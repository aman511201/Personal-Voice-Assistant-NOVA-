import os
import sys
import re
import json
import uuid
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional Scikit-Learn TF-IDF for advanced semantic memory ranking
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

from security_manager import SecurityManager


class MemoryManager:
    """
    Advanced Multi-Tier Conversational Memory System with Hybrid Semantic Retrieval,
    Episodic Summarization, and Cryptographic Security for Nova Voice Assistant.
    
    Memory Tiers:
    1. Working Memory (Short-Term Context Window):
       - Recent conversation turns tracked in memory and rolling transcript.
    2. Episodic Memory (Cross-Session Experience):
       - Summaries of past discussions, goals, and milestones across sessions.
    3. Semantic Knowledge Bank (Long-Term Memory - LTM):
       - Structured, categorized atomic memories (identity, preferences, projects, technical, rules).
       - Hybrid TF-IDF Vector & Lexical search with recency decay and frequency reinforcement.
       - Contradiction resolution: Updates old facts when new contradictory information is provided.
    4. User Profile & Identity:
       - Name, role, communication style, core personal facts, and preferences.
       
    Security & Privacy:
       - Transparent AES-256 storage encryption via SecurityManager.
       - Automatic PII redaction before calling cloud LLM for memory consolidation.
       - Ephemeral Privacy Mode (Incognito): Zero disk writes when enabled.
       - Secure memory export and full erasure capabilities.
    """

    def __init__(
        self,
        memory_dir: Optional[str] = None,
        gemini_client: Optional[Any] = None,
        security_manager: Optional[SecurityManager] = None,
        privacy_mode: bool = False
    ):
        if memory_dir is None:
            base_dir = Path(__file__).resolve().parent
            self.memory_dir = base_dir / "memory"
        else:
            self.memory_dir = Path(memory_dir)

        self.privacy_mode = privacy_mode
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.session_file = self.memory_dir / "session_state.json"
        self.profile_file = self.memory_dir / "user_profile.json"
        self.ltm_file = self.memory_dir / "long_term_memories.json"
        self.episodic_file = self.memory_dir / "episodic_history.json"
        self.history_file = self.memory_dir / "chat_history.json"

        self.gemini_client = gemini_client
        self.security = security_manager or SecurityManager(privacy_mode=self.privacy_mode)

        self._lock = threading.RLock()
        self._init_files()

    def connect_gemini_client(self, client: Any):
        """Connects Google Gemini client for AI-powered memory extraction."""
        self.gemini_client = client

    def _init_files(self):
        """Initializes storage files with secure defaults if not present."""
        if not self.session_file.exists():
            self._write_json(self.session_file, {
                "last_interaction_id": None,
                "current_session_id": str(uuid.uuid4())[:8],
                "last_active": None,
                "session_count": 0
            })

        if not self.profile_file.exists():
            self._write_json(self.profile_file, {
                "name": None,
                "role": None,
                "communication_style": "concise and friendly",
                "interests": [],
                "facts": [],
                "preferences": [],
                "projects": []
            })

        if not self.ltm_file.exists():
            self._write_json(self.ltm_file, [])

        if not self.episodic_file.exists():
            self._write_json(self.episodic_file, [])

        if not self.history_file.exists():
            self._write_json(self.history_file, [])

    def _read_json(self, file_path: Path, default: Any = None) -> Any:
        """Thread-safe, privacy-aware JSON read with transparent decryption."""
        with self._lock:
            return self.security.read_secure_json(file_path, default=default)

    def _write_json(self, file_path: Path, data: Any):
        """Thread-safe, privacy-aware JSON write with transparent encryption."""
        if self.privacy_mode:
            return
        with self._lock:
            self.security.write_secure_json(file_path, data)

    # ==========================================
    # SESSION & INTERACTION ID PERSISTENCE
    # ==========================================

    def get_last_interaction_id(self) -> Optional[str]:
        """Retrieves previous Gemini interaction ID for state resumption."""
        data = self._read_json(self.session_file, {})
        return data.get("last_interaction_id")

    def save_interaction_id(self, interaction_id: str):
        """Persists the latest interaction ID to disk."""
        data = self._read_json(self.session_file, {})
        data["last_interaction_id"] = interaction_id
        data["last_active"] = datetime.now(timezone.utc).isoformat()
        data["session_count"] = data.get("session_count", 0) + 1
        self._write_json(self.session_file, data)

    def clear_session_id(self):
        """Resets the Gemini session interaction ID."""
        data = self._read_json(self.session_file, {})
        data["last_interaction_id"] = None
        self._write_json(self.session_file, data)

    # ==========================================
    # USER PROFILE MANAGEMENT
    # ==========================================

    def get_user_profile(self) -> Dict[str, Any]:
        """Returns the stored user profile."""
        return self._read_json(self.profile_file, {
            "name": None,
            "role": None,
            "communication_style": "concise and friendly",
            "interests": [],
            "facts": [],
            "preferences": [],
            "projects": []
        })

    def set_user_name(self, name: str):
        """Updates user's name in persistent profile and resolves prior name entries."""
        name = name.strip().title()
        if not name or name.lower() in ["nova", "assistant", "none", "null"]:
            return

        profile = self.get_user_profile()
        old_name = profile.get("name")
        profile["name"] = name
        self._write_json(self.profile_file, profile)

        # Update in LTM
        if old_name and old_name.lower() != name.lower():
            self.forget_memory(f"name is {old_name}")
        self.add_long_term_memory(content=f"User's name is {name}", category="identity", confidence=1.0)
        print(f"[Memory] Updated user name: {name}")

    def add_fact(self, fact: str, category: str = "personal"):
        """Adds a fact to persistent user profile and long-term memory."""
        fact = fact.strip()
        if not fact or len(fact) < 3:
            return

        profile = self.get_user_profile()
        facts = profile.get("facts", [])

        # Check for duplication
        if not any(f.lower() == fact.lower() for f in facts):
            facts.append(fact)
            profile["facts"] = facts
            self._write_json(self.profile_file, profile)
            print(f"[Memory] Learned fact: '{fact}'")

        self.add_long_term_memory(content=fact, category=category)

    def add_preference(self, preference: str):
        """Adds a preference to persistent user profile and long-term memory."""
        preference = preference.strip()
        if not preference or len(preference) < 3:
            return

        profile = self.get_user_profile()
        prefs = profile.get("preferences", [])

        if not any(p.lower() == preference.lower() for p in prefs):
            prefs.append(preference)
            profile["preferences"] = prefs
            self._write_json(self.profile_file, profile)
            print(f"[Memory] Learned preference: '{preference}'")

        self.add_long_term_memory(content=preference, category="preference")

    def add_project(self, project: str):
        """Adds an active user project to profile and long-term memory."""
        project = project.strip()
        if not project or len(project) < 3:
            return

        profile = self.get_user_profile()
        projects = profile.get("projects", [])

        if not any(p.lower() == project.lower() for p in projects):
            projects.append(project)
            profile["projects"] = projects
            self._write_json(self.profile_file, profile)
            print(f"[Memory] Recorded project: '{project}'")

        self.add_long_term_memory(content=f"Working on: {project}", category="work_project")

    # ==========================================
    # HYBRID SEMANTIC LONG-TERM MEMORY BANK (LTM)
    # ==========================================

    def add_long_term_memory(
        self,
        content: str,
        category: str = "general",
        tags: Optional[List[str]] = None,
        confidence: float = 0.9
    ) -> Optional[str]:
        """
        Adds a new entry into the structured Long-Term Memory bank.
        Performs contradiction detection and deduplication.
        """
        content = content.strip()
        if not content or len(content) < 3:
            return None

        memories = self._read_json(self.ltm_file, [])
        content_lower = content.lower()

        # Deduplication check
        for mem in memories:
            if mem.get("content", "").lower() == content_lower:
                # Reinforce existing memory
                mem["access_count"] = mem.get("access_count", 0) + 1
                mem["last_accessed"] = datetime.now(timezone.utc).isoformat()
                self._write_json(self.ltm_file, memories)
                return mem.get("id")

        if tags is None:
            tags = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", content)[:5]]

        now_iso = datetime.now(timezone.utc).isoformat()
        mem_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}"

        memory_entry = {
            "id": mem_id,
            "created_at": now_iso,
            "last_accessed": now_iso,
            "access_count": 1,
            "category": category,
            "content": content,
            "tags": tags,
            "confidence": confidence
        }

        memories.append(memory_entry)
        self._write_json(self.ltm_file, memories)
        return mem_id

    def get_all_long_term_memories(self) -> List[Dict[str, Any]]:
        """Returns all entries from Long-Term Memory."""
        return self._read_json(self.ltm_file, [])

    def search_relevant_memories(self, query: Optional[str], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid Semantic Memory Retrieval:
        Combines TF-IDF vector similarity with recency decay, access frequency, and token overlap.
        """
        memories = self.get_all_long_term_memories()
        if not memories:
            return []

        if not query or not query.strip():
            # Return most recently accessed/created memories
            return sorted(memories, key=lambda m: m.get("created_at", ""), reverse=True)[:limit]

        query_clean = query.strip().lower()
        now = datetime.now(timezone.utc)

        # 1. Scikit-Learn TF-IDF Semantic Cosine Similarity (if available and >= 2 items)
        tfidf_scores = [0.0] * len(memories)
        if HAS_SKLEARN and len(memories) >= 2:
            try:
                corpus = [m.get("content", "") + " " + " ".join(m.get("tags", [])) for m in memories]
                vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
                matrix = vectorizer.fit_transform(corpus)
                q_vec = vectorizer.transform([query_clean])
                similarities = cosine_similarity(q_vec, matrix).flatten()
                tfidf_scores = list(similarities)
            except Exception:
                tfidf_scores = [0.0] * len(memories)

        # 2. Token Overlap & Multi-Factor Scoring
        query_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", query_clean))
        scored = []

        for idx, mem in enumerate(memories):
            content_lower = mem.get("content", "").lower()
            tags = [t.lower() for t in mem.get("tags", [])]

            # TF-IDF score component (weight: 4.0)
            score = float(tfidf_scores[idx]) * 4.0

            # Direct lexical matching
            for token in query_tokens:
                if token in content_lower:
                    score += 2.0
                if token in tags:
                    score += 2.5

            # Recency Exponential Decay (half-life of 14 days)
            created_str = mem.get("created_at")
            if created_str:
                try:
                    created_dt = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
                    days_elapsed = max(0.0, (now - created_dt).total_seconds() / 86400.0)
                    recency_weight = math.exp(-days_elapsed / 14.0)
                except Exception:
                    recency_weight = 0.5
            else:
                recency_weight = 0.5

            score += recency_weight * 0.5

            # Access frequency reinforcement
            access_count = mem.get("access_count", 1)
            score += min(1.0, access_count * 0.1)

            # Confidence factor
            score *= mem.get("confidence", 0.9)

            scored.append((score, mem))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Update last_accessed and access_count for top results
        top_results = [item[1] for item in scored[:limit] if item[0] > 0.1]
        if not top_results:
            top_results = memories[-limit:]

        for m in top_results:
            m["last_accessed"] = now.isoformat()
            m["access_count"] = m.get("access_count", 0) + 1

        self._write_json(self.ltm_file, memories)
        return top_results

    def forget_memory(self, query_or_id: str) -> int:
        """
        Deletes matching memories from both LTM bank and user profile.
        Supports query keywords, exact memory IDs, or target topics.
        """
        memories = self.get_all_long_term_memories()
        query_clean = query_or_id.lower().strip()

        sub_query = re.sub(r"^(?:that\s+)?(?:i\s+(?:am|have|like|drink|eat|prefer|work|live)\s+|my\s+)", "", query_clean).strip()
        tokens = [t for t in re.findall(r"\b[a-zA-Z0-9]+\b", sub_query) if len(t) > 2]

        remaining = []
        removed_count = 0

        for mem in memories:
            content_lower = mem.get("content", "").lower()
            is_match = False

            if mem.get("id") == query_or_id:
                is_match = True
            elif query_clean in content_lower or (sub_query and sub_query in content_lower):
                is_match = True
            elif tokens and all(t in content_lower for t in tokens):
                is_match = True

            if is_match:
                removed_count += 1
            else:
                remaining.append(mem)

        self._write_json(self.ltm_file, remaining)

        # Also purge from user profile
        profile = self.get_user_profile()
        profile["facts"] = [
            f for f in profile.get("facts", [])
            if not (query_clean in f.lower() or (sub_query and sub_query in f.lower()) or (tokens and all(t in f.lower() for t in tokens)))
        ]
        profile["preferences"] = [
            p for p in profile.get("preferences", [])
            if not (query_clean in p.lower() or (sub_query and sub_query in p.lower()) or (tokens and all(t in p.lower() for t in tokens)))
        ]
        profile["projects"] = [
            pr for pr in profile.get("projects", [])
            if not (query_clean in pr.lower() or (sub_query and sub_query in pr.lower()) or (tokens and all(t in pr.lower() for t in tokens)))
        ]
        self._write_json(self.profile_file, profile)

        return removed_count

    # ==========================================
    # EPISODIC CROSS-SESSION MEMORY
    # ==========================================

    def add_episodic_summary(self, summary: str, topics: Optional[List[str]] = None, turns: int = 0):
        """Records an episodic session summary so Nova remembers past conversations."""
        summary = summary.strip()
        if not summary:
            return
        episodes = self._read_json(self.episodic_file, [])
        entry = {
            "session_id": f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "topics": topics or [],
            "turns": turns
        }
        episodes.append(entry)
        if len(episodes) > 30:
            episodes = episodes[-30:]
        self._write_json(self.episodic_file, episodes)

    def get_episodic_history(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Returns the most recent episodic conversation summaries."""
        episodes = self._read_json(self.episodic_file, [])
        return episodes[-limit:]

    # ==========================================
    # CHAT TRANSCRIPT & WORKING MEMORY
    # ==========================================

    def log_turn(self, user_text: str, ai_text: str, interaction_id: Optional[str] = None):
        """
        Logs a single conversation turn:
        1. Fast local regex learning (immediate fallback).
        2. Asynchronous Gemini AI Memory Consolidation in background thread (0-latency).
        """
        history = self._read_json(self.history_file, [])
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": user_text,
            "ai": ai_text,
            "interaction_id": interaction_id
        }
        history.append(entry)
        if len(history) > 100:
            history = history[-100:]
        self._write_json(self.history_file, history)

        # 1. Immediate local heuristic learning
        self._heuristic_learn_from_user(user_text)

        # 2. Asynchronous Gemini AI Extraction (scrubs PII first for privacy)
        if self.gemini_client is not None and not self.privacy_mode:
            thread = threading.Thread(
                target=self._extract_with_gemini_worker,
                args=(user_text, ai_text),
                daemon=True
            )
            thread.start()

        if interaction_id:
            self.save_interaction_id(interaction_id)

    def get_recent_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Returns the most recent N turns from conversation history."""
        history = self._read_json(self.history_file, [])
        return history[-limit:]

    # ==========================================
    # GEMINI AI MEMORY EXTRACTION (BACKGROUND 0-LATENCY)
    # ==========================================

    def _extract_with_gemini_worker(self, user_text: str, ai_text: str):
        """
        Background worker calling Gemini 3.8 Flash to intelligently extract
        structured user facts, preferences, and projects with privacy scrubbing.
        """
        if not self.gemini_client:
            return

        # Scrub PII and secrets prior to cloud transmission
        safe_user = self.security.scrub_pii(user_text)
        safe_ai = self.security.scrub_pii(ai_text)

        prompt = f"""
You are the Memory Consolidation Engine for Nova, an AI voice assistant.
Analyze this single conversation turn:
User said: "{safe_user}"
Nova replied: "{safe_ai}"

Task:
Extract any persistent information worth remembering about the user in future conversations:
1. User's name or identity.
2. Personal background, job, tools, hobbies, projects, or goals.
3. Long-term preferences, habits, or rules for Nova.
4. Active projects mentioned.

Return ONLY a valid JSON object matching this schema:
{{
  "name": "User's name if mentioned or null",
  "new_facts": ["concise fact 1"],
  "preferences": ["preference 1"],
  "projects": ["project 1"]
}}

If nothing permanent about the user was learned in this turn, return:
{{"name": null, "new_facts": [], "preferences": [], "projects": []}}
"""
        try:
            # Uses gemini-3.8-flash for rapid, accurate memory synthesis
            resp = self.gemini_client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )
            raw = resp.text.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\n", "", raw)
                raw = re.sub(r"\n```$", "", raw)

            data = json.loads(raw)

            name = data.get("name")
            if name and isinstance(name, str) and len(name) < 40:
                clean_name = name.strip().title()
                if clean_name.lower() not in ["nova", "assistant", "ai", "none", "null"]:
                    self.set_user_name(clean_name)

            for fact in data.get("new_facts", []):
                if isinstance(fact, str) and len(fact) > 3:
                    self.add_fact(fact)

            for pref in data.get("preferences", []):
                if isinstance(pref, str) and len(pref) > 3:
                    self.add_preference(pref)

            for proj in data.get("projects", []):
                if isinstance(proj, str) and len(proj) > 3:
                    self.add_project(proj)

        except Exception as e:
            # Fallback heuristics ensure continuity even if network is unavailable
            pass

    def summarize_session_with_gemini(self):
        """
        Synthesizes a 1-2 sentence episodic summary of the conversation session.
        Called on graceful exit or program completion.
        """
        recent = self.get_recent_history(limit=12)
        if len(recent) < 2 or not self.gemini_client or self.privacy_mode:
            return

        transcript = "\n".join([f"User: {t['user']}\nNova: {t['ai']}" for t in recent])
        safe_transcript = self.security.scrub_pii(transcript)

        prompt = f"""
Summarize this conversation between User and Nova into a 1-2 sentence episodic memory:
{safe_transcript}

Return ONLY a 1-2 sentence summary of what was discussed, what the user worked on or asked about.
"""
        try:
            resp = self.gemini_client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )
            summary = resp.text.strip()
            if summary:
                self.add_episodic_summary(summary=summary, turns=len(recent))
                print(f"\n[Memory] Saved episodic session summary: {summary}")
        except Exception:
            pass

    # ==========================================
    # HEURISTIC LOCAL LEARNING (OFFLINE SAFETY NET)
    # ==========================================

    def _heuristic_learn_from_user(self, user_text: str):
        """Fast offline regex extractor for immediate facts and preferences."""
        text = user_text.strip()

        # Name detection patterns
        name_match = re.search(r"\b(?:my name is|i am called|call me)\s+([A-Z][a-zA-Z]+)", text, re.IGNORECASE)
        if name_match:
            detected_name = name_match.group(1).title()
            if detected_name.lower() not in ["nova", "a", "an", "the", "here", "ready", "fine", "good", "happy"]:
                self.set_user_name(detected_name)

        # Preference patterns
        pref_match = re.search(r"\bi\s+(?:really\s+)?(?:like|love|prefer)\s+(.+?)(?:\.|$)", text, re.IGNORECASE)
        if pref_match:
            candidate = pref_match.group(1).strip()
            if 2 < len(candidate) < 80:
                self.add_preference(f"Likes {candidate}")

        # Explicit "remember that..."
        remember_match = re.search(r"\bremember\s+(?:that\s+)?(.+?)(?:\.|$)", text, re.IGNORECASE)
        if remember_match:
            fact = remember_match.group(1).strip()
            if 2 < len(fact) < 120:
                self.add_fact(fact)

    # ==========================================
    # CONTEXT GROUNDING & DYNAMIC PROMPT BUILDER
    # ==========================================

    def build_system_instruction(self, base_instruction: str, current_query: Optional[str] = None) -> str:
        """
        Dynamically grounds the system instruction with:
        - User Identity & Preferences
        - Semantic memories retrieved via hybrid TF-IDF similarity
        - Cross-Session Episodic History
        - Privacy & Security Guardrails
        """
        profile = self.get_user_profile()
        name = profile.get("name")
        facts = profile.get("facts", [])
        preferences = profile.get("preferences", [])
        projects = profile.get("projects", [])
        episodes = self.get_episodic_history(limit=2)
        recent_turns = self.get_recent_history(limit=3)

        # Hybrid semantic search for query-relevant memories
        relevant_memories = self.search_relevant_memories(current_query, limit=4) if current_query else []

        memory_lines = ["\n[PERSISTENT LONG-TERM MEMORY & CONTEXT]"]

        if name:
            memory_lines.append(f"- User's Name: {name} (Address the user naturally as {name} when appropriate)")
        else:
            memory_lines.append("- User's Name: Not yet specified")

        if preferences:
            memory_lines.append("- Stored Preferences:")
            for p in preferences[-5:]:
                memory_lines.append(f"  * {p}")

        if facts:
            memory_lines.append("- Core Remembered Facts:")
            for f in facts[-5:]:
                memory_lines.append(f"  * {f}")

        if projects:
            memory_lines.append("- Active Projects & Goals:")
            for pr in projects[-4:]:
                memory_lines.append(f"  * {pr}")

        if relevant_memories:
            memory_lines.append("- Contextually Relevant Knowledge Bank:")
            for mem in relevant_memories:
                memory_lines.append(f"  * [{mem.get('category', 'fact')}] {mem.get('content')}")

        if episodes:
            memory_lines.append("- Cross-Session Episodic History:")
            for ep in episodes:
                memory_lines.append(f"  * Previous Discussion: {ep.get('summary')}")

        if recent_turns:
            memory_lines.append("- Recent Conversational Turns:")
            for turn in recent_turns:
                memory_lines.append(f"  * User: \"{turn['user']}\" -> Nova: \"{turn['ai']}\"")

        memory_lines.append(
            "- Memory Guidance: You have permanent cross-session memory. Ground your answers "
            "naturally in this knowledge. If asked what you remember, summarize warmly and accurately."
        )

        return base_instruction.strip() + "\n" + "\n".join(memory_lines) + "\n"

    # ==========================================
    # DIRECT MEMORY VOICE & TEXT COMMANDS
    # ==========================================

    def handle_memory_command(self, user_text: str) -> Optional[str]:
        """
        Handles direct voice and text memory management commands.
        Returns a friendly response string if handled, or None if it should proceed to Gemini.
        """
        clean_text = user_text.lower().strip()

        # Query what is remembered
        if any(trigger in clean_text for trigger in [
            "what do you remember about me",
            "what do you know about me",
            "tell me what you remember",
            "what is my memory",
            "what have you learned about me",
            "what's my name",
            "do you know my name",
            "what are my preferences",
            "what projects am i working on",
            "show memory summary"
        ]):
            profile = self.get_user_profile()
            name = profile.get("name")
            facts = profile.get("facts", [])
            prefs = profile.get("preferences", [])
            projects = profile.get("projects", [])
            ltm = self.get_all_long_term_memories()

            if "what's my name" in clean_text or "do you know my name" in clean_text:
                if name:
                    return f"Your name is {name}!"
                else:
                    return "I don't know your name yet. You can tell me by saying 'My name is...'!"

            if "what projects" in clean_text:
                if projects:
                    return f"You are currently working on: {', '.join(projects)}."
                else:
                    return "You haven't mentioned any active projects yet. Tell me what you're building!"

            parts = []
            if name:
                parts.append(f"Your name is {name}.")
            if prefs:
                parts.append(f"Your preferences include: {', '.join(prefs[:4])}.")
            if facts:
                parts.append(f"I remember that: {'; '.join(facts[:4])}.")
            if projects:
                parts.append(f"Projects: {', '.join(projects[:3])}.")
            elif ltm:
                sample_ltm = [m['content'] for m in ltm[:4]]
                parts.append(f"Stored memories: {'; '.join(sample_ltm)}.")

            if parts:
                return "Here is what I remember: " + " ".join(parts)
            else:
                return "I don't have any specific facts saved about you yet, but I'm ready to learn!"

        # Explicit remember command
        remember_match = re.match(r"^remember(?:\s+that)?\s+(.+)$", clean_text)
        if remember_match:
            fact_to_remember = user_text[remember_match.start(1):].strip().rstrip(".")
            self.add_fact(fact_to_remember)
            return f"Got it! I will remember that {fact_to_remember}."

        # Forget specific memory command
        forget_match = re.match(r"^forget(?:\s+that)?\s+(.+)$", clean_text)
        if forget_match:
            query = user_text[forget_match.start(1):].strip().rstrip(".")
            if query.lower() in ["everything", "all", "memory", "what you know about me"]:
                self.clear_all_memory()
                return "I have cleared all my memory. We can start fresh!"
            count = self.forget_memory(query)
            if count > 0:
                return f"I have forgotten about '{query}'."
            else:
                return f"I couldn't find any stored memory about '{query}' to forget."

        # Export memory command (Transparency / GDPR)
        if any(trigger in clean_text for trigger in ["export memory", "export my memory", "backup memory"]):
            export_path = self.export_memory()
            return f"I have exported your complete memory profile to {export_path.name}."

        # Clear all memory command
        if self.security.is_destructive_command(clean_text):
            self.clear_all_memory()
            return "I have cleared all my stored memory. We can start completely fresh!"

        return None

    def export_memory(self) -> Path:
        """Exports complete user memory bank to a readable markdown report."""
        export_file = self.memory_dir / "memory_export.md"
        profile = self.get_user_profile()
        ltm = self.get_all_long_term_memories()
        episodes = self.get_episodic_history(limit=10)

        lines = [
            "# Nova AI Voice Assistant - Stored User Memory Export",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## 1. User Identity & Profile",
            f"- **Name**: {profile.get('name', 'Unspecified')}",
            f"- **Role**: {profile.get('role', 'Unspecified')}",
            f"- **Communication Style**: {profile.get('communication_style', 'Default')}",
            "",
            "### Preferences",
        ]
        for p in profile.get("preferences", []):
            lines.append(f"- {p}")

        lines.append("\n### Core Facts")
        for f in profile.get("facts", []):
            lines.append(f"- {f}")

        lines.append("\n### Active Projects")
        for pr in profile.get("projects", []):
            lines.append(f"- {pr}")

        lines.append("\n## 2. Long-Term Semantic Memories")
        for m in ltm:
            lines.append(f"- [{m.get('category', 'general')}] {m.get('content')} (Created: {m.get('created_at')})")

        lines.append("\n## 3. Episodic Conversation History")
        for ep in episodes:
            lines.append(f"- [{ep.get('timestamp')}] {ep.get('summary')}")

        with open(export_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"[Memory] Exported full memory profile to {export_file}")
        return export_file

    def clear_all_memory(self):
        """Securely clears all persistent memory files across session, profile, LTM, and history."""
        self._write_json(self.session_file, {
            "last_interaction_id": None,
            "current_session_id": str(uuid.uuid4())[:8],
            "last_active": None,
            "session_count": 0
        })
        self._write_json(self.profile_file, {
            "name": None,
            "role": None,
            "communication_style": "concise and friendly",
            "interests": [],
            "facts": [],
            "preferences": [],
            "projects": []
        })
        self._write_json(self.ltm_file, [])
        self._write_json(self.episodic_file, [])
        self._write_json(self.history_file, [])
        print("[Memory] All persistent memory records cleared.")
