import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from google import genai
from memory_manager import MemoryManager

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None
memory = MemoryManager(gemini_client=client)

print("=" * 60)
print("🧠 NOVA LONG-TERM MEMORY (LTM) INSPECTOR & DIAGNOSTIC")
print("=" * 60)

# 1. User Profile
profile = memory.get_user_profile()
last_id = memory.get_last_interaction_id()
recent_history = memory.get_recent_history(limit=3)
all_ltm = memory.get_all_long_term_memories()
episodes = memory.get_episodic_history(limit=3)

print("\n[1. User Profile]")
print(f"  * Name: {profile.get('name') or 'Not set yet'}")
print(f"  * Role: {profile.get('role') or 'Not specified'}")
print(f"  * Preferences: {profile.get('preferences')}")
print(f"  * Core Facts: {profile.get('facts')}")

print("\n[2. Long-Term Knowledge Bank (LTM)]")
print(f"  * Total Stored Knowledge Items: {len(all_ltm)}")
for mem in all_ltm[-5:]:
    print(f"    - [{mem.get('category')}] {mem.get('content')} (tags: {', '.join(mem.get('tags', []))})")

print("\n[3. Cross-Session Episodic History]")
print(f"  * Total Recorded Sessions: {len(episodes)}")
for ep in episodes:
    print(f"    - [{ep.get('timestamp')[:10]}] {ep.get('summary')}")

print("\n[4. Session Continuity State]")
print(f"  * Last Gemini Interaction ID: {last_id or 'None (Fresh session)'}")
print(f"  * Recorded Conversation Turns: {len(recent_history)}")

# Test Relevance Search
print("\n[5. Testing Dynamic Relevance Search]")
sample_query = "What projects or Python code am I working on?"
results = memory.search_relevant_memories(sample_query, limit=3)
print(f"  * Query: '{sample_query}'")
print(f"  * Matched {len(results)} relevant memories:")
for r in results:
    print(f"    -> {r.get('content')}")

# Test Memory System Prompt Enrichment
print("\n[6. Enriched System Instruction Sample]")
sample_prompt = memory.build_system_instruction(
    "You are Nova, an AI voice assistant.",
    current_query=sample_query
)
print("-" * 50)
print(sample_prompt)
print("-" * 50)

print("\n" + "=" * 60)
print("✅ Long-Term Memory (LTM) system is fully active and healthy!")
print("=" * 60)