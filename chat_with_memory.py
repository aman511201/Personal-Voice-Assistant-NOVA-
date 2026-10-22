import os
import sys
import argparse
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pyttsx3
from dotenv import load_dotenv
from google import genai

from security_manager import SecurityManager
from memory_manager import MemoryManager

# Parse optional security arguments
parser = argparse.ArgumentParser(description="Nova Text Console with LTM & Security")
parser.add_argument("--privacy-mode", action="store_true",
                    help="Incognito session: No memories or chat history saved to disk")
parser.add_argument("--encrypt", action="store_true",
                    help="Enable AES-256 transparent encryption for memory storage")
parser.add_argument("--mute", action="store_true",
                    help="Disable spoken audio replies (text-only mode)")
args = parser.parse_args()

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env")
    sys.exit(1)

security = SecurityManager(
    enable_encryption=args.encrypt,
    privacy_mode=args.privacy_mode
)
masked_key = security.mask_api_key(api_key)

client = genai.Client(api_key=api_key)

# Connect MemoryManager with Gemini 3.8
memory = MemoryManager(
    gemini_client=client,
    security_manager=security,
    privacy_mode=args.privacy_mode
)
previous_interaction_id = memory.get_last_interaction_id()

# TTS setup
engine = None
if not args.mute:
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 170)
    except Exception:
        engine = None

SYSTEM_INSTRUCTION = """
You are Nova, a friendly, intelligent, and privacy-conscious AI assistant.

Rules:
- Keep your answers short and conversational.
- Use simple language.
- Address the user warmly if their name is known.
- Be friendly, helpful, and natural.
- Use your stored long-term memories when relevant.
"""

print("=" * 65)
print("🤖 NOVA AI ASSISTANT (PERSISTENT LTM CONSOLE)")
print("=" * 65)
print(f"🔒 Security: Key {masked_key} | Encryption: {'ON' if args.encrypt else 'OFF'} | Privacy Mode: {'ACTIVE' if args.privacy_mode else 'OFF'}")

if previous_interaction_id:
    print(f"[Memory] Resumed existing session: {previous_interaction_id}")
else:
    print("[Memory] Starting fresh session.")

profile = memory.get_user_profile()
if profile.get("name"):
    print(f"[Memory] Recognized User: {profile.get('name')}")

ltm_count = len(memory.get_all_long_term_memories())
print(f"[Memory] Stored Long-Term Knowledge Items: {ltm_count}")

episodes = memory.get_episodic_history(limit=1)
if episodes:
    print(f"[Memory] Last Session Context: {episodes[0].get('summary')}")

print("\nType your message below.")
print("Commands: 'exit', 'what do you remember about me', 'remember that...', 'forget that...', 'export memory', 'reset memory'.\n")

while True:
    try:
        user_text = input("You: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Goodbye!")
        memory.summarize_session_with_gemini()
        break

    if not user_text:
        continue

    if user_text.lower() in ["exit", "quit", "bye", "goodbye"]:
        print("Nova: Goodbye! Have a great day.")
        if engine:
            engine.say("Goodbye! Have a great day.")
            engine.runAndWait()
        memory.summarize_session_with_gemini()
        break

    # 1. Direct memory commands
    memory_reply = memory.handle_memory_command(user_text)
    if memory_reply:
        print(f"\nNova: {memory_reply}\n")
        if engine:
            engine.say(memory_reply)
            engine.runAndWait()
        memory.log_turn(user_text, memory_reply, previous_interaction_id)
        continue

    # 2. Enrich system instruction with contextually relevant long-term memories
    active_instruction = memory.build_system_instruction(
        SYSTEM_INSTRUCTION,
        current_query=user_text
    )

    # 3. Guard against prompt injections & redact PII
    guarded_instruction, guarded_input = security.wrap_safe_prompt(
        active_instruction,
        user_text
    )

    # 4. Request Gemini 3.8 response
    try:
        if previous_interaction_id is None:
            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=guarded_input,
                system_instruction=guarded_instruction
            )
        else:
            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=guarded_input,
                previous_interaction_id=previous_interaction_id,
                system_instruction=guarded_instruction
            )
    except Exception as e:
        print(f"[Memory] Session ended or error ({e}). Starting fresh session...")
        memory.clear_session_id()
        try:
            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=guarded_input,
                system_instruction=guarded_instruction
            )
        except Exception as err:
            print(f"[Error] Request failed: {err}\n")
            continue

    previous_interaction_id = response.id
    ai_text = response.output_text.strip()

    print(f"\nNova: {ai_text}\n")
    if engine:
        engine.say(ai_text)
        engine.runAndWait()

    # 5. Save turn to history and extract memories in background via Gemini
    memory.log_turn(user_text, ai_text, previous_interaction_id)
