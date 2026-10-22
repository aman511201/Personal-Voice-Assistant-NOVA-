import os
import sys
import re
import queue
import threading
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
from voice_detector import VoiceDetector
from wake_word_detector import WakeWordDetector


# ==========================================
# 1. PARSE ARGUMENTS & CONFIGURATION
# ==========================================

parser = argparse.ArgumentParser(description="Nova Streaming Voice Assistant with Threaded TTS, Wake Word & LTM")
parser.add_argument("--wake-word", type=str, default="hey_nova",
                    help="Wake word to listen for (default: 'hey_nova')")
parser.add_argument("--no-wake-word", action="store_true",
                    help="Disable wake word and use continuous voice activity detection")
parser.add_argument("--push-to-talk", action="store_true",
                    help="Use press-Enter push-to-talk mode")
parser.add_argument("--privacy-mode", action="store_true",
                    help="Incognito mode: Zero disk persistence for chat history or memories")
parser.add_argument("--encrypt", action="store_true",
                    help="Enable AES-256 transparent encryption for memory storage")
args = parser.parse_args()


# ==========================================
# 2. SETUP & SECURITY INITIALIZATION
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file.")
    sys.exit(1)

security = SecurityManager(
    enable_encryption=args.encrypt,
    privacy_mode=args.privacy_mode
)
masked_key = security.mask_api_key(api_key)

client = genai.Client(api_key=api_key)

# Connect MemoryManager with Gemini client
memory = MemoryManager(
    gemini_client=client,
    security_manager=security,
    privacy_mode=args.privacy_mode
)
previous_interaction_id = memory.get_last_interaction_id()

print("=" * 70)
print("🤖 NOVA STREAMING VOICE ASSISTANT (THREADED TTS, WAKE WORD & LTM)")
print("=" * 70)
print(f"🔒 Security: API Key {masked_key} | Encryption: {'ON' if args.encrypt else 'OFF'} | Privacy: {'ACTIVE' if args.privacy_mode else 'OFF'}")

profile = memory.get_user_profile()
if profile.get("name"):
    print(f"👋 Recognized User: {profile['name']}")
if previous_interaction_id:
    print(f"🧠 Resumed previous session context ({previous_interaction_id})")
else:
    print("🧠 Starting fresh session.")

# Initialize Wake Word Detector
use_wake_word = not args.no_wake_word and not args.push_to_talk
wake_detector = None

if use_wake_word:
    wake_words = [args.wake_word.lower().strip()]
    if "nova" in wake_words[0]:
        wake_words = ["nova", "hey_jarvis"]
    wake_detector = WakeWordDetector(
        wake_words=wake_words,
        threshold=0.50,
        sample_rate=16000,
        play_chime=True
    )
    print(f"👂 Local Wake Word Engine: Listening for {wake_words} (Zero idle audio uploaded)")
else:
    mode_str = "Push-to-Talk" if args.push_to_talk else "Continuous VAD"
    print(f"🎙️ Voice Mode: {mode_str}")

# Initialize Advanced Voice Activity Detector (VAD) with ultra-fast turnaround
detector = VoiceDetector(
    sample_rate=16000,
    channels=1,
    chunk_duration=0.03,        # 30ms chunks
    pre_roll_duration=0.4,       # 400ms pre-speech buffer
    silence_duration=0.65,       # 650ms trailing silence (dynamic down to 500ms)
    min_speech_duration=0.25     # Captures quick conversational answers
)

# Calibrate microphone to room ambient noise
detector.calibrate_ambient_noise(duration=0.5)

BASE_SYSTEM_INSTRUCTION = """
You are Nova, an ultra-fast, intelligent, friendly, and privacy-conscious AI voice assistant.

Rules:
- Keep answers concise, punchy, and conversational (1-2 sentences unless details are requested).
- Use simple, easy-to-understand language.
- Avoid unnecessary markdown, asterisks, or complicated formatting.
- Address the user naturally if their name is known.
- Integrate remembered facts seamlessly without reciting bulleted lists.
- Your responses will be spoken aloud.
"""


# ==========================================
# 3. ASYNCHRONOUS TEXT-TO-SPEECH (TTS)
# ==========================================

engine = pyttsx3.init()
voices = engine.getProperty("voices")
if len(voices) > 1:
    engine.setProperty("voice", voices[1].id)
engine.setProperty("rate", 185)

tts_queue = queue.Queue()


def speech_worker():
    """Worker thread that speaks text queued asynchronously."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        print(f"\n🔊 Nova: {text}")
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()


speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


# ==========================================
# 4. SPEECH TO TEXT
# ==========================================

def speech_to_text(filename):
    """Transcribes audio using Gemini Transcribe and securely wipes temp file."""
    print("\n📝 Converting speech to text...")
    try:
        audio_file = client.files.upload(file=filename)
        interaction = client.interactions.create(
            model="gemini-3.5-transcribe",
            input=[
                {
                    "type": "audio",
                    "uri": audio_file.uri,
                    "mime_type": audio_file.mime_type,
                }
            ],
        )
        return interaction.output_text.strip()
    except Exception as e:
        print(f"❌ STT Error: {e}")
        return ""
    finally:
        # Privacy: Secure shred of audio recording
        security.secure_delete_file(filename)


# ==========================================
# 5. RESPONSE GENERATION WITH LTM & SECURITY
# ==========================================

def generate_response(user_text):
    global previous_interaction_id

    # 1. Direct memory command check
    memory_reply = memory.handle_memory_command(user_text)
    if memory_reply:
        print(f"\n[Nova]: {memory_reply}")
        tts_queue.put(memory_reply)
        memory.log_turn(user_text, memory_reply, previous_interaction_id)
        return

    # 2. Enrich system instruction with contextually relevant LTM
    active_instruction = memory.build_system_instruction(
        BASE_SYSTEM_INSTRUCTION,
        current_query=user_text
    )

    # 3. Wrap with defensive delimiters and PII redaction
    guarded_instruction, guarded_input = security.wrap_safe_prompt(
        active_instruction,
        user_text
    )

    print("\n[Nova]: ", end="", flush=True)

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
        print(f"\n[Memory] Session error ({e}). Recovering fresh session...")
        memory.clear_session_id()
        try:
            response = client.interactions.create(
                model="gemini-3.8-flash",
                input=guarded_input,
                system_instruction=guarded_instruction
            )
        except Exception as err:
            print(f"\n[Error] Gemini request failed: {err}")
            tts_queue.put("Sorry, I encountered an issue processing your request.")
            return

    previous_interaction_id = response.id
    answer = response.output_text.strip()

    print(answer)
    tts_queue.put(answer)

    # 4. Log turn: updates chat history and triggers background memory consolidation
    memory.log_turn(user_text, answer, previous_interaction_id)


# ==========================================
# 6. MAIN LOOP
# ==========================================

print("\nSay 'exit' or press Ctrl+C anytime to stop.\n")

try:
    while True:
        # Step 1: Wait for local wake word (if enabled)
        if use_wake_word and wake_detector:
            wake_result = wake_detector.wait_for_wake_word()
            if not wake_result:
                continue

        # Step 2: Record voice using advanced VAD or Push-to-Talk
        if args.push_to_talk:
            audio_file = detector.record_push_to_talk("voice_input.wav")
        else:
            audio_file = detector.record_voice("voice_input.wav", show_meter=True)

        if not audio_file or not os.path.exists(audio_file):
            continue

        # Step 3: Transcribe & Shred
        user_text = speech_to_text(audio_file)

        if not user_text:
            print("⚠️ Could not understand audio.")
            continue

        print(f"\n👤 You: {user_text}")

        # Handle Standalone Wake Word or strip leading "Hey Nova" prefix
        clean_lower = user_text.lower().strip()
        if clean_lower in ["hey nova", "nova", "hi nova", "hello nova", "hey nova!"]:
            ack = "I'm listening! What can I help you with?"
            print(f"\n[Nova]: {ack}")
            tts_queue.put(ack)
            continue

        user_text = re.sub(r"^(?:hey\s+nova|nova|hello\s+nova)[,\s]+", "", user_text, flags=re.IGNORECASE).strip()
        if not user_text:
            continue

        # Step 4: Exit command
        if user_text.lower().strip() in ["exit", "quit", "goodbye", "stop", "bye"]:
            print("\n👋 Goodbye!")
            tts_queue.put("Goodbye! Have a great day.")
            tts_queue.join()
            memory.summarize_session_with_gemini()
            break

        # Step 5: Generate response & queue audio
        generate_response(user_text)

except KeyboardInterrupt:
    print("\n\n👋 Shutting down Nova...")
    tts_queue.join()
    memory.summarize_session_with_gemini()

print("\n🤖 Nova stopped.")
