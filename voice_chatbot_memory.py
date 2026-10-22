import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pyttsx3
from dotenv import load_dotenv
from google import genai
from memory_manager import MemoryManager
from voice_detector import VoiceDetector


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ Error: GEMINI_API_KEY not found in .env file.")
    exit(1)

client = genai.Client(api_key=api_key)

# Initialize TTS
engine = pyttsx3.init()
engine.setProperty("rate", 170)

# Initialize Persistent Long-Term Memory
memory = MemoryManager(gemini_client=client)
previous_interaction_id = memory.get_last_interaction_id()

print("=" * 60)
print("🤖 NOVA VOICE CHATBOT (LTM & ADAPTIVE VAD)")
print("=" * 60)

if previous_interaction_id:
    print(f"🧠 Resumed previous session ID: {previous_interaction_id}")
else:
    print("🧠 Starting fresh session.")

# Initialize Voice Detector with pre-roll buffer and ambient noise calibration
detector = VoiceDetector(
    sample_rate=16000,
    channels=1,
    silence_duration=1.4,
    pre_roll_duration=0.4
)
detector.calibrate_ambient_noise(duration=0.5)

BASE_INSTRUCTION = """
You are Nova, a helpful AI voice assistant.
Keep answers brief, natural, and conversational.
Recall stored user facts naturally when relevant.
"""

print("\nSpeak into your microphone. Say 'exit' to quit.\n")

try:
    while True:
        # 1. Record voice using smart VAD
        audio_path = detector.record_voice("recorded.wav", show_meter=True)
        if not audio_path:
            continue

        # 2. Upload audio
        print("\n📤 Processing speech...")
        try:
            audio_file = client.files.upload(file="recorded.wav")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            continue

        # 3. Speech → Text
        try:
            transcription = client.interactions.create(
                model="gemini-3.5-transcribe",
                input=[
                    {
                        "type": "audio",
                        "uri": audio_file.uri,
                        "mime_type": audio_file.mime_type,
                    }
                ],
            )
            user_text = transcription.output_text.strip()
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            continue

        if not user_text:
            print("⚠️ Could not recognize speech. Please try again.")
            continue

        print(f"\n👤 You: {user_text}")

        # 4. Exit
        if user_text.lower().strip() in ["exit", "quit", "bye", "goodbye"]:
            engine.say("Goodbye!")
            engine.runAndWait()
            memory.summarize_session_with_gemini()
            break

        # 5. Check direct memory commands
        memory_reply = memory.handle_memory_command(user_text)
        if memory_reply:
            print(f"[Nova]: {memory_reply}")
            engine.say(memory_reply)
            engine.runAndWait()
            memory.log_turn(user_text, memory_reply, previous_interaction_id)
            continue

        # 6. Send message to Gemini with dynamic Long-Term Memory context
        active_instruction = memory.build_system_instruction(
            BASE_INSTRUCTION,
            current_query=user_text
        )

        try:
            if previous_interaction_id is None:
                response = client.interactions.create(
                    model="gemini-3.6-flash",
                    input=user_text,
                    system_instruction=active_instruction
                )
            else:
                response = client.interactions.create(
                    model="gemini-3.6-flash",
                    input=user_text,
                    previous_interaction_id=previous_interaction_id,
                    system_instruction=active_instruction
                )
        except Exception as e:
            print(f"[Memory] Session error or expired ({e}). Starting fresh session...")
            memory.clear_session_id()
            try:
                response = client.interactions.create(
                    model="gemini-3.6-flash",
                    input=user_text,
                    system_instruction=active_instruction
                )
            except Exception as err:
                print(f"❌ Request failed: {err}")
                continue

        # 7. Save interaction ID and log to persistent history
        previous_interaction_id = response.id
        ai_text = response.output_text.strip()

        print(f"🤖 Nova: {ai_text}")

        # 8. Speak response
        engine.say(ai_text)
        engine.runAndWait()

        # Persist turn and extract memories in background
        memory.log_turn(user_text, ai_text, previous_interaction_id)

except KeyboardInterrupt:
    print("\n👋 Stopping...")
    memory.summarize_session_with_gemini()

print("🤖 Nova stopped.")