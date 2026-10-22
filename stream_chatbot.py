import os
import re
import pyttsx3
import threading
import queue

from dotenv import load_dotenv
from google import genai


# ==========================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ GEMINI_API_KEY not found in .env file.")
    exit()


# ==========================================
# 2. CREATE GEMINI CLIENT
# ==========================================

client = genai.Client(api_key=api_key)


# ==========================================
# 3. NOVA PERSONALITY
# ==========================================

SYSTEM_INSTRUCTION = """
You are Nova, a friendly AI voice assistant.

Rules:
- Keep answers short and conversational.
- Use simple language.
- Avoid unnecessary markdown.
- Explain technical questions step by step.
- Your responses will be spoken aloud.
- Do not give unnecessarily long answers.
- Be friendly and natural.
"""


# ==========================================
# 4. SETUP TEXT-TO-SPEECH
# ==========================================

engine = pyttsx3.init()

voices = engine.getProperty("voices")

# Use Microsoft Zira
if len(voices) > 1:
    engine.setProperty("voice", voices[1].id)

engine.setProperty("rate", 170)


# ==========================================
# 5. CREATE TTS QUEUE
# ==========================================

tts_queue = queue.Queue()


# ==========================================
# 6. TTS WORKER
# ==========================================

def speech_worker():

    while True:

        text = tts_queue.get()

        # Stop worker
        if text is None:
            break

        print("\n🔊 Nova is speaking...")

        engine.say(text)
        engine.runAndWait()

        tts_queue.task_done()


# ==========================================
# 7. START TTS THREAD
# ==========================================

speech_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

speech_thread.start()


# ==========================================
# 8. START NOVA
# ==========================================

print("🤖 Nova is ready!")
print("Type 'exit' to stop.\n")


# ==========================================
# 9. MAIN CHAT LOOP
# ==========================================

while True:

    # Get user input
    user_text = input("You: ")

    # Exit command
    if user_text.lower().strip() == "exit":

        print("👋 Goodbye!")

        # Stop TTS worker
        tts_queue.put(None)

        break


    # Prevent empty input
    if not user_text.strip():

        print("⚠️ Please enter something.\n")

        continue


    print("\nNova: ", end="", flush=True)


    # ======================================
    # 10. STREAM GEMINI RESPONSE
    # ======================================

    sentence_buffer = ""


    try:

        for chunk in client.models.generate_content_stream(

            model="gemini-3.6-flash",

            contents=user_text,

            config={
                "system_instruction": SYSTEM_INSTRUCTION
            }

        ):

            if chunk.text:

                text = chunk.text

                # Print streamed text
                print(text, end="", flush=True)

                # Add text to speech buffer
                sentence_buffer += text


                # ==================================
                # 11. DETECT COMPLETE SENTENCE
                # ==================================

                if re.search(r"[.!?]\s*$", sentence_buffer):

                    sentence = sentence_buffer.strip()

                    # Put sentence into TTS queue
                    tts_queue.put(sentence)

                    # Clear buffer
                    sentence_buffer = ""


        # ==================================
        # 12. SPEAK REMAINING TEXT
        # ==================================

        if sentence_buffer.strip():

            tts_queue.put(sentence_buffer.strip())


        print("\n")


    except Exception as e:

        print("\n❌ Error:", e)

        print()