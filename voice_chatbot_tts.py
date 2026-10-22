import os
import sounddevice as sd
import soundfile as sf
import pyttsx3

from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)


# Initialize text-to-speech
engine = pyttsx3.init()


while True:

    # 1. Record voice
    duration = 10
    sample_rate = 16000

    print("\n🎤 Speak now...")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1
    )

    sd.wait()

    sf.write("recorded.wav", audio, sample_rate)

    print("✅ Recording complete!")


    # 2. Upload audio
    audio_file = client.files.upload(
        file="recorded.wav"
    )


    # 3. Speech → Text
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

    user_text = transcription.output_text

    print("📝 You:", user_text)


    # 4. Exit command
    if user_text.lower().strip() == "exit":

        engine.say("Goodbye Aman!")

        engine.runAndWait()

        break


    # 5. Send text to Gemini
    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=user_text
    )

    ai_text = response.output_text

    print("🤖 AI:", ai_text)


    # 6. Text → Speech
    engine.say(ai_text)

    engine.runAndWait()