import os
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

while True:

    # 1. Record voice
    duration = 5
    sample_rate = 16000

    print("\n🎤 Speak now...")

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1
    )

    sd.wait()

    sf.write("recorded.wav", audio, sample_rate)

    # 2. Upload audio
    audio_file = client.files.upload(
        file="recorded.wav"
    )

    # 3. Convert speech to text
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
        print("👋 Goodbye!")
        break

    # 5. Send text to Gemini
    response = client.interactions.create(
        model="gemini-3.8-flash",
        input=user_text
    )

    print("🤖 AI:", response.output_text)