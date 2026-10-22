import os
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

# Record voice
duration = 5
sample_rate = 16000

print("🎤 Speak now...")

audio = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1
)

sd.wait()

sf.write("recorded.wav", audio, sample_rate)

print("✅ Recording complete!")

# Upload audio to Gemini
audio_file = client.files.upload(
    file="recorded.wav"
)

# Transcribe
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

print("\n📝 You said:")
print(interaction.output_text)