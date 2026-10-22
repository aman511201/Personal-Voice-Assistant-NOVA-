import sounddevice as sd
import soundfile as sf

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

print("✅ Recording saved as recorded.wav")