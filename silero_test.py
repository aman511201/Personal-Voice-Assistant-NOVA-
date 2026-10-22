
import sounddevice as sd
import numpy as np
import time

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

print("Stay quiet for 2 seconds...")
noise_levels = []

for _ in range(20):
    audio = sd.rec(
        CHUNK_SIZE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    noise_levels.append(float(np.sqrt(np.mean(audio ** 2))))

noise_level = float(np.median(noise_levels))
threshold = max(0.005, noise_level * 3)

print(f"Background level: {noise_level:.5f}")
print(f"Speech threshold: {threshold:.5f}")
print("\nSpeak normally. Press Ctrl+C to stop.")

try:
    while True:
        audio = sd.rec(
            CHUNK_SIZE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32"
        )
        sd.wait()

        level = float(np.sqrt(np.mean(audio ** 2)))
        is_loud = level > threshold

        status = "SOUND" if is_loud else "QUIET"
        print(f"\rLevel: {level:.5f} | {status}   ", end="")

except KeyboardInterrupt:
    print("\nTest finished.")
