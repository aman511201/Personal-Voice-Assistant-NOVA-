import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from voice_detector import VoiceDetector


def main():
    print("=" * 60)
    print("🎙️ NOVA ADVANCED VOICE DETECTION TEST (VAD BENCHMARK)")
    print("=" * 60)
    print("1. Automatic Adaptive VAD (Hands-free, stops when silent)")
    print("2. Push-to-Talk (Press Enter to start, Press Enter to stop)")
    print("=" * 60)

    choice = input("Select mode (1 or 2, default: 1): ").strip()

    detector = VoiceDetector(
        sample_rate=16000,
        channels=1,
        chunk_duration=0.05,
        pre_roll_duration=0.4,
        silence_duration=1.4,
        min_speech_duration=0.4
    )

    if choice == "2":
        output_file = detector.record_push_to_talk("voice_input.wav")
    else:
        # Step 1: Calibrate ambient noise floor
        detector.calibrate_ambient_noise(duration=0.6)
        print("\nReady! Speak into your microphone...")
        output_file = detector.record_voice("voice_input.wav", show_meter=True)

    if output_file and os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"\n🎉 Success! Audio cleanly recorded and normalized.")
        print(f"📁 Output file: {output_file} ({file_size:,} bytes)")
    else:
        print("\n⚠️ No valid audio was captured.")


if __name__ == "__main__":
    main()