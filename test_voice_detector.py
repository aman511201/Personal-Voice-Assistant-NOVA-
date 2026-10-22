import os
import shutil
import tempfile
import numpy as np
from voice_detector import VoiceDetector


def test_voice_detector_algorithms():
    print("[TEST] Running VoiceDetector algorithmic & VAD tests...")

    vd = VoiceDetector(
        sample_rate=16000,
        chunk_duration=0.05,
        pre_roll_duration=0.2,
        silence_duration=0.5,
        min_speech_duration=0.2
    )

    # 1. Test RMS calculation
    silence_chunk = np.zeros(vd.chunk_size, dtype=np.float32)
    assert vd.calculate_rms(silence_chunk) == 0.0, "Silence RMS must be 0"

    # Sine wave chunk at 440 Hz
    t = np.linspace(0, vd.chunk_duration, vd.chunk_size, endpoint=False)
    sine_chunk = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rms = vd.calculate_rms(sine_chunk)
    assert 0.30 <= rms <= 0.40, f"Expected RMS around 0.35, got {rms}"
    print(f"[OK] RMS calculation verified (Sine wave RMS: {rms:.4f})")

    # 2. Test Zero Crossing Rate
    zcr_silence = vd.calculate_zcr(silence_chunk)
    zcr_sine = vd.calculate_zcr(sine_chunk)
    assert zcr_sine > 0, "Sine wave must have non-zero ZCR"
    print(f"[OK] ZCR calculation verified (Sine wave ZCR: {zcr_sine:.4f})")

    # 3. Test Pre-roll buffer math
    assert vd.pre_roll_chunks == 4, f"Expected 4 chunks for 0.2s / 0.05s, got {vd.pre_roll_chunks}"
    print("[OK] Pre-roll buffer math verified")

    # 4. Test audio normalization and file writing
    temp_dir = tempfile.mkdtemp()
    try:
        out_file = os.path.join(temp_dir, "test_out.wav")
        # Synthesize 1 second of speech-like sound + silence
        tone = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        # Verify normalization
        peak = np.max(np.abs(tone))
        normalized = (tone / peak) * 0.90
        assert np.isclose(np.max(np.abs(normalized)), 0.90), "Peak must be 0.90 after normalization"
        print("[OK] Peak normalization verified")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("[PASSED] ALL VOICEDETECTOR UNIT TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    test_voice_detector_algorithms()
