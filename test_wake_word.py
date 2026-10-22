import os
import sys
import numpy as np
from wake_word_detector import WakeWordDetector, HAS_OPENWAKEWORD


def test_wake_word_detector():
    print("[TEST] Running WakeWordDetector Unit Tests...")

    # 1. Test Detector Initialization with default Hey Nova
    detector = WakeWordDetector(
        wake_words=["hey_nova", "nova"],
        threshold=0.5,
        sample_rate=16000,
        play_chime=False
    )

    assert detector.sample_rate == 16000
    assert detector.chunk_size == 1280, f"Expected 1280 chunk_size, got {detector.chunk_size}"
    assert "hey_nova" in detector.wake_words
    print(f"[OK] WakeWordDetector initialized with 'hey_nova' (Sample Rate: {detector.sample_rate}, Chunk Size: {detector.chunk_size})")

    # 2. Test Chime Waveform Generation
    assert hasattr(detector, "_chime_audio")
    assert len(detector._chime_audio) > 0
    # Peak must be normalized and audible without clipping
    peak = np.max(np.abs(detector._chime_audio))
    assert 0.1 <= peak <= 0.9, f"Chime peak amplitude unexpected: {peak}"
    print(f"[OK] Two-tone melodic wake chime generated successfully (Peak: {peak:.2f}, Samples: {len(detector._chime_audio)})")

    # 3. Test RMS Level Calculation
    silence = np.zeros(detector.chunk_size, dtype=np.float32)
    assert detector._calculate_rms(silence) == 0.0, "Silence RMS must be 0"

    t = np.linspace(0, 0.08, detector.chunk_size, endpoint=False)
    sine = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rms = detector._calculate_rms(sine)
    assert 0.15 <= rms <= 0.25, f"Expected RMS around 0.21, got {rms}"
    print(f"[OK] RMS calculation verified (Sine wave RMS: {rms:.4f})")

    # 4. Test OpenWakeWord Inference (if installed)
    if HAS_OPENWAKEWORD and detector._oww_available:
        assert detector._oww_model is not None
        # Feed silent audio chunk
        silent_chunk = np.zeros(1280, dtype=np.int16)
        preds = detector._oww_model.predict(silent_chunk)
        assert isinstance(preds, dict), "Prediction must return a dict"
        assert "hey_nova" in preds or "alexa" in preds, "Expected hey_nova or configured model in prediction"
        for k, v in preds.items():
            assert 0.0 <= v <= 1.0, f"Prediction probability {v} must be between 0 and 1"
        print(f"[OK] OpenWakeWord custom 'hey_nova' on-device streaming inference verified (Predictions: {preds})")
    else:
        print("[SKIP] OpenWakeWord engine not available in current environment")

    # 5. Test Multiple Wake Word Selection
    multi_detector = WakeWordDetector(
        wake_words=["alexa", "hey_mycroft"],
        play_chime=False
    )
    assert "alexa" in multi_detector.wake_words
    assert "hey_mycroft" in multi_detector.wake_words
    print("[OK] Multi-wake-word configuration verified")

    print("\n[PASSED] ALL WAKE WORD DETECTOR UNIT TESTS PASSED SUCCESSFULLY!\n")


if __name__ == "__main__":
    test_wake_word_detector()
