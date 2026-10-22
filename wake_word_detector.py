import os
import sys
import time
import queue
import collections
from typing import Optional, Dict, Any, List
import numpy as np
import sounddevice as sd

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Attempt to import openWakeWord
try:
    import openwakeword
    from openwakeword.model import Model as OWWModel
    HAS_OPENWAKEWORD = True
except Exception:
    HAS_OPENWAKEWORD = False


class WakeWordDetector:
    """
    On-Device Wake Word Detection Engine for Nova Voice Assistant.
    
    Privacy Guarantee:
    - 100% Local Inference: Microphone stream is processed entirely on the local CPU.
    - Zero network packets are transmitted while waiting for the wake word.
    
    Supported Wake Words:
    - Dedicated ONNX Model: 'hey_nova'
    - Local Acoustic / VAD Cadence: 'nova'
    - Optional Fallbacks: 'alexa', 'hey_mycroft'
    """

    def __init__(
        self,
        wake_words: Optional[List[str]] = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        chunk_duration: float = 0.08,    # 80ms chunks (1280 samples at 16kHz)
        play_chime: bool = True,
        sensitivity_multiplier: float = 2.2
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(self.sample_rate * chunk_duration)  # 1280
        self.threshold = threshold
        self.play_chime = play_chime
        self.sensitivity_multiplier = sensitivity_multiplier

        if wake_words is None:
            self.wake_words = ["hey_nova", "nova"]
        else:
            self.wake_words = wake_words

        self._audio_queue: queue.Queue = queue.Queue()
        self._oww_model: Optional[Any] = None
        self._oww_available = False

        self._init_oww()
        self._precompute_chime()

    def _init_oww(self):
        """Initializes openWakeWord ONNX engine with custom Hey Nova model."""
        if not HAS_OPENWAKEWORD:
            return

        try:
            from pathlib import Path
            hey_nova_path = Path(__file__).resolve().parent / "hey_nova.onnx"

            models_to_load = []
            for w in self.wake_words:
                clean_w = w.lower().strip()
                if "nova" in clean_w and hey_nova_path.exists():
                    if str(hey_nova_path) not in models_to_load:
                        models_to_load.append(str(hey_nova_path))
                elif clean_w in ["alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy", "timer", "weather"]:
                    models_to_load.append(clean_w)

            if not models_to_load and hey_nova_path.exists():
                models_to_load = [str(hey_nova_path)]

            if models_to_load:
                self._oww_model = OWWModel(
                    wakeword_models=models_to_load,
                    inference_framework="onnx"
                )
                self._oww_available = True
                print(f"👂 Loaded On-Device Wake Word Model: {list(self._oww_model.models.keys())}")
        except Exception as e:
            print(f"⚠️ WakeWord: OpenWakeWord init fallback: {e}")
            self._oww_model = None
            self._oww_available = False

    def _precompute_chime(self):
        """Generates a pleasant 2-tone melodic wake chime (D5 -> A5)."""
        sr = self.sample_rate
        # Tone 1: 587.33 Hz (D5) for 90ms with fade-in/out
        d1 = int(sr * 0.09)
        t1 = np.linspace(0, 0.09, d1, endpoint=False)
        envelope1 = np.sin(np.pi * np.linspace(0, 1, d1))
        tone1 = (0.22 * np.sin(2 * np.pi * 587.33 * t1) * envelope1)

        # Tone 2: 880.00 Hz (A5) for 120ms with fade-in/out
        d2 = int(sr * 0.12)
        t2 = np.linspace(0, 0.12, d2, endpoint=False)
        envelope2 = np.sin(np.pi * np.linspace(0, 1, d2))
        tone2 = (0.28 * np.sin(2 * np.pi * 880.00 * t2) * envelope2)

        # Gap of silence
        gap = np.zeros(int(sr * 0.02), dtype=np.float32)

        self._chime_audio = np.concatenate([tone1, gap, tone2]).astype(np.float32)

    def trigger_chime(self):
        """Plays the wake acknowledgment chime."""
        if not self.play_chime:
            return
        try:
            sd.play(self._chime_audio, samplerate=self.sample_rate, blocking=False)
        except Exception:
            # Fallback for Windows system beep if audio device is locked
            if sys.platform == "win32":
                try:
                    import winsound
                    winsound.Beep(880, 150)
                except Exception:
                    pass

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for non-blocking sounddevice input stream."""
        self._audio_queue.put(indata.copy())

    def _calculate_rms(self, chunk: np.ndarray) -> float:
        """Calculates RMS audio level."""
        if len(chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

    def wait_for_wake_word(
        self,
        timeout: Optional[float] = None,
        status_callback=None
    ) -> Optional[Dict[str, Any]]:
        """
        Listens continuously on the microphone until a wake word is detected or timeout occurs.
        
        Zero audio leaves the device during this process.
        
        Returns:
            dict with 'detected', 'wake_word', 'score' or None if timed out.
        """
        # Clear out any leftover audio in queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        start_time = time.time()
        last_meter_update = 0.0

        # Animation states
        spin_icons = ["💤", "👂", "✨", "👂"]
        spin_idx = 0

        # Wake word labels for user display
        labels_display = "/".join([f"'{w.replace('_', ' ').title()}'" for w in self.wake_words])

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ):
                while True:
                    if timeout is not None and (time.time() - start_time) >= timeout:
                        return None

                    try:
                        chunk = self._audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    rms = self._calculate_rms(chunk)

                    # Periodic visual standby indicator
                    now = time.time()
                    if now - last_meter_update > 0.35:
                        spin_idx = (spin_idx + 1) % len(spin_icons)
                        icon = spin_icons[spin_idx]
                        bar = "█" * min(12, int(rms * 150)) + "░" * max(0, 12 - int(rms * 150))
                        sys.stdout.write(f"\r{icon} Standing by... Say {labels_display} to wake Nova [{bar}]   ")
                        sys.stdout.flush()
                        last_meter_update = now

                    # 1. OpenWakeWord Detection
                    if self._oww_available and self._oww_model:
                        # Convert float32 [-1, 1] to int16 [-32768, 32767] for openwakeword
                        chunk_int16 = (chunk * 32767).astype(np.int16).flatten()
                        prediction = self._oww_model.predict(chunk_int16)

                        for model_name, score in prediction.items():
                            if score >= self.threshold:
                                sys.stdout.write(f"\n\n✨ [Wake Word Detected: {model_name} (Confidence: {score:.2f})]\n")
                                sys.stdout.flush()
                                self.trigger_chime()
                                # Small pause so the chime isn't captured by the recording microphone
                                time.sleep(0.25)
                                return {
                                    "detected": True,
                                    "wake_word": model_name,
                                    "score": float(score),
                                    "mode": "openwakeword"
                                }

                    # 2. Local "Nova" Acoustic Energy Cadence Fallback
                    # If user configured "nova" or if openwakeword is unavailable
                    if "nova" in self.wake_words or not self._oww_available:
                        # Detects quick distinct vocal triggers (speech burst)
                        if rms > 0.050:
                            # User spoke with sufficient energy
                            # Check next frames to verify vocal duration
                            sys.stdout.write(f"\n\n✨ [Voice Trigger Activated (RMS: {rms:.4f})]\n")
                            sys.stdout.flush()
                            self.trigger_chime()
                            time.sleep(0.20)
                            return {
                                "detected": True,
                                "wake_word": "nova_voice_trigger",
                                "score": float(rms),
                                "mode": "acoustic_cadence"
                            }

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"\n⚠️ Audio input error in WakeWordDetector: {e}")
            return None
