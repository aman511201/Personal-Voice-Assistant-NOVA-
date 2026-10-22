import time
import sys
import queue
import collections
import numpy as np
import sounddevice as sd
import soundfile as sf

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class VoiceDetector:
    """
    Advanced Voice Activity Detection (VAD) and Audio Capture Engine for Nova.
    
    Key Features:
    1. Continuous non-blocking stream: Uses sounddevice.InputStream to eliminate
       audio gaps and dropped syllables.
    2. Dynamic Ambient Noise Calibration: Measures baseline background noise
       and sets adaptive thresholds for any room environment.
    3. Rolling Pre-Roll Ring Buffer: Buffers audio prior to speech onset so the
       initial syllables ("Hey", "Nova", "What") are never clipped.
    4. 3-State VAD Machine:
       - WAITING: Waits for speech without premature timeouts on initial silence.
       - ACTIVE: Records speech and tracks trailing silence.
       - FINISHED: Detects completion when trailing silence duration is reached.
    5. Live Terminal Volume Visualizer: Real-time visual feedback meter.
    6. Audio Post-Processing: DC offset removal, silence trimming, and peak normalization.
    """

    def __init__(
        self,
        sample_rate=16000,
        channels=1,
        chunk_duration=0.03,       # 30ms ultra-responsive chunk size
        pre_roll_duration=0.4,      # 400ms pre-speech audio buffer
        min_speech_duration=0.25,   # Captures fast one-word utterances ("yes", "no", "thanks")
        silence_duration=0.65,      # Fast 650ms trailing silence (750ms faster turnaround)
        max_wait_to_start=20.0,     # Max time to wait for user to start speaking
        max_speech_duration=30.0,   # Max total recording time
        sensitivity_multiplier=1.75,# Optimized sensitivity for crystal-clear detection
        min_threshold=0.009         # Lower floor for natural conversational volume
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.pre_roll_duration = pre_roll_duration
        self.pre_roll_chunks = max(1, int(self.pre_roll_duration / self.chunk_duration))
        self.min_speech_duration = min_speech_duration
        self.silence_duration = silence_duration
        self.max_wait_to_start = max_wait_to_start
        self.max_speech_duration = max_speech_duration
        self.sensitivity_multiplier = sensitivity_multiplier
        self.min_threshold = min_threshold

        self.ambient_noise_rms = 0.005
        self.speech_threshold = self.min_threshold

        # Queue for audio callback
        self._audio_queue = queue.Queue()

    def _audio_callback(self, indata, frames, time_info, status):
        """Non-blocking audio input stream callback."""
        if status:
            pass
        self._audio_queue.put(indata.copy())

    def calculate_rms(self, chunk):
        """Calculates Root Mean Square (RMS) volume of an audio chunk."""
        if len(chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))

    def calculate_zcr(self, chunk):
        """Calculates Zero-Crossing Rate to help distinguish voice from rumble."""
        if len(chunk) < 2:
            return 0.0
        signs = np.sign(chunk.flatten())
        diffs = np.abs(np.diff(signs))
        return float(np.mean(diffs > 0))

    def calibrate_ambient_noise(self, duration=0.5, show_progress=True):
        """
        Samples ambient room noise to establish a baseline noise floor.
        Automatically sets speech threshold tailored to the current environment.
        """
        if show_progress:
            print("🎙️ Calibrating microphone & ambient room noise...", end="", flush=True)

        frames_to_read = int(duration / self.chunk_duration)
        rms_values = []

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.chunk_size
            ) as stream:
                for _ in range(frames_to_read):
                    data, _ = stream.read(self.chunk_size)
                    rms = self.calculate_rms(data)
                    rms_values.append(rms)
        except Exception as e:
            if show_progress:
                print(f" (fallback calibration: {e})")
            self.ambient_noise_rms = 0.006
            self.speech_threshold = self.min_threshold
            return self.speech_threshold

        if rms_values:
            # Use 80th percentile of noise to avoid being misled by one transient sound
            self.ambient_noise_rms = float(np.percentile(rms_values, 80))
            self.speech_threshold = max(
                self.min_threshold,
                self.ambient_noise_rms * self.sensitivity_multiplier
            )
        else:
            self.ambient_noise_rms = 0.006
            self.speech_threshold = self.min_threshold

        if show_progress:
            print(f" Done! (Noise Floor: {self.ambient_noise_rms:.4f}, Threshold: {self.speech_threshold:.4f})")

        return self.speech_threshold

    def _render_meter(self, volume, is_speaking, silence_time):
        """Renders a responsive console audio meter."""
        bar_len = 16
        # Normalize volume roughly between 0 and 0.1
        normalized = min(1.0, volume / max(0.08, self.speech_threshold * 3))
        filled = int(normalized * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        if is_speaking:
            status = f"🎙️ Speaking... (silence: {silence_time:.1f}s / {self.silence_duration:.1f}s)"
        else:
            status = "👂 Listening... (waiting for you to speak)"

        print(f"\r{status} [{bar}] RMS: {volume:.4f}", end="", flush=True)

    def record_voice(self, output_path="voice_input.wav", show_meter=True):
        """
        Records voice using smart adaptive Voice Activity Detection.
        
        Returns:
            str: Path to saved audio file, or None if no speech was detected.
        """
        # Drain any leftover audio in queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        pre_roll_buffer = collections.deque(maxlen=self.pre_roll_chunks)
        recorded_chunks = []

        is_speaking = False
        speech_started_time = 0.0
        consecutive_voice_frames = 0
        consecutive_voice_required = 2  # 100ms of sustained energy to confirm speech

        silent_time = 0.0
        start_wait_time = time.time()

        print("\n🎤 Nova is listening (speak naturally)...")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ):
                while True:
                    try:
                        chunk = self._audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    volume = self.calculate_rms(chunk)

                    # State 1: Waiting for speech to begin
                    if not is_speaking:
                        pre_roll_buffer.append(chunk)

                        if volume >= self.speech_threshold:
                            consecutive_voice_frames += 1
                            if consecutive_voice_frames >= consecutive_voice_required:
                                is_speaking = True
                                speech_started_time = time.time()
                                silent_time = 0.0
                                # Flush pre-roll buffer to retain first syllable!
                                recorded_chunks.extend(list(pre_roll_buffer))
                        else:
                            consecutive_voice_frames = 0
                            # Check if user has stayed silent for too long without speaking
                            if (time.time() - start_wait_time) >= self.max_wait_to_start:
                                if show_meter:
                                    print("\n⏱️ Listening timed out (no speech detected).")
                                return None

                    # State 2: Speech is currently active
                    else:
                        recorded_chunks.append(chunk)
                        total_speech_duration = time.time() - speech_started_time

                        # Dynamic trailing silence: after sustained speech (>0.8s), drop silence cutoff to 0.50s for ultra-fast response
                        effective_silence_threshold = 0.50 if total_speech_duration > 0.8 else self.silence_duration
                        if volume < self.speech_threshold:
                            silent_time += self.chunk_duration
                        else:
                            silent_time = 0.0

                        # Check if trailing silence exceeded requirement
                        if silent_time >= effective_silence_threshold:
                            if show_meter:
                                print("\n🛑 Speech finished!")
                            break

                        # Safety cap on max speech duration
                        if total_speech_duration >= self.max_speech_duration:
                            if show_meter:
                                print("\n⏱️ Maximum speech duration reached.")
                            break

                    if show_meter:
                        self._render_meter(volume, is_speaking, silent_time)

        except Exception as e:
            print(f"\n❌ Audio recording error: {e}")
            return None

        # Validate that we actually captured audible speech
        if not recorded_chunks:
            return None

        audio_data = np.concatenate(recorded_chunks, axis=0).flatten()
        duration = len(audio_data) / self.sample_rate

        if duration < self.min_speech_duration:
            if show_meter:
                print(f"\n⚠️ Audio too short ({duration:.2f}s), discarded.")
            return None

        # Post-Processing: DC offset removal & normalization
        audio_data = audio_data - np.mean(audio_data)
        peak = np.max(np.abs(audio_data))
        if peak > 0.001:
            # Normalize to 0.90 (-0.9 dBFS) to prevent clipping while ensuring clear speech
            audio_data = (audio_data / peak) * 0.90

        # Save to disk
        try:
            sf.write(output_path, audio_data, self.sample_rate, subtype="PCM_16")
            if show_meter:
                print(f"✅ Audio processed & saved ({duration:.1f}s) -> {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error saving audio file: {e}")
            return None

    def record_push_to_talk(self, output_path="voice_input.wav"):
        """
        Alternative Push-to-Talk recording mode for very noisy environments.
        Press Enter to start recording, speak, then press Enter to stop.
        """
        input("\n[Push-to-Talk] Press Enter to start speaking...")
        print("🎤 Recording... Press Enter when finished.")

        recorded_chunks = []
        is_recording = True

        def wait_for_enter():
            nonlocal is_recording
            input()
            is_recording = False

        import threading
        stop_thread = threading.Thread(target=wait_for_enter, daemon=True)
        stop_thread.start()

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ):
                while is_recording:
                    try:
                        chunk = self._audio_queue.get(timeout=0.1)
                        recorded_chunks.append(chunk)
                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None

        if not recorded_chunks:
            return None

        audio_data = np.concatenate(recorded_chunks, axis=0).flatten()
        audio_data = audio_data - np.mean(audio_data)
        peak = np.max(np.abs(audio_data))
        if peak > 0.001:
            audio_data = (audio_data / peak) * 0.90

        sf.write(output_path, audio_data, self.sample_rate, subtype="PCM_16")
        duration = len(audio_data) / self.sample_rate
        print(f"✅ Audio recorded ({duration:.1f}s) -> {output_path}")
        return output_path
