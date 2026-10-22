import os
import sys
import numpy as np
import torch
import torch.nn as nn
from openwakeword.utils import AudioFeatures

# Force UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def synthesize_hey_nova(sr=16000, f0=150, duration=0.85):
    """Synthesizes an acoustic approximation of 'Hey Nova' using formant synthesis."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros_like(t)

    # 1. Pitch pulse train (glottal source)
    glottal = np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(4 * np.pi * f0 * t) + 0.25 * np.sin(6 * np.pi * f0 * t)

    # Segment 1: "Hey" (0.0 to 0.25s) - F1=450, F2=2100
    m_hey = (t >= 0.05) & (t < 0.28)
    e_hey = np.sin(np.pi * (t[m_hey] - 0.05) / 0.23)
    audio[m_hey] += glottal[m_hey] * e_hey * (np.sin(2 * np.pi * 450 * t[m_hey]) + 0.4 * np.sin(2 * np.pi * 2100 * t[m_hey]))

    # Segment 2: "No" (0.28 to 0.55s) - F1=480, F2=1050
    m_no = (t >= 0.28) & (t < 0.55)
    e_no = np.sin(np.pi * (t[m_no] - 0.28) / 0.27)
    audio[m_no] += glottal[m_no] * e_no * (np.sin(2 * np.pi * 480 * t[m_no]) + 0.5 * np.sin(2 * np.pi * 1050 * t[m_no]))

    # Segment 3: "va" (0.55 to 0.80s) - Frication + F1=550, F2=1400
    m_va = (t >= 0.55) & (t < 0.80)
    e_va = np.sin(np.pi * (t[m_va] - 0.55) / 0.25)
    frication = np.random.normal(0, 0.2, len(t[m_va]))
    audio[m_va] += e_va * (glottal[m_va] * (np.sin(2 * np.pi * 550 * t[m_va]) + 0.3 * np.sin(2 * np.pi * 1400 * t[m_va])) + frication * 0.15)

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0.001:
        audio = (audio / peak) * 0.85
    return audio


def build_hey_nova_model(output_path="hey_nova.onnx"):
    print("🚀 Synthesizing acoustic training data for 'Hey Nova'...")
    af = AudioFeatures(inference_framework="onnx")

    pos_samples = []
    # Generate 40 pitch and duration variations of "Hey Nova"
    for f0 in [110, 130, 150, 175, 200, 220, 240]:
        for dur in [0.75, 0.85, 0.95]:
            wav = synthesize_hey_nova(f0=f0, duration=dur)
            # Pad with silence to 1.5s
            padded = np.zeros(int(16000 * 1.5), dtype=np.float32)
            padded[int(16000 * 0.2):int(16000 * 0.2) + len(wav)] = wav
            wav_int16 = (padded * 32767).astype(np.int16)

            af.reset()
            # Feed in 1280 chunks
            for start in range(0, len(wav_int16) - 1280, 1280):
                af(wav_int16[start:start + 1280])

            feats = af.get_features()
            if feats is not None and feats.shape == (1, 16, 96):
                pos_samples.append(feats[0])

    print(f"✅ Generated {len(pos_samples)} positive 'Hey Nova' feature vectors.")

    # Generate negative samples: silence, random noise, and non-target tones
    neg_samples = []
    for _ in range(len(pos_samples) * 3):
        noise = np.random.normal(0, 0.1, (16, 96)).astype(np.float32)
        neg_samples.append(noise)

    # Non-target speech tones
    for f in [300, 800, 1800, 3000]:
        t = np.linspace(0, 1.5, int(16000 * 1.5))
        sine_wav = (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)
        wav_int16 = (sine_wav * 32767).astype(np.int16)
        af.reset()
        for start in range(0, len(wav_int16) - 1280, 1280):
            af(wav_int16[start:start + 1280])
        feats = af.get_features()
        if feats is not None and feats.shape == (1, 16, 96):
            neg_samples.append(feats[0])

    X = np.concatenate([pos_samples, neg_samples], axis=0)  # (N, 16, 96)
    y = np.array([1.0] * len(pos_samples) + [0.0] * len(neg_samples), dtype=np.float32).reshape(-1, 1)

    # 3-Layer MLP (openWakeWord standard)
    class HeyNovaModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(16 * 96, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.net(x)

    model = HeyNovaModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()

    inputs = torch.tensor(X, dtype=torch.float32)
    labels = torch.tensor(y, dtype=torch.float32)

    for epoch in range(60):
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    print(f"🎯 Model trained. Loss: {loss.item():.4f}")

    dummy = torch.randn(1, 16, 96, dtype=torch.float32)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["onnx::Flatten_0"],
        output_names=["output"]
    )
    print(f"✨ Successfully exported {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")
    return output_path


if __name__ == "__main__":
    build_hey_nova_model()
