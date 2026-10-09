"""Export Silero VAD v5's 16 kHz weights (from the `silero-vad` package's TorchScript model) to a
safetensors checkpoint that `asr_silero_vad_compile` lowers into a `vad.silero.v1` packet.

  python scripts/asr/silero_export.py <out dir>      (needs torch, silero-vad, safetensors)

Writes <out dir>/model.safetensors and <out dir>/reference.json: per-frame speech probabilities
for a fixed test signal, which the packet's Rust executor is checked against."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from silero_vad import load_silero_vad

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
vad = load_silero_vad()
model = vad._model
state = {k: v.detach().float().contiguous() for k, v in model.state_dict().items()}
names = {
    "stft.forward_basis_buffer": "vad.stft.basis",
    "decoder.rnn.weight_ih": "vad.lstm.weight_ih",
    "decoder.rnn.weight_hh": "vad.lstm.weight_hh",
    "decoder.rnn.bias_ih": "vad.lstm.bias_ih",
    "decoder.rnn.bias_hh": "vad.lstm.bias_hh",
    "decoder.decoder.2.weight": "vad.out.weight",
    "decoder.decoder.2.bias": "vad.out.bias",
}
for i in range(4):
    names[f"encoder.{i}.reparam_conv.weight"] = f"vad.encoder.{i}.weight"
    names[f"encoder.{i}.reparam_conv.bias"] = f"vad.encoder.{i}.bias"
if set(names) != set(state):
    raise SystemExit(f"unexpected Silero weights: {sorted(set(state) ^ set(names))}")
save_file({names[k]: v for k, v in state.items()}, out / "model.safetensors",
          metadata={"model": "silero-vad-v5", "sample_rate": "16000", "frame": "512", "context": "64"})

# Reference: a deterministic signal (tone bursts, noise, silence) through the official model.
rng = np.random.default_rng(7)
t = np.arange(16000 * 4) / 16000
signal = 0.02 * rng.standard_normal(t.size)
signal[16000:32000] += 0.3 * np.sin(2 * np.pi * 220 * t[16000:32000]) * np.sin(2 * np.pi * 3 * t[16000:32000])
signal[40000:48000] = 0
signal = signal.astype(np.float32)
vad.reset_states()
probs = []
with torch.no_grad():
    for start in range(0, signal.size - 511, 512):
        probs.append(float(vad(torch.from_numpy(signal[start:start + 512]).unsqueeze(0), 16000)))
json.dump({"signal_seed": 7, "signal": signal.tolist(), "probabilities": probs}, open(out / "reference.json", "w"))
print(f"wrote {out}/model.safetensors ({len(state)} tensors) and {len(probs)} reference probabilities")
