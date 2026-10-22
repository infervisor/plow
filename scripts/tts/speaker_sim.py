#!/usr/bin/env python3
"""Speaker similarity of generated speech to a reference voice: cosine of Chatterbox's
VoiceEncoder speaker embeddings (ResembleAI/chatterbox `ve.safetensors`, 16 kHz).

Prints, per group of wavs, the median cosine to REF. A cloned voice should sit near the official
clone's similarity and well above the model's default voice.

  venv-cbx/bin/python scripts/tts/speaker_sim.py REF.wav LABEL=a.wav,b.wav [LABEL2=c.wav ...]
"""
import sys

import librosa
import numpy as np
import torch


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    from chatterbox.models.voice_encoder import VoiceEncoder
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(hf_hub_download("ResembleAI/chatterbox", "ve.safetensors")))
    ve.eval()

    def embed(paths):
        wavs = [librosa.load(p, sr=16000)[0] for p in paths]
        with torch.inference_mode():
            return ve.embeds_from_wavs(wavs, sample_rate=16000)

    ref = embed([sys.argv[1]])[0]
    for arg in sys.argv[2:]:
        label, files = arg.split("=", 1)
        e = embed([f for f in files.split(",") if f])
        cos = e @ ref / (np.linalg.norm(e, axis=1) * np.linalg.norm(ref))
        print(f"{label}: n={len(cos)} median_cos={float(np.median(cos)):.3f} min={float(cos.min()):.3f}")


if __name__ == "__main__":
    main()
