#!/usr/bin/env python3
"""Write the ECAPA parity fixtures for crates/plowrt/src/tts/ecapa.rs (`tests::parity`).

    <maya2 venv>/bin/python scripts/tts/ecapa_cases.py [OUT_DIR] [--audio /home/ssm-user/asr-work/audio]

OUT_DIR (default crates/plowrt/tests/data/ecapa) gets:
  <clip>.s16          24 kHz mono PCM, int16 LE; the reference input is s16 / 32768
  fbank_ls01.f32      Fbank(n_mels=80) of ls01 before sentence normalisation, [T, 80] f32 LE
  cases.json          {"cases": [{"name", "parts": [clip, ...], "samples", "embedding": [192]}], "fbank": {...}}
A case's input is the concatenation of its parts truncated to `samples`. The embedding is Maya's `Ecapa.__call__`
(maya3/api.py) evaluated on CPU in fp32.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

SR = 24000


def load_ecapa():
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.processing.features import InputNormalization
    ckpt = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/*/embedding_model.ckpt")))[0]
    fbank = Fbank(n_mels=80).eval()
    norm = InputNormalization(norm_type="sentence", std_norm=False).eval()
    emb = ECAPA_TDNN(input_size=80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1],
                     dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192).eval()
    emb.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)

    @torch.inference_mode()
    def call(w24):
        w16 = torch.from_numpy(resample_poly(np.asarray(w24, np.float32), 2, 3).astype(np.float32))[None]
        lens = torch.ones(1)
        fb = fbank(w16)
        e = emb(norm(fb, lens), lens).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).numpy(), fb[0].numpy()
    return call


def quantize(x):
    return np.round(np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="crates/plowrt/tests/data/ecapa")
    ap.add_argument("--audio", default="/home/ssm-user/asr-work/audio")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)

    t3 = np.arange(3 * SR) / SR
    t4 = np.arange(4 * SR) / SR
    clips = {}
    for n in ("ls00", "ls01"):
        x, sr = sf.read(f"{a.audio}/{n}.wav", dtype="float32")
        clips[n] = resample_poly(x, SR // 1000, sr // 1000) if sr != SR else x
    clips["chirp"] = 0.3 * np.sin(2 * np.pi * (100 * t3 + (6000 - 100) / (2 * 3) * t3 ** 2))
    clips["noise_tone"] = 0.05 * np.random.default_rng(0).standard_normal(4 * SR) + 0.2 * np.sin(2 * np.pi * 220 * t4)
    clips["short"] = clips["ls01"][int(0.5 * SR):int(0.7 * SR)]
    pcm = {}
    for n, x in clips.items():
        q = quantize(x)
        q.tofile(od / f"{n}.s16")
        pcm[n] = q.astype(np.float32) / 32768.0

    ecapa = load_ecapa()
    specs = [("ls00", ["ls00"], None), ("ls01", ["ls01"], None), ("chirp", ["chirp"], None),
             ("noise_tone", ["noise_tone"], None), ("short", ["short"], None),
             ("ten", ["ls00", "ls01"], 10 * SR),
             ("long", ["ls00", "ls01", "chirp", "noise_tone", "ls00", "ls01"], 25 * SR)]
    cases = []
    for name, parts, n in specs:
        x = np.concatenate([pcm[p] for p in parts])[:n]
        e, fb = ecapa(x)
        cases.append({"name": name, "parts": parts, "samples": int(len(x)), "embedding": [float(v) for v in e]})
        if name == "ls01":
            fb.astype("<f4").tofile(od / "fbank_ls01.f32")
            fbank = {"case": "ls01", "frames": int(fb.shape[0]), "file": "fbank_ls01.f32"}
        print(f"{name}: {len(x)} samples")
    (od / "cases.json").write_text(json.dumps({"cases": cases, "fbank": fbank}) + "\n")


if __name__ == "__main__":
    main()
