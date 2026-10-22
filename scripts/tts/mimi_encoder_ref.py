#!/usr/bin/env python3
"""Breeze TTS 2 reference-audio encoder (Qwen3-TTS-Tokenizer-12Hz encoder = HF Mimi), fp32 CPU.

Runs the official `Qwen3TTSTokenizer.encode` on each wav and saves, per wav, in OUT/<stem>/:
  wav24k.npy  [n] f32       mono 24 kHz input the encoder saw (after the wrapper's resample)
  seanet.npy  [T25][512]    SEANet output (25 Hz rows)
  xf.npy      [T25][512]    encoder_transformer output
  down.npy    [T12][512]    downsample output (12.5 Hz rows)
  codes.npy   [F][16] i32   codes the model returns (F = ceil(n / 1920))

  venv-breeze/bin/python scripts/tts/mimi_encoder_ref.py OUT REVISION a.wav [b.wav ...] \
      [--golden newtts/breeze/golden/clone/input_values.npy]
"""
import argparse
import os

import numpy as np
import torch

REPO = "models--BreezeBlue--Breeze-TTS-2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("revision")
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--golden", help="codes [1][F][16] the official clone run encoded from wavs[0]")
    args = ap.parse_args()
    from qwen_tts import Qwen3TTSTokenizer
    import soundfile as sf

    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    tok = Qwen3TTSTokenizer.from_pretrained(
        os.path.join(hub, REPO, "snapshots", args.revision, "audio_tokenizer"), device_map="cpu")
    enc = tok.model.encoder
    seen = {}

    def keep(name):
        def hook(_m, _i, o):
            o = o[0] if isinstance(o, tuple) else o
            o = getattr(o, "last_hidden_state", o)
            seen[name] = o.detach().float().numpy()
        return hook

    enc.encoder.register_forward_hook(keep("seanet"))
    enc.encoder_transformer.register_forward_hook(keep("xf"))
    enc.downsample.register_forward_hook(keep("down"))
    for i, path in enumerate(args.wavs):
        wav, sr = sf.read(path, always_2d=True, dtype="float32")
        wav = np.mean(wav, axis=1)
        wav24 = tok._normalize_audio_inputs(wav, sr=sr)[0]
        codes = tok.encode(wav, sr=sr)["audio_codes"][0].numpy().astype(np.int32)
        d = os.path.join(args.out, os.path.splitext(os.path.basename(path))[0])
        os.makedirs(d, exist_ok=True)
        np.save(f"{d}/wav24k.npy", wav24)
        # seanet / down are [1][C][T]; xf is [1][T][C].
        np.save(f"{d}/seanet.npy", seen["seanet"][0].T)
        np.save(f"{d}/xf.npy", seen["xf"][0])
        np.save(f"{d}/down.npy", seen["down"][0].T)
        np.save(f"{d}/codes.npy", codes)
        msg = f"{path}: {len(wav24)} samples -> codes {codes.shape}"
        if i == 0 and args.golden:
            g = np.load(args.golden)[0].astype(np.int32)
            same = (g == codes).all(axis=1) if g.shape == codes.shape else None
            msg += f"; golden {g.shape}: " + (
                f"{int(same.sum())}/{len(same)} frames identical" if same is not None else "shape differs")
        print(msg, flush=True)


if __name__ == "__main__":
    main()
