#!/usr/bin/env python3
"""Per-frame speech-probability parity of a `vad.silero.v1` packet (plowrt's host executor,
`--asr-vad-packet`) vs the Silero VAD TorchScript reference, on a fixed audio set: LibriSpeech clips (an ASR manifest), the first clips
joined by pauses of silence and of noise, and pure silence / noise segments.

  python scripts/asr/silero_vad_parity.py --packet DIR/silero_vad.pkt --jit silero_vad.jit \
      --manifest MANIFEST.json --runner TARGET/release/examples/asr_vad_check --out DIR \
      [--max-dp 1e-4] [--min-agree 0.999]

Writes DIR/audio/*.f32 (16 kHz f32le), DIR/ref.json, DIR/plow.json and DIR/parity.json, and
prints `VAD_PARITY max_dp=.. mean_dp=.. agree=.. flips=.. frames=..` (decisions at 0.5).
Exit 1 when a threshold fails.
"""
import argparse
import json
import os
import subprocess
import sys
import wave

import numpy as np

RATE = 16000
FRAME = 512


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getframerate() != RATE or w.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected 16 kHz s16 WAV")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
    return x


def audio_set(manifest):
    clips = {}
    for e in json.load(open(manifest)):
        clips[os.path.splitext(os.path.basename(e["path"]))[0]] = read_wav(e["path"])
    rng = np.random.default_rng(1234)
    names = sorted(clips)
    a, b, c = (clips[n] for n in names[:3])
    noise = lambda s, db: (rng.standard_normal(int(s * RATE)) * 10 ** (db / 20)).astype(np.float32)
    zeros = lambda s: np.zeros(int(s * RATE), np.float32)
    extra = {
        "pauses_silence": np.concatenate([zeros(0.5), a, zeros(2.0), b, zeros(1.5), c, zeros(1.0)]),
        "pauses_noise": np.concatenate([noise(0.5, -50), a + noise(len(a) / RATE, -50), noise(2.0, -45), b, noise(1.0, -45)]),
        "silence": zeros(8.0),
        "noise_white_m40": noise(8.0, -40),
        "noise_white_m25": noise(8.0, -25),
        "hum_60hz": (0.05 * np.sin(2 * np.pi * 60 * np.arange(8 * RATE) / RATE)).astype(np.float32),
    }
    clips.update(extra)
    return clips


def reference(jit, clips):
    import torch

    torch.set_num_threads(1)
    m = torch.jit.load(jit, map_location="cpu").eval()
    out = {}
    with torch.no_grad():
        for name, x in clips.items():
            m.reset_states()
            t = torch.from_numpy(x)
            out[name] = [float(m(t[i:i + FRAME].unsqueeze(0), RATE)[0, 0]) for i in range(0, len(x) - FRAME + 1, FRAME)]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--packet", required=True)
    p.add_argument("--jit", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--runner", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-dp", type=float, default=1e-4)
    p.add_argument("--min-agree", type=float, default=0.999)
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "audio"), exist_ok=True)
    clips = audio_set(a.manifest)
    paths = {}
    for name, x in clips.items():
        paths[name] = os.path.join(a.out, "audio", name + ".f32")
        x.astype("<f4").tofile(paths[name])
    ref = reference(a.jit, clips)
    json.dump(ref, open(os.path.join(a.out, "ref.json"), "w"))
    plow_json = os.path.join(a.out, "plow.json")
    subprocess.run([a.runner, a.packet, "probs", plow_json, *paths.values()], check=True)
    got = {name: json.load(open(plow_json))[path] for name, path in paths.items()}
    rows, dps, agree, flips = {}, [], 0, []
    for name in clips:
        r, g = np.array(ref[name]), np.array(got[name])
        if len(r) != len(g):
            raise SystemExit(f"{name}: {len(g)} plow frames vs {len(r)} reference")
        d = np.abs(r - g)
        dps.append(d)
        same = (r >= a.threshold) == (g >= a.threshold)
        agree += int(same.sum())
        flips += [(name, int(i), float(r[i]), float(g[i])) for i in np.flatnonzero(~same)]
        rows[name] = dict(frames=len(r), max_dp=float(d.max()), speech_frames=int((r >= a.threshold).sum()))
    d = np.concatenate(dps)
    res = dict(frames=int(d.size), max_dp=float(d.max()), mean_dp=float(d.mean()), agree=agree / d.size,
               flips=flips, threshold=a.threshold, clips=rows,
               max_dp_limit=a.max_dp, min_agree_limit=a.min_agree)
    res["pass"] = res["max_dp"] <= a.max_dp and res["agree"] >= a.min_agree
    json.dump(res, open(os.path.join(a.out, "parity.json"), "w"), indent=1)
    print(f"VAD_PARITY max_dp={res['max_dp']:.3e} mean_dp={res['mean_dp']:.3e} agree={res['agree']:.6f} "
          f"flips={len(flips)} frames={res['frames']} clips={len(clips)} pass={res['pass']}")
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
