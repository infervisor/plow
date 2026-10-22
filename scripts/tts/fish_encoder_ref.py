#!/usr/bin/env python3
"""Torch reference of the Fish S2 Pro reference encoder over scripts/tts/fish_encoder_export.py's
export (the lowering's semantics, crates/devgen/src/fish_encoder.rs): 44.1 kHz PCM -> [F][10] codes.

  python scripts/tts/fish_encoder_ref.py EXPORT_DIR PCM.f32 OUT_CODES.npy [--codes GOLD.npy]
      [--rope-bf16] [--f64]

--rope-bf16 rounds the RoPE cos/sin table to bf16 as the official module does (plow rotates with
fp32 angles); --codes prints the per-codebook match against official codes.
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("pcm")
    ap.add_argument("out")
    ap.add_argument("--codes")
    ap.add_argument("--rope-bf16", action="store_true")
    ap.add_argument("--f64", action="store_true")
    a = ap.parse_args()
    from safetensors.torch import load_file

    dt = torch.float64 if a.f64 else torch.float32
    W = {k: v.to(dt) for k, v in load_file(f"{a.export}/model.safetensors").items()}
    cfg = json.load(open(f"{a.export}/config.json"))
    pcm = np.fromfile(a.pcm, dtype=np.float32)
    fs = cfg["frame_samples"]
    frames = -(-len(pcm) // fs)
    x = torch.zeros(1, 1, frames * fs, dtype=dt)
    x[0, 0, :len(pcm)] = torch.from_numpy(pcm).to(dt)

    def conv(x, w, b, stride=1, dil=1, groups=1):
        k = W[w].shape[2]
        x = F.pad(x, ((k - 1) * dil + 1 - stride, 0))
        return F.conv1d(x, W[w], W[b], stride=stride, dilation=dil, groups=groups)

    def snake(x, a):
        al = W[a][: x.shape[1]][None, :, None]
        return x + (1.0 / (al + 1e-9)) * torch.sin(al * x) ** 2

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg["eps"]) * W[w]

    def transformer(x, o, layers, window):
        t, d = x.shape
        h, hd = cfg["heads"], cfg["head_dim"]
        inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
        ang = torch.arange(t, dtype=torch.float64)[:, None] * inv[None, :]
        cos, sin = torch.cos(ang), torch.sin(ang)
        if a.rope_bf16:
            cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        cos, sin = cos.to(dt), sin.to(dt)
        i = torch.arange(t)
        mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window)

        def rope(q):  # [t, h, hd] half-split
            q1, q2 = q[..., : hd // 2], q[..., hd // 2:]
            c, s = cos[:, None, :], sin[:, None, :]
            return torch.cat([q1 * c - q2 * s, q2 * c + q1 * s], -1)

        for l in range(layers):
            p = f"{o}.{l}"
            y = rms(x, f"{p}.attn_norm")
            qkv = y @ W[f"{p}.wqkv"].t()
            q, k, v = qkv.split(d, -1)
            q, k = rope(q.view(t, h, hd)), rope(k.view(t, h, hd))
            v = v.view(t, h, hd)
            att = torch.einsum("qhd,khd->hqk", q, k) / math.sqrt(hd)
            att = att.masked_fill(~mask[None], float("-inf")).softmax(-1)
            y = torch.einsum("hqk,khd->qhd", att, v).reshape(t, d)
            x = x + y @ W[f"{p}.wo_s"].t()
            y = rms(x, f"{p}.ffn_norm")
            x = x + (F.silu(y @ W[f"{p}.w1"].t()) * (y @ W[f"{p}.w3"].t())) @ W[f"{p}.w2_s"].t()
        return rms(x, f"{o}.norm")

    with torch.no_grad():
        x = conv(x, "enc.c0.w", "enc.c0.b")
        for i, s in enumerate(cfg["strides"]):
            o = f"enc.b{i}"
            for j, d in enumerate(cfg["dilations"]):
                r = f"{o}.ru{j}"
                y = conv(snake(x, f"{r}.a1"), f"{r}.c1.w", f"{r}.c1.b", dil=d)
                x = x + conv(snake(y, f"{r}.a2"), f"{r}.c2.w", f"{r}.c2.b")
            x = conv(snake(x, f"{o}.snake"), f"{o}.down.w", f"{o}.down.b", stride=s)
        x = transformer(x[0].t(), "xf", cfg["transformer"]["layers"], cfg["transformer"]["window"])
        x = conv(snake(x.t()[None], "enc.snake"), "enc.out.w", "enc.out.b")
        for u in range(cfg["downsample"]["count"]):
            o = f"dn{u}"
            x = conv(x, f"{o}.conv.w", f"{o}.conv.b", stride=2)
            y = conv(x, f"{o}.cnx.dw.w", f"{o}.cnx.dw.b", groups=x.shape[1])[0].t()
            y = F.layer_norm(y, (y.shape[1],), W[f"{o}.cnx.ln.w"], W[f"{o}.cnx.ln.b"],
                             cfg["downsample"]["ln_eps"])
            y = F.gelu(y @ W[f"{o}.cnx.1.w"].t() + W[f"{o}.cnx.1.b"])
            x = x + (y @ W[f"{o}.cnx.2.w_s"].t() + W[f"{o}.cnx.2.b_s"]).t()[None]
        r = transformer(x[0].t(), "pre", cfg["pre"]["layers"], cfg["pre"]["window"])
        codes = []
        for k in range(len(cfg["codebook_sizes"])):
            ze = r @ W[f"vq{k}.win"].t() + W[f"vq{k}.bin"]
            idx = (ze @ W[f"vq{k}.C"].t()).argmax(-1)
            codes.append(idx)
            r = r - W[f"vq{k}.T"][idx]
    codes = torch.stack(codes, 1).numpy().astype(np.uint32)
    np.save(a.out, codes)
    print(f"frames {frames} codes {codes.shape}")
    if a.codes:
        g = np.load(a.codes)
        print("match per codebook:", " ".join(f"{(codes[:, k] == g[:, k]).mean():.3f}" for k in range(g.shape[1])))


if __name__ == "__main__":
    main()
