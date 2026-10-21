#!/usr/bin/env python3
"""Fish S2 Pro codec decode from scripts/tts/fish_codec_export.py's tensors (torch, fp32/fp64).

The reference the plow lowering is gated against: codes [10][T] (i32 file, codebook-major) in,
PCM f32 out (2048 samples per frame).

  python scripts/tts/fish_codec_ref.py EXPORT_DIR CODES.i32 FRAMES OUT.f32 [--f64]
  python scripts/tts/fish_codec_ref.py EXPORT_DIR --vs-official CODEC_PTH [--frames N]
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def snake(x, a):
    a = a[None, :, None]
    return x + (a + 1e-9).reciprocal() * torch.sin(a * x).pow(2)


def causal_conv(x, w, b, dilation=1):
    pad = (w.shape[-1] - 1) * dilation
    return F.conv1d(F.pad(x, (pad, 0)), w, b, dilation=dilation, groups=x.shape[1] // w.shape[1])


def causal_convt(x, w, b, stride):
    y = F.conv_transpose1d(x, w, b, stride=stride)
    trim = w.shape[-1] - stride
    return y[..., : y.shape[-1] - trim] if trim else y


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos, theta):
    hd = x.shape[-1]
    inv = theta ** (-torch.arange(0, hd, 2, dtype=x.dtype) / hd)
    ang = pos[:, None].to(x.dtype) * inv[None, :]
    # The official table is stored in bf16 (modded_dac.py precompute_freqs_cis).
    cos, sin = (t.to(torch.bfloat16).to(x.dtype) for t in (ang.cos(), ang.sin()))
    a, b = x[..., : hd // 2], x[..., hd // 2 :]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], -1)


def decode(W, cfg, codes):
    """codes: int64 [10][T] -> pcm [T*2048]."""
    dt = W["vq.T0"].dtype
    sizes = cfg["codebook_sizes"]
    T = codes.shape[1]
    z = sum(W[f"vq.T{k}"][codes[k].clamp(0, sizes[k] - 1)] for k in range(len(sizes)))
    tc = cfg["transformer"]
    H, hd, win = tc["heads"], tc["head_dim"], tc["window"]
    pos = torch.arange(T)
    i, j = pos[:, None], pos[None, :]
    mask = (j <= i) & (j > i - win)
    x = z
    for l in range(tc["layers"]):
        p = f"post.layers.{l}"
        h = rms(x, W[f"{p}.attn_norm"], tc["eps"])
        q, k, v = (h @ W[f"{p}.wqkv"].t()).split(tc["dim"], -1)
        q, k, v = (t.reshape(T, H, hd).transpose(0, 1) for t in (q, k, v))
        q, k = rope(q, pos, tc["rope_theta"]), rope(k, pos, tc["rope_theta"])
        s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
        s = s.masked_fill(~mask, float("-inf")).softmax(-1)
        a = (s @ v).transpose(0, 1).reshape(T, tc["dim"])
        x = x + W[f"{p}.ls_attn"] * (a @ W[f"{p}.wo"].t())
        h = rms(x, W[f"{p}.ffn_norm"], tc["eps"])
        f = F.silu(h @ W[f"{p}.w1"].t()) * (h @ W[f"{p}.w3"].t())
        x = x + W[f"{p}.ls_ffn"] * (f @ W[f"{p}.w2"].t())
    x = rms(x, W["post.norm"], tc["eps"]).t()[None]
    for u, uc in enumerate(cfg["upsample"]):
        p = f"up{u}"
        x = causal_convt(x, W[f"{p}.convt.w"], W[f"{p}.convt.b"], uc["stride"])
        r = x
        y = causal_conv(x, W[f"{p}.cnx.dw.w"], W[f"{p}.cnx.dw.b"])[0].t()
        y = F.layer_norm(y, (y.shape[-1],), W[f"{p}.cnx.ln.w"], W[f"{p}.cnx.ln.b"], uc["ln_eps"])
        y = F.gelu(y @ W[f"{p}.cnx.1.w"].t() + W[f"{p}.cnx.1.b"])
        y = (y @ W[f"{p}.cnx.2.w"].t() + W[f"{p}.cnx.2.b"]) * W[f"{p}.cnx.gamma"]
        x = r + y.t()[None]
    x = causal_conv(x, W["dec.in.w"], W["dec.in.b"])
    for b, bc in enumerate(cfg["decoder"]["blocks"]):
        p = f"dec.blk{b}"
        x = causal_convt(snake(x, W[f"{p}.snake"]), W[f"{p}.up.w"], W[f"{p}.up.b"], bc["stride"])
        for r, d in enumerate(bc["dilations"]):
            q = f"{p}.ru{r}"
            y = causal_conv(snake(x, W[f"{q}.a1"]), W[f"{q}.c1.w"], W[f"{q}.c1.b"], d)
            x = x + causal_conv(snake(y, W[f"{q}.a2"]), W[f"{q}.c2.w"], W[f"{q}.c2.b"])
    x = causal_conv(snake(x, W["dec.out.snake"]), W["dec.out.w"], W["dec.out.b"])
    return torch.tanh(x).reshape(-1).to(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("codes", nargs="?")
    ap.add_argument("frames", nargs="?", type=int)
    ap.add_argument("out", nargs="?")
    ap.add_argument("--f64", action="store_true")
    ap.add_argument("--vs-official")
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    cfg = json.load(open(f"{a.export}/config.json"))
    dt = torch.float64 if a.f64 else torch.float32
    W = {k: v.to(dt) for k, v in load_file(f"{a.export}/model.safetensors").items()}
    torch.set_grad_enabled(False)
    if a.vs_official:
        from fish_speech.models.text2semantic.inference import decode_to_audio, load_codec_model

        g = torch.Generator().manual_seed(0)
        codes = torch.stack([torch.randint(0, n, (a.n,), generator=g) for n in cfg["codebook_sizes"]])
        codec = load_codec_model(a.vs_official, "cpu", torch.float32)
        ref = decode_to_audio(codes.clone(), codec).float().reshape(-1)
        ours = decode(W, cfg, codes).float()
        n = min(ref.numel(), ours.numel())
        d = (ref[:n] - ours[:n]).norm() / ref[:n].norm()
        print(f"official {ref.numel()} ours {ours.numel()} rel-L2 {d:.3e} max {(ref[:n]-ours[:n]).abs().max():.3e}")
        return
    codes = torch.from_numpy(np.fromfile(a.codes, np.int32).astype(np.int64)).reshape(-1, a.frames)
    decode(W, cfg, codes).float().numpy().tofile(a.out)


if __name__ == "__main__":
    main()
