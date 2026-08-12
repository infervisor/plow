#!/usr/bin/env python3
"""Per-tile fp8-e4m3 round-trip error analysis for Qwen3-4B bf16 weights.

Scheme (verified, runtime/amd/op_gemm.h:1439):
  per-output-channel (row of [N,K]) scale = amax_n / 448
  W8   = rne_e4m3(W / scale)          (torch.float8_e4m3fn)
  Wdq  = W8 * scale                   (multiplier applied once in epilogue)

Tiling: (BN,BK) = (128,64) over row-major [N,K].
"""
import json, os, sys, math
import torch
from safetensors.torch import safe_open

MODEL = "/root/models/Qwen3-4B"
BN, BK = 128, 64
PROJ = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def tensor_files():
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))
    return idx["weight_map"]

def quant_rt(W):
    """W: [N,K] float32. Returns (Wdq float32, scale [N,1])."""
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = amax / 448.0
    q = (W / scale).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn).float()
    return q * scale, scale

def tile_stats(W, Wdq):
    """W,Wdq: [N,K] float32, N%BN==0, K%BK==0. Returns dict of [nT] tensors."""
    N, K = W.shape
    nb, kb = N // BN, K // BK
    w = W.view(nb, BN, kb, BK).permute(0, 2, 1, 3).reshape(nb * kb, BN * BK)
    d = Wdq.view(nb, BN, kb, BK).permute(0, 2, 1, 3).reshape(nb * kb, BN * BK)
    err = d - w
    num = err.pow(2).sum(dim=1).sqrt()
    den = w.pow(2).sum(dim=1).sqrt().clamp_min(1e-30)
    aw = w.abs()
    amax = aw.amax(dim=1)
    amin = aw.clamp_min(1e-12).amin(dim=1)
    mean = w.mean(dim=1, keepdim=True)
    c = w - mean
    var = c.pow(2).mean(dim=1).clamp_min(1e-30)
    kurt = c.pow(4).mean(dim=1) / var.pow(2)
    std = var.sqrt()
    out4 = (aw > 4 * std.unsqueeze(1)).sum(dim=1).float()
    rows = torch.arange(nb).view(nb, 1).expand(nb, kb).reshape(-1)
    cols = torch.arange(kb).view(1, kb).expand(nb, kb).reshape(-1)
    return dict(relL2=num / den, amax=amax, dynrange=amax / amin, kurt=kurt,
                out4=out4, std=std, ti=rows, tj=cols, nb=nb, kb=kb)

def main():
    wmap = tensor_files()
    files = {}
    def get(name):
        f = wmap[name]
        if f not in files:
            files[f] = safe_open(os.path.join(MODEL, f), framework="pt")
        return files[f].get_tensor(name).float()

    recs = []
    names = []
    for L in range(36):
        for p in PROJ:
            sub = "self_attn" if p.endswith("o_proj") or p[0] in "qkv" else "mlp"
            n = f"model.layers.{L}.{sub}.{p}.weight"
            names.append((L, p, n))
    names.append((-1, "lm_head", "model.embed_tokens.weight"))  # tied

    cols = {}
    meta = []
    for L, p, n in names:
        W = get(n)
        N, K = W.shape
        if N % BN or K % BK:
            # embed_tokens 151936 x 2560: 151936 % 128 == 0? -> yes
            print(f"SKIP {n} {tuple(W.shape)} not tileable", file=sys.stderr)
            continue
        Wdq, _ = quant_rt(W)
        s = tile_stats(W, Wdq)
        nT = s["relL2"].numel()
        meta.append((L, p, N, K, s["nb"], s["kb"], nT))
        for k in ["relL2", "amax", "dynrange", "kurt", "out4", "std", "ti", "tj"]:
            cols.setdefault(k, []).append(s[k])
        cols.setdefault("layer", []).append(torch.full((nT,), L, dtype=torch.int32))
        cols.setdefault("proj", []).append(torch.full((nT,), PROJ.index(p) if p in PROJ else 7, dtype=torch.int32))
        # whole-tensor relL2
        e = (Wdq - W)
        tl2 = (e.pow(2).sum().sqrt() / W.pow(2).sum().sqrt()).item()
        print(f"{n:52s} {str(tuple(W.shape)):16s} tiles={nT:5d} tensor_relL2={tl2:.5f} "
              f"tile_relL2 min={s['relL2'].min():.5f} med={s['relL2'].median():.5f} max={s['relL2'].max():.5f}",
              flush=True)
        del W, Wdq, e

    out = {k: torch.cat(v) for k, v in cols.items()}
    torch.save({"cols": out, "meta": meta}, "/workspace/tilequant/tiles.pt")
    print("saved", out["relL2"].numel(), "tiles")

if __name__ == "__main__":
    main()
