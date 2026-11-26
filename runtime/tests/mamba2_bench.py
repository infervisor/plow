#!/usr/bin/env python3
"""Per-kernel Mamba-2 op timing vs the calibrated roofline (GB10 ceilings), standalone cubin.

  python3 runtime/tests/mamba2_bench.py --cubin mamba2_ops.cubin --ckpt <dir> \
      --ceilings ceilings-gb10.json

Decode B in 1,2,4,8,16 (conv + step + gated norm), prefill T in --rows (conv + 3 chunk phases +
gated norm) for each prefill arm in --pf (simt = op_mamba.cuh, tc = op_mamba_pf.cuh). Each kernel is timed alone (CUDA events, median of 30 after 5 warmups). Bytes and
FLOPs come from scripts/bench/op_roof.py (the same model block_roofline prices the packet with);
floor = max(bytes / bw, flops / bf16 peak).
"""
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "bench"))
import mamba2_oracle as mo  # noqa: E402
import op_roof  # noqa: E402

H, P, N, G, W, CONV, DI, PROJ, GS = mo.H, mo.P, mo.N, mo.G, mo.W, mo.CONV, mo.DI, mo.PROJ, mo.GSIZE


def timed(ops, name, args, smem=0, reps=30):
    for _ in range(5):
        ops.launch(name, *args, smem=smem)
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        ops.launch(name, *args, smem=smem)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cubin", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ceilings", required=True)
    ap.add_argument("--json")
    ap.add_argument("--pf", default="simt,tc", help="prefill arms, comma list of simt,tc")
    ap.add_argument("--rows", default="128,2048,8192", help="prefill T list")
    a = ap.parse_args()
    c = json.load(open(a.ceilings))
    bw = c.get("bw_gbps") or c.get("hbm_gbps") or c["bandwidth_gbps"]
    tf = c.get("bf16_tflops") or c["bf16"]
    ops = mo.Ops(a.cubin)
    w = mo.layer_weights(mo.Ckpt(a.ckpt), 0)
    gen = torch.Generator(device="cuda").manual_seed(3)
    src, dsrc = PROJ | (DI << 16), PROJ | ((DI + CONV) << 16)
    rows = []

    def add(case, op, ms, p, nrows):
        by, fl = op_roof.cost(op, p, nrows, 0, 2, 2)
        floor = max(by / (bw * 1e6), fl / (tf * 1e9))
        rows.append(dict(case=case, op=op, ms=ms, floor_ms=floor, pct=100 * floor / ms,
                         gbps=by / ms / 1e6))

    for B in (1, 2, 4, 8, 16):
        proj = mo.make_proj(w, B, gen)
        hist = torch.zeros(B, CONV, W - 1, device="cuda", dtype=torch.bfloat16)
        state = torch.zeros(B, H, P, N, device="cuda", dtype=torch.float16)
        active = torch.ones(B + 1, device="cuda", dtype=torch.int32)  # slots + the step seed word
        conv = torch.empty(B, CONV, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(B, DI, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(B, DI, device="cuda", dtype=torch.bfloat16)
        t = timed(ops, "conv", (conv, proj, w["conv_w"], w["conv_b"], hist, active, CONV, W, B, 0, 0, src))
        add(f"decode B={B}", "Mamba2Conv", t, dict(rows=B, C=CONV, W=W, prefill=0), B)
        t = timed(ops, "step", (y, conv, proj, w["A_log"], w["D"], w["dt_bias"], state, active,
                                H, P, N, G, B, dsrc, 0.0, float("inf"), B, 0))
        add(f"decode B={B}", "Mamba2Step", t, dict(B=B, H=H, P=P, N=N, G=G, state_bytes=2), B)
        t = timed(ops, "norm", (out, y, proj, w["gamma"], active, G, GS, B, 1e-5, PROJ))
        add(f"decode B={B}", "Mamba2GatedNorm", t, dict(rows=B, groups=G, group=GS), B)

    for arm, T in [(arm, int(t)) for arm in a.pf.split(",") for t in a.rows.split(",")]:
        ops = mo.Ops(a.cubin, arm)
        proj = mo.make_proj(w, T, gen)
        hist = torch.zeros(CONV, W - 1, device="cuda", dtype=torch.bfloat16)
        state = torch.zeros(H, P, N, device="cuda", dtype=torch.float16)
        conv = torch.empty(T, CONV, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(T, DI, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(T, DI, device="cuda", dtype=torch.bfloat16)
        cstate = torch.empty(T // 128, H, P, N, device="cuda")
        dtv = torch.empty(H, T, device="cuda")
        cum = torch.empty(H, T, device="cuda")
        smem = ops.smem
        pc = dict(H=H, P=P, R=T, T=T, N=N, G=G, state_bytes=2)
        t = timed(ops, "conv", (conv, proj, w["conv_w"], w["conv_b"], hist, None, CONV, W, T, T, 1, src))
        add(f"prefill T={T} {arm}", "Mamba2Conv", t, dict(rows=T, C=CONV, W=W, prefill=1), T)
        t = timed(ops, "chunk_state", (cstate, conv, proj, w["A_log"], w["dt_bias"], dtv, cum,
                                       H, P, T, T, N, G, 0.0, float("inf"), dsrc), smem)
        add(f"prefill T={T} {arm}", "Mamba2ChunkState", t, pc, T)
        t = timed(ops, "chunk_pass", (cstate, cum, state, H, P, T, T, N))
        add(f"prefill T={T} {arm}", "Mamba2ChunkPass", t, pc, T)
        if arm == "tc":
            # Fused walk (PLOW_NV_MAMBA_SCAN_FUSE): priced as CHUNK_STATE (same reads, entering states
            # written in place of the local ones).
            t = timed(ops, "chunk_scan", (cstate, conv, proj, w["A_log"], w["dt_bias"], dtv, cum, state,
                                          H, P, T, T, N, G, 0.0, float("inf"), dsrc), smem)
            add(f"prefill T={T} {arm}", "Mamba2ChunkState", t, pc, T)
            rows[-1]["op"] = "Mamba2ChunkState+Pass (fused)"
            t = timed(ops, "chunk_scan_bf", (cstate, conv, proj, w["A_log"], w["dt_bias"], dtv, cum, state,
                                             H, P, T, T, N, G, 0.0, float("inf"), dsrc), smem)
            add(f"prefill T={T} {arm}", "Mamba2ChunkState", t, pc, T)
            rows[-1]["op"] = "Mamba2ChunkState+Pass (fused, bf16 entering)"
            t = timed(ops, "chunk_out_bf", (y, conv, cstate, cum, dtv, w["D"], H, P, T, T, N, G), smem)
            add(f"prefill T={T} {arm}", "Mamba2ChunkOut", t, pc, T)
            rows[-1]["op"] = "Mamba2ChunkOut (bf16 entering)"
        t = timed(ops, "chunk_out", (y, conv, cstate, cum, dtv, w["D"], H, P, T, T, N, G), smem)
        add(f"prefill T={T} {arm}", "Mamba2ChunkOut", t, pc, T)
        t = timed(ops, "norm", (out, y, proj, w["gamma"], None, G, GS, T, 1e-5, PROJ))
        add(f"prefill T={T} {arm}", "Mamba2GatedNorm", t, dict(rows=T, groups=G, group=GS), T)

    print(f"ceilings: {bw} GB/s, {tf} BF16 TFLOP/s ({c.get('source', a.ceilings)})")
    print("| case | op | time us | floor us | % of floor | GB/s |")
    print("|---|---|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['case']} | {r['op']} | {r['ms'] * 1e3:.1f} | {r['floor_ms'] * 1e3:.1f} | "
              f"{r['pct']:.0f}% | {r['gbps']:.0f} |")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
