"""Grouped fp4 MoE GEMM: correctness against the reference fp4_gemm (per expert) and achieved
weight bandwidth at V4.1 shapes.

  perf-data/tools/gpulease -n 1 dsv41-moe <venv-python> scripts/dsv41_nv/bench_moe.py <cubin> [T ...]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cudrv import Cubin, i32, i64  # noqa: E402
from pyengine import MOE_SMEM  # noqa: E402

REF = os.environ.get(
    "DSV41_REF",
    "/root/dsv41/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/inference",
)
sys.path.insert(0, REF)
torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(0)
dev = "cuda"
K = Cubin(sys.argv[1])
Ts = [int(t) for t in sys.argv[2:]] or [1, 9, 64, 1024]
E, TOPK, H, MI = 384, 6, 5120, 2304
BM = 64


def fp4_weights(N, Kd):
    w = torch.randint(0, 256, (E, N, Kd // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(118, 124, (E, N, Kd // 32), dtype=torch.uint8, device=dev)
    return w, s


def route(T):
    idx = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).int()
    n = T * TOPK
    counts = torch.zeros(E, dtype=torch.int32, device=dev)
    K.launch("dsv_moe_count", ((n + 255) // 256,), (256,), [counts, idx, i32(n)])
    max_tiles = (n + BM - 1) // BM + min(n, E)
    offs = torch.empty(E + 1, dtype=torch.int32, device=dev)
    tiles = torch.empty(max_tiles * 2, dtype=torch.int32, device=dev)
    meta = torch.empty(1, dtype=torch.int32, device=dev)
    ctr = torch.empty(E, dtype=torch.int32, device=dev)
    K.launch("dsv_moe_offsets", (1,), (512,), [offs, tiles, meta, ctr, counts, i32(E), i32(BM)])
    rows = torch.empty(n, dtype=torch.int32, device=dev)
    rowpos = torch.empty(n, dtype=torch.int32, device=dev)
    roww = torch.empty(n, dtype=torch.float32, device=dev)
    wt = torch.ones(T, TOPK, dtype=torch.float32, device=dev)
    K.launch("dsv_moe_fill", ((n + 255) // 256,), (256,), [rows, rowpos, roww, ctr, offs, idx, wt, i32(n), i32(TOPK)])
    return idx, offs, tiles, meta, rows, max_tiles, n


def run(C, A, sa, w, s, tiles, meta, offs, rows, by_row, N, Kd, max_tiles):
    K.launch("dsv_moe_gemm_fp4", ((N + 127) // 128, max_tiles), (128,),
             [C, A, sa, w, s, tiles, meta, offs, rows, i32(by_row), i32(N), i32(Kd), i64(N * Kd // 2), i64(N * Kd // 32)],
             smem=MOE_SMEM(Kd))


import kernel as kr  # noqa: E402

w13, s13 = fp4_weights(2 * MI, H)
for T in Ts:
    idx, offs, tiles, meta, rows, max_tiles, n = route(T)
    x = torch.randn(T, H, device=dev)
    xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    C = torch.empty(n, 2 * MI, device=dev)
    args = (C, xq.view(torch.uint8), xs.view(torch.uint8), w13, s13, tiles, meta, offs, rows, 0, 2 * MI, H, max_tiles)
    run(*args)
    torch.cuda.synchronize()
    # correctness: a few experts against the reference kernel
    o = offs.cpu()
    r = rows.cpu()
    worst = 0.0
    for e in sorted(set(idx.flatten().tolist()))[:6]:
        p0, p1 = int(o[e]), int(o[e + 1])
        toks = r[p0:p1].long().to(dev)
        ref = kr.fp4_gemm(xq[toks].contiguous(), xs[toks].contiguous(), w13[e].view(torch.float4_e2m1fn_x2),
                          s13[e].view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
        got = C[p0:p1]
        worst = max(worst, ((got.float() - ref.float()).norm() / ref.float().norm()).item())
    # timing
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    reps = 20
    ev[0].record()
    for _ in range(reps):
        run(*args)
    ev[1].record()
    torch.cuda.synchronize()
    ms = ev[0].elapsed_time(ev[1]) / reps
    n_exp = len(set(idx.flatten().tolist()))
    wbytes = n_exp * 2 * MI * (H // 2 + H // 32)
    print(f"T={T:5d} experts={n_exp:3d} rows={n:6d} mma : {ms*1e3:8.1f} us  weights {wbytes/1e6:7.1f} MB -> {wbytes/ms/1e6:7.2f} TB/s  "
          f"| rel err vs fp4_gemm {worst:.2e} {'PASS' if worst < 5e-3 else 'FAIL'}", flush=True)
    if T <= 64:  # the decode (GEMV) form
        C2 = torch.empty_like(C)
        gargs = [C2, xq.view(torch.uint8), xs.view(torch.uint8), w13, s13, tiles, meta, offs, rows, i32(0), i32(2 * MI), i32(H),
                 i64(2 * MI * H // 2), i64(2 * MI * H // 32)]
        grid = ((2 * MI + 7) // 8, max_tiles)
        K.launch("dsv_moe_gemv_fp4", grid, (256,), gargs)
        torch.cuda.synchronize()
        g_err = ((C2.float() - C.float()).norm() / C.float().norm()).item()
        ev[0].record()
        for _ in range(reps):
            K.launch("dsv_moe_gemv_fp4", grid, (256,), gargs)
        ev[1].record()
        torch.cuda.synchronize()
        gms = ev[0].elapsed_time(ev[1]) / reps
        print(f"T={T:5d} experts={n_exp:3d} rows={n:6d} gemv: {gms*1e3:8.1f} us  weights {wbytes/1e6:7.1f} MB -> {wbytes/gms/1e6:7.2f} TB/s  "
              f"| rel diff vs mma {g_err:.2e} {'PASS' if g_err < 5e-3 else 'FAIL'}", flush=True)
