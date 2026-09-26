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
GEMM = {64: "dsv_moe_gemm_fp4", 32: "dsv_moe_gemm_fp4_m32", 16: "dsv_moe_gemm_fp4_m16"}


def fp4_weights(N, Kd):
    w = torch.randint(0, 256, (E, N, Kd // 2), dtype=torch.uint8, device=dev)
    s = torch.randint(118, 124, (E, N, Kd // 32), dtype=torch.uint8, device=dev)
    return w, s


def route(idx, BM):
    T = idx.shape[0]
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
    return offs, tiles, meta, rows, max_tiles, n


def run(bm, C, A, sa, w, s, tiles, meta, offs, rows, by_row, N, Kd, max_tiles):
    K.launch(GEMM[bm], ((N + 127) // 128, max_tiles), (128,),
             [C, A, sa, w, s, tiles, meta, offs, rows, i32(by_row), i32(N), i32(Kd), i64(N * Kd // 2), i64(N * Kd // 32)],
             smem=MOE_SMEM(Kd, bm))


import kernel as kr  # noqa: E402

w13, s13 = fp4_weights(2 * MI, H)
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
reps = 20


def timed(f):
    f()
    torch.cuda._sleep(50_000_000)  # hold the stream so the ctypes launches queue up back to back
    ev[0].record()
    for _ in range(reps):
        f()
    ev[1].record()
    torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) / reps


for T in Ts:
    idx = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).int()
    x = torch.randn(T, H, device=dev)
    xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    n_exp = len(set(idx.flatten().tolist()))
    wbytes = n_exp * 2 * MI * (H // 2 + H // 32)
    C64 = None
    for bm in (64, 32, 16):
        offs, tiles, meta, rows, max_tiles, n = route(idx, bm)
        C = torch.empty(n, 2 * MI, device=dev)
        args = (bm, C, xq.view(torch.uint8), xs.view(torch.uint8), w13, s13, tiles, meta, offs, rows, 0, 2 * MI, H, max_tiles)
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
        ms = timed(lambda: run(*args))
        print(f"T={T:5d} experts={n_exp:3d} rows={n:6d} mma{bm:<2d}: {ms*1e3:8.1f} us  weights {wbytes/1e6:7.1f} MB -> {wbytes/ms/1e9:6.3f} TB/s  "
              f"| rel err vs fp4_gemm {worst:.2e} {'PASS' if worst < 5e-3 else 'FAIL'}", flush=True)
        if bm == 64:
            C64, tiles64, meta64, offs64, rows64, mt64 = C, tiles, meta, offs, rows, max_tiles
    if T <= 64:  # the decode (swap-AB GEMV) form, on 8-row tiles
        offs8, tiles8, meta8, rows8, mt8, _ = route(idx, 8)
        C2 = torch.empty_like(C64)
        gargs = [C2, xq.view(torch.uint8), xs.view(torch.uint8), w13, s13, tiles8, meta8, offs8, rows8, i32(0), i32(2 * MI), i32(H),
                 i64(2 * MI * H // 2), i64(2 * MI * H // 32), None, i32(n)]
        grid = ((2 * MI + 63) // 64, mt8)
        gms = timed(lambda: K.launch("dsv_moe_gemv_fp4", grid, (128,), gargs))
        # against the reference per expert on the 8-row routing (moe_fill orders an expert's rows by
        # atomics, so a row-for-row comparison with the 64-row routing is not meaningful)
        o8, r8 = offs8.cpu(), rows8.cpu()
        g_err = 0.0
        for e in sorted(set(idx.flatten().tolist()))[:6]:
            p0, p1 = int(o8[e]), int(o8[e + 1])
            toks = r8[p0:p1].long().to(dev)
            ref = kr.fp4_gemm(xq[toks].contiguous(), xs[toks].contiguous(), w13[e].view(torch.float4_e2m1fn_x2),
                              s13[e].view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
            g_err = max(g_err, ((C2[p0:p1].float() - ref.float()).norm() / ref.float().norm()).item())
        print(f"T={T:5d} experts={n_exp:3d} rows={n:6d} gemv : {gms*1e3:8.1f} us  weights {wbytes/1e6:7.1f} MB -> {wbytes/gms/1e9:6.3f} TB/s  "
              f"| rel err vs fp4_gemm {g_err:.2e} {'PASS' if g_err < 5e-3 else 'FAIL'}", flush=True)
