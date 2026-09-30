"""Hopper wgmma prefill GEMMs (dsv41_wg.cu): correctness against the reference kernels and
throughput against the mma.sync kernels they replace, at V4.1 prefill shapes.

  perf-data/tools/gpulease -n 1 dsv41-wg <venv-python> scripts/dsv41_nv/bench_wg.py <cubin> [T ...]
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
Ts = [int(t) for t in sys.argv[2:]] or [1024, 4096, 16384]
WG_BN = int(os.environ.get("WG_BN", "256"))  # the cubin's tile width (dsv41_wg.cu WG_BN)
WG_ST = 3 if WG_BN == 256 else 5
WG_SMEM = WG_ST * (128 * 64 * 2 + WG_BN * 64 * 2 + WG_BN * 80) + 3 * WG_ST * 8 + 128 * 4 + 1024
W8_SMEM = 3 * (64 + 128) * 144
import kernel as kr  # noqa: E402

ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]


def timed(f, reps=10):
    f()
    ev[0].record()
    for _ in range(reps):
        f()
    ev[1].record()
    torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) / reps


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def fake_quant(x):
    xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    fq = (xq.float() * xs.float().repeat_interleave(32, 1)).to(torch.bfloat16)
    return xq, xs, fq


# ------------------------------------------------------------------------------ dense fp8 (W8A8)
for T in Ts:
    for N, Kd in ((5120, 8192), (1024, 5120), (32768, 1280)):
        x = torch.randn(T, Kd, device=dev)
        w = (torch.randn(N, Kd, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ws = torch.randint(118, 124, ((N + 31) // 32, Kd // 32), device=dev, dtype=torch.uint8)
        xq, xs, fq = fake_quant(x)
        c = torch.empty(T, N, device=dev, dtype=torch.float32)
        wg = lambda: K.launch("dsv_gemm_wg_fp8", ((N + WG_BN - 1) // WG_BN, (T + 127) // 128, 1), (384,),
                              [c, fq, w.view(torch.uint8), ws, i32(T), i32(N), i32(Kd), i64(Kd), i64(N), i32(1),
                               i64(0), i64(0), i64(0), i64(0)], smem=WG_SMEM)
        wg()
        torch.cuda.synchronize()
        ref = kr.fp8_gemm(xq, xs, w, ws.view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
        r = rel(c, ref)
        c2 = torch.empty(T, N, device=dev, dtype=torch.float32)
        old = lambda: K.launch("dsv_gemm_w8a8", ((N + 127) // 128, (T + 63) // 64, 1), (128,),
                               [c2, xq.view(torch.uint8), xs.view(torch.uint8), w.view(torch.uint8), ws, i32(T), i32(N), i32(Kd),
                                i64(N), i32(1), i32(1), None], smem=W8_SMEM)
        tf = 2 * T * N * Kd / 1e9
        ms_wg, ms_old = timed(wg), timed(old)
        print(f"fp8 T={T:5d} N={N:5d} K={Kd:5d}: wgmma {ms_wg*1e3:8.1f} us {tf/ms_wg:6.1f} TFLOP/s | mma.sync {ms_old*1e3:8.1f} us "
              f"{tf/ms_old:6.1f} TFLOP/s | x{ms_old/ms_wg:4.2f} | rel vs fp8_gemm {r:.2e} {'PASS' if r < 2e-3 else 'FAIL'}", flush=True)

# ------------------------------------------------------------------------------ grouped fp4 (MoE w13)
E, TOPK, H, MI = 384, 6, 5120, 2304
w13 = torch.randint(0, 256, (E, 2 * MI, H // 2), dtype=torch.uint8, device=dev)
s13 = torch.randint(118, 124, (E, 2 * MI, H // 32), dtype=torch.uint8, device=dev)


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


for T in Ts:
    idx = torch.stack([torch.randperm(E, device=dev)[:TOPK] for _ in range(T)]).int()
    x = torch.randn(T, H, device=dev)
    xq, xs, fq = fake_quant(x)
    N = 2 * MI
    offs, tiles, meta, rows, max_tiles, n = route(idx, 128)
    C = torch.empty(n, N, device=dev)
    # w13 form: A rows gathered through `rows` (token order in, expert order out)
    wg = lambda: K.launch("dsv_moe_gemm_wg_fp4", ((N + WG_BN - 1) // WG_BN, max_tiles), (384,),
                          [C, fq, w13, s13, tiles, meta, offs, rows, i32(0), i32(N), i32(H), i64(N * H // 2), i64(N * H // 32)],
                          smem=WG_SMEM)
    wg()
    # w2 form: A already in expert order (a_by_row) -- same result from the pre-gathered rows
    C_by_row = torch.empty(n, N, device=dev)
    a_sorted = fq[rows.long()].contiguous()
    wg_by_row = lambda: K.launch("dsv_moe_gemm_wg_fp4", ((N + WG_BN - 1) // WG_BN, max_tiles), (384,),
                                 [C_by_row, a_sorted, w13, s13, tiles, meta, offs, rows, i32(1), i32(N), i32(H),
                                  i64(N * H // 2), i64(N * H // 32)], smem=WG_SMEM)
    wg_by_row()
    torch.cuda.synchronize()
    assert torch.equal(C, C_by_row), "a_by_row and gathered forms differ"
    o, r = offs.cpu(), rows.cpu()
    worst = 0.0
    for e in sorted(set(idx.flatten().tolist()))[:6]:
        p0, p1 = int(o[e]), int(o[e + 1])
        toks = r[p0:p1].long().to(dev)
        ref = kr.fp4_gemm(xq[toks].contiguous(), xs[toks].contiguous(), w13[e].view(torch.float4_e2m1fn_x2),
                          s13[e].view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32)
        worst = max(worst, rel(C[p0:p1], ref))
    # the mma.sync kernel it replaces, at its best tile height for this T (64)
    o64, t64, m64, r64, mt64, _ = route(idx, 64)
    C2 = torch.empty(n, N, device=dev)
    old = lambda: K.launch("dsv_moe_gemm_fp4", ((N + 127) // 128, mt64), (128,),
                           [C2, xq.view(torch.uint8), xs.view(torch.uint8), w13, s13, t64, m64, o64, r64, i32(0), i32(N), i32(H),
                            i64(N * H // 2), i64(N * H // 32)], smem=MOE_SMEM(H, 64))
    tf = 2 * n * N * H / 1e9
    ms_wg, ms_old = timed(wg), timed(old)
    ms_br = timed(wg_by_row)
    print(f"fp4 MoE w13 T={T:5d}: wgmma on pre-gathered rows {ms_br*1e3:8.1f} us {2 * n * N * H / 1e9 / ms_br:6.1f} TFLOP/s", flush=True)
    print(f"fp4 MoE w13 T={T:5d} rows={n:6d}: wgmma {ms_wg*1e3:8.1f} us {tf/ms_wg:6.1f} TFLOP/s | mma.sync(64) {ms_old*1e3:8.1f} us "
          f"{tf/ms_old:6.1f} TFLOP/s | x{ms_old/ms_wg:4.2f} | rel vs fp4_gemm {worst:.2e} {'PASS' if worst < 1e-3 else 'FAIL'}", flush=True)
