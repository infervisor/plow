"""Decode GEMVs (swap-AB tensor-core, dsv41_gemm.cu dsv_gemv_w8a8_t*): correctness against kernel.py
fp8_gemm and weight bandwidth against the 64-row mma.sync W8A8 kernel, at V4.1 decode shapes.

  perf-data/tools/gpulease -n 1 dsv41-gemv <venv-python> scripts/dsv41_nv/bench_gemv.py <cubin> [M ...]
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cudrv import Cubin, i32, i64  # noqa: E402

REF = os.environ.get(
    "DSV41_REF",
    "/root/dsv41/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/inference",
)
sys.path.insert(0, REF)
import kernel as kr  # noqa: E402

torch.set_default_dtype(torch.bfloat16)
torch.manual_seed(0)
dev = "cuda"
K = Cubin(sys.argv[1])
Ms = [int(m) for m in sys.argv[2:]] or [1, 4, 16, 32]
W8_SMEM = 3 * (64 + 128) * 144
SHAPES = ((1280, 5120), (32768, 1280), (512, 5120), (5120, 8192), (4608, 5120), (5120, 2304))
ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]


def timed(f, reps=50):
    # GPU time of back-to-back launches: a sleep kernel holds the stream while the (slow, ctypes)
    # host enqueues, so the timed launches run without host gaps
    f()
    torch.cuda._sleep(50_000_000)
    ev[0].record()
    for _ in range(reps):
        f()
    ev[1].record()
    torch.cuda.synchronize()
    return ev[0].elapsed_time(ev[1]) / reps


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def ksplit_old(tiles, kb):  # stage.rs ksplit
    if tiles >= 132:
        return 1
    return max(1, min((264 + tiles - 1) // tiles, kb // 4))


for M in Ms:
    name = "dsv_gemv_w8a8_t8" if M <= 8 else "dsv_gemv_w8a8_t16" if M <= 16 else "dsv_gemv_w8a8_t32"
    for N, Kd in SHAPES:
        KB = Kd // 32
        x = torch.randn(M, Kd, device=dev, dtype=torch.bfloat16)
        w = (torch.randn(N, Kd, device=dev) * 0.05).to(torch.float8_e4m3fn)
        ws = torch.randint(118, 124, ((N + 31) // 32, KB), device=dev, dtype=torch.uint8)
        xq, xs = kr.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
        ref = kr.fp8_gemm(xq, xs, w, ws.view(torch.float8_e8m0fnu), torch.float8_e8m0fnu, 32).float()
        a8, s8, w8 = xq.view(torch.uint8), xs.view(torch.uint8), w.view(torch.uint8)
        c = torch.empty(M, N, device=dev, dtype=torch.float32)
        gx = (N + 63) // 64
        best = None
        for ks in (1, 2, 4, 8, 16, 32):
            if ks > KB // 4:
                break
            part = torch.empty(ks, M, N, device=dev, dtype=torch.float32) if ks > 1 else None

            def run(ks=ks, part=part):
                K.launch(name, (gx, ks), (128,), [c, a8, s8, w8, ws, i32(M), i32(N), i32(Kd), i64(N), i32(1), part])
                if ks > 1:
                    K.launch("dsv_splitk_reduce", (min((M * N + 255) // 256, 4096),), (256,),
                             [c, part, i32(ks), i32(1), i32(M), i32(N), i64(N), i64(0), i32(1)])
            ms = timed(run)
            r = rel(c, ref)
            if best is None or ms < best[0]:
                best = (ms, ks, r)
        # the mma.sync kernel with the engine's split-K choice
        tiles = ((N + 127) // 128) * ((M + 63) // 64)
        ko = ksplit_old(tiles, KB)
        c2 = torch.empty(M, N, device=dev, dtype=torch.float32)
        p2 = torch.empty(ko, M, N, device=dev, dtype=torch.float32) if ko > 1 else None

        def old():
            K.launch("dsv_gemm_w8a8", ((N + 127) // 128, (M + 63) // 64, ko), (128,),
                     [c2, a8, s8, w8, ws, i32(M), i32(N), i32(Kd), i64(N), i32(1), i32(ko), p2], smem=W8_SMEM)
            if ko > 1:
                K.launch("dsv_splitk_reduce", (min((M * N + 255) // 256, 4096),), (256,),
                         [c2, p2, i32(ko), i32(1), i32(M), i32(N), i64(N), i64(0), i32(1)])
        ms_old = timed(old)
        wb = N * Kd * (1 + 1 / 1024)
        ms, ks, r = best
        print(f"M={M:3d} N={N:5d} K={Kd:5d}: gemv {ms*1e3:7.1f} us {wb/ms/1e9:5.2f} TB/s (ks={ks:2d}) | mma.sync {ms_old*1e3:7.1f} us "
              f"{wb/ms_old/1e9:5.2f} TB/s (ks={ko:2d}) | x{ms_old/ms:4.2f} | rel vs fp8_gemm {r:.2e} {'PASS' if r < 2e-3 else 'FAIL'}", flush=True)
