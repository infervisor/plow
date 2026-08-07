#!/usr/bin/env python3
"""What hipBLASLt/Tensile achieves on Gemma-4's decode GEMV shapes, HBM-bound.

The point of the ROTATION: a single 118 MB weight fits inside MI300X's 256 MB
Infinity Cache, so a naive repeat-loop measures cache bandwidth (plow's own
test_kernels GEMV bench reads 2.9 TB/s that way, against a printed "~8000 GB/s"
peak that is MI355X's number, not MI300X's 5.3 TB/s). The real decode streams
22.2 GiB of DISTINCT weights per token and never re-reads one. Cycling over
enough buffers to blow the cache reproduces that.
"""
import sys, time
import torch

assert torch.cuda.is_available(), "no ROCm device"
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False

# (label, N, K) — Gemma-4 12B decode projections, M=1.
SHAPES = [
    ("gate/up  ", 15360, 3840),
    ("down     ", 3840, 15360),
    ("qkv-ish  ", 4096, 3840),
    ("o_proj   ", 3840, 4096),
    ("lm_head  ", 262144, 3840),
    ("g31 gate ", 21504, 5376),   # the 31B shape, for the size comparison
    ("g31 down ", 5376, 21504),
]

CACHE_BYTES = 256 << 20  # MI300X Infinity Cache


def bench(N, K, iters=30):
    row = N * K * 2
    # Scale the iteration count so every shape runs >= 40 ms of GPU work. A single
    # 31 MB GEMV is ~20 us, the same order as an eager-mode launch, so a fixed
    # 30-iteration timing of the SMALL shapes measures dispatch, not bandwidth.
    iters = max(iters, int(40e-3 / max(row / 3.0e12, 1e-7)))
    # Enough distinct buffers to exceed the cache by 3x, capped so we fit VRAM.
    nbuf = max(2, min(24, int(3 * CACHE_BYTES / row) + 1))
    ws = [torch.randn(N, K, device=dev, dtype=torch.bfloat16) for _ in range(nbuf)]
    x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
    for w in ws[: min(4, nbuf)]:
        torch.nn.functional.linear(x, w)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        torch.nn.functional.linear(x, ws[i % nbuf])
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    del ws
    torch.cuda.empty_cache()
    return dt, row / dt / 1e9, nbuf, nbuf * row / (1 << 20)


print(f"torch {torch.__version__}  {torch.cuda.get_device_name(0)}")
print(f"{'shape':<10} {'N':>8} {'K':>7} {'bufs':>5} {'MiB':>8} {'ms':>8} {'GB/s':>9}")
for label, N, K in SHAPES:
    try:
        dt, gbs, nbuf, mib = bench(N, K)
        print(f"{label:<10} {N:>8} {K:>7} {nbuf:>5} {mib:>8.0f} {dt*1e3:>8.3f} {gbs:>9.0f}")
    except RuntimeError as e:
        print(f"{label:<10} {N:>8} {K:>7}   FAILED: {str(e)[:60]}")
