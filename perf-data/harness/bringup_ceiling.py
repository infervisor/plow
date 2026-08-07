#!/usr/bin/env python3
"""Practical GEMM ceiling on THIS box at YOUR model's exact prefill shapes
(BRINGUP.md §4). cuBLASLt via torch._scaled_mm (fp8) and torch.matmul (bf16).

Knowing the real ceiling is what separates "the kernel is slow" from "the box
is the wall": on GH200/Gemma-12B it measured fp8 1324-1468 / bf16 804-861 TF/s
and directly justified the 384-thread warp-specialized object.

Edit SHAPES for your model: (M=chunk_rows, N=out_features, K=in_features, name).
Run under gpulease: perf-data/harness/gpulease ceil python3 bringup_ceiling.py
"""
import time

import torch

SHAPES = [
    (4096, 4096, 3840, "q_proj"),
    (4096, 512, 3840, "kv_proj"),
    (4096, 3840, 4096, "o_proj"),
    (4096, 15360, 3840, "gate/up"),
    (4096, 3840, 15360, "down"),
]

def bench(f, it=20, warm=5):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(it):
        f()
    torch.cuda.synchronize()
    return (time.time() - t0) / it

def main():
    torch.cuda.init()
    print(f"{'shape':10s} {'fp8 ms':>8s} {'fp8 TF/s':>9s} {'bf16 ms':>8s} {'bf16 TF/s':>10s}")
    for m, n, k, name in SHAPES:
        fl = 2.0 * m * n * k
        a8 = torch.randn(m, k, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
        b8 = torch.randn(n, k, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
        sa = torch.ones(m, 1, device="cuda")
        sb = torch.ones(1, n, device="cuda")
        dt8 = bench(lambda: torch._scaled_mm(a8, b8.t(), scale_a=sa, scale_b=sb,
                                             out_dtype=torch.bfloat16))
        ab = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        bb = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        dtb = bench(lambda: ab @ bb)
        print(f"{name:10s} {dt8*1e3:8.3f} {fl/dt8/1e12:9.1f} {dtb*1e3:8.3f} {fl/dtb/1e12:10.1f}")

if __name__ == "__main__":
    main()
