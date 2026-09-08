#!/usr/bin/env python3
"""Measured HBM read/copy ceiling on ONE GPU, for the roofline's y-axis.

A roofline drawn against a spec sheet flatters the kernel: MI300X advertises 5325 GB/s and no
real kernel sees it (this tree's own MI325X entry records 4164 measured against 6000 spec).
The gap decides whether "10x off roofline" means the kernels are bad or the sheet is.

    perf-data/tools/gpulease -n 1 hbm python3 scripts/glm53_hbm_ceiling.py
"""
import sys, time
import torch

if not torch.cuda.is_available():
    sys.exit("no GPU visible")
dev = torch.device("cuda:0")
print(torch.cuda.get_device_name(0))

GIB = 1 << 30


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


for gib in (2, 8):
    n = gib * GIB // 2  # bf16 elements
    a = torch.empty(n, dtype=torch.bfloat16, device=dev)
    b = torch.empty(n, dtype=torch.bfloat16, device=dev)
    a.fill_(1.0)

    t = timeit(lambda: b.copy_(a))
    print(f"  copy   {gib:>2} GiB : {2*gib*GIB/t/1e9:8.1f} GB/s  (read+write)")

    t = timeit(lambda: torch.sum(a))
    print(f"  read   {gib:>2} GiB : {gib*GIB/t/1e9:8.1f} GB/s  (reduction, read only)")

    del a, b
    torch.cuda.empty_cache()

# The shape decode actually runs: a tall-skinny bf16 GEMV, weights streamed once.
for (n, k, name) in ((8192, 2048, "q_absorb TP4"), (6144, 4096, "o_proj TP4"),
                     (6144, 2048, "o_proj TP8")):
    w = torch.randn(n, k, dtype=torch.bfloat16, device=dev)
    x = torch.randn(k, dtype=torch.bfloat16, device=dev)
    t = timeit(lambda: torch.mv(w, x), warmup=5, iters=50)
    print(f"  GEMV {name:<14} [{n},{k}] : {w.numel()*2/t/1e9:8.1f} GB/s  ({t*1e6:7.1f} us)")
    del w, x
    torch.cuda.empty_cache()
