#!/usr/bin/env python3
"""Measured HBM bandwidth ($BW_BOUND): device-to-device copy and a read-only reduction over
buffers far larger than L2, median of 20 after warmup. Prints GB/s (1e9)."""
import statistics, torch

n = 4 << 30  # 4 GiB
a = torch.empty(n // 2, dtype=torch.bfloat16, device="cuda")
b = torch.empty_like(a)
a.normal_()
def timed(f, bytes_moved):
    for _ in range(3):
        f()
    ts = []
    for _ in range(20):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); f(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return bytes_moved / statistics.median(ts) / 1e9
copy = timed(lambda: b.copy_(a), 2 * n)
read = timed(lambda: a.sum(dtype=torch.float32), n)
print(f"copy {copy:.0f} GB/s (read+write)  read-reduce {read:.0f} GB/s  device {torch.cuda.get_device_name()}")
