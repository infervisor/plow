#!/usr/bin/env python3
"""Measure this GPU's practical ceilings for block/op rooflines and write them as JSON.

    scripts/bench/gpuq.py submit ceil 1 /opt/pytorch/bin/python3 scripts/bench/calibrate_ceilings.py \
        --out <scratch>/ceilings.json

Writes {bw_gbps, bf16_tflops, fp8_tflops, ...}: `op_roof.py --ceilings` and
`block_roofline.py --ceilings` take it instead of the registry's datasheet numbers.

  bw_gbps      streaming HBM read (a sum over a buffer far larger than L2): the denominator of
               weight/KV-stream floors. The copy rate (read + write) is recorded beside it.
  bf16_tflops  dense BF16 GEMM (torch.matmul, cuBLAS) at a square shape that saturates the part.
  fp8_tflops   dense e4m3 GEMM (torch._scaled_mm).

Medians of CUDA-event-timed repeats after warmup. Run it on an otherwise idle card (the queue
guarantees that); the SM clock and power limit at the time are recorded, since a clocked-down or
power-capped part measures a lower ceiling that is not the box's.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "perf-data", "tools"))
from bringup_ceiling import fp8_dtype  # noqa: E402


def timed(fn, reps, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) / 1e3)
    out.sort()
    return out[len(out) // 2], (out[-1] - out[0]) / out[len(out) // 2]


def smi(fields):
    try:
        text = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits",
                               "-i", str(torch.cuda.current_device())],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        return dict(zip(fields.split(","), (v.strip() for v in text.split(","))))
    except (OSError, subprocess.SubprocessError):
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bw-gib", type=float, default=8.0, help="streamed buffer size (>> L2)")
    ap.add_argument("--gemm", type=int, default=8192, help="square GEMM edge")
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()

    torch.cuda.init()
    clocks = smi("clocks.sm,clocks.max.sm,power.limit,driver_version")
    n = int(a.bw_gib * (1 << 30)) // 2
    buf = torch.ones(n, device="cuda", dtype=torch.bfloat16)
    dst = torch.empty_like(buf)
    read_s, read_spread = timed(lambda: buf.sum(dtype=torch.float32), a.reps)
    copy_s, copy_spread = timed(lambda: dst.copy_(buf), a.reps)
    del buf, dst

    m = a.gemm
    x = torch.randn(m, m, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(m, m, device="cuda", dtype=torch.bfloat16)
    bf16_s, bf16_spread = timed(lambda: x @ y, a.reps)
    f8 = fp8_dtype()
    x8, y8 = x.to(f8), y.to(f8).t().contiguous().t()
    one = torch.ones(1, device="cuda")
    fp8_s, fp8_spread = timed(lambda: torch._scaled_mm(x8, y8, scale_a=one, scale_b=one,
                                                       out_dtype=torch.bfloat16), a.reps)
    flops = 2.0 * m ** 3
    record = {
        "bw_gbps": round(2 * n / read_s / 1e9, 1),
        "copy_gbps": round(4 * n / copy_s / 1e9, 1),
        "bf16_tflops": round(flops / bf16_s / 1e12, 1),
        "fp8_tflops": round(flops / fp8_s / 1e12, 1),
        "spread": {"read": round(read_spread, 4), "copy": round(copy_spread, 4),
                   "bf16": round(bf16_spread, 4), "fp8": round(fp8_spread, 4)},
        "device": torch.cuda.get_device_name(),
        "fp8_encoding": str(f8).split(".")[-1],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_state": clocks,
        "method": f"read: bf16 sum over {a.bw_gib:g} GiB; GEMM: {m}^3 bf16 matmul / e4m3 _scaled_mm; "
                  f"median of {a.reps} CUDA-event repeats",
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
    }
    with open(a.out, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
