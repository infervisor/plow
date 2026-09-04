#!/usr/bin/env python3
"""Publish grouped-MoE decode route measurements into the tuning store.

One qualified record per route for one exact geometry cell. The selector in
`crates/tunedb/src/moe_decode.rs` reroutes only when BOTH routes are present and
current, so this script refuses a single-route publication. Samples are the
per-layer GLU+DOWN pair body in microseconds, measured under an exclusive
`gpulease` from a device trace (`scripts/k3_trace_report.py`) of each route with
the same packet geometry and toolchain.

Example:
  scripts/tune_moe_decode_publish.py --root tuning --hardware amd/gfx950/mi350x \
    --n-cu 256 --rung 1 --topk 16 --hidden 3584 --inter-local 384 --experts 896 \
    --enc mxfp4 --digest gfx950-870078e93f2c92f0 --toolchain rocm-7.14.0-nix \
    --campaign k3-moe-decode-20260904 \
    --interpreter-us 34.9 35.1 34.8 35.0 34.9 --standalone-us 16.8 16.7 16.8 16.9 16.7
"""
import argparse
import json
import os
import statistics
import sys

ORACLE = "moe-decode-pair-bitexact-v1"
MIN_SAMPLES = 5


def stats(us):
    ns = sorted(u * 1000.0 for u in us)
    n = len(ns)
    if n < MIN_SAMPLES:
        raise SystemExit(f"need >= {MIN_SAMPLES} samples per route, got {n}")

    def pct(p):
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return ns[i]

    return {
        "median_ns": statistics.median(ns),
        "p10_ns": pct(0.10),
        "p90_ns": pct(0.90),
        "min_ns": ns[0],
        "samples": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="tuning")
    ap.add_argument("--hardware", required=True)
    ap.add_argument("--n-cu", type=int, required=True)
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--topk", type=int, required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--inter-local", type=int, required=True)
    ap.add_argument("--experts", type=int, required=True)
    ap.add_argument("--enc", required=True)
    ap.add_argument("--digest", required=True, help="implementation/interpreter digest label")
    ap.add_argument("--toolchain", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--interpreter-us", type=float, nargs="+", required=True)
    ap.add_argument("--standalone-us", type=float, nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cell = {
        "hardware": a.hardware,
        "n_cu": a.n_cu,
        "decode_rung": a.rung,
        "topk": a.topk,
        "hidden": a.hidden,
        "inter_local": a.inter_local,
        "experts": a.experts,
        "weight_enc": a.enc,
    }
    digests = {
        "implementation": a.digest,
        "interpreter": a.digest,
        "toolchain": a.toolchain,
        "oracle": ORACLE,
    }
    records = []
    for route, samples in (("interpreter", a.interpreter_us), ("standalone", a.standalone_us)):
        records.append(
            {
                "cell": cell,
                "route": route,
                "digests": digests,
                "stats": stats(samples),
                "correctness": "pass",
                "state": {"state": "qualified"},
                "campaign": a.campaign,
            }
        )
    path = os.path.join(a.root, a.hardware, "moe_decode_measurement.jsonl")
    lines = [json.dumps(r, separators=(",", ":")) for r in records]
    if a.dry_run:
        print("\n".join(lines))
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for line in lines:
            f.write(line + "\n")
    interp = records[0]["stats"]["median_ns"]
    stand = records[1]["stats"]["median_ns"]
    print(f"published 2 records to {path}: interpreter {interp/1000:.3f} us, standalone {stand/1000:.3f} us")


if __name__ == "__main__":
    sys.exit(main())
