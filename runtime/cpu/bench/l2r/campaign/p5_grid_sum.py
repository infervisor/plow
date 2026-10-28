"""p5_grid_sum.py: ctl (g0) vs PLOW_CPU_COMBINE=16 (g16) per cell, means of the repeats; ratio = g16 / g0."""
import json, glob, os, statistics as st

R = "/tmp/g4c/l2r/results/p5_grid"
KEYS = [("ttft p50", "median_ttft_ms"), ("ttft p99", "p99_ttft_ms"), ("tpot p50", "median_tpot_ms"),
        ("tpot p99", "p99_tpot_ms"), ("out tok/s", "output_throughput")]


def cell(d, c):
    vals = {}
    for f in sorted(glob.glob(f"{d}/g{c}.r*/bench.json")):
        j = json.load(open(f))
        for _, k in KEYS:
            vals.setdefault(k, []).append(j[k])
    return {k: st.mean(v) for k, v in vals.items()}, {k: (max(v) - min(v)) / st.mean(v) for k, v in vals.items()}


print("| model | ISL | c | " + " | ".join(f"{n} ctl → g16 (ratio)" for n, _ in KEYS) + " |")
print("|---|---:|---:|" + "---|" * len(KEYS))
for m in ("gemma-4-E2B-it", "gemma-4-E4B-it"):
    for isl in (1900, 15900):
        for c in (1, 4, 16):
            a, sa = cell(f"{R}/{m}.isl{isl}.g0", c)
            b, sb = cell(f"{R}/{m}.isl{isl}.g16", c)
            row = []
            for _, k in KEYS:
                fl = "*" if max(sa[k], sb[k]) > 0.10 else ""
                row.append(f"{a[k]:.1f} → {b[k]:.1f} ({b[k] / a[k]:.2f}){fl}")
            print(f"| {m.split('-')[2]} | {isl} | {c} | " + " | ".join(row) + " |")
print("\n`*` = repeat spread > 10% on either arm")
