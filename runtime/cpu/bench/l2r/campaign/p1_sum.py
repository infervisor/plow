"""p1_sum.py <dir>: l2r_lock matrix -> markdown. One row per (ways, slice, kernel, scenario): plain vs locked
per-core GB/s (mean over reps of the summary per_core_mean; min core over reps), worst per-step p99 µs over
cores (median over reps), rep spread of per_core_mean, and L2-held fraction after the run (locked only)."""
import json, glob, os, re, sys, statistics as st
from collections import defaultdict

d = sys.argv[1]
cells = defaultdict(list)
for f in glob.glob(f"{d}/w*.s*.l*.*.*.r*.jsonl"):
    m = re.match(r"w(\d+)\.s(\d+)\.l(\d)\.([A-E])\.(amx|avx)\.r(\d)\.jsonl", os.path.basename(f))
    if not m:
        continue
    w, s, lk, sc, k, r = m.groups()
    summ = [json.loads(l) for l in open(f) if '"summary"' in l]
    if summ:
        cells[(int(w), int(s), k, sc, int(lk))].append(summ[0])


def agg(v):
    if not v:
        return None
    mean = [x["per_core_mean"] for x in v]
    return dict(n=len(v), gbs=st.mean(mean), min=min(x["min_gbs"] for x in v),
                p99=st.median(x["worst_p99_us"] for x in v),
                spread=(max(mean) - min(mean)) / st.mean(mean) * 100,
                held=min(x.get("held_l2_after", float("nan")) for x in v))


print("| L2 ways | slice KiB | kernel | scen | plain GB/s/core (min) | locked GB/s/core (min) | plain p99 µs | locked p99 µs | locked/plain | held L2 after | rep spread plain / locked |")
print("|---:|---:|---|---|---|---|---:|---:|---:|---:|---|")
keys = sorted({k[:4] for k in cells})
for w, s, k, sc in keys:
    a, b = agg(cells.get((w, s, k, sc, 0), [])), agg(cells.get((w, s, k, sc, 1), []))
    fmt = lambda x: f"{x['gbs']:.1f} ({x['min']:.1f})" if x else "-"
    ratio = f"{b['gbs'] / a['gbs']:.3f}" if a and b else "-"
    held = f"{b['held']:.4f}" if b else "-"
    sp = f"{a['spread']:.1f}% / {b['spread']:.1f}%" if a and b else "-"
    pa = "%.2f" % a["p99"] if a else "-"
    pb = "%.2f" % b["p99"] if b else "-"
    print(f"| {w} | {s // 1024} | {k} | {sc} | {fmt(a)} | {fmt(b)} | {pa} | {pb} | {ratio} | {held} | {sp} |")
