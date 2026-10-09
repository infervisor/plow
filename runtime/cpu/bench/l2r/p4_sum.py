"""p4_sum.py <dir> [tag-regex]: P4 KV matrix -> markdown. Runs named <cell>.r<N>.json are grouped per cell (median
over reps, p50 spread). Columns: step p50 / p99, attention phase (critical-path compute of phase 1), the GEMV phases
(0, 3-7, critical-path compute summed), KV MiB read per step, and the effective KV rate (KV bytes / phase-1 time)."""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict

d, pat = sys.argv[1], re.compile(sys.argv[2] if len(sys.argv) > 2 else ".")
cells = defaultdict(list)
for f in sorted(glob.glob(f"{d}/*.json")):
    tag = os.path.basename(f)[:-5]
    cell = re.sub(r"\.r\d+$", "", tag)
    if not pat.search(cell):
        continue
    try:
        cells[cell].append(json.loads(open(f).read().strip().splitlines()[-1]))
    except (ValueError, IndexError):
        print(f"<!-- {tag}: no result -->")
print("| cell | reps | step p50 / p99 µs (p50 spread) | attention µs | GEMV phases µs | KV MiB / step | KV GB/s |")
print("|---|---:|---|---:|---:|---:|---:|")
for cell, v in cells.items():
    p50 = [x["step_us"]["p50"] for x in v]
    att = st.median(x["phase_us"][1]["compute_max"] for x in v)
    gem = st.median(sum(x["phase_us"][i]["compute_max"] for i in (0, 3, 4, 5, 6, 7)) for x in v)
    kvb = v[0]["kv"]["bytes_per_step"]
    spread = (max(p50) - min(p50)) / st.mean(p50) * 100 if len(v) > 1 else 0
    print(f"| {cell} | {len(v)} | {st.median(p50):.1f} / {st.median(x['step_us']['p99'] for x in v):.1f} ({spread:.1f}%) "
          f"| {att:.1f} | {gem:.1f} | {kvb / 2**20:.1f} | {kvb / att / 1e3:.0f} |")
