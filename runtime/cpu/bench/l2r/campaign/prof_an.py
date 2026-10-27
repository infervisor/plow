"""prof_an.py <dump.csv> [workers cpus]: decode-step trace analysis.
Per worker: busy, by SNC node. Per GEMV-family instruction: slice time min/median/max (imbalance) and effective
bandwidth (weight bytes / max slice time). Critical path: sum over instructions of the max slice time vs the span."""
import csv, statistics as st, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    for k in ("inst", "slice", "worker", "t0_ns", "t1_ns", "i0", "i1", "i2"):
        r[k] = int(r[k])
cpus = [c for a in "2-31,34-63,66-95".split(",") for c in range(int(a.split("-")[0]), int(a.split("-")[1]) + 1)]
t0 = min(r["t0_ns"] for r in rows); t1 = max(r["t1_ns"] for r in rows)
span = (t1 - t0) / 1e3
busy = defaultdict(int)
for r in rows:
    busy[r["worker"]] += r["t1_ns"] - r["t0_ns"]
print(f"span {span:.0f} us, packets {len(rows)}")
# worker -> cpu: physical_worker_cpus is node-interleaved; report by worker index mod 3 as the node proxy
bynode = defaultdict(list)
for w, b in busy.items():
    bynode[w % 3].append(b / 1e3)
for n, v in sorted(bynode.items()):
    print(f"workers w%3={n}: n {len(v)} busy us min {min(v):.0f} median {st.median(v):.0f} max {max(v):.0f}")
ws = sorted(busy.items(), key=lambda x: x[1])
print("lowest busy workers:", [(w, round(b / 1e3)) for w, b in ws[:6]], "highest:", [(w, round(b / 1e3)) for w, b in ws[-6:]])
# per instruction
per = defaultdict(list)
for r in rows:
    per[r["inst"]].append(r)
fam = defaultdict(lambda: [0, 0.0, 0.0, 0.0, []])  # count, sum max, sum median, sum min, ratios
crit = 0.0
for i, v in per.items():
    d = [(x["t1_ns"] - x["t0_ns"]) / 1e3 for x in v]
    op = v[0]["op"]
    f = fam[op]
    f[0] += 1; f[1] += max(d); f[2] += st.median(d); f[3] += min(d)
    if len(d) > 4:
        f[4].append(max(d) / max(st.median(d), 1e-3))
    crit += max(d)
print(f"sum over instructions of the slowest slice: {crit:.0f} us (span {span:.0f} us)")
print("| op | insts | sum max-slice us | sum median-slice us | sum min-slice us | median max/median |")
for op, f in sorted(fam.items(), key=lambda x: -x[1][1]):
    print(f"| {op} | {f[0]} | {f[1]:.0f} | {f[2]:.0f} | {f[3]:.0f} | {st.median(f[4]) if f[4] else 0:.2f} |")
# gaps: per worker, time between consecutive packets (waiting), summed
gap = defaultdict(float)
byw = defaultdict(list)
for r in rows:
    byw[r["worker"]].append((r["t0_ns"], r["t1_ns"]))
for w, v in byw.items():
    v.sort()
    gap[w] = sum(max(0, b[0] - a[1]) for a, b in zip(v, v[1:])) / 1e3
print(f"per-worker wait between packets: median {st.median(gap.values()):.0f} us, min {min(gap.values()):.0f}, max {max(gap.values()):.0f}")
