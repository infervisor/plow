"""prof_bump.py <dump.csv>: time spent bumping successors (t2 - t1) per op family, and for every coarse edge
(single-consumer-start) the delay from the last producer's bump (t2) to the first consumer start."""
import csv, statistics as st, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
for r in rows:
    for k in ("inst", "slice", "worker", "t0_ns", "t1_ns", "t2_ns"):
        r[k] = int(r[k])
b = defaultdict(list)
for r in rows:
    b[r["op"]].append((r["t2_ns"] - r["t1_ns"]) / 1e3)
print("| op | packets | bump us median | p90 | max |")
for op, v in sorted(b.items(), key=lambda x: -st.median(x[1])):
    v.sort()
    print(f"| {op[9:]} | {len(v)} | {st.median(v):.2f} | {v[len(v) * 9 // 10]:.2f} | {v[-1]:.2f} |")
per = defaultdict(list)
for r in rows:
    per[r["inst"]].append(r)
order = sorted(per, key=lambda i: min(x["t0_ns"] for x in per[i]))
gaps_t1, gaps_t2 = [], []
for a, c in zip(order, order[1:]):
    last1 = max(x["t1_ns"] for x in per[a]); last2 = max(x["t2_ns"] for x in per[a])
    first = min(x["t0_ns"] for x in per[c])
    if first > last2:
        gaps_t1.append((first - last1) / 1e3); gaps_t2.append((first - last2) / 1e3)
print(f"consecutive-instruction edges: {len(gaps_t1)}; gap from last kernel end: sum {sum(gaps_t1):.0f} us, median {st.median(gaps_t1):.1f};"
      f" gap from last bump: sum {sum(gaps_t2):.0f} us, median {st.median(gaps_t2):.1f}")
