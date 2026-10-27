"""prof_tl.py <dump.csv> <first_inst> <n>: per-instruction timeline (first start / median end / last end, us from
step start, slices, busy) for n instructions in issue order starting at the given index of the sorted list."""
import csv, statistics as st, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1])))
per = defaultdict(list)
for r in rows:
    per[int(r["inst"])].append(r)
t0 = min(int(r["t0_ns"]) for r in rows)
order = sorted(per, key=lambda i: min(int(x["t0_ns"]) for x in per[i]))
a, n = int(sys.argv[2]), int(sys.argv[3])
prev_last = None
for i in order[a:a + n]:
    v = per[i]
    s = [(int(x["t0_ns"]) - t0) / 1e3 for x in v]
    e = [(int(x["t1_ns"]) - t0) / 1e3 for x in v]
    d = [b - c for b, c in zip(e, s)]
    print(f"#{i:<5} {v[0]['op'][9:]:<22} sl {len(v):>3}  start {min(s):8.1f}  med-end {st.median(e):8.1f}  last-end {max(e):8.1f}"
          f"  slice med {st.median(d):6.2f} max {max(d):6.2f}  i {v[0]['i0']}x{v[0]['i1']}x{v[0]['i2']}")
