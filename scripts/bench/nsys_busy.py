#!/usr/bin/env python3
"""nsys_busy.py <trace.sqlite> [top]: GPU busy (kernel + memcpy/memset + graph union) over the traced
window, idle time by gap size, and the top kernels by total time. The same numbers for plow and
the reference make the "device busy" / "GPU idle inside ticks" rows of an audit waterfall.

  nsys launch --session=S -t cuda --cuda-graph-trace=graph <server...> &   # then nsys start/stop
  nsys export --type sqlite <trace>.nsys-rep && nsys_busy.py <trace>.sqlite
"""
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
iv = []
k = db.execute("select start, end, shortName, demangledName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
for s, e, sn, dn in k: iv.append((s, e, names.get(sn, "?")))
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
    for s, e in db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY"): iv.append((s, e, "[memcpy]"))
if "CUPTI_ACTIVITY_KIND_MEMSET" in tables:
    for s, e in db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET"): iv.append((s, e, "[memset]"))
if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables:
    for s_, e_, g in db.execute("select start, end, graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"): iv.append((s_, e_, f"[graph {g}]"))
iv.sort()
t0, t1 = iv[0][0], max(x[1] for x in iv)
busy = 0; cur_s, cur_e = iv[0][0], iv[0][1]; gaps = []
for s, e, _ in iv[1:]:
    if s > cur_e:
        busy += cur_e - cur_s; gaps.append(s - cur_e); cur_s, cur_e = s, e
    else: cur_e = max(cur_e, e)
busy += cur_e - cur_s
wall = t1 - t0
print(f"window {wall/1e9:.3f} s, GPU busy {busy/1e9:.3f} s ({100*busy/wall:.1f}%), idle {100-100*busy/wall:.1f}%")
hist = collections.Counter()
for g in gaps:
    b = "<10us" if g < 1e4 else "<100us" if g < 1e5 else "<1ms" if g < 1e6 else "<10ms" if g < 1e7 else ">=10ms"
    hist[b] += g
print("idle by gap size (ms):", {b: round(v/1e6, 1) for b, v in sorted(hist.items())})
tot = collections.Counter(); cnt = collections.Counter()
for s, e, n in iv: tot[n] += e - s; cnt[n] += 1
ksum = sum(tot.values())
print(f"kernels+copies {len(iv)}, summed {ksum/1e9:.3f} s")
for n, v in tot.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 25):
    print(f"  {100*v/ksum:5.1f}% {v/1e6:8.1f} ms n={cnt[n]:6d} mean {v/cnt[n]/1e3:8.1f} us  {n[:90]}")
