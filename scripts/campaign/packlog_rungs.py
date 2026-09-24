#!/usr/bin/env python3
"""Price riding decode and rung padding, from a PLOW_PF_PACKLOG=1 server log.

    packlog_rungs.py <server.log>

Joins each launch's three log lines -- `PACKLOG PACK` (prefill rows, decode_feeds),
`PACKLOG R=` (total rows, bucket) and the following `PACKLOG TICK` (prefill_ms) -- and groups by
(prefill rows -> bucket), so a launch that fits its rung can be compared against one that spills.

This is what shows that riding decode is cheap in-rung (0.456 ms/row at 4068+28 -> 4096) and
expensive across a rung hole (1.67 ms/row at 2048+30 -> 2078, which has no bucket below 4096).
Aggregate prefill_ms cannot see that difference; only the per-launch join can.
"""
import re,sys,collections,statistics as st
P=re.compile(r"PACKLOG PACK reqs=(\d+) rows=(\d+) decode_feeds=(\d+)")
R=re.compile(r"PACKLOG R=(\d+) rows=(\d+) bucket=(\d+)")
T=re.compile(r"PACKLOG TICK t_ms=[\d.]+ prefill_ms=([\d.]+) decode_ms=([\d.]+) did_prefill=(\d+)")
ANSI=re.compile(r"\x1b\[[0-9;]*m")
pk=None; rr=None; rows=[]
for line in open(sys.argv[1],errors="replace"):
    line=ANSI.sub("",line)
    m=P.search(line)
    if m: pk=(int(m[2]),int(m[3])); continue
    m=R.search(line)
    if m: rr=(int(m[2]),int(m[3])); continue
    m=T.search(line)
    if m and m[3]=="1" and pk and rr:
        rows.append((pk[0],pk[1],rr[0],rr[1],float(m[1]))); pk=rr=None
print(f"joined launches: {len(rows)}")
wd=[r for r in rows if r[1]>0]; nd=[r for r in rows if r[1]==0]
print(f"  with decode_feeds>0: n={len(wd)}  prefill_ms total={sum(r[4] for r in wd)/1000:.2f} s")
print(f"  with decode_feeds=0: n={len(nd)}  prefill_ms total={sum(r[4] for r in nd)/1000:.2f} s")
print()
print("by (pf_rows -> bucket), decode_feeds>0:")
by=collections.defaultdict(list)
for pfr,df,tr,b,ms in wd: by[(pfr,b)].append((ms,df))
for k,v in sorted(by.items(), key=lambda kv:-sum(x[0] for x in kv[1]))[:8]:
    ms=[x[0] for x in v]
    print(f"  pf_rows={k[0]:5d} bucket={k[1]:5d} n={len(ms):3d} med={st.median(ms):8.3f} ms total={sum(ms)/1000:5.2f} s feeds={st.median([x[1] for x in v]):.0f}")
print()
print("by (pf_rows -> bucket), decode_feeds=0:")
by=collections.defaultdict(list)
for pfr,df,tr,b,ms in nd: by[(pfr,b)].append(ms)
for k,v in sorted(by.items(), key=lambda kv:-sum(kv[1]))[:8]:
    print(f"  pf_rows={k[0]:5d} bucket={k[1]:5d} n={len(v):3d} med={st.median(v):8.3f} ms total={sum(v)/1000:5.2f} s")
