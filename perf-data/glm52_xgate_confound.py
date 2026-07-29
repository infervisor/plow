#!/usr/bin/env python3
"""Isolate the data-dependent-MoE confound from the rendezvous protocol.

Inter-request interval MINUS the prefill drain = the DECODE half + host overhead. The
decode code object is BYTE-IDENTICAL in every arm (only the prefill MLA+MoE object was
swapped), and decode's own collectives are the untouched shipping path — so any arm-to-arm
difference in THIS column can only come from the garbage activations a numerically-wrong
prefill hands it. It is a pure read-out of "does garbage make GLM faster", measured on the
same runs, with no extra lease.
"""
import glob, os, re, statistics as st, datetime as dt
TS   = re.compile(r"(20\d\d-\d\d-\d\dT[\d:.]+Z)")
PLAN = re.compile(r"TP prefill plan")
DR   = re.compile(r"PF CHUNK .*drain=([0-9.]+) ms")
D = os.sys.argv[1] if len(os.sys.argv) > 1 else "rt_logs"
print(f"{'arm':<10} {'n':>3} {'req interval':>13} {'drain':>10} {'decode+host':>12}")
rows = {}
for path in sorted(glob.glob(f"{D}/*.*.log"), key=os.path.getmtime):
    arm = os.path.basename(path).split(".")[0]
    plans, drains = [], []
    for line in open(path, errors="replace"):
        clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = DR.search(clean)
        if m: drains.append(float(m.group(1)))
        if PLAN.search(clean):
            t = TS.search(clean)
            if t: plans.append(dt.datetime.fromisoformat(t.group(1).replace("Z", "+00:00")))
    if len(plans) < 4: continue
    iv = [(plans[i+1]-plans[i]).total_seconds()*1e3 for i in range(len(plans)-1)][2:]
    dr = drains[2:]
    if not iv: continue
    mi, md = st.median(iv), st.median(dr)
    rows.setdefault(arm, []).append(mi - md)
    print(f"{arm:<10} {len(iv):>3} {mi:>13.1f} {md:>10.1f} {mi-md:>12.1f}")
print()
base = st.median(rows.get("base", [0]))
for a, v in sorted(rows.items()):
    x = st.median(v)
    print(f"  {a:<10} decode+host {x:>8.1f} ms   vs base {x-base:+8.1f} ms "
          f"({100*(x-base)/base:+.1f}%)")
