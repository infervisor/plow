"""p3_sum.py <dir>: sync/broadcast mode sweep -> markdown. Per (stage, mode): step p50 / p99 (median over reps,
p50 spread), the sum of per-phase critical-path compute, the summed barrier wait, and the sync+broadcast cost =
step p50 minus the no-all-gather compute critical path of the same stage."""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict

d = sys.argv[1]
cells = defaultdict(list)
for f in glob.glob(f"{d}/*.r[0-9].json"):
    m = re.match(r"(.+)\.(\w+)\.(diss|hier)\.r\d\.json", os.path.basename(f))
    if m:
        cells[(m.group(1), f"{m.group(2)}.{m.group(3)}")].append(json.loads(open(f).read().strip().splitlines()[-1]))
gate = {}
for l in open(f"{d}/gate.txt") if os.path.exists(f"{d}/gate.txt") else []:
    if l.startswith(("PASS", "FAIL")):
        gate.setdefault(l.split()[1], set()).add(l.split()[0])
floor = {ref: st.mean(sum(p["compute_max"] for p in x["phase_us"]) for x in v) for (ref, mode), v in cells.items() if mode == "nobcast.diss"}
order = ["direct.diss", "rep.diss", "repnt.diss", "repnt.hier", "fid.diss", "nobcast.diss"]
print("| stage | broadcast.barrier | step p50 / p99 µs (p50 spread) | phase compute sum | barrier wait sum | sync + broadcast µs |")
print("|---|---|---|---:|---:|---:|")
for ref in sorted({k[0] for k in cells}):
    for mode in order:
        v = cells.get((ref, mode))
        if not v:
            continue
        p50 = [x["step_us"]["p50"] for x in v]
        comp = st.mean(sum(p["compute_max"] for p in x["phase_us"]) for x in v)
        bar = st.mean(sum(p["barrier_mean"] for p in x["phase_us"]) for x in v)
        sync = st.median(p50) - floor[ref] if ref in floor else float("nan")
        print(f"| {ref} | {mode} | {st.median(p50):.1f} / {st.median(x['step_us']['p99'] for x in v):.1f} "
              f"({(max(p50) - min(p50)) / st.mean(p50) * 100:.1f}%) | {comp:.1f} | {bar:.1f} | {sync:.1f} |")
print(f"\ngate: {', '.join(f'{k} {sorted(v)}' for k, v in sorted(gate.items()))}")
