"""p2_sum.py <dir>: l2r_layer sweep -> markdown. One row per (ref, gemv, all-gather on/off): step p50 / p95 / p99
(median over reps; spread of p50 over reps), per-phase critical path (max-worker compute, mean over reps) and
mean barrier time, weight bytes per worker, gate status."""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict

d = sys.argv[1]
PH = ["qkv", "attn", "comb", "o", "gate/up", "down", "ple g", "ple p"]
cells = defaultdict(list)
for f in sorted(glob.glob(f"{d}/*.json")):
    m = re.match(r"(.+)\.(avx|amx)\.nb(\d)\.r\d\.json", os.path.basename(f))
    if m:
        cells[(m.group(1), m.group(2), int(m.group(3)))].append(json.loads(open(f).read().strip().splitlines()[-1]))
gate = {}
for l in open(f"{d}/gate.txt") if os.path.exists(f"{d}/gate.txt") else []:
    if l.startswith(("PASS", "FAIL")):
        p = l.split()
        gate.setdefault((p[1], p[2]), set()).add(p[0])
print("| layer | ctx rows | GEMV | all-gather | KiB/worker | step p50 / p95 / p99 µs (p50 spread) | "
      + " | ".join(PH) + " | barrier total | gate |")
print("|---|---:|---|---|---:|---|" + "---:|" * len(PH) + "---:|---|")
for (ref, gv, nb), v in sorted(cells.items()):
    p50 = [x["step_us"]["p50"] for x in v]
    s = f"{st.median(p50):.1f} / {st.median(x['step_us']['p95'] for x in v):.1f} / {st.median(x['step_us']['p99'] for x in v):.1f} ({(max(p50) - min(p50)) / st.mean(p50) * 100:.1f}%)"
    ph = [st.mean(x["phase_us"][i]["compute_max"] for x in v) for i in range(len(PH))]
    bar = st.mean(sum(p["barrier_mean"] for p in x["phase_us"]) for x in v)
    g = "/".join(sorted(gate.get((ref, gv), {"?"})))
    print(f"| {ref} | {v[0]['ctx_rows']} | {gv} | {'off' if nb else 'on'} | {v[0]['weight_bytes_per_worker'] / 1024:.0f} | {s} | "
          + " | ".join(f"{x:.2f}" for x in ph) + f" | {bar:.1f} | {g} |")
