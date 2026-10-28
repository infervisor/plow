"""p5_stab_sum.py <dir>: stability summary over consecutive stage runs `stab.NN.json` (+ `.ctr.json`) and `turbostat.txt`.

Per run: step p50 / p99, lock held fraction after the run, DRAM MiB and L2 lines in per worker per step, and the set
of failing (row, boundary) pairs of the fixed gate. Prints first / last / min / max / drift (last-quarter mean over
first-quarter mean) and the turbostat busy frequency and package / DRAM power over the whole window."""
import glob, json, os, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_gate  # noqa: E402

d = sys.argv[1]
runs = []
for f in sorted(glob.glob(f"{d}/stab.*.json")):
    if f.endswith(".ctr.json"):
        continue
    lines = open(f).read().strip().splitlines()
    if not lines:
        continue
    j = json.loads(lines[-1])
    fails = set()
    ok, rows = p2_gate.gate(j["err"], j["ref"])
    fails |= {f"row 0 {x.split(':')[0]}" for x in rows if x.endswith("!")}
    for i, r in enumerate(j.get("rows", [])):
        _, rr = p2_gate.gate(r["err"], r["dir"])
        fails |= {f"row {i} {x.split(':')[0]}" for x in rr if x.endswith("!")}
    c = json.load(open(f[:-5] + ".ctr.json"))
    runs.append(dict(p50=j["step_us"]["p50"], p99=j["step_us"]["p99"], held=j["lock"]["held_l2_after_min"],
                     held0=j["lock"]["held_l2_before_min"], dram=c["dram_mib_step"], l2=c["l2_lines_in"],
                     fails=tuple(sorted(fails)), cos=min(j["err"]["out"][1], j["err"]["out_last"][1])))
n = len(runs)
q = max(1, n // 4)


def line(k, fmt="{:.1f}"):
    v = [r[k] for r in runs]
    drift = st.mean(v[-q:]) / st.mean(v[:q])
    print(f"{k:5s} first {fmt.format(v[0])} last {fmt.format(v[-1])} min {fmt.format(min(v))} max {fmt.format(max(v))} "
          f"mean {fmt.format(st.mean(v))} drift {drift:.3f}")


print(f"{n} runs")
for k in ("p50", "p99", "dram", "l2"):
    line(k)
line("held", "{:.4f}")
print("held before min", f"{min(r['held0'] for r in runs):.4f}", "out cos min", f"{min(r['cos'] for r in runs):.8f}")
fs = {}
for r in runs:
    fs[r["fails"]] = fs.get(r["fails"], 0) + 1
for k, v in fs.items():
    print(f"gate failures {list(k) or 'none'}: {v} runs")
t = os.path.join(d, "turbostat.txt")
if os.path.exists(t):
    hdr, vals = None, []
    for l in open(t):
        p = l.split()
        if not p:
            continue
        if p[0] == "Time_Of_Day_Seconds":
            hdr = p
            continue
        if hdr and len(p) == len(hdr):
            try:
                vals.append({h: float(x) for h, x in zip(hdr, p)})
            except ValueError:
                pass
    # turbostat prints a package summary line then per-cpu lines when not --quiet summary-only; keep summary rows
    # (they carry PkgWatt)
    s = [v for v in vals if "PkgWatt" in v]
    if s:
        for k in ("Bzy_MHz", "Busy%", "PkgWatt", "RAMWatt"):
            if k in s[0]:
                x = [v[k] for v in s]
                print(f"turbostat {k}: mean {st.mean(x):.1f} min {min(x):.1f} max {max(x):.1f} ({len(x)} samples)")
