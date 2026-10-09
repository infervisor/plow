"""p5_sum.py <dir> [regex]: P5 batched-stage summary, one row per (ref, GEMV, batch, broadcast), means over repeats.

Files `<ref>.<amx|avx>.b<B>[.<bcast>].r<N>.json` (and `.<arm>` without repeats). Columns: step p50 / p99, us per token
(p50 / B), GEMV phases (critical-path compute of qkv, o, gate/up, down, ple gate, ple proj), attention (attention +
combine), barrier wait sum, the gate verdict (worst row and boundary when it fails) and, with counters, DRAM MiB and
L2 lines in per worker per step."""
import glob, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_gate  # noqa: E402  (gate() only; the module's CLI loop sees no files)

GEMV = (0, 3, 4, 5, 6, 7)
d = sys.argv[1]
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
cells = {}
for f in sorted(glob.glob(f"{d}/*.json")):
    if f.endswith(".ctr.json"):
        continue
    name = os.path.basename(f)[:-5]
    if pat and not pat.search(name):
        continue
    m = re.match(r"(.+?)\.(amx|avx)\.b(\d+)(?:\.(?!r\d+$)([a-z0-9]+))?(?:\.r\d+)?$", name)
    if not m:
        continue
    lines = open(f).read().strip().splitlines()
    if not lines:
        continue
    cells.setdefault((m[1], m[2], int(m[3]), m[4] or ""), []).append((f, json.loads(lines[-1])))

print("| stage | GEMV | B | variant | step p50 / p99 us | us / token | GEMV us | attention us | barrier us | gate | DRAM MiB / L2 lines per worker per step |")
print("|---|---|---:|---|---:|---:|---:|---:|---:|---|---|")
for (ref, g, b, var), runs in sorted(cells.items()):
    js = [j for _, j in runs]
    p50 = st.mean(j["step_us"]["p50"] for j in js)
    p99 = st.mean(j["step_us"]["p99"] for j in js)
    gemv = st.mean(sum(j["phase_us"][i]["compute_max"] for i in GEMV) for j in js)
    attn = st.mean(j["phase_us"][1]["compute_max"] + j["phase_us"][2]["compute_max"] for j in js)
    bar = st.mean(sum(p["barrier_mean"] for p in j["phase_us"]) for j in js)
    verdict = "PASS"
    for _, j in runs:
        ok, rows = p2_gate.gate(j["err"], j["ref"])
        bad = [x for x in rows if x.endswith("!")]
        for i, r in enumerate(j.get("rows", [])):
            rok, rr = p2_gate.gate(r["err"], r["dir"])
            ok &= rok
            bad += [f"row {i} {x}" for x in rr if x.endswith("!")]
        if not ok:
            verdict = "FAIL " + "; ".join(sorted(set(bad)))
            break
    ctr = ""
    cf = [f[:-5] + ".ctr.json" for f, _ in runs if os.path.exists(f[:-5] + ".ctr.json")]
    if cf:
        c = [json.load(open(x)) for x in cf]
        ctr = f"{st.mean(x['dram_mib_step'] for x in c):.1f} / {st.mean(x['l2_lines_in'] for x in c):,.0f}"
    print(f"| {ref} | {g} | {b} | {var or '-'} | {p50:.1f} / {p99:.1f} | {p50 / b:.1f} | {gemv:.1f} | {attn:.1f} | {bar:.1f} | {verdict} | {ctr} |")
