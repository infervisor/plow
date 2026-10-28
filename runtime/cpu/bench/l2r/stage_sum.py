"""stage_sum.py <dir> [regex]: summary of l2r_layer stage runs `<stage>.<amx|avx>.b<B>[.<variant>].r<N>.json` (dense layers,
TP slices, MoE head and expert sockets): mean over reps of step p50 / p99, us per token, the critical-path compute of the
GEMV phases, attention (+ combine), the summed barrier wait, the fixed P2 gate on every row, and weight bytes per worker."""
import glob, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_gate  # noqa: E402

d = sys.argv[1]
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
name = re.compile(r"(.+?)\.(amx|avx)\.b(\d+)(?:\.(?!r\d+$)([a-z0-9]+))?(?:\.r\d+)?$")
groups = {}
for f in sorted(glob.glob(f"{d}/*.json")):
    if f.endswith(".ctr.json"):
        continue
    m = name.match(os.path.basename(f)[:-5])
    if not m or (pat and not pat.search(m.group(1))):
        continue
    lines = open(f).read().strip().splitlines()
    if not lines:
        continue
    j = json.loads(lines[-1])
    ok, rows = p2_gate.gate(j["err"], j["ref"])
    bad = [f"row 0 {x.split(':')[0]}" for x in rows if x.endswith("!")]
    for i, r in enumerate(j.get("rows", [])):
        rok, rr = p2_gate.gate(r["err"], r["dir"])
        ok &= rok
        bad += [f"row {i} {x.split(':')[0]}" for x in rr if x.endswith("!")]
    ph = j["phase_us"]
    if j.get("stage") == "experts":
        gemv, attn = sum(p["compute_max"] for p in ph), 0.0
    else:
        gemv = sum(ph[i]["compute_max"] for i in (0, 3, 4, 5, 6, 7) if i < len(ph))
        attn = ph[1]["compute_max"] + ph[2]["compute_max"]
    k = (m.group(1), m.group(2), int(m.group(3)), m.group(4) or "-")
    groups.setdefault(k, []).append(dict(p50=j["step_us"]["p50"], p99=j["step_us"]["p99"], gemv=gemv, attn=attn,
                                         bar=sum(p["barrier_mean"] for p in ph), ok=ok, bad=bad,
                                         wpw=j.get("weight_bytes_per_worker", 0), pairs=j.get("pairs")))
print("| stage | GEMV | B | variant | step p50 / p99 us | us / token | GEMV us | attention us | barrier us | weights / worker | gate |")
print("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|")
for k in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[3])):
    v = groups[k]
    mean = lambda f: st.mean(x[f] for x in v)
    gate = "PASS" if all(x["ok"] for x in v) else "FAIL " + "; ".join(sorted(set(sum((x["bad"] for x in v), []))))
    extra = f" ({v[0]['pairs']} pairs)" if v[0]["pairs"] is not None else ""
    print(f"| {k[0]}{extra} | {k[1]} | {k[2]} | {k[3]} | {mean('p50'):.1f} / {mean('p99'):.1f} | {mean('p50') / k[2]:.1f} | "
          f"{mean('gemv'):.1f} | {mean('attn'):.1f} | {mean('bar'):.1f} | {v[0]['wpw']:,.0f} | {gate} |")
