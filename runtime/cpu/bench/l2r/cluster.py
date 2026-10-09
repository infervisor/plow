"""cluster.py <plan_dir> <results_dir>... : Groq-style multi-socket pipeline from plowc stage plans and measured single-socket
stage runs. Plans: `<plan_dir>/<model>.k<kib>.b<B>.json` (plowc stage-plan --split tp). Runs: l2r_layer stage JSON named
`<model>.L<layer>.c<ctx>[.tp<S>r0|.head|.ex<G>g<g>].<amx|avx>.b<B>.r<N>.json` (stage_sum.py naming).

A plan stage is measured when every piece maps to a measured run of the same model, layer class (sliding / full /
KV-shared), socket count and role (layer, MoE head, expert group): its on-socket time is the sum of those runs' step p50
(the faster GEMV of the two at that B; a stage with several layers runs them back to back). Other stages take the
planner prediction scaled by the model's measured / predicted ratio at that B (all models' ratio when the model has
none). Cross-socket all-reduces (planner count x --allreduce-us) and hops (--hop-us) are not measured and are added
from the planner cost model. Long context (16K): full-attention stages take their 16K run, or their 2K time scaled by
the model's measured long - 2K full-layer delta per full-attention layer; cold-KV runs (`.A`, P4 path A) are preferred at
long context, since ~50 microbatches in flight do not keep one sequence's KV in L3."""
import argparse, glob, json, os, re, statistics as st, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p2_gate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("plans")
ap.add_argument("results", nargs="+")
ap.add_argument("--hop-us", type=float, default=5.0)
ap.add_argument("--allreduce-us", type=float, default=10.0)
ap.add_argument("--hf", default="/tmp/models/google/gemma-4-{}-it")
ap.add_argument("--pick", default="", help="model=kib,... budget per model (default: every budget)")
args = ap.parse_args()

MODELS = {"e2b": "E2B", "e4b": "E4B", "12b": "12B", "26b": "26B-A4B", "31b": "31B"}
NAME = re.compile(r"(e2b|e4b|12b|26b|31b)\.L(\d+)\.c(\d+)(?:\.(tp(\d+)r0|head|ex(\d+)g\d+))?\.(amx|avx)\.b(\d+)\.(r\d+|A)$")


def layer_class(model):
    c = json.load(open(args.hf.format(model) + "/config.json"))["text_config"]
    lt, ns = c["layer_types"], c.get("num_kv_shared_layers") or 0
    return lambda l: ("shared-" if l >= len(lt) - ns else "") + lt[l].split("_")[0]


CLS = {m: layer_class(m) for m in MODELS.values()}

# runs[(model, class, sockets, role, ctx, B)] = (best p50 over GEMV of the rep-mean, gemv, gate ok)
acc = {}
for d in args.results:
    for f in glob.glob(f"{d}/*.json"):
        m = NAME.match(os.path.basename(f)[:-5])
        if not m:
            continue
        lines = open(f).read().strip().splitlines()
        if not lines:
            continue
        j = json.loads(lines[-1])
        ok, _ = p2_gate.gate(j["err"], j["ref"])
        for r in j.get("rows", []):
            ok &= p2_gate.gate(r["err"], r["dir"])[0]
        mk, layer, ctx, sl, tp, ex, g, b, rep = m.groups()
        model = MODELS[mk]
        role, socks = ("experts", int(ex)) if ex else ("head", 1) if sl == "head" else ("layer", int(tp) if tp else 1)
        k = (model, CLS[model](int(layer)), socks, role, int(ctx), int(b))
        acc.setdefault(k, {}).setdefault(rep == "A", {}).setdefault(g, []).append((j["step_us"]["p50"], ok))
runs = {}
for k, by_kv in acc.items():  # long context: a cold-KV run (P4 path A, KV from DRAM) wins over a warm one
    by = by_kv[True] if k[4] > 2048 and True in by_kv else by_kv.get(False) or by_kv[True]
    runs[k] = min(((st.mean(x for x, _ in v), g, all(o for _, o in v)) for g, v in by.items()), key=lambda t: t[0])

pick = dict(x.split("=") for x in args.pick.split(",") if x)


def stage_key(model, s, ctx, b):
    """the measured-run keys for each layer of the stage, or None when a piece has no measured counterpart"""
    kinds = {p["kind"] for p in s["pieces"]}
    layers = sorted({p["layer"] for p in s["pieces"] if p.get("layer") is not None})
    if not layers or kinds - {"attn", "ffn", "ple", "router", "experts"}:
        return None
    for l in layers:  # a row-split layer (pipe packing) has no single-stage measurement
        ps = [p for p in s["pieces"] if p.get("layer") == l]
        if any(p["kind"] == "ffn" and (p["row0"], p["row1"]) != (0, p["of_rows"]) for p in ps) or \
                ("experts" not in kinds and {"attn", "ffn"} - {p["kind"] for p in ps}):
            return None
    role = "experts" if kinds == {"experts"} else "head" if "router" in kinds else "layer"
    keys = [(model, CLS[model](l), s["sockets"], role, ctx, b) for l in layers]
    return keys if all(k in runs for k in keys) else None


def onsocket(s):
    return s["pred"]["total_us"] - s["pred"]["allreduce_us"]


plans = {}
for f in sorted(glob.glob(f"{args.plans}/*.json")):
    m = re.match(r"(.+)\.k(\d+)\.b(\d+)\.json", os.path.basename(f))
    if m:
        plans[(m.group(1), int(m.group(2)), int(m.group(3)))] = json.load(open(f))

# calibration: measured / predicted on-socket time over each model's measured stages, per B
ratios = {}
for (model, kib, b), p in plans.items():
    for s in p["stages"]:
        keys = stage_key(model, s, 2048, b)
        if keys:
            ratios.setdefault((model, b), []).append(sum(runs[k][0] for k in keys) / onsocket(s))
cal = {k: st.mean(v) for k, v in ratios.items()}
cal_all = {b: st.mean(sum((v for (m, bb), v in ratios.items() if bb == b), [])) for b in {b for _, b in ratios}}

# long-context extra time of one full-attention layer (us) per model, ctx and B from measured long / 2K pairs
long_delta = {}
for (model, cls, socks, role, ctx, b), v in runs.items():
    if ctx > 2048 and cls.endswith("full") and (model, cls, socks, role, 2048, b) in runs:
        long_delta.setdefault((model, ctx, b), []).append(v[0] - runs[(model, cls, socks, role, 2048, b)][0])
long_delta = {k: st.mean(v) for k, v in long_delta.items()}
CTXS = sorted({c for (_, c, _) in long_delta} | {2048})


def evaluate(model, p, b, ctx):
    times, measured, bad = [], 0, 0
    for s in p["stages"]:
        keys = stage_key(model, s, 2048, b)
        if keys:
            t, measured = 0.0, measured + 1
            for k in keys:
                kl = k[:4] + (ctx, b)
                if ctx != 2048 and k[1].endswith("full"):
                    t += runs[kl][0] if kl in runs else runs[k][0] + long_delta.get((model, ctx, b), float("nan"))
                else:
                    t += runs[k][0]
                bad += not runs[k][2]
        else:
            t = onsocket(s) * cal.get((model, b), cal_all[b])
            if ctx != 2048:
                nfull = sum(p_["kind"] == "attn" and CLS[model](p_["layer"]).endswith("full") for p_ in s["pieces"])
                t += nfull * long_delta.get((model, ctx, b), float("nan")) if nfull else 0.0
        times.append((t + s["allreduces"] * args.allreduce_us, s["label"]))
    n = len(times)
    bn, bl = max(times)
    lat = sum(t for t, _ in times) + args.hop_us * (n - 1)
    sat = n * (bn + args.hop_us)
    return dict(n=n, sockets=p["summary"]["sockets"], measured=measured, bad=bad, bn=bn, bl=bl, lat=lat, sat=sat,
                tps=b * 1e6 / (bn + args.hop_us), seq=b * n)


print(f"hop {args.hop_us:g} us, all-reduce {args.allreduce_us:g} us (planner cost model, not measured)\n")
print("| model | L2 budget | B | ctx | stages / sockets | measured stages | bottleneck stage | bottleneck us | token latency ms | "
      "TPOT at saturation ms | tok/s (sequences) | tok/s / socket |")
print("|---|---|---:|---:|---|---:|---|---:|---:|---:|---|---:|")
for (model, kib, b), p in sorted(plans.items(), key=lambda x: (list(MODELS.values()).index(x[0][0]), x[0][1], x[0][2])):
    if model in pick and pick[model] != str(kib):
        continue
    for ctx in CTXS:
        r = evaluate(model, p, b, ctx)
        if r["lat"] != r["lat"]:  # no measured long-context full layer for this model and B
            continue
        flag = f" ({r['bad']} gate-flagged)" if r["bad"] else ""
        print(f"| {model} | {kib} KiB | {b} | {ctx // 1024}K | {r['n']} / {r['sockets']} | {r['measured']} / {r['n']}{flag} | "
              f"{r['bl']} | {r['bn']:.1f} | {r['lat'] / 1e3:.2f} | {r['sat'] / 1e3:.2f} | {r['tps']:,.0f} ({r['seq']}) | "
              f"{r['tps'] / r['sockets']:.1f} |")
print("\ncalibration (measured / predicted on-socket):",
      ", ".join(f"{m} B{b} {v:.2f}" for (m, b), v in sorted(cal.items())))
print("long-context extra us per full-attention layer:",
      ", ".join(f"{m} {c // 1024}K B{b} {v:.0f}" for (m, c, b), v in sorted(long_delta.items())))
