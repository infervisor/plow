#!/usr/bin/env python3
"""Tabulate `vllm bench serve` result JSONs into one markdown table per metric.

Reads build-glm53/bench/<label>-tp<N>/in<L>_c<C>.json (whatever --save-result wrote) and
prints plow-vs-vLLM cells side by side. Ratios are vLLM/plow for throughput (>1 = vLLM ahead)
and plow/vLLM for the latency metrics (>1 = vLLM ahead), so ">1 always means plow loses".
"""
import json, os, re, sys, collections

root = sys.argv[1] if len(sys.argv) > 1 else "build-glm53/bench"
rows = collections.defaultdict(dict)          # (tp, inlen, conc) -> {label: metrics}
for d in sorted(os.listdir(root)):
    m = re.match(r"(.+)-tp(\d+)$", d)
    if not m:
        continue
    label, tp = m.group(1), int(m.group(2))
    for fn in sorted(os.listdir(os.path.join(root, d))):
        if not fn.endswith(".json"):
            continue
        g = re.match(r"in(\d+)_c(\d+)\.json$", fn)
        if not g:
            continue
        j = json.load(open(os.path.join(root, d, fn)))
        rows[(tp, int(g.group(1)), int(g.group(2)))][label] = j

METRICS = [
    ("output_throughput",   "output tok/s",   "higher"),
    ("median_ttft_ms",      "median TTFT ms", "lower"),
    ("median_tpot_ms",      "median TPOT ms", "lower"),
    ("median_itl_ms",       "median ITL ms",  "lower"),
    ("median_e2el_ms",      "median E2EL ms", "lower"),
]
labels = sorted({l for v in rows.values() for l in v})
for key, name, better in METRICS:
    print(f"\n### {name} ({'higher' if better=='higher' else 'lower'} is better)\n")
    print("| TP | input | conc | " + " | ".join(labels) + " | vLLM ahead by |")
    print("|---:|---:|---:|" + "|".join(["---:"] * (len(labels) + 1)) + "|")
    for (tp, inlen, conc) in sorted(rows):
        cells, vals = [], {}
        for l in labels:
            j = rows[(tp, inlen, conc)].get(l)
            if j is None or key not in j:
                cells.append("—"); continue
            vals[l] = j[key]; cells.append(f"{j[key]:.2f}")
        # Best PLOW arm of whatever arms are present, against the vLLM reference. Oriented so
        # >1 always means plow loses, whichever direction the metric runs.
        ratio = "—"
        parms = {k: v for k, v in vals.items() if k != "vllm"}
        if parms and "vllm" in vals:
            best = max(parms.values()) if better == "higher" else min(parms.values())
            if best:
                r = vals["vllm"] / best if better == "higher" else best / vals["vllm"]
                ratio = f"{r:.2f}x"
        print(f"| {tp} | {inlen} | {conc} | " + " | ".join(cells) + f" | {ratio} |")
