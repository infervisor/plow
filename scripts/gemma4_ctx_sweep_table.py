#!/usr/bin/env python3
"""Tabulate `bench_packed_serve.py` JSONL sweeps into one markdown table.

Each argument is `label=path[,path...]`; repeats within a cell are reduced by
median, matching how the other Gemma tables in docs/amd are built. Ratios are
oriented so that a value below 1.0 always means the first label is behind.
"""
import argparse
import collections
import json
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("arms", nargs="+", help="label=path[,path...]")
ap.add_argument("--ratio", nargs=2, metavar=("NUM", "DEN"),
                help="add ratio columns NUM/DEN (throughput) and DEN/NUM (latency)")
ap.add_argument("--json-out")
args = ap.parse_args()

cells = collections.defaultdict(dict)  # (input, concurrency) -> label -> metrics
labels = []
for arm in args.arms:
    label, _, paths = arm.partition("=")
    labels.append(label)
    runs = collections.defaultdict(list)
    for path in paths.split(","):
        for line in open(path):
            r = json.loads(line)
            runs[(r["input"], r["concurrency"])].append(r)
    for key, group in runs.items():
        cells[key][label] = {
            "repeats": len(group),
            "tok_s": statistics.median(r["output_tok_s"] for r in group),
            "ttft_ms": statistics.median(r["ttft_ms"] for r in group),
            "tpot_ms": statistics.median(r["tpot_ms"]["p50"] for r in group),
        }

header = "| Input / concurrency | " + " | ".join(
    f"{l} tok/s / TTFT ms / TPOT ms" for l in labels) + " |"
if args.ratio:
    header += " tok/s ratio | TTFT ratio | TPOT ratio |"
print(header)
print("|---|" + "---:|" * (len(labels) + (3 if args.ratio else 0)))
for key in sorted(cells):
    row = [f"| {key[0]} / {key[1]} "]
    for label in labels:
        m = cells[key].get(label)
        row.append(f"| {m['tok_s']:.2f} / {m['ttft_ms']:.1f} / {m['tpot_ms']:.2f} "
                   if m else "| — ")
    if args.ratio:
        num, den = (cells[key].get(x) for x in args.ratio)
        if num and den:
            row.append(f"| {num['tok_s']/den['tok_s']:.2f}x "
                       f"| {den['ttft_ms']/num['ttft_ms']:.2f}x "
                       f"| {den['tpot_ms']/num['tpot_ms']:.2f}x ")
        else:
            row.append("| — | — | — ")
    print("".join(row) + "|")

if args.json_out:
    out = {f"{i}/{c}": v for (i, c), v in sorted(cells.items())}
    json.dump(out, open(args.json_out, "w"), indent=2)
