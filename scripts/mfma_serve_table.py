#!/usr/bin/env python3
"""Fold the interleaved control/candidate bench_packed_serve.py runs into one table.

    python3 scripts/mfma_serve_table.py <serve-ab-dir>

Each cell is the median over every timed repetition of every round for that arm, so the
control/candidate ordering swap between rounds cancels rather than biasing one arm.
"""
import glob
import json
import os
import statistics
import sys


def load(root, arm):
    cells = {}
    for p in sorted(glob.glob(os.path.join(root, f"bench_{arm}_r*.json"))):
        for line in open(p):
            r = json.loads(line)
            k = (r["input"], r["concurrency"])
            c = cells.setdefault(k, {"tok": [], "ttft": [], "tpot": [], "text": set()})
            c["tok"].append(r["output_tok_s"])
            c["ttft"].append(r["ttft_ms"])
            c["tpot"].append(r["tpot_ms"]["p50"])
            for q in r["requests"]:
                c["text"].add((q["prompt_sha256"], q["text"]))
    return cells


def main():
    root = sys.argv[1]
    ctl, cand = load(root, "ctl"), load(root, "cand")
    print(f"{'in/conc':>10} | {'ctl tok/s':>9} {'cand tok/s':>10} {'d%':>7} | "
          f"{'ctl TTFT':>9} {'cand TTFT':>10} {'d%':>7} | "
          f"{'ctl TPOT':>9} {'cand TPOT':>10} {'d%':>7} | {'reps':>5}")
    for k in sorted(set(ctl) & set(cand)):
        a, b = ctl[k], cand[k]
        row = []
        for f in ("tok", "ttft", "tpot"):
            x, y = statistics.median(a[f]), statistics.median(b[f])
            row += [x, y, (y - x) / x * 100.0]
        print(f"{k[0]:>6}/{k[1]:<3} | {row[0]:>9.2f} {row[1]:>10.2f} {row[2]:>+6.1f}% | "
              f"{row[3]:>9.1f} {row[4]:>10.1f} {row[5]:>+6.1f}% | "
              f"{row[6]:>9.2f} {row[7]:>10.2f} {row[8]:>+6.1f}% | {len(a['tok']):>5}")

    # completion identity: at concurrency 1 the served tier is lowrung1 (MM=1, VALU), so those
    # texts MUST be character-identical; at 4 they may differ and the count is the evidence.
    print()
    for k in sorted(set(ctl) & set(cand)):
        ta = {p: t for p, t in ctl[k]["text"]}
        tb = {p: t for p, t in cand[k]["text"]}
        shared = set(ta) & set(tb)
        same = sum(ta[p] == tb[p] for p in shared)
        first = []
        for p in sorted(shared):
            if ta[p] != tb[p]:
                i = next(i for i, (x, y) in enumerate(zip(ta[p], tb[p])) if x != y)
                first.append(i)
        print(f"  in={k[0]:>5} conc={k[1]}: {same}/{len(shared)} completions character-identical"
              + (f"; first char divergence at {sorted(first)}" if first else ""))


main()
