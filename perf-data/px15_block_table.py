#!/usr/bin/env python3
"""Render a px15 block-ranking jsonl as a per-cell table.

One table row per (batch, ctx) CELL, one column per arm — because the whole
point of px15 is that a knob's winner is a function of the cell, so a ranking
pooled across batches is not a smaller version of this table, it is a different
and wrong claim.
"""
import collections
import json
import sys


def main(paths):
    for path in paths:
        try:
            rows = [json.loads(l) for l in open(path) if l.strip()]
        except FileNotFoundError:
            print(f"-- {path}: absent")
            continue
        if not rows:
            print(f"-- {path}: empty")
            continue
        cells = collections.defaultdict(dict)
        for r in rows:
            cells[(r["batch"], r["ctx"])][r["arm"]] = r["latency_us_median"]
        arms = sorted({r["arm"] for r in rows})
        regs = {r["arm"]: r["registers"] for r in rows}
        layer = rows[0]["layer"]
        kvh = rows[0]["kv_heads"]
        print(f"\n== {path}  (layer {layer}, kv_heads {kvh}) ==")
        print("regs: " + "  ".join(f"{a}={regs[a]}" for a in arms))
        head = "".join(f"{a:>26}" for a in arms)
        print(f"{'B':>3} {'ctx':>8}{head}   winner   margin")
        for k in sorted(cells):
            v = cells[k]
            best = min(v, key=v.get)
            body = "".join(f"{v.get(a, float('nan')):>26.1f}" for a in arms)
            vals = sorted(v.values())
            marg = (vals[1] - vals[0]) / vals[1] * 100 if len(vals) > 1 else 0.0
            print(f"{k[0]:>3} {k[1]:>8}{body}   {best:<24} -{marg:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1:])
