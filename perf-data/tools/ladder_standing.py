#!/usr/bin/env python3
"""Rung-by-rung standing: plow vs the vLLM 0.28 reference, per model, per cell.

Lower is better for ttft/tpot/itl_p99; higher is better for out_tok_s. A cell is a WIN only if
plow is better on every one of those four.
"""
import csv
import pathlib
import sys

ROOT = pathlib.Path("/home/lava/plow/.claude/worktrees/gemma4-26b-beat-vllm/perf-data/campaign")

LOWER = ["ttft_ms", "tpot_ms", "itl_p99"]
HIGHER = ["out_tok_s"]


def load(p):
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                k = (int(r["input_len"]), int(r["concurrency"]))
            except (KeyError, ValueError):
                continue
            out[k] = r
    return out


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return None


def report(name, plow_file, ref_file):
    a, b = load(ROOT / plow_file), load(ROOT / ref_file)
    print(f"\n===== {name} =====")
    if a is None or b is None:
        print(f"  missing: {plow_file if a is None else ref_file}")
        return
    cells = sorted(set(a) & set(b))
    if not cells:
        print(f"  no overlapping cells (plow {len(a)}, ref {len(b)})")
        return
    print(f"  {'in':>6} {'C':>3} | {'TTFT plow/vllm':>20} | {'TPOT':>16} | "
          f"{'p99 ITL':>16} | {'tok/s':>16} | verdict")
    wins = losses = 0
    for k in cells:
        ra, rb = a[k], b[k]
        parts, ok = [], True
        for m in LOWER + HIGHER:
            x, y = num(ra, m), num(rb, m)
            if x is None or y is None:
                parts.append("      -/-      ")
                ok = False
                continue
            better = (x < y) if m in LOWER else (x > y)
            ok &= better
            parts.append(f"{x:8.1f}/{y:7.1f}{'*' if better else ' '}")
        verdict = "WIN " if ok else "lose"
        wins += ok
        losses += not ok
        print(f"  {k[0]:6d} {k[1]:3d} | {parts[0]:>20} | {parts[1]:>16} | "
              f"{parts[2]:>16} | {parts[3]:>16} | {verdict}")
    print(f"  -> {wins} cells win on ALL FOUR metrics, {losses} do not, of {len(cells)}")


report("12B ladder16k", "gemma4-12b.h100.bf16-ladder16k.csv",
       "gemma4-12b.h100.reference-vllm028-bf16.csv")
report("12B c32-16k", "gemma4-12b.h100.bf16-c32-16k.csv",
       "gemma4-12b.h100.reference-vllm028-bf16.csv")
report("26B ctx16k", "gemma4-26b-a4b.h100.bf16-ctx16k.csv",
       "gemma4-26b-a4b.h100.reference-vllm028-bf16.csv")
report("26B c32-16k", "gemma4-26b-a4b.h100.bf16-c32-16k.csv",
       "gemma4-26b-a4b.h100.reference-vllm028-bf16.csv")
