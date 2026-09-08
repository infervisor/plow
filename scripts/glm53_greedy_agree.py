#!/usr/bin/env python3
"""Greedy-agreement between two `facts_gate.py run` JSONs, per context length.

`facts_gate.py verdict` answers "is the candidate WORSE" (paired McNemar over
machine-checkable answers). That is the right question for an arm that changes
arithmetic on purpose — fp8 KV, sparse attention — but it deliberately says
nothing about how far the two token streams stayed together, and character
identity says nothing about quality. This reports the other half: the fraction
of cells whose completion is character-identical, and where the rest separated.

Divergence position is reported as a CHARACTER index and as a fraction of the
baseline answer, because that is what `facts_gate` already records for the
answer's own position — the two are directly comparable, so a divergence that
lands AFTER the checkable token is visibly harmless and one that lands before it
is visibly not.
"""
import argparse, json, statistics, sys


def cells(path):
    j = json.load(open(path))
    return {(c["tokens"], c["item"]): c for c in j["cells"]}, j


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return -1 if len(a) == len(b) else n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--json", help="write the per-length table here")
    a = ap.parse_args()

    bmap, bj = cells(a.baseline)
    cmap, cj = cells(a.candidate)
    keys = sorted(bmap.keys() & cmap.keys())
    if not keys:
        print("no shared cells", file=sys.stderr)
        return 2

    by_len = {}
    for k in keys:
        bt, ct = bmap[k]["text"], cmap[k]["text"]
        d = first_div(bt, ct)
        by_len.setdefault(k[0], []).append(
            (d, len(bt), bmap[k]["answer_char"], bmap[k]["correct"], cmap[k]["correct"])
        )

    rows = []
    print(f"{'ctx':>7} {'cells':>6} {'identical':>10} {'first-div char':>16} "
          f"{'div frac':>9} {'base ok':>8} {'cand ok':>8}")
    for ln in sorted(by_len):
        v = by_len[ln]
        ident = sum(1 for d, *_ in v if d < 0)
        divs = [d for d, *_ in v if d >= 0]
        fracs = [d / n for d, n, *_ in v if d >= 0 and n]
        row = {
            "len": ln, "cells": len(v), "identical": ident,
            "identical_frac": ident / len(v),
            "median_first_div_char": statistics.median(divs) if divs else None,
            "median_div_frac": statistics.median(fracs) if fracs else None,
            "baseline_ok": sum(1 for *_, bo, co in v if bo),
            "candidate_ok": sum(1 for *_, bo, co in v if co),
        }
        rows.append(row)
        md = f"{row['median_first_div_char']:.0f}" if divs else "-"
        mf = f"{row['median_div_frac']:.2f}" if divs else "-"
        print(f"{ln:>7} {len(v):>6} {ident:>4}/{len(v):<5} {md:>16} {mf:>9} "
              f"{row['baseline_ok']:>4}/{len(v):<3} {row['candidate_ok']:>4}/{len(v):<3}")

    tot = sum(r["cells"] for r in rows)
    ident = sum(r["identical"] for r in rows)
    print(f"\noverall: {ident}/{tot} character-identical "
          f"({100.0 * ident / tot:.1f}%), "
          f"answers correct {sum(r['baseline_ok'] for r in rows)}/{tot} baseline "
          f"vs {sum(r['candidate_ok'] for r in rows)}/{tot} candidate")
    if a.json:
        json.dump({"baseline": bj.get("arm"), "candidate": cj.get("arm"), "rows": rows},
                  open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
