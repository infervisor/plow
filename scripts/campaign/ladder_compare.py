#!/usr/bin/env python3
"""Plow vs reference on a bench ladder, cell by cell.

    ladder_compare.py --plow results.csv [results2.csv ...] --ref reference.csv [ref2.csv ...]

Both sides are `bench_*_serve.sh` CSVs (input_len,concurrency,ttft_ms,...,tpot_ms,...,out_tok_s).
Later files override earlier ones for the same (input_len, concurrency). Ratios are plow/ref for
latencies (lower is better) and ref/plow for throughput, so > 1.00 always means plow is behind.
"""
import argparse
import csv


def load(paths):
    rows = {}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(l for l in f if not l.startswith("#")):
                try:
                    key = (int(r["input_len"]), int(r["concurrency"]))
                    rows[key] = {k: float(r[k]) for k in ("ttft_ms", "tpot_ms", "out_tok_s") if r.get(k)}
                except (KeyError, ValueError):
                    continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plow", nargs="+", required=True)
    ap.add_argument("--ref", nargs="+", required=True)
    a = ap.parse_args()
    plow, ref = load(a.plow), load(a.ref)
    print(f"{'in':>6} {'C':>3} | {'TTFT plow':>10} {'ref':>9} {'x':>5} | {'TPOT plow':>9} {'ref':>7} {'x':>5} |"
          f" {'tok/s plow':>10} {'ref':>8} {'x':>5}")
    wins = total = 0
    for key in sorted(plow, key=lambda k: (k[1], k[0])):
        p, r = plow[key], ref.get(key)
        if not r:
            print(f"{key[0]:>6} {key[1]:>3} | {p['ttft_ms']:>10.1f} {'-':>9} {'':>5} | {p['tpot_ms']:>9.2f} {'-':>7} {'':>5} |"
                  f" {p['out_tok_s']:>10.1f} {'-':>8}")
            continue
        x = (p["ttft_ms"] / r["ttft_ms"], p["tpot_ms"] / r["tpot_ms"], r["out_tok_s"] / p["out_tok_s"])
        wins += sum(v < 1.0 for v in x)
        total += 3
        mark = lambda v: f"{v:>5.2f}" + ("*" if v < 1.0 else " ")
        print(f"{key[0]:>6} {key[1]:>3} | {p['ttft_ms']:>10.1f} {r['ttft_ms']:>9.1f} {mark(x[0])}| "
              f"{p['tpot_ms']:>8.2f} {r['tpot_ms']:>7.2f} {mark(x[1])}| {p['out_tok_s']:>10.1f} {r['out_tok_s']:>8.1f} {mark(x[2])}")
    print(f"plow ahead on {wins} of {total} metric-cells (* = ahead)")


if __name__ == "__main__":
    main()
