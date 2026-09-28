#!/usr/bin/env python3
"""waterfall.py <plow-resdir> <ref-resdir> [--max-prefix-hit 0.05] [--max-spread 0.05]

The audit tables for one matched grid (scripts/bench/llm_grid.sh, or any run dirs with pb_cell's
cells.log, pb_bench result dirs, the plow server.log with PLOW_PF_PACKLOG=1 and the reference's
metrics.tsv from pb_metrics_start), as markdown:

1. grid: out tok/s, TTFT p50, TPOT p50 per cell, mean over repeats (<cell>.r<k>) with the spread
   (max-min)/mean; spreads above --max-spread are flagged.
2. hygiene: the reference's prefix-cache hit rate per cell; above --max-prefix-hit the cell reused
   prompts and is not a matched comparison (exit 1).
3. waterfall: wall ms per request split by where the device time went. plow (PACKLOG, segments in
   cell order): mixed launches (prefill + riders), prefill-only launches, decode ticks, host gap,
   idle, and inside the launches the bucket padding and the rider cost over a pure launch.
   Reference (/metrics, 50 ms polls): engine time in intervals that scheduled prompt tokens
   (mixed) vs pure decode.
4. decode steps: plow decode-tick ms/step and rows/step vs the reference decode-phase ms/step.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packlog_audit  # noqa: E402
import vllm_metrics  # noqa: E402


def cells_log(res):
    order = []
    for ln in open(f"{res}/cells.log"):
        k, tag, _ = ln.split()
        if k == "CELL_BEGIN":
            order.append(tag)
    return order


def bench(res, tag):
    for p in (f"{res}/{tag}/bench.json", f"{res}/{tag}/main/bench.json",
              *glob.glob(f"{res}/{tag}/**/*.json", recursive=True)):
        if os.path.isfile(p):
            return json.load(open(p))
    return None


def base(tag):
    return re.sub(r"\.r\d+$", "", tag)


def grid(res):
    g = collections.defaultdict(list)
    for tag in cells_log(res):
        d = bench(res, tag)
        if d:
            g[base(tag)].append(d)
    return g


def agg(runs, key):
    xs = [r[key] for r in runs]
    m = sum(xs) / len(xs)
    return m, (max(xs) - min(xs)) / m if len(xs) > 1 and m else None


def plow_split(res):
    """PACKLOG segments mapped to cells in order; None when the counts disagree."""
    tags = cells_log(res)
    segs = [s for s in packlog_audit.segments(f"{res}/server.log") if len(s["ticks"]) >= 3]
    if len(segs) != len(tags):
        print(f"warning: {res}: {len(segs)} PACKLOG segments vs {len(tags)} cells; waterfall skipped "
              "(was PLOW_PF_PACKLOG=1 set, and nothing else sent traffic?)", file=sys.stderr)
        return {}
    return {t: packlog_audit.account(s) for t, s in zip(tags, segs)}


def fmt(x, w=7, p=2):
    return "-".rjust(w) if x is None else f"{x:{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plow")
    ap.add_argument("ref")
    ap.add_argument("--max-prefix-hit", type=float, default=0.05)
    ap.add_argument("--max-spread", type=float, default=0.05)
    a = ap.parse_args()
    gp, gr = grid(a.plow), grid(a.ref)
    rm = vllm_metrics.cells(a.ref) if os.path.isfile(f"{a.ref}/metrics.tsv") else {}
    ps = plow_split(a.plow) if os.path.isfile(f"{a.plow}/server.log") else {}
    bad = False

    print("## Grid (mean over repeats; spread = (max-min)/mean)\n")
    print("| cell | n | plow tok/s | spread | ref tok/s | spread | plow/ref | TTFT p50 plow / ref | TPOT p50 plow / ref |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in sorted(set(gp) | set(gr), key=lambda t: (t[0], int(re.sub(r"\D.*", "", t[1:]) or 0), t)):
        p, r = gp.get(c), gr.get(c)
        pt, psd = agg(p, "output_throughput") if p else (None, None)
        rt, rsd = agg(r, "output_throughput") if r else (None, None)
        flag = lambda s: "" if s is None else (f"{100 * s:.1f}%" + (" **!**" if s > a.max_spread else ""))
        ttft = " / ".join(fmt(agg(x, "median_ttft_ms")[0], 1, 1) if x else "-" for x in (p, r))
        tpot = " / ".join(fmt(agg(x, "median_tpot_ms")[0], 1, 2) if x else "-" for x in (p, r))
        ratio = f"{pt / rt:.2f}" if pt and rt else "-"
        print(f"| {c} | {len(p or r)} | {fmt(pt, 1, 0)} | {flag(psd)} | {fmt(rt, 1, 0)} | {flag(rsd)} | {ratio} | {ttft} | {tpot} |")

    if rm:
        print("\n## Reference prefix-cache hits\n")
        hits = {t: s["prefix_hit"] for t, s in rm.items()}
        over = [t for t, h in hits.items() if h > a.max_prefix_hit]
        print(" ".join(f"{t} {100 * h:.1f}%" for t, h in hits.items()))
        if over:
            bad = True
            print(f"\n**FAIL: prefix-cache hit rate > {100 * a.max_prefix_hit:.0f}% in {over}** — the reference "
                  "skipped prefill for reused prompts; re-run with unique prompts per cell.")

    print("\n## Waterfall: wall ms per request\n")
    print("| cell | plow wall | mixed | prefill-only | decode | host gap | idle | (padding) | (riders over pure) "
          "| ref wall | ref mixed | ref decode |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in cells_log(a.plow):
        d, b = ps.get(tag), bench(a.plow, tag)
        if not d or not b:
            continue
        n = b["completed"]
        acc = d["acc"]
        row = [d["wall"], acc["mixed"]["ms"], acc["prefill"]["ms"], acc["decode"]["ms"], d["gap"], d["idle"],
               d["pad_ms"], d["ride_extra"]]
        ref = rm.get(tag)
        rb = bench(a.ref, tag)
        if ref and rb:
            rr = [ref["active_s"] * 1e3, ref["mixed_s"] * 1e3, ref["decode_s"] * 1e3]
            rn = rb["completed"]
        else:
            rr, rn = [None] * 3, 1
        print(f"| {tag} | " + " | ".join(fmt(x / n) for x in row) + " | " +
              " | ".join(fmt(None if x is None else x / rn) for x in rr) + " |")

    print("\n## Decode step (ms/step)\n")
    print("| cell | plow decode ms/step | plow rows/step | ref decode-phase ms/step | ref running |")
    print("|---|---|---|---|---|")
    for tag in cells_log(a.plow):
        d = ps.get(tag)
        dec = d["acc"]["decode"] if d else {}
        ref = rm.get(tag, {})
        pms = dec["ms"] / dec["steps"] if dec.get("steps") else None
        prow = dec["rowsteps"] / dec["steps"] if dec.get("steps") else None
        print(f"| {tag} | {fmt(pms)} | {fmt(prow, 5, 1)} | {fmt(ref.get('decode_ms_per_step'))} | "
              f"{fmt(ref.get('decode_running'), 5, 0)} |")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
