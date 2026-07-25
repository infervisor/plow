#!/usr/bin/env python3
"""scripts/block_compare.py — diff a plow block sweep against a baseline sweep.

Ties the two halves of the block harness together:

  plow side      `block_run <asset> bench`        -> /dev/shm/block-asset/bench/sweep.json
  baseline side  `block_layer_bench.py` (vLLM's own decoder layer)
                 or `block_baseline.py` (hand-written block)

Both emit the same row schema, keyed by (batch, ctx):

    {"batch": B, "ctx": T, "latency_us_median": .., "latency_us_p95": .., "tok_s": ..}

so this joins them on (B,T) and reports the ratio. ratio < 1 means plow is
faster. Rows present in only one file are listed, not silently dropped, and a
non-zero exit is returned if the two files share no comparable points.

Usage:
  python3 scripts/block_compare.py --plow sweep.json --baseline 12b-layer.json
  python3 scripts/block_compare.py --plow a.json --baseline b.json --phase prefill
  python3 scripts/block_compare.py --plow a.json --baseline b.json --json out.json

No dependencies (stdlib only) so it runs outside the vLLM venv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DECODE_KEY = "latency_us_median"
PREFILL_KEY = "prefill_ms_median"


def load(path: str) -> tuple[dict, dict]:
    d = json.loads(Path(path).read_text())
    rows = {(int(r["batch"]), int(r["ctx"])): r for r in d.get("sweep", [])}
    if not rows:
        raise SystemExit(f"{path}: no 'sweep' rows")
    return d, rows


def main() -> int:
    ap = argparse.ArgumentParser(description="compare plow vs baseline block sweeps")
    ap.add_argument("--plow", required=True, help="sweep.json from block_run bench")
    ap.add_argument("--baseline", required=True, help="sweep.json from a baseline harness")
    ap.add_argument("--phase", default="decode", choices=["decode", "prefill"])
    ap.add_argument("--json", dest="out", default=None, help="write comparison json here")
    args = ap.parse_args()

    pmeta, prows = load(args.plow)
    bmeta, brows = load(args.baseline)
    key = DECODE_KEY if args.phase == "decode" else PREFILL_KEY
    unit = "us" if args.phase == "decode" else "ms"

    pdev, bdev = pmeta.get("device", "?"), bmeta.get("device", "?")
    print(f"phase={args.phase}  plow={Path(args.plow).name}  baseline={Path(args.baseline).name}"
          f" ({bmeta.get('baseline','?')})")
    print(f"  plow device={pdev}")
    print(f"  base device={bdev}")
    if pdev != bdev and "?" not in (pdev, bdev):
        print("  *** DEVICE MISMATCH — cross-GPU block numbers are NOT comparable ***")

    shared = sorted(set(prows) & set(brows))
    if not shared:
        print("no (batch,ctx) points in common", file=sys.stderr)
        return 2

    print(f"\n  {'B':>3} {'T':>7} {'plow':>12} {'baseline':>12} {'ratio':>8}  (ratio<1 = plow faster)")
    out_rows, ratios = [], []
    for b, t in shared:
        pv, bv = prows[(b, t)].get(key), brows[(b, t)].get(key)
        if pv is None or bv is None:
            print(f"  {b:>3} {t:>7}  {'—':>12} {'—':>12}  (missing {key})")
            continue
        ratio = pv / bv if bv else float("inf")
        ratios.append(ratio)
        print(f"  {b:>3} {t:>7} {pv:>10.2f}{unit} {bv:>10.2f}{unit} {ratio:>7.2f}x")
        out_rows.append({"batch": b, "ctx": t, "plow": pv, "baseline": bv,
                         "ratio": round(ratio, 4)})

    for label, only in (("plow", set(prows) - set(brows)), ("baseline", set(brows) - set(prows))):
        if only:
            print(f"  ({label}-only points, not compared: "
                  f"{', '.join(f'B{b}/T{t}' for b, t in sorted(only))})")

    if ratios:
        ratios.sort()
        print(f"\n  median ratio = {ratios[len(ratios)//2]:.2f}x over {len(ratios)} points")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"phase": args.phase, "plow_device": pdev, "baseline_device": bdev,
             "baseline_kind": bmeta.get("baseline"), "rows": out_rows}, indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
