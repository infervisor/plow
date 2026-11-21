#!/usr/bin/env python3
"""Automated single-block roofline certification gate (Phase 2).

Enforces the hard empirical roofline qualification thresholds before promotion:
  - Memory-bound ops (Decode GEMV, linear recurrent scan): >= 80% empirical roofline
  - Compute-bound ops (Prefill GEMM, MoE dense): >= 75% empirical roofline

Usage:
    certify_kernel.py --input <block_roofline.json | measurement.json> [--out-record <record.json>]
"""
import argparse
import json
import sys
from pathlib import Path


def evaluate_efficiency(row, min_memory_eff=0.80, min_compute_eff=0.75):
    """Evaluate whether a single row meets the certification gate."""
    phase = row.get("phase", "decode")
    pct_str = row.get("pct_of_floor") or row.get("pct_floor")
    if pct_str is None:
        eff = row.get("efficiency", 0.0)
    else:
        # pct_of_floor in block_roofline is floor/measured * 100% or measured/floor
        # In op_roofline: efficiency = floor_us / measured_us
        eff = float(pct_str) / 100.0 if float(pct_str) > 1.0 else float(pct_str)

    bound = row.get("bound", "memory").lower()
    threshold = min_compute_eff if ("compute" in bound or phase == "prefill") else min_memory_eff

    passed = eff >= threshold
    return {
        "point": row.get("point", f"{phase}:{row.get('b', 1)}:{row.get('ctx', 0)}"),
        "phase": phase,
        "bound": bound,
        "efficiency": eff,
        "threshold": threshold,
        "passed": passed,
        "reason": None if passed else f"Efficiency {eff * 100:.1f}% below required threshold {threshold * 100:.1f}%",
    }


def certify_data(data, min_memory_eff=0.80, min_compute_eff=0.75):
    rows = data.get("rows", [])
    if not rows and "efficiency" in data:
        rows = [data]

    results = []
    all_passed = True
    for row in rows:
        res = evaluate_efficiency(row, min_memory_eff, min_compute_eff)
        results.append(res)
        if not res["passed"]:
            all_passed = False

    return all_passed, results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Input JSON report from block_roofline or micro-harness")
    ap.add_argument("--min-memory", type=float, default=0.80, help="Min memory-bound efficiency [default: 0.80]")
    ap.add_argument("--min-compute", type=float, default=0.75, help="Min compute-bound efficiency [default: 0.75]")
    ap.add_argument("--out-record", help="Optional output tunedb record JSON if qualified")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        sys.exit(f"Input file not found: {args.input}")

    data = json.loads(in_path.read_text())
    all_passed, results = certify_data(data, args.min_memory, args.min_compute)

    print(f"--- Single-Block Roofline Certification Gate ---")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['point']} ({r['bound']}): efficiency {r['efficiency']*100:.2f}% (required >= {r['threshold']*100:.2f}%)")
        if not r["passed"]:
            print(f"       Reason: {r['reason']}")

    if not all_passed:
        print("\nCertification: REJECTED (one or more points failed roofline threshold)")
        sys.exit(1)

    print("\nCertification: QUALIFIED (all points meet empirical roofline thresholds)")
    if args.out_record:
        rec = {
            "state": "qualified",
            "certification": {
                "results": results,
                "verified": True,
            }
        }
        Path(args.out_record).write_text(json.dumps(rec, indent=2) + "\n")
        print(f"Wrote qualified record to {args.out_record}")


if __name__ == "__main__":
    main()
