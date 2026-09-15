#!/usr/bin/env python3
"""Score a four-arm A/B (T4) honestly.

    scripts/bench/plowbench-score.py --metric median_tpot_ms \
        ctl=<json> treat=<json> ctl2=<json> treat2=<json>

WHY THIS EXISTS. A four-arm run is ctl / treat / ctl2 / treat2, interleaved so machine drift lands
on both sides. The obvious scoring rule -- "the delta is real if it exceeds the control drift
|ctl - ctl2|" -- IS NOT SOUND, and produced a false positive in this campaign: a probe printed
`VERDICT: REAL -3.11 ms` on controls that agreed to 0.12 ms while its two IDENTICAL treatment arms
disagreed by 5.59 ms, 47x the control drift. One treatment arm was a null (-0.32 ms) and the other
an outlier (-5.91); the p99 tail fell monotonically with run order, i.e. the machine was still
warming. The control drift measured nothing about the variance the treatment arms actually had.

THE RULE HERE. Floor the effect on max(control drift, treatment spread), and REFUSE to convict at
all when the treatment arms disagree far more than the controls do -- that is evidence the run is
not in steady state, and no delta computed from it means anything.

Exit 0 = REAL, 1 = NOT CONVICTABLE, 2 = NULL (effect inside the floor).
"""
import argparse
import json
import sys


def load(spec, metric):
    tag, _, path = spec.partition("=")
    if not path:
        sys.exit(f"argument '{spec}' must be tag=path")
    with open(path) as fh:
        d = json.load(fh)
    if metric not in d:
        sys.exit(f"{path} has no '{metric}' (has: {', '.join(sorted(d))[:200]})")
    return tag, float(d[metric])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="median_tpot_ms")
    ap.add_argument("--lower-is-better", action="store_true", default=True)
    ap.add_argument("--spread-ratio", type=float, default=3.0,
                    help="refuse when treatment spread exceeds this multiple of control drift")
    ap.add_argument("arms", nargs=4, metavar="tag=path")
    a = ap.parse_args()

    vals = dict(load(s, a.metric) for s in a.arms)
    for need in ("ctl", "treat", "ctl2", "treat2"):
        if need not in vals:
            sys.exit(f"need arms ctl, treat, ctl2, treat2 — got {', '.join(vals)}")

    ctl = (vals["ctl"] + vals["ctl2"]) / 2.0
    treat = (vals["treat"] + vals["treat2"]) / 2.0
    drift = abs(vals["ctl"] - vals["ctl2"])
    tspread = abs(vals["treat"] - vals["treat2"])
    floor = max(drift, tspread)
    delta = treat - ctl

    print(f"metric: {a.metric}")
    for k in ("ctl", "treat", "ctl2", "treat2"):
        print(f"  {k:7s} {vals[k]:10.3f}")
    print(f"  control mean {ctl:.3f}   treatment mean {treat:.3f}")
    print(f"  control drift  |ctl - ctl2|     = {drift:.3f}")
    print(f"  treatment spread |treat-treat2| = {tspread:.3f}")
    print(f"  floor = max(drift, tspread)     = {floor:.3f}")
    print(f"  delta = {delta:+.3f}")
    print()

    if tspread > a.spread_ratio * max(drift, 1e-9):
        print(f"VERDICT: NOT CONVICTABLE — the two identical treatment arms disagree by "
              f"{tspread:.3f}, {tspread / max(drift, 1e-9):.1f}x the control drift.")
        print("         The run is not in steady state. Re-run; do not report this delta.")
        return 1
    if abs(delta) <= floor:
        print(f"VERDICT: NULL — |{delta:.3f}| is inside the {floor:.3f} floor.")
        return 2
    better = (delta < 0) if a.lower_is_better else (delta > 0)
    print(f"VERDICT: REAL {delta:+.3f} ({'improvement' if better else 'REGRESSION'}), "
          f"{abs(delta) / max(floor, 1e-9):.1f}x the floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
