#!/usr/bin/env python3
"""Derive max-users-under-SLO per engine/tag from b2-concurrency-*.json.

SLOs: ITL p99 <= 50 ms (TPOT), TTFT p99 <= 5000 ms (includes queueing).
Max users = highest fixed-VU (ConstantVUs) point meeting the SLO with zero
failed requests; also prints the unconstrained max-throughput point.
Scratch helper — the report tables are typed from this output.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PD = os.path.dirname(os.path.dirname(HERE))  # perf-data/../.. -> repo; adjust
PERF = os.path.join(os.path.dirname(os.path.dirname(HERE)))

ITL_SLO, TTFT_SLO = 50.0, 5000.0

for key in ("12b", "31b", "26b"):
    fn = os.path.join(PERF, f"b2-concurrency-{key}.json")
    if not os.path.exists(fn):
        continue
    d = json.load(open(fn))
    print(f"===== {key}  ({d['model']}) =====")
    tags = {}
    for r in d["runs"]:
        if r["executor"] != "ConstantVUs" or r["bench_id"] != "throughput":
            continue
        if r.get("valid") is False:
            continue
        tags.setdefault((r["tag"], r["prompt_tokens"]), []).append(r)
    for (tag, ptok), rows in sorted(tags.items()):
        rows.sort(key=lambda r: r["max_vus"])
        best_itl = best_ttft = best_both = None
        maxthru = max(rows, key=lambda r: r["aggregate_tok_s"])
        for r in rows:
            ok = r["failed_requests"] == 0
            if ok and r["itl_ms"]["p99"] <= ITL_SLO:
                best_itl = r
            if ok and r["ttft_ms"]["p99"] <= TTFT_SLO:
                best_ttft = r
            if ok and r["itl_ms"]["p99"] <= ITL_SLO and r["ttft_ms"]["p99"] <= TTFT_SLO:
                best_both = r
        def s(r):
            if r is None:
                return "none"
            return (f"VU={r['max_vus']} ({r['aggregate_tok_s']:.1f} tok/s, "
                    f"itl_p99={r['itl_ms']['p99']:.1f}, ttft_p99={r['ttft_ms']['p99']:.0f})")
        print(f"  {tag} @{ptok}tok:")
        print(f"    ITL<=50 p99 : {s(best_itl)}")
        print(f"    TTFT<=5s p99: {s(best_ttft)}")
        print(f"    both        : {s(best_both)}")
        print(f"    max thruput : {s(maxthru)}")
        for r in rows:
            print(f"      vu{r['max_vus']:<3} tok/s={r['aggregate_tok_s']:<7.1f} "
                  f"ok={r['successful_requests']:<5} fail={r['failed_requests']:<4} "
                  f"ttft avg/p99={r['ttft_ms']['avg']:.0f}/{r['ttft_ms']['p99']:.0f} "
                  f"itl avg/p99={r['itl_ms']['avg']:.1f}/{r['itl_ms']['p99']:.1f}")
