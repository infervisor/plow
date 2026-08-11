#!/usr/bin/env python3
"""Quick console summary of every ib result under harness/b2-ib (scratch)."""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
pat = sys.argv[1] if len(sys.argv) > 1 else "*"
for path in sorted(glob.glob(os.path.join(HERE, pat, "results", "*.json"))):
    rep = json.load(open(path))
    for r in rep["results"]:
        if r["id"] == "warmup":
            continue
        c = r["config"]
        ok, fail = r["successful_requests"], r["failed_requests"]
        tokreq = r["total_tokens"] / max(1, ok)
        print("%-34s %-20s vus=%-3d ok=%-5d fail=%-5d tok/s=%-7.1f tok/req=%-6.1f "
              "ttft_avg=%-8.0f ttft_p99=%-8.0f itl_avg=%-6.1f itl_p99=%-6.1f" % (
                  rep["config"]["run_id"], r["id"][:20], c["max_vus"], ok, fail,
                  r["token_throughput_secs"], tokreq,
                  r["time_to_first_token_ms"]["avg"], r["time_to_first_token_ms"]["p99"],
                  r["inter_token_latency_ms"]["avg"], r["inter_token_latency_ms"]["p99"]))
