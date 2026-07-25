#!/usr/bin/env python3
"""
px1_consolidate.py — build perf-data/px1-stage1.json from the raw
inference-benchmarker reports in perf-data/harness/b2-ib/px1-{off,on}/results/.

PX-1 stage 1 (plans/rtx-11-prefill-experiments.md): cross-request prefill
batching at the GEMM level (naive per-request-serial attention), A/B'd against
the serialized prefill on the SAME binary, SAME assets (12B ctx8k B=16 blob),
SAME harness/profile as the B2 concurrency campaign.

Every number is transcribed verbatim from the tool's own report JSON; nothing
is interpolated. The markdown companion px1-stage1.md is written by hand.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "harness", "b2-ib")

TAGS = {
    "px1-off": "serialized prefill (PLOW_PF_BATCH unset) — the baseline path",
    "px1-on": "PX-1 cross-request batched prefill (PLOW_PF_BATCH=1)",
}


def pct(p):
    return {k: p[k] for k in ("avg", "p50", "p90", "p99")}


def main():
    runs = []
    for path in sorted(glob.glob(os.path.join(RAW, "px1-*", "results", "*.json"))):
        tag = path.split(os.sep)[-3]
        if tag not in TAGS:
            continue
        with open(path) as f:
            rep = json.load(f)
        cfg = rep["config"]
        meta = cfg.get("meta") or {}
        for r in rep["results"]:
            runs.append(
                {
                    "tag": tag,
                    "engine": "plow",
                    "engine_commit": meta.get("engine_commit"),
                    "campaign": meta.get("campaign"),
                    "run_id": cfg.get("run_id"),
                    "bench_id": r["id"],
                    "executor": r["executor_type"],
                    "max_vus": r["config"]["max_vus"],
                    "rate_req_s": r["config"].get("rate"),
                    "prompt_tokens": (cfg.get("prompt_options") or {}).get("num_tokens"),
                    "decode_tokens": (cfg.get("decode_options") or {}).get("num_tokens"),
                    "duration_ms": r["duration_ms"],
                    "total_requests": r["total_requests"],
                    "successful_requests": r["successful_requests"],
                    "failed_requests": r["failed_requests"],
                    "request_rate_req_s": r["request_rate"],
                    "total_tokens_out": r["total_tokens"],
                    "total_tokens_sent": r["total_tokens_sent"],
                    "aggregate_tok_s": r["token_throughput_secs"],
                    "ttft_ms": pct(r["time_to_first_token_ms"]),
                    "itl_ms": pct(r["inter_token_latency_ms"]),
                    "e2e_ms": pct(r["e2e_latency_ms"]),
                    "source_file": os.path.relpath(path, HERE),
                }
            )

    doc = {
        "campaign": "PX-1 stage 1 (GEMM-level cross-request prefill batching)",
        "date": "2026-07-21",
        "plan": "plans/rtx-11-prefill-experiments.md PX-1, sequencing step 1",
        "model": "/workspace/models/gemma-4-12B-it",
        "served_model_name": "gemma-4-12b-it",
        "tokenizer": "google/gemma-4-12B-it",
        "dtype": "bfloat16",
        "tp": 1,
        "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB "
               "(sm_120 / cc 12.0, 188 SMs)",
        "tool": {
            "name": "huggingface/inference-benchmarker",
            "version": "1.1.0",
            "rev": "bad4f947ef5f34ef264d2451439ab90cf7cbd65c",
            "binary": "target/tools/bin/inference-benchmarker via perf-data/bench_ib.sh",
        },
        "profile": {
            "prompt_tokens": 4000,
            "decode_tokens": 128,
            "dataset": "hlarcher/inference-benchmarker github_code.json, "
                       "entries truncated to 4000 gemma tokens",
            "sampling": "temperature 0 (greedy), stream=true, max_tokens=128",
            "warmup_s": 15,
            "duration_s": 120,
        },
        "engine_configs": {
            "common": "plowrt serve (branch px1-gemm-batching), 12B ctx8k B=16 "
                      "blob (gemma4-12b-ctx8k-b16.pkt), decode cubin = "
                      "b16-mm16's interp_sm120.cubin, prefill cubin rebuilt "
                      "from branch source (interp_sm120_pf: 240 regs, 0 spill), "
                      "default flags (--slo-ms 250)",
            **TAGS,
        },
        "gates": {
            "gate_a_token_identity": "PASS — 5 prompts (short/~500/2x~4k/~6k tok) "
                                     "concurrent-burst vs solo byte-identical per "
                                     "request (packed launches confirmed in the "
                                     "server log)",
            "gate_b_bleed": "PASS — poison/victim pair, both submission orders: "
                            "victim byte-identical to solo; concat sensitivity "
                            "control flips the answer, proving the test detects "
                            "cross-request attention",
            "legacy_cross_check": "batched-solo vs serialized-solo byte-identical "
                                  "on all 8 gate prompts (decode-step first token "
                                  "== prefill lm_head token)",
            "harness": "perf-data/px1_gates.py + px1_run_gates.sh",
        },
        "metric_conventions": {
            "ttft_ms": "client-side, request POST -> first SSE content token; "
                       "INCLUDES any server-side queueing (capacity benchmark)",
            "itl_ms": "client-side inter-token gap during streaming",
            "aggregate_tok_s": "total generated tokens / benchmark wall time, "
                               "successful requests only",
        },
        "runs": runs,
    }
    out = os.path.join(HERE, "px1-stage1.json")
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {out} ({len(runs)} rows)")

    # Quick A/B table on stdout (throughput rows only).
    rows = {}
    for r in runs:
        if r["bench_id"] == "warmup":
            continue
        rows[(r["max_vus"], r["tag"])] = r
    print(f"{'VU':>4} {'tag':>8} {'tok/s':>9} {'ITLp50':>8} {'ITLp99':>8} "
          f"{'TTFTp50':>9} {'TTFTp99':>9} {'ok':>4} {'fail':>4}")
    for (vu, tag), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"{vu:>4} {tag:>8} {r['aggregate_tok_s']:>9.1f} "
              f"{r['itl_ms']['p50']:>8.1f} {r['itl_ms']['p99']:>8.1f} "
              f"{r['ttft_ms']['p50']:>9.0f} {r['ttft_ms']['p99']:>9.0f} "
              f"{r['successful_requests']:>4} {r['failed_requests']:>4}")


if __name__ == "__main__":
    main()
