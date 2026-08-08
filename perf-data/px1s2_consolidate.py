#!/usr/bin/env python3
"""
px1s2_consolidate.py — build perf-data/px1-stage2.json from the raw
inference-benchmarker reports in perf-data/harness/b2-ib/px1s2-*/results/.

PX-1 stage 2: block-diagonal varlen
flash — all packed requests' prefill attention in ONE persistent-grid kernel
pass — three-armed against the serialized prefill (off) and the stage-1
per-request-serial attention (s1), same binary, same 12B ctx8k B=16 blob,
same harness/profile as the B2 concurrency campaign.

Every number is transcribed verbatim from the tool's own report JSON; nothing
is interpolated. The markdown companion px1-stage2.md is written by hand.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "harness", "b2-ib")

TAGS = {
    "px1s2-off": "B=16 blob: serialized prefill (PLOW_PF_BATCH unset, stage-2 cubin legacy path)",
    "px1s2-s1": "B=16 blob: stage-1 batched prefill — GEMM batching + per-request-serial "
                "attention (PLOW_PF_BATCH=1, stage-1 pf cubin)",
    "px1s2-varlen": "B=16 blob: stage-2 batched prefill — GEMM batching + block-diagonal "
                    "varlen flash, one kernel pass for all requests "
                    "(PLOW_PF_BATCH=1, stage-2 pf cubin)",
    "px1s2b8-off": "B=8 blob: serialized prefill (PLOW_PF_BATCH unset)",
    "px1s2b8-s1": "B=8 blob: stage-1 batched prefill (PLOW_PF_BATCH=1, stage-1 pf cubin)",
    "px1s2b8-varlen": "B=8 blob: stage-2 varlen batched prefill "
                      "(PLOW_PF_BATCH=1, stage-2 pf cubin)",
}

SLO_ITL_P99_MS = 50.0
SLO_TTFT_P99_MS = 5000.0


def pct(p):
    return {k: p[k] for k in ("avg", "p50", "p90", "p99")}


def main():
    runs = []
    for path in sorted(glob.glob(os.path.join(RAW, "px1s2*-*", "results", "*.json"))):
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
        "campaign": "PX-1 stage 2 (block-diagonal varlen flash for batched prefill)",
        "date": "2026-07-21",
        "plan": "the design notes PX-1, sequencing step 2",
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
        "slo": {
            "itl_p99_ms": SLO_ITL_P99_MS,
            "ttft_p99_ms": SLO_TTFT_P99_MS,
            "definition": "SLO capacity = max VUs with ITL p99 <= 50 ms AND "
                          "TTFT p99 <= 5 s (campaign convention)",
        },
        "gates": {
            "oracle": "runtime/tests/sm120_interp_op_test.cu varlen suite: 6 R-request "
                      "packs (mid-tile boundaries, shuffled slots, chunked qp0, sliding "
                      "window) each BIT-EXACT vs the stage-1 per-request-serial launches "
                      "AND within flash tolerance of the T4-fixed per-request f32 "
                      "reference (relL2 <= 2.0e-3); full 134-test suite PASS",
            "gate_a_token_identity": "PASS — 5-prompt set (short/~500/2x~4k/~6k tok, "
                                     "chunk-boundary crossers) concurrent-burst vs solo "
                                     "byte-identical per request on the varlen build "
                                     "(multi-request packed launches confirmed in the log)",
            "gate_b_bleed": "PASS — poison/victim pair, both submission orders: victim "
                            "byte-identical to solo ('4'); concat sensitivity control "
                            "flips to 'PINEAPPLE', proving the test detects "
                            "cross-request attention",
            "legacy_cross_check": "batched-solo vs serialized-solo byte-identical on all "
                                  "8 gate prompts",
            "harness": "perf-data/px1_gates.py + px1_run_gates.sh (PORT=8093, B=8 assets)",
        },
        "kernel_microbench": {
            "source": "runtime/nvidia/experiments/fa_varlen_bench.cu, grid 188x256, "
                      "50-iter cudaEvent means, R requests splitting a 2048-row "
                      "(interleave quantum) or 8192-row (cold) pack, kvlen 4096/8192",
            "sliding_hd256_2048rows_kv4096_speedup": {"R1": 0.98, "R2": 1.30, "R4": 1.30, "R8": 2.57},
            "full_hd512_2048rows_kv4096_speedup": {"R1": 0.93, "R2": 0.95, "R4": 1.24, "R8": 1.26},
            "sliding_hd256_8192rows_kv8192_speedup": {"R1": 0.98, "R2": 1.07, "R4": 1.07, "R8": 1.42},
            "full_hd512_8192rows_kv8192_speedup": {"R1": 0.93, "R2": 0.93, "R4": 1.01, "R8": 1.02},
            "note": "varlen vs stage-1 device-serial loop; sliding layers win the "
                    "partial-wave tails, the HBM-bound full layers lose L2 KV locality "
                    "when requests interleave",
        },
        "pack_stats": {
            "px1s2b8-s1": {"packs_R1": 2132, "packs_R2": 227},
            "px1s2b8-varlen": {"packs_R1": 2068, "packs_R2": 240},
            "note": "~90% of packed prefill launches carry ONE request at this "
                    "profile (4k prompts, 2048-row interleave quantum, B=8) — the "
                    "varlen lever mostly idle end-to-end",
        },
        "engine_configs": {
            "common": "plowrt serve (branch px1-varlen-flash), 12B ctx8k B=16 "
                      "blob (gemma4-12b-ctx8k-b16.pkt), decode cubin = "
                      "b16-mm16's interp_sm120.cubin, prefill cubin per arm "
                      "(both 240 regs, 0 spill), default flags (--slo-ms 250), "
                      "PORT=8093",
            **TAGS,
        },
        "runs": runs,
    }
    out = os.path.join(HERE, "px1-stage2.json")
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {out} ({len(runs)} rows)")

    # Three-arm table + SLO capacity on stdout (throughput rows only).
    rows = {}
    for r in runs:
        if r["bench_id"] == "warmup":
            continue
        rows[(r["max_vus"], r["tag"])] = r
    order = ["px1s2-off", "px1s2-s1", "px1s2-varlen",
             "px1s2b8-off", "px1s2b8-s1", "px1s2b8-varlen"]
    print(f"{'VU':>4} {'tag':>13} {'tok/s':>9} {'ITLp50':>8} {'ITLp99':>8} "
          f"{'TTFTp50':>9} {'TTFTp99':>9} {'ok':>4} {'fail':>4} {'SLO':>4}")
    slo_cap = {t: 0 for t in order}
    for (vu, tag) in sorted(rows, key=lambda kv: (kv[0], order.index(kv[1]))):
        r = rows[(vu, tag)]
        ok = (r["itl_ms"]["p99"] <= SLO_ITL_P99_MS
              and r["ttft_ms"]["p99"] <= SLO_TTFT_P99_MS
              and r["failed_requests"] == 0)
        if ok and vu > slo_cap[tag]:
            slo_cap[tag] = vu
        print(f"{vu:>4} {tag:>13} {r['aggregate_tok_s']:>9.1f} "
              f"{r['itl_ms']['p50']:>8.1f} {r['itl_ms']['p99']:>8.1f} "
              f"{r['ttft_ms']['p50']:>9.0f} {r['ttft_ms']['p99']:>9.0f} "
              f"{r['successful_requests']:>4} {r['failed_requests']:>4} "
              f"{'ok' if ok else '-':>4}")
    print("SLO capacity (max qualifying VUs):",
          {t: slo_cap[t] for t in order if any(k[1] == t for k in rows)})


if __name__ == "__main__":
    main()
