#!/usr/bin/env python3
"""
consolidate_b2_ib.py — build perf-data/b2-concurrency-{12b,31b,26b}.json from
the raw inference-benchmarker reports in perf-data/tools/b2-ib/<tag>/results/.

Final-numbers campaign (v2): multi-user concurrency/capacity head-to-head,
vLLM 0.25.1 vs `plowrt serve`, whole Gemma-4 family, bf16:
  B2-ib-12b     12B, 4k prompt / 128 out   (tags vllm, vllm-rerun, plow-b8,
                plow-b16*, plow-b24*)      *flagged invalid, see below
  B2-ib-12b-16k 12B, 16k prompt / 128 out  (tags vllm-16k, plow-b8-24k)
  B2-ib-31b     31B, 4k / 128              (tags vllm-31b, plow-31b-b1)
  B2-ib-26b     26B-A4B, 4k / 128          (tags vllm-26b, plow-26b-b1/-b8)
  MM1-ib        12B+26B co-resident, one plowrt process, simultaneous load
                (tags mm-12b, mm-26b; rows land in each model's file)

Every number is transcribed verbatim from the tool's own report JSON; nothing
is interpolated. One row per (tag, benchmark id). The markdown companion
b2-concurrency-family.md is written by hand from these JSONs.

VALIDITY: rows carry "valid": false when the serving config failed its
token-identity gate. The B=16/B=24 12B blobs (HEAD emitter, post-638ce37
B>8 path) pass 4-way identity but FAIL 8-way (streams truncate, ~12 tok/req
at VU>=8, ITL ~0) — their rows are transcribed for the record but MUST NOT
be used as capacity numbers. Max gate-passing plow batch on 2026-07-21 = 8.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "harness", "b2-ib")

IB_REV = "bad4f947ef5f34ef264d2451439ab90cf7cbd65c"

# tag -> model file key
TAG_MODEL = {
    "vllm": "12b", "vllm-rerun": "12b", "plow-b8": "12b", "plow-b16": "12b",
    "plow-b24": "12b", "vllm-16k": "12b", "plow-b8-24k": "12b", "mm-12b": "12b",
    # B2-bfix campaign: the fixed batched-admission blobs (see INVALID note).
    "plow-b16-bfix": "12b", "plow-b32-bfix": "12b",
    "vllm-31b": "31b", "plow-31b-b1": "31b",
    "vllm-26b": "26b", "plow-26b-b1": "26b", "plow-26b-b8": "26b",
    "mm-26b": "26b",
}
# tags whose serving config failed the 8-way token-identity gate.
# NOTE: plow-b16/-b24's failure was NOT a kernel/emitter bug (as first
# believed) — it was the mux admission shed. `predicted_wait = live *
# service_ms` (serial M/M/1) shed every live slot once live*step_ms crossed
# --slo-ms, and a correct B=16 blob steps at ~40 ms/tok, so 8*40=320>250 killed
# the streams. Fixed in mux.rs (commit "serve: fix B>8 capacity cap"). The
# re-run tags plow-b16-bfix / plow-b32-bfix use the FIXED binary and pass the
# 8/16/32-way token-identity gate — they are the valid capacity numbers. The
# original plow-b16/-b24 rows stay flagged invalid as the pre-fix transcript.
INVALID = {
    "plow-b16": "B=16 blob shed to ~1-12 tok/req at >=8 active slots under the "
                "PRE-FIX serial admission model (live*service_ms > --slo-ms); "
                "NOT a kernel bug. Superseded by plow-b16-bfix. Kept as the "
                "pre-fix transcript, not a capacity number",
    "plow-b24": "B=24 blob: same pre-fix admission shed as plow-b16. Kept as "
                "transcript, not a capacity number",
    "plow-b32-bfix": "ctx-4096 B=32 blob (built to fit 42 GiB KV) CANNOT hold "
                "the 4k-prompt+128 profile: prompt+max_tokens = 4128 > 4096, so "
                "the ib sweep rejected every request (context-exhausted, not a "
                "shed). The B=32 blob was built for CORRECTNESS gating only "
                "(32-way token identity with short prompts + compute-sanitizer); "
                "its 4k-profile sweep rows are a ctx/profile mismatch, NOT "
                "capacity. A valid B=32 sweep needs a ctx>=~4300 blob (>47 GiB "
                "KV) — b16 already shows B>8 is bandwidth-bound (peak ties B=8)",
}

MODEL_META = {
    "12b": {
        "model": "/workspace/models/gemma-4-12B-it",
        "model_revision": "12ace6d648d72bd41519e140f1185f34d38c7e3d",
        "served_model_name": "gemma-4-12b-it",
        "tokenizer": "google/gemma-4-12B-it",
    },
    "31b": {
        "model": "/workspace/models/gemma-4-31B-it",
        "model_revision": "b9ea41a2887d8607f594846523f94c6cc75ac8a4",
        "served_model_name": "gemma-4-31b-it",
        "tokenizer": "google/gemma-4-31B-it",
    },
    "26b": {
        "model": "/workspace/models/gemma-4-26B-A4B-it",
        "model_revision": "01e5b3ee840d3a9e0b0b493c593e85398a30ef75",
        "served_model_name": "gemma-4-26b-a4b-it",
        "tokenizer": "google/gemma-4-26B-A4B-it",
    },
}

ENGINE_CONFIGS = {
    "vllm": "vLLM 0.25.1 venv, bf16, --gpu-memory-utilization 0.90, "
            "--max-model-len 8192 (24576 for the 16k profile), TP1, CUDA "
            "graphs ON (no --enforce-eager), continuous batching + default "
            "prefix caching (not handicapped)",
    "plow": "plowrt serve --release --features cuda,hf-tokenizer @ main "
            "b953a7b, DEFAULT flags (--slo-ms 250, post-2ba3fbc "
            "prefill-aware admission; chunk-interleaved prefill; "
            "PLOW_UNISEG=1 blobs). PLOW_DECODE_BATCH=B blob = B mux slots, "
            "per-slot prefill + batched decode; arrivals beyond B QUEUE in "
            "the mux (TTFT includes that wait). Blob per tag: plow-b8 = "
            "ctx8k B=8 (gpu-assets-b4), plow-b8-24k = ctx24576 B=8, "
            "plow-26b-b8 = 26B ctx8k B=8, plow-{31b,26b}-b1 = ctx132096 "
            "B=1 (S6 assets), mm-* = both 132k B=1 blobs co-resident in "
            "ONE plowrt process under simultaneous load. plow-b16-bfix = "
            "ctx8k B=16 and plow-b32-bfix = ctx4096 B=32, both served by the "
            "batched-admission-fixed binary (mux predicted_wait = "
            "ceil(live/B)*service_ms) — these pass the 8/16/32-way "
            "token-identity gate and are the valid B>8 capacity numbers "
            "(per-row engine_commit stamps the fixed build)",
}


def pct(p):
    return {k: p[k] for k in ("avg", "p50", "p90", "p99")}


def main():
    per_model = {"12b": [], "31b": [], "26b": []}
    for path in sorted(glob.glob(os.path.join(RAW, "*", "results", "*.json"))):
        tag = path.split(os.sep)[-3]
        if tag not in TAG_MODEL:
            continue
        with open(path) as f:
            rep = json.load(f)
        cfg = rep["config"]
        meta = cfg.get("meta") or {}
        for r in rep["results"]:
            row = {
                "tag": tag,
                "engine": meta.get("engine", tag.split("-")[0]),
                "engine_commit": meta.get("engine_commit"),
                "campaign": meta.get("campaign"),
                "run_id": cfg.get("run_id"),
                "bench_id": r["id"],              # warmup | throughput | constant@X.XXreq/s
                "executor": r["executor_type"],   # ConstantVUs | ConstantArrivalRate
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
            if meta.get("coresident"):
                row["coresident"] = meta["coresident"]
            if tag in INVALID:
                row["valid"] = False
                row["invalid_reason"] = INVALID[tag]
            per_model[TAG_MODEL[tag]].append(row)

    for key, runs in per_model.items():
        if not runs:
            continue
        doc = {
            "campaign": "B2-ib final-numbers v2",
            "date": "2026-07-21",
            **MODEL_META[key],
            "dtype": "bfloat16",
            "tp": 1,
            "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB "
                   "(sm_120 / cc 12.0, 188 SMs)",
            "tool": {
                "name": "huggingface/inference-benchmarker",
                "version": "1.1.0",
                "rev": IB_REV,
                "pinned_in": "Cargo.lock via tools/bench (optional dep, feature ib)",
                "binary": "target/tools/bin/inference-benchmarker via perf-data/bench_ib.sh",
            },
            "profile": {
                "prompt_tokens": "per row (4000 or 16000), variance 0",
                "decode_tokens": 128,
                "dataset": "hlarcher/inference-benchmarker github_code.json, "
                           "entries truncated to the profile's gemma token count",
                "sampling": "temperature 0 (greedy), stream=true, max_tokens=128",
                "warmup_s": 15, "duration_s": 120,
            },
            "metric_conventions": {
                "ttft_ms": "client-side, request POST -> first SSE content token; "
                           "INCLUDES any server-side queueing (capacity benchmark)",
                "itl_ms": "client-side inter-token gap during streaming",
                "aggregate_tok_s": "total generated tokens / benchmark wall time, "
                                   "successful requests only",
                "failed_requests": "HTTP errors (incl. plow 429 sheds) + streams "
                                   "closed before completion (benchmark-end cutoff)",
                "valid": "false = the serving config failed its token-identity "
                         "gate; row is transcript, not capacity evidence",
            },
            "engine_configs": ENGINE_CONFIGS,
            "runs": runs,
        }
        out = os.path.join(HERE, f"b2-concurrency-{key}.json")
        with open(out, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"wrote {out}: {len(runs)} rows from "
              f"{len(set(r['tag'] for r in runs))} tags")


if __name__ == "__main__":
    main()
