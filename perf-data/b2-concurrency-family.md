# B2-ib final-numbers — plow vs vLLM concurrency / capacity (Gemma-4, sm_120)

Campaign **B2-ib final-numbers v2**, 2026-07-21, `main` @ `b953a7b`.
Box: 1× RTX PRO 6000 Blackwell 96 GB (sm_120, 188 SMs). Harness:
`huggingface/inference-benchmarker` rev `bad4f947` (pinned; `perf-data/bench_ib.sh`),
identical tool / tokenizer / prompt profile / durations (15 s warm + 120 s
measure) for both engines. Greedy (temperature 0), streaming, 128 output
tokens. **TTFT includes server-side queueing** — the capacity convention.

Source of every number: `perf-data/b2-concurrency-12b.json` (transcribed
verbatim from the tool's own report JSON; `perf-data/harness/b2-ib/<tag>/results/`).

## Scope actually measured this pass

The GPU campaign was **stopped early by operator request** after the 12B
concurrency head-to-head and the 12B 16k-profile plow points completed. The
following did **not** run and are therefore **not** in this file:

- 16k profile **vLLM** comparator (plow-only 16k rows below).
- 31B and 26B **concurrency** sweeps (both engines).
- The 12B+26B **co-resident** (MM1) scenario.

31B and 26B are covered at **single-user** only, by the already-committed S6
served rows (`gemma4-31b-plowrt-served.md`, `gemma4-26b-plowrt-served.md`),
folded into the capacity report's headline. The bench scripts and blobs are
staged to finish those sweeps in a later lease (see the report's "what's
still open").

## Engine configs

- **plow:** `plowrt serve` (`--release --features cuda,hf-tokenizer`), **DEFAULT
  flags** (`--slo-ms 250`, post-`2ba3fbc` prefill-aware admission; chunk-
  interleaved prefill; `PLOW_UNISEG=1` blobs). A `PLOW_DECODE_BATCH=B` blob is
  **B mux slots** — per-slot prefill + batched decode; arrivals beyond B
  **queue** in the mux (their wait lands in TTFT). Valid capacity blob = **B=8**
  (`/root/gpu-assets-b4/b8`, ctx-8k). A ctx-24576 B=8 blob served the 16k rows.
- **vLLM:** 0.25.1, bf16, `--gpu-memory-utilization 0.90`, `--max-model-len 8192`
  (24576 for 16k), TP1, CUDA graphs ON, continuous batching + default prefix
  caching (not handicapped).

## Blob-validity gate (important) — UPDATED by campaign `B2-bfix`

**The original diagnosis here was wrong.** v2 claimed the 12B **B=16/B=24**
blobs were "emitter-broken (post-`638ce37`), pass 4-way but fail 8-way,
truncate to ~12 tok/req." Campaign `B2-bfix` (2026-07-21, mux fix `498a0f4`)
disproved it:

- With the admission shedder disabled (`--slo-ms 100000`) the **same** B=16 blob
  serves **8 and 16** concurrent distinct-prompt streams **byte-identical** to
  their B=1 runs; B=32 loads/decodes correctly; compute-sanitizer memcheck on
  every batched decode kernel at M≤32 = **0 errors**. The kernel is correct.
- The truncation was the **mux admission shed**: `predicted_wait = live ×
  service_ms` 429'd every live slot once `live × step_ms > --slo-ms`; a B=16 blob
  steps at ~40 ms/tok, so 8 users (320 ms) > 250 ms default → shed. Fixed to
  `ceil(live / batch) × service_ms` (B=1 byte-identical).

The re-run tags **`plow-b16-bfix`** (and `plow-b32-bfix`) use the fixed binary,
pass the **8/16/32-way** token-identity gate, run VU1..16 with **0 sheds / 0
failures**, and are the **valid** B>8 rows. The original `plow-b16`/`plow-b24`
rows stay `"valid": false` as the pre-fix transcript. **But B=16 does not raise
SLO-capacity** — peak 93 vs B=8's 97 tok/s (bandwidth-bound; `GV_MM_MAX=8` ⇒
B=16 = 2 weight passes). The real cap is decode HBM bandwidth + slow prefill,
**not** the batch size. See the capacity report §4.

---

## 12B bf16 — 4k prompt / 128 out (the head-to-head)

Fixed virtual users (ConstantVUs), 120 s each, 0 failed requests throughout.

| engine | VU | agg tok/s | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) | ITL p99 (ms) | tok/req |
|---|---:|---:|---:|---:|---:|---:|---:|
| **plow** B=8 | 1  | 35.8  | 804   | 810   | 21.8  | 21.9  | 128 |
| **plow** B=8 | 2  | 54.4  | 983   | 1483  | 28.9  | 43.4  | 127 |
| **plow** B=8 | 4  | 75.5  | 1074  | 2966  | 43.8  | 59.8  | 127 |
| **plow** B=8 | 8  | 97.2  | 1301  | 6642  | 69.9  | 81.6  | 127 |
| **plow** B=8 | 16 | 97.1  | 10524 | 16466 | 70.4  | 82.4  | 127 |
| vLLM | 1  | 44.0  | 375   | 390   | 20.0  | 20.1  | 128 |
| vLLM | 2  | 83.5  | 378   | 718   | 21.0  | 26.9  | 127 |
| vLLM | 4  | 126.1 | 900   | 1353  | 24.6  | 30.4  | 127 |
| vLLM | 8  | 177.3 | 1466  | 2675  | 32.5  | 41.8  | 128 |
| vLLM | 16 | 220.7 | 1705  | 5320  | 56.7  | 68.5  | 127 |
| vLLM | 32 | 238.8 | 2511  | 10752 | 105.9 | 122.7 | 127 |
| vLLM | 64 | 235.4 | 4993  | 22243 | 205.3 | 233.5 | 127 |

(vLLM-rerun agrees within noise: VU8 178.0 tok/s, VU16 221.3, VU32 239.2.)

### Max sustainable users per engine (12B, 4k)

Highest fixed-VU point meeting the SLO with **zero** failures:

| SLO | plow (B=8) | vLLM |
|---|---|---|
| ITL/TPOT p99 ≤ 50 ms | **2 users** (54 tok/s) | **8 users** (177 tok/s) |
| TTFT p99 ≤ 5 s        | **4 users** (76 tok/s) | **8 users** (177 tok/s)* |
| **both**              | **2 users** (54 tok/s) | **8 users** (177 tok/s) |
| unconstrained max-throughput | **8 users → 97 tok/s** | **32 users → 239 tok/s** |

\* vLLM VU16 TTFT p99 = 5.32 s just misses the 5 s bar; VU8 is the clean point.

**vLLM wins the 12B concurrency/capacity contest decisively: ~4× the
SLO-bounded users (8 vs 2) and 2.5× peak throughput (239 vs 97 tok/s).**
Two independent causes, both structural to plow today: (1) the **B=8 batch
cap** (the B=16/24 blobs are broken — above), so plow's mux serializes the 9th+
arrival into TTFT (VU16 TTFT p99 jumps to 16.5 s while decode ITL stays flat
at 82 ms — textbook queueing, not a decode regression); (2) plow's **prefill
is ~2× slower** (TTFT 804 ms vs 375 ms at VU1), documented structural (HBM-
bound flash, no TMA/FA4). vLLM's continuous batching interleaves the two.

## 12B bf16 — 16k prompt / 128 out (plow only; no vLLM comparator ran)

| engine | VU | agg tok/s | TTFT avg (ms) | TTFT p99 (ms) | ITL avg (ms) | ITL p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| **plow** B=8 (ctx24576) | 1 | 23.4 | 2491 | 2536  | 22.8  | 22.9  |
| **plow** B=8 (ctx24576) | 2 | 27.7 | 3198 | 4925  | 45.9  | 53.9  |
| **plow** B=8 (ctx24576) | 4 | 30.8 | 3942 | 10809 | 97.1  | 233.3 |
| **plow** B=8 (ctx24576) | 8 | 30.5 | 5995 | 23617 | 187.3 | 207.4 |

plow 16k SLO capacity (both SLOs, zero fails): **1 user** (23 tok/s, ITL p99
22.9 ms, TTFT p99 2.5 s). At 16k the 128 KiB/req prefill dominates and the
B=8 mux saturates by VU4. The vLLM 16k comparator is queued for the next lease.

## Data / reproduction

- `perf-data/b2-concurrency-12b.json` — all rows + percentiles + sweep grid.
- `perf-data/bench_b2_ib.sh` — parameterized driver (`CAMPAIGN`, `PROMPT_TOKS`,
  `VUS`, `DO_SWEEP`, `MODEL_NAME`, `TOKENIZER`, `ASSETS`); one server config per
  `gpulease`.
- `perf-data/harness/b2-ib/{slo_capacity,summarize}.py` — scratch reducers.
