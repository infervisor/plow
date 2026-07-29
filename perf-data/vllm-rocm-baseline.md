# vLLM ROCm baseline — 5 models on 8× MI355X (gfx950)

Measured 2026-07-27 with `scripts/bench_vllm_rocm.sh` (driver: `scripts/bench_vllm_all.sh`),
every run under a per-GPU `gpulease`. Image: **`rocm/vllm:rocm7.14.0_cdna_ubuntu24.04_py3.14_pytorch_2.11.0_vllm_0.23.0`**
→ vLLM `0.23.1.dev1+g9ddef7117.rocm714`, torch `2.11.0+rocm7.14.0`. Host ROCm 7.2.4.

**Not comparable to `vllm-tp-baseline.md`**, which used a different image
(`vllm/vllm-openai-rocm:latest`, vLLM 0.25.1+rocm723) — different vLLM *and* different ROCm.

Checkpoints served straight from the HF cache, read-only. Every run passed a
coherence gate (`/v1/chat/completions` → "Paris") before any number was taken.

## Concurrency-1 context sweep — TPOT ms/token (lower better)

| ctx | gemma-4-26B-A4B (TP1) | gemma-4-12B (TP1) | gemma-4-31B (TP1) | GLM-5.2-FP8 (TP4) | Kimi-K2.7-Code (TP4) |
|-----:|---:|---:|---:|---:|---:|
| 1k   | **4.90** | 6.80  | 13.51 | 24.93 | 23.04 |
| 4k   | **5.35** | 7.57  | 14.40 | 23.51 | 24.56 |
| 8k   | **5.95** | 8.62  | 15.57 | 25.07 | 24.06 |
| 16k  | **6.33** | 9.21  | 16.42 | 23.91 | 25.93 |
| 32k  | **7.61** | 11.18 | 19.00 | 24.16 | 26.71 |
| 64k  | **8.48** | 12.70 | 20.79 | 24.38 | 26.64 |
| 1k→64k | 1.73× | 1.87× | 1.54× | **0.98×** | 1.16× |

**GLM's TPOT does not degrade with context** (0.98× over a 64× range) — its DSA sparse
attention decouples decode cost from context length. Every dense model degrades 1.5–1.9×.
At 1k the 31B dense is 1.8× faster than GLM; by 64k GLM has overtaken it (24.38 vs 20.79 is
still behind, but the *trend* has crossed — GLM is flat while 31B keeps climbing).

## Concurrency-1 context sweep — TTFT ms (prefill)

| ctx | 26B-A4B | 12B | 31B | GLM-5.2 | Kimi |
|-----:|---:|---:|---:|---:|---:|
| 1k   | 18.1  | 28.1  | 79.8    | 37.4   | 51.4   |
| 8k   | 117.3 | 196.8 | 423.3   | 517.4  | 467.7  |
| 32k  | 797.1 | 1337.3| 3033.3  | 2100.5 | 2137.0 |
| 64k  | 2527.5| 4279.4| 10199.8 | 4776.9 | 5958.7 |

**gemma-4-31B's long-context prefill is the weak spot of the set**: 10.2 s TTFT at 64k, and
prefill throughput decays 19.4k→6.4k tok/s (3.0×) across the sweep — steeper than any other
model. Decode is not the problem; prefill is.

## General — concurrency sweep at ctx 1024, output tok/s

| conc | 26B-A4B | 12B | 31B | GLM-5.2 | Kimi |
|-----:|---:|---:|---:|---:|---:|
| 1  | 184.8 | 134.8 | 70.1  | 25.6*  | 39.8 |
| 4  | 537.3 | 479.1 | 234.1 | 139.1  | 110.7 |
| 16 | 1436.9| 1227.2| 596.4 | 400.2  | 256.5 |
| 64 | **3546.8** | 2778.5| 1416.2| 933.6  | 472.7 |

`*` GLM's conc=1 general row is warm-up-polluted (its clean conc=1 point is the ctxsweep
row: TTFT 37.4 ms). A discarded warm-up point was added to the harness afterwards; every
other row here is warm.

The 26B-A4B MoE wins on latency *and* throughput despite being larger on disk than the 12B —
~4B active parameters. The 31B dense costs ~2× the 12B's TPOT for ~2.6× the parameters.

## Per-model configuration — and why it differs

| model | TP | dtype | AITER | notes |
|---|---:|---|---|---|
| gemma-4-12B-it      | 1 | bf16 | off | |
| gemma-4-26B-A4B-it  | 1 | bf16 | off | MoE, ~4B active |
| gemma-4-31B-it      | 1 | bf16 | off | |
| zai-org/GLM-5.2-FP8 | 4 | fp8 e4m3 block | **REQUIRED** | DSA sparse attention |
| moonshotai/Kimi-K2.7-Code | 4 | int4 compressed-tensors | **MUST BE OFF** | see below |

AITER is not a free knob here — each of the two large models forces it in opposite directions.

### GLM-5.2 requires AITER
Without `VLLM_ROCM_USE_AITER=1` the engine aborts at startup:
`RuntimeError: Sparse attention indexer ROCm path is only supported on AITER.`
There is no generic ROCm kernel for the DSA indexer.

**Caveat on GLM's numbers:** AITER has no tuned GEMM config for GLM's shapes on gfx950 —
`not found tuned config in a8w8_blockscale_tuned_gemm.csv, will use default config`. These are
an untuned floor, not GLM's ceiling.

### Kimi-K2.7-Code CRASHES with AITER — likely an int4 miscompile
With AITER on, the model loads, passes the coherence gate, answers correctly on short
prompts, then dies under benchmark load with all four GPUs faulting:

```
Memory access fault by GPU node-3/4/5/6 ... Reason: Unknown.
→ EngineDeadError
```

Same log, during AITER's kernel build:

```
[aiter] Current hipcc not support: -mllvm -amdgpu-coerce-illegal-types=1, skip it.
clang: Unknown command line argument '-amdgpu-coerce-illegal-types=1'
```

AITER asks for the flag that handles **sub-byte "illegal" types** — exactly what Kimi's INT4
packed weights are — the image's bundled clang rejects it, and AITER **silently builds the
kernels without it**. With `VLLM_ROCM_USE_AITER=0` the same model serves and completes the
full matrix with **zero** memory faults.

Consistent with the rest of the set: the other four models are bf16 or fp8 (byte-or-larger
types) and never trip it. Kimi is the only INT4 checkpoint and the only one that faults.
Worth filing against AITER/the image.

## Reproducing

```bash
scripts/bench_vllm_all.sh                                   # all ready models
MODELS="google/gemma-4-12B-it:1:--dtype bfloat16" scripts/bench_vllm_all.sh
EXTRA_ENV="VLLM_ROCM_USE_AITER=1" ...                       # GLM
EXTRA_ENV="VLLM_ROCM_USE_AITER=0" \
  SERVE_EXTRA_ARGS=--trust-remote-code \
  BENCH_EXTRA_ARGS=--trust-remote-code ...                  # Kimi
```

Kimi needs `--trust-remote-code` on **both** serve and bench (`vllm bench serve` builds its
own tokenizer client-side), and runs vendor Python from the checkpoint — unlike the other
four, which load through vLLM's own implementations.

Load times (cold): 12B 240 s · 26B 210 s · 31B 260 s · GLM **3220 s** (AITER JIT) · Kimi 310 s.

## Reproducibility

gemma-4-12B ctxsweep re-measured ~2 h later through a different client path (separate
container over HTTP instead of `docker exec`): TPOT 6.830 vs 6.800 ms @1k, 7.560 vs 7.570
@4k — under 0.5% drift.

## Method notes

- `vllm bench serve --dataset-name random`, `--random-output-len 128`; ctxsweep
  `--num-prompts 3 --max-concurrency 1`; general `--num-prompts 8×conc` (cap 256).
- Every run holds a `gpulease -n <TP>` for the whole serve+bench, so a concurrent agent
  cannot land on the same cards. Lease log confirms no run was contended.
- A discarded warm-up point precedes the reported rows.
