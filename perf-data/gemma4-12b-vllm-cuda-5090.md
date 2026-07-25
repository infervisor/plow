# vLLM baseline — Gemma-4-12B-it on RTX 5090 (sm_120 / CUDA), single-user

Measured **2026-07-18**, single-user (batch 1, `--max-concurrency 1`), TP-1, one **NVIDIA RTX 5090**
(sm_120 / cc 12.0, 32607 MiB), chunked-prefill budget 8192, 128 output tok/req, 3 prompts/point,
random dataset, served endpoint, CUDA graphs ON. Reproduce with `perf-data/bench_vllm_cuda.sh`
(the CUDA/venv counterpart of `bench_vllm_docker.sh`).

This is the **CUDA counterpart** of the ROCm baselines in `vllm-fp8-baseline.md`. Methodology is
matched deliberately so the numbers are comparable to the committed AMD (gfx950/MI350X) data.

- **Model:** `google/gemma-4-12B-it`, revision `12ace6d648d72bd41519e140f1185f34d38c7e3d`, public/ungated.
  Arch is **`Gemma4UnifiedForConditionalGeneration`** (`model_type: gemma4_unified`) — note this is a
  *different* architecture from the 31B (`Gemma4ForConditionalGeneration` / `gemma4`) referenced in
  `vllm-fp8-baseline.md`. vLLM 0.25.1 registers it as `gemma4_unified`; it serves at TP-1.
- **Exact versions** (the existing baselines were burned by an unflagged version discrepancy):
  - vLLM **0.25.1** (PyPI, same major as the AMD `0.25.1+rocm723` — chosen for comparability)
  - torch **2.11.0+cu130** (CUDA **13.0**), `arch_list` includes `sm_120`
  - driver **580.65.06**; host CUDA toolkit **12.8** (V12.8.93)
- Data: `perf-data/gemma4-12b-vllm-cuda-5090.json`.

## HEADLINE FINDING #1 — vLLM does NOT run on sm_120 out of the box; one env var is required

Every config dies during **sampler warm-up** with a message that is actively misleading on a 5090:

```
RuntimeError: FlashInfer requires GPUs with sm75 or higher
  File "flashinfer/jit/core.py", line 108, in check_cuda_arch
```

Root cause (traced, not guessed): FlashInfer's `_normalize_cuda_arch()` needs **nvcc ≥ 12.9** to emit
`compute_120f` for SM 12.x. This host has CUDA **12.8**, so it raises `SM 12.x requires CUDA >= 12.9`;
the caller swallows that in a bare `except`, leaving `TARGET_CUDA_ARCHS` **empty** — and the
subsequent "is anything ≥ sm75?" check then fails. Confirmed directly:

```
nvcc>=12.9? False
Failed to get device capability: SM 12.x requires CUDA >= 12.9.
TARGET_CUDA_ARCHS: set()
```

**Workaround: `VLLM_USE_FLASHINFER_SAMPLER=0`.** This affects only the sampler JIT path — the
attention backend is unaffected (the crash occurs *after* attention init). With it set, all three
configs serve and generate coherently. The alternative fix is a CUDA ≥ 12.9 toolkit on the host.

## HEADLINE FINDING #2 — fp8 is a large decode win here (−31 % to −44 %), unlike on the small AMD models

Unlike Qwen-4B/Llama-8B on gfx950 (where vLLM fp8 was neutral-to-slower at batch 1), Gemma-4-12B is
firmly in the regime where fp8 pays off — consistent with the AMD Gemma-31B result (−10…−19 %), only
larger, because on a 32 GB card fp8 also buys back an enormous amount of KV headroom.

### Decode — TPOT ms/token (HEADLINE, lower is better)

| ctx   | bf16      | fp8       | fp8kv     |
|-------|-----------|-----------|-----------|
| 4096  | 17.990    | **12.110**| 12.250    |
| 8192  | 18.690    | 12.730    | **12.650**|
| 16384 | 23.410    | 13.190    | **12.880**|
| 32768 | **OOM**   | 14.740    | **13.850**|
| 65536 | **OOM**   | —         | **14.690**|

### Δ vs bf16 (decode TPOT, negative = faster)

| ctx   | fp8     | fp8kv   |
|-------|---------|---------|
| 4096  | −32.7 % | −31.9 % |
| 8192  | −31.9 % | −32.3 % |
| 16384 | −43.7 % | −45.0 % |

### Prefill — TTFT ms (recorded, not the target)

| ctx   | bf16     | fp8      | fp8kv    |
|-------|----------|----------|----------|
| 4096  | 554.68   | 353.64   | 316.42   |
| 8192  | 621.49   | 426.52   | 357.83   |
| 16384 | 2147.35  | 959.34   | 769.31   |
| 32768 | OOM      | 2655.13  | 1854.65  |
| 65536 | OOM      | —        | 5209.30  |

## HEADLINE FINDING #3 — bf16 has no comfortable operating point on a 32 GB card; fp8 unlocks 4× the context

Weights alone are **22.68 GiB of 31.36 GiB usable**. That leaves so little for KV + activations that
bf16 is *unstable*, and `--gpu-memory-utilization` has to be hand-tuned:

| gpu-util | bf16 outcome |
|---|---|
| 0.95 | **OOM at ctx 16384** — verbatim: `CUDA out of memory. Tried to allocate 480.00 MiB. GPU 0 has a total capacity of 31.36 GiB of which 20.00 MiB is free.` KV pool 5.84 GiB / 31,426 tok left nothing for the 8192-token prefill activations. (4k/8k did run: 18.12 / 18.77 ms.) |
| 0.92 | **stable, published column** — 17.99 / 18.69 / 23.41 ms, monotonic; 4k/8k reproduce the 0.95 run within 0.7 %. |
| 0.88 | runs, but **erratic / non-monotonic**: 4k 31.45, 8k 18.86, 16k 28.71 ms. KV pool too small (20,436 tok). Rejected. |

**ctx 32768 is unreachable in bf16**: the 0.92 KV pool is 27,182 tokens < the 32,896 required.

### VRAM headroom per config

| config | gpu-util | weights | KV pool | KV tokens | VRAM used | max ctx reached |
|---|---|---|---|---|---|---|
| bf16   | 0.92 | 22.68 GiB | 5.05 GiB  | 27,182  | 30806 / 32607 MiB | 16384 (32768 = OOM) |
| fp8    | 0.90 | 12.52 GiB | 14.6 GiB  | 144,697 | 30210 / 32607 MiB | 32768 (swept) |
| fp8kv  | 0.90 | 12.52 GiB | 14.6 GiB  | 502,058 | 30208 / 32607 MiB | 65536 (swept) |

fp8 frees **10.16 GiB** of weights, growing the KV pool 5.05 → 14.6 GiB (**27,182 → 144,697 tokens,
5.3×**). fp8kv holds **502,058 tokens** in the same 14.6 GiB — **3.47×** more than fp8.

Note the 3.47× is *not* fully explained: fp8 KV halves bytes/token, which alone predicts ~2×. The
extra factor is unexplained and most likely an artifact of how vLLM's hybrid KV manager reports a
single "tokens" figure for this model's mixed 40-sliding / 8-full layer groups. Treat the token
counts as vLLM's own accounting, not as a derived bytes/token result. **Either way, this is where
fp8/fp8kv unlock context bf16 simply cannot reach** — that part is directly observed (bf16 OOMs
past 16k; fp8 reaches 32k, fp8kv 65k).

## Correctness gate (not optional)

Per config, chat request "What is the capital of France?", greedy. All three **coherent**, identical:

| config | generated text |
|---|---|
| bf16  | `The capital of France is Paris.` |
| fp8   | `The capital of France is Paris.` |
| fp8kv | `The capital of France is Paris.` |

This matters here given the `final_logit_softcapping: 30.0` and the unusual 40-sliding / 8-full
attention mix — fp8 and fp8kv both survive it.

## Caveats

- `--gpu-memory-utilization` is **not constant across configs** (bf16 0.92, fp8/fp8kv 0.90) because
  bf16 has no stable point at 0.90+. The fp8-vs-bf16 gap (−33 %) is far larger than the bf16
  0.92-vs-0.95 spread (< 1 %), so the ranking is safe, but it is not a perfectly controlled A/B.
- ctx 1024 was not swept (the AMD baselines note it is warm-up-contaminated anyway); the first shape
  in each ladder still carries some warm-up in TTFT.
- Single run per point, 3 prompts each — no repeat-run error bars.

## Reproduce

```
export GPU_LEASE_TIMEOUT=3600
gpulease gemma-12b env \
  QUANT=bf16 KVFP8=0 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL_DIR=/workspace/models/gemma-4-12B-it \
  OUTDIR=/root/plow/perf-data/vllm_cuda_logs_gemma12b \
  PORT=8123 CTXS=4096,8192,16384 MAXLEN=16512 GPU_UTIL=0.92 \
  bash perf-data/bench_vllm_cuda.sh
```

`QUANT=fp8 KVFP8=0 CTXS=4096,8192,16384,32768 MAXLEN=32896 GPU_UTIL=0.90` for fp8;
`QUANT=fp8 KVFP8=1 CTXS=4096,8192,16384,32768,65536 MAXLEN=65664 GPU_UTIL=0.90` for fp8kv.
Venv: `/workspace/venvs/vllm` (`vllm==0.25.1`).

## Bottom line

1. **sm_120 needs `VLLM_USE_FLASHINFER_SAMPLER=0`** with CUDA 12.8 on the host, or nothing serves.
   The error message ("requires sm75 or higher") points in entirely the wrong direction.
2. **fp8 is the right default for this model on a 32 GB card** — 31–44 % faster decode *and* 2×–4×
   the reachable context. There is no regime here where bf16 wins.
3. **fp8kv is the long-context lever**: it matches fp8 at short ctx and pulls ahead as ctx grows
   (16k −2.4 %, 32k −6.0 % vs fp8), reaching 65k where bf16 OOMs past 16k. Same shape as the AMD
   finding, amplified by this card's tighter VRAM.
