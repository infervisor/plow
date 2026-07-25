# vLLM baseline — Gemma-4-12B-it on RTX PRO 6000 Blackwell (sm_120), single-user — campaign B1

Measured **2026-07-18**, single-user (batch 1, `--max-concurrency 1`), TP-1, one **NVIDIA RTX PRO 6000
Blackwell Server Edition** (sm_120 / cc 12.0, 188 SMs, 97887 MiB), chunked-prefill budget 8192,
128 output tok/req, 3 prompts/point, random dataset, served endpoint, CUDA graphs ON. These are the
**faithful vLLM numbers plow must beat** for campaign B1.

Methodology deliberately matches the repo's `perf-data/bench_vllm_cuda.sh` harness (the CUDA/venv
counterpart of the ROCm `bench_vllm_docker.sh`) so the numbers line up with the committed baselines.

- **Model:** `google/gemma-4-12B-it`, revision `12ace6d648d72bd41519e140f1185f34d38c7e3d`,
  arch **`Gemma4UnifiedForConditionalGeneration`** (`model_type: gemma4_unified`). Model max ctx
  262144; served at `--max-model-len 132096` (128k + 128 out + slack). TP-1.
- **Exact versions:**
  - vLLM **0.25.1** (PyPI, `/workspace/venvs/vllm`)
  - torch **2.11.0+cu130** (CUDA **13.0**)
  - driver **580.82.07**; host CUDA toolkit **13.0** (V13.0.88)
- **Attention backend: `TRITON_ATTN` in all three configs** — forced by vLLM because Gemma4 has
  heterogeneous head dims (`head_dim=256`, `global_head_dim=512`) so FA4 is unavailable. Verbatim log:
  `Gemma4 model has heterogeneous head dimensions ... FA4 not available, forcing TRITON_ATTN backend.`
- CUDA graphs **ON** (`cudagraph_mode FULL_AND_PIECEWISE`; `Graph capturing finished ... 0.62–0.65 GiB`
  per config). Not `--enforce-eager`.
- Data: `perf-data/gemma4-12b-vllm-sm120.json`. Raw logs: `perf-data/vllm_sm120_logs_12b/`.

## Contrast with the 5090 run: no flashinfer workaround needed here

The RTX 5090 run of this model (`gemma4-12b-vllm-cuda-5090.md`) **required `VLLM_USE_FLASHINFER_SAMPLER=0`**
because that host had CUDA **12.8** (< 12.9), so flashinfer could not emit `compute_120f` and every config
died at sampler warm-up. **This box has nvcc 13.0**, the flashinfer sampler JIT compiles cleanly, and vLLM
**serves out of the box** on sm_120 — the workaround is not applied here. (The one harness edit was the
opposite direction: line 58's hardcoded `cuda-12.8` PATH does not exist on this box and was replaced with
`PATH=/workspace/venvs/vllm/bin:/usr/local/cuda/bin:$PATH` so flashinfer's `ninja` JIT is found.)

## Decode — TPOT ms/token (HEADLINE, lower is better)

| ctx    | bf16   | fp8    | fp8kv  |
|--------|--------|--------|--------|
| 1024   | 19.780 | 12.460 | 12.820 |
| 4096   | 20.250 | 12.970 | 13.090 |
| 16384  | 21.660 | 14.270 | 13.950 |
| 32768  | 23.350 | 15.980 | 15.050 |
| 65536  | 24.760 | 17.470 | 15.780 |
| 98304  | 26.510 | 18.910 | 16.570 |
| 131072 | 28.260 | 20.710 | 17.300 |

### Δ vs bf16 (decode TPOT, negative = faster)

| ctx    | fp8     | fp8kv   |
|--------|---------|---------|
| 1024   | −37.0 % | −35.2 % |
| 4096   | −35.9 % | −35.4 % |
| 16384  | −34.1 % | −35.6 % |
| 32768  | −31.6 % | −35.5 % |
| 65536  | −29.4 % | −36.3 % |
| 98304  | −28.7 % | −37.5 % |
| 131072 | −26.7 % | −38.8 % |

fp8 is a flat ~−30 % decode win. **fp8kv holds its ~−36 % lead as context grows** (fp8's advantage
decays with ctx because its bf16 KV cache gets heavier to stream; fp8kv's half-size KV does not).

## Decode — output throughput tok/s (higher is better)

| ctx    | bf16  | fp8   | fp8kv |
|--------|-------|-------|-------|
| 1024   | 50.56 | 80.26 | 78.00 |
| 4096   | 49.38 | 77.10 | 76.39 |
| 16384  | 46.17 | 70.08 | 71.68 |
| 32768  | 42.83 | 62.58 | 66.45 |
| 65536  | 40.39 | 57.24 | 63.37 |
| 98304  | 37.72 | 52.88 | 60.35 |
| 131072 | 35.39 | 48.29 | 57.80 |

## Prefill — TTFT ms (recorded; TTFT = prefill + first decode token)

| ctx    | bf16     | fp8      | fp8kv    |
|--------|----------|----------|----------|
| 1024   | 117.46   | 94.61    | 80.12    |
| 4096   | 323.48   | 244.71   | 196.38   |
| 16384  | 1502.26  | 1220.76  | 868.07   |
| 32768  | 2815.15  | 2438.73  | 1536.75  |
| 65536  | 8469.73  | 7663.76  | 4316.36  |
| 98304  | 12392.77 | 11645.18 | 6200.49  |
| 131072 | 16271.32 | 15520.48 | 8097.43  |

**fp8kv's largest single win is long-ctx TTFT**: at 128k it is 8097 ms vs 16271 ms bf16 (−50 %) and
15520 ms fp8 (−48 %) — the fp8 KV path also accelerates attention over the growing cache, which
dominates prefill at 128k.

## Prefill — tok/s (DERIVED, convention note)

`prefill_tok_s = input_len / (Mean_TTFT_ms/1000)`. **This is not a true prefill throughput.**
`vllm bench serve` reports no standalone prefill metric; TTFT includes the first decode token +
scheduling, so this denominator is inflated and the figure **under-reports** real prefill tok/s. It is
kept only because every other served vLLM file in `perf-data/` uses the identical derivation.

| ctx    | bf16    | fp8     | fp8kv   |
|--------|---------|---------|---------|
| 1024   | 8717.9  | 10823.4 | 12780.8 |
| 4096   | 12662.3 | 16738.2 | 20857.5 |
| 16384  | 10906.2 | 13421.1 | 18874.1 |
| 32768  | 11639.9 | 13436.5 | 21322.9 |
| 65536  | 7737.7  | 8551.4  | 15183.2 |
| 98304  | 7932.4  | 8441.6  | 15854.2 |
| 131072 | 8055.4  | 8445.1  | 16186.9 |

## VRAM / KV headroom per config (util 0.90, constant across configs)

| config | weights (ckpt) | available KV | KV tokens | max-conc @132k | VRAM used | max ctx reached |
|--------|---------------|--------------|-----------|----------------|-----------|-----------------|
| bf16   | 22.28 GiB     | 61.05 GiB    | 1,668,482 | 12.63×         | 93125 / 97887 MiB | 131072 (swept) |
| fp8    | ~12.1 GiB     | 71.21 GiB    | 1,946,180 | 14.73×         | 93179 / 97887 MiB | 131072 (swept) |
| fp8kv  | ~12.1 GiB     | 71.21 GiB    | 3,892,361 | 29.47×         | 93179 / 97887 MiB | 131072 (swept) |

Unlike the 32 GB 5090 (where bf16 OOMed past 16k and `--gpu-memory-utilization` had to be hand-tuned per
config), **this 96 GB card runs all three configs at a constant util 0.90 to full 128k with no OOM** —
a clean, controlled A/B. bf16's KV pool alone holds 1.67M tokens (12.63× the 132k sequence). fp8kv
holds exactly 2× fp8's tokens (fp8 KV halves bytes/token).

## Correctness gate (per config, greedy chat)

| config | generated text |
|--------|----------------|
| bf16  | `The capital of France is Paris.` |
| fp8   | `The capital of France is Paris.` |
| fp8kv | `The capital of France is Paris.` |

All three coherent — matters given this model's `final_logit_softcapping: 30.0` and the mixed
sliding/full attention layout; fp8 and fp8kv both survive it.

## Caveats

- **`prefill_tok_s` is a derived proxy, not measured** (see convention note above).
- **ctx=1024 TTFT** carries first-shape warm-up (torch.compile specialization / cudagraph capture for
  the smallest bench shape). A chat warm-up request precedes each sweep, but the 1024 point still sees
  some of it. Decode TPOT is unaffected.
- **vLLM is not bit-exact** (torch.compile inductor + TRITON_ATTN + cudagraphs).
- Single run per point, 3 prompts each — no repeat-run error bars.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set for all configs (defensive; the 96 GB card
  has ample headroom regardless).

## Reproduce

```
export GPU_LEASE_TIMEOUT=7200
gpulease b1-12b env \
  QUANT=bf16 KVFP8=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL_DIR=/workspace/models/gemma-4-12B-it \
  OUTDIR=/root/plow/perf-data/vllm_sm120_logs_12b \
  PORT=8123 CTXS=1024,4096,16384,32768,65536,98304,131072 MAXLEN=132096 GPU_UTIL=0.90 \
  bash <scratch copy of perf-data/bench_vllm_cuda.sh with line-58 PATH fixed>
```

`QUANT=fp8 KVFP8=0` for fp8; `QUANT=fp8 KVFP8=1` for fp8kv. Same CTXS/MAXLEN/GPU_UTIL for all three.
Venv: `/workspace/venvs/vllm` (`vllm==0.25.1`). The only harness edit is line 58's PATH — see the
JSON `notes`.

## Bottom line (what plow must beat)

1. **Decode TPOT floor at batch 1, TP1, sm_120:** bf16 19.8→28.3 ms (1k→128k); fp8 12.5→20.7 ms;
   **fp8kv 12.8→17.3 ms** — fp8kv is the config to beat at long context.
2. **Prefill/TTFT floor:** fp8kv at 128k is 8.1 s (bf16 16.3 s, fp8 15.5 s).
3. All configs serve out of the box on sm_120 here (nvcc 13.0); backend is **TRITON_ATTN** throughout.
