# vLLM baseline — Gemma-4-26B-A4B-it (MoE) on RTX PRO 6000 Blackwell (sm_120), single-user — campaign B1

Measured **2026-07-18**, single-user (batch 1, `--max-concurrency 1`), TP-1, one **NVIDIA RTX PRO 6000
Blackwell Server Edition** (sm_120 / cc 12.0, 188 SMs, 97887 MiB), chunked-prefill budget 8192,
128 output tok/req, 3 prompts/point, random dataset, served endpoint, CUDA graphs ON. These are the
**faithful vLLM numbers plow must beat** for campaign B1. Same methodology, harness, and box as the
12B and 31B runs (`gemma4-12b-vllm-sm120.md`, `gemma4-31b-vllm-sm120.md`) — this file completes the
B1 baseline set.

- **Model:** `google/gemma-4-26B-A4B-it`, revision `01e5b3ee840d3a9e0b0b493c593e85398a30ef75`,
  arch **`Gemma4ForConditionalGeneration`** (`model_type: gemma4`). **MoE: 128 experts, top-8,
  ~4B active params**; 30 layers, hidden 2816, head_dim 256 / global_head_dim 512, 16 Q / 8 KV heads,
  sliding_window 1024, final_logit_softcapping 30.0. Model max ctx 262144; served at
  `--max-model-len 132096` (128k + 128 out + slack). TP-1.
- **Exact versions:** vLLM **0.25.1** (PyPI, `/workspace/venvs/vllm`); torch **2.11.0+cu130**
  (CUDA **13.0**); driver **580.82.07**; host CUDA toolkit **13.0** (V13.0.88).
- **Attention backend: `TRITON_ATTN` in all three configs** (forced — Gemma4 heterogeneous head
  dims, FA4 unavailable; same verbatim log line as the 12B/31B runs).
- CUDA graphs **ON** (`cudagraph_mode FULL_AND_PIECEWISE`; capture 11–16 s, 0.45–0.47 GiB).
- Data: `perf-data/gemma4-26b-a4b-vllm-sm120.json`. Raw logs: `perf-data/vllm_sm120_logs_26b/`.

**No OOM, no failed points: all 3 configs × 7 contexts (1k…128k) completed.**

## MoE kernel paths (the model-specific finding)

| config | MoE path (verbatim from serve logs) |
|--------|-------------------------------------|
| bf16   | `Using FlashInfer CUTLASS Unquantized MoE backend` (out of FlashInfer TRTLLM / FlashInfer CUTLASS / TRITON / BATCHED_TRITON); `trtllm::fused_moe` gemm1/gemm2 autotuned at warmup (21 profiles each) — first startup took **550 s**, mostly this autotune |
| fp8, fp8kv | `Using TRITON Fp8 MoE backend` — **with the DEFAULT (untuned) kernel config**: `Using default MoE config. Performance might be sub-optimal! Config file not found at .../E=128,N=704,device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition,dtype=fp8_w8a8.json`. Dense linears: `CutlassFP8ScaledMMLinearKernel` |

**fp8 quantization of the 128-expert MoE works on sm_120** — coherent sanity output, clean sweep —
but vLLM ships no fp8 MoE tuning file for this GPU, so the Triton fp8 expert kernels run untuned.
That is measurable headroom baked into the fp8/fp8kv numbers below.

## Decode — TPOT ms/token (HEADLINE, lower is better)

| ctx    | bf16   | fp8    | fp8kv |
|--------|--------|--------|-------|
| 1024   | 7.610  | 5.760  | 5.920 |
| 4096   | 7.900  | 6.080  | 6.190 |
| 16384  | 8.640  | 6.820  | 6.620 |
| 32768  | 9.570  | 7.740  | 7.280 |
| 65536  | 10.330 | 8.630  | 7.520 |
| 98304  | 11.340 | 9.540  | 7.940 |
| 131072 | 12.340 | 10.480 | 8.460 |

### Δ vs bf16 (decode TPOT, negative = faster)

| ctx    | fp8     | fp8kv   |
|--------|---------|---------|
| 1024   | −24.3 % | −22.2 % |
| 4096   | −23.0 % | −21.6 % |
| 16384  | −21.1 % | −23.4 % |
| 32768  | −19.1 % | −23.9 % |
| 65536  | −16.5 % | −27.2 % |
| 98304  | −15.9 % | −30.0 % |
| 131072 | −15.1 % | −31.4 % |

The MoE dilutes fp8's decode win to ~−20 % (vs ~−40 % on the dense 31B): only the ~4B active params
stream per token, so weight quantization saves proportionally less bandwidth. **fp8kv grows to
−31 % at 128k** as the (halved) KV stream dominates. Unlike the 31B, there is **no fp8kv 128k
anomaly** — TPOT scales smoothly through 128k (8 KV heads × 30 layers keeps the fp8-KV TRITON
decode kernel in its fast regime).

## Decode — output throughput tok/s (higher is better)

| ctx    | bf16   | fp8    | fp8kv  |
|--------|--------|--------|--------|
| 1024   | 131.41 | 173.61 | 168.92 |
| 4096   | 126.58 | 164.47 | 161.55 |
| 16384  | 115.74 | 146.63 | 151.06 |
| 32768  | 104.49 | 129.20 | 137.36 |
| 65536  | 96.81  | 115.87 | 132.98 |
| 98304  | 88.18  | 104.82 | 125.94 |
| 131072 | 81.04  | 95.42  | 118.20 |

Fastest decoder of the three B1 models despite the 48 GiB checkpoint — ~4B active params.

## Prefill — TTFT ms (recorded; TTFT = prefill + first decode token)

| ctx    | bf16    | fp8     | fp8kv   |
|--------|---------|---------|---------|
| 1024   | 75.34   | 87.85   | 68.37   |
| 4096   | 169.40  | 151.77  | 134.13  |
| 16384  | 799.21  | 710.24  | 525.82  |
| 32768  | 1543.51 | 1464.83 | 938.53  |
| 65536  | 4688.66 | 4650.84 | 2646.47 |
| 98304  | 6979.84 | 7131.04 | 3895.62 |
| 131072 | 9293.49 | 9623.06 | 5133.34 |

**fp8 does not improve TTFT over bf16 here** (equal to slightly worse across the ladder, including
9.62 s vs 9.29 s at 128k): fp8 prefill runs the *untuned* Triton fp8 MoE while bf16 prefill rides the
*autotuned* FlashInfer CUTLASS MoE. **fp8kv is the only config that cuts long-ctx TTFT: 5.13 s at
128k (−45 % vs bf16, −47 % vs fp8).**

## Prefill — tok/s (DERIVED, convention note)

`prefill_tok_s = input_len / (Mean_TTFT_ms/1000)`. **Not a true prefill throughput** — TTFT includes
the first decode token + scheduling, so it under-reports; kept for consistency with every other
served vLLM file in `perf-data/`.

| ctx    | bf16    | fp8     | fp8kv   |
|--------|---------|---------|---------|
| 1024   | 13591.7 | 11656.2 | 14977.3 |
| 4096   | 24179.5 | 26988.2 | 30537.5 |
| 16384  | 20500.2 | 23068.3 | 31159.0 |
| 32768  | 21229.5 | 22369.8 | 34914.2 |
| 65536  | 13977.6 | 14091.2 | 24763.6 |
| 98304  | 14084.0 | 13785.4 | 25234.5 |
| 131072 | 14103.6 | 13620.6 | 25533.5 |

## VRAM / KV headroom per config (util 0.90, constant across configs)

| config | weights (ckpt) | available KV | KV tokens | max-conc @132k | VRAM used | max ctx reached |
|--------|---------------|--------------|-----------|----------------|-----------|-----------------|
| bf16   | 48.07 GiB     | 32.42 GiB    | 1,000,429 | 7.57×          | 89389 / 97887 MiB | 131072 (swept) |
| fp8    | ~26 GiB       | 54.59 GiB    | 1,684,812 | 12.75×         | 89395 / 97887 MiB | 131072 (swept) |
| fp8kv  | ~26 GiB       | 54.59 GiB    | 3,369,625 | 25.51×         | 89395 / 97887 MiB | 131072 (swept) |

Only 30 layers × 8 KV heads means cheap KV: even bf16 holds 1M tokens. fp8kv's 3.37M-token pool
(25.51× a 132k request) is the largest concurrency headroom of any B1 config.

## Correctness gate (per config, greedy chat)

| config | generated text |
|--------|----------------|
| bf16  | `The capital of France is **Paris**.` |
| fp8   | `The capital of France is **Paris**.` |
| fp8kv | `The capital of France is **Paris**.` |

All three coherent — the sm_120 MoE routing and the fp8-quantized experts both survive the
`final_logit_softcapping: 30.0` + mixed sliding/full attention layout. (This model bolds "Paris";
the dense models did not.)

## Caveats

- **`prefill_tok_s` is a derived proxy, not measured** (see convention note above).
- **fp8/fp8kv MoE kernels are untuned** on this GPU (default Triton config; vLLM's own warning) —
  these baselines carry that slack; a tuned vLLM would be somewhat faster.
- **ctx=1024 TTFT** carries first-shape warm-up (torch.compile specialization / cudagraph capture);
  the chat warm-up request precedes each sweep but the 1024 point still sees some of it. Decode TPOT
  is unaffected.
- **vLLM is not bit-exact** (torch.compile inductor + TRITON_ATTN + cudagraphs).
- Single run per point, 3 prompts each — no repeat-run error bars.
- fp8kv logs `Checkpoint does not provide a q scaling factor. Setting it to k_scale` — harmless here
  (TRITON_ATTN, not an FP8 flash-attn/flashinfer backend).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set for all configs.
- The fp8kv config ran in a later lease than bf16/fp8 (the original driver was killed while queued
  behind other agents; the config was re-launched with identical parameters).

## Reproduce

```
export GPU_LEASE_TIMEOUT=7200
gpulease b1-26b env \
  QUANT=bf16 KVFP8=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MODEL_DIR=/workspace/models/gemma-4-26B-A4B-it \
  OUTDIR=/root/plow/perf-data/vllm_sm120_logs_26b \
  PORT=8126 CTXS=1024,4096,16384,32768,65536,98304,131072 MAXLEN=132096 GPU_UTIL=0.90 \
  HEALTH_TIMEOUT=1800 \
  bash <scratch copy of perf-data/bench_vllm_cuda.sh with line-58 PATH fixed>
```

`QUANT=fp8 KVFP8=0` for fp8; `QUANT=fp8 KVFP8=1` for fp8kv. Same CTXS/MAXLEN/GPU_UTIL for all three.
Venv: `/workspace/venvs/vllm` (`vllm==0.25.1`). The only harness edit is line 58's PATH — see the
JSON `notes`.

## Bottom line (what plow must beat)

1. **Decode TPOT floor at batch 1, TP1, sm_120:** bf16 7.61→12.34 ms (1k→128k); fp8 5.76→10.48 ms;
   **fp8kv 5.92→8.46 ms** — fp8kv is the config to beat everywhere ≥16k.
2. **Prefill/TTFT floor:** fp8kv at 128k is 5.13 s (bf16 9.29 s, fp8 9.62 s). fp8 weight
   quantization does not help prefill at all on this model.
3. vLLM's sm_120 MoE story: bf16 experts ride autotuned FlashInfer CUTLASS; **fp8 experts fall back
   to an untuned default Triton config** (no tuning file for this GPU) — the fp8/fp8kv rows carry
   that known slack.
