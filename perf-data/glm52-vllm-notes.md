# GLM-5.2-FP8 — vLLM decode baseline (gfx950 / MI350X, 8-GPU node)

Baseline that plow's GLM-5.2 bring-up will be compared against. Measurement only.

## STEP 0 — SUPPORT VERDICT: **SUPPORTED** (with a caveat)

vLLM **CAN** serve `GlmMoeDsaForCausalLM` / `model_type=glm_moe_dsa` (the DeepSeek-V3.2-DSA-class
arch: MLA + sparse-attention indexer + fine-grained block-fp8 MoE) on this gfx950 box — but only
on the newer image and only with AITER forced on.

| image | vLLM version | `GlmMoeDsaForCausalLM` in registry? |
|---|---|---|
| `rocm/vllm:latest` | 0.11.2.dev673 | **NO** |
| `vllm/vllm-openai-rocm:latest` | **0.25.1+rocm723** | **YES** |

- On 0.25.1 the arch resolves to a real implementation
  `vllm.model_executor.models.deepseek_v2.GlmMoeDsaForCausalLM` (reuses the DeepSeek-V3.2 MLA/DSA/MoE
  code path), not a stub.
- **Hard requirement: `VLLM_ROCM_USE_AITER=1`.** Without it the engine crashes at profile_run with
  `RuntimeError: Sparse attention indexer ROCm path is only supported on AITER. Please enable aiter
  with VLLM_ROCM_USE_AITER=1` (vllm/model_executor/layers/sparse_attn_indexer.py:823, forward_hip).
  The DSA indexer has no non-AITER ROCm fallback. With AITER on, it serves cleanly.
- => No plow "capability win" on this box: vLLM runs GLM-5.2-FP8. HF transformers is NOT the only
  baseline; vLLM is a valid, competitive baseline.

## Run configuration (both TP4 and TP8)
- image `vllm/vllm-openai-rocm:latest`, vLLM **0.25.1+rocm723**, torch 2.11.0 / HIP 7.2
- model `/models/GLM-5.2-FP8` (native block-fp8 checkpoint, 756 GB, 141 shards)
- **quantization = fp8, auto-detected** from checkpoint `quantization_config` (block e4m3,
  weight_block_size [128,128], dynamic acts). NOT forced via `--quantization`.
- `--dtype bfloat16` (unquantized/compute dtype), `kv_cache_dtype=auto` (bf16 MLA latent KV)
- `VLLM_ROCM_USE_AITER=1`, `--max-num-batched-tokens 8192`, `--max-model-len 133120`
- cudagraph FULL_AND_PIECEWISE (default), torch.compile/inductor ON
- **batch 1, `--max-concurrency 1`, output_len 128, num_prompts 3 per point** (single-user decode)
- backends (both TP): decode `ROCM_AITER_MLA_SPARSE`, prefill `ROCM_AITER_FA` MLA,
  MoE `AITER Fp8` (per_1x128 block-scale fused_moe, gfx950)
- weight load ~1–2 min (fast NVMe); engine init incl. compile+graph capture: TP4 ~219 s, TP8 ~330 s
- KV cache headroom: TP4 = 926,128 tokens (6.96x @128k); TP8 = 1,917,744 tokens (14.41x @128k)

## Sanity gate (both TP): PASSED
`"The capital of France is"` -> `" Paris. Distance from Paris to Lyon is 391 km"` (coherent, both TP4/TP8).
(GLM-5.2 is a reasoning model; the chat endpoint emits reasoning steps before answering — expected.)

## TP4 decode sweep (GPUs 0-3)
| ctx | TTFT ms | prefill tok/s | TPOT ms/tok | decode tok/s |
|---:|---:|---:|---:|---:|
| 1024   | 992.6  | 1031.6  | 19.18 | 52.14 |
| 4096   | 402.1  | 10185.8 | 20.16 | 49.60 |
| 8192   | 482.2  | 16989.5 | 20.22 | 49.46 |
| 16384  | 909.2  | 18020.4 | 20.30 | 49.26 |
| 32768  | 1854.5 | 17669.8 | 20.43 | 48.95 |
| 65536  | 3882.6 | 16879.4 | 20.78 | 48.12 |
| 131072 | 9729.3 | 13471.9 | 21.40 | 46.73 |

Decode is remarkably flat: TPOT 19.2 -> 21.4 ms across 1k -> 128k (+11.6% only). MLA's tiny
latent KV + DSA's top-2048 attention cap keep decode nearly context-independent. First point (1024)
TTFT is inflated by cold-start; steady-state prefill peaks ~18 Ktok/s around 8k–16k.

## TP8 decode sweep (all 8 GPUs)
| ctx | TTFT ms | prefill tok/s | TPOT ms/tok | decode tok/s |
|---:|---:|---:|---:|---:|
| 1024   | 4603.4* | 222.4*  | 18.52 | 54.00 |
| 4096   | 296.5   | 13816.4 | 19.10 | 52.36 |
| 8192   | 362.3   | 22608.6 | 19.16 | 52.19 |
| 16384  | 671.3   | 24407.1 | 19.28 | 51.87 |
| 32768  | 1361.1  | 24075.4 | 19.40 | 51.55 |
| 65536  | 2921.5  | 22432.6 | 19.68 | 50.81 |
| 131072 | 7921.7  | 16545.9 | 20.31 | 49.24 |

\* 1024 TTFT is a cold-start outlier (first request after graph capture); ignore for prefill trend.

TP8 vs TP4 decode: TP8 is ~6–8% faster per token (e.g. 128k: 20.31 vs 21.40 ms/tok; 8k: 19.16 vs
20.22). Modest — decode at batch 1 is latency/launch-bound, not bandwidth-bound (MLA KV is tiny, DSA
caps attention), so doubling TP mostly halves the per-GPU weight-read but adds all-reduce latency.
Prefill scales better with TP8 (peak ~24 Ktok/s vs ~18 Ktok/s at TP4). Decode stays flat across ctx
at both TP (TP8: 18.5 -> 20.3 ms, +9.7% over 1k->128k).

## Caveats
- `-mllvm -amdgpu-coerce-illegal-types=1 is not supported by hipcc` warning during AITER JIT (benign).
- aiter GEMM/MoE shapes report "not found tuned config ... using default config" — i.e. these are
  UNTUNED aiter kernels for GLM's shapes; a tuned config sweep would likely improve both TP configs.
  The baseline is therefore vLLM's out-of-the-box aiter default, not a tuned ceiling.
- aiter sampler lacks per-request generators -> falls back to PyTorch-native sampler (benign, temp=0).
- Not bit-exact (fp8 + MoE); coherence-gated only.
