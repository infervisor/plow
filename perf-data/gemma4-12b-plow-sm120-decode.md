# Gemma-4-12B bf16 DECODE — plow vs vLLM on sm_120 (RTX PRO 6000 Blackwell)

**Campaign P1-sm120-decode.** First committed plow-vs-vLLM decode table for
Gemma-4-12B on the RTX PRO 6000 Blackwell (96 GB, sm_120, 188 SMs). Single
sequence, batch 1, bf16. Measurement only — no source edits; the measured code
is committed HEAD `32cc434` (rtx), built from a clean `git worktree` at that
commit.

## Setup

- **Model:** google/gemma-4-12B-it, rev `12ace6d6`, at `/workspace/models/gemma-4-12B-it`.
  48 layers (8 full / 40 sliding), hidden 3840, 16 heads, kv 8 slide / 1 full,
  head_dim 256 slide / 512 full, sliding window 1024, tied embeddings.
- **plow build:** interp object `runtime/nvidia/interp_sm120.cu` compiled
  `-arch=sm_120a -DPLOW_NV_GEMMA=1 -DPLOW_NV_FA_GF=2` (150 regs, 0 spill, 1 block/SM);
  harness `runtime/tests/qwen3_sm120_chat.cu` (model-agnostic); packet
  `plowc gemma4 <dir> 132096 out.pkt 188`; global-queue scheduler (default, `PLOW_NV_SCHED=1`).
  Build ran OFF-lease from a clean worktree at `32cc434`.
- **Methodology (repo convention, commit `18760a5`):** consume an `input_len`-token
  prompt one token at a time to build the KV cache (no sm_120 prefill kernels yet —
  O(n) launches), then decode 128 tokens; discard the first 16 (warmup), report
  mean/median/sd over the remaining **112 timed steps**. Timed window spans
  ctx = input_len+16 .. input_len+127 — the KV grows past `input_len` exactly as
  vLLM's `output_len=128` decode does.
- **vLLM baseline:** `gemma4-12b-vllm-sm120.json` config `bf16` — vLLM 0.25.1,
  TRITON_ATTN (forced: Gemma4 heterogeneous head dims 256/512 leave FlashAttention
  off the fast path), cudagraphs ON, same box/model/revision, `output_len` 128.
  This is the like-for-like bf16 decode comparison.
- **Correctness:** Phase-0 (`32cc434`) gated this path token-identical vs HF greedy
  (64/64 on 4 prompts incl. >2k ctx). This run also checks per point that the device
  ARGMAX equals a host scan of the same bf16 logits row: **AGREE at every ctx**.

## Results — decode TPOT (ms/token, lower is better)

| ctx | plow bf16 TPOT | sd (ms) | sd (%) | plow tok/s | vLLM bf16 TPOT | vLLM tok/s | **plow vs vLLM** |
|------:|---------------:|--------:|-------:|-----------:|---------------:|-----------:|-----------------:|
|   1024 |   **18.481** | 0.011 | 0.06% | 54.11 | 19.780 | 50.56 | **-6.6%** |
|   4096 |   **18.543** | 0.010 | 0.05% | 53.93 | 20.250 | 49.38 | **-8.4%** |
|  16384 |   **19.066** | 0.011 | 0.06% | 52.45 | 21.660 | 46.17 | **-12.0%** |
|  32768 |   **19.756** | 0.013 | 0.06% | 50.62 | 23.350 | 42.83 | **-15.4%** |
|  65536 |   **21.116** | 0.016 | 0.08% | 47.36 | 24.760 | 40.39 | **-14.7%** |
|  98304 |   **22.547** | 0.017 | 0.08% | 44.35 | 26.510 | 37.72 | **-14.9%** |
| 131072 |   **23.975** | 0.016 | 0.06% | 41.71 | 28.260 | 35.39 | **-15.2%** |

plow bf16 decode **beats vLLM bf16 decode at every context**, from -6.6% at 1k
widening to -15.2% at 128k. The gap widens with ctx because plow's flash-decode
over the growing full-layer KV outruns vLLM's TRITON_ATTN fallback (no sm_120
head_dim 256/512 fast path exists in FlashAttention, so vLLM is forced to Triton).

`kernel_ms` (launch+sync — the part vLLM's cudagraph replay also pays) is within
0.045 ms of `tpot_ms` at every point; the plow-harness host prologue (kv-row patch
+ 3 scalars + counter zero) adds only ~0.04 ms/step and a graph-captured serving
path would not pay it. plow leads even if the full host prologue is charged.

## Scaling sanity — linear in FULL-layer KV only

Only the **8 full-attention layers** (1 kv head, head_dim 512) grow their KV with
ctx; the **40 sliding layers** are ring-capped at 16384 rows (window 1024). TPOT
growth is therefore expected to be linear in full-layer KV bytes. It is:

| ctx step | Δctx | Δtpot (ms) | ms per ctx-token |
|---|---:|---:|---:|
| 16384 → 32768 | 16384 | 0.690 | 4.21e-5 |
| 32768 → 65536 | 32768 | 1.360 | 4.15e-5 |
| 65536 → 131072 | 65536 | 2.859 | 4.36e-5 |

The per-ctx-token slope is constant (~4.3e-5 ms) across every doubling — **linear,
no super-linear term. No scaling bug**; the sliding-window ring is doing its job
(if the 40 sliding layers were also growing with ctx, the slope would be ~6x
steeper and rising). Overall TPOT rises just 29.7% (18.48 → 23.97 ms) across a
128x context increase.

## Notes

- **No failed points:** all 7 contexts (1k..128k) completed at bf16. Footprint
  KV 7.02 GiB + weights 22.2 GiB + activations 1.63 GiB ≈ 31 GiB — huge headroom on
  the 96 GB card (matches rtx-06 math).
- Run-to-run noise is tiny: sd ≤ 0.08% of the mean everywhere (112 timed steps, GQ
  scheduler); median tracks the mean to <0.002 ms.
- **Scope:** this is batch-1 single-user *decode latency*. Prefill/TTFT is not
  measured here — sm_120 has no prefill kernels yet (rtx-06 G5), so the harness
  consumes the prompt via O(n) decode launches. Throughput/concurrency is a
  separate campaign (B2). fp8 decode (G7) is future work; this is bf16-vs-bf16.
- Every GPU command ran under `gpulease p-sweep …`, one lease per ctx point,
  released between points to share the GPU with concurrent agents.
