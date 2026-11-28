# Gemma-4-26B-A4B-it comparison (1x H100, vLLM 0.28)

This file holds only qualified wins and their evidence: `campaign.py report` MATCHED +
EQUIVALENT, Infervisor total throughput above vLLM. Every matched row, wins and losses, is in
[comparison.csv](comparison.csv) (`campaign_source` `fin2-20261007-<workload>-<precision>`; the
round-1 rows `fin1-*` stay for history).

- Infervisor: plowrt `2cc23e9e` (gemma26b-next), packets built by `campaign.py build` at
  `73804e5e`: FP8 `2b8d3cf32064` from
  [`sm90a-h100-tp1-fp8.toml`](../../../../recipes/infervisor/gemma-4-26b-a4b/sm90a-h100-tp1-fp8.toml),
  BF16 `f33189e78e8e` from
  [`sm90a-h100-tp1-bf16.toml`](../../../../recipes/infervisor/gemma-4-26b-a4b/sm90a-h100-tp1-bf16.toml).
  Serve defaults come from the packet; server env `PLOW_LIBCUDA` and the cuBLAS 13.4 library path.
- Baseline: vLLM 0.28.0 (the round-1 runs), same client, BF16 KV for BF16 and
  `fp8_per_token_head` KV (TRITON_ATTN) for FP8, max-num-batched-tokens 8192, prefix caching on.
- 2 repeats per cell, one server per arm. Reproduce: `scripts/campaign/repro_gemma26b_h100.sh`.
- Quality: FP32-reference gate PASS twice per packet. FP8 kl_mean 0.191 / 0.192 vs vLLM 0.202,
  kl_p99 3.49 / 4.54 (vLLM 3.73). BF16 kl_mean 0.104 / 0.116 vs 0.1005, cont_frac 0.598 / 0.643
  (vLLM 0.638, limit 0.588), needle 1.0. Round 1's BF16 cont_frac misses were partly real: BF16
  scores on the cuBLASLt prefill attention route flipped FP32-certain tokens (chat-code pos 7,
  chat-long-15k pos 1, every capture); the recipe now keeps them in f32
  (`PLOW_PF_ATTN_GEMM_S32=1`). The rest of the run-to-run spread is c16 batch composition
  (decode rungs differ per batch size): the same packet's captures differ in ~20 of 46 cases.

| Cell | Precision | Total throughput | TTFT P99 | TPOT P99 |
|---|---|---:|---:|---:|
| 4096/128 c32 | FP8 | 1.32x | 0.55x | 0.79x |
| 4096/128 c128 | FP8 | 1.29x | 0.65x | 0.76x |
| 15000/128 c32 | FP8 | 1.88x | 0.49x | 0.52x |
| 15000/128 c128 | FP8 | 1.92x | 0.51x | 0.51x |
| agentic16k c32 | FP8 | 1.21x | 0.46x * | 0.76x |
| agentic16k c64 | FP8 | 1.12x | 0.77x | 0.83x |
| agentic16k c128 | FP8 | 3.50x | 0.20x * | 0.26x |
| 1024/128 c1 | FP8 | 1.02x | 1.50x * | 1.00x |
| agentic16k c32 | BF16 | 1.57x | 0.43x * | 0.49x * |
| agentic16k c64 | BF16 | 1.32x | 2.60x | 0.49x |
| agentic16k c128 | BF16 | 1.03x | 1.14x | 0.88x * |

`*` = repeat spread > 10% (FLAGGED; direction only). Not won (in the CSV): FP8 1024/128 c4 0.94x;
BF16 4096/128 c32 / c128 0.88x / 0.77x, 15000/128 c32 / c128 0.87x / 0.86x, 1024/128 c1 / c4
0.94x / 0.93x.

## Strict tables

### st4k-fp8 g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet 2b8d3cf32064) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.7 GiB | 75.2 GiB (1.05x) |
| Total throughput | 19,733 tok/s | 25,993 tok/s (1.32x) |
| Throughput / GPU | 19,733 tok/s/GPU | 25,993 tok/s/GPU (1.32x) |
| TTFT P99 | 5,596.3 ms * | 3,079.1 ms (0.55x) |
| TPOT P99 | 49.31 ms | 39.10 ms (0.79x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 6.7%; Total throughput 9.4% / 2.2%; Throughput / GPU 9.4% / 2.2%; TTFT P99 34.1% / 9.9%; TPOT P99 0.0% / 1.1%.

**FLAGGED: spread > 10%: Baseline TTFT P99 34.1%.**

### st4k-fp8 g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet 2b8d3cf32064) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.7 GiB | 77.6 GiB (1.08x) |
| Total throughput | 24,391 tok/s | 31,350 tok/s (1.29x) |
| Throughput / GPU | 24,391 tok/s/GPU | 31,350 tok/s/GPU (1.29x) |
| TTFT P99 | 18,546.0 ms | 12,105.8 ms (0.65x) |
| TPOT P99 | 168.49 ms | 127.70 ms (0.76x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 0.3% / 0.3%; Throughput / GPU 0.3% / 0.3%; TTFT P99 0.9% / 2.5%; TPOT P99 0.0% / 0.2%.

### st15k-fp8 g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 15000 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet 2b8d3cf32064) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.7 GiB | 66.8 GiB (0.93x) |
| Total throughput | 15,414 tok/s | 28,941 tok/s (1.88x) |
| Throughput / GPU | 15,414 tok/s/GPU | 28,941 tok/s/GPU (1.88x) |
| TTFT P99 | 27,922.6 ms | 13,680.4 ms (0.49x) |
| TPOT P99 | 237.02 ms | 122.08 ms (0.52x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 8.2%; Total throughput 0.0% / 0.3%; Throughput / GPU 0.0% / 0.3%; TTFT P99 0.4% / 2.1%; TPOT P99 0.1% / 0.2%.

### st15k-fp8 g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 15000 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet 2b8d3cf32064) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.7 GiB | 77.6 GiB (1.08x) |
| Total throughput | 16,124 tok/s | 30,907 tok/s (1.92x) |
| Throughput / GPU | 16,124 tok/s/GPU | 30,907 tok/s/GPU (1.92x) |
| TTFT P99 | 114,267.6 ms | 57,732.9 ms (0.51x) |
| TPOT P99 | 506.41 ms | 256.84 ms (0.51x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.0% / 0.1%; Throughput / GPU 0.0% / 0.1%; TTFT P99 0.1% / 0.7%; TPOT P99 0.0% / 1.6%.

### agentic-fp8 a32.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 32 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet 2b8d3cf32064) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.7 GiB | 77.6 GiB (1.08x) |
| Total throughput | 51,131 tok/s | 61,751 tok/s (1.21x) |
| Throughput / GPU | 51,131 tok/s/GPU | 61,751 tok/s/GPU (1.21x) |
| TTFT P99 | 3,129.5 ms * | 1,428.3 ms (0.46x) * |
| TPOT P99 | 56.32 ms | 43.07 ms (0.76x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.3%; Total throughput 2.1% / 3.2%; Throughput / GPU 2.1% / 3.2%; TTFT P99 40.4% / 18.5%; TPOT P99 0.7% / 0.7%.

**FLAGGED: spread > 10%: Baseline TTFT P99 40.4%; Infervisor TTFT P99 18.5%.**

### agentic-fp8 a64.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 64 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet 2b8d3cf32064) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.7 GiB | 77.7 GiB (1.08x) |
| Total throughput | 62,786 tok/s | 70,592 tok/s (1.12x) |
| Throughput / GPU | 62,786 tok/s/GPU | 70,592 tok/s/GPU (1.12x) |
| TTFT P99 | 3,023.9 ms | 2,319.1 ms (0.77x) |
| TPOT P99 | 93.05 ms | 77.66 ms (0.83x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.1% / 1.3%; Throughput / GPU 0.1% / 1.3%; TTFT P99 1.5% / 1.2%; TPOT P99 0.7% / 0.4%.

### agentic-fp8 a128.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 128 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet 2b8d3cf32064) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 77.8 GiB (1.08x) |
| Total throughput | 20,863 tok/s | 72,983 tok/s (3.50x) |
| Throughput / GPU | 20,863 tok/s/GPU | 72,983 tok/s/GPU (3.50x) |
| TTFT P99 | 49,942.4 ms | 9,976.4 ms (0.20x) * |
| TPOT P99 | 550.57 ms | 144.72 ms (0.26x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.2%; Total throughput 0.4% / 6.7%; Throughput / GPU 0.4% / 6.7%; TTFT P99 0.3% / 106.0%; TPOT P99 0.0% / 2.5%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 106.0%.**

### lat-fp8 g1

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 1024 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 16 prompts, request rate inf / concurrency 1 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet 2b8d3cf32064) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.7 GiB | 58.8 GiB (0.82x) |
| Total throughput | 1,546 tok/s | 1,583 tok/s (1.02x) |
| Throughput / GPU | 1,546 tok/s/GPU | 1,583 tok/s/GPU (1.02x) |
| TTFT P99 | 81.7 ms | 122.7 ms (1.50x) * |
| TPOT P99 | 5.45 ms | 5.42 ms (1.00x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.9% / 1.8%; Throughput / GPU 0.9% / 1.8%; TTFT P99 0.9% / 146.4%; TPOT P99 0.1% / 0.0%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 146.4%.**

### agentic-bf16 a32.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it (config+weights cff35dad39d6) | Same as baseline |
| Precision / quantization | weights bfloat16; KV cache bfloat16 | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 32 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet f33189e78e8e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1005, top1_decisive 0.9778, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.6 GiB | 78.4 GiB (1.09x) |
| Total throughput | 48,044 tok/s | 75,361 tok/s (1.57x) |
| Throughput / GPU | 48,044 tok/s/GPU | 75,361 tok/s/GPU (1.57x) |
| TTFT P99 | 3,540.7 ms | 1,527.6 ms (0.43x) * |
| TPOT P99 | 76.83 ms | 37.69 ms (0.49x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 1.2% / 3.6%; Throughput / GPU 1.2% / 3.6%; TTFT P99 9.4% / 50.8%; TPOT P99 1.2% / 19.8%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 50.8%; Infervisor TPOT P99 19.8%.**

### agentic-bf16 a64.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it (config+weights cff35dad39d6) | Same as baseline |
| Precision / quantization | weights bfloat16; KV cache bfloat16 | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 64 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet f33189e78e8e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1005, top1_decisive 0.9778, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.6 GiB | 77.9 GiB (1.09x) |
| Total throughput | 48,245 tok/s | 63,806 tok/s (1.32x) |
| Throughput / GPU | 48,245 tok/s/GPU | 63,806 tok/s/GPU (1.32x) |
| TTFT P99 | 10,759.5 ms | 27,940.2 ms (2.60x) |
| TPOT P99 | 94.14 ms | 46.16 ms (0.49x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.2%; Total throughput 0.2% / 3.7%; Throughput / GPU 0.2% / 3.7%; TTFT P99 1.1% / 4.3%; TPOT P99 0.0% / 5.2%.

### agentic-bf16 a128.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it (config+weights cff35dad39d6) | Same as baseline |
| Precision / quantization | weights bfloat16; KV cache bfloat16 | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 128 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 2cc23e9eb65a25a86b017775e093044ed1734631, packet f33189e78e8e) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1005, top1_decisive 0.9778, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.6 GiB | 78.0 GiB (1.09x) |
| Total throughput | 48,319 tok/s | 49,770 tok/s (1.03x) |
| Throughput / GPU | 48,319 tok/s/GPU | 49,770 tok/s/GPU (1.03x) |
| TTFT P99 | 31,581.2 ms | 35,943.0 ms (1.14x) |
| TPOT P99 | 94.67 ms | 83.60 ms (0.88x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.4%; Total throughput 0.6% / 8.1%; Throughput / GPU 0.6% / 8.1%; TTFT P99 0.8% / 0.4%; TPOT P99 0.7% / 13.4%.

**FLAGGED: spread > 10%: Infervisor TPOT P99 13.4%.**
