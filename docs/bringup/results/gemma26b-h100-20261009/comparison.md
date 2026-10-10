# Gemma-4-26B-A4B-it comparison, max_ctx 131072 (1x H100, vLLM 0.28)

Qualified wins only: `campaign.py report` MATCHED + EQUIVALENT, Infervisor total throughput above vLLM.
Every matched row is in [comparison.csv](comparison.csv) (`campaign_source` `fin131-20261009-<workload>-fp8`).

- Infervisor: plowrt `80097064` (gemma26b-next; the cells ran on its pre-rebase build, 26B step_bench
  digests identical), FP8 packet `5ce87bd28db1` from
  [`sm90a-h100-tp1-fp8.toml`](../../../../recipes/infervisor/gemma-4-26b-a4b/sm90a-h100-tp1-fp8.toml)
  (max_ctx 131072). Serve defaults from the packet; server env `PLOW_LIBCUDA` and the cuBLAS 13.4 library path.
- Baseline: vLLM 0.28.0, same client, `fp8_per_token_head` KV (TRITON_ATTN), `--max-model-len 131072`,
  max-num-batched-tokens 8192, prefix caching on. 2 repeats per cell, one server per arm.
  Reproduce: `scripts/campaign/repro_gemma26b_h100.sh` (workloads lat st4k st15k agentic lc32k lc128k).
- Quality: FP32-reference gate PASS twice (kl_mean 0.192 / 0.201 vs vLLM 0.202, kl_p99 3.14 / 3.83 vs
  3.73, needle 1.0). Long FP32 gate (20 cases to 32768 tokens) PASS (kl_mean 0.133 vs 0.165, kl_p99
  2.86 vs 3.56). Needles 54/54 at 8192..130000 tokens.

| Cell | Precision | Total throughput | TTFT P99 | TPOT P99 |
|---|---|---:|---:|---:|
| 1024/128 c1 | FP8 | 1.02x * | 1.87x | 1.00x |
| 4096/128 c32 | FP8 | 1.27x | 0.66x | 0.78x |
| 4096/128 c128 | FP8 | 1.30x | 0.65x | 0.75x |
| 15000/128 c32 | FP8 | 1.88x | 0.49x | 0.51x |
| 15000/128 c128 | FP8 | 1.92x | 0.50x | 0.51x |
| agentic16k c32 | FP8 | 1.20x | 0.52x | 0.77x |
| agentic16k c64 | FP8 | 1.13x | 0.79x | 0.84x |
| agentic16k c128 | FP8 | 3.30x * | 0.27x | 0.34x |
| 32768/128 c1 | FP8 | 2.25x * | 0.42x | 0.54x |
| 32768/128 c4 | FP8 | 2.18x | 0.41x | 0.47x |
| 32768/128 c16 | FP8 | 2.28x | 0.41x | 0.43x |
| 130944/128 c1 | FP8 | 3.18x | 0.31x | 0.49x |
| 130944/128 c4 | FP8 | 3.19x | 0.31x | 0.31x |
| 130944/128 c16 | FP8 | 3.26x | 0.31x | 0.29x |

`*` = repeat spread > 10% (FLAGGED; direction only). Not won (in the CSV): FP8 1024/128 c4 0.95x.
The open-loop production mix was not run.

BF16 at max_ctx 131072 is measured but not qualified: its long FP32 gate fails (kl_p99 2.56 vs vLLM 1.48,
cont_frac 0.560 vs 0.645), driven by the decode error after prompts of exactly 4096·k tokens described in
the BF16 recipe. The BF16 recipe stays at max_ctx 16384 (gate PASS on packet 475cb17f9986); its serving
comparison is [gemma26b-h100-20261007](../gemma26b-h100-20261007/comparison.md).

## Decisions

Prefill launch width, auto objective vs pinned wide (`PLOW_PF_INTERLEAVE=0`), same packets, 16K:
auto TTFT p50 -20..40% at c8..c64, but TTFT p99 +9..16%, tok/s -2..3% at 4K c16/c32, FP8 15K c128 -7%,
FP8 agentic c128 -11% tok/s with TTFT p99 3.6x (auto never launched wide). Narrow 4096-row launches cost
~8% more per row than 8192-row launches (96 vs 178 ms). Both recipes pin wide.

max_ctx, agentic c128 total tok/s (same binary, interleaved):

| max_ctx | FP8 | BF16 |
|---:|---:|---:|
| 16384 | 75965 | 51717 |
| 65536 | 76209 | 51049 |
| 131072 | 76154 | 50841 |
| 262144 | 66707 | 50036 |

At 262144 the FP8 KV budget drops 22.93 -> 20.18 GiB (flat FP8-KV scales, RoPE tables). st4k / st15k /
latency cells are unchanged across max_ctx. An FP32 reference past 32768 tokens does not fit one 80 GiB
GPU (51.5 GiB weights, 77.5 GiB peak at 32768), so long-context quality past 32768 rests on needles.

## Strict tables

### st4k (fp8 vs vllm-fp8)

#### g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 75.8 GiB (1.06x) |
| Total throughput | 20,451 tok/s | 26,056 tok/s (1.27x) |
| Throughput / GPU | 20,451 tok/s/GPU | 26,056 tok/s/GPU (1.27x) |
| TTFT P99 | 4,668.3 ms | 3,060.6 ms (0.66x) |
| TPOT P99 | 49.71 ms | 38.87 ms (0.78x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 4.7%; Total throughput 0.8% / 2.6%; Throughput / GPU 0.8% / 2.6%; TTFT P99 1.8% / 9.2%; TPOT P99 0.1% / 1.0%.

#### g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 77.7 GiB (1.08x) |
| Total throughput | 24,175 tok/s | 31,397 tok/s (1.30x) |
| Throughput / GPU | 24,175 tok/s/GPU | 31,397 tok/s/GPU (1.30x) |
| TTFT P99 | 18,580.7 ms | 12,096.5 ms (0.65x) |
| TPOT P99 | 169.96 ms | 127.94 ms (0.75x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.3% / 0.4%; Throughput / GPU 0.3% / 0.4%; TTFT P99 0.8% / 2.6%; TPOT P99 0.0% / 0.3%.

### st15k (fp8 vs vllm-fp8)

#### g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 15000 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 67.5 GiB (0.94x) |
| Total throughput | 15,399 tok/s | 28,928 tok/s (1.88x) |
| Throughput / GPU | 15,399 tok/s/GPU | 28,928 tok/s/GPU (1.88x) |
| TTFT P99 | 27,925.0 ms | 13,675.3 ms (0.49x) |
| TPOT P99 | 237.58 ms | 122.07 ms (0.51x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 5.9%; Total throughput 0.1% / 0.2%; Throughput / GPU 0.1% / 0.2%; TTFT P99 0.4% / 2.0%; TPOT P99 0.1% / 0.1%.

#### g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 15000 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 77.6 GiB (1.08x) |
| Total throughput | 16,105 tok/s | 30,931 tok/s (1.92x) |
| Throughput / GPU | 16,105 tok/s/GPU | 30,931 tok/s/GPU (1.92x) |
| TTFT P99 | 114,359.1 ms | 57,704.3 ms (0.50x) |
| TPOT P99 | 507.04 ms | 256.91 ms (0.51x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 0.0% / 0.1%; Throughput / GPU 0.0% / 0.1%; TTFT P99 0.1% / 0.9%; TPOT P99 0.1% / 1.5%.

### agentic (fp8 vs vllm-fp8)

#### a32.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 32 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 77.6 GiB (1.08x) |
| Total throughput | 51,416 tok/s | 61,466 tok/s (1.20x) |
| Throughput / GPU | 51,416 tok/s/GPU | 61,466 tok/s/GPU (1.20x) |
| TTFT P99 | 2,612.9 ms | 1,369.5 ms (0.52x) |
| TPOT P99 | 56.51 ms | 43.63 ms (0.77x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.2%; Total throughput 0.0% / 3.8%; Throughput / GPU 0.0% / 3.8%; TTFT P99 2.8% / 8.6%; TPOT P99 0.0% / 0.7%.

#### a64.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 64 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 77.7 GiB (1.08x) |
| Total throughput | 62,196 tok/s | 70,164 tok/s (1.13x) |
| Throughput / GPU | 62,196 tok/s/GPU | 70,164 tok/s/GPU (1.13x) |
| TTFT P99 | 3,008.4 ms | 2,371.0 ms (0.79x) |
| TPOT P99 | 93.24 ms | 78.30 ms (0.84x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.0% / 0.9%; Throughput / GPU 0.0% / 0.9%; TTFT P99 2.3% / 0.3%; TPOT P99 0.3% / 1.2%.

#### a128.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 128 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 77.8 GiB (1.08x) |
| Total throughput | 20,816 tok/s | 68,684 tok/s (3.30x) * |
| Throughput / GPU | 20,816 tok/s/GPU | 68,684 tok/s/GPU (3.30x) * |
| TTFT P99 | 50,001.3 ms | 13,337.4 ms (0.27x) * |
| TPOT P99 | 551.24 ms | 186.14 ms (0.34x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.2%; Total throughput 0.2% / 10.6%; Throughput / GPU 0.2% / 10.6%; TTFT P99 0.0% / 12.1%; TPOT P99 0.1% / 0.1%.

**FLAGGED: spread > 10%: Infervisor Total throughput 10.6%; Infervisor Throughput / GPU 10.6%; Infervisor TTFT P99 12.1%.**

### lc32k (fp8 vs vllm-fp8)

#### g1

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 32768 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 16 prompts, request rate inf / concurrency 1 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 77.6 GiB (1.08x) |
| Total throughput | 6,936 tok/s | 15,595 tok/s (2.25x) |
| Throughput / GPU | 6,936 tok/s/GPU | 15,595 tok/s/GPU (2.25x) |
| TTFT P99 | 3,314.9 ms | 1,402.5 ms (0.42x) * |
| TPOT P99 | 11.72 ms | 6.31 ms (0.54x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.2%; Total throughput 0.2% / 0.6%; Throughput / GPU 0.2% / 0.6%; TTFT P99 0.8% / 13.0%; TPOT P99 0.1% / 0.6%.

**FLAGGED: spread > 10%: Infervisor TTFT P99 13.0%.**

#### g4

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 32768 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 12 prompts, request rate inf / concurrency 4 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 77.8 GiB (1.08x) |
| Total throughput | 9,144 tok/s | 19,951 tok/s (2.18x) |
| Throughput / GPU | 9,144 tok/s/GPU | 19,951 tok/s/GPU (2.18x) |
| TTFT P99 | 12,574.0 ms | 5,210.1 ms (0.41x) |
| TPOT P99 | 86.38 ms | 40.42 ms (0.47x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.1% / 0.0%; Throughput / GPU 0.1% / 0.0%; TTFT P99 1.7% / 0.4%; TPOT P99 0.1% / 0.4%.

#### g16

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 32768 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 48 prompts, request rate inf / concurrency 16 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 77.9 GiB (1.08x) |
| Total throughput | 9,965 tok/s | 22,707 tok/s (2.28x) |
| Throughput / GPU | 9,965 tok/s/GPU | 22,707 tok/s/GPU (2.28x) |
| TTFT P99 | 49,503.8 ms | 20,153.4 ms (0.41x) |
| TPOT P99 | 385.76 ms | 167.30 ms (0.43x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.2% / 0.1%; Throughput / GPU 0.2% / 0.1%; TTFT P99 0.2% / 0.8%; TPOT P99 0.3% / 0.0%.

### lc128k (fp8 vs vllm-fp8)

#### g1

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 130944 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 16 prompts, request rate inf / concurrency 1 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 78.6 GiB (1.09x) |
| Total throughput | 3,045 tok/s | 9,683 tok/s (3.18x) |
| Throughput / GPU | 3,045 tok/s/GPU | 9,683 tok/s/GPU (3.18x) |
| TTFT P99 | 40,798.0 ms | 12,562.2 ms (0.31x) |
| TPOT P99 | 19.17 ms | 9.34 ms (0.49x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.5%; Total throughput 0.1% / 0.3%; Throughput / GPU 0.1% / 0.3%; TTFT P99 0.0% / 3.0%; TPOT P99 0.0% / 0.7%.

#### g4

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 130944 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 12 prompts, request rate inf / concurrency 4 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 78.8 GiB (1.10x) |
| Total throughput | 3,203 tok/s | 10,225 tok/s (3.19x) |
| Throughput / GPU | 3,203 tok/s/GPU | 10,225 tok/s/GPU (3.19x) |
| TTFT P99 | 156,395.7 ms | 48,033.9 ms (0.31x) |
| TPOT P99 | 967.12 ms | 300.94 ms (0.31x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 0.0% / 0.0%; Throughput / GPU 0.0% / 0.0%; TTFT P99 0.0% / 0.4%; TPOT P99 0.3% / 0.3%.

#### g16

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 130944 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 48 prompts, request rate inf / concurrency 16 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 79.2 GiB (1.10x) |
| Total throughput | 3,193 tok/s | 10,416 tok/s (3.26x) |
| Throughput / GPU | 3,193 tok/s/GPU | 10,416 tok/s/GPU (3.26x) |
| TTFT P99 | 631,210.1 ms | 197,099.3 ms (0.31x) |
| TPOT P99 | 2,565.06 ms | 752.76 ms (0.29x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.0% / 0.1%; Throughput / GPU 0.0% / 0.1%; TTFT P99 0.0% / 0.6%; TPOT P99 0.0% / 1.7%.

### lat (fp8 vs vllm-fp8)

#### g1

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 1024 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 16 prompts, request rate inf / concurrency 1 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 60.2 GiB (0.84x) |
| Total throughput | 1,559 tok/s | 1,583 tok/s (1.02x) |
| Throughput / GPU | 1,559 tok/s/GPU | 1,583 tok/s/GPU (1.02x) |
| TTFT P99 | 68.0 ms * | 126.8 ms (1.87x) * |
| TPOT P99 | 5.44 ms | 5.42 ms (1.00x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.2% / 1.9%; Throughput / GPU 0.2% / 1.9%; TTFT P99 31.9% / 148.4%; TPOT P99 0.1% / 0.0%.

**FLAGGED: spread > 10%: Baseline TTFT P99 31.9%; Infervisor TTFT P99 148.4%.**

#### g4

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-26b-a4b-it-fp8 (config+weights 541337d79dba) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 1024 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 12 prompts, request rate inf / concurrency 4 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --max-model-len 131072 --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt db1af0b6ba208d800c576cd1a9ae521b53cfa94b, packet 5ce87bd28db1) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.202, top1_decisive 0.9662, needle_acc 1) | Equivalent |
| Peak GPU memory | 71.8 GiB | 60.2 GiB (0.84x) |
| Total throughput | 4,674 tok/s | 4,424 tok/s (0.95x) |
| Throughput / GPU | 4,674 tok/s/GPU | 4,424 tok/s/GPU (0.95x) |
| TTFT P99 | 137.5 ms | 177.6 ms (1.29x) |
| TPOT P99 | 7.18 ms | 7.94 ms (1.11x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.2% / 0.5%; Throughput / GPU 0.2% / 0.5%; TTFT P99 1.1% / 1.7%; TPOT P99 0.8% / 0.0%.
