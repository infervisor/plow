# Gemma-4 12B FP8 comparison (H100)

This file holds only qualified wins and their evidence. Experimental history, internal A/Bs and
the non-matched arms are summarized in [../gemma4-h100/summary.md](../gemma4-h100/summary.md).
Every matched serving row is in [comparison.csv](comparison.csv) (`campaign_source` names the
experiment).

## Qualified wins (MATCHED + EQUIVALENT, `campaign.py report` exit 0)

- Infervisor: plowrt `66e90a4b` (gemma12b-next), packet `c47fc3f20569`, built by `campaign.py
  build` from [`recipes/infervisor/gemma-4-12b/sm90a-h100-tp1.toml`](../../../../recipes/infervisor/gemma-4-12b/sm90a-h100-tp1.toml)
  (FP8 W8A8 weights, FP8 per-token-head KV, 16K context, 128 decode slots, prefix cache on).
- Baseline: vLLM 0.28.0 with matched `fp8_per_token_head` KV. That KV dtype runs only on
  TRITON_ATTN (FLASHINFER rejects it, FLASH_ATTN does not list it). Probe at 4K/128 c32, out tok/s:
  max-num-batched-tokens 8192 = 399, 16384 = 394, 4096 = 398; 8192 is used.
- One H100 80GB HBM3, same vLLM client, greedy, 2 repeats per cell in one server per arm.
- Quality: FP32-reference gate PASS for the served packet (`gates.json` sha256 `73e61b0727c1`;
  KL mean 0.108 vs vLLM 0.128). See [below](#fp32-reference-quality-gate).

Ratios are Infervisor / vLLM, from the strict tables below. `*` = repeat spread > 10% (FLAGGED).

| Cell | Total throughput | TTFT P99 | TPOT P99 | Goodput (supplementary) |
|---|---:|---:|---:|---:|
| 4096/128 c32 | 1.38x | 0.70x | 0.70x | |
| 4096/128 c128 | 1.37x | 0.77x | 0.36x | |
| 15000/128 c32 | 1.79x | 0.57x | 0.30x | |
| 15000/128 c128 | 1.74x | 0.57x | 0.14x | |
| agentic16k c32 | 1.25x | 0.98x | 0.78x * | |
| agentic16k c64 | 1.09x | 1.40x | 0.90x | |
| agentic16k c128 | 2.41x * | 0.68x * | 0.25x * | |
| open loop, 0.628 sessions/s | 1.07x | 0.44x | 0.62x * | 1.06x (2.643 vs 2.485 req/s) |
| open loop, 0.771 sessions/s | 1.16x | 0.43x * | 0.34x * | 1.54x (3.042 vs 1.980 req/s) |
| open loop, 0.987 sessions/s | 1.39x * | 0.43x * | 0.35x * | 3.36x (2.513 vs 0.748 req/s) |

Goodput = requests per second meeting TTFT <= 2000 ms and TPOT <= 100 ms. The open-loop rates are
the vLLM-calibrated points for mean in-flight 16 / 48 / 96 (`prodbench/rates.txt`).

Caveats:
- This win is against vLLM's matched FP8-KV config, not its fastest config overall. Against vLLM
  BF16 KV on FLASH_ATTN (MATCHED with the BF16-KV plow packet `34641b1cace3`, rows
  `final-compare-20261004-*-bf16-vbf16`), plow loses: 4K 0.76x / 0.72x, 15K 0.78x / 0.78x,
  agentic 0.23x / 0.65x / 0.67x at c32 / c64 / c128. The plow FP8 arm (550 / 145 out tok/s at
  4K / 15K c32) is also below vLLM BF16 KV (780 / 235).
- agentic16k c128 is unstable across repeats: 539 vs 290 out tok/s, prefix token hits 75.6% vs
  25.8% in r2.
- Host load average (1-minute) during the closed-loop lease: median 1.4, max 7.8.
- The recipe moved from `scripts/campaign/recipes/gemma4-12b.h100.fp8kv-16k-c128.toml` to
  production after this run (label and gate table only; routes and flags unchanged).

Evidence (campaign scratch):
- Closed loop: `/opt/dlami/nvme/lava-tts/final2/` (`report/` strict reports, `res/` raw,
  `gate/` FP32-gate captures + `gates.json`, `probe/` vLLM baseline probe).
- Open loop: `/opt/dlami/nvme/lava-tts/prodbench/` (`report/` strict report, `res/` raw,
  `job.sh`, `rates.txt`).

## Strict reports

### 4096/128, closed loop

Report `/opt/dlami/nvme/lava-tts/final2/report/st4k-fp8/comparison.md`. Arms: baseline `/opt/dlami/nvme/lava-tts/final2/res/st4k/vllm`, Infervisor `/opt/dlami/nvme/lava-tts/final2/res/st4k/fp8`. Gate `/opt/dlami/nvme/lava-tts/final2/gate/fp8/gates.json` (sha256 73e61b0727c1, packet c47fc3f20569, PASS).

#### g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 66e90a4b3a3a847838771a6a0dcb3b5e05923e0d, packet c47fc3f20569) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.0 GiB | 77.5 GiB (1.06x) |
| Total throughput | 13,145 tok/s | 18,146 tok/s (1.38x) |
| Throughput / GPU | 13,145 tok/s/GPU | 18,146 tok/s/GPU (1.38x) |
| TTFT P99 | 7,770.7 ms | 5,458.5 ms (0.70x) |
| TPOT P99 | 77.85 ms | 54.43 ms (0.70x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 0.0% / 0.6%; Throughput / GPU 0.0% / 0.6%; TTFT P99 0.8% / 3.3%; TPOT P99 0.3% / 0.0%.

#### g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 66e90a4b3a3a847838771a6a0dcb3b5e05923e0d, packet c47fc3f20569) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.0 GiB | 77.7 GiB (1.06x) |
| Total throughput | 15,004 tok/s | 20,495 tok/s (1.37x) |
| Throughput / GPU | 15,004 tok/s/GPU | 20,495 tok/s/GPU (1.37x) |
| TTFT P99 | 30,953.8 ms | 23,839.2 ms (0.77x) |
| TPOT P99 | 275.42 ms | 98.30 ms (0.36x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.1% / 0.3%; Throughput / GPU 0.1% / 0.3%; TTFT P99 0.6% / 0.4%; TPOT P99 0.4% / 0.3%.

### 15000/128, closed loop

Report `/opt/dlami/nvme/lava-tts/final2/report/st15k-fp8/comparison.md`. Arms: baseline `/opt/dlami/nvme/lava-tts/final2/res/st15k/vllm`, Infervisor `/opt/dlami/nvme/lava-tts/final2/res/st15k/fp8`. Gate `/opt/dlami/nvme/lava-tts/final2/gate/fp8/gates.json` (sha256 73e61b0727c1, packet c47fc3f20569, PASS).

#### g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 15000 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 66e90a4b3a3a847838771a6a0dcb3b5e05923e0d, packet c47fc3f20569) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.0 GiB | 77.7 GiB (1.06x) |
| Total throughput | 9,567 tok/s | 17,108 tok/s (1.79x) |
| Throughput / GPU | 9,567 tok/s/GPU | 17,108 tok/s/GPU (1.79x) |
| TTFT P99 | 45,534.3 ms | 25,885.1 ms (0.57x) |
| TPOT P99 | 382.92 ms | 114.55 ms (0.30x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 0.0% / 0.2%; Throughput / GPU 0.0% / 0.2%; TTFT P99 0.1% / 0.6%; TPOT P99 0.0% / 0.3%.

#### g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 15000 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 66e90a4b3a3a847838771a6a0dcb3b5e05923e0d, packet c47fc3f20569) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.0 GiB | 77.8 GiB (1.07x) |
| Total throughput | 9,895 tok/s | 17,255 tok/s (1.74x) |
| Throughput / GPU | 9,895 tok/s/GPU | 17,255 tok/s/GPU (1.74x) |
| TTFT P99 | 186,793.9 ms | 106,631.0 ms (0.57x) |
| TPOT P99 | 825.66 ms | 115.83 ms (0.14x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.2%; Total throughput 0.1% / 0.1%; Throughput / GPU 0.1% / 0.1%; TTFT P99 0.1% / 0.3%; TPOT P99 0.0% / 0.0%.

### agentic16k (`llm_grid.sh --agentic`)

Report `/opt/dlami/nvme/lava-tts/final2/report/agentic-fp8/comparison.md`. Arms: baseline `/opt/dlami/nvme/lava-tts/final2/res/agentic/vllm`, Infervisor `/opt/dlami/nvme/lava-tts/final2/res/agentic/fp8`. Gate `/opt/dlami/nvme/lava-tts/final2/gate/fp8/gates.json` (sha256 73e61b0727c1, packet c47fc3f20569, PASS).

#### a32.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 32 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 66e90a4b3a3a847838771a6a0dcb3b5e05923e0d, packet c47fc3f20569) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.3 GiB | 77.8 GiB (1.06x) |
| Total throughput | 35,035 tok/s | 43,788 tok/s (1.25x) |
| Throughput / GPU | 35,035 tok/s/GPU | 43,788 tok/s/GPU (1.25x) |
| TTFT P99 | 3,818.5 ms | 3,742.7 ms (0.98x) |
| TPOT P99 | 82.04 ms | 63.76 ms (0.78x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.2% / 5.6%; Throughput / GPU 0.2% / 5.6%; TTFT P99 4.2% / 0.4%; TPOT P99 3.6% / 20.8%.

**FLAGGED: spread > 10%: Infervisor TPOT P99 20.8%.**

#### a64.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 64 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 66e90a4b3a3a847838771a6a0dcb3b5e05923e0d, packet c47fc3f20569) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.3 GiB | 77.8 GiB (1.06x) |
| Total throughput | 42,380 tok/s | 46,014 tok/s (1.09x) |
| Throughput / GPU | 42,380 tok/s/GPU | 46,014 tok/s/GPU (1.09x) |
| TTFT P99 | 5,371.2 ms | 7,502.0 ms (1.40x) |
| TPOT P99 | 139.29 ms | 125.06 ms (0.90x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.0% / 1.3%; Throughput / GPU 0.0% / 1.3%; TTFT P99 1.1% / 1.6%; TPOT P99 0.2% / 1.9%.

#### a128.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 128 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 66e90a4b3a3a847838771a6a0dcb3b5e05923e0d, packet c47fc3f20569) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.3 GiB | 77.8 GiB (1.06x) |
| Total throughput | 12,553 tok/s | 30,257 tok/s (2.41x) * |
| Throughput / GPU | 12,553 tok/s/GPU | 30,257 tok/s/GPU (2.41x) * |
| TTFT P99 | 82,042.6 ms | 55,982.3 ms (0.68x) * |
| TPOT P99 | 892.14 ms | 223.26 ms (0.25x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.1% / 60.0%; Throughput / GPU 0.1% / 60.0%; TTFT P99 0.1% / 129.9%; TPOT P99 0.1% / 25.4%.

**FLAGGED: spread > 10%: Infervisor Total throughput 60.0%; Infervisor Throughput / GPU 60.0%; Infervisor TTFT P99 129.9%; Infervisor TPOT P99 25.4%.**

### open-loop production mix (`llm_grid.sh --prod`)

Report `/opt/dlami/nvme/lava-tts/prodbench/report/comparison.md`. Arms: baseline `/opt/dlami/nvme/lava-tts/prodbench/res/vllm`, Infervisor `/opt/dlami/nvme/lava-tts/prodbench/res/fp8`. Gate `/opt/dlami/nvme/lava-tts/final2/gate/fp8/gates.json` (sha256 73e61b0727c1, packet c47fc3f20569, PASS).

#### q628.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | open-loop mix: 4 system prompts (lognormal median 1536), turns geometric mean 6 max 20, first message lognormal(1500, 1), tool output lognormal(700, 1), context cap 16384 / output lognormal(160, 0.7) in [16, 1024] tokens, ignore_eos | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, Poisson 0.628 sessions/s, think lognormal(5 s, 0.8) <= 60 s, 300 s (measured 75-275 s), session header on / open loop (achieved concurrency in the supplementary note), seeds 1099191/1107110 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 66e90a4b3a3a847838771a6a0dcb3b5e05923e0d, packet c47fc3f20569) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 72.9 GiB | 77.8 GiB (1.07x) |
| Total throughput | 18,611 tok/s | 19,824 tok/s (1.07x) |
| Throughput / GPU | 18,611 tok/s/GPU | 19,824 tok/s/GPU (1.07x) |
| TTFT P99 | 1,853.7 ms | 819.5 ms (0.44x) |
| TPOT P99 | 94.83 ms * | 59.24 ms (0.62x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 6.1% / 7.7%; Throughput / GPU 6.1% / 7.7%; TTFT P99 3.0% / 4.0%; TPOT P99 30.4% / 21.3%.

**FLAGGED: spread > 10%: Baseline TPOT P99 30.4%; Infervisor TPOT P99 21.3%.**

Supplementary (outside the strict table; goodput = requests meeting TTFT <= 2000 ms and TPOT <= 100 ms per second, measured window), Baseline / Infervisor: goodput 2.485 req/s / 2.643 req/s; SLO met 98.0% / 100.0%; requests 2.540 req/s / 2.643 req/s; mean in-flight 29.3 / 22.2; mean live sessions 42.7 / 36.3; TTFT P50 275.2 ms / 181.7 ms; TPOT P50 59.92 ms / 41.81 ms; cached prompt tokens 77.2% / 72.9%.

#### q771.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | open-loop mix: 4 system prompts (lognormal median 1536), turns geometric mean 6 max 20, first message lognormal(1500, 1), tool output lognormal(700, 1), context cap 16384 / output lognormal(160, 0.7) in [16, 1024] tokens, ignore_eos | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, Poisson 0.771 sessions/s, think lognormal(5 s, 0.8) <= 60 s, 300 s (measured 75-275 s), session header on / open loop (achieved concurrency in the supplementary note), seeds 1117924/1125843 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 66e90a4b3a3a847838771a6a0dcb3b5e05923e0d, packet c47fc3f20569) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 72.9 GiB | 77.8 GiB (1.07x) |
| Total throughput | 18,868 tok/s | 21,810 tok/s (1.16x) |
| Throughput / GPU | 18,868 tok/s/GPU | 21,810 tok/s/GPU (1.16x) |
| TTFT P99 | 3,790.5 ms * | 1,619.0 ms (0.43x) * |
| TPOT P99 | 238.80 ms * | 81.03 ms (0.34x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 2.8% / 6.5%; Throughput / GPU 2.8% / 6.5%; TTFT P99 64.3% / 51.5%; TPOT P99 36.6% / 20.5%.

**FLAGGED: spread > 10%: Baseline TTFT P99 64.3%; Infervisor TTFT P99 51.5%; Baseline TPOT P99 36.6%; Infervisor TPOT P99 20.5%.**

Supplementary (outside the strict table; goodput = requests meeting TTFT <= 2000 ms and TPOT <= 100 ms per second, measured window), Baseline / Infervisor: goodput 1.980 req/s / 3.042 req/s; SLO met 71.9% / 99.2%; requests 2.745 req/s / 3.067 req/s; mean in-flight 44.1 / 34.0; mean live sessions 58.1 / 50.1; TTFT P50 371.7 ms / 233.8 ms; TPOT P50 75.55 ms / 58.36 ms; cached prompt tokens 69.5% / 69.7%.

#### q987.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | open-loop mix: 4 system prompts (lognormal median 1536), turns geometric mean 6 max 20, first message lognormal(1500, 1), tool output lognormal(700, 1), context cap 16384 / output lognormal(160, 0.7) in [16, 1024] tokens, ignore_eos | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, Poisson 0.987 sessions/s, think lognormal(5 s, 0.8) <= 60 s, 300 s (measured 75-275 s), session header on / open loop (achieved concurrency in the supplementary note), seeds 1146220/1154139 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt 66e90a4b3a3a847838771a6a0dcb3b5e05923e0d, packet c47fc3f20569) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 72.9 GiB | 77.8 GiB (1.07x) |
| Total throughput | 17,143 tok/s | 23,876 tok/s (1.39x) * |
| Throughput / GPU | 17,143 tok/s/GPU | 23,876 tok/s/GPU (1.39x) * |
| TTFT P99 | 6,135.4 ms * | 2,638.6 ms (0.43x) * |
| TPOT P99 | 420.65 ms * | 147.53 ms (0.35x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.6% / 11.2%; Throughput / GPU 0.6% / 11.2%; TTFT P99 16.7% / 60.4%; TPOT P99 10.4% / 46.5%.

**FLAGGED: spread > 10%: Infervisor Total throughput 11.2%; Infervisor Throughput / GPU 11.2%; Baseline TTFT P99 16.7%; Infervisor TTFT P99 60.4%; Baseline TPOT P99 10.4%; Infervisor TPOT P99 46.5%.**

Supplementary (outside the strict table; goodput = requests meeting TTFT <= 2000 ms and TPOT <= 100 ms per second, measured window), Baseline / Infervisor: goodput 0.748 req/s / 2.513 req/s; SLO met 26.3% / 68.4%; requests 2.835 req/s / 3.660 req/s; mean in-flight 90.9 / 62.9; mean live sessions 103.2 / 80.7; TTFT P50 1,070.0 ms / 359.7 ms; TPOT P50 188.53 ms / 89.68 ms; cached prompt tokens 48.4% / 64.1%.

## FP32-reference quality gate

The qualified packet `c47fc3f20569` passed this gate at plowrt `66e90a4b`
(`/opt/dlami/nvme/lava-tts/final2/gate/fp8/gates.json`, 2026-10-04): KL mean 0.1076 vs vLLM
0.1277, KL p99 3.277 vs 3.188, top1_decisive 0.9813 vs 0.9823, cont_frac 0.656 vs 0.574, needle
1.0 vs 1.0. The calibration run below, which set the thresholds, used the earlier grid-trial
packet.

Bit-exact agreement with vLLM is replaced as the promotion gate by `campaign.py gate` kind `llm_fp32_ref` ([agent-tools.md §5](../../agent-tools.md#5-numerics-and-retrieval-gates)). Reference: the same `gemma-4-12b-it-fp8` checkpoint dequantized to FP32 weights, FP32 activations, TF32 off, exact attention (H100, torch 2.13, transformers 5.16). Prompt set: 46 cases and 1,172 teacher-forced positions. That is 20 natural-text cases (128 to 15,872 tokens), 8 agentic chat cases (54 to 15,255 tokens) and 18 chat needles (4K and 15,872 tokens). Both stacks are teacher-forced on the FP32 greedy continuation through `/v1/completions` at concurrency 16. Plow is the current best packet: grid-trial dedicated-grid 16K/128-slot FP8-KV, with the `serve-short-grid` serve env and prefix cache on. vLLM is 0.28.0 with the `serve-short-vllm-fp8pth` flags. Each stack was captured twice against the same server.

| stack | KL mean | KL p99 | top-1, FP32 margin > 1 nat (1,016 pos.) | greedy agreement | needle | KL mean, natural / chat / needle |
|---|---:|---:|---:|---:|---:|---|
| Plow r1 | 0.1038 | 2.279 | 0.9852 | 0.608 | 18/18 | 0.184 / 0.0086 / 6.2e-7 |
| Plow r2 | 0.0971 | 2.431 | 0.9852 | 0.642 | 18/18 | 0.172 / 0.0082 / 5.0e-7 |
| vLLM r1 | 0.1277 | 3.188 | 0.9823 | 0.574 | 18/18 | 0.226 / 0.0114 / 1.7e-6 |
| vLLM r2 | 0.1084 | 3.011 | 0.9882 | 0.595 | 18/18 | 0.190 / 0.0122 / 1.6e-6 |

**Verdict: PASS.** Plow passes against both vLLM repeats, and so does Plow r2. Plow is at least as close to FP32 as vLLM on every metric. Across all positions, top-1 differs from FP32 at 65–72 of 1,172 positions on both stacks. The decisive flips and the largest KL values come from repetition loops in long natural-text continuations, and they occur on both stacks. Chat and needle positions are close to exact on both (needle KL ≤ 4.4e-5).

Thresholds are kept relative to vLLM and checked against the measured repeat spread:
- KL: Plow ≤ vLLM × 1.25 + 0.002. vLLM's own repeats differ by 1.18× in mean and 1.06× in p99.
- Top-1: −0.01. vLLM repeats differ by 0.006.
- Greedy agreement: −0.05. Repeat spreads were 0.021 for vLLM and 0.034 for Plow.
- Needle: no drop allowed, and at least 0.9.

vLLM run against its own repeat passes in both directions, so the tolerances admit the reference stack's own noise. None of the provisional values needed to change.

During the run, plowrt was found to omit a stop token from completions logprobs (`finish_reason=stop`, `tokens=[]`), unlike vLLM. The capture now sends `ignore_eos`. Evidence is in `/opt/dlami/nvme/lava-tts/fp32gate/`:
- `ref.json`: sha256 `b2988c51…`.
- `prompts.json`: `10302403…`.
- Captures in `best/gate/llm_fp32_ref/`: plow `3d79bd08…`, vllm `74a7a572…`.
- Gate recipe: `best/recipe.toml`.

