# Gemma-4 12B FP8 comparison (H100)

This file holds only qualified wins and their evidence. Experimental history, internal A/Bs and
the non-matched arms are summarized in [../gemma4-h100/summary.md](../gemma4-h100/summary.md).
Every matched serving row is in [comparison.csv](comparison.csv) (`campaign_source` names the
experiment; this run's rows are `final3-20261005-*`).

## Qualified wins (MATCHED + EQUIVALENT, `campaign.py report` exit 0)

- Infervisor: plowrt `ee57f7b7` (gemma12b-next), packet `5fbe627c02af`, built by `campaign.py
  build` at gemma12b-next `6a21c8c3` from [`recipes/infervisor/gemma-4-12b/sm90a-h100-tp1.toml`](../../../../recipes/infervisor/gemma-4-12b/sm90a-h100-tp1.toml)
  (FP8 W8A8 weights, FP8 per-token-head KV, 16K context, 128 decode slots, prefix cache on). The
  bundle is self-contained: serve defaults come from the packet's `serve.json` (equal to the
  recipe `[serve.env]`); the only server env was `PLOW_LIBCUDA`.
- Baseline: vLLM 0.28.0 with matched `fp8_per_token_head` KV. That KV dtype runs only on
  TRITON_ATTN (FLASHINFER rejects it, FLASH_ATTN does not list it). Probe at 4K/128 c32, out tok/s:
  max-num-batched-tokens 8192 = 399, 16384 = 393, 4096 = 398; 8192 is used. Baseline arms are the
  final2 (closed loop) and prodbench (open loop) vLLM runs, same client and plan hash.
- One H100 80GB HBM3, same vLLM client, greedy, 2 repeats per cell in one server per arm.
- Quality: FP32-reference gate PASS for the served packet (`gates.json` sha256 `1e64f6058d79`;
  KL mean 0.1048 vs vLLM 0.1277). See [below](#fp32-reference-quality-gate).
- Rebuild at `e93e0a64` (packet `ab1c1d583f8d`): differs only in
  `gen_sm90a_attn_pf_hd512_fp8kv.cubin` (unpacked partial-chunk fix; the `5fbe627c` object faults
  on step_bench's unpacked path at non-rung prompt lengths, which serving never takes). Gate PASS,
  KL mean 0.1053; catalog bench rel_l2 and timing unchanged (within 2%).

Ratios are Infervisor / vLLM, from the strict tables below. `*` = repeat spread > 10% (FLAGGED);
flagged values are direction only.

| Cell | Total throughput | TTFT P99 | TPOT P99 | Goodput (supplementary) |
|---|---:|---:|---:|---:|
| 4096/128 c32 | 1.38x | 0.70x | 0.70x |  |
| 4096/128 c128 | 1.37x | 0.77x | 0.36x |  |
| 15000/128 c32 | 1.79x | 0.57x | 0.30x |  |
| 15000/128 c128 | 1.75x | 0.57x | 0.14x |  |
| agentic16k c32 | 1.40x | 0.57x | 0.67x |  |
| agentic16k c64 | 1.31x | 0.79x | 0.71x |  |
| agentic16k c128 | 4.02x | 0.16x | 0.24x |  |
| open loop, 0.628 sessions/s | 1.08x | 0.41x | 0.60x * | 1.08x (2.675 vs 2.485 req/s) |
| open loop, 0.771 sessions/s | 1.18x | 0.28x * | 0.31x * | 1.57x (3.103 vs 1.980 req/s) |
| open loop, 0.987 sessions/s | 1.50x | 0.23x * | 0.26x * | 4.57x (3.412 vs 0.748 req/s) |

Goodput = requests per second meeting TTFT <= 2000 ms and TPOT <= 100 ms. The open-loop rates are
the vLLM-calibrated points for mean in-flight 16 / 48 / 96 (`prodbench/rates.txt`).

Caveats:
- This win is against vLLM's matched FP8-KV config, not its fastest config overall. Against vLLM
  BF16 KV (default backend, MATCHED + EQUIVALENT with the BF16-KV plow packet `af6ee1c20678`,
  gate PASS KL 0.0821, rows `final3-20261005-*-bf16`), total throughput is 4K 0.83x / 0.79x, 15K
  0.86x / 0.86x at c32 / c128, agentic 0.68x / 2.01x / 1.39x at c32 / c64 / c128. The plow FP8 arm
  (551 / 145 out tok/s at 4K / 15K c32) is also below vLLM BF16 KV (780 / 235).
- Host load average (1-minute) during the leases, mean / max: 4K 1.2 / 1.4, 15K 1.3 / 2.7,
  agentic 2.2 / 3.0 (overlapped a niced 4-core packet build), open loop 1.0 / 1.1.

Evidence (campaign scratch):
- Infervisor arms, gates, reports: `/opt/dlami/nvme/lava-tts/final3/` (`report/` strict reports,
  `res/` raw, `gate/{fp8,bf16,fp8b}/` FP32-gate captures + `gates.json`, `res/unpacked*/`
  unpacked partial-chunk checks, `res/backcompat/`, `res/ramp/` latency/throughput mode ramp).
- vLLM arms: `/opt/dlami/nvme/lava-tts/final2/res/` (closed loop, `probe/` baseline probe) and
  `/opt/dlami/nvme/lava-tts/prodbench/res/vllm` (open loop, `rates.txt`).
- Rebuild: `scripts/campaign/repro_gemma12b_h100.sh`.

## Strict reports

### 4096/128, closed loop

Report `/opt/dlami/nvme/lava-tts/final3/report/st4k-fp8/comparison.md`. Arms: baseline `/opt/dlami/nvme/lava-tts/final2/res/st4k/vllm`, Infervisor `/opt/dlami/nvme/lava-tts/final3/res/st4k/fp8`. Gate `/opt/dlami/nvme/lava-tts/final3/gate/fp8/gates.json` (sha256 1e64f6058d79, packet 5fbe627c02af, PASS).

#### g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt ee57f7b7a084baab3295e7015bd4136a43afc17e, packet 5fbe627c02af) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.0 GiB | 77.6 GiB (1.06x) |
| Total throughput | 13,145 tok/s | 18,177 tok/s (1.38x) |
| Throughput / GPU | 13,145 tok/s/GPU | 18,177 tok/s/GPU (1.38x) |
| TTFT P99 | 7,770.7 ms | 5,419.1 ms (0.70x) |
| TPOT P99 | 77.85 ms | 54.43 ms (0.70x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.4%; Total throughput 0.0% / 1.0%; Throughput / GPU 0.0% / 1.0%; TTFT P99 0.8% / 3.6%; TPOT P99 0.3% / 0.4%.

#### g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 4096 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt ee57f7b7a084baab3295e7015bd4136a43afc17e, packet 5fbe627c02af) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.0 GiB | 77.7 GiB (1.06x) |
| Total throughput | 15,004 tok/s | 20,547 tok/s (1.37x) |
| Throughput / GPU | 15,004 tok/s/GPU | 20,547 tok/s/GPU (1.37x) |
| TTFT P99 | 30,953.8 ms | 23,694.6 ms (0.77x) |
| TPOT P99 | 275.42 ms | 98.23 ms (0.36x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.1% / 0.1%; Throughput / GPU 0.1% / 0.1%; TTFT P99 0.6% / 0.1%; TPOT P99 0.4% / 0.0%.

### 15000/128, closed loop

Report `/opt/dlami/nvme/lava-tts/final3/report/st15k-fp8/comparison.md`. Arms: baseline `/opt/dlami/nvme/lava-tts/final2/res/st15k/vllm`, Infervisor `/opt/dlami/nvme/lava-tts/final3/res/st15k/fp8`. Gate `/opt/dlami/nvme/lava-tts/final3/gate/fp8/gates.json` (sha256 1e64f6058d79, packet 5fbe627c02af, PASS).

#### g32

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 15000 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 96 prompts, request rate inf / concurrency 32 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt ee57f7b7a084baab3295e7015bd4136a43afc17e, packet 5fbe627c02af) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.0 GiB | 77.7 GiB (1.06x) |
| Total throughput | 9,567 tok/s | 17,111 tok/s (1.79x) |
| Throughput / GPU | 9,567 tok/s/GPU | 17,111 tok/s/GPU (1.79x) |
| TTFT P99 | 45,534.3 ms | 25,888.7 ms (0.57x) |
| TPOT P99 | 382.92 ms | 114.50 ms (0.30x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.1%; Total throughput 0.0% / 0.3%; Throughput / GPU 0.0% / 0.3%; TTFT P99 0.1% / 0.8%; TPOT P99 0.0% / 0.3%.

#### g128

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | 15000 / 128 tokens | Same as baseline |
| Traffic / concurrency | greedy (T=0), 384 prompts, request rate inf / concurrency 128 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt ee57f7b7a084baab3295e7015bd4136a43afc17e, packet 5fbe627c02af) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.0 GiB | 77.7 GiB (1.06x) |
| Total throughput | 9,895 tok/s | 17,269 tok/s (1.75x) |
| Throughput / GPU | 9,895 tok/s/GPU | 17,269 tok/s/GPU (1.75x) |
| TTFT P99 | 186,793.9 ms | 106,551.3 ms (0.57x) |
| TPOT P99 | 825.66 ms | 115.74 ms (0.14x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.1% / 0.1%; Throughput / GPU 0.1% / 0.1%; TTFT P99 0.1% / 0.3%; TPOT P99 0.0% / 0.0%.

### agentic16k (`llm_grid.sh --agentic`)

Report `/opt/dlami/nvme/lava-tts/final3/report/agentic-fp8/comparison.md`. Arms: baseline `/opt/dlami/nvme/lava-tts/final2/res/agentic/vllm`, Infervisor `/opt/dlami/nvme/lava-tts/final3/res/agentic/fp8`. Gate `/opt/dlami/nvme/lava-tts/final3/gate/fp8/gates.json` (sha256 1e64f6058d79, packet 5fbe627c02af, PASS).

#### a32.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 32 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt ee57f7b7a084baab3295e7015bd4136a43afc17e, packet 5fbe627c02af) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.3 GiB | 77.7 GiB (1.06x) |
| Total throughput | 35,035 tok/s | 49,087 tok/s (1.40x) |
| Throughput / GPU | 35,035 tok/s/GPU | 49,087 tok/s/GPU (1.40x) |
| TTFT P99 | 3,818.5 ms | 2,186.8 ms (0.57x) |
| TPOT P99 | 82.04 ms | 54.76 ms (0.67x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.2%; Total throughput 0.2% / 4.1%; Throughput / GPU 0.2% / 4.1%; TTFT P99 4.2% / 7.7%; TPOT P99 3.6% / 1.4%.

#### a64.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 64 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt ee57f7b7a084baab3295e7015bd4136a43afc17e, packet 5fbe627c02af) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.3 GiB | 77.8 GiB (1.06x) |
| Total throughput | 42,380 tok/s | 55,530 tok/s (1.31x) |
| Throughput / GPU | 42,380 tok/s/GPU | 55,530 tok/s/GPU (1.31x) |
| TTFT P99 | 5,371.2 ms | 4,219.9 ms (0.79x) |
| TPOT P99 | 139.29 ms | 99.35 ms (0.71x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.0% / 2.0%; Throughput / GPU 0.0% / 2.0%; TTFT P99 1.1% / 0.3%; TPOT P99 0.2% / 2.6%.

#### a128.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | agentic 10 turns, system 1536, prompt grows to 15600 / 128 tokens per turn | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, closed-loop sessions, session header on / concurrency 128 sessions | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt ee57f7b7a084baab3295e7015bd4136a43afc17e, packet 5fbe627c02af) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 73.3 GiB | 77.8 GiB (1.06x) |
| Total throughput | 12,553 tok/s | 50,475 tok/s (4.02x) |
| Throughput / GPU | 12,553 tok/s/GPU | 50,475 tok/s/GPU (4.02x) |
| TTFT P99 | 82,042.6 ms | 12,820.7 ms (0.16x) |
| TPOT P99 | 892.14 ms | 210.47 ms (0.24x) |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.1% / 0.3%; Throughput / GPU 0.1% / 0.3%; TTFT P99 0.1% / 1.7%; TPOT P99 0.1% / 0.5%.

### open-loop production mix (`llm_grid.sh --prod`)

Report `/opt/dlami/nvme/lava-tts/final3/report/prod-fp8/comparison.md`. Arms: baseline `/opt/dlami/nvme/lava-tts/prodbench/res/vllm`, Infervisor `/opt/dlami/nvme/lava-tts/final3/res/prod/fp8`. Gate `/opt/dlami/nvme/lava-tts/final3/gate/fp8/gates.json` (sha256 1e64f6058d79, packet 5fbe627c02af, PASS).

#### q628.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | open-loop mix: 4 system prompts (lognormal median 1536), turns geometric mean 6 max 20, first message lognormal(1500, 1), tool output lognormal(700, 1), context cap 16384 / output lognormal(160, 0.7) in [16, 1024] tokens, ignore_eos | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, Poisson 0.628 sessions/s, think lognormal(5 s, 0.8) <= 60 s, 300 s (measured 75-275 s), session header on / open loop (achieved concurrency in the supplementary note), seeds 1099191/1107110 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt ee57f7b7a084baab3295e7015bd4136a43afc17e, packet 5fbe627c02af) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 72.9 GiB | 77.8 GiB (1.07x) |
| Total throughput | 18,611 tok/s | 20,106 tok/s (1.08x) |
| Throughput / GPU | 18,611 tok/s/GPU | 20,106 tok/s/GPU (1.08x) |
| TTFT P99 | 1,853.7 ms | 753.9 ms (0.41x) |
| TPOT P99 | 94.83 ms * | 56.91 ms (0.60x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 6.1% / 9.4%; Throughput / GPU 6.1% / 9.4%; TTFT P99 3.0% / 4.2%; TPOT P99 30.4% / 19.0%.

**FLAGGED: spread > 10%: Baseline TPOT P99 30.4%; Infervisor TPOT P99 19.0%.**

Supplementary (outside the strict table; goodput = requests meeting TTFT <= 2000 ms and TPOT <= 100 ms per second, measured window), Baseline / Infervisor: goodput 2.485 req/s / 2.675 req/s; SLO met 98.0% / 100.0%; requests 2.540 req/s / 2.675 req/s; mean in-flight 29.3 / 21.2; mean live sessions 42.7 / 35.5; TTFT P50 275.2 ms / 157.4 ms; TPOT P50 59.92 ms / 38.49 ms; cached prompt tokens 77.2% / 79.0%.

#### q771.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | open-loop mix: 4 system prompts (lognormal median 1536), turns geometric mean 6 max 20, first message lognormal(1500, 1), tool output lognormal(700, 1), context cap 16384 / output lognormal(160, 0.7) in [16, 1024] tokens, ignore_eos | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, Poisson 0.771 sessions/s, think lognormal(5 s, 0.8) <= 60 s, 300 s (measured 75-275 s), session header on / open loop (achieved concurrency in the supplementary note), seeds 1117924/1125843 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt ee57f7b7a084baab3295e7015bd4136a43afc17e, packet 5fbe627c02af) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 72.9 GiB | 77.8 GiB (1.07x) |
| Total throughput | 18,868 tok/s | 22,195 tok/s (1.18x) |
| Throughput / GPU | 18,868 tok/s/GPU | 22,195 tok/s/GPU (1.18x) |
| TTFT P99 | 3,790.5 ms * | 1,048.5 ms (0.28x) * |
| TPOT P99 | 238.80 ms * | 74.57 ms (0.31x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 2.8% / 7.5%; Throughput / GPU 2.8% / 7.5%; TTFT P99 64.3% / 62.7%; TPOT P99 36.6% / 31.9%.

**FLAGGED: spread > 10%: Baseline TTFT P99 64.3%; Infervisor TTFT P99 62.7%; Baseline TPOT P99 36.6%; Infervisor TPOT P99 31.9%.**

Supplementary (outside the strict table; goodput = requests meeting TTFT <= 2000 ms and TPOT <= 100 ms per second, measured window), Baseline / Infervisor: goodput 1.980 req/s / 3.103 req/s; SLO met 71.9% / 99.8%; requests 2.745 req/s / 3.107 req/s; mean in-flight 44.1 / 31.9; mean live sessions 58.1 / 48.4; TTFT P50 371.7 ms / 181.8 ms; TPOT P50 75.55 ms / 54.77 ms; cached prompt tokens 69.5% / 75.8%.

#### q987.g

Comparison: **MATCHED**; quality: **EQUIVALENT**

| Metric | Baseline | Infervisor |
|---|---|---|
| Model / version | gemma-4-12b-it-fp8 (config+weights 68f098a76c2b) | Same as baseline |
| Precision / quantization | weights compressed-tensors W8 float per-channel, A8 float dynamic per-token; KV cache fp8_per_token_head | Same as baseline |
| Input / output length | open-loop mix: 4 system prompts (lognormal median 1536), turns geometric mean 6 max 20, first message lognormal(1500, 1), tool output lognormal(700, 1), context cap 16384 / output lognormal(160, 0.7) in [16, 1024] tokens, ignore_eos | Same as baseline |
| Traffic / concurrency | greedy (T=0), chat API, Poisson 0.987 sessions/s, think lognormal(5 s, 0.8) <= 60 s, 300 s (measured 75-275 s), session header on / open loop (achieved concurrency in the supplementary note), seeds 1146220/1154139 | Same as baseline |
| GPU type & count | 1 x NVIDIA H100 80GB HBM3 | Same as baseline |
| Serving stack | vLLM 0.28.0 (--gpu-memory-utilization 0.9 --max-num-seqs 256 --dtype bfloat16 --max-model-len 16384 --attention-backend TRITON_ATTN --kv-cache-dtype fp8_per_token_head --enable-prefix-caching --max-num-batched-tokens 8192 --enable-prompt-tokens-details) | Infervisor (plowrt ee57f7b7a084baab3295e7015bd4136a43afc17e, packet 5fbe627c02af) |
| Output quality / correctness | FP32-reference gate peer (kl_mean 0.1277, top1_decisive 0.9823, needle_acc 1) | Equivalent |
| Peak GPU memory | 72.9 GiB | 77.8 GiB (1.07x) |
| Total throughput | 17,143 tok/s | 25,689 tok/s (1.50x) |
| Throughput / GPU | 17,143 tok/s/GPU | 25,689 tok/s/GPU (1.50x) |
| TTFT P99 | 6,135.4 ms * | 1,381.0 ms (0.23x) * |
| TPOT P99 | 420.65 ms * | 108.05 ms (0.26x) * |

Spread over 2 repeats, (max - min) / mean, Baseline / Infervisor: Peak GPU memory 0.0% / 0.0%; Total throughput 0.6% / 6.0%; Throughput / GPU 0.6% / 6.0%; TTFT P99 16.7% / 74.6%; TPOT P99 10.4% / 48.2%.

**FLAGGED: spread > 10%: Baseline TTFT P99 16.7%; Infervisor TTFT P99 74.6%; Baseline TPOT P99 10.4%; Infervisor TPOT P99 48.2%.**

Supplementary (outside the strict table; goodput = requests meeting TTFT <= 2000 ms and TPOT <= 100 ms per second, measured window), Baseline / Infervisor: goodput 0.748 req/s / 3.412 req/s; SLO met 26.3% / 87.9%; requests 2.835 req/s / 3.888 req/s; mean in-flight 90.9 / 55.3; mean live sessions 103.2 / 74.6; TTFT P50 1,070.0 ms / 268.5 ms; TPOT P50 188.53 ms / 68.82 ms; cached prompt tokens 48.4% / 70.8%.

## FP32-reference quality gate

The qualified packet `5fbe627c02af` passed this gate at plowrt `ee57f7b7`
(`/opt/dlami/nvme/lava-tts/final3/gate/fp8/gates.json`, 2026-10-05): KL mean 0.1048 vs vLLM
0.1277, KL p99 2.957 vs 3.188, top1_decisive 0.9783 vs 0.9823, cont_frac
0.627 vs 0.574, needle 1.0 vs 1.0. The `e93e0a64` rebuild `ab1c1d583f8d` also passes
(`gate/fp8b/`, KL mean 0.1053, top1_decisive 0.981). The calibration run below, which set the
thresholds, used the earlier grid-trial packet.

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

