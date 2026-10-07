# Gemma-4 BF16 comparison (Xeon 6975P-C, CPU engine)

This file holds only qualified wins and their evidence. Every matched serving row is in
[comparison.csv](comparison.csv) (one block per model, `campaign.py report` output with `model` and
`status` columns prepended). Status: E2B, E4B, 12B, 26B-A4B final (plowrt b6924229); 31B pending
its vLLM baseline.

## Setup

- Infervisor: plowrt CPU engine (`--no-default-features --features cpu`), AMX tier, 96 threads,
  BF16 weights + BF16 KV, 2K context, 32 (E2B, 26B) / 64 (E4B, 12B) decode slots, packed prefill,
  AMX `FLASH_PREFILL` (26B: AVX-512, `PLOW_CPU_AMX_ATTN=0`), prefix cache + session resume +
  cross-slot prefix share on. Recipes: `recipes/infervisor/gemma-4-*/xeon6-amx-bf16.toml`.
- Baseline: vLLM 0.30.0+cpu, `--dtype bfloat16 --max-model-len 2048 --max-num-seqs 256`, prefix
  caching on, `VLLM_CPU_OMP_THREADS_BIND=0-95`. Same vLLM client for both arms, greedy.
- One Intel Xeon 6975P-C (96 cores, SNC3, AMX-BF16), 377 GiB DDR5. Host state for every arm:
  `kernel.numa_balancing=0`, zero-latency CPU QoS held, THP madvise, no pseudo-lock.
- llama.cpp was dropped as a baseline mid-campaign (2026-10-06).
- Packets: `campaign.py build` of each recipe (`--gpu xeon6975p --arch amx`); hashes in each report.

## Qualified wins (MATCHED + EQUIVALENT, `campaign.py report` exit 0)

Closed loop, ISL 1000 / OSL 128, 2 repeats. Ratios are Infervisor / vLLM: throughput higher is
better; TTFT, TPOT and peak memory lower is better. `*` = repeat spread > 10% (direction only).

| Model | FP32 gate kl_mean (plow / vLLM) | Cell | Total throughput | TTFT P99 | TPOT P99 | Peak memory |
|---|---|---|---:|---:|---:|---:|
| E2B | 0.0004 / 0.0005 PASS | c1 | 3.25x | 0.52x * | 0.29x | 0.27x * |
| | | c8 | 2.02x | 0.81x | 0.48x | 0.29x * |
| | | c32 | 1.34x | 0.87x | 0.74x | 0.34x * |
| E4B | 0.0005 / 0.0014 PASS | c1 | 3.26x | 0.55x | 0.29x | 0.35x * |
| | | c8 | 2.24x | 0.82x | 0.43x | 0.37x * |
| | | c32 | 1.45x | 0.91x | 0.69x | 0.43x |
| 12B | 0.0110 / 0.0215 PASS | c1 | 3.41x | 0.67x * | 0.28x | 0.53x * |
| | | c8 | 2.26x | 0.77x * | 0.44x | 0.55x * |
| | | c32 | 2.04x | 0.82x * | 0.50x | 0.63x |
| 26B-A4B | 0.0950 / 0.0893 PASS | c1 | 5.78x | 0.38x * | 0.16x | 0.72x * |
| | | c8 | 3.52x | 0.65x * | 0.28x | 0.73x * |
| | | c32 | 2.28x | 0.77x * | 0.44x | 0.77x |

Infervisor wins every row. Burst TTFT P99 at c8/c32 (all prompts arriving at once) is bound by
prefill throughput; packed prefill (2048-row launches), AMX attention and the balanced
W-stationary GEMM partition took E4B's 2048-row prefill from 0.74 s (sequential) to 0.42 s.
Deferring decode during prefill (`PLOW_PF_DEFER_DECODE=1`) was measured and rejected (E4B c32 TTFT
P99 9.9 s vs 11.1 s, vLLM 8.2 s; median TTFT 4.6x worse).

## Real-world open loop (supplementary, not a strict report; plowrt 8e5f0039, before packed prefill)

chat2k: multi-turn agentic sessions (4 apps, ~400-token system prompts, 4 turns mean, tool
outputs, think time), Poisson session arrivals, 420 s per rate, SLO TTFT <= 3 s and a per-model
TPOT bound. Identical arrival plan on both servers. Ratios Infervisor / vLLM.

| Model | sessions/s | Goodput | SLO % (plow / vLLM) | TTFT P99 | TPOT P99 | Prefix hit (plow / vLLM) |
|---|---:|---:|---:|---:|---:|---:|
| E2B | 0.5 | 1.01x | 100 / 100 | 0.52x | 0.42x | 48 / 53 % |
| | 1.0 | 1.81x | 100 / 56 | **1.19x** | 0.41x | 63 / 59 % |
| | 1.5 | 19.7x | 100 / 5 | 0.57x | 0.34x | 65 / 59 % |
| E4B | 0.25 | 1.00x | 100 / 100 | 0.48x | 0.44x | 61 / 58 % |
| | 0.5 | 1.27x | 100 / 78 | 0.56x | 0.57x | 62 / 52 % |
| | 0.75 | 1.353 vs 0 req/s | 79 / 0 | **5.28x** | 0.41x | 51 / 53 % |
| 12B | 0.1 | 0.98x | 100 / 100 | 0.66x | 0.44x | 53 / 45 % |
| | 0.2 | 1.30x | 100 / 81 | 0.86x | 0.34x | 56 / 30 % |
| | 0.3 | 238x | 96 / 0.5 | **1.14x** | 0.51x | 51 / 29 % |

E4B 0.75 is past both servers' capacity (35.6 concurrent sessions > 32 plow slots): plow keeps 79%
of requests inside the SLO, vLLM none, but plow's queueing tail sets TTFT P99. A 64-slot decode
ladder was measured there and rejected (TTFT P99 487 s, throughput collapsed).

## Not qualified

- 26B-A4B with AMX attention: FP32 gate kl_mean 0.118-0.128 (limit 0.112) at f64-measured attention
  accuracy equal to the AVX-512 kernel; the model's MoE top-k near-ties flip with the summation
  order. Its recipe keeps AVX-512 attention (above: PASS).

- 31B: vLLM baseline pending (vLLM CPU places KV on one NUMA node; 59 GiB weights + KV exceed the
  node with the current tmpfs model layout).

## Evidence (campaign scratch)

- Strict reports: `/tmp/g4c/final/<model>/report-vllm/` (comparison.md/.csv/.json), arms in
  `/tmp/g4c/final/<model>/{plow,vllm}`, FP32 gates in `/tmp/g4c/final/<model>/gate/gates.json`.
- Open loop: `/tmp/g4c/final/<model>/prod-{plow,vllm}`, table `/tmp/g4c/final/prod_table.share.md`.
- CPU-target packet A/B: `/tmp/g4c/targetab/<model>/{old,new}`.
