# Gemma-4 BF16 comparison (Xeon 6975P-C, CPU engine)

This file holds only qualified wins and their evidence. Every matched serving row is in
[comparison.csv](comparison.csv) (one block per model, `campaign.py report` output with `model` and
`status` columns prepended). Status: E2B, E4B, 12B final; 26B-A4B gate miss; 31B pending its vLLM
baseline.

## Setup

- Infervisor: plowrt CPU engine (`--no-default-features --features cpu`), AMX tier, 96 threads,
  BF16 weights + BF16 KV, 2K context, 32 decode slots, prefix cache + session resume + cross-slot
  prefix share on. Recipes: `recipes/infervisor/gemma-4-*/xeon6-amx-bf16.toml`.
- Baseline: vLLM 0.30.0+cpu, `--dtype bfloat16 --max-model-len 2048 --max-num-seqs 256`, prefix
  caching on, `VLLM_CPU_OMP_THREADS_BIND=0-95`. Same vLLM client for both arms, greedy.
- One Intel Xeon 6975P-C (96 cores, SNC3, AMX-BF16), 377 GiB DDR5. Host state for every arm:
  `kernel.numa_balancing=0`, zero-latency CPU QoS held, THP madvise, no pseudo-lock.
- llama.cpp was dropped as a baseline mid-campaign (2026-10-06).
- Packets in the strict reports were emitted `--gpu rtx6000pro --arch sm_120a`. The recipes now
  name the CPU target (`--gpu xeon6975p --arch amx`, e83faf4e); its packets are byte-identical
  except the header target hash (bytes 28-31) on all five models, and serve identical greedy
  output (E4B, max |dlogprob| 0.0), so this evidence carries over.

## Qualified wins (MATCHED + EQUIVALENT, `campaign.py report` exit 0)

Closed loop, ISL 1000 / OSL 128, 2 repeats. Ratios are Infervisor / vLLM: throughput higher is
better; TTFT, TPOT and peak memory lower is better. `*` = repeat spread > 10% (direction only).

| Model | FP32 gate kl_mean (plow / vLLM) | Cell | Total throughput | TTFT P99 | TPOT P99 | Peak memory |
|---|---|---|---:|---:|---:|---:|
| E2B | 0.0005 / 0.0005 PASS | c1 | 3.20x | 0.55x * | 0.29x | 0.27x * |
| | | c8 | 1.90x | 0.94x | 0.51x | 0.29x * |
| | | c32 | 1.20x | **1.09x** | 0.82x | 0.33x * |
| E4B | 0.0006 / 0.0014 PASS | c1 | 3.28x | 0.61x * | 0.28x | 0.35x * |
| | | c8 | 2.02x | **1.11x** * | 0.48x | 0.37x * |
| | | c32 | 1.19x | **1.36x** | 0.83x | 0.42x |
| 12B | 0.0181 / 0.0215 PASS | c1 | 3.36x | 0.75x * | 0.28x | 0.53x * |
| | | c8 | 2.04x | **1.04x** * | 0.49x | 0.55x * |
| | | c32 | 1.69x | **1.19x** | 0.60x | 0.63x |

Bold = Infervisor loses. Burst TTFT P99 at c8/c32 (all prompts arriving at once) is bound by
prefill throughput: plow prefills ~310 ms per 1000-token prompt back to back, vLLM batches
(~255 ms each). Deferring decode during prefill (`PLOW_PF_DEFER_DECODE=1`) was measured and
rejected (E4B c32 TTFT P99 9.9 s vs 11.1 s, vLLM 8.2 s; median TTFT 4.6x worse).

## Real-world open loop (supplementary, not a strict report)

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

- 26B-A4B: MATCHED, NOT EQUIVALENT. FP32 gate kl_mean 0.1167 vs vLLM 0.0893 (limit 0.1136); one
  case (`nat-code-128`) carries it. Root cause: a bf16-noise near-tie at the MoE top-k cutoff for
  the first token after BOS cascades through later layers; forcing plow's routes into the HF fp32
  model reproduces plow's output, and plow's per-layer residual error is below HF bf16's up to the
  flip. Excluding that case: 0.083 vs 0.088. Performance (not a valid comparison): throughput
  2.1-5.7x, TTFT P99 0.40-0.93x, TPOT P99 0.16-0.48x. chat2k: plow 100% SLO at 0.12 / 0.25 / 0.4
  sessions/s (TTFT P99 757 / 724 / 1597 ms); the vLLM arm did not start (KV allocation on NUMA
  node 0 short of memory) and is pending.
- 31B: vLLM baseline pending (vLLM CPU places KV on one NUMA node; 59 GiB weights + KV exceed the
  node with the current tmpfs model layout).

## Evidence (campaign scratch)

- Strict reports: `/tmp/g4c/final/<model>/report-vllm/` (comparison.md/.csv/.json), arms in
  `/tmp/g4c/final/<model>/{plow,vllm}`, FP32 gates in `/tmp/g4c/final/<model>/gate/gates.json`.
- Open loop: `/tmp/g4c/final/<model>/prod-{plow,vllm}`, table `/tmp/g4c/final/prod_table.share.md`.
- CPU-target packet A/B: `/tmp/g4c/targetab/<model>/{old,new}`.
