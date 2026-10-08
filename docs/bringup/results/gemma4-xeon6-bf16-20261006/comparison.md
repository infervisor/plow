# Gemma-4 BF16 comparison (Xeon 6975P-C, CPU engine)

This file holds only qualified wins and their evidence. Every matched serving row is in
[comparison.csv](comparison.csv) (one block per model, `campaign.py report` output with `model` and
`status` columns prepended). Status: E2B, E4B, 12B, 26B-A4B final (plowrt 59a36e3d); 31B pending
its vLLM baseline.

## Setup

- Infervisor: plowrt CPU engine (`--no-default-features --features cpu`), AMX tier, 96 threads,
  BF16 weights + BF16 KV, 2K context, 32 (E2B, 26B) / 64 (E4B, 12B) decode slots, packed prefill,
  AMX `FLASH_PREFILL` (26B: P kept f32-accurate through the TMUL PV, `PLOW_CPU_AMX_ATTN_SPLIT_P=1`),
  CPU token batch (decode rows ride in the packed prefill launch), prefix cache + session resume +
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
| E2B | 0.0004 / 0.0005 PASS | c1 | 3.24x | 0.53x * | 0.29x | 0.27x * |
| | | c8 | 2.03x | 0.78x | 0.48x | 0.29x * |
| | | c32 | 1.36x | 0.82x | 0.73x | 0.34x * |
| E4B | 0.0007 / 0.0014 PASS | c1 | 3.25x | 0.54x * | 0.29x | 0.35x * |
| | | c8 | 2.22x | 0.78x * | 0.43x | 0.37x * |
| | | c32 | 1.50x | 0.85x | 0.66x | 0.43x |
| 12B | 0.0125 / 0.0215 PASS | c1 | 3.31x | 0.68x * | 0.28x | 0.53x * |
| | | c8 | 2.28x | 0.74x * | 0.44x | 0.55x * |
| | | c32 | 2.08x | 0.78x * | 0.49x | 0.63x |
| 26B-A4B | 0.1014 / 0.0893 PASS | c1 | 5.78x | 0.36x * | 0.16x | 0.72x * |
| | | c8 | 3.62x | 0.57x * | 0.28x | 0.73x * |
| | | c32 | 2.42x | 0.67x * | 0.41x | 0.77x |
Infervisor wins every row. Against the b6924229 reports, burst TTFT P99 at c32 improved from 0.87x / 0.91x / 0.77x
to 0.82x / 0.85x / 0.67x (E2B / E4B / 26B-A4B); this binary adds the CPU token batch and, for 26B, TMUL attention.
26B-A4B's gate (0.1014, limit 0.1116) is the split-P recipe; its earlier
AVX-512-attention gate was 0.0950. Burst TTFT P99 at c8/c32 (all prompts arriving at once) is bound by
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

## Agentic sessions to 16K context (supplementary, not a strict report; plowrt 59a36e3d)

Open-loop session arrivals, closed loop inside each session (the next turn waits for the reply plus
think time):
- 4 apps; shared system prompt with a median of 1.5K tokens; first user turn with a median of 1.5K
  tokens; tool outputs with a median of 700 tokens.
- 6 turns mean (20 max), so the conversation grows to the 16,384-token context limit. Measured prompts
  average 5.3-8.5K tokens and reach 15.8K.
- Output length: median 160 tokens (16-512); think time: median 5 s.
- 600 s per rate. Requests that start in [120 s, 540 s] are measured.
- SLO: TTFT ≤ 10 s, and TPOT ≤ 150 / 200 / 400 / 400 ms for E2B / E4B / 12B / 26B-A4B.
- Greedy decoding, prefix caching on for both servers, identical arrival plan.
- `/tmp/g4c/prod16k.sh`, `agentic_turns.py` defaults.

Server configuration:
- **vLLM:** `--max-model-len 16384`, `VLLM_CPU_KVCACHE_SPACE` = 8 / 16 / 24 / 16 GiB.
- **Infervisor:** 16K emits of each recipe:
  - `--max-ctx 16384`;
  - decode ladder up to 64 slots (E2B) or 32 slots (others);
  - `PLOW_SESSION_SLACK` = 64 (E2B) or 32;
  - otherwise each recipe's emit and serve env.
  - Packets 42aadb1b2d38 / db30257effcd / f00484df6d4f / 58f0a9eced76.

The runtime is the same binary as the strict reports (sha256 9f1cef09), which includes:
- the CPU token batch: decode rows ride in the packed prefill launch;
- shortest-remaining-first prompt share;
- session resume.

Ratios are Infervisor / vLLM: goodput higher is better; latencies lower is better. **Bold** marks a
cell Infervisor loses.

| Model | Sessions/s | Goodput req/s | SLO % | TTFT P50 ms | TTFT P99 ms | TTFT P99 ms, same requests | TPOT P99 ms | Output tok/s | Prefix hit (plow) |
|---|---:|---|---|---|---|---|---|---|---:|
| E2B | 0.1 | 0.490 / 0.490 (1.00x) | 100 / 100 | 239 / 858 (0.28x) | 2,500 / 3,134 (0.80x) | 2,508 / 3,136 (0.80x) (n=198) | 72 / 122 (0.59x) | 92 / 92 | 76% |
|  | 0.2 | 0.893 / 0.607 (1.47x) | 100 / 73 | 272 / 962 (0.28x) | 1,765 / 2,553 (0.69x) | 1,760 / 3,035 (0.58x) (n=337) | 110 / 238 (0.46x) | 166 / 156 | 78% |
|  | 0.3 | 1.067 / 0.060 (17.92x) | 78 / 5 | 349 / 1,048 (0.33x) | 2,147 / 4,074 (0.53x) | 2,209 / 4,078 (0.54x) (n=483) | 191 / 321 (0.59x) | 263 / 227 | 77% |
| E4B | 0.05 | 0.176 / 0.169 (1.04x) | 100 / 100 | 481 / 1,288 (0.37x) | **4,043 / 3,352 (1.21x)** | 4,111 / 4,464 (0.92x) (n=68) | 77 / 169 (0.46x) | 39 / 39 | 64% |
|  | 0.1 | 0.483 / 0.205 (2.36x) | 100 / 47 | 480 / 1,320 (0.36x) | 3,486 / 4,774 (0.73x) | 3,419 / 4,938 (0.69x) (n=171) | 128 / 283 (0.45x) | 90 / 80 | 75% |
|  | 0.15 | 0.652 / 0.021 (30.44x) | 100 / 4 | 494 / 1,502 (0.33x) | 2,444 / 6,134 (0.40x) | 2,740 / 6,105 (0.45x) (n=209) | 155 / 341 (0.45x) | 132 / 109 | 78% |
| 12B | 0.02 | 0.100 / 0.079 (1.27x) | 100 / 100 | 1,074 / 2,641 (0.41x) | 3,748 / 6,882 (0.54x) | 3,222 / 6,790 (0.47x) (n=33) | 123 / 287 (0.43x) | 19 / 15 | 76% |
|  | 0.04 | 0.155 / 0.133 (1.16x) | 100 / 97 | 1,186 / 2,443 (0.49x) | 5,386 / 6,853 (0.79x) | 5,775 / 8,635 (0.67x) (n=55) | 126 / 362 (0.35x) | 32 / 26 | 65% |
|  | 0.06 | 0.219 / 0.171 (1.28x) | 100 / 87 | 1,080 / 2,601 (0.42x) | 5,430 / 11,914 (0.46x) | 6,410 / 11,944 (0.54x) (n=72) | 196 / 488 (0.40x) | 44 / 38 | 73% |
| 26B-A4B | 0.03 | 0.160 / 0.140 (1.14x) | 100 / 100 | 490 / 1,717 (0.29x) | 3,835 / 4,891 (0.78x) | 3,910 / 5,940 (0.66x) (n=51) | 83 / 254 (0.33x) | 35 / 32 | 73% |
|  | 0.06 | 0.210 / 0.098 (2.15x) | 100 / 54 | 544 / 2,232 (0.24x) | 4,419 / 7,993 (0.55x) | 5,316 / 8,037 (0.66x) (n=58) | 121 / 480 (0.25x) | 42 / 35 | 66% |
|  | 0.09 | 0.374 / 0.129 (2.91x) | 100 / 48 | 564 / 2,178 (0.26x) | 3,149 / 7,574 (0.42x) | 3,284 / 7,639 (0.43x) (n=99) | 158 / 624 (0.25x) | 72 / 51 | 72% |

How to read the TTFT P99 columns:
- Each session waits for its own replies, so the faster server reaches later turns inside the window.
  Those turns have longer prompts. Infervisor measured more requests and more long prompts in every
  cell (for example 26B-A4B 0.09: 157 vs 112 requests, 21 vs 6 prompts over 12K tokens).
- "TTFT P99" is each harness's own window.
- "Same requests" is the P99 over the (session, turn) pairs both servers measured
  (`/tmp/g4c/matched.py`).

Results:
- Infervisor wins every cell on TTFT P50, same-request TTFT P99 and TPOT P99.
- Goodput: equal or higher in every cell. The two equal cells are the lowest rates, where both
  servers meet every SLO (E2B 0.1: 1.00x; E4B 0.05: 1.04x).
- TTFT P50 is 0.24-0.49x; TPOT P99 is 0.25-0.59x.
- At the highest rate, Infervisor holds 78-100% SLO attainment and vLLM 4-87%.
- The one loss is the E4B 0.05 harness TTFT P99, 4.04 s vs 3.35 s:
  - Infervisor's window contains a cold 15.8K-token prompt (about 4.0 s of prefill).
  - On the 68 requests both servers measured, Infervisor is faster (4.11 s vs 4.46 s).
  - 15.5% of a 15.8K-token E4B prefill is the norm kernels (`v_rmsnorm`, `v_norm_residual`), which
    are limited by moving data between cores. Fusing the norms into the GEMMs is the open lever.

E4B 0.05 with a 1,500 s window (1,320 s measured, same arrival plan; `prod-{plow,vllm}-long3`):

| Server | Requests | TTFT P99 | SLO attainment | TPOT P99 |
|---|---:|---:|---:|---:|
| Infervisor | 267 | 3,393 ms | 100% | 68 ms |
| vLLM | 248 | 7,895 ms | 79% | 249 ms |

Read this as supplementary, not as a replacement for the 600 s cell:
- **Infervisor reproduces its 600 s run.** Over requests started in 120-540 s: TTFT P99 4,086 vs 4,099 ms,
  TPOT P50 26 vs 26 ms.
- **vLLM does not.** Over the same window with the same arrivals, it is slower than its 600 s baseline
  from 10-07: TPOT P50 153 vs 103 ms.
- **The host differed:** before the run, tmpfs pages were interleaved over the NUMA nodes and node 0 was
  freed for vLLM's KV cache. The cause of vLLM's slowdown was not isolated.

## Not qualified

- 26B-A4B with bf16-rounded P in the AMX attention PV: FP32 gate kl_mean 0.118-0.128 (limit 0.112);
  the model's MoE top-k near-ties flip with the summation order. Superseded by split P (above: PASS).

- 31B: vLLM baseline pending (vLLM CPU places KV on one NUMA node; 59 GiB weights + KV exceed the
  node with the current tmpfs model layout).

## Evidence (campaign scratch)

- Strict reports: `/tmp/g4c/final/<model>/report-vllm/` (comparison.md/.csv/.json), arms in
  `/tmp/g4c/final/<model>/{plow,vllm}`, FP32 gates in `/tmp/g4c/final/<model>/gate/gates.json`.
- Open loop: `/tmp/g4c/final/<model>/prod-{plow,vllm}`, table `/tmp/g4c/final/prod_table.share.md`.
- 16K agentic: `/tmp/g4c/final16k/<model>/prod-{plow-final,vllm}`, tables `/tmp/g4c/table16k_doc.py`
  (this file) and `/tmp/g4c/table16k.py`, same-request P99 `/tmp/g4c/matched.py`.
- Previous strict reports (b6924229): `/tmp/g4c/final/<model>/{plow,gate,report-vllm}.b6924229`.
- CPU-target packet A/B: `/tmp/g4c/targetab/<model>/{old,new}`.
