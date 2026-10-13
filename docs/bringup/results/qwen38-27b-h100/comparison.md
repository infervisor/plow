# Qwen3.8-27B comparison (1x H100 SXM5, vLLM 0.28)

Qualified wins only: `campaign.py report` MATCHED + EQUIVALENT, Infervisor ahead of vLLM on the
metric. Every rendered row, won or lost, is in [comparison.csv](comparison.csv) (`campaign_source`
`q38-20261013-grid113-bf16` for the shipped packet; `q38-20261011-grid43-bf16` and
`q38-20261011-grid3-bf16` kept for the previous ones).

- Infervisor BF16: plowrt `6af0b02c`, packet `3fb7cefaf150a710` from
  [`sm90a-h100-tp1.toml`](../../../../recipes/infervisor/qwen3.8-27b/sm90a-h100-tp1.toml)
  (`campaign.py build` at the merge pin `b44ca43f`, hash-identical to the qualified build106 packet;
  max_ctx 16384, decode ladder 1..48, live-KV allocation, packed prefill for the hybrid chain with a
  compact terminal, generated hd256 prefill-attention and flash-decode objects, cuBLASLt decode
  projections on rungs >= 16, decode rows riding prefill ticks, chain graph pieces, the compact
  terminal lm_head on cuBLASLt, prefix cache off: the recurrent GDN state has no prefix snapshot).
- Baseline: vLLM 0.28.0, `Qwen/Qwen3.8-27B` BF16 weights and KV, `--max-model-len 16384
  --gpu-memory-utilization 0.9 --max-num-seqs 256 --enable-prefix-caching`, same client (`vllm bench
  serve` from `/opt/pytorch`), 2 repeats per cell, one server per arm. 1000-token prompts, 128 output
  tokens, greedy (g) and sampled T=1 top-p 0.95 (s) at concurrency 1 / 8 / 32 / 64 / 128.

Quality: FP32-reference gate PASS at c1, c8 and c32, each against the vLLM 0.28 capture at the
same concurrency (`gate111/gates.json`; the verdict gates kl_mean, 1%-trimmed mean, p95, the count
over 0.1 nat, decisive top-1, continuation and decisive free-run flips, needles; p99 reported):

| conc | kl_mean plow / vLLM | trimmed | p95 | over 0.1 nat | top1_decisive | continuation | needles |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | 0.00078 / 0.00051 | 0.00045 / 0.00034 | 0.0025 / 0.0017 | 1 / 0 | 1.0 / 1.0 | 0.865 / 0.861 | 1.0 / 1.0 |
| c8 | 0.00142 / 0.00040 | 0.00048 / 0.00035 | 0.0026 / 0.0016 | 1 / 0 | 0.999 / 1.0 | 0.845 / 0.866 | 1.0 / 1.0 |
| c32 | 0.00123 / 0.00041 | 0.00049 / 0.00036 | 0.0026 / 0.0019 | 1 / 0 | 0.999 / 1.0 | 0.861 / 0.870 | 1.0 / 1.0 |

Decisive free-run flips: 45 of 46 cases at c8 and c32 (vLLM 46), the one being `nat-docs-1024`
position 30, the swing position documented in the recipe; the recipe's `cont_decisive_drop_max`
margin (0.022, one case) records it.

`campaign.py report`, all 10 cells MATCHED + EQUIVALENT, 2 repeats (ratios Infervisor / vLLM):

| Cell | Total tok/s vLLM | Infervisor | Ratio | TTFT p99 | TPOT p99 | Peak GiB vLLM / Inf |
|---|---:|---:|---:|---:|---:|---:|
| 1000/128 c1 greedy | 415 | 387 | 0.93x | **0.36x*** | 1.12x | 72.1 / **62.7** |
| 1000/128 c1 sampled | 412 | 385 | 0.93x | **0.48x** | 1.12x | 72.1 / **62.7** |
| 1000/128 c8 greedy | 2,336* | 2,223 | 0.95x | **0.32x*** | 1.41x | 72.3 / **63.1** |
| 1000/128 c8 sampled | 2,616 | 2,216 | 0.85x | 1.01x | 1.40x | 72.3 / **63.1** |
| 1000/128 c32 greedy | 6,235 | 5,965 | 0.96x | 0.99x | 1.25x | 72.7 / **65.4** |
| 1000/128 c32 sampled | 6,192 | 5,931 | 0.96x | 0.99x | 1.25x | 72.7 / **65.4** |
| 1000/128 c64 greedy | 8,059 | 7,056 | 0.88x | 1.69x | **0.95x** | 72.9 / **66.9** |
| 1000/128 c64 sampled | 8,012 | 7,067 | 0.88x | 1.70x | **0.94x** | 72.9 / **66.9** |
| 1000/128 c128 greedy | 7,739 | 7,106 | 0.92x | 1.21x | **0.77x** | 73.2 / **66.9** |
| 1000/128 c128 sampled | 7,757 | 7,070 | 0.91x | 1.22x | **0.77x** | 73.2 / **66.9** |

`*` = repeat spread > 10% (FLAGGED; direction only).

Qualified wins (bold): TTFT P99 at c1 (sampled 87.9 vs 184.1 ms; greedy 87.4 vs 242.4 ms sits on
the baseline's 42% spread) and c8 greedy (709 vs 2217 ms, on a 138% baseline spread); TPOT P99 at
c64 (59.0 vs 62.0 ms) and c128 (58.8 vs 76.4 ms); peak GPU memory in every cell (0.87-0.92x,
Infervisor repeats within 5.4%).

Lost (in the CSV): total throughput in every cell (0.85-0.96x), TPOT P99 at c1..c32 (1.12-1.41x),
TTFT P99 at c64 and c128 (1.7x and 1.2x: the decode ladder tops at 48 slots, requests above 48
queue behind the ladder); c8 sampled and c32 TTFT P99 are parity (0.99-1.01x).

Versus the previous qualified packet `a6d706647465f718` (`q38-20261011-grid43-bf16`): total
throughput 0.87 -> 0.93x at c1, 0.88 -> 0.95x at c8 greedy, 0.67 -> 0.96x at c32, 0.58 -> 0.88x at
c64 and 0.60 -> 0.92x at c128; TPOT P99 1.73 -> 1.25x at c32, 1.45 -> 0.95x at c64 and 1.18 ->
0.77x at c128; TTFT P99 2.46 -> 1.69x at c64 and 1.80 -> 1.21x at c128; the c1..c8 peak-memory
spread (14-17%, live-KV commit timing) is gone. The remaining gap at 1000/128 is the decode tick at
B8..B32 (the weight stream of a 27B BF16 model per rung under the 700 W board cap; the campaign's
rung-by-rung evidence is in `plans/qwen38-27b-h100.md`).

Long-context serving (4K and 15K prompts, 128 out, c16 / c32 / c64) was measured on the same packet
and client as a serving grid (grid111, plan 2026-10-13 17:50), not as a `campaign.py report`, so it
is recorded in the plan and not claimed here.

Evidence (campaign scratch, not in the repo): `/opt/dlami/nvme/lava-tts/qwen38/grid113/{plow,report}`,
`grid1/vllm` (baseline arm), `gate111/gates.json` (c1/c8/c32 on the served packet),
`fp32gate/vllm-c{1,8,32}.json` (peers), `ab113/` (build113 vs build106 step_bench digests and ride
lines), `grid43full/` (previous packet). Log: `plans/qwen38-27b-h100.md`.
