# Qwen3.8-27B comparison (1x H100 SXM5, vLLM 0.28)

Qualified wins only: `campaign.py report` MATCHED + EQUIVALENT, Infervisor ahead of vLLM on the
metric. Every rendered row, won or lost, is in [comparison.csv](comparison.csv) (`campaign_source`
`q38-20261011-grid43-bf16` for the current packet; `q38-20261011-grid3-bf16` kept for the previous one).

- Infervisor BF16: plowrt `8d25a203`, packet `a6d706647465f718` (pairing `0xc3f74a8e97e4d467`) from
  [`sm90a-h100-tp1.toml`](../../../../recipes/infervisor/qwen3.8-27b/sm90a-h100-tp1.toml)
  (`campaign.py build`, max_ctx 16384, decode ladder 1..48, live-KV allocation, **packed prefill
  for the hybrid chain** (concurrent prompts share one prefill launch, masked padding, compact
  terminal: the first token comes from the prefill launch), prefix cache off: the recurrent GDN
  state has no prefix snapshot, so plowrt refuses it). Serve defaults otherwise (objective `auto`;
  the token-batch route has no recurrent step and is declined at load).
- Baseline: vLLM 0.28.0, `Qwen/Qwen3.8-27B` BF16 weights and KV, `--max-model-len 16384
  --gpu-memory-utilization 0.9 --max-num-seqs 256 --enable-prefix-caching`, same client (`vllm bench
  serve` from `/opt/pytorch`), 2 repeats per cell, one server per arm. 1000-token prompts, 128 output
  tokens, greedy (g) and sampled T=1 top-p 0.95 (s) at concurrency 1 / 8 / 32 / 64 / 128.

Quality: FP32-reference gate PASS at c1, c8 and c32, each against the vLLM 0.28 capture at the
same concurrency (`gate43m/gates.json`; the verdict gates kl_mean, 1%-trimmed mean, p95, the count
over 0.1 nat, decisive top-1, continuation and decisive free-run flips, needles; p99 reported):

| conc | kl_mean plow / vLLM | trimmed | p95 | over 0.1 nat | top1_decisive | continuation | needles |
|---|---:|---:|---:|---:|---:|---:|---:|
| c1 | 0.0016 / 0.0007 | 0.00063 / 0.00044 | 0.0033 / 0.0025 | 4 / 0 | 1.0 / 1.0 | 0.852 / 0.845 | 1.0 / 1.0 |
| c8 | 0.0014 / 0.0007 | 0.00065 / 0.00046 | 0.0036 / 0.0024 | 3 / 0 | 1.0 / 1.0 | 0.817 / 0.826 | 1.0 / 1.0 |
| c32 | 0.0019 / 0.0007 | 0.00070 / 0.00047 | 0.0042 / 0.0028 | 6 / 1 | 0.999 / 1.0 | 0.845 / 0.854 | 1.0 / 1.0 |

`campaign.py report`, all 10 cells MATCHED + EQUIVALENT, 2 repeats (ratios Infervisor / vLLM):

| Cell | Total tok/s vLLM | Infervisor | Ratio | TTFT p99 | TPOT p99 | Peak GiB vLLM / Inf |
|---|---:|---:|---:|---:|---:|---:|
| 1000/128 c1 greedy | 415 | 360 | 0.87x | **0.48x*** | 1.19x | 72.1 / 66.9* |
| 1000/128 c1 sampled | 412 | 358 | 0.87x | **0.62x** | 1.19x | 72.1 / 66.7* |
| 1000/128 c8 greedy | 2,336* | 2,065 | 0.88x | 0.39x* | 1.50x | 72.3 / 67.6* |
| 1000/128 c8 sampled | 2,616 | 2,064 | 0.79x | 1.24x | 1.45x | 72.3 / 67.5* |
| 1000/128 c32 greedy | 6,235 | 4,150 | 0.67x | 1.27x | 1.73x | 72.7 / 70.5 |
| 1000/128 c32 sampled | 6,192 | 4,146 | 0.67x | 1.28x | 1.64x | 72.7 / 70.5 |
| 1000/128 c64 greedy | 8,059 | 4,638 | 0.58x | 2.46x | 1.45x | 72.9 / 73.0 |
| 1000/128 c64 sampled | 8,012 | 4,627 | 0.58x | 2.48x | 1.44x | 72.9 / 73.0 |
| 1000/128 c128 greedy | 7,739 | 4,668 | 0.60x | 1.80x | 1.18x | 73.2 / 73.0 |
| 1000/128 c128 sampled | 7,757 | 4,654 | 0.60x | 1.81x | 1.17x | 73.2 / 73.0 |

`*` = repeat spread > 10% (FLAGGED; direction only).

Qualified wins (bold): TTFT P99 at c1 (greedy flagged by the baseline's 42% spread; sampled clean,
114 vs 184 ms: one padded prefill bucket launch per prompt, the first token from that launch). Peak
GPU memory is lower at c1..c8 but the Infervisor repeats spread 14-17% (live-KV commit timing), so
it is recorded, not claimed. The c8 greedy TTFT P99 (0.39x) sits on a 138% baseline spread.

Lost (in the CSV): total throughput and TPOT P99 in every cell, TTFT P99 from c8 sampled up.

Versus the previous packet `949474c1acaffbd9` (`q38-20261011-grid3-bf16`, one prompt per prefill
launch): packed prefill lifted total throughput at c8 / c32 / c64 / c128 by 4 / 6 / 5 / 3% (0.85 ->
0.88x, 0.63 -> 0.67x, 0.55 -> 0.58x, 0.58 -> 0.60x), cut c8 sampled TTFT P99 1.44 -> 1.24x, c32
TTFT P99 1.68 -> 1.27x (4467 -> 3371 ms) and c32 TPOT P99 1.86 -> 1.73x (68.7 -> 63.9 ms); c1 is
unchanged (first token from the prefill launch in both). The remaining gap is the decode ladder:

- c1: decode step 23.8 ms vs vLLM TPOT 20.0 ms. The 64-layer step is weight-bandwidth bound
  (54 GB of BF16 weights per token = 16 ms at 3.35 TB/s); the GEMV walk reaches 53-82% of HBM
  bandwidth per projection and the 48 GDN steps add ~3 ms.
- c8..c48 decode ticks: 27.5 / 31.5 / 40.4 / 50.2 ms at B8 / 16 / 32 / 48 (`step_bench`, ctx 1K).
  At B32 the projections run at 55-68% of HBM bandwidth (23.6 ms against a 16 ms weight stream),
  the GDN step at 44% (6.5 ms) and the hd256 FlashDecode at 14% (4.6 ms plus a 1.2 ms merge). vLLM
  decodes at 25.1 ms per step at c32.
- c64 / c128: the ladder tops at 48 slots (the 64-slot GDN state, 151 MB per slot, leaves the planner
  0.5 GiB short on the 80 GB part), so requests above 48 queue: TTFT p99 13.0 / 26.3 s.
- Numerics lead (plan 18:50): plow sits at KL 0.30 against FP32 at `nat-docs-1024` token 30 (FP32
  decisive by 2 nats) on every build, 0.8-1.3 with the top-1 flipped in some c8 runs; vLLM 0.0004-0.07,
  HF bf16 0.16. Within the gate, but plow's bf16 decode path is further from FP32 than vLLM's at the
  rounding-sensitive positions.

Next levers, in order of served effect: the wide-rung GEMV (B32 23.6 ms vs a 16 ms weight stream),
the GDN decode step and FlashDecode bodies, decode riders on prefill ticks for the recurrent chain,
slots > 48.

Evidence (campaign scratch, not in the repo): `/opt/dlami/nvme/lava-tts/qwen38/grid43full/{plow,report}`,
`grid1/vllm` (baseline arm), `gate43m/gates.json` (c1/c8/c32), `fp32gate/vllm-c{1,8,32}.json`
(peers), `grid3/` (previous packet), `steps35b/` and `steps36/` (ladder), `probe35/` (prefill segment
timing), `sweep32/roof.b32.md` (B32 roofline). Log: `plans/qwen38-27b-h100.md`.
