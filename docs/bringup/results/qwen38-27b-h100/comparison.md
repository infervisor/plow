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

## 2026-10-27: native vs cuBLASLt route sweep (Qwen3.8-27B BF16 and FP8)

Protocol: `step_bench` on one H100, decode 48 steps `--spread`, prefill `--packed-prefill --pf-reps 3`
(best timed rep), two passes ABAB (passes agree within 1%), ms = mean of the passes. Current arm =
the abi2 packet of the recipe at 21ab0d7d. Alternative = either the same packet with one runtime
knob set, or a re-emit of the recipe with the emit knobs named in the table title changed (Qwen:
the decode threshold moves together with `PLOW_QWEN_GDN_ISOLATE`, which plowrt requires; the FP8
arms also need the devgen/policy commit 8b811e45, emitted with plowc built from it). Both arms of a
table run on the same plowrt (named in the title where it is not bin34 6a4b75bc). % floor = floor /
measured, floor = decode bytes (weights streamed once + KV rows) / 3.35 TB/s, prefill max(FLOP /
peak, bytes / 3.35 TB/s) with peak 1979 TF/s FP8, 990 BF16; not reported for the MoE packets, whose
routed expert bytes the model cannot bound. digest = TOK_STREAM fnv of the run (`equal` = both
passes of both arms). `--same` slot agreement held on every arm (8/8 .. 64/64). Not measured: the
4160/4224 rungs (they only occur with decode riders, which step_bench does not build) and, for the
native prefill arms, the 1152 and 8192 rungs. Raw: /opt/dlami/nvme/lava-tts/route-sweep (A/, B/, C/,
D/, logs/, gates/).

FP8 result: devgen and the segment_roles policy now route the W8A8 projections to cuBLASLt FP8
(commits 8b811e45, f2f7fabb: prefill admitted at the packet's 128/1024/4096/8192 buckets only), and
the recipe takes prefill on cuBLASLt plus decode from 2 rows (commit a3ee2457). Each route was also
measured alone (attribution tables): prefill alone gives the whole prefill gain with identical
single-request first-token digests (the packed 4 x 1024 digest differs), decode alone the whole decode gain. Prefill 2.5-4.5x faster (4096 rows 1232 ->
277 ms, 9% -> 39% of the FP8 floor), decode B2..B48 38-49% faster; B1 stays on the native M1 TMA
GEMV. FP32 gate (packet 2af45e892b75, plowrt 69215b1e) c1/c8/c32 PASS: kl_trim 0.0110/0.0099/0.0102
vs vLLM hub-FP8 0.0147/0.0153/0.0179, top1_decisive 0.9939/0.9951/0.9951, cont_frac
0.731/0.765/0.736, needle 1. The packet needs a plowrt carrying this policy.

BF16 result: decode cuBLASLt from 4 rows with the GDN step isolated from 4 (commit 9d6cc389): B4/B8
-7..-10%, B1/B2/B16+ unchanged with identical digests. The GDN isolation alone (cuBLASLt still from
16) is neutral (+0.1..+0.9%), so the gain is the library route. From 1 row loses at B1/B2. FP32 gate
(packet 3157ca560892) c1/c8/c32 PASS: kl_trim 0.000461/0.000450/0.000415 vs vLLM
0.000338/0.000346/0.000356, top1_decisive 0.9988/0.9988/1, cont_frac 0.862/0.874/0.883, needle 1.
Native prefill is 2.9-3.7x slower; the fused qkv matmul and the cuBLASLt terminal stay. At 32-48
slots x 16K (48 x 8K) step_bench runs out of KV memory on this packet (capacity, not a route).


**Qwen3.8 FP8 cuBLASLt prefill + decode from 2 rows (GDN isolate 2) -- PROMOTED (plowrt 69215b1e both arms)** (qwenfp8-base vs qwenfp8-fin2)

| phase | rows x ctx | qwenfp8-base ms | qwenfp8-fin2 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B1 x 1024 | 23.50 | 23.35 | -0.6% | 35 / 35 | equal |
| decode | B1 x 16000 | 24.13 | 23.99 | -0.6% | 35 / 35 | differs |
| decode | B2 x 1024 | 48.12 | 29.99 | -37.7% | 17 / 28 | differs |
| decode | B4 x 1024 | 48.72 | 29.25 | -40.0% | 17 / 29 | differs |
| decode | B8 x 1024 | 59.42 | 30.59 | -48.5% | 15 / 28 | differs |
| decode | B8 x 16000 | 62.21 | 33.41 | -46.3% | 18 / 33 | differs |
| decode | B16 x 1024 | 62.04 | 31.39 | -49.4% | 15 / 29 | differs |
| decode | B32 x 1024 | 64.95 | 32.96 | -49.2% | 16 / 31 | differs |
| decode | B32 x 16000 | 74.83 | 42.93 | -42.6% | 26 / 46 | differs |
| decode | B48 x 1024 | 68.41 | 35.27 | -48.4% | 16 / 32 | differs |
| prefill | 1 x 1024 | 320.53 | 78.49 | -75.5% | 8 / 34 | equal |
| prefill | 1 x 4096 | 1232.44 | 277.16 | -77.5% | 9 / 39 | equal |

**Qwen3.8 FP8 attribution: cuBLASLt prefill only (plowrt 69215b1e both arms)** (qwenfp8-base vs qwenfp8-pflt)

| phase | rows x ctx | qwenfp8-base ms | qwenfp8-pflt ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B1 x 1024 | 23.49 | 23.49 | -0.0% | 35 / 35 | equal |
| prefill | 1 x 128 | 90.01 | 36.53 | -59.4% | 9 / 22 | equal |
| prefill | 1 x 1024 | 319.65 | 78.55 | -75.4% | 8 / 34 | equal |
| prefill | 1 x 4096 | 1233.71 | 277.03 | -77.5% | 9 / 39 | equal |
| prefill | 1 x 8192 | 2540.45 | 559.12 | -78.0% | 8 / 38 | equal |
| prefill | 4 x 1024 | 1281.47 | 315.23 | -75.4% | 8 / 34 | differs |

**Qwen3.8 FP8 attribution: cuBLASLt decode from 16 only (plowrt bin34 both arms)** (qwenfp8-base vs qwenfp8-lt16)

| phase | rows x ctx | qwenfp8-base ms | qwenfp8-lt16 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B1 x 1024 | 23.50 | 23.35 | -0.7% | 35 / 35 | equal |
| decode | B8 x 1024 | 59.46 | 59.78 | +0.5% | 15 / 14 | equal |
| decode | B16 x 1024 | 62.08 | 31.41 | -49.4% | 15 / 29 | differs |
| decode | B16 x 16000 | 67.60 | 36.97 | -45.3% | 20 / 37 | differs |
| decode | B32 x 1024 | 64.97 | 32.99 | -49.2% | 16 / 31 | differs |
| decode | B32 x 16000 | 74.82 | 42.92 | -42.6% | 26 / 46 | differs |
| decode | B48 x 1024 | 68.44 | 35.24 | -48.5% | 16 / 32 | differs |
| prefill | 1 x 1024 | 320.48 | 314.01 | -2.0% | 8 / 9 | equal |
| prefill | 1 x 4096 | 1231.21 | 1236.12 | +0.4% | 9 / 9 | equal |

**Qwen3.8 FP8 attribution: cuBLASLt decode from 2 + GDN isolate 2 only (plowrt bin34 both arms)** (qwenfp8-base vs qwenfp8-lt2i)

| phase | rows x ctx | qwenfp8-base ms | qwenfp8-lt2i ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B1 x 1024 | 23.50 | 23.37 | -0.6% | 35 / 35 | equal |
| decode | B2 x 1024 | 48.12 | 29.99 | -37.7% | 17 / 28 | differs |
| decode | B4 x 1024 | 48.73 | 29.23 | -40.0% | 17 / 29 | differs |
| decode | B8 x 1024 | 59.44 | 30.61 | -48.5% | 15 / 28 | differs |
| decode | B8 x 16000 | 62.31 | 33.41 | -46.4% | 18 / 33 | differs |
| decode | B16 x 1024 | 62.10 | 31.42 | -49.4% | 15 / 29 | differs |
| decode | B32 x 1024 | 64.95 | 32.96 | -49.3% | 16 / 31 | differs |

**Qwen3.8 FP8 cuBLASLt prefill + decode from 1 row (GDN isolate 1)** (qwenfp8-base vs qwenfp8-fin1)

| phase | rows x ctx | qwenfp8-base ms | qwenfp8-fin1 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B1 x 1024 | 23.50 | 29.12 | +23.9% | 35 / 28 | differs |
| decode | B1 x 16000 | 24.13 | 29.76 | +23.3% | 35 / 29 | differs |
| decode | B2 x 1024 | 48.12 | 29.98 | -37.7% | 17 / 28 | differs |
| decode | B4 x 1024 | 48.72 | 29.25 | -40.0% | 17 / 29 | differs |
| decode | B8 x 1024 | 59.42 | 30.61 | -48.5% | 15 / 28 | differs |
| decode | B8 x 16000 | 62.21 | 33.44 | -46.3% | 18 / 33 | differs |
| decode | B16 x 1024 | 62.04 | 31.42 | -49.4% | 15 / 29 | differs |
| decode | B32 x 1024 | 64.95 | 32.96 | -49.2% | 16 / 31 | differs |
| decode | B32 x 16000 | 74.83 | 42.94 | -42.6% | 26 / 46 | differs |
| decode | B48 x 1024 | 68.41 | 35.27 | -48.4% | 16 / 32 | differs |
| prefill | 1 x 1024 | 320.53 | 78.58 | -75.5% | 8 / 34 | equal |
| prefill | 1 x 4096 | 1232.44 | 276.90 | -77.5% | 9 / 39 | equal |

**Qwen3.8 FP8 PLOW_TERMINAL_LT=0** (qwenfp8-base vs qwenfp8-termlt0)

| phase | rows x ctx | qwenfp8-base ms | qwenfp8-termlt0 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| prefill | 1 x 128 | 90.06 | 90.13 | +0.1% | 9 / 9 | equal |
| prefill | 1 x 1024 | 320.08 | 320.15 | +0.0% | 8 / 8 | equal |

**Qwen3.8 BF16 cuBLASLt decode from 4 rows (GDN isolate 4) -- PROMOTED** (qwenbf16-base vs qwenbf16-lt4i)

| phase | rows x ctx | qwenbf16-base ms | qwenbf16-lt4i ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B1 x 1024 | 22.19 | 22.17 | -0.1% | 70 / 70 | equal |
| decode | B1 x 16000 | 22.79 | 22.77 | -0.1% | 70 / 70 | equal |
| decode | B2 x 1024 | 23.16 | 23.12 | -0.1% | 68 / 68 | equal |
| decode | B2 x 16000 | 24.31 | 24.28 | -0.1% | 67 / 67 | equal |
| decode | B4 x 1024 | 24.74 | 22.97 | -7.1% | 64 / 69 | differs |
| decode | B4 x 16000 | 26.03 | 24.20 | -7.0% | 65 / 70 | equal |
| decode | B8 x 1024 | 26.32 | 23.81 | -9.5% | 61 / 67 | differs |
| decode | B8 x 16000 | 29.18 | 26.65 | -8.7% | 63 / 69 | differs |
| decode | B16 x 1024 | 24.99 | 25.00 | +0.0% | 66 / 66 | equal |

**Qwen3.8 BF16 attribution: GDN isolate 4 alone, cuBLASLt still from 16** (qwenbf16-base vs qwenbf16-iso4)

| phase | rows x ctx | qwenbf16-base ms | qwenbf16-iso4 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B4 x 1024 | 24.74 | 24.97 | +0.9% | 64 / 63 | equal |
| decode | B8 x 1024 | 26.32 | 26.36 | +0.1% | 61 / 61 | equal |
| decode | B8 x 16000 | 29.18 | 29.21 | +0.1% | 63 / 63 | equal |

**Qwen3.8 BF16 cuBLASLt decode from 1 row (GDN isolate 1)** (qwenbf16-base vs qwenbf16-lt1i)

| phase | rows x ctx | qwenbf16-base ms | qwenbf16-lt1i ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B1 x 1024 | 22.19 | 22.94 | +3.4% | 70 / 68 | equal |
| decode | B1 x 16000 | 22.79 | 23.53 | +3.2% | 70 / 67 | equal |
| decode | B2 x 1024 | 23.16 | 23.47 | +1.4% | 68 / 67 | differs |
| decode | B2 x 16000 | 24.31 | 24.61 | +1.3% | 67 / 66 | equal |
| decode | B4 x 1024 | 24.74 | 22.97 | -7.1% | 64 / 69 | differs |
| decode | B4 x 16000 | 26.03 | 24.21 | -7.0% | 65 / 70 | equal |
| decode | B8 x 1024 | 26.32 | 23.81 | -9.5% | 61 / 67 | differs |
| decode | B8 x 16000 | 29.18 | 26.66 | -8.6% | 63 / 69 | differs |
| decode | B16 x 1024 | 24.99 | 25.00 | +0.0% | 66 / 66 | equal |

**Qwen3.8 BF16 native prefill projections** (qwenbf16-base vs qwenbf16-pfnative)

| phase | rows x ctx | qwenbf16-base ms | qwenbf16-pfnative ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B1 x 1024 | 22.26 | 22.28 | +0.1% | 70 / 70 | equal |
| prefill | 1 x 128 | 33.07 | 98.24 | +197.1% | 47 / 16 | equal |
| prefill | 1 x 512 | 52.69 | 194.64 | +269.4% | 51 / 14 | equal |
| prefill | 1 x 1024 | 84.91 | 297.21 | +250.0% | 63 / 18 | equal |
| prefill | 1 x 4096 | 321.37 | 1006.65 | +213.2% | 67 / 21 | equal |
| prefill | 1 x 8192 | 676.29 | 1991.75 | +194.5% | 64 / 22 | equal |

**Qwen3.8 BF16 PLOW_DECODE_LT_QKV=0** (qwenbf16-base vs qwenbf16-ltqkv0)

| phase | rows x ctx | qwenbf16-base ms | qwenbf16-ltqkv0 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| decode | B16 x 1024 | 25.18 | 25.22 | +0.2% | 66 / 66 | differs |
| decode | B16 x 16000 | 30.77 | 30.84 | +0.2% | 69 / 69 | equal |
| decode | B32 x 1024 | 27.58 | 27.54 | -0.2% | 64 / 64 | equal |
| decode | B48 x 1024 | 30.10 | 30.10 | +0.0% | 62 / 62 | equal |

**Qwen3.8 BF16 PLOW_TERMINAL_LT=0** (qwenbf16-base vs qwenbf16-termlt0)

| phase | rows x ctx | qwenbf16-base ms | qwenbf16-termlt0 ms | change | % floor (cur / alt) | digest |
|---|---|---|---|---|---|---|
| prefill | 1 x 128 | 33.03 | 33.42 | +1.2% | 47 / 46 | equal |
| prefill | 1 x 1024 | 85.04 | 86.09 | +1.2% | 63 / 62 | equal |
| prefill | 1 x 4096 | 322.56 | 321.99 | -0.2% | 67 / 67 | equal |

## 2026-10-27: BF16 packet vs vLLM 0.29 on the final PR #108 build (production readiness)

**Result: not production ready against vLLM 0.29. Every cell is a loss on throughput, and every cell
but one loses on latency.** `campaign.py report` MATCHED + EQUIVALENT on every cell; rows in
[comparison.csv](comparison.csv) under `campaign_source` `q38-20261027-v029-bf16`.

- Infervisor: plowrt `bb9f389fd`, packet `3157ca560892` (route-sweep `qwenbf16-lt4i`: cuBLASLt
  decode from 4 rows, GDN step isolated from 4), server env = the recipe's `[serve.env]` (live KV,
  prefix cache + VMM prefix with the KV block store and its host tier on auto, `PLOW_PF_INTERLEAVE=1024`).
- Baseline: vLLM 0.29.0, BF16 weights and KV, `--max-model-len 16384 --gpu-memory-utilization 0.9
  --max-num-seqs 256`, prefix caching on (mamba cache mode `align`), FLASH_ATTN and the FlashInfer
  GDN prefill (defaults; the venv needs the cutlass DSL path on `PYTHONPATH`). 2 repeats, same seeds.
- Quality: FP32-reference gate c1/c8/c32 PASS on the served packet (bin43 gate).
- Open-loop rates: the production mix at 1.8 sessions/s overloads vLLM itself on this dense 27B
  (goodput 0 %, TTFT P50 75 s, 3.6 req/s served), so the mix runs at 0.45 and 0.7 sessions/s.
- Raw evidence: `/opt/dlami/nvme/lava-tts/prod-compare/qwenbf16-{prod,grid}/{plow,vllm,report-plow}`.

| Cell | Total throughput | TTFT P99 | TPOT P99 | Peak memory |
|---|---:|---:|---:|---:|
| prod mix, 0.45 sessions/s | 0.95x | 3.85x * | 1.76x * | 1.06x |
| prod mix, 0.7 sessions/s | 0.85x | 4.75x * | 0.54x | 1.03x |
| 1000/128 c1 greedy | 0.91x | 0.90x * | 1.12x | 1.01x * |
| 1000/128 c8 greedy | 0.88x | 1.32x * | 1.41x * | 1.04x |
| 1000/128 c32 greedy | 0.80x | 1.48x * | 1.60x | 1.08x |

`*` = repeat spread > 10% (FLAGGED; direction only). The c1 TTFT P99 (0.90x) sits on a 60 % plowrt
spread. The 0.7 TPOT P99 (0.54x) is not a win: vLLM's first repeat at 0.7 overloaded (TTFT P50 3.7 s,
TPOT P50 217 ms), and plowrt overloaded on both repeats (goodput 0 % and 5.5 % vs vLLM 13 % and 96 %).
Supplementary at 0.45, vLLM / Infervisor: goodput 1.68 / 1.01 req/s, SLO met 99.3 % / 62.7 %,
TTFT P50 264 / 1084 ms, TPOT P50 26.7 / 57.8 ms, in-flight 3.3 / 7.6; prefix token hit (server
metrics) vLLM 63-69 % vs plowrt 52 % then 34 % at 0.45, and 4-13 % at 0.7.

Against the 2026-10-13 report (vLLM 0.28, prefix cache off, packet `3fb7cefaf150`), c32 throughput
fell from 0.96x to 0.80x and TPOT P99 rose from 1.25x to 1.60x. The baseline is flat (6,235 vs 6,237
tok/s), so the regression is on the plowrt side: this packet, the block store and the prefix cache
are now on. Not attributed: the prefix-off control grid was withdrawn (see below).

Diagnosis (packlog_audit, 0.45 sessions/s): decode at ~4 rows runs 25 ms per step (vLLM TPOT P50
26 ms, including its prefill interference), and a 1024-row prefill chunk takes 89 ms (11.5K rows/s).
Prefill is 49 % and decode 48 % of the wall clock, and riders add 1.4-3 ms each. Prefix retention is
the second factor. It falls during the run, and the host tier froze at its reserve
(`kv store: host tier frozen at the host reserve`, 208 GB reserved) during the 0.7 cell. The decode
and prefill step gap belongs to the BF16 decode kernel work. Recipe-level variants were queued
(`PLOW_PF_INTERLEAVE=0`, plus ITL budget 60 / decode pipeline / admission overlap,
`PLOW_VMM_HOST_CACHE_MIB=0`, `PLOW_LT_RUNG_ALGOS=true`, prefix-off attribution grid) but were
withdrawn before running, to give the GPU to that work. The recipe is unchanged.
