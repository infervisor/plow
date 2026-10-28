# P2: real BF16 Gemma-4 decoder layer as a weight-stationary stage (Xeon 6975P-C)

Checklist phase P2. One real E2B / E4B text decoder layer, BF16 checkpoint weights, one decode token, on the 90
isolated cores. Raw evidence: `/tmp/g4c/l2r/{ref,results/p2v2,results/p2_ctr,results/p2_place}`. Tools in
`runtime/cpu/bench/l2r/`: `ref_layer.py` (FP32 reference), `l2r_layer.c` (stage), `p2_gate.py` (numerical gate),
`p2_sweep.sh`, `p2_sum.py`. Host state as P0/P1 (isolated cores, `cpu_dma_latency=0`, THP `madvise`).

## Scope: one E4B BF16 layer does not fit in L2

Exact per-layer BF16 weight bytes from the checkpoints' safetensors headers:

| layer | type | q / k / v / o | gate + up + down | per-layer input gate + proj | total | per worker (90) |
|---|---|---|---|---|---:|---:|
| E2B L0 | sliding, head 256, 1 KV head | 6 / 0.75 / 0.75 / 6 MiB | 54 MiB | 1.5 MiB | 69.0 MiB | 785 KiB |
| E2B L4 | full, head 512, 1 KV head | 12 / 1.5 / 1.5 / 12 MiB | 54 MiB | 1.5 MiB | 82.5 MiB | 939 KiB |
| E4B L0 | sliding, head 256, 2 KV heads | 10 / 2.5 / 2.5 / 10 MiB | 150 MiB | 2.5 MiB | 177.5 MiB | 2,020 KiB |
| E4B L5 | full, head 512, 2 KV heads | 20 / 5 / 5 / 20 MiB | 150 MiB | 2.5 MiB | 202.5 MiB | 2,304 KiB |

P1 shows 1.25-1.53 MiB per core held resident; the whole 2 MiB L2 is 180 MiB over 90 cores. An E2B layer
fits (0.77-0.92 MiB per core). An E4B layer cannot be L2-resident in BF16 on one socket. Agreed scope:
the **pure L2 stage is a real E2B layer**, and the **E4B layer runs as an L2/L3 split**. E2B's double-wide-MLP
layers (KV-shared layers 15+, 136.5 MiB = 1.52 MiB per core) are at the measured limit and were not run.

## Reference and numerical gate

`ref_layer.py <hf_dir> <layer> <ctx> <out>` runs the HF model (transformers 5.17, torch 2.13, FP32) on `<ctx>`
tokens of real text (repo docs) plus one decode token, and stops after `<layer>`. It dumps the layer's BF16
weights, its decode-step inputs (hidden state, per-layer input, RoPE cos/sin, the layer's KV cache: sliding
layers 511 rows, full layers `ctx` rows) and FP32 values at 17 op boundaries.

* `layer_ref` re-implements the layer standalone, and matches HF's own intermediates at every boundary to
  relative RMS < 1e-5, for all 6 dumps (E2B L0 2K, L4 2K/16K; E4B L0 2K, L5 2K/16K). The semantics are
  pinned from the HF source:
  * Gemma-4 RMSNorm is `x · (mean x² + ε)^-½ · w`.
  * q_norm and k_norm are followed by RoPE; v_norm has no scale.
  * Attention scaling is 1.0.
  * The sliding window means 511 cached rows plus the new one.
  * The tail is `per_layer_projection(gelu(per_layer_input_gate(h)) · per_layer_input)`, then norm, then
    residual, then `layer_scalar`.
* Its BF16 mode gives the error bar a BF16 kernel is held to: BF16 weights, every GEMV input rounded to BF16,
  BF16 KV, FP32 accumulation.
* **Gate (fixed before the first stage run):** at every boundary, stage relative RMS error vs FP32 ≤ 1.5 × the
  BF16 reference's; output cosine ≥ 0.99999 on the first and the last step.

## Stage (`l2r_layer.c`)

* One worker per isolated core. Every GEMV is output-row partitioned (4-row units for AVX-512, 16-row tiles
  for AMX). Each worker keeps its row slices of all 9 matrices in one 2 MiB-aligned, `MADV_HUGEPAGE`,
  node-local arena that it first-touches.
* KV is split by position across workers, node-local. Attention is split-K: per worker and head, the max, the
  sum and the partial output. A distributed combine follows, each worker owning a slice of the head outputs.
* Norms, RoPE and residuals are recomputed by every worker from the shared vectors. A decode step has 8
  dissemination barriers: qkv | attention | combine | o | gate,up,gelu | down | ple gate | ple proj.
* GEMV kernels:
  * AVX-512: `VDPBF16PS`, 4 rows per pass.
  * AMX: `TDPBF16PS` with M = 1 (A = the x chunk in tile row 0), weights packed into 1 KiB VNNI B tiles,
    4 C tiles per x load.
* Attention: per KV head, each K/V row is widened to FP32 once for all query heads of the group (compile-time
  group size). Softmax uses a vectorised exp: degree-7 Taylor after ln2 range reduction, < 1 ulp.
* Every step recomputes the same token (the new KV row overwrites itself), so the weights stay hot and the
  outputs must repeat bit for bit; the last step is gated as well as the first.
* `L2R_NOBCAST=1`: after step 0 every worker reads private snapshots of the shared vectors. The values are
  identical because the step repeats. This removes the activation all-gather from the timing and isolates
  compute + barriers.

## Numerical results

**72 of 72 runs PASS**: 6 dumps × AVX/AMX × all-gather on/off × 3 reps. The values are deterministic.
Per-boundary error / BF16-reference error:

* E2B L0 2K: 0.99-1.02 at every boundary, output cosine 0.99999982. AMX and AVX-512 identical.
* E2B L4 16K: 0.95-1.06 through `h2`. The per-layer-input tail is higher: `pg` 1.27, `pact` 1.20, `pp` 1.46
  (smallest margin of all runs). Output 0.95, cosine 0.99999989. Identical before and after the softmax
  change, so it comes from the stage's accumulation and rounding order, not the exp.
* E4B L0 / L5: PASS, output cosine 0.9999997.
* Repeatability: the same errors with the worker order reversed (`66-95,34-63,2-31`), with 60 workers, and
  with 180 workers on SMT siblings. First and last step agree (`out` = `out_last`) after 5,000 steps.

Not yet run: logits / full-model FP32 KL with this stage in the model (P5 integration); the existing
full-model gates (P0) cover plowrt's own kernels only.

## Performance (AVX-512 unless noted; µs; median of 3 reps, p50 spread ≤ 3%)

| layer | ctx rows | GEMV | all-gather | KiB/worker | step p50 / p95 / p99 µs (p50 spread) | qkv | attn | comb | o | gate/up | down | ple g | ple p | barrier total | gate |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| e2b.L0.c2048 | 512 | amx | on | 785 | 148.6 / 153.3 / 156.7 (0.5%) | 2.32 | 25.22 | 25.00 | 20.11 | 19.77 | 28.20 | 13.32 | 2.04 | 22.1 | PASS |
| e2b.L0.c2048 | 512 | amx | off | 785 | 34.7 / 49.2 / 67.2 (0.4%) | 2.47 | 3.56 | 3.05 | 1.75 | 6.55 | 3.50 | 2.06 | 0.72 | 20.5 | PASS |
| e2b.L0.c2048 | 512 | avx | on | 785 | 145.2 / 148.9 / 150.9 (1.7%) | 1.73 | 24.32 | 24.78 | 19.49 | 18.67 | 27.34 | 11.58 | 3.38 | 20.4 | PASS |
| e2b.L0.c2048 | 512 | avx | off | 785 | 32.9 / 47.0 / 50.3 (0.3%) | 1.64 | 3.09 | 2.87 | 1.20 | 7.19 | 3.08 | 0.88 | 0.21 | 17.2 | PASS |
| e2b.L4.c16384 | 16385 | amx | on | 939 | 195.9 / 200.8 / 205.0 (2.9%) | 4.50 | 68.60 | 23.52 | 26.19 | 21.36 | 27.49 | 13.22 | 1.85 | 34.5 | PASS |
| e2b.L4.c16384 | 16385 | amx | off | 939 | 90.5 / 92.3 / 98.9 (0.8%) | 4.30 | 49.10 | 5.41 | 4.16 | 9.03 | 5.60 | 2.53 | 0.65 | 31.1 | PASS |
| e2b.L4.c16384 | 16385 | avx | on | 939 | 189.7 / 194.3 / 200.1 (1.7%) | 3.45 | 66.21 | 23.67 | 24.98 | 19.77 | 25.18 | 10.24 | 3.86 | 27.0 | PASS |
| e2b.L4.c16384 | 16385 | avx | off | 939 | 84.9 / 91.7 / 98.3 (1.8%) | 3.40 | 46.64 | 4.71 | 2.40 | 9.40 | 3.40 | 1.43 | 0.35 | 24.8 | PASS |
| e2b.L4.c2048 | 2049 | amx | on | 939 | 164.4 / 169.4 / 173.2 (0.4%) | 3.88 | 38.76 | 24.32 | 25.22 | 19.33 | 27.05 | 13.33 | 1.72 | 27.4 | PASS |
| e2b.L4.c2048 | 2049 | amx | off | 939 | 46.1 / 47.2 / 53.6 (1.1%) | 2.96 | 9.77 | 4.88 | 2.89 | 7.06 | 3.93 | 2.09 | 0.65 | 22.3 | PASS |
| e2b.L4.c2048 | 2049 | avx | on | 939 | 165.7 / 169.9 / 172.5 (0.1%) | 3.00 | 40.69 | 24.00 | 25.06 | 18.82 | 26.11 | 10.36 | 4.47 | 23.8 | PASS |
| e2b.L4.c2048 | 2049 | avx | off | 939 | 43.2 / 47.3 / 58.6 (0.7%) | 2.81 | 9.32 | 4.52 | 2.14 | 7.58 | 3.04 | 1.13 | 0.24 | 18.1 | PASS |
| e4b.L0.c2048 | 512 | amx | on | 2020 | 205.9 / 211.6 / 214.7 (3.0%) | 12.33 | 24.62 | 20.33 | 23.23 | 62.55 | 34.71 | 14.62 | 3.10 | 52.6 | PASS |
| e4b.L0.c2048 | 512 | amx | off | 2020 | 128.3 / 139.5 / 148.5 (1.0%) | 12.80 | 10.14 | 4.29 | 5.59 | 53.90 | 24.60 | 5.41 | 0.84 | 54.9 | PASS |
| e4b.L0.c2048 | 512 | avx | on | 2020 | 208.2 / 214.1 / 218.6 (2.5%) | 8.39 | 23.52 | 26.76 | 22.54 | 57.44 | 38.79 | 12.03 | 6.17 | 32.4 | PASS |
| e4b.L0.c2048 | 512 | avx | off | 2020 | 118.0 / 120.7 / 126.2 (1.2%) | 8.95 | 11.54 | 4.19 | 4.53 | 50.22 | 22.40 | 4.04 | 0.89 | 34.4 | PASS |
| e4b.L5.c16384 | 16385 | amx | on | 2304 | 348.0 / 353.7 / 358.1 (0.6%) | 31.21 | 80.58 | 11.70 | 27.10 | 107.13 | 51.87 | 22.23 | 3.99 | 59.5 | PASS |
| e4b.L5.c16384 | 16385 | amx | off | 2304 | 303.8 / 308.1 / 312.9 (0.6%) | 31.74 | 78.82 | 7.39 | 20.41 | 97.39 | 48.38 | 7.07 | 1.53 | 63.3 | PASS |
| e4b.L5.c16384 | 16385 | avx | on | 2304 | 350.0 / 356.9 / 361.5 (1.1%) | 30.93 | 82.23 | 11.52 | 33.10 | 101.17 | 59.41 | 14.04 | 7.63 | 52.4 | PASS |
| e4b.L5.c16384 | 16385 | avx | off | 2304 | 302.8 / 306.0 / 311.6 (0.5%) | 31.60 | 79.49 | 6.92 | 19.25 | 101.24 | 47.46 | 5.45 | 1.10 | 54.9 | PASS |
| e4b.L5.c2048 | 2049 | amx | on | 2304 | 275.5 / 281.9 / 287.0 (0.6%) | 28.00 | 24.32 | 21.44 | 27.57 | 94.10 | 44.33 | 20.01 | 4.05 | 64.3 | PASS |
| e4b.L5.c2048 | 2049 | amx | off | 2304 | 214.2 / 222.5 / 229.4 (0.3%) | 28.51 | 17.28 | 7.33 | 16.69 | 85.64 | 40.64 | 6.41 | 1.07 | 66.9 | PASS |
| e4b.L5.c2048 | 2049 | avx | on | 2304 | 288.4 / 295.2 / 299.2 (1.3%) | 26.23 | 26.32 | 32.05 | 32.18 | 86.37 | 52.90 | 13.86 | 7.90 | 44.7 | PASS |
| e4b.L5.c2048 | 2049 | avx | off | 2304 | 210.7 / 213.1 / 218.3 (0.4%) | 26.55 | 19.37 | 6.91 | 16.45 | 85.71 | 38.86 | 4.60 | 1.02 | 44.9 | PASS |

128K rows (`L2R_CTX_REPEAT=8` on the 16K dumps, timing only, numerics not checked):

| layer | ctx rows | GEMV | all-gather | KiB/worker | step p50 / p95 / p99 µs (p50 spread) | qkv | attn | comb | o | gate/up | down | ple g | ple p | barrier total | gate |
|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| e2b.L4.c131072 | 131073 | avx | on | 939 | 557.7 / 568.3 / 696.7 (0.2%) | 15.93 | 396.77 | 11.61 | 26.96 | 48.84 | 29.09 | 11.63 | 5.78 | 60.0 | n/a (timing only) |
| e2b.L4.c131072 | 131073 | avx | off | 939 | 500.8 / 511.1 / 528.7 (3.6%) | 16.15 | 397.81 | 7.08 | 12.15 | 39.27 | 18.07 | 3.29 | 0.70 | 61.0 | n/a (timing only) |
| e4b.L5.c131072 | 131073 | avx | on | 2304 | 1344.8 / 1364.9 / 1437.0 (0.5%) | 40.26 | 1035.96 | 13.72 | 36.75 | 119.75 | 58.70 | 16.76 | 8.26 | 95.6 | n/a (timing only) |
| e4b.L5.c131072 | 131073 | avx | off | 2304 | 1311.4 / 1331.8 / 1431.7 (0.5%) | 40.51 | 1042.25 | 9.42 | 23.95 | 118.31 | 51.02 | 6.45 | 2.18 | 104.5 | n/a (timing only) |

Phase columns are the critical path (max over workers of the phase's own compute, mean over steps). The
barrier column is the mean over workers of time spent waiting in the 8 barriers.

### Where the time goes

* **The activation all-gather dominates.** Every worker reads each whole output vector (q, o, attn, act, down,
  pact) right after up to 90 other cores wrote its lines. That costs ~100-115 ns per 64-byte line with almost
  no overlap (instrumented prologues: reading q, 128 lines, 17 µs; o, 96 lines, 11 µs). E2B L0: 145 µs with
  the all-gather vs 33 µs without. The cost grows with the number of readers (180 SMT workers: 221 µs). This
  is P3's subject.
* **Barriers:** 8 per layer at 1.6-2.6 µs each (17-25 µs per E2B layer). P3.
* **Compute** with the all-gather removed, E2B L0: ~20 µs critical path. gate/up 7.2 µs (419 KiB per worker,
  ~60 GB/s incl. gelu); down 3.1 (~70 GB/s); attention 3.1; combine 2.9.
* **AMX is not faster at batch 1.** Same numerics, same or slightly worse time. 16-row tiles split 1,536-row
  matrices into 96 tiles over 90 workers, so 6 workers do double work (down: max 3.5 µs vs mean 1.8). M = 1
  uses 1/16 of each tile op.
* **Attention at 16K is compute-bound, not memory-bound.** DRAM reads < 0.14 MiB per step; most of each
  worker's 364 KiB KV slice stays in L2 (1,495 L2 lines in per step against 5,824 KV lines). Widening K/V
  once per row did not help (100 µs). Removing 1,456 scalar `expf` calls per worker did: 100 → 46.5 µs.
* **128K:** E2B's 256 MiB KV streams from L3 (attention 398 µs ≈ 680 GB/s). E4B's 512 MiB exceeds L3 and
  streams from DRAM (1,042 µs ≈ 515 GB/s; P0 DRAM tier 612-669 GB/s). That is P4's subject (KV placement).

### L2 residency of the stage (perf `L2_LINES_IN.ALL`, no all-gather)

| stage | weight lines per worker | L2 lines in per step per worker | reading |
|---|---:|---:|---|
| E2B L0 2K | 12,560 | 195 | weights resident (KV + vectors only) |
| E2B L4 16K | 15,024 | 1,495 | weights resident; most of the KV slice too |
| E4B L0 2K | 32,320 | 16,928 | ~52% of the slice refetched from L3 every step |

### E4B L2/L3 split

The layer is 2,020 KiB per worker, so part of it must stream every step. Explicit splits were measured
against plain LRU (E4B L0 2K, no all-gather, step p50):

| policy | step p50 µs | L2 lines in / step |
|---|---:|---:|
| plain THP, hardware replacement | **114.7** | 16,928 |
| resident budget 768 / 1,024 / 1,280 / 1,536 KiB, rest prefetched `PREFETCHNTA` | 160.6 / 152.6 / 137.2 / 138.0 | 14.7-16.0 K |
| 1,280 KiB locked through `pseudo_lock_sram` (12 ways), rest streamed in the 4 free ways | 128.0 | 16,463 |

* `PREFETCHNTA` lines still allocate in L2 on this part (lines in barely move), so it only adds instructions.
* The lock holds under an L3 stream (held fraction 0.9999 → 0.9993, unlike P1's DRAM stream). But
  everything else is then confined to 4 ways and misses more.
* Plain replacement already keeps about half the slice resident, and is fastest. The E4B layer is
  L3-bandwidth-bound: ~93 MB per step from L3 at the P0 rate of 1.36 TB/s is ~68 µs, matching gate/up + down
  (72.6 µs). E4B L5 behaves the same (plain 209.9 µs; splits 241-266 µs).

## Exit gate P2

| criterion | result |
|---|---|
| real BF16 layer passes the numerical gate | **PASS**: 72/72 runs, every boundary ≤ 1.5 × the BF16 reference error, output cosine ≥ 0.9999997, deterministic across placements and 5,000 steps |
| latency breakdown understood | yes: all-gather > barriers > compute for E2B; L3 bandwidth for the E4B FFN; attention compute (2K/16K), L3 (E2B 128K), DRAM (E4B 128K) |
| stretch: ≤ 20 µs per layer at batch 1, L2-resident | **missed**. Best real E2B layer: 32.9 µs p50 without the all-gather (compute ~20 + barriers ~17), 145 µs with it. E4B: 118 µs (L3-bound) |

**P2 exit: PASS on numerics; stretch target missed.** Carried forward:

* P3: replace the read-everything all-gather (e.g. per-node replicas pushed by the producer, flag-in-data,
  CLDEMOTE), and cut the 8 dissemination barriers, ~2 µs each. Those two cost ~110 + 17 µs of the E2B layer.
* P4: KV placement for full-attention layers beyond 16K (E2B L3-resident at 128K, E4B DRAM-bound).
* Open from the checklist:
  * AMX at batch 4/8/16 and batch-B stage timing: the stage is batch-1 only, and batch-1 AMX gives no gain.
  * A fused-vs-unfused comparison: the stage fuses gelu·up into gate/up and recomputes norms/residuals
    redundantly; the unfused variant was not built.
  * 12- vs 14-way stage runs: the lock is not used for the E2B stage, so moot; E4B locking measured at 12 ways
    only.
  * The AVX-512 GEMV reaches ~60-70 GB/s per worker in the stage vs 101 GB/s in the P0 stream.
