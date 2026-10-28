# P4: BF16 long-context KV and L3 staging (Xeon 6975P-C)

Checklist phase P4. Stage, gate and host state are as in P2/P3: real E2B/E4B layers, BF16, 90 isolated workers,
`repnt` broadcast with dissemination barriers.

* Raw evidence: `/tmp/g4c/l2r/{ref,results/p4_tile,p4_early,p4_amx,p4_matrix,p4_lock,p4_attn,p4_foot}`.
* Tools: commit e9804c64, in `runtime/cpu/bench/l2r/`.
  * `l2r_layer.c` gains the P4 knobs: `L2R_KV_COPIES`, `L2R_KV_TILE_KIB`, `L2R_KV_PFD`/`L2R_KV_PFH`,
    `L2R_KV_EARLY_KIB`, `L2R_SEQS`, `L2R_ATTN_ONLY` and `L2R_PERF_CTL`.
  * `ref_layer.py` now does chunked prefill.
  * `p4_run.sh` runs one cell, `p4_ctr.sh` runs it under counters, `p4_sum.py` summarizes.
* Build: as P3. AVX-512 GEMV unless marked AMX.
* Runs: the matrix, lock, AMX and attention cells are 3 reps; the tile/prefetch/early selection is 2 reps. Medians
  are reported with the p50 spread. Steps scale with KV bytes: 5,000 / 2,000 / 600 / 300, with the first 10%
  excluded.
* **Gate: 712/712 runs PASS**, unchanged thresholds.
  * Lowest output cosine: 0.99999939.
  * Worst boundary ratio: pp 1.46 (E2B L4 16K, inherent; same as P2).
  * With `L2R_SEQS`, every sequence's attention output is checked (worst = the BF16 reference error).

## P4.1 Capacity: checkpoint KV shapes and exact BF16 KV bytes

From the checkpoint configs:

| | E2B | E4B |
|---|---|---|
| layers | 35 | 42 |
| layers that own KV | 15 | 24 |
| full-attention layers that own KV | 3 (L4, L9, L14) | 4 (L5, L11, L17, L23) |
| sliding layers that own KV (window 512) | 12 | 20 |
| KV heads × head dim (sliding / full) | 1 × 256 / 1 × 512 | 2 × 256 / 2 × 512 |
| full-attention KV per token per layer | 2 KiB | **4 KiB** (checklist: 4 KiB, confirmed) |
| sliding KV ring per layer | 0.5 MiB (511 + 1 rows) | 1 MiB |

The remaining layers (E2B 20, E4B 18) are KV-shared: they re-read the last owning layer of their type. So the
KV read per token is larger than the KV allocated:

* E2B: 7 full-attention readers and 28 sliding.
* E4B: 7 full-attention readers and 35 sliding.

Per sequence, MiB:

| ctx | E4B full layer | E4B allocated | E4B read / token | E2B full layer | E2B allocated | E2B read / token |
|---|---:|---:|---:|---:|---:|---:|
| 2K | 8 | 52 | 91 | 4 | 18 | 42 |
| 8K | 32 | 148 | 259 | 16 | 54 | 126 |
| 16K | 64 | 276 | 483 | 32 | 102 | 238 |
| 32K | 128 | 532 | 931 | 64 | 198 | 462 |
| 64K | 256 | 1,044 | 1,827 | 128 | 390 | 910 |
| 128K | 512 | 2,068 | 3,619 | 256 | 774 | 1,806 |

* E4B's full layer matches the checklist's 8 / 32 / 64 / 128 / 256 / 512 MiB.
* At concurrency 4 and 16 multiply by 4 and 16. E4B at 128K × 16 allocates 32 GiB.
* The stage allocates exactly `2 × kv_heads × (ctx + 1) × head_dim × 2 B` per layer per sequence. The serving
  engine's own allocation is P5's to measure.

### References at 2K-128K

* `ref_layer.py` now prefills in chunks: 8,192 tokens up to 8K, 4,096 at 32K, 2,048 at 64K-128K.
  * A chunked re-run of the P2 E2B L4 16K dump agrees with the single-shot dump to FP32 rounding: x_in
    2.3e-7, attn 3.1e-7, out 1.6e-7 rel RMS.
* New dumps: full layers (E2B L4, E4B L5) at 8K, 32K, 64K and 128K; sliding layers L0 at 128K.
* All but one pass the 1e-5 HF self-check. **E2B L4 128K exceeds it on `attn`**: 1.24e-5 rel RMS vs HF;
  `out` is 1.9e-6.
* The HF-vs-reference attention difference grows steadily with context:

  | ctx | 2K | 8K | 16K | 32K | 64K | 128K |
  |---|---:|---:|---:|---:|---:|---:|
  | E2B L4 attn rel RMS | 2.7e-7 | 5.2e-7 | 1.0e-6 | 1.9e-6 | 3.5e-6 | 1.24e-5 |

  * That pattern is FP32 accumulation over N terms (HF SDPA vs an explicit softmax).
  * It is 61× below the BF16 error bar at that context (7.6e-4).
  * The dump is used as is. The self-check threshold was not changed.

## P4.2 Paths A / B / C, tile size and prefetch

The three paths:

* **A**: KV cold in DRAM. Each worker rotates through enough identical KV copies that the footprint is
  ≥ 1.5 GiB (3.2× L3), so every step reads its KV from DRAM. This is the serving case: between two reads of a
  layer's KV, the rest of the model's weights and KV pass through the caches.
* **B**: KV naturally cache-hot. One copy, re-read every step, so the KV stays in L2/L3 when it fits.
* **C**: path A plus explicit staging. Either in-loop software prefetch d rows ahead, or a bulk prefetch of the
  step's KV at step start (`EARLY`, issued before the qkv GEMV).

Selection on the 32K full-attention layers (step p50 µs, 2 reps; full table in `results/p4_tile/summary.md`):

| setting | E2B L4 B | E2B L4 A | E4B L5 B | E4B L5 A |
|---|---:|---:|---:|---:|
| one tile (whole chunk), no prefetch | 144.3 | 326.3 | 388.5 | 639.6 |
| tile 16 / 64 / 256 / 1,024 KiB | 230.8 / 174.4 / 149.6 / 145.3 | 298.5 / 253.2 / **257.2** / 324.8 | 468.9 / 394.8 / **381.6** / 388.7 | 580.6 / 543.1 / **546.4** / 637.6 |
| tile 256 + T0 prefetch, d = 8 rows | 158.9 | **192.6** | 395.7 | **464.4** |
| tile 256 + T0, d = 2 / T2, d = 2 / T2, d = 8 | 158.1 / 152.0 / 152.3 | 193.0 / 200.8 / 197.1 | 395.6 / 395.2 / 394.5 | 466.2 / 468.1 / 463.8 |
| tile 0/64 + NTA prefetch, d = 2-32 | n/a | 347-469 | n/a | 665-834 |
| tile 256 + EARLY T1/T2, 256 / 1,024 / 4,096 KiB | n/a | 255-274 | n/a | 591-630 |

**Selected: tile 256 KiB with in-loop T0 prefetch, 8 rows ahead.** Why:

* Tiled online softmax keeps one tile of V in L1/L2 for the 16-32 column passes. Untiled, V is re-walked from
  DRAM: path A goes 326 → 257 µs.
* Prefetch then hides the DRAM latency (257 → 193 µs). Attention runs at 500-600 GB/s, against the P0 DRAM read
  rate of ~645 GB/s on this worker set.
* **NTA prefetch is worse than none.** On this part an NTA line still allocates in L2 (P2), and the hint only
  shortens its life.
* **Bulk early staging loses.** Its attention phase is faster (117-138 vs 133 µs), but the burst of prefetches
  stalls the qkv / o / FFN GEMVs behind it (89-123 vs 33 µs). Step: 255-274 vs 193 µs for E2B and 591-630 vs 464 µs for E4B. Per the checklist rule,
  explicit staging stays off.
* On already-hot KV (B), prefetch costs 2-6%, and 256 KiB tiles cost E2B +3.7% / save E4B 1.8%.

### Footprint sweep: when cache reuse beats prefetch

E4B L5 32K, 128 MiB of KV per step. The worker rotates through k copies; footprint = k × 128 MiB.
Step p50 µs, 3 reps:

| copies (footprint) | 1 (128 MiB) | 2 (256) | 3 (384) | 4 (512) | 6 (768) | 8 (1 GiB) | 12 (1.5 GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| no prefetch | **377.1** | 408.2 | 463.6 | 485.8 | 514.0 | 529.8 | 542.5 |
| T0 prefetch d 8 | 390.5 | **393.0** | **395.2** | **410.2** | **432.3** | **445.9** | **459.1** |

* With no prefetch, alternating between copies degrades steadily as the footprint grows past ~1/2 of L3. L3 is
  480 MiB and non-inclusive, so DRAM reads do not all land in it.
* **Prefetch wins from a 256 MiB footprint up.** Between reuses of a layer's KV in serving, the footprint is the
  whole per-token stream: E2B weights alone are ~2.6 GiB. Even the KV-shared readers re-read their source 5-6
  layers later, behind ≥ 375 MiB of other traffic. **Serving is path C at every context.**
* 12 vs 14 lock ways is in P4.3 below.

## P4.3 Context × concurrency matrix (tile 256)

Step p50 / p99 µs, 3 reps each.

* C = cold KV + T0 prefetch.
* When the KV of all sequences is ≥ 1.5 GiB, the run is already DRAM-cold, so A = B, and C is B plus prefetch.
* Concurrency is `L2R_SEQS` (KV side only; the GEMVs stay batch 1). Max p50 spread across reps ≤ 7.0% (E2B L0
  2K c1, a 53 µs step); all other cells ≤ 6.3%.

| layer | ctx | seqs | KV MiB / step | B (hot) | A (cold) | **C (cold + prefetch)** | C attention µs (GB/s) | C GEMV phases µs |
|---|---:|---:|---:|---|---|---|---|---:|
| E2B L0 sliding | 2K / 128K | 1 | 0.5 | 53 / 67 · 54 / 67 | 54 / 68 · 54 / 68 | 54 / 68 · 54 / 68 | 10 (51) | 21 |
| E2B L0 sliding | 2K / 128K | 16 | 8 | 172 / 188 · 172 / 186 | 196 / 209 · 200 / 209 | 199 / 208 · 197 / 207 | 70-72 (117-121) | 27 |
| E2B L4 full | 2K | 1 / 4 / 16 | 4 / 16 / 64 | 66 / 73 · 123 / 136 · 373 / 420 | 74 / 87 · 155 / 169 · 480 / 539 | 70 / 82 · 142 / 156 · 433 / 456 | 19 · 56 · 224 (222-300) | 24 · 31 · 48 |
| E2B L4 full | 8K | 1 / 4 / 16 | 16 / 64 / 256 | 78 / 88 · 196 / 214 · 685 / 790 | 108 / 123 · 302 / 320 · 1,047 / 1,078 | 94 / 108 · 246 / 267 · 885 / 951 | 41 · 151 · 634 (406-444) | 25 · 38 · 84 |
| E2B L4 full | 16K | 1 / 4 / 16 | 32 / 128 / 512 | 100 / 110 · 345 / 360 · 1,380 / 1,434 | 157 / 173 · 494 / 589 · 1,671 / 1,731 | 127 / 143 · 419 / 449 · 1,427 / 1,467 | 74 · 288 · 1,171 (451-467) | 27 · 79 · 86 |
| E2B L4 full | 32K | 1 / 4 / 16 | 64 / 256 / 1,024 | 151 / 169 · 524 / 543 · 3,062 / 3,330 | 258 / 280 · 890 / 918 · 3,310 / 3,400 | 193 / 209 · 675 / 707 · **2,493 / 2,573** | 132 · 535 · 2,220 (484-509) | 33 · 85 · 98 |
| E2B L4 full | 64K | 1 / 4 / 16 | 128 / 512 / 2,048 | 299 / 310 · 1,187 / 1,243 · 6,194 / 6,334 | 471 / 495 · 1,601 / 1,670 · = B | 368 / 386 · 1,188 / 1,320 · **4,573 / 4,655** | 262 · 1,040 · 4,236 (507-516) | 78 · 87 · 138 |
| E2B L4 full | 128K | 1 / 4 / 16 | 256 / 1,024 / 4,096 | 496 / 517 · 2,830 / 2,885 · 12,781 / 12,996 | 851 / 876 · 3,107 / 3,172 · = B | 635 / 659 · **2,248 / 2,313** · **8,993 / 10,283** | 522 · 2,078 · 8,612 (499-517) | 87 · 103 · 155 |
| E4B L0 sliding | 2K / 128K | 1 | 1 | 143 / 153 · 141 / 150 | 143 / 153 · 142 / 152 | 143 / 153 · 142 / 153 | 14 (73) | 107-108 |
| E4B L0 sliding | 2K / 128K | 16 | 16 | 323 / 333 · 322 / 333 | 353 / 366 · 353 / 366 | 353 / 366 · 353 / 367 | 105 (159-160) | 150-151 |
| E4B L5 full | 2K | 1 / 4 / 16 | 8 / 32 / 128 | 230 / 237 · 310 / 319 · 586 / 598 | 238 / 247 · 355 / 368 · 766 / 792 | 236 / 244 · 343 / 355 · 718 / 742 | 28 · 93 · 358 (299-375) | 182 · 195 · 195 |
| E4B L5 full | 8K | 1 / 4 / 16 | 32 / 128 / 512 | 270 / 278 · 417 / 425 · 1,388 / 1,418 | 315 / 326 · 586 / 606 · 1,702 / 1,752 | 294 / 304 · 519 / 535 · 1,452 / 1,486 | 71 · 272 · 1,091 (474-494) | 193-199 |
| E4B L5 full | 16K | 1 / 4 / 16 | 64 / 256 / 1,024 | 312 / 320 · 571 / 586 · 2,666 / 2,703 | 397 / 413 · 889 / 942 · 2,939 / 2,984 | 356 / 369 · 739 / 763 · **2,402 / 2,443** | 126 · 492 · 2,020 (532-546) | 191-227 |
| E4B L5 full | 32K | 1 / 4 / 16 | 128 / 512 / 2,048 | 382 / 392 · 1,182 / 1,211 · 5,263 / 5,512 | 547 / 567 · 1,477 / 1,560 · = B | 464 / 480 · 1,182 / 1,210 · **4,191 / 4,457** | 240 · 931 · 3,691 (559-582) | 194-334 |
| E4B L5 full | 64K | 1 / 4 / 16 | 256 / 1,024 / 4,096 | 547 / 563 · 2,462 / 2,518 · 10,424 / 11,347 | 842 / 869 · 2,680 / 2,726 · = B | 684 / 709 · **2,094 / 2,208** · **8,116 / 8,711** | 459 · 1,819 · 7,579 (567-590) | 194-365 |
| E4B L5 full | 128K | 1 / 4 / 16 | 512 / 2,048 / 8,192 | 1,130 / 1,168 · 4,974 / 5,053 · 20,789 / 22,846 | 1,422 / 1,497 · = B · = B | **1,116 / 1,153** · **3,825 / 4,208** · **15,734 / 17,066** | 886 · 3,429 · 15,186 (566-626) | 197 · 339 · 376 |

(c1 / c4 / c16 values separated by ·; bold = C beats even the artificially hot B.)

What the matrix shows:

* **C beats A, the realistic cold case, in every full-attention cell.**
  * The gain is 14-28% for E2B and 7-22% for E4B at ≥ 8K. At 2K, where the layer is GEMV/sync-bound, it is
    1-10%.
  * C's p99 is also lower than A's in every full-attention cell.
* **From ~1 GiB of KV per step, C also beats B**, by 10-30%. B no longer fits L3 and falls back to DRAM without
  prefetch.
  * At 512 MiB per step they tie: E2B 64K c4 1,187 vs 1,188; E4B 32K c4 1,182 vs 1,182; E4B 128K c1 1,130 vs 1,116.
  * B still wins at E2B 16K c16 and E4B 8K c16.
* **Long-context attention is DRAM-bound.** C's attention reads 500-626 GB/s at ≥ 32K, i.e. 78-97% of the
  ~645 GB/s P0 DRAM rate.
  * Attention alone (`L2R_ATTN_ONLY`, `results/p4_attn`) is 4-13% faster than inside the layer:
    E2B L4 32K 129.8 vs 135.4 µs; E4B L5 128K 794 vs 887 µs. That overhead is the layer's own traffic.
* **Concurrency scales the attention cost linearly with KV bytes.** One E4B full layer at 128K × 16 costs
  15.7 ms. 7 such readers per token make the KV alone ~110 ms per decode step. Batch-16 at 128K is a
  DRAM-bandwidth problem, not a cache problem.
* **Sliding layers are context-independent.** 2K and 128K cost the same, and both pass the gate at 128K
  positions. Their KV ring is ≤ 1 MiB per sequence.

### Counters: DRAM traffic and weight residency

`p4_ctr.sh` counts over the timed steps only (perf `--control` fifo). The per-step numbers below are per worker.
"Excess" = L2 lines in − the worker's KV lines, i.e. weight + activation refetch.

| cell | step p50 µs | DRAM MiB / step (KV MiB) | L2 lines in (KV lines) | excess, % of the weight slice |
|---|---:|---|---|---:|
| E2B L4 2K B / A / C | 66.8 / 73.9 / 71.0 | 2.0 / 5.6 / 5.2 (4) | 2,254 / 2,948 / 3,012 (729) | 10 / 15 / 15 |
| E2B L4 8K B / A / C | 78.7 / 108.3 / 94.0 | 2.0 / 14.7 / 13.5 (16) | 2,370 / 5,237 / 5,200 (2,913) | −4 / 16 / 15 |
| E2B L4 32K B / A / C | 152.0 / 258.5 / 194.2 | 2.1 / 50.4 / 48.5 (64) | 6,899 / 15,941 / 16,246 (11,651) | −32 / 29 / 31 |
| E2B L4 64K B / A / C | 299.4 / 471.6 / 367.7 | 2.2 / 101.6 / 100.2 (128) | 36,995 / 38,440 / 39,962 (23,302) | 91 / 101 / 111 |
| E2B L4 128K B / A / C | 498.0 / 851.7 / 632.0 | 2.4 / 202.2 / 200.2 (256) | 65,724 / 66,349 / 65,772 (46,604) | 125 / 129 / 125 |
| E4B L5 32K B / A / C | 381.8 / 546.8 / 464.9 | 2.4 / 113.7 / 111.4 (128) | 65,790 / 65,393 / 65,541 (23,302) | 115 / 114 / 115 |
| E4B L5 128K B / A / C | 1,135.1 / 1,429.1 / 1,114.6 | 216.1 / 449.4 / 436.0 (512) | 137,144 / 137,036 / 136,762 (93,207) | 119 / 119 / 118 |

* The E2B L4 weight slice is 0.92 MiB = 15,104 lines per worker; the 2K-8K excess (10-16%) is the broadcast and
  activation baseline (P3).
* **The weights stay resident up to ~8K at c1. They are partly evicted at 32K, and fully refetched every step
  from 64K.**
  * At 32K a worker's KV is 0.71 MiB per step; at 64K it is 1.42 MiB.
  * Weight slice + KV then exceeds the ~1.5 MiB that P1 showed a core can hold.
* The GEMV phases show the cost in step time. E2B L4, path C:

  | ctx | 2K | 8K | 16K | 32K | 64K | 128K |
  |---|---:|---:|---:|---:|---:|---:|
  | GEMV phases µs | 24 | 25 | 27 | 33 | 78 | 87 |

* The E4B L5 slice (2.3 MiB) never fits (P2), so its excess is ~115% at every context. Its FFN is L3-bound
  regardless of KV.
* **Path A reads 79-92% of its KV bytes from DRAM at ≥ 8K**, despite the 3.2× rotation; B at E4B 128K reads 42%. The rest
  is served from L3. A non-inclusive L3 still keeps part of a cyclic stream (mechanism not isolated).

### Lock 12 vs 14 ways under KV traffic

E2B L4: the whole 0.92 MiB slice is locked through `/dev/pseudo_lock`, at 12 or 14 L2 ways. Driver reloaded per
way count, parameters restored after. Step p50 / p99 µs, 3 reps:

| ctx, path | plain | lock 12 ways | lock 14 ways | GEMV phases plain / 12 / 14 µs |
|---|---|---|---|---|
| 32K B | **152.6 / 170.5** | 168.1 / 175.9 | 176.7 / 186.3 | 33.9 / 34.5 / 34.1 |
| 32K A | **258.5 / 276.8** | 266.1 / 286.2 | 269.7 / 300.3 | 32.6 / 34.7 / 34.8 |
| 32K C | **193.5 / 211.5** | 206.0 / 220.9 | 220.4 / 239.4 | 33.3 / 33.9 / 34.1 |
| 128K B | 494.9 / 515.2 | **455.1 / 477.8** | 479.1 / 505.8 | 96.3 / 35.9 / 35.3 |
| 128K A | 854.4 / 938.5 | 807.4 / 842.4 | **795.7 / 831.3** | 86.0 / 37.3 / 34.7 |
| 128K C | 631.3 / 697.6 | **610.1 / 646.1** | 630.8 / 680.3 | 86.1 / 36.5 / 34.5 |

* **At 128K the lock keeps the weights.** GEMV phases drop 86 → 37 µs, and L2 lines in drop 65.8 K → 54.3 K per
  step.
  * Locked lines held after the run: ≥ 0.957 in the worst case, ≥ 0.98 in most runs.
  * But KV confined to the 4 or 2 free ways slows attention (C: 516 → 539 / 551 µs).
  * Net gain is only 3.4% (C, 12 ways), with a lower p99 (646 vs 698).
* At 32K the lock costs 3-10%. Plain LRU still holds most of the slice there, and the lock only takes ways away
  from the KV.
* **12 ways beats 14 everywhere except 128K A** (14 ways faster there by 1.5%). Two free ways are
  too few for the KV tiles.
* The lock's GEMV time at 128K (35-37 µs) is still above the 2K value (24 µs) with the weights held. The extra is
  outside the locked slice: activations and broadcast replicas are evicted by the KV stream. Inferred from
  held ≥ 0.96, not isolated.

### AMX under KV traffic (P1 carry-over)

E2B L4, path C, 3 reps, AMX vs AVX-512 GEMV (`results/p4_amx`):

| ctx | step p50 µs | GEMV phases µs |
|---|---|---|
| 2K | 70.4 vs 70.7 | 25.4 vs 24.3 |
| 32K | 204.0 vs 192.5 | 42.8 vs 33.1 |
| 128K | 647.0 vs 630.5 | 100.1 vs 86.5 |

* AMX loses 16-29% of its GEMV time once the step streams ≥ 64 MiB of KV. This matches P1's AMX-after-KV
  slowdown.
* The batch-1 stage keeps the AVX-512 GEMV.

## Selected policy

| decision | choice | basis (observed latency) |
|---|---|---|
| attention kernel | tiled online softmax, 256 KiB K+V tiles, FP32 max / sum / weighted-V | untiled path A 326 → 257 µs; ≤ 4% either way on hot KV |
| KV staging | **path C: in-loop T0 prefetch, 8 rows ahead, always on** | 7-28% over A at ≥ 8K; costs 2-6% only if the KV is already hot, which serving never has above a 256 MiB footprint |
| explicit L3 staging copy / bulk early prefetch | **off** | +27-42% step time: its cost exceeds what it saves |
| NTA hints | off | slower than no prefetch |
| L2 lock for weights under long KV | off by default; 12 ways if used | 128K: −3.4% (C), +3-10% at 32K. 14 ways worse in 5 of 6 cells |
| GEMV kernel next to long KV | AVX-512 | AMX +16-29% GEMV time under KV |
| crossovers | KV per worker > ~0.7 MiB (E2B c1 ≥ 32K): weights start to leave L2; > 1.4 MiB (≥ 64K): weights refetched every step; KV footprint ≥ 256 MiB: prefetch beats reuse | counters + matrix above |

## Exit gate P4

| criterion | result |
|---|---|
| all contexts pass correctness | **PASS**. 712/712 runs, 2K-128K, c1-c16, paths A/B/C, lock 12/14, AMX/AVX, all at unchanged thresholds |
| chosen path minimizes measured attention / layer p99 | **PASS for the serving case.** C has the lowest p99 of the cold paths in every full-attention cell. On sliding layers and 2K E2B L4, A and C are within 1-10% (the layer is GEMV/sync-bound). Hot B is lower only where a < 256 MiB footprint is re-read every step, which serving decode does not do |
| without violating the P1 weight-residency gate | **holds only up to ~8K at c1 (E2B; KV ≤ 0.2 MiB per worker).** At 16K GEMV is +12%. At 32K the excess refetch is 29% of the slice; from 64K the slice is refetched every step (GEMV +225% at 64K). The 12-way lock holds the weights at 128K (held ≥ 0.957) but nets only 3.4%. E4B never satisfied it (P2) |
| explicit L3 staging disabled if it costs more than reuse saves | **done**: early staging is off |

**P4 exit: PASS on correctness and path selection. The weight-residency premise fails beyond ~8K context at
batch 1.** Decode attention at long context is a DRAM-bandwidth problem: 500-626 GB/s, ~80-97% of peak. Pinning
weights in L2 then saves ≤ 8% of a step, which the lock mostly gives back.

Carried into P5, serving:

* **Weights will be DRAM/L3-streamed anyway.** E2B ≈ 2.6 GiB vs ~180 MiB of L2. So the serving-relevant P4
  results are:
  * tiled online-softmax attention with T0 prefetch;
  * no staging copies;
  * AVX-512 next to KV streams;
  * per-node NT broadcast replicas (P3).
* **Expected long-context decode cost from KV alone (E4B).** At ~600 GB/s, 3.6 GiB of KV reads per token at 128K
  is ≈ 6 ms per token at c1, and ≈ 100 ms at c16. Short-context decode stays weight-bound: E2B ≈ 2.6 GiB/token.
