# P5: single-socket BF16 serving, batched stage and go / no-go (Xeon 6975P-C)

Checklist phase P5 and the go / no-go decision. Two kinds of result, kept apart throughout:

* **Stage**: one real decoder layer as a weight-stationary stage on the 90 isolated cores, every weight in L2.
  This is what one socket of a multi-socket pipeline would run.
* **Serving**: the whole model served end to end by plowrt on one socket. E2B is 4.28 GiB of BF16 weights per
  token and E4B 8.75 GiB, so serving reads every weight from DRAM each token. It is **not L2-resident**, and no
  result here claims it is.

* Raw evidence: `/tmp/g4c/l2r/results/{p5_grid,p5_ab1,p5_ab2,p5_ab3,p5_ident,p5_batch,p5_batch2,p5_stab,p5_stab2,p5_stab3,p5_srvctr}`,
  `/tmp/g4c/final/<model>/{plow.comb,gate.comb,report-vllm.comb}`, plans in `/tmp/g4c/l2r/plan`.
* Tools: `runtime/cpu/bench/l2r/` (`l2r_layer.c` batch, `p5_sum.py`), `plowc stage-plan`, `PLOW_CPU_COMBINE`
  in plowrt. Commits 17921a80, 53581658, 4f03ca12.
* Host state as P0-P4. The 90 isolated workers, SNC3, a zero-latency CPU QoS request held, THP madvise,
  `numa_balancing=0`. `pseudo_lock_sram` stays loaded at 12 L2 ways except where 14 is stated.

## P5.1 Serving: what changed in plowrt

The stage work (P3) showed that synchronisation, not GEMV, sets a layer's time. A decode profile of plowrt
serving E2B (`cpu_profile`, `PROF_DUMP`) found the same thing at full-model scale.

| E2B decode step (c1) | ms |
|---|---:|
| span | 10.6 |
| median-slice GEMV (≈ the 7.1 ms DRAM floor) | 7.6 |
| slice imbalance | ~1.0 |
| single-core serial ops (norms at M = 1) | ~0.4 |
| hand-off gaps | ~1.6 |

The gaps come from the completion counters. Every slice of a full-width op `fetch_add`s one shared line. The
last bump took 2.0 µs median and 9.8 µs (GEMV) / 15.9 µs (GLU) at worst, about 1 ms per step. Changing the poll
period (every 15 / 63 / 255 spins) did not move it, so the cost is writer contention, not polling.

`PLOW_CPU_COMBINE=G` (`--cpu-combine`, `docs/runtime/cpu.md`) adds combining-tree counters. Groups of `G`
same-node executors bump a sub-counter, and the group's last arrival adds the group's share to the real counter.
Consumers and thresholds are unchanged.

* Profile: last bump 2.0 → 0.26 µs, step span 10.5 → 9.4 ms.
* Greedy outputs identical with and without it: 12/12 per model, including 4 concurrent requests.
* Tests: `cpu_interp` 12/12 (incl. `combining_tree_orders_dependencies`), `token_batch_tail` 4/4, knob specs.

Other existing flags, measured by A/B on E2B c1 TPOT p50 (2 reps, T4 order):

| arm | TPOT p50 ms | vs ctl |
|---|---:|---:|
| ctl | 10.72-10.75 | |
| 2 MiB huge pages | 10.69-10.77 | 0% |
| huge pages, no weight affinity | 10.79-10.83 | +0.6% |
| `PLOW_CPU_SRAM` (L3 lock of 209 MiB of layer tensors, L2 lock of worker scratch) | 11.26-11.27 | **+4.8%** |
| `PLOW_CPU_SRAM` with the driver at `l3_ways=0` (L2 scratch lock only) | 10.78 | +0.4% |

Locking L3 for weights costs serving time: the locked share is a small fraction of the weights read per token,
and the lock takes cache from everything else. L3 stays an unlocked cache. COMBINE gains 7-10%; nothing else
moves serving.

`G` choice at c1, ISL 1000 (2 reps): E2B ctl 10.48 / 10.49 ms vs G 8 / 16 / 30 = 9.70 / 9.65 / 9.70. E4B
18.85 / 18.88 vs 17.75 / 17.65 / 17.80. `G=16` is the recipe value.

### Grid: ctl vs `PLOW_CPU_COMBINE=16`

pk90 packets (n_cu 90, 16K max context), ISL 1900 / 15900, OSL 128, 2 reps; means. Ratio = COMBINE / ctl, lower is
better. `*` = repeat spread > 10% on either arm.

| model | ISL | c | TTFT p50 ms | TTFT P99 ms | TPOT p50 ms | TPOT P99 ms | out tok/s |
|---|---:|---:|---|---|---|---|---|
| E2B | 1900 | 1 | 265.7 → 262.9 (0.99) | 319.3 → 301.0 (0.94)* | 10.9 → 9.8 (**0.90**) | 10.9 → 9.8 (0.90) | 77.2 → 84.9 (1.10) |
| E2B | 1900 | 4 | 484.0 → 545.0 (1.13)* | 1077.4 → 1059.1 (0.98) | 17.9 → 16.5 (0.92) | 19.5 → 18.6 (0.95) | 184.4 → 193.5 (1.05) |
| E2B | 1900 | 16 | 896.8 → 552.6 (0.62) | 4487.6 → 4366.7 (0.97) | 50.2 → 50.2 (1.00) | 54.6 → 53.3 (0.98) | 280.4 → 288.1 (1.03) |
| E2B | 15900 | 1 | 1668.7 → 1667.5 (1.00) | 1774.3 → 1797.7 (1.01) | 11.6 → 10.6 (**0.91**) | 11.8 → 10.7 (0.90) | 40.6 → 42.3 (1.04) |
| E2B | 15900 | 4 | 2798.3 → 2633.1 (0.94)* | 7132.4 → 6677.9 (0.94)* | 50.8 → 50.0 (0.99)* | 61.2 → 57.4 (0.94) | 55.5 → 56.8 (1.02) |
| E2B | 15900 | 16 | 2681.9 → 3101.4 (1.16)* | 27134.9 → 27168.3 (1.00) | 251.8 → 249.6 (0.99) | 259.2 → 259.9 (1.00) | 58.6 → 58.5 (1.00) |
| E4B | 1900 | 1 | 396.4 → 394.1 (0.99) | 436.7 → 449.9 (1.03)* | 19.3 → 17.9 (**0.93**) | 19.3 → 17.9 (0.93) | 44.9 → 47.9 (1.07) |
| E4B | 1900 | 4 | 1249.2 → 1042.9 (0.83)* | 1552.6 → 1549.4 (1.00)* | 23.5 → 25.3 (1.08)* | 29.6 → 30.0 (1.01) | 119.8 → 122.4 (1.02) |
| E4B | 1900 | 16 | 2636.0 → 2657.4 (1.01) | 5463.3 → 5546.9 (1.02) | 63.1 → 62.1 (0.99) | 80.4 → 79.8 (0.99) | 191.6 → 193.1 (1.01) |
| E4B | 15900 | 1 | 3181.0 → 3180.4 (1.00) | 3361.7 → 3351.8 (1.00)* | 20.5 → 19.0 (**0.93**) | 20.6 → 19.2 (0.93) | 22.1 → 22.8 (1.03) |
| E4B | 15900 | 4 | 4957.1 → 5706.7 (1.15)* | 12548.5 → 12710.5 (1.01) | 93.7 → 87.1 (0.93)* | 108.1 → 107.1 (0.99) | 30.2 → 30.4 (1.01) |
| E4B | 15900 | 16 | 4996.2 → 4998.4 (1.00) | 51685.0 → 51441.2 (1.00) | 437.9 → 435.9 (1.00) | 454.7 → 454.0 (1.00) | 33.3 → 33.4 (1.00) |

* **c1:** decode is 7-10% faster at both lengths.
* **c4 / c16:** decode is within noise. There decode is bound by per-sequence work (16K: KV reads), not by
  hand-off.
* **TTFT P99:** 0.94-1.03 in every cell, so COMBINE never makes prefill worse. The TTFT p50 cells above 1.10
  are all spread-flagged.
* **KV growth:** at 16K, c16 TPOT is 250 ms (E2B) and 436 ms (E4B) against 11 / 20 ms at c1. Sixteen 16K
  sequences put 0.5-1 GiB of KV in every decode step.
* **Not run:** 32K and 128K serving. The recipe packets stop at 2K and the pk90 packets at 16K. Long context is
  measured at stage level (P4 and the final table below).

### Strict comparison vs vLLM 0.30 (recipes with `PLOW_CPU_COMBINE=16`)

`campaign.py report`, exit 0, **MATCHED + EQUIVALENT** on every cell. Closed loop, ISL 1000 / OSL 128, 2 repeats,
plowrt 53581658. Same packets and vLLM 0.30 arms as the 59a36e3d reports. The FP32 gate was re-captured with
this binary: E2B kl_mean 0.00041 vs vLLM 0.00051, E4B 0.00073 vs 0.00140, top-1 and needle 1/1.

| Model | Cell | Total throughput | TTFT P99 | TPOT P99 | Peak memory |
|---|---|---:|---:|---:|---:|
| E2B | c1 | 3.67x (was 3.24x) | 0.52x * | 0.25x (was 0.29x) | 0.27x * |
| | c8 | 2.17x (2.03x) | 0.75x | 0.45x (0.48x) | 0.29x * |
| | c32 | 1.46x (1.36x) | 0.81x | 0.68x (0.73x) | 0.34x * |
| E4B | c1 | 3.52x (3.25x) | 0.51x * | 0.27x (0.29x) | 0.35x * |
| | c8 | 2.37x (2.22x) | 0.77x | 0.40x (0.43x) | 0.37x * |
| | c32 | 1.55x (1.50x) | 0.84x | 0.64x (0.66x) | 0.43x |

The full 12-row tables are in `/tmp/g4c/final/<model>/report-vllm.comb/comparison.md`. The rows are in
`docs/bringup/results/gemma4-xeon6-bf16-20261006/comparison.{md,csv}`. Both recipes now set
`PLOW_CPU_COMBINE = "16"`.

**llama.cpp: no valid strict report.**

* `campaign.py report` refuses the recorded llama.cpp arm (exit 2): its own repeats differ in `io_key`.
* llama.cpp ends some greedy requests early. In g32 one repeat has outputs of 20-32 tokens where the other has
  128, so the arm cannot be paired.
* That arm was recorded 2026-10-05, and the BF16 GGUF was deleted after it.
* llama.cpp is therefore still the baseline dropped on 2026-10-06 (see `comparison.md`). A valid comparison needs
  a fixed output length on the llama.cpp server and a new GGUF.

### Serving counters (E2B, 2K, c1, recipe packet)

ISL 1000 / OSL 512, 16 requests, system-wide counters over the run, per output token (prefill included):

| TPOT p50 / p99 | DRAM read | L2 lines in | LLC misses |
|---|---:|---:|---:|
| 9.33 / 9.36 ms | 3,964 MiB | 79.5 M | 69.2 M |

Each token re-reads the whole 4.28 GiB of weights from DRAM, about 445 GB/s against the measured 612-669 GB/s.
That is why decode is a DRAM-bound 9.3 ms here, and why a single socket cannot hold the model in L2.

## P5.2 Batched stage: B ≤ 16 with AMX

The pipeline needs every stage to run a batch: B tokens of B sequences per step. That uses AMX (16-row tiles) and
spreads each weight read over B tokens.

`l2r_layer` gains `L2R_BATCH=b` and `L2R_ROWS=dir,...` (commit 53581658):

* Each row is a distinct sequence with its own dump: hidden state, per-layer input, RoPE, KV cache and FP32
  reference. `ref_layer.py` takes a text offset; rows are `r × 997` tokens apart.
* AMX GEMV: A tile rows = b. AVX-512: 4 weight rows × up to 4 batch rows in registers.
* Norms, RoPE and residuals run per row. Broadcasts carry b rows.
* **Row-group attention:** row r gets workers `[90 r / b, 90 (r + 1) / b)`. They split that row's KV positions and
  combine only each other's partials. With b = 1 the group is all 90 workers, i.e. the P2-P4 scheme; B=1 errors
  are bit-identical to the P4 binary.
  * Without it, all 90 workers attended a slice of all 16 rows and combined 90 partials per row: B=16 took 449 µs.
    With row groups it takes 240 µs: attention 106 → 19 µs, combine 89 → 5 µs.
* Vectorised GELU (`exp`-based tanh): B=16 240 → 222 µs.
* Every row is gated against its own dump (`p2_gate.py`). 76 new reference dumps passed the HF self-check.

### Batch sweep

Distinct rows, `repnt` + dissemination, P4 KV policy at 16K (256 KiB tiles, T0 prefetch distance 8), 2 reps,
warm KV. µs / token = step p50 / B. GEMV = critical-path compute of the six GEMV phases. Attention = attention +
combine.

| stage (weights / core, mean) | B | AMX step p50 / p99 µs | AMX µs / token | AVX µs / token | AMX GEMV / attention / barrier µs |
|---|---:|---:|---:|---:|---|
| E2B L0 2K (0.77 MiB) | 1 | 46.8 / 62.7 | 46.8 | 50.6 | 20.6 / 12.5 / 22.5 |
| | 4 | 70.5 / 87.0 | 17.6 | 25.6 | 39.9 / 14.1 / 28.2 |
| | 8 | 109.3 / 136.1 | 13.7 | 21.9 | 77.9 / 15.8 / 39.9 |
| | 16 | 213.1 / 236.1 | **13.3** | 20.9 | 167.1 / 23.7 / 56.8 |
| E2B L0 16K (sliding: 511 KV rows) | 16 | 212.6 / 231.7 | 13.3 | 20.9 | 167.0 / 23.4 / 56.5 |
| E2B L4 2K (0.92 MiB, full attention) | 1 | 62.9 / 96.1 | 62.9 | 65.2 | 26.2 / 24.3 / 26.9 |
| | 4 | 104.6 / 115.2 | 26.2 | 32.3 | 54.0 / 36.8 / 37.7 |
| | 16 | 389.9 / 405.8 | **24.4** | 31.4 | 233.6 / 129.9 / 76.4 |
| E2B L4 16K | 1 | 111.9 / 140.4 | 111.9 | 104.5 | 33.4 / 66.5 / 36.6 |
| | 16 | 1254.1 / 1390.1 | **78.4** | 86.4 | 291.0 / 928.7 / 238.8 |
| E4B L5 2K (2.25 MiB: does not fit) | 1 | 226.4 / 240.1 | 226.4 | 224.9 | 183.9 / 30.0 / 65.5 |
| | 16 | 691.4 / 712.7 | **43.2** | 58.9 | 450.5 / 217.7 / 144.9 |

* **AMX pays off with batch.** At B=16 an E2B layer costs 13.3 µs per token against 46.8 at B=1 (3.5x); AVX
  reaches 20.9. At B=1 the two GEMVs are equal: one row cannot fill a tile.
* **Batch cost is activation traffic.** From B=1 to 16, E2B L0's GEMV phases grow 20.6 → 167 µs, about 9.8 µs per
  extra row. The weights are read once per step either way. What grows is the all-gather: every core gathers
  every row's q / attention / o / act / down vectors, 37 KiB per row (~3.4 GB/s per core into each core). The
  per-row norms and residuals add a little.
* **Broadcast mode at B=16:** `rep` 204.8 µs and `repcld` 205.6 µs vs `repnt` 213.0 µs. With 16 rows the
  replicas are better left in L3 than streamed to DRAM. At B=1, `repnt` remained best (P3).
* **AMX slice imbalance.** AMX splits each matrix in 16-row tiles. o / down / per-layer-input projections have
  1,536 rows = 96 tiles over 90 cores, so 6 cores get two tiles. Max / mean weight bytes per core: AMX 1.59 (E2B L0,
  1,277,952 / 803,908 B), 1.52 (L4, 1,458,176 / 961,195 B), 1.16 (E4B L5); AVX (4-row units) 1.11 / 1.11 / 1.06.
  * Tried: a balanced split in 4-row units (AMX on whole tiles, AVX-512 on the 4-12-row tail). **Slower** at every
    B ≥ 2: E2B L0 B=16 278.3 µs vs 213.1, L4 2K 444.8 vs 389.9, L4 16K 1,289 vs 1,254. The AVX tail costs more than
    the idle tile slots it removes. Reverted; raw in `/tmp/g4c/l2r/results/p5_batch2`.
* **Long context:** at 16K, B=16 is 74% attention. 16 rows × 16K × 2 KiB is 512 MiB of KV per step, about
  550 GB/s. The stage is KV-bandwidth bound there, as P4 found for B=1.

**Gate: every run passes except two cases, both from one statistic.**

* E2B L4 2K, row `o5`: `pact` 1.39e-4 vs its BF16 bar 6.77e-5 (2.06x), at B ≥ 8 with AMX.
* E2B L4 16K, row `o7`: `pp` 1.79-1.85x, at B ≥ 8.
* In every row, `ref.pact` has an effective element count ((Σx²)² / Σx⁴) of 1.0-3.2 out of 256: one outlier
  channel of the per-layer input dominates. Its rel-RMS error is therefore one element's error. o5's "BF16 bar"
  is 20x below its own `pg` bar, by chance. `pp` inherits the same element.
* The failure reproduces bit for bit with the P4 binary, on AVX and AMX, with 2 workers or 90.
* In the failing rows the layer output passes: `out` ratio ≤ 1.5 and cosine ≥ 0.9999998.
* The gate's thresholds are fixed, so these runs are recorded as FAIL with this cause. A per-boundary
  effective-sample check would remove the false failure but is not added here.

### Residency at B=1 / 16: plain vs locked

| stage | B | plain p50 µs | locked p50 µs | locked / plain | L2 lines in per worker per step (plain → locked) | held after |
|---|---:|---:|---:|---:|---|---:|
| E2B L0 2K, 12 ways | 1 | 49.2 | 52.4 | 1.07 | 1,337 → 2,065 | ≥ 0.99 |
| | 16 | 218.4 | 235.1 | 1.08 | 25,901 → 32,365 | ≥ 0.99 |
| E2B L0 16K, 12 ways | 16 | 219.4 | 235.4 | 1.07 | 26,106 → 32,424 | ≥ 0.99 |
| E2B L4 2K, 14 ways | 1 | 64.7 | 81.5 | 1.26 | 2,061 → 7,168 | 0.99+ |
| | 16 | 397.9 | 425.8 | 1.07 | 52,104 → 59,982 | 0.99+ |
| E2B L4 16K, 14 ways | 1 | 111.6 | 134.9 | 1.21 | 2,769 → 14,008 | 0.99+ |
| | 16 | 1255.9 | 1270.1 | 1.01 | 138,237 → 154,708 | 0.99+ |

E2B L4's largest AMX slice is 1,458,176 B per core. That exceeds the 12-way budget (1,376,256 B), so it was locked
with 14 ways (1,605,632 B).

* **The lock holds (99%+ of locked lines) but costs 1-26%.** With 12-14 of 16 ways locked, KV, activations and
  attention tiles thrash the 2-4 free ways: L2 lines in rise 1.2-5x.
* Without the lock, the weights already stay resident at these footprints. At B=1 lines in per step are about 2x
  the KV slice (P4).
* For a stage the right setting is no lock, or as few ways as the weights need. A pipeline stage should be sized
  (plan below) so that weights plus the B-row activation working set fit L2 unlocked.

## P5.3 Stage plan: plowc decides the partition

`plowc --hf-dir <ckpt> stage-plan` (`crates/plowc/src/stage_plan.rs`; `docs/runtime/cpu.md` "Pipeline stage plan")
decides how many layers or which part of a layer each socket holds:

* **Inputs:** the checkpoint's tensor shapes, `--cores`, `--l2-weight-kib` (`auto` = the driver's lockable L2 per
  core), `--batch`, `--ctx`.
* **Units, in execution order:** per-layer-input projection; per layer attention, FFN and PLE block; the tied
  LM head. KV-shared layers carry no k/v weights.
* **Packing:** greedy, in order. FFN (intermediate rows) and head (vocabulary rows) split at multiples of
  `16 × cores` rows. Attention never splits. A KV-shared layer whose source attention sits on another stage is
  listed as needing that cache's replica.
* **Predicted time per stage:** bytes per core / GEMV rate + exchanges × exchange cost + KV bytes / KV rate +
  (B − 1) × activation bytes / all-gather rate. The constants come from P2-P5.
  * Check at B=1: E2B L0 predicted 44 µs, measured 47.
  * Check at B=16: E2B L0 predicted 209 µs (44 + 15 × 37,376 B of gathered activations per row / 3.4 GB/s), measured
    213. A pipeline stage also gathers each layer's output (6,144 B per row), which the single-layer bench does not;
    the plan includes it (236 µs for that layer).

Plans at 1,344 KiB per core (12 ways), 90 cores:

| model | B | ctx | stages | layers / stage | bottleneck stage µs | token latency ms | tok/s (sequences in flight) |
|---|---:|---:|---:|---:|---:|---:|---|
| E2B | 1 | 2K | 38 | 0.92 | 99.2 | 2.4 | 9,594 (38) |
| E2B | 1 | 16K | 38 | 0.92 | 148.2 | 2.7 | 6,529 (38) |
| E2B | 16 | 2K | 38 | 0.92 | 648.9 | 12.5 | 24,468 (608) |
| E2B | 16 | 16K | 38 | 0.92 | 1431.8 | 18.0 | 11,136 (608) |
| E4B | 1 | 2K | 77 | 0.55 | 74.1 | 3.8 | 12,640 (77) |
| E4B | 1 | 16K | 77 | 0.55 | 172.0 | 4.5 | 5,650 (77) |
| E4B | 16 | 2K | 77 | 0.55 | 614.6 | 21.1 | 25,823 (1232) |
| E4B | 16 | 16K | 77 | 0.55 | 2180.5 | 32.1 | 7,321 (1232) |

Tok/s = B × stages / (stages × bottleneck): every socket runs one batch per bottleneck period. Token latency
is the sum of stage times plus hops; a sequence's next token waits for the full ring, stages × bottleneck.

* **E2B: 38 stages, 0.92 layers each.** Layers split mid-FFN; a typical stage is one layer's attention + FFN tail
  + the next layer's FFN head.
* **E4B: 77 stages, 0.55 layers each.** Every layer spans about two sockets.
* **The tied 262,144 × hidden LM head takes 7 (E2B) / 11 (E4B) of those sockets.**
* **B=1: sync dominates every planned stage.** 7-18 exchanges × ~3.9 µs against ~20 µs of GEMV. Fewer exchanges
  per stage is worth more than more L2.
* **B=16: the activation all-gather dominates.** 268-432 µs of the 343-649 µs E2B stages at 2K; GEMV stays ~20 µs.
  At 16K the full-attention stages are KV-bound (909 µs of the 1,432 µs bottleneck).

## Final benchmark table

Stage rows: one real layer on the 90 workers, 2 reps, means.

* GEMV = critical-path compute of the six GEMV phases; Sync = mean barrier wait summed over the 8 phases;
  Attention = attention + combine; Total = step p50 (p99 in brackets).
* L2 = lines in per worker per step; DRAM = CAS reads per step.
* Accuracy = the fixed P2 gate on every boundary of every row.

| Workload | Context | Batch/concurrency | CAT ways | Weight bytes/core | GEMV µs | Sync µs | Attention µs | Total layer µs | TPOT p50/p99 | L2/LLC misses | Accuracy | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| Real E4B BF16 stage | 2K | 1 | 12 | mean 2,359,296, max 2,510,848 (1,376,256 locked) | 116.6 | 39.1 | 40.6 | 171.1 (184.0) | n/a | L2 25.1K; DRAM 2.4 MiB | PASS, out cos 0.9999998 | Baseline. AVX; 0.94 MiB/core of FFN streamed from L3; lock held ≥ 0.987 |
| Real E4B BF16 stage | 16K | 1 | 12 | mean 2,359,296, max 2,510,848 (1,376,256 locked) | 123.3 | 44.5 | 103.9 | 240.7 (254.7) | n/a | L2 36.0K; DRAM 2.4 MiB | PASS, out cos 0.9999999 | KV interference: 64 MiB KV per step served from L3; held ≥ 0.997 |
| Real E4B BF16 stage | 128K | 1 | 12 | mean 2,359,296, max 2,510,848 (1,376,256 locked) | 145.9 | 100.0 | 629.6 | 793.9 (886.4) | n/a | L2 120K; DRAM 156 MiB | PASS, out cos 0.9999999 | Long context: 512 MiB KV per step, part from DRAM; held ≥ 0.962 |
| E2B BF16 serving | 2K | 1 | mixed (no lock) | n/a | n/a | n/a | n/a | n/a | 9.33 / 9.36 ms | per token: L2 79.5M lines, LLC 69.2M misses, DRAM 3,964 MiB | FP32 gate PASS (kl 0.0004 vs vLLM 0.0005) | Whole model not L2 resident: DRAM-bound at ~445 GB/s; `PLOW_CPU_COMBINE=16`; 3.67x vLLM 0.30 throughput at c1 |
| E2B BF16 stage (added) | 2K | 16 | 0 | mean 803,908, max 1,277,952 | 167.1 | 56.8 | 23.7 | 213.1 (236.1) | n/a | L2 25.9K; DRAM 3.3 MiB | PASS | AMX, layer 0 (sliding): 13.3 µs per token |
| E2B BF16 stage (added) | 16K | 16 | 0 | mean 961,195, max 1,458,176 | 291.0 | 238.8 | 928.7 | 1254.1 (1390.1) | n/a | L2 138K; DRAM 167 MiB | FAIL pp row o7 1.79x (single-element boundary); out PASS | AMX, layer 4 (full): KV-bound, 512 MiB per step |
| E2B BF16 stage, stability (added) | 16K | 16 | 14 | mean 961,195, max 1,458,176 (locked) | 36-min run | 36-min run | 36-min run | 1,292.6 mean p50 (1,368.9 mean p99; drift 1.010 / 1.047) | n/a | L2 155.9K; DRAM 137 MiB | 80/80 runs identical: out cos ≥ 0.99999989; only pp row o7 (single element) | 80 runs × 20,000 steps; lock held ≥ 0.965; 3.56 GHz, 595 W package |

## P5.4 Stability

E2B L4 (full attention) at 16K, B=16 distinct rows, AMX.

* L2 lock 14 ways (1,605,632 B per core) over the whole resident slice.
* P4 KV policy; 512 MiB of KV per step.
* 80 back-to-back runs of 20,000 steps: 1.6 M steps, **36 min**.
* Each run is gated (16 rows) and counted (DRAM CAS, L2 lines in). The lock's held fraction is measured before
  and after each run.
* turbostat every 10 s throughout. `p5_stab_sum.py`, `/tmp/g4c/l2r/results/p5_stab3`.

| metric | first run | last run | min | max | mean | drift (last ¼ / first ¼) |
|---|---:|---:|---:|---:|---:|---:|
| step p50 µs | 1,294.0 | 1,336.9 | 1,250.4 | 1,360.6 | 1,292.6 | 1.010 |
| step p99 µs | 1,430.8 | 1,418.9 | 1,280.0 | 1,513.7 | 1,368.9 | 1.047 |
| DRAM MiB per step | 144.6 | 135.7 | 132.9 | 148.2 | 137.1 | 0.985 |
| L2 lines in per worker per step | 156,790 | 156,311 | 155,144 | 156,790 | 155,913 | 1.001 |
| lock held after the run | 0.9996 | 0.9995 | 0.9651 | 0.9998 | 0.9966 | 0.998 |

* **No drift:** p50 +1.0%, p99 +4.7% within its 1,280-1,514 µs band, counters flat.
* **Lock:** the locked weights stay ≥ 96.5% held; mean 99.7%, ≥ 99.4% before every run.
* **Correctness:** identical in every run. Output cosine ≥ 0.99999989 (first and last step of each run). The same
  single gate failure (row `o7`, `pp`, the single-element boundary of P5.2) in all 80 runs and nothing else.
* **Frequency and power:**
  * busy frequency 3,560 MHz mean, 3,499-3,642;
  * package 595 W mean, 581-606;
  * DRAM 38.4 W mean, 36.3-39.7;
  * 100% busy (the workers spin between phases).
* Binary: the committed 16-row AMX partition (the P5.2 sweep binary).

## Checklist status

| P5 item | status |
|---|---|
| Integrate the weight-stationary stage into the Plow CPU backend behind a flag | **partial**. The stage's sync finding is in plowrt as `PLOW_CPU_COMBINE` (recipe default). The L2-resident stage itself is a benchmark (`l2r_layer`), not a plowrt mode; it only pays off across several sockets |
| Keep whole-model claims separate from stage residency | done: every table labels stage vs serving |
| Real E2B / E4B end to end vs current Plow and vLLM | done: ctl vs COMBINE grid; strict reports vs vLLM 0.30 MATCHED + EQUIVALENT. llama.cpp has no valid arm (above) |
| TTFT / TPOT p50 / p99, tok/s per stream, aggregate, power | done except p95 (the bench client records p50 / p99) and serving power. Stage power is in P5.4 |
| Contexts 2K, 16K, 32K, 128K where supported | serving 2K / 16K (the packets' maximum); stage 2K-128K |
| Concurrency 1 / 4 / 16 | done (grid); stage batch 1-16 |
| 30-60 min stability with correctness checks, counters, frequency, p99 drift | P5.4 |
| Stage result and DRAM-backed serving result, labeled | final table |
| Publish failures and regressions | this file: lock costs 1-26%, `PLOW_CPU_SRAM` +4.8%, the single-element gate failures, llama.cpp |

**Exit gate P5:**

* Numerical gates pass, with the two single-element boundaries explained above.
* p99 is reproducible: the repeat spread of the qualified serving cells is ≤ 10% except where flagged.
* Stage-level L2 residency holds under BF16 KV traffic: locked lines are 96-99.9% held through 128K.

## Go / no-go

| item | status |
|---|---|
| P0 environment reproducible | yes: `isolate.sh`, P0 baseline |
| P1 L2 weight residency robust under KV interference | yes, with a cost. Held ≥ 0.96 with the lock; without it the weights stay resident at ≤ 1.4 MiB per core and short context. The lock costs 1-26% by starving KV and activations |
| P2 real layer correct, breakdown understood | yes |
| P3 sync optimised and stable | correct and stable; the ≤ 4 µs target is infeasible. 8 exchanges per layer at ~4 µs at B=1, plus ~10 µs per extra row at B=16 |
| P4 KV policy selected | yes: 256 KiB tiles + in-loop T0 prefetch; no staging, no lock for KV |
| P5 serving validated at p99 and sustained load | yes for serving (strict reports, grid); stage stability in P5.4 |

**Recommendation: GO for a two-socket chained-stage experiment. NO-GO for building the full multi-socket
pipeline yet.**

The stage numbers support the pipeline idea. An E2B layer at B=16 costs 13-24 µs per token at ≤ 2K. A 38-stage
E2B pipeline at B=16, 2K projects a 649 µs bottleneck stage: ~24,500 tok/s over 608 sequences in flight,
a per-sequence TPOT of 38 × 649 µs ≈ 24.7 ms, and ~640 tok/s per socket against ~288 for one socket serving c16
today (9.3 ms TPOT at c1). That is a projection with an assumed 5 µs hop, not a measurement.

The projection rests on unmeasured pieces. Before more sockets, the two-socket experiment must establish:

1. **The socket-to-socket hop.** The plan assumes 5 µs per hop; nothing here measures it. The activation per hop
   is B × hidden × 4 B (+ the partial down sum when an FFN splits).
2. **Two chained planned stages streaming tokens**, including a split FFN (partial sum forwarded), against a
   chained reference.
3. **Fewer exchanges per stage.**
   * Sync is 39-70 µs of a 61-99 µs planned E2B stage at B=1.
   * The activation all-gather grows ~10 µs per batch row.
   * Candidates: fold norm / RoPE into owners, fuse qkv → attention ownership, BF16 broadcasts where the
     reference rounds anyway.
4. **The LM head**: 7-11 sockets for the tied head. Vocabulary-parallel head stages, or a DRAM-backed head on
   fewer sockets.
5. **Long context is KV-bandwidth bound** (74% of a B=16 16K stage). Pipelining does not change that; KV placement
   per stage does (P4).

Blockers carried forward:

* lock costs 1-26%, so plan for unlocked residency;
* sync floor at B=1, activation all-gather at B=16 (~10 µs per extra row per layer);
* unmeasured hop;
* head size;
* KV bandwidth at ≥ 16K;
* the single-element `pact`/`pp` gate statistic;
* no valid llama.cpp arm.
