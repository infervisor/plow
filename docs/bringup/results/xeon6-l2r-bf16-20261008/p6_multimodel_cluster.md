# P6: Gemma 4 family on the L2-resident socket, and a Groq-style multi-socket design (Xeon 6975P-C)

P1-P5 for E2B, E4B, 12B, 26B-A4B and 31B, then a cluster design built from the measured single-socket stages.

* **Stage**: one socket's share of one layer. This is a whole layer, a tensor-parallel (TP) slice, an MoE head or an
  MoE expert group, run on the 90 isolated cores with every weight in L2.
* **Cluster numbers are projections.** Measured stage times are placed on the `plowc stage-plan` partition. The
  socket-to-socket hop (5 µs) and the cross-socket all-reduce (10 µs) are not measured; a sensitivity case uses
  10 / 25 µs. Nothing here ran on more than one socket.

Sources:

* Raw evidence: `/tmp/g4c/l2r/results/{m_sweep,m_res,m_cold}` and P5's `p5_batch`. Plans are in
  `/tmp/g4c/l2r/plan/{all,pipe}` and references in `/tmp/g4c/l2r/ref`.
* Tools in `runtime/cpu/bench/l2r/`:
  * `ref_layer.py` with `[tp]` = `S` or `moe:G`;
  * `l2r_layer.c` (TP slice, MoE head and expert-group stages);
  * `stage_sum.py`, `cluster.py`;
  * `p2_gate.py`, plus the zero-partial rule below.
* `plowc stage-plan --split tp|pipe`: commits 0eb35106, 474673da and this one.
* Host as in P0-P5: 90 isolated workers, SNC3. `pseudo_lock_sram` is at 12 L2 ways (1,344 KiB per core) except
  where 14 ways (1,568 KiB) is stated.

## P6.1 What one socket holds

`plowc stage-plan --split tp` chooses the partition. A dense layer that does not fit one socket becomes a TP group of
the smallest socket count S that fits:

* each rank holds whole q heads with their KV heads; a KV head is replicated when S exceeds the KV heads;
* each rank holds an FFN row block;
* o and down are split by K, so each layer needs two cross-socket all-reduces.

The bench emulates those all-reduces by adding the other ranks' FP32 partial sums from the reference dump.

An MoE layer becomes two stages:

* **Head socket:** attention, the dense MLP and the router.
* **Expert sockets:** each holds 9-10 experts, every expert striped over all 90 cores. They return a partial sum to
  the head socket.

| model | layer slice measured | weights / core, mean | largest core (AVX 4-row / AMX 16-row units) | L2 budget used |
|---|---|---:|---|---|
| E2B | whole layer L0 (sliding), L4 (full) | 0.77 / 0.92 MiB | AMX 1.22 / 1.39 MiB | 12 / 14 ways (P5) |
| E4B | TP2 rank 0: L0 (sliding), L5 (full) | 1.00 / 1.14 MiB | AVX 1.09 / 1.25; AMX 1.33 / 1.47 MiB | 12 ways |
| 12B | TP4 rank 0: L0, L5 | 1.19 / 1.31 MiB | AVX 1.26 / 1.37; AMX 1.50 / 1.59 MiB | 12 ways plain; 14 ways for L5 |
| 26B-A4B | head socket L0, L5 | 1.12 / 1.42 MiB | AVX 1.21 (L0) | 12 / 14 ways |
| 26B-A4B | expert group (13 groups of 9-10 experts) | 1.26 MiB | AVX 1.29; AMX 2.15 MiB (> L2) | 12 ways; AMX spills to own L3 |
| 31B | TP8 rank 0: L0, L5 | 1.27 / 1.44 MiB | AVX 1.43 / 1.60 MiB | 14 ways |

* **The 31B full layer is 130 MiB per TP8 socket.** It fits only at 14 ways (1,568 KiB per core).
* **Only an AVX partition fits the expert group.**
  * AMX splits each expert's 704 gate / up rows into 16-row tiles: 44 tiles over 90 cores.
  * About half the cores hold one gate / up tile of every expert and the rest none. The largest core holds
    2,252,800 B (max / mean 1.70), above L2.
  * AVX's 4-row units balance to 1,351,680 B (1.02).
  * The AMX partition still runs: the overflow stays in the socket's own L3, which holds nothing else (the socket's
    weights are 119 MB).
  * AMX is 1.3-1.6x faster than AVX at B ≥ 4 (P6.3), and the cluster projection uses it.
  * The lock tests (P6.4) use the AVX partition, which fits L2.
* **Rank 0 is the representative.**
  * Every rank has the same shape except the KV-replica assignment.
  * Ranks 0 and 3 of 12B TP4 pass the gate at AVX and AMX. The expert group of record is the group with the most
  (token, expert) pairs of row 0 (L0: g5, L5: g9).

## P6.2 Numerics (P2 gate, fixed thresholds)

Every row of every run is gated against its own reference dump. The bar is the BF16-mode reference: boundary rel
RMS ≤ 1.5 × BF16, out cosine ≥ 0.99999.

* **12B, 26B-A4B, 31B: every run passes.**
  * 62 sweep cells × 2 reps and B ≤ 16 rows each.
  * 20 residency cells and 30 stability runs.
* **One rule was added for expert groups.**
  * A row whose 8 routed experts all sit in other groups has an exactly zero partial.
  * Its BF16 bar is then 0 / 0, and the earlier gate reported it as a FAIL (NaN).
  * Such a row must now output exactly zero (max abs error 0). This is stricter, not looser.
  * Every such row in the sweep outputs exactly 0.
* **E4B TP2 fails in every cell, on single-element boundaries only.** The layer output passes in all 400 checks
  (ratio ≤ 1.01, cos ≥ 0.9999997).
  * The failing boundaries, at 1.5-3.8x their bar, are `pact` (rows 0, 2, 8), `pp` (rows 0, 15), `act` (row 13) and
    `down` (row 5).
  * Each has an effective element count ((Σx²)² / Σx⁴) of 1.4-30.
* **Row 0 L5 `pp` (3.77x) was traced to one BF16 rounding**, using a debug dump of the bench's vectors:
  * The bench's `pp` equals W · BF16(its own `pact`) to 6e-8, so the GEMV is right.
  * `pp` is 79% one `pact` element (#192).
  * In the TP slice, #192 is 32.1186. That is below the BF16 midpoint 32.125 (one ulp is 0.25 at 32), so it rounds
    to 32.0. The FP32 reference (32.1689) and the full-layer run (32.1284) round to 32.25.
  * The full E4B L5 layer on the same row passes `pp` at 1.00.
  * The TP slice moves `pact` by 0.01 because the other rank's partial sum enters exactly in FP32.
* This is the single-element statistic of P5.2 (E2B L4 `pact` / `pp`). The thresholds are fixed, so these cells are
  recorded as FAIL with this cause; the cluster table flags them.

## P6.3 Batch sweep (P5): one socket's stage at B = 1-16

Warm KV, 2K context, distinct rows (one sequence per row), `repnt`, 2 reps. Step p50 is the mean over reps. GEMV,
attention and barrier are the critical-path sums (`stage_sum.py`).

| stage | B=1 | B=4 | B=8 | B=16 AMX | µs / token at B=16 | B=16 AVX | B=16 GEMV / attention / barrier µs |
|---|---:|---:|---:|---:|---:|---:|---|
| 12B L0 TP4 (sliding) | 48.3 | 91.7 | 180.9 | 359.5 | **22.5** | 471.2 | 248 / 37 / 113 |
| 12B L5 TP4 (full) | 58.6 | 124.1 | 247.3 | 439.5 | **27.5** | 558.7 | 277 / 81 / 130 |
| 31B L0 TP8 | 56.2 | 122.0 | 246.5 | 466.1 | **29.1** | 590.9 | 330 / 38 / 151 |
| 31B L5 TP8 | 67.2 (AVX 65.1) | 172.5 | 305.1 | 541.2 | **33.8** | 681.5 | 358 / 82 / 163 |
| 26B L0 head | 67.6 (AVX 65.5) | 108.4 | 254.3 | 462.6 | **28.9** | 546.3 | 250 / 141 / 140 |
| 26B L5 head | 96.7 | 199.7 | 337.4 | 589.6 | **36.9** | 711.6 | 273 / 230 / 150 |
| 26B L0 expert group g5 | 15.2 (4 pairs) | 23.8 (8) | 28.4 (11) | 40.6 (19 pairs) | **2.5** | 65.3 | 37 / - / 15 |
| 26B L5 expert group g9 | 10.0 (2 pairs) | 13.8 (4) | 22.2 (4) | 36.0 (7 pairs) | **2.3** | 47.1 | 28 / - / 17 |
| E2B L0 (P5) | 46.8 | 70.5 | 109.3 | 213.1 | **13.3** | 334 | 167 / 24 / 57 |
| E4B L0 TP2 | 50.3 | 85.0 | 150.3 | 301.6 | **18.8** | 452.8 | 261 / 18 / 63 |
| E4B L5 TP2 | 61.2 | 111.0 | 221.9 | 399.2 | **25.0** | 519.8 | 299 / 80 / 74 |

* **The B=1 floor is ~50-65 µs for every dense slice.** About 20-25 µs is barrier and exchange: P3's 8 exchanges
  at ~3-4 µs. Bigger models cost more sockets, not more time per socket.
* **AMX pays off with batch.**
  * B=16 costs 22-37 µs per token on a dense or head slice, against 48-97 µs at B=1.
  * AVX is 1.2-1.3x slower at B=16 and equal at B=1.
  * The growth with B is the activation all-gather and barrier (P5.2): about 10 µs per extra row.
* **Expert sockets are idle most of the step.**
  * The expert group finishes in 10-41 µs while its head socket takes 65-590 µs.
  * With top-8 of 128 routing, a group sees 2-19 (token, expert) pairs per step at B ≤ 16.

## P6.4 Residency (P1): plain vs locked, under counters

AVX, 2K, the same slices. The lock covers the whole resident slice: `L2R_LOCK=1` at the stated ways, and
`L2R_EXPERTS_LOCK=1` for the expert arena. "Lines in" counts L2 lines filled per worker per step.

| stage | ways | B | plain p50 µs | locked p50 µs | locked / plain | lines in, plain → locked | lock held after |
|---|---:|---:|---:|---:|---:|---|---:|
| 12B L0 TP4 | 12 | 1 | 55.5 | 56.2 | 1.01 | 1,440 → 2,143 | 0.997 |
| | | 16 | 471.2 | 593.1 | 1.26 | 58,651 → 52,776 | 0.995 |
| 26B L0 head | 12 | 1 | 62.6 | 76.1 | 1.22 | 2,760 → 4,897 | 0.999 |
| | | 16 | 537.5 | 617.1 | 1.15 | 69,438 → 60,084 | 1.000 |
| 26B L0 expert group | 12 | 1 | 18.5 | 17.1 | 0.92 | 355 → 347 | 1.000 |
| | | 16 | 67.3 | 92.7 | 1.38 | 2,311 → 5,527 | 1.000 |
| 31B L0 TP8 | 14 | 1 | 60.2 | 79.0 | 1.31 | 1,818 → 5,768 | 0.996 |
| | | 16 | 590.3 | 901.3 | 1.53 | 72,281 → 98,975 | 0.999 |
| 12B L5 TP4 | 14 | 1 | 61.0 | 80.8 | 1.32 | 2,376 → 6,320 | 0.996 |
| | | 16 | 587.6 | 842.3 | 1.43 | 70,676 → 91,918 | 0.994 |

* **Unlocked, the weights stay resident.**
  * At B=1, lines in are only 2-8% above the weight slice (the KV and activations).
  * At B=16, the extra lines are the gathered activations of 16 rows: about 37 KiB per row per phase.
  * Weights are not re-read.
* **Locking costs 15-53% at B=16.**
  * At B=1 it costs 1-32%: 12 ways 1-22%, 14 ways 31-32%. The expert group is the exception, 8% faster locked.
  * The lock leaves 2-4 ways for the KV and activation traffic, which then misses.
  * At 14 ways only 256 KiB per core is left.
* **Design rule: run unlocked.** P5 concluded the same for E2B. The lock is kept as a diagnostic: it proves the
  slice fits.

## P6.5 Long context (P4)

KV policy as in P4: 256 KiB tiles, in-loop T0 prefetch at distance 8, and AMX.

* **Path A (cold):** the KV rotates through ≥ 1.5 GiB, so every step reads it from DRAM.
* **Path B (warm):** one copy, which stays in the 480 MiB L3 when it fits.
* **A pipeline sees path A.** Each socket holds the KV of every microbatch in flight: 28-61 sequences × B.

| stage | ctx | B | warm (B) p50 µs | cold (A) p50 µs | attention (cold) µs | KV MiB / step | KV GB/s (cold) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 12B L5 TP4 | 16K | 1 | 95.0 | 116.5 | 65 | 32 | 516 |
| 12B L5 TP4 | 16K | 16 | 1068.3 | 1332.1 | 963 | 512 | 557 |
| 31B L5 TP8 | 16K | 1 | 126.9 | 147.9 | 65 | 32 | 516 |
| 31B L5 TP8 | 16K | 16 | 1216.8 | 1465.9 | 982 | 512 | 547 |
| 26B L5 head | 16K | 1 | 247.8 | 280.2 | 163 | 64 | 412 |
| 12B L5 TP4 | 128K | 1 | 441.6 | 687.5 | 548 | 256 | 490 |
| 31B L5 TP8 | 128K | 1 | 456.7 | 667.1 | 506 | 256 | 530 |
| 26B L5 head | 128K | 1 | 901.2 | 1252.5 | 1098 | 512 | 489 |

128K runs tile the 16K dump 8x (`L2R_CTX_REPEAT=8`, timing only).

* **Sliding layers are unaffected.** Their window is 1,024 tokens (512 for E2B / E4B).
* **A full-attention slice at 128K is DRAM-bound.**
  * It streams 256-512 MiB per token at ~490-530 GB/s, one socket's DRAM bandwidth.
  * That is 10-19x a sliding stage.
* **TP does not spread this KV.**
  * The 12B full layers have 1 KV head and 31B has 4 (26B: 2).
  * A TP4 or TP8 group replicates the KV head, and every rank streams the same bytes.

## P6.6 Stability

10 back-to-back runs of ~60 s each, B=16 distinct rows, AVX, locked (the worst case of P6.4).

* Every run is gated and counted; the lock's held fraction is read before and after each run.
* turbostat every 10 s. `p5_stab_sum.py`, `/tmp/g4c/l2r/results/m_res/stab.*`.

| stage | ways | runs | p50 first → last µs (min-max) | p50 drift | p99 drift | lock held min | gate | busy MHz / package W |
|---|---:|---:|---|---:|---:|---:|---|---|
| 12B L0 TP4 | 12 | 10 | 598 → 671 (585-671) | 1.049 | 1.040 | 0.990 | 10 / 10 PASS | 3,841 / 591 |
| 26B L0 expert group | 12 | 10 | 87 → 91 (87-93) | 1.027 | 1.020 | 0.996 | 10 / 10 PASS | 3,851 / 559 |
| 31B L0 TP8 | 14 | 10 | 864 → 860 (860-943) | 0.982 | 0.986 | 0.988 | 10 / 10 PASS | 3,850 / 582 |

* **No drift beyond 5%.** Counters (DRAM, L2 lines in) are flat within 1.7%, except the expert group's lines in,
  which drifted −3.5%.
* **The lock holds:** ≥ 98.8% after every run, ≥ 99.8% before every run.
* **Correctness is identical in every run.**

## P6.7 Cluster design

### Partition (plowc decides)

`plowc stage-plan` per model at 12 and 14 ways, for B = 1, 4, 8 and 16. The partition does not change with B.

| model | split | ways | stages | sockets | layout |
|---|---|---:|---:|---:|---|
| E2B | pipe (= tp plan) | 14 | 28 | 33 | 1.25 layers per socket, layers split mid-FFN; LM head 6 sockets |
| E2B | tp | 12 | 31 | 57 | L0-14 2-3 per socket; L15-34 (KV-shared, double-wide MLP) TP2 |
| E2B | pipe | 12 | 32 | 38 | same throughput as tp with 19 fewer sockets |
| E4B | tp | 12 / 14 | 44 | 96 / 95 | every layer TP2; LM head 11 / 10 |
| E4B | pipe | 12 | 67 | 77 | layers split across 2 stages; 2.4x the stages of a TP2 layer |
| 12B | tp | 14 | 49 | 206 | every layer TP4; LM head 14 |
| 12B | tp | 12 | 49 | 241 | sliding TP4; full layers TP8; LM head 17 |
| 26B-A4B | tp | 12 | 61 | 437 | per layer: 1 head socket (2 for full layers) + 13 expert sockets; LM head 12 |
| 26B-A4B | tp | 14 | 61 | 371 | 1 head socket + 11 expert sockets (11-12 experts each); LM head 11 |
| 31B | tp | 14 | 61 | 500 | every layer TP8; LM head 20 |
| 31B | tp | 12 | 61 | 583 | sliding TP8; full TP16; LM head 23 |

### Projected cluster

`cluster.py` takes each plan stage's measured time from its single-socket run, adds 2 × all-reduce per TP layer and
a hop per stage, and uses the result as follows:

* **Stage time:** step p50 of the faster GEMV that passes the gate. A stage holding several whole layers takes the
  sum of their runs.
* **Unmeasured stages:** these are row-split layers, KV-shared layers, TP degrees not run, and the LM head. They take
  the planner's prediction × the model's measured / predicted ratio at that B (0.85-1.37).
* **Token latency:** Σ stages + hops. This is the TPOT of a lone sequence.
* **TPOT at saturation:** stages × (bottleneck + hop), with B × stages sequences in flight.
* **Throughput:** B / (bottleneck + hop).

Chosen budget per model: the plan that runs the measured slices. Each cell below reads TPOT at saturation, then
tok/s per socket in parentheses. The 16K and 128K columns use the cold-KV full-attention stages (P6.5). Sliding stages
are unchanged at long context. E4B has no 16K slice run.

| model | sockets | measured stages | B=1, 2K | B=16, 2K | B=1, 16K | B=16, 16K | B=1, 128K |
|---|---:|---:|---|---|---|---|---|
| E2B (14 ways) | 33 | 1 / 28 | 3.5 ms (242) | 17.9 ms (758) | 4.7 ms (181) | 42.1 ms (322) | |
| E4B (12 ways) | 96 | 24 / 44 (flagged) | 3.8 ms (121) | 20.2 ms (363) | | | |
| 12B (14 ways) | 206 | 48 / 49 | 4.1 ms (58) | 22.8 ms (167) | 6.9 ms (34) | 66.5 ms (57) | 34.9 ms (6.8) |
| 26B-A4B (12 ways) | 437 | 55 / 61 | 5.2 ms (27) | 28.5 ms (78) | 16.4 ms (8.5) | | 75.7 ms (1.8) |
| 31B (14 ways) | 500 | 60 / 61 | 5.5 ms (22) | 34.5 ms (57) | 10.6 ms (12) | 90.9 ms (22) | 42.2 ms (2.9) |

The same at 12 ways:

* **12B:** B=16 is 20.6 ms and 158 tok/s per socket on 241 sockets. The full layers become TP8 and stop being the
  B=1 bottleneck. At B=16 they remain it (predicted, 415 µs).
* **31B:** B=16 is 32.7 ms and 51 tok/s per socket on 583 sockets.
* **26B-A4B at 14 ways:** B=16 is 36.3 ms and 72 tok/s per socket on 371 sockets. Its 11-expert groups are
  predicted, not measured.

**Sensitivity** (hop 10 µs, all-reduce 25 µs):

* B=1 TPOT rises to 3.65 ms (E2B), 5.8 ms (12B), 7.3 ms (26B) and 7.6 ms (31B).
* B=16 tok/s drops 3-7%.

**Against one socket serving the whole model** (plowrt, strict reports vs vLLM 0.30,
`gemma4-xeon6-bf16-20261006`, 1000 / 128 tokens):

| model | 1 socket c1 TPOT p99 | 1 socket c32: TPOT p99, decode tok/s / socket (32 / TPOT) | cluster B=1 TPOT | cluster B=16: TPOT, tok/s / socket |
|---|---:|---|---:|---|
| E2B | 9.3 ms | 62.9 ms, 509 | 3.5 ms (2.7x lower) | 17.9 ms, 758 (1.5x) |
| E4B | 17.6 ms | 105.8 ms, 302 | 3.8 ms (4.6x lower) | 20.2 ms, 363 (1.2x) |
| 12B | 45.1 ms | 210.1 ms, 152 | 4.1 ms (11x lower) | 22.8 ms, 167 (1.1x) |
| 26B-A4B | 16.8 ms | 145.6 ms, 220 | 5.2 ms (3.2x lower) | 28.5 ms, 78 (0.36x) |
| 31B | (no serving baseline yet) | | 5.5 ms | 34.5 ms, 57 |

* **Dense models trade sockets for latency at equal or better per-socket throughput.**
  * 12B: TPOT 45 → 4.1 ms at B=1, and 167 tok/s per socket at B=16 against 152 at c32 on one socket. The 1-socket
    c32 TPOT is 210 ms; the cluster's B=16 TPOT is 23 ms.
  * E2B gets 1.5x the per-socket throughput as well.
* **MoE does not pay.**
  * 26B-A4B keeps 390 of its 437 sockets for experts. A group is busy 10-41 µs of a 460-590 µs head step.
  * Per-socket throughput is 1/3 of one socket streaming the active experts from DRAM.
* **Long context breaks the balance.**
  * At 128K one full-attention stage takes 670-1,250 µs against ~60-80 µs for every other stage.
  * B=1 saturated TPOT goes from 4-5 ms to 35-76 ms.

### Design

| element | choice | evidence |
|---|---|---|
| unit | one socket, 90 isolated cores, a weight slice of ≤ 1.6 MiB per core in L2, unlocked | P6.4: resident without the lock; lock costs 15-53% at B=16 |
| batch | B ≤ 16 rows per microbatch, AMX | 22-37 µs per token per dense / head slice at B=16 vs 48-97 at B=1 |
| dense layer > 1 socket | TP group of the smallest S that fits (12B S=4, 31B S=8), 2 all-reduces per layer | measured slices; S chosen by `plowc --split tp` |
| dense small model | pipe for E2B (33-38 sockets) | equal throughput to tp with fewer sockets |
| MoE | head socket per layer + expert groups (13 at 12 ways) | measured; poor utilisation, see go / no-go |
| LM head | vocabulary-parallel, 10-23 sockets, 1 all-reduce | predicted only |
| ring | stages in layer order, last stage → first (next token) | `plowc` order |
| hop payload | B × hidden × 4 B: 6-21 KiB at B=1, 96-336 KiB at B=16 | |
| all-reduce payload | ring, per socket 2(S-1)/S × B × hidden × 4 B: up to 37 KiB (B=1) / 588 KiB (B=16, 31B TP8) | 588 KiB in 10 µs = 60 GB/s per socket |
| fabric | TP groups inside one server (UPI) or a ≥ 400 Gb/s, < 5 µs NIC path; pipeline hops can cross servers | the all-reduce, not the hop, sizes the fabric |
| KV | in each stage's local DRAM, P4 tiles + prefetch, no lock | P6.5 |

**What plowc still has to decide, from these measurements:**

1. **Sequence-parallel attention for full-attention layers at long context.**
   * The KV of a full layer must be split by position over k sockets, each computing a partial softmax and
     combining with one exchange.
   * TP cannot do this, because the KV heads are replicated.
   * At 128K, k = 8 brings a 12B full stage from ~690 µs to ~140 µs + combine (59 µs + 629 / 8), 2x a sliding
     stage.
   * This is a projection: P4's attention time scales with KV bytes.
2. **One KV tensor for `attention_k_eq_v` layers.**
   * V is the same projection as K, normed without scale and without RoPE.
   * Caching the raw projection once and deriving k / v while streaming halves the KV bytes of 12B / 26B / 31B
     full layers.
3. **Balance by measured stage time, not bytes.**
   * The 14-way 12B and 31B plans are bottlenecked on the full layers: 78.6 µs vs 68.3 µs for the sliding layers
     at B=1.
   * At 12 ways plowc gives those layers TP8 / TP16, which moves the B=1 bottleneck back to the sliding layers.
   * At B=16 the TP8 full layer stays the bottleneck (predicted).

## P6.8 MoE: experts resident in L2 + L3 SRAM

Every expert stays on chip; experts that no token routes to simply don't run. The 13-group design (P6.3) does this
from L2 alone and needs 13 sockets per layer for 1.52 GB of experts (11.9 MB each), 390 of 437 sockets. L3 is SRAM
too.

**Fast tier capacity.** `l2r_bw` AMX stream on the 90 workers, THP, by per-core footprint:

| per core | 3 MiB | 4.5 | 6.5 | 8 | 9 | 12 | 16 | 24 | 32 | 64 (P0, DRAM) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TB/s | 1.35 | 1.29 | 1.31 | 1.33 | 1.30 | 1.05 | 0.88 | 0.76 | 0.71 | 0.66 |

* Up to ~9 MiB per core (~800 MiB per socket), streaming holds L3 speed.
* That is more than L2 + L3 (≈ 660 MiB): L3 is non-inclusive and keeps part of a looping stream.

**Expert sockets at 2 and 3 per layer.** 26B L0, groups of 64 (760 MB per socket) and 42-43 experts (507 MB), rows
o0-o15. `ref_layer.py ... moe:2|moe:3`; `/tmp/g4c/l2r/results/m_moe23{,ctr}`.

* `L2R_EXPERTS_ROTATE=37` shifts every step's expert indices, except the first and last steps, which the gate checks.
  Each step therefore reads a different subset of the socket's experts, as a pipeline does.
* Without rotation the same experts are read every step and stay in L2. At B=1 that is 2-4x optimistic.
* Every run passes the gate.

The stage time is the slowest group (AVX, rotating):

| experts | sockets / layer | per socket | B=1 µs | B=16 µs | DRAM per step B=1 / B=16 (slowest group) | AMX B=16 µs |
|---|---:|---:|---:|---:|---|---:|
| L2 only (P6.3) | 13 | 119 MB | 18.5 | 65.3 | ~0 | 40.6 |
| L2 + L3 | 3 | 507 MB | 50.7 | 270.0 | 1.2 / 10.3 MiB (≈ 96% of L2 misses served on chip) | 387.6 |
| L2 + L3 + DRAM | 2 | 760 MB | 72.3 | 522.2 | 21.0 / 121.5 MiB (≈ 27% from DRAM) | 522.3 |

* **AVX beats AMX once the experts stream.** It is 1.3-1.4x faster at B=16, because AMX's 16-row tiles leave a
  1.7x per-core imbalance.
* **At 3 sockets per layer the experts are SRAM-resident.** At B=16 the layer's 82 touched experts (~980 MB) stream
  as ~330 MB per socket at ~1.2 TB/s, 270 µs, under the head socket's 463 µs.

**Cluster** (projection, same model as P6.7):

* Heads use the 12-way plan: 1 socket per sliding layer and TP2 for the 5 full layers (predicted).
* The LM head takes 12 sockets (predicted).

| experts | sockets | bottleneck B=16 | tok/s at B=16 | tok/s / socket | TPOT at saturation B=1 / B=16 |
|---|---:|---|---:|---:|---|
| L2 only (13 / layer) | 437 | head 463 µs | 34,200 | 78 | 5.2 / 28.5 ms |
| **L2 + L3 (3 / layer)** | **137** | head 463 µs | 34,200 | **250** | 5.2 / 28.5 ms |
| L2 + L3 + DRAM (2 / layer) | 107 | experts 522 µs | 30,300 | 284 | 5.2 / 32.2 ms |
| one socket, whole model (plowrt c32 / c1) | 1 | | | 220 | 16.8 ms (c1) / 146 ms (c32) |

**3 expert sockets per layer is the GO candidate.**

* Same throughput and TPOT as the 13-group design on 31% of the sockets.
* 1.14x one socket's throughput per socket, and 3.2x lower TPOT at B=1.

Still needed to qualify it:

1. **Routing balance.**
   * The slowest of 3 groups sets the stage, and this is measured on 16 rows of one layer.
   * Needs a routing trace over many tokens and all 30 layers: per-group pair counts and their p99.
   * Then assign experts to groups by load, not by index.
2. **The head-to-expert transfers**: per MoE layer, h1 to the 3 expert sockets and their partial sums back. That is
   B × 2816 × 4 B, 176 KiB at B=16, each way per socket. It adds latency, not throughput.
3. **Full-layer heads as TP2**: predicted only. On one socket the full-layer head is 590 µs at B=16, which would be
   the bottleneck (27k tok/s, 205 per socket on 132 sockets).
4. **plowc**: an expert tier (L2 + L3 bytes per core from the measured curve, AVX partition) instead of L2-only
   expert packing. Today it plans 13 groups.

## Go / no-go

| item | status |
|---|---|
| P1 residency for every model's slice | yes: resident unlocked ≤ 1.6 MiB per core; lock holds ≥ 0.99 but costs 15-53% at B=16 |
| P2 numerics: TP slices, MoE head, expert groups | yes for 12B / 26B / 31B; E4B TP2 FAIL on single-element boundaries only (traced to one BF16 rounding), output passes |
| P3 sync per slice | same ~20-25 µs of barrier per stage at B=1 as P3 |
| P4 long context | measured to 128K: full-attention stages DRAM-bound, need sequence-parallel KV |
| P5 batch ≤ 16 with AMX | yes for every slice; the expert group's AMX partition spills to L3 (AVX fits L2) |
| stability | 3 × 10 min, no drift > 5%, every run gated PASS |
| cluster design from measured stages | projections above. 90-98% of stages measured for 12B / 26B / 31B; E2B 1 of 28 (row-split layers) |

**Recommendation: GO for a two-server dense pipeline (12B TP4 stages over 2 × 2 sockets, then a 4-stage chain). MoE
(26B-A4B): NO-GO for the L2-only expert layout, GO candidate with experts in L2 + L3 at 3 sockets per layer (P6.8),
pending a routing-balance trace. NO-GO for long-context full-attention layers until sequence-parallel attention
exists.**

* **Dense models:**
  * 12B projects 4.1 ms TPOT at B=1 (11x below one socket) and parity or better per socket at B=16.
  * The measured parts cover 48 of 49 stages.
* **MoE:**
  * The L2-only design spends 9 of 10 sockets on experts that are idle 90% of the step.
  * Holding the experts in L2 + L3 (P6.8) keeps every expert in on-chip SRAM on 3 sockets per layer: 137 sockets,
    250 tok/s per socket at B=16.
* **Unmeasured, and the two-server experiment must measure:**
  1. the hop and the 2-per-layer all-reduce at B = 1 and 16 (payloads above);
  2. a TP group across a UPI link and across servers;
  3. the LM head as a vocabulary-parallel stage group;
  4. chained stages streaming tokens against a chained reference.

Blockers carried forward:

* The expert group's AMX partition does not fit L2: 16-row tiles over 704 rows, max / mean 1.70.
  * It runs from L2 + L3.
  * A balanced AMX split (tiles across experts, not within one) is open.
* The 31B full layer needs 14 ways (130 MiB per TP8 socket).
* E4B and E2B KV-shared layers are predicted, not measured.
* No 31B single-socket serving baseline.
