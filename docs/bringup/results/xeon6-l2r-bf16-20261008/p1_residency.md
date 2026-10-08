# P1: L2 residency, CAT pseudo-lock vs huge-page warm cache (Xeon 6975P-C)

Checklist phase P1. Can each of the 90 isolated inference cores keep a BF16 weight slice in its private
2 MiB L2 while it also streams KV and exchanges activations? Locking uses our driver
(`runtime/cpu/driver/pseudo_lock_sram`, ABI v3, `/dev/pseudo_lock`; srcversion `BC629ACA10117C72BFACA6E`
= the loaded module, source clean at `6f35570d`). Probe: `runtime/cpu/bench/l2r/l2r_lock.c`, matrix
`p1_matrix.sh`. Raw evidence: `/tmp/g4c/l2r/results/{p1,p1_ctr,p1_long}`. Host state as P0
(`isolate.sh apply`, `cpu_dma_latency=0`, THP `madvise`).

## Method

* One thread per inference core (2-31, 34-63, 66-95), weight slice per core allocated after pinning,
  2 MiB-aligned, `MADV_HUGEPAGE` (P0: 4 KiB pages lose L2 capacity to set conflicts).
* Step = one pass over the slice: AMX-BF16 B-tile stream (`TDPBF16PS`) or AVX-512 `VDPBF16PS` GEMV.
  Per-step latency per core → p50 / p99 / max; per-core GB/s; 3 reps × 10 s per cell (AVX 1 rep).
* **plain**: the slice is only warmed by use (16-way LRU). **locked**: `PL_IOC_LOCK` level 2 on the owner
  core (fill in the lock CLOS with prefetchers off); `PL_IOC_MEASURE` before and after the run counts the
  slice's lines still served at L1/L2 latency ("held").
* Interference per step:
  * A: none.
  * B: per-SNC-node slot broadcast and reduction plus a 10 KiB hidden vector (the decoder-stage exchange).
  * C: KV stream from DRAM, 256 KiB per step from a 256 MiB per-core buffer.
  * D: KV from LLC (2 MiB per core, 180 MiB total < L3) plus DRAM streamers on the 6 housekeeping cores.
  * E: C plus a 64 KiB q block, long runs.
* Lock capacity: the driver locks 7/8 of the lock ways. 12 ways (`cbm 0xfff`) = 1,376,256 B/core;
  14 ways (`0x3fff`) = 1,605,632 B/core. The rest of the L2 (4 / 2 ways) is left for CLOS 0 (all other
  data, the worker's own KV and stack, the SMT sibling).

## P1.1/P1.2 Capacity × interference (AMX unless noted; GB/s per core, mean (slowest core))

| L2 ways | slice KiB | kernel | scen | plain GB/s/core (min) | locked GB/s/core (min) | plain p99 µs | locked p99 µs | locked/plain | held L2 after | rep spread plain / locked |
|---:|---:|---|---|---|---|---:|---:|---:|---:|---|
| 12 | 512 | amx | A | 138.8 (136.0) | 138.8 (138.5) | 4.99 | 4.25 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 12 | 512 | amx | B | 128.1 (124.5) | 128.4 (124.8) | 12.30 | 12.28 | 1.002 | 0.9999 | 0.0% / 0.1% |
| 12 | 512 | amx | C | 114.2 (111.1) | 114.4 (110.0) | 6.87 | 6.88 | 1.001 | 0.9977 | 0.2% / 0.2% |
| 12 | 512 | amx | D | 109.0 (106.1) | 106.5 (98.9) | 12.38 | 11.19 | 0.978 | 0.9983 | 0.1% / 0.1% |
| 12 | 512 | avx | A | 101.4 (100.8) | 101.4 (100.8) | 5.31 | 6.08 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 12 | 512 | avx | C | 100.6 (98.5) | 100.9 (100.1) | 8.24 | 6.32 | 1.003 | 0.9999 | 0.0% / 0.0% |
| 12 | 1024 | amx | A | 139.5 (139.3) | 139.4 (139.3) | 8.56 | 8.57 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 12 | 1024 | amx | B | 134.4 (133.1) | 134.4 (133.1) | 15.78 | 15.83 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 12 | 1024 | amx | C | 118.7 (118.0) | 118.0 (91.6) | 12.31 | 12.63 | 0.994 | 0.9859 | 0.0% / 0.8% |
| 12 | 1024 | amx | D | 120.3 (100.6) | 119.4 (48.1) | 15.61 | 16.95 | 0.992 | 0.9797 | 1.2% / 2.5% |
| 12 | 1024 | avx | A | 101.5 (100.9) | 101.5 (101.2) | 11.50 | 11.98 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 12 | 1024 | avx | C | 100.5 (97.2) | 100.9 (99.6) | 18.71 | 13.62 | 1.004 | 0.9988 | 0.0% / 0.0% |
| 12 | 1280 | amx | A | 139.6 (139.5) | 139.5 (137.9) | 10.74 | 12.30 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 12 | 1280 | amx | B | 135.5 (134.4) | 135.5 (134.4) | 17.76 | 17.75 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 12 | 1280 | amx | C | 121.3 (120.7) | 116.4 (58.6) | 14.24 | 19.69 | 0.959 | 0.9457 | 0.0% / 2.5% |
| 12 | 1280 | amx | D | 124.0 (122.1) | 122.6 (85.4) | 17.98 | 19.17 | 0.988 | 0.9823 | 0.1% / 2.8% |
| 12 | 1280 | avx | A | 101.5 (100.8) | 101.5 (100.9) | 16.55 | 15.06 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 12 | 1280 | avx | C | 100.8 (98.7) | 99.1 (77.8) | 19.18 | 19.37 | 0.983 | 0.9885 | 0.0% / 0.0% |
| 12 | 1344 | amx | A | 139.6 (139.4) | 139.6 (139.4) | 11.30 | 11.30 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 12 | 1344 | amx | B | 135.7 (134.6) | 135.8 (134.7) | 18.25 | 18.19 | 1.001 | 1.0000 | 0.0% / 0.2% |
| 12 | 1344 | amx | C | 121.9 (121.3) | 116.1 (74.6) | 14.60 | 18.93 | 0.953 | 0.9457 | 0.0% / 2.1% |
| 12 | 1344 | amx | D | 124.5 (123.2) | 123.4 (103.0) | 18.46 | 18.69 | 0.991 | 0.9916 | 0.1% / 0.7% |
| 12 | 1344 | avx | A | 101.5 (101.3) | 101.5 (101.0) | 15.68 | 15.82 | 1.000 | 1.0000 | 0.0% / 0.0% |
| 12 | 1344 | avx | C | 100.8 (100.2) | 95.8 (66.4) | 16.47 | 22.44 | 0.950 | 0.9688 | 0.0% / 0.0% |
| 14 | 512 | amx | A | 138.9 (138.8) | 138.8 (138.7) | 4.24 | 4.24 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 14 | 512 | amx | B | 128.2 (124.7) | 128.3 (124.8) | 12.34 | 12.32 | 1.001 | 0.9999 | 0.0% / 0.1% |
| 14 | 512 | amx | C | 114.3 (110.5) | 114.5 (112.1) | 6.96 | 6.93 | 1.002 | 0.9999 | 0.1% / 0.1% |
| 14 | 512 | amx | D | 108.8 (102.9) | 106.7 (96.4) | 12.41 | 11.26 | 0.980 | 0.9997 | 0.1% / 0.6% |
| 14 | 512 | avx | A | 101.3 (100.8) | 101.3 (100.8) | 5.24 | 5.75 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 14 | 512 | avx | C | 100.5 (97.2) | 100.3 (99.7) | 9.17 | 6.29 | 0.998 | 0.9999 | 0.0% / 0.0% |
| 14 | 1024 | amx | A | 139.5 (139.4) | 139.4 (139.3) | 8.47 | 8.58 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 14 | 1024 | amx | B | 134.4 (133.1) | 134.4 (133.1) | 15.77 | 15.80 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 14 | 1024 | amx | C | 118.6 (114.4) | 118.3 (97.4) | 12.33 | 12.35 | 0.998 | 0.9962 | 0.4% / 0.9% |
| 14 | 1024 | amx | D | 120.8 (119.2) | 118.4 (58.9) | 15.60 | 16.81 | 0.980 | 0.9922 | 0.1% / 2.7% |
| 14 | 1024 | avx | A | 101.5 (100.9) | 101.5 (100.9) | 12.01 | 10.48 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 14 | 1024 | avx | C | 100.8 (100.0) | 97.8 (61.3) | 12.59 | 19.03 | 0.970 | 0.9946 | 0.0% / 0.0% |
| 14 | 1280 | amx | A | 139.6 (139.4) | 139.6 (139.4) | 10.73 | 10.73 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 14 | 1280 | amx | B | 135.5 (134.5) | 135.5 (134.4) | 17.66 | 17.74 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 14 | 1280 | amx | C | 121.4 (120.5) | 120.8 (108.9) | 14.13 | 15.18 | 0.996 | 0.9987 | 0.1% / 0.2% |
| 14 | 1280 | amx | D | 124.0 (122.6) | 119.8 (71.1) | 18.03 | 20.78 | 0.966 | 0.9822 | 0.1% / 5.9% |
| 14 | 1280 | avx | A | 101.5 (100.9) | 101.4 (100.9) | 14.95 | 15.02 | 1.000 | 0.9999 | 0.0% / 0.0% |
| 14 | 1280 | avx | C | 100.8 (100.0) | 97.4 (72.4) | 15.93 | 19.73 | 0.966 | 0.9890 | 0.0% / 0.0% |
| 14 | 1568 | amx | A | 139.6 (139.5) | 139.6 (139.5) | 13.19 | 13.19 | 1.000 | 1.0000 | 0.0% / 0.0% |
| 14 | 1568 | amx | B | 136.3 (135.3) | 136.3 (135.4) | 19.87 | 19.87 | 1.000 | 0.9995 | 0.0% / 0.0% |
| 14 | 1568 | amx | C | 123.8 (123.3) | 119.7 (69.0) | 16.33 | 21.24 | 0.967 | 0.9687 | 0.0% / 3.7% |
| 14 | 1568 | amx | D | 125.8 (123.9) | 119.9 (59.3) | 20.12 | 26.09 | 0.953 | 0.9771 | 0.1% / 4.3% |
| 14 | 1568 | avx | A | 101.5 (101.2) | 98.8 (78.4) | 18.57 | 30.27 | 0.974 | 0.9897 | 0.0% / 0.0% |
| 14 | 1568 | avx | C | 100.3 (99.8) | 94.2 (58.1) | 19.31 | 30.49 | 0.940 | 0.9755 | 0.0% / 0.0% |

Readings:

* **A and B: locked = plain** within 0.2% at every size up to 1.53 MiB per core, except AVX-512 at 1.53 MiB
  (14 ways, A), where locked is 2.6% lower (slowest core 78.4). Nothing evicts a THP-backed slice of that
  size that is re-read every ~10 µs. B costs 3-8% of the stream rate (exchange traffic, P3).
* **C and D: plain is better than locked.** Plain keeps 109-126 GB/s per core with the slowest core within
  5.4% of the mean (one exception: 12 ways, 1 MiB, D, slowest 100.6 vs 120.3, rep spread 1.2%). Locked loses
  up to 6% on the mean, and its slowest core drops to 48-103 GB/s. Its p99 step
  time is up to 5.5 µs higher (12 ways, 1.25 MiB, C: 14.2 → 19.7 µs).
* Locked lines do not stay held under DRAM interference: 0.946-0.99 of the slice is still in L2 after a C
  or D run (0.9999 after A/B).

## P1.3 Counters (1.25 MiB slice, 12 ways, 10 s)

Raw core events (perf 6.1 has no event JSON for this CPU): `L2_LINES_IN.ALL` (`r1f25`),
`L2_LINES_OUT.SILENT` / `.NON_SILENT` (`r0126` / `r0226`); CHA snoop-filter evictions
(`uncore_cha/event=0x3d,umask=0x07/`, all CHAs). A fully resident slice needs only the KV step's lines:
4,096 per step in C/D, 0 in A.

| scen | lock | steps/core | L2 lines in / step / core | weight lines | KV lines/step | silent / non-silent out per step | CHA SF evictions / s | GB/s/core (min) | held after |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|
| A | 0 | 1,062,143 | 1 | 20,480 | 0 | 0 / 1 | 1,426 | 139.6 (138.6) | n/a |
| A | 1 | 1,062,142 | 2 | 20,480 | 0 | 0 / 1 | 524 | 139.6 (138.3) | 0.9999 |
| C | 0 | 219,177 | 4,121 | 20,480 | 4096 | 24 / 4,100 | 1,335,868 | 121.3 (119.2) | n/a |
| C | 1 | 218,139 | 4,250 | 20,480 | 4096 | 19 / 4,232 | 38,666 | 118.1 (96.9) | 0.9723 |
| D | 0 | 422,370 | 4,100 | 20,480 | 4096 | 3 / 4,098 | 3,876,402 | 124.0 (120.9) | n/a |
| D | 1 | 422,600 | 4,104 | 20,480 | 4096 | 2 / 4,103 | 35,161 | 123.3 (97.8) | 0.9949 |

* **Plain: the weights are resident.** Under the DRAM KV stream 25 of the slice's 20,480 lines per step
  (0.12%) are refetched; under LLC KV 4 lines.
* **Locked: 154 weight lines per step (0.75%) are refetched under C.** Over the 10 s run, 2.8% of the
  locked lines (~51 K lines over 90 cores) leave L2 although CLOS 0 cannot allocate into the lock ways. The
  eviction path is not identified. The candidate is snoop-filter back-invalidation: 38.7 K CHA SF
  evictions/s in locked C, enough to account for the loss, but the counter does not say whose lines are
  evicted. A lost line refills through CLOS 0. It then competes for the 4 unlocked ways with the 4,096-line
  KV step and is evicted again every step, while plain mode gives every line all 16 ways under LRU. The
  lock turns a one-off loss into a per-step miss.

### Weight-phase slowdown after a KV phase (plain, 1.25 MiB, AMX, C, 10 s)

The probe's GB/s times only the weight pass. In C/D the AMX weight pass runs 11-13% slower than in A
although the counters show the slice resident. The AVX-512 GEMV loses ≤ 1% in the same cells.

| KV step per core | 4 KiB | 16 KiB | 64 KiB | 256 KiB (default) | 1 MiB |
|---|---:|---:|---:|---:|---:|
| AMX weight GB/s per core (min) | 139.0 (137.7) | 136.0 (134.9) | 126.8 (125.4) | 121.5 (119.8) | 95.9 (89.1) |
| vs A (139.6) | -0.4% | -2.6% | -9.2% | -13.0% | -31% (slice + KV > L2) |

Not drain: a spin of 1 / 3 / 10 µs after each KV phase (`L2R_GAP_NS`) leaves C at 121.7 / 122.8 / 118.9 and
D at 123.9 / 122.6 / 120.8 GB/s. Not frequency: effective clock over the run is 2.95 (A) / 3.02 (C) /
2.94 (D) GHz. The loss scales with the KV bytes the core touched since its last weight pass. The
mechanism is open (candidates: L2 replacement-state or prefetcher disruption specific to the AMX tile-load
stream).

## P1.4 Long runs (scenario E, 1.25 MiB, AMX)

| run | secs | GB/s/core mean (min core) | worst p99 µs | max step µs (worst core) | step µs first 10% → last 10% (median core) | worst core drift | held L2 before → after |
|---|---:|---|---:|---:|---|---:|---|
| plain | 600 | 121.33 (120.84, cpu 42) | 14.17 | 427 | 10.81 → 10.80 | -0.5% | n/a |
| plain | 1800 | 121.32 (120.80, cpu 43) | 14.12 | 257 | 10.81 → 10.80 | -0.3% | n/a |
| locked | 600 | 111.22 (65.76, cpu 90) | 22.99 | 292 | 10.83 → 11.11 | +33.9% | 1.0000 → 0.9286 |

E runs the full C KV stream plus a 64 KiB q block per step, a synthetic attention (no softmax). The
single slowest step per run (257-427 µs) is a one-off; its cause is not attributed, and p99 is unaffected.

## Exit gate P1

| criterion | result |
|---|---|
| no sustained weight reload | **PASS, plain THP**: 25 of 20,480 weight lines per step refetched under the DRAM KV stream, 4 under LLC KV; step time flat over 30 min (median core 10.81 → 10.80 µs, worst core -0.3%). **Fails with the lock**: 7.1% of locked lines lost in 10 min, worst core +33.9% |
| < 5% slowdown under the intended KV workload | **AVX-512 GEMV: PASS** (≤ 3% in C/D at every size; 1.53 MiB locked excepted). **AMX stream: PASS only up to 16 KiB of KV per core per step** (0.4% at 4 KiB, 2.6% at 16 KiB). It is 9% at 64 KiB and 13% at 256 KiB, with the weights still resident; the cause is not reload and not post-phase drain, mechanism unidentified |
| 12 vs 14 ways | moot: the lock is not adopted. Plain THP keeps up to 1.53 MiB per core resident (C: 123.8 GB/s, slowest core 123.3) |
| cleanup after interruption | SIGKILL mid-run: 0 regions after the fd closes, CLOS 0 L2 mask back to `0xffff`, PQR CLOS 0, `MSR 0x1A4` = 0 on sampled cores |

**P1 exit: conditional PASS.** Residency is robust without the lock: 1.25-1.5 MiB per core, THP-backed,
warm by use. Carried forward:

* P2 uses plain THP slices of ≤ 1.25 MiB per core (1.53 MiB measured resident; 1.25 MiB leaves 768 KiB of
  L2 for KV, activations and the SMT sibling) and `PLOW_CPU_HUGE_PAGES=1` in plowrt.
* The AMX-after-KV slowdown is a P4 item: keep the KV bytes per core per step small (tile attention so a
  core's KV chunk is ≤ 16-64 KiB between weight passes), or use the AVX-512 GEMV, which is insensitive but
  27% slower. E4B batch-1 KV is ~2 KiB per token per layer. Spread over 90 cores that is ~45 KiB per core
  per layer at 2K context and ~360 KiB at 16K for global-attention layers (sliding-window layers stay small).
* Driver lock: not used. Lost locked lines refill into the 4 unlocked ways and keep missing. The eviction
  path (snoop-filter back-invalidation is the candidate) is open; it would matter only if the lock were
  needed to protect weights from co-runners.
* Not tested: a CAT-only mode (weights filled into a reserved way mask without the driver's
  prefetch-off fill). The same "lost lines refill into the restricted mask" behaviour is expected but not
  measured.
