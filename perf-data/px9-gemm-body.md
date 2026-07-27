# PX-9 — the w8a8 GEMM body is not the mainloop, it is the cp.async staging path

RTX 5090 (sm_120a, **170 SMs**, 96 MiB L2) · 2026-07-26 · bench `perf-data/px9_gemm_body_bench.cu`
Run under `perf-data/harness/gpulease`. Companion to PX-7; `px7_w8a8_ceiling_bench.cu` and
`px7-w8a8-ceiling.md` are campaign records and are **left untouched** — where PX-9 corrects a
PX-7 number it says so here rather than editing the older file.

## Question

PX-7 measured the w8a8 prefill GEMM at ~60% of the in-tree fp8 peak of 503.8 TFLOP/s with wave
quantization removed and nothing else running, and killed the occupancy branch. It left the
body itself unattributed. Candidates it named: smem bank conflicts on B, pipeline depth,
register spills, epilogue cost, the `QUANT_FP8` activation pass, or operand movement rather
than mma throughput. This file attributes it.

**The instrument is SM clock cycles per QMMA, not TFLOP/s.** TFLOP/s hides a clock that moves
with power draw; cycles/QMMA does not. A register-only QMMA loop gives the hardware issue
ceiling in the same unit, so every rung is a pure ratio and no "peak" number has to be trusted.

## Result 1 — the peak is real, and it is the fp32-accumulate wall

`fp8_mma_rate_probe.cu` (register-only, 4 independent chains/warp, 170 blocks), re-run today:

| instruction | TFLOP/s | FLOP/clk/SM |
|---|---|---|
| `mma.m16n8k16.f32.bf16` | 259.2 | 512 |
| `mma.m16n8k32.f32.e4m3` | **516.9** | **1024** |
| `mma.m16n8k32.f16.e4m3` | **1027.9** | **2048** |

The latency-bound arm (1 dependent chain) gives 256.9 / 515.0 / 1031.2 — identical, so the mma
is **throughput**-limited, not latency-limited, and one dependent chain already saturates it.

PX-9's own ladder rung 0 measures the same thing in cycles: **exactly 64.00 SM-cycles per
warp-QMMA**, i.e. `8192 FLOP / (64.00/8 warps) = 1024 FLOP/clk/SM`, at a clock64-derived
**2.979 GHz** → **518.5 TFLOP/s**. The in-tree 503.8 is 97% of that; the 3% is clock, not
kernel. **503.8 is a good number and PX-7's denominator was correct.**

Two things follow:

* **fp32 accumulate costs exactly 2x on this part.** e4m3 with an f16 accumulator runs at
  2048 FLOP/clk/SM. This is the documented consumer-Blackwell segmentation (the GeForce
  whitepaper's 838 vs 419 dense fp8 TFLOP/s at the 2407 MHz spec clock is the same 2:1). It is
  a **numerics** lever, not a tuning one — see Result 7.
* **The in-tree bf16 peak of 209.5 TFLOP/s is understated.** It is 512 FLOP/clk/SM at the
  2407 MHz spec clock; the part actually runs the mma loop at ~2.98 GHz, giving 259.2. Any
  "% of bf16 peak" quoted against 209.5 in PX-6/PX-7 is ~24% too generous.

## Result 2 — the mainloop is at 94.3% of the ceiling. **The smem path is not the problem.**

The ladder runs the REAL block shape — 1 block/SM, 8 warps, `acc[2][8][4]` = 64 f32/thread, the
real 2x8 mma block, the real swizzled fp8 smem tiles — and adds one thing per rung. No global
traffic anywhere.

| rung | cyc/QMMA | vs ceiling | implied TFLOP/s |
|---|---|---|---|
| 0 mma only, operands resident in registers | **64.00** | — | 518.5 |
| 1 + fragment reads from smem, AS SHIPPED (scalar u32) | 64.38 | **+0.6%** | 515.4 |
| 2 + fragment reads, vectorized (uint2) | 64.38 | +0.6% | 515.4 |
| 3 = rung 1 + `__syncthreads` per K-tile | 67.88 | +6.1% | 488.8 |
| 4 = rung 2 + `__syncthreads` per K-tile | 67.93 | +6.1% | 488.4 |
| 5 = rung 2 with the swizzle REMOVED | 66.00 | +3.1% | 502.9 |

Run twice under separate leases; the two runs agree to **0.01 cyc/QMMA on every rung**, which is
what makes the 0.6% and 6.1% figures usable at all (the TFLOP/s column moves ~0.3% between runs
purely on clock, 2.979 vs 2.969 GHz — the cycle column does not move).

**The entire smem fragment read — 48 `LDS.32` per K-tile per warp, a structural 2-way bank
conflict, and 96 `LOP3` + 26 `SHF` + 23 `IADD3` of swizzle address arithmetic per 32 QMMA
(counted in SASS) — costs 0.6%.** At 64 cycles per warp-QMMA the mma cadence is so slow that
everything else hides behind it. The barrier costs **5.4%** (rung 3 vs rung 1). The swizzle is
worth keeping: removing it costs **2.5%** (rung 5 vs rung 2), i.e. it buys back more than the
address arithmetic it charges.

**So a plow K-tile with no global traffic runs at 67.89 cyc/QMMA = 94.3% of the hardware
ceiling** (478 TFLOP/s at the 2.913 GHz that rung sustains; the ceiling at that same clock is
507). Note the percentages in this table are CYCLE ratios and are therefore clock-free — the
TFLOP/s column is not comparable across rows because each rung sustains a slightly different
clock, which is exactly why the ladder is denominated in cycles.

Every hypothesis PX-7 listed about the mainloop — bank conflicts, register spills (SASS: 0
spill bytes in both w8a8 bodies), epilogue cost (~0.6% of a K=3840 tile by instruction count),
ldmatrix/movement overhead — is worth single-digit percent or less.

## Result 3 — PX-7's per-shape "% of peak" was understated by IDLE SMs

The oracle grid `G* = largest divisor of T <= P` removes wave quantization by construction, but
it does not use the whole machine. Measuring the same shapes at `grid = P = 170` as well:

| shape | M | T | G\* | SMs used | G\* TFLOP/s | full-grid TFLOP/s | ratio |
|---|---|---|---|---|---|---|---|
| gate\|up | 8192 | 7680 | 160 | 94.1% | 305.4 | 316.9 | 1.038x |
| down | 8192 | 1920 | 160 | 94.1% | 342.7 | 342.7 | 0.998x |
| **q_full** | 8192 | 4096 | **128** | **75.3%** | **257.8** | **328.5** | **1.274x** |
| o_full | 8192 | 1920 | 160 | 94.1% | 331.3 | 329.9 | 0.996x |

**`q_full` was never a slow shape.** 4096's largest divisor below 170 is 128, so PX-7's oracle
grid left **42 of 170 SMs idle** and read 243.3 TFLOP/s (48.3%) for a shape that does 328.5
(63.4%) on the full machine — the same as the other three. PX-7's table should be read as
94.1%-of-machine numbers for three shapes and a 75.3%-of-machine number for `q_full`.

Corrected, all four shapes sit in a tight band: **61–66% of the 503.8 in-tree peak, i.e. 66–72%
of the no-global-traffic mainloop ceiling.** There is one wall, not four.

## Result 4 — cuBLASLt reaches 95–99% on the same shapes. **The gap is ours.**

cuBLASLt fp8 (e4m3 TN, per-tensor scale, bf16 out — a speed reference, not a plow twin, since
it has no per-row activation scale and picks its own grid). **Both engines under the SAME
L2-cold protocol** (16 weight replicas, cycled per iteration); see the cold/warm control below.

| shape | M | cuBLASLt (cold) | % of 503.8 | plow full-grid (cold) | plow/cuBLASLt |
|---|---|---|---|---|---|
| gate\|up | 8192 | **499.8** | 99.2% | 317.3 | 0.63x |
| down | 8192 | **481.2** | 95.5% | 341.5 | 0.71x |
| q_full | 8192 | **497.5** | 98.7% | 327.0 | 0.66x |
| o_full | 8192 | **479.0** | 95.1% | 328.5 | 0.69x |
| gate\|up | 2048 | 476.0 | 94.5% | 301.8 | 0.63x |
| down | 2048 | 477.7 | 94.8% | 338.4 | 0.71x |
| q_full | 2048 | 431.3 | 85.6% | 284.4 | 0.66x |
| o_full | 2048 | 473.7 | 94.0% | 323.9 | 0.68x |

### The cold/warm control — DRAM is not in this story at all

Run 1's cuBLASLt arm was L2-warm (one weight, 30 iterations) while plow's cycled 16 replicas,
which would have been a real confound. Re-running **both** engines at both protocols
(`PX9_NREP=16` cold vs `PX9_NREP=1` warm), full grid, M=8192:

| engine | gate\|up | down | q_full | o_full |
|---|---|---|---|---|
| plow, L2-cold (16 replicas) | 317.3 | 341.5 | 327.0 | 328.5 |
| plow, L2-warm (1 weight) | 317.9 | 341.7 | 328.5 | 328.7 |
| cuBLASLt, L2-cold | 499.8 | 481.2 | 497.5 | 479.0 |
| cuBLASLt, L2-warm | 499.9 | 481.1 | 496.3 | 477.9 |

**Every cell moves by less than 0.5% — i.e. by nothing.** The confound was not a confound, and
more importantly: at these shapes *neither engine is touching DRAM in a way that matters*. The
700 MB replication protocol PX-6 and PX-7 built is doing no work here. Whatever separates 328
from 497 is not memory-system cold-start.

cuBLASLt lands **exactly where PX-9's rung 3 says a K-tile with no global traffic belongs**
(479–500 straddling the 476.4/478.1 mainloop ceiling). So the ceiling is reachable on this silicon, PX-7's
"~90% of peak" target is not fantasy, and the 30% plow is missing is **not** in the mma, the
fragment read, the swizzle, the barrier, the epilogue, or the accumulator.

**By elimination it is the `cp.async` global→smem staging path** — the only thing in
`d_gemm_w8a8` that the ladder does not have. Per rtx-01 §1 and §6, that is precisely what
cuBLAS does differently on sm_120: it drives the operand stream with **TMA
(`cp.async.bulk.tensor`)**, which removes the per-thread address generation and all of the smem
**store** traffic that `LDGSTS` still pays ("0 SMEM stores vs 256" in the cuBLAS-beating sm_120
kernel this repo already cites).

The corroborating in-kernel evidence is Result 6: halving the number of `LDS` instructions in
the fragment read is worth **0%** in the ladder and **+6.5%** in the real kernel. The only
difference between the two is that the real kernel's LSU is simultaneously running `LDGSTS`.

### It is the request path, not the byte count

At `BM=BN=128` the arithmetic intensity against global is `2*BM*BN/(BM+BN)` = 128 FLOP per
requested byte, so the gmem-request rate the shapes actually pull is:

| shape | gmem requests | ms | TB/s now | TB/s needed at the mainloop ceiling | working set |
|---|---|---|---|---|---|
| gate\|up | 11.58 GB | 6.099 | 1.90 | 2.86 | 149 MB |
| down | 7.61 GB | 2.820 | 2.70 | 3.77 | 185 MB |
| q_full | 4.16 GB | 1.569 | 2.65 | 3.86 | 63 MB |
| o_full | 4.09 GB | 1.562 | 2.62 | 3.79 | 99 MB |

Two eliminations fall out of this table:

* **Not DRAM.** `q_full`'s entire working set (63 MB) fits the 96 MiB L2 while `down`'s
  (185 MB) does not, and they land within 1.5% of each other. L2 residency does not predict
  performance at all — and the cold/warm control above closes it: turning the L2-cold protocol
  off moves nothing.
* **Not the byte count / L2 bandwidth.** `PGM_BN=256` raises the intensity 1.33x (128 → 170.7
  FLOP/requested byte) with **zero register spills** in the plain body, and it measured
  **0.92x on `q_full`, 0.96x on `down`** — *slower*. If bytes through L2 were the wall, buying
  a third of them back would have helped.

What is left is the **cost of the requests themselves**. Each `stage()` issues 4 `LDGSTS` per
thread = **1024 outstanding `cp.async` requests per block per K-tile**, each with its own
bounds check, its own `pgm_sw8` store address, and its own smem write. TMA replaces all 1024
with one descriptor-driven bulk copy and zero smem stores. That also explains why more stages
do nothing: the limit is the request queue and the issue path, not the number of buffers.

This is the weakest link in the chain and it is stated as such: **without `ncu` it is an
elimination argument, not a counter reading.** The direct confirmation would be
`l1tex__throughput` / `lsu` stall attribution, which this container cannot collect.

## Result 5 — the `-D` knob space is exhausted. Nothing in it is worth more than 1%.

Full-grid TFLOP/s, L2-cold, ratio against the PX-9 default:

| arm | gate\|up | down | q_full | o_full | verdict |
|---|---|---|---|---|---|
| **PX-9 default** | 316.9 | 342.7 | 328.5 | 329.9 | — |
| `PGM_STAGES=4` | 1.002x | 1.006x | 1.004x | 1.006x | +0.4%, costs occ 2→1 |
| `PGM_STAGES=5` | 1.004x | 0.999x | 0.999x | 1.002x | noise |
| `PGM_STAGES=6` | 1.003x | 1.000x | 0.999x | 0.999x | noise |
| `PGM_GLU_STAGES=3` | 1.009x | — | — | — | +0.9% |
| `PGM_GLU_STAGES=4` | 1.012x | — | — | — | +1.2%, breaks the arena cap |
| `PGM_BN=64` | **1.096x** | 0.896x | 0.890x | 0.906x | shape-dependent, see below |
| `PGM_BN=256` | 0.611x | 0.957x | 0.920x | 0.958x | GLU spills 364 B, dead |
| `PGM_SW8_OFF=1` | 1.025x | 0.870x | 0.877x | 0.876x | swizzle earns its keep on plain |

**Pipeline depth is worth nothing.** This is a clean negative and it kills the "cuBLAS uses 6
stages so we should too" hypothesis: at `STAGES=6` (98,304 B of arena) the kernel is within
noise of `STAGES=3`. The staging path is not latency-bound; more buffers do not help it.

**`PGM_BN=64` is worth +9.6% on `gate|up` and −10% on everything else.** `gate|up` is ~2/3 of
prefill GEMM FLOPs and is the GLU arm, which PX-7 showed is register-limited: at `BN=64`,
`NFRAG` drops 8→4, so the GLU's two accumulators cost 64 f32/thread instead of 128 — the same
as the plain body. A per-op tile choice (`BN=64` for `GEMM_GLU`, `BN=128` for the plain
projections) is worth `0.67*1.096 + 0.33*1.00` = **+6.4% weighted**, and it is a selector
change, not a kernel change. Not taken here because it belongs to the tuning system
(`plans/arch-gemm-tuning-system.md`), but it is the largest single win in the knob space.

## Result 6 — two bit-exact body changes, +9% on the plain arm, 0% on GLU

Both are pure address bijections over the same bytes, so the mma sees identical operands in
identical lanes and the accumulation is bit-identical. Both are in `op_gemm.cuh` behind `-D`
knobs and default ON.

**(a) `PGM_W8A8_LDS64` — read the fp8 fragment as a `uint2`.** The fp8 fragment map gives lane
L byte offset `8*(L&3)` inside its row, so a 4-byte read touches **only even 4-byte words — at
most 16 of the 32 banks.** Enumerated over the whole (row, lane) space, that is a structural
2-way conflict that **no XOR swizzle can remove**. The pair `(kb, kb+4)` is always 8-byte
aligned and never crosses a 16-byte line, so `sw8(off+4) == sw8(off)+4` holds exactly and one
`LDS.64` replaces two `LDS.32` plus a second full swizzle address chain.

**(b) `PGM_SW8_V2` — match the swizzle to the actual 64-byte row.** The shipped `Swizzle<3,4,3>`
assumes a 128-byte fast dimension, but a `BK8=64` fp8 row is 64 bytes, so `SShift=3` reads
offset bits [7,10) = **row bits [1,4) and never sees row bit 0** — two adjacent rows get the
same line permutation. `off ^ ((off>>2)&0x30)` is the `Swizzle<2,4,2>` matched to the real row.

| read granularity | `Swizzle<3,4,3>` (shipped) | V2 |
|---|---|---|
| 4-byte (`LDS.32`) | 2-way | 2-way (structural) |
| 8-byte (`LDS.64`) | 2-way | **1-way, conflict-free** |

Measured, full grid, L2-cold, TFLOP/s and ratio against the **shipped** body
(`PGM_W8A8_LDS64=0 PGM_SW8_V2=0`). All four arms pass the same numeric oracle.

| arm | gate\|up/8192 | down/8192 | q_full/8192 | o_full/8192 |
|---|---|---|---|---|
| shipped: scalar read + `Swizzle<3,4,3>` | 317.0 (1.000x) | 320.2 (1.000x) | 308.3 (1.000x) | 309.6 (1.000x) |
| **(a) alone**: uint2 read + old swizzle | 315.3 (0.995x) | 340.9 (**1.065x**) | 325.5 (**1.056x**) | 328.3 (**1.060x**) |
| **(b) alone**: scalar read + V2 swizzle | 312.8 (0.987x) | 312.2 (0.975x) | 301.4 (0.978x) | 301.9 (0.975x) |
| **(a)+(b) = PX-9 default** | 316.9 (1.000x) | 342.7 (**1.070x**) | 328.5 (**1.066x**) | 329.9 (**1.066x**) |

Read the table carefully, because it is not additive:

* **(a), the uint2 read, carries the whole win: +5.6 to +6.5% by itself.**
* **(b), the V2 swizzle, is actively harmful on its own (−2.5%)** — with 4-byte reads both
  swizzles are 2-way conflicted, so V2 only changes which conflict you pay and loses the
  Swizzle<3,4,3> spread. It is worth **+0.5%** on top of (a), which is at the edge of the
  ~0.5% run-to-run spread seen between repeated identical arms. Kept because the pair measures
  best and the analysis says the pair is the conflict-free one — but the honest claim is "(a)
  is the fix, (b) is free and consistent with it", not "(b) is worth 2.5%".
* **The GLU arm does not move at all** (0.995–1.000x, i.e. noise). It is register-bound, as
  PX-7 found, so freeing LSU slots buys it nothing.

Reproduced in a second lease under the cold/warm control (full grid, L2-cold, PX-9 default vs
shipped): `gate|up` 1.001x / 1.002x, `down` 1.070x / 1.069x, `q_full` 1.061x / 1.066x, `o_full`
1.060x / 1.067x at M=8192 / M=2048. Weighted over the real op mix (`gate|up` ~2/3 of prefill
GEMM FLOPs): **+2.2%**. Small, free, bit-exact.

The interesting part is the disagreement with the ladder, which said the fragment read costs
0.6%. Both are true: with no `LDGSTS` in flight the fragment read hides completely behind the
64-cycle mma cadence; with `LDGSTS` in flight the LSU is a real co-limiter and the same
instructions cost 6.5%. **That is itself evidence that the staging path is what saturates.**

## Result 7 — what would actually close the remaining 30%

Ranked by measured evidence, not by plausibility:

1. **TMA (`cp.async.bulk.tensor`) + mbarrier for the operand stream.** This is the one change
   that targets the thing PX-9 attributes the gap to, and it is the one thing cuBLASLt does
   differently on this part. rtx-01 §1 confirms single-CTA TMA exists on sm_120a and the repo
   already has the ABI groundwork (`experiments/tma_abi_probe.cu` — runtime-indexed descriptor
   tables, two tile shapes over one tensor, handle-derived ids). What does **not** exist is an
   sm_120a TMA GEMM: every `tma_ws_gemm_*.cu` in the tree targets sm_90a and uses `wgmma`,
   which sm_120 does not have. The port is TMA producer + `mma.sync` consumer.
   Ceiling: the ~478 TFLOP/s mainloop number, i.e. **+40% on the GEMM**, 9.6 s → ~6.5 s of a
   127k prefill. This matches PX-7's sizing of the prize.
2. **Per-op `BN`** (Result 5): +6.4% weighted, selector-level, no kernel work.
3. **The PX-9 body changes** (Result 6): +2.2% weighted, already in.
4. **f16 accumulate** is a hard 2x on the tensor core (Result 1) and is the ONLY way past
   ~1024 FLOP/clk/SM on this silicon. It is not a tuning decision. e4m3 inputs have ~2 decimal
   digits; an f16 accumulator over K=3840 (60 K-tiles of k32) would need DeepGEMM-style
   two-level promotion — accumulate f16 inside a K-block, promote to f32 every 64–128 K —
   which the in-tree `tma_ws_gemm_fp8.cu` already measured on Hopper (promote-128 cut the error
   18x there). **Not recommended before (1)**: it doubles a ceiling that plow is currently 30%
   short of reaching, so it buys nothing until the staging path is fixed.

**What is definitively dead:** stage-count tuning, swizzle tuning beyond Result 6, register
spilling (there is none), epilogue cost, and — from PX-7 — occupancy. The mainloop is at 94.3%.

## Gates

| gate | result |
|---|---|
| numeric oracle — `fp8_gemm_w8a8_probe.cu` on the PX-9 default | **PASS** — 0/16384 exact mismatches on `d_gemm_w8a8`; GLU relL2 1.635e-3, band-fails 0 |
| numeric oracle — same probe on the shipped body (control) | **PASS** — byte-identical result, so the two body changes are proven bijections |
| oracle grid u = 1.000 exactly | **PASS** — asserted per cell, as PX-7 |
| full-grid arm (the PX-7 SM-idling correction) | **PASS** — measured, not inferred |
| L2-cold | **ENFORCED** — 700 MB replication + cycling on the plow arm |
| L2-cold on the cuBLASLt arm | **FIXED MID-RUN, then PASS** — run 1's cuBLASLt arm reused one weight across all 30 iterations while plow's cycled 16 replicas. Re-run under the identical replication protocol: cuBLASLt cold 499.8/481.2/497.5/479.0 vs warm 499.9/481.1/496.3/477.9. The confound was real in principle and worth nothing in fact. |
| cold/warm control on BOTH engines | **PASS** — every cell within 0.5%, so no conclusion here rests on the cold protocol |
| ladder reproducibility | **PASS** — two separate leases agree to 0.01 cyc/QMMA on all six rungs |
| megakernel register bucket unchanged | **PASS** — the real `interp_sm120.cu` prefill object (`PLOW_NV_GEMMA=1 PLOW_NV_FA_GF=2 PLOW_NV_EMBED_SMEM=1 PLOW_NV_PREFILL=1`) builds to **238 registers, 0 spill bytes, 1024 B stack, 2320 B smem** both with and without the PX-9 changes. `op_moe.cuh` shares `pgm_load_*_w8a8` and `pgm_sw8` and is in the same TU, so it inherits both changes at no register cost. This is the `arch-gemm-tuning-system.md` §2.2(7) "megakernel resources are global" hazard, explicitly checked. |
| GPU exclusive | **ENFORCED** — `gpulease` |
| `ncu` attribution | **NOT RUN** — `ERR_NVGPUCTRPERM`: this container has no GPU performance-counter permission. Every stall/bank-conflict claim in this file is therefore from **SASS instruction census + exhaustive offline bank enumeration + differential timing**, never from a hardware counter. The "it is the staging path" conclusion is by **elimination**, not by a measured `dram__`/`l1tex__` counter. |
| bf16 null control | **NOT RUN** — PX-7 already passed it (117.4 vs PX-6's 116.5) and PX-9 changes no bf16 path |
| end-to-end prefill | **NOT RUN** — no checkpoint in this worktree; all numbers are isolated-kernel |

### Bugs found mid-run, recorded

1. **The hoisted-swizzle rung was wrong and crashed the ladder** with an illegal memory access.
   The idea was to precompute the 12 swizzled fragment offsets once and add `kf` as an `LDS`
   immediate. `pgm_sw8`'s XOR mask sits on bits 4–6 and `kf = 32` flips bit 5, so
   `sw8(off+32) == sw8(off) ^ 32`, **not** `sw8(off) + 32` — 2048 of 4096 (row, lane) cases
   mismatch, and the bad addresses left the tile. The `+4` inside a 16-byte line *does* commute,
   which is what Result 6(a) relies on; the `+32` across k-subgroups does not. Rung removed; the
   dead end is recorded in the bench header because rung 1 shows the whole address-math bill is
   0.6% and a correct two-base-set hoist would not have been worth the registers anyway.
2. **`gpulease` reports its own child as a foreign process.** `foreign()` compares
   `nvidia-smi` pids against `$$`, which is the `gpulease` shell, not the CUDA process it
   `exec`s. So *every* leased run self-reports `WARN foreign-during` and exits 76 ("completed
   but contended"). rc=76 therefore carries no information about contention. Not fixed here —
   flagged because it silently devalues the one guard the campaign relies on.
