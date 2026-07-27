# PX-7 — what plow's fp8 prefill GEMM is actually worth, and why 2 blocks/SM will not save it

RTX 5090 (sm_120a, **170 SMs**, 96 MiB L2) · 2026-07-26 · bench `perf-data/px7_w8a8_ceiling_bench.cu`
Run under `perf-data/harness/gpulease`. Companion to PX-6; `px6_wavequant_bench.cu` untouched.

## Question

plow's 127k prefill implies ~190 TFLOP/s on its GEMMs against this card's in-tree measured fp8
peak of **503.8 TFLOP/s** (`op_gemm.cuh:1282`, rtx-05) — the same `mma.m16n8k32.f32.e4m3` class
vLLM runs at ~peak. Two explanations with completely different fixes:

* **(a) scheduling** — tiles are fine, the megakernel starves them (`occ_per_sm = 1`, because
  flash-prefill's 85,248 B smem sets the whole object's occupancy). Fix = make the lean occ-2
  GEMM segment object reachable from serve. **Runtime work.**
* **(b) kernel body** — a tile just costs this much. Fix = `d_gemm_w8a8`. **Kernel work.**

Measured at an **oracle grid** `G* | T` so wave quantization is zero by construction (u = 1.000
exactly, asserted), L2-cold (700 MB weight replication, cycled), 8 warm + 30 timed.

**Null control PASSES:** the bf16 arm at `q_full` gives **117.4 TFLOP/s** against PX-6's
**116.5** at u = 1.000. Same harness, same answer, so the fp8 numbers are trustworthy.

## Result 1 — the body ceiling (1 block/SM, u = 1.000)

| shape    | M    | w8a8 TFLOP/s | % of 503.8 | bf16 TFLOP/s | fp8/bf16 |
|----------|------|--------------|------------|--------------|----------|
| gate\|up | 8192 | 305.3        | 60.6%      | 145.6        | 2.10x    |
| down     | 8192 | 321.0        | 63.7%      | 151.8        | 2.12x    |
| q_full   | 8192 | 243.3        | 48.3%      | 118.3        | 2.06x    |
| o_full   | 8192 | 311.7        | 61.9%      | 150.1        | 2.08x    |

fp8 delivers a clean **2.0–2.1x over bf16** everywhere — the w8a8 arm is working as designed.
But it tops out at **~60% of fp8 peak with ZERO quantization and nothing else running.**

## Result 2 — 2 blocks/SM is worth ~5%, not 2x. **This kills option (a).**

plow sizes its persistent grid as `occ * sm_count` (`exec/gpu.rs`), so an occ-2 object launches
2P blocks. Timing the same shapes at a 1/SM and a 2/SM oracle grid:

| shape    | M    | 1 blk/SM | 2 blk/SM | ratio      |
|----------|------|----------|----------|------------|
| gate\|up | 8192 | 305.3    | 305.2    | **1.000x** |
| down     | 8192 | 321.0    | 373.5    | 1.164x     |
| q_full   | 8192 | 243.3    | 286.9    | 1.179x     |
| o_full   | 8192 | 311.7    | 363.0    | 1.165x     |

**`gate|up` does not move at all**, because `d_gemm_glu_w8a8` is *register*-limited to
`occ = 1` (measured: `occ w8a8=2 glu=1`) — its `acc[2][8][4]` plus two B operands. Handing it
320 blocks just runs two waves of 160. And `gate|up` is **~2/3 of prefill GEMM FLOPs**.

Weighted over the real op mix: `0.67*1.00 + 0.33*1.17` = **~1.05x**.

> **Do not build Step 2.** Making the occ-2 GEMM segment object reachable from serve — the
> runtime change to `check_coarse_single_segment()` and the per-segment relaunch loop — buys
> ~5% on the prefill GEMM, not the ~2x the occupancy argument suggested. The dominant op cannot
> use the second block. Cost/benefit is nowhere near the counter-protocol risk.

At M=2048 the 2/SM grid is often *worse* (0.883x on `down`, 0.900x on `o_full`): the available
divisor grid (240) is not two clean waves.

## Result 3 — re-sizing the real target

Isolated, weighted over the op mix, the GEMMs run at **~300 TFLOP/s**. Total prefill GEMM work
at 127k is `2 * 10.9e9 * 126976` = 2768 TFLOP, so the GEMMs alone should take **~9.2 s**.
The measured linear term of plow's prefill fit is **14.6 s**.

**So ~5.4 s (37%) of the prefill linear term is NOT GEMM.** The earlier "plow's GEMM runs at
190 TFLOP/s" statement conflated the two; corrected, it is ~300 TFLOP/s of GEMM plus 37%
overhead. Candidates, none yet attributed: per-packet counter gates (~910 packets/program),
`QUANT_FP8` activation quantization (w8a8 needs a per-row quant pass before every GEMM),
norms/RoPE/residuals, and wave quantization at the real grid (~6% at tm=16 by PX-6's model).

Two independent targets remain, and **both are needed** — neither alone reaches vLLM's 14.2 s:

| target                                   | now    | ceiling if fixed | method            |
|------------------------------------------|--------|------------------|-------------------|
| non-GEMM overhead in the linear term      | 14.6 s | 9.2 s            | attribute first   |
| GEMM body (60% -> ~90% of peak)           | 9.2 s  | ~6.1 s           | `d_gemm_w8a8`     |
| hd512 flash prefill (quadratic)           | 16.4 s | ~8.2 s           | `d_flash_prefill_px4` |

## Gates

| gate | result |
|---|---|
| null control (bf16 reproduces PX-6's 116.5) | **PASS** — 117.4 |
| oracle grid u = 1.000 exactly | **PASS** — asserted per cell |
| L2-cold | **ENFORCED** — 700 MB replication + cycling |
| GPU exclusive | **ENFORCED** — `gpulease` |
| numeric oracle | **NOT RUN** — random operands, no reference; no production kernel changed |

A bug found and fixed mid-run, recorded because it produced an impossible number: the bf16
control calls plain `d_gemm` but was initially credited the GLU 2x FLOP factor, reading 287.8
TFLOP/s = **137% of bf16 peak**. Physically impossible readings are the cheapest bugs to catch.

## Result 4 — the full prefill attribution (this is the actionable part)

Summing px7's per-shape isolated times over the real Gemma-4-12B layer mix (gate|up x48,
down x48, q/k/v/o_slide x40, q/k/o_full x8 — `k_eq_v` means full layers have no `v_proj`)
gives a **GEMM-only model for one prefill chunk**: **152.87 ms at M=2048**, **601.63 ms at
M=8192**. Both imply the same ~9.5 s of GEMM for a 127k prompt, as they must.

Against the measured 30.85 s (mixed fp8 KV, chunk 2048, NS=32):

| component                                  | time    | share  |
|--------------------------------------------|---------|--------|
| GEMM (px7 model, 63 x 152.87 ms)           | 9.63 s  | 31.2%  |
| **flash prefill** (fitted quadratic)       | **16.48 s** | **53.4%** |
| other linear (gates, QUANT_FP8, norms, RoPE, residual) | 4.74 s | 15.4% |

**The flash prefill is the majority of long-context prefill — not the GEMM.** Every earlier
statement in this campaign that framed the GEMM as the primary gap was wrong; it is 31%.

### Sizing the flash-prefill target

At 127k the 8 full-attention layers do `8 x 2(QK,PV) x 2 x N^2/2 x 16 heads x 512 hd` =
**2114 TFLOP**.

| engine | quadratic time | implied TFLOP/s |
|--------|----------------|-----------------|
| plow   | 16.48 s        | **128**         |
| vLLM   | 9.17 s         | **230**         |

plow is **1.80x behind**. bf16 peak on this card is 209.5 TFLOP/s, so vLLM's 230 is *above*
bf16 peak — it must be running fp8 tensor cores on **both** QK and PV. (Inference from the
number, not from reading vLLM; worth confirming before relying on it.)

plow's px4 arm already does `mma.m16n8k32.e4m3` for **QK** but dequantizes **V to bf16** for
the P.V mma — `op_attention.cuh` notes this explicitly: *"V still dequants to bf16 for the P.V
mma (fp8 P.V needs BKV=32 to fill k32 — Lever C/D)."* That is exactly the missing half, and it
explains why enabling the px4 fp8 arm recovered only 10% of the quadratic term (b 1.131e-9 ->
1.022e-9) instead of ~2x.

**Next step is therefore Lever C/D: fp8 P.V at BKV=32**, not GEMM work and not occupancy.
Ceiling if it lands: 16.48 -> ~9.2 s, i.e. total prefill 30.85 -> ~23.6 s against vLLM's 14.2.
Still not parity — the remaining 9.5 s of GEMM (at 60% of fp8 peak) and the 4.7 s of non-GEMM
linear overhead would both also have to come down.

## Result 5 — the smem budget says Lever C/D and BQ=64 are the SAME change

The px4 arm is hard-pinned: `static_assert(HD == 512 && BQ == 32 && BKV == 16)`. The generic
`d_flash_prefill<HD,BQ,BKV,FP8KV>` is parametric and allows BQ up to 64 (`WQK_M = BQ/16`,
`WQK_M * WQK_N <= 8` warps). What actually decides the tiling is shared memory. Per-block
staging bytes at HD=512 against the 101,376 B dynamic-smem cap:

| tiling          | bf16 staging | px4 today (bf16 + fp8 mirrors) | **full-fp8 staging** |
|-----------------|--------------|-------------------------------|----------------------|
| BQ=32, BKV=16   | 70,656       | ~88,000 (measured 85,248)     | 37,888               |
| BQ=64, BKV=16   | 108,032 ✗    | 142,080 ✗                     | 58,880               |
| BQ=32, BKV=32   | 108,032 ✗    | 125,440 ✗                     | 58,880               |
| **BQ=64, BKV=32** | 149,504 ✗  | 183,808 ✗                     | **83,968 ✓**         |

**This is the finding.** Today's px4 arm pays for BOTH a bf16 mirror and an fp8 staging tile,
which is why it is stuck at the smallest tiling. Dropping to fp8-ONLY staging (Qs8/Ks8/Vs8, no
bf16 Ks/Vs) makes **BQ=64, BKV=32 fit in 83,968 B — LESS than the 85,248 B the arm already
claims today.** Occupancy is unchanged; the tiling is free.

Three wins arrive as one change, which is why they should not be sequenced:

1. **BKV 16 -> 32** fills `k32`, which is the stated precondition for fp8 P.V (Lever C/D).
2. **BQ 32 -> 64 halves the KV re-read.** Flash re-streams the whole KV per q-tile, so traffic
   scales as `1/BQ`. Estimated 4.13 TB -> 2.07 TB over a 127k prefill (bf16 KV), and half that
   again with e4m3 K/V.
3. **Both mmas run e4m3.** QK is already `mma.m16n8k32.e4m3`; P.V becomes so too. P is in [0,1]
   after the online softmax, which is exactly why FlashAttention-3's fp8 path quantizes it.

Combined ceiling is therefore better than the 1.80x deficit alone suggests — the compute goes
~2x AND the traffic halves. Sizing it conservatively at 2x: flash prefill 16.48 -> ~8.2 s and
total prefill 30.85 -> ~22.6 s, against vLLM's 14.2 s.

**Still not parity on its own**, and that is the honest bottom line: the 9.6 s of GEMM (60% of
fp8 peak) and the 4.7 s of non-GEMM linear overhead would both also have to come down. But this
is now ONE well-scoped kernel change with a measured budget and a named precondition, rather
than the three-front programme it looked like two results ago.

**Required gate before it ships:** greedy-token parity against a bf16-KV run at >= 8k tokens,
plus the needle test at 69k. e4m3 P carries ~2 decimal digits; the online-softmax rescale must
be proven not to compound across tiles.
