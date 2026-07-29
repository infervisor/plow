# AITER / Tensile / CK / ThunderKittens — what transfers to plow, and what cannot

Research note for the MI355X (gfx950 / CDNA4) window. Companion to
`plans/mi355x-cdna4-readiness.md`. Written on a box with no AMD GPU, so every
quantitative claim here is either quoted from a measurement already recorded in
this repo (attributed) or derived from disassembly (reproducible with
`scripts/asm_audit.py`). Nothing is estimated.

> ## ⚠ CORRECTION (2026-07-27, measured on MI355X). §1 below is HALF WRONG.
>
> This note says so itself: *"Written on a box with no AMD GPU."* It has now been tested on one,
> and the "type error" claim in §1 holds for **decode** and fails for **prefill**.
>
> A hipBLASLt/Tensile assembly kernel was dispatched **directly from plow's own HSA backend** — no
> HIP runtime, no `libhipblaslt` — and produced **bit-exact** output (0 of 33,554,432 elements
> differ) at **1.2–1.7×** plow's best tile on real Gemma-4-31B prefill shapes.
>
> The premise that fails is "there is no host-side enqueue". **plow's prefill is already ~97
> segmented AQL dispatches** across three co-loaded code objects (`run_segmented`,
> `crates/plowrt/src/exec/amd.rs:1046`). An external object dispatched as its own segment is the
> *existing mechanism*, not a violation of it — and `module_load` already creates a per-module
> executable, so a 4th co-resident object needs **zero backend changes**.
>
> **§1 remains correct for decode**, and that distinction is the whole point: decode really is one
> cooperative launch per token, ops really are inlined arms of a switch inside a workgroup, and
> decode GEMV is already at 89% of the memory ceiling. There is nothing to win there and a verified
> property (1 dispatch/token) to lose.
>
> Two further corrections from the same measurement: the **2×** figure quoted for hipBLASLt is
> **1.2–1.7×** on real Gemma shapes (the 2× came from rocprof-depressed Qwen shapes with K=2560),
> and the bigger lever turned out not to be the assembly at all — at M=128 *both* kernels sit at
> 5–8% of peak. **The skinny/small-M deficit is a tile-inventory problem** (Tensile ships 336 macro
> tiles; plow's selector has 3, and plow's own measured-best tile is not selectable).
>
> Full detail and the ranked candidate survey: `plans/knob-contract.md` §0-EXT-RESULT.
> Working driver with the decoded ABI documented inline: `runtime/ubench/gemm_tensile_ext.c`.

## 1. The structural fact: none of them can be linked in

AITER, Tensile, hipBLASLt and Composable Kernel are **launch-per-op** libraries.
The host enqueues one kernel per GEMM and the library chooses the grid.

plow is the opposite shape. One persistent kernel, `grid == CU count`, resident
for the life of the model, walking a counter-gated instruction stream
(`runtime/common/dev_isa.h`). Ops are *inlined arms of a switch inside a
workgroup*, not launches. There is no host-side enqueue to hand to hipBLASLt,
and a library kernel cannot be called from inside a workgroup.

This is not a cost/benefit call — it is a type error. The co-residency argument
that makes plow's counter protocol sound (every workgroup resident, so a
spinning consumer can never starve a producer) is exactly what a library launch
breaks. Anyone proposing to "just call hipBLASLt for the big GEMMs" is proposing
to give up the execution model that is the point of the project.

ThunderKittens is a different case: it is a *tile abstraction library* (register
and shared-memory tile types plus explicit async pipelining), not a kernel
catalogue, and it is written against NVIDIA's `wgmma`/TMA. Its AMD story is thin
and its primitives assume hardware plow does not have here (TMA descriptors, a
warp-group MMA). The **idea** transfers; the code does not.

## 2. What actually transfers: the pipeline schedule

`runtime/amd/op_gemm.h` already records the diagnosis, from measurements taken
on real MI350X hardware:

> bf16 barrier-halving gave only +7-9%; fp8's 4x barrier cut gave parity. The
> ONLY path to the library's 1450 bf16 / 2600 fp8 is the occ-1 4-wave
> deep-pipeline rewrite (512-reg budget -> deep REGISTER prefetch that breaks
> the read->MFMA dependency, as CK's Intrawave v4/v5 pipelines do).

Two independent measurements agreeing that cutting barriers does nearly nothing
is strong evidence the bottleneck is the **LDS-read -> MFMA dependency chain**
itself: each `ds_read` feeds the very next MFMA with no independent work
between, so the wave stalls on operand latency regardless of how few barriers
it crosses.

That is precisely what CK's Intrawave v4/v5 attack, and the mechanism is not
exotic: drop to **one wave per SIMD (occupancy 1)** to get the full 512-register
budget, then spend those registers on a **register prefetch deep enough that the
operands for MFMA *n+k* are already in flight while MFMA *n* issues**. Latency
is hidden by the schedule instead of by occupancy. AITER's hand-written assembly
kernels do the same thing; that is most of why they beat compiler-scheduled HIP.

plow can express this today — `segmented dispatch` already builds a 4-wave /
`FA_DC=256` / occ-1 object for the flash-prefill segment
(`PLOW_BUCKET_FLASH` in `scripts/build_gfx950.sh`). The GEMM has simply never
been given the same treatment.

## 3. Measured here: the 4-wave GEMM needs a different TILE, not just a flag

The naive version of that rewrite — take the shipped 256x256 GEMM and compile it
at 4 waves — **does not work**, and the disassembly says so without a GPU.

Sweep over tiles at `PLOW_WG_WAVES=4` (wave grid 2x2), bf16 `d_gemm_t`, measured
with `scripts/asm_audit.py`. `burst` = longest back-to-back MFMA run, `stalled` =
MFMAs issued immediately after an `s_waitcnt`, `spill` = scratch instructions:

| tile      | MFMA | burst | stalled  | spill |
|-----------|------|-------|----------|-------|
| 256x256   | 64   | 16    | 13/64    | **792** |
| **192x256** | 48 | **12** | **4/48 (8.3%)** | **0** |
| 256x128   | 32   | 8     | 4/32     | 0 |
| 128x256   | 32   | 8     | 4/32     | 0 |
| 128x128   | 16   | 4     | 4/16     | 0 |
| 256x64    | 16   | 4     | 4/16     | 0 |
| 128x64    | 8    | 2     | 4/8      | 0 |

Shipped baseline for comparison — 256x256 at **8** waves, wave grid 2x4:
MFMA 32, burst 8, stalled 4/32 (12.5%), spill 0.

Reading:

- **256x256 at 4 waves is unusable.** A 2x2 wave grid gives each wave a 128x128
  accumulator = 256 f32 VGPRs of accumulator *alone*, before operands. 792
  scratch instructions. Halving the wave count without shrinking the tile just
  moves the work onto a register file that cannot hold it.
- **192x256 at 4 waves is the candidate.** Against the shipped 8-wave config it
  is **1.5x the MFMA burst (12 vs 8)** with a **lower stall ratio (8.3% vs
  12.5%)** and still **zero spill** — better on both pipeline metrics at once.
  It is also not a novel tile: `gemm_fp8_c5` in `test_kernels.hip` is 192x256,
  described there as "the Qwen prefill tile".
- The fp8 twin ranks the same way (fp8 rows in the same sweep), which is
  expected: the operand width changes the LDS traffic, not the accumulator
  pressure that decides the cliff.

### 3a. CORRECTION — 4 waves has already been measured, and it LOST

The table above ranks per-wave instruction scheduling. It is blind to the thing
that actually makes the shipped kernel fast, and `op_gemm.h` already records the
hardware measurement that settles it — mean over the six Gemma-31B projections,
**through the interpreter**:

```
8 waves, 2x4 grid, ping-pong, BN=256    573 TF/s   <- current
4 waves, 2x2 grid, no ping-pong, BN=256 533
8 waves, BN=128                         531
4 waves, BN=128                         484
```

**4 waves is ~7% slower, not faster.** The reason is exactly what `burst` cannot
see. The 8-wave config's throughput does not come from long per-wave MFMA runs;
it comes from a **cross-wave ping-pong**: 8 waves as two groups of 4, `wcol =
wave % 4` placing one wave of each group on each SIMD, and a single extra
`s_barrier` executed by group 1 only, which offsets that group by one cluster
for the rest of the kernel. Group 0 is then in a memory phase exactly when group
1 is in an MFMA phase. At 4 waves there is one wave per SIMD, no co-resident
partner to hand the SIMD to, so `GM_PP` compiles out and that overlap is gone.

`burst` measures instructions within one wave. The ping-pong hides latency
*between* waves on a shared SIMD. A deeper per-wave burst bought at the cost of
losing the partner wave is a bad trade, and the 533-vs-573 measurement is what
that trade costs.

So the §3 sweep should be read narrowly: it says which tiles **fit** at 4 waves
(256x256 does not — 792 spills), not which config is fastest. It does not
support moving the GEMM to 4 waves, and the earlier draft of this note that
implied otherwise was wrong.

### 3b. What the sweep is still good for

192x256 is a real win — but at **8 waves**, and `op_gemm.h` already measured it,
interleaved (256 vs 192 back-to-back, to defeat the cold-vs-warm clock ramp that
otherwise manufactures a fake winner):

- **Qwen3-4B (K=2560), through the interpreter:** 192x256 BEATS 256x256 —
  o_proj +32%, down +21%, gate +11%, q +6% (kv N=2048 is -8%).
- **Gemma-31B (K=5376):** 256x256 wins by 4-8%.
- **Standalone on an idle GPU:** 256x256 wins outright.

The mechanism: 192x256's 96-register accumulator stays in **arch VGPRs (0 AGPR,
measured)**, while 256x256's 128-AGPR accumulator brackets each MFMA with
`v_accvgpr` moves. Those moves are free when power is abundant and expensive in
the power-limited regime that 256 CUs of sustained dense MFMA actually run in —
which is the only regime serving runs in.

So the tile is **K- and power-dependent, with no global winner**, and the
correct fix is per-shape tile selection (`plans/tile-specific-gemm.md`), not a
new default. `GM_BM`/`GM_BN` are already compile-time overridable for exactly
this.

### 3c. What is genuinely open

The occ-1 deep-prefetch idea is not dead — but the honest statement is that it
has been tested only in its weak form (4 waves, same tile, ping-pong lost, -7%).
Nobody has built the strong form: 4 waves **with** a tile small enough to leave
registers for a prefetch depth that the 8-wave config cannot afford, trading the
ping-pong for a software-pipelined operand stream deep enough not to need it.
CK Intrawave v4/v5 is that strong form. The §3 sweep shows 192x256 and 256x128
have the register headroom to attempt it.

**Status: MEASURE-ON-HW, low prior.** The one existing data point on the weak
form is negative. Do not spend the hardware window on this before the cheap
certain wins (fp8 prefill end-to-end, mxfp4 decode, per-shape tile selection).

## 4. Using AITER on the rented box (as an oracle, not a dependency)

Two legitimate uses, neither of which links AITER into plow:

1. **Numerics oracle.** The new mxfp4 path has no CPU reference for the
   hardware's E8M0 rounding. AITER's mxfp4 kernels feed
   `e8m0_to_f32(byte) = 2^(byte-127)` into the same `v_cvt_scalef32_*`
   instructions, so its output is a ready-made reference for
   `d_gemv_mxfp4_k`.
2. **Roofline target.** AITER's achieved TF/s on the same shapes is the number
   to beat, and the honest measure of how much of the 1450 bf16 / 2600 fp8 gap
   the occ-1 rewrite actually closes.

## 5. What NOT to copy

- **Do not fold arbitrary f32 scales into `v_cvt_scalef32_*`.** Its `scalef32`
  operand is E8M0 (exponent-only). Correct for MX by construction; a ~22% error
  for DeepSeek/GLM block-fp8, probed on gfx950 2026-07-17 (`amd_common.h`).
  AITER feeds it only E8M0 scales — copying the call without the precondition is
  how that becomes a silent accuracy bug.
- **Do not adopt a runtime tile switch.** `interp.hip` documents the trap: the
  interpreter inlines every arm, so register allocation is the worst case over
  all of them; a runtime `cfg` switch pulled all three GEMM instantiations into
  that worst case (260 VGPR + 128 AGPR) and the dispatch was rejected outright.
  Tile choice must stay compile-time, per bucket/segment. This is the single
  biggest structural difference from how a library is organised, and it is why
  "just port the CK config space" does not work.
