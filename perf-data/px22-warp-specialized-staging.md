# PX-22 — warp specialization DOES pay on sm_120a: 1.14x on the w8a8 prefill GEMM body, bit-exact

RTX 5090 (sm_120a, 170 SMs, 96 MiB L2) · 2026-07-26 · bench `runtime/bench/nvidia/px22_ws_stage_bench.cu`,
build `perf-data/px22_build.sh`, run under `perf-data/tools/gpulease`.

## Question

PX-9 attributed plow's 61–66%-of-fp8-peak w8a8 prefill GEMM to the `cp.async` staging path **by
elimination** — mainloop at 94.3% of the ceiling, cuBLASLt at 95–99% on the same shapes, stage
depth worth 0%, `BN=256` worth less than nothing, L2 residency not predictive. PX-13 then killed
the fix PX-9 ranked first: 2-D `cp.async.bulk.tensor` measured **1.104x SLOWER** than the shipped
`cp.async.cg` on the production `[128][64]` e4m3 tile, byte-identical output.

That left one untested half of the CUTLASS recipe on this part: **producer/consumer warp
specialization**. The question here is narrow and it is the gate for everything downstream:

> Does decoupling the copy ISSUE from the mma, **inside one op body at the shipped 256 threads**,
> beat the barrier-synchronised uniform loop on the production `[128][64]` e4m3 tile?

PX-16's objection — *"all 8 warps run the same op body out of the same switch, so there is no
producer/consumer asymmetry to exploit"* — is true of the interpreter's **dispatch** and irrelevant
to the **body**. All 8 warps agreeing on *which* op to run is required (the switch is warp-uniform);
the asymmetry lives one level down, after the opcode is resolved. So no separate object, no
segmented dispatch, no `PLOW_UNISEG` are on the critical path — and, per Result 5, none are needed.

**The instrument is SM cycles per K-tile** (PX-9's unit: a clock that moves with power draw cannot
corrupt a cycle ratio). Every arm runs real global memory, the production tile, the shipped
`pgm_sw8` V2 swizzle, the shipped `LDS.64` fragment readers, the real `m16n8k32` e4m3 mma and a
real epilogue. Shape: `gate|up` at the real prefill bucket, M=1024, N=15360, K=3840, grid 160
(960 tiles / 6 per block, so no wave quantization inside the cycle counter).

Everything below is the mean of **two runs under separate leases**; they agree to **0.3% on every
cell** (raw: `perf-data/px22-warp-specialized-staging-raw.txt`).

## Result 1 — the shipped body pays 466 cyc/K-tile of EXPOSED staging. Warp specialization removes 73% of it.

| arm | cyc/K-tile | TFLOP/s | % of 518.5 | regs | spill |
|---|---|---|---|---|---|
| mma floor at 1024 FLOP/clk/SM (arithmetic) | 2048 | 518.5 | 100% | — | — |
| `mma_only` — ring pre-staged once, mainloop + barrier, **no global** | 2673 | 361 | 69.6% | 149 | 0 |
| `mma_only nobar` — same, barrier deleted | 2506 | 385 | 74.3% | 149 | 0 |
| `mma_only 4warp` — 4 warps, 2x2 grid, no barrier (**the ws consumer's own floor**) | 2619 | 370 | 71.4% | 236 | 0 |
| `stage_only` — staging + wait + barrier, **no mma** | 1160 | — | — | 54 | 0 |
| **`uniform` NS=3 — THE SHIPPED BODY** | **3139** | **309.4** | **59.7%** | 172 | 0 |
| `uniform` NS=4 | 3071 | 315 | 60.7% | 160 | 0 |
| `uniform nobar` — both per-K-tile `__syncthreads` deleted (RACY, timing-only) | 2615 | 368 | 71.0% | 172 | 0 |
| **`ws4` NS=6 — 4 producer warps + 4 consumer warps, same 256 threads** | **2745** | **349** | **67.4%** | 211 | 0 |
| `ws4` NS=5 / NS=4 / NS=3 | 2752 / 2762 / 2767 | 349 | 67.3% | 211 | 0 |

**`uniform` → `ws4` = 3139 → 2745 cyc/K-tile = 1.144x** (309.4 → 349.5 TFLOP/s = 1.13x; the TFLOP/s
ratio is smaller than the cycle ratio because the arms sustain slightly different clocks, which is
exactly why the ladder is denominated in cycles).

The decomposition, all in cyc/K-tile:

* Staging measured alone costs **1160**, but the shipped body is only **466** above its own
  no-global floor (3139 − 2673). So **60% of the staging already overlaps** in the uniform loop and
  40% is exposed. PX-13's 818.9 is the same quantity measured without the production predication,
  swizzle and barriers; 1160 is that arm re-measured with all three in place. Neither number is the
  cost that matters — **466 is**, and it was never measured before.
* Under warp specialization the exposed staging drops to **126** (2745 − 2619, against the ws
  consumer's *own* no-staging floor). **Warp specialization removes 340 of the 466 exposed cycles,
  i.e. 73% of the exposed staging cost.**

## Result 2 — the mechanism is the barrier, and it costs 3.1x more in context than in isolation

The coordinator asked for the barrier bill separately. It is worth having on its own:

| context | with barrier | without | barrier cost | as % |
|---|---|---|---|---|
| mainloop only, no global traffic (`mma_only`) | 2673 | 2506 | **167** | 6.7% |
| the real body, `cp.async` in flight (`uniform`) | 3139 | 2615 | **524** | **20.0%** |

PX-9's ladder measured the barrier at 5.4% with no `LDGSTS` in flight; this measures the same two
`__syncthreads` at **20.0%** with the real staging path running. The 3.1x difference **is** the
mechanism: every warp issues its 4 `cp.async` lines and then stops at a barrier, so there is no warp
left to issue mma while the LSU works, and the mma pipeline drains once per K-tile. That is why
stage depth was worth 0% in PX-9 — more buffers do not help a loop whose warps are all blocked at
the same barrier.

Two corroborating readings:

* `uniform nobar` (2615, racy) ≈ `mma_only 4warp` (2619) ≈ the ws consumer's floor. A
  barrier-free uniform loop is the theoretical best case for removing the serialization, and it is
  worth 524 cyc/K-tile. **`ws4` captures 394 of those 524 = 75%**, legally, bit-exactly. The
  remaining ~130 is the mbarrier handshake.
* Stage depth in `ws4` is worth **0.8%** across NS=3→6 (2767 → 2745) at the production K. Depth is
  still not the lever — decoupling is.

## Result 3 — the producer needs ONE warp. The 4+4 split is forced by tiling, not by copy throughput.

| arm | cyc/K-tile |
|---|---|
| `ws4` NS=4, all 4 producer warps issue | 2762 |
| `ws4 iss2` NS=4, only 2 of the 4 issue | 2758 |
| `ws4 iss1` NS=4, only **1** warp issues all 1024 `LDGSTS` per K-tile | 2757 |

Within noise. **One warp is enough to keep the pipe full once the copies are decoupled** — the
staging path was never issue-bandwidth-limited, it was *serialization*-limited, which is the same
conclusion Result 2 reaches from the other side.

That matters for the design: the 4-producer/4-consumer split is chosen **only** because 128x128
with whole 16-row fragments cannot be tiled by 6 or 7 warps. A 1- or 2-producer split would leave 6
or 7 warps on the mma and should be strictly better — it needs a tile whose warp grid divides
(BM=96 for 6 warps as 3x2, or BN=192 for 6 warps as 2x3). **Untested, and the single most promising
follow-up.**

The consumer's fragment shape matters a little: `ws4b` (4x1 warp grid, 2 A-frags + 16 B-frags)
measures 2805 vs `ws4`'s 2762 (2x2, 4 A-frags + 8 B-frags) — the 2x2 grid is 1.6% better and is
what the numbers above use.

## Result 4 — the win is decoupling, not the wider warp tile

At 256 threads, moving 4 warps to production forces the 4 remaining consumers onto a 2x2 grid
(`acc[4][8][4]` = 128 f32/thread instead of 64), which also halves the fragment loads per block.
That is a confound, and `mma_only 4warp` is the control that removes it: the **same** consumer
shape, no staging at all, no barrier.

| floor | cyc/K-tile |
|---|---|
| 8 warps, 4x2 grid, no barrier (`mma_only nobar`) | 2506 |
| 4 warps, 2x2 grid, no barrier (`mma_only 4warp`) | 2619 |

The retiling is **4.5% WORSE**, not better — halving the mma warps costs more than the fragment
reuse buys. So warp specialization wins *in spite of* the retiling it forces, and the whole 1.144x
is attributable to decoupling the copies. It also means a 6- or 7-consumer split (Result 3) has
that 4.5% still on the table.

## Result 5 — it costs nothing the megakernel does not already have

The coordinator asked precisely where the static register ceiling binds. **It does not bind.**

| body | registers | spill | smem at the shipped depth |
|---|---|---|---|
| megakernel prefill object today (PX-9 gate) | 238 | 0 | arena 101,376 B cap |
| `uniform` plain w8a8 (this bench) | 172 | 0 | 49,152 B |
| **`ws4` plain w8a8** | **211** | **0** | **49,280 B** (+128 B of mbarriers) |
| `uniform` GLU at the shipped `GLU_BN=128` | **250** | 0 | 49,152 B |
| **`ws` GLU at `GLU_BN=64`** | **202** | **0** | 49,280 B |

* Both warp-specialized bodies are **below the object's existing 238-register allocation**, so
  integrating them does not raise the megakernel's max-over-arms number and **no separate object is
  needed on register grounds**.
* The smem cost is **+128 B** for `2*NS` mbarriers — 49,280 B against a 101,376 B cap.
* At 256 threads and 1 block/SM the hardware ceiling is 65536/256 = **256 registers/thread**. The
  *shipped GLU body* already sits at 250, i.e. 2.3% from that wall — that, not the ws body, is
  where the register ceiling actually binds today.
* The line where a separate object *would* become necessary: a 4-consumer split at `GLU_BN=128`
  needs two accumulator sets of `[4][8][4]` = **256 f32/thread** of accumulator alone. Impossible.
  That is exactly why the warp-specialized GLU must also halve its N-tile (Result 6).
* **No `sm_120a`-only instruction is used.** `mbarrier` + `cp.async.mbarrier.arrive.noinc` are
  sm_80+. The authorization to pin `sm_120a` and use `setmaxnreg` freely turned out not to be
  needed — see Result 7.

## Result 6 — the GLU arm (2/3 of prefill GEMM FLOPs) also wins, but only via `GLU_BN=64`, and that is the risk

`d_gemm_glu_w8a8` stages one A tile and two weight tiles and holds two accumulator sets. Measured
with the same protocol (ms and TFLOP/s only — cyc/K-tile is not comparable across N-tile widths):

| arm | ms | TFLOP/s | regs | vs shipped |
|---|---|---|---|---|
| **`glu uniform BN=128 STAGES=2` — AS SHIPPED** | **0.761** | **317.6** | 250 | 1.000x |
| `glu uniform BN=128 STAGES=3` | 0.741 | 326.1 | 240 | 1.027x |
| `glu uniform BN=64 STAGES=2` (the matched control) | 0.718 | 336.9 | 146 | 1.060x |
| `glu uniform BN=64 STAGES=4` | 0.733 | 329.8 | 152 | 1.038x |
| **`glu ws BN=64 STAGES=3`** | **0.674** | **358.5** | 202 | **1.129x** |
| `glu ws BN=64 STAGES=4 / 6` | 0.687 / 0.688 | 351.6 / 351.0 | 202 | 1.108x |

**1.129x on the shipped GLU arm — but it decomposes as 1.060x from halving the N-tile and 1.065x
from warp specialization on top.** The N-tile half is `PGM_GLU_BN=64`, and **PX-13 measured that
knob alone as a 2.3% end-to-end REGRESSION** on a real 127k prefill (33.21 s vs 32.46 s, 25x
outside the run-to-run band) despite being +9.3% isolated. So the GLU integration inherits a knob
whose microbench and runtime signs have already disagreed once, on this exact op. This is the
single largest risk in the proposal and it is why the recommendation below splits the two arms.

## Result 7 — `setmaxnreg` never wins. At 256 threads and 1 block/SM there is nothing to donate.

| arm | entry | producer / consumer | cyc/K-tile | spill B |
|---|---|---|---|---|
| `ws4` free (ptxas picks 211) | — | — | **2762** | 0 |
| `clamp168` — `__maxnreg__(168)`, no donation (control) | 168 | — | 2829 | 8 |
| `smr168` — + `dec 88` / `inc 248` | 168 | 88 / 248 | 2761 | 8 |
| `clamp128` — `__maxnreg__(128)`, no donation (control) | 128 | — | **9291** | **376** |
| `smr128` — + `dec 24` / `inc 232` (the CUTLASS split) | 128 | 24 / 232 | 2785 | 24 |
| `smr128b` — + `dec 88` / `inc 168` | 128 | 88 / 168 | 2807 | 8 |

* **The donation demonstrably works**: `clamp128` without it spills 376 B and runs **3.4x slower**;
  adding `dec 24 / inc 232` recovers all of it. So this is not a "the instruction did nothing"
  result — the registers really do reach the consumer, on sm_120a, in a real GEMM body.
* **It is still never the winner.** The best `setmaxnreg` arm (2761) ties the arm with no
  `__maxnreg__` at all (2762). The reason is arithmetic: at 256 threads with 1 block/SM the CTA's
  own pool **is** the entire 65,536-register file, i.e. 256 registers/thread are already available
  statically, and `setmaxnreg` is zero-sum *within that pool*. There is nothing it can hand the
  consumer that ptxas could not already allocate. It would only matter at ≥2 blocks/SM or at a
  larger block, neither of which applies to the prefill object.
* This reproduces on sm_120a exactly what `runtime/nvidia/experiments/hopper_warpspec_prefill.cu`
  found on sm_90a: *"setmaxnreg is never the winner; warp specialization alone IS worth it."*
* **Recommendation: do not put `setmaxnreg` in `op_gemm.cuh`.** It buys nothing here and it would
  make the object `sm_120a`-only for no measured gain.

## Result 8 — ~9% of the "mainloop" floor is the epilogue, not the mainloop (new, unclaimed)

Doubling K to 7680 amortizes the per-output-tile cost (accumulator zero-init + the scalar bf16
epilogue) over twice as many K-tiles. For the `mma_only` arms this is a **clean** control — they
have no global traffic in the mainloop, so nothing else changes:

| arm | K=3840 | K=7680 | per-output-tile share |
|---|---|---|---|
| `mma_only` | 2673 | 2452 | 8.3% |
| `mma_only nobar` | 2506 | 2283 | 8.9% |
| `mma_only 4warp` | 2619 | 2362 | 9.8% |

So the true mainloop floor at the production K is ~2450, and **~9% of what this bench calls the
"mma floor" is actually the epilogue** — the store map writes 64 separate 2-byte elements per
thread per tile. PX-9 estimated the epilogue at ~0.6% *by instruction count in an isolated ladder*;
measured inside a real body with a real store map it is an order of magnitude larger. That is a
lever nobody has costed, and it is roughly the same size as the one this file just closed.

**Caveat, stated because it cuts the other way:** for the arms that stage from global, K=7680 also
doubles the weight working set to 236 MB, well past the 96 MiB L2, so those columns change memory
regime and are **not** a clean amortization control. In that different regime the GLU ws arm
actually *loses* to the shipped uniform (0.974x) while the plain ws arm wins *more* (1.189x, and
only at NS=6). Production K is 3840; the K=7680 numbers are recorded as a regime warning, not as a
result.

## Gates

| gate | result |
|---|---|
| **bit-exact output, FNV-1a over the entire C plane** | **PASS** — all 12 plain arms (uniform, every ws variant, every register variant) hash **identically**; all 4 GLU arms hash identically to the shipped GLU body. Every arm accumulates each output element over the same k sequence in the same order; only the warp that owns it changes. |
| hash is discriminating, not vacuous | **PASS** — the zero-plane hash (`c13b395e6a1d0383`), the plain-GEMM hash (`6f9f6be9902baa9a`) and the GLU hash (`c727b42ba91121ea`) are all distinct, and the bench asserts an arm never equals the zero plane. Added **after** the bug below made the gate pass for the wrong reason. |
| reproducibility, two separate leases | **PASS** — every cell within 0.3%; headline ratio 1.141x / 1.146x |
| oracle / numeric parity vs the in-tree probe | **NOT RUN** — this bench is self-contained and proves arm-to-arm bit-identity, which is the property a body swap needs. It does not re-run `fp8_gemm_w8a8_probe.cu`; that gate belongs to the integration, not the microbench. |
| **byte-identical shipped cubins when the feature is off** | **PASS, trivially** — `git diff HEAD -- runtime crates include scripts` is **empty**. No kernel or runtime source was touched; the only new files are `runtime/bench/nvidia/px22_ws_stage_bench.cu` and `perf-data/px22_build.sh`. There is no feature to turn off yet. |
| `cargo test --workspace` | **PASS** — exit 0, 0 failures |
| `cargo build --release -p plowrt --features cuda,hf-tokenizer` | **PASS** — clean (warnings only, all pre-existing) |
| no reading above the 518.5 TFLOP/s fp8 ceiling | **PASS** — asserted per row by the bench; the best arm is 349.5 (67.4%). The one arm that tripped it early was `stage_only`, which does zero FLOPs — its TFLOP/s column is now suppressed rather than printed against a meaningless denominator. |
| `PLOW_ROOT` on every build | **ENFORCED** — `px22_build.sh` honours `PLOW_ROOT` and was always invoked with it |
| GPU exclusive | **ENFORCED** — every run under `gpulease` |
| L2-cold protocol | **NOT RUN, deliberately** — PX-9 measured cold vs warm within 0.5% on both plow and cuBLASLt at these shapes. This bench measures the *issue* path; the 59 MB weight is L2-resident by construction. |
| `ncu` counter attribution | **NOT RUN** — `ERR_NVGPUCTRPERM` in this container, as in PX-9 and PX-13. Every claim here is differential timing between arms that differ in exactly one thing, never a hardware counter. |
| **end-to-end prefill** | **NOT RUN** — no checkpoint in this worktree. Every number here is isolated-kernel, and PX-13 already demonstrated on this exact op that an isolated GEMM win can invert end to end. **Nothing in this file may be scaled through a prefill budget to claim a wall-clock win.** |
| megakernel integration | **NOT STARTED** — by design; the microbench was the gate |

### Bugs found mid-run, recorded

1. **The bit-exactness gate passed vacuously in the first run.** The operands were filled with
   `xr() & 0x7f`, and **`0x7f` is NaN in E4M3** — the only NaN encoding, hit 1/128 of the time. One
   NaN anywhere in a K=3840 reduction makes the whole C plane NaN, so every arm hashed the same
   *and the plain GEMM hashed identically to the GLU body*, which is what exposed it. Fixed by
   `rnd_e4m3()` (exponent restricted to 5..9, every operand finite in [0.25, 7.5]) and by adding
   the zero-plane / cross-body hash checks above. **The timings from the NaN run were unaffected
   — fp8 mma has no NaN rate penalty and they reproduce to 0.3% — but the gate they passed was
   worthless.** All numbers in this file are from the fixed bench.
2. **`__maxnreg__`, not `__launch_bounds__`.** Confirmed on sm_120a, as
   `runtime/nvidia/experiments/README.md` warns: ptxas silently drops the `setmaxnreg` effect under
   `__launch_bounds__`. All register arms here use `__maxnreg__`.
3. **A K sweep is not a clean amortization control for arms that stage from global.** Doubling K
   doubles the weight working set past L2 and changes the memory regime. It is clean only for the
   `mma_only` arms, which is the only place Result 8 uses it.

## Verdict and recommendation

**Warp specialization DOES pay on sm_120a**, unlike TMA (PX-13), occupancy 2 (PX-7) and the tile
sweep (PX-13). The last open lever in this campaign is real:

* **1.144x** on the plain w8a8 body (3139 → 2745 cyc/K-tile; 309.4 → 349.5 TFLOP/s; 59.7% → 67.4%
  of the 518.5 ceiling), **bit-exact**, at **211 registers / 0 spills / +128 B of smem**, inside one
  op body at the shipped 256 threads.
* **1.129x** on the GLU body — but 1.060x of that is `PGM_GLU_BN=64`, the one knob PX-13 already
  measured as a 2.3% end-to-end regression.
* It needs **no separate object, no segmented dispatch, no `setmaxnreg`, and no `sm_120a`-only
  instruction**. Both bodies fit under the object's existing 238-register allocation.

**Recommendation: proceed, in two separately-gated steps, and gate both END TO END.**

1. **Take the plain w8a8 body first.** It is the safe subset: 1.144x, bit-exact, no knob with a
   known sign disagreement, ~1/3 of prefill GEMM FLOPs. If the weighted arithmetic held it would be
   worth ~1.05x on the GEMM — **but that arithmetic is exactly the "scale an isolated ratio through
   an assumed budget" move that has produced three wrong rankings in this campaign, so it must be
   measured, not computed.**
2. **Treat the GLU body as a separate experiment**, because it cannot be had without `GLU_BN=64`,
   and that knob's microbench and runtime signs have already disagreed on this op. If the plain arm
   confirms end to end and the GLU arm does not, ship the plain one alone.

Before either, the cheapest remaining experiment is **Result 3's**: 1 or 2 producer warps with 6 or
7 consumers, which requires a tile whose warp grid divides (BM=96 as 3x2, or BN=192 as 2x3). Result
3 shows one producer warp suffices and Result 4 shows the forced 4-consumer retiling costs 4.5%, so
that variant should be worth another ~4% on top — and it is a tile change, not a new mechanism.
