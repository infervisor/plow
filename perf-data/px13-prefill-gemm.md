# PX-13 — the prefill GEMM tile space, scored end to end. The microbench picks the wrong tile.

RTX 5090 (sm_120a, **170 SMs**, 96 MiB L2, driver 580.159.03, CUDA 13.0) · 2026-07-26
Gemma-4-12B-it, fp8 W8A8, the PX-12 §2b asset. Benches `runtime/bench/nvidia/px9_gemm_body_bench.cu`
(extended) and `runtime/bench/nvidia/px13_tma_stage_bench.cu` (new). Every GPU run under
`perf-data/harness/gpulease`.

Companion to PX-9 (`px9-gemm-body.md`) and PX-12 (`px12-consolidated-baseline.md`). Neither file
is edited; where PX-13 corrects one it says so here.

## Question

PX-9 left three things: an unimplemented selector win (`BN=64` for `GEMM_GLU`, "the largest
single win in the knob space", +6.4% weighted), one live hypothesis (TMA staging), and a knob
space it declared exhausted. PX-12 left the standing — plow 25.96 tok/s vs vLLM 42.49, **82% of
plow's wall serial prefill**. This file implements the selector win, sweeps the prefill tile
space, and scores every arm **end to end** rather than on the microbench.

The headline is a negative and it is the point: **PX-9's largest knob-space win is a 2.3%
end-to-end regression.** The isolated bench and the runtime disagree in *sign*.

## Result 1 — the per-opcode N-tile is implemented, bit-exact, and reproduces PX-9's microbench

`PGM_BN` was global, which is why PX-9 could not take the win. `pgm_stage_b8` and
`pgm_load_bfrags_w8a8` are now templated on the N-tile; `d_gemm_glu_w8a8` runs `PGM_GLU_BN`, the
plain w8a8 body and both MoE bodies keep `PGM_BN`. Measured full-grid, L2-cold, and **padded to
the shipped 89104 B prefill arena** so occupancy matches the megakernel (occ 1, not the occ 2 an
unpadded isolated launch would get):

| shape | M | `GLU_BN=128` | `GLU_BN=64` | ratio |
|---|---|---|---|---|
| gate\|up | 8192 | 317.8 TFLOP/s | **347.4** | **1.093x** |
| gate\|up | **1024** ← the real bucket | 299.7 | **318.1** | **1.061x** |
| gate\|up | 512 | 296.7 | 305.5 | 1.030x |
| gate\|up | 128 | 218.4 | 203.6 | 0.932x |
| down / q_full / o_full | any | unchanged | unchanged | 1.00x |

Padded vs unpadded is within 0.6% on every cell, so **the win is not occupancy** — it survives at
1 block/SM. PX-9 predicted +9.6% at M=8192 and measured +9.3% here.

**Bit-exactness (numerics gate).** The N-tiling changes only how the output plane is cut; every
element still accumulates over the whole of K in the same k32 order. A `[hash]` mode added to the
bench dumps an FNV-1a of the entire C plane on deterministic operands. `GLU_BN` 128 / 64 / 32 give
**identical hashes on all four shapes** — bit-exact, not "within tolerance".

**Correction to PX-9 Result 5.** PX-9 swept `PGM_BN` at **M=8192 only**. The deployed packet's
prefill buckets are `[128, 512, 1024]`, so the runtime never launches a prefill GEMM above
M=1024. At the M that actually ships the win is +6.1%, not +9.6%, and at M=128 it is **negative**.

## Result 2 — end to end it LOSES, 2.3%, far outside the noise band

PX-12 §2's protocol verbatim: one 126,976-token prompt, 8 output tokens, concurrency 1, so wall
== serial prefill. The two arm dirs are byte-identical except `interp_sm120_pf.cubin`; the
`GLU_BN=128` object is **md5 `a10c0d8a…`, bit-identical to PX-12's arm C**, so this reproduces
the campaign's baseline exactly. Three alternating reps:

| arm | rep 1 | rep 2 | rep 3 | rep 4 | median | vs base |
|---|---|---|---|---|---|---|
| `PGM_GLU_BN=128` (shipped) | 32.43 | 32.48 | 32.46 | 32.49 | **32.47 s** | — |
| `PGM_GLU_BN=64` | 33.23 | 33.16 | 33.24 | 33.25 | **33.24 s** | **+2.3%** |

Run-to-run spread is **±0.03 s** within each arm and the two bands do not come close to
touching. The delta is 0.77 s — **25x the noise band**. The isolated bench
predicted −0.3 s. **It is wrong in sign, not just in magnitude.**

The knob is therefore implemented and left at **128**. `-DPGM_GLU_BN=64` re-enables it for anyone
re-testing.

## Result 3 — the whole tile space, scored end to end

Thirteen objects, one per tile arm, each with a distinct md5 (so every `-D` provably reached the
object), each scored on the same conc-1 127k prefill. `REG`/`STACK` are the prefill megakernel's,
from `cuobjdump -res-usage` — recorded because an isolated tile win that raises the object's
register pressure is not a win.

The last two columns are the **isolated** bench on the same arms at M=1024, full grid, L2-cold,
arena-padded — weighted `0.67*gate|up + 0.11*(down + q_full + o_full)`, PX-9's op mix.

| arm | defines | REG | STACK | **wall (s)** | **vs base** | micro (weighted) | micro rank |
|---|---|---|---|---|---|---|---|
| **base** | — | 240 | 1024 | **32.47** | **1.000** | 283.3 | 5 |
| `stages4` | `PGM_STAGES=4` | 240 | 1024 | 32.49 | 1.001 | 283.4 | 4 |
| `glustages3` | `PGM_GLU_STAGES=3` | 240 | 1024 | 32.52 | 1.002 | 283.7 | 3 |
| `stages2` | `PGM_STAGES=2` | 240 | 1024 | 32.60 | 1.004 | 283.2 | 6 |
| `st4glu3` | `STAGES=4 GLU_STAGES=3` | 240 | 1024 | 32.63 | 1.005 | 283.7 | 3 |
| `nolds64` | `PGM_W8A8_LDS64=0` | 240 | 1024 | 32.67 | 1.006 | 273.5 | 9 |
| `nosw8v2` | `PGM_SW8_V2=0` | 240 | 1024 | 32.69 | 1.007 | 281.9 | 7 |
| `glubn64` | `PGM_GLU_BN=64` | 240 | 1024 | 33.24 | 1.024 | **295.0** | **2** |
| `bn64` | `BN=64 GLU_BN=64` | 231 | 1024 | 34.07 | 1.049 | **302.8** | **1** |
| `bm64` | `PGM_BM=64` | 225 | 1024 | 34.72 | 1.069 | 259.2 | 10 |
| `bm256` | `PGM_BM=256` | **255** | **1680** | 38.77 | 1.194 | 171.9 | 11 |
| `bn256` | `BN=256 GLU_BN=128` | **255** | **1600** | **REJECTED (arena)** | — | 279.7 | 8 |
| `stages5` | `PGM_STAGES=5` | 240 | 1024 | **REJECTED (arena)** | — | *(PX-9: "noise")* | — |

**The microbench's top two arms are the runtime's worst two deployable arms.** `bn64` is the
isolated winner by 6.9% weighted — `gate|up` 300→318, `down` 256→289, `o_full` 245→278, three
shapes out of four — and it is **5.0% slower** on the real prefill. `glubn64` is isolated #2 and
runtime #8. The two rankings agree only at the bottom, where `bm64`/`bm256` lose in both.

A prefill tile tuner scored on the microbench would have shipped `bn64`: a 5% regression, chosen
with high confidence, from a bench that passes every oracle and every L2-cold and occupancy
control this campaign knows how to apply.

**The hand-set default is the optimum.** Nothing in the `-D` tile space beats the shipped
`BM=128 / BN=128 / BK8=64 / STAGES=3 / GLU_STAGES=2`. Every arm is flat or worse. That is a
useful negative and it is stated, not buried.

**`PGM_BM` was not even a knob.** It was `#define PGM_BM 128` with no `#ifndef`, so `-DPGM_BM=64`
silently produced a byte-identical object (caught by the md5 column). Now overridable, default
unchanged.

**PX-9's body changes, re-measured end to end.** `PGM_W8A8_LDS64` (the uint2 fragment read) is
worth **+0.6%** on the real prefill, not the +2.2% weighted its microbench implied; `PGM_SW8_V2`
is worth **+0.7%**. Both are real, both keep their sign, both are ~3x smaller than the isolated
number. Keep them.

## Result 4 — two arms in the space are UNDEPLOYABLE, and the failure is silent

`PGM_STAGES=5` puts the GEMM arena at 102400 B and `PGM_BN=256` at 102400 B; both exceed this
board's 101376 B dynamic-smem opt-in. plowrt does not fail — it loads with
**`prefill_buckets=0`** and consumes the prompt one *decode* step at a time. The `STAGES=5` arm
ran for **≥707 s** (aborted at 707.03 s, 0 successful requests) against the base arm's 32.5 s,
before `perf-data/px13_e2e.sh` grew an arena guard.

**PX-9 measured `STAGES=5` and `STAGES=6` in an isolated bench and recorded them as "noise".**
They are not noise and they are not slow — they cannot be deployed at all. An isolated kernel
bench allocates its own smem and never sees the union the megakernel must fit inside. This is the
same class of error as Result 2 and it is why the tuner has to score objects, not kernels.

Recorded bug: plowrt's tracing writes ANSI colour *between* a field name and its `=`, so
`grep 'prefill_buckets=0'` on a serve log never matches. The first version of the guard was
defeated by it.

## Result 5 — why the microbench inverts: the arms that lose are the arms that re-read operands

No hardware counters here (`ncu` is `ERR_NVGPUCTRPERM` on this container, as in PX-9), so this is
a **hypothesis consistent with the ordering**, not a measured attribution. It is stated because
the ordering is monotone and mechanistic, and because it predicts which knobs to leave alone.

Among the arms whose register/stack profile does not move (REG 240 or below, STACK 1024), the
end-to-end loss tracks how many times the tile decomposition makes the kernel re-read an operand:

| arm | tiles_m | tiles_n (gate\|up) | A re-reads | W re-reads | wall |
|---|---|---|---|---|---|
| base | 8 | 120 | 120x | 8x | 32.46 |
| `glubn64` | 8 | **240** | **240x** | 8x | 33.21 |
| `bn64` | 8 | **240** (both arms) | **240x** | 8x | 34.07 |
| `bm64` | **16** | 120 | 120x | **16x** | 34.72 |

`bm64` is worst because the operand it doubles is the **weight** (118 MB for `gate|up`, far over
L2, so the extra reads are DRAM); `glubn64`/`bn64` double the **activation** (3.9 MB at M=1024, so
the extra reads hit L2). Doubling DRAM traffic hurts more than doubling L2 traffic — which is the
observed order.

The isolated bench cannot see this. Its L2-cold protocol replicates and cycles the *weight* to
defeat L2, one op at a time, with the activation resident and nothing else competing. The
megakernel streams ~12 GiB of weights through 124 chunks x 48 layers with the flash-prefill arm
interleaved. `bm256` breaks the pattern for a different and visible reason: it hits the 255-register
ceiling and spills (STACK 1024 -> 1680).

**This is the mechanism PX-9's "the gap is the cp.async staging path" conclusion needs to be read
against.** PX-9 reached it by elimination inside a bench where global traffic was deliberately
made irrelevant.

## Result 6 — TMA on sm_120a: the capability table says it does not exist, and it does

`plowc tune --gpu rtx5090 --profile prefill_dense` prints:

```
capabilities: mma_sync=true wgmma=false tcgen05=false tmem=false tma=false ...
```

`IsaLevel::Sm120a.caps().tma == false` in `crates/hwspec/src/isa.rs`, asserted by a test. But the
tree already ships a working `cp.async.bulk` arm for sm_120a (`PLOW_NV_FA_TMA`), and
`runtime/bench/nvidia/px13_tma_stage_bench.cu` — which issues
`cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` — **assembles for
`compute_120a` and links against `cuTensorMapEncodeTiled`**. A prefill tuner that gated a TMA arm
on `caps().tma` would never generate it on this board. Reported, not fixed: changing a capability
row changes kernel selection for every target and belongs behind its own regression.

## Result 7 — TMA staging is 1.10x SLOWER than `cp.async.cg`. **PX-9's one live hypothesis is dead.**

`runtime/bench/nvidia/px13_tma_stage_bench.cu` stages the production `[128][64]` e4m3 A tile and B tile into
the production 49152 B 3-deep ring, at the production 256 threads, three ways, with no mma in the
way. 120 blocks, 60 K-tiles, `verify` mode asserts all three land **byte-identical** smem.

| arm | what it issues per K-tile | cyc/K-tile | GB/s | vs `ldgsts` |
|---|---|---|---|---|
| `ldgsts` | 1024 x `cp.async.cg` 16 B across 256 threads — **AS SHIPPED** | **818.9** | 6613 | **1.000x** |
| `tma2d` | 2 x `cp.async.bulk.tensor.2d` + mbarrier, from tid 0 | 904.0 | 6026 | **1.104x SLOWER** |
| `bulk1d` | 256 x 1-D `cp.async.bulk` + mbarrier, from tid 0 — the `PLOW_NV_FA_TMA` shape | 39539.5 | 148 | **48.3x SLOWER** |

byte-identity: `bulk1d` 0/16384 differing, `tma2d` 0/16384 differing. All three are the same copy.

**TMA does not help on sm_120a, and here is why.** A 2-D TMA copy is issued by *one* thread and
the whole 8 KiB tile is serialised through one TMA unit; `LDGSTS` spreads 512 independent 16-byte
requests across 256 threads and the entire LSU. On the datacenter parts TMA pays for itself in two
ways that **do not exist on this silicon**: multicast of one tile to a whole cluster
(`dsm_cluster = false` here) and freeing whole warps for a `wgmma` producer/consumer split
(`wgmma = false` here). Strip both and a bulk descriptor engine has nothing left to give against a
256-thread `LDGSTS` storm — it is a *narrower* issue path, not a wider one. So it loses by 10%.

This also rehabilitates PX-4: its ~2x is not a different result, it is the same result seen through
a shape 48x worse than the one measured here. One-D-bulk-per-row from a single thread is
catastrophic, and PX-4's arm was only 2x down because the flash mainloop hid most of it.

**What this does to PX-9 Result 7.** PX-9 ranked "TMA + mbarrier for the operand stream" first,
ceiling "+40% on the GEMM, 9.6 s → 6.5 s of a 127k prefill". The premise was reached **by
elimination** in a bench that deliberately removed global traffic, and the instruction it named is
now measured 10% slower than the one it would replace, on the exact tile. The remaining
possibility is that TMA still wins *inside* the mainloop by freeing LSU issue slots for the
fragment reads (PX-9 Result 6's disagreement is evidence such coupling exists) — but that is a
much smaller claim than "+40%", and Result 5 says the runtime's binding constraint is operand
re-reads, which no staging instruction changes.

**Caveat, stated:** the 59 MB weight is L2-resident in this probe, so it measures the staging
**issue** path, not DRAM. That is deliberate — it is the axis PX-9 blamed after showing its own
L2-cold and L2-warm numbers were within 0.5%.

Recorded bug found writing it: initialising the mbarrier with arrive-count 1 while all 256 threads
also call `mbarrier.arrive` over-completes the barrier, flips the phase with copies in flight, and
faults with `unspecified launch failure`. Only the issuing thread may arrive.

## Result 8 — the sm_120a tuning cell

`tuning/` held exactly one cell (`nvidia/sm_90a/h100-nvl`, an H100 NVL / 132-CU / 26B-MoE
interpreter-dispatch record). There was **no sm_120a entry, no 170-SM entry, no 12B-dense entry**,
so every tile constant in the shipped cubins was a hand-set default and
`plowc tune --status` correctly reported "no kernel measurements for this cell".

This campaign writes `tuning/nvidia/sm_120a/rtx-5090/prefill_tile_measurement.jsonl` — 13 rows,
one per arm, each carrying its define set, its pf-cubin md5, the object's registers and stack, the
end-to-end score, the isolated-bench numbers **as data rather than as the ranking key**, and its
state (`qualified` / `rejected` with a reason).

**It is deliberately not a `tunedb::KernelMeasurement`.** On NVIDIA the tile is a compile-time
macro of the *object*, and `plowc tune` says so itself — all three dense-GEMM opcodes alias to one
body, so "the real tuning axis here is which object is built, not which opcode is emitted". A
`KernelMeasurement` is keyed by `op_case` + `kernel_id` and has **no field that can distinguish
`BN=128` from `BN=64`**: they are the same kernel id in different objects. Until that entity grows
a build-identity column, a prefill-tile record cannot be expressed in it. That is the concrete
blocker for `plowc tune --profile prefill_dense` ever becoming measurement-backed, and it is a
schema change, not a measurement campaign.

Two further gaps found while doing it, both reported not fixed:

* **The inventory prober does not see the w8a8 path.** `plowc tune` lists three kernels
  (`PLOW_DOP_GEMM`, `_MED`, `_SMALL`) at tile `128x128x32` — the **bf16** tile. The ops that
  actually run a fp8 prefill (`PLOW_DOP_GEMM_FP8`, `PLOW_DOP_GEMM_GLU_FP8`) and their `BK8=64`
  tile are absent, so the tuner's own view of the search space excludes the thing being tuned.
* `crates/plowc/src/tune.rs` selects through `&NoMeasurements`, so `--shape` reports tier
  `portable` on every shape. This cell does not change that, because of the schema blocker above.

## Gates

| gate | result |
|---|---|
| **numerics — `GLU_BN` bit-exactness** | **PASS** — FNV-1a of the whole C plane identical at `GLU_BN` 128 / 64 / 32 on all four shapes. Bit-exact, not tolerance-matched |
| **byte-identity at the defaults** | **PASS** — the deployed-style prefill object built from `main + PX-9 body only` and from this tree are both md5 `a10c0d8a…`, which is also PX-12's arm C object. Every px13 source change (per-opcode `GLU_BN`, `PGM_BM` made overridable, the MoE explicit template args) is a provable no-op at its default |
| byte-identity of all five script-built objects | **PASS** — `scripts/build_sm120_cubin.sh` from `git archive` of the pre-px13 tree and from this tree gives identical md5s for `interp_sm120`, `_fp8kv`, `_pf`, `_pf_fp8kv`, `sample_sm120`. **Side finding:** those objects are *also* unchanged by PX-9's `d_gemm_w8a8` body edit, which the isolated bench measures at +6.5% — so the w8a8 GEMM body is dead-stripped from everything that script emits. Consistent with PX-12 §0b: the script cannot build the deployed prefill object |
| baseline reproduces PX-12 §2 | **PASS** — 32.46 s vs PX-12's recorded 32.39 s (0.2%) |
| every `-D` provably reached the object | **PASS** — 13 arms, 13 distinct pf-cubin md5s. Caught `-DPGM_BM` reaching nothing |
| oracle grid u = 1.000 | **PASS** — asserted per cell, and the assertion now uses the **per-opcode** N-tile (using `PGM_BN` for a GLU shape would assert on a tile count the kernel never produces) |
| L2-cold | **ENFORCED** — 700 MB replication + cycling, PX-9's protocol |
| occupancy control | **ENFORCED** — `PX13_PAD_SMEM=89104` pads the isolated launches to the shipped prefill arena. Padded vs unpadded within 0.6% |
| GPU exclusive | **ENFORCED** — `gpulease` on every run |
| end-to-end, per arm | **RUN** — 13 arms, conc-1 127k, the `GLU_BN` pair at 3 alternating reps (±0.03 s) |
| **end-to-end reps for the other 11 arms** | **1 REP EACH** — the ±0.03 s band from the 4-rep base / 4-rep `glubn64` pair is the justification, but no arm inside +1% of base is separated from it by this data. The four arms that matter (`glubn64`, `bn64`, `bm64`, `bm256`) are all >2% out |
| **`ncu` attribution of Result 5** | **NOT RUN** — `ERR_NVGPUCTRPERM`, same as PX-9. Result 5 is a hypothesis consistent with a monotone ordering, not a counter measurement |
| **TMA staging byte-identity** | **PASS** — `tma2d` and `bulk1d` both land 0/16384 differing bytes vs `ldgsts` |
| **TMA staging inside the mainloop** | **NOT RUN** — the probe has no mma. It shows the staging instruction itself is 10% slower; it does not exclude a second-order win from freeing LSU slots for the fragment read. That variant needs the fragment reader's swizzle matched to a TMA smem layout (`SWIZZLE_NONE` here would force `PGM_SW8_OFF`, itself worth -13% in PX-9) |
| **TMA staging, L2-cold** | **NOT RUN** — the 59 MB weight is L2-resident by construction, so the probe measures the staging ISSUE path. Deliberate: PX-9 showed cold and warm within 0.5% |
| **TMA GEMM port** | **NOT STARTED** — and correctly so: the staging instruction it would install is measured slower than the one it replaces |
| greedy-token parity | **NOT RUN** — not needed for the shipped default (unchanged object, md5-identical to PX-12 arm C); `GLU_BN=64` is bit-exact so parity is implied, and it is not shipped |
| concurrency-8 cell | **NOT RUN** — the shipped tile did not move, so there is nothing to re-measure at conc 8 |
| MoE GLU N-tile | **NOT SWEPT** — `op_moe.cuh`'s GLU body has the same two-accumulator register pressure and would take the same knob, but Gemma-4-12B is dense and no MoE arm was measured |

## Verdict

1. **PX-9's largest knob-space win does not exist.** `BN=64` for `GEMM_GLU` is +6.1% isolated at
   the real M, bit-exact, occupancy-neutral — and **2.3% slower** on the 127k prefill, 25x outside
   the noise band. Implemented, defaulted off.
2. **No tile in the `-D` space beats the shipped one.** The hand-set default wins. Two arms in the
   space are not slow but undeployable, and an isolated bench cannot tell.
3. **The prefill GEMM is not going to be moved by tile tuning.** Everything in this space is
   between 1.000x and 1.19x, in the wrong direction. The 32.5 s stands.
4. **TMA is dead as the prefill-GEMM lever on this part.** The 2-D `cp.async.bulk.tensor` that
   PX-9 ranked first is measured **1.10x slower** than the `cp.async.cg` it would replace, on the
   exact production tile, byte-identical output. It is a single-thread issue path on silicon with
   no cluster multicast and no `wgmma` to specialise for. The port is not worth starting.
5. **The sm_120a cell now exists**, and the honest thing it records is a negative plus the schema
   blocker that stops `plowc tune --profile prefill_dense` from being measurement-backed.
6. **Where the prefill actually is.** Nothing in this file moved the 32.5 s. PX-12's arithmetic
   therefore still holds: 8 x 32.5 s of serial prefill is 82% of the conc-8 wall, and the two
   things that can reach it are prefill/decode **overlap** and the flash-prefill arm — not the
   GEMM tile, which is now swept, and not TMA, which is now measured.

## Reproduce

    perf-data/px13_build.sh                        # isolated-bench arms
    GPU_LEASE_TIMEOUT=3000 perf-data/harness/gpulease px13c perf-data/px13_run3.sh
    perf-data/px13_sweep_build.sh                  # one prefill object + asset dir per arm
    GPU_LEASE_TIMEOUT=7200 perf-data/harness/gpulease px13sweep perf-data/px13_sweep_e2e.sh 1
    perf-data/px13_emit_tuning.py <sweep stdout captures>
    perf-data/px13_build_tma.sh && perf-data/harness/gpulease px13tma /tmp/px13tma
