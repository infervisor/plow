# 11 — Tuning coverage by operator family

What the tuner can select among, per family, and what blocks each one. Derived
from the interpreters in this checkout, 2026-07-24. Where a claim here
disagrees with an older document, this one was checked against the source.

Three things decide whether a family is tunable at all:

1. **Does the opcode reach a distinct kernel?** Opcodes that fall through to one
   body are names, not choices; ranking them measures dispatch noise.
   `kernelcaps::probe::dispatch_arms` derives this from the built object.
2. **Is there a parameter to vary?** Either a `#ifndef`-guarded macro or a
   template argument at the dispatch site. A hard `#define` is not a knob —
   `-D` on it is a redefinition. `kernelcaps::sweep` classifies these.
3. **Is there a correctness oracle?** `tunedb` refuses to qualify a measurement
   whose oracle has not passed, so a family with no oracle cannot produce a
   selectable record however fast it runs.

---

## Coverage summary

| family | opcodes | distinct kernels | knob | oracle | status |
|---|---|---|---|---|---|
| Dense linear (tiled GEMM) | 19 | NV Ampere 1 body / 3 names; NV Hopper wgmma `d_gemm_sm90`; AMD 3 real | Ampere `PGM_BN`, stages; Hopper `PGM90_STAGES`/`PGM90_GLU_STAGES`/`PGM90_FP8_PROMOTE`; AMD `GM_BM`/`GM_BN` | AMD yes | **tunable** |
| Dense linear (GEMV) | — | distinct | `GV_*` unroll, `GV_MM_MAX` | AMD yes | tunable, NV oracle missing |
| MoE | 28 | all distinct (no fallthrough) | shares dense `PGM_*` | weak | blocked on oracle + shared tile |
| Attention | 7 | decode≡decode_fp8, prefill≡prefill_fp8 | NV template args; AMD `FA_*` | NV split across files | partial |
| MLA / latent | 4 | decode≡gather (GATHER flag) | GF ladder | AMD yes (DSA bench) | partial |
| DSA / sparse select | 3 | distinct | fixed `<128,32>` | **AMD yes, strongest** | tunable on AMD |
| Norm / elementwise | 13 | all distinct | few | NV oracle exists, untimed | low value |
| Token | 3 | distinct | none | — | low value |
| Collectives | 6 | 2 real, 2 stubs, 2 absent | `PLOW_XR_CUS` | AMD yes (2-GPU) | **needs N≥2 GPUs** |
| Recurrence (Mamba) | 1 | 1 | none | **none, never run on GPU** | **not tunable** |

---

## Aliasing — what must never be ranked against itself

Derived by `dispatch_arms`, confirmed by reading both interpreters.

**Dense GEMM, NVIDIA.** `GEMM=8`, `GEMM_MED=15`, `GEMM_SMALL=14` fall through to
one `d_gemm` (`interp_sm120.cu:524`). The tile is the object-wide `PGM_*` macro
triple, so all three are one kernel.

**fp8 GEMM, NVIDIA.** `GEMM_FP8=33`, `GEMM_MED_FP8=34`, `GEMM_SMALL_FP8=35` fall
through to one body in *both* build configurations — `d_gemm_w8a8` under
`PLOW_NV_W8A8=1` (`:541`), `d_gemm_fp8` under `=0` (`:566`). Different body per
flag, one body within any build. A measurement must be keyed on
`PLOW_NV_W8A8`, not on the opcode.

**Dense GEMM, NVIDIA Hopper (sm_90a).** `interp_sm90a.cu:16` hard-defines
`PLOW_NV_HOPPER=1`, so `op_gemm.cuh:512` delegates `d_gemm` to the wgmma body
`d_gemm_sm90` (and `d_gemm_glu`→`d_gemm_glu_sm90`, w8a8 likewise). The three
tile opcodes still fall through to one body, so the alias-collapse story is
unchanged — but the *tile* is `op_gemm_sm90.cuh`'s `PGM90_*` triple
(128×128×64 bf16, 128×128×128 e4m3), **not** the Ampere `PGM_*` triple, which
under a Hopper build feeds only the dead `#else` arm. The `kernelcaps` Sm90a
recipe carries `PLOW_NV_HOPPER=1` and points `tile_macros` at
`op_gemm_sm90.cuh`/`PGM90_*` accordingly, so `dense_gemm_inventory` reports the
tile the object actually executes. A measurement on sm_90a must be keyed to that
tile, and the body it records is `d_gemm_sm90`.

**Dense GEMM, AMD.** The same three opcodes reach three separately compiled
instantiations (`interp.hip:555/569/572`). Genuinely rankable. This asymmetry is
the single most important thing for a cross-vendor tuner to encode.

**Attention.** `FLASH_DECODE=12` and `FLASH_DECODE_FP8=38` are one templated
body differing in a `bool FP8KV`; likewise `FLASH_PREFILL=11` /
`FLASH_PREFILL_FP8=39`. On AMD they are mutually exclusive per object, so no
single build can benchmark both.

**MLA.** `FLASH_MLA_DECODE=50` and `FLASH_GATHER_DECODE=54` share a `case` group
and one function, differing in a `bool GATHER`. Treat as one kernel plus a shape
predicate.

**MoE.** No fallthrough aliases — every MoE opcode has its own `case`/`break` and
a uniquely named function. But two *body-level* aliases exist that the
fallthrough parser cannot see, because they are wrapper calls rather than shared
labels: AMD's `MOE_GROUP_GLU_FP8_BLK=48` is a `k`-loop over
`MOE_EXPERT_GLU_FP8_BLK=45`, and `=49` over `=46`, at default flags. They
diverge only under `-DPLOW_MOE_GROUP_FLAT=1` or `-DPLOW_MOE_MFMA=1`. **This is a
known limitation of derived alias detection** and is recorded rather than
silently missed.

---

## Sweepable parameters

Classified by `kernelcaps::sweep`, which reads the guard form.

| macro | vendor | status |
|---|---|---|
| `PGM_BN`, `PGM_STAGES`, `PGM_GLU_STAGES` | NVIDIA Ampere (sm_120a) | overridable |
| `PGM_BM`, `PGM_BK` | NVIDIA Ampere | **fixed** — `-D` collides |
| `PGM_BK8` | NVIDIA Ampere | **asserted** to 64 by `static_assert` |
| `PGM90_STAGES`, `PGM90_GLU_STAGES` | NVIDIA Hopper (sm_90a) | **bounded** — `#ifndef` but an arena `static_assert` caps them; a sweep may lower, not freely raise |
| `PGM90_FP8_PROMOTE` | NVIDIA Hopper | overridable — free 0/1 toggle for the two-level fp8 shadow accumulator |
| `PGM90_BM`, `PGM90_BN`, `PGM90_BK`, `PGM90_BK8` | NVIDIA Hopper | **fixed** — pinned to the wgmma m64n128 / 128 B swizzle shape |
| `GM_BM`, `GM_BN` | AMD | overridable — `build_gfx950_qwen.sh:29` ships `-DGM_BM=192` |
| `GM_BK` | AMD | **overridable** — `#ifndef`-guarded since the CDNA3 port; on a 64 KiB-LDS part it is the only axis that shrinks the stage without moving `GM_BN`, which fused GLU pins |
| `GV_UNROLL*`, `GV_MM_MAX`, `GV_UN16/32` | both | overridable, documented "for autotune" |
| `FA_DC`, `FA_DBUF`, `FA_BKV_D128` | AMD | overridable |
| `PLOW_NV_FA_GF`, `..._GF_FULL` | NVIDIA | overridable, `#error`-checked to {1,2,4,8} |
| `PLOW_NV_FA_HD` | NVIDIA | **fixed** |
| `PLOW_MOE_GROUP_FLAT`, `PLOW_MOE_MFMA` | AMD | overridable |
| `MAMBA_MAX_DSTATE` | NVIDIA | overridable, but a correctness bound not a tile |

The M axis is sweepable on AMD and not on NVIDIA. The K axis is sweepable on
**AMD only** — it was sweepable on neither until the CDNA3 port made `GM_BK`
`#ifndef`-guarded, because a 64 KiB workgroup cannot hold the stage at the
CDNA4 tile and `BN` is pinned by the fused-GLU `SN == 2` assert. **NVIDIA attention tiles are not macros at all** — `BQ`/`BKV` are
template literals at five dispatch sites (`d_flash_prefill_mux<256,64,32>`,
`<512,32,16>`), so sweeping them means editing those call sites. The probe reads
them out as `DispatchArm::specializations`.

Two couplings a campaign must respect:

- `GV_MM_MAX` moves the register ceiling of **every arm** in the object, because
  the interpreter inlines all of them. It is not independently tunable per op.
- MoE grouped GEMM shares the dense `PGM_*` triple, so sweeping it moves dense
  prefill too. They cannot be tuned independently without new macros.
- `PGM_BM` is packet-layout-visible: `MOE_ALIGN_GEMMA_PF=74` pads the routing
  histogram to it and the host emitter sizes buffers to match.

---

## Declared but not executable

`GEMM_NORM=9` — no arm on either backend. AMD keeps a dead wrapper
`exec_gemm_norm` (`interp.hip:381`) that nothing calls; the rationale at `:59`
records it as a measured loss (62–113 TF/s fused vs 194+ unfused). Exclude.

`XREDUCESCATTER=25`, `XALLGATHER=26` — reserved numbers, referenced nowhere
outside the enum. The decomposition exists only *inside*
`d_xreduce_twoshot_mega` as two phases, not as packets.

`FLASH_MLA_PREFILL=51`, `FLASH_GATHER_PREFILL=55` — no arm anywhere.

**Dispatch-present but kernel-absent:** AMD's `XFLASH_MERGE=27` and
`XARGMAX_FIN=28` have live `case` labels with empty bodies. These are the
dangerous ones — they would benchmark at ~0 ns and win. `ProbedObject::executes`
excludes them; `::stubs` lists them.

Asymmetries worth recording: `ATTN_SELECT=53` and `O_UV_FOLD=52` exist on AMD
only. `GEMV_ARGMAX=80`, `GEMV_SZ=78`, `GEMV_GLU_SZ=79` are NVIDIA only.
`ROW_RMS=2` is AMD only. NVIDIA has **no collective support at all**. The MoE
opcode sets are entirely disjoint between vendors: NVIDIA dispatches 61–77/81/82,
AMD dispatches 40–49/56, with no overlap.

---

## Oracles — the real bottleneck

`tunedb` will not qualify a measurement without a passing oracle, so this table
decides what can actually be tuned today.

**Harnesses that time *and* validate in one run — all four are AMD:**

| harness | op | oracle |
|---|---|---|
| `runtime/bench/interp_gemm_bench.c` | GEMM **through the interpreter** | CPU dot, `rel > 0.02` fails |
| `runtime/bench/qwen_interp_bench.c` | GEMM, Qwen shapes | same |
| `runtime/bench/dsa_gather_bench.c` | DSA score/select/gather | three independent CPU refs, incl. **exact** top-k set equality |
| `runtime/tests/tp_allreduce_bench.c` | all-reduce | bit-exact sum, 2 GPUs |

`interp_gemm_bench.c` is the template worth generalizing. Its header makes the
argument the whole tuning system rests on: a standalone GEMM and the same GEMM
inlined into the interpreter get different register allocations, measured 212
vs 770 TF/s for the same tile family. It builds a one-instruction `PlowProgram`
and dispatches it, so it inherits both interpreter fidelity and an oracle.
Parameterizing that packet gives a general per-op driver.

**On NVIDIA, timing and oracles exist for the same ops but in separate files.**
`sm120_interp_op_test.cu` validates against an inline f32 CPU oracle and includes
the same `op_*.cuh` headers the megakernel uses — but has no `cudaEvent` calls.
The `*_bw_*.cu` timers have no oracle. Merging the two is the shortest path to a
qualifying NVIDIA measurement.

Not qualifying: `runtime/ubench/` (no oracle, fixed shapes, AMD-only, and its
`BenchParams` case-format in `bench_cu.h:77` is declared but never implemented);
`moe_group_bench.cu` (compares fp8 against bf16 on the same device — a
self-consistency delta, not an independent reference); `block_run.rs`
(block-level, finiteness only, says so at `:11`).

---

## Per-family verdicts

**Dense GEMM — tunable now.** Aliasing understood, AMD has a knob and an oracle.
NVIDIA needs the oracle merge before it can qualify anything. On sm_90a the
probed body is the wgmma `d_gemm_sm90` at 128×128×64, not the Ampere
`d_gemm` — `dense_gemm_inventory` reports it correctly, and its only free knob
is `PGM90_FP8_PROMOTE` (the depth knobs are arena-bounded).

**GEMV — tunable, high value.** `GV_MM_MAX` and the unroll ladder are explicitly
"build-overridable for autotune", and decode is where the dispatch floor
dominates. Watch the global register coupling.

**MoE — blocked, despite being the largest family.** 28 opcodes, all distinct,
but the grouped GEMMs share the dense tile so they cannot be swept
independently, and there is no independent oracle. Highest-value unblock:
an oracle, then per-MoE tile macros. On sm_90a the grouped-GEMM prefill bodies
(`d_moe_group_glu_gemma_pf`/`d_moe_group_down_gemma_pf`) are compiled from their
wgmma definitions in op_moe_sm90.cuh under `PLOW_NV_HOPPER`, but the tuner
verdict is unchanged: no oracle means no qualifiable measurement, Hopper or
Ampere.

**Attention / MLA / DSA — partial.** DSA has the strongest oracle in the tree.
NVIDIA attention tiles need call-site edits rather than `-D`, so a sweep there is
a code-generation problem, not a parameter search. On sm_90a the hd256 prefill
arm runs a wgmma body (`d_flash_prefill_sm90`, op_attention_sm90.cuh) whose
`BQ`/`BKV` are template literals like the Ampere path — same code-generation
problem, plus the same missing NVIDIA timing+oracle merge.

**Collectives — park until N≥2 GPUs.** Two of six opcodes have no implementation,
two are stubs, NVIDIA has none. At N=1 the gate self-satisfies and the reduce
degenerates to a copy, so a single-GPU number would be a memcpy timing. The one
genuine missing knob is a one-shot↔two-shot crossover threshold; both bodies are
documented bit-identical, so its oracle is free.

**Mamba — not tunable, and not qualifiable.** No tiles, no chunking, single-CU by
construction (`op_mamba.cuh:73`, `if (slice != 0) return;`). Its only macro is a
correctness bound. The header carries an explicit "UNVERIFIED ON GPU" banner —
it has only been compiled, never executed — and there is no GPU oracle. It needs
implementation before it needs a tuner. Note it is included **unconditionally**
in every sm_120 object (`interp_sm120.cu:53`), so it costs registers in builds
that never run it; that cost is worth measuring even though the op is not.
