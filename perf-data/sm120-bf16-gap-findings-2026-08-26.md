# Why plow doesn't beat vLLM at matched (bf16) precision, and what was investigated as a fix

Consolidated findings from this session's diagnostic pass. Two parts: (1) the root-cause
diagnosis of the bf16-vs-bf16 gap, and (2) the segmented-GEMM-occupancy lever investigated as a
possible fix and why it isn't recommended.

## Part 1 — the diagnosis

At the one directly-comparable fixed setting (input-len 8192, concurrency 1), **plow bf16 trails
vLLM bf16 by ~18%** (plow ~1437ms vs vLLM ~1221ms TTFT). This is a *different* result from the
w8a8 win (plow fp8 beats vLLM bf16 by 27%, confirmed at 3 context lengths in
`perf-data/sm120-prefill-w8a8-multictx-2026-08-26.md`) — w8a8 uses a faster instruction
(`mma.m16n8k32.e4m3`, 2x bf16's tensor throughput) to overcome an efficiency gap that still exists
underneath it. The bf16-vs-bf16 comparison isolates that gap.

**Two measured, quantified root causes** (full detail: `perf-data/prefill-bf16-gap-attribution-
2026-08-26.md`, `perf-data/prefill-kernel-sweep-2026-08-26.md` — full-grid isolated microbenches,
real Gemma-4-12B shapes, no `ncu` needed):

1. **GEMM runs at 66-71% of cuBLASLt** on identical shapes (gate|up 71.0%, down 66.7%, q_full
   67.0%, o_full 65.7%) — a real ~30-34% gap, the same magnitude the w8a8 body showed against
   cuBLASLt separately (PX-9).
2. **Flash-attention runs at only 34-44% of the raw bf16 mma hardware ceiling** — hd256 sliding
   (40/48 layers) at 34.1%, hd512 full (8/48 layers) at 43.9%. A *bigger* relative gap than GEMM's
   — flash-attention's harder dependency chain (running softmax stats, P·V using just-computed
   probabilities) makes that plausible.

**Why this isn't a hardware ceiling**: sm_120a (consumer/workstation Blackwell) has no TMEM and no
`tcgen05`/`wgmma` (Hopper/datacenter-Blackwell only). cuBLAS on *this* GPU is also restricted to
`mma.sync`-class kernels, the same instruction class plow uses, and FlashAttention-3/4 don't run
on sm_120a either. **The likely dominant factor is cuBLAS's autotuned kernel-selection maturity**
(tile/split-K/swizzle picked per shape from a huge NVIDIA-tuned library, built over years) **vs
plow's fixed, hand-picked tile** — an engineering-depth gap, not a silicon-capability gap.

**Cheap knobs are exhausted**: `PLOW_NV_FA256_BKV=64` fails to compile (real correctness
`static_assert` on the current softmax reduction); `PGM_STAGES=4` fails to load (dynamic smem
exceeds the device's 101,376 B opt-in cap at the current tile width).

### What this session tried against it, and why it's currently blocked

The one real attempt at closing the GEMM half: ported `px22_ws_stage_bench.cu`'s proven
producer/consumer warp specialization (1.144x isolated, bit-exact, 0 spills, 0 register cost to
the megakernel) into `d_gemm_w8a8`. **The isolated kernel body is correct and fast** — verified
directly (bit-exact `memcmp` against the plain body, 0.5ms at the real production shape, matching
px22's own number closely) — but it **hangs specifically when dispatched through the real
interpreter megakernel**, not in any standalone launch. Seven separate hypotheses tested and
ruled out (shape, grid width, cold DRAM, repeated calls across the reused arena, cooperative
launch mode, an explicit `cp.async` drain, many-calls-with-varying-shapes-interleaved-with-the-
plain-body) — every standalone reproduction is fine; only the real dispatch path hangs. Root
cause not found: both `ncu` (`ERR_NVGPUCTRPERM`) and `cuda-gdb`'s CUDA debug backend ("Could not
find CUDA Debugger back-end") are unavailable in this sandbox, closing off live warp-level
introspection. Full record: `perf-data/sm120-iter2-ws-gemm-rejected-2026-08-26.md`.

cuBLASDx (a separate, non-`mbarrier` approach) is blocked outright — not installed, not in
nixpkgs, needs an NVIDIA Developer Program login/EULA this environment can't reach autonomously.

## Part 2 — segmented GEMM occupancy (`PLOW_NV_SEG_GEMM`), investigated as an alternative

Question investigated: since the full prefill megakernel runs at only 1 block/SM, can a leaner,
GEMM-only segmented object run at 2 blocks/SM and close part of the gap, via a mechanism that
doesn't share Iteration 2's `mbarrier` hang risk?

**Mechanism — real, and structurally independent of the interpreter hang.** `PLOW_NV_SEG_GEMM`
compiles a genuinely separate, smaller `__global__` kernel (own smem arena, own
`__launch_bounds__(256,2)` register cap, `interp_sm120.cu:564-574,1926-1936`), launched
independently from the flash/general object, not `mbarrier`-based intra-kernel warp roles. The
host alternates separate launches across objects (`crates/plowrt/src/exec/gpu.rs:4463`), gated by
the same pre-existing, validated `__syncthreads()` counter protocol every dispatch path already
uses. This hang risk only attaches if `SEG_GEMM` is combined with a *separate* `SEG_WS`/
`SEG_WS384` warp-splitting variant, which is not required here.

**Already measured on the correct GPU (RTX 5090, sm_120a), already rejected, with numbers.**
`perf-data/px7-w8a8-ceiling.md`:

| shape | occ-2 vs occ-1 |
|---|---|
| `gate\|up` (GLU, ~2/3 of prefill GEMM FLOPs) | **1.000x** — register-limited to occ-1 regardless of the object split |
| `down` / `q_full` / `o_full` | 1.16-1.18x |

FLOP-weighted end-to-end: only **~1.05x** — and GEMM itself is only ~31% of prefill time
(flash-attention's quadratic term is 53%). PX-7's own conclusion, verbatim: *"Do not build Step 2
... Cost/benefit is nowhere near the counter-protocol risk."* Rough math: 31% × (1 − 1/1.05) ≈
**~1.5% of total prefill wall time** — well under the mission's 5%-faster gate on its own.

**Not just a flag flip on sm_120a — real missing infrastructure.** The host loader
(`gpu.rs:3874-3898`) hardcodes sm90a cubin filenames; no `build_sm120_cubin.sh` equivalent exists
for `_pfseg`/`_pfgemm`. More importantly, the packet emitter currently slices GEMM segments for
`n_cu` blocks, not `2*n_cu` — a second resident block per SM has no work until that's built
(`runtime/CMakeLists.txt:349-352`: "the harness does NOT yet drive it ... the Stage-3
prerequisite"). PX-3's own occ-2 result was a standalone bench, not end-to-end, and measured on
RTX PRO 6000 (188 SMs), not the RTX 5090 this campaign targets — its own report says "e2e status
— blocked ... the emitter investment is not justified by this cheap check."

**A premise this investigation corrected**: the working assumption going in — that shared memory,
dominated by flash-attention's arena, is what caps the full megakernel at 1 block/SM — is
contradicted by `perf-data/prefill-occupancy-handoff-2026-08-25.md`: the real cap on the *full
megakernel* is **register pressure** (`REG:255`, the union over every inlined opcode), not smem —
"even a 0-byte arena wouldn't raise occupancy past 1." This doesn't invalidate `SEG_GEMM` itself
(a genuinely separate, smaller object sidesteps the register union differently, and PX-7's occ-2
measurement already reflects the real register story directly), but the expected payoff is PX-7's
measured ~1.05x weighted, not a naive "2x occupancy ⇒ ~2x GEMM speedup" argument.

**Assessment**: real mechanism, correctly-targeted hardware already measured, explicitly rejected
for a quantified reason (~1.05x weighted / ~1.5% of total prefill time) against real,
currently-unbuilt engineering cost (new sm_120a cubin build path + emitter re-slice) and
unspecified "counter-protocol risk." Not recommended as a priority lever.

## Where this leaves the bf16 gap

Both investigated paths this session (warp-specialized GEMM, segmented-GEMM occupancy) are
blocked or not worth their cost. The highest-leverage remaining step, if the bf16 gap is to be
closed, is root-causing the interpreter hang from Part 1 — it blocks the larger warp-
specialization and attention-pipelining levers, not just the one GEMM attempt. That needs a
minimal synthetic-program harness (via `crates/packet/src/devbuild.rs`'s `Builder`, ~30-60 lines,
plus a new ~150-250 line GPU-launch harness modeled on `crates/plowrt/tests/cuda_gpu.rs` — no
reusable "small program → real GPU interpreter run" path exists today, confirmed by a dedicated
survey; estimated 3-6 hours) to bisect the interpreter's gate/counter dispatch machinery in
isolation from the full 48-layer program — the one code path nothing in this session's
investigation reached. Not built this session; recorded as the concrete next step if pursued.
