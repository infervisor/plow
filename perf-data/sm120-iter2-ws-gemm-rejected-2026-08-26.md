# Iteration 2 — REJECTED: PX-22 warp-specialized w8a8 GEMM is bit-exact but catastrophically slow outside its own microbenchmark

Iteration:      2 (`/root/.claude/plans/glimmering-soaring-stream.md`)
Commit before change: `f11be6e` (Iteration 1)
Hypothesis:     `runtime/bench/nvidia/px22_ws_stage_bench.cu`'s producer/consumer warp
                specialization (1.144x isolated, bit-exact, 0 spills, `perf-data/px22-warp-
                specialized-staging.md`) integrates cleanly into the production `d_gemm_w8a8`
                body and wins end-to-end prefill.
Expected mechanism: 4 producer warps (4-7) issue `cp.async`, 4 consumer warps (0-3, 2x2 grid)
                run `mma` continuously with per-slot `mbarrier` handshakes instead of the
                shipped body's two per-K-tile `__syncthreads`, removing the ~466 cyc/K-tile of
                exposed staging the barrier serializes.
Expected maximum end-to-end benefit: not reached — see Decision.

## Files changed (implemented, tested, then fully reverted)

- `runtime/nvidia/op_gemm.cuh` — added `d_gemm_w8a8_ws`, a direct port of
  `px22_ws_stage_bench.cu`'s `body_ws<PGM_STAGES,4,4,2,2,false,0,0>` using production's exact
  tile constants (`PGM_BM=PGM_BN=128`, `PGM_BK8=64`, `PGM_STAGES=3`) and calling convention
  (`ascale`/`wscale`/`a_row0`/`slice`/`nblk`/`arena`), gated `#if PGM_W8A8_WS` (default 0,
  byte-identical when off — verified, see below).
- `runtime/nvidia/interp_sm120.cu` — dispatched it in place of `d_gemm_w8a8` under the same
  guard, at the existing `PLOW_DOP_GEMM_FP8` call site (only during the first correctness pass;
  reverted before the deeper investigation below, which used a standalone probe instead).

**Both fully reverted.** `git status --short` is clean except this report.

## Static verification (all passed, before any GPU run)

- WS=0 (default): rebuilt cubins, got a **byte-identical** object to the exact cubin already
  used in `prefill-beats-vllm-w8a8-2026-08-25.md`'s win (`sha256:a87ca7eb.../d7b3e784...`,
  matching `/workspace/plow-work/cubin-w8a8/` verbatim).
- WS=1 (via `PLOW_EXTRA_DEFINES`, since `PGM_W8A8_WS` is a plain `#ifndef` macro like `PGM_BN`,
  not a registered CMake option — first attempt silently no-op'd by passing `-DPGM_W8A8_WS=1`
  straight to `cmake`, which isn't a recognized option; caught by checking the register
  count/hash didn't change, then fixed): compiled cleanly, **242 registers / 1024 B stack / 0
  spills — identical to WS=0** (matches px22's own Result 5: "below the object's existing
  register allocation, no separate object needed"), `smem_pf` unchanged at 81664 B (the
  megakernel's smem union is set by flash-attention's much larger claim, not GEMM's).

## Correctness: initial full-stack test looked like a hang; standalone repro proved it isn't

Served the real `assets/gemma4-12b-prefill-w8a8-mc8192` asset with the WS=1 prefill cubin. Load
was clean (`smem_pf=81664`, no refusal). A short greedy prompt (8 tokens) did not return inside a
60s timeout; `nvidia-smi` showed 100% GPU util, 88% host CPU, zero forward progress. This read as
a deadlock and was reported as one in this file's first draft.

**It is not a deadlock — it terminates, just far slower than expected.** Built a standalone
repro (`runtime/nvidia/experiments/fp8_gemm_w8a8_probe.cu`'s own pattern: a single, uncooperative,
un-repeated kernel launch calling the production GEMM body directly with `blockIdx.x`/`gridDim.x`
as `slice`/`nblk`, bypassing the interpreter's gates/counters/cooperative-launch machinery
entirely) with a numeric oracle (small-integer e4m3 operands, exact in f32, so the kernel's output
must bit-match a CPU f64 reference — the same trick the in-tree probe uses). Bisected shape and
grid size:

| M | N | K | grid | tiles | wall time | result |
|---|---|---|---|---|---|---|
| 128 | 128 | 64 | 188 | 1 | <1s | PASS, 0 mismatches |
| 128 | 128 | 3840 | 188 | 1 (60 k-steps) | <1s | PASS, 0 mismatches |
| 256 | 256 | 64 | 1 | 4 | <1s | PASS, 0 mismatches |
| 256 | 256 | 3840 | 1 | 4 (60 k-steps ea.) | <1s | PASS, 0 mismatches |
| 128 | 15360 | 3840 | 1 | 120 (60 k-steps ea.) | **7.9s** | PASS, 0 mismatches |
| 128 | 15360 | 3840 | 120 | 120 (1 tile/block) | **7.8s** | PASS, 0 mismatches |
| 1024 | 3840 | 3840 | 170 | 240 | **13.7s** | PASS, 0 mismatches |
| 1024 | 15360 | 3840 | 170 | 960 | **53.7s** | PASS, 0 mismatches |

**Every configuration is bit-exact.** Wall time scales with total tile x k-step count, not with
grid width — `G=1` (one block, fully serial, zero cross-block interaction) and `G=120` (one tile
per block, maximum parallelism) on the *identical* 120-tile shape take the *same* 7.8-7.9s. That
rules out cross-block memory-bandwidth contention or scheduling unfairness between concurrent
blocks as the cause. The real production shape (M=1024, N=15360, K=3840, the `gate|up` GEMM at
the 8192-token bucket) computed correctly in 53.7s — against px22's own isolated bench measuring
**0.356ms** for a materially similar shape (M=1024, N=15360, K=3840, `ws4 NS=3`, 170 SMs). That is
roughly a **150,000x** slowdown, not a modest regression.

## Isolated / complete-object / end-to-end result

Not meaningfully measurable — a GEMM op that takes seconds instead of sub-millisecond is not a
candidate for any further benchmarking; the conclusion is already unambiguous from the oracle
timings above.

## Root-cause assessment (not fully chased down)

The bit-exact correctness rules out a logic bug in the handshake counts (a wrong expected-arrival
count would show up as reading stale/incomplete data, which the oracle would catch). The
proportional, grid-width-independent scaling with total tile x k-step count points at something
inherent to the spin-wait `mbarrier.try_wait.parity` polling loop itself under real (non-L2-
resident-by-construction) memory conditions — plausible candidates, none confirmed: the polling
loop's `bra`-back retry may consume the SM's instruction-issue slots aggressively enough to starve
the very producer warp it is waiting on (a starvation pattern a parking `__syncthreads()` cannot
exhibit, since it de-schedules the warp rather than re-issuing an instruction every cycle);
`px22_ws_stage_bench.cu`'s protocol deliberately keeps the 59 MB weight L2-resident across a
repeated `reps=20` launch loop ("this bench measures the ISSUE path"), while this probe's single,
cold launch pays genuine DRAM latency on every `cp.async` — if the spin-wait pattern's cost is
latency-sensitive in a way `__syncthreads()`-gated `cp.async.wait_group` is not, cold DRAM
latency could amplify rather than just add to the polling cost. Neither hypothesis was
instrumented or confirmed this iteration (`ncu` is still `ERR_NVGPUCTRPERM`-blocked;
`compute-sanitizer` is available — confirmed working this iteration — but its `synccheck`/
`racecheck` tools verify barrier *correctness*, not *timing*, so they would not have surfaced
this).

## Decision: REJECT

## Reason

This is exactly the failure mode this repo's own history repeatedly warns about and the mission
explicitly gates against: **"a microbenchmark win is not sufficient"** and **"an isolated GEMM win
can invert end to end"** (`px22-warp-specialized-staging.md` §Recommendation, citing PX-13's TMA
result). Here the inversion is not a modest one — it is catastrophic (5 orders of magnitude),
and it happens for a reason px22's own bench structurally could not observe: the bench's
repeated-launch, L2-resident protocol is explicitly *not* the real prefill memory regime. `PGM_W8A8_WS`
defaulted to 0 and was verified byte-identical to the shipped win when off, so production was
never at risk from the in-tree code; both files are fully reverted regardless, per the mission's
"remove only that experiment's production changes" rule.

This does **not** invalidate px22's own isolated finding on its own terms (1.144x, bit-exact, in
its own repeated-launch, L2-resident bench) — it invalidates the assumption that the isolated
number transfers to a real, cold, single-shot, DRAM-bound megakernel dispatch, which is exactly
what production prefill is.

## Commit

(this iteration's commit follows this report — reverted code + this report only, no production
behavior change)

## Next experiment

Given this iteration's finding, any future warp-specialization attempt on this codebase should
budget a *cold, single-launch, oracle-timed* test (exactly the harness this iteration built,
`fp8_gemm_w8a8_probe.cu`-style with `blockIdx.x`/`gridDim.x` slice/nblk and no repeated-launch
warmup) as a gate *before* the repeated-launch microbenchmark is trusted at all — the repeated
protocol should be treated as a ceiling estimate, not a projection. Re-scoping the remaining
ranked candidates (2-producer/6-consumer GEMM tiling, cuBLASDx, hd256/hd512 attention pipelining)
downward in priority until this class of risk is either explained or a cheap single-launch gate
exists to screen for it before spending more GPU time on mbarrier-based warp specialization.
