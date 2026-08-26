# Iteration 2 — REJECTED: PX-22 warp-specialized w8a8 GEMM is bit-exact and fast standalone, but hangs when dispatched from the real interpreter

**Correction, same day**: this report originally concluded the kernel itself was
"~150,000x slower cold" than px22's isolated bench. That conclusion was **wrong** — it was a
measurement-harness bug (a single-threaded O(M·N·K) CPU reference loop dominating wall-clock
time in my own probe, not GPU kernel time). Re-verified with a proper GPU-side comparison below:
the kernel is fast and bit-exact. The real, still-open finding is narrower and different: the
kernel hangs specifically when dispatched from inside the production interpreter megakernel,
not in any standalone launch. Keeping the wrong-conclusion history here rather than silently
editing it away, since a future session should know this trap exists in probe design too.

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

## Files changed (implemented, tested, then fully reverted — twice, see timeline)

- `runtime/nvidia/op_gemm.cuh` — added `d_gemm_w8a8_ws`, a direct port of
  `px22_ws_stage_bench.cu`'s `body_ws<PGM_STAGES,4,4,2,2,false,0,0>` using production's exact
  tile constants (`PGM_BM=PGM_BN=128`, `PGM_BK8=64`, `PGM_STAGES=3`) and calling convention
  (`ascale`/`wscale`/`a_row0`/`slice`/`nblk`/`arena`), gated `#if PGM_W8A8_WS` (default 0,
  byte-identical when off — verified).
- `runtime/nvidia/interp_sm120.cu` — dispatched it in place of `d_gemm_w8a8` under the same
  guard, at the existing `PLOW_DOP_GEMM_FP8` call site.

**Both fully reverted; working tree is clean.** (Re-added and re-reverted a second time during
the correction pass below — same diff each time, not a new experiment.)

## Static verification (all passed, before any GPU run)

- WS=0 (default): rebuilt cubins, byte-identical to the exact cubin already used in
  `prefill-beats-vllm-w8a8-2026-08-25.md`'s win (`sha256:a87ca7eb.../d7b3e784...`).
- WS=1 (via `PLOW_EXTRA_DEFINES`, since `PGM_W8A8_WS` is a plain `#ifndef` macro like `PGM_BN`,
  not a registered CMake option): compiled cleanly, **242 registers / 1024 B stack / 0 spills —
  identical to WS=0**, `smem_pf` unchanged at 81664 B.

## Correctness and timing: standalone kernel is correct and fast — the earlier "150,000x slower" finding was a probe bug

First pass: served the real w8a8 asset with the WS=1 prefill cubin. A short prompt didn't return
in 60s; 100% GPU util, no progress. Read as a deadlock. Built a standalone probe (mirroring
`runtime/nvidia/experiments/fp8_gemm_w8a8_probe.cu`: a single kernel launch calling the
production GEMM body directly with `blockIdx.x`/`gridDim.x` as `slice`/`nblk`, no interpreter
involved) with a numeric oracle: e4m3 small-integer operands (exact in f32) checked against a
CPU **f64 reference computed by a triple-nested `for m: for n: for k` loop**. This "passed" at
small shapes and got dramatically slower at large ones (up to 53.7s at the real M=1024, N=15360,
K=3840 `gate|up` shape), which the first draft of this report attributed to the GPU kernel.

**That CPU reference loop is O(M·N·K) — 6.04x10^10 scalar double-precision multiply-adds at the
full shape, single-threaded.** That is the actual cost that was measured, not GPU time. Re-ran
with a GPU-only comparison (`cudaEventElapsedTime`-equivalent wall clock around
`cudaDeviceSynchronize()` only, no CPU reference computation, output compared via device-to-host
`memcmp` against the *plain* `d_gemm_w8a8`'s own output instead of a CPU reference):

```
M=1024 N=15360 K=3840 G=170
plain: 0.0005 s  ok
ws:    0.0005 s  ok
bit-exact match: PASS (0 / 15728640 elements differ)
```

**The kernel is fast (matches px22's own 0.356ms bench number closely) and bit-exact**, in a
plain (non-cooperative) single-shot launch, at the exact production shape. A repeated-launch
warm-vs-cold check (one untimed "cold" launch, then 5 more) showed **every** launch — including
the first — at 0.0005s: no cold-DRAM effect at all, contradicting the first draft's "spin-wait
under cold DRAM" theory as well as its headline number.

## The real, still-open finding: it hangs specifically through the interpreter

With the standalone kernel proven correct and fast, re-tested the *actual* dispatch path: rebuilt
the real interpreter object with `-DPGM_W8A8_WS=1` (wired into `interp_sm120.cu`'s
`PLOW_DOP_GEMM_FP8` case, not a probe), served the same w8a8 asset, sent the same short prompt
with a 90s timeout this time. **It hung again — genuinely, not a measurement artifact**: `curl`
returned nothing after 90.0s, `nvidia-smi` showed 100% GPU util / 121.7 W / 90% host CPU sustained
for 110+ seconds with zero output, and this path has no CPU-side O(M·N·K) computation anywhere to
blame. Killed the server; GPU briefly showed a stale 100%-util reading with no processes (same
harmless artifact as the first pass — a fresh trivial kernel launch immediately after confirmed
the device itself is healthy, util dropped to 1%).

**So the bug is real, reproducible, and specifically about being dispatched from inside the
persistent, cooperative, multi-op interpreter megakernel** — not in the GEMM body's algorithm,
not shape-dependent, not a cold-memory effect. Not root-caused this iteration. Candidates,
untested: the interpreter's per-op dispatch loop (`interp_sm120.cu:2011-2134`) uses `__syncthreads()`
at packet-claim and gate boundaries and `cudaLaunchCooperativeKernel` for the whole grid, none of
which the standalone probe exercises; a real prefill program calls `d_gemm_w8a8_ws` many times
(once per GEMM op per layer, 48 layers), so state left in the reused shared-memory arena by one
call could interact with the next call's fresh `pgm_mbar_init` in a way a single-call probe
cannot expose; or an interaction with `in->blocks`-based partial-grid participation
(`interp_sm120.cu:2088-2092`: "a block not in the packet's set simply has no stream entry for it")
that the always-full-grid standalone probe never exercises either.

## Isolated / complete-object / end-to-end result

Isolated (standalone, single-call): fast and bit-exact, effectively confirming px22's own number
transfers cleanly outside the interpreter. Complete-object / end-to-end: **not reachable** — the
real dispatch path hangs before any request completes.

## Decision: REJECT

## Reason

The isolated GEMM body is sound — this is not the "microbenchmark win inverts end-to-end"
pattern the first draft claimed. It is a narrower, still-serious integration bug: something about
warp-specialized `mbarrier` synchronization is incompatible with this specific persistent
cooperative megakernel's dispatch loop, and it manifests as a genuine hang (not slowness) only in
that context. `PGM_W8A8_WS` defaulted to 0 and was verified byte-identical to the shipped win when
off, so production was never at risk; both files are fully reverted per the mission's "remove only
that experiment's production changes" rule. Root-causing the interpreter-specific interaction is
real, open-ended debugging work (needs either a working profiler — `ncu` is still blocked — or a
minimal single-op interpreter program to bisect against, neither built this iteration) and was not
completed within this iteration's budget.

## Commit

(this iteration's commit follows this report — reverted code + this corrected report, no
production behavior change)

## Next experiment

If this line of work continues: build a minimal ONE-packet, ONE-layer prefill program (rather
than the full 48-layer Gemma-4 program) to get a fast interpreter-level bisection loop, and check
whether the hang reproduces with a single `d_gemm_w8a8_ws` call through the interpreter's gate/
dispatch machinery — that would separate "any interpreter dispatch breaks it" from "repeated
calls across the reused arena break it." Until that is understood, treat any `mbarrier`-based
technique proposed for Iterations 3, 5, or 6 (all of which call for per-slot `mbarrier`
synchronization) as carrying this same undiagnosed risk, and gate each with a full live
interpreter dispatch test (not just a standalone probe) before trusting a microbenchmark number.
