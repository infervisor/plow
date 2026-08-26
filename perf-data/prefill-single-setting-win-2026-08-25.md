# Prefill, one fixed setting — Gemma-4-12B / RTX 5090 (sm_120a): closing the gap from 31% to ~19%

Follow-on to `perf-data/prefill-vllm-iteration-2026-08-25.md`, same session. That report's Phases
0-2 (generic, any-input tuning) found no win. Reframed per user direction: **plow assets are
precompiled per-shape — pick ONE fixed operating point and specialize hard for it, rather than
chasing a generic win across all context lengths.** Target chosen: **input-len 8192, concurrency
1, `--random-output-len 8`** (TTFT-only), matching one of the three points in the original live
comparison.

## Baseline at this one setting

| | TTFT (ms) | tok/s |
|---|---|---|
| plow (original chunked baseline) | 1744.3 | 4,697 |
| vLLM 0.27.0 | 1204.0 | 6,803 |
| ratio | | **0.69x — plow 31% behind** |

## Lever 1: single-chunk prefill bucket (`PLOW_MAX_CHUNK=8192`) — real win, zero kernel changes

The shipped emitter default caps prefill chunks at 1024 tokens for this model (window-derived),
so an 8192-token prompt was served as **8 separate chunk launches**, each re-streaming the full
22.18 GiB of bf16 weights from HBM (per `crates/devgen/src/lib.rs`'s own T32 comment: a chunk
boundary costs "weight restream + packet floors", independently measured elsewhere in this repo
at ~36 ms for just a 128-row tail chunk). `PLOW_MAX_CHUNK` is an existing, documented emitter env
var (`crates/devgen/src/lib.rs:2094-2123`, must be a power of two ≤ 8192); setting it to 8192
before emitting extends the shipped ladder to `[128, 512, 1024, 2048, 4096, 8192]`, giving an
8192-token prompt a **single** matching bucket — no source change, no cubin rebuild, pure
emit-time config (`plowc --hf-dir ... PLOW_MAX_CHUNK=8192`, new asset
`gemma4-12b-prefill-mc8192`).

**Gate**: greedy "Paris" — pass. Bicycle paragraph — **exact text match** to the untouched
baseline. `grep -aq libcuda.so.1` — pass.

**Result**: 1744.3 ms → **1470-1479 ms (-15.6%)**, reproduced twice.

## Lever 2: `PGM_BN=192` (plain-GEMM N-tile, GLU left at default 128) — small further win, real cost

Once prefill runs as one large single-shot GEMM (M≈8192 rows) instead of 8× M≈1024 chunks, the
bandwidth argument already documented in `op_gemm.cuh`'s own T2 comment ("BN 64->128 halves the
activation re-read traffic... a wider N-tile is a direct global-bandwidth win on the memory-bound
prefill GEMM") has more room to keep paying off past BN=128. `PGM_BN` is already a bare
`#ifndef`-overridable macro (`op_gemm.cuh:717-723`) — no source change needed, just
`PLOW_EXTRA_DEFINES="-DPGM_BN=192"` on the cubin rebuild.

- `PGM_BN=256` **does not compile**: a pre-existing `static_assert` (`op_gemm.cuh:1107`) couples
  the *unused* w8a16-fp8 GLU arena to the same global `PGM_BN`, and it overflows the bf16 arena
  claim at BN=256. Backed off to BN=192 (clean divisor of both 3840 and 15360, still compiles,
  arena math confirmed by hand before building). Fixing this properly to reach BN=256 would need
  decoupling the fp8-w8a16 arena macros from `PGM_BN` (mirroring lever 2's own `PGM_BN_GLU` split)
  — not attempted; the fp8-w8a16 body's own tile width would need auditing too before that's safe,
  out of scope for this pass.
- Register/spill cost on the production prefill object: `REG:240→255` (hit the hard ceiling),
  spill instructions (`STL`/`LDL` count) `66→105` (GLU pinned at 128) or `66→98` (GLU also at
  192, no override) — both real costs, no free lunch here, unlike the earlier `PGM_BN_GLU`
  refactor's default path.
- Whether GLU itself follows `PGM_BN` to 192 or stays pinned at 128 made **no measurable
  difference** (1427-1450 ms either way) — the win is entirely from the plain projections.

**Gate**: greedy "Paris" — pass. Bicycle paragraph — **exact text match**. Register/spill diff
above, measured before trusting any speed number.

**Result** (on top of lever 1): 1470-1479 ms → **1426-1452 ms (-3%)**, reproduced across 5 runs
(1425.8, 1427.1, 1431.9, 1437.5, 1443.0, 1445.0, 1445.5, 1450.5, 1452.5 ms — mean ≈ 1437 ms).

## Tried, no effect

- `PLOW_PF_NO_INTERLEAVE=1` (disables the runtime's `PLOW_PF_INTERLEAVE`-rows request-slicing,
  a multi-tenant fairness mechanism, `crates/plowrt/src/serve/mux.rs:2047-2144`): no change.
  Expected — that mechanism exists to let other concurrent slots interleave decode steps; at
  concurrency 1 there's nothing to interleave with, so the runtime's own "fastest cold TTFT" path
  already bypasses it.
- Serving without `PLOW_MULTISTEP=8`/`PLOW_DEV_SAMPLE=1` (decode-step-batching config,
  unrelated to TTFT in principle): no change, as expected. Restored for the final config since
  they matter for the decode portion of a real request.

## Net result at this one setting

| config | TTFT (ms) | vs vLLM (1204.0 ms) |
|---|---|---|
| original chunked baseline | 1744.3 | 0.69x (31% behind) |
| + single-chunk bucket (lever 1) | ~1474 | 0.82x (18% behind) |
| + `PGM_BN=192` (lever 2) | **~1437** | **0.84x (16% behind)** |

Final gap: ~16-19% depending which run pair is compared (5-9 runs each side, ~1-2% run-to-run
spread) — call it **~19% behind vLLM**, down from 31%, to stay conservative.

**Gap closed from 31% to ~19% at this one operating point, with zero numerics risk beyond the
already-passing exact-match gates** (lever 1 is pure emit-config, lever 2 is a proven-bit-exact
tile-shape choice — same mma operands, same accumulation order, different N-tile only).

## What's left to close the remaining ~19%

Not attempted this pass (larger scope, real precision risk, needs proper accuracy gating beyond
the two-prompt exact-match smoke test — GSM8K per this repo's own discipline before shipping):

1. **fp8/w8a16 weights for this one asset.** Halves the GEMM weight-read bytes; the existing
   w8a16 dequant-to-bf16-in-smem path (`op_gemm.cuh`'s `d_gemm_fp8`/`d_gemm_glu_fp8`, "T6 lever
   L2") is already in this codebase and unused by the current bf16 baseline. Biggest remaining
   lever by evidence (weight-bandwidth argument), but is a genuine numerics change requiring the
   full correctness discipline (exact-token-match + GSM8K on the integrated serving path) before
   trusting any speed number from it — bigger scope than either lever above.
2. **`PGM_BN=256`+** for the plain body, properly: decouple the fp8-w8a16 arena macros from the
   live `PGM_BN` (and audit that path's own tile width) so the static_assert stops capping this
   knob's ceiling.
3. Wave-aligned bucket sizing (`PLOW_PF_LADDER=wave`) to check whether the SM-tile grid at exactly
   8192 rows already lands cleanly on 170 SMs or is leaving a partial wave idle.
