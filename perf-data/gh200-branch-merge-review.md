# gh200-tma-gemm — merge review (2026-08-07)

High-effort review of the whole branch vs `origin/main` (66 commits, ~8.5k
insertions): TMA/ws384 prefill campaign + multi-model verification +
multi-instance design + GPU error-handling overhaul. 12 candidates verified,
10 CONFIRMED, 2 refuted. Five host-side findings are FIXED (commit 8648a38);
five kernel-side ones are open, all knob-gated or build-variant-gated.

## Fixed (8648a38)

| # | Site | Defect |
|---|------|--------|
| 1 | `exec/gpu.rs` Drop | `seg_graphs` + SegPf's three modules never released. Engines share one backend whose primary-ctx retain outlives them, so the backend-drop drain never runs while serving — each S1 swap pinned 3 cubins and leaked one ~480-node `CUgraphExec` per (bucket, slot) under `PLOW_PF_SEG_GRAPH=1`, to process exit. |
| 2 | `exec/gpu.rs` EQSMEM | `grid_gemm` re-queried at `BLOCK`(256) while the ws384 object launches at 384 → over-reported grid → `COOPERATIVE_LAUNCH_TOO_LARGE` per chunk, or a counter-wait hang under `PLOW_PF_SEG_NONCOOP=1`. |
| 3 | `exec/gpu.rs` `prefill_batched` | `PLOW_PF_BATCH` + `PLOW_PF_SEG_DIR` launched the plain `_pf` object (gq window = segment 0), silently skipping every later segment: stale KV, garbage logits, successful return. Now refused loudly. |
| 4 | `serve/mux.rs` | CUDA first-token TTFT recorded `add(0)` → dump printed a confident 0.000 ms, real cost hidden in UNACCOUNTED. |
| 5 | `exec/gpu.rs` seg loop | `PLOW_PF_SEG_FATONLY`/`_NONCOOP` read via `std::env::var` inside the ~480-iteration launch loop (~960 env-lock + String allocs per chunk). |

## Kernel-side: 4 of 5 FIXED in 132cc07

Fixed and verified on GH200 with a full canonical cubin rebuild (gate output
byte-identical, 70.6 ms @1k / 176.8 ms @4k vs the 70.7/176.6 baseline):
findings 1, 2, 4 and 5 below. Only finding 3 remains open. Details of each
fix are in the commit; the original analysis is kept below as the record.

**Finding 2's fix carries a lesson worth keeping**: the first attempt read the
capability global as `plow_fa_hd256`, but `PLOW_SYM()` suffixes every global
per object (`plow_fa_hd256_pffa`), and a missing symbol reads as
"unconstrained" — so the guard silently did nothing and the mis-built object
served happily. Caught only because the drill asserted the REFUSAL, not just
the happy path. Any future capability check must be drilled against a
deliberately mis-built object.

## Original analysis (finding 3 still open)

Ordered by severity. None fires on the shipping serve configuration used for
the campaign (`PLOW_PF_SEG_DIR` + `PURE=fp8` + `FA512=all` + `SEG_GRAPH=1`,
batch-1 bundles), which is why the gates stayed green throughout.

1. **`op_norm.cuh:58/325` — decode numerics change at batch >= 8.** The T17
   warp-per-row path gates on `rows >= PLOW_NV_WARPS(8) && feat%8==0`, a pure
   function of packet `i[0]`; decode passes `i[0] = dbatch`, and
   `PLOW_DECODE_BATCH=8` is a shipped baseline. The new per-lane partition +
   `warp_sum32` reduces in a different order than the legacy per-thread
   RN_VEC + block tree, so f32 non-associativity moves the last ulp. The
   header claims decode keeps the legacy path — it does not. Golden-token and
   byte-stability gates for batched decode would break on rebuild, with no
   opt-out flag. **Verified unaffected here only because the GH200 bundles are
   batch=1.** Fix: add a prefill/decode discriminator (or gate on `rows` far
   above any decode batch), then re-run the batch-8 golden gate.
2. **`devblob.rs:573` — no capability check on the pffa cubin.**
   `PLOW_PF_SEG_FA512=all` routes hd256 segments to `interp_sm90a_pffa`
   purely on serve-time env; against a cubin built with the default
   `PLOW_BUILD_FA_HD256=0` the hd256 arm is compiled out and the dispatch
   falls to a bare `__trap()` → LAUNCH_FAILED → (with this branch) poisoned
   context, engine Dead, all requests 503. The sibling pfgemm object queries
   `plow_block_pfgemm` three lines earlier; the mirror for fa512 is absent.
   Fix: emit an `plow_fa_hd256` global in the cubin and refuse a mismatch at
   load, the way the loader already refuses a missing file.
3. **`devblob.rs:642` — `check_coarse_single_segment` relaxed to
   `e.seg >= l2_domains`** leaning on a parse-time gate that on NVIDIA is
   satisfiable by env var alone (`L2_DISPATCH_SYM` attestation exists only on
   the AMD path). An L2-placed blob + standard cubin + stale env now loads
   where it used to error loudly, and the `#else` interpreter branch drains
   only domain 0 → unsatisfied gates, hang or garbage. Fix: add the CUDA-side
   attestation symbol, or restore the strict check until one exists.
4. **`op_gemm_sm90.cuh:1237` — OOB store in the SMEPI epilogue.**
   `PGM90_WS384_SMEPI=1` stores full 256-column rows guarded only by
   `gr < m`, with no column guard against `n` (the non-SMEPI epilogue checks
   `cc + 1 < n`): at `n % 256 == 128` it writes 256 B past each row into the
   adjacent tensor, plus OOB `wscale` reads. Dead today (knob defaults 0, the
   build script never sets it) and the commit that added it records the
   epilogue as measurement-REFUTED — so the cheapest fix is deletion.
5. **`op_attention.cuh:3289` — hd512 TMA staging is unreachable dead code.**
   `FA_PX4_ELIGIBLE(512)` is always true with the default knobs, so the
   `USE_SERIAL` arm is always taken — and only the `!USE_SERIAL` arm was given
   the new `mapkv` parameter. devgen mints the KV tensor map and the interp
   passes it, but it arrives null and hd512 silently stays on cp.async: any
   perf attributed to T33 at hd512 was not measured on the TMA path. Enabling
   the documented `-DPLOW_NV_FA512_WG=1 -DPLOW_NV_FA_PX4=0` combination
   reaches a `for (hb = 0; hb < BKV/32; hb++)` that runs zero iterations after
   the barrier is armed → `waitKV()` spins forever.

## Refuted

- FA BKV hang at hd512 via the TMA path — unreachable (see open #5).
- Dropped stream-entry flag check — traps loudly rather than hanging.

## Pre-existing, not from this branch

`devgen`'s `the_prefill_ladder_leaves_the_decode_program_byte_identical` fails
roughly 1 run in 7 of the full suite: sibling tests mutate process-global env
(`PLOW_GLM_WGFIT=0`, `PLOW_GLM_GF`) that leaks into parallel tests.
`crates/devgen/src/mla.rs` is untouched by this branch — the race is on `main`
and can redden CI on any branch.

## Merge state

66 ahead / 0 behind `origin/main` (clean fast-forward), tree clean, 109
workspace suites green, `--features cuda` and `--features hsa` both build,
GPU gate coherent, 176.6-177.2 ms @4k / 70.7 ms @1k median — unchanged from
the campaign record (176.8 / 70.4).
