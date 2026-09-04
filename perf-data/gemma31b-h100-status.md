# Gemma-4-31B / H100 prefill baseline — status (resumable)

Tracks progress on the plan at `~/.claude/plans/abundant-roaming-squirrel.md`
(Gemma-4-31B prefill: vLLM baseline vs plow, Phase 1 — benchmark only, no
tuning). Update this file at every checkpoint; commit after each phase.

Box: H100 80GB HBM3 (single GPU, `nvidia-smi -L`), driver 595.91.07.
Fresh machine for plow: no nix, no cargo/rustc, no CUDA toolkit for
`shaswot`, no `/workspace` at session start (2026-09-04).

## Phase 0 — environment bring-up: IN PROGRESS

- [x] Install Nix (Determinate installer, multi-user daemon) — `nix (Determinate Nix 3.22.3) 2.35.2`.
      Must `source /nix/var/nix/profiles/default/etc/profile.d/nix-daemon.sh` each new shell.
- [x] `nix develop -c cargo build --release -p plowc` — built, `target/release/plowc` runs.
- [ ] `nix develop -c cargo build --release -p plowrt --features cuda` — running in background
- [x] ACL grant on `/opt/dlami/nvme/hf-cache/hub/models--google--gemma-4-31B-it`
      (`sudo setfacl -R -m u:shaswot:rX ...` on the model dir, plus
      `sudo setfacl -m u:shaswot:x /opt/dlami/nvme/hf-cache` for traversal —
      `hub/` itself is already `o+rx`). Verified readable (config.json + a
      safetensors shard). Checkpoint dir:
      `/opt/dlami/nvme/hf-cache/hub/models--google--gemma-4-31B-it/snapshots/842da3794eaa0b77d5f08bae87a17459d91ff475`
      (2 safetensors shards, standard HF layout).
- [x] Install CUDA toolkit ≤13.2 (cuda-keyring apt repo already configured) —
      `cuda-toolkit-13-2` (13.2.86) installed at `/usr/local/cuda-13.2`,
      `nvcc --version` confirms release 13.2, matches driver 595.91.07.
- [x] Build sm_90a cubins: `scripts/build_sm90a_cubin.sh` run OUTSIDE
      `nix develop` with `env -i PATH=/usr/local/cuda-13.2/bin:/usr/bin:/bin
      PLOW_NVCC=/usr/local/cuda-13.2/bin/nvcc`. Both objects built + kernel
      symbols verified present:
      `assets-run/gemma4-31b-bf16/interp_sm90a.cubin` (1.30 MB, decode) and
      `interp_sm90a_pf.cubin` (678 KB, prefill).
- [x] Emit plow bf16 asset:
      `PLOW_UNISEG=1 PLOW_NS_FULL_ABS=33 ./target/release/plowc --hf-dir <ckpt>
      --emit devblob --gpu h100 --arch sm_90a --max-ctx 8192
      --out assets-run/gemma4-31b-bf16` — succeeded: 60 layers, weights
      57.2 GiB, KV cache 2.19 GiB (matches `coldstart-plow-vs-vllm-gh200.md`
      §1b exactly). **IMPORTANT for benchmarking**: `weights.json`'s
      `"network"` field (the model-routing key clients must pass as
      `"model"`) is `842da3794eaa0b77d5f08bae87a17459d91ff475` (the HF
      snapshot dir's commit-hash basename, not a friendly name) — use that
      exact string as `--model` in `vllm bench serve` calls against plow.
- [x] `plowrt serve --assets assets-run/gemma4-31b-bf16 --port 8090` — running
      as PID (see `ps`/`assets-run/plowrt-serve.log`), healthy: weights
      uploaded in 5.8s (9.92 GiB/s), `GET /v1/models` returns 200, model slug
      `842da3794eaa0b77d5f08bae87a17459d91ff475` (== the checkpoint's HF
      snapshot commit hash — pass this as `"model"` in every plow request).
- [x] Correctness gate: `libcuda.so.595.91.07` confirmed mapped into the
      plowrt process AND registered as a live compute app via `nvidia-smi
      --query-compute-apps`; greedy "Paris" — exact; bicycle-balance
      paragraph — coherent, on-topic, correctly terminated at `max_tokens=80`
      (no prior-baseline text to exact-match against — this is the first-ever
      build for this model, so coherence is the applicable bar, not
      bit-exact refactor parity).

**Phase 0 COMPLETE.**

## Phase 1 — vLLM baseline (bf16 ONLY — user dropped the fp8 leg): DONE

**SCOPE CHANGE (user, mid-session): fp8 vLLM leg dropped entirely.** Only
bf16 vLLM is compared against plow. Do not resurrect the fp8/`--quantization
fp8` leg without asking first — `assets-run/gemma-31b.service.fp8` is kept
on disk (untested beyond a syntax check) in case it's wanted later, but it
is NOT part of this campaign's deliverable.

Also **note**: also had to bump `--max-model-len 8192 -> 20000` and
`--gpu-memory-utilization 0.95 -> 0.97` from the `.bak-pre-fp8kv` defaults —
8192 doesn't fit a 16000-token test prompt, and 20000 at util 0.95 OOM'd on
KV cache (needed 14.82 GiB, had 13.3 GiB available; error suggested max
15824 tokens at that util). 0.97 fixed it. This is a required correction to
the plan's stated "base bf16 flags", not a discretionary tuning choice.

- [x] bf16 leg: `gemma-31b.service` ExecStart = `--dtype bfloat16
      --gpu-memory-utilization 0.97 --max-model-len 20000` (no kv-cache-dtype
      override, no prefix-caching, no chunked-prefill). Restarted, healthy
      after 165s.
- [x] bf16 leg: `vllm bench serve` sweep, input-len 2048/8192/16000,
      concurrency 1, `--random-output-len 8 --ignore-eos --num-prompts 5
      --seed 0` (auth via `--header "Authorization=Bearer <key>"` — note
      `KEY=VALUE` format, NOT `KEY: VALUE`, or vllm bench rejects it).
      **Zero failed requests at any input length.** Logs:
      `assets-run/bench/vllm-bf16-{2048,8192,16000}.log`.

  | input | Mean TTFT (ms) | Median TTFT (ms) |
  |---|---|---|
  | 2048  | 211.73 | 208.43 |
  | 8192  | 814.25 | 908.88 |
  | 16000 | 1855.62 | 1859.31 |

- [x] Restored `/etc/systemd/system/gemma-31b.service` to its EXACT original
      (pre-session) content — `diff` against the pre-session backup
      (`assets-run/gemma-31b.service.orig-live`) is empty. Stopped (matches
      its `inactive` state at session start); `systemctl reset-failed` run
      to clear a cosmetic "failed (timeout)" mark left by force-killing it
      mid-`torch.compile` on stop — config content was already correct
      before that, this was just clearing the status display. GPU fully
      freed (`nvidia-smi`: 0 MiB used) before starting Phase 2.

**Phase 1 COMPLETE (bf16 only).**

## Phase 2 — plow bf16 baseline: DONE

**Had to re-emit the plow asset**: the Phase 0 asset was built with
`--max-ctx 8192` (the plan's own recipe), which cannot serve a 16000-token
prompt at all. Re-ran `plowc --emit devblob ... --max-ctx 20000` (same
cubins, no rebuild needed — max-ctx only affects packet/KV sizing, not the
compiled kernels), matching vLLM's bumped `--max-model-len 20000`. KV cache
grew 2.19 -> 3.09 GiB. Re-verified the correctness gate (greedy "Paris")
after re-emit before trusting any timing.

- [x] Same `vllm bench serve` sweep against plow on :8090, identical protocol.
      **Zero failed requests at any input length.** Logs:
      `assets-run/bench/plow-bf16-{2048,8192,16000}.log`.

  | input | plow Mean TTFT (ms) | vLLM Mean TTFT (ms) | plow/vLLM |
  |---|---|---|---|
  | 2048  | 942.54  | 211.73  | 0.22x (plow **4.45x slower**) |
  | 8192  | 4005.20 | 814.25  | 0.20x (plow **4.92x slower**) |
  | 16000 | 8882.37 | 1855.62 | 0.21x (plow **4.79x slower**) |

**This gap is much larger than the 12B/RTX5090 and 12B/GH200 campaigns'
18-39% gaps.** Likely cause (not yet root-caused — this is Phase 2/tuning
territory, out of scope here): `plowc` emitted prefill buckets `[128, 512,
1024]` for this 31B object — every one of the 2048/8192/16000-token test
prompts is served as MANY small chunks with per-chunk weight re-streaming,
the exact "unoptimized chunked baseline" pattern the 12B campaign measured
as its *worst* row (before the single-chunk-bucket fix, `PLOW_MAX_CHUNK`,
brought bf16 from 1744ms to 1437ms at 8192 on RTX 5090). None of that
tuning has been applied here — this is a legitimately untuned first
baseline, exactly as scoped.
- [x] Correctness gate re-checked on the re-emitted (max-ctx 20000) asset
      before trusting numbers: greedy "Paris" — exact.

**Phase 2 COMPLETE.**

## Write-up: DONE

- [x] `perf-data/gemma4-31b-h100-prefill-baseline-2026-09-04.md` — headline
      table (plow trails vLLM bf16 by 4.45-4.92x across 2048/8192/16000),
      correctness discipline, claims/non-claims, open items for Phase 2.

**Campaign (Phase 0-2, benchmark only) COMPLETE.** plow is still serving on
:8090 (PID 108847 as of session end) — left running, not part of any
production path, stop it with `kill 108847` (or `pkill -f 'plowrt serve'`)
if you want the GPU back. vLLM
services (`gemma-12b`, `gemma-31b`) are both stopped, `gemma-31b.service`
confirmed byte-identical to its pre-session content.

## Phase 3 — tuning plan (bf16 vs bf16 ONLY): NOT STARTED — paused, GPU busy elsewhere

**Do not start until the GPU is free again** (user paused this explicitly —
"the gpu is busy right now"). Everything below is the fully-researched,
ready-to-execute plan for the next session. No fp8/w8a8/quantization work
of any kind — every lever is a scheduling/architecture change on top of
full bf16 (user-set scope, explicit: "just win for bf16 vs bf16").

**Also decode is now in scope** (user: "have to win the prefil and decode
both") — Phase 1/2 only measured TTFT (prefill); we already have the TPOT
(decode) numbers too, computed from the same `vllm bench serve` runs:

| input | vLLM TPOT (ms) | plow TPOT (ms) | plow/vLLM |
|---|---|---|---|
| 2048  | 23.77 | 32.85 | 1.38x slower |
| 8192  | 23.18 | 33.50 | 1.45x slower |
| 16000 | 22.28 | 34.21 | 1.54x slower |

So: prefill trails 4.45-4.92x, decode trails 1.38-1.54x. Both need to flip
to plow winning.

### Root-cause research (this session, two Explore agents, fully cited — do not re-derive)

**Prefill**: `crates/devgen/src/lib.rs:2094-2123,6338-6342`. The chunk
ladder `[128,512,1024,2048,4096,8192]` is filtered to `<= max_chunk`, and
`max_chunk` **defaults from the model's sliding-attention window**
(`default_chunk(window) = window.next_power_of_two().clamp(...)`), NOT from
`--max-ctx`. This checkpoint's `config.json` has `sliding_window=1024`
(50/60 layers sliding, 10 full) -> `default_chunk=1024` -> shipped
`prefill_buckets: [128,512,1024]` regardless of `--max-ctx 20000`. A
16,000-token prompt is therefore served as many separate small launches
with per-chunk weight re-streaming. **`PLOW_MAX_CHUNK` is the existing,
documented emit-time override** (clap arg `--emit-max-chunk`, env
`PLOW_MAX_CHUNK`, read at `crates/devgen/src/emit_config.rs:121,586`).
Ceiling is 8192 (hard `assert` in code, power-of-two only, max 8192 —
`PLOW_MAX_CHUNK=20000` would abort emission, do not try it).

KV memory cost (only the 50 sliding layers scale; ring =
`next_pow2(window + chunk - 1)`; full-attn layers use `ctx` directly,
unaffected):
| PLOW_MAX_CHUNK | sliding ring | total KV/seq |
|---|---|---|
| 1024 (today) | 2048 | 3.09 GiB (matches measured) |
| 2048 | 4096 | ~4.65 GiB |
| 4096 | 8192 | ~7.78 GiB |
| 8192 (ceiling) | 16384 | ~14.0 GiB |

All fit fine at concurrency 1 (80GB card, 57.2 GiB weights resident, our
benchmark protocol is concurrency 1) — cost matters only for future
multi-concurrency work, note but don't block on it.

**Decode**: the four "H100 decode fixes" baked unconditionally into every
`build_sm90a_cubin.sh` build (`PLOW_NV_GEMV_RB`, `PLOW_MOE_DOWN_LANESPLIT`,
`PLOW_NV_FA_WPR`, `PLOW_NV_FP8_RB=4`) were validated on **Gemma-4-26B-A4B, a
MoE model**. Traced through `runtime/nvidia/op_moe.cuh`: `PLOW_NV_GEMV_RB`'s
big win and `PLOW_MOE_DOWN_LANESPLIT` are both MoE-router/MoE-down logic
that **never dispatches on a dense model** — Gemma-4-31B is dense
(`config.json`: `enable_moe_block: false`). `PLOW_NV_FP8_RB=4` only affects
fp8 GEMV (irrelevant, we're bf16). **Only `PLOW_NV_FA_WPR` (flash
warp-per-row) is doing real work for this model+precision.** This exact
shape (31B, dense, bf16, H100) has never been tuned — `tuning/README-
decode-tuner.md` + `scripts/tune_decode_sweep.sh` key their tunedb on model
name (`crates/tunedb/src/decode.rs:97`) and have no entry for it yet.
Tuner sweeps: object knobs (~40s/build: `PLOW_NV_FORCE_MINBLK`,
`GV_UNROLL`, `GV_UNROLL_GLU`, `GV_MOE_UN`, `PLOW_MOE_DOWN_SG`, `GV_MM_MAX`,
`PLOW_NV_FA_*`) and packet knobs (~60s/emit: `--n-cu`, `PLOW_NS_ABS`,
`PLOW_NS_FULL_ABS`, `PLOW_DECODE_BATCH`) — full 32-config x 4-ctx grid is
~75 min per (gpu, dtype) per the tuner's own README.

Decode occupancy: currently 1 block/SM (208 regs) — confirmed in serve log
(`occ_per_sm=1`). A real, HW-validated lever exists
(`perf-data/segmented-decode-26b-h100.md`: `PLOW_NV_LEAN_DECODE=1
PLOW_NV_FORCE_MINBLK=2` reached occ-2, ~1.32x self-speedup) but **only
measured on the 26B MoE model's GEMV arms**, and is **unshipped for
sm_90a**: not in `build_sm90a_cubin.sh`, not a CMake target for sm_90a
(pinned to `sm_120a` only in `runtime/CMakeLists.txt:337-384`), no
segmented-decode dispatch path in `exec/gpu.rs` for this arch. Real
engineering, not a flag flip — `PLOW_NV_FORCE_MINBLK` itself IS already
available on the existing monolithic sm_90a decode object (shared header
`interp_sm120.cu:583-595`, included by `interp_sm90a.cu:42`) so a **cheap
probe** (just add `-DPLOW_NV_FORCE_MINBLK=2` to the existing decode build,
no object-split) is worth trying first, before committing to the fuller
port.

### The 4-stage plan (execute Stage 1 -> 2 -> checkpoint -> maybe 3/4)

**Stage 1 (prefill, low risk, no rebuild): DONE, 2026-09-04.** GPU was
stopped/restarted mid-session (`gemma-31b.service` had been started outside
this session with a drop-in override, using 77 GiB — stopped on user
confirmation before this stage began). Emitted into separate asset dirs
(`assets-run/gemma4-31b-bf16-mc{2048,4096,8192}`, cubins symlinked from the
original `gemma4-31b-bf16` dir, no rebuild) at `PLOW_MAX_CHUNK={2048,4096,
8192}`, `--max-ctx 20000` unchanged. All three correctness-gated (greedy
"Paris" — exact, every variant) and full-swept. Zero failed requests
throughout.

| PLOW_MAX_CHUNK | TTFT @2048 | TTFT @8192 | TTFT @16000 | KV/seq |
|---|---|---|---|---|
| 1024 (baseline) | 942.54 | 4005.20 | 8882.37 | 3.09 GiB |
| 2048 | 841.58 | 3481.61 | 7799.84 | 4.65 GiB |
| 4096 | 840.18 | 3360.59 | 7557.06 | 7.78 GiB |
| **8192 (winner)** | **839.55** | **3289.35** | **7398.71** | 14.03 GiB (74 GiB total used, fits) |

TPOT unaffected by chunk size, as expected (~32.8-34.2ms across all four,
matches baseline within noise).

**Result: only 10-18% faster than baseline, monotonic but heavily
diminishing returns — nowhere near closing the ~4.5-4.9x gap.** This is a
materially different (weaker) result than the 12B/RTX-5090 campaign, where
the equivalent lever closed most of a *much smaller* (31%->19%) gap. At
`PLOW_MAX_CHUNK=8192` plow still trails vLLM by **~3.97-4.04x** on TTFT
(839.55/211.73, 3289.35/814.25, 7398.71/1855.62) — down from 4.45-4.92x, but
still roughly 4x. **This strongly implies the dominant cost for this
model/GPU is NOT chunk/launch overhead** (which Stage 1 fixes) **but
underlying per-token GEMM/attention kernel efficiency** — consistent with
the sm_120 bf16-gap diagnosis in `perf-data/sm120-bf16-gap-findings-2026-
08-26.md` (GEMM at 66-71% of cuBLASLt, flash-attention at 34-44% of the raw
mma ceiling) — that diagnosis was for a different chip/model but the
mechanism (engineering-depth gap vs cuBLAS's autotuned kernels, not chunk
overhead) likely generalizes. **Practical read: Stage 3 (the segmented
TMA-GEMM architecture) is very likely necessary here, not just a fallback**
— flag this to the user before investing in Stage 2's decode-only work, or
proceed to Stage 2 anyway since it's cheap and orthogonal, then escalate.

Winning config for everything downstream: `PLOW_MAX_CHUNK=8192`, asset dir
`assets-run/gemma4-31b-bf16-mc8192`. Servers for mc2048/mc4096 stopped
after their sweeps; only mc8192 is the carried-forward baseline.

**Stage 2 (decode, low risk, no new engineering): DONE, 2026-09-04.** Built
`step_bench` example + `tunedb-decode` bin (`cargo build --release -p
plowrt --features cuda --example step_bench` /
`-p tunedb --bin tunedb-decode`) — both auto-discovered, no Cargo.toml
edits needed. Ran `scripts/tune_decode_sweep.sh` against the Stage-1
checkpoint, `--model` pointed at the real HF snapshot dir.

**Real bug found and tested (negative result, but worth recording — fixes a
documentation/tuner gap for any future sm_90a sliding-window model):**
`build_sm90a_cubin.sh`'s decode object unconditionally bakes
`-DPLOW_NV_FA_GF_FULL=4`, but this model's sliding layers have GQA=2
(`heads=32 / kv_heads=16`), and the PACKET-side emitter (`devgen`,
`crates/devgen/src/lib.rs:3049-3055`) sizes nsplit from a SINGLE global
`fa_gf_full()` value applied uniformly to every layer's
`assert_eq!((heads/kvh) % gf, 0)` check — so packets can only ever be
emitted at `PLOW_FA_GF_FULL` &lt;= 2 for this model (4 or 8 panics at emit:
"layer 0: GF 4 must divide GQA 2"). This is exactly the object/packet
mismatch pattern `tuning/README-decode-tuner.md` warns about ("a mismatch
silently mis-fills the grid") — our shipped decode cubin (GF_FULL=4 object)
has been paired with every packet we've emitted (GF_FULL=2, the devgen
default) for this entire campaign. **Tested the fix directly**: rebuilt the
decode+prefill cubins with `PLOW_EXTRA_DEFINES="-DPLOW_NV_FA_GF_FULL=2"`
(matching pairing), correctness-gated (greedy "Paris" exact, bicycle-
balance paragraph **byte-identical** to the mismatched baseline — confirms
this is purely a scheduling knob, no numerics change), then ran the full
sweep: TTFT unchanged (prefill uses its own separate `PLOW_NV_FA_GF=2`,
untouched by this), **TPOT got very slightly WORSE** (32.98 vs 32.84ms
@2048, 33.88 vs 33.45ms @8192, 34.95 vs 34.23ms @16000 — 1-2% regression).
**Conclusion: the mismatch is real but not the cause of the decode gap;
reverted to the original GF_FULL=4 object** (already our best-measured,
matches the build script's own "CONFIRMED OPTIMAL ON H100" comment,
empirically re-confirmed here). Do not "fix" this pairing again without a
new measurement — the assumption that clean pairing helps was directly
tested and refuted.

Ran a second, corrected scoped sweep (avoiding the GF_FULL axis entirely,
since only 1/2 are packet-valid for this model — not worth sweeping):
`--gv-unroll "4 8" --ns-full-abs "0 33" --ctx 8192 --reps 3` (4 grid points,
~10 min). Results (raw `step_bench` decode-step TPOT, not full-serving):

| GV_UNROLL | NS_FULL_ABS | step TPOT (ms) |
|---|---|---|
| 4 | 0 or 33 | 31.36-31.37 |
| **8** | 0 or 33 | **29.19-29.20** (6.9% faster) |

`NS_FULL_ABS` made no measurable difference to decode (expected — it's a
prefill-tuned, full-attention-layer-only split count). **GV_UNROLL=8
already IS our production default** (`runtime/nvidia/op_gemm.cuh:32-33`,
`#ifndef GV_UNROLL / #define GV_UNROLL 8`, never overridden in
`build_sm90a_cubin.sh`) — so this sweep **confirms our current decode
object is already at the local optimum on every knob tested**, it does not
identify a new win.

**Net Stage 2 result: no improvement found.** The scoped tuner pass
(GF_FULL pairing, GV_UNROLL, NS_FULL_ABS) is exhausted with zero net gain —
one real bug found and correctly ruled out as a red herring, one knob
confirmed already-optimal. TPOT stays at the Stage-1/baseline numbers:
32.84/33.45/34.23 ms @ 2048/8192/16000, vs vLLM's 23.77/23.18/22.28 —
**1.38-1.54x slower, unchanged.** This is consistent with the plan's own
Stage 4 hypothesis (occupancy, currently 1 block/SM, is the real lever —
not fine-grained constants) and rules out easy wins before committing to
that bigger engineering lift.

**Checkpoint: DONE, 2026-09-04. Plow does NOT win yet on either metric —
Stage 3 and/or Stage 4 (architecture-level work) are required, not
optional.** Best config so far (`PLOW_MAX_CHUNK=8192`, everything else
production default — `assets-run/gemma4-31b-bf16-mc8192`, currently the
live server on :8090):

| input | plow TTFT | vLLM TTFT | ratio | plow TPOT | vLLM TPOT | ratio |
|---|---|---|---|---|---|---|
| 2048  | 839.55  | 211.73  | **3.97x slower** | 32.84 | 23.77 | **1.38x slower** |
| 8192  | 3289.35 | 814.25  | **4.04x slower** | 33.45 | 23.18 | **1.45x slower** |
| 16000 | 7398.71 | 1855.62 | **3.99x slower** | 34.23 | 22.28 | **1.54x slower** |

Summary of what the cheap/no-new-engineering levers bought: prefill 10-18%
faster (chunk bucketing), decode 0% (already at local optimum, one
plausible bug tested and ruled out). Neither stage came close to flipping
the sign. Recommendation: both Stage 3 (prefill) and Stage 4 (decode) are
needed to have a realistic shot at beating vLLM — this is real kernel/
build-system engineering, not more parameter sweeping. Get explicit user
sign-off before starting either, given the effort/risk step-up from
Stage 1-2.

**Stage 3 (prefill, higher effort): ATTEMPTED, 2026-09-04. Real, large
REGRESSION found — reverted. Do not retry this exact configuration.**

Attempted the canonical GH200 bf16 recipe (`docs/flags-reference.md:
559-613`, `PLOW_BUILD_W8A8` dropped, bf16-only):
`PLOW_EXTRA_DEFINES="-DPLOW_NV_FA256_BKV=64 -DPLOW_NV_FA512_BKV=32"
PLOW_BUILD_TMA_GEMM=1 PLOW_BUILD_SEG=1 PLOW_BUILD_FATLITE=1
PLOW_BUILD_GEMM_WS384=1 PLOW_BUILD_FA512=1 PLOW_BUILD_FA_WG=1
PLOW_BUILD_FA_HD256=1`.

**The WS384 (`_pfgemm`) object failed to compile**: `d_gemm_sm90_tma_ws384_role`,
`d_gemm_sm90_tma_uni256`, `d_quant_fp8_ws384`, `sm90_reg_dec/inc` all
"undefined identifier" in `interp_sm120.cu`, despite `-DPGM90_UNI_BN256=1
-DPLOW_NV_SEG_WS384=1` both being present on the nvcc command line (verified
via `nvcc -E` — the call sites in `interp_sm120.cu` survive preprocessing,
but the DEFINITIONS in `op_gemm_sm90.cuh:717-1259` (gated `#if PGM90_UNI_BN256
&& defined(PLOW_NV_SEG_WS384) && PLOW_NV_SEG_WS384`) do NOT — confirmed by
grepping the `-E` output for a unique comment string from that block, zero
hits). Root cause not found (traced the include chain, the `#ifndef`
guards, checked for `#undef` — none found; nvcc's actual multi-pass
cudafe++ compilation may preprocess differently than a standalone `-E` run,
which would explain the discrepancy, but this wasn't confirmed). **This is
a real, reproducible build bug in this exact flag combination on this
toolchain (CUDA 13.2) — flagging for whoever touches sm_90a WS384 next,
do not assume it "just works" from the docs' canonical-recipe listing.**

**Worked around by substituting `PLOW_BUILD_GEMM_ONLY=1`** (the simpler
plain lean object) for `PLOW_BUILD_GEMM_WS384=1` — this DID compile cleanly
(all 5 objects: decode, monolithic `_pf`, `_pfseg`, `_pfgemm`, `_pffa|`
built, correct kernel symbols verified via `cuobjdump`). Emitted matching
packet (`PLOW_SEG_CLASS_SLICE=1 PLOW_NO_GLU_FUSE=1 PLOW_SEG_PURE_GEMM=1
PLOW_SEG_FA512=all`, NO `PLOW_UNISEG`). Served with `--pf-seg-dir
--pf-seg-pure 1 --pf-seg-fa512 all`. Loaded cleanly: `segmented=true,
grid_gemm=264` (132 SM x 2 blocks/SM — real occupancy-2 achieved).
**Correctness gate passed exactly** — greedy "Paris" exact, bicycle-balance
paragraph **byte-identical** to the UNISEG baseline (confirms the
architecture is genuinely token-identical, matching the campaign's claim).

**Measured performance: catastrophic regression, not a win.**

| input | seg-bf16 TTFT | mc8192 baseline TTFT | seg-bf16 TPOT | mc8192 baseline TPOT |
|---|---|---|---|---|
| 2048  | 4183.21  | 839.55  | 36.83 | 32.84 |
| 8192  | 15539.13 | 3289.35 | 37.48 | 33.45 |
| 16000 | 30684.05 | 7398.71 | 38.20 | 34.23 |

**~4.7-5x SLOWER on TTFT than our already-behind baseline** (i.e. ~18-20x
slower than vLLM), TPOT also slightly worse. Zero failed requests — this is
a real, reliable measurement, not a crash or fluke.

**Reverted immediately to the mc8192 UNISEG baseline** (`assets-run/
gemma4-31b-bf16-mc8192`), confirmed healthy.

**Reading (not confirmed by profiling — `ncu`/`cuda-gdb` unavailable in
this sandbox, same limitation prior sessions hit)**: two live hypotheses,
neither ruled out:
1. **The GEMM_ONLY substitution is itself much worse than WS384** — the
   docs are explicit the campaign's real numbers depended on the
   384-thread producer/consumer "cuBLAS shape" body, not the plain lean
   object. A regression from a worse GEMM body is plausible but a 5x
   swing seems too large to attribute to tile-body choice alone.
2. **Segmentation's benefit needs concurrency/batch to show up.** Our
   protocol is concurrency 1 (one sequential request, matching vLLM's own
   config) — multiple small launches per chunk (fat/gemm/fa objects) with
   cross-segment gate synchronization may cost more in serialized
   per-launch overhead than a single monolithic launch saves, when there's
   no second request to hide that latency behind. The GH200 campaign's own
   numbers ("seg fat-only 450.5 vs monolithic 442.6 — the 97 launches are
   CHEAP") suggest segmentation ALONE should be near-neutral, not a 5x
   regression — so if this hypothesis is right, something in OUR specific
   config (GEMM_ONLY object, FATLITE's 128-reg fat object, or an H100-
   specific launch/sync cost difference from GH200) is compounding badly.

**Recommendation: do not pursue this architecture further without either
(a) fixing the WS384 compile bug and re-testing the actual canonical
recipe, or (b) profiling tools to see where the time actually goes — both
are real, uncertain-effort investigations, not quick fixes. Stage 4
(decode) is a smaller, independent, lower-risk experiment — doing that
next rather than continuing to debug this blind.**

**Stage 4 (decode): cheap probe DONE, 2026-09-04 — marginal win, does not
close the gap.** Built decode+prefill cubins with `-DPLOW_NV_FORCE_MINBLK=2`
(fits cleanly: REG 208->128, STACK 472B, LOCAL:0 — no heavy spill). Emitted
a matching `--n-cu 264` packet (occupancy pair, per the tuner's own
warning) and a `--n-cu 132` control packet paired with the UNMODIFIED
production decode cubin. Tested via `step_bench` directly (isolated decode
step, no serving-path risk to the working mc8192 asset):

| ctx | occ-1 (baseline) step ms | occ-2 (FORCE_MINBLK=2) step ms | delta |
|---|---|---|---|
| 2048  | 28.671 | 28.488 | -0.6% |
| 8192  | 29.204 | 28.827 | -1.3% |
| 16000 | 29.881 | 29.275 | -2.0% |

**Real, reproducible, but far too small to matter** (need to close a
1.38-1.54x gap; this recovers ~1-2%). Consistent with the earlier finding
that the levers which made occ-2 a big win on the 26B campaign
(`PLOW_NV_LEAN_DECODE` arm-stripping + occ-2 together, projected 1.93x ->
~1.32x gap — itself STILL a loss to vLLM in that campaign's own numbers)
are MoE-specific machinery inert on this dense model. Bare `FORCE_MINBLK=2`
on the full (MoE-arms-included-but-unused) object recovers only the
occupancy part, not the register-pressure-reduction part, which the 26B
win depended on more heavily.

**Did not attempt the fuller `PLOW_NV_LEAN_DECODE` port** (new CMake target
for sm_90a — currently pinned to sm_120a only — plus `exec/gpu.rs`
segmented-decode dispatch plumbing the source 26B report itself calls
unfinished). Judged not worth it: even the BEST case precedent (26B, MoE
model, where lean-stripping removes real dead weight) only reached ~1.32x
behind vLLM, not parity or a win — and this model has less to strip (dense,
most of what lean-decode removes is MoE machinery already unexercised
here). Real engineering effort for a return that, on the best available
precedent, still doesn't flip the sign.

Reverted the probe assets, restored the production mc8192 server
(`assets-run/gemma4-31b-bf16-mc8192`), confirmed healthy.

## Stage 5 — op fusion audit (user idea: "fuse more so occupancy is more"): DONE, 2026-09-04

**Egglog is a dead end for this — ruled out fast, don't revisit.**
`docs/bringup/02-egglog-rewrite.md` states in its own Scope section: the
egglog rewrite pass is "a working library that is **not on the shipping
emit path**" — `plowc --emit devblob` (what builds every asset in this
campaign) does not consume a `FusedGraph`; packets are hand-written
directly in `crates/devgen`. Fusing more in egglog would never reach a
kernel we serve.

**The real lever was in the hand-written path**: the GH200 12B campaign's
own "FUSION AUDIT" (`perf-data/gemma12b-gh200-prefill-campaign.md:56-61`)
flagged "prefill runs the sandwich norms UNFUSED... decode has
NORM_RESIDUAL_NORM... ~2-3%[of TTFT]" as an identified-but-untried gap.
Confirmed in source: `gfuse = c.arch == Arch::Gemma4 && (gemv_family ||
emit_config::active().pf_gfuse)` (`crates/devgen/src/lib.rs:2932`) —
`gemv_family` is decode, so the fusion is automatic there; prefill needs
the explicit opt-in `PLOW_PF_GFUSE=1` (`emit_config.rs:514-516`), never set
in this campaign until now.

**Tested `PLOW_PF_GFUSE=1`** on top of the Stage-1-winning `mc8192` config
(same cubins, no rebuild — packet-emit-time flag only, reuses the
already-decode-proven `NormResidualNorm` op). Real packet-count drop: 896
-> 776 per prefill bucket (~13-14% fewer packets). Correctness: greedy
"Paris" exact; bicycle-balance paragraph **coherent but NOT byte-identical**
(expected and accepted — the fusion changes floating-point reduction order,
documented in-source as "last-ulp bf16 flips → token divergence possible";
this is a real but bounded numerics reassociation, not a logic bug).

| input | mc8192 TTFT | mc8192+gfuse TTFT | delta |
|---|---|---|---|
| 2048  | 839.55  | 833.72  | -0.7% |
| 8192  | 3289.35 | 3268.45 | -0.6% |
| 16000 | 7398.71 | 7368.79 | -0.4% |

Landed at the low end of the audit's own "~2-3%" estimate, not the high
end. Small, real, zero regressions, zero failed requests. **Adopted as the
new best config**: `assets-run/gemma4-31b-bf16-mc8192-gfuse` (currently
serving on :8090).

**Checked two more fusion flags for applicability, both ruled out without
needing a test run:**
- `PLOW_PF_GEMV_HEAD` (lm_head GEMV epilogue fusion, pairs with cubin flag
  `PLOW_BUILD_GEMV_HEAD=1`) — **confirmed INERT** by an existing doc,
  `perf-data/gemma4-12b-longctx-5090.md:549-552`: "-39% is per-launch on
  `lm_head`, ~0.3% of a 127k prefill at chunk 1024" — exactly the trap that
  doc itself warns about, scaling an isolated-op ratio through an assumed
  budget instead of the measured one. Not worth building.
- `PLOW_QNORM_FUSE`, `PLOW_FUSE_HNR`, `PLOW_FUSE_MERGE`, `PLOW_FUSE_ARGMAX`
  — traced in source: `qnorm_fuse` is gated `w8a8 && ...` (fp8-only, out of
  bf16-only scope); `fuse_hnr`/`fuse_merge` are gated `gemv_family && ...`
  (decode-only, already at local optimum per Stage 2). None apply to
  prefill/bf16.

**Updated best-known numbers (mc8192 + PF_GFUSE, bf16):**

| input | plow TTFT | vLLM TTFT | ratio | plow TPOT | vLLM TPOT | ratio |
|---|---|---|---|---|---|---|
| 2048  | 833.72  | 211.73  | 3.94x slower | 32.86 | 23.77 | 1.38x slower |
| 8192  | 3268.45 | 814.25  | 4.01x slower | 33.48 | 23.18 | 1.44x slower |
| 16000 | 7368.79 | 1855.62 | 3.97x slower | 34.20 | 22.28 | 1.54x slower |

Barely moves the needle (4.0-4.9x -> 3.9-4.0x on TTFT). The fusion-flag
search space for bf16 Gemma-4 prefill is now exhausted as far as static
source inspection can find it — every remaining unflipped flag is either
architecture-mismatched (K3/GLM/MoE-specific), precision-mismatched
(fp8-only), or decode-scoped (already optimal).

## Stage 6 — revisiting segmented prefill: BREAKTHROUGH, 2026-09-04

User pushed for other options after Stage 5. Re-attacked Stage 3's 5x
regression systematically instead of abandoning it.

**Bisection, three cheap server-flag tests first (no rebuild — ruled out
launch/scheduling overhead entirely):**
- `--pf-seg-graph` (submit whole segment chain as one CUDA graph): 15438ms
  @8192 — no change from the 15539ms baseline regression.
- `--pf-seg-noncoop` (non-cooperative launches): 15572ms @8192 — no change.
- Conclusion: the 5x regression is NOT host-launch/scheduling overhead —
  it's inside the kernel bodies themselves.

**Rebuild bisection**: dropped `PLOW_BUILD_FATLITE=1` from the recipe
(kept `PLOW_BUILD_TMA_GEMM=1 PLOW_BUILD_SEG=1 PLOW_BUILD_GEMM_ONLY=1
PLOW_BUILD_FA512=1 PLOW_BUILD_FA_WG=1 PLOW_BUILD_FA_HD256=1`). FATLITE
caps the fat `_pfseg` object (which still handles ALL norm/misc ops) to
128 registers for occupancy-2 — confirmed via `cuobjdump`: without it the
fat object reverts to REG:255/occ-1, no spill. **This alone fixed the
regression completely**: 15539 -> 2905.95ms @8192 (5.3x recovery), already
*faster* than the mc8192+gfuse baseline (3268.45ms). Correctness:
byte-identical to the pre-Stage-5 baseline (no PF_GFUSE in this emit).
**Root cause of Stage 3's regression, now confirmed: FATLITE's
register-starved fat object, not segmentation, TMA, or launch overhead.**

**Then found TMA was never actually active**: `PLOW_BUILD_TMA_GEMM=1`
(object/cubin flag) alone is inert — per `build_sm90a_cubin.sh`'s own
comment, the TMA GEMM body only dispatches when the packet carries
tensormap handles, which requires the SEPARATE packet-side
`PLOW_TMA_GEMM=1` (`crates/devgen/src/emit_config.rs:511`) — never set in
any Stage 3 attempt until now. Added it, plus `PLOW_PF_GFUSE=1` (Stage 5's
win, stacks independently). Same cubins, packet re-emit only.

**Result — the big one:**

| input | plow TTFT (seg+TMA+gfuse) | vLLM TTFT | ratio | plow TPOT | vLLM TPOT | ratio |
|---|---|---|---|---|---|---|
| 2048  | **533.43**  | 211.73  | **2.52x slower** | 36.90 | 23.77 | 1.55x slower |
| 8192  | **1998.86** | 814.25  | **2.45x slower** | 37.54 | 23.18 | 1.62x slower |
| 16000 | **4141.87** | 1855.62 | **2.23x slower** | 38.27 | 22.28 | 1.72x slower |

Zero failed requests at every point. Correctness: greedy "Paris" exact;
bicycle-balance paragraph coherent (expected wording drift from combined
TMA + norm-fusion numerics reassociation, same accepted category as
Stage 5's gfuse alone).

**From ~4.0x behind (Stage 5 end) to ~2.2-2.5x behind on TTFT — the
single biggest win of the whole campaign, larger than Stage 1+2+4+5
combined.** Decode (TPOT) drifted slightly worse (1.4-1.5x -> 1.55-1.72x)
— unexplained, worth checking whether the segmented prefill path is
perturbing something decode-adjacent (KV layout, arena reuse) rather than
assuming it's noise.

**Config**: cubins `assets-run/gemma4-31b-seg-nofatlite/` (decode +
monolithic `_pf` + `_pfseg`(255reg/occ1) + `_pfgemm`(occ2) + `_pffa`),
packet `assets-run/gemma4-31b-seg-nofatlite-tma-gfuse/` (`PLOW_MAX_CHUNK=8192
PLOW_TMA_GEMM=1 PLOW_PF_GFUSE=1 PLOW_SEG_CLASS_SLICE=1 PLOW_NO_GLU_FUSE=1
PLOW_SEG_PURE_GEMM=1 PLOW_SEG_FA512=all`, NO `PLOW_UNISEG`). Serve:
`--pf-seg-dir assets-run/gemma4-31b-seg-nofatlite --pf-seg-pure 1
--pf-seg-fa512 all`. **This is now the best-known config — currently
serving on :8090.**

**Chunk re-sweep (mc4096) tried**: 2020.94ms @8192, 4180.66ms @16000 — no
better than mc8192 (1998.86/4141.87), slightly worse. mc8192 confirmed
still the winner under the new architecture too.

**Decode TPOT regression investigated, inconclusive — deprioritized.**
Ruled out: the decode CUBIN itself is byte-identical in register footprint
(REG:200/STACK:0/SHARED:2448, `cuobjdump`) between old and new builds, so
the extra `PLOW_EXTRA_DEFINES` in the Stage 6 cubin build did not change
the compiled decode kernel. Found a real packet-level difference instead:
decode's own program (`prog 6, T=1`) carries the SAME packet count (676)
but ~1.9x the workgroup-packets (81035 vs 42623) in the segmented emit.
Leading hypothesis, not confirmed: `PLOW_SEG_CLASS_SLICE=1`
(`crates/packet/src/devbuild.rs:1354-1403`, required for the GEMM
segment's occ-2 re-slicing) may be doubling ops it shouldn't outside the
prefill program — traced the gating logic partway (`wave_class`,
`pure_gemm`) but did not reach a definitive per-program scoping answer in
the time available. **Given our benchmark's `--random-output-len 8` makes
TTFT the dominant term in total latency (533-4142ms TTFT vs 8×~37ms=~296ms
of decode), the net effect of adopting this config is still strongly
positive** — but this regression should be root-caused before treating the
config as final/shippable.

## Stage 7 — WS384 fixed: ROOT CAUSE found and resolved, another huge win, 2026-09-04

**Root cause of the Stage 3 WS384 compile failure, finally found.** Mapped
every `#if`/`#endif` in `runtime/nvidia/op_gemm_sm90.cuh` by line number.
`#if PLOW_NV_W8A8` opens at **line 583** and does not `#endif` until
**line 1680** — it wraps the ENTIRE UNI_BN256 (726-976), WS384 (978-1259),
AND SEG_GEMM (1261-1667) sections, INCLUDING the sections' own "bf16 twin"
bodies (e.g. `d_gemm_sm90_tma_uni256`, explicitly commented "T20b: bf16
twin of the fp8 uni256 body below"). **The bf16 WS384/UNI256 bodies are
nested inside a `W8A8`-only guard** — so dropping `PLOW_BUILD_W8A8` (as
Stage 3 did, correctly per `docs/flags-reference.md`'s general "drop W8A8
for bf16-only cubins" guidance) also removes the bf16 bodies. This
explains the original "undefined identifier" errors exactly: the call
sites in `interp_sm120.cu` aren't gated the same way, so they survive
while their definitions don't.

**Found the workaround already in the tree**: `runtime/nvidia/experiments/
ws384_probe.cu` (the standalone T31 reference probe) `#define`s
`PLOW_NV_W8A8 1` even though it ALSO builds and tests the bf16 (`E4M3=
false`) instantiation — the probe's author already knew/worked around this
exact guard.

**Fix**: build WITH `PLOW_BUILD_W8A8=1` (unlocks compiling BOTH precision
arms into one object — this is the documented design: "one lean object
serves bf16 and fp8") but do NOT set `PLOW_W8A8=1` at packet emit — that's
the separate, independent flag that actually determines whether fp8
opcodes get written into the packet. Our packets stay 100% bf16 opcodes;
the fp8 arms are compiled in but never dispatched. Recipe: `PLOW_BUILD_TMA_GEMM=1
PLOW_BUILD_SEG=1 PLOW_BUILD_GEMM_WS384=1 PLOW_BUILD_W8A8=1 PLOW_BUILD_FA512=1
PLOW_BUILD_FA_WG=1 PLOW_BUILD_FA_HD256=1` (FATLITE deliberately omitted,
per Stage 6). **All 5 objects built clean, first try.** Registers exactly
match the design doc's claims: `_pfgemm` REG:160 (the documented
`__maxnreg__(160)` entry point), `_pfseg` REG:255/occ-1 (no FATLITE).

Emitted with the same packet flags as Stage 6 (`PLOW_TMA_GEMM=1
PLOW_PF_GFUSE=1 PLOW_MAX_CHUNK=8192`, no `PLOW_UNISEG`, no `PLOW_W8A8`).
Loaded: `grid_gemm=132 block_gemm=384` — WS384 runs 384 THREADS/block
(dedicated producer warpgroup + 2 consumer warpgroups) at occ-1, a
structurally different (and per the design docs, more cuBLAS-like)
shape than GEMM_ONLY's occ-2/256-thread approach, not just "the same
thing with more registers." Correctness: greedy "Paris" exact,
bicycle-balance coherent.

**Result:**

| input | plow TTFT (WS384) | vLLM TTFT | ratio | plow TPOT | vLLM TPOT | ratio |
|---|---|---|---|---|---|---|
| 2048  | **326.80**  | 211.73  | **1.54x slower** | 37.07 | 23.77 | 1.56x slower |
| 8192  | **1257.75** | 814.25  | **1.54x slower** | 37.61 | 23.18 | 1.62x slower |
| 16000 | **2702.67** | 1855.62 | **1.46x slower** | 38.33 | 22.28 | 1.72x slower |

Zero failed requests. **From 2.2-2.5x behind (Stage 6) to 1.46-1.54x
behind — another huge win, larger than Stage 6's own jump.** Cumulative
from campaign start: **~4.0-4.9x -> ~1.5x behind on TTFT.** Decode TPOT
regression persists at roughly the same magnitude as Stage 6 (still
unexplained, still likely `PLOW_SEG_CLASS_SLICE`-related, still a much
smaller drag than the prefill win is a gain given our 8-token output
protocol).

**Config**: cubins `assets-run/gemma4-31b-seg-ws384/`, packet
`assets-run/gemma4-31b-seg-ws384-asset/`. Serve: `--pf-seg-dir
assets-run/gemma4-31b-seg-ws384 --pf-seg-pure 1 --pf-seg-fa512 all`.
**Now the best-known config — currently serving on :8090.**

## Stage 8 — decode regression FIXED too: `PLOW_SEG_CLASS_SLICE` confirmed and dropped, 2026-09-04

Immediately tested the Stage 6 hypothesis against the new WS384 config:
WS384 runs at **occ-1** (`grid_gemm=132` from the Stage 7 load log, not
264) — `PLOW_SEG_CLASS_SLICE=1`'s entire purpose is filling a SECOND
resident block/SM at occ-2 (`crates/packet/src/devbuild.rs:1354-1359`:
"occ-2 makes 2 blocks/SM resident... otherwise half the grid idles at
occ-2"). WS384 never has a second block, so the flag should be pure
dead weight for this object — worth dropping.

Re-emitted the identical Stage 7 packet with `PLOW_SEG_CLASS_SLICE`
simply omitted (nothing else changed). Confirmed via the emit log: decode's
own program (`prog 6, T=1`) workgroup-packets dropped from 81035 back to
**42623** — exactly the pre-Stage-6 baseline value. Prefill programs also
dropped somewhat (e.g. `prog 0`: 111049 -> 95077) — the flag was doubling
some prefill dispatch too, unnecessarily, since WS384 doesn't need it either.

Correctness: greedy "Paris" exact, bicycle-balance byte-identical to the
Stage 7 (with-slice) output — confirms this is a pure dispatch-sizing
change, no numerics difference at all (expected: the flag only changes
`cus` slicing counts, not what gets computed).

**Result: TTFT unchanged, TPOT fully recovered.**

| input | plow TTFT | vLLM TTFT | ratio | plow TPOT | vLLM TPOT | ratio |
|---|---|---|---|---|---|---|
| 2048  | 327.05  | 211.73  | 1.545x slower | **32.99** | 23.77 | **1.388x slower** |
| 8192  | 1261.10 | 814.25  | 1.549x slower | **33.53** | 23.18 | **1.447x slower** |
| 16000 | 2712.38 | 1855.62 | 1.462x slower | **34.31** | 22.28 | **1.540x slower** |

Zero failed requests. **Both the Stage 6 decode regression and the
Stage 3 WS384 compile bug are now fully resolved — this is unambiguously
the best config on every metric measured this campaign.**

**Cumulative progress this session: TTFT 4.0-4.9x -> 1.46-1.55x behind;
TPOT 1.38-1.54x behind (essentially unchanged from the very first
baseline — the regression introduced and then fully un-done, net zero
change on decode, all the gain is on prefill).**

**Config (current best, live on :8090)**: cubins
`assets-run/gemma4-31b-seg-ws384/` (unchanged from Stage 7), packet
`assets-run/gemma4-31b-seg-ws384-noslice/` (`PLOW_MAX_CHUNK=8192
PLOW_TMA_GEMM=1 PLOW_PF_GFUSE=1 PLOW_NO_GLU_FUSE=1 PLOW_SEG_PURE_GEMM=1
PLOW_SEG_FA512=all`, NO `PLOW_UNISEG`, NO `PLOW_SEG_CLASS_SLICE`). Serve:
`--pf-seg-dir assets-run/gemma4-31b-seg-ws384 --pf-seg-pure 1
--pf-seg-fa512 all` (no `--pf-seg-graph`, no `--pf-seg-noncoop` — neither
helped, per Stage 6).

## Stage 9 — chunk resweep (no win) + PGM90_UNI256_NS tuning (small win), 2026-09-04

**Chunk resweep on final WS384/no-slice architecture**: `PLOW_MAX_CHUNK=4096`
tested — 1268.26ms/2723.09ms @8192/16000, both slightly worse than 8192's
1261.10/2712.38. 8192 confirmed still best.

**`PGM90_UNI256_NS` tuning** (default 4, sm_90a GEMM TMA ring depth):
- `NS=3`: builds clean (smaller pfgemm object, 103720 vs 106792 bytes),
  correctness byte-identical (pure staging change). **Small, real,
  ctx-scaling win**: 327.00/1243.86/2668.36ms @2048/8192/16000 vs the
  NS=4 default's 327.05/1261.10/2712.38 — 0%/-1.4%/-1.6%, bigger at longer
  context (more pipelining to benefit from a leaner ring). **Adopted.**
- `NS=5`: **fails to load** — `dynamic smem 246864 B exceeds device
  opt-in limit 232448 B`. Confirms NS=3 is near the practical ceiling in
  this direction; the doc's own warning about `NS=2` (ring starvation,
  untested here but not worth risking given NS=3 already found) brackets
  the viable range to noticeably around 3-4.

**Final numbers, this campaign's best config:**

| input | plow TTFT | vLLM TTFT | ratio | plow TPOT | vLLM TPOT | ratio |
|---|---|---|---|---|---|---|
| 2048  | 327.00  | 211.73  | 1.545x slower | 32.97 | 23.77 | 1.387x slower |
| 8192  | 1243.86 | 814.25  | 1.528x slower | 33.52 | 23.18 | 1.446x slower |
| 16000 | 2668.36 | 1855.62 | 1.438x slower | 34.26 | 22.28 | 1.538x slower |

**Config**: cubins `assets-run/gemma4-31b-seg-ws384-ns3/` (built with
`PLOW_EXTRA_DEFINES="-DPLOW_NV_FA256_BKV=64 -DPLOW_NV_FA512_BKV=32
-DPGM90_UNI256_NS=3"` added to the Stage 7 recipe), packet
`assets-run/gemma4-31b-seg-ws384-ns3-asset/` (same emit flags as Stage 8).
Serve: same `--pf-seg-dir/--pf-seg-pure 1/--pf-seg-fa512 all` pattern.
**Live on :8090.**

## Stage 10 — searched decode for an analogous hidden-guard bug: none found, 2026-09-04

Given the prefill side's 3-for-3 hit rate finding real hidden-guard bugs
(FATLITE, TMA's two-part flag, W8A8 trapping bf16 WS384), did a structural
sweep of decode's own GEMV path for the same pattern before assuming
decode needs new engineering.

- Mapped every `#if`/`#endif` span in `runtime/nvidia/op_gemm.cuh` (the
  shared, non-Hopper-specific body file decode's dense GEMV bodies live
  in — decode is M=1, no matrix to multiply, so it never touches the
  wgmma-only `op_gemm_sm90.cuh` where the prefill bug lived). Widest span
  found: 81 lines (`PGM90_FORK_GLU`). Nothing remotely like
  `op_gemm_sm90.cuh`'s 1097-line `PLOW_NV_W8A8` trap. **Structurally
  clean — no equivalent bug here.**
- The one `#if PLOW_NV_W8A8` in `op_moe.cuh` is MoE-only, irrelevant to
  this dense model (already established in Stage 2/6).
- Checked `PLOW_DOP_GEMV_QKVG` (a QKV+Gate fused decode opcode spotted in
  the opcode name table) — **dead code**: appears only in the debug name-
  string list (`crates/devgen/src/lib.rs:5839`), zero `DevOp::` emission
  call sites anywhere. Never wired up, not a usable lever.

**Conclusion: decode's remaining ~1.4-1.55x gap is not a hidden-flag bug
like prefill's was — it reflects genuine bandwidth efficiency.** Back-of-
envelope: 57.18 GiB weights / measured decode step time implies plow runs
at roughly 56-58% of H100's realistic HBM3 bandwidth, vLLM at roughly
80-85%. Closing that gap for real needs either profiling tools this
sandbox doesn't have (`ncu`/`cuda-gdb` unavailable, established earlier
this campaign) or the previously-scoped `PLOW_NV_LEAN_DECODE` port (new
CMake target + `exec/gpu.rs` dispatch plumbing, real engineering, and its
best precedent on a different model only reached ~1.32x behind, not
parity) — not a quick find-and-flip.

### Open leads not yet tried

1. `PGM90_TILE_BAND` still untested — its own comment describes an
   occ-2/264-tile rationale that doesn't obviously apply to WS384's
   occ-1/132-block grid, so expected value is lower than `UNI256_NS` was,
   but not zero.
2. The `PLOW_NV_LEAN_DECODE` port (real engineering, sized above) is the
   only concretely-scoped remaining lever for decode.
3. Profiling tools (`ncu`, `cuda-gdb`) would materially help both sides at
   this point — everything found this session was via source-reading and
   A/B measurement, not instrumentation. Worth asking whether this
   sandbox can get either before investing more blind-search effort.
4. **Decode TPOT regression (1.4-1.5x -> 1.55-1.72x) is unexplained** —
   investigate before shipping this config; might be a KV/arena side
   effect of the segmented prefill path worth fixing, or might reverse
   once WS384/chunk-tuning change the object shapes again.

## Stage 11 — final decode search: two more candidates ruled out, session closed, 2026-09-04

Pushed on decode once more after Stage 10 rather than stop at "no bug
found." Two further checks, both genuinely dispositive:

- **`PLOW_NV_FA_KUN=2`** (K-stream pre-issue depth): built decode+prefill
  with it set, compared SASS (`cuobjdump -sass`) byte-for-byte against the
  baseline decode cubin — **identical, zero diff**. Confirmed dead code
  for this model's head dims, not just historically (the in-source comment
  at `op_attention.cuh:502` only documented it as null on the OLD pre-WPR
  access pattern; this is a fresh, direct confirmation it's still null on
  the current FA_WPR-era body).
- **`PLOW_NV_FA_GF_FULL=8`** (px16's documented 1.52x isolated win on
  flash-decode, "register-neutral," on a DIFFERENT model): **structurally
  infeasible here**, not just untested. The packet-side fusion factor is a
  single value applied to EVERY layer's `assert_eq!((heads/kvh) % gf, 0)`
  (`crates/devgen/src/lib.rs:3049-3055`, the same mechanism Stage 6 found
  for GF_FULL=4). This model's sliding layers have GQA=2 (`heads=32/
  kvh=16`); GF=8 does not divide 2, so a GF_FULL=8 packet cannot even be
  emitted — it would panic at `plowc --emit`. px16's win was presumably
  measured on a model whose GQA ratio makes 8 valid for every layer type;
  it does not transfer to Gemma-4-31B's specific head/kv_head shape.

**Considered and deliberately declined**: the full `PLOW_NV_LEAN_DECODE`
segmented-decode port (strip flash out of a lean high-occupancy GEMV
object, dispatch flash to a second object). Assessed the actual scope by
reading `crates/plowrt/src/exec/gpu.rs`'s existing prefill-segmentation
machinery (`SegPf`, touched at 8+ call sites: struct field, load path,
`seg_mode`, the wave-class-segment launch branch, cleanup) — replicating
this for decode needs equivalent new dispatch logic PLUS new decode-side
wave-classing in `crates/packet`/`crates/devgen` (currently `wave_class`'s
segment-slicing logic is entirely prefill-scoped, gated on `!uniseg &&
self.ops.iter().any(FlashPrefill)`, which decode's own ops never satisfy).
**Declined to implement in this session**: this is genuine new systems
engineering on the single hottest, most correctness-sensitive path in the
server (every token of every request goes through decode dispatch — a
bug here means silently wrong output, not just slowness, unlike every
other change this campaign). The best available precedent for the
approach (`perf-data/segmented-decode-26b-h100.md`, a different, MoE
model) only reached ~1.32x behind vLLM even in its projected best case —
not parity — for a full CMake target + dispatch-plumbing investment. Not
a responsible scope to rush.

**Decode's remaining ~1.4-1.55x gap is the honest floor reachable via
source-reading + cubin/packet-flag experimentation in this sandbox.**
Every documented, structurally-valid, register-neutral-or-better lever
has been tried or ruled out with a concrete reason. Further progress needs
either profiling tools (`ncu`/`cuda-gdb`, unavailable all session) or a
dedicated follow-up engineering effort for the segmented-decode port.

## Campaign summary (session start -> end, 2026-09-04)

| metric | session start | final | 
|---|---|---|
| TTFT (prefill) | 4.45-4.92x slower | **1.44-1.55x slower** |
| TPOT (decode) | 1.38-1.54x slower | **1.39-1.54x slower** (net ~unchanged — Stage 6 regressed it, Stage 8 fully recovered it) |

Three real, root-caused bugs fixed on the prefill side (FATLITE register
starvation, TMA's missing packet-side flag, the `PLOW_NV_W8A8` guard
trapping bf16 WS384 bodies), plus two smaller wins (chunk bucketing,
`PLOW_PF_GFUSE` sandwich-norm fusion, `PGM90_UNI256_NS=3`). Decode
investigated exhaustively (tuner sweep, occupancy probe, fusion-flag
audit, hidden-guard search, two more knobs) — genuinely at its floor
without new engineering or profiling tools.

**Final best config, live on :8090 at session end**: cubins
`assets-run/gemma4-31b-seg-ws384-ns3/`, packet
`assets-run/gemma4-31b-seg-ws384-ns3-asset/`, served with `--pf-seg-dir
assets-run/gemma4-31b-seg-ws384-ns3 --pf-seg-pure 1 --pf-seg-fa512 all`
(no `PLOW_UNISEG`, no `PLOW_SEG_CLASS_SLICE`, no `--pf-seg-graph`, no
`--pf-seg-noncoop` — none of those last two helped). vLLM services both
stopped, byte-identical to their pre-session content.

### Verification (every stage, same gate as Phase 0-2)

`libcuda`/live-compute-process check + greedy "Paris" exact match before
trusting any timing; every precision/numerics-adjacent change (fusion,
TMA, chunk-ring changes) additionally checked for exact-match or
documented-expected coherent drift against the pre-change baseline.
`vllm bench serve` sweep, zero failed requests required at every point
recorded in this file. Not committed to git — standing instruction this
session ("dont commit").

## Blockers / notes log

- 2026-09-04: GPU busy elsewhere (user, resolved by stopping a
  session-external `gemma-31b.service` instance) — noted at Phase 3 start,
  no longer relevant.
- This file is long (11 stages). To resume: read the "Campaign summary"
  above for the current best config and numbers, then the relevant Stage
  section for the reasoning/citations behind whichever lever you want to
  revisit. Do not re-run the two Explore-agent research passes from Phase
  3's start — everything load-bearing from them is already captured with
  file:line citations throughout.
