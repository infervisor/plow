# Decode lever L3: f_b GEMV folded into KdaStateStepG's prologue — status

Branch `codex/decode-l3-fb-fold` (from `codex/amd-agent-harness` @ e26daf3).
Plan row: `decode-gap-plan-20260904.md` §4 L3. Flag: `PLOW_KDA_FB_FOLD` (emit env / `plowc`
arg, and the decode object's `-D`). Default OFF everywhere; default packet byte-identical.

## What the fold does

In every K3 KDA decode layer the forget-gate up-projection `f_b_proj` (per rank N=1536, K=128,
393 KB, `Gemv` at b=256) is a packet whose only consumer is `KdaStateStepG` (192 WGs at BV=8).
With the fold the step packet carries `t4 = f_a` (`[1, 128]`, was `f_raw`), `j1 = W_fb`
(`[1536, 128]`) and flags bit 2 (`PLOW_KDA_F_FB_FOLD`); each step WG computes its head's 128
gate logits in its prologue and the GEMV packet, its gate and the `f_raw` tensor are gone
(−69 packets/token, 2 fewer packets in the Conv3→f_b→StepG chain's critical path).

Exactness: the prologue computes each column with `gemv_cols_wave` (`runtime/amd/op_gemm.h`),
which is `gemv_rows`/`gemv_rows_r`'s per-column arithmetic verbatim — lane L holds the halves
at `k = chunk*512 + 8L`, the buffer descriptor returns zero past K, `dot8` seeded at `+0.0f`,
chunks accumulated in order, `wave_sum` — then `bf2f(f2bf(t))`, the bf16 the GEMV would have
stored and the step would have re-read. The served f_b arm is `d_gemv_t<1>` staged →
`gemv_rows_rs<1,true,7,2>` (K=128: one chunk, lanes 16..63 contribute an exact `+0.0`), so
the two are bit-identical by construction; the head's 16 tiles recompute the same 128 columns.
Columns are issued `PLOW_KDA_FB_CB` (8) at a time per wave: 16 columns/wave, 2 rounds.

## Implemented

| piece | where |
|---|---|
| device: `gemv_cols_wave<CB>` (under `PLOW_KDA_FB_FOLD && PLOW_BUCKET_DECODE`) | `runtime/amd/op_gemm.h` |
| device: `PLOW_KDA_F_FB_FOLD`, `w_fb` operand on `d_kda_state_step_t`/`_g`, prologue into `lds + 4*D`, `plow_kda_fb_fold_1` marker; forced off on non-decode objects | `runtime/amd/op_kda.h` |
| interpreter arm passes `j1` under the flag bit (flag-on objects only) | `runtime/amd/interp.hip` |
| CMake option `PLOW_KDA_FB_FOLD` (OFF) → `-DPLOW_KDA_FB_FOLD=1` on decode objects | `runtime/CMakeLists.txt` |
| emit flag `kda_fb_fold` (`PLOW_KDA_FB_FOLD`, default false) | `crates/devgen/src/emit_config.rs` |
| K3 mixer: decode + plain Conv3→StepG chain only (not seq-rows, chunk, conv-step-DB or the fused-decode arm) skips the GEMV and `f_raw`, sets `t4/j1/flags` | `crates/devgen/src/kda.rs` |
| slot table / disasm: `j1=w_fb` | `crates/packet/src/dev.rs`, `slots.rs` |
| manifest: `features.kda_fb_fold`, `requires PLOW_KDA_FB_FOLD=1`, `plow_config.h` `PLOW_PACKET_REQUIRES_KDA_FB_FOLD` + `#ifndef PLOW_KDA_FB_FOLD` default from the packet | `crates/devgen/src/manifest.rs` |
| loader: `DECODE_ARM_MARKERS` entry (refuses a folded packet on an object without the arm) | `crates/plowrt/src/exec/amd.rs` |
| tests: `kda::tests::fb_fold_removes_the_f_b_gemv_and_leaves_the_default_packet_byte_identical` (unset == "0" byte-for-byte, prefill untouched), `k3::tests::kda_fb_fold_removes_one_gemv_per_kda_layer` (69 folds, −69 packets), `manifest::tests::kda_fb_fold_is_a_decode_object_requirement` | devgen |
| 1-GPU microbench | `runtime/bench/amd/kda_fb_fold_bench.hip` |

Tune-store note: the store's staleness key is the preprocessed prefill digest, so every new
line is under `#if PLOW_KDA_FB_FOLD` (default 0 in the probe) — `c8_*` census tests stay at
7650/7650 measured. The two `gfx942_*_measurements_reach_the_compiler` failures in
`tuned_tile_selection` are pre-existing on the base branch (stale gfx942 cells).

## Evidence (2026-09-04, MI355X, one GPU, `gpulease -n 1`)

`/tmp/l3/kda_fb_fold_bench 12 128 8 256 64 200` (H=12, D=128, BV=8, GEMV at 256 WGs, step at
192 WGs; 64 iterations of fresh random q/k/v/f_a/W_fb/beta/A_log/dt_bias/state, 200-rep timing):

```
gemv f_b alone (1 launch)                        3.92 us
step alone, gate from f_raw (1 launch)           4.20 us
control: gemv + step (2 launches)                8.14 us
fused: step with f_b fold (1 launch)             8.03 us
exact: 12681216 words checked; fused != control: o 0, state 0;
       control f_raw vs cpu f64 max rel 3.852e-03; status 0
```

* Exactness: over 64 random draws every `o` word and every post-step `state` f32 of the fused
  arm equals the control (GEMV then step); the control's `f_raw` sits within bf16 rounding of
  a double-precision CPU f_b (harness sanity). BV=16 (96 WGs): control 9.06 / fused 8.51 us,
  same 0/0.
* Body: the fold adds ~3.8 us to the step (8.03 vs 4.20) — one head's 128 columns per WG is
  latency-bound (two dependent load rounds + 16 `wave_sum`s per wave), about what the whole
  256-WG GEMV costs standalone. `PLOW_KDA_FB_CB=4` measures the same (8.00 us); CB=16 (one
  load round) 7.91 us for +144 B private scratch (see below) — not worth it, 8 stays the
  default. Final-source rerun: control 8.16 / fused 8.04 us, 0/0 mismatches. No foreign GPU
  context flagged by gpulease on any run.
* In-network projection: the removed cost is the f_b packet body on the critical path (9.3 us
  traced at 1 WG/CU, `decode-gap-plan` §4) + its gate (~1.4–2 us) minus the +3.8 us step
  growth ≈ −7 us/layer × 69 ≈ −0.5 ms/token (plan: −0.6..−0.7). The saving is the packet,
  not the arithmetic; the standalone pair-vs-fused delta (−0.1 us) says so.

Objects (served K3 decode recipe from `/tmp/k3-stack3` build.make, `hipcc_hsaco.sh`,
`plow_interp_dec_gfx950`, served `plow_config.h`):

| object | bytes | VGPR | occ | vgpr spill | sgpr spill | private | LDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| flag off (this tree) | 155664 | 248 | 2 | 0 | 108 | 216 B | 147496 |
| `-DPLOW_KDA_FB_FOLD=1` (CB=8) | 173504 | 248 | 2 | 0 | 108 | 284 B | 147496 |
| fold, `PLOW_KDA_FB_CB=4` | 173504 | 248 | 2 | 0 | 108 | 252 B | 147496 |
| fold, `PLOW_KDA_FB_CB=16` | 174016 | 248 | 2 | 0 | 108 | 428 B | 147496 |

Flag-off object: same size and resources as the L4 status' flag-off row (155664 B / 248 / occ 2
/ spill 0 / 108 / 216 B); all new code is under the flag. Flag-on: VGPR/occupancy/spill
unchanged; +17.8 KB of code (the fold is instantiated in all three D rungs) and +68 B private
segment (scratch for the per-column descriptor/accumulator arrays in the step arm).

## Pending

1. TP8 gate (main session owns the GPUs). Emit with the flag, build the paired objects, serve,
   3 alternating 8192→256 folds against the served bundle; checksum must be
   `fnv1a64:71a28c1449921c95`; expect TPOT −0.4..−0.7 ms and the KDA family −7..−10 us/layer
   in the critpath (KDA_CONV3 → STEP with no GEMV between):

   ```
   docs/k3-mi355x-20260904/scripts/showdown_bundle.sh /tmp/k3-l3 PLOW_KDA_FB_FOLD=1
   ```
   (`showdown_bundle.sh` hard-codes `wt=` to the served worktree; point it at a checkout of this
   branch. The cmake step needs no extra `-D`: `plow_config.h` defaults `PLOW_KDA_FB_FOLD`
   from the packet, decode objects only. `plowrt` refuses the packet on an object without
   `plow_kda_fb_fold_1`.)
2. If the gate passes: default the flag on (`emit_config.rs` `default_value_t`, `env_opt_out`,
   CMake option ON) and record the row in the campaign summary.
3. Body headroom: the +3.8 us is latency, not bytes (32 KB/WG through L2). Options if the
   critpath wants it back: issue the fold's loads before the `parked`/staging prologue, or
   have only tile 0 of each head project and publish through an in-packet per-head flag (the
   L5 rendezvous pattern) — the latter changes nothing numerically.
