# Plan — put the DECODE knobs under the tuner, with ctx as a sweep axis

Status: design, not built. Motivated by the 26B/H100 campaign
(`perf-data/gemma26b-h100-gemv-mlp.md`), which tuned these knobs by hand and found the
optimum is **not** a constant.

## Why

`plowc tune` today is scoped to **prefill** kernel selection (`--profile prefill_dense`, the
`weight_tiling` BN/BK search recorded in `tunedb`). Every knob that moved the decode number in
this campaign is a **compile-time define on the decode object**, outside that search space:

| knob | where | measured sensitivity |
|---|---|---|
| `PLOW_NV_FORCE_MINBLK` (+ packet `--n-cu`) | occupancy | **6.196 → 5.746 (7 %)** |
| `GV_MOE_UN` | MoE arms | 6.288 → 6.196 (1.5 %) |
| `GV_UNROLL` / `GV_UNROLL_GLU` | dense + GLU | 6.200 (UN 8) vs 7.055 (UN 2) at occ-1; **inverted at occ-2** (UN 4 beats UN 8) |
| `PLOW_MOE_DOWN_SG` | lane-split | 6.109 (SG 4) vs 6.443 (SG 2) |
| `PLOW_NS_ABS` | packet, flash split | 6.894 (8) vs 6.954 (default 17) |
| `PLOW_NV_FA_GF_FULL` | flash | 9.209 (4) vs 10.846 (8) **at ctx 32k** |

Two facts make this a search problem rather than a constant:

1. **The optimum flips with occupancy.** Deep unroll wins at 1 block/SM and *loses* at 2
   (registers cap at 128 and it spills). A single `#define` cannot serve both.
2. **The optimum depends on op shape.** Lane-split wins on short K (MoE down, I_moe=704) and
   loses on long K (o_proj, K=4096). Row-blocking wins standalone and loses in-context.

## Why ctx must be an axis (not a fixed point)

Decode TPOT is **not** flat in ctx — measured on the shipped object:

| ctx | 1024 | 8192 | 32768 |
|---|---|---|---|
| bf16 ms | 6.196 | 6.935 | 9.209 |

The growth is entirely the 5 full-attention layers, and the two knobs that govern them
(`FA_GF_FULL`, `NS_FULL_ABS`) are exactly the ones whose optimum is ctx-dependent — the
existing `build_sm90a_cubin.sh` comment already records a *different* `NS_FULL_ABS` cliff per
context, and `devgen`'s CU-fill formula bakes `nsplit` from `ctx` at emit time. Tuning at
ctx=1024 only and shipping that is how the current `nsplit=17` default (ragged 136 items on
132 blocks) survived.

## Design

**Key constraint learned the hard way: isolation ≠ in-context.** `gemv_lab_h100.cu` says
row-blocking wins 1.4× on every decode shape; in the megakernel it *loses*. So a microbench-only
tuner would ship the wrong arm. The scoring function must be **end-to-end TPOT**.

```
                 ┌── stage A: prune (cheap, microbench) ────────────┐
  knob grid  ──► │ gemv_lab-style standalone kernels per op shape   │──► top-K per knob
                 └─────────────────────────────────────────────────┘
                 ┌── stage B: confirm (expensive, e2e) ────────────┐
  top-K      ──► │ nvcc decode cubin (~40 s) → step_bench × ctx     │──► tunedb record
                 └─────────────────────────────────────────────────┘
```

- **Stage A** exists already (`runtime/nvidia/experiments/gemv_lab_h100.cu`). Use it only to
  *prune* obviously-bad points, never to pick the winner.
- **Stage B** is the real scorer. One `nvcc` per config (~40 s) + `step_bench` per ctx (~25 s).

**Record key**: `(gpu, arch, dtype, n_cu/occupancy, ctx_bucket, model_shape_hash)`
**Record value**: the winning define-set + measured TPOT + the runner-up margin.

`ctx_bucket` ∈ {1k, 8k, 32k, 128k} — the campaign shows 1k→8k is nearly flat but 8k→32k is
not, so buckets must be geometric, not linear.

**Deliverable shape.** These are compile-time defines, so the artifact is a *set of cubins*,
not a packet field. The runtime already selects a cubin by profile name
(`interp_sm90a.cubin`, `_pf`, `_fp8kv`), so extend that to
`interp_sm90a__occ{N}_ctx{B}.cubin` and have `exec::gpu` pick on (resident n_cu, packet
max_ctx). Packet-side knobs (`NS_ABS`, `NS_FULL_ABS`) stay where they are — emitted per packet,
which already has ctx.

## Cost

Grid: `MINBLK{1,2} × GV_UNROLL{4,8} × GV_MOE_UN{2,4} × SG{4,8} × NS_ABS{8,16}` = 32 configs
× 4 ctx. At ~40 s build + 4×25 s measure ≈ 2.3 min/config ⇒ **~75 min per (gpu, dtype)**.
Cheap enough to run per hardware target; too slow to run per build, which is why it belongs in
`tunedb` as a recorded artifact (and why `compile` must read but never write it — the existing
rule already says this).

## Ordering (do these first)

1. **Occupancy** — the 7 % knob, and it needs a matching `--n-cu` packet, so it must be swept
   as a *pair* `(FORCE_MINBLK, n_cu)`. Note the occ-2 grid carries a bounded one-time ~140 MiB
   VRAM cost (measured NOT a leak) that `gpu_lifecycle`'s 64 MiB tolerance trips on — the
   tolerance should become occupancy-aware before occ-2 is a default.
2. **`NS_ABS` × ctx** — the only knob with a *known* ctx interaction and a known-bad default.
3. `GV_UNROLL`/`GV_MOE_UN` conditioned on the chosen occupancy (they invert with it).
4. `SG` / lane-split per op class, conditioned on K (short-K only).

## What this does NOT fix

The campaign's remaining 1.20-1.28× vs vLLM is **not** a tuning gap — at occ-2 QKV already
runs at 2968 GB/s (91 % of the measured 3269 GB/s ceiling) and the SASS hot loop is ~85 % FFMA.
A tuner recovers the few percent left in the knobs and stops the defaults from rotting as
other arms change (which is exactly what happened to `GV_MOE_UN` this campaign). Closing the
rest needs the structural work listed at the end of the perf-data card.

---

## Extensibility (implemented)

Two axes the design has to survive, both now enforced by tests in `crates/tunedb/src/decode.rs`:

**New op families must not require a schema change.** `DecodeKnobs` began as seven named
fields, and that closed shape could not represent the knobs that arrived later — the
flash-attention family (`PLOW_NV_FA_WPR`, `FA_GF`, `FA_GF_FULL`, `FA_KUN`, `PLOW_NS_FULL_ABS`)
and the fp8 GEMV row-block (`PLOW_NV_FP8_RB`). The sweep script could vary them; the record
could not store them. So knobs now also carry:

- `extra_defines: BTreeMap<String,String>` — compile-time knobs, rendered as `-DNAME=VALUE`
- `extra_emit: BTreeMap<String,String>` — packet-emit knobs, passed to `plowc` as env

kept separate for the same reason `emit_env` is separate from `defines`: they land in different
artifacts and drift apart exactly when written as one string. Both are `serde(default)`, so
rows written before a family existed still load (`knobs_without_extras_still_load`), and a new
family is recordable and rebuildable with no struct change
(`a_new_op_family_rides_the_extra_maps`).

**Hardware: knob values are portable, their spelling is not.** `defines()` emitted nvcc `-D`
unconditionally, which a future AMD/HSA sweep would have silently inherited and used to build
the wrong object. There is now a `Backend` derived from the hardware key
(`nvidia/sm_90a/h100-nvl` → `Nvidia`, `amd/...` → `Hsa`, unknown → `None`), and
`defines_for(backend)` **returns `None`** for a backend whose sweep has not been built rather
than guessing (`a_backend_without_a_sweep_refuses_to_render_flags`). The record `cell` was
already hardware-keyed; this closes the rendering half.

Tests: 34 → **37**.
