# Decode lever L4: MoeCombine folded into the tagged XReduce publish — status

Branch `codex/decode-l4-combine-publish` (from `codex/amd-agent-harness` @ 986200d).
Plan row: `decode-gap-plan-20260904.md` §4 L4. Flag: `PLOW_XR_COMBINE_FOLD` (emit env /
`plowc` arg, and the decode object's `-D`). Default OFF everywhere.

## What the fold does

K3's latent `MoeCombine` (92/token, `d_moe_combine`, no residual, no shared:
`out[h] = bf16(Σ_{j<16} part[j][h])`, f32 in fixed slot order) only feeds the b=7 tagged
one-shot `XReduce`, whose publish copies that bf16 row into 8-byte tagged words. With the
fold the publishing workgroups run the same slot loop themselves (`d_xreduce_tag_publish_combine`,
`runtime/amd/op_collective.h`) and write the tagged words directly:

* same start value (0.0f), same order (slot 0..15), same f32 accumulate, same `f2bf` — bit-exact
  by construction, no deterministic-tree contract;
* the `MoeCombine` packet and its gate are not emitted (−92 packets/token, 2073 → 1981 decode
  instructions in the flag-on emit); the XReduce packet depends on the DOWN packet(s) directly;
* the plain partial slot (`act.dg_tp`, slot B) is no longer written on the latent seam.

Packet encoding (one-shot `XReduce`, op 24): `t1 = part` (`[k, n]` f32), `i7 = k` (0 = off).
Both are zero/`TENSOR_NONE` in the default packet, so a flag-off packet is byte-identical.

## Implemented

| piece | where |
|---|---|
| device: `d_xreduce_tag_publish_combine`, `cpart/ck` on `d_xreduce_tagged_mega`, `plow_xr_combine_fold_1` marker, `PLOW_XRT_FOLD_MAXK = 16` (all under `#if PLOW_XR_COMBINE_FOLD`) | `runtime/amd/op_collective.h` |
| interpreter arm passes `t1`/`i7` (flag-on objects only) | `runtime/amd/interp.hip` XReduce arm |
| CMake option `PLOW_XR_COMBINE_FOLD` (OFF) → `-DPLOW_XR_COMBINE_FOLD=1` on decode objects | `runtime/CMakeLists.txt` |
| emit flag `xr_combine_fold` (`PLOW_XR_COMBINE_FOLD`, default false) | `crates/devgen/src/emit_config.rs` |
| `emit_xreduce_combine_fold`; K3 decode MoE tail skips `MoeCombine` and emits the folded XReduce (grouped and per-slot chains) | `crates/devgen/src/lib.rs`, `crates/devgen/src/k3.rs` |
| manifest: `features.xr_combine_fold`, `requires PLOW_XR_COMBINE_FOLD=1`, `plow_config.h` `PLOW_PACKET_REQUIRES_XR_COMBINE_FOLD` + `#ifndef PLOW_XR_COMBINE_FOLD` default from the packet (so the packet-paired cmake build picks the arm up with no extra flag) | `crates/devgen/src/manifest.rs` |
| loader: `DECODE_ARM_MARKERS` entry (refuses a folded packet on an object without the arm); tagged-blob contract also requires `t1` present and `k <= 16` | `crates/plowrt/src/exec/amd.rs`, `amd_tp.rs` |
| tests: default packet unchanged / flag-on fold shape (`k3::tests::xr_combine_fold_*`), manifest requirement + header (`manifest::tests::xr_combine_fold_*`) | devgen |
| 1-GPU microbench | `runtime/bench/amd/xr_combine_fold_bench.hip` |

## Evidence (2026-09-04, MI355X, one GPU)

Microbench (`/tmp/xr_combine_fold_bench 3584 16 8 7 200`: 8 simulated ranks on one device, all
56 workgroups in one grid, the tagged protocol unmodified; 200 iterations of random f32 slot
partials in [-4, 4]):

```
control  (combine + copy-publish XReduce, 2 launches): 17.25 us
xreduce  (copy-publish XReduce alone, 1 launch):        6.92 us
fused    (combine-in-publish XReduce, 1 launch):        9.78 us
exact: 5734400 words checked; fused != control 0, fused != cpu ref 0,
       control != cpu ref 0, fused != fused(rerun) 0; status 0
```

* Exactness: every output word of every rank equals the control (production combine then
  copy-publish) and a CPU reference (f32 slot sum → bf16 → f32 rank sum in rank order → bf16),
  and reruns on the same data agree. Earlier run with the first (runtime-`k` loop) body measured
  14.3 µs; staging the 16 slot loads in registers before the adds brought it to 9.8 µs.
* Timing caveat: gpulease flagged a 40 KB foreign context on the leased GPU
  (`/tmp/plow-attnres-decode-mwg/bench`, another agent's L5 run) before and during; numbers were
  stable across three runs (9.73 / 9.78 fused, 6.80 / 6.92 copy-publish).
* Bench-level saving ≈ 7.5 µs/layer on the two-launch model; in the interpreter the removed cost
  is the combine packet body (8.4 µs) + its gate (~2 µs) minus the +2.9 µs publish growth, i.e.
  ≈ −7.5 µs × 92 ≈ −0.7 ms/token projected (plan: −0.5..−0.8).

Objects (served K3 recipe from `/tmp/k3-32k`, `hipcc_hsaco.sh`, `plow_interp_dec_gfx950`):

| object | bytes | VGPR | occ | vgpr spill | sgpr spill | private | LDS |
|---|---:|---:|---:|---:|---:|---:|---:|
| flag off (this tree) | 155664 | 248 | 2 | 0 | 108 | 216 B | 147496 |
| `-DPLOW_XR_COMBINE_FOLD=1` | 162888 | 248 | 2 | 0 | 170 | 216 B | 147496 |
| fold + `PLOW_HAS_MOE_COMBINE=0` (what a flag-on packet's config selects) | 162064 | 248 | 2 | 0 | 170 | 216 B | 147496 |

Flag-off object identity: same-path builds of the generic decode bucket before/after this
change have identical disassembly and symbol tables; the only byte difference is the
`__hip_cuid_*` symbol (source-hash CU id), as for every edit under `runtime/amd/`.

Flag-on emit (`PLOW_XR_COMBINE_FOLD=1 plowc … --emit devblob`, no Lean verify): succeeds,
7650/7650 tiles measured, `features.xr_combine_fold=true`, `requires` carries
`PLOW_XR_COMBINE_FOLD=1`, `plow_config.h` has `PLOW_PACKET_HAS_MOE_COMBINE 0` and
`PLOW_XR_COMBINE_FOLD 1`; 233 decode programs (unchanged count).

## Pending

1. TP8 gate (main session owns the GPUs). Emit with the flag, build the paired objects, serve,
   3 alternating 8192→256 folds against the served bundle; checksum must be
   `fnv1a64:71a28c1449921c95`; expect TPOT −0.5..−0.8 ms and the decode critpath to show
   XREDUCE b=7 body ≤ 11 µs on the latent seam:

   ```
   PLOW_XR_COMBINE_FOLD=1 docs/k3-mi355x-20260904/scripts/showdown_bundle.sh /tmp/k3-l4 \
       PLOW_XR_COMBINE_FOLD=1
   ```
   (`showdown_bundle.sh` hard-codes `wt=` to the served worktree; point it at a checkout of this
   branch. The cmake step needs no extra `-D`: `plow_config.h` defaults
   `PLOW_XR_COMBINE_FOLD` from the packet. `plowrt` refuses the packet on an object without
   `plow_xr_combine_fold_1`.)
2. If the gate passes: default the flag on (`emit_config.rs` `default_value_t`, `env_opt_out`,
   CMake option ON) and record the row in the campaign summary.
3. `sgpr_spill` 108 → 170 on the decode object (VGPR/occupancy/private unchanged): watch the
   per-packet GEMV body in the gate's critpath; if it moved, the staged-load unroll can drop to 8
   slots × 2 rounds.
4. L10 (publish from the DOWN epilogue) now has its target: `d_xreduce_tag_publish_combine` is
   the only reader of `part` on the latent seam.
