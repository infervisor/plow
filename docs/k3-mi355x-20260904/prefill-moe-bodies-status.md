# Prefill MoE bodies: stage-1 register-B K256 body, stage-2 64x128 body (2026-09-04)

Branch `codex/prefill-moe-bodies` (from `codex/amd-agent-harness` 8b2555d). Two exact,
default-off object variants for the lean gfx950 MoE prefill objects, plus the exact-traffic
screen of the stage-2 `part` round trip. Nothing in the packet changes; the flags are object
requests carried by the manifest config header.

## Result (one leased MI355X, T8192 / H3584 / I384 / E896 / top-k 16, 151,232 BM64 rows,
31 order-alternated compute-cache-flushed samples, medians)

| body | before (ms/layer) | after (ms/layer) | delta | FlyDSL/AITER bound |
|---|---:|---:|---:|---:|
| stage-1 A4-reuse GEMM (`plow_moe1_a4_reuse_16x16x128_gfx950`) | 1.4185 | **0.8155** | −0.603 (−42.5%) | 0.616 (`t64x256x256_w3_xcd4`, MXFP8 in / bf16 out, no quant) |
| stage-1 quant/sort (unchanged) | 0.310 | 0.310 | — | (AITER quantizes in a separate fused sort) |
| stage-1 boundary (quant + GEMM) | 1.730 | **1.126** | −0.604 (−34.9%) | — |
| stage-2 down (`plow_moe2_mxfp4_16x16x128_gfx950`) | 0.783 | **0.728** | −0.055 (−7.0%) | 0.170 @ T1024 (AITER stage-2 is an f32-atomic, not fixed-order) |
| combine (unchanged) | 0.327 | 0.327 | — | — |

Per layer the MoE body path goes 1.730 + 0.783 + 0.327 = 2.84 ms → 2.18 ms (−0.66 ms);
92 layers → **−60 ms projected TTFT** on the 955 ms stack-5 served number (boundaries and the
router/align passes are untouched). Stage-1 sits at 1.32x the AITER exact-shape body bound
with the bf16→MXFP4 quant pass still separate (0.31 ms); the remaining stage-1 gap is that pass
(AITER's is fused into its sort) and the 2-wave/SIMD occupancy of the 228-VGPR body.

Exactness: every candidate object matched the shipping object byte-for-byte on the routed
distribution above — stage-1: 0 / 29,036,544 MXFP4 payload bytes and 0 / 1,814,784 E8M0 scale
bytes (pad rows included); stage-2: 0 / 469,762,048 f32 `part` words. Flag-off objects have
the same executable `.text` SHA-256 as before the change (stage-1
`fbaa1a6a…`, stage-2 `0c697ba0…`, the manifest's pinned digest).

## What changed

### Stage-1 (`PLOW_MOE_STAGE1_BODY=1` → `-DPLOW_MOE1_BODY=1`, `runtime/bench/amd/lean_moe_stage1_ref/reuse_kernel.hip`)
Same tile (BM64 / BN128 / four waves, A-only LDS, B in registers), same MFMA sequence per
accumulator (k-steps in the same order, same operands and E8M0 scale words), so the f32
accumulators, the SiTU epilogue and the MXFP4 output are bit-identical. Data movement only:
- iteration = two 128-K steps: each lane's B loads cover a full 128-byte weight row line
  (the K128 body fetched half a line per k-step and refetched the other half after the L1
  had turned over); A tile rows are 128 B with an 8-chunk XOR swizzle; one barrier per two
  k-steps (14 instead of 28);
- E8M0 scales are staged through LDS in the packed lane order the MFMA wants: B scales
  `[branch][group][wave][lane_n][ni]` (one `ds_read_u16` per operand pair instead of four
  scattered 1-byte global loads per lane per k-step), A scales `[group][lane_n][mi]` (one
  `ds_read_b32` instead of four byte loads);
- XCD fold of the grid (`bid = (bid % 8) * (grid / 8) + bid / 8`, bijective): consecutive
  tiles of one expert land on one XCD's L2, so its 1.38 MB of gate/up rows are reused from L2
  instead of MALL. Alone the K256 body measured 0.924 ms; with the fold 0.816 ms.
- Resources: 228 VGPR (was 182), 48 SGPR, 0 spills, 0 private, occupancy 2 waves/SIMD (unchanged:
  182 already pinned 2), dynamic LDS 21,504 B in the mainloop under the existing 32,768 B bridge
  alias. The object gate for this variant is 256 registers / occupancy 2 (`_hs_moe1_maxreg`).
  Forcing 3 waves/SIMD (`-DPLOW_MOE1_WAVES=3`, ≤168 VGPR) spills 90 VGPRs and is rejected by
  the no-spill gate; the two-k-step B stage (64 VGPR live across the MFMA block) is what buys
  the full-line B fetch, so the occupancy axis stays closed at 2.

### Stage-2 (`PLOW_MOE_STAGE2_BODY=1` → `-DPLOW_MOE2_BODY=1`, `runtime/bench/amd/lean_moe_stage2_ref/native_kernel.hip`)
64x128 tiles on the unchanged launch grid (`n_tiles(256) * 2 * tiles64` is read as
`n_tiles(128) * tiles64`), so each 64-row sorted tile streams its expert's down rows once
instead of twice (B L2 traffic 3.2 → 1.6 GB/layer). Each output element sees the same three
K-ordered MFMAs with the same operands and scale words, then the same `acc * gate` store.
A stays in one 4 KiB LDS slot under the 4,352 B contract; row ids/gates alias it after the
mainloop. Resources: 90 VGPR (was 98), 34 SGPR, 0 spills, occupancy 5. Same symbol, ABI,
markers and LDS, so the runtime is untouched. Screened and rejected on the same harness: the
XCD fold (0.823 vs 0.844 same-run, noise) and non-temporal `part` stores (0.916 vs 0.846,
+8%: the write-combining L2 path beats streaming 64-byte row fragments). The flag therefore
carries only the 64x128 body. Stage-1's fold result reproduced on a second lease
(0.821 vs 1.421 ms).

### Plumbing
`PLOW_MOE_STAGE1_BODY` / `PLOW_MOE_STAGE2_BODY` are opt-in `plowc` emit knobs
(`crates/devgen/src/emit_config.rs`). When set, the manifest carries
`objects.lean.moe_stage{1,2}_body.required = true` and `plow_config.h` carries
`#define PLOW_OBJECT_MOE_STAGE{1,2}_BODY 1`; `runtime/CMakeLists.txt` and
`scripts/build_gfx950.sh` read the header and add the object define (stage-1 also lifts the
register gate to 256). Off, the manifest, pairing hash, header and objects are unchanged
(`manifest::tests::moe_body_variants_are_opt_in_object_requests`).

## Stage-2 `part` round trip: screened, no exact reduction

The `part` tensor is `f32[T*topk, H]` = 1.879 GB written by stage-2 and read once by the
fixed-order combine (3.76 GB/layer, ≈0.6 ms at 6.2 TB/s, ≈56 ms over 92 layers). Under the
fixed-slot-order contract (no reassociation) every option was ruled out on bytes:
- token-major partial layout: `part` is already token-major (`[token][slot][H]`); a sorted-row
  layout with a gather in the combine moves the same bytes.
- fused down+combine epilogue: the combine order is `((r+s)+p0)+p1+…+p15`; adding slot j needs
  the prefix through j−1. Expert-sorted tiles produce a token's 16 slots at arbitrary times, so
  the prefix must be materialised — that is the `part` buffer. A last-arriver combine keeps the
  same read bytes (only ≈14% of them can still be in the 256 MB MALL) and adds per-token
  arrival counters and release/acquire ordering; expected ≤ 4 ms total.
- slot-sequential sweeps (acc += slot j for j = 0..15, exact order): 16 f32 `[T,H]` round trips
  = 3.76 GB again, plus 16x the down-weight HBM reads (9.9 GB).
- token-major down compute (all 16 slots of a token in one workgroup): 11 MB of expert rows per
  token, 90 GB/layer; token-block-local sorting to fit a block's `part` in MALL re-reads all
  down weights per block (16 blocks → 9.9 GB).
- bf16 / split `part`, atomics, deterministic tree: not exact / not adopted.
The exact lever left in stage-2 is the body (above); the traffic lever needs the deterministic
tree contract (C2), which this work does not adopt.

## Not done / follow-ups
- Align/scatter passes: after `PLOW_MOE_ALIGN_PAR` the four align packets are 0.21 ms/layer
  isolated (18.5 ms network); the router is on the T/8 seam band. Left as is.
- Stage-1 quant/sort (0.31 ms/layer, 940 MB gather + 288 MB write) is the next stage-1 item;
  fusing it into the sort/align phase the way AITER does would remove most of the remaining gap
  to the 0.616 ms bound.
- Three `devgen` k3 tests (`truncation_works_at_t_rows_too`, `a_prefill_bucket_emits_gemms…`,
  `the_mla_prefill_arm_forces_one_split`) failed once in the full `--lib` run and pass alone and
  on a rerun (303/303); they read MLA knobs without `env_guard` while other tests hold
  `EnvScope`, a pre-existing race unrelated to these flags.

## Reproduce

```sh
nix develop -c runtime/bench/amd/lean_moe_stage1_ref/build_body.sh /tmp/moe1 \
  -DPLOW_MOE1_BODY=1        # MOE1_MAXREG=256; text_sha.sh prints .text digests
GPU_LEASE_TIMEOUT=14400 perf-data/tools/gpulease -n 1 moe1-body \
  /tmp/moe1/body_compare /tmp/moe1/shipping.elf /tmp/moe1/candidate.elf --run
nix develop -c runtime/bench/amd/lean_moe_stage2_ref/build_compare.sh /tmp/moe2 -DPLOW_MOE2_BODY=1
GPU_LEASE_TIMEOUT=14400 perf-data/tools/gpulease -n 1 moe2-body \
  /tmp/moe2/stage2_compare /tmp/moe2/shipping.elf /tmp/moe2/candidate.elf --run
```

TP8 gate (served bundle vs flags, 8192→256, checksum must stay `fnv1a64:71a28c1449921c95`):

```sh
docs/k3-mi355x-20260904/scripts/showdown_bundle.sh /tmp/k3-moe-bodies \
  PLOW_MOE_STAGE1_BODY=1 PLOW_MOE_STAGE2_BODY=1
```
