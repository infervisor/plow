# MI355X f32-mix AttnRes phase object (2026-09-04)

Plan item: `plans/k3-beat-vllm-0.28-v3.md` §2 C3 / §4 Q0. Supersedes the rejected interpreter arm in
`mi355x-attnres-f32-mix-norm-20260904.md` (42 SGPR spills, +7.25%).

Decision: **object built, oracle-qualified, routed behind a default-off flag; TP8 A/B left to the
parent session.** No default changes; `PLOW_ATTNRES_F32MIX` unset emits byte-identical packets.

## What it is

`runtime/amd/attn_res_f32mix_gfx950.hip` — `plow_attn_res_f32mix_gfx950`, one token per 4-wave
workgroup, 4 columns x 7 vectors per lane, persistent over a 768-workgroup grid. Semantics are
pinned vLLM 0.28 (`vllm/models/kimi_k3/amd/ops/attn_res.py`, BLOCK_L = 1):

* `prefix = bf16(res_pre + bf16(res_a + res_b))`, stored (the materialized tensor other packets read);
* snapshot push `ring[push_row] = push_src` (the mix reads rows `[0, nb)`);
* scores on the BF16 sources, `s_r = sum(x*w) * rsqrt(sum(x*x)/H + eps)`;
* online softmax mix in f32 in source order (ring rows, prefix last), `mixed *= 1/denom`;
* output RMSNorm on the **unrounded** f32 mix with its own `out_eps`, then one BF16 rounding.

Fixed reduction order: 28 sequential FMAs per lane, a DPP row-shift tree per wave, wave partials
through the same tree. No LDS row staging; `score_w` is staged in LDS once per workgroup.

Packet contract: an `AttnRes` whose `f[1]` (output-norm epsilon) is finite and > 0. Only
`emit_attn_res` under `PLOW_ATTNRES_F32MIX=1` writes it; the interpreter ignores `f[1]`, so an old
runtime runs the packet on its BF16-seam arm instead of failing. The eligible packet also needs
`gamma` (fused norm), `T >= 256`, `HID == 7168`, `nb <= 8 <= nb_cap`. Materialized residual inputs
(`t6/t7`, optional `i5`) are consumed by the object.

## Resources (`hipcc_hsaco.sh`, contract 168 total regs / occupancy 3)

```
{"kernel": "plow_attn_res_f32mix_gfx950", "object_bytes": 18240, "vgpr": 133, "agpr": 0,
 "total_registers": 133, "sgpr": 63, "occupancy_waves_per_simd": 3, "vgpr_spill": 0,
 "sgpr_spill": 0, "private_segment_bytes": 0, "group_segment_bytes": 28736,
 "wavefront_size": 64, "max_workgroup_size": 256, "accepted": true}
```

The 128-VGPR target is missed by 5: at a 128 budget the same body spills 2 VGPRs. The alternative
shape `-DAR_THREADS=448u -DAR_VEC=8u -DAR_DEPTH=2 -DAR_MIN_WAVES=4` is 98 VGPRs / occupancy 4 /
0 spills and 8% slower (table below). `-fno-slp-vectorize` is required: packed-f32 FMA formation
costs ~30 VGPRs of operand shuffling (128 VGPRs + 14 spills with it, 103 without, same body).

## Oracle (`runtime/bench/amd/attnres_f32mix/oracle.hip`, one leased GPU)

Reference: CPU port of `attn_res.py` in f32 with double row sums, plus an ordering-free double
reference. Three GPU arms: the served `.elf` via `hipModuleLoad` with the plowrt kernarg ABI, the
same body compiled with an f32 side output, and production `d_attn_res` (BF16 seam) as control.
Synthetic magnitudes: prefix/ring/push ~ U(-2, 2), delta ~ U(-0.35, 0.35), norm ~ U(0.75, 1.25),
proj ~ U(-0.05, 0.05), gamma ~ U(0.5, 1.5); `eps = out_eps = 1e-5`.

| case | f32 vs vLLM port relL2 | max abs (f32) | BF16 out max abs | BF16 exact | control (BF16 seam) vs port relL2 |
|---|---:|---:|---:|---:|---:|
| T1024 nb=0 res | 6.3e-8 | 7.2e-7 | 0.0156 | 99.9994% | 1.7e-5 |
| T1024 nb=1 res | 1.3e-7 | 2.1e-6 | 0.0156 | 99.991% | 2.70e-3 |
| T1024 nb=4 res | 1.8e-7 | 2.5e-6 | 0.0156 | 99.984% | 2.80e-3 |
| T1024 nb=8 res | 2.1e-7 | 3.2e-6 | 0.0156 | 99.981% | 2.82e-3 |
| T1024 nb=8 res+pre | 2.1e-7 | 3.2e-6 | 0.0156 | 99.981% | 2.82e-3 |
| T8192 nb=8 res | 2.1e-7 | 3.7e-6 | 0.0156 | 99.981% | 2.82e-3 |

All 16 T1024 cases (nb 0/1/4/8 x {res, push, res+pre, raw mix}) and 4 T8192 nb=8 cases PASS:
f32 relL2 <= 2.1e-7 (gate 1e-6), materialized prefix and pushed ring row bit-exact, repeat launches
byte-identical, object BF16 output byte-identical to the probe body. The remaining BF16 max abs
0.0156 is one BF16 ULP at |x| in [2, 4) on 0.02% of elements. The control column reproduces the
documented 2.9e-3 seam delta of the production BF16-seam fusion.

## Timing (T8192, nb=8, res_a+res_b materialization, fused norm; one leased GPU, four alternating folds)

| arm | ms | note |
|---|---:|---|
| streaming floor (same reads/writes, no barriers) | 0.221 | 1.41 GB, ~6.4 TB/s |
| interpreter body `d_materialize_residual + d_attn_res` (256 WG x 512, 147 KB LDS) | 0.589 | |
| object, 448x8 depth 2, 98 VGPR occ 4, grid 1024 | 0.282 | |
| **object, 256x4 depth 1, 133 VGPR occ 3, grid 768** | **0.260** | shipped default |
| object, 256x4 depth 2, 147 VGPR occ 3, grid 2048 | 0.263 | |

0.260 ms vs the target <= 0.25: 4% short, 18% above the streaming floor; -56% vs the interpreter
body with materialization, -45% vs the 0.47 ms interpreter figure without it. The remaining gap is
the per-row barrier chain (nb+2 barriers per token); the DPP reduction took it from 0.307 to 0.287,
the 4-wave shape (three tokens per CU) from 0.287 to 0.260.

## Reproduce

```
nix develop --command bash runtime/cmake/hipcc_hsaco.sh hipcc $ROCM_PATH/lib/llvm/bin/clang-offload-bundler \
  gfx950 /tmp/arf32/attn_res_f32mix_gfx950.elf plow_attn_res_f32mix_gfx950 168 3 -fno-slp-vectorize \
  -I runtime/amd -I runtime/common -DPLOW_LEAN_OBJECT=1 -DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1 \
  -DPLOW_REQUIRED_MARKER=plow_attn_res_f32mix_abi_1 runtime/amd/attn_res_f32mix_gfx950.hip
nix develop --command hipcc --offload-arch=gfx950 -O3 -w -fno-slp-vectorize -DPLOW_K3=1 \
  -I runtime/amd -I runtime/common runtime/bench/amd/attnres_f32mix/oracle.hip -o /tmp/arf32/oracle
perf-data/tools/gpulease -n 1 attnres-oracle /tmp/arf32/oracle --elf /tmp/arf32/attn_res_f32mix_gfx950.elf --t 1024
perf-data/tools/gpulease -n 1 attnres-timing /tmp/arf32/oracle --elf /tmp/arf32/attn_res_f32mix_gfx950.elf --t 8192 --time
```

## Route

* emit: `PLOW_ATTNRES_F32MIX=1` → `emit_attn_res` writes `f[1] = eps`; `Builder` isolates each such
  packet in its own wave class (25); manifest `objects.lean.attn_res_f32mix.required` and
  `#define PLOW_PACKET_REQUIRES_ATTN_RES_F32MIX 1` in the config header.
* build: `PLOW_HSACO_ATTN_RES_F32MIX=ON` or the config define (cmake), `PLOW_ATTNRES_F32MIX=1` or the
  config define (`scripts/build_gfx950.sh`); both require `PLOW_HSACO_CONFIG` for the pairing stamp.
* runtime: `PrefillSegmentRoute::AttnResF32Mix`; fail-closed on a missing object, missing markers,
  missing pairing stamp, kernarg size, private segment, a marked packet in a mixed segment, or a
  non-gfx950 device. Grid 768 (`PLOW_ATTNRES_F32MIX_GRID` overrides). Decode (`T < 256`) and the
  XReduce-folded AttnRes stay on their current arms.

## TP8 A/B (not run here)

Same packet build twice, interpreter control vs object candidate, order-alternated:

```
# control: unchanged emit + objects
# candidate:
PLOW_ATTNRES_F32MIX=1 <devgen emit as for the current K3 TP8 candidate>   # marks + isolates AttnRes
PLOW_HSACO_CONFIG=<candidate>/config.h PLOW_ATTNRES_F32MIX=1 scripts/build_gfx950.sh   # adds attn_res_f32mix_gfx950.elf
perf-data/tools/gpulease -n 8 attnres-f32mix-ab <the usual 8192->1 fold gate, 3 folds, order-alternated>
```

Expect: 186 prefill AttnRes segments on the object (`seg_launches` name `attn_res_f32mix`),
~-0.3 ms/site (~-55 ms at T8192), seam relL2 vs vLLM captures 2.9e-3 → ~1e-7 at every AttnRes
boundary, decode TPOT neutral (decode is untouched).
