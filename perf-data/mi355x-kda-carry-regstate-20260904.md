# gfx950 KDA carry register-resident state — 2026-09-04

Plan item P3 (`plans/k3-beat-vllm-0.28-v3.md` §5). Harness: `runtime/bench/amd/kda_carry_regstate/`
(`nix develop --command runtime/bench/amd/kda_carry_regstate/run.sh`, one GPU under
`gpulease -n 1`). Shape: dense `T8192,H12,D128,V128,BT64`, q pre-scaled (`PLOW_KDA_CHUNK_QPRE`),
21 order-rotated samples per arm. Toolchain `rocm-7.14.0-nix` (clang 23), MI355X.

## 1. Where the 2.2 ms/layer goes

`s_memtime` stamps inside the shipping body (`d_kda_chunk_carry_bt64<true>`, wave 0, mean over
96 workgroups x 128 chunks): 35,327 cycles per chunk = 8,700 (V'/from-state products) +
8,097 (output product) + 17,899 (key factors + state update) + 489 (three barriers). At
2.36 GHz that is 15 µs per chunk, 1.92 ms per layer.

The ISA explains the split: each per-element bounds predicate (`token < mv`, `s + j < valid`)
is compiled as an exec-masked branch around one `global_load_{ushort,dword}` followed by
`s_waitcnt vmcnt(0)` — 42 `vmcnt(0)` waits per chunk over 78 load instructions. U (4 loads),
Aqk (16 loads), k and g (32 loads) each pay a full memory round trip, serialized; the four
`d0` blocks of the state product add four more. Roughly 40 dependent round trips at
300-400 ns each is the chunk time. MFMA (14 issues, ~450 cycles), LDS state traffic
(~48 KB per chunk, < 400 LDS cycles), and exp2 (20 per lane) are minor. The idle waves 4-7,
the LDS-resident state, and the barriers are not the floor; the load chain is.

## 2. Register-resident carry (`runtime/amd/op_kda_carry_regstate.h`)

Four waves, WG256, one (head, V16) item per workgroup (96 workgroups). Wave `w` owns token
rows `[16w, 16w+16)` for the V'/output products and state columns `[32w, 32w+32)` for the
update, held as two f32 MFMA accumulators across the chunk loop. Mechanisms, all exact:

- Update MFMA with swapped operand roles (`A = K-factor^T`, `B = V'`): identical products,
  K order, hi/lo split and accumulator, transposed so a lane holds four consecutive `d` of one
  `v` — the `[v][d]` bf16 snapshot layout. Verified bit-exact (the MFMA reduction is the same
  for every output element).
- bf16 state snapshot `[16][136]` and V' `[16][72]` in LDS (`ds_write_b64` /
  `ds_read_b128`), two barriers per chunk; `from_state` stays in registers across barrier 1.
- Scaled-key factors `K exp2(g_last - g)` built one chunk ahead in row layout (lane = token
  row, coalesced 16-byte loads, `readlane` broadcast of `g_last`), split hi/lo with the
  branch-free RNE, staged in per-wave LDS tiles `[64][36]` and read transposed with
  immediate-offset `ds_read_u16`.
- All other chunk factors (W, q, U, Aqk, decay) loaded for chunk c+1 during chunk c from
  clamped row addresses and masked after the load: no bounds branch around any load.
- `#pragma clang fp contract(off)` plus an explicit `fmaf` for `S * decay + upd`: the
  shipping backend fuses exactly that update (`v_pk_fma_f32`) and nothing else.

Codegen hazards found on the way (both would silently break exactness): extracting odd
elements of a loaded `bf16x8` (`__bf16` ext-vector) through `__builtin_bit_cast` produced
wrong values — the key path now loads `u32x4` and unpacks halves; and the default contraction
fused `scaled - bf2f(high)` when contraction was left to the backend.

| arm | median ms | vs control | VGPR / SGPR / LDS | spill |
| --- | ---: | ---: | --- | --- |
| control V16/WG512 | 1.916 | -- | 204 / 68 / 14,336 B dyn | none |
| regstate V16/WG256 | **0.726** | **-62.1% (2.64x)** | 235 / 53 / 43,520 B static | none, private 0 |

Oracle: 0 / 12,582,912 BF16 output and 0 / 196,608 FP32 state mismatches at T=8192; also
0 / 0 at T=64 and at T=8191 (63-row tail chunk); Aqk unchanged. Isolated saving
1.19 ms/layer → 82 ms over 69 layers (the network carry is 2.2 ms/layer, so the network
saving should be ≥ that if the family object's carry behaves like the harness control).

Per-chunk attribution of the candidate (12,949 cycles): products 567 + 622, loads+barrier 1
2,366, state update 1,575, next-chunk key factors 6,989 (54%), barrier 2 568. The remaining
cost is VALU in the key-factor computation (32 `exp2` and 64 RNE splits per lane per chunk,
~26 VALU per element), which every one of the eight V-tile workgroups of a head recomputes.

Hardware-convert arm (`v_cvt_pk_bf16_f32` for every bf16 rounding, `d_kda_chunk_carry_bt64_regstate<false, true>`):
**0.572 ms (3.36x, -70.2%)**, VGPR 238 / SGPR 48, no spill, and bit-exact on the same
oracle (0 mismatches, 21-sample run: control 1.918 / regstate 0.724 / hwcvt 0.572 ms). It is
kept bench-only: the hardware convert agrees with `f2bf` on every finite and infinite input
(both RNE) but may differ in the NaN payload it produces, and the route contract here is
bit-exact for all inputs. Promoting it is a one-line template argument in
`kda_chunk_carry_regstate.hip` once a NaN-payload difference is accepted (a NaN state is
already a poisoned request).

## 3. Route (opt-in, default off)

- Emit: `PLOW_KDA_CARRY_REGSTATE=1` (devgen `EmitConfig::kda_carry_regstate`, gfx950 + qpre
  only) isolates the exact `KdaChunkWu -> KdaChunkCarry` pair like the key-factor route and
  marks the carry singleton with `SE_KDA_CARRY_REGSTATE`. `flags` has no free bit, so this is
  the wave-item bit (`8`) disambiguated by opcode; a runtime that predates the route refuses
  the packet as an impure wave-item segment. Manifest: `objects.lean.kda_carry_regstate.required`
  and `#define PLOW_PACKET_REQUIRES_KDA_CARRY_REGSTATE 1` in the build config header.
- Object: `runtime/amd/kda_chunk_carry_regstate.hip` →
  `kda_chunk_carry_regstate_gfx950.elf` (CMake `PLOW_HSACO_KDA_CARRY_REGSTATE` or
  manifest-required through `PLOW_HSACO_CONFIG`; `scripts/build_gfx950.sh` mirror), built with
  `-DPLOW_NO_SPILL=1 -DPLOW_NO_SGPR_SPILL=1`, gate `256 1`. Measured through
  `hipcc_hsaco.sh` with a stamped config: VGPR 229, SGPR 52, 0 spills, private 0,
  LDS 43,520, kernarg 88 (+256 COv5), markers `plow_kda_carry_regstate_{abi_1,
  bt64_d128_v128_qpre_1, wave64_1, no_spill_1, static_lds_43520, vgpr_le_256}` + packet hash.
- Runtime: `derive_segments` class 23 for pure marked carry segments,
  `PrefillSegmentRoute::KdaChunkCarryRegstate` (grid `heads * 8`, WG256), fail-closed load
  (object present, all markers, packet-pairing stamp, kernarg/LDS/private gate), dispatch ahead
  of the key-factor and family routes; `rebase_kda_key_factor_routes` carries `t` for chunked
  prefill. Runtime rollback is impossible by construction (a marked packet requires the object);
  the rollback is the emission flag.

## 4. TP8 A/B (not run here; TP8 network benchmarks are outside this task)

Same recipe as the qpre gate (`perf-data/kimi-k3-mi355x-kda-qpre-bf16-20260903.md`): exact
8192-token prompt, 1 output token, C1, BF16 KV, `--amd-kda-family-route=true`, promoted
P1/P2/P5 settings, three order-alternated folds under one eight-GPU lease. The only delta is
the emission flag and the packet-paired object set:

```sh
# control emit (K3_FULL=1 PLOW_FP8_KV=0 PLOW_SEG_PACKED_PREFILL=1 ... as in the qpre gate)
# candidate emit: identical env plus
PLOW_KDA_CARRY_REGSTATE=1 K3_FULL=1 PLOW_FP8_KV=0 PLOW_SEG_PACKED_PREFILL=1 \
  PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix ./target/release/plowc --hf-dir <k3_farm> \
  --emit devblob --arch gfx950 --gpu mi350 --num-gpus 8 --parallel tp --max-ctx 16384 \
  --n-cu 256 --out <assets-candidate>
# objects, per arm, paired to that arm's packet (candidate manifest requires the carry object):
cmake -S runtime -B <build-arm> -DPLOW_HSACO_ARCH=gfx950 \
  -DPLOW_HSACO_CONFIG=<assets-arm>/plow_config.h <the qpre-gate object options> && \
  cmake --build <build-arm> --target hsaco
# folds (alternate c,p,p,c,c,p), each under one lease:
perf-data/tools/gpulease -n 8 k3-kda-carry-regstate-fold <exact 8192->1 bench of the qpre gate> \
  --hsaco <build-arm>/hsaco   # candidate object dir must contain kda_chunk_carry_regstate_gfx950.elf
```

Gate: token-identical output across arms and ranks; TTFT delta from three folds; expect the
`KdaChunkCarry` family body to fall by ~1.2 ms/layer (~80 ms) if the isolated result transfers.

## 5. Follow-up with the largest remaining lever

54% of the candidate's chunk time is the scaled-key computation, recomputed by the eight
V-tile workgroups of each head. The existing key-factor Wu object already emits exactly those
hi/lo factors once per (chunk, head) into the 50 MB scratch pair; a regstate carry that reads
them (row-layout 16-byte loads straight into the LDS tiles, no exp2/split) would drop the
carry to roughly 6-7k cycles per chunk (~0.4 ms/layer) for ~0.1 ms/layer of extra Wu work.
