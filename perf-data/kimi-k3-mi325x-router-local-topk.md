# Kimi-K3 MI325X router local top-k

Date: 2026-08-11. Hardware: 8x MI325X, TP8, gfx942. Toolchain: repository
Nix ROCm 7.14.0. Client: vLLM 0.27.0 `bench serve`, one warmup.

## Experiment

K3 router top-k already uses 16 block-wide maximum reductions instead of the
old all-pairs selection. The old reduction rescanned each lane's four expert
keys on every pass. The adopted path loads and sorts those four packed keys
once, then advances only the lane that owned the global winner. Packed keys
include the expert id, so the global ordering and tie break are unchanged.

The option is default-on only for batched K3 decode objects. K3 prefill, B1,
and generic objects retain their existing path. Opt out with
`PLOW_K3_DECODE_ROUTER_LOCAL=0` or the matching CMake option.

## Result

Matched binary-tile-search plus parallel-ALIGN baseline and candidate,
C32/N32, random input 32, output 512:

| seed | control tok/s | candidate tok/s | change | control mean TPOT | candidate mean TPOT |
|---:|---:|---:|---:|---:|---:|
| 0 | 146.024 | 147.157 | +0.78% | 198.818 ms | 198.140 ms |
| 1 | 145.117 | 146.340 | +0.84% | 200.195 ms | 199.356 ms |

The fixed-identical-prompt C32/out128 gate improves 109.171 to 112.321 tok/s.
Its generated-text and input-length arrays are position-wise identical 32/32
against the adopted control. Every cell completed 32/32 requests with the
exact requested output-token count, zero failures, empty error strings, no
in-band error marker, and compact TP counter auditing.

## Static gates

Control and candidate pass the gfx942 and grouped A4W4 ISA audits. Static/GQ
resources remain 256 VGPR, 64,560/64,568 B LDS, and 32 reported spills.
`plow_exec` remains 37,756 instructions, 7,644 SALU instructions, and 673
scratch instructions. The default rebuild is byte-identical to the measured
candidate:

```text
53562abacdd00ed9169aeeac4244278ed07b7ed2e2bbbf2a805a58d362b90115  interp_decode_fp8kv_k3.elf
838b8ebc885a1b9969bc9f9fca0383c859eb690c45050a76db2818f0cda61b18  interp_decode_fp8kv_k3_gq.elf
```

CMake ON/OFF generation proves the define is scoped to batched K3 decode
rows. The build script rejects option values other than zero or one. The K3
devgen suite passes 59/59.

## Reproduction

```bash
nix develop --command env \
  PLOW_DECODE_BATCH=32 PLOW_GEMV_MM=16 PLOW_GEMV_WALK=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 JOBS=2 \
  scripts/build_gfx942.sh /tmp/k3-b32-router-default

nix develop --command env \
  PLOW_DECODE_BATCH=32 PLOW_GEMV_MM=16 PLOW_GEMV_WALK=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 PLOW_K3_DECODE_ROUTER_LOCAL=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 JOBS=2 \
  scripts/build_gfx942.sh /tmp/k3-b32-router-control
```

The server and client settings match
`perf-data/kimi-k3-mi325x-moe-align-prefix.md`, changing only the two primary
decode objects in `PLOW_HSACO`. Raw evidence:

```text
/tmp/k3-router-local-ab/candidate/identical-out128.json
/tmp/k3-router-local-ab/candidate/out512-seed0.json
/tmp/k3-router-local-ab/candidate/out512-seed1.json
/tmp/k3-align-prefix-ab/candidate/identical-out128.json
/tmp/k3-align-prefix-ab/candidate/out512-seed0.json
/tmp/k3-align-prefix-ab/candidate/out512-seed1.json
```
