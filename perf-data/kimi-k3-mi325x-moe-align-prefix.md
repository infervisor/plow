# Kimi-K3 MI325X grouped-MoE ALIGN prefix

Date: 2026-08-11. Hardware: 8x MI325X, TP8, gfx942. Toolchain: repository
Nix ROCm 7.14.0. Client: vLLM 0.27.0 `bench serve`, one warmup.

## Experiment

`MoeAlignPf` histograms routed slots by expert and builds the padded expert
prefix consumed by grouped GLU and DOWN. It must remain one workgroup, but K3
has 896 experts and the old prefix was computed serially by thread zero.

The candidate assigns each of 256 lanes a contiguous expert chunk, scans the
256 chunk totals in LDS, then lets each lane write its chunk's row offsets,
counts, tile prefix, and scatter cursor. Packet ABI, tensor layout, padding,
router output, scatter order, and grouped GEMM bodies are unchanged.

The workgroup prefix is default-on only for batched K3 decode objects. Generic
objects, K3 prefill, and B1 retain the scalar path. Opt out with
`PLOW_K3_DECODE_ALIGN_PAR_PREFIX=0` or the matching CMake option.

## Result

Matched binary-tile-search baseline and candidate, C32/N32, random input 32,
output 512:

| seed | scalar tok/s | parallel tok/s | change | scalar mean TPOT | parallel mean TPOT |
|---:|---:|---:|---:|---:|---:|
| 0 | 144.847 | 146.024 | +0.81% | 200.180 ms | 198.818 ms |
| 1 | 144.116 | 145.117 | +0.69% | 201.621 ms | 200.195 ms |

The fixed-identical-prompt gate also improves 108.517 to 109.171 tok/s
(+0.60%). Its control and candidate `generated_texts` arrays and input lengths
are position-wise identical 32/32. Every cell completed 32/32 requests and its
exact output-token count, with zero vLLM failures, empty error strings, no
in-band error marker, and compact TP counter auditing.

The gain is small because ALIGN is only one part of 92 grouped-MoE layers, but
it repeats across every B32 decode step and wins all three measured cells.

## Static gates

The default and opt-out objects both pass the gfx942 and grouped A4W4 ISA
audits. Static/GQ resources remain 256 VGPR, 64,560/64,568 B LDS, and 32
reported spills. `plow_exec` falls from 37,788 to 37,756 instructions and from
7,676 to 7,644 SALU instructions; scratch instruction count remains 673.

Final default hashes, byte-identical to the measured candidate:

```text
fe78d1674cad4547cad084877e2cc67cac987ec0010bfaaa70b3fc3919bda4c6  interp_decode_fp8kv_k3.elf
595db6ade56520dc2333ba03107d48e75fc6d019059f13be718b73a0a47355f2  interp_decode_fp8kv_k3_gq.elf
```

CMake ON/OFF generation proves the define is applied only to the four batched
K3 decode rows and not to prefill rows. The build script rejects option values
other than zero or one.

## Reproduction

```bash
nix develop --command env \
  PLOW_DECODE_BATCH=32 PLOW_GEMV_MM=16 PLOW_GEMV_WALK=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 JOBS=2 \
  scripts/build_gfx942.sh /tmp/k3-b32-align-default

nix develop --command env \
  PLOW_DECODE_BATCH=32 PLOW_GEMV_MM=16 PLOW_GEMV_WALK=1 \
  PLOW_K3_DECODE_MXFP4_PROJ=0 PLOW_K3_DECODE_ALIGN_PAR_PREFIX=0 \
  PLOW_ROWS_ONLY=interp_decode_fp8kv_k3 JOBS=2 \
  scripts/build_gfx942.sh /tmp/k3-b32-align-control
```

The server/client settings are the same as
`perf-data/kimi-k3-mi325x-moe-tile-search.md`, changing only the two primary
decode objects in `PLOW_HSACO`. Raw evidence:

```text
/tmp/k3-align-prefix-ab/candidate/server.log
/tmp/k3-align-prefix-ab/candidate/identical-out128.json
/tmp/k3-align-prefix-ab/candidate/out512-seed{0,1}.json
/tmp/k3-tile-bsearch-ab/candidate/identical-out128.json
/tmp/k3-tile-bsearch-ab/candidate/out512-seed{0,1}.json
```
