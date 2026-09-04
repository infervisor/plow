# gfx950 MLA-prefill transpose-read screen — 2026-09-04

## Decision

Keep `PLOW_MLA_PF_TR16` default-off pending the full-network gate. At the K3 trace
geometry it is bit-exact to the current V2 body and reduces the isolated body by
16.62% without changing occupancy or introducing scratch/spills.

The change is model-independent. It is selected by gfx950 capability and the generic
`DK`/`DR` V2 template, not by a model name. Candidate objects export
`plow_mla_pf_tr16_arm_1`; controls do not.

## Why this schedule

The current PV stage reconstructs each eight-value MFMA fragment with scalar LDS
reads and packs. The unrolled gfx950 ISA has 256 `ds_read_u16` and 128
`v_perm_b32` instructions in this stage. gfx950's `ds_read_b64_tr_b16` directly
transposes four adjacent LDS rows; 64 transpose reads replace that sequence while
the 64 PV MFMAs and their accumulation order remain unchanged.

Pinned AITER uses the same instruction in its gfx950 FlyDSL KDA kernels and Opus
(`runtime/amd/third_party/aiter_opus/opus/opus.hpp`). gfx942 does not provide the
instruction, so the build and source both reject that architecture.

## Gate

Device: one uncontended MI355X through `perf-data/tools/gpulease -n 1`.
Toolchain: repository Nix ROCm 7.14.0. Harness:
`runtime/bench/amd/run_mla_prefill_8k_sweep.sh`. Shape:
`B=1,T=8192,H=12,DK=512,DR=64,KV=bf16`, grid 256.

Correctness runs before timing:

- candidate `Opart` and `(m,l)` arrays are byte-identical to current V2 over the
  complete shape;
- current V2 vs the validated eight-wave MFMA oracle has maximum normalized-row
  relative error `2.416e-4` (limit `2e-2`).

Twenty-one interleaved, cache-flushed timing samples produced these medians:

| arm | median | relative | projected 24-layer body |
|---|---:|---:|---:|
| current V2+SV | 4006.959 us | 1.000x | 96.167 ms |
| TR16 | 3340.793 us | 0.8338x | 80.179 ms |

Delta: `-666.166 us/layer`, `-15.988 ms` over 24 layers. This is an isolated
projection, not a network result. The current trace charges `108.653 ms` to 24
`FlashMlaPrefill` packets (`perf-data/kimi-k3-mi355x-current-attribution-20260904.md`).

## Resources

Compiler report for the matched harness objects:

| arm | wave/WG | VGPR/AGPR | SGPR | occupancy | LDS | scratch | SGPR/VGPR spill |
|---|---|---:|---:|---:|---:|---:|---:|
| current V2+SV | 64/256 | 256/92 | 57 | 1 | 41,568 B | 0 | 0/0 |
| TR16 | 64/256 | 256/92 | 56 | 1 | 41,472 B | 0 | 0/0 |

Matched production CMake raw objects also build successfully:

| arm | wave/WG | used VGPR/AGPR | metadata SGPR | occupancy | LDS | private | SGPR/VGPR spill |
|---|---|---:|---:|---:|---:|---:|---:|
| current V2+SV | 64/256 | 255/76 | 106 | 1 | 58,368 B | 0 | 17/0 |
| TR16 | 64/256 | 253/76 | 106 | 1 | 58,368 B | 0 | 17/0 |

The SGPR spill count is unchanged and spills into VGPRs; both metadata records report
zero private bytes and zero VGPR spill.

Harness object SHA256:

- current: `d50fc57d124f7058c588046f3430f6c85d8371eb187a46d089596024c40ab710`
- TR16: `e7932734caae2c02df2e08d0c1b40a6ba1382b781d6251bf54e5aa3265dde63d`

Production raw-object SHA256:

- current: `0c39462d3033325421bf54dde354f61f0c9c5a3994dfcf9f4ca0d6439afa4322`
- TR16: `b2f848f7eea007eb7ce13cc82453bde87316eb1d64d6fdf235509b80704d3f30`

## TuneDB/profile scope

The gfx950 attention TuneDB selects decode `nsplit` for exact cells such as
`mla/dk512/dr64/h12/gf4`; it does not select a prefill-body schedule. Dense-GEMM
TuneDB likewise does not cover `FlashMlaPrefill`. TR16 is therefore an isolated
object build knob and does not change packet shape, TuneDB lookup, scratch extent,
or merge policy.

Next gate: hash-stamped current/TR16 object sets with one identical measured-TuneDB
packet, three order-alternated TP8 `8192->256` folds, exact tokens/checksums and
all-rank counters, neutral TPOT, plus a matched raw-trace attribution.
