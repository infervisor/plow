# gfx950 MLA-prefill transpose-read screen — 2026-09-04

## Decision

Enable `PLOW_MLA_PF_TR16` by default for gfx950 HSACO builds, with
`-DPLOW_MLA_PF_TR16=OFF` as the rollback. At the K3 trace geometry it is bit-exact
to the current V2 body, reduces the isolated body by 16.62%, and reduces the matched
TP8 traced body by 16.34% without changing occupancy or introducing scratch/spills.

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

## TuneDB/profile scope

The gfx950 attention TuneDB selects decode `nsplit` for exact cells such as
`mla/dk512/dr64/h12/gf4`; it does not select a prefill-body schedule. Dense-GEMM
TuneDB likewise does not cover `FlashMlaPrefill`. TR16 is therefore an isolated
object build knob and does not change packet shape, TuneDB lookup, scratch extent,
or merge policy.

## TP8 network qualification

Both arms used packet
`f1bf783dac96791b7116ffb549862c8206ba33351310c7c113504916611e8921`,
the same checkpoint and TuneDB, BF16 KV (`fp8_dir: None`), and the default
segment-major TP prefill schedule. The HSA runtime was built from this isolated tree
and has SHA256
`754ee0fbc51374f32c87b649ecc370868fa2397e838926a465573eeddf69682b`.
Every measured request enabled per-dispatch TP audit and reported completion on all
eight ranks.

The complete object directories contain 37 objects each. Only
`interp_mla_v2_sv{,_gq}.elf` differ. Object SHA256:

- current static: `040a9f2a52d821445dee20a7d9a05265d998fe2eacd0ba2e4894c5b6b85a65bb`
- current GQ: `c21c0572a56abf73355a1c1ce3221685a43917f617e6de685c5f03e33ebf4b2e`
- TR16 static: `ce1fc55bdfbb4e6cf0d0ce5343e9d97a150e099c7f2b64e042e5782a41bd53f6`
- TR16 GQ: `6ff9e108fc5db33ad4fb7e355052b2fc0d20512fcdd2de44a16d35c45a145b5c`

The current object has 256 `ds_read_u16`, 128 `v_perm_b32`, and no transpose
reads. The candidate has 64 `ds_read_b64_tr_b16`, no old reads/packs, and alone
exports `plow_mla_pf_tr16_arm_1`.

### Exact single-request folds

Three order-alternated `8192->256`, warmup-zero processes produced byte-identical
256-token arrays and checksum `fnv1a64:6bdfaa7b84ee4e7e` in every arm. They also
exposed process-start/tail instability and are retained here rather than filtered:

| fold/order | current TTFT/TPOT/E2E ms | TR16 TTFT/TPOT/E2E ms | TR16 delta ms |
|---|---:|---:|---:|
| 1 C/T | 1403.295 / 29.978 / 9047.652 | 1386.743 / 30.599 / 9189.428 | -16.552 / +0.621 / +141.776 |
| 2 T/C | 3088.215 / 29.934 / 10721.358 | 2834.663 / 29.936 / 10468.231 | -253.552 / +0.002 / -253.127 |
| 3 C/T | 1403.693 / 29.962 / 9044.120 | 1888.796 / 29.925 / 9519.698 | +485.103 / -0.037 / +475.578 |

Fold 1's TPOT increase is one 199.648 ms inter-token stall; its TR16 median/p90
are 29.931/29.959 ms versus current 29.975/30.005 ms. The unfiltered three-fold
mean deltas are TTFT `+71.666`, TPOT `+0.195`, E2E `+121.409` ms, so this noisy
warmup-zero gate alone does not qualify the change.

### Persistent-engine gate

Two order-balanced processes per arm used one warmup plus five measured sequential
C1 `8192->256` requests. Aggregate checksum `fnv1a64:7b56ea918df2c0ba` matched.

| pair/order | TTFT delta mean | TPOT delta mean | E2E delta mean |
|---|---:|---:|---:|
| 1 C/T | -18.719 ms | -0.0130 ms | -22.023 ms |
| 2 T/C | -19.005 ms | +0.0189 ms | -14.179 ms |
| balanced mean | **-18.862 ms** | **+0.0030 ms** | **-18.101 ms** |

TPOT changes by 0.010%, while TTFT and served E2E improve in both orders. This is
the production qualification used for the default decision.

### Matched trace

The exact one-token trace pair attributes the network delta directly to the changed
body:

| metric | current | TR16 | delta |
|---|---:|---:|---:|
| TTFT | 1403.875 ms | 1385.977 ms | -17.897 ms |
| `FLASH_MLA_PREFILL` body, 24 packets | 108.045 ms | 90.390 ms | -17.656 ms (-16.34%) |
| total traced body | 1025.275 ms | 1006.925 ms | -18.350 ms |
| wide-packet straggler spread | 211.562 ms | 203.532 ms | -8.030 ms |
| wide-packet convergence | 46.906 ms | 47.011 ms | +0.105 ms |

The collective convergence term is neutral; the improvement is in the MLA body and
reduced workgroup spread. Raw evidence is in
`/tmp/plow-mla-tr16-network/segment-major`.
