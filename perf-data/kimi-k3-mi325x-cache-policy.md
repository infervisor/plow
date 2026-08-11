# Kimi-K3 MI325X grouped-weight cache policy

Date: 2026-08-11. Hardware: 8x MI325X (`gfx942`, 304 CU/card).
Toolchain: flake ROCm 7.14.0. Client: flake vLLM 0.27.0 `bench serve`.

## Experiment

The adopted B8 decode object excludes dead standalone MXFP4 projection arms.
The candidate changes only the four grouped-A4W4 expert load sites from cached
`global_load_dwordx4` to `global_load_dwordx4 ... nt`.

- Static/GQ objects retain VGPR=256, LDS=64560/64568, and two spills.
- Object sizes are unchanged.
- The gfx942 assembly audit passes for both arms.
- All runs use compact TP audit, one warmup, C8/N32, and 128 forced output
  tokens through `plowrt serve` and vLLM `bench serve`.

| seed | cached tok/s | NT tok/s | delta |
|---:|---:|---:|---:|
| 0 | 51.710 | 50.486 | -2.37% |
| 1 | 51.470 | 50.291 | -2.29% |
| 2 | 51.078 | 51.174 | +0.19% |

Median: **51.470 -> 50.486 tok/s (-1.91%)**. Every arm completed 32/32
requests and 4096/4096 output tokens. Input lengths and generated text are
identical for each matched seed.

## Verdict

Reject grouped expert NT. Selected expert tiles have useful per-XCD reuse, so
cache retention is more valuable than one-use streaming semantics. Cached
loads remain the default. Dense one-use MXFP4 projection loads remain NT, but
those projection bodies are absent from the adopted B8 decode object.

The next precision experiment is not a cache-policy change. The gfx942 grouped
body currently performs W4A16 gate/up, quantizes only the SiTUv2 intermediate
to A4, then decodes A4 and W4 to BF16 for DOWN BF16 MFMA. Weight requantization
is zero. A true A8W4 FP8-MFMA arm needs explicit K32 E8M0 promotion,
OCP-to-FNUZ correction, and an independent numeric oracle before packet
integration.
