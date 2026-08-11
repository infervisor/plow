# Kimi-K3 B1 KDA gated-norm to o-projection screen (rejected)

Date: 2026-08-11. Hardware: one MI325X (`gfx942`). Toolchain: repository Nix ROCm 7.14.0.

## Question

Can the fitted two-workgroup `KdaGatedNorm` packet be absorbed into its sole o-projection
consumer? The candidate reproduces the shipping per-head RMS reduction, sigmoid gate, and BF16
materialization in each GEMV workgroup's LDS before running the unchanged production GEMV row
body. It removes the global intermediate read and one packet boundary.

The full-grid harness rotates eight exact TP8 shapes (`H=12`, `D=128`, `N=7168`, GEMV grid 299)
through 176 MiB of BF16 weights. It compares every staged normalized BF16 byte and every projected
BF16 output byte, then alternates the two-launch control and one-launch candidate for 12 samples.

## Result

| arm | eight layers | projected 69-layer time |
|---|---:|---:|
| fitted gated norm + o-projection | 0.084718 ms | 0.730693 ms |
| fused LDS gated norm + o-projection | 0.075919 ms | 0.654797 ms |
| saving | 0.008800 ms | 0.075896 ms |

The normalized input and final projection are byte-identical (`y_diff=0`, `out_diff=0`). The
candidate is 156 VGPR, 29 SGPR, zero private memory and zero spills. The GEMV control is 159 VGPR,
27 SGPR and also spill-free.

## Decision

STOP. This exact boundary fold projects only 0.076 ms/token. The already-adopted two-workgroup
norm fit removed nearly all exposed cost, so deleting the remaining packet cannot materially move
the B1 target. Do not add a new GEMV mode or run a TP8 serving campaign for this axis.

Raw result: `/tmp/k3-kda-norm-gemv-final.jsonl`.

## Reproduction

```bash
nix develop --command cmake -S runtime -B /tmp/plow-k3-kda-norm-gemv \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-kda-norm-gemv \
  --target k3_kda_norm_gemv_sweep -j2
nix develop --command perf-data/tools/gpulease -n 1 k3-kda-norm-gemv \
  /tmp/plow-k3-kda-norm-gemv/bench/k3_kda_norm_gemv_sweep \
  /tmp/plow-k3-kda-norm-gemv/bench/k3_kda_norm_gemv_sweep_gfx942.co
```
