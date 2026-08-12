# Kimi-K3 B1 BF16 GEMV R-split fit on MI325X

Date: 2026-08-11

## Decision

Reject production promotion. Keep the standalone shape-fit arm in the bring-up harness.

The adopted WG128 packet gives the hot `N=7168,K=768` shared-down projection exactly seven output
columns per wave. The shipping R2 body exposes only four weight vectors at once because the row has
two K chunks. The candidate processes all seven columns together (R7/UN2), while K1536 shapes use
an R4/R2/R1 ladder and the remaining short rows keep R2.

R7 improves isolated shared-down 2.94x, but the instance-weighted exact-shape GEMV total improves
only 5.109 -> 4.704 ms/token, a 0.405 ms ceiling. That is too small relative to the current
48.232 ms served TPOT and the 20 ms goal to justify another production object and full-model A/B.

## Current trace

The exact KDA-double-buffer plus norm-fit packet has 2,274 decode instructions. A real-weight TP8
trace at ctx5 reports a 46.597 ms device span. Body envelopes overlap, but they identify the work
families:

| family | packets | body ms |
|---|---:|---:|
| ordinary BF16 GEMV | 816 | 22.622 |
| grouped MXFP4 GLU | 92 | 4.291 |
| XReduce | 278 | 4.099 |
| KDA QKVG | 69 | 3.827 |
| grouped MXFP4 down | 92 | 3.709 |

At ctx128K only `FlashMlaDecodeFp8` changes materially, from 0.643 to 7.537 ms of body envelope.
The fixed-shape GEMV body remains 22.4--22.6 ms.

Trace SHA256:

- real ctx5: `3cb2a8033aae9d4d37cd15b67a795d346c9feb147d1919ae482681671df4a231`
- decode-only ctx128K: `89a14100368698f155c7b9943ba58e3cfec56c945ca537af19e0d3d84600729d`

## Full-grid block sweep

The harness uses all 16 current BF16 GEMV shapes, their exact WG128-era grids and packet counts,
1.5 GiB of rotating cold weights, forward/reverse interleave, and 41 timing samples. Every R-fit
output is bitwise equal to the shipping body. The A/A range is 0.9987--1.0025.

| shape | shipping us | R-fit speedup |
|---|---:|---:|
| shared-down 7168x768 | 14.739 | 2.937x |
| q-absorb 6144x1536 | 13.094 | 1.745x |
| o-proj 7168x1536 | 15.065 | 1.360x |
| f-b 1536x128 | 3.524 | 1.252x |
| router 896x7168 | 3.868 | 1.151x |
| routed-up 896x3584 | 3.438 | 1.015x |
| q-rope 768x1536 | 2.844 | 0.740x |

Instance-weighted totals:

| arm | projected ms/token |
|---|---:|
| shipping R2 | 5.109 |
| R4/R2/R1 ladder | 4.735 |
| shape-fit R7/R4/R2 | 4.704 |

The standalone candidate compiles at 248 VGPR, 69 SGPR, 0 private bytes and zero spills. The object
SHA256 is `41010d61d4066c0400bea7f81f955a87d091bf848bdbf2274a9281494d406f6c`.

## Reproduction

```bash
nix develop --command cmake -S runtime -B /tmp/plow-k3-rsplit \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-rsplit \
  --target k3_gemv_rsplit_sweep -j2
nix develop --command env GPU_LEASE_DIR=/tmp/plow-gpulease-shared \
  perf-data/tools/gpulease -n 1 k3-rsplit-fit \
  /tmp/plow-k3-rsplit/bench/k3_gemv_rsplit_sweep \
  /tmp/plow-k3-rsplit/bench/k3_gemv_grid_sweep_gfx942.co 41
```
