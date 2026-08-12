# Kimi-K3 B1 dense MXFP4 full-grid screen (rejected)

Date: 2026-08-11. Hardware: one MI325X (`gfx942`). Toolchain: repository Nix ROCm 7.14.0.

## Question

Can converting K3's 722 live dense BF16 GEMV packets to checkpoint-layout MXFP4 remove at least
5 ms/token from the fixed B1 decode body?

The existing full-grid K3 GEMV screen now runs the production `d_gemv_mxfp4` body alongside BF16,
block E4M3, and per-row E4M3. It covers all 16 emitted B1 shapes at their production grids, walks
cold weight arenas, and weights each shape by its packet count. MXFP4 uses packed low-nibble-first
E2M1 weights plus one E8M0 byte per K32 block, exactly matching the production packet layout.

## Result

Nine-sample final run:

| weighted 722-packet body | time | delta from BF16 |
|---|---:|---:|
| BF16 | 6.343 ms | -- |
| block E4M3 | 5.807 ms | -0.535 ms |
| per-row E4M3 | 6.960 ms | +0.618 ms |
| MXFP4 | 6.533 ms | +0.190 ms |

The BF16 A/A weighted ratio is 0.9998. MXFP4 wins the 163840×7168 lm-head and several narrow-K
shapes, but loses the repeated K=7168 projections that dominate K3. The packed format moves only
`0.53125 byte/weight`, yet gfx942 must software-decode E2M1+E8M0 before the dot product; this path
is VALU-bound rather than weight-bandwidth-bound.

The synthetic `N=257,K=7168` quantization oracle passes against the exact quantized MX values:
relative L2 `0.001617`. Quantization drift against the BF16-weight reference is `0.143053`, cosine
`0.990131`. This is sufficient to validate the screen but would require a full model quality gate
if performance had won.

Resource metadata for `k_mxfp4`: 248 VGPR, 97 SGPR, 16,384 B LDS, zero private memory and zero
spills. The body is already near the interpreter's register ceiling before composition.

## Reproduction

```bash
nix develop --command cmake -S runtime -B /tmp/plow-k3-mx-screen \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-mx-screen \
  --target k3_gemv_w8a16_sweep -j2

nix develop --command env GPU_LEASE_NGPU=1 \
  PLOW_K3_W8A16_JSONL=/tmp/k3-gemv-mx-screen-final.jsonl \
  perf-data/tools/gpulease -n 1 k3-gemv-mx-screen \
  /tmp/plow-k3-mx-screen/bench/k3_gemv_w8a16_sweep \
  /tmp/plow-k3-mx-screen/bench/k3_gemv_w8a16_sweep_gfx942.co 9
```

Raw output: `/tmp/k3-gemv-mx-screen-final.txt`.

## Decision

STOP. Do not quantize K3's dense BF16 tensors or add dense MXFP4 packet substitutions. The exact
body screen regresses before paying interpreter, packet, checkpoint-conversion, and quality costs.
