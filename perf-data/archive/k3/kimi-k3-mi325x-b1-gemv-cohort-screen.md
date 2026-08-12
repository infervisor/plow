# Kimi-K3 B1 GEMV cohort screen on MI325X

## Decision

Do not promote the tested same-input GEMV cohort fusions to the production packet. The exact
full-grid screen projects only **0.131 ms/token** of savings. This is well below the 5 ms promotion
bar and cannot materially close the current 48.2 ms to 20 ms B1 latency gap.

Keep the harness as a model-free first-stage filter. A literal one-workgroup test is not a valid
performance proxy: it omits the production ownership grid, CU occupancy, weight concurrency, and
global-queue scheduling. The useful unit is one isolated operation or ready-packet cohort run over
the full production grid and exact shapes. Whole-model TP8 serving remains the final adoption gate.

## Scope

The screen covers the largest sets of independent B1 BF16 GEMVs that share one input:

| cohort | logical rows | layers | control packet grids |
|---|---:|---:|---|
| KDA QKVG + `f_a` + `beta` | 6144 + 128 + 12 | 69 | 304 + 128 + 12 |
| MLA q + ckv + krot + gate | 1536 + 512 + 64 + 1536 | 24 | 128 + 128 + 64 + 128 |
| MoE router + latent down | 896 + 3584 | 92 | 128 + 128 |

Control kernels retain separate packet ownership and restage the shared activation at each packet
boundary. Candidate kernels concatenate the rows and stage the activation once. Both use the
production B1 `gemv_rows<1,...,GV_UNROLL=14>` body, K=7168, 512 threads, production-like task
counts, and a 2 GiB rotating weight arena. Candidate grids sweep 128/192/256/304. Every candidate
output is required to be bit-identical to its control.

This is an optimistic fusion ceiling: the candidate stores weights contiguously and excludes
packet/counter integration. Any production realization would retain separate checkpoint tensor
bindings or add addressing metadata.

## Result

MI325X, gfx942, ROCm 7.14, 21 samples, median:

| cohort | repeats | best grid | control us | fused us | speedup |
|---|---:|---:|---:|---:|---:|
| KDA QKVG + `f_a` + `beta` | 17 | 256 | 19.609 | 18.677 | 1.0499x |
| MLA input projections | 30 | 304 | 11.386 | 9.836 | 1.1577x |
| MoE router + latent down | 25 | 256 | 12.657 | 12.337 | 1.0259x |
| layer-count weighted projection | | | 2.791 ms | 2.660 ms | 1.0492x |

Projected saving = **0.131 ms/token**. All three candidates were bit-identical to control.

Resources remain healthy and therefore do not hide a promising candidate: controls use
156/158/156 VGPR, 62--74 SGPR, 16,384 B LDS, and zero private memory; candidates use 154 VGPR,
56 SGPR, 16,384 B LDS, and zero private memory.

## Reproduction

Base commit: `1190b42708b6a7c11b53541f6b7afa05bc2d4530`.

```sh
nix develop --command cmake -S runtime -B /tmp/plow-k3-cohort \
  -DPLOW_ROCM=ON -DPLOW_BENCH=ON -DCMAKE_BUILD_TYPE=Release
nix develop --command cmake --build /tmp/plow-k3-cohort \
  --target k3_gemv_cohort_sweep -j2
nix develop --command env GPU_LEASE_DIR=/tmp/plow-gpulease-shared \
  perf-data/tools/gpulease -n1 k3-gemv-cohort \
  /tmp/plow-k3-cohort/bench/k3_gemv_cohort_sweep \
  /tmp/plow-k3-cohort/bench/k3_gemv_cohort_sweep_gfx942.co 21
```

Toolchain: HIP 7.14.60850, AMD clang 23.0.0git. Artifact SHA256:

- host: `efeb613e4bb82d3580cbddca191a96849dd056d871e2e7972167afc61b852236`
- device bundle: `77e77cc90e1bfac219a0ee26ce31a1f928ab277159d3d6171c4e3461250b2199`

Post-run `gpulease --audit` reported no foreign compute processes.
