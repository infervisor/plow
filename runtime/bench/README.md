# Plow System-Level Benchmarks

Benchmarks that exercise **multi-CU / whole-GPU / interpreter-level** paths.
For single-CU/SM micro-benchmarks (gridDim=1, roofline analysis), see
[`../ubench/`](../ubench/README.md).

## Directory Layout

| Directory | Purpose |
|-----------|---------|
| `dispatch/` | AQL launch overhead, dispatch protocol, scheduling floor |
| `interp/` | GEMM/op throughput measured *through* the persistent interpreter |
| `gemm/` | Multi-CU GEMM/GEMV sweeps, library comparisons (hipBLASLt, rocBLAS, Tensile) |
| `nvidia/` | NVIDIA-specific ceiling probes (sm_90a / sm_120a) |
| `amd/` | AMD-specific ceiling probes (gfx942 / gfx950) |
| `util/` | Device detection and miscellaneous helpers |

## Relationship to Other Benchmark Areas

```
┌─────────────────────────────────────────────────────────────────┐
│  runtime/ubench/          MICRO: one CU, one op, roofline       │
├─────────────────────────────────────────────────────────────────┤
│  runtime/bench/           SYSTEM: multi-CU, interpreter, libs   │
├─────────────────────────────────────────────────────────────────┤
│  perf-data/               RESULTS: reports, traces, JSON data   │
├─────────────────────────────────────────────────────────────────┤
│  tuning/                  TUNEDB: measured kernel parameters     │
├─────────────────────────────────────────────────────────────────┤
│  scripts/bench_*          ORCHESTRATION: run sweeps, compare     │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Examples

### Interpreter GEMM throughput (AMD)

```bash
nix develop --command cmake -S runtime -B runtime/build -DPLOW_ROCM=ON \
  -DPLOW_HIP_ARCH=gfx950 -DPLOW_BENCH=ON
nix develop --command cmake --build runtime/build --target interp_gemm_bench
nix develop --command runtime/build/bench/interp/interp_gemm_bench
```

### Interpreter MXFP4 GEMM ladder (AMD)

Build before taking the GPU lease. This uses the runtime CU count (304 on MI325X), dispatches
each MXFP4 rung as an interpreter packet, and emits 12 timing samples plus sentinel/f64-oracle
correctness.

```bash
nix develop --command cmake -S runtime -B runtime/build -DPLOW_ROCM=ON \
  -DPLOW_HIP_ARCH=gfx942 -DPLOW_BENCH=ON
nix develop --command cmake --build runtime/build --target interp_mxfp4_bench
nix develop --command bash scripts/build_gfx942.sh /tmp/plow-gfx942-hsaco

nix develop --command env PLOW_STAGE4_CLEARED=1 PLOW_GPU=MI325X \
  PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix PLOW_BUILD_ID=<git-revision> \
  PLOW_LEASE_LABEL=mi325x-mxfp4 PLOW_GEMM_JSONL=/tmp/interp-mxfp4.jsonl \
  perf-data/harness/gpulease -n 1 mi325x-mxfp4 \
  runtime/build/bench/interp/interp_mxfp4_bench \
  /tmp/plow-gfx942-hsaco/interp_prefill_k3_moe_a4w4.elf 4096 4224 7168 k3-tp8
```

The complete Kimi-K3 TP8 shape census uses the production FP8-KV/MXFP4 interpreter,
holds one MI325X lease for the campaign, and publishes qualified rows only after all
96 MXFP4 shapes pass:

```bash
nix develop --command env \
  PLOW_GEMM_JSONL=/tmp/k3-mi325x-mxfp4-full.jsonl \
  scripts/rebench_k3_mxfp4_gfx942.sh
```

### gfx942 fused MXFP4 GLU experiment

After Stages 1–3 pass, this compares production `GemmGluMxfp4` with the counter-gated
production `GemmMxfp4 + GemmMxfp4 + Glu` program. Both arms use the same compiled gfx942 tile
and the runtime CU count. Set `PLOW_GM_BM`, `PLOW_GM_BK`, and `PLOW_GM_DBUF` to the object's
compiled geometry (defaults 192, 64, and 1). The harness requires at least ten samples, a
full-output NaN sentinel check, a sampled f64 oracle, and JSONL provenance.

```bash
nix develop --command cmake --build runtime/build \
  --target interp_mxfp4_glu_gfx942

nix develop --command env PLOW_STAGE4_CLEARED=1 PLOW_GPU=MI325X \
  PLOW_TOOLCHAIN_LABEL=rocm-7.14.0-nix PLOW_BUILD_ID=<git-revision> \
  PLOW_GM_BM=192 PLOW_GM_BK=64 PLOW_GM_DBUF=1 \
  PLOW_LEASE_LABEL=k3-mxfp4-glu PLOW_GEMM_GLU_JSONL=/tmp/k3-mxfp4-glu.jsonl \
  perf-data/harness/gpulease -n 1 k3-mxfp4-glu \
  runtime/build/bench/interp_mxfp4_glu_gfx942 \
  /tmp/plow-gfx942-hsaco/interp_prefill_k3_moe_a4w4.elf 4096 768 7168 15
```

Discard the result when `gpulease` returns 76.

### hipBLASLt comparison

```bash
# needs ROCm + hipBLASLt outside nix
hipcc -O3 -lhipblaslt runtime/bench/gemm/bench_lt.cpp -o bench_lt
./bench_lt 0   # bf16
./bench_lt 1   # fp8
```

### NVIDIA ceiling probes

```bash
nvcc -arch=sm_120a -O3 runtime/bench/nvidia/px7_w8a8_ceiling_bench.cu -o px7_bench
./px7_bench
```

## Build

These benchmarks are **not** part of the default build. Enable with:

```bash
cmake -B build -DPLOW_BENCH=ON [other flags...]
cmake --build build
```

Individual targets are listed in `CMakeLists.txt`.
