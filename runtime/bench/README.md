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
cd runtime && cmake -B build -DPLOW_ROCM=ON -DPLOW_HIP_ARCH=gfx950 -DPLOW_BENCH=ON
cmake --build build --target interp_gemm_bench
./build/bench/interp/interp_gemm_bench
```

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
