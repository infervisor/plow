# Plow μBench — Single CU/SM Kernel Profiler

A micro-benchmarker that exercises **each innermost device op** on a single
Compute Unit (AMD) or Streaming Multiprocessor (NVIDIA), measures hardware
counters via `rocprof` (or `ncu`), and reports a roofline analysis showing how
close each kernel is to the theoretical single-unit ceiling.

## Target Kernels

| Kernel | Family | Bound | Notes |
|--------|--------|-------|-------|
| `d_gemm` | MFMA | Compute | bf16 128×128×64 tile, MFMA 32×32×16 |
| `d_rmsnorm` | Row-reduce | Memory | Wave64 reduction + elementwise |
| `d_headnorm_rope` | Row-reduce | Memory | Per-head norm + half-split RoPE |
| `d_residual` | Elementwise | Memory | Pure BW (2R + 1W) |
| `d_geglu` | Elementwise | ALU | gelu_tanh transcendental |
| `d_embed` | Gather | Memory | Strided table lookup |
| `d_softcap` | Elementwise | ALU | tanh throughput |
| `interp_overhead` | Control | — | Gate/fence/signal machinery cost |

## Quick Start (AMD MI350X)

```bash
cd runtime/bench
./run_ubench_cu.sh                    # full 3-pass rocprof run
./run_ubench_cu.sh --timing-only      # fast: wall-clock timing only
./run_ubench_cu.sh --kernel d_gemm    # profile one kernel
./run_ubench_cu.sh --json out.json    # structured output
```

## Build with CMake

```bash
cd runtime
cmake -B build -DPLOW_ROCM=ON -DPLOW_UBENCH=ON -DPLOW_HIP_ARCH=gfx950
cmake --build build --target ubench       # timing only
cmake --build build --target ubench_full  # with rocprof counters
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  run_ubench_cu.sh                                       │
│  Orchestrates: compile → profiler passes → analysis     │
├──────────────────┬──────────────────────────────────────┤
│ bench_cu_gfx950  │  bench_cu_roofline.c                 │
│ .hip (device)    │  (host driver)                       │
│                  │                                      │
│ bench_gemm()     │  1. Detect GPU, select spec          │
│ bench_rmsnorm()  │  2. Allocate + fill device memory    │
│ bench_residual() │  3. Launch kernel (gridDim=1)        │
│ ...              │  4. Parse rocprof CSV                 │
│                  │  5. Compute roofline metrics          │
│ gridDim=1        │  6. Print terminal table              │
│ blockDim=256     │                                      │
│ (= 1 CU)        │  bench_cu.h (abstraction layer)      │
└──────────────────┴──────────────────────────────────────┘
```

## Key Design Principles

1. **Single CU isolation**: `gridDim=1` guarantees one workgroup on one CU
2. **Same code path**: Benchmarks call the EXACT `d_*` functions the interpreter runs
3. **Iteration loop inside kernel**: Amortizes launch overhead, accumulates counters
4. **Multi-pass profiling**: 3 rocprof passes cover ALU, memory, and LDS counters
5. **Vendor-agnostic header**: `bench_cu.h` defines a portable spec/analysis interface

## Output Example

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║  Single CU/SM Roofline Benchmark — MI350X (gfx950) @ 2.20 GHz              ║
║  Peak: 9.01 bf16 TFLOPS/CU | 31.2 GB/s HBM/CU | Ridge: 289 FLOPs/B       ║
╠═══════════════════════╤════════════╤══════════╤════════════╤═════╤═════╤═════╣
║ Kernel                │ Ach. TFLOP │ Comp %   │ Ach. GB/s  │ Mem%│MFMA%│Bound║
╟───────────────────────┼────────────┼──────────┼────────────┼─────┼─────┼─────╢
║ d_gemm                │ 7.21       │ 80.0%    │ —          │ —   │78.2%│COMP ║
║ d_rmsnorm             │ —          │ —        │ 28.5       │91.2%│ —   │MEM  ║
║ d_residual            │ —          │ —        │ 30.8       │98.6%│ —   │MEM  ║
║ d_geglu               │ —          │ —        │ 24.2       │77.4%│ —   │MEM  ║
╚═══════════════════════╧════════════╧══════════╧════════════╧═════╧═════╧═════╝
```

## Files

| File | Purpose |
|------|---------|
| `bench_cu.h` | Backend-agnostic spec + analysis interface |
| `bench_cu_gfx950.hip` | AMD HIP benchmark wrappers (one per `d_*` op) |
| `bench_cu_roofline.c` | Host driver: launch, parse CSV, print table |
| `bench_cu_counters_pass{1,2,3}.txt` | rocprof PMC counter configs |
| `run_ubench_cu.sh` | Orchestrator script |
| `CMakeLists.txt` | Build integration |

## Extending to NVIDIA

The `bench_cu.h` abstraction layer is designed for multi-vendor use:
- Create `bench_cu_sm120.cu` with `__global__` wrappers calling the sm120 `d_*` ops
- Add CUDA driver API backend in `bench_cu_roofline.c` (guarded by `#ifdef PLOW_CUDA`)
- Replace `rocprof` with `ncu --set full --csv` in the shell script
- The analysis logic (roofline computation, table printing) is shared
