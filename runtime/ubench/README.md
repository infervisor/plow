# Plow μBench — Single CU/SM Kernel Profiler

> **Scope**: This directory is STRICTLY for single-CU/SM micro-benchmarks
> (`gridDim=1`). Multi-CU sweeps, library comparisons, and interpreter-level
> benchmarks belong in [`../bench/`](../bench/README.md).

A micro-benchmarker that exercises **each innermost device op** on a single
Compute Unit (AMD) or Streaming Multiprocessor (NVIDIA), measures hardware
counters via `rocprof` (AMD) or in-kernel `clock64` timing (NVIDIA, see
[NVIDIA sm_12x](#nvidia-sm_12x-gb10--sm_121a)), and reports a roofline analysis
showing how close each kernel is to the theoretical single-unit ceiling.

Per-ISA instruction, memory, sync and ceiling reference for the ops measured here:
[`docs/bringup/isa/README.md`](../../docs/bringup/isa/README.md).

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
cd runtime/ubench
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

## NVIDIA sm_12x (GB10 = sm_121a)

| File | Purpose |
|------|---------|
| `ceiling_sm120.cu` | Single-SM ceilings (`mma.sync` bf16 / e4m3 / `kind::f8f6f4` / MXFP8 / NVFP4 / MXFP4 issue rate, FFMA, DRAM via `ld.global.v4`/`.v8`/`cp.async`/`cp.async.bulk`, L2 hit, `ld.shared`) and the harness utilities (fill, L2 flush, snapshot, digest) |
| `fp4_tile_sm120.cu` | Single-SM NVFP4 MMA tile loops: tensor pipe, resident smem, `cp.async` / bulk / TMA 2-D ring feeds (`--fp4-tile`) |
| `bench_cu_sm120.cu` | Includes `interp_sm120.cu`; `ubench_exec` runs packet instructions through the interpreter's own `plow_exec` at gridDim=1 |
| `bench_cu_roofline.c` (`-DPLOW_CUDA`) | Host driver on `../nvidia/cuda_backend.c` (driver API): ceilings, cases, per-slice timing, JSON |

Ops are the packet's **real instructions**, read from `model.pkt` (raw words, real tensor table,
real neighbours), compiled with the packet's `plow_config.h` and the define list plowc used for the
object (`interp_sm120_rs` = MTP verify decode, `interp_sm120_pf` = prefill): the object's
`nvcc <object>.cubin (...)` log line plus the manifest requires/recommends plowc appends as
`PLOW_EXTRA_DEFINES`. The script refuses to build unless that list equals, in order, the production
nvcc command in the emit's `assets/.cubin-build`. A case names an opcode
and a tensor-name substring in one program (T = decode rows or prefill bucket); producers run first
on the full grid so routing tables, NVFP4 q tensors and scan state have the real format. Per case:

* every slice `s` of the instruction's `blocks` runs alone on ONE SM (`slice = s`, `nblk = blocks`),
  L2 flushed before each launch, cycles from `clock64` (no launch overhead); the median per slice,
  the slowest slice (`slice max`) and the whole op serially on one SM;
* full grid vs one SM running all slices, from the same input snapshot: `bitwise` (output
  digests equal), `nondet` (two full-grid runs already differ: atomics), or `MISMATCH`.

Bytes and FLOPs come from `scripts/bench/op_roof.py --ubench` (op_roof's per-instruction formulas
divided by `blocks`), against the single-SM ceilings for the op's bound, and the full-GPU gap
`n x (max(slice, whole-GPU floor) - floor)` per program.

```bash
# inside nix develop; every timed run through scripts/bench/gpuq.py
BUILD_DIR=<scratch>/build runtime/ubench/run_ubench_cu.sh --nv --ceilings --json ceil.json
BUILD_DIR=<scratch>/build runtime/ubench/run_ubench_cu.sh --nv --set nemotron-dec \
    --assets <emit>/assets --build-log <emit>/build.log --object interp_sm120_rs \
    [--case moe_up_routed] --json dec.json            # sets: nemotron-dec nemotron-pf qwen-dec qwen-pf
<scratch>/build/ubench_cu_roofline_nv --list nemotron-dec   # case names
plowrt disasm <emit>/assets > model.disasm
python3 scripts/bench/op_roof.py model.disasm --gpu gb10 --ctx 1024 --kvbytes 1 \
    --sm-ceilings ceil.json --ubench dec.json --ubench pf.json
```

CMake: `-DPLOW_CUDA=ON -DPLOW_UBENCH=ON -DPLOW_CUBIN_ARCH=sm_121a` builds the ceiling cubin and
`ubench_cu_roofline_nv` (`ubench_nv_ceilings` runs them); the ops cubin also needs
`-DPLOW_CUBIN_CONFIG=<assets>/plow_config.h -DPLOW_UBENCH_OBJECT_DEFS="<object defines>"`
(the full production list, unchecked: prefer the script).

GB10 single-SM ceilings (2026-11-24, 2.58 GHz; recorded in `crates/hwspec/src/peak.rs`
`GB10_SM_CEILINGS` and `BENCH_SPEC_GB10`):

| ceiling | one SM | per clock | whole GPU (48 SMs) |
|---|---:|---:|---:|
| `mma.sync` bf16 m16n8k16 | 2.64 TFLOP/s | 1024 FLOP | 126.7 TFLOP/s |
| `mma.sync` e4m3 m16n8k32 (= `kind::f8f6f4` = MXFP8 1X) | 5.28 TFLOP/s | 2048 FLOP | 253.4 TFLOP/s |
| `mma.sync` NVFP4 m16n8k64 `mxf4nvf4` ue4m3 4X (= MXFP4 2X) | 10.56 TFLOP/s | 4096 FLOP | 506.4 TFLOP/s |
| FP32 FFMA | 0.645 TFLOP/s | 250 FLOP | 30.9 TFLOP/s |
| DRAM read (`ld.global.v4`/`.v8`, `cp.async`, `cp.async.bulk` within 6%) | 78 GB/s | 30 B | 239 GB/s (4.9 per SM) |
| L2-hit read | 83 GB/s | 32 B | 2159 GB/s |
| `ld.shared.v4` | 329 GB/s | 128 B | 15.8 TB/s |

One SM can pull a third of the chip's DRAM bandwidth, and its L2 hits are no faster than DRAM: the
per-SM load path (~32 B/clk), not DRAM, bounds a single SM.

### NVFP4 MMA tile loops (`fp4_tile_sm120.cu`, `--fp4-tile`)

The W4A4 prefill MoE walk's k-loop (`nf4v_tiled` in `../nvidia/op_moe_nvfp4_pf.cuh`) reduced to
its parts, at gridDim=1: `mma.sync` m16n8k64 `kind::mxf4nvf4` 4X ue4m3 (SASS `OMMA.SF.16864`), A/B
fragments by `ldmatrix.x4` from the production stage layout (64 B rows, 16 B chunk ^ (row >> 1 & 3),
then 8 B of e4m3 scales per row). One kernel per block tile × mode:

| mode | k-loop |
|---:|---|
| 0 | registers only: the tensor pipe (also `ft_chain*`: one warp, N independent accumulators) |
| 1 / 2 | one resident smem stage, ldmatrix + scale LDS per k64; 2 adds a block barrier per k128 |
| 3 | production `cp.async` ring (3 stages of k128, all threads stage, barrier per step) |
| 4 | as 3, software-pipelined (fragments of the next half before this half's MMAs) |
| 5 / 6 / 7 | ring staging alone / without the 8 B scale copies / mode 3 without scale copies |
| 8 | one thread fills a pre-tiled stage with 4 `cp.async.bulk` copies + mbarrier |
| 11 | A and B tiles by one TMA 2-D box each (`cp.async.bulk.tensor.2d`, u8 `[rows][K/2]` map, box 64 B × BM/BN rows, `SWIZZLE_64B` = the ring's swizzle); scales by `cp.async` |

Layouts replay the operand streams: `up` / `down` / `sdown` = Nemotron routed up (K 1024, A
reused), routed down (K 2688) and shared down (K 5376) row-major weights; `tiled` = one contiguous
block per stage; `l2` = both operands L2-resident.

```bash
BUILD_DIR=<scratch>/build runtime/ubench/run_ubench_cu.sh --nv --fp4-tile [--nk 1024] [--json fp4.json]
```

CMake target `ubench_nv_fp4_tile` does the same. Findings (GB10, nk = 1024):

* `OMMA.SF.16864` issues once per 16 clk per SMSP; two accumulators per warp saturate it (the NOP
  after each OMMA is ptxas pairing, not a hazard).
* From resident smem the production tiles reach 99-100% of 10.556 TF (128×256, 128×192, 128×128,
  64×256); a block barrier per k128 costs 3-8%.
* The `cp.async` ring feed reaches 38-63%. The loss is LSU/MIO contention between eight MMA warps'
  LDGSTS and their LDSM/LDS, not DRAM: an L2-resident operand stream is no faster. Staging alone
  moves 23-25 B/clk; dropping the 8 B scale copies lifts mode 3 from 62% to 69%. Four stages,
  software pipelining and L2 bulk prefetch do not help.
* One thread issuing bulk copies (mode 8, pre-tiled) reaches 77-79% (128×256), 75% (128×192), 70%
  (128×128); 64 B per-row bulk copies reach 4-13%. TMA 2-D (mode 11) needs no pre-tiling: 73% on
  512 B weight rows (128×256), 55-59% on 1344-2688 B rows, where the `cp.async` ring reaches 47-49%.
