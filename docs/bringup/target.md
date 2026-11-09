# The bringup target — parameter block

Stages 1–3 are target-independent. From Stage 4 on, every roofline %, block latency, recipe and
campaign number is a statement about one part. Fill this block once at the start of Stage 4 and
carry it through every later stage and agent prompt. A row you cannot fill is a blocker, not a
default.

## The block

| parameter | meaning | how to derive it |
|---|---|---|
| `$VENDOR` | `nvidia`, `amd`, `apple` or `intel` | `IsaLevel::vendor()` (`crates/hwspec/src/isa.rs`; `Vendor` in `spec.rs`) |
| `$ISA` | the `--arch` value | `IsaLevel::arch_flag()`: `sm_89`, `sm_90a`, `sm_100a`, `sm_120a`, `gfx942`, `gfx950`, `gfx1201`, `metal3`, `avx512`, `amx`, `cpu` |
| `$GPU` | part name/alias passed to `--gpu` | `plowc --list-gpus`. Do not guess one. |
| `$NCU` | executor (CU/SM/core) count the blob is compiled for | leave `--n-cu` at `0` (takes the spec's count); set it only to under-subscribe deliberately |
| `$NGPU` / `$PARALLEL` | `--num-gpus` (default `1`), `--parallel` (default `tp`) | TP is the only emit-time mode; DP and PD are serve-time (`plowrt serve --dp`, `--pd`, see [06-runtime-opt.md](06-runtime-opt.md)) |
| `$MAXCTX` | `--max-ctx` (default `131072`) | must hold `input + output` of every campaign row, or the request is refused |
| `$TOOLCHAIN` | `nvcc` / `hipcc` + version; host C compiler for CPU; none for Metal | needed even for GPU-free probing: capability is derived from a built object |
| `$BUILD` | how device objects are built | table below |
| `$FEATURES` | `plowrt` cargo feature | `cuda` (NVIDIA), `hsa` (AMD), `metal` (Apple), `cpu` (Intel CPU engine) |
| `$BW_BOUND` | measured HBM bandwidth denominator | `MemorySpec::bandwidth_for_bound()` — see below |
| `$COMPUTE_CEIL` | measured sustained matrix-core ceiling | measured on this part — see below |
| `$RESULTS` | scratch directory for raw campaign output, outside the repo | the repo keeps only `docs/bringup/results/<campaign>/comparison.{md,csv}` |

## Device objects per ISA

| `$ISA` | `$BUILD` |
|---|---|
| `gfx942` | `scripts/build_gfx942.sh` |
| `gfx1201` | `PLOW_HIP_ARCH=gfx1201 scripts/build_gfx942.sh` |
| `gfx950` | `scripts/build_gfx950.sh` |
| `sm_90a` | `scripts/build_sm90a_cubin.sh` |
| `sm_120a` | `scripts/build_sm120_cubin.sh` |
| `sm_89` | `plowc --emit devblob+cubin --arch sm_89` (the recipe's `[emit] emit`) |
| `metal3` | none: shaders compile at load from the packet |
| `amx` / `avx512` | none: CPU kernels build with `plowrt --features cpu` (`runtime/cpu/`, `docs/runtime/cpu.md`) |

A recipe build (`scripts/campaign/campaign.py build`) runs the recipe's object step; see
`scripts/campaign/README.md`.

## Per-ISA authority

Arch divergence has three owners; [`docs/arch/14-amd-arch-divergence.md`](../arch/14-amd-arch-divergence.md)
§3 says which owns what.

| layer | owns | where |
|---|---|---|
| instruction primitives | which builtin exists (MFMA shapes, fp8 encoding, MX converts) | `runtime/amd/amd_arch.h` |
| budget and shape | LDS bytes, GEMM tile, stage buffers | `hwspec::IsaLevel::geometry()` |
| profile | which arms/knobs an object is built with | the per-ISA build script |

`IsaLevel::geometry()` returns `Some` only for AMD levels. The NVIDIA GEMM tile is one object-wide
macro triple that `kernelcaps` probes from the built object: ask `plowc tune inventory`, not a
table.

Read the ISA reference for `$ISA` before Stage 4: [`isa/README.md`](isa/README.md).

## Two denominators to establish before reporting any %

- **Bandwidth.** `MemorySpec::bandwidth_for_bound()` returns `bandwidth_measured` when the
  registry entry has it and falls back to the datasheet `bandwidth` otherwise
  (`crates/hwspec/src/spec.rs`). Check the `$GPU` entry: if `bandwidth_measured` is `None`,
  either add a measured value or state that the denominator is a datasheet peak.
- **Compute.** The sustained matrix-core ceiling is a per-part measurement. Measure it on `$GPU`
  (Stage 4 step 0) and name the denominator every reported % used.

`scripts/bench/calibrate_ceilings.py` (run through `scripts/bench/gpuq.py submit`) measures
streaming bandwidth and dense BF16/FP8 GEMM ceilings into a JSON that `block_roofline.py
--ceilings` and `op_roof.py --ceilings` read in place of registry numbers.

## Rules

- Never hardcode a part in a command: write `--gpu $GPU --arch $ISA`.
- A number measured on one part does not transfer to another, including two parts at the same
  `$ISA`. Re-anchor.
- Two parts can share an `$ISA` and differ in memory (`crates/hwspec/src/amd/mi300.rs`). Capacity
  decides which experiments are runnable; it does not change emitted code.
- `$ISA` is packet metadata: `--arch` records the target in `build.json` and selects what
  `--emit devblob+cubin` builds; it does not change an emitted byte on its own.
