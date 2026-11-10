# ISA references for kernel authors

One file per instruction-set level plow builds kernels for. Read the file for your target
**before writing or changing a kernel, inline asm, intrinsics or a low-level C/CUDA/HIP/Metal
body for it** (kernel tuning, single-block sweep, rung campaigns). Each file is a distilled,
decision-oriented reference: execution model, matrix instructions, memory movement,
synchronization, toolchain flags as plow passes them, pitfalls already measured in plow, roofline
numbers (datasheet vs calibrated) and links to the vendor documents.

Facts carry their source. `[measured in plow: <ref>]` = a commit, doc or recipe in this repo;
`[vendor documented: <doc>]` = a public vendor document; `(UNVERIFIED)` = not confirmed against a
primary source, treat as a hypothesis.

## Target → file

| `IsaLevel` (`crates/hwspec/src/isa.rs`) | arch flag | SKUs plow ships | reference | recipes | kernels | build |
|---|---|---|---|---|---|---|
| `Sm90a` | `sm_90a` | H100 SXM | [sm90a.md](sm90a.md) | `recipes/infervisor/*/sm90a-*` | `runtime/nvidia/*sm90*`, `interp_sm90a*.cu` | `scripts/build_sm90a_cubin.sh` |
| `Sm89` | `sm_89` | L4, L40S | [sm89.md](sm89.md) | `recipes/infervisor/*/sm89-*` | `runtime/nvidia/interp_sm89.cu`, shared `op_*.cuh` | `plowc --emit devblob+cubin --arch sm_89` → `runtime/CMakeLists.txt` (`PLOW_SM89_CUBIN`) |
| `Sm120a` | `sm_120a` | RTX 5090, RTX PRO 6000 Blackwell | [sm120.md](sm120.md) | `recipes/infervisor/*/sm120a-*` | `runtime/nvidia/*sm120*`, `register_rtx6000.cu` | `scripts/build_sm120_cubin.sh` |
| `Gfx942` | `gfx942` | MI300X, MI325X | [gfx942.md](gfx942.md) | `recipes/infervisor/*/gfx942-*` | `runtime/amd/` | `scripts/build_gfx942.sh` |
| `Gfx950` | `gfx950` | MI350X, MI355X | [gfx950.md](gfx950.md) | `recipes/infervisor/*/gfx950-*` | `runtime/amd/*gfx950*` + shared `runtime/amd/` | `scripts/build_gfx950.sh` |
| `Gfx1201` | `gfx1201` | Radeon AI PRO R9700 | [gfx1201.md](gfx1201.md) | none | `runtime/amd/` (`PLOW_GFX12` arms) | `PLOW_HIP_ARCH=gfx1201 scripts/build_gfx942.sh` |
| `Metal3` | `metal3` | Apple M4 Pro | [apple-gpu.md](apple-gpu.md) | `recipes/infervisor/*/metal3-*` | `runtime/apple/*.metal` | MSL source compiled at load by plowrt (no offline step) |
| `Amx`, `Avx512`, `CpuRef` | `amx`, `avx512`, `cpu` | Xeon 6 (Granite Rapids); avx512 tier also EPYC Zen 4 | [x86-avx512-amx.md](x86-avx512-amx.md) | `recipes/infervisor/*/xeon6-*` | `runtime/cpu/dev/{amx,avx512}` | `crates/plowrt/build.rs` |

`Sm100a` (B200, `tcgen05`/TMEM) exists in `IsaLevel` but no plow recipe ships it; it has no
reference file yet. Do not use `sm120.md` for it: the two levels share a product name, not
instructions.

## Where the numbers live

* Datasheet capability per SKU: `GpuSpec` in `crates/hwspec/src/{nvidia,amd,apple,intel}/`.
  `MemorySpec::bandwidth_for_bound()` falls back to the datasheet peak when no measured value is
  registered ([target.md](../target.md) "Two denominators").
* Instruction availability: `IsaLevel::caps()`; AMD LDS/tile budget: `IsaLevel::geometry()`.
* AMD instruction primitives: `runtime/amd/amd_arch.h`; layer ownership:
  [docs/arch/14-amd-arch-divergence.md](../../arch/14-amd-arch-divergence.md) §3.
* Calibrated ceilings: `scripts/bench/calibrate_ceilings.py` (one short lease per box) feeding
  `scripts/campaign/block_roofline.py`; some campaign recipes under `scripts/campaign/recipes/` carry a `[roofline]` table (no production
  recipe does today); each ISA file lists where its calibrated numbers actually live. Kernel-level
  tuning records: `tuning/<vendor>/` keyed by hardware fingerprint.

## Using these files in the harness

1. Fill the parameter block in [target.md](../target.md); `$ISA` names the file to read here.
2. Before a kernel change, read the target file's checklist and pitfalls, then follow the
   single-block HARD RULE in [agent-tools.md](../agent-tools.md) and the native-kernel inner-loop
   contract in [05-single-block-sweep.md](../05-single-block-sweep.md#native-kernel-inner-loop-contract).
3. A new pitfall measured in a campaign goes into the target file's pitfall list with its commit
   or doc reference, in the same change that fixes it.
4. These files describe hardware. Policy (gates, routes, recipes) stays in the bringup docs they
   link to; do not duplicate it here.
