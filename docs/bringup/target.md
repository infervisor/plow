# The bringup target — parameter block

> Every stage from 4 onward is measured against **one specific GPU**. This file
> defines the named parameters that stand in for that GPU throughout the stage
> playbooks and the agent prompts, and says where each one is *derived from the
> repo* rather than remembered.

Stages 1–3 (nn-graph, egglog, Lean) are target-independent — they compile and
prove, they do not measure. Stages 4–7 are not: a roofline %, a block latency,
a serving recipe and a campaign number are all statements about one part. Fill
this block in **once**, at the start of Stage 4, and carry it forward.

## The block

Copy this into your working notes / the agent prompt and fill every row. A row
you cannot fill is a blocker, not a default.

| parameter | meaning | how to derive it |
|---|---|---|
| `$VENDOR` | `amd` or `nvidia` | `IsaLevel::vendor()` (`crates/hwspec/src/isa.rs`) |
| `$ISA` | the ISA level string the toolchain is given — the `--arch` value | `IsaLevel::arch_flag()`; one of `sm_89`, `sm_90a`, `sm_100a`, `sm_120a`, `gfx942`, `gfx950` |
| `$GPU` | the GPU part name/alias passed to `--gpu` | `plowc --list-gpus` prints every recognized name and alias. **Do not guess one.** |
| `$NCU` | executor (CU/SM) count the blob is compiled for | leave `--n-cu` at its default `0` — it then takes the `--gpu` spec's `sm_count`. Pass it explicitly only to *under*-subscribe deliberately. |
| `$NGPU` / `$PARALLEL` | `--num-gpus` (default `1`) and `--parallel` (default `tp`) | TP is the only wired multi-GPU mode — see `docs/arch/09-multi-gpu.md` |
| `$MAXCTX` | `--max-ctx` the blob is compiled for (default `131072`) | must hold `input + output` of every campaign row, or the request is *refused* (Stage 7) |
| `$TOOLCHAIN` | `hipcc` (AMD) or `nvcc` (NVIDIA), plus its version | required even for GPU-free probing — capability is derived from a built object |
| `$BUILD` | the per-ISA device-object build script | `scripts/build_<isa>.sh` — see the table below |
| `$FEATURES` | `plowrt` cargo feature | `--features hsa` (AMD) or `--features cuda` (NVIDIA) |
| `$BW_BOUND` | the **measured** HBM bandwidth denominator | `MemorySpec::bandwidth_for_bound()` — read the warning below |
| `$COMPUTE_CEIL` | the **measured sustained** matrix-core ceiling | measured on this part, not the datasheet peak — read the warning below |
| `$RESULTS` | where the campaign write-up lands | `perf-data/plow-<isa>/` if one exists for `$ISA`, else `perf-data/` root |

## Per-ISA specifics — point at the authority, do not restate it

Arch divergence is owned by three layers, and
[`docs/arch/14-amd-arch-divergence.md`](../arch/14-amd-arch-divergence.md) §3 is
the authority on which layer owns what. Read it before assuming a difference is
a code difference:

| layer | owns | where |
|---|---|---|
| instruction primitives | which builtin exists (MFMA shapes, fp8 encoding, MX converts) | `runtime/amd/amd_arch.h` |
| budget and shape | LDS bytes, GEMM tile, stage buffers | `hwspec::IsaLevel::geometry()` (`crates/hwspec/src/isa.rs`) — the host source of truth |
| profile | which arms/knobs a shipped object is built with | the per-ISA build script |

Build scripts in the tree today:

| `$ISA` | `$BUILD` |
|---|---|
| `gfx942` | `scripts/build_gfx942.sh` |
| `gfx950` | `scripts/build_gfx950.sh` |
| `sm_120a` | `scripts/build_sm120_cubin.sh` |
| `sm_90a` | `scripts/build_sm90a_cubin.sh` |

`IsaLevel::geometry()` returns `Some` only for the AMD levels, deliberately: the
NVIDIA GEMM tile is one object-wide macro triple that `kernelcaps` probes out of
the header per build, so there is no second copy for a table to disagree with.
On NVIDIA, ask the object (`plowc tune inventory`), not a table.

## Two denominators you must establish before reporting any %

These are the two places where a stale parameter silently flatters the result.

**Bandwidth.** `MemorySpec::bandwidth_for_bound()` returns
`bandwidth_measured` when the registry has it and **falls back to the datasheet
peak when it does not** (`crates/hwspec/src/spec.rs`). On `main` almost every
entry is `bandwidth_measured: None`, so on a new part `bandwidth_for_bound()` is
a datasheet number wearing a measured name. Before quoting a bandwidth roofline
%, either populate the registry entry for `$GPU` from a measurement, or state
explicitly that the denominator is an unmeasured datasheet peak.

**Compute.** The sustained matrix-core ceiling is likewise a per-part
measurement, not a datasheet figure, and the two differ enough to change a
verdict. Establish `$COMPUTE_CEIL` on `$GPU` itself (Stage 4 Step 0) and name
which denominator every reported % used.

> A roofline % is a ratio. Getting the numerator right and the denominator from
> another part is not a measurement of anything.

## Rules

- **Never hardcode a part into a command.** Write `--gpu $GPU --arch $ISA`. A
  script left pointing at another part's `--arch`/`--gpu`/`--n-cu` is a real,
  recurring bring-up failure, not a hypothetical one.
- **A number measured on one part does not transfer to another**, including
  between two parts at the same `$ISA`. Re-anchor.
- **Two parts can share an `$ISA` and differ in memory.** `gfx942` covers both
  MI300X and MI325X; they are the same compute die and differ in the HBM
  subsystem (`crates/hwspec/src/amd/mi300.rs`). Capacity decides which
  experiments are *runnable*; it does not change the emitted code.
- **`$ISA` is metadata for the packet** — `--arch` records the target in
  `build.json` and selects what `--emit devblob+cubin` builds for; it does not
  change an emitted byte on its own.
