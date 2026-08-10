# 12 — Using the tuner

How to ask what a target can execute, how to read the answer, and how to take a
measurement that the database will accept.

Companion docs: `11-tuning-coverage.md` (which families are tunable and what
blocks each), `tuning/README.md` (the measurement-record schema).

---

## What this is, and what it is not

The tuner answers **"which kernel should this op use on this hardware, and on
what evidence"**. It is offline: it runs at build/provisioning time, never in
the serving path.

Two properties are load-bearing, and both constrain how you use it:

- **`compile` reads, `tune` writes.** `plowc compile` may consume qualified
  records; it must never publish one. If the same command could do both, a build
  could calibrate itself against its own output.
- **Capability is derived from a built object, never declared.** There is no
  hand-written kernel table. The same `interp_sm120.cu` becomes eight different
  objects under different `-D` flags, and `interp_sm90a.cu` is that same file
  included with `PLOW_NV_HOPPER=1`, so "which kernels exist" is a question about
  *an object*. Probing therefore needs the vendor toolchain — see
  [Requirements](#requirements).

---

## Requirements

| to do this | you need |
|---|---|
| inspect an NVIDIA target | `nvcc` (preprocessor only — no GPU) |
| inspect an AMD target | `hipcc` (preprocessor only — no GPU) |
| take a measurement | the GPU itself, plus `perf-data/harness/gpulease` |

Probing is preprocessor-only, so you can inspect a target you cannot run.
You cannot inspect a target whose compiler you do not have — that is deliberate,
because the alternative is a declared table that drifts from the object.

Build the CLI with `nix develop --command cargo build -p plowc --bin plowc`.

---

## Quick start

Three subcommands, all read-only.

### 1. What can this target execute?

```console
$ plowc tune --gpu "H100 NVL"
target      : H100 NVL (sm_90a)
fingerprint : nvidia/sm_90a/h100-nvl
profile     : prefill_dense
capabilities: mma_sync=true wgmma=true tcgen05=false tmem=false tma=true mfma=false lanes=32

inventory   : probed from sm_90a-a88f334142bc2f8e
             defines: PLOW_NV_EMBED_SMEM PLOW_NV_FA_GF=2 PLOW_NV_GEMMA PLOW_NV_PREFILL

executable kernels (3):
    8  PLOW_DOP_GEMM                tile 128x128x32     dispatched
   15  PLOW_DOP_GEMM_MED            tile 128x128x32     dispatched
   14  PLOW_DOP_GEMM_SMALL          tile 128x128x32     dispatched

aliases     : opcodes reaching the SAME body. Ranking within a group
              measures dispatch noise, not kernels.
  sm_90a:d_gemm@sm_90a-a88f334142bc2f8e  <-  PLOW_DOP_GEMM, PLOW_DOP_GEMM_MED, PLOW_DOP_GEMM_SMALL

              on NVIDIA the tile is a compile-time macro per interpreter
              object, so the real tuning axis here is which object is
              built, not which opcode is emitted.
```

### 2. Which kernel would a given shape get, and why?

```console
$ plowc tune --gpu "H100 NVL" --shape 4096,4096,4096
op          : dense matmul 4096x4096x4096 (LargeM)
selected    : PLOW_DOP_GEMM (8)
tile        : 128x128x32
rationale   : 3 opcodes share one implementation
              returning the canonical opcode; there is nothing to rank
tier        : portable
fallbacks   : PLOW_DOP_GEMM_SMALL, PLOW_DOP_GEMM_MED
```

### 3. What does the database hold?

```console
$ plowc tune --gpu "H100 NVL" --status
database    : tuning
cell        : nvidia/sm_90a/h100-nvl

no kernel measurements for this cell.
selection will use the analytical model and report tier `portable`.
```

Options: `--gpu` (default `H100 SXM5`), `--profile` (default `prefill_dense`),
`--db` (default `tuning`), `--root` (repo root, for locating interpreter
sources), `--shape M,N,K`, `--status`.

---

## Reading the output

### The `inventory` line is provenance, and it matters

```
inventory   : probed from sm_90a-a88f334142bc2f8e
             defines: PLOW_NV_EMBED_SMEM PLOW_NV_FA_GF=2 PLOW_NV_GEMMA PLOW_NV_PREFILL
```

That hash is a `BuildId` over the ISA, the normalized `-D` set, the toolchain,
and a digest of the interpreter source. **Two objects with different defines are
different builds and get different hashes**, which is what lets a stale record be
detected rather than silently reused. If the defines listed do not match the
object you are actually shipping, everything below the line describes a different
binary.

### The `aliases` block is the most important thing on the page

Opcodes listed together reach **one function body**. Ranking within a group is
measuring dispatch noise; a "winner" from such a comparison will not reproduce.

The example above is the real NVIDIA case: `interp_sm120.cu:524` reads
`case PLOW_DOP_GEMM: case PLOW_DOP_GEMM_MED: case PLOW_DOP_GEMM_SMALL: d_gemm(...)`
— one body, three names. The same three opcodes on gfx950 are three separately
compiled instantiations, and *there* ranking them is real tuning. Run the same
command against `--gpu MI350X` to see the contrast.

This block is derived by parsing the preprocessed dispatch switch, not from a
table, so it covers families nobody has written a table for. Its known blind
spot: a body-level alias, where one kernel *calls* another rather than sharing a
`case` label. AMD's MoE group ops are `k`-loops over the expert ops at default
flags and will not appear here — see `11-tuning-coverage.md`.

### `rationale` and `tier` say how much to trust a selection

| rationale | meaning |
|---|---|
| `measured` | a record matched this hardware **and** this build |
| `analytical cold start` | no matching record; the cost model decided |
| `the only legal candidate` | one kernel was legal; nothing was ranked |
| `N opcodes share one implementation` | an alias group; the canonical opcode is returned |

| tier | meaning |
|---|---|
| `portable` | analytical model only |
| `architecture-seed` | measured on some SKU at this ISA level |
| `sku-calibrated` | measured on this exact SKU |
| `deployment-calibrated` | measured on this deployment under its clock/power policy |

A `portable` tier is not a failure — it is the honest state before any campaign
has run. What matters is that a bundle records which tier it actually used.

### Errors are answers too

```console
$ plowc tune --gpu MI350X
could not derive an inventory for gfx950:
  cannot run hipcc: an inventory is derived from a built object, so probing
  needs the toolchain that builds it
```

This is not a bug. There is no declared fallback, because a declared kernel list
is exactly what drifts from the object being compiled for. Install ROCm, or
inspect from a machine that has it.

An unknown `--gpu` fails rather than defaulting to some other spec, and a
non-positive `--shape` dimension is rejected.

---

## Reading the generated data

Records live under `tuning/<vendor>/<isa>/<sku>/`, keyed by
`HardwareFingerprint::tuning_path()` so two GPUs cannot be mistaken for one
population. Full schema in `tuning/README.md`; the essentials:

**One JSON object per line, appended, never rewritten.** A superseded record
becomes stale rather than disappearing, and rejections are kept with their reason
so a campaign does not rediscover the same dead end.

**Every record has a `state`:**

| state | selectable? | meaning |
|---|---|---|
| `provisional` | no | measured, not yet qualified |
| `qualified` | **yes** | passed every gate — the only selectable state |
| `rejected` | no | measured and refused; `reason` says why |
| `stale` | no | was qualified; an input digest has since moved |

**Every record has `digests`** — `implementation`, `interpreter`, `toolchain`,
`oracle`. These are checked separately, so recompiling one kernel invalidates its
own records and nothing else. When a record is unusable, the tool reports *which*
digest moved, because "stale" is not actionable but "built with cuda-13.0, you
are compiling with cuda-12.4" is.

**Timings are never a single number.** `stats` carries `median_ns`, `p10_ns`,
`p90_ns`, `min_ns`, `samples`. A candidate only beats another if its median
advantage exceeds the jitter in both — a 1 ns edge under 30 ns of spread is not a
result.

The current cell is `tuning/nvidia/sm_90a/h100-nvl/interpreter_calibration.jsonl`,
an interpreter dispatch-floor calibration. Note it is stored **`provisional`**,
with `reason_not_qualified` explaining that a dispatch-floor probe has no
correctness oracle — it is evidence, not a selectable kernel measurement. That is
the state machine working, not a gap.

---

## Taking a measurement

### Always run under the lease

```console
$ perf-data/harness/gpulease <label> <command>
```

`gpulease` audits for foreign compute processes and **exits 76 if the GPU was
contended**. Treat 76 as a *failed measurement*, never a result with a caveat —
`tunedb::measurement_is_trustworthy` accepts only exit 0. This is not
theoretical: the first campaign in `tuning/` needed three attempts before the
audit came back clean, and the two contended runs differed from the clean ones by
~35% on the gate term.

Wrap the **run**, not the build. Compiling needs no GPU.

### A worked example

`runtime/bench/dispatch/interp_dispatch_floor_nv.cu` is the reference for a measurement
done properly — cooperative launch at the resident grid, the interpreter's real
`ld.acquire.gpu`/`red.release.gpu` lowering, warm-up discarded, median plus
percentiles reported, repeated until clean:

```console
$ nvcc -arch=sm_90a -O3 -std=c++17 -w \
    runtime/bench/dispatch/interp_dispatch_floor_nv.cu -o floor_nv
$ perf-data/harness/gpulease floor ./floor_nv 400
```

Repeat until it exits 0, then take the median across clean runs — one clean run
is still one sample of the *run* distribution.

### Before you sweep a macro

Check that it is a knob. A macro the source hard-defines cannot be varied from
the command line: `-D` on it is a redefinition, and the numbers you get back
describe a tile that was never built.

```rust
use kernelcaps::{classify_macro, Sweepable};
let header = std::fs::read_to_string("runtime/nvidia/op_gemm.cuh")?;
assert_eq!(classify_macro(&header, "PGM_BN"), Sweepable::Overridable);
assert_eq!(classify_macro(&header, "PGM_BM"), Sweepable::Fixed);
```

The current picture: the M axis is sweepable on AMD (`-DGM_BM=192` is shipped by
`build_gfx950_qwen.sh`) and **not** on NVIDIA; the K axis is sweepable on
**AMD only** (`GM_BK` became `#ifndef`-guarded with the CDNA3 port). NVIDIA attention tiles are not macros at all — `BQ`/`BKV` are template
literals at five dispatch sites, so sweeping them means editing those sites.

Two couplings to respect: `GV_MM_MAX` moves the register ceiling of *every* arm
in the object, because the interpreter inlines all of them; and MoE grouped GEMM
shares the dense `PGM_*` triple, so sweeping it moves dense prefill too.

---

## Adding a hardware target

1. Add the SKU to `hwspec::registry` if absent.
2. Ensure `IsaLevel::from_spec` maps its compute capability, and that
   `IsaLevel::caps()` states what the ISA can execute. Write capabilities out
   explicitly — capability is **not** monotonic in release order (Hopper has
   `wgmma` and TMA; the newer consumer Blackwell has neither), so "at least as
   new" is not a safe inference and is deliberately not offered.
3. Add an `ObjectRecipe` in `kernelcaps::targets` mirroring the build script —
   the same TU, the same `-D` set. Do not over-specify: a `-D` the source
   *derives* collides with its own definition. The recipe guards catch this
   without a toolchain.
4. Run `plowc tune --gpu "<name>"` and check the inventory matches what the
   build produces.

Steps 1–3 need no hardware and no vendor toolchain; step 4 needs the compiler.

---

## Not implemented yet

Stated plainly so nobody plans around them:

- **`plowc tune` does not run benchmarks.** A per-op harness needs a correctness
  oracle, and the database refuses to qualify a measurement without one, so a
  benchmark subcommand today would emit authoritative-looking numbers that can
  never be selected. `11-tuning-coverage.md` identifies the four harnesses that
  do time-and-validate in one run — all AMD — and the shortest path to a
  qualifying NVIDIA measurement.
- **No gfx950 inventory has been probed.** The pipeline and recipe are in place;
  it needs one run on a ROCm machine.
- **Nothing consumes measurements yet.** `select_kernel` accepts a
  `MeasuredCosts` and the store can serve one, but no caller wires them together
  — so every selection today reports tier `portable`.
- **Bundles carry no tuning provenance.** `Manifest` still records only
  `gpu: String` and `WeightTiling{bn,bk}`.
- **Only dense GEMM is wired**, 3 of 84 opcodes. See `11-tuning-coverage.md` for
  the per-family verdicts.
