# 10 — Implementation Status

> This document separates **what exists in the current checkout** from target
> design. Component existence is not blanket support: declared, compiled,
> GPU-correct, model-verified, and tuned are distinct qualification levels.
> Hardware/model claims must name the attained level and scorecard cell.

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Complete — implemented, tested, passing CI |
| 🔧 | Partial — core logic exists, edge cases or tests incomplete |
| 🔲 | Planned — designed in docs, not yet implemented |

---

## Workspace Overview

**Build system:** Cargo workspace, 13 member crates (`nn-graph`, `hwspec`, `kernelcaps`, `tunedb`, `costmodel`, `devgen`, `rewrite`, `schedule`, `packet`, `plowc`, `lean_verify`, `plow-asset`, `plowrt`)  
**Language:** Rust (compiler/host), C/CUDA/HIP (device runtime)  
**Formal verification:** Lean 4 (optional, feature-gated)  
**CI target:** `cargo test --workspace` + CMake for runtime

---

## Crate-by-Crate Status

### `crates/nn-graph/` — Frontend / Model Graph IR

The former `frontend` crate is folded into `nn-graph`: model resolution lives
behind its `hub` feature and architecture builders behind its `models` feature.

| Component | Status | Notes |
|-----------|--------|-------|
| HuggingFace config fetch | ✅ | `nn_graph::hub::build_from_pretrained` via `hf-hub` with pure-Rust TLS |
| Symbolic graph IR | ✅ | `nn_graph::Graph`: build, symbolic shape inference, bind B/S/L |
| Model architectures | ✅ | `nn_graph::models`: Gemma 2/3/4, Llama 2/3, Qwen 2/2.5, DeepSeek v2/v3 |
| Shape bucket specialization | ✅ | `nn_graph::models::ShapeBucket` + `Bindings` |

### `crates/rewrite/` — Egglog Rewriting

| Component | Status | Notes |
|-----------|--------|-------|
| Lower nn-graph → egglog | ✅ | `lower.rs` |
| Fusion rules (egglog) | ✅ | gemm_silu, norm_gemm, swiglu, flash fusions |
| Saturation engine | ✅ | Bounded iteration + node limits |
| Cost-based extraction | ✅ | Custom extractor using costmodel |
| TileGraph assembly | ✅ | `tilegraph.rs`: DmaIn/Compute/DmaOut generation |
| Multi-unit partitioning | ✅ | N-axis split with Join nodes |
| Tile selection (explore) | ✅ | `explore.rs` + egglog selection pass |
| Collapse pass (inline DMA) | ✅ | `collapse.rs`: fold boundary DMAs into kernel |
| Cross-block pipelining | ✅ | All blocks in one plan |
| ConstraintSet generation | ✅ | Handoffs, colocation, dedup, locality |
| RelaxableHandoff cost deltas | ✅ | Per-handoff alternative costs |
| L2Local handoff kind | ✅ | Per-partition residency |
| DSM handoff kind | ✅ | Distributed shared memory |

### `crates/costmodel/` — Hardware Cost Oracle

| Component | Status | Notes |
|-----------|--------|-------|
| GEMM candidate enumeration | ✅ | Legal MMA multiples + SRAM filter |
| GEMM cost function | ✅ | max(compute, memory) model |
| Flash attention candidates | ✅ | bq × bkv enumeration |
| Flash cost function | ✅ | Per-block MMA + load overlap |
| Row-op cost | ✅ | Memory-bound model |
| Split-K support | ✅ | Decode batch-1 parallelization |
| TMEM filtering (Blackwell) | ✅ | `tcgen05` accumulator check |
| Kernel reservation bytes | ✅ | Per-arch interpreter overhead |
| Dominance pruning | ✅ | Pareto-optimal candidates only |
| SoC partitioning | ✅ | `partition_n` proportional to throughput |
| SRAM page model | ✅ | `sram.rs` with optile integration |

### `crates/kernelcaps/` — Kernel Capability Registry

| Component | Status | Notes |
|-----------|--------|-------|
| `KernelSpec` + capability filtering | ✅ | What a built interpreter object can actually execute |
| Alias detection (`alias_groups`) | ✅ | Reports opcodes that share one body (NVIDIA GEMM/MED/SMALL) |
| Resource gates (`resource`) | 🔧 | Parses both vendors' register/spill reports; NVIDIA build not yet wired to fail |
| Inventory probe from built object | 🔧 | Derives inventory from a built object; gfx950 awaits a ROCm probe run |
| `select_kernel` | 🔧 | `gemma4::pick_tile` routes through it; other model pickers not yet migrated |

### `crates/tunedb/` — Calibrated Measurement Store

| Component | Status | Notes |
|-----------|--------|-------|
| Normalized store + states | ✅ | Records keyed below serving-level (GPU/shape/tile/kernel) |
| Atomic qualified publication | ✅ | Negative/stale records retained for provenance, not selected |
| `NetworkBlockDefinition` → manifest | 🔲 | Block-driven tune manifest is open work |

### `crates/devgen/` — Legacy Device-Blob Emitter

| Component | Status | Notes |
|-----------|--------|-------|
| `gemma4` HF checkpoint → device packet program | 🔧 | **Deprecated**; superseded by `plowc --hf-dir`. Built only with `--features legacy-gemma-bins`, slated for removal |

### `crates/hwspec/` — Hardware Registry

| Component | Status | Notes |
|-----------|--------|-------|
| NVIDIA H100 | ✅ | SM 9.0, 132 SMs, 80GB HBM3 (NVL variant: 94GB HBM3e) |
| NVIDIA B200 (Blackwell datacenter) | ✅ | SM **10.0**, TMEM/`tcgen05`. The previous "SM 12.0" here was wrong |
| NVIDIA RTX 5090 / 6000 Pro (Blackwell consumer) | ✅ | SM 12.0, `mma.sync`, **no** TMEM, **no** `wgmma` |
| NVIDIA RTX 6000 Ada | ✅ | SM 8.9 |
| ISA level vs `Arch` | 🔧 | `costmodel::Arch::Blackwell` conflates SM100 and SM120; `hwspec::IsaLevel` (in `hwspec::isa`) separates them. `costmodel` still branches on `Arch` (`tile.rs`, `mma.rs`, `lib.rs`) |
| AMD MI300X | ✅ | gfx942, 304 CUs, 192GB HBM3 |
| AMD MI350X | ✅ | gfx950 |
| Arch-specific MMA specs | ✅ | Per-instruction throughput/latency |

### `crates/schedule/` — Scheduler

| Component | Status | Notes |
|-----------|--------|-------|
| IntervalSet (exclusive) | ✅ | Sorted Vec, earliest_free, reserve |
| BandwidthSet (capacity) | ✅ | Event-sweep peak, capacity_ok |
| PagePool (SRAM allocation) | ✅ | Per-SM, working-set + output pages |
| Machine model | ✅ | From Soc + Config |
| TaskGraph expansion | ✅ | TileNode → Task with durations |
| Critical-path priority | ✅ | Longest-path-to-sink heuristic |
| List scheduler (Pass A+C) | ✅ | Placement + ordering fused |
| Counter building (Pass D+E) | ✅ | Clustering + scope assignment |
| Relax pass | ✅ | Greedy SRAM demotion |
| Counter elimination (§8.1) | ✅ | Resource-order redundancy |
| Scope narrowing (§8.2) | ✅ | IntraGpu → IntraSm downgrade |
| Prefetch hoisting (§8.3) | ✅ | DMA reordering |
| SRAM temporal fit (§8.5) | ✅ | Phase 3 greedy re-promotion |
| Packet emission | ✅ | TaskGraph → binary .pkt stream |
| Simulation (sim.rs) | ✅ | Counter-gated replay, makespan |
| Verification (verify.rs) | ✅ | Rust-side counter protocol check |
| Spill reporting | ✅ | Per-tile spill diagnostics |
| TMEM slot allocation | ✅ | Blackwell accumulator columns |
| Locality domain pinning | ✅ | DSM / L2-partition aware placement |
| Bucket compilation | ✅ | Multi-bucket per model |
| Weight tiling | ✅ | M-independent layout computation |
| Memory allocation | ✅ | Arena planning, address map |
| Chrome trace export | ✅ | `trace.rs` for visualization |

### `crates/packet/` — Wire Format

| Component | Status | Notes |
|-----------|--------|-------|
| Opcode encoding (structured u16) | ✅ | Family/variant/flags |
| All body types | ✅ | Dma, Gemm, Flash, Row, Token, Layout, Rdma |
| Program encode/decode | ✅ | `to_bytes()` / `decode()` |
| Counter table | ✅ | id + threshold + scope |
| Stream header (v5) | ✅ | Magic, version, bucket_id, plan_gen |
| 4-byte alignment | ✅ | All records aligned |
| Validation on decode | ✅ | Magic, version, overflow checks |
| Round-trip tests | ✅ | Property-based + sample programs |

### `crates/plow-asset/` — Shared Schema

| Component | Status | Notes |
|-----------|--------|-------|
| Manifest | ✅ | Model metadata for runtime |
| MemoryMap | ✅ | Per-device buffer layout + validate() |
| KvPaging | ✅ | Page-table KV cache schema |
| RequestIo | ✅ | Per-request buffer descriptions |
| Blocks / Experts schemas | ✅ | PP and EP metadata |
| DecodeFlashOp / DecodeKvSchema | ✅ | Decode-phase patching |
| BufClass / BufKind / Access | ✅ | Buffer classification enums |

### `crates/plowc/` — Compiler Driver

| Component | Status | Notes |
|-----------|--------|-------|
| `compile()` entry point | ✅ | Full pipeline: source → report |
| Shape bucketing | ✅ | Prefill + decode buckets |
| Emit streams | ✅ | Per-bucket .pkt + sidecars |
| inject_sample_packet | ✅ | Token sampling instruction injection |
| inject_tokenize_packet | ✅ | Tokenization instruction injection |
| KV growable entry injection | ✅ | Decode-phase KV growth |
| SRAM fit phase 2/3 | ✅ | apply_sram_fit_phase2 |
| Lean verify dispatch | ✅ | All 6 checkpoints (feature-gated) |
| Memory report | ✅ | Per-bucket HBM/SRAM accounting |
| Footprint CSV export | ✅ | Operator tooling output |
| Multi-GPU parallel option | ✅ | Tensor(n) via Soc construction |
| Net compilation | ✅ | Network-attached model compilation |

### `crates/plowrt/` — Runtime Host

| Component | Status | Notes |
|-----------|--------|-------|
| CLI (serve + simulate) | ✅ | Tokio async + clap |
| HTTP server (TCP + UDS) | ✅ | Hyper-based |
| CPU simulation mode | ✅ | Dry-run without GPU |
| Bundle loading | ✅ | .pkt + sidecars from disk |
| Trace output | ✅ | Chrome trace JSON |

### `runtime/` — Device Runtime (C/CUDA/HIP)

| Component | Status | Notes |
|-----------|--------|-------|
| Interpreter loop (`interp.c`) | ✅ | Counter-gated sequential walk |
| Dispatch table (`dispatch.c`) | ✅ | Family-indexed function pointers |
| Memory map (`memmap.c`) | ✅ | Arena base + offset resolution |
| Decode layer (`decode.c`) | ✅ | Packet parsing + validation |
| **NVIDIA kernels** | | |
| Generic GEMM/Flash/Row/DMA/Layout sources | 🔧 | Reference and fused families exist; performant registration is not uniform by architecture |
| SM120 specialized Gemma dense/MoE decode+prefill | ✅ | Strong GPU/model benchmark evidence for explicit campaign cells |
| SM120 MLA/DSA decode | 🔧 | Dispatch arms exist in `runtime/nvidia/interp_sm120.cu`: `FLASH_MLA_DECODE`, `MLA_MERGE_FOLD`, `INDEX_SCORE`, `INDEX_SELECT`, `FLASH_GATHER_DECODE`. Not GPU-qualified here |
| SM120 MLA/DSA prefill | 🔲 | `FLASH_MLA_PREFILL=51` and `FLASH_GATHER_PREFILL=55` are declared in both ABIs with **no dispatch arm on any backend** |
| SM120 Mamba | 🔧 | `MAMBA2_SCAN=90` is in both ABIs and dispatches in `runtime/nvidia/interp_sm120.cu` (case `PLOW_DOP_MAMBA2_SCAN`). No model-level validation |
| Hopper/Blackwell generic registrations | 🔧 | Registration sources explicitly leave some performant variants unregistered |
| **AMD kernels** | | |
| GEMM/Flash/Row + HSA/device interpreter | 🔧 | Core and specialized gfx paths exist; coverage is op/model/shape specific |
| gfx MLA/DSA | 🔧 | Device implementation/reference work exists; qualify per model and target |
| AMD Mamba | 🔲 | No current production path |
| **CPU kernels** | | |
| GEMM (reference) | ✅ | `cpu/gemm.c` |
| Flash (reference) | ✅ | `cpu/flash.c` |
| Row ops | ✅ | `cpu/row.c` |
| DMA (memcpy) | ✅ | `cpu/dma.c` |
| Control loop | ✅ | `cpu/control.c` |

### `lean-plow/` — Formal Verification

| Component | Status | Notes |
|-----------|--------|-------|
| CLI binary (plow_verify) | ✅ | JSON-IPC protocol |
| Checkpoint A (rewrite soundness) | ✅ | Rule catalog matching |
| Checkpoint B (tile partition) | ✅ | Coverage + disjointness |
| Checkpoint C (SRAM fit) | ✅ | Temporal page check |
| Checkpoint D (counter protocol) | ✅ | Protocol covers deps |
| Checkpoint E (wire format) | ✅ | Round-trip proof |
| Checkpoint F (address map) | ✅ | Non-overlap + bounds |
| Universal theorems | ✅ | `WellFormed.no_deadlock`, `resourceOrdered_sub_happensBefore` |
| Performance proofs | ✅ | `CostBounds`, `FusionSavings`, `CrossOpPerf` |
| KV pool proofs | ✅ | `KvPool`, `KvPerf` |

---

## Feature Status Matrix

### Compiler Features

| Feature | Status | Blocking? |
|---------|--------|-----------|
| Single-GPU compilation | ✅ | — |
| Multi-bucket (prefill + decode) | ✅ | — |
| Tensor parallelism (N-axis) | ✅ | — |
| Cross-block pipelining | ✅ | — |
| Counter elimination | ✅ | — |
| Scope narrowing | ✅ | — |
| Prefetch hoisting | ✅ | — |
| SRAM temporal fit | ✅ | — |
| Lean verification (opt-in) | ✅ | — |
| Pipeline parallelism | 🔲 | Not blocking single-node |
| Expert parallelism (MoE) | 🔲 | Not blocking dense models |
| Wavefront counter clustering | 🔲 | Not blocking ≤200 tiles/boundary |
| Dynamic shape recompilation | 🔲 | Handled by bucketing for now |
| Iterative feedback (relax→reschedule loop) | 🔲 | Single-pass relax suffices |

### Runtime Features

| Feature | Status | Blocking? |
|---------|--------|-----------|
| Persistent/segmented device interpreters | 🔧 | Implemented on production paths; profile/resource qualification is not unified |
| Counter-gated execution | ✅ | — |
| NVIDIA backend descriptors/core | 🔧 | Do not infer uniform operator or tuned-model coverage |
| AMD backend descriptors/core | 🔧 | Do not infer uniform operator or tuned-model coverage |
| CPU simulation backend | ✅ | Reference/dry-run scope |
| HTTP serving | ✅ | — |
| Continuous batching | 🔧 | Mux/batching paths exist; production scorecard remains workload-specific |
| Multi-tenant isolation | 🔲 | Needed for shared infrastructure |
| Watchdog timer | 🔲 | Defense-in-depth |
| DPU integration | 🔲 | Needed for cross-node RDMA |
| Dynamic indirection table | 🔲 | Needed for MoE routing |
| In-place packet rewriting | 🔲 | Fine-grained dynamism |

---

## Test Coverage

| Test Category | Count | Location |
|---------------|-------|----------|
| Rewrite unit tests | ~15 | `crates/rewrite/tests/` |
| Schedule unit tests | ~10 | `crates/schedule/tests/` |
| Packet round-trip tests | ~12 | `crates/packet/src/lib.rs` (mod tests) |
| Compiler integration tests | ~15 | `crates/plowc/tests/` |
| Runtime integration tests | ~5 | `crates/plowrt/tests/` |
| Lean verification tests | ~8 | `crates/plowc/tests/lean_verify*.rs` |
| Runtime C tests | ~10 | `runtime/tests/` |
| Costmodel property tests | ~12 | `crates/costmodel/` (mod tests) |
| AMD-specific tests | ~4 | `runtime/tests/*gfx950*` |

---

## Device ISA: declared vs dispatchable

Reconciled mechanically from `crates/packet/src/dev.rs`,
`runtime/common/dev_isa.h`, and the dispatch switches, 2026-07-23.

Rust and C declare **84 opcodes each**, with no value collisions and no drift.
That agreement is now enforced by `crates/packet/tests/dev_opcodes.rs` rather
than by discipline — previously no test compared a single opcode value.

| interpreter | dispatch switch | opcodes with an arm |
|---|---|---|
| sm120 | dispatch `switch (in->op)` in `runtime/nvidia/interp_sm120.cu` | 63 / 84 |
| sm90a | shares the sm120 source | inherits, gated by `PLOW_NV_HOPPER` |
| gfx950 | dispatch `switch (in->op)` in `runtime/amd/interp.hip` | 51 / 84 |

Declared with **no dispatch arm on any backend**: `GEMM_NORM=9`,
`XREDUCESCATTER=25`, `XALLGATHER=26`, `FLASH_MLA_PREFILL=51`,
`FLASH_GATHER_PREFILL=55`.

Two caveats that make source-level counting insufficient, and which are why the
capability registry probes built artifacts instead:

- `interp_sm90a.cu` is a 42-line wrapper that `#include`s `interp_sm120.cu`
  with `PLOW_NV_HOPPER=1`. Hopper shares the dispatch table and gates
  internally, so a grep of that file finds zero opcodes.
- CMake builds **multiple distinct interpreter objects from that one translation
  unit** (`runtime/CMakeLists.txt`: `plow_interp_sm120`, `plow_interp_sm90a`,
  `plow_interp_sm120_gemma`, and variants), differing only in `-D` flags. Real
  capability is a property of an object, not of the source.

### Aliased opcodes

On NVIDIA, `GEMM`, `GEMM_MED`, and `GEMM_SMALL` all fall through to a single
`d_gemm` body in `runtime/nvidia/interp_sm120.cu`, comment: *"one body, three
tile opcodes"*. The tile is a compile-time macro per object, so the tuning axis
on NVIDIA is which object is built, not which opcode is emitted. AMD dispatches
the same three opcodes to three genuinely distinct instantiations
(`runtime/amd/interp.hip`), because a runtime tile switch would pull all three into the
interpreter's worst-case register allocation and fail the dispatch outright.

`kernelcaps::Inventory::alias_groups` reports this, so a campaign cannot rank
three names for one kernel and record the winner.

### Interpreter resource envelopes

Measured on this checkout with CUDA 13.0, against the figures in the build
scripts:

| object | registers | spill | documented |
|---|---|---|---|
| `interp_sm90a` (decode) | 208 | 0 | "150 regs" |
| `interp_sm90a_pf` (prefill) | 255 | 180 B store / 644 B load | "236 regs", zero spill required |

The `runtime/nvidia/interp_sm120.cu` header states the gate as *"must show 0
bytes spill and >= 1 block/SM"*, but no script or CMake target runs it — unlike AMD, which parses
`-Rpass-analysis=kernel-resource-usage` and **fails the build** past the
register cliff (`scripts/build_gfx950.sh`). The prefill object currently
violates its own documented gate. `kernelcaps::resource` parses both vendors'
reports and applies the gate; wiring it into the NVIDIA build is open work.

## Architecture Gaps (Design → Implementation)

These items are described in the design documents but have no corresponding implementation:

| Design Section | Gap | Priority |
|----------------|-----|----------|
| Kernel tuning architecture | `crates/kernelcaps` lands `KernelSpec`, capability filtering, alias detection, resource gates, and a probe that derives the inventory from a built object. gfx950 awaits a probe run on ROCm | 🔧 Highest |
| Block-driven tuning | `crates/tunedb` lands the normalized store, states, and atomic publication. `NetworkBlockDefinition` → manifest is open | 🔧 Highest |
| Compiler selection | `gemma4::pick_tile` (in `devgen`) now goes through `kernelcaps::select_kernel` (differential-verified over 5760 shapes). The `llama3` binary (`crates/plowc/src/bin/llama3.rs`) still has its own `pick_tile` | 🔧 Highest |
| Interpreter readiness | Resource-compatible dense/MoE/latent/Mamba profiles and complete-object gates | High |
| Extended ops | SM120 MLA/DSA completion, MLA prefill, then Mamba state/scan support | High |
| §7.4 Wavefront clustering | Counter tree for >200 tiles/boundary | Low (not needed for current models) |
| §9 Multi-GPU PP | Pipeline-parallel block partitioning | Medium (needed for >70B models) |
| §9 Multi-GPU EP | Expert-parallel MoE compilation | Medium (needed for MoE models) |
| §11 DPU integration | Cross-node RDMA via DPU executors | Low (single-node first) |
| §7.2 Indirection table | Runtime routing for MoE | Medium (tied to EP) |
| §7.3 In-place rewrite | Fine-grained dynamic shape handling | Low (bucketing suffices) |
| §12 Multi-tenancy | Tenant isolation + switching | Medium (needed for serving infra) |
| §13 Watchdog | Counter-progress monitoring | Low (Lean proves no deadlock) |
| Phase 4 Iterative feedback | Relax → reschedule loop | Low (single-pass works) |
| §8.5 Polonius-style Datalog | Egglog-native liveness analysis | Research (future optimization) |

---

## Build Configuration

### Cargo.toml Workspace

```toml
[workspace]
members = [
    "crates/nn-graph",
    "crates/hwspec",
    "crates/kernelcaps",
    "crates/tunedb",
    "crates/costmodel",
    "crates/devgen",
    "crates/rewrite",
    "crates/schedule",
    "crates/packet",
    "crates/plowc",
    "crates/lean_verify",
    "crates/plow-asset",
    "crates/plowrt",
    "tools/bench",
]

# Tuned for plowrt, the latency-critical artifact: fat LTO across a single
# codegen unit inlines the packet-decode → dispatch → counter path, and
# `panic = "abort"` drops unwind tables from it.
[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"
strip = "symbols"
```

### Feature Gates

| Feature | Crate | Effect |
|---------|-------|--------|
| `lean-verify` | `schedule`, `plowc` | Enables Lean IPC verification passes |
| `cuda` | `plowrt` | Links NVIDIA runtime |
| `hip` | `plowrt` | Links AMD ROCm runtime |

### Nix Flake

The project uses Nix for reproducible builds:
- `flake.nix` — defines devshell with Rust toolchain, CUDA toolkit, ROCm, Lean 4
- `flake.lock` — pins all inputs including `nn-graph` vendor dep
- Build: `nix develop` then `cargo build --release`
