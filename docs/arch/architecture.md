# Plow Architecture — Index

> **Scope:** This page is the landing point for the `docs/arch/` set. Plow (née
> Infervisor) is an ahead-of-time compiler and persistent-kernel runtime for
> transformer inference across heterogeneous accelerators (NVIDIA, AMD, CPU,
> DPU). The compiler (`plowc`) lowers a model and a hardware topology into
> per-executor `.pkt` streams plus JSON sidecars; the runtime (`plowrt`) loads
> those artifacts and executes them on persistent-kernel interpreters that
> coordinate through a shared atomic-counter pool, keeping the CPU off the
> critical path.

The numbered chapters below are the single source of truth. This file used to
duplicate them and is now an index only.

---

## Chapters

| # | Document | What it covers |
|---|----------|----------------|
| [00](00-overview.md) | **Overview & Index** | System identity, design philosophy, boundaries, data flow, key decisions |
| [01](01-compiler-pipeline.md) | **Compiler Pipeline** | Egglog rewriting half + list-scheduling half and the interface between them |
| [02](02-tile-graph.md) | **Tile Dependency Graph** | The central IR (DmaIn/Compute/DmaOut DAG) that connects rewriting to scheduling |
| [03](03-scheduler.md) | **Scheduler** | Machine model, interval-conflict list scheduling, and post-schedule passes |
| [04](04-packet-abi.md) | **Packet ABI** | Wire format, opcode encoding, record/body types, resource kinds |
| [05](05-counter-system.md) | **Counter Coordination** | Scope tiers, counter determination, wavefront clustering |
| [06](06-runtime.md) | **Runtime** | Execution model, interpreter loop, memory regions, KV paging, backends |
| [07](07-cost-model.md) | **Cost Model** | Hardware abstraction, tile candidate enumeration and selection |
| [08](08-formal-verification.md) | **Formal Verification** | Lean 4 checkpoint architecture (A–F) and universal theorems |
| [09](09-multi-gpu.md) | **Multi-GPU & Parallelism** | TP/PP/EP/DP strategies and collective lowering |
| [10](10-implementation-status.md) | **Implementation Status** | What exists in the checkout vs. target design, crate by crate |
| [11](11-tuning-coverage.md) | **Tuning Coverage** | Per operator family: distinct kernels, knobs, oracles, blockers |
| [12](12-using-the-tuner.md) | **Using the Tuner** | `plowc tune`, reading the output, taking a measurement |
| [13](13-prefill-chunking.md) | **Prefill Chunking** | Bucket ladder, the ragged tail, ragged-M |
| [14](14-amd-arch-divergence.md) | **AMD Arch Divergence** | gfx942 vs gfx950: one source tree, what forks, the tripwire |

---

## Workspace at a glance

Cargo workspace, 13 member crates (Rust) plus a C/CUDA/HIP device runtime under
`runtime/` and Lean 4 proofs under `lean-plow/`:

- **Frontend / IR:** `nn-graph` (symbolic operator graph IR, model hub; folds the
  former `frontend`)
- **Compiler core:** `rewrite`, `costmodel`, `schedule`, `packet`, `plowc`
- **Hardware / kernel / tuning registries:** `hwspec`, `kernelcaps`, `tunedb`
- **Verification:** `lean_verify`
- **Shared schema:** `plow-asset` (compiler↔runtime boundary types)
- **Runtime host:** `plowrt` (serve, simulate, mux, executor pool)
- **Legacy:** `devgen` (deprecated device-blob emitter, feature-gated)

See [10 — Implementation Status](10-implementation-status.md) for the full
crate-by-crate breakdown, the build profile, and the device-ISA reconciliation.
