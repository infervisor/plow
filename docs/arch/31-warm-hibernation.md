# 31 — Warm Hibernation & Dormant Model State

**Status: design proposal (checked against `unified-memory-budget`).**

## 1. Principle

Multi-model co-tenancy on GPU memory requires rapid context switching between resident models without paying the tens-of-seconds penalty of cold loading or pinning tens of gigabytes of idle VRAM.

Warm hibernation achieves 0 MiB idle physical memory retention with low-second reactivation (~3s) by decoupling virtual address space reservation from physical page backing:
- Driver Virtual Memory Management (`cuMemAddressReserve` / `hipMemAddressReserve`) allocates 64-bit address space in the process address map at microsecond cost without allocating physical DRAM.
- Physical memory blocks (`cuMemCreate` / `hipMemCreate`) are allocated and mapped into the reserved virtual addresses (`cuMemMap` + `cuMemSetAccess`).
- During hibernation, physical memory is unmapped and released (`cuMemUnmap` + `cuMemRelease`), returning 100% of physical DRAM to the driver.
- The virtual address reservations ($VA$) are preserved.

## 2. Invariants

### A. CUDA Graph Invariance
Captured and instantiated CUDA stream graphs (`CUgraphExec`) bind explicit 64-bit device virtual addresses in their kernel parameter tables:
- When physical memory is unmapped, launching the graph causes an MMU page fault.
- When physical memory is re-allocated and mapped back to the **exact same virtual address range**, the device MMU translates identical virtual addresses to the new physical pages.
- Instantiated graphs in `seg_graphs` and `cublaslt_decode_graph` remain valid. No graph re-capture (`cuStreamBeginCapture`), graph update (`cuGraphExecUpdate`), or driver instantiation (`cuGraphInstantiate`) is required.

### B. cuBLASLt Route Stability
cuBLASLt algorithm profiles and heuristic selections are functions of matrix shapes, datatypes, strides, and hardware architecture. They do not depend on physical page identities. Cached algorithm IDs remain valid across hibernation cycles.

### C. Cubin Module Permanence
Loaded kernel ELF binaries (`CUmodule` for interpreter, prefill segments, samplers, and multistep) reside in the driver context's code segment (< 1 MiB footprint). They remain loaded during dormancy.

## 3. Residency Transitions

Residency transitions extend the placement contract (`docs/arch/residency-contract.md`):

```
       [Unloaded] ──(load)──> [Resident] ──(serve)
                                   │
                             (hibernate)
                                   ▼
                               [Dormant]
                                   │
                                (awaken)
                                   ▼
                               [Resident]
```

1. **`hibernate`**:
   - Quiesces device execution (`be.synchronize()`).
   - Releases physical weight memory in `WeightSlab::Vmm` via `cuMemUnmap` + `cuMemRelease`.
   - Evicts dynamic KV cache pages (`VmmServe`).
   - Releases scratch activations.
   - Retains virtual address reservation, instantiated CUDA graphs, cubin modules, and metadata.
   - Registers model as `Residency::Dormant` in `ServeManager`.
   - Credits freed memory to `GpuMemoryBudget`.

2. **`awaken`**:
   - Ensures memory availability via `GpuMemoryBudget`.
   - Re-allocates physical memory chunks via `cuMemCreate`.
   - Maps chunks to the preserved virtual addresses via `cuMemMap` + `cuMemSetAccess`.
   - Streams weights from host storage via pinned DMA.
   - Re-arms scratch memory and counter cursors.
   - Transitions model back to `Residency::Resident`.
