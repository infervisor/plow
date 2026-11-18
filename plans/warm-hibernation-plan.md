# Warm Hibernation: Fast Cold-Start via Virtual Address Space & CUDA Graph Preservation

## 1. Goal & Context

Serve multi-model workloads (LLM, ASR, TTS) on constrained GPU VRAM by providing sub-second to low-second model reactivation (~3s vs. ~30–45s cold start) while surrendering 100% of physical model weight and KV cache memory back to the driver when dormant.

- **Target Architecture**: NVIDIA SM90a (H100 PCIe/SXM5) & AMD ROCm (MI300X gfx942).
- **Base Branch**: `unified-memory-budget` (`origin/unified-memory-budget` @ `42f1c6ff2`).
- **Target Worktree**: `/opt/dlami/nvme/plow/.claude/worktrees/sconc`.
- **Target Models**: Gemma 26B FP8, Qwen3-ASR 1.7B, Chatterbox T3, Veena 3B.

---

## 2. Verified Baseline Facts (`unified-memory-budget` @ `42f1c6ff2`)

### A. Current Load Latency Breakdown (Gemma 26B FP8, H100)
A full cold load via `ServeManager::load` / `GpuEngine::load_rank_inner` takes **~30–45 seconds** (or **~12–15s** if weights reside in host pagecache):
1. **Blob & Manifest Parse (~50–100 ms)**: `crates/plowrt/src/exec/gpu.rs:4198-4275`.
2. **Weight Allocation & DMA Streaming (~20–30s cold, ~3–5s cached)**:
   - `WeightSlab::Vmm` reserves VA and commits chunks via `wslab-map` thread (`crates/plowrt/src/memory/vmm.rs:6844-6923`).
   - Streams 26 GiB safetensors weights via pinned host buffers.
   - Synthesizes TMA descriptors (`GEN_TMAP_BF16`/`GEN_TMAP_E4M3`) and MoE scaling tables.
3. **Module Loading & ELF Parsing (~200–500 ms)**:
   - `cuModuleLoadData` for `interp_sm90a_fp8kv.cubin`, `interp_sm90a_pf.cubin`, sampler, multistep, and segmented GEMM (`gpu.rs:5875-5910`).
4. **cuBLASLt Algorithm Profiling (~4–8 seconds)**:
   - Evaluates heuristics and queries cuBLASLt search across prefill buckets ($M \in \{128, 256, 512, 1024, \dots\}$) and MoE grouped routes (`gpu.rs:4609-4730`).
5. **CUDA Graph Capture & Instantiation (~1.5–3 seconds)**:
   - Captures prefill chain sequences and decode rungs into `cuGraph` and instantiates executable graphs (`gpu.rs:13619, 13870`).
   - Populates `seg_graphs` (120–170 graph instances) and `cublaslt_decode_graph`.
6. **Mux Dispatcher Wiring (~50 ms)**:
   - Allocates initial KV cache, arms `GpuMemoryBudget`, and starts dispatcher loop (`serve/mux.rs`).

### B. Current Unload Teardown (`ServeManager::unload` / `GpuEngine::drop`)
Model unload is currently completely destructive (`crates/plowrt/src/exec/gpu.rs:16156-16247`):
- `be.synchronize()`.
- Destroys all CUDA graphs: `be.graph_destroy(g)` for `cublaslt_decode_graph` and all `seg_graphs`.
- Unloads all modules: `be.module_unload(&m)` for prefill, sampler, multistep, and segmented kernels.
- Drops `_weight_slab`: unmaps all chunks (`cuMemUnmap`), releases physical handles (`cuMemRelease`), and **frees the virtual address reservation** (`cuMemAddressFree`, `vmm.rs:6999`).
- Drops KV pages, TMA tables, scratch memory, and counter blocks.
- Calls `device.pool_trim(0)`.

*Consequence*: Subsequent requests pay the entire 30–45s cold start penalty.

---

## 3. Core Technical Insight: Virtual Address Preservation via Driver VMM

CUDA's Low-Level Virtual Memory Management (VMM) API decouples address space reservation from physical DRAM allocation:
1. `cuMemAddressReserve(&va, bytes, ...)`: Reserves contiguous 64-bit virtual address space in the user process (microseconds, 0 VRAM consumed).
2. `cuMemCreate(&handle, bytes, &prop, ...)`: Allocates physical DRAM pages on the GPU device.
3. `cuMemMap(va, bytes, offset, handle, ...)` + `cuMemSetAccess(va, bytes, ...)`: Binds physical memory to the reserved VA.
4. `cuMemUnmap(va, bytes)` + `cuMemRelease(handle)`: Unmaps and frees physical DRAM back to the driver.

### The CUDA Graph Invariant
- A captured and instantiated CUDA graph (`CUgraphExec`) encodes explicit 64-bit device virtual addresses ($VA$) in its launch parameter blocks.
- If physical memory is freed via `cuMemUnmap` + `cuMemRelease` while **keeping the virtual address reservation (`cuMemAddressReserve`) active**:
  - The virtual address pointers ($VA_{weights}$) remain permanently valid within the process address map.
  - While unmapped, launching the graph would page fault.
  - Upon remap (`cuMemCreate` + `cuMemMap` + `cuMemSetAccess`) back into the **identical reserved VA**, the exact same physical DRAM pages are accessible at the exact same virtual addresses.
  - **The CUDA graphs require ZERO re-capture, ZERO re-instantiation, and ZERO parameter patching.**
  - **cuBLASLt algorithm profiles and workspace selections remain 100% valid.**
  - **Loaded cubin modules remain resident in the driver context.**

---

## 4. State Breakdown: Preserved vs. Released

| Component | State in Dormant Mode | VRAM Footprint | Recovery on Wakeup |
| :--- | :--- | :--- | :--- |
| **Model Weights** | Unmapped & physical handles released (`cuMemRelease`). Virtual address range kept reserved (`cuMemAddressReserve`). | **0 MiB** | Re-create physical chunks + re-map to reserved $VA$ + stream DMA from host pagecache (~3s). |
| **KV Cache & Prefix** | Pages unmapped & released | **0 MiB** | Re-allocated on demand upon new request admission. |
| **Scratch & Activations** | Released | **0 MiB** | Re-allocated or re-mapped during wakeup. |
| **CUDA Graphs (`seg_graphs`)** | **PRESERVED** (`GraphExec`) | **0 MiB** (host driver struct) | **Zero re-capture, zero re-instantiation**. Runs immediately upon weight remap. |
| **cuBLASLt Route Cache** | **PRESERVED** (`cublaslt_decode`) | **0 MiB** | **Zero profiling overhead**. Retains optimal algorithm IDs. |
| **Cubin Modules** | **PRESERVED** (`module_pf`, etc.) | **< 1 MiB** (driver code text) | Ready for immediate kernel launches. |
| **Metadata & Descriptors** | **PRESERVED** (`blob`, `kernarg`) | **0 MiB** (host RAM) | Blob, tensors, TMA shapes already resolved. |

**Net VRAM consumption while Dormant: 0 MiB**.

---

## 5. Architectural Design & API Changes

### A. Residency State Machine (`crates/plowrt/src/serve/mod.rs`)
```
             ┌─────────────── load (cold, 35s) ──────────────┐
             │                                               ▼
      [Unloaded]                                         [Resident]
             ▲                                            │     ▲
             │                                   hibernate│     │awaken
             │                                     (~20ms)│     │(~3s)
             │                                            ▼     │
             └──────── unload ─────────────── [Dormant / Hibernated]
```
Add `Residency::Dormant` to `enum Residency` alongside `Resident`, `Auto`, `Unloaded`, `Unloading`.

### B. `VmmSlab` Hibernation Primitives (`crates/plowrt/src/memory/vmm.rs`)
Extend `VmmSlab` to support unmapping physical memory without freeing the virtual address reservation:
- `pub fn hibernate(&self) -> Result<u64>`:
  - Halts the background mapper thread if running.
  - Iterates over `handles`, invoking `ops.unmap(va + off, n)` and `ops.release(h)`.
  - Clears `handles` and resets `mapped` watermark to 0.
  - **Does NOT call `ops.address_free(self.va, self.reserved)`.**
  - Returns the number of physical bytes freed to the system.
- `pub fn awaken(&mut self, chunk_hint: u64) -> Result<()>`:
  - Spawns background mapper thread `wslab-map` to re-allocate physical chunks via `ops.create(n)` and re-map into `[self.va, self.va + self.reserved)` via `ops.map` + `ops.set_access`.
  - Blocks or overlaps as safetensors weights stream in.

### C. `GpuEngine` Hibernation Primitives (`crates/plowrt/src/exec/gpu.rs`)
- `pub fn hibernate(&mut self) -> Result<u64>`:
  - Synchronizes execution stream: `self.be.synchronize()?`.
  - Calls `_weight_slab.hibernate()?`.
  - Evicts all active KV pages: `if let Some(vmm) = &mut self.vmm { vmm.evict_all()?; }`.
  - Drops temporary scratch activations and counter views (`_ctr_block`, `d_ctr`).
  - Leaves `seg_graphs`, `cublaslt_decode_graph`, `module_pf`, `sampler`, `multistep`, `cublaslt_decode`, and `kernarg` completely untouched.
- `pub fn awaken(&mut self, checkpoint_dir: &Path) -> Result<()>`:
  - Re-maps physical memory into `_weight_slab` at the original base address.
  - Streams weights from `checkpoint_dir` into `_weight_slab` views using pinned host DMA buffers.
  - Re-allocates counter blocks and re-arms cursor pointers.
  - **Bypasses all module loading, cuBLASLt profiling, and CUDA graph instantiation.**

### D. `ServeManager` Integration (`crates/plowrt/src/serve/manager.rs`)
- Add endpoints:
  - `POST /v1/models/{slug}/hibernate` -> `ServeManager::hibernate(slug)`
  - `POST /v1/models/{slug}/awaken` -> `ServeManager::awaken(slug)`
- **`GpuMemoryBudget` Synchronization**:
  - On `hibernate`: credits `weights_bytes` + `kv_cache_cap` back to the budget ledger.
  - On `awaken`: ensures `GpuMemoryBudget::fits_resident` before initiating weight streaming.
- **LRU Dormancy Policy**:
  - When a new model requests admission and free memory is insufficient, `ServeManager` checks idle models in `Residency::Auto`.
  - Instead of performing destructive eviction (`unload`), it transitions the least-recently-used idle model to `Residency::Dormant`, freeing its physical VRAM while retaining its graph execution state.

---

## 6. Performance Target & Verification Criteria

| Scenario | Cold Start Today | Dormant Wakeup Target |
| :--- | :--- | :--- |
| **Gemma 26B FP8 (H100)** | ~35 seconds | **≤ 3.5 seconds (10x faster)** |
| **Physical VRAM while Idle** | 26 GiB | **0 MiB (100% freed)** |
| **CUDA Graph Re-capture** | 168 graphs re-compiled (~2.5s) | **0 graphs re-compiled (0s)** |
| **cuBLASLt Profiling** | 4–6s heuristic benchmark | **0s (retained)** |
| **Inference Accuracy** | Baseline | **Bit-exact identical token output** |

---

## 7. Implementation Roadmap & Milestones

1. **Phase 1: Memory Layer (`crates/plowrt/src/memory/vmm.rs`)**:
   - Implement `hibernate()` and `awaken()` on `VmmSlab`.
   - Unit tests: verify VA reservation persists across physical unmap/remap cycles and memory read/writes succeed post-wake.
2. **Phase 2: Engine Layer (`crates/plowrt/src/exec/gpu.rs`)**:
   - Implement `GpuEngine::hibernate()` and `GpuEngine::awaken()`.
   - Ensure graphs in `seg_graphs` launch successfully on remapped physical memory without graph rebuilding.
3. **Phase 3: Control Plane (`crates/plowrt/src/serve/`)**:
   - Add `Residency::Dormant` to residency contract and state machine.
   - Wire `ServeManager::hibernate` and `ServeManager::awaken`.
   - Update `GpuMemoryBudget` accounting on hibernation/awaken transitions.
4. **Phase 4: Hardware Validation on `gpu03`**:
   - Deploy Gemma 26B FP8 on H100.
   - Run inference turn -> trigger hibernation -> verify `nvidia-smi` drops by 26 GiB -> trigger awaken -> measure wakeup latency (< 3.5s) -> verify completion tokens match baseline bit-for-bit.
5. **Phase 5: Multi-Model Co-Tenancy & AMD Adaptation**:
   - Test interleaved hibernation/wakeup between Gemma 26B FP8 and Qwen3-ASR / TTS on a single GPU.
   - Port `VmmSlab` hibernation hooks to `exec/amd.rs` using `hipMemAddressReserve` / `hipMemUnmap`.

---

## 8. Log

- **2026-11-18**: Plan created and verified against `unified-memory-budget` codebase. Baseline facts, VMM VA mechanics, and CUDA graph preservation invariants documented.
