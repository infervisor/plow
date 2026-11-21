# Architectural Blueprint: Accelerated Model Bringup, Autonomous Kernel Tuning, and Runtime Stability Governance

## Executive Summary

This architecture specification defines the strategic redesign of `plow`'s model bringup pipeline, kernel tuning campaigns, and runtime execution engine (`plowrt`). The objectives are:
1. **Accelerate Architecture Bringup**: Reduce time-to-first-token for newly released LLMs and multi-modal architectures (e.g., hybrid linear-attention GDN/KDA, sparse MoE, multi-head latent attention MLA, audio/diffusion transformers) from multiple developer-weeks to under 48 hours.
2. **Autonomous Kernel Campaigns**: Eliminate ad-hoc kernel tuning by establishing a formal `BlockContract` and an automated search-to-certification pipeline that guarantees every production kernel operates at $\ge 80\%$ (decode GEMV/GEMM) or $\ge 75\%$ (prefill GEMM) of calibrated empirical roofline before inclusion in a recipe.
3. **Runtime Stability and Invariant Governance**: Eliminate admission flapping, memory budget traps, and scheduling regressions in `plowrt` by replacing heuristic controller logic with formally verified state invariants, deterministic discrete-event simulation (DES), and property-based fuzzing.

---

## 1. End-to-End Lifecycle Bottleneck Analysis

The current lifecycle from an upstream Hugging Face checkpoint to high-performance production serving consists of four distinct phases:
$$\text{HF Checkpoint} \xrightarrow{\text{1. Devgen Emit}} \text{Packet Graph} \xrightarrow{\text{2. Campaign Tuning}} \text{Tuned DB} \xrightarrow{\text{3. Recipe Pack}} \text{Manifest} \xrightarrow{\text{4. Production Serving}} \text{plowrt}$$

```
+----------------------------------------------------------------------------------------------------+
|                                CURRENT BRINGUP LIFECYCLE BOTTLENECKS                               |
+------------------------------------+---------------------------------------------------------------+
| Phase                              | Root Cause Bottlenecks                                        |
+------------------------------------+---------------------------------------------------------------+
| 1. HF Checkpoint -> Devgen Emit    | - Monolithic emitters (3,000 to 16,000 LOC per architecture).  |
|                                    | - Silent numerical correctness traps (RoPE interleaving,     |
|                                    |   RMSNorm epsilon placement, attention scale absorption).      |
|                                    | - Untyped instruction immediates (i[0..7] ABI packing).       |
|                                    | - Target-divergent flag semantics (PLOW_UNISEG, PLOW_FP8).    |
+------------------------------------+---------------------------------------------------------------+
| 2. Kernel Campaign Tuning          | - Fragmented toolchain (Python campaign.py vs Rust tunedb).   |
|                                    | - Heavy monolithic interpreter compilation (CUBIN/HSACO).     |
|                                    | - Violation of single-block roofline rule; manual tuning.     |
+------------------------------------+---------------------------------------------------------------+
| 3. Recipe Verification & Packaging | - Incomplete schema validation without invariant verification |
|                                    | - Build-host path leakage into release artifacts.             |
+------------------------------------+---------------------------------------------------------------+
| 4. Production Serving (plowrt)     | - Admission controller flapping on elastic burst exit.        |
|                                    | - Throughput downshift starvation trap from EWMA noise.       |
|                                    | - Hybrid sliding-ring memory budget sizing mismatch.          |
+------------------------------------+---------------------------------------------------------------+
```

### 1.1 Phase 1: Hugging Face Checkpoint $\to$ Devgen Emit

#### 1.1.1 Monolithic Architecture Proliferation
In `crates/devgen/src/`, each new model architecture is currently implemented as an isolated, monolithic emitter:
- `mla.rs`: ~16,200 lines for DeepSeek-V3/R1 multi-head latent attention and sparse MoE.
- `lib.rs`: ~12,400 lines containing base transformer logic, weight remapping, and generic graph assembly.
- `k3.rs`: ~4,500 lines for linear RNN / gated delta net (GDN).
- `qwen35.rs`: ~4,300 lines for hybrid linear/GQA architectures.

Each emitter re-implements tensor weight name resolution (`checkpoint.rs`), linear-to-packed GEMM layout transformations, KV cache buffer offset allocation, RoPE frequency table precomputation, and packet graph wiring. Because these implementations are monolithic, bug fixes in one architecture (e.g., proper alignment of FP8 quantized scale factors or bias addition) are rarely propagated to others.

#### 1.1.2 Silent Numerical Correctness Traps
New model architectures introduce subtle mathematical variants that compile cleanly into packet graphs but generate fluent-but-incoherent tokens:
- **RMSNorm Epsilon Placement**: Certain models compute $\frac{x}{\sqrt{\frac{1}{d}\sum x^2 + \epsilon}}$ while others compute $\frac{x}{\sqrt{\frac{1}{d}\sum x^2} + \epsilon}$, or downcast $\epsilon$ to FP16/BF16 before the addition.
- **RoPE Rotary Masking & Interleaving**: Divergence between Neox style (half-split: $[-x_{d/2..}, x_{..d/2}]$) and standard GPT-J style (interleaved adjacent pairs: $[-x_1, x_0, -x_3, x_2]$), as well as partial rotary application (e.g., applying RoPE to only the first 64 of 128 head dimensions and passing the remaining 64 passthrough).
- **Attention Scale Factor Absorption**: Fusing $1/\sqrt{d_k}$ into the Query projection weights versus applying it inside the softmax kernel. When combined with FP8 scales or Query/Key normalization (QK-norm), unabsorbed scale factors cause silent underflow/overflow.
- **Tied Embedding Layouts**: Models reusing input embeddings for the LM head output with transposition, scaling ($d_{\text{model}}^{-0.5}$), or untied bias vectors.
- **Recurrent State Layouts**: For hybrid linear models (Kimi KDA, Qwen GDN), state tensors $[B, H, K, V]$ require specific contiguous strides for chunked parallel prefix scans. Misaligned strides produce valid matrix operations with garbled sequential memory access.

#### 1.1.3 Untyped Instruction ABI (`i[0..7]`)
The `Instruction` packet format in `include/packet.h` and `crates/packet/` stores immediate parameters in generic 32-bit integer arrays:
```c
struct Instruction {
    uint16_t op;
    uint16_t flags;
    uint32_t i[8];
    uint64_t p[4];
};
```
Different kernels interpret `i[0..7]` inconsistently:
- `FlashPrefill`: Head dimension $D$ is passed in `i[6]`, sequence length in `i[0]`.
- `FlashMerge`: Head dimension $D$ is passed in `i[3]`, sequence length in `i[1]`.
- `GemvDecode`: Tile size $M$ is in `i[2]`, batch size in `i[0]`.

This lack of strong typing in the intermediate representation (IR) causes emitter-kernel parameter mismatch bugs that can only be diagnosed at GPU execution time through memory faults or NaN outputs.

#### 1.1.4 Target Architecture Divergence Traps
Flags in `crates/devgen/src/emit_config.rs` have divergent semantics between hardware targets:
- `PLOW_UNISEG`: Groups execution segments into a unified dispatch stream. On NVIDIA Hopper/Ada, this improves command queue latency. On AMD CDNA3/RDNA3, it forces single-queue scheduling that disrupts async wave-level copy engines and drops compute unit occupancy.
- `PLOW_FP8`: Activates FP8 GEMM/GEMV execution. AMD requires `E4M3FNUZ` or `E5M2FNUZ` format conversions, whereas NVIDIA Hopper native tensor cores expect `E4M3FN` / `E5M2`. Emitting incorrect quantization encodings produces invalid numerical results.

---

### 1.2 Phase 2: Kernel Campaign Tuning

#### 1.2.1 Fragmented Toolchain
Kernel tuning is split across multiple disconnected systems:
- `scripts/campaign/campaign.py`: High-level Python driver for running rungs, sweeps, and benchmarks.
- `scripts/gen_kernels/gen_search.py`: Candidate generator using TileLang/Triton.
- `crates/tunedb/`: Rust database storing tuned kernel dispatch selections keyed by `HardwareFingerprint`.
- `runtime/nvidia/` and `runtime/amd/`: Hand-optimized C++/CUDA/HIP template implementations.

Winning configurations found by Python sweep scripts must be manually transcribed or piped through bespoke scripts into `crates/tunedb/records/`. If a model introduces a new GEMM shape or tile ratio (e.g., MLA latent projection $M=1, K=576, N=4096$), no automated mechanism ensures it is systematically swept across SM/CU block configurations before deployment.

#### 1.2.2 Monolithic Interpreter Compilation Latency
`runtime/` builds monolithic `.cubin` (NVIDIA) and `.hsaco` (AMD) objects containing compiled dispatch templates for hundreds of block variations. Compiling these objects during candidate search creates multi-hour feedback loops, discouraging rapid exploration of architectural kernel variations.

#### 1.2.3 Violations of the Single-Block Roofline Hard Rule
Per `docs/bringup/agent-tools.md`, the hard engineering rule requires:
1. Calibrate hardware roofline (HBM bandwidth, L2 bandwidth, Tensor Core compute peak).
2. Measure single transformer block execution against theoretical roofline.
3. Optimize kernel memory traffic and arithmetic intensity until single-block roofline target is satisfied.
4. Only then proceed to full-model multi-rung serving sweeps.

In practice, engineers frequently jump directly to multi-rung serving benchmarks (`plowbench.sh`). Full-pipeline noise (queueing delays, host-device synchronization, memory bus contention) obscures kernel-level inefficiencies, resulting in suboptimal kernel selection.

---

### 1.3 Phase 3: Recipe Verification and Packaging

`scripts/recipe_common.py` and `scripts/check_recipe.py` validate recipe TOML files against a static schema. However, they lack semantic cross-validation:
- They do not verify whether `ring_slot_charge` under hybrid sliding/global attention aligns with the memory budget reserved in `crates/plowrt/src/memory/budget.rs`.
- They do not verify that the chosen decode rung configurations have certified kernel entries in `crates/tunedb/`.
- Packaging via `plowrt pack` has historically leaked absolute build-host toolchain paths into published manifests, violating reproducibility guarantees.

---

### 1.4 Phase 4: Production Serving in `plowrt`

#### 1.4.1 Admission Controller Flapping on Elastic Burst
In `crates/plowrt/src/sched/admission.rs`, the admission controller permits an elastic burst above the nominal concurrency ceiling ($C_{\text{nominal}}$) up to an elastic limit ($C_{\text{max}}$) when queue depth spikes. However, prior logic immediately collapsed the concurrency limit back to $C_{\text{nominal}}$ as soon as the queue emptied. If 40 requests were active ($C=40$) and the queue reached 0, the ceiling dropped to 32. Ingress admissions were completely blocked until active requests drained below 32, creating throughput stutter and latency spikes.

#### 1.4.2 Throughput Downshift Starvation Trap
In `crates/plowrt/src/sched/rungs.rs`, the rung scheduler dynamically adjusts active batch sizes across configured decode rungs (e.g., 1, 4, 8, 16, 32, 64) based on EWMA token throughput. Transient jitter on intermediate rungs (e.g., rung 16 momentarily measuring slightly higher tok/s than rung 32 due to short generation lengths) could cause the controller to downshift and throttle system capacity, even when dozens of requests remained queued.

#### 1.4.3 Hybrid Sliding-Ring Memory Budget Sizing
Models utilizing hybrid sliding-window and global attention (e.g., Gemma-2, Gemma-4) require dedicated circular buffer rings for sliding layers alongside standard paged KV blocks for full-attention layers. Sizing the static memory reservation for these sliding rings based on elastic burst capacity ($C_{\text{max}}$) steals memory from the general paged KV pool, increasing premature evictions or allocation failures during long-context generation.

---

## 2. Architecture for Generic, Automated Kernel Tuning & Roofline Certification

To eliminate manual tuning loops, `plow` requires a unified kernel contract, an empirical hardware calibration engine, an autonomous search coordinator, and a closed-loop certification gate.

```
+----------------------------------------------------------------------------------------------------+
|                               AUTONOMOUS TUNING & CERTIFICATION ENGINE                             |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   +-----------------------+      +-----------------------+      +------------------------------+   |
|   |  Model IR / Sub-DAG   | ---> |     BlockContract     | ---> |   calibrated_roofline.py     |   |
|   | (devgen Block Node)   |      | (FLOPs, Bytes, Bounds)|      |   (Empirical HBM/L2/Tensor)  |   |
|   +-----------------------+      +-----------------------+      +------------------------------+   |
|                                                                                 |                  |
|                                                                                 v                  |
|   +--------------------------------------------------------------------------------------------+   |
|   | Autonomous Search Pipeline (TileLang / Triton / C++ Templates)                             |   |
|   | - Tile Sizes: [M_tile, N_tile, K_tile]                                                      |   |
|   | - Warp Configurations & Pipeline Stages: [2..8 stages, 4..16 warps]                        |   |
|   | - TMA / Async Copy / Swizzle / Register Pressure Optimizations                             |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                  |                                                 |
|                                                  v                                                 |
|   +--------------------------------------------------------------------------------------------+   |
|   | Single-Block Micro-Harness (Isolate single transformer block sub-DAG on single GPU)        |   |
|   +--------------------------------------------------------------------------------------------+   |
|                                                  |                                                 |
|                                                  v                                                 |
|   +--------------------------------------------------------------------------------------------+   |
|   | Certification Gate (Hard Thresholds):                                                      |   |
|   | - Memory-Bound Op (Decode GEMV, Linear RNN Step): Efficiency >= 80% of Calibrated HBM BW   |   |
|   | - Compute-Bound Op (Prefill GEMM, MoE Dense):     Efficiency >= 75% of Calibrated TFLOPS   |   |
|   +--------------------------------------------------------------------------------------------+   |
|                   |                                                            |                   |
|                   | Pass                                                       | Fail              |
|                   v                                                            v                   |
|   +---------------------------------+                        +---------------------------------+   |
|   | crates/tunedb/records/          |                        | Genetic / Grid Refinement Loop  |   |
|   | (Auto-serialized JSON record)   |                        | (Prune invalid tile spaces)     |   |
|   +---------------------------------+                        +---------------------------------+   |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 2.1 Unified `BlockContract` and Op Descriptors

Every computational block in `devgen` must declare a strongly typed `BlockContract` implementing standard arithmetic and memory interfaces.

```rust
pub enum BoundCategory {
    MemoryBoundHbm,
    MemoryBoundL2,
    ComputeBoundTensorCore,
    ComputeBoundVectorAlu,
}

pub struct BlockContract {
    pub op_name: &'static str,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub dtype: DataType,
    pub bound_category: BoundCategory,
    /// Exact arithmetic operations required per execution invocation
    pub theoretical_flops: fn(batch: usize, seq_len: usize, d_model: usize) -> u64,
    /// Exact memory bytes transferred across the memory hierarchy
    pub minimum_bytes_transferred: fn(batch: usize, seq_len: usize, d_model: usize) -> u64,
}
```

#### Arithmetic Intensity & Theoretical Lower Bounds
For each operation, the arithmetic intensity $I$ (FLOPs/byte) determines its roofline ceiling:
$$I = \frac{\text{Theoretical FLOPs}}{\text{Minimum Memory Bytes Transferred}}$$

- **Decode Multi-Query / Multi-Head Attention ($M=1$)**:
  $$\text{FLOPs} = 2 \cdot B \cdot H_q \cdot S \cdot D, \quad \text{Bytes} = B \cdot H_{kv} \cdot S \cdot D \cdot \text{sizeof(dtype)}$$
  $$I_{\text{decode}} = \frac{2 \cdot H_q}{H_{kv} \cdot \text{sizeof(dtype)}} \quad \text{(Strictly Memory-Bound)}$$
- **Prefill Chunked GEMM ($M \ge 512$)**:
  $$\text{FLOPs} = 2 \cdot M \cdot N \cdot K, \quad \text{Bytes} = (M \cdot K + K \cdot N + M \cdot N) \cdot \text{sizeof(dtype)}$$
  $$I_{\text{prefill}} \approx \frac{2 \cdot M \cdot N}{M + N} \gg I_{\text{knee}} \quad \text{(Strictly Compute-Bound)}$$

### 2.2 Empirical Calibration Engine (`scripts/campaign/calibrated_roofline.py`)

Theoretical datasheet numbers (e.g., "3.35 TB/s HBM3" or "989 TFLOPS BF16") are never used as ground truth for optimization targets. Hardware throttling, clock variance, memory controller efficiency, and driver overhead degrade attainable peaks.

The calibration engine executes a standardized micro-benchmark suite prior to kernel sweeps:
1. **Empirical HBM Bandwidth ($BW_{\text{HBM}}$)**: Coalesced read/write streaming across 4 GB buffers using vector loads (`float4` / `int4`).
2. **Empirical L2 Cache Bandwidth ($BW_{\text{L2}}$)**: Repeated cache-line footprint streaming within the L2 boundary (e.g., 50 MB on SM90, 32 MB on CDNA3).
3. **Empirical Tensor Core Peak ($P_{\text{Tensor}}$)**: Square GEMM kernels ($N=K=4096, M=4096$) sweeping standard tile sizes to extract maximum sustained FP16, BF16, and FP8 compute rates.
4. **Empirical Host-Device Launch Latency ($t_{\text{launch}}$)**: Null packet dispatch measuring driver queue submission overhead.

Results are cached under `perf-data/probes/<hw_fingerprint>.json`:
```json
{
  "fingerprint": "sm90_h100_sxm5_80gb",
  "empirical_hbm_bw_gbps": 3052.4,
  "empirical_l2_bw_gbps": 11840.0,
  "empirical_bf16_tensor_tflops": 892.1,
  "empirical_fp8_tensor_tflops": 1740.5,
  "dispatch_latency_ns": 1820
}
```

### 2.3 Autonomous Candidate Search & Single-Block Harness

The search pipeline generates kernel variations across a multidimensional tuning space:
- **Tiling**: $M_{\text{tile}} \in [16, 32, 64, 128, 256]$, $N_{\text{tile}} \in [16, 32, 64, 128, 256]$, $K_{\text{tile}} \in [32, 64, 128]$.
- **Execution Pipeline**: Pipeline stages $\in [2, 3, 4, 5, 7]$, warp counts $\in [4, 8, 12, 16]$.
- **Architectural Primitives**: TMA (Tensor Memory Accelerator), asynchronous transaction barriers, register allocation ceilings (`maxrregcount`), swizzle patterns for bank conflict elimination.

#### Single-Block Verification Harness
Rather than evaluating full model graphs, the candidate search isolates a single transformer block's sub-DAG (e.g., 1 layer of MLA decode or 1 layer of KDA state scan):
1. Allocate inputs in VMM device memory.
2. Warm up kernel for 10 iterations to eliminate cold-cache anomalies.
3. Execute 100 timed iterations using hardware cycle counters (`cudaEventElapsedTime` / `hipEventElapsedTime`).
4. Calculate empirical efficiency:
   $$\eta_{\text{mem}} = \frac{\text{Minimum Bytes} / t_{\text{elapsed}}}{BW_{\text{HBM}}}, \quad \eta_{\text{compute}} = \frac{\text{Theoretical FLOPs} / t_{\text{elapsed}}}{P_{\text{Tensor}}}$$

### 2.4 Certification Gate & Direct `tunedb` Integration

A candidate kernel is automatically certified and admitted to production only when it satisfies the certification gate:
$$\text{Gate}(\text{Kernel}) = \begin{cases} 
\text{PASS} & \text{if } \text{category} = \text{MemoryBound} \land \eta_{\text{mem}} \ge 0.80 \\
\text{PASS} & \text{if } \text{category} = \text{ComputeBound} \land \eta_{\text{compute}} \ge 0.75 \\
\text{FAIL} & \text{otherwise}
\end{cases}$$

Upon passing:
1. The kernel configuration is automatically formatted and serialized into `crates/tunedb/records/<hw_fingerprint>.json`.
2. A cryptographic hash of the compiled kernel bytecode is registered in the record to ensure reproducibility.
3. The recipe generator (`scripts/render_recipe.py`) marks the op configuration as validated.

---

## 3. Runtime Stability Invariants, Memory Governance, and Verification Harness

Production instability in high-throughput inference engines stems primarily from heuristic-driven controllers thrashing under dynamic traffic. `plowrt` enforces mathematically defined invariants governing admission, memory allocations, and multi-rung dispatch.

```
+----------------------------------------------------------------------------------------------------+
|                                    PLOWRT STABILITY INVARIANTS                                     |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [Queue Ingress]                                                                                   |
|         |                                                                                          |
|         v                                                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | Admission Controller: Monotonic Burst Draining & Steady-State Floor                          |  |
|  | - C_max(t) = max(C_nominal, N_active(t))                                                     |  |
|  | - If Queued > 0 and Active >= C_nominal: No rung downshifting below C_nominal                 |  |
|  | - Minimum Rung Dwell Time T_dwell >= 500ms                                                   |  |
|  +----------------------------------------------------------------------------------------------+  |
|         |                                                                                          |
|         +-----------------------+                                                                  |
|         | Slot Admitted         | Slot Rejected (Backpressure)                                     |
|         v                       v                                                                  |
|  +----------------------------------------------------------------------------------------------+  |
|  | Tiered Memory Governance (Physical VMM Budget)                                               |  |
|  | - Tier 0: Static Weights + Workspace Buffers (Fixed)                                         |  |
|  | - Tier 1: Parked Base Sliding-Rings (C_nominal * W_sliding * L_sliding * D_head)                |  |
|  | - Tier 2: Dynamic Dynamic Paged KV Blocks (General Request Pool)                             |  |
|  | - Tier 3: Burst Headroom Buffer (Guaranteed 5% physical pool reserve)                       |  |
|  | * Rule: Burst slots allocate dynamic rings from Tier 3; Tier 2 never encroached.             |  |
|  +----------------------------------------------------------------------------------------------+  |
|         |                                                                                          |
|         v                                                                                          |
|  +----------------------------------------------------------------------------------------------+  |
|  | Multi-Rung Decode Scheduler                                                                  |  |
|  | - Rungs: [1, 4, 8, 16, 32, 64]                                                               |  |
|  | - Active slots packed contiguously; empty slots zero-cost bypass                             |  |
|  +----------------------------------------------------------------------------------------------+  |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 Admission Controller Invariants

#### Invariant 1: Monotonic Elastic Burst Draining
*Definition*: Let $C_{\text{nominal}}$ be the configured nominal concurrency limit, $C_{\text{burst}}$ be the maximum elastic burst ceiling, and $N_{\text{active}}(t)$ be the number of active requests executing in `plowrt` at time $t$.

When the system enters burst state ($N_{\text{active}}(t) > C_{\text{nominal}}$) and the ingress queue drains ($Q(t) = 0$), the instantaneous admission ceiling $C_{\text{admit}}(t)$ must satisfy:
$$C_{\text{admit}}(t) = \max(C_{\text{nominal}}, N_{\text{active}}(t))$$
$$\frac{d}{dt} C_{\text{admit}}(t) \le 0 \quad \text{for } N_{\text{active}}(t) > C_{\text{nominal}}$$

*Proof of Non-Blocking Ingress*: If an active slot completes at time $t_1$, $N_{\text{active}}$ drops to $N_{\text{active}}(t_1) - 1$. If a new request arrives when $N_{\text{active}} \ge C_{\text{nominal}}$, it is rejected or queued; however, existing active requests in slots $s \in [C_{\text{nominal}} + 1, C_{\text{burst}}]$ are guaranteed non-preempted completion without stalling adjacent compute streams.

#### Invariant 2: Nominal Steady-State Floor Under Saturated Demand
*Definition*: Let $R_k$ represent decode rung $k$ with batch capacity $B_k$. Let $\text{EWMA}_{\text{tok/s}}(R_k)$ be the exponentially weighted moving average of token generation throughput on rung $R_k$.

If queued demand $Q(t) > 0$ and $N_{\text{active}}(t) \ge C_{\text{nominal}}$, the scheduler is forbidden from downshifting to any rung $R_j$ where $B_j < C_{\text{nominal}}$:
$$\text{SelectedRung}(t) \ge R_{\text{nominal}} \quad \forall t \text{ where } Q(t) > 0 \land N_{\text{active}}(t) \ge C_{\text{nominal}}$$

This eliminates the failure mode where transient timing anomalies on smaller rungs trigger downshift avalanches under heavy client load.

#### Invariant 3: Minimum Rung Dwell Time and Switching Hysteresis
To prevent high-frequency oscillations between adjacent rungs (e.g., jumping between rung 16 and rung 32 every 2 steps):
$$t_{\text{switch}} - t_{\text{last\_switch}} \ge T_{\text{dwell}} \quad (T_{\text{dwell}} = 500\,\text{ms} \lor 50 \text{ decode iterations})$$
A transition to higher rung $R_{k+1}$ requires:
$$\text{EWMA}_{\text{tok/s}}(R_{k+1}) > 1.08 \cdot \text{EWMA}_{\text{tok/s}}(R_k)$$
A transition to lower rung $R_{k-1}$ requires:
$$\text{EWMA}_{\text{tok/s}}(R_{k-1}) > 1.12 \cdot \text{EWMA}_{\text{tok/s}}(R_k)$$

### 3.2 Tiered Memory Governance and Hybrid Attention Sizing

Physical GPU memory managed by `crates/plowrt/src/memory/budget.rs` and `vmm.rs` is partitioned into four strict tiers:

```
+--------------------------------------------------------------------+
| Tier 0: Static Weights, Constants, Runtime Buffers                 |
+--------------------------------------------------------------------+
| Tier 1: Parked Base Sliding-Rings (C_nominal * Size_ring)           |
+--------------------------------------------------------------------+
| Tier 2: Paged KV Cache Block Pool (General Request Allocations)    |
+--------------------------------------------------------------------+
| Tier 3: Elastic Burst Headroom Reserve (5% physical VMM headroom)  |
+--------------------------------------------------------------------+
```

#### Sizing Hybrid Sliding-Ring Allocations
In architectures combining sliding-window layers and full self-attention (e.g., Gemma-2 with sliding window $W=4096$ on alternate layers):
- The static parked allocation covers only nominal capacity:
  $$M_{\text{parked\_rings}} = C_{\text{nominal}} \times L_{\text{sliding}} \times 2 \times W_{\text{sliding}} \times D_{\text{head}} \times \text{sizeof(dtype)}$$
- When an elastic burst request $r_i$ is admitted ($C_{\text{nominal}} < i \le C_{\text{burst}}$), its sliding rings are allocated dynamically from the **Tier 3 Burst Reserve**.
- If Tier 3 reserve is exhausted, admission of request $r_i$ is rejected with backpressure before any kernel scheduling occurs. Tier 2 (paged KV cache) is never cannibalized for sliding rings.

### 3.3 Verification Harness: Discrete-Event Simulator & Controller Fuzzing

To ensure mathematical stability under all production conditions, `plowrt` incorporates a headless discrete-event simulation harness (`crates/plowrt/src/sched/sim.rs`).

```rust
pub struct RuntimeSimHarness {
    pub admission: AdmissionController,
    pub scheduler: DecodeRungScheduler,
    pub memory: MemoryBudgetTracker,
    pub clock_ns: u64,
}

impl RuntimeSimHarness {
    /// Executes randomized arrival and execution traces to fuzz runtime invariants
    pub fn fuzz_trace(&mut self, arrivals: Vec<SyntheticRequest>) {
        for req in arrivals {
            self.step_to(req.arrival_time);
            let admit_result = self.admission.try_admit(&req, &self.memory);
            // Invariant Check 1: No OOM
            assert!(self.memory.physical_allocated_bytes() <= self.memory.physical_capacity_bytes());
            // Invariant Check 2: Monotonic burst ceiling
            assert!(self.admission.verify_monotonic_drain_invariant());
            // Invariant Check 3: Concurrency bound
            assert!(self.scheduler.active_count() <= self.admission.current_ceiling());
        }
    }
}
```

#### Property-Based Fuzz Testing
Using `proptest`, the simulation harness executes 1,000,000 randomized state sequences across extreme traffic conditions:
- **Burst Traffic**: Poisson arrival spikes driving queue depth from 0 to 500 requests in 10 ms.
- **Pathological Contexts**: Prompt lengths varying randomly between 1 token and 128k tokens.
- **Kernel Jitter**: Decode step latencies sampled from multimodal Gaussian distributions with injected random tail stalls (e.g., 50 ms PCIe/bus stalls).
- **Early Client Aborts**: Up to 30% of requests randomly cancelling during generation.

Any invariant violation halts execution and produces a minimized deterministic reproduction trace.

---

## 4. Concrete Engineering Roadmap & Prioritized Workstreams

The modernization of model bringup, kernel tuning, and runtime stability is organized into four parallel workstreams executed across five concrete milestones (M0 through M4).

```
+----------------------------------------------------------------------------------------------------+
|                                    ENGINEERING ROADMAP TIMELINE                                    |
+----------------------------------------------------------------------------------------------------+
| Milestone | Timeline  | Core Focus and Deliverables                                                |
+----------------------------------------------------------------------------------------------------+
|    M0     | Weeks 1-2 | Runtime Stability Hotfix & Controller Invariant Hardening                  |
|           |           | - Implement monotonic burst draining in crates/plowrt/src/sched/admission.rs|
|           |           | - Implement nominal steady-state floor in sched/rungs.rs                   |
|           |           | - Correct Tier 1/3 sliding-ring memory budget sizing in memory/budget.rs   |
+-----------+-----------+----------------------------------------------------------------------------+
|    M1     | Weeks 3-5 | Discrete-Event Simulation & Fuzzing Harness                                |
|           |           | - Implement headless DES simulation harness (sched/sim.rs)                 |
|           |           | - Build proptest suite fuzzing 1M state transitions                        |
|           |           | - Establish automated CI invariant gate for all future scheduler PRs       |
+-----------+-----------+----------------------------------------------------------------------------+
|    M2     | Weeks 6-9 | Devgen 2.0 Componentization & Strongly Typed IR ABI                       |
|           |           | - Decompose monolithic emitters into modular components (norm, rope, moe)   |
|           |           | - Replace raw i[0..7] with typed OpInstruction structs                     |
|           |           | - Unify weight resolution and numerical correctness sanity suite          |
+-----------+-----------+----------------------------------------------------------------------------+
|    M3     | Weeks 10-13| Autonomous Tuning Pipeline & Certified Roofline Gate                      |
|           |           | - Implement calibrated_roofline.py empirical probe                         |
|           |           | - Build single-block micro-harness for candidate search                    |
|           |           | - Enforce >=80% / >=75% roofline gate with direct tunedb auto-serialization|
+-----------+-----------+----------------------------------------------------------------------------+
|    M4     | Weeks 14-16| Multi-Modal & Hybrid Architecture Rapid-Bringup Playbook                  |
|           |           | - Deliver declarative templates for Whisper/Audio, Vision MM, and KDA      |
|           |           | - Validate sub-48h bringup for novel checkpoint architectures               |
|           |           | - Cross-hardware verification across NVIDIA SM90 and AMD CDNA3             |
+----------------------------------------------------------------------------------------------------+
```

### 4.1 Detailed Workstreams

#### Workstream 1: `devgen 2.0` Declarative Model Component Library
- **Goal**: Replace 40,000+ lines of duplicated architecture code with composable, reusable building blocks.
- **Key Deliverables**:
  - `crates/devgen/src/components/norm.rs`: Generic RMSNorm, LayerNorm, and QK-Norm with explicit epsilon placement and dtype casting semantics.
  - `crates/devgen/src/components/rope.rs`: Unified rotary embedding generator handling Neox/GPT-J interleaving, partial rotary dimensions, and frequency scaling.
  - `crates/devgen/src/components/attention.rs`: Composable attention interfaces for MHA, GQA, MLA, and hybrid linear recurrent scans (GDN/KDA).
  - `crates/devgen/src/components/moe.rs`: Unified sparse MoE routing, top-k expert selection, token dropping, and shared expert accumulation.
  - `crates/packet/src/typed_abi.rs`: Strongly typed wrappers over `Instruction` immediates (`GemmArgs`, `FlashAttentionArgs`, `NormArgs`), eliminating raw indexing errors.

#### Workstream 2: Autonomous Kernel Tuning and Roofline Certification
- **Goal**: Establish push-button kernel optimization with mathematical performance guarantees.
- **Key Deliverables**:
  - `scripts/campaign/calibrated_roofline.py`: Standardized GPU micro-benchmarks measuring empirical HBM/L2 bandwidth and Tensor Core peaks.
  - `scripts/campaign/single_block_harness.py`: Isolated single-block runner measuring candidate kernel efficiency against calibrated roofline.
  - `scripts/campaign/certify_kernel.py`: Automated gate enforcing $\ge 80\%$ (decode) and $\ge 75\%$ (prefill) thresholds.
  - `crates/tunedb/src/auto_record.rs`: Automated Rust/Python bridge writing winning candidates directly to `crates/tunedb/records/`.

#### Workstream 3: `plowrt` Stability Invariants and Hardened Memory Governance
- **Goal**: Guarantee zero admission flapping, zero memory exhaustion traps, and deterministic scheduling.
- **Key Deliverables**:
  - Monotonic burst draining controller in `crates/plowrt/src/sched/admission.rs`.
  - Nominal throughput floor and dwell-time hysteresis in `crates/plowrt/src/sched/rungs.rs`.
  - Tiered memory budget allocation and sliding-ring burst protection in `crates/plowrt/src/memory/budget.rs`.
  - Headless discrete-event simulation and property-based fuzzing harness in `crates/plowrt/src/sched/sim.rs`.

#### Workstream 4: Cross-Hardware Portability & Unified ISA Abstraction
- **Goal**: Ensure seamless compilation and kernel execution across NVIDIA and AMD hardware.
- **Key Deliverables**:
  - Audit and harmonize target-divergent flags (`PLOW_UNISEG`, `PLOW_FP8`, `PLOW_L2_PLACE`).
  - Native support for AMD CDNA3/RDNA3 wave32/wave64 execution without register spills.
  - Automated FP8 quantization conversion layer (`E4M3FN` $\leftrightarrow$ `E4M3FNUZ`).

---

## 5. Verification and Acceptance Criteria

Every milestone must satisfy clear, measurable acceptance criteria before being marked complete:

| Component | Metric | Target Criteria |
| :--- | :--- | :--- |
| **Model Bringup Latency** | Time to First Token (TTFT) Bringup | $\le 48$ hours from HF checkpoint release to functional serving |
| **Decode Kernel Quality** | Empirical Memory Bandwidth Utilization | $\ge 80\%$ of calibrated HBM peak on single-block harness |
| **Prefill Kernel Quality**| Empirical Tensor Core Utilization | $\ge 75\%$ of calibrated Tensor Core peak on single-block harness |
| **Admission Stability**   | Concurrency Oscillation Rate | 0 downshift flaps during continuous saturated queueing |
| **Memory Safety**         | Out-of-Memory (OOM) Rate | 0 physical allocation failures across 1,000,000 simulated fuzz traces |
| **Cross-Platform Parity** | Target Divergence Rate | 100% test pass rate across both NVIDIA SM90 and AMD CDNA3 targets |

---

## 6. Architectural Summary

By decoupling model architecture definitions from kernel-level packing, establishing an automated single-block roofline certification gate, and replacing heuristic runtime controllers with mathematically invariant state machines, this roadmap transforms `plow` into an industrial-grade, highly agile inference platform capable of day-zero model onboarding with uncompromised serving stability.
