# Plow Runtime (`plowrt`) Reference

Architecture and operational reference for the low-latency, zero-allocation Plow serving engine.

---

## 1. Core Architecture

`plowrt` is a high-performance serving runtime designed for sub-millisecond dispatch and strict latency SLAs:
* **Zero Hot-Path Allocation:** Static VMM virtual memory reservations established at boot; zero heap allocations during steady-state request execution.
* **Tiered Memory Governance:** Enforces Tiers 0–3 (static weights, parked sliding rings, dynamic paged KV pool, elastic reserve).
* **Hardware Independence:** Packet ABI contract decoupling architecture bytecode and tensor weights from host scheduling logic.

---

## 2. Multi-Target Execution Backends

| Backend | Supported Parts | Dispatch Mechanism | Matrix Core Acceleration |
|---|---|---|---|
| **NVIDIA (CUDA)** | sm_89 (L40S), sm_90a (H100/H200), sm_120 (Blackwell) | Direct driver API (`cuLaunchKernelEx`), CUDAGraph stream captures | TMA, WGMMA, Tensor Cores, FP8/FP4 |
| **AMD (ROCm/HSA)** | `gfx942` (MI300X/MI325X), `gfx950` (MI350X/MI355X) | Direct HSA AQL hardware queue packet submission | MFMA matrix cores, CDNA3/CDNA4 |
| **Intel/AMD (CPU)** | Xeon 6 (Granite Rapids), EPYC 9654 (Zen 4) | Multi-core work-stealing worker pool, combining-tree counters | AMX (TMUL), AVX-512, BF16 |

---

## 3. Subsystem Technical Guides

* [`host-path.md`](host-path.md): Request ingress, token scheduling, and host-device dispatch loop.
* [`prefix-cache.md`](prefix-cache.md): Content-addressed physical VMM prefix caching and zero-copy reuse.
* [`sessions.md`](sessions.md): Session lifecycle, context persistence, and eviction policies.
* [`data-parallel.md`](data-parallel.md): Multi-GPU data-parallel serving across independent workers.
* [`cpu.md`](cpu.md): High-throughput CPU execution tier with AMX/AVX-512.
* [`asr.md`](asr.md) & [`asr-packet-contract.md`](asr-packet-contract.md): Speech recognition pipeline and packet ABI.
* [`tts.md`](tts.md): Speech synthesis, codec streaming, and tokenization.
* [`kernel-generators.md`](kernel-generators.md): Ahead-of-Time kernel synthesis and tuning registry.

---

## 4. Hardware & Recipe References

* [`gemma4-e4b-h100.md`](gemma4-e4b-h100.md): Gemma 4 E4B execution on NVIDIA H100.
* [`h100-speech-recipes.md`](h100-speech-recipes.md): Speech model serving configurations on H100.
* [`l40s-recipes.md`](l40s-recipes.md): sm_89 deployment recipes and throughput ceilings on L40S.
* [`attention-roofline.md`](attention-roofline.md): Attention kernel roofline analysis and memory bandwidth benchmarks.
