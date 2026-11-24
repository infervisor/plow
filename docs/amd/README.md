# AMD ROCm Architecture & Model Bringup

Crisp technical reference for AMD CDNA targets (MI300X, MI325X, MI350X, MI355X) in Plow.

---

## 1. Hardware Targets & Ceilings

| Target | Architecture | Compute Units | Memory / Bandwidth | MFMA / Matrix Capabilities |
|---|---|---|---|---|
| **MI300X / MI325X** | `gfx942` (CDNA3) | 304 CUs (8 XCDs) | 192–256 GB HBM3 / 5.3 TB/s | `v_mfma_f32_16x16x16_fp8`<br>`v_mfma_f32_32x32x8_bf16`<br>LDS 64 KB / CU |
| **MI350X / MI355X** | `gfx950` (CDNA4) | 304 CUs (8 XCDs) | 288 GB HBM3e / 8.0 TB/s | Native OCP FP4/FP6/FP8<br>2.6× FP8 FLOP density<br>Enhanced LDS dual-issue |

* **Hardware Fingerprint:** Probes are calibrated via `scripts/bench/calibrated_roofline.py` and stored in `perf-data/probes/`.
* **Queueing:** Every AMD GPU execution must use `scripts/bench/gpuq.py submit <label> <ngpu> <cmd...>`.

---

## 2. Model Support Matrix

| Model Family | Target GPU / TP | Precision | Core Architecture | Current Bringup Status |
|---|---|---|---|---|
| **DeepSeek-V4-Flash / V4.1** | MI300X (TP8) | FP8 / BF16 | MLA Attention + 256 MoE Experts + Engram CSA2 | IR lowering & block emit verified; parser qualified. |
| **Gemma-4-31B** | MI300X (TP1) | BF16 | Dense Multi-Head Attention, Context 8192 | Serving qualified; packed prefill with runtime fusion. |
| **GLM-5.3** | MI300X / MI350X (TP8) | FP8 / MXFP4 | Dense MLA + Rowband + DSA Prefill | **Production Serving Qualified** at TP8 (`glm53_fp8_gfx942_tp8`). |
| **Kimi-K3** | MI325X / MI355X (TP8) | FP8 / BF16 | Hybrid 69 KDA Linear Attn + 24 Gated MLA Layers | Single-block devblob emit and disasm qualified. |
| **Qwen-3.5-GDN** | MI300X (TP1/TP8) | BF16 / FP8 | GDN Linear Recurrent State + Conv Chunked Scan | Kernel arms in `runtime/amd/interp.hip` qualified. |

---

## 3. HSA Runtime Dispatch Architecture

Plow bypasses standard HIP runtime layers for latency-critical paths:
* **AQL Packet Submission:** `crates/plowrt/src/exec/amd.rs` enqueues Architected Queuing Language packets directly to HSA hardware queues.
* **Decode GEMV / Attention:** Dispatched as a single cooperative workgroup launch per token, achieving ~89% of practical memory bandwidth ceiling.
* **Prefill GEMM:** Executed via segmented AQL dispatches across co-loaded code objects (Tensile / hipBLASLt assembly kernels) loaded directly into the HSA address space.
* **Multi-GPU Communication:** Hardware cross-ring collectives driven via HSA signal completion barriers across the 8 XCD interconnects.

---

## 4. Numerical Invariants on CDNA

1. **Recurrent State Layout (GDN / KDA):** State tensors $[B, H, K, V]$ require strict contiguous stride layout for prefix scans. Any non-contiguous striding triggers silent divergence.
2. **MFMA Tile Divisibility:** Projection matrices $N$ and $K$ must divide cleanly by MFMA tile boundaries (`16x16x16` or `32x32x8`); ragged tiles require boundary padding in `crates/devgen`.
3. **OCP FP8 / MXFP4 Scales:** Scale factors must align with block sizes (32 or 64 elements) without intermediate precision downcasting.
4. **RMSNorm Epsilon Placement:** Epsilon must remain strictly inside the square root in FP32 prior to activation casting.

---

## 5. Verification & Testing

* **Oracle Microbenchmarks:** `runtime/tests/*_test.hip` validate individual kernel parity against host CPU implementations.
* **Single Block Sweep:** Mandatory Stage 5 screen:
  ```bash
  python3 scripts/campaign/campaign.py nv-block-ab \
    --control <base> --treatment <cand> --verb bench --ceilings perf-data/probes/<fingerprint>.json
  ```
* **Recipe Validation:** Canonical recipes live in `recipes/infervisor/<model>/<label>.toml` and validate via `python3 scripts/check_recipe.py`.
