# Gemma-4-31B on MI300X

Crisp summary of Gemma-4-31B dense model bringup, packed prefill, and serving performance on MI300X (`gfx942`).

---

## 1. Model Configuration

* **Architecture:** 31B parameter dense decoder; context length up to 8192 tokens.
* **Target Hardware:** Single MI300X (`gfx942`, 304 CUs, 192 GB HBM3), TP1.
* **Precision:** BF16 activations and weights (`bfloat16`).

---

## 2. Runtime Optimization Levers

* **Packed Prefill:** Fuses multiple independent prefix segments into uniform GEMM shapes (`PLOW_PF_CHUNK=2048`), maximizing MFMA compute occupancy.
* **Terminal-Prefill Completion:** Batches the final prefill token directly with concurrent decode requests, cutting first-token latency by 9–10%.
* **Sliding-Window Attention:** Fixed 4096-token sliding window managed via Tier 1 circular ring buffers (`PLOW_VMM_RING_PARK`).

---

## 3. Serving Verification & Status

* **Status:** Fully serving qualified at TP1.
* **Parity:** Evaluated via `llm_fp32_ref`; KL divergence trim-mean $\le 1.05\times$ reference peer.
* **Recipe:** Canonical recipe in `recipes/infervisor/gemma4-31b/gemma4_gfx942_tp1.toml`.
