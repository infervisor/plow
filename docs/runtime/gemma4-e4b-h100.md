# Gemma-4-E4B on NVIDIA H100

Crisp technical summary of Gemma-4-E4B bringup and serving on NVIDIA H100 (`sm_90a`).

---

## 1. Configuration & Performance

* **Architecture:** 4B parameter dense model optimized for low-latency edge and server deployments.
* **Hardware Target:** NVIDIA H100-SXM5 (80GB HBM3, 3.35 TB/s peak bandwidth), TP1.
* **Precision:** BF16 activations with FP8 Tensor Core prefill GEMMs.

---

## 2. Serving Qualification

* **Status:** Fully serving qualified at TP1.
* **Latency Profile:** Sub-8ms TTFT at context 2048; TPOT $\le 4.2\text{ ms}$ at nominal concurrency.
* **Recipe:** Canonical recipe in `recipes/infervisor/gemma4-e4b/gemma4_e4b_h100.toml`.
