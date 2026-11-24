# L40S Deployment Recipes

Crisp technical summary of model deployment and serving ceilings on NVIDIA L40S (`sm_89`).

---

## 1. Architecture & Limits

* **Target Hardware:** NVIDIA L40S (48GB GDDR6, 862 GB/s memory bandwidth, Ada Lovelace).
* **Execution Bound:** Strongly memory-bandwidth bound on decode GEMV; prefill accelerated by FP8 Tensor Cores.

---

## 2. Deployed Recipes

* **Gemma-4-E2B / E4B:** Full-model serving qualified at TP1; context up to 4096 tokens.
* **Speech Pipelines:** ASR and TTS audio pipelines running native Ada CUBINs without CPU offload.
