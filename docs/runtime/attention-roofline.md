# Attention Roofline Analysis

Crisp technical summary of attention kernel performance bounds and rooflines.

---

## 1. Analytical Ceiling

* **Decode Attention:** Memory bound on reading the KV cache:
  $$t_{\text{floor}} = \frac{2 \cdot B \cdot K \cdot \text{dtype\_size}}{BW_{\text{meas}}}$$
* **Prefill FlashAttention:** Compute bound on Tensor Cores for long sequences ($S \ge 1024$):
  $$t_{\text{floor}} = \frac{4 \cdot B \cdot S^2 \cdot D}{\text{TFLOPS}_{\text{meas}}}$$

---

## 2. Empirical Efficiency Floors

* Decode attention kernels must achieve $\ge 80\%$ of measured memory bandwidth floor.
* Prefill FlashAttention kernels must achieve $\ge 75\%$ of measured Tensor Core throughput floor.
* Measured via `python3 scripts/campaign/certify_kernel.py`.
