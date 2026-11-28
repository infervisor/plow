# Qwen-3.5-GDN on MI300X

Crisp summary of Qwen-3.5-GDN linear attention and gated recurrent bringup on AMD MI300X (`gfx942`).

---

## 1. Model Configuration

* **Architecture:** Gated Delta Network (GDN) linear attention with 1D short convolutions and gated RMSNorm.
* **Target Hardware:** MI300X (`gfx942`), TP1 / TP8.
* **Precision:** BF16 activations and weights; FP32 scan recurrence.

---

## 2. Kernel Arm Mapping

Plow maps GDN operations to dedicated CDNA3 kernels in `runtime/amd/interp.hip`:
* `QWEN_GDN_CONV`: 1D causal depthwise convolution across state buffers.
* `QWEN_GDN_STEP`: Recurrent step scan with exponential decay gating.
* `QWEN_GATED_NORM`: Fused Sigmoid gating with RMSNorm normalization.
* `QWEN_HEADNORM_ROPE`: Per-head normalization followed by rotary positional embedding.

---

## 3. Bringup Status

* All 8 GDN kernel arms implemented and qualified in `runtime/amd/interp.hip`.
* Numerics validated against reference PyTorch model within float32 error $\le 10^{-4}$.
* Packet emitter lowers full GDN blocks into native AQL dispatch sequences.
