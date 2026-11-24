# GLM-5.3 on MI300X & MI350X

Crisp summary of GLM-5.3 production serving bringup on AMD Instinct architectures (`gfx942` / `gfx950`).

---

## 1. Model Configuration

* **Architecture:** Dense MLA Transformer with Deep Residual Architecture (DSA) prefill acceleration.
* **Target Hardware:** 8x MI300X (`gfx942`, TP8) and 8x MI350X (`gfx950`, TP8).
* **Precision:** Production FP8 activations and weights; optional OCP MXFP4 on CDNA4.

---

## 2. Key Architecture Invariants

* **Rowband Buffering:** Row-banded memory allocations (`PLOW_GLM_ROWBAND`) reduce peak VMM footprint during long-context execution.
* **DSA Prefill:** Selective query sparsification triggered for sequence lengths exceeding threshold (`PLOW_GLM_DSA_PF_MIN_CTX=2048`).
* **Cross-Ring TP8 Collectives:** Hardware-accelerated all-reduce and all-gather via HSA signal barriers.

---

## 3. Serving Qualification

* **Production Status:** **Serving Qualified** on 8x MI300X at TP8 (`glm53_fp8_gfx942_tp8`).
* **Gate Check:** Bit-exact parity on reference prompts; throughput and TTFT exceed baseline targets across all Poisson arrival rates.
