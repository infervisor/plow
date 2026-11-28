# DeepSeek-V4-Flash on MI300X

Crisp summary of model architecture, sharding, and bringup status on AMD MI300X (`gfx942`).

---

## 1. Model Configuration

* **Architecture:** Multi-Head Latent Attention (MLA) + 256 Routed Experts + Engram CSA2.
* **Checkpoint:** 72,317 tensors across 48 shards (~167 GB).
* **Target Hardware:** 8x MI300X (`gfx942`), TP8 tensor parallelism.
* **Precision:** FP8 activation and weights with FP32 normalization.

---

## 2. Kernel & Tensor Sharding

* **MLA Attention:** Compressed latent vectors $c_t$ and $c_t^{kv}$ projected per rank; RoPE applied to decoupled query/key heads ($D_{\text{rope}} = 64$).
* **MoE Dispatch:** Top-8 routing across 256 experts (32 experts per rank at TP8); ragless routing buffers with all-to-all collective exchange.
* **Engram CSA2:** N-gram lookup table and embedding projection co-allocated in device memory.

---

## 3. Bringup Status

* **IR & Devblob:** Operator DAG shape inference qualified in `crates/nn-graph`. Single-block packet emission verified (`crates/devgen`).
* **HSA Dispatch:** Direct AQL dispatch tested; interpreter GEMM fallbacks mapped.
* **Serving Gate:** Block parity verified against reference oracle. Full-model serving under active bringup.
