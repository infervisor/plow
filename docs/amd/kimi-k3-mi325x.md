# Kimi-K3 on MI325X & MI355X

Crisp summary of Kimi-K3 hybrid architecture bringup on AMD Instinct CDNA3/CDNA4 (`gfx942` / `gfx950`).

---

## 1. Model Configuration

* **Architecture:** Hybrid Linear/Attention Model: 69 KDA (Kimi Delta Attention) layers + 24 Gated MLA layers.
* **Target Hardware:** 8x MI325X (`gfx942`, 256 GB HBM3) or 8x MI355X (`gfx950`, CDNA4, TP8).
* **Precision:** FP8 activation and weights with FP32 recurrent state accumulators.

---

## 2. KDA & Recurrent State Execution

* **Recurrent State:** State tensors $[B, H, K, V]$ require strict contiguous stride layout for chunked prefix scan kernels.
* **Paging & Rings:** Hybrid sliding rings in Tier 1; global MLA attention layers use Tier 2 dynamic paged KV pool.
* **Multi-GPU Sharding:** KDA heads sharded evenly across ranks ($H / \text{TP}$); MLA latent projections sharded via TP8 row/column splits.

---

## 3. Bringup Status

* Single-block devblob emission and disassembly verified (`crates/devgen`).
* KDA forward recurrence verified against Flash-Linear-Attention oracles (`fla/ops/kda`).
* End-to-end full serving bringup active on CDNA4 hardware.
