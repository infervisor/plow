# Data-Parallel Runtime Serving

Crisp technical specification of multi-GPU Data-Parallel (`--dp`) serving in `plowrt`.

---

## 1. Overview & Architecture

* **Single Process / Multi-Device:** A single `plowrt serve` coordinator drives independent device worker engines across 2 to 8 GPUs.
* **Independent Execution:** Each DP rank manages its own VMM allocation, KV cache pools, and CUDAGraph/AQL dispatch queues.
* **NUMA Socket Affinity:** Workers are bound to the specific NUMA node and CPU cores local to their target GPU (`PLOW_DP_NUMA_PIN=1`).

---

## 2. Load Balancing Policies

* **Round-Robin:** Dispatches incoming sessions evenly across active DP ranks.
* **Least-Loaded:** Directs requests to the worker with the lowest active token count or highest Tier 2 memory headroom.
* **Sticky Routing:** Multi-turn sessions route to the rank holding the pre-existing cached KV context.

---

## 3. Serving Invocation

```bash
plowrt serve --recipe recipes/infervisor/<model>/<label>.toml --dp 8 --port 8100
```
