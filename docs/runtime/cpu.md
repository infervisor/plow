# CPU Execution Tier (AMX & AVX-512)

Crisp technical specification of CPU inference acceleration in `plowrt`.

---

## 1. Architecture & Capabilities

* **Target Processors:** Intel Xeon 6 (AMX TMUL matrix accelerator, AVX-512) and AMD EPYC 9654 (AVX-512).
* **Matrix Acceleration:** AMX Tile Matrix Multiplication units execute 16×16 BF16 outer products at peak core frequency.
* **Virtual Executor Model:** Packet `n_cu` represents virtual executors; thread worker pool dynamically maps work slices to physical cores.

---

## 2. Key Optimizations

* **Combining-Tree Counters (`PLOW_CPU_COMBINE=16`):** Eliminates atomic contention on completion counters across wide socket topologies (up to 128 cores).
* **NUMA Interleaving:** Memory allocations bound to socket-local NUMA nodes (`PLOW_CPU_NUMA=1`) to avoid cross-socket UPI/Infinity Fabric traffic.
* **Zero Spin-Waiting:** Workers transition to low-power pause or event waits when queue is empty (`PLOW_CPU_SPIN_US=0`).

---

## 3. Verification Commands

```bash
# Run CPU execution test suite
cargo test -p plowrt --lib cpu
```
