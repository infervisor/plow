# Kernel Generators & AOT Synthesis

Crisp specification of Ahead-of-Time kernel generation and packaging in `plowrt`.

---

## 1. Architecture

* **Generator Tooling:** Python and meta-programming scripts in `scripts/gen_kernels/`, `scripts/fa4_aot/`, and `scripts/gdn_aot/`.
* **Target Emission:** Emits standalone device assembly (`.cubin` for NVIDIA, `.hsaco` for AMD).
* **Binary Integrity:** Emitted objects are bound to packets with cryptographic SHA-256 hashes recorded in `build.json`.

---

## 2. Qualification Criteria

* Every generated kernel must produce bit-exact or value-floor bounded numerical results against reference host oracles (`runtime/tests/*_test.c`).
* Kernels must achieve $\ge 80\%$ decode efficiency and $\ge 75\%$ prefill efficiency against calibrated hardware rooflines.
