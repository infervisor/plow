# Stage 4 — Kernel Tuning to the Roofline

Hardware-dependent stage. Tunes hot kernel families (prefill GEMM, decode GEMV, attention, MoE) on `$GPU` to a calibrated fraction of the hardware roofline.

* **Precondition:** Stage 3 passes. Target hardware parameters filled in [`target.md`](target.md) (`$GPU`, `$ISA`, `$BW_BOUND`, `$COMPUTE_CEIL`, `$NCU`).
* **Hardware Guide:** Consult [`isa/README.md`](isa/README.md) for target instruction constraints before editing kernel code.
* **Output:** Tuned kernel configurations recorded in `perf-data/tuning/<fingerprint>.json` and certified against calibrated roofline floors.
* **Architecture Reference:** [`docs/arch/11-tuning-coverage.md`](../arch/11-tuning-coverage.md) and [`docs/arch/12-using-the-tuner.md`](../arch/12-using-the-tuner.md).

---

## 1. Principles & Promotion Order

Follow strict tuning progression:
1. **Decode GEMV / FlashAttention:** Prioritize memory bandwidth efficiency on batch-1 / batch-nominal.
2. **Prefill GEMM:** Prioritize Tensor Core compute utilization on chunked query lengths.
3. **MoE Routing & Dispatch:** Minimize scatter/gather latency and synchronization overhead.
4. **Norms / Elementwise:** Fuse into adjacent projection kernels where possible.

Never optimize high-level scheduling knobs (Stage 6) before individual kernels reach their calibrated roofline floors.

---

## 2. Strongly Typed `BlockContract` & Arithmetic Intensity

Every kernel adheres to an analytical `BlockContract` categorizing its operational bound:

### Memory-Bound Decode Operations (`MemoryBoundHbm`)
For decode GEMV and attention reading KV cache:
$$I_{\text{decode}} = \frac{2 \cdot B \cdot K}{B \cdot K \cdot \text{dtype\_size} + B \cdot D \cdot \text{dtype\_size}} \quad [\text{FLOP/byte}]$$
$$\text{Theoretical Floor: } t_{\text{floor}} = \frac{\text{Bytes Transferred}}{BW_{\text{meas}}} \quad [\mu\text{s}]$$

### Compute-Bound Prefill Operations (`ComputeBoundTensorCore`)
For prefill GEMM ($M \times N \times K$):
$$I_{\text{prefill}} = \frac{2 \cdot M \cdot N \cdot K}{(M \cdot K + K \cdot N + M \cdot N) \cdot \text{dtype\_size}} \quad [\text{FLOP/byte}]$$
$$\text{Theoretical Floor: } t_{\text{floor}} = \frac{\text{FLOPs}}{\text{TFLOPS}_{\text{meas}}} \quad [\mu\text{s}]$$

---

## 3. Empirical Ceiling Calibration

Measure achievable hardware ceilings before running kernel sweeps:
```bash
scripts/bench/gpuq.py submit probe 1 \
  python3 scripts/bench/calibrated_roofline.py \
    --gpu "$GPU" \
    --out "perf-data/probes/${HW_FINGERPRINT}.json"
```
Produces measured peak memory bandwidth ($BW_{\text{meas}}$ in GB/s) and peak GEMM throughput ($\text{TFLOPS}_{\text{meas}}$ in TFLOPS).

---

## 4. Candidate Search & Tuning

### Step 1: Formulate Search Space
Explore tile sizes, warp layouts, pipeline stages, and register limits within hardware constraints:
* NVIDIA Hopper/Blackwell: TMA warp group layouts, cluster sizes (`cluster_m`, `cluster_n`), swizzling modes.
* AMD CDNA: LDS allocation per workgroup, MFMA instruction shapes (`32x32x8`, `16x16x16`), wave counts.

### Step 2: Validate Numerical Equivalence
Validate every candidate against host reference oracles (`runtime/tests/*_block_*_test.c`):
```bash
scripts/bench/gpuq.py submit test-oracle 1 \
  ./build/runtime/tests/gemm_oracle_test --gpu "$GPU" --tolerance 1e-4
```
Ensure float32 absolute error $\le 10^{-4}$ and relative error $\le 10^{-3}$.

### Step 3: Run Microbenchmark Sweep
```bash
scripts/bench/gpuq.py submit tune-sweep 1 \
  python3 scripts/campaign/tune_kernel.py \
    --family prefill_gemm \
    --shapes-from <scratch>/shapes.json \
    --probe "perf-data/probes/${HW_FINGERPRINT}.json" \
    --out <scratch>/candidate_sweep.json
```

---

## 5. Certification Gate & Serialization

### Certification Thresholds
Evaluate candidate measurements via `certify_kernel.py`:
```bash
python3 scripts/campaign/certify_kernel.py \
  --input <scratch>/candidate_sweep.json \
  --ceilings "perf-data/probes/${HW_FINGERPRINT}.json"
```

Hard acceptance thresholds:
* **Decode GEMV / Attention:** $\eta_{\text{mem}} = \frac{t_{\text{floor}}}{t_{\text{measured}}} \ge 80\%$.
* **Prefill GEMM:** $\eta_{\text{compute}} = \frac{t_{\text{floor}}}{t_{\text{measured}}} \ge 75\%$.

### Serialization to `tunedb`
Winning configurations are serialized to `perf-data/tuning/<fingerprint>.json`:
```bash
python3 scripts/campaign/tune_kernel.py serialize \
  --input <scratch>/candidate_sweep.json \
  --dest "perf-data/tuning/${HW_FINGERPRINT}.json"
```
The compiler automatically reads this database during packet emission.

---

## 6. Gate Checklist

Before advancing to Stage 5:
- [ ] Peak ceilings measured and written to `perf-data/probes/<fingerprint>.json`.
- [ ] All candidate kernels produce numerical outputs matching host oracles within tolerances.
- [ ] Decode kernels achieve $\ge 80\%$ of calibrated memory roofline floor.
- [ ] Prefill kernels achieve $\ge 75\%$ of calibrated Tensor Core compute floor.
- [ ] Winning parameters committed to `perf-data/tuning/<fingerprint>.json`.
