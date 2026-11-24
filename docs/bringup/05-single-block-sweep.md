# Stage 5 — Single-Block Sweep

Validates a fully fused layer block in isolation on `$GPU` before full-model assembly.

* **The Single-Block Hard Rule:** Every performance PR and campaign report must start with the block-vs-roofline table on a single block. Full-model evaluation is prohibited until the isolated block is certified.
* **Precondition:** Stage 4 passes. Tuned kernels serialized in `perf-data/tuning/<fingerprint>.json`.
* **Output:** Certified 5-column block-vs-roofline table and verified numerical equivalence against reference host oracle.
* **Architecture Reference:** [`docs/bringup/agent-tools.md`](agent-tools.md) §1 and §3.

---

## 1. Single-Block Emission

Emit the isolated layer block devblob:
```bash
nix develop --command python3 scripts/campaign/campaign.py build <recipe> \
  --block 0 \
  --out <scratch>/block_assets
```
Verifies packet structure, weight slices, and intermediate memory bindings for a single transformer layer.

---

## 2. Reference Host Oracle Verification

Execute the reference host oracle on physical GPU to confirm numerical parity:
```bash
scripts/bench/gpuq.py submit block-oracle 1 \
  ./build/runtime/tests/layer_block_oracle_test \
    --packet <scratch>/block_assets/model.pkt \
    --gpu "$GPU"
```
* **Correctness Standard:** Max absolute error $\le 10^{-4}$, relative error $\le 10^{-3}$ against un-fused reference layer outputs across identical inputs.

---

## 3. Four-Arm Interleaved A/B Protocol

To eliminate GPU frequency drift, thermal throttling, and co-tenant interference, benchmarks execute four interleaved arms in a single lease:
$$\text{Sequence: } \text{ctrl}_1 \longrightarrow \text{treat}_1 \longrightarrow \text{ctrl}_2 \longrightarrow \text{treat}_2$$

### Execution Command
```bash
scripts/bench/gpuq.py submit block-ab 1 \
  python3 scripts/campaign/campaign.py nv-block-ab \
    --control <base_assets> \
    --treatment <cand_assets> \
    --verb bench \
    --args "--batch 1,8,32 --ctx 512,2048" \
    --ceilings "perf-data/probes/${HW_FINGERPRINT}.json" \
    --out <scratch>/block_ab_results.json
```

### Noise Filter Criterion
The measurement is valid only if treatment delta exceeds control drift by a factor of 3:
$$D_{\text{ctrl}} = |t_{\text{ctrl1}} - t_{\text{ctrl2}}|$$
$$S_{\text{treat}} = |t_{\text{treat1}} - t_{\text{treat2}}|$$
$$\text{Assertion: } S_{\text{treat}} \le 3 \times D_{\text{ctrl}}$$
If $D_{\text{ctrl}}$ exceeds 2% of median control latency, the run is discarded and re-queued.

---

## 4. Mandatory 5-Column Block-vs-Roofline Table

Generate the standard 5-column report table:
```bash
python3 scripts/campaign/block_roofline.py <scratch>/block_assets \
  --ceilings "perf-data/probes/${HW_FINGERPRINT}.json" \
  --plow <scratch>/sweep.json \
  --current <prod_sweep.json> \
  --out <scratch>/block_table.md
```

### Table Schema
| Shape (B, S) | Measured ($\mu$s) | Roofline Floor ($\mu$s) | % of Floor | Delta vs Current |
|---|---|---|---|---|
| `(1, 128)` | 18.2 | 15.1 | 83.0% | -1.4 $\mu$s |
| `(8, 128)` | 24.6 | 20.8 | 84.5% | -2.1 $\mu$s |
| `(32, 512)` | 94.3 | 76.4 | 81.0% | -5.8 $\mu$s |

* `% of Floor` = $\frac{t_{\text{floor}}}{t_{\text{measured}}} \times 100\%$.
* `Delta vs Current` = candidate latency minus production baseline latency at identical batch and context.

---

## 5. Certification Gate

Validate single-block latency using `certify_kernel.py`:
```bash
python3 scripts/campaign/certify_kernel.py \
  --input <scratch>/sweep.json \
  --ceilings "perf-data/probes/${HW_FINGERPRINT}.json"
```
Gate passes only when all evaluated shapes exceed $\ge 80\%$ decode efficiency and $\ge 75\%$ prefill efficiency.

---

## 6. Gate Checklist

Before advancing to Stage 6:
- [ ] Single-block devblob emits cleanly without compilation errors.
- [ ] Numerical parity verified against host oracle ($L_{\infty} \le 10^{-4}$, $L_2 \le 10^{-3}$).
- [ ] Interleaved A/B measurement passes noise filter ($S_{\text{treat}} \le 3 \times D_{\text{ctrl}}$).
- [ ] Mandatory 5-column block-vs-roofline table populated for all key shapes.
- [ ] `certify_kernel.py` confirms block meets efficiency floors.
