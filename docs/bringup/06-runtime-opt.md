# Stage 6 — Runtime Optimization

Full-model execution stage. Configures memory governance, admission boundaries, and execution rungs on `$GPU` to maximize serving throughput under strict latency budgets.

* **Precondition:** Stage 5 passes. Single block certified against calibrated roofline.
* **Output:** Production recipe TOML in `recipes/infervisor/<model>/<label>.toml` with verified runtime stability invariants.
* **Architecture Reference:** [`docs/arch/06-runtime.md`](../arch/06-runtime.md), [`docs/arch/22-kv-cache.md`](../arch/22-kv-cache.md), and [`docs/arch/model_bringup_and_stability_roadmap.md`](../arch/model_bringup_and_stability_roadmap.md) §3.

---

## 1. Formal Runtime Stability Invariants

Serving configurations must strictly maintain three stability invariants ([`model_bringup_and_stability_roadmap.md`](../arch/model_bringup_and_stability_roadmap.md) §3):

### Invariant 1: Monotonic Burst Draining
When the system admits a burst exceeding nominal capacity $C_{\text{nominal}}$, admission capacity relaxes dynamically:
$$C_{\text{admit}}(t) = \max(C_{\text{nominal}}, N_{\text{active}}(t))$$
As active requests complete, $C_{\text{admit}}$ contracts monotonically ($\frac{d C_{\text{admit}}}{dt} \le 0$) until returning to $C_{\text{nominal}}$. No new arrivals may expand capacity above $C_{\text{nominal}}$ during draining.

### Invariant 2: Nominal Steady-State Floor
Under continuous saturated demand, the scheduler guarantees a sustained concurrency floor:
$$N_{\text{active}}(t) \ge C_{\text{nominal}}$$
System resources must never artificially throttle below nominal concurrency when the admission queue is non-empty.

### Invariant 3: Dwell Time & Hysteresis
Batch and rung reconfigurations must prevent oscillatory thrashing:
* **Minimum Dwell Time:** Any batch or rung state must persist for at least $T_{\text{dwell}} \ge 500\text{ ms}$ or $50\text{ steps}$ before re-evaluating.
* **Directional Hysteresis:** Upgrades require $+8\%$ sustained queue pressure; downgrades require $+12\%$ sustained queue deflation.

---

## 2. Tiered Memory Governance (Tiers 0–3)

Total GPU physical VRAM is governed by four non-overlapping tiers:

| Memory Tier | Allocation Type | Governance Rule |
|---|---|---|
| **Tier 0: Static Weights** | Engine & model parameters | Fixed at boot; immutable physical VMM allocation. |
| **Tier 1: Parked Sliding Rings** | Nominal concurrency rings | Pre-allocated ring buffers for $C_{\text{nominal}}$ sessions; pinned VRAM. |
| **Tier 2: Dynamic Paged KV** | Context expansion pool | Elastic page pool for token growth; **never cannibalized** by burst admissions. |
| **Tier 3: Elastic Burst Reserve** | Emergency headroom | Minimum 5% VRAM buffer reserved for transient activation spikes. |

### Hybrid Sliding-Ring Budget Calculation
For hybrid architectures combining local sliding-window layers with global full-attention layers:
$$M_{\text{parked}} = C_{\text{nominal}} \times L_{\text{sliding}} \times 2 \times N_{\text{kv\_heads}} \times D_{\text{head}} \times W_{\text{sliding}} \times \text{dtype\_size}$$
Tier 1 reserves exactly $M_{\text{parked}}$ bytes. Tier 2 receives all remaining allocatable VRAM divided into uniform page blocks (`PLOW_KV_PAGE_SIZE`).

---

## 3. Discrete-Event Simulator & Fuzzing Verification

Before deploying to physical hardware, verify scheduler logic against the discrete-event simulator:
```bash
cargo test -p plowrt --lib sched::sim
```
Runs property-based chaos testing across Poisson arrival spikes, sudden disconnections, and long-context saturation to verify Invariants 1–3 hold without deadlock or memory exhaustion.

---

## 4. Sequential Tuning Levers

Tune runtime parameters in strict sequential order:

### Lever 1: Batching & Rung Sizing
* Set prefill rungs to match input prompt distribution (e.g. 512, 2048, 8192).
* Set decode batch rungs to hardware sweet spots (e.g. 16, 32, 64, 128).
* Verify decode rungs avoid memory fragmentation and fit within CUDAGraph stream bounds.

### Lever 2: Dynamic KV Paging (`PLOW_KV_PAGE_SIZE`)
* Set KV page size (typically 16 or 32 tokens).
* Assert page allocation overhead remains under 1% of total step time.

### Lever 3: CUDAGraph Execution Bounds
* Pre-record graphs for all fixed decode rungs up to $C_{\text{nominal}}$.
* Ensure zero graph-recapture events occur during steady-state execution.

---

## 5. Recipe Generation

Document all finalized parameters into the canonical recipe TOML:
```bash
python3 scripts/campaign/render_recipe.py \
  --model <model_id> \
  --gpu "$GPU" \
  --cnominal 64 \
  --out recipes/infervisor/<model>/<label>.toml
```

Validate recipe syntax and completeness:
```bash
python3 scripts/check_recipe.py recipes/infervisor/<model>/<label>.toml
```

---

## 6. Gate Checklist

Before advancing to Stage 7:
- [ ] `cargo test -p plowrt --lib sched::sim` passes discrete-event invariant fuzzing.
- [ ] Invariants 1, 2, and 3 are verified under burst injection testing.
- [ ] Tiered Memory sizing verified: Tier 2 dynamic pool is isolated from burst cannibalization.
- [ ] CUDAGraph allocations pre-recorded for all nominal decode rungs without dynamic recaptures.
- [ ] Production recipe TOML passes `check_recipe.py` validation.
