# Stage 8 — Rung Campaign

Iterative optimization of an individual compiled rung for an already-serving model.

* **Definition:** A **rung** is an execution configuration defined by: `Program × Row Class × Prior Class × Execution Route` (e.g. 8192-row prefill program with long prior, or decode rung at fixed context).
* **Core Principle:** One rung, one lever, evaluated through the rigorous T1–T4 promotion ladder.
* **Precondition:** Stage 6 and Stage 7 complete.
* **Architecture Reference:** [`docs/bringup/agent-tools.md`](agent-tools.md) §4.

---

## 1. Six-Step Optimization Loop

```
0. Pick Rung  →  1. Rung Card  →  2. Attribute  →  3. Lever Card  →  4. T1–T4 Ladder  →  5. Decide & Record
```

### Step 0: Prioritize Rung
Rank candidate rungs by potential system impact:
$$\text{Priority} = \text{Workload Share} \times \text{Gap to Floor}$$
* **Workload Share:** Execution time fraction spent in this rung (from `PLOW_TICK_LOG`).
* **Gap to Floor:** Difference between measured latency and analytical roofline floor.

### Step 1: Complete Rung Baseline Card
Document baseline metrics before editing code:
* **Identity:** Program index, role (`plowrt disasm`), bucket dimensions, route topology.
* **Baseline Latency:** Median latency of interleaved controls (`ctrl1` and `ctrl2`) in a single lease.
* **Value Floor:** Numerical divergence floor (logits and KV rel-$L_2$ between identical controls).

### Step 2: Attribute Bottlenecks
Capture fine-grained execution trace (`PLOW_TRACE_RAW=1`, `PLOW_TICK_LOG=1`):
* Decompose execution into: Native kernels, interpreter GEMMs, glue operations, and launch boundaries.
* Isolate fixed launch costs vs per-token processing: $2 \cdot t(T/2) - t(T)$.
* Compute physical distance to floor for each individual component.

### Step 3: Complete Lever Hypothesis Card
Document proposed optimization hypothesis:
* **Hypothesis:** Specific architectural or kernel mechanism targeted.
* **Predicted Gain:** Quantitative latency reduction derived from trace attribution.
* **Scope:** Exhaustive list of affected programs, opcodes, and sister rungs.
* **Rollback Mechanism:** Registered knob name and off-value in `knob_spec.rs`.
* **Falsifier:** Cheapest test that would refute the hypothesis.

---

## 2. Promotion Ladder (T1 → T4)

Every candidate optimization must climb the four-tier promotion ladder:

| Tier | Evaluation Environment | Promotion Pass Criteria |
|---|---|---|
| **T1: CPU** | Local machine | Compiler tests pass; `model.pkt` byte identity preserved with knob off; diff restricted to scope. |
| **T2: Single GPU** | Isolated kernel harness | Outperforms baseline beyond noise floor on captured production operands. |
| **T3: Single Block** | Single layer block (`--block`) | Block latency improves beyond noise ($S_{\text{treat}} \le 3 \times D_{\text{ctrl}}$); outputs within value floor. |
| **T4: Full Serving** | Full engine under traffic | Zero failed requests; TPOT/TTFT improved without tail regression; all accuracy gates pass. |

---

## 3. NVIDIA Exact-Cell Tuner

For exact GEMM and attention shape tuning on NVIDIA architectures:
```bash
python3 scripts/gemma4_h100_kernel_tuner.py plan --spec <spec> --summary <out>
python3 scripts/gemma4_h100_kernel_tuner.py run  --spec <spec> --summary <out>
```
*(Self-leasing: do not wrap in `gpuq.py submit`).*
* Ranks tuning cells by $\text{occurrences} \times \text{control\_time}$.
* Promoted candidates must subsequently pass Tier 3 single-block evaluation before entering the production recipe.

---

## 4. Decision & Recording

* **Decision:** Land (promote to production), Park (retain branch with documented findings), or Kill (revert).
* **Production Landing:** Promoted wins update the canonical recipe in `recipes/infervisor/<model>/<label>.toml`.
* **Knob Registration:** All new knobs must be registered in `crates/*/src/knob_spec.rs` and documented in `docs/flags-reference.md`.
* **Ledger Update:** Candidate measurements are recorded in the campaign ledger in scratch. Never commit raw JSON or trial profiles to Git.

---

## 5. Gate Checklist

Before landing optimization:
- [ ] Hypothesis Card filed with quantitative gain prediction and rollback knob.
- [ ] Candidate successfully passes all four tiers (T1, T2, T3, T4) in sequence.
- [ ] Numerical drift stays strictly within baseline value floor.
- [ ] Canonical recipe TOML updated and validated with `check_recipe.py`.
- [ ] New knobs registered in `knob_spec.rs` and verified via cargo tests.
