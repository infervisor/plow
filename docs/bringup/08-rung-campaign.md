# Stage 8 — Rung Campaign

Iterative optimization of a single compiled rung for an already-serving model (Stage 6 complete).

* **Definition:** A **rung** is defined by: `Program × Row Class × Prior Class × Execution Route` (e.g. 8192-row prefill program with long prior, or decode rung at fixed context).
* **Core Principle:** One rung, one lever, evaluated through the T1–T4 promotion ladder.
* **Precondition:** Stage 6 complete. `target.md` parameters defined.

---

## 1. Six-Step Optimization Loop

```
0. Pick Rung  →  1. Rung Card  →  2. Attribute  →  3. Lever Card  →  4. T1–T4 Ladder  →  5. Decide & Record
```

### Step 0: Select Target Rung
Prioritize by: $\text{Priority} = \text{Workload Share} \times \text{Gap to Floor}$
* **Workload Share:** Proportion of execution time spent in rung (from `PLOW_TICK_LOG`).
* **Gap to Floor:** Difference between measured latency and analytical roofline floor.

### Step 1: Complete Rung Card
Document baseline before introducing changes:
* **Identity:** Program index, role (`plowrt disasm`), bucket dimensions, route topology.
* **Workload:** Dedicated isolating input (e.g. `plowrt bench --prefill-sweep --prompt-ids <ids>`).
* **Baseline:** Median latency of interleaved control runs (`ctrl1` and `ctrl2`) in a single lease.
* **Value Floor:** Numerical divergence floor (logits and KV rel-L2 between identical controls).

### Step 2: Attribute Bottlenecks
Capture isolated trace (`PLOW_TRACE_RAW=1`, `PLOW_TICK_LOG=1`, `PLOW_PREFILL_SEG_TIMING=1`):
* Classify time across: Native kernels, interpreter GEMMs, glue operations, and launch boundaries.
* Distinguish fixed launch costs from per-row processing: $2 \cdot t(T/2) - t(T)$.
* Compute distance to physical floor for each component.

### Step 3: Complete Lever Card
Document candidate hypothesis before modifying code:
* **Hypothesis:** Specific bottleneck mechanism targeted.
* **Predicted Gain:** Quantitative latency reduction derived from attribution.
* **Scope:** Exhaustive list of affected programs, opcodes, and sister rungs.
* **Numerics Contract:** Bit-identical or bounded by value floor.
* **Rollback Mechanism:** Registered knob name and off-value in `knob_spec.rs`.
* **Falsifier:** Cheapest test that would refute the hypothesis.

---

## 2. Promotion Ladder (T1 → T4)

Advancement requires measured gain exceeding noise threshold at each tier:

| Tier | Environment | Scope & Execution | Promotion Criteria |
|---|---|---|---|
| **T1: CPU** | Local machine | Compiler tests; `model.pkt` byte identity with knob disabled; instruction diff restricted to scope. | Build clean; diff matches declared scope exactly; untouched objects unchanged. |
| **T2: Single GPU** | Isolated kernel | Isolated harness on captured production operands with device-event timing. | Outperforms interpolated control beyond noise; predicted hardware counters move in target direction. |
| **T3: Single Block** | Isolated layer block | Single layer block (`--block` or `PLOW_LAYERS=N`) evaluated against calibrated block roofline. | Block latency improves beyond noise; numerical outputs stay within value floor; sister rungs un-degraded. |
| **T4: Full Serving** | Full engine | Interleaved serving arms (`ctrl1 → treat1 → ctrl2 → treat2`) under production traffic. | Zero failed requests; TPOT/TTFT improved without tail regression; all accuracy gates pass. |

### Ladder Execution Rules
1. **Control validation:** Benchmark harnesses must demonstrate zero difference between two control runs before testing candidates.
2. **Machine-checked pass/fail:** Automated scripts exit non-zero on threshold failures.
3. **Resource binding:** Kernel modifications that alter block dimensions, register counts, shared memory, or cluster topology constitute a new ABI and must be validated across the full stack.

---

## 3. NVIDIA Exact-Cell Tuner

For exact GEMM and attention shape tuning on NVIDIA architectures:
```bash
python3 scripts/gemma4_h100_kernel_tuner.py plan --spec <spec> --summary <out>
python3 scripts/gemma4_h100_kernel_tuner.py run  --spec <spec> --summary <out>
```
*(Self-leasing: do not wrap in `gpuq.py submit`).*
* Ranks tuning cells by `occurrences × control_us`.
* Compiles candidate profiles with bound cubin hashes, launch geometries, and register budgets.
* Promoted candidates must subsequently pass Tier 3 single-block evaluation.

---

## 4. Decision & Recording

* **Verdict:** Land, Park (with recorded reason), or Kill.
* **Production Landing:** Qualified wins update the canonical recipe in `recipes/infervisor/<model>/<label>.toml`.
* **Knob Registration:** New knobs must be documented in `docs/flags-reference.md` and registered in `crates/*/src/knob_spec.rs`.
* **Ledger Update:** Append candidate measurements to the campaign ledger in scratch. Never commit raw JSON or trial profiles to Git.
