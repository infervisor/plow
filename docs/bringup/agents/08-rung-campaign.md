# Agent — Rung Campaign: one rung, one lever, measured to a decision

## Target parameters — fill this in FIRST

Fill the block from [`../target.md`](../target.md) (`$VENDOR $ISA $GPU $NCU $NGPU $PARALLEL
$MAXCTX $TOOLCHAIN $BUILD $FEATURES $BW_BOUND $COMPUTE_CEIL $RESULTS`). A row you cannot fill is a
blocker. Commands use these names; never substitute a literal part name.

You are improving **one compiled rung** of a model that already serves (Stage 6 done). A
**rung** is one program × one row class × one prior class × one route. Examples:

- the 8192-row sparse prefill program at prior ≥ 2047 (full chunk, or a ragged 4K tail);
- the same program at prior 0, a request's first chunk;
- decode rung 20 at 65K context.

The loop has seven steps: rung card, attribution, lever card, ladder (T1 → T4), decision, record.
Every lever passes through the same ladder, and only one lever is under test per rung at a time.
Isolated experiments with no rung card, no floor and no decision rule are the failure mode this
prompt exists to prevent.

Read first:
- [`05-single-block-sweep.md`](05-single-block-sweep.md) and
  [`06-runtime-opt.md`](06-runtime-opt.md) for harness selection;
- the live board for this model (`plans/rung-board.md`, gitignored);
- the knob verification plan (`plans/lean-knob-verification.md`), which supplies scope and ledger
  semantics.

## 0. Pick the rung

- **Priority:** `workload share × gap to floor`.
  - Workload share: time the target workload spends in this rung, from a served run's `PLOW_TICK_LOG`.
  - Gap to floor: measured ms minus the roofline floor for its terms (compute ceiling, bandwidth,
    fabric, launch/boundary floor).
- **Work:** take the top unowned rung on the board and put your name on it. Never work two rungs
  in one lever.
- **When a lever touches several rungs:** it is still judged per rung. Choose one owner rung; every
  other touched rung is a regression check (§4).

## 1. Rung card — fill before any lever

| field | how to fill it |
|---|---|
| identity | packet stamp; program index and role (`plowrt disasm`, `build.json` `programs`); bucket T; topology (ordinary / packed / body); sparse vs dense route; prior range; live-row range |
| model path | the emitter that builds it, by `model_type` (e.g. `glm_moe_dsa` → `mla::glm_emit_full`, not `glm53_emit_full`). Confirm with `plowrt disasm` that the ops you mean exist in this program. |
| workload | the exact prompt/sweep that lands on this rung and only this rung: `amd-bench --prefill-sweep <len>` for a prefill chunk, a fixed-context decode step for a decode rung. For serving, use `--random-range-ratio 0`, since ±20% spreads prompts over neighbouring rungs. |
| baseline | median ms with `ctrl` and `ctrl2` in the **same job**. The noise floor is \|ctrl − ctrl2\| + 2·MAD. |
| value floor | cross-process last-row logits and KV-row rel-L2 of ctrl vs ctrl2. Prefill logits are not reproducible across processes (tracker #50), so this floor, not zero, is the equality bar. |
| share | ms/run this rung costs in the target workload (count × median). |

## 2. Attribute the rung

One all-rank trace of this rung only (`PLOW_TRACE_RAW=1`, `PLOW_TICK_LOG=1`,
`PLOW_PREFILL_SEG_TIMING=1`). Tabulate:

- **Op classes:** native kernels (vendor attention, MoE, indexer, BLAS), collectives (transfer vs
  wait on peers), interpreter GEMMs, each interpreter glue op separately (tail vs span), and
  native↔interpreter boundaries (count × µs).
- **Fixed vs per-row:** `2·t(T/2) − t(T)` at the same prior, same program. Fixed cost dominates
  small and ragged rungs.
- **Floors per term:** a term already at its floor gets no lever.
- **Delta vs the rung's previous attribution:** what shrank, what didn't.

## 3. Lever card — fill before touching code

| field | content |
|---|---|
| hypothesis | one sentence: which term, why it is above its floor |
| predicted ms | from the attribution, with arithmetic; a range is fine, no number is not |
| scope | every program, op class and instruction field the lever may change, plus every other rung it touches. Off must be byte-identical on all of them. |
| numerics class | bit-identical, or floor-bounded (state the floor you will hold it to) |
| rollback | knob name and off value; runtime knobs go in `RuntimeConfig`, emit knobs in `EmitConfig` |
| cheapest falsifier | the T1 or T2 measurement that would kill it |

## 4. The ladder — promote only on an effect above the tier's own noise

| tier | what | gate to promote |
|---|---|---|
| T1 CPU | build; tests; knob-off `model.pkt` sha256 identical; untouched objects disassembly-identical (dead code can shift sibling codegen, so guard new code with a define); instruction diff equals the declared scope and nothing else | all identical outside scope; diff ⊆ scope |
| T2 one GPU | kernel harness on captured production operands against **the exact dispatched route** (not a nearby shape, not the library's best, not a re-picked kernel), same inputs, device-event timing, bit or FP floor check. Run control-before/candidate/control-after anchors and emit the lever card's counters. | faster than the interpolated control beyond `abs(ctrl-before - ctrl-after) + 2*MAD`; predicted counters move in the declared direction in at least 3/4 trials; include the boundary cost the change adds or removes (6–52 µs per native↔interpreter transition) |
| T3 eight GPUs | the rung itself: truncated `--layers N` packet (N just past the changed ops, with its own object set) or the full model on the rung's workload. ctrl / treat / ctrl2 in one job; value check vs the cross-process floor; the lever's firing proven (launch counts, trace) | rung median faster beyond the noise floor; values inside the floor; every other touched rung not slower |
| T4 serve | only to flip a default: interleaved ctrl / treat / ctrl2 / treat2 arms, exact-length prompts, ≥ 32 prompts at C1 and ≥ 160 at C16, `--save-detailed`, bootstrap 95% CI of the per-request TTFT/TPOT difference; retrieval base + tail suite; serving guard | CI upper bound < 0 in every cell; no failed requests; TPOT not worse beyond the control spread; retrieval all pass |

Rules at every tier:

- **Validate the harness on control vs control first.** A compare or summary script that has not
  reproduced a zero difference on two control runs is not evidence.
  - A filename-field bug once failed a correct gate.
  - A bench summary that read one of two clients once showed a regression that didn't exist.
- **Pass/fail is machine-checked:** scripts exit nonzero on failure and write a `PASS` marker that
  the next tier checks.
- **Host-bound or GPU-bound first.** Before a host lever, confirm the step is host-bound: shrink
  host time and check the step moves. Parallel enqueue and kernarg caching measured null because
  decode was GPU-bound.
- **Re-time the production choice.** Before comparing against a library, re-time the kernel
  production actually dispatches, through the same launch path it ships on.
- **One timing method per comparison.** Every arm is timed on the same clock (device dispatch time
  through the production launch path). A host wall-clock sweep around blocking drains and a
  device-clock arm are not comparable: a re-pick that looked −7.7 ms/step at T2 on mixed clocks was
  +0.5 ms/step at T3, and a plow-tile winner table built the same way was wrong on nearly every shape.
- **Bind tile resources to the object.** Record block size, registers/thread,
  dynamic shared memory, driver-reported blocks/SM, SM count, launch blocks,
  cluster, and resulting grid waves. A tile change that alters any of these is
  a new object/entry ABI. Confirm the runtime loads that entry and recomputes
  occupancy; packet workgroups alone do not describe residency.
- **For Gemma-4 H100 exact cells, use the machine gate.** Run
  `scripts/gemma4_h100_kernel_tuner.py`; a result named
  `T2-kernel-qualified-candidate` still owes the `T3-packet-role-block` gate.
  Rank its `rung_rollup` by realized occurrence-weighted savings, and compare
  that with `weighted_predicted_savings_us` and
  `weighted_noise_floor_us` before choosing the next lever.

## 5. Decide, then record

- **Decision:** one of land-opt-in, flip-default, park (with the measured reason), or kill. A
  T3-positive lever lands opt-in (off by default, byte-identical). A default flips only on T4.
- **Ledger:** append one entry per arm in the §5.1 schema of `plans/lean-knob-verification.md`
  (rung digest, recipe digest, metric, samples, control ids) to the external ledger.
- **Tracker:** one row in `docs/bringup/tp-bringup-upstream-review-log.md` with numbers and job
  labels. One row in `docs/flags-reference.md` for any knob. Update the rung card and the board.
- **No per-experiment markdown or raw JSON in git.**

## 6. Queue and disk hygiene

- **Queue:** every GPU process goes through the shared queue. Quick jobs are T2, T3 and anything
  under 5 minutes. Preflight paths, binaries and pairing on CPU before submitting.
- **Spool entries:** never move or edit one that may be about to start.
- **Storage:** build outputs, object sets, packets and dumps live on campaign storage (`$RESULTS`),
  not the root filesystem. Use one cargo target dir per worktree, and delete superseded exports.
- **Held jobs:** a job gated on an earlier tier checks that tier's `PASS` marker itself, so a
  failed gate never burns queue time.

## Pitfalls these campaigns actually hit

| pitfall | guard |
|---|---|
| lever emitted into a different model's emitter | rung card "model path" + T1 `empty_effect`: knob on changes nothing on the target recipe |
| scope creep (a workgroup cap also narrowed a collective) | lever card scope + T1 instruction diff by op class |
| knob off not identical at the object level | disassembly compare of untouched objects |
| rung-mixing workload (±20% prompt lengths) | exact-length prompts; check input-length range per cell |
| isolated kernel win vanishes in flow | T3 on the rung is the promotion gate, never T2 |
| front-of-chunk layouts that assume a scalar `kv_len` | per-row tensors indexed from the band's own row 0; chunk rows rebased per launch |
| fixed-width native routes need 2048 causal keys | prior-class the rung card; split rows, don't widen the gate |
| comparing against a stored or cross-session baseline | ctrl and ctrl2 in the same job, always |

## Report back

- **Cards:** the rung card and the lever card, filled.
- **Attribution:** the table, fixed vs per-row, and floors.
- **Per tier:** command, job label, the numbers (median, floor, CI), and pass or fail.
- **Decision** and why.
- **Records:** tracker row, flag row, ledger entries, board update.
- **Next lever on this rung,** ranked by predicted ms against the rung's remaining gap to floor.
