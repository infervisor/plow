# Rung campaign — one rung, one lever, measured to a decision

Improve one compiled rung of a model that already serves (Stage 6 done). A **rung** is one program
× one row class × one prior class × one route, e.g. the 8192-row prefill program at a long prior,
the same program at prior 0, or one decode rung at a fixed context.

The loop has six steps: pick the rung, rung card, attribution, lever card, ladder (T1 → T4),
decide and record. One lever per rung at a time; every lever passes the same ladder.

- **Precondition:** the [`target.md`](target.md) block is filled; commands use its names, never a
  literal part. Harness selection: [05](05-single-block-sweep.md), [06](06-runtime-opt.md).
- **Cross-cutting rules:** [agent-tools.md](agent-tools.md) "HARD RULE" (block first),
  "Queue" (every GPU process), "A/B rule" (noise and repeats), "Per-rung gate and recipe
  selection" (what a recipe may carry), "Measurement hazards".

## 0. Pick the rung

- Priority = `workload share × gap to floor`. Share: time the target workload spends in the rung,
  from a served run's `PLOW_TICK_LOG`. Gap: measured ms minus the roofline floor of its terms
  (compute, bandwidth, fabric, launch/boundary).
- Take the top unowned rung on the campaign's rung board (kept in scratch) and claim it.
- A lever touching several rungs is still judged per rung: one owner rung; every other touched
  rung is a regression check (§4).

## 1. Rung card (before any lever)

| field | how to fill it |
|---|---|
| identity | packet stamp; program index and role (`plowrt disasm`, `build.json` `programs`); bucket T; topology (ordinary / packed / body); sparse vs dense route; prior range; live-row range |
| model path | the emitter that builds it, by `model_type`; confirm with `plowrt disasm` that the ops you mean exist in this program |
| workload | an input that lands on this rung only: `plowrt bench --assets <a> --prefill-sweep --prompt-ids <ids>` (or `--prefill-lengths <len>`) for a prefill chunk; a fixed-context decode step for a decode rung. Serving uses exact-length prompts (`--random-range-ratio 0`) |
| baseline | median with `ctrl` and `ctrl2` in the same job; noise per [agent-tools.md](agent-tools.md) "A/B rule" |
| value floor | cross-process last-row logits and KV-row rel-L2 of ctrl vs ctrl2; this floor, not zero, is the equality bar |
| share | ms/run this rung costs in the target workload (count × median) |

## 2. Attribute

One all-rank trace of this rung only (`PLOW_TRACE_RAW`, `PLOW_TICK_LOG=1`,
`PLOW_PREFILL_SEG_TIMING=1`). Tabulate:

- **Op classes:** native kernels (vendor attention, MoE, indexer, BLAS), collectives (transfer vs
  wait), interpreter GEMMs, each interpreter glue op (tail vs span), native↔interpreter boundaries
  (count × µs).
- **Fixed vs per-row:** `2·t(T/2) − t(T)` at the same prior and program. Fixed cost dominates small
  and ragged rungs.
- **Floor per term:** a term at its floor gets no lever.
- **Delta vs the previous attribution** of this rung.

## 3. Lever card (before touching code)

| field | content |
|---|---|
| hypothesis | one sentence: which term, why it is above its floor |
| predicted ms | from the attribution, with arithmetic |
| scope | every program, op class and instruction field it may change, plus every other rung it touches; off must be byte-identical on all of them |
| numerics class | bit-identical, or floor-bounded (state the floor) |
| rollback | knob and off value; runtime knobs in `RuntimeConfig`, emit knobs in `EmitConfig`, registered in `knob_spec.rs` |
| cheapest falsifier | the T1 or T2 measurement that would kill it |

## 4. The ladder

Promote only on an effect above the tier's own noise ([agent-tools.md](agent-tools.md) "A/B rule").

| tier | what | gate to promote |
|---|---|---|
| T1 CPU | build; tests; knob-off `model.pkt` sha256 identical; untouched objects disassembly-identical (guard new code with a define: dead code shifts sibling codegen); instruction diff equals the declared scope | identical outside scope; diff ⊆ scope |
| T2 one GPU | kernel harness on captured production operands against the exact dispatched route (not a nearby shape, not the library's best), device-event timing, bit or FP-floor check; control-before / candidate / control-after; the lever card's counters | faster than the interpolated control beyond noise; predicted counters move in the declared direction; boundary cost added or removed included |
| T3 one block | the rung on a single block: `--block` or truncated `PLOW_LAYERS=N` packet with its own object set, against the block's calibrated roofline; full model on the rung's workload only where the block harness cannot express the rung (name the gap). ctrl / treat / ctrl2 in one job; values vs the cross-process floor; the lever's firing proven (launch counts, trace) | rung faster beyond noise; values inside the floor; every other touched rung not slower |
| T4 serve | only to flip a default: interleaved ctrl / treat / ctrl2 / treat2 arms, exact-length prompts, the FP32 gate and grid per [agent-tools.md](agent-tools.md) "Gates" and "Campaign playbook" | no failed requests; TPOT not worse beyond control spread; gates pass |

Rules at every tier:

- **Validate the harness on control vs control first.** A compare script that has not reproduced a
  zero difference on two control runs is not evidence.
- **Machine-checked pass/fail:** scripts exit nonzero on failure and write a `PASS` marker that the
  next tier's job checks before it runs.
- **Host-bound or GPU-bound first.** Before a host lever, shrink host time and check the step moves.
- **Re-time the production choice** through the launch path it ships on before comparing against a
  library.
- **One clock per comparison:** every arm timed the same way (device dispatch time through the
  production launch path); a host wall-clock arm and a device-clock arm are not comparable.
- **Bind tile resources to the object:** block size, registers/thread, dynamic shared memory,
  driver-reported blocks/SM, SM count, launch blocks, cluster, grid waves. A change to any is a new
  object/entry ABI; confirm the runtime loads that entry and recomputes occupancy.

### NVIDIA exact-cell tuner

`scripts/gemma4_h100_kernel_tuner.py` (`plan|run --spec <spec> --summary <out>`; *self-leasing*, do
not wrap in `gpuq.py submit`) drives T2 for exact GEMM/attention cells from the packet audit:

1. Rank cells by `occurrences × control_us` from the packet trace; pick the largest in one rung and
   family. Attention cells include head dim, GQA, window, query rung, packed topology and live-KV
   boundary; GEMM/GEMV cells include exact M/N/K, dtype, scale layout, epilogue, split path and
   execution mode. Never average KV buckets.
2. Each candidate declares one hypothesis, one lever, predicted savings (µs) and expected counter
   movement; each compiled profile binds cubin hash, symbol, launch geometry, tile, stages,
   TMA/swizzle, registers, shared memory, stack, spill, blocks/SM and cluster. A new block size,
   register budget, shared-memory claim, cluster or entry point is a separate object.
3. The tool verifies seeds before timing and gates on stack/spill, occupancy fit, control-anchor
   stability, same-trial noise and counter direction.
4. Rank by `rung_rollup` realized occurrence-weighted savings against
   `weighted_predicted_savings_us` and `weighted_noise_floor_us`; a fast rare shape does not outrank
   a smaller saving repeated across the block.
5. A `T2-kernel-qualified-candidate` still owes `T3-packet-role-block`: route it through its
   hash-bound packet role and repeat the block A/B, reconciling planned vs observed launches,
   object transitions, grid waves and activation bytes.
6. Publish a tune-store row only for the exact packet/object identity that passed T3, and prove
   `plowc tune select` consumes it. Full logits and serving run only after every promoted cell in the
   rung is closed.

## 5. Decide and record

- **Decision:** land, park (measured reason) or kill. Landing in a production recipe follows
  [agent-tools.md](agent-tools.md) "Per-rung gate and recipe selection"; the candidate is judged
  against the same-concurrency vLLM peer. Parked or killed lever code does not land.
- **Ledger:** one entry per arm (rung digest, recipe digest, metric, samples, control ids) in the
  campaign ledger in scratch (`$RESULTS`), never in the repo.
- **Flags:** one row in [`docs/flags-reference.md`](../flags-reference.md) for any new knob.
- Update the rung card and the board. No per-experiment markdown or raw JSON in git.

## Pitfalls

| pitfall | guard |
|---|---|
| lever emitted into a different model's emitter | rung card "model path"; T1 knob-on changes nothing on another model's recipe |
| scope creep (a workgroup cap also narrowed a collective) | lever card scope + T1 instruction diff by op class |
| knob off not identical at the object level | disassembly compare of untouched objects |
| rung-mixing workload | exact-length prompts; check input-length range per cell |
| isolated kernel win vanishes in flow | T3 on the rung is the promotion gate, never T2 |
| front-of-chunk layouts assuming a scalar `kv_len` | per-row tensors indexed from the band's row 0; chunk rows rebased per launch |
| fixed-width native routes with a minimum causal key count | prior-class the rung card; split rows, do not widen the gate |
| stored or cross-session baseline | ctrl and ctrl2 in the same job |

## Report

- The rung card and lever card, filled.
- Attribution table, fixed vs per-row, floors.
- Per tier: command, job label, numbers (median, floor, noise), pass or fail.
- Decision and why; ledger entries and flag row.
- Next lever on this rung, ranked by predicted ms against the remaining gap to floor.
