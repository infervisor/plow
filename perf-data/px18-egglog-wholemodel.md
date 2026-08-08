# PX-18 — the egglog rewrite stage, end to end: coverage, and why it is zero

RTX 5090 (sm_120a, 170 SM, 32 GiB) · Gemma-4-12B-it · companion to
`px12-consolidated-baseline.md` (the 127k cell: vLLM **42.49** out tok/s vs plow **29.91**,
81% of plow's wall is prefill).

This note is on the **compiler/graph axis**: egglog rewrites and Lean-verified packets,
deliberately different from the kernel/tile/occupancy work in the rest of the campaign.

**No GPU time was used and no serving-path code was changed.** That is the finding, not an
omission — see §1.

---

## Question

1. What fraction of the Gemma-4-12B graph does egglog actually rewrite today, in **ops and
   FLOPs**, and what is hand-emitted?
2. Pushed whole-model, what additional fusions does egglog find that the hand-emitted path
   does not take?
3. Which of those need new opcodes?
4. Counter granularity, coarse vs fine, A/B'd end to end.

Questions 2–4 are answered by question 1, and the answer is that they do not arise. Q1 is
the deliverable.

---

## 1. Coverage: egglog rewrites **0%** of the compiled graph, on every emit path

Measured with `crates/plowc/examples/egglog_coverage.rs` (added by this note) against the real
`/root/gemma-4-12B-it/config.json`, bound at B=1, S=1024.

### 1a. The numbers

**Text-only graph (`text_config`, 48 layers — the model that is served):**

| | ops | GFLOP |
|---|---|---|
| Gemma-4-12B text graph | 1156 | 24,225.6 |
| Op kinds any rule LHS *could* match — the ceiling | 916 (79.2%) | 23,744.0 (**98.0%**) |
| Fusion matches egglog *finds* in saturation | 530 | — |
| **Ops reaching any emitter via egglog** | **0 (0.0%)** | **0 (0.0%)** |

**Unified graph (text + vision + audio towers), for completeness:**

| | ops | GFLOP |
|---|---|---|
| graph | 1595 | 38,774.9 |
| rule-reachable ceiling | 1216 (76.2%) | 37,578.7 (96.9%) |
| **reaching any emitter** | **0 (0.0%)** | **0 (0.0%)** |

Input-graph FLOP distribution, text-only: `linear` 289 ops / 98.0%, `attention` 48 ops / 2.0%,
everything else (rmsnorm 289, elementwise 144, reshape 192, rope 96, scale 49, act 48,
embedding 1) rounds to 0.0% combined. **This model is a GEMM, twice.**

### 1b. Four independent reasons it is exactly zero

1. **`plan_from_all_blocks` never sees a fused graph.** `crates/rewrite/src/bridge.rs:94`
   iterates `g.block_nodes(block)` on the **raw** `nn_graph::Graph`.
   `crates/plowc/src/lib.rs:2270` computes `rewrite_graph(&g)`, keeps only `stats`, and drops
   the term. The comment above it says so outright:
   *"The fused graph is not consumed (the plan below is built from the source graph)."*
2. **The one function that consumes a `FusedGraph` has no production caller.**
   `plan_from_fused` (`bridge.rs:409`) is referenced only from
   `crates/rewrite/tests/qwen_block_to_tiles.rs:261`.
3. **`devgen` — the only emitter that produces a GPU-runnable PLOWDEV blob — has no `rewrite`
   dependency at all.** `crates/devgen/Cargo.toml` lists costmodel, packet, kernelcaps, hwspec,
   plow-asset, serde_json. Verified structurally: `cargo tree -p devgen -e normal` contains
   neither `rewrite` nor `egglog`. `devgen::run_verified` receives
   `EmitArgs{dir, ctx, out, n_cu, tp, block_spec, …}` — no graph, no plan, no fused term crosses
   that boundary.
4. **Nothing egglog produces can reach a GPU even in principle.** `plowrt`'s GPU executor loads
   only `DevBlob` (`crates/plowrt/src/exec/gpu.rs:579`). The `.pkt` bucket streams the
   egglog-adjacent path emits are CPU-reference/simulator artifacts.

The single live use is `report_devblob_egglog` (`crates/plowc/src/main.rs:390`): a
`tracing::info!` line under `--lean-verify`, saturate-only, result logged and dropped.

### 1c. egglog's rule set is a strict **subset** of devgen's hand-emitted fusions

Every rule that fires on this model, against what the emitted stream already does:

| egglog rule (saturation matches) | devgen today | verdict |
|---|---|---|
| `rmsnorm-rope-fuse` + `-scale` (144) | `HeadNormRope` / `HeadNormRopeFp8` — norm + RoPE **+ KV-cache store** | devgen **superset** |
| `gated-mlp-fuse` → `SwiGLU` (48) | `GemmGluFp8` / `GemvGluFp8` — gate‖up GEMM **+** act epilogue | devgen **superset** |
| `residual-rmsnorm-fuse` (96) | `NormResidualNorm` — residual + **two** norms, crossing the layer boundary | superset in **decode**; **absent in prefill** — the only gap, see §4 |
| `embedding-scale-fuse` (1) | `Embed` with `emb_scale` baked in | devgen **superset** |
| `linear-act-fuse` (48) | subsumed by `GemmGlu` | devgen **superset** |
| `rmsnorm-linear-fuse` (**193** — the largest) | `DevOp::GemmNorm` exists but is **never emitted**; `Gemv` fold mode 2 is dead with `TENSOR_NONE` gammas | **measured 22.4 → 24.4 ms/token, a 9% regression** (`crates/devgen/src/lib.rs:1417`) |

So egglog's single biggest opportunity is one the campaign already implemented, measured, and
deliberately reverted. **There is no new opcode to add (Q3), because there is no fusion in the
rule set that devgen does not already perform or has not already rejected on measurement.**

### 1d. A structural limit, not a missing rule

`schema.egg` models attention as `(Attention Expr Expr Expr String)` with the entire attention
config collapsed into one opaque token. No rule can branch on an attention attribute. Most
concretely, Gemma-4's **`attention_k_eq_v`** (K and V share one projection) is **invisible to
this rule set even in principle** — while devgen already elides the v_proj weight, its GEMM and
its CUs. The rules are likewise blind to everything else that decides this blob: fp8 quant
placement (4 `QuantFp8`/layer, deliberately shared to avoid a race), KV ring stores, flash
split/merge, tile selection.

---

## 2. The extractor could not run on the model we serve — root-caused and fixed

`rewrite_graph` **aborted the process** on the 48-layer Gemma-4 text graph:
`egglog-2.0.0/src/extract.rs:471`, `.unwrap()` on a `None` `parent_edge`. With
`panic = "abort"` in the release profile that is process death, which is precisely why
`explore_stats` exists as a saturate-only path and why the devblob emitter never calls
`rewrite_graph`.

### 2a. Cause

`TreeAdditiveCostModel` sums its children, and `Cost` for every integer type combines with
`saturating_add` (`extract.rs:70`). **Tree cost is not DAG cost.** A residual stream references
its hidden state ~8× per layer (q/k/v read the normed hidden; the sandwich norm and the residual
add each read the stream), so the *tree* unfolding of layer `L` costs ~8^L while the DAG is
linear in `L`. That crosses `u64::MAX` around **layer 21**.

Past that, every e-class on the residual chain is pinned at `u64::MAX`, so
`new_cost < *e.get()` is never true again and Bellman-Ford's `topo_rnk` stops advancing in step
with the costs. `save_best_parent_edge` requires
`target_topo_rnk > compute_topo_rnk_hyperedge(row)` to record a parent edge; that test then
fails for *every* e-node feeding some e-class, and reconstruction unwraps `None`.

It is a **scale** bug, not a graph-shape bug. Bisected against the real config:

| layers | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | **24** | **48** |
|---|---|---|---|---|---|---|---|---|---|---|
| before | ok | ok | ok | ok | ok | ok | ok | ok | **abort** | **abort** |
| after | ok | ok | ok | ok | ok | ok | ok | ok | **ok** | **ok** |

### 2b. Fix

`extract::BigTreeAdditiveCost` — egglog's tree-additive cost in arbitrary precision
(`num::BigInt`). Deliberately surgical: **the only change is that the accumulator cannot
overflow.**

A cheaper bounded cost (critical-path depth, `max` instead of `+`) also cannot overflow, and was
tried first — but it *re-ranks the fusion space*. Measured, it flips the `pre_feedforward_norm`
site from `FusedResidualNorm` to two `FusedNormLinear`s, because depth cost cannot see that the
latter recomputes the norm once per consumer; it broke six `fuse_all_models` expectations. That
is a design change to the rewrite's objective, not a bug fix. `BigInt` reproduces the default's
extraction decisions *exactly* wherever the default did not overflow.

Head weight is hardcoded to 1, which is exact rather than approximate: `schema.egg` declares no
`:cost` on any constructor (`grep -c :cost` is 0 across all three `egl/*.egg` files), so
`TreeAdditiveCostModel::enode_cost`'s `func.decl.cost.unwrap_or(1)` is uniformly 1.
`Function::decl` is private in egglog 2.0.0.

Cost at 48 layers: ~143 bits, 3 limbs, over a ~4k-node e-graph. Immaterial.

### 2c. What the fix reveals

Post-fix extraction on the 48-layer text graph: `ops_before` 1156 → `ops_after` 913, 243 fused
nodes — `FusedResidualNorm` 96/96, `FusedNormRope` 48, `FusedNormRopeScale` 48, `SwiGLU` 48,
`FusedNormLinear` 2, `FusedEmbeddingScale` 1. Unified graph: 1595 → 1324, 325 fused.

**A correction to this note's own earlier draft.** Before the fix I measured
`FusedResidualNorm` matching 96× in saturation and extracted **0** times, and
`FusedNormRopeScale` 48 → 0, and attributed it to the missing cost annotations (§3). That
attribution was **wrong**: both were symptoms of the same overflow, and both extract at full
multiplicity once it is fixed, with no annotations added. The documentation defects in §3 are
real as documentation defects; they were not causing the lost fusions.

---

## 3. Two documentation defects, recorded and corrected

Both actively mislead a reader into believing extraction is tuned when it is not:

1. `egl/rules.egg` carried a section header *"Cost annotations: fused nodes are always preferred
   over unfused forms — (Lower cost = preferred at extraction.)"* with **no annotations under
   it**, and none anywhere in the file.
2. `egl/schema.egg:6` asserted *"explicit cost annotations reinforce this"* when there are none.

Corrected in place to state what is true, plus the trap that a future `:cost` would be
**silently ignored** because the Rust cost model hardcodes the head weight (`Function::decl` is
private). §1d's attention limitation is also now recorded in the schema.

---

## 4. The one real gap, sized and deliberately unfunded

`residual-rmsnorm-fuse` fires 96× and is the **only** rule that finds something devgen leaves on
the table: **decode** uses the fused `NormResidualNorm` (`devgen/src/lib.rs:2220`) while
**prefill** emits the split `NormResidual` + `RmsNorm` pair (`:2234`, `:2254`), gated on
`gfuse`, which is decode-only.

Everything needed is already in the tree:

* opcode `DevOp::NormResidualNorm` exists;
* `d_norm_residual_norm` (`runtime/nvidia/op_norm.cuh:277`) is **row-parallel**, so `t>1` works;
* it is already compiled into the prefill object (`interp_sm120.cu:592` sits under
  `#if PLOW_NV_GEMMA`, outside the `!PLOW_NV_PREFILL` guard, and `PLOW_NV_PREFILL` implies
  `PLOW_NV_GEMMA`);
* it is documented **bit-exact** to the pair it replaces — it rounds `resid` to bf16 before the
  second reduction, reproducing the HBM round-trip without the traffic — so greedy parity would
  hold *by construction*.

**Pre-registered size, computed before any implementation:** it removes one full pass over the
residual stream per site, 2 sites × 48 layers. At T=1024, H=3840, bf16 that is ~755 MB/chunk
≈ 0.5 ms; over a 127k prefill at chunk 1024 (124 chunks), ~62 ms of a 27,590 ms prefill =
**~0.2%**, plus ~10% fewer packets per prefill program (910 → 814).

**0.2% is below the cell's 3% reproducibility band** (PX-12 §3: control reproduced §2b to 3%).
It is therefore **not measurable on the cell**, and building it would produce a number nobody
could trust in either direction. **Deliberately not implemented.** Recorded here so the next
agent does not re-derive it.

---

## 5. Results

1. **egglog rewrites 0% of the compiled graph — 0 ops, 0 FLOPs — on every emit path.** Four
   independent reasons (§1b), each traced to a line. The ceiling, had it been wired, would have
   been 916/1156 ops (79.2%) and 98.0% of FLOPs.
2. **Its rule set is a strict subset of devgen's hand-emitted fusions** (§1c). Its largest single
   rule (`rmsnorm-linear-fuse`, 193 matches) is a **measured 9% decode regression** that the
   campaign already reverted.
3. **No new opcodes are warranted** (Q3). Nothing in the rule set is unimplemented on the
   serving path except §4, which is sized below noise.
4. **The extractor aborted on the model we serve**; root-caused to `u64` saturation of a
   tree-additive cost on a residual stream past ~21 layers, and fixed with an arbitrary-precision
   cost model that preserves the default's decisions exactly (§2).
5. **Q4 (counter granularity) was not run**, and the reason is Q1: `Builder::select_granularity`
   operates on devgen's emitted op stream, which egglog does not influence. A coarse-vs-fine A/B
   would have been a measurement of devgen, not of the rewrite axis, and duplicates work already
   owned by other branches.

---

## 6. Gates

| gate | result |
|---|---|
| Q1 coverage measured on the real checkpoint, ops **and** FLOPs | **PASSED** — §1a, `crates/plowc/examples/egglog_coverage.rs`, both text-only and unified |
| zero-coverage claim traced to code, not inferred | **PASSED** — four independent reasons, §1b, each with file:line; `cargo tree -p devgen` contains neither `rewrite` nor `egglog` |
| extractor runs on the 48-layer served model | **PASSED** — §2, `gemma4_48_layer_unroll_extracts_without_saturating_cost` |
| that test actually catches the bug | **PASSED** — verified failing on the pre-fix tree (`git stash`): same `extract.rs:471` panic; passes after |
| no regression in the rewrite suite | **PASSED** — 84 tests, 0 failures across all 9 targets. The rejected depth-cost variant broke 6; `BigInt` breaks none |
| no regression in dependent crates | **PASSED** — `plowc` + `schedule` suites green |
| **byte-identical blob** | **PASSED, structurally — stronger than a hash.** `devgen` has no dependency path to `rewrite`/`egglog` (§1b.3), so no change in this note can reach the emitter. The one call from the devblob path (`report_devblob_egglog`) is `--lean-verify`-only and its result is dropped |
| Lean rule catalog unchanged | **PASSED** — 17 `; rule:` annotations, 17 `(rewrite` forms, all 17 present in `lean-plow/Plow/Rewrite.lean:227 soundRules`. **No rewrite rule was added or modified**, so checkpoint A's input is byte-identical |
| Lean re-verification of the extractor fix | **NOT APPLICABLE** — checkpoint A certifies *rewrite rules*; the fix is to the extraction cost function, which selects among already-sound rules and cannot introduce an unsound one |
| greedy-token parity at fixed chunk | **NOT RUN — nothing on the serving path changed.** Would have been vacuous: the blob is byte-identical |
| end-to-end on the 127k cell vs 29.91 | **NOT RUN, deliberately.** No serving-path change to measure. Spending a cell to re-measure a byte-identical blob would burn shared GPU time to reproduce noise |
| coarse-vs-fine counter granularity A/B | **NOT RUN** — see §5.5; Q1 makes it a measurement of devgen, not of this axis |
| prefill `NormResidualNorm` fusion | **NOT IMPLEMENTED, deliberately** — §4, pre-registered at ~0.2%, below the cell's 3% band, therefore unmeasurable |
| the rejected depth-cost model | **MEASURED AND REJECTED** — non-overflowing but re-ranks the fusion space (§2b) |
| coverage at other shape buckets | **NOT RUN** — the ratio is structural (op counts and FLOP mix are shape-independent in the relevant sense); one bucket was sufficient to establish 0% |

---

## 7. Recommendation

The rewrite/egglog subsystem is **wired into `plowc` in name only**. Options, in order of
honesty:

1. **Leave it as an analysis tool and say so.** The doc-comment pipeline diagram in
   `crates/plowc/src/lib.rs:10` (`build graph ─▶ rewrite (fuse) ─▶ plan_from_all_blocks`) is
   **false** and should be corrected — `rewrite (fuse)` is not in that path. This is a one-line
   change and the highest value-per-byte item left in this note.
2. If the subsystem is to earn its place, the gap is not rules — it is that
   `FusedGraph → packet::devbuild::Builder` **does not exist**. Building it means reimplementing
   devgen's ~1265-line per-layer emitter (fp8 quant placement, KV ring stores, flash split/merge,
   RoPE tables, counter deps) on top of a rule set that is already a subset of what that emitter
   does. That is a large project whose *best case* is parity.
3. Do not add opcodes for these rules. §1c is the reason.

**None of this is where the 29.91 → 42.49 gap lives.** 81% of plow's wall is prefill and 98% of
prefill's FLOPs are `linear`; the leverage is in the GEMM and attention kernels, which is what
other branches are working on.
