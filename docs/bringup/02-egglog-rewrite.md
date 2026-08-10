# Bringup Stage 2 — Egglog Equality-Saturation Rewriting

> Stage 2 of the model-bringup playbook. Takes the Stage-1 operator graph and
> finds the best equivalent, fusion-optimized graph via egglog equality
> saturation. This stage also owns the Checkpoint-A soundness tie-in: every
> fusion rule the engine can fire must have a matching Lean soundness theorem.

Prerequisite: Stage 1 (frontend) produces an `nn_graph::Graph` for the model.
Next: Stage 3 (Lean verification) — Checkpoint A gates this stage's rule set.

---

## Goal

Given the operator DAG for a new model/architecture, decide whether the
existing fusion rule set already covers it, and if not, add new rewrite rules
**soundly** — meaning:

1. The graph lowers to egglog terms without error.
2. Saturation reaches a fixpoint (the rule set is a directed, non-generative
   set of one-way rewrites, so it always terminates).
3. The intended fusions fire, and extraction picks the fused form.
4. Checkpoint A still passes: every rule the engine can fire is in the Lean
   `soundRules` table with a `rule_*` soundness theorem.

### Scope note — where this runs

The rewrite pass is a **working library that is not on the shipping emit path**.
`plowc --emit devblob` (the path every asset is built with) does not consume a
`FusedGraph`; it synthesizes a `LayerPlan` directly from `config.json` and the
fused kernels in a shipped packet are hand-written in `crates/devgen`. The
egglog pass is exercised by:

- `rewrite::rewrite_graph` — lower → saturate → extract (used by the
  `crates/rewrite` integration tests, and by `plowc`'s non-emit plan path for
  statistics).
- `rewrite::explore_stats` — saturate-only, reports fusion *opportunity* counts
  without extracting (used by the advisory `report_devblob_egglog` fusion
  report that `plowc --emit devblob` prints and discards).

So this stage's concrete, enforced deliverable is: **the rule set is correct and
Checkpoint A passes.** The extracted graph is inspectable and tested, but it
does not itself drive a GPU asset today. Treat "the rule fires and the fusion is
proven sound" as the success bar, not "the fusion reaches a kernel."

See `docs/arch/01-compiler-pipeline.md` (Stage 2) for the full pipeline context
and the deviation-from-implementation warnings.

---

## When a new arch needs new rules vs reuses existing

Most new decoder-style LLMs need **zero** new rules. The existing set already
covers the structural fusions that recur across architectures:

| Fusion site | Rule | Fused target |
|---|---|---|
| norm → projection (QKV / gate / up / lm_head) | `rmsnorm-linear-fuse`, `rmsnorm-linearbias-fuse` | `FusedNormLinear(Bias)` |
| LayerNorm → projection (SigLIP, DiT) | `layernorm-linear-fuse`, `layernorm-linearbias-fuse` | `FusedLayerNormLinear(Bias)` |
| gated MLP (`act(gate) * up`) | `gated-mlp-fuse` | `SwiGLU` |
| residual add → norm (every block boundary) | `residual-rmsnorm-fuse`, `residual-layernorm-fuse` | `FusedResidualNorm` / `FusedResidualLayerNorm` |
| linear → activation (non-gated MLP) | `linear-act-fuse`, `linearbias-act-fuse` | `FusedLinearAct(Bias)` |
| embedding → scale (Gemma normalizer) | `embedding-scale-fuse` | `FusedEmbeddingScale` |
| qk_norm → RoPE (Gemma3/4) | `rmsnorm-rope-fuse`, `rmsnorm-rope-scale-fuse` | `FusedNormRope(Scale)` |
| GroupNorm → act (→ Conv3d) (VAE) | `groupnorm-act-fuse`, `groupnorm-act-conv3d(-bias)-fuse` | `FusedGroupNormAct[Conv3d[Bias]]` |
| AdaLN modulate, gated residual (DiT) | `adaln-modulate-fuse`, `gated-residual-fuse` | `FusedAdaLN`, `FusedGatedResidual` |
| output-gate (Kimi-K3 KDA / MLA) | `kda-gated-norm-fuse`, `mla-out-gate-fuse` | `FusedKdaGatedNorm`, `FusedMlaOutGate` |

**Reuse the existing rules** when the new model expresses its computation with
the ops already in the schema (`crates/rewrite/src/egl/schema.egg` — `RmsNorm`,
`LayerNorm`, `Linear`, `Rope`, `Ew`, `Act`, `GroupNorm`, `Conv2d/3d`,
`Embedding`, `Scale`, …) in the same structural shape. The Stage-1 frontend maps
the arch onto these ops; if it does, existing rules fire automatically.

**Add a new rule** only when the model introduces a *new fusible pattern* — a
composition of ops that recurs and that no existing rule matches. Two triggers:

1. **The pattern uses ops already in the schema, in a new arrangement.** Example:
   `mla-out-gate-fuse` matches `mul(attn, sigmoid(Linear(x)))` — all existing
   ops, a new composition. Just add a rule (and its target op + Lean theorem).
2. **The pattern needs an op not yet in the schema.** Then you must first extend
   `schema.egg` with the base op (and teach `lower.rs` to emit it), then add the
   rule. Kimi-K3's `Conv1dDepthwise`, `SituGlu`, `LinearAttention`,
   `BlockResidual` are examples of base ops added for one arch.

**Do not add a rule** for a one-off, non-recurring pattern, or one you cannot
prove sound in Lean. A rule with no `rule_*` theorem fails Checkpoint A and
blocks the compile under `--lean-verify`.

---

## Step-by-step: adding a fusion rule

A rule is sound-by-construction here because the fused op is *defined* to be its
unfused composition. Adding one is a four-file change kept in lockstep:

### 1. Add the fused target to the schema

`crates/rewrite/src/egl/schema.egg`, under `; === fused targets ===`. The fused
constructor must carry **every operand leaf** the unfused form referenced — no
weight or bias may be dropped, or the weight manifest goes incomplete and the
kernel cannot reproduce the math. Match the operand order to your Lean `expand`
(below). Example (existing):

```egglog
(SwiGLU String Expr Expr)   ; act kind, gate_proj_out, up_proj_out
```

If the pattern needs a base op not present, add that to the `Expr` datatype too,
and teach `crates/rewrite/src/lower.rs` `term_for` to emit it from the
corresponding `nn_graph::Op`. (Refusing to lower — `LowerError::Unsupported` —
is the correct behavior for an op with no term; do not lower to `Opaque`, which
is a single-input passthrough and would drop every operand but the first.)

### 2. Add the rewrite rule with its `; rule:` annotation

`crates/rewrite/src/egl/rules.egg`. **Every `(rewrite …)` MUST be immediately
preceded by a `; rule: <name>` annotation.** This is a hard contract:

- `plowc` parses these annotations (`parse_rule_catalog` in
  `crates/plowc/src/lib.rs`) into the live rule catalog it submits to
  Checkpoint A. Parsing is strict: an annotation with no following `(rewrite …)`,
  or a `(rewrite …)` with no preceding annotation, is a hard `RuleCatalog`
  error. One annotation binds to exactly one rewrite.
- The name must be **kebab-case** and match the Lean theorem name with dashes →
  underscores (`gated-mlp-fuse` ↔ `rule_gated_mlp_fuse`) and must appear
  verbatim in `soundRules`.

```egglog
; rule: gated-mlp-fuse
(rewrite (Ew "mul" (Act ?k ?g) ?u)
         (SwiGLU ?k ?g ?u))
```

The LHS is the pattern to match; the RHS is the fused replacement. Pattern
variables are `?name`. Constant discriminants (activation kind `"silu"`,
elementwise kind `"mul"`) can be pinned as literals to keep distinct e-nodes
distinct — see `kda-gated-norm-fuse` pinning `"sigmoid"` / `"mul"`.

### 3. Cost / extraction behavior (no `:cost` annotations)

There are deliberately **no `:cost` annotations** in `schema.egg` or
`rules.egg`. Extraction runs egglog's tree-additive cost with a uniform head
weight of 1 (`extract::BigTreeAdditiveCost`), so a fused target wins purely
because it has **strictly fewer e-nodes** than the form it replaces. If your
fusion collapses N ops into 1, extraction prefers it with no annotation needed.

Two constraints follow, both load-bearing:

- **A `:cost` you add would be silently ignored.** `egglog::Function::decl` is
  private in egglog 2.0.0, so `BigTreeAdditiveCost` hardcodes head weight 1. To
  express a non-tree-size cost, you must teach the Rust cost model first.
- **Extraction uses arbitrary-precision `BigInt`, not `u64`.** The default
  `u64` tree cost saturates and then aborts the process during reconstruction
  on residual models past ~21 layers; `BigTreeAdditiveCost` reproduces the
  default's decisions exactly wherever the default did not overflow, and does
  not abort at scale. Do not change the cost function casually — a different
  bounded cost (e.g. critical-path depth) re-ranks the fusion space and flips
  extraction decisions.

### 4. Add the Lean soundness theorem + register the rule

`lean-plow/Plow/Rewrite.lean`. This is the Checkpoint-A obligation and is
non-optional for a rule that will ship in `rules.egg`:

1. Add a fused variant to the mini-IR `inductive Op`.
2. Extend `expand : Op → Op` so the fused op unfolds to **exactly** its unfused
   composition (matching the RHS→LHS of your egglog rule).
3. Write a `rule_*` theorem asserting `expand (fused …) = <unfused composition>`;
   the proof is `rfl` because `expand` *defines* the fused op as its unfused
   form.
4. Add the rule's kebab-case name to the `soundRules` list.

```lean
theorem rule_gated_mlp_fuse (g u : Op) (k : String) :
    expand (Op.SwiGLU g u k) =
      Op.Ew "mul" (Op.Act k (expand g)) (expand u) := rfl
```

`soundRules` and the `rule_*` theorems must stay one-to-one. `checkA` rejects
any egglog rule name not in `soundRules`; a rule added to `rules.egg` alone
therefore cannot slip past verification.

> Rule *bodies* are not structurally checked against the egglog RHS — editing a
> proven rule's RHS in `rules.egg` without updating its `expand` case is outside
> Checkpoint A's scope. Keep the two in sync by hand, and add a `fuse_*` test
> that asserts the fusion fires with the correct operands.

---

## Running saturation and inspecting the graph

Use `nix develop` for all build/test commands.

### Rewrite one model end-to-end (lower → saturate → extract)

The integration tests are the canonical driver. `rewrite::rewrite_graph`
returns the `FusedGraph` plus `RewriteStats { ops_before, ops_after, fused }`:

```bash
nix develop --command cargo test -p rewrite --test fuse_all_models
# one model:
nix develop --command cargo test -p rewrite --test fuse_all_models fuse_gemma
```

To exercise your own config, mirror `fuse_gemma` (in
`crates/rewrite/tests/fuse_all_models.rs`):

```rust
let g = nn_graph::models::build_from_config_json(CONFIG_JSON).unwrap();
let (fused, stats) = rewrite::rewrite_graph(&g).unwrap();
assert!(fused.contains("FusedNormLinear"));   // your fused op name
assert!(stats.ops_after < stats.ops_before);  // fusion reduced the graph
```

`FusedGraph::contains(op)`, `fused_count()`, and `op_count()` are the inspection
API. Nodes are `FNode { op, args }`; leaves are `Input` / `Weight` carrying the
tensor name, so you can confirm no weight was dropped by the fusion (see
`fused_weights` vs `graph_weights` in `fuse_all_models.rs`).

### Saturate-only (fusion opportunity counts, no extraction)

`rewrite::explore_stats` runs `(run-schedule (saturate (run)))` then
`(print-size)` and returns `(graph_ops, [(fused_op_name, count)])` — the number
of e-graph matches per fused op, without extracting a term. This is the safe
analysis path (extraction can hit an upstream egglog panic on some large
graphs; saturation + `print-size` never enters that path). `plowc`'s
`report_devblob_egglog` uses it to print the advisory fusion report during
`--emit devblob`.

### Verify Checkpoint A (the soundness gate)

`plowc` parses the live `rules.egg` catalog and submits it to the Lean verifier.
Checkpoint A is exercised by the ignored integration test (needs the built
`plow_verify` binary):

```bash
# build the Lean verifier once:
nix develop --command bash -c "cd lean-plow && lake build"
# run the rewrite checkpoint test (feature-gated, #[ignore]d by default):
nix develop --command cargo test -p plowc --features lean-verify \
  --test lean_verify_rewrite -- --ignored
```

The Lean side (`lean-plow/`) also builds independently; `lake build` failing on
a new/renamed `rule_*` theorem or a `soundRules` mismatch is the fast local
signal.

---

## Success criteria

- **Lowers.** `rewrite::rewrite_graph(&g)` (or `explore_stats`) returns `Ok`
  for the new model's graph — no `LowerError::Unsupported` / `UnmappedTensor`.
- **Saturates.** Saturation terminates (it always does for the one-way rule set)
  and extraction returns a term.
- **Fusions fire.** The intended fused ops appear:
  `fused.contains("FusedXxx")` and `stats.ops_after < stats.ops_before`. Add a
  `fuse_<model>` test asserting the specific fusions and that no weight leaf was
  dropped.
- **Checkpoint A passes.** Every rule name in `rules.egg` is in
  `Plow.Rewrite.soundRules` with a `rule_*` theorem; `lake build` succeeds and
  the `lean_verify_rewrite` test accepts the catalog. **Every fired rule must be
  in `soundRules`** — this is the invariant Stage 3 (Lean) enforces before you
  proceed.
- **Negative behavior preserved.** Patterns that only resemble a fusion target
  do not mis-fire (see `crates/rewrite/tests/negative_fusions.rs`).

---

## Pitfalls

- **Adding a rule to `rules.egg` without the Lean theorem.** Fails Checkpoint A.
  The four-file change (schema, rules, lower if needed, Lean) is atomic.
- **Missing or misplaced `; rule:` annotation.** Every `(rewrite …)` needs
  exactly one preceding annotation; a dangling annotation or an un-annotated
  rewrite is a hard parse error in `plowc`.
- **Name mismatch.** Kebab-case in `rules.egg` and `soundRules`; the Lean
  theorem is `rule_<name_with_underscores>`. A typo passes the egglog parse but
  fails Checkpoint A.
- **Dropping a weight/bias leaf in the fused target.** The manifest goes
  incomplete and the fused kernel cannot reproduce the math. Carry every leaf;
  assert weight-set equality in a test.
- **Adding a `:cost` annotation.** Silently ignored (head weight is hardcoded to
  1). Express cost differences by making the fused form have fewer e-nodes, or
  change the Rust cost model deliberately.
- **Lowering an unsupported op to `Opaque`.** `Opaque` is a single-input
  passthrough — it drops every operand but the first. Return
  `LowerError::Unsupported` instead so the pass fails honestly.
- **Attention attributes are opaque.** `Attention` collapses its whole config
  into one token (`schema.egg`), so no rule can branch on an attention
  attribute (e.g. Gemma-4's `attention_k_eq_v`). That is a structural limit of
  the vocabulary, not a missing rule.
- **Assuming the fused graph reaches a kernel.** It does not on the shipping
  emit path (see Scope note). The deliverable is a correct, proven rule set.
- **Editing a proven rule's RHS.** Checkpoint A does not re-check rule bodies;
  keep `rules.egg` RHS and Lean `expand` in sync manually.

---

## Code pointers (by symbol)

**Rewrite crate — `crates/rewrite/`**

- `src/egl/schema.egg` — `Expr` datatype (op vocabulary) + `; === fused targets ===`.
- `src/egl/rules.egg` — the rewrite rules and their `; rule:` annotations.
- `src/lower.rs` — `lower`, `term_for`, `LowerError` (`nn_graph::Graph` → egglog terms).
- `src/extract.rs` — `run`, `BigTreeAdditiveCost`, `FusedGraph`, `FNode`, `Arg`,
  `is_fused`, `term_to_graph`.
- `src/lib.rs` — `rewrite_graph`, `explore_stats`, `rules_source`, `RewriteStats`,
  `RewriteError`; `SCHEMA`/`RULES` embedded via `include_str!`.
- `src/explore.rs` — `explore::select` (declarative argmin over costs),
  `explore_tiles` (Stage-4 tile selection; not the fusion rules).
- `src/tilegraph.rs` — `assemble`, `assemble_tuned` (Stage-4; the latter takes a
  `KernelOracle`, default `NoOracle`).

**Verification — Checkpoint A**

- `lean-plow/Plow/Rewrite.lean` — mini-IR `Op`, `expand`, `rule_*` theorems,
  `soundRules`, `isSoundRule`.
- `lean-plow/Plow/CLI/Checkpoints.lean` — `checkA`.
- `crates/plowc/src/lib.rs` — `parse_rule_catalog`, `verify_A`.
- `crates/lean_verify/src/checkpoints/rewrite.rs` — `check_rewrite_rules`,
  `RewriteRulesRequest`.
- `docs/arch/08-formal-verification.md` — Checkpoint A section.

**Frontend / context**

- `crates/nn-graph` — `models::build_from_config_json`, `Op`, `Graph`, `Origin`.
- `docs/arch/01-compiler-pipeline.md` — Stage 2 + deviation warnings.
- `docs/arch/02-tile-graph.md` — downstream tile graph.

**Tests**

- `crates/rewrite/tests/fuse_all_models.rs` — per-model fusion assertions.
- `crates/rewrite/tests/negative_fusions.rs` — non-mis-firing checks.
- `crates/plowc/tests/lean_verify_rewrite.rs` — Checkpoint A end-to-end.
