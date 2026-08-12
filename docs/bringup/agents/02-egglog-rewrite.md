# Agent — Bringup Stage 2: Egglog Equality-Saturation Rewriting

## Target parameters — none

This stage is **target-independent**: it rewrites and proves, it does not
measure. Do not name a GPU, ISA, CU count or toolchain anywhere in it — a rule
whose soundness depends on a part is not sound. The parameter block in
[`../target.md`](../target.md) is filled in at Stage 4, not here.

You are executing **Stage 2** of the plow model-bringup playbook for a new
model/architecture: egglog equality-saturation rewriting (operator graph →
fused/optimized graph), plus its Checkpoint-A soundness tie-in.

Your job: get the new model's operator graph to lower and saturate through the
fusion rules, decide whether it needs new rules, add any new rule **soundly**
(with its Lean Checkpoint-A theorem), and leave Checkpoint A passing before
handing off to Stage 3 (Lean verification).

Read `docs/bringup/02-egglog-rewrite.md` first — it is the reference for this
stage. This prompt is the executable procedure.

## Ground rules

- Use `nix develop --command <cmd>` for every build/test.
- Keep changes minimal. Add a rule only when a real, recurring fusible pattern
  has no existing match. Most new decoder LLMs need zero new rules.
- Never add a rewrite rule without its Lean soundness theorem in the same
  change. A rule in `rules.egg` with no `soundRules` entry fails Checkpoint A.
- Do not add `:cost` annotations (silently ignored — head weight is hardcoded
  to 1). Fusion wins by having fewer e-nodes.
- Do not commit. Report results.

## Context you must internalize

- The rewrite pass is a working, tested library **but not on the shipping emit
  path**: `plowc --emit devblob` synthesizes its plan from `config.json` and
  ships hand-written fused kernels from `crates/devgen`; no `FusedGraph` reaches
  a GPU today. So your success bar is **"the rule fires and is proven sound,"**
  not "the fusion reaches a kernel."
- Sound-by-construction: a fused op is *defined* in Lean (`expand`) to equal its
  unfused composition; each rule's theorem is `rfl`.

## Step 1 — Read

- `crates/rewrite/src/egl/schema.egg` — the `Expr` op vocabulary + fused targets.
- `crates/rewrite/src/egl/rules.egg` — existing rules + `; rule:` annotations.
- `crates/rewrite/src/lower.rs` — how `nn_graph::Op` becomes an egglog term.
- `crates/rewrite/tests/fuse_all_models.rs` — how a model is fused + asserted.
- `lean-plow/Plow/Rewrite.lean` — mini-IR `Op`, `expand`, `rule_*`, `soundRules`.
- The new model's Stage-1 output (its `nn_graph` builder / config).

## Step 2 — Lower & saturate the new model, see what fires

Write or adapt a `fuse_<model>` test (mirror `fuse_gemma`):

```rust
let g = nn_graph::models::build_from_config_json(CONFIG_JSON).unwrap();
let (fused, stats) = rewrite::rewrite_graph(&g).unwrap();
```

Run it:

```bash
nix develop --command cargo test -p rewrite --test fuse_all_models fuse_<model>
```

Inspect: `fused.contains("FusedXxx")`, `stats.{ops_before,ops_after,fused}`.
For opportunity counts without extraction, use `rewrite::explore_stats(&g)`.

Outcomes:
- **Lowers + intended fusions fire + `ops_after < ops_before`** → no new rule
  needed. Add assertions, go to Step 5.
- **`LowerError::Unsupported` / `UnmappedTensor`** → Stage 1 emitted an op with
  no egglog term. If it should fuse, you must extend `schema.egg` + `lower.rs`
  (Step 3a). If it is a leaf that should just pass through, that is a Stage-1
  issue — flag it, do not fake it with `Opaque` (which drops operands).
- **Lowers but an expected fusion does not fire** → a new rule is needed
  (Step 3).

## Step 3 — Decide if a new rule is warranted

Add a rule only if the unmatched pattern is a **recurring** composition
(appears every layer / block, not a one-off) AND you can prove it sound.

- Pattern uses existing schema ops in a new arrangement → add a rule + fused
  target + Lean theorem (Step 3b, 4).
- Pattern needs an op not in the schema → also extend `schema.egg` and
  `lower.rs` (Step 3a) first.

If the pattern is a one-off, or depends on an attribute the vocabulary hides
(e.g. attention config is one opaque token — no rule can branch on it), **stop
and ask** rather than adding an unsound or over-fitted rule.

### Step 3a — (only if needed) new base op

- Add the constructor to the `Expr` datatype in `schema.egg`, carrying every
  operand (no leaf dropped).
- Teach `crates/rewrite/src/lower.rs` `term_for` to emit it from the
  `nn_graph::Op`. Refuse (`LowerError::Unsupported`) rather than lower to
  `Opaque` if it genuinely has no term.

### Step 3b — the fused target + rule

- Add the fused constructor to `schema.egg` under `; === fused targets ===`,
  carrying **every** operand leaf of the unfused form.
- Add the rewrite to `rules.egg` with a `; rule: <kebab-name>` annotation
  immediately above it (one annotation ↔ one `(rewrite …)`; strict parse):

```egglog
; rule: <kebab-name>
(rewrite (<LHS pattern with ?vars>)
         (<FusedTarget ?vars>))
```

## Step 4 — Prove it (Checkpoint-A obligation)

In `lean-plow/Plow/Rewrite.lean`:

1. Add a fused variant to `inductive Op`.
2. Extend `expand` so the fused op unfolds to exactly its unfused composition
   (match your egglog RHS→LHS operand order).
3. Add `theorem rule_<name_with_underscores> … := rfl`.
4. Add `"<kebab-name>"` to `soundRules`.

Keep `soundRules` and `rule_*` one-to-one, and keep the Lean `expand` body in
sync with the egglog RHS by hand (Checkpoint A does not re-check rule bodies).

## Step 5 — Verification gate before Stage 3

All must pass:

```bash
# 1. Fusion fires, no weight dropped, graph shrinks:
nix develop --command cargo test -p rewrite --test fuse_all_models fuse_<model>
nix develop --command cargo test -p rewrite --test negative_fusions

# 2. Lean builds (fast signal on a new rule_* / soundRules mismatch):
nix develop --command bash -c "cd lean-plow && lake build"

# 3. Checkpoint A accepts the live rule catalog:
nix develop --command cargo test -p plowc --features lean-verify \
  --test lean_verify_rewrite -- --ignored
```

Gate to proceed to Stage 3 (Lean):

- [ ] Model lowers and saturates; extraction returns a term.
- [ ] Intended fusions fire; `ops_after < ops_before`; no weight leaf dropped.
- [ ] Negative-fusion tests still pass (no mis-firing).
- [ ] Every rule in `rules.egg` is in `soundRules` with a `rule_*` theorem;
      `lake build` and `lean_verify_rewrite` pass. **Every fireable rule is
      sound** — this is the invariant Stage 3 enforces.

## When to stop and ask

- The pattern needs to branch on an attribute the vocabulary collapses into an
  opaque token (attention config, grouped MoE routing) — the schema cannot
  express it; this is a vocabulary limit, not a missing rule.
- You cannot write the `rule_* := rfl` theorem because the fused op is *not*
  definitionally its unfused composition — the fusion may be unsound; do not
  force it.
- The needed change touches the cost model / extraction objective (you think a
  `:cost` is required) — that re-ranks the whole fusion space; flag it.
- The pattern is a genuine one-off with no reuse across layers/models.

## Report

State: which models were lowered and what fired (fused-op names + counts,
`ops_before → ops_after`); whether any new rule was added and its four-file
change; Checkpoint-A status (`lake build` + `lean_verify_rewrite`); and any
real-vs-ideal caveat (notably: the fused graph does not reach a GPU asset on the
shipping emit path — the deliverable is a correct, proven rule set).
