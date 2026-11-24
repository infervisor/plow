# Stage 2 — Egglog rewrite rules

Target-independent. Input: the Stage-1 `nn_graph::Graph`. Output: a rule set that covers the
model's fusible patterns, each rule proven sound (Checkpoint A).

## Where the rewrite runs

- `rewrite::rewrite_graph` lowers → saturates → extracts a `FusedGraph`.
- `rewrite::fused_sites_for_config` keys every extracted fused node by checkpoint weight name;
  `plowc --emit devblob` passes these sites to devgen (`PLOW_EMIT_REWRITE`, default on), and
  `crates/devgen/src/rewrite_lower.rs` maps fused kinds to the emitter fusions that lower them.
  A seam whose fused kind has no `Lowering` keeps the hand fusion.
- `rewrite::explore_stats` saturates without extracting (fusion-opportunity counts only).

Rules reach packets by default. A new fused kind reaches a kernel only when `rewrite_lower.rs`
lowers it.

## Reuse vs add

Most decoder LLMs need no new rule. Existing coverage (`crates/rewrite/src/egl/rules.egg`):

| fusion site | rule(s) | fused target |
|---|---|---|
| norm → projection | `rmsnorm-linear-fuse`, `rmsnorm-linearbias-fuse` | `FusedNormLinear(Bias)` |
| LayerNorm → projection | `layernorm-linear-fuse`, `layernorm-linearbias-fuse` | `FusedLayerNormLinear(Bias)` |
| gated MLP | `gated-mlp-fuse` | `SwiGLU` |
| residual add → norm | `residual-rmsnorm-fuse`, `residual-layernorm-fuse` | `FusedResidualNorm` / `FusedResidualLayerNorm` |
| linear → activation | `linear-act-fuse`, `linearbias-act-fuse` | `FusedLinearAct(Bias)` |
| embedding → scale | `embedding-scale-fuse` | `FusedEmbeddingScale` |
| qk-norm → RoPE | `rmsnorm-rope-fuse`, `rmsnorm-rope-scale-fuse` | `FusedNormRope(Scale)` |
| GroupNorm → act (→ Conv3d) | `groupnorm-act-fuse`, `groupnorm-act-conv3d(-bias)-fuse` | `FusedGroupNormAct[Conv3d[Bias]]` |
| AdaLN modulate, gated residual | `adaln-modulate-fuse`, `gated-residual-fuse` | `FusedAdaLN`, `FusedGatedResidual` |
| output gate | `kda-gated-norm-fuse`, `mla-out-gate-fuse` | `FusedKdaGatedNorm`, `FusedMlaOutGate` |

- Reuse: the model expresses its compute with schema ops (`schema.egg`) in an existing shape.
- Add a rule: a recurring composition no rule matches. Either schema ops in a new arrangement
  (add rule + target + theorem), or a new base op (extend `schema.egg` and `lower.rs` first).
- Do not add a rule for a one-off pattern or one you cannot prove sound.

## Adding a rule — schema, rules, lower (if needed), Lean in one change

1. **Fused target** — `crates/rewrite/src/egl/schema.egg` under `; === fused targets ===`. Carry
   every operand leaf of the unfused form, in the operand order of the Lean `expand`. A new base op
   goes into `Expr` with a `term_for` arm in `crates/rewrite/src/lower.rs`. An op with no term
   returns `LowerError::Unsupported`; never lower to `Opaque` (single-input passthrough, drops
   operands).
2. **Rule** — `crates/rewrite/src/egl/rules.egg`. Every `(rewrite …)` is immediately preceded by
   exactly one `; rule: <kebab-name>`; `parse_rule_catalog` (`crates/plowc/src/lib.rs`) rejects a
   dangling annotation or an unannotated rewrite. Pin constant discriminants as literals
   (`"sigmoid"`, `"mul"`) to keep distinct e-nodes distinct.
   ```egglog
   ; rule: gated-mlp-fuse
   (rewrite (Ew "mul" (Act ?k ?g) ?u)
            (SwiGLU ?k ?g ?u))
   ```
3. **Cost** — no `:cost` annotations. Extraction uses `extract::BigTreeAdditiveCost` (uniform head
   weight 1, `BigInt`), so a fused form wins by having fewer e-nodes. A `:cost` is silently ignored;
   a different cost function re-ranks the fusion space.
4. **Lean** — `lean-plow/Plow/Rewrite.lean`: add the fused variant to `inductive Op`, extend
   `expand` so it unfolds to exactly the unfused composition, add `rule_<name_with_underscores>`
   proven by `rfl`, and add the kebab name to `soundRules`.
   ```lean
   theorem rule_gated_mlp_fuse (g u : Op) (k : String) :
       expand (Op.SwiGLU g u k) =
         Op.Ew "mul" (Op.Act k (expand g)) (expand u) := rfl
   ```
5. **Emitter** — to reach a packet, map the fused kind to a `Lowering` in
   `crates/devgen/src/rewrite_lower.rs` and have the emitter query it.
6. **Tests** — a `fuse_<model>` case in `crates/rewrite/tests/fuse_all_models.rs` asserting the
   fused op names, `ops_after < ops_before`, and fused weight set = graph weight set
   (`fused_weights` vs `graph_weights`); sites in `crates/rewrite/tests/fused_sites.rs` when the
   kind is lowered.

## Gate (all before Stage 3)

```bash
nix develop --command cargo test -p rewrite --test fuse_all_models fuse_<model>
nix develop --command cargo test -p rewrite --test negative_fusions
nix develop --command bash -c "cd lean-plow && lake build"
nix develop --command cargo test -p plowc --test lean_verify_rewrite -- --ignored
```

- [ ] The model lowers and saturates; extraction returns a term (no `LowerError::Unsupported` /
      `UnmappedTensor`).
- [ ] Intended fusions fire; `ops_after < ops_before`; no weight leaf dropped.
- [ ] Negative-fusion tests pass (no mis-firing).
- [ ] Every rule in `rules.egg` is in `soundRules` with a `rule_*` theorem; `lake build` and
      `lean_verify_rewrite` pass.

## Pitfalls

- A rule without its Lean theorem fails Checkpoint A (`checkA` rejects names not in `soundRules`).
- Name mismatch: kebab-case in `rules.egg` and `soundRules`, `rule_<underscores>` in Lean.
- Checkpoint A does not re-check rule bodies: keep the `rules.egg` RHS and `expand` in sync and pin
  operands in a `fuse_*` test.
- `Attention` collapses its config into one opaque token: no rule can branch on an attention
  attribute. That is a vocabulary limit, not a missing rule.
- Extraction can hit an upstream egglog panic on some large graphs; `explore_stats` (saturate +
  `print-size`) never enters that path.

## Stop and ask

- The pattern must branch on an attribute the vocabulary makes opaque (attention config, grouped
  routing).
- The `rule_* := rfl` theorem does not hold: the fusion may be unsound; do not force it.
- The change needs a cost-model or extraction-objective change.
- The pattern is a one-off with no reuse.

## Code pointers

- `crates/rewrite/src/`: `egl/schema.egg`, `egl/rules.egg`, `lower.rs` (`lower`, `term_for`),
  `extract.rs` (`BigTreeAdditiveCost`, `FusedGraph`), `lib.rs` (`rewrite_graph`,
  `fused_sites_for_config`, `explore_stats`).
- `crates/devgen/src/rewrite_lower.rs` — fused kind → emitter fusion.
- `lean-plow/Plow/Rewrite.lean` (`expand`, `soundRules`); `lean-plow/Plow/CLI/Checkpoints.lean`
  (`checkA`); `crates/plowc/src/lib.rs` (`parse_rule_catalog`);
  `crates/lean_verify/src/checkpoints/rewrite.rs`.
- `docs/arch/08-formal-verification.md` (Checkpoint A), `docs/arch/01-compiler-pipeline.md`.
