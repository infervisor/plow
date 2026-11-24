# Stage 2 — Egglog Rewrite Rules

Target-independent stage. Extends the term rewrite engine to cover new fusible subgraphs, ensuring every rule is formally proven sound via Checkpoint A.

* **Input:** The `nn_graph::Graph` from Stage 1.
* **Output:** Egglog rewrite rules covering recurring operator compositions, proven in Lean 4.

---

## 1. Architecture

* **Engine:** `crates/rewrite` lowers the graph to an e-graph, saturates rewrite rules, and extracts a `FusedGraph`.
* **Devblob bridge:** `plowc --emit devblob` extracts fused sites (`rewrite::fused_sites_for_config`) and passes them to `devgen::rewrite_lower`.
* **Execution default:** Rewrite lowering is enabled by default (`PLOW_EMIT_REWRITE=1`). Unmatched subgraphs retain hand fusions.

---

## 2. Existing Coverage

Review existing rules in `crates/rewrite/src/egl/rules.egg` before creating new ones:

| Fusion Pattern | Existing Rule | Fused Target |
|---|---|---|
| RMSNorm / LayerNorm → Projection | `rmsnorm-linear(-bias)-fuse`<br>`layernorm-linear(-bias)-fuse` | `FusedNormLinear(Bias)`<br>`FusedLayerNormLinear(Bias)` |
| Gated MLP (Gate + Up + Act) | `gated-mlp-fuse` | `SwiGLU` |
| Residual Add → Norm | `residual-rmsnorm-fuse`<br>`residual-layernorm-fuse` | `FusedResidualNorm`<br>`FusedResidualLayerNorm` |
| Linear → Activation | `linear-act-fuse`<br>`linearbias-act-fuse` | `FusedLinearAct(Bias)` |
| Embedding → Scale | `embedding-scale-fuse` | `FusedEmbeddingScale` |
| QK-Norm → RoPE | `rmsnorm-rope(-scale)-fuse` | `FusedNormRope(Scale)` |
| GroupNorm → Act → Conv3D | `groupnorm-act-conv3d-fuse` | `FusedGroupNormActConv3d` |
| AdaLN Modulation / Gated Residual | `adaln-modulate-fuse`<br>`gated-residual-fuse` | `FusedAdaLN`<br>`FusedGatedResidual` |
| Output Gate Projections | `kda-gated-norm-fuse`<br>`mla-out-gate-fuse` | `FusedKdaGatedNorm`<br>`FusedMlaOutGate` |

*Rule policy:* Do not add rules for one-off patterns. Only add rules for compositions that repeat across layers and have hardware kernel lowering support.

---

## 3. Adding a New Rewrite Rule

Adding a rule requires updates across `crates/rewrite`, `crates/devgen`, and `lean-plow` in a single change.

### Step 1: Define Fused Target
In `crates/rewrite/src/egl/schema.egg`:
* Declare the fused node constructor under `; === fused targets ===`.
* Include all leaf operands required by the unfused composition, matching Lean expansion order.

### Step 2: Implement Rewrite Rule
In `crates/rewrite/src/egl/rules.egg`:
* Every rule must be preceded by an exact kebab-case identifier comment: `; rule: <name>`.
* Keep constant discriminants (e.g. `"mul"`, `"silu"`) as explicit string literals.
```egglog
; rule: gated-mlp-fuse
(rewrite (Ew "mul" (Act ?k ?g) ?u)
         (SwiGLU ?k ?g ?u))
```

### Step 3: Formal Verification in Lean 4
In `lean-plow/Plow/Rewrite.lean`:
1. Add the fused variant to `inductive Op`.
2. Extend `def expand` to unfold the fused variant to its exact unfused syntax.
3. Prove soundness via reflexivity:
   ```lean
   theorem rule_gated_mlp_fuse (g u : Op) (k : String) :
       expand (Op.SwiGLU g u k) =
         Op.Ew "mul" (Op.Act k (expand g)) (expand u) := rfl
   ```
4. Register the kebab-case name in `def soundRules : List String`.

### Step 4: Emitter Lowering
In `crates/devgen/src/rewrite_lower.rs`:
* Map the extracted fused kind to the target emitter constructor.

### Step 5: Unit Tests
* Add test case in `crates/rewrite/tests/fuse_all_models.rs`.
* Verify `ops_after < ops_before` and assert all original weights are preserved (`fused_weights == graph_weights`).

---

## 4. Gate Checklist

Execute within `nix develop`:
```bash
nix develop --command cargo test -p rewrite --test fuse_all_models fuse_<model>
nix develop --command cargo test -p rewrite --test negative_fusions
nix develop --command bash -c "cd lean-plow && lake build"
nix develop --command cargo test -p plowc --test lean_verify_rewrite -- --ignored
```

- [ ] Graph lowers, saturates, and extracts without `LowerError::Unsupported` or `UnmappedTensor`.
- [ ] Target fusions fire; extracted operation count decreases (`ops_after < ops_before`).
- [ ] No weight tensors are dropped or duplicated.
- [ ] `lake build` succeeds with zero warnings (`warningAsError := true`).
- [ ] Every rule in `rules.egg` has a corresponding verified theorem in `soundRules`.

---

## 5. Common Pitfalls

* **Name discrepancies:** Rule names must match exactly: kebab-case in `rules.egg` and `soundRules`, snake-case (`rule_<name>`) in Lean theorems.
* **Opaque terms:** Never lower unhandled operators to `Opaque` (which drops inputs). Return `LowerError::Unsupported`.
* **Cost annotations:** Do not add `:cost` annotations to Egglog rules. Extraction relies on `BigTreeAdditiveCost`, optimizing strictly for e-node count.
