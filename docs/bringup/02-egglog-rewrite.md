# Stage 2 — Egglog Rewrite Rules

Target-independent stage (CPU-only). Extends the term rewrite engine to discover and lower fusible operator subgraphs, with every rewrite rule proven sound in Lean 4.

* **Input:** The `nn_graph::Graph` from Stage 1.
* **Output:** Egglog rewrite rules covering recurring operator compositions, proven sound in Lean 4 (`soundRules`), and lowered into `devgen::rewrite_lower`.
* **Architecture Reference:** [`docs/arch/01-compiler-pipeline.md`](../arch/01-compiler-pipeline.md) §4 and [`docs/arch/08-formal-verification.md`](../arch/08-formal-verification.md).

---

## 1. Rewrite Engine Architecture

* **Engine:** `crates/rewrite` converts the operator DAG to an e-graph, runs equality saturation using Egglog rules, and extracts a minimal-cost `FusedGraph`.
* **Devblob Bridge:** `plowc --emit devblob` queries fused sites via `rewrite::fused_sites_for_config` and lowers them through `devgen::rewrite_lower`.
* **Default Behavior:** Rewrite lowering is active by default (`PLOW_EMIT_REWRITE=1`). Subgraphs not matching any rewrite rule fall back to standard non-fused emitter lowerings.

---

## 2. Standard Pattern Coverage

Review existing rules in `crates/rewrite/src/egl/rules.egg` before authoring new patterns:

| Fusion Pattern | Existing Egglog Rule | Fused Target Node |
|---|---|---|
| RMSNorm / LayerNorm → Projection | `rmsnorm-linear(-bias)-fuse`<br>`layernorm-linear(-bias)-fuse` | `FusedNormLinear(Bias)`<br>`FusedLayerNormLinear(Bias)` |
| Gated MLP (Gate + Up + Act) | `gated-mlp-fuse` | `SwiGLU` |
| Residual Add → Norm | `residual-rmsnorm-fuse`<br>`residual-layernorm-fuse` | `FusedResidualNorm`<br>`FusedResidualLayerNorm` |
| Linear → Activation | `linear-act-fuse`<br>`linearbias-act-fuse` | `FusedLinearAct(Bias)` |
| Embedding → Scale | `embedding-scale-fuse` | `FusedEmbeddingScale` |
| QK-Norm → RoPE | `rmsnorm-rope(-scale)-fuse` | `FusedNormRope(Scale)` |
| GroupNorm → Act → Conv3D | `groupnorm-act-conv3d-fuse` | `FusedGroupNormActConv3d` |
| AdaLN Modulation / Gated Residual | `adaln-modulate-fuse`<br>`gated-residual-fuse` | `FusedAdaLN`<br>`FusedGatedResidual` |
| Output Gate Projection | `output-gate-fuse` | `FusedLinearAct` |

---

## 3. Step-by-Step Rule Addition Workflow

### Step 1: Define Egglog Rule (`crates/rewrite/src/egl/rules.egg`)
* Express the pattern using canonical `Op` syntax:
  ```lisp
  (rule ((= fused (Add (Linear ?x ?w1) (Linear ?x ?w2))))
        ((union fused (FusedDualLinear ?x ?w1 ?w2))))
  ```
* Invariant: Never eliminate an intermediate node that has consumers outside the matched pattern.
* Invariant: External input tensors and weight bindings must be preserved bit-for-bit.

### Step 2: Formulate Lean 4 Soundness Proof (`lean-plow/LeanPlow/Rewrites.lean`)
Every rule added to `rules.egg` must have a corresponding semantic equivalence proof in Lean 4:
1. Declare the AST pattern in `inductive Op`.
2. Define the denotational semantics in `def expand : Op → Semantics`.
3. State and prove the equivalence theorem:
   ```lean
   theorem <rule_name>_sound (x w1 w2 : Tensor) :
     expand (Op.Add (Op.Linear x w1) (Op.Linear x w2)) =
     expand (Op.FusedDualLinear x w1 w2) := by
     ...
   ```
4. Append the theorem to `def soundRules : List RewriteRule`.

### Step 3: Implement Devblob Lowering (`crates/devgen/src/rewrite_lower.rs`)
* Wire the matched pattern to device generation:
  - Add extraction logic in `rewrite_lower.rs`.
  - Validate tensor shapes, strides, and memory alignments for the target fused kernel.
  - Emit packet instruction descriptors and register fused weight slices.

### Step 4: Unit & Property Tests (`crates/rewrite/tests/`)
* Verify pattern discovery:
  ```rust
  #[test]
  fn test_<pattern>_fires() {
      let graph = build_test_subgraph();
      let fused = rewrite::optimize(&graph);
      assert!(fused.has_op(Op::Fused<Pattern>));
  }
  ```
* Verify weight manifest preservation: assert count and hashes of weights remain identical before and after saturation.

---

## 4. Machine-Checked Gate Checklist

Before advancing to Stage 3:
- [ ] `cargo test -p rewrite` passes with 0 failures and 0 warnings.
- [ ] Lean 4 verification passes: `cd lean-plow && lake build` builds with 0 errors.
- [ ] The new rule is registered in `LeanPlow.Rewrites.soundRules`.
- [ ] `cargo test -p devgen --lib rewrite` passes for single-block and full-model emission.
- [ ] No intermediate tensors with external consumers are dropped during e-graph extraction.
- [ ] Emitted packet weight manifest matches checkpoint safetensors index bit-for-bit.
