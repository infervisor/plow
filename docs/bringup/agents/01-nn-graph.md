# Agent prompt — Stage 1: add a model architecture to the nn-graph IR

You are a coding agent bringing up a new model in **plow**. This is **Stage 1**:
add the architecture to the `nn-graph` operator IR so its `config.json` parses,
builds a faithful operator graph, and infers shapes. You are NOT making the model
run on a GPU — that is later-stage work. Stay in scope.

Fill these in before starting:
- `MODEL`: the target (HF id, e.g. `Qwen/Qwen3-8B`, or a local config path).
- `REFERENCE`: the reference implementation (`modeling_*.py` and the model's
  `config.json` + `model.safetensors.index.json`). This is ground truth.

---

## Read first (do not skip)

1. `docs/bringup/01-nn-graph.md` — the playbook for this stage. Follow it.
2. `crates/nn-graph/src/lib.rs` — the "Where this sits in plow, honestly" note.
   Confirm you are editing a `models/` builder, not `DType`.
3. `crates/nn-graph/src/builder.rs` — the `Nn` API. Every op you can emit.
4. `crates/nn-graph/src/op.rs` and `src/infer.rs` — the operator set and its
   shape rules. Know what exists before considering a new op.
5. The **closest existing builder** as your template — pick from:
   `models/llama.rs` (dense GQA+SwiGLU), `models/qwen3.rs` (+ qk-norm),
   `models/deepseek.rs` (MLA+MoE), `models/gemma.rs` (local/global + softcap),
   `models/kimi_k3.rs` (hybrid linear attention), `models/siglip.rs` (ViT),
   `models/qwen_image_*.rs` (diffusion, resolution-bucketed).
6. `crates/nn-graph/tests/build_graphs.rs` — copy this test style verbatim.
7. `REFERENCE` — read the actual forward pass. Note: norm placement, activation
   (SiLU vs GeLU-tanh vs QuickGELU), RoPE style (interleaved vs half-split,
   partial), head geometry (`head_dim` vs `hidden/heads`, GQA, MLA
   qk/v head-dim split), MoE routing (flat top-k vs group-limited), and any
   per-layer heterogeneity.

Then confirm your baseline is green:

```bash
nix develop --command bash -c 'cargo build -p nn-graph --no-default-features && cargo test -p nn-graph'
```

---

## Edits to make

Touch only `crates/nn-graph`. Match existing style exactly.

1. **Config** — `crates/nn-graph/src/models/config/<arch>.rs`: a
   `#[derive(Deserialize)]` struct with only the geometry the builder needs.
   `#[serde(default = "…")]` **only** for fields the reference truly defaults;
   never blanket `#[serde(default)]` over required dimensions. `#[serde(alias)]`
   for alternate spellings. Add `kv_heads()` / `head_dim()` helpers as needed.

2. **Register** — `crates/nn-graph/src/models/config/mod.rs`: `mod <arch>;`,
   `pub use <arch>::<Arch>Config;`, a `ModelConfig::<Arch>(…)` variant, an arm in
   `from_json` matching the `model_type`, and (if the model may ship without one)
   an `architectures[]` arm in `model_type()`. Explicitly `Unsupported`-reject a
   near-variant you are not implementing.

3. **Builder** — `crates/nn-graph/src/models/<arch>.rs`: `pub fn build(cfg:
   &<Arch>Config) -> Graph` driving `Nn`. Per-layer `begin_block`/`end_block`.
   Emit ops in reference order. **Weight names must match the checkpoint exactly.**
   `mark_output` the logits (decoder) or final hidden states (encoder). Register
   in `models/mod.rs::build_graph` (and `build_encoder_graph` if applicable).

4. **New op only if forced** — if the IR cannot express an op faithfully with
   what exists, add an `Op` variant (`op.rs`) + shape rule (`infer.rs`) + `Nn`
   helper (`builder.rs`) + `Op::name`. The bar: a plausible expression with
   existing ops would be numerically-close-but-wrong. Otherwise reuse ops.

5. **Test** — add a case to `tests/build_graphs.rs` with a scaled-down but
   structurally faithful inline config. Assert `assert_fully_inferred(&g)`, the
   expected `output_shape_str(&g)`, and per-op `count_ops` for the ops that
   define this architecture (attention count, MoE routers, RoPE interleave,
   broadcasts, convs — whatever distinguishes it). Add a rejection test if the
   config has required geometry that must not default.

---

## Verification gate (all must pass before Stage 2)

```bash
nix develop --command bash -c '
  cargo build -p nn-graph --no-default-features &&   # pure IR still compiles
  cargo build -p nn-graph &&
  cargo build -p nn-graph --features hub &&
  cargo test -p nn-graph'
```

Then a build/inference smoke check of the real config (optional but recommended):

```bash
nix develop --command bash -c 'cargo run -q -p plowc -- viz --hf-dir <dir> --port 0 --out /tmp/graph.html'
# or --model <hf-id> to fetch config.json from the hub
```

Gate criteria:
- `--no-default-features` compiles (no model/serde leak into the core IR).
- All `nn-graph` tests pass, including your new build test and (if applicable)
  a "missing-field is rejected, not defaulted" test.
- Every node output has an inferred shape (`assert_fully_inferred`).
- `weight_manifest()` names match the checkpoint's tensor names and shapes —
  verify by printing it in a throwaway test:
  ```rust
  for w in nn_graph::models::build_from_config_json(cfg).unwrap().weight_manifest() {
      println!("{:32} {:?} {:?}", w.name, w.dtype, w.shape);
  }
  ```
- Op order/attributes match `REFERENCE` (norms, activation, RoPE, routing).

Only when this gate is green is the graph ready for **Stage 2 (egglog)**.

---

## When to stop and ask

- The reference does something no existing `Op` can faithfully express AND the
  "add a new op" bar is genuinely met — surface the proposed op (name, inputs,
  shape rule, why existing ops can't express it) before adding it.
- `config.json` field names/semantics are ambiguous vs the reference weights, or
  the checkpoint's tensor names don't match what your builder emits.
- Architecture detection would collide with an existing family (a shared
  `model_type` or `architectures` prefix).
- `--no-default-features` breaks and the fix would mean moving something into the
  core IR.
- The model is multimodal / multi-network (text encoder + DiT + VAE) and it's
  unclear whether to build one graph or a `PipelineConfig` of several.

Do not: touch crates other than `nn-graph`; edit `DType`; attempt emission or
GPU work; commit. Report what you changed, the passing gate output, and any
place the model departed from its template builder.
