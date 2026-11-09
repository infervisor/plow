# Stage 1 — Add the architecture to the nn-graph IR and the devblob emitter

Target-independent: name no GPU, ISA, CU count or toolchain here. The target block
([`target.md`](target.md)) is filled at Stage 4.

Output: a shape-inferred `nn_graph::Graph` for the model, and the same model wired through the
shipping `--emit devblob` path.

## Inputs

- `MODEL`: HF id (e.g. `Qwen/Qwen3-8B`) or a local checkpoint dir.
- `REFERENCE`: `modeling_*.py`, `config.json`, `model.safetensors.index.json`. Ground truth for op
  order, norms, activation, RoPE style, head geometry and weight names.

## What the IR feeds

- `crates/nn-graph` turns `config.json` into a typed operator DAG, infers shapes and lists the
  per-op weights (`Graph::weight_manifest`).
- The graph reaches packets through the rewrite: `rewrite::fused_sites_for_config` extracts fused
  sites and devgen lowers them at the seams a rule covers (`PLOW_EMIT_REWRITE`, default on).
  Without a rewrite for the checkpoint the emit keeps the hand fusions; only an explicit
  `PLOW_EMIT_REWRITE=1` refuses. `plowc` also builds the graph for fusion coverage
  (`crates/plowc/src/fusion_coverage.rs`).
- The device `LayerPlan`, weight binding and packets come from `crates/plowc/src/hf_config.rs` →
  `crates/devgen`. A builder in `nn-graph` alone does not make the model emit; step 7 does.
- `DType` is load-bearing (`costmodel::dtype_cost`, every `rewrite::OpSpec`). Do not touch it for a
  new model.

## Geometry audit — record before building

| field | decides downstream |
|---|---|
| `layer_types` (full vs sliding) | flash arms needed; sliding layers get a KV ring |
| `head_dim` per family | one kernel instantiation per head dim |
| `num_key_value_heads` per family | distinct GEMV/flash shapes per kvh |
| `sliding_window` | KV ring floor in the Stage 6 VRAM budget |
| `vocab_size` | lm_head GEMV width, sampler scratch |
| `rope_parameters` | per-family RoPE (partial RoPE, per-layer-type theta) |
| MoE vs dense | router / experts / combine op family |
| `num_hidden_layers` | segment count (the loader caps segments) |

Check GEMM tile divisibility for every projection: `N` must divide the object's N-tile, `K` its
K-tile, or the GEMM leaves the tiled path (refuses at emit or falls back). AMD tile:
`hwspec::IsaLevel::geometry()`; NVIDIA: read it off the built object (`plowc tune inventory`). See
[`target.md`](target.md).

## Templates

| new model is | copy from |
|---|---|
| dense GQA + SwiGLU, RoPE, pre-norm | `models/llama.rs` |
| + per-head qk-norm / explicit `head_dim` | `models/qwen3.rs` |
| MLA + MoE | `models/deepseek.rs` |
| local/global attention, softcap | `models/gemma.rs` |
| hybrid softmax/linear attention | `models/kimi_k3.rs` |
| ViT / vision encoder | `models/siglip.rs` |
| diffusion DiT / VAE (resolution-bucketed) | `models/qwen_image_*.rs` |

Paths are under `crates/nn-graph/src/`.

## Steps

All commands run inside `nix develop`.

0. **Baseline green.**
   ```bash
   nix develop --command bash -c '
     cargo build -p nn-graph --no-default-features &&
     cargo build -p nn-graph &&
     cargo build -p nn-graph --features hub &&
     cargo test -p nn-graph'
   ```
1. **Config** — `models/config/<arch>.rs`: `#[derive(Deserialize)]` with only the geometry the
   builder needs. `#[serde(default = "…")]` only for fields the reference defaults; never blanket
   `#[serde(default)]` over required dimensions. `#[serde(alias)]` for alternate spellings. Add
   `kv_heads()` / `head_dim()` helpers.
2. **Detection** — `models/config/mod.rs`: `mod <arch>;`, `pub use <arch>::<Arch>Config;`, a
   `ModelConfig::<Arch>` variant, a `from_json` arm on `model_type`, and an `architectures[]` arm in
   `model_type()` if the model may ship without `model_type`. Reject near-variants you do not
   implement with `ConfigError::Unsupported`.
3. **Builder** — `models/<arch>.rs`: `pub fn build(cfg: &<Arch>Config) -> Graph` driving `Nn`
   (`src/builder.rs`). Symbolic `B`/`S`; per layer `begin_block("layers.{i}")` … `end_block()`; ops
   in reference order; weight names exactly as in the checkpoint; `mark_output` logits (decoder) or
   final hidden states (encoder). Add the arm to `models/mod.rs::build_graph` (and
   `build_encoder_graph` for a text encoder).
4. **New op only if forced** — `Op` variant (`op.rs`) + shape rule (`infer.rs`) + `Nn` helper +
   `Op::name`. Bar: any expression with existing ops would be numerically close but wrong (see the
   `op.rs` doc comments).
5. **Test** — a case in `tests/build_graphs.rs` with a scaled-down, structurally faithful inline
   config: `assert_fully_inferred(&g)`, expected `output_shape_str(&g)`, per-op `count_ops` for the
   ops that define the architecture. Add a "missing required field is rejected" test.
6. **Weight manifest and binding** — compare `weight_manifest()` names/shapes to the checkpoint
   index (pattern: `tests/bind_and_weights.rs`). Bind a bucket with
   `g.bind(&Bindings::new().set("B", 1).set("S", 512))`; diffusion models specialize at build time
   via `build_from_config_json_at(json, &ShapeBucket::square(N))`.
7. **Devblob emit wiring** — wire the model independently through:
   - `crates/plowc/src/hf_config.rs`: parse and validate the real config, synthesize the full-model
     `LayerPlan`, refuse unsupported near-variants;
   - `crates/devgen`: dispatch the family and emit full-model and `--block` packets with
     checkpoint-exact weight names and shapes.

   Mirror the Stage-1 geometry and manifest. Add parser/plan tests in `hf_config.rs`, emitter tests
   in the matching `devgen` module and an `hf_dir_compile`-style test
   (`crates/plowc/tests/hf_dir_compile.rs`). A structural fixture may use zero weights; its manifest
   is compared to the real checkpoint index.
   ```bash
   nix develop --command cargo test -p plowc -p devgen
   nix develop --command plowc --hf-dir <checkpoint> --emit devblob --out <scratch>/emit
   nix develop --command plowrt simulate --assets <scratch>/emit --all-buckets
   ```

## Inspecting a graph

`plowc viz` builds every network of a source, infers shapes and writes or serves a viewer. Source
flags are top-level: `plowc --hf-dir <dir> viz …` or `plowc --model <hf-id> viz …`.

```bash
nix develop --command plowc --hf-dir <dir> viz --port 0 --out <scratch>/graph.html
nix develop --command plowc --model <hf-id> viz --revision <commit> --validate-checkpoint \
  --port 0 --dump-assets <scratch>/viz
nix develop --command plowc viz --packet <assets>/model.pkt --open false
```

`--validate-checkpoint` checks graph weights against the safetensors headers only; it does not
prove numerical equivalence. Multi-stage models get named stages from an arm in
`viz::bundle_from_model_config`. Weightless structure-only packets: `scripts/tts/header_only.py`.

## Gate (all before Stage 2)

- `nn-graph` builds with `--no-default-features`, default and `--features hub`; `cargo test -p
  nn-graph` green including the new `build_graphs` case and the rejection test.
- `assert_fully_inferred` passes; `weight_manifest()` matches the checkpoint names and shapes.
- Op order and attributes match `REFERENCE` (norms, activation, RoPE, routing).
- Full and `--block` emit succeed, `plowrt simulate --all-buckets` terminates on every program, and
  the shipping manifest matches the Stage-1 / reference tensor names and shapes.

## Pitfalls

- Blanket `#[serde(default)]` fabricates models: a partial config builds a plausible wrong one
  (`kimi_partial_config_is_rejected_not_defaulted`).
- Config keys vary by checkpoint (`num_expert_group` vs `n_group`, `dtype` vs `torch_dtype`); use
  aliases.
- An `architectures[]` prefix can route a new model to an older family's builder. Claim the prefix
  explicitly.
- `rope` (pairs `x[i]`, `x[i+d/2]`) and `rope_interleaved` (pairs `x[2i]`, `x[2i+1]`) are different
  rotations; partial RoPE (`dim < head_dim`) is common.
- `GeluTanh` ≠ `Gelu` ≠ `QuickGelu`; SwiGLU is `mul(silu(gate), up)`.
- Drive per-layer heterogeneity (local/global, dense-before-MoE, hybrid attention) from the config's
  layer list, never a hardcoded stride.
- Flat top-k and group-limited routing select different experts; use `moe_router_grouped` with
  `MoeGroups` when the checkpoint was trained grouped.
- Attention output width is the value head dim (`qk_head_dim != v_head_dim` under MLA).
- Anything needing serde/hub stays behind the `models` / `hub` features.

## Stop and ask

- A new `Op` meets the bar: propose name, inputs, shape rule and why existing ops cannot express it.
- `config.json` semantics or checkpoint tensor names disagree with the reference.
- Detection would collide with an existing family (`model_type` or `architectures` prefix).
- Keeping `--no-default-features` green would mean moving something into the core IR.
- Multimodal / multi-network model: unclear whether to build one graph or a `PipelineConfig`.

Do not edit `DType` or start GPU work in this stage.
