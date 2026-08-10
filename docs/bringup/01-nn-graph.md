# Stage 1 — Add the architecture to the nn-graph operator IR

Part of the model-bringup playbook. Bringing up a new model in plow is a staged
process; this is the first stage. Its output — a shape-inferred operator graph —
is what every later stage consumes.

> Where this sits: `crates/nn-graph` is the **Stage-1 frontend** of the compiler
> pipeline (see `docs/arch/01-compiler-pipeline.md`). It turns a HuggingFace
> `config.json` into a typed operator DAG (`nn_graph::Graph`), runs symbolic
> shape inference, and enumerates the per-op weights a loader will parse later.

---

## Goal

Add a new model architecture to the `nn-graph` model zoo so that:

1. A HuggingFace `config.json` for the architecture parses into a typed config
   struct and is routed to the architecture by `model_type` / `architectures`.
2. A builder emits a faithful `nn_graph::Graph` over the existing operator IR.
3. Shape inference fills every node output; binding `B`/`S` (and, for diffusion,
   the resolution bucket) produces concrete shapes.
4. The graph's `weight_manifest()` names every checkpoint tensor the graph needs.

The deliverable is a graph that *builds and infers*. Faithfulness to the
reference model (which ops, in which order, with which attributes) is the whole
point — a graph that builds but models the wrong architecture is the failure
mode this IR is designed to make hard to reach.

---

## Inputs

- A HuggingFace model id (e.g. `Qwen/Qwen3-8B`) or a local `config.json`.
- The reference implementation (usually `modeling_*.py` in the HF repo / the
  `transformers` or `diffusers` source) — the ground truth for op order,
  norms, activation choice, RoPE style, and head geometry.
- The published weight names (`model.safetensors.index.json` or the tensor
  names in the checkpoint) — the manifest must match these exactly.

---

## What you are editing, and what it affects

Read the crate's own honesty note (`crates/nn-graph/src/lib.rs`, "Where this
sits in plow, honestly"). Two consumers, very unequal:

- **`DType`** is load-bearing — it reaches emitted GPU code via
  `costmodel::dtype_cost`. **You will almost never touch it** for a new model.
- **A builder in `models/`** changes analysis/visualization output and nothing
  on the shipping `--emit devblob` path. Adding an architecture here is safe and
  self-contained; it will not by itself make the model *run* on a GPU.

So Stage 1 is about a correct, inspectable IR — not about emission.

---

## The recipe (what existing models did)

Every text model in the zoo (`llama`, `qwen3`, `gemma`, `deepseek`, `glm`,
`kimi`, `kimi_k3`) follows the same four-part shape. Use the closest existing
architecture as your template:

| If the new model is…                                   | Copy from            |
|--------------------------------------------------------|----------------------|
| Dense GQA + SwiGLU, RoPE, pre-norm (Llama/Mistral-like)| `models/llama.rs`    |
| Same but with per-head qk-norm / explicit `head_dim`   | `models/qwen3.rs`    |
| MLA + MoE (DeepSeek-V3-like)                            | `models/deepseek.rs` |
| MLA + MoE + per-layer local/global, softcap (Gemma-3/4)| `models/gemma.rs`    |
| Hybrid softmax/linear-attention, exotic residual       | `models/kimi_k3.rs`  |
| ViT / vision encoder (conv patch embed, non-causal)    | `models/siglip.rs`   |
| Diffusion DiT / VAE (resolution-bucketed)              | `models/qwen_image_*`|

The four parts:

### 1. Config struct — `crates/nn-graph/src/models/config/<arch>.rs`

A serde `Deserialize` struct holding exactly the geometry the builder needs.
Pattern (see `config/llama.rs`, `config/qwen3.rs`):

- Fields are `i64` / `u32` / `f32` / `Option<…>` matching the HF config keys.
- Use `#[serde(default = "…")]` **only** for fields the reference genuinely
  defaults (e.g. `rope_theta`, `tie_word_embeddings`). **Do not** blanket
  `#[serde(default)]` over the whole geometry — a config missing a required
  dimension must *error*, not silently build a plausible-but-wrong model. This
  was a real bug: `KimiConfig` was `#[serde(default)]` end to end and a partial
  config fabricated a 61-layer model (see the `kimi_partial_config_is_rejected`
  test in `tests/build_graphs.rs`).
- Use `#[serde(alias = "…")]` when the checkpoint spells a field differently
  (Qwen3's `dtype` vs `torch_dtype`; K3's `num_expert_group` vs `n_group`).
- Add derived-geometry helper methods (`kv_heads()`, `head_dim()`) rather than
  recomputing in the builder.

Register the module and re-export the type in `config/mod.rs`:
- add `mod <arch>;`
- add `pub use <arch>::<Arch>Config;`
- add a variant to the `ModelConfig` enum.

### 2. Architecture detection — `config/mod.rs::from_json`

Route the checkpoint to your config. Two paths already exist:

- `ModelConfig::from_json` matches on `model_type` (and, for diffusers
  components, `_class_name`). Add an arm mapping the model's `model_type`
  string(s) to `Ok(ModelConfig::<Arch>(serde_json::from_value(v)?))`.
- `model_type()` provides an `architectures[]` fallback for configs with no
  `model_type`. Add a `starts_with(...)` arm if the model may ship that way.

If a *near* architecture already claims the name but is not actually supported,
reject it explicitly with `ConfigError::Unsupported` and a message that says why
(see the Gemma-1/2 and Qwen2-VL rejection arms). A wrong-but-successful build is
worse than a clear refusal.

### 3. Builder — `crates/nn-graph/src/models/<arch>.rs`

`pub fn build(cfg: &<Arch>Config) -> Graph`, driving `Nn` (`src/builder.rs`):

- `let mut nn = Nn::new(act_dtype, weight_dtype);` — usually both
  `parse_dtype(cfg.torch_dtype.as_deref())`.
- Symbolic dims: `let b = nn.sym("B"); let s = nn.sym("S");`
- Input: `nn.input("input_ids", nn.shape([b, s]), DType::I32);`
- Per layer, wrap in `nn.begin_block(&format!("layers.{layer}"))` …
  `nn.end_block()` so the graph's block structure matches the reference. The
  block label is what `viz` groups on.
- Emit ops via `Nn` helpers (`linear`, `rmsnorm`, `layernorm`, `embedding`,
  `rope` / `rope_interleaved`, `attention`, `act`, `add`/`mul`, `reshape`,
  `moe_router` / `moe_router_grouped`, `conv2d`, `conv3d`, `groupnorm`, …).
  Weight names are built from the `name` prefix you pass — **make them match the
  checkpoint's tensor names exactly** (`{prefix}.self_attn.q_proj.weight`, etc).
- Mark outputs with `nn.mark_output(id)` (logits for a decoder; final hidden
  states for an encoder build).
- Return `nn.finish()`.

Wire the builder into `models/mod.rs::build_graph` (add a `match` arm on your
`ModelConfig` variant → `<arch>::build(c)`). `build_graph` calls
`infer_shapes` for you. If the model can be used as a text encoder (no lm_head),
also add a `build_encoder` arm to `build_encoder_graph` — see llama/qwen3.

If your model needs an operator the IR does not have, that is a bigger change:
add an `Op` variant in `src/op.rs`, a shape rule in `src/infer.rs`, an `Nn`
helper in `src/builder.rs`, and a name in `Op::name`. Prefer reusing existing
ops. Only add an op when a wrong-but-plausible expression of it with existing
ops would ship silently (the `SituGlu` / `BlockResidual` / `LinearAttention`
doc comments in `op.rs` explain when that bar is met).

### 4. Weight manifest (free, but verify)

`Graph::weight_manifest()` (`src/graph.rs`) already enumerates every `Weight`
input across all nodes, tagged with the op, name, dtype, and inferred shape. You
get it for free by declaring weights through `Nn::param` / the layer helpers.
Your job is to confirm it lists the right names and shapes (see below).

---

## Step-by-step

All commands run inside the dev shell.

**0. Sanity: the crate builds three ways.**

```bash
nix develop --command bash -c '
  cargo build -p nn-graph --no-default-features &&   # pure IR, no serde
  cargo build -p nn-graph &&                         # + models
  cargo build -p nn-graph --features hub'            # + hub (hf-hub/TLS)
```

**1. Get the config.** Either point at a local dir or fetch from the hub:

```bash
# Local: you already have config.json.
# Hub (needs the `hub` feature; downloads config.json via hf-hub 0.4):
nix develop --command bash -c 'cargo run -q -p plowc -- viz --model <hf-id> --port 0 --out /tmp/graph.html'
```

`plowc viz` is the end-to-end Stage-1 driver: it resolves the source
(`--hf-dir <dir>`, `--model <hf-id>`, or `--net <file.json>`), builds the graph,
runs inference, and writes a self-contained HTML DAG viewer (or serves it with
`--port <n>`). Use it to eyeball op order and per-tensor shapes.

**2. Add the config struct + detection + builder** as in the recipe above.

**3. Add a build test** in `crates/nn-graph/tests/build_graphs.rs`. Follow the
existing pattern: a *scaled-down but structurally faithful* inline `config.json`
(few layers, small dims), then assert:

- `assert_fully_inferred(&g)` — every node output has a shape.
- `output_shape_str(&g)` equals the expected symbolic output (e.g.
  `"[B, S, 1000]"` for a decoder's logits).
- `g.count_ops(|o| matches!(o, Op::Attention { .. }))` etc. match the expected
  per-layer op counts — this is how you pin "the right ops in the right places"
  (see the deepseek / gemma4 / kimi_k3 tests for MoE routers, broadcasts, RoPE
  interleave counts, etc).

**4. Run the tests:**

```bash
nix develop --command bash -c 'cargo test -p nn-graph'
# or just this stage's suite:
nix develop --command bash -c 'cargo test -p nn-graph --test build_graphs'
```

**5. Inspect the weight manifest** for exact-name/shape agreement with the
checkpoint. There is no CLI dump; assert it in a test (see
`tests/bind_and_weights.rs`, `weight_manifest_lists_per_op_weights`) or add a
throwaway one:

```rust
let g = nn_graph::models::build_from_config_json(cfg).unwrap();
for w in g.weight_manifest() {
    println!("{:32} {:?} {:?}", w.name, w.dtype, w.shape);
}
```

**6. Bind to a concrete bucket** to confirm specialization works:

```rust
let mut g = nn_graph::models::build_from_config_json(cfg).unwrap();
g.bind(&nn_graph::Bindings::new().set("B", 1).set("S", 512));
```

Text models keep `B`/`S` symbolic until `bind`. Diffusion models are specialized
to resolution at *build* time via `ShapeBucket` (it changes graph structure, not
just dim values) — use `build_from_config_json_at(json, &ShapeBucket::square(N))`.

---

## Success criteria

- `cargo build -p nn-graph --no-default-features` compiles the pure IR (no
  serde, no models) — proves you didn't leak a model dependency into the core IR.
- `cargo build -p nn-graph` and `--features hub` compile.
- `cargo test -p nn-graph` is green, including a new `build_graphs` test that
  asserts the output shape and the per-op counts for the new architecture.
- `assert_fully_inferred` passes — inference reaches every node.
- `weight_manifest()` lists the checkpoint's tensor names with correct shapes.
- The config is rejected (not defaulted) when required geometry is missing.

---

## Pitfalls (all drawn from how existing models were added)

- **Blanket `#[serde(default)]` fabricates models.** A config missing a
  dimension must error. (`kimi_partial_config_is_rejected_not_defaulted`.)
- **Config-key spelling varies between checkpoints.** Use `#[serde(alias)]`;
  don't assume the DeepSeek spelling when the model uses its own
  (`num_expert_group` vs `n_group`; `dtype` vs `torch_dtype`).
- **`architectures[]`-fallback can route a new model to the wrong builder.**
  Kimi-K3's `KimiLinear*` prefix used to match the K2 arm and built a 61-layer
  K2 for a 93-layer K3 — a silent success. Claim your prefix explicitly.
- **RoPE style matters.** `rope` (Llama-style pairing `(x[0], x[d/2])`) vs
  `rope_interleaved` (GLM-style `(x[0], x[1])`) are different rotations. Partial
  RoPE (`dim < head_dim`) is common (Gemma-4 global layers, DeepSeek MLA).
- **Activation choice is load-bearing.** SwiGLU is `mul(silu(gate), up)`.
  `GeluTanh` ≠ `Gelu` ≠ `QuickGelu`; Qwen2.5-VL's merger uses plain `Gelu`
  (`qwen_vl_merger_uses_gelu`). Match the reference exactly.
- **Per-layer heterogeneity.** Gemma alternates local (windowed) / global
  attention; DeepSeek/Kimi have `first_k_dense_replace` dense layers before MoE;
  K3 is a per-layer MLA/KDA hybrid. Drive the choice from the config's own layer
  list, never from a hardcoded stride.
- **MoE routing is not a boolean.** Flat top-k and DeepSeek group-limited
  (`noaux_tc`) routing select *different* expert sets. Use
  `moe_router_grouped` with `MoeGroups` when the checkpoint was trained grouped;
  `num_experts` must be divisible by `n_group` (checked in `infer`).
- **Weight names must match the checkpoint byte-for-byte.** The manifest is the
  loader's contract; a `.weight` suffix or a `self_attn` vs `attn` prefix
  mismatch is a load failure later.
- **Attention output takes the *value* head dim, not the query's.** For MLA
  `qk_head_dim != v_head_dim`; the `Attention`/`LinearAttention` shape rules
  read the output width off `v` — reshape your merged output accordingly.
- **Don't add a new `Op` when existing ops express it faithfully.** But when a
  plausible expression with existing ops would be numerically-close-but-wrong
  (K3's `situ` GLU transforms the up-branch too; `BlockResidual` is not `Add`),
  a dedicated op is the correct call. Read the `op.rs` doc comments.
- **Keep the pure-IR build clean.** Anything the builder needs from serde/hub
  must stay behind the `models` / `hub` features. If `--no-default-features`
  stops compiling, you reached into the core IR.

---

## Code pointers

| Symbol / path                                             | What it is                              |
|----------------------------------------------------------|-----------------------------------------|
| `nn_graph::Nn` — `src/builder.rs`                        | Ergonomic layer builder (all op helpers)|
| `nn_graph::Op` — `src/op.rs`                             | Operator enum + attributes + doc bar    |
| `nn_graph::infer::infer_shapes` — `src/infer.rs`        | Per-op symbolic shape rules             |
| `nn_graph::Graph::weight_manifest` — `src/graph.rs`     | Per-op weight list (loader contract)    |
| `nn_graph::Graph::bind` / `Bindings` — `src/bind.rs`    | B/S/L specialization                    |
| `models::build_graph` / `build_from_config_json[_at]` — `src/models/mod.rs` | Build dispatch + inference |
| `models::build_encoder_graph` — `src/models/mod.rs`     | Encoder (no lm_head) build              |
| `models::config::ModelConfig::from_json` — `src/models/config/mod.rs` | Config parse + arch detection |
| `models::config::parse_dtype` — `src/models/config/mod.rs` | torch dtype string → `DType`         |
| `hub::build_from_pretrained` / `fetch_config` — `src/hub.rs` | Resolve hf-id → config → graph      |
| `viz::graph_to_html` — `src/viz.rs`; `plowc viz` — `crates/plowc/src/main.rs` | DAG viewer            |
| `tests/build_graphs.rs`, `tests/bind_and_weights.rs`    | Verification patterns to copy           |
| `models/llama.rs`, `models/qwen3.rs`, `models/deepseek.rs`, `models/gemma.rs`, `models/kimi_k3.rs`, `models/siglip.rs`, `models/qwen_image_*.rs` | Builder templates |

---

## Where the real process differs from the idealized pipeline

`docs/arch/01-compiler-pipeline.md` shows Stage 1 feeding an egglog rewriter
(Stage 2) and on to emission. In the shipping build that is not what happens:
the `plowc --emit devblob` path synthesizes a `LayerPlan` directly from
`config.json` and does not build an `nn_graph::Graph` at all; `devgen` does not
depend on this crate. So a builder you add here powers analysis and
visualization (and the advisory egglog fusion-count report), but does **not** by
itself make the model emit-able. Getting the model onto a GPU is later-stage
work. This stage's job is a correct, inspected, shape-inferred IR — the ground
truth every later stage checks itself against.
