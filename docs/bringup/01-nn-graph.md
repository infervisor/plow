# Stage 1 — NN-Graph IR & Devblob Emitter Wiring

Target-independent stage. Maps a model configuration to a shape-inferred `nn_graph::Graph` and wires it through the production `--emit devblob` pipeline.

* **Inputs:** Model identifier (e.g. `Qwen/Qwen3-8B`) or local checkpoint directory containing `config.json`, `modeling_*.py`, and `model.safetensors.index.json`.
* **Output:** Shape-inferred operator DAG in `crates/nn-graph` and functional `--emit devblob` lowering in `crates/devgen` / `crates/plowc`.

---

## 1. Geometry Audit

Record these parameters from `config.json` before implementing:

| Parameter | Downstream Impact |
|---|---|
| `layer_types` (full vs sliding) | Determines attention kernel selection; sliding layers allocate KV ring buffer. |
| `head_dim` per family | Controls kernel instantiation per head dimension. |
| `num_key_value_heads` per family | Fixes GEMV and flash attention reduction shapes. |
| `sliding_window` | Sets floor for KV ring allocation in Stage 6 memory budget. |
| `vocab_size` | Dictates `lm_head` projection shape and sampler scratch space. |
| `rope_parameters` | Configures partial RoPE dimension and per-layer theta values. |
| `num_experts` / `num_experts_per_tok` | Selects router, expert dispatch, and combine kernel families. |
| `num_hidden_layers` | Limits total execution segments. |

*GEMM divisibility:* Verify projection dimensions $N$ and $K$ divide hardware tile boundaries (`hwspec::IsaLevel::geometry()` for AMD; object metadata for NVIDIA).

---

## 2. Reference Implementation Templates

Locate starting architectures under `crates/nn-graph/src/models/`:

| Architecture Family | Reference File |
|---|---|
| Dense GQA, SwiGLU, RoPE, RMSNorm | `models/llama.rs` |
| QK-Norm, explicit head dimension | `models/qwen3.rs` |
| Multi-head Latent Attention (MLA) + MoE | `models/deepseek.rs` |
| Local/global hybrid attention, softcapping | `models/gemma.rs` |
| Hybrid linear/softmax attention | `models/kimi_k3.rs` |
| Vision encoder (ViT) | `models/siglip.rs` |
| Diffusion DiT / VAE | `models/qwen_image_*.rs` |

---

## 3. Implementation Steps

All steps run in `nix develop`.

### Step 0: Baseline Verification
```bash
nix develop --command bash -c '
  cargo build -p nn-graph --no-default-features &&
  cargo build -p nn-graph &&
  cargo test -p nn-graph'
```

### Step 1: Configuration Deserialization
Add `crates/nn-graph/src/models/config/<arch>.rs`:
* Define `#[derive(Deserialize)]` struct extracting only required dimensions.
* Use `#[serde(alias = "...")]` for varying field names across checkpoints.
* Never use blanket `#[serde(default)]` over required structural dimensions.
* Expose `kv_heads()` and `head_dim()` accessors.

### Step 2: Architecture Registration
In `crates/nn-graph/src/models/config/mod.rs`:
* Register module and add variant to `ModelConfig::<Arch>`.
* Map `model_type` and `architectures[]` in `from_json`. Reject unsupported variants with `ConfigError::Unsupported`.

### Step 3: Graph Builder
Add `crates/nn-graph/src/models/<arch>.rs`:
* Implement `pub fn build(cfg: &<Arch>Config) -> Graph` using `Nn` builder (`src/builder.rs`).
* Maintain symbolic batch (`B`) and sequence (`S`) dimensions.
* Wrap layers in `begin_block("layers.{i}")` ... `end_block()`.
* Match checkpoint tensor names and reference operator order exactly.
* Mark output logits via `mark_output`.

### Step 4: Graph Validation Tests
Add test case to `crates/nn-graph/tests/build_graphs.rs`:
* Construct structural config fixture.
* Validate with `assert_fully_inferred(&g)`.
* Assert expected operator counts via `count_ops`.
* Add negative test confirming incomplete configurations are rejected.

### Step 5: Devblob Emitter Wiring
1. **Config parsing:** Extend `crates/plowc/src/hf_config.rs` to validate configuration and build the full `LayerPlan`.
2. **Emitter lowering:** Extend `crates/devgen` to emit full-model and `--block` packets matching checkpoint tensor names.
3. **Compilation test:**
   ```bash
   nix develop --command cargo test -p plowc -p devgen
   nix develop --command plowc --hf-dir <checkpoint> --emit devblob --out <scratch>/emit
   nix develop --command plowrt simulate --assets <scratch>/emit --all-buckets
   ```

---

## 4. Graph Inspection Tools

```bash
# Export static HTML visualization of the operator graph
nix develop --command plowc --hf-dir <dir> viz --port 0 --out <scratch>/graph.html

# Validate graph weight names against safetensors index
nix develop --command plowc --model <hf-id> viz --validate-checkpoint --port 0 --dump-assets <scratch>/viz

# Inspect emitted packet structure
nix develop --command plowc viz --packet <assets>/model.pkt --open false
```

---

## 5. Gate Checklist

Before advancing to Stage 2:
- [ ] `cargo test -p nn-graph` passes with default and `--no-default-features`.
- [ ] `assert_fully_inferred` passes; `weight_manifest()` matches checkpoint tensor names and shapes.
- [ ] Operator sequences, activations, norms, and RoPE rotations match ground truth reference.
- [ ] `plowc --emit devblob` succeeds for both full model and `--block`.
- [ ] `plowrt simulate --assets <dir> --all-buckets` executes to completion without deadlock.

---

## 6. Common Pitfalls

* **Default deserialization:** Blanket `#[serde(default)]` silently defaults missing fields, constructing invalid topologies.
* **RoPE variants:** Standard RoPE pairs adjacent elements `(x[i], x[i+d/2])`; interleaved RoPE pairs `(x[2i], x[2i+1])`.
* **Attention output dimensions:** In MLA architectures, projection output width reflects value head dimension (`v_head_dim`), not query/key head dimension.
* **Layer heterogeneity:** Local vs. global attention and dense vs. MoE layers must be driven by the configuration's explicit layer list, never hardcoded strides.
