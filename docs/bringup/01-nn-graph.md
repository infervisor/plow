# Stage 1 — NN-Graph IR & Devblob Emitter

Target-independent stage (CPU-only). Name no GPU, ISA, CU count, or vendor toolchain here; target hardware parameters are populated in [`target.md`](target.md) at Stage 4.

* **Input:** Hugging Face checkpoint directory or model ID containing `config.json`, `modeling_*.py`, and `model.safetensors.index.json`.
* **Output:** Shape-inferred `nn_graph::Graph` and fully wired `plowc --emit devblob` pipeline that simulates cleanly.
* **Architecture Reference:** [`docs/arch/01-compiler-pipeline.md`](../arch/01-compiler-pipeline.md) and [`docs/arch/model_bringup_and_stability_roadmap.md`](../arch/model_bringup_and_stability_roadmap.md) §1.1.

---

## 1. Silent Numerical Correctness Invariants

Before writing IR code, audit the reference implementation (`modeling_*.py`) against five critical numerical traps ([`model_bringup_and_stability_roadmap.md`](../arch/model_bringup_and_stability_roadmap.md) §1.1.2):

| Invariant Area | Upstream Divergence | Verification Contract in IR |
|---|---|---|
| **1. RMSNorm Epsilon** | $\sqrt{\frac{1}{d}\sum x^2 + \epsilon}$ vs $\sqrt{\frac{1}{d}\sum x^2} + \epsilon$ | Epsilon must be placed strictly inside the square root prior to reciprocal. Ensure epsilon precision (FP32) is preserved before downcasting to activation dtype. |
| **2. RoPE Interleaving** | Neox style ($[-x_{d/2..}, x_{..d/2}]$) vs GPT-J style ($[-x_1, x_0, -x_3, x_2]$) | Inspect HF RotaryEmbedding implementation. For partial RoPE ($D_{\text{rope}} < D_{\text{head}}$), assert pass-through slice $[D_{\text{rope}}..D_{\text{head}}]$ remains unrotated. |
| **3. Attention Scale** | Fused $1/\sqrt{d_k}$ in Q weights vs in-kernel scaling | Never absorb scale factor into weights if model uses QK-Norm or FP8 scales; unabsorbed factors cause silent underflow/overflow. |
| **4. Tied Embeddings** | Reused input embedding for LM head | Check if output projection applies $d_{\text{model}}^{-0.5}$ scaling, transposition, or un-tied bias vectors. |
| **5. Recurrent State** | Hybrid linear models (GDN / KDA / linear RNN) | State tensors $[B, H, K, V]$ require strict contiguous memory layout for chunked prefix scans. Strides must be verified at emit. |

---

## 2. Geometry Audit

Record these parameters from `config.json` before building the model:

| Parameter | Meaning & Downstream Decision |
|---|---|
| `layer_types` | Full-attention vs sliding-window layers; determines KV ring allocation in Stage 6. |
| `head_dim` | Distinct head dimensions require separate attention kernel instantiations. |
| `num_key_value_heads` | Number of KV heads determines GEMV launch geometry for decode. |
| `sliding_window` | Ring buffer size floor in the VRAM budget calculation. |
| `vocab_size` | Dimensions of LM head GEMV and sampler scratch allocation. |
| `rope_parameters` | Base frequency $\theta$, partial rotary dimension, and scaling factors. |
| MoE vs Dense | Number of routed experts, top-$k$, shared experts, and routing activation. |
| `num_hidden_layers` | Segment count; loader enforces physical segment limits. |

**GEMM Divisibility Rule:** For all projections ($Q, K, V, O$, MLP gate/up/down), verify that projection dimensions $N$ and $K$ divide the target GEMM tile (probed via `plowc tune inventory` or `hwspec::IsaLevel::geometry()`). Non-divisible dimensions require padding or fall back to slow unfused kernels.

---

## 3. Template Reference

| Architecture Pattern | Template Source (`crates/nn-graph/src/models/`) |
|---|---|
| Dense GQA + SwiGLU + RoPE (Standard Pre-Norm) | `models/llama.rs` |
| GQA + Per-Head QK-Norm + Explicit Head Dim | `models/qwen3.rs` |
| Multi-Head Latent Attention (MLA) + Sparse MoE | `models/deepseek.rs` |
| Hybrid Local Sliding-Window + Global Softcap | `models/gemma.rs` |
| Hybrid Linear Attention (GDN / KDA) | `models/kimi_k3.rs` |
| Vision Encoder / Multi-Modal Transformer | `models/siglip.rs` |
| Diffusion DiT / Resolution-Bucketed VAE | `models/qwen_image_*.rs` |

---

## 4. Step-by-Step Implementation

Execute all commands inside `nix develop`.

### Step 0: Ensure Clean Baseline
```bash
nix develop --command bash -c '
  cargo build -p nn-graph --no-default-features &&
  cargo build -p nn-graph &&
  cargo build -p nn-graph --features hub &&
  cargo test -p nn-graph'
```
Must pass with exit code 0.

### Step 1: Config Parsing (`crates/nn-graph/src/models/config/<arch>.rs`)
* Create `#[derive(Deserialize)]` struct containing only dimensions the graph builder requires.
* Apply `#[serde(default = "...")]` strictly to fields where the reference config specifies defaults; never apply blanket `#[serde(default)]` over required geometry.
* Provide helper methods `pub fn kv_heads(&self) -> usize` and `pub fn head_dim(&self) -> usize`.

### Step 2: Register Config in Dispatch (`crates/nn-graph/src/models/config/mod.rs`)
* Add `mod <arch>;` and `pub use <arch>::<Arch>Config;`.
* Add variant `ModelConfig::<Arch>(<Arch>Config)`.
* Wire `from_json` on `model_type` and add matching identifier in `architectures[]`. Reject unsupported variants with `ConfigError::Unsupported`.

### Step 3: Implement Graph Builder (`crates/nn-graph/src/models/<arch>.rs`)
* Expose `pub fn build(cfg: &<Arch>Config) -> Graph` using `Nn` (`src/builder.rs`).
* Use symbolic dimensions `"B"` and `"S"`.
* Wrap each layer body in `builder.begin_block("layers.{i}")` ... `builder.end_block()`.
* Operator ordering and weight tensor naming must match checkpoint keys bit-for-bit.
* Mark final outputs: logits for decoders (`mark_output`), hidden states for encoders.
* Wire builder in `models/mod.rs::build_graph` (and `build_encoder_graph` if applicable).

### Step 4: Operator Addition (Only If Unavoidable)
* Add new `Op` variant in `crates/nn-graph/src/op.rs` only when existing operators cannot express the computation without precision or performance degradation.
* Implement shape inference in `crates/nn-graph/src/infer.rs`.
* Provide builder helper on `Nn` and wire operator name in `Op::name`.

### Step 5: Unit Tests (`crates/nn-graph/tests/build_graphs.rs`)
* Add a test case with a scaled-down, structurally faithful inline configuration.
* Assert:
  ```rust
  assert_fully_inferred(&graph);
  assert_eq!(output_shape_str(&graph), "[B, S, vocab_size]");
  assert_eq!(graph.count_ops(Op::FlashAttention), expected_layer_count);
  ```
* Add a negative test verifying missing required fields trigger descriptive `ConfigError` failures.

### Step 6: Weight Manifest & Shape Binding (`crates/nn-graph/tests/bind_and_weights.rs`)
* Assert all tensor names and shapes from `graph.weight_manifest()` match the checkpoint's `model.safetensors.index.json`.
* Bind concrete runtime dimensions:
  ```rust
  let bound_graph = graph.bind(&Bindings::new().set("B", 1).set("S", 512));
  assert!(bound_graph.is_fully_bound());
  ```

### Step 7: Devblob Pipeline Wiring
* In `crates/plowc/src/hf_config.rs`: Parse model config, validate layer plan, and synthesize `LayerPlan`.
* In `crates/devgen`: Implement architecture dispatch to emit full-model and `--block` packets matching exact checkpoint weight layouts.

### Step 8: CPU Simulation Sanity Check
Confirm packet dependency DAG resolves without hardware:
```bash
nix develop --command ./scripts/block_sim.sh <scratch>/assets decode:1:128
```
Verifies makespan is finite, topological sort is acyclic, and memory buffers remain within bounds.

---

## 5. Gate Checklist

Before advancing to Stage 2:
- [ ] `cargo test -p nn-graph` passes with 0 failures and 0 warnings.
- [ ] Graph builds cleanly from reference `config.json` with `assert_fully_inferred`.
- [ ] Weight manifest names and shapes match `model.safetensors.index.json` exactly.
- [ ] All 5 numerical correctness invariants (§1) are verified against upstream reference code.
- [ ] `plowc --emit devblob` produces valid full-model and single-block packets.
- [ ] `./scripts/block_sim.sh` executes clean CPU simulation with finite makespan.
