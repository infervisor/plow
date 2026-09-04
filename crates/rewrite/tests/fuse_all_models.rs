//! Integration tests: build every model, lower to egglog, run Stage-2 fusion,
//! and verify the pipeline completes and fusion rules fire where expected.

use nn_graph::models::{build_from_config_json, build_from_config_json_at, ShapeBucket};
use nn_graph::Origin;
use std::collections::BTreeSet;

/// Weight-leaf names referenced by the fused graph.
fn fused_weights(f: &rewrite::FusedGraph) -> BTreeSet<String> {
    f.nodes
        .iter()
        .filter(|n| n.op == "Weight")
        .filter_map(|n| match n.args.first() {
            Some(rewrite::Arg::Str(s)) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

/// Weight-leaf names declared by the input graph.
fn graph_weights(g: &nn_graph::Graph) -> BTreeSet<String> {
    g.tensors
        .iter()
        .filter(|t| matches!(t.origin, Origin::Weight))
        .filter_map(|t| t.name.clone())
        .collect()
}

/// Weight leaves consumed by the compute DAG. Routed-expert parameters live in
/// the compact expert manifest instead of cloning every expert into the DAG.
fn graph_dag_weights(g: &nn_graph::Graph) -> BTreeSet<String> {
    g.nodes
        .iter()
        .flat_map(|node| node.inputs.iter())
        .filter_map(|id| {
            let tensor = g.tensor(*id);
            matches!(tensor.origin, Origin::Weight)
                .then(|| tensor.name.clone())
                .flatten()
        })
        .collect()
}

fn assert_expert_bindings_in_checkpoint(g: &nn_graph::Graph) {
    let mut bound = BTreeSet::new();
    for layer in &g.expert_bindings {
        assert_eq!(layer.routed_experts.len(), layer.num_experts as usize);
        for expert in &layer.routed_experts {
            for projection in [&expert.gate, &expert.up, &expert.down] {
                assert!(bound.insert(projection.weight.clone()));
                if let Some(scale) = &projection.scale {
                    assert!(bound.insert(scale.clone()));
                }
            }
        }
    }

    let checkpoint_experts = g
        .checkpoint_manifest()
        .into_iter()
        .filter(|weight| weight.name.contains(".mlp.experts."))
        .map(|weight| weight.name.to_string())
        .collect();
    assert_eq!(bound, checkpoint_experts, "expert manifest is incomplete");
}

// --- Gemma (existing test, kept for parity) ---

const GEMMA: &str = r#"{
    "model_type": "gemma3",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "sliding_window_pattern": 2
}"#;

#[test]
fn fuse_gemma() {
    let g = build_from_config_json(GEMMA).expect("build gemma");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire"
    );
    assert!(fused.contains("SwiGLU"), "act·mul fusion did not fire");
    // Embedding * sqrt(hidden) → one node.
    assert!(
        fused.contains("FusedEmbeddingScale"),
        "embedding+scale fusion did not fire"
    );
    // Residual-add + RMSNorm at block boundaries.
    assert!(
        fused.contains("FusedResidualNorm"),
        "residual+norm fusion did not fire"
    );
    assert!(
        stats.fused >= 6,
        "expected >= 6 fused nodes, got {}",
        stats.fused
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
}

// --- Gemma 4 (shared K/V, heterogeneous head dims, per-layer RoPE) ---

const GEMMA4: &str = r#"{
    "model_type": "gemma4_unified_text",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_global_key_value_heads": 4,
    "global_head_dim": 128,
    "attention_k_eq_v": true,
    "use_qk_norm": true,
    "query_pre_attn_scalar": 128.0,
    "sliding_window": 512,
    "layer_types": ["sliding_attention", "full_attention"],
    "rope_parameters": {
        "full_attention": {
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.5
        },
        "sliding_attention": {
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0
        }
    }
}"#;

#[test]
fn fuse_gemma4() {
    let g = build_from_config_json(GEMMA4).expect("build gemma4");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // Norm→Linear fusions fire on q_proj, kv_proj, gate_proj, up_proj (per block) + lm_head.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire"
    );
    // GeGLU MLP → SwiGLU.
    assert!(fused.contains("SwiGLU"), "act·mul fusion did not fire");
    // Norm→Rope fusions fire on qk_norm paths (use_qk_norm=true).
    assert!(
        fused.contains("FusedNormRope"),
        "norm→rope fusion did not fire (qk_norm paths)"
    );

    // Gemma 4 uses an attention scale of 1.0, so Q and K both use FusedNormRope.
    assert!(
        stats.fused >= 13,
        "expected >= 13 fused nodes, got {}",
        stats.fused
    );

    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );

    // Weight-manifest completeness: no weight dropped or duplicated by fusion.
    assert_eq!(
        fused_weights(&fused),
        graph_weights(&g),
        "fusion dropped or duplicated weight leaves"
    );
}

// --- Gemma 4 MoE (gemma-4-26B-A4B style: sparse MoE layers) ---

const GEMMA4_MOE: &str = r#"{
    "model_type": "gemma4_unified_text",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_global_key_value_heads": 4,
    "global_head_dim": 128,
    "attention_k_eq_v": true,
    "use_qk_norm": true,
    "query_pre_attn_scalar": 128.0,
    "sliding_window": 512,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
    "layer_types": ["moe_sliding_attention", "moe_full_attention"],
    "rope_parameters": {
        "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.5},
        "sliding_attention": {"rope_theta": 10000.0}
    }
}"#;

#[test]
fn gemma4_moe_fails_closed_before_rewrite() {
    let error = build_from_config_json(GEMMA4_MOE)
        .expect_err("Gemma4 MoE must not use a dense or representative-expert fallback")
        .to_string();
    assert!(error.contains("Gemma 4 MoE"), "{error}");
    assert!(
        error.contains("exhaustive expert checkpoint binding"),
        "{error}"
    );
    assert!(error.contains("refusing"), "{error}");
}

// --- DeepSeek (MLA + MoE) ---

const DEEPSEEK: &str = r#"{
    "model_type": "deepseek_v3",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_rope_head_dim": 16,
    "qk_nope_head_dim": 48,
    "v_head_dim": 64,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "first_k_dense_replace": 2,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 2,
    "topk_group": 1,
    "norm_topk_prob": true,
    "routed_scaling_factor": 2.5
}"#;

#[test]
fn fuse_deepseek() {
    let g = build_from_config_json(DEEPSEEK).expect("build deepseek");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // DeepSeek uses RmsNorm → Linear pattern at q_a_proj, q_b_proj, kv_b_proj, etc.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire"
    );
    // SwiGLU from silu(gate)*up in the MLP.
    assert!(fused.contains("SwiGLU"), "act·mul fusion did not fire");
    // Residual-add + RMSNorm at block boundaries.
    assert!(
        fused.contains("FusedResidualNorm"),
        "residual+norm fusion did not fire in DeepSeek"
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
    // Compute-DAG leaves survive fusion; compact expert tensors survive through
    // the exhaustive checkpoint binding manifest.
    let fw = fused_weights(&fused);
    let gw = graph_dag_weights(&g);
    assert_eq!(
        fw, gw,
        "fusion dropped or duplicated weight leaves in DeepSeek"
    );
    assert_expert_bindings_in_checkpoint(&g);
}

// --- SigLIP (LayerNorm + Conv2d + Transpose + Reduce) ---

const SIGLIP: &str = r#"{
    "model_type": "siglip",
    "hidden_size": 192,
    "intermediate_size": 384,
    "num_hidden_layers": 2,
    "num_attention_heads": 3,
    "num_channels": 3,
    "image_size": 56,
    "patch_size": 14,
    "layer_norm_eps": 1e-6,
    "hidden_act": "gelu_pytorch_tanh"
}"#;

#[test]
fn fuse_siglip() {
    let g = build_from_config_json(SIGLIP).expect("build siglip");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // SigLIP: LayerNorm → biased Linear fires on Q/K/V projections (where the
    // norm's input is not from an add). When the norm reads from an add,
    // extraction may prefer FusedResidualLayerNorm instead.
    assert!(
        fused.contains("FusedLayerNormLinearBias") || fused.contains("FusedResidualLayerNorm"),
        "norm→linear or residual+norm fusion did not fire in SigLIP"
    );
    // Linear+Act fusion fires on the MLP up-projection (GeluTanh).
    assert!(
        fused.contains("FusedLinearBiasAct"),
        "linear+act fusion did not fire in SigLIP MLP"
    );
    // The bias-preserving fusion drops NO weight (norm bias + linear bias kept).
    assert_eq!(
        fused_weights(&fused),
        graph_weights(&g),
        "fusion dropped weight leaves"
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
}

// --- Qwen2.5-VL vision encoder (Slice + Concat + RmsNorm→Linear) ---

const QWEN_VL: &str = r#"{
    "model_type": "qwen2_5_vl",
    "vision_config": {
        "depth": 2,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 14,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 256
    }
}"#;

#[test]
fn fuse_qwen_vl() {
    let g = build_from_config_json(QWEN_VL).expect("build qwen_vl");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // QwenVL: RmsNorm → biased Linear fires for Q/K/V projections after norm1.
    assert!(
        fused.contains("FusedNormLinearBias"),
        "rmsnorm→linear fusion did not fire in QwenVL"
    );
    // Gated SiLU MLP → SwiGLU.
    assert!(
        fused.contains("SwiGLU"),
        "act·mul fusion did not fire in QwenVL"
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
}

// --- Qwen-Image DiT (LayerNorm + AdaLN modulate + gated residual) ---

const QWEN_DIT: &str = r#"{
    "_class_name": "QwenImageTransformer2DModel",
    "num_layers": 2,
    "num_attention_heads": 4,
    "attention_head_dim": 32,
    "in_channels": 16,
    "out_channels": 16,
    "patch_size": 2,
    "joint_attention_dim": 128,
    "pooled_projection_dim": 128,
    "axes_dims_rope": [4, 14, 14]
}"#;

#[test]
fn fuse_qwen_dit() {
    let bucket = ShapeBucket::square(64);
    let g = build_from_config_json_at(QWEN_DIT, &bucket).expect("build qwen_dit");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // DiT: txt_norm (RMSNorm) → txt_in (biased Linear) fuses; the LayerNorm img/txt
    // norms feed AdaLN modulate, not a Linear.
    assert!(
        fused.contains("FusedNormLinearBias"),
        "norm→linear fusion did not fire in DiT"
    );
    // Gated residual: x + gate * y.
    assert!(
        fused.contains("FusedGatedResidual"),
        "gated residual fusion did not fire in DiT"
    );
    // Linear+Act fires on the MLP up-projections (GeluTanh).
    assert!(
        fused.contains("FusedLinearBiasAct"),
        "linear+act fusion did not fire in DiT MLP"
    );
    // Norm→Rope fusions fire on the QK norms in joint attention.
    assert!(
        fused.contains("FusedNormRope"),
        "norm→rope fusion did not fire in DiT attention"
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
}

// --- Qwen-Image VAE (GroupNorm + Act fusion) ---

const QWEN_VAE: &str = r#"{
    "_class_name": "AutoencoderKLQwenImage",
    "base_dim": 32,
    "dim_mult": [1, 2, 4],
    "num_res_blocks": 1,
    "z_dim": 4,
    "temperal_downsample": [false, true]
}"#;

#[test]
fn fuse_qwen_vae() {
    let bucket = ShapeBucket::square(64);
    let g = build_from_config_json_at(QWEN_VAE, &bucket).expect("build qwen_vae");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // VAE resnet blocks: GroupNorm → SiLU → Conv3d. The full three-way fuses.
    assert!(
        fused.contains("FusedGroupNormActConv3dBias"),
        "groupnorm+act+conv3d fusion did not fire in VAE"
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
}

// --- Llama / Mistral (standard pre-norm decoder) ---

const LLAMA: &str = r#"{
    "model_type": "llama",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0
}"#;

#[test]
fn fuse_llama() {
    let g = build_from_config_json(LLAMA).expect("build llama");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // Norm→Linear fusions fire on q/k/v/gate/up projections + lm_head.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire in Llama"
    );
    // SwiGLU from silu(gate)*up in each MLP block.
    assert!(
        fused.contains("SwiGLU"),
        "act·mul fusion did not fire in Llama"
    );
    // Residual-add + RMSNorm at block boundaries.
    assert!(
        fused.contains("FusedResidualNorm"),
        "residual+norm fusion did not fire in Llama"
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
    // Weight-manifest completeness.
    assert_eq!(
        fused_weights(&fused),
        graph_weights(&g),
        "fusion dropped or duplicated weight leaves in Llama"
    );
}

// --- GLM-5.2 MoE-DSA (MLA + MoE + Dense-Sparse Attention) ---

const GLM5: &str = r#"{
    "model_type": "glm_moe_dsa",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 64,
    "rms_norm_eps": 1e-5,
    "attention_bias": false,
    "hidden_act": "silu",
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_head_dim": 64,
    "qk_nope_head_dim": 48,
    "qk_rope_head_dim": 16,
    "v_head_dim": 64,
    "rope_interleave": true,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "first_k_dense_replace": 2,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "mlp_layer_types": ["dense", "dense", "sparse", "sparse"],
    "scoring_func": "sigmoid",
    "routed_scaling_factor": 2.5,
    "norm_topk_prob": true,
    "n_group": 2,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "moe_router_dtype": "float32",
    "indexer_types": ["full", "full", "full", "shared"],
    "index_head_dim": 32,
    "index_n_heads": 4,
    "index_topk": 64,
    "index_topk_freq": 1,
    "indexer_rope_interleave": true,
    "index_skip_topk_offset": 2,
    "num_nextn_predict_layers": 1,
    "index_share_for_mtp_iteration": true,
    "torch_dtype": "bfloat16",
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128]
    }
}"#;

#[test]
fn fuse_glm() {
    let g = build_from_config_json(GLM5).expect("build glm5");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // MLA: RmsNorm → Linear pattern at q_a_proj, kv_a_proj_with_mqa, kv_b_proj, etc.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire in GLM-5.2"
    );
    // SwiGLU from silu(gate)*up in dense MLP and MoE expert MLPs.
    assert!(
        fused.contains("SwiGLU"),
        "act·mul fusion did not fire in GLM-5.2"
    );
    // Residual-add + RMSNorm at block boundaries.
    assert!(
        fused.contains("FusedResidualNorm"),
        "residual+norm fusion did not fire in GLM-5.2"
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
    // Weight-manifest completeness. FP8 scales and routed expert tensors are
    // carried by packet metadata rather than the extracted compute dataflow.
    // The base-model final norm is dead when the fixture's MTP output is used.
    let fw = fused_weights(&fused);
    assert!(
        graph_weights(&g).contains("model.norm.weight"),
        "the input graph must still declare the base-head final norm"
    );
    assert!(
        !fw.contains("model.norm.weight"),
        "rewrite extracts the last (MTP) output, where the base-head final norm is dead"
    );
    let gw: BTreeSet<String> = graph_weights(&g)
        .into_iter()
        .filter(|w| {
            !w.ends_with(".weight_scale_inv")
                && !w.contains(".mlp.experts.")
                && w != "model.norm.weight"
        })
        .collect();
    assert_eq!(
        fw, gw,
        "fusion dropped or duplicated weight leaves in GLM-5.2"
    );
    assert_expert_bindings_in_checkpoint(&g);
}

// --- Kimi K2 (Moonshot): MLA + MoE ---

const KIMI: &str = r#"{
    "model_type": "kimi",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_rope_head_dim": 16,
    "qk_nope_head_dim": 48,
    "v_head_dim": 64,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "first_k_dense_replace": 2,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 2,
    "topk_group": 1,
    "norm_topk_prob": true,
    "routed_scaling_factor": 2.827
}"#;

#[test]
fn fuse_kimi() {
    let g = build_from_config_json(KIMI).expect("build kimi");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // MLA: RmsNorm → Linear pattern at q_a_proj, kv_a_proj_with_mqa, kv_b_proj.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire in Kimi"
    );
    // SwiGLU from silu(gate)*up in dense MLP and MoE expert MLPs.
    assert!(
        fused.contains("SwiGLU"),
        "act·mul fusion did not fire in Kimi"
    );
    // Residual-add + RMSNorm at block boundaries.
    assert!(
        fused.contains("FusedResidualNorm"),
        "residual+norm fusion did not fire in Kimi"
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
    // Compute-DAG leaves survive fusion; compact expert tensors survive through
    // the exhaustive checkpoint binding manifest.
    let fw = fused_weights(&fused);
    let gw = graph_dag_weights(&g);
    assert_eq!(fw, gw, "fusion dropped or duplicated weight leaves in Kimi");
    assert_expert_bindings_in_checkpoint(&g);
}

// --- Kimi K3 (Moonshot): hybrid KDA/MLA + latent MoE ---
//
// K3 could not be lowered at all before the four K3 terms existed: `lower.rs`
// REFUSED `Conv1dDepthwise`/`LinearAttention`/`SituGlu`/`BlockResidual` rather
// than dropping them to `Opaque`, which is a single-input passthrough and would
// have dropped every weight leaf but the first. So this test is first of all a
// test that the graph lowers.

const KIMI_K3: &str = r#"{
    "model_type": "kimi_k3",
    "text_config": {
      "model_type": "kimi_linear",
      "vocab_size": 1000, "hidden_size": 256, "intermediate_size": 512,
      "num_hidden_layers": 4, "num_attention_heads": 4,
      "q_lora_rank": 64, "kv_lora_rank": 32,
      "qk_rope_head_dim": 16, "qk_nope_head_dim": 32, "v_head_dim": 32,
      "mla_use_output_gate": true,
      "num_experts": 8, "num_experts_per_token": 2, "num_shared_experts": 1,
      "moe_intermediate_size": 128, "routed_expert_hidden_size": 192,
      "first_k_dense_replace": 1,
      "attn_res_block_size": 2,
      "linear_attn_config": {
        "num_heads": 4, "head_dim": 32, "short_conv_kernel_size": 4,
        "use_full_rank_gate": true,
        "full_attn_layers": [1, 3], "kda_layers": [2, 4]
      }
    }
}"#;

#[test]
fn fuse_kimi_k3() {
    let g = build_from_config_json(KIMI_K3).expect("build kimi_k3");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // The two K3 gate fusions. Both targets are opcodes that already exist and
    // already pass a real-weight numeric gate (KdaGatedNorm = 103,
    // MlaOutGate = 106), so reaching them costs no new kernel work.
    assert!(
        fused.contains("FusedKdaGatedNorm"),
        "KDA gated-norm fusion did not fire in K3"
    );
    assert!(
        fused.contains("FusedMlaOutGate"),
        "MLA output-gate fusion did not fire in K3"
    );
    assert!(
        fused.contains("FusedMaterializedResidualBlock"),
        "cross-block residual materialization fusion did not fire"
    );
    // The generic fusions must still fire on a hybrid graph.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire in K3"
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );

    // Weight-manifest completeness — the reason `BlockResidual` lowers to a cons
    // chain rather than a fixed-arity term. A dropped snapshot or score weight
    // shows up here and nowhere else.
    let fw = fused_weights(&fused);
    let gw: BTreeSet<String> = graph_weights(&g)
        .into_iter()
        .filter(|w| !w.ends_with(".mlp.gate.weight"))
        .collect();
    assert_eq!(fw, gw, "fusion dropped or duplicated weight leaves in K3");
}

#[test]
fn grouped_router_remains_explicit_in_the_complete_rewrite_graph() {
    let cfg = KIMI_K3.replacen(
        "\"num_experts\": 8",
        "\"num_experts\": 8, \"n_group\": 2, \"topk_group\": 1",
        1,
    );
    let g = build_from_config_json(&cfg).expect("build grouped-router graph");
    let (_, stats) = rewrite::rewrite_graph(&g).expect("rewrite complete grouped-router graph");
    assert_eq!(stats.ops_before, g.nodes.len());
}

// --- Qwen3 / Qwen2.5: GQA + SwiGLU (dense) ---

const QWEN3: &str = r#"{
    "model_type": "qwen3",
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": false
}"#;

#[test]
fn fuse_qwen3() {
    let g = build_from_config_json(QWEN3).expect("build qwen3");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // Norm→Linear fusions fire on q/k/v/gate/up projections + lm_head.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire in Qwen3"
    );
    // SwiGLU from silu(gate)*up in each MLP block.
    assert!(
        fused.contains("SwiGLU"),
        "act·mul fusion did not fire in Qwen3"
    );
    // Residual-add + RMSNorm at block boundaries.
    assert!(
        fused.contains("FusedResidualNorm"),
        "residual+norm fusion did not fire in Qwen3"
    );
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
    // Weight-manifest completeness.
    assert_eq!(
        fused_weights(&fused),
        graph_weights(&g),
        "fusion dropped or duplicated weight leaves in Qwen3"
    );
}

// --- Qwen3.5/3.8: hybrid Gated DeltaNet + gated full attention ---

const QWEN35: &str = r#"{
    "model_type": "qwen3_5",
    "text_config": {
        "model_type": "qwen3_5_text",
        "vocab_size": 1000,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 32,
        "layer_types": ["linear_attention", "linear_attention",
                        "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 16,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 6,
        "linear_value_head_dim": 16,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "rope_type": "default",
            "mrope_interleaved": true
        },
        "attention_bias": false,
        "attn_output_gate": true,
        "hidden_act": "silu",
        "mamba_ssm_dtype": "float32",
        "output_gate_type": "swish",
        "tie_word_embeddings": false,
        "dtype": "bfloat16"
    }
}"#;

#[test]
fn fuse_qwen35() {
    let g = build_from_config_json(QWEN35).expect("build qwen3.5 hybrid");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite qwen3.5 hybrid");

    for op in [
        "FusedZeroCenteredNormLinear",
        "FusedZeroCenteredNormRope",
        "FusedResidualZeroCenteredNorm",
        "FusedRmsNormSiluGate",
        "FusedPackedAttnGate",
        "SwiGLU",
    ] {
        assert!(fused.contains(op), "{op} did not fire in Qwen3.5/3.8");
    }
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce Qwen3.5 ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
    assert_eq!(
        fused_weights(&fused),
        graph_weights(&g),
        "Qwen3.5 fusion dropped or duplicated weight leaves"
    );
}
