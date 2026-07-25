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
        fused.contains("FusedNormRope") || fused.contains("FusedNormRopeScale"),
        "norm→rope fusion did not fire (qk_norm paths)"
    );
    // Q path: norm→rope→scale fuses to FusedNormRopeScale (1 per block = 2 total).
    assert!(
        fused.contains("FusedNormRopeScale"),
        "norm→rope→scale fusion did not fire on Q path"
    );

    // With attention_k_eq_v: 4 FusedNormLinear/block + 1 SwiGLU + 1 FusedNormRope(K)
    // + 1 FusedNormRopeScale(Q) = 7 per block × 2 blocks + 1 tail lm_head = 15.
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
fn fuse_gemma4_moe() {
    let g = build_from_config_json(GEMMA4_MOE).expect("build gemma4 moe");
    let (fused, stats) = rewrite::rewrite_graph(&g).expect("rewrite");

    // Norm→Linear fusions still fire on q_proj, kv_proj (per block) + lm_head.
    assert!(
        fused.contains("FusedNormLinear"),
        "rmsnorm→linear fusion did not fire in MoE variant"
    );
    // MoE experts still use GeGLU ⇒ SwiGLU fires on expert FFN.
    assert!(
        fused.contains("SwiGLU"),
        "act·mul fusion did not fire in MoE expert"
    );
    // Norm→Rope fusions on qk_norm paths.
    assert!(
        fused.contains("FusedNormRopeScale"),
        "norm→rope→scale fusion did not fire on Q path"
    );
    // Fusion reduced ops.
    assert!(
        stats.ops_after < stats.ops_before,
        "fusion did not reduce ops: {} -> {}",
        stats.ops_before,
        stats.ops_after
    );
    // Weight-manifest completeness (excluding MoE router weights: the router's
    // output is not on the extracted dataflow path — it's scheduling metadata).
    let fw = fused_weights(&fused);
    let gw: BTreeSet<String> = graph_weights(&g)
        .into_iter()
        .filter(|w| !w.ends_with(".mlp.gate.weight"))
        .collect();
    assert_eq!(
        fw, gw,
        "fusion dropped or duplicated weight leaves in Gemma4 MoE"
    );
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
    "first_k_dense_replace": 2
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
    // Weight-manifest completeness (excluding MoE router weights: the router's
    // output is not on the extracted dataflow path — it's scheduling metadata).
    let fw = fused_weights(&fused);
    let gw: BTreeSet<String> = graph_weights(&g)
        .into_iter()
        .filter(|w| !w.ends_with(".mlp.gate.weight"))
        .collect();
    assert_eq!(
        fw, gw,
        "fusion dropped or duplicated weight leaves in DeepSeek"
    );
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
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_head_dim": 64,
    "qk_nope_head_dim": 48,
    "qk_rope_head_dim": 16,
    "v_head_dim": 64,
    "rope_interleave": true,
    "first_k_dense_replace": 2,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 256,
    "indexer_types": ["full", "full", "full", "shared"],
    "index_head_dim": 32,
    "index_n_heads": 4,
    "index_topk": 64,
    "index_skip_topk_offset": 2,
    "num_nextn_predict_layers": 1
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
    // Weight-manifest completeness. Excluded leaves:
    // - MoE router weights (`.mlp.gate.weight`) — scheduling metadata, not dataflow.
    // - DSA index_head weights — dead-end side projection (`let _index_score = ...`).
    // - `lm_head.weight` — merged with `mtp_heads.0.lm_head.weight` during extraction
    //   (both read from the same norm output; the extractor deduplicates them).
    let fw = fused_weights(&fused);
    let gw: BTreeSet<String> = graph_weights(&g)
        .into_iter()
        .filter(|w| {
            !w.ends_with(".mlp.gate.weight")
                && !w.contains("index_head")
                && w != "lm_head.weight"
        })
        .collect();
    assert_eq!(
        fw, gw,
        "fusion dropped or duplicated weight leaves in GLM-5.2"
    );
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
    "first_k_dense_replace": 2
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
    // Weight-manifest completeness. Exclude MoE router weights (scheduling metadata).
    let fw = fused_weights(&fused);
    let gw: BTreeSet<String> = graph_weights(&g)
        .into_iter()
        .filter(|w| !w.ends_with(".mlp.gate.weight"))
        .collect();
    assert_eq!(
        fw, gw,
        "fusion dropped or duplicated weight leaves in Kimi"
    );
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
