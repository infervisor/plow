//! Graph visualization export — JSON, SVG, self-contained HTML viewer, and asset dumper.
//!
//! Produces rich structural and architectural metadata for single networks and
//! multi-stage pipelines (ASR, LLM, TTS, Multimodal), with first-class support
//! for Cytoscape.js canvas rendering, offline asset dumping, and embeddable iframe widgets.

pub mod dump;
pub mod html;
pub mod svg;
pub mod types;

pub use dump::dump_assets;
pub use html::render_html;
pub use svg::bundle_to_svg;
pub use types::{
    ModelModality, OpCategory, PipelineStage, VizBlock, VizGraph, VizModelBundle, VizNode,
    VizOrigin, VizSubNetwork, VizSummary, VizTensor,
};

use crate::graph::{Graph, Origin};
use crate::op::Op;
use std::collections::HashMap;

/// Convert an [`Op`] to its classification category.
pub fn op_to_category(op: &Op) -> OpCategory {
    match op {
        Op::Linear { .. } | Op::MatMul | Op::Scale(_) => OpCategory::Compute,
        Op::Attention { .. }
        | Op::LinearAttention { .. }
        | Op::DsaAttention { .. }
        | Op::Rope { .. }
        | Op::Softmax { .. } => OpCategory::Attention,
        Op::Conv1d { .. }
        | Op::Conv1dDepthwise { .. }
        | Op::ConvTranspose1d { .. }
        | Op::Conv2d { .. }
        | Op::Conv3d { .. } => OpCategory::Convolution,
        Op::RmsNorm { .. }
        | Op::RmsNormZeroCentered { .. }
        | Op::RmsNormInputScaled { .. }
        | Op::LayerNorm { .. }
        | Op::GroupNorm { .. } => OpCategory::Normalization,
        Op::Act(_) | Op::SituGlu { .. } | Op::ResidualGate { .. } | Op::Elementwise(_) => {
            OpCategory::Activation
        }
        Op::MoeRouter { .. } | Op::MoeExperts { .. } | Op::DsaIndexer { .. } => OpCategory::Routing,
        Op::Reshape { .. }
        | Op::Transpose { .. }
        | Op::Broadcast { .. }
        | Op::Concat { .. }
        | Op::Slice { .. }
        | Op::Reduce { .. } => OpCategory::Structural,
        Op::Embedding | Op::BlockResidual { .. } => OpCategory::AudioCodec,
    }
}

/// Extract op-specific parameters as a JSON value for inspection.
pub fn op_detail(op: &Op) -> serde_json::Value {
    use serde_json::json;
    match op {
        Op::Linear { out_features, bias } => json!({
            "out_features": out_features,
            "bias": bias,
        }),
        Op::MatMul => json!({}),
        Op::RmsNorm { eps } => json!({ "eps": eps }),
        Op::RmsNormZeroCentered { eps } => json!({ "eps": eps }),
        Op::RmsNormInputScaled { eps } => json!({ "eps": eps }),
        Op::LayerNorm { eps } => json!({ "eps": eps }),
        Op::Rope {
            dim,
            theta,
            interleave,
            frequency_dim,
        } => json!({
            "dim": dim, "theta": theta, "interleave": interleave,
            "frequency_dim": frequency_dim,
        }),
        Op::Act(kind) => json!({ "activation": format!("{kind:?}") }),
        Op::Elementwise(kind) => json!({ "kind": format!("{kind:?}") }),
        Op::Scale(s) => json!({ "scale": s }),
        Op::Softmax { axis } => json!({ "axis": axis }),
        Op::Attention {
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            sliding_window,
            logit_softcap,
        } => json!({
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "causal": causal,
            "sliding_window": sliding_window,
            "logit_softcap": logit_softcap,
        }),
        Op::Embedding => json!({}),
        Op::Conv2d { stride, padding } => json!({
            "stride": [stride.0, stride.1],
            "padding": [padding.0, padding.1],
        }),
        Op::Conv3d { stride, padding } => json!({
            "stride": [stride.0, stride.1, stride.2],
            "padding": [padding.0, padding.1, padding.2],
        }),
        Op::GroupNorm { groups, eps } => json!({ "groups": groups, "eps": eps }),
        Op::Reshape { shape } => json!({ "shape": format!("{shape}") }),
        Op::Transpose { perm } => json!({ "perm": perm }),
        Op::Broadcast { shape } => json!({ "shape": format!("{shape}") }),
        Op::Concat { axis } => json!({ "axis": axis }),
        Op::Slice { axis, start, len } => json!({
            "axis": axis, "start": format!("{start}"), "len": format!("{len}"),
        }),
        Op::Reduce {
            kind,
            axis,
            keepdim,
        } => json!({
            "kind": format!("{kind:?}"), "axis": axis, "keepdim": keepdim,
        }),
        Op::MoeRouter {
            num_experts,
            top_k,
            group,
            scoring,
            norm_topk,
            route_scale,
            correction_bias,
            logit_scale,
        } => json!({
            "num_experts": num_experts,
            "top_k": top_k,
            "scoring": format!("{scoring:?}"),
            "norm_topk": norm_topk,
            "route_scale": route_scale,
            "correction_bias": correction_bias,
            "logit_scale": logit_scale,
            "group": group.map(|g| serde_json::json!({
                "n_group": g.n_group,
                "topk_group": g.topk_group,
            })),
        }),
        Op::MoeExperts {
            num_experts,
            top_k,
            intermediate_size,
            block_fp8,
            post_expert_norm,
        } => json!({
            "num_experts": num_experts,
            "top_k": top_k,
            "intermediate_size": intermediate_size,
            "block_fp8": block_fp8,
            "post_expert_norm_eps": post_expert_norm,
        }),
        Op::DsaIndexer {
            num_heads,
            head_dim,
            rope_dim,
            top_k,
            theta,
        } => json!({
            "num_heads": num_heads, "head_dim": head_dim, "rope_dim": rope_dim,
            "top_k": top_k, "theta": theta,
        }),
        Op::DsaAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            top_k,
        } => json!({
            "num_heads": num_heads, "num_kv_heads": num_kv_heads,
            "head_dim": head_dim, "top_k": top_k,
        }),
        Op::Conv1dDepthwise { kernel } => json!({ "kernel": kernel }),
        Op::Conv1d {
            stride,
            dilation,
            groups,
            padding,
            pad_mode,
        } => json!({
            "stride": stride, "dilation": dilation, "groups": groups,
            "padding": [padding.0, padding.1], "pad_mode": format!("{pad_mode:?}"),
        }),
        Op::ConvTranspose1d {
            stride,
            groups,
            crop,
            output_padding,
        } => json!({
            "stride": stride, "groups": groups, "crop": [crop.0, crop.1],
            "output_padding": output_padding,
        }),
        Op::LinearAttention {
            kind,
            num_heads,
            head_dim,
        } => json!({
            "kind": format!("{kind:?}"),
            "num_heads": num_heads,
            "head_dim": head_dim,
        }),
        Op::SituGlu { beta, linear_beta } => json!({
            "beta": beta, "linear_beta": linear_beta,
        }),
        Op::BlockResidual { max_snapshots } => json!({ "max_snapshots": max_snapshots }),
        Op::ResidualGate { temperature, eps } => json!({
            "temperature": temperature, "eps": eps,
        }),
    }
}

/// Convert a [`Graph`] into a [`VizGraph`].
pub fn graph_to_viz_graph(
    g: &Graph,
    title: &str,
    model_type: &str,
    modality: ModelModality,
) -> VizGraph {
    let mut total_params = 0u64;
    let mut total_weight_bytes = 0u64;
    let mut category_counts = HashMap::new();

    // 1. Process tensors
    let tensors: Vec<VizTensor> = g
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let shape_str = t
                .shape
                .as_ref()
                .map(|s| s.display_with(&g.syms))
                .unwrap_or_else(|| "?".into());

            let elements = t.shape.as_ref().and_then(|s| {
                let mut prod = 1i64;
                for d in s.dims() {
                    prod = prod.checked_mul(d.as_static()?)?;
                }
                u64::try_from(prod).ok()
            });

            let bytes = elements.map(|elem| {
                let bit_width = match t.dtype {
                    crate::DType::F32 | crate::DType::I32 => 32,
                    crate::DType::F16 | crate::DType::BF16 => 16,
                    crate::DType::F8E4M3 | crate::DType::F8E5M2 => 8,
                    _ => 16,
                };
                (elem * bit_width + 7) / 8
            });

            let origin = match &t.origin {
                Origin::Input => VizOrigin::Input,
                Origin::Weight => {
                    if let Some(e) = elements {
                        total_params += e;
                    }
                    if let Some(b) = bytes {
                        total_weight_bytes += b;
                    }
                    VizOrigin::Weight
                }
                Origin::Node(nid) => VizOrigin::Activation { node: nid.0 },
            };

            VizTensor {
                id: i as u32,
                name: t.name.clone().unwrap_or_default(),
                dtype: format!("{}", t.dtype),
                shape: shape_str,
                elements,
                bytes,
                origin,
            }
        })
        .collect();

    // 2. Process nodes
    let nodes: Vec<VizNode> = g
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let cat = op_to_category(&n.op);
            *category_counts.entry(cat.label().to_string()).or_insert(0) += 1;

            let block = n.block.and_then(|idx| g.block_label(idx)).map(String::from);

            // Compute weight params/bytes used directly by this node
            let mut param_count = 0u64;
            let mut weight_bytes = 0u64;
            for &tid in &n.inputs {
                if let Some(t) = tensors.get(tid.0 as usize) {
                    if matches!(t.origin, VizOrigin::Weight) {
                        param_count += t.elements.unwrap_or(0);
                        weight_bytes += t.bytes.unwrap_or(0);
                    }
                }
            }

            VizNode {
                id: i as u32,
                op: n.op.name().to_string(),
                category: cat,
                detail: op_detail(&n.op),
                inputs: n.inputs.iter().map(|t| t.0).collect(),
                output: n.output.0,
                block,
                stage: None,
                param_count,
                weight_bytes,
            }
        })
        .collect();

    // 3. Process blocks
    let mut block_map: HashMap<String, (Vec<u32>, u64, u64)> = HashMap::new();
    for n in &nodes {
        if let Some(ref blk) = n.block {
            let entry = block_map
                .entry(blk.clone())
                .or_insert_with(|| (Vec::new(), 0, 0));
            entry.0.push(n.id);
            entry.1 += n.param_count;
            entry.2 += n.weight_bytes;
        }
    }

    let blocks: Vec<VizBlock> = g
        .blocks
        .iter()
        .map(|label| {
            let (node_ids, p_count, w_bytes) = block_map.remove(label).unwrap_or_default();
            VizBlock {
                id: label.clone(),
                label: label.clone(),
                stage: None,
                node_ids,
                param_count: p_count,
                weight_bytes: w_bytes,
            }
        })
        .collect();

    let summary = VizSummary {
        title: title.to_string(),
        model_type: model_type.to_string(),
        modality,
        total_params,
        total_weight_bytes,
        num_layers: g.blocks.len() as u32,
        hidden_size: None,
        num_heads: None,
        num_kv_heads: None,
        vocab_size: None,
        category_counts,
        total_nodes: nodes.len(),
        total_tensors: tensors.len(),
    };

    VizGraph {
        tensors,
        nodes,
        blocks,
        inputs: g.inputs.iter().map(|t| t.0).collect(),
        outputs: g.outputs.iter().map(|t| t.0).collect(),
        summary,
    }
}

/// Create a single-network [`VizModelBundle`].
pub fn bundle_from_graph(
    g: &Graph,
    title: &str,
    source: &str,
    model_type: &str,
    modality: ModelModality,
) -> VizModelBundle {
    let viz_graph = graph_to_viz_graph(g, title, model_type, modality);
    let summary = viz_graph.summary.clone();

    let mut stages = Vec::new();
    match modality {
        ModelModality::Asr => {
            stages.push(PipelineStage {
                id: "audio_encoder".into(),
                name: "Audio Encoder Tower".into(),
                description: "2D Conv subsampling + conformer/transformer audio encoder".into(),
                modality: "audio".into(),
                layer_count: Some(summary.num_layers),
                param_count: summary.total_params,
                input_shapes: vec!["[B, 1, T_mel, 128]".into()],
                output_shapes: vec!["[B, T_sub, hidden]".into()],
                subnetwork_name: Some("composite".into()),
            });
            stages.push(PipelineStage {
                id: "text_decoder".into(),
                name: "Thinker Decoder".into(),
                description: "Causal language model decoder".into(),
                modality: "text".into(),
                layer_count: Some(summary.num_layers),
                param_count: summary.total_params,
                input_shapes: vec!["[B, T_combined, hidden]".into()],
                output_shapes: vec!["[B, S, vocab_size]".into()],
                subnetwork_name: Some("composite".into()),
            });
        }
        ModelModality::Tts => {
            stages.push(PipelineStage {
                id: "token_encoder".into(),
                name: "Token Encoder".into(),
                description: "Text/phoneme token embedding & conformer layers".into(),
                modality: "text".into(),
                layer_count: Some(6),
                param_count: summary.total_params / 3,
                input_shapes: vec!["[B, S]".into()],
                output_shapes: vec!["[B, T_mel, 80]".into()],
                subnetwork_name: Some("composite".into()),
            });
            stages.push(PipelineStage {
                id: "vocoder".into(),
                name: "Audio Vocoder / Codec".into(),
                description: "Transposed convolution & Snake ResBlocks to waveform PCM".into(),
                modality: "audio".into(),
                layer_count: Some(4),
                param_count: summary.total_params * 2 / 3,
                input_shapes: vec!["[B, T_mel, 80]".into()],
                output_shapes: vec!["[B, T_pcm]".into()],
                subnetwork_name: Some("composite".into()),
            });
        }
        _ => {
            stages.push(PipelineStage {
                id: "backbone".into(),
                name: "Model Backbone".into(),
                description: format!("{model_type} transformer DAG"),
                modality: "tensor".into(),
                layer_count: Some(summary.num_layers),
                param_count: summary.total_params,
                input_shapes: vec!["[B, S]".into()],
                output_shapes: vec!["[B, S, vocab_size]".into()],
                subnetwork_name: Some("composite".into()),
            });
        }
    }

    let mut networks = HashMap::new();
    networks.insert(
        "composite".into(),
        VizSubNetwork {
            name: "composite".into(),
            title: title.to_string(),
            description: format!("{model_type} network graph"),
            graph: viz_graph,
        },
    );

    VizModelBundle {
        title: title.to_string(),
        source: source.to_string(),
        modality,
        default_network: "composite".into(),
        stages,
        networks,
        summary,
    }
}

/// Build a comprehensive multi-network [`VizModelBundle`] from a [`crate::models::ModelConfig`].
pub fn bundle_from_model_config(
    cfg: &crate::models::ModelConfig,
    title: &str,
    source: &str,
) -> Result<VizModelBundle, crate::models::BuildError> {
    use crate::models::{build_graph, ModelConfig, ShapeBucket};

    match cfg {
        ModelConfig::Qwen3Asr(c) => {
            let mut networks = HashMap::new();

            // 1. Composite pipeline graph
            let comp_g = crate::models::build_composite(c);
            let comp_viz = graph_to_viz_graph(&comp_g, title, "qwen3_asr", ModelModality::Asr);
            networks.insert(
                "composite".into(),
                VizSubNetwork {
                    name: "composite".into(),
                    title: "Composite ASR Pipeline".into(),
                    description: "End-to-end Audio Encoder + Thinker Decoder".into(),
                    graph: comp_viz,
                },
            );

            // 2. Audio Encoder standalone
            let enc_g = crate::models::build_audio_encoder(c);
            let enc_viz = graph_to_viz_graph(&enc_g, &format!("{title} (Audio Tower)"), "qwen3_audio_encoder", ModelModality::Asr);
            networks.insert(
                "audio_encoder".into(),
                VizSubNetwork {
                    name: "audio_encoder".into(),
                    title: "Audio Encoder Tower".into(),
                    description: "2D Conv 4x Subsampler + Audio Transformer Layers + Projection".into(),
                    graph: enc_viz,
                },
            );

            // 3. Thinker Decoder standalone
            let dec_g = crate::models::build_text_decoder(c);
            let dec_viz = graph_to_viz_graph(&dec_g, &format!("{title} (Thinker LM)"), "qwen3_decoder", ModelModality::Llm);
            networks.insert(
                "thinker_decoder".into(),
                VizSubNetwork {
                    name: "thinker_decoder".into(),
                    title: "Thinker Decoder".into(),
                    description: "Qwen3 Language Model Decoder Backbone".into(),
                    graph: dec_viz,
                },
            );

            let stages = vec![
                PipelineStage {
                    id: "audio_frontend".into(),
                    name: "1. Audio Front-End".into(),
                    description: "Two Conv2D stages (kernel 3, stride 2, pad 1) with GELU subsampling 4x in time".into(),
                    modality: "audio".into(),
                    layer_count: Some(2),
                    param_count: 480 * 1 * 9 + 480 * 480 * 9,
                    input_shapes: vec!["[B, 1, T_mel, 128]".into()],
                    output_shapes: vec!["[B, T_sub, 7680]".into()],
                    subnetwork_name: Some("audio_encoder".into()),
                },
                PipelineStage {
                    id: "audio_encoder".into(),
                    name: "2. Audio Transformer Encoder".into(),
                    description: format!("{} layers of LayerNorm, Multi-Head Self-Attention, and GELU MLP", c.audio_config.encoder_layers),
                    modality: "audio".into(),
                    layer_count: Some(c.audio_config.encoder_layers),
                    param_count: (c.audio_config.encoder_layers as u64) * (c.audio_config.d_model as u64) * (c.audio_config.d_model as u64 * 4 + c.audio_config.encoder_ffn_dim as u64 * 2),
                    input_shapes: vec!["[B, T_sub, d_model]".into()],
                    output_shapes: vec!["[B, T_sub, d_model]".into()],
                    subnetwork_name: Some("audio_encoder".into()),
                },
                PipelineStage {
                    id: "audio_projection".into(),
                    name: "3. Audio Projection".into(),
                    description: "Linear projection aligning audio encoder representations with LLM hidden dimension".into(),
                    modality: "audio".into(),
                    layer_count: Some(1),
                    param_count: (c.audio_config.d_model as u64) * (c.audio_config.output_dim as u64),
                    input_shapes: vec!["[B, T_sub, d_model]".into()],
                    output_shapes: vec!["[B, T_sub, hidden]".into()],
                    subnetwork_name: Some("audio_encoder".into()),
                },
                PipelineStage {
                    id: "thinker_decoder".into(),
                    name: "4. Qwen3 Thinker Decoder".into(),
                    description: format!("{} blocks of RMSNorm, MRoPE Attention, and SwiGLU MLP", c.text_config.num_hidden_layers),
                    modality: "text".into(),
                    layer_count: Some(c.text_config.num_hidden_layers),
                    param_count: (c.text_config.num_hidden_layers as u64) * (c.text_config.hidden_size as u64) * (c.text_config.hidden_size as u64 * 4 + c.text_config.intermediate_size as u64 * 3),
                    input_shapes: vec!["[B, T_combined, hidden]".into()],
                    output_shapes: vec!["[B, T_combined, hidden]".into()],
                    subnetwork_name: Some("thinker_decoder".into()),
                },
                PipelineStage {
                    id: "lm_head".into(),
                    name: "5. LM Head".into(),
                    description: "Final RMSNorm and projection to vocabulary logits".into(),
                    modality: "text".into(),
                    layer_count: Some(1),
                    param_count: (c.text_config.hidden_size as u64) * (c.text_config.vocab_size as u64),
                    input_shapes: vec!["[B, T_combined, hidden]".into()],
                    output_shapes: vec!["[B, T_combined, 151936]".into()],
                    subnetwork_name: Some("composite".into()),
                },
            ];

            let default_network = "composite".into();
            let summary = networks["composite"].graph.summary.clone();

            Ok(VizModelBundle {
                title: title.to_string(),
                source: source.to_string(),
                modality: ModelModality::Asr,
                default_network,
                stages,
                networks,
                summary,
            })
        }
        ModelConfig::S3Gen(c) => {
            let mut networks = HashMap::new();
            let comp_g = crate::models::build_graph(cfg, &ShapeBucket::default())?;
            let comp_viz = graph_to_viz_graph(&comp_g, title, "s3gen", ModelModality::Tts);
            networks.insert(
                "composite".into(),
                VizSubNetwork {
                    name: "composite".into(),
                    title: "S3Gen TTS Pipeline".into(),
                    description: "Token Encoder + CFM Estimator + HiFT Vocoder".into(),
                    graph: comp_viz,
                },
            );

            let stages = vec![
                PipelineStage {
                    id: "token_encoder".into(),
                    name: "1. Token Conformer Encoder".into(),
                    description: "Embedding, pre-lookahead convs, 6 conformers, 2x upsampling, 4 conformers".into(),
                    modality: "text".into(),
                    layer_count: Some(c.encoder_layers.0 + c.encoder_layers.1),
                    param_count: (c.encoder_dim as u64) * (c.encoder_dim as u64 * 12),
                    input_shapes: vec!["[B, S_tokens]".into()],
                    output_shapes: vec!["[B, T_mel, 80]".into()],
                    subnetwork_name: Some("composite".into()),
                },
                PipelineStage {
                    id: "cfm_estimator".into(),
                    name: "2. CFM Estimator".into(),
                    description: "Conditional Flow Matching Euler step diffusion via ResNet1D and Transformer blocks".into(),
                    modality: "diffusion".into(),
                    layer_count: Some(c.transformer_blocks + c.mid_blocks),
                    param_count: ((c.transformer_blocks + c.mid_blocks) as u64) * (c.estimator_dim as u64) * (c.estimator_dim as u64 * 8),
                    input_shapes: vec!["[B, T_mel, 80]".into()],
                    output_shapes: vec!["[B, T_mel, 80]".into()],
                    subnetwork_name: Some("composite".into()),
                },
                PipelineStage {
                    id: "hift_vocoder".into(),
                    name: "3. HiFT Neural Vocoder".into(),
                    description: "F0 harmonic source merge, transposed-conv upsampling with Snake ResBlocks to 24 kHz PCM".into(),
                    modality: "audio".into(),
                    layer_count: Some(c.upsample.len() as u32),
                    param_count: (c.hift_channels as u64) * 50_000,
                    input_shapes: vec!["[B, T_mel, 80]".into()],
                    output_shapes: vec!["[B, T_pcm]".into()],
                    subnetwork_name: Some("composite".into()),
                },
            ];

            let summary = networks["composite"].graph.summary.clone();
            Ok(VizModelBundle {
                title: title.to_string(),
                source: source.to_string(),
                modality: ModelModality::Tts,
                default_network: "composite".into(),
                stages,
                networks,
                summary,
            })
        }
        ModelConfig::Snac(c) => {
            let comp_g = crate::models::build_graph(cfg, &ShapeBucket::default())?;
            let comp_viz = graph_to_viz_graph(&comp_g, title, "snac", ModelModality::AudioCodec);
            let mut networks = HashMap::new();
            networks.insert(
                "composite".into(),
                VizSubNetwork {
                    name: "composite".into(),
                    title: "SNAC Neural Audio Codec".into(),
                    description: "Multi-scale RVQ codebook embeddings and Snake ResBlocks".into(),
                    graph: comp_viz,
                },
            );

            let stages = vec![
                PipelineStage {
                    id: "rvq_embeddings".into(),
                    name: "1. Multi-Level RVQ Embeddings".into(),
                    description: format!("Sum of {} codebook levels (vq strides {:?})", c.vq_strides.len(), c.vq_strides),
                    modality: "code".into(),
                    layer_count: Some(c.vq_strides.len() as u32),
                    param_count: (c.codebook_size as u64) * (c.latent_dim as u64) * (c.vq_strides.len() as u64),
                    input_shapes: vec!["[B, F * stride]".into()],
                    output_shapes: vec!["[B, T_latent, latent_dim]".into()],
                    subnetwork_name: Some("composite".into()),
                },
                PipelineStage {
                    id: "upsampling_blocks".into(),
                    name: "2. Snake ConvTranspose Upsampling".into(),
                    description: format!("{} upsampling blocks with Snake activations and noise injection", c.blocks.len()),
                    modality: "audio".into(),
                    layer_count: Some(c.blocks.len() as u32),
                    param_count: c.blocks.iter().map(|b| (b.cin * b.cout * (b.kernel as i64)) as u64).sum(),
                    input_shapes: vec!["[B, T_latent, latent_dim]".into()],
                    output_shapes: vec!["[B, T_pcm, 1]".into()],
                    subnetwork_name: Some("composite".into()),
                },
                PipelineStage {
                    id: "output_filter".into(),
                    name: "3. Tanh Output Waveform".into(),
                    description: "Causal projection to single audio channel with Tanh saturation".into(),
                    modality: "audio".into(),
                    layer_count: Some(1),
                    param_count: 7,
                    input_shapes: vec!["[B, T_pcm, 1]".into()],
                    output_shapes: vec!["[B, T_pcm]".into()],
                    subnetwork_name: Some("composite".into()),
                },
            ];

            let summary = networks["composite"].graph.summary.clone();
            Ok(VizModelBundle {
                title: title.to_string(),
                source: source.to_string(),
                modality: ModelModality::AudioCodec,
                default_network: "composite".into(),
                stages,
                networks,
                summary,
            })
        }
        _ => {
            let g = build_graph(cfg, &ShapeBucket::default())?;
            let modality = match cfg {
                ModelConfig::Gemma4Multimodal(_) | ModelConfig::QwenVl(_) => ModelModality::Multimodal,
                ModelConfig::Siglip(_) | ModelConfig::QwenImageDit(_) | ModelConfig::QwenImageVae(_) => ModelModality::Vision,
                _ => ModelModality::Llm,
            };
            Ok(bundle_from_graph(&g, title, source, "model", modality))
        }
    }
}

// =============================================================================
// Legacy compatibility shims
// =============================================================================

/// Serialize the graph to a JSON string (legacy compatible schema).
pub fn graph_to_json(g: &Graph) -> String {
    let v = graph_to_value(g);
    serde_json::to_string_pretty(&v).expect("graph serialization cannot fail")
}

/// Serialize the graph to a `serde_json::Value`.
pub fn graph_to_value(g: &Graph) -> serde_json::Value {
    serde_json::to_value(graph_to_viz_graph(g, "graph", "model", ModelModality::Llm))
        .expect("serialization cannot fail")
}

/// Generate an HTML page that visualizes the graph in the browser.
pub fn graph_to_html(g: &Graph, title: &str, source: &str) -> String {
    let bundle = bundle_from_graph(g, title, source, "model", ModelModality::Llm);
    render_html(&bundle, false)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{ModelConfig, Qwen3AsrConfig};

    #[test]
    fn test_asr_bundle_generation() {
        let mut asr_cfg = Qwen3AsrConfig::default();
        asr_cfg.audio_config.encoder_layers = 2;
        asr_cfg.text_config.num_hidden_layers = 2;
        let cfg = ModelConfig::Qwen3Asr(asr_cfg);

        let bundle = bundle_from_model_config(&cfg, "Qwen3-ASR-Test", "test/path")
            .expect("bundle generation failed");

        assert_eq!(bundle.modality, ModelModality::Asr);
        assert_eq!(bundle.stages.len(), 5);
        assert!(bundle.networks.contains_key("composite"));
        assert!(bundle.networks.contains_key("audio_encoder"));
        assert!(bundle.networks.contains_key("thinker_decoder"));

        let audio_enc = &bundle.networks["audio_encoder"].graph;
        assert!(!audio_enc.nodes.is_empty());
        assert!(!audio_enc.tensors.is_empty());

        let svg = bundle_to_svg(&bundle);
        assert!(svg.contains("<svg"));
        assert!(svg.contains("Qwen3-ASR-Test"));
        assert!(svg.contains("Audio Front-End"));

        let html = render_html(&bundle, false);
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("mode-standalone"));

        let embed_html = render_html(&bundle, true);
        assert!(embed_html.contains("mode-embed"));
    }

    #[test]
    fn test_tts_snac_bundle() {
        let snac_cfg = crate::models::config::SnacConfig::default();
        let cfg = ModelConfig::Snac(snac_cfg);

        let bundle = bundle_from_model_config(&cfg, "SNAC-24kHz", "test/snac")
            .expect("bundle generation failed");

        assert_eq!(bundle.modality, ModelModality::AudioCodec);
        assert_eq!(bundle.stages.len(), 3);
        assert!(bundle.networks.contains_key("composite"));

        let svg = bundle_to_svg(&bundle);
        assert!(svg.contains("SNAC-24kHz"));
    }

    #[test]
    fn test_tts_s3gen_bundle() {
        const S3GEN: &str = r#"{"model_type": "chatterbox_s3gen", "vocab": 6561,
          "upsample": [[8, 16, 4], [5, 11, 3], [3, 7, 2]], "source_downs": [[30, 15, 7], [6, 3, 1], [1, 1, 0]],
          "source_resblocks": [[7, [1, 3, 5]], [7, [1, 3, 5]], [11, [1, 3, 5]]],
          "resblocks": [[3, [1, 3, 5]], [7, [1, 3, 5]], [11, [1, 3, 5]], [3, [1, 3, 5]], [7, [1, 3, 5]],
            [11, [1, 3, 5]], [3, [1, 3, 5]], [7, [1, 3, 5]], [11, [1, 3, 5]]],
          "n_fft": 16, "harmonics": 9}"#;

        let cfg = ModelConfig::from_json(S3GEN).expect("parse s3gen");
        let bundle = bundle_from_model_config(&cfg, "S3Gen-Test", "test/s3gen")
            .expect("bundle generation failed");

        assert_eq!(bundle.modality, ModelModality::Tts);
        assert_eq!(bundle.stages.len(), 3);
        assert!(bundle.networks.contains_key("composite"));
    }

    #[test]
    fn test_dump_assets() {
        let mut asr_cfg = Qwen3AsrConfig::default();
        asr_cfg.audio_config.encoder_layers = 1;
        asr_cfg.text_config.num_hidden_layers = 1;
        let cfg = ModelConfig::Qwen3Asr(asr_cfg);
        let bundle = bundle_from_model_config(&cfg, "Dump-Test", "test/dump").unwrap();

        let tmp_dir = std::env::temp_dir().join("plow_viz_test_dump");
        let _ = std::fs::remove_dir_all(&tmp_dir);
        dump_assets(&bundle, &tmp_dir).expect("dump assets failed");

        assert!(tmp_dir.join("index.html").exists());
        assert!(tmp_dir.join("embed.html").exists());
        assert!(tmp_dir.join("model.json").exists());
        assert!(tmp_dir.join("pipeline.json").exists());
        assert!(tmp_dir.join("pipeline.svg").exists());

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
