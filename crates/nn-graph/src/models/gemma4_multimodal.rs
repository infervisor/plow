//! Gemma 4 multimodal composite graph: SigLIP vision encoder → projector → text decoder.
//!
//! This builds the full multimodal pipeline as a single graph. The architecture:
//! 1. **Vision encoder** (SigLIP): image → patch embeddings → ViT → per-patch features
//! 2. **Projector** (linear): maps vision hidden → text hidden size
//! 3. **Text decoder** (Gemma 4): processes the interleaved text + projected image tokens
//!
//! At the graph level, the vision side produces `[B, N_img_tokens, text_hidden]`
//! which is concatenated with the text embeddings along the sequence axis. The
//! text decoder then attends over the full interleaved sequence.

use super::config::{parse_dtype, Gemma4MultimodalConfig, GemmaConfig, SiglipConfig};
use crate::op::ActKind;
use crate::Nn;
use crate::{DType, Dim, Graph, TensorId};

pub fn build(cfg: &Gemma4MultimodalConfig) -> Graph {
    let text_cfg = &cfg.text_config;
    let vis_cfg = &cfg.vision_config;
    let dt = parse_dtype(text_cfg.torch_dtype.as_deref());
    let mut nn = Nn::new(dt, dt);

    let h_text = text_cfg.hidden_size;
    let h_vis = vis_cfg.hidden_size;
    let n_img_tokens = cfg.image_token_count();
    let proj_out = cfg.projector_output();
    let eps = text_cfg.rms_norm_eps;

    // Symbolic dims.
    let b = nn.sym("B");
    let s_text = nn.sym("S"); // text-only sequence length

    // --- Vision encoder: image → [B, N_img_tokens, h_vis] ---
    let img = nn.input(
        "pixel_values",
        nn.shape([
            b.clone(),
            Dim::stat(vis_cfg.num_channels),
            Dim::stat(vis_cfg.image_size),
            Dim::stat(vis_cfg.image_size),
        ]),
        dt,
    );
    let vis_embed = vision_encoder(&mut nn, vis_cfg, img, &b);

    // --- Projector: [B, N_img_tokens, h_vis] → [B, N_img_tokens, h_text] ---
    let projected = nn.linear(
        "multi_modal_projector.linear",
        vis_embed,
        h_vis,
        proj_out,
        true,
    );

    // --- Text input: token embeddings ---
    let ids = nn.input(
        "input_ids",
        nn.shape([b.clone(), s_text.clone()]),
        DType::I32,
    );
    let text_embed = nn.embedding("embed_tokens", ids, text_cfg.vocab_size, h_text);
    let text_embed = nn.scale(text_embed, (h_text as f32).sqrt());

    // --- Interleave: concat vision embeddings into the text sequence ---
    // The combined sequence is [image_tokens | text_tokens] along axis 1.
    let combined = nn.concat(1, vec![projected, text_embed]);

    // The combined sequence dim: N_img_tokens + S_text.
    let s_combined = Dim::stat(n_img_tokens).add(&s_text);

    // --- Text decoder layers ---
    let mut x = combined;
    for layer in 0..text_cfg.num_hidden_layers {
        let p = format!("layers.{layer}");
        nn.begin_block(&p);
        let is_global = text_cfg.layer_is_global(layer);
        let sliding = if is_global {
            None
        } else {
            Some(text_cfg.sliding_window)
        };

        // --- attention block (pre-norm, post-norm, residual) ---
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.input_layernorm"), x, h_text, eps);
        let attn =
            text_attention(&mut nn, text_cfg, &p, normed, &b, &s_combined, is_global, sliding);
        let attn = nn.rmsnorm(&format!("{p}.post_attention_layernorm"), attn, h_text, eps);
        x = nn.add(residual, attn);

        // --- MLP block (pre-norm, post-norm, residual) ---
        let residual = x;
        let normed = nn.rmsnorm(&format!("{p}.pre_feedforward_layernorm"), x, h_text, eps);
        let mlp = if text_cfg.layer_is_moe(layer) {
            moe_ffn(&mut nn, text_cfg, &p, normed, h_text)
        } else {
            geglu_mlp(&mut nn, text_cfg, &p, normed, h_text)
        };
        let mlp = nn.rmsnorm(&format!("{p}.post_feedforward_layernorm"), mlp, h_text, eps);
        x = nn.add(residual, mlp);
    }
    nn.end_block();

    x = nn.rmsnorm("norm", x, h_text, eps);

    // LM head.
    let logits = nn.linear("lm_head", x, h_text, text_cfg.vocab_size, false);
    nn.mark_output(logits);
    nn.finish()
}

// =============================================================================
// Text decoder helpers (attention, MLP, MoE) — same logic as gemma.rs but with
// an explicit sequence dim parameter for the multimodal combined sequence.
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn text_attention(
    nn: &mut Nn,
    cfg: &GemmaConfig,
    prefix: &str,
    x: TensorId,
    b: &Dim,
    s: &Dim,
    is_global: bool,
    sliding_window: Option<u32>,
) -> TensorId {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let nkv = cfg.kv_heads_for(is_global);
    let hd = cfg.head_dim_for(is_global);
    let eps = cfg.rms_norm_eps;
    let q_dim = nh as i64 * hd;
    let kv_dim = nkv as i64 * hd;

    // Projections.
    let q = nn.linear(&format!("{prefix}.self_attn.q_proj"), x, h, q_dim, false);
    let (k_lin, v_lin) = if cfg.attention_k_eq_v {
        let kv = nn.linear(&format!("{prefix}.self_attn.kv_proj"), x, h, kv_dim, false);
        (kv, kv)
    } else {
        let k = nn.linear(&format!("{prefix}.self_attn.k_proj"), x, h, kv_dim, false);
        let v = nn.linear(&format!("{prefix}.self_attn.v_proj"), x, h, kv_dim, false);
        (k, v)
    };

    // Split heads: [B, S, n*hd] -> [B, S, n, hd].
    let mut q = nn.reshape(
        q,
        [b.clone(), s.clone(), Dim::stat(nh as i64), Dim::stat(hd)],
    );
    let mut k = nn.reshape(
        k_lin,
        [b.clone(), s.clone(), Dim::stat(nkv as i64), Dim::stat(hd)],
    );
    let v = nn.reshape(
        v_lin,
        [b.clone(), s.clone(), Dim::stat(nkv as i64), Dim::stat(hd)],
    );

    // Gemma3/4 query/key RMSNorm.
    if cfg.use_qk_norm {
        q = nn.rmsnorm(&format!("{prefix}.self_attn.q_norm"), q, hd, eps);
        k = nn.rmsnorm(&format!("{prefix}.self_attn.k_norm"), k, hd, eps);
    }

    // RoPE.
    let (theta, rotary_factor) = cfg.rope_for(is_global);
    let rotary_dim = ((hd as f32) * rotary_factor).round() as u32;
    q = nn.rope(q, rotary_dim, theta);
    k = nn.rope(k, rotary_dim, theta);

    // Query scaling.
    let scalar = if cfg.query_pre_attn_scalar > 0.0 {
        cfg.query_pre_attn_scalar
    } else {
        hd as f32
    };
    q = nn.scale(q, 1.0 / scalar.sqrt());

    let attn = nn.attention(q, k, v, nh, nkv, hd as u32, true, sliding_window, None);

    // Merge heads back: [B, S, n, hd] -> [B, S, n*hd].
    let merged = nn.reshape(attn, [b.clone(), s.clone(), Dim::stat(q_dim)]);
    nn.linear(
        &format!("{prefix}.self_attn.o_proj"),
        merged,
        q_dim,
        h,
        false,
    )
}

/// GeGLU MLP: `down(act(gate(x)) * up(x))`.
fn geglu_mlp(nn: &mut Nn, cfg: &GemmaConfig, prefix: &str, x: TensorId, h: i64) -> TensorId {
    let inter = cfg.intermediate_size;
    let gate = nn.linear(&format!("{prefix}.mlp.gate_proj"), x, h, inter, false);
    let up = nn.linear(&format!("{prefix}.mlp.up_proj"), x, h, inter, false);
    let gate = nn.act(ActKind::GeluTanh, gate);
    let hidden = nn.mul(gate, up);
    nn.linear(&format!("{prefix}.mlp.down_proj"), hidden, inter, h, false)
}

/// Single expert GeGLU FFN.
fn expert_geglu(nn: &mut Nn, prefix: &str, x: TensorId, h: i64, inter: i64) -> TensorId {
    let gate = nn.linear(&format!("{prefix}.gate_proj"), x, h, inter, false);
    let up = nn.linear(&format!("{prefix}.up_proj"), x, h, inter, false);
    let gate = nn.act(ActKind::GeluTanh, gate);
    let hidden = nn.mul(gate, up);
    nn.linear(&format!("{prefix}.down_proj"), hidden, inter, h, false)
}

/// MoE FFN: router + representative expert.
fn moe_ffn(nn: &mut Nn, cfg: &GemmaConfig, prefix: &str, x: TensorId, h: i64) -> TensorId {
    let _logits = nn.moe_router(
        &format!("{prefix}.mlp.gate"),
        x,
        h,
        cfg.num_local_experts,
        cfg.num_experts_per_tok,
    );
    expert_geglu(
        nn,
        &format!("{prefix}.mlp.experts.0"),
        x,
        h,
        cfg.intermediate_size,
    )
}

// =============================================================================
// Vision encoder (SigLIP-based, per-patch output — no mean pooling).
// =============================================================================

/// Build the SigLIP-based vision encoder, returning per-patch features
/// `[B, N_patches, hidden]` (no mean pooling — we need per-patch tokens for the
/// multimodal projector).
fn vision_encoder(nn: &mut Nn, cfg: &SiglipConfig, img: TensorId, b: &Dim) -> TensorId {
    let h = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let head_dim = h / nh as i64;
    let eps = cfg.layer_norm_eps;
    let n_patch = cfg.num_patches();

    // Patch embedding via strided conv: [B, C, I, I] -> [B, H, P, P].
    let patches = nn.conv2d(
        "vision_tower.vision_model.embeddings.patch_embedding",
        img,
        cfg.num_channels,
        h,
        (cfg.patch_size, cfg.patch_size),
        (cfg.patch_size as u32, cfg.patch_size as u32),
        (0, 0),
        true,
    );
    // Flatten spatial and move channels last: [B, H, P*P] -> [B, P*P, H].
    let patches = nn.reshape(patches, [b.clone(), Dim::stat(h), Dim::stat(n_patch)]);
    let mut x = nn.transpose(patches, vec![0, 2, 1]);

    // Learned position embedding.
    let pos = nn.param(
        "vision_tower.vision_model.embeddings.position_embedding.weight",
        [Dim::stat(n_patch), Dim::stat(h)],
    );
    x = nn.add(x, pos);

    for layer in 0..cfg.num_hidden_layers {
        let p = format!("vision_tower.vision_model.encoder.layers.{layer}");
        nn.begin_block(&p);

        let residual = x;
        let normed = nn.layernorm(&format!("{p}.layer_norm1"), x, h, eps);
        let attn = vision_self_attention(nn, &p, normed, b, n_patch, h, nh, head_dim);
        x = nn.add(residual, attn);

        let residual = x;
        let normed = nn.layernorm(&format!("{p}.layer_norm2"), x, h, eps);
        let mlp = vision_mlp(nn, &p, normed, h, cfg.intermediate_size, &cfg.hidden_act);
        x = nn.add(residual, mlp);
    }
    nn.end_block();

    // Post-layernorm yields per-patch features (no pooling for multimodal).
    nn.layernorm("vision_tower.vision_model.post_layernorm", x, h, eps)
}

#[allow(clippy::too_many_arguments)]
fn vision_self_attention(
    nn: &mut Nn,
    p: &str,
    x: TensorId,
    b: &Dim,
    seq: i64,
    h: i64,
    nh: u32,
    head_dim: i64,
) -> TensorId {
    let q = nn.linear(&format!("{p}.self_attn.q_proj"), x, h, h, true);
    let k = nn.linear(&format!("{p}.self_attn.k_proj"), x, h, h, true);
    let v = nn.linear(&format!("{p}.self_attn.v_proj"), x, h, h, true);

    let q = nn.reshape(
        q,
        [
            b.clone(),
            Dim::stat(seq),
            Dim::stat(nh as i64),
            Dim::stat(head_dim),
        ],
    );
    let k = nn.reshape(
        k,
        [
            b.clone(),
            Dim::stat(seq),
            Dim::stat(nh as i64),
            Dim::stat(head_dim),
        ],
    );
    let v = nn.reshape(
        v,
        [
            b.clone(),
            Dim::stat(seq),
            Dim::stat(nh as i64),
            Dim::stat(head_dim),
        ],
    );

    // Bidirectional (non-causal) full attention.
    let attn = nn.attention(q, k, v, nh, nh, head_dim as u32, false, None, None);
    let merged = nn.reshape(attn, [b.clone(), Dim::stat(seq), Dim::stat(h)]);
    nn.linear(&format!("{p}.self_attn.out_proj"), merged, h, h, true)
}

fn vision_mlp(nn: &mut Nn, p: &str, x: TensorId, h: i64, inter: i64, act: &str) -> TensorId {
    let up = nn.linear(&format!("{p}.mlp.fc1"), x, h, inter, true);
    let up = nn.act(act_kind(act), up);
    nn.linear(&format!("{p}.mlp.fc2"), up, inter, h, true)
}

fn act_kind(s: &str) -> ActKind {
    match s {
        "gelu" => ActKind::Gelu,
        "gelu_pytorch_tanh" | "gelu_tanh" => ActKind::GeluTanh,
        "relu" => ActKind::Relu,
        "quick_gelu" => ActKind::QuickGelu,
        _ => ActKind::GeluTanh,
    }
}
