//! DeepSeek-V4.1-Flash configuration, read from the checkpoint's `config.json` (`text_config`) and
//! `model.safetensors.index.json` (for the tensor list plowrt's `Checkpoint` does not expose).

use std::path::Path;

use serde_json::Value;

use crate::error::{Result, RuntimeError};

#[derive(Clone, Debug)]
pub struct Cfg {
    pub vocab: usize,
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub q_lora: usize,
    pub o_groups: usize,
    pub o_lora: usize,
    pub n_experts: usize,
    pub topk: usize,
    pub moe_inter: usize,
    pub route_scale: f32,
    pub norm_topk: bool,
    pub swiglu_limit: f32,
    pub eps: f32,
    pub window: usize,
    pub compress_ratios: Vec<usize>,
    pub kv_source: Vec<usize>,
    pub index_source: Vec<usize>,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_topk: usize,
    pub candidate_source: Option<usize>,
    pub cand_topk_blocks: usize,
    pub cand_block: usize,
    pub hc_mult: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub engram_layers: Vec<usize>,
    pub engram_n_heads: usize,
    pub engram_head_dim: usize,
    pub engram_max_ngram: usize,
    pub engram_vocab_size: i64,
    pub engram_compressed_vocab: usize,
    pub engram_pad_id: usize,
    pub rope_theta: f64,
    pub compress_rope_theta: f64,
    pub yarn_factor: f64,
    pub yarn_orig: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub eos_id: u32,
    /// Every tensor name in the shard index.
    pub names: Vec<String>,
}

fn us(v: &Value, k: &str) -> Result<usize> {
    v[k].as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| RuntimeError::Device(format!("dsv41 config: missing {k}")))
}
fn fl(v: &Value, k: &str) -> Result<f64> {
    v[k].as_f64().ok_or_else(|| RuntimeError::Device(format!("dsv41 config: missing {k}")))
}
fn list(v: &Value, k: &str) -> Result<Vec<usize>> {
    v[k].as_array()
        .ok_or_else(|| RuntimeError::Device(format!("dsv41 config: missing {k}")))?
        .iter()
        .map(|x| x.as_u64().map(|u| u as usize).ok_or_else(|| RuntimeError::Device(format!("dsv41 config: bad {k}"))))
        .collect()
}

impl Cfg {
    pub fn load(dir: &Path) -> Result<Cfg> {
        let read = |f: &str| -> Result<Value> {
            let s = std::fs::read_to_string(dir.join(f)).map_err(|e| RuntimeError::Device(format!("{f}: {e}")))?;
            serde_json::from_str(&s).map_err(|e| RuntimeError::Device(format!("{f}: {e}")))
        };
        let root = read("config.json")?;
        if root["model_type"].as_str() != Some("deepseek_v41") {
            return Err(RuntimeError::Device(format!(
                "dsv41: {} is not a deepseek_v41 checkpoint",
                dir.display()
            )));
        }
        let t = &root["text_config"];
        let rs = &t["rope_scaling"];
        let index = read("model.safetensors.index.json")?;
        let mut names: Vec<String> = index["weight_map"]
            .as_object()
            .ok_or_else(|| RuntimeError::Device("dsv41: index has no weight_map".into()))?
            .keys()
            .cloned()
            .collect();
        names.sort();
        let cand = t["candidate_source_layer_id"].as_i64().unwrap_or(-1);
        Ok(Cfg {
            vocab: us(t, "vocab_size")?,
            hidden: us(t, "hidden_size")?,
            n_layers: us(t, "num_hidden_layers")?,
            n_heads: us(t, "num_attention_heads")?,
            head_dim: us(t, "head_dim")?,
            rope_dim: us(t, "qk_rope_head_dim")?,
            q_lora: us(t, "q_lora_rank")?,
            o_groups: us(t, "o_groups")?,
            o_lora: us(t, "o_lora_rank")?,
            n_experts: us(t, "n_routed_experts")?,
            topk: us(t, "num_experts_per_tok")?,
            moe_inter: us(t, "moe_intermediate_size")?,
            route_scale: fl(t, "routed_scaling_factor")? as f32,
            norm_topk: t["norm_topk_prob"].as_bool().unwrap_or(true),
            swiglu_limit: fl(t, "swiglu_limit")? as f32,
            eps: fl(t, "rms_norm_eps")? as f32,
            window: us(t, "sliding_window")?,
            compress_ratios: list(t, "compress_ratios")?,
            kv_source: list(t, "kv_source_layer_ids")?,
            index_source: list(t, "index_source_layer_ids")?,
            index_heads: us(t, "index_n_heads")?,
            index_dim: us(t, "index_head_dim")?,
            index_topk: us(t, "index_topk")?,
            candidate_source: (cand >= 0).then_some(cand as usize),
            cand_topk_blocks: us(t, "candidate_topk_blocks")?,
            cand_block: us(t, "candidate_block_size")?,
            hc_mult: us(t, "hc_mult")?,
            sinkhorn_iters: us(t, "hc_sinkhorn_iters")?,
            hc_eps: fl(t, "hc_eps")? as f32,
            engram_layers: list(t, "engram_layer_ids")?,
            engram_n_heads: us(t, "engram_n_heads")?,
            engram_head_dim: us(t, "engram_head_dim")?,
            engram_max_ngram: us(t, "engram_max_ngram_size")?,
            engram_vocab_size: t["engram_vocab_size"].as_i64().unwrap_or(0),
            engram_compressed_vocab: t["engram_compressed_vocab_size"].as_u64().unwrap_or(0) as usize,
            engram_pad_id: t["engram_pad_token_id"].as_u64().unwrap_or(2) as usize,
            rope_theta: fl(t, "rope_theta")?,
            compress_rope_theta: fl(t, "compress_rope_theta")?,
            yarn_factor: fl(rs, "factor")?,
            yarn_orig: us(rs, "original_max_position_embeddings")?,
            beta_fast: fl(rs, "beta_fast")?,
            beta_slow: fl(rs, "beta_slow")?,
            eos_id: root["eos_token_id"].as_u64().unwrap_or(1) as u32,
            names,
        })
    }

    /// The layer's tensors that live on the GPU (everything but the Engram tables).
    pub fn layer_tensor_names(&self, l: usize) -> Vec<String> {
        let p = format!("layers.{l}.");
        self.names
            .iter()
            .filter(|n| n.starts_with(&p) && !n.contains(".engram.embed."))
            .cloned()
            .collect()
    }

    /// The most recent kv source at or before layer `l` (whose compressed cache `l` reads).
    pub fn kv_src(&self, l: usize) -> Option<usize> {
        self.kv_source.iter().copied().filter(|&s| s <= l).max()
    }
    /// The most recent index source at or before `l` (whose top-k picks `l` attends over).
    pub fn index_src(&self, l: usize) -> Option<usize> {
        self.index_source.iter().copied().filter(|&s| s <= l).max()
    }

    /// cos/sin tables [max_pos][rope_dim/2] (model.py precompute_freqs_cis): plain rope for the
    /// window-only layers, YaRN at `compress_rope_theta` for layers with a compressed KV.
    pub fn rope_tables(&self, max_pos: usize, compressed: bool) -> (Vec<f32>, Vec<f32>) {
        let dim = self.rope_dim;
        let (base, orig) = if compressed { (self.compress_rope_theta, self.yarn_orig) } else { (self.rope_theta, 0) };
        let base32 = base as f32;
        let mut freqs: Vec<f32> = (0..dim / 2).map(|i| 1.0f32 / base32.powf((2 * i) as f32 / dim as f32)).collect();
        if orig > 0 {
            let corrected = |rot: f64| dim as f64 * (orig as f64 / (rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln());
            let low = (corrected(self.beta_fast).floor()).max(0.0);
            let high = (corrected(self.beta_slow).ceil()).min((dim - 1) as f64);
            let span = (high - low).max(1e-3) as f32;
            for (i, f) in freqs.iter_mut().enumerate() {
                let ramp = ((i as f32 - low as f32) / span).clamp(0.0, 1.0);
                let smooth = 1.0 - ramp;
                *f = *f / self.yarn_factor as f32 * (1.0 - smooth) + *f * smooth;
            }
        }
        let half = dim / 2;
        let mut c = vec![0f32; max_pos * half];
        let mut s = vec![0f32; max_pos * half];
        for p in 0..max_pos {
            for i in 0..half {
                let a = p as f32 * freqs[i];
                c[p * half + i] = a.cos();
                s[p * half + i] = a.sin();
            }
        }
        (c, s)
    }
}
