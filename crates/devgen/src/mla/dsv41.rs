use super::*;

// ===== DeepSeek-V4.1-Flash (`deepseek_v41` / text `deepseek_v41_text`) — FRONT END ============
//
// The geometry comes from `nn_graph::models::config::DeepSeekV41Config`, which already parses AND
// validates this checkpoint (compress-ratio reconciliation, the two-level index split, the Engram
// cross-checks) with a test against the released shards. devgen reuses it rather than growing a
// second reader: `costmodel` already depends on `nn-graph`, so the edge costs no crate, and this
// config's whole difficulty is in fields that are easy to read and easy to MISREAD --
// `compress_ratios` and `kv_source_layer_ids` disagree by construction.
//
// What this module adds on top of the parse is the thing a config cannot tell you: whether the
// TENSORS on disk have the shapes the emit is about to assume.

/// Resolved V4.1 attention geometry, in the terms the emit needs rather than the config's.
///
/// V4.1's MLA is FULLY ABSORBED, and that is visible in the shards rather than the config: there
/// is no `kv_b` tensor at all. `wkv` is `[512, 5120]` -- one 512-wide latent per token, shared by
/// all 64 heads (`num_key_value_heads = 1`), serving as both K and V. So the latent width is
/// `head_dim`, not a separate v projection, and it is REPLICATED under TP rather than sharded.
/// Fields the emit path will read once it exists (`dsv41_gaps` lists what stands in the way).
/// Kept rather than trimmed to what today's refusal happens to print, for the reason `K3Cfg`
/// states: re-deriving a dimension later from a different source is how two values for one
/// quantity appear, and at emit time a wrong one is indistinguishable from a right one.
#[allow(dead_code)]
pub(crate) struct Dsv41Cfg {
    pub layers: u32,
    pub hidden: u32,
    pub heads: u32,
    /// `head_dim` = 512. The full per-head q width AND the latent width.
    pub head_dim: u32,
    /// 64 of the 512 are rotated; the other 448 are content ("nope").
    pub qk_rope: u32,
    pub q_lora: u32,
    pub eps: f32,
    /// Grouped output LoRA: `o_groups` x `o_lora_rank` = 8 x 1024. See [`Dsv41Cfg::wo_a_groups`].
    pub o_groups: u32,
    pub o_lora_rank: u32,
    pub n_exp: u32,
    pub top_k: u32,
    pub moe_inter: u32,
    pub index_heads: u32,
    pub index_dim: u32,
    pub index_topk: u32,
    /// Layers owning a CSA2 compressor. EVERY other layer READS their cache.
    pub kv_source: Vec<u32>,
    /// Layers owning indexer queries (8). Only the `kv_source` 4 own index KEYS.
    pub index_source: Vec<u32>,
    pub engram_layers: Vec<u32>,
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,
    pub sliding_window: u32,
}

#[allow(dead_code)]
impl Dsv41Cfg {
    /// `wo_a` is BLOCK-DIAGONAL over `o_groups`, which the shard shape states and a `Linear` would
    /// not: `[8192, 4096]` is 8 stacked `[o_lora_rank, heads/o_groups * head_dim]` blocks, and the
    /// reference spells it `einsum("bsgd,grd->bsgr")` precisely because a plain `Linear` over the
    /// whole 32768-wide attention output would be a different -- and 8x larger -- operator.
    ///
    /// Returns `(groups, out_per_group, in_per_group)`.
    pub fn wo_a_groups(&self) -> (u32, u32, u32) {
        (
            self.o_groups,
            self.o_lora_rank,
            self.heads * self.head_dim / self.o_groups,
        )
    }

    /// `wo_b`'s input width: every group's LoRA output concatenated.
    pub fn wo_b_in(&self) -> u32 {
        self.o_groups * self.o_lora_rank
    }

    /// Softmax scale is `head_dim ** -0.5` over the FULL 512, not over the 64 rotated dims.
    pub fn attn_scale(&self) -> f32 {
        (self.head_dim as f32).powf(-0.5)
    }
}

/// Parse the released checkpoint through nn-graph, wrapper or flat text tower.
///
/// `ModelConfig::from_json` deliberately REFUSES the released checkpoint: it is the multimodal
/// wrapper, its ViT and aligner are unmodeled, and the parser points at the text-generation
/// frontend rather than silently dropping the vision half. The text tower is `text_config` -- and
/// that sub-object does not parse alone either, because `quantization_config` and `dtype` live at
/// the wrapper's TOP level. `sub_config` is the merge that lifts them down.
pub(crate) fn parse_dsv41(dir: &Path) -> Option<nn_graph::models::config::DeepSeekV41Config> {
    use nn_graph::models::config::{DeepSeekV41Config, ModelConfig};
    fn v41_of(text: &str) -> Option<DeepSeekV41Config> {
        match ModelConfig::from_json(text) {
            Ok(ModelConfig::DeepSeekV41(v)) => Some(v),
            _ => None,
        }
    }
    let raw = std::fs::read_to_string(dir.join("config.json")).ok()?;
    v41_of(&raw).or_else(|| {
        let v: Value = serde_json::from_str(&raw).ok()?;
        v.get("text_config")?;
        let tower = nn_graph::models::config::sub_config(&v, "text_config");
        v41_of(&serde_json::to_string(&tower).ok()?)
    })
}

pub(crate) fn cfg_dsv41(dir: &Path) -> Result<Dsv41Cfg, String> {
    let c =
        parse_dsv41(dir).ok_or_else(|| "config.json is not a deepseek_v41 tower".to_string())?;
    c.validate().map_err(|e| e.to_string())?;
    let kv_source = (0..c.num_hidden_layers)
        .filter(|l| c.is_kv_source(*l))
        .collect();
    let index_source = (0..c.num_hidden_layers)
        .filter(|l| c.is_index_source(*l))
        .collect();
    Ok(Dsv41Cfg {
        layers: c.num_hidden_layers,
        hidden: c.hidden_size as u32,
        heads: c.num_attention_heads,
        head_dim: c.head_dim,
        qk_rope: c.qk_rope_head_dim,
        q_lora: c.q_lora_rank,
        eps: c.rms_norm_eps,
        o_groups: c.o_groups,
        o_lora_rank: c.o_lora_rank,
        n_exp: c.n_routed_experts,
        top_k: c.num_experts_per_tok,
        moe_inter: c.moe_intermediate_size as u32,
        index_heads: c.index_n_heads,
        index_dim: c.index_head_dim,
        index_topk: c.index_topk,
        kv_source,
        index_source,
        engram_layers: c.engram_layer_ids.clone(),
        hc_mult: c.hc_mult,
        hc_sinkhorn_iters: c.hc_sinkhorn_iters,
        sliding_window: c.sliding_window,
    })
}

/// Every layer-0 attention shape the emit would otherwise assume, checked against the shards.
///
/// A tensor's ABSENCE proves nothing -- the download may be partial -- so a miss is skipped and
/// only a CONTRADICTION is reported. That asymmetry is the point: this can tell you the emit is
/// wrong, never that it is right.
pub(crate) fn dsv41_shard_check(c: &Dsv41Cfg, dir: &Path) -> Vec<String> {
    let (hdr, _have, _total) = super::kimi_k3::k3_shard_headers(dir);
    if hdr.is_empty() {
        return Vec::new();
    }
    let (g, orow, ocol) = c.wo_a_groups();
    let want: &[(&str, [u32; 2])] = &[
        ("attn.wkv.weight", [c.head_dim, c.hidden]),
        ("attn.kv_norm.weight", [c.head_dim, 0]),
        ("attn.wq_a.weight", [c.q_lora, c.hidden]),
        ("attn.q_norm.weight", [c.q_lora, 0]),
        ("attn.wq_b.weight", [c.heads * c.head_dim, c.q_lora]),
        ("attn.wo_a.weight", [g * orow, ocol]),
        ("attn.wo_b.weight", [c.hidden, c.wo_b_in()]),
        ("attn.attn_sink", [c.heads, 0]),
    ];
    let mut bad = Vec::new();
    for (suffix, dims) in want {
        let name = format!("layers.0.{suffix}");
        let Some((_dt, shape)) = hdr
            .get(&name)
            .or_else(|| hdr.get(&format!("model.{name}")))
        else {
            continue; // absent proves nothing
        };
        let expect: Vec<i64> = if dims[1] == 0 {
            vec![dims[0] as i64]
        } else {
            vec![dims[0] as i64, dims[1] as i64]
        };
        if *shape != expect {
            bad.push(format!(
                "{name}: shard has {shape:?}, config implies {expect:?}"
            ));
        }
    }
    bad
}

/// What the EMITTER still needs, itemised. Ordered as
/// `docs/amd/deepseek-v41-flash-mi300x.md` section 5.6.
pub(crate) fn dsv41_gaps(c: &Dsv41Cfg) -> Vec<String> {
    let in_per_group = c.heads * c.head_dim / c.o_groups;
    vec![
        format!(
            "CSA2 emit: the compressor runs on the {} kv_source layers {:?} and every one of the \
             {} layers READS that cache, so a per-layer compressor is the wrong shape (ops \
             180/181 exist and dispatch)",
            c.kv_source.len(),
            c.kv_source,
            c.layers
        ),
        format!(
            "two-level indexer: {} layers own indexer queries but only the {} kv_source layers own \
             index KEYS, because keys derive from the compressor latent",
            c.index_source.len(),
            c.kv_source.len()
        ),
        format!(
            "Engram host side: the tokenizer-derived compressed-token map and the per-step lookback \
             cache, for layers {:?}. The hash tables themselves are done \
             (DeepSeekV41Config::engram_hash_tables); device ops 182/183 dispatch",
            c.engram_layers
        ),
        format!(
            "grouped output LoRA: wo_a is block-diagonal, {} groups of [{}, {}]. NOT a blocker for \
             a PREFILL emit -- it is {} ordinary GEMMs, each [T, {}] x [{}, {}], and at T=8192 one \
             alone fills 304 CUs. A fused grouped kernel is a DECODE optimisation (T=1 makes them \
             {} tiny GEMVs), not a correctness gap",
            c.o_groups,
            c.o_lora_rank,
            in_per_group,
            c.o_groups,
            in_per_group,
            in_per_group,
            c.o_lora_rank,
            c.o_groups,
        ),
        format!(
            "sparse_attn operand shape: V4.1 concatenates window KV and compressed KV into ONE \
             cache and one index list over a flat {}-wide latent with num_key_value_heads=1, where \
             FlashGather* is MLA-shaped (Qabs/Qrope/Ckv/Krope)",
            c.head_dim
        ),
        format!(
            "full-model emit: fork glm53_emit_full, which already emits prefill buckets plus decode \
             rungs with mHC at V4.1's own constants (hc_mult={}, sinkhorn={})",
            c.hc_mult, c.hc_sinkhorn_iters
        ),
    ]
}
