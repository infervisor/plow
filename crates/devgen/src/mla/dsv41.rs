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
    /// The validated config itself. Kept rather than fully flattened so that derived quantities
    /// -- the ue8m0 block grid, `compressor_has_gate`, `engram_rows` -- have exactly ONE
    /// definition, in nn-graph, where they are already tested against the released shards. A
    /// second copy of the `[32, 32]` grid here is precisely how V4's `[128, 128]` would creep back
    /// in on a checkpoint that does not use it.
    pub raw: nn_graph::models::config::DeepSeekV41Config,
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
        raw: c,
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
    let [ob, ib] = c.raw.quantization_config.weight_block_size;
    vec![
        // FIRST, because it is now the whole remaining job. The kernel gate that used to lead this
        // list is CLOSED: op 184 / d_gemm_t<WFP8MX> reads the [{ob}, {ib}] ue8m0 grid, passes 12/12
        // on gfx942, and `emit_pf_gemm_fp8_mx` emits it. What is left is the emitter around it.
        format!(
            "full-model emit: there is no `declare_dsv41_rows_batched` and no \
             `emit_dsv41_program`. Fork `glm53_emit_full`, which already does prefill buckets plus \
             decode rungs with mHC at V4.1's own constants (hc_mult={}, sinkhorn={}) -- but the \
             driver is 82 lines and the work is under it: `declare_glm_rows_batched` is 1182 lines \
             and V4.1's is a REWRITE, not a copy, since the tensors differ throughout. The target \
             is fully specified -- `dsv41_layer_tensors` is checked against all 96 085 shard \
             tensors in both directions -- so this is a long job, not an open question. Start from \
             `glm_emit_block`'s 61-line --block sibling",
            c.hc_mult, c.hc_sinkhorn_iters
        ),
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
    ]
}

/// How one weight divides across tensor-parallel ranks.
///
/// Every tensor is `[out, in]`. `Replicated` means each rank holds the whole thing; `OutSplit`
/// means rank `r` holds rows `[r*out/tp, (r+1)*out/tp)`; `InSplit` means it holds columns
/// `[r*in/tp, (r+1)*in/tp)` and the results are reduced afterwards.
///
/// Stated per tensor rather than inferred, because the inference is wrong for this model in two
/// places and both are silent. `wkv` LOOKS column-parallel and is not -- the absorbed latent is
/// one 512-wide row shared by all 64 heads, so every rank needs all of it (`num_key_value_heads`
/// is 1, and there is no `kv_b` to expand). `wq_a` looks shardable and is not -- the q-LoRA rank
/// is shared across heads, so the split can only happen at `wq_b`, after the rank.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum Shard {
    Replicated,
    OutSplit,
    InSplit,
}

/// The TP sharding of every weight this emit binds today.
///
/// Returns `None` for a tensor whose sharding has not been established. At `tp > 1` that is a
/// REFUSAL rather than a default: guessing `Replicated` wastes memory but guessing a split gives
/// every rank a fraction of a weight it needed whole, and the model merely gets worse.
pub(crate) fn dsv41_shard_of(suffix: &str) -> Option<Shard> {
    use Shard::*;
    Some(match suffix {
        // Norms are per-feature over the full hidden width; every rank normalises its own copy.
        "attn_norm.weight" | "ffn_norm.weight" | "attn.q_norm.weight" | "attn.kv_norm.weight" => {
            Replicated
        }
        // The q-LoRA DOWN projection. The rank (1280) is shared across heads, so there is nothing
        // head-shaped to split here; the head split happens at wq_b below.
        "attn.wq_a.weight" | "attn.wq_a.scale" => Replicated,
        // The q UP projection: [heads * head_dim, q_lora], column-parallel by head.
        "attn.wq_b.weight" | "attn.wq_b.scale" => OutSplit,
        // THE ABSORBED LATENT IS REPLICATED. One 512-wide row serves all 64 heads as both K and
        // V, so a rank holding 1/8 of it could not attend with the heads it owns. Section 7.
        "attn.wkv.weight" | "attn.wkv.scale" => Replicated,
        // The output LoRA. Block-diagonal over o_groups, and o_groups == 8 == tp, so rank r's
        // OutSplit rows [r*1024, (r+1)*1024) ARE group r's [1024, 4096] block -- the grouped
        // structure and the tensor-parallel split are the same cut.
        "attn.wo_a.weight" | "attn.wo_a.scale" => OutSplit,
        // wo_b reduces over every group's LoRA output, so it splits on the INPUT and the ranks
        // reduce afterwards.
        "attn.wo_b.weight" | "attn.wo_b.scale" => InSplit,
        // Shared expert: gate/up are column-parallel over the intermediate, down reduces over it.
        "ffn.shared_experts.w1.weight"
        | "ffn.shared_experts.w1.scale"
        | "ffn.shared_experts.w3.weight"
        | "ffn.shared_experts.w3.scale" => OutSplit,
        "ffn.shared_experts.w2.weight" | "ffn.shared_experts.w2.scale" => InSplit,
        // One sink per head, so it follows the heads.
        "attn.attn_sink" => OutSplit,
        // The ROUTER is replicated: every rank scores every expert, because a rank that saw only
        // its own 48 experts could not pick the global top-6. 384 * 5120 bf16 is 3.9 MB, so the
        // replication costs nothing worth splitting for.
        "ffn.gate.weight" | "ffn.gate.bias" | "ffn.gate.bias_vl" => Replicated,
        // The routed experts under the DEFAULT (TP) placement: gate/up column-parallel over
        // `moe_inter`, down reducing over it, exactly like the shared expert. `PLOW_MOE_PREFILL_EP`
        // would instead give each rank 384/8 = 48 WHOLE experts, which is a different answer for
        // these names -- but it is opt-in and OFF by default (`knob_spec.rs:762`), and this emit
        // does not implement it, so the default is what is encoded here.
        _ if is_routed_expert(suffix, "w1") || is_routed_expert(suffix, "w3") => OutSplit,
        _ if is_routed_expert(suffix, "w2") => InSplit,
        // mHC mixes the RESIDUAL stream, which is replicated -- restoring it is what the reduce
        // after `wo_b` is for. These are f32 and tiny; the largest is [mix, hc_mult, hidden].
        "hc_attn_fn" | "hc_attn_base" | "hc_attn_scale" | "hc_ffn_fn" | "hc_ffn_base"
        | "hc_ffn_scale" => Replicated,
        // The CSA2 compressor produces THE LATENT, and the latent is replicated for the same
        // reason `attn.wkv` is: one 512-wide row serves all 64 heads.
        "attn.compressor.wkv.weight"
        | "attn.compressor.norm.weight"
        | "attn.compressor.wgate.weight" => Replicated,
        // Indexer queries are per index-head (32 of them, 4 per rank at tp=8).
        "attn.indexer.wq_b.weight"
        | "attn.indexer.wq_b.scale"
        | "attn.indexer.weights_proj.weight" => OutSplit,
        // The index KEY is derived from the compressor's latent and is likewise one shared row.
        "attn.indexer.wk.weight" | "attn.indexer.k_norm.weight" => Replicated,
        // `engram.embed` CANNOT be replicated: the two tables are 202.8 GB and an MI300X has 192 GB
        // of HBM, so a replicated copy does not fit on one GPU at all. Row-split is the only
        // placement that fits (25.35 GB per rank), which makes this OutSplit by capacity rather
        // than by preference. `engram.wkv`/`q_weight`/`k_weight` are deliberately NOT listed: their
        // sharding follows from how the row gather exchanges between ranks, and that is not
        // designed yet. They live on layers 1 and 14 only, so layer 0 does not meet them.
        "engram.embed.weight" | "engram.embed.scale" => OutSplit,
        _ => return None,
    })
}

/// `ffn.experts.<e>.<w>.{weight,scale}` for a given `w`, for any expert index.
fn is_routed_expert(suffix: &str, w: &str) -> bool {
    let Some(rest) = suffix.strip_prefix("ffn.experts.") else {
        return false;
    };
    let Some((e, tail)) = rest.split_once('.') else {
        return false;
    };
    !e.is_empty()
        && e.bytes().all(|b| b.is_ascii_digit())
        && (tail == format!("{w}.weight") || tail == format!("{w}.scale"))
}

/// Every WEIGHT handle for a V4.1 layer, keyed by the checkpoint name without its
/// `layers.{l}.` prefix.
///
/// A map rather than a struct of named fields, which is the opposite of what
/// `declare_glm_rows_batched` does, and deliberate. GLM's declaration computes each shape inline,
/// so naming the fields is how it stays readable. V4.1's shapes come from
/// [`dsv41_layer_tensors`], which is already checked against all 96 085 shard tensors in both
/// directions -- so the list IS the contract, and a struct would be a second, hand-maintained copy
/// of it that can drift. Ask by the name the checkpoint uses.
///
/// This carries weights ONLY. Activation scratch is deliberately absent: its shapes follow the
/// emit's dataflow (what is fused, what is split per bucket, what survives a seam) rather than the
/// checkpoint, so declaring it before the emit exists would be guessing at the answer.
pub(crate) struct Dsv41Weights {
    /// `per_layer[l]` maps a suffix like `attn.wq_a.weight` to its tensor handle.
    per_layer: Vec<std::collections::BTreeMap<String, u32>>,
    /// Total declared bytes, which is what a capacity claim is checked against.
    pub(crate) bytes: u64,
}

impl Dsv41Weights {
    /// The handle for one weight, or a panic naming what was asked for.
    ///
    /// Panics rather than returning `Option` because every caller is an emit site that cannot
    /// proceed without the handle: a `TENSOR_NONE` fallback would bind a null pointer and the
    /// kernel would read it, which is the silent-wrongness shape this whole path avoids.
    pub(crate) fn get(&self, layer: u32, suffix: &str) -> u32 {
        *self.per_layer[layer as usize].get(suffix).unwrap_or_else(|| {
            panic!(
                "layer {layer} has no weight {suffix:?}. `dsv41_layer_tensors` decides what a \
                 layer carries and it is checked against the shards, so this is a typo or a \
                 tensor this layer genuinely does not have -- the optional groups (compressor, \
                 indexer keys, engram) are on specific layers only"
            )
        })
    }

    /// Whether a layer carries an optional weight, for the groups that are not on every layer.
    pub(crate) fn has(&self, layer: u32, suffix: &str) -> bool {
        self.per_layer[layer as usize].contains_key(suffix)
    }
}

/// Declare every weight the given layers bind, in checkpoint order.
///
/// The names carry their full `layers.{l}.` prefix into the tensor table, because that is what the
/// loader matches against the shards -- the suffix is only the lookup key on this side.
pub(crate) fn declare_dsv41_weights(
    b: &mut Builder,
    c: &Dsv41Cfg,
    layers: &[u32],
    tp: u32,
) -> Dsv41Weights {
    assert!(tp >= 1, "tp must be at least 1");
    let mut per_layer = vec![std::collections::BTreeMap::new(); c.layers as usize];
    let mut bytes = 0u64;
    for &l in layers {
        // The two tables the grouped-GEMM arms actually read. `bind_packed_experts` finds them by
        // the `expert_weight_table` / `expert_scale_table` suffix, takes the expert count from the
        // declared size (`bytes / 24`), resolves the checkpoint's own spelling, and PACKS every
        // expert's slice into one buffer -- so the 2304 per-expert checkpoint tensors must NOT be
        // declared. Declaring them would upload 1.7 GB per layer per rank that no op reads, beside
        // the packed copy that every op does.
        let ewt = b.tensor(
            &format!("layers.{l}.ffn.expert_weight_table"),
            (c.n_exp as u64) * 3 * 8,
        );
        let est = b.tensor(
            &format!("layers.{l}.ffn.expert_scale_table"),
            (c.n_exp as u64) * 3 * 8,
        );
        per_layer[l as usize].insert("ffn.expert_weight_table".to_string(), ewt);
        per_layer[l as usize].insert("ffn.expert_scale_table".to_string(), est);
        for (suffix, full) in dsv41_layer_tensors(c, l) {
            // `dsv41_layer_tensors` is the CHECKPOINT contract, checked against the shards at
            // full size. What a PACKET declares is this rank's share, which is a different
            // number whenever the tensor splits -- and the loader binds rank r's slice to it.
            let sz = if tp == 1 {
                full
            } else {
                match dsv41_shard_of(&suffix) {
                    Some(Shard::Replicated) => full,
                    Some(Shard::OutSplit) | Some(Shard::InSplit) => {
                        assert_eq!(
                            full % tp as u64,
                            0,
                            "layer {l} {suffix:?} is {full} bytes, which tp={tp} does not divide"
                        );
                        full / tp as u64
                    }
                    None => panic!(
                        "layer {l} {suffix:?} has no established TP sharding, so a tp={tp} \
                         declaration cannot size it. Guessing Replicated wastes memory; guessing \
                         a split hands every rank a fraction of a weight it needed whole and the \
                         model merely gets worse. Add it to `dsv41_shard_of`. Missing \
                         capability: `dsv41_shard_{suffix}`."
                    ),
                }
            };
            // `bytes` is the WEIGHT BUDGET -- what this rank must hold -- so the routed experts
            // count here even though they are not declared. They are resident either way; the
            // packed buffer is where, not whether.
            bytes += sz;
            if is_routed_expert(&suffix, "w1")
                || is_routed_expert(&suffix, "w2")
                || is_routed_expert(&suffix, "w3")
            {
                continue;
            }
            let h = b.tensor(&format!("layers.{l}.{suffix}"), sz);
            let prev = per_layer[l as usize].insert(suffix.clone(), h);
            assert!(
                prev.is_none(),
                "layer {l} declared {suffix:?} twice; `dsv41_layer_tensors` must not repeat a name"
            );
        }
    }
    Dsv41Weights { per_layer, bytes }
}

/// The activation scratch ONE layer's attention projection chain needs, and nothing else.
///
/// Declared next to the emit that uses it rather than up front with the weights, because these
/// shapes follow the DATAFLOW, not the checkpoint: `q_a` is `q_lora` wide because `wq_a` factors
/// the query, `kv` is 576 wide because the absorbed MLA shares one latent across all 64 heads.
/// A checkpoint-driven declaration cannot produce either number.
pub(crate) struct Dsv41ProjAct {
    /// Input after the pre-attention RMSNorm, `[T][hidden]` bf16.
    pub(crate) xn: u32,
    /// `x @ wq_a`, the query LoRA down-projection, `[T][q_lora]` bf16.
    pub(crate) q_a: u32,
    /// `q_a` after its own RMSNorm, `[T][q_lora]` bf16.
    pub(crate) q_an: u32,
    /// `q_an @ wq_b`, the query up-projection, `[T][heads * head_dim]` bf16. `head_dim`
    /// already contains the rope half (64 of its 512); there is no separate rope width to add.
    pub(crate) q: u32,
    /// `xn @ wkv`, the shared latent, `[T][head_dim]` bf16. ONE row per token for ALL 64
    /// heads -- the absorbed MLA's defining shape, and why num_key_value_heads is 1.
    pub(crate) kv: u32,
}

/// Emit one layer's attention PROJECTION chain: the four GEMMs that feed the attention core.
///
/// This is where op 184 earns its place. Every GEMM here reads `[32, 32]` ue8m0 block-fp8 weights,
/// and together with `wo_a`/`wo_b` they are **82.98 TFLOP of the 281.6 TFLOP 8k prefill** -- the
/// single largest term after the routed experts, and the reason the block-fp8 arm was item 0.
///
/// Returns the scratch plus the instruction id of the last op, so the caller can chain the
/// attention core onto it.
///
/// # What is NOT here
///
/// The attention core, the output projection, mHC, CSA2, Engram and the MoE. This emits the
/// projections and stops, because a chain that produces `q` and `kv` from real weights is
/// independently checkable -- against the shard shapes and against the opcode -- and a
/// half-written whole layer is not.
pub(crate) fn emit_dsv41_attn_proj(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    cus: &[u32],
    l: u32,
    tp: u32,
    x: u32,
    t: u32,
    deps: &[u32],
) -> (Dsv41ProjAct, u32) {
    let hidden = c.hidden;
    let q_lora = c.q_lora;
    assert_eq!(c.heads % tp, 0, "tp={tp} must divide {} heads", c.heads);
    // `head_dim` (512) ALREADY CONTAINS the rope half: qk_rope is 64 of it and nope is the other
    // 448. Adding qk_rope here would widen every projection by 64 per head against weights that
    // are not that shape -- the tensor list says `wq_b` is `heads * head_dim * q_lora` and `wkv`
    // is `head_dim * hidden`, and both are checked against the shards.
    //
    // PER-RANK, because `wq_b` is OutSplit and `declare_dsv41_weights` sized it that way. Reading
    // the full 64 heads against a per-rank declaration is the silent bug the operand check in
    // `emit_pf_gemm_fp8_mx` exists for: every rank would compute heads 0..63 out of rank 0's
    // shard. `wkv` is REPLICATED -- one 512-wide latent serves all 64 heads on every rank -- so
    // it does NOT divide.
    let q_out = c.heads / tp * c.head_dim;
    // ONE row for every head, which is what "fully absorbed" means -- there is no kv_b to expand
    // it, and num_key_value_heads is 1.
    let kv_out = c.head_dim;

    let act = Dsv41ProjAct {
        xn: b.tensor(&format!("act.l{l}.xn"), (t as u64) * (hidden as u64) * 2),
        q_a: b.tensor(&format!("act.l{l}.q_a"), (t as u64) * (q_lora as u64) * 2),
        q_an: b.tensor(&format!("act.l{l}.q_an"), (t as u64) * (q_lora as u64) * 2),
        q: b.tensor(&format!("act.l{l}.q"), (t as u64) * (q_out as u64) * 2),
        kv: b.tensor(&format!("act.l{l}.kv"), (t as u64) * (kv_out as u64) * 2),
    };

    let all = cus.to_vec();
    // eps is 1e-20 on this model, not the 1e-6 every other family uses. Reading it from the
    // config rather than writing a constant is the same rule the scale grid follows.
    let eps = c.eps;
    let c_xn = b.emit(DevOp::RmsNorm, all.clone(), deps, |d| {
        d.t[0] = act.xn;
        d.t[1] = x;
        d.t[2] = w.get(l, "attn_norm.weight");
        d.i[0] = t;
        d.i[1] = hidden;
        d.f[0] = eps;
    });

    let c_qa = emit_pf_gemm_fp8_mx(
        b,
        cus,
        act.q_a,
        act.xn,
        w.get(l, "attn.wq_a.weight"),
        w.get(l, "attn.wq_a.scale"),
        t,
        q_lora,
        hidden,
        &[c_xn],
    );
    let c_qan = b.emit(DevOp::RmsNorm, all.clone(), &[c_qa], |d| {
        d.t[0] = act.q_an;
        d.t[1] = act.q_a;
        d.t[2] = w.get(l, "attn.q_norm.weight");
        d.i[0] = t;
        d.i[1] = q_lora;
        d.f[0] = eps;
    });
    let c_q = emit_pf_gemm_fp8_mx(
        b,
        cus,
        act.q,
        act.q_an,
        w.get(l, "attn.wq_b.weight"),
        w.get(l, "attn.wq_b.scale"),
        t,
        q_out,
        q_lora,
        &[c_qan],
    );
    // The latent reads the NORMED input, not `q_an` -- it is a parallel branch off the same xn,
    // not a continuation of the query chain. Chaining it after `c_q` would serialise two GEMMs
    // that have no data dependence.
    let c_kv = emit_pf_gemm_fp8_mx(
        b,
        cus,
        act.kv,
        act.xn,
        w.get(l, "attn.wkv.weight"),
        w.get(l, "attn.wkv.scale"),
        t,
        kv_out,
        hidden,
        &[c_xn],
    );
    let _ = c_q;
    (act, c_kv)
}

/// The mHC residual stream and its scratch, shared by every layer.
///
/// V4.1's mHC is GLM-5.3's hyper-connection: the same `HyperConnPre`/`HyperConnPost` pair over the
/// same constants (`hc_mult` 4, 20 Sinkhorn iterations, `hc_eps` 1e-6), so this binds V4.1's
/// tensors to [`super::emit_mhc_pre`]/[`super::emit_mhc_post`] rather than emitting a second copy
/// of the algorithm.
///
/// The residual is a PING-PONG pair, not one buffer. Each sublayer reads one copy and the POST
/// writes the other, because `HyperConnPost` mixes the sublayer's output back across all `hc_mult`
/// copies -- writing in place would have it read rows it had already overwritten.
pub(crate) struct Dsv41Mhc {
    pub(crate) residual: [u32; 2],
    pub(crate) layer_input: u32,
    pub(crate) mixes: u32,
    pub(crate) post_mix: u32,
    pub(crate) comb_mix: u32,
}

/// Declare the mHC stream for `t` rows. `mix` is `(2 + hc_mult) * hc_mult`, the same derivation
/// `dsv41_layer_tensors` sizes `hc_*_fn` with -- one formula, so the weight and the scratch cannot
/// disagree about how many mix rows there are.
pub(crate) fn declare_dsv41_mhc(b: &mut Builder, c: &Dsv41Cfg, t: u32) -> Dsv41Mhc {
    let (r, n, w) = (t as u64, c.hc_mult as u64, c.hidden as u64);
    let mix = (2 + n) * n;
    Dsv41Mhc {
        residual: [
            b.tensor("act.hc_residual_a", r * n * w * 2),
            b.tensor("act.hc_residual_b", r * n * w * 2),
        ],
        layer_input: b.tensor("act.hc_layer_input", r * w * 2),
        mixes: b.tensor("act.hc_mixes", r * mix * 4),
        post_mix: b.tensor("act.hc_post_mix", r * n * 4),
        comb_mix: b.tensor("act.hc_comb_mix", r * n * n * 4),
    }
}

/// The mHC PRE half for one sublayer: collapse the residual copies into `m.layer_input`.
///
/// `ffn` picks which of the layer's TWO mHC weight sets to use. Every layer carries both
/// (`hc_attn_*` and `hc_ffn_*`), which is why there is no mHC-free block to extract from this
/// model and why the rung has to emit it before anything downstream is meaningful.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dsv41_mhc_pre(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    m: &Dsv41Mhc,
    l: u32,
    ffn: bool,
    ri: usize,
    t: u32,
    deps: &[u32],
) -> u32 {
    let side = if ffn { "ffn" } else { "attn" };
    let h = super::MhcHandles {
        fn_w: w.get(l, &format!("hc_{side}_fn")),
        base: w.get(l, &format!("hc_{side}_base")),
        scale: w.get(l, &format!("hc_{side}_scale")),
        residual: m.residual[ri],
        mixes: m.mixes,
        post_mix: m.post_mix,
        comb_mix: m.comb_mix,
        layer_input: m.layer_input,
    };
    super::emit_mhc_pre(
        b,
        &h,
        c.hidden,
        c.hc_mult,
        c.hc_sinkhorn_iters,
        c.eps,
        c.raw.hc_eps,
        t,
        deps,
    )
}

/// The mHC POST half: mix `raw` back across the residual copies, into the OTHER buffer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dsv41_mhc_post(
    b: &mut Builder,
    c: &Dsv41Cfg,
    m: &Dsv41Mhc,
    raw: u32,
    ri: usize,
    t: u32,
    deps: &[u32],
) -> u32 {
    super::emit_mhc_post(
        b,
        m.residual[ri ^ 1],
        raw,
        m.residual[ri],
        m.post_mix,
        m.comb_mix,
        c.hidden,
        c.hc_mult,
        t,
        deps,
    )
}

/// Emit V4.1's MoE: the router and the 384 MXFP4 routed experts.
///
/// # This calls GLM's prefill MoE body rather than growing a second one
///
/// The routed experts are 139.16 TFLOP of an 8k prefill -- 49.4% of the model, the largest single
/// block in the census -- so this is the part where a hand-rolled emit would cost the 90 ms target
/// outright. `emit_glm_moe_ffn_prefill` is 516 measured lines: `pick_tile` geometry, the align op's
/// padded row bound, lean segments, and the EP/TP placement rule. V4.1 wants all of it, unchanged.
///
/// What V4.1 does NOT have is a `declare_glm_rows_batched` -- 1182 lines whose tensors differ
/// throughout -- so the adaptation runs the other way: build the nine `GlmCfg` fields that body
/// reads, bind V4.1's weights into a `GlmLW`, fill the MoE scratch of a `GlmTn::none()`, and leave
/// every other handle absent. `TENSOR_NONE` rather than handle 0, so a field this turns out to
/// need is a loud missing operand rather than a silent read of somebody else's tensor.
///
/// The alternative -- parameterising the GLM body over its config and scratch -- was tried first
/// and abandoned: it hands `c` and `n` whole to `glm_sp`, `glm_shared_fold` and the tile pickers,
/// so it would have cascaded through the helpers of a shipped, measured path for no gain over
/// adapting the caller. The sizing that IS genuinely shared -- the padded gathered-row bound, where
/// a wrong answer is an out-of-bounds device write -- moved into `declare_moe_pf_scratch`, which
/// both families now call.
///
/// # The router
///
/// `sqrtsoftplus`, not sigmoid. Both are monotone, so a sigmoid emit would still select plausible
/// experts and only the gate weights would be wrong -- which is exactly how the bit-2 collision in
/// `op_moe.h` went unnoticed. The flags are built from the config's own `scoring_func`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dsv41_moe(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    l: u32,
    tp: u32,
    t: u32,
    x_out: u32,
    xn2: u32,
    c_norm: u32,
    // The shared expert's output tensor and completion dep. V4.1's shared expert is block-FP8
    // while these experts are MXFP4, so the caller emits it on op 184 and this only combines it.
    shared: (u32, u32),
    xgate: &mut u32,
    cus: &[u32],
) -> u32 {
    let shared_pre = Some(shared.1);
    assert_eq!(
        c.raw.scoring_func, "sqrtsoftplus",
        "the emit reads the score transform from the checkpoint; a new value needs a kernel arm, \
         not a substitution. Missing capability: `dsv41_router_scoring_{}`.",
        c.raw.scoring_func
    );
    assert_eq!(
        c.raw.topk_method, "noaux_tc",
        "selection bias is bound on the noaux_tc contract. Missing capability: \
         `dsv41_router_topk_{}`.",
        c.raw.topk_method
    );
    // Group-limited routing is the IDENTITY here: V4.1's config carries no `n_group`/`topk_group`
    // at all, so every expert is in play. `1`/`1` is how the GLM body spells that.
    let gc = super::GlmCfg::moe_only(
        c.hidden,
        c.n_exp,
        c.top_k,
        c.moe_inter,
        tp,
        false, // EP is opt-in and this emit does not implement it; see `dsv41_shard_of`.
        c.raw.routed_scaling_factor,
        1,
        1,
    );
    let mut lw = super::GlmLW::none();
    lw.wr = w.get(l, "ffn.gate.weight");
    lw.bias = w.get(l, "ffn.gate.bias");
    // The routed-expert weight and scale TABLES. The grouped GEMM resolves each expert's base from
    // these rather than taking 384 operands, which is why one handle stands for 384 tensors.
    // The pointer tables, NOT expert 0's weight. These are `[n_exp][3]` u64 device addresses that
    // `bind_packed_experts` fills at load; pointing them at a real checkpoint tensor would have the
    // grouped GEMM read fp4 mantissas as if they were pointers.
    lw.ewt = w.get(l, "ffn.expert_weight_table");
    lw.est = w.get(l, "ffn.expert_scale_table");
    // The shared expert is emitted separately (`emit_dsv41_ffn_shared`): it is block-FP8 on the
    // [32,32] grid while these are MXFP4, so it cannot fold into the routed table -- which is the
    // same mixed-encoding fact `glm_shared_fold` refuses on (`enc == Fp8Blk`).
    let mut n = super::GlmTn::none();
    n.lw = vec![lw];
    n.shared = shared.0;
    n.xn2 = xn2;
    // RAW OUTPUT: the body writes the FFN's own answer into `n.attn` and adds NO residual. That is
    // what an mHC layer needs -- `HyperConnPost` is what mixes this back into the residual stream,
    // and the ordinary path's `x_out = xmid + ffn` would add a second, wrong one on top of it. It
    // is not a cosmetic difference: `xmid` is the post-attention residual in GLM's layout and this
    // emitter has no such buffer, so the plain path read a tensor nothing had written.
    n.attn = x_out;
    n.rlogit = b.tensor(
        &format!("act.l{l}.rlogit"),
        (t as u64) * (c.n_exp as u64) * 2,
    );
    n.tab = b.tensor(&format!("act.l{l}.tab"), (t as u64) * (c.top_k as u64) * 8);
    if tp > 1 {
        // The combine's partial. `GlmTn::none()` leaves this TENSOR_NONE, and the body writes it
        // unconditionally at tp>1 (`emit_glm_moe_ffn_prefill`: `d.t[0] = n.dg_tp`), so leaving it
        // unset would aim every rank's routed output at the null handle.
        n.dg_tp = b.tensor(PEER_SLOT_MOE, (t as u64) * (c.hidden as u64) * 2);
    }
    // `fold` is false for V4.1 (it requires Fp8Blk experts), so e_all/tk_all are the plain counts.
    let sc = super::declare_moe_pf_scratch(
        b,
        t as u64,
        c.hidden,
        c.top_k,
        c.top_k,
        c.n_exp,
        c.moe_inter / tp,
        super::MoeEnc::Mxfp4,
    );
    n.part = sc.part;
    n.meta = sc.meta;
    n.row_token = sc.row_token;
    n.row_partidx = sc.row_partidx;
    n.row_gate = sc.row_gate;
    n.fu_g = sc.fu_g;
    n.fu_scale = sc.fu_scale;
    n.slot_b = (t as u64 * c.hidden as u64 * 2) as u32;
    super::emit_glm_moe_ffn_prefill(
        b,
        &gc,
        &n,
        0,
        t,
        super::MoeEnc::Mxfp4,
        x_out,
        c_norm,
        xgate,
        cus,
        true, // raw_output -- see `n.attn` above
        shared_pre,
        super::router_flag::SQRTSOFTPLUS
            | super::router_flag::BIAS
            | if c.raw.norm_topk_prob { super::router_flag::NORM_TOPK } else { 0 },
    )
}

/// Scratch for the attention core.
pub(crate) struct Dsv41CoreAct {
    /// `q` after the interior RoPE, `[T][nh_l][head_dim]` bf16.
    pub(crate) qr: u32,
    /// The shared latent after the interior RoPE, `[T][head_dim]` bf16.
    pub(crate) kvr: u32,
    /// Flash partials: `[T][nh_l][DK]` f32 and `[T][nh_l][2]` f32 at nsplit 1.
    pub(crate) opart: u32,
    pub(crate) mlpart: u32,
    /// The merged output, `[T][nh_l][head_dim]` bf16 -- which IS `wo_a`'s input.
    pub(crate) o: u32,
}

/// Emit the attention core: interior RoPE, the windowed absorbed-MLA flash, and the merge.
///
/// # Every operand here is already built
///
/// `head_dim` is 512 with the rope INSIDE it, so there is no separate rope strip and the flash runs
/// its NoPE arm over the whole 512 (`dsv41_attn_core_shape`). `d_flash_mla_prefill<512, 0>` is
/// instantiated, and `i[3]` carries both facts at once: bit 31 is NoPE and the low bits are the
/// sliding WINDOW, which is why layer 0 costs `T * 128` and not `T^2`. GLM passes `1 << 31` there
/// with a zero window, meaning full causal; V4.1 passes `1 << 31 | 128`.
///
/// So the core needs no gathered flash and no selection table. That is worth stating because the
/// obvious reading -- window attention is sparse attention, sparse attention is op 55 -- leads to
/// `FlashGatherPrefill` and its `[b][t][top_k]` idx array, which is a much larger job for the same
/// answer. A CONTIGUOUS causal window is what the dense arm's own `keep` predicate already is.
///
/// `t3`/`t5` repeat `t2`/`t4`: under NoPE the rope operands are unused, and GLM spells the same
/// aliasing (`if dr > 0 { n.qr } else { n.qa }`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dsv41_attn_core(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    cus: &[u32],
    l: u32,
    tp: u32,
    q: u32,
    kv: u32,
    kvlen: u32,
    pos: u32,
    cos: u32,
    sin: u32,
    t: u32,
    ctx: u32,
    deps: &[u32],
) -> (Dsv41CoreAct, u32) {
    let (hd, nope, rope) = dsv41_attn_core_shape(c);
    assert_eq!(
        c.heads % tp,
        0,
        "tp={tp} must divide the {} attention heads",
        c.heads
    );
    let nh_l = c.heads / tp;
    // The rope range must start on a wave boundary -- the kernel indexes a REGISTER by
    // `rot_offset / 64`, so an offset that is not a multiple of 64 would rotate a neighbouring
    // register's elements. 448 = 7 * 64 on this checkpoint.
    assert_eq!(
        nope % 64,
        0,
        "the interior rope starts at {nope}, which is not a register boundary; \
         `d_qwen_headnorm_rope` indexes x[rot_offset / 64]. Missing capability: \
         `rope_interior_unaligned_{nope}`"
    );
    assert!(
        rope <= 64 && rope.is_power_of_two(),
        "the rotary width {rope} must fit one register and be a power of two"
    );
    let act = Dsv41CoreAct {
        qr: b.tensor(
            &format!("act.l{l}.qr"),
            (t as u64) * (nh_l as u64) * (hd as u64) * 2,
        ),
        kvr: b.tensor(&format!("act.l{l}.kvr"), (t as u64) * (hd as u64) * 2),
        opart: b.tensor(
            &format!("act.l{l}.opart"),
            (t as u64) * (nh_l as u64) * (hd as u64) * 4,
        ),
        mlpart: b.tensor(
            &format!("act.l{l}.mlpart"),
            (t as u64) * (nh_l as u64) * 2 * 4,
        ),
        o: b.tensor(
            &format!("act.l{l}.attn_o"),
            (t as u64) * (nh_l as u64) * (hd as u64) * 2,
        ),
    };
    let all = cus.to_vec();
    // The interior RoPE, on q and on the shared latent. `normalize = 0`: q's norm was applied to
    // the q-LoRA before `wq_b` and the latent's by `kv_norm`, so this op only rotates.
    let mut rope_op = |b: &mut Builder, out: u32, x: u32, heads: u32, dep: &[u32]| {
        b.emit(DevOp::QwenHeadNormRope, all.clone(), dep, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = TENSOR_NONE;
            d.t[3] = cos;
            d.t[4] = sin;
            d.t[5] = pos;
            d.i[0] = heads;
            d.i[1] = hd;
            d.i[2] = rope;
            d.i[3] = t;
            d.i[4] = 0; // no cache ring: prefill writes row-major
            d.i[5] = 0; // normalize off
            d.i[6] = 1; // prefill
            d.i[7] = nope; // rotate [nope, head_dim) -- the SUFFIX
        })
    };
    let c_q = rope_op(b, act.qr, q, nh_l, deps);
    // ONE latent row per token, shared by every head: heads = 1, not nh_l.
    let c_kv = rope_op(b, act.kvr, kv, 1, deps);
    let scale = 1.0f32 / (hd as f32).sqrt();
    let window = c.sliding_window;
    let c_fl = b.emit(DevOp::FlashMlaPrefill, all.clone(), &[c_q, c_kv], |d| {
        d.t[0] = act.opart;
        d.t[1] = act.mlpart;
        d.t[2] = act.qr;
        d.t[3] = act.qr; // NoPE: the rope operands are unused
        d.t[4] = act.kvr;
        d.t[5] = act.kvr;
        d.t[6] = kvlen;
        d.i[0] = 1; // one sequence per prefill chunk
        d.i[1] = nh_l;
        d.i[2] = ctx;
        // Bit 31 NoPE + the sliding window in the low bits. A zero window here is FULL CAUSAL:
        // the same answer at 32x the arithmetic, which shows up only as a missed latency target.
        d.i[3] = (1u32 << 31) | window;
        d.i[4] = t;
        d.i[5] = KV_MASK_NONE;
        d.f[0] = scale;
    });
    // nsplit = 1, so the merge is the softmax normalisation plus the attention SINKS -- one extra
    // unscaled logit per head that joins the denominator with no value row. V4.1 has one per head
    // and `FlashMerge` is the only place it can be folded.
    let sink = w.get(l, "attn.attn_sink");
    let c_mg = b.emit(DevOp::FlashMerge, all.clone(), &[c_fl], |d| {
        d.t[0] = act.o;
        d.t[1] = act.opart;
        d.t[2] = act.mlpart;
        d.t[3] = sink;
        d.i[0] = t;
        d.i[1] = nh_l;
        d.i[2] = 1; // nsplit
        d.i[3] = hd;
    });
    (act, c_mg)
}

/// The o_proj TP partial's tensor name, fixed by `plowrt`'s peer-slot table (slot 0).
///
/// `crates/plowrt/src/exec/amd.rs` matches this string literally to bind the tensor into the peer
/// region instead of local VRAM. Renaming it here silently un-peers the buffer.
pub(crate) const PEER_SLOT_O: &str = "act.og_tp";
/// Peer slot 2, where the shared expert's down projection writes its partial. Slot 1
/// (`act.dg_tp`) is the routed combine's and slot 0 is attention's; see `emit_dsv41_ffn_shared`
/// for why this one cannot share either.
pub(crate) const PEER_SLOT_SHARED: &str = "act.ug_tp";
/// Peer slot 1, the routed-expert combine's partial. The GLM MoE body writes `GlmTn::dg_tp` and
/// reduces it, and a `TENSOR_NONE` there would make the combine write nothing at tp>1.
pub(crate) const PEER_SLOT_MOE: &str = "act.dg_tp";

/// Scratch for the output side: the per-group LoRA output and the projected residual.
pub(crate) struct Dsv41OutAct {
    /// This rank's output-LoRA group, `[T][o_lora_rank]` bf16.
    pub(crate) o_a: u32,
    /// This rank's PARTIAL `[T][hidden]` bf16. It is a partial, not an output: it is only group
    /// `r`'s contribution, and no rank's copy is the answer until the reduce runs.
    ///
    /// It MUST be declared as `act.og_tp`. The name is the contract: `plowrt` binds exactly six
    /// names into the TP peer region (`exec/amd.rs`'s `is_peer_slot`), `act.og_tp` being slot 0,
    /// the o_proj partial. Any other name is an ordinary device-local activation, and then every
    /// rank's `XReduce` sums peer slots its peers never wrote -- which that code's own comment
    /// describes as "a wrong token, with no fault and no message".
    pub(crate) o_part: u32,
    /// The projected residual contribution after the all-reduce, `[T][hidden]` bf16.
    pub(crate) o: u32,
}

/// Emit the attention OUTPUT projection: the grouped LoRA down-projection then `wo_b`.
///
/// # The grouped LoRA is free at TP8, and that is not a coincidence
///
/// `wo_a` is block-diagonal over `o_groups` -- `[8192, 4096]` is 8 stacked `[1024, 4096]`, and the
/// reference spells it `einsum("bsgd,grd->bsgr")`. A single dense GEMM CANNOT do that: a dense
/// `[T, 32768] x [8192, 32768]` would need a weight 8x larger than the one on disk, and the
/// missing 7/8 is exactly the block-diagonal saving.
///
/// But `heads / o_groups` is `64 / 8` = 8, so at **TP8 one rank owns exactly one group**: its
/// share of the heads is `o_group_in_features` = 4096 wide, its slice of `wo_a` is `[1024, 4096]`,
/// and the GEMM is ORDINARY. The group count and the tensor-parallel degree are the same number,
/// so the structure that would need 8 GEMMs at TP1 needs one per rank at TP8 -- which is the
/// configuration this model is served in.
///
/// At TP1 this refuses rather than pretending. Eight GEMMs would need eight weight handles, and
/// the tensor table binds a name to a whole checkpoint tensor -- there is no sub-tensor view -- so
/// a TP1 path needs either a grouped-GEMM opcode or a loader that can bind a slice. Neither
/// exists, and inventing a name like `wo_a.g3` would bind nothing and read zeros.
///
/// # `wo_b` reduces, it does not gather
///
/// Rank `r` produces group `r`'s `[T, 1024]`, which is exactly input columns
/// `[r*1024, (r+1)*1024)` of `wo_b` -- so `wo_b` is INPUT-parallel and the ranks sum afterwards.
/// The tempting alternative (all-gather the eight groups, then run the full `[5120, 8192]` GEMM on
/// every rank) is what this emit used to do and it is 8x the arithmetic: 687 GFLOP per rank per
/// layer instead of 86, which over 40 layers is 27.5 TFLOP against a 281.6 TFLOP budget -- a ~10%
/// inflation of the whole model's compute, bought for nothing. The reduce is GLM's
/// [`crate::emit_xreduce`], which already picks the bandwidth-optimal two-shot for prefill and the
/// latency-optimal one-shot for decode.
pub(crate) fn emit_dsv41_attn_out(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    cus: &[u32],
    l: u32,
    tp: u32,
    attn_out: u32,
    t: u32,
    xgate: &mut u32,
    deps: &[u32],
) -> (Dsv41OutAct, u32) {
    let (groups, orow, ocol) = c.wo_a_groups();
    assert_eq!(
        tp, groups,
        "the output LoRA is block-diagonal over {groups} groups and this emit maps ONE group to \
         one rank, so it needs tp == o_groups (tp is {tp}). At tp=1 the same weight is 8 separate \
         GEMMs over 8 slices of one checkpoint tensor, and the tensor table has no sub-tensor \
         view to express that -- it would need a grouped-GEMM opcode or a slicing loader. \
         Missing capability: `emit_dsv41_out_lora_tp{tp}`."
    );
    let act = Dsv41OutAct {
        o_a: b.tensor(&format!("act.l{l}.o_a"), (t as u64) * (orow as u64) * 2),
        // NOT per-layer: the name is fixed by the runtime's peer-slot table, and the reduce
        // consumes the partial in the same layer that writes it, so one buffer serves all 40.
        o_part: b.tensor(PEER_SLOT_O, (t as u64) * (c.hidden as u64) * 2),
        o: b.tensor(&format!("act.l{l}.o"), (t as u64) * (c.hidden as u64) * 2),
    };
    // This rank's group: [T, ocol] x [orow, ocol]^T -> [T, orow].
    let c_oa = emit_pf_gemm_fp8_mx(
        b,
        cus,
        act.o_a,
        attn_out,
        w.get(l, "attn.wo_a.weight"),
        w.get(l, "attn.wo_a.scale"),
        t,
        orow,
        ocol,
        deps,
    );
    // This rank's SLICE of wo_b: [T, orow] x [hidden, orow]^T -> a [T, hidden] PARTIAL.
    let c_ob = emit_pf_gemm_fp8_mx(
        b,
        cus,
        act.o_part,
        act.o_a,
        w.get(l, "attn.wo_b.weight"),
        w.get(l, "attn.wo_b.scale"),
        t,
        c.hidden,
        orow,
        &[c_oa],
    );
    let xr = super::xr_cus_capped(b.n_cu(), cus);
    let c_xr = crate::emit_xreduce(
        b,
        xgate,
        false,
        &xr,
        c_ob,
        act.o,
        t * c.hidden,
        tp,
        0,
    );
    (act, c_xr)
}

/// V4.1's GLU is the CLAMPED SwiGLU, `PLOW_ACT_SWIGLU_CLAMP_` in `op_elementwise.h`.
///
/// Not `GLM_ACT_SILU` (1), which every other family in this emitter uses. The clamp limit is a
/// config field (`swiglu_limit`, 10.0 on the released checkpoint) and rides `f[1]`; emitting act 1
/// here would drop the clamp silently, and the gfx942 test that covers this arm measured
/// 3160 of 4096 elements actually clamped, so the difference is not academic.
const DSV41_ACT_SWIGLU_CLAMP: u32 = 4;

/// Scratch for the FFN's pre-norm and the shared expert.
pub(crate) struct Dsv41FfnAct {
    /// Residual after the FFN RMSNorm, `[T][hidden]` bf16.
    pub(crate) xn: u32,
    /// Shared-expert gate and up, each `[T][moe_inter / tp]` bf16.
    pub(crate) sh_gate: u32,
    pub(crate) sh_up: u32,
    /// After the clamped SwiGLU, `[T][moe_inter / tp]` bf16.
    pub(crate) sh_act: u32,
    /// The down projection's PARTIAL, in peer slot 2, `[T][hidden]` bf16. At tp=1 nothing
    /// reduces it and `sh_out` is left untouched, so the caller reads this one.
    pub(crate) sh_part: u32,
    /// Shared-expert output after the cross-rank sum, `[T][hidden]` bf16.
    pub(crate) sh_out: u32,
}

/// Emit the FFN pre-norm and the SHARED expert (not the routed ones).
///
/// The shared expert is block-FP8 on the `[32, 32]` ue8m0 grid like every other dense projection,
/// which is why it is 23.19 TFLOP of the 8k prefill and why it lands here on op 184 rather than
/// with the routed experts. The ROUTED experts are MXFP4 on ops 85/86 -- a different fetch path
/// entirely, and the reason this checkpoint needs a mixed encoding at all.
///
/// Gate and up are emitted as TWO GEMMs plus a `Glu`, not one fused `GemmGlu`. There is no fused
/// block-fp8-at-32 GLU arm: `d_gemm_t<WFP8MX>` static_asserts `!GLU` outright, because the
/// promotion into a second accumulator every 32 K already costs what the fusion would save. The
/// GLM emitter measured the equivalent split at -0.017 ms, i.e. nothing.
pub(crate) fn emit_dsv41_ffn_shared(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    cus: &[u32],
    l: u32,
    tp: u32,
    x: u32,
    t: u32,
    xgate: &mut u32,
    deps: &[u32],
) -> (Dsv41FfnAct, u32) {
    let hidden = c.hidden;
    assert_eq!(
        c.moe_inter % tp,
        0,
        "tp={tp} must divide the shared expert's intermediate {}",
        c.moe_inter
    );
    // PER-RANK. `w1`/`w3` are OutSplit over `moe_inter` and `w2` is InSplit over it, so all three
    // are DECLARED at `inter / tp` and every intermediate activation is that wide too.
    let inter = c.moe_inter / tp;
    let act = Dsv41FfnAct {
        xn: b.tensor(&format!("act.l{l}.ffn_xn"), (t as u64) * (hidden as u64) * 2),
        sh_gate: b.tensor(&format!("act.l{l}.sh_gate"), (t as u64) * (inter as u64) * 2),
        sh_up: b.tensor(&format!("act.l{l}.sh_up"), (t as u64) * (inter as u64) * 2),
        sh_act: b.tensor(&format!("act.l{l}.sh_act"), (t as u64) * (inter as u64) * 2),
        // The DOWN projection's output is a PARTIAL over this rank's slice of the intermediate,
        // and it has to land where peers can read it. `d_xreduce` sums `peer_scratch[r] + slot`
        // over every rank and never reads `out`, so a partial written to an ordinary arena tensor
        // contributes nothing and the reduce returns the sum of whatever else is at that offset.
        // SLOT 2 (`act.ug_tp`), not slot 0: slot 0 is this same layer's attention partial, and
        // passing the attention reduce only proves every peer ARRIVED, not that every peer has
        // finished READING -- a fast rank would overwrite slot 0 under a slow peer. K3 reuses
        // slot 0 safely only because its shared expert is ordered after a SECOND collective
        // (`k3.rs:1609`); V4.1's shared expert runs before the MoE combine, so it cannot be.
        sh_part: if tp == 1 {
            // A group of one has nothing to reduce, and slot 2 does not exist at tp=1 (the host
            // maps no peer region), so the down projection writes the output directly.
            b.tensor(&format!("act.l{l}.sh_out"), (t as u64) * (hidden as u64) * 2)
        } else {
            b.tensor(PEER_SLOT_SHARED, (t as u64) * (hidden as u64) * 2)
        },
        sh_out: b.tensor(&format!("act.l{l}.sh_out"), (t as u64) * (hidden as u64) * 2),
    };
    let all = cus.to_vec();
    let eps = c.eps;
    let c_xn = b.emit(DevOp::RmsNorm, all.clone(), deps, |d| {
        d.t[0] = act.xn;
        d.t[1] = x;
        d.t[2] = w.get(l, "ffn_norm.weight");
        d.i[0] = t;
        d.i[1] = hidden;
        d.f[0] = eps;
    });
    // Gate and up read the SAME input and have no dependence on each other.
    let c_g = emit_pf_gemm_fp8_mx(
        b,
        cus,
        act.sh_gate,
        act.xn,
        w.get(l, "ffn.shared_experts.w1.weight"),
        w.get(l, "ffn.shared_experts.w1.scale"),
        t,
        inter,
        hidden,
        &[c_xn],
    );
    let c_u = emit_pf_gemm_fp8_mx(
        b,
        cus,
        act.sh_up,
        act.xn,
        w.get(l, "ffn.shared_experts.w3.weight"),
        w.get(l, "ffn.shared_experts.w3.scale"),
        t,
        inter,
        hidden,
        &[c_xn],
    );
    let limit = c.raw.swiglu_limit;
    let c_act = b.emit(DevOp::Glu, all.clone(), &[c_g, c_u], |d| {
        d.t[0] = act.sh_act;
        d.t[1] = act.sh_gate;
        d.t[2] = act.sh_up;
        d.i[0] = t * inter;
        d.i[1] = DSV41_ACT_SWIGLU_CLAMP;
        d.f[1] = limit;
    });
    let c_down = emit_pf_gemm_fp8_mx(
        b,
        cus,
        act.sh_part,
        act.sh_act,
        w.get(l, "ffn.shared_experts.w2.weight"),
        w.get(l, "ffn.shared_experts.w2.scale"),
        t,
        hidden,
        inter,
        &[c_act],
    );
    if tp == 1 {
        // `sh_part` IS `sh_out` here, so the answer is already in place.
        return (act, c_down);
    }
    let xr = super::xr_cus_capped(b.n_cu(), cus);
    let c_xr = crate::emit_xreduce(
        b,
        xgate,
        false,
        &xr,
        c_down,
        act.sh_out,
        t * hidden,
        tp,
        // Byte offset, not an index: the host binds `act.ug_tp` at `scratch_base + 2 * slot_b`
        // (`exec/amd.rs:8838`) and `slot_b` is the blob's `t * hidden * 2`.
        2 * t * hidden * 2,
    );
    (act, c_xr)
}

/// One sublayer a V4.1 rung has to emit, and whether it is emitted yet.
///
/// This exists so the answer to "what is missing" comes from RUNNING plowc rather than from
/// reading the source and hoping the prose kept up. `dsv41_gaps` describes the shape of the work;
/// this enumerates it per layer against the config, so a rung either emits or says precisely
/// which of its parts does not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Part {
    Done,
    Todo,
}

/// What the attention core needs, and the one thing in the way.
///
/// Read off the checkpoint, not the config prose: `attn.wkv.weight` is `[512, 5120]` and
/// `attn.wq_b.weight` is `[32768, 1280]` = 64 heads x 512. So both the query and the shared latent
/// are 512 wide, and `qk_rope_head_dim` 64 lives INSIDE that 512 (448 nope + 64 rope). There is no
/// separate `k_rot` tensor and no `kv_lora_rank` key in the config at all.
///
/// That is NOT DeepSeek-V3's shape, which is what makes this worth writing down. V3 and GLM carry
/// `kv_lora_rank` 512 as the latent AND a separate 64-wide rope strip, so their per-head query is
/// 576 and the kernel is `d_flash_mla_prefill<DK=512, DR=64>` -- `DR` being the EXTRA width
/// appended to K for scoring. V4.1 appends nothing.
///
/// Where the widths land:
///   * score over the full 512 (448 unrotated, 64 rotated)
///   * V is the whole 512 latent, so O is 512 per head -- which is exactly the 64*512/8 = 4096
///     that `wo_a` reads per group, and the arithmetic only closes if V is the full latent
///   * therefore `DK = 512, DR = 0`: the shipped `<512, 0>` NoPE arm, IF the rope is already applied
///
/// **The blocker is that "if", and it is a SUFFIX rope.** The reference rotates the last `rd` dims
/// everywhere -- `apply_rotary_emb(q[..., -rd:], ...)` at `inference/model.py:772`, and the same
/// `[..., -rd:]` slice for `kv` (706), the compressor latent (758) and the sliding-window pair
/// (1059, 1061). Nothing in this tree rotates an interior range: `d_headnorm_rope` is templated on
/// `HD` and pairs `(i, i + HD/2)` across the WHOLE head vector, and `d_qwen_headnorm_rope`, which
/// does carry a separate `rotary` count, applies it as the PREFIX (`if (lane < rotary)`) and copies
/// the rest through. V4.1 needs `[448, 512)` of a 512-wide row. Slicing the strip into its own
/// tensor is not available either -- the tensor table binds a name to a whole checkpoint tensor,
/// the same wall `emit_dsv41_out_lora_tp1` hits.
///
/// So this is one bounded kernel change -- a rotary OFFSET, turning `lane < rotary` into
/// `lane - off < rotary`, so the prefix form is `off = 0` and every existing blob is unchanged --
/// plus a gfx942 numerics run. It is NOT a new `<DK, DR>` instantiation: only `<512, 64>` and
/// `<512, 0>` are built and V4.1 wants one that already exists.
/// Missing capability: `rope_interior_range`.
///
/// The output side needs the same offset with the rotation INVERTED: `model.py:781` and `:1068`
/// run `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` on the attention output before the output
/// LoRA reads it. Op 181 already does inverse RoPE; it needs the same interior range.
///
/// Emitting `<512, 64>` here instead would read 64 bytes past every latent row and still produce
/// fluent output, which is the failure this rung refuses on purpose.
/// EVERY layer attends over a sliding window, and that is the cheapest fact in this model.
///
/// `inference/model.py:78` states it outright -- "every layer attends over a sliding window, and
/// may add compressed KV on top" -- with `window_size` 128 and a per-layer `compress_ratios` where
/// 0 means window-only. Layer 0 is window-only: `kv_source_layer_ids` is [2, 8, 14, 20].
///
/// So layer 0's attention is O(T * 128), not O(T^2). At 8k and tp=8 that is
/// `8192 * 128 * 8 heads * 512 * 4` = 17.2 GFLOP per rank, against 4.4 TFLOP for the full causal
/// form across the ranks -- roughly 32x less work. The attention core is NOT where the 90 ms goes;
/// the projections (82.98 TFLOP) and the experts (139.16 + 23.19) are.
///
/// And it needs no new kernel. `get_window_topk_idxs` (`model.py:410`) materialises exactly a
/// `[b, m, topk] int32` table -- "sparse_attn needs real [b, m, topk] int32 memory" -- which is
/// what [`DevOp::FlashGatherPrefill`] (op 55) takes in `t7`, with `top_k` in `i6`.
/// `d_flash_gather_prefill` is built and dispatched, and the `nope` branch is instantiated at
/// `<512, 0>` -- V4.1's geometry exactly, once the interior rope has been applied.
///
/// Op 55 had no emit site, and the reason was the SELECTOR, not the flash: a learned top-k needs
/// T-row `IndexScore`/`IndexSelect`, which `mla.rs`'s scoping note calls a real design question. A
/// WINDOW needs none of that -- row `t` is `clamp(t - 127, 0) ..= t`, pure arithmetic. It also
/// discharges the causality obligation that note flags, since the gather flash applies no mask and
/// trusts the selector: a window cannot name a future row, and `-1` marks a slot before the
/// sequence started.
///
/// (NOT op 51 with `t7` set. That slot on 51 is the DSA per-64-query-tile UNION table, which only
/// the four-wave V2 object understands; the eight-wave object traps on it rather than silently
/// running dense attention where the model was trained sparse.)
///
/// So what remains for the core is the index table, the emit around it, and the decode ring form
/// (`model.py:422`, oldest-first over a `start_pos % window` rotation) which prefill does not need.
pub(crate) fn dsv41_attn_core_shape(c: &Dsv41Cfg) -> (u32, u32, u32) {
    let nope = c.head_dim - c.qk_rope;
    (c.head_dim, nope, c.qk_rope)
}

/// Emit ONE V4.1 layer as a prefill rung: a tensor table, one program, and a descriptor.
///
/// The entry is `act.x`, uploaded by the harness, and the exit is whichever residual buffer the mHC
/// ping-pong left the answer in. There is no embed and no tail -- a rung is a validation artifact
/// for one layer, not a servable model.
///
/// Layer 0 only, for now. Every other layer adds a group this does not emit: a compressor
/// (2, 8, 14, 20), indexer queries (those plus 24, 28, 32, 36) or an Engram table (1, 14). The
/// refusal names which.
pub(crate) fn emit_dsv41_block(
    c: &Dsv41Cfg,
    l: u32,
    tp: u32,
    n_cu: u32,
    ctx: u32,
    t: u32,
) -> (crate::Model, plow_asset::BlockDescriptor) {
    let parts = dsv41_layer_parts(c, l);
    let todo: Vec<&str> = parts
        .iter()
        .filter(|(_, st)| *st == Part::Todo)
        .map(|(n, _)| *n)
        .collect();
    assert!(
        todo.is_empty(),
        "layer {l} cannot be emitted as a rung: {} of {} parts are missing -- {todo:?}. \
         Missing capability: `emit_dsv41_block_l{l}`.",
        todo.len(),
        parts.len()
    );
    let (hd, _nope, rope) = dsv41_attn_core_shape(c);
    let nh_l = c.heads / tp;

    let mut tb = Builder::new(n_cu);
    tb.set_tensor_dedup(true);
    let w = declare_dsv41_weights(&mut tb, c, &[l], tp);
    let x = tb.tensor("act.x", (t as u64) * (c.hidden as u64) * 2);
    let xnext = tb.tensor("act.xnext", (t as u64) * (c.hidden as u64) * 2);
    let pos = tb.tensor("in.pos", (ctx as u64) * 4);
    let kvlen = tb.tensor("in.kvlen", 4);
    // Layer 0 is pure sliding-window attention, and the reference disables YaRN there outright
    // (`model.py:686`: "disable YaRN and use base rope_theta in pure sliding-window attention").
    // Materialising a scaled table here would rotate every query by the wrong angle and still
    // produce fluent output.
    let [cos_t, sin_t] = packet::rope::GenTensor::rope_pair(
        ctx,
        rope,
        c.raw.rope_theta as f64,
        1.0,
        packet::rope::RopeScale::None,
    );
    let cos = tb.tensor_gen("in.cos", cos_t.byte_len(), cos_t);
    let sin = tb.tensor_gen("in.sin", sin_t.byte_len(), sin_t);
    let mhc = declare_dsv41_mhc(&mut tb, c, t);
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();

    let mut b = Builder::new(n_cu);
    b.set_tensor_dedup(true);
    b.adopt_tensors(tensors);
    let all = b.all();
    let mut xgate = 0u32;

    // The residual stream starts in copy 0. Each sublayer reads one copy and its POST writes the
    // other, which is why `ri` flips twice per layer -- once for attention, once for the FFN.
    let mut ri = 0usize;
    let c_seed = b.emit(DevOp::Residual, all.clone(), &[], |d| {
        d.t[0] = mhc.residual[0];
        d.t[1] = x;
        d.t[2] = x;
        d.i[0] = t * c.hidden;
        d.f[0] = 0.5; // (x + x) * 0.5 == x: seed every mHC copy from the block entry
    });

    let c_pre = emit_dsv41_mhc_pre(&mut b, c, &w, &mhc, l, false, ri, t, &[c_seed]);
    let (proj, c_proj) = emit_dsv41_attn_proj(&mut b, c, &w, &all, l, tp, mhc.layer_input, t, &[c_pre]);
    let (core, c_core) = emit_dsv41_attn_core(
        &mut b, c, &w, &all, l, tp, proj.q, proj.kv, kvlen, pos, cos, sin, t, ctx, &[c_proj],
    );
    let (_out, c_out) =
        emit_dsv41_attn_out(&mut b, c, &w, &all, l, tp, core.o, t, &mut xgate, &[c_core]);
    let c_post = emit_dsv41_mhc_post(&mut b, c, &mhc, _out.o, ri, t, &[c_out]);
    ri ^= 1;

    let c_pre2 = emit_dsv41_mhc_pre(&mut b, c, &w, &mhc, l, true, ri, t, &[c_post]);
    let (ffn, c_sh) = emit_dsv41_ffn_shared(
        &mut b, c, &w, &all, l, tp, mhc.layer_input, t, &mut xgate, &[c_pre2],
    );
    let c_moe = emit_dsv41_moe(
        &mut b, c, &w, l, tp, t, xnext, ffn.xn, c_sh, (ffn.sh_out, c_sh), &mut xgate, &all,
    );
    let _ = emit_dsv41_mhc_post(&mut b, c, &mhc, xnext, ri, t, &[c_moe]);
    ri ^= 1;

    let prog = b.finish();
    let out_name = if ri == 0 { "act.hc_residual_a" } else { "act.hc_residual_b" };
    let tensors = prog.tensors.clone();

    // A TRAILING, DELIBERATELY EMPTY DECODE PROGRAM.
    //
    // `derive_roles` reads a parent blob's roles POSITIONALLY: `decode_rung_lo` puts the boundary
    // at `len - 1`, so the last program is always a decode rung and a ONE-program blob has no
    // prefill bucket at all. The rung emitted above is prefill -- 1024 rows of it -- and a host
    // filtering on `role.is_prefill_bucket()` would find nothing to run.
    //
    // It is EMPTY rather than a copy of the prefill program or a guessed decode chain. V4.1 has no
    // decode emit yet: layer 0 is sliding-window over a KV ring nothing here allocates, and the
    // mHC residual is carried between calls by state this rung does not declare. `block.json` says
    // `decode_t: 0` to match, and `the_rung_states_it_cannot_decode` pins the pair, so a real
    // decode emit has to update both or fail.
    let mut db = Builder::new(n_cu);
    db.adopt_tensors(prog.tensors.clone());
    let decode = db.finish();
    assert!(decode.insts.is_empty(), "the decode placeholder must stay empty");

    let m = crate::Model {
        n_cu,
        target: 0,
        tensors,
        progs: vec![prog, decode],
        prog_t: vec![t, 1],
        kv_row_insts: Vec::new(),
        gen,
    };
    use plow_asset::*;
    let hidden = c.hidden as i64;
    let d = BlockDescriptor {
        model: "DeepSeek-V4.1-Flash".into(),
        arch: "mla_moe_swa".into(),
        layer: l,
        kind: vec!["mla_attn_swa".into(), "moe_ffn".into()],
        hidden,
        // The ACTIVATIONS are bf16. The weights are mixed -- block-FP8 dense and shared, MXFP4
        // routed -- which this field has no way to say, so it does not try to.
        dtype: "bf16".into(),
        dims: BlockDims {
            heads: Some(c.heads as i64),
            // The absorbed latent, which is `head_dim` here and NOT a separate kv_lora_rank:
            // V4.1's config has no such key and the 512 includes the 64 rope dims.
            kv_lora: Some(hd as i64),
            q_lora: Some(c.q_lora as i64),
            n_exp: Some(c.n_exp as i64),
            top_k: Some(c.top_k as i64),
            shared_exp: Some(1),
            moe_inter: Some(c.moe_inter as i64),
            ..Default::default()
        },
        // No DSA indexer on layer 0; `index_source_layer_ids` starts at 2.
        dsa_role: None,
        inputs: vec![BlockTensor {
            name: "act.x".into(),
            shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(hidden)],
            dtype: "bf16".into(),
        }],
        outputs: vec![BlockTensor {
            name: out_name.into(),
            shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(hidden)],
            dtype: "bf16".into(),
        }],
        // Layer 0 carries nothing between calls: pure sliding window, one chunk per run.
        carried_state: Vec::new(),
        weights: BlockWeights {
            mode: "symlink".into(),
            ckpt: "DeepSeek-V4.1-Flash".into(),
            prefix: format!("layers.{l}."),
        },
        programs: BlockPrograms {
            // Prefill only. There is no decode program: the rung exists to prove the prefill
            // chain, and a `decode_t: 1` here would advertise one that was never emitted.
            prefill_buckets: vec![t as i64],
            decode_t: 0,
        },
    };
    let _ = nh_l;
    (m, d)
}

/// The ops one layer needs, in dataflow order, each marked done or not.
///
/// Per LAYER rather than per model because the optional groups differ: only the kv_source layers
/// carry a compressor, only two carry Engram. A rung that picks its layer well can therefore be
/// completable long before the whole model is.
pub(crate) fn dsv41_layer_parts(c: &Dsv41Cfg, l: u32) -> Vec<(&'static str, Part)> {
    let mut p = vec![
        ("mhc_pre (ops 128/129)", Part::Done),
        ("attn_norm + q_a/q_b/wkv projections (op 184)", Part::Done),
    ];
    if c.kv_source.contains(&l) {
        p.push(("csa2 compressor (ops 180/181)", Part::Todo));
    }
    if c.index_source.contains(&l) {
        p.push(("indexer queries (two-level)", Part::Todo));
    }
    p.push((
        "attention core (interior rope + windowed absorbed MLA + sink merge)",
        Part::Done,
    ));
    p.push(("output projection wo_a + wo_b (op 184)", Part::Done));
    p.push(("output all-reduce (XReduce, wo_b is input-parallel)", Part::Done));
    p.push(("ffn_norm", Part::Done));
    if c.engram_layers.contains(&l) {
        p.push(("engram gate + embed (ops 182/183)", Part::Todo));
    }
    p.push(("moe router + routed experts (ops 85/86, MXFP4)", Part::Done));
    p.push(("shared expert (op 184 + clamped SwiGLU)", Part::Done));
    p.push(("mhc_post", Part::Done));
    p
}

/// Emit ONE layer as a ladder rung, or refuse naming exactly what is not emitted yet.
///
/// The target is `plowc --block L` / `PLOW_LAYERS=single:L`, which is gpuq's tier-3 bring-up path.
/// A rung is explicitly a VALIDATION artifact and not a serving model -- the same contract
/// `glm_emit_block` has -- so it may be narrower than the full emit, but it must never be WRONG:
/// a blob missing its attention core would load and run and produce fluent-looking garbage, which
/// is the exact failure this campaign has spent its time avoiding. So the refusal is the feature
/// until every part is done.
///
/// Returns `Ok` with the finished parts once there are none left, `Err` with a per-part report
/// otherwise.
pub(crate) fn dsv41_emit_block_plan(c: &Dsv41Cfg, l: u32) -> Result<Vec<&'static str>, String> {
    assert!(
        l < c.layers,
        "layer {l} is out of range for a {}-layer model",
        c.layers
    );
    let parts = dsv41_layer_parts(c, l);
    let todo: Vec<&str> = parts
        .iter()
        .filter(|(_, st)| *st == Part::Todo)
        .map(|(n, _)| *n)
        .collect();
    if todo.is_empty() {
        return Ok(parts.iter().map(|(n, _)| *n).collect());
    }
    let done: Vec<&str> = parts
        .iter()
        .filter(|(_, st)| *st == Part::Done)
        .map(|(n, _)| *n)
        .collect();
    Err(format!(
        "deepseek_v41 layer {l} cannot be emitted as a rung yet: {} of {} parts are done.\n\
         \n  emitted:\n{}\n  not emitted:\n{}\n\
         \nA rung is a validation artifact, not a serving model, so it MAY be narrower than the \
         full emit -- but it must not be wrong. A blob missing its attention core loads, runs, and \
         produces fluent-looking garbage, so this refuses instead of writing one.\n\
         The kernels are NOT the gap: ops 180/181, 182/183 and 184 all exist and pass on gfx942. \
         What is missing is the emit around them. Missing capability: `emit_dsv41_block`.",
        done.len(),
        parts.len(),
        done.iter()
            .map(|n| format!("    + {n}\n"))
            .collect::<String>(),
        todo.iter()
            .map(|n| format!("    - {n}\n"))
            .collect::<String>(),
    ))
}

/// One weight the emit binds: the checkpoint name (without the `layers.{l}.` prefix) and its
/// size in BYTES.
///
/// Bytes rather than elements because that is what `Builder::tensor` takes and what a shard
/// header can be checked against directly. The dtype is folded in here rather than carried
/// alongside: `F8_E4M3`, `F8_E8M0` and the nibble-packed `I8` expert weights are all 1 byte per
/// stored element, `BF16` is 2 and `F32` is 4, and conflating those is how a tensor table ends up
/// half the size it should be with no error until the load reads past a buffer.
pub(crate) type LayerTensor = (String, u64);

fn f8(name: &str, elems: u64) -> LayerTensor {
    (name.to_string(), elems)
}
fn bf16(name: &str, elems: u64) -> LayerTensor {
    (name.to_string(), elems * 2)
}
fn f32t(name: &str, elems: u64) -> LayerTensor {
    (name.to_string(), elems * 4)
}

/// Every tensor layer `l` carries, as `(name, bytes)`.
///
/// The OPTIONAL groups are the whole difficulty of this checkpoint and none of them are stated
/// directly by the config -- they are derived (`kv_source_layer_ids` gives a compressor,
/// `index_source_layer_ids` gives indexer queries, only the intersection gives indexer KEYS), and
/// each derivation is a place to read V4.1 as V4. `dsv41_layer_tensors_match_the_shards` asserts
/// the derivations reproduce what is actually on disk, layer by layer, for all 40.
pub(crate) fn dsv41_layer_tensors(c: &Dsv41Cfg, l: u32) -> Vec<LayerTensor> {
    let (hidden, heads, hd) = (c.hidden as u64, c.heads as u64, c.head_dim as u64);
    let q_lora = c.q_lora as u64;
    let (g, orow, ocol) = c.wo_a_groups();
    let (g, orow, ocol) = (g as u64, orow as u64, ocol as u64);
    // The ue8m0 scale grid comes from the CONFIG's own `weight_block_size` via nn-graph, not
    // from a constant here: V4.1 is [32, 32] where V4 was [128, 128], and a second copy of that
    // number in devgen is how the wrong one survives a checkpoint change.
    let blk = |r: u64, cdim: u64| {
        let [a, b] = c.raw.fp8_scale_shape(r as i64, cdim as i64);
        (a * b) as u64
    };
    let mxblk = |out: u64, inf: u64| {
        let [a, b] = c.raw.mxfp4_scale_shape(out as i64, inf as i64);
        (a * b) as u64
    };
    let mut t = vec![
        bf16("attn_norm.weight", hidden),
        bf16("ffn_norm.weight", hidden),
        // One sink per head, and F32 where plow's FlashMerge `t3` slot is bf16.
        f32t("attn.attn_sink", heads),
        bf16("attn.q_norm.weight", q_lora),
        bf16("attn.kv_norm.weight", hd),
        f8("attn.wq_a.weight", q_lora * hidden),
        f8("attn.wq_a.scale", blk(q_lora, hidden)),
        f8("attn.wq_b.weight", heads * hd * q_lora),
        f8("attn.wq_b.scale", blk(heads * hd, q_lora)),
        // The single latent: 512 wide, shared by every head. No kv_b exists.
        f8("attn.wkv.weight", hd * hidden),
        f8("attn.wkv.scale", blk(hd, hidden)),
        f8("attn.wo_a.weight", g * orow * ocol),
        f8("attn.wo_a.scale", blk(g * orow, ocol)),
        f8("attn.wo_b.weight", hidden * g * orow),
        f8("attn.wo_b.scale", blk(hidden, g * orow)),
    ];

    // mHC, on EVERY layer -- which is why there is no mHC-free block to extract.
    let mix = ((2 + c.hc_mult) * c.hc_mult) as u64;
    for side in ["attn", "ffn"] {
        t.push(f32t(&format!("hc_{side}_fn"), mix * c.hc_mult as u64 * hidden));
        t.push(f32t(&format!("hc_{side}_base"), mix));
        t.push(f32t(&format!("hc_{side}_scale"), 3));
    }

    // MoE. The routed experts are MXFP4 (two nibbles per byte, so the stored row is half as wide);
    // the SHARED expert is block-FP8, not fp4 -- one layer, two encodings.
    let inter = c.moe_inter as u64;
    t.push(bf16("ffn.gate.weight", c.n_exp as u64 * hidden));
    t.push(f32t("ffn.gate.bias", c.n_exp as u64));
    // The image-span routing bias: a SECOND bias, used for vl tokens (`noaux_tc_for_vl`).
    t.push(f32t("ffn.gate.bias_vl", c.n_exp as u64));
    for e in 0..c.n_exp as u64 {
        for w in ["w1", "w3"] {
            t.push(f8(&format!("ffn.experts.{e}.{w}.weight"), inter * hidden / 2));
            t.push(f8(&format!("ffn.experts.{e}.{w}.scale"), mxblk(inter, hidden)));
        }
        t.push(f8(&format!("ffn.experts.{e}.w2.weight"), hidden * inter / 2));
        t.push(f8(&format!("ffn.experts.{e}.w2.scale"), mxblk(hidden, inter)));
    }
    for w in ["w1", "w3"] {
        t.push(f8(&format!("ffn.shared_experts.{w}.weight"), inter * hidden));
        t.push(f8(&format!("ffn.shared_experts.{w}.scale"), blk(inter, hidden)));
    }
    t.push(f8("ffn.shared_experts.w2.weight", hidden * inter));
    t.push(f8("ffn.shared_experts.w2.scale", blk(hidden, inter)));

    // CSA2: only the kv_source layers own a compressor, and every other layer READS its cache.
    // BF16 where `model.py:446` declares f32 -- the shard wins (section 7).
    if c.kv_source.contains(&l) {
        t.push(bf16("attn.compressor.wkv.weight", hd * hidden));
        t.push(bf16("attn.compressor.norm.weight", hd));
        // Layer 20 runs at ratio 1, which is a plain projection with NO softmax gate -- so the
        // gate is not simply "every compressor has one".
        if c.raw.compressor_has_gate(l) {
            t.push(bf16("attn.compressor.wgate.weight", hd * hidden));
        }
    }

    // The two-level indexer. Queries on 8 layers; KEYS only where a compressor lives, because the
    // index key is derived from the compressor's latent.
    if c.index_source.contains(&l) {
        let (ih, idim) = (c.index_heads as u64, c.index_dim as u64);
        t.push(f8("attn.indexer.wq_b.weight", ih * idim * q_lora));
        t.push(f8("attn.indexer.wq_b.scale", blk(ih * idim, q_lora)));
        t.push(bf16("attn.indexer.weights_proj.weight", ih * hidden));
        if c.kv_source.contains(&l) {
            t.push(bf16("attn.indexer.wk.weight", idim * hd));
            t.push(bf16("attn.indexer.k_norm.weight", idim));
        }
    }

    // Engram, on 2 layers. `embed` is the 98 GB table; `q_weight`/`k_weight` are [hc_mult, hidden]
    // -- one row per hyper-connection copy, which is what op 182 takes as `qw`/`kw`.
    if let Some(rows) = c.raw.engram_rows(l) {
        let rows = rows as u64;
        let (ehd, ehe) = (c.raw.engram_head_dim as u64, c.raw.engram_n_heads as u64);
        t.push(f8("engram.embed.weight", rows * ehd));
        // The scale is ue8m0 at a 32-element block along the ROW, which is why `engram.embed.scale`
        // is [rows, 8] for a 256-wide row rather than [rows, 2] as V4's [128, 128] would give.
        t.push(f8("engram.embed.scale", rows * ehd.div_ceil(32)));
        let wkv_out = hidden * (c.hc_mult as u64 + 1);
        // The `wkv` GEMM consumes ALL the gathered rows flattened, not one head's worth: a token
        // fetches `(max_ngram - 1) * n_heads` = 24 rows of `head_dim`, so the input is 24 * 256 =
        // 6144. Sizing this as `n_heads * head_dim` gives 2048 and a tensor a third of the right
        // size, which is what `dsv41_layer_tensors_match_the_shards` caught.
        let cols = (c.raw.engram_max_ngram_size as u64 - 1) * ehe;
        let wkv_in = cols * ehd;
        t.push(f8("engram.wkv.weight", wkv_out * wkv_in));
        t.push(f8("engram.wkv.scale", blk(wkv_out, wkv_in)));
        t.push(bf16("engram.q_weight", c.hc_mult as u64 * hidden));
        t.push(bf16("engram.k_weight", c.hc_mult as u64 * hidden));
    }
    t
}
