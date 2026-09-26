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
        // list is CLOSED: op 198 / d_gemm_t<WFP8MX> reads the [{ob}, {ib}] ue8m0 grid, passes 12/12
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
             194/195 exist and dispatch)",
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
        // The routed experts: gate/up column-parallel over `moe_inter`, down reducing over it,
        // exactly like the shared expert. `PLOW_MOE_PREFILL_EP` gives each rank 384/8 = 48 WHOLE
        // experts at the full `moe_inter` instead -- but these arms are reached ONLY for the byte
        // budget (the routed experts are packed by `bind_packed_experts`, never declared as packet
        // tensors), and the budget is the same number either way: 48 * 2304 == 384 * 288. So one
        // answer serves both placements here, and the placement itself is chosen in the runtime.
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
        // THE INDEXER IS REPLICATED, and that is the tree's own standing recommendation rather
        // than this emit's preference. `ColumnParallelLinear` would give each rank 4 of the 32
        // index heads at tp=8, and `d_index_score_pf_row`'s A tile IS the head axis:
        //
        //     static_assert(HIc % 32 == 0, "the 32x32 MFMA A-tile is 32 index heads; HIc must be
        //                                   a multiple of it (replicate a sharded indexer
        //                                   instead)")   [DSV4-IDX]
        //
        // 4 heads would leave seven eighths of the tile idle, and the alternative -- sharding and
        // all-reducing `index_score` -- is a `[T][compress_len]` f32 reduce: 268 MB per layer at
        // 8k and ratio 1, on 8 layers. Replicating costs 5.2 MB of `wq_b` and 328 KB of
        // `weights_proj` per rank and no collective at all. The ARITHMETIC is the reference's
        // either way; only the head-sum order differs.
        //
        // The next step past this is to split the QUERY ROWS instead -- one eighth of the score
        // work per rank and a `[T][512]` i32 all-gather, 16 MB -- which is not attempted here and
        // is recorded rather than guessed at.
        "attn.indexer.wq_b.weight"
        | "attn.indexer.wq_b.scale"
        | "attn.indexer.weights_proj.weight"
        // The index KEY is derived from the compressor's latent and is likewise one shared row.
        | "attn.indexer.wk.weight"
        | "attn.indexer.k_norm.weight" => Replicated,
        // `engram.embed` CANNOT be replicated: the two tables are 202.8 GB and an MI300X has 192 GB
        // of HBM, so a replicated copy does not fit on one GPU at all. Row-split is the only
        // placement that fits (25.35 GB per rank), which makes this OutSplit by capacity rather
        // than by preference. `ParallelEngramEmbedding` shards exactly this way and no other way
        // (`model.py:296-325`): `part_num_embeddings = ceil(rows / world_size)` -- 384 006 168 / 8
        // = 48 000 771, no ragged last shard -- `vocab_start = rank * part`, an out-of-shard id
        // contributes zeros, and the ranks are summed by an `all_reduce` of the VALUES.
        "engram.embed.weight" | "engram.embed.scale" => OutSplit,
        // ...and everything downstream of that all-reduce is REPLICATED, which the reference
        // settles rather than leaves open: `self.wkv = Linear(...)` (`model.py:345`) is the plain
        // class, not `ColumnParallelLinear` or `RowParallelLinear`, and `q_weight`/`k_weight` are
        // plain `nn.Parameter`s of `[hc_mult, dim]`. Replicated costs 157 MB per rank per engram
        // layer for `wkv` and 40 KB for the gate weights -- nothing against the 12.29 GB table
        // slice beside it. An earlier note here said this "follows from how the row gather
        // exchanges between ranks, and that is not designed yet"; it does not, because the
        // exchange is the embed's own all-reduce and it happens before `wkv` ever runs.
        "engram.wkv.weight" | "engram.wkv.scale" | "engram.q_weight" | "engram.k_weight" => {
            Replicated
        }
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
            &format!("layers.{l}.ffn.expert_weight_table{}", prof_table_suffix()),
            (c.n_exp as u64) * 3 * 8,
        );
        let est = b.tensor(
            &format!("layers.{l}.ffn.expert_scale_table{}", prof_table_suffix()),
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
/// This is where op 198 earns its place. Every GEMM here reads `[32, 32]` ue8m0 block-fp8 weights,
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

/// CSA2's write side: the compressed-KV cache the whole chain shares, and one source's scratch.
pub(crate) struct Dsv41Compress {
    /// THE SHARED CACHE, `[ctx][head_dim]` bf16, ONE buffer for the whole chain. In the reference
    /// it is a module-level global -- `shared_attn.compress_kv` (`model.py:745`) -- written by
    /// whichever `kv_source` layer ran last and read by every layer after it until the next one
    /// overwrites it. Sized at `ctx` rows because layer 20 compresses at ratio 1; the ratio-2
    /// sources use the prefix.
    pub(crate) cache: u32,
    /// `xn @ compressor.wkv^T`, `[T][head_dim]` bf16.
    kv: u32,
    /// `xn @ compressor.wgate^T`, `[T][head_dim]` bf16. Unused at ratio 1, which has no gate.
    gate: u32,
    /// The post-norm, PRE-RoPE latent, `[ctx][head_dim]` bf16 -- what the INDEXER reads, and the
    /// reason op 194 stops after the norm. Separate from `cache` because op 199 must not
    /// overwrite what the indexer still has to read.
    pub(crate) latent: u32,
}

/// Whether any layer in the chain touches the shared compressed cache -- as a WRITER
/// (`kv_source`) or as a READER (`compress_ratio != 0`). The two disagree by construction, and
/// declaring on the writer list alone is what left a reader chain without a cache to read.
fn dsv41_chain_uses_compress(c: &Dsv41Cfg, layers: &[u32]) -> bool {
    layers.iter().any(|&l| {
        c.kv_source.contains(&l)
            || !matches!(
                c.raw.attn_kind(l),
                nn_graph::models::config::V41Attn::Window
            )
    })
}

/// Declare the compressor's scratch. `None` when no layer in the chain writes OR reads the cache.
pub(crate) fn declare_dsv41_compress(
    b: &mut Builder,
    c: &Dsv41Cfg,
    layers: &[u32],
    t: u32,
    ctx: u32,
) -> Option<Dsv41Compress> {
    if !dsv41_chain_uses_compress(c, layers) {
        return None;
    }
    let (r, hd, cx) = (t as u64, c.head_dim as u64, ctx as u64);
    Some(Dsv41Compress {
        cache: b.tensor("act.compress_kv", cx * hd * 2),
        kv: b.tensor("act.compress_wkv", r * hd * 2),
        gate: b.tensor("act.compress_wgate", r * hd * 2),
        latent: b.tensor("act.compress_latent", cx * hd * 2),
    })
}

/// Emit ONE `kv_source` layer's compressor: the pooled latent, and the cache row it becomes.
///
/// # Two shapes, because `compress_ratio` is not one number
///
/// `kv_source_layer_ids` is `[2, 8, 14, 20]` and `compress_ratios` gives 2, 2, 2 and **1**. At
/// ratio 1 `Compressor.forward` returns on its first line -- `self.norm(self.wkv(x))`, no gate, no
/// fp32, no pooling (`model.py:461-462`) -- so layer 20 is a plain GEMM plus an RMSNorm and op 194
/// must not run at all. `compressor_has_gate` is what the tensor table already keys the `wgate`
/// weight on, and this uses the same predicate rather than a second reading of the ratio.
///
/// # The pooled form
///
/// Op 194 with `coff = 1` and a NULL `ape`. V4.1 has neither the overlap transform nor the
/// position-in-block bias -- `scripts/dsv41_csa2_oracle.py` check [1] puts op 194 at max err
/// 0.000e+00 against the reference on those terms, and check [2] shows a nonzero `ape` is worth
/// 1.65, so omitting it is a claim and not a formality.
///
/// `i7 = 2` stops the op after the norm. The rope and the fake quant are op 199, because
/// `Compressor.forward` returns the latent BEFORE RoPE for the indexer's sake and `_compress_kv`
/// finishes it afterwards (`model.py:751-761`).
///
/// # What it does NOT emit
///
/// The ragged tail. `n_pools` is `t / ratio` and `model.py:331-341` stashes the `seqlen % ratio`
/// leftover tokens into `kv_state` for the next call -- a decode concern, and a prefill rung is
/// one chunk. The emit refuses a `t` the ratio does not divide rather than silently dropping the
/// remainder, because a dropped tail is invisible in the output.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dsv41_compressor(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    cus: &[u32],
    cp: &Dsv41Compress,
    l: u32,
    xn: u32,
    cos: u32,
    sin: u32,
    t: u32,
    deps: &[u32],
) -> u32 {
    let hd = c.head_dim;
    let hidden = c.hidden;
    let ratio = match c.raw.attn_kind(l) {
        nn_graph::models::config::V41Attn::Compressed { ratio } => ratio,
        nn_graph::models::config::V41Attn::Window => {
            panic!("layer {l} is a kv_source with compress_ratio 0, which cannot compress")
        }
    };
    assert_eq!(
        t % ratio,
        0,
        "t={t} leaves a {}-token tail at ratio {ratio}, and `Compressor` carries that tail into \
         the NEXT call through `kv_state` (model.py:331-341) rather than dropping it. A prefill \
         rung is one chunk, so the emit refuses instead of losing it silently.",
        t % ratio
    );
    let all = cus.to_vec();
    let n_pools = t / ratio;

    // bf16 weights -- `dsv41_layer_tensors` declares the compressor in bf16 because that is what
    // the SHARD holds, whatever `model.py:446` promotes to f32 in the pooling.
    let bf16_gemm = |b: &mut Builder, out: u32, x: u32, weight: u32, n: u32, dep: &[u32]| {
        let op = crate::pick_tile(t, n, hidden, b.n_cu(), kernelcaps::QuantScheme::None);
        b.emit(op, all.clone(), dep, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = weight;
            d.i[0] = t;
            d.i[1] = n;
            d.i[2] = hidden;
        })
    };

    let c_lat = if ratio == 1 {
        // Layer 20. No gate, no pool: `self.norm(self.wkv(x))` and nothing else.
        let c_kv = bf16_gemm(
            b,
            cp.kv,
            xn,
            w.get(l, "attn.compressor.wkv.weight"),
            hd,
            deps,
        );
        b.emit(DevOp::RmsNorm, all.clone(), &[c_kv], |d| {
            d.t[0] = cp.latent;
            d.t[1] = cp.kv;
            d.t[2] = w.get(l, "attn.compressor.norm.weight");
            d.i[0] = t;
            d.i[1] = hd;
            d.f[0] = c.eps;
        })
    } else {
        assert!(
            c.raw.compressor_has_gate(l),
            "layer {l} pools at ratio {ratio} but has no `wgate`; the softmax gate is what pooling \
             IS"
        );
        let c_kv = bf16_gemm(
            b,
            cp.kv,
            xn,
            w.get(l, "attn.compressor.wkv.weight"),
            hd,
            deps,
        );
        let c_gt = bf16_gemm(
            b,
            cp.gate,
            xn,
            w.get(l, "attn.compressor.wgate.weight"),
            hd,
            deps,
        );
        b.emit(DevOp::CompressPool, all.clone(), &[c_kv, c_gt], |d| {
            d.t[0] = cp.latent;
            d.t[1] = cp.kv;
            d.t[2] = cp.gate;
            d.t[3] = TENSOR_NONE; // no `ape`: V4.1's Compressor has no such parameter
            d.t[4] = w.get(l, "attn.compressor.norm.weight");
            d.t[5] = TENSOR_NONE; // no cos/sin: arm 2 stops before the rope
            d.t[6] = TENSOR_NONE;
            d.t[7] = TENSOR_NONE; // prefill: the output slot comes from i6, not from a step
            d.i[0] = n_pools;
            d.i[1] = ratio;
            d.i[2] = 1; // coff: no overlap transform on V4.1
            d.i[3] = hd;
            d.i[4] = 0; // rd unused on arm 2
            d.i[5] = 0; // qblk unused on arm 2
            d.i[6] = 0; // out_base: one chunk, starting at position 0
            d.i[7] = 2; // pool, norm, STOP
            d.f[0] = c.eps;
        })
    };

    // The cache row: rope the last `qk_rope` at the group's FIRST token, then fp4 e2m1 at blocks
    // of 16 with an E4M3 scale. Both numbers are the reference's own call
    // (`fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)`, model.py:760) and the
    // scale format is worth 15.3% of the latent's amax, not a rounding detail.
    b.emit(DevOp::CompressRopeQuant, all.clone(), &[c_lat], |d| {
        d.t[0] = cp.cache;
        d.t[1] = cp.latent;
        d.t[2] = cos;
        d.t[3] = sin;
        d.t[4] = TENSOR_NONE;
        d.i[0] = n_pools;
        d.i[1] = hd;
        d.i[2] = c.qk_rope;
        d.i[3] = DSV41_KV_QBLK;
        d.i[4] = ratio;
        d.i[5] = 0;
        d.i[6] = 2; // PLOW_CMP_Q_FP4_E4M3
    })
}

/// `fp4_act_quant(latent, 16, ...)` -- the compressed cache's quant block, model.py:760.
pub(crate) const DSV41_KV_QBLK: u32 = 16;

/// The two-level indexer's scratch, and the two tables it shares across the chain.
pub(crate) struct Dsv41Index {
    /// THE SHARED INDEX KEYS, `[ctx][index_head_dim]` bf16. `shared_attn.index_k`
    /// (`model.py:548`) -- only the four `kv_source` layers derive keys, from their own
    /// compressor latent, and the other four index layers read whichever was written last.
    pub(crate) keys: u32,
    /// THE SHARED SELECTION, `[T][index_topk]` i32. `shared_attn.topk_idxs`
    /// (`model.py:731`): an index layer publishes it and every layer up to the next index
    /// layer reuses it without running an indexer at all (`_compress_topk_idxs`, 722-731).
    pub(crate) idx: u32,
    /// `qr @ indexer.wq_b^T`, then roped and fake-quantized in place of itself:
    /// `[T][index_n_heads][index_head_dim]` bf16.
    q: u32,
    qr: u32,
    /// `xn @ indexer.weights_proj^T`, `[T][index_n_heads]` bf16 -- the per-head ReLU weights.
    w: u32,
    /// `latent @ indexer.wk^T` and its norm, `[ctx][index_head_dim]` bf16.
    k: u32,
    kn: u32,
    /// `[T][ctx/min_ratio]` f32. The widest score a layer in this chain needs.
    score: u32,
    /// THE PER-PACK UNION (op 119), and the `u64[n_qt][kv_stride]` membership scratch it builds
    /// it through. This is what lets the gathered pass run the V2 arm instead of the scalar one:
    /// 8 queries share one staged KV slab and carry a 64-bit mask per row (8 queries x 8 heads),
    /// so the latent is read once per PACK rather than once per query.
    uni: u32,
    umask: u32,
}

/// Queries per union pack. Fixed by the kernel (`op_attention_common.h`, `QP`), which puts the
/// pack's 8 queries x 8 heads on the MFMA M dimension and therefore requires `n_head == 8`.
pub(crate) const DSV41_UNION_PACK: u32 = 8;

/// The union table's per-pack row capacity. A pack cannot name more rows than its 8 queries
/// select between them, nor more than the cache holds.
pub(crate) fn dsv41_union_cap(topk: u32, kv_rows: u32) -> u32 {
    (DSV41_UNION_PACK * topk).min(kv_rows)
}

/// Declare the indexer's scratch. `None` when no layer in the chain runs an indexer or reads a
/// selection -- the same writer/reader split the cache has.
pub(crate) fn declare_dsv41_index(
    b: &mut Builder,
    c: &Dsv41Cfg,
    layers: &[u32],
    t: u32,
    ctx: u32,
) -> Option<Dsv41Index> {
    if !dsv41_chain_uses_compress(c, layers) {
        return None;
    }
    let (r, cx) = (t as u64, ctx as u64);
    let (hi, di) = (c.index_heads as u64, c.index_dim as u64);
    Some(Dsv41Index {
        keys: b.tensor("act.index_k", cx * di * 2),
        idx: b.tensor("act.index_idx", r * c.index_topk as u64 * 4),
        q: b.tensor("act.index_q", r * hi * di * 2),
        qr: b.tensor("act.index_qr", r * hi * di * 2),
        w: b.tensor("act.index_w", r * hi * 2),
        k: b.tensor("act.index_kpre", cx * di * 2),
        kn: b.tensor("act.index_kn", cx * di * 2),
        // The score is [T][pools] and `pools` is widest at the SMALLEST ratio in the chain.
        score: b.tensor("act.index_score", r * cx * 4),
        // Sized for the WIDEST case in the chain, like `score`: ratio 1, so `kv_rows == ctx`.
        // Layout (dev_isa.h op 119): u32 count[n_qt], 256 B aligned, then per pack
        // [cap i32 pos][cap u32 maskLo][cap u32 maskHi].
        uni: b.tensor(
            "act.index_uni",
            {
                let n_qt = (r + DSV41_UNION_PACK as u64 - 1) / DSV41_UNION_PACK as u64;
                let cap = dsv41_union_cap(c.index_topk, ctx) as u64;
                (n_qt * 4).next_multiple_of(256) + n_qt * cap * 12
            },
        ),
        umask: b.tensor(
            "act.index_umask",
            ((r + DSV41_UNION_PACK as u64 - 1) / DSV41_UNION_PACK as u64) * cx * 8,
        ),
    })
}

/// Emit ONE `index_source` layer's indexer: keys where it owns them, then the score and the
/// top-k the 38 reader layers select their compressed positions with.
///
/// # This is the selector op 55 was waiting for
///
/// [`DevOp::FlashGatherPrefill`]'s own doc: *"What kept this without an emit site is the
/// SELECTOR, not the flash: a learned top-k needs T-row `IndexScore`/`IndexSelect`, which is a
/// real design problem."* Ops 117/118 are that pair, built for GLM-5.3's lightning indexer, and
/// V4.1's is the same computation at the same `index_head_dim` and the same 32 heads --
/// `scripts/dsv41_indexer_oracle.py` check [1] puts op 117 at 1.1e-07 relative against
/// `Indexer.forward`'s own expression.
///
/// # Three differences, and only one needed a kernel change
///
///   1. **The columns are POOLS.** V4.1's index keys are compressed entries, so a query reaches
///      `(t + 1) / ratio` of them and not `t + 1`. That is op 117's new `i4 = pool_size` and op
///      118's existing `i3`, and the two MUST agree. Check [2].
///   2. **The indexer is REPLICATED**, not column-parallel. See `dsv41_shard_of`: the MFMA A
///      tile is the head axis and a tp=8 shard would leave it one-eighth full, which is what
///      `d_index_score_pf_row`'s own `static_assert` tells an emitter to avoid.
///   3. **The two-level candidate stage is SKIPPED**, and that is a statement about this
///      context length rather than about the model. `candidate_topk_blocks` is 2048 over
///      `candidate_block_size` 8 = 16384 compressed positions, and 8k gives 8192 at ratio 1.
///      Level one therefore keeps every reachable block, and the mask level two applies is a
///      block-rounded SUPERSET of the reachability mask `Indexer.forward` has already applied
///      -- so it removes nothing. Checks [3] and [4], and the assert below is what stops this
///      being silently wrong at a longer context.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dsv41_indexer(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    cus: &[u32],
    ix: &Dsv41Index,
    cp: Option<&Dsv41Compress>,
    l: u32,
    xn: u32,
    qr: u32,
    kvlen: u32,
    cos: u32,
    sin: u32,
    t: u32,
    ctx: u32,
    deps: &[u32],
) -> u32 {
    let ratio = match c.raw.attn_kind(l) {
        nn_graph::models::config::V41Attn::Compressed { ratio } => ratio,
        nn_graph::models::config::V41Attn::Window => {
            panic!("layer {l} is an index_source with compress_ratio 0")
        }
    };
    let (hi, di) = (c.index_heads, c.index_dim);
    let rd = c.qk_rope;
    let n_pool = t / ratio;
    let blocks = (ctx / ratio).div_ceil(c.raw.candidate_block_size as u32);
    assert!(
        blocks <= c.raw.candidate_topk_blocks as u32,
        "ctx={ctx} at ratio {ratio} gives {blocks} candidate blocks against \
         candidate_topk_blocks={}, so `select_candidate_blocks` would actually DROP blocks and \
         this emit's decision to skip the two-level selection stops being a no-op. \
         Missing capability: `dsv41_indexer_two_level`.",
        c.raw.candidate_topk_blocks
    );
    // `topk = min(self.index_topk, end_pos // ratio)` (model.py:578). A chunk with fewer pools
    // than `index_topk` selects all of them, and the WIDTH the table is written at is what op 55
    // then reads, so the two must be derived the same way -- `dsv41_index_topk` is that one
    // derivation.
    let topk = dsv41_index_topk(c, t, ratio);
    let all = cus.to_vec();

    let bf16_gemm = |b: &mut Builder, out: u32, x: u32, weight: u32, n: u32, k: u32, dep: &[u32]| {
        let op = crate::pick_tile(t, n, k, b.n_cu(), kernelcaps::QuantScheme::None);
        b.emit(op, all.clone(), dep, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = weight;
            d.i[0] = t;
            d.i[1] = n;
            d.i[2] = k;
        })
    };

    // ---- the index KEYS, on the four layers that own a compressor --------------------------
    // "the index keys are derived from the compressor's latent, so only a layer that compresses
    // its own KV can produce them; every other indexer reads them from that layer's cache"
    // (model.py:498-500). And they are derived from the PRE-RoPE latent, which is the whole
    // reason op 194 stops after the norm.
    let mut key_dep: Vec<u32> = Vec::new();
    if c.kv_source.contains(&l) {
        let cp = cp.expect("a kv_source layer has a compressor");
        let c_k = {
            let op = crate::pick_tile(n_pool, di, c.head_dim, b.n_cu(), kernelcaps::QuantScheme::None);
            b.emit(op, all.clone(), deps, |d| {
                d.t[0] = ix.k;
                d.t[1] = cp.latent;
                d.t[2] = w.get(l, "attn.indexer.wk.weight");
                d.i[0] = n_pool;
                d.i[1] = di;
                d.i[2] = c.head_dim;
            })
        };
        let c_kn = b.emit(DevOp::RmsNorm, all.clone(), &[c_k], |d| {
            d.t[0] = ix.kn;
            d.t[1] = ix.k;
            d.t[2] = w.get(l, "attn.indexer.k_norm.weight");
            d.i[0] = n_pool;
            d.i[1] = di;
            d.f[0] = c.eps;
        });
        // `fp4_act_quant(k, fp4_block_size, True)` -- blocks of 32 with an E8M0 scale, which is
        // the OTHER of op 199's two settings. A key stands for its group's first token, so it
        // ropes at `j * ratio` exactly as the cache row does.
        key_dep.push(b.emit(DevOp::CompressRopeQuant, all.clone(), &[c_kn], |d| {
            d.t[0] = ix.keys;
            d.t[1] = ix.kn;
            d.t[2] = cos;
            d.t[3] = sin;
            d.t[4] = TENSOR_NONE;
            d.i[0] = n_pool;
            d.i[1] = di;
            d.i[2] = rd;
            d.i[3] = DSV41_IDX_QBLK;
            d.i[4] = ratio;
            d.i[5] = 0;
            d.i[6] = 1; // PLOW_CMP_Q_FP4_POW2 -- E8M0, the fp4_act_quant default
            d.i[7] = 1;
        }));
    }

    // ---- the queries, and the per-head weights ---------------------------------------------
    let c_q = emit_pf_gemm_fp8_mx(
        b,
        cus,
        ix.q,
        qr,
        w.get(l, "attn.indexer.wq_b.weight"),
        w.get(l, "attn.indexer.wq_b.scale"),
        t,
        hi * di,
        c.q_lora,
        deps,
    );
    // One angle per TOKEN across all 32 heads, then fp4 at 32 over each head's whole 128 --
    // op 199's third call site, `i7 = n_head`.
    let c_qr = b.emit(DevOp::CompressRopeQuant, all.clone(), &[c_q], |d| {
        d.t[0] = ix.qr;
        d.t[1] = ix.q;
        d.t[2] = cos;
        d.t[3] = sin;
        d.t[4] = TENSOR_NONE;
        d.i[0] = t;
        d.i[1] = di;
        d.i[2] = rd;
        d.i[3] = DSV41_IDX_QBLK;
        d.i[4] = 1; // a query stands for its OWN token
        d.i[5] = 0;
        d.i[6] = 1;
        d.i[7] = hi;
    });
    let c_w = bf16_gemm(
        b,
        ix.w,
        xn,
        w.get(l, "attn.indexer.weights_proj.weight"),
        hi,
        c.hidden,
        deps,
    );

    // ---- score, then select ------------------------------------------------------------------
    // `scale` is `softmax_scale * n_heads**-0.5` (model.py:555), folded into op 117's epilogue
    // instead of into `weights`: `part * scale` and `(w * scale) * relu` are the same expression
    // by distributivity, and the epilogue is where the op already has a multiply.
    let scale = (di as f32).powf(-0.5) * (hi as f32).powf(-0.5);
    let kv_stride = ctx / ratio;
    let mut score_deps = vec![c_qr, c_w];
    score_deps.extend_from_slice(&key_dep);
    let c_sc = b.emit(DevOp::IndexScorePf, all.clone(), &score_deps, |d| {
        d.t[0] = ix.score;
        d.t[1] = ix.qr;
        d.t[2] = ix.keys;
        d.t[3] = ix.w;
        d.t[4] = kvlen;
        d.i[0] = t;
        d.i[1] = hi;
        d.i[2] = kv_stride;
        d.i[3] = di;
        d.i[4] = ratio; // POOLS, not tokens
        d.f[0] = scale;
    });
    let c_sel = b.emit(DevOp::IndexSelectPf, all.clone(), &[c_sc], |d| {
        d.t[0] = ix.idx;
        d.t[1] = ix.score;
        d.t[2] = kvlen;
        d.i[0] = t;
        d.i[1] = topk;
        d.i[2] = kv_stride;
        d.i[3] = ratio; // MUST equal op 117's i4
    });
    // THE PER-PACK UNION. Built here, beside the selection it summarises and once per PUBLICATION
    // rather than once per reader: the 38 reader layers share `ix.idx`, so they share this too.
    //
    // It is what the gathered flash needs to take the V2 arm. The scalar arm walks one query's
    // top-k at a time and re-reads the latent for each; the V2 arm stages the pack's union ONCE
    // and masks per row. Measured on this model's own selections at 8k, a pack of 8 unions to
    // 1582 rows against the 8 x 480 = 3840 the scalar arm reads: 2.43x fewer row reads.
    b.emit(DevOp::IndexUnionPf, all.clone(), &[c_sel], |d| {
        d.t[0] = ix.uni;
        d.t[1] = ix.umask;
        d.t[2] = ix.idx;
        d.t[3] = kvlen;
        d.i[0] = t;
        d.i[1] = topk;
        // THE SCAN BOUND IS IN TOKENS, THE SELECTIONS ARE IN POOLS, and `i2` has to cover the
        // larger of the two. `d_index_union_pf` walks `[0, tile_end)` with
        // `tile_end = kv_len[0] - n_tok + q_hi + 1` -- a TOKEN position -- while indexing
        // `umask[slice * kv_stride + s]`. GLM reaches here after `DsaPoolExpand`, so its two
        // spaces agree; V4.1 selects COMPRESSED entries directly and `ctx / ratio` would be half
        // the row the scan writes, running off the end of the scratch (a hardware exception, not
        // a wrong answer). `ctx` is the bound that covers it; positions above `ctx / ratio` are
        // simply never set, and `cap` is unchanged because 8 * top_k is the binding term.
        d.i[2] = ctx;
        d.i[3] = dsv41_union_cap(topk, ctx);
        d.i[4] = DSV41_UNION_PACK;
    })
}

/// How many compressed positions one query selects: `min(index_topk, end_pos // ratio)`
/// (`model.py:578`).
///
/// ONE derivation, because op 118 WRITES the table at this width and op 55 READS it at this
/// width, from two different emit functions. A chunk shorter than `index_topk * ratio` selects
/// every pool it has, and a reader that assumed the full 512 would walk past the row.
pub(crate) fn dsv41_index_topk(c: &Dsv41Cfg, t: u32, ratio: u32) -> u32 {
    c.index_topk.min(t / ratio)
}

/// `fp4_act_quant(k, fp4_block_size, True)` -- the indexer's quant block, model.py:546/552.
pub(crate) const DSV41_IDX_QBLK: u32 = 32;

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
    /// The CROSS-SUBLAYER `pre` gates, `[2][t][hc_mult]` f32 -- both halves in one tensor because
    /// `HyperConnPre` has 8 tensor slots and already used 7. Sublayer `k` reads half `k % 2` and
    /// publishes into the other, so consecutive sublayers alternate. See [`super::MhcPre`].
    pub(crate) pre_pair: u32,
}

/// Declare the mHC stream for `t` rows. `mix` is `(2 + hc_mult) * hc_mult`, the same derivation
/// `dsv41_layer_tensors` sizes `hc_*_fn` with -- one formula, so the weight and the scratch cannot
/// disagree about how many mix rows there are.
pub(crate) fn declare_dsv41_mhc(b: &mut Builder, c: &Dsv41Cfg, t: u32, tp: u32) -> Dsv41Mhc {
    let (r, n, w) = (t as u64, c.hc_mult as u64, c.hidden as u64);
    let mix = (2 + n) * n;
    // Under sequence parallelism the mixes and the cross-sublayer gates are PURE SCRATCH between
    // one `pre` and its `post`: no collective reads them and nothing outside the band does. So
    // they shrink to the band and are indexed densely, rows `[0, t/tp)`. The residual cannot --
    // it is the block's entry AND its exit, so it stays full size and is addressed through
    // `dsv41_band` views, which is also why `pre_pair` could not simply be viewed: it is
    // `[2][t][hc_mult]`, half-major, and one `@band` offset cannot slice both halves.
    let sr = if dsv41_sp(t, tp) { (t / tp) as u64 } else { r };
    Dsv41Mhc {
        residual: [
            b.tensor("act.hc_residual_a", r * n * w * 2),
            b.tensor("act.hc_residual_b", r * n * w * 2),
        ],
        layer_input: b.tensor("act.hc_layer_input", r * w * 2),
        mixes: b.tensor("act.hc_mixes", sr * mix * 4),
        post_mix: b.tensor("act.hc_post_mix", sr * n * 4),
        comb_mix: b.tensor("act.hc_comb_mix", sr * n * n * 4),
        pre_pair: b.tensor("act.hc_pre_pair", 2 * sr * n * 4),
    }
}

/// The peer slot UNIT for this packet: the widest per-token collective message it carries.
///
/// # Why this is not just `c.hidden`
///
/// `DevBlob::parse` recovers two numbers from the collectives and nothing else: `hidden` as
/// `max(i[0] / program.t)` and `slot_bytes` as `max(i[2])`, the byte offset of partial slot B.
/// `AmdTpGroup::load` then REFUSES a packet where `slot_bytes % (hidden * 2) != 0`, because it
/// divides one by the other to recover `max_tokens`.
///
/// Every collective in this model is `hidden` = 5120 wide except ONE: Engram's all-reduce of the
/// row-split gather, which is `n_cols * head_dim` = 6144. That single op raises the recovered
/// `hidden` to 6144 while the routed combine's slot offset stays at `t * 5120 * 2`, and
/// 83 886 080 is not a multiple of 12 288 -- so a layer-1 packet built with `c.hidden` here
/// LOADS NOWHERE. It is caught at load rather than silently, which is the good case, but it is
/// caught after an object build and a queue wait.
///
/// So the unit is the max, and it is conditional on the chain actually containing an engram
/// layer: widening it unconditionally would break every OTHER packet the same way, since their
/// recovered `hidden` stays 5120 and 6144-based offsets are not multiples of 10 240 either.
pub(crate) fn dsv41_peer_width(c: &Dsv41Cfg, layers: &[u32]) -> u32 {
    if layers.iter().any(|l| c.engram_layers.contains(l)) {
        c.hidden
            .max(engram_cols(c) * c.raw.engram_head_dim as u32)
    } else {
        c.hidden
    }
}

/// Engram's scratch, declared only for a chain that contains an engram layer.
///
/// `ids` is an INPUT, not an activation: the n-gram hash is integer work over token ids alone
/// (`crates/plowrt/src/text/engram.rs`), so it arrives host-built like `in.pos`.
pub(crate) struct Dsv41Engram {
    /// This rank's contribution to the gather, `[T][n_cols * head_dim]` bf16, in PEER SLOT 0 --
    /// the table is row-split so a rank writes zeros for every id outside its shard, and the
    /// all-reduce below is what makes the row whole. Reusing attention's slot is safe for the same
    /// reason the shared expert's reuse is: Engram runs at the TOP of the layer and its reduce has
    /// completed before attention's begins. It is WIDER than attention's use of the slot (6144 vs
    /// hidden=5120), which costs nothing because `Builder::tensor` takes the max on re-declaration.
    pub(crate) part: u32,
    /// The summed gather, which is what `wkv` consumes.
    pub(crate) emb: u32,
    /// `wkv`'s output: `hc_mult` keys then ONE shared value, `[T][(hc_mult + 1) * hidden]` bf16.
    pub(crate) kv: u32,
    /// `[T][n_cols]` i32, the n-gram row ids. GLOBAL ids, not shard-local: op 197 does the
    /// signed subtract itself so one uploaded tensor serves every rank.
    pub(crate) ids: u32,
}

/// Declare Engram's scratch for `t` rows. `None` when no layer in the chain has an Engram.
pub(crate) fn declare_dsv41_engram(
    b: &mut Builder,
    c: &Dsv41Cfg,
    layers: &[u32],
    t: u32,
) -> Option<Dsv41Engram> {
    if !layers.iter().any(|l| c.engram_layers.contains(l)) {
        return None;
    }
    let (r, cols, hd) = (t as u64, engram_cols(c) as u64, c.raw.engram_head_dim as u64);
    let n = c.hc_mult as u64;
    Some(Dsv41Engram {
        part: b.tensor(PEER_SLOT_O, r * cols * hd * 2),
        emb: b.tensor("act.engram_emb", r * cols * hd * 2),
        kv: b.tensor("act.engram_kv", r * (n + 1) * c.hidden as u64 * 2),
        ids: b.tensor("in.engram_ids", r * cols * 4),
    })
}

/// `n_hash_cols` -- `(max_ngram_size - 1) * n_heads`, 24 for this checkpoint.
///
/// ONE derivation, shared by the tensor table, the scratch and the emit, because three copies of
/// `(n - 1) * heads` is three chances to disagree about how wide a row of `wkv`'s input is.
pub(crate) fn engram_cols(c: &Dsv41Cfg) -> u32 {
    (c.raw.engram_max_ngram_size - 1) * c.raw.engram_n_heads
}

/// Engram: the n-gram lookup written into the residual stream, BEFORE the block.
///
/// # This runs ahead of `mhc_pre`, not inside the FFN
///
/// `Block.__init__` constructs `self.engram` but `Block.forward` never calls it.
/// `Transformer.forward` does, in the layer loop and ahead of the block (`model.py:1262-1267`),
/// so op 196 is in place on the hc-EXPANDED stream `[T][hc_mult][hidden]` before this layer's mHC
/// collapses it. See `dsv41_layer_parts` for what that cost when the table said otherwise.
///
/// # Four ops, and the all-reduce is the reference's own
///
///   1. **op 197**, the gather. The table is 384 006 168 rows of 256 fp8 -- 98.31 GB, so a
///      replicated copy does not fit on a 192 GB MI300X at all -- and is row-split. A rank writes
///      ZEROS for any id outside its shard.
///   2. **`XReduce`**, which sums those shards. This is `ParallelEngramEmbedding.forward`'s own
///      `dist.all_reduce(values)` (`model.py:323-324`), not an artifact of this emit, and it is
///      why the gather lands in a peer slot rather than in ordinary VRAM.
///   3. **op 198**, `wkv`. Block-fp8 at a `[32, 32]` ue8m0 grid, `[T][6144] -> [T][25600]`. The
///      reduce comes BEFORE it, as the reference has it. `wkv` is linear and bias-free so the two
///      orders agree in exact arithmetic, but reducing after would quantize partial sums, and it
///      would also reduce a tensor four times wider.
///   4. **op 196**, the gate and mix, in place on the residual.
///
/// `token_mask` is `TENSOR_NONE`: it selects image spans, which take part in no n-gram, and a
/// text-only prefill has none. A VL path must pass it -- a masked token has to pass through
/// UNTOUCHED, which is not the same as adding a zero value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dsv41_engram(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    cus: &[u32],
    e: &Dsv41Engram,
    l: u32,
    tp: u32,
    t: u32,
    residual: u32,
    xgate: &mut u32,
    deps: &[u32],
) -> u32 {
    let cols = engram_cols(c);
    let hd = c.raw.engram_head_dim as u32;
    let rows = c
        .raw
        .engram_rows(l)
        .expect("emit_dsv41_engram called for a layer with no engram table");
    // `part_num_embeddings = ceil(num_embeddings / world_size)`, model.py:303. The interpreter
    // derives THIS rank's base as `rank * part_rows` -- see `DevOp::EngramEmbed`'s i4 sentinel.
    let part_rows = (rows as u64).div_ceil(tp as u64);
    assert!(
        part_rows <= u32::MAX as u64,
        "engram shard of {part_rows} rows does not fit an i32 row index"
    );

    let c_gather = b.emit(DevOp::EngramEmbed, cus.to_vec(), deps, |d| {
        d.t[0] = e.part;
        d.t[1] = w.get(l, "engram.embed.weight");
        d.t[2] = w.get(l, "engram.embed.scale");
        d.t[3] = e.ids;
        d.i[0] = t;
        d.i[1] = cols;
        d.i[2] = hd;
        // `weight_block_size` is [32, 32] for V4.1, NOT the [128, 128] V4 used. A wrong value
        // rescales every lookup and faults nowhere -- so it comes off the config, not a literal.
        d.i[3] = c.raw.quantization_config.weight_block_size[1] as u32;
        d.i[4] = TENSOR_NONE_I; // derive the shard base from the rank
        d.i[5] = part_rows as u32;
    });

    let xr = super::xr_cus_capped(b.n_cu(), cus);
    let c_xr = crate::emit_xreduce(
        b,
        xgate,
        false,
        &xr,
        c_gather,
        e.emb,
        t * cols * hd,
        tp,
        early_reduce_slot(t, cols * hd),
    );

    let kv_out = c.hidden * (c.hc_mult + 1);
    let c_kv = super::emit_pf_gemm_fp8_mx(
        b,
        cus,
        e.kv,
        e.emb,
        w.get(l, "engram.wkv.weight"),
        w.get(l, "engram.wkv.scale"),
        t,
        kv_out,
        cols * hd,
        &[c_xr],
    );

    // ONE WORKGROUP PER TOKEN, as `d_engram_gate`'s header requires: its three reductions per hc
    // copy are over `hidden` and want the whole workgroup. Same grid rule as `HyperConnPre`.
    b.emit(
        DevOp::EngramGate,
        (0..t.min(b.n_cu())).collect(),
        &[c_kv],
        |d| {
            d.t[0] = residual;
            d.t[1] = e.kv;
            d.t[2] = w.get(l, "engram.q_weight");
            d.t[3] = w.get(l, "engram.k_weight");
            d.t[4] = TENSOR_NONE; // no token_mask: text-only prefill has no image spans
            d.i[0] = t;
            d.i[1] = c.hc_mult;
            d.i[2] = c.hidden;
            d.f[0] = c.eps;
        },
    )
}

/// The mHC PRE half for one sublayer: collapse the residual copies into `m.layer_input`.
///
/// `ffn` picks which of the layer's TWO mHC weight sets to use. Every layer carries both
/// (`hc_attn_*` and `hc_ffn_*`), which is why there is no mHC-free block to extract from this
/// model and why the rung has to emit it before anything downstream is meaningful.
///
/// # `pi` -- V4.1's mHC is CROSS-SUBLAYER, GLM-5.3's is not
///
/// `pi` is the sublayer's index in the emitted chain, counting BOTH halves of every layer:
/// `2 * (layer index within the chain) + (ffn as usize)`. It picks which half of `m.pre_pair`
/// this sublayer reads, because V4.1 gates `hc_pre` with the coefficients the PREVIOUS sublayer
/// produced -- attention with the previous block's `ffn_pre`, the FFN with this block's
/// `attn_pre` (`model.py:965-996`, and the `Block` docstring says it outright). Binding V4.1
/// onto GLM's same-sublayer ordering was worth 51.9% relative error on layer 0 ALONE, measured
/// by `scripts/dsv41_mhc_oracle.py`; `post` and `comb` are same-sublayer in both and unchanged.
///
/// `pi == 0` uses [`super::MhcPre::Seed`], the one-hot on copy 0 that `make_identity_pre_mix`
/// supplies before the model's first block (`model.py:1159-1163`). For a chain that starts at
/// layer 0 that IS the model; for a rung starting elsewhere it is the rung's synthetic entry, of
/// a piece with the synthetic residual stream the harness uploads into `act.hc_residual_a`.
#[allow(clippy::too_many_arguments)]
/// Rank-relative band view `<base>@band<t>`: rows `[rank*t/tp, (rank+1)*t/tp)` of a `[t][row]`
/// tensor. The host binds every `<base>@band...` at `base + rank * bytes` (`exec/amd.rs`'s
/// `is_band_view`), so the view is pure addressing -- storage belongs to the base.
pub(crate) fn dsv41_band(b: &mut Builder, base: u32, t: u32, tp: u32, row_bytes: u64) -> u32 {
    let name = format!("{}@band{t}", b.tensor_name(base));
    match (0..b.n_tensors() as u32).find(|&h| b.tensor_name(h) == name) {
        Some(h) => h,
        None => b.tensor(&name, (t / tp) as u64 * row_bytes),
    }
}

/// `PLOW_DSV41_SEQ_PAR`: the layer's per-token work runs on this rank's `t/tp` band.
///
/// The mHC stream and the norms depend only on their own token, so eight ranks running all `t`
/// rows of them is eight times the work for one answer (12.81 prices it at 2.55 ms/layer). Under
/// this the two TP seams become reduce-scatter instead of all-reduce, the mHC runs on the owned
/// band, and the GEMM blocks -- which DO need every token, being head- and expert-parallel -- take
/// an all-gather of `layer_input` in front of them. A two-shot all-reduce is already a
/// reduce-scatter plus an all-gather, so the fabric moves the same bytes either way.
///
/// The floor is the runtime's: `check_seq_par_seams` refuses a decode rung or a `t` that is not a
/// multiple of `tp`, and grows the peer region to `SEQ_PAR_SLOTS` when it sees the collectives.
/// Find-or-create a peer RESULT slot by name. Both the block emit and the MoE emit need the same
/// handle and `Builder::tensor` does not deduplicate, so a second plain declaration would publish
/// two tensors with one name and the runtime would bind whichever it found first.
pub(crate) fn dsv41_sp_slot(b: &mut Builder, name: &str, bytes: u64) -> u32 {
    match (0..b.n_tensors() as u32).find(|&h| b.tensor_name(h) == name) {
        Some(h) => h,
        None => b.tensor(name, bytes),
    }
}

pub(crate) fn dsv41_sp(t: u32, tp: u32) -> bool {
    crate::emit_config::active().dsv41_seq_par && tp > 1 && t > 1 && t % tp == 0
}

/// CEILING INSTRUMENT ONLY (`PLOW_DSV41_SP_ABL=1`): run the per-token mHC and norm packets over
/// `t/tp` rows instead of `t`.
///
/// The ANSWER IS WRONG -- 7/8 of the rows are never written, so the residual stream is garbage
/// from the first layer on. What it measures is right: these packets are replicated across the
/// ranks today (every rank computes all `t` rows of work that depends only on its own token), and
/// sequence parallelism would leave each rank exactly this band. So the run time of this emit is
/// the run time SP would reach, without building SP's reduce-scatter/all-gather plumbing or the
/// third peer-slot region the dsv41 layout would need for it.
///
/// Read it against the unablated control and the difference is the SP budget, nothing more.
fn sp_abl_rows(t: u32, tp: u32) -> u32 {
    // The real thing and the instrument that prices it land on the same row count; only the real
    // one also moves the addresses, which is the whole difference between a band and a wrong answer.
    if dsv41_sp(t, tp) || (crate::emit_config::active().dsv41_sp_abl && tp > 1) {
        t / tp
    } else {
        t
    }
}

pub(crate) fn emit_dsv41_mhc_pre(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    m: &Dsv41Mhc,
    l: u32,
    ffn: bool,
    ri: usize,
    pi: usize,
    t: u32,
    tp: u32,
    deps: &[u32],
) -> u32 {
    let side = if ffn { "ffn" } else { "attn" };
    let sp = dsv41_sp(t, tp);
    // Under SP the residual is addressed through its band view and `layer_input` is PUBLISHED into
    // peer result slot 3 rather than written whole: the GEMM block that consumes it is head- and
    // expert-parallel, so it needs every token, and the all-gather in front of it is what turns
    // these bands back into `m.layer_input`.
    let (res, li) = if sp {
        let hb = (t as u64) * (c.hidden as u64) * 2;
        let h2 = dsv41_sp_slot(b, "act.h2_tp", hb);
        (
            dsv41_band(b, m.residual[ri], t, tp, c.hc_mult as u64 * c.hidden as u64 * 2),
            dsv41_band(b, h2, t, tp, c.hidden as u64 * 2),
        )
    } else {
        (m.residual[ri], m.layer_input)
    };
    let h = super::MhcHandles {
        fn_w: w.get(l, &format!("hc_{side}_fn")),
        base: w.get(l, &format!("hc_{side}_base")),
        scale: w.get(l, &format!("hc_{side}_scale")),
        residual: res,
        mixes: m.mixes,
        post_mix: m.post_mix,
        comb_mix: m.comb_mix,
        layer_input: li,
        pre: {
            let (pair, in_half) = (m.pre_pair, (pi % 2) as u32);
            if pi == 0 {
                super::MhcPre::Seed { pair, in_half }
            } else {
                super::MhcPre::Deferred { pair, in_half }
            }
        },
    };
    super::emit_mhc_pre(
        b,
        &h,
        c.hidden,
        c.hc_mult,
        c.hc_sinkhorn_iters,
        c.eps,
        c.raw.hc_eps,
        sp_abl_rows(t, tp),
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
    tp: u32,
    deps: &[u32],
) -> u32 {
    let (out, rin, rw) = if dsv41_sp(t, tp) {
        let rb = c.hc_mult as u64 * c.hidden as u64 * 2;
        (
            dsv41_band(b, m.residual[ri ^ 1], t, tp, rb),
            dsv41_band(b, m.residual[ri], t, tp, rb),
            t / tp,
        )
    } else {
        (m.residual[ri ^ 1], m.residual[ri], sp_abl_rows(t, tp))
    };
    super::emit_mhc_post(
        b,
        out,
        raw,
        rin,
        m.post_mix,
        m.comb_mix,
        c.hidden,
        c.hc_mult,
        rw,
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
    // while these experts are MXFP4, so the caller emits it on op 198 and this only combines it.
    shared: (u32, u32),
    xgate: &mut u32,
    cus: &[u32],
    peer_w: u32,
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
        // The GLM MoE descriptor's own EP flag stays FALSE even under `PLOW_MOE_PREFILL_EP`: the
        // placement is applied by the whole-graph rewrite in `Builder::finish`, not by emitting
        // pre-widened ops here. See the note at `set_moe_prefill_ep_degree`.
        false,
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
        (t as u64) * (c.n_exp as u64) * 4,
    );
    n.tab = b.tensor(&format!("act.l{l}.tab"), (t as u64) * (c.top_k as u64) * 8);
    if tp > 1 {
        // The combine's partial. `GlmTn::none()` leaves this TENSOR_NONE, and the body writes it
        // unconditionally at tp>1 (`emit_glm_moe_ffn_prefill`: `d.t[0] = n.dg_tp`), so leaving it
        // unset would aim every rank's routed output at the null handle.
        n.dg_tp = b.tensor(PEER_SLOT_MOE, (t as u64) * (c.hidden as u64) * 2);
    }
    if dsv41_sp(t, tp) {
        // The three peer RESULT slots the sequence-parallel seams publish bands into. The runtime
        // binds them by NAME (`exec/amd.rs`'s `is_peer_slot`: h2 = slot 3, xe = 4, rt = 5) and
        // `check_seq_par_seams` refuses a blob that carries XReduceScatter/XAllGather without all
        // three, so they are declared together whether or not each seam uses one.
        let hb = (t as u64) * (c.hidden as u64) * 2;
        n.h2_tp = dsv41_sp_slot(b, "act.h2_tp", hb);
        n.xe_tp = dsv41_sp_slot(b, "act.xe_tp", hb);
        n.rt_tp = dsv41_sp_slot(b, "act.rt_tp", hb);
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
    // The UNIT, not this op's width: see `dsv41_peer_width`. The combine writes `t * hidden * 2`
    // at this offset either way; what changes is where slot B begins.
    n.slot_b = (t as u64 * peer_w as u64 * 2) as u32;
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
            | super::router_flag::F32_LOGIT
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
    // The shared compressed cache, the shared selection and `index_topk`, when this layer reads
    // them. `None` is a layer whose `compress_ratio` is 0 -- 0 and 1 only.
    compressed: Option<(u32, u32, u32)>,
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
    // TWO PARTIALS on a layer that also reads the compressed cache: the window pass and the
    // gathered pass. `Attention.forward` runs ONE `sparse_attn` over `cat([window_kv,
    // compress_kv])` with the sink added once at the end (model.py:775-779); a two-partial merge
    // is that same expression, and it avoids copying the SHARED cache into a per-layer buffer
    // and re-basing the index table by the window length.
    // The gathered pass then splits ITS partial further: 1024 query packs over 304 workgroups is
    // 3.37 rounds, so the packet pays 4, and the last round is the expensive one because a pack's
    // union grows with its position until top-k caps it. `gsplit` items per pack make the
    // quantization finer -- the same lever the dense arm's causal KV-split already pulls.
    let gsplit = crate::emit_config::active().mla_gather_split.max(1).min(8);
    let nsplit = if compressed.is_some() { 1 + gsplit as u64 } else { 1u64 };
    let act = Dsv41CoreAct {
        qr: b.tensor(
            &format!("act.l{l}.qr"),
            (t as u64) * (nh_l as u64) * (hd as u64) * 2,
        ),
        kvr: b.tensor(&format!("act.l{l}.kvr"), (t as u64) * (hd as u64) * 2),
        opart: b.tensor(
            &format!("act.l{l}.opart"),
            (t as u64) * (nh_l as u64) * (hd as u64) * 4 * nsplit,
        ),
        mlpart: b.tensor(
            &format!("act.l{l}.mlpart"),
            (t as u64) * (nh_l as u64) * 2 * 4 * nsplit,
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
            // BIT 31 = INTERLEAVED. `apply_rotary_emb` takes "adjacent element pairs as complex
            // numbers" (`model.py:392`); op 142's default arm pairs `(i, i + rotary/2)`. The two
            // are different operators and the mistake does NOT cancel when q and the latent take
            // it together -- `scripts/dsv41_rope_oracle.py` prices the swap at 41.2% of the
            // maximum attention score, on a tensor of exactly the right shape.
            d.i[2] = rope | (1u32 << 31);
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
        // Write-only output split: this is partial 0 of `nsplit`. Zero when there is one partial,
        // which is the layout every other emitter in the tree produces.
        d.i[7] = if nsplit > 1 { ((nsplit as u32) << 8) | 0 } else { 0 };
        d.f[0] = scale;
    });
    // THE COMPRESSED READ. `compress_ratios[l] != 0` on 38 of 40 layers and every one of them
    // attends over the shared CSA2 cache ON TOP of its window -- the parts table called this
    // "38 of 40 layers silently attending over a fraction of what they should".
    //
    // No causal mask: the selector produced the set, and op 118 scanned only
    // `(t + 1) / ratio` pools, so a query cannot name a compressed entry that is not complete
    // yet (`scripts/dsv41_indexer_oracle.py` check [2]).
    let c_fl = match compressed {
        None => c_fl,
        // OP 51, NOT OP 55, and `t7` is the per-pack UNION rather than the per-query top-k.
        //
        // Op 55 is `d_flash_gather_prefill`, the scalar body: one query at a time, its own top-k,
        // the latent re-read for each. The same packet as op 51 with the NoPE bit and a union in
        // t7 selects `d_flash_mla_prefill_v2<512, 0, GATHER=true>` instead, which stages the
        // pack's union once for 8 queries with all 8 per-rank heads on the MFMA M dimension.
        // That arm requires `n_head == 8`, which is what TP8 gives this model.
        //
        // i6 is the union's `cap` here, disambiguated by t7 exactly as the DR=64 chain does it.
        Some((cache, uni, topk)) => b.emit(DevOp::FlashMlaPrefill, all.clone(), &[c_q, c_kv, c_fl], |d| {
            d.t[0] = act.opart;
            d.t[1] = act.mlpart;
            d.t[2] = act.qr;
            d.t[3] = act.qr;
            d.t[4] = cache;
            d.t[5] = cache;
            d.t[6] = kvlen;
            d.t[7] = uni;
            d.i[0] = 1;
            d.i[1] = nh_l;
            d.i[2] = ctx; // cache rows; the selection never names one past `ctx / ratio`
            d.i[3] = 1u32 << 31; // NoPE, and NO window: the gather arm takes the whole set
            d.i[4] = t;
            d.i[5] = KV_MASK_NONE;
            d.i[6] = dsv41_union_cap(topk, ctx);
            // Partials 1..gsplit of nsplit; the body adds its split index to `out_sp0`.
            d.i[7] = (gsplit << 16) | ((nsplit as u32) << 8) | 1;
            d.f[0] = scale;
        }),
    };
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
        d.i[2] = nsplit as u32;
        d.i[3] = hd;
    });
    // THE INVERSE ROPE, which this emit used to leave out entirely.
    //
    // `Attention.forward` ends `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` (model.py:781) on
    // EVERY layer, window-only ones included -- op 195's own header calls it "STRUCTURAL, not
    // cosmetic", because `o` mixes cached rows each rotated by its OWN position and de-rotating by
    // the QUERY's position is what leaves a position-independent latent for the fixed `wo_a`. The
    // op has existed since the CSA2 wiring and had no emit site; a layer without it runs, stays
    // finite, and is wrong. `scripts/dsv41_rope_oracle.py` check [5].
    //
    // `i4 = 0`: a rung is one chunk starting at position 0, and `t3 = TENSOR_NONE` because the
    // tensor form carries ONE step and exists for decode.
    let c_ir = b.emit(DevOp::RopeInverseO, all.clone(), &[c_mg], |d| {
        d.t[0] = act.o;
        d.t[1] = cos;
        d.t[2] = sin;
        d.t[3] = TENSOR_NONE;
        d.i[0] = t;
        d.i[1] = nh_l;
        d.i[2] = hd;
        d.i[3] = rope;
        d.i[4] = 0;
    });
    (act, c_ir)
}

/// The o_proj TP partial's tensor name, fixed by `plowrt`'s peer-slot table (slot 0).
///
/// `crates/plowrt/src/exec/amd.rs` matches this string literally to bind the tensor into the peer
/// region instead of local VRAM. Renaming it here silently un-peers the buffer.
pub(crate) const PEER_SLOT_O: &str = "act.og_tp";
/// Where the shared expert's down projection writes its partial: SLOT 0, the same one attention
/// used earlier in the layer. See `emit_dsv41_ffn_shared` for why reusing it is safe here and why
/// a third slot is not available.
pub(crate) const PEER_SLOT_SHARED: &str = PEER_SLOT_O;
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
) -> (Dsv41OutAct, Vec<u32>) {
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
    let xr = super::xr_cus_capped(b.n_cu(), cus);
    let kb = dsv41_xr_band(t);
    let c_xr = if dsv41_sp(t, tp) {
        // Sequence-parallel attention seam: the o_proj partial is reduce-SCATTERED, so this rank
        // keeps its own band of the attention output in `act.og_tp` and the mHC post reads it
        // there. No all-gather here -- the next one that matters is in front of the FFN's GEMM.
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
        vec![crate::emit_xreduce_scatter(
            b,
            xgate,
            &xr,
            &[c_ob],
            act.o_part,
            t * c.hidden,
            tp,
            early_reduce_slot(t, c.hidden),
            None,
            None,
        )]
    } else if tp > 1 && kb > 1 {
        // BANDED TP SEAM: K row-band wo_b GEMMs, each feeding its own two-shot, so band 0's
        // fabric transfer overlaps bands 1..K-1's compute. The bands are disjoint rows of the
        // same tiles, so the sums are bit-identical to the unbanded emit.
        let rows = t / kb;
        let bcus = dsv41_xr_band_cus(&xr);
        let ends: Vec<u32> = (0..kb)
            .map(|i| {
                let c_p = super::emit_pf_gemm_fp8_mx_band(
                    b,
                    cus,
                    act.o_part,
                    act.o_a,
                    w.get(l, "attn.wo_b.weight"),
                    w.get(l, "attn.wo_b.scale"),
                    rows,
                    c.hidden,
                    orow,
                    i * rows,
                    &[c_oa],
                );
                crate::emit_xreduce_twoshot_band(
                    b,
                    xgate,
                    &bcus,
                    &[c_p],
                    act.o,
                    rows * c.hidden,
                    tp,
                    0,
                    i * rows * c.hidden,
                    None,
                )
            })
            .collect();
        ends
    } else {
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
        vec![crate::emit_xreduce(
            b,
            xgate,
            false,
            &xr,
            c_ob,
            act.o,
            t * c.hidden,
            tp,
            early_reduce_slot(t, c.hidden),
        )]
    };
    (act, c_xr)
}

/// Band count for the V4.1 attention TP seam (`PLOW_DSV41_XR_BAND=K`, 2..=8; 1 = the unbanded
/// emit, byte-identical). K row-bands of the `wo_b` GEMM each feed their own two-shot, so band
/// 0's fabric transfer runs while bands 1..K-1 are still in the GEMM. The `t/K >= 512` floor keeps
/// a band's GEMM from under-filling 304 CUs.
pub(crate) fn dsv41_xr_band(t: u32) -> u32 {
    let k = crate::emit_config::active().dsv41_xr_band;
    if (2..=8).contains(&k) && t % k == 0 && t / k >= 512 {
        k
    } else {
        1
    }
}

/// The band collectives take a PREFIX of the seam's CUs so the workgroups outside it walk past
/// them on the global queue and claim the next band's GEMM -- without that there is nothing for
/// the transfer to overlap WITH.
pub(crate) fn dsv41_xr_band_cus(xr: &[u32]) -> Vec<u32> {
    match crate::emit_config::active().dsv41_xr_band_cus {
        Some(c) if c > 0 && (c as usize) < xr.len() => xr[..c as usize].to_vec(),
        _ => xr.to_vec(),
    }
}

/// Suffix for the packed-expert table names: EMPTY in a real packet.
///
/// Under a `PLOW_DSV41_OPS` cut that lands before the grouped GLU, the tables are still DECLARED
/// -- truncation drops instructions, not tensors -- and `bind_packed_experts` then refuses the
/// blob: "expert_weight_table is declared but no decode instruction streams experts through it".
/// The refusal is right; the profiling blob simply has no MoE left. Renaming the tables takes
/// them out of that scan, since it keys on the exact `expert_weight_table` suffix.
///
/// `_moe2` specifically, and the choice is forced from both sides: `is_host_filled_table` must
/// still MATCH the name (or the loader hunts the checkpoint for a weight by that name and fails
/// with "MISSING WEIGHT"), while `bind_packed_experts`' layer scan must NOT -- it strips the exact
/// `expert_weight_table` suffix, which `..._moe2` does not end with. The stage-2 companion lookup
/// that does use `_moe2` only runs for a prefix the layer scan already found, and it finds none.
///
/// Profiling only, and only below the GLU cut: a real packet must keep the real names or its
/// experts never get packed.
fn prof_table_suffix() -> &'static str {
    match crate::emit_config::active().dsv41_ops {
        Some(n) if n < 30 => "_moe2",
        _ => "",
    }
}

/// Slot for the two EARLY reduces (attention, shared expert): 0 in a real packet.
///
/// Under `PLOW_DSV41_OPS` it becomes `slot_b` instead, and ONLY so that a truncated packet still
/// loads. `DevBlob::parse` recovers `slot_bytes` as `max(i[2])` over the collectives, and in the
/// shipped two-slot layout the ROUTED COMBINE's reduce is the only one carrying a non-zero
/// `i[2]` -- so a prefix cut anywhere before it recovers `slot_bytes = 0` and the TP loader
/// refuses the blob ("partial slot is 0 B, not a multiple of hidden*2"). Putting the early
/// reduces at `slot_b` keeps the max non-zero for every prefix that contains one.
///
/// It is a PROFILING CUT, not an alternative design: with this on, two reduces share `slot_b`
/// and the layer's numerics are not the model's. `emit_dsv41_block` prints the warning.
fn early_reduce_slot(t: u32, hidden: u32) -> u32 {
    if crate::emit_config::active().dsv41_ops.is_some() {
        t * hidden * 2
    } else {
        0
    }
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
    /// The down projection's PARTIAL, in peer slot 2, `[T][hidden]` bf16 -- and the partial is
    /// what the routed combine takes, at every `tp`. Nothing reduces it here; the band reduce
    /// after `d_moe_combine_pf` is the one cross-rank sum the shared expert gets.
    pub(crate) sh_part: u32,
}

/// Emit the FFN pre-norm and the SHARED expert (not the routed ones).
///
/// The shared expert is block-FP8 on the `[32, 32]` ue8m0 grid like every other dense projection,
/// which is why it is 23.19 TFLOP of the 8k prefill and why it lands here on op 198 rather than
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
        //
        // SLOT 0, reusing attention's, because THERE ARE ONLY TWO SLOTS TO SPEND. The host does not
        // read a slot count from the packet: `DevBlob::parse` recovers `slot_bytes` as `max(i[2])`
        // over the collectives (`devblob.rs:191`, "slot A carries 0 and slot B carries the" unit),
        // so an op passing `2 * slot_b` does not buy a third slot -- it redefines the UNIT as
        // twice what every other op meant by it, and the host then binds `act.dg_tp` at
        // `2 * slot_b` while the combine reads `slot_b`. That is what this emit did on its first
        // TP8 run: every reduce past attention read an offset nothing was bound at, and the run
        // came back with 2498560 NaN and a 30-second collective timeout on every iteration.
        //
        // Reusing slot 0 is safe HERE for the reason K3's could not be (`k3.rs:1609`): K3's worry
        // is a ONE-SHOT gate, which says every peer ARRIVED and not that every peer finished
        // READING. Attention's reduce is `XReduceTwoShot` -- reduce-scatter then all-gather -- so
        // a rank cannot complete it until every peer has both contributed and published its band,
        // which is exactly "finished reading slot 0". `AmdTpGroup::run_rung` also drains every rank
        // between segments, and the two reduces are in different segments.
        sh_part: if tp == 1 {
            // A group of one has no peer region, so the down projection writes an ordinary buffer.
            b.tensor(&format!("act.l{l}.sh_out"), (t as u64) * (hidden as u64) * 2)
        } else {
            b.tensor(PEER_SLOT_SHARED, (t as u64) * (hidden as u64) * 2)
        },
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
    // NO CROSS-RANK SUM HERE, and that is the contract, not an omission. `d_moe_combine_pf`
    // computes `out = residual + shared + SUM_slot part` and the band reduce that FOLLOWS it sums
    // `out` across ranks, so the `shared` it is handed must be the row-parallel PARTIAL. GLM says
    // so by construction: its own body emits the shared down straight into `n.shared` with
    // nothing between that GEMM and the combine. Reducing here and passing the reduced buffer
    // made every rank add the FULL shared expert before a sum over 8 ranks -- the shared expert
    // counted `tp` times, which no shape check can see and which cost 40 redundant all-reduces
    // (~29 ms at 8k) on the way. `sh_part` is the answer the caller wants at every `tp`.
    (act, c_down)
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
/// LoRA reads it. Op 195 already does inverse RoPE; it needs the same interior range.
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
/// Emit a CHAIN of layers into one program.
///
/// `layers` is the `--block l..r` range, in order. A layer's output tensor IS its input tensor --
/// `act.hc_residual_a`, because `ri` flips twice per layer -- so the chain needs no copy and no
/// extra buffer between layers; each layer's last op is simply the next layer's dependency.
///
/// The weight table was already per-layer (`Dsv41Weights::per_layer` is indexed by layer id and
/// `declare_dsv41_weights` already took a slice), so a chain declares every layer's tensors and the
/// body picks its own by `l` exactly as a single-layer emit did. A one-element slice reproduces the
/// previous behaviour.
pub(crate) fn emit_dsv41_block(
    c: &Dsv41Cfg,
    layers: &[u32],
    tp: u32,
    n_cu: u32,
    ctx: u32,
    t: u32,
) -> (crate::Model, plow_asset::BlockDescriptor) {
    assert!(!layers.is_empty(), "--block needs at least one layer");
    let l = layers[0];
    for &li in layers {
        let parts = dsv41_layer_parts(c, li);
        let todo: Vec<&str> = parts
            .iter()
            .filter(|(_, st)| *st == Part::Todo)
            .map(|(n, _)| *n)
            .collect();
        assert!(
            todo.is_empty(),
            "layer {li} cannot be emitted as a rung: {} of {} parts are missing -- {todo:?}. \
             Missing capability: `emit_dsv41_block_l{li}`.",
            todo.len(),
            parts.len()
        );
    }
    let (hd, _nope, rope) = dsv41_attn_core_shape(c);
    let nh_l = c.heads / tp;

    let mut tb = Builder::new(n_cu);
    tb.set_tensor_dedup(true);
    let w = declare_dsv41_weights(&mut tb, c, layers, tp);
    let xnext = tb.tensor("act.xnext", (t as u64) * (c.hidden as u64) * 2);
    let pos = tb.tensor("in.pos", (ctx as u64) * 4);
    let kvlen = tb.tensor("in.kvlen", 4);
    // TWO ROPE TABLES, and which one a layer takes is decided by `compress_ratio`, not by what
    // the layer does with it. `Attention.__init__` (`model.py:680-687`) builds ONE `freqs_cis` per
    // layer and every rope in that layer -- the query's, the latent's, the compressor's, the
    // indexer's -- reads it:
    //
    //     if self.compress_ratio:   original_seq_len, rope_theta = args.original_seq_len,
    //                                                              args.compress_rope_theta
    //     else:                     original_seq_len, rope_theta = 0, args.rope_theta
    //                               # "disable YaRN and use base rope_theta in pure
    //                               #  sliding-window attention"
    //
    // So layers 0 and 1 rotate at theta 10000 with no scaling, and layers 2-39 rotate at theta
    // 160000 under YaRN. Handing a layer the other table rotates every query by the wrong angle
    // and still produces fluent output.
    //
    // `YarnDeepSeek`, not `Yarn`: the interpolation is identical and the ATTENTION FACTOR is not.
    // `precompute_freqs_cis` returns `polar(ones_like(freqs), freqs)` -- magnitude exactly 1, no
    // mscale anywhere -- and generic YaRN's `0.1*ln(factor)+1` would be 1.277 at factor 16,
    // multiplied into every cos and sin.
    let rope_tables = |tb: &mut Builder, name: &str, theta: f64, scale: packet::rope::RopeScale| {
        let [cos_t, sin_t] = packet::rope::GenTensor::rope_pair(ctx, rope, theta, 1.0, scale);
        (
            tb.tensor_gen(&format!("in.cos{name}"), cos_t.byte_len(), cos_t),
            tb.tensor_gen(&format!("in.sin{name}"), sin_t.byte_len(), sin_t),
        )
    };
    let (cos, sin) = rope_tables(
        &mut tb,
        "",
        c.raw.rope_theta as f64,
        packet::rope::RopeScale::None,
    );
    let rs = &c.raw.rope_scaling;
    let (cos_c, sin_c) = if layers
        .iter()
        .any(|&l| !matches!(c.raw.attn_kind(l), nn_graph::models::config::V41Attn::Window))
    {
        rope_tables(
            &mut tb,
            "_yarn",
            c.raw.compress_rope_theta as f64,
            packet::rope::RopeScale::YarnDeepSeek {
                factor: rs.factor as f64,
                beta_fast: rs.beta_fast as f64,
                beta_slow: rs.beta_slow as f64,
                orig: rs.original_max_position_embeddings as f64,
                truncate: true,
            },
        )
    } else {
        (cos, sin)
    };
    let mhc = declare_dsv41_mhc(&mut tb, c, t, tp);
    let engram = declare_dsv41_engram(&mut tb, c, layers, t);
    let compress = declare_dsv41_compress(&mut tb, c, layers, t, ctx);
    let index = declare_dsv41_index(&mut tb, c, layers, t, ctx);
    let peer_w = dsv41_peer_width(c, layers);
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();

    let mut b = Builder::new(n_cu);
    b.set_tensor_dedup(true);
    // EXPERT PARALLEL for the routed MoE (PLOW_MOE_PREFILL_EP, opt-in). Under the default TP
    // placement each rank holds all 384 experts sliced to moe_inter/tp = 288, and 288 is what makes
    // `down`'s k-loop too short to amortise the grouped GEMM's per-tile expert-weight reload --
    // the term 12.63 measured as the MoE pair's binding cost. EP gives each rank 384/tp = 48 WHOLE
    // experts at the full 2304 instead.
    //
    // It needs no new collective: every rank already holds the whole replicated residual, so each
    // one runs only its LOCAL experts and writes zeros elsewhere, and the XReduce that already sums
    // the shared-expert partials folds the routed ones in the SAME reduction. `Builder::finish`
    // does the whole-graph rewrite (`rewrite_replicated_moe_prefill_ep`), which finds the
    // align/GLU/down/combine -> TP-reduction chains, swaps the expert weight/scale tables for `_ep`
    // companions the runtime binds with local bases, and derives the full width as i[0] * degree --
    // so the emit below keeps its TP-sliced `moe_inter / tp` and does NOT pre-divide anything.
    b.set_moe_prefill_ep_degree(
        (crate::emit_is_amd() && crate::emit_config::active().moe_prefill_ep).then_some(tp),
    );
    b.adopt_tensors(tensors);
    let all = b.all();
    let mut xgate = 0u32;

    // The residual stream starts in copy 0. Each sublayer reads one copy and its POST writes the
    // other, which is why `ri` flips twice per layer -- once for attention, once for the FFN.
    let mut ri = 0usize;

    // NO SEED OP. The entry IS `act.hc_residual_a`, the mHC stream itself, all `hc_mult` copies of
    // it, uploaded by the harness.
    //
    // The seed this replaces wrote `t * hidden` elements into a `[hc_mult][T][hidden]` buffer, so
    // copies 1..hc_mult were never written and `HyperConnPre` mixed three copies of whatever the
    // arena held. On hardware that is exactly what it looked like: 167104 Inf out of 20971520, on
    // a layer whose input was bounded by +-0.04.
    //
    // It could not be fixed by widening `i[0]`: `Residual` reads `t[1]`/`t[2]` for as many elements
    // as it writes, so a 4-copy write would run 3 copies off the end of a 1-copy `act.x`. There is
    // no output offset on the op to write the copies one at a time with. And the seed was a
    // RUNG-ONLY fiction in the first place -- in a whole model the embedding produces this stream
    // -- so the honest entry for a rung is the stream, not a hidden state plus a fake expansion.
    // The previous layer's last op. Empty for the first, so a one-layer chain emits exactly the
    // instruction stream it did before this was a loop.
    // A RUNG THAT READS A CACHE NOTHING IN IT WROTE. `--block 3` is a legal one-layer rung and
    // layer 3 attends over the cache layer 2 published; in a chain that starts at 3 nothing
    // publishes it, so op 55 gathers from whatever the arena holds. That is a rung's usual
    // bargain -- the entry residual is uploaded too -- but it is the difference between a
    // timing artifact and a numerical one, so it is said out loud rather than discovered.
    {
        let first_src = layers.iter().position(|l| c.kv_source.contains(l));
        let first_rd = layers.iter().position(|&l| {
            !matches!(
                c.raw.attn_kind(l),
                nn_graph::models::config::V41Attn::Window
            )
        });
        // `act.index_k` has its OWN owner list: the four `kv_source` layers derive index keys
        // from their compressor latent, and the other four index layers read them. A chain
        // starting at 24 runs an indexer against arena state, which is not the same gap as
        // reading an unwritten cache and is not covered by the same test.
        let first_idx = layers.iter().position(|l| c.index_source.contains(l));
        let unwritten = match (first_rd, first_src) {
            (Some(rd), src) => src.is_none_or(|s| s > rd),
            (None, _) => false,
        } || match (first_idx, first_src) {
            (Some(ix), src) => src.is_none_or(|s| s > ix),
            (None, _) => false,
        };
        if unwritten {
            eprintln!(
                "  NOTE: this chain reads shared CSA2 state that no earlier layer in it writes, \
                 so `act.compress_kv`, `act.index_k` and `act.index_idx` enter as arena state. \
                 TIME this blob, do not read it; a chain starting at or before the relevant \
                 kv_source ({:?}) is the numerically meaningful one.",
                c.kv_source
            );
        }
    }
    let mut deps: Vec<u32> = Vec::new();
    // The sublayer index, counting attention and FFN separately: it is what picks the `pre_pair`
    // half, so it must advance twice per layer and never reset. See `emit_dsv41_mhc_pre`.
    let mut pi = 0usize;
    for &l in layers {
        // BEFORE the block, in place on the residual stream's hc copies -- `Transformer.forward`
        // runs Engram in the layer loop ahead of `layer(...)`, not inside it (model.py:1262-1267).
        // Its completion becomes the mHC pre's only dependency, so the chain stays a chain.
        if c.engram_layers.contains(&l) {
            let e = engram
                .as_ref()
                .expect("declare_dsv41_engram saw this layer in `layers`");
            deps = vec![emit_dsv41_engram(
                &mut b,
                c,
                &w,
                &all,
                e,
                l,
                tp,
                t,
                mhc.residual[ri],
                &mut xgate,
                &deps,
            )];
        }
        let sp = dsv41_sp(t, tp);
        let xr_sp = super::xr_cus_capped(b.n_cu(), &all);
        let slot3 = 3 * (t as u64 * peer_w as u64 * 2) as u32;
        // Gather this rank's `layer_input` band out of peer slot 3 into the whole `[t, hidden]`
        // the GEMM blocks read. Attention is head-parallel and the MoE is expert-parallel, so
        // both need EVERY token; the mHC either side of them does not.
        let sp_gather = |b: &mut Builder, xg: &mut u32, dep: u32| -> u32 {
            crate::emit_xall_gather(
                b,
                xg,
                &xr_sp,
                &[dep],
                &[(mhc.layer_input, t * c.hidden, slot3)],
                tp,
            )
        };
        let c_pre = emit_dsv41_mhc_pre(&mut b, c, &w, &mhc, l, false, ri, pi, t, tp, &deps);
        pi += 1;
        let c_pre = if sp {
            sp_gather(&mut b, &mut xgate, c_pre)
        } else {
            c_pre
        };
        let (proj, c_proj) =
            emit_dsv41_attn_proj(&mut b, c, &w, &all, l, tp, mhc.layer_input, t, &[c_pre]);
        // Every rope in a layer reads that layer's ONE `freqs_cis`, and which table that is comes
        // from `compress_ratio` alone -- see the two tables above.
        let (lcos, lsin) = if matches!(
            c.raw.attn_kind(l),
            nn_graph::models::config::V41Attn::Window
        ) {
            (cos, sin)
        } else {
            (cos_c, sin_c)
        };
        // CSA2's write side, off the same normed input the projections read (`Compressor` takes
        // `Attention.forward`'s `x`, as `wq_a` and `wkv` do). It joins the attention core's
        // dependency list even though the core does not read the cache yet: the cache is shared
        // with LATER layers, and an op outside the chain's dependency order is an op the
        // scheduler may float past the layer that reads it.
        let mut core_deps = vec![c_proj];
        if c.kv_source.contains(&l) {
            let cp = compress
                .as_ref()
                .expect("declare_dsv41_compress saw this layer in `layers`");
            core_deps.push(emit_dsv41_compressor(
                &mut b, c, &w, &all, cp, l, proj.xn, lcos, lsin, t, &[c_proj],
            ));
        }
        // THE SELECTION, on the 8 `index_source` layers. Every other layer REUSES the one its
        // source published -- `_compress_topk_idxs` returns `shared_attn.topk_idxs` unchanged
        // when `is_index_source` is false (model.py:722-731) -- so the emit runs an indexer
        // exactly where the reference constructs one, and the layers between simply read the
        // table. That reuse is why 8 indexers serve 38 readers.
        if c.index_source.contains(&l) {
            let ix = index
                .as_ref()
                .expect("declare_dsv41_index saw this layer in `layers`");
            core_deps.push(emit_dsv41_indexer(
                &mut b,
                c,
                &w,
                &all,
                ix,
                compress.as_ref(),
                l,
                proj.xn,
                proj.q_an,
                kvlen,
                lcos,
                lsin,
                t,
                ctx,
                &core_deps.clone(),
            ));
        }
        // `compress_ratios[l]`, not `kv_source`: who WRITES the cache and who READS it disagree
        // by construction, and reading the writer list is what left 38 layers window-only.
        let compressed = match c.raw.attn_kind(l) {
            nn_graph::models::config::V41Attn::Window => None,
            nn_graph::models::config::V41Attn::Compressed { ratio } => {
                let cp = compress.as_ref().expect("a compressed layer needs a cache");
                let ix = index.as_ref().expect("a compressed layer needs a selection");
                Some((cp.cache, ix.uni, dsv41_index_topk(c, t, ratio)))
            }
        };
        let (core, c_core) = emit_dsv41_attn_core(
            &mut b, c, &w, &all, l, tp, proj.q, proj.kv, kvlen, pos, lcos, lsin, compressed, t,
            ctx, &core_deps,
        );
        let (_out, c_out) =
            emit_dsv41_attn_out(&mut b, c, &w, &all, l, tp, core.o, t, &mut xgate, &[c_core]);
        // Under SP the attention answer never becomes `_out.o`: the seam reduce-SCATTERED it, so
        // this rank's rows are the band of the `act.og_tp` partial and the mHC reads them there.
        let attn_out = if sp {
            dsv41_band(&mut b, _out.o_part, t, tp, c.hidden as u64 * 2)
        } else {
            _out.o
        };
        let c_post = emit_dsv41_mhc_post(&mut b, c, &mhc, attn_out, ri, t, tp, &c_out);
        ri ^= 1;

        let c_pre2 = emit_dsv41_mhc_pre(&mut b, c, &w, &mhc, l, true, ri, pi, t, tp, &[c_post]);
        pi += 1;
        let c_pre2 = if sp {
            sp_gather(&mut b, &mut xgate, c_pre2)
        } else {
            c_pre2
        };
        let (ffn, c_sh) = emit_dsv41_ffn_shared(
            &mut b, c, &w, &all, l, tp, mhc.layer_input, t, &[c_pre2],
        );
        let c_moe = emit_dsv41_moe(
            &mut b, c, &w, l, tp, t, xnext, ffn.xn, c_sh, (ffn.sh_part, c_sh), &mut xgate, &all,
            peer_w,
        );
        // CAPTURED, not discarded: it is the next layer's only dependency, and the thing that
        // makes the chain a chain rather than 40 layers racing on one residual buffer.
        // Same for the FFN seam: `emit_dsv41_moe` returns the reduce-scatter under SP and leaves
        // this rank's rows in `act.dg_tp`, so `xnext` is never written and never read.
        let moe_out = if sp {
            let dg = dsv41_sp_slot(&mut b, PEER_SLOT_MOE, (t as u64) * (c.hidden as u64) * 2);
            dsv41_band(&mut b, dg, t, tp, c.hidden as u64 * 2)
        } else {
            xnext
        };
        let c_layer = emit_dsv41_mhc_post(&mut b, c, &mhc, moe_out, ri, t, tp, &[c_moe]);
        ri ^= 1;
        deps = vec![c_layer];
    }

    // PLOW_DSV41_OPS=<n>: emit only the first n ops of the layer. A profiling cut, not a
    // feature -- the run times of successive prefixes difference into a per-op cost, which is
    // the only way to get one out of a megakernel interpreter.
    if let Some(n) = crate::emit_config::active().dsv41_ops {
        b.truncate_ops(n as usize);
        eprintln!(
            "  PLOW_DSV41_OPS={n}: emitting a PREFIX of the layer, for profiling only. The two \
             early reduces move to slot_b so a truncated packet still loads, so this blob's \
             NUMERICS ARE NOT THE MODEL'S -- time it, do not read it."
        );
    }
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
        // `[hc_mult][T][hidden]`, NOT `[T][hidden]`. The mHC carries `hc_mult` parallel residual
        // streams and `HyperConnPre` mixes all of them; a descriptor claiming `[T][hidden]` would
        // have a harness upload one copy and leave the rest to the arena.
        inputs: vec![BlockTensor {
            name: "act.hc_residual_a".into(),
            shape: vec![
                Dim::Fixed(c.hc_mult as i64),
                Dim::Symbolic("T".into()),
                Dim::Fixed(hidden),
            ],
            dtype: "bf16".into(),
        }],
        outputs: vec![BlockTensor {
            name: out_name.into(),
            shape: vec![
                Dim::Fixed(c.hc_mult as i64),
                Dim::Symbolic("T".into()),
                Dim::Fixed(hidden),
            ],
            dtype: "bf16".into(),
        }],
        // Layer 0 carries nothing between calls: pure sliding window, one chunk per run.
        carried_state: Vec::new(),
        weights: BlockWeights {
            mode: "symlink".into(),
            ckpt: "DeepSeek-V4.1-Flash".into(),
            // A chain draws on every layer it emits, so naming one of them would send a reader
            // looking for a single-layer blob.
            prefix: if layers.len() == 1 {
                format!("layers.{l}.")
            } else {
                "layers.".to_string()
            },
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
    // "mhc_pre" here means the CROSS-SUBLAYER one. It read Done for as long as this emit bound
    // V4.1 onto GLM-5.3's same-sublayer `hc_pre`, which `scripts/dsv41_mhc_oracle.py` prices at
    // 51.9% relative on layer 0 alone -- V4.1 collapses layer 0's attention with a ONE-HOT
    // (`make_identity_pre_mix`) where that emit used a learned mix off `hc_attn_fn`. It is Done
    // again only because `MhcPre::{Seed, Deferred}` now carries the deferral; a caller that
    // reverts to `MhcPre::Own` makes this line a lie.
    //
    // STILL MISSING, and NOT per-layer so it has no row here: the model's TAIL. After the last
    // block `Transformer.forward` spends the dangling `ffn_pre` on one final `hc_pre`
    // (`model.py:1268`). A block emit's output is the residual stream, so nothing is wrong yet,
    // but a whole-model emit owes that collapse.
    let mut p = vec![];

    // ENGRAM RUNS BEFORE THE BLOCK, NOT INSIDE IT. This row used to sit between `ffn_norm` and the
    // MoE, which is where a reading of `Block.__init__` puts it -- the module is constructed on
    // the Block. But `Block.forward` never calls it. `Transformer.forward` does, in the layer
    // loop and BEFORE the block:
    //
    //     if layer.engram is not None:
    //         h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], mask)
    //     h, pre_mix = layer(h, start_pos, pre_mix, image_mask)   # model.py:1262-1267
    //
    // so it reads and writes the hc-EXPANDED residual stream (`[T, hc_mult, dim]`, op 196 is in
    // place on it) before this layer's mHC pre ever runs -- not the post-attention activation an
    // FFN-sublayer position would hand it. `scripts/dsv41_engram_oracle.py` confirms ops 196/197
    // themselves are the reference's, to 0.000e+00 on the embed and 2.2e-16 on the gate, so the
    // whole of what is left here is placement and plumbing.
    if c.engram_layers.contains(&l) {
        p.push((
            "engram gather + all-reduce + wkv + gate (ops 197/198/196), BEFORE mhc_pre",
            Part::Done,
        ));
    }

    p.push(("mhc_pre (ops 128/129, cross-sublayer `pre`)", Part::Done));
    p.push(("attn_norm + q_a/q_b/wkv projections (op 198)", Part::Done));
    if c.kv_source.contains(&l) {
        // Ops 194 (arm 2: pool, norm, STOP) and 199 (rope + fp4/E4M3 quant), or at layer 20's
        // ratio 1 a plain GEMM and RMSNorm with no op 194 at all. NOT 195 -- that is the inverse
        // rope on the attention OUTPUT, which every layer runs and which now sits in the core.
        p.push(("csa2 compressor (ops 194/199), writes the shared cache", Part::Done));
    }
    if c.index_source.contains(&l) {
        // Ops 117/118, with the new pool-granular causal bound. "Two-level" is the candidate
        // stage, and at 8k it is INERT -- `candidate_topk_blocks` 2048 x `candidate_block_size`
        // 8 covers 16384 compressed positions and an 8k context has at most 8192, so level one
        // keeps every reachable block and the mask level two applies is a SUPERSET of the
        // reachability mask already applied. `emit_dsv41_indexer` asserts that rather than
        // assuming it; a longer context trips the assert instead of quietly over-selecting.
        p.push((
            "indexer queries + keys + top-k (ops 117/118), publishes the shared selection",
            Part::Done,
        ));
    }
    p.push((
        "attention core (interior rope + windowed absorbed MLA + sink merge + inverse rope)",
        Part::Done,
    ));
    // THE READ SIDE OF CSA2, which this table used to omit entirely.
    //
    // `kv_source` says who WRITES the shared cache -- 4 layers. `compress_ratios[l]` says which
    // cache layer `l` READS, and 0 is the only value meaning "sliding window only". They
    // "disagree by construction" (this module's own header), and the table consulted the writer
    // list alone: every layer that merely reads the cache was marked fully Done, and the emit
    // above dispatches `FlashMlaPrefill` with `KV_MASK_NONE` and nothing but the 128-token
    // window. That is 38 of 40 layers silently attending over a fraction of what they should --
    // exactly the "loads, runs, produces fluent-looking garbage" outcome the refusal exists to
    // prevent, produced BY the refusal saying the layer was complete.
    //
    // Only layers 0 and 1 are genuinely window-only (`compress_ratios` is [0, 0, 2 x18, 1 x20,
    // 0, 0, 0] -- the trailing three are the DSpark blocks, not layers).
    if !matches!(c.raw.attn_kind(l), nn_graph::models::config::V41Attn::Window) {
        p.push((
            "compressed-KV attention (op 55 beside the window flash, merged at nsplit 2)",
            Part::Done,
        ));
    }
    p.push(("output projection wo_a + wo_b (op 198)", Part::Done));
    p.push(("output all-reduce (XReduce, wo_b is input-parallel)", Part::Done));
    p.push(("ffn_norm", Part::Done));
    p.push(("moe router + routed experts (ops 85/86, MXFP4)", Part::Done));
    p.push(("shared expert (op 198 + clamped SwiGLU)", Part::Done));
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
         Mostly the emit is the gap -- ops 194/195, 196/197, 198 and 55 exist and pass on gfx942 \
         -- but not entirely, and the one kernel difference is easy to miss: `op_compress.h` \
         implements V4's contract, where the fake-quant rounds the per-block scale to a POWER OF \
         TWO. V4.1's compressed KV wants an E4M3 scale at group 16 (`fp4_act_quant(latent, 16, \
         True, scale_dtype=torch.float8_e4m3fn)`, inference/model.py:672). `qblk` already takes \
         16 and ROTATE already selects e2m1, so the gap is the scale format alone. \
         Missing capability: `emit_dsv41_block`.",
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
    // -- one row per hyper-connection copy, which is what op 196 takes as `qw`/`kw`.
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
