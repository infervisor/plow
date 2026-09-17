//! DeepSeek-V4.1-Flash configuration (`model_type: "deepseek_v41"`).
//!
//! V4.1 is **not** a retune of [`super::DeepSeekV4Config`]. It keeps V4's MoE
//! shape, its grouped output LoRA and its mHC residual stream, and replaces
//! essentially everything about how a layer obtains KV. Each difference below
//! silently produces a plausible wrong model if it is defaulted away, so none
//! of them has a `#[serde(default)]`:
//!
//! * **CSA2 shares one KV cache across layers.** In V4 a nonzero
//!   `compress_ratios[l]` meant layer `l` runs its own compressor. In V4.1 it
//!   does not: only `kv_source_layer_ids` compress, and every other layer reads
//!   that cache. `compress_ratios[l]` now says which cache layer `l` reads
//!   (`0` = sliding window only). The checkpoint proves it — 40 layers carry
//!   `attn.wq_a`, but only **4** carry `attn.compressor.*`.
//! * **`compress_ratio == 1` is legal here.** V4 rejects it as "use 0 to mean
//!   no compressor". In V4.1 the 20 decoder layers run at ratio 1: a plain
//!   per-token projection with no softmax pooling, which is why layer 20's
//!   compressor ships `wkv` and `norm` but **no `wgate`** (3 `wgate` tensors
//!   for 4 compressors). Reusing V4's validator would reject the released
//!   checkpoint.
//! * **A two-level index.** `index_source_layer_ids` (8 layers) carry an
//!   indexer, but only those that are also `kv_source_layer_ids` (4) own index
//!   keys — hence 8 `indexer.wq_b` against 4 `indexer.wk`. From
//!   `candidate_source_layer_id` onward the indexer scores only inside the
//!   candidate blocks that layer selected, `candidate_topk_blocks` of
//!   `candidate_block_size` each.
//! * **Engram conditional memory.** `engram_layer_ids` names the two layers
//!   carrying an n-gram lookup table. These are the checkpoint's largest
//!   tensors by far: `engram_num_embeddings` is per-layer and in the hundreds
//!   of millions, one fp8 row of `engram_head_dim` each.
//! * **Block-FP8 at a [32,32] grid.** V4 is `[128, 128]`. A `[32, 32]` grid is
//!   16x as many scales, and the V4 validator hard-codes `[128, 128]`.
//! * **`quantization_config.expert_dtype`.** V4 carries `expert_dtype` at the
//!   top level and a `fmt` field inside `quantization_config`; V4.1 moves the
//!   former inside and drops the latter, so neither struct deserializes the
//!   other's document.
//!
//! # Reconciling `compress_ratios` with the layer count
//!
//! As in V4, the trailing entries are the DSpark blocks:
//! `num_hidden_layers` (40) + `dspark_target_layer_ids.len()` (3) = 43.
//! `num_nextn_predict_layers` counts speculative ITERATIONS and does not
//! reconcile the length.

use serde::Deserialize;

use super::ConfigError;

/// Elements per E8M0 scale in the MXFP4 routed-expert encoding. Fixed by the
/// OCP microscaling format, not by `config.json`.
pub const V41_MXFP4_GROUP: i64 = 32;

/// `quantization_config`. Note this carries `expert_dtype` (V4 has it at the
/// top level) and has no `fmt` field (V4 requires one), so the two documents
/// are not interchangeable.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV41QuantizationConfig {
    pub activation_scheme: String,
    pub quant_method: String,
    pub scale_fmt: String,
    pub weight_block_size: [i64; 2],
    /// Storage encoding of the ROUTED experts only; every projection and the
    /// shared expert stay block-FP8.
    pub expert_dtype: String,
}

/// `rope_scaling` — YaRN over `original_max_position_embeddings`.
#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV41RopeScaling {
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub factor: f32,
    pub original_max_position_embeddings: i64,
    #[serde(rename = "type", alias = "rope_type")]
    pub kind: String,
}

/// What a layer's attention reads, beyond its own sliding window.
///
/// This is a statement about the CACHE the layer reads, not about work it
/// does: only a `kv_source_layer_ids` layer writes that cache.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum V41Attn {
    /// `compress_ratio == 0`: sliding window only, no compressed cache read.
    Window,
    /// Reads the compressed cache built at `ratio` tokens per entry. `ratio`
    /// of 1 is a per-token projection, not a pooling.
    Compressed { ratio: u32 },
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeepSeekV41Config {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    /// **1** on the released checkpoint: one shared KV head, so the KV latent
    /// is replicated under tensor parallelism rather than sharded.
    pub num_key_value_heads: u32,
    /// Per-head query dim AND the width of the single KV latent.
    pub head_dim: u32,
    /// RoPE-carrying prefix of `head_dim`; the rest is the content part.
    pub qk_rope_head_dim: u32,
    pub q_lora_rank: u32,
    pub rms_norm_eps: f32,
    pub attention_bias: bool,
    pub hidden_act: String,
    pub max_position_embeddings: i64,
    pub tie_word_embeddings: bool,

    // --- grouped output LoRA (replaces `o_proj`) ---
    pub o_lora_rank: u32,
    pub o_groups: u32,

    // --- CSA2 ---
    /// One entry per layer, then one per DSpark block. `0` = sliding window
    /// only; `r > 0` = reads the cache at `r` tokens per entry. Unlike V4,
    /// `1` is a legal value and means a per-token projection.
    pub compress_ratios: Vec<u32>,
    /// RoPE base for the compressed cache — deliberately different from
    /// `rope_theta`, since compressed entries span `ratio` positions.
    pub compress_rope_theta: f32,
    /// Local span kept uncompressed alongside the compressed summary.
    pub sliding_window: u32,
    /// The only layers that RUN a compressor and write the shared KV cache.
    pub kv_source_layer_ids: Vec<u32>,

    // --- two-level lightning indexer ---
    /// Layers carrying an indexer. A superset of `kv_source_layer_ids`; the
    /// ones that are not also KV sources read index keys rather than owning
    /// `indexer.wk` / `indexer.k_norm`.
    pub index_source_layer_ids: Vec<u32>,
    pub index_head_dim: u32,
    pub index_n_heads: u32,
    pub index_topk: u32,
    /// Layer whose indexer selects the coarse candidate blocks that every
    /// LATER indexer then scores inside.
    pub candidate_source_layer_id: u32,
    pub candidate_topk_blocks: u32,
    pub candidate_block_size: u32,

    // --- RoPE ---
    pub rope_theta: f32,
    pub rope_scaling: DeepSeekV41RopeScaling,

    // --- MoE ---
    pub n_routed_experts: u32,
    pub n_shared_experts: u32,
    pub num_experts_per_tok: u32,
    pub moe_intermediate_size: i64,
    pub scoring_func: String,
    pub topk_method: String,
    pub norm_topk_prob: bool,
    pub routed_scaling_factor: f32,
    /// Clamp on the SwiGLU branches; 0 would silently mean "no clamp".
    pub swiglu_limit: f32,

    // --- mHC residuals ---
    pub hc_eps: f32,
    /// Residual-stream expansion. `hc_*_fn` reads `hc_mult * hidden_size`.
    pub hc_mult: u32,
    pub hc_sinkhorn_iters: u32,

    // --- Engram conditional memory ---
    /// Layers carrying an n-gram lookup table.
    pub engram_layer_ids: Vec<u32>,
    /// Rows in each layer's table, in `engram_layer_ids` order. These are the
    /// checkpoint's largest tensors.
    pub engram_num_embeddings: Vec<i64>,
    pub engram_max_ngram_size: u32,
    pub engram_vocab_size: i64,
    pub engram_n_heads: u32,
    pub engram_head_dim: i64,
    pub engram_pad_token_id: i64,
    pub engram_compressed_vocab_size: i64,

    // --- DSpark speculative module ---
    pub num_nextn_predict_layers: u32,
    pub dspark_block_size: u32,
    pub dspark_noise_token_id: i64,
    /// Main-model layers whose hidden states feed DSpark. Its LENGTH is the
    /// number of MTP blocks in the checkpoint.
    pub dspark_target_layer_ids: Vec<u32>,
    pub dspark_markov_rank: i64,
    /// The DSpark blocks carry their OWN, narrower MoE.
    pub dspark_n_routed_experts: u32,
    pub dspark_num_experts_per_tok: u32,

    #[serde(alias = "dtype")]
    pub torch_dtype: Option<String>,
    pub quantization_config: DeepSeekV41QuantizationConfig,
}

impl DeepSeekV41Config {
    /// Non-RoPE (content) portion of `head_dim`.
    pub fn qk_nope_head_dim(&self) -> u32 {
        self.head_dim - self.qk_rope_head_dim
    }

    /// Width of the cached KV latent, per token per layer.
    pub fn kv_latent_dim(&self) -> i64 {
        self.num_key_value_heads as i64 * self.head_dim as i64
    }

    /// Query heads in one output-LoRA group.
    pub fn heads_per_o_group(&self) -> u32 {
        self.num_attention_heads / self.o_groups
    }

    /// Input width of one output-LoRA group's down projection.
    pub fn o_group_in_features(&self) -> i64 {
        self.heads_per_o_group() as i64 * self.head_dim as i64
    }

    /// Number of DSpark MTP blocks — the length of `dspark_target_layer_ids`,
    /// NOT `num_nextn_predict_layers`.
    pub fn dspark_blocks(&self) -> usize {
        self.dspark_target_layer_ids.len()
    }

    /// What base layer `layer` reads. `validate()` guarantees the index is in
    /// range.
    pub fn attn_kind(&self, layer: u32) -> V41Attn {
        match self.compress_ratios[layer as usize] {
            0 => V41Attn::Window,
            ratio => V41Attn::Compressed { ratio },
        }
    }

    /// Whether `layer` runs a compressor and writes the shared KV cache.
    pub fn is_kv_source(&self, layer: u32) -> bool {
        self.kv_source_layer_ids.contains(&layer)
    }

    /// Whether `layer` carries an indexer.
    pub fn is_index_source(&self, layer: u32) -> bool {
        self.index_source_layer_ids.contains(&layer)
    }

    /// Whether `layer`'s compressor pools (softmax gate) rather than
    /// projecting one token per entry. Only a pooling compressor ships
    /// `attn.compressor.wgate`.
    pub fn compressor_has_gate(&self, layer: u32) -> bool {
        self.is_kv_source(layer) && self.compress_ratios[layer as usize] > 1
    }

    /// Engram's n-gram hash tables: one multiplier per (engram layer, lookback), and one
    /// prime-sized bucket range per (engram layer, n-gram size, head) with the offset it
    /// starts at.
    ///
    /// THE PRIMES ARE DERIVED AND SELF-CHECKING. The reference walks upward from
    /// `engram_vocab_size - 1` with `sympy.isprime`, handing each (n-gram size, head) pair the
    /// next prime never yet handed out -- a single GLOBAL walk across all engram layers, which
    /// is what keeps the ranges disjoint. A plain trial-division walk reproduces it, and the
    /// result proves itself: each layer's primes must SUM to that layer's
    /// `engram_num_embeddings`, because the table is exactly those ranges laid end to end.
    /// [`Self::validate`] asserts it, so a wrong walk cannot reach a lookup.
    ///
    /// THE MULTIPLIERS ARE CONSTANTS, and that is deliberate. The reference draws them from
    /// `np.random.default_rng(10007 * layer_id).integers(...)` -- numpy's PCG64 plus its
    /// bounded-integer algorithm. Reproducing that bit-exactly in Rust is a real undertaking
    /// whose failure mode is silent: every hash comes out wrong and the model merely gets worse.
    /// The draw is also TINY -- one value per (layer, lookback), eight numbers for the released
    /// checkpoint -- so they are what they actually are, fixed constants of that checkpoint,
    /// extracted once rather than re-derived. Any OTHER V4.1 checkpoint needs its own extraction;
    /// [`Self::engram_hash_tables`] returns `None` rather than guessing when the shape it was
    /// extracted for does not match.
    pub fn engram_hash_tables(&self) -> Option<EngramHashTables> {
        /// Extracted from the released DeepSeek-V4.1-Flash with numpy, indexed
        /// `[engram layer][lookback]`. Valid only for `engram_layer_ids == [1, 14]`,
        /// `engram_max_ngram_size == 4` and `engram_compressed_vocab_size == 99_092`, since the
        /// draw's bound is `(i64::MAX / compressed_vocab) / 2`.
        const MULTIPLIERS: [[i64; 4]; 2] = [
            [76_632_096_046_245, 4_839_876_093_313, 35_959_672_319_349, 73_987_337_458_391],
            [67_716_810_739_261, 51_510_806_800_915, 30_921_347_202_721, 82_619_226_485_591],
        ];
        if self.engram_layer_ids != [1, 14]
            || self.engram_max_ngram_size != 4
            || self.engram_compressed_vocab_size != 99_092
        {
            return None;
        }

        let per_layer = (self.engram_max_ngram_size as usize - 1) * self.engram_n_heads as usize;
        let mut seen: Vec<i64> = Vec::with_capacity(per_layer * self.engram_layer_ids.len());
        let mut primes = Vec::with_capacity(self.engram_layer_ids.len());
        let mut offsets = Vec::with_capacity(self.engram_layer_ids.len());

        for _ in &self.engram_layer_ids {
            let mut flat = Vec::with_capacity(per_layer);
            for _ in 0..self.engram_max_ngram_size - 1 {
                // `current` restarts at the vocab bound for every n-gram size, but `seen`
                // persists -- so each search skips past every prime already handed out.
                let mut current = self.engram_vocab_size - 1;
                for _ in 0..self.engram_n_heads {
                    loop {
                        current += 1;
                        if is_prime(current) && !seen.contains(&current) {
                            break;
                        }
                    }
                    seen.push(current);
                    flat.push(current);
                }
            }
            let mut offs = Vec::with_capacity(per_layer);
            let mut acc = 0i64;
            for p in &flat {
                offs.push(acc);
                acc += *p;
            }
            primes.push(flat);
            offsets.push(offs);
        }

        Some(EngramHashTables {
            multipliers: MULTIPLIERS.iter().map(|r| r.to_vec()).collect(),
            primes,
            offsets,
        })
    }

    /// Rows in `layer`'s Engram table, if it has one.
    pub fn engram_rows(&self, layer: u32) -> Option<i64> {
        let at = self.engram_layer_ids.iter().position(|l| *l == layer)?;
        self.engram_num_embeddings.get(at).copied()
    }

    /// `[out_blocks, in_blocks]` of a block-FP8 ue8m0 scale grid.
    pub fn fp8_scale_shape(&self, out_features: i64, in_features: i64) -> [i64; 2] {
        let [out_block, in_block] = self.quantization_config.weight_block_size;
        [
            (out_features + out_block - 1) / out_block,
            (in_features + in_block - 1) / in_block,
        ]
    }

    /// `[out_features, in_features / 32]` — the MXFP4 routed-expert scale grid.
    pub fn mxfp4_scale_shape(&self, out_features: i64, in_features: i64) -> [i64; 2] {
        [
            out_features,
            (in_features + V41_MXFP4_GROUP - 1) / V41_MXFP4_GROUP,
        ]
    }

    /// Every piece this compiler cannot lower yet, listed. The config parses
    /// and validates; the *emit* refuses, so the gap is a checklist rather
    /// than a dead end at `config.json`.
    pub fn unimplemented(&self) -> String {
        let mut ratios: Vec<u32> = self.compress_ratios.iter().copied().filter(|r| *r > 0).collect();
        ratios.sort_unstable();
        ratios.dedup();
        format!(
            "deepseek_v41 graph emit is not implemented (config parses and validates; refusing \
             a DeepSeek-V4 or V3 fallback, which would build a plausible wrong model out of \
             V4.1 weights). Unimplemented: \
             (1) CSA2 cache SHARING -- {nkv} kv_source layers ({kv:?}) write one compressed \
             cache at ratios {ratios:?} that all {layers} layers read, so the per-layer \
             compressor the V4 kernels assume is the wrong shape; \
             (2) the ratio-1 compressor (a per-token projection with no softmax gate, \
             {ngate} of {nkv} compressors carry `wgate`); \
             (3) the TWO-LEVEL indexer -- {nidx} indexers ({idx:?}) over {kheads}x{khd} index \
             heads with top-{topk}, of which only the {nkv} kv_source layers own index keys, \
             and layers after {cand} score only inside {cblocks} candidate blocks of \
             {cbsz} positions; \
             (4) Engram conditional memory at layers {eng:?} ({erows:?} fp8 rows of {ehd}, \
             {enheads} heads, up to {engram}-grams over a {evocab}-entry vocabulary); \
             (5) single-pass mHC (hc_mult={mult}, {sink} Sinkhorn iterations) where each \
             sublayer's mix feeds the NEXT sublayer, not its own; \
             (6) block-FP8 at a {blk:?} scale grid with {edt} routed experts -- the existing \
             fp8 arms are written for V4's [128,128] grid; \
             (7) SWA Bounded Replay over the {win}-token window; \
             (8) the attached DSpark module ({blocks} MTP blocks, its own {dexp}-expert \
             top-{dtopk} MoE, markov rank {mrank}, block size {bsz}) -- OPTIONAL for a first \
             bringup, which is bit-exact on base output without it. \
             Still missing on the EMITTER side: a deepseek_v41 graph builder",
            layers = self.num_hidden_layers,
            nkv = self.kv_source_layer_ids.len(),
            kv = self.kv_source_layer_ids,
            ngate = self
                .kv_source_layer_ids
                .iter()
                .filter(|l| self.compressor_has_gate(**l))
                .count(),
            nidx = self.index_source_layer_ids.len(),
            idx = self.index_source_layer_ids,
            topk = self.index_topk,
            kheads = self.index_n_heads,
            khd = self.index_head_dim,
            cand = self.candidate_source_layer_id,
            cblocks = self.candidate_topk_blocks,
            cbsz = self.candidate_block_size,
            eng = self.engram_layer_ids,
            erows = self.engram_num_embeddings,
            ehd = self.engram_head_dim,
            enheads = self.engram_n_heads,
            engram = self.engram_max_ngram_size,
            evocab = self.engram_vocab_size,
            mult = self.hc_mult,
            sink = self.hc_sinkhorn_iters,
            blk = self.quantization_config.weight_block_size,
            edt = self.quantization_config.expert_dtype,
            win = self.sliding_window,
            blocks = self.dspark_blocks(),
            dexp = self.dspark_n_routed_experts,
            dtopk = self.dspark_num_experts_per_tok,
            mrank = self.dspark_markov_rank,
            bsz = self.dspark_block_size,
        )
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        let bad = |message: String| Err(ConfigError::Unsupported(message));

        if self.num_hidden_layers == 0
            || self.hidden_size <= 0
            || self.vocab_size <= 0
            || self.moe_intermediate_size <= 0
            || self.head_dim == 0
            || self.num_attention_heads == 0
        {
            return bad("deepseek_v41 dimensions must be positive".into());
        }

        // --- attention geometry ---
        if self.qk_rope_head_dim == 0 || self.qk_rope_head_dim >= self.head_dim {
            return bad(format!(
                "deepseek_v41 qk_rope_head_dim={} must be a proper prefix of head_dim={}",
                self.qk_rope_head_dim, self.head_dim
            ));
        }
        if self.num_key_value_heads == 0
            || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return bad(format!(
                "deepseek_v41 num_attention_heads={} must be a positive multiple of \
                 num_key_value_heads={}",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if self.q_lora_rank == 0 {
            return bad(
                "deepseek_v41 requires a query LoRA; q_lora_rank=0 has no wq_a/q_norm to bind"
                    .into(),
            );
        }
        // The output LoRA is block-diagonal over `o_groups`, so a group is the
        // indivisible unit of head assignment.
        if self.o_groups == 0 || self.num_attention_heads % self.o_groups != 0 {
            return bad(format!(
                "deepseek_v41 o_groups={} must divide num_attention_heads={}",
                self.o_groups, self.num_attention_heads
            ));
        }
        if self.o_lora_rank == 0 {
            return bad("deepseek_v41 o_lora_rank must be positive".into());
        }
        if self.attention_bias {
            return bad("deepseek_v41 attention_bias=true is not supported".into());
        }
        if self.sliding_window == 0 {
            return bad("deepseek_v41 sliding_window must be positive".into());
        }
        if self.compress_rope_theta <= 0.0 || self.rope_theta <= 0.0 {
            return bad("deepseek_v41 RoPE bases must be positive".into());
        }

        // --- CSA2 ---
        let expected = self.num_hidden_layers as usize + self.dspark_blocks();
        if self.compress_ratios.len() != expected {
            return bad(format!(
                "deepseek_v41 compress_ratios has {} entries; expected {expected} = \
                 num_hidden_layers ({}) + dspark_target_layer_ids ({} MTP blocks). \
                 num_nextn_predict_layers ({}) counts speculative iterations and does not \
                 reconcile the length",
                self.compress_ratios.len(),
                self.num_hidden_layers,
                self.dspark_blocks(),
                self.num_nextn_predict_layers
            ));
        }
        // Unlike V4, ratio 1 is legal and means "per-token projection".
        check_layer_set(
            "kv_source_layer_ids",
            &self.kv_source_layer_ids,
            self.num_hidden_layers,
        )?;
        check_layer_set(
            "index_source_layer_ids",
            &self.index_source_layer_ids,
            self.num_hidden_layers,
        )?;
        if self.kv_source_layer_ids.is_empty() {
            return bad(
                "deepseek_v41 kv_source_layer_ids is empty; no layer would write the shared \
                 compressed KV cache"
                    .into(),
            );
        }
        // A compressor with ratio 0 has nothing to write.
        for layer in &self.kv_source_layer_ids {
            if self.compress_ratios[*layer as usize] == 0 {
                return bad(format!(
                    "deepseek_v41 kv_source layer {layer} has compress_ratio 0; a layer that \
                     writes the compressed cache must have a positive ratio"
                ));
            }
        }
        // Index keys are derived from the compressor latent, so every KV source
        // must also carry an indexer.
        for layer in &self.kv_source_layer_ids {
            if !self.index_source_layer_ids.contains(layer) {
                return bad(format!(
                    "deepseek_v41 kv_source layer {layer} is not in index_source_layer_ids; \
                     index keys are derived from the compressor latent, so every KV source \
                     owns an indexer"
                ));
            }
        }
        if !self
            .index_source_layer_ids
            .contains(&self.candidate_source_layer_id)
        {
            return bad(format!(
                "deepseek_v41 candidate_source_layer_id={} is not an index source; the coarse \
                 candidate blocks are selected by an indexer",
                self.candidate_source_layer_id
            ));
        }
        if self.candidate_topk_blocks == 0 || self.candidate_block_size == 0 {
            return bad(
                "deepseek_v41 candidate_topk_blocks and candidate_block_size must be positive"
                    .into(),
            );
        }

        // --- indexer ---
        if self.index_head_dim == 0 || self.index_n_heads == 0 || self.index_topk == 0 {
            return bad("deepseek_v41 indexer dimensions must be positive".into());
        }
        if (self.index_topk as i64) > self.max_position_embeddings {
            return bad(format!(
                "deepseek_v41 index_topk={} exceeds max_position_embeddings={}",
                self.index_topk, self.max_position_embeddings
            ));
        }

        // --- RoPE ---
        if self.rope_scaling.kind != "yarn" {
            return bad(format!(
                "deepseek_v41 unsupported RoPE scaling {:?}",
                self.rope_scaling.kind
            ));
        }
        if self.rope_scaling.factor <= 0.0
            || self.rope_scaling.original_max_position_embeddings <= 0
            || self.rope_scaling.beta_fast <= self.rope_scaling.beta_slow
        {
            return bad("deepseek_v41 invalid YaRN parameters".into());
        }
        let scaled = (self.rope_scaling.original_max_position_embeddings as f64
            * self.rope_scaling.factor as f64) as i64;
        if scaled != self.max_position_embeddings {
            return bad(format!(
                "deepseek_v41 YaRN factor {} over original_max_position_embeddings {} gives \
                 {scaled}, but max_position_embeddings is {}",
                self.rope_scaling.factor,
                self.rope_scaling.original_max_position_embeddings,
                self.max_position_embeddings
            ));
        }

        // --- MoE ---
        if self.n_routed_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.n_routed_experts
        {
            return bad(format!(
                "deepseek_v41 invalid expert routing: {} of {} routed experts per token",
                self.num_experts_per_tok, self.n_routed_experts
            ));
        }
        if self.n_shared_experts != 1 {
            return bad(format!(
                "deepseek_v41 expects exactly one shared expert; got {}",
                self.n_shared_experts
            ));
        }
        if self.dspark_n_routed_experts == 0
            || self.dspark_num_experts_per_tok == 0
            || self.dspark_num_experts_per_tok > self.dspark_n_routed_experts
        {
            return bad(format!(
                "deepseek_v41 invalid DSpark expert routing: {} of {}",
                self.dspark_num_experts_per_tok, self.dspark_n_routed_experts
            ));
        }
        if self.hidden_act != "silu"
            || self.scoring_func != "sqrtsoftplus"
            || self.topk_method != "noaux_tc"
        {
            return bad(format!(
                "deepseek_v41 requires silu experts with sqrtsoftplus/noaux_tc routing; got \
                 hidden_act={:?}, scoring_func={:?}, topk_method={:?}",
                self.hidden_act, self.scoring_func, self.topk_method
            ));
        }
        if !self.norm_topk_prob || self.routed_scaling_factor <= 0.0 || self.swiglu_limit <= 0.0 {
            return bad(
                "deepseek_v41 requires normalized, positively scaled top-k weights and a \
                 positive swiglu_limit"
                    .into(),
            );
        }

        // --- Engram ---
        check_layer_set(
            "engram_layer_ids",
            &self.engram_layer_ids,
            self.num_hidden_layers,
        )?;
        if self.engram_layer_ids.len() != self.engram_num_embeddings.len() {
            return bad(format!(
                "deepseek_v41 engram_num_embeddings has {} entries for {} engram layers; the \
                 table size is per-layer",
                self.engram_num_embeddings.len(),
                self.engram_layer_ids.len()
            ));
        }
        if self.engram_num_embeddings.iter().any(|n| *n <= 0) {
            return bad("deepseek_v41 engram_num_embeddings must be positive".into());
        }
        if self.engram_max_ngram_size == 0
            || self.engram_vocab_size <= 0
            || self.engram_n_heads == 0
            || self.engram_head_dim <= 0
            || self.engram_compressed_vocab_size <= 0
        {
            return bad("deepseek_v41 Engram dimensions must be positive".into());
        }
        if self.engram_pad_token_id < 0 || self.engram_pad_token_id >= self.vocab_size {
            return bad(format!(
                "deepseek_v41 engram_pad_token_id={} is outside vocab_size={}",
                self.engram_pad_token_id, self.vocab_size
            ));
        }

        // --- storage encodings ---
        let q = &self.quantization_config;
        if q.expert_dtype != "fp4" {
            return bad(format!(
                "deepseek_v41 expert_dtype {:?} is not the released MXFP4 encoding",
                q.expert_dtype
            ));
        }
        for (what, k) in [
            ("hidden_size", self.hidden_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
        ] {
            if k % V41_MXFP4_GROUP != 0 {
                return bad(format!(
                    "deepseek_v41 MXFP4 routed experts need {what}={k} to be a multiple of \
                     the {V41_MXFP4_GROUP}-element microscaling group"
                ));
            }
        }
        if q.quant_method != "fp8"
            || q.activation_scheme != "dynamic"
            || q.scale_fmt != "ue8m0"
            || q.weight_block_size != [32, 32]
        {
            return bad(format!(
                "deepseek_v41 requires dynamic block-FP8 [32,32] with ue8m0 scales; got \
                 method={:?}, activation={:?}, scale_fmt={:?}, block={:?}",
                q.quant_method, q.activation_scheme, q.scale_fmt, q.weight_block_size
            ));
        }
        if self.torch_dtype.as_deref() != Some("bfloat16") {
            return bad("deepseek_v41 requires dtype=bfloat16".into());
        }
        if self.tie_word_embeddings {
            return bad(
                "deepseek_v41 ships an untied head.weight; tie_word_embeddings=true would \
                 discard it"
                    .into(),
            );
        }

        // --- mHC ---
        if self.hc_mult == 0 || self.hc_sinkhorn_iters == 0 || self.hc_eps <= 0.0 {
            return bad(format!(
                "deepseek_v41 mHC needs hc_mult>0, hc_sinkhorn_iters>0 and hc_eps>0; got {}, \
                 {}, {}",
                self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps
            ));
        }

        // --- DSpark ---
        if self.dspark_blocks() == 0 {
            return bad(
                "deepseek_v41 dspark_target_layer_ids is empty; the checkpoint's mtp.* blocks \
                 have no source layers"
                    .into(),
            );
        }
        check_layer_set(
            "dspark_target_layer_ids",
            &self.dspark_target_layer_ids,
            self.num_hidden_layers,
        )?;
        if self.num_nextn_predict_layers == 0 {
            return bad("deepseek_v41 num_nextn_predict_layers must be positive".into());
        }
        if self.dspark_block_size == 0 || self.dspark_markov_rank <= 0 {
            return bad(
                "deepseek_v41 dspark_block_size and dspark_markov_rank must be positive".into(),
            );
        }
        if self.dspark_noise_token_id < 0 || self.dspark_noise_token_id >= self.vocab_size {
            return bad(format!(
                "deepseek_v41 dspark_noise_token_id={} is outside vocab_size={}",
                self.dspark_noise_token_id, self.vocab_size
            ));
        }
        // The Engram hash ranges must tile the table exactly. Each (n-gram size, head) pair owns
        // a prime-sized bucket range and the ranges are laid end to end, so a layer's primes SUM
        // to its `engram_num_embeddings` -- which makes the checkpoint's own row counts a
        // checksum on the prime walk in `engram_hash_tables`. A walk that drifts (a missed
        // `seen`, a wrong restart point) lands here rather than in a silently wrong lookup.
        if let Some(t) = self.engram_hash_tables() {
            for (at, (primes, rows)) in
                t.primes.iter().zip(self.engram_num_embeddings.iter()).enumerate()
            {
                let span: i64 = primes.iter().sum();
                if span != *rows {
                    return bad(format!(
                        "deepseek_v41 engram layer {} (id {}): the hash bucket ranges span {span} \
                         rows but the table has {rows}; the prime walk does not reproduce this \
                         checkpoint's layout",
                        at,
                        self.engram_layer_ids.get(at).copied().unwrap_or(u32::MAX),
                    ));
                }
            }
        }

        Ok(())
    }
}

/// A list of layer indices must be distinct and in range. Every such list in
/// this config selects which layers carry a tensor group, so a duplicate or an
/// out-of-range entry misbinds the checkpoint rather than failing loudly.
fn check_layer_set(what: &str, ids: &[u32], num_layers: u32) -> Result<(), ConfigError> {
    let mut seen = ids.to_vec();
    seen.sort_unstable();
    seen.dedup();
    if seen.len() != ids.len() {
        return Err(ConfigError::Unsupported(format!(
            "deepseek_v41 {what} must be distinct"
        )));
    }
    if let Some(id) = seen.last() {
        if *id >= num_layers {
            return Err(ConfigError::Unsupported(format!(
                "deepseek_v41 {what} names layer {id} but num_hidden_layers is {num_layers}"
            )));
        }
    }
    Ok(())
}

/// See [`DeepSeekV41Config::engram_hash_tables`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngramHashTables {
    /// `[engram layer][lookback]`, always odd and bounded so `token_id * multiplier` cannot
    /// overflow `i64`.
    pub multipliers: Vec<Vec<i64>>,
    /// `[engram layer][flat (n-gram size, head) column]` bucket modulus.
    pub primes: Vec<Vec<i64>>,
    /// `[engram layer][flat column]` first row of that column's bucket range.
    pub offsets: Vec<Vec<i64>>,
}

/// Trial division, which is plenty: the primes wanted sit just above `engram_vocab_size`
/// (~1.6e7), so the loop runs to ~4000.
fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3i64;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::config::ModelConfig;

    /// deepseek-ai/DeepSeek-V4.1-Flash's `text_config`, VERBATIM, with the
    /// outer `dtype` and `quantization_config` folded in -- exactly the
    /// document the text-generation frontend hands to the parser.
    const RELEASED: &str = r#"{
  "model_type": "deepseek_v41_text",
  "vocab_size": 129280,
  "hidden_size": 5120,
  "moe_intermediate_size": 2304,
  "num_hidden_layers": 40,
  "num_attention_heads": 64,
  "num_key_value_heads": 1,
  "head_dim": 512,
  "qk_rope_head_dim": 64,
  "q_lora_rank": 1280,
  "o_lora_rank": 1024,
  "o_groups": 8,
  "hidden_act": "silu",
  "swiglu_limit": 10.0,
  "rms_norm_eps": 1e-20,
  "attention_bias": false,
  "attention_dropout": 0.0,
  "initializer_range": 0.02,
  "use_cache": true,
  "tie_word_embeddings": false,
  "max_position_embeddings": 1048576,
  "rope_theta": 10000,
  "rope_scaling": {
    "rope_type": "yarn",
    "factor": 16,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 65536
  },
  "n_routed_experts": 384,
  "n_shared_experts": 1,
  "num_experts_per_tok": 6,
  "scoring_func": "sqrtsoftplus",
  "topk_method": "noaux_tc",
  "norm_topk_prob": true,
  "routed_scaling_factor": 1.5,
  "sliding_window": 128,
  "compress_ratios": [
    0,
    0,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0
  ],
  "compress_rope_theta": 160000,
  "kv_source_layer_ids": [
    2,
    8,
    14,
    20
  ],
  "index_source_layer_ids": [
    2,
    8,
    14,
    20,
    24,
    28,
    32,
    36
  ],
  "index_n_heads": 32,
  "index_head_dim": 128,
  "index_topk": 512,
  "candidate_source_layer_id": 20,
  "candidate_topk_blocks": 2048,
  "candidate_block_size": 8,
  "hc_mult": 4,
  "hc_sinkhorn_iters": 20,
  "hc_eps": 1e-06,
  "engram_layer_ids": [
    1,
    14
  ],
  "engram_num_embeddings": [
    384006168,
    384016682
  ],
  "engram_max_ngram_size": 4,
  "engram_vocab_size": 16000000,
  "engram_n_heads": 8,
  "engram_head_dim": 256,
  "engram_pad_token_id": 2,
  "engram_compressed_vocab_size": 99092,
  "num_nextn_predict_layers": 3,
  "dspark_block_size": 5,
  "dspark_noise_token_id": 128799,
  "dspark_target_layer_ids": [
    37,
    38,
    39
  ],
  "dspark_markov_rank": 256,
  "dspark_n_routed_experts": 128,
  "dspark_num_experts_per_tok": 3,
  "dtype": "bfloat16",
  "quantization_config": {
    "quant_method": "fp8",
    "activation_scheme": "dynamic",
    "weight_block_size": [
      32,
      32
    ],
    "scale_fmt": "ue8m0",
    "expert_dtype": "fp4"
  }
}"#;

    fn released() -> DeepSeekV41Config {
        match ModelConfig::from_json(RELEASED).expect("released config parses") {
            ModelConfig::DeepSeekV41(c) => c,
            other => panic!("released V4.1 config parsed as {other:?}"),
        }
    }

    /// `DeepseekV41ForCausalLM` must NOT fall through the `DeepseekV4` prefix.
    /// Before the V4.1 arm existed it did, which handed a V4.1 checkpoint to
    /// the V4 validator -- where it is rejected only by accident (the [32,32]
    /// scale grid), and where a future V4 relaxation would let it build.
    #[test]
    fn architectures_prefix_does_not_alias_v4() {
        let doc = r#"{"architectures":["DeepseekV41ForCausalLM"]}"#;
        let err = ModelConfig::from_json(doc).unwrap_err().to_string();
        assert!(!err.contains("deepseek_v4 "), "resolved to V4: {err}");
    }

    #[test]
    fn released_config_validates() {
        let cfg = released();
        assert_eq!(cfg.num_hidden_layers, 40);
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.head_dim, 512);
        assert_eq!(cfg.qk_nope_head_dim(), 448);
        // One KV head at a 512-wide latent: replicated, not sharded, under TP.
        assert_eq!(cfg.kv_latent_dim(), 512);
        assert_eq!(cfg.heads_per_o_group(), 8);
        assert_eq!(cfg.o_group_in_features(), 4096);
        // 40 layers + 3 DSpark blocks.
        assert_eq!(cfg.compress_ratios.len(), 43);
        assert_eq!(cfg.dspark_blocks(), 3);
    }

    /// The V4.1 shape the V4 validator gets wrong: ratio 1 is a legal mode, and
    /// only a ratio > 1 compressor carries a softmax gate. The checkpoint ships
    /// 4 `compressor.wkv` but only 3 `compressor.wgate`.
    #[test]
    fn ratio_one_is_a_mode_not_an_error() {
        let cfg = released();
        assert_eq!(cfg.attn_kind(0), V41Attn::Window);
        assert_eq!(cfg.attn_kind(2), V41Attn::Compressed { ratio: 2 });
        assert_eq!(cfg.attn_kind(20), V41Attn::Compressed { ratio: 1 });
        assert_eq!(cfg.attn_kind(39), V41Attn::Compressed { ratio: 1 });

        let gated = cfg
            .kv_source_layer_ids
            .iter()
            .filter(|l| cfg.compressor_has_gate(**l))
            .count();
        assert_eq!(cfg.kv_source_layer_ids.len(), 4, "4 compressor.wkv");
        assert_eq!(gated, 3, "3 compressor.wgate");
        assert!(
            !cfg.compressor_has_gate(20),
            "the ratio-1 compressor has no gate"
        );
    }

    /// Only 4 of the 40 layers compress; the other 36 read that cache. A
    /// per-layer-compressor reading of `compress_ratios` (V4's) would expect 38
    /// compressors and misbind the checkpoint.
    #[test]
    fn only_kv_sources_compress() {
        let cfg = released();
        assert_eq!(cfg.kv_source_layer_ids, vec![2, 8, 14, 20]);
        let reads_compressed = (0..cfg.num_hidden_layers)
            .filter(|l| cfg.attn_kind(*l) != V41Attn::Window)
            .count();
        assert_eq!(reads_compressed, 38);
        let writes = (0..cfg.num_hidden_layers)
            .filter(|l| cfg.is_kv_source(*l))
            .count();
        assert_eq!(writes, 4);
    }

    /// 8 indexers, 4 of which own index keys -- the checkpoint's 8
    /// `indexer.wq_b` against 4 `indexer.wk`.
    #[test]
    fn index_sources_are_a_superset_of_kv_sources() {
        let cfg = released();
        assert_eq!(cfg.index_source_layer_ids.len(), 8);
        for l in &cfg.kv_source_layer_ids {
            assert!(cfg.is_index_source(*l), "kv source {l} has no indexer");
        }
        let key_owners = (0..cfg.num_hidden_layers)
            .filter(|l| cfg.is_index_source(*l) && cfg.is_kv_source(*l))
            .count();
        assert_eq!(key_owners, 4);
        assert_eq!(cfg.candidate_source_layer_id, 20);
    }

    #[test]
    fn engram_tables_are_per_layer() {
        let cfg = released();
        assert_eq!(cfg.engram_layer_ids, vec![1, 14]);
        assert_eq!(cfg.engram_rows(1), Some(384_006_168));
        assert_eq!(cfg.engram_rows(14), Some(384_016_682));
        assert_eq!(cfg.engram_rows(0), None);
        assert_eq!(cfg.engram_head_dim, 256);
    }

    /// A [32,32] grid is 16x as many scales as V4's [128,128]; binding one
    /// shape for the other reads the wrong number of bytes.
    #[test]
    fn fp8_scale_grid_is_32x32() {
        let cfg = released();
        assert_eq!(cfg.quantization_config.weight_block_size, [32, 32]);
        assert_eq!(cfg.fp8_scale_shape(1280, 5120), [40, 160]);
        // MXFP4 keeps one E8M0 byte per 32 values along K, independent of the
        // block-FP8 grid.
        assert_eq!(cfg.mxfp4_scale_shape(2304, 5120), [2304, 160]);
    }

    /// The emit refuses, and the refusal names the pieces rather than being a
    /// bare "unsupported".
    #[test]
    fn unimplemented_names_the_gaps() {
        let text = released().unimplemented();
        for needle in [
            "CSA2 cache SHARING",
            "ratio-1 compressor",
            "TWO-LEVEL indexer",
            "Engram",
            "single-pass mHC",
            "SWA Bounded Replay",
            "DSpark",
            "graph builder",
        ] {
            assert!(
                text.contains(needle),
                "unimplemented() omits {needle:?}: {text}"
            );
        }
    }

    // ---- negative controls: each mutation must be REJECTED ----

    fn mutated(f: impl FnOnce(&mut serde_json::Value)) -> String {
        let mut v: serde_json::Value = serde_json::from_str(RELEASED).unwrap();
        f(&mut v);
        match ModelConfig::from_json(&v.to_string()) {
            Ok(_) => panic!("mutation was accepted"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn rejects_kv_source_without_a_ratio() {
        // Layer 2 compresses; zero its ratio and it has nothing to write.
        let err = mutated(|v| v["compress_ratios"][2] = 0.into());
        assert!(err.contains("compress_ratio 0"), "{err}");
    }

    #[test]
    fn rejects_kv_source_missing_an_indexer() {
        let err = mutated(|v| v["index_source_layer_ids"] = serde_json::json!([8, 14, 20]));
        assert!(err.contains("not in index_source_layer_ids"), "{err}");
    }

    #[test]
    fn rejects_compress_ratios_length_mismatch() {
        let err = mutated(|v| {
            let arr = v["compress_ratios"].as_array().unwrap().clone();
            v["compress_ratios"] = serde_json::Value::Array(arr[..40].to_vec());
        });
        assert!(err.contains("compress_ratios has 40 entries"), "{err}");
    }

    #[test]
    fn rejects_v4_scale_grid() {
        let err =
            mutated(|v| v["quantization_config"]["weight_block_size"] = serde_json::json!([128, 128]));
        assert!(err.contains("block-FP8 [32,32]"), "{err}");
    }

    #[test]
    fn rejects_engram_table_count_mismatch() {
        let err = mutated(|v| v["engram_num_embeddings"] = serde_json::json!([384006168]));
        assert!(err.contains("engram_num_embeddings has 1 entries"), "{err}");
    }

    #[test]
    fn rejects_candidate_source_that_is_not_an_indexer() {
        let err = mutated(|v| v["candidate_source_layer_id"] = 21.into());
        assert!(err.contains("is not an index source"), "{err}");
    }

    #[test]
    fn rejects_duplicate_layer_ids() {
        let err = mutated(|v| v["engram_layer_ids"] = serde_json::json!([1, 1]));
        assert!(err.contains("must be distinct"), "{err}");
    }
}
