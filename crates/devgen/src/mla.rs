//! MLA + MoE emit family: GLM-5.2 / Kimi K2.7 / DeepSeek-V3 (shared MLA+MoE emit)
//! and Nemotron-3 (Mamba-2 hybrid). Split out of `lib.rs` (module breakdown). All
//! are `--block` extraction on this path; `run_verified` dispatches on model_type.
use std::path::Path;

use packet::dev::{DevOp, TENSOR_NONE, TENSOR_NONE_I, WG_THREADS, WG_WAVES};
use packet::devbuild::{Builder, Model};
use packet::rope::{GenTensor, RopeScale};
use serde_json::Value;

use super::*;
use crate::block::{parse_block, write_block_descriptor};

/// The design notes. `H`/`NH`/`DK`(kv_lora)/`QL`(q_lora)/`QN`(qk_nope)/`DR`(qk_rope)/
/// `VD`(v_head) name the MLA geometry the kernels carry as compile-time operands.
#[derive(Clone)]
pub(crate) struct GlmCfg {
    layers: u32,  // 78 (layer 78 = MTP head, skipped)
    hidden: u32,  // H 6144
    heads: u32,   // NH 64
    kv_lora: u32, // DK 512 (latent cache width)
    q_lora: u32,  // QL 2048
    qk_nope: u32, // QN 192 (absorbed into the latent)
    qk_rope: u32, // DR 64  (partial rope, interleaved)
    v_head: u32,  // VD 256
    vocab: u32,   // 154880
    eps: f32,     // 1e-5
    n_exp: u32,   // E 256 routed experts
    top_k: u32,   // 8
    // GROUP-LIMITED ROUTING (DeepSeek noaux_tc). Experts are partitioned into `n_group` contiguous
    // groups, each scored by the SUM OF ITS TOP-2 biased scores; the top `topk_group` groups are
    // kept and the top-k runs only inside them. `n_group <= 1` makes the rule the identity, which is
    // why GLM-5.2 (n_group=1) matched its HF oracle while the flat top-k was the only thing
    // implemented — and why Kimi/DeepSeek, which DO group, were selecting a different expert set.
    n_group: u32,
    topk_group: u32,
    moe_inter: u32,     // IMOE 2048 (per-expert intermediate)
    dense_inter: u32,   // 12288 (layers < first_k_dense)
    first_k_dense: u32, // 3 (layers 0,1,2 dense FFN; 3-77 MoE)
    route_scale: f32,   // 2.5 (routed_scaling_factor)
    attn_scale: f32,    // 1/sqrt(qk_head_dim = qk_nope+qk_rope = 256) = 0.0625
    /// Interleaved partial-RoPE theta for the `qk_rope` decoupled dims, or `None` for a **NoPE**
    /// MLA (`mla_use_nope`), where those dims are carried as extra CONTENT dims and never rotated.
    ///
    /// `Option`, and not defaulted, on purpose. This field read `v["rope_theta"].as_f64()
    /// .unwrap_or(8_000_000.0)`, which is wrong in both directions:
    ///
    /// * a NoPE checkpoint (Kimi-K3 sets `mla_use_nope: true` and ships no `rope_theta` at all)
    ///   picked up GLM's 8e6 and had a rotation applied that its own modeling code cannot express
    ///   — `self.rotary_emb = None`, `assert self.use_nope`. Plausible logits, wrong model;
    /// * and GLM-5.2 itself has NO top-level `rope_theta` either — transformers 5.x moved it to
    ///   `rope_parameters.rope_theta` — so the shipping model was riding the literal `8_000_000.0`
    ///   in this file. It happens to be the right number, which is exactly why nobody noticed; a
    ///   GLM variant retrained at a different theta would have been silently wrong.
    ///
    /// `cfg_glm` now REFUSES rather than defaulting, and [`GlmCfg::rope_theta`] is the only reader.
    rope_theta: Option<f64>,
    rope_scale: RopeScale,
    /// The namespace this checkpoint's weights live under, `.`-terminated — `model.` for GLM /
    /// DeepSeek / Kimi-K2.7, `language_model.model.` for a multimodal wrapper like Kimi-K3.
    ///
    /// A cfg PROPERTY, mirroring `Cfg::prefix` on the Gemma path, because it belongs to the
    /// checkpoint and not to this file. `declare_glm` used to spell `format!("model.layers.…")`
    /// at eight sites; a checkpoint under a wrapper prefix would have had every one of those
    /// names wrong at once, and the loader's answer to a name it cannot find used to be a zero
    /// fill rather than an error (see `packet::names`).
    prefix: String,
    tp: u32,
    // EP (expert-parallel) over the same `tp` world: attention/shared/dense stay TP-sharded (the
    // "floor" is parallelized), but the ROUTED experts are distributed WHOLE across ranks (256/tp
    // per rank, full moe_inter width — no CU-starve) instead of TP-sliced. Each rank fires only its
    // LOCAL chosen experts (host binds local expert bases, NULL for remote; the kernel skips a null
    // base). The combine XReduce (already summing shared partials over tp) folds the per-rank whole-
    // expert partials in the SAME collective — no new op. See the design notes.
    ep: bool,
    // Collapse the per-slot expert packets (2*top_k) into 2 grouped packets (ops 48/49) — the op-count
    // lever for M=1 decode. Bit-identical output; block-fp8 only.
    group: bool,
    // DSA lightning indexer (GlmMoeDsa). ctx>2048 => indexer->select->gather; ctx<=2048 => dense.
    index_heads: u32,        // index_n_heads = 32
    index_dim: u32,          // index_head_dim = 128 (rope on the first qk_rope=64; pass the rest)
    index_topk: u32,         // index_topk = 2048
    indexer_full: Vec<bool>, // per-layer: true='full' (owns an indexer), false='shared' (reuse last full)
    // Whether this arch HAS the DSA lightning indexer at all. GLM-5.2 (glm_moe_dsa) => true.
    // Kimi K2.7 / DeepSeek-V3 are plain MLA (NO indexer), so `has_dsa=false` holds the DSA gate off
    // at EVERY ctx — declare_glm allocates no indexer scratch and emit_glm_mla stays on FlashMlaDecode
    // (the dense MLA path), reusing the same emit as GLM below the crossover.
    has_dsa: bool,
}
impl GlmCfg {
    /// Full per-head qk width = nope + rope = 256. The attention softmax scale is
    /// 1/sqrt of THIS (0.0625) — NOT 1/sqrt(128); the absorbed MLA keeps the full-width scale.
    fn qk_head(&self) -> u32 {
        self.qk_nope + self.qk_rope
    }
    /// The RoPE theta, or a refusal. The MLA emit is CONDITIONAL on this being present, and the
    /// condition is enforced here rather than at the four `HeadNormRope` sites, because the k-side
    /// one is not a rotation that can simply be dropped — see the message.
    ///
    /// `require_mla_rope` fires first, at config-parse time, so in practice this is unreachable;
    /// it exists so a future cfg constructed in code cannot route a NoPE model into the rope emit.
    fn rope_theta(&self) -> f64 {
        self.rope_theta.unwrap_or_else(|| {
            panic!(
                "MLA NoPE reached the RoPE emit. This cfg carries no rope_theta \
                 (`mla_use_nope`), so there is no rotation to apply — but DROPPING the two \
                 HeadNormRope ops is NOT the emit: the k-side one at mla.rs `emit_glm_mla` is \
                 also the only writer of the `kv.{{l}}.krot` cache row (t[0] = n.krot[slot], \
                 t[1] = n.krr) — and the instruction the AMD loader's kv_row_writer scan and \
                 runtime/tests/glm52_decode.c both LOOK FOR to patch that row's position each \
                 step. Remove it and the rope half of every cached key stays uninitialised while \
                 FlashMlaDecode keeps reading it at i[5], and the layer silently drops out of the \
                 KV-row-writer list — garbage that grows with context and never faults. A NoPE \
                 MLA needs a raw row-write into krot (or an identity cos=1/sin=0 table, which \
                 makes HeadNormRope an exact copy); until one exists the emit refuses. See \
                 `require_mla_rope`."
            )
        })
    }
    /// Layers `[0, first_k_dense)` are dense-FFN (intermediate 12288); the rest are MoE.
    fn is_dense(&self, layer: u32) -> bool {
        layer < self.first_k_dense
    }
    /// DSA gate: sparse (indexer->select->gather) only above the dense-attention CROSSOVER — the ctx
    /// where the gather's FIXED per-full-layer overhead (indexer score + top-k select on 21 layers)
    /// plus the constant top_k=2048 gather flash first UNDERCUTS the ctx-linear dense flash. MEASURED
    /// on the real full 78-layer model (TP4, MI350X 4-7, median-11) AFTER the MFMA-indexer + 32-WG-select
    /// interp wiring: gather tpot is flat ~48.6ms; dense grows ~0.136ms/1k-ctx from 41.4ms@16k, so the
    /// two cross at ~69k (BEFORE the wiring: ~91k). Below the crossover the whole-model tpot is
    /// MoE/projection-floor-dominated (~40ms) and dense-flash is cheap, so gather LOSES (0.85-0.90x
    /// across 16-32k) — those ctx are gated to dense, the measured winner. `CROSSOVER=65536` keeps the
    /// 16k-32k band (and up to 64k) on dense and arms gather only where it wins. NOTE: this is the TP4
    /// crossover (the session's GPU budget is 4 cards); a TP8 deployment halves the parallel floor and
    /// per-rank attention shrinks, lowering the crossover — recalibrate with an 8-GPU sweep before
    /// serving TP8 (design-doc projection puts the TP8 band nearer the crossover).
    /// PLOW_GLM_DSA=0 forces the dense path even at long ctx (the apples-to-apples decode baseline).
    fn dsa(&self, ctx: u32) -> bool {
        const CROSSOVER: u32 = 65536; // measured full-model TP4 dense/gather crossover (~69k, rounded)
        let on = self.has_dsa
            && ctx > CROSSOVER
            && emit_config::active().glm_dsa.as_deref() != Some("0");
        if on {
            // THE ONE GATE every DSA arming passes through, so the geometry check lives here rather
            // than at each of the four sites that consume `index_heads`/`index_dim`.
            //
            // `cfg_glm` PARSES `index_n_heads`/`index_head_dim` out of config.json with no
            // validation, but the kernel does not read them: `interp.hip`'s INDEX_SCORE arm
            // hardcodes `constexpr int DI_ = 128, HI_ = 32`, and `d_index_score_mfma` carries
            // `static_assert(HIc == 32, "MFMA subtile assumes index_n_heads == 32")` — HI=32 is
            // baked into the 32x32 accumulator fragment layout, it is not a template knob that
            // happens to be instantiated at one value. So a checkpoint with a different indexer
            // geometry used to parse cleanly, size `qidx` as `hi*di`, and then be STRIDED by the
            // kernel as 32*128 — wrong scores, wrong top-2048, silently attending to wrong rows.
            // A repo fixture already carries `index_heads: 8` (`k3_ref_cfg`), so this is reachable,
            // not hypothetical.
            //
            // REFUSE rather than plumb: making HI a real template parameter means a second MFMA
            // fragment layout inside a megakernel measured at 254/256 VGPR. Nothing in the roadmap
            // needs it, and an emit-time panic is where this tree puts "loud" (see op_moe.h's
            // `moe_bound_topk`: the interpreter's dispatch `default:` is a deliberate silent NOP,
            // so the compile/emit refusal is the only loud failure available).
            assert_eq!(
                self.index_heads, 32,
                "DSA indexer: index_n_heads={} but d_index_score_mfma static_asserts HIc==32 and \
                 interp.hip hardcodes HI_=32. Refusing to emit a blob whose scores would be \
                 strided wrong. Plumb i[1] into a second MFMA layout or run with PLOW_GLM_DSA=0.",
                self.index_heads
            );
            assert_eq!(
                self.index_dim, 128,
                "DSA indexer: index_head_dim={} but interp.hip hardcodes DI_=128 (and the RoPE \
                 tables, qidx/kidx sizing and LDS tiles are all cut to it). Refusing to emit.",
                self.index_dim
            );
        }
        on
    }
    /// A 'full' indexer layer owns its own indexer; 'shared' layers reuse the last full layer's idx.
    fn indexer_is_full(&self, layer: u32) -> bool {
        self.indexer_full
            .get(layer as usize)
            .copied()
            .unwrap_or(false)
    }

    /// Is the DSA SPARSE PREFILL armed (`PLOW_GLM_DSA_PF=1`)? Independent of [`Self::dsa`],
    /// which gates the DECODE indexer at a 64k crossover.
    ///
    /// The prefill selection is per QUERY TOKEN, so its economics are the opposite of decode's:
    /// a dense prefill is O(T·S) while the gathered one is O(T·top_k) + the indexer's O(T·S)
    /// *small* GEMM (128-dim keys vs the 576-dim latent, and no PV), so it pays as soon as the
    /// causal mean row length exceeds `index_topk` — i.e. from T ≈ 2·2048 up. Below that the
    /// selection is the identity and dense must be kept (`glm_dsa_pf_bucket`).
    ///
    /// Requires the same 32×128 indexer geometry [`Self::dsa`] asserts, and the same 'full'
    /// layer weights — enforced where the chain is emitted.
    fn dsa_pf(&self) -> bool {
        self.has_dsa && emit_config::active().glm_dsa_pf
    }
}

/// (See GLM_DSA_PF_PACK below: unions are per 8-query PACK since B2; the cap stays the exact
/// bound so op 119's compaction can never silently drop a selected row.)
/// Union capacity per B2 query PACK (8 queries): the exact bound min(8*top_k, ctx), so the
/// compaction can never truncate. The pack width is 8 because the head-batched gather walk
/// fills the V2 kernel's 64 M-rows with (8 queries x 8 heads) — see d_flash_mla_prefill_v2.
pub(crate) const GLM_DSA_PF_PACK: u32 = 8;
fn glm_dsa_pf_cap(c: &GlmCfg, ctx: u32) -> u32 {
    (GLM_DSA_PF_PACK as u64 * c.index_topk as u64).min(ctx as u64) as u32
}

/// Does prefill bucket `t` at context `ctx` run the SPARSE (gathered) attention arm?
///
/// Two floors, both structural rather than tuned:
///   * `t >= 2048` — the V2 flash's BQ=64 fill floor, and the same bound the host's segment
///     router applies (`exec/amd.rs derive_segments`). A smaller bucket has no V2 arm to
///     gather with, and the 8-wave dense kernel ignores `t7`.
///   * `t > index_topk` — below it the causal rows are shorter than the selection, every
///     top-k is the identity, and the indexer is pure overhead (the measured decode fact).
fn glm_dsa_pf_bucket(c: &GlmCfg, t: u32) -> bool {
    c.dsa_pf() && t >= 2048 && t > c.index_topk
}

fn cfg_glm(dir: &Path) -> GlmCfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .unwrap();
    let g = |k: &str| {
        v[k].as_u64()
            .unwrap_or_else(|| panic!("config.json missing {k}")) as u32
    };
    let qk_head = g("qk_nope_head_dim") + g("qk_rope_head_dim");
    // GROUP-LIMITED ROUTING (DeepSeek `noaux_tc`). `MOE_ROUTER_TOPK`/`_PF` take these as i[6]/i[7]
    // and apply the rule as a mask on the packed selection key, so it stays one fused kernel. Absent
    // keys default to 1/1, which the kernel treats as the identity — every GLM/Qwen/Mixtral packet
    // stays bit-identical. `topk_group >= n_group` is also the identity (every group kept).
    let n_group = v["n_group"].as_u64().unwrap_or(1).max(1) as u32;
    let topk_group = v["topk_group"].as_u64().unwrap_or(n_group as u64) as u32;
    assert!(
        n_group <= 1 || g("n_routed_experts") % n_group == 0,
        "n_routed_experts={} is not divisible by n_group={n_group}; the kernel partitions experts \
         into CONTIGUOUS equal groups (gsz = n_exp / n_group) and a remainder would silently drop \
         the tail experts from every group score",
        g("n_routed_experts")
    );
    assert!(
        topk_group >= 1,
        "topk_group=0 would select no expert group at all (n_group={n_group})"
    );
    let model = v["model_type"].as_str().unwrap_or("<unknown model_type>");
    crate::require_moe_topk(g("num_experts_per_tok"), model);
    // RoPE theta: READ, never defaulted. Both spellings, because transformers 5.x moved the key —
    // GLM-5.2's config.json has NO top-level `rope_theta` and carries
    // `rope_parameters: {rope_theta: 8000000, rope_type: "default"}`. `mla_use_nope` (Kimi-K3)
    // means there is no theta to find and none may be invented. `require_mla_rope` decides.
    let rp = &v["rope_parameters"];
    let rope_theta = v["rope_theta"]
        .as_f64()
        .or_else(|| rp["rope_theta"].as_f64());
    crate::require_mla_rope(
        rope_theta,
        v["mla_use_nope"].as_bool().unwrap_or(false),
        rp["rope_type"].as_str(),
        v["rope_scaling"].as_object().is_some(),
        model,
    );
    GlmCfg {
        layers: g("num_hidden_layers"),
        hidden: g("hidden_size"),
        heads: g("num_attention_heads"),
        kv_lora: g("kv_lora_rank"),
        q_lora: g("q_lora_rank"),
        qk_nope: g("qk_nope_head_dim"),
        qk_rope: g("qk_rope_head_dim"),
        v_head: g("v_head_dim"),
        vocab: g("vocab_size"),
        eps: v["rms_norm_eps"].as_f64().unwrap() as f32,
        n_exp: g("n_routed_experts"),
        top_k: g("num_experts_per_tok"),
        n_group,
        topk_group,
        moe_inter: g("moe_intermediate_size"),
        dense_inter: g("intermediate_size"),
        first_k_dense: g("first_k_dense_replace"),
        route_scale: v["routed_scaling_factor"].as_f64().unwrap() as f32,
        attn_scale: (qk_head as f32).powf(-0.5),
        rope_theta,
        rope_scale: RopeScale::None,
        // Flat checkpoint: GLM / DeepSeek / Kimi-K2.7 all ship `model.layers.…` at the root.
        // A nested (multimodal) variant sets this from its own wrapper, and nothing else changes.
        prefix: "model.".to_string(),
        tp: 1,
        ep: emit_config::active().glm_ep,
        group: emit_config::active().glm_group,
        index_heads: v["index_n_heads"].as_u64().unwrap_or(32) as u32,
        index_dim: v["index_head_dim"].as_u64().unwrap_or(128) as u32,
        index_topk: v["index_topk"].as_u64().unwrap_or(2048) as u32,
        indexer_full: v["indexer_types"]
            .as_array()
            .map(|a| a.iter().map(|t| t.as_str() == Some("full")).collect())
            .unwrap_or_default(),
        has_dsa: true, // GLM-5.2 (glm_moe_dsa) has the DSA lightning indexer.
    }
}

/// Kimi K2.7 / DeepSeek-V2/V3 cfg (M3). These are plain
/// MLA + MoE — the SAME DeepSeek-derived config schema GLM uses (q/kv_lora, qk_nope/rope, v_head,
/// n_routed_experts, moe_intermediate_size, first_k_dense_replace, routed_scaling_factor) but with
/// NO DSA lightning indexer. So the cfg reuses `cfg_glm`'s parse verbatim and only forces the DSA
/// gate off (`has_dsa=false`): the indexer fields default (indexer_types absent => empty) and never
/// fire, so declare_glm / emit_glm_mla take the dense-MLA path at every ctx. This is the reuse
/// seam — Kimi is GLM-below-the-crossover with different dims. NOT `rewrite/kimi.rs` (that lowers to
/// the wire-packet backend `GpuEngine` cannot load; see plan §5.0).
fn cfg_kimi(dir: &Path) -> GlmCfg {
    let mut c = cfg_glm(dir);
    c.has_dsa = false;
    c
}

fn cfg_k3_dspark(dir: &Path) -> GlmCfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .unwrap();
    let g = |k: &str| {
        v[k].as_u64()
            .unwrap_or_else(|| panic!("config.json missing {k}")) as u32
    };
    assert_eq!(v["model_type"].as_str(), Some("k3_dspark"));
    assert_eq!(v["mla_use_nope"].as_bool(), Some(false));
    assert_eq!(v["mla_use_output_gate"].as_bool(), Some(false));
    let rp = &v["rope_parameters"];
    assert_eq!(rp["rope_type"].as_str(), Some("yarn"));
    let factor = rp["factor"].as_f64().expect("rope_parameters.factor");
    let orig = rp["original_max_position_embeddings"]
        .as_f64()
        .expect("rope_parameters.original_max_position_embeddings");
    let beta_fast = rp["beta_fast"].as_f64().expect("rope_parameters.beta_fast");
    let beta_slow = rp["beta_slow"].as_f64().expect("rope_parameters.beta_slow");
    let mscale_all_dim = rp["mscale_all_dim"].as_f64().unwrap_or(0.0);
    let mscale = if factor <= 1.0 {
        1.0
    } else {
        0.1 * mscale_all_dim * factor.ln() + 1.0
    };
    let qk_head = g("qk_nope_head_dim") + g("qk_rope_head_dim");
    let layers = g("num_hidden_layers");
    GlmCfg {
        layers,
        hidden: g("hidden_size"),
        heads: g("num_attention_heads"),
        kv_lora: g("kv_lora_rank"),
        q_lora: g("q_lora_rank"),
        qk_nope: g("qk_nope_head_dim"),
        qk_rope: g("qk_rope_head_dim"),
        v_head: g("v_head_dim"),
        vocab: g("vocab_size"),
        eps: v["rms_norm_eps"].as_f64().expect("rms_norm_eps") as f32,
        n_exp: 1,
        top_k: 1,
        n_group: 1,
        topk_group: 1,
        moe_inter: g("intermediate_size"),
        dense_inter: g("intermediate_size"),
        first_k_dense: layers,
        route_scale: 1.0,
        attn_scale: (qk_head as f32).powf(-0.5) * (mscale * mscale) as f32,
        rope_theta: Some(
            rp["rope_theta"]
                .as_f64()
                .expect("rope_parameters.rope_theta"),
        ),
        rope_scale: RopeScale::Yarn {
            factor,
            beta_fast,
            beta_slow,
            orig,
        },
        prefix: String::new(),
        tp: 1,
        ep: false,
        group: false,
        index_heads: 0,
        index_dim: 0,
        index_topk: 0,
        indexer_full: Vec::new(),
        has_dsa: false,
    }
}

/// MLA head-fusion factor `d_flash_mla_decode<512,64,GF>`, chosen PER PACKET from the pkt's fixed
/// max_ctx and baked into FlashMlaDecode i[7] (the interp instantiates GF∈{2,4,8} and dispatches on
/// i[7]; LDS/registers are sized for the GF=8 max, so occ is unchanged). GLM's MLA latent is
/// HEAD-SHARED, so GF query heads re-stream the compact latent once per head-group => latent HBM
/// traffic ~ n_head/GF. TRADEOFF (measured, MI350X full-model TP4 decode): GF=4 CUTS long ctx
/// (128k 125 vs 140 ms/tok; 8k-32k 1.3-1.6x on the MLA chain) but ADDS split/merge overhead that
/// HURTS short ctx (1k 79 vs 58 ms/tok — the tiny 1k latent stream isn't worth the extra splits).
/// So: GF=2 for short-ctx pkts (preserve the router-split ~58ms@1k), GF=4 for long-ctx pkts.
/// PLOW_GLM_GF pins GF∈{2,4,8} (crossover sweeps). Crossover ~4k (see perf-data/glm52-plow-decode-tuned.json).
/// Long-ctx / DEFAULT head-fusion factor: the GF a packet gets when `i[7]` is 0, and the GF
/// `glm_nsplit`'s chip-fill cap is written against. It is NOT the max the interpreter can run —
/// `exec_flash_mla_decode` instantiates 2, 4 and 8, and `GLM_MLA_GF_MAX = 8` in op_attention.h is
/// what the LDS union is sized for.
pub(crate) const GLM_MLA_GF: u32 = 4;
const GLM_GF_CROSSOVER: u32 = 4096; // max_ctx <= this -> GF=2; else GF=8

/// The head-fusion factor for a DECODE packet, clamped to a GF this rank's head shard can express.
///
/// ## `nh_l` is a HARD constraint, and it became one the moment the GF=8 arm landed
///
/// The kernel computes `n_grp = n_head / GF` with integer division and iterates
/// `n_batch*n_tok*n_grp*nsplit` work items, so `GF > nh_l` makes `n_grp == 0` and the flash does
/// **nothing at all** — no partials written, no error, garbage attention. That is reachable on
/// GLM-5.2 (n_head 64): tp8 gives nh_l=8 (GF=8 is exactly expressible), tp16 gives nh_l=4 (it is
/// not). Before the arm existed the bug was masked, because `i[7]=8` ran the GF=4 body and 4 <= 4.
/// **Implementing the arm converted a dead field into a live divide-to-zero, so the clamp is part
/// of the arm, not a nicety.** `glm_gf_prefill` has carried the same clamp for the same reason.
///
/// The clamp also applies to the `PLOW_GLM_GF` pin: a sweep that pins 8 on a tp16 blob would
/// otherwise emit silent all-zero attention rather than a slow arm.
///
/// ## `GF <= nh_l` IS NOT ENOUGH — it must DIVIDE. (Found on Kimi-K3, fixed for everyone.)
///
/// `n_grp = nh_l / GF` truncates, and the kernel's only head cursor is `h0 = hg * GF` for
/// `hg in [0, n_grp)`. So it visits exactly `n_grp * GF` heads and the remainder
/// `nh_l % GF` is **never visited at all**: no partials written for those heads, and
/// `FlashMerge`/`MlaMergeFold` then read uninitialised `opart`/`mlpart` for them. The old
/// predicate here was `g <= nh_l` — "leaves at least one head-group on this rank" — which is
/// exactly the condition that stops `n_grp` reaching zero and says nothing about the tail.
///
/// **This never fired on any shipping model, which is why it survived.** Every `nh_l` GLM-5.2,
/// Kimi-K2.7 and DeepSeek-V3 produce is a power of two (64/128 heads over a power-of-two tp), and
/// every power of two is divisible by 8, 4 and 2. The selection is therefore **byte-identical**
/// for all of them — this changes no emitted packet for any model in the tree today.
///
/// Kimi-K3 is the first model with a non-power-of-two head count: **96**. At the reference TP8
/// that is `nh_l = 12`, and above the crossover the old rule picked GF=8 because `8 <= 12` —
/// `n_grp = 1`, heads 8..11 dropped, **4 of every rank's 12 heads silently missing**. TP16
/// (`nh_l = 6`) picked 4 and dropped 2 of 6. Both produce fluent output from two thirds of the
/// attention, which is the failure mode this tree has shipped fourteen times.
///
/// So the predicate is divisibility, and `require_gf_divides` is the backstop for a shard no
/// instantiated GF can express at all (any odd `nh_l`, e.g. K3 at tp32 -> 3).
fn glm_gf(ctx: u32, nh_l: u32) -> u32 {
    // ORIGIN OF THE 1.5-1.9x CLAIM, AND ITS STATUS. The number comes from
    // the design notes, which is an **sm120 (NVIDIA)** measurement, and it travelled
    // into this comment as if it were a property of the op. On AMD it was never measurable at all
    // until now: the interpreter had no GF=8 arm, so every ctx>4096 GLM blob emitted i[7]=8 and
    // executed GF=4. Treat 1.5-1.9x as an NVIDIA result and an UNPROVEN hypothesis on gfx950.
    //
    // ## THE DEFAULT IS 4, NOT 8, AND THAT IS A MEASURED FIX. Read before changing it back.
    //
    // `PLOW_GLM_GF8_ARM` (op_attention.h) DEFAULTS TO 0 since 3344543 — the GF=8 instantiation is
    // a +32% decode regression merely by being compiled in (persistent megakernel, shared
    // instruction stream). With it compiled out `i[7]=8` runs the GF=4 body. Returning 8 anyway
    // does NOT then reproduce the pre-9dc27bb state, because 9dc27bb ALSO made `flash_mla_cus`
    // read `i[7]` literally instead of mirroring the dispatch:
    //
    //     emitted i[7]   kernel body   n_work = (nh_l/GF_body)*ns   workgroups dispatched
    //     8              GF=4          16/4 * 64 = 256              (16/8)*64 = 128   <- HALF
    //     4              GF=4          256                          256
    //
    // 256 work items on 128 workgroups is CORRECT (the body grid-strides) and runs at half the
    // parallelism. It is the original §6g-GF8 bug with the arrow reversed a second time: there the
    // SELECTOR existed and the ARM did not; here the arm is gone and the selector still narrows.
    //
    // MEASURED, GLM-5.2 TP4, default (arm-absent) object, `glm52_decode` trace, per-layer chain:
    //     live ctx   i[7]=8 (128 wg)   i[7]=4 (256 wg)
    //      8192        97.6 us           83.7 us     -14.2%
    //     32768       168.1 us          135.9 us     -19.2%
    // End-to-end, full 78 layers, plowrt + `vllm bench serve`, median ITL (control run twice,
    // drift 0.02 / 0.05 ms):
    //      8192       28.58 / 28.60 ms   27.45 ms    -1.14 ms  (-4.0%)
    //     32768       34.81 / 34.86 ms   31.49 ms    -3.35 ms  (-9.6%)
    // Token-identical at every arm (Plow.SplitK; the 24-token greedy generate matches exactly).
    //
    // So: 4 is what the DEFAULT OBJECT can run at full width. `PLOW_GLM_GF=8` still pins 8, and it
    // is only meaningful against an object built with `-DPLOW_GLM_GF8_ARM=1`; measured there, GF=8
    // at matched work items is NOT a win (see `glm_nsplit`'s table).
    // Read the pin from env directly (not from OnceLock) so that A/B measurement scripts
    // can override mid-process. The EmitConfig field is the production path via CLI; this
    // env read is the low-ceremony measurement path.
    let pin = std::env::var("PLOW_GLM_GF")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .or(emit_config::active().glm_gf);
    let want = pin
        .filter(|&v| v == 2 || v == 4 || v == 8)
        .unwrap_or(if ctx <= GLM_GF_CROSSOVER { 2 } else { 4 });
    // SAY SO WHEN GF=8 IS PINNED BY HAND. The default object is built with PLOW_GLM_GF8_ARM=0
    // (op_attention.h:1309), where the GF=8 `else if` is preprocessed away and an emitted gf=8
    // falls through to the GF=4 body — which is CORRECT, because that body grid-strides. What
    // does not fall through is `flash_mla_cus`: it sizes the CU list from the literal i[7], so
    // the packet dispatches (nh_l/8)*ns workgroups for (nh_l/4)*ns work items and runs at HALF
    // the parallelism. Measured -3.35 ms/token at ctx 32768; nothing faults and nothing warns.
    //
    // The emitter cannot see how the object was built, so this is a warning and not a refusal —
    // but the trap is one env var away from the defect three commits were just spent fixing, and
    // an unannounced 10% is worse than a noisy line here.
    if want == 8 {
        eprintln!(
            "  PLOW_GLM_GF=8: only meaningful against an object built -DPLOW_GLM_GF8_ARM=1. \
             The DEFAULT object compiles that arm out, and gf=8 there dispatches half the \
             workgroups for the same work (measured -3.35 ms/token at ctx 32768). Unset it \
             unless you built the arm."
        );
    }
    // Largest instantiated GF that EXACTLY PARTITIONS this rank's head shard. `g <= nh_l` is the
    // divide-to-zero guard and is implied by `nh_l % g == 0` for g >= 1; what it is NOT is a
    // guarantee that every head is visited (see the header).
    let gf = [8u32, 4, 2]
        .into_iter()
        .find(|&g| g <= want && nh_l % g == 0)
        .unwrap_or(2);
    require_gf_divides(gf, nh_l, "decode");
    gf
}

/// Refuse a head-fusion factor that does not partition this rank's head shard.
///
/// The backstop, not the mechanism: `glm_gf`/`glm_gf_prefill` already prefer a GF that divides, so
/// this only fires when NO instantiated GF does — i.e. an odd `nh_l`, which no `heads / tp` in the
/// tree produces today and which Kimi-K3 reaches at tp32 (96/32 = 3). Refusing at emit is the whole
/// point: the runtime cannot detect it, because unvisited heads are not an error condition
/// anywhere — they are simply memory nobody wrote, read back by the merge as if it were a partial.
fn require_gf_divides(gf: u32, nh_l: u32, phase: &str) {
    assert!(
        gf >= 1 && nh_l % gf == 0,
        "MLA {phase}: head-fusion factor GF={gf} does not divide this rank's head shard \
         nh_l={nh_l} (heads/tp). The flash kernel walks `n_grp = nh_l / GF` head-groups at \
         `h0 = hg * GF`, so it would visit only {} of {nh_l} heads and leave the remaining {} \
         with NO partial written — FlashMerge/MlaMergeFold then read uninitialised opart/mlpart \
         for them and the model is fluent with part of its attention missing. The interpreter \
         instantiates GF in {{2,4,8}} only, so an odd nh_l cannot be expressed at all: choose a \
         tensor-parallel width whose head shard is even (for 96 heads: tp 1/2/4/8/16, not 32).",
        (nh_l / gf) * gf,
        nh_l - (nh_l / gf) * gf,
    );
}
/// MLA flash-decode KV-split count, CTX-ADAPTIVE (mirrors Gemma's PLOW_NS_MUL/ABS). The flash
/// splits its work into `n_grp*nsplit = (heads/GF)*nsplit` items over 256 CUs; nsplit must fill
/// the machine (n_grp*nsplit >= n_cu) without over-splitting short contexts (FlashMerge crit-path
/// busy scales with nsplit). fill_base = ceil(n_cu/n_grp); halve it below 2k ctx. Env PLOW_GLM_NS
/// pins nsplit directly (occupancy sweeps).
///
/// `heads` MUST be the PER-RANK head count (nh_l = n_head/tp), NOT the global n_head. The kernel
/// runs this rank's nh_l head-shard, so its work-item count is (nh_l/GF)*nsplit, and the chip-fill
/// CAP is `fill = ceil(n_cu / (nh_l/GF))`. Sizing that cap from the global n_head (the pre-TP bug)
/// pinned it to tp=1's 16 head-groups => the FlashMlaDecode op ran on 32 of 256 CUs at tp=8.
///
/// The split count is NOT simply "fill the chip": MLA decode is latent-HBM-reread-bound, so more
/// splits => more CUs streaming the latent in parallel => the decode OP drops ~1/nsplit (measured:
/// tp=8 ctx-32k decode 154->56us, 2.77x, decode_eff 244->676 GB/s at full fill). BUT plow's
/// FlashMerge is a SEPARATE O(nsplit) pass, so past a point its growth (tp=8 merge 28->54us at
/// ns 16->128) eats the decode saving — full fill REGRESSES the decode CHAIN at mid ctx (tp=8 8k
/// chain 123->155us at ns 128). The cost optimum balances the two: d/dns[ latent/nsplit + k*nsplit ]
/// = 0 => nsplit grows with the latent stream (~ctx), capped at `fill` (chip) and `kv_tiles` (no
/// empty splits). tp=1 is fill-capped to 16 (already chip-full), so byte-identical. See
/// the design notes and Plow.SplitK (the split reduction equals the sequential sum for
/// ANY nsplit; occupancy is monotone in the split count up to n_cu).
///
/// ## THE LADDER, RE-MEASURED END TO END (GLM-5.2 TP4, gfx950, default arm-absent object)
///
/// The previous constant (`ctx/512`) was fitted against a `MlaMergeFold` that no longer exists —
/// its body was rewritten and its dispatch then narrowed 256 -> 128 workgroups (3699ff1). The
/// docstring's argument is a BALANCE, so a cheaper merge moves the balance point. It did, by
/// exactly one doubling, and only at one rung. Per-layer CHAIN in us (min(t_ready) -> p90(t_end)
/// over the flash, the merge, and the gate between them, summed over layers; `glm52_decode
/// --sweep` + `scripts/glm52_trace_analyze.py --chain`, GLM_NLAYERS=8 search vehicle):
///
///  live ctx |   1024   2048   4096   8192  16384  32768  65536
///  ---------+------------------------------------------------
///  ns =  16 |  61.06  63.05  66.62  90.12 118.80 202.19 343.67
///  ns =  32 |  65.85  67.09  67.23  73.27 103.67 148.18 223.50
///  ns =  64 |  81.11  79.72  79.77  81.28  88.11 135.86 183.05
///  ns = 128 |      -      -      -  118.66 251.86 141.27 220.11
///  optimum  |     16     16     16     32     64     64     64
///
/// `ctx/512` picks 16 at 8192 (want 32) and 32 at 16384 (want 64) — WRONG at two of the seven
/// rungs. `ctx/256` reproduces ALL SEVEN. The floor stays 16: it is the measured optimum at 1024,
/// 2048 and 4096, and the two arms are within 5.6 us there, so lowering the knee costs nothing at
/// the short end. Nothing above 64 is ever wanted — ns=128 is the worst arm at every ctx it was
/// run at, which is the O(nsplit) merge growth this rule exists to bound, still intact.
///
/// The change is INERT for any blob whose `ctx` is >= 16384: `fill` = 64 at TP4/TP8 and the
/// clamp binds first. It moves the max_ctx-8192 blob (16 -> 32, -16.85 us/layer = -1.31 ms/token
/// over 78 layers) and the max_ctx-16384 blob (32 -> 64, -15.56 us/layer = -1.21 ms/token).
///
/// ## `ctx` HERE IS THE EMIT-TIME max_ctx, NOT THE LIVE LENGTH — and the table above says that
/// ## costs real time. `d_flash_mla_decode` splits over the LIVE `kv_len`, so ONE blob runs ONE
/// nsplit across every context it serves: a 135168-max-ctx server blob runs ns=64 at ctx 1024
/// (81.1 us vs 61.1 achievable, +33%) and at ctx 8192 (81.3 vs 73.3, +11%). Making `i[4]` track
/// `kv_len` is a runtime change, not a constant change — the AMD tick already re-patches
/// instruction fields per step (`glm52_decode.c` STEP patches ckv/krot `i[]`) so the mechanism
/// exists. It is NOT done here and is the largest remaining item in this knob.
///
/// That also makes the row picked above a CONVENTION, and it should be read as one: `ns(M)` is
/// sized for a decode AT `M`, the top of the range the blob serves and the most expensive point in
/// it. It is the same convention the previous constant used, and it is not free at the other end —
/// the max_ctx-8192 blob's 16 -> 32 is -16.85 us/layer at live 8192 and **+4.79 us/layer at live
/// 1024**. Worth revisiting only together with the live-`kv_len` change, which dissolves the
/// trade-off rather than re-balancing it.
pub(crate) fn glm_nsplit(ctx: u32, heads: u32) -> u32 {
    /// KV rows staged per flash step (op_attention.h FA_BKV) — the KV-tile granularity a split
    /// divides. A split covering zero whole tiles writes -inf and is pure overhead (a launched
    /// workgroup + an extra O(nsplit) merge input), so nsplit is capped at the tile count.
    const FA_BKV: u32 = 32;
    /// Latent bytes per split at which the decode saving stops beating the O(nsplit) merge growth.
    /// ns scales as ctx/NS_PER (measured knee) below the fill cap. 512 -> 256 is the re-measured
    /// knee after the `MlaMergeFold` rewrite + dispatch narrowing made the merge cheaper; see the
    /// ladder in the header. Do not move it without re-running that table — 512 and 256 differ on
    /// exactly two rungs and both of those rungs were measured, not interpolated.
    const NS_PER: u32 = 256;
    /// Split floor: below this the fixed decode overhead already dominates, so extra splits only
    /// add merge cost. RE-MEASURED and unchanged: ns=16 is still the chain optimum at ctx 1024,
    /// 2048 and 4096 (61.1/63.1/66.6 us vs ns=32's 65.9/67.1/67.2), so the cheaper merge did NOT
    /// move the short end — only the 8192 rung.
    const NS_FLOOR: u32 = 16;
    /// The largest split count ANY measured context preferred, and a deliberate limit on the blast
    /// radius of `NS_PER`. The ladder in the header is TP4 (nh_l=16), where `fill` is 64 and this
    /// ceiling never binds. It binds only where `fill` is larger — nh_l=8 (tp8, fill 128) and
    /// nh_l=4 (tp16, fill 256) — and THOSE WERE NOT SWEPT. Without it, halving `NS_PER` would
    /// silently double tp8's long-ctx nsplit 64 -> 128 on the strength of a tp4 measurement, and
    /// the tp8 trade is genuinely different: `mla_fold_cus` sizes the merge to `nh_l*ceil(v/VT)`,
    /// so tp8 reduces the same total merge work through HALF the workgroups at TWICE the depth.
    /// ns=128 was also the worst arm at every tp4 ctx it ran at (118.7 / 251.9 / 141.3 / 220.1 us
    /// at 8k / 16k / 32k / 64k). Raise it only with a tp8 ladder behind it.
    const NS_CEIL_MEASURED: u32 = 64;
    let n_grp = (heads / GLM_MLA_GF).max(1);
    let fill = ((256 + n_grp - 1) / n_grp).max(1); // chip-fill cap: splits to cover 256 CUs
    let kv_tiles = ctx.div_ceil(FA_BKV).max(1); // never split finer than there are KV tiles
                                                // ctx-scaled cost optimum, floored, then capped by the chip and the KV-tile count.
    let ns = (ctx / NS_PER)
        .max(NS_FLOOR)
        .min(NS_CEIL_MEASURED)
        .min(fill)
        .min(kv_tiles)
        .max(1);
    emit_config::active()
        .glm_ns
        .filter(|&v| v >= 1)
        .unwrap_or(ns)
}
/// Weight encoding for the MoE expert path — the `i[3]` field ops 85/86 dispatch on.
///
/// A runtime FIELD, not an opcode or a rebuild, and that is the measured design: the kernel side
/// carries bf16 + block-fp8 + A4W4 bodies in ONE object at 256 VGPR / 0 AGPR / occupancy 2 / spill 2,
/// identical to bf16 + block-fp8 alone. So a precision change is a field change. Values mirror
/// `PLOW_MOE_ENC_*` in `runtime/amd/op_moe.h`; `Bf16`/`Fp8Blk` reproduce today's emission byte for
/// byte, so nothing that does not ask for MXFP4 moves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MoeEnc {
    Bf16 = 0,
    Fp8Blk = 1,
    /// A4W4: MXFP4 on BOTH operands. The gate/up op quantizes the gathered activation to fp4
    /// during its A-staging (so there is no separate activation-quant op) and its epilogue IS the
    /// bridge — SwiGLU, MXFP4 quantize and the E8M0 scale write all happen there, so `fu` is
    /// written already-fp4 in the sorted layout DOWN reads and no bf16 intermediate exists.
    Mxfp4 = 2,
}

impl MoeEnc {
    /// The `i[]` slot the encoding travels in on the PREFILL grouped ops (85/86). There `n_exp` is
    /// `i[2]`, so `i[3]` was free.
    pub(crate) const PREFILL_SLOT: usize = 3;
    /// The `i[]` slot on the DECODE expert ops (45/46/48/49). It is NOT the same slot, and the
    /// difference is load-bearing rather than cosmetic: those four ops predate the encoding field
    /// and already use `i[3]` for `n_exp`. Writing the encoding there would set `n_exp = 2`, every
    /// expert id >= 2 would hit `if (eid >= n_exp) return;` — the sentinel skip — and the layer
    /// would produce ZEROS. No fault, no trap (the AMD dispatch default writes nothing either), just
    /// a dead MoE behind fluent-looking output. `i[6]` was free on all four.
    pub(crate) const DECODE_SLOT: usize = 6;

    fn code(self) -> u32 {
        self as u32
    }
    fn from_flags(use_fp8: bool, mxfp4: bool) -> Self {
        match (mxfp4, use_fp8) {
            (true, _) => MoeEnc::Mxfp4,
            (false, true) => MoeEnc::Fp8Blk,
            (false, false) => MoeEnc::Bf16,
        }
    }
}

/// The weight encoding of the DENSE prefill GEMM this MoE/MLA family emits, as the tile
/// selector sees it.
///
/// Only two answers, and that is not an oversight: the dense projections in this family are
/// either fp4-weight (`MoeEnc::Mxfp4`, w4a16 — the A operand and the MFMA stay bf16) or plain
/// bf16. `MoeEnc::Fp8Blk` is **block-scaled** fp8 over a `[128,128]` grid with arbitrary f32
/// scales, and there is no **DENSE** prefill GEMM arm for it on gfx950 — so mapping it to
/// `QuantScheme::W8A8` here would ask the selector for a per-row/per-channel fp8 rung that cannot
/// express this weight layout, and it would emit an opcode that reads the scales wrongly. `None`
/// keeps it on the bf16 path it already takes.
///
/// **CORRECTION (2026-07-28).** This comment used to justify itself with "every `*_FP8_BLK` opcode
/// is decode-only". **That is false**, and it had already propagated into an external code review.
/// Ops 85/86 (`MoeGroupGluPf` / `MoeGroupDownPf`) carry a genuine block-fp8 PREFILL arm —
/// `d_moe_group_pf_t<FP8=true, …>` at `runtime/amd/op_moe.h:1675,1701`, selected by `i[3] ==
/// PLOW_MOE_ENC_FP8BLK` and reading the real `[128,128]` grid via `KB = (K + 127) >> 7`. It is what
/// the whole-layer GLM prefill already runs its routed experts on.
///
/// The conclusion above survives the correction, because those two are **grouped-MoE** ops: their
/// contract is the expert weight/scale tables, `MoeAlignPf`'s `meta` row-count table, `row_token`
/// gather indices and `row_partidx`/`row_gate` scatter+scale maps, with DOWN writing an f32
/// `part[T*k, H]`. Nothing there can serve a plain `o_proj` or a dense GLU.
///
/// **SECOND CORRECTION (`GemmFp8Blk`, opcode 107).** The dense arm now exists, so
/// "block-fp8 has a grouped prefill arm and no dense one" is no longer the state of the world and
/// `glm_linear_fp8` is no longer refused on a prefill emit. This function is STILL right to answer
/// `None` for `Fp8Blk`, for a different reason than it used to have: `GemmFp8Blk` is not in
/// `gfx950_gemm_inventory` and is not tile-SELECTABLE at all. It carries ONE rung, because an
/// arbitrary-f32 block scale must be promoted into a second accumulator and the top two of the five
/// rungs cannot be built at 8 waves (see `emit_pf_gemm_fp8_blk` and `d_gemm_fp8_blk`). So the
/// emitters that need it call it DIRECTLY and never route through `pick_tile`; asking the selector
/// for a `BlockFp8` rung here would still be asking for something that does not exist.
fn mxfp4_quant(enc: MoeEnc) -> kernelcaps::QuantScheme {
    match enc {
        MoeEnc::Mxfp4 => kernelcaps::QuantScheme::Mxfp4,
        _ => kernelcaps::QuantScheme::None,
    }
}

/// MXFP4 block size: one E8M0 scale byte per 32 packed fp4 values (OCP microscaling). Mirrors the
/// `>> 5` block index in `op_moe.h`'s A/C staging.
pub(crate) const MX_BLOCK: u32 = 32;

/// `GLM_SHARD_HEAD=1` — vocab-column-parallel `lm_head`, and this rank's slice of the vocab.
///
/// The replicated default costs every rank the FULL `vocab*hidden` bf16 table on every decode step
/// — 1.90 GB/rank/token at GLM-5.2's 154880x6144, measured at 0.29-0.33 ms and **106% of the
/// 6200 GB/s ceiling**, i.e. entirely a sharding gap and not a kernel one. It is deliberate: the
/// column-parallel arm needs `XArgmaxFin` to fold the per-rank maxima, and that op was a no-op stub
/// (`runtime/amd/interp.hip`), so sharding without it gives every rank a quarter of the logits and
/// they disagree on the first token. The body now exists (`d_xargmax_fin_mega`), so this arm is
/// real — but the two must move TOGETHER, which is why one env var gates both.
///
/// Refuses a vocab that does not divide: a ragged last shard would need a per-rank vocab_l in the
/// packet, and the packet is rank-agnostic by construction.
///
/// **MEASURED −0.26 ms/token** on GLM-5.2 TP4 at ctx 1k, bit-identical over 256 generated tokens
/// including 11 whose ids fall outside rank 0's vocab shard (`perf-data/glm52-decode-emitter-abs.md`
/// §1). That matches the −0.230 ms of bandwidth floor the change removes, which is what an op at
/// 106% of the HBM ceiling is supposed to do. The host side must agree: `glm52_decode.c`'s
/// `glm_col` and `plowrt`'s `asset::shard::slice_for` both key it on the DECLARED size, so a packet
/// built without this knob still binds the full table.
fn glm_shard_head(c: &GlmCfg) -> bool {
    let on = c.tp > 1 && emit_config::active().glm_shard_head;
    assert!(
        !on || c.vocab % c.tp == 0,
        "GLM_SHARD_HEAD needs vocab ({}) divisible by tp ({}) — one blob serves every rank, so a \
         ragged last shard has no place to record its own width",
        c.vocab,
        c.tp
    );
    on
}

/// This rank's lm_head vocab shard (== `c.vocab` when replicated).
fn glm_vocab_l(c: &GlmCfg) -> u32 {
    if glm_shard_head(c) {
        c.vocab / c.tp
    } else {
        c.vocab
    }
}

/// `GLM_LINEAR_FP8=1` — keep `o_proj` and the shared expert on their CHECKPOINT block-fp8 form
/// instead of the bf16 the host prep dequantises them to.
///
/// GLM-5.2 is a block-fp8 checkpoint, but `scripts/glm52_prep.py` writes `o_proj` and the three
/// `shared_experts.*` projections as `p_bf16(dequant_blockfp8(...))` — 5.09 GB of the 19.10
/// GB/rank/token weight stream carried at 2 B/elt. The fp8 bytes and the `[128,128]`
/// `weight_scale_inv` grid are both on disk as WHOLE tensors, so republishing them is a byte copy:
/// verified element-wise, `bf16_round(fp8 * scale)` equals the prepped bf16 tensor bit for bit, so
/// this arm is the UN-ROUNDED form of the same weight, not a requantisation.
///
/// Scope, and why it is exactly these four projections: the conversion needs an opcode that reads a
/// `[128,128]` grid, and at decode that is `GEMV_FP8_BLK` (44) and `DENSE_GLU_FP8_BLK` (47). Those
/// cover `o_proj` (plain GEMV), the shared gate/up (fused GLU) and the shared down (plain GEMV).
/// The rest of the dequantised stream sits behind `GemvQkv` (fusions A and G) and `MlaMergeFold`,
/// neither of which has a block-fp8 arm — full accounting in
/// `perf-data/glm52-weight-stream-split.md`.
///
/// **PREFILL AND DECODE, since [`DevOp::GemmFp8Blk`] landed.** It was decode-only, and
/// `declare_glm_rows` REFUSED a stacked emit outright, for one reason: the knob re-declares these
/// four handles at 1 B/elt and only the decode emitters routed them to a block-fp8 opcode (44/47),
/// so `emit_glm_mla_prefill`'s `o_proj` and `emit_glm_block_prefill`'s shared expert would have put
/// a bf16 `Gemm`/`GemmGlu` on fp8 bytes and run off the end of all four tensors — no fault,
/// plausible output, wrong model. That is knob-contract §4's bug shape with the polarity reversed:
/// *the weight was swapped under an arm that was never told.*
///
/// Both prefill emitters now consult this function and route to [`emit_pf_gemm_fp8_blk`]:
/// `o_proj` on EVERY layer including the dense ones (`emit_glm_dense_block_prefill` calls the same
/// MLA emitter), and the shared expert as two `GemmFp8Blk` halves plus a `Glu` — unfused, because
/// there is no `GemmGluFp8Blk`, exactly as the MXFP4 prefill arm already unfuses for the same
/// reason. So the refusal is gone; the four handles have a T-row arm on both phases.
///
/// **MEASURED — and the recorded verdict has moved twice, in both directions.** It removes 2547
/// MB/rank/token = **−0.431 ms of floor** (`perf-data/glm52-weight-stream-split.md`). At the token,
/// against its own contemporaneous control each time:
///
/// | object | verdict |
/// |---|--:|
/// | pre-`MLA_MERGE_FOLD`-rewrite | −0.05 ms (noise) |
/// | post-fold-rewrite (`glm52-decode-emitter-abs.md` §2) | **+0.39 ms** (a regression) |
/// | 2026-07-28 12:19 object, `CORESIDENT=1` (`glm52-moe-tail-ab.md` §3.1) | **−0.44 ms** |
/// | `CORESIDENT=1`, decode-only (`glm52-linear-fp8-reeval.md` §3.1, n=3) | −0.13 ± 0.10 (noise) |
/// | SHIPPING decode knobs, decode-only (§3.2, n=6) | **−0.31 ± 0.14** (74% of floor) |
/// | **SHIPPING knobs, STACKED** (`glm52-gemm-fp8-blk.md` §8, n=6) | **−0.417 ± 0.175** (101%) |
///
/// This is §6b-STALE in both directions on one knob: the blob is unchanged and the interpreter
/// under it is not. **Do not quote a number here without checking which object produced it** —
/// re-derive against the current one with interleaved controls.
///
/// The reason it kept landing on both sides of zero is arithmetic, not sloppiness: `GemvFp8Blk` runs
/// at **966 GB/s** where bf16 `Gemv` on the same shapes runs at **1728**, so half the bytes through
/// a kernel 1.8x slower per byte is **break-even by construction** — the knob sits on the zero line
/// and the surrounding slack decides the sign. The last row is the first time it reached its whole
/// predicted floor, and that is the token around it having got ~9% shorter, NOT the kernel gap
/// closing. `gemv_rows_fp8_blk`'s memory-level parallelism is still the KERNEL item; it is now what
/// would take this knob PAST its floor rather than up to it.
///
/// Gate for shipping it is `scripts/glm52_prefill_gate.sh`'s single-block B4 oracle plus, for the
/// PREFILL arm this knob now carries, `runtime/tests/block_fp8_gfx950_test.c`'s `run_gemm_blk` —
/// NOT token identity: greedy decode on this checkpoint forks within 3 tokens between every arm
/// INCLUDING the bf16 control, so the token stream carries no signal about a precision change here.
///
/// Requires the weight dir published by `scripts/glm52_prep_fp8_linear.py` (`.weight_fp8` +
/// `.weight_scale_inv`, additive; the bf16 `.weight` stays where it is).
fn glm_linear_fp8(enc: MoeEnc) -> bool {
    enc == MoeEnc::Fp8Blk && emit_config::active().glm_linear_fp8
}

/// OPT-IN (`PLOW_GLM_OFOLD=1`): the MlaMergeFold+o_proj fusion (fusion-audit seam 1).
///
/// At prefill nsplit==1 the merge is exactly `oat = normalize(opart) × W_uv`, so the pair
/// collapses to ONE GEMM `og_tp = opart_norm × W_ofold` against the prep-derived
/// `derived.o_fold.weight` (= W_uv·W_o per head, scripts/glm52_prep_ofold.py). The V2 flash
/// writes NORMALIZED bf16 rows via packet i[6] (its epilogue holds each row's `l`), the
/// merge packet disappears, and the [T, nh_l*VD] `oat` round trip plus the mlpart write go
/// with it. ~1.84× more o-GEMM FLOPs, but measured o_proj rate makes the fused GEMM cheaper
/// than the pair (audit table). bf16 GEMM path only — the block-fp8 and MXFP4 o_proj arms
/// have no fused weight; and it REQUIRES the V2 flash routing at serve (the 8-wave kernel
/// ignores i[6] and would leave unnormalized f32 partials for the GEMM to read as bf16) —
/// plowrt refuses an ofold blob unless PLOW_MLA_PF_V2=1 and the flash object carries
/// `plow_glm_ofold_arm`. Numerics: reassociated (fused weight, f32 carried through K),
/// logit-gate class, NOT bit-identical.
fn glm_ofold(enc: MoeEnc) -> bool {
    emit_config::active().glm_ofold && !glm_linear_fp8(enc) && enc != MoeEnc::Mxfp4
}

/// OPT-IN (PLOW_GLM_FP8_KV=1): store the MLA latent cache (`kv.{l}.ckv`) as e4m3 with a per-row
/// f32 scale (`kv.{l}.scale`), exactly the K3 form — the latent writer becomes `HeadNormRopeFp8`
/// (RMSNorm + quantize + scale record in one pass) and the dense flash swaps to the `*Fp8` MLA
/// opcodes (109/110), which read half the KV bytes. The 64-wide rope cache (`krot`) stays bf16
/// (`i[6]=0`), matching the shipped K3 arm. NOT bit-identical to the bf16 arm — e4m3 has 3
/// mantissa bits — so this ships only behind its own serve gate. Unset = byte-identical emit.
///
/// Incompatible with an armed DSA gate: `FlashGatherDecode` has no fp8 twin (its `t7` is the idx
/// table, where the scale array would live), so a gathered packet would read fp8 bytes as bf16.
/// The emitters assert.
fn glm_fp8_kv() -> bool {
    emit_config::active().glm_fp8_kv
}

/// One DENSE prefill GEMM against CHECKPOINT block-fp8 weights — [`DevOp::GemmFp8Blk`] (107).
///
/// `C[t, nn] bf16 = A[t, k] bf16 · W[nn, k] e4m3`, with the checkpoint's own `[128,128]`
/// `weight_scale_inv` grid at `t[3]`. The kernel indexes it `S[(n >> 7) * ceil(K/128) + (k >> 7)]`
/// — byte for byte the convention `gemv_rows_fp8_blk` (44), `d_dense_glu_fp8_blk` (47) and
/// `d_moe_group_pf_t<FP8=true>` (85/86) already read, which is why this takes the SAME handles the
/// decode emitters bind rather than a re-quantised copy.
///
/// # Why this exists rather than a re-route
///
/// It is the arm whose absence made [`glm_linear_fp8`] decode-only. `GemmFp8` (33) / `GemmGluFp8`
/// (36) are the **w8a8** rung — one f32 per output CHANNEL plus a per-row activation scale from
/// `QuantFp8` — so they can neither address a `[128,128]` grid nor run without an fp8 A operand
/// this path does not produce. Ops 85/86 do carry a real block-fp8 prefill body, but only under
/// the grouped-MoE contract (expert weight/scale TABLES, `MoeAlignPf`'s `meta`, `row_token` gather
/// indices, `row_partidx`/`row_gate` scatter+scale maps, f32 `part[T*k, H]` output). A plain
/// `o_proj` has none of that. See `d_gemm_fp8_blk` in `runtime/amd/op_gemm.h`.
///
/// # No tile selection, and it is a register fact
///
/// Emitted DIRECTLY, not through [`pick_tile`]. An arbitrary-f32 block scale cannot ride the cvt's
/// E8M0 `scalef32` operand, so it must be PROMOTED into a second f32 accumulator every 128 K —
/// which DOUBLES a tile's accumulator cost. Of the five rungs the bf16/w8a8/mxfp4 families carry,
/// 192x256 would need 192 accumulator registers and 256x256 would need 256 (the whole AGPR file,
/// i.e. it cannot run 8 waves at all). Only 64x128 / 128x128 / 128x256 are buildable, and a
/// `QuantScheme::BlockFp8` row in `gfx950_gemm_inventory` must name five opcodes. Adding rungs
/// there also re-stales every tunedb record. One rung, chosen once, recorded in the kernel header.
fn emit_pf_gemm_fp8_blk(
    b: &mut Builder,
    cus: &[u32],
    out: u32,
    x: u32,
    wt: u32,
    sc: u32,
    t: u32,
    nn: u32,
    k: u32,
    deps: &[u32],
) -> u32 {
    // A block-fp8 weight whose scale handle is TENSOR_NONE is a null pointer in the kernel's
    // promotion, i.e. a fault or garbage rather than a wrong number. Both handles come from the
    // same `lin_fp8` branch in `declare_glm_rows`, so this can only fire on a future edit that
    // splits them — which is exactly when it is worth having.
    assert!(
        wt != TENSOR_NONE && sc != TENSOR_NONE,
        "GemmFp8Blk needs BOTH the .weight_fp8 bytes and the .weight_scale_inv grid \
         (weight={wt}, scale={sc}); `declare_glm_rows` declares them as a pair under \
         GLM_LINEAR_FP8 and neither is optional"
    );
    // KEXACT. `d_gemm_fp8_blk` is instantiated with the exact-K B-fetch, so a K that is not a whole
    // number of BK=64 tiles reads 8 halves past the row end and the MFMA silently accumulates the
    // NEXT output channel's weights — plausible output, wrong model, no fault. A ragged N is fine
    // (guarded per element) and a ragged M is fine (guarded per row); only K is unforgiving.
    // Every real block-fp8 K is a 128-multiple by construction (the scale grid is [128,128]), so
    // this can only fire on a shape the checkpoint could not have quantised this way.
    assert!(
        k % 64 == 0,
        "GemmFp8Blk needs K % 64 == 0 (K = {k}); the kernel is the KEXACT instantiation and a \
         ragged K-tile would read past the weight row into the next output channel with no fault"
    );
    b.emit(DevOp::GemmFp8Blk, cus.to_vec(), deps, |d| {
        d.t[0] = out;
        d.t[1] = x;
        d.t[2] = wt;
        d.t[3] = sc;
        d.i[0] = t;
        d.i[1] = nn;
        d.i[2] = k;
    })
}

/// `GLM_SHARED_GLU_SPLIT=1` — run `GLM_LINEAR_FP8`'s shared gate/up as TWO CO-RESIDENT
/// `GEMV_FP8_BLK` (44) halves + a `Glu` (5) instead of ONE `DENSE_GLU_FP8_BLK` (47).
///
/// **MEASURED AT −0.017 ms, i.e. NOTHING, and OPT-IN for that reason.** GLM-5.2 TP4, 4x gfx950,
/// real weights, ctx 1024, 65 steps, 3 interleaved folds, own contemporaneous control:
/// 29.504 / 29.524 / 29.551 -> 29.493 / 29.513 / 29.521 ms/token. The kernel arithmetic below is
/// right and the isolated ratio is 3.8x; it did not survive contact with the interpreter, which is
/// the third instance of an isolated-kernel win being falsified at the endpoint
/// (`perf-data/glm52-moe-tail-ab.md` §3). It also costs ONE EXTRA bf16 ROUNDING — op 47 keeps `g`
/// and `u` in f32 registers through the SwiGLU, the split writes both to bf16 and re-reads them —
/// so with no time to show for it there is no reason to make it the default. Kept as the
/// instrument that priced it, and as the record of the mechanism.
///
/// **Op 47's kernel is genuinely 2.2x slower per byte, and the cause is its WORK WALK, not its
/// bytes.** Isolated, GLM TP4 shape `(imoe_l=512, H=6144)`, 6200 GB/s denominator:
///
/// | shared gate/up | us | % of ceiling |
/// |---|--:|--:|
/// | bf16 `GemvGlu` (19), shipped today | 3.26 | 62.2 |
/// | fp8 `DenseGluFp8Blk` (47) | **7.17** | 14.2 |
///
/// Half the weight bytes, 2.2x the wall time. `d_dense_glu_fp8_blk` walks
/// `n = slice*PLOW_WAVES + wave`, so at `N=512` over `nblk=256` only slices `0..63` get any output
/// at all: **192 of 256 workgroups run empty**, and the 512 waves that do run use
/// `wave_dot_fp8_blk`, which keeps ONE load in flight per wave and reads `x` from GLOBAL. Op 44's
/// `gemv_rows_fp8_blk` spreads the same outputs over ALL `nblk` workgroups
/// (`gv_per = ceil(N/nblk)`), keeps `UN=3` chunks in flight per column and stages `x` in LDS.
///
/// Splitting the two projections onto DISJOINT CU halves is what recovers the parallelism the
/// concatenated `[2*imoe_l, H]` block would have bought without touching a weight on disk: gate on
/// `shared_cus[..half]` and up on `shared_cus[half..]`, both gated only on the post-attn norm, is
/// `2 * (512 waves over 128 CUs)` = 4 waves/CU — the same wave-per-CU occupancy as one `N=1024`
/// packet over 256 CUs, and 4x the in-flight weight bytes of op 47. Emitting the two halves on the
/// SAME CU set instead would serialise them per workgroup and halve that back.
///
/// Cost: two extra gates per sparse layer (the second GEMV and the `Glu`). `Glu` is 512 elements on
/// ONE workgroup, so it is a gate, not work.
///
/// This is the same unfusing `emit_glm_moe_prefill_block` already does on the MXFP4 arm (two
/// matmuls + `Glu` because there is no `GemmGluMxfp4`) — same operand slots, same `n.shfu_up`.
fn glm_shared_glu_split(enc: MoeEnc) -> bool {
    glm_linear_fp8(enc) && emit_config::active().glm_shared_glu_split
}

/// The two DISJOINT CU halves the co-resident gate/up GEMVs run on.
///
/// Disjointness IS the mechanism. Two packets on the SAME workgroup set run one after the other on
/// each workgroup — the interpreter walks the stream per workgroup — so emitting both halves on
/// `shared_cus` would give 2 waves/CU in flight at a time, exactly the serialisation this exists to
/// undo. A 1-CU slice cannot be split and degenerates to the serial arrangement; `b.emit` refuses
/// an empty CU set, so returning the whole slice twice is the only legal answer there.
fn glm_glu_halves(cus: &[u32]) -> (Vec<u32>, Vec<u32>) {
    if cus.len() < 2 {
        return (cus.to_vec(), cus.to_vec());
    }
    let half = cus.len() / 2;
    (cus[..half].to_vec(), cus[half..].to_vec())
}

/// Gathered-row tile height of the grouped MoE prefill GEMM. Mirrors `MPF_BM` in
/// `runtime/amd/op_moe.h`: the align op pads each expert's gathered-row range up to a whole tile, so
/// the padded row bound is `T*k + n_exp*(MPF_BM-1)` and NOT `T*k`. Sizing the gathered arrays from
/// `T*k` would be an out-of-bounds device write that is invisible at small expert counts.
// 128, not 64: this is a SIZING upper bound, and the OCC4 batched-decode object builds the
// grouped tile at MPF_BM=128 (build_gfx942.sh) while the prefill objects keep 64. The align
// op pads each expert to the OBJECT'S tile height, so the host buffer bound must cover the
// largest variant — undersizing is an out-of-bounds device write with no symptom at low
// expert counts (the header note above). Costs ~25 MB/rank of fu_g at T=8192, n_exp=256.
pub(crate) const MPF_BM: u32 = 128;

/// Router flags: bit0 sigmoid, bit1 norm_topk, bit2 apply e_score_correction_bias to SELECTION
/// only (DeepSeek/GLM noaux_tc). Mirrors FLAGS in the B4 harness.
const GLM_ROUTER_FLAGS: u32 = 1 | 2 | 4;
/// Expert/shared GLU activation = SiLU (SwiGLU). Mirrors ACT in the B4 harness.
const GLM_ACT_SILU: u32 = 1;

/// Per-layer GLM weights. Derived (absorbed / rope-folded) tensors are bf16 and named under a
/// `.derived.` segment the host weight-prep writes; the block-fp8 projections, router, and experts
/// keep their checkpoint names. `TENSOR_NONE` for the sub-block a layer does not have (dense vs MoE).
struct GlmLW {
    // MXFP4 (w4a16) E8M0 scale rows for the matmul weights below — one byte per 32 K-elements,
    // row stride K/32. TENSOR_NONE on the bf16 and block-fp8 arms, which have no such tensor: bf16
    // needs no scale and block-fp8 keeps its own [N/128][K/128] f32 grid under the checkpoint's
    // `weight_scale_inv` name. Norms and the router bias are NOT here — they are not matmul weights
    // and stay bf16/f32 under every encoding, exactly as they do in every fp4 checkpoint.
    qad_s: u32,
    wqa_s: u32,
    wqr_s: u32,
    ckvd_s: u32,
    krotd_s: u32,
    wo_s: u32,
    wr_s: u32,
    shg_s: u32,
    shu_s: u32,
    shd_s: u32,
    // MLA attention (bf16 norms + derived absorbed/rope-folded weights).
    gin: u32,    // input_layernorm
    qad: u32,    // q_a_proj (H->QL)
    gqa: u32,    // q_a_layernorm
    wqa: u32,    // DERIVED absorbed q_nope   [NH*DK, QL]
    wqr: u32, // DERIVED RAW q_rope down (q_b_rope, NOT folded) [NH*DR, QL]; RoPE applied dynamically
    ckvd: u32, // DERIVED kv_a latent down    [DK, H]
    gkva: u32, // kv_a_layernorm
    krotd: u32, // DERIVED RAW k_rope down (kv_a rope slice, NOT folded) [DR, H]; RoPE applied dynamically
    wuv: u32,   // DERIVED absorbed value       [NH*DK, VD]
    wo: u32,    // o_proj (NH*VD -> H)
    wofold: u32, // DERIVED fused W_uv·W_o (NH*DK -> H), prefill ofold arm only; else NONE
    gpost: u32, // post_attention_layernorm
    // MoE (sparse layers): router + shared expert + the two loader-filled pointer tables.
    wr: u32,   // mlp.gate.weight [E,H] bf16
    bias: u32, // mlp.gate.e_score_correction_bias [E] f32
    shg: u32,  // shared_experts.gate_proj
    shu: u32,  // shared_experts.up_proj
    shd: u32,  // shared_experts.down_proj
    ewt: u32,  // expert_weight_table [E*3] u64 device ptrs (loader-filled from bound experts)
    est: u32,  // expert_scale_table  [E*3] u64 device ptrs (block-fp8 scale grids)
    // PRESHUFFLED twin of `ewt` (PLOW_MOE_PF_SHUF=1, else TENSOR_NONE): pointers into a SECOND
    // packed slab whose per-projection layout is B'[K/64][R][64], so the grouped prefill GEMM's
    // per-k-tile B stream is one contiguous 16 KiB block instead of 64 B row-slices at K-stride.
    // Prefill-only — the decode expert GEMVs keep streaming whole rows from `ewt`'s slab.
    ewt_pf: u32,
    // Dense FFN (layers < first_k_dense): block-fp8 gate/up/down + their weight_scale_inv grids.
    // TENSOR_NONE on MoE layers.
    dgate: u32,
    dgate_s: u32,
    dup: u32,
    dup_s: u32,
    ddown: u32,
    ddown_s: u32,
    // PREFILL-ONLY pointer tables for the dense FFN, in the SAME [e*3] = {gate, up, down} layout
    // `ewt`/`est` use for the routed experts — because the dense prefill runs on the grouped
    // expert arms (ops 85/86) with n_exp = 1. See `emit_glm_dense_block_prefill`. Host-filled
    // with DEVICE ADDRESSES of `dgate`/`dup`/`ddown` (+ their `weight_scale_inv` grids), exactly
    // as `bind_packed_experts` fills `ewt`/`est`; they are not checkpoint tensors. TENSOR_NONE on
    // MoE layers AND on a decode-only emit, which is what keeps existing blobs byte-identical.
    dwt: u32,
    dst: u32,
    // DSA lightning indexer (TENSOR_NONE except on 'full' layers with the DSA gate on).
    iwqb: u32, // indexer.wq_b.weight (fp8 [HI*DI, QL]) + iwqb_s scale grid
    iwqb_s: u32,
    iwk: u32, // indexer.wk.weight (fp8 [DI, H]) + iwk_s scale grid
    iwk_s: u32,
    iknw: u32, // indexer.k_norm.weight [DI] bf16
    iknb: u32, // indexer.k_norm.bias   [DI] bf16
    iwp: u32,  // indexer.weights_proj.weight [HI, H] bf16
}

/// The GLM tensor table. Decode-shaped activations (one row) + per-layer latent/rope caches +
/// per-layer weights. Prefill activations are a later (B-sweep) concern.
// ids/pos/emb/fin/head/logits/amax are the embed + lm_head scaffolding the SINGLE-layer gate
// declares but does not yet consume — the full 78-layer decode phase (next milestone) wires them.
#[allow(dead_code)]
pub(crate) struct GlmTn {
    ids: u32,
    pos: u32,
    kvlen: u32,
    cos: u32,
    sin: u32,
    emb: u32,
    fin: u32,
    head: u32,
    // MLA activations
    x: u32,
    xn: u32,
    qlr: u32,
    qlat: u32,
    ckvraw: u32,
    qa: u32,
    qrr: u32, // raw q_rope (pre-RoPE) [NH*DR]
    qr: u32,
    krr: u32, // raw k_rope (pre-RoPE) [DR]
    opart: u32,
    mlpart: u32,
    olat: u32,
    oat: u32,
    attn: u32,
    xmid: u32,
    xn2: u32,
    /// fp8 twin of xn2 + its per-token f32 scales (PLOW_MOE_PF_A8); TENSOR_NONE otherwise.
    xn2q: u32,
    xn2s: u32,
    // MoE activations
    tab: u32,
    rlogit: u32, // router score-GEMV output [n_exp] bf16 (feeds MoeRouterTopk)
    shfu: u32,
    /// `up` half of the shared-expert GLU — MXFP4 prefill only (no fused GemmGluMxfp4 exists).
    shfu_up: u32,
    shared: u32,
    fu: u32,
    dfu: u32, // dense-FFN intermediate [dense_inter] (layers 0-2)
    part: u32,
    xnext: u32,
    logits: u32,
    amax: u32,
    // TP peer partials + zero residual (TENSOR_NONE at tp==1)
    og_tp: u32,
    dg_tp: u32,
    zero_h: u32,
    /// Byte offset of peer partial slot **B** (`dg_tp`) inside the peer-scratch region — the
    /// `i[2]` operand of every FFN `XReduce`/`XReduceTwoShot` this tensor set emits.
    ///
    /// ONE value for the WHOLE blob, `rows_max * hidden * 2`, exactly as the non-MLA emitter
    /// computes it (`crates/devgen/src/lib.rs`, `slot_b = rows_max * c.hidden * BF16`) and exactly
    /// what the host binds `act.dg_tp` at (`exec/amd.rs`: `scratch_base + TpBind::slot_b`, taken
    /// from `DevBlob::tp.slot_bytes` = `max(i[2])` over EVERY program).
    ///
    /// It must not be per-program. This used to be `t * hidden * 2` in the prefill emitters and
    /// `hidden * 2` in the decode ones, so a bucket ladder + decode blob baked FOUR different
    /// offsets; the host can bind `dg_tp` at only one, so at most one program's FFN all-reduce
    /// found its own partial and the rest reduced whatever untouched peer memory sat at the offset
    /// they named. Zeros, silently: every layer's FFN contribution vanished in prefill AND in
    /// decode, which is why a decode-only blob (max(i[2]) == hidden*2, so the one baked value and
    /// the bound one agree) stayed healthy while the same decode program inside a prefill bundle
    /// emitted a constant token.
    slot_b: u32,
    // DSA indexer (TENSOR_NONE when the DSA gate is off). qidx/kidx_raw/kidx_normed/widx are per-step
    // scratch; iscore/iidx/ighist/igctl are the score+select scratch (shared across layers, sequential);
    // icos/isin are the [ctx][DI/2] identity-tail interleaved-RoPE tables (first qk_rope/2 real, rest 1/0).
    // DSA SPARSE PREFILL scratch (PLOW_GLM_DSA_PF, TENSOR_NONE otherwise). Sized for the widest
    // prefill bucket the ladder carries, like every other prefill activation.
    qidx_pf: u32,     // rope'd indexer queries [T][HI*DI] bf16
    kidx_pf: u32,     // this chunk's indexer keys [T][DI] bf16 (written at the chunk base)
    widx_pf: u32,     // per-token indexer weights [T][HI] bf16
    iscore_pf: u32,   // f32 [T][ctx] per-query indexer scores
    iidx_pf: u32,     // i32 [T][top_k] per-query selection (-1 pads)
    iuni: u32,        // per-64-query-tile union table (see op 119)
    iumask: u32,      // u64 [n_qt][ctx] membership scratch for the union build
    qidx: u32,        // rope'd indexer query [HI*DI]
    kidx_raw: u32,    // wk @ xn [DI] (pre-norm)
    kidx_normed: u32, // k_norm(kidx_raw) [DI] (pre-rope)
    widx: u32,        // weights_proj @ xn [HI]
    iscore: u32,      // f32 [ctx] indexer scores
    iidx: u32,        // i32 [index_topk] selected positions (the gather idx; shared reuse target)
    ighist: u32,      // u32 [7*256] radix histograms (host-zeroed once)
    igctl: u32,       // u32 [3] grid-barrier ctl (host-zeroed once)
    icos: u32,
    isin: u32,
    // MoE PREFILL scratch (TENSOR_NONE when rows == 1). The token-sorted grouped-expert path:
    // `meta` is the [3*n_exp+1] i32 rowoff|cnt|tile-prefix the align op writes and both grouped
    // GEMMs read; row_token/row_partidx/row_gate are the per-GATHERED-ROW maps, sized for the
    // MPF_BM-padded upper bound; fu_g is the gathered GLU output the down GEMM consumes.
    meta: u32,
    row_token: u32,
    row_partidx: u32,
    row_gate: u32,
    fu_g: u32,
    /// E8M0 scale rows for `fu_g` under A4W4 — WRITTEN by the gate/up epilogue, READ by DOWN.
    /// TENSOR_NONE on the bf16 / block-fp8 arms, which have no per-block scale.
    fu_scale: u32,
    // per-emitted-layer caches + weights (index i <-> layer_ids[i]); kidx = indexer key cache [ctx][DI]
    // on 'full' layers (TENSOR_NONE otherwise).
    ckv: Vec<u32>,
    krot: Vec<u32>,
    /// Per-row f32 dequant scales for the fp8 latent cache ([`glm_fp8_kv`]); TENSOR_NONE per
    /// layer on the bf16 arm.
    kv_scale: Vec<u32>,
    kidx: Vec<u32>,
    lw: Vec<GlmLW>,
}

/// Declare the GLM tensor set for the layers in `layer_ids` (real layer indices; the weight names
/// carry the real index so the prepped dir binds them). `lw[i]`/`ckv[i]`/`krot[i]` correspond to
/// `layer_ids[i]`. Activations are decode-shaped (one row).
pub(crate) fn declare_glm(b: &mut Builder, c: &GlmCfg, ctx: u32, layer_ids: &[u32]) -> GlmTn {
    declare_glm_rows(b, c, ctx, layer_ids, 1, MoeEnc::Fp8Blk)
}

/// As [`declare_glm`], but sizes the ROW-DIMENSIONED activations for `rows` — the largest prefill
/// bucket — instead of decode's single row.
///
/// A [`Model`] carries ONE tensor table for every program it holds, so prefill and decode SHARE it
/// and it has to be sized for whichever phase needs more. That is the same rule the dense-GQA
/// `declare` follows ("Prefill and decode SHARE this table"); getting it wrong is not a slowdown but
/// an out-of-bounds device write, which is exactly how the dense path's `ns_pre` under-estimate
/// showed up at tp=8. `rows == 1` reproduces the decode-only table byte-for-byte, so every existing
/// call site is unaffected.
///
/// Only the activations the PREFILL program actually widens are scaled. The ones that stay ONE ROW
/// wide are deliberate, and each for its own reason:
///   * `fu` / `dfu` are the DECODE per-slot expert buffers; prefill's equivalent is the gathered
///     `fu_g`, sized on the MPF_BM-padded row bound instead.
///   * `logits` / `amax` are the lm_head tail, and prefill samples exactly ONE row — the last real
///     one, selected by the lm_head's `a_row0` (see [`emit_glm_tail`]). A [T, vocab] logit matrix
///     would be a 152k-wide GEMM per prompt token thrown away.
///   * the DSA indexer scratch, because the gathered-prefill selector is per-QUERY and nothing
///     emits it yet (see the `FlashGatherPrefill` note on [`emit_glm_mla_prefill`]).
pub(crate) fn declare_glm_rows(
    b: &mut Builder,
    c: &GlmCfg,
    ctx: u32,
    layer_ids: &[u32],
    rows: u32,
    enc: MoeEnc,
) -> GlmTn {
    declare_glm_rows_batched(b, c, ctx, layer_ids, rows, 1, enc)
}

/// As [`declare_glm_rows`], with a DECODE batch: `dbatch` slots share the blob. Three families
/// scale on it and nothing else does:
///   * `in.kvlen` — `[dbatch]` i32. This is also how the loader learns the engine batch
///     (`AmdEngine::load` derives `batch` from `in.kvlen`'s byte length and cross-checks it
///     against the decode program's `t`).
///   * the per-layer KV caches — `[dbatch][ctx][...]`; slot `s`'s block is `s * bytes/dbatch`,
///     which is the stride contract `kv_slot_stride` rebases prefill onto.
///   * the lm_head tail (`logits`/`amax`) and the flash partials (`opart`/`mlpart`) — decode
///     produces one row PER SEQUENCE, unlike prefill's last-row-only sample.
/// Row-dimensioned activations take `max(rows, dbatch)`: a B<=16 decode row set fits inside any
/// real prefill bucket, so on a bucketed blob this is a no-op. `dbatch == 1` reproduces
/// [`declare_glm_rows`] byte-for-byte.
pub(crate) fn declare_glm_rows_batched(
    b: &mut Builder,
    c: &GlmCfg,
    ctx: u32,
    layer_ids: &[u32],
    rows: u32,
    dbatch: u32,
    enc: MoeEnc,
) -> GlmTn {
    let dbatch = dbatch.max(1);
    let rows = (rows.max(1) as u64).max(dbatch as u64);
    let (h, nh, dk, dr, vd, ql, e, tk, imoe) = (
        c.hidden,
        c.heads,
        c.kv_lora,
        c.qk_rope,
        c.v_head,
        c.q_lora,
        c.n_exp,
        c.top_k,
        c.moe_inter,
    );
    // TENSOR-PARALLEL local shards (mirror the dense-GQA declare()): head-, expert- and
    // dense-intermediate-dimensioned tensors run 1/tp wide. tp==1 => *_l == full, byte-identical.
    let tp = c.tp;
    let nh_l = nh / tp; // this rank's q/v heads (column-parallel by head)
    let imoe_l = imoe / tp; // this rank's SHARED-expert/dense intermediate lanes (TP-sharded)
                            // Routed-expert intermediate width: full moe_inter under EP (whole experts, distributed across
                            // ranks — no CU-starve), else the TP shard. Sizes the `fu` gate/up buffer.
    let imoe_e = if c.ep { imoe } else { imoe_l };
    let di_l = c.dense_inter / tp; // this rank's dense-FFN intermediate lanes
    let ib = imoe.div_ceil(128); // expert scale-grid rows (I/128)
    let hb = h.div_ceil(128); // expert scale-grid cols (H/128)
    let db_l = di_l.div_ceil(128); // sharded dense scale-grid rows/cols (di_l/128)
    let db = c.dense_inter.div_ceil(128);
    let ac = |b: &mut Builder, n: &str, sz: u64| b.tensor(&format!("act.{n}"), sz);

    let ids = b.tensor("in.ids", ctx as u64 * I32);
    let pos = b.tensor("in.pos", ctx as u64 * I32);
    let kvlen = b.tensor("in.kvlen", dbatch as u64 * I32);
    // Interleaved partial-RoPE cos/sin tables for the 64 rope dims (GLM theta 8e6, from the
    // config — full rotation of DR).
    // Same [ctx][DR/2] layout the half-split path uses (freq index = element>>1); the interp's HD=64
    // dispatch selects the INTERLEAVE=true template. See rope_tables + op_norm.h.
    // `c.rope_theta()`, not the field: a NoPE cfg refuses here rather than materialising tables
    // for a rotation the model does not have.
    let [cos_t, sin_t] = GenTensor::rope_pair(ctx, c.qk_rope, c.rope_theta(), 1.0, c.rope_scale);
    let cos = b.tensor_gen("in.cos", cos_t.byte_len(), cos_t);
    let sin = b.tensor_gen("in.sin", sin_t.byte_len(), sin_t);
    let emb = b.tensor(
        &format!("{}embed_tokens.weight", c.prefix),
        (c.vocab * h) as u64 * BF16,
    );
    let fin = b.tensor(&format!("{}norm.weight", c.prefix), h as u64 * BF16);
    // lm_head. REPLICATED by default (`crates/plowrt/src/asset/shard.rs`'s module note): every rank
    // computes the full-vocab argmax so they agree on the token without a cross-rank fold, at the
    // cost of all N ranks streaming the whole `vocab*hidden` table every step. GLM_SHARD_HEAD=1
    // takes the vocab-column-parallel arm instead — `glm_vocab_l` rows per rank, a local argmax over
    // this rank's shard, and an XARGMAX_FIN that rebases to global vocab space and folds the N
    // packed keys. Both the emit and the host bind change together; see `glm_emit_full`.
    let head = b.tensor("lm_head.weight", (glm_vocab_l(c) * h) as u64 * BF16);

    let x = ac(b, "x", rows * h as u64 * BF16);
    let xn = ac(b, "xn", rows * h as u64 * BF16);
    let qlr = ac(b, "qlr", rows * ql as u64 * BF16);
    let qlat = ac(b, "qlat", rows * ql as u64 * BF16);
    let ckvraw = ac(b, "ckvraw", rows * dk as u64 * BF16);
    // Head-dimensioned activations shrink to nh_l heads under TP (the flash/merge/uv/o-fold ops run
    // this rank's head-shard); expert/dense-intermediate activations shrink to imoe_l/di_l lanes.
    let qa = ac(b, "qa", rows * (nh_l * dk) as u64 * BF16);
    let qrr = ac(b, "qrr", rows * (nh_l * dr) as u64 * BF16);
    let qr = ac(b, "qr", rows * (nh_l * dr) as u64 * BF16);
    let krr = ac(b, "krr", rows * dr as u64 * BF16);
    // TP-sharded head count (nh_l) x ctx-adaptive nsplit (glm_nsplit, from glm-tune-flash).
    // nh_l (not global c.heads) so the fill target matches this rank's actual work-item count.
    let ns = glm_nsplit(ctx, nh_l);
    // The flash partials are [b][t][head][nsplit][DK] on BOTH phases (`d_flash_mla_decode`'s `oh`
    // index; decode is its n_tok=1 case). Decode writes nh_l*ns of them, prefill writes rows*nh_l*1
    // — nsplit is FORCED to 1 there, because a per-token causal bound leaves an early token's later
    // splits empty and an empty split emits l=0 for the merge to divide by. So the buffer is the max
    // of the two, not their product.
    // PLOW_GLM_PF_NS widens the PREFILL share: the V2 flash's causal KV-split writes
    // rows*nh_l*pf_ns partials (each split a ceil-equal share of its tile's causal range;
    // dead splits carry m=-inf/l=0, which d_mla_merge_fold weighs 0 branch-free — the old
    // "empty split divides the merge" objection predates that merge rewrite).
    let osplits = (ns * dbatch).max(rows as u32 * glm_pf_ns());
    let opart = ac(b, "opart", (nh_l * osplits * dk) as u64 * F32);
    let mlpart = ac(b, "mlpart", (nh_l * osplits * 2) as u64 * F32);
    let olat = ac(b, "olat", (nh_l * dk) as u64 * BF16);
    let oat = ac(b, "oat", rows * (nh_l * vd) as u64 * BF16);
    let attn = ac(b, "attn", rows * h as u64 * BF16);
    let xmid = ac(b, "xmid", rows * h as u64 * BF16);
    let xn2 = ac(b, "xn2", rows * h as u64 * BF16);
    let (xn2q, xn2s) = if rows > 1 && moe_pf_a8() {
        (ac(b, "xn2q", rows * h as u64), ac(b, "xn2s", rows * F32))
    } else {
        (TENSOR_NONE, TENSOR_NONE)
    };
    // MoE activations. Row-dimensioned ones widen for the prefill FFN (the grouped path is
    // token-sorted, so the routing table, the [T,n_exp] logits, the shared-expert lanes and the
    // per-slot partials all carry T tokens); `fu`/`dfu` are the DECODE per-slot buffers and stay
    // one row — prefill's gathered equivalent is `fu_g` below, sized on the padded row bound.
    let tab = ac(b, "tab", rows * tk as u64 * 8);
    let rlogit = ac(b, "rlogit", rows * e as u64 * BF16); // router score output [T][n_exp] bf16
    let shfu = ac(b, "shfu", rows * imoe_l as u64 * BF16);
    // The `up` half of the shared expert's GLU. Needed on the MXFP4 PREFILL arm, where the absence
    // of a GemmGluMxfp4 forces gate and up into separate GEMMs with an explicit Glu between; on the
    // block-fp8 PREFILL arm, where the absence of a GemmGluFp8Blk does the same; and on the
    // `GLM_SHARED_GLU_SPLIT` DECODE arm, which unfuses op 47 into two co-resident op-44 halves for
    // the same reason in reverse (op 47's fusion is what costs it the CU spread).
    let shfu_up = if (rows > 1
        && (enc == MoeEnc::Mxfp4
            || glm_linear_fp8(enc)
            // TASK-9 FIT GATE: the bf16 shared expert also splits into two halves + Glu
            // when rows*K exceeds the staged-LDS fit (see the emit site), and the up half
            // needs this buffer.
            || rows as u64 * h as u64 > crate::gm_lds_halves()))
        || glm_shared_glu_split(enc)
    {
        ac(b, "shfu_up", rows * imoe_l as u64 * BF16)
    } else {
        TENSOR_NONE
    };
    let shared = ac(b, "shared", rows * h as u64 * BF16);
    // Routed-expert gate/up buffer: full moe_inter width per slot under EP (whole experts), else TP shard.
    let fu = ac(b, "fu", (tk * imoe_e) as u64 * BF16);
    let dfu = ac(b, "dfu", di_l as u64 * BF16);
    // part16: bf16 at prefill. The DECODE expert ops still write f32 `part` for their single
    // token — tk*h*F32 bytes, under the prefill bf16 size for every real bucket (rows >= 128) —
    // so one buffer serves both widths with no cross-program flow through it.
    let part_w = if moe_pf_part16() { BF16 } else { F32 };
    // A FUSED 86 -> 87 arm collapses the k per-slot rows into ONE accumulator row per token, so
    // the prefill side of this allocation drops by k (atomic, f32) or k/2 (det, f64) — 1.611 GB
    // -> 0.201 / 0.403 GB per rank at T=8192, H=6144, k=8. The size comes from `moe_pf_fuse`,
    // the SAME function the packet fields come from, because a size that disagreed with the
    // kernel arm is a silent k-fold heap overrun rather than a fault. The `.max()` term is the
    // DECODE expert ops, which still write f32 `part` for their single token out of this buffer.
    let part_pf = match moe_pf_fuse(tk) {
        MoePfFuse::Det => rows * h as u64 * 8,
        MoePfFuse::Atomic => rows * h as u64 * F32,
        MoePfFuse::None => rows * (tk * h) as u64 * part_w,
    };
    let part = ac(b, "part", part_pf.max((tk * h) as u64 * F32));
    // Grouped MoE prefill scratch. `MPF_MAX_ROWS(T,k,n_exp) = T*k + n_exp*(MPF_BM-1)` is the padded
    // gathered-row bound: the align op pads each expert's row range up to a whole MPF_BM tile, so
    // every expert can waste at most MPF_BM-1 rows. Sizing this from T*k alone is an out-of-bounds
    // device write with no symptom at low expert counts and a guaranteed one at 384.
    let (meta, row_token, row_partidx, row_gate, fu_g, fu_scale) = if rows > 1 {
        let pad_rows = rows * tk as u64 + (e * (MPF_BM - 1)) as u64;
        // The gathered GLU output is bf16 on the bf16/block-fp8 arms and PACKED fp4 under A4W4 —
        // half a byte per value plus one E8M0 byte per 32. The buffer is sized for whichever the
        // packet asks for; the fp4 form is SMALLER, so a bf16-sized allocation would merely waste,
        // but the E8M0 rows have no bf16 counterpart and must be declared or the bridge writes to a
        // null handle.
        let fug_bytes = match enc {
            MoeEnc::Mxfp4 => pad_rows * (imoe_e / 2) as u64,
            _ => pad_rows * imoe_e as u64 * BF16,
        };
        (
            ac(b, "moe_meta", (3 * e + 1) as u64 * I32),
            ac(b, "moe_rowtok", pad_rows * I32),
            ac(b, "moe_rowpart", pad_rows * I32),
            ac(b, "moe_rowgate", pad_rows * F32),
            ac(b, "moe_fug", fug_bytes),
            if enc == MoeEnc::Mxfp4 {
                ac(b, "moe_fuscale", pad_rows * (imoe_e / MX_BLOCK) as u64)
            } else {
                TENSOR_NONE
            },
        )
    } else {
        (
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
        )
    };
    let xnext = ac(b, "xnext", rows * h as u64 * BF16);
    let logits = ac(b, "logits", dbatch as u64 * glm_vocab_l(c) as u64 * BF16);
    let amax = ac(b, "amax.part", dbatch as u64 * AMAX_BLOCKS as u64 * 8);
    // TP peer-mapped partials (§7a) — only under sharding; the host binds these into peer scratch at
    // offset 0 / slot_b so the row-parallel o_proj + MoE/dense down write peer-visible partials that
    // XReduce sums. zero_h is a persistent zero buffer used as the MoeCombine residual under TP (the
    // real residual xmid is added AFTER the all-reduce, so it is not summed N times).
    // og_tp is the o_proj partial on BOTH phases — prefill all-reduces a [T,hidden] partial through
    // XReduceTwoShot, so it is row-dimensioned. dg_tp only ever carries the
    // decode FFN partial (the prefill FFN has no kernel), so it stays one row.
    let og_tp = if tp > 1 {
        ac(b, "og_tp", rows * h as u64 * BF16)
    } else {
        TENSOR_NONE
    };
    // ROW-DIMENSIONED, like og_tp. The prefill FFN's `MoeCombinePf` writes a [T, hidden] partial
    // here (`i[2] = t`), so a one-row declaration under-declares it by `rows`x. It is bound as a
    // VIEW into the peer region, so the short length never faulted — it just described the wrong
    // buffer to everything that reads `bytes`.
    let dg_tp = if tp > 1 {
        ac(b, "dg_tp", rows * h as u64 * BF16)
    } else {
        TENSOR_NONE
    };
    let zero_h = if tp > 1 {
        b.tensor_init("act.zero_h", vec![0u8; rows as usize * h as usize * 2])
    } else {
        TENSOR_NONE
    };

    // --- DSA lightning indexer scratch (ctx>2048 only). qidx/kidx/widx are per-step; iscore/iidx/
    //     ighist/igctl are the score+select scratch (shared across layers — decode runs them
    //     sequentially); icos/isin are the identity-tail interleaved-RoPE tables. ighist/igctl are
    //     tensor_init'd to ZERO (the coop select requires them clean on entry and leaves them clean). ---
    let dsa = c.dsa(ctx);
    let (hi, di, itk) = (c.index_heads, c.index_dim, c.index_topk.min(ctx));
    // Anything the INDEXER needs regardless of which side (decode gate or sparse prefill)
    // consumes it: the wq_b/wk/k_norm/weights_proj weights, the per-layer key cache, and the
    // identity-tail RoPE tables.
    let idx_on = dsa || c.dsa_pf();
    let (qidx, kidx_raw, kidx_normed, widx, iscore, iidx, ighist, igctl) = if dsa {
        let [ct, st] = GenTensor::rope_idx_pair(ctx, dr, di, c.rope_theta());
        (
            ac(b, "qidx", (hi * di) as u64 * BF16),
            ac(b, "kidx_raw", di as u64 * BF16),
            ac(b, "kidx_normed", di as u64 * BF16),
            ac(b, "widx", hi as u64 * BF16),
            ac(b, "iscore", ctx as u64 * F32),
            ac(b, "iidx", itk as u64 * I32),
            b.tensor_init("act.ighist", vec![0u8; 7 * 256 * 4]),
            b.tensor_init("act.igctl", vec![0u8; 3 * 4]),
        )
    } else {
        (
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
        )
    };
    let (icos, isin) = if idx_on {
        let [ct, st] = GenTensor::rope_idx_pair(ctx, dr, di, c.rope_theta());
        (
            b.tensor_gen("in.icos", ct.byte_len(), ct),
            b.tensor_gen("in.isin", st.byte_len(), st),
        )
    } else {
        (TENSOR_NONE, TENSOR_NONE)
    };

    // --- DSA SPARSE PREFILL scratch. Sized for `rows` (the widest bucket) exactly as every other
    //     prefill activation is; all zero-init-free (each is fully written before it is read within
    //     one bucket's chain). `iuni`'s cap bounds a 64-query tile's union at min(64*top_k, ctx) —
    //     that bound is exact, not a heuristic, so the compaction can never truncate. ---
    let rows64 = rows as u64;
    let dsa_pf = c.dsa_pf() && rows as u32 >= 2048 && rows as u32 > c.index_topk;
    let (qidx_pf, kidx_pf, widx_pf, iscore_pf, iidx_pf, iuni, iumask) = if dsa_pf {
        let n_qt = rows64.div_ceil(GLM_DSA_PF_PACK as u64);
        let cap = glm_dsa_pf_cap(c, ctx) as u64;
        let hdr = (n_qt * 4).div_ceil(256) * 256;
        (
            ac(b, "qidx_pf", rows64 * hi as u64 * di as u64 * BF16),
            ac(b, "kidx_pf", rows64 * di as u64 * BF16),
            ac(b, "widx_pf", rows64 * hi as u64 * BF16),
            ac(b, "iscore_pf", rows64 * ctx as u64 * F32),
            ac(b, "iidx_pf", rows64 * itk as u64 * I32),
            ac(b, "iuni", hdr + n_qt * cap * 12),
            // SLICE-indexed inside op 119 (one scratch row per in-flight workgroup), so the
            // size is bound by the op's block count (<= n_cu), not the 8x-denser pack count.
            ac(b, "iumask", b.n_cu() as u64 * ctx as u64 * 8),
        )
    } else {
        (
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
        )
    };

    let lin_fp8 = glm_linear_fp8(enc);
    // The checkpoint's weight namespace, from the cfg — NOT the literal `model.` these closures
    // used to carry. `kv.*` above stays compiler-owned and is deliberately not prefixed.
    let pfx = c.prefix.as_str();
    // NO `rows == 1` REFUSAL HERE ANY MORE, and the reason is a kernel that now exists rather than
    // a relaxed rule. This used to assert decode-only under `lin_fp8`, because the four handles
    // below are re-declared at 1 B/elt and only the DECODE emitters routed them to a block-fp8
    // opcode — a stacked blob would have read fp8 bytes as bf16 with no fault. Both prefill
    // emitters now route them to `GemmFp8Blk` (107); see `glm_linear_fp8`'s header.
    let fp8kv = glm_fp8_kv();
    let mut ckv = Vec::new();
    let mut krot = Vec::new();
    let mut kv_scale = Vec::new();
    let mut kidx = Vec::new();
    let mut lw = Vec::new();
    for &l in layer_ids {
        // fp8 latent ([`glm_fp8_kv`]): 1 B/elt e4m3 plus one f32 scale per cached row, the K3
        // sizing exactly. bf16 keeps the historical 2 B/elt and no scale handle.
        ckv.push(b.tensor(
            &format!("kv.{l}.ckv"),
            dbatch as u64 * (ctx * dk) as u64 * if fp8kv { 1 } else { BF16 },
        ));
        krot.push(b.tensor(
            &format!("kv.{l}.krot"),
            dbatch as u64 * (ctx * dr) as u64 * BF16,
        ));
        kv_scale.push(if fp8kv {
            b.tensor(&format!("kv.{l}.scale"), dbatch as u64 * ctx as u64 * 4)
        } else {
            TENSOR_NONE
        });
        // per-'full'-layer indexer key cache [ctx][DI] (accumulates like ckv/krot); shared layers none.
        let full = idx_on && c.indexer_is_full(l);
        kidx.push(if full {
            b.tensor(
                &format!("kv.{l}.kidx"),
                dbatch as u64 * (ctx * di) as u64 * BF16,
            )
        } else {
            TENSOR_NONE
        });
        let t = |b: &mut Builder, s: &str, sz: u64| b.tensor(&format!("{pfx}layers.{l}.{s}"), sz);
        // A MATMUL weight [N,K], sized for the packet's encoding: bf16 = 2 B/elt, MXFP4 = packed
        // fp4 at half a byte. `mxs` declares the matching E8M0 scale rows (one byte per 32 K), or
        // TENSOR_NONE off the MXFP4 arm. Keeping these two together is the point — a packed weight
        // whose scale handle was forgotten is a tensor the kernel reads as NULL.
        let tw = |b: &mut Builder, s: &str, n: u64, k: u64| match enc {
            MoeEnc::Mxfp4 => b.tensor(&format!("{pfx}layers.{l}.{s}"), n * k / 2),
            _ => b.tensor(&format!("{pfx}layers.{l}.{s}"), n * k * BF16),
        };
        let mxs = |b: &mut Builder, s: &str, n: u64, k: u64| {
            if enc == MoeEnc::Mxfp4 {
                b.tensor(
                    &format!("{pfx}layers.{l}.{s}_scale"),
                    n * k.div_ceil(MX_BLOCK as u64),
                )
            } else {
                TENSOR_NONE
            }
        };
        let dense = c.is_dense(l);
        // GLM_LINEAR_FP8: `o_proj` / shared-expert weights come from the checkpoint's block-fp8
        // bytes (1 B/elt) under a `.weight_fp8` name — the bf16 `.weight` still exists in the same
        // weight dir, so the two must not share a name — paired with the checkpoint's own
        // `[N/128][K/128]` f32 `weight_scale_inv` grid. Both names keep the projection substring
        // the host's column/row shard predicates match on, so the TP slicing of the weight and of
        // its scale grid needs no host change.
        let q8 = |b: &mut Builder, s: &str, n: u64, k: u64| {
            b.tensor(&format!("{pfx}layers.{l}.{s}.weight_fp8"), n * k)
        };
        let q8s = |b: &mut Builder, s: &str, n: u64, k: u64| {
            b.tensor(
                &format!("{pfx}layers.{l}.{s}.weight_scale_inv"),
                n.div_ceil(128) * k.div_ceil(128) * F32,
            )
        };
        // The 256 per-expert block-fp8 weights + scale grids are NOT declared as .pkt tensors: the
        // loader binds them by name-pattern (model.layers.{l}.mlp.experts.{e}.{proj}.weight[_scale_inv])
        // straight from the prepped dir, packs them, and fills expert_weight_table/expert_scale_table.
        // Declaring 75*256*6 handles would bloat the tensor table for zero emit benefit (the MoE ops
        // index the tables, never the individual expert handles). ib/hb below size the scale grids.
        let _ = (ib, db);
        // Weight tensors carry this rank's SHARDED byte size (the host binds the matching slice):
        //   column-parallel (q/v absorb, q_rope, shared+dense+expert gate/up) -> nh_l/imoe_l/di_l rows;
        //   row-parallel (o_proj, shared+dense down) -> nh_l/imoe_l/di_l input lanes. tp==1 => full.
        //   Replicated (norms, q_a_proj, kv_a_latent, k_rope, router, bias) keep full dims.
        lw.push(GlmLW {
            qad_s: mxs(b, "self_attn.q_a_proj.weight", ql as u64, h as u64),
            wqa_s: mxs(
                b,
                "self_attn.derived.q_absorb.weight",
                (nh_l * dk) as u64,
                ql as u64,
            ),
            wqr_s: mxs(
                b,
                "self_attn.derived.q_rope.weight",
                (nh_l * dr) as u64,
                ql as u64,
            ),
            ckvd_s: mxs(
                b,
                "self_attn.derived.kv_a_latent.weight",
                dk as u64,
                h as u64,
            ),
            krotd_s: mxs(b, "self_attn.derived.k_rope.weight", dr as u64, h as u64),
            wo_s: if lin_fp8 {
                q8s(b, "self_attn.o_proj", h as u64, (nh_l * vd) as u64)
            } else {
                mxs(b, "self_attn.o_proj.weight", h as u64, (nh_l * vd) as u64)
            },
            wr_s: if dense {
                TENSOR_NONE
            } else {
                mxs(b, "mlp.gate.weight", e as u64, h as u64)
            },
            shg_s: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8s(b, "mlp.shared_experts.gate_proj", imoe_l as u64, h as u64)
            } else {
                mxs(
                    b,
                    "mlp.shared_experts.gate_proj.weight",
                    imoe_l as u64,
                    h as u64,
                )
            },
            shu_s: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8s(b, "mlp.shared_experts.up_proj", imoe_l as u64, h as u64)
            } else {
                mxs(
                    b,
                    "mlp.shared_experts.up_proj.weight",
                    imoe_l as u64,
                    h as u64,
                )
            },
            shd_s: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8s(b, "mlp.shared_experts.down_proj", h as u64, imoe_l as u64)
            } else {
                mxs(
                    b,
                    "mlp.shared_experts.down_proj.weight",
                    h as u64,
                    imoe_l as u64,
                )
            },
            gin: t(b, "input_layernorm.weight", h as u64 * BF16),
            qad: tw(b, "self_attn.q_a_proj.weight", ql as u64, h as u64),
            gqa: t(b, "self_attn.q_a_layernorm.weight", ql as u64 * BF16),
            wqa: tw(
                b,
                "self_attn.derived.q_absorb.weight",
                (nh_l * dk) as u64,
                ql as u64,
            ),
            wqr: tw(
                b,
                "self_attn.derived.q_rope.weight",
                (nh_l * dr) as u64,
                ql as u64,
            ),
            ckvd: tw(
                b,
                "self_attn.derived.kv_a_latent.weight",
                dk as u64,
                h as u64,
            ),
            gkva: t(b, "self_attn.kv_a_layernorm.weight", dk as u64 * BF16),
            krotd: tw(b, "self_attn.derived.k_rope.weight", dr as u64, h as u64),
            // W_uv stays bf16 under EVERY encoding: MLA_MERGE_FOLD / O_UV_FOLD take it as
            // `const bf16*` with no encoding parameter, so there is nowhere to put an fp4 form.
            // It is DERIVED by host weight-prep (a fold of kv_b_proj), not a checkpoint tensor, so
            // a bf16 copy exists whatever the checkpoint stores — unlike the expert weights, where
            // fp4 bytes read as bf16 would be noise. Recorded as an explicit exception rather than
            // hidden; see `mla_mxfp4_wuv_is_the_declared_exception`.
            wuv: t(
                b,
                "self_attn.derived.v_absorb.weight",
                (nh_l * dk * vd) as u64 * BF16,
            ),
            wo: if lin_fp8 {
                q8(b, "self_attn.o_proj", h as u64, (nh_l * vd) as u64)
            } else {
                tw(b, "self_attn.o_proj.weight", h as u64, (nh_l * vd) as u64)
            },
            // bf16 under every encoding, like W_uv (its factor): the fused arm is bf16-GEMM
            // only, and the weight is DERIVED (prep multiplies the bf16 pair), never a
            // checkpoint quantization. Declared only when the arm emits — TENSOR_NONE keeps
            // every non-ofold blob byte-identical.
            wofold: if glm_ofold(enc) {
                t(
                    b,
                    "self_attn.derived.o_fold.weight",
                    h as u64 * (nh_l * dk) as u64 * BF16,
                )
            } else {
                TENSOR_NONE
            },
            gpost: t(b, "post_attention_layernorm.weight", h as u64 * BF16),
            wr: if dense {
                TENSOR_NONE
            } else {
                tw(b, "mlp.gate.weight", e as u64, h as u64)
            },
            bias: if dense {
                TENSOR_NONE
            } else {
                t(b, "mlp.gate.e_score_correction_bias", e as u64 * F32)
            },
            shg: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8(b, "mlp.shared_experts.gate_proj", imoe_l as u64, h as u64)
            } else {
                tw(
                    b,
                    "mlp.shared_experts.gate_proj.weight",
                    imoe_l as u64,
                    h as u64,
                )
            },
            shu: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8(b, "mlp.shared_experts.up_proj", imoe_l as u64, h as u64)
            } else {
                tw(
                    b,
                    "mlp.shared_experts.up_proj.weight",
                    imoe_l as u64,
                    h as u64,
                )
            },
            shd: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8(b, "mlp.shared_experts.down_proj", h as u64, imoe_l as u64)
            } else {
                tw(
                    b,
                    "mlp.shared_experts.down_proj.weight",
                    h as u64,
                    imoe_l as u64,
                )
            },
            ewt: if dense {
                TENSOR_NONE
            } else {
                t(b, "mlp.expert_weight_table", (e * 3) as u64 * 8)
            },
            est: if dense {
                TENSOR_NONE
            } else {
                t(b, "mlp.expert_scale_table", (e * 3) as u64 * 8)
            },
            ewt_pf: if dense || !emit_config::active().moe_pf_shuf {
                TENSOR_NONE
            } else {
                t(b, "mlp.expert_weight_table_pf", (e * 3) as u64 * 8)
            },
            // Dense-FFN weights. Block-fp8 keeps the checkpoint's own byte layout and its
            // [N/128][K/128] f32 `weight_scale_inv` grid; MXFP4 halves the weight and swaps the
            // grid for E8M0 rows. `mxd`/`mxd_s` pick per encoding so the two never disagree.
            dgate: if dense {
                match enc {
                    MoeEnc::Mxfp4 => t(b, "mlp.gate_proj.weight", (di_l * h) as u64 / 2),
                    _ => t(b, "mlp.gate_proj.weight", (di_l * h) as u64),
                }
            } else {
                TENSOR_NONE
            },
            dgate_s: if dense {
                match enc {
                    MoeEnc::Mxfp4 => mxs(b, "mlp.gate_proj.weight", di_l as u64, h as u64),
                    MoeEnc::Fp8Blk => t(
                        b,
                        "mlp.gate_proj.weight_scale_inv",
                        (db_l * hb) as u64 * F32,
                    ),
                    MoeEnc::Bf16 => TENSOR_NONE,
                }
            } else {
                TENSOR_NONE
            },
            dup: if dense {
                match enc {
                    MoeEnc::Mxfp4 => t(b, "mlp.up_proj.weight", (di_l * h) as u64 / 2),
                    _ => t(b, "mlp.up_proj.weight", (di_l * h) as u64),
                }
            } else {
                TENSOR_NONE
            },
            dup_s: if dense {
                match enc {
                    MoeEnc::Mxfp4 => mxs(b, "mlp.up_proj.weight", di_l as u64, h as u64),
                    MoeEnc::Fp8Blk => {
                        t(b, "mlp.up_proj.weight_scale_inv", (db_l * hb) as u64 * F32)
                    }
                    MoeEnc::Bf16 => TENSOR_NONE,
                }
            } else {
                TENSOR_NONE
            },
            ddown: if dense {
                match enc {
                    MoeEnc::Mxfp4 => t(b, "mlp.down_proj.weight", (h * di_l) as u64 / 2),
                    _ => t(b, "mlp.down_proj.weight", (h * di_l) as u64),
                }
            } else {
                TENSOR_NONE
            },
            ddown_s: if dense {
                match enc {
                    MoeEnc::Mxfp4 => mxs(b, "mlp.down_proj.weight", h as u64, di_l as u64),
                    MoeEnc::Fp8Blk => t(
                        b,
                        "mlp.down_proj.weight_scale_inv",
                        (hb * db_l) as u64 * F32,
                    ),
                    MoeEnc::Bf16 => TENSOR_NONE,
                }
            } else {
                TENSOR_NONE
            },
            // Only a PREFILL emit (`rows > 1`) runs the dense FFN on the grouped arms, so only a
            // prefill emit declares the tables. A decode-only blob declares neither and is
            // therefore byte-identical to one emitted before this path existed.
            dwt: if dense && rows > 1 {
                t(b, "mlp.dense_weight_table", 3 * 8)
            } else {
                TENSOR_NONE
            },
            dst: if dense && rows > 1 && enc == MoeEnc::Fp8Blk {
                t(b, "mlp.dense_scale_table", 3 * 8)
            } else {
                TENSOR_NONE
            },
            // DSA indexer weights (fp8 wq_b/wk copied VERBATIM for GemvFp8Blk + f32 [128,128] scale
            // grids; k_norm weight/bias + weights_proj bf16). REPLICATED across TP ranks (the indexer
            // is tiny and its idx is head-shared). Only bound on 'full' layers with the DSA gate on.
            iwqb: if full {
                t(b, "self_attn.indexer.wq_b.weight", (hi * di * ql) as u64)
            } else {
                TENSOR_NONE
            },
            iwqb_s: if full {
                t(
                    b,
                    "self_attn.indexer.wq_b.weight_scale_inv",
                    ((hi * di).div_ceil(128) * ql.div_ceil(128)) as u64 * F32,
                )
            } else {
                TENSOR_NONE
            },
            iwk: if full {
                t(b, "self_attn.indexer.wk.weight", (di * h) as u64)
            } else {
                TENSOR_NONE
            },
            iwk_s: if full {
                t(
                    b,
                    "self_attn.indexer.wk.weight_scale_inv",
                    (di.div_ceil(128) * hb) as u64 * F32,
                )
            } else {
                TENSOR_NONE
            },
            iknw: if full {
                t(b, "self_attn.indexer.k_norm.weight", di as u64 * BF16)
            } else {
                TENSOR_NONE
            },
            iknb: if full {
                t(b, "self_attn.indexer.k_norm.bias", di as u64 * BF16)
            } else {
                TENSOR_NONE
            },
            iwp: if full {
                t(
                    b,
                    "self_attn.indexer.weights_proj.weight",
                    (hi * h) as u64 * BF16,
                )
            } else {
                TENSOR_NONE
            },
        });
    }

    GlmTn {
        ids,
        pos,
        kvlen,
        cos,
        sin,
        emb,
        fin,
        head,
        x,
        xn,
        qlr,
        qlat,
        ckvraw,
        qa,
        qrr,
        qr,
        krr,
        opart,
        mlpart,
        olat,
        oat,
        attn,
        xmid,
        xn2,
        xn2q,
        xn2s,
        tab,
        rlogit,
        shfu,
        shfu_up,
        shared,
        fu,
        dfu,
        part,
        xnext,
        logits,
        amax,
        og_tp,
        dg_tp,
        zero_h,
        // Slot A is [0, rows*h*2); slot B starts where it ends. `rows == 1` (a decode-only emit)
        // gives `h*2`, the value every shipped decode blob already carries.
        slot_b: rows as u32 * h * BF16 as u32,
        meta,
        row_token,
        row_partidx,
        row_gate,
        fu_g,
        fu_scale,
        qidx_pf,
        kidx_pf,
        widx_pf,
        iscore_pf,
        iidx_pf,
        iuni,
        iumask,
        qidx,
        kidx_raw,
        kidx_normed,
        widx,
        iscore,
        iidx,
        ighist,
        igctl,
        icos,
        isin,
        ckv,
        krot,
        kv_scale,
        kidx,
        lw,
    }
}

/// LAYER-SEAM FOLD (`PLOW_GLM_FUSE_SEAM=1`, opt-in): the MoE/dense tail's `Residual` and the NEXT
/// layer's `input_layernorm` `RmsNorm` are the SAME `AddNorm` pair the attention seam already fuses
/// under `PLOW_GLM_FUSE_B1` — `x_out = xmid + ffn` immediately followed by `xn = rmsnorm(x_out)`,
/// with nothing in between and no other consumer. One `AddNorm` packet writes both.
///
/// It deletes ONE single-workgroup packet per layer. That shape is the whole point: on the traced
/// decode step (perf-data/plow-gfx942/glm52-decode-packet-folds.md) the tail `Residual` and the
/// next layer's `RmsNorm` are b=1 packets of 4.1 and 5.8 us sitting back-to-back on the serial
/// chain, and the census's standing rule is that a 1-workgroup packet is a pure CAUSE of gate
/// stall — it waits for nothing (0.001 ms/CU blocked) and everything waits for it.
///
/// It needs NO new plumbing, which is why it is cheap: the next layer's gamma is `n.lw[slot+1].gin`
/// (the blocks are emitted in `slot` order over the same table), and "my input is already normed"
/// is exactly `slot > 0`. The LAST layer keeps the plain `Residual` — `emit_glm_tail`'s final
/// `model.norm` is a different gamma and a different consumer — and so does any single-block
/// bring-up program, where `n.lw.len() == 1`.
///
/// NOT byte-identical, same reason `PLOW_GLM_FUSE_B1` is not: `d_add_norm` reduces over the
/// UN-ROUNDED `a + b` while the split path norms the bf16-rounded `x_out`. The residual stream
/// itself is unchanged (still the bf16 `x_out`), so the drift is confined to one norm per layer —
/// but it is a logit-gate change, not a `cmp` change, and it ships opt-in behind the HF-coherence
/// gate like B1 did.
/// TP-ONLY, and that is structural rather than a scope cut. At `tp == 1` there IS no tail
/// `Residual` to fold: the `MoeCombine`/dense-down packet writes `x_out = xmid + ffn` itself
/// (the collective is what forces the partial+Residual split in the first place), so the seam is
/// a lone `RmsNorm` and folding it would need a norm epilogue inside the combine — a different
/// change. `PLOW_NO_XREDUCE` takes the same tp==1-shaped branch and is excluded for the same
/// reason. Both ends of the seam ask this one predicate, so producer and consumer cannot disagree.
fn glm_fuse_seam(tp: u32) -> bool {
    emit_config::active().glm_fuse_seam && tp > 1 && !emit_config::active().no_xreduce
}

/// The Q-NORM FOLD gate (`PLOW_GLM_FUSE_QNORM`): may `q_a_layernorm` be computed inside fusion
/// G's `GemvQkv` staging instead of as its own one-workgroup packet?
///
/// Every condition here is a REFUSAL, not a silent downgrade, except the knob itself — a fold
/// that quietly did not happen would show up only as a missing win, and a fold that quietly DID
/// happen on a shape the kernel cannot take would read an LDS arena it never filled.
///
/// * `fuse_g` — the fold has exactly one consumer only because A/G fusion made it one. The split
///   form has two (`q_absorb` and `q_rope` as separate GEMVs), and op_gemm.h's `norm == 2` note
///   is explicit that a fold across N consumers costs (N-1) extra reductions. `fuse_g` is
///   hardcoded on today; this states the dependency rather than assuming it.
/// * `!mx` — under MXFP4 fusion G is `GemvQkvMxfp4` (op 114), whose i5/i6/i7 already carry the
///   three E8M0 scale rows and whose body has no fold arm. Refuse rather than emit a packet
///   whose reading depends on which opcode the object happens to implement.
/// * `!dsa` — an armed DSA gate gives `n.qlat` a SECOND consumer, the lightning indexer's
///   `q_idx` projection (`gemv_blk(.., n.qidx, n.qlat, w.iwqb, ..)`). Folding the norm into
///   fusion G alone would leave that GEMV reading a tensor nothing writes.
/// * the shape — mirrors `d_gemv_t`'s `fold` test and `d_rmsnorm`'s `fits` test, the same pair
///   [`crate::k3::fuse_norm_gemv`] checks. The arena comes from [`crate::gm_lds_halves`], i.e.
///   the arena the DECODE OBJECT FOR THIS PART actually has — 15,360 halves on gfx942's shipped
///   `PLOW_OCC4` profile, 73,728 on CDNA4. It was written against the flat CDNA4 constant, which
///   made the bound optimistic by 4.8× on gfx942 and left the kernel's trap as the only thing
///   standing between a too-large `ql` and a silent LDS overrun. GLM's `ql` is 1536 and clears
///   either, so this is exact-instead-of-lucky rather than a behaviour change.
fn glm_fuse_qnorm(fuse_g: bool, mx: bool, dsa: bool, ql: u32) -> bool {
    if !emit_config::active().glm_fuse_qnorm {
        return false;
    }
    assert!(
        fuse_g && !mx,
        "PLOW_GLM_FUSE_QNORM=1 needs the A/G-fused bf16 `GemvQkv` (op 22): fuse_g={fuse_g}          mxfp4={mx}. The split form gives q_a_layernorm two consumers and op 114 has no fold arm."
    );
    assert!(
        !dsa,
        "PLOW_GLM_FUSE_QNORM=1 with an armed DSA gate: the lightning indexer's q_idx projection          is a SECOND consumer of n.qlat, so deleting the q_a_layernorm packet would leave it          reading a tensor nothing writes. Emit with PLOW_GLM_DSA=0 (the shipped best config) or          without this knob."
    );
    assert!(
        ql as u64 + crate::GV_NORM_SCRATCH <= crate::gm_lds_halves()
            && ql <= crate::RN_REG * crate::PLOW_THREADS
            && ql % 8 == 0,
        "PLOW_GLM_FUSE_QNORM=1 at q_lora_rank={ql}: the fold needs the LDS-staged arm          (ql + GV_NORM_SCRATCH <= GM_LDS_HALVES) and d_rmsnorm's register path          (ql <= RN_REG*PLOW_THREADS, ql % 8 == 0)."
    );
    true
}

/// The gamma the layer-seam fold's `AddNorm` normalizes with: the NEXT layer's `input_layernorm`,
/// or `None` when there is no next layer in this program (last layer, or a single-block bring-up).
fn seam_next_gin(n: &GlmTn, slot: usize, tp: u32) -> Option<u32> {
    if !glm_fuse_seam(tp) {
        return None;
    }
    n.lw.get(slot + 1).map(|w| w.gin)
}

/// Workgroup list for the ELEMENTWISE spine ops (`Residual`).
///
/// They ship on `vec![0u32]` — ONE workgroup of 256 — and that is not a correctness
/// requirement: `d_residual` is a pure elementwise `out = (a+b)*scale` strided by
/// `nblk * PLOW_THREADS * 8`, so any width covers every element exactly once and the output
/// is BIT-IDENTICAL. It is, per the 2026-07-28 stall census, the second-largest single cause
/// of gate stall in GLM decode: 156 packets/token, 3.7 us each, during which the other 255
/// CUs spin.
///
/// `GLM_SPINE_CUS=k` is a CEILING INSTRUMENT, not a default. knob-contract §6b-i is the reason
/// it is opt-in: widening a producer takes its consumer from waiting on a max over 1 straggler
/// to a max over k, and that has already cost more than it saved once (`flash_merge` 32->256,
/// +0.555 ms). Measure before changing the default.
fn spine_cus(n_cu: u32) -> Vec<u32> {
    match emit_config::active()
        .glm_spine_cus
        .as_deref()
        .and_then(|v| v.parse::<u32>().ok())
    {
        Some(k) if k > 1 => (0..k.min(n_cu)).collect(),
        _ => vec![0u32],
    }
}

/// The workgroups a `HeadNormRope` packet can actually use, as a slice of `cus` starting at
/// `start`.
///
/// `d_headnorm_rope` gives one WAVE per work item —
/// `for (w = slice*PLOW_WAVES + wave_in_blk; w < ntok*nhead; w += nblk*PLOW_WAVES)` — so the
/// packet saturates at `ceil(ntok*nhead / PLOW_WAVES)` workgroups. Every workgroup past that
/// runs the loop ZERO times: it polls the packet's arrival counter, takes the acquire fence,
/// and exits. GLM decode hands all 256 to a q-side rope that needs **2** (`nh_l` = 16 heads over
/// 8 waves) and a k-side rope that needs **1** (the shared single-head rope).
///
/// This is knob-contract §4's recurring bug shape and the dense emitter already knows the rule
/// (`lib.rs` `headnorm_items`/`WAVES`, and the load-bearing `8` at `lib.rs:2504`); the MLA
/// emitter never got it, exactly as it never got `emit_xreduce`'s sizing.
///
/// MEASURED, GLM-5.2 TP4 decode, ctx 1024, one lease, control interleaved at positions 1/4/6
/// (26.768 / 26.799 / 26.650, mean 26.739, sd 0.076): **−1.148 ms/token (−4.3%) on its own**,
/// −1.674 together with `elem_cus`, and **token-identical over 24 generated ids with 0
/// cross-rank disagreements** on every arm. On the shipping blob's own trace the 254 empty
/// workgroups burn 0.968 ms/CU/token — the same estimator that said 1.779 for the collective
/// before `xrfit` took 1.82 ms out. (`perf-data/glm52-narrow-op-sizing.md`)
///
/// `start` is load-bearing. The q and k ropes are CONCURRENT siblings — both gated only on the
/// QKV GEMV, and all 78 pairs overlap in the shipping trace. Narrowing both onto `cus[..need]`
/// would land them on the SAME workgroups, and the interpreter walks the packet stream per
/// workgroup, so they would run one after the other. `glm_glu_halves` exists for precisely this
/// reason; disjoint slices keep the pair concurrent.
///
/// Pure narrowing of whatever the caller allowed, and BIT-IDENTICAL: slices `0..need` own
/// exactly the work items they own today, only the empty slices go away. Prefill needs no arm —
/// its `ntok*nhead` exceeds `PLOW_WAVES * cus.len()`, so `need` saturates at `cus.len()`,
/// `start` clamps to 0, and the packet is unchanged.
fn rope_cus(cus: &[u32], start: usize, ntok: u32, nhead: u32) -> Vec<u32> {
    let items = ntok as u64 * nhead as u64;
    let need = (items.div_ceil(WG_WAVES as u64).max(1) as usize).min(cus.len());
    let start = start.min(cus.len() - need);
    cus[start..start + need].to_vec()
}

/// The workgroups a flat elementwise `[n]` packet can actually use.
///
/// `d_moe_combine` gives one element per thread (`gid = slice*PLOW_THREADS + threadIdx.x`,
/// stepped by `nblk*PLOW_THREADS`), so it saturates at `ceil(n / PLOW_THREADS)` — **12** for
/// GLM's hidden 6144, against the 256 the MLA emitter hands it. Same rule and same constant as
/// `emit_xreduce`'s `ceil(xr_elems/512)`, and the same reason the dense emitter's
/// *"Elementwise ops sized to their ACTUAL work, not handed the whole machine"* comment exists.
///
/// No `start`: the combine is the MoE block's join, gated on every expert plus the shared
/// branch, so it has no concurrent sibling to be disjoint from.
///
/// MEASURED on the same lease as `rope_cus`: **−0.506 ms/token (−1.9%) on its own**, and the two
/// compose additively (−1.148 + −0.506 = −1.654 predicted, **−1.674 measured**, inside the
/// control's own 0.076 sd). Token-identical, 0 cross-rank disagreements.
///
/// **This is an EMIT-TIME change: every already-built `.pkt` keeps the old width until it is
/// re-emitted** — the same caveat `xrfit` carries.
fn elem_cus(cus: &[u32], n: u32) -> Vec<u32> {
    let need = (n.div_ceil(WG_THREADS).max(1) as usize).min(cus.len());
    cus[..need].to_vec()
}

/// `PLOW_GLM_WGFIT=0` restores the un-narrowed widths for the three sizing rules below
/// (`mla_fold_cus`, `flash_mla_cus`, `blocked_gemv_cus`), so the control arm of an A/B comes out
/// of the SAME `plowc` binary as the fixed one. Default ON: all three are pure narrowings of a
/// work-item map the kernel already owns, so the emitted arithmetic is unchanged and the only
/// thing that goes away is empty workgroups. Same shape as `PLOW_GLM_FUSE_A`, and the same reason
/// it exists.
///
/// Default ON. `PLOW_GLM_WGFIT=0` disables the narrowing for the A/B control arm.
fn wgfit() -> bool {
    emit_config::active().glm_wgfit
}

/// V-tile `exec_mla_merge_fold` (runtime/amd/interp.hip) picks for a fold packet of `bh` =
/// `n_batch * n_head_local` rows at width `v`, dispatched on `nblk` workgroups.
///
/// Mirrored here, and NOT approximated, because the tile decides the packet's work-item count AND
/// its fold map — `NV`/`LS`/`BL` are all functions of VT, so two VTs reassociate the `l` sum
/// differently and do not agree bit for bit. A width that flipped the branch would therefore be a
/// numerics change wearing a dispatch change's clothes.
const MLA_FOLD_VT: u32 = 32; // matches the op_attention.h PLOW_MLA_FOLD_VT default
fn mla_fold_vt(bh: u32, nblk: u32, v: u32) -> u32 {
    if bh != 0 && bh * 8 <= nblk {
        MLA_FOLD_VT
    } else if (128..256).contains(&v) {
        128
    } else {
        256
    }
}

/// The workgroups a `MlaMergeFold` packet can actually use.
///
/// `d_mla_merge_fold` grid-strides `for (w = slice; w < n_work; w += nblk)` over
/// `n_work = n_batch * n_head * ceil(v / VT)` — one (row, V-tile) item per step — so the packet
/// saturates at `n_work` workgroups, and every one past that polls the arrival counter, takes the
/// system-scope acquire fence and exits without touching a float. GLM-5.2 at TP4 is
/// `1 * 16 * ceil(256/32)` = **128**, against the 256 the MLA emitter hands it.
///
/// MEASURED on the shipping trace (`glm52_skew/tr/xrfit`, the post-`xrfit` control): **exactly
/// 128.0 empty workgroups per packet** on all 78 packets — bodies split cleanly at ~2700 ticks
/// (slices 0..127) against ~550 (slices 128..255) — burning **0.220 ms/CU/token**. That is the
/// same estimator that read 0.968 for `rope_cus` (measured −1.148 ms on the token) and 1.779 for
/// the collective before `xrfit` (measured −1.820), so it is calibrated near 1:1.
///
/// After the narrowing `n_work == nblk`: workgroup `slice` owns work item `slice` and nothing
/// else, exactly as it does today, so the packet is BIT-IDENTICAL — and no straggler max grows,
/// because knob-contract §6b-i is about WIDENING a producer and the workgroups removed here did
/// no work to be waited on.
///
/// The refusal at the end is load-bearing. At `v = 128` (Kimi-K3) with `bh*8 <= nblk`, VT is 32
/// and `need = bh*4`, which no longer satisfies `bh*8 <= need` — the interpreter would re-pick
/// VT=128, a different fold map and a different sum order. Narrowing is legal only when it leaves
/// the branch where it found it.
fn mla_fold_cus(cus: &[u32], bh: u32, v: u32) -> Vec<u32> {
    let nblk = cus.len() as u32;
    if !wgfit() || nblk == 0 || bh == 0 || v == 0 {
        return cus.to_vec();
    }
    let vt = mla_fold_vt(bh, nblk, v);
    let need = (bh * v.div_ceil(vt)).clamp(1, nblk);
    if mla_fold_vt(bh, need, v) != vt {
        return cus.to_vec();
    }
    cus[..need as usize].to_vec()
}

/// The workgroups a `FlashMlaDecode` / `FlashGatherDecode` packet can actually use.
///
/// `d_flash_mla_decode` grid-strides `for (w = slice; w < n_work; w += nblk)` over
/// `n_work = n_batch * n_tok * (n_head / GF) * nsplit`, so the packet saturates there.
///
/// `GF` is read LITERALLY from `i[7]`, because the interpreter now dispatches every value it can
/// carry. It did not always: `exec_flash_mla_decode` used to instantiate `GF ∈ {2, 4}` only, so an
/// `i[7]` of 8 ran the GF=4 body and this helper had to MIRROR THE DISPATCH rather than the field
/// — reading 8 literally would have halved the width and DROPPED WORK. With the GF=8 arm in place
/// the mirror is gone and the field is the truth.
///
/// At GF=4 the narrowing is INERT, and that is not luck: `glm_nsplit`'s chip-fill cap is
/// `ceil(n_cu / (nh_l / GLM_MLA_GF))` with the same `GLM_MLA_GF = 4`, so the two cancel and
/// `n_work` lands exactly on `n_cu`. It does not cancel at either end:
///
/// - **GF=2** (max_ctx <= 4096): `n_grp` doubles to `nh_l/2` while `nsplit` stays capped for GF=4,
///   so GLM-5.2 TP4 at max_ctx 1024 is `(16/2) * 16 = 128` work items on 256 workgroups. Short-ctx
///   GLM blobs — which the ctx sweeps emit — carried the same 2x over-dispatch `MlaMergeFold` had.
/// - **GF=8** (max_ctx > 4096): `n_grp` HALVES to `nh_l/8`, so TP4 at ctx 32768 is `2 * 64 = 128`
///   work items where GF=4 had 256. Narrowing to 128 stops launching 128 workgroups that grid-
///   stride straight past `n_work` and exit — the §6c/L6 shape, pure narrowing, bit-identical —
///   but it does NOT recover the parallelism: at GF=8 half the chip has no flash work to do
///   unless `PLOW_GLM_NS` doubles nsplit. That is the whole GF=8 trade and it is why the arm has
///   to be measured end-to-end rather than argued from latent bytes.
fn flash_mla_cus(
    cus: &[u32],
    n_batch: u32,
    n_tok: u32,
    nh_l: u32,
    gf_field: u32,
    nsplit: u32,
) -> Vec<u32> {
    let nblk = cus.len() as u32;
    if !wgfit() || nblk == 0 {
        return cus.to_vec();
    }
    let gf = if gf_field == 0 { GLM_MLA_GF } else { gf_field };
    // `n_head / GF`, integer, NO clamp — the kernel does not clamp either, and a shape where this
    // is 0 runs no work at all today. Narrowing must not paper over that; hand the list back.
    // (`glm_gf` clamps GF to `nh_l` so the emitter never produces that shape; this stays defensive
    // because `flash_mla_cus` is also reachable from tests with a hand-written `gf_field`.)
    let n_work = n_batch * n_tok * (nh_l / gf) * nsplit;
    if n_work == 0 {
        return cus.to_vec();
    }
    cus[..n_work.min(nblk) as usize].to_vec()
}

/// The workgroups a GV_BLOCKED gemv-family packet can use for `n` output columns.
///
/// `gemv_rows` / `gemv_qkv_rows` / `gemv_glu_rows` all own columns in CONTIGUOUS runs (op_gemm.h,
/// `GV_BLOCKED=1`, the default and the form `PLOW_FINE`'s dependency map assumes): workgroup
/// `slice` takes `[slice*per, slice*per + per)` with `per = ceil(n / nblk)`. When `n` is not a
/// multiple of `nblk` that CEILING leaves a tail of workgroups whose `n0` is already past `n` —
/// GLM's fused q_a|kv_a|k_rope packet is `n = 2624` over 256 workgroups, `per = 11`, so slices
/// 239..255 own **nothing**. Measured on the same trace: 17 empty workgroups on each of the 78
/// fusion-A packets, ~950 ticks each (they still stage `x` into LDS before finding no columns),
/// **0.049 ms/CU/token**.
///
/// `need = ceil(n / per)` is a FIXED POINT of the kernel's own arithmetic —
/// `ceil(n / ceil(n / per)) == per` — so `per`, and therefore every surviving workgroup's
/// `[n0, n1)`, is unchanged and the packet is bit-identical. Under `GV_BLOCKED=0` the interleaved
/// map covers every column at any `nblk`, so the narrowing stays CORRECT there; it is simply not
/// the shape it was derived for.
pub(crate) fn blocked_gemv_cus(cus: &[u32], n: u32) -> Vec<u32> {
    let mut nblk = cus.len() as u32;
    if !wgfit() || nblk == 0 || n == 0 {
        return cus.to_vec();
    }
    // PLOW_GLM_GEMV_WG caps the dispatch width of every blocked GEMV — a
    // rows-per-wave probe (gemv-mlp-and-tensile.md: GEMV efficiency tracks work
    // per wave; at 304 wgs the narrow fusions run ~1 row/wave). Unset ⇒
    // byte-identical.
    if let Some(cap) = emit_config::active().glm_gemv_wg {
        nblk = nblk.min(cap.max(1));
    }
    let per = n.div_ceil(nblk);
    let need = n.div_ceil(per).clamp(1, nblk);
    cus[..need as usize].to_vec()
}

/// Apply an opt-in shape-keyed cap, then reuse the standard fixed-point mapping.
pub(crate) fn blocked_gemv_cus_tuned(cus: &[u32], n: u32, k: u32) -> Vec<u32> {
    let Some(cap) = emit_config::active().gemv_wg_for(n, k) else {
        return blocked_gemv_cus(cus, n);
    };
    if !wgfit() || cus.is_empty() || n == 0 {
        return cus.to_vec();
    }
    let mut scoped = cus.to_vec();
    scoped.truncate(cap.min(scoped.len() as u32).max(1) as usize);
    let per = n.div_ceil(scoped.len() as u32);
    let need = n.div_ceil(per).clamp(1, scoped.len() as u32);
    scoped.truncate(need as usize);
    scoped
}

/// Emit the shared MLA attention sub-block (input norm -> q/kv down + absorbed folds -> dynamic
/// interleaved RoPE on the 64 rope dims -> FLASH_MLA_DECODE -> merge -> O_UV_FOLD -> o_proj ->
/// residual -> post-attention norm). Writes `n.xn2` (the FFN input) and returns the post-attn-norm
/// completion dep. IDENTICAL for the dense (0-2) and MoE (3-77) layers, so both blocks call it.
///
/// `rows` is the DECODE BATCH — sequences, one new token each, not a token axis. At `rows == 1`
/// every packet below is byte-identical to before the parameter existed. At `rows > 1`:
///   * every projection runs at `M = rows` (all in `GEMV_BUCKET_OPS`, object `PLOW_GEMV_MM` must
///     cover it — `check_gemv_capacity` refuses the mismatch at load);
///   * the two KV writers switch to the batch-major ring form (`i[6] = n_batch_kv`, `j[0] = ctx`):
///     row `t` writes ITS OWN slot's ring at `pos[t]`, which is what frees the host from patching
///     `i[2]`/`i[3]` per step (`patch_kvrow` runs only at batch 1);
///   * the flash reads per-row `kv_len` (ragged, proven by `probes/raggedkv_*`).
#[allow(clippy::too_many_arguments)]
fn emit_glm_mla(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    rows: u32,
    dbatch: u32,
    enc: MoeEnc,
    x_in: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    let all = b.all();
    let one = vec![0u32];
    let (h, nh, dk, dr, vd, ql) = (c.hidden, c.heads, c.kv_lora, c.qk_rope, c.v_head, c.q_lora);
    let tp = c.tp;
    let nh_l = nh / tp; // this rank's head-shard (column-parallel by head); tp==1 => nh
    let w = &n.lw[slot];
    let eps = c.eps;
    // GEMV helper (M=1 decode, no norm fold) — the bf16 projection form both B4 passes used, or its
    // MXFP4 (w4a16) twin. `sc` is the E8M0 scale handle, TENSOR_NONE off the MXFP4 arm. w4a16 and
    // not A4W4 here on purpose: at M=1 this is weight-bandwidth-bound, and the MX scale folds into
    // the fp4->bf16 convert EXACTLY (E8M0 is a power of two), so there is no dequant in the epilogue
    // to pay for — the same conclusion `d_gemv_mxfp4` reached.
    let gemv = |b: &mut Builder,
                out: u32,
                x: u32,
                wt: u32,
                sc: u32,
                nn: u32,
                k: u32,
                deps: &[u32]|
     -> u32 {
        let op = if enc == MoeEnc::Mxfp4 {
            DevOp::GemvMxfp4
        } else {
            DevOp::Gemv
        };
        b.emit(op, all.clone(), deps, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = wt;
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = sc;
            }
            d.i[0] = rows;
            d.i[1] = nn;
            d.i[2] = k;
            d.f[0] = 1.0;
        })
    };
    // Standard Gemma-proven decode fusions. Each defaults ON; set the
    // env to "0" to emit the unfused baseline for a before/after measurement. A and G are byte-exact
    // (GemvQkv concatenates output columns — identical per-column dot/wave_sum/f2bf as the split
    // GEMVs); B1 is algebraically exact (AddNorm reduces over the un-rounded sum — see note below).
    // MXFP4 now has its own GEMV_QKV twin (`DevOp::GemvQkvMxfp4`, op 110), so A and G fuse under it
    // as well. The three E8M0 scale rows the fused bf16 op had nowhere to put ride i5/i6/i7 as
    // TENSOR HANDLES — the demotion `DevOp::GemvQkvg` established, applied to the operand that is
    // strictly safer to demote than the weight it belongs to (read-only, and a wrong one is off by
    // a per-block power of two, i.e. visible in the first token). Byte-exact to the split
    // `GemvMxfp4` calls for the same reason the bf16 pair is byte-exact to its split Gemvs.
    let mx = enc == MoeEnc::Mxfp4;
    let lin_fp8 = glm_linear_fp8(enc);
    // TASK-9 FIT GATE (perf-data glm52-batched-decode-r4.md, TASK 9 ROOT CAUSE). GemvQkv
    // stages rows*K halves of x in LDS unconditionally — op_gemm.h: "x is ALWAYS staged in
    // LDS here: plowc emits this op only when M*K fits GM_LDS_HALVES" — so choosing the fused
    // opcode IS that promise, and at rows > 1 it must be checked, not assumed (the dense-GQA
    // path grew this gate at §6g-WALK; this emitter never called it — the separate-path
    // disease). rows=8 at K=h=6144 is 96 KiB against a 64 KiB LDS window: rows past ~5.3 read
    // zero/garbage x, and the whole rung-8 attractor family follows. The unfused arms below
    // route through DevOp::Gemv, whose body has its own per-op fit fallback and is correct at
    // any M. rows=1 fits both products on every arch in the tree, so single-row emits stay
    // byte-identical.
    let lds_fit = crate::gm_lds_halves();
    let fuse_a = (rows as u64) * (h as u64) <= lds_fit;
    let fuse_g = (rows as u64) * (ql as u64) <= lds_fit;
    // B1 defaults OFF (opt-in): AddNorm reduces over the un-rounded a+b sum, so unlike A/G it is NOT
    // byte-identical to the split Residual+RmsNorm — a reorder-level fp diff that flips one early
    // greedy argmax and cascades. Ship it only behind the HF-coherence gate; PLOW_GLM_FUSE_B1=1 opts in.
    let fuse_b1 = emit_config::active().glm_fuse_b1;

    // --- MLA ---
    // 1 input_layernorm
    // `pre` chains this layer's first op to the PREVIOUS layer's output (x_in), so the 78 layers run
    // in sequence rather than racing on the shared scratch/x buffers. Empty for the single-layer gate
    // (x_in is pre-uploaded before the launch, so no on-device producer to wait on).
    // LAYER-SEAM FOLD: under `PLOW_GLM_FUSE_SEAM` the PREVIOUS layer's tail already wrote
    // `n.xn = rmsnorm(x_in)` in the same `AddNorm` that produced `x_in` itself, so this packet is
    // not emitted and the chain hangs straight off `pre`. `slot > 0` is the whole condition: the
    // blocks are emitted in slot order, layer 0's input comes from the embedding (nothing upstream
    // to fold into), and `seam_next_gin` is what guarantees the producer actually did it (it
    // returns None for the last slot, so the two ends of the seam agree by construction).
    let c_rn1 = if glm_fuse_seam(tp) && slot > 0 {
        assert_eq!(
            pre.len(),
            1,
            "layer-seam fold expects exactly one upstream dep (the previous block's AddNorm); \
             got {} — the chain shape changed and the fold's producer is no longer identifiable",
            pre.len()
        );
        pre[0]
    } else {
        b.emit(DevOp::RmsNorm, one.clone(), pre, |d| {
            d.t[0] = n.xn;
            d.t[1] = x_in;
            d.t[2] = w.gin;
            d.i[0] = rows;
            d.i[1] = h;
            d.f[0] = eps;
        })
    };
    // 2/6/8 down-projections. FUSION A (audit §A): q_a, kv_a and k_rope ALL read n.xn with K=h, so
    //   their output columns concatenate into ONE GemvQkv (Nq=ql q_a, Nk=dk kv_a, Nv=dr k_rope) that
    //   fills every wave (fixing the k_rope/kv_a CU-starvation) and deletes 2 gates/layer. Byte-exact
    //   to the three Gemvs. Legal: M*K = h fits GM_LDS_HALVES.
    let (c_qad, c_ckvd, c_krr) = if fuse_a {
        // n = ql + dk + dr concatenated columns; `blocked_gemv_cus` drops the ceiling tail that
        // owns none of them (GLM TP4: 2624 over 256 => slices 239..255 are empty).
        let fa_cus = blocked_gemv_cus(&all, ql + dk + dr);
        let fa_op = if mx {
            DevOp::GemvQkvMxfp4
        } else {
            DevOp::GemvQkv
        };
        let c_fa = b.emit(fa_op, fa_cus, &[c_rn1], |d| {
            d.t[0] = n.qlr;
            d.t[1] = n.xn;
            d.t[2] = w.qad; // q_a   -> Nq=ql
            d.t[3] = n.ckvraw;
            d.t[4] = w.ckvd; // kv_a  -> Nk=dk
            d.t[5] = n.krr;
            d.t[6] = w.krotd; // k_rope-> Nv=dr
            d.i[0] = rows;
            d.i[1] = ql;
            d.i[2] = h;
            d.i[3] = dk;
            d.i[4] = dr;
            // The tenth, eleventh and twelfth pointers. `t[8]` holds seven of them; the three
            // E8M0 scale rows go in the three integer slots op 22 leaves empty, as handles.
            if mx {
                d.i[5] = w.qad_s;
                d.i[6] = w.ckvd_s;
                d.i[7] = w.krotd_s;
            }
        });
        (c_fa, c_fa, c_fa)
    } else {
        (
            gemv(b, n.qlr, n.xn, w.qad, w.qad_s, ql, h, &[c_rn1]),
            gemv(b, n.ckvraw, n.xn, w.ckvd, w.ckvd_s, dk, h, &[c_rn1]),
            gemv(b, n.krr, n.xn, w.krotd, w.krotd_s, dr, h, &[c_rn1]),
        )
    };
    // 3 q_a_layernorm
    //
    // Q-NORM FOLD (`PLOW_GLM_FUSE_QNORM=1`, opt-in): fusion G's GemvQkv computes this norm
    // into the LDS copy of `x` it already stages, and the packet is not emitted at all.
    //
    // WHY THIS ONE. It is the biggest packet-boundary window left on the decode chain. The
    // census (perf-data/plow-gfx942/glm52-decode-packet-folds.md §1b) traces `GemvQkv-A` end
    // -> `GemvQkv-G` start at **12.2 us** for a **4.6 us** one-workgroup body: a b=1 packet
    // between two ~149-wide GEMVs, so the chain pays a full gate round trip and then runs on
    // one CU while 303 idle. 63.7% of all in-packet CU time in that census is gate-wait, and
    // the serial packet-boundary dead time is 78.2 us of a 355 us layer — removing a BOUNDARY
    // pays more than removing work.
    //
    // THE FAN-OUT LAW (op_gemm.h's `norm == 2` note) IS SATISFIED, and it is the first thing
    // to check: folding a norm into its consumer costs (N-1) extra reductions, which is why
    // the same fold LOST on Gemma (one norm, five consumers). Here `n.qlat` has exactly ONE
    // consumer, the fused G packet — the split (unfused) form has two, which is why `fuse_g`
    // is a precondition and not an optimisation, and the DSA indexer's q projection is a
    // SECOND consumer, which is why an armed DSA gate refuses below. Every workgroup of the
    // one surviving packet recomputes the reduction redundantly (149 of them at
    // PLOW_GLM_GEMV_WG=152), but each does so over a row it has ALREADY staged, off the
    // weight-stream pipe, and none of them is a new chain level.
    //
    // BIT-EXACT BY CONSTRUCTION, the `gemv_norm_lds` argument verbatim: the fold normalizes
    // the staged copy IN PLACE, rounding to bf16 exactly as `d_rmsnorm`'s `fits` path does
    // (same PLOW_THREADS, same per-thread element map, same serial accumulation order, same
    // `block_sum`, same `rsqrtf(ss/feat + eps)`), and the ORDINARY un-normed hot loop then
    // runs over LDS bytes identical to the HBM bytes the deleted packet would have written.
    // It is NOT mode 1: multiplying `x*rms*gamma` inside the k-loop in f32 is a different
    // number from the bf16-rounded value the deleted packet stored.
    let fuse_qnorm = glm_fuse_qnorm(fuse_g, mx, c.dsa(ctx), ql);
    // The q-norm fold's `norm == 2` mechanism normalizes the STAGED x row in place; its M>1
    // behaviour has never been checked. Refuse rather than serve rows 1.. an un-normed q_a.
    assert!(
        !(fuse_qnorm && rows > 1),
        "PLOW_GLM_FUSE_QNORM=1 with a batched decode program (rows={rows}): the fold's \
         per-row staging is unvalidated at M>1. Emit the ladder blob without the fold."
    );
    let c_rnq = if fuse_qnorm {
        c_qad
    } else {
        b.emit(DevOp::RmsNorm, one.clone(), &[c_qad], |d| {
            d.t[0] = n.qlat;
            d.t[1] = n.qlr;
            d.t[2] = w.gqa;
            d.i[0] = rows;
            d.i[1] = ql;
            d.f[0] = eps;
        })
    };
    // 4/5 absorbed q_nope (Wqa: QL -> NH_l*DK) and q_rope raw down (Wqr: QL -> NH_l*DR). FUSION G
    //   (audit §G): both read n.qlat with K=ql, so fuse into ONE GemvQkv with Nv=0 (q half + k half).
    //   Byte-exact. q_rope then gets a dynamic INTERLEAVED RoPE per head at pos (no norm); HD=64
    //   selects the interleaved template; q is not cached (out_row0/stride 0).
    let (c_qa, c_qrr) = if fuse_g {
        let fg_cus = blocked_gemv_cus(&all, nh_l * dk + nh_l * dr);
        let fg_op = if mx {
            DevOp::GemvQkvMxfp4
        } else {
            DevOp::GemvQkv
        };
        let c_fg = b.emit(fg_op, fg_cus, &[c_rnq], |d| {
            d.t[0] = n.qa;
            d.t[1] = n.qlat;
            d.t[2] = w.wqa; // q_nope   -> Nq=nh_l*dk
            d.t[3] = n.qrr;
            d.t[4] = w.wqr; // q_rope raw-> Nk=nh_l*dr
            d.t[5] = TENSOR_NONE;
            d.t[6] = TENSOR_NONE; // Nv=0 (v branch never taken)
            d.i[0] = rows;
            d.i[1] = nh_l * dk;
            d.i[2] = ql;
            d.i[3] = nh_l * dr;
            d.i[4] = 0;
            if fuse_qnorm {
                // Q-NORM FOLD. `t[1]` carries the RAW q_a projection instead of the normed
                // one — SAME SLOT — and `t[7]` the gamma, which is the ONLY slot this costs:
                // op 22's three streams spend t[0..6] and t[7] has never been used on it
                // (op 108 `GemvQkvg`'s t[7] is `g_out`, a DIFFERENT opcode, and ops 114/115
                // put their scale rows in i5/i6/i7 and leave t[7] alone). The interpreter
                // discriminates on `t[7] != NONE`, which is unambiguous for the same reason
                // it is on op 50: no other feature of op 22 reads that slot.
                //
                // NOT A BITFIELD, and nothing here is: op 51's `i[6]` (low 8 bits = causal
                // KV-split `ns`, bit 8 = W_ofold; the sparse GATHER arm reuses it whole as
                // `cap`, discriminated by t7) is a prefill opcode and never meets this one.
                d.t[1] = n.qlr;
                d.t[7] = w.gqa;
                d.f[0] = eps;
            }
            // The TWO-STREAM form. `i7` must carry the sentinel and not be left 0: 0 is a legal
            // handle, and the arm's absence check is what stops a packet missing a scale row from
            // running a narrower sweep than it names.
            if mx {
                d.i[5] = w.wqa_s;
                d.i[6] = w.wqr_s;
                d.i[7] = TENSOR_NONE_I;
            }
        });
        (c_fg, c_fg)
    } else {
        (
            gemv(b, n.qa, n.qlat, w.wqa, w.wqa_s, nh_l * dk, ql, &[c_rnq]),
            gemv(b, n.qrr, n.qlat, w.wqr, w.wqr_s, nh_l * dr, ql, &[c_rnq]),
        )
    };
    // The decode ropes take DISJOINT slices of `all` (see `rope_cus`): q needs ceil(nh_l/8)
    // workgroups, the shared-head k rope needs 1, and the two are concurrent siblings.
    let rq = rope_cus(&all, 0, rows, nh_l);
    let rk = rope_cus(&all, rq.len(), rows, 1);
    // Q-ROPE FOLD (`PLOW_GLM_FUSE_ROPE=1`, opt-in): the flash applies this rope itself, inside
    // the staging that already reads every one of these `nh_l*dr` elements, and the packet is
    // not emitted at all. `rk`'s CU slice is computed from `rq.len()` either way so the k rope
    // and the DSA indexer ropes keep the CU allocation they have today — the fold must not move
    // a packet it does not delete.
    //
    // WHY THE Q ROPE AND NOT THE K ROPE. Measured, not assumed (perf-data/plow-gfx942/
    // glm52-decode-packet-folds.md): on the traced decode step the k rope and the kv_a norm run
    // CONCURRENTLY with the q_a norm and finish ~30 us before the flash gate opens — they are
    // not on the critical path, and folding them would buy nothing while taking on the KV-cache
    // write hazard (the current row must be written for FUTURE steps, and the writer would have
    // to be the one split whose range covers `pos`). The q rope IS on the path: it is a
    // single-workgroup packet sitting between a 149-wide GEMV and the 128-wide flash, so the
    // chain pays a full gate round trip plus a 1-CU body for 512 elements of work.
    let fuse_rope = emit_config::active().glm_fuse_rope;
    let c_qr = if fuse_rope {
        c_qrr
    } else {
        b.emit(DevOp::HeadNormRope, rq.clone(), &[c_qrr], |d| {
            d.t[0] = n.qr;
            d.t[1] = n.qrr;
            d.t[2] = TENSOR_NONE;
            d.t[3] = n.cos;
            d.t[4] = n.sin;
            d.t[5] = n.pos;
            // Row t's angle comes from pos[t] on BOTH arms: at out_stride == 0 the write is the
            // plain row-major ((out_row0 + t)*nhead + hh)*hd, which is exactly the [b][head][dr]
            // layout the flash reads its Q from — q is not cached, so no batch-ring form needed.
            d.i[0] = rows;
            d.i[1] = nh_l;
            d.i[2] = dr;
            d.i[3] = 0;
            d.i[4] = 1;
            d.f[0] = eps;
            d.j[0] = 0;
            d.j[1] = KV_MASK_NONE;
        })
    };
    // 7 kv_a_layernorm -> writes the latent cache (current row = row 0 here; the loader/decode
    //   step rebases the output to the current position, matching the ckv-row write of a decode step).
    //   Reads n.ckvraw from the fused (or split) down-projection above.
    //   Under [`glm_fp8_kv`] the fp8 spelling reuses HeadNormRopeFp8 with no trig tables (the K3
    //   form): same RMSNorm, then quantize the row and record its scale in one pass. The loader's
    //   kv_row_writer scan patches its i[3]; the bf16 RmsNorm writer's row is i[2].
    let fp8kv = glm_fp8_kv();
    assert!(
        !(fp8kv && rows > 1),
        "PLOW_GLM_FP8_KV=1 with a batched decode program (rows={rows}): the fp8 latent writer's \
         batch-ring form is unvalidated on GLM. Emit the ladder blob without fp8-KV."
    );
    let c_rnkv = if fp8kv {
        b.emit(DevOp::HeadNormRopeFp8, one.clone(), &[c_ckvd], |d| {
            d.t[0] = n.ckv[slot];
            d.t[1] = n.ckvraw;
            d.t[2] = w.gkva;
            d.t[5] = n.pos;
            d.t[6] = n.kv_scale[slot];
            d.i[0] = 1;
            d.i[1] = 1;
            d.i[2] = dk;
            d.i[3] = 0; // row, patched per step
            d.i[4] = 0; // apply RMSNorm before quantizing
            d.f[0] = eps;
            d.j[1] = KV_MASK_NONE;
        })
    } else if rows > 1 || dbatch > 1 {
        // BATCHED latent writer. The single-row form is a plain RmsNorm whose write row is a
        // host-patched immediate (`i[2]`) — no per-row form exists on that op, and `patch_kvrow`
        // deliberately does not run at ENGINE batch > 1. That skip is keyed on the BLOB batch,
        // not the rung: on a laddered blob even the rows==1 rung must take this form, or every
        // rung-1 step writes its KV into ring row 0 (measured: needle@3000 answers '741' — the
        // model retrieves from prefill rows and loses every decode-written one). This is the
        // same kernel family as the k-rope below: HeadNormRope at HD=dk, ONE head, gamma armed
        // (RMSNorm over the head), trig tables ABSENT (rope skipped), and the batch-major ring
        // write `((t*nhead + hh)*out_stride + pos[t]) * hd` selected by `i[6] != 0 && j[0] != 0`.
        b.emit(
            DevOp::HeadNormRope,
            rope_cus(&all, rq.len() + rk.len(), rows, 1),
            &[c_ckvd],
            |d| {
                d.t[0] = n.ckv[slot];
                d.t[1] = n.ckvraw;
                d.t[2] = w.gkva;
                d.t[3] = TENSOR_NONE;
                d.t[4] = TENSOR_NONE;
                d.t[5] = n.pos;
                d.i[0] = rows;
                d.i[1] = 1;
                d.i[2] = dk;
                d.i[3] = 0;
                // i[4] is SKIP_NORM on this op (interleave is selected by HD, not a field).
                // The rope packets pass 1 because they must NOT norm; this writer exists to
                // apply kv_a_layernorm, so it passes 0 — the one-field difference that, wrong,
                // caches an unnormalized latent and turns every batched row to garbage.
                d.i[4] = 0;
                d.i[6] = rows; // n_batch_kv: row t writes slot t's ring at pos[t]
                d.f[0] = eps;
                d.j[0] = ctx; // out_stride = the per-slot ring length
                d.j[1] = KV_MASK_NONE;
            },
        )
    } else {
        b.emit(DevOp::RmsNorm, one.clone(), &[c_ckvd], |d| {
            d.t[0] = n.ckv[slot];
            d.t[1] = n.ckvraw;
            d.t[2] = w.gkva;
            d.i[0] = 1;
            d.i[1] = dk;
            d.f[0] = eps;
        })
    };
    // 8 k_rope dynamic INTERLEAVED RoPE (shared 1-head) on n.krr from the fused (or split) down-proj,
    //   writing the rope cache at row=out_row0 (i[3]; the decode step patches it to the current pos).
    //   At rows > 1 the write switches to the batch-major ring form (i[6]/j[0]), same as the latent
    //   writer above — row t's angle AND write row both come from pos[t].
    let c_krd = b.emit(DevOp::HeadNormRope, rk.clone(), &[c_krr], |d| {
        d.t[0] = n.krot[slot];
        d.t[1] = n.krr;
        d.t[2] = TENSOR_NONE;
        d.t[3] = n.cos;
        d.t[4] = n.sin;
        d.t[5] = n.pos;
        d.i[0] = rows;
        d.i[1] = 1;
        d.i[2] = dr;
        d.i[3] = 0;
        d.i[4] = 1;
        if rows > 1 || dbatch > 1 {
            d.i[6] = rows;
            d.j[0] = ctx;
        } else {
            d.j[0] = 0;
        }
        d.f[0] = eps;
        d.j[1] = KV_MASK_NONE;
    });
    // --- DSA lightning indexer (G2/G5): ctx>2048 => project q_idx/k_idx/w, score, top-k select ->
    //     idx table, then FLASH_GATHER over the top_k selected latent rows. ctx<=2048 => dense flash
    //     (top-k is a no-op). 'full' layers own the indexer; 'shared' layers reuse the last full
    //     layer's idx (sequential layer chain => n.iidx already holds it). q_idx/k_idx use a HD=DI GPT-J
    //     interleaved RoPE with the identity-tail table (rope the first qk_rope=DR dims, pass the rest).
    let dsa = c.dsa(ctx);
    // The indexer is single-row end to end: `IndexSelect` has no batch axis (one cooperative
    // grid-sync over one [ctx] score array) and every scratch row (`iscore`/`iidx`/`qidx`/...)
    // is declared one row wide. A batched gate would need per-row histograms and a widened
    // scratch family; refuse until that exists.
    assert!(
        !(dsa && rows > 1),
        "DSA indexer with a batched decode program (rows={rows}): IndexSelect and the indexer \
         scratch are single-row. Emit the ladder blob with PLOW_GLM_DSA=0 or max-ctx <= 65536."
    );
    // The DSA lightning indexer has NO MXFP4 path: `wq_b`/`wk` are block-fp8 (GemvFp8Blk) and
    // `weights_proj` is bf16, and none of the three ops takes an encoding. Under MXFP4 the indexer
    // would therefore be a block-fp8/bf16 island inside an otherwise fp4 packet — and worse, the
    // `weights_proj` GEMV would go through the encoding-aware helper and reach GEMV_MXFP4 with a
    // NULL E8M0 scale. Refuse the combination rather than emit either. GLM-5.2 is the only arch with
    // an indexer and the gate only arms above 64k ctx, so this does not touch Kimi or DeepSeek
    // (`has_dsa=false` holds the gate off at every ctx).
    assert!(
        !(dsa && enc == MoeEnc::Mxfp4),
        "MXFP4 with the DSA indexer armed (ctx={ctx}) would leave the indexer's block-fp8 wq_b/wk          and bf16 weights_proj inside an otherwise all-MXFP4 packet. Missing capability:          `dsa_indexer_mxfp4` (an encoding parameter on INDEX_SCORE's producers). Use PLOW_GLM_DSA=0          to hold the gate off, or emit block-fp8."
    );
    let full = dsa && w.iwqb != TENSOR_NONE; // 'full' indexer layer (weights bound only there)
    let itk = c.index_topk.min(ctx);
    let (hi, di) = (c.index_heads, c.index_dim);
    let gemv_blk = |b: &mut Builder,
                    out: u32,
                    x: u32,
                    wt: u32,
                    sc: u32,
                    nn: u32,
                    k: u32,
                    deps: &[u32]|
     -> u32 {
        b.emit(DevOp::GemvFp8Blk, all.clone(), deps, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = wt;
            d.t[5] = sc;
            d.i[0] = 1;
            d.i[1] = nn;
            d.i[2] = k;
            d.i[4] = 0;
        })
    };
    let c_sel = if full {
        // q_idx = interleaved_rope(reshape_HIxDI(wq_b @ q_lat)); rope in-place (reads staged first).
        let c_q0 = gemv_blk(b, n.qidx, n.qlat, w.iwqb, w.iwqb_s, hi * di, ql, &[c_rnq]);
        // The indexer ropes are concurrent with each other AND with the q/k pair above (all four
        // hang off the input norm), so they continue the same disjoint allocation.
        let riq = rope_cus(&all, rq.len() + rk.len(), 1, hi);
        let rik = rope_cus(&all, rq.len() + rk.len() + riq.len(), 1, 1);
        let c_qi = b.emit(DevOp::HeadNormRope, riq.clone(), &[c_q0], |d| {
            d.t[0] = n.qidx;
            d.t[1] = n.qidx;
            d.t[2] = TENSOR_NONE;
            d.t[3] = n.icos;
            d.t[4] = n.isin;
            d.t[5] = n.pos;
            d.i[0] = 1;
            d.i[1] = hi;
            d.i[2] = di;
            d.i[3] = 0;
            d.i[4] = 1;
            d.i[5] = 1;
            d.f[0] = eps;
            d.j[0] = 0;
            d.j[1] = KV_MASK_NONE;
        });
        // k_idx = interleaved_rope(k_norm_LAYERNORM+BIAS(wk @ xn)) cached [ctx][DI] at pos (like krot).
        let c_k0 = gemv_blk(b, n.kidx_raw, n.xn, w.iwk, w.iwk_s, di, h, &[c_rn1]);
        let c_kn = b.emit(DevOp::LayerNorm, one.clone(), &[c_k0], |d| {
            d.t[0] = n.kidx_normed;
            d.t[1] = n.kidx_raw;
            d.t[2] = w.iknw;
            d.t[3] = w.iknb;
            d.i[0] = 1;
            d.i[1] = di;
            d.i[3] = 0;
            d.f[0] = 1e-6; // k_norm eps
        });
        let c_ki = b.emit(DevOp::HeadNormRope, rik.clone(), &[c_kn], |d| {
            d.t[0] = n.kidx[slot];
            d.t[1] = n.kidx_normed;
            d.t[2] = TENSOR_NONE;
            d.t[3] = n.icos;
            d.t[4] = n.isin;
            d.t[5] = n.pos;
            d.i[0] = 1;
            d.i[1] = 1;
            d.i[2] = di;
            d.i[3] = 0;
            d.i[4] = 1;
            d.i[5] = 1;
            d.f[0] = eps;
            d.j[0] = 0;
            d.j[1] = KV_MASK_NONE;
        });
        // w = weights_proj @ xn  [HI]  (bf16 GEMV)
        // Plain bf16 GEMV, explicitly — NOT the encoding-aware helper. Under MXFP4 that helper
        // would emit GEMV_MXFP4 against a bf16 weight with a null scale; the assert above makes the
        // combination unreachable, and this keeps it that way if the assert is ever relaxed.
        let c_w = b.emit(DevOp::Gemv, all.clone(), &[c_rn1], |d| {
            d.t[0] = n.widx;
            d.t[1] = n.xn;
            d.t[2] = w.iwp;
            d.i[0] = 1;
            d.i[1] = hi;
            d.i[2] = h;
            d.f[0] = 1.0;
        });
        // score[t] = Σ_h w[h]·ReLU(q_idx[h]·k_idx[t]) · scale  (scale = 1/√DI · 1/√HI; selection is
        // scale-invariant, this reproduces HF numerically).
        let c_sc = b.emit(DevOp::IndexScore, all.clone(), &[c_qi, c_ki, c_w], |d| {
            d.t[0] = n.iscore;
            d.t[1] = n.qidx;
            d.t[2] = n.kidx[slot];
            d.t[3] = n.widx;
            d.t[4] = n.kvlen;
            d.i[0] = 1;
            // i1/i3 are the indexer geometry the ISA contract has always specified (dev_isa.h:419)
            // and that this emitter left at ZERO, while `interp.hip` hardcoded `DI_=128, HI_=32`.
            // They are now WRITTEN — see `glm_assert_indexer_geom` for why writing them is not the
            // same as making them free.
            d.i[1] = hi;
            d.i[3] = di;
            d.i[2] = ctx;
            d.f[0] = (di as f32).powf(-0.5) * (hi as f32).powf(-0.5);
        });
        // top-k SELECT -> n.iidx (ONE cooperative launch: grid-sync radix). Perf floor 2: emit on a
        // 32-CU slice, NOT all 256. The selector is grid-barrier CONTENTION-bound, not bandwidth-bound
        // (the score array is only ctx*4 B); cutting the co-resident WG count 256->32 drops the atomic
        // contention on the grid-sync counter and the shared histogram bins (~204->144us @128k, STILL
        // set-EXACT). The kernel reads nwg from in->blocks (=32) and partitions the score array by the
        // entry's LOGICAL SLICE (0..31) — not by blockIdx.x, which under the global-queue decode
        // scheduler is whichever workgroup claimed the entry (that confusion is the end-to-end DSA bug;
        // see d_index_select_coop). All 32 are co-resident under the persistent interp (256 CUs
        // resident, this op gates on INDEX_SCORE, so its 32 WGs run together).
        let sel_wgs: Vec<u32> = (0..32.min(b.n_cu())).collect();
        b.emit(DevOp::IndexSelect, sel_wgs, &[c_sc], |d| {
            d.t[0] = n.iidx;
            d.t[1] = n.iscore;
            d.t[2] = n.ighist;
            d.t[3] = n.igctl;
            // The LIVE kv occupancy. `i[0]` below is only the packet's max ctx, and INDEX_SCORE
            // writes `iscore[pos]` for `pos < kvlen` ONLY — so without this operand the radix
            // ranked `ctx - kvlen` never-written words and selected rows past the end of the
            // latent cache, which the gather then read unmasked. DSA arms only above a 64k
            // crossover, so that gap was the overwhelming majority of every scan.
            d.t[4] = n.kvlen;
            d.i[0] = ctx;
            d.i[1] = itk;
        })
    } else {
        0
    };
    // 9 FLASH (MLA) DECODE — dense (ctx<=2048) or GATHER over the top_k selected latent rows (ctx>2048).
    //   Runs this rank's nh_l head-shard; the latent ckv/krot caches are REPLICATED (all heads read
    //   the same shared latent), so the cache stays full-width on every rank. Under DSA the flash reads
    //   ONLY the top_k rows via n.iidx (constant work ~ top_k regardless of ctx).
    let ns_attn = if dsa {
        glm_nsplit(itk, nh_l)
    } else {
        glm_nsplit(ctx, nh_l)
    };
    // Under the q-rope fold `c_qr` IS `c_qrr` (= the fused q GEMV, which is also `c_qa`), so the
    // list would name one producer twice. Dedup: a repeated dep is not wrong, but it inflates the
    // packet's wait set and the counter analysis reads it as a wider gate than it is.
    let mut fl_deps = vec![c_qa, c_qr, c_rnkv, c_krd];
    fl_deps.dedup();
    if full {
        fl_deps.push(c_sel);
    }
    //   Sized to `n_batch*n_tok*(nh_l/GF)*nsplit` (`flash_mla_cus`), which is exactly `n_cu` at
    //   GF=4 and half of it at GF=2 — see the helper for why the two do not cancel.
    // `FlashGatherDecode` has no fp8 twin (t7 = idx table, where the scales would live), so a
    // gathered packet would read fp8 latent bytes as bf16 — refuse the combination at emit.
    assert!(
        !(fp8kv && dsa),
        "PLOW_GLM_FP8_KV=1 with an armed DSA gate (ctx={ctx}): FlashGatherDecode keeps its bf16 \
         arm in the fp8kv object and cannot dequant an fp8 latent. Set PLOW_GLM_DSA=0."
    );
    // The q-rope fold spends `t[7]` (cos) and `i[6]` (sin, as a demoted handle — the
    // `DevOp::GemvQkvg` rule). BOTH other features on this packet already own those two slots:
    // GATHER puts its idx table in t7 and its top_k in i6, fp8-KV puts its per-row scale strip
    // in t7. The three are therefore SLOT-EXCLUSIVE, exactly as op 51's i[6] bitfield makes
    // ofold and the causal KV-split exclusive. Refuse at emit rather than emit a packet whose
    // reading depends on which arm the object happens to take.
    assert!(
        !(fuse_rope && (dsa || fp8kv)),
        "PLOW_GLM_FUSE_ROPE=1 with {} (ctx={ctx}): the q-rope fold needs t[7] for the cos table \
         and i[6] for the sin handle, which that arm already spends (GATHER: idx/top_k, fp8-KV: \
         kv_scale). Pick one.",
        if dsa {
            "an armed DSA gate"
        } else {
            "PLOW_GLM_FP8_KV=1"
        }
    );
    let c_fl = b.emit(
        match (dsa, fp8kv) {
            (true, _) => DevOp::FlashGatherDecode,
            (false, false) => DevOp::FlashMlaDecode,
            (false, true) => DevOp::FlashMlaDecodeFp8,
        },
        flash_mla_cus(&all, rows, 1, nh_l, glm_gf(ctx, nh_l), ns_attn),
        &fl_deps,
        |d| {
            d.t[0] = n.opart;
            d.t[1] = n.mlpart;
            d.t[2] = n.qa;
            d.t[3] = n.qr;
            d.t[4] = n.ckv[slot];
            d.t[5] = n.krot[slot];
            d.t[6] = n.kvlen;
            d.i[0] = rows;
            d.i[1] = nh_l;
            d.i[2] = ctx;
            d.i[4] = ns_attn;
            d.i[5] = KV_MASK_NONE;
            d.i[7] = glm_gf(ctx, nh_l); // per-pkt head-fusion factor (interp dispatches 2/4/8 on this)
            d.f[0] = c.attn_scale;
            if fuse_rope {
                // Q-ROPE FOLD. `t[3]` carries the RAW q_rope projection instead of the roped
                // one — SAME SLOT, so the fold costs one t and one i, not three: the position
                // needs no operand (`qpos = kv_len[b] - 1`, and every decode entry in plowrt
                // calls with `kvlen == pos + 1`, so that is bit-exactly the `pos[0]` the rope
                // packet read), and `kv_len` is already t[6].
                //
                // `i[6]` AS A TENSOR HANDLE is the demotion `DevOp::GemvQkvg` established and
                // `GemvQkvMxfp4` reuses for its three E8M0 scale rows: legal for a read-only
                // generated table, which both trig tables are. It is NOT a bitfield here and
                // must not be confused with op 51's `i[6]` (low 8 = causal KV-split ns, bit 8 =
                // W_ofold) — different opcode, different meaning, and the two never meet
                // because op 50 is decode and op 51 is prefill.
                //
                // The interpreter discriminates on `t[7] != NONE`, which is unambiguous on op
                // 50: the GATHER and fp8 twins are separate opcodes, and dense op 50 has never
                // carried a t[7]. `i[6]` cannot be the discriminator — 0 is a legal handle.
                d.t[3] = n.qrr;
                d.t[7] = n.cos;
                d.i[6] = n.sin;
            }
            if fp8kv {
                d.t[7] = n.kv_scale[slot]; // per-row dequant scales; the kernel traps on NULL
                d.i[6] = 0; // krot stays bf16 (the shipped K3 form)
            }
            if dsa {
                d.t[7] = n.iidx; // idx table (this or the last full layer's selection)
                d.i[6] = itk; // top_k rows to gather
            }
        },
    );
    // 10 FUSED MLA MERGE+FOLD: online-softmax-merge the ns_attn latent partials (Opart/mlpart) in
    //    LDS, then fold olat @ W_uv straight to v_head_dim — replaces FLASH_MERGE<512> + O_UV_FOLD,
    //    killing the Olat[nh_l*DK] HBM round-trip and one dependency gate (validated rms ~0.004;
    //    ~1.1-1.24x on the MLA chain, composing with the ctx-scaled nsplit to 1.59x at 32k).
    //    Sized to its own work items (`mla_fold_cus`): the fold grid-strides
    //    `n_batch * nh_l * ceil(vd/VT)` times, which is 128 at GLM TP4 — half the machine used to
    //    sit in this packet's gate doing nothing. n_batch is 1 for every decode packet here.
    let c_uv = b.emit(
        DevOp::MlaMergeFold,
        mla_fold_cus(&all, rows * nh_l, vd),
        &[c_fl],
        |d| {
            d.t[0] = n.oat;
            d.t[1] = n.opart;
            d.t[2] = n.mlpart;
            d.t[3] = w.wuv;
            d.i[0] = rows;
            d.i[1] = nh_l;
            d.i[2] = vd;
            d.i[4] = ns_attn;
        },
    );
    // 12 o_proj (NH_l*VD -> H)  [row-parallel]: each rank sums its head-shard into a PARTIAL H-vector.
    //   Under TP the partial goes to the peer-mapped og_tp slot and an XReduce all-reduces the N
    //   partials into n.attn; at tp==1 o_proj writes n.attn directly (byte-identical).
    // PLOW_NO_XREDUCE (diagnostic): drop the 156 all-reduce collectives (o_proj writes n.attn
    // directly with only this rank's partial) — numerically WRONG but same graph minus the
    // cross-GPU rendezvous, to isolate the XReduce cost. Never set for a real decode.
    let no_xr = tp > 1 && emit_config::active().no_xreduce;
    // GLM_LINEAR_FP8: `o_proj` is a whole block-fp8 checkpoint tensor the prep dequantises. On the
    // fp8 arm it goes through GEMV_FP8_BLK (44) reading the checkpoint's own [128,128] grid from
    // t5 — the same opcode/slot the dense-FFN down projection already uses (`dense_down_op`).
    let o_gemv = |b: &mut Builder, out: u32, deps: &[u32]| -> u32 {
        if lin_fp8 {
            b.emit(DevOp::GemvFp8Blk, all.clone(), deps, |d| {
                d.t[0] = out;
                d.t[1] = n.oat;
                d.t[2] = w.wo;
                d.t[5] = w.wo_s;
                d.i[0] = rows;
                d.i[1] = h;
                d.i[2] = nh_l * vd;
                d.i[4] = 0;
            })
        } else {
            gemv(b, out, n.oat, w.wo, w.wo_s, h, nh_l * vd, deps)
        }
    };
    // FUSION XRN (GLM_FUSE_XRN=1, needs B1 + TP): the all-reduce AND the AddNorm as ONE
    // single-workgroup packet (DevOp::XReduceAddNorm, op 116) — the kernel rounds the peer
    // reduction to bf16 before the residual add, so this is BIT-IDENTICAL to the
    // [XReduce -> AddNorm] pair below and deletes one packet + one gate per layer. n.attn is
    // never materialised on this arm (its only consumer was the seam itself).
    // XReduceAddNorm is a one-row kernel (feat in i[0]); at rows > 1 fall back to the unfused
    // pair, which it is bit-identical to by construction — no numerics change, one packet more.
    let fuse_xrn = fuse_b1 && tp > 1 && !no_xr && rows == 1 && emit_config::active().glm_fuse_xrn;
    if fuse_xrn {
        assert!(
            h <= 2 * 8 * 512 && h % 8 == 0,
            "XReduceAddNorm mirrors d_add_norm's fits path at XRN_VEC=2: feat <= 8192 and \
             8-aligned; h={h}"
        );
        let c_p = o_gemv(b, n.og_tp, &[c_uv]);
        let gate = *xgate;
        *xgate += 1;
        return b.emit(DevOp::XReduceAddNorm, one.clone(), &[c_p], |d| {
            d.t[0] = n.xn2;
            d.t[1] = n.xmid;
            d.t[2] = x_in;
            d.t[3] = w.gpost;
            d.i[0] = h;
            d.i[1] = tp;
            d.i[2] = 0; // attn-seam partial slot (same as the unfused emit_xreduce call)
            d.i[3] = gate;
            d.f[0] = eps;
        });
    }
    let c_op = if tp > 1 && !no_xr {
        let c_p = o_gemv(b, n.og_tp, &[c_uv]);
        emit_xreduce(b, xgate, true, xr_cus, c_p, n.attn, rows * h, tp, 0)
    } else {
        o_gemv(b, n.attn, &[c_uv])
    };
    // 13/14 post-attn residual + post_attention_layernorm. FUSION B1 (audit §B1): the plain add
    //   (xmid = x_in + attn) and the RmsNorm that re-reads it are the Qwen/Llama AddNorm pair — ONE
    //   packet writes BOTH the residual stream (xmid, consumed by the FFN combine) and its norm (xn2,
    //   the FFN input), deleting a gate/layer. NOTE: d_add_norm reduces over the UN-rounded a+b sum
    //   whereas the split path norms the bf16-rounded xmid, so this is algebraically exact but NOT
    //   guaranteed byte-identical to the split — the decode stream is verified before it is kept.
    if fuse_b1 {
        b.emit(DevOp::AddNorm, one.clone(), &[c_op], |d| {
            d.t[0] = n.xn2;
            d.t[1] = n.xmid;
            d.t[2] = x_in;
            d.t[3] = n.attn;
            d.t[4] = w.gpost;
            d.i[0] = rows;
            d.i[1] = h;
            d.f[0] = eps;
        })
    } else {
        let c_rs = b.emit(DevOp::Residual, spine_cus(b.n_cu()), &[c_op], |d| {
            d.t[0] = n.xmid;
            d.t[1] = x_in;
            d.t[2] = n.attn;
            d.i[0] = rows * h;
            d.f[0] = 1.0;
        });
        b.emit(DevOp::RmsNorm, one.clone(), &[c_rs], |d| {
            d.t[0] = n.xn2;
            d.t[1] = n.xmid;
            d.t[2] = w.gpost;
            d.i[0] = rows;
            d.i[1] = h;
            d.f[0] = eps;
        })
    }
}

/// MLA head-fusion factor for a PREFILL packet. The decode `glm_gf` returns 8 above its crossover
/// AND the decode interpreter now runs it, but `exec_flash_mla_prefill` (runtime/amd/interp.hip)
/// still instantiates the prefill wrapper at GF 2 and 4 ONLY — `if (gf == 2) … else <4>` — so
/// baking 8 into i[7] would not select a GF=8 body, it would silently run GF=4 under a packet that
/// claims otherwise. Emit what the object actually has. **The decode and prefill twins diverged
/// deliberately when the decode GF=8 arm landed**, for two independent reasons: the prefill bucket
/// is at 256 VGPR / occ 2 / spill 2 with no headroom for a third instantiation, and prefill pins
/// `nsplit = 1` and gets its parallelism from `n_tok`, so raising GF there halves the work-item
/// count with no split axis to win it back. If a prefill GF=8 arm is ever added, this clamp and
/// that `if` are the two places that have to move together.
///
/// The second clamp is `nh_l`: the kernel computes `n_grp = n_head / GF` and its work-item count is
/// `n_batch*n_tok*n_grp*nsplit`, so a GF larger than this rank's head count makes `n_grp == 0` and
/// the flash silently does NOTHING. That is reachable — it is the per-rank-head-count rule the
/// `glm_nsplit` header records, one level up: at tp=8 a 64-head model has nh_l=8, at tp=16 it has 4.
///
/// And as on the decode twin, `nh_l >= 4` is the divide-to-zero guard, NOT a guarantee that GF=4
/// partitions the shard. `nh_l = 6` (Kimi-K3 at tp16) satisfies it and drops heads 4 and 5. See
/// [`glm_gf`]'s header for the full argument; the predicate here is divisibility for the same
/// reason, and it is byte-identical on every power-of-two `nh_l` in the tree.
/// Workgroup list for the T-row norm/residual packets of a PREFILL bucket.
///
/// Decode keeps `spine_cus` (one workgroup) because its bodies are ~µs and widening a producer
/// makes the consumer wait on a max over k stragglers (knob-contract §6b-i). Prefill is the
/// opposite regime, measured on the 2026-08-07 T=2048 trace: the b=1 RmsNorm/Residual packets are
/// 2.5–3.3 ms EACH while 303 CUs idle — 17.8 of every 24.4 ms layer span, 73% of TTFT. Both
/// kernels stride their work by `nblk` (`d_rmsnorm` rows, `d_residual` elements), each row still
/// reduces inside one workgroup, so any width is BIT-IDENTICAL. `rows` caps the norm width at one
/// workgroup per row (the 128-row bucket cannot fill 304). `PLOW_GLM_PF_WIDE=0` restores the
/// single-workgroup emit for A/B.
fn pf_wide_cus(n_cu: u32, rows: u32) -> Vec<u32> {
    if rows <= 1 || !emit_config::active().glm_pf_wide {
        vec![0u32]
    } else {
        (0..n_cu.min(rows)).collect()
    }
}

/// Band count for a prefill TP seam (`PLOW_GLM_XR_BAND=K`, 2..=8; unset/1 = the unbanded
/// emit, byte-identical to before the knob existed).
///
/// K row-bands pipeline the TP data movement against the producer: band 0's two-shot starts
/// its fabric transfer as soon as band 0's producer rows are done, while bands 1..K-1 still
/// compute — see `emit_xreduce_twoshot_band`. The floor (`t/K >= 512`) keeps a band's GEMM
/// from under-filling 304 CUs into oblivion; small buckets emit unbanded.
/// PLOW_GLM_XR_RES=1: fold the post-collective Residual into the two-shot's all-gather
/// (`emit_xreduce_twoshot_band`'s `res` operands). Deletes one packet and a full [T,H]
/// round trip per TP seam; bit-identical to the split emit (the gathered value is already
/// bf16 and the add order is d_residual's). Default off.
fn xr_res_fold() -> bool {
    emit_config::active().glm_xr_res
}

/// Causal KV-split factor for the V2 MLA prefill flash (`PLOW_GLM_PF_NS=k`, 2..=8; unset/1 =
/// the un-split emit, byte-identical to before the knob existed).
///
/// Splits each (q-tile, head) work item into k near-equal shares of ITS OWN causal range, so
/// the item count multiplies and the tail-round quantization dissolves (measured @8k: 1024
/// items on 304 CUs = 3.37 rounds, 34% of the machine idle inside the flash packet). Dense V2
/// arm only — the emitter keeps it off the sparse and fp8-KV arms, sizes `opart`/`mlpart` by
/// it, and the blob grows a `PLOW_MLA_PF_NS` requires (old objects refuse; serving without
/// PLOW_MLA_PF_V2 routing refuses at load rather than degrading onto the 8-wave kernel).
fn glm_pf_ns() -> u32 {
    let k = emit_config::active().glm_pf_ns.unwrap_or(1);
    if (2..=8).contains(&k) {
        k
    } else {
        1
    }
}

/// Per-band collective WIDTH for a banded prefill TP seam (`PLOW_GLM_XR_BAND_CUS=c`,
/// 1..=n_cu; unset = the full `xr_cus` set, i.e. today's banded emit byte-for-byte).
///
/// THIS IS WHAT MAKES BANDING A PIPELINE INSTEAD OF K SERIAL BARRIERS. A packet's workgroup
/// count IS its `cus` list length (`devbuild::Builder::emit` → `inst.blocks`), the global
/// queue hands packets out on ONE monotonic cursor, and a two-shot's `gate_ag` blocks every
/// participating workgroup until `nranks*nblk` arrivals. So a band collective emitted on all
/// 304 CUs consumes the whole grid: no workgroup is left to claim the NEXT band's producer,
/// and K bands cost K full-grid cross-GPU barriers back to back — measured +3-8% TTFT, which
/// is why the band axis shipped default-off. Emitting the band collective on a PREFIX of
/// `xr_cus` leaves `304 - c` workgroups free to walk past it in the stream and claim band
/// i+1's GEMM/combine, which is the overlap the whole construction is for.
///
/// It also makes the K EXTRA BARRIERS affordable, which is the half that actually moved the
/// number. One rendezvous PAIR costs 13.6 + 0.0227*nblk us (measured, probes/xrwg.hip): 82.7
/// us at nblk=304 but 65 us at 112 — so K=4 bands at the shipped full width spend 331 us of
/// barrier per seam where the unbanded seam spends 82, i.e. ~52 ms per 8k prefill chunk of
/// pure added synchronisation. At c=112 the four barriers cost LESS than the one they replace.
/// (The same prefix on the UNBANDED seam is the existing `PLOW_XR_CUS` knob.)
///
/// MEASURED, and this knob exists to RECORD A NULL, not to ship one. Interleaved served A/B,
/// control round-to-round spread 0.2-0.5%, TTFT vs the unbanded control:
///   band4, no subset  +7.9% @4k / +4.6% @8k        (what shipped default-off)
///   band4, c=112      +1.2% / +0.7% / +1.0% @16k
///   band4, c=76       +1.7% / +1.2% / +1.5%
/// The subset removes 85% of the penalty and does not cross zero. The overlap it enables is
/// REAL — the trace shows band i+1's o_proj starting 500+ us INSIDE band i's collective — but
/// on a SATURATED grid "overlap" is a static partition of the same CUs, and the producer pays
/// 1:1 for what the collective gains: the banded o_proj chain goes 643 -> 1407 us while the
/// collective chain goes 1063 -> 1996. Do not re-commission this lever; see
/// perf-data/plow-gfx942/glm52-band-pipeline-cusubset.md.
///
/// Bit-identical at every width: `d_xreduce_twoshot_mega` gives each element to exactly one
/// thread and sums `r = 0..nranks-1` in order, so only the element→workgroup partition moves.
/// `PLOW_XR_CUS` applied to a CU set: a PREFIX of `all`, or `all` when the knob is unset.
///
/// The decode program has honoured this since the TP8 NUMA-crossing lever; the
/// PREFILL program did not — it hardcoded `pall`, so every prefill two-shot ran on
/// all 304 CUs and the knob was silently decode-only. That is not a tuning oversight, it is
/// the whole cost: the two-shot is FABRIC bound, and 304 workgroups x 512 threads
/// OVERSUBSCRIBE the fabric badly. Measured on this box with the shipped kernel at the @8k
/// GLM-5.2 shape (n = 8192*6144 bf16 = 100.7 MB partial, probe in
/// perf-data/plow-gfx942/glm52-band-pipeline-cusubset.md):
///
///   nblk  304    256    224    192    176    152    128     96     76     38
///   us   1279    974    918    873    873    863    894    992   1133   2028
///
/// A broad optimum at 152-192 and the shipped 304 is 48% past it. Note the earlier stagger
/// report's "233 GB/s ceiling" was probed at 304 WGs x 256 THREADS = 77,824 threads; the
/// kernel runs 512 threads/WG, so at 304 workgroups it is at 155,648 and on the far side of
/// the peak. The controlling variable is threads in flight, and the only handle the emitter
/// has on it is the workgroup count.
///
/// AND THE ISOLATED PROBE OVERSTATES IT — do not re-derive this lever from the
/// microbenchmark alone. IN SITU (trace @8k, same objects, same layer) the collective's
/// wall time is FLAT between 152 and 304: 1063/1167 us at nblk=304 vs 1115/1120 at 152,
/// while its CU-time halves (634,275 -> 362,762 CU-us per layer). Inside the megakernel the
/// two-shot is already saturated at 152 workgroups, so narrowing it frees half the machine
/// for free -- and the freed half has nothing to do, because the only packet after the
/// collective is the Residual that depends on it. Served A/B is accordingly a NULL
/// (-0.9% @4k, +0.2% @8k, +0.5% @16k against a 0.2-0.5% control spread). Kept because it is
/// correct, cheap, and halves the collective's CU-time; not adopted into the emit recipe.
///
/// Bit-identical at every width: `d_xreduce_twoshot_mega` (and `d_xreduce`) give each element
/// to exactly one thread and sum `r = 0..nranks-1` in order, so only the element->workgroup
/// partition moves. Decode is unaffected above 12: its one-shot message is `hidden` = 6144
/// elements and `emit_xreduce_gather` already saturates it at `ceil(6144/512)` = 12
/// workgroups, so any cap >= 12 leaves the decode packets byte-identical.
fn xr_cus_capped(n_cu: u32, all: &[u32]) -> Vec<u32> {
    match emit_config::active().xr_cus {
        Some(k) if k > 0 && k < n_cu => (0..k).collect(),
        _ => all.to_vec(),
    }
}

fn xr_band_cus(xr_cus: &[u32]) -> Vec<u32> {
    match emit_config::active().glm_xr_band_cus {
        Some(c) if c > 0 && (c as usize) < xr_cus.len() => xr_cus[..c as usize].to_vec(),
        _ => xr_cus.to_vec(),
    }
}

fn xr_band_k(t: u32, seam: &str) -> u32 {
    // PLOW_GLM_XR_BAND_SEAM=attn|moe restricts banding to one seam — a bisect
    // instrument for isolating which seam a bad blob's divergence comes from.
    if let Some(s) = emit_config::active().glm_xr_band_seam.as_deref() {
        if s != seam {
            return 1;
        }
    }
    let k = emit_config::active().glm_xr_band.unwrap_or(1);
    if (2..=8).contains(&k) && t % k == 0 && t / k >= 512 {
        k
    } else {
        1
    }
}

/// `PLOW_MOE_PF_PART16=1`: the grouped MoE prefill's `part` scatter is bf16 instead of f32 —
/// halves the pair's LARGEST stream (the DOWN scatter + the combine readback; the preshuffle
/// record's corrected traffic model). Numerics-changing (each expert output rounds to bf16
/// before the k-way f32 slot sum); the packet carries it in i[7] on ops 86/87 and the loader
/// refuses objects without the `plow_moe_pf_part16_arm` marker. Decode is untouched — its
/// expert ops keep writing f32 into the same (larger) buffer.
fn moe_pf_part16() -> bool {
    emit_config::active().moe_pf_part16
}

/// `PLOW_MOE_PF_ATOMIC=1`: FUSE the grouped MoE prefill's ops 86 -> 87. Op 86 stops scattering
/// `part[T*k, H]` and instead ATOMICALLY ADDS into a `[T, H]` f32 accumulator (the head of the
/// same `act.part` allocation), op 83 zeroes that accumulator as a prologue, and op 87 runs
/// UNCHANGED with `k = 1`. Removes `T*k*H*4` written by 86 and the same read by 87 — 1.611 GB
/// each way per layer per rank at T=8192 — and turns 87's k strided streams into one contiguous
/// one. Requires a power-of-two top-k (`tok = pidx >> log2(k)`) and an object built with
/// `-DPLOW_MOE_PF_ATOMIC=1`; the `plow_moe_pf_atomic_arm` marker is what refuses the mismatch.
/// NUMERICS-CHANGING: the k-way f32 sum happens in atomic-arrival order instead of fixed slot
/// order, so it is not bit-identical and not run-to-run deterministic. See op_moe.h's header.
fn moe_pf_atomic() -> bool {
    emit_config::active().moe_pf_atomic
}

/// `PLOW_MOE_PF_DET=1`: the DETERMINISTIC form of the same 86 -> 87 fusion. Op 86 accumulates
/// `rint(gate * value * 2^32)` into a `[T, H]` **f64** accumulator with a device-scope f64
/// atomic; every partial sum is an integer below 2^53, hence exact, hence INDEPENDENT of the
/// order the k workgroups arrive in. Op 87 reads one contiguous stream and scales by 2^-32.
/// Costs twice the accumulator bytes of `PLOW_MOE_PF_ATOMIC` and buys run-to-run
/// bit-reproducibility, which that arm does not have. Requires an object built with
/// `-DPLOW_MOE_PF_DET=1` (`plow_moe_pf_det_arm`). See op_moe.h's header and
/// perf-data/plow-gfx942/glm52-moe-deterministic-writer.md.
fn moe_pf_det() -> bool {
    emit_config::active().moe_pf_det
}

/// Which fused 86 -> 87 decomposition is in force. ONE function because the `act.part` SIZE and
/// the packet FIELDS are two halves of the same decision: a sizing that disagreed with the
/// kernel arm is a silent k-fold heap overrun, which is exactly why the atomic branch left the
/// allocation alone. `k` must be a power of two (`tok = pidx >> log2(k)`) and, for DET, at most
/// 16 (the f64 exact-integer bound, op_moe.h MPF_DET_SCALE). part16 excludes both: the
/// accumulator is f32/f64 by construction, so a bf16 combine would misread it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum MoePfFuse {
    None,
    Atomic,
    Det,
}
fn moe_pf_fuse(tk: u32) -> MoePfFuse {
    if moe_pf_part16() || tk == 0 || !tk.is_power_of_two() {
        return MoePfFuse::None;
    }
    if moe_pf_det() && tk <= 16 {
        MoePfFuse::Det
    } else if moe_pf_atomic() {
        MoePfFuse::Atomic
    } else {
        MoePfFuse::None
    }
}

/// `PLOW_MOE_PF_A8=1`: the grouped GLU's gathered activations are fp8 — the post-attention
/// norm's fused quant (d_rmsnorm t3/t4) writes xn2q + per-token scales, and op 85 reads them
/// (t6/t7, i[7]=1), halving the gathered-A stream. Numerics-changing; marker-gated like
/// part16. Block-fp8 experts only (A4W4 owns those operand slots).
fn moe_pf_a8() -> bool {
    emit_config::active().moe_pf_a8
}

fn glm_gf_prefill(ctx: u32, nh_l: u32) -> u32 {
    let gf = if nh_l % 4 == 0 && ctx > GLM_GF_CROSSOVER {
        4
    } else {
        2
    };
    require_gf_divides(gf, nh_l, "prefill");
    gf
}

/// Emit the MLA attention sub-block for a PREFILL bucket of `t` query rows: the T-row twin of
/// [`emit_glm_mla`]. Writes `n.xn2` (what the FFN would consume) and returns its completion dep.
///
/// ## Why this is a twin and not a `Mode` flag on the decode emitter
///
/// The two streams share no ops. Decode's shape is what earns it `GemvQkv` (fusions A and G:
/// three, then two, projections that share an activation collapse into one packet) and the whole
/// `Gemv*` family; NEITHER has a GEMM counterpart in the ISA, so at T rows every one of those emits
/// splits back into separate tiled GEMMs. Threading a mode through would have produced a function
/// whose every branch is taken by exactly one caller, and would have put the decode path — which is
/// op-for-op pinned to a validated gfx950 result by `glm_block_matches_reference_*` — one editing
/// mistake away from moving. The dense path's `Mode` enum earns its keep because prefill and decode
/// there genuinely share a spine; here they do not.
///
/// ## What this does NOT emit: the FFN
///
/// This ends at the post-attention norm, and that is a KERNEL gap, not an oversight. Every FFN op
/// this family uses takes ONE activation row:
///
///   * `d_moe_expert_glu_fp8_blk` / `d_moe_group_glu_fp8_blk` / `d_moe_expert_glu` / `d_moe_combine`
///     and `d_dense_glu_fp8_blk` (runtime/amd/op_moe.h) all take a single `const bf16* x`. The
///     GROUPED op is the near miss: pass `k' = T*top_k` and its `slot`-major `fu`/`part` indexing is
///     already the [T][k][…] layout — only the activation base, fixed at `x` for every slot instead
///     of `x + (slot/k)*H`, is single-row.
///   * There is no block-fp8 GEMM opcode at all: 8/14/15/20 are bf16 and 33-36 carry a PER-CHANNEL
///     fp8 scale, not DeepSeek's [128,128] `weight_scale_inv` grid, so the dense (first_k_dense) FFN
///     and the routed experts have no tiled arm to lower to either.
///   * The routed-expert weights are not declared tensors — the loader fills
///     `expert_weight_table`/`expert_scale_table` with device pointers — so an emitter cannot route
///     around the missing kernel by addressing them from a GEMM.
///
/// The dense-GQA path already takes exactly this position for the same reason, one line:
/// `if c.moe && !moe_pf { break; }` — a MoE model gets a decode-only blob unless grouped-MoE prefill
/// kernels exist. The difference here is that the ATTENTION half has kernels (`FLASH_MLA_PREFILL`,
/// `MLA_MERGE_FOLD`, built into `interp_prefill_mla`), and that half is the one whose absence meant
/// these models could not prefill through their own attention at all.
#[allow(clippy::too_many_arguments)]
/// Emit the DSA SPARSE-PREFILL selection chain for one 'full' indexer layer: the T-row indexer
/// projections + RoPE, the per-query score (117), the per-row exact top-k (118), and the
/// per-64-query-tile union table (119). Returns the union's completion dep — what the gathered
/// flash gates on.
///
/// The projections are the DECODE chain's, at T rows instead of one:
///   q_idx = interleaved_rope(reshape_HIxDI(wq_b @ q_lat))     [T][HI*DI]
///   k_idx = interleaved_rope(k_norm(wk @ xn))                 [T][DI], written at the chunk base
///   w     = weights_proj @ xn                                 [T][HI]
/// `wq_b`/`wk` are block-fp8, so they route through [`emit_pf_gemm_fp8_blk`] exactly as the
/// linear-fp8 prefill arm does; `weights_proj` is bf16 and takes the plain tiled GEMM.
///
/// KEY CACHE. The scorer reads the indexer keys for the WHOLE context `[0, kv_len)`, of which this
/// chunk contributes rows `[q_pos0, q_pos0 + t)` — the same append discipline `kv.{l}.kidx` already
/// has on the decode path, which is why the k_idx rope writes straight into that cache (`out_row0`
/// = the chunk base, patched per chunk by the host exactly as `ckv`/`krot` are). `kidx_pf` is the
/// pre-rope staging only.
#[allow(clippy::too_many_arguments)]
fn emit_glm_dsa_prefill_select(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    w: &GlmLW,
    slot: usize,
    ctx: u32,
    t: u32,
    all: &[u32],
    pre: &[u32],
) -> u32 {
    let (hi, di, h, ql) = (c.index_heads, c.index_dim, c.hidden, c.q_lora);
    let itk = c.index_topk.min(ctx);
    let n_cu = b.n_cu();
    // q_idx: [T][HI*DI] from the q-latent; k_idx: [T][DI] from the input norm.
    let c_q0 = emit_pf_gemm_fp8_blk(
        b,
        all,
        n.qidx_pf,
        n.qlat,
        w.iwqb,
        w.iwqb_s,
        t,
        hi * di,
        ql,
        &[pre[1]],
    );
    let c_qi = b.emit(DevOp::HeadNormRope, all.to_vec(), &[c_q0], |d| {
        d.t[0] = n.qidx_pf;
        d.t[1] = n.qidx_pf;
        d.t[2] = TENSOR_NONE;
        d.t[3] = n.icos;
        d.t[4] = n.isin;
        d.t[5] = n.pos;
        d.i[0] = t;
        d.i[1] = hi;
        d.i[2] = di;
        d.i[3] = 0;
        d.i[4] = 1;
        d.i[5] = 1;
        d.f[0] = c.eps;
        d.j[0] = 0;
        d.j[1] = KV_MASK_NONE;
    });
    let c_k0 = emit_pf_gemm_fp8_blk(b, all, n.kidx_pf, n.xn, w.iwk, w.iwk_s, t, di, h, &[pre[0]]);
    let c_kn = b.emit(DevOp::LayerNorm, pf_wide_cus(n_cu, t), &[c_k0], |d| {
        d.t[0] = n.kidx_pf;
        d.t[1] = n.kidx_pf;
        d.t[2] = w.iknw;
        d.t[3] = w.iknb;
        d.i[0] = t;
        d.i[1] = di;
        d.i[3] = 0;
        d.f[0] = 1e-6; // k_norm eps, the decode chain's constant
    });
    // ...into the SHARED key cache at the chunk base (out_row0), like krot.
    let c_ki = b.emit(DevOp::HeadNormRope, all.to_vec(), &[c_kn], |d| {
        d.t[0] = n.kidx[slot];
        d.t[1] = n.kidx_pf;
        d.t[2] = TENSOR_NONE;
        d.t[3] = n.icos;
        d.t[4] = n.isin;
        d.t[5] = n.pos;
        d.i[0] = t;
        d.i[1] = 1;
        d.i[2] = di;
        d.i[3] = 0;
        d.i[4] = 1;
        d.i[5] = 1;
        d.f[0] = c.eps;
        d.j[0] = 0;
        d.j[1] = KV_MASK_NONE;
    });
    let c_w = b.emit(
        pick_tile(t, hi, h, n_cu, kernelcaps::QuantScheme::None),
        all.to_vec(),
        &[pre[0]],
        |d| {
            d.t[0] = n.widx_pf;
            d.t[1] = n.xn;
            d.t[2] = w.iwp;
            d.i[0] = t;
            d.i[1] = hi;
            d.i[2] = h;
            d.f[0] = c.eps;
        },
    );
    let c_sc = b.emit(DevOp::IndexScorePf, all.to_vec(), &[c_qi, c_ki, c_w], |d| {
        d.t[0] = n.iscore_pf;
        d.t[1] = n.qidx_pf;
        d.t[2] = n.kidx[slot];
        d.t[3] = n.widx_pf;
        d.t[4] = n.kvlen;
        d.i[0] = t;
        d.i[1] = hi;
        d.i[2] = ctx;
        d.i[3] = di;
        d.f[0] = (di as f32).powf(-0.5) * (hi as f32).powf(-0.5);
    });
    // One workgroup per query row; grid-strided, so any width is correct and `min(t, n_cu)` is
    // the point where extra workgroups would idle.
    let c_se = b.emit(
        DevOp::IndexSelectPf,
        (0..n_cu.min(t)).collect::<Vec<_>>(),
        &[c_sc],
        |d| {
            d.t[0] = n.iidx_pf;
            d.t[1] = n.iscore_pf;
            d.t[2] = n.kvlen;
            d.i[0] = t;
            d.i[1] = itk;
            d.i[2] = ctx;
        },
    );
    // One workgroup per query PACK (B2: 8 queries share a union; all 8 heads share the walk).
    let n_qt = t.div_ceil(GLM_DSA_PF_PACK);
    b.emit(
        DevOp::IndexUnionPf,
        (0..n_cu.min(n_qt)).collect::<Vec<_>>(),
        &[c_se],
        |d| {
            d.t[0] = n.iuni;
            d.t[1] = n.iumask;
            d.t[2] = n.iidx_pf;
            d.t[3] = n.kvlen;
            d.i[0] = t;
            d.i[1] = itk;
            d.i[2] = ctx;
            d.i[3] = glm_dsa_pf_cap(c, ctx);
            d.i[4] = GLM_DSA_PF_PACK; // queries per union tile (kernel: 0 = legacy 64)
        },
    )
}

fn emit_glm_mla_prefill(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    t: u32,
    enc: MoeEnc,
    x_in: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    assert!(t > 1, "prefill bucket must carry more than one row (t={t})");
    let all = b.all();
    let n_cu = b.n_cu();
    let (h, nh, dk, dr, vd, ql) = (c.hidden, c.heads, c.kv_lora, c.qk_rope, c.v_head, c.q_lora);
    let tp = c.tp;
    // PER-RANK head count. Everything head-dimensioned below — the q projections' N, the flash's
    // i[1], the merge-fold's i[1], o_proj's K — is this rank's shard, never the global c.heads.
    // Sizing any of them from the global count is the measured tp=8 bug the `glm_nsplit` header
    // records (the flash ran on 32 of 256 CUs); prefill has strictly more work items than decode, so
    // the same mistake would be just as invisible and just as expensive here.
    let nh_l = nh / tp;
    let w = &n.lw[slot];
    let eps = c.eps;
    // Tiled GEMM at prefill shapes. `pick_tile` is the same static cost model the dense prefill path
    // uses, so a narrow M (the 128-row bucket) reaches GemmSmall/GemmMed rather than paying for a
    // 256x256 tile that leaves most of the chip idle.
    let gemm = |b: &mut Builder,
                out: u32,
                x: u32,
                wt: u32,
                sc: u32,
                nn: u32,
                k: u32,
                deps: &[u32]|
     -> u32 {
        // MXFP4 prefill USED to be one opcode (93) with no tile family, and this line pinned it
        // to whatever `GM_BM`/`GM_BN` the object was built with — 256x256 — for every shape.
        // The comment that justified it read: "`d_gemm_mxfp4` reuses the bf16 wide-K MFMA and
        // only the weight fetch differs, so `pick_tile`'s choice does not apply and there is
        // nothing to pick." The premise is right and the conclusion is backwards: because only
        // the weight fetch differs, the tiling geometry — and therefore the CU-fill problem the
        // selector exists to solve — is IDENTICAL to bf16's. "Nothing to pick" followed from
        // there being one opcode, not from any property of the kernel.
        //
        // What it cost: Kimi's `kv_a_proj` (M=128, N=576) is THREE 256x256 tiles on 256 CUs,
        // measured at ≈0.4% of peak — the worst number in the campaign. `op_gemm.h` now
        // instantiates the fp4 body at all five rungs, so the selection applies.
        let op = pick_tile(t, nn, k, n_cu, mxfp4_quant(enc));
        b.emit(op, all.clone(), deps, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = wt;
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = sc;
            }
            d.i[0] = t;
            d.i[1] = nn;
            d.i[2] = k;
            d.i[3] = 0;
            d.i[4] = 0;
            d.f[0] = eps;
        })
    };

    // 1 input_layernorm, T rows.
    let c_rn1 = b.emit(DevOp::RmsNorm, pf_wide_cus(n_cu, t), pre, |d| {
        d.t[0] = n.xn;
        d.t[1] = x_in;
        d.t[2] = w.gin;
        d.i[0] = t;
        d.i[1] = h;
        d.f[0] = eps;
    });
    // 2/6/8 the three down-projections. Decode fuses these into ONE GemvQkv (fusion A); there is no
    // GemmQkv, so prefill keeps them split — the same call the dense path makes ("prefill keeps the
    // split; T rows already parallelise"). Each is a separate tiled GEMM over the whole machine.
    let c_qad = gemm(b, n.qlr, n.xn, w.qad, w.qad_s, ql, h, &[c_rn1]);
    let c_ckvd = gemm(b, n.ckvraw, n.xn, w.ckvd, w.ckvd_s, dk, h, &[c_rn1]);
    let c_krr = gemm(b, n.krr, n.xn, w.krotd, w.krotd_s, dr, h, &[c_rn1]);
    // 3 q_a_layernorm, T rows.
    let c_rnq = b.emit(DevOp::RmsNorm, pf_wide_cus(n_cu, t), &[c_qad], |d| {
        d.t[0] = n.qlat;
        d.t[1] = n.qlr;
        d.t[2] = w.gqa;
        d.i[0] = t;
        d.i[1] = ql;
        d.f[0] = eps;
    });
    // 4/5 absorbed q_nope and raw q_rope (decode's fusion G, likewise unfused here). Output layout
    // [T][nh_l*DK] is exactly the [b][t][head][DK] the flash indexes with b=1.
    let c_qa = gemm(b, n.qa, n.qlat, w.wqa, w.wqa_s, nh_l * dk, ql, &[c_rnq]);
    let c_qrr = gemm(b, n.qrr, n.qlat, w.wqr, w.wqr_s, nh_l * dr, ql, &[c_rnq]);
    // q_rope: dynamic interleaved RoPE over T tokens. i[0]=t is the only change from decode — the
    // per-token angle comes from in.pos[t], which the host already fills for a prefill chunk.
    let c_qr = b.emit(DevOp::HeadNormRope, all.clone(), &[c_qrr], |d| {
        d.t[0] = n.qr;
        d.t[1] = n.qrr;
        d.t[2] = TENSOR_NONE;
        d.t[3] = n.cos;
        d.t[4] = n.sin;
        d.t[5] = n.pos;
        d.i[0] = t;
        d.i[1] = nh_l;
        d.i[2] = dr;
        d.i[3] = 0;
        d.i[4] = 1;
        d.f[0] = eps;
        d.j[0] = 0;
        d.j[1] = KV_MASK_NONE;
    });
    // 7 kv_a_layernorm -> T latent cache rows, written at out_row0 (the chunk base; 0 for a fresh
    //   prompt, rebased by the host for a later chunk exactly as the decode step's row is).
    //   Under [`glm_fp8_kv`]: HeadNormRopeFp8 with no trig (the K3 form) — RMSNorm + quantize +
    //   per-row scale in one pass, T rows; the chunk rebase patches its i[3] (kv_write_row_field).
    let fp8kv = glm_fp8_kv();
    let c_rnkv = if fp8kv {
        b.emit(
            DevOp::HeadNormRopeFp8,
            pf_wide_cus(n_cu, t),
            &[c_ckvd],
            |d| {
                d.t[0] = n.ckv[slot];
                d.t[1] = n.ckvraw;
                d.t[2] = w.gkva;
                d.t[5] = n.pos;
                d.t[6] = n.kv_scale[slot];
                d.i[0] = t;
                d.i[1] = 1;
                d.i[2] = dk;
                d.i[3] = 0; // chunk base, patched per chunk
                d.i[4] = 0; // apply RMSNorm before quantizing
                d.f[0] = eps;
                d.j[1] = KV_MASK_NONE;
            },
        )
    } else {
        b.emit(DevOp::RmsNorm, pf_wide_cus(n_cu, t), &[c_ckvd], |d| {
            d.t[0] = n.ckv[slot];
            d.t[1] = n.ckvraw;
            d.t[2] = w.gkva;
            d.i[0] = t;
            d.i[1] = dk;
            d.f[0] = eps;
        })
    };
    // 8 k_rope: T rows of the shared (1-head) rope key, straight into the rope cache.
    let c_krd = b.emit(DevOp::HeadNormRope, all.clone(), &[c_krr], |d| {
        d.t[0] = n.krot[slot];
        d.t[1] = n.krr;
        d.t[2] = TENSOR_NONE;
        d.t[3] = n.cos;
        d.t[4] = n.sin;
        d.t[5] = n.pos;
        d.i[0] = t;
        d.i[1] = 1;
        d.i[2] = dr;
        d.i[3] = 0;
        d.i[4] = 1;
        d.f[0] = eps;
        d.j[0] = 0;
        d.j[1] = KV_MASK_NONE;
    });
    // 9 FLASH_MLA_PREFILL. Operands are the decode twin's, with ONE reinterpretation: i[4] carried
    //   `nsplit` and now carries `n_tok`. That is forced, not opportunistic — nsplit MUST be 1 here,
    //   because under a per-token causal bound an early token's later splits cover nothing and an
    //   empty split emits l=0 for the merge to divide by (runtime/amd/op_attention.h, the
    //   d_flash_mla_prefill header). Prefill has n_tok*n_grp work items and does not need the split.
    //
    //   SPARSE (PLOW_GLM_DSA_PF): the per-query selector this comment used to say did not exist
    //   is ops 117/118/119 below — T-row indexer score, per-row exact top-k, and the per-64-query
    //   UNION table the V2 flash's GATHER arm walks (t7). The blocker was never the flash kernel;
    //   it was that `IndexScore` scores ONE query and `IndexSelect` emits ONE row, so a gather
    //   against them gave every query the LAST token's selection. Dense stays the arm at every
    //   bucket the gate does not cover (`glm_dsa_pf_bucket`), and a sparse blob run WITHOUT the V2
    //   routing degrades to dense (the 8-wave kernel ignores t7) rather than corrupting.
    // B2's head-batched gather fills the 64 M-rows with (8 queries x 8 heads); a shard with
    // any other per-rank head count has no row map, so those configs stay dense.
    let sparse = glm_dsa_pf_bucket(c, t) && w.iwqb != TENSOR_NONE && nh_l == 8;
    // The indexer chain is emitted on the layers that carry its weights ('full' layers); shared
    // layers reuse the previous full layer's union table, exactly as the decode chain reuses its
    // idx. `c_uni` threads that reuse: it is the last emitted union's dep, or the flash's own deps
    // when no chain has run yet in this program.
    let c_sel_pf = if sparse {
        Some(emit_glm_dsa_prefill_select(
            b,
            c,
            n,
            w,
            slot,
            ctx,
            t,
            &all,
            &[c_rn1, c_rnq],
        ))
    } else {
        None
    };
    assert!(
        !(sparse && fp8kv),
        "PLOW_GLM_DSA_PF=1 with fp8 latent KV: the V2 flash GATHER arm is bf16-only (op 110 never \
         routes to the flash object), so the gathered packet would read fp8 latent bytes as bf16. \
         Unset one of PLOW_GLM_FP8_KV / PLOW_GLM_DSA_PF."
    );
    // OFOLD (fusion-audit seam 1): flash writes normalized bf16 partials (i[6]), the merge
    // packet disappears, o_proj becomes the fused W_ofold GEMM. Refuse the arms it cannot
    // compose with rather than silently downgrade — both would corrupt (sparse/fp8kv own
    // t7/i6 meanings on their packets).
    //
    // `t >= 2048` MIRRORS THE HOST'S V2 ROUTING GUARD (exec/amd.rs derive_segments:
    // `v2 = env && prog.t >= 2048`): a smaller bucket runs the 8-WAVE kernel, which ignores
    // i[6] and leaves unnormalized f32 partials — the fused GEMM then reads them as bf16 and
    // every short prompt serves garbage (measured: '!!!!' on the quality gate, which rides
    // the t=128 bucket, while the t>=2048 amd-bench runs looked plausible). Small buckets
    // keep the split merge+o_proj emit; the two arms coexist per bucket in one blob.
    // Per-bucket policy when BOTH knobs are on: ofold owns t in [2048, 8192) (measured
    // -4.0% @4k vs ns2's -3.2%) and the causal KV-split owns t >= 8192 (measured -7.7/-9.2%
    // @8k/16k vs ofold's -4.1%) — they are mutually exclusive on a bucket because the fold
    // consumes the UN-SPLIT l (ns==1 is its premise).
    let ofold = glm_ofold(enc) && t >= 2048 && !(glm_pf_ns() > 1 && t >= 8192);
    assert!(
        !(ofold && (sparse || fp8kv)),
        "PLOW_GLM_OFOLD=1 cannot combine with PLOW_GLM_DSA_PF or PLOW_GLM_FP8_KV: the ofold \
         epilogue is the dense V2 arm's i[6], and the fused GEMM reads bf16 partials. Unset one."
    );
    let mut fl_deps = vec![c_qa, c_qr, c_rnkv, c_krd];
    if let Some(d) = c_sel_pf {
        fl_deps.push(d);
    }
    // Causal KV-split (PLOW_GLM_PF_NS, V2 flash only): items become (q-tile, head, split),
    // each split a ceil-equal share of ITS tile's causal range — what fixes the tail-round
    // quantization (1024 items on 304 CUs = 3.37 rounds, 34% of the machine idle inside the
    // packet, 8k trace). i[6] carries it on the DENSE arm only; the sparse arm owns i[6]=cap.
    // t >= 2048 mirrors the host's V2 routing threshold (exec/amd.rs): below it the packet
    // runs on the 8-wave kernel, which does not honor the split — the host refuses that
    // combination, so the emitter must not produce it for the small rungs.
    let pf_ns = if fp8kv || sparse || t < 2048 || ofold {
        1 // ofold owns this bucket: the fold consumes the un-split l (ns==1 premise)
    } else {
        glm_pf_ns()
    };
    let c_fl = b.emit(
        if fp8kv {
            DevOp::FlashMlaPrefillFp8
        } else {
            DevOp::FlashMlaPrefill
        },
        all.clone(),
        &fl_deps,
        |d| {
            d.t[0] = n.opart;
            d.t[1] = n.mlpart;
            d.t[2] = n.qa;
            d.t[3] = n.qr;
            d.t[4] = n.ckv[slot];
            d.t[5] = n.krot[slot];
            d.t[6] = n.kvlen;
            if fp8kv {
                d.t[7] = n.kv_scale[slot]; // per-row dequant scales; the kernel traps on NULL
            } else if sparse {
                d.t[7] = n.iuni; // the union table; its presence selects the GATHER arm
                d.i[6] = glm_dsa_pf_cap(c, ctx);
            } else if ofold {
                // Dense V2 arm bitfield, bit 8: the W_ofold fused epilogue (normalized bf16
                // partials for the fused o-GEMM; requires ns==1 — the fold consumes the
                // un-split l). Low 8 bits stay 0 (unsplit). manifest `requires` PLOW_GLM_OFOLD.
                d.i[6] = 1 << 8;
            } else if pf_ns > 1 {
                // Dense V2 arm bitfield, low 8 bits: the causal KV-split ns.
                // manifest `requires` PLOW_MLA_PF_NS.
                d.i[6] = pf_ns;
            }
            d.i[0] = 1; // n_batch (single sequence per prefill chunk)
            d.i[1] = nh_l; // PER-RANK heads
            d.i[2] = ctx; // kv_stride
            d.i[3] = 0; // window: 0 = full causal (MLA has no sliding regime)
            d.i[4] = t; // n_tok — the slot decode used for nsplit
            d.i[5] = KV_MASK_NONE;
            d.i[7] = glm_gf_prefill(ctx, nh_l);
            d.f[0] = c.attn_scale;
        },
    );
    // 10 FUSED MLA MERGE+FOLD, nsplit=1. The partials are [b][t][head][nsplit][DK] and the fold
    //    indexes them as (b*n_head + h) — so the token axis is folded into i[0]: n_batch := 1*t.
    //    That is the same identity the flash uses (`qrow = (b*n_tok + t)*n_head`), not a trick.
    //    At nsplit=1 the online-softmax merge is a pass-through and the op is purely the W_uv fold,
    //    which is why the separate OUvFold opcode is not emitted: MlaMergeFold subsumes it and the
    //    AMD prefill object carries both, so this costs one dependency gate less.
    //    Same sizing rule as decode. INERT at every real prefill bucket: `bh = t*nh_l` and
    //    `bh*8 > n_cu` for any t >= 16, so VT is 256, `need = bh >= 256` and the clamp hands the
    //    whole machine straight back. It is here so the invariant does not have to be rediscovered
    //    on this emitter the way `emit_xreduce`'s sizing had to be (knob-contract §7c).
    // Under OFOLD the merge packet does not exist: the flash's normalized bf16 partials ARE
    // the fused GEMM's A operand, so the o-GEMM deps straight on the flash.
    let c_uv = if ofold {
        c_fl
    } else {
        b.emit(
            DevOp::MlaMergeFold,
            mla_fold_cus(&all, t * nh_l, vd),
            &[c_fl],
            |d| {
                d.t[0] = n.oat;
                d.t[1] = n.opart;
                d.t[2] = n.mlpart;
                d.t[3] = w.wuv;
                d.i[0] = t;
                d.i[1] = nh_l;
                d.i[2] = vd;
                // nsplit: 1 unless the causal KV-split is on — its dead splits carry
                // m=-inf/l=0, which this merge weighs 0 branch-free (the "empty split"
                // precondition the old comment described predates that merge rewrite).
                d.i[4] = pf_ns;
            },
        )
    };
    // 12 o_proj, row-parallel over this rank's head shard. Under TP the [T,hidden] partial goes
    //    through the TWO-SHOT all-reduce (reduce-scatter + all-gather), not decode's one-shot: the
    //    partial is bandwidth-bound at T rows, so the two-shot moves ~tp/2x less over the fabric
    //    (see the design notes). `emit_xreduce(decode=false)` is that path, already in the tree.
    //
    //    Under GLM_LINEAR_FP8 `w.wo`/`w.wo_s` are the CHECKPOINT's block-fp8 bytes and its
    //    [128,128] weight_scale_inv grid, not the bf16 the prep dequantises to — the same pair the
    //    decode emitter binds to `GemvFp8Blk` (44). This is the T-row arm; before it existed
    //    `declare_glm_rows` refused the combination rather than put a bf16 `Gemm` on fp8 bytes.
    //    It is on the DENSE layers too: `emit_glm_dense_block_prefill` calls this same emitter.
    let lin_fp8 = glm_linear_fp8(enc);
    let oproj = |b: &mut Builder, out: u32, deps: &[u32]| -> u32 {
        if ofold {
            // Fused W_ofold GEMM: A = the flash's normalized bf16 partials in the opart
            // allocation ([T, nh_l*DK] row-major), K = nh_l*DK. Replaces merge + o_proj.
            gemm(b, out, n.opart, w.wofold, TENSOR_NONE, h, nh_l * dk, deps)
        } else if lin_fp8 {
            emit_pf_gemm_fp8_blk(b, &all, out, n.oat, w.wo, w.wo_s, t, h, nh_l * vd, deps)
        } else {
            gemm(b, out, n.oat, w.wo, w.wo_s, h, nh_l * vd, deps)
        }
    };
    let no_xr = tp > 1 && emit_config::active().no_xreduce;
    // Banding is bf16-GEMM-only: the five tiled bf16 wrappers carry c_row0 (i[5]); the
    // block-fp8 and MXFP4 GEMM families do not, and banding them would silently overwrite
    // band 0's rows.
    // (ofold also forces kb=1: the banded arm's inline GEMMs hardcode oat/wo.)
    let kb = if lin_fp8 || enc == MoeEnc::Mxfp4 || ofold {
        1
    } else {
        xr_band_k(t, "attn")
    };
    let xr_deps: Vec<u32> = if tp > 1 && !no_xr && kb > 1 {
        // BANDED TP seam (see emit_xreduce_twoshot_band): K row-band o_proj GEMMs, each
        // feeding its own two-shot — band 0's fabric transfer overlaps bands 1..K-1's
        // compute. a_row0/c_row0 keep the bands inside the ONE og_tp partial, so the
        // per-band slot/e0 arithmetic below is the same [T,h] layout the unbanded emit uses.
        let rows = t / kb;
        // CU-SUBSET SCHEDULING (see `xr_band_cus`): the band collectives run on a PREFIX of
        // `xr_cus` so the workgroups outside it walk past them on the global queue and claim
        // the next band's o_proj. The producer keeps `all` — it is a plain counter-gated
        // packet, so whatever workgroups are free drain its 304 slices.
        let bcus = xr_band_cus(xr_cus);
        (0..kb)
            .map(|i| {
                let op = pick_tile(rows, h, nh_l * vd, n_cu, mxfp4_quant(enc));
                let c_p = b.emit(op, all.clone(), &[c_uv], |d| {
                    d.t[0] = n.og_tp;
                    d.t[1] = n.oat;
                    d.t[2] = w.wo;
                    if enc == MoeEnc::Mxfp4 {
                        d.t[3] = w.wo_s;
                    }
                    d.i[0] = rows;
                    d.i[1] = h;
                    d.i[2] = nh_l * vd;
                    d.i[4] = i * rows; // a_row0
                    d.i[5] = i * rows; // c_row0
                    d.f[0] = eps;
                });
                emit_xreduce_twoshot_band(
                    b,
                    xgate,
                    &bcus,
                    &[c_p],
                    n.attn,
                    rows * h,
                    tp,
                    0, // region base — the band offset rides in e0 (loader infers slot_bytes from max i2)
                    i * rows * h,
                    None,
                )
            })
            .collect()
    } else if tp > 1 && !no_xr {
        let c_p = oproj(b, n.og_tp, &[c_uv]);
        if xr_res_fold() {
            // XR+Residual fold: the all-gather writes xmid = x_in + reduced directly and the
            // Residual packet below is not emitted. Same packet fields as emit_xreduce plus
            // t1/t2 — kb==1 through the band fn is the proven-identical emit shape.
            vec![emit_xreduce_twoshot_band(
                b,
                xgate,
                xr_cus,
                &[c_p],
                n.attn,
                t * h,
                tp,
                0,
                0,
                Some((x_in, n.xmid)),
            )]
        } else {
            vec![emit_xreduce(
                b,
                xgate,
                false,
                xr_cus,
                c_p,
                n.attn,
                t * h,
                tp,
                0,
            )]
        }
    } else {
        vec![oproj(b, n.attn, &[c_uv])]
    };
    // 13/14 post-attn residual + post_attention_layernorm. The decode path can fuse these into one
    //   AddNorm (fusion B1, opt-in); prefill keeps them split for the same reason the dense path
    //   does — T rows already parallelise the norm, so the fusion buys a gate and costs the
    //   byte-identity that made B1 opt-in in the first place. Under the XR_RES fold the
    //   Residual was folded into the collective above and the norm gates on the XR directly.
    let folded_attn = tp > 1 && !no_xr && kb == 1 && xr_res_fold();
    let rs_deps: Vec<u32> = if folded_attn {
        xr_deps
    } else {
        vec![
            b.emit(DevOp::Residual, pf_wide_cus(n_cu, t * h), &xr_deps, |d| {
                d.t[0] = n.xmid;
                d.t[1] = x_in;
                d.t[2] = n.attn;
                d.i[0] = t * h;
                d.f[0] = 1.0;
            }),
        ]
    };
    b.emit(DevOp::RmsNorm, pf_wide_cus(n_cu, t), &rs_deps, |d| {
        d.t[0] = n.xn2;
        d.t[1] = n.xmid;
        d.t[2] = w.gpost;
        // PLOW_MOE_PF_A8: the fused w8a8 quant epilogue also writes xn2q + per-token scales
        // for the grouped GLU's fp8 gathered-A arm (bit-identical xn2 either way).
        if n.xn2q != TENSOR_NONE {
            d.t[3] = n.xn2q;
            d.t[4] = n.xn2s;
        }
        d.i[0] = t;
        d.i[1] = h;
        d.f[0] = eps;
    })
}

/// Emit ONE MoE (sparse) decoder block for a PREFILL bucket of `t` rows: MLA attention at T rows
/// then the TOKEN-SORTED grouped-expert FFN. Returns the combine's completion dep; writes `x_out`.
///
/// The FFN here is not the decode FFN with a row loop — it is a different decomposition, and it has
/// to be. Decode routes ONE token, so `top_k` expert packets each stream one expert's weights for a
/// single activation row: pure weight bandwidth, no reuse to find. At T rows the same experts are
/// hit by many tokens, so the win is to SORT the `T*k` (token, expert) slots by expert and run one
/// grouped GEMM whose A operand is gathered — every expert's weights cross HBM once for all the
/// tokens that chose it. That is what ops 83-87 implement and why they are separate opcodes rather
/// than a row count on ops 45/46.
///
/// The pieces that are NOT expert-routed stay ordinary prefill GEMMs: the router score is a plain
/// `[T,n_exp]` `Gemm` (the top-k tail is the only new router op), and the shared expert is
/// `GemmGlu` + `Gemm` on bf16 weights.
#[allow(clippy::too_many_arguments)]
fn emit_glm_block_prefill(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    t: u32,
    enc: MoeEnc,
    x_in: u32,
    x_out: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    let c_rn2 = emit_glm_mla_prefill(b, c, n, slot, ctx, t, enc, x_in, pre, xgate, xr_cus);
    let all = b.all();
    let one = vec![0u32];
    let n_cu = b.n_cu();
    let (h, e, tk, imoe) = (c.hidden, c.n_exp, c.top_k, c.moe_inter);
    let tp = c.tp;
    let imoe_l = imoe / tp;
    // EP: routed experts are distributed WHOLE across ranks (n_exp/tp per rank, full moe_inter), so
    // no rank ever runs a CU-starved fragment of an expert. Without EP they are TP-sliced. Identical
    // rule to the decode block — the grouped prefill GEMM resolves its weight bases from the same
    // expert_weight_table, and the host binds NULL for a non-local expert either way.
    let imoe_e = if c.ep { imoe } else { imoe_l };
    let w = &n.lw[slot];
    // EXPERT-COUNT BOUND. Two LDS carves scale with n_exp and neither is checked on device: the
    // align op's `cnt[n_exp] | cur[n_exp] | tot` and the router tail's `scores[n_exp] | keys[n_exp]`.
    // Both fit the AMD raw arena at 384 (the largest, keys, is 384*8 = 3 KiB against ~144 KiB), which
    // is the number Kimi K2.7 actually routes; `d_moe_align_pf`'s header states the bound it was
    // written for as 512. Asserting it HERE is the only place it can be checked at all — the kernel
    // sizes from the runtime operand and would just overrun. Note the sm_120 twin hardcodes
    // `PLOW_MOE_MAXE 256` (runtime/nvidia/op_moe.cuh), so this family is gfx950-only past 256.
    const MOE_PF_MAX_EXPERTS: u32 = 512;
    assert!(
        e <= MOE_PF_MAX_EXPERTS,
        "n_routed_experts={e} exceeds the grouped MoE prefill LDS bound of {MOE_PF_MAX_EXPERTS} \
         (the align histogram and the router key array are both carved from the shared arena)"
    );
    let gemm = |b: &mut Builder,
                out: u32,
                x: u32,
                wt: u32,
                sc: u32,
                nn: u32,
                k: u32,
                deps: &[u32]|
     -> u32 {
        // Same fix as the sibling emitter above: the fp4 prefill GEMM goes through `pick_tile`
        // like every other encoding, instead of being pinned to the object's default tile.
        let op = pick_tile(t, nn, k, n_cu, mxfp4_quant(enc));
        b.emit(op, all.clone(), deps, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = wt;
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = sc;
            }
            d.i[0] = t;
            d.i[1] = nn;
            d.i[2] = k;
        })
    };

    // 15a router SCORE: the [T, n_exp] logit matrix is an ordinary tiled GEMM — the router split's
    //     decode half was already "the ordinary multi-CU GEMV", and this is its T-row twin.
    let c_score = gemm(b, n.rlogit, n.xn2, w.wr, w.wr_s, e, h, &[c_rn2]);
    // 15b router TOP-K tail, block-per-token. Bit-identical PER TOKEN to the decode tail (the kernel
    //     is literally that kernel under a token loop), so the 8-of-384 selection a prefill chunk
    //     makes is the selection decode would have made for the same row.
    // PLOW_MOE_PF_ATOMIC: this packet also zeroes the [T, H] f32 accumulator op 86 will
    // atomically add into. It is the earliest packet of the MoE chain (router -> align -> GLU ->
    // DOWN), so the existing edges already order the zero before every writer — no new packet,
    // no new gate, and the zero (201 MB at T=8192) rides a packet that is 1.8% of the layer.
    // `!moe_pf_part16()` is a real exclusion, not tidiness: the accumulator is f32 by
    // construction (the atomic is an f32 add), so a part16 blob would have op 87 read it as bf16.
    // `tk.is_power_of_two()` is what makes `tok = pidx >> log2(k)` exact.
    let fuse = moe_pf_fuse(tk);
    let atom = fuse == MoePfFuse::Atomic;
    // PLOW_MOE_PF_DET: same decomposition, f64 fixed-point accumulator, order-independent sum.
    let det = fuse == MoePfFuse::Det;
    let c_router = b.emit(DevOp::MoeRouterTopkPf, all.clone(), &[c_score], |d| {
        d.t[0] = n.tab;
        d.t[1] = n.rlogit;
        d.t[3] = w.bias;
        if atom || det {
            d.t[2] = n.part;
            d.i[0] = h;
        }
        d.i[1] = e;
        d.i[2] = tk;
        d.i[3] = GLM_ROUTER_FLAGS;
        d.i[4] = t;
        // Same group operands as the decode tail — the prefill kernel is that kernel under a token
        // loop, so an emitter that set them on one and not the other would make prefill and decode
        // route the same token to DIFFERENT experts.
        d.i[6] = c.n_group;
        d.i[7] = c.topk_group;
        d.f[0] = c.route_scale;
    });
    // 15c ALIGN/SORT — ONE workgroup, and it must be: the MPF_BM-padded row prefix is a global scan.
    //     The other 255 CUs are gated behind it by the counter DAG exactly as they are behind the
    //     decode router, so this is the same shape of serialization the decode path already pays.
    let c_align = b.emit(DevOp::MoeAlignPf, one.clone(), &[c_router], |d| {
        d.t[0] = n.meta;
        d.t[1] = n.tab;
        d.t[2] = n.row_token;
        d.t[3] = n.row_partidx;
        d.t[4] = n.row_gate;
        d.i[0] = t;
        d.i[1] = e;
        d.i[2] = tk;
    });
    // 16 shared expert gate|up — GemmGlu, the T-row twin of decode's GemvGlu (same operand slots,
    //    M in i[0]). Column-parallel: this rank's imoe_l lanes. Routing-independent, so it gates
    //    only on the post-attn norm and overlaps the whole router/align/expert chain.
    // MXFP4 now HAS a fused prefill arm (`DevOp::GemmGluMxfp4`, op 109), the T-row twin of decode's
    // op 92, so the pair fuses whenever 256x256 is the winning fp4 rung for this shape — the same
    // test the bf16 path applies, because the epilogue is instantiated at that tile only. When a
    // narrower rung wins the pair still UNFUSES into two GemmMxfp4 plus a Glu (op 5): that is a
    // tile decision, not a precision one, and `glu_fusion_wins_mxfp4` prices it (the unfused triple
    // materialises gate AND up to HBM and reads both back — ~8 B per output element).
    // BLOCK-FP8 (GLM_LINEAR_FP8) still unfuses ALWAYS, because there is no `GemmGluFp8Blk`: gate
    // and up are two `GemmFp8Blk` (107) against the checkpoint's own fp8 bytes + [128,128]
    // weight_scale_inv, with an explicit `Glu`. Same operand slots, same `n.shfu_up`. This is the
    // T-row arm the decode side has had as op 47 all along. Its fused twin is NOT the same edit as
    // the mxfp4 one: an arbitrary-f32 block scale must be PROMOTED into a second f32 accumulator
    // every 128 K rather than folded into the convert, so a fused gate|up would need TWO promotion
    // sets on top of two accumulators and cannot be built at 8 waves (see `d_gemm_fp8_blk`'s note
    // on why that family has one tile rung at all). MXFP4 has no such cost — its E8M0 scale folds
    // into the cvt exactly — which is why the fp4 arm falls out and the block-fp8 one does not.
    let lin_fp8 = glm_linear_fp8(enc);
    let c_shglu = if enc == MoeEnc::Mxfp4 && glu_fusion_wins_mxfp4(t, imoe_l, h, n_cu) {
        b.emit(DevOp::GemmGluMxfp4, all.clone(), &[c_rn2], |d| {
            d.t[0] = n.shfu;
            d.t[1] = n.xn2;
            d.t[2] = w.shg;
            d.t[3] = w.shg_s;
            d.t[4] = w.shu_s;
            d.t[5] = w.shu;
            d.i[0] = t;
            d.i[1] = imoe_l;
            d.i[2] = h;
            d.i[5] = GLM_ACT_SILU;
        })
    } else if enc == MoeEnc::Mxfp4 || lin_fp8 {
        let (c_g, c_u) = if lin_fp8 {
            (
                emit_pf_gemm_fp8_blk(
                    b,
                    &all,
                    n.shfu,
                    n.xn2,
                    w.shg,
                    w.shg_s,
                    t,
                    imoe_l,
                    h,
                    &[c_rn2],
                ),
                emit_pf_gemm_fp8_blk(
                    b,
                    &all,
                    n.shfu_up,
                    n.xn2,
                    w.shu,
                    w.shu_s,
                    t,
                    imoe_l,
                    h,
                    &[c_rn2],
                ),
            )
        } else {
            (
                gemm(b, n.shfu, n.xn2, w.shg, w.shg_s, imoe_l, h, &[c_rn2]),
                gemm(b, n.shfu_up, n.xn2, w.shu, w.shu_s, imoe_l, h, &[c_rn2]),
            )
        };
        b.emit(DevOp::Glu, all.clone(), &[c_g, c_u], |d| {
            d.t[0] = n.shfu;
            d.t[1] = n.shfu;
            d.t[2] = n.shfu_up;
            d.i[0] = t * imoe_l;
            d.i[1] = GLM_ACT_SILU;
        })
    } else {
        b.emit(DevOp::GemmGlu, all.clone(), &[c_rn2], |d| {
            d.t[0] = n.shfu;
            d.t[1] = n.xn2;
            d.t[2] = w.shg;
            d.t[5] = w.shu;
            d.i[0] = t;
            d.i[1] = imoe_l;
            d.i[2] = h;
            d.i[5] = GLM_ACT_SILU;
        })
    };
    // 17 shared expert down — row-parallel (imoe_l input): a PARTIAL [T,H] under TP.
    let c_shd = if lin_fp8 {
        emit_pf_gemm_fp8_blk(
            b,
            &all,
            n.shared,
            n.shfu,
            w.shd,
            w.shd_s,
            t,
            h,
            imoe_l,
            &[c_shglu],
        )
    } else {
        gemm(b, n.shared, n.shfu, w.shd, w.shd_s, h, imoe_l, &[c_shglu])
    };
    // 18 grouped gate/up + GLU over the sorted rows. A is gathered from xn2 by row_token, so an
    //    expert's gate|up crosses HBM ONCE for every token that chose it — the reuse decode cannot
    //    have. i[3] picks the block-fp8 or bf16 weight arm from the same tables decode uses.
    // PLOW_MOE_PF_SHUF: read the routed weights from the PRESHUFFLED prefill slab (i6=1 tells
    // the kernel the B'[K/64][R][64] layout). Scales and every other operand are unchanged.
    let shuf = w.ewt_pf != TENSOR_NONE;
    let c_g = b.emit(DevOp::MoeGroupGluPf, all.clone(), &[c_align, c_rn2], |d| {
        d.t[0] = n.fu_g;
        d.t[1] = n.xn2;
        d.t[2] = if shuf { w.ewt_pf } else { w.ewt };
        d.t[3] = w.est;
        d.t[4] = n.meta;
        d.t[5] = n.row_token;
        d.i[6] = u32::from(shuf);
        // A4W4 binds two more: t6 = row_partidx, so the fused bridge can tell a PAD row from a live
        // one and skip it (the bf16/fp8 arms let pad rows fall out in DOWN's scatter instead, but a
        // bridge that quantized them would write E8M0 bytes for rows nothing reads); t7 = the E8M0
        // scale rows it WRITES, because the bridge is this op's epilogue rather than a separate op.
        if enc == MoeEnc::Mxfp4 {
            d.t[6] = n.row_partidx;
            d.t[7] = n.fu_scale;
        }
        // PLOW_MOE_PF_A8 (block-fp8 experts only — A4W4 owns t6/t7): fp8 gathered A.
        if enc == MoeEnc::Fp8Blk && n.xn2q != TENSOR_NONE {
            d.t[6] = n.xn2q;
            d.t[7] = n.xn2s;
            d.i[7] = 1;
        }
        d.i[0] = imoe_e;
        d.i[1] = h;
        d.i[2] = e;
        d.i[MoeEnc::PREFILL_SLOT] = enc.code();
        d.i[5] = GLM_ACT_SILU;
    });
    // 19 grouped down + gate-scale + SCATTER into part[T*k, H]. row_partidx carries each gathered
    //    row's fixed destination, so the align op's nondeterministic within-expert row ORDER does
    //    not reach the output — the combine below still sums part in FIXED slot order.
    let c_d = b.emit(DevOp::MoeGroupDownPf, all.clone(), &[c_g], |d| {
        d.t[0] = n.part;
        d.t[1] = n.fu_g;
        d.t[2] = if shuf { w.ewt_pf } else { w.ewt };
        d.t[3] = w.est;
        d.t[4] = n.meta;
        d.i[6] = u32::from(shuf);
        // t5 = the E8M0 rows the bridge wrote — DOWN's A operand is fp4 + these, so under A4W4 the
        // activation never returns to bf16 between the two GEMMs.
        if enc == MoeEnc::Mxfp4 {
            d.t[5] = n.fu_scale;
        }
        d.t[6] = n.row_partidx;
        d.t[7] = n.row_gate;
        d.i[0] = h;
        d.i[1] = imoe_e;
        d.i[2] = e;
        d.i[MoeEnc::PREFILL_SLOT] = enc.code();
        d.i[7] = u32::from(moe_pf_part16()); // bf16 part scatter
                                             // PLOW_MOE_PF_ATOMIC: log2(k)+1. `row_partidx[row] == token*k + slot` (d_moe_align_pf),
                                             // so the epilogue recovers the token with one shift and adds into acc[token][H].
        if atom {
            d.i[4] = tk.trailing_zeros() + 1;
        }
        // PLOW_MOE_PF_DET: the SAME log2(k)+1, in a DIFFERENT field, so an object carrying one
        // arm can never read a blob emitted for the other as its own.
        if det {
            d.i[5] = tk.trailing_zeros() + 1;
        }
    });
    // 20 T-token combine. Under TP `shared`/`part` are PARTIALS, so the combine residual must NOT be
    //    xmid (XReduce would sum it tp times): it writes the partial with a zero residual, the
    //    two-shot all-reduce folds the ranks, and a Residual then adds the real xmid. tp==1 keeps
    //    the fused xmid combine — the same structure the decode block uses, one phase up.
    let no_xr = tp > 1 && emit_config::active().no_xreduce;
    if tp > 1 && !no_xr {
        // Banded TP seam (see emit_xreduce_twoshot_band): K row-band combines, each feeding
        // its own two-shot, so a band's fabric transfer overlaps the later bands' combines
        // — and, K>1 aside, the LAST band's transfer is all that stays exposed before the
        // Residual. kb==1 emits the exact pre-banding packets.
        let kb = xr_band_k(t, "moe");
        let rows = t / kb;
        // CU-SUBSET SCHEDULING — see `xr_band_cus` and the attn seam's twin. Gated on kb>1
        // so `PLOW_GLM_XR_BAND_CUS` never narrows the UNBANDED collective (that is
        // `PLOW_XR_CUS`'s job) and the two seams stay symmetric under every knob combination.
        let bcus = if kb > 1 {
            xr_band_cus(xr_cus)
        } else {
            xr_cus.to_vec()
        };
        let xr_deps: Vec<u32> = (0..kb)
            .map(|i| {
                let c_cmb = b.emit(DevOp::MoeCombinePf, all.clone(), &[c_shd, c_d], |d| {
                    d.t[0] = n.dg_tp;
                    // TENSOR_NONE, not `n.zero_h`. Under TP the combine's residual must be ZERO
                    // (xmid is added after the all-reduce, or XReduce would sum it tp times) —
                    // and `d_moe_combine_pf` ALREADY spells zero as a null pointer
                    // (`residual ? bf2f(residual[i]) : 0.0f`). Naming a [T,H] bf16 buffer of
                    // literal zeros made the kernel READ 100.7 MB per layer per rank at T=8192
                    // (7.5 GB per 8k chunk over 75 MoE layers) to add 0.0f. Bit-identical by
                    // inspection of that ternary, and it removes one of this op's four streams.
                    d.t[1] = TENSOR_NONE;
                    d.t[2] = n.shared;
                    d.t[3] = n.part;
                    d.i[0] = h;
                    // PLOW_MOE_PF_ATOMIC: op 86 already summed the k slots in place, so this
                    // reads ONE contiguous f32 stream. Same kernel, same expression, k = 1.
                    d.i[1] = if atom || det { 1 } else { tk };
                    d.i[2] = rows;
                    d.i[3] = i * rows; // t_row0
                    d.i[4] = u32::from(det); // f64 fixed-point accumulator (PLOW_MOE_PF_DET)
                    d.i[7] = u32::from(moe_pf_part16());
                });
                // `n.slot_b`, NOT `t * h * 2`: the offset is a property of the BLOB (where
                // the host binds `act.dg_tp`), not of this bucket. See GlmTn.
                emit_xreduce_twoshot_band(
                    b,
                    xgate,
                    &bcus,
                    &[c_cmb],
                    n.attn,
                    rows * h,
                    tp,
                    n.slot_b, // region base — the band offset rides in e0
                    i * rows * h,
                    (kb == 1 && xr_res_fold()).then_some((n.xmid, x_out)),
                )
            })
            .collect();
        if kb == 1 && xr_res_fold() {
            // XR+Residual fold — the collective wrote x_out = xmid + reduced itself.
            xr_deps[0]
        } else {
            b.emit(DevOp::Residual, pf_wide_cus(n_cu, t * h), &xr_deps, |d| {
                d.t[0] = x_out;
                d.t[1] = n.xmid;
                d.t[2] = n.attn;
                d.i[0] = t * h;
                d.f[0] = 1.0;
            })
        }
    } else {
        b.emit(DevOp::MoeCombinePf, all.clone(), &[c_shd, c_d], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = if atom || det { 1 } else { tk }; // see the banded twin
            d.i[2] = t;
            d.i[4] = u32::from(det); // see the banded twin
            d.i[7] = u32::from(moe_pf_part16());
        })
    }
}

/// How much of a layer a requested prefill bucket is asked to cover.
///
/// Two scopes rather than a bool, because "prefill" is not one capability here: the ATTENTION half
/// has verified kernels (`FLASH_MLA_PREFILL` / `FLASH_GATHER_PREFILL` / `MLA_MERGE_FOLD` /
/// `O_UV_FOLD`, all dispatched by `interp_prefill_mla`) and the FFN half has none on any backend.
/// Collapsing them into one flag is what would let a "prefill" request quietly produce a packet that
/// runs and is wrong — the AMD dispatch `default:` does not trap, it leaves the output buffer
/// untouched, so a MoE op in a prefill bucket reads as an accuracy bug, not a crash.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PrefillScope {
    /// MLA attention only, ending at `post_attention_layernorm`. What is emittable today.
    Attn,
    /// Whole layer, attention + FFN — MoE layers on the grouped expert arms (ops 83-87), dense
    /// (`< first_k_dense_replace`) layers on the SAME arms with degenerate 1-expert routing
    /// (see [`emit_glm_dense_block_prefill`]).
    Full,
}

/// Emit ONE DENSE (`layer < first_k_dense_replace`) decoder block for a PREFILL bucket of `t` rows:
/// MLA attention at T rows, then the dense SwiGLU FFN run on the GROUPED EXPERT ARMS with
/// degenerate routing. Returns the completion dep; writes `x_out`. The T-row twin of
/// [`emit_glm_dense_block`].
///
/// ## Why the dense FFN runs on the MoE arms
///
/// This layer's decode FFN is `DenseGluFp8Blk` (47) + `GemvFp8Blk` (44) against DeepSeek's
/// `[128,128]` `weight_scale_inv` grids. Both are M=1 wave-per-output, and there is NO block-fp8
/// GEMM opcode to lower them to at T rows — 8/14/15/20 are bf16 and 33-36 carry a per-CHANNEL fp8
/// scale, which is a different quantisation. That gap used to make this function a `panic!`, and
/// because a prefill program must cover EVERY layer, three dense layers with no T-row arm meant
/// GLM-5.2 got no prefill program at all — hence a 1024-token prompt walked through the decode
/// program one token at a time, 1024 dispatches, and a measured 37.9 s TTFT against vLLM's 1.9 s.
///
/// The kernel that was "missing" already exists. Ops 85/86 (`MoeGroupGluPf` / `MoeGroupDownPf`)
/// ARE a T-row block-fp8 GEMM against a `[128,128]` grid — `d_moe_group_pf_t<FP8=true>` in
/// `runtime/amd/op_moe.h`, whose `KB = (K+127)>>7` is that grid. They were only ever reached
/// through a router. A dense FFN is the same GEMM with the routing degenerated: ONE expert, every
/// token assigned to it, gate 1. So this emits `n_exp = 1`, `top_k = 1` and lets `MoeAlignPf`
/// synthesise the routing (`t[1] = TENSOR_NONE`; the construction is documented ONCE, in
/// `d_moe_align_pf`'s header, and must not be replicated).
///
/// Cost of the reuse, measured on gfx950 / ROCm 7.2.4: **zero**. The prefill object is
/// 256 VGPR / occ 2 / spill 2 with and without the MoE prefill arms, so this does not move the
/// 256/occ-2 cliff `scripts/build_gfx950.sh` gates on.
///
/// `GemmFp8Blk` (107) — the real dense arm this paragraph used to call "the better long-term
/// shape" — now exists, and the dense FFN prefill still does NOT use it. That is deliberate. The
/// grouped arms cost this object nothing (measured above) and give the dense FFN the same
/// gather/scatter machinery every other layer already runs; switching it to three `GemmFp8Blk`
/// packets plus a `Glu` would trade a proven zero-cost reuse for a new emission shape on three of
/// 78 layers. `GemmFp8Blk` was added for the case the grouped arms genuinely cannot serve —
/// `o_proj` and the shared expert, which have no expert tables, no row maps and no f32 `part`
/// output. It would also serve a block-fp8 model with no MoE at all, where there are no grouped
/// arms to borrow; nothing in the tree is that model yet.
///
/// ## What is NOT reused
///
/// The router (op 83) and the shared expert. A dense layer has neither: there is nothing to score
/// and no `mlp.shared_experts.*` in the checkpoint. The combine therefore takes `shared` =
/// `TENSOR_NONE`, which `d_moe_combine_pf` already honours (`if (shared) acc += ...`).
#[allow(clippy::too_many_arguments)]
fn emit_glm_dense_block_prefill(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    t: u32,
    enc: MoeEnc,
    x_in: u32,
    x_out: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    let c_rn2 = emit_glm_mla_prefill(b, c, n, slot, ctx, t, enc, x_in, pre, xgate, xr_cus);
    let all = b.all();
    let one = vec![0u32];
    let n_cu = b.n_cu();
    let (h, di) = (c.hidden, c.dense_inter);
    let tp = c.tp;
    let di_l = di / tp; // this rank's dense-FFN intermediate lanes (column-parallel gate/up)
    let w = &n.lw[slot];
    assert!(
        w.dwt != TENSOR_NONE,
        "dense prefill needs the dense weight-pointer table; declare_glm_rows only emits it for \
         rows > 1, so this layer was declared decode-only"
    );
    // MXFP4 has no dense prefill arm: op 85/86's A4W4 path is `PLOW_MOE_PF_A4W4` and expects the
    // fused-bridge scale rows, which the dense emit does not carry. Refuse rather than emit a
    // packet whose encoding field points at an arm nothing bound operands for.
    assert!(
        enc != MoeEnc::Mxfp4,
        "dense-FFN prefill is implemented for bf16 and block-fp8 only (enc={enc:?}); the MXFP4 \
         grouped arm needs the fused-bridge scale rows this path does not declare"
    );

    // The DEGENERATE routing. `n_exp = 1`, `k = 1`, table = TENSOR_NONE => `d_moe_align_pf`
    // synthesises "every token -> expert 0, gate 1" instead of reading a router's output. It still
    // writes the same meta/row_token/row_partidx/row_gate the MoE path uses, so ops 85/86 below are
    // the identical emits with a different expert count.
    const DENSE_N_EXP: u32 = 1;
    const DENSE_TOP_K: u32 = 1;
    let c_align = b.emit(DevOp::MoeAlignPf, one.clone(), &[c_rn2], |d| {
        d.t[0] = n.meta;
        d.t[1] = TENSOR_NONE; // no routing table — see d_moe_align_pf's header
        d.t[2] = n.row_token;
        d.t[3] = n.row_partidx;
        d.t[4] = n.row_gate;
        d.i[0] = t;
        d.i[1] = DENSE_N_EXP;
        d.i[2] = DENSE_TOP_K;
    });
    // gate/up + SwiGLU over the (identity-)gathered rows. Column-parallel: this rank's di_l lanes.
    let c_g = b.emit(DevOp::MoeGroupGluPf, all.clone(), &[c_align, c_rn2], |d| {
        d.t[0] = n.fu_g;
        d.t[1] = n.xn2;
        d.t[2] = w.dwt;
        d.t[3] = w.dst;
        d.t[4] = n.meta;
        d.t[5] = n.row_token;
        // PLOW_MOE_PF_A8: same fp8 gathered-A arm as the routed GLU (dense = 1-expert routing).
        if enc == MoeEnc::Fp8Blk && n.xn2q != TENSOR_NONE {
            d.t[6] = n.xn2q;
            d.t[7] = n.xn2s;
            d.i[7] = 1;
        }
        d.i[0] = di_l;
        d.i[1] = h;
        d.i[2] = DENSE_N_EXP;
        d.i[MoeEnc::PREFILL_SLOT] = enc.code();
        d.i[5] = GLM_ACT_SILU;
    });
    // down + scatter. Row-parallel (di_l input) => a PARTIAL [T,H] under TP. row_gate is all 1s, so
    // the DOWN op's unconditional gate multiply is the identity here.
    let c_d = b.emit(DevOp::MoeGroupDownPf, all.clone(), &[c_g], |d| {
        d.t[0] = n.part;
        d.t[1] = n.fu_g;
        d.t[2] = w.dwt;
        d.t[3] = w.dst;
        d.t[4] = n.meta;
        d.t[6] = n.row_partidx;
        d.t[7] = n.row_gate;
        d.i[0] = h;
        d.i[1] = di_l;
        d.i[2] = DENSE_N_EXP;
        d.i[MoeEnc::PREFILL_SLOT] = enc.code();
        d.i[7] = u32::from(moe_pf_part16()); // bf16 part scatter
    });
    // Combine. Identical TP structure to the MoE prefill block: under TP the partial is combined
    // with a ZERO residual, two-shot all-reduced, and the real residual added after — folding xmid
    // in before the all-reduce would sum it tp times.
    let no_xr = tp > 1 && emit_config::active().no_xreduce;
    if tp > 1 && !no_xr {
        let c_cmb = b.emit(DevOp::MoeCombinePf, all.clone(), &[c_d], |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = TENSOR_NONE; // see the MoE twin — a null residual IS the zero residual
            d.t[2] = TENSOR_NONE; // no shared expert on a dense layer
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = DENSE_TOP_K;
            d.i[2] = t;
            d.i[7] = u32::from(moe_pf_part16());
        });
        // `n.slot_b`, NOT `t * h * 2`: the offset is a property of the BLOB (where the host binds
        // `act.dg_tp`), not of this bucket. See the field's header on GlmTn.
        if xr_res_fold() {
            // XR+Residual fold — same seam shape as the MoE block's, degenerate routing aside.
            emit_xreduce_twoshot_band(
                b,
                xgate,
                xr_cus,
                &[c_cmb],
                n.attn,
                t * h,
                tp,
                n.slot_b,
                0,
                Some((n.xmid, x_out)),
            )
        } else {
            let c_xr = emit_xreduce(b, xgate, false, xr_cus, c_cmb, n.attn, t * h, tp, n.slot_b);
            b.emit(DevOp::Residual, pf_wide_cus(n_cu, t * h), &[c_xr], |d| {
                d.t[0] = x_out;
                d.t[1] = n.xmid;
                d.t[2] = n.attn;
                d.i[0] = t * h;
                d.f[0] = 1.0;
            })
        }
    } else {
        b.emit(DevOp::MoeCombinePf, all.clone(), &[c_d], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = TENSOR_NONE; // no shared expert on a dense layer
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = DENSE_TOP_K;
            d.i[2] = t;
            d.i[7] = u32::from(moe_pf_part16());
        })
    }
}

/// The one weight that stays bf16 under `MoeEnc::Mxfp4`, and why that is an EXCEPTION rather than
/// a mixed packet.
///
/// `W_uv` — the derived v_absorb the MLA epilogue folds with — is `const bf16*` in
/// `d_mla_merge_fold` and `d_o_uv_fold`, neither of which takes an encoding parameter. There is
/// nowhere to put an fp4 form, so it stays bf16 on every arm.
///
/// The reason this is safe where the decode experts were not: `W_uv` is DERIVED by host weight-prep
/// (a fold of `kv_b_proj`), not read from the checkpoint. Weight-prep dequantizes to compute the
/// fold regardless, so a bf16 copy exists whatever the checkpoint stores. The expert weights were
/// the opposite case — fp4 bytes on disk read as bf16 by an op with no fp4 arm, which is noise.
///
/// Size, so nobody has to guess how much of the model this is: `n_head * kv_lora * v_head`. On Kimi
/// K2.7 that is 64*512*128 = 4.19M values against ~16.9G in one layer's experts — about 0.025%.
///
/// It is reported through `build.json` rather than left implicit, because "all-MXFP4 except one
/// derived 4M tensor" is a fact a benchmark comparison needs to state, and the alternative — a
/// number nobody can reconcile with the claimed dtype — is the failure this whole line of work
/// exists to prevent. Closing it needs an encoding parameter on ops 52/57.
fn mxfp4_bf16_exceptions() -> &'static [&'static str] {
    &["MlaMergeFold/Wuv", "OUvFold/Wuv"]
}

/// What the CHECKPOINT says its weights are, read from `config.json`.
///
/// `None` = no `quantization_config`, i.e. an unquantized checkpoint or a host-prepped bf16 dir —
/// the historical case, where the env flags decide and nothing changes.
///
/// WHY THIS EXISTS. The first real GLM-5.2 emit produced a bf16 block from a checkpoint that is
/// block-fp8 on disk: `weight_enc: "bf16"`, `MoeExpertGlu` instead of `MoeExpertGluFp8Blk`. Nothing
/// on this path read `quantization_config` at all — the encoding came only from `PLOW_FP8`, so
/// omitting the flag silently asked for bf16 weights THAT DO NOT EXIST, and the block-fp8 expert
/// arms built for this exact model family were never reached.
///
/// Two traps in the parsing, both live in this file's target config:
///
///   * the dtype key is `dtype`, not `torch_dtype` — HF renamed it, and GLM-5.2-FP8 carries only
///     the new spelling. A `torch_dtype`-keyed probe finds nothing and falls back to bf16;
///   * `dtype` is `"bfloat16"` on this checkpoint ANYWAY, because it describes the COMPUTE dtype.
///     The storage dtype is in `quantization_config`. Inferring the weight encoding from the dtype
///     field would read "bfloat16" off an fp8 checkpoint and be confidently wrong.
///
/// So this reads `quantization_config` and nothing else.
fn mla_ckpt_enc(dir: &Path) -> Option<MoeEnc> {
    let v: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json")).ok()?).ok()?;
    let q = v.get("quantization_config")?;
    let method = q.get("quant_method").and_then(|m| m.as_str()).unwrap_or("");
    let fmt = q.get("fmt").and_then(|m| m.as_str()).unwrap_or("");
    // The block-fp8 the expert/dense arms implement: e4m3 with a [128,128] scale grid. The 128 is
    // not a parameter anywhere in this emitter — `div_ceil(128)` is written into every scale-grid
    // size — so a checkpoint quantized at any other block size would bind grids of the wrong shape.
    // Check it rather than assume it; the field exists precisely because it can vary.
    if method == "fp8" {
        let blk: Vec<u64> = q
            .get("weight_block_size")
            .and_then(|b| b.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
            .unwrap_or_default();
        assert!(
            fmt.is_empty() || fmt == "e4m3",
            "checkpoint quantization_config.fmt = {fmt:?}; the block-fp8 arms are e4m3 only. \
             Missing capability: `fp8_fmt_{fmt}`."
        );
        assert!(
            blk == vec![128, 128],
            "checkpoint quantization_config.weight_block_size = {blk:?}, but every scale-grid size \
             in this emitter is written as div_ceil(128) — a different block size would bind grids \
             of the wrong shape against weights that look fine. Missing capability: \
             `fp8_block_size_{blk:?}`."
        );
        return Some(MoeEnc::Fp8Blk);
    }
    // Anything else is a quantization we cannot emit. REFUSE rather than fall back to bf16: the
    // weights on disk are not bf16, so a bf16 packet is a WRONG packet, not an unoptimised one —
    // the same rule that makes w8a16-on-gfx950 a refusal rather than a silent substitution.
    panic!(
        "checkpoint quantization_config.quant_method = {method:?}, which this emitter cannot emit \
         for the MLA+MoE family. Missing capability: `ckpt_quant_{method}`. It supports \
         quant_method \"fp8\" with fmt e4m3 and weight_block_size [128,128] (ops 45/46/48/49), or \
         an unquantized checkpoint. Emitting bf16 here would ask the loader to bind bf16 weights \
         that do not exist in this checkpoint."
    )
}

/// The MoE weight encoding an emit run asks for, from the environment.
///
/// `PLOW_MXFP4=1` now RETURNS [`MoeEnc::Mxfp4`] — both blockers are cleared. The decode experts
/// took an encoding field (`i[6]`, not `i[3]`; see [`MoeEnc::DECODE_SLOT`]) and the combined object
/// carries the A4W4 experts and the w4a16 projections together, so the packet no longer strands
/// half its ops in an object that cannot run them.
///
/// Setting `PLOW_FP8=1` alongside asks for two encodings in one packet and is refused: a run is all
/// of one thing. The single documented exception is [`mxfp4_bf16_exceptions`].
fn mla_moe_enc_env(dir: &Path) -> MoeEnc {
    // THE AXIS NAMES MEAN THE SAME THING HERE AS ON THE DENSE PATH — which is why one of them is
    // refused rather than aliased.
    //
    // `PLOW_W8A16` IS this family's fp8 profile: the block-fp8 expert arms take fp8 weights and
    // leave the activation bf16 ("x stays bf16 (w8a16)", runtime/amd/op_moe.h). So it is a true
    // alias for `PLOW_FP8` and accepted as one.
    //
    // `PLOW_W8A8` is NOT. There is no activation-quantized fp8 expert arm: ops 45/46/48/49 are
    // w8a16 in every instantiation. Accepting it as an alias would hand back w8a16 when w8a8 was
    // asked for — silently substituting a different computation under a flag that named one
    // precisely, which is the failure this renaming existed to remove. The activation-quantizing
    // option for this family is A4W4 (`PLOW_MXFP4`), where BOTH operands are 4-bit.
    let cfg = emit_config::active();
    let w8a16 = cfg.w8a16;
    assert!(
        !cfg.w8a8,
        "PLOW_W8A8=1 is not implementable for the MLA+MoE family: its block-fp8 expert arms (ops \
         45/46/48/49) are w8a16 in every instantiation — fp8 weights, bf16 activations — so there \
         is nothing to quantize the activation with. Missing capability: `moe_w8a8`. Use \
         PLOW_W8A16=1 (or PLOW_FP8=1) for this family's fp8 profile, or PLOW_MXFP4=1 for A4W4, \
         which is the one path here that narrows the activation too."
    );
    let use_fp8 = cfg.fp8 || w8a16;
    let mxfp4 = cfg.mxfp4;
    assert!(
        !(mxfp4 && use_fp8),
        "PLOW_MXFP4=1 and PLOW_FP8=1 together would ask for two weight encodings in one packet. \
         A run is ALL-mxfp4 or ALL-fp8 or ALL-bf16; pick one."
    );
    let asked = MoeEnc::from_flags(use_fp8, mxfp4);
    // THE CHECKPOINT WINS, because it is a FACT and the flags are a REQUEST. What is on disk
    // decides what the ops can read; a flag can only ask. Where they agree this is a no-op, where
    // the checkpoint is unquantized the flags decide exactly as before, and where they CONTRADICT
    // the request is refused rather than silently granted or silently ignored.
    match mla_ckpt_enc(dir) {
        None => asked,
        Some(ck) if ck == asked => ck,
        // Nothing was asked for and the checkpoint is quantized: adopt it. This is the case that
        // was silently emitting bf16 against fp8 weights.
        Some(ck) if asked == MoeEnc::Bf16 => {
            eprintln!(
                "  weight encoding {ck:?} detected from the checkpoint's quantization_config \
                 (no PLOW_FP8/PLOW_MXFP4 set)."
            );
            ck
        }
        Some(ck) => panic!(
            "the checkpoint's quantization_config says its weights are {ck:?}, but the environment \
             asked for {asked:?}. The bytes on disk decide what the ops can read, so this cannot be \
             honoured either way round — unset the flag to use the checkpoint as quantized, or \
             point at a checkpoint in the requested encoding."
        ),
    }
}

/// Why `FlashGatherPrefill` (op 55) still has no emit site, and what would give it one.
///
/// NOT an emitter oversight and NOT a missing flash kernel — `d_flash_gather_prefill` exists, is
/// dispatched, and is correct. What is missing is its `idx` operand. Its own header states the
/// shape: "`idx` is one top_k row PER QUERY — `[b][t][top_k]` — because a sparse prefill selects a
/// different set for every query token, which is exactly the axis the dense decode gather did not
/// have." Nothing in the tree produces that array.
///
/// The two ops that would have to grow a token axis, with what they are today:
///
/// | op | today | what a T-row form needs |
/// |----|-------|--------------------------|
/// | `IndexScore` (58) | `t0=Score(f32[ctx])` for ONE query; `i0=n_batch i1=index_heads i2=kv_stride i3=index_head_dim`, `f0=scale` | `t0=Score(f32[T][ctx])`, `t1=Qidx(bf16[T][HI][DI])`, plus `i4=n_tok`. The MFMA form already tiles the key matrix through LDS and re-streams it per query; at T rows that key tile is shared by every query in the tile, so this should get FASTER per token, not slower — the same reuse the grouped MoE GEMM found. |
/// | `IndexSelect` (59) | `t0=iidx(i32[top_k])`, one cooperative launch, 7 radix passes over `ctx` packed keys, `i0=ctx i1=top_k` | `t0=iidx(i32[T][top_k])` and `i2=n_tok`. The grid-barrier structure is the hard part: today ONE selection owns the whole 32-CU slice and its `ighist`/`igctl` scratch. T selections either serialize over that (T grid-syncs, likely far too slow) or need per-token histogram/ctl scratch (`[T][7][256]` u32 and `[T][3]` u32) so tokens can be selected concurrently across the slice. The second is the interesting one and is a real design question, not a mechanical widening — which is why this is a kernel-side scoping note and not something to guess at from here. |
///
/// A causality constraint comes with it that decode never had to state: query token `t` sits at
/// `kv_len - n_tok + t`, so token `t`'s selection must be drawn only from KV rows `<= that`. The
/// gather flash applies NO mask (`window = 0`, and the dense `keep` predicate is bypassed under
/// `GATHER`) precisely because "the selected set is assumed causal — the selector produced it". So
/// a T-row selector owns causality entirely; a flat top-k over all `ctx` for every query would let
/// early tokens attend to the future, and the flash would not catch it.
///
/// Until then this family's prefill emits the DENSE `FlashMlaPrefill` at every ctx, including with
/// the DSA gate armed. That is correct — it is the baseline the crossover was measured against —
/// and ctx-linear rather than ctx-constant, so the gather's win at long prompts is unclaimed.
///
/// The emitter side is ready: `n.iidx` would be declared `[T][index_topk]` by the same `rows`
/// widening every other prefill activation already goes through, and the flash emit is one branch
/// (`if dsa { FlashGatherPrefill } else { FlashMlaPrefill }` with `t7 = n.iidx`, `i6 = itk`) exactly
/// as the decode path already does.
fn _flash_gather_prefill_contract() {}

/// The prefill bucket ladder for the MLA family, capped at `ctx`.
///
/// Same rungs as the dense-GQA path's shipped ladder — a 20-token prompt must not pay for a
/// 4096-row program, and T is a compile-time constant of a packet, so the compiler emits several and
/// the runtime picks the smallest that fits. The rungs are not re-derived here: the wave-quantised
/// ladder (`PLOW_PF_LADDER=wave`) is tied to the sm_120 128x128 tile, and this family's prefill
/// object is gfx950-only, so guessing tread positions for it would be worse than the power-of-two
/// default until there is a measurement to place them.
pub(crate) fn glm_prefill_buckets(ctx: u32) -> Vec<u32> {
    [128u32, 512, 1024, 2048, 4096, 8192]
        .into_iter()
        .filter(|&x| x <= ctx)
        .collect()
}

/// Prefill buckets requested for the MLA `--block` emit, from the environment, plus the scope.
///
/// OFF by default, and deliberately: the prefill program stops at the post-attention norm (no FFN
/// kernel exists — see [`emit_glm_mla_prefill`]), so a blob carrying prefill buckets makes
/// `Engine::has_prefill()` true and flips the block harness onto a path whose output tensor the
/// program never writes. Opting in is how a bring-up run asks for the MLA prefill arm without
/// changing what every existing decode asset emits.
///
///   * unset / `0`      — decode-only, byte-identical to before this change;
///   * `1`              — attention-only buckets on the ladder above;
///   * `128,512`        — attention-only buckets on the given rungs;
///   * `full`           — whole-layer prefill (attention + FFN) on the whole ladder;
///   * `full:128,512`   — whole-layer prefill on the given rungs ONLY.
///
/// The scoped-list form exists because the ladder's top rungs are not free in DEVICE MEMORY: every
/// activation is declared for the WIDEST bucket, and `act.part` alone is `T * top_k * hidden` f32 —
/// 1.6 GiB at T=8192 on GLM-5.2. A run that will only ever see 1k prompts should not pay for the
/// 8192 rung, and before this form the only way to limit the ladder also silently downgraded the
/// scope to attention-only, which for a MODEL emit produces a blob that cannot sample.
pub(crate) fn glm_prefill_buckets_env(ctx: u32) -> (Vec<u32>, PrefillScope) {
    let parse_list = |list: &str| -> Vec<u32> {
        list.split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .filter(|&x| x > 1 && x <= ctx)
            .collect()
    };
    match emit_config::active().mla_prefill.as_deref() {
        None | Some("") | Some("0") => (Vec::new(), PrefillScope::Attn),
        Some("1") => (glm_prefill_buckets(ctx), PrefillScope::Attn),
        Some("full") => (glm_prefill_buckets(ctx), PrefillScope::Full),
        Some(s) if s.starts_with("full:") => (parse_list(&s["full:".len()..]), PrefillScope::Full),
        Some(list) => (parse_list(list), PrefillScope::Attn),
    }
}

/// Emit ONE MoE (sparse) GLM decoder block — the exact block validated by the B4 harness. `slot`
/// indexes `tn.lw`/`tn.ckv`/`tn.krot`. `use_fp8` selects the block-fp8 expert opcodes (45/46) over
/// the bf16 ones (41/42). Returns the MoeCombine completion dep.
/// The BATCHED-DECODE MoE FFN seam: `rows` sequences, one token each, routed through the
/// PREFILL grouped-expert family at `T = rows`.
///
/// Exists because the decode MoE ops carry no token dimension at all (`glm52-batched-decode-scope.md`
/// correction 3), and extending four decode kernels with one would also be the wrong shape: at
/// B=16 with top-8 of 256 experts the grouped form reads each touched expert's weights ONCE for
/// every row that chose it, which is the amortisation batching exists for. Direct precedent for
/// reusing the `*Pf` family on a non-prefill shape: `emit_glm_dense_block_prefill`'s degenerate
/// `n_exp = 1, top_k = 1` dense FFN.
///
/// Differences from the prefill seam, each deliberate:
///   * projections stay GEMV-family at `M = rows` (a 16-row GEMM tile would pad 8x; the GEMV
///     rung ladder is the object's decode shape and `check_gemv_capacity` gates the pairing);
///   * the TP fold is the decode ONE-SHOT `XReduce` over `rows*h` elements, not the row-banded
///     two-shot — banding exists to overlap an 8k-row chunk's fabric time, and at 16 rows the
///     band bookkeeping costs more than it hides;
///   * the `PLOW_MOE_PF_A8` gathered-fp8-A arm is NOT taken: its producer is the prefill
///     AddNorm's quant epilogue, which the decode chain does not emit.
#[allow(clippy::too_many_arguments)]
fn emit_glm_moe_ffn_rows(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    rows: u32,
    enc: MoeEnc,
    x_out: u32,
    c_rn2: u32,
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    let all = b.all();
    let one = vec![0u32];
    let (h, e, tk, imoe) = (c.hidden, c.n_exp, c.top_k, c.moe_inter);
    let tp = c.tp;
    let imoe_l = imoe / tp;
    let imoe_e = if c.ep { imoe } else { imoe_l };
    let w = &n.lw[slot];
    let lin_fp8 = glm_linear_fp8(enc);
    assert!(
        n.meta != TENSOR_NONE,
        "batched decode FFN needs the grouped-prefill scratch (meta/row_token/...) — \
         declare_glm_rows_batched gates it on max(rows, dbatch) > 1"
    );

    // Router score at M = rows, then the PREFILL top-k tail (the decode tail under a token loop,
    // bit-identical per token) and the align/sort that the grouped ops read.
    let score_op = if enc == MoeEnc::Mxfp4 {
        DevOp::GemvMxfp4
    } else {
        DevOp::Gemv
    };
    let c_score = b.emit(score_op, all.clone(), &[c_rn2], |d| {
        d.t[0] = n.rlogit;
        d.t[1] = n.xn2;
        d.t[2] = w.wr;
        if enc == MoeEnc::Mxfp4 {
            d.t[3] = w.wr_s;
        }
        d.i[0] = rows;
        d.i[1] = e;
        d.i[2] = h;
        d.f[0] = 1.0;
    });
    let fuse = moe_pf_fuse(tk);
    let atom = fuse == MoePfFuse::Atomic;
    let det = fuse == MoePfFuse::Det;
    let c_router = b.emit(DevOp::MoeRouterTopkPf, all.clone(), &[c_score], |d| {
        d.t[0] = n.tab;
        d.t[1] = n.rlogit;
        d.t[3] = w.bias;
        if atom || det {
            d.t[2] = n.part;
            d.i[0] = h;
        }
        d.i[1] = e;
        d.i[2] = tk;
        d.i[3] = GLM_ROUTER_FLAGS;
        d.i[4] = rows;
        d.i[6] = c.n_group;
        d.i[7] = c.topk_group;
        d.f[0] = c.route_scale;
    });
    let c_align = b.emit(DevOp::MoeAlignPf, one.clone(), &[c_router], |d| {
        d.t[0] = n.meta;
        d.t[1] = n.tab;
        d.t[2] = n.row_token;
        d.t[3] = n.row_partidx;
        d.t[4] = n.row_gate;
        d.i[0] = rows;
        d.i[1] = e;
        d.i[2] = tk;
    });

    // Shared expert at M = rows. The decode fused ops that cannot carry M (`DenseGluFp8Blk` is
    // i0=N) unfuse into their GEMV halves; the plain arm is `GemvGlu`, which takes M directly.
    let c_shglu = if lin_fp8 {
        let gemv_half = |b: &mut Builder, out: u32, wt: u32, ws: u32| {
            b.emit(DevOp::GemvFp8Blk, all.clone(), &[c_rn2], |d| {
                d.t[0] = out;
                d.t[1] = n.xn2;
                d.t[2] = wt;
                d.t[5] = ws;
                d.i[0] = rows;
                d.i[1] = imoe_l;
                d.i[2] = h;
                d.i[4] = 0;
            })
        };
        let c_g = gemv_half(b, n.shfu, w.shg, w.shg_s);
        let c_u = gemv_half(b, n.shfu_up, w.shu, w.shu_s);
        b.emit(DevOp::Glu, one.clone(), &[c_g, c_u], |d| {
            d.t[0] = n.shfu;
            d.t[1] = n.shfu;
            d.t[2] = n.shfu_up;
            d.i[0] = rows * imoe_l;
            d.i[1] = GLM_ACT_SILU;
        })
    } else if (rows as u64) * (h as u64) > crate::gm_lds_halves() {
        // TASK-9 FIT GATE, GemvGlu leg: gemv_glu_rows is in op_gemm.h's always-staged family
        // ("read x ONLY through LDS"), so the fused form carries the same rows*K promise the
        // GemvQkv gate above enforces. Past the fit, split into the two plain-Gemv halves +
        // Glu — the same shape the fp8 arm ships — Gemv's body has its own fit fallback and
        // is correct at any M. MXFP4 has no split precedent here; refuse loudly rather than
        // emit a fluent-but-wrong blob.
        assert!(
            enc != MoeEnc::Mxfp4,
            "shared-expert GemvGluMxfp4 at rows={rows} K={h} exceeds the staged-LDS fit; \
             no split form exists for the mxfp4 shared expert — cap the decode ladder"
        );
        let gemv_half = |b: &mut Builder, out: u32, wt: u32| {
            b.emit(DevOp::Gemv, all.clone(), &[c_rn2], |d| {
                d.t[0] = out;
                d.t[1] = n.xn2;
                d.t[2] = wt;
                d.i[0] = rows;
                d.i[1] = imoe_l;
                d.i[2] = h;
            })
        };
        let c_g = gemv_half(b, n.shfu, w.shg);
        let c_u = gemv_half(b, n.shfu_up, w.shu);
        b.emit(DevOp::Glu, one.clone(), &[c_g, c_u], |d| {
            d.t[0] = n.shfu;
            d.t[1] = n.shfu;
            d.t[2] = n.shfu_up;
            d.i[0] = rows * imoe_l;
            d.i[1] = GLM_ACT_SILU;
        })
    } else {
        let shglu_op = if enc == MoeEnc::Mxfp4 {
            DevOp::GemvGluMxfp4
        } else {
            DevOp::GemvGlu
        };
        b.emit(shglu_op, all.clone(), &[c_rn2], |d| {
            d.t[0] = n.shfu;
            d.t[1] = n.xn2;
            d.t[2] = w.shg;
            d.t[5] = w.shu;
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = w.shg_s;
                d.t[4] = w.shu_s;
            }
            d.i[0] = rows;
            d.i[1] = imoe_l;
            d.i[2] = h;
            d.i[5] = GLM_ACT_SILU;
        })
    };
    let c_shd = if lin_fp8 {
        b.emit(DevOp::GemvFp8Blk, all.clone(), &[c_shglu], |d| {
            d.t[0] = n.shared;
            d.t[1] = n.shfu;
            d.t[2] = w.shd;
            d.t[5] = w.shd_s;
            d.i[0] = rows;
            d.i[1] = h;
            d.i[2] = imoe_l;
            d.i[4] = 0;
        })
    } else {
        let shd_op = if enc == MoeEnc::Mxfp4 {
            DevOp::GemvMxfp4
        } else {
            DevOp::Gemv
        };
        b.emit(shd_op, all.clone(), &[c_shglu], |d| {
            d.t[0] = n.shared;
            d.t[1] = n.shfu;
            d.t[2] = w.shd;
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = w.shd_s;
            }
            d.i[0] = rows;
            d.i[1] = h;
            d.i[2] = imoe_l;
            d.f[0] = 1.0;
        })
    };

    // Grouped gate/up + down over the sorted (row, expert) slots — the prefill pair verbatim
    // at T = rows, minus the A8 arm (see the header).
    let shuf = w.ewt_pf != TENSOR_NONE;
    let c_g = b.emit(DevOp::MoeGroupGluPf, all.clone(), &[c_align, c_rn2], |d| {
        d.t[0] = n.fu_g;
        d.t[1] = n.xn2;
        d.t[2] = if shuf { w.ewt_pf } else { w.ewt };
        d.t[3] = w.est;
        d.t[4] = n.meta;
        d.t[5] = n.row_token;
        d.i[6] = u32::from(shuf);
        if enc == MoeEnc::Mxfp4 {
            d.t[6] = n.row_partidx;
            d.t[7] = n.fu_scale;
        }
        d.i[0] = imoe_e;
        d.i[1] = h;
        d.i[2] = e;
        d.i[MoeEnc::PREFILL_SLOT] = enc.code();
        d.i[5] = GLM_ACT_SILU;
    });
    let c_d = b.emit(DevOp::MoeGroupDownPf, all.clone(), &[c_g], |d| {
        d.t[0] = n.part;
        d.t[1] = n.fu_g;
        d.t[2] = if shuf { w.ewt_pf } else { w.ewt };
        d.t[3] = w.est;
        d.t[4] = n.meta;
        if enc == MoeEnc::Mxfp4 {
            d.t[5] = n.fu_scale;
        }
        d.t[6] = n.row_partidx;
        d.t[7] = n.row_gate;
        d.i[0] = h;
        d.i[1] = imoe_e;
        d.i[2] = e;
        d.i[MoeEnc::PREFILL_SLOT] = enc.code();
        d.i[7] = u32::from(moe_pf_part16());
        if atom {
            d.i[4] = tk.trailing_zeros() + 1;
        }
        if det {
            d.i[5] = tk.trailing_zeros() + 1;
        }
    });

    // Combine + the decode-shaped TP seam: one-shot XReduce over rows*h, then the layer-seam
    // AddNorm (or Residual), exactly as the single-row decode block ends.
    let no_xr = tp > 1 && emit_config::active().no_xreduce;
    if tp > 1 && !no_xr {
        let c_cmb = b.emit(DevOp::MoeCombinePf, all.clone(), &[c_shd, c_d], |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = TENSOR_NONE; // residual rides AFTER the all-reduce (else summed tp times)
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = if atom || det { 1 } else { tk };
            d.i[2] = rows;
            d.i[3] = 0;
            d.i[4] = u32::from(det);
            d.i[7] = u32::from(moe_pf_part16());
        });
        let c_xr = emit_xreduce(
            b,
            xgate,
            true,
            xr_cus,
            c_cmb,
            n.attn,
            rows * h,
            tp,
            n.slot_b,
        );
        if let Some(gin_next) = seam_next_gin(n, slot, tp) {
            b.emit(DevOp::AddNorm, one.clone(), &[c_xr], |d| {
                d.t[0] = n.xn;
                d.t[1] = x_out;
                d.t[2] = n.xmid;
                d.t[3] = n.attn;
                d.t[4] = gin_next;
                d.i[0] = rows;
                d.i[1] = h;
                d.f[0] = c.eps;
            })
        } else {
            b.emit(DevOp::Residual, spine_cus(b.n_cu()), &[c_xr], |d| {
                d.t[0] = x_out;
                d.t[1] = n.xmid;
                d.t[2] = n.attn;
                d.i[0] = rows * h;
                d.f[0] = 1.0;
            })
        }
    } else {
        b.emit(DevOp::MoeCombinePf, all.clone(), &[c_shd, c_d], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = if atom || det { 1 } else { tk };
            d.i[2] = rows;
            d.i[3] = 0;
            d.i[4] = u32::from(det);
            d.i[7] = u32::from(moe_pf_part16());
        })
    }
}

pub(crate) fn emit_glm_block(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    rows: u32,
    dbatch: u32,
    enc: MoeEnc,
    x_in: u32,
    x_out: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    assert!(slot < n.lw.len(), "slot out of range");
    // The MXFP4 arm lives INSIDE the block-fp8 kernels (`d_moe_expert_glu_fp8_blk` dispatches on
    // the encoding), so MXFP4 rides the same opcodes 45/46/48/49 that block-fp8 does and differs
    // only in `i[6]`. bf16 keeps the separate 41/42 opcodes. `use_fp8` therefore means "the
    // scale-table-carrying opcode pair", which is true for both quantized encodings.
    let use_fp8 = enc != MoeEnc::Bf16;
    let lin_fp8 = glm_linear_fp8(enc);
    let glu_split = glm_shared_glu_split(enc);
    let c_rn2 = emit_glm_mla(
        b, c, n, slot, ctx, rows, dbatch, enc, x_in, pre, xgate, xr_cus,
    );
    if rows > 1 {
        // BATCHED DECODE FFN: the decode MoE op family carries no token dimension
        // (`MoeRouterTopk`/`MoeGroupGluFp8Blk`/`MoeGroupDownFp8Blk`/`MoeCombine` are all
        // single-row), so at rows > 1 the seam is emitted with the PREFILL family at T = rows —
        // the grouped form also happens to be the right shape: each touched expert's weights
        // cross HBM once for all the rows that chose it. Same move as the dense twin's
        // `emit_glm_dense_block_prefill` degenerate routing.
        return emit_glm_moe_ffn_rows(b, c, n, slot, rows, enc, x_out, c_rn2, xgate, xr_cus);
    }
    let all = b.all();
    let one = vec![0u32];
    let (h, e, tk, imoe) = (c.hidden, c.n_exp, c.top_k, c.moe_inter);
    let tp = c.tp;
    let imoe_l = imoe / tp; // this rank's SHARED-expert intermediate lanes (TP-sharded); tp==1 => imoe
                            // Routed-expert intermediate width: full moe_inter under EP (whole experts distributed across
                            // ranks — no CU-starve), else the TP shard. Under EP the host binds LOCAL experts (256/tp) whole,
                            // NULL for remote, and the kernel skips a null base; the combine XReduce folds the per-rank whole-
                            // expert partials in the same collective that already sums the shared partials.
    let imoe_e = if c.ep { imoe } else { imoe_l };
    let w = &n.lw[slot];
    // CONCURRENT EXPERT SEGMENTS: the M=1 experts underfill 256 CUs
    // (latency-starved, ~12x above the weight-bandwidth roofline), so run the top_k chosen experts as
    // CO-RESIDENT segments — each owns a DISJOINT CU slice (tk experts x 256/tk CUs), all gated on the
    // SAME router counter, so all tk run at once instead of serially on all-256. Pure work-PARTITION
    // change (the kernel's slice/nblk mechanism does the rest): 0 = serial all-256 baseline, 1 =
    // concurrent experts (shared serial), 2 = concurrent experts + co-resident (proactive) shared expert.
    // SHIP DEFAULT = 1 (co-resident experts): bit-exact, measured -17.4% on the MoE block (the M=1
    // experts collapse from serial-all-256 to tk concurrent 256/tk-CU segments). GLM_MOE_CORESIDENT=0
    // restores the serial baseline; =2 adds the proactive co-resident shared expert (marginal, opt-in).
    let cores: u32 = emit_config::active().glm_moe_coresident.unwrap_or(1);
    // Under cores>=2 the shared expert gets its own slice (parts = tk+1, slot tk), concurrent with the
    // tk routed experts; else it stays on all-256 (serial, ahead of the experts in the stream).
    //
    // GLM_SHARED_CUS makes that slice's WIDTH a knob instead of a consequence of `split(tk+1, ·)`.
    // `split` floors, so `tk+1 = 9` parts of 256 gives the tk routed experts 28 CUs each and the
    // shared expert the 32-CU remainder — the co-resident arrangement is ALREADY non-uniform, and
    // 28 is what costs the routed experts their even 2-channels-per-wave fill
    // (perf-data/glm52-kernel-review.md §4: GLU 24.72 -> 19.58 us, DOWN 25.94 -> 22.72 us at 32 CU).
    // The default below reproduces `split(tk+1, ·)` EXACTLY, so nothing moves unless asked.
    let n_cu = b.n_cu();
    let shared_w: u32 = emit_config::active()
        .glm_shared_cus
        .filter(|&s| s > 0 && s < n_cu)
        .unwrap_or(n_cu - tk * (n_cu / (tk + 1)));
    let routed_w = (n_cu - shared_w) / tk;
    assert!(
        routed_w > 0,
        "GLM_SHARED_CUS={shared_w} leaves no CUs for the {tk} routed experts"
    );
    let shared_cus: Vec<u32> = if cores >= 2 {
        ((n_cu - shared_w)..n_cu).collect()
    } else {
        all.clone()
    };
    // Per-slot routed-expert CU set: disjoint 1/tk (cores 1) or the `routed_w`-wide slice below the
    // shared expert's (cores 2), else all-256.
    let expert_cus = |b: &Builder, sl: u32| -> Vec<u32> {
        if cores >= 2 {
            (sl * routed_w..(sl + 1) * routed_w).collect()
        } else if cores >= 1 {
            b.split(tk, sl)
        } else {
            (0..n_cu).collect()
        }
    };
    // GLM_ROUTER_OFF_SHARED=1: keep the router score GEMV OFF the shared expert's slice. The router
    // is emitted before the shared expert and runs on all 256 CUs, so under cores>=2 the shared
    // expert — which is gated only on c_rn2 and could start immediately — actually waits for the
    // router's workgroups on ITS OWN CUs to retire. The router GEMV is N=256 outputs over 2048
    // waves (7/8 of them idle, 4.4% of the bandwidth ceiling), so narrowing it to the routed-expert
    // CUs costs nothing it was using and lets the shared expert start ~a router earlier.
    //
    // MEASURED: **+0.12 ms, i.e. nothing or slightly worse** (full model, TP4, ctx 1k, median of
    // 65 — perf-data/glm52-decode-emitter-abs.md §3). Kept as the instrument that priced it. Do not
    // re-propose this as a lever.
    let router_cus: Vec<u32> = if cores >= 2 && emit_config::active().glm_router_off_shared {
        (0..n_cu - shared_w).collect()
    } else {
        all.clone()
    };

    // --- MoE ---
    // 15 router. DEFAULT (split): the 256-expert x K=6144 score matmul is the ordinary MULTI-CU
    //   wave-cooperative GEMV (all.clone()) — was the single-CU scalar dot that measured 73% of the
    //   MoE layer — feeding a cheap 1-CU MoeRouterTopk tail (bit-exact selection). GLM_ROUTER_OLD=1
    //   emits the fused single-CU d_moe_router for the before/after A/B.
    let c_router = if emit_config::active().glm_router_old {
        b.emit(DevOp::MoeRouter, one.clone(), &[c_rn2], |d| {
            d.t[0] = n.tab;
            d.t[1] = n.xn2;
            d.t[2] = w.wr;
            d.t[3] = w.bias;
            d.i[0] = h;
            d.i[1] = e;
            d.i[2] = tk;
            d.i[3] = GLM_ROUTER_FLAGS;
            d.f[0] = c.route_scale;
        })
    } else {
        let score_op = if enc == MoeEnc::Mxfp4 {
            DevOp::GemvMxfp4
        } else {
            DevOp::Gemv
        };
        let c_score = b.emit(score_op, router_cus.clone(), &[c_rn2], |d| {
            d.t[0] = n.rlogit;
            d.t[1] = n.xn2;
            d.t[2] = w.wr;
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = w.wr_s;
            }
            d.i[0] = 1;
            d.i[1] = e;
            d.i[2] = h;
            d.f[0] = 1.0;
        });
        b.emit(DevOp::MoeRouterTopk, one.clone(), &[c_score], |d| {
            d.t[0] = n.tab;
            d.t[1] = n.rlogit;
            d.t[3] = w.bias;
            d.i[1] = e;
            d.i[2] = tk;
            d.i[3] = GLM_ROUTER_FLAGS;
            // GROUP-LIMITED top-k (noaux_tc). Inert at n_group<=1, which is every GLM/Qwen/Mixtral
            // packet — so this stays byte-identical there and fixes Kimi/DeepSeek, which group.
            d.i[6] = c.n_group;
            d.i[7] = c.topk_group;
            d.f[0] = c.route_scale;
        })
    };
    // 16 shared expert gate|up (fused GLU) — column-parallel: this rank's imoe_l lanes. Under cores>=2
    //   it runs on its OWN slice (shared_cus), CO-RESIDENT with the routed experts (it is routing-
    //   independent — gated only on c_rn2 — so it overlaps the expert chain instead of preceding it).
    // The shared expert HAS a fused MXFP4 GLU arm at decode (op 92) — unlike prefill, where there is
    // no GemmGluMxfp4 and the pair has to unfuse. Same operand slots, plus the two E8M0 scale rows.
    // GLM_LINEAR_FP8: the shared expert is three whole block-fp8 checkpoint tensors the prep
    // dequantises. On the fp8 arm gate/up go through DENSE_GLU_FP8_BLK (47) and down through
    // GEMV_FP8_BLK (44) — the exact pair `emit_glm_dense_block` already uses for the dense FFN,
    // same operand slots (t2/t5 weights, t3/t4 grids on 47; t5 grid on 44) and same i[] meanings.
    // Op 47 is i0=N i1=K where GemvGlu is i0=M i1=N i2=K, which is why this is a separate emit.
    let c_shglu = if glu_split {
        // GLM_SHARED_GLU_SPLIT (see `glm_shared_glu_split`): op 47 leaves 192 of 256 workgroups
        // empty at N=imoe_l and keeps one load in flight per wave. Run gate and up as two
        // GEMV_FP8_BLK on DISJOINT halves of the shared slice — same weights, same scale grids, no
        // concatenation on disk — so both halves are in flight at once at 4 waves/CU, then a
        // one-workgroup Glu folds them. Operand slots are the shared DOWN's exactly (t2 weight,
        // t5 [128,128] f32 grid; i0=M i1=N i2=K i4=x_row).
        let (sh_g, sh_u) = glm_glu_halves(&shared_cus);
        let gemv_half = |b: &mut Builder, cus: Vec<u32>, out: u32, wt: u32, ws: u32| {
            b.emit(DevOp::GemvFp8Blk, cus, &[c_rn2], |d| {
                d.t[0] = out;
                d.t[1] = n.xn2;
                d.t[2] = wt;
                d.t[5] = ws;
                d.i[0] = 1;
                d.i[1] = imoe_l;
                d.i[2] = h;
                d.i[4] = 0;
            })
        };
        let c_g = gemv_half(b, sh_g, n.shfu, w.shg, w.shg_s);
        let c_u = gemv_half(b, sh_u, n.shfu_up, w.shu, w.shu_s);
        // On the SHARED slice's first CU, not `one` (= CU 0). Under `GLM_MOE_CORESIDENT=2` the
        // shared expert owns the TOP of the machine and CU 0 belongs to a routed expert, so a Glu
        // on CU 0 would put the shared chain behind a routed expert's queue.
        b.emit(DevOp::Glu, shared_cus[..1].to_vec(), &[c_g, c_u], |d| {
            d.t[0] = n.shfu;
            d.t[1] = n.shfu;
            d.t[2] = n.shfu_up;
            d.i[0] = imoe_l;
            d.i[1] = GLM_ACT_SILU;
        })
    } else if lin_fp8 {
        b.emit(DevOp::DenseGluFp8Blk, shared_cus.clone(), &[c_rn2], |d| {
            d.t[0] = n.shfu;
            d.t[1] = n.xn2;
            d.t[2] = w.shg;
            d.t[5] = w.shu;
            d.t[3] = w.shg_s;
            d.t[4] = w.shu_s;
            d.i[0] = imoe_l;
            d.i[1] = h;
            d.i[5] = GLM_ACT_SILU;
        })
    } else {
        let shglu_op = if enc == MoeEnc::Mxfp4 {
            DevOp::GemvGluMxfp4
        } else {
            DevOp::GemvGlu
        };
        b.emit(shglu_op, shared_cus.clone(), &[c_rn2], |d| {
            d.t[0] = n.shfu;
            d.t[1] = n.xn2;
            d.t[2] = w.shg;
            d.t[5] = w.shu;
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = w.shg_s;
                d.t[4] = w.shu_s;
            }
            d.i[0] = 1;
            d.i[1] = imoe_l;
            d.i[2] = h;
            d.i[5] = GLM_ACT_SILU;
        })
    };
    // 17 shared expert down — row-parallel (imoe_l input): writes a PARTIAL H-vector under TP
    let c_shd = if lin_fp8 {
        b.emit(DevOp::GemvFp8Blk, shared_cus.clone(), &[c_shglu], |d| {
            d.t[0] = n.shared;
            d.t[1] = n.shfu;
            d.t[2] = w.shd;
            d.t[5] = w.shd_s;
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = imoe_l;
            d.i[4] = 0;
        })
    } else {
        let shd_op = if enc == MoeEnc::Mxfp4 {
            DevOp::GemvMxfp4
        } else {
            DevOp::Gemv
        };
        b.emit(shd_op, shared_cus.clone(), &[c_shglu], |d| {
            d.t[0] = n.shared;
            d.t[1] = n.shfu;
            d.t[2] = w.shd;
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = w.shd_s;
            }
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = imoe_l;
            d.f[0] = 1.0;
        })
    };
    // 18..33 the top-8 routed experts (gate/up GLU then down). imoe_e = full moe_inter under EP (whole
    //   experts, host binds the LOCAL 256/tp experts + NULL for remote; the kernel skips a null base),
    //   else the imoe_l TP shard. Each expert's part[slot] is an H-vector partial the combine XReduce
    //   folds. c.group collapses the 2*tk per-slot packets into 2 grouped packets (ops 48/49, fp8 only).
    let downs: Vec<u32> = if c.group && use_fp8 {
        // ONE grouped gate/up packet + ONE grouped down packet (op-count collapse for M=1 decode).
        let c_g = b.emit(DevOp::MoeGroupGluFp8Blk, all.clone(), &[c_router], |d| {
            d.t[0] = n.fu;
            d.t[1] = n.xn2;
            d.t[2] = n.tab;
            d.t[3] = w.ewt;
            d.t[4] = w.est;
            d.i[0] = tk;
            d.i[1] = imoe_e;
            d.i[2] = h;
            d.i[3] = e;
            d.i[5] = GLM_ACT_SILU;
            d.i[MoeEnc::DECODE_SLOT] = enc.code(); // NOT the prefill slot — see MoeEnc::DECODE_SLOT
        });
        let c_d = b.emit(DevOp::MoeGroupDownFp8Blk, all.clone(), &[c_g], |d| {
            d.t[0] = n.part;
            d.t[1] = n.fu;
            d.t[2] = n.tab;
            d.t[3] = w.ewt;
            d.t[4] = w.est;
            d.i[0] = tk;
            d.i[1] = h;
            d.i[2] = imoe_e;
            d.i[3] = e;
            d.i[MoeEnc::DECODE_SLOT] = enc.code();
        });
        vec![c_d]
    } else {
        let (glu_op, down_op) = if use_fp8 {
            (DevOp::MoeExpertGluFp8Blk, DevOp::MoeExpertDownFp8Blk)
        } else {
            (DevOp::MoeExpertGlu, DevOp::MoeExpertDown)
        };
        let mut downs = Vec::with_capacity(tk as usize);
        for sl in 0..tk {
            // cores 0: all-256 (serial). cores 1: 256/tk each. cores 2: `routed_w` each, with the
            //   shared expert co-resident on the remainder. All gated on c_router, so they run
            //   concurrently on disjoint CU sets.
            let ecus = expert_cus(b, sl);
            let c_g = b.emit(glu_op, ecus.clone(), &[c_router], |d| {
                d.t[0] = n.fu;
                d.t[1] = n.xn2;
                d.t[2] = n.tab;
                d.t[3] = w.ewt;
                if use_fp8 {
                    d.t[4] = w.est;
                }
                d.i[0] = sl;
                d.i[1] = imoe_e;
                d.i[2] = h;
                d.i[3] = e;
                d.i[5] = GLM_ACT_SILU;
                d.i[MoeEnc::DECODE_SLOT] = enc.code();
            });
            let c_d = b.emit(down_op, ecus, &[c_g], |d| {
                d.t[0] = n.part;
                d.t[1] = n.fu;
                d.t[2] = n.tab;
                d.t[3] = w.ewt;
                if use_fp8 {
                    d.t[4] = w.est;
                }
                d.i[0] = sl;
                d.i[1] = h;
                d.i[2] = imoe_e;
                d.i[3] = e;
                d.i[MoeEnc::DECODE_SLOT] = enc.code();
            });
            downs.push(c_d);
        }
        downs
    };
    // 34 combine: sum shared + Σ gate·expert (f32 acc, fixed slot order). Under TP shared/part are
    //   PARTIALS, so the combine residual must NOT be xmid (it would be summed N times by XReduce);
    //   it writes the partial (residual = zero_h) into dg_tp, XReduce all-reduces into n.attn, and a
    //   Residual then adds the real xmid -> x_out. tp==1 keeps the fused xmid combine (byte-identical).
    let mut deps = Vec::with_capacity(1 + downs.len());
    deps.push(c_shd);
    deps.extend_from_slice(&downs);
    let no_xr = tp > 1 && emit_config::active().no_xreduce;
    if tp > 1 && !no_xr {
        let c_cmb = b.emit(DevOp::MoeCombine, elem_cus(&all, h), &deps, |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = n.zero_h;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
        });
        // partial_A = og_tp @ 0, partial_B = dg_tp @ n.slot_b. Blob-wide, so a decode program
        // sharing a tensor table with prefill buckets names the SAME offset they do.
        let c_xr = emit_xreduce(b, xgate, true, xr_cus, c_cmb, n.attn, h, tp, n.slot_b);
        // LAYER-SEAM FOLD: one AddNorm writes the residual stream AND the next layer's normed
        // input, deleting the next block's `RmsNorm` packet. See `glm_fuse_seam`.
        if let Some(gin_next) = seam_next_gin(n, slot, tp) {
            b.emit(DevOp::AddNorm, one.clone(), &[c_xr], |d| {
                d.t[0] = n.xn; // the NEXT layer's input_layernorm output
                d.t[1] = x_out; // the residual stream, unchanged (still bf16)
                d.t[2] = n.xmid;
                d.t[3] = n.attn;
                d.t[4] = gin_next;
                d.i[0] = 1;
                d.i[1] = h;
                d.f[0] = c.eps;
            })
        } else {
            b.emit(DevOp::Residual, spine_cus(b.n_cu()), &[c_xr], |d| {
                d.t[0] = x_out;
                d.t[1] = n.xmid;
                d.t[2] = n.attn;
                d.i[0] = h;
                d.f[0] = 1.0;
            })
        }
    } else if tp > 1 && no_xr {
        // diagnostic: combine this rank's partials straight onto the residual, no all-reduce
        b.emit(DevOp::MoeCombine, elem_cus(&all, h), &deps, |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
        })
    } else {
        b.emit(DevOp::MoeCombine, elem_cus(&all, h), &deps, |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
        })
    }
}

/// The dense-FFN down projection's opcode for an encoding.
fn dense_down_op(enc: MoeEnc) -> DevOp {
    match enc {
        MoeEnc::Bf16 => DevOp::Gemv,
        MoeEnc::Fp8Blk => DevOp::GemvFp8Blk,
        MoeEnc::Mxfp4 => DevOp::GemvMxfp4,
    }
}

/// Emit ONE DENSE (first_k_dense_replace) GLM decoder block — layers 0-2. The MLA attention is
/// identical to the MoE block; the FFN is a straight block-fp8 SwiGLU (no router/experts/shared):
/// DENSE_GLU_FP8_BLK (gate/up, H->dense_inter) -> GEMV_FP8_BLK (down, dense_inter->H) -> residual.
/// Returns the final residual completion dep (writes `n.xnext`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_glm_dense_block(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    rows: u32,
    dbatch: u32,
    enc: MoeEnc,
    x_in: u32,
    x_out: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    assert!(slot < n.lw.len(), "slot out of range");
    let c_rn2 = emit_glm_mla(
        b, c, n, slot, ctx, rows, dbatch, enc, x_in, pre, xgate, xr_cus,
    );
    if rows > 1 {
        // BATCHED DECODE dense FFN. `DenseGluFp8Blk` (op 47) is i0=N — it has no M axis at all —
        // so at rows > 1 the FFN takes the same degenerate grouped-`*Pf` route the dense PREFILL
        // does (`emit_glm_dense_block_prefill`: n_exp = 1, top_k = 1, align synthesises the
        // routing), with the decode-shaped one-shot TP seam.
        return emit_glm_dense_ffn_rows(b, c, n, slot, rows, enc, x_out, c_rn2, xgate, xr_cus);
    }
    let all = b.all();
    let (h, di) = (c.hidden, c.dense_inter);
    let tp = c.tp;
    let di_l = di / tp; // this rank's dense-FFN intermediate lanes; tp==1 => di
    let w = &n.lw[slot];
    // dense SwiGLU gate|up (block-fp8, op 47) — column-parallel: this rank's di_l lanes
    // Under MXFP4 the dense SwiGLU becomes the w4a16 fused GLU (op 92) — same operand slots, and
    // the E8M0 rows land in t3/t4 exactly where the block-fp8 grids did. i[0]/i[1] swap meaning
    // between the two ops (op 47 is i0=N i1=K; op 92 is i0=M i1=N i2=K), which is why this is a
    // separate emit rather than an opcode substitution.
    let c_glu = match enc {
        MoeEnc::Mxfp4 => b.emit(DevOp::GemvGluMxfp4, all.clone(), &[c_rn2], |d| {
            d.t[0] = n.dfu;
            d.t[1] = n.xn2;
            d.t[2] = w.dgate;
            d.t[5] = w.dup;
            d.t[3] = w.dgate_s;
            d.t[4] = w.dup_s;
            d.i[0] = 1;
            d.i[1] = di_l;
            d.i[2] = h;
            d.i[5] = GLM_ACT_SILU;
        }),
        MoeEnc::Fp8Blk => b.emit(DevOp::DenseGluFp8Blk, all.clone(), &[c_rn2], |d| {
            d.t[0] = n.dfu;
            d.t[1] = n.xn2;
            d.t[2] = w.dgate;
            d.t[5] = w.dup;
            d.t[3] = w.dgate_s;
            d.t[4] = w.dup_s;
            d.i[0] = di_l;
            d.i[1] = h;
            d.i[5] = GLM_ACT_SILU;
        }),
        MoeEnc::Bf16 => b.emit(DevOp::GemvGlu, all.clone(), &[c_rn2], |d| {
            d.t[0] = n.dfu;
            d.t[1] = n.xn2;
            d.t[2] = w.dgate;
            d.t[5] = w.dup;
            d.i[0] = 1;
            d.i[1] = di_l;
            d.i[2] = h;
            d.i[5] = GLM_ACT_SILU;
        }),
    };
    // dense down (block-fp8 GEMV, op 44) — row-parallel (di_l input). Under TP writes a PARTIAL into
    //   the dg_tp peer slot, XReduce all-reduces into n.attn, then residual; at tp==1 writes n.shared
    //   and the residual reads it directly (byte-identical).
    let no_xr = tp > 1 && emit_config::active().no_xreduce;
    if tp > 1 && !no_xr {
        let c_down = b.emit(dense_down_op(enc), all.clone(), &[c_glu], |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = n.dfu;
            d.t[2] = w.ddown;
            // GEMV_FP8_BLK reads its scale from t5; GEMV_MXFP4 reads E8M0 from t3. Different slot,
            // so bind by encoding rather than assuming the quantized arms agree.
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = w.ddown_s;
            } else {
                d.t[5] = w.ddown_s;
            }
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = di_l;
            d.i[4] = 0;
        });
        let c_xr = emit_xreduce(b, xgate, true, xr_cus, c_down, n.attn, h, tp, n.slot_b);
        // LAYER-SEAM FOLD, dense-FFN twin of the MoE tail's. Same packet, same argument.
        if let Some(gin_next) = seam_next_gin(n, slot, tp) {
            b.emit(DevOp::AddNorm, vec![0u32], &[c_xr], |d| {
                d.t[0] = n.xn;
                d.t[1] = x_out;
                d.t[2] = n.xmid;
                d.t[3] = n.attn;
                d.t[4] = gin_next;
                d.i[0] = 1;
                d.i[1] = h;
                d.f[0] = c.eps;
            })
        } else {
            b.emit(DevOp::Residual, spine_cus(b.n_cu()), &[c_xr], |d| {
                d.t[0] = x_out;
                d.t[1] = n.xmid;
                d.t[2] = n.attn;
                d.i[0] = h;
                d.f[0] = 1.0;
            })
        }
    } else if tp > 1 && no_xr {
        let c_down = b.emit(dense_down_op(enc), all.clone(), &[c_glu], |d| {
            d.t[0] = n.shared;
            d.t[1] = n.dfu;
            d.t[2] = w.ddown;
            // GEMV_FP8_BLK reads its scale from t5; GEMV_MXFP4 reads E8M0 from t3. Different slot,
            // so bind by encoding rather than assuming the quantized arms agree.
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = w.ddown_s;
            } else {
                d.t[5] = w.ddown_s;
            }
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = di_l;
            d.i[4] = 0;
        });
        b.emit(DevOp::Residual, spine_cus(b.n_cu()), &[c_down], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
    } else {
        let c_down = b.emit(dense_down_op(enc), all.clone(), &[c_glu], |d| {
            d.t[0] = n.shared;
            d.t[1] = n.dfu;
            d.t[2] = w.ddown;
            // GEMV_FP8_BLK reads its scale from t5; GEMV_MXFP4 reads E8M0 from t3. Different slot,
            // so bind by encoding rather than assuming the quantized arms agree.
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = w.ddown_s;
            } else {
                d.t[5] = w.ddown_s;
            }
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = di_l;
            d.i[4] = 0;
        });
        b.emit(DevOp::Residual, spine_cus(b.n_cu()), &[c_down], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
    }
}

/// The BATCHED-DECODE dense FFN: `rows` sequences through the degenerate grouped-`*Pf` route
/// (n_exp = 1, top_k = 1) — see [`emit_glm_dense_block_prefill`] for why that route exists and
/// [`emit_glm_moe_ffn_rows`] for the decode-shaped seam conventions this follows.
#[allow(clippy::too_many_arguments)]
fn emit_glm_dense_ffn_rows(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    rows: u32,
    enc: MoeEnc,
    x_out: u32,
    c_rn2: u32,
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    let all = b.all();
    let one = vec![0u32];
    let (h, di) = (c.hidden, c.dense_inter);
    let tp = c.tp;
    let di_l = di / tp;
    let w = &n.lw[slot];
    assert!(
        w.dwt != TENSOR_NONE,
        "batched dense decode needs the dense weight-pointer table; declare_glm_rows_batched \
         emits it for max(rows, dbatch) > 1"
    );
    assert!(
        enc != MoeEnc::Mxfp4,
        "batched dense decode is bf16/block-fp8 only (enc={enc:?}), same bound as the dense \
         prefill twin"
    );
    const DENSE_N_EXP: u32 = 1;
    const DENSE_TOP_K: u32 = 1;
    let c_align = b.emit(DevOp::MoeAlignPf, one.clone(), &[c_rn2], |d| {
        d.t[0] = n.meta;
        d.t[1] = TENSOR_NONE;
        d.t[2] = n.row_token;
        d.t[3] = n.row_partidx;
        d.t[4] = n.row_gate;
        d.i[0] = rows;
        d.i[1] = DENSE_N_EXP;
        d.i[2] = DENSE_TOP_K;
    });
    let c_g = b.emit(DevOp::MoeGroupGluPf, all.clone(), &[c_align, c_rn2], |d| {
        d.t[0] = n.fu_g;
        d.t[1] = n.xn2;
        d.t[2] = w.dwt;
        d.t[3] = w.dst;
        d.t[4] = n.meta;
        d.t[5] = n.row_token;
        d.i[0] = di_l;
        d.i[1] = h;
        d.i[2] = DENSE_N_EXP;
        d.i[MoeEnc::PREFILL_SLOT] = enc.code();
        d.i[5] = GLM_ACT_SILU;
    });
    let c_d = b.emit(DevOp::MoeGroupDownPf, all.clone(), &[c_g], |d| {
        d.t[0] = n.part;
        d.t[1] = n.fu_g;
        d.t[2] = w.dwt;
        d.t[3] = w.dst;
        d.t[4] = n.meta;
        d.t[6] = n.row_partidx;
        d.t[7] = n.row_gate;
        d.i[0] = h;
        d.i[1] = di_l;
        d.i[2] = DENSE_N_EXP;
        d.i[MoeEnc::PREFILL_SLOT] = enc.code();
        d.i[7] = u32::from(moe_pf_part16());
    });
    let no_xr = tp > 1 && emit_config::active().no_xreduce;
    if tp > 1 && !no_xr {
        let c_cmb = b.emit(DevOp::MoeCombinePf, all.clone(), &[c_d], |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = TENSOR_NONE;
            d.t[2] = TENSOR_NONE; // no shared expert on a dense layer
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = DENSE_TOP_K;
            d.i[2] = rows;
            d.i[7] = u32::from(moe_pf_part16());
        });
        let c_xr = emit_xreduce(
            b,
            xgate,
            true,
            xr_cus,
            c_cmb,
            n.attn,
            rows * h,
            tp,
            n.slot_b,
        );
        if let Some(gin_next) = seam_next_gin(n, slot, tp) {
            b.emit(DevOp::AddNorm, one.clone(), &[c_xr], |d| {
                d.t[0] = n.xn;
                d.t[1] = x_out;
                d.t[2] = n.xmid;
                d.t[3] = n.attn;
                d.t[4] = gin_next;
                d.i[0] = rows;
                d.i[1] = h;
                d.f[0] = c.eps;
            })
        } else {
            b.emit(DevOp::Residual, spine_cus(b.n_cu()), &[c_xr], |d| {
                d.t[0] = x_out;
                d.t[1] = n.xmid;
                d.t[2] = n.attn;
                d.i[0] = rows * h;
                d.f[0] = 1.0;
            })
        }
    } else {
        b.emit(DevOp::MoeCombinePf, all.clone(), &[c_d], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = TENSOR_NONE;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = DENSE_TOP_K;
            d.i[2] = rows;
            d.i[7] = u32::from(moe_pf_part16());
        })
    }
}

/// GLM-5.2 emit entry (Stack B: .pkt bound by name from the host-prepped weight dir). Milestone-1
/// emits the SINGLE-layer MoE block program the validation harness runs against the HF oracle; the
/// full 78-layer decode + dense layers + TP sharding are the next milestones.
/// Full 78-layer GLM-5.2 DECODE program (M=1): embed -> [dense 0-2 | MoE 3-77] ping-ponged -> final
/// norm -> lm_head -> argmax (writes the sampled id back into in.ids). Layers 0-77 (78 = MTP head,
/// skipped). Per-layer ckv/krot caches; the decode loop patches the current-token cache row per step
/// (k_rope HeadNormRope out_row0 via kv_row_insts; ckv RMSNORM output via a per-step pointer rebind).
/// `use_fp8` selects the block-fp8 expert kernels (45/46) for the MoE layers.
#[allow(clippy::too_many_arguments)]
fn glm_emit_full(
    dir: &Path,
    ctx: u32,
    out: &str,
    n_cu: u32,
    tp: u32,
    use_fp8: bool,
    rope_gen: bool,
    target: &str,
    l2_layout: Option<packet::devbuild::L2Layout>,
    verify: Option<&crate::VerifyHook>,
) {
    // The decode-batch ladder: one decode program per rung, ascending, after the prefill
    // buckets (`decode_rung_lo` separates the two ranges by width alone, so every rung must sit
    // strictly below the narrowest bucket — 32 < 128 holds by construction). Unset, this is
    // `[1]` and the emit is byte-identical to the pre-ladder one; that anchor is the gate the
    // batched path is validated against.
    let rungs: Vec<u32> = emit_config::active().decode_rungs();
    let dbatch: u32 = *rungs.last().expect("decode_rungs is non-empty");
    let mut c = cfg_glm(dir);
    c.tp = tp;
    // --glm-layers cap truncates the model to the first N layers — a single-GPU smoke test of the
    // decode LOOP mechanics (embed/chain/KV-row patch/argmax/multi-step) that fits without TP or
    // all 78 layers' weights. Default = full 0..77 (layer 78 = MTP, skipped).
    let (_full, cap, _single) = emit_config::active().glm_layer_cfg();
    let nl = cap.unwrap_or(c.layers).min(c.layers);
    let layers: Vec<u32> = (0..nl).collect();
    let enc = MoeEnc::from_flags(use_fp8, false);

    // PREFILL BUCKETS. `PLOW_MLA_PREFILL=full` turns the serving blob from decode-only (n_prog = 1)
    // into a bucket ladder + decode. Decode-only is what made GLM's TTFT 20x vLLM's: with no
    // prefill program the runtime walks the prompt through the DECODE program one token at a time
    // (`AmdServe::prefill`'s `decode_only` arm), i.e. one dispatch per prompt token.
    //
    // A MODEL emit only accepts the whole-layer scope. `Attn` stops at the post-attention norm and
    // never writes `act.logits`, so it would produce a blob whose prefill programs cannot sample —
    // and `Engine::has_prefill()` would still be true, so the runtime would USE them. Refuse.
    let (pf, scope) = glm_prefill_buckets_env(ctx);
    assert!(
        pf.is_empty() || scope == PrefillScope::Full,
        "GLM_FULL emits a whole MODEL: its prefill buckets must cover the whole layer \
         (PLOW_MLA_PREFILL=full). PLOW_MLA_PREFILL=1 is the attention-only scope, which ends at \
         post_attention_layernorm and never writes act.logits — the runtime would select those \
         programs and sample from a buffer nothing wrote."
    );
    let max_rows = pf.iter().copied().max().unwrap_or(1);

    let mut tb = Builder::new(n_cu);
    // Row-parameterised declare: activations are sized for the WIDEST bucket, so one tensor table
    // serves every program. max_rows == 1 reproduces `declare_glm` exactly, which is what keeps a
    // decode-only emit byte-identical to one from before this path existed. `dbatch` widens only
    // the KV rings, `in.kvlen`, the lm_head tail and the flash partials — see the declare's doc.
    let tn = declare_glm_rows_batched(&mut tb, &c, ctx, &layers, max_rows, dbatch, enc);
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();

    // One program per prefill bucket, ahead of decode. Same layer chain, same dense/MoE split, the
    // T-row emitters throughout; the lm_head tail samples the LAST row (see `emit_glm_tail`).
    let mut progs = Vec::new();
    let mut prog_t = Vec::new();
    for &t in &pf {
        let mut pb = Builder::new(n_cu);
        // PLOW_L2_PLACE reaches the DECODE builder unconditionally (below). GLM's prefill
        // program is uni-segment, so `Builder::finish` WOULD place it too — but the
        // shipped prefill objects are built without -DPLOW_L2_PLACE_DISPATCH
        // (build_gfx942.sh adds it to decode objects by default, to prefill objects only
        // under PLOW_L2HIER_PF=1) and the loader refuses a placed program on an unbuilt
        // object. PLOW_GLM_PLACE_PF=1 opts prefill placement in; pair it with objects
        // from a PLOW_L2HIER_PF=1 build.
        if emit_config::active().glm_place_pf {
            pb.set_l2_placement(l2_layout);
        }
        pb.adopt_tensors(tensors.clone());
        let pall = pb.all();
        // PLOW_XR_CUS caps the PREFILL two-shots too — see `xr_cus_capped`. This used to be
        // `pall.clone()` unconditionally, which pinned every prefill collective to 304
        // workgroups on the wrong side of the fabric's concurrency peak.
        let pxr: Vec<u32> = xr_cus_capped(n_cu, &pall);
        let mut pxgate = 0u32;
        let c_emb = pb.emit(DevOp::Embed, pall.clone(), &[], |d| {
            d.t[0] = tn.x;
            d.t[1] = tn.emb;
            d.t[2] = tn.ids;
            d.i[0] = t;
            d.i[1] = c.hidden;
            d.f[0] = 1.0;
        });
        let mut cur = tn.x;
        let mut dep = vec![c_emb];
        for (slot, &l) in layers.iter().enumerate() {
            let nxt = if cur == tn.x { tn.xnext } else { tn.x };
            let d = if c.is_dense(l) {
                emit_glm_dense_block_prefill(
                    &mut pb,
                    &c,
                    &tn,
                    slot,
                    ctx,
                    t,
                    enc,
                    cur,
                    nxt,
                    &dep,
                    &mut pxgate,
                    &pxr,
                )
            } else {
                emit_glm_block_prefill(
                    &mut pb,
                    &c,
                    &tn,
                    slot,
                    ctx,
                    t,
                    enc,
                    cur,
                    nxt,
                    &dep,
                    &mut pxgate,
                    &pxr,
                )
            };
            dep = vec![d];
            cur = nxt;
        }
        emit_glm_tail(&mut pb, &c, &tn, cur, &dep, t, false, &mut pxgate);
        progs.push(pb.finish());
        prog_t.push(t);
    }

    // The decode-rung separation `decode_rung_lo` depends on: every rung strictly below every
    // prefill bucket. 32 < 128 by construction, but assert like the Gemma ladder does — a future
    // bucket table edit must not silently fold a rung into the prefill range.
    if let Some(&min_bucket) = pf.iter().min() {
        assert!(
            dbatch < min_bucket,
            "decode rung {dbatch} >= narrowest prefill bucket {min_bucket}: \
             decode_rung_lo separates the ranges by width alone"
        );
    }
    // One decode program per rung, ascending. 78 decoder layers each, ping-ponging x <-> xnext so
    // layer l+1 reads layer l's output; each layer's first op waits on the previous layer's
    // completion (`dep`). XReduce collectives (decode one-shot): each o_proj + FFN-down
    // all-reduce takes a unique xctr gate id per program. At tp==1 no XReduce is emitted.
    let mut n_ops = 0;
    for &rb in &rungs {
        let mut b = Builder::new(n_cu);
        b.set_l2_placement(l2_layout); // PLOW_L2_PLACE: None ⇒ byte-identical
        b.adopt_tensors(tensors.clone());
        let all = b.all();

        // embed: in.ids[0..rb] -> x  (GLM has no embedding scale)
        let c_emb = b.emit(DevOp::Embed, all.clone(), &[], |d| {
            d.t[0] = tn.x;
            d.t[1] = tn.emb;
            d.t[2] = tn.ids;
            d.i[0] = rb;
            d.i[1] = c.hidden;
            d.f[0] = 1.0;
        });
        let mut xgate: u32 = 0;
        let xr_cus: Vec<u32> = xr_cus_capped(n_cu, &all);
        let mut cur = tn.x;
        let mut dep = c_emb;
        for (slot, &l) in layers.iter().enumerate() {
            let nxt = if cur == tn.x { tn.xnext } else { tn.x };
            dep = if c.is_dense(l) {
                emit_glm_dense_block(
                    &mut b,
                    &c,
                    &tn,
                    slot,
                    ctx,
                    rb,
                    dbatch,
                    MoeEnc::from_flags(use_fp8, false),
                    cur,
                    nxt,
                    &[dep],
                    &mut xgate,
                    &xr_cus,
                )
            } else {
                emit_glm_block(
                    &mut b,
                    &c,
                    &tn,
                    slot,
                    ctx,
                    rb,
                    dbatch,
                    MoeEnc::from_flags(use_fp8, false),
                    cur,
                    nxt,
                    &[dep],
                    &mut xgate,
                    &xr_cus,
                )
            };
            cur = nxt;
        }
        // final RMSNorm (model.norm) -> xn, then lm_head GEMV -> logits, greedy argmax -> in.ids.
        emit_glm_tail(&mut b, &c, &tn, cur, &[dep], rb, true, &mut xgate);
        let prog = b.finish();
        n_ops = prog.insts.len();
        progs.push(prog);
        prog_t.push(rb);
    }

    let mut m = Model {
        n_cu,
        target: 0, // GLM/Kimi/DeepSeek/Nemotron: GPU fingerprint not threaded here yet
        tensors,
        progs,
        kv_row_insts: Vec::new(),
        prog_t,
        gen,
    };
    if !rope_gen {
        m.bake_gen();
    }
    // PLOW_DECODE_BATCH IS SILENTLY DROPPED ON THIS PATH — SAY SO. The GLM emitter builds its
    // decode chain at one row (`declare_glm_rows(rows = 1)`) and never consults `decode_batch`,
    // so `PLOW_DECODE_BATCH=4` emits `decode M=1` and the operator gets an UNBATCHED blob that
    // looks exactly like a batched one from the outside.
    //
    // IGNORED RATHER THAN REFUSED, by this tree's own rule (`lib.rs` "REFUSE vs IGNORE is decided
    // by one question: what does the caller get if the flag is dropped?"): dropping it yields a
    // CORRECT packet, merely a single-row one, so refusing would deny the caller a blob they can
    // use. But it MUST be loud — `PLOW_L2_PLACE` sets that precedent on the dense-GQA path.
    //
    // Silence here is the expensive failure: someone sets the flag, measures concurrency, and
    // reports a batching result from a run that never batched. That is `LESSONS.md` §1 exactly —
    // "a null has two explanations: the work is free, or the work was never removed".
    let dec_batch = emit_config::active().decode_batch;
    if dec_batch > 1 {
        eprintln!(
            "  PLOW_DECODE_BATCH={dec_batch} IGNORED: the GLM decode chain is emitted at one row \
             and never reads it — this blob is `decode M=1`. Concurrency against it measures \
             QUEUEING, not batching. See task #24 / `glm52-experiments.md`."
        );
    }
    // Gate BEFORE the bytes land: a rejected program must never exist on disk.
    let lean = crate::apply_verify_gate(&m, verify);
    std::fs::write(out, m.to_blob()).unwrap();
    eprintln!(
        "glm52-FULL: {} layers (0-{}) hidden={} experts={}/top{} vocab={} {} -> {out}\n  \
         {n_ops} ops, decode M=1, ctx={ctx}, tp={tp}",
        layers.len(),
        layers.len().saturating_sub(1),
        c.hidden,
        c.n_exp,
        c.top_k,
        c.vocab,
        if use_fp8 { "block-fp8" } else { "bf16" }
    );
    report_mla_prefill(&m, &pf, scope, enc);
    write_mla_manifest(&m, out, target, enc, &lean);
}

/// The program tail shared by the decode program and every prefill bucket: final RMSNorm
/// (`model.norm`) -> lm_head GEMV -> greedy argmax -> `in.ids`.
///
/// `rows` is the bucket width. The norm runs over all `rows` rows (cheap, and the whole residual is
/// live), but the lm_head deliberately does NOT: sampling needs exactly ONE row — the LAST real one
/// — so `i[0]` stays M=1 and `i[4]` (`a_row0`) selects `rows - 1`. Emitting a [T, vocab] logit
/// matrix instead would cost a 152k-wide GEMM per prompt token to throw all but one row away.
///
/// `a_row0` is also the ONE field the host rewrites per chunk (`patch_prefill` sets it to
/// `clen - 1`), because a chunked prompt's last real row is not the bucket's last row. Baking
/// `rows - 1` here keeps a single-chunk prompt correct even if that patch never ran, and makes the
/// packet readable statically — the value is the right one, not a placeholder zero.
///
/// ## Why the sharded-lm_head fold lives HERE and is not gated to decode
///
/// Under `GLM_SHARD_HEAD` this rank owns `vocab/tp` logits, so the argmax over them is a LOCAL
/// winner and `ArgmaxFin` would write it straight into `in.ids` — four ranks, four different
/// tokens, no fault. That is true of EVERY program that argmaxes a sharded head, prefill buckets
/// included. So the fold is not a decode feature to be switched on per program: **sharding the
/// head and folding the maxima are one change**, and the only safe shapes are "replicated head +
/// `ArgmaxFin`" or "sharded head + `XArgmaxFin`". Gating the fold off in prefill would leave the
/// strictly worse of the two.
///
/// `rows > 1` is fine and does not touch the fold's batch bound: `rows` is a TOKEN axis, and the
/// lm_head collapses it to ONE logit row (`i[0] = 1`, `i[4] = rows - 1`) before the argmax ever
/// runs, so `n_batch` is 1 in every bucket. The batch axis is SEQUENCES, and prefill is
/// single-sequence.
///
/// `xgate` is the program's xctr id allocator, threaded in because each prefill bucket is its own
/// `Builder` with its own counter — the tail cannot invent ids without colliding with the layer
/// collectives of whichever program it was emitted into.
///
/// NOT YET MEASURED ON A PREFILL BUCKET: `GLM_SHARD_HEAD` is validated on the decode program
/// (256/256 tokens bit-identical vs the replicated head, 11 of them outside rank 0's shard). The
/// prefill tail is emitted by the same code and is structurally identical, but the TP4 prefill
/// path has not been run with a sharded head.
/// `dec_batch` flips `rows` from a TOKEN axis to a SEQUENCE axis: the prefill tail norms all
/// rows but samples only the LAST (`i[0]=1, a_row0=rows-1`); a batched decode tail produces one
/// logit row and one argmax PER ROW (`i[0]=rows, a_row0=0`, `n_batch=rows` on the argmax pair).
/// `rows == 1` is byte-identical on both settings, which is what the B=1 ladder rung's
/// re-emit gate leans on.
fn emit_glm_tail(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    x_final: u32,
    dep: &[u32],
    rows: u32,
    dec_batch: bool,
    xgate: &mut u32,
) {
    let all = b.all();
    // `vocab_l` is the full vocab when replicated and vocab/tp under GLM_SHARD_HEAD. The GEMV, the
    // argmax and the tensor declarations all read this same helper, so the arms cannot drift.
    let vocab_l = glm_vocab_l(c);
    // At prefill (`rows` = the bucket's T) this norms every row of the chunk even though the head
    // GEMV reads only the last — d_rmsnorm has no input row offset — so at least spread the rows
    // across the machine; decode (rows=1) keeps the single workgroup.
    let c_f = b.emit(DevOp::RmsNorm, pf_wide_cus(b.n_cu(), rows), dep, |d| {
        d.t[0] = n.xn;
        d.t[1] = x_final;
        d.t[2] = n.fin;
        d.i[0] = rows;
        d.i[1] = c.hidden;
        d.f[0] = c.eps;
    });
    // Batched decode: per-sequence logits. The Gemma reference for the argmax batching is
    // `lib.rs`' `nb_argmax` (`if decode && t > 1 { t } else { 0 }`) — 0 at one row keeps the
    // packet byte-identical to the pre-batch emit.
    let nb = if dec_batch && rows > 1 { rows } else { 0 };
    let c_lm = b.emit(DevOp::Gemv, all, &[c_f], |d| {
        d.t[0] = n.logits;
        d.t[1] = n.xn;
        d.t[2] = n.head;
        d.i[0] = if nb > 0 { rows } else { 1 };
        d.i[1] = vocab_l;
        d.i[2] = c.hidden;
        // a_row0: the last real row (host re-patches per chunk); 0 on the batched decode tail —
        // every row samples.
        d.i[4] = if nb > 0 { 0 } else { rows - 1 };
    });
    let c_am = b.emit(DevOp::Argmax, (0..AMAX_BLOCKS).collect(), &[c_lm], |d| {
        d.t[0] = n.amax;
        d.t[1] = n.logits;
        d.i[0] = vocab_l;
        d.i[1] = nb;
    });
    if glm_shard_head(c) {
        // XARGMAX_FIN SUBSUMES ArgmaxFin: it folds the AMAX_BLOCKS partials itself, rebases the
        // winning index by rank*vocab_l and takes the cross-rank max, so emitting both would fold
        // twice and write the LOCAL winner's id first. Two xctr ids from this program's allocator:
        // the arrival gate and the peer-visible 8-byte value slot — distinct, because the gate is
        // an atomic counter and the slot is data.
        // The fold publishes one u64 per sequence, 16 keys per 128-byte xctr counter line;
        // above 16 the value slot spans TWO consecutive ids (kernel: PLOW_XAMAX_LINE), so
        // the ceiling is 32. The extra id is allocated only when the rung needs it — a
        // narrow blob's id map is unchanged. Assert rather than leave ids[32..] holding the
        // previous step's token.
        const XAMAX_MAX_BATCH: u32 = 32;
        let n_batch = nb.max(1);
        assert!(
            n_batch <= XAMAX_MAX_BATCH,
            "XARGMAX_FIN carries at most {XAMAX_MAX_BATCH} sequences across two xctr counter \
             lines (asked for {n_batch}); cap the decode ladder at 32 under GLM_SHARD_HEAD"
        );
        let gate = *xgate;
        *xgate += if n_batch > 16 { 3 } else { 2 };
        b.emit(DevOp::XArgmaxFin, vec![0u32], &[c_am], |d| {
            d.t[0] = n.ids;
            d.t[1] = n.amax;
            d.i[0] = AMAX_BLOCKS;
            d.i[1] = n_batch;
            d.i[2] = vocab_l;
            d.i[3] = gate;
            d.i[4] = gate + 1;
        });
    } else {
        b.emit(DevOp::ArgmaxFin, vec![0u32], &[c_am], |d| {
            d.t[0] = n.ids;
            d.t[1] = n.amax;
            d.i[0] = AMAX_BLOCKS;
            d.i[1] = nb;
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_main(
    dir: &Path,
    ctx: u32,
    out: &str,
    n_cu: u32,
    tp: u32,
    rope_gen: bool,
    target: &str,
    l2_layout: Option<packet::devbuild::L2Layout>,
    verify: Option<&crate::VerifyHook>,
) {
    let enc = mla_moe_enc_env(dir);
    let use_fp8 = enc == MoeEnc::Fp8Blk;
    // Full 78-layer serving decode program (GLM_FULL=1) vs the single-layer validation gate (default).
    let (glm_full, _glm_cap, glm_single) = emit_config::active().glm_layer_cfg();
    if glm_full {
        glm_emit_full(
            dir, ctx, out, n_cu, tp, use_fp8, rope_gen, target, l2_layout, verify,
        );
        return;
    }
    let mut c = cfg_glm(dir);
    c.tp = tp;
    assert_eq!(
        tp, 1,
        "GLM TP sharding is milestone-3; use --tp 1 for the single-layer bring-up"
    );
    // Which layer to emit for the single-layer vs-HF gate (default = first MoE layer, matching the
    // B4 oracle's layer 3).
    let layer: u32 = glm_single.unwrap_or(c.first_k_dense);
    let dense = c.is_dense(layer);

    let mut tb = Builder::new(n_cu);
    let tn = declare_glm(&mut tb, &c, ctx, &[layer]);
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();
    let mut b = Builder::new(n_cu);
    b.adopt_tensors(tensors.clone());
    let mut xgate = 0u32; // tp==1 single-layer gate: no XReduce, so xgate/xr_cus are unused
    if dense {
        emit_glm_dense_block(
            &mut b,
            &c,
            &tn,
            0,
            ctx,
            1,
            1,
            MoeEnc::from_flags(use_fp8, false),
            tn.x,
            tn.xnext,
            &[],
            &mut xgate,
            &[],
        );
    } else {
        emit_glm_block(
            &mut b,
            &c,
            &tn,
            0,
            ctx,
            1,
            1,
            MoeEnc::from_flags(use_fp8, false),
            tn.x,
            tn.xnext,
            &[],
            &mut xgate,
            &[],
        );
    }
    let prog = b.finish();

    let n_ops = prog.insts.len();
    let mut m = Model {
        n_cu,
        target: 0, // GLM/Kimi/DeepSeek/Nemotron: GPU fingerprint not threaded here yet
        tensors,
        progs: vec![prog],
        kv_row_insts: Vec::new(),
        prog_t: vec![1],
        gen,
    };
    if !rope_gen {
        m.bake_gen();
    }
    // Single-layer bring-up gate: verified like every other path, but it writes
    // NO `build.json`, so there is no artifact to record the outcome in. That is
    // acceptable only because there is also nothing here to over-claim — a
    // consumer that finds no manifest cannot read a verification status off it.
    let _lean = crate::apply_verify_gate(&m, verify);
    std::fs::write(out, m.to_blob()).unwrap();
    eprintln!(
        "glm52: GlmMoeDsa {} layers hidden={} heads={} kv_lora={} q_lora={} qk={}+{} v={} \
         experts={}/top{} moe_inter={} scale={:.4}",
        c.layers,
        c.hidden,
        c.heads,
        c.kv_lora,
        c.q_lora,
        c.qk_nope,
        c.qk_rope,
        c.v_head,
        c.n_exp,
        c.top_k,
        c.moe_inter,
        c.attn_scale
    );
    eprintln!(
        "  single-layer {} {} block: layer {layer}, {n_ops} ops, max_ctx={ctx} -> {out}",
        if dense {
            "block-fp8 DENSE"
        } else if use_fp8 {
            "block-fp8 MoE"
        } else {
            "bf16 MoE"
        },
        if dense {
            "SwiGLU (op47/44)"
        } else if use_fp8 {
            "MoeExpertGluFp8Blk/DownFp8Blk"
        } else {
            "MoeExpertGlu/Down"
        }
    );
    let _ = c.qk_head();
}

/// MLA+MoE emit flavor. The Model build (declare_glm + emit_glm_block/dense) is IDENTICAL across
/// these — only the descriptor's arch tag, mixer `kind`, and whether the DSA indexer role/dims/
/// carried indices apply differ. GLM-5.2 has the DSA lightning indexer; Kimi K2.7 / DeepSeek-V3 are
/// plain MLA (their cfg holds `has_dsa=false`, so the shared emit never takes the DSA path).
#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum MlaArch {
    Glm,
    Kimi,
    DeepSeek,
    K3DSpark,
}

/// Build a single-block (layers `block`) MLA+MoE program + its descriptor, no file IO — the testable
/// core of `--block` on the GLM emit path and, via `arch`,
/// the Kimi/DeepSeek reuse of that same emit (§5.0, M3). No embed / no final-norm+lm_head+argmax
/// tail: `act.x` in, the last layer's residual out. The emitter is slot-indexed (per-layer vectors
/// are built from `layer_ids`), so a range extraction is the existing single-layer bring-up
/// (glm_main default) generalized to N layers. `arch` selects only descriptor metadata — the ops
/// come from the shared GLM emit, DSA-gated on `c.dsa(ctx)` (held off for Kimi via cfg `has_dsa`).
// Decode-only shorthand. The CLI entry points go through `glm_build_block_pf` (they have prefill
// buckets to pass); this stays because it is what the op-sequence tests assert against, and those
// assertions are the offline proof that the decode stream still matches the validated gfx950 block.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn glm_build_block(
    c: &GlmCfg,
    ctx: u32,
    n_cu: u32,
    block: std::ops::Range<usize>,
    use_fp8: bool,
    model: &str,
    arch: MlaArch,
) -> (Model, plow_asset::BlockDescriptor) {
    let enc = MoeEnc::from_flags(use_fp8, false);
    glm_build_block_pf(
        c,
        ctx,
        n_cu,
        block,
        use_fp8,
        model,
        arch,
        &[],
        PrefillScope::Attn,
        enc,
    )
}

/// As [`glm_build_block`], plus one PREFILL bucket program per entry of `pf` ahead of the decode
/// program.
///
/// Program order is buckets-then-decode, which is not a convention this function is free to choose:
/// `devgen::manifest` derives `kind` from `pi == progs.len()-1` and `plowrt`'s loader reads
/// `progs[..len-1]` as the prefill set. `prog_t` carries each bucket's T and the decode 1.
///
/// `pf` is a PARAMETER rather than an environment read so tests can drive it deterministically —
/// process-global env under a parallel test runner is a race, and the emitted program set is exactly
/// what these tests exist to pin. The CLI entry points do the env read (`glm_prefill_buckets_env`).
///
/// A prefill bucket is emitted only for a SINGLE-layer block: the program ends at the
/// post-attention norm (`act.xn2`) because the FFN has no T-row kernel, so it does not produce a
/// residual stream a following layer could read. Asking for prefill on a multi-layer extraction is a
/// request that cannot be honoured, so it asserts rather than silently emitting a broken chain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_build_block_pf(
    c: &GlmCfg,
    ctx: u32,
    n_cu: u32,
    block: std::ops::Range<usize>,
    use_fp8: bool,
    model: &str,
    arch: MlaArch,
    pf: &[u32],
    scope: PrefillScope,
    enc: MoeEnc,
) -> (Model, plow_asset::BlockDescriptor) {
    use plow_asset::*;
    let layers: Vec<u32> = block.clone().map(|l| l as u32).collect();
    // A whole-layer prefill needs a T-row FFN for EVERY layer in the block. Both kinds now have
    // one: MoE layers on the grouped expert arms, dense layers on those SAME arms with degenerate
    // 1-expert routing (`emit_glm_dense_block_prefill`). MXFP4 is the remaining hole — its grouped
    // arm is the A4W4 fused-bridge path, which the dense emit does not carry operands for.
    if scope == PrefillScope::Full && !pf.is_empty() && enc == MoeEnc::Mxfp4 {
        for &l in &layers {
            assert!(
                !c.is_dense(l),
                "whole-layer MXFP4 prefill is not implemented for DENSE layer {l} \
                 (< first_k_dense_replace): the grouped MXFP4 arm is the A4W4 fused-bridge path \
                 and the dense emit declares none of its scale rows. Use block-fp8 or bf16, or \
                 extract a MoE layer (--block >= first_k_dense_replace)."
            );
        }
    }
    // ATTENTION-ONLY buckets stop at the post-attention norm, so they produce no residual stream for
    // a following layer — that scope is single-layer by construction. `Full` chains like decode does.
    assert!(
        pf.is_empty() || scope == PrefillScope::Full || layers.len() == 1,
        "attention-only MLA prefill buckets need a single-layer --block: the program ends at the \
         post-attention norm, so block {:?} has no residual stream to hand the next layer. Use \
         PLOW_MLA_PREFILL=full for the whole-layer (MoE) prefill, which chains.",
        block
    );

    let mut tb = Builder::new(n_cu);
    // One tensor table serves every program, so it is sized for the widest bucket (1 = decode-only).
    let tn = declare_glm_rows(
        &mut tb,
        c,
        ctx,
        &layers,
        pf.iter().copied().max().unwrap_or(1),
        enc,
    );
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();

    // PREFILL bucket programs, ahead of decode.
    let mut progs = Vec::new();
    let mut prog_t = Vec::new();
    for &t in pf {
        let mut pb = Builder::new(n_cu);
        pb.adopt_tensors(tensors.clone());
        let pall = pb.all();
        let mut pxgate = 0u32;
        match scope {
            PrefillScope::Attn => {
                emit_glm_mla_prefill(
                    &mut pb,
                    c,
                    &tn,
                    0,
                    ctx,
                    t,
                    enc,
                    tn.x,
                    &[],
                    &mut pxgate,
                    &pall,
                );
            }
            PrefillScope::Full => {
                // Same ping-pong chain the decode program uses: layer l+1 reads layer l's residual.
                // Dense vs MoE is chosen per layer by exactly the rule the decode chain uses.
                let mut cur = tn.x;
                let mut dep: Vec<u32> = Vec::new();
                for (slot, &l) in layers.iter().enumerate() {
                    let nxt = if cur == tn.x { tn.xnext } else { tn.x };
                    let d = if c.is_dense(l) {
                        emit_glm_dense_block_prefill(
                            &mut pb,
                            c,
                            &tn,
                            slot,
                            ctx,
                            t,
                            enc,
                            cur,
                            nxt,
                            &dep,
                            &mut pxgate,
                            &pall,
                        )
                    } else {
                        emit_glm_block_prefill(
                            &mut pb,
                            c,
                            &tn,
                            slot,
                            ctx,
                            t,
                            enc,
                            cur,
                            nxt,
                            &dep,
                            &mut pxgate,
                            &pall,
                        )
                    };
                    dep = vec![d];
                    cur = nxt;
                }
            }
        }
        progs.push(pb.finish());
        prog_t.push(t);
    }

    let mut b = Builder::new(n_cu);
    b.adopt_tensors(tensors.clone());
    let all = b.all();

    // Layer chain, ping-ponging x <-> xnext (layer l+1 reads layer l's output). The
    // first layer's first op has NO dependency (empty deps) — the block entry is
    // `act.x`, uploaded by the harness. tp==1 single-block: no XReduce, so xgate/xr_cus
    // are inert (mirrors glm_main's single-layer bring-up).
    let mut xgate = 0u32;
    let xr_cus = all.clone();
    let mut cur = tn.x;
    let mut dep: Vec<u32> = Vec::new();
    for (slot, &l) in layers.iter().enumerate() {
        let nxt = if cur == tn.x { tn.xnext } else { tn.x };
        let d = if c.is_dense(l) {
            emit_glm_dense_block(
                &mut b, c, &tn, slot, ctx, 1, 1, enc, cur, nxt, &dep, &mut xgate, &xr_cus,
            )
        } else {
            emit_glm_block(
                &mut b, c, &tn, slot, ctx, 1, 1, enc, cur, nxt, &dep, &mut xgate, &xr_cus,
            )
        };
        dep = vec![d];
        cur = nxt;
    }
    // After N layers the residual is back in `x` (even) or in `xnext` (odd).
    let out_name = if cur == tn.x { "act.x" } else { "act.xnext" };
    progs.push(b.finish());
    prog_t.push(1);
    let m = Model {
        n_cu,
        target: 0, // GLM/Kimi/DeepSeek/Nemotron: GPU fingerprint not threaded here yet
        tensors,
        progs,
        kv_row_insts: Vec::new(),
        prog_t,
        gen,
    };

    // Descriptor. l0 = the extracted layer (block start); its DSA role + FFN kind
    // drive the arch-agnostic fields.
    let l0 = block.start as u32;
    let dsa_on = c.dsa(ctx);
    let full = c.indexer_is_full(l0);
    let dense = c.is_dense(l0);
    let hidden = c.hidden as i64;

    // MLA latent caches (ckv/krot) per layer; the indexer key cache (kidx) too on
    // 'full' indexer layers under an armed DSA gate.
    let mut kv_tensors = Vec::new();
    for l in block.clone() {
        kv_tensors.push(format!("kv.{l}.ckv"));
        kv_tensors.push(format!("kv.{l}.krot"));
        if glm_fp8_kv() {
            kv_tensors.push(format!("kv.{l}.scale"));
        }
        if dsa_on && c.indexer_is_full(l as u32) {
            kv_tensors.push(format!("kv.{l}.kidx"));
        }
    }
    let mut carried_state = vec![CarriedState {
        role: "kv".into(),
        tensors: kv_tensors,
        layout: "mla_latent".into(),
    }];
    // IndexShare (§7): a 'reuse' layer under an armed DSA gate consumes the previous
    // indexer layer's top-k selection — a carried INPUT, since the block does not
    // recompute it. (Gate off, or an 'indexer' layer => computed in-block => no carry.)
    if dsa_on && !full {
        carried_state.push(CarriedState {
            role: "dsa_indices".into(),
            tensors: vec!["act.iidx".into()],
            layout: "topk_positions".into(),
        });
    }

    // Arch-flavor metadata: GLM carries the DSA mixer kind + indexer role + index_* dims; Kimi/
    // DeepSeek are plain MLA (mla_attn, no dsa_role, no index_* dims). The ops are the same.
    let (arch_tag, mixer_kind) = match arch {
        MlaArch::Glm => ("glm_mla_dsa", "mla_dsa"),
        MlaArch::Kimi => ("kimi_mla_moe", "mla_attn"),
        MlaArch::DeepSeek => ("deepseek_mla_moe", "mla_attn"),
        MlaArch::K3DSpark => ("k3_dspark", "mla_draft"),
    };
    let is_glm = arch == MlaArch::Glm;
    let desc = BlockDescriptor {
        model: model.to_string(),
        arch: arch_tag.into(),
        layer: l0,
        kind: vec![
            mixer_kind.into(),
            if dense { "dense_ffn" } else { "moe_ffn" }.into(),
        ],
        hidden,
        dtype: if use_fp8 { "fp8".into() } else { "bf16".into() },
        dims: BlockDims {
            heads: Some(c.heads as i64),
            kv_lora: Some(c.kv_lora as i64),
            q_lora: Some(c.q_lora as i64),
            n_exp: (!dense).then_some(c.n_exp as i64),
            top_k: (!dense).then_some(c.top_k as i64),
            shared_exp: (!dense).then_some(1),
            moe_inter: (!dense).then_some(c.moe_inter as i64),
            index_heads: is_glm.then_some(c.index_heads as i64),
            index_dim: is_glm.then_some(c.index_dim as i64),
            index_topk: is_glm.then_some(c.index_topk as i64),
            ..Default::default()
        },
        // DSA role only on GLM; plain-MLA archs have no indexer (dsa_role absent).
        dsa_role: is_glm.then(|| {
            if full {
                "indexer".into()
            } else {
                "reuse".into()
            }
        }),
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
        carried_state,
        weights: BlockWeights {
            mode: "symlink".into(),
            ckpt: model.to_string(),
            prefix: format!("{}layers.{l0}.", c.prefix),
        },
        programs: BlockPrograms {
            // Non-empty only under an explicit prefill request. These buckets carry the MLA
            // ATTENTION at T rows; the FFN half is still decode-only (no T-row kernel).
            prefill_buckets: pf.iter().map(|&t| t as i64).collect(),
            decode_t: 1,
        },
    };
    (m, desc)
}

/// `--block` on the GLM (glm_moe_dsa) emit path. Emits ONE block (layers `spec`) as a
/// GPU-loadable PLOWDEV blob with a `SECT_METADATA` `block.json` descriptor + sibling
/// file — the GLM analogue of the gemma `--block` path (decode-only; the GLM emitter
/// has no prefill program).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_emit_block(
    dir: &Path,
    ctx: u32,
    out: &str,
    n_cu: u32,
    tp: u32,
    spec: &str,
    rope_gen: bool,
    target: &str,
    verify: Option<&crate::VerifyHook>,
) {
    let mut c = cfg_glm(dir);
    c.tp = tp;
    // `--num-gpus N --parallel tp` reaches here as tp=N. The block emit is TP-parameterized end to
    // end — every head-dimensioned tensor and op field is this rank's nh_l = heads/tp shard, the
    // shared/dense intermediates are imoe_l/di_l, the routed experts stay WHOLE under EP, and the
    // o_proj / FFN-down partials go through XReduce — so the assert that used to pin this to tp=1
    // was describing an earlier state of the emitter, not a limitation of it. Verified by
    // `mla_prefill_tp_shapes_scale_with_tp` / `mla_prefill_tp_emits_two_shot_allreduce`.
    // The RUNTIME cannot serve tp>1 yet (no cross-GPU collectives); this is compiler-side emission.
    assert!(
        tp >= 1 && c.heads % tp == 0,
        "tp={tp} must divide n_head={} (each rank owns a whole head shard)",
        c.heads
    );
    let enc = mla_moe_enc_env(dir);
    let use_fp8 = enc == MoeEnc::Fp8Blk;
    let block = parse_block(spec, c.layers as usize);
    let model = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.to_string_lossy().into_owned());
    let (pf, scope) = glm_prefill_buckets_env(ctx);
    let (mut m, desc) = glm_build_block_pf(
        &c,
        ctx,
        n_cu,
        block.clone(),
        use_fp8,
        &model,
        MlaArch::Glm,
        &pf,
        scope,
        enc,
    );
    let section = write_block_descriptor(out, &desc);
    if !rope_gen {
        m.bake_gen();
    }
    let lean = crate::apply_verify_gate(&m, verify);
    std::fs::write(out, m.to_blob_v6(&[section])).unwrap();
    eprintln!(
        "glm52 --block {block:?}: {} block, {} layer(s), {} decode ops, dsa_role={} ctx={ctx} -> {out}",
        if use_fp8 { "block-fp8" } else { "bf16" },
        block.len(),
        m.progs.last().map(|p| p.insts.len()).unwrap_or(0),
        desc.dsa_role.as_deref().unwrap_or("-"),
    );
    report_mla_prefill(&m, &pf, scope, enc);
    write_mla_manifest(&m, out, target, enc, &lean);
    eprintln!("  block.json sibling written next to {out}");
}

/// Write `build.json` beside the emitted `.pkt` — what the packet REQUIRES of the object that runs
/// it, derived from the instruction stream that was just serialized.
///
/// The dense-GQA path has done this since the manifest landed; the MLA `--block` path never did, and
/// with a decode-only blob that was survivable because every decode arm is in every decode object.
/// It stops being survivable now: a prefill bucket needs `interp_prefill_mla`, and a WHOLE-LAYER one
/// needs the MoE arms on top (`interp_prefill_mla_moe`). Pairing a packet with an object that lacks
/// an arm is the failure this file exists to prevent — and on AMD it does not even trap, the
/// dispatch `default:` leaves the output untouched.
fn write_mla_manifest(m: &Model, out: &str, target: &str, enc: MoeEnc, lean: &crate::LeanReport) {
    if target.is_empty() {
        return; // legacy CLI path: output unchanged
    }
    let mut man = crate::manifest::build(m, target, lean);
    // The MXFP4 exception list, stated in the artifact a comparison reads rather than left to a
    // code comment. "all-MXFP4" with one derived 4M-value tensor in bf16 is a fact a dtype
    // comparison has to be able to quote; a number nobody can reconcile with the claimed encoding is
    // the failure this whole line of work exists to prevent.
    if enc == MoeEnc::Mxfp4 {
        man["weight_encoding"] = serde_json::json!({
            "moe": "mxfp4_a4w4",
            "projections": "mxfp4_w4a16",
            "bf16_exceptions": mxfp4_bf16_exceptions(),
            "note": "W_uv is DERIVED by weight-prep (a fold of kv_b_proj), so a bf16 copy exists                      whatever the checkpoint stores; MLA_MERGE_FOLD / O_UV_FOLD take it as                      `const bf16*` with no encoding parameter. Norms and the router bias are not                      matmul weights and stay bf16/f32 under every encoding.",
        });
    }
    let mpath = std::path::Path::new(out).with_file_name("build.json");
    match serde_json::to_vec_pretty(&man).map(|b| std::fs::write(&mpath, b)) {
        Ok(Ok(())) => eprintln!("  build manifest -> {}", mpath.display()),
        Ok(Err(e)) => eprintln!("  WARN: build.json not written: {e}"),
        Err(e) => eprintln!("  WARN: build.json not serialized: {e}"),
    }
}

/// One line per emitted prefill bucket, and a loud line about what they do NOT contain.
///
/// The gap is invisible in the blob — a bucket program looks like any other program — so state it
/// where the person who ran the emit will read it, not only in a doc comment.
fn report_mla_prefill(m: &Model, pf: &[u32], scope: PrefillScope, enc: MoeEnc) {
    if enc == MoeEnc::Mxfp4 {
        eprintln!(
            "  weight encoding: MXFP4 — A4W4 experts (i[3]=2 prefill / i[6]=2 decode), w4a16 \
             projections. bf16 exceptions: {:?} (derived, no encoding parameter on ops 52/57).",
            mxfp4_bf16_exceptions()
        );
    }
    if pf.is_empty() {
        return;
    }
    for (i, &t) in pf.iter().enumerate() {
        eprintln!("  prefill bucket T={t}: {} ops", m.progs[i].insts.len());
    }
    match scope {
        PrefillScope::Attn => eprintln!(
            "  NOTE: these buckets cover the ATTENTION sub-block only (through \
             post_attention_layernorm -> act.xn2); the block output act.x/act.xnext is written by \
             the DECODE program only. PLOW_MLA_PREFILL=full emits the whole layer (MoE FFN too)."
        ),
        PrefillScope::Full => eprintln!(
            "  whole-layer prefill: MLA attention + token-sorted grouped MoE FFN (ops 83-87). \
             Requires an object built with PLOW_MLA_PREFILL=1 AND the MoE prefill arms."
        ),
    }
}

/// `--block` on the Kimi K2.7 / DeepSeek MLA+MoE path (M3).
/// Emits ONE block (layers `spec`) as a GPU-loadable PLOWDEV blob with a `SECT_METADATA` `block.json`
/// descriptor + sibling file. REUSES the GLM MLA + MoE emit verbatim (glm_build_block) with a Kimi
/// cfg (`has_dsa=false`) — no DSA, KV latent (ckv/krot) carried state, decode-only (the GLM emit has
/// no prefill program, so programs.prefill_buckets stays empty). `arch` picks the Kimi vs DeepSeek
/// descriptor tag.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn kimi_emit_block(
    dir: &Path,
    ctx: u32,
    out: &str,
    n_cu: u32,
    tp: u32,
    spec: &str,
    arch: MlaArch,
    rope_gen: bool,
    target: &str,
    verify: Option<&crate::VerifyHook>,
) {
    let mut c = if arch == MlaArch::K3DSpark {
        cfg_k3_dspark(dir)
    } else {
        cfg_kimi(dir)
    };
    c.tp = tp;
    // See the note in `glm_emit_block`: the shared MLA+MoE emit is TP-parameterized, so `--num-gpus
    // N` sharding is emission-complete on this path too. tp must divide the head count.
    assert!(
        tp >= 1 && c.heads % tp == 0,
        "tp={tp} must divide n_head={} (each rank owns a whole head shard)",
        c.heads
    );
    let enc = mla_moe_enc_env(dir);
    let use_fp8 = enc == MoeEnc::Fp8Blk;
    let block = parse_block(spec, c.layers as usize);
    let model = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.to_string_lossy().into_owned());
    let (pf, scope) = glm_prefill_buckets_env(ctx);
    let (mut m, desc) = glm_build_block_pf(
        &c,
        ctx,
        n_cu,
        block.clone(),
        use_fp8,
        &model,
        arch,
        &pf,
        scope,
        enc,
    );
    if arch == MlaArch::K3DSpark {
        let declared = m
            .tensors
            .iter()
            .map(|tensor| tensor.name.clone())
            .collect::<Vec<_>>();
        crate::checkpoint::validate_coverage(
            dir,
            "layers.",
            &declared,
            Some(block.clone()),
            &[
                "q_b_proj.weight",
                "kv_b_proj.weight",
                "kv_a_proj_with_mqa.weight",
            ],
            &[],
            &["derived."],
        )
        .unwrap_or_else(|e| panic!("k3_dspark: {e}"));
        let mut expected = Vec::with_capacity(block.len() * 5);
        for layer in block.clone() {
            let prefix = format!("layers.{layer}.self_attn.");
            expected.extend([
                (
                    format!("{prefix}derived.q_absorb.weight"),
                    vec![(c.heads * c.kv_lora) as u64, c.q_lora as u64],
                ),
                (
                    format!("{prefix}derived.q_rope.weight"),
                    vec![(c.heads * c.qk_rope) as u64, c.q_lora as u64],
                ),
                (
                    format!("{prefix}derived.kv_a_latent.weight"),
                    vec![c.kv_lora as u64, c.hidden as u64],
                ),
                (
                    format!("{prefix}derived.k_rope.weight"),
                    vec![c.qk_rope as u64, c.hidden as u64],
                ),
                (
                    format!("{prefix}derived.v_absorb.weight"),
                    vec![(c.heads * c.kv_lora) as u64, c.v_head as u64],
                ),
            ]);
        }
        crate::checkpoint::validate_bf16_sidecar(
            dir,
            "model-idx-derived-dspark.safetensors",
            &expected,
        )
        .unwrap_or_else(|e| panic!("k3_dspark: {e}"));
    }
    let section = write_block_descriptor(out, &desc);
    if !rope_gen {
        m.bake_gen();
    }
    let lean = crate::apply_verify_gate(&m, verify);
    std::fs::write(out, m.to_blob_v6(&[section])).unwrap();
    eprintln!(
        "{} --block {block:?}: {} block, {} layer(s), {} decode ops, ctx={ctx} tp={tp} -> {out}",
        desc.arch,
        if use_fp8 { "block-fp8" } else { "bf16" },
        block.len(),
        m.progs.last().map(|p| p.insts.len()).unwrap_or(0),
    );
    report_mla_prefill(&m, &pf, scope, enc);
    write_mla_manifest(&m, out, target, enc, &lean);
    eprintln!("  block.json sibling written next to {out}");
}

// ===== Nemotron-3 Mamba-2 hybrid (M4). =========
// Nemotron-3 Nano 30B-A3B is a HYBRID: 52 layers = 23 Mamba-2 mixers + 23 MoE FFNs + 6 GQA
// attentions, interleaved by a `hybrid_override_pattern` string. The Mamba-2 mixer is the
// genuinely NEW piece (the first state-space op in the tree — DevOp::Mamba2Scan, op_mamba.cuh,
// and the `mamba_ref` golden below). The GQA-attention and MoE layers REUSE existing DevOps
// (the same attn/MoE ops gemma/kimi emit), so only the mamba mixer is new work.

/// One Nemotron layer's role. The `hybrid_override_pattern` chars map: 'M' => Mamba-2 mixer,
/// '*' => GQA attention, '-' => MoE FFN (Nemotron-3 is the MoE variant, so the MLP slot is MoE).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum NemoKind {
    Mamba,
    Attn,
    Moe,
}

impl NemoKind {
    /// Descriptor `kind` tag for this layer.
    fn tag(self) -> &'static str {
        match self {
            NemoKind::Mamba => "mamba2",
            NemoKind::Attn => "gqa_attn",
            NemoKind::Moe => "moe_ffn",
        }
    }
}

/// Nemotron-3 hybrid config. Small synthetic values in tests; `cfg_nemotron` fills real dims from
/// `config.json`. Reference geometry (Nemotron-H / Nemotron-3 Nano 30B-A3B, assumption where a key
/// is absent): hidden 4096, mamba d_inner 8192 (expand 2), n_head 128, head_dim 64, d_state 128,
/// d_conv 4, n_groups 8; attn 32 heads / 8 kv-heads / head_dim 128; MoE 128 routed + 1 shared,
/// top_k 6, moe_inter 768.
pub(crate) struct NemoCfg {
    layers: usize,
    hidden: u32,
    // Mamba-2 mixer.
    d_inner: u32,
    n_head: u32,   // mamba_n_heads
    head_dim: u32, // d_inner / n_head
    d_state: u32,
    d_conv: u32,
    n_groups: u32,
    // GQA attention.
    attn_heads: u32,
    attn_kv_heads: u32,
    attn_head_dim: u32,
    // MoE.
    n_exp: u32,
    top_k: u32,
    shared_exp: u32,
    moe_inter: u32,
    eps: f32,
    kinds: Vec<NemoKind>,
}

impl NemoCfg {
    /// conv_dim = d_inner + 2*n_groups*d_state — the width the depthwise conv1d runs over (x,B,C).
    fn conv_dim(&self) -> u32 {
        self.d_inner + 2 * self.n_groups * self.d_state
    }
}

/// Parse a Nemotron-3 `config.json`. Where a key is absent (this box has no checkpoint), falls back
/// to the reference geometry above and NOTES it via the returned defaults. The per-layer pattern
/// comes from `hybrid_override_pattern` (M/*/- chars); absent, it synthesizes the documented
/// 23-mamba / 6-attn / 23-moe interleave.
fn cfg_nemotron(dir: &Path) -> NemoCfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .unwrap();
    let gu = |k: &str, d: u32| v[k].as_u64().map(|x| x as u32).unwrap_or(d);
    let hidden = gu("hidden_size", 4096);
    let expand = gu("mamba_expand", 2);
    let d_inner = v["mamba_d_inner"]
        .as_u64()
        .map(|x| x as u32)
        .unwrap_or(expand * hidden);
    let n_head = gu("mamba_n_heads", 128);
    let head_dim = v["mamba_head_dim"]
        .as_u64()
        .map(|x| x as u32)
        .unwrap_or(d_inner / n_head.max(1));
    let layers = gu("num_hidden_layers", 52) as usize;
    // Per-layer kind pattern.
    let kinds: Vec<NemoKind> = match v["hybrid_override_pattern"].as_str() {
        Some(p) => p
            .chars()
            .filter(|c| matches!(c, 'M' | '*' | '-'))
            .map(|c| match c {
                'M' => NemoKind::Mamba,
                '*' => NemoKind::Attn,
                _ => NemoKind::Moe,
            })
            .collect(),
        // Assumption: no pattern in config -> the documented interleave. Attention every ~9th
        // layer (6 of 52), Mamba/MoE alternating otherwise.
        None => (0..layers)
            .map(|l| {
                if l % 9 == 4 {
                    NemoKind::Attn
                } else if l % 2 == 0 {
                    NemoKind::Mamba
                } else {
                    NemoKind::Moe
                }
            })
            .collect(),
    };
    let c = NemoCfg {
        layers: kinds.len().max(layers),
        hidden,
        d_inner,
        n_head,
        head_dim,
        d_state: gu("mamba_d_state", 128),
        d_conv: gu("mamba_d_conv", 4),
        n_groups: gu("mamba_n_groups", 8),
        attn_heads: gu("num_attention_heads", 32),
        attn_kv_heads: gu("num_key_value_heads", 8),
        attn_head_dim: gu("attention_head_dim", 128),
        n_exp: gu("n_routed_experts", 128),
        top_k: gu("num_experts_per_tok", 6),
        shared_exp: gu("n_shared_experts", 1),
        moe_inter: gu("moe_intermediate_size", 768),
        eps: v["rms_norm_eps"].as_f64().map(|x| x as f32).unwrap_or(1e-5),
        kinds,
    };
    crate::require_moe_topk(c.top_k, "nemotron");
    // `attention_head_dim` REACHES A BARE if-CHAIN WITH NO `else`. `exec_flash_decode`
    // (interp.hip), the FLASH_MERGE arms and HEADNORM_ROPE all dispatch on hd with
    // `if (hd == 128) ... else if (hd == 256) ... else if (hd == 512) ...` and NOTHING after it.
    // Unlike the NVIDIA interpreter this one has no `default: __trap()` anywhere, so hd=64 or 192
    // selects no arm at all: Q is never RoPE'd, Opart is never written, O is never written, and
    // the model emits whatever was already in those buffers. The only existing shape gate is
    // `hd % 8 == 0` (the vectorised load width), which 64 and 192 both pass.
    //
    // Checked HERE, at the config.json boundary, and deliberately not at emit time: the synthetic
    // `nemo_ref_cfg` fixture uses `attn_head_dim: 16` for op-sequence tests that never build a
    // runnable object, and breaking those would be collateral damage in another model's tests.
    // This is the edge a real checkpoint crosses.
    assert!(
        matches!(c.attn_head_dim, 128 | 256 | 512),
        "nemotron: attention_head_dim={} — the AMD interpreter instantiates the flash decode / \
         merge / head-norm-rope arms at 128, 256 and 512 ONLY, and its dispatch has no `else` and \
         no trap, so this would emit a program that silently writes no attention output at all. \
         Add the instantiation in runtime/amd/interp.hip before emitting this checkpoint.",
        c.attn_head_dim
    );
    c
}

/// Emit ONE Mamba-2 mixer layer (decode, M=1). RmsNorm -> 3 projection GEMVs (z / xBC / dt; these
/// are the column slices of the single in_proj) -> the new DevOp::Mamba2Scan mixer core (conv1d +
/// SSD scan + gated RMSNorm, reading/writing conv_state + ssm_state) -> out_proj GEMV -> residual.
/// Returns the residual counter (the block chain dep). `mamba.{l}.conv_state`/`ssm_state` are the
/// carried tensors the descriptor advertises.
fn emit_nemotron_mamba(
    b: &mut Builder,
    c: &NemoCfg,
    l: u32,
    cur: u32,
    nxt: u32,
    deps: &[u32],
) -> u32 {
    let bf = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 2);
    let f32t = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 4);
    let (h, di, nh, hd, ds, dc) = (
        c.hidden as u64,
        c.d_inner as u64,
        c.n_head as u64,
        c.head_dim as u64,
        c.d_state as u64,
        c.d_conv as u64,
    );
    let cd = c.conv_dim() as u64;
    let pfx = format!("mamba.{l}.");
    let cus = b.all();
    let one = vec![0u32];
    // input RMSNorm
    let xn = bf(b, format!("{pfx}xn"), di.max(h));
    let g_in = bf(b, format!("{pfx}norm_in.w"), h);
    let d_norm = b.emit(DevOp::RmsNorm, cus.clone(), deps, |i| {
        i.t[0] = xn;
        i.t[1] = cur;
        i.t[2] = g_in;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.f[0] = c.eps;
    });
    // z / xBC / dt projections (in_proj column slices)
    let z = bf(b, format!("{pfx}z"), di);
    let wz = bf(b, format!("{pfx}z_proj.w"), di * h);
    let d_z = b.emit(DevOp::Gemv, cus.clone(), &[d_norm], |i| {
        i.t[0] = z;
        i.t[1] = xn;
        i.t[2] = wz;
        i.i[0] = 1;
        i.i[1] = c.d_inner;
        i.i[2] = c.hidden;
    });
    let xbc = bf(b, format!("{pfx}xbc"), cd);
    let wxbc = bf(b, format!("{pfx}xbc_proj.w"), cd * h);
    let d_xbc = b.emit(DevOp::Gemv, cus.clone(), &[d_norm], |i| {
        i.t[0] = xbc;
        i.t[1] = xn;
        i.t[2] = wxbc;
        i.i[0] = 1;
        i.i[1] = c.conv_dim();
        i.i[2] = c.hidden;
    });
    let dt = bf(b, format!("{pfx}dt"), nh);
    let wdt = bf(b, format!("{pfx}dt_proj.w"), nh * h);
    let d_dt = b.emit(DevOp::Gemv, cus.clone(), &[d_norm], |i| {
        i.t[0] = dt;
        i.t[1] = xn;
        i.t[2] = wdt;
        i.i[0] = 1;
        i.i[1] = c.n_head;
        i.i[2] = c.hidden;
    });
    // Mixer core (single-CU, correctness-first). Packed params: A_log|D|dt_bias|conv_b|norm_w.
    let mixed = bf(b, format!("{pfx}y"), di);
    let conv_w = bf(b, format!("{pfx}conv1d.w"), cd * dc);
    let params = f32t(b, format!("{pfx}ssm_params"), 3 * nh + cd + di);
    let conv_state = f32t(b, format!("{pfx}conv_state"), (dc - 1) * cd);
    let ssm_state = f32t(b, format!("{pfx}ssm_state"), nh * hd * ds);
    let d_scan = b.emit(DevOp::Mamba2Scan, one, &[d_z, d_xbc, d_dt], |i| {
        i.t[0] = mixed;
        i.t[1] = xbc;
        i.t[2] = dt;
        i.t[3] = z;
        i.t[4] = conv_w;
        i.t[5] = params;
        i.t[6] = conv_state;
        i.t[7] = ssm_state;
        i.i[0] = 1; // T (decode)
        i.i[1] = c.d_inner;
        i.i[2] = c.n_head;
        i.i[3] = c.head_dim;
        i.i[4] = c.d_state;
        i.i[5] = c.n_groups;
        i.i[6] = c.d_conv;
        i.i[7] = c.conv_dim();
        i.f[0] = c.eps;
    });
    // out_proj + residual.
    let op = bf(b, format!("{pfx}out"), h);
    let wout = bf(b, format!("{pfx}out_proj.w"), h * di);
    let d_out = b.emit(DevOp::Gemv, cus.clone(), &[d_scan], |i| {
        i.t[0] = op;
        i.t[1] = mixed;
        i.t[2] = wout;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.i[2] = c.d_inner;
    });
    b.emit(DevOp::Residual, cus, &[d_out], |i| {
        i.t[0] = nxt;
        i.t[1] = cur;
        i.t[2] = op;
        i.i[0] = c.hidden;
        i.f[0] = 1.0;
    })
}

/// Emit ONE GQA attention layer (decode, M=1) reusing the existing attention DevOps
/// (the gemma/kimi decode path): RmsNorm -> fused GemvQkv -> HeadNormRope (q RoPE) ->
/// FlashDecode -> FlashMerge -> o_proj GEMV -> residual. KV cache (`kv.{l}.k/v`) is the
/// carried state.
fn emit_nemotron_attn(
    b: &mut Builder,
    c: &NemoCfg,
    l: u32,
    ctx: u32,
    cur: u32,
    nxt: u32,
    deps: &[u32],
) -> u32 {
    let bf = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 2);
    let f32t = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 4);
    let (h, nh, kvh, hd) = (
        c.hidden as u64,
        c.attn_heads as u64,
        c.attn_kv_heads as u64,
        c.attn_head_dim as u64,
    );
    let nq = nh * hd;
    let nkv = kvh * hd;
    let pfx = format!("attn.{l}.");
    let cus = b.all();
    let nsplit = 1u32;
    let xn = bf(b, format!("{pfx}xn"), h);
    let g_in = bf(b, format!("{pfx}norm_in.w"), h);
    let d_norm = b.emit(DevOp::RmsNorm, cus.clone(), deps, |i| {
        i.t[0] = xn;
        i.t[1] = cur;
        i.t[2] = g_in;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.f[0] = c.eps;
    });
    let q = bf(b, format!("{pfx}q"), nq);
    let k = bf(b, format!("{pfx}kv.{l}.k"), (ctx as u64) * nkv);
    let vv = bf(b, format!("{pfx}kv.{l}.v"), (ctx as u64) * nkv);
    let wq = bf(b, format!("{pfx}q_proj.w"), nq * h);
    let wk = bf(b, format!("{pfx}k_proj.w"), nkv * h);
    let wv = bf(b, format!("{pfx}v_proj.w"), nkv * h);
    let d_qkv = b.emit(DevOp::GemvQkv, cus.clone(), &[d_norm], |i| {
        i.t[0] = q;
        i.t[1] = xn;
        i.t[2] = wq;
        i.t[3] = k;
        i.t[4] = wk;
        i.t[5] = vv;
        i.t[6] = wv;
        i.i[0] = 1;
        i.i[1] = nq as u32;
        i.i[2] = c.hidden;
        i.i[3] = nkv as u32;
        i.i[4] = nkv as u32;
    });
    let cos = bf(b, format!("{pfx}rope.cos"), (ctx as u64) * hd);
    let sin = bf(b, format!("{pfx}rope.sin"), (ctx as u64) * hd);
    let pos = b.tensor(&format!("{pfx}pos"), 4);
    let d_rope = b.emit(DevOp::HeadNormRope, cus.clone(), &[d_qkv], |i| {
        i.t[0] = q;
        i.t[1] = q;
        i.t[3] = cos;
        i.t[4] = sin;
        i.t[5] = pos;
        i.i[0] = 1;
        i.i[1] = c.attn_heads;
        i.i[2] = c.attn_head_dim;
    });
    let opart = f32t(b, format!("{pfx}opart"), nq * (nsplit as u64));
    let mlpart = f32t(b, format!("{pfx}mlpart"), nh * (nsplit as u64) * 2);
    let kvlen = b.tensor(&format!("{pfx}kvlen"), 4);
    let d_fd = b.emit(DevOp::FlashDecode, cus.clone(), &[d_rope], |i| {
        i.t[0] = opart;
        i.t[1] = mlpart;
        i.t[2] = q;
        i.t[3] = k;
        i.t[4] = vv;
        i.t[5] = kvlen;
        i.i[0] = 1;
        i.i[1] = c.attn_heads;
        i.i[2] = c.attn_kv_heads;
        i.i[3] = nkv as u32;
        i.i[5] = nsplit;
        i.i[6] = c.attn_head_dim;
        i.f[0] = (c.attn_head_dim as f32).powf(-0.5);
    });
    let ao = bf(b, format!("{pfx}ao"), nq);
    let d_merge = b.emit(DevOp::FlashMerge, cus.clone(), &[d_fd], |i| {
        i.t[0] = ao;
        i.t[1] = opart;
        i.t[2] = mlpart;
        i.i[0] = 1;
        i.i[1] = c.attn_heads;
        i.i[2] = nsplit;
        i.i[3] = c.attn_head_dim;
    });
    let op = bf(b, format!("{pfx}o"), h);
    let wo = bf(b, format!("{pfx}o_proj.w"), h * nq);
    let d_o = b.emit(DevOp::Gemv, cus.clone(), &[d_merge], |i| {
        i.t[0] = op;
        i.t[1] = ao;
        i.t[2] = wo;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.i[2] = nq as u32;
    });
    b.emit(DevOp::Residual, cus, &[d_o], |i| {
        i.t[0] = nxt;
        i.t[1] = cur;
        i.t[2] = op;
        i.i[0] = c.hidden;
        i.f[0] = 1.0;
    })
}

/// Emit ONE MoE FFN layer (decode, M=1) reusing the existing MoE DevOps (the kimi/GLM MoE path):
/// RmsNorm -> router score GEMV -> MoeRouterTopk -> shared expert (GemvGlu + down GEMV) ->
/// top_k × (MoeExpertGlu, MoeExpertDown) -> MoeCombine. No carried state.
fn emit_nemotron_moe(
    b: &mut Builder,
    c: &NemoCfg,
    l: u32,
    cur: u32,
    nxt: u32,
    deps: &[u32],
) -> u32 {
    let bf = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 2);
    let f32t = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 4);
    let (h, ne, kk, im) = (
        c.hidden as u64,
        c.n_exp as u64,
        c.top_k as u64,
        c.moe_inter as u64,
    );
    let pfx = format!("moe.{l}.");
    let cus = b.all();
    let xn = bf(b, format!("{pfx}xn"), h);
    let g_in = bf(b, format!("{pfx}norm_in.w"), h);
    let d_norm = b.emit(DevOp::RmsNorm, cus.clone(), deps, |i| {
        i.t[0] = xn;
        i.t[1] = cur;
        i.t[2] = g_in;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.f[0] = c.eps;
    });
    let logit = bf(b, format!("{pfx}logit"), ne);
    let wr = bf(b, format!("{pfx}router.w"), ne * h);
    let d_score = b.emit(DevOp::Gemv, cus.clone(), &[d_norm], |i| {
        i.t[0] = logit;
        i.t[1] = xn;
        i.t[2] = wr;
        i.i[0] = 1;
        i.i[1] = c.n_exp;
        i.i[2] = c.hidden;
    });
    let table = f32t(b, format!("{pfx}routing_table"), kk * 2);
    let d_topk = b.emit(DevOp::MoeRouterTopk, vec![0u32], &[d_score], |i| {
        i.t[0] = table;
        i.t[1] = logit;
        i.i[1] = c.n_exp;
        i.i[2] = c.top_k;
        i.f[0] = 1.0;
    });
    // Shared expert (always on).
    let sh_fu = bf(b, format!("{pfx}shared.fu"), im);
    let sh_gu = bf(b, format!("{pfx}shared.gate_up.w"), 2 * im * h);
    let d_sgu = b.emit(DevOp::GemvGlu, cus.clone(), &[d_norm], |i| {
        i.t[0] = sh_fu;
        i.t[1] = xn;
        i.t[2] = sh_gu;
        i.t[5] = sh_gu;
        i.i[0] = 1;
        i.i[1] = c.moe_inter;
        i.i[2] = c.hidden;
        i.i[5] = 1; // silu
    });
    let shared = bf(b, format!("{pfx}shared.out"), h);
    let sh_dn = bf(b, format!("{pfx}shared.down.w"), h * im);
    let d_sdn = b.emit(DevOp::Gemv, cus.clone(), &[d_sgu], |i| {
        i.t[0] = shared;
        i.t[1] = sh_fu;
        i.t[2] = sh_dn;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.i[2] = c.moe_inter;
    });
    // Routed experts (per-slot, one glu+down each).
    let ewt = b.tensor(&format!("{pfx}expert_weight_table"), ne * 3 * 8);
    let fu = bf(b, format!("{pfx}fu"), kk * im);
    let part = f32t(b, format!("{pfx}part"), kk * h);
    let mut d_parts = Vec::new();
    for slot in 0..c.top_k {
        let d_glu = b.emit(DevOp::MoeExpertGlu, cus.clone(), &[d_topk], |i| {
            i.t[0] = fu;
            i.t[1] = xn;
            i.t[2] = table;
            i.t[3] = ewt;
            i.i[0] = slot;
            i.i[1] = c.moe_inter;
            i.i[2] = c.hidden;
            i.i[3] = c.n_exp;
            i.i[5] = 1;
        });
        let d_dn = b.emit(DevOp::MoeExpertDown, cus.clone(), &[d_glu], |i| {
            i.t[0] = part;
            i.t[1] = fu;
            i.t[2] = table;
            i.t[3] = ewt;
            i.i[0] = slot;
            i.i[1] = c.hidden;
            i.i[2] = c.moe_inter;
            i.i[3] = c.n_exp;
        });
        d_parts.push(d_dn);
    }
    let mut combine_deps = vec![d_sdn];
    combine_deps.extend(&d_parts);
    b.emit(DevOp::MoeCombine, cus, &combine_deps, |i| {
        i.t[0] = nxt;
        i.t[1] = cur;
        i.t[2] = shared;
        i.t[3] = part;
        i.i[0] = c.hidden;
        i.i[1] = c.top_k;
    })
}

/// Build a single-block (layers `block`) Nemotron-3 program + its descriptor, no file IO — the
/// testable core of `--block` on the nemotron_h path (§5.3, §7). Per the extracted layer's kind it
/// emits the NEW Mamba-2 mixer op, or reuses the GQA-attention emit, or reuses the MoE emit. No
/// embed / no final-norm+lm_head+argmax tail: `act.x` in, the last layer's residual out. Decode-only
/// (M=1), like the GLM/Kimi block path.
pub(crate) fn nemotron_build_block(
    c: &NemoCfg,
    ctx: u32,
    n_cu: u32,
    block: std::ops::Range<usize>,
    model: &str,
) -> (Model, plow_asset::BlockDescriptor) {
    use plow_asset::*;
    let mut b = Builder::new(n_cu);
    let x = b.tensor("act.x", (c.hidden as u64) * 2);
    let xnext = b.tensor("act.xnext", (c.hidden as u64) * 2);
    // Mandatory GpuEngine handles, zero-stubbed (mirrors declare_glm / the gemma
    // block path). The block emits no Embed / lm_head, so in.ids and act.logits
    // are inert; in.pos / in.kvlen are patched per decode step by the runtime.
    // Without these, GpuEngine::load rejects the blob ("missing in.ids/in.pos/
    // in.kvlen/act.logits") before any kernel launches.
    b.tensor("in.ids", ctx as u64 * 4);
    b.tensor("in.pos", ctx as u64 * 4);
    b.tensor("in.kvlen", 4); // batch = kvlen_bytes/4 = 1
    b.tensor("act.logits", 1024 * 2); // vocab stub (bf16); unused (no head)
    let mut cur = x;
    let mut dep: Vec<u32> = Vec::new();
    for &l in block.clone().collect::<Vec<_>>().iter() {
        let nxt = if cur == x { xnext } else { x };
        let kind = c.kinds[l];
        let d = match kind {
            NemoKind::Mamba => emit_nemotron_mamba(&mut b, c, l as u32, cur, nxt, &dep),
            NemoKind::Attn => emit_nemotron_attn(&mut b, c, l as u32, ctx, cur, nxt, &dep),
            NemoKind::Moe => emit_nemotron_moe(&mut b, c, l as u32, cur, nxt, &dep),
        };
        dep = vec![d];
        cur = nxt;
    }
    let out_name = if cur == x { "act.x" } else { "act.xnext" };
    let tensors = b.tensors();
    let gen = b.gen_tensors();
    let prog = b.finish();
    let m = Model {
        n_cu,
        target: 0, // GLM/Kimi/DeepSeek/Nemotron: GPU fingerprint not threaded here yet
        tensors,
        progs: vec![prog],
        kv_row_insts: Vec::new(),
        prog_t: vec![1],
        gen,
    };

    // Descriptor. kind = per-layer tags; carried_state = union (conv+ssm per mamba, kv per attn,
    // none for moe); dims populated for whichever layer kinds appear in the block.
    let l0 = block.start as u32;
    let kinds: Vec<NemoKind> = block.clone().map(|l| c.kinds[l]).collect();
    let has_mamba = kinds.contains(&NemoKind::Mamba);
    let has_attn = kinds.contains(&NemoKind::Attn);
    let has_moe = kinds.contains(&NemoKind::Moe);
    let mut carried_state = Vec::new();
    for l in block.clone() {
        match c.kinds[l] {
            NemoKind::Mamba => {
                carried_state.push(CarriedState {
                    role: "conv".into(),
                    tensors: vec![format!("mamba.{l}.conv_state")],
                    layout: "conv".into(),
                });
                carried_state.push(CarriedState {
                    role: "ssm".into(),
                    tensors: vec![format!("mamba.{l}.ssm_state")],
                    layout: "ssm_head_major".into(),
                });
            }
            NemoKind::Attn => carried_state.push(CarriedState {
                role: "kv".into(),
                tensors: vec![format!("kv.{l}.k"), format!("kv.{l}.v")],
                layout: "head_major".into(),
            }),
            NemoKind::Moe => {}
        }
    }
    let hidden = c.hidden as i64;
    let desc = BlockDescriptor {
        model: model.to_string(),
        arch: "nemotron_h".into(),
        layer: l0,
        kind: kinds.iter().map(|k| k.tag().to_string()).collect(),
        hidden,
        dtype: "bf16".into(),
        dims: BlockDims {
            // Mamba-2.
            d_inner: has_mamba.then_some(c.d_inner as i64),
            n_head: has_mamba.then_some(c.n_head as i64),
            d_state: has_mamba.then_some(c.d_state as i64),
            d_conv: has_mamba.then_some(c.d_conv as i64),
            n_groups: has_mamba.then_some(c.n_groups as i64),
            // head_dim: mamba head width if a mamba layer, else attn head width.
            head_dim: if has_mamba {
                Some(c.head_dim as i64)
            } else if has_attn {
                Some(c.attn_head_dim as i64)
            } else {
                None
            },
            // GQA attention.
            heads: has_attn.then_some(c.attn_heads as i64),
            kv_heads: has_attn.then_some(c.attn_kv_heads as i64),
            // MoE.
            n_exp: has_moe.then_some(c.n_exp as i64),
            top_k: has_moe.then_some(c.top_k as i64),
            shared_exp: has_moe.then_some(c.shared_exp as i64),
            moe_inter: has_moe.then_some(c.moe_inter as i64),
            ..Default::default()
        },
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
        carried_state,
        weights: BlockWeights {
            mode: "symlink".into(),
            ckpt: model.to_string(),
            prefix: format!("backbone.layers.{l0}."),
        },
        programs: BlockPrograms {
            prefill_buckets: Vec::new(), // decode-only (M=1) block emit
            decode_t: 1,
        },
    };
    (m, desc)
}

/// `--block` on the Nemotron-3 (nemotron_h) hybrid path (M4). Emits ONE block (layers `spec`) as a
/// GPU-loadable PLOWDEV blob with a `SECT_METADATA` `block.json` descriptor + sibling file. Per-layer
/// dispatch: the NEW Mamba-2 mixer op (mamba layers) or the reused GQA-attn / MoE emit.
pub(crate) fn nemotron_emit_block(
    dir: &Path,
    ctx: u32,
    out: &str,
    n_cu: u32,
    tp: u32,
    spec: &str,
    rope_gen: bool,
) {
    assert_eq!(
        tp, 1,
        "Nemotron TP sharding is a later milestone; use --tp 1 for --block"
    );
    let c = cfg_nemotron(dir);
    let block = parse_block(spec, c.layers);
    let model = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.to_string_lossy().into_owned());
    let (mut m, desc) = nemotron_build_block(&c, ctx, n_cu, block.clone(), &model);
    let section = write_block_descriptor(out, &desc);
    if !rope_gen {
        m.bake_gen();
    }
    std::fs::write(out, m.to_blob_v6(&[section])).unwrap();
    eprintln!(
        "nemotron_h --block {block:?}: {} layer(s), {} ops, kinds={:?}, ctx={ctx} -> {out}",
        block.len(),
        m.progs[0].insts.len(),
        desc.kind,
    );
    eprintln!("  block.json sibling written next to {out}");
}

mod kimi_k3;
pub(crate) use kimi_k3::{k3_emit_full, kimi_k3_emit};

// ===== tests moved from lib.rs (module breakdown): access mla internals directly =====
#[cfg(test)]
#[path = "mla/ckpt_quant_tests.rs"]
mod ckpt_quant_tests;

#[cfg(test)]
#[path = "mla/glm_tests.rs"]
mod glm_tests;

#[cfg(test)]
#[path = "mla/kimi_tests.rs"]
mod kimi_tests;

#[cfg(test)]
#[path = "mla/nemotron_tests.rs"]
mod nemotron_tests;
