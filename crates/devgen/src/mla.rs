//! MLA + MoE emit family: GLM-5.2 / Kimi K2.7 / DeepSeek-V3 (shared MLA+MoE emit)
//! and Nemotron-3 (Mamba-2 hybrid). Split out of `lib.rs` (module breakdown). All
//! are `--block` extraction on this path; `run_verified` dispatches on model_type.
use std::path::Path;

use packet::dev::{DevOp, TENSOR_NONE, TENSOR_NONE_I, WG_THREADS, WG_WAVES};
use packet::devbuild::{Builder, Model};
use packet::rope::{GenTensor, RopeScale};
use serde_json::Value;

use crate::block::{parse_block, write_block_descriptor};
use super::*;

/// `plans/glm52-arch.md`. `H`/`NH`/`DK`(kv_lora)/`QL`(q_lora)/`QN`(qk_nope)/`DR`(qk_rope)/
/// `VD`(v_head) name the MLA geometry the kernels carry as compile-time operands.
#[derive(Clone)]
pub(crate) struct GlmCfg {
    layers: u32,        // 78 (layer 78 = MTP head, skipped)
    hidden: u32,        // H 6144
    heads: u32,         // NH 64
    kv_lora: u32,       // DK 512 (latent cache width)
    q_lora: u32,        // QL 2048
    qk_nope: u32,       // QN 192 (absorbed into the latent)
    qk_rope: u32,       // DR 64  (partial rope, interleaved)
    v_head: u32,        // VD 256
    vocab: u32,         // 154880
    eps: f32,           // 1e-5
    n_exp: u32,         // E 256 routed experts
    top_k: u32,         // 8
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
    // expert partials in the SAME collective — no new op. See plans/moe-ep-kernels.md §5a.
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
            && std::env::var("PLOW_GLM_DSA").ok().as_deref() != Some("0");
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
    let rope_theta = v["rope_theta"].as_f64().or_else(|| rp["rope_theta"].as_f64());
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
        // Flat checkpoint: GLM / DeepSeek / Kimi-K2.7 all ship `model.layers.…` at the root.
        // A nested (multimodal) variant sets this from its own wrapper, and nothing else changes.
        prefix: "model.".to_string(),
        tp: 1,
        ep: std::env::var("GLM_EP").ok().as_deref() == Some("1"),
        group: std::env::var("GLM_GROUP").ok().as_deref() == Some("1"),
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

/// Kimi K2.7 / DeepSeek-V2/V3 cfg (plans/block-asset-harness.md §5.0/§5.3, M3). These are plain
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
    // plans/mla-sm120-kernels.md §7, which is an **sm120 (NVIDIA)** measurement, and it travelled
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
    let want = std::env::var("PLOW_GLM_GF")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
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
/// plans/glm-mla-flash-tuning.md and Plow.SplitK (the split reduction equals the sequential sum for
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
    std::env::var("PLOW_GLM_NS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
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
    let on = c.tp > 1 && std::env::var("GLM_SHARD_HEAD").ok().as_deref() == Some("1");
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
    enc == MoeEnc::Fp8Blk && std::env::var("GLM_LINEAR_FP8").ok().as_deref() == Some("1")
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
    glm_linear_fp8(enc) && std::env::var("GLM_SHARED_GLU_SPLIT").ok().as_deref() == Some("1")
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
pub(crate) const MPF_BM: u32 = 64;

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
    gin: u32,   // input_layernorm
    qad: u32,   // q_a_proj (H->QL)
    gqa: u32,   // q_a_layernorm
    wqa: u32,   // DERIVED absorbed q_nope   [NH*DK, QL]
    wqr: u32, // DERIVED RAW q_rope down (q_b_rope, NOT folded) [NH*DR, QL]; RoPE applied dynamically
    ckvd: u32, // DERIVED kv_a latent down    [DK, H]
    gkva: u32, // kv_a_layernorm
    krotd: u32, // DERIVED RAW k_rope down (kv_a rope slice, NOT folded) [DR, H]; RoPE applied dynamically
    wuv: u32,   // DERIVED absorbed value       [NH*DK, VD]
    wo: u32,    // o_proj (NH*VD -> H)
    gpost: u32, // post_attention_layernorm
    // MoE (sparse layers): router + shared expert + the two loader-filled pointer tables.
    wr: u32,   // mlp.gate.weight [E,H] bf16
    bias: u32, // mlp.gate.e_score_correction_bias [E] f32
    shg: u32,  // shared_experts.gate_proj
    shu: u32,  // shared_experts.up_proj
    shd: u32,  // shared_experts.down_proj
    ewt: u32,  // expert_weight_table [E*3] u64 device ptrs (loader-filled from bound experts)
    est: u32,  // expert_scale_table  [E*3] u64 device ptrs (block-fp8 scale grids)
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
    let rows = rows.max(1) as u64;
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
    let kvlen = b.tensor("in.kvlen", I32);
    // Interleaved partial-RoPE cos/sin tables for the 64 rope dims (GLM theta 8e6, from the
    // config — full rotation of DR).
    // Same [ctx][DR/2] layout the half-split path uses (freq index = element>>1); the interp's HD=64
    // dispatch selects the INTERLEAVE=true template. See rope_tables + op_norm.h.
    // `c.rope_theta()`, not the field: a NoPE cfg refuses here rather than materialising tables
    // for a rotation the model does not have.
    let [cos_t, sin_t] = GenTensor::rope_pair(ctx, c.qk_rope, c.rope_theta(), 1.0, RopeScale::None);
    let cos = b.tensor_gen("in.cos", cos_t.byte_len(), cos_t);
    let sin = b.tensor_gen("in.sin", sin_t.byte_len(), sin_t);
    let emb = b.tensor(&format!("{}embed_tokens.weight", c.prefix), (c.vocab * h) as u64 * BF16);
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
    let osplits = ns.max(rows as u32);
    let opart = ac(b, "opart", (nh_l * osplits * dk) as u64 * F32);
    let mlpart = ac(b, "mlpart", (nh_l * osplits * 2) as u64 * F32);
    let olat = ac(b, "olat", (nh_l * dk) as u64 * BF16);
    let oat = ac(b, "oat", rows * (nh_l * vd) as u64 * BF16);
    let attn = ac(b, "attn", rows * h as u64 * BF16);
    let xmid = ac(b, "xmid", rows * h as u64 * BF16);
    let xn2 = ac(b, "xn2", rows * h as u64 * BF16);
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
    let shfu_up = if (rows > 1 && (enc == MoeEnc::Mxfp4 || glm_linear_fp8(enc)))
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
    let part = ac(b, "part", rows * (tk * h) as u64 * F32);
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
        (TENSOR_NONE, TENSOR_NONE, TENSOR_NONE, TENSOR_NONE, TENSOR_NONE, TENSOR_NONE)
    };
    let xnext = ac(b, "xnext", rows * h as u64 * BF16);
    let logits = ac(b, "logits", glm_vocab_l(c) as u64 * BF16);
    let amax = ac(b, "amax.part", AMAX_BLOCKS as u64 * 8);
    // TP peer-mapped partials (§7a) — only under sharding; the host binds these into peer scratch at
    // offset 0 / slot_b so the row-parallel o_proj + MoE/dense down write peer-visible partials that
    // XReduce sums. zero_h is a persistent zero buffer used as the MoeCombine residual under TP (the
    // real residual xmid is added AFTER the all-reduce, so it is not summed N times).
    // og_tp is the o_proj partial on BOTH phases — prefill all-reduces a [T,hidden] partial through
    // XReduceTwoShot (plans/tp-prefill.md), so it is row-dimensioned. dg_tp only ever carries the
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
    let (qidx, kidx_raw, kidx_normed, widx, iscore, iidx, ighist, igctl, icos, isin) = if dsa {
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
            b.tensor_gen("in.icos", ct.byte_len(), ct),
            b.tensor_gen("in.isin", st.byte_len(), st),
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
    let mut ckv = Vec::new();
    let mut krot = Vec::new();
    let mut kidx = Vec::new();
    let mut lw = Vec::new();
    for &l in layer_ids {
        ckv.push(b.tensor(&format!("kv.{l}.ckv"), (ctx * dk) as u64 * BF16));
        krot.push(b.tensor(&format!("kv.{l}.krot"), (ctx * dr) as u64 * BF16));
        // per-'full'-layer indexer key cache [ctx][DI] (accumulates like ckv/krot); shared layers none.
        let full = dsa && c.indexer_is_full(l);
        kidx.push(if full {
            b.tensor(&format!("kv.{l}.kidx"), (ctx * di) as u64 * BF16)
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
                b.tensor(&format!("{pfx}layers.{l}.{s}_scale"), n * k.div_ceil(MX_BLOCK as u64))
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
            wqa_s: mxs(b, "self_attn.derived.q_absorb.weight", (nh_l * dk) as u64, ql as u64),
            wqr_s: mxs(b, "self_attn.derived.q_rope.weight", (nh_l * dr) as u64, ql as u64),
            ckvd_s: mxs(b, "self_attn.derived.kv_a_latent.weight", dk as u64, h as u64),
            krotd_s: mxs(b, "self_attn.derived.k_rope.weight", dr as u64, h as u64),
            wo_s: if lin_fp8 {
                q8s(b, "self_attn.o_proj", h as u64, (nh_l * vd) as u64)
            } else {
                mxs(b, "self_attn.o_proj.weight", h as u64, (nh_l * vd) as u64)
            },
            wr_s: if dense { TENSOR_NONE } else { mxs(b, "mlp.gate.weight", e as u64, h as u64) },
            shg_s: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8s(b, "mlp.shared_experts.gate_proj", imoe_l as u64, h as u64)
            } else {
                mxs(b, "mlp.shared_experts.gate_proj.weight", imoe_l as u64, h as u64)
            },
            shu_s: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8s(b, "mlp.shared_experts.up_proj", imoe_l as u64, h as u64)
            } else {
                mxs(b, "mlp.shared_experts.up_proj.weight", imoe_l as u64, h as u64)
            },
            shd_s: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8s(b, "mlp.shared_experts.down_proj", h as u64, imoe_l as u64)
            } else {
                mxs(b, "mlp.shared_experts.down_proj.weight", h as u64, imoe_l as u64)
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
            ckvd: tw(b, "self_attn.derived.kv_a_latent.weight", dk as u64, h as u64),
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
                tw(b, "mlp.shared_experts.gate_proj.weight", imoe_l as u64, h as u64)
            },
            shu: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8(b, "mlp.shared_experts.up_proj", imoe_l as u64, h as u64)
            } else {
                tw(b, "mlp.shared_experts.up_proj.weight", imoe_l as u64, h as u64)
            },
            shd: if dense {
                TENSOR_NONE
            } else if lin_fp8 {
                q8(b, "mlp.shared_experts.down_proj", h as u64, imoe_l as u64)
            } else {
                tw(b, "mlp.shared_experts.down_proj.weight", h as u64, imoe_l as u64)
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
                    _ => t(b, "mlp.gate_proj.weight_scale_inv", (db_l * hb) as u64 * F32),
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
                    _ => t(b, "mlp.up_proj.weight_scale_inv", (db_l * hb) as u64 * F32),
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
                    _ => t(b, "mlp.down_proj.weight_scale_inv", (hb * db_l) as u64 * F32),
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
        kidx,
        lw,
    }
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
    match std::env::var("GLM_SPINE_CUS").ok().and_then(|v| v.parse::<u32>().ok()) {
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
fn wgfit() -> bool {
    std::env::var("PLOW_GLM_WGFIT").ok().as_deref() != Some("0")
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
    let nblk = cus.len() as u32;
    if !wgfit() || nblk == 0 || n == 0 {
        return cus.to_vec();
    }
    let per = n.div_ceil(nblk);
    let need = n.div_ceil(per).clamp(1, nblk);
    cus[..need as usize].to_vec()
}

/// Emit the shared MLA attention sub-block (input norm -> q/kv down + absorbed folds -> dynamic
/// interleaved RoPE on the 64 rope dims -> FLASH_MLA_DECODE -> merge -> O_UV_FOLD -> o_proj ->
/// residual -> post-attention norm). Writes `n.xn2` (the FFN input) and returns the post-attn-norm
/// completion dep. IDENTICAL for the dense (0-2) and MoE (3-77) layers, so both blocks call it.
#[allow(clippy::too_many_arguments)]
fn emit_glm_mla(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
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
        let op = if enc == MoeEnc::Mxfp4 { DevOp::GemvMxfp4 } else { DevOp::Gemv };
        b.emit(op, all.clone(), deps, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = wt;
            if enc == MoeEnc::Mxfp4 {
                d.t[3] = sc;
            }
            d.i[0] = 1;
            d.i[1] = nn;
            d.i[2] = k;
            d.f[0] = 1.0;
        })
    };
    // Standard Gemma-proven decode fusions (plans/glm52-fusion-audit.md). Each defaults ON; set the
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
    let fuse_a = std::env::var("PLOW_GLM_FUSE_A").ok().as_deref() != Some("0");
    let fuse_g = std::env::var("PLOW_GLM_FUSE_G").ok().as_deref() != Some("0");
    // B1 defaults OFF (opt-in): AddNorm reduces over the un-rounded a+b sum, so unlike A/G it is NOT
    // byte-identical to the split Residual+RmsNorm — a reorder-level fp diff that flips one early
    // greedy argmax and cascades. Ship it only behind the HF-coherence gate; PLOW_GLM_FUSE_B1=1 opts in.
    let fuse_b1 = std::env::var("PLOW_GLM_FUSE_B1").ok().as_deref() == Some("1");

    // --- MLA ---
    // 1 input_layernorm
    // `pre` chains this layer's first op to the PREVIOUS layer's output (x_in), so the 78 layers run
    // in sequence rather than racing on the shared scratch/x buffers. Empty for the single-layer gate
    // (x_in is pre-uploaded before the launch, so no on-device producer to wait on).
    let c_rn1 = b.emit(DevOp::RmsNorm, one.clone(), pre, |d| {
        d.t[0] = n.xn;
        d.t[1] = x_in;
        d.t[2] = w.gin;
        d.i[0] = 1;
        d.i[1] = h;
        d.f[0] = eps;
    });
    // 2/6/8 down-projections. FUSION A (audit §A): q_a, kv_a and k_rope ALL read n.xn with K=h, so
    //   their output columns concatenate into ONE GemvQkv (Nq=ql q_a, Nk=dk kv_a, Nv=dr k_rope) that
    //   fills every wave (fixing the k_rope/kv_a CU-starvation) and deletes 2 gates/layer. Byte-exact
    //   to the three Gemvs. Legal: M*K = h fits GM_LDS_HALVES.
    let (c_qad, c_ckvd, c_krr) = if fuse_a {
        // n = ql + dk + dr concatenated columns; `blocked_gemv_cus` drops the ceiling tail that
        // owns none of them (GLM TP4: 2624 over 256 => slices 239..255 are empty).
        let fa_cus = blocked_gemv_cus(&all, ql + dk + dr);
        let fa_op = if mx { DevOp::GemvQkvMxfp4 } else { DevOp::GemvQkv };
        let c_fa = b.emit(fa_op, fa_cus, &[c_rn1], |d| {
            d.t[0] = n.qlr;
            d.t[1] = n.xn;
            d.t[2] = w.qad; // q_a   -> Nq=ql
            d.t[3] = n.ckvraw;
            d.t[4] = w.ckvd; // kv_a  -> Nk=dk
            d.t[5] = n.krr;
            d.t[6] = w.krotd; // k_rope-> Nv=dr
            d.i[0] = 1;
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
    let c_rnq = b.emit(DevOp::RmsNorm, one.clone(), &[c_qad], |d| {
        d.t[0] = n.qlat;
        d.t[1] = n.qlr;
        d.t[2] = w.gqa;
        d.i[0] = 1;
        d.i[1] = ql;
        d.f[0] = eps;
    });
    // 4/5 absorbed q_nope (Wqa: QL -> NH_l*DK) and q_rope raw down (Wqr: QL -> NH_l*DR). FUSION G
    //   (audit §G): both read n.qlat with K=ql, so fuse into ONE GemvQkv with Nv=0 (q half + k half).
    //   Byte-exact. q_rope then gets a dynamic INTERLEAVED RoPE per head at pos (no norm); HD=64
    //   selects the interleaved template; q is not cached (out_row0/stride 0).
    let (c_qa, c_qrr) = if fuse_g {
        let fg_cus = blocked_gemv_cus(&all, nh_l * dk + nh_l * dr);
        let fg_op = if mx { DevOp::GemvQkvMxfp4 } else { DevOp::GemvQkv };
        let c_fg = b.emit(fg_op, fg_cus, &[c_rnq], |d| {
            d.t[0] = n.qa;
            d.t[1] = n.qlat;
            d.t[2] = w.wqa; // q_nope   -> Nq=nh_l*dk
            d.t[3] = n.qrr;
            d.t[4] = w.wqr; // q_rope raw-> Nk=nh_l*dr
            d.t[5] = TENSOR_NONE;
            d.t[6] = TENSOR_NONE; // Nv=0 (v branch never taken)
            d.i[0] = 1;
            d.i[1] = nh_l * dk;
            d.i[2] = ql;
            d.i[3] = nh_l * dr;
            d.i[4] = 0;
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
    let rq = rope_cus(&all, 0, 1, nh_l);
    let rk = rope_cus(&all, rq.len(), 1, 1);
    let c_qr = b.emit(DevOp::HeadNormRope, rq.clone(), &[c_qrr], |d| {
        d.t[0] = n.qr;
        d.t[1] = n.qrr;
        d.t[2] = TENSOR_NONE;
        d.t[3] = n.cos;
        d.t[4] = n.sin;
        d.t[5] = n.pos;
        d.i[0] = 1;
        d.i[1] = nh_l;
        d.i[2] = dr;
        d.i[3] = 0;
        d.i[4] = 1;
        d.f[0] = eps;
        d.j[0] = 0;
        d.j[1] = KV_MASK_NONE;
    });
    // 7 kv_a_layernorm -> writes the latent cache (current row = row 0 here; the loader/decode
    //   step rebases the output to the current position, matching the ckv-row write of a decode step).
    //   Reads n.ckvraw from the fused (or split) down-projection above.
    let c_rnkv = b.emit(DevOp::RmsNorm, one.clone(), &[c_ckvd], |d| {
        d.t[0] = n.ckv[slot];
        d.t[1] = n.ckvraw;
        d.t[2] = w.gkva;
        d.i[0] = 1;
        d.i[1] = dk;
        d.f[0] = eps;
    });
    // 8 k_rope dynamic INTERLEAVED RoPE (shared 1-head) on n.krr from the fused (or split) down-proj,
    //   writing the rope cache at row=out_row0 (i[3]; the decode step patches it to the current pos).
    let c_krd = b.emit(DevOp::HeadNormRope, rk.clone(), &[c_krr], |d| {
        d.t[0] = n.krot[slot];
        d.t[1] = n.krr;
        d.t[2] = TENSOR_NONE;
        d.t[3] = n.cos;
        d.t[4] = n.sin;
        d.t[5] = n.pos;
        d.i[0] = 1;
        d.i[1] = 1;
        d.i[2] = dr;
        d.i[3] = 0;
        d.i[4] = 1;
        d.f[0] = eps;
        d.j[0] = 0;
        d.j[1] = KV_MASK_NONE;
    });
    // --- DSA lightning indexer (G2/G5): ctx>2048 => project q_idx/k_idx/w, score, top-k select ->
    //     idx table, then FLASH_GATHER over the top_k selected latent rows. ctx<=2048 => dense flash
    //     (top-k is a no-op). 'full' layers own the indexer; 'shared' layers reuse the last full
    //     layer's idx (sequential layer chain => n.iidx already holds it). q_idx/k_idx use a HD=DI GPT-J
    //     interleaved RoPE with the identity-tail table (rope the first qk_rope=DR dims, pass the rest).
    let dsa = c.dsa(ctx);
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
    let mut fl_deps = vec![c_qa, c_qr, c_rnkv, c_krd];
    if full {
        fl_deps.push(c_sel);
    }
    //   Sized to `n_batch*n_tok*(nh_l/GF)*nsplit` (`flash_mla_cus`), which is exactly `n_cu` at
    //   GF=4 and half of it at GF=2 — see the helper for why the two do not cancel.
    let c_fl = b.emit(
        if dsa {
            DevOp::FlashGatherDecode
        } else {
            DevOp::FlashMlaDecode
        },
        flash_mla_cus(&all, 1, 1, nh_l, glm_gf(ctx, nh_l), ns_attn),
        &fl_deps,
        |d| {
            d.t[0] = n.opart;
            d.t[1] = n.mlpart;
            d.t[2] = n.qa;
            d.t[3] = n.qr;
            d.t[4] = n.ckv[slot];
            d.t[5] = n.krot[slot];
            d.t[6] = n.kvlen;
            d.i[0] = 1;
            d.i[1] = nh_l;
            d.i[2] = ctx;
            d.i[4] = ns_attn;
            d.i[5] = KV_MASK_NONE;
            d.i[7] = glm_gf(ctx, nh_l); // per-pkt head-fusion factor (interp dispatches 2/4/8 on this)
            d.f[0] = c.attn_scale;
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
    let c_uv = b.emit(DevOp::MlaMergeFold, mla_fold_cus(&all, nh_l, vd), &[c_fl], |d| {
        d.t[0] = n.oat;
        d.t[1] = n.opart;
        d.t[2] = n.mlpart;
        d.t[3] = w.wuv;
        d.i[0] = 1;
        d.i[1] = nh_l;
        d.i[2] = vd;
        d.i[4] = ns_attn;
    });
    // 12 o_proj (NH_l*VD -> H)  [row-parallel]: each rank sums its head-shard into a PARTIAL H-vector.
    //   Under TP the partial goes to the peer-mapped og_tp slot and an XReduce all-reduces the N
    //   partials into n.attn; at tp==1 o_proj writes n.attn directly (byte-identical).
    // PLOW_NO_XREDUCE (diagnostic): drop the 156 all-reduce collectives (o_proj writes n.attn
    // directly with only this rank's partial) — numerically WRONG but same graph minus the
    // cross-GPU rendezvous, to isolate the XReduce cost. Never set for a real decode.
    let no_xr = tp > 1 && std::env::var("PLOW_NO_XREDUCE").ok().as_deref() == Some("1");
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
                d.i[0] = 1;
                d.i[1] = h;
                d.i[2] = nh_l * vd;
                d.i[4] = 0;
            })
        } else {
            gemv(b, out, n.oat, w.wo, w.wo_s, h, nh_l * vd, deps)
        }
    };
    let c_op = if tp > 1 && !no_xr {
        let c_p = o_gemv(b, n.og_tp, &[c_uv]);
        emit_xreduce(b, xgate, true, xr_cus, c_p, n.attn, h, tp, 0)
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
            d.i[0] = 1;
            d.i[1] = h;
            d.f[0] = eps;
        })
    } else {
        let c_rs = b.emit(DevOp::Residual, spine_cus(b.n_cu()), &[c_op], |d| {
            d.t[0] = n.xmid;
            d.t[1] = x_in;
            d.t[2] = n.attn;
            d.i[0] = h;
            d.f[0] = 1.0;
        });
        b.emit(DevOp::RmsNorm, one.clone(), &[c_rs], |d| {
            d.t[0] = n.xn2;
            d.t[1] = n.xmid;
            d.t[2] = w.gpost;
            d.i[0] = 1;
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
    let one = vec![0u32];
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
    let c_rn1 = b.emit(DevOp::RmsNorm, one.clone(), pre, |d| {
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
    let c_rnq = b.emit(DevOp::RmsNorm, one.clone(), &[c_qad], |d| {
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
    let c_rnkv = b.emit(DevOp::RmsNorm, one.clone(), &[c_ckvd], |d| {
        d.t[0] = n.ckv[slot];
        d.t[1] = n.ckvraw;
        d.t[2] = w.gkva;
        d.i[0] = t;
        d.i[1] = dk;
        d.f[0] = eps;
    });
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
    //   NOT FlashGatherPrefill, even when the DSA gate is armed. The gathered prefill wants one
    //   top_k row PER QUERY (`idx[b][t][top_k]`), and nothing produces that: `IndexScore` scores a
    //   single query (`t0=Score(f32[ctx])`) and `IndexSelect` emits one `iidx[top_k]`. Emitting the
    //   gather against the decode selector's single row would silently give every query token the
    //   LAST token's selection. Dense MLA prefill is correct at every ctx (it is what the crossover
    //   compares against), just ctx-linear — so that is what is emitted until a per-query selector
    //   exists. See the report note; this is the one arm the parent task asked for that is blocked.
    let c_fl = b.emit(DevOp::FlashMlaPrefill, all.clone(), &[c_qa, c_qr, c_rnkv, c_krd], |d| {
        d.t[0] = n.opart;
        d.t[1] = n.mlpart;
        d.t[2] = n.qa;
        d.t[3] = n.qr;
        d.t[4] = n.ckv[slot];
        d.t[5] = n.krot[slot];
        d.t[6] = n.kvlen;
        d.i[0] = 1; // n_batch (single sequence per prefill chunk)
        d.i[1] = nh_l; // PER-RANK heads
        d.i[2] = ctx; // kv_stride
        d.i[3] = 0; // window: 0 = full causal (MLA has no sliding regime)
        d.i[4] = t; // n_tok — the slot decode used for nsplit
        d.i[5] = KV_MASK_NONE;
        d.i[7] = glm_gf_prefill(ctx, nh_l);
        d.f[0] = c.attn_scale;
    });
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
    let c_uv = b.emit(DevOp::MlaMergeFold, mla_fold_cus(&all, t * nh_l, vd), &[c_fl], |d| {
        d.t[0] = n.oat;
        d.t[1] = n.opart;
        d.t[2] = n.mlpart;
        d.t[3] = w.wuv;
        d.i[0] = t;
        d.i[1] = nh_l;
        d.i[2] = vd;
        d.i[4] = 1; // nsplit — forced to 1 by the prefill precondition above
    });
    // 12 o_proj, row-parallel over this rank's head shard. Under TP the [T,hidden] partial goes
    //    through the TWO-SHOT all-reduce (reduce-scatter + all-gather), not decode's one-shot: the
    //    partial is bandwidth-bound at T rows, so the two-shot moves ~tp/2x less over the fabric
    //    (plans/tp-prefill.md §4). `emit_xreduce(decode=false)` is that path, already in the tree.
    //
    //    Under GLM_LINEAR_FP8 `w.wo`/`w.wo_s` are the CHECKPOINT's block-fp8 bytes and its
    //    [128,128] weight_scale_inv grid, not the bf16 the prep dequantises to — the same pair the
    //    decode emitter binds to `GemvFp8Blk` (44). This is the T-row arm; before it existed
    //    `declare_glm_rows` refused the combination rather than put a bf16 `Gemm` on fp8 bytes.
    //    It is on the DENSE layers too: `emit_glm_dense_block_prefill` calls this same emitter.
    let lin_fp8 = glm_linear_fp8(enc);
    let oproj = |b: &mut Builder, out: u32, deps: &[u32]| -> u32 {
        if lin_fp8 {
            emit_pf_gemm_fp8_blk(b, &all, out, n.oat, w.wo, w.wo_s, t, h, nh_l * vd, deps)
        } else {
            gemm(b, out, n.oat, w.wo, w.wo_s, h, nh_l * vd, deps)
        }
    };
    let no_xr = tp > 1 && std::env::var("PLOW_NO_XREDUCE").ok().as_deref() == Some("1");
    let c_op = if tp > 1 && !no_xr {
        let c_p = oproj(b, n.og_tp, &[c_uv]);
        emit_xreduce(b, xgate, false, xr_cus, c_p, n.attn, t * h, tp, 0)
    } else {
        oproj(b, n.attn, &[c_uv])
    };
    // 13/14 post-attn residual + post_attention_layernorm. The decode path can fuse these into one
    //   AddNorm (fusion B1, opt-in); prefill keeps them split for the same reason the dense path
    //   does — T rows already parallelise the norm, so the fusion buys a gate and costs the
    //   byte-identity that made B1 opt-in in the first place.
    let c_rs = b.emit(DevOp::Residual, one.clone(), &[c_op], |d| {
        d.t[0] = n.xmid;
        d.t[1] = x_in;
        d.t[2] = n.attn;
        d.i[0] = t * h;
        d.f[0] = 1.0;
    });
    b.emit(DevOp::RmsNorm, one.clone(), &[c_rs], |d| {
        d.t[0] = n.xn2;
        d.t[1] = n.xmid;
        d.t[2] = w.gpost;
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
    let c_router = b.emit(DevOp::MoeRouterTopkPf, all.clone(), &[c_score], |d| {
        d.t[0] = n.tab;
        d.t[1] = n.rlogit;
        d.t[3] = w.bias;
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
                emit_pf_gemm_fp8_blk(b, &all, n.shfu, n.xn2, w.shg, w.shg_s, t, imoe_l, h, &[c_rn2]),
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
        emit_pf_gemm_fp8_blk(b, &all, n.shared, n.shfu, w.shd, w.shd_s, t, h, imoe_l, &[c_shglu])
    } else {
        gemm(b, n.shared, n.shfu, w.shd, w.shd_s, h, imoe_l, &[c_shglu])
    };
    // 18 grouped gate/up + GLU over the sorted rows. A is gathered from xn2 by row_token, so an
    //    expert's gate|up crosses HBM ONCE for every token that chose it — the reuse decode cannot
    //    have. i[3] picks the block-fp8 or bf16 weight arm from the same tables decode uses.
    let c_g = b.emit(DevOp::MoeGroupGluPf, all.clone(), &[c_align, c_rn2], |d| {
        d.t[0] = n.fu_g;
        d.t[1] = n.xn2;
        d.t[2] = w.ewt;
        d.t[3] = w.est;
        d.t[4] = n.meta;
        d.t[5] = n.row_token;
        // A4W4 binds two more: t6 = row_partidx, so the fused bridge can tell a PAD row from a live
        // one and skip it (the bf16/fp8 arms let pad rows fall out in DOWN's scatter instead, but a
        // bridge that quantized them would write E8M0 bytes for rows nothing reads); t7 = the E8M0
        // scale rows it WRITES, because the bridge is this op's epilogue rather than a separate op.
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
    // 19 grouped down + gate-scale + SCATTER into part[T*k, H]. row_partidx carries each gathered
    //    row's fixed destination, so the align op's nondeterministic within-expert row ORDER does
    //    not reach the output — the combine below still sums part in FIXED slot order.
    let c_d = b.emit(DevOp::MoeGroupDownPf, all.clone(), &[c_g], |d| {
        d.t[0] = n.part;
        d.t[1] = n.fu_g;
        d.t[2] = w.ewt;
        d.t[3] = w.est;
        d.t[4] = n.meta;
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
    });
    // 20 T-token combine. Under TP `shared`/`part` are PARTIALS, so the combine residual must NOT be
    //    xmid (XReduce would sum it tp times): it writes the partial with a zero residual, the
    //    two-shot all-reduce folds the ranks, and a Residual then adds the real xmid. tp==1 keeps
    //    the fused xmid combine — the same structure the decode block uses, one phase up.
    let no_xr = tp > 1 && std::env::var("PLOW_NO_XREDUCE").ok().as_deref() == Some("1");
    if tp > 1 && !no_xr {
        let c_cmb = b.emit(DevOp::MoeCombinePf, all.clone(), &[c_shd, c_d], |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = n.zero_h;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
            d.i[2] = t;
        });
        // `n.slot_b`, NOT `t * h * 2`: the offset is a property of the BLOB (where the host binds
        // `act.dg_tp`), not of this bucket. See the field's header on GlmTn.
        let c_xr = emit_xreduce(b, xgate, false, xr_cus, c_cmb, n.attn, t * h, tp, n.slot_b);
        b.emit(DevOp::Residual, one.clone(), &[c_xr], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.attn;
            d.i[0] = t * h;
            d.f[0] = 1.0;
        })
    } else {
        b.emit(DevOp::MoeCombinePf, all.clone(), &[c_shd, c_d], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
            d.i[2] = t;
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
    });
    // Combine. Identical TP structure to the MoE prefill block: under TP the partial is combined
    // with a ZERO residual, two-shot all-reduced, and the real residual added after — folding xmid
    // in before the all-reduce would sum it tp times.
    let no_xr = tp > 1 && std::env::var("PLOW_NO_XREDUCE").ok().as_deref() == Some("1");
    if tp > 1 && !no_xr {
        let c_cmb = b.emit(DevOp::MoeCombinePf, all.clone(), &[c_d], |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = n.zero_h;
            d.t[2] = TENSOR_NONE; // no shared expert on a dense layer
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = DENSE_TOP_K;
            d.i[2] = t;
        });
        // `n.slot_b`, NOT `t * h * 2`: the offset is a property of the BLOB (where the host binds
        // `act.dg_tp`), not of this bucket. See the field's header on GlmTn.
        let c_xr = emit_xreduce(b, xgate, false, xr_cus, c_cmb, n.attn, t * h, tp, n.slot_b);
        b.emit(DevOp::Residual, one.clone(), &[c_xr], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.attn;
            d.i[0] = t * h;
            d.f[0] = 1.0;
        })
    } else {
        b.emit(DevOp::MoeCombinePf, all.clone(), &[c_d], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = TENSOR_NONE; // no shared expert on a dense layer
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = DENSE_TOP_K;
            d.i[2] = t;
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
    let w8a16 = std::env::var("PLOW_W8A16").ok().as_deref() == Some("1");
    assert!(
        std::env::var("PLOW_W8A8").ok().as_deref() != Some("1"),
        "PLOW_W8A8=1 is not implementable for the MLA+MoE family: its block-fp8 expert arms (ops \
         45/46/48/49) are w8a16 in every instantiation — fp8 weights, bf16 activations — so there \
         is nothing to quantize the activation with. Missing capability: `moe_w8a8`. Use \
         PLOW_W8A16=1 (or PLOW_FP8=1) for this family's fp8 profile, or PLOW_MXFP4=1 for A4W4, \
         which is the one path here that narrows the activation too."
    );
    let use_fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1") || w8a16;
    let mxfp4 = std::env::var("PLOW_MXFP4").ok().as_deref() == Some("1");
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
    match std::env::var("PLOW_MLA_PREFILL").ok().as_deref() {
        None | Some("") | Some("0") => (Vec::new(), PrefillScope::Attn),
        Some("1") => (glm_prefill_buckets(ctx), PrefillScope::Attn),
        Some("full") => (glm_prefill_buckets(ctx), PrefillScope::Full),
        Some(s) if s.starts_with("full:") => {
            (parse_list(&s["full:".len()..]), PrefillScope::Full)
        }
        Some(list) => (parse_list(list), PrefillScope::Attn),
    }
}

/// Emit ONE MoE (sparse) GLM decoder block — the exact block validated by the B4 harness. `slot`
/// indexes `tn.lw`/`tn.ckv`/`tn.krot`. `use_fp8` selects the block-fp8 expert opcodes (45/46) over
/// the bf16 ones (41/42). Returns the MoeCombine completion dep.
pub(crate) fn emit_glm_block(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
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
    let c_rn2 = emit_glm_mla(b, c, n, slot, ctx, enc, x_in, pre, xgate, xr_cus);
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
    // CONCURRENT EXPERT SEGMENTS (plans/glm52-coresident.md): the M=1 experts underfill 256 CUs
    // (latency-starved, ~12x above the weight-bandwidth roofline), so run the top_k chosen experts as
    // CO-RESIDENT segments — each owns a DISJOINT CU slice (tk experts x 256/tk CUs), all gated on the
    // SAME router counter, so all tk run at once instead of serially on all-256. Pure work-PARTITION
    // change (the kernel's slice/nblk mechanism does the rest): 0 = serial all-256 baseline, 1 =
    // concurrent experts (shared serial), 2 = concurrent experts + co-resident (proactive) shared expert.
    // SHIP DEFAULT = 1 (co-resident experts): bit-exact, measured -17.4% on the MoE block (the M=1
    // experts collapse from serial-all-256 to tk concurrent 256/tk-CU segments). GLM_MOE_CORESIDENT=0
    // restores the serial baseline; =2 adds the proactive co-resident shared expert (marginal, opt-in).
    let cores: u32 = std::env::var("GLM_MOE_CORESIDENT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
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
    let shared_w: u32 = std::env::var("GLM_SHARED_CUS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&s| s > 0 && s < n_cu)
        .unwrap_or(n_cu - tk * (n_cu / (tk + 1)));
    let routed_w = (n_cu - shared_w) / tk;
    assert!(routed_w > 0, "GLM_SHARED_CUS={shared_w} leaves no CUs for the {tk} routed experts");
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
    let router_cus: Vec<u32> =
        if cores >= 2 && std::env::var("GLM_ROUTER_OFF_SHARED").ok().as_deref() == Some("1") {
            (0..n_cu - shared_w).collect()
        } else {
            all.clone()
        };

    // --- MoE ---
    // 15 router. DEFAULT (split): the 256-expert x K=6144 score matmul is the ordinary MULTI-CU
    //   wave-cooperative GEMV (all.clone()) — was the single-CU scalar dot that measured 73% of the
    //   MoE layer — feeding a cheap 1-CU MoeRouterTopk tail (bit-exact selection). GLM_ROUTER_OLD=1
    //   emits the fused single-CU d_moe_router for the before/after A/B.
    let c_router = if std::env::var("GLM_ROUTER_OLD").ok().as_deref() == Some("1") {
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
        let score_op = if enc == MoeEnc::Mxfp4 { DevOp::GemvMxfp4 } else { DevOp::Gemv };
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
        let shglu_op = if enc == MoeEnc::Mxfp4 { DevOp::GemvGluMxfp4 } else { DevOp::GemvGlu };
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
        let shd_op = if enc == MoeEnc::Mxfp4 { DevOp::GemvMxfp4 } else { DevOp::Gemv };
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
    let no_xr = tp > 1 && std::env::var("PLOW_NO_XREDUCE").ok().as_deref() == Some("1");
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
        b.emit(DevOp::Residual, spine_cus(b.n_cu()), &[c_xr], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.attn;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
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

/// The dense-FFN down projection's opcode for an encoding. Block-fp8 and bf16 both go through
/// `GEMV_FP8_BLK` (the dense down has no bf16-specific op); MXFP4 has its own.
fn dense_down_op(enc: MoeEnc) -> DevOp {
    if enc == MoeEnc::Mxfp4 {
        DevOp::GemvMxfp4
    } else {
        DevOp::GemvFp8Blk
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
    enc: MoeEnc,
    x_in: u32,
    x_out: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    assert!(slot < n.lw.len(), "slot out of range");
    let c_rn2 = emit_glm_mla(b, c, n, slot, ctx, enc, x_in, pre, xgate, xr_cus);
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
    let c_glu = if enc == MoeEnc::Mxfp4 {
        b.emit(DevOp::GemvGluMxfp4, all.clone(), &[c_rn2], |d| {
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
        })
    } else {
        b.emit(DevOp::DenseGluFp8Blk, all.clone(), &[c_rn2], |d| {
            d.t[0] = n.dfu;
            d.t[1] = n.xn2;
            d.t[2] = w.dgate;
            d.t[5] = w.dup;
            d.t[3] = w.dgate_s;
            d.t[4] = w.dup_s;
            d.i[0] = di_l;
            d.i[1] = h;
            d.i[5] = GLM_ACT_SILU;
        })
    };
    // dense down (block-fp8 GEMV, op 44) — row-parallel (di_l input). Under TP writes a PARTIAL into
    //   the dg_tp peer slot, XReduce all-reduces into n.attn, then residual; at tp==1 writes n.shared
    //   and the residual reads it directly (byte-identical).
    let no_xr = tp > 1 && std::env::var("PLOW_NO_XREDUCE").ok().as_deref() == Some("1");
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
        b.emit(DevOp::Residual, spine_cus(b.n_cu()), &[c_xr], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.attn;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
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

/// GLM-5.2 emit entry (Stack B: .pkt bound by name from the host-prepped weight dir). Milestone-1
/// emits the SINGLE-layer MoE block program the validation harness runs against the HF oracle; the
/// full 78-layer decode + dense layers + TP sharding are the next milestones.
/// Full 78-layer GLM-5.2 DECODE program (M=1): embed -> [dense 0-2 | MoE 3-77] ping-ponged -> final
/// norm -> lm_head -> argmax (writes the sampled id back into in.ids). Layers 0-77 (78 = MTP head,
/// skipped). Per-layer ckv/krot caches; the decode loop patches the current-token cache row per step
/// (k_rope HeadNormRope out_row0 via kv_row_insts; ckv RMSNORM output via a per-step pointer rebind).
/// `use_fp8` selects the block-fp8 expert kernels (45/46) for the MoE layers.
#[allow(clippy::too_many_arguments)]
fn glm_emit_full(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, use_fp8: bool, rope_gen: bool, target: &str, verify: Option<&crate::VerifyHook>) {
    let mut c = cfg_glm(dir);
    c.tp = tp;
    // GLM_NLAYERS truncates the model to the first N layers — a single-GPU smoke test of the decode
    // LOOP mechanics (embed/chain/KV-row patch/argmax/multi-step) that fits without TP or all 78
    // layers' weights. Default = full 0..77 (layer 78 = MTP, skipped).
    let nl = std::env::var("GLM_NLAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(c.layers)
        .min(c.layers);
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
    // decode-only emit byte-identical to one from before this path existed.
    let tn = declare_glm_rows(&mut tb, &c, ctx, &layers, max_rows, enc);
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();

    // One program per prefill bucket, ahead of decode. Same layer chain, same dense/MoE split, the
    // T-row emitters throughout; the lm_head tail samples the LAST row (see `emit_glm_tail`).
    let mut progs = Vec::new();
    let mut prog_t = Vec::new();
    for &t in &pf {
        let mut pb = Builder::new(n_cu);
        pb.adopt_tensors(tensors.clone());
        let pall = pb.all();
        let pxr: Vec<u32> = pall.clone();
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
                    &mut pb, &c, &tn, slot, ctx, t, enc, cur, nxt, &dep, &mut pxgate, &pxr,
                )
            } else {
                emit_glm_block_prefill(
                    &mut pb, &c, &tn, slot, ctx, t, enc, cur, nxt, &dep, &mut pxgate, &pxr,
                )
            };
            dep = vec![d];
            cur = nxt;
        }
        emit_glm_tail(&mut pb, &c, &tn, cur, &dep, t, &mut pxgate);
        progs.push(pb.finish());
        prog_t.push(t);
    }

    let mut b = Builder::new(n_cu);
    b.adopt_tensors(tensors.clone());
    let all = b.all();

    // embed: in.ids[0] -> x  (GLM has no embedding scale)
    let c_emb = b.emit(DevOp::Embed, all.clone(), &[], |d| {
        d.t[0] = tn.x;
        d.t[1] = tn.emb;
        d.t[2] = tn.ids;
        d.i[0] = 1;
        d.i[1] = c.hidden;
        d.f[0] = 1.0;
    });
    // 78 decoder layers, ping-ponging x <-> xnext so layer l+1 reads layer l's output. Each layer's
    // first op waits on the previous layer's completion (`dep`) — the layers run in sequence.
    // XReduce collectives (decode one-shot): each o_proj + FFN-down all-reduce takes a unique xctr
    // gate id (allocated by xgate). At tp==1 no XReduce is emitted. The all-reduce runs on `all` CUs
    // by default; PLOW_XR_CUS caps it (the TP8 NUMA-crossing lever, plans/tp-design.md §8b).
    let mut xgate: u32 = 0;
    let xr_cus: Vec<u32> = {
        let k = std::env::var("PLOW_XR_CUS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        match k {
            Some(k) if k > 0 && k < n_cu => (0..k).collect(),
            _ => all.clone(),
        }
    };
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
    emit_glm_tail(&mut b, &c, &tn, cur, &[dep], 1, &mut xgate);
    let prog = b.finish();
    let n_ops = prog.insts.len();
    progs.push(prog);
    prog_t.push(1);

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
fn emit_glm_tail(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    x_final: u32,
    dep: &[u32],
    rows: u32,
    xgate: &mut u32,
) {
    let all = b.all();
    // `vocab_l` is the full vocab when replicated and vocab/tp under GLM_SHARD_HEAD. The GEMV, the
    // argmax and the tensor declarations all read this same helper, so the arms cannot drift.
    let vocab_l = glm_vocab_l(c);
    let c_f = b.emit(DevOp::RmsNorm, vec![0u32], dep, |d| {
        d.t[0] = n.xn;
        d.t[1] = x_final;
        d.t[2] = n.fin;
        d.i[0] = rows;
        d.i[1] = c.hidden;
        d.f[0] = c.eps;
    });
    let c_lm = b.emit(DevOp::Gemv, all, &[c_f], |d| {
        d.t[0] = n.logits;
        d.t[1] = n.xn;
        d.t[2] = n.head;
        d.i[0] = 1;
        d.i[1] = vocab_l;
        d.i[2] = c.hidden;
        d.i[4] = rows - 1; // a_row0: the last real row; host re-patches per chunk
    });
    let c_am = b.emit(DevOp::Argmax, (0..AMAX_BLOCKS).collect(), &[c_lm], |d| {
        d.t[0] = n.amax;
        d.t[1] = n.logits;
        d.i[0] = vocab_l;
    });
    if glm_shard_head(c) {
        // XARGMAX_FIN SUBSUMES ArgmaxFin: it folds the AMAX_BLOCKS partials itself, rebases the
        // winning index by rank*vocab_l and takes the cross-rank max, so emitting both would fold
        // twice and write the LOCAL winner's id first. Two xctr ids from this program's allocator:
        // the arrival gate and the peer-visible 8-byte value slot — distinct, because the gate is
        // an atomic counter and the slot is data.
        let gate = *xgate;
        *xgate += 2;
        // The fold publishes one u64 per sequence into ONE 128-byte xctr counter line, so it can
        // carry at most PLOW_XAMAX_MAX_BATCH = 16. See the header for why this is 1 at every
        // `rows`; assert rather than let a future batched emit silently leave ids[16..] holding
        // the previous step's token.
        const XAMAX_MAX_BATCH: u32 = 16;
        let n_batch = 1u32;
        assert!(
            n_batch <= XAMAX_MAX_BATCH,
            "XARGMAX_FIN carries at most {XAMAX_MAX_BATCH} sequences in one xctr counter line"
        );
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
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_main(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, rope_gen: bool, target: &str, verify: Option<&crate::VerifyHook>) {
    let enc = mla_moe_enc_env(dir);
    let use_fp8 = enc == MoeEnc::Fp8Blk;
    // Full 78-layer serving decode program (GLM_FULL=1) vs the single-layer validation gate (default).
    if std::env::var("GLM_FULL").ok().as_deref() == Some("1") {
        glm_emit_full(dir, ctx, out, n_cu, tp, use_fp8, rope_gen, target, verify);
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
    let layer: u32 = std::env::var("GLM_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(c.first_k_dense);
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
}

/// Build a single-block (layers `block`) MLA+MoE program + its descriptor, no file IO — the testable
/// core of `--block` on the GLM emit path (plans/block-asset-harness.md §5.3, §7) and, via `arch`,
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
    glm_build_block_pf(c, ctx, n_cu, block, use_fp8, model, arch, &[], PrefillScope::Attn, enc)
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
    let tn = declare_glm_rows(&mut tb, c, ctx, &layers, pf.iter().copied().max().unwrap_or(1), enc);
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
                emit_glm_mla_prefill(&mut pb, c, &tn, 0, ctx, t, enc, tn.x, &[], &mut pxgate, &pall);
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
                            &mut pb, c, &tn, slot, ctx, t, enc, cur, nxt, &dep, &mut pxgate, &pall,
                        )
                    } else {
                        emit_glm_block_prefill(
                            &mut pb, c, &tn, slot, ctx, t, enc, cur, nxt, &dep, &mut pxgate, &pall,
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
            emit_glm_dense_block(&mut b, c, &tn, slot, ctx, enc, cur, nxt, &dep, &mut xgate, &xr_cus)
        } else {
            emit_glm_block(
                &mut b, c, &tn, slot, ctx, enc, cur, nxt, &dep, &mut xgate, &xr_cus,
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
        dsa_role: is_glm.then(|| if full { "indexer".into() } else { "reuse".into() }),
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
pub(crate) fn glm_emit_block(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, spec: &str, rope_gen: bool, target: &str, verify: Option<&crate::VerifyHook>) {
    let mut c = cfg_glm(dir);
    c.tp = tp;
    // `--num-gpus N --parallel tp` reaches here as tp=N. The block emit is TP-parameterized end to
    // end — every head-dimensioned tensor and op field is this rank's nh_l = heads/tp shard, the
    // shared/dense intermediates are imoe_l/di_l, the routed experts stay WHOLE under EP, and the
    // o_proj / FFN-down partials go through XReduce — so the assert that used to pin this to tp=1
    // was describing an earlier state of the emitter, not a limitation of it. Verified by
    // `mla_prefill_tp_shapes_scale_with_tp` / `mla_prefill_tp_emits_two_shot_allreduce`.
    // The RUNTIME cannot serve tp>1 yet (no cross-GPU collectives); this is compiler-side emission.
    assert!(tp >= 1 && c.heads % tp == 0,
        "tp={tp} must divide n_head={} (each rank owns a whole head shard)", c.heads);
    let enc = mla_moe_enc_env(dir);
    let use_fp8 = enc == MoeEnc::Fp8Blk;
    let block = parse_block(spec, c.layers as usize);
    let model = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.to_string_lossy().into_owned());
    let (pf, scope) = glm_prefill_buckets_env(ctx);
    let (mut m, desc) =
        glm_build_block_pf(&c, ctx, n_cu, block.clone(), use_fp8, &model, MlaArch::Glm, &pf, scope, enc);
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

/// `--block` on the Kimi K2.7 / DeepSeek MLA+MoE path (plans/block-asset-harness.md §5.0/§5.3, M3).
/// Emits ONE block (layers `spec`) as a GPU-loadable PLOWDEV blob with a `SECT_METADATA` `block.json`
/// descriptor + sibling file. REUSES the GLM MLA + MoE emit verbatim (glm_build_block) with a Kimi
/// cfg (`has_dsa=false`) — no DSA, KV latent (ckv/krot) carried state, decode-only (the GLM emit has
/// no prefill program, so programs.prefill_buckets stays empty). `arch` picks the Kimi vs DeepSeek
/// descriptor tag.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn kimi_emit_block(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, spec: &str, arch: MlaArch, rope_gen: bool, target: &str, verify: Option<&crate::VerifyHook>) {
    let mut c = cfg_kimi(dir);
    c.tp = tp;
    // See the note in `glm_emit_block`: the shared MLA+MoE emit is TP-parameterized, so `--num-gpus
    // N` sharding is emission-complete on this path too. tp must divide the head count.
    assert!(tp >= 1 && c.heads % tp == 0,
        "tp={tp} must divide n_head={} (each rank owns a whole head shard)", c.heads);
    let enc = mla_moe_enc_env(dir);
    let use_fp8 = enc == MoeEnc::Fp8Blk;
    let block = parse_block(spec, c.layers as usize);
    let model = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.to_string_lossy().into_owned());
    let (pf, scope) = glm_prefill_buckets_env(ctx);
    let (mut m, desc) =
        glm_build_block_pf(&c, ctx, n_cu, block.clone(), use_fp8, &model, arch, &pf, scope, enc);
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

// ===== Nemotron-3 Mamba-2 hybrid (plans/block-asset-harness.md §7 Nemotron, §11 M4). =========
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
pub(crate) fn nemotron_emit_block(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, spec: &str, rope_gen: bool) {
    assert_eq!(tp, 1, "Nemotron TP sharding is a later milestone; use --tp 1 for --block");
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

// ===== Kimi-K3 (`kimi_k3` / text `kimi_linear`) — COMPILER FRONT END ONLY =====================
//
// K3 is a MULTIMODAL wrapper (`KimiK3ForConditionalGeneration`) around a `kimi_linear` text tower.
// The text tower is a HYBRID: 24 of its 93 layers are DeepSeek-style MLA, the other 69 are KDA
// (Kimi Delta Attention — a LINEAR attention with carried recurrent state). Its MoE is LATENT: the
// 896 routed experts read a 3584-wide projection of the hidden state, not the 7168 hidden state,
// and their GEMMs are mxfp4.
//
// Nothing here emits. The job of this section is to get the front end as far as it can honestly
// go — parse every field, resolve the per-layer attention map, cross-check every dimension against
// the safetensors headers actually on disk — and then refuse with an ITEMISED list of what is not
// implemented. A precise refusal is the deliverable; the alternative (reuse `cfg_kimi` because the
// MLA keys happen to have the same spelling) compiles a blob that loads, runs, and is wrong.

/// Per-layer attention implementation in the K3 text tower.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum K3Attn {
    /// DeepSeek-style MLA (q_a/q_b/kv_a_with_mqa/kv_b), 24 layers.
    Mla,
    /// Kimi Delta Attention: linear attention, recurrent state, short convs on q/k/v,
    /// low-rank forget gate. 69 layers. See `docs/kimi-k3-kda.md` (sibling agent).
    Kda,
}

/// Resolved Kimi-K3 text-tower geometry. Every field is REQUIRED — there are no defaults, because
/// a default here is indistinguishable from a correct value at emit time and only shows up as
/// fluent-but-wrong output.
pub(crate) struct K3Cfg {
    layers: u32,
    hidden: u32,
    heads: u32,
    vocab: u32,
    /// Read by the emit path, which is still blocked (see `k3_gaps`). Kept rather than
    /// dropped: re-deriving it later from a different source is how two eps values appear.
    #[allow(dead_code)]
    eps: f32,
    // --- MLA (the 24 full-attention layers) ---
    kv_lora: u32,
    q_lora: u32,
    qk_nope: u32,
    qk_rope: u32,
    v_head: u32,
    /// `mla_use_nope`: MLA carries NO positional encoding (KDA supplies position). The 64 "rope"
    /// dims still exist in the tensors — they are simply never rotated.
    mla_nope: bool,
    /// `mla_use_output_gate`: `self_attn.g_proj` gates the attention output before `o_proj`.
    mla_out_gate: bool,
    /// `rope_theta` if the config carries one. K3's `text_config` carries NONE (consistent with
    /// `mla_use_nope`), and this stays `None` — it is NOT defaulted. `cfg_glm` used to substitute
    /// GLM's 8e6 here; it is now `Option<f64>` too and refuses via `devgen::require_mla_rope`.
    rope_theta: Option<f64>,
    // --- KDA (the 69 linear-attention layers) ---
    kda_heads: u32,
    kda_head_dim: u32,
    kda_conv: u32,
    kda_full_rank_gate: bool,
    kda_gate_lower_bound: f64,
    // --- MoE ---
    n_exp: u32,
    top_k: u32,
    shared_exp: u32,
    moe_inter: u32,
    /// `routed_expert_hidden_size` — the LATENT width the routed-expert GEMMs actually run at.
    moe_latent: u32,
    latent_norm: bool,
    dense_inter: u32,
    first_k_dense: u32,
    route_scale: f32,
    router_sigmoid: bool,
    renormalize: bool,
    n_group: u32,
    topk_group: u32,
    // --- activation ---
    hidden_act: String,
    situ_beta: f64,
    situ_linear_beta: f64,
    // --- residual blocks ---
    attn_res_block: u32,
    // --- quantization ---
    quant_format: String,
    quant_group: u32,
    quant_bits: u32,
    /// Per-layer attention map, 0-BASED and FIRST-CLASS. `attn.len() == layers`.
    attn: Vec<K3Attn>,
    // --- vision (OUT OF SCOPE, refused by name — never silently dropped) ---
    vision: Option<K3Vision>,
}

/// The MoonViT tower + mm_projector this compiler explicitly does NOT implement. Recorded so the
/// refusal can name what it is refusing; never used to emit anything.
struct K3Vision {
    layers: u32,
    hidden: u32,
    projector: String,
}

impl K3Cfg {
    fn n_mla(&self) -> usize {
        self.attn.iter().filter(|&&k| k == K3Attn::Mla).count()
    }
    fn n_kda(&self) -> usize {
        self.attn.iter().filter(|&&k| k == K3Attn::Kda).count()
    }
}

/// Parse `config.json` into a [`K3Cfg`]. Panics (the emitter convention) with a message naming the
/// exact field on anything missing or unexpected.
///
/// Three traps, all of which have a checkpoint-verified answer below:
///  * the geometry lives under `text_config`, not at the root;
///  * the MoE keys are Kimi spellings (`num_experts`, `num_experts_per_token`,
///    `num_shared_experts`), NOT the DeepSeek spellings `cfg_glm` reads;
///  * `linear_attn_config.{full_attn_layers,kda_layers}` are **1-BASED**
///    (`configuration_kimi_k3.py::is_kda_layer` tests `(layer_idx + 1) in kda_layers`).
fn cfg_kimi_k3(dir: &Path) -> K3Cfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .unwrap();
    k3_cfg_from(&v)
}

/// [`cfg_kimi_k3`] on an already-parsed `config.json`. Split out so the parse rules — above all
/// the 1-based layer lists and the latent-vs-moe_inter choice — are unit-testable without a
/// 618 GB checkpoint on disk.
fn k3_cfg_from(v: &Value) -> K3Cfg {
    // Vision is OUT OF SCOPE and is refused BY NAME (see `kimi_k3_emit`'s SCOPE REFUSAL block and
    // the final panic). It is recorded rather than asserted on here so the text-tower analysis
    // still runs to completion — the report is worth more than an early abort, and the refusal is
    // just as explicit either way. What is NOT acceptable is dropping it silently: a text-only
    // blob for a multimodal checkpoint loads, runs, and is wrong on every image prompt.
    let vision = v
        .get("vision_config")
        .filter(|c| c.is_object())
        .map(|vc| K3Vision {
            layers: vc["vt_num_hidden_layers"].as_u64().unwrap_or(0) as u32,
            hidden: vc["vt_hidden_size"].as_u64().unwrap_or(0) as u32,
            projector: vc["mm_projector_type"].as_str().unwrap_or("?").to_string(),
        });
    let t = &v["text_config"];
    assert!(
        t.is_object(),
        "kimi_k3: config.json has no `text_config` object; the text geometry lives there, not at \
         the root"
    );
    assert_eq!(
        t["model_type"].as_str(),
        Some("kimi_linear"),
        "kimi_k3: text_config.model_type is {:?}, expected \"kimi_linear\"",
        t["model_type"]
    );
    let g = |k: &str| {
        t[k].as_u64()
            .unwrap_or_else(|| panic!("kimi_k3: text_config missing required field {k:?}")) as u32
    };
    let gf = |k: &str| {
        t[k].as_f64()
            .unwrap_or_else(|| panic!("kimi_k3: text_config missing required field {k:?}"))
    };
    let gb = |k: &str| {
        t[k].as_bool()
            .unwrap_or_else(|| panic!("kimi_k3: text_config missing required field {k:?}"))
    };
    let layers = g("num_hidden_layers");
    let attn = k3_attn_map(t, layers);

    let lac = &t["linear_attn_config"];
    let q = &t["quantization_config"];
    let qw = &q["config_groups"]["group_0"]["weights"];
    let act = t["hidden_act"]
        .as_str()
        .expect("kimi_k3: text_config missing required field \"hidden_act\"")
        .to_string();

    K3Cfg {
        layers,
        hidden: g("hidden_size"),
        heads: g("num_attention_heads"),
        vocab: g("vocab_size"),
        eps: gf("rms_norm_eps") as f32,
        kv_lora: g("kv_lora_rank"),
        q_lora: g("q_lora_rank"),
        qk_nope: g("qk_nope_head_dim"),
        qk_rope: g("qk_rope_head_dim"),
        v_head: g("v_head_dim"),
        mla_nope: gb("mla_use_nope"),
        mla_out_gate: gb("mla_use_output_gate"),
        // NOT defaulted. Absent means "this model has no RoPE", which is a fact to act on.
        rope_theta: t["rope_theta"].as_f64(),
        kda_heads: lac["num_heads"]
            .as_u64()
            .expect("kimi_k3: linear_attn_config.num_heads") as u32,
        kda_head_dim: lac["head_dim"]
            .as_u64()
            .expect("kimi_k3: linear_attn_config.head_dim") as u32,
        kda_conv: lac["short_conv_kernel_size"]
            .as_u64()
            .expect("kimi_k3: linear_attn_config.short_conv_kernel_size")
            as u32,
        kda_full_rank_gate: lac["use_full_rank_gate"].as_bool().unwrap_or(false),
        kda_gate_lower_bound: lac["gate_lower_bound"].as_f64().unwrap_or(f64::NEG_INFINITY),
        n_exp: g("num_experts"),
        top_k: g("num_experts_per_token"),
        shared_exp: g("num_shared_experts"),
        moe_inter: g("moe_intermediate_size"),
        moe_latent: g("routed_expert_hidden_size"),
        latent_norm: gb("latent_moe_use_norm"),
        dense_inter: g("intermediate_size"),
        first_k_dense: g("first_k_dense_replace"),
        route_scale: gf("routed_scaling_factor") as f32,
        router_sigmoid: t["moe_router_activation_func"].as_str() == Some("sigmoid"),
        renormalize: gb("moe_renormalize"),
        n_group: g("num_expert_group"),
        topk_group: g("topk_group"),
        situ_beta: t["activation_situ_beta"].as_f64().unwrap_or(f64::NAN),
        situ_linear_beta: t["activation_situ_linear_beta"].as_f64().unwrap_or(f64::NAN),
        hidden_act: act,
        attn_res_block: g("attn_res_block_size"),
        quant_format: q["format"].as_str().unwrap_or("<none>").to_string(),
        quant_group: qw["group_size"].as_u64().unwrap_or(0) as u32,
        quant_bits: qw["num_bits"].as_u64().unwrap_or(0) as u32,
        attn,
        vision,
    }
}

/// Resolve `linear_attn_config.{full_attn_layers,kda_layers}` into a 0-based per-layer map.
///
/// Both lists are read and both are checked: together they must PARTITION `0..layers` — no gap, no
/// overlap, nothing out of range. Deriving one by complement of the other is the §4 bug shape: a
/// truncated list would then silently reclassify layers, and a KDA layer compiled as MLA binds
/// tensor names the checkpoint does not have (`q_a_proj` on a layer that ships `q_proj`).
fn k3_attn_map(t: &Value, layers: u32) -> Vec<K3Attn> {
    let lac = &t["linear_attn_config"];
    assert!(
        lac.is_object(),
        "kimi_k3: text_config has no `linear_attn_config`. Without it there is no way to know \
         which layers are MLA and which are KDA, and guessing the stride mis-binds 69 of 93 layers."
    );
    let list = |k: &str| -> Vec<i64> {
        lac[k]
            .as_array()
            .unwrap_or_else(|| panic!("kimi_k3: linear_attn_config.{k} missing or not an array"))
            .iter()
            .map(|x| {
                x.as_i64()
                    .unwrap_or_else(|| panic!("kimi_k3: linear_attn_config.{k} non-integer entry"))
            })
            .collect()
    };
    let mut out: Vec<Option<K3Attn>> = vec![None; layers as usize];
    for (src, kind) in [
        (list("full_attn_layers"), K3Attn::Mla),
        (list("kda_layers"), K3Attn::Kda),
    ] {
        for one_based in src {
            // 1-BASED -> 0-based, converted exactly once, here.
            let l = one_based - 1;
            assert!(
                (0..layers as i64).contains(&l),
                "kimi_k3: linear_attn_config lists layer {one_based} (1-based; {l} 0-based) but \
                 num_hidden_layers is {layers}"
            );
            assert!(
                out[l as usize].is_none(),
                "kimi_k3: 0-based layer {l} appears in BOTH full_attn_layers and kda_layers"
            );
            out[l as usize] = Some(kind);
        }
    }
    let missing: Vec<usize> = out
        .iter()
        .enumerate()
        .filter(|(_, k)| k.is_none())
        .map(|(i, _)| i)
        .collect();
    assert!(
        missing.is_empty(),
        "kimi_k3: linear_attn_config covers {} of {layers} layers; 0-based layers {:?} are in \
         neither list",
        layers as usize - missing.len(),
        &missing[..missing.len().min(8)]
    );
    out.into_iter().map(|k| k.unwrap()).collect()
}

/// Tensor name -> (dtype, shape) for every `*.safetensors` shard PRESENT in `dir`.
///
/// Deliberately NOT `checkpoint::shard_files`, which panics on an incomplete shard set. K3 is 96
/// shards and a download in progress is the normal case, so this reads whatever has landed and
/// reports the count. **A tensor's absence proves nothing** — every caller below must only ever
/// use this to CONTRADICT the config, never to conclude something does not exist.
fn k3_shard_headers(dir: &Path) -> (std::collections::BTreeMap<String, (String, Vec<i64>)>, u32, u32) {
    use std::io::Read;
    let mut out = std::collections::BTreeMap::new();
    let (mut have, mut total) = (0u32, 0u32);
    let Ok(rd) = std::fs::read_dir(dir) else {
        return (out, 0, 0);
    };
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for ent in rd.flatten() {
        let fname = ent.file_name();
        let Some(f) = fname.to_str() else { continue };
        let base = f
            .strip_suffix(".partial.safetensors")
            .or_else(|| f.strip_suffix(".safetensors"));
        let Some(base) = base else { continue };
        if let Some((_, n)) = base.rsplit_once("-of-") {
            total = total.max(n.parse::<u32>().unwrap_or(0));
        }
        files.push(ent.path());
    }
    files.sort();
    for p in &files {
        let Ok(mut f) = std::fs::File::open(p) else { continue };
        let mut len8 = [0u8; 8];
        if f.read_exact(&mut len8).is_err() {
            continue;
        }
        let hlen = u64::from_le_bytes(len8);
        if hlen == 0 || hlen > 256 * 1024 * 1024 {
            continue;
        }
        let mut hbuf = vec![0u8; hlen as usize];
        if f.read_exact(&mut hbuf).is_err() {
            continue; // still downloading: header not fully written yet
        }
        let Ok(hdr) = serde_json::from_slice::<Value>(&hbuf) else { continue };
        let Some(obj) = hdr.as_object() else { continue };
        have += 1;
        for (k, val) in obj {
            if k == "__metadata__" {
                continue;
            }
            let shape: Vec<i64> = val["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
                .unwrap_or_default();
            out.insert(
                k.clone(),
                (val["dtype"].as_str().unwrap_or("?").to_string(), shape),
            );
        }
    }
    (out, have, total.max(have))
}

/// Cross-check every config dimension against the shard headers on disk. **Tensors win.**
///
/// This exists because of the GLM-5.2 lesson (§ knob-contract): `AutoConfig` reported
/// `qk_rope_head_dim=192` where the tensors said 64 and it cost a day. Returns one line per
/// disagreement; an empty vector means every dim the checkpoint can speak to agrees with the config.
/// Tensors that have not downloaded yet are simply not checked.
fn k3_config_vs_tensors(c: &K3Cfg, h: &std::collections::BTreeMap<String, (String, Vec<i64>)>) -> Vec<String> {
    let mut errs = Vec::new();
    let mut check = |name: String, want: Vec<i64>| {
        if let Some((_, got)) = h.get(&name) {
            if *got != want {
                errs.push(format!("{name}: config implies {want:?}, tensor is {got:?}"));
            }
        }
    };
    let (hd, nh) = (c.hidden as i64, c.heads as i64);
    // Pick the first MLA and first KDA layer that actually exist on disk.
    let mla_l = c.attn.iter().position(|&k| k == K3Attn::Mla);
    let kda_l = c.attn.iter().position(|&k| k == K3Attn::Kda);
    let p = "language_model.model.layers";
    if let Some(l) = mla_l {
        let a = format!("{p}.{l}.self_attn");
        check(format!("{a}.q_a_proj.weight"), vec![c.q_lora as i64, hd]);
        check(
            format!("{a}.q_b_proj.weight"),
            vec![nh * (c.qk_nope + c.qk_rope) as i64, c.q_lora as i64],
        );
        check(
            format!("{a}.kv_a_proj_with_mqa.weight"),
            vec![(c.kv_lora + c.qk_rope) as i64, hd],
        );
        check(
            format!("{a}.kv_b_proj.weight"),
            vec![nh * (c.qk_nope + c.v_head) as i64, c.kv_lora as i64],
        );
        check(format!("{a}.o_proj.weight"), vec![hd, nh * c.v_head as i64]);
        if c.mla_out_gate {
            check(format!("{a}.g_proj.weight"), vec![nh * c.v_head as i64, hd]);
        }
    }
    if let Some(l) = kda_l {
        let a = format!("{p}.{l}.self_attn");
        let w = (c.kda_heads * c.kda_head_dim) as i64;
        for proj in ["q_proj", "k_proj", "v_proj", "g_proj"] {
            check(format!("{a}.{proj}.weight"), vec![w, hd]);
        }
        for cv in ["q_conv1d", "k_conv1d", "v_conv1d"] {
            check(format!("{a}.{cv}.weight"), vec![w, 1, c.kda_conv as i64]);
        }
        check(format!("{a}.b_proj.weight"), vec![c.kda_heads as i64, hd]);
        check(format!("{a}.o_proj.weight"), vec![hd, w]);
    }
    // MoE on the first MoE layer present.
    if let Some(l) = (c.first_k_dense..c.layers).next() {
        let m = format!("{p}.{l}.block_sparse_moe");
        check(format!("{m}.gate.weight"), vec![c.n_exp as i64, hd]);
        check(
            format!("{m}.gate.e_score_correction_bias"),
            vec![c.n_exp as i64],
        );
        check(
            format!("{m}.routed_expert_down_proj.weight"),
            vec![c.moe_latent as i64, hd],
        );
        check(
            format!("{m}.routed_expert_up_proj.weight"),
            vec![hd, c.moe_latent as i64],
        );
        check(format!("{m}.routed_expert_norm.weight"), vec![c.moe_latent as i64]);
        let sh = (c.shared_exp * c.moe_inter) as i64;
        check(format!("{m}.shared_experts.gate_proj.weight"), vec![sh, hd]);
        check(format!("{m}.shared_experts.up_proj.weight"), vec![sh, hd]);
        check(format!("{m}.shared_experts.down_proj.weight"), vec![hd, sh]);
        // mxfp4 routed expert 0. `weight_packed` is [N, K/2] (2 fp4 per byte) and
        // `weight_scale` is [N, K/group] (one E8M0 byte per group) — the SAME layout
        // DevOp::GemvMxfp4 documents (crates/packet/src/dev.rs:622). K is the LATENT
        // width, not moe_inter: that is the load-bearing check in this whole function.
        let (li, lo) = (c.moe_latent as i64, c.moe_inter as i64);
        let grp = c.quant_group.max(1) as i64;
        for w13 in ["w1", "w3"] {
            check(format!("{m}.experts.0.{w13}.weight_packed"), vec![lo, li / 2]);
            check(format!("{m}.experts.0.{w13}.weight_scale"), vec![lo, li / grp]);
        }
        check(format!("{m}.experts.0.w2.weight_packed"), vec![li, lo / 2]);
        check(format!("{m}.experts.0.w2.weight_scale"), vec![li, lo / grp]);
    }
    if c.first_k_dense > 0 {
        let m = format!("{p}.0.mlp");
        check(format!("{m}.gate_proj.weight"), vec![c.dense_inter as i64, hd]);
        check(format!("{m}.up_proj.weight"), vec![c.dense_inter as i64, hd]);
        check(format!("{m}.down_proj.weight"), vec![hd, c.dense_inter as i64]);
    }
    errs
}

/// One K3 capability: what it is, why it blocks, and where the fix goes — plus, once it lands,
/// the evidence that it did.
///
/// # Why closed items STAY in this list
///
/// This report's own preamble says a gap list on its own "invites the next agent to rebuild
/// machinery that exists — the mirror image of §4's *an arm exists and nothing routes to it*".
/// That happened. Between `3f64b3c` and `6603cf7` SIX of these entries were implemented and
/// validated against real-weight oracles on gfx950 — KDA, `situ`, AttnRes, LatentMoE, the MLA
/// output gate and NoPE — and the list went on printing all six as unimplemented blockers. A
/// reader, human or agent, who trusts it is told to write seven opcodes that already dispatch
/// (88, 89, 102, 103, 104, 105, 106, all of them in `GFX950_DISPATCHED`), and is NOT told that
/// the one thing actually standing between this checkpoint and a token is the model-level
/// assembly. The report said "8 unimplemented capabilities"; the true count was 2.
///
/// Deleting a closed entry is the opposite failure: the next agent re-derives whether it was ever
/// needed. So entries are RETIRED, not removed — `done` carries the commit and the measured
/// residual, and only `done.is_none()` entries count as blockers.
struct K3Gap {
    what: &'static str,
    scope: String,
    why: String,
    fix: &'static str,
    /// `Some(evidence)` once this landed and passed a real-weight numeric gate on hardware.
    /// The string is printed verbatim in the CLOSED section and is the reason not to rebuild it.
    done: Option<&'static str>,
}

/// The ranked missing-capability list for Kimi-K3, blocker first.
///
/// Ordering rule: a gap that blocks EVERY layer outranks one that blocks a subset, and a gap whose
/// SEMANTICS are unknown outranks one that is only unwired — you cannot schedule work you cannot
/// specify.
fn k3_gaps(c: &K3Cfg) -> Vec<K3Gap> {
    let mut g = Vec::new();
    g.push(K3Gap {
        what: "KDA (Kimi Delta Attention) linear attention",
        scope: format!("{}/{} layers", c.n_kda(), c.layers),
        why: format!(
            "linear attention with CARRIED RECURRENT STATE ([{}h x {}d x {}d] per layer), short \
             depthwise convs (k={}) on q/k/v, a {} forget gate (f_a_proj/f_b_proj) and A_log/dt_bias. \
             SEMANTICS ARE SPECIFIED by `docs/kimi-k3-kda.md`, and they are now IMPLEMENTED in the \
             dataflow form that doc's §7.2 `KdaScan` proposal was overruled in favour of: the \
             recurrent state is a DECLARED HBM TENSOR with counter-gated tile dependencies, not a \
             monolithic state-carrying kernel.",
            c.kda_heads,
            c.kda_head_dim,
            c.kda_head_dim,
            c.kda_conv,
            if c.kda_full_rank_gate { "full-rank" } else { "low-rank" },
        ),
        fix: "DONE — crates/devgen/src/kda.rs (declare_kda_weights/declare_kda_state/\
              emit_kda_layer) + runtime/amd/op_kda.h. What is NOT done is the CALL: \
              `emit_kda_layer` is reached by nothing outside kda.rs's own unit tests, because \
              there is no model-level K3 emitter to call it (see the full-model-emit gap).",
        done: Some(
            "3f64b3c. FOUR opcodes, not one, all in GFX950_DISPATCHED and all reached by the \
             emitter: KdaConv=88, KdaGate=89, KdaStateStep=102, KdaGatedNorm=103. Real layer-0 \
             weights, 16 packets on a leased gfx950, T=1 AND T=4: conv+SiLU ~2.4e-03, gate \
             2.0e-04, beta 2.5e-04, STATE (f32, V-first) 1.4e-04, block out 8.1e-04. The state \
             row is the load-bearing one — against the TRANSPOSED reading of the same reference \
             it is 1.408e+00, 10100x larger, and both readings have identical norms, so no \
             magnitude check would have caught a transpose. REGISTER COST ZERO: decode stayed \
             248 VGPR / occ 2 / spill 0 (the predicted 32 VGPR/lane was the cost of one workgroup \
             per head, not of KDA — one WAVE owns one column, so the state is 2 f32/lane). \
             `Mamba2Scan(90)`, the dead-opcode cautionary tale this was written against, is still \
             dead and still has no AMD arm.",
        ),
    });
    if c.hidden_act != "silu" && c.hidden_act != "gelu_pytorch_tanh" && c.hidden_act != "gelu_tanh" {
        g.push(K3Gap {
            what: "`situ` activation",
            scope: "every FFN: dense layer 0, 2 shared experts, 896 routed experts, all 93 layers"
                .into(),
            why: format!(
                "hidden_act = {:?} (beta {}, linear_beta {}). CLOSED FOR DECODE, OPEN FOR PREFILL. \
                 The original diagnosis — \"plow's activation operand is ONE BIT\" — was right \
                 about the constraint and wrong about the fix: situ transforms the UP branch as \
                 well as the gate, `beta*tanh(g/beta)*sigmoid(g) * lbeta*tanh(u/lbeta)`, so the \
                 EXPRESSION SHAPE changes from `act(g)*u` to `A(g)*B(u)` and a third `act` code \
                 alone would have left `up` un-clipped — a small error at |u| < {} that grows with \
                 the tail, i.e. plausible output and the wrong model. WHAT IS STILL OPEN: the two \
                 GROUPED PREFILL GLU epilogues (runtime/amd/op_moe.h:1285, :1584) were not \
                 converted to the pair form, so `moe_act` returns NaN for code 2 rather than \
                 silently computing gelu_tanh(g)*u. Prefill for a K3 MoE layer is therefore \
                 REFUSED-BY-NaN, not supported.",
                c.hidden_act, c.situ_beta, c.situ_linear_beta, c.situ_linear_beta
            ),
            fix: "DONE for decode. OPEN: runtime/amd/op_moe.h:1285 and :1584 (convert the grouped \
                  prefill epilogues to the pair form `moe_glu(g, u, act, beta, lbeta)`), which is \
                  a precondition for any K3 PREFILL program.",
            done: Some(
                "50d9ed5 (dense/shared) + the routed-expert half. NOT a third `act` code: \
                 PLOW_DOP_SITU_GLU = 105 for the dense and shared FFNs, and PLOW_MOE_ACT_SITU = 2 \
                 with a PAIR-form `moe_glu` inside the 896 routed experts. The two betas ride in \
                 `f0`/`f1`, which were FREE on every GLU-family op, so no `i` slot moved and every \
                 pre-K3 packet is byte-identical. Measured on real weights: dense situ act \
                 3.177e-03 (rung 1), expert situ GLU 3.553e-03 (rung 2), 1.815e-03 (rung 3). \
                 Register cost ZERO across all four objects. `moe_act` returning NaN for code 2 is \
                 deliberate — this interpreter's dispatch `default:` is a silent NOP and there is \
                 no device trap, so NaN is the loudest primitive available for the two epilogues \
                 that were not converted.",
            ),
        });
    }
    g.push(K3Gap {
        what: "residual-ATTENTION blocks (`attn_res_block_size`) — not a residual add at all",
        scope: format!(
            "all {} layers, TWICE each (post-attention and post-MLP), plus once at the model output",
            c.layers
        ),
        why: format!(
            "`_apply_attn_res` (modeling_kimi_linear.py:1075) replaces `x = x + f(x)` with a SOFTMAX \
             over up to {} candidates: the running prefix sum plus one snapshot per completed \
             {}-layer block. It RMS-normalises each candidate, scores it against \
             `norm.weight * proj.weight` and takes a probability-weighted mixture. So every layer \
             ships `self_attention_res_norm` [{h}] + `self_attention_res_proj` [1,{h}] and \
             `mlp_res_norm` [{h}] + `mlp_res_proj` [1,{h}], the model ships an `output_attn_res_*` \
             pair, and a new block snapshot is pushed when `layer_idx % {} == 0`. \
             `score_weight = norm.weight * proj.weight` is a constant [{h}] vector, so the two \
             tensors FOLD INTO ONE at weight-prep time and neither factor reaches the device.\n\
             THE DETECTABILITY FINDING, which is the reason this entry is worth reading even \
             though the op is done: at a SNAPSHOT layer (l % {} == 0, i.e. {} of {}) the block \
             output is `attn + ffn` and a plain-residual wiring differs by 1.0. At EVERY OTHER \
             layer `prefix = prefix_in + attn`, so the block output is `prefix_in + attn + ffn` — \
             EXACTLY what a plain residual produces, measured at 3.0e-03 on real layer-1 weights \
             against 8.1e-01 at the AttnRes outputs themselves. **A block-output-only gate does \
             not see AttnRes at {} of {} layers.** Any future K3 gate must score the two AttnRes \
             outputs, not the block output.",
            c.layers / c.attn_res_block.max(1) + 1,
            c.attn_res_block,
            c.attn_res_block,
            c.attn_res_block,
            c.layers.div_ceil(c.attn_res_block.max(1)),
            c.layers,
            c.layers - c.layers.div_ceil(c.attn_res_block.max(1)),
            c.layers,
            h = c.hidden
        ),
        fix: "DONE — runtime/amd/op_k3.h (op 104) + crates/devgen/src/k3.rs emit_attn_res. What is \
              NOT done is the model-level plumbing: the `block_residual` ring (<=8 live H-wide \
              snapshots) as CARRIED STATE across the layer loop, which belongs to the full-model \
              emit gap, not here.",
        done: Some(
            "50d9ed5. PLOW_DOP_ATTN_RES = 104, ONE packet (three packets x 186/token would be \
             3.3 ms of pure protocol). Real weights: AttnRes(attn) 0.000e+00 and AttnRes(mlp) \
             1.109e-03 at rung 2; 1.000e-03 at rung 1; EXACTLY 0 at rung 3. Controls at the \
             sub-layer inputs, which is where they have to be: h_a vs a plain residual 8.04e-01, \
             h2 vs plain 7.70e-01. Three things the gate pinned that a code read would not: \
             `score_weight` folds at prep time; the mix is over the RAW rows v, not the normalised \
             k (the natural misreading — right shape, wrong per-row magnitude); and \
             `variance = mean(x^2)` is RMSNorm's, not mean-centred. KNOWN COST, UNMEASURED: \
             AttnRes is ONE WORKGROUP PER TOKEN (blocks = 1 of 256 at T=1) because both reductions \
             span the full H-wide row and the softmax couples the rows. 186 invocations/token on \
             1 CU. The batched form is the fix and it is not written.",
        ),
    });
    if c.moe_latent > 0 && c.moe_latent != c.hidden {
        g.push(K3Gap {
            what: "LATENT MoE (routed experts do not read the hidden state)",
            scope: format!("{} MoE layers", c.layers - c.first_k_dense),
            why: format!(
                "resolved order, from modeling_kimi_linear.py:815-837 — the ROUTER scores the \
                 HIDDEN state ({}), then `routed_expert_down_proj` projects hidden -> latent {}, \
                 every routed expert runs at K={}, the gated expert sum is RMS-normed by \
                 `routed_expert_norm` [{}] (latent_moe_use_norm={}) and only then does \
                 `routed_expert_up_proj` [{},{}] return to hidden. The shared experts read the \
                 ORIGINAL hidden and are added AFTER the up-projection. THE KERNELS NEEDED \
                 NOTHING: H is a runtime operand and the scale-row arithmetic needs only 128- and \
                 32-divisibility, which {}/128 and {}/32 satisfy exactly. What is STILL OPEN is \
                 the DECLARE: `declare_glm` sizes every expert weight with K = hidden, wrong here \
                 by {}/{} = 2x, and the combine accumulator must run at latent rather than hidden \
                 width (the four decode `MoeCombine` `d.i[0] = h` sites and the two prefill \
                 combine sites, which perf-data/kimi-k3-kernel-gap.md §5e omits from its list).",
                c.hidden, c.moe_latent, c.moe_latent, c.moe_latent, c.latent_norm,
                c.hidden, c.moe_latent, c.moe_latent, c.moe_latent, c.hidden, c.moe_latent,
            ),
            fix: "crates/devgen/src/mla.rs declare_glm (expert weight/scale sizing keyed on the \
                  latent width) and the MoE emit (down/norm/up around the expert loop). The GRAPH \
                  is proven (see below); this is the width plumbing, and it belongs with the \
                  full-model emit.",
            done: Some(
                "50d9ed5, GRAPH ONLY — the kernels were already sufficient. Validated on real \
                 layer-1 weights, top-16 of 896, on real mxfp4 bytes: latent down 2.158e-03, \
                 expert situ GLU 3.553e-03, MoeCombine(no residual) 3.378e-03, latent RMSNorm \
                 3.673e-03, latent up 4.123e-03, shared expert 3.993e-03. One kernel line \
                 changed: `d_moe_combine`'s `residual` is now OPTIONAL (op_moe.h:819 decode, \
                 :1689 prefill) — it was an unconditional null deref, and a latent-width combine \
                 has no hidden-width residual to add. TWO OPERAND FACTS the gate pins that a code \
                 read would not: the shared expert reads the PRE-DOWN hidden `h3` (feeding it the \
                 latent fails loudly on width; feeding it `h2` would fail QUIETLY), and the gate \
                 weight multiplies inside the DOWN kernel, not in the combine.",
            ),
        });
    }
    if c.top_k > crate::MOE_MAX_TOPK {
        g.push(K3Gap {
            what: "top-k beyond PLOW_MOE_MAX_TOPK",
            scope: format!("top-{} routing on {} MoE layers", c.top_k, c.layers - c.first_k_dense),
            why: format!(
                "`#define PLOW_MOE_MAX_TOPK` (runtime/amd/op_moe.h) sizes both routers\' winner/gate \
                 arrays and the `wl` LDS carve the rank pass writes into. This checkpoint routes \
                 top-{}, past the current bound of {}. The emit refuses (devgen::require_moe_topk) \
                 rather than letting the kernel truncate: slots above the bound are never written, \
                 every expert body loops to the packet\'s unbounded top_k operand and reads them as \
                 uninitialised scratch, and the renormalisation denominator covers only the kept \
                 gates. Raise both constants together — a drift test enforces the pair.",
                c.top_k, crate::MOE_MAX_TOPK
            ),
            fix: "runtime/amd/op_moe.h:57 (raise the bound, re-check the LDS carve at :299), and \
                  turn the two silent clamps at :135/:314 into a hard failure so the next model \
                  past the bound is loud instead of wrong.",
            done: None,
        });
    }
    if c.mla_out_gate {
        g.push(K3Gap {
            what: "MLA output gate (`mla_use_output_gate`)",
            scope: format!("{} MLA layers", c.n_mla()),
            why: format!(
                "`self_attn.g_proj.weight` [{}, {}] = [heads*v_head_dim, hidden] gates the \
                 attention output before o_proj; plow's MLA chain was flash -> OUvFold -> o_proj \
                 with nothing in between. Now expressed as its own opcode rather than folded into \
                 `MlaMergeFold`'s epilogue: the fold is a REDUCTION over KV splits and the gate is \
                 a per-element multiply on its RESULT, so folding it in would have applied the \
                 sigmoid once per split.",
                c.heads * c.v_head, c.hidden
            ),
            fix: "DONE — runtime/amd/op_k3.h (op 106) + crates/devgen/src/k3.rs \
                  emit_mla_out_gate. The CALL site is the model-level MLA emit, which does not \
                  exist yet (see the full-model emit gap).",
            done: Some(
                "6603cf7 (rung 3). PLOW_DOP_MLA_OUT_GATE = 106. Real layer-3 weights: MLA OUTPUT \
                 GATE 3.468e-05, block output 7.324e-04, with the control `gated vs ungated \
                 attention` at 5.17e-01 — i.e. the gate is not a rounding-level effect and a \
                 missing one would not have hidden in the block output.",
            ),
        });
    }
    if c.mla_nope || c.rope_theta.is_none() {
        g.push(K3Gap {
            what: "MLA with NO positional encoding (`mla_use_nope`)",
            scope: format!("{} MLA layers", c.n_mla()),
            why: format!(
                "mla_use_nope={} and text_config carries NO `rope_theta` at all — KDA supplies \
                 position, so the {dr} decoupled dims exist in q_b/kv_a but are never rotated \
                 (modeling_kimi_linear.py: `self.rotary_emb = None`, `assert self.use_nope`; \
                 q_rot/k_rot are split off and concatenated back UNCHANGED, i.e. they are extra \
                 CONTENT dims of the {}-wide key).\n\
                 THE SILENT DEFAULT IS CLOSED: `cfg_glm` no longer reads `rope_theta` as \
                 `.unwrap_or(8_000_000.0)`; it is `Option<f64>`, both config spellings are tried, \
                 and `devgen::require_mla_rope` REFUSES a NoPE checkpoint at parse time instead of \
                 substituting GLM's theta. (That default was also load-bearing for GLM-5.2 itself, \
                 whose config has no top-level `rope_theta` — the key moved to \
                 `rope_parameters.rope_theta` in transformers 5.x.)\n\
                 WHAT REMAINS IS THE EMIT, and it is not a deletion. `emit_glm_mla` has two \
                 HeadNormRope ops; the k-side one (mla.rs, `d.t[0] = n.krot[slot]`) is ALSO THE \
                 ONLY WRITER OF THE `kv.{{l}}.krot` CACHE ROW. Drop it and the rope half of every \
                 cached key is never written while FlashMlaDecode keeps reading it at i[5] — \
                 uninitialised memory that grows with context and never faults. WORSE, that op is \
                 also how the RUNTIME FINDS the per-layer KV-row writer: `plowrt::exec::amd`'s \
                 kv_row_writer classifier and runtime/tests/glm52_decode.c:419 both SCAN the \
                 instruction stream for a HeadNormRope whose t[0] is a `kv.*.krot` tensor and \
                 patch its out_row to the current position every step. Delete it and the scan \
                 simply finds fewer layers — no error, no count check. So a NoPE MLA needs the \
                 WRITE KEPT and the ROTATION removed, not the op removed; \
                 perf-data/kimi-k3-kernel-gap.md 8c and item #2 (\"a removal, effort XS\") are \
                 wrong on this point. The KV layout does NOT change: krot stays [ctx][{dr}] and \
                 holds the raw, unrotated k_rot.",
                c.mla_nope,
                c.qk_nope + c.qk_rope,
                dr = c.qk_rope,
            ),
            fix: "TECHNIQUE PROVEN, RUST EMIT NOT WRITTEN. `crates/devgen/src/k3.rs \
                  k3_nope_rope_pair` builds the identity table and a unit test checks it BITWISE; \
                  what is missing is `emit_glm_mla` / `emit_glm_mla_prefill` selecting it off \
                  `rope_theta == None` and keeping both HeadNormRope emits. NOTE FOR ANYONE TOLD \
                  \"just open require_mla_rope for K3\": that gate is on the `cfg_glm` path and \
                  the K3 config parse (`k3_cfg_from`) NEVER CALLS IT — it is not what blocks the \
                  93-layer emit, and opening it changes nothing. The blocker is the model-level \
                  emitter below.",
            done: Some(
                "6603cf7, TECHNIQUE ONLY — proven in the rung-3 C harness, not in devgen. NoPE is \
                 done with an IDENTITY cos=1/sin=0 table (both exact in bf16, so HeadNormRope is a \
                 bit-exact row copy), keeping BOTH HeadNormRope emits so the krot cache write and \
                 the runtime's kv-row-writer scan both survive. Real layer-3 weights: absorbed \
                 q_nope 1.724e-07, FLASH_MLA+MERGE_FOLD 1.069e-03, and both KV writes EXACTLY 0. \
                 THE CONTROL IS THE PART WORTH INHERITING: the first version rotated q and every \
                 cached k at the SAME position and measured 1.2e-07, i.e. 'RoPE is harmless here' \
                 — a common rotation is ORTHOGONAL and preserves every dot product exactly. RoPE \
                 is RELATIVE: key t must be rotated by t, query by qpos. Corrected control \
                 2.459e-01. A control that proves nothing is worse than no control.",
            ),
        });
    }
    g.push(K3Gap {
        what: "full-model emit for a hybrid MLA arch — THE ONE REMAINING BLOCKER",
        scope: "the whole blob".into(),
        why: "EVERY OP K3 NEEDS NOW EXISTS AND PASSES A REAL-WEIGHT GATE (see CLOSED, above). What \
              does not exist is anything that CALLS them together. Concretely, and this is the \
              honest state of the tree rather than a plan:\n\
              * `crates/devgen/src/k3.rs` and `crates/devgen/src/kda.rs` are reached by NOTHING \
                outside their own `#[cfg(test)]` modules. `emit_kda_layer` emits a whole KDA \
                mixer; `emit_attn_res`/`emit_situ_glu`/`emit_mla_out_gate`/`emit_k3_block_out` \
                emit one packet each. No function composes them into even ONE complete layer, and \
                there is no loop over layers anywhere.\n\
              * the three rung gates build their instruction streams BY HAND IN C \
                (`runtime/tests/k3_{block,moe_block,mla_block}_gfx950_test.c`, a private \
                `emitop()` each), against fixtures pinned to a single `K3_LAYER`. So the C \
                harnesses and the devgen modules are two independent transcriptions of the same \
                graph with nothing tying them together — a drift hazard that only a shared emit \
                closes.\n\
              * `glm_emit_full` cannot be reused as-is: it assumes a UNIFORM MLA layer and a PLAIN \
                RESIDUAL ADD, and K3 breaks both. It needs the per-layer attention map to select \
                layer L's ops AND its carried state (a KDA recurrent state + 3 conv states on 69 \
                layers, a ckv/krot KV ring on 24), plus the `block_residual` snapshot ring.\n\
              * there is no embed / final-norm / lm_head / argmax tail for K3 in ANY form: all \
                three C gates are block-only, `act.x` in and a residual out.\n\
              * and there is a SECOND, independent refusal outside devgen: \
                `crates/plowc/src/hf_config.rs` `build_full_model_plan` asserts \
                `arch != HfArch::KimiK3`, locked by `test_kimi_k3_has_no_full_model_plan`. Both \
                have to open.\n\
              THERE IS ALSO NO TRUNCATION KNOB. GLM's `GLM_NLAYERS` (mla.rs, `glm_emit_full`) is \
              what makes a cheap iteration loop possible — a truncated model loads in seconds \
              instead of the 4-minute 183 GiB/rank full load. K3 has no equivalent and cannot \
              have one until there is a loop to truncate. When it is written: layers 0..3 is the \
              minimum honest span, because 0/1/2 are KDA and 3 is the first MLA, so anything \
              shorter is not testing the hybrid at all."
            .into(),
        fix: "crates/devgen/src/lib.rs (dispatch: stop routing kimi_k3 unconditionally into \
              `kimi_k3_emit`), crates/plowc/src/hf_config.rs (the second refusal), and a new \
              `k3_main`/`k3_emit_full` in crates/devgen/src/mla.rs or a k3 module: a declare keyed \
              on the per-layer attention map, the layer loop composing kda.rs + k3.rs + the MLA \
              emit, a K3_NLAYERS truncation knob, and the embed/tail.",
        done: None,
    });
    g.push(K3Gap {
        what: "MoE sub-namespace + expert-name template",
        scope: "every expert tensor on 92 MoE layers".into(),
        why: "PARTLY CLOSED. The wrapper prefix itself is no longer a gap in either half:\n\
              * emit — `GlmCfg::prefix` is now cfg data (mirroring `Cfg::prefix` on the Gemma \
                path) and `declare_glm` builds every name from it, so a tower spelled \
                `language_model.model.layers.{L}.…` needs a field, not a patch;\n\
              * bind — the loaders no longer allowlist weight prefixes. `packet::names::\
                is_checkpoint_weight` classifies by EXCLUSION of the compiler's own namespaces \
                (`act.`/`in.`/`kv.`/`moe.` + the host-filled pointer tables), so an unknown name \
                is demanded of the checkpoint and a missing one is `MISSING WEIGHT: <name>`. \
                Under the old `starts_with(\"model.\")` all 497 052 of this checkpoint's \
                language-tower tensors — none of which starts with `model.` — would have been \
                allocated, never uploaded, zero-filled and decoded from.\n\
              What is left is BELOW the prefix: the MoE block is `block_sparse_moe.…` with \
              experts `experts.{e}.w1|w2|w3` (Mixtral naming: w1=gate, w2=down, w3=up) and \
              mxfp4 `weight_packed`/`weight_scale`, where `declare_glm` and \
              `bind_packed_experts` both spell `mlp.experts.{e}.{gate,up,down}_proj.weight`.\n\
              A THIRD SITE, not previously recorded here and the dangerous one: the TP shard \
              classifier `crates/plowrt/src/asset/shard.rs` keys on projection SUBSTRINGS — `COL` \
              holds \"gate_proj.weight\"/\"up_proj.weight\", `ROW` holds \
              \"o_proj.weight\"/\"down_proj.weight\", matched with `name.contains(s)`. Mixtral \
              `w1`/`w2`/`w3` match NEITHER list, so every routed-expert tensor would fall through \
              to `Shard::Replicated` — no error, no missing weight, just every rank holding the \
              whole expert and column-parallel work done redundantly against a row-parallel \
              layout. It is a substring default, so it fails by SILENCE in exactly the way this \
              report exists to prevent.\n\
              A FOURTH, for the same template: there is NO mxfp4 expert-bind path in the AMD \
              runtime at all. `bind_packed_experts` binds `.weight` + `.weight_scale_inv` \
              (block-fp8) only; K3 ships `weight_packed` + `weight_scale`. The decode KERNEL arm \
              exists and is validated — what is missing is the host-side bind."
            .into(),
        fix: "crates/devgen/src/mla.rs declare_glm (expert-name template as cfg data, next to \
              `GlmCfg::prefix`), crates/plowrt/src/exec/amd.rs bind_packed_experts (same \
              template, read from the packet rather than hardcoded, plus a weight_packed/\
              weight_scale arm), and crates/plowrt/src/asset/shard.rs (an axis tag per \
              projection, or the same template — NOT another substring literal).",
        done: None,
    });
    g
}

/// `--emit devblob` on a Kimi-K3 checkpoint. Parses and validates everything the front end can,
/// prints the state of the checkpoint and the itemised missing-capability report, then aborts.
///
/// This function never returns: there is no correct blob to emit. It exists so the failure is a
/// specific, accurate statement of what is not implemented rather than an `Option::unwrap` panic
/// three crates deep (which is what `kimi_k3` produced before: `crates/devgen/src/config.rs:93`,
/// because the `text_config` probe routed it into the Gemma-4 parser).
/// Emit the full K3 decode blob. `K3_FULL=1` selects this; the default stays
/// the capability report in [`kimi_k3_emit`], which is still the honest answer
/// for anyone who has not read what is missing.
///
/// `K3_NLAYERS` truncates, and it is what makes iteration affordable — the same
/// role `GLM_NLAYERS` plays. **0..3 is the minimum honest span**: 0/1/2 are KDA
/// and 3 is the first MLA, so anything shorter does not exercise the hybrid at
/// all. Truncation shrinks the tensor table, so a short model loads in seconds
/// instead of paying the full-checkpoint load.
///
/// What this does NOT yet do, and what will therefore fail at LOAD rather than
/// here: the host-side mxfp4 expert bind (`bind_packed_experts` knows
/// `.weight` + `.weight_scale_inv`, K3 ships `weight_packed` + `weight_scale`)
/// and the Mixtral `w1/w2/w3` expert-name template. Both fail loudly with a
/// missing weight, which is the right failure — but they are why this is gated
/// rather than default.
#[allow(clippy::too_many_arguments)]
pub(crate) fn k3_emit_full(
    dir: &Path,
    ctx: u32,
    out: &str,
    n_cu: u32,
    tp: u32,
    rope_gen: bool,
    verify: Option<&crate::VerifyHook>,
    l2_layout: Option<packet::devbuild::L2Layout>,
) {
    let c = cfg_kimi_k3(dir);
    let pf = k3_prefill_buckets(ctx);
    let mut m = k3_build_model(dir, ctx, n_cu, tp, &pf, l2_layout);
    k3_ablate_bodies(&mut m);
    // Leave the position tables as GENERATED tensors unless asked to bake them.
    // The runtime materialises them at load (`exec/amd.rs` `g.generate()`), and
    // `DevBlob::parse` refuses a gen tensor it cannot produce, so nothing is
    // taken on trust. Baking is `--no-rope-gen`.
    //
    // It matters more here than anywhere else in the tree: K3 is NoPE, so its
    // table is the IDENTITY — cos = 1, sin = 0. Baking writes
    // `ctx * qk_rope * 2B * 2` bytes of ones and zeros into the blob, which at
    // ctx 131072 is 33.5 MiB of constants, and it is the ONLY thing in a K3
    // blob that grows with context.
    if !rope_gen {
        m.bake_gen();
    }
    let layers = k3_emit_layers(&c);
    // COVERAGE GATE — and the reason it is here is a bug it would have caught on day one.
    //
    // `checkpoint::validate_coverage` is bidirectional and fatal, and its own header names
    // Kimi-K3 as the case it was written for. It was reachable only from the dense path
    // (`lib.rs`); THIS function went straight from `apply_verify_gate` to `fs::write`. So when
    // the model-level `_apply_output_attn_res` was never emitted, its two weights
    // (`output_attn_res_{norm,proj}`) sat in the checkpoint claimed by nothing, every per-layer
    // golden test stayed green, all 8 ranks agreed, and the model decoded one constant token
    // forever. A missing OP is invisible; the weight it fails to read is not.
    //
    // The gate is run on the DECLARED names before the blob is written, so a program that would
    // drop a weight never reaches disk to be benchmarked by someone else.
    //
    // Truncation is passed through as `block`: under `K3_NLAYERS` the other layers' weights are
    // legitimately uncovered, and without this every truncated emit would read as "an
    // architecture this emitter does not implement".
    let truncated = (layers.len() as u32) < c.layers;
    match crate::checkpoint::validate_coverage(
        dir,
        K3_PREFIX,
        &m.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        truncated.then(|| 0..layers.len()),
        K3_INDIRECT,
        K3_PAIRED,
        K3_SYNTHESIZED,
    ) {
        Ok(()) => {}
        Err(e) if std::env::var("PLOW_SKIP_COVERAGE").ok().as_deref() == Some("1") => {
            eprintln!("*** PLOW_SKIP_COVERAGE=1 — EMITTING A MODEL KNOWN TO BE WRONG ***\n{e}");
        }
        Err(e) => {
            eprintln!("kimi_k3: {e}");
            std::process::exit(1);
        }
    }
    // Gate BEFORE the bytes land: a rejected program must never exist on disk.
    crate::apply_verify_gate(&m, verify);
    std::fs::write(out, m.to_blob()).expect("write k3 devblob");
    eprintln!(
        "kimi_k3: emitted {} layers ({} KDA, {} MLA), tp={tp}, {} tensors, {} decode \
         instructions -> {out}\n  prefill buckets {pf:?} ({} programs incl. decode), ctx={ctx}",
        layers.len(),
        layers.iter().filter(|&&l| matches!(c.attn[l as usize], K3Attn::Kda)).count(),
        layers.iter().filter(|&&l| !matches!(c.attn[l as usize], K3Attn::Kda)).count(),
        m.tensors.len(),
        m.progs.last().map(|p| p.insts.len()).unwrap_or(0),
        m.progs.len(),
    );
}

/// The 0-based layer span a K3 emit covers. `K3_NLAYERS` truncates it, and BOTH program kinds are
/// built from this one list, so a truncation cannot leave prefill and decode at different depths.
/// Kimi-K3's coverage waivers — the only checkpoint tensors a correct K3 blob leaves undeclared.
///
/// Each names a mechanism that DOES read the bytes; see `validate_coverage`'s `indirect`
/// contract. Adding an entry here is how a missing op gets hidden, so an addition needs the
/// mechanism, not a plausible story about the weight being unused.
/// Kimi-K3's checkpoint prefix. K3 nests its text tower under a multimodal wrapper, so of the
/// checkpoint's 497,052 language-tower tensors ZERO start with `model.`.
///
/// Shared by the emitter and the coverage gate deliberately: `validate_coverage` filters BOTH
/// sides by the prefix, so a prefix that matches nothing compares two empty sets and passes. A
/// gate keyed on a second copy of this string would silently stop gating the moment the two
/// drifted — which is the same class of failure the gate exists to catch.
pub(crate) const K3_PREFIX: &str = "language_model.model.";

/// Kimi-K3's coverage waivers — the only checkpoint tensors a correct K3 blob leaves undeclared.
/// Names a K3 blob declares that the CHECKPOINT does not ship, because they are produced before
/// the bind. The mirror of [`K3_INDIRECT`]; same rule — each entry names a producer.
const K3_SYNTHESIZED: &[&str] = &[
    // `fold_res_score` (plowrt exec/amd.rs:1912) computes this [H] f32 at load from the
    // checkpoint's `_res_norm`/`_res_proj` pair. It is the twin of the `_res_{norm,proj}`
    // waivers below: one mechanism, one weight consumed, one weight produced.
    "_res_score.weight",
    // Supplied by scripts/kimi_k3_prep.py's `--derived` sidecar, which `shard_files` above
    // deliberately cannot see: it accepts only `model.safetensors` and `model-{i}-of-{n}`, and
    // the sidecar is named `model-idx-derived-*.safetensors` precisely so the COMPILER ignores it
    // while the RUNTIME (which globs every `*.safetensors`) picks it up. So devgen cannot check
    // these here even though they are present on disk at serve time.
    "derived.",
];

/// Conditional waivers — covered only if the consumer is emitted. See `validate_coverage`'s
/// `paired` contract for why these are NOT flat entries in [`K3_INDIRECT`].
///
/// `fold_res_score` turns each `{stem}_res_norm.weight` + `{stem}_res_proj.weight` pair into one
/// `{stem}_res_score.weight`. Three stems exist: `self_attention`, `mlp`, and the model-level
/// `output_attn` — and it was the third whose op went missing. Keying on the produced name means
/// dropping that op un-covers its two weights and fails the emit, which is the whole point.
const K3_PAIRED: &[(&str, &str)] = &[
    ("_res_norm.weight", "_res_score.weight"),
    ("_res_proj.weight", "_res_score.weight"),
];

const K3_INDIRECT: &[&str] = &[
    ".experts.",           // bind_packed_experts, by name pattern (494,592 tensors)
    "q_b_proj.weight",     // absorbed host-side into derived.{q_absorb,q_rope}
    "kv_b_proj.weight",    // absorbed host-side into derived.{q_absorb,v_absorb}
];

fn k3_emit_layers(c: &K3Cfg) -> Vec<u32> {
    let nl = std::env::var("K3_NLAYERS")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(c.layers)
        .min(c.layers);
    (0..nl).collect()
}

/// [`k3_emit_full`]'s model, with the prefill ladder as a PARAMETER rather than an environment
/// read, and without the file I/O.
///
/// The seam is a parameter for the reason `emit_kda_mixer_ex`'s is: a test that flips an env var
/// races every other test in the binary. It also lets the decode-identity gate build the SAME
/// model twice, once with an empty ladder and once with a full one, and compare the decode
/// program byte for byte — which is the only way to state "the ladder did not move decode" as a
/// fact rather than a hope.
/// BODY ABLATION, and it is a MEASUREMENT INSTRUMENT that produces WRONG TOKENS.
///
/// `PLOW_K3_ABLATE=<opcode>[,<opcode>...]` rewrites the named ops to `Nop` **after** the graph is
/// built, so `stream`, `waits`, `succs`, the counter count and every packet's dispatch width are
/// byte-for-byte what they were — the ONLY thing that goes away is the op's body. Subtracting the
/// ablated run from the full one is therefore that op family's BODY time, the way
/// `PLOW_CHAIN_BYPASS` isolates its CHAIN DEPTH. The two answer different questions and this tree
/// had only the second one on AMD: `PLOW_NV_ABLATE_LO/HI` is NVIDIA-only
/// (`scripts/tune_decode_sweep.sh:399`), which is why K3's per-layer cost had never been attributed.
///
/// Consumers read stale buffers, so tokens are garbage. That is intended and is the same standing
/// as `PLOW_CHAIN_BYPASS`: wrong numerics are a valid instrument for scheduling and for cost.
fn k3_ablate_bodies(m: &mut Model) {
    let Ok(spec) = std::env::var("PLOW_K3_ABLATE") else { return };
    let want: Vec<u16> = spec.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    if want.is_empty() {
        return;
    }
    let mut hit = 0usize;
    for p in m.progs.iter_mut() {
        for i in p.insts.iter_mut() {
            if want.contains(&i.op) {
                i.op = packet::dev::DevOp::Nop as u16;
                hit += 1;
            }
        }
    }
    eprintln!("  PLOW_K3_ABLATE: {hit} instruction(s) rewritten to Nop — TOKENS ARE GARBAGE, this is a cost instrument");
}

/// FlashMLA's decode `nsplit` for K3, and the reason it is not simply `glm_nsplit`.
///
/// The work-item count is `(nh_l / gf) * nsplit`, which at TP8 is `3 * nsplit` — so `nsplit = 4`
/// dispatches **12** items and leaves 244 of 256 workgroups empty on all 24 MLA layers. Splitting
/// the KV range further is the only way to fill the machine on this op.
///
/// It is NOT a free widening: `MlaMergeFold` reduces over the `nsplit` partials, so the merge grows
/// as the flash shrinks and the net is a U-shape whose minimum `mla.rs`'s own `NS_CEIL_MEASURED`
/// note records as UNSWEPT at TP8. `PLOW_K3_NS` is therefore the sweep handle and the default is
/// the measured winner; do not change the default without re-running the sweep.
fn k3_nsplit(ctx: u32) -> u32 {
    if let Some(v) = std::env::var("PLOW_K3_NS").ok().and_then(|v| v.parse::<u32>().ok()) {
        return v.max(1);
    }
    let _ = ctx;
    // SWEPT on the real TP8 asset at ctx 8000, 32 steps, fp8 KV, everything else fixed:
    //   ns  4 -> 42.226 ms/token   (flash dispatched on 12 of 256 workgroups)
    //   ns 16 -> 39.700 ms/token   (48)
    //   ns 32 -> 39.582 ms/token   (96)
    // 16 and 32 are tied inside run-to-run spread, which is the U-shape this doc predicts: the
    // flash keeps getting more parallel while `MlaMergeFold` reduces over more partials. 16 is the
    // knee — same time as 32 for half the `o_part`/`ml_part` scratch.
    16
}

fn k3_build_model(
    dir: &Path,
    ctx: u32,
    n_cu: u32,
    tp: u32,
    pf: &[u32],
    l2_layout: Option<packet::devbuild::L2Layout>,
) -> Model {
    use crate::k3::{K3MlaCfg, K3ModelCfg, K3MoeCfg};
    let c = cfg_kimi_k3(dir);
    let layers = k3_emit_layers(&c);

    let mcfg = K3ModelCfg {
        block: crate::k3::K3BlockCfg {
            hidden: c.hidden,
            eps: c.eps,
            attn_res_block_size: c.attn_res_block,
            situ_beta: c.situ_beta as f32,
            situ_linear_beta: c.situ_linear_beta as f32,
        },
        kda: crate::kda::KdaCfg {
            hidden: c.hidden,
            heads: c.kda_heads,
            head_dim: c.kda_head_dim,
            conv_w: c.kda_conv,
            gate_lower_bound: Some(c.kda_gate_lower_bound as f32),
            eps: c.eps,
            // BV must shrink with the local head count or the state step strands
            // the chip: at tp8 (12 heads) a fixed 16 gives 96 of 256 items.
            bv: if tp >= 8 { 8 } else { 16 },
        },
        mla: K3MlaCfg {
            hidden: c.hidden,
            heads: c.heads,
            q_lora: c.q_lora,
            kv_lora: c.kv_lora,
            qk_rope: c.qk_rope,
            v_head: c.v_head,
            eps: c.eps,
            scale: 1.0 / ((c.qk_nope + c.qk_rope) as f32).sqrt(),
            n_split: k3_nsplit(ctx),
            gf: 4,
            fp8_kv: std::env::var("PLOW_FP8_KV").ok().as_deref() == Some("1")
                || std::env::var("PLOW_KV_FP8").ok().as_deref() == Some("1"),
        },
        moe: K3MoeCfg {
            hidden: c.hidden,
            latent: c.moe_latent,
            moe_inter: c.moe_inter,
            shared_inter: c.shared_exp * c.moe_inter,
            n_exp: c.n_exp,
            top_k: c.top_k,
            route_flags: u32::from(c.router_sigmoid) | (u32::from(c.renormalize) << 1),
            route_scale: c.route_scale,
            n_group: c.n_group,
            topk_group: c.topk_group,
            enc: MoeEnc::Mxfp4 as u32,
            // The grouped ops passed the full TP8 K3 gate at 4K+16: 103.161 -> 62.893 ms/token,
            // with all 17 dumped logit vectors byte-identical. Keep `0` as the reproducible
            // baseline arm; every other spelling, including unset, ships the measured winner.
            group_decode: std::env::var("K3_MOE_GROUP").ok().as_deref() != Some("0"),
        },
        vocab: c.vocab,
        first_k_dense: c.first_k_dense,
        dense_inter: c.dense_inter,
        prefix: K3_PREFIX.into(),
        tp,
    };

    // THE PROGRAM SET: one per prefill rung, then decode. `k3_emit_full` used to set
    // `prog_t: vec![1]` — decode only — which means a prompt longer than one token has NOTHING to
    // run and the runtime walks it through the decode program a token at a time
    // (`AmdServe::prefill`'s `decode_only` arm, one dispatch per prompt token). That is the whole
    // of TTFT, and with the host phase now measured at 3% of a decode token it is the largest
    // remaining serving gap on this path.
    //
    // DECODE IS BUILT FIRST, and the order is load-bearing rather than tidy. Every program in a
    // blob shares ONE tensor table; `Builder::set_tensor_dedup` lets a later builder adopt the
    // previous table and get the SAME handle back for a name it re-declares, growing the byte
    // count to the max. Building decode into an empty table means its handles are exactly what a
    // decode-only emit produced, so every instruction of the decode program is byte-identical and
    // the buckets can only APPEND (`k3_decode_program_is_unchanged_by_the_prefill_ladder` pins
    // it). The `progs` vector is reordered to buckets-then-decode below, because that is the
    // convention `Model::prog_t` and `manifest` read.
    let mut tensors: Vec<packet::devbuild::TensorDecl> = Vec::new();
    let mut gen = Vec::new();
    let mut built: Vec<packet::devbuild::Program> = Vec::new();
    let slot_rows = pf.iter().copied().max().unwrap_or(1);
    // BATCHED DECODE. `PLOW_DECODE_BATCH=B` makes the DECODE program carry B INDEPENDENT
    // SEQUENCES rather than one, which is a different thing from a prefill bucket's `t` rows and
    // is why it is paired with `RowKind::Sequences` rather than just a larger `t`
    // (perf-data/k3-batched-decode-design.md §1). B=1 is byte-identical to the pre-batch blob.
    //
    // Capped at PLOW_GEMV_MAXM=16: `AmdEngine::load` refuses above it because the gfx950 decode
    // GEMV is a compile-time row bucket and sequences 16.. would get zero logits. The KDA state
    // is the other bound — 0.44 GiB per slot, constant in context.
    let dbatch: u32 = std::env::var("PLOW_DECODE_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .clamp(1, 16);
    for (i, &t) in std::iter::once(&dbatch).chain(pf.iter()).enumerate() {
        let mut b = Builder::new(n_cu);
        b.set_tensor_dedup(true);
        // PLOW_L2_PLACE: `None` => byte-identical. Until this line the flag reached the dense-GQA
        // builders only, and `kimi_k3` is absent from the arch list that warns about being
        // ignored (`lib.rs:4327`), so setting it on K3 was a silent no-op.
        b.set_l2_placement(l2_layout);
        b.adopt_tensors(tensors.clone());
        crate::k3::emit_k3_model(&mut b,
            &mcfg,
            &|l| matches!(c.attn[l as usize], K3Attn::Kda),
            &layers,
            ctx,
            t,
            slot_rows,
            n_cu,
            // The DECODE program (i == 0) carries independent sequences when batched; every
            // prefill bucket is always consecutive tokens of one sequence.
            //
            // PLOW_K3_SEQ_ROWS FORCES the sequence-row carriers on at dbatch == 1. It is a
            // BISECTION INSTRUMENT, not a serving knob: at one row every carrier is a no-op by
            // construction (the only slot is slot 0, at offset 0 under either addressing), so a
            // B=1 emit with it on MUST reproduce the known-good B=1 stream token for token. If it
            // does not, the carrier that broke it is separable from batching itself — which is the
            // one question a B>1 run cannot answer, because at B>1 there is no reference stream to
            // compare against.
            if i == 0 && (dbatch > 1 || std::env::var_os("PLOW_K3_SEQ_ROWS").is_some()) {
                crate::k3::RowKind::Sequences
            } else {
                crate::k3::RowKind::Tokens
            },
        );
        // Every builder re-declares the same NoPE recipes and, under dedup, gets the same handles,
        // so any one of the lists is the whole set. Take the first — decode's — because it is the
        // one a decode-only emit would also have produced.
        if i == 0 {
            gen = b.gen_tensors();
        }
        let prog = b.finish();
        tensors = prog.tensors.clone();
        built.push(prog);
    }
    // buckets ascending, decode LAST — the order `Model::prog_t` and `manifest.rs` both assume
    // (`prog_t`'s last entry is the decode program; everything before it is a prefill bucket).
    let decode = built.remove(0);
    built.push(decode);
    // The decode entry is `dbatch`, not 1: at PLOW_DECODE_BATCH = B the decode program is emitted
    // at B rows (`RowKind::Sequences`), and `AmdEngine::load` cross-checks `prog_t.last()` against
    // `in.kvlen`'s row count. Leaving 1 here made a B=4 blob refuse itself at load with
    // "in.kvlen is 4 rows but the decode program is compiled for t=1" — a real mismatch between
    // two records of the same fact, which is exactly what that check is for.
    let prog_t: Vec<u32> = pf.iter().copied().chain(std::iter::once(dbatch)).collect();

    Model {
        n_cu,
        target: 0,
        tensors,
        progs: built,
        kv_row_insts: Vec::new(),
        prog_t,
        gen,
    }
}

/// The prefill rungs a K3 emit builds programs for.
///
/// `T` is a COMPILE-TIME constant of a packet, so the ladder is the only way a 20-token prompt and
/// a 4096-token one can both avoid paying for the other's program. The rungs are GLM's — this
/// family's prefill object is gfx950-only and there is no K3 measurement to place treads with, so
/// re-deriving them here would be guessing.
///
/// ON BY DEFAULT, unlike `PLOW_MLA_PREFILL`. That knob is off because the GLM MLA prefill arm can
/// be built at an ATTENTION-ONLY scope that never writes `act.logits` — a blob whose prefill
/// programs cannot sample while `Engine::has_prefill()` is true. K3 has no such scope: every
/// bucket here is a whole model, embed through argmax, so there is no half-built state to opt into.
///
///   * unset / `1` / `full` — the whole ladder, capped at `ctx`;
///   * `0`                  — decode only, byte-identical to before this path existed;
///   * `512,1024`           — those rungs only.
///
/// The list form is not cosmetic: every activation is declared for the WIDEST bucket, and
/// `act.pf.moe.part` alone is `T * top_k * latent` f32 — **1.9 GiB at T = 8192** on the shipped
/// geometry. A deployment that will only ever see 1k prompts should not pay for the 8192 rung.
pub(crate) fn k3_prefill_buckets(ctx: u32) -> Vec<u32> {
    match std::env::var("K3_PREFILL").ok().as_deref() {
        Some("0") => Vec::new(),
        None | Some("") | Some("1") | Some("full") => glm_prefill_buckets(ctx),
        Some(list) => list
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .filter(|&x| x > 1 && x <= ctx)
            .collect(),
    }
}

pub(crate) fn kimi_k3_emit(dir: &Path, ctx: u32, tp: u32, block_spec: Option<&str>) -> ! {
    let c = cfg_kimi_k3(dir);
    let (hdrs, have, total) = k3_shard_headers(dir);
    let mismatches = k3_config_vs_tensors(&c, &hdrs);

    eprintln!("kimi_k3: config ACCEPTED, emission REFUSED.\n");
    eprintln!("  checkpoint  {}", dir.display());
    if total == 0 {
        eprintln!("  shards      none on disk — every dimension below is CONFIG-ONLY, unverified");
    } else {
        eprintln!(
            "  shards      {have}/{total} readable, {} tensors{}",
            hdrs.len(),
            if have < total {
                "  (PARTIAL: a tensor's absence proves nothing)"
            } else {
                ""
            }
        );
    }
    eprintln!(
        "  text tower  {} layers = {} MLA + {} KDA | hidden {} | heads {} | vocab {} | ctx {ctx} | tp {tp}",
        c.layers, c.n_mla(), c.n_kda(), c.hidden, c.heads, c.vocab
    );
    eprintln!(
        "  MLA         q_lora {} kv_lora {} qk {}+{} v {} | nope={} out_gate={} rope_theta={}",
        c.q_lora,
        c.kv_lora,
        c.qk_nope,
        c.qk_rope,
        c.v_head,
        c.mla_nope,
        c.mla_out_gate,
        match c.rope_theta {
            Some(t) => format!("{t}"),
            None => "ABSENT".into(),
        }
    );
    eprintln!(
        "  KDA         {} heads x {} dim, conv k={}, {} gate, lower bound {}",
        c.kda_heads,
        c.kda_head_dim,
        c.kda_conv,
        if c.kda_full_rank_gate { "full-rank" } else { "low-rank" },
        c.kda_gate_lower_bound
    );
    eprintln!(
        "  MoE         {} routed (top-{}) + {} shared | inter {} | LATENT {} | norm={} | dense L<{} inter {}",
        c.n_exp, c.top_k, c.shared_exp, c.moe_inter, c.moe_latent, c.latent_norm,
        c.first_k_dense, c.dense_inter
    );
    eprintln!(
        "  router      {} | renorm={} | scale {} | groups {}/{} | act {:?}",
        if c.router_sigmoid { "sigmoid" } else { "softmax" },
        c.renormalize,
        c.route_scale,
        c.topk_group,
        c.n_group,
        c.hidden_act
    );
    eprintln!(
        "  quant       {} | {} bits | group {} | routed experts ONLY (attn/shared/dense/lm_head stay bf16)",
        c.quant_format, c.quant_bits, c.quant_group
    );
    eprintln!(
        "  attn map    0-based MLA layers {:?}{}",
        c.attn
            .iter()
            .enumerate()
            .filter(|(_, &k)| k == K3Attn::Mla)
            .map(|(i, _)| i)
            .take(6)
            .collect::<Vec<_>>(),
        if c.n_mla() > 6 { " …" } else { "" }
    );
    if let Some(spec) = block_spec {
        eprintln!("  --block     {spec:?} (accepted, but no layer kind is emittable — see below)");
    }
    if let Some(vis) = &c.vision {
        eprintln!(
            "\nSCOPE REFUSAL: this checkpoint is MULTIMODAL and plow implements the TEXT tower \
             only.\n  refused     MoonViT vision_tower ({} layers x {} hidden) + mm_projector \
             ({:?}).\n              Not skipped, not partially compiled: a text-only blob for a \
             multimodal\n              checkpoint loads, runs, and is wrong on every image prompt. \
             Strip\n              `vision_config` from config.json to ask for the text tower \
             explicitly.",
            vis.layers, vis.hidden, vis.projector
        );
    }

    if mismatches.is_empty() {
        eprintln!(
            "\n  config vs tensors: AGREE on every dimension the {} readable tensors can speak to.",
            hdrs.len()
        );
    } else {
        eprintln!(
            "\n  config vs tensors: {} DISAGREEMENT(S). TRUST THE TENSORS (GLM-5.2 lesson):",
            mismatches.len()
        );
        for m in &mismatches {
            eprintln!("    - {m}");
        }
    }

    // The other half of an honest refusal: what is ALREADY there. A gap list on its own invites
    // the next agent to rebuild machinery that exists — the mirror image of §4's "an arm exists
    // and nothing routes to it".
    eprintln!("\nALREADY COVERED — do not rebuild these:");
    eprintln!(
        "  mxfp4 routed experts  the MoE expert path already carries a weight ENCODING field \
         (MoeEnc::Mxfp4 = 2,\n                        i[6] decode / i[3] prefill) with an emitter \
         selector, and `wave_dot_mxfp4`\n                        (runtime/amd/op_moe.h:395) is \
         w4a16 — bf16 activations against packed fp4,\n                        exactly this \
         checkpoint's scheme (`input_activations: null`). The on-disk\n                        \
         layout is byte-exact: weight_packed [N, K/2], weight_scale [N, K/{}] u8.\n                 \
        Nothing to pack, nothing to convert.",
        c.quant_group.max(1)
    );
    eprintln!(
        "  E8M0 bias             127, per knob-contract §2, and CONFIRMED empirically from this \
         checkpoint:\n                        w1 scale bytes span 115-122, i.e. 2^-12..2^-5. Under \
         a bias of 0 the same\n                        bytes would mean 2^115, so the convention is \
         not in doubt."
    );
    eprintln!(
        "  router width          {} experts is inside the analysed bound: the LDS note at \
         runtime/amd/op_moe.h:50-56\n                        works the arena out to n_exp ~12000 \
         and the packed key gives the id 20 bits.\n                        (The \"n_exp<=256 is \
         tiny\" remark at op_moe.h:93 is stale, not a limit.)",
        c.n_exp
    );
    eprintln!(
        "  MLA geometry          q_lora/kv_lora/qk_nope/qk_rope/v_head are the SAME schema \
         cfg_glm parses, and\n                        every one agrees with the tensors. The \
         absorbed form (q_absorb/v_absorb) is\n                        already produced and \
         numerically verified by scripts/kimi_k3_prep.py."
    );

    let gaps = k3_gaps(&c);
    let (closed, open): (Vec<_>, Vec<_>) = gaps.iter().partition(|g| g.done.is_some());

    // CLOSED FIRST, and in full. The point of this section is to stop the next reader building
    // what already runs — which is a failure this report has actually caused, not a hypothetical.
    if !closed.is_empty() {
        eprintln!(
            "\nCLOSED — {} capabilities that WERE on this list and now LAND, each with the \
             real-weight\nevidence. DO NOT REBUILD THESE. Read the `done:` line before writing any \
             opcode.\n",
            closed.len()
        );
        for (i, g) in closed.iter().enumerate() {
            eprintln!("C{:<2} {}  [{}]", i + 1, g.what, g.scope);
            for line in textwrap72(g.done.unwrap()) {
                eprintln!("      {line}");
            }
            eprintln!("      residual work, if any:");
            for line in textwrap72(&g.fix.split_whitespace().collect::<Vec<_>>().join(" ")) {
                eprintln!("        {line}");
            }
            eprintln!();
        }
    }

    eprintln!(
        "\nMISSING CAPABILITIES — {} of them, ranked (blocker first). Each names where the fix \
         goes.\n",
        open.len()
    );
    for (i, g) in open.iter().enumerate() {
        eprintln!("{:>2}. {}  [{}]", i + 1, g.what, g.scope);
        for line in textwrap72(&g.why) {
            eprintln!("      {line}");
        }
        eprintln!("      fix:");
        for line in textwrap72(&g.fix.split_whitespace().collect::<Vec<_>>().join(" ")) {
            eprintln!("        {line}");
        }
        eprintln!();
    }
    panic!(
        "kimi_k3: {} unimplemented capabilities (listed above; {} further capabilities are CLOSED \
         and must not be rebuilt); no correct devblob exists for this checkpoint{}.",
        open.len(),
        closed.len(),
        if c.vision.is_some() {
            ", and its vision tower is out of scope and REFUSED (text-only)"
        } else {
            ""
        }
    );
}

/// Minimal greedy wrap so the capability report stays readable in a terminal.
fn textwrap72(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for w in s.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + w.len() > 92 {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(w);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// Kimi-K3 FRONT-END tests. There is no emit to test; what these lock down is the config
/// contract, because every one of them is a place a wrong answer would be silent.
#[cfg(test)]
mod kimi_k3_tests {
    use super::*;

    /// A faithful miniature of the real `config.json`: same key spellings, same nesting, same
    /// 1-based layer lists, scaled to 6 layers (2 MLA at 1-based {3,6}, 4 KDA).
    fn k3_json(patch: &[(&str, &str)]) -> Value {
        let base = r#"{
          "model_type": "kimi_k3",
          "architectures": ["KimiK3ForConditionalGeneration"],
          "vision_config": {"vt_num_hidden_layers": 27, "vt_hidden_size": 1024,
                            "mm_projector_type": "patchmergerv2"},
          "text_config": {
            "model_type": "kimi_linear",
            "hidden_size": 256, "num_attention_heads": 8, "num_hidden_layers": 6,
            "vocab_size": 1000, "intermediate_size": 512, "rms_norm_eps": 1e-5,
            "q_lora_rank": 64, "kv_lora_rank": 32, "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8, "v_head_dim": 16,
            "mla_use_nope": true, "mla_use_output_gate": true,
            "num_experts": 32, "num_experts_per_token": 4, "num_shared_experts": 2,
            "moe_intermediate_size": 96, "routed_expert_hidden_size": 128,
            "latent_moe_use_norm": true, "first_k_dense_replace": 1,
            "routed_scaling_factor": 1.0, "moe_router_activation_func": "sigmoid",
            "moe_renormalize": true, "num_expert_group": 1, "topk_group": 1,
            "hidden_act": "situ", "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0, "attn_res_block_size": 12,
            "linear_attn_config": {
              "num_heads": 8, "head_dim": 32, "short_conv_kernel_size": 4,
              "use_full_rank_gate": true, "gate_lower_bound": -5.0,
              "full_attn_layers": [3, 6], "kda_layers": [1, 2, 4, 5]
            },
            "quantization_config": {"format": "mxfp4-pack-quantized",
              "config_groups": {"group_0": {"weights": {"group_size": 32, "num_bits": 4}}}}
          }
        }"#;
        let mut v: Value = serde_json::from_str(base).unwrap();
        for (path, val) in patch {
            let seg: Vec<&str> = path.split('/').collect();
            let mut cur = &mut v;
            for s in &seg[..seg.len() - 1] {
                cur = cur.get_mut(s).unwrap();
            }
            let last = seg[seg.len() - 1];
            match *val {
                "<remove>" => {
                    cur.as_object_mut().unwrap().remove(last);
                }
                s => {
                    cur[last] = serde_json::from_str(s).unwrap();
                }
            }
        }
        v
    }

    /// The layer lists are 1-BASED (`configuration_kimi_k3.py::is_kda_layer` tests
    /// `(layer_idx + 1) in kda_layers`) and the real checkpoint proves it: MLA tensors live on
    /// 0-based layers 3, 7, 11, … while `full_attn_layers` starts at 4. An off-by-one here binds
    /// `q_a_proj` to a layer that ships `q_proj` — or, worse, does not fail and mixes the two.
    #[test]
    fn layer_lists_are_one_based_and_partition_the_tower() {
        let c = k3_cfg_from(&k3_json(&[]));
        assert_eq!(
            c.attn,
            vec![
                K3Attn::Kda, // 1-based 1
                K3Attn::Kda, // 2
                K3Attn::Mla, // 3
                K3Attn::Kda, // 4
                K3Attn::Kda, // 5
                K3Attn::Mla, // 6
            ]
        );
        assert_eq!((c.n_mla(), c.n_kda()), (2, 4));
        assert_eq!(c.attn.len(), c.layers as usize);
    }


    /// A checkpoint DIRECTORY holding the miniature config, for the two tests that drive
    /// `k3_build_model` end to end. There is no K3 checkpoint on this machine — only weights are
    /// missing, and `k3_build_model` never reads any.
    fn k3_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("plow_k3_{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        // head_dim 64, not the fixture's 32: `emit_kda_mixer` refuses a head_dim that is not a
        // multiple of the 64-lane wave, and these two tests are the only ones here that EMIT.
        let cfg = k3_json(&[("text_config/linear_attn_config/head_dim", "64")]);
        std::fs::write(d.join("config.json"), cfg.to_string()).unwrap();
        d
    }

    /// **THE DECODE PROGRAM IS UNCHANGED BY THE PREFILL LADDER — byte for byte.**
    ///
    /// This is the whole reason decode is built FIRST into an empty tensor table. Every program in
    /// a blob shares one table, so a bucket that declared a handle ahead of decode would renumber
    /// every `t[]` slot in the decode program: same graph, different bytes, and a regression no
    /// op-census test would see. Building decode first means the buckets can only APPEND.
    ///
    /// It compares the SERIALIZED instruction stream, not an op count. `DevInst` carries the
    /// tensor handles, the immediates and the counter wiring; two programs whose op sequences agree
    /// can still differ in every one of those.
    #[test]
    fn the_prefill_ladder_leaves_the_decode_program_byte_identical() {
        let d = k3_dir("ladder");
        let bare = k3_build_model(&d, 4096, 256, 1, &[], None);
        let laddered = k3_build_model(&d, 4096, 256, 1, &[128, 512, 1024], None);

        assert_eq!(bare.prog_t, vec![1], "no ladder ⇒ decode only, as before this path existed");
        assert_eq!(laddered.prog_t, vec![128, 512, 1024, 1], "buckets ascending, decode LAST");
        assert_eq!(laddered.progs.len(), 4);

        let a = bare.progs.last().unwrap();
        let b = laddered.progs.last().unwrap();
        assert_eq!(a.insts.len(), b.insts.len(), "decode op count moved");
        for (i, (x, y)) in a.insts.iter().zip(b.insts.iter()).enumerate() {
            assert_eq!(x.op, y.op, "decode inst {i}: opcode");
            assert_eq!(x.t, y.t, "decode inst {i}: tensor handles renumbered");
            assert_eq!(x.i, y.i, "decode inst {i}: immediates");
            assert_eq!(x.f, y.f, "decode inst {i}: floats");
            assert_eq!(x.blocks, y.blocks, "decode inst {i}: block count");
        }
        assert_eq!(a.stream, b.stream, "decode per-CU streams");
        assert_eq!(a.waits, b.waits, "decode counter waits");
        assert_eq!(a.succs, b.succs, "decode counter successors");

        // The shared table is a SUPERSET whose common prefix is decode's own, in order — that is
        // what makes the handles above stable. Sizes GROW (activations are declared for the widest
        // bucket); names and order do not move.
        for (i, td) in bare.tensors.iter().enumerate() {
            assert_eq!(td.name, laddered.tensors[i].name, "tensor {i} moved");
            assert!(laddered.tensors[i].bytes >= td.bytes, "tensor {} shrank", td.name);
        }
        assert!(laddered.tensors.len() > bare.tensors.len(), "buckets declare their own scratch");
        // And the ladder actually widened the shared activations rather than merely appending.
        let ring = |m: &Model| m.tensors.iter().find(|t| t.name == "kv.blkres").unwrap().bytes;
        assert_eq!(ring(&laddered), 1024 * ring(&bare), "the ring is [T][nb_cap][hidden]");
    }

    /// Every program must address the same peer slot B. The host has one peer layout for the
    /// whole blob and binds `act.dg_tp` at the blob-wide offset; a per-program `t*hidden*2`
    /// immediate makes decode reduce unrelated memory whenever a prefill ladder is present.
    #[test]
    fn k3_tp_peer_slot_is_program_invariant() {
        let d = k3_dir("tp_slot");
        let m = k3_build_model(&d, 4096, 256, 2, &[128, 512], None);
        let hidden = cfg_kimi_k3(&d).hidden;
        let want = 512 * hidden * 2;
        for (pi, p) in m.progs.iter().enumerate() {
            let slots: std::collections::BTreeSet<u32> = p
                .insts
                .iter()
                .filter(|i| {
                    i.op == DevOp::XReduce as u16 || i.op == DevOp::XReduceTwoShot as u16
                })
                .map(|i| i.i[2])
                .collect();
            assert_eq!(
                slots,
                [0, want].into_iter().collect(),
                "program {pi} (T={}) disagrees with the blob-wide peer layout",
                m.prog_t[pi]
            );
        }
    }

    /// The ladder itself: rungs are capped at `ctx`, and `K3_PREFILL=0` is the identity.
    ///
    /// The cap is not cosmetic — a program for `T > ctx` can never be invoked, and every
    /// activation in the blob is declared for the WIDEST bucket, so an uncapped ladder charges a
    /// 4k-context deployment for an 8192-row program it cannot run.
    #[test]
    fn the_prefill_ladder_is_capped_at_the_context() {
        assert_eq!(k3_prefill_buckets(131072), vec![128, 512, 1024, 2048, 4096, 8192]);
        assert_eq!(k3_prefill_buckets(2048), vec![128, 512, 1024, 2048]);
        assert_eq!(k3_prefill_buckets(64), Vec::<u32>::new(), "no rung fits — decode only");
    }

    /// Every bucket is a WHOLE MODEL: embed through argmax, with the grouped-MoE FFN.
    ///
    /// GLM's `PLOW_MLA_PREFILL=1` has an attention-only scope that stops at the post-attention
    /// norm and never writes `act.logits` — and `Engine::has_prefill()` is still true, so the
    /// runtime selects those programs and samples from a buffer nothing wrote. There is no such
    /// scope here, and this is what says so.
    #[test]
    fn every_prefill_bucket_is_a_whole_model() {
        let d = k3_dir("whole");
        let m = k3_build_model(&d, 4096, 256, 1, &[128, 512], None);
        let c = cfg_kimi_k3(&d);
        for (pi, p) in m.progs.iter().enumerate() {
            let t = m.prog_t[pi];
            let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();
            assert_eq!(n(DevOp::Embed), 1, "prog {pi} (T={t}): no embed prologue");
            assert_eq!(n(DevOp::ArgmaxFin), 1, "prog {pi} (T={t}): cannot sample");
            // Two AttnRes mixes on every layer, on every program. An AttnRes present in one bucket
            // and not another would make the two phases compute DIFFERENT MODELS.
            // ...and ONE model-level mix (`_apply_output_attn_res`) on every program too — the
            // site whose absence left `model.norm` reading only the post-snapshot partial sum.
            assert_eq!(n(DevOp::AttnRes), 2 * c.layers as usize + 1, "prog {pi} (T={t})");
            // The FFN half is present on both, in the spelling that phase has a kernel for.
            let moe_layers = (c.layers - c.first_k_dense) as usize;
            if t == 1 {
                assert_eq!(n(DevOp::MoeRouterTopk), moe_layers);
                assert_eq!(n(DevOp::MoeCombine), moe_layers);
                assert_eq!(n(DevOp::MoeCombinePf), 0);
            } else {
                assert_eq!(n(DevOp::MoeRouterTopkPf), moe_layers);
                assert_eq!(n(DevOp::MoeCombinePf), moe_layers);
                assert_eq!(n(DevOp::MoeCombine), 0);
            }
        }
    }

    /// The map is FIRST-CLASS data, not a count: a config whose MLA layers sit at different
    /// indices must produce a different map even though the counts are identical.
    #[test]
    fn attn_map_is_not_a_count() {
        let a = k3_cfg_from(&k3_json(&[]));
        let b = k3_cfg_from(&k3_json(&[
            ("text_config/linear_attn_config/full_attn_layers", "[1, 4]"),
            ("text_config/linear_attn_config/kda_layers", "[2, 3, 5, 6]"),
        ]));
        assert_eq!((a.n_mla(), a.n_kda()), (b.n_mla(), b.n_kda()));
        assert_ne!(a.attn, b.attn, "same counts must not mean the same model");
    }

    #[test]
    #[should_panic(expected = "appears in BOTH")]
    fn overlapping_layer_lists_are_rejected() {
        k3_cfg_from(&k3_json(&[(
            "text_config/linear_attn_config/kda_layers",
            "[1, 2, 3, 4, 5]",
        )]));
    }

    #[test]
    #[should_panic(expected = "are in neither list")]
    fn incomplete_layer_lists_are_rejected() {
        // Dropping 1-based layer 5 leaves 0-based layer 4 unclassified. Deriving KDA as the
        // complement of full_attn_layers would hide this; the partition check is the point.
        k3_cfg_from(&k3_json(&[(
            "text_config/linear_attn_config/kda_layers",
            "[1, 2, 4]",
        )]));
    }

    #[test]
    #[should_panic(expected = "num_hidden_layers is 6")]
    fn out_of_range_layer_index_is_rejected() {
        k3_cfg_from(&k3_json(&[(
            "text_config/linear_attn_config/full_attn_layers",
            "[3, 7]",
        )]));
    }

    /// `routed_expert_hidden_size` (3584 on the real model), NOT `moe_intermediate_size`, is the
    /// routed-expert GEMM's K. Verified against the checkpoint: `experts.0.w1.weight_packed` is
    /// [moe_inter, routed_expert_hidden_size/2].
    #[test]
    fn routed_experts_run_at_the_latent_width() {
        let c = k3_cfg_from(&k3_json(&[]));
        assert_eq!(c.moe_latent, 128);
        assert_eq!(c.moe_inter, 96);
        assert_ne!(c.moe_latent, c.moe_inter);
        assert_ne!(c.moe_latent, c.hidden);
        // The shape predicate the emitter will have to satisfy, spelled out.
        let (k, n) = (c.moe_latent as i64, c.moe_inter as i64);
        assert_eq!(vec![n, k / 2], vec![96, 64], "w1.weight_packed = [N, K/2]");
        assert_eq!(
            vec![n, k / c.quant_group as i64],
            vec![96, 4],
            "w1.weight_scale = [N, K/group]"
        );
    }

    /// The Kimi spellings differ from DeepSeek's. Reading `n_routed_experts` here would either
    /// hard-error or, with a default, silently compile a dense model.
    #[test]
    #[should_panic(expected = "missing required field \"num_experts\"")]
    fn deepseek_moe_spellings_are_not_accepted() {
        k3_cfg_from(&k3_json(&[("text_config/num_experts", "<remove>")]));
    }

    /// `rope_theta` is ABSENT from this config and must stay `None`. Silently applying GLM's RoPE
    /// to a NoPE model is a silent-corruption bug, not a missing feature; `cfg_glm`'s matching
    /// half is `require_mla_rope` (see `mla_rope_tests` in lib.rs).
    #[test]
    fn absent_rope_theta_is_none_not_a_default() {
        let c = k3_cfg_from(&k3_json(&[]));
        assert_eq!(c.rope_theta, None);
        assert!(c.mla_nope);
        assert!(
            k3_gaps(&c).iter().any(|g| g.what.contains("NO positional")),
            "a NoPE model must produce an explicit gap"
        );
    }

    /// Vision is recorded and REFUSED by name — never silently dropped.
    #[test]
    fn vision_is_recorded_for_explicit_refusal() {
        let c = k3_cfg_from(&k3_json(&[]));
        let v = c.vision.expect("vision_config must be recorded, not ignored");
        assert_eq!((v.layers, v.hidden), (27, 1024));
        // A text-only re-export has none and must not be flagged.
        let text_only = k3_cfg_from(&k3_json(&[("vision_config", "<remove>")]));
        assert!(text_only.vision.is_none());
    }

    /// Every gap must name a concrete fix site; a report that says "not supported" and stops is
    /// the failure mode this whole path exists to replace.
    #[test]
    fn every_gap_names_a_fix_site() {
        let gaps = k3_gaps(&k3_cfg_from(&k3_json(&[])));
        assert!(gaps.len() >= 8, "expected the full ranked list, got {}", gaps.len());
        for g in &gaps {
            assert!(
                g.fix.contains(".rs") || g.fix.contains(".h"),
                "gap {:?} names no file to change",
                g.what
            );
            assert!(!g.scope.is_empty() && !g.why.is_empty());
        }
    }

    /// A CLOSED capability must carry its evidence, and an OPEN one must not claim any.
    ///
    /// This is the assertion that keeps the report honest in both directions. Printing a landed
    /// capability as an unimplemented blocker sends the next agent to rebuild four opcodes that
    /// already dispatch (it did); marking one closed without the measured residual next to it
    /// makes the claim unfalsifiable.
    #[test]
    fn closed_gaps_carry_evidence_and_open_gaps_do_not_claim_any() {
        let gaps = k3_gaps(&k3_cfg_from(&k3_json(&[])));
        let closed: Vec<_> = gaps.iter().filter(|g| g.done.is_some()).collect();
        let open: Vec<_> = gaps.iter().filter(|g| g.done.is_none()).collect();
        assert!(
            closed.len() >= 5,
            "KDA, situ, AttnRes, LatentMoE, the MLA output gate and NoPE all landed with \
             real-weight gates; got {} closed",
            closed.len()
        );
        for g in &closed {
            // "validated" means a number, not an adjective.
            assert!(
                g.done.unwrap().contains("e-0") || g.done.unwrap().contains("e+0"),
                "closed gap {:?} cites no measured residual",
                g.what
            );
        }
        assert!(!open.is_empty(), "the full-model emit is still open");
        assert!(
            open[0].what.contains("full-model emit"),
            "the model-level assembly is THE remaining blocker and must rank first among the \
             open items; got {:?}",
            open[0].what
        );
    }

    /// The K3 config parse must NOT route through `require_mla_rope`.
    ///
    /// `require_mla_rope` lives on the `cfg_glm` path. A reader who sees it refuse NoPE naturally
    /// concludes it is what blocks the 93-layer K3 emit and "opens it for K3" — which changes
    /// nothing at all, because `k3_cfg_from` never calls it. Pin the fact so the next reader is
    /// not sent to the wrong file: K3's refusal comes from `kimi_k3_emit`, and the thing behind it
    /// is the absent model-level emitter.
    #[test]
    fn k3_is_refused_by_the_gap_report_not_by_require_mla_rope() {
        // A NoPE K3 config parses CLEANLY here. If `require_mla_rope` were on this path, this
        // call would panic instead of returning.
        let c = k3_cfg_from(&k3_json(&[]));
        assert!(c.mla_nope && c.rope_theta.is_none());
        // And the NoPE entry is CLOSED as a technique, not open as a blocker.
        let nope = k3_gaps(&c)
            .into_iter()
            .find(|g| g.what.contains("NO positional"))
            .expect("a NoPE model must still produce an explicit entry");
        assert!(nope.done.is_some(), "rung 3 proved the identity-table technique");
        assert!(
            nope.fix.contains("require_mla_rope"),
            "the entry must say, in the fix text, that opening require_mla_rope is NOT the fix"
        );
    }

    /// The top-k gap is a real threshold against `PLOW_MOE_MAX_TOPK 8u`, not a blanket "kimi is
    /// unsupported": a top-8 config must NOT raise it. The clamp it guards
    /// (`runtime/amd/op_moe.h:135`) is silent, so this is the one gap whose absence would be
    /// indistinguishable from correctness at runtime.
    #[test]
    fn topk_gap_is_conditional_on_the_kernel_bound() {
        let has = |c: &K3Cfg| k3_gaps(c).iter().any(|g| g.what.contains("top-k beyond"));
        // Against the constant, never a literal — the bound moved 8 -> 16 for this very model,
        // and a hardcoded test would then have asserted the opposite of what it means.
        let over = (crate::MOE_MAX_TOPK + 1).to_string();
        let at = crate::MOE_MAX_TOPK.to_string();
        assert!(has(&k3_cfg_from(&k3_json(&[(
            "text_config/num_experts_per_token",
            &over
        )]))));
        assert!(!has(&k3_cfg_from(&k3_json(&[(
            "text_config/num_experts_per_token",
            &at
        )]))));
    }

    /// K3's real top-16 is now INSIDE the bound, so the gap must be gone from the shipped report.
    /// This is the assertion that the raise actually removed a blocker rather than renaming one.
    #[test]
    fn kimi_k3_real_topk_is_within_the_raised_bound() {
        assert!(16 <= crate::MOE_MAX_TOPK, "K3 routes top-16");
        let c = k3_cfg_from(&k3_json(&[("text_config/num_experts_per_token", "16")]));
        assert!(!k3_gaps(&c).iter().any(|g| g.what.contains("top-k beyond")));
    }

    /// The latent-MoE gap must fire only when the routed experts really do read a different
    /// width; a hidden-width MoE (DeepSeek/GLM shape) is already covered by the existing emit.
    #[test]
    fn latent_moe_gap_is_conditional_on_the_width() {
        let has = |c: &K3Cfg| k3_gaps(c).iter().any(|g| g.what.contains("LATENT MoE"));
        assert!(has(&k3_cfg_from(&k3_json(&[]))));
        assert!(!has(&k3_cfg_from(&k3_json(&[(
            "text_config/routed_expert_hidden_size",
            "256" // == hidden_size
        )]))));
    }
}

// ===== tests moved from lib.rs (module breakdown): access mla internals directly =====
#[cfg(test)]
mod ckpt_quant_tests {
    //! Reading the weight encoding off the CHECKPOINT rather than off a flag.
    //!
    //! The first real GLM-5.2 emit produced a bf16 block from a checkpoint that is block-fp8 on
    //! disk — asking the loader to bind bf16 weights that do not exist, and never reaching the
    //! block-fp8 expert arms built for that exact model family. These pin the parse against the
    //! shapes that actually appear in `zai-org/GLM-5.2-FP8`'s `config.json`.
    use super::*;

    fn cfg_dir(name: &str, body: &str) -> std::path::PathBuf {
        // CARGO_TARGET_TMPDIR is only defined for integration tests, not unit tests.
        let d = std::env::temp_dir().join(format!("plow_{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("config.json"), body).unwrap();
        d
    }

    /// The real GLM-5.2-FP8 shape, including the two things that would fool a dtype-keyed probe:
    /// the key is `dtype` and not `torch_dtype`, and its value is "bfloat16" — the COMPUTE dtype —
    /// on a checkpoint whose weights are e4m3.
    #[test]
    fn block_fp8_checkpoint_is_detected_despite_a_bfloat16_dtype_field() {
        let d = cfg_dir(
            "ckpt_fp8",
            r#"{"model_type":"glm_moe_dsa","dtype":"bfloat16",
                "quantization_config":{"activation_scheme":"dynamic","fmt":"e4m3",
                "quant_method":"fp8","weight_block_size":[128,128]}}"#,
        );
        assert_eq!(mla_ckpt_enc(&d), Some(MoeEnc::Fp8Blk));
    }

    /// No `quantization_config` => the historical path, where the env flags decide and nothing
    /// about an existing workflow changes.
    #[test]
    fn unquantized_checkpoint_leaves_the_decision_to_the_flags() {
        let d = cfg_dir("ckpt_plain", r#"{"model_type":"kimi_k2","dtype":"bfloat16"}"#);
        assert_eq!(mla_ckpt_enc(&d), None);
    }

    /// 128 is not a parameter anywhere in this emitter — every scale-grid size is written as
    /// `div_ceil(128)` — so a checkpoint quantized at another block size would bind grids of the
    /// wrong shape against weights that look perfectly fine. The field exists because it can vary.
    #[test]
    #[should_panic(expected = "fp8_block_size")]
    fn a_different_block_size_is_refused() {
        let d = cfg_dir(
            "ckpt_blk64",
            r#"{"quantization_config":{"quant_method":"fp8","fmt":"e4m3",
                "weight_block_size":[64,64]}}"#,
        );
        mla_ckpt_enc(&d);
    }

    /// A quantization this emitter has no arms for must REFUSE, not fall back to bf16: the weights
    /// on disk are not bf16, so a bf16 packet is a WRONG packet rather than an unoptimised one.
    /// Same rule as w8a16-on-gfx950.
    #[test]
    #[should_panic(expected = "ckpt_quant_awq")]
    fn an_unsupported_quantization_is_refused_rather_than_downgraded() {
        let d = cfg_dir("ckpt_awq", r#"{"quantization_config":{"quant_method":"awq"}}"#);
        mla_ckpt_enc(&d);
    }

    /// A non-e4m3 fp8 flavour is refused too — `fmt` is checked, not assumed.
    #[test]
    #[should_panic(expected = "fp8_fmt_e5m2")]
    fn a_non_e4m3_fp8_format_is_refused() {
        let d = cfg_dir(
            "ckpt_e5m2",
            r#"{"quantization_config":{"quant_method":"fp8","fmt":"e5m2",
                "weight_block_size":[128,128]}}"#,
        );
        mla_ckpt_enc(&d);
    }
}

#[cfg(test)]
mod glm_tests {
    //! The GLM-5.2 (GlmMoeDsa) single-layer emit is the FIRST milestone-1 gate: the emitted op
    //! sequence must be identical to the 34-op MoE block that runtime/tests/
    //! glm52_real_block_gfx950_test.c validated on gfx950 against the HF oracle (real 256 experts,
    //! real [128,128] block-fp8 scales — plans/glm52-campaign.md "B4-CORE DONE"). Asserting op-for-op
    //! equality here, offline, means the emitted layer inherits that passing GPU result. No GPU, no
    //! weights — a pure structural equivalence proof, exactly as the Gemma pick_tile tests lock in
    //! the tile choice offline.
    use super::*;

    /// The real GLM-5.2-FP8 config dims (plans/glm52-arch.md). `layers` is trimmed — the single
    /// block only touches one layer.
    fn glm_ref_cfg() -> GlmCfg {
        GlmCfg {
            layers: 4,
            hidden: 6144,
            heads: 64,
            kv_lora: 512,
            q_lora: 2048,
            qk_nope: 192,
            qk_rope: 64,
            v_head: 256,
            vocab: 154880,
            eps: 1e-5,
            n_exp: 256,
            top_k: 8,
            n_group: 1,   // GLM-5.2 does not group-limit (why the flat top-k matched its oracle)
            topk_group: 1,
            moe_inter: 2048,
            dense_inter: 12288,
            first_k_dense: 3,
            route_scale: 2.5,
            attn_scale: (256f32).powf(-0.5),
            rope_theta: Some(8_000_000.0),
            prefix: "model.".into(),
            tp: 1,
            ep: false,
            group: false,
            index_heads: 32,
            index_dim: 128,
            index_topk: 2048,
            // indexer_types[0..4] = full,full,full,shared (real GLM-5.2 pattern); irrelevant to these
            // ctx=512 offline tests (DSA is gated OFF at ctx<=2048) but set for completeness.
            indexer_full: vec![true, true, true, false],
            has_dsa: true,
        }
    }

    fn emitted_ops(use_fp8: bool) -> Vec<u16> {
        let c = glm_ref_cfg();
        let mut b = Builder::new(256);
        // Emit MoE layer 3 (the B4 oracle's layer), matching the harness.
        let tn = declare_glm(&mut b, &c, 512, &[3]);
        let tensors = b.tensors();
        let mut b2 = Builder::new(256);
        b2.adopt_tensors(tensors);
        let mut xgate = 0u32;
        emit_glm_block(
            &mut b2,
            &c,
            &tn,
            0,
            512,
            MoeEnc::from_flags(use_fp8, false),
            tn.x,
            tn.xnext,
            &[],
            &mut xgate,
            &[],
        );
        b2.finish().insts.iter().map(|d| d.op).collect()
    }

    /// The reference MoE-block op sequence, in emission order. This is the B4 harness sequence
    /// (glm52_real_block_gfx950_test.c) with the two rope-slice GEMVs each followed by a dynamic
    /// interleaved HeadNormRope (HD=64) instead of a position-FOLDED GEMV — the production form that
    /// runtime/tests/glm52_run.c validates on gfx950 (dynamic rope at a fixed position reproduces the
    /// folded B4 numbers). The folded B4 result is inherited by transitivity: dynamic-at-fixed-pos ==
    /// the fold, proven numerically by the glm52_run ms1 gate.
    fn ref_sequence(use_fp8: bool) -> Vec<u16> {
        use DevOp::*;
        let (glu, down) = if use_fp8 {
            (MoeExpertGluFp8Blk, MoeExpertDownFp8Blk)
        } else {
            (MoeExpertGlu, MoeExpertDown)
        };
        let mut ops = vec![
            RmsNorm,        // input_layernorm
            GemvQkv, // FUSED A: q_a + kv_a + k_rope input projections (share xn) -> one GemvQkv
            RmsNorm, // q_a_layernorm
            GemvQkv, // FUSED G: Wqa (absorbed q_nope) + Wqr (q_rope) -> one GemvQkv
            HeadNormRope, // q_rope dynamic interleaved RoPE (HD=64)
            RmsNorm, // kv_a_layernorm -> latent cache
            HeadNormRope, // k_rope dynamic interleaved RoPE -> rope cache
            FlashMlaDecode, // MLA flash
            MlaMergeFold, // fused latent merge + W_uv fold (was FlashMerge + OUvFold)
            Gemv,    // o_proj
            Residual, // x_mid
            RmsNorm, // post_attention_layernorm
            Gemv,    // router SCORE GEMV (multi-CU wave-cooperative; the router split)
            MoeRouterTopk, // router tail: sigmoid+bias+norm_topk+scale (1-CU bit-exact selection)
            GemvGlu, // shared expert gate|up
            Gemv,    // shared expert down
        ];
        for _ in 0..8 {
            ops.push(glu);
            ops.push(down);
        }
        ops.push(MoeCombine);
        ops.into_iter().map(|o| o as u16).collect()
    }

    #[test]
    fn glm_block_matches_reference_bf16() {
        assert_eq!(
            emitted_ops(false),
            ref_sequence(false),
            "bf16 op sequence != reference"
        );
    }

    #[test]
    fn glm_block_matches_reference_fp8() {
        assert_eq!(
            emitted_ops(true),
            ref_sequence(true),
            "block-fp8 op sequence != reference"
        );
    }

    /// The dense (layers 0-2) block op sequence: shared MLA (16 ops) + block-fp8 SwiGLU (dense GLU
    /// op 47, dense down GEMV_FP8_BLK op 44) + residual = 19 ops.
    fn emitted_dense_ops() -> Vec<u16> {
        let c = glm_ref_cfg();
        let mut b = Builder::new(256);
        let tn = declare_glm(&mut b, &c, 512, &[0]); // layer 0 is dense (first_k_dense_replace=3)
        let tensors = b.tensors();
        let mut b2 = Builder::new(256);
        b2.adopt_tensors(tensors);
        let mut xgate = 0u32;
        emit_glm_dense_block(
            &mut b2,
            &c,
            &tn,
            0,
            512,
            MoeEnc::Fp8Blk,
            tn.x,
            tn.xnext,
            &[],
            &mut xgate,
            &[],
        );
        b2.finish().insts.iter().map(|d| d.op).collect()
    }

    /// Emit ONE MoE layer (slot layer 3) at `ctx`, with the indexer 'full'/'shared'/off, and return
    /// the op sequence. `full` binds an indexer on layer 3; `ctx>2048` arms the DSA gate.
    fn emitted_ops_dsa(ctx: u32, full: bool) -> Vec<u16> {
        let mut c = glm_ref_cfg();
        c.indexer_full = vec![false, false, false, full]; // layer 3 = MoE; full toggles its indexer
        let mut b = Builder::new(256);
        let tn = declare_glm(&mut b, &c, ctx, &[3]);
        let tensors = b.tensors();
        let mut b2 = Builder::new(256);
        b2.adopt_tensors(tensors);
        let mut xgate = 0u32;
        emit_glm_block(
            &mut b2,
            &c,
            &tn,
            0,
            ctx,
            MoeEnc::Fp8Blk,
            tn.x,
            tn.xnext,
            &[],
            &mut xgate,
            &[],
        );
        b2.finish().insts.iter().map(|d| d.op).collect()
    }

    #[test]
    fn glm_dsa_gate_off_below_cutover() {
        use DevOp::*;
        // ctx<=CROSSOVER (65536): NO DSA ops, dense FlashMlaDecode — byte-identical to the non-DSA MoE
        // block. 32768 is in the mid-ctx band, where the measured full-model TP4 winner is dense.
        let ops = emitted_ops_dsa(32768, true);
        assert!(
            ops.contains(&(FlashMlaDecode as u16)),
            "dense flash below cutover"
        );
        assert!(
            !ops.contains(&(FlashGatherDecode as u16)),
            "no gather below cutover"
        );
        assert!(
            !ops.contains(&(IndexScore as u16)),
            "no indexer below cutover"
        );
        assert_eq!(
            ops,
            ref_sequence(true),
            "ctx<=2048 == plain MoE block (DSA off)"
        );
    }

    /// Emit one MoE layer at `ctx`/`tp` and hand back the MLA flash-decode packet itself, so a
    /// test can read the fields the INTERPRETER dispatches on rather than just the opcode. Either
    /// opcode counts: `FlashGatherDecode` (the DSA arm, above the 64k cutover) and
    /// `FlashMlaDecode` are two instantiations of ONE wrapper, `exec_flash_mla_decode`, and both
    /// read GF from `i[7]` — so both had the missing GF=8 arm and both have it now.
    fn glm_flash_pkt(ctx: u32, tp: u32) -> crate::DevInst {
        let mut c = glm_ref_cfg();
        c.tp = tp;
        c.indexer_full = vec![false, false, false, false]; // keep the dense flash arm, not GATHER
        let mut b = Builder::new(256);
        let tn = declare_glm(&mut b, &c, ctx, &[3]);
        let tensors = b.tensors();
        let mut b2 = Builder::new(256);
        b2.adopt_tensors(tensors);
        let mut xgate = 0u32;
        emit_glm_block(
            &mut b2, &c, &tn, 0, ctx, MoeEnc::Fp8Blk, tn.x, tn.xnext, &[], &mut xgate, &[],
        );
        b2.finish()
            .insts
            .into_iter()
            .find(|d| {
                d.op == DevOp::FlashMlaDecode as u16 || d.op == DevOp::FlashGatherDecode as u16
            })
            .expect("an MLA flash-decode packet")
    }

    /// THE REVERSE COVERAGE CHECK for the GF=8 arm (knob-contract §4, read in the direction that
    /// guard does NOT cover: *an arm exists — does anything route to it, and does the packet the
    /// emitter builds match the body the interpreter will pick?*).
    ///
    /// This is the test that would have caught the original bug. `glm_gf` returned 8 on every
    /// long-context GLM blob for as long as the crossover has existed, `exec_flash_mla_decode`
    /// dispatched `if (gf == 2) <2> else <4>`, and the GF=8 body did not exist — so `i[7] = 8`
    /// selected the GF=4 arm silently. Nothing failed, nothing warned, and `flash_mla_cus` was
    /// written to MIRROR the wrong dispatch so the workgroup count stayed self-consistent with it.
    ///
    /// `blocks` is the load-bearing half: the kernel grid-strides `w = slice; w < n_work` over
    /// `n_batch*n_tok*(nh_l/GF)*nsplit`, so the packet's width has to be derived from the SAME GF
    /// the interpreter will instantiate. If these two ever disagree again, either work is dropped
    /// (width > n_work is only wasteful; width derived from a LARGER GF than the body uses drops
    /// items) or the chip is under-filled without anyone noticing.
    #[test]
    fn glm_flash_decode_packet_matches_the_arm_the_interpreter_dispatches() {
        // The packet-level harness is TP1 only: `emit_glm_mla` at tp>1 hands some collective an
        // empty CU list in this single-block fixture ("an op must run at least one CU"), at every
        // ctx and on both sides of this change. The TP4 shape is asserted arithmetically instead,
        // in `mla_fold_is_sized_to_its_work_items_and_never_flips_vt`.
        let fl = glm_flash_pkt(32768, 1);
        // GF=4, NOT 8, and this assertion is the whole point of the test now. `PLOW_GLM_GF8_ARM`
        // defaults to 0 (op_attention.h — the arm is a +32% decode regression by mere presence),
        // so the interpreter instantiates {2,4} and an emitted 8 would run the GF=4 body. It would
        // ALSO narrow `blocks` to (nh_l/8)*nsplit, because 9dc27bb made `flash_mla_cus` read i[7]
        // literally: the emitter would hand HALF the workgroups to a body that has full-GF work to
        // do. Measured cost of that mismatch on GLM-5.2 TP4 (arm-absent object, per-layer chain):
        // 97.6 -> 83.7 us at ctx 8192 and 168.1 -> 135.9 us at 32768; end-to-end median ITL over
        // 78 layers 28.58 -> 27.45 ms and 34.81 -> 31.49 ms, token-identical.
        assert_eq!(fl.i[7], 4, "long ctx bakes the GF the default object actually instantiates");
        // nsplit is capped for GLM_MLA_GF=4: fill = ceil(256/(64/4)) = 16, below ctx/NS_PER = 128.
        assert_eq!(fl.i[4], 16, "nsplit");
        // ... so the work-item count is (64/4)*16 = 256, the whole chip, and `blocks` matches it.
        assert_eq!(fl.blocks, 256, "GF=4 => (nh_l/4)*nsplit workgroups, chip-wide");

        // Short ctx stays on the GF=2 arm, which has always existed and always been dispatched.
        let sh = glm_flash_pkt(1024, 1);
        assert_eq!((sh.i[7], sh.blocks), (2, 256), "max_ctx <= 4096 keeps GF=2, chip-wide");

        // Every GF the emitter can bake MUST be one the interpreter instantiates. The set is
        // {2,4,8} and `exec_flash_mla_decode` dispatches exactly those three; anything else lands
        // in the `else` and silently runs GF=4, which is the bug this test exists to prevent.
        for &ctx in &[512u32, 1024, 4096, 8192, 32768, 131072] {
            let g = glm_flash_pkt(ctx, 1).i[7];
            assert!(matches!(g, 2 | 4 | 8), "ctx={ctx}: uninstantiated GF {g}");
        }
    }

    /// THE SELECTOR MUST BE TOLD THE LIVE KV LENGTH, AND THE INDEXER MUST DECLARE ITS GEOMETRY.
    ///
    /// Both halves are field-level and therefore invisible to every op-sequence test in this file:
    /// the ops were all present and in the right order the whole time.
    ///
    /// 1. `IndexSelect.t[4] = in.kvlen`. `i[0]` is the packet's MAX ctx, but `INDEX_SCORE` writes
    ///    `iscore[pos]` only for `pos < kvlen`. Without the operand the radix ranked `ctx - kvlen`
    ///    never-written words — and since DSA arms only above a 64k crossover, that was nearly the
    ///    whole array on any real decode step. The selector then handed the gather positions past
    ///    the end of the cache and `d_flash_mla_decode<...,GATHER=true>` applies NO mask, so those
    ///    rows were read as if they were real. `runtime/nvidia/op_dsa.cuh` records the same class of
    ///    defect against this kernel as `[RAG]`.
    /// 2. `IndexScore.i[1]/i[3]` = the indexer geometry the ISA contract has always specified. They
    ///    were left at ZERO while `interp.hip` hardcoded `DI_=128, HI_=32`, so a checkpoint with a
    ///    different geometry parsed cleanly and was silently strided wrong. `GlmCfg::dsa` now
    ///    refuses that outright; these fields make the packet self-describing so the two cannot
    ///    drift again without the assert catching it.
    #[test]
    fn glm_dsa_selector_is_bound_to_the_live_kv_length_and_declares_its_geometry() {
        let mut c = glm_ref_cfg();
        c.indexer_full = vec![false, false, false, true];
        let ctx = 131072; // above CROSSOVER, so the DSA arm is live
        let mut b = Builder::new(256);
        let tn = declare_glm(&mut b, &c, ctx, &[3]);
        let tensors = b.tensors();
        let mut b2 = Builder::new(256);
        b2.adopt_tensors(tensors.clone());
        let mut xgate = 0u32;
        emit_glm_block(
            &mut b2, &c, &tn, 0, ctx, MoeEnc::Fp8Blk, tn.x, tn.xnext, &[], &mut xgate, &[],
        );
        let insts = b2.finish().insts;
        let kvlen = tensors
            .iter()
            .position(|t| t.name == "in.kvlen")
            .expect("in.kvlen declared") as u32;

        let sel = insts
            .iter()
            .find(|d| d.op == DevOp::IndexSelect as u16)
            .expect("an IndexSelect packet");
        assert_eq!(
            sel.t[4], kvlen,
            "IndexSelect must read the LIVE kv length; i[0]={} is only the max ctx, and the score \
             kernel writes nothing past kvlen",
            sel.i[0]
        );
        assert_eq!(sel.i[0], ctx, "i[0] stays the max ctx (the scan upper bound)");
        assert_eq!(sel.i[1], c.index_topk, "i[1] is the top_k ceiling");

        let sc = insts
            .iter()
            .find(|d| d.op == DevOp::IndexScore as u16)
            .expect("an IndexScore packet");
        assert_eq!(sc.i[1], c.index_heads, "i[1] = index_heads, per dev_isa.h");
        assert_eq!(sc.i[3], c.index_dim, "i[3] = index_head_dim, per dev_isa.h");
        // ...and those are the ONLY values the kernel can execute, so the emitter must not be able
        // to produce anything else. `d_index_score_mfma` static_asserts HIc == 32.
        assert_eq!((sc.i[1], sc.i[3]), (32, 128), "the geometry interp.hip hardcodes");
    }

    #[test]
    fn glm_dsa_full_layer_emits_indexer() {
        use DevOp::*;
        // ctx>CROSSOVER, 'full': indexer (2 fp8 projections + LayerNorm + 2 rope + weights_proj GEMV +
        // score + select) then FLASH_GATHER (not dense).
        let ops = emitted_ops_dsa(131072, true);
        assert!(
            ops.contains(&(IndexScore as u16)),
            "full layer scores the indexer"
        );
        assert!(
            ops.contains(&(IndexSelect as u16)),
            "full layer selects top-k"
        );
        assert!(
            ops.contains(&(LayerNorm as u16)),
            "full layer k_norm LayerNorm"
        );
        assert!(ops.contains(&(FlashGatherDecode as u16)), "gather flash");
        assert!(
            !ops.contains(&(FlashMlaDecode as u16)),
            "no dense flash under DSA"
        );
    }

    #[test]
    fn glm_dsa_shared_layer_reuses_idx() {
        use DevOp::*;
        // ctx>CROSSOVER, 'shared': NO indexer ops (reuses the last full layer's idx) but still GATHERs.
        let ops = emitted_ops_dsa(131072, false);
        assert!(
            !ops.contains(&(IndexScore as u16)),
            "shared layer emits no score"
        );
        assert!(
            !ops.contains(&(IndexSelect as u16)),
            "shared layer emits no select"
        );
        assert!(
            !ops.contains(&(LayerNorm as u16)),
            "shared layer emits no k_norm"
        );
        assert!(
            ops.contains(&(FlashGatherDecode as u16)),
            "shared layer still gathers"
        );
    }

    #[test]
    fn glm_dense_block_sequence() {
        use DevOp::*;
        // Fused MLA (A+G): the 3 input GEMVs (q_a/kv_a/k_rope) -> one GemvQkv, and Wqa+Wqr -> one GemvQkv.
        let mla = vec![
            RmsNorm,
            GemvQkv,
            RmsNorm,
            GemvQkv,
            HeadNormRope,
            RmsNorm,
            HeadNormRope,
            FlashMlaDecode,
            MlaMergeFold,
            Gemv,
            Residual,
            RmsNorm,
        ];
        let mut want: Vec<u16> = mla.into_iter().map(|o| o as u16).collect();
        want.extend([DenseGluFp8Blk as u16, GemvFp8Blk as u16, Residual as u16]);
        assert_eq!(emitted_dense_ops(), want, "dense block op sequence");
        assert_eq!(emitted_dense_ops().len(), 15);
    }

    #[test]
    fn glm_block_op_count() {
        // 16 attention/pre-MoE ops after the A/G fusion (input q_a/kv_a/k_rope -> 1 GemvQkv, Wqa/Wqr
        // -> 1 GemvQkv; 2 dynamic-rope HeadNormRope + the 2-op router split + fused MlaMergeFold)
        // + 8*(glu+down) + 1 combine = 33 (was 36 pre-fusion).
        assert_eq!(emitted_ops(false).len(), 33);
    }

    // --- `--block` extraction path (M2, glm_build_block) ---------------------------
    // These exercise the actual single-block emit + descriptor build on the CPU with the
    // synthetic ref cfg (no checkpoint, no GPU): the block path must add NOTHING beyond
    // the validated per-layer block (no embed/tail), and the descriptor must reflect the
    // DSA IndexShare role + carried state.

    fn block_ops(c: &GlmCfg, ctx: u32, block: std::ops::Range<usize>) -> Vec<u16> {
        let (m, _desc) = glm_build_block(c, ctx, 256, block, true, "glm-ref", MlaArch::Glm);
        m.progs[0].insts.iter().map(|d| d.op).collect()
    }

    /// A single MoE-layer `--block 3` extraction emits EXACTLY the validated MoE block
    /// op sequence — no embed, no final-norm/lm_head/argmax tail. This is the numeric
    /// coverage lever: the block inherits glm_block_matches_reference_*'s GPU parity.
    #[test]
    fn glm_block_extract_matches_reference() {
        let c = glm_ref_cfg();
        assert_eq!(
            block_ops(&c, 512, 3..4),
            ref_sequence(true),
            "single-block --block 3 op sequence != validated MoE block"
        );
    }

    /// The weight namespace is a CFG PROPERTY, and it is the only thing a wrapper prefix moves.
    ///
    /// Kimi-K3's tower is `language_model.model.layers.{L}.…` — 497 052 of its 497 220 tensors,
    /// and NOT ONE under `model.`. Two properties are asserted, both against `c.prefix` rather
    /// than against a spelled-out string, so changing the prefix cannot leave the test agreeing
    /// with itself for the wrong reason:
    ///
    ///  * every checkpoint-bound tensor moves with the prefix, and
    ///  * no compiler-owned tensor moves at all (`kv.`/`act.`/`in.` are plow's, not the model's).
    ///
    /// Third property, and the one that guards the shipping models: switching the prefix changes
    /// only the SPELLING, so the two declarations are the same tensors in the same order with the
    /// same byte sizes.
    #[test]
    fn the_weight_prefix_is_cfg_data_and_moves_only_the_weights() {
        let decl = |pfx: &str| {
            let mut c = glm_ref_cfg();
            c.prefix = pfx.to_string();
            let mut b = Builder::new(256);
            let _ = declare_glm(&mut b, &c, 512, &[3]);
            b.tensors()
                .iter()
                .map(|t| (t.name.clone(), t.bytes))
                .collect::<Vec<_>>()
        };
        let flat = decl("model.");
        let nested = decl("language_model.model.");
        assert_eq!(flat.len(), nested.len(), "a prefix must not add or drop tensors");

        let (mut moved, mut fixed) = (0usize, 0usize);
        for ((fname, fbytes), (nname, nbytes)) in flat.iter().zip(nested.iter()) {
            assert_eq!(fbytes, nbytes, "{fname}: a prefix must not change a byte size");
            match fname.strip_prefix("model.") {
                // Anything the checkpoint names — weights AND the pointer tables declared beside
                // them, which `bind_packed_experts` resolves by that same prefix.
                Some(tail) => {
                    assert_eq!(
                        *nname,
                        format!("language_model.model.{tail}"),
                        "a name under the model prefix did not follow the cfg prefix"
                    );
                    moved += 1;
                }
                // `lm_head.weight` and every compiler-owned tensor are outside the prefix by
                // construction and must NOT move: `kv.3.krot` is plow's, not the model's.
                None => {
                    assert_eq!(fname, nname, "{fname} is not the checkpoint's to rename");
                    fixed += 1;
                }
            }
        }
        // Both sides non-trivial, so neither arm can pass by being empty.
        assert!(moved >= 15 && fixed >= 5, "moved {moved}, fixed {fixed}");

        // And the compiler-owned namespaces really are compiler-owned: none of them is ever
        // demanded of a checkpoint, under either spelling.
        for (n, _) in flat.iter().chain(nested.iter()) {
            if packet::names::is_runtime_tensor(n) {
                assert!(!packet::names::is_checkpoint_weight(n), "{n}");
            }
        }
    }

    /// The loaders' weight predicate must be a SUPERSET of the prefix allowlist it replaced —
    /// otherwise a shipping model would stop binding something it used to bind.
    ///
    /// The old rule was `starts_with("model.") || starts_with("fp8/")` (`exec/gpu.rs`,
    /// `serve/manager.rs`) plus `|| starts_with("lm_head")` on the AMD loader only. Asserted over
    /// a real GLM declaration, so it covers the expert POINTER tables — the one family that looks
    /// like a weight, lives under the model prefix, and must NOT be demanded of a checkpoint.
    #[test]
    fn the_new_weight_predicate_binds_everything_the_old_one_did() {
        let c = glm_ref_cfg();
        let mut b = Builder::new(256);
        let _ = declare_glm(&mut b, &c, 512, &[3]);
        let (mut n_old, mut n_tables) = (0usize, 0usize);
        for t in b.tensors() {
            let n = t.name.as_str();
            let old = n.starts_with("model.") || n.starts_with("fp8/") || n.starts_with("lm_head");
            let table = packet::names::is_host_filled_table(n);
            if table {
                n_tables += 1;
                assert!(!packet::names::is_checkpoint_weight(n), "{n}: host-filled, not a weight");
                continue;
            }
            if old {
                n_old += 1;
                assert!(
                    packet::names::is_checkpoint_weight(n),
                    "{n} used to bind from the checkpoint and no longer would"
                );
            }
        }
        assert!(n_old >= 15 && n_tables > 0, "weights {n_old}, tables {n_tables}");
    }

    /// A multi-layer `--block 2..4` extraction is the per-layer blocks concatenated
    /// (dense layer 2 then MoE layer 3), and the residual ping-pong lands the output in
    /// `act.x` after an even layer count.
    #[test]
    fn glm_block_extract_multi_layer_chains() {
        let c = glm_ref_cfg();
        let mut want = emitted_dense_ops(); // layer 2 (dense)
        want.extend(ref_sequence(true)); // layer 3 (MoE)
        assert_eq!(block_ops(&c, 512, 2..4), want, "2-layer block != dense++moe");
        let (_, desc) = glm_build_block(&c, 512, 256, 2..4, true, "glm-ref", MlaArch::Glm);
        assert_eq!(desc.outputs[0].name, "act.x", "even layer count -> act.x out");
        assert_eq!(desc.layer, 2, "descriptor.layer = block start");
    }

    /// Descriptor for a single MoE block: arch/kind/dims + `act.xnext` output (odd
    /// layer count) + kv carried state, DSA gate OFF at this ctx (no dsa_indices).
    #[test]
    fn glm_block_descriptor_moe() {
        let c = glm_ref_cfg(); // indexer_full[3] = false (reuse)
        let (_, d) = glm_build_block(&c, 512, 256, 3..4, true, "glm-ref", MlaArch::Glm);
        assert_eq!(d.arch, "glm_mla_dsa");
        assert_eq!(d.kind, vec!["mla_dsa", "moe_ffn"]);
        assert_eq!(d.dtype, "fp8");
        assert_eq!(d.dims.kv_lora, Some(512));
        assert_eq!(d.dims.q_lora, Some(2048));
        assert_eq!(d.dims.n_exp, Some(256));
        assert_eq!(d.dims.top_k, Some(8));
        assert_eq!(d.dims.shared_exp, Some(1));
        assert_eq!(d.dims.moe_inter, Some(2048));
        assert_eq!(d.dims.index_topk, Some(2048));
        assert_eq!(d.outputs[0].name, "act.xnext", "odd layer count -> act.xnext");
        assert_eq!(d.weights.prefix, "model.layers.3.");
        // Prefill is OPT-IN (`PLOW_MLA_PREFILL`), so the default block emit is still decode-only —
        // and must stay so, or every existing GLM asset gains buckets whose FFN half does not exist.
        assert!(
            d.programs.prefill_buckets.is_empty(),
            "GLM block emit is decode-only unless prefill buckets are requested"
        );
        // DSA gate off (ctx <= CROSSOVER): reuse role, but NO dsa_indices carried.
        assert_eq!(d.dsa_role.as_deref(), Some("reuse"));
        assert_eq!(d.carried_state.len(), 1);
        assert_eq!(d.carried_state[0].role, "kv");
        assert_eq!(d.carried_state[0].tensors, vec!["kv.3.ckv", "kv.3.krot"]);
    }

    /// Descriptor for a DENSE block (`--block 0`): no MoE dims, dense_ffn kind.
    #[test]
    fn glm_block_descriptor_dense() {
        let c = glm_ref_cfg();
        let (_, d) = glm_build_block(&c, 512, 256, 0..1, true, "glm-ref", MlaArch::Glm);
        assert_eq!(d.kind, vec!["mla_dsa", "dense_ffn"]);
        assert_eq!(d.dims.n_exp, None, "dense block has no MoE dims");
        assert_eq!(d.dims.moe_inter, None);
        assert_eq!(d.dims.kv_lora, Some(512), "MLA dims still present");
    }

    /// IndexShare (§7): under an ARMED DSA gate (ctx > CROSSOVER=65536), a 'reuse'
    /// layer carries `dsa_indices` in (it does not recompute the top-k), while an
    /// 'indexer' layer computes them in-block (kv carries its kidx cache instead).
    #[test]
    fn glm_block_dsa_indexshare_carried_state() {
        // 'reuse' layer 3 (indexer_types[3] = shared).
        let mut c = glm_ref_cfg();
        c.indexer_full = vec![false, false, false, false];
        let (_, reuse) = glm_build_block(&c, 131072, 256, 3..4, true, "glm-ref", MlaArch::Glm);
        assert_eq!(reuse.dsa_role.as_deref(), Some("reuse"));
        let dsa = reuse
            .carried_state
            .iter()
            .find(|s| s.role == "dsa_indices")
            .expect("reuse layer carries dsa_indices");
        assert_eq!(dsa.tensors, vec!["act.iidx"]);

        // 'indexer' layer 3 (indexer_types[3] = full): computes indices in-block, so
        // no dsa_indices carry; its kidx key cache joins the kv carried state.
        c.indexer_full = vec![false, false, false, true];
        let (_, idx) = glm_build_block(&c, 131072, 256, 3..4, true, "glm-ref", MlaArch::Glm);
        assert_eq!(idx.dsa_role.as_deref(), Some("indexer"));
        assert!(
            idx.carried_state.iter().all(|s| s.role != "dsa_indices"),
            "indexer layer does not carry dsa_indices in"
        );
        assert!(
            idx.carried_state[0].tensors.contains(&"kv.3.kidx".to_string()),
            "indexer layer carries its kidx cache"
        );
    }

    /// `MlaMergeFold` is sized to its OWN work-item count, never to `n_cu`, and the narrowing must
    /// leave the interpreter's VT branch exactly where it found it — a different VT is a different
    /// fold map and therefore different arithmetic (`exec_mla_merge_fold`, op_attention.h).
    #[test]
    fn mla_fold_is_sized_to_its_work_items_and_never_flips_vt() {
        let all: Vec<u32> = (0..256u32).collect();
        // GLM-5.2, v_head 256: bh*ceil(256/32) = bh*8, and bh*8 <= nblk keeps VT at 32.
        for &(nh_l, want) in &[(16u32, 128usize), (8, 64), (4, 32), (32, 256)] {
            let got = mla_fold_cus(&all, nh_l, 256);
            assert_eq!(got.len(), want, "GLM nh_l={nh_l}");
            assert_eq!(got[0], 0, "the narrowing keeps slice 0 == workgroup 0");
            // The width IS the work-item count, so no workgroup is left without an item.
            let vt = mla_fold_vt(nh_l, got.len() as u32, 256);
            assert_eq!(vt, mla_fold_vt(nh_l, 256, 256), "VT branch must not move");
            assert_eq!(nh_l * 256u32.div_ceil(vt), got.len() as u32, "sized to n_work");
        }
        // Kimi-K3, v_head 128: bh=16 would pick VT=32 at 256 wgs and VT=128 at the narrowed 64,
        // which reassociates the fold. The rule must REFUSE rather than narrow.
        assert_eq!(mla_fold_cus(&all, 16, 128).len(), 256, "v=128 narrowing flips VT — refuse");
        // ... but a bh too large for VT=32 in the first place narrows safely.
        assert_eq!(mla_fold_cus(&all, 96, 128).len(), 96, "v=128, VT=256 both sides");
        // Prefill (n_batch = t folded into bh) hands the whole machine back.
        assert_eq!(mla_fold_cus(&all, 128 * 16, 256).len(), 256, "prefill bucket is inert");
        // The flash-decode rule cancels EXACTLY at GF=4 — `glm_nsplit`'s fill cap uses the same
        // `GLM_MLA_GF`, so `(nh_l/4)*fill == n_cu` and the long-ctx blob is chip-wide. THIS IS THE
        // REGRESSION GUARD FOR THE HALF-WIDTH DEFECT: while `glm_gf` returned 8 against an object
        // built without `-DPLOW_GLM_GF8_ARM=1`, these packets carried 128 workgroups for 256 work
        // items — correct output (the body grid-strides) at half the parallelism, worth a measured
        // -3.35 ms/token at ctx 32768 end-to-end. If this ever reads 128 again, either `glm_gf`
        // went back to 8 or `flash_mla_cus` stopped agreeing with the body.
        // tp4 (nh_l=16) and tp2 (nh_l=32) land exactly on the chip. tp8 (nh_l=8) does NOT, and
        // that is pre-existing and deliberate: its `fill` is 128, but `NS_CEIL_MEASURED` holds
        // nsplit at 64 because the ladder behind the ceiling is a tp4 ladder. tp8's long-ctx flash
        // therefore runs 2*64 = 128 items on 256 CUs — HALF THE CHIP — and whether raising it pays
        // is an open, measurable question, not an assumption to bake in. See `glm_nsplit`.
        for &(nh_l, ctx, want) in &[(16u32, 65536u32, 256usize), (32, 65536, 256), (8, 65536, 128)]
        {
            let got = flash_mla_cus(&all, 1, 1, nh_l, glm_gf(ctx, nh_l), glm_nsplit(ctx, nh_l));
            assert_eq!(got.len(), want, "nh_l={nh_l} ctx={ctx}: GF=4 work items");
        }
        // The GF=8 arm, when someone builds it (-DPLOW_GLM_GF8_ARM=1) and pins PLOW_GLM_GF=8,
        // halves the work items and needs 2x nsplit to be chip-wide again — at 2x the merge
        // inputs, and the merge is a function of nsplit ALONE (measured: gf4/ns64 26.5 us vs
        // gf8/ns64 26.5; gf4/ns128 41.7 vs gf8/ns128 47.2). That is why matching WORK ITEMS across
        // GF is not a fair trade and GF=8 lost the matched-item A/B at both ctx.
        assert_eq!(
            flash_mla_cus(&all, 1, 1, 16, 8, glm_nsplit(65536, 16)).len(),
            128,
            "GF=8 at the GF=4 nsplit is half the chip"
        );
        assert_eq!(
            flash_mla_cus(&all, 1, 1, 16, 8, 2 * glm_nsplit(65536, 16)).len(),
            256,
            "GF=8 at 2x nsplit restores full fill"
        );
        assert_eq!(
            flash_mla_cus(&all, 1, 1, 16, glm_gf(1024, 16), glm_nsplit(1024, 16)).len(),
            128,
            "max_ctx 1024 is GF=2 with a GF=4-sized nsplit: 128 items, not 256"
        );
        // i[7] is read LITERALLY by `flash_mla_cus`, so the value the emitter bakes MUST be the
        // one the interpreter instantiates — and with `PLOW_GLM_GF8_ARM=0` (the default) that set
        // is {2,4}. This pair is the invariant: the emitted GF and the dispatch width agree.
        assert_eq!(glm_gf(65536, 16), 4, "long ctx bakes the GF the default object runs");
        assert_eq!(flash_mla_cus(&all, 1, 1, 16, 4, 64).len(), 256, "GF=4 => nh_l/4 groups");
        // `n_grp = nh_l / GF` is integer: a GF larger than this rank's head shard makes the flash
        // do NOTHING. GLM-5.2 n_head=64 reaches nh_l=4 at tp16, so the clamp is live, not
        // hypothetical.
        assert_eq!(glm_gf(65536, 8), 4, "tp8 (nh_l=8): 4, the default arm");
        assert_eq!(glm_gf(65536, 4), 4, "tp16 (nh_l=4) must clamp to 4, not divide to zero");
        assert_eq!(glm_gf(65536, 2), 2, "nh_l=2 clamps all the way to 2");
        // `PLOW_GLM_GF=8` still reaches the arm — it is the only way to run its A/B — and it is
        // still clamped by divisibility. Asserted at the end of this test, where the env var is
        // set (§6g-GF8: a pinned 8 on a tp16 blob would otherwise emit all-zero attention).
        // GF MUST DIVIDE nh_l, NOT MERELY FIT IN IT. `n_grp = nh_l / GF` truncates and the only
        // head cursor is `h0 = hg*GF`, so the `nh_l % GF` tail is never visited and its opart /
        // mlpart rows are read back by the merge uninitialised. Kimi-K3 is the first model in the
        // tree with a non-power-of-two head count (96), and it is the reference TP that breaks:
        //   tp8  -> nh_l=12: the old `g <= nh_l` rule took GF=8, n_grp=1, heads 8..11 DROPPED
        //   tp16 -> nh_l=6 : took GF=4, n_grp=1, heads 4..5 DROPPED
        assert_eq!(glm_gf(65536, 12), 4, "K3 tp8 (nh_l=12): 8 does not divide 12, 4 does");
        assert_eq!(glm_gf(65536, 6), 2, "K3 tp16 (nh_l=6): 4 does not divide 6, 2 does");
        assert_eq!(glm_gf(65536, 24), 4, "K3 tp4 (nh_l=24): 4, the default arm (8 is pin-only)");
        assert_eq!(glm_gf_prefill(65536, 12), 4, "prefill twin: 12 % 4 == 0");
        assert_eq!(glm_gf_prefill(65536, 6), 2, "prefill twin: 6 % 4 != 0, fall to 2");
        // Every nh_l a shipping model produces is a power of two, and every power of two is
        // divisible by 8, 4 and 2 — so this change moves NO emitted packet for GLM-5.2, Kimi-K2.7
        // or DeepSeek-V3. Pin that, because "it is byte-identical" is the claim that makes this
        // safe to land without re-validating those blobs.
        for nh_l in [2u32, 4, 8, 16, 32, 64, 128] {
            for ctx in [1024u32, 65536] {
                assert_eq!(
                    glm_gf(ctx, nh_l),
                    [8u32, 4, 2]
                        .into_iter()
                        .find(|&g| g <= if ctx <= GLM_GF_CROSSOVER { 2 } else { 4 } && g <= nh_l)
                        .unwrap_or(2),
                    "power-of-two nh_l={nh_l} ctx={ctx} must be unchanged by the divisibility rule"
                );
            }
        }
        // The pin is clamped too: a sweep must not be able to emit all-zero attention.
        std::env::set_var("PLOW_GLM_GF", "8");
        assert_eq!(glm_gf(1024, 16), 8, "the pin overrides the crossover");
        assert_eq!(glm_gf(1024, 4), 4, "the pin is still clamped by nh_l");
        std::env::remove_var("PLOW_GLM_GF");
        // The knob restores the control arm exactly.
        std::env::set_var("PLOW_GLM_WGFIT", "0");
        assert_eq!(mla_fold_cus(&all, 16, 256).len(), 256);
        assert_eq!(blocked_gemv_cus(&all, 2624).len(), 256);
        std::env::remove_var("PLOW_GLM_WGFIT");
    }

    /// An ODD head shard cannot be expressed by ANY instantiated GF, and the emit must say so.
    ///
    /// The interpreter instantiates GF in {2,4,8}; none divides an odd `nh_l`, so there is no
    /// correct packet to emit and the only honest outcome is a refusal at compile time. The
    /// runtime cannot catch this — unvisited heads are not an error condition anywhere, they are
    /// memory nobody wrote that the merge consumes as if it were a partial. Reachable on Kimi-K3
    /// (96 heads) at tp32: 96/32 = 3.
    #[test]
    #[should_panic(expected = "does not divide this rank's head shard")]
    fn an_odd_head_shard_is_refused_rather_than_silently_truncated() {
        glm_gf(65536, 3);
    }

    /// A GV_BLOCKED gemv packet owns columns in runs of `per = ceil(n/nblk)`; the narrowing drops
    /// only the ceiling tail that owns none, and it is a FIXED POINT of that arithmetic, so every
    /// surviving workgroup's column run is byte-for-byte the one it had before.
    #[test]
    fn blocked_gemv_drops_only_the_empty_ceiling_tail() {
        let all: Vec<u32> = (0..256u32).collect();
        for &n in &[1u32, 63, 255, 256, 257, 512, 2624, 6144, 9216, 154880] {
            let got = blocked_gemv_cus(&all, n);
            let per = n.div_ceil(256);
            let per_after = n.div_ceil(got.len() as u32);
            assert_eq!(per_after, per, "n={n}: `per` moved, the column map is not preserved");
            // every surviving workgroup owns at least one column ...
            assert!((got.len() as u32 - 1) * per < n, "n={n}: kept an empty workgroup");
            // ... and no column is dropped.
            assert!(got.len() as u32 * per >= n, "n={n}: dropped columns");
        }
        // GLM-5.2 TP4: fusion A is 2048+512+64 over 256 workgroups.
        assert_eq!(blocked_gemv_cus(&all, 2048 + 512 + 64).len(), 239);
        // fusion G (16*512 + 16*64) already divides evenly.
        assert_eq!(blocked_gemv_cus(&all, 16 * 512 + 16 * 64).len(), 256);
    }

    /// The MLA flash-decode split factor is the ctx-scaled cost optimum, capped by the ACTUAL
    /// per-rank chip-fill `fill = ceil(n_cu / (nh_l/GF))` and the KV-tile count. `glm_nsplit` takes
    /// nh_l (= n_head/tp) so the cap is correct under TP/EP — the pre-fix bug sized it from the
    /// global n_head=64, pinning the cap to tp=1's fill regardless of TP. Asserts the caps and the
    /// measured (MI350X mla_perf) chain optima: ns~16 up to 8k, ns~64 at 32k.
    #[test]
    fn glm_nsplit_is_ctx_scaled_and_capped_per_rank() {
        let n_cu = 256u32;
        for &(_tp, nh_l) in &[(1u32, 64u32), (2, 32), (4, 16), (8, 8)] {
            let n_grp = (nh_l / GLM_MLA_GF).max(1);
            let fill = (n_cu + n_grp - 1) / n_grp;
            let mut prev = 0u32;
            for &ctx in &[1024u32, 4096, 8192, 16384, 32768, 65536, 131072] {
                let ns = glm_nsplit(ctx, nh_l);
                let kv_tiles = ctx.div_ceil(32);
                // Cap 1 — never over-split past the chip (the nh_l-aware fill).
                assert!(
                    ns <= fill,
                    "nh_l={nh_l} ctx={ctx}: ns={ns} exceeds chip-fill {fill}"
                );
                // Cap 2 — never split finer than there are KV tiles (no empty splits).
                assert!(
                    ns <= kv_tiles,
                    "nh_l={nh_l} ctx={ctx}: ns={ns} exceeds {kv_tiles} KV tiles"
                );
                // Monotone non-decreasing in ctx (more latent => more useful splits).
                assert!(
                    ns >= prev,
                    "nh_l={nh_l} ctx={ctx}: ns={ns} < prev {prev} (not ctx-monotone)"
                );
                prev = ns;
            }
        }
        // MEASURED chain optima locked in, one assert per rung of the ladder in the header
        // (GLM-5.2 TP4, arm-absent object, per-layer chain us; the whole table is there). These
        // are the rows this rule exists to reproduce, so they are pinned individually rather than
        // as "ns grows with ctx" — the previous constant satisfied that and still missed two.
        for &nh_l in &[8u32, 16] {
            for &(ctx, want, why) in &[
                (1024u32, 16u32, "61.1 vs ns32's 65.8"),
                (4096, 16, "66.6 vs ns32's 67.2 — still the floor's rung"),
                (8192, 32, "73.3 vs ns16's 90.1 — the rung ctx/512 got WRONG"),
                (16384, 64, "88.1 vs ns32's 103.7 — the other rung ctx/512 got wrong"),
                (32768, 64, "135.9, and 128 regresses to 141.3"),
                (65536, 64, "183.0, fill-capped anyway"),
            ] {
                assert_eq!(
                    glm_nsplit(ctx, nh_l),
                    want,
                    "nh_l={nh_l} ctx={ctx}: measured optimum is ns={want} ({why})"
                );
            }
        }
        // tp=1 is chip-full at ns=16 (n_grp=16), so the fill cap pins every ctx to 16 — byte-identical
        // to the pre-fix path (no regression on single-GPU decode).
        for &ctx in &[1024u32, 8192, 32768, 131072] {
            assert_eq!(
                glm_nsplit(ctx, 64),
                16,
                "tp=1 ctx={ctx}: fill-capped to 16 (unchanged)"
            );
        }
        // The refined rule must NOT full-fill mid ctx (the measured 8k regression at ns=128): at tp=8
        // 8k it stays at the floor, not fill=128.
        assert!(
            glm_nsplit(8192, 8) < ((256 + 1) / 2),
            "tp=8 8k must not full-fill (mid-ctx merge regression)"
        );
    }

    #[test]
    fn glm_cfg_qk_scale() {
        let c = glm_ref_cfg();
        assert_eq!(c.qk_head(), 256);
        assert!(
            (c.attn_scale - 0.0625).abs() < 1e-6,
            "MLA scale = 1/sqrt(256)"
        );
        assert!(
            c.is_dense(0) && c.is_dense(2) && !c.is_dense(3),
            "first_k_dense_replace=3"
        );
    }

    /// `GLM_LINEAR_FP8` re-declares four tensors per layer at HALF their bf16 size, and the
    /// PREFILL emitters have to be told. They were not, for three interpreters: `declare_glm_rows`
    /// REFUSED a stacked emit (`require_lin_fp8_decode_only`) rather than put a bf16 `Gemm` on fp8
    /// bytes, because no dense T-row block-fp8 GEMM existed. `GemmFp8Blk` (107) is that GEMM, so
    /// what is pinned here is the ROUTE, not the refusal.
    ///
    /// Pinned through `emit_pf_gemm_fp8_blk` rather than by setting `GLM_LINEAR_FP8` and calling
    /// the emitters: the knob is process-global env state, cargo runs tests in parallel threads,
    /// and a sibling test that counts tensors sees the four extra `weight_scale_inv` handles appear
    /// under it. That is not hypothetical — it broke
    /// `the_weight_prefix_is_cfg_data_and_moves_only_the_weights` (58 vs 54) when the old version of
    /// this test set the var. Test the pure part as a pure function.
    #[test]
    fn glm_linear_fp8_prefill_routes_to_the_block_fp8_gemm() {
        let mut b = Builder::new(256);
        let w = b.tensor("w.weight_fp8", 6144 * 4096);
        let s = b.tensor("w.weight_scale_inv", 48 * 32 * F32);
        let x = b.tensor("act.x", 512 * 4096 * BF16);
        let o = b.tensor("act.o", 512 * 6144 * BF16);
        let all: Vec<u32> = (0..256u32).collect();
        emit_pf_gemm_fp8_blk(&mut b, &all, o, x, w, s, 512, 6144, 4096, &[]);
        let p = b.finish();
        assert_eq!(p.insts.len(), 1);
        let d = &p.insts[0];
        assert_eq!(
            d.op,
            DevOp::GemmFp8Blk as u16,
            "the prefill arm must be the block-fp8 GEMM, never a bf16 Gemm on fp8 bytes"
        );
        // The scale grid is NOT optional and must ride t[3]: a null there is a wrong number, not a
        // fault, because the kernel's promotion multiplies by whatever it reads.
        assert_eq!([d.t[0], d.t[1], d.t[2], d.t[3]], [o, x, w, s]);
        assert_eq!([d.i[0], d.i[1], d.i[2]], [512, 6144, 4096]);
    }

    /// A block-fp8 weight without its scale grid is a NULL pointer inside the kernel's promotion.
    /// The two handles are declared as a pair; refuse rather than emit half of one.
    #[test]
    fn glm_linear_fp8_prefill_refuses_a_weight_with_no_scale_grid() {
        let mut b = Builder::new(256);
        let w = b.tensor("w.weight_fp8", 64);
        let x = b.tensor("act.x", 64);
        let o = b.tensor("act.o", 64);
        let all: Vec<u32> = (0..256u32).collect();
        let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            emit_pf_gemm_fp8_blk(&mut b, &all, o, x, w, TENSOR_NONE, 8, 8, 8, &[]);
        }))
        .err()
        .expect("a scale-less block-fp8 GEMM must be refused, not emitted");
        let msg = e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(msg.contains("weight_scale_inv"), "name the missing handle; got: {msg}");
    }
}

#[cfg(test)]
mod kimi_tests {
    //! Kimi K2.7 / DeepSeek MLA+MoE `--block` extraction (M3, plans/block-asset-harness.md §5.0/
    //! §5.3/§7). Kimi REUSES the GLM MLA + MoE emit verbatim (glm_build_block) with a cfg that holds
    //! the DSA gate off (`has_dsa=false`) — so a Kimi block is the SAME op sequence as a GLM block
    //! BELOW the DSA crossover, minus every indexer artifact: no DSA scratch, FlashMlaDecode (never
    //! FlashGatherDecode) at ANY ctx, and a descriptor with no dsa_role / no index_* dims. These
    //! synthetic-CPU tests are the only verification available on this box (no Kimi checkpoint, no
    //! transformers → no real blob, no GPU parity). They lock in the op sequence + descriptor exactly
    //! as glm_tests does for GLM.
    use super::*;

    /// Synthetic small Kimi cfg (structurally faithful: DeepSeek-schema MLA + MoE, first_k_dense=1
    /// so layer 0 is dense and 1+ are MoE, has_dsa=false). Real K2.7 geometry is hidden 7168 / 64
    /// heads / kv_lora 512 / q_lora 1536 / qk_nope 128 / qk_rope 64 / v_head 128 / 384 exp / top_k 8
    /// / moe_inter 2048; the shape logic is dim-agnostic, so small dims exercise the same emit.
    fn kimi_ref_cfg() -> GlmCfg {
        GlmCfg {
            layers: 4,
            hidden: 256,
            heads: 4,
            kv_lora: 64,
            q_lora: 96,
            qk_nope: 32,
            qk_rope: 16,
            v_head: 32,
            vocab: 1000,
            eps: 1e-5,
            n_exp: 16,
            top_k: 4,
            n_group: 1,
            topk_group: 1,
            moe_inter: 128,
            dense_inter: 256,
            first_k_dense: 1,
            route_scale: 2.5,
            attn_scale: (48f32).powf(-0.5), // 1/sqrt(qk_nope+qk_rope = 48)
            rope_theta: Some(50_000.0),
            prefix: "model.".into(),
            tp: 1,
            ep: false,
            group: false,
            // Indexer fields are inert under has_dsa=false (never read); set placeholders.
            index_heads: 8,
            index_dim: 32,
            index_topk: 64,
            indexer_full: Vec::new(), // Kimi/DeepSeek config has no `indexer_types`
            has_dsa: false,
        }
    }

    fn block_ops(c: &GlmCfg, ctx: u32, block: std::ops::Range<usize>, arch: MlaArch) -> Vec<u16> {
        let (m, _d) = glm_build_block(c, ctx, 256, block, true, "kimi-ref", arch);
        m.progs[0].insts.iter().map(|d| d.op).collect()
    }

    /// Expected MoE-block op sequence: shared MLA (12 ops) + router split (2) + shared expert (2) +
    /// top_k×(glu, down) + MoeCombine. IDENTICAL shape to glm_tests::ref_sequence but parameterized
    /// on top_k — the reuse the arch is built on.
    fn kimi_moe_sequence(use_fp8: bool, top_k: usize) -> Vec<u16> {
        use DevOp::*;
        let (glu, down) = if use_fp8 {
            (MoeExpertGluFp8Blk, MoeExpertDownFp8Blk)
        } else {
            (MoeExpertGlu, MoeExpertDown)
        };
        let mut ops = vec![
            RmsNorm,        // input_layernorm
            GemvQkv,        // FUSED A: q_a + kv_a + k_rope down-projections
            RmsNorm,        // q_a_layernorm
            GemvQkv,        // FUSED G: q_absorb + q_rope down
            HeadNormRope,   // q_rope dynamic interleaved RoPE
            RmsNorm,        // kv_a_layernorm -> latent cache
            HeadNormRope,   // k_rope dynamic RoPE -> rope cache
            FlashMlaDecode, // MLA flash (NO DSA gather)
            MlaMergeFold,   // fused latent merge + W_uv fold
            Gemv,           // o_proj
            Residual,       // post-attn residual
            RmsNorm,        // post_attention_layernorm
            Gemv,           // router score GEMV
            MoeRouterTopk,  // router top-k select
            GemvGlu,        // shared expert gate|up
            Gemv,           // shared expert down
        ];
        for _ in 0..top_k {
            ops.push(glu);
            ops.push(down);
        }
        ops.push(MoeCombine);
        ops.into_iter().map(|o| o as u16).collect()
    }

    /// Expected DENSE-block op sequence: shared MLA (12) + block-fp8 SwiGLU (gate/up + down) +
    /// residual. The GLM emitter's dense FFN is block-fp8 regardless of `use_fp8`, so Kimi's dense
    /// layer (layer 0) inherits those opcodes.
    fn kimi_dense_sequence() -> Vec<u16> {
        use DevOp::*;
        vec![
            RmsNorm,
            GemvQkv,
            RmsNorm,
            GemvQkv,
            HeadNormRope,
            RmsNorm,
            HeadNormRope,
            FlashMlaDecode,
            MlaMergeFold,
            Gemv,
            Residual,
            RmsNorm,
            DenseGluFp8Blk,
            GemvFp8Blk,
            Residual,
        ]
        .into_iter()
        .map(|o| o as u16)
        .collect()
    }

    /// A single MoE-layer `--block 1` extraction emits EXACTLY the MLA+MoE block — no embed, no
    /// final-norm/lm_head/argmax tail, `act.x` in and out.
    #[test]
    fn kimi_block_extract_matches_mla_moe_sequence() {
        let c = kimi_ref_cfg();
        assert_eq!(
            block_ops(&c, 512, 1..2, MlaArch::Kimi),
            kimi_moe_sequence(true, 4),
            "single-block --block 1 op sequence != MLA+MoE block (fp8)"
        );
        assert_eq!(
            {
                let (m, _) = glm_build_block(&c, 512, 256, 1..2, false, "kimi-ref", MlaArch::Kimi);
                m.progs[0].insts.iter().map(|d| d.op).collect::<Vec<_>>()
            },
            kimi_moe_sequence(false, 4),
            "bf16 op sequence != MLA+MoE block"
        );
    }

    /// Descriptor for a Kimi MoE block: arch tag, mla_attn+moe_ffn kind, NO dsa_role, MLA+MoE dims,
    /// NO index_* dims, KV latent (ckv/krot) carried state only, decode-only programs.
    #[test]
    fn kimi_block_descriptor_moe() {
        let c = kimi_ref_cfg();
        let (_, d) = glm_build_block(&c, 512, 256, 1..2, true, "kimi-k2.7", MlaArch::Kimi);
        assert_eq!(d.arch, "kimi_mla_moe");
        assert_eq!(d.kind, vec!["mla_attn", "moe_ffn"]);
        assert_eq!(d.dtype, "fp8");
        assert_eq!(d.dsa_role, None, "plain MLA has no DSA indexer role");
        assert_eq!(d.dims.heads, Some(4));
        assert_eq!(d.dims.kv_lora, Some(64));
        assert_eq!(d.dims.q_lora, Some(96));
        assert_eq!(d.dims.n_exp, Some(16));
        assert_eq!(d.dims.top_k, Some(4));
        assert_eq!(d.dims.shared_exp, Some(1));
        assert_eq!(d.dims.moe_inter, Some(128));
        assert_eq!(d.dims.index_heads, None, "no DSA => no index dims");
        assert_eq!(d.dims.index_dim, None);
        assert_eq!(d.dims.index_topk, None);
        assert_eq!(d.layer, 1);
        assert_eq!(d.weights.prefix, "model.layers.1.");
        assert_eq!(d.outputs[0].name, "act.xnext", "odd layer count -> act.xnext");
        // Decode-only unless prefill buckets are asked for; `kimi_prefill_descriptor_lists_buckets`
        // covers the opted-in shape.
        assert!(
            d.programs.prefill_buckets.is_empty(),
            "GLM/Kimi block emit is decode-only unless prefill buckets are requested"
        );
        assert_eq!(d.programs.decode_t, 1);
        // KV latent carried state only — no kidx, no dsa_indices.
        assert_eq!(d.carried_state.len(), 1);
        assert_eq!(d.carried_state[0].role, "kv");
        assert_eq!(d.carried_state[0].layout, "mla_latent");
        assert_eq!(
            d.carried_state[0].tensors,
            vec!["kv.1.ckv", "kv.1.krot"],
            "MLA latent caches only (no indexer kidx)"
        );
    }

    /// Descriptor for a Kimi DENSE block (layer 0, first_k_dense=1): dense_ffn kind, no MoE dims,
    /// MLA dims still present.
    #[test]
    fn kimi_block_descriptor_dense() {
        let c = kimi_ref_cfg();
        let (_, d) = glm_build_block(&c, 512, 256, 0..1, true, "kimi-ref", MlaArch::Kimi);
        assert_eq!(d.kind, vec!["mla_attn", "dense_ffn"]);
        assert_eq!(d.dims.n_exp, None, "dense block has no MoE dims");
        assert_eq!(d.dims.moe_inter, None);
        assert_eq!(d.dims.kv_lora, Some(64), "MLA dims still present");
        assert_eq!(d.dsa_role, None);
    }

    /// A multi-layer `--block 0..2` extraction chains dense layer 0 then MoE layer 1, and the
    /// residual ping-pong lands the output back in `act.x` after an even layer count.
    #[test]
    fn kimi_block_multi_layer_chains() {
        let c = kimi_ref_cfg();
        let mut want = kimi_dense_sequence(); // layer 0 (dense)
        want.extend(kimi_moe_sequence(true, 4)); // layer 1 (MoE)
        assert_eq!(
            block_ops(&c, 512, 0..2, MlaArch::Kimi),
            want,
            "2-layer block != dense++moe"
        );
        let (_, d) = glm_build_block(&c, 512, 256, 0..2, true, "kimi-ref", MlaArch::Kimi);
        assert_eq!(d.outputs[0].name, "act.x", "even layer count -> act.x out");
        assert_eq!(d.layer, 0, "descriptor.layer = block start");
    }

    /// The DSA gate is held OFF at EVERY ctx (has_dsa=false): even at 131072 (well past GLM's 65536
    /// crossover) the block emits FlashMlaDecode — never FlashGatherDecode — and carries no
    /// dsa_indices / no kidx. This is what "reuse GLM MLA without DSA" means structurally.
    #[test]
    fn kimi_no_dsa_at_long_ctx() {
        let c = kimi_ref_cfg();
        let ops = block_ops(&c, 131072, 1..2, MlaArch::Kimi);
        assert!(
            ops.contains(&(DevOp::FlashMlaDecode as u16)),
            "dense MLA flash present"
        );
        assert!(
            !ops.contains(&(DevOp::FlashGatherDecode as u16)),
            "no DSA gather flash for Kimi"
        );
        let (_, d) = glm_build_block(&c, 131072, 256, 1..2, true, "kimi-ref", MlaArch::Kimi);
        assert_eq!(d.dsa_role, None);
        assert!(
            d.carried_state.iter().all(|s| s.role != "dsa_indices"),
            "no dsa_indices carried"
        );
        assert!(
            d.carried_state[0].tensors.iter().all(|t| !t.contains("kidx")),
            "no indexer kidx cache"
        );
    }

    // ===== PREFILL buckets (FLASH_MLA_PREFILL + MLA_MERGE_FOLD at T rows) ======================
    // These are the offline gate on the arm whose ABSENCE meant Kimi K2.7 / DeepSeek / GLM-5.2
    // could decode on gfx950 but could not prefill through their own attention. Same discipline as
    // the decode tests above: no GPU, no weights — pin the emitted stream and the operand fields
    // that select the kernel body, so a hardware run inherits a checked packet rather than a guess.

    /// A cfg with 8 heads, so tp ∈ {1,2,4,8} all divide the head count (the real models have 64).
    /// A TP sweep fixture whose head shard stays EMITTABLE at every `tp` it is used with.
    ///
    /// This was `heads = 8`, and the sweeps below run `tp` up to 8 — so the tp=8 arm had
    /// `nh_l = 1`. The smallest GF the interpreter instantiates is 2, so that arm was emitting a
    /// `FlashMlaPrefill` with `n_grp = nh_l / GF = 0`: `n_work = 0`, the kernel's work loop never
    /// executes, no partial is ever written, and `MlaMergeFold` consumes uninitialised `opart`.
    /// The test asserted the packet's OPERANDS and they were all correct, so it passed for a
    /// packet that computes nothing — the assertion was on `i[1]`, never on whether the shape was
    /// runnable.
    ///
    /// `require_gf_divides` now refuses that shape at emit. 16 heads keeps the sweep's four `tp`
    /// points (nh_l 16/8/4/2) inside what the kernel can express; `nh_l = 1` has its own
    /// `#[should_panic]` test below, because it is a REAL limitation of the current arm set and
    /// not a fixture detail.
    fn kimi_tp_cfg(tp: u32) -> GlmCfg {
        let mut c = kimi_ref_cfg();
        c.heads = 16;
        c.tp = tp;
        c
    }

    /// `heads == tp` gives `nh_l = 1`, and NO instantiated GF can express it.
    ///
    /// Found by `require_gf_divides` firing on a test fixture that had been passing. Both
    /// selectors fall back to GF=2 when nothing fits (`glm_gf`'s `unwrap_or(2)`, `glm_gf_prefill`'s
    /// `else` branch), and 2 > 1, so this was a silent divide-to-zero on BOTH the decode and
    /// prefill paths — not a new restriction, a newly-visible one. A model sharded until each rank
    /// owns a single MLA head must be refused at emit or it produces attention from nothing.
    #[test]
    #[should_panic(expected = "does not divide this rank's head shard")]
    fn a_single_head_per_rank_cannot_be_expressed_by_any_gf() {
        let mut c = kimi_ref_cfg();
        c.heads = 8;
        c.tp = 8; // nh_l = 1
        pf_block(&c, 512, &[128]);
    }

    /// Build one MLA block with prefill buckets and return (model, descriptor).
    fn pf_block(c: &GlmCfg, ctx: u32, pf: &[u32]) -> (Model, plow_asset::BlockDescriptor) {
        glm_build_block_pf(c, ctx, 256, 1..2, true, "kimi-ref", MlaArch::Kimi, pf, PrefillScope::Attn, MoeEnc::Fp8Blk)
    }

    fn find_op(p: &packet::devbuild::Program, op: DevOp) -> &packet::dev::DevInst {
        p.insts
            .iter()
            .find(|d| d.op == op as u16)
            .unwrap_or_else(|| panic!("{op:?} not emitted"))
    }

    /// The co-resident shared gate/up halves must be DISJOINT and cover the slice. Overlap is
    /// silent: the packets still compute the right numbers, they just run one after the other on
    /// the shared workgroups, which is the entire cost `glm_shared_glu_split` exists to remove.
    #[test]
    fn glm_shared_glu_halves_are_disjoint_and_total() {
        for n in [2usize, 3, 8, 32, 224, 256] {
            let cus: Vec<u32> = (0..n as u32).collect();
            let (g, u) = glm_glu_halves(&cus);
            assert!(!g.is_empty() && !u.is_empty(), "n={n}: an empty CU set is not emittable");
            assert!(g.iter().all(|c| !u.contains(c)), "n={n}: halves overlap");
            let mut all: Vec<u32> = g.iter().chain(u.iter()).copied().collect();
            all.sort_unstable();
            assert_eq!(all, cus, "n={n}: the halves must cover the slice exactly");
        }
        // A 1-CU slice cannot be split; the fallback is the serial arrangement, not an empty set.
        assert_eq!(glm_glu_halves(&[7]), (vec![7], vec![7]));
    }

    /// The prefill program IS the MLA attention sub-block at T rows, and it is built from the GEMM
    /// family — not one decode-shaped op survives into it.
    #[test]
    fn mla_prefill_bucket_op_sequence() {
        use DevOp::*;
        let c = kimi_ref_cfg();
        let (m, _) = pf_block(&c, 512, &[128]);
        // Buckets FIRST, decode LAST — manifest.rs and plowrt both key off that order.
        assert_eq!(m.progs.len(), 2, "one prefill bucket + decode");
        assert_eq!(m.prog_t, vec![128, 1]);
        let ops: Vec<u16> = m.progs[0].insts.iter().map(|d| d.op).collect();
        assert_eq!(ops.len(), 15, "norm + 3 down GEMMs + norm + 2 q GEMMs + 2 rope + norm + \
                                   flash + merge-fold + o_proj + residual + norm");
        let tiled = [Gemm as u16, GemmMed as u16, GemmSmall as u16];
        // Positions 1,2,3 (q_a/kv_a/k_rope), 5,6 (q_absorb/q_rope) and 12 (o_proj) are the tiled
        // GEMMs that replace decode's GemvQkv fusions and its o_proj Gemv.
        for i in [1usize, 2, 3, 5, 6, 12] {
            assert!(tiled.contains(&ops[i]), "op {i} = {:?} is not a tiled GEMM arm", ops[i]);
        }
        assert_eq!(
            [ops[0], ops[4], ops[7], ops[8], ops[9], ops[10], ops[11], ops[13], ops[14]],
            [
                RmsNorm as u16,
                RmsNorm as u16,
                HeadNormRope as u16,
                RmsNorm as u16,
                HeadNormRope as u16,
                FlashMlaPrefill as u16,
                MlaMergeFold as u16,
                Residual as u16,
                RmsNorm as u16
            ]
        );
        // No decode-family op may leak into a prefill bucket: those bodies are compiled into the
        // DECODE object, and the AMD dispatch default silently no-ops an opcode with no arm.
        for bad in [
            Gemv, GemvQkv, GemvGlu, GemvFp8Blk, FlashMlaDecode, FlashGatherDecode, MoeRouterTopk,
            MoeExpertGluFp8Blk, MoeCombine, DenseGluFp8Blk,
        ] {
            assert!(!ops.contains(&(bad as u16)), "{bad:?} leaked into the prefill bucket");
        }
    }

    /// The flash + fold operand fields, which are what select the kernel body and its work
    /// decomposition. `i[4] = n_tok` (not nsplit) and `nsplit = 1` are the prefill PRECONDITION:
    /// under a per-token causal bound an early token's later splits are empty and an empty split
    /// emits l=0 for the merge to divide by.
    #[test]
    fn mla_prefill_flash_and_fold_operands() {
        let c = kimi_ref_cfg();
        let (m, _) = pf_block(&c, 512, &[128]);
        let p = &m.progs[0];
        let fl = find_op(p, DevOp::FlashMlaPrefill);
        assert_eq!(fl.i[0], 1, "n_batch: one sequence per prefill chunk");
        assert_eq!(fl.i[1], c.heads, "n_head = per-rank head count (tp=1 => all)");
        assert_eq!(fl.i[2], 512, "kv_stride = ctx");
        assert_eq!(fl.i[3], 0, "window 0 = full causal");
        assert_eq!(fl.i[4], 128, "i[4] carries n_tok on the prefill arm, not nsplit");
        assert_eq!(fl.i[5], KV_MASK_NONE);
        assert!(matches!(fl.i[7], 2 | 4), "GF must be an instantiated prefill body, got {}", fl.i[7]);
        assert_eq!(fl.t[6], c_kvlen(&m), "kv_len operand bound");
        let fold = find_op(p, DevOp::MlaMergeFold);
        assert_eq!(fold.i[0], 128, "the token axis folds into n_batch: partials are (b*n_tok+t)");
        assert_eq!(fold.i[1], c.heads);
        assert_eq!(fold.i[2], c.v_head);
        assert_eq!(fold.i[4], 1, "nsplit MUST be 1 on the prefill arm");
    }

    /// `in.kvlen`'s handle, so the operand check above is against the real tensor, not an index.
    fn c_kvlen(m: &Model) -> u32 {
        m.tensors.iter().position(|t| t.name == "in.kvlen").expect("in.kvlen declared") as u32
    }

    /// EVERY head-dimensioned prefill field is the PER-RANK count nh_l = n_head/tp. Sizing any of
    /// them from the global head count is the measured tp=8 bug the `glm_nsplit` header records —
    /// the flash ran on 32 of 256 CUs — and prefill has strictly more work items than decode, so it
    /// would be just as invisible here.
    #[test]
    fn mla_prefill_tp_shapes_scale_with_tp() {
        for tp in [1u32, 2, 4, 8] {
            let c = kimi_tp_cfg(tp);
            let nh_l = c.heads / tp;
            let (m, _) = pf_block(&c, 512, &[128]);
            let p = &m.progs[0];
            assert_eq!(find_op(p, DevOp::FlashMlaPrefill).i[1], nh_l, "tp={tp} flash n_head");
            assert_eq!(find_op(p, DevOp::MlaMergeFold).i[1], nh_l, "tp={tp} fold n_head");
            // o_proj is row-parallel: K = this rank's nh_l*v_head lanes, N = full hidden.
            let o = p
                .insts
                .iter()
                .rev()
                .find(|d| matches!(d.op, x if x == DevOp::Gemm as u16
                                        || x == DevOp::GemmMed as u16
                                        || x == DevOp::GemmSmall as u16))
                .expect("o_proj GEMM");
            assert_eq!(o.i[1], c.hidden, "tp={tp} o_proj N = full hidden");
            assert_eq!(o.i[2], nh_l * c.v_head, "tp={tp} o_proj K = per-rank head lanes");
            // The q projections are column-parallel by head.
            let qa = &p.insts[5];
            assert_eq!(qa.i[1], nh_l * c.kv_lora, "tp={tp} q_absorb N");
        }
    }

    /// PREFILL all-reduces the [T,hidden] o_proj partial with the TWO-SHOT collective, not decode's
    /// one-shot: the partial is bandwidth-bound at T rows, so the two-shot moves ~tp/2x less over
    /// the fabric (plans/tp-prefill.md §4). tp=1 emits no collective at all.
    #[test]
    fn mla_prefill_tp_emits_two_shot_allreduce() {
        let (m1, _) = pf_block(&kimi_tp_cfg(1), 512, &[128]);
        assert!(
            !m1.progs[0].insts.iter().any(|d| d.op == DevOp::XReduceTwoShot as u16
                || d.op == DevOp::XReduce as u16),
            "tp=1 emits no collective"
        );
        for tp in [2u32, 4, 8] {
            let c = kimi_tp_cfg(tp);
            let (m, _) = pf_block(&c, 512, &[128]);
            let p = &m.progs[0];
            let xr = find_op(p, DevOp::XReduceTwoShot);
            assert_eq!(xr.i[0], 128 * c.hidden, "tp={tp} reduces t*hidden elements");
            assert_eq!(xr.i[1], tp, "tp={tp} n_gpu");
            assert!(
                !p.insts.iter().any(|d| d.op == DevOp::XReduce as u16),
                "tp={tp}: prefill must not use decode's one-shot all-reduce"
            );
        }
    }

    /// EP (expert-parallel) survives on the DECODE half that the prefill buckets sit alongside:
    /// attention stays TP-sharded (nh_l above) while the ROUTED experts are distributed WHOLE
    /// across ranks — full `moe_inter` per expert, not the TP slice — so a rank never runs a
    /// CU-starved fragment of an expert.
    #[test]
    fn mla_ep_keeps_routed_experts_whole_beside_prefill() {
        let mut c = kimi_tp_cfg(4);
        c.ep = true;
        let (m, _) = pf_block(&c, 512, &[128]);
        let dec = m.progs.last().unwrap();
        let glu = find_op(dec, DevOp::MoeExpertGluFp8Blk);
        assert_eq!(glu.i[1], c.moe_inter, "EP: routed expert keeps the FULL moe_inter");
        // The SHARED expert stays TP-sharded — that is the floor EP deliberately does not touch.
        let sh = find_op(dec, DevOp::GemvGlu);
        assert_eq!(sh.i[1], c.moe_inter / 4, "shared expert stays TP-sharded under EP");
        // And the attention half of the SAME asset is still per-rank sharded.
        assert_eq!(find_op(&m.progs[0], DevOp::FlashMlaPrefill).i[1], c.heads / 4);

        let mut c_tp = c.clone();
        c_tp.ep = false;
        let (m2, _) = pf_block(&c_tp, 512, &[128]);
        assert_eq!(
            find_op(m2.progs.last().unwrap(), DevOp::MoeExpertGluFp8Blk).i[1],
            c.moe_inter / 4,
            "without EP the routed expert IS TP-sliced"
        );
    }

    /// One tensor table serves every program, so the row-dimensioned activations are sized for the
    /// widest bucket. Under-sizing them is an out-of-bounds DEVICE write, not a slowdown.
    #[test]
    fn mla_prefill_widens_the_shared_tensor_table() {
        let c = kimi_ref_cfg();
        let bytes = |m: &Model, n: &str| {
            m.tensors.iter().find(|t| t.name == n).unwrap_or_else(|| panic!("{n}")).bytes
        };
        let (dec_only, _) = pf_block(&c, 512, &[]);
        let (with_pf, _) = pf_block(&c, 512, &[128, 512]);
        let h = c.hidden as u64;
        assert_eq!(bytes(&dec_only, "act.x"), h * 2, "decode-only: one row");
        assert_eq!(bytes(&with_pf, "act.x"), 512 * h * 2, "sized for the WIDEST bucket");
        assert_eq!(bytes(&with_pf, "act.xn2"), 512 * h * 2);
        assert_eq!(bytes(&with_pf, "act.oat"), 512 * (c.heads * c.v_head) as u64 * 2);
        // The flash partials are [t][head][nsplit][DK] with nsplit=1 at prefill and ns at decode —
        // the MAX of the two, not their product.
        let ns = glm_nsplit(512, c.heads);
        assert_eq!(
            bytes(&with_pf, "act.opart"),
            (c.heads * 512.max(ns) * c.kv_lora) as u64 * 4
        );
        // The MoE partials are [T*k, H] f32 — the grouped prefill FFN scatters into them.
        assert_eq!(bytes(&with_pf, "act.part"), 512 * (c.top_k * c.hidden) as u64 * 4);
        assert_eq!(bytes(&dec_only, "act.part"), (c.top_k * c.hidden) as u64 * 4, "decode: one row");
        // The DECODE per-slot gate/up buffer stays one row — the grouped path uses moe_fug instead.
        assert_eq!(bytes(&with_pf, "act.fu"), (c.top_k * c.moe_inter) as u64 * 2);
    }

    /// The descriptor reports the buckets it actually emitted.
    #[test]
    fn kimi_prefill_descriptor_lists_buckets() {
        let c = kimi_ref_cfg();
        let (_, d) = pf_block(&c, 512, &[128, 512]);
        assert_eq!(d.programs.prefill_buckets, vec![128, 512]);
        assert_eq!(d.programs.decode_t, 1);
    }

    /// The bucket ladder is capped at ctx — a rung above the compiled context can never be invoked.
    #[test]
    fn mla_prefill_bucket_ladder_is_ctx_capped() {
        assert_eq!(glm_prefill_buckets(512), vec![128, 512]);
        assert_eq!(glm_prefill_buckets(100), Vec::<u32>::new());
        assert_eq!(
            glm_prefill_buckets(1 << 20),
            vec![128, 512, 1024, 2048, 4096, 8192]
        );
    }

    /// A multi-layer extraction cannot carry prefill: the program ends at the post-attention norm,
    /// so there is no residual stream for layer l+1 to read. It must refuse, not emit a broken chain.
    #[test]
    #[should_panic(expected = "single-layer")]
    fn mla_prefill_refuses_a_multi_layer_block() {
        let c = kimi_ref_cfg();
        glm_build_block_pf(&c, 512, 256, 0..2, true, "kimi-ref", MlaArch::Kimi, &[128], PrefillScope::Attn, MoeEnc::Fp8Blk);
    }

    /// The attention-only scope emits the verified flash-prefill arm and stops there.
    #[test]
    fn mla_prefill_attn_scope_still_emits() {
        let (m, _) = pf_block(&kimi_ref_cfg(), 512, &[128]);
        assert!(m.progs[0].insts.iter().any(|d| d.op == DevOp::FlashMlaPrefill as u16));
    }

    // ===== WHOLE-LAYER prefill: MLA attention + token-sorted grouped MoE FFN (ops 83-87) ========

    fn pf_full_enc(c: &GlmCfg, ctx: u32, pf: &[u32], block: std::ops::Range<usize>, enc: MoeEnc) -> Model {
        glm_build_block_pf(c, ctx, 256, block, true, "kimi-ref", MlaArch::Kimi, pf, PrefillScope::Full, enc).0
    }

    fn pf_full(c: &GlmCfg, ctx: u32, pf: &[u32], block: std::ops::Range<usize>) -> Model {
        pf_full_enc(c, ctx, pf, block, MoeEnc::Fp8Blk)
    }

    /// The whole-layer prefill bucket, op for op. The FFN is the GROUPED path — not the decode
    /// per-slot ops with a row loop — so no `MoeExpertGlu*`/`MoeCombine` may appear.
    #[test]
    fn mla_full_prefill_bucket_op_sequence() {
        use DevOp::*;
        let m = pf_full(&kimi_ref_cfg(), 512, &[128], 1..2);
        let ops: Vec<u16> = m.progs[0].insts.iter().map(|d| d.op).collect();
        // 15 attention ops, then: router score GEMM, router top-k tail, align, shared GLU,
        // shared down, grouped glu, grouped down, combine = 23.
        assert_eq!(ops.len(), 23, "attention (15) + MoE FFN (8)");
        assert_eq!(
            [ops[16], ops[17], ops[18], ops[20], ops[21], ops[22]],
            [
                MoeRouterTopkPf as u16, // router top-k tail (15 = the [T,n_exp] score GEMM)
                MoeAlignPf as u16,      // token-sort / MPF_BM-padded prefix
                GemmGlu as u16,         // shared expert gate|up (19 = its down GEMM)
                MoeGroupGluPf as u16,   // grouped gate/up over the sorted rows
                MoeGroupDownPf as u16,
                MoeCombinePf as u16
            ]
        );
        assert!(ops.contains(&(FlashMlaPrefill as u16)), "attention half still present");
        for bad in [MoeExpertGluFp8Blk, MoeExpertDownFp8Blk, MoeCombine, MoeRouterTopk, GemvGlu, Gemv] {
            assert!(!ops.contains(&(bad as u16)), "{bad:?} is a DECODE op; it must not appear");
        }
    }

    /// The grouped-FFN operand fields, including the ones a wrong value would make silently wrong:
    /// `n_exp` on both grouped GEMMs (the table indirection), `T` on the router tail and combine.
    #[test]
    fn mla_full_prefill_moe_operands() {
        let c = kimi_k27_code_cfg(4);
        let m = pf_full(&c, 1024, &[256], 1..2);
        let p = &m.progs[0];
        let rt = find_op(p, DevOp::MoeRouterTopkPf);
        assert_eq!((rt.i[1], rt.i[2], rt.i[4]), (384, 8, 256), "n_exp, k, T");
        assert_eq!(rt.i[3], GLM_ROUTER_FLAGS);
        let al = find_op(p, DevOp::MoeAlignPf);
        assert_eq!((al.i[0], al.i[1], al.i[2]), (256, 384, 8), "T, n_exp, k");
        assert_eq!(al.blocks, 1, "align is a single-workgroup global scan");
        let g = find_op(p, DevOp::MoeGroupGluPf);
        assert_eq!((g.i[0], g.i[1], g.i[2], g.i[3]), (c.moe_inter, c.hidden, 384, 1),
                   "I_moe (EP: whole), H, n_exp, fp8");
        let dn = find_op(p, DevOp::MoeGroupDownPf);
        assert_eq!((dn.i[0], dn.i[1], dn.i[2]), (c.hidden, c.moe_inter, 384));
        let cb = find_op(p, DevOp::MoeCombinePf);
        assert_eq!((cb.i[0], cb.i[1], cb.i[2]), (c.hidden, 8, 256), "H, k, T");
    }

    /// The gathered arrays are sized on the MPF_BM-PADDED bound `T*k + n_exp*(MPF_BM-1)`, not `T*k`.
    /// Sizing them from `T*k` is an out-of-bounds device write that hides at small expert counts and
    /// is guaranteed at 384: the padding alone is 384*63 = 24192 rows.
    #[test]
    fn mla_full_prefill_pads_the_gathered_rows() {
        let c = kimi_k27_code_cfg(4);
        let m = pf_full(&c, 1024, &[256], 1..2);
        let bytes = |n: &str| m.tensors.iter().find(|t| t.name == n).unwrap().bytes;
        let pad = 256u64 * 8 + 384 * (MPF_BM as u64 - 1);
        assert_eq!(pad, 2048 + 24192);
        assert_eq!(bytes("act.moe_rowtok"), pad * 4);
        assert_eq!(bytes("act.moe_rowpart"), pad * 4);
        assert_eq!(bytes("act.moe_rowgate"), pad * 4);
        assert_eq!(bytes("act.moe_fug"), pad * c.moe_inter as u64 * 2, "EP: full moe_inter");
        assert_eq!(bytes("act.moe_meta"), (3 * 384 + 1) as u64 * 4);
        // part is [T*k, H] f32 — no padding, the down op scatters by row_partidx.
        assert_eq!(bytes("act.part"), 256 * (8 * c.hidden) as u64 * 4);
    }

    /// Whole-layer prefill CHAINS across layers (unlike the attention-only scope), because the
    /// combine produces a real residual stream.
    #[test]
    fn mla_full_prefill_chains_multiple_layers() {
        let mut c = kimi_ref_cfg();
        c.first_k_dense = 0; // every layer MoE, so a 2-layer block is all-prefillable
        let m = pf_full(&c, 512, &[128], 1..3);
        let ops: Vec<u16> = m.progs[0].insts.iter().map(|d| d.op).collect();
        assert_eq!(ops.len(), 46, "two whole layers");
        assert_eq!(ops.iter().filter(|&&o| o == DevOp::FlashMlaPrefill as u16).count(), 2);
        assert_eq!(ops.iter().filter(|&&o| o == DevOp::MoeCombinePf as u16).count(), 2);
    }

    /// TP: the shared expert and the routed partials all-reduce through the TWO-SHOT collective at
    /// `t*hidden`, and the combine writes into the peer slot rather than onto the residual (which
    /// XReduce would otherwise sum tp times).
    #[test]
    fn mla_full_prefill_tp_combine_is_a_partial() {
        let c = kimi_k27_code_cfg(4);
        let m = pf_full(&c, 1024, &[256], 1..2);
        let p = &m.progs[0];
        let xrs: Vec<_> = p.insts.iter().filter(|d| d.op == DevOp::XReduceTwoShot as u16).collect();
        assert_eq!(xrs.len(), 2, "one for o_proj, one for the FFN combine");
        for x in &xrs {
            assert_eq!(x.i[0], 256 * c.hidden, "reduces t*hidden");
            assert_eq!(x.i[1], 4);
        }
        assert_ne!(xrs[0].i[2], xrs[1].i[2], "the two partials occupy DIFFERENT peer slots");
        assert_eq!(xrs[1].i[2], 256 * c.hidden * 2, "FFN partial sits past the o_proj partial");
        // The combine's residual is the ZERO buffer, not xmid — xmid is added after the all-reduce.
        let zero = m.tensors.iter().position(|t| t.name == "act.zero_h").unwrap() as u32;
        assert_eq!(find_op(p, DevOp::MoeCombinePf).t[1], zero);
    }

    /// The manifest is what pairs a packet with an object, so a whole-layer bucket must declare the
    /// MoE prefill axis — `interp_prefill_mla` alone would hit `default:` on ops 83-87 and write
    /// nothing.
    #[test]
    fn mla_full_prefill_declares_the_moe_prefill_axis() {
        let m = pf_full(&kimi_k27_code_cfg(4), 1024, &[256], 1..2);
        let man = crate::manifest::build(&m, "gfx950", &crate::LeanReport::skipped("test: gate not run"));
        assert_eq!(man["features"]["moe_prefill"], true);
        assert_eq!(man["features"]["prefill"], true);
        let ops: Vec<&str> =
            man["opcodes"].as_array().unwrap().iter().filter_map(|v| v.as_str()).collect();
        for o in ["MoeRouterTopkPf", "MoeAlignPf", "MoeGroupGluPf", "MoeGroupDownPf", "MoeCombinePf"] {
            assert!(ops.contains(&o), "{o} missing from the manifest");
        }
    }

    // ===== GROUP-LIMITED ROUTING (DeepSeek noaux_tc) ===========================================

    /// Both router tails must carry the group operands, and they must AGREE. The prefill tail is the
    /// decode tail under a token loop, so an emitter that set them on one and not the other would
    /// route the same token to different experts depending on which program ran it — the exact
    /// class of prefill/decode divergence that reads as a model bug rather than a compiler one.
    #[test]
    fn mla_group_routing_reaches_both_router_tails() {
        let mut c = kimi_k27_code_cfg(4);
        c.n_group = 8;
        c.topk_group = 4;
        let m = pf_full(&c, 1024, &[256], 1..2);
        let pf = find_op(&m.progs[0], DevOp::MoeRouterTopkPf);
        assert_eq!((pf.i[6], pf.i[7]), (8, 4), "prefill tail carries n_group/topk_group");
        let dec = find_op(m.progs.last().unwrap(), DevOp::MoeRouterTopk);
        assert_eq!((dec.i[6], dec.i[7]), (8, 4), "decode tail carries the SAME pair");
        assert_eq!((pf.i[1], pf.i[2]), (dec.i[1], dec.i[2]), "n_exp/k agree too");
    }

    /// At `n_group <= 1` the rule is the identity, and the emitter must still say so explicitly —
    /// the kernel treats 1 as inert, so every GLM / Qwen / Mixtral packet stays bit-identical.
    #[test]
    fn mla_group_routing_is_inert_for_ungrouped_models() {
        let c = kimi_ref_cfg(); // n_group = 1
        let (m, _) = pf_block(&c, 512, &[]);
        let dec = find_op(m.progs.last().unwrap(), DevOp::MoeRouterTopk);
        assert_eq!((dec.i[6], dec.i[7]), (1, 1), "ungrouped => identity operands");
    }

    // ===== A4W4 (MXFP4 on both operands) for the grouped expert path ===========================

    /// `i[3]` selects the encoding, and A4W4 binds the two extra operands the fused bridge needs:
    /// `t7` on GLU is the E8M0 rows it WRITES (the bridge is its epilogue), `t5` on DOWN is the same
    /// rows READ back. Getting either wrong means the intermediate silently loses its scales.
    #[test]
    fn mla_a4w4_expert_path_binds_the_bridge_operands() {
        let c = kimi_k27_code_cfg(4);
        let m = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
        let p = &m.progs[0];
        let scale = m.tensors.iter().position(|t| t.name == "act.moe_fuscale").unwrap() as u32;
        let rowpart = m.tensors.iter().position(|t| t.name == "act.moe_rowpart").unwrap() as u32;
        let g = find_op(p, DevOp::MoeGroupGluPf);
        assert_eq!(g.i[3], 2, "i[3]=2 selects the MXFP4 body");
        assert_eq!(g.t[7], scale, "GLU writes the E8M0 rows (its epilogue IS the bridge)");
        assert_eq!(g.t[6], rowpart, "GLU needs row_partidx so the bridge skips PAD rows");
        let d = find_op(p, DevOp::MoeGroupDownPf);
        assert_eq!(d.i[3], 2);
        assert_eq!(d.t[5], scale, "DOWN reads the same E8M0 rows back");
        assert_eq!(d.t[1], g.t[0], "DOWN's A operand IS the bridge's fp4 output");
    }

    /// THE ENCODING SLOT IS NOT THE SAME ON BOTH PHASES, and this test exists because getting it
    /// wrong is silent.
    ///
    /// Prefill ops 85/86 carry `n_exp` in `i[2]`, so the encoding took `i[3]`. Decode ops
    /// 45/46/48/49 predate the field and already use `i[3]` for `n_exp`, so theirs is `i[6]`.
    /// Writing the encoding into `i[3]` on a DECODE op sets `n_exp = 2`; every expert id >= 2 then
    /// hits `if (eid >= n_exp) return;` and the op writes nothing at all. Combined with the AMD
    /// dispatch default, which also writes nothing, the result is a layer that emits ZEROS with no
    /// fault and no diagnostic — a dead MoE behind fluent-looking output.
    ///
    /// So: pin the slot per op, and pin that `n_exp` still lands where the kernel reads it. Either
    /// assertion alone would miss the failure; the pair is what makes it impossible.
    #[test]
    fn mla_encoding_slot_differs_between_decode_and_prefill() {
        let c = kimi_k27_code_cfg(4);
        let m = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
        let (pf, dec) = (&m.progs[0], m.progs.last().unwrap());

        // PREFILL: encoding in i[3], n_exp in i[2].
        for op in [DevOp::MoeGroupGluPf, DevOp::MoeGroupDownPf] {
            let x = find_op(pf, op);
            assert_eq!(x.i[MoeEnc::PREFILL_SLOT], 2, "{op:?}: encoding is i[3] on prefill");
            assert_eq!(x.i[2], c.n_exp, "{op:?}: n_exp stays in i[2]");
        }
        // DECODE: encoding in i[6], n_exp in i[3]. If these two ever swap, n_exp becomes 2.
        for op in [DevOp::MoeExpertGluFp8Blk, DevOp::MoeExpertDownFp8Blk] {
            let x = find_op(dec, op);
            assert_eq!(x.i[MoeEnc::DECODE_SLOT], 2, "{op:?}: encoding is i[6] on decode");
            assert_eq!(x.i[3], c.n_exp, "{op:?}: i[3] is n_exp — writing the encoding here kills it");
            assert_ne!(x.i[3], 2, "n_exp=2 is the silent-zeros signature");
        }
        assert_ne!(MoeEnc::PREFILL_SLOT, MoeEnc::DECODE_SLOT, "the slots differ, deliberately");
    }

    /// The grouped decode pair (48/49) takes the encoding in the same slot as the per-slot pair.
    #[test]
    fn mla_encoding_slot_on_the_grouped_decode_ops() {
        let mut c = kimi_k27_code_cfg(4);
        c.group = true; // collapses the 2*top_k per-slot packets into ops 48/49
        let m = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
        let dec = m.progs.last().unwrap();
        for op in [DevOp::MoeGroupGluFp8Blk, DevOp::MoeGroupDownFp8Blk] {
            let x = find_op(dec, op);
            assert_eq!(x.i[MoeEnc::DECODE_SLOT], 2, "{op:?}: encoding is i[6]");
            assert_eq!(x.i[3], c.n_exp, "{op:?}: i[3] is n_exp");
        }
    }

    /// On the EXPERT path a precision change is a field change: the same opcodes, one operand
    /// different. That is the property the kernel side bought by making the encoding `i[3]`/`i[6]`
    /// instead of new opcodes, and it is worth pinning because it is what makes an A/B across
    /// encodings a controlled comparison rather than two different programs.
    ///
    /// It does NOT extend to the PROJECTIONS, and the test says so rather than pretending: w4a16 is
    /// a genuinely different kernel family (`GemmMxfp4` reuses the bf16 wide-K MFMA but fetches fp4
    /// with the MX scale folded into the convert), so those change opcode. Precision is a field on
    /// the experts and an opcode on the projections; asserting the stronger claim everywhere would
    /// have been false.
    #[test]
    fn mla_encoding_is_a_field_on_the_expert_path() {
        let c = kimi_k27_code_cfg(4);
        let ops = |enc| -> Vec<u16> {
            pf_full_enc(&c, 1024, &[256], 1..2, enc).progs[0].insts.iter().map(|d| d.op).collect()
        };
        assert_eq!(ops(MoeEnc::Bf16), ops(MoeEnc::Fp8Blk), "bf16 vs block-fp8: same stream");
        // Same expert opcodes under all three; only i[3] moves.
        for (enc, code) in [(MoeEnc::Bf16, 0), (MoeEnc::Fp8Blk, 1), (MoeEnc::Mxfp4, 2)] {
            let m = pf_full_enc(&c, 1024, &[256], 1..2, enc);
            assert_eq!(find_op(&m.progs[0], DevOp::MoeGroupGluPf).i[3], code);
            assert_eq!(find_op(&m.progs[0], DevOp::MoeGroupDownPf).i[3], code);
            // ... and on decode, in the OTHER slot. bf16 rides its own opcodes (41/42) rather
            // than the scale-table-carrying pair, so look at whichever the encoding selects.
            let dop = if enc == MoeEnc::Bf16 {
                DevOp::MoeExpertGlu
            } else {
                DevOp::MoeExpertGluFp8Blk
            };
            assert_eq!(find_op(m.progs.last().unwrap(), dop).i[6], code, "decode slot i[6]");
        }
    }

    /// An all-MXFP4 packet: EVERY matmul weight consumer, in BOTH programs, on an MXFP4 arm — and
    /// nothing left on a block-fp8 one. This is the whole point of the encoding work, and the check
    /// is by absence as much as by presence: a single surviving `GemvFp8Blk` or `Gemv` would be a
    /// mixed run reported as an MXFP4 one.
    #[test]
    fn mla_all_mxfp4_packet_has_no_other_encoding() {
        use DevOp::*;
        let c = kimi_k27_code_cfg(4);
        let m = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
        let pf: Vec<u16> = m.progs[0].insts.iter().map(|d| d.op).collect();
        let dec: Vec<u16> = m.progs.last().unwrap().insts.iter().map(|d| d.op).collect();

        // PREFILL: projections are on SOME mxfp4 tile rung, the shared GLU unfused into two of
        // them plus a Glu. Which rung is a per-shape decision now — pinning `GemmMxfp4` here
        // would re-assert the very thing the T3 fix removed, namely that every fp4 prefill GEMM
        // takes the 256x256 tile whatever its shape. What must hold is that the ENCODING is
        // uniform, which is what the absence check below states.
        const MXFP4_TILES: [DevOp; 5] =
            [GemmMxfp4, GemmMedMxfp4, GemmSmallMxfp4, GemmWideMxfp4, GemmC5Mxfp4];
        assert!(
            MXFP4_TILES.iter().any(|t| pf.contains(&(*t as u16))),
            "no mxfp4 prefill GEMM at all; stream = {pf:?}"
        );
        assert!(pf.contains(&(Glu as u16)), "no GemmGluMxfp4 => explicit Glu");
        // DECODE: projections are GemvMxfp4, the shared GLU IS fused (op 92 exists at decode).
        assert!(dec.contains(&(GemvMxfp4 as u16)));
        assert!(dec.contains(&(GemvGluMxfp4 as u16)));

        // Nothing may remain on a bf16 or block-fp8 matmul arm, in EITHER program.
        for (name, ops) in [("prefill", &pf), ("decode", &dec)] {
            for bad in [
                Gemv, GemvGlu, GemvQkv, GemvFp8Blk, DenseGluFp8Blk, Gemm, GemmMed, GemmSmall,
                GemmWide, GemmC5, GemmGlu, GemmFp8, GemmMedFp8, GemmSmallFp8, GemmWideFp8,
                GemmC5Fp8, GemmGluFp8,
            ] {
                assert!(
                    !ops.contains(&(bad as u16)),
                    "{name}: {bad:?} survived into an all-MXFP4 packet"
                );
            }
        }
        // The expert path carries the encoding in its two phase-dependent slots.
        assert_eq!(find_op(&m.progs[0], MoeGroupGluPf).i[MoeEnc::PREFILL_SLOT], 2);
        assert_eq!(
            find_op(m.progs.last().unwrap(), MoeExpertGluFp8Blk).i[MoeEnc::DECODE_SLOT],
            2
        );
    }

    /// MXFP4 weights are packed at half a byte with one E8M0 byte per 32 — and the scale handle
    /// must be BOUND, not merely declared. A packed weight whose scale operand is TENSOR_NONE is a
    /// null pointer in the kernel.
    #[test]
    fn mla_mxfp4_weights_are_packed_and_their_scales_bound() {
        let c = kimi_k27_code_cfg(4);
        let mx = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
        let bf = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Fp8Blk);
        let bytes = |m: &Model, n: &str| m.tensors.iter().find(|t| t.name == n).map(|t| t.bytes);
        let nm = "model.layers.1.self_attn.o_proj.weight";
        let (n, k) = (c.hidden as u64, (c.heads / 4 * c.v_head) as u64);
        assert_eq!(bytes(&bf, nm), Some(n * k * 2), "bf16: 2 B/elt");
        assert_eq!(bytes(&mx, nm), Some(n * k / 2), "packed fp4: half a byte");
        assert_eq!(bytes(&mx, &format!("{nm}_scale")), Some(n * k / MX_BLOCK as u64));
        assert_eq!(bytes(&bf, &format!("{nm}_scale")), None, "no E8M0 rows off the MXFP4 arm");
        // Every MXFP4 projection op must carry a real scale handle in t3.
        for p in &mx.progs {
            for i in p.insts.iter().filter(|d| {
                d.op == DevOp::GemvMxfp4 as u16 || d.op == DevOp::GemmMxfp4 as u16
            }) {
                assert_ne!(i.t[3], TENSOR_NONE, "MXFP4 projection with an unbound E8M0 scale");
            }
        }
    }

    /// The ONE bf16 tensor left under MXFP4, declared as an exception rather than hidden. It is
    /// safe precisely because it is DERIVED (weight-prep folds kv_b_proj, so a bf16 copy exists
    /// whatever the checkpoint stores) — unlike the expert weights, where fp4 bytes read as bf16
    /// would be noise.
    #[test]
    fn mla_mxfp4_wuv_is_the_declared_exception() {
        let c = kimi_k27_code_cfg(4);
        let mx = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
        let nm = "model.layers.1.self_attn.derived.v_absorb.weight";
        let w = mx.tensors.iter().find(|t| t.name == nm).unwrap();
        let (nh_l, dk, vd) = (c.heads / 4, c.kv_lora, c.v_head);
        assert_eq!(w.bytes, (nh_l * dk * vd) as u64 * 2, "W_uv stays bf16 under MXFP4");
        assert!(mx.tensors.iter().all(|t| t.name != format!("{nm}_scale")));
        assert!(mxfp4_bf16_exceptions().contains(&"MlaMergeFold/Wuv"));
        // How much of the model this is, at the REAL Kimi K2.7 geometry rather than the scaled
        // fixture — the number a dtype comparison has to be able to quote instead of guess.
        // W_uv = n_head*kv_lora*v_head; experts = n_exp*3*moe_inter*hidden.
        let wuv_real: u64 = 64 * 512 * 128;
        let experts_real: u64 = 384 * 3 * 2048 * 7168;
        assert_eq!(wuv_real, 4_194_304);
        assert!(
            wuv_real * 4000 < experts_real,
            "W_uv is {:.4}% of one layer's expert weights",
            wuv_real as f64 * 100.0 / experts_real as f64
        );
    }

    /// The manifest must not call an MXFP4 packet fp8 just because the expert opcodes still carry
    /// fp8-era NAMES. Once the encoding became a runtime field, `MoeExpertGluFp8Blk` with `i[6]=2`
    /// is an MXFP4 instruction — the name stopped being a fact and became a label.
    #[test]
    fn mla_manifest_does_not_call_an_mxfp4_packet_fp8() {
        let c = kimi_k27_code_cfg(4);
        let mx = crate::manifest::build(&pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4), "gfx950", &crate::LeanReport::skipped("test: gate not run"));
        assert_eq!(mx["shapes"]["moe_enc"], serde_json::json!([2]), "ONE encoding");
        assert_eq!(mx["features"]["moe_enc_mixed"], false);
        assert_eq!(mx["features"]["a4w4"], true);
        assert_eq!(mx["features"]["mxfp4_weights"], true);
        assert_eq!(mx["features"]["fp8_weights"], false, "no fp8 weight anywhere in this packet");
        // The block-fp8 packet is still reported as fp8 — the correction must not overreach.
        let fp8 = crate::manifest::build(&pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Fp8Blk), "gfx950", &crate::LeanReport::skipped("test: gate not run"));
        assert_eq!(fp8["shapes"]["moe_enc"], serde_json::json!([1]));
        assert_eq!(fp8["features"]["fp8_weights"], true);
        assert_eq!(fp8["features"]["a4w4"], false);
    }

    /// PLOW_MXFP4=1 and PLOW_FP8=1 together ask for two encodings in one packet.
    #[test]
    fn mla_two_encodings_at_once_is_refused() {
        assert_eq!(MoeEnc::from_flags(true, true), MoeEnc::Mxfp4, "mxfp4 wins the enum");
        // The env-level guard is what actually refuses; exercised through the CLI.
    }

    /// A4W4 halves the gathered intermediate and adds one E8M0 byte per 32 values. The scale rows
    /// have no bf16 counterpart, so they must be DECLARED or the bridge writes to a null handle.
    #[test]
    fn mla_a4w4_sizes_the_packed_intermediate() {
        let c = kimi_k27_code_cfg(4);
        let pad = 256u64 * 8 + 384 * (MPF_BM as u64 - 1);
        let bytes = |m: &Model, n: &str| m.tensors.iter().find(|t| t.name == n).map(|t| t.bytes);
        let fp8 = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Fp8Blk);
        let mx = pf_full_enc(&c, 1024, &[256], 1..2, MoeEnc::Mxfp4);
        assert_eq!(bytes(&fp8, "act.moe_fug"), Some(pad * c.moe_inter as u64 * 2));
        assert_eq!(bytes(&mx, "act.moe_fug"), Some(pad * (c.moe_inter / 2) as u64), "packed fp4");
        assert_eq!(bytes(&mx, "act.moe_fuscale"), Some(pad * (c.moe_inter / MX_BLOCK) as u64));
        assert_eq!(bytes(&fp8, "act.moe_fuscale"), None, "no E8M0 rows on the block-fp8 arm");
    }

    /// bf16 and block-fp8 emission must be bit-identical to before the encoding field existed.
    #[test]
    fn mla_default_encoding_is_block_fp8_and_unchanged() {
        let c = kimi_ref_cfg();
        let (m, _) = pf_block(&c, 512, &[128]);
        let dec = m.progs.last().unwrap();
        // The decode expert ops are untouched by the encoding work — they have no such field.
        assert!(dec.insts.iter().any(|d| d.op == DevOp::MoeExpertGluFp8Blk as u16));
        assert_eq!(MoeEnc::from_flags(true, false), MoeEnc::Fp8Blk);
        assert_eq!(MoeEnc::from_flags(false, false), MoeEnc::Bf16);
        assert_eq!(MoeEnc::from_flags(false, true), MoeEnc::Mxfp4);
        assert_eq!((MoeEnc::Bf16.code(), MoeEnc::Fp8Blk.code(), MoeEnc::Mxfp4.code()), (0, 1, 2));
    }

    /// 384 experts is inside the grouped prefill's LDS bound; past it the align histogram and the
    /// router key array would overrun the shared arena with nothing on device to notice.
    #[test]
    #[should_panic(expected = "exceeds the grouped MoE prefill LDS bound")]
    fn mla_full_prefill_bounds_the_expert_count() {
        let mut c = kimi_k27_code_cfg(4);
        assert_eq!(c.n_exp, 384, "Kimi K2.7 routes 384 — inside the bound");
        c.n_exp = 1024;
        pf_full(&c, 512, &[128], 1..2);
    }

    /// A DENSE layer prefills on the GROUPED EXPERT ARMS with degenerate 1-expert routing, because
    /// there is no block-fp8 T-row GEMM opcode and ops 85/86 already are one. This pins the whole
    /// construction: the align op gets NO routing table (that is what makes it synthesise
    /// "every token -> expert 0, gate 1"), the two grouped GEMMs carry `n_exp = 1` and the dense
    /// weight-pointer tables rather than the expert ones, and there is no router and no shared
    /// expert. Getting any of these wrong produces a packet that RUNS and is wrong — the AMD
    /// dispatch `default:` leaves outputs untouched rather than trapping.
    #[test]
    fn mla_full_prefill_dense_layer_uses_synthetic_single_expert_routing() {
        let c = kimi_ref_cfg(); // first_k_dense = 1, so layer 0 is dense
        let m = pf_full(&c, 512, &[128], 0..1);
        let pf = &m.progs[0];
        let ops: Vec<u16> = pf.insts.iter().map(|d| d.op).collect();

        // No router on a dense layer: nothing to score, and `mlp.gate.weight` does not exist.
        assert!(
            !ops.contains(&(DevOp::MoeRouterTopkPf as u16)),
            "a dense layer has no router — its routing is synthesised by the align op"
        );

        let align = pf
            .insts
            .iter()
            .find(|d| d.op == DevOp::MoeAlignPf as u16)
            .expect("dense prefill still aligns: the grouped GEMMs read its meta/row maps");
        assert_eq!(
            align.t[1], TENSOR_NONE,
            "the routing table operand MUST be TENSOR_NONE — that is the signal d_moe_align_pf \
             reads to synthesise single-expert routing. Binding a real table here would route the \
             dense FFN through whatever the previous MoE layer left in `tab`."
        );
        assert_eq!((align.i[1], align.i[2]), (1, 1), "n_exp = 1, top_k = 1");

        for (op, name) in [
            (DevOp::MoeGroupGluPf, "gate/up"),
            (DevOp::MoeGroupDownPf, "down"),
        ] {
            let d = pf
                .insts
                .iter()
                .find(|d| d.op == op as u16)
                .unwrap_or_else(|| panic!("dense prefill must emit the grouped {name} arm"));
            assert_eq!(d.i[2], 1, "{name}: exactly one 'expert'");
            assert_eq!(
                d.i[3],
                MoeEnc::Fp8Blk.code(),
                "{name}: block-fp8 goes in the PREFILL encoding slot i[3], not decode's i[6]"
            );
            assert_ne!(d.t[2], TENSOR_NONE, "{name}: dense weight-pointer table must be bound");
            assert_ne!(d.t[3], TENSOR_NONE, "{name}: dense scale-pointer table must be bound");
        }

        // The combine takes no shared expert — a dense layer has none, and d_moe_combine_pf
        // already honours a null `shared`.
        let cmb = pf
            .insts
            .iter()
            .find(|d| d.op == DevOp::MoeCombinePf as u16)
            .expect("dense prefill combines part into the residual");
        assert_eq!(cmb.t[2], TENSOR_NONE, "no shared expert on a dense layer");
        assert_eq!(cmb.i[1], 1, "k = 1: one part slot per token");
    }

    /// The dense prefill borrows the MoE arms, so it must NOT borrow the MoE weight tables. Binding
    /// `ewt`/`est` there would read 256 routed experts that a dense layer does not have.
    #[test]
    fn mla_full_prefill_dense_binds_dense_tables_not_expert_tables() {
        let c = kimi_ref_cfg();
        let m = pf_full(&c, 512, &[128], 0..1);
        let names = m.tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>();
        assert!(
            names.iter().any(|n| n.contains("mlp.dense_weight_table")),
            "dense prefill declares its own [3] u64 pointer table; got {names:?}"
        );
        let glu = m.progs[0]
            .insts
            .iter()
            .find(|d| d.op == DevOp::MoeGroupGluPf as u16)
            .unwrap();
        let bound = m.tensors[glu.t[2] as usize].name.as_str();
        assert!(
            bound.contains("dense_weight_table"),
            "the grouped GLU on a DENSE layer must read the dense table, got `{bound}`"
        );
    }

    /// EVERY program in one blob must name the SAME peer partial-slot-B offset.
    ///
    /// The host binds `act.dg_tp` ONCE, at `scratch_base + DevBlob::tp.slot_bytes`, and recovers
    /// `slot_bytes` as `max(i[2])` over every `XReduce`/`XReduceTwoShot` in the blob
    /// (`plowrt/src/asset/devblob.rs`). So a blob that bakes a different `i[2]` per program has, by
    /// construction, at most ONE program whose FFN all-reduce reads the buffer its own combine
    /// wrote — the others reduce untouched peer memory and their FFN contribution silently
    /// disappears. That is exactly what a bucket ladder + decode bundle did: `t*h*2` per bucket and
    /// `h*2` for decode, four values, and the decode program (healthy on its own, where its value
    /// IS the max) started emitting a constant token the moment prefill buckets joined it.
    #[test]
    fn every_program_shares_one_peer_partial_slot_offset() {
        let mut c = kimi_ref_cfg();
        c.tp = 2; // TP is what puts XReduce in the program at all
        let m = glm_build_block_pf(
            &c, 2048, 256, 0..2, true, "kimi-ref", MlaArch::Kimi, &[128, 512],
            PrefillScope::Full, MoeEnc::Fp8Blk,
        )
        .0;
        assert!(m.progs.len() >= 3, "two buckets + decode; got {}", m.progs.len());
        let want = 512 * c.hidden * 2; // rows_max * hidden * 2, the widest bucket
        let mut slots: Vec<u32> = m
            .progs
            .iter()
            .flat_map(|p| p.insts.iter())
            .filter(|d| {
                d.op == DevOp::XReduce as u16 || d.op == DevOp::XReduceTwoShot as u16
            })
            .map(|d| d.i[2])
            .collect();
        assert!(!slots.is_empty(), "tp=2 must emit collectives");
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(
            slots,
            vec![0, want],
            "slot A is 0 and slot B is rows_max*hidden*2 in EVERY program; a per-bucket offset \
             here means the host binds act.dg_tp where most programs do not look"
        );
        // ...and the declared buffer must actually be that wide, since the prefill combine writes
        // [T, hidden] into it.
        let dg = m
            .tensors
            .iter()
            .find(|t| t.name == "act.dg_tp")
            .expect("tp>1 declares act.dg_tp");
        assert_eq!(dg.bytes, want as u64, "dg_tp is row-dimensioned, like og_tp");
    }

    /// MXFP4 is the one encoding with no dense prefill arm: its grouped path is the A4W4
    /// fused-bridge, whose scale rows the dense emit does not declare. Refuse rather than emit an
    /// encoding field pointing at an arm nothing bound operands for.
    #[test]
    #[should_panic(expected = "MXFP4 prefill is not implemented for DENSE layer")]
    fn mla_full_prefill_refuses_a_dense_mxfp4_layer() {
        let c = kimi_ref_cfg();
        glm_build_block_pf(
            &c, 512, 256, 0..1, false, "glm-ref", MlaArch::Glm, &[128], PrefillScope::Full,
            MoeEnc::Mxfp4,
        );
    }

    /// The DSA gate does NOT reach for `FlashGatherPrefill`, even armed. A gathered prefill wants
    /// one top_k row PER QUERY (`idx[b][t][top_k]`) and the selector produces a single row, so
    /// emitting the gather would give every query token the last token's selection.
    #[test]
    fn mla_prefill_stays_dense_under_an_armed_dsa_gate() {
        let mut c = kimi_ref_cfg();
        c.has_dsa = true;
        // `kimi_ref_cfg` carries a SYNTHETIC indexer geometry (index_heads 8, index_topk 64) that
        // no AMD kernel can execute — `d_index_score_mfma` static_asserts HIc==32 and interp.hip
        // hardcodes DI_=128 — and this test arms DSA only to check that PREFILL stays dense while
        // DECODE gathers. Give it the real GLM geometry so the fixture describes an emittable blob;
        // the routing assertions below are unchanged and are what the test is actually about.
        c.index_heads = 32;
        c.index_dim = 128;
        c.indexer_full = vec![false, true, false, false];
        let (m, _) =
            glm_build_block_pf(&c, 131072, 256, 1..2, true, "glm-ref", MlaArch::Glm, &[128], PrefillScope::Attn, MoeEnc::Fp8Blk);
        let ops: Vec<u16> = m.progs[0].insts.iter().map(|d| d.op).collect();
        assert!(ops.contains(&(DevOp::FlashMlaPrefill as u16)), "dense MLA prefill");
        assert!(!ops.contains(&(DevOp::FlashGatherPrefill as u16)), "no per-query selector exists");
        assert!(!ops.contains(&(DevOp::IndexScore as u16)), "the indexer is decode-shaped");
        // The DECODE program of the same asset still gathers — the gate is armed, only prefill opts out.
        let dec: Vec<u16> = m.progs.last().unwrap().insts.iter().map(|d| d.op).collect();
        assert!(dec.contains(&(DevOp::FlashGatherDecode as u16)), "decode still gathers at 128k");
    }

    /// The manifest must SEE the prefill buckets, since that is what tells an object builder it
    /// needs the `PLOW_MLA_PREFILL=1` arms. Derived from the instruction stream, not from intent.
    #[test]
    fn mla_prefill_shows_up_in_the_build_manifest() {
        let c = kimi_ref_cfg();
        let (m, _) = pf_block(&c, 512, &[128, 512]);
        let man = crate::manifest::build(&m, "gfx950", &crate::LeanReport::skipped("test: gate not run"));
        let ops: Vec<&str> = man["opcodes"].as_array().unwrap().iter().filter_map(|v| v.as_str()).collect();
        assert!(ops.contains(&"FlashMlaPrefill"));
        assert!(ops.contains(&"MlaMergeFold"));
        assert_eq!(man["features"]["mla"], true);
        // `prefill` must be true for an MLA packet: its prefill flash is the LATENT one, and this
        // flag is what tells the object builder it needs PLOW_MLA_PREFILL=1.
        assert_eq!(man["features"]["prefill"], true);
        assert_eq!(man["features"]["moe_prefill"], false, "attention-only scope");
        // block-fp8 experts ARE fp8 weights, [128,128] scale grid or not.
        assert_eq!(man["features"]["fp8_weights"], true);
        assert_eq!(man["shapes"]["prefill_buckets"], serde_json::json!([128, 512]));
        let progs = man["programs"].as_array().unwrap();
        assert_eq!(progs[0]["kind"], "prefill");
        assert_eq!(progs[0]["bucket"], 128);
        assert_eq!(progs.last().unwrap()["kind"], "decode");
    }

    // ===== Kimi K2.7-Code at the ATOM/AITER comparison point: 384 experts, top-8, TP=4 ==========

    /// Kimi K2.7-Code routing geometry. The GLM-5.2 numbers this emitter was first written against
    /// are 256 experts / top-8; K2.7-Code is 384 / top-8, so anything sized for 256 is a Kimi bug.
    fn kimi_k27_code_cfg(tp: u32) -> GlmCfg {
        let mut c = kimi_ref_cfg();
        c.heads = 64; // real K2.7 head count — the shape that has to divide by tp=4
        c.n_exp = 384;
        c.top_k = 8;
        c.tp = tp;
        c.ep = true; // 384/4 = 96 WHOLE experts per rank
        c
    }

    /// Nothing in the emit is sized for 256 experts: every expert-dimensioned field follows
    /// `c.n_exp`, and the co-resident CU partition follows `top_k`, not the expert count.
    #[test]
    fn kimi_k27_code_384_experts_top8_tp4() {
        let c = kimi_k27_code_cfg(4);
        let (m, d) = pf_block(&c, 1024, &[128]);
        let dec = m.progs.last().unwrap();
        // Router: the score GEMV's N is the expert count, and the top-k tail carries it too.
        let topk = find_op(dec, DevOp::MoeRouterTopk);
        assert_eq!(topk.i[1], 384, "router top-k must see all 384 experts");
        assert_eq!(topk.i[2], 8, "top_k = 8");
        // One (glu, down) pair per top_k slot — 8, not 2 and not 256.
        let n_glu = dec.insts.iter().filter(|x| x.op == DevOp::MoeExpertGluFp8Blk as u16).count();
        assert_eq!(n_glu, 8, "one expert packet per top_k slot");
        for g in dec.insts.iter().filter(|x| x.op == DevOp::MoeExpertGluFp8Blk as u16) {
            assert_eq!(g.i[3], 384, "expert op carries n_exp=384 (table bound)");
            // EP at tp=4: 384/4 = 96 WHOLE experts per rank, so each keeps the FULL moe_inter.
            assert_eq!(g.i[1], c.moe_inter, "EP: whole expert, full moe_inter — not the TP slice");
        }
        assert_eq!(d.dims.n_exp, Some(384));
        assert_eq!(d.dims.top_k, Some(8));
        // 384 divides by every TP degree we care about: 4 -> 96 whole experts per rank.
        assert_eq!(384 % 4, 0);
        // And the attention half of the SAME asset is per-rank sharded at 64/4 = 16 heads.
        assert_eq!(find_op(&m.progs[0], DevOp::FlashMlaPrefill).i[1], 16);
        assert_eq!(find_op(&m.progs[0], DevOp::MlaMergeFold).i[1], 16);
    }

    /// TP=4 is the primary serving degree for this comparison, so pin the whole prefill bucket at
    /// it: real head count, real expert count, both programs present, no decode op in the bucket.
    #[test]
    fn kimi_k27_code_tp4_prefill_bucket_is_complete_attention() {
        let c = kimi_k27_code_cfg(4);
        let (m, _) = pf_block(&c, 8192, &[128, 1024, 8192]);
        assert_eq!(m.prog_t, vec![128, 1024, 8192, 1], "3 buckets + decode");
        for (i, &t) in [128u32, 1024, 8192].iter().enumerate() {
            let p = &m.progs[i];
            assert_eq!(find_op(p, DevOp::FlashMlaPrefill).i[4], t, "n_tok = bucket");
            assert_eq!(find_op(p, DevOp::FlashMlaPrefill).i[1], 16, "nh_l = 64/4");
            assert_eq!(find_op(p, DevOp::MlaMergeFold).i[0], t, "fold n_batch = tokens");
            assert_eq!(find_op(p, DevOp::MlaMergeFold).i[4], 1, "nsplit = 1");
            assert_eq!(find_op(p, DevOp::XReduceTwoShot).i[1], 4, "tp=4 all-reduce");
        }
    }

    /// The DeepSeek flavor differs only in the descriptor arch tag; the emit + kind + no-DSA are
    /// identical to Kimi.
    #[test]
    fn deepseek_arch_tag() {
        let c = kimi_ref_cfg();
        let (_, d) = glm_build_block(&c, 512, 256, 1..2, true, "deepseek-v3", MlaArch::DeepSeek);
        assert_eq!(d.arch, "deepseek_mla_moe");
        assert_eq!(d.kind, vec!["mla_attn", "moe_ffn"]);
        assert_eq!(d.dsa_role, None);
    }
}


#[cfg(test)]
mod nemotron_tests {
    use super::*;

    // ---- reference Mamba-2 SSD math (f32) ------------------------------------------------------

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
    fn softplus(x: f32) -> f32 {
        // numerically-stable log(1+e^x)
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    /// Deterministic pseudo-random stream in [-amp, amp] (reproducible, no rand dep).
    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self, amp: f32) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((self.0 >> 33) as f32) / ((1u64 << 31) as f32); // [0,1)
            (u * 2.0 - 1.0) * amp
        }
    }

    struct Dims {
        t: usize,
        n_head: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    }
    impl Dims {
        fn d_inner(&self) -> usize {
            self.n_head * self.head_dim
        }
        fn hpg(&self) -> usize {
            self.n_head / self.n_groups
        }
    }

    /// SSD selective scan, STATEFUL recurrence form (what the device kernel / block emit mirror).
    /// `x` [T, d_inner], `b`/`cc` [T, n_groups*d_state], `dt_eff` [T, n_head] (already softplus'd),
    /// `a` [n_head] (= -exp(A_log)), `dd` [n_head] (the D skip). `ssm` [n_head*head_dim*d_state] is
    /// read as the initial state and OVERWRITTEN with the final state. Returns yscan [T, d_inner].
    fn scan_recurrence(
        d: &Dims,
        x: &[f32],
        b: &[f32],
        cc: &[f32],
        dt_eff: &[f32],
        a: &[f32],
        dd: &[f32],
        ssm: &mut [f32],
    ) -> Vec<f32> {
        let (nh, hd, ds, ng) = (d.n_head, d.head_dim, d.d_state, d.n_groups);
        let di = d.d_inner();
        let hpg = d.hpg();
        let mut y = vec![0.0f32; d.t * di];
        for t in 0..d.t {
            for h in 0..nh {
                let dtv = dt_eff[t * nh + h];
                let da = (dtv * a[h]).exp();
                let g = h / hpg;
                for p in 0..hd {
                    let xv = x[t * di + h * hd + p];
                    let mut acc = 0.0f32;
                    for n in 0..ds {
                        let bn = b[t * ng * ds + g * ds + n];
                        let cn = cc[t * ng * ds + g * ds + n];
                        let si = h * hd * ds + p * ds + n;
                        ssm[si] = da * ssm[si] + dtv * xv * bn;
                        acc += cn * ssm[si];
                    }
                    y[t * di + h * hd + p] = acc + dd[h] * xv;
                }
            }
        }
        y
    }

    /// SSD selective scan, INDEPENDENT closed-form dual: h_t = exp(cum_t)·h_init +
    /// Σ_{s≤t} exp(cum_t − cum_s)·dt_s·x_s⊗B_s, y_t = Σ_n C_t·h_t + D·x_t. Materializes the decay
    /// per (t,s) and sums — a structurally different computation (different float order) than the
    /// stateful recurrence, so agreement to tolerance validates the recurrence. `ssm_init` is the
    /// carried-in state; does NOT mutate it.
    fn scan_dual(
        d: &Dims,
        x: &[f32],
        b: &[f32],
        cc: &[f32],
        dt_eff: &[f32],
        a: &[f32],
        dd: &[f32],
        ssm_init: &[f32],
    ) -> Vec<f32> {
        let (nh, hd, ds, ng) = (d.n_head, d.head_dim, d.d_state, d.n_groups);
        let di = d.d_inner();
        let hpg = d.hpg();
        let mut y = vec![0.0f32; d.t * di];
        for h in 0..nh {
            // cumulative log-decay per t: cum[t] = Σ_{r=0}^{t} dt_r·A_h
            let mut cum = vec![0.0f32; d.t];
            let mut run = 0.0f32;
            for t in 0..d.t {
                run += dt_eff[t * nh + h] * a[h];
                cum[t] = run;
            }
            let g = h / hpg;
            for t in 0..d.t {
                for p in 0..hd {
                    let mut acc = dd[h] * x[t * di + h * hd + p];
                    for n in 0..ds {
                        let cn = cc[t * ng * ds + g * ds + n];
                        // initial-state contribution
                        let mut hval = cum[t].exp() * ssm_init[h * hd * ds + p * ds + n];
                        // input contributions from all s ≤ t
                        for s in 0..=t {
                            let decay = (cum[t] - cum[s]).exp();
                            let xs = x[s * di + h * hd + p];
                            let bs = b[s * ng * ds + g * ds + n];
                            hval += decay * dt_eff[s * nh + h] * xs * bs;
                        }
                        acc += cn * hval;
                    }
                    y[t * di + h * hd + p] = acc;
                }
            }
        }
        y
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// The NEW SSM math: the stateful recurrence (kernel/emit form) equals the independent
    /// closed-form dual to f32 tolerance — with a NON-ZERO carried-in ssm_state, so the initial
    /// state term is exercised. Reports the max-abs error vs the golden.
    #[test]
    fn mamba2_scan_matches_independent_recurrence() {
        let d = Dims { t: 6, n_head: 4, head_dim: 5, d_state: 3, n_groups: 2 };
        let di = d.d_inner();
        let gd = d.n_groups * d.d_state;
        let mut r = Lcg(0x1234_5678_9abc_def0);
        let x: Vec<f32> = (0..d.t * di).map(|_| r.f(0.5)).collect();
        let b: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
        let cc: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
        // dt already softplus'd (positive); A = -exp(a_log) (negative) => stable decay in (0,1).
        let dt_eff: Vec<f32> = (0..d.t * d.n_head).map(|_| softplus(r.f(1.0))).collect();
        let a: Vec<f32> = (0..d.n_head).map(|_| -(r.f(0.5) + 0.7).exp()).collect();
        let dd: Vec<f32> = (0..d.n_head).map(|_| r.f(0.5)).collect();
        let ssm_init: Vec<f32> = (0..d.n_head * d.head_dim * d.d_state).map(|_| r.f(0.3)).collect();

        let mut ssm = ssm_init.clone();
        let y_rec = scan_recurrence(&d, &x, &b, &cc, &dt_eff, &a, &dd, &mut ssm);
        let y_dual = scan_dual(&d, &x, &b, &cc, &dt_eff, &a, &dd, &ssm_init);
        let err = max_abs(&y_rec, &y_dual);
        eprintln!("mamba2 SSM scan: max-abs err (recurrence vs independent dual) = {err:e}");
        assert!(err < 1e-4, "SSM scan diverges from independent golden: max-abs {err:e}");
    }

    /// Prefill/decode equivalence: running the scan as ONE T-step prefill leaves the same
    /// ssm_state, and yields the same last-token output, as feeding the tokens one at a time
    /// through single-step decode calls (each carrying the state forward). This is the
    /// state-carry contract the harness relies on (§6, §7).
    #[test]
    fn mamba2_decode_equals_prefill() {
        let d = Dims { t: 5, n_head: 3, head_dim: 4, d_state: 3, n_groups: 1 };
        let di = d.d_inner();
        let gd = d.n_groups * d.d_state;
        let mut r = Lcg(0xdead_beef_cafe_1234);
        let x: Vec<f32> = (0..d.t * di).map(|_| r.f(0.5)).collect();
        let b: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
        let cc: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
        let dt_eff: Vec<f32> = (0..d.t * d.n_head).map(|_| softplus(r.f(1.0))).collect();
        let a: Vec<f32> = (0..d.n_head).map(|_| -(r.f(0.3) + 0.7).exp()).collect();
        let dd: Vec<f32> = (0..d.n_head).map(|_| r.f(0.5)).collect();

        // Full prefill scan.
        let mut ssm_pf = vec![0.0f32; d.n_head * d.head_dim * d.d_state];
        let y_pf = scan_recurrence(&d, &x, &b, &cc, &dt_eff, &a, &dd, &mut ssm_pf);

        // Token-at-a-time decode, carrying ssm_state forward.
        let mut ssm_dec = vec![0.0f32; d.n_head * d.head_dim * d.d_state];
        let mut y_last = vec![0.0f32; di];
        for t in 0..d.t {
            let d1 = Dims { t: 1, ..copy_dims(&d) };
            let xr = &x[t * di..(t + 1) * di];
            let br = &b[t * gd..(t + 1) * gd];
            let cr = &cc[t * gd..(t + 1) * gd];
            let dtr = &dt_eff[t * d.n_head..(t + 1) * d.n_head];
            y_last = scan_recurrence(&d1, xr, br, cr, dtr, &a, &dd, &mut ssm_dec);
        }
        let err_state = max_abs(&ssm_pf, &ssm_dec);
        let err_y = max_abs(&y_pf[(d.t - 1) * di..], &y_last);
        eprintln!("mamba2 prefill-vs-decode: ssm_state err={err_state:e} last-token err={err_y:e}");
        assert!(err_state < 1e-5, "decode state != prefill state: {err_state:e}");
        assert!(err_y < 1e-5, "decode last-token != prefill: {err_y:e}");
    }

    fn copy_dims(d: &Dims) -> Dims {
        Dims {
            t: d.t,
            n_head: d.n_head,
            head_dim: d.head_dim,
            d_state: d.d_state,
            n_groups: d.n_groups,
        }
    }

    // ---- emit op-sequence + descriptor ---------------------------------------------------------

    /// Synthetic small Nemotron-3 hybrid cfg (structurally faithful: mamba mixer + GQA attn + MoE).
    /// Layer 0 = mamba, 1 = attn, 2 = moe (a minimal one-of-each pattern the block extraction walks).
    fn nemo_ref_cfg() -> NemoCfg {
        NemoCfg {
            layers: 3,
            hidden: 64,
            d_inner: 128,
            n_head: 8,
            head_dim: 16, // d_inner / n_head
            d_state: 16,
            d_conv: 4,
            n_groups: 2,
            attn_heads: 8,
            attn_kv_heads: 2,
            attn_head_dim: 16,
            n_exp: 16,
            top_k: 4,
            shared_exp: 1,
            moe_inter: 96,
            eps: 1e-5,
            kinds: vec![NemoKind::Mamba, NemoKind::Attn, NemoKind::Moe],
        }
    }

    fn block_ops(c: &NemoCfg, block: std::ops::Range<usize>) -> Vec<u16> {
        let (m, _d) = nemotron_build_block(c, 512, 256, block, "nemotron-ref");
        m.progs[0].insts.iter().map(|d| d.op).collect()
    }

    /// Mamba mixer block: input RMSNorm, 3 in_proj GEMVs (z/xBC/dt), the NEW Mamba2Scan, out_proj
    /// GEMV, residual — `act.x` in and out, no embed/tail.
    #[test]
    fn nemotron_mamba_block_sequence() {
        use DevOp::*;
        let c = nemo_ref_cfg();
        assert_eq!(
            block_ops(&c, 0..1),
            vec![RmsNorm, Gemv, Gemv, Gemv, Mamba2Scan, Gemv, Residual]
                .into_iter()
                .map(|o| o as u16)
                .collect::<Vec<_>>(),
            "mamba mixer block sequence"
        );
    }

    /// GQA attention block reuses the existing attn DevOps.
    #[test]
    fn nemotron_attn_block_sequence() {
        use DevOp::*;
        let c = nemo_ref_cfg();
        assert_eq!(
            block_ops(&c, 1..2),
            vec![RmsNorm, GemvQkv, HeadNormRope, FlashDecode, FlashMerge, Gemv, Residual]
                .into_iter()
                .map(|o| o as u16)
                .collect::<Vec<_>>(),
            "gqa attention block sequence"
        );
    }

    /// MoE block reuses the existing MoE DevOps (router split + shared expert + top_k experts +
    /// combine), matching the kimi MoE structure.
    #[test]
    fn nemotron_moe_block_sequence() {
        use DevOp::*;
        let c = nemo_ref_cfg();
        let mut want = vec![RmsNorm, Gemv, MoeRouterTopk, GemvGlu, Gemv];
        for _ in 0..c.top_k {
            want.push(MoeExpertGlu);
            want.push(MoeExpertDown);
        }
        want.push(MoeCombine);
        assert_eq!(
            block_ops(&c, 2..3),
            want.into_iter().map(|o| o as u16).collect::<Vec<_>>(),
            "moe block sequence"
        );
    }

    /// Mamba block descriptor: arch nemotron_h, kind ["mamba2"], Mamba-2 dims, conv+ssm carried
    /// state (NO kv), no attn/MoE dims.
    #[test]
    fn nemotron_mamba_descriptor() {
        let c = nemo_ref_cfg();
        let (_, d) = nemotron_build_block(&c, 512, 256, 0..1, "Nemotron-3");
        assert_eq!(d.arch, "nemotron_h");
        assert_eq!(d.kind, vec!["mamba2"]);
        assert_eq!(d.layer, 0);
        assert_eq!(d.dims.d_inner, Some(128));
        assert_eq!(d.dims.n_head, Some(8));
        assert_eq!(d.dims.head_dim, Some(16));
        assert_eq!(d.dims.d_state, Some(16));
        assert_eq!(d.dims.d_conv, Some(4));
        assert_eq!(d.dims.n_groups, Some(2));
        assert_eq!(d.dims.heads, None, "mamba block has no attn dims");
        assert_eq!(d.dims.n_exp, None, "mamba block has no MoE dims");
        assert_eq!(d.carried_state.len(), 2);
        assert_eq!(d.carried_state[0].role, "conv");
        assert_eq!(d.carried_state[0].layout, "conv");
        assert_eq!(d.carried_state[0].tensors, vec!["mamba.0.conv_state"]);
        assert_eq!(d.carried_state[1].role, "ssm");
        assert_eq!(d.carried_state[1].layout, "ssm_head_major");
        assert_eq!(d.carried_state[1].tensors, vec!["mamba.0.ssm_state"]);
        assert_eq!(d.weights.prefix, "backbone.layers.0.");
        assert!(d.programs.prefill_buckets.is_empty());
        assert_eq!(d.outputs[0].name, "act.xnext", "one (odd) layer -> act.xnext");
    }

    /// Attention block descriptor: kind ["gqa_attn"], GQA dims, kv carried state.
    #[test]
    fn nemotron_attn_descriptor() {
        let c = nemo_ref_cfg();
        let (_, d) = nemotron_build_block(&c, 512, 256, 1..2, "Nemotron-3");
        assert_eq!(d.kind, vec!["gqa_attn"]);
        assert_eq!(d.dims.heads, Some(8));
        assert_eq!(d.dims.kv_heads, Some(2));
        assert_eq!(d.dims.head_dim, Some(16));
        assert_eq!(d.dims.d_inner, None, "attn block has no mamba dims");
        assert_eq!(d.dims.n_exp, None);
        assert_eq!(d.carried_state.len(), 1);
        assert_eq!(d.carried_state[0].role, "kv");
        assert_eq!(d.carried_state[0].tensors, vec!["kv.1.k", "kv.1.v"]);
    }

    /// MoE block descriptor: kind ["moe_ffn"], MoE dims, NO carried state.
    #[test]
    fn nemotron_moe_descriptor() {
        let c = nemo_ref_cfg();
        let (_, d) = nemotron_build_block(&c, 512, 256, 2..3, "Nemotron-3");
        assert_eq!(d.kind, vec!["moe_ffn"]);
        assert_eq!(d.dims.n_exp, Some(16));
        assert_eq!(d.dims.top_k, Some(4));
        assert_eq!(d.dims.shared_exp, Some(1));
        assert_eq!(d.dims.moe_inter, Some(96));
        assert_eq!(d.dims.d_inner, None);
        assert_eq!(d.dims.heads, None);
        assert!(d.carried_state.is_empty(), "MoE block carries no state");
    }

    /// A multi-layer block chains all three layer kinds; kind lists each, carried_state unions the
    /// mamba (conv+ssm) and attn (kv) entries, and the residual ping-pong lands the output in
    /// `act.xnext` after 3 (odd) layers.
    #[test]
    fn nemotron_multi_layer_chains() {
        use DevOp::*;
        let c = nemo_ref_cfg();
        let ops = block_ops(&c, 0..3);
        // mamba(7) + attn(7) + moe(5 + 2*top_k + 1)
        assert_eq!(ops[0], RmsNorm as u16);
        assert_eq!(ops[4], Mamba2Scan as u16, "mamba mixer first");
        assert!(ops.contains(&(FlashDecode as u16)), "attn layer present");
        assert!(ops.contains(&(MoeCombine as u16)), "moe layer present");
        let (_, d) = nemotron_build_block(&c, 512, 256, 0..3, "Nemotron-3");
        assert_eq!(d.kind, vec!["mamba2", "gqa_attn", "moe_ffn"]);
        assert_eq!(d.layer, 0);
        assert_eq!(d.outputs[0].name, "act.xnext", "3 layers (odd) -> act.xnext");
        // conv + ssm (mamba L0) + kv (attn L1); moe contributes none.
        let roles: Vec<&str> = d.carried_state.iter().map(|s| s.role.as_str()).collect();
        assert_eq!(roles, vec!["conv", "ssm", "kv"]);
        // all Mamba-2 dims and attn dims and MoE dims populated.
        assert_eq!(d.dims.d_inner, Some(128));
        assert_eq!(d.dims.kv_heads, Some(2));
        assert_eq!(d.dims.n_exp, Some(16));
    }
}
