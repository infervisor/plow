//! MLA + MoE emit family: GLM-5.2 / Kimi K2.7 / DeepSeek-V3 (shared MLA+MoE emit)
//! and Nemotron-3 (Mamba-2 hybrid). Split out of `lib.rs` (module breakdown). All
//! are `--block` extraction on this path; `run_verified` dispatches on model_type.
use std::path::Path;

use packet::dev::{DevOp, TENSOR_NONE};
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
    moe_inter: u32,     // IMOE 2048 (per-expert intermediate)
    dense_inter: u32,   // 12288 (layers < first_k_dense)
    first_k_dense: u32, // 3 (layers 0,1,2 dense FFN; 3-77 MoE)
    route_scale: f32,   // 2.5 (routed_scaling_factor)
    attn_scale: f32,    // 1/sqrt(qk_head_dim = qk_nope+qk_rope = 256) = 0.0625
    rope_theta: f64,    // 8e6 (interleaved partial RoPE on the 64 rope dims)
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
        self.has_dsa
            && ctx > CROSSOVER
            && std::env::var("PLOW_GLM_DSA").ok().as_deref() != Some("0")
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
        moe_inter: g("moe_intermediate_size"),
        dense_inter: g("intermediate_size"),
        first_k_dense: g("first_k_dense_replace"),
        route_scale: v["routed_scaling_factor"].as_f64().unwrap() as f32,
        attn_scale: (qk_head as f32).powf(-0.5),
        rope_theta: v["rope_theta"].as_f64().unwrap_or(8_000_000.0),
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
/// max_ctx and baked into FlashMlaDecode i[7] (the interp instantiates GF∈{2,4} and dispatches on
/// i[7]; LDS/registers are sized for the GF=4 max, so occ is unchanged). GLM's MLA latent is
/// HEAD-SHARED, so GF query heads re-stream the compact latent once per head-group => latent HBM
/// traffic ~ n_head/GF. TRADEOFF (measured, MI350X full-model TP4 decode): GF=4 CUTS long ctx
/// (128k 125 vs 140 ms/tok; 8k-32k 1.3-1.6x on the MLA chain) but ADDS split/merge overhead that
/// HURTS short ctx (1k 79 vs 58 ms/tok — the tiny 1k latent stream isn't worth the extra splits).
/// So: GF=2 for short-ctx pkts (preserve the router-split ~58ms@1k), GF=4 for long-ctx pkts.
/// PLOW_GLM_GF pins GF∈{2,4} (crossover sweeps). Crossover ~4k (see perf-data/glm52-plow-decode-tuned.json).
/// Long-ctx / MAX head-fusion factor. Matches the op_attention.h GLM_MLA_GF define (the interp
/// sizes the MLA-decode LDS + registers for this GF), and is the GF `glm_nsplit`'s chip-fill cap
/// assumes (the per-pkt glm_gf never exceeds it, so nsplit stays a safe over-estimate at GF=2).
pub(crate) const GLM_MLA_GF: u32 = 4;
const GLM_GF_CROSSOVER: u32 = 4096; // max_ctx <= this -> GF=2; else GF=8
fn glm_gf(ctx: u32) -> u32 {
    // GF=8 measured 1.5-1.9x faster than GF=4 at every ctx>=8192 (P2, plans/
    // mla-sm120-kernels.md §7): the NH/GF latent-reread cut dominates; merge is
    // GF-independent (nsplit unchanged) and 134 regs < the 225 megakernel cap so
    // occupancy is unaffected. nsplit still sized for GLM_MLA_GF=4 (a conservative
    // under-split at GF=8 -> slight chip under-fill only at batch=1; GF=8 wins there
    // anyway). PLOW_GLM_GF pins {2,4,8}.
    std::env::var("PLOW_GLM_GF")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v == 2 || v == 4 || v == 8)
        .unwrap_or(if ctx <= GLM_GF_CROSSOVER { 2 } else { 8 })
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
/// empty splits). Measured MI350X chain optima (mla_perf, tp4 & tp8): ns~16 up to 8k, ns~64 at 32k;
/// `ctx/512` floored at 16 reproduces them and yields the 32k win (tp8 242->165us 1.47x, tp4
/// 240->176us 1.36x) with no mid-ctx regression. tp=1 is fill-capped to 16 (already chip-full), so
/// byte-identical. See plans/glm-mla-flash-tuning.md and Plow.SplitK (the split reduction equals
/// the sequential sum for ANY nsplit; occupancy is monotone in the split count up to n_cu).
pub(crate) fn glm_nsplit(ctx: u32, heads: u32) -> u32 {
    /// KV rows staged per flash step (op_attention.h FA_BKV) — the KV-tile granularity a split
    /// divides. A split covering zero whole tiles writes -inf and is pure overhead (a launched
    /// workgroup + an extra O(nsplit) merge input), so nsplit is capped at the tile count.
    const FA_BKV: u32 = 32;
    /// Latent bytes per split at which the decode saving stops beating the O(nsplit) merge growth.
    /// ns scales as ctx/NS_PER (measured knee) below the fill cap.
    const NS_PER: u32 = 512;
    /// Split floor: below this the fixed decode overhead already dominates, so extra splits only
    /// add merge cost (measured: ns=16 is the chain optimum for ctx<=8k at every TP degree).
    const NS_FLOOR: u32 = 16;
    let n_grp = (heads / GLM_MLA_GF).max(1);
    let fill = ((256 + n_grp - 1) / n_grp).max(1); // chip-fill cap: splits to cover 256 CUs
    let kv_tiles = ctx.div_ceil(FA_BKV).max(1); // never split finer than there are KV tiles
                                                // ctx-scaled cost optimum, floored, then capped by the chip and the KV-tile count.
    let ns = (ctx / NS_PER).max(NS_FLOOR).min(fill).min(kv_tiles).max(1);
    std::env::var("PLOW_GLM_NS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(ns)
}
/// Router flags: bit0 sigmoid, bit1 norm_topk, bit2 apply e_score_correction_bias to SELECTION
/// only (DeepSeek/GLM noaux_tc). Mirrors FLAGS in the B4 harness.
const GLM_ROUTER_FLAGS: u32 = 1 | 2 | 4;
/// Expert/shared GLU activation = SiLU (SwiGLU). Mirrors ACT in the B4 harness.
const GLM_ACT_SILU: u32 = 1;

/// Per-layer GLM weights. Derived (absorbed / rope-folded) tensors are bf16 and named under a
/// `.derived.` segment the host weight-prep writes; the block-fp8 projections, router, and experts
/// keep their checkpoint names. `TENSOR_NONE` for the sub-block a layer does not have (dense vs MoE).
struct GlmLW {
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
    // Interleaved partial-RoPE cos/sin tables for the 64 rope dims (theta=8e6, full rotation of DR).
    // Same [ctx][DR/2] layout the half-split path uses (freq index = element>>1); the interp's HD=64
    // dispatch selects the INTERLEAVE=true template. See rope_tables + op_norm.h.
    let [cos_t, sin_t] = GenTensor::rope_pair(ctx, c.qk_rope, c.rope_theta, 1.0, RopeScale::None);
    let cos = b.tensor_gen("in.cos", cos_t.byte_len(), cos_t);
    let sin = b.tensor_gen("in.sin", sin_t.byte_len(), sin_t);
    let emb = b.tensor("model.embed_tokens.weight", (c.vocab * h) as u64 * BF16);
    let fin = b.tensor("model.norm.weight", h as u64 * BF16);
    let head = b.tensor("lm_head.weight", (c.vocab * h) as u64 * BF16);

    let x = ac(b, "x", h as u64 * BF16);
    let xn = ac(b, "xn", h as u64 * BF16);
    let qlr = ac(b, "qlr", ql as u64 * BF16);
    let qlat = ac(b, "qlat", ql as u64 * BF16);
    let ckvraw = ac(b, "ckvraw", dk as u64 * BF16);
    // Head-dimensioned activations shrink to nh_l heads under TP (the flash/merge/uv/o-fold ops run
    // this rank's head-shard); expert/dense-intermediate activations shrink to imoe_l/di_l lanes.
    let qa = ac(b, "qa", (nh_l * dk) as u64 * BF16);
    let qrr = ac(b, "qrr", (nh_l * dr) as u64 * BF16);
    let qr = ac(b, "qr", (nh_l * dr) as u64 * BF16);
    let krr = ac(b, "krr", dr as u64 * BF16);
    // TP-sharded head count (nh_l) x ctx-adaptive nsplit (glm_nsplit, from glm-tune-flash).
    // nh_l (not global c.heads) so the fill target matches this rank's actual work-item count.
    let ns = glm_nsplit(ctx, nh_l);
    let opart = ac(b, "opart", (nh_l * ns * dk) as u64 * F32);
    let mlpart = ac(b, "mlpart", (nh_l * ns * 2) as u64 * F32);
    let olat = ac(b, "olat", (nh_l * dk) as u64 * BF16);
    let oat = ac(b, "oat", (nh_l * vd) as u64 * BF16);
    let attn = ac(b, "attn", h as u64 * BF16);
    let xmid = ac(b, "xmid", h as u64 * BF16);
    let xn2 = ac(b, "xn2", h as u64 * BF16);
    let tab = ac(b, "tab", tk as u64 * 8);
    let rlogit = ac(b, "rlogit", e as u64 * BF16); // router score-GEMV output [n_exp] bf16
    let shfu = ac(b, "shfu", imoe_l as u64 * BF16);
    let shared = ac(b, "shared", h as u64 * BF16);
    // Routed-expert gate/up buffer: full moe_inter width per slot under EP (whole experts), else TP shard.
    let fu = ac(b, "fu", (tk * imoe_e) as u64 * BF16);
    let dfu = ac(b, "dfu", di_l as u64 * BF16);
    let part = ac(b, "part", (tk * h) as u64 * F32);
    let xnext = ac(b, "xnext", h as u64 * BF16);
    let logits = ac(b, "logits", c.vocab as u64 * BF16);
    let amax = ac(b, "amax.part", AMAX_BLOCKS as u64 * 8);
    // TP peer-mapped partials (§7a) — only under sharding; the host binds these into peer scratch at
    // offset 0 / slot_b so the row-parallel o_proj + MoE/dense down write peer-visible partials that
    // XReduce sums. zero_h is a persistent zero buffer used as the MoeCombine residual under TP (the
    // real residual xmid is added AFTER the all-reduce, so it is not summed N times).
    let og_tp = if tp > 1 {
        ac(b, "og_tp", h as u64 * BF16)
    } else {
        TENSOR_NONE
    };
    let dg_tp = if tp > 1 {
        ac(b, "dg_tp", h as u64 * BF16)
    } else {
        TENSOR_NONE
    };
    let zero_h = if tp > 1 {
        b.tensor_init("act.zero_h", vec![0u8; h as usize * 2])
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
        let [ct, st] = GenTensor::rope_idx_pair(ctx, dr, di, c.rope_theta);
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
        let t = |b: &mut Builder, s: &str, sz: u64| b.tensor(&format!("model.layers.{l}.{s}"), sz);
        let dense = c.is_dense(l);
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
            gin: t(b, "input_layernorm.weight", h as u64 * BF16),
            qad: t(b, "self_attn.q_a_proj.weight", (ql * h) as u64 * BF16),
            gqa: t(b, "self_attn.q_a_layernorm.weight", ql as u64 * BF16),
            wqa: t(
                b,
                "self_attn.derived.q_absorb.weight",
                (nh_l * dk * ql) as u64 * BF16,
            ),
            wqr: t(
                b,
                "self_attn.derived.q_rope.weight",
                (nh_l * dr * ql) as u64 * BF16,
            ),
            ckvd: t(
                b,
                "self_attn.derived.kv_a_latent.weight",
                (dk * h) as u64 * BF16,
            ),
            gkva: t(b, "self_attn.kv_a_layernorm.weight", dk as u64 * BF16),
            krotd: t(b, "self_attn.derived.k_rope.weight", (dr * h) as u64 * BF16),
            wuv: t(
                b,
                "self_attn.derived.v_absorb.weight",
                (nh_l * dk * vd) as u64 * BF16,
            ),
            wo: t(b, "self_attn.o_proj.weight", (h * nh_l * vd) as u64 * BF16),
            gpost: t(b, "post_attention_layernorm.weight", h as u64 * BF16),
            wr: if dense {
                TENSOR_NONE
            } else {
                t(b, "mlp.gate.weight", (e * h) as u64 * BF16)
            },
            bias: if dense {
                TENSOR_NONE
            } else {
                t(b, "mlp.gate.e_score_correction_bias", e as u64 * F32)
            },
            shg: if dense {
                TENSOR_NONE
            } else {
                t(
                    b,
                    "mlp.shared_experts.gate_proj.weight",
                    (imoe_l * h) as u64 * BF16,
                )
            },
            shu: if dense {
                TENSOR_NONE
            } else {
                t(
                    b,
                    "mlp.shared_experts.up_proj.weight",
                    (imoe_l * h) as u64 * BF16,
                )
            },
            shd: if dense {
                TENSOR_NONE
            } else {
                t(
                    b,
                    "mlp.shared_experts.down_proj.weight",
                    (h * imoe_l) as u64 * BF16,
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
            dgate: if dense {
                t(b, "mlp.gate_proj.weight", (di_l * h) as u64)
            } else {
                TENSOR_NONE
            },
            dgate_s: if dense {
                t(
                    b,
                    "mlp.gate_proj.weight_scale_inv",
                    (db_l * hb) as u64 * F32,
                )
            } else {
                TENSOR_NONE
            },
            dup: if dense {
                t(b, "mlp.up_proj.weight", (di_l * h) as u64)
            } else {
                TENSOR_NONE
            },
            dup_s: if dense {
                t(b, "mlp.up_proj.weight_scale_inv", (db_l * hb) as u64 * F32)
            } else {
                TENSOR_NONE
            },
            ddown: if dense {
                t(b, "mlp.down_proj.weight", (h * di_l) as u64)
            } else {
                TENSOR_NONE
            },
            ddown_s: if dense {
                t(
                    b,
                    "mlp.down_proj.weight_scale_inv",
                    (hb * db_l) as u64 * F32,
                )
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

/// Emit the shared MLA attention sub-block (input norm -> q/kv down + absorbed folds -> dynamic
/// interleaved RoPE on the 64 rope dims -> FLASH_MLA_DECODE -> merge -> O_UV_FOLD -> o_proj ->
/// residual -> post-attention norm). Writes `n.xn2` (the FFN input) and returns the post-attn-norm
/// completion dep. IDENTICAL for the dense (0-2) and MoE (3-77) layers, so both blocks call it.
fn emit_glm_mla(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
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
    // GEMV helper (M=1 decode, no norm fold) — the bf16 projection form both B4 passes used.
    let gemv = |b: &mut Builder, out: u32, x: u32, wt: u32, nn: u32, k: u32, deps: &[u32]| -> u32 {
        b.emit(DevOp::Gemv, all.clone(), deps, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = wt;
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
        let c_fa = b.emit(DevOp::GemvQkv, all.clone(), &[c_rn1], |d| {
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
        });
        (c_fa, c_fa, c_fa)
    } else {
        (
            gemv(b, n.qlr, n.xn, w.qad, ql, h, &[c_rn1]),
            gemv(b, n.ckvraw, n.xn, w.ckvd, dk, h, &[c_rn1]),
            gemv(b, n.krr, n.xn, w.krotd, dr, h, &[c_rn1]),
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
        let c_fg = b.emit(DevOp::GemvQkv, all.clone(), &[c_rnq], |d| {
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
        });
        (c_fg, c_fg)
    } else {
        (
            gemv(b, n.qa, n.qlat, w.wqa, nh_l * dk, ql, &[c_rnq]),
            gemv(b, n.qrr, n.qlat, w.wqr, nh_l * dr, ql, &[c_rnq]),
        )
    };
    let c_qr = b.emit(DevOp::HeadNormRope, all.clone(), &[c_qrr], |d| {
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
    let c_krd = b.emit(DevOp::HeadNormRope, all.clone(), &[c_krr], |d| {
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
        let c_qi = b.emit(DevOp::HeadNormRope, all.clone(), &[c_q0], |d| {
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
        let c_ki = b.emit(DevOp::HeadNormRope, all.clone(), &[c_kn], |d| {
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
        let c_w = gemv(b, n.widx, n.xn, w.iwp, hi, h, &[c_rn1]);
        // score[t] = Σ_h w[h]·ReLU(q_idx[h]·k_idx[t]) · scale  (scale = 1/√DI · 1/√HI; selection is
        // scale-invariant, this reproduces HF numerically).
        let c_sc = b.emit(DevOp::IndexScore, all.clone(), &[c_qi, c_ki, c_w], |d| {
            d.t[0] = n.iscore;
            d.t[1] = n.qidx;
            d.t[2] = n.kidx[slot];
            d.t[3] = n.widx;
            d.t[4] = n.kvlen;
            d.i[0] = 1;
            d.i[2] = ctx;
            d.f[0] = (di as f32).powf(-0.5) * (hi as f32).powf(-0.5);
        });
        // top-k SELECT -> n.iidx (ONE cooperative launch: grid-sync radix). Perf floor 2: emit on a
        // 32-CU slice, NOT all 256. The selector is grid-barrier CONTENTION-bound, not bandwidth-bound
        // (the score array is only ctx*4 B); cutting the co-resident WG count 256->32 drops the atomic
        // contention on the grid-sync counter and the shared histogram bins (~204->144us @128k, STILL
        // set-EXACT). The kernel reads nwg from in->blocks (=32) and grid-strides blockIdx.x over 0..31,
        // so CUs 0..31 give full, exact coverage; all 32 are trivially co-resident under the persistent
        // interp (256 CUs resident, this op gates on INDEX_SCORE, so its 32 WGs run together).
        let sel_wgs: Vec<u32> = (0..32.min(b.n_cu())).collect();
        b.emit(DevOp::IndexSelect, sel_wgs, &[c_sc], |d| {
            d.t[0] = n.iidx;
            d.t[1] = n.iscore;
            d.t[2] = n.ighist;
            d.t[3] = n.igctl;
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
    let c_fl = b.emit(
        if dsa {
            DevOp::FlashGatherDecode
        } else {
            DevOp::FlashMlaDecode
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
            d.i[0] = 1;
            d.i[1] = nh_l;
            d.i[2] = ctx;
            d.i[4] = ns_attn;
            d.i[5] = KV_MASK_NONE;
            d.i[7] = glm_gf(ctx); // per-pkt head-fusion factor (interp dispatches GF=2/4 on this)
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
    let c_uv = b.emit(DevOp::MlaMergeFold, all.clone(), &[c_fl], |d| {
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
    let c_op = if tp > 1 && !no_xr {
        let c_p = gemv(b, n.og_tp, n.oat, w.wo, h, nh_l * vd, &[c_uv]);
        emit_xreduce(b, xgate, true, xr_cus, c_p, n.attn, h, tp, 0)
    } else {
        gemv(b, n.attn, n.oat, w.wo, h, nh_l * vd, &[c_uv])
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
        let c_rs = b.emit(DevOp::Residual, one.clone(), &[c_op], |d| {
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

/// Emit ONE MoE (sparse) GLM decoder block — the exact block validated by the B4 harness. `slot`
/// indexes `tn.lw`/`tn.ckv`/`tn.krot`. `use_fp8` selects the block-fp8 expert opcodes (45/46) over
/// the bf16 ones (41/42). Returns the MoeCombine completion dep.
pub(crate) fn emit_glm_block(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    use_fp8: bool,
    x_in: u32,
    x_out: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    assert!(slot < n.lw.len(), "slot out of range");
    let c_rn2 = emit_glm_mla(b, c, n, slot, ctx, x_in, pre, xgate, xr_cus);
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
    let gemv = |b: &mut Builder, out: u32, x: u32, wt: u32, nn: u32, k: u32, deps: &[u32]| -> u32 {
        b.emit(DevOp::Gemv, all.clone(), deps, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = wt;
            d.i[0] = 1;
            d.i[1] = nn;
            d.i[2] = k;
            d.f[0] = 1.0;
        })
    };
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
    let shared_cus = if cores >= 2 {
        b.split(tk + 1, tk)
    } else {
        all.clone()
    };
    // Per-slot routed-expert CU set: disjoint 1/tk (cores 1) or 1/(tk+1) (cores 2) slice, else all-256.
    let expert_parts = if cores >= 2 { tk + 1 } else { tk };

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
        let c_score = gemv(b, n.rlogit, n.xn2, w.wr, e, h, &[c_rn2]);
        b.emit(DevOp::MoeRouterTopk, one.clone(), &[c_score], |d| {
            d.t[0] = n.tab;
            d.t[1] = n.rlogit;
            d.t[3] = w.bias;
            d.i[1] = e;
            d.i[2] = tk;
            d.i[3] = GLM_ROUTER_FLAGS;
            d.f[0] = c.route_scale;
        })
    };
    // 16 shared expert gate|up (fused GLU) — column-parallel: this rank's imoe_l lanes. Under cores>=2
    //   it runs on its OWN slice (shared_cus), CO-RESIDENT with the routed experts (it is routing-
    //   independent — gated only on c_rn2 — so it overlaps the expert chain instead of preceding it).
    let c_shglu = b.emit(DevOp::GemvGlu, shared_cus.clone(), &[c_rn2], |d| {
        d.t[0] = n.shfu;
        d.t[1] = n.xn2;
        d.t[2] = w.shg;
        d.t[5] = w.shu;
        d.i[0] = 1;
        d.i[1] = imoe_l;
        d.i[2] = h;
        d.i[5] = GLM_ACT_SILU;
    });
    // 17 shared expert down — row-parallel (imoe_l input): writes a PARTIAL H-vector under TP
    let c_shd = b.emit(DevOp::Gemv, shared_cus.clone(), &[c_shglu], |d| {
        d.t[0] = n.shared;
        d.t[1] = n.shfu;
        d.t[2] = w.shd;
        d.i[0] = 1;
        d.i[1] = h;
        d.i[2] = imoe_l;
        d.f[0] = 1.0;
    });
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
            // cores 0: all-256 (serial). cores>=1: disjoint 1/expert_parts slice → the tk experts
            //   (+ shared under cores 2) are co-resident and run concurrently, gated on c_router.
            let ecus = if cores >= 1 {
                b.split(expert_parts, sl)
            } else {
                all.clone()
            };
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
        let c_cmb = b.emit(DevOp::MoeCombine, all.clone(), &deps, |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = n.zero_h;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
        });
        let slot_b = h * 2; // dg_tp peer offset (partial_A = og_tp @ 0, partial_B = dg_tp @ h*2)
        let c_xr = emit_xreduce(b, xgate, true, xr_cus, c_cmb, n.attn, h, tp, slot_b);
        b.emit(DevOp::Residual, one.clone(), &[c_xr], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.attn;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
    } else if tp > 1 && no_xr {
        // diagnostic: combine this rank's partials straight onto the residual, no all-reduce
        b.emit(DevOp::MoeCombine, all.clone(), &deps, |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
        })
    } else {
        b.emit(DevOp::MoeCombine, all.clone(), &deps, |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
        })
    }
}

/// Emit ONE DENSE (first_k_dense_replace) GLM decoder block — layers 0-2. The MLA attention is
/// identical to the MoE block; the FFN is a straight block-fp8 SwiGLU (no router/experts/shared):
/// DENSE_GLU_FP8_BLK (gate/up, H->dense_inter) -> GEMV_FP8_BLK (down, dense_inter->H) -> residual.
/// Returns the final residual completion dep (writes `n.xnext`).
pub(crate) fn emit_glm_dense_block(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    x_in: u32,
    x_out: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    assert!(slot < n.lw.len(), "slot out of range");
    let c_rn2 = emit_glm_mla(b, c, n, slot, ctx, x_in, pre, xgate, xr_cus);
    let all = b.all();
    let one = vec![0u32];
    let (h, di) = (c.hidden, c.dense_inter);
    let tp = c.tp;
    let di_l = di / tp; // this rank's dense-FFN intermediate lanes; tp==1 => di
    let w = &n.lw[slot];
    // dense SwiGLU gate|up (block-fp8, op 47) — column-parallel: this rank's di_l lanes
    let c_glu = b.emit(DevOp::DenseGluFp8Blk, all.clone(), &[c_rn2], |d| {
        d.t[0] = n.dfu;
        d.t[1] = n.xn2;
        d.t[2] = w.dgate;
        d.t[5] = w.dup;
        d.t[3] = w.dgate_s;
        d.t[4] = w.dup_s;
        d.i[0] = di_l;
        d.i[1] = h;
        d.i[5] = GLM_ACT_SILU;
    });
    // dense down (block-fp8 GEMV, op 44) — row-parallel (di_l input). Under TP writes a PARTIAL into
    //   the dg_tp peer slot, XReduce all-reduces into n.attn, then residual; at tp==1 writes n.shared
    //   and the residual reads it directly (byte-identical).
    let no_xr = tp > 1 && std::env::var("PLOW_NO_XREDUCE").ok().as_deref() == Some("1");
    if tp > 1 && !no_xr {
        let c_down = b.emit(DevOp::GemvFp8Blk, all.clone(), &[c_glu], |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = n.dfu;
            d.t[2] = w.ddown;
            d.t[5] = w.ddown_s;
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = di_l;
            d.i[4] = 0;
        });
        let slot_b = h * 2;
        let c_xr = emit_xreduce(b, xgate, true, xr_cus, c_down, n.attn, h, tp, slot_b);
        b.emit(DevOp::Residual, one.clone(), &[c_xr], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.attn;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
    } else if tp > 1 && no_xr {
        let c_down = b.emit(DevOp::GemvFp8Blk, all.clone(), &[c_glu], |d| {
            d.t[0] = n.shared;
            d.t[1] = n.dfu;
            d.t[2] = w.ddown;
            d.t[5] = w.ddown_s;
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = di_l;
            d.i[4] = 0;
        });
        b.emit(DevOp::Residual, one.clone(), &[c_down], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
    } else {
        let c_down = b.emit(DevOp::GemvFp8Blk, all.clone(), &[c_glu], |d| {
            d.t[0] = n.shared;
            d.t[1] = n.dfu;
            d.t[2] = w.ddown;
            d.t[5] = w.ddown_s;
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = di_l;
            d.i[4] = 0;
        });
        b.emit(DevOp::Residual, one.clone(), &[c_down], |d| {
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
fn glm_emit_full(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, use_fp8: bool, rope_gen: bool) {
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

    let mut tb = Builder::new(n_cu);
    let tn = declare_glm(&mut tb, &c, ctx, &layers);
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();
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
                use_fp8,
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
    let c_f = b.emit(DevOp::RmsNorm, vec![0u32], &[dep], |d| {
        d.t[0] = tn.xn;
        d.t[1] = cur;
        d.t[2] = tn.fin;
        d.i[0] = 1;
        d.i[1] = c.hidden;
        d.f[0] = c.eps;
    });
    let c_lm = b.emit(DevOp::Gemv, all.clone(), &[c_f], |d| {
        d.t[0] = tn.logits;
        d.t[1] = tn.xn;
        d.t[2] = tn.head;
        d.i[0] = 1;
        d.i[1] = c.vocab;
        d.i[2] = c.hidden;
        d.i[4] = 0;
    });
    let c_am = b.emit(DevOp::Argmax, (0..AMAX_BLOCKS).collect(), &[c_lm], |d| {
        d.t[0] = tn.amax;
        d.t[1] = tn.logits;
        d.i[0] = c.vocab;
    });
    b.emit(DevOp::ArgmaxFin, vec![0u32], &[c_am], |d| {
        d.t[0] = tn.ids;
        d.t[1] = tn.amax;
        d.i[0] = AMAX_BLOCKS;
    });
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
}

pub(crate) fn glm_main(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, rope_gen: bool) {
    let use_fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1");
    // Full 78-layer serving decode program (GLM_FULL=1) vs the single-layer validation gate (default).
    if std::env::var("GLM_FULL").ok().as_deref() == Some("1") {
        glm_emit_full(dir, ctx, out, n_cu, tp, use_fp8, rope_gen);
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
            use_fp8,
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
pub(crate) fn glm_build_block(
    c: &GlmCfg,
    ctx: u32,
    n_cu: u32,
    block: std::ops::Range<usize>,
    use_fp8: bool,
    model: &str,
    arch: MlaArch,
) -> (Model, plow_asset::BlockDescriptor) {
    use plow_asset::*;
    let layers: Vec<u32> = block.clone().map(|l| l as u32).collect();

    let mut tb = Builder::new(n_cu);
    let tn = declare_glm(&mut tb, c, ctx, &layers);
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();
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
            emit_glm_dense_block(&mut b, c, &tn, slot, ctx, cur, nxt, &dep, &mut xgate, &xr_cus)
        } else {
            emit_glm_block(
                &mut b, c, &tn, slot, ctx, use_fp8, cur, nxt, &dep, &mut xgate, &xr_cus,
            )
        };
        dep = vec![d];
        cur = nxt;
    }
    // After N layers the residual is back in `x` (even) or in `xnext` (odd).
    let out_name = if cur == tn.x { "act.x" } else { "act.xnext" };
    let prog = b.finish();
    let mut m = Model {
        n_cu,
        target: 0, // GLM/Kimi/DeepSeek/Nemotron: GPU fingerprint not threaded here yet
        tensors,
        progs: vec![prog],
        kv_row_insts: Vec::new(),
        prog_t: vec![1],
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
            prefix: format!("model.layers.{l0}."),
        },
        programs: BlockPrograms {
            prefill_buckets: Vec::new(), // GLM emit path is decode-only (M=1)
            decode_t: 1,
        },
    };
    (m, desc)
}

/// `--block` on the GLM (glm_moe_dsa) emit path. Emits ONE block (layers `spec`) as a
/// GPU-loadable PLOWDEV blob with a `SECT_METADATA` `block.json` descriptor + sibling
/// file — the GLM analogue of the gemma `--block` path (decode-only; the GLM emitter
/// has no prefill program).
pub(crate) fn glm_emit_block(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, spec: &str, rope_gen: bool) {
    let mut c = cfg_glm(dir);
    c.tp = tp;
    assert_eq!(
        tp, 1,
        "GLM TP sharding is milestone-3; use --tp 1 for --block extraction"
    );
    let use_fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1");
    let block = parse_block(spec, c.layers as usize);
    let model = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.to_string_lossy().into_owned());
    let (mut m, desc) = glm_build_block(&c, ctx, n_cu, block.clone(), use_fp8, &model, MlaArch::Glm);
    let section = write_block_descriptor(out, &desc);
    if !rope_gen {
        m.bake_gen();
    }
    std::fs::write(out, m.to_blob_v6(&[section])).unwrap();
    eprintln!(
        "glm52 --block {block:?}: {} block, {} layer(s), {} ops, dsa_role={} ctx={ctx} -> {out}",
        if use_fp8 { "block-fp8" } else { "bf16" },
        block.len(),
        m.progs[0].insts.len(),
        desc.dsa_role.as_deref().unwrap_or("-"),
    );
    eprintln!("  block.json sibling written next to {out}");
}

/// `--block` on the Kimi K2.7 / DeepSeek MLA+MoE path (plans/block-asset-harness.md §5.0/§5.3, M3).
/// Emits ONE block (layers `spec`) as a GPU-loadable PLOWDEV blob with a `SECT_METADATA` `block.json`
/// descriptor + sibling file. REUSES the GLM MLA + MoE emit verbatim (glm_build_block) with a Kimi
/// cfg (`has_dsa=false`) — no DSA, KV latent (ckv/krot) carried state, decode-only (the GLM emit has
/// no prefill program, so programs.prefill_buckets stays empty). `arch` picks the Kimi vs DeepSeek
/// descriptor tag.
pub(crate) fn kimi_emit_block(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, spec: &str, arch: MlaArch, rope_gen: bool) {
    let mut c = cfg_kimi(dir);
    c.tp = tp;
    assert_eq!(
        tp, 1,
        "Kimi/DeepSeek TP sharding is a later milestone; use --tp 1 for --block extraction"
    );
    let use_fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1");
    let block = parse_block(spec, c.layers as usize);
    let model = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.to_string_lossy().into_owned());
    let (mut m, desc) = glm_build_block(&c, ctx, n_cu, block.clone(), use_fp8, &model, arch);
    let section = write_block_descriptor(out, &desc);
    if !rope_gen {
        m.bake_gen();
    }
    std::fs::write(out, m.to_blob_v6(&[section])).unwrap();
    eprintln!(
        "{} --block {block:?}: {} block, {} layer(s), {} ops, ctx={ctx} -> {out}",
        desc.arch,
        if use_fp8 { "block-fp8" } else { "bf16" },
        block.len(),
        m.progs[0].insts.len(),
    );
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
    NemoCfg {
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
    }
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
    let mut m = Model {
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

// ===== tests moved from lib.rs (module breakdown): access mla internals directly =====
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
            moe_inter: 2048,
            dense_inter: 12288,
            first_k_dense: 3,
            route_scale: 2.5,
            attn_scale: (256f32).powf(-0.5),
            rope_theta: 8_000_000.0,
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
            use_fp8,
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
            true,
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
        assert!(d.programs.prefill_buckets.is_empty(), "GLM is decode-only");
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
        // Measured chain optima locked in (fill-permitting): ns=16 up to 8k, ns=64 at 32k.
        for &nh_l in &[8u32, 16] {
            assert_eq!(
                glm_nsplit(1024, nh_l),
                16,
                "nh_l={nh_l}: 1k optimum is ns=16"
            );
            assert_eq!(
                glm_nsplit(8192, nh_l),
                16,
                "nh_l={nh_l}: 8k optimum is ns=16"
            );
            assert_eq!(
                glm_nsplit(32768, nh_l),
                64,
                "nh_l={nh_l}: 32k optimum is ns=64"
            );
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
            moe_inter: 128,
            dense_inter: 256,
            first_k_dense: 1,
            route_scale: 2.5,
            attn_scale: (48f32).powf(-0.5), // 1/sqrt(qk_nope+qk_rope = 48)
            rope_theta: 50_000.0,
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
        assert!(
            d.programs.prefill_buckets.is_empty(),
            "GLM/Kimi emit is decode-only"
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
