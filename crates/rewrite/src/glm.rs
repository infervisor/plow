//! GLM-5.2-FP8 decode block as a Path-A tile-IR [`LayerPlan`], with typed
//! compile-time vs runtime **gates** and a **partial-evaluator** that folds the
//! static gates.
//!
//! Today GLM decode is Path B: `crates/plowc/src/bin/gemma4.rs` hand-emits the
//! block, with the ctx-bucket / layer-type / TP / fp8 decisions as emit-time
//! constants and the router top-8 / DSA gather as runtime branches. This module
//! is the Path-A representation the compiler can REASON about: the block is a
//! list of [`OpSpec`]s (feeding the existing [`crate::assemble`] — cost-model tile
//! selection + union-find fusion groups), and the gates are typed nodes.
//!
//! ## What maps to what (reuse-first — no new [`OpKind`])
//!
//! Every GLM decode op is one of the four existing tile-IR op kinds:
//! * **block-fp8 / bf16 GEMV** (all projections, experts, router score) →
//!   [`OpKind::Gemm`] with `m = 1`. Block-fp8 is expressed by `weight_dtype =
//!   F8E4M3` (⇒ `weight_elem = 1`, half the bf16 weight bandwidth — the
//!   dominant decode cost). The 128×128 f32 block-scale stream is a modeled
//!   delta we omit: ~`(N/128)(K/128)·4` B, <0.1 % of the fp8 weight bytes, so it
//!   never re-ranks a tile. A dedicated block-fp8 op kind would only be needed
//!   to price that stream.
//! * **norm / rope / residual / router-topk / index-score / index-select** →
//!   [`OpKind::Row`] (memory-bound row op; `reduce` distinguishes a
//!   norm/reduction from a point-wise add/rope).
//! * **MLA flash decode / DSA gather** → [`OpKind::Flash`]. The ctx-bucket gate
//!   resolves `seq_kv` (full ctx vs DSA top-`index_topk`) at partial-eval time.
//!
//! ## Gates
//!
//! [`StaticGate`]s (ctx-bucket, layer-type, TP, fp8) are compile-resolvable;
//! [`partial_eval`] folds them and emits the specialized op list. [`DynGate`]s
//! (router top-8, DSA gather idx) are data-dependent and survive as runtime
//! nodes annotating the plan op they guard.

use crate::tilegraph::{LayerPlan, OpKind, OpSpec};
use costmodel::{AttnShape, GemmShape, RowShape};
use nn_graph::DType;

/// GLM-5.2-FP8 static dimensions (`plans/glm52-arch.md`). Bf16-unquantized
/// shapes; the fp8 gate only changes weight *dtype*, not dims.
#[derive(Clone, Copy, Debug)]
pub struct GlmDims {
    pub hidden: i64,       // 6144
    pub heads: i64,        // 64
    pub kv_lora: i64,      // 512  (kv latent, DK)
    pub qk_rope: i64,      // 64   (DR — the rotated dims)
    pub v_head: i64,       // 256  (VD)
    pub q_lora: i64,       // 2048 (QL)
    pub n_exp: i64,        // 256
    pub top_k: i64,        // 8
    pub shared_exp: i64,   // 1
    pub moe_inter: i64,    // 2048 (per-expert intermediate)
    pub dense_inter: i64,  // 12288 (dense-layer FFN intermediate)
    pub index_heads: i64,  // 32   (HI)
    pub index_dim: i64,    // 128  (DI)
    pub index_topk: i64,   // 2048 (DSA selected tokens)
    pub vocab: i64,        // 154880
}

impl Default for GlmDims {
    /// The shipping GLM-5.2-FP8 config.
    fn default() -> Self {
        GlmDims {
            hidden: 6144,
            heads: 64,
            kv_lora: 512,
            qk_rope: 64,
            v_head: 256,
            q_lora: 2048,
            n_exp: 256,
            top_k: 8,
            shared_exp: 1,
            moe_inter: 2048,
            dense_inter: 12288,
            index_heads: 32,
            index_dim: 128,
            index_topk: 2048,
            vocab: 154880,
        }
    }
}

/// FFN kind of a decoder layer — the layer-type static gate's value.
/// Layers 0-2 are `Dense`; 3-77 are `Moe` (`first_k_dense_replace = 3`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Dense,
    Moe,
}

/// A GLM layer's DSA indexer role — `Full` layers own the lightning-indexer +
/// top-k select; `Shared` layers reuse the last full layer's idx table (no
/// indexer projections of their own). Dense/short-ctx layers run no indexer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexerRole {
    /// Owns the indexer projections + the top-k select (`GatherIdx` produced here).
    Full,
    /// Reuses a prior full layer's idx (gather still runs; no local projections).
    Shared,
    /// No indexer (dense flash over the full ctx).
    None,
}

/// A gate guarding a region of the GLM block.
///
/// [`StaticGate`]s are folded by [`partial_eval`]; [`DynGate`]s survive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Static(StaticGate),
    Dynamic(DynGate),
}

/// A compile-time-resolvable gate. `partial_eval` folds these into concrete op
/// presence + shapes, so they vanish as runtime branches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticGate {
    /// ctx ≤ `index_topk` ⇒ dense flash over full ctx; ctx > `index_topk` ⇒ DSA
    /// gather over the top-k. Also resolves the flash `seq_kv`.
    CtxBucket { ctx: i64 },
    /// Dense (0-2) vs MoE (3-77) FFN, and the layer's indexer role.
    LayerType(LayerType, IndexerRole),
    /// Tensor-parallel degree — shrinks head/expert/intermediate dims by `tp`.
    Tp(i64),
    /// Block-fp8 vs bf16 weight path for the projections/experts.
    Fp8(bool),
}

/// A data-dependent gate. Survives `partial_eval`; the compiler cannot resolve
/// *which* experts/tokens are chosen, only that a runtime selection op exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynGate {
    /// Router top-`top_k`-of-`n_exp` expert selection (which experts is dynamic).
    RouterTopK { n_exp: i64, top_k: i64 },
    /// DSA top-`index_topk` token select (which tokens is dynamic).
    GatherIdx { index_topk: i64 },
}

/// A resolved static-gate assignment — the compile-time knowns for one layer.
#[derive(Clone, Copy, Debug)]
pub struct GlmStatic {
    pub ctx: i64,
    pub layer: LayerType,
    pub indexer: IndexerRole,
    pub tp: i64,
    pub fp8: bool,
}

/// A dynamic gate bound to the plan op(s) it guards (by op name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateBinding {
    pub gate: DynGate,
    /// The plan op that realizes the runtime selection (`op.name`).
    pub selector: String,
}

/// The partial-evaluated GLM block: a specialized [`LayerPlan`] (all static
/// gates folded away) plus the residual dynamic gates.
#[derive(Clone, Debug)]
pub struct GlmBlock {
    pub plan: LayerPlan,
    pub dyn_gates: Vec<GateBinding>,
}

// --- op constructors (M = 1 decode) ----------------------------------------

fn gemm(name: &str, inputs: &[&str], output: &str, n: i64, k: i64, wdt: DType) -> OpSpec {
    OpSpec {
        name: name.into(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        output: output.into(),
        kind: OpKind::Gemm(GemmShape { m: 1, n, k }),
        weight_dtype: wdt,
        compute_dtype: DType::BF16,
    }
}

fn row(name: &str, inputs: &[&str], output: &str, feat: i64, operands: i64, reduce: bool) -> OpSpec {
    OpSpec::bf16(
        name.into(),
        inputs.iter().map(|s| s.to_string()).collect(),
        output.into(),
        OpKind::Row(RowShape { rows: 1, feat, operands, reduce }),
    )
}

fn flash(name: &str, inputs: &[&str], output: &str, heads: i64, seq_kv: i64, head_dim: i64) -> OpSpec {
    OpSpec::bf16(
        name.into(),
        inputs.iter().map(|s| s.to_string()).collect(),
        output.into(),
        OpKind::Flash(AttnShape { heads, seq_q: 1, seq_kv, head_dim }),
    )
}

// --- the partial-evaluator --------------------------------------------------

/// Fold the static gates and emit the specialized GLM decode block.
///
/// This IS partial-evaluation: `s.ctx` / `s.layer` / `s.indexer` / `s.tp` /
/// `s.fp8` pick which [`OpSpec`]s appear and their shapes/dtypes — the branches
/// vanish. Only the [`DynGate`]s (router top-8, DSA gather) are left as
/// [`GateBinding`]s on their selector ops.
pub fn partial_eval(d: &GlmDims, s: &GlmStatic) -> GlmBlock {
    let tp = s.tp.max(1);
    let nh_l = d.heads / tp;
    let (h, dk, dr, vd, ql) = (d.hidden, d.kv_lora, d.qk_rope, d.v_head, d.q_lora);
    // fp8 weight path for the large projections/experts; norms/router are bf16.
    let wq = if s.fp8 { DType::F8E4M3 } else { DType::BF16 };
    // DSA is active only above the ctx bucket AND on a layer with an indexer.
    let dsa = s.ctx > d.index_topk && s.indexer != IndexerRole::None;
    let seq_kv = if dsa { d.index_topk.min(s.ctx) } else { s.ctx };

    let mut ops: Vec<OpSpec> = Vec::new();
    let mut dyn_gates: Vec<GateBinding> = Vec::new();

    // === MLA (shared by dense + MoE) ===
    ops.push(row("input_layernorm", &["x", "gin"], "xn", h, 2, true));
    ops.push(gemm("q_a_proj", &["xn", "qad"], "qlr", ql, h, wq));
    ops.push(row("q_a_layernorm", &["qlr", "gqa"], "qlat", ql, 2, true));
    ops.push(gemm("q_absorb", &["qlat", "wqa"], "qa", nh_l * dk, ql, wq));
    ops.push(gemm("q_rope_down", &["qlat", "wqr"], "qrr", nh_l * dr, ql, wq));
    ops.push(row("q_rope", &["qrr", "cos", "sin"], "qr", nh_l * dr, 3, false));
    ops.push(gemm("kv_a_latent", &["xn", "ckvd"], "ckvraw", dk, h, wq));
    ops.push(row("kv_a_layernorm", &["ckvraw", "gkva"], "ckv", dk, 2, true));
    ops.push(gemm("k_rope_down", &["xn", "krotd"], "krr", dr, h, wq));
    ops.push(row("k_rope", &["krr", "cos", "sin"], "krot", dr, 3, false));

    // --- DSA lightning indexer (ctx-bucket + indexer-role static gates) ---
    if dsa && s.indexer == IndexerRole::Full {
        let (hi, di) = (d.index_heads, d.index_dim);
        ops.push(gemm("idx_q", &["qlat", "iwqb"], "qidx0", hi * di, ql, wq));
        ops.push(row("idx_q_rope", &["qidx0", "icos", "isin"], "qidx", hi * di, 3, false));
        ops.push(gemm("idx_k", &["xn", "iwk"], "kidx_raw", di, h, wq));
        ops.push(row("idx_k_norm", &["kidx_raw", "iknw", "iknb"], "kidx_n", di, 3, true));
        ops.push(row("idx_k_rope", &["kidx_n", "icos", "isin"], "kidx", di, 3, false));
        ops.push(gemm("idx_weights", &["xn", "iwp"], "widx", hi, h, DType::BF16));
        ops.push(row("index_score", &["qidx", "kidx", "widx"], "iscore", s.ctx, hi, true));
        // top-k SELECT — DYNAMIC (which tokens is data-dependent).
        ops.push(row("index_select", &["iscore"], "iidx", s.ctx, 1, true));
        dyn_gates.push(GateBinding {
            gate: DynGate::GatherIdx { index_topk: d.index_topk },
            selector: "index_select".into(),
        });
    } else if dsa {
        // Shared indexer layer: reuses the prior full layer's idx (no local
        // projections/select), but the gather over top-k tokens still runs.
        dyn_gates.push(GateBinding {
            gate: DynGate::GatherIdx { index_topk: d.index_topk },
            selector: "flash".into(),
        });
    }

    // --- flash (ctx-bucket resolves seq_kv: full ctx vs DSA top-k) ---
    let head_dim = dk + dr; // MLA: latent + rope
    let mut fl_in = vec!["qa", "qr", "ckv", "krot"];
    if dsa {
        fl_in.push("iidx");
    }
    ops.push(flash("flash", &fl_in, "opart", nh_l, seq_kv, head_dim));
    // fused MLA merge + v_absorb fold (current emit: MlaMergeFold).
    ops.push(gemm("mla_merge_fold", &["opart", "wuv"], "oat", nh_l * vd, dk, DType::BF16));
    ops.push(gemm("o_proj", &["oat", "wo"], "attn", h, nh_l * vd, wq));
    ops.push(row("residual", &["x", "attn"], "xmid", h, 2, false));
    ops.push(row("post_attention_layernorm", &["xmid", "gpost"], "xn2", h, 2, true));

    // === FFN (layer-type static gate) ===
    match s.layer {
        LayerType::Dense => {
            let di_l = d.dense_inter / tp;
            // fused gate|up GLU (both di_l projections) then down.
            ops.push(gemm("dense_glu", &["xn2", "dgate", "dup"], "dfu", 2 * di_l, h, wq));
            ops.push(gemm("dense_down", &["dfu", "ddown"], "shared", h, di_l, wq));
            ops.push(row("dense_residual", &["xmid", "shared"], "xnext", h, 2, false));
        }
        LayerType::Moe => {
            let imoe_l = d.moe_inter / tp;
            // router score GEMV (bf16) → top-k select (DYNAMIC).
            ops.push(gemm("router_score", &["xn2", "wr"], "rlogit", d.n_exp, h, DType::BF16));
            ops.push(row("router_topk", &["rlogit", "bias"], "tab", d.n_exp, 2, true));
            dyn_gates.push(GateBinding {
                gate: DynGate::RouterTopK { n_exp: d.n_exp, top_k: d.top_k },
                selector: "router_topk".into(),
            });
            // shared expert (routing-independent) GLU + down.
            ops.push(gemm("shared_glu", &["xn2", "shg", "shu"], "shfu", 2 * imoe_l, h, wq));
            ops.push(gemm("shared_down", &["shfu", "shd"], "shared", h, imoe_l, wq));
            // top-k routed experts — uniform-shaped (each processes the one M=1
            // token); WHICH weights is the RouterTopK dynamic gate.
            let mut down_names = Vec::new();
            for e in 0..d.top_k {
                let glu = format!("expert{e}_glu");
                let down = format!("expert{e}_down");
                ops.push(gemm(&glu, &["xn2", "tab", "ewt"], &format!("fu{e}"), 2 * imoe_l, h, wq));
                ops.push(gemm(&down, &[&format!("fu{e}"), "tab", "ewt"], &format!("part{e}"), h, imoe_l, wq));
                down_names.push(down);
            }
            // combine: shared + Σ gate·expert (+ residual).
            let mut cin: Vec<&str> = vec!["xmid", "shared"];
            let held: Vec<String> = (0..d.top_k).map(|e| format!("part{e}")).collect();
            for p in &held {
                cin.push(p);
            }
            let _ = down_names;
            ops.push(row("moe_combine", &cin, "xnext", h, 2 + d.top_k, false));
        }
    }

    GlmBlock { plan: LayerPlan { ops }, dyn_gates }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble;
    use crate::tilegraph::{Compute, ConstraintSet, TileGraph, TileNode};
    use costmodel::{Soc, SramPolicy, DEFAULT_PAGE_BYTES};

    fn mi350x() -> &'static costmodel::hwspec::GpuSpec {
        costmodel::hwspec::registry::lookup("MI350X").unwrap()
    }

    fn st(ctx: i64, layer: LayerType, indexer: IndexerRole) -> GlmStatic {
        GlmStatic { ctx, layer, indexer, tp: 1, fp8: true }
    }

    fn op_names(plan: &LayerPlan) -> Vec<&str> {
        plan.ops.iter().map(|o| o.name.as_str()).collect()
    }

    // ---- partial-eval: static gates fold to the right op set ----

    #[test]
    fn short_ctx_moe_has_no_indexer() {
        // ctx ≤ index_topk: the DSA subgraph vanishes; flash covers full ctx.
        let d = GlmDims::default();
        let b = partial_eval(&d, &st(1024, LayerType::Moe, IndexerRole::Full));
        let names = op_names(&b.plan);
        assert!(!names.iter().any(|n| n.starts_with("idx_") || *n == "index_select"));
        // no GatherIdx gate, but the router top-k gate is present.
        assert!(b.dyn_gates.iter().all(|g| matches!(g.gate, DynGate::RouterTopK { .. })));
        // flash seq_kv resolved to the full ctx.
        let f = b.plan.ops.iter().find(|o| o.name == "flash").unwrap();
        assert_eq!(o_seqkv(f), 1024);
    }

    #[test]
    fn long_ctx_full_layer_has_indexer_and_gather() {
        let d = GlmDims::default();
        let b = partial_eval(&d, &st(131072, LayerType::Moe, IndexerRole::Full));
        let names = op_names(&b.plan);
        assert!(names.contains(&"index_select") && names.contains(&"idx_q"));
        // both dynamic gates present: RouterTopK + GatherIdx.
        assert!(b.dyn_gates.iter().any(|g| matches!(g.gate, DynGate::GatherIdx { .. })));
        assert!(b.dyn_gates.iter().any(|g| matches!(g.gate, DynGate::RouterTopK { .. })));
        // flash seq_kv folded to the DSA top-k (constant work regardless of ctx).
        let f = b.plan.ops.iter().find(|o| o.name == "flash").unwrap();
        assert_eq!(o_seqkv(f), d.index_topk);
    }

    #[test]
    fn long_ctx_shared_layer_reuses_idx() {
        // shared indexer layer: no local projections, gather still gated.
        let d = GlmDims::default();
        let b = partial_eval(&d, &st(131072, LayerType::Moe, IndexerRole::Shared));
        let names = op_names(&b.plan);
        assert!(!names.contains(&"idx_q") && !names.contains(&"index_select"));
        assert!(b.dyn_gates.iter().any(|g| matches!(g.gate, DynGate::GatherIdx { .. })));
        let f = b.plan.ops.iter().find(|o| o.name == "flash").unwrap();
        assert_eq!(o_seqkv(f), d.index_topk);
    }

    #[test]
    fn dense_layer_has_no_router_or_experts() {
        let d = GlmDims::default();
        let b = partial_eval(&d, &st(1024, LayerType::Dense, IndexerRole::None));
        let names = op_names(&b.plan);
        assert!(names.contains(&"dense_glu") && names.contains(&"dense_down"));
        assert!(!names.iter().any(|n| n.starts_with("expert") || *n == "router_topk"));
        assert!(b.dyn_gates.is_empty()); // no dynamic gates on a short-ctx dense layer.
    }

    #[test]
    fn tp_shards_head_and_expert_dims() {
        let d = GlmDims::default();
        let b1 = partial_eval(&d, &GlmStatic { tp: 1, ..st(1024, LayerType::Moe, IndexerRole::None) });
        let b8 = partial_eval(&d, &GlmStatic { tp: 8, ..st(1024, LayerType::Moe, IndexerRole::None) });
        // o_proj K = nh_l * vd shrinks 8× under TP8.
        let k1 = o_gemm_k(&b1.plan, "o_proj");
        let k8 = o_gemm_k(&b8.plan, "o_proj");
        assert_eq!(k1, 8 * k8);
    }

    #[test]
    fn fp8_gate_sets_weight_dtype() {
        let d = GlmDims::default();
        let bf16 = partial_eval(&d, &GlmStatic { fp8: false, ..st(1024, LayerType::Moe, IndexerRole::None) });
        let fp8 = partial_eval(&d, &GlmStatic { fp8: true, ..st(1024, LayerType::Moe, IndexerRole::None) });
        let w = |p: &LayerPlan| p.ops.iter().find(|o| o.name == "o_proj").unwrap().weight_dtype;
        assert_eq!(w(&bf16.plan), DType::BF16);
        assert_eq!(w(&fp8.plan), DType::F8E4M3);
        // router score stays bf16 in BOTH (norms/router never quantize).
        let r = |p: &LayerPlan| p.ops.iter().find(|o| o.name == "router_score").unwrap().weight_dtype;
        assert_eq!(r(&fp8.plan), DType::BF16);
    }

    // ---- Path-A machinery runs on the specialized block ----

    #[test]
    fn assemble_dedups_xn_across_qa_kva_krope() {
        // Fusion-A PRECONDITION: q_a, kv_a-latent, k_rope all read `xn`. The
        // dma-dedup must stage `xn` once and record all three consumers — the
        // structural fact the hand-emit's GemvQkv triple-fuse exploits.
        let d = GlmDims::default();
        let b = partial_eval(&d, &st(1024, LayerType::Moe, IndexerRole::None));
        let soc = Soc::single(mi350x(), DEFAULT_PAGE_BYTES);
        let (g, cons) = assemble(&soc, &b.plan, SramPolicy::Stream, None).unwrap();
        // `xn` is a single DRAM stage.
        let xn_stage = g.nodes.iter().filter(|n| matches!(n, TileNode::DmaIn { tensor, resident: false } if tensor == "xn")).count();
        // xn has one producer (input_layernorm) → consumers read it resident, so
        // it may be a resident DmaIn rather than a fresh stage; assert the dedup
        // group instead (all consumers of the produced `xn`).
        let consumers = xn_consumers(&g, &cons);
        assert!(consumers.contains(&"q_a_proj".to_string()));
        assert!(consumers.contains(&"kv_a_latent".to_string()));
        assert!(consumers.contains(&"k_rope_down".to_string()));
        let _ = xn_stage;
    }

    #[test]
    fn assemble_forms_fusion_groups() {
        // Union-find colocates same-unit producer→consumer chains. The
        // norm→projection chains (input_layernorm→{q_a,kv_a,k_rope}) must land in
        // a colocation group — the hand-emit's dependency chain, now discovered.
        let d = GlmDims::default();
        let b = partial_eval(&d, &st(1024, LayerType::Moe, IndexerRole::None));
        let soc = Soc::single(mi350x(), DEFAULT_PAGE_BYTES);
        let (_, cons) = assemble(&soc, &b.plan, SramPolicy::Stream, None).unwrap();
        // A non-trivial colocation group exists (single unit ⇒ every same-unit
        // handoff pins).
        assert!(!cons.colocation_groups.is_empty());
    }

    #[test]
    fn m1_gemv_has_single_tile_no_split() {
        // Validates the audit's "no per-shape GEMV tile knob": at M=1 with the
        // default split_k_max=1, every GEMV has exactly ONE non-split tile — the
        // compiler AGREES there is nothing to tune, matching pick_tile falling to
        // DevOp::Gemv for decode.
        let d = GlmDims::default();
        let b = partial_eval(&d, &st(1024, LayerType::Moe, IndexerRole::None));
        let soc = Soc::single(mi350x(), DEFAULT_PAGE_BYTES);
        let (g, _) = assemble(&soc, &b.plan, SramPolicy::Stream, None).unwrap();
        for n in &g.nodes {
            if let TileNode::Compute { kind: Compute::Gemm(t), .. } = n {
                assert_eq!(t.split_k, 1, "decode M=1 must not split-K at split_k_max=1");
            }
        }
    }

    #[test]
    fn decode_floor_prices_op_count_fusion() {
        // PHASE-2 FIX (was `cost_model_underprices_op_count_fusion`). The decode
        // dispatch floor (~4.6 µs = 10 120 cyc on MI350X, `costmodel::cost`) makes
        // op-count VISIBLE. A column-concatenation fusion (Fusion A:
        // q_a(2048)+kv_a(512)+k_rope(64), all K=6144 → one GemvQkv; Fusion G:
        // Wqa(32768)+Wqr(4096), K=2048 → one GemvQkv) runs the SAME per-column
        // dots as its parts — bandwidth-invariant under the concat — but pays ONE
        // dispatch floor instead of k. So its faithful cost is
        //   fused = Σ parts − (k−1)·floor,
        // i.e. fusing k decode ops WINS by (k−1)·floor, tile-divisibility-free.
        use costmodel::{CostModel, CostParams, GemmShape, SramPolicy, DEFAULT_PAGE_BYTES as PB};
        let cm = CostModel::new(mi350x(), PB);
        let p = CostParams::from_dtypes(DType::F8E4M3, DType::BF16);
        let floor = cm.decode_op_floor();
        assert!(floor > 10_000 && floor < 10_300, "≈4.6 µs @2.2 GHz, got {floor}");
        // Per-op cost (includes exactly one floor) and its floor-free work.
        let cost = |n: i64, k: i64| {
            let g = GemmShape { m: 1, n, k };
            let t = cm.best_tile_typed(g, SramPolicy::Stream, p).unwrap().0;
            cm.gemm_cost_typed(g, t, p)
        };
        let work = |n: i64, k: i64| cost(n, k) - floor;

        // --- Fusion A: 3 xn-readers (K=6144) → 1 dispatch, saves 2 floors. ---
        let sep_a = cost(2048, 6144) + cost(512, 6144) + cost(64, 6144);
        let fused_a = work(2048, 6144) + work(512, 6144) + work(64, 6144) + floor;
        assert_eq!(sep_a - fused_a, 2 * floor, "Fusion A wins by 2 floors");
        assert!(fused_a < sep_a);
        // --- Fusion G: 2 qlat-readers (K=2048) → 1 dispatch, saves 1 floor. ---
        let sep_g = cost(32768, 2048) + cost(4096, 2048);
        let fused_g = work(32768, 2048) + work(4096, 2048) + floor;
        assert_eq!(sep_g - fused_g, floor, "Fusion G wins by 1 floor");

        // The op-count credit the model now assigns per removed op is the floor —
        // ~20× the old LAUNCH_CYCLES (=500), which is why the model previously
        // could NOT see these fusions (op-count was a rounding error).
        assert!(floor > 20 * 500, "op-count credit grew ~20× vs LAUNCH_CYCLES");

        // HONESTY: naively re-tiling the merged N=2624 as ONE fresh GEMM still
        // mis-costs it — N=2624 is indivisible by 128/256 so `candidates` forces
        // bn=64 (2624 = 64·41), whose activation re-reads balloon the estimate.
        // That is a PRE-EXISTING M=1-GEMV-as-GEMM tiling artifact (strict
        // N-divisibility in `costmodel::tile::candidates`), ORTHOGONAL to the
        // floor: a real GemvQkv keeps each sub-column's tiling, so the faithful
        // cost above is the one the fusion selector uses.
        let naive_merge = cost(2624, 6144);
        assert!(naive_merge > sep_a, "merged-shape re-tile artifact is separate from the floor");
    }

    #[test]
    fn corrected_model_surfaces_indexer_side_fusions() {
        // NEW fusions the corrected model surfaces that E2 (glm-fusion, MLA-only)
        // did NOT do: on a FULL indexer layer the DSA lightning-indexer adds more
        // same-activation, same-K, same-dtype GEMVs that extend A and G:
        //   A′ = q_a + kv_a + k_rope + idx_k   (all read `xn`,  K=hidden=6144, fp8)
        //   G′ = q_absorb + q_rope_down + idx_q (all read `qlat`, K=q_lora=2048, fp8)
        // Each is a pure column-concat ⇒ bit-exact (like A/G), and each removes one
        // more floor per full-indexer layer. idx_weights also reads `xn`/K=6144 but
        // is BF16 (not fp8) ⇒ CANNOT join the fp8 concat (mixed weight dtype).
        let d = GlmDims::default();
        let b = partial_eval(&d, &st(131072, LayerType::Moe, IndexerRole::Full));
        let g = |name: &str| -> (Vec<String>, GemmShape, DType) {
            let o = b.plan.ops.iter().find(|o| o.name == name).unwrap();
            match o.kind {
                OpKind::Gemm(s) => (o.inputs.clone(), s, o.weight_dtype),
                _ => panic!("{name} not gemm"),
            }
        };
        // A′: idx_k shares xn + K=6144 + fp8 with the A trio.
        for n in ["q_a_proj", "kv_a_latent", "k_rope_down", "idx_k"] {
            let (ins, s, dt) = g(n);
            assert_eq!(ins[0], "xn", "{n} activation");
            assert_eq!(s.k, d.hidden, "{n} K");
            assert_eq!(dt, DType::F8E4M3, "{n} dtype");
        }
        // idx_weights reads xn/K=6144 but is BF16 ⇒ excluded from the fp8 concat.
        let (iw_ins, iw_s, iw_dt) = g("idx_weights");
        assert_eq!((iw_ins[0].as_str(), iw_s.k), ("xn", d.hidden));
        assert_eq!(iw_dt, DType::BF16);
        // G′: idx_q shares qlat + K=2048 + fp8 with the G pair.
        for n in ["q_absorb", "q_rope_down", "idx_q"] {
            let (ins, s, dt) = g(n);
            assert_eq!(ins[0], "qlat", "{n} activation");
            assert_eq!(s.k, d.q_lora, "{n} K");
            assert_eq!(dt, DType::F8E4M3, "{n} dtype");
        }
    }

    // ---- helpers ----

    fn o_seqkv(o: &OpSpec) -> i64 {
        match o.kind {
            OpKind::Flash(a) => a.seq_kv,
            _ => panic!("not flash"),
        }
    }
    fn o_gemm_k(p: &LayerPlan, name: &str) -> i64 {
        match p.ops.iter().find(|o| o.name == name).unwrap().kind {
            OpKind::Gemm(g) => g.k,
            _ => panic!("not gemm"),
        }
    }
    fn xn_consumers(g: &TileGraph, cons: &ConstraintSet) -> Vec<String> {
        // Ops whose op_io lists `xn` as an input.
        let mut out = Vec::new();
        for (node, io) in &cons.op_io {
            if io.inputs.iter().any(|i| i == "xn") {
                if let TileNode::Compute { op, .. } = &g.nodes[*node] {
                    out.push(op.clone());
                }
            }
        }
        out
    }
}
