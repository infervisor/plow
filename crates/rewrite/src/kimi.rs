//! Kimi K2 / DeepSeek V2-V3 decode block as a Path-A tile-IR [`LayerPlan`],
//! with typed compile-time vs runtime **gates** and a **partial-evaluator** that
//! folds the static gates.
//!
//! Architecturally these models are MLA (multi-head latent attention) + MoE,
//! almost identical — the config field names differ slightly but the graph shape
//! is the same. This module covers both.
//!
//! ## Op mapping (reuse-first — no new [`OpKind`])
//!
//! * **block-fp8 / bf16 GEMV** (all projections, experts, router score) →
//!   [`OpKind::Gemm`] with `m = 1`.
//! * **norm / rope / residual / router-topk** → [`OpKind::Row`].
//! * **MLA flash decode** → [`OpKind::Flash`].
//!
//! ## Gates
//!
//! [`StaticGate`]s (layer-type, TP, fp8) are compile-resolvable;
//! [`partial_eval`] folds them and emits the specialized op list.
//! [`DynGate::RouterTopK`] is data-dependent and survives as a runtime node.

use crate::tilegraph::{LayerPlan, OpKind, OpSpec};
use costmodel::{AttnShape, GemmShape, RowShape};
use nn_graph::DType;

// ─── Dimensions ─────────────────────────────────────────────────────────────

/// Architecture dimensions for Kimi K2 / DeepSeek V3.
/// Covers both models (they share the same field set).
#[derive(Clone, Copy, Debug)]
pub struct MlaMoeDims {
    pub hidden: i64,            // 7168
    pub heads: i64,             // 128
    pub q_lora_rank: i64,       // 1536
    pub kv_lora_rank: i64,      // 512
    pub qk_rope_head_dim: i64,  // 64
    pub qk_nope_head_dim: i64,  // 128
    pub v_head_dim: i64,        // 128
    pub n_routed_experts: i64,  // 256
    pub n_shared_experts: i64,  // 1
    pub num_experts_per_tok: i64, // 8
    pub moe_inter: i64,         // 2048
    pub dense_inter: i64,       // 18432
    pub vocab: i64,             // 160000 (Kimi) / 129280 (DeepSeek)
}

impl Default for MlaMoeDims {
    /// Kimi K2 / DeepSeek V3 default dimensions.
    fn default() -> Self {
        MlaMoeDims {
            hidden: 7168,
            heads: 128,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            qk_nope_head_dim: 128,
            v_head_dim: 128,
            n_routed_experts: 256,
            n_shared_experts: 1,
            num_experts_per_tok: 8,
            moe_inter: 2048,
            dense_inter: 18_432,
            vocab: 160_000,
        }
    }
}

// ─── Gates ──────────────────────────────────────────────────────────────────

/// FFN kind of a decoder layer.
/// First `first_k_dense_replace` layers are `Dense`; rest are `Moe`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Dense,
    Moe,
}

/// A compile-time-resolvable gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticGate {
    /// Dense vs MoE FFN.
    LayerType(LayerType),
    /// Tensor-parallel degree — shrinks head/expert/intermediate dims by `tp`.
    Tp(i64),
    /// Block-fp8 vs bf16 weight path for the projections/experts.
    Fp8(bool),
}

/// A data-dependent gate — survives `partial_eval`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DynGate {
    /// Router top-`top_k`-of-`n_exp` expert selection.
    RouterTopK { n_exp: i64, top_k: i64 },
}

/// A resolved static-gate assignment for one layer.
#[derive(Clone, Copy, Debug)]
pub struct MlaStatic {
    pub ctx: i64,
    pub layer: LayerType,
    pub tp: i64,
    pub fp8: bool,
}

/// A dynamic gate bound to the plan op it guards.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateBinding {
    pub gate: DynGate,
    /// The plan op that realizes the runtime selection.
    pub selector: String,
}

/// The partial-evaluated Kimi/DeepSeek block: a specialized [`LayerPlan`] plus
/// residual dynamic gates.
#[derive(Clone, Debug)]
pub struct MlaBlock {
    pub plan: LayerPlan,
    pub dyn_gates: Vec<GateBinding>,
}

// ─── Op constructors (M = 1 decode) ─────────────────────────────────────────

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

// ─── Partial evaluator ──────────────────────────────────────────────────────

/// Fold the static gates and emit the specialized Kimi/DeepSeek decode block.
///
/// `s.layer` / `s.tp` / `s.fp8` pick which ops appear and their shapes/dtypes —
/// the static branches vanish. Only [`DynGate::RouterTopK`] survives on MoE layers.
pub fn partial_eval(d: &MlaMoeDims, s: &MlaStatic) -> MlaBlock {
    let tp = s.tp.max(1);
    let nh_l = d.heads / tp;
    let (h, dk, dr, nope, vd, ql) = (
        d.hidden,
        d.kv_lora_rank,
        d.qk_rope_head_dim,
        d.qk_nope_head_dim,
        d.v_head_dim,
        d.q_lora_rank,
    );
    // fp8 weight path for the large projections/experts; norms/router are bf16.
    let wq = if s.fp8 { DType::F8E4M3 } else { DType::BF16 };

    let mut ops: Vec<OpSpec> = Vec::new();
    let mut dyn_gates: Vec<GateBinding> = Vec::new();

    // === MLA (shared by dense + MoE) ===
    ops.push(row("input_layernorm", &["x", "gin"], "xn", h, 2, true));

    // Query path: down-project → norm → absorb + rope
    ops.push(gemm("q_a_proj", &["xn", "qad"], "qlr", ql, h, wq));
    ops.push(row("q_a_layernorm", &["qlr", "gqa"], "qlat", ql, 2, true));
    ops.push(gemm("q_absorb", &["qlat", "wqa"], "qa", nh_l * nope, ql, wq));
    ops.push(gemm("q_rope_down", &["qlat", "wqr"], "qrr", nh_l * dr, ql, wq));
    ops.push(row("q_rope", &["qrr", "cos", "sin"], "qr", nh_l * dr, 3, false));

    // KV path: latent down-project → norm, + k_rope
    ops.push(gemm("kv_a_proj", &["xn", "ckvd"], "ckvraw", dk, h, wq));
    ops.push(row("kv_a_layernorm", &["ckvraw", "gkva"], "ckv", dk, 2, true));
    ops.push(gemm("k_rope_down", &["xn", "krotd"], "krr", dr, h, wq));
    ops.push(row("k_rope", &["krr", "cos", "sin"], "krot", dr, 3, false));

    // Flash decode: full context (no DSA in Kimi/DeepSeek)
    let head_dim = nope + dr; // MLA: content + rope
    ops.push(flash("flash", &["qa", "qr", "ckv", "krot"], "opart", nh_l, s.ctx, head_dim));

    // MLA merge + v_absorb fold → o_proj
    ops.push(gemm("mla_merge_fold", &["opart", "wuv"], "oat", nh_l * vd, dk, DType::BF16));
    ops.push(gemm("o_proj", &["oat", "wo"], "attn", h, nh_l * vd, wq));
    ops.push(row("residual", &["x", "attn"], "xmid", h, 2, false));
    ops.push(row("post_attention_layernorm", &["xmid", "gpost"], "xn2", h, 2, true));

    // === FFN (layer-type static gate) ===
    match s.layer {
        LayerType::Dense => {
            let di_l = d.dense_inter / tp;
            ops.push(gemm("dense_gate", &["xn2", "dgate"], "dg", di_l, h, wq));
            ops.push(gemm("dense_up", &["xn2", "dup"], "du", di_l, h, wq));
            ops.push(row("dense_swiglu", &["dg", "du"], "dfu", di_l, 2, false));
            ops.push(gemm("dense_down", &["dfu", "ddown"], "dout", h, di_l, wq));
            ops.push(row("dense_residual", &["xmid", "dout"], "xnext", h, 2, false));
        }
        LayerType::Moe => {
            let imoe_l = d.moe_inter / tp;
            // Router score → top-k select (DYNAMIC)
            ops.push(gemm("router_score", &["xn2", "wr"], "rlogit", d.n_routed_experts, h, DType::BF16));
            ops.push(row("router_topk", &["rlogit"], "tab", d.n_routed_experts, 1, true));
            dyn_gates.push(GateBinding {
                gate: DynGate::RouterTopK { n_exp: d.n_routed_experts, top_k: d.num_experts_per_tok },
                selector: "router_topk".into(),
            });
            // Shared expert(s) GLU + down
            let sh_inter = imoe_l * d.n_shared_experts;
            ops.push(gemm("shared_gate", &["xn2", "shg"], "shg_out", sh_inter, h, wq));
            ops.push(gemm("shared_up", &["xn2", "shu"], "shu_out", sh_inter, h, wq));
            ops.push(row("shared_swiglu", &["shg_out", "shu_out"], "shfu", sh_inter, 2, false));
            ops.push(gemm("shared_down", &["shfu", "shd"], "shared", h, sh_inter, wq));
            // Top-k routed experts — each processes the one M=1 token
            let mut part_names = Vec::new();
            for e in 0..d.num_experts_per_tok {
                let g_name = format!("expert{e}_gate");
                let u_name = format!("expert{e}_up");
                let glu_name = format!("expert{e}_swiglu");
                let down_name = format!("expert{e}_down");
                ops.push(gemm(&g_name, &["xn2", "tab", "ewg"], &format!("eg{e}"), imoe_l, h, wq));
                ops.push(gemm(&u_name, &["xn2", "tab", "ewu"], &format!("eu{e}"), imoe_l, h, wq));
                ops.push(row(&glu_name, &[&format!("eg{e}"), &format!("eu{e}")], &format!("fu{e}"), imoe_l, 2, false));
                ops.push(gemm(&down_name, &[&format!("fu{e}"), "tab", "ewd"], &format!("part{e}"), h, imoe_l, wq));
                part_names.push(format!("part{e}"));
            }
            // Combine: shared + Σ gate·expert + residual
            let mut cin: Vec<&str> = vec!["xmid", "shared"];
            for p in &part_names {
                cin.push(p);
            }
            ops.push(row("moe_combine", &cin, "xnext", h, 2 + d.num_experts_per_tok, false));
        }
    }

    MlaBlock { plan: LayerPlan { ops }, dyn_gates }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_static(layer: LayerType) -> MlaStatic {
        MlaStatic { ctx: 4096, layer, tp: 1, fp8: true }
    }

    #[test]
    fn moe_layer_has_router_gate() {
        let b = partial_eval(&MlaMoeDims::default(), &default_static(LayerType::Moe));
        assert_eq!(b.dyn_gates.len(), 1);
        assert_eq!(b.dyn_gates[0].selector, "router_topk");
        match b.dyn_gates[0].gate {
            DynGate::RouterTopK { n_exp, top_k } => {
                assert_eq!(n_exp, 256);
                assert_eq!(top_k, 8);
            }
        }
    }

    #[test]
    fn dense_layer_has_no_dynamic_gates() {
        let b = partial_eval(&MlaMoeDims::default(), &default_static(LayerType::Dense));
        assert!(b.dyn_gates.is_empty());
    }

    #[test]
    fn fp8_gate_sets_weight_dtype() {
        let d = MlaMoeDims::default();
        let b_fp8 = partial_eval(&d, &MlaStatic { ctx: 4096, layer: LayerType::Moe, tp: 1, fp8: true });
        let b_bf16 = partial_eval(&d, &MlaStatic { ctx: 4096, layer: LayerType::Moe, tp: 1, fp8: false });
        // q_a_proj should have FP8 weight dtype when fp8=true
        let qa_fp8 = b_fp8.plan.ops.iter().find(|o| o.name == "q_a_proj").unwrap();
        let qa_bf16 = b_bf16.plan.ops.iter().find(|o| o.name == "q_a_proj").unwrap();
        assert_eq!(qa_fp8.weight_dtype, DType::F8E4M3);
        assert_eq!(qa_bf16.weight_dtype, DType::BF16);
    }

    #[test]
    fn tp_shards_head_and_expert_dims() {
        let d = MlaMoeDims::default();
        let b1 = partial_eval(&d, &MlaStatic { ctx: 4096, layer: LayerType::Moe, tp: 1, fp8: true });
        let b2 = partial_eval(&d, &MlaStatic { ctx: 4096, layer: LayerType::Moe, tp: 2, fp8: true });
        // q_absorb N should halve with tp=2: heads/tp * nope_dim
        let qa1 = b1.plan.ops.iter().find(|o| o.name == "q_absorb").unwrap();
        let qa2 = b2.plan.ops.iter().find(|o| o.name == "q_absorb").unwrap();
        match (qa1.kind, qa2.kind) {
            (OpKind::Gemm(g1), OpKind::Gemm(g2)) => {
                assert_eq!(g1.n, 128 * 128); // 128 heads * 128 nope
                assert_eq!(g2.n, 64 * 128);  // 64 heads * 128 nope
            }
            _ => panic!("expected Gemm"),
        }
    }

    #[test]
    fn moe_op_count() {
        let d = MlaMoeDims::default();
        let b = partial_eval(&d, &default_static(LayerType::Moe));
        // MLA: norm + q_a_proj + q_a_norm + q_absorb + q_rope_down + q_rope +
        //      kv_a_proj + kv_a_norm + k_rope_down + k_rope +
        //      flash + merge_fold + o_proj + residual + post_norm = 15
        // MoE: router_score + router_topk + shared_gate + shared_up + shared_swiglu +
        //      shared_down + 8*(gate + up + swiglu + down) + combine = 6 + 32 + 1 = 39
        // Total: 15 + 39 = 54
        assert_eq!(b.plan.ops.len(), 54);
    }

    #[test]
    fn dense_op_count() {
        let d = MlaMoeDims::default();
        let b = partial_eval(&d, &default_static(LayerType::Dense));
        // MLA: 15 + Dense: gate + up + swiglu + down + residual = 5 → total 20
        assert_eq!(b.plan.ops.len(), 20);
    }

    #[test]
    fn flash_seq_kv_equals_ctx() {
        let d = MlaMoeDims::default();
        let b = partial_eval(&d, &MlaStatic { ctx: 8192, layer: LayerType::Dense, tp: 1, fp8: true });
        let fl = b.plan.ops.iter().find(|o| o.name == "flash").unwrap();
        match fl.kind {
            OpKind::Flash(a) => assert_eq!(a.seq_kv, 8192),
            _ => panic!("expected Flash"),
        }
    }

    #[test]
    fn assemble_produces_tile_graph() {
        use crate::assemble;
        use crate::tilegraph::TileNode;
        use costmodel::{Soc, SramPolicy, DEFAULT_PAGE_BYTES};

        fn mi350x() -> &'static costmodel::hwspec::GpuSpec {
            costmodel::hwspec::registry::lookup("MI350X").unwrap()
        }

        let d = MlaMoeDims::default();
        let b = partial_eval(&d, &default_static(LayerType::Moe));
        let soc = Soc::single(mi350x(), DEFAULT_PAGE_BYTES);
        let (g, cons) = assemble(&soc, &b.plan, SramPolicy::Stream, None).unwrap();
        // Should have at least as many compute nodes as ops
        let compute_count = g.nodes.iter().filter(|n| {
            matches!(n, TileNode::Compute { .. })
        }).count();
        assert!(compute_count >= b.plan.ops.len(), "too few compute nodes: {compute_count}");
        // Constraint set should have placements for every compute
        assert!(cons.placement.len() >= b.plan.ops.len());
    }
}
