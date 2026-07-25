//! End-to-end: build a Qwen3.5-4B transformer block as an `nn_graph` graph,
//! infer its shapes at a concrete sequence bucket, bridge it to a tiling
//! `LayerPlan`, and assemble the per-layer tile graph + constraint set.
//!
//! This exercises the whole front half of the compiler on one realistic decoder
//! layer: typed IR → shape inference → op→tile bridge → tile-graph assembly,
//! including the cross-op *tile* dependencies (RMSNorm → projection M-coupling)
//! the scheduler consumes.

use costmodel::{DEFAULT_PAGE_BYTES, Soc, SramPolicy};
use nn_graph::{infer_shapes, ActKind, DType, Nn};
use rewrite::{
    assemble, consumer_thresholds, materialize_tile_deps, plan_from_block, plan_from_fused,
    rewrite_graph, Compute, GraphNode, TileDomain,
};
use std::collections::HashMap;

/// Count compute nodes of a given tile-graph by their `Compute` kind predicate.
fn count_computes(tg: &rewrite::TileGraph, pred: impl Fn(&Compute) -> bool) -> usize {
    tg.nodes
        .iter()
        .filter(|n| matches!(n, GraphNode::Compute { kind, .. } if pred(kind)))
        .count()
}

// Qwen3-4B dense decoder config (the "3.5-4B" generation): GQA attention with
// per-head QK-norm + RoPE, SwiGLU MLP.
const H: i64 = 2560; // hidden size
const NH: i64 = 32; // attention heads
const NKV: i64 = 8; // key/value heads (GQA)
const HD: i64 = 128; // head dim
const QD: i64 = NH * HD; // 4096, query projection out
const KVD: i64 = NKV * HD; // 1024, k/v projection out
const IM: i64 = 9728; // MLP intermediate size
const T: i64 = 2048; // sequence bucket (tokens)

fn h100() -> &'static costmodel::hwspec::GpuSpec {
    costmodel::hwspec::registry::lookup("H100 SXM5").unwrap()
}

/// One Qwen3 decoder block over `x: [T, H]`, returned shape-inferred.
fn qwen_block() -> nn_graph::Graph {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let x = nn.input("x", nn.shape([T.into(), H.into()]), DType::BF16);

    nn.begin_block("layers.0");
    // --- attention ---
    let h1 = nn.rmsnorm("input_norm", x, H, 1e-6);
    let q = nn.linear("q_proj", h1, H, QD, false);
    let k = nn.linear("k_proj", h1, H, KVD, false);
    let v = nn.linear("v_proj", h1, H, KVD, false);
    // split heads (token-major: [T, heads, head_dim])
    let qh = nn.reshape(q, [T.into(), NH.into(), HD.into()]);
    let kh = nn.reshape(k, [T.into(), NKV.into(), HD.into()]);
    let vh = nn.reshape(v, [T.into(), NKV.into(), HD.into()]);
    // Qwen3 per-head QK-norm, then RoPE
    let qn = nn.rmsnorm("q_norm", qh, HD, 1e-6);
    let kn = nn.rmsnorm("k_norm", kh, HD, 1e-6);
    let qr = nn.rope(qn, HD as u32, 1e6);
    let kr = nn.rope(kn, HD as u32, 1e6);
    let attn = nn.attention(
        qr, kr, vh, NH as u32, NKV as u32, HD as u32, true, None, None,
    );
    let ao = nn.reshape(attn, [T.into(), QD.into()]);
    let o = nn.linear("o_proj", ao, QD, H, false);
    let r1 = nn.add(x, o);
    // --- MLP (SwiGLU) ---
    let h2 = nn.rmsnorm("post_norm", r1, H, 1e-6);
    let gate = nn.linear("gate_proj", h2, H, IM, false);
    let up = nn.linear("up_proj", h2, H, IM, false);
    let ga = nn.act(ActKind::Silu, gate);
    let gu = nn.mul(ga, up);
    let down = nn.linear("down_proj", gu, IM, H, false);
    let out = nn.add(r1, down);
    nn.end_block();

    nn.mark_output(out);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("infer shapes");
    g
}

/// Drive the block all the way to a tile graph and check the structural facts
/// the scheduler relies on.
#[test]
fn qwen_block_lowers_to_tile_graph() {
    let g = qwen_block();
    let plan = plan_from_block(&g, 0).expect("bridge block → plan");

    // The bridge covered every op of the block.
    assert_eq!(plan.ops.len(), g.block_nodes(0).count());

    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (tg, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).expect("assemble tile graph");

    // The seven linears (q/k/v/o/gate/up/down) became GEMM compute nodes, and
    // attention became a flash-attention compute node.
    let gemms = tg
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n,
                GraphNode::Compute {
                    kind: Compute::Gemm(_),
                    ..
                }
            )
        })
        .count();
    let flashes = tg
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n,
                GraphNode::Compute {
                    kind: Compute::Flash(_),
                    ..
                }
            )
        })
        .count();
    assert_eq!(gemms, 7, "expected 7 GEMMs (q/k/v/o/gate/up/down)");
    assert_eq!(flashes, 1, "expected 1 flash-attention node");

    // Single unit ⇒ every producer→consumer hand-off stays in SRAM (no HBM
    // round-trip) and is colocated, never a cross-unit transfer.
    assert!(!cons.handoffs.is_empty());
    for hf in &cons.handoffs {
        assert!(!hf.cross_unit);
        assert!(matches!(
            tg.nodes[hf.producer_dma_out],
            GraphNode::DmaOut { resident: true, .. }
        ));
        assert!(matches!(
            tg.nodes[hf.consumer_dma_in],
            GraphNode::DmaIn { resident: true, .. }
        ));
    }

    // Every distinct weight/activation is staged from DRAM exactly once (dedup):
    // the norm output feeding q/k/v is staged zero times (it's resident).
    let mut staged = HashMap::new();
    for n in &tg.nodes {
        if let GraphNode::DmaIn {
            tensor,
            resident: false,
        } = n
        {
            *staged.entry(tensor.clone()).or_insert(0) += 1;
        }
    }
    assert!(
        staged.values().all(|&c| c == 1),
        "an operand was staged from DRAM more than once"
    );
    // All seven projection weights are present as distinct DRAM stages.
    for w in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ] {
        assert!(
            staged.keys().any(|t| t == &format!("{w}.weight")),
            "missing weight stage for {w}"
        );
    }
}

/// The headline cross-op tile dependency: each RMSNorm → GEMM edge couples the
/// GEMM's M-tiles to exactly the norm row-blocks covering their rows (the N /
/// output-feature axis is free). This is what lets a projection's first M-tile
/// start while the norm is still producing later rows.
#[test]
fn rmsnorm_to_projection_couples_on_token_axis() {
    let g = qwen_block();
    let plan = plan_from_block(&g, 0).expect("bridge");
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (_, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).expect("assemble");

    // Collect the Row→Gemm tile deps (input_norm→q/k/v, post_norm→gate/up, …).
    let row_to_gemm: Vec<_> = cons
        .tile_deps
        .iter()
        .filter(|d| {
            matches!(cons.domains[&d.producer], TileDomain::Row { .. })
                && matches!(cons.domains[&d.consumer], TileDomain::Gemm { .. })
        })
        .collect();
    // At least input_norm→{q,k,v} (3) and post_norm→{gate,up} (2).
    assert!(
        row_to_gemm.len() >= 5,
        "expected >= 5 norm→gemm couplings, got {}",
        row_to_gemm.len()
    );

    for d in row_to_gemm {
        // Couples the token/row axis: norm row axis 0 ↔ GEMM M axis 0.
        assert_eq!(d.dep.couple.len(), 1);
        assert_eq!(d.dep.couple[0].producer_axis, 0);
        assert_eq!(d.dep.couple[0].consumer_axis, 0);

        let (pd, cd) = (cons.domains[&d.producer], cons.domains[&d.consumer]);
        let (TileDomain::Row { br, .. }, TileDomain::Gemm { bm, .. }) = (pd, cd) else {
            unreachable!()
        };

        // Every materialized edge is a genuine M-range overlap between the two
        // blockings; the counter threshold equals the per-tile in-degree.
        let edges = materialize_tile_deps(&pd, &cd, &d.dep);
        assert!(!edges.is_empty());
        for e in &edges {
            let (i, r) = (e.consumer_coord[0], e.producer_coord[0]);
            assert!(
                r * br < i * bm + bm && i * bm < r * br + br,
                "tile {i} ⟂ row-block {r}"
            );
        }
        // N is free: for a fixed M-tile i, every N-tile j shares the same producers.
        let mut by_i: HashMap<i64, Vec<Vec<i64>>> = HashMap::new();
        let mut by_ij: HashMap<(i64, i64), Vec<i64>> = HashMap::new();
        for e in &edges {
            by_ij
                .entry((e.consumer_coord[0], e.consumer_coord[1]))
                .or_default()
                .push(e.producer_coord[0]);
        }
        for ((i, _), mut set) in by_ij {
            set.sort_unstable();
            by_i.entry(i).or_default().push(set);
        }
        for sets in by_i.values() {
            assert!(
                sets.windows(2).all(|w| w[0] == w[1]),
                "N-tiles disagree on producers"
            );
        }

        let thresholds = consumer_thresholds(&pd, &cd, &d.dep);
        let mut counted: HashMap<Vec<i64>, u32> = HashMap::new();
        for e in &edges {
            *counted.entry(e.consumer_coord.clone()).or_insert(0) += 1;
        }
        assert_eq!(thresholds, counted);
    }
}

/// The full compiler front half: nn-graph → egglog fusion → fused-graph bridge →
/// tiling. Fusion folds each RMSNorm into the projection that consumes it, so
/// the norm→linear hand-off disappears at the tile level — yet the GEMM/flash
/// structure and the *remaining* cross-op tile dependencies are intact.
#[test]
fn fused_block_lowers_to_tile_graph() {
    let g = qwen_block();
    let (fused, _stats) = rewrite_graph(&g).expect("rewrite (fuse)");
    let plan = plan_from_fused(&fused, &g).expect("bridge fused → plan");

    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (tg, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).expect("assemble");

    // Same compute skeleton as the unfused path: 7 projections + 1 attention.
    assert_eq!(count_computes(&tg, |k| matches!(k, Compute::Gemm(_))), 7);
    assert_eq!(count_computes(&tg, |k| matches!(k, Compute::Flash(_))), 1);

    // Fusion reached tiling: q/k/v read the *raw block input* `x` directly —
    // their input RMSNorm was folded into the GEMM (FusedNormLinear), so there
    // is no separate norm node feeding them.
    let qkv_on_x = plan
        .ops
        .iter()
        .filter(|o| {
            matches!(o.kind, rewrite::OpKind::Gemm(_))
                && o.inputs.first().map(String::as_str) == Some("x")
        })
        .count();
    assert_eq!(
        qkv_on_x, 3,
        "q/k/v should consume the raw input with the norm fused in"
    );

    // Fusion strictly reduced the number of row/elementwise tasks vs. the
    // unfused plan (the standalone norms are gone).
    let raw = plan_from_block(&g, 0).expect("raw plan");
    let raw_rows = raw
        .ops
        .iter()
        .filter(|o| matches!(o.kind, rewrite::OpKind::Row(_)))
        .count();
    let fused_rows = plan
        .ops
        .iter()
        .filter(|o| matches!(o.kind, rewrite::OpKind::Row(_)))
        .count();
    assert!(
        fused_rows < raw_rows,
        "fusion did not reduce row ops: {fused_rows} vs {raw_rows}"
    );

    // Norm weights survive fusion as distinct DRAM-staged operands, even with no
    // standalone norm compute node (the fused kernel still needs them).
    let staged: Vec<&String> = tg
        .nodes
        .iter()
        .filter_map(|n| match n {
            GraphNode::DmaIn {
                tensor,
                resident: false,
            } => Some(tensor),
            _ => None,
        })
        .collect();
    for w in ["input_norm.weight", "post_norm.weight"] {
        assert!(
            staged.iter().any(|t| *t == w),
            "fused-away norm weight {w} not staged"
        );
    }

    // Cross-op tile dependencies still exist between fused ops — e.g. the
    // residual (Row) feeding gate/up projections (Gemm) on the token axis — and
    // every materialized edge is a genuine M-range overlap.
    let row_to_gemm: Vec<_> = cons
        .tile_deps
        .iter()
        .filter(|d| {
            matches!(cons.domains[&d.producer], TileDomain::Row { .. })
                && matches!(cons.domains[&d.consumer], TileDomain::Gemm { .. })
        })
        .collect();
    assert!(
        !row_to_gemm.is_empty(),
        "expected surviving Row→Gemm couplings"
    );
    for d in row_to_gemm {
        assert_eq!(d.dep.couple[0].producer_axis, 0);
        assert_eq!(d.dep.couple[0].consumer_axis, 0);
        let (pd, cd) = (cons.domains[&d.producer], cons.domains[&d.consumer]);
        let edges = materialize_tile_deps(&pd, &cd, &d.dep);
        assert!(!edges.is_empty());
        let (TileDomain::Row { br, .. }, TileDomain::Gemm { bm, .. }) = (pd, cd) else {
            unreachable!()
        };
        for e in &edges {
            let (i, r) = (e.consumer_coord[0], e.producer_coord[0]);
            assert!(r * br < i * bm + bm && i * bm < r * br + br);
        }
    }
}
