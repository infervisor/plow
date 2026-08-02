//! End-to-end test that **baking all blocks** produces one plan whose tile
//! graph chains fine-grained tile dependencies across block boundaries — the
//! consumer-block's first op tiles unblock as their specific producer-block
//! last-op tiles complete, not after the whole tensor.
//!
//! Backs the design in `plans/lean-formal-verification-analysis.md §5.10-D`
//! and the `plan_from_all_blocks` API.

use costmodel::{Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use nn_graph::{infer_shapes, ActKind, DType, Nn};
use rewrite::{
    assemble, materialize_tile_deps, plan_from_all_blocks, plan_from_block, rewrite_graph, Compute,
    GraphNode,
};

// Small Llama-like decoder dims — enough shape to force meaningful tile grids
// without exploding test time.
const H: i64 = 512;
const NH: i64 = 8;
const NKV: i64 = 2;
const HD: i64 = 64;
const QD: i64 = NH * HD;
const KVD: i64 = NKV * HD;
const IM: i64 = 1024;
const T: i64 = 128;

fn h100() -> &'static costmodel::hwspec::GpuSpec {
    costmodel::hwspec::registry::lookup("H100 SXM5").unwrap()
}

/// Build a small decoder with `n_blocks` stacked decoder layers. Each block is
/// a full attention+MLP + residuals unit; blocks share the residual stream
/// via nn-graph's ambient tensor-name plumbing.
fn stacked_decoder(n_blocks: u32) -> nn_graph::Graph {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let mut x = nn.input("x", nn.shape([T.into(), H.into()]), DType::BF16);

    for i in 0..n_blocks {
        nn.begin_block(&format!("layers.{i}"));
        // Attention branch.
        let h1 = nn.rmsnorm(&format!("l{i}_input_norm"), x, H, 1e-6);
        let q = nn.linear(&format!("l{i}_q_proj"), h1, H, QD, false);
        let k = nn.linear(&format!("l{i}_k_proj"), h1, H, KVD, false);
        let v = nn.linear(&format!("l{i}_v_proj"), h1, H, KVD, false);
        let qh = nn.reshape(q, [T.into(), NH.into(), HD.into()]);
        let kh = nn.reshape(k, [T.into(), NKV.into(), HD.into()]);
        let vh = nn.reshape(v, [T.into(), NKV.into(), HD.into()]);
        let qr = nn.rope(qh, HD as u32, 1e6);
        let kr = nn.rope(kh, HD as u32, 1e6);
        let attn = nn.attention(
            qr, kr, vh, NH as u32, NKV as u32, HD as u32, true, None, None,
        );
        let ao = nn.reshape(attn, [T.into(), QD.into()]);
        let o = nn.linear(&format!("l{i}_o_proj"), ao, QD, H, false);
        let r1 = nn.add(x, o);
        // MLP (SwiGLU) branch.
        let h2 = nn.rmsnorm(&format!("l{i}_post_norm"), r1, H, 1e-6);
        let gate = nn.linear(&format!("l{i}_gate_proj"), h2, H, IM, false);
        let up = nn.linear(&format!("l{i}_up_proj"), h2, H, IM, false);
        let ga = nn.act(ActKind::Silu, gate);
        let gu = nn.mul(ga, up);
        let down = nn.linear(&format!("l{i}_down_proj"), gu, IM, H, false);
        let out = nn.add(r1, down);
        nn.end_block();
        // The next block reads this block's output.
        x = out;
    }

    nn.mark_output(x);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("infer shapes");
    g
}

#[test]
fn plan_from_all_blocks_concatenates_ops_in_block_order() {
    let g = stacked_decoder(3);
    let all = plan_from_all_blocks(&g).expect("all-blocks plan");
    let per_block: Vec<_> = (0..3u32)
        .map(|b| plan_from_block(&g, b).expect("per-block plan"))
        .collect();

    // Concatenation preserves per-block op count exactly.
    let per_block_ops: usize = per_block.iter().map(|p| p.ops.len()).sum();
    assert_eq!(
        all.ops.len(),
        per_block_ops,
        "combined plan op count must equal sum of per-block plans"
    );

    // The name suffix carries the block index, so ops from different blocks
    // are distinguishable in the combined plan.
    let first_block_names: Vec<_> = all
        .ops
        .iter()
        .filter(|o| o.name.ends_with("_L0"))
        .map(|o| o.name.clone())
        .collect();
    let last_block_names: Vec<_> = all
        .ops
        .iter()
        .filter(|o| o.name.ends_with("_L2"))
        .map(|o| o.name.clone())
        .collect();
    assert_eq!(first_block_names.len(), per_block[0].ops.len());
    assert_eq!(last_block_names.len(), per_block[2].ops.len());

    // Block order is preserved: every _L0 op appears before every _L2 op.
    let first_l2 = all
        .ops
        .iter()
        .position(|o| o.name.ends_with("_L2"))
        .expect("some _L2 op");
    let last_l0 = all
        .ops
        .iter()
        .rposition(|o| o.name.ends_with("_L0"))
        .expect("some _L0 op");
    assert!(
        last_l0 < first_l2,
        "block order violated: last _L0 at {last_l0} ≥ first _L2 at {first_l2}"
    );
}

#[test]
fn assembled_multi_block_has_tile_deps_across_block_boundary() {
    let g = stacked_decoder(2);
    let (_fused, _stats) = rewrite_graph(&g).expect("rewrite");
    let plan = plan_from_all_blocks(&g).expect("all-blocks plan");
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (tg, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).expect("assemble");

    // Map each compute node → its op name (which carries the `_L{block}` suffix).
    let node_op: Vec<Option<String>> = tg
        .nodes
        .iter()
        .map(|n| match n {
            GraphNode::Compute { op, .. } => Some(op.clone()),
            _ => None,
        })
        .collect();

    // Find at least one tile_dep whose producer is in block 0 and consumer in
    // block 1. This is the boundary-spanning tile dependency that lets block
    // 1's first tiles unblock per-tile as block 0's last-op tiles finish.
    let mut cross_block = 0usize;
    for d in &cons.tile_deps {
        let (Some(pn), Some(cn)) = (&node_op[d.producer], &node_op[d.consumer]) else {
            continue;
        };
        let pblock = block_suffix(pn);
        let cblock = block_suffix(cn);
        if let (Some(pb), Some(cb)) = (pblock, cblock) {
            if pb != cb {
                assert!(pb < cb, "producer must be in an earlier block");
                cross_block += 1;
                // Materialize the tile edges and check at least one exists.
                let (pd, cd) = (cons.domains[&d.producer], cons.domains[&d.consumer]);
                let edges = materialize_tile_deps(&pd, &cd, &d.dep);
                assert!(
                    !edges.is_empty(),
                    "cross-block tile dep {pn} → {cn} materialized to zero edges"
                );
            }
        }
    }
    assert!(
        cross_block > 0,
        "no cross-block tile dependencies found — cross-block pipelining is broken"
    );
    eprintln!(
        "[multi-block] {} cross-block tile dependencies observed",
        cross_block
    );
}

#[test]
fn all_gemms_and_flashes_show_up_across_blocks() {
    let g = stacked_decoder(2);
    let plan = plan_from_all_blocks(&g).expect("plan");
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (tg, _cons) = assemble(&soc, &plan, SramPolicy::Stream, None).expect("assemble");

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
    // 2 blocks × (7 GEMMs + 1 flash) = 14 GEMMs, 2 flashes.
    assert_eq!(gemms, 14, "expected 14 GEMMs across 2 blocks");
    assert_eq!(
        flashes, 2,
        "expected 2 flash-attention nodes across 2 blocks"
    );
}

fn block_suffix(op_name: &str) -> Option<u32> {
    let idx = op_name.rfind("_L")?;
    op_name[idx + 2..].parse().ok()
}
