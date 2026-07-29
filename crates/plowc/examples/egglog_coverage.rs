//! Q1 coverage probe: how much of a real checkpoint's graph does the egglog
//! rewrite stage actually cover, in ops AND in FLOPs?
//!
//! Reads a HuggingFace `config.json`, builds the Stage-1 graph at a bound
//! shape bucket, then reports:
//!   * the op histogram of the input graph, weighted by FLOPs;
//!   * what `explore_stats` (saturate-only) discovers;
//!   * what `rewrite_graph` (saturate + EXTRACT) produces, or the failure.
//!
//! Usage: `cargo run --release -p plowc --example egglog_coverage -- <dir> [B] [S]`

use std::collections::BTreeMap;

use nn_graph::{Dim, Graph, Op, Origin, Shape};

/// FLOPs for one node, given its resolved input/output shapes.
/// Counts multiply-accumulate as 2 FLOPs; elementwise/norm as 1 per element
/// (they are bandwidth-bound anyway — the point of this column is to separate
/// "big" ops from "small" ones, not to be a roofline).
fn node_flops(g: &Graph, n: &nn_graph::Node) -> u128 {
    let numel = |t: nn_graph::TensorId| -> u128 {
        g.tensors[t.0 as usize]
            .shape
            .as_ref()
            .map(shape_numel)
            .unwrap_or(0)
    };
    let out = numel(n.output);
    match &n.op {
        Op::Linear { .. } => {
            // out_elems * K * 2
            let k = n
                .inputs
                .first()
                .and_then(|&t| g.tensors[t.0 as usize].shape.as_ref())
                .and_then(|s| s.last())
                .and_then(|d| d.as_static())
                .unwrap_or(0) as u128;
            out * k * 2
        }
        Op::MatMul => {
            let k = n
                .inputs
                .first()
                .and_then(|&t| g.tensors[t.0 as usize].shape.as_ref())
                .and_then(|s| s.last())
                .and_then(|d| d.as_static())
                .unwrap_or(0) as u128;
            out * k * 2
        }
        Op::Attention {
            num_heads,
            head_dim,
            ..
        } => {
            // 2 GEMMs of [S,hd]x[hd,S] and [S,S]x[S,hd] per head; `out` is
            // [B,S,H*hd], so tokens = out / (H*hd).
            let hd = *head_dim as u128;
            let h = *num_heads as u128;
            let tokens = if h * hd > 0 { out / (h * hd) } else { 0 };
            // Causal ⇒ ~half. Sequence length is tokens/batch; approximate with
            // tokens (batch is 1 in the probe buckets).
            2 * 2 * h * hd * tokens * tokens / 2
        }
        Op::Embedding => out,
        // Norms, activations, elementwise, rope, scale, softmax, reshape…
        _ => out,
    }
}

fn shape_numel(s: &Shape) -> u128 {
    let mut acc: u128 = 1;
    for d in s.dims() {
        match d.as_static() {
            Some(v) if v >= 0 => acc *= v as u128,
            _ => return 0,
        }
    }
    acc
}

/// The `nn_graph` op names that each egglog rule's LHS can match. Anything
/// outside this set is provably untouchable by the current rule set, whatever
/// the graph looks like.
fn rule_reachable(op: &Op) -> bool {
    matches!(
        op,
        Op::Linear { .. }
            | Op::RmsNorm { .. }
            | Op::LayerNorm { .. }
            | Op::Rope { .. }
            | Op::Act(_)
            | Op::Elementwise(_)
            | Op::Scale(_)
            | Op::GroupNorm { .. }
            | Op::Conv3d { .. }
            | Op::Embedding
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "/root/gemma-4-12B-it".into());
    let b: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let s: i64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1024);

    let json = std::fs::read_to_string(format!("{dir}/config.json")).expect("config.json");
    let mut g = nn_graph::models::build_from_config_json_at(&json, &nn_graph::models::ShapeBucket::default())
        .expect("build graph");
    g.bind(&nn_graph::Bindings::new().set("B", b).set("S", s));

    println!("== {dir}  (B={b}, S={s}) ==");
    println!("graph nodes: {}", g.nodes.len());

    // --- Input-graph histogram, ops and FLOPs ---
    let mut by_op: BTreeMap<&'static str, (usize, u128, usize)> = BTreeMap::new();
    let mut total_flops: u128 = 0;
    let mut reach_ops = 0usize;
    let mut reach_flops: u128 = 0;
    for n in &g.nodes {
        let f = node_flops(&g, n);
        total_flops += f;
        let r = rule_reachable(&n.op);
        if r {
            reach_ops += 1;
            reach_flops += f;
        }
        let e = by_op.entry(n.op.name()).or_insert((0, 0, 0));
        e.0 += 1;
        e.1 += f;
        e.2 += r as usize;
    }

    println!("\n-- input graph: ops / GFLOP / rule-reachable ops --");
    let mut rows: Vec<_> = by_op.iter().collect();
    rows.sort_by_key(|(_, v)| std::cmp::Reverse(v.1));
    for (name, (cnt, fl, rch)) in rows {
        println!(
            "{name:>14}  {cnt:>7}  {:>12.3}  {rch:>7}  {:>5.1}% of FLOPs",
            *fl as f64 / 1e9,
            100.0 * *fl as f64 / total_flops.max(1) as f64
        );
    }
    println!(
        "\nTOTAL {} ops, {:.3} GFLOP",
        g.nodes.len(),
        total_flops as f64 / 1e9
    );
    println!(
        "RULE-REACHABLE (any rule LHS could match this op kind): {reach_ops} ops ({:.1}%), \
         {:.3} GFLOP ({:.1}% of FLOPs)",
        100.0 * reach_ops as f64 / g.nodes.len() as f64,
        reach_flops as f64 / 1e9,
        100.0 * reach_flops as f64 / total_flops.max(1) as f64
    );

    // --- Saturate-only: what does egglog FIND? ---
    match rewrite::explore_stats(&g) {
        Ok((ops, fused)) => {
            let total: usize = fused.iter().map(|(_, c)| c).sum();
            println!("\n-- explore_stats (saturate, no extract): {ops} ops, {total} matches --");
            let mut f = fused.clone();
            f.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            for (name, c) in f {
                println!("  {name:>28}  {c}");
            }
        }
        Err(e) => println!("\nexplore_stats FAILED: {e}"),
    }

    // --- Full extract: does the pipeline that would feed an emitter work? ---
    println!("\n-- rewrite_graph (saturate + EXTRACT) --");
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rewrite::rewrite_graph(&g))) {
        Ok(Ok((fused, stats))) => {
            println!(
                "  OK: ops_before={} ops_after={} fused={}",
                stats.ops_before, stats.ops_after, stats.fused
            );
            let mut h: BTreeMap<String, usize> = BTreeMap::new();
            for n in &fused.nodes {
                *h.entry(n.op.clone()).or_default() += 1;
            }
            let mut rows: Vec<_> = h.into_iter().collect();
            rows.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
            for (name, c) in rows {
                println!("  {name:>28}  {c}");
            }
        }
        Ok(Err(e)) => println!("  EXTRACT ERROR: {e}"),
        Err(_) => println!("  EXTRACT PANICKED (upstream egglog 2.0.0 extract.rs:471)"),
    }

    // --- The bridge the devblob path would need ---
    println!("\n-- plan_from_all_blocks (the LayerPlan an emitter consumes) --");
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rewrite::plan_from_all_blocks(&g)
    })) {
        Ok(Ok(plan)) => {
            println!("  OK: {} ops in plan", plan.ops.len());
            let mut h: BTreeMap<String, usize> = BTreeMap::new();
            for o in &plan.ops {
                *h.entry(format!("{:?}", std::mem::discriminant(&o.kind))).or_default() += 1;
            }
            println!("  distinct op kinds: {}", h.len());
        }
        Ok(Err(e)) => println!("  BRIDGE ERROR: {e}"),
        Err(_) => println!("  BRIDGE PANICKED"),
    }

    let _ = Origin::Weight;
    let _ = Dim::stat(1);
}
