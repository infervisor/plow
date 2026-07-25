//! Measure how many candidates §8.4 dominance-pruning drops on realistic
//! GEMM shapes. Not a correctness test — the unit tests handle that; this
//! quantifies the win.

use costmodel::dominance::prune_dominated;
use costmodel::{CostModel, GemmShape, SramPolicy, DEFAULT_PAGE_BYTES};
use hwspec::registry;

fn cm() -> CostModel<'static> {
    CostModel::new(registry::lookup("H100 SXM5").unwrap(), DEFAULT_PAGE_BYTES)
}

#[test]
fn reports_pruning_savings_on_representative_gemms() {
    let cm = cm();
    let shapes = [
        ("prefill-q",   GemmShape { m: 1024, n: 4096, k: 4096 }),
        ("prefill-kv",  GemmShape { m: 1024, n: 1024, k: 4096 }),
        ("mlp-up",      GemmShape { m: 1024, n: 14336, k: 4096 }),
        ("mlp-down",    GemmShape { m: 1024, n: 4096, k: 14336 }),
        ("decode-q",    GemmShape { m: 1, n: 4096, k: 4096 }),
        ("decode-mlp",  GemmShape { m: 1, n: 14336, k: 4096 }),
    ];
    for (name, g) in shapes {
        let cands = cm.candidates(g, SramPolicy::Stream);
        let (pruned, rep) = prune_dominated(&cands, g, &cm);
        assert!(pruned.len() <= cands.len());
        eprintln!(
            "[dominance] {name:<12} m={:>5} n={:>5} k={:>5} — {} → {} tiles ({:.1}% dropped)",
            g.m, g.n, g.k, rep.before, rep.kept, rep.savings_pct(),
        );
    }
}
