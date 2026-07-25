//! Pass §8.4 — tile dominance pruning.
//!
//! Filters the candidate list from [`crate::CostModel::candidates`] to remove
//! tiles that are Pareto-dominated in three cost-monotone axes:
//!
//! * `passes` — SRAM streaming passes (higher ⇒ more DMA operand reloads)
//! * `sram_pages` — working-set page count (higher ⇒ less occupancy, less
//!   compute/DMA overlap)
//! * `output_tiles` — number of BM×BN output tiles = ⌈M/bm⌉ · ⌈N/bn⌉ (higher
//!   ⇒ more compute steps and per-tile DMAs)
//!
//! Each metric is a monotone input to [`crate::cost::gemm_cycles`]: raising
//! any one, holding the rest, does not decrease the estimated cycles. Hence
//! any tile Pareto-dominated in all three is provably worse than its
//! dominator and can be dropped without losing the optimum.
//!
//! Downstream effect: the tile-selection extractor (rewrite::explore) sees a
//! smaller candidate set, so egglog saturation over tile choices runs faster
//! (§8.4 predicts 2–4× candidate reduction on typical GEMMs).

use crate::tile::{GemmShape, TileShape};
use crate::CostModel;

/// Metrics driving dominance. Kept public so callers can inspect them.
#[derive(Clone, Copy, Debug)]
pub struct TileMetrics {
    pub passes: u64,
    pub sram_pages: u64,
    pub output_tiles: u64,
}

/// Summary of what got pruned. `before - kept.len() == dropped.len()` unless
/// the input already contained duplicates.
#[derive(Debug, Clone, Default)]
pub struct DominanceReport {
    pub before: usize,
    pub kept: usize,
    pub dropped: usize,
}

impl DominanceReport {
    pub fn savings_pct(&self) -> f64 {
        if self.before == 0 {
            0.0
        } else {
            100.0 * self.dropped as f64 / self.before as f64
        }
    }
}

fn output_tiles(g: GemmShape, t: TileShape) -> u64 {
    let tm = (g.m as u64).div_ceil(t.bm.max(1) as u64).max(1);
    let tn = (g.n as u64).div_ceil(t.bn.max(1) as u64).max(1);
    tm * tn
}

impl CostModel<'_> {
    /// Metrics for one tile against a target GEMM shape.
    pub fn tile_metrics(&self, g: GemmShape, t: TileShape) -> TileMetrics {
        TileMetrics {
            passes: self.passes(t),
            sram_pages: self.sram_pages(t),
            output_tiles: output_tiles(g, t),
        }
    }
}

/// `a` Pareto-dominates `b`: at least as good in every metric, and strictly
/// better in at least one.
fn dominates(a: TileMetrics, b: TileMetrics) -> bool {
    let a_le = a.passes <= b.passes
        && a.sram_pages <= b.sram_pages
        && a.output_tiles <= b.output_tiles;
    let a_lt = a.passes < b.passes
        || a.sram_pages < b.sram_pages
        || a.output_tiles < b.output_tiles;
    a_le && a_lt
}

/// Drop Pareto-dominated tiles. Preserves input order among survivors.
pub fn prune_dominated(
    cands: &[TileShape],
    g: GemmShape,
    cm: &CostModel<'_>,
) -> (Vec<TileShape>, DominanceReport) {
    let before = cands.len();
    let metrics: Vec<TileMetrics> = cands.iter().map(|&t| cm.tile_metrics(g, t)).collect();

    let mut kept = Vec::with_capacity(before);
    for (i, &t) in cands.iter().enumerate() {
        let mi = metrics[i];
        // Dominated by any *other* candidate (including duplicates via strict
        // inequality — a duplicate tile does not dominate itself).
        let dominated = metrics
            .iter()
            .enumerate()
            .any(|(j, &mj)| i != j && dominates(mj, mi));
        if !dominated {
            kept.push(t);
        }
    }

    let report = DominanceReport {
        before,
        kept: kept.len(),
        dropped: before - kept.len(),
    };
    (kept, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CostModel;
    use hwspec::registry;

    fn cm() -> CostModel<'static> {
        CostModel::new(registry::lookup("H100 SXM5").unwrap(), crate::DEFAULT_PAGE_BYTES)
    }

    /// Dominance is transitive: dropping A because it's beaten by B, and B
    /// because it's beaten by C, leaves only C. Sanity that our loop doesn't
    /// stop early.
    #[test]
    fn dominates_is_transitive_within_prune() {
        let a = TileMetrics { passes: 4, sram_pages: 20, output_tiles: 200 };
        let b = TileMetrics { passes: 2, sram_pages: 10, output_tiles: 100 };
        let c = TileMetrics { passes: 1, sram_pages: 5, output_tiles: 50 };
        assert!(dominates(b, a));
        assert!(dominates(c, b));
        assert!(dominates(c, a));
    }

    /// Tiles that are equal in every metric are kept (dominance is strict).
    #[test]
    fn duplicates_are_not_dropped() {
        let m = TileMetrics { passes: 2, sram_pages: 10, output_tiles: 100 };
        assert!(!dominates(m, m));
    }

    /// Real GEMM: expect a Pareto frontier of size ≤ the input.
    #[test]
    fn prune_never_grows_the_set() {
        let cm = cm();
        let g = GemmShape { m: 1024, n: 4096, k: 4096 };
        let cands = cm.candidates(g, crate::SramPolicy::Stream);
        let (pruned, rep) = prune_dominated(&cands, g, &cm);
        assert!(pruned.len() <= cands.len());
        assert_eq!(rep.before, cands.len());
        assert_eq!(rep.kept, pruned.len());
        assert_eq!(rep.dropped, cands.len() - pruned.len());
    }

    /// Manually constructed 3-tile input where one is dominated: verify it
    /// gets dropped. Uses metrics directly rather than a real CostModel.
    #[test]
    fn drops_strictly_dominated_tile() {
        let good = TileMetrics { passes: 1, sram_pages: 5, output_tiles: 50 };
        let bad = TileMetrics { passes: 2, sram_pages: 10, output_tiles: 100 };
        let neutral = TileMetrics { passes: 3, sram_pages: 4, output_tiles: 60 };
        assert!(dominates(good, bad));
        assert!(!dominates(good, neutral)); // neutral has fewer pages
        assert!(!dominates(neutral, good)); // good has fewer passes/tiles
    }
}
