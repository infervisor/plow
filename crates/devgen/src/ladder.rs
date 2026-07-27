//! Wave-aligned prefill bucket ladder (PX-6, `perf-data/px6-sm-quantization.md`).
//!
//! The prefill GEMM body walks `T = ceil(M/BM) * ceil(N/BN)` output tiles with a grid-stride
//! loop keyed on the packet's `(slice, nblk)` (`runtime/nvidia/op_gemm.cuh:893-900`), so one
//! launch of `tm = ceil(t/BM)` row-tiles costs
//!
//! ```text
//!     sum_op  ceil(tm * tn_op / n_cu) * tau_op
//! ```
//!
//! That is a STAIRCASE in `tm`, flat between wave boundaries — rows added inside a tread are
//! free, and one row past a tread top costs a whole extra wave of every op that stepped.
//! Measured on a 170-SM RTX 5090 the effect is not subtle: `N = 170*128` runs 1 wave at
//! 0.18362 ms and `N = 171*128` runs 2 waves at 0.30368 ms — **0.6% more work, 65% more time**.
//!
//! `tau_op` needs no measurement: it is linear in `k` to within 5% (measured tau/k =
//! 0.0355 / 0.0359 / 0.0372 / 0.0373 us per k-unit at k = 15360 / 8192 / 4096 / 3840), so the
//! whole staircase is computable at emit time from `(tn, k, glu, n_cu)` alone. The fused GLU
//! reads two weight matrices and is charged 2x, which is ~10% conservative (its measured
//! tau/k is 0.0337 because the two matmuls share the A-stage).
//!
//! **Why this matters.** The shipped rungs `[128, 512, 1024, 2048, 4096]` sit on powers of two,
//! which is unrelated to where the treads are. Scored against optimal multi-launch covering on
//! the real Gemma-4-12B op mix at n_cu=170, that ladder gives up **9.6% of prefill GEMM time on
//! average** over L = 128..4096, worst cells +41.9% (640 rows, which must be served as 128+512)
//! and +31.6% (1280 rows -> 512+1024). Adding four tread-top rungs — 1408, 2176, 640, 1792 —
//! takes the mean loss to **1.4%**. None of them is a power of two: they are `11*128`, `17*128`,
//! `5*128`, `14*128`.
//!
//! The ladder is a function of `n_cu`, so it is NOT portable across GPUs: 170 = 2*5*17 and
//! 188 = 2^2*47 put the treads in completely different places. That is the whole point of
//! deriving it instead of hardcoding it.

/// One prefill GEMM, as the ladder cost model sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LadderOp {
    /// Output column tiles, `ceil(N / bn)`.
    pub(crate) tn: u32,
    /// Reduction depth. Measured `tau` is linear in this.
    pub(crate) k: u32,
    /// Fused gate|up reads two weight matrices per tile.
    pub(crate) glu: bool,
    /// How many layers carry this op.
    pub(crate) count: u32,
}

/// Relative cost of ONE prefill launch of `tm` row-tiles. Arbitrary units (`tau ∝ k`), so only
/// ratios are meaningful — which is all the rung selection below needs.
pub(crate) fn launch_cost(n_cu: u32, ops: &[LadderOp], tm: u32) -> u64 {
    let p = n_cu.max(1) as u64;
    ops.iter()
        .map(|o| {
            let waves = ((tm as u64 * o.tn as u64) + p - 1) / p;
            waves * o.k as u64 * if o.glu { 2 } else { 1 } * o.count as u64
        })
        .sum()
}

/// Tread tops: every `tm` whose cost is strictly below `tm + 1`'s, i.e. the largest launch size
/// before some op's wave count increments. `max_tm` is always a candidate.
fn tread_tops(n_cu: u32, ops: &[LadderOp], max_tm: u32) -> Vec<u32> {
    let mut v: Vec<u32> = (1..max_tm)
        .filter(|&tm| launch_cost(n_cu, ops, tm) < launch_cost(n_cu, ops, tm + 1))
        .collect();
    v.push(max_tm);
    v
}

/// `out[l]` = min cost to cover `l` row-tiles with `ladder`, multi-launch allowed.
/// `ladder` must contain 1 or long prompts are uncoverable.
fn cover_cost(n_cu: u32, ops: &[LadderOp], ladder: &[u32], max_tm: u32) -> Vec<u64> {
    let mut best = vec![u64::MAX; max_tm as usize + 1];
    best[0] = 0;
    for l in 1..=max_tm as usize {
        for &r in ladder {
            let prev = l.saturating_sub(r as usize);
            if best[prev] == u64::MAX {
                continue;
            }
            let c = best[prev].saturating_add(launch_cost(n_cu, ops, r));
            if c < best[l] {
                best[l] = c;
            }
        }
    }
    best
}

/// Total covering cost over every prompt length — the objective rung selection minimizes.
fn total_cover(n_cu: u32, ops: &[LadderOp], ladder: &[u32], max_tm: u32) -> u64 {
    cover_cost(n_cu, ops, ladder, max_tm)
        .iter()
        .skip(1)
        .fold(0u64, |a, &b| a.saturating_add(b))
}

/// Choose `budget` prefill rungs (in row-tiles) for this GPU and model.
///
/// Greedy forward selection over the tread tops against optimal multi-launch covering. Greedy is
/// the right tool here and not a shortcut: the objective is supermodular-ish in practice and the
/// measured greedy sequence (1408, 2176, 640, 1792) recovers 8.2 of the 9.6 available points in
/// four rungs, so an exact search would buy under a point at 2^23 the cost.
///
/// Rung 1 is always present — it is `MIN_CHUNK` and without it long prompts are uncoverable.
/// Returns sorted row-tile counts; the caller scales by `bm` to get rows.
pub(crate) fn wave_ladder(n_cu: u32, ops: &[LadderOp], max_tm: u32, budget: usize) -> Vec<u32> {
    assert!(max_tm >= 1, "max_tm must be >= 1");
    if ops.is_empty() || max_tm == 1 {
        return vec![1];
    }
    let cands = tread_tops(n_cu, ops, max_tm);
    let mut sel: Vec<u32> = vec![1];
    if !sel.contains(&max_tm) {
        sel.push(max_tm);
    }
    while sel.len() < budget.max(sel.len()) {
        let cur = total_cover(n_cu, ops, &sel, max_tm);
        let mut best: Option<(u64, u32)> = None;
        for &c in &cands {
            if sel.contains(&c) {
                continue;
            }
            let mut trial = sel.clone();
            trial.push(c);
            let t = total_cover(n_cu, ops, &trial, max_tm);
            if t < cur && best.is_none_or(|(bt, _)| t < bt) {
                best = Some((t, c));
            }
        }
        match best {
            Some((_, c)) => sel.push(c),
            None => break, // no remaining rung helps
        }
    }
    sel.sort_unstable();
    sel.dedup();
    sel
}

/// The prefill GEMM mix of a dense-GQA model, per the emitter's own sharding.
///
/// `q`/`k`/`v` are emitted on DISJOINT proportional CU sets (`split3`, `lib.rs:1780-1815`) and run
/// concurrently, so they are modelled as ONE op of `tn_q + tn_k (+ tn_v)` tiles against all
/// `n_cu` — not as separate ops each against the whole machine. Modelling them serialized
/// overstates the chunk-128 floor by 13% (60.8 vs 53.9 ms on Gemma-4-12B at n_cu=170).
///
/// `gate|up` is the fused `GemmGlu` (one op, `glu = true`). `o` and `down` take `b.all()`.
pub(crate) fn ladder_ops(c: &crate::config::Cfg, bn: u32) -> Vec<LadderOp> {
    let bn = bn.max(1);
    let tp = c.tp.max(1);
    let full = c.is_full.iter().filter(|&&f| f).count() as u32;
    let slide = c.layers.saturating_sub(full);
    let hidden_t = c.hidden.div_ceil(bn);
    let inter_l = (c.inter / tp).max(1);

    let mut ops = Vec::with_capacity(8);
    for (count, hd, kvh) in [(slide, c.hd_slide, c.kvh_slide), (full, c.hd_full, c.kvh_full)] {
        if count == 0 {
            continue;
        }
        let qd = (c.heads / tp).max(1) * hd;
        let kd = (kvh / tp).max(1) * hd;
        // q + k (+ v when the arch does not share k as v) as one concurrent group.
        let vt = if c.k_eq_v { 0 } else { kd.div_ceil(bn) };
        ops.push(LadderOp {
            tn: qd.div_ceil(bn) + kd.div_ceil(bn) + vt,
            k: c.hidden,
            glu: false,
            count,
        });
        // o_proj: N = hidden, K = qd.
        ops.push(LadderOp { tn: hidden_t, k: qd, glu: false, count });
    }
    // gate|up fused, and down — identical in every layer.
    ops.push(LadderOp { tn: inter_l.div_ceil(bn), k: c.hidden, glu: true, count: c.layers });
    ops.push(LadderOp { tn: hidden_t, k: inter_l, glu: false, count: c.layers });
    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gemma-4-12B at tp=1: hidden 3840, inter 15360, 16 heads, 8 kv heads,
    /// 40 sliding layers (hd 256) + 8 full layers (hd 512), k_eq_v.
    pub(super) fn g4_12b() -> Vec<LadderOp> {
        vec![
            LadderOp { tn: 32 + 16, k: 3840, glu: false, count: 40 }, // qkv sliding
            LadderOp { tn: 30, k: 4096, glu: false, count: 40 },      // o sliding
            LadderOp { tn: 64 + 32, k: 3840, glu: false, count: 8 },  // qkv full
            LadderOp { tn: 30, k: 8192, glu: false, count: 8 },       // o full
            LadderOp { tn: 120, k: 3840, glu: true, count: 48 },      // gate|up
            LadderOp { tn: 30, k: 15360, glu: false, count: 48 },     // down
        ]
    }

    #[test]
    fn cost_is_a_staircase_flat_between_wave_boundaries() {
        let ops = g4_12b();
        // down/o carry tn=30: ceil(30*tm/170) is constant for tm in 1..=5, so 5 costs the same
        // as 4 for those ops. The measured layer total at tm=4 and tm=5 differs only by the ops
        // that actually stepped -- it is never a smooth ramp.
        let c: Vec<u64> = (1..=8).map(|tm| launch_cost(170, &ops, tm)).collect();
        assert!(c.windows(2).all(|w| w[1] >= w[0]), "cost must be monotone in tm");
        // tm=6 is the big cliff: tn=30 ops (down, o) step from 1 wave to 2.
        let step6 = c[5] - c[4];
        let step7 = c[6] - c[5];
        assert!(step6 > step7 * 3, "tm=6 must be the dominant cliff, got {step6} vs {step7}");
    }

    #[test]
    fn tread_tops_are_not_powers_of_two() {
        let tops = tread_tops(170, &g4_12b(), 32);
        // 5 (=640 rows), 11 (=1408), 17 (=2176) are tread tops; 4 (=512) and 16 (=2048) are not.
        for tm in [5u32, 11, 17] {
            assert!(tops.contains(&tm), "tm {tm} should be a tread top, got {tops:?}");
        }
        assert!(!tops.contains(&16), "tm 16 (2048 rows) is mid-tread, got {tops:?}");
    }

    #[test]
    fn wave_ladder_beats_the_shipped_power_of_two_rungs() {
        let ops = g4_12b();
        let shipped = [1u32, 4, 8, 16, 32]; // 128, 512, 1024, 2048, 4096
        let wave = wave_ladder(170, &ops, 32, shipped.len());
        assert_eq!(wave.len(), shipped.len(), "same rung budget: {wave:?}");
        let best = total_cover(170, &ops, &(1..=32).collect::<Vec<_>>(), 32) as f64;
        let s = total_cover(170, &ops, &shipped, 32) as f64;
        let w = total_cover(170, &ops, &wave, 32) as f64;
        let (sl, wl) = (100.0 * (s - best) / best, 100.0 * (w - best) / best);
        assert!(w < s, "wave ladder {wave:?} ({wl:.2}%) must beat shipped ({sl:.2}%)");
        assert!(wl < sl / 2.0, "expected >2x better, shipped {sl:.2}% vs wave {wl:.2}%");
    }

    #[test]
    fn ladder_depends_on_sm_count() {
        // 170 = 2*5*17 and 188 = 2^2*47 put the treads in different places. A ladder tuned for
        // one is wrong on the other -- which is why this is derived, not hardcoded.
        let ops = g4_12b();
        assert_ne!(wave_ladder(170, &ops, 32, 8), wave_ladder(188, &ops, 32, 8));
    }

    #[test]
    fn rung_one_always_present_and_ladder_is_sorted_unique() {
        for p in [1u32, 64, 132, 170, 188, 256] {
            let l = wave_ladder(p, &g4_12b(), 32, 6);
            assert_eq!(l[0], 1, "MIN_CHUNK rung missing for n_cu={p}: {l:?}");
            assert!(l.windows(2).all(|w| w[0] < w[1]), "not sorted/unique: {l:?}");
        }
    }

    #[test]
    fn degenerate_inputs_do_not_panic() {
        assert_eq!(wave_ladder(170, &[], 32, 4), vec![1]);
        assert_eq!(wave_ladder(170, &g4_12b(), 1, 4), vec![1]);
        assert_eq!(launch_cost(0, &g4_12b(), 1), launch_cost(1, &g4_12b(), 1));
    }
}

/// Pins the ladders the cost model actually produces. These are the numbers quoted in
/// `perf-data/px6-sm-quantization.md`; a change here is a deliberate change to the cost model,
/// not an incidental one.
#[cfg(test)]
mod pinned {
    use super::*;
    use super::tests::g4_12b;

    fn rows(p: u32, cap_tm: u32, budget: usize) -> Vec<u32> {
        wave_ladder(p, &g4_12b(), cap_tm, budget).into_iter().map(|t| t * 128).collect()
    }
    fn loss(p: u32, cap_tm: u32, ladder: &[u32]) -> f64 {
        let ops = g4_12b();
        let all: Vec<u32> = (1..=cap_tm).collect();
        let best = total_cover(p, &ops, &all, cap_tm) as f64;
        let tm: Vec<u32> = ladder.iter().map(|x| x / 128).collect();
        100.0 * (total_cover(p, &ops, &tm, cap_tm) as f64 - best) / best
    }

    const SHIPPED: [u32; 6] = [128, 512, 1024, 2048, 4096, 8192];

    #[test]
    fn ladder_170sm_8192cap() {
        assert_eq!(rows(170, 64, 6), vec![128, 512, 1408, 2176, 2688, 8192]);
        assert!((loss(170, 64, &SHIPPED) - 7.03).abs() < 0.05, "{}", loss(170, 64, &SHIPPED));
        assert!(loss(170, 64, &rows(170, 64, 6)) < 1.5);
    }

    #[test]
    fn ladder_188sm_8192cap_is_different_and_shipped_is_worse_there() {
        assert_eq!(rows(188, 64, 6), vec![128, 384, 768, 2176, 3200, 8192]);
        // The power-of-two ladder is WORSE on 188 SMs than on 170 -- and 188 is the card most
        // existing campaigns in perf-data/ used.
        assert!(loss(188, 64, &SHIPPED) > loss(170, 64, &SHIPPED));
        assert!((loss(188, 64, &SHIPPED) - 11.41).abs() < 0.05, "{}", loss(188, 64, &SHIPPED));
        assert!(loss(188, 64, &rows(188, 64, 6)) < 1.5);
    }

    #[test]
    fn no_chosen_rung_is_a_power_of_two_except_the_endpoints() {
        for &r in &rows(170, 64, 6) {
            if r == 128 || r == 512 || r == 8192 {
                continue; // endpoints / MIN_CHUNK can coincide with the old ladder
            }
            assert!(!r.is_power_of_two(), "rung {r} is a power of two -- treads are not");
        }
    }
}
