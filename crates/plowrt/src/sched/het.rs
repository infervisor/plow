//! Deciding whether a queued request's prompt head runs on the CPU, and how
//! long a head that does may be.
//!
//! A head exists only because a queue exists. Every decision here is driven by
//! the estimates the mux already publishes — `LoadEstimator`'s arrival EWMA and
//! measured service time, and `predicted_wait_ms` — so this adds a policy over
//! the admission signals rather than a second scheduler.
//!
//! Pure: no engine, no device, no clock. The mux supplies the numbers.

use rustc_hash::FxHashMap;

/// Utilisation at or above which a head may run. The same threshold
/// `sched::admission::admit` uses to separate "run now" from "hold to form a
/// batch", for the same reason: below it there is no wait to hide a head in, so
/// a head is pure added TTFT.
pub const ARM_UTIL: f64 = 0.85;
/// Utilisation below which the head pool disarms. Lower than [`ARM_UTIL`] so a
/// burst does not flap the pool in and out of its spin state.
pub const DISARM_UTIL: f64 = 0.75;

/// What one CPU head pass costs, measured per model on the serving host.
///
/// `fixed_ms` dominates for a big MoE — a pass streams the weight set from DRAM
/// and 28 rows read almost what 512 do — so a head is budgeted in wall time and
/// there is no saving in running a shorter one than the wait affords.
#[derive(Clone, Copy, Debug)]
pub struct HeadCost {
    pub fixed_ms: f64,
    pub per_row_ms: f64,
    /// KV bytes one prompt row adds to the handoff.
    pub xfer_bytes_per_row: u64,
    /// Effective host-to-device bandwidth for the handoff.
    pub xfer_bytes_per_ms: f64,
}

impl HeadCost {
    /// Wall time of a head of `rows`, including the transfer that must land
    /// before the device may run the body.
    pub fn total_ms(&self, rows: u32) -> f64 {
        let compute = self.fixed_ms + self.per_row_ms * f64::from(rows);
        let xfer = if self.xfer_bytes_per_ms > 0.0 {
            (self.xfer_bytes_per_row * u64::from(rows)) as f64 / self.xfer_bytes_per_ms
        } else {
            0.0
        };
        compute + xfer
    }

    /// Largest head that finishes and transfers inside `wait_ms`, leaving
    /// `margin_ms` for the tick that will admit the request.
    ///
    /// Zero when the fixed cost alone does not fit: a head that will not
    /// complete is one the device abandons, so starting it buys nothing.
    pub fn budget_rows(&self, wait_ms: f64, margin_ms: f64, cap: u32) -> u32 {
        let have = wait_ms - margin_ms;
        if have <= 0.0 || self.total_ms(0) > have {
            return 0;
        }
        let per_row = self.per_row_ms
            + if self.xfer_bytes_per_ms > 0.0 {
                self.xfer_bytes_per_row as f64 / self.xfer_bytes_per_ms
            } else {
                0.0
            };
        if per_row <= 0.0 {
            return cap;
        }
        let rows = ((have - self.fixed_ms) / per_row).floor();
        if rows <= 0.0 {
            return 0;
        }
        (rows as u64).min(u64::from(cap)) as u32
    }
}

/// A head the planner decided to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HetPlan {
    pub head_rows: u32,
}

/// Choose a head length for a prompt of `prompt_rows` facing `wait_ms` of
/// queue: the LARGEST head the wait affords.
///
/// # There is deliberately no snap to a rung boundary
///
/// The obvious refinement is to nudge the head so the device is left a whole
/// number of compiled rungs. It cannot help, and the reason is the same
/// property that makes any head cover-safe: **device cost is monotone
/// non-increasing in rows removed.** AMD's ragged cover is
/// `ceil(rows / max_bucket)` launches, and any cover valid for `r + 1` rows is
/// valid for `r`, so NVIDIA's DP cost cannot rise as rows fall either.
///
/// A larger head is therefore never a worse cover than a smaller one, and the
/// largest affordable head already lands on or past whatever boundary a snap
/// would have aimed for. Alignment is subsumed, not forgone: if the budget
/// reaches the remainder, the rung is won automatically; if it does not, no
/// choice within the window would have won it either.
///
/// This is also why the backend's cover planner is neither called nor
/// reimplemented here. It stays exactly as it is, and this cannot disagree with
/// it.
///
/// `min_gpu_rows` keeps real work on the device: a head must not quietly grow
/// into a CPU-served request, which is a separate decision with its own quality
/// contract.
pub fn plan_head(
    util: f64,
    wait_ms: f64,
    prompt_rows: u32,
    cost: &HeadCost,
    cap: u32,
    margin_ms: f64,
    min_gpu_rows: u32,
) -> Option<HetPlan> {
    if util < ARM_UTIL {
        return None;
    }
    let headroom = prompt_rows.saturating_sub(min_gpu_rows);
    let head_rows = cost.budget_rows(wait_ms, margin_ms, cap).min(headroom);
    (head_rows > 0).then_some(HetPlan { head_rows })
}

/// Arming hysteresis: the head pool follows the queue, not each tick.
#[derive(Clone, Debug, Default)]
pub struct Arm {
    armed: bool,
    hi_ticks: u32,
}

impl Arm {
    /// Fold one tick's utilisation in and return whether heads may run.
    ///
    /// Disarming is immediate in both of its causes — falling load and a
    /// tripped contention guard — while arming needs `needed` consecutive busy
    /// ticks. Asymmetric on purpose: the cost of arming late is a missed
    /// offload, the cost of disarming late is GPU tick latency.
    pub fn observe(&mut self, util: f64, guard_tripped: bool, needed: u32) -> bool {
        if guard_tripped || util < DISARM_UTIL {
            self.armed = false;
            self.hi_ticks = 0;
            return false;
        }
        if util >= ARM_UTIL {
            self.hi_ticks = self.hi_ticks.saturating_add(1);
            if self.hi_ticks >= needed.max(1) {
                self.armed = true;
            }
        } else {
            self.hi_ticks = 0;
        }
        self.armed
    }

    pub fn armed(&self) -> bool {
        self.armed
    }
}

/// The closed-loop half of the non-regression guarantee.
///
/// Affinity and `SCHED_IDLE` bound what the head pool can preempt; neither
/// bounds memory bandwidth or PCIe, and those leak through static isolation. So
/// the guard watches the statistic the runtime already maintains — the mux's
/// measured per-tick service time — against what it was with the pool idle, and
/// disarms when the armed value drifts past a limit.
///
/// **Baselines are per shape.** Service time is a function of the decode rung
/// and live batch, so one global baseline would trip on a rung change and stay
/// silent through real contention at another. A shape never seen disarmed is
/// not judged at all: no baseline, no verdict, rather than a guess.
pub struct ContentionGuard {
    baseline_ms: FxHashMap<u32, f64>,
    breaches: u32,
    tripped: bool,
    limit_pct: f64,
    needed: u32,
    alpha: f64,
}

impl ContentionGuard {
    pub fn new(limit_pct: f64, needed: u32) -> Self {
        ContentionGuard {
            baseline_ms: FxHashMap::default(),
            breaches: 0,
            tripped: false,
            limit_pct,
            needed: needed.max(1),
            alpha: 0.2,
        }
    }

    /// Record a tick that ran with the head pool idle.
    pub fn observe_disarmed(&mut self, shape: u32, service_ms: f64) {
        if service_ms <= 0.0 {
            return;
        }
        let e = self.baseline_ms.entry(shape).or_insert(service_ms);
        *e = self.alpha * service_ms + (1.0 - self.alpha) * *e;
    }

    /// Record a tick that ran with heads in flight; returns whether the guard
    /// is now tripped.
    pub fn observe_armed(&mut self, shape: u32, service_ms: f64) -> bool {
        let Some(&base) = self.baseline_ms.get(&shape) else {
            return self.tripped;
        };
        if service_ms > base * (1.0 + self.limit_pct / 100.0) {
            self.breaches += 1;
            if self.breaches >= self.needed {
                self.tripped = true;
            }
        } else {
            self.breaches = 0;
        }
        self.tripped
    }

    pub fn tripped(&self) -> bool {
        self.tripped
    }

    /// Clear the trip after a quiet period. The baselines are kept: they
    /// describe the machine, not the episode.
    pub fn rearm(&mut self) {
        self.tripped = false;
        self.breaches = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A big MoE: the pass streams the weight set, so it is fixed-cost heavy.
    fn moe() -> HeadCost {
        HeadCost {
            fixed_ms: 800.0,
            per_row_ms: 2.0,
            xfer_bytes_per_row: 88_000,
            xfer_bytes_per_ms: 50_000_000.0,
        }
    }

    #[test]
    fn no_queue_means_no_head() {
        // Below the arm threshold there is no wait to hide the pass in.
        assert_eq!(plan_head(0.5, 5_000.0, 4096, &moe(), 4096, 50.0, 128), None);
    }

    #[test]
    fn a_wait_shorter_than_the_fixed_pass_buys_nothing() {
        // The head would be abandoned before it completed, so it is not started.
        assert_eq!(plan_head(0.95, 700.0, 4096, &moe(), 4096, 50.0, 128), None);
        assert_eq!(moe().budget_rows(700.0, 50.0, 4096), 0);
        // Just past the fixed cost the head is small but real: 800 ms of idle
        // CPU for 24 rows is a poor rate and a free one, and those are rows the
        // device does not run.
        assert_eq!(moe().budget_rows(900.0, 50.0, 4096), 24);
    }

    #[test]
    fn a_deep_queue_affords_a_real_head() {
        // The conc=20 regime: minutes of wait, so the budget is rows not tokens.
        let n = moe().budget_rows(30_000.0, 50.0, 100_000);
        assert!(n > 5_000, "budget was {n}");
        // The transfer is part of the budget, not a footnote: at 88 KB/row it
        // is ~1.76 ms per 1000 rows against 2 ms/row of compute.
        let with_xfer = moe().total_ms(n);
        assert!(with_xfer <= 30_000.0 - 50.0, "budget overruns the wait");
    }

    #[test]
    fn the_budget_leaves_a_margin_for_the_admitting_tick() {
        let cost = HeadCost {
            fixed_ms: 0.0,
            per_row_ms: 1.0,
            xfer_bytes_per_row: 0,
            xfer_bytes_per_ms: 0.0,
        };
        assert_eq!(cost.budget_rows(1_000.0, 200.0, 10_000), 800);
    }

    #[test]
    fn a_head_never_consumes_the_whole_prompt() {
        // min_gpu_rows keeps real work on the device: a head must not quietly
        // become a CPU-served request.
        let cost = HeadCost {
            fixed_ms: 0.0,
            per_row_ms: 0.001,
            xfer_bytes_per_row: 0,
            xfer_bytes_per_ms: 0.0,
        };
        let p = plan_head(0.95, 1e9, 4096, &cost, u32::MAX, 0.0, 128).unwrap();
        assert_eq!(p.head_rows, 4096 - 128);
    }

    /// The 4124-row case, and why no rung snap is needed to serve it.
    ///
    /// A ladder stepping at 4096 makes "leave the device 4096 rows" the cheap
    /// cover, reached by a 28-row head. A budget that affords 40 rows takes 40
    /// — leaving 4084, still one rung, and twelve more rows the device does not
    /// run. Snapping back to 28 would have been strictly worse.
    #[test]
    fn a_budget_past_the_rung_boundary_keeps_the_extra_rows() {
        let cover = |rows: u32| if rows <= 4096 { 1u64 } else { 2 };
        let cost = HeadCost {
            fixed_ms: 0.0,
            per_row_ms: 0.001,
            xfer_bytes_per_row: 0,
            xfer_bytes_per_ms: 0.0,
        };
        let p = plan_head(0.95, 40.0, 4124, &cost, 40, 0.0, 1).unwrap();
        assert_eq!(p.head_rows, 40);
        assert_eq!(cover(4124 - p.head_rows), 1, "the rung is won anyway");
    }

    /// And when the budget does not reach the boundary, no choice would have.
    #[test]
    fn a_budget_short_of_the_boundary_cannot_win_the_rung_either() {
        let cover = |rows: u32| if rows <= 4096 { 1u64 } else { 2 };
        let cost = HeadCost {
            fixed_ms: 0.0,
            per_row_ms: 0.001,
            xfer_bytes_per_row: 0,
            xfer_bytes_per_ms: 0.0,
        };
        let p = plan_head(0.95, 20.0, 4124, &cost, 20, 0.0, 1).unwrap();
        assert_eq!(p.head_rows, 20);
        assert_eq!(cover(4124 - p.head_rows), 2);
    }

    #[test]
    fn arming_needs_a_sustained_queue_but_disarming_is_immediate() {
        let mut arm = Arm::default();
        assert!(!arm.observe(0.9, false, 3));
        assert!(!arm.observe(0.9, false, 3));
        assert!(arm.observe(0.9, false, 3));
        // One quiet tick inside the band keeps it armed (hysteresis).
        assert!(arm.observe(0.8, false, 3));
        // Below the disarm threshold it drops at once.
        assert!(!arm.observe(0.7, false, 3));
        // And re-arming starts the count over.
        assert!(!arm.observe(0.9, false, 3));
    }

    #[test]
    fn a_tripped_guard_disarms_regardless_of_load() {
        let mut arm = Arm::default();
        for _ in 0..5 {
            arm.observe(0.99, false, 3);
        }
        assert!(arm.armed());
        assert!(!arm.observe(0.99, true, 3));
    }

    #[test]
    fn contention_trips_only_after_sustained_regression() {
        let mut g = ContentionGuard::new(5.0, 3);
        for _ in 0..10 {
            g.observe_disarmed(8, 100.0);
        }
        // Inside the limit: no breach.
        assert!(!g.observe_armed(8, 104.0));
        // Past it, but not yet for long enough.
        assert!(!g.observe_armed(8, 120.0));
        assert!(!g.observe_armed(8, 120.0));
        assert!(g.observe_armed(8, 120.0));
        assert!(g.tripped());
    }

    #[test]
    fn a_run_of_good_ticks_clears_the_breach_count() {
        let mut g = ContentionGuard::new(5.0, 3);
        g.observe_disarmed(8, 100.0);
        assert!(!g.observe_armed(8, 120.0));
        assert!(!g.observe_armed(8, 100.0));
        assert!(!g.observe_armed(8, 120.0));
        assert!(!g.observe_armed(8, 120.0));
        assert!(!g.tripped());
    }

    #[test]
    fn an_unseen_shape_is_not_judged() {
        // Service time is a function of the rung and live batch. Comparing a
        // rung-16 tick against a rung-8 baseline would trip on a shape change
        // and stay silent through real contention at another.
        let mut g = ContentionGuard::new(5.0, 1);
        g.observe_disarmed(8, 100.0);
        assert!(!g.observe_armed(16, 10_000.0));
        assert!(!g.tripped());
        // The shape it does know still trips.
        assert!(g.observe_armed(8, 10_000.0));
    }

    #[test]
    fn rearming_keeps_the_baselines() {
        let mut g = ContentionGuard::new(5.0, 1);
        g.observe_disarmed(8, 100.0);
        assert!(g.observe_armed(8, 200.0));
        g.rearm();
        assert!(!g.tripped());
        // The baseline describes the machine, not the episode.
        assert!(g.observe_armed(8, 200.0));
    }
}
