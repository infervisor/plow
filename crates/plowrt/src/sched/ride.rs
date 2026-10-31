//! Whether a tick's decode rows ride its prefill launch or take their own decode step.
//!
//! A decode row inside a prefill launch shares the launch's weight pass but pays the prefill
//! attention path per row; a standalone decode step pays its own weight pass once for all rows.
//! Which is cheaper depends on the model, the context and the batch (Gemma-4-E4B at ISL 1000:
//! riding 49 rows costs 7.9 ms against a 13.4 ms step, riding 97 rows 15.7 ms against ~16 ms and
//! the non-riding arm serves 10% more at c128; Gemma-4-12B at 8192 rides at 0.50 ms/row against
//! 0.27). So the choice is measured per engine, not configured: every launch and step feeds the
//! estimates, and the launch rides when `ride_ms(bucket) * rows <= step_ms(rows)`.
//!
//! Riding is the default while an estimate is missing. One launch in [`EXPLORE`] per bucket takes
//! the arm the policy did not choose, so neither estimate goes stale while the other wins.
//!
//! A launch's cost is the median of its bucket's last [`K`] samples, not a running mean: one
//! stalled launch (a 70 ms rung-128 tail with 3 riders against a 31 ms norm) priced a rider at
//! 13 ms, and the bucket then declined 1210 of 1276 rides at ~free rider cost (Qwen3.8-27B FP8).
//! Until a bucket holds `K` riding samples, and for `K / 2` explorations after its decision
//! flips, the other arm runs every [`EXPLORE_FAST`] launches so a contaminated window refills.

/// Launches per bucket between forced runs of the arm the policy is not using.
pub const EXPLORE: u32 = 32;
/// The exploration interval while a bucket's riding window is filling or just flipped.
pub const EXPLORE_FAST: u32 = 4;
/// Samples per estimate window.
pub const K: usize = 8;
const ALPHA: f64 = 0.2;
const SUMMARY_EVERY: u64 = 2048;

/// The last [`K`] samples; `median` of an even count is the mean of the middle pair.
#[derive(Clone, Copy, Debug, Default)]
struct Window {
    v: [f64; K],
    len: usize,
    next: usize,
}

impl Window {
    fn add(&mut self, x: f64) {
        self.v[self.next] = x;
        self.next = (self.next + 1) % K;
        self.len = (self.len + 1).min(K);
    }

    fn median(&self) -> Option<f64> {
        if self.len == 0 {
            return None;
        }
        let mut v = self.v;
        let s = &mut v[..self.len];
        s.sort_unstable_by(f64::total_cmp);
        let m = self.len / 2;
        Some(if self.len % 2 == 1 {
            s[m]
        } else {
            (s[m - 1] + s[m]) / 2.0
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Ewma(Option<f64>);

impl Ewma {
    fn add(&mut self, x: f64) {
        self.0 = Some(self.0.map_or(x, |m| m + ALPHA * (x - m)));
    }
}

#[derive(Clone, Debug, Default)]
struct Bucket {
    rows: usize,
    /// Launch ms with no decode rows.
    pure: Window,
    /// Per riding row: launch ms over the `pure` median, divided by the riding rows.
    per_row: Window,
    /// Launch ms with or without decode rows: what a riding row waits for.
    total: Ewma,
    since_explore: u32,
    /// Explorations left at [`EXPLORE_FAST`] after the decision flipped, and the last decision.
    fast: u32,
    last: Option<bool>,
    /// Decisions: rode, declined, and how many of either were forced explorations.
    rode: u64,
    declined: u64,
    explored: u64,
}

/// Per-engine cost estimates for [`RideCost::ride`].
#[derive(Debug, Default)]
pub struct RideCost {
    buckets: Vec<Bucket>,
    /// Standalone decode step ms, keyed by the covering width in `widths`.
    steps: Vec<(usize, Ewma)>,
    widths: Vec<usize>,
    /// The engine's prefill rungs, ascending (for [`RideCost::rows_within`]).
    ladder: Vec<usize>,
    /// [`RideCost::rows_within`] calls since the measured rung above the cap last re-ran.
    since_reprobe: u32,
    decisions: u64,
}

impl RideCost {
    /// `widths`: the engine's decode rung widths, ascending (step costs are kept per rung).
    pub fn set_widths(&mut self, widths: &[usize]) {
        if self.widths != widths {
            self.widths = widths.to_vec();
            self.steps = widths.iter().map(|&w| (w, Ewma::default())).collect();
        }
    }

    pub fn needs_widths(&self) -> bool {
        self.widths.is_empty()
    }

    pub fn set_ladder(&mut self, ladder: &[usize]) {
        if self.ladder != ladder {
            self.ladder = ladder.to_vec();
        }
    }

    pub fn needs_ladder(&self) -> bool {
        self.ladder.is_empty()
    }

    fn bucket(&mut self, rows: usize) -> &mut Bucket {
        let i = match self.buckets.iter().position(|b| b.rows == rows) {
            Some(i) => i,
            None => {
                self.buckets.push(Bucket {
                    rows,
                    ..Default::default()
                });
                self.buckets.len() - 1
            }
        };
        &mut self.buckets[i]
    }

    fn step(&self, rows: usize) -> Option<f64> {
        let i = self.widths.iter().position(|&w| w >= rows)?;
        self.steps[i].1 .0
    }

    /// One prefill launch of `bucket` rows that carried `decode_rows` riding rows, `ms` long.
    pub fn observe_launch(&mut self, bucket: usize, decode_rows: usize, ms: f64) {
        let b = self.bucket(bucket);
        match (decode_rows, b.pure.median()) {
            (0, _) => b.pure.add(ms),
            (d, Some(pure)) => b.per_row.add((ms - pure).max(0.0) / d as f64),
            _ => {}
        }
    }

    /// One launch of the rung actually launched (`bucket` rows, riders included), `ms` long.
    pub fn observe_rung(&mut self, bucket: usize, ms: f64) {
        self.bucket(bucket).total.add(ms);
    }

    /// One standalone decode step over `rows` rows, `ms` per step.
    pub fn observe_step(&mut self, rows: usize, ms: f64) {
        if let Some(i) = self.widths.iter().position(|&w| w >= rows) {
            self.steps[i].1.add(ms);
        }
    }

    /// `(any rung measured, index of the widest measured rung within budget_ms)`.
    fn fit(&self, budget_ms: f64) -> (bool, Option<usize>) {
        let mut cap = None;
        let mut any = false;
        for (i, &rows) in self.ladder.iter().enumerate() {
            if let Some(m) = self.rung_ms(rows) {
                any = true;
                if m <= budget_ms {
                    cap = Some(i);
                }
            }
        }
        (any, cap)
    }

    fn rung_ms(&self, rows: usize) -> Option<f64> {
        self.buckets
            .iter()
            .find(|b| b.rows == rows)
            .and_then(|b| b.total.0)
    }

    /// The widest measured rung within `budget_ms` (the narrowest when none fits), without the
    /// exploration [`RideCost::rows_within`] adds. `None` until a rung has been measured.
    pub fn rows_fit(&self, budget_ms: f64) -> Option<usize> {
        let (any, cap) = self.fit(budget_ms);
        any.then(|| self.ladder[cap.unwrap_or(0)])
    }

    /// The widest rung of the ladder a riding row can wait for under `budget_ms`: the widest
    /// measured rung within the budget, or the rung right above it while that one is unmeasured
    /// (its first launch measures it) and once per [`EXPLORE`] calls when it is measured.
    /// `None` until a rung has been measured.
    pub fn rows_within(&mut self, budget_ms: f64) -> Option<usize> {
        let (any, cap) = self.fit(budget_ms);
        if !any {
            return None;
        }
        let next = cap.map_or(0, |i| i + 1);
        let explore = self
            .ladder
            .get(next)
            .is_some_and(|&rows| self.rung_ms(rows).is_none());
        // Only launching a rung re-measures it, and a budgeted launch never runs the rung above
        // the cap: one slow sample (a host stall) priced it over the budget for good, so the cap
        // could only fall, rung by rung (G26 at 2.2 sessions/s: 1152 -> 128 rows in 7 minutes).
        let reprobe = !explore && next < self.ladder.len() && {
            self.since_reprobe += 1;
            self.since_reprobe >= EXPLORE
        };
        if reprobe {
            self.since_reprobe = 0;
        }
        Some(
            self.ladder[if explore || reprobe {
                next
            } else {
                cap.unwrap_or(0)
            }],
        )
    }

    /// Whether `decode_rows` rows should ride the next launch of `bucket` rows.
    pub fn ride(&mut self, bucket: usize, decode_rows: usize) -> bool {
        let step = self.step(decode_rows);
        let b = self.bucket(bucket);
        let prefer = match (b.pure.median(), b.per_row.median(), step) {
            (Some(_), Some(r), Some(s)) => r * decode_rows as f64 <= s,
            // Without a pure sample neither the per-row cost nor its comparison exists: measure it.
            (None, _, _) => false,
            _ => true,
        };
        if b.last.is_some_and(|last| last != prefer) {
            b.fast = (K / 2) as u32;
        }
        b.last = Some(prefer);
        let interval = if b.per_row.len < K || b.fast > 0 {
            EXPLORE_FAST
        } else {
            EXPLORE
        };
        b.since_explore += 1;
        let explore = b.pure.len > 0 && b.since_explore >= interval;
        if explore {
            b.since_explore = 0;
            b.fast = b.fast.saturating_sub(1);
            b.explored += 1;
        }
        let ride = prefer != explore;
        if ride {
            b.rode += 1;
        } else {
            b.declined += 1;
        }
        // A server stopped by a signal never drops its engines: log the counts as they grow too.
        self.decisions += 1;
        if self.decisions.is_multiple_of(SUMMARY_EVERY) {
            if let Some(summary) = self.decision_summary() {
                tracing::info!(%summary, "ride decisions per prefill bucket (rows:rode/declined/explored)");
            }
        }
        ride
    }

    /// One line per bucket with any decision: `rows:rode/declined/explored`.
    pub fn decision_summary(&self) -> Option<String> {
        let parts: Vec<String> = self
            .buckets
            .iter()
            .filter(|b| b.rode + b.declined > 0)
            .map(|b| format!("{}:{}/{}/{}", b.rows, b.rode, b.declined, b.explored))
            .collect();
        (!parts.is_empty()).then(|| parts.join(" "))
    }
}

impl Drop for RideCost {
    fn drop(&mut self) {
        if let Some(summary) = self.decision_summary() {
            tracing::info!(%summary, "ride decisions per prefill bucket (rows:rode/declined/explored)");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warm(c: &mut RideCost, pure: f64, per_row: f64, step: f64) {
        c.set_widths(&[32, 64, 128]);
        for _ in 0..K {
            c.observe_launch(2048, 0, pure);
        }
        for _ in 0..K {
            c.observe_launch(2048, 50, pure + 50.0 * per_row);
        }
        for rows in [20, 50, 100] {
            c.observe_step(rows, step);
        }
    }

    /// Decisions over `n` launches of `bucket` with `rows` riders, each observed at the cost of
    /// its arm (`launch(ride)`); returns how many rode.
    fn run(
        c: &mut RideCost,
        n: u32,
        bucket: usize,
        rows: usize,
        launch: impl Fn(bool) -> f64,
    ) -> u32 {
        let mut rode = 0;
        for _ in 0..n {
            let r = c.ride(bucket, rows);
            rode += u32::from(r);
            c.observe_launch(bucket, if r { rows } else { 0 }, launch(r));
        }
        rode
    }

    #[test]
    fn one_stalled_launch_does_not_stop_riding() {
        // Rung-128 tail: 31 ms pure, ~free riders, 17 ms decode step.
        let mut c = RideCost::default();
        warm(&mut c, 31.0, 0.05, 17.0);
        c.observe_launch(2048, 3, 70.8);
        c.observe_launch(2048, 2, 45.2);
        assert!(
            c.ride(2048, 3),
            "two stalls in a window of eight leave the median cheap"
        );
        let rode = run(&mut c, 64, 2048, 4, |r| if r { 31.2 } else { 31.0 });
        assert!(rode >= 64 - 64 / EXPLORE - 1, "rode {rode} of 64");
    }

    #[test]
    fn a_sustained_high_rider_cost_still_declines() {
        // Gemma-4-12B at 8192: 0.50 ms/row riding against a 0.27 ms/row step.
        let mut c = RideCost::default();
        warm(&mut c, 120.0, 0.50, 27.0);
        let rode = run(&mut c, 64, 2048, 100, |r| if r { 170.0 } else { 120.0 });
        assert!(
            rode <= 64 / EXPLORE + 1,
            "rode {rode} of 64: only explorations"
        );
    }

    #[test]
    fn a_contaminated_start_recovers_within_a_bounded_number_of_launches() {
        let mut c = RideCost::default();
        c.set_widths(&[64]);
        c.observe_step(8, 17.0);
        c.observe_launch(128, 0, 31.0);
        // The first riding samples all stalled.
        for _ in 0..K {
            c.observe_launch(128, 4, 90.0);
        }
        assert!(!c.ride(128, 4));
        // Riding really costs ~nothing: the fast explorations refill the window.
        run(&mut c, 8 * EXPLORE, 128, 4, |r| if r { 31.4 } else { 31.0 });
        let rode = run(&mut c, 64, 128, 4, |r| if r { 31.4 } else { 31.0 });
        assert!(
            rode >= 64 - 64 / EXPLORE - 1,
            "rode {rode} of 64 after recovery"
        );
    }

    #[test]
    fn a_turn_from_cheap_to_expensive_riders_is_followed() {
        let mut c = RideCost::default();
        warm(&mut c, 31.0, 0.05, 17.0);
        run(
            &mut c,
            4 * EXPLORE,
            2048,
            8,
            |r| if r { 80.0 } else { 31.0 },
        );
        let rode = run(&mut c, 64, 2048, 8, |r| if r { 80.0 } else { 31.0 });
        assert!(
            rode <= 64 / EXPLORE + 1,
            "rode {rode} of 64 once riders turned expensive"
        );
    }

    #[test]
    fn decisions_are_counted_per_bucket() {
        let mut c = RideCost::default();
        warm(&mut c, 31.0, 0.05, 17.0);
        run(&mut c, EXPLORE, 2048, 4, |_| 31.0);
        let s = c.decision_summary().unwrap();
        assert_eq!(s, format!("2048:{}/1/1", EXPLORE - 1));
    }

    #[test]
    fn rides_only_while_the_rows_cost_less_than_a_step() {
        let mut c = RideCost::default();
        warm(&mut c, 45.0, 0.16, 14.0);
        assert!(c.ride(2048, 50)); // 8.0 <= 14
        assert!(!c.ride(2048, 100)); // 16.0 > 14
    }

    #[test]
    fn a_missing_pure_sample_is_measured_first_and_explores_the_other_arm() {
        let mut c = RideCost::default();
        c.set_widths(&[64]);
        assert!(
            !c.ride(1024, 10),
            "no pure sample yet: run the launch without riders"
        );
        warm(&mut c, 45.0, 0.16, 14.0);
        let arms: Vec<bool> = (0..EXPLORE).map(|_| c.ride(2048, 50)).collect();
        assert_eq!(
            arms.iter().filter(|&&r| !r).count(),
            1,
            "one explored non-ride per window"
        );
    }

    #[test]
    fn the_itl_budget_takes_the_widest_measured_rung_and_explores_one_above() {
        let mut c = RideCost::default();
        c.set_ladder(&[128, 512, 1024, 1088, 2048, 4096, 8192]);
        assert_eq!(c.rows_within(60.0), None, "nothing measured yet");
        c.observe_rung(512, 35.0);
        assert_eq!(c.rows_within(60.0), Some(1024), "1024 unmeasured: try it");
        c.observe_rung(1024, 50.0);
        assert_eq!(c.rows_within(60.0), Some(1088), "1088 unmeasured: try it");
        c.observe_rung(1088, 80.0);
        assert_eq!(c.rows_within(60.0), Some(1024), "1088 over budget");
        assert_eq!(
            c.rows_within(90.0),
            Some(2048),
            "2048 unmeasured above 1088"
        );
        c.observe_rung(2048, 64.0);
        assert_eq!(c.rows_within(60.0), Some(1024));
        assert_eq!(
            c.rows_within(70.0),
            Some(4096),
            "2048 fits 70 ms; the over-budget 1088 below it does not block exploring upward"
        );
        c.observe_rung(4096, 115.0);
        c.observe_rung(8192, 220.0);
        assert_eq!(c.rows_within(70.0), Some(2048));
        assert_eq!(
            c.rows_within(10.0),
            Some(128),
            "nothing fits: the narrowest rung"
        );
    }

    #[test]
    fn an_over_budget_rung_is_re_measured_so_the_cap_can_climb_back() {
        let mut c = RideCost::default();
        c.set_ladder(&[512, 1024, 1152, 2048]);
        c.observe_rung(512, 25.0);
        c.observe_rung(1024, 36.0);
        c.observe_rung(1152, 44.0);
        c.observe_rung(2048, 90.0);
        // One stalled 1152 launch prices it over the budget.
        c.observe_rung(1152, 250.0);
        assert_eq!(
            c.rows_within(60.0),
            Some(1024),
            "1152 priced over the budget"
        );
        for _ in 0..20 * EXPLORE {
            let cap = c.rows_within(60.0).unwrap();
            // Each launch measures its rung at its real cost.
            c.observe_rung(
                cap,
                [36.0, 44.0, 90.0][[1024, 1152, 2048].iter().position(|&r| r == cap).unwrap()],
            );
        }
        let picks: Vec<usize> = (0..EXPLORE).map(|_| c.rows_within(60.0).unwrap()).collect();
        assert_eq!(
            picks.iter().filter(|&&r| r == 1152).count() as u32,
            EXPLORE - 1,
            "the cap climbed back to 1152 once its re-runs fit the budget"
        );
        assert_eq!(
            picks.iter().filter(|&&r| r == 2048).count(),
            1,
            "the over-budget rung above re-runs once per window"
        );
    }

    #[test]
    fn a_step_estimate_is_needed_to_leave_the_default() {
        let mut c = RideCost::default();
        c.set_widths(&[64]);
        c.observe_launch(2048, 0, 45.0);
        c.observe_launch(2048, 40, 70.0);
        assert!(c.ride(2048, 40), "no step estimate: keep riding");
    }
}
