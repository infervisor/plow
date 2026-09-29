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

/// Launches per bucket between forced runs of the arm the policy is not using.
pub const EXPLORE: u32 = 32;
const ALPHA: f64 = 0.2;

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
    pure: Ewma,
    /// Extra launch ms over `pure`, and the riding rows that caused it (their ratio is the
    /// per-row cost, weighted by rows so a one-row launch's jitter does not set it).
    extra: Ewma,
    riders: Ewma,
    since_explore: u32,
}

/// Per-engine cost estimates for [`RideCost::ride`].
#[derive(Clone, Debug, Default)]
pub struct RideCost {
    buckets: Vec<Bucket>,
    /// Standalone decode step ms, keyed by the covering width in `widths`.
    steps: Vec<(usize, Ewma)>,
    widths: Vec<usize>,
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

    fn bucket(&mut self, rows: usize) -> &mut Bucket {
        let i = match self.buckets.iter().position(|b| b.rows == rows) {
            Some(i) => i,
            None => {
                self.buckets.push(Bucket { rows, ..Default::default() });
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
        match (decode_rows, b.pure.0) {
            (0, _) => b.pure.add(ms),
            (d, Some(pure)) => {
                b.extra.add((ms - pure).max(0.0));
                b.riders.add(d as f64);
            }
            _ => {}
        }
    }

    /// One standalone decode step over `rows` rows, `ms` per step.
    pub fn observe_step(&mut self, rows: usize, ms: f64) {
        if let Some(i) = self.widths.iter().position(|&w| w >= rows) {
            self.steps[i].1.add(ms);
        }
    }

    /// Whether `decode_rows` rows should ride the next launch of `bucket` rows.
    pub fn ride(&mut self, bucket: usize, decode_rows: usize) -> bool {
        let step = self.step(decode_rows);
        let b = self.bucket(bucket);
        let per_row = b.extra.0.zip(b.riders.0).map(|(e, d)| e / d);
        let prefer = match (b.pure.0, per_row, step) {
            (Some(_), Some(r), Some(s)) => r * decode_rows as f64 <= s,
            // Without a pure sample neither the per-row cost nor its comparison exists: measure it.
            (None, _, _) => false,
            _ => true,
        };
        b.since_explore += 1;
        if b.since_explore >= EXPLORE {
            b.since_explore = 0;
            return !prefer;
        }
        prefer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warm(c: &mut RideCost, pure: f64, per_row: f64, step: f64) {
        c.set_widths(&[32, 64, 128]);
        c.observe_launch(2048, 0, pure);
        c.observe_launch(2048, 50, pure + 50.0 * per_row);
        for rows in [20, 50, 100] {
            c.observe_step(rows, step);
        }
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
        assert!(!c.ride(1024, 10), "no pure sample yet: run the launch without riders");
        warm(&mut c, 45.0, 0.16, 14.0);
        let arms: Vec<bool> = (0..EXPLORE).map(|_| c.ride(2048, 50)).collect();
        assert_eq!(arms.iter().filter(|&&r| !r).count(), 1, "one explored non-ride per window");
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
