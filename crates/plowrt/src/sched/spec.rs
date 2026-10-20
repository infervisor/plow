//! Per-tick speculative-decode plan: whether a pure decode tick drafts, and how deep per row.
//!
//! A spec tick costs one draft launch (`K` steps over the drafting rows) plus one verify launch at
//! the rung covering every row's span; it yields each row's accepted drafts plus one token. The
//! plan maximizes expected tokens per millisecond over per-row depths `k_i` in `{K - 1, K}`
//! (capped per row), so a tick fills the rung it pays for and never jumps one for a single row,
//! and runs only when that beats the plain step at the batch's own rung by [`MARGIN`]. Costs are
//! measured per engine (step ms per rung width, draft ms per step per program rows) and
//! acceptance per draft position; an unmeasured width is extrapolated from the nearest measured
//! one. Every [`EXPLORE`] eligible ticks the best plan runs regardless, so acceptance estimates
//! recover after a stretch below the margin.

pub const MARGIN: f64 = 1.03;
pub const EXPLORE: u32 = 64;
const ALPHA: f64 = 0.1;
const ACCEPT_ALPHA: f64 = 0.02;
const ACCEPT_PRIOR: f64 = 0.7;
/// Step-time growth per doubling of rows, for widths not measured yet.
const WIDTH_GROWTH: f64 = 1.2;
/// Draft ms per step, as a fraction of the narrowest measured step, until measured.
const DRAFT_PRIOR: f64 = 0.12;

/// A tick's speculation: draft depth `k` and the verify rung width it fills.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub k: u32,
    pub width: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct Ewma(Option<f64>);

impl Ewma {
    fn add(&mut self, x: f64, alpha: f64) {
        self.0 = Some(self.0.map_or(x, |m| m + alpha * (x - m)));
    }
}

#[derive(Clone, Debug)]
pub struct SpecPolicy {
    /// Decode rung widths, ascending, and the step ms measured at each.
    widths: Vec<usize>,
    step: Vec<Ewma>,
    /// Widest verify launch the engine can run.
    max_rows: usize,
    kmax: u32,
    /// Draft program row counts, ascending, and the ms per draft step measured at each.
    draft_rows: Vec<usize>,
    draft: Vec<Ewma>,
    /// `accept[j]`: P(draft `j + 1` accepted | drafts `..=j` accepted).
    accept: Vec<f64>,
    since_spec: u32,
}

impl SpecPolicy {
    pub fn new(widths: &[usize], max_rows: usize, kmax: u32, draft_rows: &[usize]) -> Self {
        Self {
            widths: widths.to_vec(),
            step: vec![Ewma::default(); widths.len()],
            max_rows,
            kmax,
            draft_rows: draft_rows.to_vec(),
            draft: vec![Ewma::default(); draft_rows.len()],
            accept: vec![ACCEPT_PRIOR; kmax as usize],
            since_spec: 0,
        }
    }

    fn width(&self, rows: usize) -> Option<usize> {
        self.widths.iter().position(|&w| w >= rows)
    }

    /// One plain decode step or verify launch over `rows` rows took `ms`.
    pub fn observe_step(&mut self, rows: usize, ms: f64) {
        if let Some(i) = self.width(rows) {
            self.step[i].add(ms, ALPHA);
        }
    }

    /// One draft launch of `k` steps over `rows` drafting rows took `ms`.
    pub fn observe_draft(&mut self, rows: usize, k: u32, ms: f64) {
        if let Some(i) = self.draft_rows.iter().position(|&r| r >= rows) {
            self.draft[i].add(ms / f64::from(k.max(1)), ALPHA);
        }
    }

    /// A span of `k` drafts accepted its first `n_acc`.
    pub fn observe_verdict(&mut self, k: u32, n_acc: u32) {
        for j in 0..k.min(n_acc + 1).min(self.kmax) as usize {
            let hit = if (j as u32) < n_acc { 1.0 } else { 0.0 };
            self.accept[j] += ACCEPT_ALPHA * (hit - self.accept[j]);
        }
    }

    /// Expected tokens a row with `k` drafts emits.
    fn expected(&self, k: u32) -> f64 {
        let mut p = 1.0;
        let mut e = 1.0;
        for &a in &self.accept[..k as usize] {
            p *= a;
            e += p;
        }
        e
    }

    fn step_ms(&self, i: usize) -> Option<f64> {
        if let Some(m) = self.step[i].0 {
            return Some(m);
        }
        let (j, m) = self
            .step
            .iter()
            .enumerate()
            .filter_map(|(j, s)| s.0.map(|m| (j, m)))
            .min_by_key(|&(j, _)| j.abs_diff(i))?;
        let doublings = (self.widths[i] as f64 / self.widths[j] as f64).log2();
        Some(m * WIDTH_GROWTH.powf(doublings))
    }

    fn draft_ms(&self, i: usize) -> Option<f64> {
        self.draft[i].0.or_else(|| {
            let narrowest = self.step.iter().find_map(|s| s.0)?;
            Some(DRAFT_PRIOR * narrowest)
        })
    }

    /// Plan one pure decode tick. `caps[i]`: the most drafts row `i` may carry (0 = it decodes
    /// plainly inside the verify). `None` leaves the tick to the plain path; the rows' depths are
    /// [`Self::depths`] of the plan.
    pub fn plan(&mut self, caps: &[u32]) -> Option<Plan> {
        let b = caps.len();
        let plain_ms = self.width(b).and_then(|i| self.step_ms(i))?;
        let base = b as f64 / plain_ms;
        let mut best: Option<(f64, u32, usize)> = None;
        for k in 1..=self.kmax {
            for (wi, &w) in self.widths.iter().enumerate() {
                if w < b || w > self.max_rows {
                    continue;
                }
                let Some(ks) = Self::fill(caps, k, w) else {
                    continue;
                };
                let drafting = ks.iter().filter(|&&x| x > 0).count();
                let deepest = ks.iter().copied().max().unwrap_or(0);
                if drafting == 0 || deepest < k {
                    continue;
                }
                let Some(di) = self.draft_rows.iter().position(|&r| r >= drafting) else {
                    continue;
                };
                let (Some(s), Some(d)) = (self.step_ms(wi), self.draft_ms(di)) else {
                    continue;
                };
                let tokens: f64 = ks.iter().map(|&x| self.expected(x)).sum();
                let gain = tokens / (s + d * f64::from(k)) / base;
                if best.is_none_or(|(g, _, _)| gain > g) {
                    best = Some((gain, k, w));
                }
            }
        }
        let (gain, k, width) = best?;
        self.since_spec += 1;
        if gain < MARGIN && self.since_spec < EXPLORE {
            return None;
        }
        self.since_spec = 0;
        Some(Plan { k, width })
    }

    /// Per-row depths of `plan` for `caps` (which may be tighter than the ones it was planned
    /// for); all zero when the base does not fit.
    pub fn depths(caps: &[u32], plan: Plan, out: &mut Vec<u32>) {
        out.clear();
        match Self::fill(caps, plan.k, plan.width) {
            Some(ks) => out.extend(ks),
            None => out.resize(caps.len(), 0),
        }
    }

    /// Depths `min(cap, k - 1)`, then rows raised to `min(cap, k)` in order while the total
    /// rows stay within `width`. `None` when even the base does not fit.
    fn fill(caps: &[u32], k: u32, width: usize) -> Option<Vec<u32>> {
        let mut ks: Vec<u32> = caps.iter().map(|&c| c.min(k - 1)).collect();
        let mut rows = caps.len() + ks.iter().map(|&x| x as usize).sum::<usize>();
        if rows > width {
            return None;
        }
        for (x, &c) in ks.iter_mut().zip(caps) {
            if rows >= width {
                break;
            }
            if c >= k {
                *x = k;
                rows += 1;
            }
        }
        Some(ks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIDTHS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 128];
    const STEP: [f64; 8] = [4.37, 4.97, 5.85, 6.81, 9.01, 14.61, 20.92, 34.23];

    fn measured(accept: f64) -> SpecPolicy {
        let mut p = SpecPolicy::new(&WIDTHS, 16, 7, &[1, 2, 4, 8, 16]);
        for (&w, &ms) in WIDTHS.iter().zip(&STEP) {
            p.observe_step(w, ms);
        }
        for (r, ms) in [(1, 0.54), (2, 0.56), (4, 0.58), (8, 0.60), (16, 0.70)] {
            p.observe_draft(r, 7, ms * 7.0);
        }
        p.accept = vec![accept; 7];
        p
    }

    #[test]
    fn single_row_drafts_deep_and_fills_its_rung() {
        let mut p = measured(0.78);
        let mut ks = Vec::new();
        let plan = p.plan(&[7]).expect("speculates");
        SpecPolicy::depths(&[7], plan, &mut ks);
        assert!(ks[0] >= 3, "{ks:?}");
        assert_eq!(ks[0] as usize + 1, plan.width, "the span fills its rung");
    }

    #[test]
    fn rows_never_exceed_the_verify_limit_or_caps() {
        let mut p = measured(0.9);
        let mut ks = Vec::new();
        for b in 1..=16usize {
            let caps: Vec<u32> = (0..b).map(|i| [7, 0, 3, 7][i % 4]).collect();
            if let Some(plan) = p.plan(&caps) {
                SpecPolicy::depths(&caps, plan, &mut ks);
                let rows = b + ks.iter().sum::<u32>() as usize;
                assert!(rows <= 16, "B{b}: {rows} rows");
                assert!(ks.iter().zip(&caps).all(|(k, c)| k <= c));
                let deepest = *ks.iter().max().unwrap();
                assert!(ks
                    .iter()
                    .all(|&k| k == 0 || k + 1 >= deepest || caps.contains(&k)));
            }
        }
        assert!(p.plan(&[7; 16]).is_none(), "B16 has no room for a draft");
    }

    #[test]
    fn low_acceptance_stays_plain_until_explore() {
        let mut p = measured(0.05);
        let ran = (0..EXPLORE).filter(|_| p.plan(&[7; 4]).is_some()).count();
        assert_eq!(ran, 1, "one exploration tick per window");
    }

    #[test]
    fn unmeasured_widths_extrapolate() {
        let mut p = SpecPolicy::new(&WIDTHS, 16, 7, &[1, 2, 4, 8, 16]);
        assert!(p.plan(&[7]).is_none(), "nothing measured: plain");
        p.observe_step(1, 4.4);
        assert!(p.plan(&[7]).is_some(), "priors carry the first spec tick");
    }

    #[test]
    fn verdicts_move_acceptance_per_position() {
        let mut p = SpecPolicy::new(&WIDTHS, 16, 3, &[1]);
        for _ in 0..500 {
            p.observe_verdict(3, 1);
        }
        assert!(p.accept[0] > 0.95 && p.accept[1] < 0.05);
        assert!(
            (p.accept[2] - ACCEPT_PRIOR).abs() < 1e-9,
            "position 3 never evaluated"
        );
        assert!(
            (p.expected(3) - (1.0 + p.accept[0] * (1.0 + p.accept[1] * (1.0 + p.accept[2])))).abs()
                < 1e-9
        );
    }
}
