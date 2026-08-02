//! Resource timelines — the data structure the user asked to "initiate".
//!
//! Two primitives (design §5):
//! * [`IntervalSet`] — *exclusive* reservations (SM, DMA engine, DPU, host
//!   thread): the borrow-checker `&mut` rule, no two holds overlap.
//! * [`BandwidthSet`] — *capacity* reservations (HBM, interconnect): weighted
//!   holds whose peak concurrent sum must stay under a limit (§5.4).
//!
//! Both are backed by sorted `Vec`s rather than an augmented BST: per-resource
//! reservation counts are modest (tasks per SM), so O(n) insert/query is simpler
//! and fast enough, while the API (`overlaps` / `earliest_free` / `peak` /
//! `reserve`) is exactly what an interval tree would expose — so it can be
//! swapped for one later if profiling demands.

/// Scheduler time base (cost-model cycles).
pub type Cycle = u64;

/// Exclusive reservations on one resource, kept sorted by start and
/// non-overlapping.
#[derive(Clone, Debug, Default)]
pub struct IntervalSet {
    res: Vec<(Cycle, Cycle)>, // [start, end)
}

impl IntervalSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Does any reservation overlap `[start, end)`?
    pub fn overlaps(&self, start: Cycle, end: Cycle) -> bool {
        self.res.iter().any(|&(s, e)| start < e && s < end)
    }

    /// Earliest cycle ≥ `after` at which a `dur`-long interval fits with no
    /// overlap. Single forward pass (reservations are sorted + disjoint): push
    /// the candidate start past every reservation that overlaps its window.
    pub fn earliest_free(&self, after: Cycle, dur: Cycle) -> Cycle {
        if dur == 0 {
            return after;
        }
        let mut t = after;
        // Reservations are disjoint and start-sorted, so ends are monotone too:
        // binary-search past everything already behind the candidate window
        // instead of walking the whole timeline from cycle 0.
        let first = self.res.partition_point(|&(_, e)| e <= after);
        for &(s, e) in &self.res[first..] {
            if s >= t + dur {
                break; // the gap [t, t+dur) fits before this reservation
            }
            t = t.max(e); // overlaps → slide past it
        }
        t
    }

    /// Reserve `[start, start+dur)`. Caller ensures it is free (use
    /// [`IntervalSet::earliest_free`]).
    pub fn reserve(&mut self, start: Cycle, dur: Cycle) {
        let iv = (start, start + dur);
        let pos = self.res.partition_point(|&(s, _)| s < iv.0);
        self.res.insert(pos, iv);
    }

    pub fn reservations(&self) -> &[(Cycle, Cycle)] {
        &self.res
    }
}

/// Weighted capacity reservations on one resource (bandwidth).
///
/// Stored as a **stepwise aggregate-load profile**, not a reservation list:
/// `levels[i] = (t, load)` means the summed load is `load` on
/// `[t, levels[i+1].0)`; the final level extends to ∞ and is 0 once every
/// reservation has ended. Always non-empty, anchored at `(0, 0.0)`.
///
/// The old flat `Vec<(start, end, w)>` made every `peak` query rescan and
/// re-sort every reservation ever made — O(R log R) per probe with R growing
/// per placed task, i.e. O(T²·log) over a whole schedule. The profile answers
/// `peak` with a binary search plus a walk over only the pieces inside the
/// window, and can answer "earliest feasible start" directly
/// ([`BandwidthSet::next_feasible`]), which replaces the scheduler's blind
/// probe ladders.
#[derive(Clone, Debug)]
pub struct BandwidthSet {
    levels: Vec<(Cycle, f64)>,
}

impl Default for BandwidthSet {
    fn default() -> Self {
        Self {
            levels: vec![(0, 0.0)],
        }
    }
}

impl BandwidthSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index of the piece whose span contains `t`.
    fn piece_at(&self, t: Cycle) -> usize {
        // levels[0].0 == 0 and t >= 0, so the result is always >= 1.
        self.levels.partition_point(|&(s, _)| s <= t) - 1
    }

    /// Peak concurrent weight over `[start, end)`. O(log N + pieces in window).
    pub fn peak(&self, start: Cycle, end: Cycle) -> f64 {
        if start >= end {
            return 0.0;
        }
        let mut peak = 0.0f64;
        let mut i = self.piece_at(start);
        while i < self.levels.len() && self.levels[i].0 < end {
            peak = peak.max(self.levels[i].1);
            i += 1;
        }
        peak
    }

    /// Would adding weight `w` over `[start, end)` keep the peak ≤ `limit`?
    /// (The new hold is constant `w` over the window, so the combined peak is
    /// `peak(existing) + w`.)
    pub fn capacity_ok(&self, start: Cycle, end: Cycle, w: f64, limit: f64) -> bool {
        self.peak(start, end) + w <= limit + 1e-9
    }

    /// Earliest `t ≥ after` such that adding `w` over `[t, t+dur)` stays under
    /// `limit`. The profile's tail level is 0 (every reservation ends), so a
    /// feasible start always exists when `w ≤ limit` — callers pre-check the
    /// `w > limit` model-mismatch case.
    pub fn next_feasible(&self, after: Cycle, dur: Cycle, w: f64, limit: f64) -> Cycle {
        let dur = dur.max(1);
        let mut t = after;
        let mut i = self.piece_at(t);
        loop {
            // Scan the pieces overlapping [t, t+dur) for the first violation.
            let mut k = i;
            let mut violation = None;
            while k < self.levels.len() && self.levels[k].0 < t.saturating_add(dur) {
                if self.levels[k].1 + w > limit + 1e-9 {
                    violation = Some(k);
                    break;
                }
                k += 1;
            }
            let Some(k) = violation else { return t };
            if k + 1 >= self.levels.len() {
                // Defensive: a violating final piece can only mean w > limit
                // (tail load is 0); the caller handles that case before us.
                return t;
            }
            // The window cannot start before the violating piece ends.
            t = self.levels[k + 1].0;
            i = k + 1;
        }
    }

    /// Ensure a breakpoint exists at `t`; return its index.
    fn ensure_breakpoint(&mut self, t: Cycle) -> usize {
        match self.levels.binary_search_by_key(&t, |&(s, _)| s) {
            Ok(i) => i,
            Err(i) => {
                // Split the containing piece; the new point inherits its load.
                let level = self.levels[i - 1].1;
                self.levels.insert(i, (t, level));
                i
            }
        }
    }

    pub fn reserve(&mut self, start: Cycle, end: Cycle, w: f64) {
        if start >= end {
            return;
        }
        let i = self.ensure_breakpoint(start);
        let j = self.ensure_breakpoint(end); // j > i: end > start
        for lvl in &mut self.levels[i..j] {
            lvl.1 += w;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exclusive_overlap_and_free() {
        let mut s = IntervalSet::new();
        s.reserve(0, 10); // [0,10)
        s.reserve(20, 5); // [20,25)
        assert!(s.overlaps(5, 15));
        assert!(!s.overlaps(10, 20)); // touching, not overlapping
                                      // A 5-long task after cycle 0 must wait until 10 (first reservation ends).
        assert_eq!(s.earliest_free(0, 5), 10);
        // A 5-long task fits exactly in the gap [10,20)? earliest_free(10,5)=10.
        assert_eq!(s.earliest_free(10, 5), 10);
        // A 12-long task can't fit in [10,20), must wait past [20,25) → 25.
        assert_eq!(s.earliest_free(10, 12), 25);
    }

    #[test]
    fn earliest_free_empty_and_zero() {
        let s = IntervalSet::new();
        assert_eq!(s.earliest_free(7, 3), 7);
        assert_eq!(s.earliest_free(7, 0), 7);
    }

    #[test]
    fn bandwidth_capacity() {
        let mut b = BandwidthSet::new();
        b.reserve(0, 10, 0.6); // 60% of the link over [0,10)
                               // A second 0.6 transfer overlapping would peak at 1.2 > 1.0 → rejected.
        assert!(!b.capacity_ok(5, 15, 0.6, 1.0));
        // A 0.3 transfer overlapping fits (0.6 + 0.3 ≤ 1.0).
        assert!(b.capacity_ok(5, 15, 0.3, 1.0));
        // A transfer in the disjoint window [10,20) sees no concurrency.
        assert!(b.capacity_ok(10, 20, 0.9, 1.0));
        assert_eq!(b.peak(0, 10), 0.6);
    }
}
