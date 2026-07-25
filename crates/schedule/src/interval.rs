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
        for &(s, e) in &self.res {
            if e <= t {
                continue; // already behind the candidate window
            }
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
#[derive(Clone, Debug, Default)]
pub struct BandwidthSet {
    res: Vec<(Cycle, Cycle, f64)>, // [start, end), weight
}

impl BandwidthSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Peak concurrent weight over `[start, end)`. Event sweep: clamp every
    /// overlapping reservation to the window, emit (+w at on, −w at off)
    /// events, sort, and walk once accumulating — O(R log R) per call.
    pub fn peak(&self, start: Cycle, end: Cycle) -> f64 {
        let mut events: Vec<(Cycle, f64)> = Vec::new();
        for &(s, e, w) in &self.res {
            if s < end && e > start {
                events.push((s.max(start), w));
                events.push((e.min(end), -w));
            }
        }
        if events.is_empty() {
            return 0.0;
        }
        // At equal times, apply ends (−w) before starts (+w): intervals are
        // half-open, so a hold ending at t does not overlap one starting at t.
        events.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut cur = 0.0f64;
        let mut peak = 0.0f64;
        let mut i = 0;
        while i < events.len() {
            let t = events[i].0;
            while i < events.len() && events[i].0 == t {
                cur += events[i].1;
                i += 1;
            }
            if t < end {
                peak = peak.max(cur);
            }
        }
        peak
    }

    /// Would adding weight `w` over `[start, end)` keep the peak ≤ `limit`?
    /// (The new hold is constant `w` over the window, so the combined peak is
    /// `peak(existing) + w`.)
    pub fn capacity_ok(&self, start: Cycle, end: Cycle, w: f64, limit: f64) -> bool {
        self.peak(start, end) + w <= limit + 1e-9
    }

    pub fn reserve(&mut self, start: Cycle, end: Cycle, w: f64) {
        self.res.push((start, end, w));
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
