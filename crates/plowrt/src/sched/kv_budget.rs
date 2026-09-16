//! §I KV seating — how many sequences of a given length this rank can hold, and the `mem_ok`
//! predicate [`super::admission::admit`] takes.
//!
//! Under decode context parallelism the per-token KV figure stops meaning "bytes to hold one
//! token of this sequence" and starts meaning "bytes THIS DEVICE holds for one token of this
//! sequence". The conversion is NOT a division by the DCP degree: ownership is quantised to
//! pages, so the binding constraint is the worst shard, which exceeds the mean by up to one
//! page per sequence. Seating on the mean over-admits and the worst-ranked device OOMs while
//! the budget still reports headroom.
//!
//! The replicated layout (`degree == 1`) makes every figure here identical to the pre-DCP
//! arithmetic, so a caller threads a [`DcpLayout`] through unconditionally.

use packet::dcp::DcpLayout;

/// The KV seating budget for one rank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvBudget {
    /// Device bytes this rank may spend on KV rows.
    pub budget_bytes: u64,
    /// KV bytes one token of one sequence costs with the cache REPLICATED — the figure
    /// recovered from the blob as `sum(bytes of "kv.*") / (max_ctx * batch)`. It is the
    /// replicated figure even under DCP, because the emitter declares a smaller extent and the
    /// row layout, not the row width, is what shards.
    pub per_token_replicated: u64,
    /// How KV rows are spread over this rank's DCP group.
    pub layout: DcpLayout,
}

impl KvBudget {
    pub const fn new(budget_bytes: u64, per_token_replicated: u64, layout: DcpLayout) -> Self {
        KvBudget {
            budget_bytes,
            per_token_replicated,
            layout,
        }
    }

    /// Bytes this rank must hold to seat one `len`-token sequence.
    ///
    /// `None` on overflow, which is a refusal, not a zero.
    pub fn seat_bytes(&self, len: u64) -> Option<u64> {
        self.layout.local_kv_bytes(self.per_token_replicated, len)
    }

    /// Sequences of `len` tokens this rank can seat from an empty budget.
    pub fn seats(&self, len: u64) -> u64 {
        self.layout
            .seats(self.budget_bytes, self.per_token_replicated, len)
    }

    /// Bytes left after seating `seated`, saturating at zero.
    pub fn headroom(&self, seated: impl IntoIterator<Item = u64>) -> u64 {
        self.budget_bytes.saturating_sub(self.used(seated))
    }

    /// Bytes the given sequence lengths occupy on this rank. Saturates rather than wrapping: a
    /// saturated total is over budget, which is the correct verdict.
    pub fn used(&self, seated: impl IntoIterator<Item = u64>) -> u64 {
        seated.into_iter().fold(0u64, |acc, len| {
            acc.saturating_add(self.seat_bytes(len).unwrap_or(u64::MAX))
        })
    }

    /// The `mem_ok` argument of [`super::admission::admit`]: can this rank seat `incoming`
    /// alongside what it already holds?
    ///
    /// `seated` is the CURRENT length of every live sequence and `incoming` the length the new
    /// one will reach — its max, not its prompt, because a sequence that no longer fits halfway
    /// through generation has to be preempted, and preemption at long context costs more than
    /// the admission it saves.
    pub fn fits(&self, seated: impl IntoIterator<Item = u64>, incoming: u64) -> bool {
        let Some(want) = self.seat_bytes(incoming) else {
            return false;
        };
        self.used(seated)
            .checked_add(want)
            .is_some_and(|total| total <= self.budget_bytes)
    }

    /// Amortised per-token bytes on this rank, for a cost model that wants a rate rather than a
    /// seat. Equals `per_token_replicated` at degree 1 and approaches `/degree` as `len` grows
    /// past one page per shard.
    pub fn bytes_per_token(&self, len: u64) -> f64 {
        if len == 0 {
            return 0.0;
        }
        self.seat_bytes(len).unwrap_or(u64::MAX) as f64 / len as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The measured GLM-5.3 TP8 figure.
    const PER_TOKEN: u64 = 55_608;
    const BUDGET: u64 = 50 * (1 << 30);

    fn rep() -> KvBudget {
        KvBudget::new(BUDGET, PER_TOKEN, DcpLayout::replicated(8))
    }
    fn dcp() -> KvBudget {
        KvBudget::new(BUDGET, PER_TOKEN, DcpLayout::new(8, 8, 64))
    }

    #[test]
    fn the_replicated_budget_is_the_pre_dcp_arithmetic() {
        let b = rep();
        for len in [1u64, 1000, 8192, 70_016, 81_920] {
            assert_eq!(b.seat_bytes(len), Some(PER_TOKEN * len));
            assert_eq!(b.seats(len), BUDGET / (PER_TOKEN * len));
            assert_eq!(b.bytes_per_token(len), PER_TOKEN as f64);
        }
    }

    #[test]
    fn eight_way_sharding_buys_about_eight_times_the_seats() {
        let (r, d) = (rep(), dcp());
        for len in [8192u64, 32_768, 70_016] {
            let (sr, sd) = (r.seats(len), d.seats(len));
            assert!(sr > 0, "len {len}");
            assert!(sd >= 8 * sr, "len {len}: {sd} vs 8 x {sr}");
            // and no more than eight times plus the page quantisation slack
            assert!(sd <= 9 * sr + 8, "len {len}: {sd}");
        }
    }

    #[test]
    fn seating_uses_the_worst_shard_not_the_mean() {
        let d = dcp();
        // 65 tokens: shard 0 holds a whole 64-row page, shard 1 holds 1 row, the rest nothing.
        assert_eq!(d.seat_bytes(65), Some(PER_TOKEN * 64));
        // The mean would have said 65/8 = 8 rows and over-admitted shard 0 eightfold.
        assert!(d.seat_bytes(65).unwrap() > PER_TOKEN * (65 / 8));
        // Short sequences therefore get NO sharding benefit at all, which is correct: they have
        // fewer pages than shards.
        assert_eq!(d.seat_bytes(64), rep().seat_bytes(64));
    }

    #[test]
    fn fits_refuses_when_the_worst_shard_would_overflow() {
        let len = 70_016u64;
        let d = dcp();
        let per_seat = d.seat_bytes(len).unwrap();
        let seats = d.seats(len);
        let full: Vec<u64> = std::iter::repeat_n(len, seats as usize).collect();
        assert!(d.used(full.iter().copied()) + per_seat > BUDGET);
        assert!(!d.fits(full.iter().copied(), len), "over-admitted");
        let one_short: Vec<u64> = full[1..].to_vec();
        assert!(d.fits(one_short.iter().copied(), len));
    }

    #[test]
    fn an_empty_budget_seats_nothing_and_a_zero_length_costs_nothing() {
        let d = dcp();
        assert_eq!(d.seat_bytes(0), Some(0));
        assert!(d.fits([], 0));
        assert_eq!(KvBudget::new(0, PER_TOKEN, DcpLayout::new(8, 8, 64)).seats(8192), 0);
        assert_eq!(d.bytes_per_token(0), 0.0);
    }

    #[test]
    fn overflow_is_a_refusal_not_a_wrap() {
        let d = KvBudget::new(BUDGET, u64::MAX / 2, DcpLayout::new(8, 8, 64));
        assert_eq!(d.seat_bytes(u64::MAX / 2), None);
        assert!(!d.fits([], u64::MAX / 2));
        assert_eq!(d.seats(u64::MAX / 2), 0);
    }

    #[test]
    fn the_amortised_rate_approaches_one_eighth() {
        let d = dcp();
        let rate = d.bytes_per_token(70_016);
        let ideal = PER_TOKEN as f64 / 8.0;
        assert!(rate >= ideal, "{rate} < {ideal}");
        assert!(rate <= ideal * 1.01, "{rate} vs {ideal}");
    }

    #[test]
    fn headroom_tracks_used() {
        let d = dcp();
        let live = [8192u64, 16_384, 70_016];
        assert_eq!(d.headroom(live), BUDGET - d.used(live));
        assert_eq!(d.headroom(std::iter::repeat_n(u64::MAX, 2)), 0);
    }
}
