//! §I Admission control — queuing-theory arrival/service estimation + shed.

/// Exponentially-weighted moving-average rate estimator (arrival λ / service μ).
#[derive(Clone, Copy, Debug)]
pub struct Ewma {
    value: f64,
    alpha: f64,
}

impl Ewma {
    pub fn new(alpha: f64) -> Self {
        Ewma { value: 0.0, alpha }
    }

    #[inline]
    pub fn update(&mut self, sample: f64) -> f64 {
        self.value = self.alpha * sample + (1.0 - self.alpha) * self.value;
        self.value
    }

    pub fn get(&self) -> f64 {
        self.value
    }
}

/// Per-slug load estimate: arrival rate λ and per-batch service rate μ(B).
pub struct LoadEstimator {
    pub lambda: Ewma,
    pub service_ms: Ewma,
}

impl Default for LoadEstimator {
    fn default() -> Self {
        LoadEstimator {
            lambda: Ewma::new(0.2),
            service_ms: Ewma::new(0.2),
        }
    }
}

impl LoadEstimator {
    /// Utilization ρ = λ / μ given the current batch's service time.
    pub fn utilization(&self) -> f64 {
        let mu = if self.service_ms.get() > 0.0 {
            1000.0 / self.service_ms.get()
        } else {
            f64::INFINITY
        };
        if mu.is_finite() && mu > 0.0 {
            self.lambda.get() / mu
        } else {
            0.0
        }
    }
}

/// Admission verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admit {
    /// Run this iteration now.
    Now,
    /// Hold briefly to form a larger batch (queuing gain outweighs the wait).
    Defer,
    /// Reject — predicted wait blows the SLO or memory can't seat it.
    Shed,
}

/// Decide admission from utilization, predicted wait, and the SLO.
pub fn admit(util: f64, predicted_wait_ms: f64, slo_ms: f64, mem_ok: bool) -> Admit {
    if !mem_ok || predicted_wait_ms > slo_ms {
        return Admit::Shed;
    }
    // Below saturation there's headroom (and no queue to batch with): run
    // immediately — deferring an isolated request only adds latency. Near
    // saturation (ρ ≥ 0.85), hold briefly so the batch fills; the queuing
    // (throughput) gain outweighs the short wait.
    if util < 0.85 {
        Admit::Now
    } else {
        Admit::Defer
    }
}

/// What the device can actually back, so admission stops at memory instead of at slot count.
///
/// The mux used to admit on free SLOTS alone. On a packet with `max_ctx` 81920 and 20 slots
/// that is fine at 8k prompts (20 x 8192 rows = 8.4 GiB) and fatal at 70k (20 x 70,000 rows =
/// 72.1 GiB against 55.59 GiB free), where the overcommit surfaced as an async queue fault
/// instead of as backpressure. Admitting against bytes makes the long-context case queue.
///
/// `bytes_per_token` is per RANK and counts every KV tensor the packet declares, because a
/// sequence's rows are written on every rank. The pools reserve VA for `max_ctx` and map
/// physical at the frontier, so this prices what an admitted sequence may grow into, which is
/// exactly what admission must not oversubscribe.
pub const MAX_KV_BLOCK_GROUPS: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvBlockGroup {
    pub block_rows: u64,
    pub block_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvBudget {
    pub bytes_per_token: u64,
    pub budget_bytes: u64,
    pub block_groups: [KvBlockGroup; MAX_KV_BLOCK_GROUPS],
    pub block_group_count: u8,
}

impl KvBudget {
    pub fn linear(bytes_per_token: u64, budget_bytes: u64) -> Self {
        Self {
            bytes_per_token,
            budget_bytes,
            block_groups: [KvBlockGroup::default(); MAX_KV_BLOCK_GROUPS],
            block_group_count: 0,
        }
    }

    pub fn with_block_groups(mut self, groups: &[(u64, u64)]) -> Option<Self> {
        if groups.is_empty() || groups.len() > MAX_KV_BLOCK_GROUPS {
            return None;
        }
        for (dst, &(block_rows, block_bytes)) in self.block_groups.iter_mut().zip(groups.iter()) {
            if block_rows == 0 || block_bytes == 0 {
                return None;
            }
            *dst = KvBlockGroup {
                block_rows,
                block_bytes,
            };
        }
        self.block_group_count = groups.len() as u8;
        Some(self)
    }

    #[inline]
    pub fn bytes_for_rows(&self, rows: u64) -> u64 {
        if self.block_group_count == 0 {
            return rows.saturating_mul(self.bytes_per_token);
        }
        self.block_groups[..self.block_group_count as usize]
            .iter()
            .fold(0u64, |total, group| {
                // Row zero of every slot/group is mapped before free memory is sampled.
                let extra_blocks = rows
                    .saturating_add(group.block_rows - 1)
                    .checked_div(group.block_rows)
                    .unwrap_or(u64::MAX)
                    .saturating_sub(1);
                total.saturating_add(extra_blocks.saturating_mul(group.block_bytes))
            })
    }

    pub fn fits_requests(&self, rows: impl IntoIterator<Item = u64>) -> bool {
        rows.into_iter()
            .try_fold(0u64, |total, rows| {
                total.checked_add(self.bytes_for_rows(rows))
            })
            .is_some_and(|bytes| bytes <= self.budget_bytes)
    }

    /// Whether `committed_rows` already-promised rows plus `want_rows` more still fit.
    ///
    /// Saturating throughout: a budget this large only overflows if the geometry is nonsense,
    /// and refusing every request would be a worse failure than admitting one.
    pub fn fits(&self, committed_rows: u64, want_rows: u64) -> bool {
        if self.block_group_count != 0 {
            return self.fits_requests([committed_rows, want_rows]);
        }
        committed_rows
            .saturating_add(want_rows)
            .saturating_mul(self.bytes_per_token)
            <= self.budget_bytes
    }

    /// Rows the budget can back in total. For logs and for the "cannot ever fit" case.
    pub fn max_rows(&self) -> u64 {
        if self.block_group_count != 0 {
            let mut lo = 0u64;
            let mut hi = 1u64;
            while self.bytes_for_rows(hi) <= self.budget_bytes && hi < u64::MAX / 2 {
                hi *= 2;
            }
            while lo + 1 < hi {
                let mid = lo + (hi - lo) / 2;
                if self.bytes_for_rows(mid) <= self.budget_bytes {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return lo;
        }
        self.budget_bytes / self.bytes_per_token.max(1)
    }
}

#[cfg(test)]
mod kv_budget_tests {
    use super::KvBudget;

    /// The measured 70k/C20 cell: the arm that faulted must not be admissible, and the arms
    /// that ran clean must still be. 55,296 B/token/rank, 55.59 GiB free, 0.9 headroom.
    #[test]
    fn the_faulting_cell_is_refused_and_the_clean_ones_are_not() {
        let free = (55.58984375f64 * (1u64 << 30) as f64) as u64;
        let b = KvBudget::linear(55_296, (free as f64 * 0.9) as u64);

        // 20 x 8192 = the shape every clean run used.
        assert!(b.fits(0, 20 * 8192));
        // 1 and 4 sequences at 70k: arms A and B, both clean on hardware.
        assert!(b.fits(0, 70_000));
        assert!(b.fits(3 * 70_000, 70_000));
        // 20 x 70,000: arms C and D, both faulted on hardware.
        assert!(!b.fits(19 * 70_000, 70_000));
        // The knee sits where the arithmetic says: ~13 at 0.9 headroom of 55.59 GiB.
        let n = (1..=20).take_while(|i| b.fits((i - 1) as u64 * 70_000, 70_000)).count();
        assert_eq!(n, 13, "max concurrent 70k sequences");
    }

    /// A row count that would wrap `rows * bytes_per_token` must refuse, not wrap to a small
    /// product and admit. Saturation makes the overflowing case the most-refused one.
    #[test]
    fn saturates_instead_of_overflowing() {
        let b = KvBudget::linear(55_296, 8 << 30);
        assert!(!b.fits(u64::MAX, 1));
        assert!(!b.fits(1, u64::MAX));
        assert!(!b.fits(u64::MAX / 2, u64::MAX / 2));
        // An unreadable geometry divides by zero rather than panicking.
        assert_eq!(KvBudget::linear(0, 8).max_rows(), 8);
    }

    #[test]
    fn block_rounded_glm_budget_admits_fourteen_70k_sequences() {
        let free = (55.58984375f64 * (1u64 << 30) as f64) as u64;
        let b = KvBudget::linear(55_608, (free as f64 * 0.9) as u64)
            .with_block_groups(&[
                (16_384, 78 * (2 << 20)),
                (8_192, 21 * (2 << 20)),
                (4_096, 78 * (2 << 20)),
            ])
            .unwrap();

        assert_eq!(
            b.bytes_for_rows(1),
            0,
            "the prepaid first column is not charged twice"
        );
        assert!(b.fits_requests(std::iter::repeat_n(70_700, 14)));
        assert!(!b.fits_requests(std::iter::repeat_n(70_700, 15)));
    }

    #[test]
    fn block_rounding_is_per_sequence() {
        let b = KvBudget::linear(1, 2 << 20)
            .with_block_groups(&[(4_096, 2 << 20)])
            .unwrap();
        assert!(b.fits_requests([4_096, 4_096]));
        assert!(!b.fits_requests([4_097, 4_097]));
    }
}
