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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvBudget {
    pub bytes_per_token: u64,
    pub budget_bytes: u64,
}

impl KvBudget {
    /// Whether `committed_rows` already-promised rows plus `want_rows` more still fit.
    ///
    /// Saturating throughout: a budget this large only overflows if the geometry is nonsense,
    /// and refusing every request would be a worse failure than admitting one.
    pub fn fits(&self, committed_rows: u64, want_rows: u64) -> bool {
        committed_rows
            .saturating_add(want_rows)
            .saturating_mul(self.bytes_per_token)
            <= self.budget_bytes
    }

    /// Rows the budget can back in total. For logs and for the "cannot ever fit" case.
    pub fn max_rows(&self) -> u64 {
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
        let b = KvBudget { bytes_per_token: 55_296, budget_bytes: (free as f64 * 0.9) as u64 };

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
        let b = KvBudget { bytes_per_token: 55_296, budget_bytes: 8 << 30 };
        assert!(!b.fits(u64::MAX, 1));
        assert!(!b.fits(1, u64::MAX));
        assert!(!b.fits(u64::MAX / 2, u64::MAX / 2));
        // An unreadable geometry divides by zero rather than panicking.
        assert_eq!(KvBudget { bytes_per_token: 0, budget_bytes: 8 }.max_rows(), 8);
    }
}
