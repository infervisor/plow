//! §I Admission control — queuing-theory arrival/service estimation + seating.
//!
//! There is no SLO-driven shed, and that is deliberate.
//!
//! `Admit::Shed` used to sit on the tick path, and the only thing it did was take every live
//! slot and answer it 429. That is not backpressure: a live slot has already paid for its
//! prefill, so dropping it burns the most expensive work in the system and hands the client a
//! retry that arrives right back. Its three recorded regressions were each patched by making it
//! fire less, and by the end it could not fire at all — `predicted_wait` was
//! `ceil(live / rung_width) * service_ms`, the covering rung is never narrower than the live
//! extent, so the ratio was always 1 against an SLO floored at 8 service ticks.
//!
//! Backpressure lives where a request has not started yet and refusing it is free:
//!
//! * memory — [`seat`] returns [`Denied::KvBudgetFull`] and the mux keeps the request queued;
//! * latency — [`crate::sched::rungs::RungController`] consumes `slo_ms` and `oldest_wait_ms`
//!   and moves the admission window ([`crate::sched::rungs::RungReason::Slo`]);
//! * staleness — the mux sheds a QUEUED request whose wait passed its TTL, which is the one
//!   place shedding costs no completed work.

use std::time::Instant;

/// Time constant of the arrival-rate estimator. Matched to the rung controller's narrowing
/// window (`NARROW_TICKS` x `MIN_DWELL_TICKS` at a ~40 ms decode tick is a few seconds), so λ
/// has fallen most of the way to zero by the time narrowing is eligible.
const ARRIVAL_TAU_S: f64 = 2.0;

/// Exponentially-weighted moving-average estimator over VALUE samples (service time).
///
/// Sample-indexed, not time-indexed: it is only correct for a quantity that is observed once per
/// event and whose staleness does not matter. A RATE needs [`ArrivalRate`].
#[derive(Clone, Copy, Debug)]
pub struct Ewma {
    value: f64,
    alpha: f64,
}

impl Ewma {
    pub fn new(alpha: f64) -> Self {
        Ewma { value: 0.0, alpha }
    }

    /// The first sample seeds the average. Starting from zero reads low by `(1 - alpha)^n` over
    /// the first n samples (59% of the truth after four at alpha 0.2), and the rung controller
    /// compares rungs whose sample counts differ by orders of magnitude: a rung with a handful
    /// of samples looked 1.7x faster than it was and won the throughput seat from the widest.
    #[inline]
    pub fn update(&mut self, sample: f64) -> f64 {
        self.value = if self.value == 0.0 {
            sample
        } else {
            self.alpha * sample + (1.0 - self.alpha) * self.value
        };
        self.value
    }

    pub fn get(&self) -> f64 {
        self.value
    }
}

/// Arrival intensity λ, in requests per second, decaying toward zero with elapsed time.
///
/// An [`Ewma`] over `1/dt` samples was wrong twice over. It never decayed — updated only on an
/// arrival, λ held the burst rate for as long as the model then sat idle, which is exactly when
/// the rung controller has to see load fall, so [`crate::sched::rungs::RungController`] widened
/// and never narrowed. And averaging reciprocals is biased high by Jensen's inequality
/// (E[1/dt] ≥ 1/E[dt]), so even a live stream read hot.
///
/// This is the exponential intensity estimator instead: every arrival deposits `1/τ` and the
/// deposit decays as `exp(-Δt/τ)`. For a Poisson stream of rate `r` its expectation is exactly
/// `r` (∫ r·(1/τ)·e^(-s/τ) ds = r) — no reciprocal is ever averaged — and with no arrivals it
/// falls to zero on the τ timescale.
#[derive(Clone, Copy, Debug)]
pub struct ArrivalRate {
    /// Intensity as of `last`. Undecayed; every reader decays it forward.
    value: f64,
    tau_s: f64,
    last: Option<Instant>,
}

impl ArrivalRate {
    pub fn new(tau_s: f64) -> Self {
        ArrivalRate {
            value: 0.0,
            tau_s,
            last: None,
        }
    }

    /// λ as of `now`. Pure: decaying on read costs one `exp` and keeps every reader honest
    /// without needing `&mut` on the tick path.
    #[inline]
    pub fn rate(&self, now: Instant) -> f64 {
        let Some(last) = self.last else {
            return 0.0;
        };
        let dt = now.saturating_duration_since(last).as_secs_f64();
        self.value * (-dt / self.tau_s).exp()
    }

    /// Time since the previous arrival, if there was one.
    pub fn since_last(&self, now: Instant) -> Option<std::time::Duration> {
        self.last.map(|last| now.saturating_duration_since(last))
    }

    /// Record one arrival at `now`.
    #[inline]
    pub fn observe(&mut self, now: Instant) -> f64 {
        self.value = self.rate(now) + 1.0 / self.tau_s;
        self.last = Some(now);
        self.value
    }
}

/// Per-slug load estimate: arrival rate λ and per-batch service rate μ(B).
pub struct LoadEstimator {
    pub lambda: ArrivalRate,
    pub service_ms: Ewma,
}

impl Default for LoadEstimator {
    fn default() -> Self {
        LoadEstimator {
            lambda: ArrivalRate::new(ARRIVAL_TAU_S),
            service_ms: Ewma::new(0.2),
        }
    }
}

impl LoadEstimator {
    /// Utilization ρ = λ / μ given the current batch's service time.
    ///
    /// Reported as the `plowrt_utilization` gauge. Nothing schedules on it: the rung controller
    /// computes its own per-rung utilization, which is the one that moves the admission window.
    pub fn utilization(&self, now: Instant) -> f64 {
        let mu = if self.service_ms.get() > 0.0 {
            1000.0 / self.service_ms.get()
        } else {
            f64::INFINITY
        };
        if mu.is_finite() && mu > 0.0 {
            self.lambda.rate(now) / mu
        } else {
            0.0
        }
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
/// Why a request did not take a slot.
///
/// The whole seat decision lives in [`seat`], so this enum is the complete list of ways a request
/// can fail to be admitted. Only [`Denied::KvBudgetFull`] is retryable: the caller keeps that
/// request queued and tries again once a slot retires. Every other variant is terminal and the
/// caller answers the stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Denied {
    /// The engine is poisoned; admitting would dispatch into a dead context.
    EngineDead,
    /// Arrival met a full slot table inside the admission window.
    NoFreeSlot,
    /// No amount of retirement frees enough — deferring would wedge the queue behind it.
    ExceedsWholeBudget { want: u64, max_rows: u64 },
    /// Live sequences hold the KV today. Retryable.
    KvBudgetFull { want: u64 },
}

impl Denied {
    /// Whether the caller should keep the request queued rather than answer its stream.
    pub fn is_retryable(self) -> bool {
        matches!(self, Denied::KvBudgetFull { .. })
    }
}

/// KV rows a sequence still has promised: its prompt plus the generation it has not produced yet.
///
/// The charge only ever shrinks, so subtracting `generated_tokens` is safe without preemption.
pub fn reserved_kv_rows(prompt_tokens: usize, max_tokens: usize, generated_tokens: usize) -> u64 {
    prompt_tokens
        .saturating_add(max_tokens.max(1))
        .saturating_sub(generated_tokens) as u64
}

/// THE seat decision, in one place and in this order: liveness, then a slot, then the whole-device
/// bound, then what live sequences already hold.
///
/// Pure — no metrics, no streams, no allocation — so the order of the checks reads in one screen
/// and is testable without an engine. The caller supplies `free_slot` (the first idle index inside
/// the decode-rung admission window) and `committed_rows` over EVERY live slot, including those
/// above that window, because their KV is just as resident.
pub fn seat(
    engine_dead: bool,
    free_slot: Option<usize>,
    want_rows: u64,
    committed_rows: impl IntoIterator<Item = u64>,
    budget: Option<KvBudget>,
) -> Result<usize, Denied> {
    if engine_dead {
        return Err(Denied::EngineDead);
    }
    let slot = free_slot.ok_or(Denied::NoFreeSlot)?;
    if let Some(budget) = budget {
        if !budget.fits_requests([want_rows]) {
            return Err(Denied::ExceedsWholeBudget {
                want: want_rows,
                max_rows: budget.max_rows(),
            });
        }
        if !budget.fits_requests(committed_rows.into_iter().chain(std::iter::once(want_rows))) {
            return Err(Denied::KvBudgetFull { want: want_rows });
        }
    }
    Ok(slot)
}

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
                // Charge every block the sequence will physically map. There is no per-sequence
                // prepaid block to credit back: `SharedPrefix::new` reserves VA only and leaves
                // every frontier at 0, and `free` is sampled after it, so the budget already
                // reflects zero resident KV. The one genuinely prepaid block per group
                // (`blocks_kept`) is device-wide and only exists after a sequence retires;
                // crediting it per sequence handed back ~354 MiB each and moved the 70k knee
                // from 12 to 14.
                let blocks = rows
                    .saturating_add(group.block_rows - 1)
                    .checked_div(group.block_rows)
                    .unwrap_or(u64::MAX);
                total.saturating_add(blocks.saturating_mul(group.block_bytes))
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
mod seat_tests {
    use super::{seat, Denied, KvBudget};

    fn glm_budget() -> KvBudget {
        let free = (55.58984375f64 * (1u64 << 30) as f64) as u64;
        KvBudget::linear(55_608, (free as f64 * 0.9) as u64)
            .with_block_groups(&[
                (16_384, 78 * (2 << 20)),
                (8_192, 21 * (2 << 20)),
                (4_096, 78 * (2 << 20)),
            ])
            .unwrap()
    }

    /// The order of the checks is the contract: a dead engine is reported as dead even when the
    /// slot table is also full, so the caller answers with the fault that killed the engine
    /// rather than a misleading 429.
    #[test]
    fn liveness_is_decided_before_capacity() {
        assert_eq!(seat(true, None, 1, [], None), Err(Denied::EngineDead));
        assert_eq!(seat(false, None, 1, [], None), Err(Denied::NoFreeSlot));
        assert_eq!(seat(false, Some(3), 1, [], None), Ok(3));
    }

    /// A request larger than the whole device is terminal, not retryable: deferring it would
    /// wedge the queue behind a request no retirement can ever seat.
    #[test]
    fn a_request_over_the_whole_budget_is_terminal_not_queued() {
        let b = glm_budget();
        let max = b.max_rows();
        let err = seat(false, Some(0), max + 1, [], Some(b)).unwrap_err();
        assert!(matches!(err, Denied::ExceedsWholeBudget { .. }));
        assert!(!err.is_retryable());
    }

    /// Pressure from live sequences is retryable, and is charged over every live slot — including
    /// slots above the admission window, whose KV is just as resident.
    #[test]
    fn live_sequences_push_a_fitting_request_back_onto_the_queue() {
        let b = glm_budget();
        assert_eq!(seat(false, Some(0), 70_700, [], Some(b)), Ok(0));
        let err = seat(false, Some(0), 70_700, [70_700; 12], Some(b)).unwrap_err();
        assert_eq!(err, Denied::KvBudgetFull { want: 70_700 });
        assert!(err.is_retryable());
        // 11 already seated leaves room for the twelfth.
        assert_eq!(seat(false, Some(0), 70_700, [70_700; 11], Some(b)), Ok(0));
    }

    /// Without a budget the seat decision is slots alone — the pre-KV-admission behaviour.
    #[test]
    fn no_budget_means_slots_alone() {
        assert_eq!(seat(false, Some(0), u64::MAX, [u64::MAX; 4], None), Ok(0));
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
        let n = (1..=20)
            .take_while(|i| b.fits((i - 1) as u64 * 70_000, 70_000))
            .count();
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

    /// 12, not 14: `bytes_for_rows` charges every block the sequence maps. The earlier `-1`
    /// credited a prepaid row-zero block per sequence per group — 354 MiB each — that nothing
    /// funds, because `SharedPrefix::new` reserves VA only and `free` is sampled after it.
    #[test]
    fn block_rounded_glm_budget_admits_twelve_70k_sequences() {
        let free = (55.58984375f64 * (1u64 << 30) as f64) as u64;
        let b = KvBudget::linear(55_608, (free as f64 * 0.9) as u64)
            .with_block_groups(&[
                (16_384, 78 * (2 << 20)),
                (8_192, 21 * (2 << 20)),
                (4_096, 78 * (2 << 20)),
            ])
            .unwrap();

        // One row still maps one block in every group — 156 + 42 + 156 MiB.
        assert_eq!(b.bytes_for_rows(1), 354 * (1 << 20));
        assert_eq!(b.bytes_for_rows(70_700), 3966 * (1 << 20));
        assert!(b.fits_requests(std::iter::repeat_n(70_700, 12)));
        assert!(!b.fits_requests(std::iter::repeat_n(70_700, 13)));
    }

    /// Each sequence rounds up on its own; the budget never rounds the SUM of their rows, which
    /// would hide a partial block behind another sequence's remainder.
    #[test]
    fn block_rounding_is_per_sequence() {
        let b = KvBudget::linear(1, 3 * (2 << 20))
            .with_block_groups(&[(4_096, 2 << 20)])
            .unwrap();
        // 2 + 1 blocks, exactly the budget.
        assert!(b.fits_requests([4_097, 1]));
        // 2 + 2 blocks. Rounding the sum instead (8,194 rows -> 3 blocks) would have admitted it.
        assert!(!b.fits_requests([4_097, 4_097]));
    }
}

#[cfg(test)]
mod arrival_rate_tests {
    use super::{ArrivalRate, LoadEstimator};
    use std::time::{Duration, Instant};

    /// The defect: λ was updated only on an arrival, so a burst left it pinned at the burst rate
    /// for as long as the model then sat idle. It must fall toward zero on elapsed time alone.
    #[test]
    fn a_quiet_period_decays_lambda_toward_zero() {
        let t0 = Instant::now();
        let mut lambda = ArrivalRate::new(2.0);
        for i in 0..40 {
            lambda.observe(t0 + Duration::from_millis(50 * i));
        }
        let busy = lambda.rate(t0 + Duration::from_millis(2_000));
        assert!(busy > 5.0, "20 arrivals/s should read hot, got {busy}");

        // Decay is monotone and reaches zero; no arrival is needed to move it.
        let mut prev = busy;
        for quiet_s in 1..=10 {
            let now = t0 + Duration::from_millis(2_000) + Duration::from_secs(quiet_s);
            let idle = lambda.rate(now);
            assert!(idle < prev, "λ must fall while idle: {idle} !< {prev}");
            prev = idle;
        }
        assert!(prev < busy * 0.01, "10 s of quiet leaves {prev} of {busy}");
    }

    /// Averaging `1/dt` is biased high by Jensen's inequality. A depositing estimator is not:
    /// under a steady rate it converges ON the rate, which is what the rung controller compares
    /// against a capacity in the same units.
    #[test]
    fn a_steady_stream_reads_its_own_rate_without_jensen_bias() {
        let t0 = Instant::now();
        let mut lambda = ArrivalRate::new(2.0);
        // 10/s for 20 s, so the estimator is well past its transient.
        for i in 0..200 {
            lambda.observe(t0 + Duration::from_millis(100 * i));
        }
        let rate = lambda.rate(t0 + Duration::from_millis(19_900));
        assert!((rate - 10.0).abs() < 1.0, "expected ~10/s, got {rate}");

        // The jittered stream the old estimator overstated most: the same mean rate with dt
        // alternating an order of magnitude either side of it. An EWMA of 1/dt reads several
        // times high here; a depositing one still reads the mean.
        let mut jittered = ArrivalRate::new(2.0);
        let mut at = 0u64;
        for i in 0..200 {
            at += if i % 2 == 0 { 20 } else { 180 };
            jittered.observe(t0 + Duration::from_millis(at));
        }
        let jr = jittered.rate(t0 + Duration::from_millis(at));
        assert!((jr - 10.0).abs() < 2.0, "jitter must not inflate λ: {jr}");
    }

    /// Never observed means no load, not a divide-by-zero or a NaN into the controller.
    #[test]
    fn an_unobserved_estimator_is_zero_not_nan() {
        let lambda = ArrivalRate::new(2.0);
        assert_eq!(lambda.rate(Instant::now()), 0.0);
        let load = LoadEstimator::default();
        assert_eq!(load.utilization(Instant::now()), 0.0);
        // A reader whose clock is behind the last arrival must not raise e to a positive power.
        let t0 = Instant::now();
        let mut l = ArrivalRate::new(2.0);
        l.observe(t0 + Duration::from_secs(5));
        assert!(l.rate(t0).is_finite() && l.rate(t0) <= l.rate(t0 + Duration::from_secs(5)));
    }
}
