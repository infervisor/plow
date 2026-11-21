//! Decode-rung admission policy.

use std::num::{NonZeroU32, NonZeroUsize};

use super::admission::Ewma;

const WIDEN_UTIL: f64 = 0.85;
const NARROW_UTIL: f64 = 0.60;
const NARROW_TICKS: u32 = 128;
const MIN_DWELL_TICKS: u32 = 64;
const MIN_THROUGHPUT_SAMPLES: u64 = 4;
const THROUGHPUT_HYSTERESIS: f64 = 1.03;
/// Narrowing under a backlog idles the slots it takes back while requests wait for them, so it
/// needs a win above sampling noise: Veena at c128 narrowed 64 -> 32 on a ~3% estimate (samples
/// taken beside the codec's GPU work) and held 50 requests queued for 10 s.
const THROUGHPUT_BACKLOG_HYSTERESIS: f64 = 1.10;
const THROUGHPUT_DWELL_TICKS: u32 = 8;

/// Number of leading slots a decode dispatch must cover.
///
/// Slots keep their backend-owned recurrent state until release, so a live slot
/// cannot be compacted across a gap without moving that state.
#[inline]
pub fn occupied_extent(occupied: impl IntoIterator<Item = bool>) -> usize {
    occupied.into_iter().enumerate().fold(
        0,
        |extent, (slot, live)| {
            if live {
                slot + 1
            } else {
                extent
            }
        },
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RungError {
    Empty,
    Zero,
    NotAscending,
    WidestMismatch { widest: usize, capacity: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RungReason {
    Legacy,
    Hold,
    Occupied,
    Backlog,
    Slo,
    Utilization,
    Throughput,
    LowLoad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RungDecision {
    pub actual: usize,
    pub admission: usize,
    pub reason: RungReason,
}

#[derive(Clone, Copy, Debug)]
pub struct RungLoad {
    pub occupied_extent: usize,
    pub queued: usize,
    pub oldest_wait_ms: f64,
    pub arrival_rps: f64,
    pub mean_output_tokens: f64,
    pub slo_ms: f64,
    pub kv_used: f64,
}

#[derive(Clone, Copy, Debug)]
struct RungStat {
    service_ms: Ewma,
    step_ms: Ewma,
    samples: u64,
}

impl Default for RungStat {
    fn default() -> Self {
        Self {
            service_ms: Ewma::new(0.2),
            step_ms: Ewma::new(0.2),
            samples: 0,
        }
    }
}

/// Validated ascending decode widths. Allocation happens once, at model load.
#[derive(Debug)]
pub struct DecodeRungs {
    widths: Box<[NonZeroU32]>,
}

impl DecodeRungs {
    pub fn new(widths: &[u32], capacity: usize) -> Result<Self, RungError> {
        if widths.is_empty() {
            return Err(RungError::Empty);
        }
        let mut out = Vec::with_capacity(widths.len());
        let mut prev = 0u32;
        for &width in widths {
            let Some(nz) = NonZeroU32::new(width) else {
                return Err(RungError::Zero);
            };
            if width <= prev {
                return Err(RungError::NotAscending);
            }
            prev = width;
            out.push(nz);
        }
        let widest = out.last().expect("non-empty").get() as usize;
        if widest != capacity {
            return Err(RungError::WidestMismatch { widest, capacity });
        }
        Ok(Self {
            widths: out.into_boxed_slice(),
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.widths.len()
    }

    #[inline]
    pub fn width(&self, index: usize) -> usize {
        self.widths[index].get() as usize
    }

    #[inline]
    pub fn covering(&self, extent: usize) -> usize {
        self.widths
            .iter()
            .position(|w| w.get() as usize >= extent.max(1))
            .unwrap_or(self.widths.len() - 1)
    }
}

/// Per-model controller. `decide` and `observe_decode` never allocate or block.
pub struct RungController {
    rungs: DecodeRungs,
    stats: Box<[RungStat]>,
    target: usize,
    min_seat: usize,
    dwell_ticks: u32,
    low_load_ticks: u32,
    fast_probe: bool,
    throughput: bool,
    nominal_seat: Option<usize>,
    elastic_seat: Option<usize>,
    elastic_wait_ms: f64,
}

impl RungController {
    pub fn new(rungs: DecodeRungs) -> Self {
        let stats = vec![RungStat::default(); rungs.len()].into_boxed_slice();
        Self {
            rungs,
            stats,
            target: 0,
            min_seat: 0,
            dwell_ticks: MIN_DWELL_TICKS,
            low_load_ticks: 0,
            fast_probe: false,
            throughput: true,
            nominal_seat: None,
            elastic_seat: None,
            elastic_wait_ms: 250.0,
        }
    }

    /// Set minimum admission rung floor (cannot downshift below this seat).
    pub fn with_min_seat(mut self, min_seat: usize) -> Self {
        self.min_seat = min_seat.min(self.rungs.len().saturating_sub(1));
        self.target = self.target.max(self.min_seat);
        self
    }

    /// Set nominal steady-state admission ceiling by target width.
    pub fn with_nominal_max(mut self, width: Option<usize>) -> Self {
        self.nominal_seat = width.map(|w| self.rungs.covering(w));
        self
    }

    /// Set elastic burst admission ceiling by target width.
    pub fn with_elastic_max(mut self, width: Option<usize>) -> Self {
        self.elastic_seat = width.map(|w| self.rungs.covering(w));
        self
    }

    /// Set queue wait threshold in ms before elastic burst expansion activates.
    pub fn with_elastic_wait_ms(mut self, ms: f64) -> Self {
        self.elastic_wait_ms = ms.max(0.0);
        self
    }

    /// Probe the widest rung after one penultimate sample instead of four (the throughput
    /// objective).
    pub fn with_fast_probe(mut self, on: bool) -> Self {
        self.fast_probe = on;
        self
    }

    /// Whether a narrower rung may take the seat on its measured capacity. Off while a coupled
    /// stage (a codec) runs inside the model's ticks: its work lags the rows that caused it, so a
    /// rung's service time is not its own and narrow rungs read cheap (Qwen3-TTS at 40 req/s
    /// narrowed 192 -> 16 and held it with requests waiting upstream). Demand still sets the seat.
    pub fn set_throughput(&mut self, on: bool) {
        self.throughput = on;
    }

    /// Follow a serving-policy switch: the probe is a per-window choice, not a fixed property.
    pub fn set_fast_probe(&mut self, on: bool) {
        self.fast_probe = on;
    }

    /// Clamp the admission target to at most `seat` (e.g. when ITL rises).
    #[inline]
    pub fn clamp_target(&mut self, seat: usize) {
        self.target = self.target.min(seat).max(self.min_seat);
    }

    #[inline]
    pub fn is_legacy(&self) -> bool {
        self.rungs.len() == 1
    }

    #[inline]
    pub fn covering(&self, extent: usize) -> usize {
        self.rungs.covering(extent)
    }

    #[inline]
    pub fn width(&self, index: usize) -> usize {
        self.rungs.width(index)
    }

    #[inline]
    pub fn admission_limit(&self) -> usize {
        self.rungs.width(self.target)
    }

    /// The widest admission limit the controller can reach: its top rung.
    pub fn ceiling(&self) -> usize {
        self.rungs.width(self.rungs.len() - 1)
    }

    pub fn observe_decode(&mut self, rung: usize, service_ms: f64, steps: NonZeroUsize) {
        if !service_ms.is_finite() || service_ms <= 0.0 || rung >= self.stats.len() {
            return;
        }
        self.stats[rung].service_ms.update(service_ms);
        self.stats[rung]
            .step_ms
            .update(service_ms / steps.get() as f64);
        self.stats[rung].samples = self.stats[rung].samples.saturating_add(1);
    }

    pub fn nominal_idx(&self) -> usize {
        let widest_idx = self.rungs.len() - 1;
        self.nominal_seat.unwrap_or(widest_idx).min(widest_idx)
    }

    pub fn decide(&mut self, load: RungLoad) -> RungDecision {
        if self.is_legacy() {
            return RungDecision {
                actual: 0,
                admission: 0,
                reason: RungReason::Legacy,
            };
        }

        self.dwell_ticks = self.dwell_ticks.saturating_add(1);
        let actual = self.rungs.covering(load.occupied_extent);
        let widest_idx = self.rungs.len() - 1;
        let nominal_idx = self.nominal_idx();
        let elastic_idx = self
            .elastic_seat
            .unwrap_or(widest_idx)
            .min(widest_idx)
            .max(nominal_idx);

        let burst_active = load.kv_used < 0.85
            && ((load.queued > 0
                && (load.oldest_wait_ms >= self.elastic_wait_ms || load.queued >= 4))
                || (self.target > nominal_idx
                    && load.occupied_extent > self.rungs.width(nominal_idx)));
        let allowed_ceiling_idx = if burst_active {
            elastic_idx
        } else {
            nominal_idx
        };

        let demanded = load
            .occupied_extent
            .saturating_add(load.queued)
            .min(self.rungs.width(allowed_ceiling_idx));
        let demand_seat = self.rungs.covering(demanded).min(allowed_ceiling_idx);
        let seat = if burst_active && demand_seat > nominal_idx {
            demand_seat
        } else {
            self.throughput_seat(demand_seat, demanded, load.queued, self.target)
                .min(allowed_ceiling_idx)
                .max(self.min_seat)
        };

        // Invariant 2: Nominal Steady-State Floor Under Saturated Demand.
        // When queued > 0 and occupied_extent >= nominal width, scheduler is forbidden
        // from downshifting below nominal_idx due to intermediate rung throughput noise.
        let seat = if load.queued > 0 && load.occupied_extent >= self.rungs.width(nominal_idx) {
            seat.max(nominal_idx)
        } else {
            seat
        };

        let mut reason = if actual > self.target {
            RungReason::Occupied
        } else {
            RungReason::Hold
        };

        // Invariant 1: Monotonic Elastic Burst Draining.
        // When active in elastic burst (target > nominal_idx) and queue has drained (queued == 0),
        // the admission ceiling monotonically tracks draining active requests down to nominal floor:
        // C_admit(t) = max(C_nominal, covering(occupied_extent)).
        if self.target > nominal_idx && load.queued == 0 {
            let drain_seat = self.rungs.covering(load.occupied_extent).max(nominal_idx);
            if drain_seat < self.target {
                self.target = drain_seat;
                self.dwell_ticks = 0;
                self.low_load_ticks = 0;
                reason = RungReason::Throughput;
            }
        }

        if self.target > allowed_ceiling_idx && self.dwell_ticks >= THROUGHPUT_DWELL_TICKS {
            self.target = allowed_ceiling_idx.max(self.min_seat);
            self.dwell_ticks = 0;
            self.low_load_ticks = 0;
            reason = RungReason::Throughput;
        } else if seat < demand_seat
            && seat < self.target
            && self.dwell_ticks >= THROUGHPUT_DWELL_TICKS
        {
            self.target = seat;
            self.dwell_ticks = 0;
            self.low_load_ticks = 0;
            reason = RungReason::Throughput;
        } else if load.queued > 0 && seat > self.target {
            let wait = self.projected_wait_ms(self.target, load);
            let slo = load.slo_ms.max(8.0 * self.service_ms(self.target));
            self.target = seat;
            self.dwell_ticks = 0;
            self.low_load_ticks = 0;
            reason = if wait > slo {
                RungReason::Slo
            } else {
                RungReason::Backlog
            };
        } else if self.target + 1 < self.rungs.len()
            && self.target + 1 <= seat
            && self.utilization(self.target, load) >= WIDEN_UTIL
        {
            self.target += 1;
            self.dwell_ticks = 0;
            self.low_load_ticks = 0;
            reason = RungReason::Utilization;
        } else if load.queued == 0
            && self.target > self.min_seat
            && load.occupied_extent <= self.rungs.width(self.target) / 2
            && load.occupied_extent <= self.rungs.width(self.target - 1)
            && self.utilization(self.target - 1, load) <= NARROW_UTIL
        {
            self.low_load_ticks = self.low_load_ticks.saturating_add(1);
            if self.low_load_ticks >= NARROW_TICKS && self.dwell_ticks >= MIN_DWELL_TICKS {
                self.target = (self.target - 1).max(self.min_seat);
                self.dwell_ticks = 0;
                self.low_load_ticks = 0;
                reason = RungReason::LowLoad;
            }
        } else {
            self.low_load_ticks = 0;
        }

        RungDecision {
            actual,
            admission: self.target,
            reason,
        }
    }

    fn service_ms(&self, rung: usize) -> f64 {
        self.estimate_ms(rung, |stat| stat.service_ms.get())
    }

    fn step_ms(&self, rung: usize) -> f64 {
        self.estimate_ms(rung, |stat| stat.step_ms.get())
    }

    fn estimate_ms(&self, rung: usize, value: impl Fn(&RungStat) -> f64) -> f64 {
        if self.stats[rung].samples > 0 {
            return value(&self.stats[rung]);
        }
        (0..rung)
            .rev()
            .find(|&i| self.stats[i].samples > 0)
            // Until the wider rung has a sample, assume work scales with its
            // width. Reusing the narrow latency unchanged overstates wider
            // capacity and can admit a burst on evidence it does not have.
            .map(|i| {
                value(&self.stats[i]) * self.rungs.width(rung) as f64 / self.rungs.width(i) as f64
            })
            .unwrap_or(0.0)
    }

    fn utilization(&self, rung: usize, load: RungLoad) -> f64 {
        let step_ms = self.step_ms(rung);
        if step_ms <= 0.0 {
            return 0.0;
        }
        let demand_tps = load.arrival_rps.max(0.0) * load.mean_output_tokens.max(1.0);
        let capacity_tps = self.rungs.width(rung) as f64 * 1000.0 / step_ms;
        demand_tps / capacity_tps
    }

    fn projected_wait_ms(&self, rung: usize, load: RungLoad) -> f64 {
        let width = self.rungs.width(rung);
        let waves = load.queued.div_ceil(width.max(1));
        load.oldest_wait_ms.max(0.0) + waves as f64 * self.service_ms(rung)
    }

    fn throughput_seat(
        &self,
        demand_seat: usize,
        demanded: usize,
        queued: usize,
        _current_target: usize,
    ) -> usize {
        if demand_seat == 0 || !self.throughput {
            return demand_seat;
        }
        let widest = self.rungs.len() - 1;
        // Cold start seats the penultimate rung, then probes the widest: a rung only gains samples
        // by running, so holding the widest back until it has samples would cap it forever.
        // `width(widest) > 4`: don't hold back ladders capped at small widths (e.g. realtime
        // profiles with PLOW_DECODE_MAX_RUNG <= 4) where all rungs fit within the latency budget.
        // Waiting for four penultimate samples stalls the rest of a cold burst: 16 x 128-row
        // prompts on a fresh server ran 1 + 7, then four decode-only ticks (43 ms) at rung 8,
        // then the last 8 (12B 128/C16 P99 TTFT 200 ms). Seating the widest outright instead
        // packs all 15 into one launch, which synchronizes the cohort for the whole run (every
        // later wave a 16-burst: mean TTFT 78 -> 107 ms). `fast_probe` keeps the penultimate
        // bootstrap (the 7 + 8 stagger) and probes the widest after ONE sample.
        let probe_after = if self.fast_probe {
            1
        } else {
            MIN_THROUGHPUT_SAMPLES
        };
        if demand_seat == widest
            && self.rungs.width(widest) > 4
            && self.stats[demand_seat].samples < MIN_THROUGHPUT_SAMPLES
            && self.stats[demand_seat - 1].samples < probe_after
        {
            return (demand_seat - 1).max(self.min_seat);
        }
        if self.stats[demand_seat].samples < MIN_THROUGHPUT_SAMPLES
            || self.stats[demand_seat - 1].samples < MIN_THROUGHPUT_SAMPLES
        {
            return demand_seat.max(self.min_seat);
        }
        let demand_step_ms = self.step_ms(demand_seat);
        if demand_step_ms <= 0.0 {
            return demand_seat.max(self.min_seat);
        }
        let capacity = |rung: usize| {
            let ms = self.step_ms(rung);
            if ms > 0.0 {
                self.rungs.width(rung) as f64 * 1000.0 / ms
            } else {
                0.0
            }
        };
        let nominal_idx = self.nominal_idx();
        if self.nominal_seat.is_some() && demand_seat <= nominal_idx {
            return demand_seat.max(self.min_seat);
        }
        let demanded_capacity = if demand_seat <= nominal_idx {
            capacity(demand_seat)
        } else {
            (demanded.min(self.rungs.width(demand_seat)) as f64) * 1000.0 / demand_step_ms
        };
        let min_candidate = if self.nominal_seat.is_some() {
            nominal_idx
        } else {
            self.min_seat
        };
        (min_candidate..demand_seat)
            .filter(|&rung| self.stats[rung].samples >= MIN_THROUGHPUT_SAMPLES)
            .max_by(|&a, &b| capacity(a).total_cmp(&capacity(b)))
            .filter(|&rung| {
                let margin = if queued > 0 {
                    THROUGHPUT_BACKLOG_HYSTERESIS
                } else {
                    THROUGHPUT_HYSTERESIS
                };
                capacity(rung) > demanded_capacity * margin
            })
            .unwrap_or(demand_seat)
            .max(self.min_seat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupied_extent_preserves_slot_identity() {
        assert_eq!(occupied_extent([]), 0);
        assert_eq!(occupied_extent([false, false]), 0);
        assert_eq!(occupied_extent([true, false, false]), 1);
        assert_eq!(occupied_extent([false, false, true, false]), 3);
    }

    fn controller(widths: &[u32]) -> RungController {
        RungController::new(DecodeRungs::new(widths, *widths.last().unwrap() as usize).unwrap())
    }

    #[test]
    fn ceiling_is_the_top_rung_while_admission_starts_at_the_bottom() {
        let c = controller(&[8, 16, 32, 64]);
        assert_eq!(c.admission_limit(), 8);
        assert_eq!(c.ceiling(), 64);
    }

    fn load(occupied_extent: usize, queued: usize) -> RungLoad {
        RungLoad {
            occupied_extent,
            queued,
            oldest_wait_ms: 0.0,
            arrival_rps: 0.0,
            mean_output_tokens: 1.0,
            slo_ms: 250.0,
            kv_used: 0.0,
        }
    }

    /// A rung with few samples must not win the throughput seat on a half-converged average.
    ///
    /// The defect this guards (h100-sxm5, Gemma-4-12B at C16): rung 8 only ran during ramps, so
    /// its zero-seeded EWMA read ~7.4 ms for a 12.5 ms step, its capacity beat rung 16's
    /// 16 / 14.3 ms, and admission narrowed to 8 with 8 requests queued for up to 18 s.
    #[test]
    fn a_sparsely_sampled_rung_does_not_steal_the_throughput_seat() {
        let mut c = controller(&[1, 2, 4, 8, 16]);
        let one = NonZeroUsize::new(1).unwrap();
        for _ in 0..MIN_THROUGHPUT_SAMPLES {
            c.observe_decode(3, 12.5, one);
        }
        for _ in 0..200 {
            c.observe_decode(4, 14.3, one);
        }
        assert_eq!(c.step_ms(3), 12.5);
        c.target = 4;
        let decision = c.decide(load(16, 8));
        assert_eq!(c.admission_limit(), 16, "{decision:?}");
    }

    /// With a coupled stage in the ticks, a cheaper-looking narrow rung does not take the seat.
    #[test]
    fn a_coupled_stage_keeps_the_demanded_rung() {
        let mut c = controller(&[16, 192]);
        for _ in 0..20 {
            c.observe_decode(0, 5.0, NonZeroUsize::MIN);
            c.observe_decode(1, 90.0, NonZeroUsize::MIN);
        }
        c.target = 1;
        c.decide(load(192, 0));
        assert_eq!(c.admission_limit(), 16);
        c.set_throughput(false);
        c.target = 1;
        let decision = c.decide(load(192, 0));
        assert_eq!(c.admission_limit(), 192, "{decision:?}");
    }

    #[test]
    fn occupied_extent_over_half_rung_does_not_narrow_on_low_load() {
        let mut c = controller(&[1, 2, 4, 8, 16, 32]);
        for rung in 0..6 {
            for _ in 0..20 {
                c.observe_decode(rung, 10.0, NonZeroUsize::MIN);
            }
        }
        c.target = 5; // 32
        for _ in 0..200 {
            let decision = c.decide(load(30, 0));
            assert_eq!(
                c.admission_limit(),
                32,
                "occupied 30/32 must not downshift: {decision:?}"
            );
        }
    }

    #[test]
    fn queued_work_does_not_narrow_on_low_load() {
        let mut c = controller(&[1, 2, 4, 8, 16, 32]);
        for rung in 0..6 {
            for _ in 0..20 {
                c.observe_decode(rung, 10.0, NonZeroUsize::MIN);
            }
        }
        c.target = 5; // 32
        for _ in 0..200 {
            let decision = c.decide(load(8, 2));
            assert_eq!(
                c.admission_limit(),
                32,
                "queued work must not downshift: {decision:?}"
            );
        }
    }

    #[test]
    fn min_seat_floor_prevents_downshift() {
        let mut c = controller(&[1, 2, 4, 8, 16, 32]).with_min_seat(5);
        c.target = 5; // 32
        for _ in 0..500 {
            let decision = c.decide(load(0, 0));
            assert_eq!(
                c.admission_limit(),
                32,
                "min_seat 5 must pin rung 32: {decision:?}"
            );
        }
    }

    /// A burst must widen the admission window and a QUIET PERIOD must narrow it back.
    ///
    /// The defect this guards: λ was an EWMA updated only on an arrival, so once a burst ended
    /// it held the burst rate forever. `utilization` never fell to `NARROW_UTIL`, the low-load
    /// branch was never taken, and the window stayed at its burst width for the life of the
    /// process. Nothing here calls `observe` during the quiet phase — elapsed time alone has to
    /// move λ.
    #[test]
    fn a_quiet_period_narrows_the_rung_a_burst_widened() {
        use crate::sched::admission::ArrivalRate;
        use std::time::{Duration, Instant};

        const TICK: Duration = Duration::from_millis(10);
        let mut c = controller(&[1, 2, 4, 8]);
        for rung in 0..4 {
            for _ in 0..20 {
                c.observe_decode(rung, 10.0, NonZeroUsize::MIN);
            }
        }
        let busy = |c: &mut RungController, rps: f64| {
            let mut l = load(8, 0);
            l.arrival_rps = rps;
            l.mean_output_tokens = 100.0;
            c.decide(l)
        };

        // Burst: 8 req/s for 4 s. λ deposits and the window walks out to the widest rung.
        let t0 = Instant::now();
        let mut lambda = ArrivalRate::new(2.0);
        let mut next_arrival = t0;
        for tick in 0..400 {
            let now = t0 + TICK * tick;
            while next_arrival <= now {
                lambda.observe(next_arrival);
                next_arrival += Duration::from_millis(125);
            }
            busy(&mut c, lambda.rate(now));
        }
        let widened = c.admission_limit();
        assert_eq!(widened, 8, "a sustained burst must reach the widest rung");

        // Quiet: not one arrival. Slots drain, so the occupied extent goes to zero too.
        let quiet_from = t0 + TICK * 400;
        let hot = lambda.rate(quiet_from);
        let mut narrowed_at = None;
        for tick in 0..1_200 {
            let now = quiet_from + TICK * tick;
            let mut l = load(0, 0);
            l.arrival_rps = lambda.rate(now);
            l.mean_output_tokens = 100.0;
            c.decide(l);
            if narrowed_at.is_none() && c.admission_limit() < widened {
                narrowed_at = Some(tick);
            }
        }
        assert!(
            narrowed_at.is_some(),
            "12 s of silence must narrow the window"
        );
        assert_eq!(
            c.admission_limit(),
            1,
            "idle must return to the narrowest rung"
        );

        // Control: with λ PINNED at the burst value — the old estimator's behaviour — the same
        // 12 s of silence narrows nothing, because utilization never reaches NARROW_UTIL.
        let mut stale = controller(&[1, 2, 4, 8]);
        for rung in 0..4 {
            for _ in 0..20 {
                stale.observe_decode(rung, 10.0, NonZeroUsize::MIN);
            }
        }
        for _ in 0..400 {
            busy(&mut stale, hot);
        }
        assert_eq!(stale.admission_limit(), 8);
        for _ in 0..1_200 {
            let mut l = load(0, 0);
            l.arrival_rps = hot;
            l.mean_output_tokens = 100.0;
            stale.decide(l);
        }
        assert_eq!(
            stale.admission_limit(),
            8,
            "control: a λ that never decays never narrows"
        );
    }

    #[test]
    fn validates_arbitrary_ascending_widths_including_one() {
        let r = DecodeRungs::new(&[1, 3, 7, 16, 32], 32).unwrap();
        assert_eq!(r.covering(1), 0);
        assert_eq!(r.covering(2), 1);
        assert_eq!(r.covering(8), 3);
        assert_eq!(r.covering(31), 4);
        assert_eq!(DecodeRungs::new(&[], 1).unwrap_err(), RungError::Empty);
        assert_eq!(
            DecodeRungs::new(&[1, 4, 4], 4).unwrap_err(),
            RungError::NotAscending
        );
    }

    #[test]
    fn single_rung_is_an_exact_policy_bypass() {
        let mut c = controller(&[32]);
        for occupied in [0, 1, 17, 32] {
            let d = c.decide(load(occupied, 100));
            assert_eq!(d.reason, RungReason::Legacy);
            assert_eq!(c.admission_limit(), 32);
            assert_eq!(d.actual, 0);
        }
    }

    #[test]
    fn occupied_extent_not_live_count_selects_the_execution_rung() {
        let mut c = controller(&[1, 3, 7, 16, 32]);
        let d = c.decide(load(7, 0));
        assert_eq!(c.width(d.actual), 7);
        assert_eq!(d.reason, RungReason::Occupied);
    }

    #[test]
    fn backlog_widens_to_the_smallest_covering_rung() {
        let mut c = controller(&[1, 3, 7, 16, 32]);
        let d = c.decide(load(1, 5));
        assert_eq!(c.width(d.admission), 7);
        assert_eq!(d.reason, RungReason::Backlog);
    }

    #[test]
    fn backlog_stays_on_the_faster_sampled_standard_rung() {
        let mut c = controller(&[1, 2, 4, 8, 16, 32]);
        for _ in 0..MIN_THROUGHPUT_SAMPLES {
            c.observe_decode(4, 55.0, NonZeroUsize::MIN);
            c.observe_decode(5, 125.0, NonZeroUsize::MIN);
        }
        c.target = 5;
        let d = c.decide(load(20, 80));
        assert_eq!(c.width(d.admission), 16);
        assert_eq!(d.reason, RungReason::Throughput);
        let mut saturated = load(20, 80);
        saturated.arrival_rps = 100.0;
        saturated.mean_output_tokens = 700.0;
        let admission = c.decide(saturated).admission;
        assert_eq!(c.width(admission), 16);
    }

    /// A marginal capacity edge narrows an idle queue's window but not a backlog's.
    #[test]
    fn a_backlog_keeps_the_wide_rung_on_a_marginal_throughput_edge() {
        let mut c = controller(&[1, 2, 4, 8, 16, 32]);
        for _ in 0..MIN_THROUGHPUT_SAMPLES {
            // 16 rows / 10 ms vs 32 rows / 21 ms: rung 16 is ~5% faster.
            c.observe_decode(4, 10.0, NonZeroUsize::MIN);
            c.observe_decode(5, 21.0, NonZeroUsize::MIN);
        }
        c.target = 5;
        c.decide(load(32, 40));
        assert_eq!(c.admission_limit(), 32);
        let d = c.decide(load(32, 0));
        assert_eq!(
            (c.admission_limit(), d.reason),
            (16, RungReason::Throughput)
        );
    }

    #[test]
    fn cold_saturated_backlog_bootstraps_on_the_penultimate_rung() {
        let mut c = controller(&[1, 2, 4, 8, 16, 32]);
        let d = c.decide(load(1, 99));
        assert_eq!(c.width(d.admission), 16);
        assert_eq!(d.reason, RungReason::Backlog);
    }

    #[test]
    fn fast_probe_widens_after_one_penultimate_sample() {
        let mut c = controller(&[1, 2, 4, 8, 16]).with_fast_probe(true);
        let d = c.decide(load(1, 15));
        assert_eq!(
            c.width(d.admission),
            8,
            "a cold backlog still bootstraps on the penultimate rung"
        );
        assert_eq!(d.reason, RungReason::Backlog);
        c.observe_decode(3, 47.0, NonZeroUsize::MIN);
        let admission = c.decide(load(8, 8)).admission;
        assert_eq!(
            c.width(admission),
            16,
            "one sample is enough to probe the widest"
        );
    }

    #[test]
    fn saturated_backlog_probes_the_widest_rung_once_the_penultimate_is_sampled() {
        let mut c = controller(&[1, 2, 4, 8, 16]);
        let admission = c.decide(load(1, 99)).admission;
        assert_eq!(c.width(admission), 8);
        for _ in 0..MIN_THROUGHPUT_SAMPLES {
            c.observe_decode(3, 47.0, NonZeroUsize::MIN);
        }
        let admission = c.decide(load(8, 92)).admission;
        assert_eq!(c.width(admission), 16);
        for _ in 0..MIN_THROUGHPUT_SAMPLES {
            c.observe_decode(4, 60.0, NonZeroUsize::MIN);
        }
        let admission = c.decide(load(16, 84)).admission;
        assert_eq!(c.width(admission), 16);
    }

    #[test]
    fn throughput_cap_waits_for_adjacent_samples_and_a_real_gain() {
        let mut c = controller(&[1, 8, 16, 32]);
        for _ in 0..MIN_THROUGHPUT_SAMPLES {
            c.observe_decode(3, 125.0, NonZeroUsize::MIN);
        }
        let admission = c.decide(load(20, 80)).admission;
        assert_eq!(c.width(admission), 32);

        for _ in 0..MIN_THROUGHPUT_SAMPLES {
            c.observe_decode(2, 63.0, NonZeroUsize::MIN);
        }
        let admission = c.decide(load(20, 80)).admission;
        assert_eq!(c.width(admission), 32);
    }

    #[test]
    fn narrowing_changes_admission_before_high_slots_drain() {
        let mut c = controller(&[1, 4, 16]);
        c.target = 2;
        assert_eq!(c.admission_limit(), 16);
        for _ in 0..64 {
            c.decide(load(12, 0));
        }
        assert_eq!(c.admission_limit(), 4);
        assert_eq!(c.width(c.covering(12)), 16);
    }

    #[test]
    fn service_and_slo_are_keyed_by_actual_rung() {
        let mut c = controller(&[1, 4, 16]);
        c.observe_decode(0, 40.0, NonZeroUsize::MIN);
        let mut l = load(1, 3);
        l.oldest_wait_ms = 300.0;
        let d = c.decide(l);
        assert_eq!(c.width(d.admission), 4);
        assert_eq!(d.reason, RungReason::Slo);
        assert_eq!(c.stats[0].samples, 1);
        assert_eq!(c.stats[1].samples, 0);
    }

    #[test]
    fn unseen_wider_rung_uses_conservative_width_scaled_service() {
        let mut c = controller(&[1, 4, 16]);
        c.observe_decode(0, 2.5, NonZeroUsize::MIN);
        let b1 = c.service_ms(0);
        assert!(b1 > 0.0);
        assert_eq!(c.service_ms(1), b1 * 4.0);
        assert_eq!(c.service_ms(2), b1 * 16.0);
        c.observe_decode(1, 6.0, NonZeroUsize::MIN);
        assert_eq!(c.service_ms(2), c.service_ms(1) * 4.0);
    }

    #[test]
    fn quantum_length_does_not_inflate_utilization_or_widen_admission() {
        let mut single = controller(&[1, 4, 16]);
        let mut multi = controller(&[1, 4, 16]);
        let mut unnormalized = controller(&[1, 4, 16]);
        for _ in 0..32 {
            single.observe_decode(0, 10.0, NonZeroUsize::MIN);
            multi.observe_decode(0, 40.0, NonZeroUsize::new(4).unwrap());
            unnormalized.observe_decode(0, 40.0, NonZeroUsize::MIN);
        }
        let demand = RungLoad {
            arrival_rps: 1.0,
            mean_output_tokens: 50.0,
            ..load(4, 0)
        };
        assert_eq!(single.utilization(0, demand), multi.utilization(0, demand));
        let single_admission = single.decide(demand).admission;
        let multi_admission = multi.decide(demand).admission;
        assert_eq!(single.width(single_admission), 1);
        assert_eq!(multi.width(multi_admission), 1);
        assert_eq!(unnormalized.decide(demand).reason, RungReason::Utilization);
    }

    #[test]
    fn waiting_uses_the_whole_quantum_even_when_capacity_uses_steps() {
        let mut c = controller(&[1, 4, 16]);
        c.observe_decode(0, 40.0, NonZeroUsize::new(4).unwrap());
        let demand = RungLoad {
            oldest_wait_ms: 20.0,
            ..load(1, 2)
        };
        assert_eq!(c.projected_wait_ms(0, demand), 20.0 + 2.0 * c.service_ms(0));
        assert!(c.projected_wait_ms(0, demand) > 20.0 + 2.0 * c.step_ms(0));
        assert_eq!(c.step_ms(1), c.step_ms(0) * 4.0);
        assert_eq!(c.service_ms(1), c.service_ms(0) * 4.0);
    }

    #[test]
    fn variable_completed_quanta_keep_a_per_step_average() {
        let mut single = controller(&[1, 4]);
        let mut variable = controller(&[1, 4]);
        for steps in [4, 4, 2, 1, 3, 1] {
            single.observe_decode(0, 10.0, NonZeroUsize::MIN);
            variable.observe_decode(0, 10.0 * steps as f64, NonZeroUsize::new(steps).unwrap());
        }
        assert_eq!(single.step_ms(0), variable.step_ms(0));
        assert!(variable.service_ms(0) > variable.step_ms(0));
    }

    #[test]
    fn invalid_timing_does_not_poison_either_estimate() {
        let mut c = controller(&[1, 4]);
        for sample in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            c.observe_decode(0, sample, NonZeroUsize::MIN);
        }
        assert_eq!(c.stats[0].samples, 0);
        assert_eq!(c.service_ms(0), 0.0);
        assert_eq!(c.step_ms(0), 0.0);
    }

    #[test]
    fn fully_occupied_target_rung_does_not_narrow() {
        let mut c = controller(&[1, 2, 4, 8, 16]);
        c.target = 2; // width 4
        assert_eq!(c.admission_limit(), 4);
        for _ in 0..128 {
            c.decide(load(4, 0));
        }
        assert_eq!(
            c.admission_limit(),
            4,
            "fully occupied target rung must not narrow"
        );
    }

    #[test]
    fn small_batch_ladder_cold_start_does_not_throttle() {
        let mut c = controller(&[1, 2, 4]);
        let d = c.decide(load(1, 3));
        assert_eq!(
            c.width(d.admission),
            4,
            "cold start on width 4 ladder must admit width 4 directly"
        );
        assert_eq!(d.reason, RungReason::Backlog);
    }

    #[test]
    fn min_seat_prevents_false_throughput_downshift_and_dwell_reset() {
        let mut c = controller(&[1, 2, 4, 8, 16, 32]).with_min_seat(5); // pinned to 32
                                                                        // Observe faster token throughput on rung 4 (width 16) than rung 5 (width 32)
        for _ in 0..10 {
            c.observe_decode(4, 10.0, NonZeroUsize::new(1).unwrap());
            c.observe_decode(5, 50.0, NonZeroUsize::new(1).unwrap());
        }
        c.decide(load(32, 0));
        assert_eq!(c.admission_limit(), 32);
        let d = c.decide(load(32, 0));
        assert_eq!(c.admission_limit(), 32);
        assert_ne!(
            d.reason,
            RungReason::Throughput,
            "min_seat floor must not report Throughput downshift"
        );
        assert!(
            c.dwell_ticks > 1,
            "dwell ticks must not reset when min_seat clamps"
        );
    }

    #[test]
    fn elastic_admission_stretches_under_backlog_and_contracts_when_cleared() {
        let mut c = controller(&[1, 2, 4, 8, 16, 24, 32, 40, 48])
            .with_nominal_max(Some(32))
            .with_elastic_max(Some(40))
            .with_elastic_wait_ms(250.0);

        // Steady state with no backlog: held at nominal 32
        let _ = c.decide(load(30, 0));
        assert!(c.admission_limit() <= 32);

        // Small wait time < 250ms and small queue < 4: holds at nominal 32
        let mut l = load(32, 2);
        l.oldest_wait_ms = 100.0;
        let _ = c.decide(l);
        assert_eq!(c.admission_limit(), 32);

        // Backlog arrival: queue >= 4 or oldest_wait_ms >= 250ms
        let mut burst = load(32, 8);
        burst.oldest_wait_ms = 300.0;
        let d = c.decide(burst);
        assert_eq!(c.admission_limit(), 40, "elastic burst stretches to 40");
        assert!(d.reason == RungReason::Backlog || d.reason == RungReason::Slo);

        // While burst work is still in flight (occupied 35 > nominal 32), burst ceiling stays active to avoid admission lock
        let d = c.decide(load(35, 0));
        assert_eq!(
            c.admission_limit(),
            40,
            "burst ceiling stays open while work occupies elastic slots"
        );

        // Once burst work drains to nominal (occupied <= 32) and queue is clear: contracts back to nominal 32 after dwell
        let mut contracted = false;
        for _ in 0..THROUGHPUT_DWELL_TICKS {
            if c.decide(load(32, 0)).reason == RungReason::Throughput {
                contracted = true;
                break;
            }
        }
        assert!(contracted, "must contract with Throughput reason");
        assert_eq!(c.admission_limit(), 32, "contracts back to nominal 32");
    }

    #[test]
    fn partially_occupied_wide_rung_does_not_beat_fully_occupied_narrow_rung_when_throughput_is_lower(
    ) {
        let mut c = controller(&[1, 2, 4, 8, 16, 32, 64]).with_nominal_max(Some(32));
        // Observe rung 5 (32) at 18ms and rung 6 (64) at 26ms
        for _ in 0..10 {
            c.observe_decode(5, 18.0, NonZeroUsize::new(1).unwrap());
            c.observe_decode(6, 26.0, NonZeroUsize::new(1).unwrap());
        }
        // At 34 demanded tokens:
        // Rung 64 useful throughput = 34 * 1000 / 26 = 1307.7 tok/s
        // Rung 32 useful throughput = 32 * 1000 / 18 = 1777.7 tok/s
        // 1777.7 > 1307.7 * 1.10 = 1438.5 -> rung 32 must win!
        let seat_34 = c.throughput_seat(6, 34, 2, c.target);
        assert_eq!(
            seat_34, 5,
            "at 34 demanded, rung 32 has higher useful throughput than rung 64"
        );

        // At 50 demanded tokens:
        // Rung 64 useful throughput = 50 * 1000 / 26 = 1923.1 tok/s
        // 1777.7 < 1923.1 * 1.10 = 2115.4 -> rung 64 must win!
        let seat_50 = c.throughput_seat(6, 50, 18, c.target);
        assert_eq!(
            seat_50, 6,
            "at 50 demanded, rung 64 has genuinely higher throughput than rung 32"
        );
    }

    #[test]
    fn elastic_burst_contracts_even_with_residual_queued_work() {
        let mut c = controller(&[1, 2, 4, 8, 16, 24, 32, 40, 48])
            .with_nominal_max(Some(32))
            .with_elastic_max(Some(40))
            .with_elastic_wait_ms(250.0);

        // Burst expands to 40
        let mut burst = load(32, 8);
        burst.oldest_wait_ms = 300.0;
        let _ = c.decide(burst);
        assert_eq!(c.admission_limit(), 40);

        // Burst subsides: queued drops below burst threshold (queued < 4 and wait < 250ms)
        // Even with residual queued work (queued = 1), after THROUGHPUT_DWELL_TICKS it must contract to 32
        let mut residual = load(32, 1);
        residual.oldest_wait_ms = 10.0;
        let mut d = c.decide(residual);
        for _ in 1..THROUGHPUT_DWELL_TICKS {
            d = c.decide(residual);
        }
        assert_eq!(
            c.admission_limit(),
            32,
            "must contract to nominal 32 even with residual queued work"
        );
        assert_eq!(d.reason, RungReason::Throughput);
    }

    #[test]
    fn generic_dynamic_nominal_governance_and_kv_pressure_guard() {
        let mut c = controller(&[1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64])
            .with_nominal_max(Some(32))
            .with_elastic_max(Some(48));
        c.target = 8; // width 32

        // Steady state with 32 occupied, 0 queued: must stay at nominal 32, NOT expand to 48
        let _ = c.decide(load(32, 0));
        assert_eq!(
            c.admission_limit(),
            32,
            "steady state must govern at nominal 32"
        );

        // High KV pressure (kv_used = 0.90) with queued requests: burst must be INHIBITED
        let mut kv_pressured = load(32, 8);
        kv_pressured.oldest_wait_ms = 300.0;
        kv_pressured.kv_used = 0.90;
        let _ = c.decide(kv_pressured);
        assert_eq!(
            c.admission_limit(),
            32,
            "burst must be inhibited when kv_used >= 0.85"
        );

        // Normal KV pressure (kv_used = 0.50) with queued requests: burst stretches past nominal
        let mut burst = load(32, 8);
        burst.oldest_wait_ms = 300.0;
        burst.kv_used = 0.50;
        let _ = c.decide(burst);
        assert!(c.admission_limit() > 32, "burst stretches past nominal 32");

        // Once queue clears: contracts back to nominal 32 after dwell
        let mut contracted = false;
        for _ in 0..THROUGHPUT_DWELL_TICKS {
            if c.decide(load(32, 0)).reason == RungReason::Throughput {
                contracted = true;
                break;
            }
        }
        assert!(contracted, "must contract with Throughput reason");
        assert_eq!(c.admission_limit(), 32, "contracts back to nominal 32");
    }

    #[test]
    fn nominal_seat_prevents_throughput_downshift_under_steady_load() {
        let mut c = controller(&[1, 2, 4, 8, 16, 32, 64])
            .with_nominal_max(Some(32))
            .with_elastic_max(Some(64));
        c.target = 5; // width 32

        // Observe rung 4 (16) faster in per-token throughput than rung 5 (32):
        // 16 rows at 8.2ms (1951 tok/s) vs 32 rows at 17.5ms (1828 tok/s)
        for _ in 0..10 {
            c.observe_decode(4, 8.2, NonZeroUsize::new(1).unwrap());
            c.observe_decode(5, 17.5, NonZeroUsize::new(1).unwrap());
        }

        // Even though rung 16 has 6.7% higher token rate, nominal steady state (32)
        // must NOT downshift to 16 when occupied extent is 32 and queued is 0!
        for _ in 0..20 {
            let d = c.decide(load(32, 0));
            assert_eq!(
                c.admission_limit(),
                32,
                "steady state must not throttle below nominal"
            );
            assert_ne!(
                d.reason,
                RungReason::Throughput,
                "must not report throughput downshift"
            );
        }
    }

    #[test]
    fn test_discrete_event_simulation_fuzz_invariants() {
        // Deterministic pseudo-random generator
        struct Rng(u64);
        impl Rng {
            fn next_u64(&mut self) -> u64 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                self.0
            }
            fn gen_range(&mut self, low: usize, high: usize) -> usize {
                if low >= high {
                    return low;
                }
                low + (self.next_u64() as usize % (high - low + 1))
            }
            fn gen_f64(&mut self) -> f64 {
                (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
            }
        }

        let mut rng = Rng(0x5EED_CAFE_DEAD_BEEF);
        let widths = [1, 2, 4, 8, 16, 24, 32, 40, 48, 64];
        let nominal_cap = 32;
        let elastic_cap = 48;
        let mut controller = DecodeRungs::new(&widths, 64)
            .map(|r| {
                RungController::new(r)
                    .with_nominal_max(Some(nominal_cap))
                    .with_elastic_max(Some(elastic_cap))
                    .with_elastic_wait_ms(200.0)
            })
            .unwrap();

        // Seed baseline EWMA samples
        for (i, &w) in widths.iter().enumerate() {
            let base_lat = 2.0 + (w as f64) * 0.45;
            for _ in 0..10 {
                controller.observe_decode(i, base_lat, NonZeroUsize::new(1).unwrap());
            }
        }

        let mut active = 0usize;
        let mut queued = 0usize;
        let mut oldest_wait_ms = 0.0f64;
        let mut prev_admit_limit = controller.admission_limit();
        let mut in_burst_drain = false;

        // Execute 10,000 synthetic discrete-event transitions
        for step in 0..10_000 {
            let regime = (step / 500) % 4;
            match regime {
                0 => {
                    // Steady-state Poisson
                    if rng.gen_f64() < 0.4 {
                        queued = queued.saturating_add(rng.gen_range(1, 3));
                    }
                }
                1 => {
                    // Burst spike
                    if step % 50 == 0 {
                        queued = queued.saturating_add(rng.gen_range(20, 50));
                        oldest_wait_ms = 350.0;
                    }
                }
                2 => {
                    // Burst drain phase
                    queued = 0;
                    oldest_wait_ms = 0.0;
                }
                _ => {
                    // Random churn
                    if rng.gen_f64() < 0.25 {
                        queued = rng.gen_range(0, 30);
                        oldest_wait_ms = rng.gen_f64() * 400.0;
                    }
                }
            }

            let observed_rung = controller.covering(active.max(1));
            let noise = if rng.gen_f64() < 0.05 { 2.5 } else { 1.0 };
            let lat = (2.0 + (widths[observed_rung] as f64) * 0.45) * noise;
            controller.observe_decode(observed_rung, lat, NonZeroUsize::new(1).unwrap());

            let load = RungLoad {
                occupied_extent: active,
                queued,
                oldest_wait_ms,
                arrival_rps: if queued > 0 { 50.0 } else { 5.0 },
                mean_output_tokens: 30.0,
                slo_ms: 1000.0,
                kv_used: if regime == 1 { 0.70 } else { 0.40 },
            };

            let decision = controller.decide(load);
            let admit_limit = controller.admission_limit();

            // Invariant 1: Monotonic Burst Drain
            if active > nominal_cap && queued == 0 {
                if in_burst_drain {
                    assert!(
                        admit_limit <= prev_admit_limit,
                        "step {step}: Invariant 1 violation: burst drain flared admit_limit from {prev_admit_limit} up to {admit_limit}"
                    );
                }
                in_burst_drain = true;
            } else if queued > 0 {
                in_burst_drain = false;
            }

            // Invariant 2: Nominal Steady-State Floor Under Saturated Demand
            if queued > 0 && active >= nominal_cap {
                assert!(
                    admit_limit >= nominal_cap,
                    "step {step}: Invariant 2 violation: admit_limit {admit_limit} fell below nominal {nominal_cap} with queue {queued} and active {active}"
                );
            }

            // Invariant 3: Strict Capacity Bound
            assert!(
                admit_limit <= elastic_cap,
                "step {step}: Invariant 3 violation: admit_limit {admit_limit} exceeded elastic ceiling {elastic_cap}"
            );
            assert!(
                decision.actual <= widths.len() - 1,
                "step {step}: Invariant 3 violation: actual rung index out of bounds"
            );

            // Lifecycle transitions
            let completions = rng.gen_range(0, (active / 4).max(1));
            active = active.saturating_sub(completions);

            if queued > 0 && active < admit_limit {
                let admits = queued.min(admit_limit - active);
                active += admits;
                queued -= admits;
            }

            if queued == 0 {
                oldest_wait_ms = 0.0;
            } else {
                oldest_wait_ms += lat;
            }
            prev_admit_limit = admit_limit;
        }
    }
}
