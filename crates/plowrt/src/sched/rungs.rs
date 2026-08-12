//! Decode-rung admission policy.

use std::num::NonZeroU32;

use super::admission::Ewma;

const WIDEN_UTIL: f64 = 0.85;
const NARROW_UTIL: f64 = 0.60;
const NARROW_TICKS: u32 = 32;
const MIN_DWELL_TICKS: u32 = 64;

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
}

#[derive(Clone, Copy, Debug)]
struct RungStat {
    service_ms: Ewma,
    samples: u64,
}

impl Default for RungStat {
    fn default() -> Self {
        Self {
            service_ms: Ewma::new(0.2),
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
    dwell_ticks: u32,
    low_load_ticks: u32,
}

impl RungController {
    pub fn new(rungs: DecodeRungs) -> Self {
        let stats = vec![RungStat::default(); rungs.len()].into_boxed_slice();
        Self {
            rungs,
            stats,
            target: 0,
            dwell_ticks: 0,
            low_load_ticks: 0,
        }
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

    pub fn observe_decode(&mut self, rung: usize, service_ms: f64) {
        if service_ms <= 0.0 || rung >= self.stats.len() {
            return;
        }
        self.stats[rung].service_ms.update(service_ms);
        self.stats[rung].samples = self.stats[rung].samples.saturating_add(1);
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
        let demanded = load
            .occupied_extent
            .saturating_add(load.queued)
            .min(self.rungs.width(self.rungs.len() - 1));
        let seat = self.rungs.covering(demanded);
        let mut reason = if actual > self.target {
            RungReason::Occupied
        } else {
            RungReason::Hold
        };

        if load.queued > 0 && seat > self.target {
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
            && self.utilization(self.target, load) >= WIDEN_UTIL
        {
            self.target += 1;
            self.dwell_ticks = 0;
            self.low_load_ticks = 0;
            reason = RungReason::Utilization;
        } else if load.queued == 0
            && self.target > 0
            && self.utilization(self.target - 1, load) <= NARROW_UTIL
        {
            self.low_load_ticks = self.low_load_ticks.saturating_add(1);
            if self.low_load_ticks >= NARROW_TICKS && self.dwell_ticks >= MIN_DWELL_TICKS {
                self.target -= 1;
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
        if self.stats[rung].samples > 0 {
            return self.stats[rung].service_ms.get();
        }
        (0..rung)
            .rev()
            .find(|&i| self.stats[i].samples > 0)
            .map(|i| self.stats[i].service_ms.get())
            .unwrap_or(0.0)
    }

    fn utilization(&self, rung: usize, load: RungLoad) -> f64 {
        let service_ms = self.service_ms(rung);
        if service_ms <= 0.0 {
            return 0.0;
        }
        let demand_tps = load.arrival_rps.max(0.0) * load.mean_output_tokens.max(1.0);
        let capacity_tps = self.rungs.width(rung) as f64 * 1000.0 / service_ms;
        demand_tps / capacity_tps
    }

    fn projected_wait_ms(&self, rung: usize, load: RungLoad) -> f64 {
        let width = self.rungs.width(rung);
        let waves = load.queued.div_ceil(width.max(1));
        load.oldest_wait_ms.max(0.0) + waves as f64 * self.service_ms(rung)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn controller(widths: &[u32]) -> RungController {
        RungController::new(DecodeRungs::new(widths, *widths.last().unwrap() as usize).unwrap())
    }

    fn load(occupied_extent: usize, queued: usize) -> RungLoad {
        RungLoad {
            occupied_extent,
            queued,
            oldest_wait_ms: 0.0,
            arrival_rps: 0.0,
            mean_output_tokens: 1.0,
            slo_ms: 250.0,
        }
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
    fn narrowing_changes_admission_before_high_slots_drain() {
        let mut c = controller(&[1, 4, 16]);
        c.decide(load(1, 15));
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
        c.observe_decode(0, 40.0);
        let mut l = load(1, 3);
        l.oldest_wait_ms = 300.0;
        let d = c.decide(l);
        assert_eq!(c.width(d.admission), 4);
        assert_eq!(d.reason, RungReason::Slo);
        assert_eq!(c.stats[0].samples, 1);
        assert_eq!(c.stats[1].samples, 0);
    }
}
