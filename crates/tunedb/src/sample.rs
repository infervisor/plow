//! Timing samples and the statistics kept from them.

use serde::{Deserialize, Serialize};

/// Why a set of samples could not become a measurement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SampleError {
    /// Fewer samples than the minimum. One timing is an anecdote: it carries no
    /// dispersion, so nothing downstream can tell a real win from noise.
    TooFew { got: usize, need: usize },
    /// A non-finite or non-positive duration.
    NotPositive,
    /// The GPU was contended while measuring. `gpulease` exits 76 for this and
    /// says so: "a contended run silently invalidates timings". Such a run is
    /// discarded, never stored with a caveat.
    Contended,
}

impl std::fmt::Display for SampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleError::TooFew { got, need } => {
                write!(
                    f,
                    "{got} samples is below the {need} required for a robust statistic"
                )
            }
            SampleError::NotPositive => write!(f, "a sample was not a positive finite duration"),
            SampleError::Contended => {
                write!(
                    f,
                    "the GPU was contended during measurement (gpulease rc=76)"
                )
            }
        }
    }
}

impl std::error::Error for SampleError {}

/// Robust summary of a timed run.
///
/// Deliberately not a single number. The architecture requires median plus
/// dispersion and a sample count, because a decode-latency decision made on a
/// mean is a decision made on the tail of whatever else the machine was doing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub median_ns: f64,
    pub p10_ns: f64,
    pub p90_ns: f64,
    pub min_ns: f64,
    pub samples: usize,
}

impl Stats {
    /// Minimum samples for a publishable statistic.
    pub const MIN_SAMPLES: usize = 5;

    pub fn from_samples(mut ns: Vec<f64>) -> Result<Self, SampleError> {
        if ns.len() < Self::MIN_SAMPLES {
            return Err(SampleError::TooFew {
                got: ns.len(),
                need: Self::MIN_SAMPLES,
            });
        }
        if ns.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(SampleError::NotPositive);
        }
        ns.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        Ok(Stats {
            median_ns: percentile(&ns, 0.50),
            p10_ns: percentile(&ns, 0.10),
            p90_ns: percentile(&ns, 0.90),
            min_ns: ns[0],
            samples: ns.len(),
        })
    }

    /// Spread between p90 and the median — the jitter term in the objective.
    pub fn jitter_ns(&self) -> f64 {
        self.p90_ns - self.median_ns
    }

    /// Relative dispersion. A candidate whose spread swamps its margin over a
    /// rival has not been shown to be faster.
    pub fn relative_spread(&self) -> f64 {
        if self.median_ns <= 0.0 {
            return f64::INFINITY;
        }
        (self.p90_ns - self.p10_ns) / self.median_ns
    }

    /// Whether this measurement is faster than `other` by a margin that exceeds
    /// the noise in both. Used instead of comparing medians directly, so a
    /// campaign cannot promote a winner that is inside its own dispersion.
    pub fn beats(&self, other: &Stats) -> bool {
        let noise = (self.jitter_ns()).max(other.jitter_ns());
        self.median_ns + noise < other.median_ns
    }
}

/// Nearest-rank percentile on already-sorted data.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let idx = ((sorted.len() as f64 - 1.0) * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_single_timing_is_not_a_measurement() {
        assert_eq!(
            Stats::from_samples(vec![100.0]),
            Err(SampleError::TooFew { got: 1, need: 5 })
        );
    }

    #[test]
    fn rejects_non_positive_and_non_finite() {
        assert_eq!(
            Stats::from_samples(vec![1.0, 2.0, 3.0, 4.0, 0.0]),
            Err(SampleError::NotPositive)
        );
        assert_eq!(
            Stats::from_samples(vec![1.0, 2.0, 3.0, 4.0, f64::NAN]),
            Err(SampleError::NotPositive)
        );
    }

    #[test]
    fn summarises_robustly() {
        let s = Stats::from_samples((1..=101).map(|v| v as f64).collect()).unwrap();
        assert_eq!(s.median_ns, 51.0);
        assert_eq!(s.p10_ns, 11.0);
        assert_eq!(s.p90_ns, 91.0);
        assert_eq!(s.min_ns, 1.0);
        assert_eq!(s.samples, 101);
    }

    /// The guard that stops a campaign promoting noise: a 1% median edge inside
    /// a 20% spread is not a win.
    #[test]
    fn a_win_inside_the_noise_is_not_a_win() {
        let noisy_fast = Stats::from_samples(vec![90.0, 95.0, 99.0, 105.0, 130.0]).unwrap();
        let noisy_slow = Stats::from_samples(vec![91.0, 96.0, 100.0, 106.0, 131.0]).unwrap();
        assert!(
            !noisy_fast.beats(&noisy_slow),
            "1 ns apart with ~30 ns jitter"
        );

        let tight_fast = Stats::from_samples(vec![50.0, 50.0, 50.0, 51.0, 51.0]).unwrap();
        let tight_slow = Stats::from_samples(vec![99.0, 100.0, 100.0, 100.0, 101.0]).unwrap();
        assert!(tight_fast.beats(&tight_slow), "2x apart with ~1 ns jitter");
    }

    #[test]
    fn jitter_and_spread_are_reported() {
        let s = Stats::from_samples(vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        assert_eq!(s.median_ns, 30.0);
        assert_eq!(s.jitter_ns(), 20.0);
        assert!((s.relative_spread() - (50.0 - 10.0) / 30.0).abs() < 1e-9);
    }
}
