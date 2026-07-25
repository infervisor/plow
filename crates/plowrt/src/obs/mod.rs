//! §K Profiling & telemetry — per-SM/per-task timing + aggregate metrics.

pub mod trace;

use std::sync::atomic::{AtomicU64, Ordering};

/// Aggregate runtime metrics, exported as Prometheus text on `/metrics`. Cheap
/// atomic counters; always on (per-task tracing in [`trace`] is sampled/toggled).
#[derive(Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub tokens: AtomicU64,
    pub rejected: AtomicU64,
    pub deadlocks: AtomicU64,
    pub kv_preemptions: AtomicU64,
    /// λ, μ, ρ scaled ×1000 (atomics are integer; f64 read back on export).
    pub lambda_milli: AtomicU64,
    pub util_milli: AtomicU64,

    /// §I Bucket muxer: sum-of-batch-sizes and count of batches formed. Their
    /// ratio is the mean batch size actually run (queue-aware, not per-request).
    pub batch_size_sum: AtomicU64,
    pub batch_count: AtomicU64,
    /// Adaptive batch-formation hold: sum-of-hold-ms and count of holds taken.
    pub hold_ms_sum: AtomicU64,
    pub hold_count: AtomicU64,
    /// Requests shed by admission (predicted wait > SLO or memory OOM).
    pub admit_shed: AtomicU64,
}

impl Metrics {
    #[inline]
    pub fn inc(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    #[inline]
    pub fn add(counter: &AtomicU64, v: u64) {
        counter.fetch_add(v, Ordering::Relaxed);
    }

    /// Render Prometheus exposition text.
    pub fn to_prometheus(&self) -> String {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let mean_batch = {
            let (s, n) = (g(&self.batch_size_sum), g(&self.batch_count));
            if n == 0 { 0.0 } else { s as f64 / n as f64 }
        };
        let mean_hold = {
            let (s, n) = (g(&self.hold_ms_sum), g(&self.hold_count));
            if n == 0 { 0.0 } else { s as f64 / n as f64 }
        };
        format!(
            "# plowrt metrics\n\
             plowrt_requests_total {}\n\
             plowrt_tokens_total {}\n\
             plowrt_rejected_total {}\n\
             plowrt_deadlocks_total {}\n\
             plowrt_kv_preemptions_total {}\n\
             plowrt_arrival_rate {:.3}\n\
             plowrt_utilization {:.3}\n\
             plowrt_batch_size_mean {:.3}\n\
             plowrt_batch_count_total {}\n\
             plowrt_hold_ms_mean {:.3}\n\
             plowrt_admit_shed_total {}\n",
            g(&self.requests),
            g(&self.tokens),
            g(&self.rejected),
            g(&self.deadlocks),
            g(&self.kv_preemptions),
            g(&self.lambda_milli) as f64 / 1000.0,
            g(&self.util_milli) as f64 / 1000.0,
            mean_batch,
            g(&self.batch_count),
            mean_hold,
            g(&self.admit_shed),
        )
    }
}
