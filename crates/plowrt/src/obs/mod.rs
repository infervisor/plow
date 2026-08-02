//! §K Profiling & telemetry — per-SM/per-task timing + aggregate metrics.

pub mod dstep;
pub mod pfx;
pub mod trace;
pub mod ttft;

use std::sync::atomic::{AtomicU64, Ordering};

/// Aggregate runtime metrics, exported as Prometheus text on `/metrics`. Cheap
/// atomic counters; always on (per-task tracing in [`trace`] is sampled/toggled).
#[derive(Default)]
pub struct Metrics {
    pub requests: AtomicU64,
    pub tokens: AtomicU64,
    /// Requests refused before or during service: capacity (no free slot),
    /// empty prompt, prompt at/over `max_ctx`, position past `max_ctx`.
    ///
    /// DISTINCT from [`Metrics::admit_shed`], which is the admission
    /// controller dropping live slots because predicted wait exceeded the SLO.
    /// Both end as a 429; conflating them hides which pressure caused it.
    pub rejected: AtomicU64,
    /// λ and ρ scaled ×1000 (atomics are integer; f64 read back on export).
    /// ρ here is the QUEUEING utilization λ/μ from `sched::admission`, not a
    /// memory or SM occupancy figure — see `LoadEstimator::utilization`.
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
    ///
    /// # The two means are LIFETIME means, and the sums are exported beside them
    ///
    /// `batch_size_mean` and `hold_ms_mean` divide by the count since process
    /// start, so after a few hours of serving they are dominated by history and
    /// stop responding to a behaviour change. They are kept because dashboards
    /// read them, but the `_sum`/`_count` pairs are what a scraper should use:
    /// `rate(plowrt_batch_size_sum[5m]) / rate(plowrt_batch_count_total[5m])`
    /// is a WINDOWED mean and is the number anyone actually wants.
    ///
    /// Exporting the components also removes a second, quieter defect: each
    /// counter is an independent relaxed load, so a numerator read after an
    /// update and a denominator read before one could produce a ratio above the
    /// true maximum batch size. A scraper dividing two counters it sampled
    /// together does not have that problem.
    ///
    /// # HELP/TYPE
    ///
    /// Emitted, because without them every series scrapes as UNTYPED: `rate()`
    /// still works but tooling loses the counter/gauge distinction, and a
    /// `_total` suffix on an untyped series is exactly what exposition linters
    /// flag.
    pub fn to_prometheus(&self) -> String {
        let g = |a: &AtomicU64| a.load(Ordering::Relaxed);
        let mean = |s: u64, n: u64| if n == 0 { 0.0 } else { s as f64 / n as f64 };
        let (bs, bc) = (g(&self.batch_size_sum), g(&self.batch_count));
        let (hs, hc) = (g(&self.hold_ms_sum), g(&self.hold_count));
        format!(
            "# HELP plowrt_requests_total Requests accepted for service.\n\
             # TYPE plowrt_requests_total counter\n\
             plowrt_requests_total {}\n\
             # HELP plowrt_tokens_total Tokens generated (prompt + generation, not split).\n\
             # TYPE plowrt_tokens_total counter\n\
             plowrt_tokens_total {}\n\
             # HELP plowrt_rejected_total Requests refused: capacity, empty prompt, or past max_ctx.\n\
             # TYPE plowrt_rejected_total counter\n\
             plowrt_rejected_total {}\n\
             # HELP plowrt_admit_shed_total Live slots dropped by admission (predicted wait over SLO).\n\
             # TYPE plowrt_admit_shed_total counter\n\
             plowrt_admit_shed_total {}\n\
             # HELP plowrt_arrival_rate EWMA arrival rate lambda, requests/s.\n\
             # TYPE plowrt_arrival_rate gauge\n\
             plowrt_arrival_rate {:.3}\n\
             # HELP plowrt_utilization Queueing utilization rho = lambda/mu. NOT memory or SM occupancy.\n\
             # TYPE plowrt_utilization gauge\n\
             plowrt_utilization {:.3}\n\
             # HELP plowrt_batch_size_sum Sum of formed batch sizes; divide by batch_count for a windowed mean.\n\
             # TYPE plowrt_batch_size_sum counter\n\
             plowrt_batch_size_sum {}\n\
             # HELP plowrt_batch_count_total Batches formed.\n\
             # TYPE plowrt_batch_count_total counter\n\
             plowrt_batch_count_total {}\n\
             # HELP plowrt_batch_size_mean Lifetime mean batch size; prefer rate(sum)/rate(count).\n\
             # TYPE plowrt_batch_size_mean gauge\n\
             plowrt_batch_size_mean {:.3}\n\
             # HELP plowrt_hold_ms_sum Sum of batch-formation hold, ms.\n\
             # TYPE plowrt_hold_ms_sum counter\n\
             plowrt_hold_ms_sum {}\n\
             # HELP plowrt_hold_count_total Holds taken.\n\
             # TYPE plowrt_hold_count_total counter\n\
             plowrt_hold_count_total {}\n\
             # HELP plowrt_hold_ms_mean Lifetime mean hold; prefer rate(sum)/rate(count).\n\
             # TYPE plowrt_hold_ms_mean gauge\n\
             plowrt_hold_ms_mean {:.3}\n",
            g(&self.requests),
            g(&self.tokens),
            g(&self.rejected),
            g(&self.admit_shed),
            g(&self.lambda_milli) as f64 / 1000.0,
            g(&self.util_milli) as f64 / 1000.0,
            bs,
            bc,
            mean(bs, bc),
            hs,
            hc,
            mean(hs, hc),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EVERY EXPORTED SERIES MUST HAVE A WRITER SOMEWHERE IN THE CRATE.
    ///
    /// This is the check that was missing. Four metrics shipped exported and
    /// never incremented — `plowrt_rejected_total`, `plowrt_deadlocks_total`,
    /// `plowrt_kv_preemptions_total` and `plowrt_arrival_rate` — so they read a
    /// hard zero forever. Three of those describe FAILURE conditions, and a
    /// monitoring system reading zero concludes the failure is not happening;
    /// that is worse than not exporting them, because absence prompts a question
    /// and a confident zero does not.
    ///
    /// `deadlocks` and `kv_preemptions` were DELETED rather than wired: there is
    /// no deadlock detector anywhere in the crate, and `Reclaim::Preempt` is
    /// unreachable from the serving path (`memory/streamer.rs` defines it; no
    /// `serve/` or `exec/` caller reaches it). A metric for a subsystem that
    /// does not exist cannot be fixed by incrementing it somewhere.
    ///
    /// Greps the source rather than exercising the server because the failure
    /// is a MISSING call, and only a whole-crate search can see one.
    #[test]
    fn every_exported_metric_has_a_writer() {
        let text = Metrics::default().to_prometheus();
        let exported: Vec<&str> = text
            .lines()
            .filter(|l| !l.starts_with('#'))
            .filter_map(|l| l.split_whitespace().next())
            .collect();
        assert!(
            exported.len() >= 10,
            "expected the full series set, got {exported:?}"
        );

        // Field name for each series: strip the prefix and the Prometheus
        // `_total` suffix, which the field does not carry.
        let src = {
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
            let mut acc = String::new();
            let mut stack = vec![dir];
            while let Some(d) = stack.pop() {
                for e in std::fs::read_dir(&d).expect("read src") {
                    let p = e.expect("entry").path();
                    if p.is_dir() {
                        stack.push(p);
                    } else if p.extension().is_some_and(|x| x == "rs") && !p.ends_with("obs/mod.rs")
                    {
                        acc.push_str(&std::fs::read_to_string(&p).unwrap_or_default());
                    }
                }
            }
            acc
        };

        let mut dead = Vec::new();
        for series in &exported {
            let field = series
                .trim_start_matches("plowrt_")
                .trim_end_matches("_total");
            // The two `_mean` series are DERIVED from a sum and a count; their
            // components are exported beside them and are what gets written.
            if field.ends_with("_mean") {
                continue;
            }
            // `arrival_rate` and `utilization` are stored scaled x1000.
            let candidates = [
                format!("metrics.{field}"),
                format!(".{field},"),
                format!(".{field}\n"),
                match field {
                    "arrival_rate" => "lambda_milli".into(),
                    "utilization" => "util_milli".into(),
                    _ => format!("{field}_milli"),
                },
            ];
            if !candidates.iter().any(|c| src.contains(c.as_str())) {
                dead.push(*series);
            }
        }
        assert!(
            dead.is_empty(),
            "exported but never written — these read a permanent 0 and lie to a scraper: {dead:?}"
        );
    }
}
