use std::fmt::Write;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::Metrics;

const LATENCY: &[u64] = &[
    100, 500, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 250000, 500000, 1000000, 2500000,
    5000000, 10000000, 30000000, 60000000, 120000000, 300000000, 600000000,
];
const TOKENS: &[u64] = &[
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
    262144, 1048576,
];

#[derive(Default)]
pub struct Histogram {
    bins: [AtomicU64; 21],
    sum: AtomicU64,
}

impl Histogram {
    fn observe(&self, value: u64, bounds: &[u64]) {
        self.bins[bounds.partition_point(|&b| b < value)].fetch_add(1, Relaxed);
        self.sum.fetch_add(value, Relaxed);
    }

    pub fn duration(&self, value: Duration) {
        self.observe(value.as_micros().min(u64::MAX as u128) as u64, LATENCY);
    }

    pub fn tokens(&self, value: usize) {
        self.observe(value as u64, TOKENS);
    }

    fn write(&self, out: &mut String, name: &str, labels: &str, bounds: &[u64], scale: f64) {
        let mut count = 0;
        for (i, bin) in self.bins.iter().enumerate() {
            count += bin.load(Relaxed);
            if let Some(&bound) = bounds.get(i) {
                let _ = writeln!(
                    out,
                    "{name}_bucket{{{labels},le=\"{}\"}} {count}",
                    bound as f64 / scale
                );
            } else {
                let _ = writeln!(out, "{name}_bucket{{{labels},le=\"+Inf\"}} {count}");
            }
        }
        let _ = writeln!(
            out,
            "{name}_sum{{{labels}}} {}",
            self.sum.load(Relaxed) as f64 / scale
        );
        let _ = writeln!(out, "{name}_count{{{labels}}} {count}");
    }
}

#[derive(Default)]
pub struct ServingMetrics {
    pub running: AtomicU64,
    pub prompt_computed: AtomicU64,
    pub prompt_cached: AtomicU64,
    pub generation: AtomicU64,
    pub success: [AtomicU64; 2],
    pub aborted: AtomicU64,
    pub preempted: AtomicU64,
    pub tick_errors: AtomicU64,
    /// Completed requests judged against `PLOW_TTFT_SLO_MS` / `PLOW_TBT_SLO_MS`, and how many
    /// met the TTFT target, the TBT target (mean inter-token time), and both (goodput).
    pub slo_requests: AtomicU64,
    pub slo_ttft_met: AtomicU64,
    pub slo_tbt_met: AtomicU64,
    pub slo_met: AtomicU64,
    pub ttft: Histogram,
    pub itl: Histogram,
    pub tpot: Histogram,
    pub e2e: Histogram,
    pub queue: Histogram,
    pub inference: Histogram,
    pub prefill: Histogram,
    pub decode: Histogram,
    pub prompt_tokens: Histogram,
    pub computed_tokens: Histogram,
    pub output_tokens: Histogram,
    pub max_tokens: Histogram,
    pub ticks: [Histogram; 3],
    pub tick_tokens: Histogram,
    pub tick_batch: Histogram,
    pub run_packets: Histogram,
}

pub fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

pub fn family(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
}

impl ServingMetrics {
    pub fn write(out: &mut String, models: &[(String, Arc<Metrics>, bool)]) {
        let labels: Vec<String> = models
            .iter()
            .map(|(slug, _, _)| format!("model_name=\"{}\",engine=\"0\"", escape_label(slug)))
            .collect();
        macro_rules! scalar {
            ($name:expr, $kind:expr, $help:expr, $value:expr) => {{
                family(out, $name, $kind, $help);
                for ((_, metrics, ready), labels) in models.iter().zip(&labels) {
                    let value = $value(metrics, *ready);
                    let _ = writeln!(out, "{}{{{labels}}} {value}", $name);
                }
            }};
        }
        scalar!(
            "plowrt_model_ready",
            "gauge",
            "Dispatcher installed and model admits requests.",
            |_: &Metrics, ready| u8::from(ready)
        );
        scalar!(
            "vllm:num_requests_running",
            "gauge",
            "Requests occupying execution slots.",
            |m: &Metrics, _| m.serving.running.load(Relaxed)
        );
        scalar!(
            "vllm:num_requests_waiting",
            "gauge",
            "Requests waiting outside execution slots.",
            |m: &Metrics, _| m.queued_requests.load(Relaxed)
        );
        scalar!(
            "vllm:prompt_tokens_total",
            "counter",
            "Computed prompt tokens, recorded at first generated token.",
            |m: &Metrics, _| m.serving.prompt_computed.load(Relaxed)
        );
        scalar!(
            "vllm:prompt_tokens_cached_total",
            "counter",
            "Prompt tokens reused, recorded at first generated token.",
            |m: &Metrics, _| m.serving.prompt_cached.load(Relaxed)
        );
        scalar!(
            "vllm:generation_tokens_total",
            "counter",
            "Generated tokens including terminal tokens.",
            |m: &Metrics, _| m.serving.generation.load(Relaxed)
        );
        scalar!("plowrt_requests_aborted_total", "counter", "Admitted requests dropped without normal completion or preemption; includes errors and disconnects.", |m: &Metrics, _| m.serving.aborted.load(Relaxed));
        scalar!(
            "plowrt_requests_preempted_total",
            "counter",
            "Requests terminated by model residency preemption.",
            |m: &Metrics, _| m.serving.preempted.load(Relaxed)
        );
        scalar!(
            "plowrt_tick_errors_total",
            "counter",
            "Ticks returning a device fault or worker failure.",
            |m: &Metrics, _| m.serving.tick_errors.load(Relaxed)
        );
        scalar!(
            "plowrt_slo_requests_total",
            "counter",
            "Completed requests judged against the configured TTFT/TBT targets.",
            |m: &Metrics, _| m.serving.slo_requests.load(Relaxed)
        );
        scalar!(
            "plowrt_slo_ttft_met_total",
            "counter",
            "Judged requests whose time to first token met PLOW_TTFT_SLO_MS.",
            |m: &Metrics, _| m.serving.slo_ttft_met.load(Relaxed)
        );
        scalar!(
            "plowrt_slo_tbt_met_total",
            "counter",
            "Judged requests whose mean inter-token time met PLOW_TBT_SLO_MS.",
            |m: &Metrics, _| m.serving.slo_tbt_met.load(Relaxed)
        );
        scalar!(
            "plowrt_slo_met_total",
            "counter",
            "Judged requests meeting every configured target (goodput).",
            |m: &Metrics, _| m.serving.slo_met.load(Relaxed)
        );
        family(
            out,
            "vllm:request_success_total",
            "counter",
            "Requests completed normally by finish reason.",
        );
        for ((_, m, _), labels) in models.iter().zip(&labels) {
            for (i, reason) in ["stop", "length"].iter().enumerate() {
                let _ = writeln!(
                    out,
                    "vllm:request_success_total{{{labels},finished_reason=\"{reason}\"}} {}",
                    m.serving.success[i].load(Relaxed)
                );
            }
        }
        macro_rules! hist {
            ($name:expr, $field:ident, $bounds:expr, $scale:expr, $help:expr) => {{
                family(out, $name, "histogram", $help);
                for ((_, m, _), labels) in models.iter().zip(&labels) {
                    m.serving.$field.write(out, $name, labels, $bounds, $scale);
                }
            }};
        }
        hist!(
            "vllm:time_to_first_token_seconds",
            ttft,
            LATENCY,
            1e6,
            "Arrival through first generated token; excludes network transit."
        );
        hist!(
            "vllm:inter_token_latency_seconds",
            itl,
            LATENCY,
            1e6,
            "Time between generated tokens observed by the host."
        );
        hist!(
            "vllm:request_time_per_output_token_seconds",
            tpot,
            LATENCY,
            1e6,
            "Per completed request mean time between generated tokens."
        );
        hist!(
            "vllm:e2e_request_latency_seconds",
            e2e,
            LATENCY,
            1e6,
            "Arrival through normal completion."
        );
        hist!(
            "vllm:request_queue_time_seconds",
            queue,
            LATENCY,
            1e6,
            "Dispatcher submission through slot admission."
        );
        hist!(
            "vllm:request_inference_time_seconds",
            inference,
            LATENCY,
            1e6,
            "Slot admission through normal completion."
        );
        hist!(
            "vllm:request_prefill_time_seconds",
            prefill,
            LATENCY,
            1e6,
            "Slot admission through first generated token."
        );
        hist!(
            "vllm:request_decode_time_seconds",
            decode,
            LATENCY,
            1e6,
            "First through last generated token of completed requests."
        );
        hist!(
            "vllm:request_prompt_tokens",
            prompt_tokens,
            TOKENS,
            1.0,
            "Prompt length of normally completed requests."
        );
        hist!(
            "vllm:request_prefill_kv_computed_tokens",
            computed_tokens,
            TOKENS,
            1.0,
            "Uncached prompt length of normally completed requests."
        );
        hist!(
            "vllm:request_generation_tokens",
            output_tokens,
            TOKENS,
            1.0,
            "Generated length of normally completed requests."
        );
        hist!(
            "vllm:request_params_max_tokens",
            max_tokens,
            TOKENS,
            1.0,
            "Requested maximum output tokens at submission."
        );
        hist!(
            "plowrt_tick_generation_tokens",
            tick_tokens,
            TOKENS,
            1.0,
            "Generated tokens per completed mux tick; excludes prefill input rows."
        );
        hist!(
            "plowrt_tick_batch_size",
            tick_batch,
            TOKENS,
            1.0,
            "Occupied request slots at tick submission."
        );
        hist!(
            "plowrt_run_packets",
            run_packets,
            TOKENS,
            1.0,
            "Executor-reported packet count per normally completed request."
        );
        family(
            out,
            "plowrt_tick_duration_seconds",
            "histogram",
            "Host wall duration including worker submission and execution, not isolated GPU time.",
        );
        for ((_, m, _), labels) in models.iter().zip(&labels) {
            for (phase, h) in ["prefill", "decode", "mixed"].iter().zip(&m.serving.ticks) {
                h.write(
                    out,
                    "plowrt_tick_duration_seconds",
                    &format!("{labels},phase=\"{phase}\""),
                    LATENCY,
                    1e6,
                );
            }
        }
    }
}

pub struct RequestMetrics {
    metrics: Arc<Metrics>,
    arrived: Instant,
    admitted: Instant,
    first: Option<Instant>,
    last: Option<Instant>,
    tokens: usize,
    prompt: usize,
    cached: usize,
    finished: bool,
}

impl RequestMetrics {
    pub fn new(metrics: Arc<Metrics>, arrived: Instant, queued: Instant, prompt: usize) -> Self {
        let admitted = Instant::now();
        metrics.serving.running.fetch_add(1, Relaxed);
        metrics
            .serving
            .queue
            .duration(admitted.saturating_duration_since(queued));
        Self {
            metrics,
            arrived,
            admitted,
            first: None,
            last: None,
            tokens: 0,
            prompt,
            cached: 0,
            finished: false,
        }
    }

    pub fn token(&mut self, cached: usize) {
        let now = Instant::now();
        let m = &self.metrics.serving;
        if let Some(last) = self.last {
            m.itl.duration(now.saturating_duration_since(last));
        } else {
            self.first = Some(now);
            self.cached = cached.min(self.prompt);
            m.ttft.duration(now.saturating_duration_since(self.arrived));
            m.prefill
                .duration(now.saturating_duration_since(self.admitted));
            m.prompt_computed
                .fetch_add((self.prompt - self.cached) as u64, Relaxed);
            m.prompt_cached.fetch_add(self.cached as u64, Relaxed);
        }
        self.last = Some(now);
        self.tokens += 1;
        m.generation.fetch_add(1, Relaxed);
    }

    pub fn finish(&mut self, reason: crate::serve::stream::FinishReason, packets: usize) {
        if self.finished {
            return;
        }
        self.finished = true;
        let m = &self.metrics.serving;
        use crate::serve::stream::FinishReason;
        if matches!(reason, FinishReason::Preempted) {
            m.preempted.fetch_add(1, Relaxed);
            return;
        }
        m.success[usize::from(matches!(reason, FinishReason::Length))].fetch_add(1, Relaxed);
        let now = Instant::now();
        m.e2e.duration(now.saturating_duration_since(self.arrived));
        m.inference
            .duration(now.saturating_duration_since(self.admitted));
        if let (Some(first), Some(last)) = (self.first, self.last) {
            let decode = last.saturating_duration_since(first);
            m.decode.duration(decode);
            if self.tokens > 1 {
                m.tpot.duration(decode.div_f64((self.tokens - 1) as f64));
            }
            let targets = crate::config::RuntimeConfig::get().slo_targets();
            if targets.active() {
                let ttft_ms = first.saturating_duration_since(self.arrived).as_secs_f64() * 1e3;
                let tpot_ms = decode.as_secs_f64() * 1e3 / self.tokens.saturating_sub(1).max(1) as f64;
                let (ttft_ok, tbt_ok) = crate::sched::slo::attained(targets, ttft_ms, tpot_ms);
                m.slo_requests.fetch_add(1, Relaxed);
                m.slo_ttft_met.fetch_add(u64::from(ttft_ok), Relaxed);
                m.slo_tbt_met.fetch_add(u64::from(tbt_ok), Relaxed);
                m.slo_met.fetch_add(u64::from(ttft_ok && tbt_ok), Relaxed);
            }
        }
        m.prompt_tokens.tokens(self.prompt);
        m.computed_tokens.tokens(self.prompt - self.cached);
        m.output_tokens.tokens(self.tokens);
        m.run_packets.tokens(packets);
    }
}

impl Drop for RequestMetrics {
    fn drop(&mut self) {
        self.metrics.serving.running.fetch_sub(1, Relaxed);
        if !self.finished {
            self.metrics.serving.aborted.fetch_add(1, Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::stream::FinishReason;

    #[test]
    fn histogram_is_cumulative_and_has_consistent_infinity_count() {
        let h = Histogram::default();
        for value in [0, 1, 2, 2, 1048577] {
            h.tokens(value);
        }
        let mut out = String::new();
        h.write(&mut out, "test", "model_name=\"a\"", TOKENS, 1.0);
        assert!(out.contains("le=\"1\"} 2\n"));
        assert!(out.contains("le=\"2\"} 4\n"));
        assert!(out.contains("le=\"+Inf\"} 5\n"));
        assert!(out.contains("test_count{model_name=\"a\"} 5\n"));
        assert!(out.contains("test_sum{model_name=\"a\"} 1048582\n"));
    }

    #[test]
    fn request_lifecycle_distinguishes_success_abort_and_preemption() {
        let m = Arc::new(Metrics::default());
        let now = Instant::now();
        let mut r = RequestMetrics::new(m.clone(), now, now, 100);
        assert_eq!(m.serving.running.load(Relaxed), 1);
        r.token(40);
        r.token(40);
        r.finish(FinishReason::Length, 3);
        r.finish(FinishReason::Length, 3);
        drop(r);
        assert_eq!(m.serving.running.load(Relaxed), 0);
        assert_eq!(m.serving.success[1].load(Relaxed), 1);
        assert_eq!(m.serving.prompt_computed.load(Relaxed), 60);
        assert_eq!(m.serving.prompt_cached.load(Relaxed), 40);
        assert_eq!(m.serving.generation.load(Relaxed), 2);
        let mut aborted = RequestMetrics::new(m.clone(), now, now, 20);
        aborted.token(0);
        drop(aborted);
        let mut preempted = RequestMetrics::new(m.clone(), now, now, 20);
        preempted.finish(FinishReason::Preempted, 0);
        drop(preempted);
        assert_eq!(m.serving.running.load(Relaxed), 0);
        assert_eq!(m.serving.aborted.load(Relaxed), 1);
        assert_eq!(m.serving.preempted.load(Relaxed), 1);
        assert_eq!(m.serving.success[0].load(Relaxed), 0);
    }

    #[test]
    fn concurrent_models_have_escaped_labels_and_one_metadata_declaration() {
        let a = Arc::new(Metrics::default());
        let b = Arc::new(Metrics::default());
        a.serving.generation.store(11, Relaxed);
        b.serving.generation.store(23, Relaxed);
        let mut out = String::new();
        ServingMetrics::write(
            &mut out,
            &[("a\"\\\n".into(), a, true), ("b".into(), b, false)],
        );
        assert_eq!(
            out.matches("# TYPE vllm:generation_tokens_total counter")
                .count(),
            1
        );
        assert!(out.contains(
            "vllm:generation_tokens_total{model_name=\"a\\\"\\\\\\n\",engine=\"0\"} 11\n"
        ));
        assert!(out.contains("vllm:generation_tokens_total{model_name=\"b\",engine=\"0\"} 23\n"));
        assert!(!out.contains("vllm:kv_cache_usage_perc"));
    }
}
