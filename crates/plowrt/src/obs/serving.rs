use std::fmt::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
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

    /// A latency histogram in seconds.
    pub fn write_seconds(&self, out: &mut String, name: &str, labels: &str) {
        self.write(out, name, labels, LATENCY, 1e6);
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
pub struct AsrMetrics {
    pub cohort: AtomicBool,
    pub http_requests: AtomicU64,
    pub websocket_sessions: AtomicU64,
    pub active_sessions: AtomicU64,
    pub jobs: AtomicU64,
    pub queued: AtomicU64,
    pub running: AtomicU64,
    pub completed: AtomicU64,
    pub errors: AtomicU64,
    pub cancelled: AtomicU64,
    pub rejected: AtomicU64,
    pub audio_duration_ms: AtomicU64,
    pub first_transcript: Histogram,
    pub e2e: Histogram,
    pub queue: Histogram,
}

#[derive(Default)]
pub struct TtsMetrics {
    pub active: AtomicBool,
    pub http_requests: AtomicU64,
    pub websocket_sessions: AtomicU64,
    pub active_streams: AtomicU64,
    pub completed: AtomicU64,
    pub errors: AtomicU64,
    pub cancelled: AtomicU64,
    pub playback_stalls: AtomicU64,
    pub watchdog_recovers: AtomicU64,
    pub audio_duration_ms: AtomicU64,
    pub first_audio: Histogram,
    pub e2e: Histogram,
}

#[derive(serde::Serialize)]
pub struct LatencySnapshot {
    pub count: u64,
    pub sum_ms: f64,
    /// Conservative bucket upper bounds; null means empty or beyond the last finite bucket.
    pub p50_ms: Option<f64>,
    pub p95_ms: Option<f64>,
    pub p99_ms: Option<f64>,
}

impl Histogram {
    pub fn snapshot(&self) -> LatencySnapshot {
        let bins: Vec<_> = self.bins.iter().map(|b| b.load(Relaxed)).collect();
        let count = bins.iter().sum();
        let quantile = |percent: u64| {
            if count == 0 {
                return None;
            }
            let rank = (count as u128 * percent as u128).div_ceil(100);
            let mut cumulative = 0u128;
            for (index, bin) in bins.iter().enumerate() {
                cumulative += *bin as u128;
                if cumulative >= rank {
                    return LATENCY.get(index).map(|us| *us as f64 / 1000.0);
                }
            }
            None
        };
        LatencySnapshot {
            count,
            sum_ms: self.sum.load(Relaxed) as f64 / 1000.0,
            p50_ms: quantile(50),
            p95_ms: quantile(95),
            p99_ms: quantile(99),
        }
    }
}

#[derive(serde::Serialize)]
pub struct AsrSnapshot {
    pub job_counters_available: bool,
    pub http_requests: u64,
    pub websocket_sessions: u64,
    pub active_sessions: u64,
    pub jobs: u64,
    pub queued: u64,
    pub running: u64,
    pub completed: u64,
    pub errors: u64,
    pub cancelled: u64,
    pub rejected: u64,
    pub audio_duration_seconds: f64,
    pub first_transcript: LatencySnapshot,
    pub e2e: LatencySnapshot,
    pub queue: LatencySnapshot,
}

#[derive(serde::Serialize)]
pub struct TtsSnapshot {
    pub http_requests: u64,
    pub websocket_sessions: u64,
    pub active_streams: u64,
    pub completed: u64,
    pub errors: u64,
    pub cancelled: u64,
    pub playback_stalls: u64,
    pub watchdog_recovers: u64,
    pub audio_duration_seconds: f64,
    pub first_audio: LatencySnapshot,
    pub e2e: LatencySnapshot,
}

impl AsrMetrics {
    pub fn snapshot(&self) -> Option<AsrSnapshot> {
        let http_requests = self.http_requests.load(Relaxed);
        let websocket_sessions = self.websocket_sessions.load(Relaxed);
        let cohort = self.cohort.load(Relaxed);
        if !cohort && http_requests == 0 && websocket_sessions == 0 {
            return None;
        }
        Some(AsrSnapshot {
            job_counters_available: cohort,
            http_requests,
            websocket_sessions,
            active_sessions: self.active_sessions.load(Relaxed),
            jobs: self.jobs.load(Relaxed),
            queued: self.queued.load(Relaxed),
            running: self.running.load(Relaxed),
            completed: self.completed.load(Relaxed),
            errors: self.errors.load(Relaxed),
            cancelled: self.cancelled.load(Relaxed),
            rejected: self.rejected.load(Relaxed),
            audio_duration_seconds: self.audio_duration_ms.load(Relaxed) as f64 / 1000.0,
            first_transcript: self.first_transcript.snapshot(),
            e2e: self.e2e.snapshot(),
            queue: self.queue.snapshot(),
        })
    }
}

impl TtsMetrics {
    pub fn snapshot(&self) -> Option<TtsSnapshot> {
        let http_requests = self.http_requests.load(Relaxed);
        let websocket_sessions = self.websocket_sessions.load(Relaxed);
        let active = self.active.load(Relaxed);
        if !active && http_requests == 0 && websocket_sessions == 0 {
            return None;
        }
        Some(TtsSnapshot {
            http_requests,
            websocket_sessions,
            active_streams: self.active_streams.load(Relaxed),
            completed: self.completed.load(Relaxed),
            errors: self.errors.load(Relaxed),
            cancelled: self.cancelled.load(Relaxed),
            playback_stalls: self.playback_stalls.load(Relaxed),
            watchdog_recovers: self.watchdog_recovers.load(Relaxed),
            audio_duration_seconds: self.audio_duration_ms.load(Relaxed) as f64 / 1000.0,
            first_audio: self.first_audio.snapshot(),
            e2e: self.e2e.snapshot(),
        })
    }
}

#[derive(serde::Serialize)]
pub struct ModelSnapshot {
    pub id: String,
    pub ready: bool,
    pub requests: u64,
    pub queued: u64,
    pub running: u64,
    pub completed: u64,
    pub aborted: u64,
    pub preempted: u64,
    pub rejected: u64,
    pub tick_errors: u64,
    pub prompt_tokens: u64,
    pub cached_tokens: u64,
    pub generated_tokens: u64,
    pub ttft: LatencySnapshot,
    pub e2e: LatencySnapshot,
    pub queue: LatencySnapshot,
    pub asr: Option<AsrSnapshot>,
    pub tts: Option<TtsSnapshot>,
}

#[derive(serde::Serialize)]
pub struct RuntimeSnapshot {
    pub object: &'static str,
    pub started_at_unix_ms: u64,
    pub observed_at_unix_ms: u64,
    pub models: Vec<ModelSnapshot>,
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

pub fn started_at_unix_ms() -> u64 {
    static STARTED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *STARTED.get_or_init(unix_ms)
}

pub fn snapshot(models: &[(String, Arc<Metrics>, bool)]) -> RuntimeSnapshot {
    RuntimeSnapshot {
        object: "runtime.metrics",
        started_at_unix_ms: started_at_unix_ms(),
        observed_at_unix_ms: unix_ms(),
        models: models
            .iter()
            .map(|(id, m, ready)| {
                let s = &m.serving;
                let asr = s.asr.snapshot();
                let tts = s.tts.snapshot();
                let cohort = s.asr.cohort.load(Relaxed);
                ModelSnapshot {
                    id: id.clone(),
                    ready: *ready,
                    requests: if cohort {
                        s.asr.jobs.load(Relaxed)
                    } else {
                        m.requests.load(Relaxed)
                    },
                    queued: if cohort {
                        s.asr.queued.load(Relaxed)
                    } else {
                        m.queued_requests.load(Relaxed)
                    },
                    running: if cohort {
                        s.asr.running.load(Relaxed)
                    } else {
                        s.running.load(Relaxed)
                    },
                    completed: if cohort {
                        s.asr.completed.load(Relaxed)
                    } else {
                        s.success.iter().map(|n| n.load(Relaxed)).sum()
                    },
                    aborted: if cohort {
                        s.asr.cancelled.load(Relaxed)
                    } else {
                        s.aborted.load(Relaxed)
                    },
                    preempted: s.preempted.load(Relaxed),
                    rejected: if cohort {
                        s.asr.rejected.load(Relaxed)
                    } else {
                        m.rejected.load(Relaxed)
                    },
                    tick_errors: s.tick_errors.load(Relaxed),
                    prompt_tokens: s.prompt_computed.load(Relaxed),
                    cached_tokens: s.prompt_cached.load(Relaxed),
                    generated_tokens: s.generation.load(Relaxed),
                    ttft: s.ttft.snapshot(),
                    e2e: if cohort {
                        s.asr.e2e.snapshot()
                    } else {
                        s.e2e.snapshot()
                    },
                    queue: if cohort {
                        s.asr.queue.snapshot()
                    } else {
                        s.queue.snapshot()
                    },
                    asr,
                    tts,
                }
            })
            .collect(),
    }
}

#[derive(Default)]
pub struct ServingMetrics {
    pub asr: AsrMetrics,
    pub tts: TtsMetrics,
    pub running: AtomicU64,
    pub prompt_computed: AtomicU64,
    pub prompt_cached: AtomicU64,
    pub generation: AtomicU64,
    pub success: [AtomicU64; 2],
    pub aborted: AtomicU64,
    /// Aborts that came before the first generated token, and arrival-to-abort time of every abort.
    pub aborted_before_first: AtomicU64,
    pub aborted_elapsed: Histogram,
    pub preempted: AtomicU64,
    pub tick_errors: AtomicU64,
    /// Completed requests judged against `PLOW_TTFT_SLO_MS` / `PLOW_TBT_SLO_MS`, and how many
    /// met the TTFT target, the TBT target (mean inter-token time), and both (goodput).
    pub slo_requests: AtomicU64,
    pub slo_ttft_met: AtomicU64,
    /// `X-Session-Id` retention (see `serve::session`): admissions that resumed retained rows,
    /// that found none, the rows resumed, entries evicted, and what is retained now.
    pub session_hits: AtomicU64,
    pub session_misses: AtomicU64,
    pub session_reused_rows: AtomicU64,
    pub session_evictions: AtomicU64,
    pub session_retained: AtomicU64,
    pub session_retained_rows: AtomicU64,
    pub session_retained_bytes: AtomicU64,
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
    /// Time to first output per turn stage (`serve::turns::Kind`: ASR final, LLM TTFT, TTS TTFA).
    pub turn_stage: [Histogram; 3],
    /// Chat requests whose rendered prompt has reasoning on, and the generated tokens their
    /// reasoning splitter attributed to the trace (see [`ReasoningTally`]).
    pub thinking_requests: AtomicU64,
    pub reasoning_tokens: AtomicU64,
    pub request_reasoning_tokens: Histogram,
    /// Thinking control (`serve::think`): trace blocks, blocks ended by forcing, requests the
    /// reply cap ended, grace tokens spent per controlled request, and the time to the first
    /// reply byte of a thinking request (what a voice caller waits for).
    pub thinking_blocks: AtomicU64,
    pub thinking_forced: AtomicU64,
    pub reply_capped: AtomicU64,
    pub thinking_grace_tokens: Histogram,
    pub first_reply: Histogram,
    /// Voice reply guard (`serve::reply_guard`): responses with a moved planning line, fallback
    /// lines by kind (`wait`, `goodbye`), and answers whose first word marks reasoning.
    pub guard_moved: AtomicU64,
    pub guard_fallback: [AtomicU64; 2],
    pub guard_answer_reasoning: AtomicU64,
    pub spec: SpecMetrics,
}

/// Speculative decoding counters with vLLM's semantics, counted at verification: one draft per span
/// with `k > 0`, its `k` draft tokens, the drafts the verify accepted (an EOS inside the span ends
/// the count; a host stop string later in the span does not), and per draft position the spans that
/// accepted it. `kmax` 0 = speculation off (families not written).
#[derive(Default)]
pub struct SpecMetrics {
    pub kmax: AtomicU64,
    pub drafts: AtomicU64,
    pub draft_tokens: AtomicU64,
    pub accepted: AtomicU64,
    pub accepted_per_pos: [AtomicU64; crate::orch::speculative::SPEC_KMAX],
    /// Decode ticks that ran at least one draft.
    pub ticks: AtomicU64,
    /// Decode ticks that ran plain (or verified without drafts), by first blocker.
    pub skipped: [AtomicU64; SpecSkip::ALL.len()],
    /// Decode rows that could draft (drafting sampling params, tokens and context left), summed
    /// over decode ticks.
    pub eligible_rows: AtomicU64,
}

/// Why a decode tick of a speculation-serving engine carried no drafts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecSkip {
    PrefillTick,
    Backlog,
    CfgRow,
    PlainRow,
    SeededSampledRow,
    RowsOverLimit,
    NoGreedyRows,
    PolicyNoGain,
    NotReady,
    Refused,
}

impl SpecSkip {
    pub const ALL: [SpecSkip; 10] = [
        SpecSkip::PrefillTick,
        SpecSkip::Backlog,
        SpecSkip::CfgRow,
        SpecSkip::PlainRow,
        SpecSkip::SeededSampledRow,
        SpecSkip::RowsOverLimit,
        SpecSkip::NoGreedyRows,
        SpecSkip::PolicyNoGain,
        SpecSkip::NotReady,
        SpecSkip::Refused,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SpecSkip::PrefillTick => "prefill_tick",
            SpecSkip::Backlog => "backlog",
            SpecSkip::CfgRow => "cfg_row",
            SpecSkip::PlainRow => "plain_row",
            SpecSkip::SeededSampledRow => "seeded_sampled_row",
            SpecSkip::RowsOverLimit => "rows_over_limit",
            SpecSkip::NoGreedyRows => "no_greedy_rows",
            SpecSkip::PolicyNoGain => "policy_no_gain",
            SpecSkip::NotReady => "not_ready",
            SpecSkip::Refused => "refused",
        }
    }
}

impl SpecMetrics {
    pub fn record(&self, k: u32, n_acc: u32) {
        if k == 0 {
            return;
        }
        self.drafts.fetch_add(1, Relaxed);
        self.draft_tokens.fetch_add(u64::from(k), Relaxed);
        self.accepted.fetch_add(u64::from(n_acc), Relaxed);
        for c in &self.accepted_per_pos[..(n_acc as usize).min(self.accepted_per_pos.len())] {
            c.fetch_add(1, Relaxed);
        }
    }

    pub fn skip(&self, why: SpecSkip) {
        self.skipped[why as usize].fetch_add(1, Relaxed);
    }
}

/// One chat request's reasoning accounting, recorded once when its response finishes.
pub struct ReasoningTally {
    pub metrics: Arc<Metrics>,
    /// The rendered prompt has reasoning on (counted at admission).
    pub thinking: bool,
    /// Request arrival, until the first reply byte is recorded.
    arrived: Option<Instant>,
}

impl ReasoningTally {
    /// Counts the admitted request when `thinking`.
    pub fn admitted(metrics: Arc<Metrics>, thinking: bool, arrived: Instant) -> Self {
        if thinking {
            metrics.serving.thinking_requests.fetch_add(1, Relaxed);
        }
        Self {
            metrics,
            thinking,
            arrived: thinking.then_some(arrived),
        }
    }

    /// A guarded response's outcome.
    pub fn guarded(
        &self,
        g: &crate::serve::reply_guard::ReplyGuard,
        fallback: Option<crate::serve::reply_guard::Fallback>,
    ) {
        let m = &self.metrics.serving;
        if g.moved {
            m.guard_moved.fetch_add(1, Relaxed);
        }
        if g.answer_reasoning {
            m.guard_answer_reasoning.fetch_add(1, Relaxed);
        }
        if let Some(f) = fallback {
            m.guard_fallback[f as usize].fetch_add(1, Relaxed);
        }
    }

    /// The response's first answer text after its trace; recorded once, for thinking requests.
    pub fn reply_started(&mut self) {
        if let Some(t) = self.arrived.take() {
            self.metrics.serving.first_reply.duration(t.elapsed());
        }
    }

    /// `trace_tokens`: the splitter's count, the same number `reasoning_tokens` reports in
    /// `usage`. Requests that neither asked for reasoning nor produced a trace stay out of the
    /// histogram. `think`: the thinking controller's report, when the request had one.
    pub fn finished(&self, trace_tokens: u64, think: Option<crate::serve::think::ThinkReport>) {
        let m = &self.metrics.serving;
        if trace_tokens > 0 {
            m.reasoning_tokens.fetch_add(trace_tokens, Relaxed);
        }
        if self.thinking || trace_tokens > 0 {
            m.request_reasoning_tokens.tokens(trace_tokens as usize);
        }
        if let Some(t) = think {
            m.thinking_blocks.fetch_add(u64::from(t.blocks), Relaxed);
            m.thinking_forced.fetch_add(u64::from(t.forced), Relaxed);
            m.thinking_grace_tokens.tokens(t.grace_tokens as usize);
        }
    }
}

pub fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// `model_name="…",engine="…"` for a model's metrics: a DP rank (`slug#r`) reports its model
/// and rank, anything else engine 0.
pub fn model_labels(key: &str) -> String {
    let (model, engine) = crate::serve::dp::split_key(key).unwrap_or((key, 0));
    format!("model_name=\"{}\",engine=\"{engine}\"", escape_label(model))
}

pub fn family(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
}

impl ServingMetrics {
    /// The model's cumulative tick time, µs: the device time its dispatcher has spent ticking.
    pub fn device_us(&self) -> u64 {
        self.ticks.iter().map(|h| h.sum.load(Relaxed)).sum()
    }

    pub fn write(out: &mut String, models: &[(String, Arc<Metrics>, bool)]) {
        let labels: Vec<String> = models
            .iter()
            .map(|(slug, _, _)| model_labels(slug))
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
            |m: &Metrics, _| if m.serving.asr.cohort.load(Relaxed) {
                m.serving.asr.running.load(Relaxed)
            } else {
                m.serving.running.load(Relaxed)
            }
        );
        scalar!(
            "vllm:num_requests_waiting",
            "gauge",
            "Requests waiting outside execution slots.",
            |m: &Metrics, _| if m.serving.asr.cohort.load(Relaxed) {
                m.serving.asr.queued.load(Relaxed)
            } else {
                m.queued_requests.load(Relaxed)
            }
        );
        scalar!(
            "vllm:prompt_tokens_total",
            "counter",
            "Prompt tokens including prefix-cache hits, recorded at first generated token.",
            |m: &Metrics, _| m.serving.prompt_computed.load(Relaxed)
                + m.serving.prompt_cached.load(Relaxed)
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
        scalar!(
            "vllm:prefix_cache_queries_total",
            "counter",
            "Prefix cache queries, in terms of number of queried tokens.",
            |m: &Metrics, _| m.serving.prompt_computed.load(Relaxed)
                + m.serving.prompt_cached.load(Relaxed)
        );
        scalar!(
            "vllm:prefix_cache_hits_total",
            "counter",
            "Prefix cache hits, in terms of number of cached tokens.",
            |m: &Metrics, _| m.serving.prompt_cached.load(Relaxed)
        );
        scalar!("plowrt_requests_aborted_total", "counter", "Admitted requests dropped without normal completion or preemption; includes errors and disconnects.", |m: &Metrics, _| if m.serving.asr.cohort.load(Relaxed) { m.serving.asr.cancelled.load(Relaxed) } else { m.serving.aborted.load(Relaxed) });
        scalar!(
            "plowrt_requests_aborted_before_first_token_total",
            "counter",
            "Aborted requests that had not produced a token (client gave up while queued or in prefill).",
            |m: &Metrics, _| m.serving.aborted_before_first.load(Relaxed)
        );
        scalar!(
            "plowrt_requests_thinking_total",
            "counter",
            "Admitted chat requests whose rendered prompt has reasoning on (template thinking switch or a prompt that leaves the trace open).",
            |m: &Metrics, _| m.serving.thinking_requests.load(Relaxed)
        );
        scalar!(
            "plowrt_reasoning_tokens_total",
            "counter",
            "Generated tokens attributed to the reasoning trace, markers included; counted when a chat response finishes.",
            |m: &Metrics, _| m.serving.reasoning_tokens.load(Relaxed)
        );
        scalar!(
            "plowrt_thinking_blocks_total",
            "counter",
            "Reasoning-trace blocks of chat requests under thinking control (a block re-opened after the answer began counts again).",
            |m: &Metrics, _| m.serving.thinking_blocks.load(Relaxed)
        );
        scalar!(
            "plowrt_thinking_budget_forced_total",
            "counter",
            "Trace blocks whose `thinking_token_budget` ran out and whose end the server forced.",
            |m: &Metrics, _| m.serving.thinking_forced.load(Relaxed)
        );
        scalar!(
            "plowrt_reply_capped_total",
            "counter",
            "Chat requests ended by `max_reply_tokens` (finish_reason length, x_plow_finish_reason reply_cap).",
            |m: &Metrics, _| m.serving.reply_capped.load(Relaxed)
        );
        scalar!(
            "plowrt_requests_preempted_total",
            "counter",
            "Requests terminated by model residency preemption.",
            |m: &Metrics, _| m.serving.preempted.load(Relaxed)
        );
        // The same count under vLLM's name. A vLLM dashboard's preemption panel
        // is one of the few that reads as a hard incident, and it matched
        // nothing here.
        scalar!(
            "vllm:num_preemptions_total",
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
            "plowrt_session_hits_total",
            "counter",
            "Session requests that resumed retained KV rows.",
            |m: &Metrics, _| m.serving.session_hits.load(Relaxed)
        );
        scalar!(
            "plowrt_session_misses_total",
            "counter",
            "Session requests that found no retained KV rows to resume.",
            |m: &Metrics, _| m.serving.session_misses.load(Relaxed)
        );
        scalar!(
            "plowrt_session_reused_tokens_total",
            "counter",
            "Prompt rows resumed from retained session KV instead of prefilled.",
            |m: &Metrics, _| m.serving.session_reused_rows.load(Relaxed)
        );
        scalar!(
            "plowrt_session_evictions_total",
            "counter",
            "Retained session sequences dropped: idle TTL, a live request, the KV budget or the cap.",
            |m: &Metrics, _| m.serving.session_evictions.load(Relaxed)
        );
        scalar!(
            "plowrt_session_retained",
            "gauge",
            "Session sequences whose KV is retained.",
            |m: &Metrics, _| m.serving.session_retained.load(Relaxed)
        );
        scalar!(
            "plowrt_session_retained_tokens",
            "gauge",
            "KV rows retained for sessions (a CFG pair counts both members).",
            |m: &Metrics, _| m.serving.session_retained_rows.load(Relaxed)
        );
        scalar!(
            "plowrt_session_retained_bytes",
            "gauge",
            "KV bytes retained for sessions (rows times the engine's KV bytes per row).",
            |m: &Metrics, _| m.serving.session_retained_bytes.load(Relaxed)
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
        scalar!(
            "plowrt_reply_guard_moved_total",
            "counter",
            "Guarded chat responses that had a planning line moved from content to reasoning.",
            |m: &Metrics, _| m.serving.guard_moved.load(Relaxed)
        );
        scalar!(
            "plowrt_reply_guard_answer_reasoning_total",
            "counter",
            "Guarded chat responses whose answer starts with Wait/Alternatively/Hmm/Actually, moved or not.",
            |m: &Metrics, _| m.serving.guard_answer_reasoning.load(Relaxed)
        );
        family(
            out,
            "plowrt_reply_guard_fallback_total",
            "counter",
            "Guarded chat responses left with nothing to say and answered with a fallback line, by kind.",
        );
        for ((_, m, _), labels) in models.iter().zip(&labels) {
            for (i, kind) in ["wait", "goodbye"].iter().enumerate() {
                let _ = writeln!(
                    out,
                    "plowrt_reply_guard_fallback_total{{{labels},kind=\"{kind}\"}} {}",
                    m.serving.guard_fallback[i].load(Relaxed)
                );
            }
        }
        family(
            out,
            "vllm:request_success_total",
            "counter",
            "Finished requests by finish reason; `abort` as vLLM counts it (client disconnect or error).",
        );
        for ((_, m, _), labels) in models.iter().zip(&labels) {
            let cohort = m.serving.asr.cohort.load(Relaxed);
            for (i, reason) in ["stop", "length", "abort"].iter().enumerate() {
                let _ = writeln!(
                    out,
                    "vllm:request_success_total{{{labels},finished_reason=\"{reason}\"}} {}",
                    match (cohort, i) {
                        (true, 0) => m.serving.asr.completed.load(Relaxed),
                        (true, 2) => m.serving.asr.cancelled.load(Relaxed),
                        (true, _) => 0,
                        (false, 2) => m.serving.aborted.load(Relaxed),
                        (false, _) => m.serving.success[i].load(Relaxed),
                    }
                );
            }
        }
        if models
            .iter()
            .any(|(_, m, _)| m.serving.spec.kmax.load(Relaxed) > 0)
        {
            let spec = |name: &str, help: &str, value: &dyn Fn(&SpecMetrics) -> u64| {
                let mut out = String::new();
                family(&mut out, name, "counter", help);
                for ((_, m, _), labels) in models.iter().zip(&labels) {
                    if m.serving.spec.kmax.load(Relaxed) > 0 {
                        let _ = writeln!(out, "{name}{{{labels}}} {}", value(&m.serving.spec));
                    }
                }
                out
            };
            out.push_str(&spec(
                "vllm:spec_decode_num_drafts_total",
                "Speculative verify spans that carried drafts.",
                &|s| s.drafts.load(Relaxed),
            ));
            out.push_str(&spec(
                "vllm:spec_decode_num_draft_tokens_total",
                "Draft tokens proposed.",
                &|s| s.draft_tokens.load(Relaxed),
            ));
            out.push_str(&spec(
                "vllm:spec_decode_num_accepted_tokens_total",
                "Draft tokens accepted.",
                &|s| s.accepted.load(Relaxed),
            ));
            out.push_str(&spec(
                "plowrt_spec_ticks_total",
                "Decode ticks that ran at least one draft.",
                &|s| s.ticks.load(Relaxed),
            ));
            out.push_str(&spec(
                "plowrt_spec_eligible_rows_total",
                "Decode rows that could draft, summed over decode ticks.",
                &|s| s.eligible_rows.load(Relaxed),
            ));
            family(
                out,
                "plowrt_spec_skipped_ticks_total",
                "counter",
                "Decode ticks without drafts, by first blocker.",
            );
            for ((_, m, _), labels) in models.iter().zip(&labels) {
                if m.serving.spec.kmax.load(Relaxed) == 0 {
                    continue;
                }
                for why in SpecSkip::ALL {
                    let _ = writeln!(
                        out,
                        "plowrt_spec_skipped_ticks_total{{{labels},reason=\"{}\"}} {}",
                        why.label(),
                        m.serving.spec.skipped[why as usize].load(Relaxed)
                    );
                }
            }
            family(
                out,
                "vllm:spec_decode_num_accepted_tokens_per_pos_total",
                "counter",
                "Drafts accepted at each draft position.",
            );
            for ((_, m, _), labels) in models.iter().zip(&labels) {
                let kmax = m.serving.spec.kmax.load(Relaxed) as usize;
                for (pos, c) in m
                    .serving
                    .spec
                    .accepted_per_pos
                    .iter()
                    .enumerate()
                    .take(kmax)
                {
                    let _ = writeln!(
                        out,
                        "vllm:spec_decode_num_accepted_tokens_per_pos_total{{{labels},position=\"{pos}\"}} {}",
                        c.load(Relaxed)
                    );
                }
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
            "plowrt_request_aborted_elapsed_seconds",
            aborted_elapsed,
            LATENCY,
            1e6,
            "Arrival through abort; a peak at a fixed time points at a client timeout."
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
            "plowrt_request_reasoning_tokens",
            request_reasoning_tokens,
            TOKENS,
            1.0,
            "Reasoning-trace tokens per finished chat request that had reasoning on or produced a trace."
        );
        hist!(
            "plowrt_thinking_grace_tokens",
            thinking_grace_tokens,
            TOKENS,
            1.0,
            "Trace tokens past the budget spent waiting for a line end, per request under thinking control."
        );
        hist!(
            "plowrt_first_reply_token_seconds",
            first_reply,
            LATENCY,
            1e6,
            "Arrival through the first answer text after the trace, per thinking chat request."
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
            "vllm:request_max_num_generation_tokens",
            output_tokens,
            TOKENS,
            1.0,
            "Generated length of normally completed requests (one sequence per request)."
        );
        hist!(
            "vllm:request_params_max_tokens",
            max_tokens,
            TOKENS,
            1.0,
            "Requested maximum output tokens of normally completed requests."
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
        family(
            out,
            "plowrt_turn_stage_seconds",
            "histogram",
            "Stage start through its first output: ASR final transcript, LLM first token, TTS first audio.",
        );
        for ((_, m, _), labels) in models.iter().zip(&labels) {
            for (stage, h) in ["asr_final", "llm_ttft", "tts_ttfa"]
                .iter()
                .zip(&m.serving.turn_stage)
            {
                if h.bins.iter().any(|b| b.load(Relaxed) > 0) {
                    h.write_seconds(
                        out,
                        "plowrt_turn_stage_seconds",
                        &format!("{labels},stage=\"{stage}\""),
                    );
                }
            }
        }
        scalar!("plowrt_asr_http_requests_total", "counter", "HTTP transcriptions routed to a known model; includes invalid parameters after routing.", |m: &Metrics, _| m.serving.asr.http_requests.load(Relaxed));
        scalar!(
            "plowrt_asr_websocket_sessions_total",
            "counter",
            "WebSocket sessions with a valid start and known model.",
            |m: &Metrics, _| m.serving.asr.websocket_sessions.load(Relaxed)
        );
        scalar!(
            "plowrt_asr_active_sessions",
            "gauge",
            "Active routed audio WebSocket sessions.",
            |m: &Metrics, _| m.serving.asr.active_sessions.load(Relaxed)
        );
        scalar!(
            "plowrt_asr_jobs_total",
            "counter",
            "Admitted cohort transcription jobs, including partial revisions.",
            |m: &Metrics, _| m.serving.asr.jobs.load(Relaxed)
        );
        scalar!(
            "plowrt_asr_jobs_queued",
            "gauge",
            "Cohort jobs waiting for engine admission, including batch formation.",
            |m: &Metrics, _| m.serving.asr.queued.load(Relaxed)
        );
        scalar!(
            "plowrt_asr_jobs_running",
            "gauge",
            "Cohort transcription jobs currently executing.",
            |m: &Metrics, _| m.serving.asr.running.load(Relaxed)
        );
        scalar!(
            "plowrt_asr_jobs_completed_total",
            "counter",
            "Cohort transcription jobs completed successfully.",
            |m: &Metrics, _| m.serving.asr.completed.load(Relaxed)
        );
        scalar!(
            "plowrt_asr_jobs_errors_total",
            "counter",
            "Cohort transcription jobs ending in an engine error.",
            |m: &Metrics, _| m.serving.asr.errors.load(Relaxed)
        );
        scalar!(
            "plowrt_asr_jobs_cancelled_total",
            "counter",
            "Cohort transcription jobs cancelled or disconnected before completion.",
            |m: &Metrics, _| m.serving.asr.cancelled.load(Relaxed)
        );
        scalar!(
            "plowrt_asr_jobs_rejected_total",
            "counter",
            "Cohort job admission refused because the engine queue was full or closed.",
            |m: &Metrics, _| m.serving.asr.rejected.load(Relaxed)
        );
        for (name, help, select) in [
            ("plowrt_asr_first_transcript_seconds", "Cohort submission through its first transcript; excludes audio upload and network transit.", (|m: &AsrMetrics| &m.first_transcript) as fn(&AsrMetrics) -> &Histogram),
            ("plowrt_asr_job_duration_seconds", "Cohort submission through terminal success, error or cancellation.", (|m: &AsrMetrics| &m.e2e) as fn(&AsrMetrics) -> &Histogram),
            ("plowrt_asr_queue_seconds", "Cohort submission through engine admission.", (|m: &AsrMetrics| &m.queue) as fn(&AsrMetrics) -> &Histogram),
        ] {
            family(out, name, "histogram", help);
            for ((_, m, _), labels) in models.iter().zip(&labels) {
                if m.serving.asr.cohort.load(Relaxed) { select(&m.serving.asr).write_seconds(out, name, labels); }
            }
        }
        scalar!(
            "plowrt_asr_audio_duration_seconds_total",
            "counter",
            "Cumulative duration of processed input audio in seconds.",
            |m: &Metrics, _| m.serving.asr.audio_duration_ms.load(Relaxed) as f64 / 1000.0
        );
        scalar!(
            "plowrt_tts_http_requests_total",
            "counter",
            "HTTP speech synthesis requests routed to this model.",
            |m: &Metrics, _| m.serving.tts.http_requests.load(Relaxed)
        );
        scalar!(
            "plowrt_tts_websocket_sessions_total",
            "counter",
            "WebSocket TTS streaming sessions routed to this model.",
            |m: &Metrics, _| m.serving.tts.websocket_sessions.load(Relaxed)
        );
        scalar!(
            "plowrt_tts_active_streams",
            "gauge",
            "Active concurrent TTS streams currently generating or streaming audio.",
            |m: &Metrics, _| m.serving.tts.active_streams.load(Relaxed)
        );
        scalar!(
            "plowrt_tts_completed_total",
            "counter",
            "TTS generation/streaming requests completed successfully.",
            |m: &Metrics, _| m.serving.tts.completed.load(Relaxed)
        );
        scalar!(
            "plowrt_tts_errors_total",
            "counter",
            "TTS synthesis requests ending in an error.",
            |m: &Metrics, _| m.serving.tts.errors.load(Relaxed)
        );
        scalar!(
            "plowrt_tts_cancelled_total",
            "counter",
            "TTS requests or streaming sessions cancelled or aborted by client disconnect.",
            |m: &Metrics, _| m.serving.tts.cancelled.load(Relaxed)
        );
        scalar!(
            "plowrt_tts_playback_stalls_total",
            "counter",
            "Observed audio buffer underruns or playback stalls across streaming sessions.",
            |m: &Metrics, _| m.serving.tts.playback_stalls.load(Relaxed)
        );
        scalar!(
            "plowrt_tts_watchdog_recovers_total",
            "counter",
            "Mute or drone anomalies detected and recovered by audio watchdog.",
            |m: &Metrics, _| m.serving.tts.watchdog_recovers.load(Relaxed)
        );
        scalar!(
            "plowrt_tts_audio_duration_seconds_total",
            "counter",
            "Cumulative duration of synthesized audio output in seconds.",
            |m: &Metrics, _| m.serving.tts.audio_duration_ms.load(Relaxed) as f64 / 1000.0
        );
        for (name, help, select) in [
            ("plowrt_tts_first_audio_seconds", "Time from speech request arrival to first playable audio frame (TTFA).", (|m: &TtsMetrics| &m.first_audio) as fn(&TtsMetrics) -> &Histogram),
            ("plowrt_tts_e2e_duration_seconds", "Time from speech request arrival to full utterance completion.", (|m: &TtsMetrics| &m.e2e) as fn(&TtsMetrics) -> &Histogram),
        ] {
            family(out, name, "histogram", help);
            for ((_, m, _), labels) in models.iter().zip(&labels) {
                if m.serving.tts.active.load(Relaxed) || m.serving.tts.http_requests.load(Relaxed) > 0 || m.serving.tts.websocket_sessions.load(Relaxed) > 0 {
                    select(&m.serving.tts).write_seconds(out, name, labels);
                }
            }
        }
        crate::serve::turns::write_metrics(out);
        family(out, "plowrt_overload_level", "gauge", "Overload level: 0 normal, 1 ASR partials stretched, 2 TTS render-ahead capped, 3 new sessions shed.");
        let _ = writeln!(
            out,
            "plowrt_overload_level {}",
            crate::serve::overload::level()
        );
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
    max_tokens: usize,
    finished: bool,
}

impl RequestMetrics {
    pub fn new(
        metrics: Arc<Metrics>,
        arrived: Instant,
        queued: Instant,
        prompt: usize,
        max_tokens: usize,
    ) -> Self {
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
            max_tokens,
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
        if matches!(reason, FinishReason::ReplyCap) {
            m.reply_capped.fetch_add(1, Relaxed);
        }
        m.success[usize::from(matches!(
            reason,
            FinishReason::Length | FinishReason::ReplyCap
        ))]
        .fetch_add(1, Relaxed);
        let now = Instant::now();
        m.e2e.duration(now.saturating_duration_since(self.arrived));
        m.inference
            .duration(now.saturating_duration_since(self.admitted));
        if let (Some(first), Some(last)) = (self.first, self.last) {
            let decode = last.saturating_duration_since(first);
            m.decode.duration(decode);
            m.tpot.duration(if self.tokens > 1 {
                decode.div_f64((self.tokens - 1) as f64)
            } else {
                Duration::ZERO
            });
            let targets = crate::config::RuntimeConfig::get().slo_targets();
            if targets.active() {
                let ttft_ms = first.saturating_duration_since(self.arrived).as_secs_f64() * 1e3;
                let tpot_ms =
                    decode.as_secs_f64() * 1e3 / self.tokens.saturating_sub(1).max(1) as f64;
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
        if self.max_tokens > 0 {
            m.max_tokens.tokens(self.max_tokens);
        }
        m.run_packets.tokens(packets);
    }
}

impl Drop for RequestMetrics {
    fn drop(&mut self) {
        self.metrics.serving.running.fetch_sub(1, Relaxed);
        if !self.finished {
            let m = &self.metrics.serving;
            m.aborted.fetch_add(1, Relaxed);
            if self.first.is_none() {
                m.aborted_before_first.fetch_add(1, Relaxed);
            }
            m.aborted_elapsed.duration(self.arrived.elapsed());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::stream::FinishReason;

    #[test]
    fn latency_snapshot_uses_bucket_upper_bounds_without_inventing_empty_or_overflow_values() {
        let histogram = Histogram::default();
        assert!(histogram.snapshot().p95_ms.is_none());
        histogram.duration(Duration::from_millis(2));
        histogram.duration(Duration::from_millis(30));
        let snapshot = histogram.snapshot();
        assert_eq!((snapshot.count, snapshot.sum_ms), (2, 32.0));
        assert_eq!(snapshot.p50_ms, Some(5.0));
        assert_eq!(snapshot.p95_ms, Some(50.0));
        histogram.duration(Duration::from_secs(700));
        assert!(histogram.snapshot().p99_ms.is_none());
    }

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
        let mut r = RequestMetrics::new(m.clone(), now, now, 100, 8);
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
        let mut aborted = RequestMetrics::new(m.clone(), now, now, 20, 8);
        aborted.token(0);
        drop(aborted);
        drop(RequestMetrics::new(m.clone(), now, now, 20, 8));
        let mut preempted = RequestMetrics::new(m.clone(), now, now, 20, 8);
        preempted.finish(FinishReason::Preempted, 0);
        drop(preempted);
        assert_eq!(m.serving.running.load(Relaxed), 0);
        assert_eq!(m.serving.aborted.load(Relaxed), 2);
        assert_eq!(m.serving.aborted_before_first.load(Relaxed), 1);
        assert_eq!(m.serving.preempted.load(Relaxed), 1);
        assert_eq!(m.serving.success[0].load(Relaxed), 0);
    }

    /// Each thinking-control family moves once per request: the controller's report at finish,
    /// the first reply byte once, the reply cap through the request's finish reason.
    #[test]
    fn thinking_control_metrics_count_once_per_request() {
        let m = Arc::new(Metrics::default());
        let now = Instant::now();
        let mut t = ReasoningTally::admitted(m.clone(), true, now);
        t.reply_started();
        t.reply_started();
        t.finished(
            12,
            Some(crate::serve::think::ThinkReport {
                blocks: 2,
                forced: 1,
                grace_tokens: 3,
                guard_moved: false,
            }),
        );
        ReasoningTally::admitted(m.clone(), false, now).reply_started();
        let mut r = RequestMetrics::new(m.clone(), now, now, 10, 16);
        r.token(0);
        r.finish(FinishReason::ReplyCap, 1);
        drop(r);
        let mut out = String::new();
        ServingMetrics::write(&mut out, &[("g".into(), m, true)]);
        let l = "{model_name=\"g\",engine=\"0\"}";
        for line in [
            format!("plowrt_thinking_blocks_total{l} 2\n"),
            format!("plowrt_thinking_budget_forced_total{l} 1\n"),
            format!("plowrt_reply_capped_total{l} 1\n"),
            format!("plowrt_thinking_grace_tokens_count{l} 1\n"),
            format!("plowrt_thinking_grace_tokens_sum{l} 3\n"),
            format!("plowrt_first_reply_token_seconds_count{l} 1\n"),
            format!("vllm:request_success_total{{model_name=\"g\",engine=\"0\",finished_reason=\"length\"}} 1\n"),
        ] {
            assert!(out.contains(&line), "missing {line}");
        }
    }

    #[test]
    fn vllm_families_match_vllm_request_accounting() {
        let m = Arc::new(Metrics::default());
        let now = Instant::now();
        let mut r = RequestMetrics::new(m.clone(), now, now, 100, 16);
        r.token(40);
        r.finish(FinishReason::Stop, 1);
        drop(r);
        let mut aborted = RequestMetrics::new(m.clone(), now, now, 10, 32);
        aborted.token(0);
        drop(aborted);
        let mut out = String::new();
        ServingMetrics::write(&mut out, &[("g".into(), m, true)]);
        let l = "{model_name=\"g\",engine=\"0\"}";
        for line in [
            format!("vllm:prompt_tokens_total{l} 110\n"),
            format!("vllm:prompt_tokens_cached_total{l} 40\n"),
            format!("vllm:prefix_cache_queries_total{l} 110\n"),
            format!("vllm:time_to_first_token_seconds_count{l} 2\n"),
            format!("vllm:e2e_request_latency_seconds_count{l} 1\n"),
            format!("vllm:request_time_per_output_token_seconds_count{l} 1\n"),
            format!("vllm:request_params_max_tokens_count{l} 1\n"),
            format!("vllm:request_params_max_tokens_sum{l} 16\n"),
            format!("vllm:request_max_num_generation_tokens_count{l} 1\n"),
            format!("vllm:request_max_num_generation_tokens_sum{l} 1\n"),
            format!("vllm:request_success_total{{model_name=\"g\",engine=\"0\",finished_reason=\"stop\"}} 1\n"),
            format!("vllm:request_success_total{{model_name=\"g\",engine=\"0\",finished_reason=\"abort\"}} 1\n"),
            format!("plowrt_requests_aborted_before_first_token_total{l} 0\n"),
            format!("plowrt_request_aborted_elapsed_seconds_count{l} 1\n"),
        ] {
            assert!(out.contains(&line), "missing {line}");
        }
    }

    #[test]
    fn spec_decode_counters_use_vllm_names_and_positions() {
        let m = Arc::new(Metrics::default());
        let mut out = String::new();
        ServingMetrics::write(&mut out, &[("g".into(), Arc::clone(&m), true)]);
        assert!(
            !out.contains("spec_decode"),
            "absent while speculation is off"
        );
        m.serving.spec.kmax.store(3, Relaxed);
        m.serving.spec.record(3, 2);
        m.serving.spec.record(2, 0);
        m.serving.spec.record(0, 0);
        out.clear();
        ServingMetrics::write(&mut out, &[("g".into(), m, true)]);
        let l = "{model_name=\"g\",engine=\"0\"";
        for line in [
            format!("vllm:spec_decode_num_drafts_total{l}}} 2\n"),
            format!("vllm:spec_decode_num_draft_tokens_total{l}}} 5\n"),
            format!("vllm:spec_decode_num_accepted_tokens_total{l}}} 2\n"),
            format!("vllm:spec_decode_num_accepted_tokens_per_pos_total{l},position=\"0\"}} 1\n"),
            format!("vllm:spec_decode_num_accepted_tokens_per_pos_total{l},position=\"1\"}} 1\n"),
            format!("vllm:spec_decode_num_accepted_tokens_per_pos_total{l},position=\"2\"}} 0\n"),
        ] {
            assert!(out.contains(&line), "missing {line}");
        }
        assert!(!out.contains("position=\"3\""));
        assert!(out.contains(&format!(
            "plowrt_spec_skipped_ticks_total{l},reason=\"prefill_tick\"}} 0\n"
        )));
        assert!(out.contains(&format!("plowrt_spec_eligible_rows_total{l}}} 0\n")));
        assert_eq!(
            out.matches("# TYPE vllm:spec_decode_num_drafts_total counter")
                .count(),
            1
        );
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

    #[test]
    fn tts_and_asr_metrics_exposition_and_snapshots() {
        let m = Arc::new(Metrics::default());
        m.serving.tts.active.store(true, Relaxed);
        m.serving.tts.http_requests.store(5, Relaxed);
        m.serving.tts.websocket_sessions.store(10, Relaxed);
        m.serving.tts.active_streams.store(3, Relaxed);
        m.serving.tts.completed.store(14, Relaxed);
        m.serving.tts.errors.store(1, Relaxed);
        m.serving.tts.cancelled.store(2, Relaxed);
        m.serving.tts.playback_stalls.store(0, Relaxed);
        m.serving.tts.watchdog_recovers.store(1, Relaxed);
        m.serving.tts.audio_duration_ms.store(150_000, Relaxed);
        m.serving.tts.first_audio.duration(Duration::from_millis(115));
        m.serving.tts.e2e.duration(Duration::from_millis(500));

        m.serving.asr.cohort.store(true, Relaxed);
        m.serving.asr.audio_duration_ms.store(200_000, Relaxed);

        let snap = m.serving.tts.snapshot().expect("tts snapshot");
        assert_eq!(snap.http_requests, 5);
        assert_eq!(snap.websocket_sessions, 10);
        assert_eq!(snap.active_streams, 3);
        assert_eq!(snap.completed, 14);
        assert_eq!(snap.errors, 1);
        assert_eq!(snap.cancelled, 2);
        assert_eq!(snap.playback_stalls, 0);
        assert_eq!(snap.watchdog_recovers, 1);
        assert_eq!(snap.audio_duration_seconds, 150.0);
        assert_eq!(snap.first_audio.count, 1);

        let mut out = String::new();
        ServingMetrics::write(&mut out, &[("maya2".into(), m, true)]);
        let l = "{model_name=\"maya2\",engine=\"0\"}";
        assert!(out.contains(&format!("plowrt_tts_http_requests_total{l} 5\n")));
        assert!(out.contains(&format!("plowrt_tts_websocket_sessions_total{l} 10\n")));
        assert!(out.contains(&format!("plowrt_tts_active_streams{l} 3\n")));
        assert!(out.contains(&format!("plowrt_tts_completed_total{l} 14\n")));
        assert!(out.contains(&format!("plowrt_tts_errors_total{l} 1\n")));
        assert!(out.contains(&format!("plowrt_tts_cancelled_total{l} 2\n")));
        assert!(out.contains(&format!("plowrt_tts_playback_stalls_total{l} 0\n")));
        assert!(out.contains(&format!("plowrt_tts_watchdog_recovers_total{l} 1\n")));
        assert!(out.contains(&format!("plowrt_tts_audio_duration_seconds_total{l} 150\n")));
        assert!(out.contains(&format!("plowrt_asr_audio_duration_seconds_total{l} 200\n")));
        assert!(out.contains("plowrt_tts_first_audio_seconds_count"));
        assert!(out.contains("plowrt_tts_e2e_duration_seconds_count"));
    }
}
