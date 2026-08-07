//! §TTFT — a wall-clock breakdown of one request's time-to-first-token.
//!
//! `PLOW_TTFT_LOG=1`. Off by default; every call site is behind [`on`], which is
//! a cached `OnceLock<bool>` load, so a serving build pays nothing.
//!
//! # Why a global and not a per-request struct
//!
//! The measurement this exists for is `vllm bench serve --max-concurrency 1`, so
//! exactly ONE request is in flight from arrival to first token and a global is
//! unambiguous. Threading a timeline through `Job` -> `Slot` -> `ServeEngine` ->
//! `AmdTpGroup` -> `AmdEngine` would touch six signatures on the decode critical
//! path to measure a phase that only happens once per request. If this is ever
//! wanted at concurrency > 1 it needs the plumbing; until then the counters are
//! reset by the arriving request and read by the first token of the SAME request.
//!
//! The phases partition the interval `[handler entry, first SSE frame]`, which is
//! what `vllm bench serve`'s chat backend stamps as TTFT (the first chunk carrying
//! a `choices` array — see `serve::chat::sse_response`).

use std::sync::atomic::{AtomicU64, Ordering};

/// Whether TTFT logging is active (`--ttft-log` / `PLOW_TTFT_LOG=1`).
/// Reads from [`RuntimeConfig::get`](crate::config::RuntimeConfig::get) —
/// the CLI-parsed global, else its cached env-only snapshot (tests).
pub fn on() -> bool {
    crate::config::RuntimeConfig::get().ttft_log
}

/// One accumulator. `&'static str` label so the dump is self-describing and the
/// call sites read as prose.
pub struct Phase {
    pub label: &'static str,
    ns: AtomicU64,
    /// How many times this phase fired (chunks, segments, launches).
    count: AtomicU64,
}

impl Phase {
    pub const fn new(label: &'static str) -> Self {
        Phase {
            label,
            ns: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }
    #[inline]
    pub fn add(&self, ns: u64) {
        self.ns.fetch_add(ns, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
    }
    /// Count an event that carries no time of its own (segments, launches).
    #[inline]
    pub fn tally(&self, n: u64) {
        self.count.fetch_add(n, Ordering::Relaxed);
    }
    pub fn reset(&self) {
        self.ns.store(0, Ordering::Relaxed);
        self.count.store(0, Ordering::Relaxed);
    }
    pub fn read(&self) -> (u64, u64) {
        (
            self.ns.load(Ordering::Relaxed),
            self.count.load(Ordering::Relaxed),
        )
    }
}

/// Request path, host-side, before the engine ever sees the prompt.
pub static TEMPLATE: Phase = Phase::new("chat template render");
pub static ENCODE: Phase = Phase::new("tokenize (HF BPE)");
/// Submit -> the tick that calls `prefill`: dispatcher wake, formation hold,
/// admission, engine-thread handoff.
pub static QUEUE: Phase = Phase::new("queue: submit -> prefill call");

/// Inside `AmdServe::prefill` / `AmdTpGroup::prefill`.
pub static PF_PLAN: Phase = Phase::new("  plan_chunks + chunk_steps");
pub static PF_PREPARE: Phase = Phase::new("  prefill_prepare (ids/pos/patch upload)");
pub static PF_REARM: Phase = Phase::new("  rearm_prog (counter zeroing)");
pub static PF_XCTR: Phase = Phase::new("  zero_xctr (cross-GPU gates)");
pub static PF_ENQUEUE: Phase = Phase::new("  enqueue_segment (AQL launch)");
pub static PF_DRAIN: Phase = Phase::new("  drain (host barrier, per segment per rank)");
pub static PF_READ: Phase = Phase::new("  read_sampled (D2H of in.ids)");
pub static PF_SEGMENTS: Phase = Phase::new("  (segments x ranks x chunks)");
/// The whole `ServeEngine::prefill` call, as the mux sees it.
pub static PREFILL: Phase = Phase::new("prefill TOTAL (engine thread)");

/// First token: detokenise + push onto the response channel.
pub static FIRST_TOK: Phase = Phase::new("first token detok + channel send");

const PHASES: &[&Phase] = &[
    &TEMPLATE,
    &ENCODE,
    &QUEUE,
    &PREFILL,
    &PF_PLAN,
    &PF_PREPARE,
    &PF_REARM,
    &PF_XCTR,
    &PF_ENQUEUE,
    &PF_DRAIN,
    &PF_READ,
    &PF_SEGMENTS,
    &FIRST_TOK,
];

/// Zero every phase. Called by the arriving request; safe at concurrency 1 only.
pub fn reset() {
    for p in PHASES {
        p.reset();
    }
}

/// Time `f`, adding the elapsed wall to `p`. Returns `f`'s value.
#[inline]
pub fn timed<T>(p: &Phase, f: impl FnOnce() -> T) -> T {
    if !on() {
        return f();
    }
    let t = std::time::Instant::now();
    let out = f();
    p.add(t.elapsed().as_nanos() as u64);
    out
}

/// The padded row cover `plan_chunks` chose, and the chunk list that produced
/// it — the padding question ("what does the bucket ladder cost at 1024?") is
/// answered by this line, not by a phase.
pub static COVER: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// Record the chunk cover for the dump.
pub fn set_cover(chunks: &[u32]) {
    if !on() {
        return;
    }
    let rows: u32 = chunks.iter().sum();
    if let Ok(mut c) = COVER.lock() {
        *c = format!("{chunks:?} = {rows} padded rows");
    }
}

/// Emit the breakdown. `ttft_ns` is arrival -> first SSE frame.
pub fn dump(ttft_ns: u64, prompt_tokens: usize) {
    if !on() {
        return;
    }
    let cover = COVER.lock().map(|c| c.clone()).unwrap_or_default();
    let ms = |ns: u64| ns as f64 / 1e6;
    let mut out = String::new();
    out.push_str(&format!(
        "\nTTFT BREAKDOWN  prompt={prompt_tokens} tok  cover={cover}  \
         TTFT={:.2} ms\n{:<44} {:>10} {:>8} {:>7}\n",
        ms(ttft_ns),
        "phase",
        "ms",
        "n",
        "%",
    ));
    // PREFILL is the parent of the PF_* rows; counting both would double.
    let mut accounted = 0u64;
    for p in PHASES {
        let (ns, n) = p.read();
        if !p.label.starts_with("  ") {
            accounted += ns;
        }
        out.push_str(&format!(
            "{:<44} {:>10.3} {:>8} {:>6.1}%\n",
            p.label,
            ms(ns),
            n,
            100.0 * ns as f64 / ttft_ns.max(1) as f64,
        ));
    }
    out.push_str(&format!(
        "{:<44} {:>10.3} {:>8} {:>6.1}%\n",
        "UNACCOUNTED (HTTP, axum, SSE serialise)",
        ms(ttft_ns.saturating_sub(accounted)),
        "",
        100.0 * ttft_ns.saturating_sub(accounted) as f64 / ttft_ns.max(1) as f64,
    ));
    eprint!("{out}");
}
