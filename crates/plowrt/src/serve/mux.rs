//! §I Per-model request muxer — slot-oriented continuous-batching engine.
//!
//! Each loaded model has one dispatcher task with a bounded MPSC ingress.
//! The dispatcher holds a fixed-size **slot table** sized to the largest
//! compiled `bucket.batch` in the bundle. Between decode ticks it admits new
//! arrivals from ingress into idle slots — so a fresh request doesn't wait for
//! prior in-flight requests to reach `max_tokens` before its first token.
//!
//! **Loop (async, per tick).**
//!
//! 1. If no live slots: block on `rx.recv().await`; admit the arrival.
//! 2. Non-blocking drain: `try_recv` while there's an idle slot. Update EWMA λ.
//! 3. Pick the covering bucket for `(Decode, live_slots, max seq)` — this is
//!    the bucket ladder responding to **live shape**, re-picked whenever the
//!    slot composition would round up a different rung.
//! 4. `admit()` gates the tick: `Shed` fails every occupied slot (429 to
//!    each waiter); `Now | Defer` proceeds.
//! 5. Run one tick that advances each live slot by one token against the
//!    picked bucket — on the model's dedicated engine thread (GPU) or the
//!    blocking pool (CPU reference). Finished slots (newline / `max_tokens`)
//!    get their `oneshot` fired and are freed.
//! 6. Update metrics; loop.
//!
//! **Streaming.** Each request carries a `ChunkSender` (an mpsc). The mux emits
//! one `Token { id, text }` per produced token, with `text` the *incremental*
//! detokenized delta (the running decode minus what was already sent), and a
//! final `Done { reason }` on stop. Cancellation is implicit: if the HTTP
//! handler drops the receiver, `send()` fails and the slot is freed next tick.
//!
//! **Not here (yet):** true batched exec that fires all live slots in *one*
//! bucket walk (needs per-slot KV routing / a `SAMPLE_BATCH` opcode) — see
//! the design notes.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use tokio::sync::mpsc;

use crate::asset::{BucketKey, ModelBundle, Phase};
use crate::device::cpu::StreamSet;
use crate::exec::counters::CounterPool;
use crate::exec::indirection::slots as ind_slots;
use crate::exec::oob::{OobChannel, OobMsg};
use crate::memory::streamer::{KvArena, SlotHandle};
use crate::memory::AddressSpace;
use crate::obs::serving::RequestMetrics;
use crate::obs::Metrics;
use crate::sched::admission::LoadEstimator;
use crate::sched::batching::{cold_start_hold_ms, select_bucket};
use crate::sched::multistep::MultiStep;
use crate::sched::rungs::{DecodeRungs, RungController, RungLoad};
use crate::serve::session::{Retention, Seat, SessionTicket};
use crate::serve::stream::{ChunkSender, FinishReason, StreamChunk};
use crate::serve::{
    bucket_has_sample_batch, reference_logits_row, sample_vocab, AppState, GenParams,
    RunObserver,
};
use crate::Result;

// The `PLOW_PF_PACKLOG=1` tick accumulator. It is a DIFFERENT measurement from
// `exec/gpu.rs:3319`, which writes the per-launch `PACKLOG R=... rows=... bucket=...
// chunks=[...]` line the RTX-12 packing bench parses: that one prices a single prefill
// launch, this one prices the WHOLE mux tick and splits it prefill-vs-decode, which is
// what §DISAGG phase-0 needs to bound what disaggregation could recover.
//
// Both gated arms call it (`cuda` at the fused prefill+decode tick, `hsa` at the AMD tick,
// which is either/or), so a build without either feature has no caller — that is the shape
// that got it deleted once already. Keep the call sites and this module in the same commit.
#[allow(dead_code)]
mod packlog {
    use std::sync::atomic::{AtomicU64, Ordering};

    static PREFILL_NS: AtomicU64 = AtomicU64::new(0);
    static DECODE_NS: AtomicU64 = AtomicU64::new(0);
    static PREFILL_TICKS: AtomicU64 = AtomicU64::new(0);
    static DECODE_TICKS: AtomicU64 = AtomicU64::new(0);
    static DECODE_ROWS: AtomicU64 = AtomicU64::new(0);
    static TICKS: AtomicU64 = AtomicU64::new(0);

    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

    /// One line per mux tick: when it ended, what its prefill pass and decode launch cost, and
    /// how many rows decoded. The cumulative summary below hides WHICH ticks carried prefill.
    /// `steps` is the decode quantum the launch ran, `tokens` the tokens the tick emitted, and
    /// `live`/`prefilling` the occupied slots and those still in prefill when the tick began.
    pub(crate) fn tick(
        prefill_ns: u64,
        decode_ns: u64,
        did_prefill: bool,
        rows: usize,
        steps: usize,
        tokens: usize,
        live: usize,
        prefilling: usize,
    ) {
        let t = START.get_or_init(std::time::Instant::now).elapsed();
        eprintln!(
            "PACKLOG TICK t_ms={:.1} prefill_ms={:.2} decode_ms={:.2} did_prefill={} decode_rows={} \
             steps={} tokens={} live={} prefilling={}",
            t.as_secs_f64() * 1e3,
            prefill_ns as f64 / 1e6,
            decode_ns as f64 / 1e6,
            did_prefill as u8,
            rows,
            steps,
            tokens,
            live,
            prefilling
        );
    }

    /// Whether pack-log is active (`--pf-packlog` / `PLOW_PF_PACKLOG=1`).
    /// Reads from `RuntimeConfig::get()` — one atomic load, hot-path safe.
    pub(crate) fn on() -> bool {
        crate::config::RuntimeConfig::get().pf_packlog
    }

    /// Record one mux tick's prefill-pass and decode-launch wall times (ns).
    /// `rows` is the decode batch width that tick, so the reader can turn
    /// `decode_ns` into a per-row cost. Emits a cumulative summary every 1000
    /// ticks; the bench slices the log by line-count brackets for per-cell deltas.
    pub(crate) fn record(
        prefill_ns: u64,
        decode_ns: u64,
        did_prefill: bool,
        did_decode: bool,
        rows: usize,
    ) {
        if on() {
            let t = START.get_or_init(std::time::Instant::now).elapsed();
            eprintln!(
                "PACKLOG PHASE t_ms={:.1} prefill_ms={:.2} decode_ms={:.2} did_prefill={} did_decode={} decode_rows={}",
                t.as_secs_f64() * 1e3,
                prefill_ns as f64 / 1e6,
                decode_ns as f64 / 1e6,
                did_prefill as u8,
                did_decode as u8,
                rows,
            );
        }
        PREFILL_NS.fetch_add(prefill_ns, Ordering::Relaxed);
        DECODE_NS.fetch_add(decode_ns, Ordering::Relaxed);
        if did_prefill {
            PREFILL_TICKS.fetch_add(1, Ordering::Relaxed);
        }
        if did_decode {
            DECODE_TICKS.fetch_add(1, Ordering::Relaxed);
            DECODE_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
        }
        let n = TICKS.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 1000 == 0 {
            eprintln!(
                "PACKLOG WALL prefill_ns={} decode_ns={} prefill_ticks={} decode_ticks={} \
                 decode_rows={} ticks={}",
                PREFILL_NS.load(Ordering::Relaxed),
                DECODE_NS.load(Ordering::Relaxed),
                PREFILL_TICKS.load(Ordering::Relaxed),
                DECODE_TICKS.load(Ordering::Relaxed),
                DECODE_ROWS.load(Ordering::Relaxed),
                n,
            );
        }
    }
}

/// Dispatcher config. Cold-path — set once at startup from CLI flags.
#[derive(Clone, Copy, Debug)]
pub struct MuxConfig {
    /// Upper bound on the arrival-rate-driven batch-formation hold (ms) used
    /// only in the cold-start path (empty slot table) — with slots always
    /// draining, the hot path never sleeps.
    pub max_hold_ms: f64,
    /// Latency target (ms) the decode-rung controller admits against, and the
    /// floor under a queued request's TTL. It moves the admission WINDOW and
    /// drops requests that have not started; it never touches a live slot.
    pub slo_ms: f64,
    /// Enable multi-step decode: produce `n` tokens per tick (SGLang overlap
    /// scheduling). Steps scale inversely with batch size — small batches are
    /// host-turnaround-bound so more steps hide the latency.
    pub multi_step: bool,
    /// GPU packet queue depth. `0` = synchronous (no queue), matching current
    /// behavior. Non-zero enables double-buffered work submission via
    /// [`PacketQueue`](crate::exec::queue::PacketQueue) to overlap host
    /// bookkeeping with device launch latency.
    pub queue_depth: usize,
    /// Maximum requests waiting outside the engine slot table. `0` derives a
    /// bound of four full engine batches.
    pub max_queued_requests: usize,
    /// Skip or end the cold-start hold once nothing else is queued or tokenizing.
    pub idle_dispatch: bool,
}

impl Default for MuxConfig {
    fn default() -> Self {
        MuxConfig {
            max_hold_ms: 8.0,
            slo_ms: 250.0,
            multi_step: true,
            queue_depth: 0,
            max_queued_requests: 0,
            idle_dispatch: true,
        }
    }
}

/// One request through the mux. Tokens stream out on `respond` as they are
/// produced; `Done`/`Err` terminates the stream. The receiver drives the
/// HTTP handler (SSE frames or a buffered non-streaming response).
pub struct Job {
    /// Tokenized prompt. Encoding happens on the submitting task (the HTTP
    /// handler) — a large prompt must not stall the dispatcher loop, which is
    /// the serialized decode critical path for every live stream.
    pub prompt_ids: Vec<u32>,
    pub gen: GenParams,
    pub arrived: Instant,
    pub respond: ChunkSender,
    pub opts: JobOpts,
}

/// Per-request options beyond the text sampling contract. `Default` is a plain text request.
#[derive(Default)]
pub struct JobOpts {
    pub class: JobClass,
    /// Stream token ids only: no detokenization, and a consumer that falls behind is parked
    /// (not fed) instead of cut, up to [`PARK_TIMEOUT`].
    pub raw_tokens: bool,
    /// Host-built prefill rows for a speech/audio packet; `None` for text.
    pub speech: Option<Box<SpeechJob>>,
    /// `X-Session-Id`: resume the session's retained rows, and retain this request's.
    pub session: Option<Box<SessionTicket>>,
    /// The voice turn this request serves ([`crate::serve::turns`]).
    pub turn: Option<crate::serve::turns::TurnKey>,
    /// A later turn of a session that already had one: seated ahead of requests opening a
    /// session, until either has waited [`crate::serve::cosched::max_wait`].
    pub continuing: bool,
    /// The prompt's prefix-cache block hashes, when the DP router already computed them.
    pub prefix: Option<crate::memory::vmm::PrefixKey>,
    /// Multimodal soft-token rows the prompt's bit-31 ids name (`serve::mm`).
    pub mm: Option<Box<crate::serve::mm::MmJob>>,
}

/// Queue priority. Orders the waiting queue and serial prefill, and scales the queue TTL.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum JobClass {
    /// A user is waiting on this one's whole, short output: an ASR final (end of speech to
    /// transcript is the turn's first deadline).
    Final = 0,
    /// A user is waiting on this one's first output (a stream's first audio).
    Critical = 1,
    #[default]
    Normal = 2,
    /// Yields to everything else, and is superseded if it waits: revisable partial results.
    Bulk = 3,
}

impl JobClass {
    /// Multiple of the queue TTL this class may wait. A bulk result that has queued a tenth of it
    /// has been superseded by the time it would run.
    fn ttl_scale(self) -> f64 {
        match self {
            JobClass::Final | JobClass::Critical | JobClass::Normal => 1.0,
            JobClass::Bulk => 0.1,
        }
    }
}

/// Ends a running multistep quantum early (`GpuEngine::multi_step_sampled_at_most`'s `cut`) when
/// a co-tenant with more urgent work starts waiting for the device (`--co-sched deadline`): an
/// 8-step LLM quantum is 50-100 ms of an ASR final's budget. (Cutting on a new arrival as well
/// measured nothing at c1-c32: adaptive multistep already runs single steps while one is likely.)
#[cfg(feature = "cuda")]
struct QuantumCut {
    turn: Option<(Arc<crate::serve::cosched::DeviceTurn>, crate::serve::cosched::Due)>,
}
#[cfg(not(feature = "cuda"))]
struct QuantumCut;

#[cfg(feature = "cuda")]
impl QuantumCut {
    fn fire(&self) -> bool {
        self.turn.as_ref().is_some_and(|(dt, mine)| dt.outranked_due(*mine))
    }
}

/// How long a critical job keeps deadline urgency: a speech stream past it is sustained by
/// throughput, not by jumping co-tenants for every token.
const CRITICAL_SPAN: std::time::Duration = std::time::Duration::from_millis(1000);

/// Tokens a critical job keeps deadline urgency for however long it has waited: a speech stream's
/// first chunks. Past both this and [`CRITICAL_SPAN`] it is throughput.
const CRITICAL_TOKENS: usize = 48;

/// The class of one job's work, for [`crate::serve::cosched::CoSched::Deadline`] outside a voice
/// turn: an ASR final first; then a prompt still owed its first token, or a speech stream's start,
/// is a deadline; decode is throughput; bulk work (partial transcripts) waits.
fn job_urgency(class: JobClass, step: usize, arrived: Instant, now: Instant) -> crate::serve::cosched::Urgency {
    use crate::serve::cosched::Urgency;
    match class {
        JobClass::Bulk => Urgency::Bulk,
        JobClass::Final => Urgency::Final,
        _ if step == 0 => Urgency::Deadline,
        JobClass::Critical if step < CRITICAL_TOKENS || now.saturating_duration_since(arrived) < CRITICAL_SPAN => {
            Urgency::Deadline
        }
        _ => Urgency::Normal,
    }
}

/// When this model's next tick is due on the device ([`crate::serve::cosched::CoSched::Deadline`]):
/// the best-ranked job ([`crate::serve::cosched::Due::rank`]). Jobs outside a voice turn keep the class deadline (the most urgent class,
/// anchored at the oldest first output or ASR final of that class; ongoing work from now). Turn
/// jobs take their stage deadline ([`crate::serve::deadlines`]) from the slot's turn times
/// ([`refresh_turns`]); a queued turn job looks its turn up, the first [`QUEUED_TURN_LOOKUPS`] only (the
/// queue head is all the next tick can admit). Costs come from [`crate::sched::cost`] for
/// `model`, this model's cost-model id.
fn tick_due(
    slots: &[Option<Slot>],
    waiting: &std::collections::VecDeque<(Job, Instant)>,
    model: usize,
    now: Instant,
) -> crate::serve::cosched::Due {
    use crate::sched::cost::{estimate_id, Op};
    use crate::serve::cosched::{Due, Urgency};
    let width = slots.iter().flatten().count();
    let tick = estimate_id(model, Op::DecodeTick { width });
    let legacy = || {
        let live = slots.iter().flatten().filter(|s| s.turn.is_none()).map(|s| (s.class, s.step, s.arrived));
        live.chain(waiting.iter().filter(|(j, _)| j.opts.turn.is_none()).map(|(j, _)| (j.opts.class, 0, j.arrived)))
    };
    let mut best = legacy().map(|(class, step, arrived)| job_urgency(class, step, arrived, now)).min().map(|u| {
        let since = legacy()
            .filter(|&(class, step, arrived)| (step == 0 || class == JobClass::Final) && job_urgency(class, step, arrived, now) == u)
            .map(|(_, _, arrived)| arrived)
            .fold(now, Instant::min);
        Due { cost: tick, ..Due::from_urgency(u, since) }
    });
    let mut ests = None;
    let raw_first = |raw_tokens: bool, speech: Option<&SpeechJob>| {
        raw_tokens.then(|| speech.map_or(0, |sp| sp.first_tokens)).map(|n| if n == 0 { CRITICAL_TOKENS } else { n })
    };
    let mut tighter = |class: JobClass, raw_first: Option<usize>, step: usize, rows: usize, arrived: Instant, t: &crate::serve::turns::TurnTimes| {
        let e = ests.get_or_insert_with(crate::serve::deadlines::ests);
        let prefill = if rows > 0 { estimate_id(model, Op::Prefill { rows }) } else { std::time::Duration::ZERO };
        let d = turn_job_due(model, class, raw_first, step, prefill, tick, arrived, t, e, now);
        if best.is_none_or(|b| d.rank(now) < b.rank(now)) {
            best = Some(d);
        }
    };
    for s in slots.iter().flatten() {
        if let Some(t) = &s.turn {
            let rows = if s.step == 0 { s.prompt_ids.len().saturating_sub(s.pf_pos) } else { 0 };
            tighter(s.class, raw_first(s.raw_tokens, s.speech.as_deref()), s.step, rows, s.arrived, t);
        }
    }
    for (j, _) in waiting.iter().filter(|(j, _)| j.opts.turn.is_some()).take(QUEUED_TURN_LOOKUPS) {
        if let Some(t) = j.opts.turn.as_ref().and_then(|k| crate::serve::turns::table().times(k)) {
            tighter(j.opts.class, raw_first(j.opts.raw_tokens, j.opts.speech.as_deref()), 0, j.prompt_ids.len(), j.arrived, &t);
        }
    }
    best.unwrap_or_else(|| Due { cost: tick, ..Due::from_urgency(Urgency::Bulk, now) })
}

/// Re-read the turn times of slots whose turn has no first audio yet: later stages (LLM first
/// token, TTS first audio) land in the turn table after admission.
fn refresh_turns(slots: &mut [Option<Slot>]) {
    for s in slots.iter_mut().flatten() {
        if s.turn.is_some_and(|t| t.tts_first_audio.is_none()) {
            if let Some(t) = s.turn_key.as_ref().and_then(|k| crate::serve::turns::table().times(k)) {
                s.turn = Some(t);
            }
        }
    }
}

/// Queued turn jobs [`tick_due`] looks up per tick.
const QUEUED_TURN_LOOKUPS: usize = 2;

/// One turn job's stage and `Due`, without model names: an ASR final or partial by class, a
/// speech LM (raw-token stream, `raw_first`: tokens to its first audio) by its first chunk, else
/// an LLM prompt or its decode.
#[allow(clippy::too_many_arguments)]
fn turn_job_due(
    model: usize,
    class: JobClass,
    raw_first: Option<usize>,
    step: usize,
    prefill: std::time::Duration,
    tick: std::time::Duration,
    arrived: Instant,
    t: &crate::serve::turns::TurnTimes,
    e: &crate::serve::deadlines::Ests,
    now: Instant,
) -> crate::serve::cosched::Due {
    use crate::serve::deadlines::{claim_stage, partial, stage_cost, stage_due};
    use crate::serve::turns::Stage;
    let (stage, since) = match class {
        JobClass::Bulk => return partial(arrived, stage_cost(Stage::AsrFinal, prefill, tick, e)),
        JobClass::Final => (Stage::AsrFinal, arrived),
        _ if raw_first.is_some_and(|n| step < n) => (Stage::TtsFirst, arrived),
        _ if raw_first.is_some() => (Stage::TtsStream, now),
        _ if step == 0 => (Stage::LlmFirst, arrived),
        _ => (Stage::LlmDecode, now),
    };
    claim_stage(stage, model);
    stage_due(stage, t, e, stage_cost(stage, prefill, tick, e), since)
}

/// How often a dispatcher re-reads its engine's KV admission budget from device memory.
#[cfg(feature = "cuda")]
const KV_BUDGET_REFRESH: std::time::Duration = std::time::Duration::from_millis(50);

/// How long a consumer that stopped reading may hold its slot parked before the request is cut.
const PARK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

type HeldToken = (u32, String, Option<Box<crate::text::logprobs::TokenLogprobs>>);

/// A request whose prefill splices host rows over prompt positions (`EmbedOverlayBf16`: the
/// packet's `in.encoder_overlay` / `in.encoder_overlay_index`) and whose decode embedding may
/// offset a learned position by a per-slot base (`EmbedPosBf16`: `in.pos_base`).
///
/// Both packet ops index their inputs by LAUNCH row, and the overlay is one global tensor: every
/// prefill launch rewrites it for the rows it carries (serial: one chunk; packed: every member's
/// rows at their launch offsets), and decode runs the plain launch (launch row == slot).
///
/// Packed prefill runs `prompt_ids[..n - 1]` and embeds the last prompt row through the decode
/// program (`EmbedPosBf16` at `pos_base`), so when an overlay covers that row, `prompt_ids[n - 1]`
/// must be the token whose decode embedding the overlay row is.
#[derive(Default)]
pub struct SpeechJob {
    /// `[rows][hidden]` f32, one row per entry of `overlay_pos`.
    pub overlay: Vec<f32>,
    /// Absolute prompt position each overlay row replaces, strictly increasing.
    pub overlay_pos: Vec<u32>,
    /// Written to `in.pos_base[slot]` once the prompt is prefilled (decode position base).
    pub pos_base: Option<u32>,
    /// Classifier-free guidance: the request also runs unconditionally on the partner slot
    /// (owner + 1), and each token is drawn from the combined logits on the host.
    pub cfg: Option<CfgJob>,
    /// A voice turn's stream: tokens its first audio renders from. Its ticks are the turn's first
    /// audio until then and stream decode after, so the first render outranks them (0: unknown).
    pub first_tokens: usize,
}

/// The unconditional member of a CFG pair and the guided sampling chain.
pub struct CfgJob {
    /// The unconditional member's prefill rows, at `SpeechJob::overlay_pos`.
    pub uncond_overlay: Vec<f32>,
    pub params: crate::text::sample::CfgParams,
    /// Penalty history ahead of the generated tokens (the reference counts its BOS).
    pub history: Vec<u32>,
    /// `None` = greedy guided decoding.
    pub seed: Option<u64>,
}

/// A CFG owner's host state: its draws and reused logits buffers.
struct CfgRun {
    rng: Option<crate::text::sample::SplitMix>,
    /// The partner's prefill frontier on the packed route (the owner's is `Slot::pf_pos`).
    partner_pf: usize,
    cond: Vec<f32>,
    uncond: Vec<f32>,
    scratch: Vec<f32>,
    /// The engine holds this pair's penalty history: its draws run on the device.
    dev_seeded: bool,
}

/// Handle to a per-model dispatcher — cheap to clone (wraps a Sender).
#[derive(Clone)]
pub struct ModelMux {
    tx: mpsc::Sender<MuxMsg>,
    metrics: Arc<Metrics>,
    /// Preempt request flag, checked by the dispatcher at every loop top —
    /// the ONLY signal that reaches it while a full slot table keeps it away
    /// from the message channel (see [`ModelMux::preempt`]).
    preempt: Arc<std::sync::atomic::AtomicBool>,
    preempt_notify: Arc<tokio::sync::Notify>,
    /// A non-bulk job was sent: a dispatcher waiting for its device turn (`--co-sched deadline`)
    /// re-queues with its new most urgent work instead of the `Due` it queued with.
    arrival_notify: Arc<tokio::sync::Notify>,
    /// Requests past model lookup whose job is not on `tx` yet (still tokenizing).
    ingress: Arc<Ingress>,
    /// Set by [`ModelMux::preempt`] and never cleared: held work submitting later fails retryably.
    preempted: Arc<std::sync::atomic::AtomicBool>,
}

/// What a request submitting to a preempted mux is told (503-class).
pub const PREEMPTED: &str = "model preempted — retry";

/// Requests past model lookup; the last to leave wakes a gracefully draining dispatcher.
#[derive(Default)]
struct Ingress {
    count: std::sync::atomic::AtomicUsize,
    idle: tokio::sync::Notify,
}

impl Ingress {
    fn enter(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    fn leave(&self) {
        if self.count.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.idle.notify_one();
        }
    }

    fn pending(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }
}

/// Holds one count in [`ModelMux::ingress`] until dropped.
pub struct IngressGuard<'a>(&'a Ingress);

/// [`IngressGuard`] that outlives the borrow: a request that works before it submits (an ASR
/// encode) holds it, so a graceful drain waits for its job instead of closing on it. A preempt
/// does not wait for it; the job's later submission fails with [`PREEMPTED`].
pub struct OwnedIngress(Arc<Ingress>);

impl Drop for OwnedIngress {
    fn drop(&mut self) {
        self.0.leave();
    }
}

impl Drop for IngressGuard<'_> {
    fn drop(&mut self) {
        self.0.leave();
    }
}

/// Internal messages to the dispatcher: jobs or control signals.
enum MuxMsg {
    Job(Job, Instant),
    /// Graceful drain: stop admitting new requests, finish in-flight slots,
    /// then signal completion through the oneshot.
    Drain(tokio::sync::oneshot::Sender<()>),
}

pub enum SubmitError {
    Full(Job),
    Closed(Job),
}

/// Per-dispatcher engine health, driven by the device faults a tick surfaces.
///
/// Only [`crate::DeviceErrorInfo`] faults move this — the CPU reference path
/// never produces one, so it can never misfire there — and only `Dead` gates
/// anything: a fatal fault means the device context is poisoned for good, so
/// the dispatcher fails its live slots once and rejects every later arrival
/// up front (a fatal `DeviceFault`, mapping to 503) instead of dispatching
/// into the dead context and flooding the log. `Degraded` counts consecutive
/// non-fatal faulted ticks; a clean tick resets it.
enum EngineHealth {
    Healthy,
    Degraded { consecutive_failures: u32 },
    Dead(crate::DeviceErrorInfo),
}

/// Pure transition function (free-standing for tests): `Dead` is terminal,
/// a fatal fault is `Dead`, a non-fatal fault bumps `Degraded`, and a clean
/// tick resets to `Healthy`.
fn advance_health(health: EngineHealth, fault: Option<crate::DeviceErrorInfo>) -> EngineHealth {
    if let EngineHealth::Dead(_) = health {
        return health;
    }
    match fault {
        Some(f) if f.fatal => EngineHealth::Dead(f),
        Some(_) => EngineHealth::Degraded {
            consecutive_failures: match health {
                EngineHealth::Degraded {
                    consecutive_failures,
                } => consecutive_failures + 1,
                _ => 1,
            },
        },
        None => EngineHealth::Healthy,
    }
}

/// Record the first device fault a tick sees (per-slot errors keep flowing to
/// their waiters; this is only the dispatcher's health signal).
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
fn note_fault(tick_fault: &mut Option<crate::DeviceErrorInfo>, err: &crate::RuntimeError) {
    if tick_fault.is_none() {
        *tick_fault = err.device_fault().cloned();
    }
}

/// Per-slot copy of a batch error for fan-out to every affected waiter: a
/// typed device fault stays typed (its fatality drives the 503 mapping);
/// anything else degrades to the stringified `Msg` as before.
#[allow(dead_code)]
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu", test))]
fn fanout_err(err: &crate::RuntimeError, msg: &str) -> crate::RuntimeError {
    match err.device_fault() {
        Some(info) => crate::RuntimeError::DeviceFault { info: info.clone() },
        None => crate::RuntimeError::Msg(msg.to_string()),
    }
}

#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu", test))]
fn decode_feed_extent(feeds: &[(usize, u32)]) -> Option<usize> {
    feeds.iter().map(|&(slot, _)| slot + 1).max()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DecodeProgress {
    extent: usize,
    steps: std::num::NonZeroUsize,
}

#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu", test))]
fn completed_decode(feeds: &[(usize, u32)], steps: usize) -> Option<DecodeProgress> {
    Some(DecodeProgress {
        extent: decode_feed_extent(feeds)?,
        steps: std::num::NonZeroUsize::new(steps)?,
    })
}

impl ModelMux {
    /// Whether any request is queued, tokenizing or holding an engine slot.
    pub fn in_flight(&self) -> bool {
        self.ingress.pending() > 0
            || self.metrics.queued_requests.load(Ordering::Relaxed) > 0
            || self.metrics.slots_active.load(Ordering::Relaxed) > 0
    }

    pub fn ingress_owned(&self) -> OwnedIngress {
        self.ingress.enter();
        OwnedIngress(Arc::clone(&self.ingress))
    }

    /// Count a request as pending before it tokenizes; hand the guard to `submit_arrived`.
    pub fn ingress(&self) -> IngressGuard<'_> {
        self.ingress.enter();
        IngressGuard(&self.ingress)
    }

    /// [`Self::preempt`] has run: this dispatcher is gone or going, and serves nothing more.
    pub fn preempted(&self) -> bool {
        self.preempted.load(Ordering::Acquire)
    }

    /// The dispatcher has exited (a drain finished).
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }

    /// Requests submitted and not yet in an engine slot.
    pub fn pending(&self) -> usize {
        self.metrics.queued_requests.load(Ordering::Relaxed) as usize
    }

    /// Submit a job. Returns immediately; the caller awaits the stream.
    pub fn submit(&self, job: Job) -> std::result::Result<(), SubmitError> {
        let arrived = job.arrived;
        self.submit_arrived(job, arrived, None)
    }

    pub fn submit_arrived(
        &self,
        job: Job,
        arrived: Instant,
        ingress: Option<IngressGuard<'_>>,
    ) -> std::result::Result<(), SubmitError> {
        Metrics::inc(&self.metrics.requests);
        self.metrics.serving.max_tokens.tokens(job.gen.max_tokens);
        Metrics::inc(&self.metrics.queued_requests);
        // Released before the send: a dispatcher that dequeues this job must not see it as a peer.
        drop(ingress);
        let urgent = job.opts.class != JobClass::Bulk;
        match self.tx.try_send(MuxMsg::Job(job, arrived)) {
            Ok(()) => {
                if urgent {
                    self.arrival_notify.notify_one();
                }
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(MuxMsg::Job(job, _))) => {
                self.metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
                Metrics::inc(&self.metrics.rejected);
                Err(SubmitError::Full(job))
            }
            Err(mpsc::error::TrySendError::Closed(MuxMsg::Job(job, _))) => {
                self.metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
                Metrics::inc(&self.metrics.rejected);
                Err(SubmitError::Closed(job))
            }
            Err(_) => unreachable!(),
        }
    }

    /// As [`Self::submit`], but a full ingress queue waits for room instead of failing: for a front
    /// that bounds its own requests in flight.
    pub async fn submit_wait(&self, job: Job) -> std::result::Result<(), SubmitError> {
        Metrics::inc(&self.metrics.requests);
        self.metrics.serving.max_tokens.tokens(job.gen.max_tokens);
        Metrics::inc(&self.metrics.queued_requests);
        let arrived = job.arrived;
        let urgent = job.opts.class != JobClass::Bulk;
        self.tx.send(MuxMsg::Job(job, arrived)).await.map_err(|mpsc::error::SendError(msg)| {
            self.metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
            Metrics::inc(&self.metrics.rejected);
            match msg {
                MuxMsg::Job(job, _) => SubmitError::Closed(job),
                MuxMsg::Drain(_) => unreachable!("sent a job"),
            }
        })?;
        if urgent {
            self.arrival_notify.notify_one();
        }
        Ok(())
    }

    /// Initiate graceful drain: no new requests accepted, all live slots run to
    /// completion. Returns when every in-flight slot has finished. Use before
    /// `Registry::unload` to avoid mid-generation errors.
    pub async fn drain(&self) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(MuxMsg::Drain(tx)).await;
        let _ = rx.await;
    }

    /// Preemptive drain: close EVERY live stream now — `Done` with
    /// [`FinishReason::Preempted`] and the usage so far — and free the slots,
    /// instead of letting generations run out. Bounds an S1 switch's drain
    /// phase at ~one tick where a graceful [`Self::drain`] is O(max_tokens ×
    /// service_ms) — a 2048-token slot at 40 ms/token is an 82 s wait.
    /// Returns when the dispatcher has exited.
    pub async fn preempt(&self) {
        self.preempted.store(true, Ordering::Release);
        self.preempt.store(true, Ordering::Release);
        self.preempt_notify.notify_one();
        // The Drain message wakes a dispatcher blocked on recv (idle path)
        // and carries the completion signal; the flag is what the tick loop
        // sees when a full slot table keeps it off the channel.
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(MuxMsg::Drain(tx)).await;
        let _ = rx.await;
    }
}

/// One live request occupying a slot in the engine.
struct Slot {
    telemetry: Option<RequestMetrics>,
    prompt_ids: Vec<u32>,
    out_ids: Vec<u32>,
    gen: GenParams,
    respond: ChunkSender,
    /// Incremental-detokenize window (TGI scheme): `prefix_offset..read_offset`
    /// is the last emitted token span; each new token decodes only
    /// `out_ids[prefix_offset..]` (O(window), not O(total)) and streams the
    /// byte delta past the prefix decode.
    prefix_offset: usize,
    read_offset: usize,
    executed: usize,
    step: usize,
    /// GPU-path prefill frontier: prompt tokens consumed so far by
    /// `prefill_chunk`. `step == 0 && pf_pos < prompt_ids.len()` = mid-prefill.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pf_pos: usize,
    /// Prompt tokens the engine served from the prefix cache for this
    /// sequence (GPU path, `PLOW_VMM_PREFIX=1`); 0 = cold. Reported as
    /// OpenAI `usage.prompt_tokens_details.cached_tokens`.
    cached_tokens: usize,
    /// KV arena handle held for this slot's lifetime. `None` when the bundle
    /// has no `KvPaging` (test bundles / models without attention).
    kv: Option<SlotHandle>,
    /// When the request was handed to the dispatcher ([`Job::arrived`]). Read
    /// only by the §TTFT breakdown (`PLOW_TTFT_LOG=1`), to charge the interval
    /// between submit and the tick that actually prefills it — which today only
    /// the gfx950 arm does.
    #[cfg_attr(not(feature = "hsa"), allow(dead_code))]
    arrived: Instant,
    /// Tail of the text streamed so far, for OpenAI `stop` matching. Only the
    /// last `max(stop)` bytes are kept — a stop sequence can straddle token
    /// boundaries, so matching per-delta would miss it, and keeping the whole
    /// answer to scan it would be O(n^2) over a long generation.
    stop_tail: String,
    /// Bytes withheld from the client because they are a proper prefix of a stop string and
    /// the rest of it may still be generated. Released once a later token proves otherwise.
    stop_pending: String,
    class: JobClass,
    raw_tokens: bool,
    /// The voice turn's times ([`tick_due`]), refreshed by [`refresh_turns`] until its first audio.
    turn: Option<crate::serve::turns::TurnTimes>,
    turn_key: Option<crate::serve::turns::TurnKey>,
    speech: Option<Box<SpeechJob>>,
    /// Set on the owner of a CFG pair; its partner slot (owner + 1) stays `None` in the table
    /// and is reserved while the owner lives (see [`slot_free`]).
    cfg: Option<Box<CfgRun>>,
    /// Chunks produced while the consumer's channel was full, oldest first. A parked slot is
    /// not fed; see [`flush_parked`].
    held: Vec<HeldToken>,
    held_finish: Option<FinishReason>,
    parked_at: Option<Instant>,
    /// A session request's retention seat; dropped with the slot, it retires the rows.
    session: Option<Seat>,
    /// Retained cache rows this sequence starts from (the engine keeps them at `begin_slot`).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    resume: usize,
    /// The next token's OpenAI logprobs, set by the sampler when the request asked for them.
    lp: Option<Box<crate::text::logprobs::TokenLogprobs>>,
    /// [`JobOpts::prefix`], consumed by the prefix-cache attach.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    prefix: Option<crate::memory::vmm::PrefixKey>,
    /// [`JobOpts::mm`]: staged into the LM's slab before any launch reads the prompt; its rows
    /// are released when the slot is dropped.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    mm: Option<Box<crate::serve::mm::MmJob>>,
}

impl Slot {
    /// Rows the mixed-step program cannot carry: it does not stage overlays (see [`SpeechJob`]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn serial_prefill(&self) -> bool {
        self.speech.is_some()
    }

    /// Rows that take the plain one-step decode launch: `EmbedPosBf16` reads the position base by
    /// launch row, and speech outputs are short and stop unpredictably, so a device multi-step
    /// quantum mostly decodes past the stop (Qwen3-ASR served, 73 clips, C16: 246.7 vs 227.3 RTFx
    /// one-step vs K=8).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn plain_decode(&self) -> bool {
        self.speech.is_some()
    }
}

/// Buffers reused across ticks for one bucket. Reallocated only when the live
/// shape changes buckets (cold path — new arrival straddles a ladder rung).
struct BucketBufs {
    key: BucketKey,
    pool: CounterPool,
    streams: StreamSet,
    vocab: usize,
}

/// KV allocator plus the physical allocation its pool bases point into.
/// Keeping both under the mux-owned `Arc` prevents the address space from
/// freeing device memory while live `KvArena` handles still reference it.
struct KvState {
    arena: KvArena,
    /// `None` only for the existing zero-base fallback after allocation failure.
    _addr_space: Option<AddressSpace>,
}

type SharedKvState = Arc<Mutex<KvState>>;

/// Spawn the dispatcher for `slug` and return the mux handle. The task lives
/// for the lifetime of the process (dropping every `ModelMux` clone closes
/// the channel and shuts the task down cleanly).
pub fn spawn(
    slug: String,
    bundle: Arc<ModelBundle>,
    state: Arc<AppState>,
    cfg: MuxConfig,
) -> ModelMux {
    let preempt_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let preempt_seen = Arc::clone(&preempt_flag);
    let preempt_notify = Arc::new(tokio::sync::Notify::new());
    let preempt_wake = Arc::clone(&preempt_notify);
    let arrival_notify = Arc::new(tokio::sync::Notify::new());
    let arrival_wake = Arc::clone(&arrival_notify);
    let ingress = Arc::new(Ingress::default());
    let ingress_seen = Arc::clone(&ingress);
    let metrics = state.model_metrics(&slug);
    let handle_metrics = Arc::clone(&metrics);

    // Slot capacity from the compiler-emitted ladder — the largest decode
    // bucket sets the ceiling for concurrent live requests.
    let capacity = bundle
        .bucket_keys()
        .filter(|k| k.phase == Phase::Decode)
        .map(|k| k.batch.max(1) as usize)
        .max()
        .unwrap_or(1);
    // GPU-engine bundles are bucketless: take both capacity and the optional
    // decode ladder from the loaded engine once, before the hot loop starts.
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    let gpu_shape = state.gpu_engine(&slug).map(|e| {
        let e = e.lock();
        (e.batch(), e.decode_rungs())
    });
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    let capacity = gpu_shape.as_ref().map(|x| x.0).unwrap_or(capacity);
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    let rung_widths = gpu_shape.map(|x| x.1);
    #[cfg(not(any(feature = "cuda", feature = "hsa", feature = "cpu")))]
    let rung_widths: Option<Box<[u32]>> = None;
    let (capacity, rung_widths) = {
        let max_rung = crate::config::RuntimeConfig::get().decode_max_rung;
        let min_rung = crate::config::RuntimeConfig::get().amd.decode_min_rung;
        if max_rung.is_some() || min_rung.is_some() {
            if let Some(widths) = rung_widths {
                let filtered: Box<[u32]> = widths
                    .iter()
                    .copied()
                    .filter(|&w| max_rung.map_or(true, |max| w <= max) && min_rung.map_or(true, |min| w >= min))
                    .collect();
                if let Some(&widest) = filtered.last() {
                    (capacity.min(widest as usize), Some(filtered))
                } else {
                    (capacity, Some(widths))
                }
            } else {
                let cap = max_rung.map_or(capacity, |m| capacity.min(m as usize));
                (cap, None)
            }
        } else {
            (capacity, rung_widths)
        }
    };
    let mut rung_controller =
        rung_widths
            .as_deref()
            .and_then(|widths| match DecodeRungs::new(widths, capacity) {
                Ok(rungs) if rungs.len() > 1 => Some(
                    RungController::new(rungs)
                        .with_fast_probe(crate::serve::policy::fast_probe()),
                ),
                Ok(_) => None,
                Err(err) => {
                    tracing::warn!(?err, ?widths, capacity, "decode rung policy disabled");
                    None
                }
            });
    let ingress_capacity = if cfg.max_queued_requests == 0 {
        capacity.saturating_mul(4).max(1)
    } else {
        cfg.max_queued_requests
    };
    let (tx, mut rx) = mpsc::channel::<MuxMsg>(ingress_capacity);
    tracing::info!(%slug, capacity, ingress_capacity, "mux capacity resolved");
    metrics.slots_capacity.store(capacity as u64, Ordering::Relaxed);

    // WHAT THE DEVICE CAN BACK, taken once like `gpu_shape` above. Admission used free SLOTS
    // alone, which is right at 8k prompts and fatal at 70k: 20 x 70,000 rows wants 72.1 GiB of
    // a 55.59 GiB budget, and the overcommit arrived as an async queue fault instead of as
    // backpressure. `None` keeps slot-count admission for every engine without a budget.
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    let mut kv_budget = state
        .gpu_engine(&slug)
        .and_then(|e| e.lock().kv_admission_budget());
    #[cfg(not(any(feature = "cuda", feature = "hsa", feature = "cpu")))]
    let kv_budget: Option<crate::sched::admission::KvBudget> = None;
    #[cfg(feature = "cuda")]
    let pf_rungs = state.gpu_engine(&slug).and_then(|e| e.lock().pf_launch_rungs());
    #[cfg(feature = "cuda")]
    let prefix_probe = state.gpu_engine(&slug).and_then(|e| e.lock().vmm_prefix_probe());
    #[cfg(not(feature = "cuda"))]
    let prefix_probe: Option<crate::memory::vmm::PrefixProbe> = None;
    #[cfg(feature = "cuda")]
    let (resume_supported, prefix_cache, kv_row_bytes) = state.gpu_engine(&slug).map_or((false, false, 0), |e| {
        let e = e.lock();
        (e.slot_resume_supported(), e.is_cuda() && e.prefix_cache_enabled(), e.kv_row_bytes())
    });
    #[cfg(all(not(feature = "cuda"), feature = "cpu"))]
    let (resume_supported, prefix_cache, kv_row_bytes) =
        (state.gpu_engine(&slug).is_some_and(|e| e.lock().slot_resume_supported()), false, 0);
    #[cfg(not(any(feature = "cuda", feature = "cpu")))]
    let (resume_supported, prefix_cache, kv_row_bytes) = (false, false, 0);
    if let Some(b) = kv_budget {
        tracing::info!(
            %slug,
            bytes_per_token = b.bytes_per_token,
            budget_gib = b.budget_bytes as f64 / (1u64 << 30) as f64,
            max_rows = b.max_rows(),
            "mux: KV-capacity admission armed"
        );
    }
    // Jobs the KV budget could not back yet. Retried in arrival order ahead of anything
    // newer, so the budget never reorders the queue.
    //
    // NOT a single blocking slot. A large request at the head must not idle slots that a
    // smaller one behind it would fill: with no preemption path, a held request frees nothing
    // by waiting, so refusing to look past it converts "this one does not fit" into "nothing
    // runs". vLLM tolerates the same head-of-line stall only because it preempts a running
    // request to make room; until plow does, backfilling is what keeps the batch full.
    // The internal deque is capped at `ingress_capacity`; once full, new arrivals stay in the
    // bounded channel until a waiter is admitted. A request that can never fit is answered, not
    // parked (see `admit_into`).
    let mut waiting: std::collections::VecDeque<(Job, Instant)> = std::collections::VecDeque::new();

    // Per-model KV arena from the first decode bucket that declares paging
    // (all rungs share the same paging shape). `None` when no bucket carries
    // KvPaging (test bundles / attention-less models) — admission then skips
    // KV allocation and mux behaves like phase 2.
    //
    // The per-layer physical bases come from an `AddressSpace` built off the
    // same bucket's memory map: `AddressSpace::kv_layer_bases` walks each
    // `KvLayerPaging::buffer_name`, looks up the compiled `MemEntry`, and
    // resolves its `phys_addr`. This is the compiler → runtime seam: plowc
    // emitted the offsets, the runtime honors them.
    let kv_config = bundle.bucket_keys().find_map(|k| {
        let b = bundle.bucket(k)?;
        let paging = b.map.kv_paging.clone()?;
        Some((b, paging))
    });
    let n_layers = kv_config
        .as_ref()
        .map(|(_, paging)| paging.per_layer.len())
        .unwrap_or(0);
    let kv_pages_range = ind_slots::kv_pages(n_layers, capacity);
    let indirection_size = ind_slots::table_size(n_layers, capacity);

    let arena: Option<SharedKvState> = kv_config.and_then(|(bucket, paging)| {
        match AddressSpace::allocate(Arc::clone(state.execset.backend()), bucket.map.clone()) {
            Ok(addr) => {
                let bases = addr.kv_layer_bases(&paging);
                Some(Arc::new(Mutex::new(KvState {
                    arena: KvArena::new(paging, &bases),
                    _addr_space: Some(addr),
                })))
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "kv arena: could not allocate AddressSpace; falling back to zero bases"
                );
                let bases: Vec<u64> = paging.per_layer.iter().map(|_| 0u64).collect();
                Some(Arc::new(Mutex::new(KvState {
                    arena: KvArena::new(paging, &bases),
                    _addr_space: None,
                })))
            }
        }
    });

    // A GPU model's dispatcher gets its own OS thread and runs each tick inline, so a tick costs
    // no engine-thread wake and no tokio-worker wake on return (26B-A4B H100: handoff 50-87 ->
    // 0.3 us per tick). Nothing else changes: the loop below already waits for every tick before
    // touching the queue.
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    let inline_tick = state.gpu_engine(&slug).is_some_and(|engine| engine.lock().is_gpu());
    #[cfg(not(any(feature = "cuda", feature = "hsa", feature = "cpu")))]
    let inline_tick = false;
    let dispatcher_name = format!("plow-mux-{slug}");
    let dp_cpus = state
        .dp_rank(&slug)
        .map(|(set, r)| set.ranks[r].cpus.clone())
        .filter(|cpus| !cpus.is_empty() && crate::config::RuntimeConfig::get().dp_numa_pin);
    let downstream = state.downstream(&slug);

    let dispatcher = async move {
        let mut slots: Vec<Option<Slot>> = (0..capacity).map(|_| None).collect();
        let mut retention = Retention::new(
            resume_supported,
            prefix_cache,
            kv_budget.map_or(kv_row_bytes, |b| b.bytes_per_token),
            rung_widths.as_deref().unwrap_or(&[]),
        );
        let mut freed_last_tick = false;
        let mut load = LoadEstimator::default();
        let mut host_window = crate::obs::host::Window::default();
        let mut host_last_return: Option<Instant> = None;
        // Cache one BucketBufs per BucketKey — rung swaps are a swap-in, not
        // a rebuild. The dispatcher owns the map; each tick takes the entry
        // out, hands it to the tick thread, and puts it back on return.
        let mut bufs_cache: FxHashMap<BucketKey, BucketBufs> = FxHashMap::default();
        // Lazily built and round-tripped through each tick; `None` only before
        // the first tick (and after a tick panic) so the hot path never
        // allocates a placeholder observer.
        let mut obs: Option<RunObserver> = None;
        // Engine health: Dead (fatal device fault) rejects every new arrival
        // at admission; anything else changes nothing on the tick path.
        let mut health = EngineHealth::Healthy;
        // Drain protocol state: once set, stop admitting new requests and
        // complete in-flight slots, then signal the oneshot.
        let mut draining = false;
        let mut drain_done: Option<tokio::sync::oneshot::Sender<()>> = None;
        // Per-model OOB channel: executor→host feedback (faults, checkpoints,
        // speculative verdicts). Drained after each tick.
        let oob = Arc::new(OobChannel::default());
        let mut oob_events: Vec<OobMsg> = Vec::new();
        // GPU-engine models never run the CPU bucket walk — skip the ladder
        // scan + bufs machinery on the dispatcher critical path entirely.
        #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
        let has_gpu = state.gpu_engine(&slug).is_some();
        // Whose turn it is on this model's device. `None` when no turns were
        // installed, and a no-op under `--co-sched free` (the default), where
        // whoever is ready goes first and nothing here orders anything.
        //
        // Not vendor-gated. Taking turns is host-side sequencing of ticks, and
        // every backend that can hold two models on one device wants it — AMD
        // most of all, since HSA has no cooperative-launch refusal to turn CU
        // oversubscription into an error instead of a hang.
        let device_turn = state.device_turn(&slug);
        // Held across ticks so a quantum can span them; dropped whenever this
        // model parks, so an idle model never sits on a device its co-tenant is
        // waiting for.
        let mut turn = crate::serve::cosched::Turn::default();
        let cost_id = crate::sched::cost::id(&slug);
        #[cfg(not(any(feature = "cuda", feature = "hsa", feature = "cpu")))]
        let has_gpu = false;
        // Dedicated engine/submission thread for GPU models: every tick runs
        // on ONE persistent OS thread (CUDA context bound once, no
        // blocking-pool dispatch). CPU-reference models keep spawn_blocking.
        // An inline dispatcher already is that thread.
        let engine_thread = (has_gpu && !inline_tick)
            .then(|| crate::exec::engine_thread::EngineThread::spawn(format!("plow-eng-{slug}")));
        let token_group = bundle.decode_token_group();
        #[cfg(feature = "cuda")]
        let (mut kv_refreshed, mut kv_pressure_seen) = (Instant::now(), (0u64, 0u64));
        let (mut preempted, mut late_drains) = (false, Vec::new());

        loop {
            // Preempt ([`ModelMux::preempt`]): kill every live slot NOW.
            // Checked at every loop top because a full slot table never
            // reaches the message channel between ticks — this flag is the
            // one bounded-latency path in.
            retention.collect(Instant::now());
            retention.publish(&metrics);
            if preempt_seen.swap(false, Ordering::AcqRel) {
                turn.release();
                preempt_slots(&mut slots, &arena).await;
                retention.table.clear();
                while let Some((job, _)) = waiting.pop_front() {
                    metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
                    Metrics::inc(&metrics.rejected);
                    let _ = job.respond.try_send(StreamChunk::Err(crate::RuntimeError::Rejected(
                        "model preempted for an S1 switch — retry".into(),
                    )));
                }
                draining = true;
                preempted = true;
            }
            let live = slots.iter().filter(|s| s.is_some()).count();
            metrics.slots_active.store(live as u64, Ordering::Relaxed);
            // The budget follows device memory (other models load, grow and leave); memory
            // pressure asks the planner to evict an idle co-tenant so this model can grow.
            #[cfg(feature = "cuda")]
            if kv_budget.is_some() && kv_refreshed.elapsed() >= KV_BUDGET_REFRESH {
                kv_refreshed = Instant::now();
                let pressure = state.gpu_engine(&slug).map(|e| {
                    let e = e.lock();
                    kv_budget = e.kv_admission_budget().or(kv_budget);
                    e.kv_pressure_events()
                });
                let seen = (metrics.kv_budget_denials.load(Ordering::Relaxed), pressure.unwrap_or(0));
                if seen != kv_pressure_seen {
                    kv_pressure_seen = seen;
                    tracing::debug!(%slug, denials = seen.0, oom_waits = seen.1, "mux: KV memory short — asking to grow");
                    if let Some(mgr) = state.manager_for(&slug) {
                        mgr.request_growth(&slug);
                    }
                }
            }

            // Drain completion: if draining and no in-flight slots remain,
            // signal the drain future and exit the dispatcher loop. A graceful drain waits for a
            // request past model lookup (`ingress`): closing on it would fail it. A preempt does not
            // (an idle stream can hold one indefinitely); its later submission fails retryably.
            let drained = draining && live == 0 && waiting.is_empty() && (preempted || ingress_seen.pending() == 0);
            // A graceful drain serves what was submitted before the mux left the routing table,
            // however late its job reached the channel.
            if drained && !preempted {
                while let Ok(msg) = rx.try_recv() {
                    note_dequeued(&msg, &metrics);
                    match msg {
                        MuxMsg::Drain(done) => late_drains.push(done),
                        MuxMsg::Job(job, arrived) => {
                            Metrics::inc(&metrics.queued_requests);
                            waiting.push_back((job, arrived));
                        }
                    }
                }
            }
            if drained && waiting.is_empty() {
                turn.release();
                for done in drain_done.take().into_iter().chain(late_drains.drain(..)) {
                    let _ = done.send(());
                }
                // A preempt's completion oneshot rides a Drain message that
                // may still be queued (the flag outran the channel) — answer
                // every pending one before exiting. Queued JOBS get an
                // explicit Err, not a silent drop: the preempt flag bypasses
                // channel order, so unlike a message-initiated drain these
                // jobs never had their chance to be dequeued and admitted,
                // and a stream that ends with no terminal chunk is
                // indistinguishable from a crash (the shed path at the
                // admission gate sets the precedent).
                while let Ok(msg) = rx.try_recv() {
                    note_dequeued(&msg, &metrics);
                    match msg {
                        MuxMsg::Drain(done) => {
                            let _ = done.send(());
                        }
                        MuxMsg::Job(job, _) => {
                            Metrics::inc(&metrics.rejected);
                            let _ = job.respond.try_send(StreamChunk::Err(
                                crate::RuntimeError::Rejected(
                                    "model preempted for an S1 switch — retry".into(),
                                ),
                            ));
                        }
                    }
                }
                break;
            }

            // Cold start: no live slots — block until an arrival (or exit
            // when every ModelMux clone has dropped and the channel closes).
            if live == 0 && waiting.is_empty() {
                metrics.decode_occupied_extent.store(0, Ordering::Relaxed);
                metrics.decode_rung_actual.store(0, Ordering::Relaxed);
                // Parking with the turn held would starve a co-tenant for as
                // long as this model has nothing to do, which is unbounded.
                turn.release();
                let msg = tokio::select! {
                    msg = rx.recv() => msg,
                    // A graceful drain held only by ingress: the last request leaving without
                    // submitting must wake it, as no message will.
                    _ = ingress_seen.idle.notified(), if draining => continue,
                };
                let Some(msg) = msg else { break };
                note_dequeued(&msg, &metrics);
                match msg {
                    MuxMsg::Job(job, arrived) => {
                        note_arrival(job.arrived, &mut load, &metrics);
                        let held = admit_session(
                            &mut slots,
                            usize::MAX,
                            job,
                            arrived,
                            arena.as_ref(),
                            &metrics,
                            &health,
                            kv_budget,
                            downstream.full(),
                            &mut retention,
                        );
                        if let Some(j) = held {
                            Metrics::inc(&metrics.queued_requests);
                            waiting.push_back(j);
                        }
                    }
                    // No in-flight work: the drain check at the loop top finishes, after serving
                    // any request that was past model lookup when the drain began.
                    MuxMsg::Drain(done) => {
                        draining = true;
                        match drain_done {
                            None => drain_done = Some(done),
                            Some(_) => late_drains.push(done),
                        }
                    }
                }
            }

            // A decode ladder controls ADMISSION separately from execution.
            // New jobs use only the low slot prefix under `admission_limit`;
            // existing high slots are never moved and continue to pin the
            // engine's occupied-extent rung until they drain.
            let admission_limit = if let Some(controller) = rung_controller.as_mut() {
                let occupied_extent = slots
                    .iter()
                    .rposition(Option::is_some)
                    .map(|i| i + 1)
                    .unwrap_or(1);
                // Read the receiver directly for an exact local admission snapshot.
                let queued = rx.len().saturating_add(waiting.len());
                let (sum, n) = slots
                    .iter()
                    .flatten()
                    .fold((0usize, 0usize), |(s, n), slot| {
                        (
                            s.saturating_add(
                                slot.gen
                                    .max_tokens
                                    .saturating_sub(slot.out_ids.len())
                                    .max(1),
                            ),
                            n + 1,
                        )
                    });
                let mean_output_tokens = if n == 0 { 1.0 } else { sum as f64 / n as f64 };
                let before = controller.admission_limit();
                let now = Instant::now();
                let oldest_wait_ms = waiting
                    .front()
                    .map(|(_, arrived)| now.saturating_duration_since(*arrived).as_secs_f64() * 1e3)
                    .unwrap_or(0.0);
                let decision = controller.decide(RungLoad {
                    occupied_extent,
                    queued,
                    oldest_wait_ms,
                    arrival_rps: load.lambda.rate(now),
                    mean_output_tokens,
                    slo_ms: cfg.slo_ms,
                });
                let admission = controller.admission_limit();
                metrics
                    .decode_rung_admission
                    .store(admission as u64, Ordering::Relaxed);
                metrics
                    .decode_occupied_extent
                    .store(occupied_extent as u64, Ordering::Relaxed);
                let kv_used = kv_used(kv_budget, &slots);
                metrics.kv_used_milli.store((kv_used * 1000.0) as u64, Ordering::Relaxed);
                if crate::serve::policy::observe(crate::serve::policy::Load {
                    width: occupied_extent,
                    queued: waiting.len(),
                    kv_used,
                }) {
                    Metrics::inc(&metrics.serve_mode_switches);
                }
                metrics.serve_mode.store(
                    u64::from(crate::serve::policy::class() == crate::serve::policy::Class::HighConcurrency),
                    Ordering::Relaxed,
                );
                if let Some(rc) = rung_controller.as_mut() {
                    rc.set_fast_probe(crate::serve::policy::fast_probe());
                }
                if admission != before {
                    Metrics::inc(&metrics.decode_rung_switches);
                    tracing::info!(
                        from = before,
                        to = admission,
                        occupied_extent,
                        queued,
                        reason = ?decision.reason,
                        "decode admission rung"
                    );
                }
                admission
            } else {
                capacity
            };
            #[cfg(feature = "cuda")]
            if let Some((request_rows, top_rows)) = pf_rungs {
                // Only demand past every slot: a rung admission hold (E4B c128's cold start
                // queued 32K rows behind a 96-slot rung) is not overload.
                let overflow_rows = if slots.iter().all(Option::is_some) {
                    waiting.iter().map(|(job, _)| job.prompt_ids.len()).sum()
                } else {
                    0
                };
                crate::serve::policy::observe_prefill(crate::serve::policy::PrefillLoad {
                    overflow_rows,
                    queued: waiting.len() + rx.len(),
                    top_rows,
                    kv_used: kv_used(kv_budget, &slots),
                });
                metrics
                    .prefill_launch_rows
                    .store(pf_launch_rows(request_rows, top_rows).0 as u64, Ordering::Relaxed);
            }

            let mut queued_behind = false;
            // Non-blocking drain: fill every idle slot the queue can serve.
            // A short hold when we still have empty slots and no live work
            // yet keeps us from waking up on a single arrival amid a burst.
            if !waiting.is_empty() {
                drain_waiting_session(
                    &mut waiting,
                    &mut slots,
                    admission_limit,
                    Instant::now(),
                    cfg.slo_ms,
                    arena.as_ref(),
                    &metrics,
                    &health,
                    kv_budget,
                    downstream.full(),
                    &mut retention,
                    prefix_probe.as_ref(),
                );
            }
            let idle = slots[..admission_limit]
                .iter()
                .filter(|s| s.is_none())
                .count();
            if !draining && waiting.len() < ingress_capacity && idle > 0 {
                let lambda = load.lambda.rate(Instant::now());
                // Only hold when the slot table is empty (cold-start burst);
                // if any slot is already live, spinning up the tick delivers
                // TTFT faster than waiting for more arrivals.
                let hold_ms = if slots.iter().all(Option::is_none) {
                    cold_start_hold_ms(
                        lambda,
                        cfg.max_hold_ms,
                        cfg.idle_dispatch,
                        rx.len() + ingress_seen.pending(),
                    )
                } else {
                    0.0
                };
                if hold_ms > 0.0 {
                    turn.release();
                    Metrics::add(&metrics.hold_ms_sum, hold_ms as u64);
                    Metrics::inc(&metrics.hold_count);
                    let deadline =
                        Instant::now() + std::time::Duration::from_secs_f64(hold_ms / 1000.0);
                    while waiting.len() < ingress_capacity
                        && slots[..admission_limit].iter().any(|s| s.is_none())
                    {
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        match tokio::time::timeout(remaining, rx.recv()).await {
                            Ok(Some(msg)) => {
                                note_dequeued(&msg, &metrics);
                                match msg {
                                    MuxMsg::Job(job, arrived) => {
                                        note_arrival(job.arrived, &mut load, &metrics);
                                        // Behind anything already waiting: `drain_waiting` orders it.
                                        let held = if waiting.is_empty() {
                                            admit_session(
                                                &mut slots,
                                                admission_limit,
                                                job,
                                                arrived,
                                                arena.as_ref(),
                                                &metrics,
                                                &health,
                                                kv_budget,
                                                downstream.full(),
                                                &mut retention,
                                            )
                                        } else {
                                            queued_behind = true;
                                            Some((job, arrived))
                                        };
                                        if let Some(j) = held {
                                            Metrics::inc(&metrics.queued_requests);
                                            waiting.push_back(j);
                                        }
                                        if cfg.idle_dispatch
                                            && rx.is_empty()
                                            && ingress_seen.pending() == 0
                                        {
                                            break;
                                        }
                                    }
                                    MuxMsg::Drain(done) => {
                                        draining = true;
                                        drain_done = Some(done);
                                        break;
                                    }
                                }
                            }
                            Ok(None) => break,
                            Err(_) => break,
                        }
                    }
                }
                // Any additional pending arrivals (no wait).
                while !draining
                    && waiting.len() < ingress_capacity
                    && slots[..admission_limit].iter().any(|s| s.is_none())
                {
                    match rx.try_recv() {
                        Ok(msg) => {
                            note_dequeued(&msg, &metrics);
                            match msg {
                                MuxMsg::Job(job, arrived) => {
                                    note_arrival(job.arrived, &mut load, &metrics);
                                    let held = if waiting.is_empty() {
                                        admit_session(
                                            &mut slots,
                                            admission_limit,
                                            job,
                                            arrived,
                                            arena.as_ref(),
                                            &metrics,
                                            &health,
                                            kv_budget,
                                            downstream.full(),
                                            &mut retention,
                                        )
                                    } else {
                                        queued_behind = true;
                                        Some((job, arrived))
                                    };
                                    if let Some(j) = held {
                                        Metrics::inc(&metrics.queued_requests);
                                        waiting.push_back(j);
                                    }
                                }
                                MuxMsg::Drain(done) => {
                                    draining = true;
                                    drain_done = Some(done);
                                    break;
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
            // An arrival never takes a slot ahead of an older waiter: it joined the queue, which
            // is drained again in its seat order.
            if std::mem::take(&mut queued_behind) {
                drain_waiting_session(
                    &mut waiting,
                    &mut slots,
                    admission_limit,
                    Instant::now(),
                    cfg.slo_ms,
                    arena.as_ref(),
                    &metrics,
                    &health,
                    kv_budget,
                    downstream.full(),
                    &mut retention,
                    prefix_probe.as_ref(),
                );
            }
            // A full slot table leaves arrivals in the channel in arrival order; in `waiting`,
            // `drain_waiting` seats them by class as slots free.
            while !draining
                && waiting.len() < ingress_capacity
                && !rx.is_empty()
                && slots[..admission_limit].iter().all(Option::is_some)
            {
                let Ok(msg) = rx.try_recv() else { break };
                note_dequeued(&msg, &metrics);
                match msg {
                    MuxMsg::Job(job, arrived) => {
                        note_arrival(job.arrived, &mut load, &metrics);
                        Metrics::inc(&metrics.queued_requests);
                        waiting.push_back((job, arrived));
                    }
                    MuxMsg::Drain(done) => {
                        draining = true;
                        drain_done = Some(done);
                    }
                }
            }

            let live = slots.iter().filter(|s| s.is_some()).count();
            if live == 0 {
                // Everything queued waits on the downstream stage: sleep until it finishes work.
                if !waiting.is_empty() && downstream.full() {
                    turn.release();
                    tokio::select! {
                        _ = downstream.released() => {}
                        _ = tokio::time::sleep(std::time::Duration::from_millis(5)) => {}
                    }
                }
                continue;
            }
            // Every live slot is parked on a slow consumer: nothing to launch.
            if slots.iter().flatten().all(|s| s.parked_at.is_some()) {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }

            // Pick the covering bucket for (Decode, live, max seq requirement)
            // and its cached bufs — CPU reference path only; the GPU engine
            // ignores both.
            let (key, mut taken_bufs) = if has_gpu {
                (None, None)
            } else {
                let max_seq = slots
                    .iter()
                    .filter_map(|s| s.as_ref())
                    .map(|s| (s.prompt_ids.len() + s.out_ids.len()).max(1) as i64)
                    .max()
                    .unwrap_or(1);
                let key = select_bucket(&bundle, Phase::Decode, live as i64, max_seq)
                    .or_else(|| bundle.bucket_keys().find(|k| k.phase == Phase::Decode))
                    .or_else(|| bundle.bucket_keys().next());

                // Look up (or build once) the cached bufs for this bucket key.
                // Take the entry out so we can move it into the blocking task.
                let bufs = if let Some(k) = key {
                    if !bufs_cache.contains_key(&k) {
                        if let Some(b) = bundle.bucket(k) {
                            let pool = CounterPool::from_counters(&b.program.counters);
                            let streams = StreamSet::new(&b.program, pool.len());
                            bufs_cache.insert(
                                k,
                                BucketBufs {
                                    key: k,
                                    pool,
                                    streams,
                                    vocab: sample_vocab(b),
                                },
                            );
                        }
                    }
                    bufs_cache.remove(&k)
                } else {
                    None
                };
                (key, bufs)
            };

            // ρ = λ/μ, exported as the `plowrt_utilization` gauge. Reported only: the decode-rung
            // controller computes its own per-rung utilization and that is what moves the
            // admission window. Nothing on this path may refuse a LIVE slot.
            metrics.util_milli.store(
                (load.utilization(Instant::now()) * 1000.0) as u64,
                Ordering::Relaxed,
            );

            Metrics::add(&metrics.batch_size_sum, live as u64);
            Metrics::inc(&metrics.batch_count);

            // One tick: advance every live slot by N tokens (multi-step).
            // Handed to the blocking pool so the dispatcher task stays hot
            // for arrivals.
            refresh_turns(&mut slots);
            let due = tick_due(&slots, &waiting, cost_id, Instant::now());
            // A co-tenant with more urgent work is waiting for the device: a K-step quantum
            // here would hold it for all K. Under overload, voice turns already ahead of their
            // deadlines give the device back sooner too.
            let outranked = device_turn.as_ref().is_some_and(|dt| dt.outranked_due(due))
                || (crate::serve::deadlines::ahead_under_overload(due, Instant::now())
                    && slots.iter().flatten().any(|s| s.turn.is_some()));
            let steps = if outranked {
                1
            } else if cfg.multi_step {
                #[cfg(any(feature = "cuda", feature = "hsa"))]
                let device_quantum = crate::serve::policy::decode_k(false, token_group);
                #[cfg(not(any(feature = "cuda", feature = "hsa")))]
                let device_quantum = 0;

                if device_quantum > 1
                    && (freed_last_tick
                        || !waiting.is_empty()
                        || slots.iter().flatten().any(|s| s.step == 0))
                {
                    // Prefill is pending, so the next chunk must not wait behind a K-step
                    // quantum. A slot freed last tick counts: its successor is usually a round
                    // trip away, and a K-step quantum here lets the next completion land in the
                    // same wave (two prefills back to back). 15000/C4 TTFT 1099 -> 627 ms.
                    1
                } else if device_quantum > 1 {
                    group_aligned(device_quantum.max(MultiStep::for_batch(live as i64).steps), token_group)
                } else {
                    MultiStep::for_batch(live as i64).steps
                }
            } else {
                1
            };
            #[cfg(feature = "cuda")]
            let quantum_cut = QuantumCut {
                turn: device_turn
                    .clone()
                    .filter(|dt| dt.mode() == crate::serve::cosched::CoSched::Deadline)
                    .map(|dt| (dt, due)),
            };
            #[cfg(not(feature = "cuda"))]
            let quantum_cut = QuantumCut;
            let bundle_ref = Arc::clone(&bundle);
            let state_ref = Arc::clone(&state);
            let slug_for_tick = slug.clone();
            let key_for_tick = key;
            let vocab_for_tick = taken_bufs.as_ref().map(|b| b.vocab).unwrap_or(256);

            // Snapshot the live KV handles before the slots move into the
            // blocking task: if the tick panics, the slots are dropped inside
            // the closure without releasing their arena seq-slots (Slot has no
            // Drop), permanently shrinking the arena. Releasing an
            // already-released handle is a no-op, so the snapshot is safe to
            // replay on the panic path.
            let kv_snapshot: Vec<SlotHandle> = if arena.is_some() {
                live_kv_rows(&slots).collect()
            } else {
                Vec::new()
            };

            // A downstream stage rendering a first chunk has the device to itself.
            while downstream.urgent() && !preempt_seen.load(Ordering::Acquire) {
                tokio::select! {
                    _ = downstream.released() => {}
                    _ = tokio::time::sleep(std::time::Duration::from_millis(1)) => {}
                }
            }
            if let Some(dt) = &device_turn {
                tokio::select! {
                    biased;
                    _ = preempt_wake.notified() => continue,
                    _ = turn.take_due(dt, due) => {}
                    _ = arrival_wake.notified(), if dt.mode() == crate::serve::cosched::CoSched::Deadline => continue,
                }
            }
            if preempt_seen.load(Ordering::Acquire) {
                continue;
            }
            let co_scheduled = device_turn.as_ref().is_some_and(|dt| dt.ordered());
            let pf_rows: usize = slots.iter().flatten().filter(|s| s.step == 0).map(|s| s.prompt_ids.len().saturating_sub(s.pf_pos)).sum();
            let taken_slots = std::mem::take(&mut slots);
            let taken_obs = obs.take().unwrap_or_else(|| {
                let mut obs = RunObserver::new(state.record_trace, indirection_size);
                obs.set_kv_pages_range(kv_pages_range.clone());
                obs
            });
            let arena_ref = arena.clone();
            let kv_pages_for_tick = kv_pages_range.clone();

            metrics.serving.tick_batch.tokens(live);
            let t_service_start = Instant::now();
            let host_timed = crate::obs::host::on();
            let tick = move || {
                let t_body = host_timed.then(Instant::now);
                let out = run_one_tick(
                    &state_ref,
                    &slug_for_tick,
                    &bundle_ref,
                    key_for_tick,
                    vocab_for_tick,
                    taken_slots,
                    taken_bufs,
                    taken_obs,
                    arena_ref,
                    kv_pages_for_tick,
                    steps,
                    cfg.multi_step,
                    co_scheduled,
                    quantum_cut,
                );
                (out, t_body.map_or(0, |t| t.elapsed().as_nanos() as u64))
            };
            // GPU models tick on the dedicated engine thread (or inline on
            // this dispatcher's own thread); the dispatcher task stays hot for
            // arrivals/cancellation either way.
            let joined = match &engine_thread {
                Some(t) => t.run(tick).await,
                None if inline_tick => crate::exec::engine_thread::run_inline(tick),
                None => tokio::task::spawn_blocking(tick)
                    .await
                    .map_err(|e| e.to_string()),
            };

            let t_returned = Instant::now();
            let service = t_returned - t_service_start;
            let ms = service.as_secs_f64() * 1e3;
            let host_disp_ns = host_last_return
                .replace(t_returned)
                .map_or(0, |t| (t_service_start - t).as_nanos() as u64);

            match joined {
                Ok((
                    (
                        returned_slots,
                        returned_bufs,
                        returned_obs,
                        tokens_produced,
                        did_prefill,
                        tick_fault,
                        decode_progress,
                    ),
                    tick_body_ns,
                )) => {
                    if host_timed {
                        let times = crate::obs::host::TickTimes {
                            tick_ns: tick_body_ns,
                            handoff_ns: (service.as_nanos() as u64).saturating_sub(tick_body_ns),
                            disp_ns: host_disp_ns,
                        };
                        if let Some(line) =
                            host_window.tick(times, !did_prefill && decode_progress.is_some())
                        {
                            tracing::info!(%slug, "{line}");
                        }
                    }
                    // Decode-service EWMA: prefill ticks are excluded — see
                    // `service_sample`. Updating on them poisons the admission
                    // predictor and sheds live decode streams.
                    let phase = match (did_prefill, decode_progress.is_some()) {
                        (true, true) => 2,
                        (true, false) => 0,
                        _ => 1,
                    };
                    metrics.serving.ticks[phase].duration(t_service_start.elapsed());
                    metrics.serving.tick_tokens.tokens(tokens_produced);
                    if tick_fault.is_some() {
                        Metrics::inc(&metrics.serving.tick_errors);
                    }
                    // A decode sample is per step: a K-step quantum is compared against per-token deadlines.
                    let (op, took) = if did_prefill {
                        (crate::sched::cost::Op::Prefill { rows: pf_rows.min(crate::config::RuntimeConfig::get().pf_chunk_rows()) }, service)
                    } else {
                        let k = decode_progress.as_ref().map_or(steps, |p| p.steps.get() as u32).max(1);
                        (crate::sched::cost::Op::DecodeTick { width: live }, service / k)
                    };
                    crate::sched::cost::record_id(cost_id, op, took);
                    let sample = service_sample(ms, did_prefill);
                    if let Some(sample) = sample {
                        load.service_ms.update(sample);
                    }
                    if let (Some(controller), Some(progress), Some(sample)) =
                        (rung_controller.as_mut(), decode_progress, sample)
                    {
                        let rung = controller.covering(progress.extent);
                        metrics
                            .decode_rung_actual
                            .store(controller.width(rung) as u64, Ordering::Relaxed);
                        controller.observe_decode(rung, sample, progress.steps);
                    } else if rung_controller.is_some() {
                        metrics.decode_rung_actual.store(0, Ordering::Relaxed);
                    }
                    freed_last_tick = returned_slots.iter().flatten().count() < live;
                    slots = returned_slots;
                    if let Some(b) = returned_bufs {
                        bufs_cache.insert(b.key, b);
                    }
                    obs = Some(returned_obs);
                    Metrics::add(&metrics.tokens, tokens_produced as u64);

                    // Health transition. On the transition INTO Dead (once):
                    // fail every remaining live slot with the fault — ticking
                    // them against a poisoned context could only re-error —
                    // and every later arrival is rejected at admission.
                    let was_dead = matches!(health, EngineHealth::Dead(_));
                    health = advance_health(health, tick_fault);
                    if !was_dead {
                        if let EngineHealth::Dead(f) = &health {
                            metrics.engine_dead.store(true, std::sync::atomic::Ordering::Relaxed);
                            tracing::error!(
                                %slug,
                                error_op = %f.operation,
                                error_code = f.code,
                                error_name = %f.name,
                                "engine dead: fatal device fault — rejecting new requests"
                            );
                            for s in slots.iter_mut() {
                                if let Some(slot) = s.take() {
                                    release_kv(&arena, slot.kv);
                                    let _ = slot.respond.try_send(StreamChunk::Err(
                                        crate::RuntimeError::DeviceFault { info: f.clone() },
                                    ));
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    Metrics::inc(&metrics.serving.tick_errors);
                    tracing::error!(%slug, error = %e, "mux tick task panicked");
                    // Reinitialize; every in-flight slot is lost. Bufs cache
                    // and observer rebuild lazily — the panic is per-tick.
                    // The panicked closure dropped the slots without releasing
                    // their KV seq-slots — return them from the snapshot.
                    for h in kv_snapshot {
                        release_kv(&arena, Some(h));
                    }
                    slots = (0..capacity).map(|_| None).collect();
                    obs = None;
                }
            }

            // Post-tick: drain OOB events from the executor. Handles faults,
            // checkpoints, and speculative verdicts. The channel is lock-free
            // on the hot (emit) side; drain is cold-path per tick.
            oob.drain_events(&mut oob_events);
            for ev in oob_events.drain(..) {
                use crate::exec::oob::OobKind;
                let kind_raw = ev.kind;
                match kind_raw {
                    x if x == OobKind::Fault as u16 => {
                        tracing::warn!(exec = ev.exec, arg0 = ev.arg0, "executor fault");
                    }
                    x if x == OobKind::Checkpoint as u16 => {
                        // §K tracing: record timestamp checkpoint.
                    }
                    x if x == OobKind::SpecVerdict as u16 => {
                        // Speculative decode acceptance length — handled by
                        // the multi-model orchestrator when wired.
                    }
                    _ => {}
                }
            }
        }
        metrics.decode_rung_actual.store(0, Ordering::Relaxed);
        metrics.decode_rung_admission.store(0, Ordering::Relaxed);
        metrics.decode_occupied_extent.store(0, Ordering::Relaxed);
    };
    if inline_tick {
        std::thread::Builder::new()
            .name(dispatcher_name)
            .spawn(move || {
                crate::exec::engine_thread::pin_serving();
                #[cfg(any(feature = "hsa", feature = "cuda"))]
                if let Some(cpus) = dp_cpus {
                    if let Err(e) = crate::exec::engine_affinity::pin_current_thread(&cpus) {
                        tracing::warn!(error = %e, "dp: pinning the rank's dispatcher failed; left unpinned");
                    }
                }
                #[cfg(not(any(feature = "hsa", feature = "cuda")))]
                let _ = dp_cpus;
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("inline mux dispatcher runtime")
                    .block_on(dispatcher)
            })
            .expect("spawn inline mux dispatcher");
    } else {
        tokio::spawn(dispatcher);
    }

    ModelMux {
        tx,
        metrics: handle_metrics,
        preempt: preempt_flag,
        preempt_notify,
        arrival_notify,
        ingress,
        preempted: Arc::new(std::sync::atomic::AtomicBool::new(false)),
    }
}

fn note_dequeued(msg: &MuxMsg, metrics: &Metrics) {
    if matches!(msg, MuxMsg::Job(_, _)) {
        metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

fn note_arrival(now: Instant, load: &mut LoadEstimator, metrics: &Metrics) {
    if packlog::on() {
        if let Some(gap) = load.lambda.since_last(now) {
            eprintln!("PACKLOG ARRIVE gap_us={}", gap.as_micros());
        }
    }
    let lambda = load.lambda.observe(now);
    metrics
        .lambda_milli
        .store((lambda * 1000.0) as u64, Ordering::Relaxed);
}

use crate::sched::admission::{reserved_kv_rows, seat, Denied};

/// How long a request may sit in `waiting` before it stops yielding to younger arrivals, as a
/// multiple of the SLO. The queue is backfill-first on purpose — a small request should take a
/// slot a large one cannot use — but unbounded backfill is precisely what starves the large one,
/// because every small admission it yields to shrinks the budget it is waiting for.
const AGING_SLO_MULTIPLE: f64 = 4.0;
/// Floor under the aging bound, so a tiny `--slo-ms` cannot turn the queue strictly FIFO.
const AGING_FLOOR_MS: f64 = 1_000.0;
/// How long a request may sit in `waiting` at all, as a multiple of the SLO.
const QUEUE_TTL_SLO_MULTIPLE: f64 = 40.0;
/// Floor under the TTL. Deliberately far above the aging bound: a 70k prompt queueing behind
/// live sequences for several seconds is the system working, not a failure, so the TTL only
/// catches a request nothing is going to serve.
const QUEUE_TTL_FLOOR_MS: f64 = 30_000.0;

/// Wait after which a queued request blocks the backfill behind it.
#[inline]
fn queue_aging_ms(slo_ms: f64) -> f64 {
    (slo_ms.max(0.0) * AGING_SLO_MULTIPLE).max(AGING_FLOOR_MS)
}

/// Wait after which a queued request is shed: never under the throughput class, else derived.
#[inline]
fn queue_ttl_ms(slo_ms: f64) -> f64 {
    queue_ttl_with(
        slo_ms,
        crate::serve::policy::queue_ttl_ms(),
    )
}

/// `Some(ms <= 0)` never sheds.
#[inline]
fn queue_ttl_with(slo_ms: f64, set: Option<f64>) -> f64 {
    match set {
        Some(ms) if ms > 0.0 => ms,
        Some(_) => f64::INFINITY,
        None => (slo_ms.max(0.0) * QUEUE_TTL_SLO_MULTIPLE).max(QUEUE_TTL_FLOOR_MS),
    }
}

/// What to do with an entry in `waiting` before it is retried.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Queued {
    /// Try to seat it.
    Retry,
    /// The client is gone. Drop it silently — there is nobody to answer.
    Disconnected,
    /// It has waited past the TTL. Answer 429 and drop it.
    Expired,
}

/// The queue-entry policy, kept pure so the two ways a waiting request leaves without running
/// are decided in one place.
#[inline]
fn queue_verdict(closed: bool, waited_ms: f64, slo_ms: f64, class: JobClass) -> Queued {
    if closed {
        Queued::Disconnected
    } else if waited_ms > queue_ttl_ms(slo_ms) * class.ttl_scale() {
        Queued::Expired
    } else {
        Queued::Retry
    }
}

/// Seat order of a queued request: class first; within a class, every request that has waited
/// `bound` oldest first, then continuing session turns, then requests opening a session, each by
/// arrival. A continuing turn overtakes a new session by at most `bound`.
#[inline]
fn seat_order(class: JobClass, continuing: bool, arrived: Instant, now: Instant, bound: std::time::Duration) -> (JobClass, u8, Instant) {
    let tier = if now.saturating_duration_since(arrived) >= bound {
        0
    } else if continuing {
        1
    } else {
        2
    };
    (class, tier, arrived)
}

/// The head of the queue keeps its seat against [`cache_first`] once it has waited this long.
const CACHE_FIRST_WAIT_MS: f64 = QUEUE_TTL_FLOOR_MS;

/// Cache-aware admission, the throughput class's retention rule: of the waiters in the head's
/// class (`heads` = class and wait per waiter, in seat order; `rows` = rows each attaches from
/// the prefix cache), seat first the one attaching the most rows when that beats the head by a
/// block. FIFO under cyclic session reuse seats exactly the session whose prefix LRU evicted
/// last; this seats sessions still cached, so their turns keep it. The head keeps its seat once
/// it has waited [`CACHE_FIRST_WAIT_MS`]. Returns the waiter to seat ahead of the head.
fn cache_first(heads: &[(JobClass, f64)], rows: &[u32], block: u32) -> Option<usize> {
    let (&(class, waited), &head_rows) = heads.first().zip(rows.first())?;
    if waited >= CACHE_FIRST_WAIT_MS {
        return None;
    }
    let peers = heads.iter().take_while(|h| h.0 == class).count().min(rows.len());
    let (best, &best_rows) = rows[..peers].iter().enumerate().rev().max_by_key(|&(_, r)| *r)?;
    (best > 0 && best_rows >= head_rows.saturating_add(block)).then_some(best)
}

#[inline]
fn waited_ms(now: Instant, arrived: Instant) -> f64 {
    now.saturating_duration_since(arrived).as_secs_f64() * 1e3
}

/// One pass over the wait queue: sweep what will never run, then seat what fits.
///
/// Two fairness rules on top of the backfill:
///
/// * **Sweep.** A disconnected client and a request past its TTL both hold a queue entry they
///   will never use. The per-request disconnect check in [`admit_into`] only fires when a slot
///   is free, so with a full slot table those entries used to sit in `waiting` indefinitely,
///   counting against `ingress_capacity` and inflating the backlog the rung controller widens
///   against. Sweeping is unconditional and costs one atomic load per queued entry.
/// * **Aging.** The first request that does not fit and has waited past
///   [`queue_aging_ms`] stops the pass. Younger requests keep backfilling until then, so the
///   throughput win is preserved; after it, the aged request is only waiting on retirement,
///   which is bounded. Without this a large request is starved forever by a stream of small
///   ones: each small admission it yields to consumes the very budget it needs.
///
/// A request that can never fit is not this function's problem — [`seat`] answers
/// [`Denied::ExceedsWholeBudget`] terminally, so the aging rule can never block on one.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn drain_waiting(
    waiting: &mut std::collections::VecDeque<(Job, Instant)>,
    slots: &mut [Option<Slot>],
    admission_limit: usize,
    now: Instant,
    slo_ms: f64,
    arena: Option<&SharedKvState>,
    metrics: &Arc<Metrics>,
    health: &EngineHealth,
    kv_budget: Option<crate::sched::admission::KvBudget>,
    downstream_full: bool,
) {
    drain_waiting_session(
        waiting,
        slots,
        admission_limit,
        now,
        slo_ms,
        arena,
        metrics,
        health,
        kv_budget,
        downstream_full,
        &mut Retention::off(),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn drain_waiting_session(
    waiting: &mut std::collections::VecDeque<(Job, Instant)>,
    slots: &mut [Option<Slot>],
    admission_limit: usize,
    now: Instant,
    slo_ms: f64,
    arena: Option<&SharedKvState>,
    metrics: &Arc<Metrics>,
    health: &EngineHealth,
    kv_budget: Option<crate::sched::admission::KvBudget>,
    downstream_full: bool,
    retention: &mut Retention,
    prefix: Option<&crate::memory::vmm::PrefixProbe>,
) {
    waiting.retain(|(job, arrived)| {
        let class = job.opts.class;
        match queue_verdict(job.respond.is_closed(), waited_ms(now, *arrived), slo_ms, class) {
            Queued::Retry => true,
            Queued::Disconnected => {
                metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
                false
            }
            Queued::Expired => {
                Metrics::inc(&metrics.admit_shed);
                metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
                let _ = job
                    .respond
                    .try_send(StreamChunk::Err(crate::RuntimeError::Rejected(format!(
                        "queued {:.0} ms past the {:.0} ms queue TTL",
                        waited_ms(now, *arrived),
                        queue_ttl_ms(slo_ms) * class.ttl_scale()
                    ))));
                false
            }
        }
    });

    let bound = crate::serve::cosched::max_wait();
    let order = |(job, arrived): &(Job, Instant)| seat_order(job.opts.class, job.opts.continuing, *arrived, now, bound);
    let queue = waiting.make_contiguous();
    if !queue.is_sorted_by_key(order) {
        queue.sort_by_key(order);
    }
    let aging = queue_aging_ms(slo_ms);
    // Attachable rows per waiter, once per pass: admissions within it barely move the cache.
    let probe = prefix.filter(|_| {
        crate::serve::policy::cache_aware_admission()
            && slots[..admission_limit.min(slots.len())].iter().any(Option::is_none)
    });
    let mut rows: std::collections::VecDeque<u32> = probe
        .map(|p| waiting.iter().map(|(job, _)| p.cached_rows_keyed(&job.prompt_ids, job.opts.prefix.as_ref())).collect())
        .unwrap_or_default();
    let mut still: std::collections::VecDeque<(Job, Instant)> =
        std::collections::VecDeque::new();
    while slots[..admission_limit.min(slots.len())]
        .iter()
        .any(Option::is_none)
    {
        if let Some(p) = probe {
            let heads: Vec<(JobClass, f64)> =
                waiting.iter().map(|(job, arrived)| (job.opts.class, waited_ms(now, *arrived))).collect();
            if let Some(i) = cache_first(&heads, rows.make_contiguous(), p.block_rows()) {
                let entry = waiting.remove(i).expect("in range");
                waiting.push_front(entry);
                let r = rows.remove(i).expect("in range");
                rows.push_front(r);
                Metrics::inc(&metrics.cache_first_admissions);
            }
        }
        rows.pop_front();
        let Some((job, arrived)) = waiting.pop_front() else {
            break;
        };
        match admit_session(
            slots,
            admission_limit,
            job,
            arrived,
            arena,
            metrics,
            health,
            kv_budget,
            downstream_full,
            retention,
        ) {
            Some(held) => {
                let blocks = waited_ms(now, held.1) >= aging;
                still.push_back(held);
                if blocks {
                    break;
                }
            }
            None => {
                // Seated: count it active now, not at the next loop top a long prefill tick away,
                // so a load reader (the DP router) never sees it in neither gauge.
                metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
                metrics.slots_active.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    still.append(waiting);
    *waiting = still;
}

/// Place a job into the first idle slot, asking the arena for the KV
/// footprint upfront. Under temporary KV pressure the request remains queued without occupying
/// a slot. The prompt arrives pre-tokenized (see [`Job::prompt_ids`]).
/// Returns the job UNADMITTED when the KV budget cannot back it yet; the caller holds it and
/// retries before taking anything newer. That is the whole backpressure mechanism: a request
/// the device cannot back stays in the queue instead of being dispatched into a fault.
///
/// `limit` bounds which slots a NEW request may take (the decode-rung admission window); the
/// budget is charged over every live slot, including those above the limit, because their KV is
/// just as resident.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn admit_into(
    slots: &mut [Option<Slot>],
    limit: usize,
    job: Job,
    arrived: Instant,
    arena: Option<&SharedKvState>,
    metrics: &Arc<Metrics>,
    health: &EngineHealth,
    kv_budget: Option<crate::sched::admission::KvBudget>,
    downstream_full: bool,
) -> Option<(Job, Instant)> {
    admit_session(
        slots,
        limit,
        job,
        arrived,
        arena,
        metrics,
        health,
        kv_budget,
        downstream_full,
        &mut Retention::off(),
    )
}

/// [`admit_into`] with session retention: a session request resumes its retained slot when its
/// rows still match; a retained slot never holds back a live request (the least recently used
/// one is evicted for it, and for the KV budget).
#[allow(clippy::too_many_arguments)]
fn admit_session(
    slots: &mut [Option<Slot>],
    limit: usize,
    job: Job,
    arrived: Instant,
    arena: Option<&SharedKvState>,
    metrics: &Arc<Metrics>,
    health: &EngineHealth,
    kv_budget: Option<crate::sched::admission::KvBudget>,
    downstream_full: bool,
    retention: &mut Retention,
) -> Option<(Job, Instant)> {
    if job.respond.is_closed() {
        return None;
    }

    let pair = job.opts.speech.as_ref().is_some_and(|s| s.cfg.is_some());
    let seq_upper = reserved_kv_rows(job.prompt_ids.len(), job.gen.max_tokens, 0) as i64;
    // A CFG pair holds two engine slots' KV.
    let want = seq_upper as u64 * (1 + pair as u64);
    let lim = limit.min(slots.len());
    let session = job.opts.session.as_deref().map(|t| t.session.clone());
    let resume = job
        .opts
        .session
        .as_deref()
        .and_then(|t| retention.table.lookup(&t.session, &t.keys, pair, lim, Instant::now()))
        .filter(|r| slots[r.slot].is_none() && (!pair || slots[r.slot + 1].is_none()));
    let find_free = |slots: &[Option<Slot>], table: &crate::serve::session::RetainTable| {
        if pair {
            (0..lim.saturating_sub(1)).step_by(2).find(|&i| {
                slots[i].is_none() && slots[i + 1].is_none() && !table.holds(i) && !table.holds(i + 1)
            })
        } else {
            (0..lim).find(|&i| slot_free(slots, i) && !table.holds(i))
        }
    };
    // Decode launches cover every slot up to the highest live one, so a retained slot must not push
    // a live request past the rung the live set already runs (or the slack): evict the least
    // recently used retained slot below it instead, and resume only where a fresh seat would not
    // sit lower.
    let floor = retention.floor(slots.iter().rposition(Option::is_some).map_or(0, |i| i + 1));
    let resume = resume.filter(|r| r.slot < floor || slots[..r.slot].iter().all(Option::is_some));
    let mut free_slot = match resume {
        Some(r) => Some(r.slot),
        None => find_free(slots, &retention.table),
    };
    while free_slot.is_none() && retention.table.evict_lru(lim, None) {
        free_slot = find_free(slots, &retention.table);
    }
    while let Some(f) = free_slot.filter(|&f| resume.is_none() && f >= floor) {
        if !retention.table.evict_lru(f, None) {
            break;
        }
        free_slot = find_free(slots, &retention.table);
    }
    // Idle slots that cannot take this job (a lone slot for a pair, a live pair's partner) free
    // up as requests retire: queue, do not refuse.
    if free_slot.is_none() && !(pair && slots.len() < 2) && slots[..lim].iter().any(Option::is_none) {
        return Some((job, arrived));
    }
    let seated = loop {
        let committed = slots
            .iter()
            .flatten()
            .map(|s| {
                reserved_kv_rows(s.prompt_ids.len(), s.gen.max_tokens, s.out_ids.len())
                    * (1 + s.cfg.is_some() as u64)
            })
            .chain(retention.table.rows_except(session.as_deref()));
        match seat(
            matches!(health, EngineHealth::Dead(_)),
            free_slot,
            want,
            committed,
            kv_budget,
            downstream_full,
        ) {
            Err(Denied::KvBudgetFull { .. })
                if retention.table.evict_lru(usize::MAX, session.as_deref()) => {}
            seated => break seated,
        }
    };
    let idx = match seated {
        Ok(idx) => idx,
        Err(denied) => {
            // Retryable: stays queued, says nothing, costs no slot. This is the whole
            // backpressure mechanism.
            if denied.is_retryable() {
                if let Denied::KvBudgetFull { want } = denied {
                    Metrics::inc(&metrics.kv_budget_denials);
                    tracing::debug!(
                        want,
                        max_rows = kv_budget.map(|b| b.max_rows()).unwrap_or(0),
                        "mux: KV budget full — request stays queued"
                    );
                }
                return Some((job, arrived));
            }
            // Terminal: answer the stream with the reason this request can never be seated.
            let err = match denied {
                Denied::EngineDead => {
                    // The poisoning itself was logged once; this stays at debug so a request
                    // flood does not become a log flood.
                    tracing::debug!("mux: engine dead — request rejected");
                    let EngineHealth::Dead(info) = health else {
                        unreachable!("seat only returns EngineDead for a dead engine")
                    };
                    crate::RuntimeError::DeviceFault { info: info.clone() }
                }
                Denied::NoFreeSlot => {
                    tracing::warn!(
                        capacity = slots.len(),
                        "mux: no free slot — request rejected"
                    );
                    crate::RuntimeError::Rejected("engine at capacity — no free slot".into())
                }
                Denied::ExceedsWholeBudget { want, max_rows } => {
                    tracing::warn!(
                        want,
                        max_rows,
                        "mux: request exceeds the whole KV budget — rejected"
                    );
                    crate::RuntimeError::ContextLength(format!(
                        "request reserves {want} tokens; this device can back {max_rows} across \
                         all concurrent sequences"
                    ))
                }
                Denied::KvBudgetFull { .. } | Denied::DownstreamFull => {
                    unreachable!("handled as retryable above")
                }
            };
            // Counted for every terminal denial. Kept separate from the admission-shed path:
            // both end as a 429, but shedding is the controller dropping live work because
            // predicted wait passed the SLO, while these are arrival meeting a hard limit.
            Metrics::inc(&metrics.rejected);
            let _ = job.respond.try_send(StreamChunk::Err(err));
            return None;
        }
    };

    let kv = if let Some(arena) = arena {
        match arena.lock().arena.allocate_slot(seq_upper) {
            Ok(h) => Some(h),
            Err(e) => {
                Metrics::inc(&metrics.rejected);
                tracing::warn!(error = %e, seq_upper, "mux: kv arena OOM — request shed");
                let _ = job
                    .respond
                    .try_send(StreamChunk::Err(crate::RuntimeError::Oom(format!(
                        "kv: {e}"
                    ))));
                return None;
            }
        }
    } else {
        None
    };

    let telemetry = Some(RequestMetrics::new(
        Arc::clone(metrics),
        arrived,
        job.arrived,
        job.prompt_ids.len(),
    ));
    if let Some(session) = &session {
        retention.table.forget(session);
    }
    let resume = resume.filter(|r| r.slot == idx).map_or(0, |r| r.rows);
    let speech = job.opts.speech.is_some();
    slots[idx] = Some(Slot {
        session: retention.seat(job.opts.session, idx, pair, speech, resume, metrics),
        resume,
        telemetry,
        prompt_ids: job.prompt_ids,
        out_ids: Vec::new(),
        gen: job.gen,
        arrived: job.arrived,
        stop_tail: String::new(),
        stop_pending: String::new(),
        respond: job.respond,
        prefix_offset: 0,
        read_offset: 0,
        executed: 0,
        step: 0,
        pf_pos: 0,
        cached_tokens: 0,
        kv,
        class: job.opts.class,
        raw_tokens: job.opts.raw_tokens,
        turn: job.opts.turn.as_ref().and_then(|k| crate::serve::turns::table().times(k)),
        turn_key: job.opts.turn,
        cfg: pair.then(|| {
            Box::new(CfgRun {
                rng: job.opts.speech.as_ref().and_then(|s| s.cfg.as_ref()?.seed).map(crate::text::sample::SplitMix::new),
                partner_pf: 0,
                cond: Vec::new(),
                uncond: Vec::new(),
                scratch: Vec::new(),
                dev_seeded: false,
            })
        }),
        speech: job.opts.speech,
        held: Vec::new(),
        held_finish: None,
        parked_at: None,
        lp: None,
        prefix: job.opts.prefix,
        mm: job.opts.mm,
    });
    None
}

/// Whether slot `i` is idle and not the reserved partner of a live CFG owner.
fn slot_free(slots: &[Option<Slot>], i: usize) -> bool {
    slots[i].is_none() && !(i % 2 == 1 && slots[i - 1].as_ref().is_some_and(|s| s.cfg.is_some()))
}

/// Return a slot's KV blocks to the arena (no-op when the slot never got one).
fn release_kv(arena: &Option<SharedKvState>, handle: Option<SlotHandle>) {
    if let (Some(a), Some(h)) = (arena.as_ref(), handle) {
        a.lock().arena.release_slot(h);
    }
}

/// Terminate a failing slot: release its KV allocation and notify the client.
#[allow(dead_code)]
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu", test))]
fn fail_slot(
    slot_opt: &mut Option<Slot>,
    arena: &Option<SharedKvState>,
    err: crate::RuntimeError,
) {
    if let Some(taken) = slot_opt.take() {
        release_kv(arena, taken.kv);
        let _ = taken.respond.try_send(StreamChunk::Err(err));
    }
}

/// Terminate all active feed slots with the given error.
#[allow(dead_code)]
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu", test))]
fn fail_feeds(
    slots: &mut [Option<Slot>],
    feeds: &[(usize, u32)],
    arena: &Option<SharedKvState>,
    err: &crate::RuntimeError,
) {
    let msg = err.to_string();
    for &(slot, _) in feeds {
        fail_slot(&mut slots[slot], arena, fanout_err(err, &msg));
    }
}

/// Close every live slot with `FinishReason::Preempted` — the stream carries
/// everything generated so far plus honest usage, and the slot frees exactly
/// as it does when a client disconnects mid-generation (the sanctioned
/// teardown path: take the slot, release its KV; the engine's sequence slot
/// is reclaimed on the next admit).
async fn preempt_slots(slots: &mut [Option<Slot>], arena: &Option<SharedKvState>) {
    for slot_opt in slots.iter_mut() {
        let Some(mut slot) = slot_opt.take() else {
            continue;
        };
        if let Some(telemetry) = slot.telemetry.as_mut() {
            telemetry.finish(FinishReason::Preempted, slot.executed);
        }
        let done = StreamChunk::Done {
            executed: slot.executed,
            reason: FinishReason::Preempted,
            usage: crate::serve::stream::TokenUsage {
                prompt_tokens: slot.prompt_ids.len(),
                cached_tokens: slot.cached_tokens,
                completion_tokens: slot.out_ids.len(),
            },
        };
        // A dropped terminal is not a lost token — it is a 500 ("stream ended
        // without a finish reason") on the buffered path and a stream that
        // just stops on SSE. `try_send` alone lost it whenever the client's
        // 32-slot channel happened to be full, which is precisely the client
        // that is behind and most needs telling why its answer stopped.
        //
        // `Closed` needs no delivery (the handler is gone). `Full` waits, but
        // bounded: the dispatcher is on its way out and must not be pinned by
        // a client that has stopped reading without disconnecting.
        match slot.respond.try_send(done) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Closed(_)) => {}
            Err(mpsc::error::TrySendError::Full(done)) => {
                if tokio::time::timeout(TERMINAL_DELIVERY, slot.respond.send(done))
                    .await
                    .is_err()
                {
                    tracing::warn!(
                        "preempt: client did not drain in {}ms — terminal chunk dropped",
                        TERMINAL_DELIVERY.as_millis()
                    );
                }
            }
        }
        release_kv(arena, slot.kv);
    }
}

/// How long a preempt waits for a backpressured client to make room for its
/// terminal chunk before giving up on it.
const TERMINAL_DELIVERY: std::time::Duration = std::time::Duration::from_millis(250);

/// Refresh `obs.indirection[KV_PAGES]` from the arena: for each live row (in
/// compact order — idle slots and slots without a KV handle are skipped) write
/// the per-`(row, layer)` **seq-slot base** — the address of that sequence's
/// `(kv=0, head=0)` head-slot in the layer's pool. The attention kernel adds the
/// separable per-head tail `(kv·kv_heads + head)·max_seqs·head_slot_bytes` using
/// the pool geometry, so one entry per `(row, layer)` suffices (no per-head slot
/// blow-up in the model-sized `KV_PAGES` region).
///
/// Layout is **row-major**: `KV_PAGES.start + row*n_layers + layer`, matching how
/// a batched attention kernel indexes a `B×L` tile. The range is allocated for the
/// mux capacity up front, so every live row fits by construction.
///
/// Called from both the batched and fallback tick paths just before the bucket
/// walk so a real attention kernel would see fresh addresses on dispatch. In the
/// CPU reference the interpreter's `StepObserver::on_fire` for `Body::Flash`
/// snapshots this range into `obs.kv_writes` — the phase-4b1 consumer — proving
/// the compiler-emitted addresses reach the fire site.
///
/// `live` is an iterator over each active slot's `SlotHandle`, in the row order
/// the caller intends. Free-standing so unit tests can drive it without
/// constructing a mux.
fn refresh_indirection<I>(
    obs: &mut RunObserver,
    live: I,
    arena: &Option<SharedKvState>,
    kv_pages_range: std::ops::Range<usize>,
) where
    I: IntoIterator<Item = SlotHandle>,
{
    // Wipe the region so stale entries from a prior (possibly wider) tick
    // never leak into the current dispatch.
    for i in kv_pages_range.clone() {
        obs.indirection.set(i, 0);
    }
    let Some(arena) = arena.as_ref() else { return };
    let arena = arena.lock();
    let n_layers = arena.arena.n_layers();
    if n_layers == 0 {
        return;
    }

    for (row, handle) in live.into_iter().enumerate() {
        for layer in 0..n_layers {
            let idx = kv_pages_range.start + row * n_layers + layer;
            debug_assert!(idx < kv_pages_range.end);
            let addr = arena.arena.seq_slot_base(handle, layer).unwrap_or(0);
            obs.indirection.set(idx, addr);
        }
    }
}

/// Iterator adapter: the `SlotHandle` of every live slot with a KV allocation.
/// Idle slots and no-KV slots are skipped so the row order stays compact.
fn live_kv_rows(slots: &[Option<Slot>]) -> impl Iterator<Item = SlotHandle> + '_ {
    slots.iter().filter_map(|s| s.as_ref()?.kv)
}

/// Run one decode step for every live slot; fire and free finished slots.
///
/// **Batched path.** When the bucket carries `TOKEN_SAMPLE_BATCH` the mux
/// packs every live slot's logits into a `B×vocab` tile, sets per-row
/// params/rng, and fires the bucket **once** via `AppState::step_batch`. A GPU
/// packet may instead cover those decode rows and waiting prefill rows in one
/// mixed launch. The
/// produced tokens land in `obs.host.slot_tokens[row]`. This is the phase-3
/// "one bucket walk per tick" path.
///
/// **Fallback path.** When the bucket has no `SAMPLE_BATCH` (batch=1 rungs,
/// legacy buckets, tests using tiny_program), fall back to per-slot serial
/// ticks via `AppState::step_token` — the phase-1 behavior.
fn run_one_tick(
    state: &AppState,
    // The API slug, which is the key engines are INSTALLED under. Distinct
    // from `bundle.network()`, the manifest's network name: `--assets DIR
    // --slug other` registers under `other` while the manifest still says
    // whatever it says. Looking the engine up by network therefore missed it
    // for any slug override and fell through to the CPU reference
    // interpreter — fluent, wrong, and fast, with nothing in the log. It only
    // ever bites when a slug differs from a network name, which is to say when
    // more than one model is in play.
    slug: &str,
    bundle: &ModelBundle,
    key: Option<BucketKey>,
    vocab: usize,
    mut slots: Vec<Option<Slot>>,
    mut bufs: Option<BucketBufs>,
    mut obs: RunObserver,
    arena: Option<SharedKvState>,
    kv_pages_range: std::ops::Range<usize>,
    steps: u32,
    // `--multi-step` itself, kept apart from `steps` because `steps == 1` is ambiguous: it is
    // both "multi-step is off" and "`MultiStep::for_batch` collapsed at this batch size".
    #[cfg_attr(
        not(any(feature = "hsa", feature = "cpu")),
        allow(unused_variables)
    )]
    multi_step: bool,
    #[cfg_attr(
        not(any(feature = "cuda", feature = "hsa", feature = "cpu")),
        allow(unused_variables)
    )]
    co_scheduled: bool,
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))] quantum_cut: QuantumCut,
) -> (
    Vec<Option<Slot>>,
    Option<BucketBufs>,
    RunObserver,
    usize,
    bool,
    // First device fault seen this tick — the dispatcher's EngineHealth
    // signal. Always `None` on the CPU reference path.
    Option<crate::DeviceErrorInfo>,
    // Only successful decode contributes a rung sample. Steps count device
    // execution, including tokens discarded after a stop within the quantum.
    Option<DecodeProgress>,
) {
    let bucket = key.and_then(|k| bundle.bucket(k));
    for slot in slots.iter_mut() {
        if slot.as_ref().is_some_and(|s| s.parked_at.is_some()) {
            flush_parked(slot, &arena);
        }
    }
    let mut tokens_this_tick = 0usize;
    #[cfg_attr(not(any(feature = "cuda", feature = "hsa")), allow(unused_mut))]
    let mut tick_fault: Option<crate::DeviceErrorInfo> = None;

    // GPU path: when this model has an sm_120 engine, every token comes from
    // the persistent interpreter on the device — the CPU reference walk and
    // its stand-in logits are bypassed entirely. The engine drives B
    // independent sequence slots (the compiled PLOW_DECODE_BATCH; the slot
    // table is sized to it at spawn, so mux slot i IS engine slot i). Per
    // tick: a packet-declared mixed variant combines live decode and waiting
    // prefill rows when available; otherwise prefill runs before one batched
    // decode launch.
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    if let Some(eng) = state.gpu_engine(slug) {
        let mut guard = eng.lock();
        // Per-backend tick bodies, because the two engines differ in kind: the
        // sm_120 engine is slotted (B sequences, chunked prefill, prefix
        // sharing, device sampling) and the gfx950 one is single-sequence.
        //
        // The CUDA arm's body is deliberately left at its original indentation
        // and otherwise untouched — the only edits are `&mut e` -> `&mut *e`,
        // now that `e` is a `&mut GpuEngine` rather than the lock guard. Keeping
        // it un-reindented is what makes the diff prove the shipped path did not
        // change; reflow it only in a commit that changes nothing else.
        #[cfg(feature = "cuda")]
        #[allow(irrefutable_let_patterns)]
        if let crate::serve::engine::ServeEngine::Cuda(e) = &mut *guard {
            let mut disconnected = smallvec::SmallVec::<[bool; 128]>::new();
            disconnected.resize(e.batch(), false);
            let result = (|| {
            let stop = Arc::clone(e.stop_ids());
            let cap = e.batch();

            // Slots past the engine batch cannot be served — only reachable on a
            // capacity/engine mismatch, and better a loud 429 than a hang.
            for slot_opt in slots.iter_mut().skip(cap) {
                if let Some(taken) = slot_opt.take() {
                    tracing::warn!(cap, "gpu: slot past engine batch rejected");
                    release_kv(&arena, taken.kv);
                    let _ =
                        taken
                            .respond
                            .try_send(StreamChunk::Err(crate::RuntimeError::Rejected(format!(
                                "GPU engine serves {cap} sequence slot(s)"
                            ))));
                }
            }

            // Multimodal soft-token rows reach this engine's slab before any launch this tick reads
            // them; a prompt row the table does not hold fails the request instead of reading pad.
            for i in 0..cap.min(slots.len()) {
                let Some(s) = slots[i].as_mut() else { continue };
                let Some(mm) = s.mm.as_mut().filter(|mm| !mm.staged()) else { continue };
                let staged = match state.mm_model(slug) {
                    Some(own) => own.slab().stage(mm, &s.prompt_ids, |t, off, b| e.write_tensor_ordered(t, off, b)),
                    None => Err(crate::RuntimeError::Rejected("multimodal request on an engine without multimodal state".into())),
                };
                if let Err(err) = staged {
                    tracing::warn!(%err, "multimodal rows not staged");
                    if let Some(taken) = slots[i].take() {
                        release_kv(&arena, taken.kv);
                        let _ = taken.respond.try_send(StreamChunk::Err(err));
                    }
                }
            }

            // Whether this tick does any prefill work — reported to the dispatcher
            // so prefill tick durations never enter the decode-service EWMA.
            let did_prefill =
                (0..cap.min(slots.len())).any(|i| slots[i].as_ref().is_some_and(|s| s.step == 0));

            // Decode feeds, gathered BEFORE the prefill pass so a slot prefilled
            // this tick (which just produced its first token) doesn't also step.
            let mut feeds = gpu_decode_feeds(&slots, cap);

            // PX-17: throughput mode — while any slot is mid-prefill, drop the decode
            // feeds so the prefill chain runs uninterrupted and no decode launch pays
            // its fixed cost at a partial batch. Every deferred row is picked up by a
            // full-batch decode tick once prefill drains.
            let defer_decode = pf_defer_decode();
            if defer_decode && did_prefill {
                feeds.clear();
            }

            // A pipelined decode step may still be in flight from the previous tick. Anything
            // but its exact continuation (a prefill, a changed row set, a row the device cannot
            // sample) completes it first, streams its tokens, and re-gathers.
            if e.pipe_busy()
                && (did_prefill || !e.pipe_covers(&feeds) || !gpu_pipe_rows(&feeds, &slots))
            {
                let mut done = std::mem::take(&mut obs.host.pipe_tokens);
                match e.pipe_drain(&mut done) {
                    Ok(()) => {
                        for &(i, token) in &done {
                            if slots[i].is_some() {
                                disconnected[i] |= gpu_emit_slot_token(
                                    &mut slots[i],
                                    &arena,
                                    bundle,
                                    token,
                                    &mut tokens_this_tick,
                                    stop.as_slice(),
                                );
                            }
                        }
                    }
                    Err(err) => {
                        note_fault(&mut tick_fault, &err);
                        fail_feeds(&mut slots, &feeds, &arena, &err);
                    }
                }
                obs.host.pipe_tokens = done;
                feeds = gpu_decode_feeds(&slots, cap);
                if defer_decode && did_prefill {
                    feeds.clear();
                }
            }

            // A mixed packet variant executes existing decode rows and a
            // prefix of waiting prefill rows in one compiler-emitted program.
            // Keep each prompt's last token for the ordinary decode path so
            // it can produce that request's first output token. Admit only a
            // cold prefill span until repeated mixed-span parity is qualified.
            if !feeds.is_empty() && did_prefill {
                let available_prefill: usize = slots
                    .iter()
                    .take(cap)
                    .filter_map(|slot| slot.as_ref())
                    .filter(|slot| {
                        slot.step == 0
                            && slot.pf_pos == 0
                            && !slot.respond.is_closed()
                            && !slot.serial_prefill()
                    })
                    .map(|slot| {
                        slot.prompt_ids
                            .len()
                            .saturating_sub(1)
                            .saturating_sub(slot.pf_pos)
                    })
                    .sum();
                if let Some(rows) = e.mixed_step_rows(feeds.len(), available_prefill) {
                    let prefill_capacity = rows as usize - feeds.len();
                    let mut pack = Vec::<(usize, usize, usize)>::new();
                    let mut remaining = prefill_capacity;
                    let mut oldest_first: smallvec::SmallVec<[usize; 64]> =
                        (0..slots.len().min(cap)).filter(|&i| slots[i].is_some()).collect();
                    oldest_first.sort_unstable_by_key(|&i| slots[i].as_ref().map(|s| s.arrived));
                    for i in oldest_first {
                        if remaining == 0 {
                            break;
                        }
                        let Some(slot) = slots[i].as_ref() else {
                            continue;
                        };
                        if slot.step != 0
                            || slot.pf_pos != 0
                            || slot.respond.is_closed()
                            || slot.serial_prefill()
                        {
                            continue;
                        }
                        if slot.pf_pos == 0 {
                            let reserve = slot.prompt_ids.len() + slot.gen.max_tokens.max(1);
                            if let Err(err) = e.begin_slot(i, reserve) {
                                tracing::warn!(
                                    slot = i,
                                    error = %err,
                                    error_code = ?err.device_code(),
                                    fatal = err.is_fatal(),
                                    "gpu: mixed-step begin failed"
                                );
                                note_fault(&mut tick_fault, &err);
                                if let Some(taken) = slots[i].take() {
                                    release_kv(&arena, taken.kv);
                                    let _ = taken.respond.try_send(StreamChunk::Err(err));
                                }
                                continue;
                            }
                            let attached = {
                                let request = slots[i].as_mut().expect("checked Some");
                                e.attach_prompt_keyed(i, &request.prompt_ids, request.prefix.take())
                            };
                            match attached {
                                Ok(frontier) => {
                                    let request = slots[i].as_mut().expect("checked Some");
                                    request.pf_pos = frontier;
                                    request.cached_tokens = e.attached_rows(i) as usize;
                                }
                                Err(err) => {
                                    note_fault(&mut tick_fault, &err);
                                    if let Some(taken) = slots[i].take() {
                                        release_kv(&arena, taken.kv);
                                        let _ = taken.respond.try_send(StreamChunk::Err(err));
                                    }
                                    continue;
                                }
                            }
                        }
                        let request = slots[i].as_ref().expect("checked Some");
                        if request.pf_pos != 0 {
                            continue;
                        }
                        let start = request.pf_pos;
                        let available = request
                            .prompt_ids
                            .len()
                            .saturating_sub(1)
                            .saturating_sub(start);
                        if available == 0 {
                            continue;
                        }
                        let mut take = available
                            .min(remaining)
                            .min(pf_chunk_rows())
                            .min(e.pf_request_max_rows());
                        if request.mm.as_deref().is_some_and(|j| j.spans()) {
                            take = plow_asset::multimodal::span_safe_rows(
                                &request.prompt_ids,
                                start,
                                take,
                                e.pf_stage_rows(),
                            );
                            if take == 0 {
                                continue;
                            }
                        }
                        pack.push((i, start, take));
                        remaining -= take;
                    }
                    if pack.last().is_some_and(|&(_, start, len)| {
                        start
                            .checked_add(len)
                            .and_then(|end| end.checked_add(remaining))
                            .is_none_or(|end| end > e.max_ctx())
                    }) {
                        pack.clear();
                    }
                    if !pack.is_empty() {
                        let decode: Vec<_> = feeds
                            .iter()
                            .map(|&(slot, token)| plow_asset::mixed_step::DecodeRequest {
                                slot: slot as u32,
                                state_slot: slot as u32,
                                token,
                            })
                            .collect();
                        let prefill: Vec<_> = pack
                            .iter()
                            .map(|&(slot, start, len)| {
                                let request = slots[slot].as_ref().expect("packed slot is Some");
                                plow_asset::mixed_step::PrefillRequest {
                                    slot: slot as u32,
                                    state_slot: slot as u32,
                                    start: start as u32,
                                    tokens: &request.prompt_ids[start..start + len],
                                    prompt_len: request.prompt_ids.len() as u32,
                                }
                            })
                            .collect();
                        let mut toks = std::mem::take(&mut obs.host.slot_tokens);
                        toks.resize(feeds.len(), 0);
                        let mixed = e.mixed_step(rows, &decode, &prefill, &mut toks);
                        drop(prefill);
                        match mixed {
                            Ok(()) => {
                                tracing::debug!(
                                    decode = feeds.len(),
                                    prefill = pack.len(),
                                    prefill_rows =
                                        pack.iter().map(|&(_, _, len)| len).sum::<usize>(),
                                    rows,
                                    "gpu: mixed prefill/decode launch"
                                );
                                for &(slot, start, len) in &pack {
                                    slots[slot].as_mut().expect("packed slot is Some").pf_pos =
                                        start + len;
                                }
                                if let Err(err) = gpu_batch_logprobs(
                                    &mut *e,
                                    feeds.iter().enumerate().map(|(row, &(slot, _))| (row, slot, toks[row])),
                                    &mut slots,
                                ) {
                                    tracing::warn!(error = %err, "gpu: batched logprob stats failed; per-row path");
                                }
                                for (row, &(slot, _)) in feeds.iter().enumerate() {
                                    let slot_opt = &mut slots[slot];
                                    let Some(request) = slot_opt.as_mut() else {
                                        continue;
                                    };
                                    match gpu_finish_token(&mut *e, row, request, toks[row]) {
                                        Ok(token) => disconnected[slot] |= handle_produced_token(
                                            slot_opt,
                                            &arena,
                                            bundle,
                                            token,
                                            1,
                                            &mut tokens_this_tick,
                                            Some(stop.as_slice()),
                                        ),
                                        Err(err) => {
                                            note_fault(&mut tick_fault, &err);
                                            fail_slot(slot_opt, &arena, err);
                                        }
                                    }
                                }
                            }
                            Err(err) => {
                                tracing::warn!(
                                    error = %err,
                                    error_code = ?err.device_code(),
                                    fatal = err.is_fatal(),
                                    decode = feeds.len(),
                                    prefill = pack.len(),
                                    rows,
                                    "gpu: mixed-step launch failed"
                                );
                                note_fault(&mut tick_fault, &err);
                                let msg = err.to_string();
                                fail_feeds(&mut slots, &feeds, &arena, &err);
                                for &(slot, _, _) in &pack {
                                    fail_slot(&mut slots[slot], &arena, fanout_err(&err, &msg));
                                }
                            }
                        }
                        obs.host.slot_tokens = toks;
                        return (slots, bufs, obs, tokens_this_tick, true, tick_fault, None);
                    }
                }
            }

            let pack_t = packlog::on().then(Instant::now);
            let (pack_live, pack_prefilling) = if pack_t.is_some() {
                let live = slots.iter().take(cap).flatten();
                (live.clone().count(), live.filter(|s| s.step == 0).count())
            } else {
                (0, 0)
            };
            // The unified token batch decodes `feeds` inside the prefill pass and clears
            // them; the rung controller still owes that step its progress.
            let feeds_before = decode_feed_extent(&feeds);
            let mut unified_progress = None;
            if e.pf_batch_enabled() {
                let compact = e.has_packed_terminal();
                let mut completed = std::mem::take(&mut obs.host.prefill_tokens);
                completed.clear();
                let mut dev_sampled = false;
                if obs.host.ride.needs_widths() {
                    let widths: Vec<usize> = e.effective_decode_rungs().iter().map(|&w| w as usize).collect();
                    obs.host.ride.set_widths(&widths);
                }
                if let Some(f) = gpu_prefill_batched_pass(
                    &mut *e, &mut slots, cap, &arena, feeds.is_empty(), co_scheduled, &mut completed,
                    &mut feeds, &mut obs.host.token_batch_tokens, &mut dev_sampled,
                    &mut obs.host.ride,
                ) {
                    if tick_fault.is_none() {
                        tick_fault = Some(f);
                    }
                } else if feeds.is_empty() {
                    unified_progress = feeds_before.and_then(|extent| {
                        Some(DecodeProgress {
                            extent,
                            steps: std::num::NonZeroUsize::new(1)?,
                        })
                    });
                }
                let mut guided: smallvec::SmallVec<[usize; 8]> = Default::default();
                if let Err(err) = gpu_batch_logprobs(
                    &mut *e,
                    completed.iter().enumerate().map(|(row, &(i, token))| (row, i, token)),
                    &mut slots,
                ) {
                    tracing::warn!(error = %err, "gpu: batched logprob stats failed; per-row path");
                }
                for (row, &(i, token)) in completed.iter().enumerate() {
                    // A speech prompt's last row: stage its position base; a CFG member's logits
                    // wait in its owner until both members have their row.
                    match gpu_speech_prompt_done(&mut *e, &mut slots, row, i) {
                        Ok(None) => continue,
                        Ok(Some((owner, true))) => {
                            guided.push(owner);
                            continue;
                        }
                        Ok(Some((_, false))) => {}
                        Err(err) => {
                            note_fault(&mut tick_fault, &err);
                            let owner = if slots[i].is_some() { i } else { i - 1 };
                            fail_slot(&mut slots[owner], &arena, err);
                            continue;
                        }
                    }
                    let was_dev = dev_sampled && slots[i].as_ref().and_then(dev_row_spec).is_some();
                    gpu_finish_and_emit_token(
                        &mut *e,
                        row,
                        i,
                        &mut slots[i],
                        &arena,
                        bundle,
                        token,
                        was_dev,
                        &mut tokens_this_tick,
                        stop.as_slice(),
                        &mut tick_fault,
                        &mut disconnected,
                    );
                }
                guided.dedup();
                for owner in guided {
                    let Some(slot) = slots[owner].as_mut() else { continue };
                    if slot.step != 0 || !packed_prompt_done(&slots, owner, 0) {
                        continue;
                    }
                    let token = cfg_draw(slots[owner].as_mut().expect("checked Some"));
                    disconnected[owner] |= gpu_emit_slot_token(
                        &mut slots[owner],
                        &arena,
                        bundle,
                        token,
                        &mut tokens_this_tick,
                        stop.as_slice(),
                    );
                }
                obs.host.prefill_tokens = completed;
                if !compact {
                    for i in 0..slots.len().min(cap) {
                        let Some(s) = slots[i].as_ref() else { continue };
                        if s.step != 0 {
                            continue;
                        }
                        let n = s.prompt_ids.len();
                        if n == 0 {
                            fail_slot(
                                &mut slots[i],
                                &arena,
                                crate::RuntimeError::Rejected("empty prompt".into()),
                            );
                            continue;
                        }
                        if !e.packed_slot_ready(i)
                            || s.pf_pos + 1 != n
                            || !packed_prompt_done(&slots, i, 1)
                        {
                            continue; // still mid-prefill
                        }
                        if s.respond.is_closed() {
                            disconnected[i] = true;
                            if let Some(taken) = slots[i].take() {
                                release_kv(&arena, taken.kv);
                            }
                            continue;
                        }
                        let last = *s.prompt_ids.last().expect("n >= 1");
                        let pair = s.cfg.is_some();
                        // The decode launch embeds the last prompt row at its position base.
                        if let Some(base) = s.speech.as_ref().and_then(|sp| sp.pos_base) {
                            let written = [i, i + 1][..1 + pair as usize].iter().try_for_each(|&row| {
                                e.write_tensor("in.pos_base", (row * 4) as u64, &base.to_le_bytes())
                            });
                            if let Err(err) = written {
                                note_fault(&mut tick_fault, &err);
                                fail_slot(&mut slots[i], &arena, err);
                                continue;
                            }
                        }
                        feeds.push((i, last));
                        if pair {
                            feeds.push((i + 1, last));
                        }
                    }
                }
            } else {
                // Prefill pass — chunk-interleaved continuous batching. With live
                // decoders, at most ONE capped prefill chunk runs per tick, so a
                // mid-decode arrival stalls the running streams by one chunk (not one
                // whole prompt); the decode launch below runs between chunks. With no
                // decoders live, stop once a request becomes ready for decode,
                // unless the caller explicitly defers decode.
                let cap_rows = if co_scheduled {
                    co_sched_prefill_rows(e.pf_max_rows(), pf_interleave_rows())
                } else if feeds.is_empty() {
                    usize::MAX
                } else {
                    pf_interleave_rows()
                };
                // Short serial-only prompts (speech) share one tick up to the interleave budget:
                // nothing packs them, and one per tick leaves the decode batch starved.
                let mut serial_rows = 0usize;
                loop {
                    let Some(i) = (0..slots.len().min(cap))
                        .filter(|&i| slots[i].as_ref().is_some_and(|s| s.step == 0))
                        .min_by_key(|&i| slots[i].as_ref().map(|s| (s.class, s.arrived)))
                    else {
                        break;
                    };
                    // Client gone mid-prefill (chunks span ticks now) — don't spend
                    // launches building KV for a dead stream.
                    if slots[i]
                        .as_ref()
                        .map(|s| s.respond.is_closed())
                        .unwrap_or(false)
                    {
                        disconnected[i] = true;
                        if let Some(taken) = slots[i].take() {
                            release_kv(&arena, taken.kv);
                        }
                        continue;
                    }
                    let slot_opt = &mut slots[i];
                    // §TTFT (CUDA arm): queue = submit -> this tick; prefill = the chunk call.
                    if crate::obs::ttft::on() {
                        if let Some(sr) = slot_opt.as_ref() {
                            if sr.pf_pos == 0 {
                                crate::obs::ttft::QUEUE.add(sr.arrived.elapsed().as_nanos() as u64);
                            }
                        }
                    }
                    let t_pf = std::time::Instant::now();
                    let (serial, pf_before) =
                        slot_opt.as_ref().map_or((false, 0), |s| (s.serial_prefill(), s.pf_pos));
                    let res = gpu_prefill_advance(
                        &mut *e,
                        i,
                        slot_opt.as_mut().expect("checked Some"),
                        cap_rows,
                    );
                    crate::obs::ttft::PREFILL.add(t_pf.elapsed().as_nanos() as u64);
                    match res {
                        Ok(Some(token)) => {
                            tracing::debug!(token, slot = i, step = 0usize, "gpu: token");
                            // Detok + channel send, timed like the AMD arm: the
                            // old `add(0)` counted a sample with zero time, so
                            // the CUDA TTFT dump reported a confident 0.000 ms
                            // for this phase and rolled the real cost into
                            // UNACCOUNTED. Gated like the QUEUE site above.
                            let t_tok = crate::obs::ttft::on().then(std::time::Instant::now);
                            disconnected[i] |= handle_produced_token(
                                slot_opt,
                                &arena,
                                bundle,
                                token,
                                1,
                                &mut tokens_this_tick,
                                Some(stop.as_slice()),
                            );
                            if let Some(t) = t_tok {
                                crate::obs::ttft::FIRST_TOK.add(t.elapsed().as_nanos() as u64);
                            }
                        }
                        Ok(None) => {
                            // Mid-prefill: the frontier advanced one chunk.
                        }
                        Err(err) => {
                            tracing::warn!(
                                slot = i,
                                error = %err,
                                error_code = ?err.device_code(),
                                fatal = err.is_fatal(),
                                model = bundle.network(),
                                "gpu: prefill failed"
                            );
                            note_fault(&mut tick_fault, &err);
                            fail_slot(slot_opt, &arena, err);
                        }
                    }
                    if serial {
                        serial_rows += slot_opt
                            .as_ref()
                            .map_or(usize::MAX, |s| s.pf_pos.saturating_sub(pf_before));
                    }
                    if co_scheduled
                        || (gpu_prefill_should_yield(
                            !feeds.is_empty(),
                            defer_decode,
                            slot_opt.as_ref(),
                        ) && !(serial && serial_rows < pf_interleave_rows()))
                    {
                        // New decoders join next tick; their first token was already emitted.
                        break;
                    }
                }
            }

            let pack_prefill_ns = pack_t.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);
            let pack_had_feeds = !feeds.is_empty();
            let mut decode_progress = unified_progress;
            let dec_t = packlog::on().then(Instant::now);

            // One batched decode launch for every slot already past prefill. The
            // token buffer round-trips through `obs.host.slot_tokens` so the
            // per-tick hot path allocates nothing.
            if !feeds.is_empty() {
                let mut toks = std::mem::take(&mut obs.host.slot_tokens);
                // Bounded device multi-step (plan stage 5): when enabled and EVERY
                // fed row uses unmodified greedy logits (the device advance uses the argmax
                // token), run a K-token quantum with one host sync and stream up to
                // K tokens per row, stopping a row as soon as handle_produced_token
                // frees it (mid-quantum EOS — extra device tokens past the stop
                // are discarded). Remaining output budgets cap K. Any sampling adjustment
                // falls through to the per-token path below.
                let use_pipe = e.pipe_enabled() && gpu_pipe_rows(&feeds, &slots);
                // Device-sampleable rows ride the quantum too when the sampler is loaded:
                // `plow_sample` runs between each decode and advance.
                let sampled_multi = e.multistep_sampling();
                let use_multi = !use_pipe
                    && steps > 1
                    && e.multistep_quantum().is_some()
                    && feeds.iter().all(|&(i, _)| {
                        slots[i]
                            .as_ref()
                            .map(|s| {
                                // A CFG pair rides one step per tick: K=8 vs 1 lost Chatterbox c16
                                // 17.25 vs 18.75 aps (the quantum overshoots the stop and delays
                                // prefill and first-chunk renders).
                                if s.cfg.is_some() {
                                    return false;
                                }
                                !s.plain_decode()
                                    && (gpu_argmax_eligible(&s.gen.params)
                                        || (sampled_multi && dev_sample_spec(s).is_some()))
                            })
                            .unwrap_or(true)
                    });
                if use_pipe {
                    // Look ahead only while every row owes a token past the one this tick
                    // completes; a stop the host cannot foresee costs one discarded step.
                    let lookahead = feeds.iter().all(|&(i, _)| {
                        slots[i].as_ref().is_some_and(|s| {
                            s.gen.max_tokens.max(1).saturating_sub(s.step) >= 2
                        })
                    });
                    let mut done = std::mem::take(&mut obs.host.pipe_tokens);
                    let t_call = crate::obs::host::on().then(Instant::now);
                    match e.pipe_step(&feeds, lookahead, &mut done) {
                        Ok(()) => {
                            let t_emit = host_engine_call(t_call, feeds.len(), tokens_this_tick);
                            decode_progress = completed_decode(&feeds, 1);
                            for &(i, token) in &done {
                                if slots[i].is_none() {
                                    continue;
                                }
                                tracing::debug!(token, slot = i, "gpu: token (pipelined)");
                                disconnected[i] |= gpu_emit_slot_token(
                                    &mut slots[i],
                                    &arena,
                                    bundle,
                                    token,
                                    &mut tokens_this_tick,
                                    stop.as_slice(),
                                );
                            }
                            host_emit_done(t_emit, tokens_this_tick);
                        }
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                error_code = ?err.device_code(),
                                fatal = err.is_fatal(),
                                fed = feeds.len(),
                                model = bundle.network(),
                                "gpu: pipelined decode failed"
                            );
                            note_fault(&mut tick_fault, &err);
                            fail_feeds(&mut slots, &feeds, &arena, &err);
                        }
                    }
                    obs.host.pipe_tokens = done;
                } else if use_multi {
                    let remaining = feeds
                        .iter()
                        .filter_map(|&(i, _)| slots[i].as_ref())
                        .map(|slot| slot.gen.max_tokens.max(1).saturating_sub(slot.step))
                        .min()
                        .unwrap_or(1);
                    let requested = multistep_requested(
                        remaining,
                        steps as usize,
                        e.multistep_quantum().unwrap_or(1),
                    );
                    let quantum = requested.min(e.multistep_quantum().unwrap_or(1));
                    let mut specs: Vec<crate::exec::gpu::DevSample> = Vec::new();
                    let mut draws: Vec<f32> = Vec::new();
                    for &(i, _) in &feeds {
                        if let Some(spec) = slots[i].as_ref().and_then(dev_sample_spec) {
                            if specs.is_empty() {
                                specs = vec![crate::exec::gpu::DevSample::greedy(); e.batch()];
                                draws = vec![0.0; e.batch() * quantum];
                            }
                            specs[i] = spec;
                            let slot = slots[i].as_ref().expect("fed slot");
                            for k in 0..quantum {
                                draws[i * quantum + k] = slot_rng01_at(slot, k);
                            }
                        }
                    }
                    let rng = |b: usize, k: usize| draws.get(b * quantum + k).copied().unwrap_or(0.0);
                    let sample = (!specs.is_empty())
                        .then_some((specs.as_slice(), &rng as &dyn Fn(usize, usize) -> f32));
                    let staged = gpu_stage_cfg(&mut *e, &feeds, &mut slots, requested);
                    let t_call = crate::obs::host::on().then(Instant::now);
                    let t_step = Instant::now();
                    let cut = || quantum_cut.fire();
                    let cut = quantum_cut.turn.is_some().then_some(&cut as &dyn Fn() -> bool);
                    let res = staged.and_then(|_| e.multi_step_sampled_at_most(&feeds, requested, sample, cut, &mut toks));
                    match res {
                        Ok(k) => {
                            obs.host.ride.observe_step(feeds.len(), t_step.elapsed().as_secs_f64() * 1e3 / k.max(1) as f64);
                            gpu_cfg_advance(&feeds, &mut slots, k);
                            let t_emit = host_engine_call(t_call, feeds.len(), tokens_this_tick);
                            decode_progress = completed_decode(&feeds, k);
                            for (ri, &(i, _)) in feeds.iter().enumerate() {
                                for s in 0..k {
                                    if slots[i].is_none() {
                                        break; // row stopped earlier this quantum
                                    }
                                    let token = toks[ri * k + s];
                                    tracing::debug!(token, slot = i, "gpu: token (multi-step)");
                                    disconnected[i] |= gpu_emit_slot_token(
                                        &mut slots[i],
                                        &arena,
                                        bundle,
                                        token,
                                        &mut tokens_this_tick,
                                        stop.as_slice(),
                                    );
                                }
                            }
                            host_emit_done(t_emit, tokens_this_tick);
                        }
                        Err(err) => {
                            tracing::warn!(
                                error = %err,
                                error_code = ?err.device_code(),
                                fatal = err.is_fatal(),
                                fed = feeds.len(),
                                model = bundle.network(),
                                "gpu: multi-step failed"
                            );
                            note_fault(&mut tick_fault, &err);
                            fail_feeds(&mut slots, &feeds, &arena, &err);
                        }
                    }
                } else {
                    // Device sampling (plan stage 4): when the engine has a sampler,
                // build a batch-wide spec array so eligible temperature>0 rows are
                // sampled on-device (token lands in in.ids, no vocab-row D2H); a
                // row is device-sampled iff temp>0 with no penalties/logit-bias
                // (those still need the host path). `dev_sampled` marks which rows
                // must NOT be host-resampled afterwards.
                let dev_specs: Option<Vec<crate::exec::gpu::DevSample>> = if e.dev_sample_enabled()
                {
                    let cap = e.batch();
                    let mut v = vec![crate::exec::gpu::DevSample::greedy(); cap];
                    for &(i, _) in &feeds {
                        if let Some(slot) = slots[i].as_ref() {
                            if let Some(spec) = dev_sample_spec(slot) {
                                v[i] = spec;
                            }
                        }
                    }
                    Some(v)
                } else {
                    None
                };
                let t_call = crate::obs::host::on().then(Instant::now);
                let t_step = Instant::now();
                let (cfg_dev, step_res) = match gpu_stage_cfg(&mut *e, &feeds, &mut slots, 1) {
                    Ok(d) => (d, e.step_slots_sampled(&feeds, dev_specs.as_deref(), &mut toks)),
                    Err(err) => (false, Err(err)),
                };
                match step_res {
                    Ok(()) => {
                        obs.host.ride.observe_step(feeds.len(), t_step.elapsed().as_secs_f64() * 1e3);
                        let t_emit = host_engine_call(t_call, feeds.len(), tokens_this_tick);
                        decode_progress = completed_decode(&feeds, 1);
                        let cfg_drawn = if cfg_dev {
                            gpu_cfg_advance(&feeds, &mut slots, 1);
                            true
                        } else {
                            match gpu_cfg_draws(&mut *e, &feeds, &mut slots, &mut toks) {
                                Ok(drawn) => drawn,
                                Err(err) => {
                                    note_fault(&mut tick_fault, &err);
                                    fail_feeds(&mut slots, &feeds, &arena, &err);
                                    false
                                }
                            }
                        };
                        if let Err(err) = gpu_batch_logprobs(
                            &mut *e,
                            feeds.iter().zip(toks.iter()).map(|(&(i, _), &t)| (i, i, t)),
                            &mut slots,
                        ) {
                            tracing::warn!(error = %err, "gpu: batched logprob stats failed; per-row path");
                        }
                        for (&(i, _), &argmax_tok) in feeds.iter().zip(toks.iter()) {
                            let slot_opt = &mut slots[i];
                            let was_dev = cfg_drawn && slot_opt.as_ref().is_some_and(|s| s.cfg.is_some())
                                || dev_specs
                                    .as_ref()
                                    .map(|_| slot_opt.as_ref().map_or(false, |s| dev_sample_spec(s).is_some()))
                                    .unwrap_or(false);
                            gpu_finish_and_emit_token(
                                &mut *e,
                                i,
                                i,
                                slot_opt,
                                &arena,
                                bundle,
                                argmax_tok,
                                was_dev,
                                &mut tokens_this_tick,
                                stop.as_slice(),
                                &mut tick_fault,
                                &mut disconnected,
                            );
                        }
                        host_emit_done(t_emit, tokens_this_tick);
                    }
                    Err(err) => {
                        // The batched launch failed — every fed slot loses.
                        tracing::warn!(
                            error = %err,
                            error_code = ?err.device_code(),
                            fatal = err.is_fatal(),
                            fed = feeds.len(),
                            model = bundle.network(),
                            "gpu: decode launch failed"
                        );
                        note_fault(&mut tick_fault, &err);
                        fail_feeds(&mut slots, &feeds, &arena, &err);
                    }
                }
            }
            obs.host.slot_tokens = toks;
        }
            if let Some(dt) = dec_t {
                let decode_ns = dt.elapsed().as_nanos() as u64;
                packlog::tick(
                    pack_prefill_ns,
                    decode_ns,
                    did_prefill,
                    feeds.len(),
                    decode_progress.map_or(0, |p| p.steps.get()),
                    tokens_this_tick,
                    pack_live,
                    pack_prefilling,
                );
                packlog::record(
                    pack_prefill_ns,
                    decode_ns,
                    did_prefill,
                    pack_had_feeds,
                    feeds.len(),
                );
            }
            return (
                slots,
                bufs,
                obs,
                tokens_this_tick,
                did_prefill,
                tick_fault,
                decode_progress,
            );
            })();
            for (slot, request) in result.0.iter().enumerate().take(e.batch()) {
                if request.is_none() && slot_free(&result.0, slot) {
                    e.retire_slot(slot, result.5.is_none() && !disconnected[slot]);
                }
            }
            return result;
        }

        // gfx950: B independent sequence slots, one decode dispatch for all of
        // them. Mux slot `i` IS engine slot `i`, so no owner map is needed —
        // the engine's slot table and this one are the same indices.
        //
        // A packet-bound mixed program can advance prefill and decode together.
        // Otherwise, a tick runs one isolated/packed prefill submission followed
        // by bounded decode. Those separate programs share scratch and run sequentially.
        //
        // Irrefutable in an hsa-only build (the enum then has one variant) and
        // refutable alongside `cuda` — the same arm has to compile as both.
        // Single-sequence engines (gfx950, CPU) share one tick body through
        // `SeqEngine`; the packed/multistep branches are AMD-only and the CPU
        // engine declines them (`None`), falling to whole-prompt prefill + step.
        #[cfg(any(feature = "hsa", feature = "cpu"))]
        if let Some(e) = guard.seq_engine_mut() {
            e.bind_engine_thread();
            let stop = Arc::clone(e.stop_ids());
            let b = e.batch();

            // `capacity` is `batch()`, so this is only reachable on a mismatch;
            // a loud rejection beats a hang.
            for slot_opt in slots.iter_mut().skip(b) {
                if let Some(taken) = slot_opt.take() {
                    tracing::warn!("amd: slot past engine batch rejected");
                    release_kv(&arena, taken.kv);
                    let _ =
                        taken
                            .respond
                            .try_send(StreamChunk::Err(crate::RuntimeError::Rejected(format!(
                                "AMD engine serves {b} sequence slots"
                            ))));
                }
            }

            // Client gone — don't spend a launch on a dead stream, and free its
            // engine slot so the next arrival can have it.
            for i in 0..b.min(slots.len()) {
                if slots[i]
                    .as_ref()
                    .map(|s| s.respond.is_closed())
                    .unwrap_or(false)
                {
                    if let Some(taken) = slots[i].take() {
                        release_kv(&arena, taken.kv);
                    }
                    e.release(i);
                }
            }

            // The gfx950 engine samples on device and the host never sees the
            // logit row, so there is no host resample to apply — every token is
            // the device argmax. Say so ONCE rather than let a `temperature`
            // the caller set be silently discarded: greedy output that claims to
            // be sampled is the failure mode worth being loud about.
            if slots
                .iter()
                .flatten()
                .any(|s| s.gen.params.temperature > 0.0)
            {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    tracing::warn!(
                        "amd: temperature > 0 requested, but the gfx950 engine samples \
                         greedily on device — serving the argmax. Penalties, top_p, \
                         top_k and logit_bias are ignored on this backend."
                    );
                });
            }

            // ONE prefill chunk per tick, oldest pending request first. Slot
            // indices are reused, so index order can starve an older high slot
            // when short requests repeatedly refill lower slots.
            //
            // PREFILL AND DECODE NOW SHARE THE TICK. This arm used to `return`,
            // which made a tick EITHER a prefill OR a decode and meant every
            // live decode stream stalled for the whole of someone else's
            // prefill — measured 49.3 tok/s through `serve` at concurrency 16
            // against 91.3 from the same packet under `amd-bench`, and TTFT
            // 3.3 s because N arrivals serialise into N prefill-only ticks.
            //
            // Falling through instead costs nothing and needs no kernel change,
            // and the ordering is what makes it sound:
            //
            //   * The prefill runs FIRST, so by the time `feeds` is built the
            //     new slot's first token is already in `out_ids` and it decodes
            //     in the same tick rather than waiting for the next one.
            //   * `prefill_slot` rebases the KV table onto the slot and restores
            //     base 0 before returning, and `decode_step_batched` refuses a
            //     non-zero base — so the decode below cannot run rebased.
            //   * The two programs share `in.ids`/`in.pos`/`in.kvlen` and the
            //     `act.*` scratch, which is safe ONLY because they alternate
            //     rather than overlap: each phase fully re-stages its own inputs
            //     (`seed_ids` + `decode_prepare_batched` on one side,
            //     `prefill_prepare` on the other) before it reads them.
            //
            // The separate prefill/decode fallback runs sequentially because it
            // shares input/activation buffers. The parked-row mask prevents a
            // mid-prefill KDA state from advancing during the decode dispatch.
            let mut did_prefill = false;
            let _tick = crate::obs::tick::begin();
            let rt = crate::config::RuntimeConfig::get();
            let slo_targets = rt.slo_targets();
            let slo_on = slo_targets.active();
            let slo_t0 = slo_on.then(Instant::now);
            let mut slo_pf_ms = 0.0f64;
            let decode_rows = slots[..b.min(slots.len())]
                .iter()
                .filter(|slot| slot.as_ref().is_some_and(|slot| slot.step > 0))
                .count();
            let has_decode = decode_rows > 0;
            let tick_max = amd_prefill_tick_cap(
                has_decode || co_scheduled,
                rt.pf_defer_decode && !co_scheduled,
                rt.pf_interleave_amd(),
            );
            if slots[..b.min(slots.len())]
                    .iter()
                    .flatten()
                    .filter(|slot| slot.step == 0)
                    .take(2)
                    .count()
                    == 2
            {
                for (i, slot_opt) in slots.iter_mut().enumerate().take(b) {
                    if e.resume_before_pack() {
                        if let Some(s) = slot_opt
                            .as_mut()
                            .filter(|s| s.step == 0 && s.pf_pos == 0 && s.resume > 0)
                        {
                            s.resume = if s.cfg.is_none() { e.resume_slot(i, s.resume) } else { 0 };
                            s.pf_pos = s.resume;
                            s.cached_tokens = s.resume;
                        }
                    }
                    let Some(slot) = slot_opt.as_ref().filter(|slot| slot.step == 0) else {
                        continue;
                    };
                    if let Err(err) = e.prepare_packed_prefill_slot(i, &slot.prompt_ids, tick_max) {
                        note_fault(&mut tick_fault, &err);
                        if let Some(taken) = slot_opt.take() {
                            release_kv(&arena, taken.kv);
                            let _ = taken.respond.try_send(StreamChunk::Err(err));
                        }
                        e.release(i);
                    }
                }
            }
            let terminal: Vec<_> = slots
                .iter()
                .enumerate()
                .take(b)
                .filter_map(|(i, slot)| {
                    let slot = slot.as_ref()?;
                    (slot.step == 0 && e.terminal_prefill_ready(i, &slot.prompt_ids)).then_some(i)
                })
                .collect();
            if !terminal.is_empty() {
                let feeds: Vec<_> = slots
                    .iter()
                    .enumerate()
                    .take(b)
                    .filter_map(|(i, slot)| {
                        if rt.pf_defer_decode {
                            return None;
                        }
                        let slot = slot.as_ref().filter(|s| s.parked_at.is_none())?;
                        slot.out_ids.last().map(|&token| (i, token))
                    })
                    .collect();
                let result = {
                    let members: Vec<_> = terminal
                        .iter()
                        .map(|&i| {
                            (
                                i,
                                slots[i]
                                    .as_ref()
                                    .expect("terminal slot")
                                    .prompt_ids
                                    .as_slice(),
                            )
                        })
                        .collect();
                    e.finish_prefill_batch(&feeds, &members)
                };
                match result {
                    Ok(output) => {
                        e.advance_prefill_turn(*terminal.last().expect("terminal slots"));
                        tracing::debug!(
                            prefill = terminal.len(),
                            decode = feeds.len(),
                            "AMD batched prefill completion"
                        );
                        for (i, token) in output {
                            if let Some(slot) = slots[i].as_mut() {
                                if slot.step == 0 {
                                    slot.pf_pos = slot.prompt_ids.len();
                                }
                            }
                            handle_produced_token(
                                &mut slots[i],
                                &arena,
                                bundle,
                                token,
                                1,
                                &mut tokens_this_tick,
                                Some(stop.as_slice()),
                            );
                            if slots[i].is_none() {
                                e.release(i);
                            }
                        }
                    }
                    Err(err) => {
                        note_fault(&mut tick_fault, &err);
                        let msg = err.to_string();
                        for i in terminal.into_iter().chain(feeds.iter().map(|&(i, _)| i)) {
                            if let Some(taken) = slots[i].take() {
                                release_kv(&arena, taken.kv);
                                let _ = taken
                                    .respond
                                    .try_send(StreamChunk::Err(fanout_err(&err, &msg)));
                            }
                            e.release(i);
                        }
                    }
                }
                return (slots, bufs, obs, tokens_this_tick, true, tick_fault, None);
            }
            // ---------------------------------------------------------------------------
            // UNIFIED TOKEN BATCH (plans/unified-token-batch.md §7).
            //
            // Placed ahead of the mixed arm and never active beside it — the engine loads one
            // route or the other. Two things differ from the mixed arm, and both are the point
            // of the route:
            //
            //  * It does NOT require a decode row. A step of one completing prompt is legal,
            //    which is what lets a prompt's first generated token come out of the launch
            //    that consumed the prompt instead of a second, decode-shaped pass.
            //  * A member may consume its prompt to the end. `token_batch_prefill_rows` does
            //    not hold the last token back, and the sampled ids for prompts that finish
            //    here follow the decode feeds in `output`.
            if !rt.pf_defer_decode
                && !slo_on
                && e.token_batch_rows(1, 0, 1).is_some()
            {
                let feeds: Vec<(usize, u32)> = slots
                    .iter()
                    .enumerate()
                    .take(b)
                    .filter_map(|(i, slot)| {
                        let slot = slot.as_ref().filter(|s| s.parked_at.is_none())?;
                        (slot.step > 0).then(|| (i, *slot.out_ids.last().expect("decode output")))
                    })
                    .collect();
                // A body can only pack a slot that already has a cursor, and a fresh request
                // acquires one otherwise only through the isolated path below — which a
                // prefix-cache hit would then run alone at the smallest rung that holds its
                // suffix. Seed every waiting slot first so the suffix is a candidate here.
                for (i, slot_opt) in slots.iter_mut().enumerate().take(b) {
                    if e.resume_before_pack() {
                        if let Some(s) = slot_opt
                            .as_mut()
                            .filter(|s| s.step == 0 && s.pf_pos == 0 && s.resume > 0)
                        {
                            s.resume = if s.cfg.is_none() { e.resume_slot(i, s.resume) } else { 0 };
                            s.pf_pos = s.resume;
                            s.cached_tokens = s.resume;
                        }
                    }
                    let Some(slot) = slot_opt
                        .as_ref()
                        .filter(|slot| slot.step == 0 && !slot.respond.is_closed())
                    else {
                        continue;
                    };
                    if e.next_prefill_rows(i).is_some() {
                        continue;
                    }
                    if let Err(err) = e.prepare_packed_prefill_slot(i, &slot.prompt_ids, tick_max) {
                        note_fault(&mut tick_fault, &err);
                        if let Some(taken) = slot_opt.take() {
                            release_kv(&arena, taken.kv);
                            let _ = taken.respond.try_send(StreamChunk::Err(err));
                        }
                        e.release(i);
                    }
                }
                // WHOLE CHUNKS ONLY, oldest first. A member's next chunk is taken as planned or
                // not at all: cutting a chunk to fit the body would run a slice of a sparse
                // 8192 step through a dense body, which at long context costs several times the
                // sparse launch it displaces. Chunks no body can hold stay with the planner arm
                // below (one isolated launch each), and never block the ones that fit.
                let capacity_for = |members: usize, rows: u32| {
                    let samples = feeds.len() + members;
                    e.token_batch_rows(samples, feeds.len(), rows as usize)
                        .map(|t| e.token_batch_prefill_capacity(t, feeds.len()).min(tick_max))
                };
                let mut candidates: Vec<(usize, Instant, u32)> = slots
                    .iter()
                    .enumerate()
                    .take(b)
                    .filter_map(|(i, slot)| {
                        let slot = slot.as_ref()?;
                        if slot.step != 0 || slot.respond.is_closed() {
                            return None;
                        }
                        let rows = e.token_batch_prefill_rows(i, &slot.prompt_ids, u32::MAX);
                        let fits = rows > 0
                            && rows <= tick_max
                            && capacity_for(1, rows).is_some_and(|capacity| {
                                rows <= capacity && e.token_batch_prefill_fits(i, capacity)
                            });
                        fits.then_some((i, slot.arrived, rows))
                    })
                    .collect();
                let (rotate, turn) = (rt.pf_rotate(), e.prefill_turn());
                candidates.sort_by_key(|&(slot, arrived, _)| {
                    let cap = b.max(1);
                    let distance = if rotate { (slot + cap - turn % cap) % cap } else { 0 };
                    (distance, arrived, slot)
                });
                // `leading` bounds the selection stage and is not known until the pack is
                // formed, so try the widest pack first and shrink. Assuming every member
                // completes over-counts `leading`, which under-counts the prefill capacity —
                // the safe direction: a plan that fits the assumed bucket also fits the real
                // one. The pool is the `take` oldest candidates, so shrinking drops the
                // youngest and every accepted pack is a prefix of the arrival order.
                let mut chosen: Option<(u32, Vec<(usize, u32)>)> = None;
                for take in (1..=candidates.len()).rev() {
                    let samples = feeds.len() + take;
                    let want: u32 = candidates[..take]
                        .iter()
                        .fold(0u32, |sum, c| sum.saturating_add(c.2));
                    let Some(rows) = e.token_batch_rows(samples, feeds.len(), want as usize) else {
                        continue;
                    };
                    let capacity = e.token_batch_prefill_capacity(rows, feeds.len()).min(tick_max);
                    let pack = amd_token_batch_pack(&candidates[..take], capacity);
                    if pack.len() != take {
                        continue;
                    }
                    // Every member must fit THIS pack's body (its attention kind and span
                    // admissibility), not only the narrower body it was vetted against alone.
                    if !pack.iter().all(|&(slot, _)| e.token_batch_member_fits_body(slot, rows)) {
                        continue;
                    }
                    // Which prompts FINISH here decides whether the step has an output segment
                    // at all: `S = 0` means none, and this route has no way to run a body
                    // without one, so such a pack is skipped rather than staged for the engine
                    // to refuse.
                    let completing = pack
                        .iter()
                        .filter(|&&(slot, take)| {
                            let frontier = e.prefill_frontier(slot).unwrap_or(0) as u32;
                            slots[slot].as_ref().is_some_and(|s| {
                                frontier.saturating_add(take) == s.prompt_ids.len() as u32
                            })
                        })
                        .count();
                    // ADMIT ONLY A STEP THAT ACTUALLY PACKS PREFILLS.
                    //
                    // `feeds + completing > 0` is the CORRECTNESS floor: without a sampled row
                    // there is no output segment to run. Two prefill members is the policy on
                    // top of it. Decode rows do not make a lone prefill a packed-prefill step:
                    // at 70K/C20, fusing decode with one full 8192-row member was 4.5% slower
                    // than the ordinary independent paths.
                    // A lone member measured -3.5% tok/s and -35% TTFT at 512 input (Gemma-4
                    // 31B, C1): it runs a far wider rung with nothing to pack it with.
                    if feeds.len() + completing > 0 && pack.len() >= 2 {
                        chosen = Some((rows, pack));
                        break;
                    }
                }
                let mut fell_through = false;
                if let Some((rows, pack)) = chosen {
                    // `(slot, id)` pairs — the delivery shape both backends' token-batch
                    // routes share; the buffer round-trips through `obs.host` so the hot
                    // path allocates nothing.
                    let mut tokens = std::mem::take(&mut obs.host.token_batch_tokens);
                    let started = Instant::now();
                    let mut finished: Vec<usize> = Vec::new();
                    let mut result = {
                        let members: Vec<_> = pack
                            .iter()
                            .map(|&(slot, take)| {
                                (
                                    slot,
                                    slots[slot]
                                        .as_ref()
                                        .expect("token-batch member")
                                        .prompt_ids
                                        .as_slice(),
                                    take,
                                )
                            })
                            .collect();
                        e.token_batch_step(rows, &feeds, &members, &mut tokens)
                    };
                    crate::obs::ttft::PREFILL.add(started.elapsed().as_nanos() as u64);
                    if result.is_ok() {
                        // Sampled ids that are not decode feeds belong to prompts that
                        // completed in this step.
                        finished.extend(
                            tokens
                                .iter()
                                .map(|&(slot, _)| slot as usize)
                                .filter(|slot| !feeds.iter().any(|&(f, _)| f == *slot)),
                        );
                        // Members that did NOT finish keep a cursor; the ones that did have
                        // consumed their whole prompt.
                        let pending: Vec<usize> = pack
                            .iter()
                            .map(|&(slot, _)| slot)
                            .filter(|slot| !finished.contains(slot))
                            .collect();
                        match amd_packed_frontier_updates(pending, |slot| e.prefill_frontier(slot))
                        {
                            Ok(updates) => {
                                for (slot, frontier) in updates {
                                    slots[slot].as_mut().expect("token-batch member").pf_pos =
                                        frontier;
                                }
                            }
                            Err(slot) => {
                                result = Err(crate::RuntimeError::Device(format!(
                                    "token-batch prefill slot {slot} lost its cursor after dispatch"
                                )))
                            }
                        }
                    }
                    if let Some(&(slot, _)) = pack.last() {
                        e.advance_prefill_turn(slot);
                    }
                    match result {
                        Ok(()) => {
                            tracing::debug!(
                                rows,
                                decode = feeds.len(),
                                prefill = pack.len(),
                                completed = finished.len(),
                                fires = true,
                                "AMD token batch"
                            );
                            // Delivery is by LOGICAL REQUEST: each pair names the slot it
                            // belongs to, in sample order — the decode feeds first, then the
                            // prompts that completed here.
                            for &(id, token) in tokens.iter() {
                                let slot = id as usize;
                                if finished.contains(&slot) {
                                    if let Some(s) = slots[slot].as_mut() {
                                        if s.step == 0 {
                                            s.pf_pos = s.prompt_ids.len();
                                            s.cached_tokens = e.cached_rows(slot);
                                        }
                                    }
                                }
                                seq_host_logprobs(&*e, slot, &mut slots[slot], token);
                                handle_produced_token(
                                    &mut slots[slot],
                                    &arena,
                                    bundle,
                                    token,
                                    1,
                                    &mut tokens_this_tick,
                                    Some(stop.as_slice()),
                                );
                                if slots[slot].is_none() {
                                    e.release(slot);
                                }
                            }
                        }
                        // A REFUSAL IS NOT A FAULT. `RuntimeError::Rejected` from this route
                        // is decided before anything reaches the device and leaves host state
                        // as it was, so the tick falls through to the ordinary path — the same
                        // "selection, not truncation" rule the D-class span limit follows. A
                        // device error is a different thing and still retires the slots.
                        Err(crate::RuntimeError::Rejected(reason)) => {
                            tracing::debug!(%reason, "AMD token batch declined this tick");
                            fell_through = true;
                        }
                        Err(err) => {
                            note_fault(&mut tick_fault, &err);
                            let msg = err.to_string();
                            for i in feeds
                                .iter()
                                .map(|&(slot, _)| slot)
                                .chain(pack.iter().map(|&(slot, _)| slot))
                            {
                                if let Some(taken) = slots[i].take() {
                                    release_kv(&arena, taken.kv);
                                    let _ = taken
                                        .respond
                                        .try_send(StreamChunk::Err(fanout_err(&err, &msg)));
                                }
                                e.release(i);
                            }
                        }
                    }
                    obs.host.token_batch_tokens = tokens;
                    if !fell_through {
                        return (slots, bufs, obs, tokens_this_tick, true, tick_fault, None);
                    }
                }
            }
            if has_decode
                && !rt.pf_defer_decode
                && !slo_on
                && e.mixed_step_rows(decode_rows, 1).is_some()
            {
                let mut candidates: Vec<_> = slots
                    .iter()
                    .enumerate()
                    .take(b)
                    .filter_map(|(i, slot)| {
                        let slot = slot.as_ref()?;
                        if slot.step != 0 || slot.respond.is_closed() {
                            return None;
                        }
                        let rows = e.mixed_prefill_rows(i, &slot.prompt_ids, tick_max);
                        (rows > 0).then_some((i, slot.arrived, rows))
                    })
                    .collect();
                let available = candidates
                    .iter()
                    .fold(0u32, |sum, candidate| sum.saturating_add(candidate.2))
                    .min(tick_max);
                if let Some(rows) = e.mixed_step_rows(decode_rows, available as usize) {
                    let prefill_capacity = rows.saturating_sub(decode_rows as u32);
                    candidates.retain(|&(slot, _, _)| e.mixed_prefill_fits(slot, prefill_capacity));
                    let capacity = prefill_capacity.min(tick_max);
                    let pack = amd_mixed_prefill_pack(
                        candidates,
                        capacity,
                        rt.pf_rotate(),
                        e.prefill_turn(),
                        b,
                    );
                    if !pack.is_empty() {
                        let feeds: Vec<_> = slots
                            .iter()
                            .enumerate()
                            .take(b)
                            .filter_map(|(i, slot)| {
                                let slot = slot.as_ref().filter(|s| s.parked_at.is_none())?;
                                (slot.step > 0)
                                    .then(|| (i, *slot.out_ids.last().expect("decode output")))
                            })
                            .collect();
                        let mut tokens = std::mem::take(&mut obs.host.slot_tokens);
                        tokens.resize(feeds.len(), 0);
                        let started = Instant::now();
                        let mut result = {
                            let members: Vec<_> = pack
                                .iter()
                                .map(|&(slot, take)| {
                                    (
                                        slot,
                                        slots[slot]
                                            .as_ref()
                                            .expect("mixed member")
                                            .prompt_ids
                                            .as_slice(),
                                        take,
                                    )
                                })
                                .collect();
                            e.mixed_step(rows, &feeds, &members, &mut tokens)
                        };
                        crate::obs::ttft::PREFILL.add(started.elapsed().as_nanos() as u64);
                        if result.is_ok() {
                            match amd_packed_frontier_updates(
                                pack.iter().map(|&(slot, _)| slot),
                                |slot| e.prefill_frontier(slot),
                            ) {
                                Ok(updates) => {
                                    for (slot, frontier) in updates {
                                        slots[slot].as_mut().expect("mixed member").pf_pos =
                                            frontier;
                                    }
                                }
                                Err(slot) => {
                                    result = Err(crate::RuntimeError::Device(format!(
                                        "mixed prefill slot {slot} lost its cursor after dispatch"
                                    )))
                                }
                            }
                        }
                        e.advance_prefill_turn(pack.last().expect("mixed pack").0);
                        match result {
                            Ok(()) => {
                                tracing::debug!(
                                    decode = feeds.len(),
                                    prefill = pack.len(),
                                    rows,
                                    "AMD mixed prefill/decode launch"
                                );
                                for (row, &(slot, _)) in feeds.iter().enumerate() {
                                    handle_produced_token(
                                        &mut slots[slot],
                                        &arena,
                                        bundle,
                                        tokens[row],
                                        1,
                                        &mut tokens_this_tick,
                                        Some(stop.as_slice()),
                                    );
                                    if slots[slot].is_none() {
                                        e.release(slot);
                                    }
                                }
                            }
                            Err(err) => {
                                tracing::warn!(error = %err, error_code = ?err.device_code(), fatal = err.is_fatal(), "AMD mixed prefill/decode failed");
                                note_fault(&mut tick_fault, &err);
                                let msg = err.to_string();
                                for slot in feeds
                                    .iter()
                                    .map(|&(slot, _)| slot)
                                    .chain(pack.iter().map(|&(slot, _)| slot))
                                {
                                    if let Some(taken) = slots[slot].take() {
                                        release_kv(&arena, taken.kv);
                                        let _ = taken
                                            .respond
                                            .try_send(StreamChunk::Err(fanout_err(&err, &msg)));
                                    }
                                    e.release(slot);
                                }
                            }
                        }
                        obs.host.slot_tokens = tokens;
                        // Mixed latency includes prefill and must not train decode-rung service estimates.
                        return (slots, bufs, obs, tokens_this_tick, true, tick_fault, None);
                    }
                }
            }
            // ONE PLANNER FOR EVERY BACKEND (`crate::sched::step`, the shape of vLLM's
            // `schedule()`): decodes first, then the mid-prefill requests' spans under one step
            // budget — a pack of several requests where the backend can run one, else one
            // request's planned chunk oldest-first — then fresh admissions into an untouched
            // budget. This arm only LOWERS the plan: a pack is one `advance_packed_prefill`, a
            // single span one `prefill_chunked_at_most`, and the decodes are the batched
            // dispatch below. What the engine cannot run it refuses by name; the plan is never
            // narrowed here.
            let pf_batch = true;
            let cap = b.min(slots.len()).min(u128::BITS as usize);
            let backend = e.step_backend();
            let now = Instant::now();
            let full_budget = tick_max.min(backend.step_budget);
            let candidates: Vec<crate::sched::step::Candidate> = (0..cap)
                .filter_map(|i| {
                    let slot = slots[i]
                        .as_ref()
                        .filter(|s| s.step == 0 && !s.respond.is_closed())?;
                    let arrival = arrival_key(slot.arrived, now);
                    if let Some(span) = pf_batch
                        .then(|| e.packable_prefill_span(i, full_budget))
                        .flatten()
                    {
                        return Some(crate::sched::step::Candidate {
                            span,
                            arrival,
                            packable: true,
                            planned: true,
                        });
                    }
                    let planned = e.next_prefill_rows(i);
                    let rows = planned.unwrap_or(u32::MAX);
                    Some(crate::sched::step::Candidate {
                        span: packet::dev::PrefillSpan {
                            row0: 0,
                            n_rows: rows,
                            slot: i as u32,
                            flags: 0,
                            kv_row0: 0,
                            kv_len: rows,
                            state_slot: i as u32,
                            program: 0,
                        },
                        arrival,
                        packable: false,
                        planned: planned.is_some(),
                    })
                })
                .collect();
            let step_tick = crate::sched::step::Tick {
                cap_rows: tick_max,
                packing: pf_batch,
                rotate: rt.pf_rotate(),
                turn: e.prefill_turn(),
                slots: cap,
            };
            let step = if slo_on {
                amd_slo_plan(&*e, &slots, backend, step_tick, slo_targets, &mut obs.slo, now)
            } else {
                crate::sched::step::plan(
                    backend,
                    step_tick,
                    (0..cap).filter(|&i| slots[i].as_ref().is_some_and(|s| s.step > 0)).map(|i| i as u32),
                    &candidates,
                    |program| e.prefill_prog_t(program as usize),
                    |program| e.packed_prefill_span_limit(program as usize),
                )
            };
            for launch in &step.launches {
                if launch.is_pack() {
                    let packed = launch.spans.clone();
                        let pk_t = packlog::on().then(Instant::now);
                        for span in &packed {
                            let slot = slots[span.slot as usize]
                                .as_ref()
                                .expect("packed candidate still occupies its slot");
                            crate::obs::ttft::QUEUE.add(slot.arrived.elapsed().as_nanos() as u64);
                        }
                        let t_pf = Instant::now();
                        let mut result = {
                            let members: Vec<_> = packed
                                .iter()
                                .map(|span| {
                                    let i = span.slot as usize;
                                    (
                                        i,
                                        slots[i]
                                            .as_ref()
                                            .expect("packed candidate still occupies its slot")
                                            .prompt_ids
                                            .as_slice(),
                                    )
                                })
                                .collect();
                            e.advance_packed_prefill(&members)
                        };
                        crate::obs::ttft::PREFILL.add(t_pf.elapsed().as_nanos() as u64);
                        crate::obs::tick::prefill(
                            t_pf.elapsed().as_nanos() as u64,
                            packed.iter().map(|span| span.n_rows).sum(),
                        );
                        if result.is_ok() {
                            match amd_packed_frontier_updates(
                                packed.iter().map(|span| span.slot as usize),
                                |i| e.prefill_frontier(i),
                            ) {
                                Ok(updates) => {
                                    for (i, frontier) in updates {
                                        slots[i]
                                            .as_mut()
                                            .expect("packed member still occupies its slot")
                                            .pf_pos = frontier;
                                    }
                                }
                                Err(i) => {
                                    result = Err(crate::RuntimeError::Device(format!(
                                        "packed prefill slot {i} lost its cursor after dispatch"
                                    )));
                                }
                            }
                        }
                        e.advance_prefill_turn(packed.last().expect("pack has members").slot as usize);
                        match result {
                            Ok(()) => {
                                // ONE info line the first time it actually fires. "Armed" (the
                                // load-time line in exec/amd.rs) and "firing" are different
                                // claims, and only the second one is evidence that a measured
                                // delta belongs to this feature. Everything after that stays at
                                // debug so the steady state is quiet.
                                static FIRED: std::sync::Once = std::sync::Once::new();
                                FIRED.call_once(|| {
                                    tracing::info!(
                                        spans = packed.len(),
                                        program = packed[0].program,
                                        "AMD packed prefill fired"
                                    )
                                });
                                tracing::debug!(spans = packed.len(), "AMD packed prefill advanced");
                                for span in &packed {
                                    let i = span.slot as usize;
                                    if let Some(s) = slots[i].as_mut() {
                                        s.cached_tokens = e.cached_rows(i);
                                    }
                                }
                                for (i, token) in e.take_packed_tokens() {
                                    if let Some(s) = slots[i].as_mut() {
                                        s.pf_pos = s.prompt_ids.len();
                                    }
                                    let t_tok = std::time::Instant::now();
                                    seq_host_logprobs(&*e, i, &mut slots[i], token);
                                    handle_produced_token(
                                        &mut slots[i],
                                        &arena,
                                        bundle,
                                        token,
                                        1,
                                        &mut tokens_this_tick,
                                        Some(stop.as_slice()),
                                    );
                                    crate::obs::ttft::FIRST_TOK.add(t_tok.elapsed().as_nanos() as u64);
                                    if slots[i].is_none() {
                                        e.release(i);
                                    }
                                }
                            }
                            Err(err) => {
                                tracing::warn!(
                                    error = %err,
                                    error_code = ?err.device_code(),
                                    fatal = err.is_fatal(),
                                    members = packed.len(),
                                    model = bundle.network(),
                                    "AMD packed prefill failed"
                                );
                                note_fault(&mut tick_fault, &err);
                                let msg = err.to_string();
                                for span in &packed {
                                    let i = span.slot as usize;
                                    if let Some(taken) = slots[i].take() {
                                        release_kv(&arena, taken.kv);
                                        let _ = taken
                                            .respond
                                            .try_send(StreamChunk::Err(fanout_err(&err, &msg)));
                                    }
                                    e.release(i);
                                }
                            }
                        }
                        if let Some(t) = pk_t {
                            packlog::record(t.elapsed().as_nanos() as u64, 0, true, false, 0);
                        }
                    did_prefill = true;
                } else {
                    let i = launch.spans[0].slot as usize;
                    if pf_batch {
                        e.advance_prefill_turn(i);
                    }
                    // AMD prefill and decode share scratch and run sequentially.
                    // This interval measures isolated prefill, not mixed-kernel overlap.
                    let pk_t = packlog::on().then(Instant::now);
                    // A retained session's rows: the engine keeps them and prefills only the
                    // suffix, or refuses and the prompt starts cold.
                    if let Some(s) = slots[i].as_mut().filter(|s| s.pf_pos == 0 && s.resume > 0) {
                        s.resume = if s.cfg.is_none() { e.resume_slot(i, s.resume) } else { 0 };
                        s.pf_pos = s.resume;
                        s.cached_tokens = s.resume;
                    }
                    let slot_ref = slots[i].as_ref().expect("found above");
                    // §TTFT: everything between `mux.submit` and this line — the
                    // dispatcher wake, the formation hold, admission, and the
                    // engine-thread handoff.
                    crate::obs::ttft::QUEUE.add(slot_ref.arrived.elapsed().as_nanos() as u64);
                    let t_pf = std::time::Instant::now();
                    // ONE CHUNK, not the whole prompt. `Ok(None)` means this slot has more chunks to
                    // go; it stays `step == 0` for a later tick. Planned against the FULL tick cap,
                    // never the planner's remainder: a fresh cursor built against a partial budget
                    // would plan the whole prompt in narrower rungs, and the planner only admits a
                    // planned chunk that fits.
                    let fresh = slo_on && e.next_prefill_rows(i).is_none();
                    let (pf, ran) = if slo_on {
                        let rows = launch.spans[0].n_rows;
                        match e.prefill_chunk_rows(i, &slot_ref.prompt_ids, tick_max, rows) {
                            Ok((token, ran)) => (Ok(token), ran),
                            Err(err) => (Err(err), None),
                        }
                    } else {
                        (e.prefill_chunked_at_most(i, &slot_ref.prompt_ids, tick_max), None)
                    };
                    if let Some(ran) = ran {
                        let ms = t_pf.elapsed().as_secs_f64() * 1e3;
                        slo_pf_ms += ms;
                        obs.slo.cost.observe_launch(ran.bucket, ran.clen, ran.c0, fresh, ms);
                    }
                    let frontier = e.prefill_frontier(i).unwrap_or(slot_ref.prompt_ids.len());
                    crate::obs::tick::prefill(
                        t_pf.elapsed().as_nanos() as u64,
                        frontier.saturating_sub(slot_ref.pf_pos) as u32,
                    );
                    if let Some(s) = slots[i].as_mut() { s.cached_tokens = e.cached_rows(i); }
                    crate::obs::ttft::PREFILL.add(t_pf.elapsed().as_nanos() as u64);
                    match pf {
                        Ok(None) => {
                            if let Some(slot) = slots[i].as_mut() {
                                slot.pf_pos = frontier;
                            }
                        }
                        Ok(Some(token)) => {
                            if let Some(s) = slots[i].as_mut() {
                                s.pf_pos = s.prompt_ids.len();
                            }
                            tracing::debug!(token, slot = i, "amd: prefill token");
                            let t_tok = std::time::Instant::now();
                            seq_host_logprobs(&*e, i, &mut slots[i], token);
                            handle_produced_token(
                                &mut slots[i],
                                &arena,
                                bundle,
                                token,
                                1,
                                &mut tokens_this_tick,
                                Some(stop.as_slice()),
                            );
                            crate::obs::ttft::FIRST_TOK.add(t_tok.elapsed().as_nanos() as u64);
                            if slots[i].is_none() {
                                e.release(i);
                            }
                        }
                        Err(err) => {
                            tracing::warn!(
                                slot = i,
                                error = %err,
                                error_code = ?err.device_code(),
                                fatal = err.is_fatal(),
                                model = bundle.network(),
                                "amd: prefill failed"
                            );
                            note_fault(&mut tick_fault, &err);
                            if let Some(taken) = slots[i].take() {
                                release_kv(&arena, taken.kv);
                                let _ = taken.respond.try_send(StreamChunk::Err(err));
                            }
                            e.release(i);
                        }
                    }
                    if let Some(t) = pk_t {
                        packlog::record(t.elapsed().as_nanos() as u64, 0, true, false, 0);
                    }
                    did_prefill = true;
                }
            }
            if did_prefill {
                let prefill_remains = slots[..b.min(slots.len())]
                    .iter()
                    .any(|s| s.as_ref().is_some_and(|s| s.step == 0));
                if amd_defer_decode(rt.pf_defer_decode, prefill_remains) {
                    tracing::debug!("amd: decode deferred while prefill remains");
                    return (slots, bufs, obs, tokens_this_tick, true, tick_fault, None);
                }
            }

            if did_prefill
                && slots.iter().enumerate().take(b).any(|(i, slot)| {
                    slot.as_ref().is_some_and(|slot| {
                        slot.step == 0 && e.terminal_prefill_ready(i, &slot.prompt_ids)
                    })
                })
            {
                return (slots, bufs, obs, tokens_this_tick, true, tick_fault, None);
            }

            // Decode: every live slot feeds the token it last produced.
            let feeds: Vec<(usize, u32)> = (0..b.min(slots.len()))
                .filter_map(|i| {
                    let s = slots[i].as_ref().filter(|s| s.parked_at.is_none())?;
                    Some((i, *s.out_ids.last()?))
                })
                .collect();
            let mut decode_progress = None;
            if feeds.is_empty() {
                return (
                    slots,
                    bufs,
                    obs,
                    tokens_this_tick,
                    did_prefill,
                    tick_fault,
                    None,
                );
            }
            let pk_t = packlog::on().then(Instant::now);
            let pk_rows = feeds.len();
            // §DSTEP owns the whole tick from here, so `TOKEN` is a real total
            // and not a sum of parts. One tick is one token on the TP path,
            // which is the only shape `PLOW_DECODE_BATCH=1` admits.
            let t_tick = crate::obs::dstep::begin_token();
            let remaining = feeds
                .iter()
                .filter_map(|&(i, _)| slots[i].as_ref())
                .map(|slot| slot.gen.max_tokens.saturating_sub(slot.out_ids.len()))
                .min()
                .unwrap_or(1);
            let requested = amd_multistep_requested(
                remaining,
                multi_step,
                crate::serve::policy::decode_k(true, 1) as usize,
            );
            let multi = e.multistep_quantum(&feeds, requested);
            let mut deferred = std::mem::take(&mut obs.host.slot_tokens);
            let t_dec = (crate::obs::tick::on() || slo_on).then(Instant::now);
            let t_call = crate::obs::host::on().then(Instant::now);
            let step_result = if let Some(quantum) = multi {
                let call_res = e.multi_step(&feeds, quantum, &mut deferred);
                let t_emit = host_engine_call(t_call, feeds.len(), tokens_this_tick);
                let res = call_res.and_then(|quantum| {
                    for &(i, _) in &feeds {
                        for step in 0..quantum {
                            if slots[i].is_none() {
                                break;
                            }
                            let token = deferred_token(&deferred, i, step, quantum)?;
                            tracing::debug!(token, slot = i, "amd: token (deferred read)");
                            handle_produced_token(
                                &mut slots[i],
                                &arena,
                                bundle,
                                token,
                                1,
                                &mut tokens_this_tick,
                                Some(stop.as_slice()),
                            );
                        }
                        if slots[i].is_none() {
                            e.release(i);
                        }
                    }
                    Ok(quantum)
                });
                host_emit_done(t_emit, tokens_this_tick);
                res
            } else {
                let call_res = e.step_batch(&feeds);
                let t_emit = host_engine_call(t_call, feeds.len(), tokens_this_tick);
                let res = call_res.map(|out| {
                    for (i, token) in out {
                        tracing::debug!(token, slot = i, "amd: token");
                        let t_stream = crate::obs::dstep::on().then(Instant::now);
                        seq_host_logprobs(&*e, i, &mut slots[i], token);
                        handle_produced_token(
                            &mut slots[i],
                            &arena,
                            bundle,
                            token,
                            1,
                            &mut tokens_this_tick,
                            Some(stop.as_slice()),
                        );
                        if let Some(t) = t_stream {
                            crate::obs::dstep::STREAM.add(t.elapsed().as_nanos() as u64);
                        }
                        if slots[i].is_none() {
                            e.release(i);
                        }
                    }
                    1
                });
                host_emit_done(t_emit, tokens_this_tick);
                res
            };
            obs.host.slot_tokens = deferred;
            if let Some(t) = t_dec {
                crate::obs::tick::decode(t.elapsed().as_nanos() as u64, feeds.len() as u32);
                if let (Some(t0), None, true) = (slo_t0, multi, step_result.is_ok()) {
                    let dec_ms = t.elapsed().as_secs_f64() * 1e3;
                    obs.slo.cost.observe_decode(feeds.len() as u32, did_prefill, dec_ms);
                    obs.slo.cost.observe_host(t0.elapsed().as_secs_f64() * 1e3 - slo_pf_ms - dec_ms);
                    if let Some(o) = obs.slo.last.take().filter(|o| o.predicted_ms > 0.0) {
                        let ratio = t0.elapsed().as_secs_f64() * 1e3 / o.predicted_ms;
                        obs.slo.cost.observe_margin(o.class, ratio);
                    }
                }
            }
            match step_result {
                Ok(quantum) => {
                    decode_progress = completed_decode(&feeds, quantum);
                }
                Err(err) => {
                    // The batched launch failed — every fed slot loses.
                    tracing::warn!(
                        error = %err,
                        error_code = ?err.device_code(),
                        fatal = err.is_fatal(),
                        fed = feeds.len(),
                        model = bundle.network(),
                        "amd: decode failed"
                    );
                    note_fault(&mut tick_fault, &err);
                    let msg = err.to_string();
                    for &(i, _) in &feeds {
                        if let Some(taken) = slots[i].take() {
                            release_kv(&arena, taken.kv);
                            let _ = taken
                                .respond
                                .try_send(StreamChunk::Err(fanout_err(&err, &msg)));
                        }
                        e.release(i);
                    }
                }
            }
            crate::obs::dstep::finish_token(t_tick);
            if let Some(t) = pk_t {
                packlog::record(0, t.elapsed().as_nanos() as u64, false, true, pk_rows);
            }
            return (
                slots,
                bufs,
                obs,
                tokens_this_tick,
                did_prefill,
                tick_fault,
                decode_progress,
            );
        }

        #[allow(unreachable_code)]
        {
            unreachable!("ServeEngine variant with no tick body")
        }
    }

    // Owner map for the batched path: row `b` in the SAMPLE_BATCH tile is the
    // request in `slots[owner_of_row[b]]`. Rebuilt each tick to skip idle
    // slots (compact packing so the bucket's B×vocab tile has no gaps).
    let batched = bucket.map(bucket_has_sample_batch).unwrap_or(false);

    if batched {
        if let (Some(bucket), Some(bufs_mut)) = (bucket, bufs.as_mut()) {
            let v = bufs_mut.vocab;
            // Compact the live slots into row order.
            let mut owner_of_row: Vec<usize> = Vec::new();
            for (i, s) in slots.iter().enumerate() {
                if s.is_some() {
                    owner_of_row.push(i);
                }
            }
            let b = owner_of_row.len();
            if b > 0 && v > 0 {
                // Per-tick reset for the FLASH-consumer trace before we
                // re-populate the indirection table.
                obs.clear_tick_traces();
                refresh_indirection(
                    &mut obs,
                    live_kv_rows(&slots),
                    &arena,
                    kv_pages_range.clone(),
                );

                obs.host.logits.clear();
                obs.host.logits.resize(b * v, 0.0);
                obs.host.slot_params.clear();
                obs.host.slot_rng01.clear();
                obs.host.slot_tokens.clear();
                obs.host.slot_tokens.resize(b, 0);
                obs.host.tokens.clear();

                for (row, &slot_idx) in owner_of_row.iter().enumerate() {
                    let slot = slots[slot_idx].as_ref().expect("owner is Some");
                    obs.host.slot_params.push(slot.gen.params.clone());
                    obs.host.slot_rng01.push(slot_rng01(slot));
                    let row_slice = &mut obs.host.logits[row * v..(row + 1) * v];
                    reference_logits_row(&slot.prompt_ids, &slot.out_ids, row_slice);
                }

                match state.step_batch(bucket, &bufs_mut.pool, &mut bufs_mut.streams, &mut obs) {
                    Ok(executed) => {
                        // The per-tick executed count is shared across all
                        // rows — attribute it evenly for the per-slot total.
                        let per_slot_exec = if b > 0 { executed / b } else { 0 };
                        // Swap slot_tokens out (zero-alloc) before the mutable
                        // per-slot loop; swap back after so capacity is reused.
                        let mut produced: Vec<u32> = std::mem::take(&mut obs.host.slot_tokens);
                        for (row, &slot_idx) in owner_of_row.iter().enumerate() {
                            let token = produced[row];
                            handle_produced_token(
                                &mut slots[slot_idx],
                                &arena,
                                bundle,
                                token,
                                per_slot_exec,
                                &mut tokens_this_tick,
                                None,
                            );
                        }
                        // Return capacity to obs for the next tick.
                        produced.clear();
                        obs.host.slot_tokens = produced;
                    }
                    Err(e) => {
                        // Batched exec failed — every live row loses. The
                        // error message is duplicated per slot (RuntimeError
                        // doesn't Clone) since each waiter needs its own copy.
                        tracing::warn!(error = %e, "mux: batched exec failed");
                        let msg = e.to_string();
                        for &slot_idx in &owner_of_row {
                            if let Some(slot) = slots[slot_idx].take() {
                                release_kv(&arena, slot.kv);
                                let _ = slot.respond.try_send(StreamChunk::Err(
                                    crate::RuntimeError::Msg(msg.clone()),
                                ));
                            }
                        }
                    }
                }
            }
            return (slots, bufs, obs, tokens_this_tick, false, tick_fault, None);
        }
    }

    // Fallback: per-slot serial ticks against a scalar SAMPLE (phase 1).
    // Multi-step: produce up to `steps` tokens per slot before returning
    // control to the dispatcher (SGLang overlap scheduling). Early-exit if
    // every slot finishes within the window.
    for _step in 0..steps {
        if slots.iter().all(|s| s.is_none()) {
            break;
        }
        // Refresh indirection once per step for the live shape; slots that
        // finish mid-loop drop out but entries stay valid until next refresh.
        obs.clear_tick_traces();
        refresh_indirection(
            &mut obs,
            live_kv_rows(&slots),
            &arena,
            kv_pages_range.clone(),
        );
        for slot_opt in slots.iter_mut() {
            let Some(slot) = slot_opt.as_mut().filter(|s| s.parked_at.is_none()) else {
                continue;
            };

            obs.host.params = slot.gen.params.clone();

            let token_res: Result<(u32, usize)> =
                if let (Some(bucket), Some(bufs)) = (bucket, bufs.as_mut()) {
                    state.step_token(
                        bucket,
                        &bufs.pool,
                        &mut bufs.streams,
                        &mut obs,
                        &slot.prompt_ids,
                        &slot.out_ids,
                        slot.step,
                        bufs.vocab,
                        slot.gen.seed,
                    )
                } else {
                    // No bucket in the bundle — direct-sample against reference
                    // logits. Matches the fallback in generate_with_bucket.
                    obs.host.tokens.clear();
                    obs.host.rng01 = crate::serve::seeded_unit_with(
                        &slot.prompt_ids,
                        &slot.out_ids,
                        slot.step,
                        slot.gen.seed,
                    );
                    crate::serve::reference_logits(
                        &slot.prompt_ids,
                        &slot.out_ids,
                        vocab,
                        &mut obs.host.logits,
                    );
                    let tok = crate::text::sample::sample(
                        &obs.host.logits,
                        &obs.host.params,
                        None,
                        obs.host.rng01,
                    );
                    Ok((tok, 0))
                };

            match token_res {
                Ok((token, exec)) => {
                    handle_produced_token(
                        slot_opt,
                        &arena,
                        bundle,
                        token,
                        exec,
                        &mut tokens_this_tick,
                        None,
                    );
                }
                Err(e) => {
                    tracing::warn!(error = %e, "mux: step_token failed");
                    if let Some(slot) = slot_opt.take() {
                        release_kv(&arena, slot.kv);
                        let _ = slot.respond.try_send(StreamChunk::Err(e));
                    }
                }
            }
        }
    }

    (slots, bufs, obs, tokens_this_tick, false, tick_fault, None)
}

#[cfg_attr(not(feature = "hsa"), allow(dead_code))]
/// Decode feeds: every live slot past prefill, with its last token.
#[cfg(feature = "cuda")]
fn gpu_decode_feeds(slots: &[Option<Slot>], cap: usize) -> Vec<(usize, u32)> {
    let mut feeds = Vec::with_capacity(cap);
    for (i, s) in slots.iter().enumerate().take(cap) {
        let Some(s) = s.as_ref() else { continue };
        if s.step == 0 || s.parked_at.is_some() {
            continue;
        }
        let last = *s.out_ids.last().expect("step > 0 implies output");
        feeds.push((i, last));
        // The CFG partner steps on the same token.
        if s.cfg.is_some() && i + 1 < cap {
            feeds.push((i + 1, last));
        }
    }
    feeds
}

/// Whether every fed row can run in the decode pipeline: the device advance feeds the
/// argmax token, so a row that needs host sampling cannot.
#[cfg(feature = "cuda")]
fn gpu_pipe_rows(feeds: &[(usize, u32)], slots: &[Option<Slot>]) -> bool {
    !feeds.is_empty()
        && feeds.iter().all(|&(i, _)| {
            slots[i]
                .as_ref()
                .map(|s| gpu_argmax_eligible(&s.gen.params) && !s.plain_decode())
                .unwrap_or(true)
        })
}

fn deferred_token(tokens: &[u32], slot: usize, step: usize, quantum: usize) -> Result<u32> {
    tokens
        .get(slot.saturating_mul(quantum).saturating_add(step))
        .copied()
        .ok_or_else(|| {
            crate::RuntimeError::Device(format!(
                "deferred token ring missing slot {slot} step {step} at quantum {quantum}"
            ))
        })
}

/// Whether a tick's wall time is a valid **decode** service sample. Prefill
/// ticks are excluded: a chunk-interleaved prefill tick is bounded by design
/// (`PLOW_PF_INTERLEAVE` rows) and a long prompt runs MANY of them, so feeding
/// them to the service EWMA reports a decode tick that costs an order of
/// magnitude more than it does. The rung controller floors its SLO at eight of
/// these (`RungController::decide`), so a poisoned sample moves the admission
/// window on evidence from the wrong phase. Free-standing for tests.
fn service_sample(ms: f64, did_prefill: bool) -> Option<f64> {
    (ms > 0.0 && !did_prefill).then_some(ms)
}

/// A prefill launch's fixed cost, in rows.
#[cfg(feature = "cuda")]
fn pf_chunk_cost_rows() -> usize {
    crate::exec::gpu::PF_CHUNK_COST_ROWS
}

/// Reads `RuntimeConfig::get().pf_interleave_rows()`.
#[cfg(feature = "cuda")]
fn pf_interleave_rows() -> usize {
    crate::config::RuntimeConfig::get().pf_interleave_rows()
}

/// Packed prefill rows per launch while requests decode, and the launch rows riders included:
/// `PLOW_PF_INTERLEAVE` when set (`0` pins the widest launch), else the objective's width
/// (`policy::prefill_launch_rows`). The objective's width bounds the whole launch like a ladder
/// top: bounding only the prompt rows ran 2048 of them + 3 riders as a 4096-row launch (E4B c64,
/// 31 of 98 launches; tok/s -14%).
#[cfg(feature = "cuda")]
fn pf_launch_rows(request: usize, top: usize) -> (usize, usize) {
    match crate::config::RuntimeConfig::get().pf_interleave {
        Some(_) => (pf_interleave_rows().min(top), top),
        None => {
            let rows = crate::serve::policy::prefill_launch_rows(request, top);
            (rows, rows)
        }
    }
}

/// Share of the KV-capacity budget the live sequences reserve (0 without a budget).
fn kv_used(kv_budget: Option<crate::sched::admission::KvBudget>, slots: &[Option<Slot>]) -> f64 {
    kv_budget.filter(|b| b.max_rows() > 0).map_or(0.0, |b| {
        let rows: u64 = slots
            .iter()
            .flatten()
            .map(|s| reserved_kv_rows(s.prompt_ids.len(), s.gen.max_tokens, s.out_ids.len()))
            .sum();
        rows as f64 / b.max_rows() as f64
    })
}

/// Prefill rows for one launch, from the queue (the latency objective's packing).
///
/// `rows` are the waiting prompts' offered rows, oldest first. A launch costs a fixed
/// `chunk_cost` rows of time plus its rows, and every prompt packed into it finishes when the
/// launch does. So packing prompt `j + 1` (r rows) delays the `j` prompts already in by r rows and
/// saves the `n - j` prompts not yet in one fixed cost each: it joins while
/// `j * r < (n - j) * chunk_cost`. Short prompts share a launch, long ones run alone and oldest
/// first, and a deeper queue packs more. Measured on h100-sxm5, Gemma-4-12B launch ms at
/// 128/512/1024/2048/4096 rows 16.8/26.6/45.2/86.6/172.3: four 1024-row prompts packed all finish
/// at 172 ms, alone they finish at 45/90/135/180.
///
/// A prompt no launch can hold whole (`r >= bound`) still FILLS this one, as the static bound
/// does: stopping short there ran a long prompt's tail as two padded launches instead of one full
/// one (12B 15000 in, C4: TPOT 25.1 -> 26.6 ms, 105.8 -> 102.0 tok/s). A shorter prompt that does
/// not fit waits for the next launch: splitting it costs it a launch.
#[cfg(feature = "cuda")]
fn queue_pack_rows(rows: &[usize], chunk_cost: usize, bound: usize) -> usize {
    let Some((&first, rest)) = rows.split_first() else {
        return bound;
    };
    let mut total = first.min(bound);
    for (j, &r) in rest.iter().enumerate() {
        let (packed, waiting) = (j + 1, rows.len() - (j + 1));
        if total + r > bound {
            return if r >= bound { bound } else { total };
        }
        if packed * r >= waiting * chunk_cost {
            break;
        }
        total += r;
    }
    total
}

/// Whether a launch of `bucket` prefill rows trims its prefill so `decode_rows` riders fit, rather
/// than running in the `spilled` bucket that holds both. A wide spill is a padded rung, not padded
/// rows: 12B on H100 ran 1024 + riders in 1088 at 42.9 ms against 32.4 ms for 1024, and 4096 + 63
/// in 4160 at 127.0 against 117.1 ms; the trimmed rows join the next launch. Below
/// `pf_chunk_cost_rows()` a spill costs less than the tail launch a trim can leave. So does a
/// narrow spill whose trimmed `rows` tail no other waiting row (`beyond`) would join: 26B at C4
/// trimmed 1024 + 3 riders to 1021 rows (31.1 ms) and paid a lone 3-row tail launch (16 ms), where
/// 1088 holds all of it in 32.6 ms.
#[cfg(feature = "cuda")]
fn trim_for_riders(bucket: usize, spilled: usize, decode_rows: usize, rows: usize, beyond: usize) -> bool {
    let cost = pf_chunk_cost_rows();
    let tail = (rows + decode_rows).saturating_sub(bucket);
    let lone_tail = tail + beyond < cost && spilled - bucket < cost;
    decode_rows > 0 && spilled > bucket && bucket >= cost && !lone_tail
}

/// Prompt rows one model may consume while holding its device turn, when there
/// is no prefill object and the prompt is fed token by token. A bound on
/// launches per turn rather than on a bucket's rows: ~256 decode-shaped
/// launches is a turn hold in the low milliseconds, which is the same order as
/// the quantum's worth of decode ticks it is competing with.
#[cfg(feature = "cuda")]
const CO_SCHED_DECODE_ONLY_ROWS: usize = 256;

/// Prompt rows to consume in one co-scheduled tick.
///
/// `bucket` is `GpuEngine::pf_max_rows()`, which is 0 EXACTLY when the engine
/// has no prefill object — and that is also the only case where the caller
/// feeds the prompt token by token. Flooring the whole expression at 1 (the
/// obvious reading) therefore made every such prompt advance ONE token per
/// tick under `--co-sched rr`, each with a full turn acquire/release: a 4k
/// prompt became 4096 ticks. `pf_interleave_rows()` is no help on its own
/// either — it is `usize::MAX` when interleaving is unbounded, which would put
/// the entire prompt back inside one turn.
#[cfg(feature = "cuda")]
fn co_sched_prefill_rows(bucket: usize, interleave: usize) -> usize {
    if bucket == 0 {
        interleave.min(CO_SCHED_DECODE_ONLY_ROWS)
    } else {
        bucket.min(interleave)
    }
    .max(1)
}

/// PX-17 throughput mode: with `--pf-defer-decode` /
/// `PLOW_PF_DEFER_DECODE=1`, a tick that still has
/// ANY slot mid-prefill runs its prefill chain to completion and skips the
/// decode launch entirely, so every decode tick later runs at the full batch.
///
/// This is the scheduler-only bound on what prefill⊕decode FUSION can win. A
/// decode launch costs `a + b*B` — `a` (the 12 GiB weight re-read plus launch
/// turnaround) is paid whatever `B` is. Interleaving spends `a` on ~992 ticks at
/// an average `B` well under the engine batch; deferring spends it on the
/// minimum number of full-batch ticks instead. Fusion attacks the same `a` from
/// the other side (fold it into the prefill launch that is running anyway), so
/// the two bound each other.
///
/// Costs streaming latency: no token leaves the server until every prompt is
/// resident. Off by default — this is a measurement/throughput knob, not a
/// serving default.
#[cfg(feature = "cuda")]
fn pf_defer_decode() -> bool {
    crate::config::RuntimeConfig::get().pf_defer_decode
}

#[cfg(feature = "cuda")]
fn gpu_prefill_should_yield(has_feeds: bool, defer_decode: bool, slot: Option<&Slot>) -> bool {
    has_feeds || (!defer_decode && slot.is_some_and(|s| s.step > 0 && !s.respond.is_closed()))
}

#[cfg(any(feature = "hsa", feature = "cpu"))]
fn amd_prefill_tick_cap(has_decode: bool, defer_decode: bool, interleave: u32) -> u32 {
    if !has_decode || defer_decode || interleave == 0 {
        u32::MAX
    } else {
        interleave
    }
}

#[cfg(any(feature = "hsa", feature = "cpu"))]
fn amd_defer_decode(enabled: bool, prefill_remains: bool) -> bool {
    enabled && prefill_remains
}

/// A quantum of whole token groups when it holds at least one (Veena's 7-code frames: 8 -> 7).
/// Measured H100 Veena served: c1 TTFA 100.0 -> 89.4 ms, stream c8 19.1 -> 20.2 aps.
fn group_aligned(steps: u32, group: usize) -> u32 {
    let g = group as u32;
    if g > 1 && steps >= g {
        steps / g * g
    } else {
        steps
    }
}

fn multistep_requested(remaining: usize, scheduler_steps: usize, configured: usize) -> usize {
    remaining.min(scheduler_steps).min(configured.max(1))
}

/// The AMD deferred-read decode quantum to ask for.
///
/// Deliberately NOT `multistep_requested`: that one is bounded by `MultiStep::for_batch`, which
/// collapses to a single step above B=8 on the premise that a wide batch already amortises the
/// launch. The premise belongs to the NVIDIA device-multistep object. This path issues one
/// host-visible dispatch per token, so under continuous batching — where B is essentially always
/// above 8 — the quantum was ALWAYS 1 and decode paid a full host turnaround on every token.
///
/// Arming a quantum while a DIFFERENT slot is mid-prefill is safe, and by construction rather
/// than by luck: `feeds` carries only slots that have already produced a token, the engine
/// clears a slot's prefill cursor at the instant that token is produced, and `multi_step`
/// advances nothing but the slots it is fed. A prefill cursor moves only when the host calls
/// `prefill_chunk_rows` on a later tick. The quantum therefore changes WHEN the next prefill
/// chunk is issued — a TTFT tradeoff for the prefilling slot, governed by `--pf-defer-decode`
/// and the interleave bound — never WHETHER it is correct.
///
/// What remains are the bounds that mean something: the client's own remaining output, the
/// configured knob, and (applied downstream by `decode_quantum`) `DEFERRED_TOKEN_MAX_STEPS`.
#[cfg(any(feature = "hsa", feature = "cpu"))]
fn amd_multistep_requested(remaining: usize, enabled: bool, configured: usize) -> usize {
    if !enabled {
        return 1;
    }
    remaining.min(configured.max(1))
}

#[cfg(any(feature = "hsa", feature = "cpu"))]
fn amd_prefill_pick(
    candidates: impl IntoIterator<Item = (usize, Instant)>,
    fair: bool,
    start: usize,
    cap: usize,
) -> Option<usize> {
    let cap = cap.max(1);
    let start = start % cap;
    candidates
        .into_iter()
        .min_by(|&(slot_a, arrived_a), &(slot_b, arrived_b)| {
            if fair {
                ((slot_a + cap - start) % cap).cmp(&((slot_b + cap - start) % cap))
            } else {
                (arrived_a, slot_a).cmp(&(arrived_b, slot_b))
            }
        })
        .map(|(slot, _)| slot)
}

/// The step plan under `PLOW_TBT_SLO_MS` / `PLOW_TTFT_SLO_MS` (`crate::sched::slo`): isolated
/// launches only, each chunk sized in milliseconds against the online tick cost model.
#[cfg(any(feature = "hsa", feature = "cpu"))]
fn amd_slo_plan(
    e: &dyn crate::serve::engine::SeqEngine,
    slots: &[Option<Slot>],
    backend: crate::sched::step::Backend,
    tick: crate::sched::step::Tick,
    targets: crate::sched::slo::Targets,
    state: &mut crate::sched::slo::SloState,
    now: Instant,
) -> crate::sched::step::Plan {
    use crate::sched::slo::{Ladder, SloCandidate};
    let ladder = state.ladder.take().unwrap_or_else(|| Ladder {
        rungs: e.prefill_ladder(),
        tail_sparse_ctx: crate::config::RuntimeConfig::get().amd_tail_sparse_ctx(),
        tail_sparse_min_pairs: crate::config::RuntimeConfig::get().amd.tail_sparse_min_pairs,
    });
    let candidates: Vec<SloCandidate> = (0..tick.slots)
        .filter_map(|i| {
            let slot = slots[i].as_ref().filter(|s| s.step == 0 && !s.respond.is_closed())?;
            let planned = e.next_prefill_rows(i);
            let rows = planned.unwrap_or(u32::MAX);
            let prior = u32::try_from(e.prefill_frontier(i).unwrap_or(0)).unwrap_or(u32::MAX);
            Some(SloCandidate {
                base: crate::sched::step::Candidate {
                    span: packet::dev::PrefillSpan {
                        row0: 0,
                        n_rows: rows,
                        slot: i as u32,
                        flags: 0,
                        kv_row0: 0,
                        kv_len: rows,
                        state_slot: i as u32,
                        program: 0,
                    },
                    arrival: arrival_key(slot.arrived, now),
                    packable: false,
                    planned: planned.is_some(),
                },
                prior,
                remaining: u32::try_from(slot.prompt_ids.len())
                    .unwrap_or(u32::MAX)
                    .saturating_sub(prior),
                age_ms: now.saturating_duration_since(slot.arrived).as_secs_f64() * 1e3,
            })
        })
        .collect();
    let (plan, outcome) = crate::sched::slo::plan_tick(
        targets,
        backend,
        tick,
        (0..tick.slots)
            .filter(|&i| slots[i].as_ref().is_some_and(|s| s.step > 0))
            .map(|i| i as u32),
        &candidates,
        &ladder,
        &state.cost,
        &mut state.skipped,
        |program| e.prefill_prog_t(program as usize),
        |program| e.packed_prefill_span_limit(program as usize),
    );
    if crate::obs::tick::on() {
        let launches: Vec<String> = plan
            .launches
            .iter()
            .flat_map(|l| &l.spans)
            .map(|s| {
                let prior = candidates
                    .iter()
                    .find(|c| c.base.span.slot == s.slot)
                    .map_or(0, |c| c.prior);
                format!("{}:{}@{}", s.slot, s.n_rows, prior)
            })
            .collect();
        eprintln!(
            "SLOPLAN budget={:.1} pred={:.1} margin={:.3} class={:?} progress={} k={} decode_rows={} waiting={} launches=[{}]",
            outcome.budget_ms,
            outcome.predicted_ms,
            outcome.margin,
            outcome.class,
            outcome.progress,
            outcome.k,
            plan.decodes.len(),
            candidates.len(),
            launches.join(",")
        );
    }
    state.ladder = Some(ladder);
    state.last = Some(outcome);
    plan
}

/// Members of one token-batch body from an ordered candidate pool: every candidate whose whole
/// next chunk fits what is left of `capacity`, in pool order, none of them cut. A chunk that
/// does not fit is skipped, not sliced — the planner's whole-span rule.
#[cfg(any(feature = "hsa", feature = "cpu"))]
fn amd_token_batch_pack(candidates: &[(usize, Instant, u32)], capacity: u32) -> Vec<(usize, u32)> {
    let mut budget = capacity;
    let mut selected = Vec::new();
    for &(slot, _, rows) in candidates {
        if rows > 0 && rows <= budget {
            selected.push((slot, rows));
            budget -= rows;
        }
    }
    selected
}

#[cfg(any(feature = "hsa", feature = "cpu"))]
fn amd_mixed_prefill_pack(
    mut candidates: Vec<(usize, Instant, u32)>,
    mut budget: u32,
    fair: bool,
    start: usize,
    cap: usize,
) -> Vec<(usize, u32)> {
    let mut selected = Vec::new();
    while budget > 0 {
        let Some(slot) = amd_prefill_pick(
            candidates
                .iter()
                .filter(|candidate| candidate.2 > 0)
                .map(|&(slot, arrived, _)| (slot, arrived)),
            fair,
            start,
            cap,
        ) else {
            break;
        };
        let index = candidates
            .iter()
            .position(|candidate| candidate.0 == slot)
            .expect("selected candidate");
        let (_, _, rows) = candidates.remove(index);
        let take = rows.min(budget);
        selected.push((slot, take));
        budget -= take;
    }
    selected
}

/// Form one ragged AMD prefill pack from already-planned request spans.
///
/// All members use the first selected span's compiled program. Rows are packed
/// densely under `row_budget`; KV and recurrent coordinates remain request-local.
/// Invalid descriptors are excluded so the caller can retain the isolated path.
#[cfg(any(feature = "hsa", feature = "cpu"))]
fn amd_prefill_pack(
    candidates: impl IntoIterator<Item = packet::dev::PrefillSpan>,
    row_budget: u32,
    start: usize,
    cap: usize,
    program_rows: impl Fn(u32) -> Option<u32>,
    program_span_limit: impl Fn(u32) -> u32,
) -> Vec<packet::dev::PrefillSpan> {
    let mut pack = crate::sched::prefill::admit(
        candidates,
        row_budget,
        start,
        cap,
        crate::sched::prefill::SpanPolicy::Whole,
        program_rows,
    );
    // The D-class limit (plans/unified-token-batch.md §5.4) consulted BEFORE cursors move, so
    // the tick offers a legal plan rather than one `stage_packed_prefill` will refuse. The
    // refusal there stays as the backstop; this is what keeps it from being a liveness bug.
    if let Some(program) = pack.spans().first().map(|span| span.program) {
        pack.limit_spans(program_span_limit(program) as usize);
    }
    pack.spans().to_vec()
}

#[cfg(any(feature = "hsa", feature = "cpu"))]
fn amd_packed_frontier_updates(
    members: impl IntoIterator<Item = usize>,
    mut frontier: impl FnMut(usize) -> Option<usize>,
) -> std::result::Result<Vec<(usize, usize)>, usize> {
    members
        .into_iter()
        .map(|slot| frontier(slot).map(|value| (slot, value)).ok_or(slot))
        .collect()
}

/// Arrival order key for the step planner: smaller is older.
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
fn arrival_key(arrived: Instant, now: Instant) -> u64 {
    u64::MAX - u64::try_from(now.saturating_duration_since(arrived).as_nanos()).unwrap_or(u64::MAX)
}

/// RTX-12 chunked packing: per-REQUEST cap on the prefill rows one request may
/// contribute to a single batched launch. `PLOW_PF_CHUNK=C` clamps each waiting
/// request's slice to `C` rows so the `per_launch` budget (`PLOW_PF_INTERLEAVE`)
/// is shared by up to `R ≈ budget/C` requests instead of monopolized by the
/// first big prompt. `0` = uncapped = today's byte-identical behaviour (the A/B
/// canary; packing is numerics-neutral, so C only changes which requests share
/// a launch, never any request's tokens). Default `0` (off), an expert override.
/// Unset and `0` both mean uncapped, so the default IS the zero sentinel.
/// Reads `RuntimeConfig::get().pf_chunk_rows()`.
#[cfg(feature = "cuda")]
fn pf_chunk_rows() -> usize {
    crate::config::RuntimeConfig::get().pf_chunk_rows()
}

/// Pack prompt rows and optional decode feeds under the prefill quantum. Compact outputs
/// stop the pass before another launch can overwrite their logits. The legacy
/// route leaves the last prompt row for decode.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gpu_prefill_batched_pass(
    e: &mut crate::exec::gpu::GpuEngine,
    slots: &mut [Option<Slot>],
    cap: usize,
    arena: &Option<SharedKvState>,
    cold: bool,
    bounded_tick: bool,
    completed: &mut Vec<(usize, u32)>,
    feeds: &mut Vec<(usize, u32)>,
    unified_output: &mut Vec<(u32, u32)>,
    dev_sampled: &mut bool,
    ride: &mut crate::sched::ride::RideCost,
) -> Option<crate::DeviceErrorInfo> {
    use crate::exec::gpu::PfBatchReq;

    completed.clear();
    *dev_sampled = false;
    let compact = e.has_packed_terminal();
    let mut unified = e.token_batch_enabled();
    // Whether this tick's decode rows may still leave the launch for their own step
    // (`sched::ride`): decided once, when the launch's bucket is known.
    let mut ride_open = unified && !feeds.is_empty();
    let withheld = usize::from(!compact);
    let mut decode_rows = if unified { feeds.len() } else { 0 };
    let mut tick_fault: Option<crate::DeviceErrorInfo> = None;
    let budget_max = e.pf_max_rows();
    if budget_max == 0 {
        return tick_fault;
    }
    // Overlay rows one launch can stage, when the packet splices host rows.
    let overlay_rows = e.tensor_bytes("in.encoder_overlay").map(|bytes| {
        let hidden = slots
            .iter()
            .flatten()
            .filter_map(|s| s.speech.as_deref())
            .find(|sp| !sp.overlay_pos.is_empty())
            .map_or(1, |sp| sp.overlay.len() / sp.overlay_pos.len());
        bytes as usize / 4 / hidden.max(1)
    });
    let (launch_cap, launch_max) = if cold && !bounded_tick {
        (budget_max, budget_max)
    } else {
        pf_launch_rows(e.pf_request_max_rows(), budget_max)
    };
    // Bound each candidate before fair sharing. Short requests return unused
    // rows to later candidates in the same launch.
    let chunk_cap = pf_chunk_rows().min(e.pf_request_max_rows());
    let stage = e.pf_stage_rows();
    let adaptive = crate::serve::policy::adaptive_packing();
    loop {
        let host_t = packlog::on().then(Instant::now);
        for (i, slot) in slots.iter_mut().enumerate().take(cap) {
            let Some(request) = slot.as_mut().filter(|s| s.step == 0) else {
                continue;
            };
            if request.respond.is_closed() {
                if let Some(taken) = slot.take() {
                    release_kv(arena, taken.kv);
                }
                e.retire_slot(i, false);
                continue;
            }
            if request.prompt_ids.is_empty() {
                continue;
            }
            fit_speech_budget(request, e.max_ctx());
            let total = request.prompt_ids.len() + request.gen.max_tokens.max(1);
            let pair = request.cfg.is_some();
            let admitted = (|| -> Result<()> {
                for row in [i, i + 1].into_iter().take(1 + pair as usize) {
                    let fresh = !e.packed_slot_ready(row);
                    if fresh && request.resume > 0 {
                        request.resume = e.resume_slot(row, request.resume);
                    }
                    if fresh && row == i && request.prefix.is_some() {
                        e.stage_prefix_key(row, request.prefix.take());
                    }
                    let Some(frontier) = e.admit_packed_slot(row, &request.prompt_ids, total)? else {
                        continue;
                    };
                    if let Some(ttl) = request.session.as_ref().and_then(|s| s.pin_ttl()).filter(|_| fresh) {
                        e.hold_session_prefix(row, ttl);
                    }
                    if fresh && request.speech.as_ref().is_some_and(|sp| sp.pos_base.is_some()) {
                        // A decode launch covering this row before its prompt is in must not find
                        // a stale base above the reset position (`EmbedPosBf16` traps on pos < base).
                        e.write_tensor("in.pos_base", (row * 4) as u64, &0u32.to_le_bytes())?;
                    }
                    match request.cfg.as_mut().filter(|_| row != i) {
                        Some(run) => run.partner_pf = frontier,
                        None => {
                            request.pf_pos = frontier;
                            request.cached_tokens = (e.attached_rows(i) as usize).max(request.resume.min(frontier));
                        }
                    }
                }
                Ok(())
            })();
            match admitted {
                Ok(_) => {}
                Err(err) => {
                    tracing::warn!(
                        slot = i,
                        error = %err,
                        error_code = ?err.device_code(),
                        fatal = err.is_fatal(),
                        "gpu: packed KV admission failed"
                    );
                    note_fault(&mut tick_fault, &err);
                    fail_slot(slot, arena, err);
                }
            }
        }
        // Gather candidate spans directly; counting admitted rows and filtering slots in one pass.
        let now = Instant::now();
        let candidates: Vec<crate::sched::step::Candidate> = (0..cap.min(slots.len()))
            .filter_map(|i| {
                let (s, pf_pos) = pack_row(slots, i)?;
                let n = s.prompt_ids.len();
                if !e.packed_slot_ready(i) || s.step != 0 || n == 0 || pf_pos + withheld >= n {
                    return None;
                }
                let publish = e.pf_publish_cap(i, pf_pos);
                let mut remaining = e.pf_plan_slice(n - withheld - pf_pos, chunk_cap).min(publish);
                if s.mm.as_deref().is_some_and(|j| j.spans()) {
                    remaining = plow_asset::multimodal::span_chunk_rows(
                        &s.prompt_ids[..n - withheld],
                        pf_pos,
                        remaining,
                        chunk_cap.min(publish),
                        stage,
                    );
                }
                let n_rows = u32::try_from(remaining).ok().filter(|&r| r > 0)?;
                let slot_u32 = u32::try_from(i).ok()?;
                let kv_row0 = u32::try_from(pf_pos).ok()?;
                let span = packet::dev::PrefillSpan {
                    row0: 0,
                    n_rows,
                    slot: slot_u32,
                    flags: 0,
                    kv_row0,
                    kv_len: kv_row0 + n_rows,
                    state_slot: slot_u32,
                    program: 0,
                };
                Some(crate::sched::step::Candidate {
                    arrival: arrival_key(s.arrived, now),
                    span,
                    packable: true,
                    planned: true,
                })
            })
            .collect();
        if candidates.is_empty() {
            return tick_fault;
        }
        let avail: usize = candidates.iter().map(|c| c.span.n_rows as usize).sum();
        // `RideCost` keys both the decision and the launch it observes by the prefill bucket:
        // keyed by the launch's own bucket, a riding launch that spills (4096 + 63 into 4160)
        // records under a bucket the decision never asks about, and the rider cost is never learned.
        let ride_key = e.pf_pack_budget(avail.min(launch_cap));
        if std::mem::take(&mut ride_open) && !ride.ride(ride_key, feeds.len()) {
            unified = false;
            decode_rows = 0;
        }
        let per_launch = launch_cap.min(launch_max.saturating_sub(decode_rows));
        // Every overlay row of a launch is staged at once: bound its rows by the overlay's.
        let per_launch = per_launch.min(overlay_rows.unwrap_or(usize::MAX)).max(1);
        let per_launch = if adaptive {
            let mut queue: Vec<(u64, usize)> =
                candidates.iter().map(|c| (c.arrival, c.span.n_rows as usize)).collect();
            queue.sort_unstable_by_key(|&(arrival, _)| arrival);
            let rows: Vec<usize> = queue.into_iter().map(|(_, rows)| rows).collect();
            queue_pack_rows(&rows, pf_chunk_cost_rows(), per_launch)
        } else {
            per_launch
        };
        let rows = avail.min(per_launch);
        let bucket = e.pf_pack_budget(rows);
        // Under the unified token batch the decode rows ride in this launch, and the batch takes the
        // smallest bucket holding every row.
        let trim = trim_for_riders(
            bucket,
            e.pf_pack_budget(rows.min(bucket) + decode_rows),
            decode_rows,
            rows,
            avail - rows,
        );
        let per_launch = if trim {
            bucket.saturating_sub(decode_rows).max(1)
        } else {
            bucket
        }
        .min(per_launch);
        let step = crate::sched::step::plan(
            e.step_backend(),
            crate::sched::step::Tick {
                cap_rows: u32::try_from(per_launch).unwrap_or(u32::MAX),
                packing: true,
                rotate: crate::config::RuntimeConfig::get().pf_rotate(),
                turn: e.prefill_turn(),
                slots: cap.min(slots.len()),
            },
            if unified { feeds.iter().map(|&(i, _)| i as u32).collect::<Vec<_>>() } else { Vec::new() },
            &candidates,
            |_| u32::try_from(per_launch).ok(),
            |_| u32::MAX,
        );
        let pack: Vec<_> = step
            .launches
            .first()
            .map(|launch| launch.spans.as_slice())
            .unwrap_or(&[])
            .iter()
            .map(|span| (span.slot as usize, span.kv_row0 as usize, span.n_rows as usize))
            .filter_map(|(i, c0, len)| {
                // A share cut by the planner must still not split a media span.
                let s = slots[i].as_ref()?;
                if !s.mm.as_deref().is_some_and(|j| j.spans()) {
                    return Some((i, c0, len));
                }
                let len = plow_asset::multimodal::span_safe_rows(&s.prompt_ids, c0, len, stage);
                (len > 0).then_some((i, c0, len))
            })
            .collect();
        if pack.is_empty() {
            return tick_fault;
        }
        if let Some(t) = host_t {
            eprintln!(
                "PACKLOG PACK reqs={} rows={} decode_feeds={} unified={} admit_plan_us={:.0}",
                pack.len(),
                pack.iter().map(|p| p.2).sum::<usize>(),
                feeds.len(),
                unified,
                t.elapsed().as_secs_f64() * 1e6
            );
        }
        let riders = if unified { feeds.len() } else { 0 };
        let t_launch = Instant::now();
        let res = if unified {
            use plow_asset::token_batch::{Phase, Request, Selection};
            // A feed or pack entry whose slot vanished mid-tick is skipped, not a panic:
            // the tick thread must outlive one inconsistent row, and the slot re-feeds
            // next tick from live state.
            let decode = feeds.iter().filter_map(|&(i, ref token)| {
                let slot = slots[i].as_ref()?;
                Some(Request {
                    id: i as u32,
                    slot: i as u32,
                    state_slot: i as u32,
                    generation: e.slot_generation(i)?,
                    phase: Phase::Decode,
                    tokens: std::slice::from_ref(token),
                    prompt_len: slot.prompt_ids.len() as u32,
                    selection: Selection::default(),
                })
            });
            let prefill = pack.iter().filter_map(|&(i, c0, len)| {
                let slot = slots[i].as_ref()?;
                Some(Request {
                    id: i as u32,
                    slot: i as u32,
                    state_slot: i as u32,
                    generation: e.slot_generation(i)?,
                    phase: Phase::Prefill,
                    tokens: slot.prompt_ids.get(c0..c0.checked_add(len)?)?,
                    prompt_len: slot.prompt_ids.len() as u32,
                    selection: Selection::default(),
                })
            });
            let requests: smallvec::SmallVec<[_; 16]> = decode.chain(prefill).collect();
            let result = if e.dev_sample_enabled() {
                // Stochastic rows draw on the device: a host draw downloads the row's 512 KiB of
                // logits and walks the vocabulary once per row, serially, inside the tick.
                *dev_sampled = true;
                e.token_batch_step_sampled(&requests, unified_output, &|slot| {
                    slots.get(slot as usize)?.as_ref().and_then(dev_row_spec)
                })
            } else {
                e.token_batch_step(&requests, unified_output)
            };
            if result.is_ok() {
                completed.extend(
                    unified_output.iter().map(|&(slot, token)| (slot as usize, token)),
                );
            }
            result
        } else {
            let staged = match overlay_rows {
                Some(_) => gpu_speech_pack_inputs(e, slots, &pack),
                None => Ok(()),
            };
            let reqs: Vec<PfBatchReq> = pack
                .iter()
                .map(|&(i, c0, len)| PfBatchReq {
                    slot: i,
                    prompt: &pack_row(slots, i).expect("packed row is live").0.prompt_ids,
                    c0,
                    len,
                })
                .collect();
            staged
                .and_then(|()| {
                    if compact {
                        e.prefill_batched_complete(&reqs, completed)
                    } else {
                        e.prefill_batched(&reqs)
                    }
                })
                .and_then(|()| {
                    // Completed prompts' logits sit in compact rows 0.. in `completed` order.
                    if !e.dev_sample_enabled() || completed.is_empty() {
                        return Ok(());
                    }
                    let specs: smallvec::SmallVec<[crate::exec::gpu::DevSample; 16]> = completed
                        .iter()
                        .map(|&(i, _)| {
                            slots[i]
                                .as_ref()
                                .and_then(dev_row_spec)
                                .unwrap_or_else(crate::exec::gpu::DevSample::greedy)
                        })
                        .collect();
                    if specs.iter().all(|s| s.temp <= 0.0) {
                        return Ok(());
                    }
                    let mut ids: smallvec::SmallVec<[u32; 16]> = smallvec::smallvec![0; specs.len()];
                    e.sample_rows_on_device(&specs, &mut ids)?;
                    for (entry, id) in completed.iter_mut().zip(ids) {
                        entry.1 = id;
                    }
                    *dev_sampled = true;
                    Ok(())
                })
        };
        if res.is_ok() {
            ride.observe_launch(ride_key, riders, t_launch.elapsed().as_secs_f64() * 1e3);
        }
        match res {
            Ok(()) => {
                if unified {
                    feeds.clear();
                }
                for &(i, c0, len) in &pack {
                    set_pack_frontier(slots, i, c0 + len);
                }
                let last_slot = pack.last().expect("pack is non-empty").0;
                let last_finished = slots[last_slot]
                    .as_ref()
                    .map(|s| s.pf_pos + withheld >= s.prompt_ids.len())
                    .unwrap_or(true);
                if last_finished {
                    e.advance_prefill_turn(last_slot);
                }
            }
            Err(err) => {
                // The shared launch failed — every packed request loses.
                tracing::warn!(
                    error = %err,
                    error_code = ?err.device_code(),
                    fatal = err.is_fatal(),
                    packed = pack.len(),
                    "gpu: batched prefill failed"
                );
                note_fault(&mut tick_fault, &err);
                let msg = err.to_string();
                if unified {
                    for (i, _) in feeds.drain(..) {
                        fail_slot(&mut slots[i], arena, fanout_err(&err, &msg));
                        e.retire_slot(i, false);
                    }
                }
                for &(i, _, _) in &pack {
                    let owner = if slots[i].is_some() { i } else { i - 1 };
                    fail_slot(&mut slots[owner], arena, fanout_err(&err, &msg));
                    if unified {
                        e.retire_slot(i, false);
                    }
                }
                return tick_fault;
            }
        }
        if !completed.is_empty() || !cold || bounded_tick {
            return tick_fault; // bounded stall: decode now, next pack next tick
        }
        // Cold path: stop as soon as any request is ready so its first token
        // fires this tick; the rest continue next tick (with decoders live).
        let any_ready = pack.iter().any(|&(i, _, _)| {
            slots[i].as_ref().is_some_and(|s| {
                e.packed_slot_ready(i)
                    && s.step == 0
                    && !s.prompt_ids.is_empty()
                    && packed_prompt_done(slots, i, withheld)
            })
        });
        if any_ready {
            return tick_fault;
        }
    }
}

/// A speech job's `max_tokens` is its packet's cap (an ASR transcript, a TTS utterance), not a
/// client's ask: under a live context bound (`--live-ctx-models`) the stream ends at the context
/// instead of the request failing.
#[cfg(feature = "cuda")]
fn fit_speech_budget(slot: &mut Slot, max_ctx: usize) {
    if slot.speech.is_some() {
        let room = max_ctx.saturating_sub(slot.prompt_ids.len()).max(1);
        slot.gen.max_tokens = slot.gen.max_tokens.min(room);
    }
}

/// Advance a prefilling slot: one prompt chunk through the prefill bucket
/// chain into engine slot `slot_idx`'s KV ring (bucket capped at `cap_rows`),
/// or whole-prompt decode-only consumption (one launch per prompt token) when
/// no `_pf` object is loaded. Returns `Ok(None)` while the prompt is still
/// being consumed; `Ok(Some(token))` once `in.ids[0]` holds the first
/// generated token and the slot's `pos == n_prompt`. Greedy consumes the
/// device `ARGMAX_FIN` token; `temperature > 0` downloads the logits row and
/// reuses the host sampler (prefill logits land in row 0 regardless of slot).
#[cfg(feature = "cuda")]
fn gpu_prefill_advance(
    e: &mut crate::exec::gpu::GpuEngine,
    slot_idx: usize,
    slot: &mut Slot,
    cap_rows: usize,
) -> Result<Option<u32>> {
    use crate::exec::gpu::PrefillStep;

    if slot.prompt_ids.is_empty() {
        return Err(crate::RuntimeError::Rejected("empty prompt".into()));
    }
    fit_speech_budget(slot, e.max_ctx());
    let total = slot.prompt_ids.len() + slot.gen.max_tokens.max(1);
    if slot.pf_pos == 0 {
        let kept = if slot.resume > 0 && slot.cfg.is_none() { e.resume_slot(slot_idx, slot.resume) } else { 0 };
        // A CFG pair's partner begins with its owner, so memory for both is there or neither.
        let partner = (slot.cfg.is_some() && slot.speech.as_ref().is_some_and(|sp| sp.cfg.is_some()))
            .then_some(slot_idx + 1);
        let begun = e
            .begin_slot(slot_idx, total)
            .and_then(|()| partner.map_or(Ok(()), |p| e.begin_slot(p, total)));
        if let Err(err) = begun {
            // Out of device memory for live KV: wait (the prompt stays at row 0), and ask the
            // planner for room instead of failing the request.
            let short = matches!(err, crate::RuntimeError::Oom(_)) || err.device_code() == Some(2);
            if !short || err.is_fatal() {
                return Err(err);
            }
            e.retire_slot(slot_idx, false);
            if let Some(p) = partner {
                e.retire_slot(p, false);
            }
            e.note_kv_pressure();
            return Ok(None);
        }
        if kept > 0 {
            slot.resume = kept;
            slot.pf_pos = kept;
            slot.cached_tokens = kept;
        }
        if let Some(ttl) = slot.session.as_ref().and_then(|s| s.pin_ttl()) {
            e.hold_session_prefix(slot_idx, ttl);
        }
    }
    if let Some(sp) = slot.speech.as_deref() {
        // The pair prefills as one unit: the unconditional member to completion on the partner
        // first, its last-row logits stashed before the owner's prefill overwrites row 0.
        if let (0, Some(cfg), Some(run)) = (slot.pf_pos, sp.cfg.as_ref(), slot.cfg.as_mut()) {
            let partner = slot_idx + 1;
            let mut c0 = 0;
            loop {
                gpu_speech_prefill_inputs(e, partner, c0, &cfg.uncond_overlay, &sp.overlay_pos, sp.pos_base)?;
                match e.prefill_chunk(partner, &slot.prompt_ids, usize::MAX)? {
                    PrefillStep::Progress(end) => c0 = end,
                    PrefillStep::Done(_) => break,
                }
            }
            e.logits_row(0, &mut run.uncond)?;
            if let Some(base) = sp.pos_base {
                e.write_tensor("in.pos_base", (partner * 4) as u64, &base.to_le_bytes())?;
            }
        }
        gpu_speech_prefill_inputs(e, slot_idx, slot.pf_pos, &sp.overlay, &sp.overlay_pos, sp.pos_base)?;
    }
    let tok = if e.has_prefill() {
        let staged = slot.pf_pos == 0 && slot.prefix.is_some();
        if staged {
            e.stage_prefix_key(slot_idx, slot.prefix.take());
        }
        let step = e.prefill_chunk(slot_idx, &slot.prompt_ids, cap_rows);
        if staged {
            e.stage_prefix_key(slot_idx, None);
        }
        match step? {
            PrefillStep::Progress(frontier) => {
                slot.pf_pos = frontier;
                // First chunk consulted the prefix cache — record the hit.
                slot.cached_tokens = e.attached_rows(slot_idx) as usize;
                return Ok(None);
            }
            PrefillStep::Done(tok) => {
                slot.pf_pos = slot.prompt_ids.len();
                slot.cached_tokens = e.attached_rows(slot_idx) as usize;
                tok
            }
        }
    } else {
        let mut tok = 0u32;
        let mut toks = Vec::with_capacity(1);
        // Decode-only prompt consumption still gets prefix sharing: attach
        // maps the cached rows, so only the tail is fed token by token.
        // `consume_prompt` overlaps host submit with the in-flight
        // interpreter (one D2H+sync after the last token).
        let start = if slot.pf_pos == 0 {
            let attached = e.attach_prompt_keyed(slot_idx, &slot.prompt_ids, slot.prefix.take())?;
            slot.cached_tokens = attached;
            attached
        } else {
            slot.pf_pos
        };
        let end = start
            .saturating_add(cap_rows.max(1))
            .min(slot.prompt_ids.len());
        let Some(tail) = slot.prompt_ids.get(start..end) else {
            return Err(crate::RuntimeError::Msg(format!(
                "prompt cursor {start} is past the prompt ({} tokens)",
                slot.prompt_ids.len()
            )));
        };
        if !tail.is_empty() {
            tok = e.consume_prompt(slot_idx, tail, &mut toks)?;
        }
        slot.pf_pos = end;
        if end < slot.prompt_ids.len() {
            return Ok(None);
        }
        tok
    };
    // Parity artifact: these ids + the per-step tokens are what the
    // standalone gemma4_sm120_chat harness is diffed against.
    tracing::debug!(
        prompt_ids = ?slot.prompt_ids,
        first_token = tok,
        slot = slot_idx,
        "gpu: prompt consumed"
    );
    if let Some(base) = slot.speech.as_ref().and_then(|s| s.pos_base) {
        e.write_tensor("in.pos_base", (slot_idx * 4) as u64, &base.to_le_bytes())?;
    }
    if let Some(run) = slot.cfg.as_mut() {
        e.logits_row(0, &mut run.cond)?;
        return Ok(Some(cfg_draw(slot)));
    }
    if let Some(spec) = dev_row_spec(slot).filter(|_| e.dev_sample_enabled()) {
        let mut tok = [0u32];
        e.sample_rows_on_device(&[spec], &mut tok)?;
        return Ok(Some(tok[0]));
    }
    gpu_finish_token(e, 0, slot, tok).map(Some)
}

/// Refuse a packet with per-row host inputs (`in.encoder_overlay`, `in.pos_base`) whose prefill
/// may run through a route that does not stage them per launch row: the mixed step and the unified
/// token batch. Serial and packed prefill stage every member's rows ([`gpu_speech_pack_inputs`]).
#[cfg(feature = "cuda")]
pub fn check_speech_packet(e: &crate::exec::gpu::GpuEngine) -> Result<()> {
    let overlay = e.tensor_bytes("in.encoder_overlay").is_some();
    if !overlay && e.tensor_bytes("in.pos_base").is_none() {
        return Ok(());
    }
    if e.token_batch_enabled() || e.mixed_step_rows(1, 1).is_some() {
        return Err(crate::RuntimeError::Rejected(
            "packet has per-row host inputs (overlay / position base) and a mixed-step or \
             token-batch program, which do not stage them per row"
                .into(),
        ));
    }
    if overlay && !e.has_prefill() {
        return Err(crate::RuntimeError::Rejected(
            "overlay packet has no prefill programs".into(),
        ));
    }
    Ok(())
}

/// Stage a speech job's per-slot inputs for its next prefill chunk starting at prompt row `c0`:
/// the overlay rows that chunk covers (compacted to overlay row 0) and a chunk-relative
/// `in.encoder_overlay_index`. Rewritten before EVERY chunk: the overlay tensor is shared by all
/// slots and another job may have prefilled in between.
#[cfg(feature = "cuda")]
fn gpu_speech_prefill_inputs(
    e: &mut crate::exec::gpu::GpuEngine,
    slot_idx: usize,
    c0: usize,
    overlay: &[f32],
    overlay_pos: &[u32],
    pos_base: Option<u32>,
) -> Result<()> {
    // A decode launch covering this row before the prompt is in must not find a stale base above
    // its reset position (`EmbedPosBf16` traps on pos < base); the real base lands at Done.
    if c0 == 0 && pos_base.is_some() {
        e.write_tensor("in.pos_base", (slot_idx * 4) as u64, &0u32.to_le_bytes())?;
    }
    if overlay_pos.is_empty() {
        return Ok(());
    }
    if !e.has_prefill() {
        return Err(crate::RuntimeError::Rejected(
            "overlay prompts need the packet's prefill programs".into(),
        ));
    }
    let hidden = overlay.len() / overlay_pos.len();
    let index_rows = e
        .tensor_bytes("in.encoder_overlay_index")
        .ok_or_else(|| crate::RuntimeError::Rejected("packet has no in.encoder_overlay_index".into()))?
        as usize
        / 4;
    // Every row a prefill launch may read, padded bucket rows included.
    let window = index_rows.min(e.pf_max_rows().max(1));
    let mut index = Vec::new();
    let (lo, hi) = speech_overlay_index(overlay_pos, c0, window, &mut index);
    if hi > lo {
        e.write_tensor(
            "in.encoder_overlay",
            0,
            bytemuck::cast_slice(&overlay[lo * hidden..hi * hidden]),
        )?;
    }
    e.write_tensor("in.encoder_overlay_index", 0, bytemuck::cast_slice(&index))
}

/// The chunk-relative overlay index for launch rows `[c0, c0 + window)`: the overlay rows
/// `lo..hi` whose prompt positions fall there, renumbered from 0; every other row `u32::MAX`.
#[cfg(any(feature = "cuda", test))]
fn speech_overlay_index(pos: &[u32], c0: usize, window: usize, index: &mut Vec<u32>) -> (usize, usize) {
    let lo = pos.partition_point(|&p| (p as usize) < c0);
    let hi = pos.partition_point(|&p| (p as usize) < c0.saturating_add(window));
    index.clear();
    index.resize(window, u32::MAX);
    for (k, &p) in pos[lo..hi].iter().enumerate() {
        index[p as usize - c0] = k as u32;
    }
    (lo, hi)
}

/// The request behind engine row `i` and that row's prefill frontier: the slot's own, or its CFG
/// owner's partner frontier when `i` is a live pair's partner.
#[cfg(any(feature = "cuda", test))]
fn pack_row(slots: &[Option<Slot>], i: usize) -> Option<(&Slot, usize)> {
    match slots.get(i)?.as_ref() {
        Some(s) => Some((s, s.pf_pos)),
        None => {
            let owner = slots.get(i.checked_sub(1)?)?.as_ref()?;
            Some((owner, owner.cfg.as_ref()?.partner_pf))
        }
    }
}

#[cfg(any(feature = "cuda", test))]
fn set_pack_frontier(slots: &mut [Option<Slot>], i: usize, frontier: usize) {
    if let Some(s) = slots[i].as_mut() {
        s.pf_pos = frontier;
    } else if let Some(run) = i.checked_sub(1).and_then(|o| slots[o].as_mut()?.cfg.as_mut()) {
        run.partner_pf = frontier;
    }
}

/// Whether owner `i`'s packed prefill (both members of a CFG pair) left only the `withheld` rows.
#[cfg(any(feature = "cuda", test))]
fn packed_prompt_done(slots: &[Option<Slot>], i: usize, withheld: usize) -> bool {
    slots[i].as_ref().is_some_and(|s| {
        let n = s.prompt_ids.len();
        s.pf_pos + withheld >= n && s.cfg.as_ref().is_none_or(|run| run.partner_pf + withheld >= n)
    })
}

/// The overlay and its launch-row index for one packed launch: each member's overlay rows inside
/// its chunk, at the chunk's launch offset, compacted in pack order. Written for every launch of
/// an overlay packet, so a row no member overlays never reads a stale index.
#[cfg(any(feature = "cuda", test))]
fn speech_pack_overlay(
    slots: &[Option<Slot>],
    pack: &[(usize, usize, usize)],
    window: usize,
    rows: &mut Vec<f32>,
    index: &mut Vec<u32>,
) {
    rows.clear();
    index.clear();
    index.resize(window, u32::MAX);
    let mut launch_row = 0usize;
    let mut k = 0u32;
    for &(i, c0, len) in pack {
        let partner = slots[i].is_none();
        if let Some(sp) = pack_row(slots, i).and_then(|(s, _)| s.speech.as_deref()) {
            let src = match (partner, sp.cfg.as_ref()) {
                (true, Some(cfg)) => &cfg.uncond_overlay,
                _ => &sp.overlay,
            };
            let hidden = src.len() / sp.overlay_pos.len().max(1);
            let lo = sp.overlay_pos.partition_point(|&p| (p as usize) < c0);
            let hi = sp.overlay_pos.partition_point(|&p| (p as usize) < c0 + len);
            for &p in &sp.overlay_pos[lo..hi] {
                if let Some(row) = index.get_mut(launch_row + p as usize - c0) {
                    *row = k;
                }
                k += 1;
            }
            rows.extend_from_slice(&src[lo * hidden..hi * hidden]);
        }
        launch_row += len;
    }
}

#[cfg(feature = "cuda")]
fn gpu_speech_pack_inputs(
    e: &mut crate::exec::gpu::GpuEngine,
    slots: &[Option<Slot>],
    pack: &[(usize, usize, usize)],
) -> Result<()> {
    thread_local! {
        static STAGE: std::cell::RefCell<(Vec<f32>, Vec<u32>)> =
            const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
    }
    let index_rows = e
        .tensor_bytes("in.encoder_overlay_index")
        .ok_or_else(|| crate::RuntimeError::Rejected("packet has no in.encoder_overlay_index".into()))?
        as usize
        / 4;
    let window = index_rows.min(e.pf_max_rows().max(1));
    STAGE.with_borrow_mut(|(rows, index)| {
        speech_pack_overlay(slots, pack, window, rows, index);
        if !rows.is_empty() {
            e.write_tensor("in.encoder_overlay", 0, bytemuck::cast_slice(rows))?;
        }
        e.write_tensor("in.encoder_overlay_index", 0, bytemuck::cast_slice(index))
    })
}

/// A packed launch completed engine row `i`'s prompt (compact terminal, logits row `row`). Writes
/// a speech job's decode position base and, for a CFG member, stashes the row's logits in its
/// owner: `Some((owner, true))`, drawn once both members are in. `Some((i, false))` for an
/// ordinary row, `None` for a row whose request is gone.
#[cfg(feature = "cuda")]
fn gpu_speech_prompt_done(
    e: &mut crate::exec::gpu::GpuEngine,
    slots: &mut [Option<Slot>],
    row: usize,
    i: usize,
) -> Result<Option<(usize, bool)>> {
    let partner = slots[i].is_none();
    let owner = if partner { i.checked_sub(1).filter(|&o| slots[o].is_some()) } else { Some(i) };
    let Some(owner) = owner else { return Ok(None) };
    let Some(slot) = slots[owner].as_mut() else { return Ok(None) };
    if let Some(base) = slot.speech.as_ref().and_then(|sp| sp.pos_base) {
        e.write_tensor("in.pos_base", (i * 4) as u64, &base.to_le_bytes())?;
    }
    let Some(run) = slot.cfg.as_mut() else { return Ok(Some((owner, false))) };
    e.logits_row(row, if partner { &mut run.uncond } else { &mut run.cond })?;
    Ok(Some((owner, true)))
}

/// The per-step stochastic draw for one slot, with the request's OpenAI `seed`
/// mixed in.
///
/// EVERY sampling site goes through this. The three that mattered each called
/// a seedless draw helper directly, so `seed` was honoured only on the
/// no-bucket reference path — i.e. only where there is no model to sample from.
fn slot_rng01(slot: &Slot) -> f32 {
    crate::serve::seeded_unit_with(&slot.prompt_ids, &slot.out_ids, slot.step, slot.gen.seed)
}

/// The draw for the `k`-th token of a device multi-step quantum. Tokens produced inside the
/// quantum are not on the host yet, so the draw keys on the quantum-start history and the step.
#[cfg(feature = "cuda")]
fn slot_rng01_at(slot: &Slot, k: usize) -> f32 {
    crate::serve::seeded_unit_with(&slot.prompt_ids, &slot.out_ids, slot.step + k, slot.gen.seed)
}

/// Device-sampling eligibility for a slot (plan stage 4). `Some(spec)` when the
/// row is `temperature>0` and needs no per-row token history — the device
/// sampler handles temperature/top_k/top_p/min_p but not the penalties or the
/// logit bias, so those rows keep the host path (see
/// [`SamplingParams::needs_host_logits`]). `rng01` is the same per-step seeded
/// draw the host uses, so a fixed seed stays reproducible. Greedy rows return
/// `None` (the device argmax already equals `ARGMAX_FIN`).
#[cfg(feature = "cuda")]
fn dev_sample_spec(slot: &Slot) -> Option<crate::exec::gpu::DevSample> {
    let p = &slot.gen.params;
    if p.temperature <= 0.0 || p.needs_host_logits() {
        return None;
    }
    Some(crate::exec::gpu::DevSample {
        temp: p.temperature,
        top_k: p.top_k as i32,
        top_p: p.top_p,
        min_p: p.min_p,
        rng01: slot_rng01(slot),
    })
}

/// The spec a token-batch row is drawn with on the device: a stochastic, non-CFG row that needs
/// no host logits. `None` keeps the argmax and the host path.
#[cfg(feature = "cuda")]
fn dev_row_spec(slot: &Slot) -> Option<crate::exec::gpu::DevSample> {
    slot.cfg.is_none().then(|| dev_sample_spec(slot)).flatten()
}

/// Unmodified greedy requests keep the device argmax. Sampling adjustments
/// download logits row `row` and use the host sampler.
#[cfg(feature = "cuda")]
fn gpu_finish_token(
    e: &mut crate::exec::gpu::GpuEngine,
    row: usize,
    slot: &mut Slot,
    argmax_tok: u32,
) -> Result<u32> {
    if let Some(run) = slot.cfg.as_mut() {
        e.logits_row(row + 1, &mut run.uncond)?;
        e.logits_row(row, &mut run.cond)?;
        return Ok(cfg_draw(slot));
    }
    if !gpu_argmax_eligible(&slot.gen.params) {
        // Filled for this step by `gpu_batch_logprobs`.
        if slot.lp.is_some() && greedy_stats_k(&slot.gen.params).is_some() {
            return Ok(argmax_tok);
        }
        if let Some(lp) = greedy_device_logprobs(e, row, &slot.gen.params, argmax_tok)? {
            slot.lp = Some(Box::new(lp));
            return Ok(argmax_tok);
        }
        let mut logits = e.take_logits_buf();
        logits.clear();
        e.logits_row(row, &mut logits)?;
        let p = &slot.gen.params;
        let stats = p.logprobs.map(|r| crate::text::logprobs::RowStats::of(&logits, r));
        let adjusted = p.repetition_penalty != 1.0
            || p.presence_penalty != 0.0
            || p.frequency_penalty != 0.0
            || !p.logit_bias.is_empty();
        // Greedy logprobs rows keep the device argmax (same bf16 row, same lowest-id tie-break).
        let tok = if p.temperature <= 0.0 && !adjusted {
            argmax_tok
        } else {
            let raw = (stats.is_some() && adjusted).then(|| logits.clone());
            crate::text::sample::apply_penalties(&mut logits, &slot.out_ids, &slot.gen.params);
            let tok = crate::text::sample::sample(&logits, &slot.gen.params, None, slot_rng01(slot));
            if let Some(raw) = raw {
                logits = raw;
            }
            tok
        };
        if let Some(stats) = stats {
            slot.lp = Some(Box::new(stats.finish(logits[tok as usize])));
        }
        e.return_logits_buf(logits);
        return Ok(tok);
    }
    Ok(argmax_tok)
}

/// OpenAI logprobs for a single-sequence engine that exposes its logits to the host (CPU): only
/// rows that asked for them pay the vocab read. The served token stays the engine's argmax.
#[cfg(any(feature = "hsa", feature = "cpu"))]
fn seq_host_logprobs(e: &dyn super::engine::SeqEngine, i: usize, slot: &mut Option<Slot>, token: u32) {
    let Some(s) = slot.as_mut() else { return };
    let Some(req) = s.gen.params.logprobs else { return };
    let mut logits = Vec::new();
    if e.logits_row(i, &mut logits) && (token as usize) < logits.len() {
        let stats = crate::text::logprobs::RowStats::of(&logits, req);
        s.lp = Some(Box::new(stats.finish(logits[token as usize])));
    }
}

/// The top-k a greedy, unadjusted logprobs row asks the device stats kernel for; `None` when the
/// row needs the host path.
#[cfg(feature = "cuda")]
fn greedy_stats_k(p: &crate::text::sample::SamplingParams) -> Option<u32> {
    let req = p.logprobs?;
    let adjusted = p.repetition_penalty != 1.0
        || p.presence_penalty != 0.0
        || p.frequency_penalty != 0.0
        || !p.logit_bias.is_empty();
    (p.temperature <= 0.0 && !adjusted).then(|| u32::from(req.top.min(crate::text::logprobs::MAX_TOP_LOGPROBS)))
}

/// Logprobs from device statistics `out` (`plow_logprob_stats` layout) for `k` alternatives.
#[cfg(feature = "cuda")]
fn stats_logprobs(out: &[f32], k: usize, raw_logits: bool) -> crate::text::logprobs::TokenLogprobs {
    let top = (0..k).map(|i| (out[3 + i].to_bits(), out[3 + k + i])).filter(|&(t, _)| t != u32::MAX).collect();
    crate::text::logprobs::RowStats::from_parts(out[0], raw_logits, top).finish(out[1])
}

/// Every greedy logprobs row of a step through one `plow_logprob_stats_rows` launch and one
/// readback, instead of a launch and a synchronous read per row. `rows` = (logits row, slot,
/// token). Fills `slot.lp` for the rows it serves; the rest (an inexact top-k, an object without
/// the batched kernel) keep [`gpu_finish_token`]'s per-row path.
#[cfg(feature = "cuda")]
fn gpu_batch_logprobs(
    e: &mut crate::exec::gpu::GpuEngine,
    rows: impl Iterator<Item = (usize, usize, u32)>,
    slots: &mut [Option<Slot>],
) -> Result<()> {
    let mut reqs: smallvec::SmallVec<[(u32, u32, u32); 32]> = Default::default();
    let mut owners: smallvec::SmallVec<[usize; 32]> = Default::default();
    for (row, i, tok) in rows {
        let Some(slot) = slots[i].as_ref().filter(|s| s.cfg.is_none()) else { continue };
        if let Some(k) = greedy_stats_k(&slot.gen.params) {
            reqs.push((row as u32, tok, k));
            owners.push(i);
        }
    }
    if reqs.is_empty() {
        return Ok(());
    }
    let mut out = Vec::with_capacity(reqs.len());
    if !e.logprob_stats_rows(&reqs, &mut out)? {
        return Ok(());
    }
    for ((&i, &(_, _, k)), stats) in owners.iter().zip(&reqs).zip(&out) {
        if stats[2] != 0.0 {
            continue;
        }
        let slot = slots[i].as_mut().expect("checked Some");
        let raw = slot.gen.params.logprobs.is_some_and(|r| r.raw_logits);
        slot.lp = Some(Box::new(stats_logprobs(stats, k as usize, raw)));
    }
    Ok(())
}

/// A greedy, unadjusted logprobs row: its statistics come from `plow_logprob_stats` on device
/// (43 floats back instead of the 512 KiB row). `None` when the row needs the host path, the
/// sampler object has no stats kernel, or its top-k is flagged inexact.
#[cfg(feature = "cuda")]
fn greedy_device_logprobs(
    e: &mut crate::exec::gpu::GpuEngine,
    row: usize,
    p: &crate::text::sample::SamplingParams,
    tok: u32,
) -> Result<Option<crate::text::logprobs::TokenLogprobs>> {
    let (Some(req), Some(k)) = (p.logprobs, greedy_stats_k(p)) else { return Ok(None) };
    let mut out = [0f32; crate::exec::gpu::LOGPROB_STATS_MAX];
    if !e.logprob_stats(row, tok, k, &mut out)? || out[2] != 0.0 {
        return Ok(None);
    }
    Ok(Some(stats_logprobs(&out, k as usize, req.raw_logits)))
}

/// Draw every fed CFG owner's token from ONE download of the step's logits rows (instead of two
/// synchronous row reads per pair) into its `toks` entry. Returns whether any was drawn.
#[cfg(feature = "cuda")]
fn gpu_cfg_draws(
    e: &mut crate::exec::gpu::GpuEngine,
    feeds: &[(usize, u32)],
    slots: &mut [Option<Slot>],
    toks: &mut [u32],
) -> Result<bool> {
    if !feeds.iter().any(|&(i, _)| slots[i].as_ref().is_some_and(|s| s.cfg.is_some())) {
        return Ok(false);
    }
    thread_local! {
        static RAW: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let vocab = e.vocab();
    let rows = decode_feed_extent(feeds).unwrap_or(0);
    RAW.with_borrow_mut(|raw| {
        raw.resize(rows * vocab * 2, 0);
        e.read_tensor_range("act.logits", 0, raw)?;
        let raw: &[u8] = raw;
        let row = |r: usize| &raw[r * vocab * 2..(r + 1) * vocab * 2];
        // The owners' draws are independent (each reads its own rows, writes its own slot) and
        // ~40 us each at vocab 8194, serial they were 2.4 ms of GPU-idle time per step at 64
        // pairs: spread them over the rayon pool.
        let mut at = vec![usize::MAX; slots.len()];
        for (k, &(i, _)) in feeds.iter().enumerate() {
            at[i] = k;
        }
        let mut owners: Vec<(usize, &mut Slot)> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| {
                let s = s.as_mut().filter(|s| s.cfg.is_some() && at[i] != usize::MAX)?;
                Some((i, s))
            })
            .collect();
        use rayon::prelude::*;
        let drawn: Vec<(usize, u32)> = owners
            .par_iter_mut()
            .map(|(i, slot)| (at[*i], cfg_draw_bf16(slot, row(*i), row(*i + 1))))
            .collect();
        for (k, tok) in drawn {
            toks[k] = tok;
        }
        Ok(true)
    })
}

/// Stage every fed CFG owner's next `steps` guided draws on the device (`plow_sample_cfg`),
/// seeding its penalty history on the first. `Ok(false)` = no owner fed or no device CFG sampler
/// (the host draws). The uniforms are the owner's next `steps` SplitMix draws, the host path's.
#[cfg(feature = "cuda")]
fn gpu_stage_cfg(
    e: &mut crate::exec::gpu::GpuEngine,
    feeds: &[(usize, u32)],
    slots: &mut [Option<Slot>],
    steps: usize,
) -> Result<bool> {
    if !gpu_cfg_device(e) {
        return Ok(false);
    }
    let mut rows = Vec::new();
    let mut draws = Vec::new();
    for &(i, _) in feeds {
        let Some(slot) = slots[i].as_mut() else { continue };
        let (Some(run), Some(cfg)) = (slot.cfg.as_mut(), slot.speech.as_ref().and_then(|s| s.cfg.as_ref())) else {
            continue;
        };
        if !run.dev_seeded {
            e.cfg_seed_history(i, cfg.history.iter().chain(&slot.out_ids).copied())?;
            run.dev_seeded = true;
        }
        let p = &cfg.params;
        let mut rng = run.rng.clone();
        rows.push((
            i,
            crate::exec::gpu::DevCfg {
                weight: p.cfg_weight,
                penalty: p.repetition_penalty,
                temp: if rng.is_some() { p.temperature } else { 0.0 },
                top_p: p.top_p,
                min_p: p.min_p,
            },
        ));
        draws.extend((0..steps).map(|_| rng.as_mut().map_or(0.0, |r| r.unit())));
    }
    if rows.is_empty() {
        return Ok(false);
    }
    e.stage_cfg(&rows, &draws, steps)?;
    Ok(true)
}

/// CFG pairs draw on the device: the sampler object has `plow_sample_cfg` and `--cfg-device`.
#[cfg(feature = "cuda")]
fn gpu_cfg_device(e: &crate::exec::gpu::GpuEngine) -> bool {
    e.cfg_sampling() && crate::config::RuntimeConfig::get().nv.cfg_device
}

/// Advance every fed CFG owner's SplitMix past the `k` device draws it consumed.
#[cfg(feature = "cuda")]
fn gpu_cfg_advance(feeds: &[(usize, u32)], slots: &mut [Option<Slot>], k: usize) {
    for &(i, _) in feeds {
        if let Some(rng) = slots[i].as_mut().and_then(|s| s.cfg.as_mut()?.rng.as_mut()) {
            for _ in 0..k {
                rng.unit();
            }
        }
    }
}

/// A CFG owner's next token from its (cond, uncond) rows of the step's bf16 logits.
#[cfg(feature = "cuda")]
fn cfg_draw_bf16(slot: &mut Slot, cond: &[u8], uncond: &[u8]) -> u32 {
    let (Some(run), Some(cfg)) = (slot.cfg.as_mut(), slot.speech.as_ref().and_then(|s| s.cfg.as_ref())) else {
        unreachable!("cfg_draw on a CFG owner")
    };
    let u = run.rng.as_mut().map(|r| r.unit());
    let history = cfg.history.iter().chain(&slot.out_ids).copied();
    crate::text::sample::sample_cfg_bf16(&cfg.params, cond, uncond, history, u, &mut run.scratch)
}

/// A CFG owner's next token from its stashed (cond, uncond) logits.
#[cfg(feature = "cuda")]
fn cfg_draw(slot: &mut Slot) -> u32 {
    let (Some(run), Some(cfg)) = (slot.cfg.as_mut(), slot.speech.as_ref().and_then(|s| s.cfg.as_ref())) else {
        unreachable!("cfg_draw on a CFG owner")
    };
    let u = run.rng.as_mut().map(|r| r.unit());
    let history = cfg.history.iter().chain(&slot.out_ids).copied();
    crate::text::sample::sample_cfg(&cfg.params, &run.cond, &run.uncond, history, u, &mut run.scratch)
}

#[cfg(feature = "cuda")]
fn gpu_argmax_eligible(params: &crate::text::sample::SamplingParams) -> bool {
    params.temperature <= 0.0 && !params.needs_host_logits()
}

/// §HOSTT: close a timed decode engine call and open its emit loop (`tokens` = the tick's count
/// so far).
fn host_engine_call(t_call: Option<Instant>, rows: usize, tokens: usize) -> Option<(Instant, usize)> {
    let t = t_call?;
    crate::obs::host::engine_call(t.elapsed().as_nanos() as u64, rows);
    Some((Instant::now(), tokens))
}

fn host_emit_done(t_emit: Option<(Instant, usize)>, tokens: usize) {
    if let Some((t, before)) = t_emit {
        crate::obs::host::emit(t.elapsed().as_nanos() as u64, tokens - before);
    }
}

#[cfg(feature = "cuda")]
fn gpu_emit_slot_token(
    slot_opt: &mut Option<Slot>,
    arena: &Option<SharedKvState>,
    bundle: &ModelBundle,
    token: u32,
    tokens_this_tick: &mut usize,
    stop: &[u32],
) -> bool {
    handle_produced_token(
        slot_opt,
        arena,
        bundle,
        token,
        1,
        tokens_this_tick,
        Some(stop),
    )
}

#[cfg(feature = "cuda")]
fn gpu_finish_and_emit_token(
    e: &mut crate::exec::gpu::GpuEngine,
    row: usize,
    slot_idx: usize,
    slot_opt: &mut Option<Slot>,
    arena: &Option<SharedKvState>,
    bundle: &ModelBundle,
    raw_token: u32,
    skip_host_sample: bool,
    tokens_this_tick: &mut usize,
    stop: &[u32],
    tick_fault: &mut Option<crate::DeviceErrorInfo>,
    disconnected: &mut [bool],
) {
    let Some(slot) = slot_opt.as_mut() else { return };
    let finished = if skip_host_sample {
        Ok(raw_token)
    } else {
        gpu_finish_token(e, row, slot, raw_token)
    };
    match finished {
        Ok(token) => {
            tracing::debug!(
                token,
                slot = slot_idx,
                step = slot.step,
                "gpu: token"
            );
            disconnected[slot_idx] |= gpu_emit_slot_token(
                slot_opt,
                arena,
                bundle,
                token,
                tokens_this_tick,
                stop,
            );
        }
        Err(err) => {
            tracing::warn!(
                slot = slot_idx,
                error = %err,
                error_code = ?err.device_code(),
                fatal = err.is_fatal(),
                model = bundle.network(),
                "gpu: sample failed"
            );
            note_fault(tick_fault, &err);
            fail_slot(slot_opt, arena, err);
        }
    }
}

/// Incremental detokenize over a bounded window (TGI scheme): decode only
/// `out_ids[*prefix..]` — O(window) per token instead of O(total) — and emit
/// the bytes past the `*prefix..*read` span's decode. The window advances only
/// when new visible bytes appear; a trailing replacement char (partial UTF-8
/// sequence mid-multibyte-token) holds the delta back until the sequence
/// completes, unless `last`: the stream ends here and nothing will complete it.
/// Free-standing so tests can drive it without a mux.
fn incremental_delta(
    tok: &dyn crate::text::tokenizer::Tokenize,
    out_ids: &[u32],
    prefix: &mut usize,
    read: &mut usize,
    last: bool,
    keep_special: bool,
) -> String {
    const MAX_DETOKENIZE_WINDOW: usize = 16;
    let len = out_ids.len();
    let safe_start = len
        .saturating_sub(MAX_DETOKENIZE_WINDOW)
        .max((*prefix).min(len));
    let effective_read = (*read).clamp(safe_start, len);
    thread_local! {
        static TEXT: std::cell::RefCell<(String, String)> = const { std::cell::RefCell::new((String::new(), String::new())) };
    }
    TEXT.with_borrow_mut(|(prefix_text, new_text)| {
        prefix_text.clear();
        new_text.clear();
        tok.decode_append(&out_ids[safe_start..effective_read], keep_special, prefix_text);
        tok.decode_append(&out_ids[safe_start..], keep_special, new_text);
        match new_text.get(prefix_text.len()..) {
            Some(d) if !d.is_empty() && (last || !new_text.ends_with('\u{FFFD}')) => {
                let d = d.to_string();
                *prefix = effective_read;
                *read = len;
                d
            }
            _ => String::new(),
        }
    })
}

/// Common per-slot bookkeeping for a produced token: append to `out_ids`,
/// stream the incremental delta, and close/free the slot on stop conditions.
/// Shared by the batched, fallback, and GPU paths so error/exit semantics
/// stay identical. `stop_ids` overrides the reference path's newline-byte
/// heuristic with the model's real eos set (GPU path).
fn handle_produced_token(
    slot_opt: &mut Option<Slot>,
    arena: &Option<SharedKvState>,
    bundle: &ModelBundle,
    token: u32,
    exec: usize,
    tokens_this_tick: &mut usize,
    stop_ids: Option<&[u32]>,
) -> bool {
    // A finished slot parked on its consumer only drains; a token past its end is not output.
    let Some(slot) = slot_opt.as_mut().filter(|s| s.held_finish.is_none()) else {
        return false;
    };
    if let Some(telemetry) = slot.telemetry.as_mut() {
        telemetry.token(slot.cached_tokens);
    }
    slot.out_ids.push(token);
    slot.executed += exec;
    slot.step += 1;
    if let Some(seat) = slot.session.as_mut() {
        seat.on_token(token);
    }
    *tokens_this_tick += 1;

    // Token sends leave one channel entry for Done/Err. Backpressure ends
    // this request explicitly without blocking another model's submission thread.
    // Stop conditions: the model's eos set when known (GPU path), else the
    // reference path's newline-byte heuristic; and max_tokens.
    //
    // COMPUTED BEFORE THE SEND, because a stop token's TEXT is framing and must not
    // reach the caller. It used to be computed after, which was invisible while every
    // stop id rendered as the empty string — `skip_special_tokens` drops a token flagged
    // `special`. Kimi-K3's turn ends at `<|close|>`, which its own
    // `added_tokens_decoder` flags `"special": false`, so it renders literally and the
    // answer came back as `The capital of France is Paris.<|close|>`.
    // `min_tokens` holds EVERY stop condition off, not just the eos set: a
    // request that asks for at least N tokens and is cut short by a `stop`
    // string at token 3 has had the parameter ignored just as surely.
    let below_min = slot.out_ids.len() < slot.gen.min_tokens;
    let stop_token = !slot.gen.ignore_eos
        && !below_min
        && (slot.gen.stop_token_ids.contains(&token)
            || match stop_ids {
                Some(ids) => ids.contains(&token),
                None => token % 256 == u32::from(b'\n'),
            });
    let stop_max = slot.step >= slot.gen.max_tokens.max(1);
    // A stop token's own text stays out of the decode; the stream's last token flushes bytes held
    // for a UTF-8 continuation that will now never come.
    let delta = if slot.raw_tokens {
        String::new()
    } else {
        let n = slot.out_ids.len() - usize::from(stop_token);
        incremental_delta(
            bundle.tokenizer().as_ref(),
            &slot.out_ids[..n],
            &mut slot.prefix_offset,
            &mut slot.read_offset,
            stop_token || stop_max,
            slot.gen.keep_special_tokens,
        )
    };

    // OPENAI `stop` STRINGS. The request field was not parsed at all before, so
    // a client that relied on `stop` to end a step — every LangChain ReAct or
    // structured-output chain does — got an over-generated answer it then
    // mis-parsed. Matched on the streamed TEXT, not on ids, because a stop
    // sequence need not be a token and can straddle a token boundary.
    //
    // `ignore_eos` suppresses this too: a benchmark that asks for exactly
    // `--random-output-len` tokens must not be cut short by an accidental match.
    // Stop-string bookkeeping over the run of generated-but-unemitted bytes. Extracted so it
    // can be driven token by token in a test: both bugs it has carried lived in the SEQUENCING
    // of hold, release and cut, not in either helper, and nothing exercised that.
    let (delta, stop_string) =
        if !slot.gen.ignore_eos && !below_min && !slot.gen.stop.is_empty() {
            apply_stop_strings(
                &mut slot.stop_tail,
                &mut slot.stop_pending,
                delta,
                &slot.gen.stop,
            )
        } else {
            (delta, false)
        };
    // Held bytes are text once the request ends without a stop-string match; dropping them
    // truncated the answer.
    let delta = if (stop_token || stop_max) && !stop_string && !slot.stop_pending.is_empty() {
        let mut d = delta;
        d.push_str(&std::mem::take(&mut slot.stop_pending));
        d
    } else {
        delta
    };
    // A stop token's chunk carries only bytes released by the end of the stream.
    if !stop_token || !delta.is_empty() {
        let (id, lp) = if stop_token { (crate::serve::stream::TEXT_ONLY, None) } else { (token, slot.lp.take()) };
        if send_or_hold(slot, (id, delta, lp)) {
            if let Some(taken) = slot_opt.take() {
                release_kv(arena, taken.kv);
            }
            return true;
        }
    }
    if slot.step == 1 && crate::obs::host::on() {
        crate::obs::host::first_token(slot.prompt_ids.len(), slot.arrived.elapsed());
    }
    if stop_token || stop_max || stop_string {
        let reason = if stop_max && !stop_token && !stop_string {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        if slot.parked_at.is_some() {
            slot.held_finish = Some(reason);
            return false;
        }
        if let Some(telemetry) = slot.telemetry.as_mut() {
            telemetry.finish(reason, slot.executed);
        }
        let disconnected = slot.respond.try_send(StreamChunk::Done {
            executed: slot.executed,
            reason,
            usage: crate::serve::stream::TokenUsage {
                prompt_tokens: slot.prompt_ids.len(),
                cached_tokens: slot.cached_tokens,
                completion_tokens: slot.out_ids.len(),
            },
        }).is_err();
        if let Some(taken) = slot_opt.take() {
            release_kv(arena, taken.kv);
        }
        return disconnected;
    }
    false
}

/// Send one token chunk, or hold it and park the slot while the consumer is behind or earlier
/// chunks still wait; one channel entry stays free for the terminal. True when the consumer is
/// gone.
fn send_or_hold(slot: &mut Slot, (id, text, logprobs): HeldToken) -> bool {
    if slot.parked_at.is_none() && slot.respond.capacity() > 1 {
        return slot.respond.try_send(StreamChunk::Token { id, text, logprobs }).is_err();
    }
    slot.held.push((id, text, logprobs));
    slot.parked_at.get_or_insert_with(Instant::now);
    false
}

/// Drain a parked slot into its consumer's free capacity (one entry stays reserved for
/// the terminal). Returns true when the slot was freed: finished, disconnected, or parked past
/// [`PARK_TIMEOUT`].
fn flush_parked(slot_opt: &mut Option<Slot>, arena: &Option<SharedKvState>) -> bool {
    let Some(slot) = slot_opt.as_mut() else { return false };
    let Some(since) = slot.parked_at else { return false };
    if slot.respond.is_closed() {
        if let Some(taken) = slot_opt.take() {
            release_kv(arena, taken.kv);
        }
        return true;
    }
    let room = slot.respond.capacity().saturating_sub(1).min(slot.held.len());
    for (id, text, logprobs) in slot.held.drain(..room) {
        let _ = slot.respond.try_send(StreamChunk::Token { id, text, logprobs });
    }
    // PARK_TIMEOUT bounds a consumer that stopped reading, not one that reads slowly.
    if room > 0 && !slot.held.is_empty() {
        slot.parked_at = Some(Instant::now());
        return false;
    }
    if slot.held.is_empty() {
        let Some(reason) = slot.held_finish.take() else {
            slot.parked_at = None;
            return false;
        };
        if let Some(telemetry) = slot.telemetry.as_mut() {
            telemetry.finish(reason, slot.executed);
        }
        let _ = slot.respond.try_send(StreamChunk::Done {
            executed: slot.executed,
            reason,
            usage: crate::serve::stream::TokenUsage {
                prompt_tokens: slot.prompt_ids.len(),
                cached_tokens: slot.cached_tokens,
                completion_tokens: slot.out_ids.len(),
            },
        });
    } else if since.elapsed() > PARK_TIMEOUT {
        let _ = slot.respond.try_send(StreamChunk::Err(crate::RuntimeError::Rejected(
            "response consumer is too slow".into(),
        )));
    } else {
        return false;
    }
    if let Some(taken) = slot_opt.take() {
        release_kv(arena, taken.kv);
    }
    true
}

/// One token's worth of stop-string bookkeeping.
///
/// `pending` is the run of bytes GENERATED BUT NOT YET EMITTED: whatever an earlier token
/// withheld because it could still begin a stop string. This token's `delta` goes on the END of
/// that run, and every decision is taken over the whole run, in stream order. Returns the text
/// to emit and whether a stop string ended generation.
///
/// ONE FUNCTION, because two ordering bugs lived in the seam between the steps:
///
///   * holding and releasing used to be separate blocks in that order, so a token that withheld
///     a tail met the release block in the SAME call and got those bytes prepended back in
///     front of what remained — stop "three" against "Count: " withheld the final "t" and
///     emitted "tCoun";
///   * and the cut was an index into THIS token's delta while it was applied to the run, which
///     is a different string.
///
/// The cut is computed against the run directly. `earliest_stop_cut` clamps a match that STARTS
/// before the run to 0, which is the right answer and the subtle half: those bytes are the
/// beginning of the stop string itself, so the run is dropped rather than emitted. Adding the
/// carried length back to that 0 re-emitted them — stop "France is" over " France is" returned
/// " France" instead of " ".
fn apply_stop_strings(
    tail: &mut String,
    pending: &mut String,
    delta: String,
    stops: &[String],
) -> (String, bool) {
    let carried = std::mem::take(pending);
    let carried_len = carried.len();
    let mut run = if carried.is_empty() {
        delta
    } else {
        let mut run = carried;
        run.push_str(&delta);
        run
    };
    if run.is_empty() {
        return (run, false);
    }
    let keep = stops.iter().map(|s| s.len()).max().unwrap_or(0).saturating_sub(1);
    // Only this token's own bytes are new to the match window; the carried ones entered it when
    // they first arrived.
    tail.push_str(&run[carried_len..]);
    // The run is a suffix of the tail (the tail keeps at least `keep` bytes and a hold never
    // exceeds `keep`), so this maps a tail offset onto a run offset.
    let run_start = tail.len().saturating_sub(run.len());
    if let Some(cut) = earliest_stop_cut(tail, run_start, run.len(), stops) {
        let cut = (0..=cut)
            .rev()
            .find(|&c| run.is_char_boundary(c))
            .unwrap_or(0);
        run.truncate(cut);
        return (run, true);
    }
    if tail.len() > keep {
        let cut = tail.len() - keep;
        let cut = (0..=cut).rev().find(|&c| tail.is_char_boundary(c)).unwrap_or(0);
        tail.drain(..cut);
    }
    // WITHHOLD A TRAILING STOP PREFIX. Matching alone is not enough: the tail of the run may be
    // the start of a stop string whose remainder has not been generated yet, and sent bytes
    // cannot be recalled.
    let held = stop_prefix_held(tail, stops, run.len());
    if held > 0 {
        let keep_to = (0..=run.len() - held)
            .rev()
            .find(|&c| run.is_char_boundary(c))
            .unwrap_or(0);
        *pending = run[keep_to..].to_string();
        run.truncate(keep_to);
    }
    (run, false)
}

/// Byte offset within THIS delta at which output must stop, or `None` for no match.
///
/// `stop` is a SET, not a priority order: the cut is the earliest match in the text, over every
/// pattern. Scanning in list order and breaking on the first pattern that matched anywhere
/// returned text past an earlier stop — with `["BB", "A"]` over "xxAyyBB", "xxAyy" instead of
/// "xx". `base` is the accumulated tail's length before this delta was appended.
fn earliest_stop_cut(tail: &str, base: usize, delta_len: usize, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter_map(|pat| tail.find(pat.as_str()))
        .map(|i| i.saturating_sub(base).min(delta_len))
        .min()
}

/// Trailing bytes of this delta that must be withheld because they are a proper prefix of some
/// stop string whose remainder may still be generated.
///
/// Matching alone is not enough. Sent bytes cannot be recalled, so with stop "STOP" the deltas
/// "abcST" then "OPdef" match correctly on the second while the first has already emitted the
/// stop string's own prefix. Returns the longest such suffix, capped at this delta's length —
/// a prefix reaching back into earlier deltas was already sent and is not recoverable here.
fn stop_prefix_held(tail: &str, stops: &[String], delta_len: usize) -> usize {
    stops
        .iter()
        .filter_map(|pat| {
            (1..pat.len())
                .rev()
                .find(|&n| pat.is_char_boundary(n) && tail.ends_with(&pat[..n]))
        })
        .max()
        .unwrap_or(0)
        .min(delta_len)
}

/// A dispatcher with no engine: each job is answered with `script(&job)`'s pieces, one token
/// each, then `Done` (`length` when the pieces reach `max_tokens`). For HTTP-level tests of
/// the handlers.
#[cfg(test)]
pub(crate) fn scripted_mux(script: impl Fn(&Job) -> Vec<String> + Send + Sync + 'static) -> ModelMux {
    let (tx, mut rx) = mpsc::channel::<MuxMsg>(64);
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            let MuxMsg::Job(job, _) = msg else { continue };
            let pieces = script(&job);
            let n = pieces.len();
            for text in pieces {
                let logprobs = job.gen.params.logprobs.map(|_| Box::new(crate::text::logprobs::TokenLogprobs { logprob: -0.25, top: Vec::new() }));
                if job.respond.send(StreamChunk::Token { id: 7, text, logprobs }).await.is_err() {
                    break;
                }
            }
            let reason = if n >= job.gen.max_tokens { FinishReason::Length } else { FinishReason::Stop };
            let usage = crate::serve::stream::TokenUsage { prompt_tokens: job.prompt_ids.len(), cached_tokens: 0, completion_tokens: n };
            let _ = job.respond.send(StreamChunk::Done { executed: n, reason, usage }).await;
        }
    });
    ModelMux {
        tx,
        metrics: Arc::new(Metrics::default()),
        preempt: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        preempt_notify: Arc::new(tokio::sync::Notify::new()),
        arrival_notify: Arc::new(tokio::sync::Notify::new()),
        ingress: Arc::default(),
        preempted: Arc::default(),
    }
}

#[cfg(test)]
mod tests {
    /// The reported defect: a stop string split across two deltas leaked its own prefix.
    ///
    /// With stop "STOP" and deltas "abcST" then "OPdef", the match is only findable on the
    /// second — by which time "abcST" has been sent. `stop_prefix_held` is what withholds the
    /// "ST" so the client sees "abc".
    #[test]
    fn speech_overlay_index_is_chunk_relative_and_compacted() {
        // Audio rows at prompt positions 3..9 of a prompt chunked 4 rows at a time.
        let pos: Vec<u32> = (3..9).collect();
        let mut index = Vec::new();
        assert_eq!(speech_overlay_index(&pos, 0, 4, &mut index), (0, 1));
        assert_eq!(index, [u32::MAX, u32::MAX, u32::MAX, 0]);
        assert_eq!(speech_overlay_index(&pos, 4, 4, &mut index), (1, 5));
        assert_eq!(index, [0, 1, 2, 3]);
        assert_eq!(speech_overlay_index(&pos, 8, 4, &mut index), (5, 6));
        assert_eq!(index, [0, u32::MAX, u32::MAX, u32::MAX]);
        assert_eq!(speech_overlay_index(&pos, 12, 4, &mut index), (6, 6));
        assert!(index.iter().all(|&i| i == u32::MAX));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn raw_token_jobs_stream_ids_without_text() {
        let (mut slot, mut rx) = prefill_test_slot();
        slot.as_mut().unwrap().raw_tokens = true;
        let bundle = prefill_test_bundle("raw-tokens");
        let mut n = 0;
        assert!(!handle_produced_token(&mut slot, &None, &bundle, 104, 1, &mut n, Some(&[])));
        match rx.try_recv().unwrap() {
            StreamChunk::Token { id, text, .. } => assert_eq!((id, text.as_str()), (104, "")),
            _ => panic!("expected a token"),
        }
    }

    #[test]
    fn a_stop_string_split_across_deltas_does_not_leak_its_prefix() {
        let stops = vec!["STOP".to_string()];
        // First delta: no match yet, but "ST" is a live prefix and must be held back.
        assert_eq!(super::earliest_stop_cut("abcST", 0, 5, &stops), None);
        assert_eq!(super::stop_prefix_held("abcST", &stops, 5), 2);
        // Second delta completes it: nothing of "OPdef" survives.
        assert_eq!(
            super::earliest_stop_cut("abcSTOPdef", 5, 5, &stops),
            Some(0)
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn co_scheduled_prefill_rows_never_collapse_to_one_token() {
        // No prefill object (`pf_max_rows() == 0`) with interleaving unbounded:
        // the bound has to come from the decode-only cap, not from 1.
        assert_eq!(
            super::co_sched_prefill_rows(0, usize::MAX),
            super::CO_SCHED_DECODE_ONLY_ROWS
        );
        // A tighter interleave setting still wins.
        assert_eq!(super::co_sched_prefill_rows(0, 32), 32);
        // With a prefill object, one bucket's rows, capped by interleave.
        assert_eq!(super::co_sched_prefill_rows(8192, usize::MAX), 8192);
        assert_eq!(super::co_sched_prefill_rows(8192, 512), 512);
        // Never zero: the caller uses this as a chunk width.
        assert_eq!(super::co_sched_prefill_rows(0, 0), 1);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn riders_trim_wide_launches_that_spill_a_rung() {
        assert!(super::trim_for_riders(4096, 4160, 63, 4096, 4096));
        assert!(super::trim_for_riders(1024, 1088, 1, 1024, 1024));
        assert!(super::trim_for_riders(4224, 8192, 3, 4224, 0), "a wide spill costs more than a tail");
        assert!(!super::trim_for_riders(4096, 4096, 63, 4000, 0), "riders fit the bucket");
        assert!(!super::trim_for_riders(4096, 4160, 0, 4096, 0));
        assert!(!super::trim_for_riders(128, 256, 63, 128, 0), "a narrow spill beats a tail launch");
        assert!(!super::trim_for_riders(1024, 1088, 3, 1024, 0), "no waiting row would join the tail");
        assert!(super::trim_for_riders(1024, 1088, 3, 1024, 600), "the tail joins waiting rows");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn queue_pack_rows_pack_short_prompts_and_run_long_ones_alone() {
        // An empty queue leaves the static bound; a lone prompt takes its rows.
        assert_eq!(super::queue_pack_rows(&[], 256, 4224), 4224);
        assert_eq!(super::queue_pack_rows(&[1024], 256, 4224), 1024);
        // Long prompts run alone: packing a second delays the first by more than it saves.
        assert_eq!(super::queue_pack_rows(&[1024; 4], 256, 4224), 1024);
        // Short prompts share a launch, more of them the deeper the queue.
        assert_eq!(super::queue_pack_rows(&[128; 4], 256, 4224), 384);
        assert_eq!(super::queue_pack_rows(&[128; 16], 256, 4224), 1408);
        // Never past the bound, and the oldest prompt is cut to it rather than dropped.
        assert_eq!(super::queue_pack_rows(&[128; 16], 256, 512), 512);
        assert_eq!(super::queue_pack_rows(&[8192, 128], 256, 4224), 4224);
        // A long prompt's tail is topped up by the next long prompt: neither finishes sooner alone.
        assert_eq!(super::queue_pack_rows(&[2328, 4224], 256, 4221), 4221);
        // A prompt that a later launch holds whole is not split to top this one up.
        assert_eq!(super::queue_pack_rows(&[1024; 16], 512, 4224), 4096);
        assert_eq!(super::queue_pack_rows(&[4096, 4096], 512, 4224), 4096);
    }

    /// A held prefix that turns out not to begin a match is released, not dropped.
    #[test]
    fn a_prefix_that_does_not_complete_is_released() {
        let stops = vec!["STOP".to_string()];
        assert_eq!(super::stop_prefix_held("abcST", &stops, 5), 2);
        // "STx" is no longer a prefix of "STOP", so nothing is withheld and the earlier
        // "ST" is released ahead of this delta by the caller.
        assert_eq!(super::stop_prefix_held("abcSTx", &stops, 1), 0);
        assert_eq!(super::earliest_stop_cut("abcSTx", 5, 1, &stops), None);
    }

    /// Drive the whole hold/release/cut sequence over a token stream, which is where both of
    /// this function's bugs lived — each helper was already unit-tested and still correct.
    fn run_stops(stops: &[&str], tokens: &[&str]) -> (String, bool) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let (mut tail, mut pending, mut out) = (String::new(), String::new(), String::new());
        for t in tokens {
            let (emit, stopped) =
                super::apply_stop_strings(&mut tail, &mut pending, t.to_string(), &stops);
            out.push_str(&emit);
            if stopped {
                return (out, true);
            }
        }
        (out, false)
    }

    /// A hold must not be released by the very call that created it. Stop "three" over
    /// "Count: " withholds the final "t" (a live prefix) and used to emit "tCoun".
    #[test]
    fn a_hold_is_not_released_into_its_own_token() {
        let (out, stopped) = run_stops(&["three"], &["Count", ":", " one", " two"]);
        assert_eq!(out, "Count: one two");
        assert!(!stopped);
    }

    /// The same stream, now actually reaching the stop string.
    #[test]
    fn the_held_prefix_is_dropped_when_the_match_completes() {
        let (out, stopped) = run_stops(&["three"], &["Count", ":", " one", " two", " three", "!"]);
        assert_eq!(out, "Count: one two ");
        assert!(stopped);
    }

    /// A match that STARTS inside the held bytes must drop them, not re-emit them. Stop
    /// "France is" over tokens " France" + " is" returned " France" once the cut was rebased
    /// onto the run by adding the carried length to a clamped 0.
    #[test]
    fn a_match_starting_inside_the_held_bytes_drops_them() {
        let (out, stopped) = run_stops(&["France is"], &[" France", " is", " Paris"]);
        assert_eq!(out, " ");
        assert!(stopped);
    }

    /// A stop string spanning two tokens, cutting mid-token.
    #[test]
    fn a_stop_spanning_several_tokens_cuts_at_its_start() {
        let (out, stopped) = run_stops(&["STOP"], &["abcST", "OPdef"]);
        assert_eq!(out, "abc");
        assert!(stopped);
    }

    /// Nothing is lost when no stop ever matches.
    #[test]
    fn a_stream_with_no_match_emits_every_byte() {
        let (out, stopped) = run_stops(&["zzz"], &["hello", " wor", "ld", "!"]);
        assert_eq!(out, "hello world!");
        assert!(!stopped);
    }

    /// Multibyte text must not be split inside a character.
    #[test]
    fn multibyte_text_survives_the_hold() {
        let (out, stopped) = run_stops(&["END"], &["\u{65e5}\u{672c}\u{8a9e}", "E", "ND", "x"]);
        assert_eq!(out, "\u{65e5}\u{672c}\u{8a9e}");
        assert!(stopped);
    }

    /// `stop` is a set: the cut is the earliest match in the TEXT, not the first pattern listed.
    #[test]
    fn multiple_stops_cut_at_the_earliest_match_not_the_first_listed() {
        let stops = vec!["BB".to_string(), "A".to_string()];
        // "A" at 2 precedes "BB" at 5; list order would have returned 5.
        assert_eq!(super::earliest_stop_cut("xxAyyBB", 0, 7, &stops), Some(2));
    }

    /// Longest live prefix wins when several stops share a lead, and a stop already fully
    /// matched withholds nothing (the cut handles it).
    #[test]
    fn the_longest_live_prefix_is_the_one_withheld() {
        let stops = vec!["END".to_string(), "ENDING".to_string()];
        assert_eq!(super::stop_prefix_held("textEN", &stops, 6), 2);
        // Capped at the delta: bytes from earlier deltas are already sent.
        assert_eq!(super::stop_prefix_held("textEN", &stops, 1), 1);
        // No stop and no prefix.
        assert_eq!(super::stop_prefix_held("plain", &stops, 5), 0);
    }

    use super::*;
    use crate::exec::indirection::slots as ind_slots;
    use crate::serve::RunObserver;
    use plow_asset::{KvLayerPaging, KvPaging};

    /// Every ingress event deposits into λ, and the gauge reports what the rung controller
    /// reads. A one-per-second stream must settle on ~1/s, not on the reciprocal EWMA's answer.
    #[test]
    fn arrival_rate_is_updated_once_per_ingress_event() {
        let metrics = Metrics::default();
        let mut load = LoadEstimator::default();
        let first = Instant::now();

        note_arrival(first, &mut load, &metrics);
        assert_eq!(metrics.lambda_milli.load(Ordering::Relaxed), 500);
        for i in 1..30 {
            note_arrival(first + std::time::Duration::from_secs(i), &mut load, &metrics);
        }
        let lambda = metrics.lambda_milli.load(Ordering::Relaxed) as f64 / 1000.0;
        assert!((lambda - 1.0).abs() < 0.3, "1 req/s should read ~1, got {lambda}");
    }

    /// Idle time alone must move λ, and with it the utilization gauge — the whole point of the
    /// decaying estimator. Nothing calls `note_arrival` between the two reads here.
    #[test]
    fn utilization_falls_during_a_quiet_period_with_no_ingress_event() {
        let mut load = LoadEstimator::default();
        let t0 = Instant::now();
        for i in 0..20 {
            load.lambda.observe(t0 + std::time::Duration::from_millis(100 * i));
        }
        load.service_ms.update(40.0);
        let busy = load.utilization(t0 + std::time::Duration::from_millis(2_000));
        let quiet = load.utilization(t0 + std::time::Duration::from_secs(12));
        assert!(busy > 0.0 && quiet < busy * 0.01, "busy {busy} quiet {quiet}");
    }

    fn test_job() -> Job {
        let (respond, _rx) = crate::serve::stream::channel();
        Job {
            prompt_ids: vec![1],
            gen: GenParams::default(),
            arrived: Instant::now(),
            respond,
            opts: Default::default(),
        }
    }

    #[test]
    fn deferred_token_ring_is_row_major_and_bounds_checked() {
        let ring = [10, 11, 12, 20, 21, 22];
        assert_eq!(deferred_token(&ring, 1, 2, 3).unwrap(), 22);
        assert!(deferred_token(&ring, 2, 0, 3).is_err());
    }

    /// §B1. The AMD deferred-read quantum must not collapse at the batch sizes continuous
    /// batching actually runs at. `MultiStep::for_batch` returns 1 for every B > 8, so routing
    /// the AMD request through it made `quantum >= 2` unreachable at C16 and decode paid a full
    /// host turnaround on every token.
    #[cfg(feature = "hsa")]
    #[test]
    fn the_amd_quantum_survives_a_batch_wider_than_eight() {
        // The defect, stated as the old call would have computed it.
        for batch in [9, 16, 32, 70] {
            let collapsed =
                multistep_requested(512, MultiStep::for_batch(batch).steps as usize, 8);
            assert_eq!(collapsed, 1, "control: for_batch collapses at B={batch}");
        }
        // Batch size no longer bounds it; the knob does.
        assert_eq!(amd_multistep_requested(512, true, 8), 8);
        assert_eq!(amd_multistep_requested(512, true, 4), 4);
        // `--multi-step` off still means single-step, which `steps == 1` could not express.
        assert_eq!(amd_multistep_requested(512, false, 8), 1);
        // The client's remaining output still bounds it, and a zero knob cannot wedge it.
        assert_eq!(amd_multistep_requested(3, true, 8), 3);
        assert_eq!(amd_multistep_requested(0, true, 8), 0);
        assert_eq!(amd_multistep_requested(512, true, 0), 1);
        // And the nominal default is now reachable on this backend, which a ceiling of 4 made
        // impossible: `decode_quantum` clamps the request to DEFERRED_TOKEN_MAX_STEPS.
        let positions = [0u32; 16];
        assert_eq!(
            crate::sched::multistep::decode_quantum(
                0..16,
                &positions,
                81_920,
                amd_multistep_requested(512, true, 8),
                crate::exec::amd::DEFERRED_TOKEN_MAX_STEPS,
            ),
            Ok(8)
        );
    }

    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    #[test]
    fn multistep_honors_scheduler_runtime_and_output_caps() {
        for (batch, expected) in [(1, 4), (4, 2), (16, 1)] {
            let steps = MultiStep::for_batch(batch).steps as usize;
            assert_eq!(multistep_requested(16, steps, 8), expected);
        }
        assert_eq!(multistep_requested(16, 1, 8), 1);
        assert_eq!(multistep_requested(16, 8, 4), 4);
        assert_eq!(multistep_requested(16, 8, 1), 1);
        assert_eq!(multistep_requested(16, 8, 0), 1);
        assert_eq!(multistep_requested(3, 8, 4), 3);
        assert_eq!(multistep_requested(0, 8, 4), 0);
    }

    #[test]
    fn bounded_ingress_reports_full_closed_and_depth() {
        let metrics = Arc::new(Metrics::default());
        let (tx, mut rx) = mpsc::channel(1);
        let mux = ModelMux {
            tx,
            metrics: Arc::clone(&metrics),
            preempt: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            preempt_notify: Arc::new(tokio::sync::Notify::new()),
            arrival_notify: Arc::new(tokio::sync::Notify::new()),
            ingress: Arc::default(),
            preempted: Arc::default(),
        };

        let (a, b) = (mux.ingress(), mux.ingress());
        assert_eq!(mux.ingress.pending(), 2);
        drop(a);
        assert_eq!(mux.ingress.pending(), 1);
        drop(b);
        assert_eq!(mux.ingress.pending(), 0);

        assert!(mux.submit(test_job()).is_ok());
        assert_eq!(metrics.queued_requests.load(Ordering::Relaxed), 1);
        assert!(matches!(mux.submit(test_job()), Err(SubmitError::Full(_))));
        assert_eq!(metrics.queued_requests.load(Ordering::Relaxed), 1);

        let msg = rx.try_recv().unwrap();
        note_dequeued(&msg, &metrics);
        assert_eq!(metrics.queued_requests.load(Ordering::Relaxed), 0);
        drop(rx);
        assert!(matches!(
            mux.submit(test_job()),
            Err(SubmitError::Closed(_))
        ));
        assert_eq!(metrics.queued_requests.load(Ordering::Relaxed), 0);
    }

    #[cfg(feature = "cuda")]
    fn idle_test_mux(name: &str) -> ModelMux {
        let backend: Arc<dyn crate::device::Backend> = Arc::new(crate::device::cpu::CpuBackend::new(1));
        let execset = Arc::new(crate::exec::ExecutorSet::bringup(backend).unwrap());
        let state = Arc::new(AppState::new(crate::orch::Registry::new(), execset));
        spawn(name.into(), Arc::new(prefill_test_bundle(name)), state, MuxConfig::default())
    }

    /// A DP rank taken out mid-burst: the router stops choosing it the moment its dispatcher leaves
    /// the table, and a job already holding its (now closed) dispatcher is resubmitted to the other
    /// rank instead of failing.
    #[cfg(feature = "cuda")]
    #[tokio::test]
    async fn a_rank_removed_mid_burst_loses_no_request() {
        use crate::serve::dp::{DpRouter, DpSet, RouteCfg};
        let backend: Arc<dyn crate::device::Backend> = Arc::new(crate::device::cpu::CpuBackend::new(1));
        let execset = Arc::new(crate::exec::ExecutorSet::bringup(backend).unwrap());
        let state = Arc::new(AppState::new(crate::orch::Registry::new(), execset));
        let mut router = DpRouter::new(RouteCfg::default());
        router.add(DpSet::new("m", (0..2).map(|r| (r as u32, r, state.model_metrics(&format!("m#{r}")))).collect()));
        state.install_dp(router);
        for r in 0..2 {
            let key = format!("m#{r}");
            let mux = spawn(key.clone(), Arc::new(prefill_test_bundle(&format!("dp-burst-{r}"))), Arc::clone(&state), MuxConfig::default());
            state.install_mux(key, mux);
        }
        let set = Arc::clone(state.dp_set("m").unwrap());
        let mut seen = [0usize; 2];
        for _ in 0..6 {
            let (rank, mux, _, _pick) = state.dp_route(&set, None, Some(&[1, 2, 3]), 0).unwrap();
            seen[rank] += 1;
            assert!(state.submit_routed(Some((&set, rank)), None, &mux, test_job(), Instant::now(), None).is_ok());
        }
        assert!(seen.iter().all(|&n| n > 0), "both ranks take work: {seen:?}");

        let stale = state.mux("m#0").unwrap();
        let removed = state.remove_mux("m#0").unwrap();
        removed.drain().await;
        assert!(stale.is_closed());
        for _ in 0..16 {
            let (rank, _, _, _) = state.dp_route(&set, Some("s"), None, 0).unwrap();
            assert_eq!(rank, 1, "a removed rank is never routed to");
        }
        assert!(state.submit_routed(Some((&set, 0)), None, &stale, test_job(), Instant::now(), None).is_ok());
        assert_eq!(set.stats.retries.load(Ordering::Relaxed), 1);
        state.remove_mux("m#1");
        assert!(state.dp_route(&set, None, None, 0).is_none());
        assert!(matches!(
            state.submit_routed(Some((&set, 0)), None, &stale, test_job(), Instant::now(), None),
            Err(SubmitError::Closed(_))
        ));
    }

    /// An idle stream holding ingress must not hold a preempt: it completes, and the held work's
    /// later submission is told the model was preempted.
    #[cfg(feature = "cuda")]
    #[tokio::test]
    async fn preempt_does_not_wait_for_held_ingress() {
        let mux = idle_test_mux("preempt-held-ingress");
        let held = mux.ingress_owned();
        tokio::time::timeout(std::time::Duration::from_secs(5), mux.preempt()).await.expect("preempt waited on ingress");
        assert!(mux.preempted());
        assert!(matches!(mux.submit_wait(test_job()).await, Err(SubmitError::Closed(_))));
        drop(held);
    }

    /// A graceful drain held only by ingress completes once that request leaves without submitting.
    #[cfg(feature = "cuda")]
    #[tokio::test]
    async fn graceful_drain_wakes_when_ingress_leaves() {
        let mux = idle_test_mux("drain-ingress-leaves");
        let held = mux.ingress_owned();
        let draining = {
            let mux = mux.clone();
            tokio::spawn(async move { mux.drain().await })
        };
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(!draining.is_finished(), "a graceful drain closed on a request past model lookup");
        drop(held);
        tokio::time::timeout(std::time::Duration::from_secs(5), draining).await.expect("drain never woke").unwrap();
        assert!(!mux.preempted());
    }

    #[test]
    fn disconnected_job_is_not_admitted() {
        let metrics = Arc::new(Metrics::default());
        let mut slots: Vec<Option<Slot>> = std::iter::once_with(|| None).collect();

        assert!(admit_into(
            &mut slots,
            1,
            test_job(),
            Instant::now(),
            None,
            &metrics,
            &EngineHealth::Healthy,
            None,
            false,
        )
        .is_none());
        assert!(slots[0].is_none());
    }

    /// A queued job whose client is still connected; the caller must keep the receiver alive.
    fn queued_job(
        prompt: usize,
        arrived: Instant,
    ) -> ((Job, Instant), crate::serve::stream::ChunkReceiver) {
        let (respond, rx) = crate::serve::stream::channel();
        let job = Job {
            prompt_ids: vec![1; prompt],
            gen: GenParams {
                max_tokens: 1,
                ..GenParams::default()
            },
            arrived,
            respond,
            opts: Default::default(),
        };
        ((job, arrived), rx)
    }

    /// 1000 rows of budget at one byte per row; a slot table nothing else is using.
    fn starvation_fixture() -> (
        Arc<Metrics>,
        Vec<Option<Slot>>,
        crate::sched::admission::KvBudget,
        Vec<crate::serve::stream::ChunkReceiver>,
    ) {
        let metrics = Arc::new(Metrics::default());
        let budget = crate::sched::admission::KvBudget::linear(1, 1_000);
        let mut slots: Vec<Option<Slot>> = (0..8).map(|_| None).collect();
        // One live sequence already holding 301 rows, so a 701-row request cannot be seated
        // until something retires, but a 101-row one can.
        let ((job, arrived), rx) = queued_job(300, Instant::now());
        assert!(admit_into(
            &mut slots,
            8,
            job,
            arrived,
            None,
            &metrics,
            &EngineHealth::Healthy,
            Some(budget),
            false,
        )
        .is_none());
        (metrics, slots, budget, vec![rx])
    }

    /// The backfill is a throughput win and stays on until the head has actually waited. Both
    /// halves are the contract, so both are asserted here.
    #[test]
    fn a_young_head_yields_to_backfill_and_an_aged_one_stops_it() {
        let now = Instant::now();
        let aging = queue_aging_ms(250.0);
        let young = now - std::time::Duration::from_millis(10);
        let aged = now - std::time::Duration::from_secs_f64(aging / 1e3 + 1.0);

        for (head_arrived, admitted_behind, remaining) in [(young, 3, 1), (aged, 0, 4)] {
            let (metrics, mut slots, budget, mut keep) = starvation_fixture();
            let live_before = slots.iter().flatten().count();
            let mut waiting: std::collections::VecDeque<(Job, Instant)> =
                std::collections::VecDeque::new();
            // A 701-row head that does not fit behind the live sequence...
            let (entry, rx) = queued_job(700, head_arrived);
            keep.push(rx);
            waiting.push_back(entry);
            // ...and three 101-row requests that do.
            for _ in 0..3 {
                let (entry, rx) = queued_job(100, now - std::time::Duration::from_millis(5));
                keep.push(rx);
                waiting.push_back(entry);
            }

            drain_waiting(
                &mut waiting,
                &mut slots,
                8,
                now,
                250.0,
                None,
                &metrics,
                &EngineHealth::Healthy,
                Some(budget),
                false,
            );

            assert_eq!(
                slots.iter().flatten().count() - live_before,
                admitted_behind,
                "younger requests admitted past the head"
            );
            assert_eq!(waiting.len(), remaining);
            // The head keeps its place either way — aging reorders nothing.
            assert_eq!(waiting.front().unwrap().0.prompt_ids.len(), 700);
        }
    }

    /// Repeated passes must not starve the aged head: once it blocks the backfill, the only
    /// thing it waits on is retirement, and it takes the slot the moment one frees.
    #[test]
    fn an_aged_head_is_seated_as_soon_as_a_live_sequence_retires() {
        let now = Instant::now();
        let (metrics, mut slots, budget, mut keep) = starvation_fixture();
        let mut waiting: std::collections::VecDeque<(Job, Instant)> =
            std::collections::VecDeque::new();
        let (entry, rx) = queued_job(700, now - std::time::Duration::from_secs(5));
        keep.push(rx);
        waiting.push_back(entry);
        for _ in 0..3 {
            let (entry, rx) = queued_job(100, now);
            keep.push(rx);
            waiting.push_back(entry);
        }

        let mut pass = |slots: &mut Vec<Option<Slot>>, waiting: &mut _| {
            drain_waiting(
                waiting,
                slots,
                8,
                now,
                250.0,
                None,
                &metrics,
                &EngineHealth::Healthy,
                Some(budget),
                false,
            );
        };
        pass(&mut slots, &mut waiting);
        assert_eq!(waiting.len(), 4, "blocked while the budget is held");

        // The live sequence retires, freeing its 301 rows. The head takes them, and the
        // backfill resumes behind it for as many of the 101-row requests as still fit
        // (701 + 101 + 101 = 903 of 1000; the fourth is held again, this time fairly).
        slots[0] = None;
        pass(&mut slots, &mut waiting);
        assert!(
            slots.iter().flatten().any(|s| s.prompt_ids.len() == 700),
            "the aged head goes first"
        );
        assert_eq!(waiting.len(), 1);
        assert_eq!(waiting.front().unwrap().0.prompt_ids.len(), 100);
    }

    /// A queued entry whose client vanished must be swept even when no slot is free — the
    /// per-request check inside `admit_into` only runs when there is somewhere to put it.
    #[test]
    fn waiting_entries_are_swept_for_disconnects_with_a_full_slot_table() {
        let now = Instant::now();
        let metrics = Arc::new(Metrics::default());
        let mut slots: Vec<Option<Slot>> = Vec::new();
        let mut waiting: std::collections::VecDeque<(Job, Instant)> =
            std::collections::VecDeque::new();
        metrics.queued_requests.store(2, Ordering::Relaxed);
        let (gone, rx) = queued_job(10, now);
        drop(rx);
        waiting.push_back(gone);
        let (live, keep) = queued_job(10, now);
        waiting.push_back(live);

        drain_waiting(
            &mut waiting,
            &mut slots,
            0,
            now,
            250.0,
            None,
            &metrics,
            &EngineHealth::Healthy,
            None,
            false,
        );

        assert_eq!(waiting.len(), 1, "the disconnected entry is gone");
        assert_eq!(metrics.queued_requests.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.admit_shed.load(Ordering::Relaxed), 0, "not a shed");
        drop(keep);
    }

    /// The TTL is the one shed plow performs, and it only ever touches a request that has not
    /// started. It answers the stream rather than dropping it silently.
    #[test]
    fn queue_ttl_override_sets_or_disables_the_shed() {
        assert_eq!(queue_ttl_with(250.0, None), QUEUE_TTL_FLOOR_MS);
        assert_eq!(queue_ttl_with(1_000.0, None), 40_000.0);
        assert_eq!(queue_ttl_with(250.0, Some(600_000.0)), 600_000.0);
        assert!(queue_ttl_with(250.0, Some(0.0)).is_infinite());
        assert_eq!(queue_verdict(false, 1e9, 250.0, JobClass::Normal), Queued::Expired);
    }

    #[test]
    fn a_request_past_the_queue_ttl_is_shed_with_an_answer() {
        let now = Instant::now();
        let metrics = Arc::new(Metrics::default());
        let mut slots: Vec<Option<Slot>> = Vec::new();
        let mut waiting: std::collections::VecDeque<(Job, Instant)> =
            std::collections::VecDeque::new();
        metrics.queued_requests.store(1, Ordering::Relaxed);
        let ttl = queue_ttl_ms(250.0);
        let (stale, mut rx) = queued_job(
            10,
            now - std::time::Duration::from_secs_f64(ttl / 1e3 + 1.0),
        );
        waiting.push_back(stale);

        drain_waiting(
            &mut waiting,
            &mut slots,
            0,
            now,
            250.0,
            None,
            &metrics,
            &EngineHealth::Healthy,
            None,
            false,
        );

        assert!(waiting.is_empty());
        assert_eq!(metrics.admit_shed.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.queued_requests.load(Ordering::Relaxed), 0);
        assert!(matches!(
            rx.try_recv(),
            Ok(StreamChunk::Err(crate::RuntimeError::Rejected(_)))
        ));
    }

    /// A critical arrival is seated ahead of older normal ones; a bulk one past its (shorter)
    /// TTL is shed while a normal one of the same age still waits.
    #[test]
    fn waiting_is_served_by_class_and_bulk_goes_stale_first() {
        let now = Instant::now();
        let metrics = Arc::new(Metrics::default());
        let mut slots: Vec<Option<Slot>> = std::iter::repeat_with(|| None).take(1).collect();
        let mut waiting = std::collections::VecDeque::new();
        let ms = |v: f64| std::time::Duration::from_secs_f64(v / 1e3);
        let (normal, _rx_n) = queued_job(10, now - ms(50.0));
        let ((mut critical, t), _rx_c) = queued_job(20, now - ms(10.0));
        critical.opts.class = JobClass::Critical;
        let stale = now - ms(queue_ttl_ms(250.0) * 0.5);
        let ((mut bulk, tb), mut rx_b) = queued_job(30, stale);
        bulk.opts.class = JobClass::Bulk;
        let (old_normal, _rx_o) = queued_job(40, stale);
        waiting.extend([normal, (critical, t), (bulk, tb), old_normal]);
        metrics.queued_requests.store(4, Ordering::Relaxed);
        drain_waiting(
            &mut waiting, &mut slots, 1, now, 250.0, None, &metrics, &EngineHealth::Healthy, None,
            false,
        );
        assert_eq!(slots[0].as_ref().unwrap().prompt_ids.len(), 20);
        assert!(matches!(rx_b.try_recv(), Ok(StreamChunk::Err(_))));
        let left: Vec<_> = waiting.iter().map(|(j, _)| j.prompt_ids.len()).collect();
        assert_eq!(left, [40, 10], "oldest first within a class");
    }

    #[test]
    fn cache_first_seats_the_most_cached_waiter_by_a_block_within_the_head_class() {
        use super::{cache_first, JobClass::*, CACHE_FIRST_WAIT_MS};
        let n = |w: f64| (Normal, w);
        // A block better than the head wins; ties go to the earlier waiter.
        assert_eq!(cache_first(&[n(5.0), n(4.0), n(3.0)], &[1536, 6144, 6144], 2048), Some(1));
        // Less than a block better: FIFO.
        assert_eq!(cache_first(&[n(5.0), n(4.0)], &[1536, 3500], 2048), None);
        // The head already holds the most cache.
        assert_eq!(cache_first(&[n(5.0), n(4.0)], &[8192, 2048], 2048), None);
        // Only the head's class competes: a cached bulk waiter never overtakes it.
        assert_eq!(cache_first(&[n(5.0), (Bulk, 9.0)], &[0, 8192], 2048), None);
        // The head keeps its seat once it has waited the bound.
        assert_eq!(cache_first(&[n(CACHE_FIRST_WAIT_MS), n(1.0)], &[0, 8192], 2048), None);
        assert_eq!(cache_first(&[], &[], 2048), None);
    }

    /// A continuing session turn takes the slot ahead of a session opening, but only until the
    /// opening request has waited the bound; past it, arrival order decides.
    #[test]
    fn continuing_turns_go_first_within_the_wait_bound() {
        let now = Instant::now();
        let bound = crate::serve::cosched::max_wait();
        let seat = |opening_waited: std::time::Duration| {
            let metrics = Arc::new(Metrics::default());
            let mut slots: Vec<Option<Slot>> = std::iter::repeat_with(|| None).take(1).collect();
            let (opening, _rx_o) = queued_job(10, now - opening_waited);
            let ((mut next, t), _rx_c) = queued_job(20, now);
            next.opts.continuing = true;
            let mut waiting: std::collections::VecDeque<_> = [opening, (next, t)].into();
            metrics.queued_requests.store(2, Ordering::Relaxed);
            drain_waiting(
                &mut waiting, &mut slots, 1, now, 250.0, None, &metrics, &EngineHealth::Healthy, None,
                false,
            );
            slots[0].as_ref().unwrap().prompt_ids.len()
        };
        assert_eq!(seat(bound / 2), 20, "the continuing turn overtakes a young opening");
        assert_eq!(seat(bound + std::time::Duration::from_millis(1)), 10, "never past the bound");

        let o = |continuing, ago_ms| {
            seat_order(JobClass::Normal, continuing, now - std::time::Duration::from_millis(ago_ms), now, bound)
        };
        let aged = bound.as_millis() as u64 + 10;
        assert!(o(false, aged + 5) < o(false, aged), "aged requests are oldest first");
        assert!(o(false, aged) < o(true, 0));
        assert!(o(true, 0) < o(false, 500));
        assert!(o(false, 500) < o(false, 100));
        let critical = seat_order(JobClass::Critical, false, now, now, bound);
        assert!(critical < o(false, aged), "class still comes first");
    }

    fn cfg_job(prompt: usize) -> ((Job, Instant), crate::serve::stream::ChunkReceiver) {
        let ((mut job, t), rx) = queued_job(prompt, Instant::now());
        job.opts.speech = Some(Box::new(SpeechJob {
            cfg: Some(CfgJob {
                uncond_overlay: Vec::new(),
                params: crate::text::sample::CfgParams {
                    cfg_weight: 0.5,
                    temperature: 1.0,
                    min_p: 0.0,
                    top_p: 1.0,
                    repetition_penalty: 1.0,
                },
                history: Vec::new(),
                seed: None,
            }),
            ..Default::default()
        }));
        ((job, t), rx)
    }

    /// A CFG pair takes an even slot and its odd partner, charges KV for both, and keeps the
    /// partner from every other job while the owner lives; a job that finds only such slots
    /// waits instead of being refused.
    #[test]
    fn cfg_pairs_seat_on_even_odd_slots_and_reserve_the_partner() {
        let metrics = Arc::new(Metrics::default());
        let mut slots: Vec<Option<Slot>> = std::iter::repeat_with(|| None).take(4).collect();
        let admit = |slots: &mut Vec<Option<Slot>>, (job, t): (Job, Instant), budget| {
            admit_into(slots, 4, job, t, None, &metrics, &EngineHealth::Healthy, budget, false)
        };
        let (plain, _r0) = queued_job(10, Instant::now());
        assert!(admit(&mut slots, plain, None).is_none());
        let (pair, _r1) = cfg_job(10);
        assert!(admit(&mut slots, pair, None).is_none());
        assert!(slots[2].as_ref().is_some_and(|s| s.cfg.is_some()) && slots[3].is_none());
        assert!(!slot_free(&slots, 3) && slot_free(&slots, 1));
        let (plain, _r2) = queued_job(10, Instant::now());
        assert!(admit(&mut slots, plain, None).is_none());
        assert!(slots[1].is_some() && slots[3].is_none());
        // Only the reserved partner is idle: both kinds wait.
        let (pair, mut r3) = cfg_job(10);
        assert!(admit(&mut slots, pair, None).is_some());
        let (plain, mut r4) = queued_job(10, Instant::now());
        assert!(admit(&mut slots, plain, None).is_some());
        assert!(r3.try_recv().is_err() && r4.try_recv().is_err());
        // The pair's KV is charged twice: 3 live rows-holders x 11 + a new pair's 22 > 50.
        slots[1] = None;
        let budget = crate::sched::admission::KvBudget::linear(1, 50);
        let (pair, _r5) = cfg_job(10);
        assert!(admit(&mut slots, pair, Some(budget)).is_some());
    }

    /// The partner steps on its owner's token, right after it.
    #[cfg(feature = "cuda")]
    #[test]
    fn cfg_owners_feed_their_partner() {
        let (mut owner, _rx) = prefill_test_slot();
        {
            let s = owner.as_mut().unwrap();
            s.step = 1;
            s.out_ids.push(7);
            s.cfg = Some(Box::new(CfgRun { rng: None, partner_pf: 0, cond: Vec::new(), uncond: Vec::new(), scratch: Vec::new(), dev_seeded: false }));
        }
        let slots = vec![owner, None];
        assert_eq!(gpu_decode_feeds(&slots, 2), [(0, 7), (1, 7)]);
    }

    /// A packed launch stages each member's overlay rows at its launch offset, in pack order: a
    /// pair's partner reads the unconditional rows, and a row no member overlays is unmapped.
    #[test]
    fn packed_overlay_rows_land_at_each_members_launch_offset() {
        let metrics = Arc::new(Metrics::default());
        let mut slots: Vec<Option<Slot>> = std::iter::repeat_with(|| None).take(4).collect();
        // Slot 0: an audio job, overlay rows at prompt positions 1..3, hidden 2.
        let ((mut job, t), _r0) = queued_job(6, Instant::now());
        job.opts.speech = Some(Box::new(SpeechJob {
            overlay: vec![1.0, 1.0, 2.0, 2.0],
            overlay_pos: vec![1, 2],
            ..Default::default()
        }));
        assert!(admit_into(&mut slots, 4, job, t, None, &metrics, &EngineHealth::Healthy, None, false).is_none());
        // Slots 2/3: a CFG pair whose every row is an overlay.
        let ((mut job, t), _r1) = cfg_job(3);
        {
            let sp = job.opts.speech.as_mut().unwrap();
            sp.overlay = vec![10.0, 10.0, 11.0, 11.0, 12.0, 12.0];
            sp.overlay_pos = vec![0, 1, 2];
            sp.cfg.as_mut().unwrap().uncond_overlay = vec![20.0, 20.0, 21.0, 21.0, 22.0, 22.0];
        }
        assert!(admit_into(&mut slots, 4, job, t, None, &metrics, &EngineHealth::Healthy, None, false).is_none());
        assert!(slots[2].as_ref().is_some_and(|s| s.cfg.is_some()) && slots[3].is_none());

        // Launch: slot 0 rows [0, 2), partner rows [1, 3), owner rows [0, 2).
        let pack = [(0, 0, 2), (3, 1, 2), (2, 0, 2)];
        let (mut rows, mut index) = (Vec::new(), Vec::new());
        speech_pack_overlay(&slots, &pack, 8, &mut rows, &mut index);
        assert_eq!(index, [u32::MAX, 0, 1, 2, 3, 4, u32::MAX, u32::MAX]);
        assert_eq!(rows, [1.0, 1.0, 21.0, 21.0, 22.0, 22.0, 10.0, 10.0, 11.0, 11.0]);

        // Frontiers: the partner's lives in its owner; the pair is done when both members are.
        assert_eq!(pack_row(&slots, 3).map(|(_, pf)| pf), Some(0));
        set_pack_frontier(&mut slots, 2, 2);
        assert!(!packed_prompt_done(&slots, 2, 1));
        set_pack_frontier(&mut slots, 3, 2);
        assert_eq!(pack_row(&slots, 3).map(|(_, pf)| pf), Some(2));
        assert!(packed_prompt_done(&slots, 2, 1) && !packed_prompt_done(&slots, 2, 0));
        assert!(pack_row(&slots, 1).is_none());
    }

    /// A full downstream stage keeps a request queued, answered with nothing.
    #[test]
    fn a_full_downstream_keeps_the_request_queued() {
        let metrics = Arc::new(Metrics::default());
        let mut slots: Vec<Option<Slot>> = std::iter::repeat_with(|| None).take(1).collect();
        let ((job, arrived), mut rx) = queued_job(10, Instant::now());
        let held = admit_into(
            &mut slots, 1, job, arrived, None, &metrics, &EngineHealth::Healthy, None, true,
        );
        assert!(held.is_some() && slots[0].is_none());
        assert!(rx.try_recv().is_err());
    }

    /// A raw-token consumer that falls behind is parked, not cut: its tokens are held in order,
    /// the slot leaves the decode feeds, and they are delivered with the terminal once it drains.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_slow_raw_token_consumer_is_parked_then_drained() {
        let (mut slot, mut rx) = prefill_test_slot();
        {
            let s = slot.as_mut().unwrap();
            s.raw_tokens = true;
            s.gen.max_tokens = 40;
        }
        let bundle = prefill_test_bundle("park");
        let mut n = 0;
        for t in 0..34u32 {
            assert!(!handle_produced_token(&mut slot, &None, &bundle, t, 1, &mut n, Some(&[99])));
        }
        let s = slot.as_ref().unwrap();
        assert!(s.parked_at.is_some() && s.held.len() == 2);
        assert!(gpu_decode_feeds(std::slice::from_ref(&slot), 1).is_empty());
        assert!(!handle_produced_token(&mut slot, &None, &bundle, 99, 1, &mut n, Some(&[99])));
        let mut got = Vec::new();
        while let Ok(StreamChunk::Token { id, .. }) = rx.try_recv() {
            got.push(id);
        }
        assert!(!flush_parked(&mut slot, &None) || slot.is_none());
        while let Ok(c) = rx.try_recv() {
            match c {
                StreamChunk::Token { id, .. } => got.push(id),
                StreamChunk::Done { reason, .. } => assert!(matches!(reason, FinishReason::Stop)),
                StreamChunk::Err(e) => panic!("{e}"),
            }
        }
        assert!(slot.is_none(), "finished after the drain");
        assert_eq!(got, (0..34).collect::<Vec<_>>());
    }

    /// A parked consumer that never drains is cut after the park timeout.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_parked_consumer_is_cut_after_the_timeout() {
        let (mut slot, _rx) = prefill_test_slot();
        slot.as_mut().unwrap().raw_tokens = true;
        let bundle = prefill_test_bundle("park-timeout");
        let mut n = 0;
        for t in 0..40u32 {
            handle_produced_token(&mut slot, &None, &bundle, t, 1, &mut n, Some(&[]));
        }
        assert!(!flush_parked(&mut slot, &None));
        slot.as_mut().unwrap().parked_at = Some(Instant::now() - PARK_TIMEOUT - PARK_TIMEOUT);
        assert!(flush_parked(&mut slot, &None));
        assert!(slot.is_none());
    }

    /// A text consumer that falls behind is parked too, not cut at the 33rd chunk: every delta
    /// arrives in order, then the terminal.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_slow_text_consumer_is_parked_then_drained() {
        let (mut slot, mut rx) = prefill_test_slot();
        slot.as_mut().unwrap().gen.max_tokens = 40;
        let bundle = prefill_test_bundle("park-text");
        let mut n = 0;
        for _ in 0..40 {
            assert!(!handle_produced_token(&mut slot, &None, &bundle, u32::from(b'a'), 1, &mut n, Some(&[])));
        }
        assert!(slot.as_ref().is_some_and(|s| s.parked_at.is_some() && s.held_finish.is_some()));
        assert!(gpu_decode_feeds(std::slice::from_ref(&slot), 1).is_empty());
        let (mut text, mut done) = (String::new(), false);
        for _ in 0..4 {
            while let Ok(c) = rx.try_recv() {
                match c {
                    StreamChunk::Token { text: t, .. } => text.push_str(&t),
                    StreamChunk::Done { reason, .. } => done = matches!(reason, FinishReason::Length),
                    StreamChunk::Err(e) => panic!("{e}"),
                }
            }
            flush_parked(&mut slot, &None);
        }
        assert!(done && slot.is_none());
        assert_eq!(text, "a".repeat(40));
    }

    /// Aging must always come first, or a request would be shed before it ever blocks the
    /// backfill and the fairness rule would be unreachable.
    #[test]
    fn the_aging_bound_is_always_inside_the_ttl() {
        for slo in [0.0, 1.0, 250.0, 5_000.0, 1e9] {
            assert!(
                queue_aging_ms(slo) < queue_ttl_ms(slo),
                "slo {slo}: aging {} ttl {}",
                queue_aging_ms(slo),
                queue_ttl_ms(slo)
            );
        }
        // A disconnect outranks a TTL expiry: there is nobody left to answer 429.
        assert_eq!(queue_verdict(true, 1e9, 250.0, JobClass::Normal), Queued::Disconnected);
        assert_eq!(queue_verdict(false, 1e9, 250.0, JobClass::Normal), Queued::Expired);
        assert_eq!(queue_verdict(false, 0.0, 250.0, JobClass::Normal), Queued::Retry);
        // A nonsensical SLO must not produce a zero or negative bound.
        assert!(queue_aging_ms(f64::NAN).is_finite() && queue_aging_ms(-1.0) > 0.0);
    }

    fn fault(fatal: bool) -> crate::DeviceErrorInfo {
        crate::DeviceErrorInfo {
            operation: "cuStreamSynchronize".into(),
            code: 719,
            name: "CUDA_ERROR_LAUNCH_FAILED".into(),
            fatal,
        }
    }

    #[test]
    fn health_degrades_on_nonfatal_and_recovers_on_success() {
        let h = advance_health(EngineHealth::Healthy, Some(fault(false)));
        assert!(matches!(
            h,
            EngineHealth::Degraded {
                consecutive_failures: 1
            }
        ));
        let h = advance_health(h, Some(fault(false)));
        assert!(matches!(
            h,
            EngineHealth::Degraded {
                consecutive_failures: 2
            }
        ));
        let h = advance_health(h, None);
        assert!(matches!(h, EngineHealth::Healthy));
    }

    #[test]
    fn health_dies_on_fatal_and_stays_dead() {
        let h = advance_health(EngineHealth::Healthy, Some(fault(true)));
        assert!(matches!(h, EngineHealth::Dead(_)));
        // Terminal: neither a clean tick nor a non-fatal fault revives it.
        let h = advance_health(h, None);
        assert!(matches!(h, EngineHealth::Dead(_)));
        let h = advance_health(h, Some(fault(false)));
        assert!(matches!(h, EngineHealth::Dead(_)));
    }

    #[test]
    fn health_ignores_clean_ticks() {
        assert!(matches!(
            advance_health(EngineHealth::Healthy, None),
            EngineHealth::Healthy
        ));
    }

    /// A batch failure fans out to every fed slot — the typed fault must
    /// survive the copy (a fatal fault maps to 503; a Msg would read as 500).
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    #[test]
    fn fanout_preserves_the_typed_fault() {
        let f = crate::RuntimeError::DeviceFault { info: fault(true) };
        assert!(fanout_err(&f, "ignored").is_fatal());
        let plain = crate::RuntimeError::Msg("boom".into());
        assert!(matches!(
            fanout_err(&plain, "boom"),
            crate::RuntimeError::Msg(m) if m == "boom"
        ));
    }

    fn paging_2layers(max_seqs: i64) -> KvPaging {
        KvPaging {
            block_tokens: 4,
            block_bytes: 64,
            kv_heads: 2,
            head_dim: 8,
            kv_factor: 2,
            max_seqs,
            head_slot_bytes: 64,
            per_layer: vec![
                KvLayerPaging {
                    layer_idx: 0,
                    buffer_name: "kv_cache_L0".into(),
                    initial_blocks: 4,
                },
                KvLayerPaging {
                    layer_idx: 1,
                    buffer_name: "kv_cache_L1".into(),
                    initial_blocks: 4,
                },
            ],
        }
    }

    /// Prefill ticks must never enter the decode-service EWMA: the rung
    /// controller floors its SLO at eight service ticks, so one 420 ms prefill
    /// tick moves that floor past a 250 ms target on its own.
    #[test]
    fn service_sample_excludes_prefill_ticks() {
        assert_eq!(service_sample(420.0, true), None);
        assert_eq!(service_sample(18.3, false), Some(18.3));
        assert_eq!(service_sample(0.0, false), None);

        let mut svc = crate::sched::admission::Ewma::new(0.2);
        for _ in 0..16 {
            svc.update(service_sample(420.0, true).unwrap_or(18.0));
        }
        assert!(svc.get() < 250.0, "decode EWMA stays under the SLO");
        let mut poisoned = crate::sched::admission::Ewma::new(0.2);
        for _ in 0..16 {
            poisoned.update(420.0);
        }
        assert!(poisoned.get() > 250.0, "control: unfiltered EWMA blows it");
    }

    #[test]
    fn kv_reservation_releases_generated_output_credit() {
        assert_eq!(reserved_kv_rows(70_000, 700, 0), 70_700);
        assert_eq!(reserved_kv_rows(70_000, 700, 512), 70_188);
        assert_eq!(reserved_kv_rows(4, 0, 0), 5);
        assert_eq!(reserved_kv_rows(4, 0, usize::MAX), 0);
    }

    #[test]
    fn decode_rung_attribution_uses_only_fed_slots() {
        let controller = RungController::new(DecodeRungs::new(&[1, 4], 4).unwrap());
        let live_slots = [0usize, 3];
        let feeds = [(live_slots[0], 7u32)];
        let extent = decode_feed_extent(&feeds).unwrap();

        assert_eq!(controller.width(controller.covering(extent)), 1);
        assert_eq!(decode_feed_extent(&[]), None);
    }

    #[test]
    fn decode_progress_uses_completed_steps_and_physical_extent() {
        let feeds = [(0, 7), (3, 9)];
        let progress = completed_decode(&feeds, 2).unwrap();
        assert_eq!(progress.extent, 4);
        assert_eq!(progress.steps.get(), 2);
        assert_eq!(completed_decode(&feeds, 0), None);
        assert_eq!(completed_decode(&[], 4), None);
    }

    #[cfg(feature = "cuda")]
    fn prefill_test_bundle(name: &str) -> ModelBundle {
        let dir =
            std::env::temp_dir().join(format!("plowrt-prefill-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("weights.json"),
            r#"{"network":"prefill-test","gpu":"cpu","num_gpus":1,"parallel":"none","weight_shared":false,"buckets":[]}"#,
        )
        .unwrap();
        let bundle = ModelBundle::load(&dir).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        bundle
    }

    fn prefill_test_slot() -> (Option<Slot>, crate::serve::stream::ChunkReceiver) {
        let (respond, rx) = crate::serve::stream::channel();
        (
            Some(Slot {
                telemetry: None,
                prompt_ids: vec![1, 2, 3],
                out_ids: Vec::new(),
                gen: GenParams::default(),
                respond,
                prefix_offset: 0,
                read_offset: 0,
                executed: 0,
                step: 0,
                pf_pos: 2,
                cached_tokens: 0,
                kv: None,
                arrived: Instant::now(),
                stop_tail: String::new(),
                stop_pending: String::new(),
                class: JobClass::Normal,
                raw_tokens: false,
                turn: None,
                turn_key: None,
                speech: None,
                mm: None,
                cfg: None,
                held: Vec::new(),
                held_finish: None,
                parked_at: None,
                session: None,
                resume: 0,
                lp: None,
                prefix: None,
            }),
            rx,
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tick_due_takes_turn_deadlines_and_keeps_classes_without_turns() {
        use crate::serve::cosched::{Due, Urgency};
        let model = crate::sched::cost::id("mux-test-tick-due");
        let waiting = std::collections::VecDeque::new();
        let (mut prompt, _rx) = prefill_test_slot();
        let (mut fin, _rx2) = prefill_test_slot();
        let now = Instant::now();
        let arrived = prompt.as_ref().unwrap().arrived;
        fin.as_mut().unwrap().class = JobClass::Final;
        // No turns: the class mapping, anchored at the oldest job of the winning class.
        let slots = vec![prompt.take(), None];
        assert_eq!(tick_due(&slots, &waiting, model, now).deadline, Due::from_urgency(Urgency::Deadline, arrived).deadline);
        let mut slots = slots;
        slots[1] = fin;
        assert_eq!(tick_due(&slots, &waiting, model, now).deadline, Due::from_urgency(Urgency::Final, slots[1].as_ref().unwrap().arrived).deadline);
        // A prompt whose turn ended speech 1.4 s ago is tighter than a fresh ASR final.
        let speech_end = now.checked_sub(std::time::Duration::from_millis(1400)).unwrap();
        slots[0].as_mut().unwrap().turn =
            Some(crate::serve::turns::TurnTimes { speech_end: Some(speech_end), budget: std::time::Duration::from_millis(1500), ..Default::default() });
        let d = tick_due(&slots, &waiting, model, now);
        assert!(d.deadline <= speech_end + std::time::Duration::from_millis(1500));
        assert!(d.slack(now) < Due::from_urgency(Urgency::Final, now).slack(now));
        // The same turn with time to spare yields to the final.
        slots[0].as_mut().unwrap().turn.as_mut().unwrap().speech_end = Some(now);
        let d = tick_due(&slots, &waiting, model, now);
        assert_eq!(d.deadline, Due::from_urgency(Urgency::Final, slots[1].as_ref().unwrap().arrived).deadline);
    }

    #[test]
    fn a_live_slot_sees_its_turns_first_audio() {
        use crate::serve::turns::{table, Kind, Stage};
        let now = Instant::now();
        let session: Arc<str> = "mux-test-refresh-turn".into();
        let key = table().join(&session, None, Kind::Llm, now, None, None, None).key;
        let (mut slot, _rx) = prefill_test_slot();
        let s = slot.as_mut().unwrap();
        s.turn = table().times(&key);
        s.turn_key = Some(key.clone());
        let mut slots = vec![slot];
        refresh_turns(&mut slots);
        assert!(slots[0].as_ref().unwrap().turn.unwrap().tts_first_audio.is_none());
        table().stamp(&key, Stage::TtsFirst, now);
        refresh_turns(&mut slots);
        assert_eq!(slots[0].as_ref().unwrap().turn.unwrap().tts_first_audio, Some(now));
    }

    /// A turn's speech stream is its first audio only until the tokens that audio renders from;
    /// after them its ticks are stream decode, behind the first render.
    #[test]
    fn a_speech_stream_is_first_audio_until_its_first_tokens() {
        use crate::serve::cosched::Band;
        let (model, now) = (crate::sched::cost::id("mux-test-first-tokens"), Instant::now());
        let t = crate::serve::turns::TurnTimes { speech_end: Some(now), ..Default::default() };
        let e = crate::serve::deadlines::Ests::default();
        let zero = std::time::Duration::ZERO;
        let band = |first: Option<usize>, step| turn_job_due(model, JobClass::Critical, first, step, zero, zero, now, &t, &e, now).band;
        assert_eq!(band(Some(20), 19), Band::First);
        assert_eq!(band(Some(20), 20), Band::Stream);
        assert_eq!(band(None, 0), Band::First, "an LLM prompt");
        assert_eq!(band(None, 1), Band::Stream, "LLM decode");
        let asr = turn_job_due(model, JobClass::Final, Some(CRITICAL_TOKENS), 3, zero, zero, now, &t, &e, now);
        assert_eq!(asr.band, Band::Final, "an ASR final, every step");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_argmax_requires_unmodified_greedy_logits() {
        let mut params = crate::text::sample::SamplingParams::default();
        assert!(!gpu_argmax_eligible(&params));
        params.temperature = 0.0;
        assert!(gpu_argmax_eligible(&params));
        params.repetition_penalty = 1.1;
        assert!(!gpu_argmax_eligible(&params));
        params.repetition_penalty = 1.0;
        params.logit_bias.push((1, 10.0));
        assert!(!gpu_argmax_eligible(&params));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_cold_prefill_yields_after_first_token_without_advancing_twice() {
        let bundle = prefill_test_bundle("ready");
        let (mut slot, mut rx) = prefill_test_slot();
        let feeds: Vec<(usize, u32)> = Vec::new();
        let mut tokens = 0;
        assert!(!gpu_prefill_should_yield(false, false, slot.as_ref()));
        assert!(gpu_prefill_should_yield(true, false, slot.as_ref()));

        slot.as_mut().unwrap().pf_pos = 3;
        assert!(!handle_produced_token(&mut slot, &None, &bundle, 65, 1, &mut tokens, Some(&[])));
        assert!(gpu_prefill_should_yield(
            !feeds.is_empty(),
            false,
            slot.as_ref()
        ));
        assert!(!gpu_prefill_should_yield(
            !feeds.is_empty(),
            true,
            slot.as_ref()
        ));
        assert!(feeds.is_empty(), "new decoder waits for the next tick");
        let ready = slot.as_ref().unwrap();
        assert_eq!((ready.step, ready.executed, tokens), (1, 1, 1));
        assert_eq!(ready.out_ids, [65]);
        assert!(matches!(
            rx.try_recv().unwrap(),
            StreamChunk::Token { id: 65, .. }
        ));
        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));

        rx.close();
        assert!(!gpu_prefill_should_yield(false, false, slot.as_ref()));
    }

    /// Bytes held back as a possible stop-string prefix are released when the request ends on
    /// its budget or on a stop token instead of being dropped.
    #[cfg(feature = "cuda")]
    #[test]
    fn held_stop_prefix_is_released_at_finish() {
        let bundle = prefill_test_bundle("stop-held");
        // "STOP" never completes: "ST" is held, then the budget (2) or the stop token (0) ends it.
        for (ids, max_tokens) in [(&[b'S' as u32, b'T' as u32][..], 2), (&[b'S' as u32, b'T' as u32, 0][..], 8)] {
            let (mut slot, mut rx) = prefill_test_slot();
            let s = slot.as_mut().unwrap();
            s.gen.max_tokens = max_tokens;
            s.gen.stop = vec!["STOP".into()];
            let mut tokens = 0;
            for &id in ids {
                handle_produced_token(&mut slot, &None, &bundle, id, 1, &mut tokens, Some(&[0]));
            }
            assert!(slot.is_none(), "the request finished");
            let mut text = String::new();
            let mut done = false;
            while let Ok(chunk) = rx.try_recv() {
                match chunk {
                    StreamChunk::Token { text: t, .. } => text.push_str(&t),
                    StreamChunk::Done { .. } => done = true,
                    StreamChunk::Err(e) => panic!("{e}"),
                }
            }
            assert!(done);
            assert_eq!(text, "ST", "max_tokens {max_tokens}");
        }
    }

    /// A partial UTF-8 sequence held for its continuation is emitted when the stream ends on its
    /// budget or a stop token, once, and the stop token's own text stays out.
    #[cfg(feature = "cuda")]
    #[test]
    fn a_held_partial_utf8_tail_is_flushed_at_finish() {
        let bundle = prefill_test_bundle("fffd-held");
        for (ids, max_tokens) in [(&[u32::from(b'a'), 0xC3][..], 2), (&[u32::from(b'a'), 0xC3, u32::from(b'Z')][..], 8)] {
            let (mut slot, mut rx) = prefill_test_slot();
            slot.as_mut().unwrap().gen.max_tokens = max_tokens;
            let mut tokens = 0;
            for &id in ids {
                handle_produced_token(&mut slot, &None, &bundle, id, 1, &mut tokens, Some(&[u32::from(b'Z')]));
            }
            assert!(slot.is_none(), "the request finished");
            let (mut text, mut done) = (String::new(), false);
            while let Ok(chunk) = rx.try_recv() {
                match chunk {
                    StreamChunk::Token { text: t, .. } => text.push_str(&t),
                    StreamChunk::Done { .. } => done = true,
                    StreamChunk::Err(e) => panic!("{e}"),
                }
            }
            assert!(done);
            assert_eq!(text, "a\u{FFFD}", "max_tokens {max_tokens}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_cold_prefill_continues_after_first_token_finishes_or_cancels() {
        let bundle = prefill_test_bundle("finished");
        for (stop_ids, max_tokens, cancel) in [
            (&[65][..], 4, false),
            (&[][..], 1, false),
            (&[][..], 4, true),
        ] {
            let (mut slot, mut rx) = prefill_test_slot();
            slot.as_mut().unwrap().gen.max_tokens = max_tokens;
            if cancel {
                rx.close();
            }
            let mut tokens = 0;
            let disconnected = handle_produced_token(
                &mut slot,
                &None,
                &bundle,
                65,
                1,
                &mut tokens,
                Some(stop_ids),
            );
            assert_eq!(disconnected, cancel);
            assert!(slot.is_none());
            assert!(!gpu_prefill_should_yield(false, false, slot.as_ref()));
            assert!(!gpu_prefill_should_yield(false, true, slot.as_ref()));
            assert!(gpu_prefill_should_yield(true, false, slot.as_ref()));
        }
    }

    #[cfg(any(feature = "hsa", feature = "cpu"))]
    #[test]
    fn amd_mixed_admission_obeys_rotation_age_and_total_prefill_budget() {
        let now = Instant::now();
        let candidates = vec![
            (0, now, 512),
            (2, now - std::time::Duration::from_secs(1), 512),
            (3, now, 0),
        ];
        assert_eq!(
            amd_mixed_prefill_pack(candidates.clone(), 700, true, 1, 4),
            [(2, 512), (0, 188)]
        );
        assert_eq!(
            amd_mixed_prefill_pack(candidates.clone(), 127, false, 0, 4),
            [(2, 127)]
        );
        assert!(amd_mixed_prefill_pack(candidates, 0, true, 0, 4).is_empty());
    }

    #[cfg(any(feature = "hsa", feature = "cpu"))]
    #[test]
    fn amd_token_batch_pack_takes_whole_chunks_in_pool_order_and_never_cuts() {
        let now = Instant::now();
        let pool = vec![(4, now, 700), (1, now, 300), (7, now, 2048), (2, now, 0), (9, now, 900)];
        // 700 + 300 fit; 2048 does not and is skipped, not sliced; 900 fits what is left.
        assert_eq!(amd_token_batch_pack(&pool, 2028), [(4, 700), (1, 300), (9, 900)]);
        // Exactly full.
        assert_eq!(amd_token_batch_pack(&pool, 1000), [(4, 700), (1, 300)]);
        let suffixes = vec![(0, now, 512), (1, now, 512)];
        assert_eq!(amd_token_batch_pack(&suffixes, 1024), [(0, 512), (1, 512)]);
        // Nothing fits: no member is cut to fit.
        assert!(amd_token_batch_pack(&pool, 299).is_empty());
        assert!(amd_token_batch_pack(&[], 4096).is_empty());
    }

    #[cfg(any(feature = "hsa", feature = "cpu"))]
    #[test]
    fn amd_prefill_scheduler_controls_are_bounded() {
        assert_eq!(amd_prefill_tick_cap(false, false, 2048), u32::MAX);
        assert_eq!(amd_prefill_tick_cap(true, false, 0), u32::MAX);
        assert_eq!(amd_prefill_tick_cap(true, true, 2048), u32::MAX);
        assert_eq!(amd_prefill_tick_cap(true, false, 2048), 2048);
        assert!(amd_defer_decode(true, true));
        assert!(!amd_defer_decode(true, false));
        assert!(!amd_defer_decode(false, true));

        let t0 = Instant::now();
        let t1 = t0 + std::time::Duration::from_millis(1);
        let candidates = || [(0, t0), (1, t1), (2, t1)];
        assert_eq!(amd_prefill_pick(candidates(), false, 2, 3), Some(0));
        assert_eq!(amd_prefill_pick(candidates(), true, 1, 3), Some(1));
        assert_eq!(amd_prefill_pick(candidates(), true, 2, 3), Some(2));
        assert_eq!(amd_prefill_pick(candidates(), true, 3, 3), Some(0));
        assert_eq!(amd_prefill_pick([(3, t0), (2, t0)], false, 0, 4), Some(2));
    }

    #[cfg(feature = "hsa")]
    #[test]
    fn amd_prefill_pack_rotates_filters_and_respects_budget() {
        use packet::dev::{PrefillSpan, PREFILL_SPAN_RESET_STATE};

        let span = |slot, rows, program| PrefillSpan {
            row0: 99,
            n_rows: rows,
            slot,
            flags: if slot == 0 {
                PREFILL_SPAN_RESET_STATE
            } else {
                0
            },
            kv_row0: slot * 10,
            kv_len: slot * 10 + rows,
            state_slot: slot,
            program,
        };

        // Fair rotation starts at slot 2. Its program defines compatibility;
        // program 4 is excluded and the remaining rows are densely rebased.
        let pack = amd_prefill_pack(
            [span(0, 4, 3), span(1, 2, 4), span(2, 3, 3), span(3, 2, 3)],
            7,
            2,
            4,
            |_| Some(7),
            |_| u32::MAX,
        );
        assert_eq!(pack.iter().map(|s| s.slot).collect::<Vec<_>>(), [2, 3]);
        assert_eq!(pack.iter().map(|s| s.row0).collect::<Vec<_>>(), [0, 3]);
        assert!(pack.iter().all(|s| s.program == 3));

        // A candidate that does not fit is skipped rather than exceeding the
        // token budget, allowing a later compatible short span to fill it.
        let pack = amd_prefill_pack(
            [span(0, 5, 7), span(1, 2, 7), span(2, 1, 7)],
            3,
            0,
            3,
            |_| Some(3),
            |_| u32::MAX,
        );
        assert_eq!(pack.iter().map(|s| s.slot).collect::<Vec<_>>(), [1, 2]);
        assert_eq!(pack.iter().map(|s| s.row0).collect::<Vec<_>>(), [0, 2]);
        assert_eq!(pack.iter().map(|s| s.n_rows).sum::<u32>(), 3);

        let pack = amd_prefill_pack(
            [span(0, 3, 7), span(1, 3, 7), span(2, 1, 7)],
            8,
            0,
            3,
            |_| Some(4),
            |_| u32::MAX,
        );
        assert_eq!(pack.iter().map(|s| s.slot).collect::<Vec<_>>(), [0, 2]);
        assert_eq!(pack.iter().map(|s| s.n_rows).sum::<u32>(), 4);

        // §5.4's D-class limit reaches admission: the same candidates, capped to one span. It
        // is the FIRST span in rotation order that survives, so the request that loses its span
        // is simply not admitted this tick -- and `packed.len() >= 2` at the call site then
        // routes the tick to the isolated prefill path instead of staging a plan the engine
        // would refuse.
        let pack = amd_prefill_pack(
            [span(0, 3, 7), span(1, 3, 7), span(2, 1, 7)],
            8,
            0,
            3,
            |_| Some(4),
            |_| 1,
        );
        assert_eq!(pack.iter().map(|s| s.slot).collect::<Vec<_>>(), [0]);
        assert!(
            amd_prefill_pack([span(0, 3, 7), span(1, 3, 7)], 8, 0, 3, |_| Some(4), |_| 0,)
                .is_empty()
        );
    }

    #[cfg(feature = "hsa")]
    #[test]
    fn amd_prefill_pack_rejects_invalid_descriptors() {
        use packet::dev::PrefillSpan;

        let valid = PrefillSpan {
            row0: 0,
            n_rows: 2,
            slot: 1,
            flags: 0,
            kv_row0: 8,
            kv_len: 10,
            state_slot: 1,
            program: 5,
        };
        let mut zero = valid;
        zero.n_rows = 0;
        let mut wrong_state = valid;
        wrong_state.state_slot = 0;
        let mut wrong_len = valid;
        wrong_len.kv_len = 11;
        let mut past_capacity = valid;
        past_capacity.slot = 4;
        past_capacity.state_slot = 4;

        assert_eq!(
            amd_prefill_pack(
                [zero, wrong_state, wrong_len, past_capacity],
                8,
                0,
                4,
                |_| Some(8),
                |_| u32::MAX,
            ),
            []
        );
        assert_eq!(
            amd_prefill_pack([valid], 0, 0, 4, |_| Some(8), |_| u32::MAX),
            []
        );
        assert_eq!(
            amd_prefill_pack([valid], 8, 0, 4, |_| None, |_| u32::MAX),
            []
        );
    }

    #[cfg(feature = "hsa")]
    #[test]
    fn packed_prefill_frontiers_are_all_or_error_in_member_order() {
        let updates = amd_packed_frontier_updates([2, 0, 3], |slot| Some(slot + 10)).unwrap();
        assert_eq!(updates, [(2, 12), (0, 10), (3, 13)]);
        assert_eq!(
            amd_packed_frontier_updates([2, 0, 3], |slot| (slot != 0).then_some(slot + 10)),
            Err(0)
        );
    }

    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    #[test]
    fn arrival_key_orders_older_requests_first() {
        let now = Instant::now();
        let older = now - std::time::Duration::from_millis(50);
        let newer = now - std::time::Duration::from_millis(5);
        assert!(arrival_key(older, now) < arrival_key(newer, now));
        assert_eq!(arrival_key(now, now), u64::MAX);
    }

    /// The incremental (windowed) detokenizer must reconstruct exactly the
    /// full decode, including multibyte UTF-8 split across ids, streaming the
    /// held-back bytes once the sequence completes.
    #[test]
    fn incremental_delta_matches_full_decode() {
        use crate::text::tokenizer::{ByteTokenizer, Tokenize};
        let tok = ByteTokenizer;
        let text = "héllo wörld — ok\n";
        let ids: Vec<u32> = text.bytes().map(u32::from).collect();

        let (mut prefix, mut read) = (0usize, 0usize);
        let mut streamed = String::new();
        let mut fed: Vec<u32> = Vec::new();
        for &id in &ids {
            fed.push(id);
            streamed.push_str(&incremental_delta(&tok, &fed, &mut prefix, &mut read, false, false));
        }
        assert_eq!(streamed, tok.decode(&ids));
        // The window stays bounded: prefix has advanced with the stream.
        assert!(
            prefix >= ids.len() - 4,
            "window failed to advance: {prefix}"
        );
    }

    #[test]
    fn kv_state_keeps_address_space_alive() {
        use crate::device::cpu::CpuBackend;
        use crate::device::Backend;
        use plow_asset::{MemoryMap, Segment};

        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(1));
        let weak_backend = Arc::downgrade(&backend);
        let map = MemoryMap {
            arena_bytes: 64,
            growable_base: 64,
            segments: vec![Segment {
                device: 0,
                global_base: 0,
                size: 64,
                growable_base: 64,
            }],
            entries: Vec::new(),
            kv_paging: None,
        };
        let addr_space = AddressSpace::allocate(Arc::clone(&backend), map).unwrap();
        let state = KvState {
            arena: KvArena::new(paging_2layers(4), &[0x1000, 0x2000]),
            _addr_space: Some(addr_space),
        };

        drop(backend);
        assert!(weak_backend.upgrade().is_some());
        drop(state);
        assert!(weak_backend.upgrade().is_none());
    }

    #[test]
    fn refresh_populates_all_entries_for_32_layers_batch_4() {
        let mut paging = paging_2layers(4);
        paging.per_layer = (0..32)
            .map(|layer_idx| KvLayerPaging {
                layer_idx,
                buffer_name: format!("kv_cache_L{layer_idx}"),
                initial_blocks: 4,
            })
            .collect();
        let bases: Vec<u64> = (0..32).map(|layer| 0x1000 + layer * 0x10000).collect();
        let arena = Some(Arc::new(Mutex::new(KvState {
            arena: KvArena::new(paging, &bases),
            _addr_space: None,
        })));
        let handles: Vec<SlotHandle> = {
            let mut state = arena.as_ref().unwrap().lock();
            (0..4)
                .map(|_| state.arena.allocate_slot(8).unwrap())
                .collect()
        };

        let kv_pages = ind_slots::kv_pages(32, 4);
        let mut obs = RunObserver::new(false, ind_slots::table_size(32, 4));
        obs.set_kv_pages_range(kv_pages.clone());
        refresh_indirection(&mut obs, handles, &arena, kv_pages.clone());

        let populated: Vec<u64> = kv_pages.map(|slot| obs.indirection.get(slot)).collect();
        assert_eq!(populated.len(), 128);
        assert!(populated.iter().all(|&addr| addr != 0));
        for row in 0..4 {
            for layer in 0..32 {
                assert_eq!(populated[row * 32 + layer], bases[layer] + row as u64 * 64);
            }
        }
    }

    #[test]
    fn refresh_wipes_stale_kv_entries() {
        // Prior tick left non-zero KV_PAGES; refresh with zero live rows must
        // wipe them.
        let arena_inner = KvArena::new(paging_2layers(4), &[0x1000, 0x2000]);
        let arena = Some(Arc::new(Mutex::new(KvState {
            arena: arena_inner,
            _addr_space: None,
        })));
        let kv_pages = ind_slots::kv_pages(2, 4);
        let mut obs = RunObserver::new(false, ind_slots::table_size(2, 4));
        obs.set_kv_pages_range(kv_pages.clone());
        for i in kv_pages.clone() {
            obs.indirection.set(i, 0xDEADBEEF);
        }
        refresh_indirection(&mut obs, std::iter::empty(), &arena, kv_pages.clone());
        for i in kv_pages {
            assert_eq!(obs.indirection.get(i), 0, "slot {i} not wiped");
        }
    }

    #[test]
    fn refresh_without_arena_is_a_wipe() {
        let kv_pages = ind_slots::kv_pages(2, 4);
        let mut obs = RunObserver::new(false, ind_slots::table_size(2, 4));
        obs.set_kv_pages_range(kv_pages.clone());
        for i in kv_pages.clone() {
            obs.indirection.set(i, 0xDEADBEEF);
        }
        refresh_indirection(
            &mut obs,
            std::iter::empty::<SlotHandle>(),
            &None,
            kv_pages.clone(),
        );
        for i in kv_pages {
            assert_eq!(obs.indirection.get(i), 0);
        }
    }

    /// FLASH → indirection → observer trace (§4b1). Populate `KV_PAGES` by
    /// hand, run a tiny program with a `Body::Flash` gated on a `Body::Host`
    /// producer, and assert `obs.kv_writes` captured the exact snapshot the
    /// interpreter saw at fire time. Guards the compiler → runtime seam:
    /// whatever `refresh_indirection` writes reaches the FLASH consumer.
    #[test]
    fn flash_records_kv_pages_snapshot() {
        use crate::device::cpu::CpuBackend;
        use crate::device::Backend;
        use crate::exec::ExecutorSet;
        use packet::{Body, Counter, Inst, Program, ResourceKind};

        // A tiny program: one Host op producing counter 0, then a Flash op
        // gated on it. The Flash body values are arbitrary — the interpreter
        // fires it and the observer records the indirection snapshot.
        let program = Program {
            insts: vec![
                Inst {
                    resource: ResourceKind::Sm,
                    unit: 0,
                    index: 0,
                    body: Body::Host,
                    wait: vec![],
                    succ: vec![0],
                },
                Inst {
                    resource: ResourceKind::Sm,
                    unit: 0,
                    index: 1,
                    body: Body::Flash {
                        coord: [0, 0],
                        seq_q: 1,
                        seq_kv: 1,
                        head_dim: 8,
                        bq: 1,
                        bkv: 1,
                        heads: 1,
                        kv_heads: 1,
                        window: 0,
                        out: 0,
                        tmem: 0,
                        variant: packet::Opcode::VARIANT_FLASH_CAUSAL_BF16,
                    },
                    wait: vec![0],
                    succ: vec![],
                },
            ],
            counters: vec![Counter {
                id: 0,
                threshold: 1,
                scope: 1,
                _pad: [0; 3],
            }],
            bucket_id: 0,
            plan_gen: 0,
            flags: 0,
        };

        // Populate KV_PAGES with a distinctive pattern before firing so the
        // captured snapshot has content the assertion can pin.
        let kv_pages = ind_slots::kv_pages(2, 4);
        let mut obs = RunObserver::new(false, ind_slots::table_size(2, 4));
        obs.set_kv_pages_range(kv_pages.clone());
        let expected: Vec<u64> = kv_pages
            .clone()
            .enumerate()
            .map(|(i, slot)| {
                let addr = 0x1000 + i as u64 * 64;
                obs.indirection.set(slot, addr);
                addr
            })
            .collect();

        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(1));
        let execset = ExecutorSet::bringup(backend).unwrap();
        let pool = execset.counter_pool(&program);
        let mut streams = crate::device::cpu::StreamSet::new(&program, pool.len());
        execset.run_reference_traced_reuse(&program, &pool, &mut obs, &mut streams);

        // Exactly one FLASH fired.
        assert_eq!(obs.kv_writes.len(), 1);
        let write = &obs.kv_writes[0];
        assert_eq!(write.packet_index, 1);
        assert_eq!(write.addresses, expected);
    }

    #[test]
    fn no_flash_leaves_kv_writes_empty() {
        // Programs with no attention (only Host / Token) must produce zero
        // kv_writes entries — the observer only records on FLASH fire.
        use crate::device::cpu::CpuBackend;
        use crate::device::Backend;
        use crate::exec::ExecutorSet;
        use packet::{Body, Counter, Inst, Program, ResourceKind};

        let program = Program {
            insts: vec![
                Inst {
                    resource: ResourceKind::Sm,
                    unit: 0,
                    index: 0,
                    body: Body::Host,
                    wait: vec![],
                    succ: vec![0],
                },
                Inst {
                    resource: ResourceKind::Sm,
                    unit: 0,
                    index: 1,
                    body: Body::Host,
                    wait: vec![0],
                    succ: vec![],
                },
            ],
            counters: vec![Counter {
                id: 0,
                threshold: 1,
                scope: 1,
                _pad: [0; 3],
            }],
            bucket_id: 0,
            plan_gen: 0,
            flags: 0,
        };

        let kv_pages = ind_slots::kv_pages(2, 4);
        let mut obs = RunObserver::new(false, ind_slots::table_size(2, 4));
        obs.set_kv_pages_range(kv_pages.clone());
        for slot in kv_pages {
            obs.indirection.set(slot, 0xDEADBEEF);
        }
        let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(1));
        let execset = ExecutorSet::bringup(backend).unwrap();
        let pool = execset.counter_pool(&program);
        let mut streams = crate::device::cpu::StreamSet::new(&program, pool.len());
        execset.run_reference_traced_reuse(&program, &pool, &mut obs, &mut streams);

        assert!(obs.kv_writes.is_empty());
    }
}

/// Host-path microbenchmarks, CPU only. Inputs by path: `HOSTBENCH_TOKENIZER` (a
/// `tokenizer.json`), `HOSTBENCH_PROMPTS` (a directory of `prompts-<L>.json`, JSON string arrays
/// of `vllm bench serve --dataset-name random` prompts) and `HOSTBENCH_TEXTS` (a JSON string array
/// of generated outputs to replay through the detokenizer). Encode splitting follows the serve
/// env (`PLOW_ENCODE_THREADS`, `PLOW_ENCODE_SPLIT_MIN`).
#[cfg(all(test, feature = "hf-tokenizer"))]
mod host_bench {
    use super::*;
    use crate::serve::openai::{CompletionChoice, CompletionRequest, CompletionResponse};
    use crate::text::tokenizer::{HfTokenizer, Tokenize};
    use std::time::Duration;

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    fn p90(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() * 9 / 10]
    }

    fn spin(d: Duration) -> Duration {
        let t = Instant::now();
        while t.elapsed() < d {
            std::hint::spin_loop();
        }
        t.elapsed()
    }

    fn strings(path: &std::path::Path) -> Vec<String> {
        serde_json::from_slice(&std::fs::read(path).expect("read input")).expect("string array")
    }

    #[test]
    #[ignore]
    fn host_path_microbench() {
        let var = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("set {k}"));
        let tok = HfTokenizer::from_file(std::path::Path::new(&var("HOSTBENCH_TOKENIZER")))
            .expect("tokenizer");
        let prompts_dir = std::path::PathBuf::from(var("HOSTBENCH_PROMPTS"));

        println!("HOSTBENCH request path (median of 8 prompts x 5 runs)");
        for len in [128usize, 1024, 4096, 8192, 15000] {
            let prompts = strings(&prompts_dir.join(format!("prompts-{len}.json")));
            let (mut enc, mut parse, mut ids_n) = (Vec::new(), Vec::new(), 0);
            for p in prompts.iter().take(8) {
                let body = serde_json::to_vec(&serde_json::json!({
                    "model": "m", "prompt": p, "max_tokens": 128, "stream": true,
                    "ignore_eos": true, "temperature": 0.0,
                    "stream_options": {"include_usage": true},
                }))
                .unwrap();
                for _ in 0..5 {
                    let t = Instant::now();
                    let req: CompletionRequest = serde_json::from_slice(&body).unwrap();
                    parse.push(t.elapsed().as_secs_f64() * 1e3);
                    drop(req);
                    let t = Instant::now();
                    ids_n = tok.encode_with_special_tokens(p, true).len();
                    enc.push(t.elapsed().as_secs_f64() * 1e3);
                }
            }
            println!(
                "  L={len:>5} ids={ids_n:>6} json_parse_ms={:.3} encode_ms={:.3} (p90 {:.3})",
                median(parse),
                median(enc.clone()),
                p90(enc)
            );
            // The chat path's extra step: the checkpoint template over one user turn.
            if let Some(t) = std::env::var("HOSTBENCH_ASSETS")
                .ok()
                .and_then(|d| crate::serve::template::ChatTemplate::load(std::path::Path::new(&d)))
            {
                let msgs = vec![serde_json::json!({"role": "user", "content": prompts[0]})];
                let render: Vec<f64> = (0..20)
                    .map(|_| {
                        let t0 = Instant::now();
                        std::hint::black_box(t.render(&msgs).unwrap());
                        t0.elapsed().as_secs_f64() * 1e3
                    })
                    .collect();
                println!("  L={len:>5} chat_template_render_ms={:.3}", median(render));
            }
        }

        // Per-token detokenize over real generations, and the SSE frame the handler builds.
        let texts = strings(std::path::Path::new(&var("HOSTBENCH_TEXTS")));
        let (mut detok_ns, mut n_tok) = (0u128, 0usize);
        for text in &texts {
            let ids: Vec<u32> = tok.encode(text).into_iter().take(128).collect();
            let (mut prefix, mut read) = (0usize, 0usize);
            let mut fed = Vec::with_capacity(ids.len());
            let t = Instant::now();
            for &id in &ids {
                fed.push(id);
                std::hint::black_box(incremental_delta(&tok, &fed, &mut prefix, &mut read, false, false));
            }
            detok_ns += t.elapsed().as_nanos();
            n_tok += ids.len();
        }
        let (model, id) = ("gemma-4-26b-a4b-it".to_string(), "cmpl-0123456789abcdef".to_string());
        let frames = 20_000;
        let t = Instant::now();
        for i in 0..frames {
            let frame = CompletionResponse {
                id: id.clone(),
                object: "text_completion",
                created: 1_789_920_673,
                model: model.clone(),
                choices: vec![CompletionChoice {
                    index: 0,
                    text: if i % 2 == 0 { " the".into() } else { ".".into() },
                    logprobs: None,
                    finish_reason: None,
                    x_plow_finish_reason: None,
                }],
                usage: None,
                token_ids: None,
            };
            let _ = std::hint::black_box(
                axum::response::sse::Event::default()
                    .data(crate::serve::stream::chunk_data(&frame)),
            );
        }
        let serde_event_us = t.elapsed().as_secs_f64() * 1e6 / frames as f64;
        // The served path: the stream's fixed head serialized once, one choice per frame.
        let head = crate::serve::stream::FrameHead::new(&id, "text_completion", 1_789_920_673, &model);
        let t = Instant::now();
        for i in 0..frames {
            let choice = CompletionChoice {
                index: 0,
                text: if i % 2 == 0 { " the".into() } else { ".".into() },
                logprobs: None,
                finish_reason: None,
                x_plow_finish_reason: None,
            };
            let _ = std::hint::black_box(head.frame(&choice));
        }
        println!(
            "HOSTBENCH per token: detok_us={:.2} ({n_tok} tokens) sse_frame_us={:.2} (serde+Event {:.2})",
            detok_ns as f64 / 1e3 / n_tok.max(1) as f64,
            t.elapsed().as_secs_f64() * 1e6 / frames as f64,
            serde_event_us,
        );

        // Per-tick dispatcher <-> engine handoff around a 2 ms tick body, the mux's own shape.
        let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
        let tick = Duration::from_millis(2);
        let handoff: Vec<f64> = rt.block_on(async {
            let eng = crate::exec::engine_thread::EngineThread::spawn("hostbench-eng".into());
            let mut v = Vec::new();
            for _ in 0..300 {
                let t = Instant::now();
                let body = eng.run(move || spin(tick)).await.unwrap();
                v.push((t.elapsed() - body).as_secs_f64() * 1e6);
            }
            v
        });
        let inline: Vec<f64> = (0..300)
            .map(|_| {
                let t = Instant::now();
                let body = crate::exec::engine_thread::run_inline(|| spin(tick)).unwrap();
                (t.elapsed() - body).as_secs_f64() * 1e6
            })
            .collect();
        println!(
            "HOSTBENCH per tick: engine_thread_handoff_us={:.1} (p90 {:.1}) inline_us={:.2}",
            median(handoff.clone()),
            p90(handoff),
            median(inline)
        );

        // Engine-thread emit: one try_send per live stream to a parked handler task.
        for streams in [1usize, 16] {
            let (lat_tx, mut lat_rx) = tokio::sync::mpsc::unbounded_channel::<f64>();
            let mut txs = Vec::new();
            for _ in 0..streams {
                let (tx, mut rx) = tokio::sync::mpsc::channel::<(Instant, String)>(33);
                let lat_tx = lat_tx.clone();
                rt.spawn(async move {
                    while let Some((sent, text)) = rx.recv().await {
                        std::hint::black_box(text);
                        let _ = lat_tx.send(sent.elapsed().as_secs_f64() * 1e6);
                    }
                });
                txs.push(tx);
            }
            drop(lat_tx);
            let send_us: Vec<f64> = std::thread::spawn(move || {
                let mut v = Vec::new();
                for _ in 0..200 {
                    spin(tick);
                    let t = Instant::now();
                    for tx in &txs {
                        tx.try_send((Instant::now(), " the".to_string())).unwrap();
                    }
                    v.push(t.elapsed().as_secs_f64() * 1e6 / txs.len() as f64);
                }
                v
            })
            .join()
            .unwrap();
            let wake: Vec<f64> = rt.block_on(async {
                let mut v = Vec::new();
                while let Some(x) = lat_rx.recv().await {
                    v.push(x);
                }
                v
            });
            println!(
                "HOSTBENCH emit streams={streams}: try_send_us/token={:.2} (p90 {:.2}) handler_wake_us={:.1} (p90 {:.1})",
                median(send_us.clone()),
                p90(send_us),
                median(wake.clone()),
                p90(wake)
            );
        }
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use crate::serve::session::{RetainTable, SessionTicket};
    use std::time::Duration;

    fn retention() -> Retention {
        // One launch width covering the whole table: seating never widens it.
        Retention::with_table(RetainTable::new(Duration::from_secs(60), 0), &[16], 0)
    }

    fn job(prompt: &[u32], session: Option<&str>) -> ((Job, Instant), crate::serve::stream::ChunkReceiver) {
        let (respond, rx) = crate::serve::stream::channel();
        let t = Instant::now();
        let job = Job {
            prompt_ids: prompt.to_vec(),
            gen: GenParams { max_tokens: 4, ..GenParams::default() },
            arrived: t,
            respond,
            opts: JobOpts {
                session: session.map(|s| {
                    Box::new(SessionTicket {
                        session: s.into(),
                        request: "r".into(),
                        keys: crate::serve::session::row_keys(prompt, &[], &[]),
                        report: None,
                    })
                }),
                ..Default::default()
            },
        };
        ((job, t), rx)
    }

    fn admit(
        slots: &mut [Option<Slot>],
        r: &mut Retention,
        (job, t): (Job, Instant),
        budget: Option<crate::sched::admission::KvBudget>,
    ) -> Option<(Job, Instant)> {
        let metrics = Arc::new(Metrics::default());
        admit_session(slots, slots.len(), job, t, None, &metrics, &EngineHealth::Healthy, budget, false, r)
    }

    /// Run slot `i` to one produced token and free it, as a finished request would.
    fn finish(slots: &mut [Option<Slot>], r: &mut Retention, i: usize) {
        let mut slot = slots[i].take().expect("live");
        if let Some(seat) = slot.session.as_mut() {
            seat.on_token(99);
        }
        drop(slot);
        r.collect(Instant::now());
    }

    #[test]
    fn a_session_resumes_its_retained_slot_and_other_requests_keep_off_it() {
        let mut slots: Vec<Option<Slot>> = std::iter::repeat_with(|| None).take(4).collect();
        let mut r = retention();
        let (a, _ra) = job(&[1, 2, 3, 4, 5, 6], Some("s"));
        assert!(admit(&mut slots, &mut r, a, None).is_none());
        assert_eq!(slots[0].as_ref().unwrap().resume, 0);
        finish(&mut slots, &mut r, 0);
        assert!(r.table.holds(0), "the finished session request is retained");

        let (plain, _rp) = job(&[1, 2, 3], None);
        assert!(admit(&mut slots, &mut r, plain, None).is_none());
        assert!(slots[0].is_none() && slots[1].is_some(), "a live request takes a free slot first");
        let (other, _ro) = job(&[1, 2, 3, 4, 5, 6, 7], Some("t"));
        assert!(admit(&mut slots, &mut r, other, None).is_none());
        assert_eq!(slots[2].as_ref().unwrap().resume, 0, "sessions never share rows");

        let (next, _rn) = job(&[1, 2, 3, 4, 9, 9, 9], Some("s"));
        assert!(admit(&mut slots, &mut r, next, None).is_none());
        assert_eq!(slots[0].as_ref().unwrap().resume, 4, "the shared prefix resumes");
        assert!(!r.table.holds(0));
    }

    #[test]
    fn retained_slots_yield_to_live_requests_and_the_evicted_session_recomputes() {
        let mut slots: Vec<Option<Slot>> = std::iter::repeat_with(|| None).take(2).collect();
        let mut r = retention();
        for (i, s) in ["a", "b"].into_iter().enumerate() {
            let (j, _rx) = job(&[1, 2, 3], Some(s));
            assert!(admit(&mut slots, &mut r, j, None).is_none());
            finish(&mut slots, &mut r, i);
        }
        assert_eq!(r.table.len(), 2);
        let (plain, _rp) = job(&[5, 6], None);
        assert!(admit(&mut slots, &mut r, plain, None).is_none(), "retained KV never queues a live request");
        assert!(slots[0].is_some() && !r.table.holds(0), "the least recently used session went");
        let (again, _rg) = job(&[1, 2, 3], Some("a"));
        assert!(admit(&mut slots, &mut r, again, None).is_none());
        assert_eq!(slots[1].as_ref().unwrap().resume, 0, "evicted: a plain recompute");
        assert!(r.table.is_empty());
    }

    #[test]
    fn retained_slots_never_push_a_live_request_past_the_slack() {
        let mut slots: Vec<Option<Slot>> = std::iter::repeat_with(|| None).take(16).collect();
        let mut r = Retention::with_table(RetainTable::new(Duration::from_secs(60), 0), &[1, 2, 4, 8, 16], 0);
        // Ten sessions live at once (the launch is 16 wide), then all idle.
        for i in 0..10 {
            let (j, _rx) = job(&[1, 2, 3], Some(&format!("s{i}")));
            assert!(admit(&mut slots, &mut r, j, None).is_none());
        }
        for i in 0..10 {
            finish(&mut slots, &mut r, i);
        }
        assert_eq!(r.table.len(), 10);
        // Slots 0..10 are retained; the next free one (10) would widen the idle launch (1) to 16:
        // the least recently used retained slot (0) goes and the request decodes at width 1.
        let (plain, _rp) = job(&[5, 6], None);
        assert!(admit(&mut slots, &mut r, plain, None).is_none());
        assert!(slots[0].is_some() && r.table.len() == 9);
    }

    #[test]
    fn retained_rows_count_against_the_kv_budget_and_are_evicted_for_it() {
        let mut slots: Vec<Option<Slot>> = std::iter::repeat_with(|| None).take(4).collect();
        let mut r = retention();
        let (a, _ra) = job(&[1; 20], Some("a"));
        assert!(admit(&mut slots, &mut r, a, None).is_none());
        finish(&mut slots, &mut r, 0);
        assert_eq!(r.table.rows(), 20);
        // 20 retained + 24 wanted > 40: the retained rows go, the request is seated.
        let budget = crate::sched::admission::KvBudget::linear(1, 40);
        let (b, _rb) = job(&[2; 20], None);
        assert!(admit(&mut slots, &mut r, b, Some(budget)).is_none());
        assert!(r.table.is_empty());
    }

    #[test]
    fn without_a_ticket_or_with_retention_off_nothing_is_retained() {
        let mut slots: Vec<Option<Slot>> = std::iter::repeat_with(|| None).take(2).collect();
        let mut r = retention();
        let (a, _ra) = job(&[1, 2, 3], None);
        assert!(admit(&mut slots, &mut r, a, None).is_none());
        finish(&mut slots, &mut r, 0);
        let mut off = Retention::off();
        let (b, _rb) = job(&[1, 2, 3], Some("s"));
        assert!(admit(&mut slots, &mut off, b, None).is_none());
        assert!(slots[0].as_ref().unwrap().session.is_none());
        finish(&mut slots, &mut off, 0);
        assert!(r.table.is_empty() && off.table.is_empty());
    }
}
