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
use crate::obs::Metrics;
use crate::sched::admission::{admit, Admit, LoadEstimator};
use crate::sched::batching::{formation_window_ms, select_bucket};
use crate::sched::multistep::MultiStep;
use crate::sched::rungs::{DecodeRungs, RungController, RungLoad};
use crate::serve::stream::{ChunkSender, FinishReason, StreamChunk};
use crate::serve::{
    bucket_has_sample_batch, reference_logits_row, sample_vocab, seeded_unit, AppState, GenParams,
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

/// Admission sheds only when the predicted wait exceeds this many ticks of the
/// engine's observed service time (or the configured `slo_ms` floor, whichever
/// is larger). 8 was chosen to clear the worst spurious shed observed — a B=32
/// Gemma-4-12B blob at `predicted_wait_ms=329` against `service_ms≈58` (5.7
/// ticks) — with margin, while still catching a genuinely wedged queue.
const SLO_SERVICE_TICKS: f64 = 8.0;

/// Dispatcher config. Cold-path — set once at startup from CLI flags.
#[derive(Clone, Copy, Debug)]
pub struct MuxConfig {
    /// Upper bound on the arrival-rate-driven batch-formation hold (ms) used
    /// only in the cold-start path (empty slot table) — with slots always
    /// draining, the hot path never sleeps.
    pub max_hold_ms: f64,
    /// SLO used by admission (predicted wait above this sheds the request).
    /// Treated as a FLOOR: the effective SLO is
    /// `max(slo_ms, SLO_SERVICE_TICKS * service_ms)` so it scales with the
    /// decode batch instead of shedding every request on a wide blob.
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
}

impl Default for MuxConfig {
    fn default() -> Self {
        MuxConfig {
            max_hold_ms: 8.0,
            slo_ms: 250.0,
            multi_step: true,
            queue_depth: 0,
            max_queued_requests: 0,
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
}

/// Internal messages to the dispatcher: jobs or control signals.
enum MuxMsg {
    Job(Job),
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
#[cfg(any(feature = "cuda", feature = "hsa"))]
fn note_fault(tick_fault: &mut Option<crate::DeviceErrorInfo>, err: &crate::RuntimeError) {
    if tick_fault.is_none() {
        *tick_fault = err.device_fault().cloned();
    }
}

/// Per-slot copy of a batch error for fan-out to every affected waiter: a
/// typed device fault stays typed (its fatality drives the 503 mapping);
/// anything else degrades to the stringified `Msg` as before.
#[cfg(any(feature = "cuda", feature = "hsa"))]
fn fanout_err(err: &crate::RuntimeError, msg: &str) -> crate::RuntimeError {
    match err.device_fault() {
        Some(info) => crate::RuntimeError::DeviceFault { info: info.clone() },
        None => crate::RuntimeError::Msg(msg.to_string()),
    }
}

impl ModelMux {
    /// Submit a job. Returns immediately; the caller awaits the stream.
    pub fn submit(&self, job: Job) -> std::result::Result<(), SubmitError> {
        Metrics::inc(&self.metrics.queued_requests);
        match self.tx.try_send(MuxMsg::Job(job)) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(MuxMsg::Job(job))) => {
                self.metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
                Err(SubmitError::Full(job))
            }
            Err(mpsc::error::TrySendError::Closed(MuxMsg::Job(job))) => {
                self.metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
                Err(SubmitError::Closed(job))
            }
            Err(_) => unreachable!(),
        }
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
        self.preempt.store(true, Ordering::Release);
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
    let metrics = Arc::clone(&state.metrics);
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
    #[cfg(any(feature = "cuda", feature = "hsa"))]
    let gpu_shape = state.gpu_engine(bundle.network()).map(|e| {
        let e = e.lock();
        (e.batch(), e.decode_rungs())
    });
    #[cfg(any(feature = "cuda", feature = "hsa"))]
    let capacity = gpu_shape.as_ref().map(|x| x.0).unwrap_or(capacity);
    #[cfg(any(feature = "cuda", feature = "hsa"))]
    let rung_widths = gpu_shape.map(|x| x.1);
    #[cfg(not(any(feature = "cuda", feature = "hsa")))]
    let rung_widths: Option<Box<[u32]>> = None;
    let mut rung_controller =
        rung_widths
            .as_deref()
            .and_then(|widths| match DecodeRungs::new(widths, capacity) {
                Ok(rungs) if rungs.len() > 1 => Some(RungController::new(rungs)),
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

    tokio::spawn(async move {
        let mut slots: Vec<Option<Slot>> = (0..capacity).map(|_| None).collect();
        let mut load = LoadEstimator::default();
        let mut last_arrival: Option<Instant> = None;
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
        #[cfg(any(feature = "cuda", feature = "hsa"))]
        let has_gpu = state.gpu_engine(bundle.network()).is_some();
        #[cfg(not(any(feature = "cuda", feature = "hsa")))]
        let has_gpu = false;
        // Dedicated engine/submission thread for GPU models: every tick runs
        // on ONE persistent OS thread (CUDA context bound once, no
        // blocking-pool dispatch). CPU-reference models keep spawn_blocking.
        let engine_thread = has_gpu
            .then(|| crate::exec::engine_thread::EngineThread::spawn(format!("plow-eng-{slug}")));

        loop {
            // Preempt ([`ModelMux::preempt`]): kill every live slot NOW.
            // Checked at every loop top because a full slot table never
            // reaches the message channel between ticks — this flag is the
            // one bounded-latency path in.
            if preempt_seen.swap(false, Ordering::AcqRel) {
                preempt_slots(&mut slots, &arena);
                draining = true;
            }
            let live = slots.iter().filter(|s| s.is_some()).count();

            // Drain completion: if draining and no in-flight slots remain,
            // signal the drain future and exit the dispatcher loop.
            if draining && live == 0 {
                if let Some(done) = drain_done.take() {
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
                        MuxMsg::Job(job) => {
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
            if live == 0 {
                let Some(msg) = rx.recv().await else { break };
                note_dequeued(&msg, &metrics);
                match msg {
                    MuxMsg::Job(job) => {
                        admit_into(
                            &mut slots,
                            job,
                            &mut load,
                            &mut last_arrival,
                            arena.as_ref(),
                            &metrics,
                            &health,
                        );
                    }
                    MuxMsg::Drain(done) => {
                        // No in-flight work; signal immediately.
                        let _ = done.send(());
                        break;
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
                let queued = metrics.queued_requests.load(Ordering::Relaxed) as usize;
                let (sum, n) = slots
                    .iter()
                    .flatten()
                    .fold((0usize, 0usize), |(s, n), slot| {
                        (s.saturating_add(slot.gen.max_tokens.max(1)), n + 1)
                    });
                let mean_output_tokens = if n == 0 { 1.0 } else { sum as f64 / n as f64 };
                let before = controller.admission_limit();
                let decision = controller.decide(RungLoad {
                    occupied_extent,
                    queued,
                    oldest_wait_ms: 0.0,
                    arrival_rps: load.lambda.get(),
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

            // Non-blocking drain: fill every idle slot the queue can serve.
            // A short hold when we still have empty slots and no live work
            // yet keeps us from waking up on a single arrival amid a burst.
            let idle = slots[..admission_limit]
                .iter()
                .filter(|s| s.is_none())
                .count();
            if !draining && idle > 0 {
                let lambda = load.lambda.get();
                let live_now = admission_limit - idle;
                // Only hold when the slot table is empty (cold-start burst);
                // if any slot is already live, spinning up the tick delivers
                // TTFT faster than waiting for more arrivals.
                let hold_ms = if live_now == 0 {
                    formation_window_ms(lambda, cfg.max_hold_ms)
                } else {
                    0.0
                };
                if hold_ms > 0.0 {
                    Metrics::add(&metrics.hold_ms_sum, hold_ms as u64);
                    Metrics::inc(&metrics.hold_count);
                    let deadline =
                        Instant::now() + std::time::Duration::from_secs_f64(hold_ms / 1000.0);
                    while slots[..admission_limit].iter().any(|s| s.is_none()) {
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        match tokio::time::timeout(remaining, rx.recv()).await {
                            Ok(Some(msg)) => {
                                note_dequeued(&msg, &metrics);
                                match msg {
                                    MuxMsg::Job(job) => admit_into(
                                        &mut slots[..admission_limit],
                                        job,
                                        &mut load,
                                        &mut last_arrival,
                                        arena.as_ref(),
                                        &metrics,
                                        &health,
                                    ),
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
                while !draining && slots[..admission_limit].iter().any(|s| s.is_none()) {
                    match rx.try_recv() {
                        Ok(msg) => {
                            note_dequeued(&msg, &metrics);
                            match msg {
                                MuxMsg::Job(job) => admit_into(
                                    &mut slots[..admission_limit],
                                    job,
                                    &mut load,
                                    &mut last_arrival,
                                    arena.as_ref(),
                                    &metrics,
                                    &health,
                                ),
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

            let live = slots.iter().filter(|s| s.is_some()).count();
            if live == 0 {
                continue;
            }
            let (service_capacity, tick_rung) = if let Some(controller) = rung_controller.as_ref() {
                let occupied_extent = slots
                    .iter()
                    .rposition(Option::is_some)
                    .map(|i| i + 1)
                    .unwrap_or(1);
                let rung = controller.covering(occupied_extent);
                let width = controller.width(rung);
                metrics
                    .decode_rung_actual
                    .store(width as u64, Ordering::Relaxed);
                (width, Some(rung))
            } else {
                (capacity, None)
            };

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

            // Admit gate — reads updated λ/μ. Shed drops every live slot.
            let util = load.utilization();
            metrics
                .util_milli
                .store((util * 1000.0) as u64, Ordering::Relaxed);
            // The GPU engine (and any SAMPLE_BATCH bucket) advances ALL live
            // slots in ONE batched launch, so `service_ms` is the per-token
            // service time for the WHOLE batch, not per slot. The wait a
            // joining request sees is the number of *serial* batches ahead of
            // it: ceil(live / capacity). For a batch-1 engine (the shipped B=1
            // GPU path and the CPU reference walk) capacity == 1, so this is
            // exactly `live * service_ms` — byte-identical admission. For B > 1
            // it removes the serial-M/M/1 overestimate that 429'd every live
            // stream once `live * step_ms` crossed the SLO: a correct B=16
            // batch decoding at ~40 ms/token was sched-killed at 7 users even
            // though every user's real inter-token latency was ~40 ms (well
            // under the SLO). Proven on GPU: b16 8-way passes token-identity
            // with the shed off, sheds to 1 token/req with it on.
            let predicted_wait = predicted_wait_ms(live, service_capacity, load.service_ms.get());
            // A FLAT SLO cannot be right across decode batches. `service_ms` is
            // one tick of real work and grows with B (measured Gemma-4-12B:
            // 22 ms at B=8, 26 ms at B=16, 58 ms at B=32), so a constant 250 ms
            // silently becomes "shed everything" as the blob gets wider — a
            // B=32 blob 429'd every request at `predicted_wait_ms=329` with a
            // single live stream, and `vllm bench` reports those 429s as
            // SUCCESSFUL requests, which reads as a 2592 tok/s result. Floor the
            // SLO at a few ticks of the service time the engine actually shows,
            // so the knob keeps its meaning (a user waiting far longer than
            // normal) instead of tracking the blob's width.
            let slo = cfg.slo_ms.max(SLO_SERVICE_TICKS * load.service_ms.get());
            match admit(util, predicted_wait, slo, true) {
                Admit::Shed => {
                    Metrics::add(&metrics.admit_shed, live as u64);
                    tracing::warn!(
                        %slug,
                        live,
                        predicted_wait_ms = predicted_wait,
                        slo_ms = slo,
                        slo_ms_configured = cfg.slo_ms,
                        service_ms = load.service_ms.get(),
                        util,
                        "admission shed: dropping every live slot (429 to each)"
                    );
                    for s in slots.iter_mut() {
                        if let Some(slot) = s.take() {
                            release_kv(&arena, slot.kv);
                            let _ = slot.respond.try_send(StreamChunk::Err(
                                crate::RuntimeError::Rejected("arrival-rate admission shed".into()),
                            ));
                        }
                    }
                    if let Some(b) = taken_bufs.take() {
                        bufs_cache.insert(b.key, b);
                    }
                    continue;
                }
                Admit::Defer => {
                    let hold_ms = formation_window_ms(load.lambda.get(), cfg.max_hold_ms);
                    if hold_ms > 0.0 {
                        tokio::time::sleep(std::time::Duration::from_secs_f64(hold_ms / 1000.0))
                            .await;
                    }
                }
                Admit::Now => {}
            }

            Metrics::add(&metrics.batch_size_sum, live as u64);
            Metrics::inc(&metrics.batch_count);

            // One tick: advance every live slot by N tokens (multi-step).
            // Handed to the blocking pool so the dispatcher task stays hot
            // for arrivals.
            let steps = if cfg.multi_step {
                MultiStep::for_batch(live as i64).steps
            } else {
                1
            };
            let bundle_ref = Arc::clone(&bundle);
            let state_ref = Arc::clone(&state);
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

            let taken_slots = std::mem::take(&mut slots);
            let taken_obs = obs.take().unwrap_or_else(|| {
                let mut obs = RunObserver::new(state.record_trace, indirection_size);
                obs.set_kv_pages_range(kv_pages_range.clone());
                obs
            });
            let arena_ref = arena.clone();
            let kv_pages_for_tick = kv_pages_range.clone();

            let t_service_start = Instant::now();
            let tick = move || {
                run_one_tick(
                    &state_ref,
                    &bundle_ref,
                    key_for_tick,
                    vocab_for_tick,
                    taken_slots,
                    taken_bufs,
                    taken_obs,
                    arena_ref,
                    kv_pages_for_tick,
                    steps,
                )
            };
            // GPU models tick on the dedicated engine thread; the dispatcher
            // task stays hot for arrivals/cancellation either way.
            let joined = match &engine_thread {
                Some(t) => t.run(tick).await,
                None => tokio::task::spawn_blocking(tick)
                    .await
                    .map_err(|e| e.to_string()),
            };

            let ms = t_service_start.elapsed().as_millis() as f64;

            match joined {
                Ok((
                    returned_slots,
                    returned_bufs,
                    returned_obs,
                    tokens_produced,
                    did_prefill,
                    tick_fault,
                )) => {
                    // Decode-service EWMA: prefill ticks are excluded — see
                    // `service_sample`. Updating on them poisons the admission
                    // predictor and sheds live decode streams.
                    if let Some(sample) = service_sample(ms, did_prefill) {
                        load.service_ms.update(sample);
                        if let (Some(controller), Some(rung)) =
                            (rung_controller.as_mut(), tick_rung)
                        {
                            controller.observe_decode(rung, sample);
                        }
                    }
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
    });

    ModelMux {
        tx,
        metrics: handle_metrics,
        preempt: preempt_flag,
    }
}

fn note_dequeued(msg: &MuxMsg, metrics: &Metrics) {
    if matches!(msg, MuxMsg::Job(_)) {
        metrics.queued_requests.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Place a job into the first idle slot, asking the arena for the KV
/// footprint upfront. On KV OOM the request is dropped with a typed error —
/// the mux doesn't hold the slot open under memory pressure. The prompt
/// arrives pre-tokenized (see [`Job::prompt_ids`]).
fn admit_into(
    slots: &mut [Option<Slot>],
    job: Job,
    load: &mut LoadEstimator,
    last_arrival: &mut Option<Instant>,
    arena: Option<&SharedKvState>,
    metrics: &Metrics,
    health: &EngineHealth,
) {
    // A dead engine cannot serve anyone — reject up front with the fault that
    // killed it (fatal DeviceFault → 503), instead of admitting into a
    // poisoned context. The poisoning itself was logged once; this stays at
    // debug so a request flood doesn't become a log flood.
    if let EngineHealth::Dead(info) = health {
        tracing::debug!("mux: engine dead — request rejected");
        Metrics::inc(&metrics.rejected);
        let _ = job
            .respond
            .try_send(StreamChunk::Err(crate::RuntimeError::DeviceFault {
                info: info.clone(),
            }));
        return;
    }

    // Refresh λ from the inter-arrival gap.
    let now = job.arrived;
    if let Some(prev) = last_arrival.replace(now) {
        let dt = now.duration_since(prev).as_secs_f64();
        if dt > 1e-6 {
            load.lambda.update(1.0 / dt);
            // PUBLISH IT. `lambda_milli` existed and was exported as
            // `plowrt_arrival_rate` with no writer anywhere, so it read a flat
            // 0.000 next to a live `plowrt_utilization` — which scrapes as "no
            // traffic", not as "not implemented". The estimate was already
            // here; only the store was missing.
            metrics
                .lambda_milli
                .store((load.lambda.get() * 1000.0) as u64, Ordering::Relaxed);
        }
    } else {
        load.lambda.update(1.0);
    }

    let Some(idx) = slots.iter().position(|s| s.is_none()) else {
        // Capacity exhausted — reject fast rather than sitting on the request.
        tracing::warn!(
            capacity = slots.len(),
            "mux: no free slot — request rejected"
        );
        // Counted here and NOT in the admission-shed path above: both end as a
        // 429, but shedding is the controller dropping live work because
        // predicted wait passed the SLO, while this is arrival meeting a full
        // slot table. Adding them together hides which pressure caused the 429.
        Metrics::inc(&metrics.rejected);
        let _ = job
            .respond
            .try_send(StreamChunk::Err(crate::RuntimeError::Rejected(
                "engine at capacity — no free slot".into(),
            )));
        return;
    };

    let seq_upper = (job.prompt_ids.len() + job.gen.max_tokens.max(1)) as i64;

    let kv = if let Some(arena) = arena {
        match arena.lock().arena.allocate_slot(seq_upper) {
            Ok(h) => Some(h),
            Err(e) => {
                tracing::warn!(error = %e, seq_upper, "mux: kv arena OOM — request shed");
                let _ = job
                    .respond
                    .try_send(StreamChunk::Err(crate::RuntimeError::Oom(format!(
                        "kv: {e}"
                    ))));
                return;
            }
        }
    } else {
        None
    };

    slots[idx] = Some(Slot {
        prompt_ids: job.prompt_ids,
        out_ids: Vec::new(),
        gen: job.gen,
        arrived: job.arrived,
        respond: job.respond,
        prefix_offset: 0,
        read_offset: 0,
        executed: 0,
        step: 0,
        pf_pos: 0,
        cached_tokens: 0,
        kv,
    });
}

/// Return a slot's KV blocks to the arena (no-op when the slot never got one).
fn release_kv(arena: &Option<SharedKvState>, handle: Option<SlotHandle>) {
    if let (Some(a), Some(h)) = (arena.as_ref(), handle) {
        a.lock().arena.release_slot(h);
    }
}

/// Close every live slot with `FinishReason::Preempted` — the stream carries
/// everything generated so far plus honest usage, and the slot frees exactly
/// as it does when a client disconnects mid-generation (the sanctioned
/// teardown path: take the slot, release its KV; the engine's sequence slot
/// is reclaimed on the next admit).
fn preempt_slots(slots: &mut [Option<Slot>], arena: &Option<SharedKvState>) {
    for slot_opt in slots.iter_mut() {
        let Some(slot) = slot_opt.take() else {
            continue;
        };
        let _ = slot.respond.try_send(StreamChunk::Done {
            executed: slot.executed,
            reason: FinishReason::Preempted,
            usage: crate::serve::stream::TokenUsage {
                prompt_tokens: slot.prompt_ids.len(),
                cached_tokens: slot.cached_tokens,
                completion_tokens: slot.out_ids.len(),
            },
        });
        release_kv(arena, slot.kv);
    }
}

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
/// params/rng, and fires the bucket **once** via `AppState::step_batch`. The
/// produced tokens land in `obs.host.slot_tokens[row]`. This is the phase-3
/// "one bucket walk per tick" path.
///
/// **Fallback path.** When the bucket has no `SAMPLE_BATCH` (batch=1 rungs,
/// legacy buckets, tests using tiny_program), fall back to per-slot serial
/// ticks via `AppState::step_token` — the phase-1 behavior.
fn run_one_tick(
    state: &AppState,
    bundle: &ModelBundle,
    key: Option<BucketKey>,
    vocab: usize,
    mut slots: Vec<Option<Slot>>,
    mut bufs: Option<BucketBufs>,
    mut obs: RunObserver,
    arena: Option<SharedKvState>,
    kv_pages_range: std::ops::Range<usize>,
    steps: u32,
) -> (
    Vec<Option<Slot>>,
    Option<BucketBufs>,
    RunObserver,
    usize,
    bool,
    // First device fault seen this tick — the dispatcher's EngineHealth
    // signal. Always `None` on the CPU reference path.
    Option<crate::DeviceErrorInfo>,
) {
    let bucket = key.and_then(|k| bundle.bucket(k));
    let mut tokens_this_tick = 0usize;
    #[cfg_attr(not(any(feature = "cuda", feature = "hsa")), allow(unused_mut))]
    let mut tick_fault: Option<crate::DeviceErrorInfo> = None;

    // GPU path: when this model has an sm_120 engine, every token comes from
    // the persistent interpreter on the device — the CPU reference walk and
    // its stand-in logits are bypassed entirely. The engine drives B
    // independent sequence slots (the compiled PLOW_DECODE_BATCH; the slot
    // table is sized to it at spawn, so mux slot i IS engine slot i). Per
    // tick: prefill each new arrival into its own KV slot (sequential), then
    // ONE batched decode launch advances every already-running slot.
    #[cfg(any(feature = "cuda", feature = "hsa"))]
    if let Some(eng) = state.gpu_engine(bundle.network()) {
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

            // Whether this tick does any prefill work — reported to the dispatcher
            // so prefill tick durations never enter the decode-service EWMA.
            let did_prefill = slots
                .iter()
                .take(cap)
                .any(|s| s.as_ref().map(|s| s.step == 0).unwrap_or(false));

            // Decode feeds, gathered BEFORE the prefill pass so a slot prefilled
            // this tick (which just produced its first token) doesn't also step.
            let mut feeds: Vec<(usize, u32)> = slots
                .iter()
                .enumerate()
                .take(cap)
                .filter_map(|(i, s)| {
                    let s = s.as_ref()?;
                    (s.step > 0).then(|| (i, *s.out_ids.last().expect("step > 0 implies output")))
                })
                .collect();

            // PX-17: throughput mode — while any slot is mid-prefill, drop the decode
            // feeds so the prefill chain runs uninterrupted and no decode launch pays
            // its fixed cost at a partial batch. Every deferred row is picked up by a
            // full-batch decode tick once prefill drains.
            if pf_defer_decode() && did_prefill {
                feeds.clear();
            }

            let pack_t = packlog::on().then(Instant::now);
            if e.pf_batch_enabled() {
                // PX-1 cross-request batched prefill: pack every waiting request's
                // next chunk (up to `n-1` prompt rows each) into shared launches
                // under a token budget, then feed each finished request's LAST
                // prompt token through the batched decode step below — which both
                // writes its final KV row and produces its first token, batched
                // with every live decode stream.
                if let Some(f) =
                    gpu_prefill_batched_pass(&mut *e, &mut slots, cap, &arena, feeds.is_empty())
                {
                    if tick_fault.is_none() {
                        tick_fault = Some(f);
                    }
                }
                for i in 0..slots.len().min(cap) {
                    let Some(s) = slots[i].as_ref() else { continue };
                    if s.step != 0 {
                        continue;
                    }
                    let n = s.prompt_ids.len();
                    if n == 0 {
                        if let Some(taken) = slots[i].take() {
                            release_kv(&arena, taken.kv);
                            let _ = taken.respond.try_send(StreamChunk::Err(
                                crate::RuntimeError::Rejected("empty prompt".into()),
                            ));
                        }
                        continue;
                    }
                    if s.pf_pos + 1 != n {
                        continue; // still mid-prefill
                    }
                    if s.respond.is_closed() {
                        if let Some(taken) = slots[i].take() {
                            release_kv(&arena, taken.kv);
                        }
                        continue;
                    }
                    // A 1-token prompt never enters the batched pass — its slot is
                    // ready at pf_pos 0 but still needs its sequence begun.
                    if n == 1 && s.pf_pos == 0 {
                        if let Err(err) = e.begin_slot(i, n + s.gen.max_tokens.max(1)) {
                            note_fault(&mut tick_fault, &err);
                            if let Some(taken) = slots[i].take() {
                                release_kv(&arena, taken.kv);
                                let _ = taken.respond.try_send(StreamChunk::Err(err));
                            }
                            continue;
                        }
                    }
                    let last = *slots[i]
                        .as_ref()
                        .expect("checked Some")
                        .prompt_ids
                        .last()
                        .expect("n >= 1");
                    feeds.push((i, last));
                }
            } else {
                // Prefill pass — chunk-interleaved continuous batching. With live
                // decoders, at most ONE capped prefill chunk runs per tick, so a
                // mid-decode arrival stalls the running streams by one chunk (not one
                // whole prompt); the decode launch below runs between chunks. With no
                // decoders live, the chain runs to completion (fastest cold TTFT —
                // the pre-interleave behavior).
                let cap_rows = if feeds.is_empty() {
                    usize::MAX
                } else {
                    pf_interleave_rows()
                };
                loop {
                    let Some(i) = (0..slots.len().min(cap))
                        .find(|&i| slots[i].as_ref().map(|s| s.step == 0).unwrap_or(false))
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
                            handle_produced_token(
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
                            if let Some(taken) = slot_opt.take() {
                                release_kv(&arena, taken.kv);
                                let _ = taken.respond.try_send(StreamChunk::Err(err));
                            }
                        }
                    }
                    if !feeds.is_empty() {
                        break; // bounded stall: decode now, next chunk next tick
                    }
                }
            }

            let pack_prefill_ns = pack_t.map(|t| t.elapsed().as_nanos() as u64).unwrap_or(0);
            let pack_had_feeds = !feeds.is_empty();
            let dec_t = packlog::on().then(Instant::now);

            // One batched decode launch for every slot already past prefill. The
            // token buffer round-trips through `obs.host.slot_tokens` so the
            // per-tick hot path allocates nothing.
            if !feeds.is_empty() {
                let mut toks = std::mem::take(&mut obs.host.slot_tokens);
                // Bounded device multi-step (plan stage 5): when enabled and EVERY
                // fed row is greedy (temp==0; the device advance uses the argmax
                // token), run a K-token quantum with one host sync and stream up to
                // K tokens per row, stopping a row as soon as handle_produced_token
                // frees it (mid-quantum EOS / max_tokens — the extra device tokens
                // past the stop are discarded, bounded by K). Any stochastic row
                // falls through to the per-token path below.
                let use_multi = e.multistep_quantum().is_some()
                    && feeds.iter().all(|&(i, _)| {
                        slots[i]
                            .as_ref()
                            .map(|s| s.gen.params.temperature <= 0.0)
                            .unwrap_or(true)
                    });
                if use_multi {
                    match e.multi_step(&feeds, &mut toks) {
                        Ok(k) => {
                            for (ri, &(i, _)) in feeds.iter().enumerate() {
                                for s in 0..k {
                                    if slots[i].is_none() {
                                        break; // row stopped earlier this quantum
                                    }
                                    let token = toks[ri * k + s];
                                    tracing::debug!(token, slot = i, "gpu: token (multi-step)");
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
                            }
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
                            let msg = err.to_string();
                            for &(i, _) in &feeds {
                                if let Some(taken) = slots[i].take() {
                                    release_kv(&arena, taken.kv);
                                    let _ = taken
                                        .respond
                                        .try_send(StreamChunk::Err(fanout_err(&err, &msg)));
                                }
                            }
                        }
                    }
                    obs.host.slot_tokens = toks;
                    if let Some(dt) = dec_t {
                        packlog::record(
                            pack_prefill_ns,
                            dt.elapsed().as_nanos() as u64,
                            did_prefill,
                            pack_had_feeds,
                            feeds.len(),
                        );
                    }
                    return (slots, bufs, obs, tokens_this_tick, did_prefill, tick_fault);
                }
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
                let step_res = e.step_slots_sampled(&feeds, dev_specs.as_deref(), &mut toks);
                match step_res {
                    Ok(()) => {
                        for (&(i, _), &argmax_tok) in feeds.iter().zip(toks.iter()) {
                            let slot_opt = &mut slots[i];
                            let Some(slot) = slot_opt.as_mut() else {
                                continue;
                            };
                            // Device-sampled rows already hold their final token in
                            // `argmax_tok` (the sampler wrote in.ids); skip the host
                            // resample.
                            let was_dev = dev_specs
                                .as_ref()
                                .map(|_| dev_sample_spec(slot).is_some())
                                .unwrap_or(false);
                            let finished = if was_dev {
                                Ok(argmax_tok)
                            } else {
                                gpu_finish_token(&mut *e, i, slot, argmax_tok)
                            };
                            match finished {
                                Ok(token) => {
                                    tracing::debug!(
                                        token,
                                        slot = i,
                                        step = slot.step,
                                        "gpu: token"
                                    );
                                    handle_produced_token(
                                        slot_opt,
                                        &arena,
                                        bundle,
                                        token,
                                        1,
                                        &mut tokens_this_tick,
                                        Some(stop.as_slice()),
                                    );
                                }
                                Err(err) => {
                                    tracing::warn!(
                                        slot = i,
                                        error = %err,
                                        error_code = ?err.device_code(),
                                        fatal = err.is_fatal(),
                                        model = bundle.network(),
                                        "gpu: sample failed"
                                    );
                                    note_fault(&mut tick_fault, &err);
                                    if let Some(taken) = slot_opt.take() {
                                        release_kv(&arena, taken.kv);
                                        let _ = taken.respond.try_send(StreamChunk::Err(err));
                                    }
                                }
                            }
                        }
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
                        let msg = err.to_string();
                        for &(i, _) in &feeds {
                            if let Some(taken) = slots[i].take() {
                                release_kv(&arena, taken.kv);
                                let _ = taken
                                    .respond
                                    .try_send(StreamChunk::Err(fanout_err(&err, &msg)));
                            }
                        }
                    }
                }
                obs.host.slot_tokens = toks;
            }
            if let Some(dt) = dec_t {
                packlog::record(
                    pack_prefill_ns,
                    dt.elapsed().as_nanos() as u64,
                    did_prefill,
                    pack_had_feeds,
                    feeds.len(),
                );
            }
            return (slots, bufs, obs, tokens_this_tick, did_prefill, tick_fault);
        }

        // gfx950: B independent sequence slots, one decode dispatch for all of
        // them. Mux slot `i` IS engine slot `i`, so no owner map is needed —
        // the engine's slot table and this one are the same indices.
        //
        // A tick is EITHER one prefill (the prefill program is single-sequence
        // and holds the whole device, so at most one per tick) followed by one
        // batched decode step across every live slot. The two phases interleave
        // by tick but never overlap on the device because they share scratch.
        //
        // Irrefutable in an hsa-only build (the enum then has one variant) and
        // refutable alongside `cuda` — the same arm has to compile as both.
        #[cfg(feature = "hsa")]
        #[allow(irrefutable_let_patterns)]
        if let crate::serve::engine::ServeEngine::Amd(e) = &mut *guard {
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
            // Prefill and decode remain sequential inside the tick because they
            // share input/activation buffers. The parked-row mask prevents a
            // mid-prefill KDA state from advancing during the decode dispatch.
            //
            // `--pf-no-interleave` / `PLOW_PF_NO_INTERLEAVE=1` restores the old
            // prefill-only tick.
            let mut did_prefill = false;
            let nv = &crate::config::RuntimeConfig::get().nv;
            let no_interleave = nv.pf_no_interleave;
            let has_decode = slots[..b.min(slots.len())]
                .iter()
                .any(|s| s.as_ref().is_some_and(|s| s.step > 0));
            let tick_max = amd_prefill_tick_cap(
                has_decode,
                no_interleave,
                nv.pf_defer_decode,
                nv.pf_interleave,
            );
            let pending = amd_prefill_pick(
                (0..b.min(slots.len())).filter_map(|i| {
                    let slot = slots[i].as_ref()?;
                    (slot.step == 0).then_some((i, slot.arrived))
                }),
                nv.pf_batch,
                e.prefill_turn(),
                b.min(slots.len()),
            );
            if let Some(i) = pending {
                if nv.pf_batch {
                    e.advance_prefill_turn(i);
                }
                // §DISAGG phase-0. The AMD tick is EITHER a prefill OR a decode
                // (this arm returns before the decode launch below), so unlike
                // the CUDA arm there is no fused launch and `mixed_decode_ns`
                // is ZERO BY CONSTRUCTION. What disaggregation can recover here
                // is therefore the WHOLE prefill tick, during which every live
                // decode stream is stalled — so `prefill_ns / (prefill_ns +
                // decode_ns)` is the ceiling, not the mixed ratio.
                let pk_t = packlog::on().then(Instant::now);
                let slot_ref = slots[i].as_ref().expect("found above");
                // §TTFT: everything between `mux.submit` and this line — the
                // dispatcher wake, the formation hold, admission, and the
                // engine-thread handoff.
                crate::obs::ttft::QUEUE.add(slot_ref.arrived.elapsed().as_nanos() as u64);
                let t_pf = std::time::Instant::now();
                // ONE CHUNK, not the whole prompt. `Ok(None)` means this slot has more chunks to
                // go; it stays `step == 0` for a later tick. The default keeps the oldest prompt
                // active, while `PLOW_PF_BATCH=1` rotates fairly across pending AMD slots.
                let pf = e.prefill_chunked_at_most(i, &slot_ref.prompt_ids, tick_max);
                let frontier = e.prefill_frontier(i).unwrap_or(slot_ref.prompt_ids.len());
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
                if no_interleave {
                    return (slots, bufs, obs, tokens_this_tick, true, tick_fault);
                }
                let prefill_remains = slots[..b.min(slots.len())]
                    .iter()
                    .any(|s| s.as_ref().is_some_and(|s| s.step == 0));
                if amd_defer_decode(nv.pf_defer_decode, prefill_remains) {
                    tracing::debug!("amd: decode deferred while prefill remains");
                    return (slots, bufs, obs, tokens_this_tick, true, tick_fault);
                }
            }

            // Decode: every live slot feeds the token it last produced.
            let feeds: Vec<(usize, u32)> = (0..b.min(slots.len()))
                .filter_map(|i| {
                    let s = slots[i].as_ref()?;
                    Some((i, *s.out_ids.last()?))
                })
                .collect();
            if feeds.is_empty() {
                return (slots, bufs, obs, tokens_this_tick, did_prefill, tick_fault);
            }
            let pk_t = packlog::on().then(Instant::now);
            let pk_rows = feeds.len();
            // §DSTEP owns the whole tick from here, so `TOKEN` is a real total
            // and not a sum of parts. One tick is one token on the TP path,
            // which is the only shape `PLOW_DECODE_BATCH=1` admits.
            let t_tick = crate::obs::dstep::on().then(Instant::now);
            let remaining = feeds
                .iter()
                .filter_map(|&(i, _)| slots[i].as_ref())
                .map(|slot| slot.gen.max_tokens.saturating_sub(slot.out_ids.len()))
                .min()
                .unwrap_or(1);
            let requested = amd_multistep_requested(
                remaining,
                steps as usize,
                crate::config::RuntimeConfig::get().nv.multistep,
            );
            let multi = e.multistep_quantum(&feeds, requested);
            let mut deferred = std::mem::take(&mut obs.host.slot_tokens);
            let step_result = if let Some(quantum) = multi {
                e.multi_step(&feeds, quantum, &mut deferred)
                    .and_then(|quantum| {
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
                        Ok(())
                    })
            } else {
                e.step_batch(&feeds).map(|out| {
                    for (i, token) in out {
                        tracing::debug!(token, slot = i, "amd: token");
                        let t_stream = crate::obs::dstep::on().then(Instant::now);
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
                })
            };
            obs.host.slot_tokens = deferred;
            match step_result {
                Ok(out) => {
                    let _ = out;
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
            if let Some(t) = t_tick {
                crate::obs::dstep::token(t.elapsed().as_nanos() as u64);
            }
            if let Some(t) = pk_t {
                packlog::record(0, t.elapsed().as_nanos() as u64, false, true, pk_rows);
            }
            return (slots, bufs, obs, tokens_this_tick, did_prefill, tick_fault);
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
                    obs.host.slot_rng01.push(seeded_unit(
                        &slot.prompt_ids,
                        &slot.out_ids,
                        slot.step,
                    ));
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
            return (slots, bufs, obs, tokens_this_tick, false, tick_fault);
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
            let Some(slot) = slot_opt.as_mut() else {
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
                    )
                } else {
                    // No bucket in the bundle — direct-sample against reference
                    // logits. Matches the fallback in generate_with_bucket.
                    obs.host.tokens.clear();
                    obs.host.rng01 =
                        crate::serve::seeded_unit(&slot.prompt_ids, &slot.out_ids, slot.step);
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

    (slots, bufs, obs, tokens_this_tick, false, tick_fault)
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
/// (`PLOW_PF_INTERLEAVE` rows) and a long prompt runs MANY of them — feeding
/// them to the admission EWMA inflates `predicted_wait` past the SLO and the
/// shed kills every live decode stream (measured: any 32k prompt at the
/// default `--slo-ms 250` shed itself and its neighbors). Free-standing for
/// tests.
fn service_sample(ms: f64, did_prefill: bool) -> Option<f64> {
    (ms > 0.0 && !did_prefill).then_some(ms)
}

/// Predicted admission wait for a joining request, in ms.
///
/// `service_ms` is the wall time of ONE decode tick, which advances EVERY live
/// slot in a single batched launch (GPU engine / SAMPLE_BATCH bucket). A
/// joining request therefore waits on the number of *serial* batches ahead of
/// it — `ceil(live / capacity)` — not on `live` serial services. For a
/// batch-1 engine (`capacity == 1`: the shipped B=1 GPU path and the CPU
/// reference walk) this is exactly `live * service_ms`, byte-identical to the
/// pre-fix formula; for a B>1 batch it collapses to a single tick while any
/// slot is free, which is what a data-parallel launch actually costs.
fn predicted_wait_ms(live: usize, capacity: usize, service_ms: f64) -> f64 {
    live.div_ceil(capacity.max(1)) as f64 * service_ms
}

/// The serve-layer interleave bound: max prefill-chunk rows per tick while
/// other slots are mid-decode. `PLOW_PF_INTERLEAVE` overrides (rows; `0` =
/// whole prompt in one tick, the pre-interleave behavior). Read once.
/// Reads `RuntimeConfig::get().nv.pf_interleave_rows()`.
#[cfg(feature = "cuda")]
fn pf_interleave_rows() -> usize {
    crate::config::RuntimeConfig::get().nv.pf_interleave_rows()
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
    crate::config::RuntimeConfig::get().nv.pf_defer_decode
}

#[cfg(feature = "hsa")]
fn amd_prefill_tick_cap(
    has_decode: bool,
    no_interleave: bool,
    defer_decode: bool,
    interleave: u32,
) -> u32 {
    if !has_decode || no_interleave || defer_decode || interleave == 0 {
        u32::MAX
    } else {
        interleave
    }
}

#[cfg(feature = "hsa")]
fn amd_defer_decode(enabled: bool, prefill_remains: bool) -> bool {
    enabled && prefill_remains
}

#[cfg(feature = "hsa")]
fn amd_multistep_requested(remaining: usize, scheduler_steps: usize, configured: u32) -> usize {
    remaining
        .min(scheduler_steps)
        .min(configured.max(1) as usize)
}

#[cfg(feature = "hsa")]
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

/// RTX-12 chunked packing: per-REQUEST cap on the prefill rows one request may
/// contribute to a single batched launch. `PLOW_PF_CHUNK=C` clamps each waiting
/// request's slice to `C` rows so the `per_launch` budget (`PLOW_PF_INTERLEAVE`)
/// is shared by up to `R ≈ budget/C` requests instead of monopolized by the
/// first big prompt. `0` = uncapped = today's byte-identical behaviour (the A/B
/// canary; packing is numerics-neutral, so C only changes which requests share
/// a launch, never any request's tokens). Read once. Default `0` (off) — this
/// is opt-in alongside `PLOW_PF_BATCH=1`, matching the rest of the PX-1 knobs.
/// Unset and `0` both mean uncapped, so the default IS the zero sentinel.
/// Reads `RuntimeConfig::get().nv.pf_chunk_rows()`.
#[cfg(feature = "cuda")]
fn pf_chunk_rows() -> usize {
    crate::config::RuntimeConfig::get().nv.pf_chunk_rows()
}

/// PX-1: pack every mid-prefill slot's next chunk into shared batched-prefill
/// launches. Per launch, each waiting request contributes up to its remaining
/// `n-1` prompt rows (the last prompt token is fed through the batched decode
/// step instead — that step writes its KV row AND yields the first generated
/// token, so no per-request lm_head is needed in the shared launch). The row
/// budget per launch is the largest prefill bucket when no decode stream is
/// live (fastest cold TTFT), else `PLOW_PF_INTERLEAVE` rows — the same bounded
/// stall as the serialized path, now shared by N requests instead of one.
/// Chunk boundaries are bit-invariant for the fused (nsplit=1) flash — masked
/// tiles are exact no-ops and the KV tile grid is absolutely aligned — so
/// packing decisions cannot change any request's tokens.
#[cfg(feature = "cuda")]
fn gpu_prefill_batched_pass(
    e: &mut crate::exec::gpu::GpuEngine,
    slots: &mut [Option<Slot>],
    cap: usize,
    arena: &Option<SharedKvState>,
    cold: bool,
) -> Option<crate::DeviceErrorInfo> {
    use crate::exec::gpu::PfBatchReq;

    let mut tick_fault: Option<crate::DeviceErrorInfo> = None;
    let budget_max = e.pf_max_rows();
    if budget_max == 0 {
        return tick_fault;
    }
    let per_launch = if cold {
        budget_max
    } else {
        pf_interleave_rows().min(budget_max)
    };
    // RTX-12: per-request slice cap. With `chunk_cap < per_launch`, the first
    // big prompt no longer consumes the whole budget, so the slot-order pack
    // loop below hands the remaining budget to the next waiting request(s) —
    // P1 single-slice-per-request co-packing (R ≈ per_launch / chunk_cap).
    let chunk_cap = pf_chunk_rows();
    loop {
        // Minimal-padding budget: cap this pack at the largest bucket the
        // waiting rows can FILL (else the smallest covering bucket), so a
        // cold 4.5k-row pack runs [4096, tail] instead of an 8192 at ~45% pad.
        let avail: usize = slots
            .iter()
            .take(cap)
            .filter_map(|s| s.as_ref())
            .filter(|s| s.step == 0 && !s.respond.is_closed())
            .map(|s| (s.prompt_ids.len().max(1) - 1).saturating_sub(s.pf_pos))
            .sum();
        if avail == 0 {
            return tick_fault;
        }
        let per_launch = e.pf_pack_budget(avail.min(per_launch)).min(per_launch);
        // Assemble one pack: (slot, c0, len) per waiting request, in slot order.
        let mut pack: Vec<(usize, usize, usize)> = Vec::new();
        let mut budget = per_launch;
        for i in 0..slots.len().min(cap) {
            if budget == 0 {
                break;
            }
            let Some(s) = slots[i].as_ref() else { continue };
            if s.step != 0 {
                continue;
            }
            let n = s.prompt_ids.len();
            if n == 0 || s.pf_pos + 1 >= n {
                continue; // empty / ready — handled by the feed collection
            }
            if s.respond.is_closed() {
                if let Some(taken) = slots[i].take() {
                    release_kv(arena, taken.kv);
                }
                continue;
            }
            if s.pf_pos == 0 {
                if let Err(err) = e.begin_slot(i, n + s.gen.max_tokens.max(1)) {
                    tracing::warn!(
                        slot = i,
                        error = %err,
                        error_code = ?err.device_code(),
                        fatal = err.is_fatal(),
                        "gpu: pf-batch begin failed"
                    );
                    note_fault(&mut tick_fault, &err);
                    if let Some(taken) = slots[i].take() {
                        release_kv(arena, taken.kv);
                        let _ = taken.respond.try_send(StreamChunk::Err(err));
                    }
                    continue;
                }
            }
            let s = slots[i].as_ref().expect("checked Some");
            let take = (n - 1 - s.pf_pos).min(budget).min(chunk_cap);
            pack.push((i, s.pf_pos, take));
            budget -= take;
        }
        if pack.is_empty() {
            return tick_fault;
        }
        let reqs: Vec<PfBatchReq> = pack
            .iter()
            .map(|&(i, c0, len)| PfBatchReq {
                slot: i,
                prompt: &slots[i].as_ref().expect("packed slot is Some").prompt_ids,
                c0,
                len,
            })
            .collect();
        let res = e.prefill_batched(&reqs);
        drop(reqs);
        match res {
            Ok(()) => {
                for &(i, c0, len) in &pack {
                    slots[i].as_mut().expect("packed slot is Some").pf_pos = c0 + len;
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
                for &(i, _, _) in &pack {
                    if let Some(taken) = slots[i].take() {
                        release_kv(arena, taken.kv);
                        let _ = taken
                            .respond
                            .try_send(StreamChunk::Err(fanout_err(&err, &msg)));
                    }
                }
                return tick_fault;
            }
        }
        if !cold {
            return tick_fault; // bounded stall: decode now, next pack next tick
        }
        // Cold path: stop as soon as any request is ready so its first token
        // fires this tick; the rest continue next tick (with decoders live).
        let any_ready = slots.iter().take(cap).any(|s| {
            s.as_ref()
                .map(|s| {
                    s.step == 0 && !s.prompt_ids.is_empty() && s.pf_pos + 1 == s.prompt_ids.len()
                })
                .unwrap_or(false)
        });
        if any_ready {
            return tick_fault;
        }
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
    if slot.pf_pos == 0 {
        e.begin_slot(slot_idx, slot.prompt_ids.len() + slot.gen.max_tokens.max(1))?;
    }
    let tok = if e.has_prefill() {
        match e.prefill_chunk(slot_idx, &slot.prompt_ids, cap_rows)? {
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
        let start = e.attach_prompt(slot_idx, &slot.prompt_ids)?;
        slot.cached_tokens = start;
        for &t in &slot.prompt_ids[start..] {
            e.step_slots(&[(slot_idx, t)], &mut toks)?;
            tok = toks[0];
        }
        slot.pf_pos = slot.prompt_ids.len();
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
    gpu_finish_token(e, 0, slot, tok).map(Some)
}

/// Device-sampling eligibility for a slot (plan stage 4). `Some(spec)` when the
/// row is `temperature>0` with NO repetition penalty and NO logit bias — the
/// device sampler handles temperature/top_k/top_p/min_p but not penalties or
/// bias (those need per-row history the device doesn't own yet), so those rows
/// keep the host path. `rng01` is the same per-step seeded draw the host uses,
/// so a fixed seed stays reproducible. Greedy rows return `None` (the device
/// argmax already equals `ARGMAX_FIN`).
#[cfg(feature = "cuda")]
fn dev_sample_spec(slot: &Slot) -> Option<crate::exec::gpu::DevSample> {
    let p = &slot.gen.params;
    if p.temperature <= 0.0 || p.repetition_penalty != 1.0 || !p.logit_bias.is_empty() {
        return None;
    }
    Some(crate::exec::gpu::DevSample {
        temp: p.temperature,
        top_k: p.top_k as i32,
        top_p: p.top_p,
        min_p: p.min_p,
        rng01: seeded_unit(&slot.prompt_ids, &slot.out_ids, slot.step),
    })
}

/// Turn a slot's device-argmax token into the slot's produced token: greedy
/// passes it through untouched; `temperature > 0` downloads logits row `row`
/// and reuses the host sampler with penalties.
#[cfg(feature = "cuda")]
fn gpu_finish_token(
    e: &mut crate::exec::gpu::GpuEngine,
    row: usize,
    slot: &mut Slot,
    argmax_tok: u32,
) -> Result<u32> {
    if slot.gen.params.temperature > 0.0 {
        let mut logits = e.take_logits_buf();
        logits.clear();
        e.logits_row(row, &mut logits)?;
        crate::text::sample::apply_penalties(&mut logits, &slot.out_ids, &slot.gen.params);
        let rng = seeded_unit(&slot.prompt_ids, &slot.out_ids, slot.step);
        let tok = crate::text::sample::sample(&logits, &slot.gen.params, None, rng);
        e.return_logits_buf(logits);
        return Ok(tok);
    }
    Ok(argmax_tok)
}

/// Incremental detokenize over a bounded window (TGI scheme): decode only
/// `out_ids[*prefix..]` — O(window) per token instead of O(total) — and emit
/// the bytes past the `*prefix..*read` span's decode. The window advances only
/// when new visible bytes appear; a trailing replacement char (partial UTF-8
/// sequence mid-multibyte-token) holds the delta back until the sequence
/// completes. Free-standing so tests can drive it without a mux.
fn incremental_delta(
    tok: &dyn crate::text::tokenizer::Tokenize,
    out_ids: &[u32],
    prefix: &mut usize,
    read: &mut usize,
) -> String {
    const MAX_DETOKENIZE_WINDOW: usize = 16;
    let len = out_ids.len();
    let safe_start = len
        .saturating_sub(MAX_DETOKENIZE_WINDOW)
        .max((*prefix).min(len));
    let effective_read = (*read).clamp(safe_start, len);
    let prefix_text = tok.decode(&out_ids[safe_start..effective_read]);
    let new_text = tok.decode(&out_ids[safe_start..]);
    match new_text.get(prefix_text.len()..) {
        Some(d) if !d.is_empty() && !new_text.ends_with('\u{FFFD}') => {
            let d = d.to_string();
            *prefix = effective_read;
            *read = len;
            d
        }
        _ => String::new(),
    }
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
) {
    let Some(slot) = slot_opt.as_mut() else {
        return;
    };
    slot.out_ids.push(token);
    slot.executed += exec;
    slot.step += 1;
    *tokens_this_tick += 1;

    let delta = incremental_delta(
        bundle.tokenizer().as_ref(),
        &slot.out_ids,
        &mut slot.prefix_offset,
        &mut slot.read_offset,
    );

    // `try_send` fails two ways and they are NOT the same event: `Closed` is
    // the client dropping (implicit cancellation, the intended path) and `Full`
    // is a consumer 32 tokens behind (`serve/stream.rs` bounds the channel at
    // 32 — ~1.3 s at 40 ms/token, reachable on a slow SSE reader or TCP
    // backpressure). Both free the slot WITHOUT a `Done` or an `Err`, so the
    // receiver only sees the stream end. The receiving side must therefore
    // never render a terminal-less stream as a clean stop — see
    // `chat::buffer_and_reply` and `chat::sse_response`, which is where that is
    // enforced.
    // Stop conditions: the model's eos set when known (GPU path), else the
    // reference path's newline-byte heuristic; and max_tokens.
    //
    // COMPUTED BEFORE THE SEND, because a stop token's TEXT is framing and must not
    // reach the caller. It used to be computed after, which was invisible while every
    // stop id rendered as the empty string — `skip_special_tokens` drops a token flagged
    // `special`. Kimi-K3's turn ends at `<|close|>`, which its own
    // `added_tokens_decoder` flags `"special": false`, so it renders literally and the
    // answer came back as `The capital of France is Paris.<|close|>`.
    let stop_token = !slot.gen.ignore_eos
        && match stop_ids {
            Some(ids) => ids.contains(&token),
            None => token % 256 == u32::from(b'\n'),
        };
    if !stop_token
        && slot
            .respond
            .try_send(StreamChunk::Token {
                id: token,
                text: delta,
            })
            .is_err()
    {
        if let Some(taken) = slot_opt.take() {
            release_kv(arena, taken.kv);
        }
        return;
    }
    let stop_max = slot.step >= slot.gen.max_tokens.max(1);
    if stop_token || stop_max {
        let reason = if stop_max && !stop_token {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        let _ = slot.respond.try_send(StreamChunk::Done {
            executed: slot.executed,
            reason,
            usage: crate::serve::stream::TokenUsage {
                prompt_tokens: slot.prompt_ids.len(),
                cached_tokens: slot.cached_tokens,
                completion_tokens: slot.out_ids.len(),
            },
        });
        if let Some(taken) = slot_opt.take() {
            release_kv(arena, taken.kv);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::indirection::slots as ind_slots;
    use crate::serve::RunObserver;
    use plow_asset::{KvLayerPaging, KvPaging};

    fn test_job() -> Job {
        let (respond, _rx) = crate::serve::stream::channel();
        Job {
            prompt_ids: vec![1],
            gen: GenParams::default(),
            arrived: Instant::now(),
            respond,
        }
    }

    #[test]
    fn deferred_token_ring_is_row_major_and_bounds_checked() {
        let ring = [10, 11, 12, 20, 21, 22];
        assert_eq!(deferred_token(&ring, 1, 2, 3).unwrap(), 22);
        assert!(deferred_token(&ring, 2, 0, 3).is_err());
    }

    #[cfg(feature = "hsa")]
    #[test]
    fn amd_multistep_honors_the_runtime_cap() {
        assert_eq!(amd_multistep_requested(16, 8, 4), 4);
        assert_eq!(amd_multistep_requested(16, 8, 1), 1);
        assert_eq!(amd_multistep_requested(3, 8, 4), 3);
    }

    #[test]
    fn bounded_ingress_reports_full_closed_and_depth() {
        let metrics = Arc::new(Metrics::default());
        let (tx, mut rx) = mpsc::channel(1);
        let mux = ModelMux {
            tx,
            metrics: Arc::clone(&metrics),
            preempt: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };

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
    #[cfg(any(feature = "cuda", feature = "hsa"))]
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

    /// Prefill ticks must never enter the decode-service EWMA (the admission
    /// shed regression: one long prefill tick > slo_ms killed every live
    /// decode stream). Decode ticks always do.
    #[test]
    fn service_sample_excludes_prefill_ticks() {
        assert_eq!(service_sample(420.0, true), None);
        assert_eq!(service_sample(18.3, false), Some(18.3));
        assert_eq!(service_sample(0.0, false), None);

        // The shed math this guards: live=1, slo=250 — a decode-scale EWMA
        // admits, a prefill-poisoned one sheds.
        let mut svc = crate::sched::admission::Ewma::new(0.2);
        for _ in 0..16 {
            svc.update(service_sample(420.0, true).unwrap_or(18.0));
        }
        assert!(svc.get() < 250.0, "decode EWMA stays under the SLO");
        let mut poisoned = crate::sched::admission::Ewma::new(0.2);
        for _ in 0..16 {
            poisoned.update(420.0);
        }
        assert!(poisoned.get() > 250.0, "control: unfiltered EWMA sheds");
    }

    #[cfg(feature = "hsa")]
    #[test]
    fn amd_prefill_scheduler_controls_are_bounded() {
        assert_eq!(amd_prefill_tick_cap(false, false, false, 2048), u32::MAX);
        assert_eq!(amd_prefill_tick_cap(true, true, false, 2048), u32::MAX);
        assert_eq!(amd_prefill_tick_cap(true, false, false, 0), u32::MAX);
        assert_eq!(amd_prefill_tick_cap(true, false, true, 2048), u32::MAX);
        assert_eq!(amd_prefill_tick_cap(true, false, false, 2048), 2048);
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

    /// The batched-engine admission model: a decode tick advances every live
    /// slot in ONE launch, so `predicted_wait` must not scale with `live` up to
    /// the batch capacity. Regression guard for the B>8 serving-capacity bug —
    /// a correct B=16 blob at ~40 ms/token was shed-killed at 7 live users
    /// because the old `live * service_ms` formula predicted 280 ms > 250 ms SLO
    /// even though every user's real inter-token latency was ~40 ms.
    #[test]
    fn predicted_wait_is_batched_not_serial() {
        // Batch-1 engine (shipped B=1 GPU path / CPU reference walk): identical
        // to the old serial formula `live * service_ms`.
        for live in 0..4 {
            assert_eq!(
                predicted_wait_ms(live, 1, 40.0),
                live as f64 * 40.0,
                "capacity=1 must equal the pre-fix serial formula"
            );
        }
        // B=16 engine at 40 ms/token: all 8 (and all 16) live users share one
        // batched tick, so the wait stays at one service time — under a 250 ms
        // SLO where the old formula (8*40=320) sheds.
        assert_eq!(predicted_wait_ms(8, 16, 40.0), 40.0);
        assert_eq!(predicted_wait_ms(16, 16, 40.0), 40.0);
        assert!(predicted_wait_ms(8, 16, 40.0) <= 250.0, "b16@8 must admit");
        assert!(8.0 * 40.0 > 250.0, "control: old serial formula would shed");
        // No live slots → no wait.
        assert_eq!(predicted_wait_ms(0, 16, 40.0), 0.0);
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
            streamed.push_str(&incremental_delta(&tok, &fed, &mut prefix, &mut read));
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
