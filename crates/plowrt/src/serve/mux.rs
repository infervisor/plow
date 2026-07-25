//! §I Per-model request muxer — slot-oriented continuous-batching engine.
//!
//! Each loaded model has one dispatcher task with an unbounded MPSC ingress.
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
//! `plans/serve-request-mux.md`.

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
use crate::serve::stream::{ChunkSender, FinishReason, StreamChunk};
use crate::serve::{
    bucket_has_sample_batch, reference_logits_row, sample_vocab, seeded_unit, AppState, GenParams,
    RunObserver,
};
use crate::Result;

/// PX-1 packing measurement (RTX-12 baseline). When `PLOW_PF_PACKLOG=1`, the
/// GPU tick accumulates wall time spent in the batched-prefill pass vs the
/// batched-decode launch and periodically emits a `PACKLOG WALL ...` stderr
/// line, so the bench can report the prefill/decode wall-time split per cell.
/// Off by default → the hot path pays only cached relaxed atomic loads.
mod packlog {
    use std::sync::atomic::{AtomicU64, Ordering};

    static PREFILL_NS: AtomicU64 = AtomicU64::new(0);
    static DECODE_NS: AtomicU64 = AtomicU64::new(0);
    static PREFILL_TICKS: AtomicU64 = AtomicU64::new(0);
    static DECODE_TICKS: AtomicU64 = AtomicU64::new(0);
    static TICKS: AtomicU64 = AtomicU64::new(0);

    crate::env_flag!(pub fn on, "PLOW_PF_PACKLOG");

    /// Record one GPU tick's prefill-pass and decode-launch wall times (ns).
    /// `did_prefill`/`did_decode` count how many ticks touched each phase.
    /// Emits a cumulative summary every 1000 ticks; the bench slices the log
    /// by line-count brackets to get per-cell deltas.
    pub fn record(prefill_ns: u64, decode_ns: u64, did_prefill: bool, did_decode: bool) {
        PREFILL_NS.fetch_add(prefill_ns, Ordering::Relaxed);
        DECODE_NS.fetch_add(decode_ns, Ordering::Relaxed);
        if did_prefill {
            PREFILL_TICKS.fetch_add(1, Ordering::Relaxed);
        }
        if did_decode {
            DECODE_TICKS.fetch_add(1, Ordering::Relaxed);
        }
        let n = TICKS.fetch_add(1, Ordering::Relaxed) + 1;
        if n % 1000 == 0 {
            emit();
        }
    }

    fn emit() {
        eprintln!(
            "PACKLOG WALL prefill_ns={} decode_ns={} prefill_ticks={} decode_ticks={} ticks={}",
            PREFILL_NS.load(Ordering::Relaxed),
            DECODE_NS.load(Ordering::Relaxed),
            PREFILL_TICKS.load(Ordering::Relaxed),
            DECODE_TICKS.load(Ordering::Relaxed),
            TICKS.load(Ordering::Relaxed),
        );
    }
}

/// Dispatcher config. Cold-path — set once at startup from CLI flags.
#[derive(Clone, Copy, Debug)]
pub struct MuxConfig {
    /// Upper bound on the arrival-rate-driven batch-formation hold (ms) used
    /// only in the cold-start path (empty slot table) — with slots always
    /// draining, the hot path never sleeps.
    pub max_hold_ms: f64,
    /// SLO used by admission (predicted wait above this sheds the request).
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
}

impl Default for MuxConfig {
    fn default() -> Self {
        MuxConfig {
            max_hold_ms: 8.0,
            slo_ms: 250.0,
            multi_step: true,
            queue_depth: 0,
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
    tx: mpsc::UnboundedSender<MuxMsg>,
}

/// Internal messages to the dispatcher: jobs or control signals.
enum MuxMsg {
    Job(Job),
    /// Graceful drain: stop admitting new requests, finish in-flight slots,
    /// then signal completion through the oneshot.
    Drain(tokio::sync::oneshot::Sender<()>),
}

impl ModelMux {
    /// Submit a job. Returns immediately; the caller awaits the stream.
    pub fn submit(&self, job: Job) -> std::result::Result<(), Job> {
        self.tx.send(MuxMsg::Job(job)).map_err(|e| match e.0 {
            MuxMsg::Job(j) => j,
            _ => unreachable!(),
        })
    }

    /// Initiate graceful drain: no new requests accepted, all live slots run to
    /// completion. Returns when every in-flight slot has finished. Use before
    /// `Registry::unload` to avoid mid-generation errors.
    pub async fn drain(&self) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.tx.send(MuxMsg::Drain(tx));
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
    let (tx, mut rx) = mpsc::unbounded_channel::<MuxMsg>();
    let metrics = Arc::clone(&state.metrics);

    // Slot capacity from the compiler-emitted ladder — the largest decode
    // bucket sets the ceiling for concurrent live requests.
    let capacity = bundle
        .bucket_keys()
        .filter(|k| k.phase == Phase::Decode)
        .map(|k| k.batch.max(1) as usize)
        .max()
        .unwrap_or(1);
    // GPU-engine bundles are bucketless: the ceiling is the engine's compiled
    // decode batch (PLOW_DECODE_BATCH), one slot per engine sequence slot —
    // mux slot i IS engine slot i.
    #[cfg(feature = "cuda")]
    let capacity = state
        .gpu_engine(bundle.network())
        .map(|e| e.lock().batch())
        .unwrap_or(capacity);

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
        #[cfg(feature = "cuda")]
        let has_gpu = state.gpu_engine(bundle.network()).is_some();
        #[cfg(not(feature = "cuda"))]
        let has_gpu = false;
        // Dedicated engine/submission thread for GPU models: every tick runs
        // on ONE persistent OS thread (CUDA context bound once, no
        // blocking-pool dispatch). CPU-reference models keep spawn_blocking.
        let engine_thread =
            has_gpu.then(|| crate::exec::engine_thread::EngineThread::spawn(format!("plow-eng-{slug}")));

        loop {
            let live = slots.iter().filter(|s| s.is_some()).count();

            // Drain completion: if draining and no in-flight slots remain,
            // signal the drain future and exit the dispatcher loop.
            if draining && live == 0 {
                if let Some(done) = drain_done.take() {
                    let _ = done.send(());
                }
                break;
            }

            // Cold start: no live slots — block until an arrival (or exit
            // when every ModelMux clone has dropped and the channel closes).
            if live == 0 {
                let Some(msg) = rx.recv().await else { break };
                match msg {
                    MuxMsg::Job(job) => {
                        admit_into(
                            &mut slots,
                            job,
                            &mut load,
                            &mut last_arrival,
                            arena.as_ref(),
                        );
                    }
                    MuxMsg::Drain(done) => {
                        // No in-flight work; signal immediately.
                        let _ = done.send(());
                        break;
                    }
                }
            }

            // Non-blocking drain: fill every idle slot the queue can serve.
            // A short hold when we still have empty slots and no live work
            // yet keeps us from waking up on a single arrival amid a burst.
            let idle = slots.iter().filter(|s| s.is_none()).count();
            if !draining && idle > 0 {
                let lambda = load.lambda.get();
                let live_now = capacity - idle;
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
                    while slots.iter().any(|s| s.is_none()) {
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        match tokio::time::timeout(remaining, rx.recv()).await {
                            Ok(Some(MuxMsg::Job(job))) => admit_into(
                                &mut slots,
                                job,
                                &mut load,
                                &mut last_arrival,
                                arena.as_ref(),
                            ),
                            Ok(Some(MuxMsg::Drain(done))) => {
                                draining = true;
                                drain_done = Some(done);
                                break;
                            }
                            Ok(None) => break,
                            Err(_) => break,
                        }
                    }
                }
                // Any additional pending arrivals (no wait).
                while !draining && slots.iter().any(|s| s.is_none()) {
                    match rx.try_recv() {
                        Ok(MuxMsg::Job(job)) => admit_into(
                            &mut slots,
                            job,
                            &mut load,
                            &mut last_arrival,
                            arena.as_ref(),
                        ),
                        Ok(MuxMsg::Drain(done)) => {
                            draining = true;
                            drain_done = Some(done);
                            break;
                        }
                        Err(_) => break,
                    }
                }
            }

            let live = slots.iter().filter(|s| s.is_some()).count();
            if live == 0 {
                continue;
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
            let predicted_wait = predicted_wait_ms(live, capacity, load.service_ms.get());
            match admit(util, predicted_wait, cfg.slo_ms, true) {
                Admit::Shed => {
                    Metrics::add(&metrics.admit_shed, live as u64);
                    tracing::warn!(
                        %slug,
                        live,
                        predicted_wait_ms = predicted_wait,
                        slo_ms = cfg.slo_ms,
                        util,
                        "admission shed: dropping every live slot (429 to each)"
                    );
                    for s in slots.iter_mut() {
                        if let Some(slot) = s.take() {
                            release_kv(&arena, slot.kv);
                            let _ =
                                slot.respond
                                    .try_send(StreamChunk::Err(crate::RuntimeError::Rejected(
                                        "arrival-rate admission shed".into(),
                                    )));
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
                None => tokio::task::spawn_blocking(tick).await.map_err(|e| e.to_string()),
            };

            let ms = t_service_start.elapsed().as_millis() as f64;

            match joined {
                Ok((returned_slots, returned_bufs, returned_obs, tokens_produced, did_prefill)) => {
                    // Decode-service EWMA: prefill ticks are excluded — see
                    // `service_sample`. Updating on them poisons the admission
                    // predictor and sheds live decode streams.
                    if let Some(sample) = service_sample(ms, did_prefill) {
                        load.service_ms.update(sample);
                    }
                    slots = returned_slots;
                    if let Some(b) = returned_bufs {
                        bufs_cache.insert(b.key, b);
                    }
                    obs = Some(returned_obs);
                    Metrics::add(&metrics.tokens, tokens_produced as u64);
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

    ModelMux { tx }
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
) {
    // Refresh λ from the inter-arrival gap.
    let now = job.arrived;
    if let Some(prev) = last_arrival.replace(now) {
        let dt = now.duration_since(prev).as_secs_f64();
        if dt > 1e-6 {
            load.lambda.update(1.0 / dt);
        }
    } else {
        load.lambda.update(1.0);
    }

    let Some(idx) = slots.iter().position(|s| s.is_none()) else {
        // Capacity exhausted — reject fast rather than sitting on the request.
        tracing::warn!(capacity = slots.len(), "mux: no free slot — request rejected");
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
) -> (Vec<Option<Slot>>, Option<BucketBufs>, RunObserver, usize, bool) {
    let bucket = key.and_then(|k| bundle.bucket(k));
    let mut tokens_this_tick = 0usize;

    // GPU path: when this model has an sm_120 engine, every token comes from
    // the persistent interpreter on the device — the CPU reference walk and
    // its stand-in logits are bypassed entirely. The engine drives B
    // independent sequence slots (the compiled PLOW_DECODE_BATCH; the slot
    // table is sized to it at spawn, so mux slot i IS engine slot i). Per
    // tick: prefill each new arrival into its own KV slot (sequential), then
    // ONE batched decode launch advances every already-running slot.
    #[cfg(feature = "cuda")]
    if let Some(eng) = state.gpu_engine(bundle.network()) {
        let mut e = eng.lock();
        let stop = Arc::clone(e.stop_ids());
        let cap = e.batch();

        // Slots past the engine batch cannot be served — only reachable on a
        // capacity/engine mismatch, and better a loud 429 than a hang.
        for slot_opt in slots.iter_mut().skip(cap) {
            if let Some(taken) = slot_opt.take() {
                tracing::warn!(cap, "gpu: slot past engine batch rejected");
                release_kv(&arena, taken.kv);
                let _ = taken
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

        let pack_t = packlog::on().then(Instant::now);
        if e.pf_batch_enabled() {
            // PX-1 cross-request batched prefill: pack every waiting request's
            // next chunk (up to `n-1` prompt rows each) into shared launches
            // under a token budget, then feed each finished request's LAST
            // prompt token through the batched decode step below — which both
            // writes its final KV row and produces its first token, batched
            // with every live decode stream.
            gpu_prefill_batched_pass(&mut e, &mut slots, cap, &arena, feeds.is_empty());
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
                let res = gpu_prefill_advance(
                    &mut e,
                    i,
                    slot_opt.as_mut().expect("checked Some"),
                    cap_rows,
                );
                match res {
                    Ok(Some(token)) => {
                        tracing::debug!(token, slot = i, step = 0usize, "gpu: token");
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
                    Ok(None) => {
                        // Mid-prefill: the frontier advanced one chunk.
                    }
                    Err(err) => {
                        tracing::warn!(slot = i, error = %err, "gpu: prefill failed");
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
                        tracing::warn!(error = %err, fed = feeds.len(), "gpu: multi-step failed");
                        let msg = err.to_string();
                        for &(i, _) in &feeds {
                            if let Some(taken) = slots[i].take() {
                                release_kv(&arena, taken.kv);
                                let _ = taken.respond.try_send(StreamChunk::Err(
                                    crate::RuntimeError::Msg(msg.clone()),
                                ));
                            }
                        }
                    }
                }
                obs.host.slot_tokens = toks;
                if let Some(dt) = dec_t {
                    packlog::record(pack_prefill_ns, dt.elapsed().as_nanos() as u64, did_prefill, pack_had_feeds);
                }
                return (slots, bufs, obs, tokens_this_tick, did_prefill);
            }
            // Device sampling (plan stage 4): when the engine has a sampler,
            // build a batch-wide spec array so eligible temperature>0 rows are
            // sampled on-device (token lands in in.ids, no vocab-row D2H); a
            // row is device-sampled iff temp>0 with no penalties/logit-bias
            // (those still need the host path). `dev_sampled` marks which rows
            // must NOT be host-resampled afterwards.
            let dev_specs: Option<Vec<crate::exec::gpu::DevSample>> = if e.dev_sample_enabled() {
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
                            gpu_finish_token(&mut e, i, slot, argmax_tok)
                        };
                        match finished {
                            Ok(token) => {
                                tracing::debug!(token, slot = i, step = slot.step, "gpu: token");
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
                                tracing::warn!(slot = i, error = %err, "gpu: sample failed");
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
                    tracing::warn!(error = %err, fed = feeds.len(), "gpu: decode launch failed");
                    let msg = err.to_string();
                    for &(i, _) in &feeds {
                        if let Some(taken) = slots[i].take() {
                            release_kv(&arena, taken.kv);
                            let _ = taken
                                .respond
                                .try_send(StreamChunk::Err(crate::RuntimeError::Msg(msg.clone())));
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
            );
        }
        return (slots, bufs, obs, tokens_this_tick, did_prefill);
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
                                let _ = slot
                                    .respond
                                    .try_send(StreamChunk::Err(crate::RuntimeError::Msg(msg.clone())));
                            }
                        }
                    }
                }
            }
            return (slots, bufs, obs, tokens_this_tick, false);
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
        refresh_indirection(&mut obs, live_kv_rows(&slots), &arena, kv_pages_range.clone());
        for slot_opt in slots.iter_mut() {
            let Some(slot) = slot_opt.as_mut() else {
                continue;
            };

            obs.host.params = slot.gen.params.clone();

            let token_res: Result<(u32, usize)> = if let (Some(bucket), Some(bufs)) =
                (bucket, bufs.as_mut())
            {
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

    (slots, bufs, obs, tokens_this_tick, false)
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
#[cfg(feature = "cuda")]
fn pf_interleave_rows() -> usize {
    use std::sync::OnceLock;
    static ROWS: OnceLock<usize> = OnceLock::new();
    *ROWS.get_or_init(|| {
        let rows = std::env::var("PLOW_PF_INTERLEAVE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2048);
        if rows == 0 {
            usize::MAX
        } else {
            rows
        }
    })
}

/// RTX-12 chunked packing: per-REQUEST cap on the prefill rows one request may
/// contribute to a single batched launch. `PLOW_PF_CHUNK=C` clamps each waiting
/// request's slice to `C` rows so the `per_launch` budget (`PLOW_PF_INTERLEAVE`)
/// is shared by up to `R ≈ budget/C` requests instead of monopolized by the
/// first big prompt. `0` = uncapped = today's byte-identical behaviour (the A/B
/// canary; packing is numerics-neutral, so C only changes which requests share
/// a launch, never any request's tokens). Read once. Default `0` (off) — this
/// is opt-in alongside `PLOW_PF_BATCH=1`, matching the rest of the PX-1 knobs.
#[cfg(feature = "cuda")]
fn pf_chunk_rows() -> usize {
    use std::sync::OnceLock;
    static ROWS: OnceLock<usize> = OnceLock::new();
    *ROWS.get_or_init(|| {
        let c = std::env::var("PLOW_PF_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        if c == 0 {
            usize::MAX
        } else {
            c
        }
    })
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
) {
    use crate::exec::gpu::PfBatchReq;

    let budget_max = e.pf_max_rows();
    if budget_max == 0 {
        return;
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
            return;
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
                    tracing::warn!(slot = i, error = %err, "gpu: pf-batch begin failed");
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
            return;
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
                tracing::warn!(error = %err, packed = pack.len(), "gpu: batched prefill failed");
                let msg = err.to_string();
                for &(i, _, _) in &pack {
                    if let Some(taken) = slots[i].take() {
                        release_kv(arena, taken.kv);
                        let _ = taken
                            .respond
                            .try_send(StreamChunk::Err(crate::RuntimeError::Msg(msg.clone())));
                    }
                }
                return;
            }
        }
        if !cold {
            return; // bounded stall: decode now, next pack next tick
        }
        // Cold path: stop as soon as any request is ready so its first token
        // fires this tick; the rest continue next tick (with decoders live).
        let any_ready = slots.iter().take(cap).any(|s| {
            s.as_ref()
                .map(|s| s.step == 0 && !s.prompt_ids.is_empty() && s.pf_pos + 1 == s.prompt_ids.len())
                .unwrap_or(false)
        });
        if any_ready {
            return;
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
        let tok = crate::text::sample::sample(
            &logits,
            &slot.gen.params,
            None,
            rng,
        );
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

    // Client dropped → free the slot (implicit cancellation).
    if slot
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

    // Stop conditions: the model's eos set when known (GPU path), else the
    // reference path's newline-byte heuristic; and max_tokens.
    let stop_token = match stop_ids {
        Some(ids) => ids.contains(&token),
        None => token % 256 == u32::from(b'\n'),
    };
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
