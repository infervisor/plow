//! §G OpenAI-compatible API server.

pub mod chat;
/// The loaded device engine behind a slug, as one type over both backends —
/// the seam that lets `serve` stop being CUDA-only.
#[cfg(any(feature = "cuda", feature = "hsa"))]
pub mod engine;
#[cfg(feature = "cuda")]
pub mod manager;
pub mod models;
pub mod mux;
pub mod openai;
pub mod stream;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;

use crate::asset::{Bucket, BucketKey, Phase};
use crate::device::cpu::StepObserver;
use crate::exec::host::HostExecutor;
use crate::exec::indirection::IndirectionTable;
use crate::exec::ExecutorSet;
use crate::obs::trace::{TaskSpan, Timeline};
use crate::obs::Metrics;
use crate::orch::Registry;
use crate::sched::{batching, Scheduler};
use crate::text::sample::{self, SamplingParams};
use crate::{Result, RuntimeError};

/// Per-request generation controls, built from the API request.
#[derive(Clone, Debug)]
pub struct GenParams {
    pub max_tokens: usize,
    pub params: SamplingParams,
    /// Suppress the model's eos/stop set so generation runs to `max_tokens`.
    /// vLLM's `ignore_eos`; `vllm bench serve` sets it for the synthetic
    /// datasets so every request emits exactly `--random-output-len` tokens.
    /// Without it a benchmark measures a mix of short and capped responses and
    /// under-reports steady-state throughput (measured: 161 vs 512 tokens per
    /// request on the same prompts, i.e. 3.2x the prefill churn per output
    /// token). Default `false` — normal serving is unchanged.
    pub ignore_eos: bool,
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams {
            max_tokens: 4096,
            params: SamplingParams::default(),
            ignore_eos: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{status_for, GenParams};
    use crate::{DeviceErrorInfo, RuntimeError};
    use axum::http::StatusCode;

    #[test]
    fn generation_default_allows_long_responses() {
        assert_eq!(GenParams::default().max_tokens, 4096);
    }

    fn fault(fatal: bool) -> RuntimeError {
        RuntimeError::DeviceFault {
            info: DeviceErrorInfo {
                operation: "cuStreamSynchronize".into(),
                code: 719,
                name: "CUDA_ERROR_LAUNCH_FAILED".into(),
                fatal,
            },
        }
    }

    #[test]
    fn status_maps_fatal_fault_to_503_and_transient_to_500() {
        assert_eq!(status_for(&fault(true)), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(status_for(&fault(false)), StatusCode::INTERNAL_SERVER_ERROR);
        // Existing mappings preserved.
        assert_eq!(
            status_for(&RuntimeError::UnknownModel("x".into())),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            status_for(&RuntimeError::Rejected("shed".into())),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            status_for(&RuntimeError::Oom("kv".into())),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            status_for(&RuntimeError::Device("validation".into())),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}

/// A run observer that both executes host token ops (via [`HostExecutor`]) and,
/// when tracing is on, records timeline spans — one pass over the schedule.
///
/// The `indirection` table is the SETUP_INDIRECTION analogue for the mux:
/// before each tick the mux writes per-slot per-layer KV block addresses into
/// its `KV_PAGES` region. The CPU reference interpreter doesn't consume them
/// yet (it uses `reference_logits`), but the plumbing is in place — real
/// attention kernels resolve slot → address through this table.
pub(crate) struct RunObserver {
    pub host: HostExecutor,
    pub record: bool,
    pub spans: Vec<TaskSpan>,
    /// Per-tick indirection table. The mux sizes this for its complete
    /// `max_batch × n_layers` KV region (with 64 as the legacy minimum).
    pub indirection: IndirectionTable,
    /// Model-specific slice of `indirection` occupied by KV addresses.
    kv_pages_range: std::ops::Range<usize>,
    /// Per-`FLASH`-fire trace: the KV_PAGES snapshot the interpreter saw when
    /// the attention op fired. The reference interpreter doesn't compute real
    /// attention, but reading the indirection here proves the compiler →
    /// runtime KV-address wire end-to-end: whatever `refresh_indirection`
    /// wrote before the tick shows up in exactly the fires that would have
    /// consumed it on a real device.
    pub kv_writes: Vec<KvWrite>,
}

/// One recorded consumption of `KV_PAGES` by a `FLASH`-family packet.
#[derive(Debug, Clone)]
pub struct KvWrite {
    /// The packet index (position in the bucket's inst vec) that fired.
    pub packet_index: u32,
    /// The `KV_PAGES` slice contents at fire time — one address per KV slot.
    pub addresses: Vec<u64>,
}

/// Default capacity of the per-tick indirection table.
pub(crate) const RUN_INDIRECTION_SIZE: usize = 64;

impl RunObserver {
    pub(crate) fn new(record: bool, table_size: usize) -> Self {
        RunObserver {
            host: HostExecutor::new(),
            record,
            spans: Vec::new(),
            indirection: IndirectionTable::new(table_size.max(RUN_INDIRECTION_SIZE)),
            kv_pages_range: crate::exec::indirection::slots::kv_pages(0, 0),
            kv_writes: Vec::new(),
        }
    }

    pub(crate) fn set_kv_pages_range(&mut self, range: std::ops::Range<usize>) {
        debug_assert!(range.end <= self.indirection.len());
        self.kv_pages_range = range;
    }

    /// Clear per-tick traces (`kv_writes`) without touching persistent state.
    /// The mux calls this at the top of each tick so the trace doesn't grow
    /// across ticks.
    pub(crate) fn clear_tick_traces(&mut self) {
        self.kv_writes.clear();
    }
}

impl StepObserver for RunObserver {
    #[inline]
    fn run_math(&self) -> bool {
        false
    }
    fn on_fire(&mut self, i: usize, inst: &packet::Inst, t0: u64, t1: u64) {
        self.host.on_fire(i, inst, t0, t1);

        // FLASH consumes KV: snapshot the `KV_PAGES` region from the
        // indirection table so the mux (or a test) can prove the compiler-
        // emitted addresses reached the interpreter dispatch. This is the
        // §4b1 seam — real attention would consume the same addresses.
        if let packet::Body::Flash { .. } = inst.body {
            let addresses: Vec<u64> = self
                .kv_pages_range
                .clone()
                .map(|slot| self.indirection.get(slot))
                .collect();
            self.kv_writes.push(KvWrite {
                packet_index: i as u32,
                addresses,
            });
        }

        if self.record {
            self.spans.push(TaskSpan {
                exec: inst.index as u32,
                task: i as u32,
                opcode: inst.body.opcode().0,
                t_start: t0,
                t_end: t1,
            });
        }
    }
}

/// The vocab width the bucket's SAMPLE packet declares (256 if none present).
pub(crate) fn sample_vocab(bucket: &Bucket) -> usize {
    bucket
        .program
        .insts
        .iter()
        .find_map(|i| match i.body {
            packet::Body::Token { vocab, .. } if vocab > 0 => Some(vocab as usize),
            _ => None,
        })
        .unwrap_or(256)
}

/// Does this bucket carry a `TOKEN_SAMPLE_BATCH` packet? When true the mux
/// fires the whole batch axis in one bucket walk; when false it falls back to
/// per-slot serial ticks against a scalar SAMPLE packet.
pub(crate) fn bucket_has_sample_batch(bucket: &Bucket) -> bool {
    bucket.program.insts.iter().any(|i| {
        matches!(
            i.body,
            packet::Body::Token { kind, .. } if kind == packet::Opcode::TOKEN_SAMPLE_BATCH
        )
    })
}

/// A deterministic `[0,1)` draw seeded by the request state (for stochastic
/// sampling in the reference path — reproducible, no wall-clock entropy).
pub(crate) fn seeded_unit(prompt: &[u32], out: &[u32], step: usize) -> f32 {
    (fnv_seed(prompt, out, step) % 10_000) as f32 / 10_000.0
}

fn fnv_seed(prompt: &[u32], out: &[u32], step: usize) -> u64 {
    let mut h = 1469598103934665603u64;
    for &t in prompt.iter().chain(out.iter()) {
        h ^= t as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h ^= step as u64;
    h.wrapping_mul(1099511628211)
}

/// Reference logits distribution: deterministic, request-seeded, peaked at a
/// printable ASCII letter. Stands in for real numerics+weights so the runtime
/// serving loop (bucket exec → gated host sample → detok) produces varied,
/// reproducible output. Replace with the arena's computed logits once golden
/// numerics + weight loading are wired.
pub(crate) fn reference_logits(prompt: &[u32], out: &[u32], vocab: usize, v: &mut Vec<f32>) {
    let vocab = vocab.max(1);
    v.clear();
    v.resize(vocab, -12.0f32);
    reference_logits_row(prompt, out, v);
}

/// Fill a caller-owned `vocab`-wide slice with the reference distribution for
/// `(prompt, out)`. Used by `step_batch` to build the `B×vocab` tile row by
/// row without allocation.
pub(crate) fn reference_logits_row(prompt: &[u32], out: &[u32], row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let vocab = row.len();
    let h = fnv_seed(prompt, out, out.len());
    for x in row.iter_mut() {
        *x = -12.0;
    }
    for (i, x) in row.iter_mut().enumerate() {
        let b = (i % 256) as u8;
        if b == b' ' || b.is_ascii_graphic() {
            *x = 0.4 + ((h.wrapping_add(i as u64) % 13) as f32) * 0.05;
        }
    }
    // A seeded peak on a lowercase letter.
    let peak = (b'a' as usize + (h as usize % 26)).min(vocab - 1);
    row[peak] += 4.0;
}

/// Shared server state. `Arc`-wrapped for axum handlers.
pub struct AppState {
    pub registry: Registry,
    pub execset: Arc<ExecutorSet>,
    pub metrics: Arc<Metrics>,
    /// Per-slug bucket muxer handles. Populated at startup by `main::serve`
    /// after the registry is loaded; read (Sender-clone) on the request path.
    muxes: RwLock<FxHashMap<String, mux::ModelMux>>,
    /// Per-slug GPU engines ([`engine::ServeEngine`] — sm_120 or gfx950).
    /// Installed at startup for bundles that ship a device blob; when present
    /// the mux drives real GPU decode steps instead of the CPU reference.
    #[cfg(any(feature = "cuda", feature = "hsa"))]
    gpu: RwLock<FxHashMap<String, Arc<Mutex<engine::ServeEngine>>>>,
    /// Engine-lock-free VMM stats readers for `/metrics` (one per slug with
    /// prefix sharing up) — a scrape must never queue behind a tick.
    #[cfg(feature = "cuda")]
    vmm_stats: RwLock<FxHashMap<String, crate::memory::vmm::VmmStatsHandle>>,
    /// The S1 multi-model manager (residency + VRAM planner). Installed once
    /// at startup when any bundle is GPU-managed; `None` on CPU-only serves.
    #[cfg(feature = "cuda")]
    manager: std::sync::OnceLock<Arc<manager::ModelManager>>,
    /// When set, each run records a timeline dumpable at `GET /trace`.
    record_trace: bool,
    trace: Mutex<Timeline>,
}

impl AppState {
    pub fn new(registry: Registry, execset: Arc<ExecutorSet>) -> Self {
        Self::with_trace(registry, execset, false)
    }

    /// Construct with per-run timeline recording enabled/disabled.
    pub fn with_trace(registry: Registry, execset: Arc<ExecutorSet>, record_trace: bool) -> Self {
        AppState {
            registry,
            execset,
            metrics: Arc::new(Metrics::default()),
            muxes: RwLock::new(FxHashMap::default()),
            #[cfg(any(feature = "cuda", feature = "hsa"))]
            gpu: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "cuda")]
            vmm_stats: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "cuda")]
            manager: std::sync::OnceLock::new(),
            record_trace,
            trace: Mutex::new(Timeline::new()),
        }
    }

    /// Register a GPU engine for a model slug. Called once at startup.
    #[cfg(any(feature = "cuda", feature = "hsa"))]
    pub fn install_gpu_engine(&self, slug: String, engine: engine::ServeEngine) {
        #[cfg(feature = "cuda")]
        if let Some(h) = engine.vmm_stats_handle() {
            self.vmm_stats.write().insert(slug.clone(), h);
        }
        self.gpu.write().insert(slug, Arc::new(Mutex::new(engine)));
    }

    /// The GPU engine serving `slug`, when one was installed.
    #[cfg(any(feature = "cuda", feature = "hsa"))]
    pub(crate) fn gpu_engine(&self, slug: &str) -> Option<Arc<Mutex<engine::ServeEngine>>> {
        self.gpu.read().get(slug).cloned()
    }

    /// Whether `slug` is served by a GPU engine (drives e.g. the chat-template
    /// choice). Always `false` without a vendor backend feature.
    pub fn has_gpu_engine(&self, slug: &str) -> bool {
        #[cfg(any(feature = "cuda", feature = "hsa"))]
        {
            return self.gpu.read().contains_key(slug);
        }
        #[cfg(not(any(feature = "cuda", feature = "hsa")))]
        {
            let _ = slug;
            false
        }
    }

    /// Remove a GPU engine (S1 eviction). The caller drops the returned `Arc`
    /// — the last drop is the model unload that returns the VRAM.
    #[cfg(feature = "cuda")]
    pub fn remove_gpu_engine(&self, slug: &str) -> Option<Arc<Mutex<engine::ServeEngine>>> {
        self.vmm_stats.write().remove(slug);
        self.gpu.write().remove(slug)
    }

    /// Install the S1 model manager (once, at startup).
    #[cfg(feature = "cuda")]
    pub fn install_manager(&self, m: Arc<manager::ModelManager>) {
        let _ = self.manager.set(m);
    }

    /// The S1 model manager, when multi-model GPU serving is active.
    #[cfg(feature = "cuda")]
    pub fn manager(&self) -> Option<&Arc<manager::ModelManager>> {
        self.manager.get()
    }

    /// Register a dispatcher for a model slug. Called once at startup.
    pub fn install_mux(&self, slug: String, m: mux::ModelMux) {
        self.muxes.write().insert(slug, m);
    }

    /// Remove a dispatcher (S1 eviction) — new lookups fail fast; the caller
    /// drains the returned handle before dropping the engine.
    pub fn remove_mux(&self, slug: &str) -> Option<mux::ModelMux> {
        self.muxes.write().remove(slug)
    }

    /// Look up the dispatcher for a slug (clones the underlying Sender).
    pub fn mux(&self, slug: &str) -> Option<mux::ModelMux> {
        self.muxes.read().get(slug).cloned()
    }

    /// Run one request end-to-end: tokenize the prompt, then autoregressively
    /// decode `gen.max_tokens` tokens, each produced by running the decode
    /// bucket's schedule and letting the **gated host `SAMPLE` packet** turn the
    /// logits into a token id on the [`HostExecutor`] (or, if the bucket carries
    /// no sample packet, sampling directly). Real runtime path: bucket select →
    /// counter-gated execution → host sample → detokenize → stream.
    ///
    /// Logits come from the arena's logits buffer; with no checkpoint loaded the
    /// CPU golden numerics are the documented seam, so [`reference_logits`]
    /// supplies a deterministic, request-seeded distribution — the *mechanics*
    /// (gating, host sampling, detok) are real; only the logit values stand in
    /// until weights + numerics are wired.
    pub fn generate(&self, model: &str, prompt: &str, gen: &GenParams) -> Result<(String, usize)> {
        self.generate_with_bucket(model, prompt, gen, None)
    }

    /// Same as [`generate`], but with an optional bucket override chosen by the
    /// muxer (which sees the joined batch, not this single request).
    pub fn generate_with_bucket(
        &self,
        model: &str,
        prompt: &str,
        gen: &GenParams,
        bucket_override: Option<BucketKey>,
    ) -> Result<(String, usize)> {
        Metrics::inc(&self.metrics.requests);
        let bundle = self.registry.get(model)?;
        // The model's own tokenizer (real HF `tokenizer.json` when present).
        let tok = bundle.tokenizer();
        let prompt_ids = tok.encode(prompt);

        let scheduler = Scheduler::new(&self.execset);
        let seq = prompt_ids.len().max(1) as i64;
        let key = bucket_override.or_else(|| {
            batching::select_bucket(&bundle, Phase::Decode, 1, seq)
                .or_else(|| bundle.bucket_keys().find(|k| k.phase == Phase::Decode))
                .or_else(|| bundle.bucket_keys().next())
        });
        let bucket = key.and_then(|k| bundle.bucket(k));
        let vocab = bucket.map(sample_vocab).unwrap_or(256);

        // Per-token hot-loop state, built once per request: the observer (host
        // executor + span buffer), sampling params, the counter pool (reset by
        // `run_reference_traced_reuse` each token), and the program's stream
        // bucketing. The loop below performs no per-token allocation.
        let mut obs = RunObserver::new(self.record_trace, RUN_INDIRECTION_SIZE);
        obs.host.params = gen.params.clone();
        let mut run = bucket.map(|b| {
            let pool = scheduler.pool_for(b);
            let streams = crate::device::cpu::StreamSet::new(&b.program, pool.len());
            (b, pool, streams)
        });

        let mut out_ids: Vec<u32> = Vec::new();
        let mut executed_total = 0usize;

        for step in 0..gen.max_tokens.max(1) {
            obs.host.tokens.clear();
            obs.host.rng01 = seeded_unit(&prompt_ids, &out_ids, step);
            reference_logits(&prompt_ids, &out_ids, vocab, &mut obs.host.logits);

            let token = if let Some((bucket, pool, streams)) = run.as_mut() {
                let stats = self.execset.run_reference_traced_reuse(
                    &bucket.program,
                    pool,
                    &mut obs,
                    streams,
                );
                if !stats.completed {
                    return Err(RuntimeError::Deadlock(format!(
                        "bucket did not complete: {}/{} fired",
                        stats.executed, stats.total
                    )));
                }
                executed_total += stats.executed;
                if self.record_trace {
                    let mut tl = self.trace.lock();
                    for span in obs.spans.drain(..) {
                        tl.push(span);
                    }
                }
                // The gated SAMPLE packet wrote a token; else sample directly.
                obs.host.tokens.last().copied().unwrap_or_else(|| {
                    sample::sample(&obs.host.logits, &obs.host.params, None, obs.host.rng01)
                })
            } else {
                sample::sample(&obs.host.logits, &obs.host.params, None, obs.host.rng01)
            };

            out_ids.push(token);
            // Stop on a newline byte (a simple, deterministic stop condition).
            if token % 256 == u32::from(b'\n') {
                break;
            }
        }

        Ok((tok.decode(&out_ids), executed_total))
    }

    /// Chrome-trace JSON accumulated from traced runs (empty if `--trace` off).
    pub fn trace_json(&self) -> String {
        self.trace.lock().to_chrome_json()
    }

    /// One batched decode step: fire the bucket **once** for every live slot.
    /// The mux prepares `obs.host` before calling (row-major `B×vocab`
    /// logits, per-row params/rng, `slot_tokens.resize(B, 0)`); on return the
    /// row-`b` produced token is at `obs.host.slot_tokens[b]`. Requires the
    /// bucket to carry a `TOKEN_SAMPLE_BATCH` packet — the fallback to per-slot
    /// [`step_token`] is the caller's responsibility.
    pub(crate) fn step_batch(
        &self,
        bucket: &Bucket,
        pool: &crate::exec::counters::CounterPool,
        streams: &mut crate::device::cpu::StreamSet,
        obs: &mut RunObserver,
    ) -> Result<usize> {
        let stats = self
            .execset
            .run_reference_traced_reuse(&bucket.program, pool, obs, streams);
        if !stats.completed {
            return Err(RuntimeError::Deadlock(format!(
                "bucket did not complete: {}/{} fired",
                stats.executed, stats.total
            )));
        }
        if self.record_trace {
            let mut tl = self.trace.lock();
            for span in obs.spans.drain(..) {
                tl.push(span);
            }
        }
        Ok(stats.executed)
    }

    /// One decode step for a single slot: fill reference logits, run the
    /// counter-gated bucket walk once, return the sampled token. The mux
    /// engine calls this per live slot per tick; caller-owned buffers
    /// (observer, counter pool, stream set) keep the hot path allocation-free.
    pub(crate) fn step_token(
        &self,
        bucket: &Bucket,
        pool: &crate::exec::counters::CounterPool,
        streams: &mut crate::device::cpu::StreamSet,
        obs: &mut RunObserver,
        prompt_ids: &[u32],
        out_ids: &[u32],
        step: usize,
        vocab: usize,
    ) -> Result<(u32, usize)> {
        obs.host.tokens.clear();
        obs.host.rng01 = seeded_unit(prompt_ids, out_ids, step);
        reference_logits(prompt_ids, out_ids, vocab, &mut obs.host.logits);

        let stats = self
            .execset
            .run_reference_traced_reuse(&bucket.program, pool, obs, streams);
        if !stats.completed {
            return Err(RuntimeError::Deadlock(format!(
                "bucket did not complete: {}/{} fired",
                stats.executed, stats.total
            )));
        }
        if self.record_trace {
            let mut tl = self.trace.lock();
            for span in obs.spans.drain(..) {
                tl.push(span);
            }
        }
        let token = obs.host.tokens.last().copied().unwrap_or_else(|| {
            sample::sample(&obs.host.logits, &obs.host.params, None, obs.host.rng01)
        });
        Ok((token, stats.executed))
    }
}

/// Build the axum app.
pub fn app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/models", get(models::list_models))
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics_handler))
        .route("/trace", get(trace_handler))
        .with_state(state)
}

/// `GET /trace` — Chrome-trace JSON from traced live runs (§O, `--trace`).
async fn trace_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    ([("content-type", "application/json")], state.trace_json()).into_response()
}

async fn healthz() -> &'static str {
    "ok"
}

async fn metrics_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> String {
    #[allow(unused_mut)]
    let mut out = state.metrics.to_prometheus();
    // Prefix-cache (VMM) counters, one block per GPU-served model, read
    // through the engine-lock-free stats handles — series stay continuous
    // under sustained inference (only the pool mutex is taken, µs holds).
    #[cfg(feature = "cuda")]
    for (slug, h) in state.vmm_stats.read().iter() {
        let s = h.stats();
        use std::fmt::Write;
        let _ = write!(
            out,
            "plowrt_prefix_attach_hits_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_attach_misses_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_tokens_attached_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_hash_collisions_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_blocks_shared_mapped_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_nodes_evicted_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_blocks_live{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_cache_blocks{{model=\"{slug}\"}} {}\n",
            s.attach_hits,
            s.attach_misses,
            s.tokens_attached,
            s.hash_collisions,
            s.blocks_shared_mapped,
            s.nodes_evicted,
            s.blocks_live,
            s.cache_blocks,
        );
    }
    out
}

/// Map a runtime error to an HTTP status. A fatal device fault means the
/// device context is dead — 503 (retry another instance), not a 500 that
/// reads as a plowrt bug; a non-fatal fault stays a 500.
pub(crate) fn status_for(err: &RuntimeError) -> axum::http::StatusCode {
    use axum::http::StatusCode;
    match err {
        RuntimeError::UnknownModel(_) => StatusCode::NOT_FOUND,
        RuntimeError::Rejected(_) | RuntimeError::Oom(_) => StatusCode::TOO_MANY_REQUESTS,
        RuntimeError::DeviceFault { info } if info.fatal => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}
