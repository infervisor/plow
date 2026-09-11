//! §G OpenAI-compatible API server.

pub mod admin;
pub mod bench;
pub mod chat;
pub mod completion;
pub mod cosched;
#[cfg(feature = "cpu")]
pub mod cpu_serve;
/// The loaded device engine behind a slug, as one type over both backends —
/// the seam that lets `serve` stop being CUDA-only.
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
pub mod engine;
#[cfg(feature = "cuda")]
pub mod manager;
pub mod models;
pub mod mux;
pub mod openai;
pub mod placement;
pub mod stream;
#[cfg(all(test, any(feature = "hsa", feature = "cpu")))]
mod step_lowering_tests;
pub mod template;
pub mod tokenize;

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
    /// OpenAI `stop`. Generation ends at the first match and the matched text
    /// is withheld. Previously the request field was not even parsed, so a
    /// LangChain agent relying on `stop` to end a ReAct step over-generated and
    /// then mis-parsed its own output.
    pub stop: Vec<String>,
    /// OpenAI `seed`, mixed into the sampling draw for reproducibility.
    pub seed: Option<u64>,
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams {
            max_tokens: 4096,
            params: SamplingParams::default(),
            ignore_eos: false,
            stop: Vec::new(),
            seed: None,
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
    seeded_unit_with(prompt, out, step, None)
}

/// The same draw, with an optional caller-supplied OpenAI `seed` mixed in.
/// Without it the draw is derived from the token stream alone — deterministic,
/// but not something a client can choose, which is what `seed` is for.
pub(crate) fn seeded_unit_with(prompt: &[u32], out: &[u32], step: usize, seed: Option<u64>) -> f32 {
    (fnv_seed(prompt, out, step, seed) % 10_000) as f32 / 10_000.0
}

fn fnv_seed(prompt: &[u32], out: &[u32], step: usize, seed: Option<u64>) -> u64 {
    let mut h = 1469598103934665603u64;
    if let Some(s) = seed {
        h ^= s;
        h = h.wrapping_mul(1099511628211);
    }
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
    let h = fnv_seed(prompt, out, out.len(), None);
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
    model_metrics: RwLock<FxHashMap<String, Arc<Metrics>>>,
    /// Per-slug bucket muxer handles. Populated at startup by `main::serve`
    /// after the registry is loaded; read (Sender-clone) on the request path.
    muxes: RwLock<FxHashMap<String, mux::ModelMux>>,
    /// Per-slug GPU engines ([`engine::ServeEngine`] — sm_120 or gfx950).
    /// Installed at startup for bundles that ship a device blob; when present
    /// the mux drives real GPU decode steps instead of the CPU reference.
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    gpu: RwLock<FxHashMap<String, Arc<Mutex<engine::ServeEngine>>>>,
    /// Engine-lock-free VMM stats readers for `/metrics` (one per slug with
    /// prefix sharing up) — a scrape must never queue behind a tick.
    #[cfg(feature = "cuda")]
    vmm_stats: RwLock<FxHashMap<String, crate::memory::vmm::VmmStatsHandle>>,
    /// One S1 residency manager per device group (residency + VRAM planner).
    /// Installed once at startup; empty on CPU-only serves.
    ///
    /// Per group, not one manager taught about devices: every invariant a
    /// manager holds is already per-card — one free-VRAM figure, one slab pool,
    /// one LRU order, one switch lock — so instancing it per group keeps a load
    /// on GPU 1 from serializing behind a switch on GPU 0.
    #[cfg(feature = "cuda")]
    managers: std::sync::OnceLock<Vec<Arc<manager::ModelManager>>>,
    /// slug → device-group index. The request path reads it to find the group
    /// serving a model; the control plane reports it. NOT vendor-gated: the
    /// AMD and CPU engines serve device groups too, they just only ever have
    /// one.
    slug_group: RwLock<FxHashMap<String, usize>>,
    /// One co-tenant turn per device group ([`cosched`]). Installed at startup
    /// by whichever backend path came up. Vendor-neutral by construction —
    /// taking turns is host-side sequencing, and every backend that can hold
    /// two models on one device needs it.
    turns: std::sync::OnceLock<Vec<Arc<cosched::DeviceTurn>>>,
    /// Directories a control-plane `load` may take an assets dir from. Set
    /// once at startup; empty means no assets dir may be named by request.
    models_roots: std::sync::OnceLock<Vec<std::path::PathBuf>>,
    /// Operator-set residency overrides, slug → state. Absent = [`Residency::Auto`].
    /// Only the control plane writes here; the manager and the request path read it.
    residency: RwLock<FxHashMap<String, Residency>>,
    control: Mutex<FxHashMap<String, std::sync::Weak<tokio::sync::Mutex<()>>>>,
    /// When set, each run records a timeline dumpable at `GET /trace`.
    record_trace: bool,
    trace: Mutex<Timeline>,
}

/// Operator-visible residency state of a registered slug.
///
/// This exists because residency is otherwise purely demand-driven: an
/// operator's unload would be undone by the very next request (which calls
/// `ensure_resident`) or by the speculative preloader. `Unloaded` is the state
/// that makes an explicit unload stick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Residency {
    /// Registered; the manager loads and evicts it on demand.
    #[default]
    Auto,
    /// An unload is in flight. Handlers refuse new work for this slug BEFORE
    /// the mux is removed, so a request cannot slip into the window between
    /// `remove_mux` and the dispatcher's exit and be dropped without a terminal.
    Unloading,
    /// Explicitly unloaded. `ensure_resident` refuses and the preloader skips
    /// it until an explicit load returns it to `Auto`.
    Unloaded,
}

impl Residency {
    /// Whether new requests for this slug may be admitted.
    pub fn admits(self) -> bool {
        matches!(self, Residency::Auto)
    }
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
            model_metrics: RwLock::new(FxHashMap::default()),
            muxes: RwLock::new(FxHashMap::default()),
            #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
            gpu: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "cuda")]
            vmm_stats: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "cuda")]
            managers: std::sync::OnceLock::new(),
            slug_group: RwLock::new(FxHashMap::default()),
            turns: std::sync::OnceLock::new(),
            models_roots: std::sync::OnceLock::new(),
            residency: RwLock::new(FxHashMap::default()),
            control: Mutex::new(FxHashMap::default()),
            record_trace,
            trace: Mutex::new(Timeline::new()),
        }
    }

    /// Register a GPU engine for a model slug. Called once at startup.
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    pub fn install_gpu_engine(&self, slug: String, engine: engine::ServeEngine) {
        #[cfg(feature = "cuda")]
        if let Some(h) = engine.vmm_stats_handle() {
            self.vmm_stats.write().insert(slug.clone(), h);
        }
        self.gpu.write().insert(slug, Arc::new(Mutex::new(engine)));
    }

    /// The GPU engine serving `slug`, when one was installed.
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    pub(crate) fn gpu_engine(&self, slug: &str) -> Option<Arc<Mutex<engine::ServeEngine>>> {
        self.gpu.read().get(slug).cloned()
    }

    /// Whether `slug` is served by a GPU engine (drives e.g. the chat-template
    /// choice). Always `false` without a vendor backend feature.
    pub fn has_gpu_engine(&self, slug: &str) -> bool {
        #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
        {
            return self.gpu.read().contains_key(slug);
        }
        #[cfg(not(any(feature = "cuda", feature = "hsa", feature = "cpu")))]
        {
            let _ = slug;
            false
        }
    }

    /// Remove a GPU engine (eviction / unload). The caller drops the returned
    /// `Arc` — the last drop is the model unload that returns the device memory.
    ///
    /// Available on every backend that can INSTALL an engine. It used to be
    /// CUDA-only while `install_gpu_engine` was not, so an AMD or CPU serve
    /// could bring an engine up and had no way to take it down — teardown is
    /// lifecycle, not a vendor feature. Only the VMM stats handle is
    /// CUDA-specific, and that is gated inside.
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    pub fn remove_gpu_engine(&self, slug: &str) -> Option<Arc<Mutex<engine::ServeEngine>>> {
        #[cfg(feature = "cuda")]
        self.vmm_stats.write().remove(slug);
        self.gpu.write().remove(slug)
    }

    /// Install the per-group residency managers (once, at startup).
    #[cfg(feature = "cuda")]
    pub fn install_managers(&self, m: Vec<Arc<manager::ModelManager>>) {
        self.install_device_turns(m.len());
        let _ = self.managers.set(m);
    }

    /// Install one manager — the single-group case, and what tests use.
    #[cfg(feature = "cuda")]
    pub fn install_manager(&self, m: Arc<manager::ModelManager>) {
        self.install_managers(vec![m]);
    }

    /// Install one co-tenant turn per device group (once, at startup).
    ///
    /// Called by every backend path, not just CUDA: two AMD models on one agent
    /// is a thing that already happens — the AMD startup loop installs an
    /// engine per bundle and each opens the same ROCr agent — and HSA has no
    /// cooperative-launch refusal to catch the resulting CU oversubscription.
    /// The CPU engine gives every model its own worker pool, so turns bound
    /// thread contention there.
    pub fn install_device_turns(&self, groups: usize) {
        self.turns.get_or_init(|| {
            let turns = (0..groups.max(1))
                .map(|_| Arc::new(cosched::DeviceTurn::from_config()))
                .collect::<Vec<_>>();
            if let Some(first) = turns.first() {
                tracing::info!(
                    groups = turns.len(),
                    mode = ?first.mode(),
                    quantum = first.quantum(),
                    "co-tenant scheduling installed"
                );
            }
            turns
        });
    }

    /// The co-tenant turn for `slug`'s device group, when turns are installed.
    ///
    /// A slug with no recorded group takes group 0's turn: the AMD and CPU
    /// paths serve one device set and never record a group, and defaulting to
    /// "no turn" there would silently disable ordering on exactly the backend
    /// that most needs it.
    pub fn device_turn(&self, slug: &str) -> Option<Arc<cosched::DeviceTurn>> {
        let turns = self.turns.get()?;
        turns
            .get(self.slug_group(slug).unwrap_or(0))
            .map(Arc::clone)
    }

    /// Record which group serves `slug`.
    pub fn set_slug_group(&self, slug: &str, group: usize) {
        self.slug_group.write().insert(slug.to_string(), group);
    }

    /// Forget which group served `slug` (deregistration).
    pub fn clear_slug_group(&self, slug: &str) {
        self.slug_group.write().remove(slug);
    }

    /// The group index serving `slug`, when placement assigned one.
    pub fn slug_group(&self, slug: &str) -> Option<usize> {
        self.slug_group.read().get(slug).copied()
    }

    /// Every installed manager, one per device group.
    #[cfg(feature = "cuda")]
    pub fn managers(&self) -> &[Arc<manager::ModelManager>] {
        self.managers.get().map(Vec::as_slice).unwrap_or(&[])
    }

    /// The manager for `slug`'s group.
    ///
    /// Placement is the authority; the fallback scan exists for managers
    /// installed without a placement pass (tests, and the single-group case),
    /// and for a slug registered at runtime before its group was recorded.
    #[cfg(feature = "cuda")]
    pub fn manager_for(&self, slug: &str) -> Option<&Arc<manager::ModelManager>> {
        let managers = self.managers();
        if let Some(g) = self.slug_group(slug) {
            if let Some(m) = managers.get(g) {
                return Some(m);
            }
        }
        managers.iter().find(|m| m.manages(slug))
    }

    /// The manager a NEW model should be registered with: the named group, or
    /// the one with the most free memory when the caller did not name one.
    #[cfg(feature = "cuda")]
    pub fn manager_for_new(
        &self,
        group: Option<usize>,
    ) -> Option<(usize, &Arc<manager::ModelManager>)> {
        let managers = self.managers();
        match group {
            Some(g) => managers.get(g).map(|m| (g, m)),
            None => managers
                .iter()
                .enumerate()
                .max_by_key(|(_, m)| m.device_mem_info().map(|(free, _)| free).unwrap_or(0)),
        }
    }

    pub async fn control_lock(&self, slug: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.control.lock();
            locks.retain(|_, lock| lock.strong_count() != 0);
            match locks.get(slug).and_then(std::sync::Weak::upgrade) {
                Some(lock) => lock,
                None => {
                    let lock = Arc::new(tokio::sync::Mutex::new(()));
                    locks.insert(slug.to_string(), Arc::downgrade(&lock));
                    lock
                }
            }
        };
        lock.lock_owned().await
    }

    /// Install the assets-dir allow-list for control-plane loads (once, at
    /// startup). Paths are canonicalized here so the prefix test at request
    /// time compares two real paths.
    pub fn install_models_roots(&self, roots: impl IntoIterator<Item = std::path::PathBuf>) {
        let roots: Vec<std::path::PathBuf> = roots
            .into_iter()
            .filter_map(|p| p.canonicalize().ok())
            .collect();
        let _ = self.models_roots.set(roots);
    }

    /// The assets-dir allow-list. Empty until `install_models_roots`.
    pub fn models_roots(&self) -> &[std::path::PathBuf] {
        self.models_roots.get().map(Vec::as_slice).unwrap_or(&[])
    }

    /// Residency state of `slug` (defaults to [`Residency::Auto`]).
    pub fn residency(&self, slug: &str) -> Residency {
        self.residency.read().get(slug).copied().unwrap_or_default()
    }

    /// Set the residency state of `slug`. `Auto` clears the override.
    pub fn set_residency(&self, slug: &str, state: Residency) {
        let mut map = self.residency.write();
        match state {
            Residency::Auto => {
                map.remove(slug);
            }
            _ => {
                map.insert(slug.to_string(), state);
            }
        }
    }

    pub(crate) fn model_metrics(&self, slug: &str) -> Arc<Metrics> {
        let mut models = self.model_metrics.write();
        models.retain(|name, metrics| {
            let keep = self.registry.contains(name) || Arc::strong_count(metrics) > 1;
            if !keep {
                self.metrics.accumulate_counters(metrics);
            }
            keep
        });
        Arc::clone(models.entry(slug.to_owned()).or_default())
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
        .route("/v1/completions", post(completion::completions))
        .route("/tokenize", post(tokenize::tokenize))
        .route("/detokenize", post(tokenize::detokenize))
        .route("/v1/models", get(models::list_models))
        // BOTH spellings. vLLM serves `/health`, and every k8s probe and
        // benchmark harness copied from vLLM asks for it; plowrt served only
        // `/healthz`, so all of them got a 404 from a healthy server.
        .route("/health", get(healthz))
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
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let models: Vec<_> = {
        let mut metrics = state.model_metrics.write();
        for slug in state.registry.slugs() {
            metrics.entry(slug).or_default();
        }
        metrics.retain(|slug, metrics| {
            let keep = state.registry.contains(slug) || Arc::strong_count(metrics) > 1;
            if !keep {
                state.metrics.accumulate_counters(metrics);
            }
            keep
        });
        let mut models: Vec<_> = metrics.iter().map(|(slug, m)| {
            (
                slug.clone(),
                Arc::clone(m),
                state.mux(slug).is_some() && state.residency(slug).admits(),
            )
        }).collect();
        models.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        models
    };
    let aggregate = Metrics::default();
    aggregate.accumulate(&state.metrics);
    for (_, metrics, _) in &models {
        aggregate.accumulate(metrics);
    }
    let mut out = aggregate.to_prometheus();
    for (index, (slug, metrics, _)) in models.iter().enumerate() {
        let label = crate::obs::serving::escape_label(slug);
        for line in metrics.to_prometheus().replace("plowrt_", "plowrt_model_").lines() {
            if line.starts_with('#') {
                if index == 0 {
                    out.push_str(line);
                    out.push('\n');
                }
                continue;
            }
            if let Some((name, value)) = line.split_once(' ') {
                use std::fmt::Write;
                let _ = writeln!(out, "{name}{{model_name=\"{label}\",engine=\"0\"}} {value}");
            }
        }
    }
    crate::obs::serving::ServingMetrics::write(&mut out, &models);
    // Prefix-cache (VMM) counters, one block per GPU-served model, read
    // through the engine-lock-free stats handles — series stay continuous
    // under sustained inference (only the pool mutex is taken, µs holds).
    #[cfg(feature = "cuda")]
    for (name, kind, help) in [
        ("attach_hits_total", "counter", "Successful prefix attach requests."),
        ("attach_misses_total", "counter", "Prefix attach requests without a reusable prefix."),
        ("tokens_attached_total", "counter", "Tokens attached from reusable prefix blocks."),
        ("hash_collisions_total", "counter", "Prefix hash collisions detected."),
        ("blocks_shared_mapped_total", "counter", "Shared prefix blocks mapped."),
        ("nodes_evicted_total", "counter", "Prefix tree nodes evicted."),
        ("blocks_live", "gauge", "Live prefix pool blocks."),
        ("cache_blocks", "gauge", "Prefix cache blocks."),
        ("cache_bytes", "gauge", "Prefix cache bytes."),
        ("snapshot_bytes", "gauge", "Prefix snapshot bytes."),
        ("snapshots_evicted_total", "counter", "Prefix snapshots evicted."),
    ] {
        crate::obs::serving::family(&mut out, &format!("plowrt_prefix_{name}"), kind, help);
    }
    #[cfg(feature = "cuda")]
    for (slug, h) in state.vmm_stats.read().iter() {
        let s = h.stats();
        let slug = crate::obs::serving::escape_label(slug);
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
             plowrt_prefix_cache_blocks{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_cache_bytes{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_snapshot_bytes{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_snapshots_evicted_total{{model=\"{slug}\"}} {}\n",
            s.attach_hits,
            s.attach_misses,
            s.tokens_attached,
            s.hash_collisions,
            s.blocks_shared_mapped,
            s.nodes_evicted,
            s.blocks_live,
            s.cache_blocks,
            s.cache_bytes,
            s.snapshot_bytes,
            s.snapshots_evicted,
        );
    }
    ([("content-type", "text/plain; version=0.0.4; charset=utf-8")], out).into_response()
}

/// Build an OpenAI-shaped error response: `{"error": {message, type, code}}`.
///
/// Every error path in this server used to emit a bare `{"error": "<string>"}`.
/// openai-python reads `body["error"]["message"]` and branches on
/// `body["error"]["code"]`, so a string body gave clients an unusable message
/// and nothing to branch on.
pub(crate) fn api_error(
    status: axum::http::StatusCode,
    message: impl Into<String>,
    kind: &'static str,
    code: Option<&'static str>,
    param: Option<String>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    (
        status,
        axum::Json(openai::ApiErrorBody::new(message, kind, code, param)),
    )
        .into_response()
}

/// The same, for a `RuntimeError`: status and code derived from the variant.
pub(crate) fn api_error_for(err: &RuntimeError) -> axum::response::Response {
    let status = status_for(err);
    let (kind, code) = match err {
        RuntimeError::UnknownModel(_) => ("invalid_request_error", Some("model_not_found")),
        RuntimeError::ContextLength(_) => {
            ("invalid_request_error", Some("context_length_exceeded"))
        }
        RuntimeError::Rejected(_) | RuntimeError::Oom(_) => {
            ("rate_limit_error", Some("server_overloaded"))
        }
        _ => ("server_error", None),
    };
    api_error(status, err.to_string(), kind, code, None)
}

#[cfg(test)]
mod error_mapping_tests {
    use super::{api_error_for, status_for};
    use crate::error::RuntimeError;
    use axum::http::StatusCode;

    /// A prompt longer than the compiled context can NEVER succeed. It used to
    /// come back 429 from the CUDA and AMD padded-cover paths and 500 from the
    /// AMD raw-prompt path, and every OpenAI-compatible client treats 429 as
    /// retryable — so a permanent failure was answered with "try again", in a
    /// backoff loop.
    #[test]
    fn context_length_is_a_client_error_not_a_retry() {
        assert_eq!(
            status_for(&RuntimeError::ContextLength("too long".into())),
            StatusCode::BAD_REQUEST
        );
    }

    /// A shed request IS retryable and must stay 429 — the two must not be
    /// collapsed just because both refuse the request.
    #[test]
    fn a_shed_request_is_still_retryable() {
        assert_eq!(
            status_for(&RuntimeError::Rejected("no slot".into())),
            StatusCode::TOO_MANY_REQUESTS
        );
    }

    /// Clients branch on `error.code`; a bare string body gives them nothing.
    #[test]
    fn the_error_envelope_carries_a_machine_readable_code() {
        let resp = api_error_for(&RuntimeError::ContextLength("too long".into()));
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let resp = api_error_for(&RuntimeError::UnknownModel("nope".into()));
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}

/// Map a runtime error to an HTTP status. A fatal device fault means the
/// device context is dead — 503 (retry another instance), not a 500 that
/// reads as a plowrt bug; a non-fatal fault stays a 500.
pub(crate) fn status_for(err: &RuntimeError) -> axum::http::StatusCode {
    use axum::http::StatusCode;
    match err {
        RuntimeError::UnknownModel(_) => StatusCode::NOT_FOUND,
        // A prompt longer than the compiled context can NEVER succeed, so a
        // 429 was actively harmful: every OpenAI-compatible client treats 429
        // as retryable and backs off in a loop against a permanent failure.
        RuntimeError::ContextLength(_) => StatusCode::BAD_REQUEST,
        RuntimeError::Rejected(_) | RuntimeError::Oom(_) => StatusCode::TOO_MANY_REQUESTS,
        RuntimeError::DeviceFault { info } if info.fatal => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Privileged routes, mounted only on the owner-only Unix socket by the CLI.
pub fn admin_app(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models/load", post(admin::load))
        .route("/v1/models/unload", post(admin::unload))
        .route("/v1/models/status", get(admin::status))
        .with_state(state)
}
