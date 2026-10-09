//! §G OpenAI-compatible API server.

pub mod admin;
pub mod auth;
#[cfg(feature = "cpu")]
pub mod portable;
pub mod bench;
pub mod chat;
pub mod completion;
pub mod config;
pub mod cosched;
pub mod deadlines;
pub mod dp;
#[cfg(feature = "cpu")]
pub mod cpu_serve;
/// The loaded device engine behind a slug, as one type over both backends —
/// the seam that lets `serve` stop being CUDA-only.
#[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
pub mod engine;
/// The CPU prefill-head pool: the twin packet and the cores reserved for it.
// Allowed dead until the mux drives it.
#[cfg(feature = "cpu")]
#[allow(dead_code)]
pub mod head;
pub mod policy;
#[cfg(feature = "cuda")]
pub mod manager;
pub mod logprobs;
pub mod models;
pub mod mux;
pub mod openai;
pub mod overload;
pub mod placement;
pub mod reasoning;
pub mod stream;
pub mod session;
#[cfg(all(test, any(feature = "hsa", feature = "cpu")))]
mod step_lowering_tests;
pub mod template;
pub mod tokenize;
pub mod turns;

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
    /// vLLM `min_tokens`: neither the eos set nor a `stop` string may end this
    /// request before it has produced this many tokens.
    pub min_tokens: usize,
    /// vLLM `stop_token_ids`: request-supplied ids that end generation, on top
    /// of the checkpoint's own eos set.
    pub stop_token_ids: Vec<u32>,
}

/// The generation budget of a request that named none: the default, capped to the context the
/// prompt leaves (as vLLM does), so a long prompt is served instead of refused for a budget the
/// client never asked for.
pub(crate) fn default_max_tokens(default: usize, max_ctx: Option<usize>, n_prompt: usize) -> usize {
    match max_ctx {
        Some(ctx) if n_prompt < ctx => default.min(ctx - n_prompt),
        _ => default,
    }
}

/// Refuse up front a request whose prompt plus budget cannot fit the context. The CUDA engine
/// refuses it at seating; the AMD batch only noticed when the slot's position ran past the
/// context, and that failed every request in the batch.
pub(crate) fn context_overflow(max_ctx: Option<usize>, n_prompt: usize, max_tokens: usize) -> Option<RuntimeError> {
    let ctx = max_ctx?;
    let total = n_prompt.saturating_add(max_tokens.max(1));
    (total > ctx).then(|| {
        RuntimeError::ContextLength(format!(
            "prompt ({n_prompt} tokens) + max_tokens ({max_tokens}) = {total} exceeds the context {ctx}"
        ))
    })
}

/// Refuse, before encoding, a prompt too long for the context under any tokenization: it has more
/// than `max_ctx` tokens once no token can cover all `max_token_bytes` of its share.
pub(crate) fn prompt_bytes_overflow(max_ctx: Option<usize>, max_token_bytes: usize, bytes: usize) -> Option<RuntimeError> {
    let ctx = max_ctx?;
    (bytes / max_token_bytes.max(1) > ctx).then(|| {
        RuntimeError::ContextLength(format!("prompt ({bytes} bytes) has more tokens than the context {ctx}"))
    })
}

/// Prompts at least this long encode with [`tokio::task::block_in_place`]: a 64 KiB prompt already
/// takes tens of ms, and a 64 MiB one stalled every connection on its worker, `/health` included.
const BLOCKING_ENCODE_BYTES: usize = 64 * 1024;

pub(crate) fn encode_prompt(text: &str, encode: impl FnOnce(&str) -> Vec<u32>) -> Vec<u32> {
    let multi_thread = || {
        tokio::runtime::Handle::try_current()
            .is_ok_and(|h| h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread)
    };
    if text.len() >= BLOCKING_ENCODE_BYTES && multi_thread() {
        tokio::task::block_in_place(|| encode(text))
    } else {
        encode(text)
    }
}

impl Default for GenParams {
    fn default() -> Self {
        GenParams {
            max_tokens: 4096,
            params: SamplingParams::default(),
            ignore_eos: false,
            stop: Vec::new(),
            seed: None,
            min_tokens: 0,
            stop_token_ids: Vec::new(),
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

    #[test]
    fn default_budget_fits_the_context_the_prompt_leaves() {
        use super::default_max_tokens;
        assert_eq!(default_max_tokens(4096, Some(16384), 14000), 2384);
        assert_eq!(default_max_tokens(4096, Some(16384), 100), 4096);
        // At or past the context the prompt is refused as too long, not given a zero budget.
        assert_eq!(default_max_tokens(4096, Some(16384), 16384), 4096);
        assert_eq!(default_max_tokens(4096, None, 14000), 4096);
    }

    #[test]
    fn prompt_plus_budget_past_the_context_is_a_context_error() {
        use super::context_overflow;
        assert!(context_overflow(Some(16384), 14000, 2384).is_none());
        assert!(matches!(context_overflow(Some(16384), 14000, 2385), Some(RuntimeError::ContextLength(_))));
        assert!(matches!(context_overflow(Some(16384), 16384, 4096), Some(RuntimeError::ContextLength(_))));
        assert!(context_overflow(None, 1 << 30, 1 << 30).is_none());
    }

    #[test]
    fn a_prompt_longer_than_any_tokenization_fits_is_refused_unencoded() {
        use super::prompt_bytes_overflow;
        let byte = crate::text::tokenizer::ByteTokenizer;
        let tok: &dyn crate::text::tokenizer::Tokenize = &byte;
        assert!(prompt_bytes_overflow(Some(8), tok.max_token_bytes(), 8).is_none());
        assert!(matches!(prompt_bytes_overflow(Some(8), tok.max_token_bytes(), 9), Some(RuntimeError::ContextLength(_))));
        assert!(prompt_bytes_overflow(Some(8), 400, 8 * 400 + 399).is_none());
        assert!(prompt_bytes_overflow(Some(8), 400, 9 * 400).is_some());
        assert!(prompt_bytes_overflow(Some(8), usize::MAX, usize::MAX).is_none());
        assert!(prompt_bytes_overflow(None, 1, usize::MAX).is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_long_prompt_encodes_off_the_worker_with_the_same_ids() {
        let long = "a".repeat(super::BLOCKING_ENCODE_BYTES + 1);
        let tok = crate::text::tokenizer::ByteTokenizer;
        use crate::text::tokenizer::Tokenize;
        assert_eq!(super::encode_prompt(&long, |t| tok.encode(t)).len(), long.len());
        assert_eq!(super::encode_prompt("hi", |t| tok.encode(t)), vec![104, 105]);
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
    /// SLO tick planner state (`PLOW_TBT_SLO_MS` / `PLOW_TTFT_SLO_MS`); untouched when unset.
    pub slo: crate::sched::slo::SloState,
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
            slo: Default::default(),
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

/// A deterministic `[0,1)` draw seeded by the request state, with the caller's
/// OpenAI `seed` mixed in when it set one.
///
/// THERE IS DELIBERATELY NO SEEDLESS VARIANT. There used to be, and every
/// sampling site on a real backend called it, so `seed` was honoured only on
/// the reference path that has no model. Taking `Option<u64>` makes forgetting
/// the seed a thing you have to type.
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
    /// Per-slug backlog of the stage the model's output feeds; outlives reloads so a stage
    /// bound once keeps gating every later dispatcher.
    downstream: RwLock<FxHashMap<String, Arc<crate::sched::admission::DownstreamCredit>>>,
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
    #[cfg(feature = "cpu")]
    portable: std::sync::OnceLock<Arc<portable::PortableManager>>,
    /// Operator-set residency overrides, slug → state. Absent = [`Residency::Auto`].
    /// Only the control plane writes here; the manager and the request path read it.
    residency: RwLock<FxHashMap<String, Residency>>,
    control: Mutex<FxHashMap<String, std::sync::Weak<tokio::sync::Mutex<()>>>>,
    /// Unix seconds at which this process began offering models. Stamped ONCE:
    /// `GET /v1/models` used to call `now_secs()` per card, so `created`
    /// changed on every scrape and no client could treat it as an identity.
    started: u64,
    /// slug -> compiled context length, captured when the engine is installed
    /// so a model card never has to take the engine mutex behind a live tick.
    max_ctx: RwLock<FxHashMap<String, usize>>,
    /// slug -> whether its backend applies the request's sampling parameters.
    /// Captured at install for the same reason as `max_ctx`.
    sampling_honoured: RwLock<FxHashMap<String, bool>>,
    /// When set, each run records a timeline dumpable at `GET /trace`.
    record_trace: bool,
    trace: Mutex<Timeline>,
    /// Set once on SIGTERM/SIGINT: `/health` turns 503 and the ASR front refuses new work and
    /// ends sessions still receiving audio.
    pub shutdown: tokio::sync::watch::Sender<bool>,
    /// Data-parallel models and their ranks (`--dp`). Unset when every model has one rank.
    dp: std::sync::OnceLock<dp::DpRouter>,
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

    /// The wire spelling, shared by the admin status and the model card.
    pub fn as_str(self) -> &'static str {
        match self {
            Residency::Auto => "auto",
            Residency::Unloading => "unloading",
            Residency::Unloaded => "unloaded",
        }
    }
}

impl AppState {
    pub fn new(registry: Registry, execset: Arc<ExecutorSet>) -> Self {
        Self::with_trace(registry, execset, false)
    }

    /// Construct with per-run timeline recording enabled/disabled.
    pub fn with_trace(registry: Registry, execset: Arc<ExecutorSet>, record_trace: bool) -> Self {
        let _ = crate::obs::serving::started_at_unix_ms();
        AppState {
            registry,
            execset,
            metrics: Arc::new(Metrics::default()),
            model_metrics: RwLock::new(FxHashMap::default()),
            muxes: RwLock::new(FxHashMap::default()),
            downstream: RwLock::new(FxHashMap::default()),
            #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
            gpu: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "cuda")]
            vmm_stats: RwLock::new(FxHashMap::default()),
            #[cfg(feature = "cuda")]
            managers: std::sync::OnceLock::new(),
            slug_group: RwLock::new(FxHashMap::default()),
            turns: std::sync::OnceLock::new(),
            models_roots: std::sync::OnceLock::new(),
            #[cfg(feature = "cpu")]
            portable: std::sync::OnceLock::new(),
            residency: RwLock::new(FxHashMap::default()),
            control: Mutex::new(FxHashMap::default()),
            started: openai::now_secs(),
            max_ctx: RwLock::new(FxHashMap::default()),
            sampling_honoured: RwLock::new(FxHashMap::default()),
            record_trace,
            trace: Mutex::new(Timeline::new()),
            shutdown: tokio::sync::watch::channel(false).0,
            dp: std::sync::OnceLock::new(),
        }
    }

    /// Install the DP router (once, at startup, before any rank loads).
    pub fn install_dp(&self, router: dp::DpRouter) {
        let _ = self.dp.set(router);
    }

    /// The DP router, when any model has more than one rank.
    pub fn dp(&self) -> Option<&dp::DpRouter> {
        self.dp.get()
    }

    /// `model`'s ranks, when it is served data-parallel.
    #[inline]
    pub fn dp_set(&self, model: &str) -> Option<&Arc<dp::DpSet>> {
        self.dp.get()?.set(model)
    }

    /// The DP set and rank of an instance key.
    pub fn dp_rank(&self, key: &str) -> Option<(&Arc<dp::DpSet>, usize)> {
        self.dp.get()?.rank(key)
    }

    /// The registry model an instance key serves: itself unless it is a DP rank.
    pub fn model_of<'a>(&'a self, key: &'a str) -> &'a str {
        self.dp_rank(key).map_or(key, |(set, _)| set.model.as_str())
    }

    /// The CUDA device an instance key runs on (0 when no manager places it).
    pub fn ordinal_of(&self, key: &str) -> u8 {
        #[cfg(feature = "cuda")]
        if let Some(m) = self.manager_for(key) {
            return m.ordinal();
        }
        let _ = key;
        0
    }

    fn refresh_up(&self, key: &str) {
        if let Some((set, r)) = self.dp_rank(key) {
            set.ranks[r].set_up(self.muxes.read().contains_key(key) && self.residency(key).admits());
        }
    }

    /// Pick a rank of `set` and its dispatcher. Ranks in `exclude`, and any whose dispatcher
    /// is gone or preempted by the time it is looked up, are skipped.
    /// The returned [`dp::Pick`] counts the request against its rank until dropped: hold it
    /// until the job is submitted.
    pub fn dp_route<'s>(
        &self,
        set: &'s dp::DpSet,
        session: Option<&str>,
        prompt: Option<&[u32]>,
        mut exclude: u32,
    ) -> Option<(usize, mux::ModelMux, Option<crate::memory::vmm::PrefixKey>, dp::Pick<'s>)> {
        let cfg = self.dp.get()?.cfg;
        loop {
            let routed = set.route(&cfg, session, prompt, exclude)?;
            match self.mux(&set.ranks[routed.rank].key) {
                Some(m) if !m.preempted() => return Some((routed.rank, m, routed.key, routed.pick)),
                _ => exclude |= 1 << routed.rank,
            }
        }
    }

    /// Submit `job` to `mux`, the dispatcher of `set`'s rank `rank` when routed data-parallel. A
    /// rank that has closed or is full hands the same job (tokens and prefix key included) to
    /// another rank.
    pub(crate) fn submit_routed(
        &self,
        routed: Option<(&dp::DpSet, usize)>,
        session: Option<&str>,
        mux: &mux::ModelMux,
        job: mux::Job,
        arrived: std::time::Instant,
        ingress: Option<mux::IngressGuard<'_>>,
    ) -> std::result::Result<(), mux::SubmitError> {
        let Some((set, mut rank)) = routed else {
            return mux.submit_arrived(job, arrived, ingress);
        };
        let mut exclude = 0u32;
        let mut next: Option<mux::ModelMux> = None;
        let mut held: Option<dp::Pick<'_>> = None;
        let mut job = job;
        let mut ingress = ingress;
        loop {
            let target = next.as_ref().unwrap_or(mux);
            let refused = match target.submit_arrived(job, arrived, ingress.take()) {
                Ok(()) => return Ok(()),
                Err(e) => e,
            };
            drop(held.take());
            exclude |= 1 << rank;
            let Some((r, m, _, pick)) = self.dp_route(set, session, None, exclude) else {
                return Err(refused);
            };
            job = match refused {
                mux::SubmitError::Full(j) | mux::SubmitError::Closed(j) => j,
            };
            rank = r;
            next = Some(m);
            held = Some(pick);
        }
    }

    /// Make some rank of `set` serve: a no-op while one does; otherwise load one whose residency
    /// admits (S1 switch on its group).
    #[cfg(feature = "cuda")]
    pub async fn dp_admit(&self, set: &dp::DpSet) -> std::result::Result<(), manager::EnsureError> {
        if set.ranks.iter().any(|r| r.is_up()) {
            return Ok(());
        }
        let mut last = manager::EnsureError::Unloaded;
        for r in set.ranks.iter().filter(|r| self.residency(&r.key).admits()) {
            let Some(mgr) = self.manager_for(&r.key) else { continue };
            match mgr.ensure_resident(&r.key).await {
                Ok(()) => return Ok(()),
                Err(e) => last = e,
            }
        }
        Err(last)
    }

    /// Register a GPU engine for a model slug. Called once at startup.
    #[cfg(any(feature = "cuda", feature = "hsa", feature = "cpu"))]
    pub fn install_gpu_engine(&self, slug: String, engine: engine::ServeEngine) {
        #[cfg(feature = "cuda")]
        if let Some(h) = engine.vmm_stats_handle() {
            self.vmm_stats.write().insert(slug.clone(), h);
        }
        self.max_ctx.write().insert(slug.clone(), engine.max_ctx());
        self.sampling_honoured
            .write()
            .insert(slug.clone(), engine.honours_sampling());
        if let Some((set, r)) = self.dp_rank(&slug) {
            self.max_ctx.write().insert(set.model.clone(), engine.max_ctx());
            self.sampling_honoured.write().insert(set.model.clone(), engine.honours_sampling());
            #[cfg(feature = "cuda")]
            set.set_probe(r, engine.vmm_prefix_probe());
            #[cfg(not(feature = "cuda"))]
            let _ = r;
        }
        self.gpu.write().insert(slug, Arc::new(Mutex::new(engine)));
    }

    #[cfg(feature = "cpu")]
    pub fn install_portable_manager(&self, manager: portable::PortableManager) {
        let _ = self.portable.set(Arc::new(manager));
    }

    #[cfg(feature = "cpu")]
    pub fn portable_manager(&self) -> Option<&Arc<portable::PortableManager>> {
        self.portable.get()
    }

    /// Unix seconds this process started offering models.
    pub fn started(&self) -> u64 {
        self.started
    }

    /// The compiled context length for `slug`, when an engine reported one.
    pub fn max_ctx(&self, slug: &str) -> Option<usize> {
        self.max_ctx.read().get(slug).copied()
    }

    /// Whether `slug`'s backend applies the request's sampling parameters.
    /// `true` when no engine is installed — the CPU reference path samples on
    /// the host, so there is nothing to warn about.
    pub fn sampling_honoured(&self, slug: &str) -> bool {
        self.sampling_honoured.read().get(slug).copied().unwrap_or(true)
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
            if self.gpu.read().contains_key(slug) {
                return true;
            }
            return self.dp_set(slug).is_some_and(|set| {
                let gpu = self.gpu.read();
                set.ranks.iter().any(|r| gpu.contains_key(&r.key))
            });
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
        if let Some((set, r)) = self.dp_rank(slug) {
            set.set_probe(r, None);
        }
        self.gpu.write().remove(slug)
    }

    /// Install the per-group residency managers (once, at startup).
    #[cfg(feature = "cuda")]
    pub fn install_managers(&self, m: Vec<Arc<manager::ModelManager>>) {
        self.install_device_turns(m.len(), true);
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
    pub fn install_device_turns(&self, groups: usize, cuda: bool) {
        self.turns.get_or_init(|| {
            let mode = policy::co_sched(self.registry.slugs().len(), !cuda);
            policy::set_co_sched(mode);
            let turns = (0..groups.max(1))
                .map(|_| Arc::new(cosched::DeviceTurn::serving(mode)))
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

    /// `slug`'s downstream credit (created on first use; unlimited until a stage sets a limit).
    pub fn downstream(&self, slug: &str) -> Arc<crate::sched::admission::DownstreamCredit> {
        let known = self.downstream.read().get(slug).cloned();
        let credit = known.unwrap_or_else(|| Arc::clone(self.downstream.write().entry(slug.to_string()).or_default()));
        if credit.device_turn().is_none() {
            if let Some(turn) = self.device_turn(slug) {
                credit.set_device_turn(turn);
            }
        }
        credit
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
        {
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
        self.refresh_up(slug);
    }

    pub fn model_metrics(&self, slug: &str) -> Arc<Metrics> {
        // A DP model's own counters (a front's, before it routes) land on rank 0.
        if let Some(set) = self.dp_set(slug) {
            return Arc::clone(&set.ranks[0].metrics);
        }
        if let Some(metrics) = self.model_metrics.read().get(slug) {
            return metrics.clone();
        }
        let mut models = self.model_metrics.write();
        models.retain(|name, metrics| {
            let keep = self.registry.contains(name) || Arc::strong_count(metrics) > 1 || self.dp_rank(name).is_some();
            if !keep {
                self.metrics.accumulate_counters(metrics);
            }
            keep
        });
        Arc::clone(models.entry(slug.to_owned()).or_default())
    }

    /// Register a dispatcher for a model slug. Called once at startup.
    pub fn install_mux(&self, slug: String, m: mux::ModelMux) {
        let dp = self.dp_rank(&slug).is_some();
        let key = dp.then(|| slug.clone());
        self.muxes.write().insert(slug, m);
        if let Some(key) = key {
            self.refresh_up(&key);
        }
    }

    /// Remove a dispatcher (S1 eviction) — new lookups fail fast; the caller
    /// drains the returned handle before dropping the engine.
    pub fn remove_mux(&self, slug: &str) -> Option<mux::ModelMux> {
        let m = self.muxes.write().remove(slug);
        self.refresh_up(slug);
        m
    }

    /// Every installed dispatcher.
    pub fn all_muxes(&self) -> Vec<mux::ModelMux> {
        self.muxes.read().values().cloned().collect()
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
            obs.host.rng01 = seeded_unit_with(&prompt_ids, &out_ids, step, gen.seed);
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
        seed: Option<u64>,
    ) -> Result<(u32, usize)> {
        obs.host.tokens.clear();
        obs.host.rng01 = seeded_unit_with(prompt_ids, out_ids, step, seed);
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

/// Maximum request body this server accepts.
///
/// axum's default is 2 MiB, chosen for ordinary web forms, and it is applied by
/// the `Json` extractor BEFORE any handler runs — so a long-context chat
/// request was refused with a bare 413 and no error envelope. A million-token
/// conversation is several MiB of JSON; the cap belongs at a size that reflects
/// what this server is for.
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

/// Build the axum app.
pub fn app(state: Arc<AppState>) -> Router {
    let router = Router::new();
    #[cfg(feature = "cuda")]
    let router = router.route("/v1/audio/speech", post(crate::tts::serving::speech));
    let router = router
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completion::completions))
        .route("/tokenize", post(tokenize::tokenize))
        .route("/detokenize", post(tokenize::detokenize))
        .route("/v1/models", get(models::list_models))
        // GET only, with a 404 fallback for every other method. The admin
        // routes live at `/v1/models/load|unload|status`, and a bare
        // `get(...)` here answered a POST to one of those with 405 on the
        // PUBLIC router — announcing that the control plane exists on a
        // listener that does not serve it. Static paths still win on the
        // merged router, so admin keeps working where it is mounted.
        .route(
            "/v1/models/:id",
            get(models::get_model).fallback(models::model_route_fallback),
        )
        // BOTH spellings. vLLM serves `/health`, and every k8s probe and
        // benchmark harness copied from vLLM asks for it; plowrt served only
        // `/healthz`, so all of them got a 404 from a healthy server.
        .route("/health", get(healthz))
        .route("/healthz", get(healthz))
        .route("/metrics", get(metrics_handler))
        .route("/v1/metrics", get(metrics_snapshot_handler))
        .route("/trace", get(trace_handler))
        .route("/v1/turns/:session", get(turns::session_turns))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(Arc::clone(&state));
    #[cfg(feature = "cuda")]
    let router = router.merge(crate::asr::serving::AsrServer::for_serve(state).transcription_router(true));
    router
}

/// `GET /trace` — Chrome-trace JSON from traced live runs (§O, `--trace`).
async fn trace_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    ([("content-type", "application/json")], state.trace_json()).into_response()
}

async fn healthz(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> (axum::http::StatusCode, String) {
    let dead: Vec<String> = state
        .model_metrics
        .read()
        .iter()
        .filter(|(_, m)| m.engine_dead.load(std::sync::atomic::Ordering::Relaxed))
        .map(|(slug, _)| slug.clone())
        .collect();
    #[cfg(feature = "cuda")]
    let dead = [dead, crate::asr::serving::dead_encoders()].concat();
    if *state.shutdown.borrow() {
        return (axum::http::StatusCode::SERVICE_UNAVAILABLE, "shutting down".into());
    }
    // A DP model with a live rank still serves: report the dead ranks, answer 200.
    let (degraded, dead): (Vec<String>, Vec<String>) = dead.into_iter().partition(|key| {
        state
            .dp_rank(key)
            .is_some_and(|(set, _)| set.ranks.iter().any(|r| !dead_rank(&r.metrics)))
    });
    if dead.is_empty() && !degraded.is_empty() {
        return (axum::http::StatusCode::OK, format!("degraded: DP ranks dead: {}", degraded.join(",")));
    }
    if dead.is_empty() {
        (axum::http::StatusCode::OK, "ok".into())
    } else {
        (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            format!("engine dead (fatal device fault or ASR encoder exit): {}", dead.join(",")),
        )
    }
}

fn dp_metrics(out: &mut String, router: &dp::DpRouter, state: &AppState) {
    use crate::obs::serving::{escape_label, family};
    use std::fmt::Write;
    use std::sync::atomic::Ordering::Relaxed;
    let mut sets: Vec<_> = router.sets().collect();
    sets.sort_unstable_by(|a, b| a.model.cmp(&b.model));
    family(out, "plowrt_dp_rank_info", "gauge", "DP rank placement: the device its group starts at.");
    for set in &sets {
        for r in &set.ranks {
            let m = escape_label(&set.model);
            let _ = writeln!(out, "plowrt_dp_rank_info{{model_name=\"{m}\",engine=\"{}\",device=\"{}\"}} 1", r.rank, r.ordinal);
        }
    }
    family(out, "plowrt_dp_rank_up", "gauge", "DP rank has a dispatcher and admits requests.");
    for set in &sets {
        for r in &set.ranks {
            let m = escape_label(&set.model);
            let _ = writeln!(out, "plowrt_dp_rank_up{{model_name=\"{m}\",engine=\"{}\"}} {}", r.rank, u8::from(r.is_up() && state.residency(&r.key).admits()));
        }
    }
    family(out, "plowrt_dp_rank_load", "gauge", "DP router load signal: (queued + active) / slots + KV pressure.");
    for set in &sets {
        for r in &set.ranks {
            let m = escape_label(&set.model);
            let _ = writeln!(out, "plowrt_dp_rank_load{{model_name=\"{m}\",engine=\"{}\"}} {}", r.rank, dp::load(&r.cand()));
        }
    }
    family(out, "plowrt_dp_route_decisions_total", "counter", "DP routing decisions by the rule that chose the rank.");
    for set in &sets {
        for reason in dp::Reason::ALL {
            let m = escape_label(&set.model);
            let _ = writeln!(out, "plowrt_dp_route_decisions_total{{model_name=\"{m}\",reason=\"{}\"}} {}", reason.as_str(), set.stats.decisions[reason as usize].load(Relaxed));
        }
    }
    for (name, help, read) in [
        ("plowrt_dp_route_retries_total", "Requests resubmitted to another rank after the first closed.", (|s: &dp::DpStats| s.retries.load(Relaxed)) as fn(&dp::DpStats) -> u64),
        ("plowrt_route_probes_total", "Prefix-cache probes the DP router made.", |s| s.probes.load(Relaxed)),
        ("plowrt_route_probe_contended_total", "Prefix-cache probes skipped because the cache lock was held.", |s| s.probe_contended.load(Relaxed)),
    ] {
        family(out, name, "counter", help);
        for set in &sets {
            let _ = writeln!(out, "{name}{{model_name=\"{}\"}} {}", escape_label(&set.model), read(&set.stats));
        }
    }
    family(out, "plowrt_dp_route_seconds", "gauge", "DP route wall time, upper bound of the log2 bucket at the quantile.");
    for set in &sets {
        for q in [0.5, 0.99, 0.999] {
            let _ = writeln!(out, "plowrt_dp_route_seconds{{model_name=\"{}\",quantile=\"{q}\"}} {}", escape_label(&set.model), set.stats.route_ns_quantile(q) as f64 * 1e-9);
        }
    }
    family(out, "plowrt_dp_sessions", "gauge", "Sessions pinned to a DP rank.");
    for set in &sets {
        let _ = writeln!(out, "plowrt_dp_sessions{{model_name=\"{}\"}} {}", escape_label(&set.model), set.sessions());
    }
}

fn dead_rank(m: &Metrics) -> bool {
    m.engine_dead.load(std::sync::atomic::Ordering::Relaxed)
}

fn packet_model(slug: &str) -> bool {
    #[cfg(feature = "cuda")]
    return crate::asr::serving::packet_model_names().iter().any(|n| n == slug);
    #[cfg(not(feature = "cuda"))]
    {
        let _ = slug;
        false
    }
}

fn metrics_models(state: &AppState) -> Vec<(String, Arc<Metrics>, bool)> {
        let mut metrics = state.model_metrics.write();
        for slug in state.registry.slugs() {
            match state.dp_set(&slug) {
                Some(set) => {
                    for r in &set.ranks {
                        metrics.entry(r.key.clone()).or_insert_with(|| Arc::clone(&r.metrics));
                    }
                }
                None => {
                    metrics.entry(slug).or_default();
                }
            }
        }
        metrics.retain(|slug, metrics| {
            let keep = state.registry.contains(slug) && state.dp_set(slug).is_none()
                || Arc::strong_count(metrics) > 1
                || state.dp_rank(slug).is_some();
            if !keep {
                state.metrics.accumulate_counters(metrics);
            }
            keep
        });
        let mut models: Vec<_> = metrics.iter().map(|(slug, m)| {
            (
                slug.clone(),
                Arc::clone(m),
                state.mux(slug).is_some() && state.residency(slug).admits() || packet_model(slug),
            )
        }).collect();
        models.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        models
}

async fn metrics_snapshot_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::Json<crate::obs::serving::RuntimeSnapshot> {
    axum::Json(crate::obs::serving::snapshot(&metrics_models(&state)))
}

async fn metrics_handler(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let models = metrics_models(&state);
    let aggregate = Metrics::default();
    aggregate.accumulate(&state.metrics);
    for (_, metrics, _) in &models {
        aggregate.accumulate(metrics);
    }
    let mut out = aggregate.to_prometheus();
    for (index, (slug, metrics, _)) in models.iter().enumerate() {
        let label = crate::obs::serving::model_labels(slug);
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
                let _ = writeln!(out, "{name}{{{label}}} {value}");
            }
        }
    }
    crate::obs::serving::ServingMetrics::write(&mut out, &models);
    if let Some(router) = state.dp() {
        dp_metrics(&mut out, router, &state);
    }
    // Prefix-cache (VMM) counters, one block per GPU-served model, read
    // through the engine-lock-free stats handles — series stay continuous
    // under sustained inference (only the pool mutex is taken, µs holds).
    #[cfg(feature = "cuda")]
    for (name, kind, help) in [
        ("attach_hits_total", "counter", "Successful prefix attach requests."),
        ("attach_misses_total", "counter", "Prefix attach requests without a reusable prefix."),
        ("tokens_queried_total", "counter", "Prompt tokens queried for prefix cache reuse."),
        ("tokens_attached_total", "counter", "Tokens attached from reusable prefix blocks."),
        ("hash_collisions_total", "counter", "Prefix hash collisions detected."),
        ("blocks_shared_mapped_total", "counter", "Shared prefix blocks mapped."),
        ("nodes_evicted_total", "counter", "Prefix tree nodes evicted."),
        ("blocks_live", "gauge", "Live prefix pool blocks."),
        ("blocks_stale", "gauge", "Retired-window blocks still mapped, awaiting the reclaimer."),
        ("blocks_pooled", "gauge", "Zero-reference blocks parked in the reuse pool."),
        ("cache_blocks", "gauge", "Prefix cache blocks."),
        ("cache_bytes", "gauge", "Prefix cache bytes."),
        ("snapshot_bytes", "gauge", "Prefix snapshot bytes."),
        ("snapshots_evicted_total", "counter", "Prefix snapshots evicted."),
        ("eviction_units_total", "counter", "Suffix evictions: a boundary plus the blocks only it kept attachable."),
        ("eviction_unit_bytes_total", "counter", "Bytes freed by suffix evictions."),
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
             plowrt_prefix_tokens_queried_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_tokens_attached_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_hash_collisions_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_blocks_shared_mapped_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_nodes_evicted_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_blocks_live{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_blocks_stale{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_blocks_pooled{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_cache_blocks{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_cache_bytes{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_snapshot_bytes{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_snapshots_evicted_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_eviction_units_total{{model=\"{slug}\"}} {}\n\
             plowrt_prefix_eviction_unit_bytes_total{{model=\"{slug}\"}} {}\n",
            s.attach_hits,
            s.attach_misses,
            s.tokens_queried,
            s.tokens_attached,
            s.hash_collisions,
            s.blocks_shared_mapped,
            s.nodes_evicted,
            s.blocks_live,
            s.blocks_stale,
            s.blocks_pooled,
            s.cache_blocks,
            s.cache_bytes,
            s.snapshot_bytes,
            s.snapshots_evicted,
            s.eviction_units,
            s.eviction_unit_bytes,
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
        RuntimeError::Rejected(_) | RuntimeError::Oom(_) | RuntimeError::Overloaded(_) => {
            ("rate_limit_error", Some("server_overloaded"))
        }
        _ => ("server_error", None),
    };
    api_error(status, err.to_string(), kind, code, None)
}

#[cfg(test)]
mod health_tests {
    use super::{healthz, AppState};
    use axum::http::StatusCode;
    use std::sync::{atomic::Ordering, Arc};

    /// A poisoned CUDA context rejects every request; `/health` must say so, or an
    /// orchestrator keeps routing traffic to an instance that can never answer.
    #[tokio::test]
    async fn health_reports_a_dead_engine() {
        let backend: Arc<dyn crate::device::Backend> = Arc::new(crate::device::cpu::CpuBackend::new(1));
        let execset = Arc::new(crate::exec::ExecutorSet::bringup(backend).unwrap());
        let state = Arc::new(AppState::new(crate::orch::Registry::new(), execset));
        let metrics = state.model_metrics("m");
        let (code, _) = healthz(axum::extract::State(Arc::clone(&state))).await;
        assert_eq!(code, StatusCode::OK);
        metrics.engine_dead.store(true, Ordering::Relaxed);
        let (code, body) = healthz(axum::extract::State(state)).await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
        assert!(body.contains('m'));
    }

    /// One dead DP rank degrades the model; only every rank dead fails `/health`.
    #[tokio::test]
    async fn health_is_degraded_until_every_dp_rank_is_dead() {
        let backend: Arc<dyn crate::device::Backend> = Arc::new(crate::device::cpu::CpuBackend::new(1));
        let execset = Arc::new(crate::exec::ExecutorSet::bringup(backend).unwrap());
        let state = Arc::new(AppState::new(crate::orch::Registry::new(), execset));
        let mut router = super::dp::DpRouter::new(Default::default());
        router.add(super::dp::DpSet::new("g", (0..2).map(|r| (r, r as usize, state.model_metrics(&format!("g#{r}")))).collect()));
        state.install_dp(router);
        let set = Arc::clone(state.dp_set("g").unwrap());
        set.ranks[0].metrics.engine_dead.store(true, Ordering::Relaxed);
        let (code, body) = healthz(axum::extract::State(Arc::clone(&state))).await;
        assert_eq!(code, StatusCode::OK);
        assert!(body.starts_with("degraded") && body.contains("g#0"), "{body}");
        set.ranks[1].metrics.engine_dead.store(true, Ordering::Relaxed);
        let (code, _) = healthz(axum::extract::State(state)).await;
        assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn dp_ranks_label_their_model_and_engine() {
        use crate::obs::serving::model_labels;
        assert_eq!(model_labels("gemma"), "model_name=\"gemma\",engine=\"0\"");
        assert_eq!(model_labels("gemma#5"), "model_name=\"gemma\",engine=\"5\"");
        assert_eq!(model_labels("a\"b#1"), "model_name=\"a\\\"b\",engine=\"1\"");
    }
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
        RuntimeError::Rejected(_) | RuntimeError::Oom(_) | RuntimeError::Overloaded(_) => {
            StatusCode::TOO_MANY_REQUESTS
        }
        RuntimeError::Unavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
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
