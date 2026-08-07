//! §I.5 Multi-model serving — S1 switching with a VRAM budget planner.
//!
//! Decision of record (5e880f6): **S1 switching only** — at most the
//! VRAM-fitting subset of registered models is RESIDENT (weights + KV + tables
//! on-device, mux live). A request for a non-resident model triggers a switch:
//! evict LRU resident engines until the planner says the target fits, then
//! load it. No S2 resident multi-tenancy (no weight paging, no shared arenas).
//!
//! **Planner.** Per model, from the PLOWDEV blob header alone (no GPU):
//! weights bytes (`model.*` / `fp8/*` tensors), KV arena bytes at the compiled
//! context (`kv.*` tensors), activation/table bytes (the rest). The engine
//! allocates every blob tensor at load, so `tensor total + overhead` is the
//! footprint; `overhead` (modules, program tables, counter blocks, allocator
//! slack) is measured against `cuMemGetInfo` at the first load of each assets
//! dir and cached — the planner self-calibrates.
//!
//! **Cross-model execution.** Resident models keep their per-model dispatchers
//! (one `ModelMux` each); GPU execution serializes naturally: every engine
//! shares ONE `CudaBackend` (one context) and launches cooperatively on the
//! NULL stream, so kernels from different models are stream-ordered — a tick's
//! launch can never interleave with another model's launch, and each engine
//! synchronizes under its own mutex before reading results. No global launch
//! lock is needed.
//!
//! **Admission.** Resident models shed on their own KV pools / slot tables as
//! before. A non-resident model's request first passes [`ModelManager::
//! ensure_resident`]: switches serialize on one async lock (concurrent
//! requests for the same target coalesce); if eviction cannot make the target
//! fit, the request is shed with [`EnsureError::WontFit`] (503 + Retry-After
//! at the HTTP layer). A switch waits for the victim's in-flight generations
//! to complete (mux drain) — S1 does not preempt.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::Instant;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::asset::devblob::DevBlob;
use crate::device::cuda::CudaBackend;
use crate::memory::vmm::VmmOps;
use crate::serve::mux::{self, MuxConfig};
use crate::serve::AppState;
use crate::{Result, RuntimeError};

/// Planner headroom kept free on top of a model's requirement (driver slack,
/// transient staging).
const RESERVE: u64 = 256 << 20;
/// Non-tensor overhead assumed before the first load measures the real value.
const DEFAULT_OVERHEAD: u64 = 512 << 20;

/// Per-model VRAM plan, derived from the blob header (sizes only — the blob
/// carries no addresses).
#[derive(Clone, Copy, Debug)]
pub struct BlobPlan {
    /// Checkpoint weights (`model.*`, `fp8/*`).
    pub weights_bytes: u64,
    /// KV arena at the compiled context (`kv.*`).
    pub kv_bytes: u64,
    /// Activations, IO, MoE tables — every other blob tensor.
    pub other_bytes: u64,
}

impl BlobPlan {
    /// Sum of every device tensor the engine will allocate.
    pub fn tensor_total(&self) -> u64 {
        self.weights_bytes + self.kv_bytes + self.other_bytes
    }

    /// Bytes of this model's load the weight slab can satisfy from pooled
    /// physical chunks (`VmmOps::pool_take`): the slab backs every non-VMM-KV
    /// blob tensor, which is at least weights + other. (With the VMM prefix
    /// cache off, KV rides the slab too — this then under-credits, which only
    /// costs reuse, never correctness.)
    pub fn slab_reusable(&self) -> u64 {
        self.weights_bytes + self.other_bytes
    }

    /// Classify one blob tensor into the plan. Free-standing so it is
    /// unit-testable without a real blob.
    ///
    /// The weight test is `packet::names::is_checkpoint_weight` — the same predicate the two
    /// loaders bind on, so the plan cannot disagree with what is actually uploaded. It used to be
    /// a local `starts_with("model.") || starts_with("fp8/")`, which under-counted an untied
    /// `lm_head.weight` (declared at the top level) and would have counted a wrapper-prefixed
    /// tower (Kimi-K3's `language_model.model.…`) as zero weight bytes.
    fn add(&mut self, name: &str, bytes: u64) {
        if name.starts_with("kv.") {
            self.kv_bytes += bytes;
        } else if packet::names::is_checkpoint_weight(name) {
            self.weights_bytes += bytes;
        } else {
            self.other_bytes += bytes;
        }
    }

    /// Parse the plan from the assets dir's PLOWDEV blob. Reads the blob file
    /// once (header + tensor decls; the init section ride-along is the cost of
    /// the shared parser — startup-only, never on the request path).
    pub fn from_dir(dir: &Path) -> Result<BlobPlan> {
        let pkt = DevBlob::find_in_dir(dir)?
            .ok_or_else(|| RuntimeError::Device(format!("no PLOWDEV blob in {}", dir.display())))?;
        let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io {
            path: pkt.clone(),
            source,
        })?;
        // METADATA ONLY -- this sums tensor bytes for the memory plan and never dispatches a
        // packet, so L2-domain placement is irrelevant here. The real guard is in the engine,
        // which checks the CODE OBJECT for `plow_l2_place_dispatch_1`. Using the strict parse
        // made `serve` reject every placed blob before the engine ever saw it.
        let blob = DevBlob::parse_l2(&raw, true)?;
        let mut plan = BlobPlan {
            weights_bytes: 0,
            kv_bytes: 0,
            other_bytes: 0,
        };
        for t in &blob.tensors {
            plan.add(&t.name, t.bytes);
        }
        Ok(plan)
    }
}

/// One registered (not necessarily resident) model.
struct Managed {
    slug: String,
    dir: PathBuf,
    ckpt: PathBuf,
    plan: BlobPlan,
}

/// Phase timings of the last completed switch — the honest S1 cost record
/// (read by the GPU gate test and logged per switch).
#[derive(Clone, Debug, Default)]
pub struct SwitchReport {
    pub target: String,
    /// `(victim slug, drain_ms, unload_ms)` per evicted model.
    pub evicted: Vec<(String, f64, f64)>,
    /// `GpuEngine::load` wall time (dominated by the checkpoint H2D).
    pub load_ms: f64,
    /// VRAM used (bytes) after the switch settled.
    pub vram_used: u64,
}

/// Why `ensure_resident` refused.
#[derive(Debug)]
pub enum EnsureError {
    /// Even after evicting every other resident model the target cannot fit.
    /// Shed with 503 + Retry-After.
    WontFit { need: u64, free: u64 },
    /// The engine load (or drain plumbing) failed.
    Load(RuntimeError),
}

impl std::fmt::Display for EnsureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnsureError::WontFit { need, free } => write!(
                f,
                "model does not fit: needs {} MiB, {} MiB free after eviction",
                need >> 20,
                free >> 20
            ),
            EnsureError::Load(e) => write!(f, "model load failed: {e}"),
        }
    }
}

/// The S1 model manager: residency, planner, switch orchestration.
pub struct ModelManager {
    /// `Weak` breaks the `AppState ↔ manager` cycle; the state outlives every
    /// request, so upgrades only fail during process teardown.
    state: Weak<AppState>,
    be: Arc<CudaBackend>,
    mux_cfg: MuxConfig,
    models: Vec<Managed>,
    /// Serializes switches. Resident-model requests never take it.
    switch: tokio::sync::Mutex<()>,
    /// slug → last request time (LRU eviction order).
    last_use: Mutex<FxHashMap<String, Instant>>,
    /// slug → measured non-tensor overhead (first-load calibration).
    overhead: Mutex<FxHashMap<String, u64>>,
    /// Optional VRAM budget cap (bytes): the planner pretends the card has
    /// only this much. Forces switching on hardware where models co-fit
    /// (tests, A/B) — `PLOW_VRAM_BUDGET_MIB` in `main`.
    budget: Option<u64>,
    /// Last completed switch (None until the first one).
    pub last_switch: Mutex<Option<SwitchReport>>,
}

impl ModelManager {
    /// Build the manager for `(slug, assets dir, checkpoint dir)` triples —
    /// every registered bundle that ships a PLOWDEV blob. Parses each blob's
    /// plan up front (startup-only file reads).
    pub fn new(
        be: Arc<CudaBackend>,
        state: &Arc<AppState>,
        mux_cfg: MuxConfig,
        models: Vec<(String, PathBuf, PathBuf)>,
        budget: Option<u64>,
    ) -> Result<ModelManager> {
        let mut managed = Vec::with_capacity(models.len());
        for (slug, dir, ckpt) in models {
            let plan = BlobPlan::from_dir(&dir)?;
            tracing::info!(
                %slug,
                weights_gib = gib(plan.weights_bytes),
                kv_gib = gib(plan.kv_bytes),
                other_gib = gib(plan.other_bytes),
                total_gib = gib(plan.tensor_total()),
                "planner: model registered"
            );
            managed.push(Managed {
                slug,
                dir,
                ckpt,
                plan,
            });
        }
        if managed.len() > 1 {
            // Multi-model serving: keep dropped weight slabs' physical chunks
            // pooled so a switch re-maps them (µs-class) instead of re-paying
            // the driver's serial page commit (~13 GiB/s). Safe to default on
            // HERE because this planner credits the pool in its fit check
            // (`pool_bytes`) and trims what a target cannot consume
            // (`pool_trim`). `PLOW_SLAB_KEEP=0` force-disables.
            crate::memory::vmm::set_slab_keep_default(true);
            tracing::info!("planner: slab chunk pool enabled (multi-model)");
        }
        Ok(ModelManager {
            state: Arc::downgrade(state),
            be,
            mux_cfg,
            models: managed,
            switch: tokio::sync::Mutex::new(()),
            last_use: Mutex::new(FxHashMap::default()),
            overhead: Mutex::new(FxHashMap::default()),
            budget,
            last_switch: Mutex::new(None),
        })
    }

    /// Whether `slug` is under manager control (has a blob plan).
    pub fn manages(&self, slug: &str) -> bool {
        self.models.iter().any(|m| m.slug == slug)
    }

    /// The parsed plan for `slug` (tests / capacity reporting).
    pub fn plan(&self, slug: &str) -> Option<BlobPlan> {
        self.models.iter().find(|m| m.slug == slug).map(|m| m.plan)
    }

    /// Planner requirement for `slug`: tensor total + (measured|default)
    /// overhead. `None` for unmanaged slugs.
    pub fn required(&self, slug: &str) -> Option<u64> {
        let m = self.models.iter().find(|m| m.slug == slug)?;
        let ovh = self
            .overhead
            .lock()
            .get(slug)
            .copied()
            .unwrap_or(DEFAULT_OVERHEAD);
        Some(m.plan.tensor_total() + ovh)
    }

    fn touch(&self, slug: &str) {
        self.last_use
            .lock()
            .insert(slug.to_string(), Instant::now());
    }

    /// Effective free VRAM under the optional budget cap: with a budget B on a
    /// card of total T, pretend the card has only B — free' = free − (T − B).
    fn free_vram(&self) -> Result<u64> {
        let (free, total) = self.be.mem_info()?;
        Ok(match self.budget {
            Some(b) if b < total => free.saturating_sub(total - b),
            _ => free,
        })
    }

    fn state(&self) -> std::result::Result<Arc<AppState>, EnsureError> {
        self.state
            .upgrade()
            .ok_or_else(|| EnsureError::Load(RuntimeError::Msg("server shutting down".into())))
    }

    /// Resident = engine installed.
    pub fn is_resident(&self, slug: &str) -> bool {
        self.state
            .upgrade()
            .map(|s| s.has_gpu_engine(slug))
            .unwrap_or(false)
    }

    /// Startup: load models in registration order while the planner says they
    /// fit (no eviction — earlier models win). At least one model must load.
    pub async fn load_initial(&self) -> Result<()> {
        let mut loaded = 0usize;
        for m in &self.models {
            let need = self.required(&m.slug).expect("managed") + RESERVE;
            let free = self.free_vram()?;
            if free < need {
                tracing::info!(
                    slug = %m.slug,
                    need_gib = gib(need),
                    free_gib = gib(free),
                    "planner: not resident at startup (S1 switch will load on demand)"
                );
                continue;
            }
            self.load_model(m)
                .await
                .map_err(|e| RuntimeError::Msg(format!("{}: {e}", m.slug)))?;
            loaded += 1;
        }
        if loaded == 0 && !self.models.is_empty() {
            return Err(RuntimeError::Msg(
                "planner: no model fits the VRAM budget at startup".into(),
            ));
        }
        Ok(())
    }

    /// Make `slug` resident, evicting LRU residents if the planner requires
    /// it. Fast no-lock path when already resident. Concurrent callers for a
    /// non-resident model serialize on the switch lock and coalesce on the
    /// post-lock recheck (a speculative preload in flight for `slug` counts —
    /// the caller waits on the same lock and finds the model resident).
    pub async fn ensure_resident(
        self: &Arc<Self>,
        slug: &str,
    ) -> std::result::Result<(), EnsureError> {
        if !self.manages(slug) {
            return Err(EnsureError::Load(RuntimeError::UnknownModel(slug.into())));
        }
        if self.is_resident(slug) {
            self.touch(slug);
            return Ok(());
        }

        let _g = self.switch.lock().await;
        if self.is_resident(slug) {
            self.touch(slug);
            return Ok(());
        }
        let m = self
            .models
            .iter()
            .find(|m| m.slug == slug)
            .expect("managed");
        let need = self.required(slug).expect("managed") + RESERVE;
        let mut report = SwitchReport {
            target: slug.to_string(),
            ..SwitchReport::default()
        };

        // Evict LRU residents until the target fits (or nothing is left).
        // Pooled slab chunks (an evicted model's kept physical memory) count
        // toward the target: the incoming weight slab re-maps them instead of
        // allocating fresh VRAM, so `free + pool` is the honest capacity.
        // The trim comes FIRST, every iteration (each evict can grow the
        // pool): chunks beyond what the target's slab can consume must be
        // REAL free memory before the load — tables, VMM KV blocks and
        // overhead all allocate fresh — and holding them uncredited squeezed
        // a fitting switch into WontFit (a 30 GiB victim pool credited at a
        // 24 GiB target's cap left 5.7 GiB dead).
        let reusable_cap = m.plan.slab_reusable();
        loop {
            let trimmed = VmmOps::pool_trim(&*self.be, reusable_cap);
            if trimmed > 0 {
                tracing::info!(
                    trimmed_mib = trimmed >> 20,
                    "planner: released pooled chunks the target cannot reuse"
                );
            }
            let free = self.free_vram().map_err(EnsureError::Load)?;
            let credit = VmmOps::pool_bytes(&*self.be);
            if free + credit >= need {
                break;
            }
            let victims: Vec<String> = self
                .models
                .iter()
                .filter(|v| v.slug != slug && self.is_resident(&v.slug))
                .map(|v| v.slug.clone())
                .collect();
            let Some(victim) = pick_victim(&victims, &self.last_use.lock()) else {
                tracing::warn!(
                    %slug,
                    need_gib = gib(need),
                    free_gib = gib(free + credit),
                    "planner: switch cannot fit — shedding"
                );
                return Err(EnsureError::WontFit {
                    need,
                    free: free + credit,
                });
            };
            let (drain_ms, unload_ms) = self.evict(&victim).await?;
            report.evicted.push((victim, drain_ms, unload_ms));
        }

        let t0 = Instant::now();
        self.load_model(m).await?;
        report.load_ms = ms(t0);
        let (free, total) = self.be.mem_info().map_err(EnsureError::Load)?;
        report.vram_used = total - free;
        tracing::info!(
            %slug,
            evicted = ?report.evicted,
            load_ms = report.load_ms,
            vram_used_gib = gib(report.vram_used),
            "S1 switch complete"
        );
        *self.last_switch.lock() = Some(report);
        self.touch(slug);
        self.maybe_preload();
        Ok(())
    }

    /// After a switch: if spare VRAM fits the hottest non-resident model
    /// WITHOUT evicting anyone, load it in the background. A later request
    /// for it then finds it resident — or coalesces on the switch lock while
    /// the preload finishes — instead of paying the full 2 s-class switch.
    /// A no-op when nothing fits; `PLOW_PRELOAD=0` disables.
    fn maybe_preload(self: &Arc<Self>) {
        if std::env::var("PLOW_PRELOAD").ok().as_deref() == Some("0") {
            return;
        }
        let mgr = Arc::clone(self);
        tokio::spawn(async move {
            // The switch lock makes residency and free-VRAM stable for the
            // candidate pick, and lets a real request for the preloading
            // model coalesce instead of double-loading. A request for a
            // THIRD model queues behind the preload — the cost cap is one
            // load, which that request was already going to pay.
            let _g = mgr.switch.lock().await;
            let Some(slug) = mgr.preload_candidate() else {
                return;
            };
            let m = mgr
                .models
                .iter()
                .find(|m| m.slug == slug)
                .expect("candidate is managed");
            VmmOps::pool_trim(&*mgr.be, m.plan.slab_reusable());
            let t0 = Instant::now();
            match mgr.load_model(m).await {
                Ok(()) => {
                    tracing::info!(%slug, load_ms = ms(t0), "planner: speculative preload complete")
                }
                Err(e) => tracing::warn!(%slug, error = %e, "planner: speculative preload failed"),
            }
        });
    }

    /// Hottest (most-recently-requested) non-resident model whose requirement
    /// fits free VRAM plus usable pooled chunks — no eviction considered.
    fn preload_candidate(&self) -> Option<String> {
        let free = self.free_vram().ok()?;
        let pool = VmmOps::pool_bytes(&*self.be);
        let last_use = self.last_use.lock();
        self.models
            .iter()
            .filter(|m| !self.is_resident(&m.slug))
            .filter(|m| {
                let credit = pool.min(m.plan.slab_reusable());
                self.required(&m.slug).expect("managed") + RESERVE <= free + credit
            })
            .max_by_key(|m| last_use.get(&m.slug).copied())
            .map(|m| m.slug.clone())
    }

    /// Tear one resident model down: remove the mux (new submits fail fast),
    /// drain in-flight generations, then drop the engine — VRAM returns to the
    /// driver (a36aa30/d058626 lifecycle). Returns `(drain_ms, unload_ms)`.
    ///
    /// The drain is graceful (every live slot runs to completion — unbounded,
    /// O(max_tokens × service_ms)) unless `PLOW_DRAIN_TIMEOUT_MS` is set:
    /// then generations get that long to finish before the remainder is
    /// preempted (`ModelMux::preempt` — streams close with
    /// `finish_reason: "preempted"` and the tokens produced so far), bounding
    /// the switch's drain phase. `0` preempts immediately.
    async fn evict(&self, slug: &str) -> std::result::Result<(f64, f64), EnsureError> {
        let state = self.state()?;
        let t0 = Instant::now();
        if let Some(mux) = state.remove_mux(slug) {
            match drain_timeout_ms() {
                None => mux.drain().await,
                Some(ms) => {
                    let deadline = std::time::Duration::from_millis(ms);
                    if tokio::time::timeout(deadline, mux.drain()).await.is_err() {
                        tracing::info!(
                            %slug,
                            timeout_ms = ms,
                            "drain deadline passed — preempting live generations"
                        );
                        mux.preempt().await;
                    }
                }
            }
        }
        let drain_ms = ms(t0);

        let t1 = Instant::now();
        let engine = state.remove_gpu_engine(slug);
        if let Some(e) = &engine {
            let holders = Arc::strong_count(e);
            if holders > 1 {
                // A tick clone should be impossible after drain (the mux loop
                // exited); a held Arc means the VRAM will NOT free and the
                // planner's ledger is wrong — loud, not silent.
                tracing::error!(%slug, holders, "engine still referenced at evict — VRAM leak");
            }
        }
        drop(engine);
        let unload_ms = ms(t1);
        let (free, _) = self.be.mem_info().map_err(EnsureError::Load)?;
        tracing::info!(%slug, drain_ms, unload_ms, free_gib = gib(free), "model evicted");
        Ok((drain_ms, unload_ms))
    }

    /// Load `m`'s engine (blocking H2D on the blocking pool), install it, and
    /// spawn its dispatcher. Calibrates the overhead cache on first load.
    async fn load_model(&self, m: &Managed) -> std::result::Result<(), EnsureError> {
        let state = self.state()?;
        let bundle = state.registry.get(&m.slug).map_err(EnsureError::Load)?;
        if bundle.tokenizer().is_byte_fallback() {
            return Err(EnsureError::Load(RuntimeError::Msg(format!(
                "{}: GPU engine requires a real tokenizer.json in {}",
                m.slug,
                m.dir.display()
            ))));
        }

        let (free_before, _) = self.be.mem_info().map_err(EnsureError::Load)?;
        let pool_before = VmmOps::pool_bytes(&*self.be);
        let be = Arc::clone(&self.be);
        let (dir, ckpt) = (m.dir.clone(), m.ckpt.clone());
        let engine =
            tokio::task::spawn_blocking(move || crate::exec::gpu::GpuEngine::load(be, &dir, &ckpt))
                .await
                .map_err(|e| EnsureError::Load(RuntimeError::Msg(format!("load task: {e}"))))?
                .map_err(EnsureError::Load)?;
        let (free_after, _) = self.be.mem_info().map_err(EnsureError::Load)?;
        let pool_after = VmmOps::pool_bytes(&*self.be);

        // First-load calibration: measured footprint minus the planned tensor
        // bytes = module/table/allocator overhead for this assets dir. Pool
        // chunks the slab consumed moved from the pool ledger into the engine
        // without touching `free` — count both ledgers or reused chunks make
        // the model look smaller than it is.
        let used = (free_before + pool_before).saturating_sub(free_after + pool_after);
        let measured = used.saturating_sub(m.plan.tensor_total());
        self.overhead
            .lock()
            .entry(m.slug.clone())
            .or_insert(measured);
        tracing::info!(
            slug = %m.slug,
            used_gib = gib(used),
            planned_gib = gib(m.plan.tensor_total()),
            overhead_mib = measured >> 20,
            "planner: load measured"
        );

        state.install_gpu_engine(
            m.slug.clone(),
            crate::serve::engine::ServeEngine::Cuda(engine),
        );
        let mux = mux::spawn(m.slug.clone(), bundle, Arc::clone(&state), self.mux_cfg);
        state.install_mux(m.slug.clone(), mux);
        Ok(())
    }
}

/// `PLOW_DRAIN_TIMEOUT_MS`: how long an eviction lets in-flight generations
/// finish before preempting them. Unset = graceful unbounded drain (the
/// pre-existing behavior); read per-evict so an in-process flip is honored.
fn drain_timeout_ms() -> Option<u64> {
    std::env::var("PLOW_DRAIN_TIMEOUT_MS").ok()?.parse().ok()
}

/// Least-recently-used pick: the resident slug with the oldest (or absent)
/// last-use stamp. Free-standing for unit tests.
fn pick_victim(residents: &[String], last_use: &FxHashMap<String, Instant>) -> Option<String> {
    residents
        .iter()
        .min_by_key(|s| last_use.get(*s).copied())
        .cloned()
}

#[inline]
fn ms(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1000.0
}

#[inline]
fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_classifies_tensor_names() {
        let mut p = BlobPlan {
            weights_bytes: 0,
            kv_bytes: 0,
            other_bytes: 0,
        };
        p.add("model.layers.0.self_attn.q_proj.weight", 100);
        p.add("fp8/model.layers.0.mlp.down_proj", 50);
        p.add("kv.k.0", 30);
        p.add("kv.v.0", 30);
        p.add("in.ids", 4);
        p.add("act.logits", 16);
        p.add("moe.ewt.0", 8);
        assert_eq!(p.weights_bytes, 150);
        assert_eq!(p.kv_bytes, 60);
        assert_eq!(p.other_bytes, 28);
        assert_eq!(p.tensor_total(), 238);
    }

    /// The VRAM plan must count as WEIGHTS exactly what the loaders will demand of the
    /// checkpoint, and the loaders' predicate is `packet::names::is_checkpoint_weight`. The two
    /// cases this test exists for both used to land in `other_bytes` while the AMD loader
    /// uploaded them as weights — an under-count on the residency path, and on the CUDA loader a
    /// zero fill:
    ///
    ///  * `lm_head.weight`, which devgen declares at the top level for an untied head;
    ///  * a wrapper-prefixed tower (Kimi-K3 spells all 497 052 of its language-tower tensors
    ///    `language_model.model.…`, and none of them starts with `model.`).
    ///
    /// Asserted through the shared predicate rather than by re-spelling it, because a re-spelt
    /// copy is precisely how the five sites diverged in the first place.
    #[test]
    fn the_plan_counts_the_same_weights_the_loaders_bind() {
        let mut p = BlobPlan {
            weights_bytes: 0,
            kv_bytes: 0,
            other_bytes: 0,
        };
        let weights = [
            "lm_head.weight",
            "language_model.model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
            "language_model.lm_head.weight",
            "model.layers.0.mlp.down_proj.weight",
            "fp8/model.layers.0.mlp.down_proj.weight_scale",
        ];
        for n in weights {
            assert!(
                packet::names::is_checkpoint_weight(n),
                "{n} must bind from the checkpoint"
            );
            p.add(n, 10);
        }
        // Host-filled pointer tables live under the model prefix and are NOT weights — no
        // checkpoint contains them, and counting them as weights is how the CUDA loader used to
        // report `MISSING WEIGHT` for a GLM packet.
        for n in ["model.layers.3.mlp.expert_weight_table", "moe.ewt.3"] {
            assert!(!packet::names::is_checkpoint_weight(n), "{n}");
            p.add(n, 7);
        }
        p.add("kv.3.krot", 5);
        p.add("act.x", 3);
        assert_eq!(p.weights_bytes, 10 * weights.len() as u64);
        assert_eq!(p.kv_bytes, 5);
        assert_eq!(p.other_bytes, 7 * 2 + 3);
    }

    #[test]
    fn victim_is_least_recently_used() {
        let residents = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut lru = FxHashMap::default();
        let t0 = Instant::now();
        lru.insert("a".to_string(), t0);
        lru.insert("b".to_string(), t0 + std::time::Duration::from_secs(1));
        // c never used → evicted first (None sorts before Some).
        assert_eq!(pick_victim(&residents, &lru).as_deref(), Some("c"));
        lru.insert("c".to_string(), t0 + std::time::Duration::from_secs(2));
        assert_eq!(pick_victim(&residents, &lru).as_deref(), Some("a"));
        assert_eq!(pick_victim(&[], &lru), None);
    }
}
