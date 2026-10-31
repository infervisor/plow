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
//! (one `ModelMux` each), each with its own `EngineThread`, so two ticks do run
//! concurrently on two OS threads against the one shared `CUcontext`.
//!
//! This paragraph used to claim they were ordered by the NULL stream. They are
//! not: `GpuEngine` owns a private stream created `CU_STREAM_NON_BLOCKING`
//! (`exec/gpu.rs`, `device/cuda.rs` — "work here never implicitly orders
//! against the legacy default stream"), and every serving launch goes to it.
//! Nothing at the driver level orders one model's launch against another's.
//!
//! It is nonetheless safe, for a different reason: **every mutable device
//! buffer is per-engine** — instruction stream, tensor tables, counter block,
//! GQ cursor, pinned staging, kernarg, graph caches, the cuBLASLt handle and
//! its workspace, and the VMM KV slab. Each engine even loads its OWN
//! `CUmodule` (the backend allocates a fresh id per load and never dedups by
//! image), so the `__device__` trace globals are per-engine too. Addresses are
//! driver-assigned (`cuMemAddressReserve` with a null hint) and the blob binds
//! tensors by handle, never by absolute pointer, so two resident models cannot
//! alias by construction. No lock is needed for MEMORY safety.
//!
//! What co-residency does cost is throughput, not correctness: the interpreter
//! grid is `occupancy × sm_count`, i.e. the whole device, and a cooperative
//! launch is all-blocks-co-resident-or-fail. A second model's grid is therefore
//! admitted only once the first VACATES — which is why segment boundaries are
//! the switch points, and why a long unsegmented prefill blocks a co-tenant for
//! its full duration.
//!
//! **Admission.** Resident models shed on their own KV pools / slot tables as
//! before. A non-resident model's request first passes [`ModelManager::
//! ensure_resident`]: switches serialize on one async lock (concurrent
//! requests for the same target coalesce); if eviction cannot make the target
//! fit, the request is shed with [`EnsureError::WontFit`] (503 + Retry-After
//! at the HTTP layer). A switch waits for the victim's in-flight generations
//! to complete (mux drain); with `PLOW_DRAIN_TIMEOUT_MS` set, a drain past
//! the deadline preempts the victim's live generations instead
//! (`FinishReason::Preempted`, queued jobs rejected with a retryable 429).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashMap;

use crate::asset::devblob::DevBlob;
#[cfg(feature = "cuda")]
use crate::device::cuda::CudaBackend;
use crate::memory::vmm::VmmOps;
use crate::serve::mux::{self, MuxConfig};
use crate::serve::AppState;
use crate::{Result, RuntimeError};

/// Planner headroom kept free on top of a model's requirement (driver slack,
/// transient staging).
pub const RESERVE: u64 = 256 << 20;

/// The startup refusal under `--pin-resident` for a model the planner cannot fit.
fn pinned_misfit(slug: &str, need: u64, free: u64) -> String {
    format!(
        "planner: {slug} needs {:.1} GiB and {:.1} GiB is left; --pin-resident keeps every model \
         resident, so all of them must fit (narrow them with --live-ctx-models or \
         --vmm-live-rings-models, or serve fewer models)",
        gib(need),
        gib(free)
    )
}

/// Run a switch's blocking step on a thread of its own: the blocking pool can be saturated by
/// the very request load that needs this switch to finish.
async fn on_own_thread<T: Send + 'static>(
    name: &str,
    f: impl FnOnce() -> T + Send + 'static,
) -> Result<T> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let _ = tx.send(f());
        })
        .map_err(|e| RuntimeError::Msg(format!("{name} thread: {e}")))?;
    rx.await
        .map_err(|_| RuntimeError::Msg(format!("{name} thread panicked")))
}

/// A load or bind that failed for device memory, however the layer below spelled it.
fn is_oom(e: &RuntimeError) -> bool {
    matches!(e, RuntimeError::Oom(_))
        || e.device_code() == Some(2)
        || e.to_string().contains("OUT_OF_MEMORY")
}

/// A model that received a request this recently is not idle.
const IDLE_GRACE: Duration = Duration::from_secs(2);

/// How long an eviction waits for the evicted model's sibling packet runtimes to free.
const SIDECAR_RELEASE_WAIT: Duration = Duration::from_secs(60);

/// Non-tensor overhead assumed before the first load measures the real value.
pub const DEFAULT_OVERHEAD: u64 = 512 << 20;

/// Tensor-parallel fan-out declared by a bundle's blob, or 1 when it declares
/// none. Read before any device is opened, so placement can size groups.
pub fn tp_degree(dir: &Path) -> Result<u32> {
    let Some(pkt) = DevBlob::find_in_dir(dir)? else {
        return Ok(1);
    };
    let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io {
        path: pkt.clone(),
        source,
    })?;
    // METADATA ONLY, as in `BlobPlan::from_dir_with_granularity` — this reads
    // the TP header and never dispatches, so the L2-placement guard (which is
    // the engine's job, against the code object) must not reject it here.
    Ok(DevBlob::parse_l2(&raw, true)?
        .tp
        .map(|t| t.n_gpu)
        .unwrap_or(1)
        .max(1))
}

/// Per-model VRAM plan, derived from the blob header (sizes only — the blob
/// carries no addresses).
#[derive(Clone, Copy, Debug)]
pub struct BlobPlan {
    /// Checkpoint weights (`model.*`, `fp8/*`).
    pub weights_bytes: u64,
    /// KV bytes (`kv.*`): compiled extent, or resident startup bytes in a live device plan.
    pub kv_bytes: u64,
    /// Activations, IO, MoE tables — every other blob tensor.
    pub other_bytes: u64,
    /// Speech pipeline sibling packets, allocated when the pipeline binds after the engine load.
    pub sidecar_bytes: u64,
}

impl BlobPlan {
    /// Sum of the planned tensor bytes.
    pub fn tensor_total(&self) -> u64 {
        self.weights_bytes + self.kv_bytes + self.other_bytes + self.sidecar_bytes
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
        Self::from_dir_with_device(dir, None, None)
    }

    pub fn from_dir_with_granularity(dir: &Path, granularity: Option<u64>) -> Result<BlobPlan> {
        Self::from_dir_with_device(dir, granularity.map(|g| (g, (0, 0))), None)
    }

    pub fn from_dir_with_device(
        dir: &Path,
        device_vmm: Option<(u64, (u32, u32))>,
        checkpoint: Option<&Path>,
    ) -> Result<BlobPlan> {
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
        let mut blob = DevBlob::parse_l2(&raw, true)?;
        // Plan the bound the engine will load at (`GpuEngine::load` narrows the same way).
        let narrowed = match crate::config::RuntimeConfig::get().live_ctx_for(dir) {
            Some(want) if crate::exec::ctx_bound::declared_ctx(&blob).is_some_and(|c| want < c) => {
                crate::exec::ctx_bound::narrow(&mut blob, want)?;
                true
            }
            _ => false,
        };
        let mut plan = BlobPlan {
            weights_bytes: 0,
            kv_bytes: 0,
            other_bytes: 0,
            sidecar_bytes: sidecar_bytes(&pkt)?,
        };
        for t in &blob.tensors {
            plan.add(&t.name, t.bytes);
        }
        #[cfg(feature = "cuda")]
        if let Some((granularity, capability)) = device_vmm {
            let config = crate::config::RuntimeConfig::get();
            let manifest = match narrowed {
                true => blob.with_packet_view(plow_asset::live_kv::emit).ok(),
                false => crate::memory::vmm::LiveKvLayout::manifest(&blob, &raw)?,
            };
            let packed = blob
                .reserved_metadata(&raw, plow_asset::packed_prefill::SECTION)?
                .is_some();
            let full = manifest
                .as_ref()
                .is_some_and(|m| m.caches.iter().any(|c| c.window == 0));
            let kv = match checkpoint {
                Some(path) => crate::asset::serve::resolve(dir, path)?.manifest.kv,
                None => None,
            };
            let prefix_layout = checkpoint.and_then(|_| {
                crate::exec::gpu::GpuEngine::select_vmm_prefix_layout(
                    &blob,
                    kv.as_ref(),
                    config,
                    capability,
                    granularity,
                    manifest
                        .as_ref()
                        .map_or(Some(false), |m| m.full_rows_interleaved()),
                )
            });
            let prefix_requested = config.nv_prefix_requested(prefix_layout.is_some());
            if config.nv_live_kv_enabled(packed, full, prefix_requested)
                && (config.nv_vmm_live()
                    || crate::exec::gpu::live_kv_mappable(&blob, manifest.as_ref(), granularity))
            {
                let layout = match manifest.as_ref() {
                    Some(m) => crate::memory::vmm::LiveKvLayout::from_manifest(&blob, m)?,
                    None => crate::memory::vmm::LiveKvLayout::from_blob(&blob)?,
                };
                let virtual_bytes: u64 = layout
                    .full_tensors
                    .iter()
                    .flatten()
                    .map(|&id| blob.tensors[id].bytes)
                    .sum();
                let geo = &layout.geometry;
                let hint = u64::from(config.vmm_block_mib()) << 20;
                if let Some(full) = layout.slot_granular_full(&blob, granularity, hint) {
                    // Slots commit on admission (`kv_row_charge`); plan the shared idle backing,
                    // one admitted slot and the recycled-unit pool.
                    let tensors: Vec<_> = full.into_iter().chain(layout.ring_tensors).collect();
                    let virtual_bytes: u64 =
                        tensors.iter().map(|t| blob.tensors[t.tensor].bytes).sum();
                    let slot: u64 = tensors
                        .iter()
                        .filter(|t| t.slot_bytes % granularity == 0)
                        .map(|t| t.slot_bytes)
                        .sum();
                    let startup = crate::memory::vmm::VmmRings::idle_backed_startup_bytes(
                        &tensors,
                        geo.batch as usize,
                        granularity,
                    );
                    plan.kv_bytes = plan.kv_bytes.checked_sub(virtual_bytes).ok_or_else(|| {
                        RuntimeError::Rejected("live KV plan tensor classification".into())
                    })? + startup
                        + slot
                        + crate::memory::vmm::kv_pool_cap();
                    return Ok(plan);
                }
                let live_rings = crate::exec::gpu::live_rings_for_capacity(
                    config.nv_vmm_live_rings_for(dir),
                    true,
                    Some(geo.max_ctx),
                    Some(geo.batch),
                );
                let ring_virtual_bytes: u64 = live_rings
                    .then(|| {
                        layout
                            .ring_tensors
                            .iter()
                            .map(|ring| blob.tensors[ring.tensor].bytes)
                            .sum()
                    })
                    .unwrap_or(0);
                let block =
                    geo.block_bytes(granularity, u64::from(config.vmm_block_mib()) << 20)?;
                // Load maps row zero for every slot, including idle decode lanes.
                let resident = block
                    * u64::from(geo.batch)
                    * u64::from(geo.kvh_full)
                    * layout.full_tensors.len() as u64
                    * 2;
                let demand_virtual_bytes = virtual_bytes
                    .checked_add(ring_virtual_bytes)
                    .ok_or_else(|| {
                        RuntimeError::Rejected("live KV virtual byte overflow".into())
                    })?;
                plan.kv_bytes =
                    plan.kv_bytes
                        .checked_sub(demand_virtual_bytes)
                        .ok_or_else(|| {
                            RuntimeError::Rejected("live KV plan tensor classification".into())
                        })?
                        + resident
                        + crate::memory::vmm::kv_pool_cap();
            } else if let Some(layout) = prefix_layout {
                let geo = &layout.geo;
                geo.block_bytes(granularity, u64::from(config.vmm_block_mib()) << 20)?;
                let tracks = geo.full_layers.len() as u64 * 2;
                let ring_bytes = match config.nv_vmm_live_rings_for(dir) {
                    Some(true) => layout.slide_kv_bytes(&blob),
                    _ => 0,
                };
                let virtual_bytes = geo.full_tensor_bytes() * tracks + ring_bytes;
                // Prefix slots start unmapped (`VmmKv::new`) and map on admission against the KV
                // budget; at load only the block pool's precreated columns commit, inside
                // `kv_pool_cap`. Charging a block per slot and head (4 GiB on Gemma-4 E4B, which
                // measures 3.6 GiB under that plan) kept E4B from fitting beside ASR and TTS.
                plan.kv_bytes = plan.kv_bytes.checked_sub(virtual_bytes).ok_or_else(|| {
                    RuntimeError::Rejected("prefix KV plan tensor classification".into())
                })? + crate::memory::vmm::kv_pool_cap();
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (device_vmm, checkpoint, narrowed);
        Ok(plan)
    }
}

const GRAPH_BYTES_PER_PROGRAM: u64 = 320 << 10;

/// Device bytes of the speech pipeline's sibling packets beside `pkt` (Qwen3-ASR `encoder.pkt`,
/// codec `codec.pkt`, Chatterbox `s3gen.pkt`), bound after the engine load. Each binds a CUDA
/// packet runtime (`CudaPacketRuntime::load_on`) that `cuMemAlloc`s every packet tensor
/// (activation and batch buffers at their compiled capacity), one counter slab and eight
/// uploads per program, then captures every program sequence as a CUDA graph.
fn sidecar_bytes(pkt: &Path) -> Result<u64> {
    // Allocations of 1 MiB and up take whole 2 MiB pages.
    let alloc = |bytes: u64| match bytes.max(4) {
        b if b >= 1 << 20 => b.next_multiple_of(2 << 20),
        b => b.next_multiple_of(512),
    };
    let asset = crate::exec::packet_runtime::PacketAsset::load_if_present(pkt)?;
    let mut total = 0u64;
    for (key, legacy) in [
        ("encoder.packet", "encoder.pkt"),
        ("codec.packet", "codec.pkt"),
        ("vocoder.packet", "s3gen.pkt"),
    ] {
        let name = match &asset {
            Some(asset) => asset.stage_file(key, legacy)?,
            None => legacy.to_string(),
        };
        let path = pkt.with_file_name(name);
        if !path.is_file() {
            continue;
        }
        let raw = std::fs::read(&path).map_err(|source| RuntimeError::Io {
            path: path.clone(),
            source,
        })?;
        let blob = DevBlob::parse_l2(&raw, true)?;
        let ctr = u64::from(packet::dev::CTR_STRIDE) * 4;
        let mut counters = 0u64;
        for p in &blob.progs {
            let segments = p.gq_seg_ofs.len().saturating_sub(1) as u64;
            counters +=
                (u64::from(p.n_counter) * ctr).max(4).next_multiple_of(256) + segments * ctr;
            total += [
                std::mem::size_of_val(&p.insts[..]),
                std::mem::size_of_val(&p.stream[..]),
                std::mem::size_of_val(&p.stream_ofs[..]),
                std::mem::size_of_val(&p.stream_len[..]),
                std::mem::size_of_val(&p.waits[..]),
                std::mem::size_of_val(&p.succs[..]),
                std::mem::size_of_val(&p.gq_stream[..]),
                std::mem::size_of_val(&p.gq_seg_ofs[..]),
            ]
            .into_iter()
            .map(|b| alloc(b as u64))
            .sum::<u64>();
        }
        // Graph capture at bind: S3Gen 602 MiB over 1931 programs, Qwen3-ASR encoder ~65 MiB
        // over 460 (L40S).
        total += blob.progs.len() as u64 * GRAPH_BYTES_PER_PROGRAM;
        total += blob.tensors.iter().map(|t| alloc(t.bytes)).sum::<u64>()
            + alloc(blob.tensors.len() as u64 * 8)
            + alloc(counters);
    }
    Ok(total)
}

/// One registered (not necessarily resident) model.
#[derive(Clone)]
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

/// How an eviction ends the victim's in-flight generations.
#[derive(Clone, Copy, Debug)]
enum Stop {
    /// Let live slots run to completion, bounded by `PLOW_DRAIN_TIMEOUT_MS`
    /// when it is set. The S1 switch path.
    Graceful,
    /// Stop at the next tick top and flush what each slot produced. The
    /// control-plane unload path.
    Now,
    /// Let live slots finish for up to this long, then stop the rest: a DP rank's unload, whose
    /// new requests already route to its siblings.
    Bounded(Duration),
}

/// What one control-plane unload did.
#[derive(Clone, Debug, Default)]
pub struct UnloadReport {
    pub slug: String,
    /// Time spent stopping and flushing in-flight generations.
    pub stop_ms: f64,
    /// Time spent dropping the engine.
    pub unload_ms: f64,
    /// Bytes released from the slab chunk pool by the trim.
    pub pool_trimmed: u64,
    /// Device free-memory delta across the whole operation.
    pub freed: u64,
}

/// Why `ensure_resident` refused.
#[derive(Debug)]
pub enum EnsureError {
    /// Even after evicting every other resident model the target cannot fit.
    /// Shed with 503 + Retry-After.
    WontFit { need: u64, free: u64 },
    /// The operator unloaded this model (or is unloading it). Demand loading is
    /// deliberately NOT attempted — otherwise the next request would silently
    /// undo the unload. Cleared by an explicit load.
    Unloaded,
    /// The engine load (or drain plumbing) failed.
    Load(RuntimeError),
    /// Another switch held the switch lock past `PLOW_SWITCH_TIMEOUT_MS`. Shed with 503 +
    /// Retry-After.
    SwitchTimeout(Duration),
    /// The model's last load failed and its retry backoff has not expired (its bundle files are
    /// unchanged). Shed with 503 + Retry-After `retry_after`.
    Backoff { why: String, retry_after: Duration },
}

impl EnsureError {
    /// `r` with this refusal's Retry-After header, when it has one.
    pub fn with_retry_after(&self, mut r: axum::response::Response) -> axum::response::Response {
        if let Some(d) = self.retry_after() {
            r.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from(d.as_secs().max(1)),
            );
        }
        r
    }

    /// Retry-After for a shed request, `None` when retrying unchanged cannot help.
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            EnsureError::WontFit { .. } | EnsureError::SwitchTimeout(_) => {
                Some(Duration::from_secs(30))
            }
            EnsureError::Backoff { retry_after, .. } => Some(*retry_after),
            EnsureError::Unloaded | EnsureError::Load(_) => None,
        }
    }
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
            EnsureError::Unloaded => write!(
                f,
                "model was unloaded by the operator — load it again to serve it"
            ),
            EnsureError::Load(e) => write!(f, "model load failed: {e}"),
            EnsureError::SwitchTimeout(d) => write!(
                f,
                "another model switch is still in progress after {} s; retry later",
                d.as_secs()
            ),
            EnsureError::Backoff { why, retry_after } => write!(
                f,
                "model load failed: {why}; next attempt in {} s",
                retry_after.as_secs().max(1)
            ),
        }
    }
}

/// The device abstraction required by [`ModelManager`].
pub trait ManagerDevice: Send + Sync {
    /// Device ordinal (GPU index).
    fn device_ordinal(&self) -> u8;
    /// Available and total device memory in bytes.
    fn mem_info(&self) -> Result<(u64, u64)>;
    /// Allocation granularity in bytes.
    fn granularity(&self) -> Result<u64>;
    /// Compute capability `(major, minor)`, or `None` if non-CUDA.
    fn compute_capability(&self) -> Option<(u32, u32)>;
    /// Bytes in physical chunk pool.
    fn pool_bytes(&self) -> u64;
    /// Trim chunk pool to `keep_bytes`.
    fn pool_trim(&self, keep_bytes: u64) -> u64;
    /// Load an engine for `dir` and `ckpt`.
    fn load_engine(&self, dir: &Path, ckpt: &Path) -> Result<crate::serve::engine::ServeEngine>;
    /// Check whether sibling packets are resident.
    fn is_sibling_resident(&self, dir: &Path) -> bool;
    /// Whether this device backend supports tensor parallelism.
    fn supports_tp(&self) -> bool {
        false
    }
    /// Modules loaded on the device, when the backend counts them (lifecycle accounting).
    fn live_modules(&self) -> Option<usize> {
        None
    }
    /// Devices in this manager's group, from its first ordinal: the widest TP it can host.
    fn group_width(&self) -> u32 {
        1
    }
    /// Co-resident engines here are safe only under round-robin turns (HSA: separate queues
    /// give no whole-grid residency, and no cooperative-launch refusal turns that into an error).
    fn co_tenants_need_rr(&self) -> bool {
        false
    }
}

#[cfg(feature = "cuda")]
pub struct CudaManagerDevice {
    be: Arc<CudaBackend>,
}

#[cfg(feature = "cuda")]
impl CudaManagerDevice {
    pub fn new(be: Arc<CudaBackend>) -> Self {
        Self { be }
    }
}

#[cfg(feature = "cuda")]
impl ManagerDevice for CudaManagerDevice {
    fn device_ordinal(&self) -> u8 {
        self.be.device_ordinal
    }
    fn mem_info(&self) -> Result<(u64, u64)> {
        self.be.mem_info()
    }
    fn granularity(&self) -> Result<u64> {
        self.be.granularity()
    }
    fn compute_capability(&self) -> Option<(u32, u32)> {
        Some(self.be.compute_capability())
    }
    fn pool_bytes(&self) -> u64 {
        VmmOps::pool_bytes(&*self.be)
    }
    fn pool_trim(&self, keep_bytes: u64) -> u64 {
        VmmOps::pool_trim(&*self.be, keep_bytes)
    }
    fn load_engine(&self, dir: &Path, ckpt: &Path) -> Result<crate::serve::engine::ServeEngine> {
        let be = Arc::clone(&self.be);
        let mut engine = self
            .be
            .unload_modules_on_err(|| crate::exec::gpu::GpuEngine::load(be, dir, ckpt))?;
        if crate::config::RuntimeConfig::get().nv.spec {
            match engine.spec_serve_enable(dir) {
                Ok(limits) => tracing::info!(?limits, "speculative decoding on"),
                Err(e) => {
                    tracing::warn!(error = %e, "PLOW_SPEC set but speculative decoding unavailable")
                }
            }
        }
        Ok(crate::serve::engine::ServeEngine::Cuda(engine))
    }
    fn is_sibling_resident(&self, dir: &Path) -> bool {
        crate::exec::gpu::packet_exec::resident_in(dir, self.be.device_ordinal)
    }
    fn live_modules(&self) -> Option<usize> {
        Some(self.be.live_modules())
    }
}

#[cfg(feature = "hsa")]
pub struct HsaManagerDevice {
    be: Arc<crate::device::hsa::HsaBackend>,
    width: u32,
}

#[cfg(feature = "hsa")]
impl HsaManagerDevice {
    /// `width`: devices in the group starting at `be`'s ordinal.
    pub fn new(be: Arc<crate::device::hsa::HsaBackend>, width: u32) -> Self {
        Self { be, width }
    }
}

#[cfg(feature = "hsa")]
impl ManagerDevice for HsaManagerDevice {
    fn device_ordinal(&self) -> u8 {
        self.be.device_ordinal
    }
    fn mem_info(&self) -> Result<(u64, u64)> {
        let free = self.be.free_bytes().unwrap_or(0);
        let total = self.be.vram_bytes();
        Ok((free, total))
    }
    fn granularity(&self) -> Result<u64> {
        self.be.granularity()
    }
    fn compute_capability(&self) -> Option<(u32, u32)> {
        None
    }
    fn pool_bytes(&self) -> u64 {
        VmmOps::pool_bytes(&*self.be)
    }
    fn pool_trim(&self, keep_bytes: u64) -> u64 {
        VmmOps::pool_trim(&*self.be, keep_bytes)
    }
    fn load_engine(&self, dir: &Path, ckpt: &Path) -> Result<crate::serve::engine::ServeEngine> {
        let blob = crate::asset::devblob::DevBlob::find_in_dir(dir)?
            .ok_or_else(|| RuntimeError::Msg(format!("no blob in {}", dir.display())))?;
        let hsaco = crate::config::RuntimeConfig::get()
            .amd
            .hsaco
            .clone()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let own = dir.join("hsaco");
                if own.is_dir() {
                    own
                } else {
                    dir.to_path_buf()
                }
            });
        let first = self.be.device_ordinal;
        let engine = crate::serve::engine::AmdServe::load_on(&blob, &hsaco, Some(ckpt), first)?;
        Ok(crate::serve::engine::ServeEngine::Amd(engine))
    }
    fn is_sibling_resident(&self, _dir: &Path) -> bool {
        false
    }
    fn supports_tp(&self) -> bool {
        true
    }
    fn group_width(&self) -> u32 {
        self.width
    }
    fn co_tenants_need_rr(&self) -> bool {
        true
    }
}

pub struct HostManagerDevice {
    ordinal: u8,
}

impl HostManagerDevice {
    pub fn new() -> Self {
        Self { ordinal: 0 }
    }

    pub fn with_ordinal(ordinal: u8) -> Self {
        Self { ordinal }
    }
}

impl Default for HostManagerDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl ManagerDevice for HostManagerDevice {
    fn device_ordinal(&self) -> u8 {
        self.ordinal
    }
    fn mem_info(&self) -> Result<(u64, u64)> {
        Ok((32 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024))
    }
    fn granularity(&self) -> Result<u64> {
        Ok(4096)
    }
    fn compute_capability(&self) -> Option<(u32, u32)> {
        None
    }
    fn pool_bytes(&self) -> u64 {
        0
    }
    fn pool_trim(&self, _keep_bytes: u64) -> u64 {
        0
    }
    fn load_engine(&self, _dir: &Path, _ckpt: &Path) -> Result<crate::serve::engine::ServeEngine> {
        Err(RuntimeError::Rejected(
            "Host/CPU dynamic engine loading not yet supported".into(),
        ))
    }
    fn is_sibling_resident(&self, _dir: &Path) -> bool {
        false
    }
}

/// The S1 model manager: residency, planner, switch orchestration.
pub struct ModelManager {
    /// `Weak` breaks the `AppState ↔ manager` cycle; the state outlives every
    /// request, so upgrades only fail during process teardown.
    state: Weak<AppState>,
    device: Arc<dyn ManagerDevice>,
    mux_cfg: MuxConfig,
    /// Registered models. Behind a lock because the control plane registers
    /// new assets dirs while the server is live; cloned out of the lock at every
    /// use, since every caller then awaits a load.
    models: RwLock<Vec<Managed>>,
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
    /// A growth eviction ([`Self::request_growth`]) is in flight.
    growing: std::sync::atomic::AtomicBool,
    /// The server runtime. A dispatcher asking for growth runs its own executor on a busy
    /// thread, where a task spawned locally would wait for the load to stop.
    runtime: Option<tokio::runtime::Handle>,
    /// slug → its last failed load (other than out of memory). Demand loads back off before
    /// evicting anyone: a deterministic failure otherwise took a healthy co-tenant down on every
    /// request. A success, an explicit load or deregistration clears it.
    failed: Mutex<FxHashMap<String, LoadFailure>>,
}

/// A failed load and when the next demand load may retry it.
#[derive(Clone, Debug)]
struct LoadFailure {
    why: String,
    at: Instant,
    delay: Duration,
    /// [`bundle_stamp`] at the failure: a changed bundle retries at once.
    stamp: u64,
}

const LOAD_BACKOFF_MIN: Duration = Duration::from_secs(30);
const LOAD_BACKOFF_MAX: Duration = Duration::from_secs(600);

/// Name, size and mtime of every file directly in the bundle and checkpoint dirs, hashed: a
/// bundle being copied or replaced changes it.
fn bundle_stamp(m: &Managed) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for dir in [&m.dir, &m.ckpt] {
        let Ok(rd) = std::fs::read_dir(dir) else {
            continue;
        };
        let mut files: Vec<_> = rd
            .flatten()
            .filter_map(|e| {
                let md = e.metadata().ok()?;
                Some((e.file_name(), md.len(), md.modified().ok()))
            })
            .collect();
        files.sort();
        files.hash(&mut h);
    }
    h.finish()
}

impl ModelManager {
    /// Build the manager for `(slug, assets dir, checkpoint dir)` triples —
    /// every registered bundle that ships a PLOWDEV blob. Parses each blob's
    /// plan up front (startup-only file reads).
    #[cfg(feature = "cuda")]
    pub fn new(
        be: Arc<CudaBackend>,
        state: &Arc<AppState>,
        mux_cfg: MuxConfig,
        models: Vec<(String, PathBuf, PathBuf)>,
        budget: Option<u64>,
    ) -> Result<ModelManager> {
        Self::with_device(
            Arc::new(CudaManagerDevice::new(be)),
            state,
            mux_cfg,
            models,
            budget,
        )
    }

    #[cfg(feature = "hsa")]
    pub fn from_hsa(
        be: Arc<crate::device::hsa::HsaBackend>,
        width: u32,
        state: &Arc<AppState>,
        mux_cfg: MuxConfig,
        models: Vec<(String, PathBuf, PathBuf)>,
        budget: Option<u64>,
    ) -> Result<ModelManager> {
        Self::with_device(
            Arc::new(HsaManagerDevice::new(be, width)),
            state,
            mux_cfg,
            models,
            budget,
        )
    }

    /// Build the manager over any device implementing [`ManagerDevice`].
    pub fn with_device(
        device: Arc<dyn ManagerDevice>,
        state: &Arc<AppState>,
        mux_cfg: MuxConfig,
        models: Vec<(String, PathBuf, PathBuf)>,
        budget: Option<u64>,
    ) -> Result<ModelManager> {
        let mut managed = Vec::with_capacity(models.len());
        for (slug, dir, ckpt) in models {
            check_device_tp(&*device, &slug, &dir)?;
            let vmm = device
                .compute_capability()
                .and_then(|cap| device.granularity().ok().map(|g| (g, cap)));
            let cfg = crate::config::bundle_config(&dir)?;
            let plan = crate::config::scoped(Some(cfg), || {
                BlobPlan::from_dir_with_device(&dir, vmm, Some(&ckpt))
            })?;
            tracing::info!(
                %slug,
                weights_gib = gib(plan.weights_bytes),
                kv_gib = gib(plan.kv_bytes),
                other_gib = gib(plan.other_bytes),
                sidecar_gib = gib(plan.sidecar_bytes),
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
            device,
            mux_cfg,
            models: RwLock::new(managed),
            switch: tokio::sync::Mutex::new(()),
            last_use: Mutex::new(FxHashMap::default()),
            overhead: Mutex::new(FxHashMap::default()),
            budget,
            last_switch: Mutex::new(None),
            growing: std::sync::atomic::AtomicBool::new(false),
            failed: Mutex::default(),
            runtime: tokio::runtime::Handle::try_current().ok(),
        })
    }

    /// Whether `slug` is under manager control (has a blob plan).
    pub fn manages(&self, slug: &str) -> bool {
        self.models.read().iter().any(|m| m.slug == slug)
    }

    /// Clone `slug`'s record out from under the lock. Every caller goes on to
    /// await a load, and the lock must not be held across an await.
    fn managed(&self, slug: &str) -> Option<Managed> {
        self.models.read().iter().find(|m| m.slug == slug).cloned()
    }

    /// Register a model the process was not started with, so a later `load`
    /// can bring it up. Parses the blob plan up front (the same startup-only
    /// file read `new` does) so a bad bundle is refused here rather than
    /// halfway through a load. Idempotent for an already-registered slug.
    pub fn register(&self, slug: &str, dir: PathBuf, ckpt: PathBuf) -> Result<()> {
        if let Some(existing) = self.managed(slug) {
            // Idempotent for the SAME bundle, refused for a different one.
            // Returning Ok here regardless meant a re-register with new assets
            // kept the old dir, checkpoint and memory plan while the registry
            // took the new bundle — a new tokenizer driving old weights, which
            // produces fluent wrong output rather than an error.
            if same_path(&existing.dir, &dir) && same_path(&existing.ckpt, &ckpt) {
                return Ok(());
            }
            return Err(RuntimeError::Rejected(format!(
                "{slug} is already registered from {} (checkpoint {}); unload it with \
                 \"deregister\": true before registering {} in its place",
                existing.dir.display(),
                existing.ckpt.display(),
                dir.display()
            )));
        }
        check_device_tp(&*self.device, slug, &dir)?;
        let vmm = self
            .device
            .compute_capability()
            .and_then(|cap| self.device.granularity().ok().map(|g| (g, cap)));
        let cfg = crate::config::bundle_config(&dir)?;
        let plan = crate::config::scoped(Some(cfg), || {
            BlobPlan::from_dir_with_device(&dir, vmm, Some(&ckpt))
        })?;
        tracing::info!(
            %slug,
            dir = %dir.display(),
            total_gib = gib(plan.tensor_total()),
            "planner: model registered at runtime"
        );
        self.models.write().push(Managed {
            slug: slug.to_string(),
            dir,
            ckpt,
            plan,
        });
        Ok(())
    }

    /// Drop `slug` from this manager entirely: its plan, its measured
    /// overhead and its LRU position.
    ///
    /// The engine and mux must already be gone (`unload`). Without this, a
    /// deregistered slug kept its `Managed` record, so re-registering it with a
    /// different assets dir silently reused the old checkpoint, packet path and
    /// memory plan. Registration and deregistration are one lifecycle.
    fn deregister(&self, slug: &str) -> bool {
        let mut models = self.models.write();
        let Some(i) = models.iter().position(|m| m.slug == slug) else {
            return false;
        };
        models.remove(i);
        drop(models);
        self.overhead.lock().remove(slug);
        self.last_use.lock().remove(slug);
        self.failed.lock().remove(slug);
        tracing::info!(%slug, "planner: model deregistered");
        true
    }

    /// Every registered slug, in registration order.
    pub fn slugs(&self) -> Vec<String> {
        self.models.read().iter().map(|m| m.slug.clone()).collect()
    }

    /// The device this manager's engines and their sibling packets run on.
    pub fn ordinal(&self) -> u8 {
        self.device.device_ordinal()
    }

    /// The device ordinals this manager's group covers. One today (a manager
    /// owns one backend); a tensor-parallel group would report its whole run.
    pub fn ordinals(&self) -> Vec<u32> {
        vec![u32::from(self.device.device_ordinal())]
    }

    /// Device free/total bytes, for capacity reporting on the control plane.
    pub fn device_mem_info(&self) -> Result<(u64, u64)> {
        self.device.mem_info()
    }

    /// The parsed plan for `slug` (tests / capacity reporting).
    pub fn plan(&self, slug: &str) -> Option<BlobPlan> {
        self.models
            .read()
            .iter()
            .find(|m| m.slug == slug)
            .map(|m| m.plan)
    }

    /// Planner requirement for `slug`: tensor total + (measured|default)
    /// overhead. `None` for unmanaged slugs.
    pub fn required(&self, slug: &str) -> Option<u64> {
        let m = self.managed(slug)?;
        let ovh = self
            .overhead
            .lock()
            .get(slug)
            .copied()
            .unwrap_or(DEFAULT_OVERHEAD);
        Some(m.plan.tensor_total() + ovh)
    }

    /// Least-recently-used resident model other than `target`, idle ones first.
    fn lru_victim(&self, target: &str) -> Option<String> {
        if crate::config::RuntimeConfig::get().nv.pin_resident {
            return None;
        }
        let victims: Vec<String> = self
            .models
            .read()
            .iter()
            .filter(|v| v.slug != target && self.is_resident(&v.slug))
            .map(|v| v.slug.clone())
            .collect();
        let idle: Vec<String> = victims
            .iter()
            .filter(|v| self.is_idle(v))
            .cloned()
            .collect();
        let last_use = self.last_use.lock();
        pick_victim(&idle, &last_use).or_else(|| pick_victim(&victims, &last_use))
    }

    /// No request queued, tokenizing, in an engine slot or in a downstream stage, and none
    /// arrived for [`IDLE_GRACE`] (an ASR request encodes before it reaches the mux).
    fn is_idle(&self, slug: &str) -> bool {
        let Some(state) = self.state.upgrade() else {
            return false;
        };
        let recent = self
            .last_use
            .lock()
            .get(slug)
            .is_some_and(|t| t.elapsed() < IDLE_GRACE);
        !recent
            && state.mux(slug).is_none_or(|m| !m.in_flight())
            && state.downstream(slug).pending() == 0
    }

    /// `slug` is short of device memory for its KV: evict one idle resident model (never one
    /// with work in flight) so it can grow toward its full ladder. Coalesced: one at a time, and
    /// skipped while a switch holds the lock (that switch changes residency anyway).
    pub fn request_growth(self: &Arc<Self>, slug: &str) {
        use std::sync::atomic::Ordering;
        if crate::config::RuntimeConfig::get().nv.pin_resident
            || self.growing.swap(true, Ordering::AcqRel)
        {
            return;
        }
        let Some(runtime) = self.runtime.clone() else {
            self.growing.store(false, Ordering::Release);
            return;
        };
        let mgr = Arc::clone(self);
        let slug = slug.to_string();
        runtime.spawn(async move {
            let Ok(_g) = mgr.switch.try_lock() else {
                tracing::debug!(%slug, "planner: growth skipped — a switch is in progress");
                mgr.growing.store(false, Ordering::Release);
                return;
            };
            let idle: Vec<String> = mgr
                .models
                .read()
                .iter()
                .filter(|v| v.slug != slug && mgr.is_serving(&v.slug))
                .map(|v| v.slug.clone())
                .filter(|v| mgr.is_idle(v))
                .collect();
            let victim = pick_victim(&idle, &mgr.last_use.lock());
            match victim {
                Some(victim) => {
                    tracing::info!(%slug, %victim, "planner: evicting an idle model so a busy one can grow");
                    if let Err(e) = mgr.evict(&victim, Stop::Graceful).await {
                        tracing::warn!(%victim, error = %e, "planner: growth eviction failed");
                    }
                }
                None => tracing::debug!(%slug, "planner: growth found no idle model to evict"),
            }
            mgr.growing.store(false, Ordering::Release);
        });
    }

    fn touch(&self, slug: &str) {
        self.last_use
            .lock()
            .insert(slug.to_string(), Instant::now());
    }

    /// Effective free VRAM under the optional budget cap: with a budget B on a
    /// card of total T, pretend the card has only B — free' = free − (T − B).
    fn free_vram(&self) -> Result<u64> {
        let (free, total) = self.device.mem_info()?;
        Ok(match self.budget {
            Some(b) if b < total => free.saturating_sub(total - b),
            _ => free,
        })
    }

    /// Bytes a load may plan on: free VRAM plus `credit` (reusable pooled chunks), but never
    /// what resident models are planned to still commit (live KV slots, KV pools).
    fn fit_capacity(&self, credit: u64) -> Result<u64> {
        Ok((self.free_vram()? + credit).min(self.planned_headroom()?))
    }

    /// The device (or budget) less every resident model's planned requirement.
    fn planned_headroom(&self) -> Result<u64> {
        let (_, total) = self.device.mem_info()?;
        let cap = self.budget.map_or(total, |b| b.min(total));
        let overhead = self.overhead.lock();
        let planned: u64 = self
            .models
            .read()
            .iter()
            .filter(|m| self.is_resident(&m.slug))
            .map(|m| {
                m.plan.tensor_total() + overhead.get(&m.slug).copied().unwrap_or(DEFAULT_OVERHEAD)
            })
            .sum();
        Ok(cap.saturating_sub(planned))
    }

    fn state(&self) -> std::result::Result<Arc<AppState>, EnsureError> {
        self.state
            .upgrade()
            .ok_or_else(|| EnsureError::Load(RuntimeError::Msg("server shutting down".into())))
    }

    /// Serving = resident with its mux installed (not mid-eviction).
    fn is_serving(&self, slug: &str) -> bool {
        self.state
            .upgrade()
            .is_some_and(|s| s.has_gpu_engine(slug) && s.mux(slug).is_some())
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
        let pinned = crate::config::RuntimeConfig::get().nv.pin_resident;
        for m in self.models.read().clone() {
            let need = self.required(&m.slug).expect("managed") + RESERVE;
            let free = self.fit_capacity(0)?;
            if free < need {
                if pinned {
                    return Err(RuntimeError::Msg(pinned_misfit(&m.slug, need, free)));
                }
                tracing::info!(
                    slug = %m.slug,
                    need_gib = gib(need),
                    free_gib = gib(free),
                    "planner: not resident at startup (S1 switch will load on demand)"
                );
                continue;
            }
            self.load_model(&m)
                .await
                .map_err(|e| RuntimeError::Msg(format!("{}: {e}", m.slug)))?;
            loaded += 1;
        }
        if loaded == 0 && !self.models.read().is_empty() {
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
        // Fast path: resident AND not being taken down. The residency test
        // belongs here too — mid-unload a model is briefly still resident, and
        // admitting then would hand the request a mux that is about to vanish.
        let state = self.state()?;
        if self.is_serving(slug) && state.residency(slug).admits() {
            self.touch(slug);
            return Ok(());
        }

        // The switch runs on a task of its own: a request future dropped mid-switch (client gone)
        // released the lock between installing the engine and its dispatcher, and the model then
        // read as resident with nothing serving it.
        let mgr = Arc::clone(self);
        let slug = slug.to_string();
        tokio::spawn(async move {
            let _g = match switch_timeout() {
                Some(d) => tokio::time::timeout(d, mgr.switch.lock())
                    .await
                    .map_err(|_| EnsureError::SwitchTimeout(d))?,
                None => mgr.switch.lock().await,
            };
            mgr.ensure_resident_locked(&slug).await
        })
        .await
        .map_err(|e| EnsureError::Load(RuntimeError::Msg(format!("model switch task: {e}"))))?
    }

    /// An engine with no dispatcher (a load cut off between the two) serves nothing and is not
    /// resident for any caller: take it down so the load runs again.
    async fn clear_partial(&self, slug: &str) -> std::result::Result<(), EnsureError> {
        if self.is_resident(slug) && !self.is_serving(slug) {
            tracing::warn!(%slug, "planner: engine installed without a dispatcher — reloading");
            self.evict(slug, Stop::Now).await?;
        }
        Ok(())
    }

    async fn ensure_resident_locked(
        self: &Arc<Self>,
        slug: &str,
    ) -> std::result::Result<(), EnsureError> {
        let state = self.state()?;
        // RE-READ residency under the lock. Checking it before the wait and
        // never again let a request queued behind an in-flight unload sail past
        // the check, then demand-load the model the operator had just taken
        // down — undoing the unload with no trace. An explicitly unloaded model
        // must STAY unloaded until an explicit load.
        if !state.residency(slug).admits() {
            return Err(EnsureError::Unloaded);
        }
        self.clear_partial(slug).await?;
        if self.is_resident(slug) {
            self.touch(slug);
            return Ok(());
        }
        let m = self
            .managed(slug)
            .ok_or_else(|| EnsureError::Load(RuntimeError::UnknownModel(slug.into())))?;
        self.load_backoff(&m)?;
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
            loop {
                let trimmed = self.device.pool_trim(reusable_cap);
                if trimmed > 0 {
                    tracing::info!(
                        trimmed_mib = trimmed >> 20,
                        "planner: released pooled chunks the target cannot reuse"
                    );
                }
                let avail = self
                    .fit_capacity(self.device.pool_bytes())
                    .map_err(EnsureError::Load)?;
                if avail >= need {
                    break;
                }
                let Some(victim) = self.lru_victim(slug) else {
                    tracing::warn!(
                        %slug,
                        need_gib = gib(need),
                        free_gib = gib(avail),
                        "planner: switch cannot fit — shedding"
                    );
                    return Err(EnsureError::WontFit { need, free: avail });
                };
                let (drain_ms, unload_ms) = self.evict(&victim, Stop::Graceful).await?;
                report.evicted.push((victim, drain_ms, unload_ms));
            }

            let t0 = Instant::now();
            match self.load_model(&m).await {
                Ok(()) => {
                    report.load_ms = ms(t0);
                    break;
                }
                // The plan fit but the device did not (an under-counted footprint): evict one more
                // resident and retry, and shed only when nothing is left to evict.
                Err(EnsureError::Load(e)) if is_oom(&e) => {
                    let Some(victim) = self.lru_victim(slug) else {
                        tracing::warn!(%slug, error = %e, "planner: switch load out of memory — shedding");
                        let free = self.free_vram().map_err(EnsureError::Load)?;
                        return Err(EnsureError::WontFit { need, free });
                    };
                    tracing::warn!(%slug, %victim, error = %e, "planner: switch load out of memory — evicting");
                    let (drain_ms, unload_ms) = self.evict(&victim, Stop::Graceful).await?;
                    report.evicted.push((victim, drain_ms, unload_ms));
                }
                Err(e) => return Err(e),
            }
        }
        let (free, total) = self.device.mem_info().map_err(EnsureError::Load)?;
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

    /// Control-plane unload: stop and flush this model's in-flight work, then
    /// release its device memory. Every other model keeps serving.
    ///
    /// Three deliberate differences from a switch eviction:
    ///
    /// * the slug is marked `Unloading` **before** the mux is removed, so a
    ///   request cannot land in the window between removal and the dispatcher's
    ///   exit — where it would be dropped with no terminal chunk;
    /// * in-flight generations are stopped, not drained. A graceful drain is
    ///   `O(max_tokens × service_ms)` and an admin call must not hang on it;
    /// * the slab chunk pool is trimmed to zero. A switch keeps an evicted
    ///   model's physical chunks pooled on purpose (that is what makes the next
    ///   switch µs-class), but an operator who unloads a model and sees
    ///   `nvidia-smi` unchanged will reasonably call it broken.
    ///
    /// Ends in `Residency::Unloaded`, which is what stops the next request (or
    /// the speculative preloader) from silently loading it again.
    pub async fn unload(&self, slug: &str) -> std::result::Result<UnloadReport, EnsureError> {
        self.unload_and_deregister(slug, false).await
    }

    /// Unload a DP rank: in-flight work drains for up to `bound` (new work already routes to the
    /// other ranks), then the rest is stopped.
    pub async fn unload_rank(
        &self,
        key: &str,
        bound: Duration,
    ) -> std::result::Result<UnloadReport, EnsureError> {
        let _g = self.switch.lock().await;
        self.unload_locked(key, Stop::Bounded(bound)).await
    }

    pub async fn unload_and_deregister(
        &self,
        slug: &str,
        deregister: bool,
    ) -> std::result::Result<UnloadReport, EnsureError> {
        self.unload_with_drain(slug, deregister, None).await
    }

    /// Unload this model with an optional graceful drain bound. `None` stops in-flight work
    /// immediately; `Some(bound)` lets in-flight generations drain up to `bound`.
    pub async fn unload_with_drain(
        &self,
        slug: &str,
        deregister: bool,
        drain: Option<Duration>,
    ) -> std::result::Result<UnloadReport, EnsureError> {
        if deregister && self.state()?.dp_rank(slug).is_some() {
            return Err(EnsureError::Load(RuntimeError::Rejected(format!(
                "{slug} is a DP rank; deregister its model instead"
            ))));
        }
        let stop = match drain {
            Some(bound) => Stop::Bounded(bound),
            None => Stop::Now,
        };
        let _g = self.switch.lock().await;
        let report = self.unload_locked(slug, stop).await?;
        if deregister {
            let state = self.state()?;
            let _ = state.registry.unload(slug);
            self.deregister(slug);
            state.clear_slug_group(slug);
            state.set_residency(slug, crate::serve::Residency::Auto);
        }
        Ok(report)
    }

    async fn unload_locked(
        &self,
        slug: &str,
        stop: Stop,
    ) -> std::result::Result<UnloadReport, EnsureError> {
        if !self.manages(slug) {
            return Err(EnsureError::Load(RuntimeError::UnknownModel(slug.into())));
        }
        let state = self.state()?;

        // Before `remove_mux`, not after: this is what closes the late-submit
        // window rather than merely narrowing it.
        state.set_residency(slug, crate::serve::Residency::Unloading);

        let free_before = self.device.mem_info().map_or(0, |(free, _)| free);
        // Unloaded on failure too: the dispatcher is already gone, and only an explicit load
        // (which this state asks for) brings the model back — `Unloading` stuck it refusing.
        let evicted = self.evict(slug, stop).await;
        state.set_residency(slug, crate::serve::Residency::Unloaded);
        let (stop_ms, unload_ms) = evicted?;
        let pool_trimmed = self.device.pool_trim(0);
        let free_after = self.device.mem_info().map_or(0, |(free, _)| free);

        self.last_use.lock().remove(slug);

        let report = UnloadReport {
            slug: slug.to_string(),
            stop_ms,
            unload_ms,
            pool_trimmed,
            freed: free_after.saturating_sub(free_before),
        };
        tracing::info!(
            %slug,
            stop_ms = report.stop_ms,
            unload_ms = report.unload_ms,
            pool_trimmed_mib = report.pool_trimmed >> 20,
            freed_gib = gib(report.freed),
            "model unloaded"
        );
        Ok(report)
    }

    /// Control-plane load: make `slug` resident now and clear any operator
    /// unload. Returns the load wall time in ms (`0.0` when already resident).
    ///
    /// `evict_lru` is off by default on purpose — an admin load must not
    /// silently take down someone else's model. Without it a model that does
    /// not fit is refused with [`EnsureError::WontFit`] carrying the shortfall.
    pub async fn load(
        self: &Arc<Self>,
        slug: &str,
        evict_lru: bool,
    ) -> std::result::Result<f64, EnsureError> {
        if !self.manages(slug) {
            return Err(EnsureError::Load(RuntimeError::UnknownModel(slug.into())));
        }
        let _g = self.switch.lock().await;
        let state = self.state()?;
        self.failed.lock().remove(slug);
        let was = state.residency(slug);
        // `ensure_resident` and the fit path below both refuse a pinned slug,
        // so the pin has to come off first — and go back on if the load fails,
        // or a failed load would quietly re-arm demand loading.
        state.set_residency(slug, crate::serve::Residency::Auto);

        let result = self.load_inner(slug, evict_lru).await;
        if result.is_err() {
            state.set_residency(slug, was);
        }
        result
    }

    async fn load_inner(
        self: &Arc<Self>,
        slug: &str,
        evict_lru: bool,
    ) -> std::result::Result<f64, EnsureError> {
        self.clear_partial(slug).await?;
        if self.is_resident(slug) {
            self.touch(slug);
            return Ok(0.0);
        }
        if evict_lru {
            let t0 = Instant::now();
            self.ensure_resident_locked(slug).await?;
            return Ok(ms(t0));
        }

        let m = self
            .managed(slug)
            .ok_or_else(|| EnsureError::Load(RuntimeError::UnknownModel(slug.into())))?;
        let need = self.required(slug).expect("managed") + RESERVE;
        // Same accounting as the switch path: chunks the incoming slab cannot
        // consume have to be real free memory before the load.
        self.device.pool_trim(m.plan.slab_reusable());
        let avail = self
            .fit_capacity(self.device.pool_bytes())
            .map_err(EnsureError::Load)?;
        if avail < need {
            return Err(EnsureError::WontFit { need, free: avail });
        }
        let t0 = Instant::now();
        self.load_model(&m).await?;
        self.touch(slug);
        let load_ms = ms(t0);
        tracing::info!(%slug, load_ms, "model loaded");
        Ok(load_ms)
    }

    /// After a switch: if spare VRAM fits the hottest non-resident model
    /// WITHOUT evicting anyone, load it in the background. A later request
    /// for it then finds it resident — or coalesces on the switch lock while
    /// the preload finishes — instead of paying the full 2 s-class switch.
    /// A no-op when nothing fits; `PLOW_PRELOAD=0` disables.
    fn maybe_preload(self: &Arc<Self>) {
        if !crate::config::RuntimeConfig::get().preload {
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
            let m = mgr.managed(&slug).expect("candidate is managed");
            mgr.device.pool_trim(m.plan.slab_reusable());
            let t0 = Instant::now();
            match mgr.load_model(&m).await {
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
        let pool = self.device.pool_bytes();
        let (free, headroom) = (self.free_vram().ok()?, self.planned_headroom().ok()?);
        let state = self.state.upgrade()?;
        // Cloned out first: `required` takes the models lock again, and a recursive read behind
        // a queued registration's write deadlocks.
        let models = self.models.read().clone();
        let fits: Vec<String> = models
            .iter()
            .filter(|m| !self.is_resident(&m.slug))
            // Never speculatively reload what an operator unloaded.
            .filter(|m| state.residency(&m.slug).admits())
            .filter(|m| self.load_backoff(m).is_ok())
            .filter(|m| {
                let credit = pool.min(m.plan.slab_reusable());
                self.required(&m.slug)
                    .is_some_and(|need| need + RESERVE <= (free + credit).min(headroom))
            })
            .map(|m| m.slug.clone())
            .collect();
        let last_use = self.last_use.lock();
        fits.into_iter().max_by_key(|s| last_use.get(s).copied())
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
    async fn evict(&self, slug: &str, stop: Stop) -> std::result::Result<(f64, f64), EnsureError> {
        let state = self.state()?;
        let t0 = Instant::now();
        if let Some(mux) = state.remove_mux(slug) {
            match stop {
                // Control-plane unload: an admin call must not block on
                // `max_tokens`. Live generations stop at the next tick top and
                // close with what they produced.
                Stop::Now => mux.preempt().await,
                Stop::Bounded(bound) => {
                    if tokio::time::timeout(bound, mux.drain()).await.is_err() {
                        tracing::info!(%slug, bound_ms = bound.as_millis() as u64, "rank drain bound passed — preempting");
                        mux.preempt().await;
                    }
                }
                Stop::Graceful => match drain_timeout_ms() {
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
                },
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
        let cfg = state.engine_config(slug);
        on_own_thread("plow-unload", move || {
            crate::config::scoped(cfg, || drop(engine))
        })
        .await
        .map_err(EnsureError::Load)?;
        let engine_ms = ms(t1);
        if let Ok(bundle) = state.registry.get(state.model_of(slug)) {
            let device = self.ordinal();
            crate::asr::serving::release(&bundle.dir, device);
            crate::tts::serving::release(&bundle.dir, device);
            // Their worker threads drop the packet runtimes once the last request lets go.
            let deadline = Instant::now() + SIDECAR_RELEASE_WAIT;
            while self.device.is_sibling_resident(&bundle.dir) {
                if Instant::now() > deadline {
                    tracing::error!(%slug, "sibling packets still resident after evict — VRAM leak");
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
        let unload_ms = ms(t1);
        let (free, _) = self.device.mem_info().map_err(EnsureError::Load)?;
        let modules = self.device.live_modules();
        tracing::info!(%slug, drain_ms, unload_ms, engine_ms, free_gib = gib(free), ?modules, "model evicted");
        Ok((drain_ms, unload_ms))
    }

    /// Load `m`'s engine (blocking H2D on the blocking pool), install it, and
    /// spawn its dispatcher. Calibrates the overhead cache on first load.
    async fn load_model(&self, m: &Managed) -> std::result::Result<(), EnsureError> {
        let r = self.load_model_once(m).await;
        match &r {
            Err(EnsureError::Load(e)) if !is_oom(e) => {
                let stamp = bundle_stamp(m);
                let mut failed = self.failed.lock();
                // Doubles per failure of the same bundle files; a changed bundle starts over.
                let delay = match failed.get(&m.slug) {
                    Some(f) if f.stamp == stamp => (f.delay * 2).min(LOAD_BACKOFF_MAX),
                    _ => LOAD_BACKOFF_MIN,
                };
                tracing::warn!(slug = %m.slug, error = %e, retry_s = delay.as_secs(), "planner: load failed — demand loads back off");
                failed.insert(
                    m.slug.clone(),
                    LoadFailure {
                        why: e.to_string(),
                        at: Instant::now(),
                        delay,
                        stamp,
                    },
                );
            }
            Ok(()) => {
                self.failed.lock().remove(&m.slug);
            }
            Err(_) => {}
        }
        r
    }

    /// `Err(Backoff)` while `m`'s last failed load is inside its backoff and its bundle files are
    /// unchanged; a demand load may try it otherwise.
    fn load_backoff(&self, m: &Managed) -> std::result::Result<(), EnsureError> {
        let Some(f) = self.failed.lock().get(&m.slug).cloned() else {
            return Ok(());
        };
        let left = f.delay.saturating_sub(f.at.elapsed());
        if left.is_zero() || bundle_stamp(m) != f.stamp {
            return Ok(());
        }
        Err(EnsureError::Backoff {
            why: f.why,
            retry_after: left,
        })
    }

    async fn load_model_once(&self, m: &Managed) -> std::result::Result<(), EnsureError> {
        let state = self.state()?;
        let bundle = state
            .registry
            .get(state.model_of(&m.slug))
            .map_err(EnsureError::Load)?;
        if bundle.tokenizer().is_byte_fallback() {
            return Err(EnsureError::Load(RuntimeError::Msg(format!(
                "{}: GPU engine requires a real tokenizer.json in {}",
                m.slug,
                m.dir.display()
            ))));
        }

        // `PLOW_PF_SEG_NONCOOP=1` drops the driver's co-residency admission on
        // an explicitly stated premise: "the grid equals the module's queried
        // resident capacity AND THE STREAM IS OTHERWISE IDLE, so all blocks
        // schedule together" (`exec/gpu.rs`). A co-tenant on the same device
        // makes the second half false, and the documented consequence is a
        // spin "on counters owned by blocks that were never resident" — a hang,
        // not an error, with no launch failure to catch it.
        //
        // So this is a refusal, not a warning: a diagnostic knob must not be
        // able to deadlock a device by being left set.
        // This bundle's own knobs: every step of its load below runs under them.
        let cfg = crate::config::bundle_config(&m.dir).map_err(EnsureError::Load)?;
        state.set_engine_config(&m.slug, cfg);
        // Either side running non-cooperative launches is the hazard: per-engine configs let
        // one model set it and its co-tenant not.
        {
            let co_tenants: Vec<String> = self
                .models
                .read()
                .iter()
                .filter(|o| o.slug != m.slug)
                .map(|o| o.slug.clone())
                .filter(|slug| self.is_resident(slug))
                .collect();
            let noncoop = |c: Option<&crate::config::RuntimeConfig>| {
                c.unwrap_or_else(|| crate::config::RuntimeConfig::get())
                    .nv
                    .pf_seg_noncoop
            };
            if !co_tenants.is_empty()
                && (cfg.nv.pf_seg_noncoop
                    || co_tenants.iter().any(|s| noncoop(state.engine_config(s))))
            {
                return Err(EnsureError::Load(RuntimeError::Rejected(format!(
                    "refusing to make {} co-resident with {:?} while PLOW_PF_SEG_NONCOOP is set: \
                     non-cooperative launch assumes the device is otherwise idle, and a co-tenant \
                     turns that assumption into a counter spin-hang. Unset it to share a device.",
                    m.slug, co_tenants
                ))));
            }
        }

        // Turns are installed from the startup tenant count: a model loaded later beside another
        // raises its group's turn from `Free` (CUDA); an HSA group not started round-robin is
        // refused (`AppState::upgrade_turn`).
        let others = self
            .models
            .read()
            .iter()
            .filter(|o| o.slug != m.slug && self.is_resident(&o.slug))
            .count();
        state.upgrade_turn(&m.slug, others, self.device.co_tenants_need_rr());
        if unturned_co_tenancy(
            self.device.co_tenants_need_rr(),
            state.device_turn(&m.slug).map(|t| t.mode()),
            others > 0,
        ) {
            return Err(EnsureError::Load(RuntimeError::Rejected(format!(
                "refusing to make {} co-resident on a device group whose co-tenant turns are not \
                 round-robin: HSA queues give no whole-grid residency, so two engines can wait on \
                 each other's blocks forever. Start the server with --co-sched rr to share a group.",
                m.slug
            ))));
        }

        let (free_before, _) = self.device.mem_info().map_err(EnsureError::Load)?;
        let pool_before = self.device.pool_bytes();
        let device = Arc::clone(&self.device);
        let (dir, ckpt) = (m.dir.clone(), m.ckpt.clone());
        let engine = on_own_thread("plow-load", move || {
            crate::config::scoped(Some(cfg), || device.load_engine(&dir, &ckpt))
        })
        .await
        .and_then(|engine| engine)
        .map_err(EnsureError::Load)?;
        #[cfg(feature = "cuda")]
        if let crate::serve::engine::ServeEngine::Cuda(ref gpu) = engine {
            crate::config::scoped(Some(cfg), || mux::check_speech_packet(gpu))
                .map_err(EnsureError::Load)?;
        }
        let (free_after, _) = self.device.mem_info().map_err(EnsureError::Load)?;
        let pool_after = self.device.pool_bytes();

        // First-load calibration: measured footprint minus the planned tensor
        // bytes = module/table/allocator overhead for this assets dir. Pool
        // chunks the slab consumed moved from the pool ledger into the engine
        // without touching `free` — count both ledgers or reused chunks make
        // the model look smaller than it is.
        let used = (free_before + pool_before).saturating_sub(free_after + pool_after);
        // Sibling packets bind later, outside this measurement.
        let measured = used.saturating_sub(m.plan.tensor_total() - m.plan.sidecar_bytes);
        self.overhead
            .lock()
            .entry(m.slug.clone())
            .or_insert(measured);
        tracing::info!(
            slug = %m.slug,
            used_gib = gib(used),
            planned_gib = gib(m.plan.tensor_total() - m.plan.sidecar_bytes),
            sidecar_gib = gib(m.plan.sidecar_bytes),
            overhead_mib = measured >> 20,
            "planner: load measured"
        );

        state.install_gpu_engine(m.slug.clone(), engine);
        // Sibling packets (ASR encoder, codec, vocoder) bind with the engine and before its mux,
        // so residency and the planner's charge cover them, a request never binds them itself,
        // and an eviction frees them.
        let bind_state = Arc::clone(&state);
        let slug = m.slug.clone();
        let bound = on_own_thread("plow-bind", move || -> Result<()> {
            crate::config::scoped(Some(cfg), || {
                crate::asr::serving::bind(&bind_state, &slug)?;
                crate::tts::serving::bind(&bind_state, &slug).map_err(RuntimeError::Msg)
            })
        })
        .await
        .and_then(|bound| bound);
        if let Err(e) = bound {
            let _ = self.evict(&m.slug, Stop::Now).await;
            return Err(EnsureError::Load(e));
        }
        let mux = mux::spawn(m.slug.clone(), bundle, Arc::clone(&state), self.mux_cfg);
        state.install_mux(m.slug.clone(), mux);
        Ok(())
    }
}

/// A load that would put a second engine on a device needing round-robin turns while its
/// turn is not round-robin.
fn unturned_co_tenancy(
    needs_rr: bool,
    turn: Option<crate::serve::cosched::CoSched>,
    co_resident: bool,
) -> bool {
    needs_rr && co_resident && turn != Some(crate::serve::cosched::CoSched::Rr)
}

/// Refuse a bundle compiled for tensor parallelism when the device does not support TP.
fn check_device_tp(device: &dyn ManagerDevice, slug: &str, dir: &Path) -> Result<()> {
    check_tp(device, slug, tp_degree(dir)?)
}

fn check_tp(device: &dyn ManagerDevice, slug: &str, tp: u32) -> Result<()> {
    if tp > 1 && !device.supports_tp() {
        return Err(RuntimeError::Rejected(format!(
            "{slug} declares tensor-parallel degree {tp}, and this model manager device is \
             single-device (one backend per engine). Serving it here would load one shard \
             and answer with it. Compile a TP1 bundle, or serve it on a backend supporting TP (e.g. AMD HSA)."
        )));
    }
    // The engine spans `tp` devices from the group's first: past the group they belong to other
    // managers, whose planners never see this load.
    if tp > device.group_width() {
        return Err(RuntimeError::Rejected(format!(
            "{slug} declares tensor-parallel degree {tp}, and its device group starting at {} \
             has {} device(s); start the server with a placement whose groups are {tp} wide",
            device.device_ordinal(),
            device.group_width()
        )));
    }
    Ok(())
}

/// `--drain-timeout-ms` / `PLOW_DRAIN_TIMEOUT_MS`: how long an eviction lets
/// in-flight generations finish before preempting them. Unset = graceful
/// unbounded drain (the pre-existing behavior).
fn drain_timeout_ms() -> Option<u64> {
    crate::config::RuntimeConfig::get().drain_timeout_ms()
}

/// `--switch-timeout-ms` / `PLOW_SWITCH_TIMEOUT_MS` (600 s when unset, 0 = unbounded).
fn switch_timeout() -> Option<Duration> {
    let ms = crate::config::RuntimeConfig::get()
        .switch_timeout_ms
        .unwrap_or(600_000);
    (ms > 0).then(|| Duration::from_millis(ms))
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
    fn the_switch_lock_wait_is_bounded_by_default() {
        if crate::config::RuntimeConfig::get()
            .switch_timeout_ms
            .is_none()
        {
            assert_eq!(switch_timeout(), Some(Duration::from_secs(600)));
        }
        let e = EnsureError::SwitchTimeout(Duration::from_secs(600));
        assert!(e.to_string().contains("600 s"));
    }

    #[test]
    fn plan_classifies_tensor_names() {
        let mut p = BlobPlan {
            weights_bytes: 0,
            kv_bytes: 0,
            other_bytes: 0,
            sidecar_bytes: 0,
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
        p.sidecar_bytes = 12;
        assert_eq!(
            p.tensor_total(),
            250,
            "sibling packets are part of the footprint"
        );
        assert_eq!(p.slab_reusable(), 178, "but never ride the weight slab");
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
            sidecar_bytes: 0,
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

    struct Group(u32);

    impl ManagerDevice for Group {
        fn device_ordinal(&self) -> u8 {
            2
        }
        fn mem_info(&self) -> Result<(u64, u64)> {
            Ok((0, 0))
        }
        fn granularity(&self) -> Result<u64> {
            Ok(4096)
        }
        fn compute_capability(&self) -> Option<(u32, u32)> {
            None
        }
        fn pool_bytes(&self) -> u64 {
            0
        }
        fn pool_trim(&self, _: u64) -> u64 {
            0
        }
        fn load_engine(&self, _: &Path, _: &Path) -> Result<crate::serve::engine::ServeEngine> {
            unreachable!()
        }
        fn is_sibling_resident(&self, _: &Path) -> bool {
            false
        }
        fn supports_tp(&self) -> bool {
            true
        }
        fn group_width(&self) -> u32 {
            self.0
        }
    }

    /// A manager over one model, `m` in `dir`, whose every load fails (no registry entry).
    fn failing_manager(dir: &Path) -> (Arc<AppState>, Arc<ModelManager>) {
        let backend: Arc<dyn crate::device::Backend> =
            Arc::new(crate::device::cpu::CpuBackend::new(1));
        let execset = Arc::new(crate::exec::ExecutorSet::bringup(backend).unwrap());
        let state = Arc::new(AppState::new(crate::orch::Registry::new(), execset));
        let plan = BlobPlan {
            weights_bytes: 1 << 20,
            kv_bytes: 0,
            other_bytes: 0,
            sidecar_bytes: 0,
        };
        let mgr = Arc::new(ModelManager {
            state: Arc::downgrade(&state),
            device: Arc::new(HostManagerDevice::new()),
            mux_cfg: MuxConfig::default(),
            models: RwLock::new(vec![Managed {
                slug: "m".into(),
                dir: dir.into(),
                ckpt: dir.into(),
                plan,
            }]),
            switch: tokio::sync::Mutex::new(()),
            last_use: Mutex::default(),
            overhead: Mutex::default(),
            budget: None,
            last_switch: Mutex::new(None),
            growing: std::sync::atomic::AtomicBool::new(false),
            failed: Mutex::default(),
            runtime: None,
        });
        (state, mgr)
    }

    /// A model whose load failed (other than out of memory) is shed by later demand loads with
    /// the recorded error and a Retry-After, before the planner evicts anyone; an explicit load
    /// retries it, and deregistration forgets it.
    #[tokio::test]
    async fn a_failed_load_backs_off_demand_loads() {
        let (_state, mgr) = failing_manager(Path::new("/nonexistent"));
        let first = mgr.ensure_resident("m").await.unwrap_err();
        assert!(matches!(first, EnsureError::Load(_)), "{first}");
        let again = mgr.ensure_resident("m").await.unwrap_err();
        let EnsureError::Backoff { why, retry_after } = &again else {
            panic!("{again}");
        };
        assert!(first.to_string().contains(why.as_str()), "{again}");
        assert!(*retry_after <= LOAD_BACKOFF_MIN && again.retry_after() == Some(*retry_after));
        assert!(mgr.preload_candidate().is_none());
        let explicit = mgr.load("m", false).await.unwrap_err();
        assert!(
            matches!(explicit, EnsureError::Load(_)),
            "an explicit load retries: {explicit}"
        );
        assert!(mgr.deregister("m"));
        assert!(mgr.failed.lock().is_empty());
    }

    /// The backoff expires: the next demand load retries, and a repeat failure of the same
    /// bundle doubles the wait up to the cap. Changed bundle files retry at once and start over.
    #[tokio::test]
    async fn a_load_backoff_expires_doubles_and_resets_on_a_changed_bundle() {
        let dir = std::env::temp_dir().join(format!("plowrt_backoff_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.pkt"), b"partial").unwrap();
        let (_state, mgr) = failing_manager(&dir);
        let delay = |mgr: &ModelManager| mgr.failed.lock()["m"].delay;
        let expire = |mgr: &ModelManager| {
            let mut f = mgr.failed.lock();
            let f = f.get_mut("m").unwrap();
            f.at = Instant::now().checked_sub(f.delay).unwrap();
        };
        assert!(matches!(
            mgr.ensure_resident("m").await,
            Err(EnsureError::Load(_))
        ));
        assert_eq!(delay(&mgr), LOAD_BACKOFF_MIN);
        let mut want = LOAD_BACKOFF_MIN;
        for _ in 0..6 {
            assert!(matches!(
                mgr.ensure_resident("m").await,
                Err(EnsureError::Backoff { .. })
            ));
            expire(&mgr);
            assert!(
                mgr.preload_candidate().is_some(),
                "expired: the preloader may try it"
            );
            assert!(matches!(
                mgr.ensure_resident("m").await,
                Err(EnsureError::Load(_))
            ));
            want = (want * 2).min(LOAD_BACKOFF_MAX);
            assert_eq!(delay(&mgr), want);
        }
        assert_eq!(want, LOAD_BACKOFF_MAX);
        // The copy finishes: the file grew, so the next request retries inside the backoff.
        std::fs::write(dir.join("model.pkt"), b"partial, now complete").unwrap();
        assert!(matches!(
            mgr.ensure_resident("m").await,
            Err(EnsureError::Load(_))
        ));
        assert_eq!(
            delay(&mgr),
            LOAD_BACKOFF_MIN,
            "a changed bundle starts over"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A TP engine spans `tp` devices from its group's first; past the group they belong to
    /// other managers, which never planned for it.
    #[test]
    fn a_tp_bundle_must_fit_its_device_group() {
        assert!(check_tp(&Group(4), "m", 4).is_ok());
        assert!(check_tp(&Group(4), "m", 2).is_ok());
        let e = check_tp(&Group(1), "m", 4).unwrap_err().to_string();
        assert!(e.contains("starting at 2 has 1 device"), "{e}");
        assert!(check_tp(&HostManagerDevice::new(), "m", 2).is_err());
        assert!(check_tp(&HostManagerDevice::new(), "m", 1).is_ok());
    }

    /// A device group installed with one tenant (or none: a zero-asset start) runs free turns,
    /// and the mode is fixed at install. On HSA a second engine loaded there later is refused.
    #[test]
    fn hsa_co_tenancy_needs_round_robin_turns() {
        use crate::serve::cosched::CoSched;
        assert!(unturned_co_tenancy(true, Some(CoSched::Free), true));
        assert!(unturned_co_tenancy(true, None, true));
        assert!(!unturned_co_tenancy(true, Some(CoSched::Rr), true));
        assert!(!unturned_co_tenancy(true, Some(CoSched::Free), false));
        assert!(!unturned_co_tenancy(false, Some(CoSched::Free), true));
        assert!(HostManagerDevice::new().group_width() == 1);
        assert!(!HostManagerDevice::new().co_tenants_need_rr());
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

fn same_path(a: &Path, b: &Path) -> bool {
    a == b
        || a.canonicalize()
            .ok()
            .zip(b.canonicalize().ok())
            .is_some_and(|(a, b)| a == b)
}
