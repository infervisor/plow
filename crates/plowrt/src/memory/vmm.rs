//! §L2 VMM-backed prefix sharing for the full-attention KV cache
//! ("Implementation V1").
//!
//! ## What this buys
//!
//! Full-layer KV for a shared prompt prefix is held **once** in HBM and
//! multi-mapped (`cuMemMap`, same physical handle) into every sharing
//! sequence's virtual window — 10 GiB deduped **per sharer** at 31B@128k, the
//! difference between batch 1 and batch 8 on a 95 GiB card. The measured
//! economics (feasibility review in the plan): `cuMemSetAccess` is ~69 µs per
//! granule mapping, so sharing blocks are **64 MiB-class** (attach 11.4 ms for
//! a 10 GiB prefix vs a 14.7 ms D2D copy); the win is the dedup, not latency.
//!
//! ## Address layout (no kernel change)
//!
//! Per (full layer, K|V) one contiguous VA reservation spans the whole
//! batch-major tensor `[batch][kvh][max_ctx][hd]` — exactly the shape the
//! emitter already declares (`gemma4.rs::kv_ring` gives full layers
//! `kvr = ctx = max_ctx`), so the tensor table keeps ONE base and both the
//! flash-decode addressing and `bind_kv_slot`'s `base + b·stride` are
//! untouched. Sequence `b`, head `h` owns the sub-window at
//! `(b·kvh + h)·max_ctx·row_bytes`; physical blocks are mapped there at the
//! decode frontier (a background pre-mapper keeps the next block mapped ahead
//! — probe [5]: mapping during kernel execution is safe, no implicit sync).
//!
//! ## Prefix policy
//!
//! The radix tree in [`super::prefix`] is kept verbatim (refcount, COW, LRU,
//! tombstone); this module keys two side tables off `Match::placed`:
//! node → physical block ids, and published-boundary → snapshot. Attach
//! shares whole blocks and restores sliding windows plus any partial full-KV
//! block from a snapshot. Partial blocks remain private, so subsequent
//! prefill never writes into a shared block. Boundaries shorter than one
//! physical block use snapshots without a radix node.
//!
//! ## Eviction (leak-audit finding #9)
//!
//! `cuMemCreate` OOM and the `cache_cap_bytes` soft cap both drive
//! `PrefixCache::evict_lru` until satisfied; physical blocks are released at
//! refcount 0 and boundary snapshots freed with their node. Short-prefix
//! snapshots use a second-chance policy and stay pinned during restoration.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::memory::pool::GrowablePool;
use crate::memory::prefix::{BlockHash, PrefixCache};
use crate::{Result, RuntimeError};

/// The VMM driver surface the pool drives. Implemented by
/// `crate::device::cuda::CudaBackend` (dlopen'd driver entry points) and by a
/// mock in the unit tests. Teardown-side calls are infallible by design
/// (Drop has no error channel; implementations log).
pub trait VmmOps: Send + Sync {
    /// Physical allocation granularity (recommended; 2 MiB measured).
    fn granularity(&self) -> Result<u64>;
    /// Reserve a VA range (no physical backing).
    fn reserve(&self, bytes: u64) -> Result<u64>;
    fn address_free(&self, va: u64, bytes: u64);
    /// Create one physical block; returns the generic allocation handle.
    fn create(&self, bytes: u64) -> Result<u64>;
    fn release(&self, handle: u64);
    /// Map `handle` at `va` (multi-map of one handle is legal — probe [2]).
    fn map(&self, va: u64, bytes: u64, handle: u64) -> Result<()>;
    fn unmap(&self, va: u64, bytes: u64);
    /// Grant RW device access to a fully-mapped range.
    fn set_access(&self, va: u64, bytes: u64) -> Result<()>;
    /// Plain device allocation (sliding-window snapshots).
    fn alloc(&self, bytes: u64) -> Result<u64>;
    fn free(&self, va: u64);
    fn copy_dtod(&self, dst: u64, src: u64, bytes: u64) -> Result<()>;
    fn copy_dtod_batch(&self, pairs: &[(u64, u64, u64)]) -> Result<()> {
        for &(dst, src, bytes) in pairs {
            self.copy_dtod(dst, src, bytes)?;
        }
        Ok(())
    }

    /// Take every pooled physical chunk `(handle, bytes)` a previous
    /// [`VmmSlab`] kept via [`Self::pool_put`]. Re-mapping a pooled chunk is
    /// ~free (map+set_access, µs-class) where creating one pays the driver's
    /// serial page-commit rate (~13 GiB/s measured) — the whole point of the
    /// pool. Default: nothing pooled.
    fn pool_take(&self) -> Vec<(u64, u64)> {
        Vec::new()
    }

    /// Keep physical chunks for a future slab, or dispose of them. The
    /// default releases immediately (no pool) — safe everywhere; a backend
    /// that pools MUST release leftovers when it is dropped.
    fn pool_put(&self, chunks: Vec<(u64, u64)>) {
        for (h, _) in chunks {
            self.release(h);
        }
    }

    /// Bytes currently held in the backend's physical-chunk pool. The serve
    /// planner credits this against a switch target's requirement — pooled
    /// chunks are VRAM the incoming slab re-maps instead of re-creating.
    fn pool_bytes(&self) -> u64 {
        0
    }

    /// Release pooled chunks until at most `keep_bytes` remain; returns the
    /// bytes actually released. The planner trims to what the incoming
    /// model's slab can consume, so credited-but-unconsumable pool VRAM goes
    /// back to the driver before the load needs it as free memory.
    fn pool_trim(&self, keep_bytes: u64) -> u64 {
        let _ = keep_bytes;
        0
    }
}

/// Uniform full-KV allocation geometry. Prefix mode resolves checkpoint
/// metadata; live allocation validates packet operands with [`LiveKvLayout`].
#[derive(Clone, Debug)]
pub struct VmmGeometry {
    /// Layer indices with full attention (VMM-backed).
    pub full_layers: Vec<u32>,
    pub kvh_full: u32,
    pub hd_full: u32,
    /// Layer indices with sliding attention (stay on cudaMalloc rings).
    pub slide_layers: Vec<u32>,
    pub kvh_slide: u32,
    pub hd_slide: u32,
    /// Sliding attention window (rows restored on attach).
    pub window: u32,
    /// Bytes per FULL-layer KV element: 2 = bf16/fp16, 1 = fp8 e4m3 (with
    /// per-row f32 scale tensors — see the engine's snapshot layout).
    /// Resolved by the engine from the blob (scale-tensor presence), not
    /// from `config.json`.
    pub elem: u32,
    /// Bytes per SLIDING-layer KV element. Differs from `elem` in the mixed
    /// fp8-KV mode (`PLOW_FP8_KV_FULL=1`: e4m3 full layers, bf16 rings).
    pub elem_slide: u32,
    /// Rows per head window == the compiled context == full-layer kv_stride.
    pub max_ctx: u32,
    /// Engine sequence slots (compiled decode batch).
    pub batch: u32,
}

impl VmmGeometry {
    /// Parse the checkpoint's `config.json` (`text_config` or top level).
    /// Gemma-family: `layer_types` splits full/sliding layers and
    /// `sliding_window` is required. No `layer_types` (Qwen/Llama-family):
    /// every `num_hidden_layers` layer is full attention, no rings, no
    /// boundary snapshots. Heads/dims come from
    /// `num_global_key_value_heads`/`num_key_value_heads` and
    /// `global_head_dim`/`head_dim`. `None` when the shape isn't there — the
    /// caller then leaves VMM off.
    pub fn from_config(checkpoint_dir: &std::path::Path, max_ctx: u32, batch: u32) -> Option<Self> {
        let bytes = std::fs::read(checkpoint_dir.join("config.json")).ok()?;
        let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        let t = v.get("text_config").unwrap_or(&v);
        let mut full_layers = Vec::new();
        let mut slide_layers = Vec::new();
        match t.get("layer_types").and_then(|x| x.as_array()) {
            Some(layer_types) => {
                for (l, ty) in layer_types.iter().enumerate() {
                    match ty.as_str()? {
                        "full_attention" => full_layers.push(l as u32),
                        "sliding_attention" => slide_layers.push(l as u32),
                        _ => return None,
                    }
                }
            }
            None => {
                let n = t.get("num_hidden_layers")?.as_u64()? as u32;
                full_layers = (0..n).collect();
            }
        }
        let u = |k: &str| t.get(k).and_then(|x| x.as_u64()).map(|x| x as u32);
        let kvh_slide = u("num_key_value_heads")?;
        let kvh_full = u("num_global_key_value_heads").unwrap_or(kvh_slide);
        let hd_slide = u("head_dim")?;
        let hd_full = u("global_head_dim").unwrap_or(hd_slide);
        // Required only when sliding layers exist — their snapshot geometry
        // depends on it. All-full models carry window 0 (nothing to restore).
        let window = match slide_layers.is_empty() {
            true => u("sliding_window").unwrap_or(0),
            false => u("sliding_window")?,
        };
        if full_layers.is_empty()
            || kvh_full == 0
            || hd_full == 0
            || batch == 0
            || max_ctx == 0
            || (!slide_layers.is_empty() && (kvh_slide == 0 || hd_slide == 0 || window == 0))
        {
            return None;
        }
        Some(VmmGeometry {
            full_layers,
            kvh_full,
            hd_full,
            slide_layers,
            kvh_slide,
            hd_slide,
            window,
            elem: 2,
            elem_slide: 2,
            max_ctx,
            batch,
        })
    }

    pub(crate) fn block_bytes(&self, gran: u64, block_hint: u64) -> Result<u64> {
        let row_bytes = self.row_bytes();
        let head_span = self.max_ctx as u64 * row_bytes;
        if gran == 0 || row_bytes == 0 || head_span < gran || head_span % gran != 0 {
            return Err(RuntimeError::Device(format!(
                "vmm: head window {head_span} B not a multiple of granularity {gran}"
            )));
        }
        let block_bytes = block_hint.clamp(gran, head_span);
        if !block_bytes.is_power_of_two()
            || head_span % block_bytes != 0
            || block_bytes % gran != 0
            || block_bytes % row_bytes != 0
        {
            return Err(RuntimeError::Device(format!(
                "vmm: block {block_bytes} B must be a pow2 multiple of granularity \
                 {gran} and row {row_bytes}, dividing the head window {head_span}"
            )));
        }
        Ok(block_bytes)
    }

    /// Expected byte size of one full-layer `kv.{l}.k`/`.v` tensor — the
    /// validation gate against the blob's declared sizes.
    pub fn full_tensor_bytes(&self) -> u64 {
        self.batch as u64 * self.kvh_full as u64 * self.max_ctx as u64 * self.row_bytes()
    }

    /// One token row's bytes in a full-layer head window.
    pub fn row_bytes(&self) -> u64 {
        self.hd_full as u64 * self.elem as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveRingTensor {
    pub tensor: usize,
    pub slot_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LiveRingStats {
    pub reserved_bytes: u64,
    pub resident_bytes: u64,
    pub mapped_slots: usize,
    pub mapped_prefix: usize,
}

struct RingWindow {
    tensor: usize,
    va: u64,
    bytes: u64,
    slot_bytes: u64,
    map_bytes: u64,
    handles: Vec<Option<u64>>,
    refs: Vec<usize>,
}

/// Whole-slot ring backing is committed while at least one logical slot uses each mapping unit.
pub struct VmmRings {
    ops: Arc<dyn VmmOps>,
    windows: Vec<RingWindow>,
    mapped: Vec<bool>,
    prefix: usize,
    stats: LiveRingStats,
}

impl VmmRings {
    pub fn new(ops: Arc<dyn VmmOps>, tensors: &[LiveRingTensor], batch: usize) -> Result<Self> {
        let reject = |slot_bytes: u64, granularity: u64| {
            RuntimeError::Rejected(format!(
                "invalid live ring allocation geometry: slot_bytes={slot_bytes}, granularity={granularity}"
            ))
        };
        if batch == 0 {
            return Err(reject(0, 0));
        }
        let granularity = ops.granularity()?;
        if granularity == 0 {
            return Err(reject(0, granularity));
        }
        let mut reserved_bytes = 0u64;
        for (i, t) in tensors.iter().enumerate() {
            if t.slot_bytes == 0 || tensors[..i].iter().any(|p| p.tensor == t.tensor) {
                return Err(reject(t.slot_bytes, granularity));
            }
            let logical_bytes = t
                .slot_bytes
                .checked_mul(batch as u64)
                .ok_or_else(|| reject(t.slot_bytes, granularity))?;
            let bytes = logical_bytes
                .checked_add(granularity - 1)
                .map(|n| n / granularity * granularity)
                .ok_or_else(|| reject(t.slot_bytes, granularity))?;
            reserved_bytes = reserved_bytes
                .checked_add(bytes)
                .ok_or_else(|| reject(t.slot_bytes, granularity))?;
        }
        let mut rings = Self {
            ops,
            windows: Vec::with_capacity(tensors.len()),
            mapped: vec![false; batch],
            prefix: 0,
            stats: LiveRingStats {
                reserved_bytes,
                resident_bytes: 0,
                mapped_slots: 0,
                mapped_prefix: 0,
            },
        };
        for t in tensors {
            let logical_bytes = t.slot_bytes * batch as u64;
            let bytes = (logical_bytes + granularity - 1) / granularity * granularity;
            let map_bytes = if t.slot_bytes % granularity == 0 {
                t.slot_bytes
            } else {
                granularity
            };
            let va = rings.ops.reserve(bytes)?;
            if va == 0 || va % granularity != 0 || va.checked_add(bytes).is_none() {
                rings.ops.address_free(va, bytes);
                return Err(reject(t.slot_bytes, granularity));
            }
            rings.windows.push(RingWindow {
                tensor: t.tensor,
                va,
                bytes,
                slot_bytes: t.slot_bytes,
                map_bytes,
                handles: vec![None; (bytes / map_bytes) as usize],
                refs: vec![0; (bytes / map_bytes) as usize],
            });
        }
        Ok(rings)
    }

    pub fn tensor_va(&self, tensor: usize) -> Option<u64> {
        self.windows
            .iter()
            .find(|w| w.tensor == tensor)
            .map(|w| w.va)
    }

    pub fn stats(&self) -> LiveRingStats {
        self.stats
    }

    pub fn ensure_slot(&mut self, slot: usize) -> Result<()> {
        if slot >= self.mapped.len() {
            return Err(RuntimeError::Rejected(
                "live ring slot out of bounds".into(),
            ));
        }
        if self.mapped[slot] {
            return Ok(());
        }
        let mut touched = Vec::new();
        for i in 0..self.windows.len() {
            let va_base = self.windows[i].va;
            let slot_bytes = self.windows[i].slot_bytes;
            let map_bytes = self.windows[i].map_bytes;
            let first = slot as u64 * slot_bytes / map_bytes;
            let end = (slot as u64 + 1) * slot_bytes;
            let last = (end + map_bytes - 1) / map_bytes;
            for unit in first..last {
                let unit = unit as usize;
                if self.windows[i].refs[unit] == 0 {
                    debug_assert!(self.windows[i].handles[unit].is_none());
                    let va = va_base + unit as u64 * map_bytes;
                    let result = (|| {
                        let handle = self.ops.create(map_bytes)?;
                        if let Err(e) = self.ops.map(va, map_bytes, handle) {
                            self.ops.release(handle);
                            return Err(e);
                        }
                        if let Err(e) = self.ops.set_access(va, map_bytes) {
                            self.ops.unmap(va, map_bytes);
                            self.ops.release(handle);
                            return Err(e);
                        }
                        Ok(handle)
                    })();
                    match result {
                        Ok(handle) => {
                            self.windows[i].handles[unit] = Some(handle);
                            self.stats.resident_bytes += map_bytes;
                        }
                        Err(e) => {
                            for &(window, unit) in touched.iter().rev() {
                                self.release_unit(window, unit);
                            }
                            return Err(e);
                        }
                    }
                }
                self.windows[i].refs[unit] += 1;
                touched.push((i, unit));
            }
        }
        self.mapped[slot] = true;
        while self.prefix < self.mapped.len() && self.mapped[self.prefix] {
            self.prefix += 1;
        }
        self.stats.mapped_slots += 1;
        self.stats.mapped_prefix = self.prefix;
        Ok(())
    }

    pub fn release_slot(&mut self, slot: usize) {
        if slot >= self.mapped.len() || !self.mapped[slot] {
            return;
        }
        for i in 0..self.windows.len() {
            let slot_bytes = self.windows[i].slot_bytes;
            let map_bytes = self.windows[i].map_bytes;
            let first = slot as u64 * slot_bytes / map_bytes;
            let end = (slot as u64 + 1) * slot_bytes;
            let last = end.div_ceil(map_bytes);
            for unit in first..last {
                self.release_unit(i, unit as usize);
            }
        }
        self.mapped[slot] = false;
        self.prefix = self.prefix.min(slot);
        self.stats.mapped_slots -= 1;
        self.stats.mapped_prefix = self.prefix;
    }

    fn release_unit(&mut self, window: usize, unit: usize) {
        let w = &mut self.windows[window];
        debug_assert!(w.refs[unit] > 0);
        w.refs[unit] -= 1;
        if w.refs[unit] != 0 {
            return;
        }
        let handle = w.handles[unit].take().expect("referenced ring mapping");
        self.ops
            .unmap(w.va + unit as u64 * w.map_bytes, w.map_bytes);
        self.ops.release(handle);
        self.stats.resident_bytes -= w.map_bytes;
    }

    pub fn ensure_prefix(&mut self, rows: usize) -> Result<()> {
        if rows > self.mapped.len() {
            return Err(RuntimeError::Rejected(
                "live ring prefix out of bounds".into(),
            ));
        }
        while self.prefix < rows {
            self.ensure_slot(self.prefix)?;
        }
        Ok(())
    }
}

impl Drop for VmmRings {
    fn drop(&mut self) {
        for w in &mut self.windows {
            for (slot, handle) in w.handles.iter_mut().enumerate() {
                if let Some(handle) = handle.take() {
                    self.ops
                        .unmap(w.va + slot as u64 * w.map_bytes, w.map_bytes);
                    self.ops.release(handle);
                }
            }
            self.ops.address_free(w.va, w.bytes);
        }
    }
}

pub struct LiveKvLayout {
    pub geometry: VmmGeometry,
    /// Synthetic allocator track index → actual packet K/V handles.
    pub full_tensors: Vec<[usize; 2]>,
    pub ring_tensors: Vec<LiveRingTensor>,
    pub cache_tensors: Vec<usize>,
}

impl LiveKvLayout {
    pub fn manifest(
        blob: &crate::asset::devblob::DevBlob,
        raw: &[u8],
    ) -> Result<Option<plow_asset::live_kv::Manifest>> {
        let Some(bytes) = blob.reserved_metadata(raw, plow_asset::live_kv::SECTION)? else {
            return Ok(None);
        };
        let m: plow_asset::live_kv::Manifest = serde_json::from_slice(bytes)
            .map_err(|e| RuntimeError::Rejected(format!("invalid LIVE KV manifest: {e}")))?;
        blob.with_packet_view(|packet| m.validate(packet))
            .map_err(RuntimeError::Rejected)?;
        Ok(Some(m))
    }
    pub fn from_manifest(
        blob: &crate::asset::devblob::DevBlob,
        m: &plow_asset::live_kv::Manifest,
    ) -> Result<Self> {
        blob.with_packet_view(|packet| m.validate(packet))
            .map_err(RuntimeError::Rejected)?;
        Self::from_validated_manifest(blob, m)
    }
    fn from_validated_manifest(
        blob: &crate::asset::devblob::DevBlob,
        m: &plow_asset::live_kv::Manifest,
    ) -> Result<Self> {
        let mut full_tensors = Vec::new();
        let mut ring_tensors = Vec::new();
        let mut cache_tensors = Vec::new();
        let mut full_shape = None;
        for c in &m.caches {
            cache_tensors.extend(c.pair.map(usize::from));
            let elem = if c.scales.is_some() { 1 } else { 2 };
            if c.window == 0 {
                if full_shape.is_some_and(|shape| shape != (c.heads, c.hd, elem)) {
                    return Err(RuntimeError::Rejected(
                        "LIVE allocator requires uniform full-cache head geometry and encoding".into(),
                    ));
                }
                full_shape = Some((c.heads, c.hd, elem));
                full_tensors.push(c.pair.map(usize::from));
            } else {
                for tensor in c.pair.map(usize::from) {
                    ring_tensors.push(LiveRingTensor {
                        tensor,
                        slot_bytes: blob.tensors[tensor].bytes / u64::from(m.batch),
                    });
                }
            }
            if let Some(scales) = c.scales {
                for tensor in scales.map(usize::from) {
                    cache_tensors.push(tensor);
                    ring_tensors.push(LiveRingTensor {
                        tensor,
                        slot_bytes: blob.tensors[tensor].bytes / u64::from(m.batch),
                    });
                }
            }
        }
        let full_shape = full_shape.ok_or_else(|| {
            RuntimeError::Rejected(
                "LIVE allocator does not support sliding-only packets; declared geometry is valid"
                    .into(),
            )
        })?;
        Ok(Self {
            geometry: VmmGeometry {
                full_layers: (0..full_tensors.len() as u32).collect(),
                kvh_full: full_shape.0,
                hd_full: full_shape.1,
                slide_layers: Vec::new(),
                kvh_slide: 0,
                hd_slide: 0,
                window: 0,
                elem: full_shape.2,
                elem_slide: 2,
                max_ctx: m.max_ctx,
                batch: m.batch,
            },
            full_tensors,
            ring_tensors,
            cache_tensors,
        })
    }
    pub fn from_blob(blob: &crate::asset::devblob::DevBlob) -> Result<Self> {
        let manifest = blob
            .with_packet_view(plow_asset::live_kv::emit)
            .map_err(RuntimeError::Rejected)?;
        Self::from_validated_manifest(blob, &manifest)
    }
}

/// Result of a successful prefix attach.
#[derive(Clone, Copy, Debug)]
pub struct Attach {
    /// Reused rows. Whole blocks are shared; a partial block is restored from the snapshot.
    pub rows: u32,
    /// Device VA of the boundary snapshot taken at `rows`; the engine owns
    /// its layout and restores the borrower's rings and partial full-KV block.
    pub snap_va: u64,
    pub snap_bytes: u64,
}

/// Point-in-time pool counters (tests, metrics, the perf campaign).
#[derive(Clone, Copy, Debug, Default)]
pub struct VmmStats {
    /// Physical blocks created (`cuMemCreate`).
    pub blocks_created: u64,
    /// Shared mappings made by attach (each deduped `block_bytes` of HBM).
    pub blocks_shared_mapped: u64,
    /// Radix nodes evicted (their blocks dereferenced).
    pub nodes_evicted: u64,
    /// Physical blocks currently live.
    pub blocks_live: u64,
    /// Blocks currently referenced by the cache (upper bound on evictable).
    pub cache_blocks: u64,
    /// Full-KV blocks and boundary snapshots retained by the cache, including pinned entries.
    pub cache_bytes: u64,
    pub snapshot_bytes: u64,
    pub snapshots_evicted: u64,
    /// Hash collisions caught by radix token verification — each one was a
    /// would-be wrong-KV serve, downgraded to a miss.
    pub hash_collisions: u64,
    /// Attaches that shared a published prefix.
    pub attach_hits: u64,
    /// Fresh sequences that found no attachable prefix.
    pub attach_misses: u64,
    /// Prompt rows served from the cache across all attaches (KV never
    /// recomputed) — the numerator of the fleet hit-rate.
    pub tokens_attached: u64,
    /// Physical blocks currently parked in the reuse pool (unmapped, holding
    /// VRAM). See [`VmmKv::enable_block_pool`].
    pub blocks_pooled: u64,
    /// Block requests served from the reuse pool instead of `create` — each
    /// one a driver page-commit skipped on the request path.
    pub blocks_reused: u64,
}

/// One physical sharing block: driver handle + mapping/cache refcount.
/// The driver refcounts mappings itself (probe [2]) — this host count is
/// policy: it decides when WE release the handle.
struct Block {
    handle: u64,
    refs: u32,
}

/// Per-cache-tensor VA reservation and its mapping table:
/// `slots[(seq·kvh + head)·bph + k]` = the block mapped at that window slot.
struct Track {
    layer: u32,
    /// Caller-supplied tensor role; the legacy K/V layout uses 0/1.
    tensor: u32,
    va: u64,
    slots: Vec<Option<u32>>,
}

#[derive(Clone, Copy)]
struct BoundaryKey {
    node: Option<(u32, u32)>,
    va: u64,
}

#[derive(Default)]
struct SlotSeq {
    hashes: Vec<BlockHash>,
    tokens: Vec<u32>,
    prompt_rows: usize,
    held: usize,
    snapshot: Option<BoundaryKey>,
}

/// A published boundary's sliding-window snapshot buffer.
struct Snap {
    va: u64,
    bytes: u64,
    rows: u32,
    tail: Vec<u32>,
    users: usize,
    last_used: u64,
    referenced: bool,
    reusable_prompt: bool,
}

struct Inner {
    tracks: Vec<Track>,
    blocks: Vec<Block>,
    free_ids: Vec<u32>,
    /// Zero-ref physical handles kept for reuse instead of released
    /// ([`VmmKv::enable_block_pool`]); every entry is `block_bytes` long.
    pooled: Vec<u64>,
    /// Whole blocks mapped per sequence (uniform across tracks/heads).
    seq_blocks: Vec<u32>,
    cache: PrefixCache,
    /// Radix node identity → the block ids backing it, ordered [track][head].
    node_blocks: FxHashMap<(u32, u32), Vec<u32>>,
    /// Whole-block prefix → snapshots at boundaries within the next block.
    /// None holds prefixes shorter than one physical block.
    published: FxHashMap<Option<(u32, u32)>, Vec<Snap>>,
    snapshot_tick: u64,
    /// Monotonic publish id, used as the radix `owner_seq` so slot reuse can
    /// never collide two nodes on the same key.
    next_pub: u32,
    seqs: Vec<SlotSeq>,
    stats: VmmStats,
}

enum PublishLocked {
    Done { unused_snapshot: Option<u64> },
    NeedSnapshot,
}

struct Shared {
    ops: Arc<dyn VmmOps>,
    geo: VmmGeometry,
    block_bytes: u64,
    block_rows: u32,
    /// Blocks per head window (`head_span / block_bytes`).
    bph: u32,
    head_span: u64,
    /// Cache soft cap in bytes (0 = only OOM-driven eviction).
    cache_cap: u64,
    /// Max handles the reuse pool may hold (0 = pooling off, the default).
    /// Set once by [`VmmKv::enable_block_pool`]; atomic only so `deref_block`
    /// can read it without threading a config borrow through `Inner`.
    pool_cap: AtomicU32,
    inner: Mutex<Inner>,
    /// Per-seq mapped-row frontier, readable lock-free on the decode path.
    frontier: Vec<AtomicU32>,
    generation: Vec<AtomicU64>,
}

/// The VMM-backed KV pool + prefix cache. One per engine; owns the VA
/// reservations, the physical block slab, the radix cache and the pre-mapper
/// thread. Everything is reclaimed on Drop (the lifecycle-test contract).
pub struct VmmKv {
    prefix_reuse: bool,
    shared: Arc<Shared>,
    premap_tx: Option<std::sync::mpsc::Sender<(u32, u32, u64)>>,
    premap_join: Option<std::thread::JoinHandle<()>>,
    /// Pre-creator thread ([`Self::enable_block_pool`]): stop flag + join.
    precreate: Option<(
        Arc<std::sync::atomic::AtomicBool>,
        std::thread::JoinHandle<()>,
    )>,
}

impl VmmKv {
    /// Reserve every (full layer, K|V) VA window (cheap — µs each, probe [6]),
    /// size the sharing block, and spawn the pre-mapper.
    ///
    /// `block_hint` is the requested sharing-block size in bytes (64 MiB-class
    /// per the feasibility review); it is clamped to the head window and
    /// rounded to the driver granularity. `cache_cap` soft-caps cache-held
    /// bytes (0 = unbounded, eviction on OOM only).
    pub fn new(
        ops: Arc<dyn VmmOps>,
        geo: VmmGeometry,
        block_hint: u64,
        cache_cap: u64,
    ) -> Result<Self> {
        Self::new_with_policy(ops, geo, block_hint, cache_cap, true, None)
    }

    /// Share explicitly listed, equally shaped tensors, including unpaired MLA caches.
    pub fn new_tensors(
        ops: Arc<dyn VmmOps>,
        geo: VmmGeometry,
        block_hint: u64,
        cache_cap: u64,
        tensors: &[(u32, u32)],
    ) -> Result<Self> {
        Self::new_with_policy(ops, geo, block_hint, cache_cap, true, Some(tensors))
    }

    pub fn new_live(ops: Arc<dyn VmmOps>, geo: VmmGeometry, block_hint: u64) -> Result<Self> {
        Self::new_with_policy(ops, geo, block_hint, 0, false, None)
    }

    pub fn prefix_reuse(&self) -> bool {
        self.prefix_reuse
    }

    fn new_with_policy(
        ops: Arc<dyn VmmOps>,
        geo: VmmGeometry,
        block_hint: u64,
        cache_cap: u64,
        prefix_reuse: bool,
        tensors: Option<&[(u32, u32)]>,
    ) -> Result<Self> {
        let tensors = tensors.map_or_else(
            || geo.full_layers.iter().flat_map(|&layer| [(layer, 0), (layer, 1)]).collect(),
            <[_]>::to_vec,
        );
        let mut unique = rustc_hash::FxHashSet::default();
        if tensors.is_empty()
            || tensors.iter().any(|&(layer, tensor)| {
                !geo.full_layers.contains(&layer) || !unique.insert((layer, tensor))
            })
        {
            return Err(RuntimeError::Rejected("vmm: invalid or duplicate cache tensor tracks".into()));
        }
        let gran = ops.granularity()?;
        let row_bytes = geo.row_bytes();
        let head_span = geo.max_ctx as u64 * row_bytes;
        let block_bytes = geo.block_bytes(gran, block_hint)?;
        let block_rows = (block_bytes / row_bytes) as u32;
        let bph = (head_span / block_bytes) as u32;

        let span = geo.batch as u64 * geo.kvh_full as u64 * head_span;
        let ntracks = tensors.len();
        let tracks = Vec::with_capacity(ntracks);
        let nslots = (geo.batch * geo.kvh_full * bph) as usize;

        // Dummy pool: the radix cache's `runs` are unused here (the payload
        // rides the `placed` side tables), so the geometry only has to be
        // internally consistent and overflow-free. `block_rows` is the REAL
        // block granularity though — the cache verifies node tokens against
        // the prompt at that chunking.
        let cache = PrefixCache::new(
            GrowablePool {
                base: 0,
                kv_factor: 1,
                kv_heads: 1,
                max_seqs: u32::MAX,
                head_slot_bytes: 1 << 32,
            },
            block_rows as u64,
            1,
        )
        .expect("dummy radix pool geometry");

        let batch = geo.batch as usize;
        let shared = Arc::new(Shared {
            ops,
            geo,
            block_bytes,
            block_rows,
            bph,
            head_span,
            cache_cap,
            pool_cap: AtomicU32::new(0),
            inner: Mutex::new(Inner {
                tracks,
                blocks: Vec::new(),
                free_ids: Vec::new(),
                pooled: Vec::new(),
                seq_blocks: vec![0; batch],
                cache,
                node_blocks: FxHashMap::default(),
                published: FxHashMap::default(),
                snapshot_tick: 0,
                next_pub: 0,
                seqs: (0..batch).map(|_| SlotSeq::default()).collect(),
                stats: VmmStats::default(),
            }),
            frontier: (0..batch).map(|_| AtomicU32::new(0)).collect(),
            generation: (0..batch).map(|_| AtomicU64::new(0)).collect(),
        });
        let mut pool = VmmKv {
            prefix_reuse,
            shared: Arc::clone(&shared),
            premap_tx: None,
            premap_join: None,
            precreate: None,
        };
        {
            let mut inner = shared.inner.lock();
            for (layer, tensor) in tensors {
                let va = shared.ops.reserve(span)?;
                inner.tracks.push(Track {
                    layer,
                    tensor,
                    va,
                    slots: vec![None; nslots],
                });
            }
        }

        // Pre-mapper: keeps the NEXT block mapped ahead of decode growth so
        // the 2048-token boundary never stalls a step (plan verdict §4 —
        // ~6.8 ms if synchronous, free when overlapped). `advise` feeds it;
        // `ensure_rows` in the step path is the correctness backstop.
        let (tx, rx) = std::sync::mpsc::channel::<(u32, u32, u64)>();
        pool.premap_tx = Some(tx);
        let premap_shared = Arc::clone(&shared);
        let join = std::thread::Builder::new()
            .name("vmm-premap".into())
            .spawn(move || {
                while let Ok((seq, pos, generation)) = rx.recv() {
                    let s = &premap_shared;
                    let target = ((pos / s.block_rows) + 2)
                        .saturating_mul(s.block_rows)
                        .min(s.geo.max_ctx);
                    if s.frontier[seq as usize].load(Ordering::Acquire) < target {
                        if let Err(e) = ensure_rows(s, seq as usize, target, Some(generation)) {
                            // Non-fatal here: the synchronous backstop in the
                            // step path surfaces the real error.
                            tracing::warn!(error = %e, seq, target, "vmm pre-map failed");
                        }
                    }
                }
            })
            .map_err(|e| RuntimeError::Device(format!("vmm premap thread: {e}")))?;
        pool.premap_join = Some(join);

        tracing::info!(
            full_layers = shared.geo.full_layers.len(),
            kv_elem = shared.geo.elem,
            kv_elem_slide = shared.geo.elem_slide,
            block_mib = block_bytes >> 20,
            block_rows,
            bph,
            va_gib = (span * ntracks as u64) as f64 / (1u64 << 30) as f64,
            "vmm kv pool up (full layers VMM-backed, sliding on cudaMalloc)"
        );
        Ok(pool)
    }

    /// Turn on physical-block reuse: zero-ref blocks park in a pool (up to
    /// `cap_bytes`) instead of being released, and `ensure_rows` draws from
    /// the pool before calling the driver — the request path pays map +
    /// set_access (µs-class) instead of `cuMemCreate`'s serial page commit
    /// (~13 GiB/s measured, 5-30 ms of TTFT for a fresh sequence's first
    /// window). A background thread pre-creates roughly two window columns'
    /// worth of blocks so even the FIRST request after load skips the commit.
    ///
    /// Opt-in (engines call this off `PLOW_KV_POOL_MIB`, default 512): pooled
    /// blocks hold VRAM while idle, and the exact-driver-call-count unit tests
    /// rely on the default-off behavior.
    pub fn enable_block_pool(&mut self, cap_bytes: u64) {
        use std::sync::atomic::AtomicBool;
        let s = &self.shared;
        let cap_blocks = (cap_bytes / s.block_bytes).min(u32::MAX as u64) as u32;
        s.pool_cap.store(cap_blocks, Ordering::Relaxed);
        if cap_blocks == 0 || self.precreate.is_some() {
            return;
        }
        // Two window columns for a fresh sequence: what `ensure_rows` maps
        // before decode settles into the premap thread's +2 lookahead.
        let ntracks = s.inner.lock().tracks.len() as u32;
        let target = (ntracks * s.geo.kvh_full * 2).min(cap_blocks) as usize;
        if target == 0 {
            return;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let (t_s, t_stop) = (Arc::clone(&self.shared), Arc::clone(&stop));
        let join = std::thread::Builder::new()
            .name("vmm-kv-pool".into())
            .spawn(move || {
                for _ in 0..target {
                    if t_stop.load(Ordering::Acquire) || t_s.inner.lock().pooled.len() >= target {
                        break;
                    }
                    // Create OUTSIDE the lock — this thread must never add
                    // its commit latency to a concurrent request's
                    // `ensure_rows`.
                    match t_s.ops.create(t_s.block_bytes) {
                        Ok(h) => {
                            let mut inner = t_s.inner.lock();
                            inner.pooled.push(h);
                            inner.stats.blocks_pooled += 1;
                            inner.stats.blocks_created += 1;
                        }
                        // Best-effort: a card too full to pre-create serves
                        // demand through create_block's evict-on-OOM path.
                        Err(_) => break,
                    }
                }
            });
        if let Ok(j) = join {
            self.precreate = Some((stop, j));
        }
    }

    /// Recycle retired blocks up to `cap_bytes` without committing HBM before demand.
    pub fn enable_block_recycling(&mut self, cap_bytes: u64) {
        let cap_blocks = (cap_bytes / self.shared.block_bytes).min(u32::MAX as u64) as u32;
        self.shared.pool_cap.store(cap_blocks, Ordering::Relaxed);
    }

    /// VA base of the (layer, tensor) full-layer KV tensor — what the engine
    /// puts in the tensor table instead of a cudaMalloc base. `tensor`: 0 = K,
    /// 1 = V. `None` when `layer` is not a full layer.
    pub fn tensor_va(&self, layer: u32, tensor: u32) -> Option<u64> {
        let inner = self.shared.inner.lock();
        inner
            .tracks
            .iter()
            .find(|t| t.layer == layer && t.tensor == tensor)
            .map(|t| t.va)
    }

    /// Tokens per sharing block — the prefix-match granularity.
    pub fn block_rows(&self) -> u32 {
        self.shared.block_rows
    }

    pub fn geometry(&self) -> &VmmGeometry {
        &self.shared.geo
    }

    /// Mapped-row frontier for `seq` (lock-free; the per-step fast check).
    pub fn mapped_rows(&self, seq: usize) -> u32 {
        self.shared.frontier[seq].load(Ordering::Acquire)
    }

    /// Synchronously map blocks until at least `rows` rows are writable for
    /// `seq` across every full-layer track and head. OOM evicts cache LRU
    /// nodes before failing.
    pub fn ensure_rows(&self, seq: usize, rows: u32) -> Result<()> {
        ensure_rows(&self.shared, seq, rows, None)
    }

    /// Decode-growth hint: ask the pre-mapper to keep the next block mapped
    /// beyond `pos`. Never blocks; drops silently after shutdown.
    pub fn advise(&self, seq: usize, pos: u32) {
        if let Some(tx) = &self.premap_tx {
            let generation = self.shared.generation[seq].load(Ordering::Acquire);
            let _ = tx.send((seq as u32, pos, generation));
        }
    }

    /// Start a new sequence in `seq`: release the previous sequence's radix
    /// references and unmap+deref its window (cached blocks survive through
    /// the cache's own references).
    pub fn begin_seq(&self, seq: usize) {
        let s = &self.shared;
        let mut inner = s.inner.lock();
        s.generation[seq].fetch_add(1, Ordering::Release);
        release_prefix_hold(&mut inner, seq);
        release_window(s, &mut inner, seq);
        trim_cache(s, &mut inner);
    }

    /// Release a finished request's cache holds while retaining its writable KV mappings.
    /// CUDA's inactive decode rows can still write at the saved frontier.
    pub fn release_prefix(&self, seq: usize) {
        let s = &self.shared;
        let mut inner = s.inner.lock();
        release_prefix_hold(&mut inner, seq);
        trim_cache(s, &mut inner);
    }

    pub fn finish_attach(&self, seq: usize) {
        let s = &self.shared;
        let mut inner = s.inner.lock();
        release_snapshot_hold(&mut inner, seq);
        trim_cache(s, &mut inner);
    }

    /// Try to attach a cached prefix of `prompt` into `seq`'s windows:
    /// longest radix match, clipped to the longest **published boundary**
    /// (snapshot available) and to `< prompt.len()` so at least the last
    /// token is recomputed by prefill. On a hit the shared
    /// blocks are multi-mapped (refcounted) and the snapshot handle returned.
    /// On miss the prompt's hashes are still recorded for `publish`.
    pub fn try_attach(&self, seq: usize, prompt: &[u32]) -> Result<Option<Attach>> {
        if !self.prefix_reuse {
            return Ok(None);
        }
        let s = &self.shared;
        let hashes = hash_blocks(prompt, s.block_rows);
        let aligned = &prompt[..hashes.len() * s.block_rows as usize];
        let mut inner = s.inner.lock();

        let m = inner.cache.lookup(&hashes, aligned);
        let mut chosen = None;
        for blocks in 0..=m.blocks {
            let node = blocks.checked_sub(1).map(|i| m.placed[i]);
            if let Some(snapshots) = inner.published.get(&node) {
                let start = blocks * s.block_rows as usize;
                for snap in snapshots {
                    if (snap.rows as usize) < prompt.len()
                        && prompt.get(start..snap.rows as usize) == Some(snap.tail.as_slice())
                        && chosen.is_none_or(|(_, _, rows)| snap.rows > rows)
                    {
                        chosen = Some((blocks, BoundaryKey { node, va: snap.va }, snap.rows));
                    }
                }
            }
        }
        let Some((pick, snap_key, rows)) = chosen else {
            inner.stats.attach_misses += 1;
            inner.cache.release(&hashes, m.blocks);
            inner.seqs[seq] = SlotSeq {
                hashes,
                tokens: prompt.to_vec(),
                prompt_rows: prompt.len(),
                held: 0,
                snapshot: None,
            };
            return Ok(None);
        };
        inner.stats.attach_hits += 1;
        inner.stats.tokens_attached += rows as u64;
        if pick < m.blocks {
            // Keep references only on the attached prefix of the path.
            inner.cache.release(&hashes, m.blocks);
            let again = inner.cache.lookup(&hashes[..pick], aligned);
            debug_assert_eq!(again.blocks, pick);
        }
        let placed: Vec<(u32, u32)> = m.placed[..pick].to_vec();
        inner.seqs[seq] = SlotSeq {
            hashes,
            tokens: prompt.to_vec(),
            prompt_rows: prompt.len(),
            held: pick,
            snapshot: Some(snap_key),
        };
        inner.snapshot_tick += 1;
        let tick = inner.snapshot_tick;
        let snap = inner.published.get_mut(&snap_key.node).unwrap()
            .iter_mut().find(|s| s.va == snap_key.va).unwrap();
        snap.users += 1;
        snap.last_used = tick;
        snap.referenced = true;
        snap.reusable_prompt = true;
        let attach = Attach { rows, snap_va: snap.va, snap_bytes: snap.bytes };

        // COMMIT the whole attach under the lock — slot table, refcounts,
        // frontier — but only COLLECT the driver work. The map/set_access
        // calls are ~69 µs each (ms-scale for a long prefix) and used to
        // serialize the pre-mapper and every other slot behind this lock.
        // Deferring them is safe: slot ownership is arbitrated here, so
        // concurrent ensure_rows can only ever touch OTHER window slots
        // (disjoint VAs), and the driver itself is thread-safe. The blocks
        // can't die mid-flight — the refs taken here keep them.
        let mut unmaps: Vec<u64> = Vec::new();
        let mut frees: Vec<u64> = Vec::new();
        let mut maps: Vec<(u64, u64)> = Vec::new();

        // `begin_seq` pre-maps row 0 (idle-row garbage writes land there);
        // that private block occupies window slot 0, which the shared prefix
        // is about to claim — drop the fresh window before multi-mapping.
        if inner.seq_blocks[seq] > 0 {
            for t in 0..inner.tracks.len() {
                for h in 0..s.geo.kvh_full {
                    for k in 0..s.bph {
                        let slot = slot_index(s, seq, h, k);
                        if let Some(id) = inner.tracks[t].slots[slot].take() {
                            unmaps.push(slot_va(s, &inner.tracks[t], seq, h, k));
                            if let Some(h) = unref_block(s, &mut inner, id) {
                                frees.push(h);
                            }
                        }
                    }
                }
            }
            inner.seq_blocks[seq] = 0;
            s.frontier[seq].store(0, Ordering::Release);
        }

        // Multi-map every shared block into this sequence's window slots.
        let kvh = s.geo.kvh_full as usize;
        for (k, key) in placed.iter().enumerate() {
            let ids = inner
                .node_blocks
                .get(key)
                .ok_or_else(|| {
                    RuntimeError::Device(format!("vmm: node {key:?} has no block payload"))
                })?
                .clone();
            debug_assert_eq!(ids.len(), inner.tracks.len() * kvh);
            for (j, &id) in ids.iter().enumerate() {
                let (t, h) = (j / kvh, j % kvh);
                let va = slot_va(s, &inner.tracks[t], seq, h as u32, k as u32);
                maps.push((va, inner.blocks[id as usize].handle));
                inner.blocks[id as usize].refs += 1;
                let slot = slot_index(s, seq, h as u32, k as u32);
                debug_assert!(inner.tracks[t].slots[slot].is_none());
                inner.tracks[t].slots[slot] = Some(id);
                inner.stats.blocks_shared_mapped += 1;
            }
            inner.seq_blocks[seq] = k as u32 + 1;
            s.frontier[seq].store((k as u32 + 1) * s.block_rows, Ordering::Release);
        }
        drop(inner);

        // Drive the driver lock-free. Unmaps precede maps (slot 0 is reused).
        for &va in &unmaps {
            s.ops.unmap(va, s.block_bytes);
        }
        for &h in &frees {
            s.ops.release(h);
        }
        // Map every granule, then grant access with ONE set_access per
        // contiguous VA run — set_access is the ~69 µs call, and consecutive
        // blocks of one (track, head) window are VA-adjacent, so an N-block
        // prefix costs O(tracks × heads) grants instead of O(N × ...).
        maps.sort_unstable_by_key(|&(va, _)| va);
        let mut mapped = 0usize; // maps[..mapped] are live on the device
        let mut err: Option<RuntimeError> = None;
        for &(va, handle) in &maps {
            match s.ops.map(va, s.block_bytes, handle) {
                Ok(()) => mapped += 1,
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        if err.is_none() {
            let mut i = 0usize;
            while i < maps.len() {
                let start = maps[i].0;
                let mut end = start + s.block_bytes;
                while i + 1 < maps.len() && maps[i + 1].0 == end {
                    i += 1;
                    end += s.block_bytes;
                }
                if let Err(e) = s.ops.set_access(start, end - start) {
                    err = Some(e);
                    break;
                }
                i += 1;
            }
        }
        if let Some(e) = err {
            // Unwind. Unmap ONLY the ranges this attach actually mapped —
            // the backend's unmap contract is exact mapped ranges
            // (`device/cuda.rs`), never a blanket sweep — then tear the
            // window bookkeeping down to empty and drop the path references.
            for &(va, _) in &maps[..mapped] {
                s.ops.unmap(va, s.block_bytes);
            }
            let mut inner = s.inner.lock();
            inner.stats.blocks_shared_mapped -= maps.len() as u64;
            let mut orphans: Vec<u64> = Vec::new();
            for t in 0..inner.tracks.len() {
                for h in 0..s.geo.kvh_full {
                    for k in 0..s.bph {
                        let slot = slot_index(s, seq, h, k);
                        if let Some(id) = inner.tracks[t].slots[slot].take() {
                            if let Some(h) = unref_block(s, &mut inner, id) {
                                orphans.push(h);
                            }
                        }
                    }
                }
            }
            inner.seq_blocks[seq] = 0;
            s.frontier[seq].store(0, Ordering::Release);
            release_prefix_hold(&mut inner, seq);
            drop(inner);
            for hnd in orphans {
                s.ops.release(hnd);
            }
            return Err(e);
        }
        Ok(Some(attach))
    }

    /// Publish `seq`'s computed rows: insert `tokens`' whole blocks into
    /// the radix tree (COW — pre-existing nodes are left alone), reference
    /// the backing physical blocks from the cache, and store the boundary's
    /// sliding-window snapshot (`snap_bytes` device bytes, written by `fill`
    /// into a fresh buffer — the engine D2D-copies its rings there).
    ///
    /// `tokens` is the token id per KV row the slot holds — the prompt at
    /// prefill completion, prompt + generated at sequence end (the tail
    /// publish). It must extend the stream recorded at attach time; the
    /// chained block hashes guarantee any prefix the slot already holds
    /// references resolves identically.
    pub fn publish(
        &self,
        seq: usize,
        tokens: &[u32],
        snap_bytes: u64,
        fill: impl FnOnce(u64) -> Result<()>,
    ) -> Result<()> {
        let rows = (tokens.len() / self.block_rows() as usize) as u32 * self.block_rows();
        self.publish_at(seq, tokens, rows, snap_bytes, fill)
    }

    pub fn publish_at(
        &self,
        seq: usize,
        tokens: &[u32],
        rows: u32,
        snap_bytes: u64,
        fill: impl FnOnce(u64) -> Result<()>,
    ) -> Result<()> {
        if !self.prefix_reuse {
            return Err(RuntimeError::Rejected(
                "live KV allocation cannot publish prefixes".into(),
            ));
        }
        let s = &self.shared;
        if rows == 0 {
            return Ok(());
        }
        let generation = s.generation[seq].load(Ordering::Acquire);
        {
            let mut inner = s.inner.lock();
            if let PublishLocked::Done { unused_snapshot } =
                publish_locked(s, &mut inner, seq, tokens, rows, snap_bytes, None)?
            {
                debug_assert!(unused_snapshot.is_none());
                return Ok(());
            }
        }

        let va = alloc_snapshot(s, snap_bytes)?;
        if let Err(error) = fill(va) {
            s.ops.free(va);
            return Err(error);
        }
        if s.generation[seq].load(Ordering::Acquire) != generation {
            s.ops.free(va);
            return Err(RuntimeError::Rejected(
                "vmm: sequence changed during prefix publication".into(),
            ));
        }

        let result = {
            let mut inner = s.inner.lock();
            publish_locked(s, &mut inner, seq, tokens, rows, snap_bytes, Some(va))
        };
        match result {
            Ok(PublishLocked::Done { unused_snapshot }) => {
                if let Some(unused) = unused_snapshot {
                    s.ops.free(unused);
                }
                Ok(())
            }
            Ok(PublishLocked::NeedSnapshot) => unreachable!("snapshot was supplied"),
            Err(error) => {
                s.ops.free(va);
                Err(error)
            }
        }
    }

    pub fn stats(&self) -> VmmStats {
        stats_of(&self.shared)
    }

    /// Cloneable stats reader for exporters. Takes only the POOL mutex
    /// (µs-scale holds), never the engine mutex — a metrics scrape must not
    /// queue behind a running tick. Reads after the pool drops return the
    /// last values (the Arc keeps the bookkeeping alive, not the device
    /// resources).
    pub fn stats_handle(&self) -> VmmStatsHandle {
        VmmStatsHandle(Arc::clone(&self.shared))
    }
}

/// See [`VmmKv::stats_handle`].
#[derive(Clone)]
pub struct VmmStatsHandle(Arc<Shared>);

impl VmmStatsHandle {
    pub fn stats(&self) -> VmmStats {
        stats_of(&self.0)
    }
}

fn alloc_snapshot(s: &Shared, bytes: u64) -> Result<u64> {
    loop {
        match s.ops.alloc(bytes) {
            Ok(va) => return Ok(va),
            Err(error) if matches!(&error, RuntimeError::Oom(_)) => {
                let mut inner = s.inner.lock();
                if !evict_one(s, &mut inner, false) {
                    return Err(error);
                }
            }
            Err(error) => return Err(error),
        }
    }
}

fn publish_locked(
    s: &Shared,
    inner: &mut Inner,
    seq: usize,
    tokens: &[u32],
    rows: u32,
    snap_bytes: u64,
    snapshot: Option<u64>,
) -> Result<PublishLocked> {
    if rows as usize > tokens.len()
        || rows > s.geo.max_ctx
        || rows.div_ceil(s.block_rows) > inner.seq_blocks[seq]
        || snap_bytes == 0
    {
        return Err(RuntimeError::Rejected("vmm: unpublished rows or empty snapshot".into()));
    }
    let prior = &inner.seqs[seq].tokens;
    let overlap = prior.len().min(tokens.len());
    if tokens[..overlap] != prior[..overlap] {
        return Err(RuntimeError::Rejected("vmm: published tokens changed the attached stream".into()));
    }
    let hashes = hash_blocks(&tokens[..rows as usize], s.block_rows);
    let n_pub = hashes.len();
    let prompt_rows = match inner.seqs[seq].prompt_rows {
        0 => tokens.len(),
        rows => rows,
    };
    let reusable_prompt = (rows as usize) < prompt_rows;
    let tail = &tokens[n_pub * s.block_rows as usize..rows as usize];

    let m = inner.cache.lookup(&hashes, tokens);
    let matched_key = if n_pub == 0 {
        Some(None)
    } else if n_pub <= m.blocks {
        Some(Some(m.placed[n_pub - 1]))
    } else {
        None
    };
    let snapshot_exists = matched_key.is_some_and(|key| {
        inner.published.get(&key).is_some_and(|list| {
            list.iter().any(|snap| snap.rows == rows && snap.tail == tail)
        })
    });
    if snapshot.is_none() && !snapshot_exists {
        inner.cache.release(&hashes, m.blocks);
        return Ok(PublishLocked::NeedSnapshot);
    }

    let pid = inner.next_pub;
    inner.next_pub += 1;
    let n_ok = inner.cache.insert(&hashes, tokens, pid, m.blocks);
    if n_ok != n_pub {
        inner.cache.release(&hashes, m.blocks);
        return Err(RuntimeError::Rejected("vmm: prefix publication collided".into()));
    }

    let kvh = s.geo.kvh_full as usize;
    for idx in m.blocks..n_pub {
        let mut ids = Vec::with_capacity(inner.tracks.len() * kvh);
        for t in 0..inner.tracks.len() {
            for h in 0..kvh {
                let slot = slot_index(s, seq, h as u32, idx as u32);
                let id = inner.tracks[t].slots[slot]
                    .expect("published block below the mapped frontier");
                ids.push(id);
            }
        }
        for &id in &ids {
            inner.blocks[id as usize].refs += 1;
        }
        inner.stats.cache_blocks += ids.len() as u64;
        inner.stats.cache_bytes += ids.len() as u64 * s.block_bytes;
        inner.node_blocks.insert((pid, idx as u32), ids);
    }
    release_prefix_hold(inner, seq);
    inner.seqs[seq] = SlotSeq {
        tokens: tokens.to_vec(),
        prompt_rows,
        hashes,
        held: n_pub,
        snapshot: None,
    };

    let bkey = if n_pub == 0 {
        None
    } else if n_pub <= m.blocks {
        Some(m.placed[n_pub - 1])
    } else {
        Some((pid, n_pub as u32 - 1))
    };
    inner.snapshot_tick += 1;
    let tick = inner.snapshot_tick;
    let unused_snapshot = if let Some(snap) = inner.published.get_mut(&bkey)
        .and_then(|list| list.iter_mut().find(|snap| snap.rows == rows && snap.tail == tail))
    {
        snap.last_used = tick;
        snap.reusable_prompt |= reusable_prompt;
        snapshot
    } else {
        let va = snapshot.expect("preflight cannot commit a missing snapshot");
        inner.published.entry(bkey).or_default().push(Snap {
            va,
            bytes: snap_bytes,
            rows,
            tail: tail.to_vec(),
            users: 0,
            last_used: tick,
            referenced: false,
            reusable_prompt,
        });
        inner.stats.snapshot_bytes += snap_bytes;
        inner.stats.cache_bytes += snap_bytes;
        None
    };

    trim_cache(s, inner);
    Ok(PublishLocked::Done { unused_snapshot })
}

fn stats_of(s: &Shared) -> VmmStats {
    let inner = s.inner.lock();
    let mut out = inner.stats;
    out.hash_collisions = inner.cache.collisions();
    out
}

impl Drop for VmmKv {
    /// Full teardown: stop the pre-mapper, unmap every window, drop every
    /// cache reference, release every physical block and snapshot, free the
    /// VA reservations. The engine synchronizes the device before dropping.
    fn drop(&mut self) {
        drop(self.premap_tx.take());
        if let Some(j) = self.premap_join.take() {
            let _ = j.join();
        }
        if let Some((stop, j)) = self.precreate.take() {
            stop.store(true, Ordering::Release);
            let _ = j.join();
        }
        let s = &self.shared;
        let mut inner = s.inner.lock();
        for seq in 0..s.geo.batch as usize {
            release_window(s, &mut inner, seq);
        }
        for (_, ids) in std::mem::take(&mut inner.node_blocks) {
            for id in ids {
                deref_block(s, &mut inner, id);
            }
        }
        for (_, snapshots) in std::mem::take(&mut inner.published) {
            for snap in snapshots {
                s.ops.free(snap.va);
            }
        }
        inner.stats.cache_blocks = 0;
        inner.stats.cache_bytes = 0;
        inner.stats.snapshot_bytes = 0;
        for handle in std::mem::take(&mut inner.pooled) {
            s.ops.release(handle);
        }
        inner.stats.blocks_pooled = 0;
        debug_assert_eq!(inner.stats.blocks_live, 0, "vmm blocks leaked at drop");
        let span = s.geo.batch as u64 * s.geo.kvh_full as u64 * s.head_span;
        for t in &inner.tracks {
            s.ops.address_free(t.va, span);
        }
    }
}

/// Window-slot index for `(seq, head, block k)` within a track.
#[inline]
fn slot_index(s: &Shared, seq: usize, head: u32, k: u32) -> usize {
    ((seq as u32 * s.geo.kvh_full + head) * s.bph + k) as usize
}

/// Device VA of that slot.
#[inline]
fn slot_va(s: &Shared, track: &Track, seq: usize, head: u32, k: u32) -> u64 {
    track.va
        + (seq as u64 * s.geo.kvh_full as u64 + head as u64) * s.head_span
        + k as u64 * s.block_bytes
}

fn ensure_rows(s: &Shared, seq: usize, rows: u32, generation: Option<u64>) -> Result<()> {
    let rows = rows.min(s.geo.max_ctx);
    if s.frontier[seq].load(Ordering::Acquire) >= rows {
        return Ok(());
    }
    let mut inner = s.inner.lock();
    // A queued hint must not recreate a retired or reused sequence's mappings.
    if generation.is_some_and(|g| g != s.generation[seq].load(Ordering::Acquire)) {
        return Ok(());
    }
    let target = rows.div_ceil(s.block_rows);
    let kvh = s.geo.kvh_full;
    for k in inner.seq_blocks[seq]..target {
        for t in 0..inner.tracks.len() {
            for h in 0..kvh {
                let slot = slot_index(s, seq, h, k);
                if inner.tracks[t].slots[slot].is_some() {
                    continue;
                }
                let id = create_block(s, &mut inner)?;
                let va = slot_va(s, &inner.tracks[t], seq, h, k);
                let handle = inner.blocks[id as usize].handle;
                if let Err(error) = s.ops.map(va, s.block_bytes, handle) {
                    deref_block(s, &mut inner, id);
                    return Err(error);
                }
                if let Err(error) = s.ops.set_access(va, s.block_bytes) {
                    s.ops.unmap(va, s.block_bytes);
                    deref_block(s, &mut inner, id);
                    return Err(error);
                }
                inner.tracks[t].slots[slot] = Some(id);
            }
        }
        inner.seq_blocks[seq] = k + 1;
        s.frontier[seq].store((k + 1) * s.block_rows, Ordering::Release);
    }
    Ok(())
}

/// Create one physical block — reuse-pool first ([`VmmKv::enable_block_pool`]),
/// then the driver, evicting cache LRU on allocation failure (the OOM half of
/// finding #9's auto-eviction).
///
/// The pool check lives INSIDE the retry loop: an eviction's zero-ref blocks
/// park in the pool rather than freeing VRAM, so after `evict_one` the next
/// usable block is a pooled handle, not a driver create — checking the pool
/// only on entry would spin `create`-fail/evict until the cache ran dry and
/// then report a spurious OOM with reusable handles in hand.
fn create_block(s: &Shared, inner: &mut Inner) -> Result<u32> {
    loop {
        if let Some(handle) = inner.pooled.pop() {
            inner.stats.blocks_pooled -= 1;
            inner.stats.blocks_reused += 1;
            inner.stats.blocks_live += 1;
            return Ok(install_block(inner, Block { handle, refs: 1 }));
        }
        match s.ops.create(s.block_bytes) {
            Ok(handle) => {
                inner.stats.blocks_created += 1;
                inner.stats.blocks_live += 1;
                return Ok(install_block(inner, Block { handle, refs: 1 }));
            }
            Err(e) => {
                // A fatal fault (poisoned context) is not memory pressure —
                // evicting the cache cannot help and would spin it dry.
                if e.is_fatal() {
                    return Err(e);
                }
                if !evict_one(s, inner, false) {
                    return Err(RuntimeError::Oom(format!("vmm kv block: {e}")));
                }
            }
        }
    }
}

/// Slot a block into the table, recycling a freed id when one exists.
fn install_block(inner: &mut Inner, block: Block) -> u32 {
    match inner.free_ids.pop() {
        Some(id) => {
            inner.blocks[id as usize] = block;
            id
        }
        None => {
            inner.blocks.push(block);
            (inner.blocks.len() - 1) as u32
        }
    }
}

fn trim_cache(s: &Shared, inner: &mut Inner) {
    while s.cache_cap > 0 && inner.stats.cache_bytes > s.cache_cap {
        if !evict_one(s, inner, true) {
            break;
        }
    }
}

fn release_snapshot_hold(inner: &mut Inner, seq: usize) {
    if let Some(key) = inner.seqs[seq].snapshot.take() {
        let snap = inner.published.get_mut(&key.node).unwrap()
            .iter_mut().find(|snap| snap.va == key.va).unwrap();
        snap.users -= 1;
    }
}

fn release_prefix_hold(inner: &mut Inner, seq: usize) {
    release_snapshot_hold(inner, seq);
    let held = std::mem::take(&mut inner.seqs[seq]);
    if held.held > 0 {
        inner.cache.release(&held.hashes, held.held);
    }
}

fn free_snapshot(s: &Shared, inner: &mut Inner, snap: Snap) {
    assert_eq!(snap.users, 0, "evicting an in-flight snapshot");
    inner.stats.snapshot_bytes -= snap.bytes;
    inner.stats.cache_bytes -= snap.bytes;
    inner.stats.snapshots_evicted += 1;
    s.ops.free(snap.va);
}

/// Reclaim output snapshots first, then LRU cache entries. `false` when pinned.
fn evict_one(s: &Shared, inner: &mut Inner, preserve_hot: bool) -> bool {
    // Output-only boundaries cannot replay the original prompt. Reclaim them
    // before removing prompt snapshots or the KV blocks those snapshots need.
    if let Some((node, index)) = inner.published.iter()
        .flat_map(|(&node, snaps)| snaps.iter().enumerate().map(move |(i, snap)| (node, i, snap)))
        .filter(|(_, _, snap)| snap.users == 0 && !snap.reusable_prompt)
        .min_by_key(|(_, _, snap)| snap.last_used)
        .map(|(node, index, _)| (node, index))
    {
        remove_snapshot(s, inner, node, index);
        return true;
    }
    let Some(key) = inner.cache.evict_lru() else {
        // A radix lease protects shared KV, but snapshots are only needed while
        // restoring an attachment. Protect the most recently reused snapshot
        // against unique-tail bursts; the rest remain LRU so new prefixes fit.
        let protected = inner.published.values().flatten()
            .filter(|snap| snap.users == 0 && snap.referenced)
            .max_by_key(|snap| snap.last_used)
            .map(|snap| snap.va);
        let Some((node, index)) = inner.published.iter()
                .flat_map(|(&node, snaps)| snaps.iter().enumerate().map(move |(i, snap)| (node, i, snap)))
                .filter(|(_, _, snap)| snap.users == 0)
                .min_by_key(|(_, _, snap)| (
                    Some(snap.va) == protected,
                    snap.last_used,
                ))
                .map(|(node, index, _)| (node, index))
        else { return false };
        // Active radix leases can exceed the soft budget. Keep their hot
        // snapshot until retirement, but allow OOM reclamation to remove it.
        if preserve_hot && inner.stats.cache_blocks > 0
            && Some(inner.published[&node][index].va) == protected
        {
            return false;
        }
        remove_snapshot(s, inner, node, index);
        return true;
    };
    if let Some(ids) = inner.node_blocks.remove(&key) {
        inner.stats.cache_blocks -= ids.len() as u64;
        inner.stats.cache_bytes -= ids.len() as u64 * s.block_bytes;
        for id in ids {
            deref_block(s, inner, id);
        }
    }
    if let Some(snapshots) = inner.published.remove(&Some(key)) {
        for snap in snapshots { free_snapshot(s, inner, snap); }
    }
    inner.stats.nodes_evicted += 1;
    true
}

fn remove_snapshot(s: &Shared, inner: &mut Inner, node: Option<(u32, u32)>, index: usize) {
    let snap = inner.published.get_mut(&node).unwrap().swap_remove(index);
    if inner.published[&node].is_empty() { inner.published.remove(&node); }
    free_snapshot(s, inner, snap);
}

fn deref_block(s: &Shared, inner: &mut Inner, id: u32) {
    if let Some(h) = unref_block(s, inner, id) {
        s.ops.release(h);
    }
}

/// Drop one reference; at zero, park the handle in the reuse pool (under the
/// cap) or hand it back. A returned handle is the CALLER's to release — the
/// attach path batches its releases outside the `inner` lock.
fn unref_block(s: &Shared, inner: &mut Inner, id: u32) -> Option<u64> {
    let b = &mut inner.blocks[id as usize];
    b.refs -= 1;
    if b.refs != 0 {
        return None;
    }
    let handle = b.handle;
    inner.stats.blocks_live -= 1;
    inner.free_ids.push(id);
    if inner.pooled.len() < s.pool_cap.load(Ordering::Relaxed) as usize {
        inner.pooled.push(handle);
        inner.stats.blocks_pooled += 1;
        None
    } else {
        Some(handle)
    }
}

/// Unmap and dereference every block mapped in `seq`'s windows.
fn release_window(s: &Shared, inner: &mut Inner, seq: usize) {
    for t in 0..inner.tracks.len() {
        for h in 0..s.geo.kvh_full {
            for k in 0..s.bph {
                let slot = slot_index(s, seq, h, k);
                if let Some(id) = inner.tracks[t].slots[slot].take() {
                    let va = slot_va(s, &inner.tracks[t], seq, h, k);
                    s.ops.unmap(va, s.block_bytes);
                    deref_block(s, inner, id);
                }
            }
        }
    }
    inner.seq_blocks[seq] = 0;
    s.frontier[seq].store(0, Ordering::Release);
}

/// `PLOW_SLAB_KEEP=1` keeps a dropped [`VmmSlab`]'s PHYSICAL chunks in the
/// backend pool ([`VmmOps::pool_put`]) instead of releasing them, so the next
/// load re-maps them (µs-class) instead of re-paying the driver's serial
/// ~13 GiB/s page commit — a same-box model reload or S1 switch drops from
/// ~1.9 s to ~0.4 s.
///
/// **Off by default outside multi-model serving:** pooled chunks hold VRAM
/// between loads, and only `serve::manager`'s planner credits the pool
/// ([`VmmOps::pool_bytes`] in its fit check, [`VmmOps::pool_trim`] before a
/// load). The manager flips the process default on when it manages more than
/// one model ([`set_slab_keep_default`]); everything else — single-model
/// serves, the lifecycle tests' return-to-baseline assert — keeps the
/// release-on-drop behavior. `PLOW_SLAB_KEEP=1`/`0` or `--rt-slab-keep`
/// force-overrides either way.
fn slab_keep_enabled() -> bool {
    crate::config::RuntimeConfig::get()
        .slab_keep_override()
        .unwrap_or_else(|| SLAB_KEEP_DEFAULT.load(Ordering::Relaxed))
}

static SLAB_KEEP_DEFAULT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set the process-wide default for keeping dropped slabs' physical chunks
/// (see [`slab_keep_enabled`]). Called by the serve manager when its planner
/// is pool-aware; `PLOW_SLAB_KEEP` still overrides.
pub fn set_slab_keep_default(on: bool) {
    SLAB_KEEP_DEFAULT.store(on, Ordering::Relaxed);
}

/// `--kv-pool-mib` / `PLOW_KV_POOL_MIB` — cap (MiB) for the KV physical-block
/// reuse pool the engines hand to [`VmmKv::enable_block_pool`]. Default 512;
/// 0 disables pooling entirely (every zero-ref block released, pre-creator
/// not spawned).
pub fn kv_pool_cap() -> u64 {
    crate::config::RuntimeConfig::get().kv_pool_mib() << 20
}

/// Physical-commit chunk for [`VmmSlab`]-backed weight slabs. The CUDA driver
/// commits at ~13 GiB/s (`perf-data/coldstart-plow-vs-vllm-gh200.md` §4b) so
/// this is ~20 ms per chunk — small enough that the first carve's wait is
/// invisible, large enough that a 60 GiB slab is a few hundred driver calls,
/// not tens of thousands of granules.
pub const WEIGHT_SLAB_CHUNK: u64 = 256 << 20;

/// Where a loader's non-VMM-prefix tensors' storage comes from. Shared by the
/// CUDA and AMD engines (see each engine's `_weight_slab` field for its
/// measurements and default): the fallback chain is Vmm → Flat → PerTensor,
/// each arm strictly less demanding on the driver than the one before.
pub enum WeightSlab {
    /// One VA reservation (µs) whose pages a background mapper commits
    /// front-to-back, overlapped with the upload — the upfront page-commit
    /// stall is off the critical path entirely.
    Vmm(VmmSlab),
    /// One flat device allocation, paying the driver's full upfront commit.
    Flat(crate::device::DeviceMem),
    /// Per-tensor allocation (`PLOW_WEIGHT_SLAB=0`, or the flat allocation
    /// was refused — a fragmented card can satisfy many small blocks and not
    /// one big one).
    PerTensor,
}

/// A lazily-committed linear device slab: one VA reservation (µs — probe [6])
/// whose physical backing a background thread creates and maps front-to-back,
/// so a consumer writing the slab in address order overlaps the driver's page
/// commit instead of paying it as one upfront `cuMemAlloc` stall.
///
/// Built for the weight loader: on GH200 the driver commits at ~13 GiB/s
/// (`perf-data/coldstart-plow-vs-vllm-gh200.md` §4b) while the upload feeds
/// the slab at ~6 GiB/s, so the mapper stays ahead of the writes after the
/// first chunk and [`Self::wait_mapped`] almost never blocks.
///
/// The consumer contract is [`Self::wait_mapped`] before touching any range —
/// VMM has no demand paging; an access below the watermark on an unmapped
/// page is fatal, not slow. Teardown on Drop: join the mapper, unmap and
/// release every chunk, free the reservation.
pub struct VmmSlab {
    ops: Arc<dyn VmmOps>,
    va: u64,
    /// Caller-visible slab size (what may be carved).
    bytes: u64,
    /// VA reservation span (`bytes` rounded up to the granularity).
    reserved: u64,
    shared: Arc<SlabShared>,
    join: Option<std::thread::JoinHandle<()>>,
}

struct SlabShared {
    m: Mutex<SlabState>,
    cv: parking_lot::Condvar,
}

#[derive(Default)]
struct SlabState {
    /// Bytes from the base that are mapped AND access-granted.
    mapped: u64,
    /// Physical handles in map order (`(handle, bytes)`), for teardown.
    handles: Vec<(u64, u64)>,
    /// First commit error; the mapper stops on it and waiters return it.
    err: Option<String>,
    /// Drop asked the mapper to quit early.
    stop: bool,
}

impl VmmSlab {
    /// Reserve `bytes` of VA and start committing physical chunks of
    /// `chunk_hint` (rounded to the granularity) from the base upward.
    pub fn new(ops: Arc<dyn VmmOps>, bytes: u64, chunk_hint: u64) -> Result<Self> {
        let gran = ops.granularity()?;
        let reserved = bytes.div_ceil(gran) * gran;
        let chunk = chunk_hint.max(gran).div_ceil(gran) * gran;
        let va = ops.reserve(reserved)?;
        let shared = Arc::new(SlabShared {
            m: Mutex::new(SlabState::default()),
            cv: parking_lot::Condvar::new(),
        });
        // Pooled physical chunks from a previous slab's PLOW_SLAB_KEEP drop:
        // re-mapping one skips the driver's serial page commit entirely, so a
        // reload/S1 switch pays map+set_access (µs-class) instead of
        // ~13 GiB/s of cuMemCreate. Drained unconditionally — a stale pool
        // (flag flipped off between loads) is still valid physical memory and
        // reusing it is strictly cheaper than re-creating.
        let mut pooled = ops.pool_take();
        let (t_ops, t_shared) = (Arc::clone(&ops), Arc::clone(&shared));
        let join = std::thread::Builder::new()
            .name("wslab-map".into())
            .spawn(move || {
                let mut off = 0u64;
                while off < reserved {
                    if t_shared.m.lock().stop {
                        break;
                    }
                    let n = chunk.min(reserved - off);
                    // Exact-size match only: every chunk but the tail is the
                    // uniform `chunk` size, so cross-model reuse works and at
                    // most one tail per generation misses.
                    let reuse = pooled
                        .iter()
                        .position(|&(_, b)| b == n)
                        .map(|i| pooled.swap_remove(i).0);
                    let r = match reuse {
                        Some(h) => Ok(h),
                        None => t_ops.create(n),
                    }
                    .and_then(|h| {
                        t_ops
                            .map(va + off, n, h)
                            .and_then(|()| t_ops.set_access(va + off, n))
                            .inspect_err(|_| {
                                // Map/access failed: the chunk is not (fully)
                                // usable — unwind it so Drop's walk only sees
                                // consistent (mapped, released-once) chunks.
                                t_ops.unmap(va + off, n);
                                t_ops.release(h);
                            })
                            .map(|()| h)
                    });
                    let mut st = t_shared.m.lock();
                    match r {
                        Ok(h) => {
                            st.handles.push((h, n));
                            off += n;
                            st.mapped = off;
                        }
                        Err(e) => {
                            st.err = Some(e.to_string());
                            t_shared.cv.notify_all();
                            break;
                        }
                    }
                    t_shared.cv.notify_all();
                }
                // Whatever the pool held beyond this slab's needs goes back
                // (or is released, per the backend's default) — never leaked.
                if !pooled.is_empty() {
                    t_ops.pool_put(pooled);
                }
            })
            .map_err(|e| RuntimeError::Device(format!("vmm slab mapper thread: {e}")))?;
        Ok(VmmSlab {
            ops,
            va,
            bytes,
            reserved,
            shared,
            join: Some(join),
        })
    }

    /// Base device address of the slab.
    pub fn base(&self) -> u64 {
        self.va
    }

    /// Caller-visible slab length in bytes.
    pub fn len(&self) -> u64 {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    /// Block until `[base, base+upto)` is mapped and device-accessible.
    /// Propagates the mapper's commit error (OOM mid-slab is fatal to the
    /// load — views were already carved, there is nothing to fall back to).
    pub fn wait_mapped(&self, upto: u64) -> Result<()> {
        debug_assert!(upto <= self.bytes, "wait past the slab end");
        let mut st = self.shared.m.lock();
        loop {
            if st.mapped >= upto {
                return Ok(());
            }
            if let Some(e) = &st.err {
                return Err(RuntimeError::Device(format!("vmm slab commit: {e}")));
            }
            self.shared.cv.wait(&mut st);
        }
    }
}

impl Drop for VmmSlab {
    fn drop(&mut self) {
        self.shared.m.lock().stop = true;
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
        let mut st = self.shared.m.lock();
        let mut off = 0u64;
        for &(_, n) in &st.handles {
            self.ops.unmap(self.va + off, n);
            off += n;
        }
        let handles = std::mem::take(&mut st.handles);
        if slab_keep_enabled() {
            // PLOW_SLAB_KEEP=1: hand the physical chunks to the backend pool
            // so the next load re-maps instead of re-committing. This holds
            // VRAM between loads BY DESIGN — the planner does not credit it,
            // so it is opt-in (see `slab_keep_enabled`).
            self.ops.pool_put(handles);
        } else {
            for (h, _) in handles {
                self.ops.release(h);
            }
        }
        self.ops.address_free(self.va, self.reserved);
    }
}

/// Chained per-block hashes of the prompt at sharing-block granularity
/// (`floor(len / block_rows)` whole blocks; the tail never matches).
pub fn hash_blocks(prompt: &[u32], block_rows: u32) -> Vec<BlockHash> {
    use std::hash::Hasher;
    let bt = block_rows as usize;
    let mut out = Vec::with_capacity(prompt.len() / bt);
    let mut prev: u64 = 0xcbf2_9ce4_8422_2325;
    for chunk in prompt.chunks_exact(bt) {
        let mut h = rustc_hash::FxHasher::default();
        h.write_u64(prev);
        for &t in chunk {
            h.write_u32(t);
        }
        prev = h.finish();
        out.push(prev);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, AtomicU64};

    fn live_blob() -> crate::asset::devblob::DevBlob {
        use crate::asset::devblob::{DevBlob, DevProg, DevTensor};
        use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
        let program = |rows, prefill| {
            let mut insts = Vec::new();
            for id in [1, 2] {
                let mut writer = DevInst64 {
                    op: DevOp::HeadNormRope as u16,
                    t: [TENSOR_NONE16; 8],
                    ..Default::default()
                };
                writer.t[0] = id;
                writer.t[5] = 0;
                writer.i = [rows, 1, 256, 0, 0, 0, if prefill { 0 } else { rows }, 0];
                writer.fj = [0, 1024, u32::MAX];
                insts.push(writer);
            }
            let mut reader = DevInst64 {
                op: if prefill {
                    DevOp::FlashPrefill
                } else {
                    DevOp::FlashDecode
                } as u16,
                t: [TENSOR_NONE16; 8],
                ..Default::default()
            };
            reader.t[3] = 1;
            reader.t[4] = 2;
            if !prefill {
                reader.t[5] = 3;
            }
            reader.i = if prefill {
                [rows, rows, 8, 1, 0, 0, 256, 1]
            } else {
                [rows, 8, 1, 1024, 0, 33, 256, u32::MAX]
            };
            if prefill {
                reader.fj = [0, 1024, u32::MAX];
            }
            insts.push(reader);
            DevProg {
                t: rows,
                packed_prefill_only: false,
                token_batch_body: false,
                n_counter: 0,
                insts,
                stream: Vec::new(),
                stream_ofs: Vec::new(),
                stream_len: Vec::new(),
                waits: Vec::new(),
                succs: Vec::new(),
                gq_stream: Vec::new(),
                gq_seg_ofs: Vec::new(),
                l2_domains: 0,
            }
        };
        DevBlob {
            n_cu: 1,
            flags: 0,
            target: 0,
            tensors: vec![
                DevTensor {
                    name: "in.pos".into(),
                    bytes: 4096,
                    init: None,
                },
                DevTensor {
                    name: "cache.key".into(),
                    bytes: 4 * 1024 * 256 * 2,
                    init: None,
                },
                DevTensor {
                    name: "cache.value".into(),
                    bytes: 4 * 1024 * 256 * 2,
                    init: None,
                },
                DevTensor {
                    name: "in.kvlen".into(),
                    bytes: 16,
                    init: None,
                },
            ],
            init: Vec::new(),
            kvrow: Vec::new(),
            progs: vec![program(128, true), program(1, false), program(4, false)],
            sections: Vec::new(),
            gen: Vec::new(),
            tp: None,
        }
    }

    #[test]
    fn declared_geometry_matches_legacy_and_rejects_stale_accesses() {
        let mut blob = live_blob();
        let m = blob.with_packet_view(plow_asset::live_kv::emit).unwrap();
        let legacy = LiveKvLayout::from_blob(&blob).unwrap();
        let declared = LiveKvLayout::from_manifest(&blob, &m).unwrap();
        assert_eq!(legacy.full_tensors, declared.full_tensors);
        assert_eq!(legacy.cache_tensors, declared.cache_tensors);
        assert_eq!(legacy.geometry.batch, declared.geometry.batch);
        assert_eq!(legacy.geometry.max_ctx, declared.geometry.max_ctx);
        assert_eq!(legacy.geometry.kvh_full, declared.geometry.kvh_full);
        blob.progs[1].insts[0].fj[1] += 32;
        assert!(blob.with_packet_view(|p| m.validate(p)).is_err());
        let mut bad = m.clone();
        bad.programs[1] =
            blob.with_packet_view(|p| plow_asset::live_kv::program_digest(&p.programs[1]));
        assert!(blob.with_packet_view(|p| bad.validate(p)).is_err());
    }
    #[test]
    fn fp8_live_geometry_tracks_cache_bytes_and_scale_slots() {
        use crate::asset::devblob::DevTensor;
        use packet::dev::DevOp;
        let mut blob = live_blob();
        for h in [1, 2] {
            blob.tensors[h].bytes /= 2;
        }
        for name in ["cache.key.scale", "cache.value.scale"] {
            blob.tensors.push(DevTensor {
                name: name.into(),
                bytes: 4 * 1024 * 4,
                init: None,
            });
        }
        for p in &mut blob.progs {
            for d in &mut p.insts {
                match DevOp::from_u16(d.op).unwrap() {
                    DevOp::HeadNormRope => {
                        d.op = DevOp::HeadNormRopeFp8 as u16;
                        d.t[6] = if d.t[0] == 1 { 4 } else { 5 };
                    }
                    DevOp::FlashPrefill | DevOp::FlashDecode => {
                        d.op = if d.op == DevOp::FlashPrefill as u16 {
                            DevOp::FlashPrefillFp8 as u16
                        } else {
                            DevOp::FlashDecodeFp8 as u16
                        };
                        d.t[6..8].copy_from_slice(&[4, 5]);
                    }
                    _ => unreachable!(),
                }
            }
        }
        let manifest = blob.with_packet_view(plow_asset::live_kv::emit).unwrap();
        assert_eq!(manifest.version, 2);
        assert_eq!(manifest.caches[0].scales, Some([4, 5]));
        for operand in [6, 7] {
            let saved = blob.progs[0].insts[2].t[operand];
            blob.progs[0].insts[2].t[operand] = if operand == 6 { 5 } else { 4 };
            assert!(blob.with_packet_view(plow_asset::live_kv::emit).is_err());
            blob.progs[0].insts[2].t[operand] = saved;
        }
        let layout = LiveKvLayout::from_manifest(&blob, &manifest).unwrap();
        assert_eq!(layout.geometry.elem, 1);
        assert_eq!(layout.geometry.full_tensor_bytes(), blob.tensors[1].bytes);
        assert_eq!(layout.full_tensors, [[1, 2]]);
        assert_eq!(layout.cache_tensors, [1, 2, 4, 5]);
        assert_eq!(
            layout
                .ring_tensors
                .iter()
                .map(|t| (t.tensor, t.slot_bytes))
                .collect::<Vec<_>>(),
            [(4, 1024 * 4), (5, 1024 * 4)]
        );
        let ops = Arc::new(MockVmm::default());
        let mut scales = VmmRings::new(ops.clone(), &layout.ring_tensors, 4).unwrap();
        scales.ensure_slot(3).unwrap();
        assert_eq!(scales.stats().resident_bytes, 2 * 1024 * 4);
        scales.ensure_prefix(4).unwrap();
        assert_eq!(scales.stats().resident_bytes, 4 * 2 * 1024 * 4);
        scales.ensure_slot(3).unwrap();
        assert_eq!(ops.creates.load(Ordering::SeqCst), 8);
        drop(scales);
        assert_eq!(ops.releases.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn mixed_kv_layout_keeps_each_cache_and_scale_slot_extent() {
        use crate::asset::devblob::DevTensor;
        use packet::dev::{DevOp, TENSOR_NONE16};
        for full_fp8 in [false, true] {
            for slide_fp8 in [false, true] {
                let mut blob = live_blob();
                for (index, fp8) in [full_fp8, slide_fp8].into_iter().enumerate() {
                    let bytes = 4 * 1024 * 256 * if fp8 { 1 } else { 2 };
                    let pair = if index == 0 {
                        blob.tensors[1].bytes = bytes;
                        blob.tensors[2].bytes = bytes;
                        [1, 2]
                    } else {
                        let h = blob.tensors.len() as u16;
                        for name in ["slide.key", "slide.value"] {
                            blob.tensors.push(DevTensor {
                                name: name.into(),
                                bytes,
                                init: None,
                            });
                        }
                        [h, h + 1]
                    };
                    let scales = if fp8 {
                        let h = blob.tensors.len() as u16;
                        for which in ["key", "value"] {
                            blob.tensors.push(DevTensor {
                                name: format!("scale.{index}.{which}"),
                                bytes: 4 * 1024 * 4,
                                init: None,
                            });
                        }
                        [h, h + 1]
                    } else {
                        [TENSOR_NONE16; 2]
                    };
                    let window = if index == 0 { 0 } else { 512 };
                    let mask = if index == 0 { u32::MAX } else { 1023 };
                    for (pi, p) in blob.progs.iter_mut().enumerate() {
                        let mut ops = p.insts[..3].to_vec();
                        for k in 0..2 {
                            ops[k].op = if fp8 {
                                DevOp::HeadNormRopeFp8
                            } else {
                                DevOp::HeadNormRope
                            } as u16;
                            ops[k].t[0] = pair[k];
                            ops[k].t[6] = scales[k];
                            ops[k].fj[2] = mask;
                        }
                        let d = &mut ops[2];
                        d.op = match (pi == 0, fp8) {
                            (true, true) => DevOp::FlashPrefillFp8,
                            (true, false) => DevOp::FlashPrefill,
                            (false, true) => DevOp::FlashDecodeFp8,
                            (false, false) => DevOp::FlashDecode,
                        } as u16;
                        d.t[3..5].copy_from_slice(&pair);
                        d.t[6..8].copy_from_slice(&scales);
                        if pi == 0 {
                            d.i[5] = window;
                            d.fj[2] = mask;
                        } else {
                            d.i[4] = window;
                            d.i[7] = mask;
                        }
                        if index == 0 {
                            p.insts = ops;
                        } else {
                            p.insts.extend(ops);
                        }
                    }
                }
                let manifest = blob.with_packet_view(plow_asset::live_kv::emit).unwrap();
                let layout = LiveKvLayout::from_manifest(&blob, &manifest).unwrap();
                assert_eq!(layout.geometry.elem, if full_fp8 { 1 } else { 2 });
                assert_eq!(layout.geometry.full_tensor_bytes(), blob.tensors[1].bytes);
                for c in &manifest.caches {
                    for h in c.pair.into_iter().chain(c.scales.into_iter().flatten()) {
                        assert!(layout.cache_tensors.contains(&(h as usize)));
                        if c.window != 0 || c.scales.is_some_and(|pair| pair.contains(&h)) {
                            let region = layout
                                .ring_tensors
                                .iter()
                                .find(|t| t.tensor == h as usize)
                                .unwrap();
                            assert_eq!(region.slot_bytes * 4, blob.tensors[h as usize].bytes);
                        }
                    }
                }
                for p in &mut blob.progs {
                    for d in &mut p.insts[3..5] {
                        d.fj[2] = u32::MAX;
                    }
                    let d = &mut p.insts[5];
                    if p.t == 128 {
                        d.i[5] = 0;
                        d.fj[2] = u32::MAX;
                    } else {
                        d.i[4] = 0;
                        d.i[7] = u32::MAX;
                    }
                }
                let manifest = blob.with_packet_view(plow_asset::live_kv::emit).unwrap();
                assert_eq!(
                    LiveKvLayout::from_manifest(&blob, &manifest).is_ok(),
                    full_fp8 == slide_fp8
                );
            }
        }
    }

    #[test]
    fn sliding_only_manifest_is_valid_but_allocator_limitation_is_explicit() {
        let mut blob = live_blob();
        for p in &mut blob.progs {
            for d in &mut p.insts {
                if d.op == packet::dev::DevOp::HeadNormRope as u16 {
                    d.fj[2] = 1023;
                }
                if d.op == packet::dev::DevOp::FlashPrefill as u16 {
                    d.i[5] = 512;
                    d.fj[2] = 1023;
                }
                if d.op == packet::dev::DevOp::FlashDecode as u16 {
                    d.i[4] = 512;
                    d.i[7] = 1023;
                }
            }
        }
        let manifest = blob.with_packet_view(plow_asset::live_kv::emit).unwrap();
        assert!(blob.with_packet_view(|p| manifest.validate(p)).is_ok());
        let error = LiveKvLayout::from_manifest(&blob, &manifest)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("allocator does not support sliding-only"));
    }
    #[test]
    fn malformed_present_manifest_cannot_fall_back() {
        let mut blob = live_blob();
        assert!(LiveKvLayout::manifest(&blob, &[]).unwrap().is_none());
        blob.sections.push(crate::asset::devblob::DevSection {
            kind: packet::devbuild::SECT_METADATA,
            name: plow_asset::live_kv::SECTION.into(),
            offset: 0,
            size: 2,
        });
        assert!(LiveKvLayout::manifest(&blob, b"{}").is_err());
    }

    #[test]
    fn live_geometry_comes_from_packet_handles_and_all_rungs() {
        let layout = LiveKvLayout::from_blob(&live_blob()).unwrap();
        assert_eq!(layout.full_tensors, [[1, 2]]);
        assert_eq!(layout.cache_tensors, [1, 2]);
        assert_eq!(layout.geometry.batch, 4);
        assert_eq!(layout.geometry.max_ctx, 1024);
        assert_eq!(layout.geometry.kvh_full, 1);
        assert_eq!(layout.geometry.hd_full, 256);
        assert_eq!(layout.geometry.full_tensor_bytes(), 4 * 1024 * 256 * 2);
    }

    #[test]
    fn live_geometry_rejects_conflicting_or_unbounded_access() {
        use packet::dev::DevOp;
        let mutations: &[fn(&mut crate::asset::devblob::DevBlob)] = &[
            |b| b.tensors[1].bytes /= 2,
            |b| b.tensors[3].bytes = 4,
            |b| b.progs[1].insts[2].t[5] = 0,
            |b| b.progs[1].insts[2].i[1] = 0,
            |b| b.tensors[2].init = Some(0..1),
            |b| b.progs[0].insts[2].fj[1] = 512,
            |b| b.progs[1].insts[2].i[6] = 512,
            |b| b.progs[1].insts[0].i[6] = 0,
            |b| b.progs[2].insts[0].fj[2] = 1023,
            |b| b.progs[0].insts[0].i[6] = 128,
            |b| b.progs[0].insts[0].t[7] = 0,
            |b| b.progs[2].insts[2].i[7] = 0,
            |b| b.progs[2].insts[0].t[1] = 2,
            |b| {
                b.progs[1].insts.pop();
            },
            |b| b.progs[0].packed_prefill_only = true,
            |b| b.progs[2].insts[2].op = DevOp::FlashDecodeFp8 as u16,
        ];
        for (i, mutate) in mutations.iter().enumerate() {
            let mut blob = live_blob();
            mutate(&mut blob);
            assert!(LiveKvLayout::from_blob(&blob).is_err(), "mutation {i}");
        }
    }

    #[test]
    fn live_geometry_validates_tensormaps_across_batch_sizes() {
        use crate::asset::devblob::DevTensor;
        for batch in [1, 4, 16] {
            let fixture = || {
                let mut blob = live_blob();
                if batch == 1 {
                    blob.progs.pop();
                }
                let decode = blob.progs.last_mut().unwrap();
                decode.t = batch;
                for writer in &mut decode.insts[..2] {
                    writer.i[0] = batch;
                    writer.i[6] = batch;
                }
                decode.insts[2].i[0] = batch;
                blob.tensors[3].bytes = u64::from(batch) * 4;
                for id in [1, 2] {
                    blob.tensors[id].bytes = u64::from(batch) * 1024 * 256 * 2;
                }
                blob.tensors.push(DevTensor {
                    name: "tmap.cache".into(),
                    bytes: 256,
                    init: None,
                });
                let mut map = packet::rope::GenTensor::tmap_kv_pair(1, 2, 1024, 256, 1);
                map.tensor = 4;
                blob.gen.push(map);
                blob.progs[0].insts[2].t[7] = 4;
                blob
            };
            let blob = fixture();
            let layout = LiveKvLayout::from_blob(&blob).unwrap();
            assert_eq!(layout.geometry.batch, batch);
            assert_eq!(layout.full_tensors, [[1, 2]]);
            let mutations: &[fn(&mut crate::asset::devblob::DevBlob)] = &[
                |b| b.gen[0].ctx = 512,
                |b| b.gen[0].hd = 128,
                |b| b.gen[0].frac = 2.0,
                |b| b.gen[0].scale = 1,
                |b| b.tensors[4].bytes = 128,
                |b| b.tensors[1].bytes /= 2,
                |b| b.progs[0].insts[2].t[7] = 3,
                |b| {
                    b.progs.last_mut().unwrap().insts[2].op =
                        packet::dev::DevOp::FlashDecodeFp8 as u16
                },
            ];
            for (case, mutate) in mutations.iter().enumerate() {
                let mut invalid = fixture();
                mutate(&mut invalid);
                assert!(
                    LiveKvLayout::from_blob(&invalid).is_err(),
                    "batch={batch} case={case}"
                );
            }
        }
    }

    #[test]
    fn live_allocator_never_hashes_attaches_or_publishes_and_releases_on_reset() {
        let ops = Arc::new(MockVmm::default());
        let geo = uniform_pool(ops.clone()).geometry().clone();
        let p = VmmKv::new_live(ops.clone(), geo, 64).unwrap();
        assert!(!p.prefix_reuse());
        for _ in 0..2 {
            p.ensure_rows(0, 17).unwrap();
            p.ensure_rows(1, 9).unwrap();
            let other_rows = p.mapped_rows(1);
            assert!(p.try_attach(1, &prompt(17)).unwrap().is_none());
            assert!(p
                .publish(0, &prompt(17), 4, |_| panic!("snapshot must not run"))
                .is_err());
            let inner = p.shared.inner.lock();
            assert!(inner.published.is_empty());
            assert!(inner.node_blocks.is_empty());
            assert!(inner
                .seqs
                .iter()
                .all(|s| s.hashes.is_empty() && s.tokens.is_empty()));
            drop(inner);
            p.begin_seq(0);
            assert_eq!(p.mapped_rows(0), 0);
            assert_eq!(p.mapped_rows(1), other_rows);
            assert!(p.stats().blocks_live > 0);
            p.begin_seq(1);
            assert_eq!(p.stats().blocks_live, 0);
        }
        assert_eq!(p.stats().attach_hits, 0);
        assert_eq!(p.stats().attach_misses, 0);
        assert_eq!(ops.allocs.load(Ordering::SeqCst), 0);
    }

    /// Records every driver call; `fail_creates` makes the next N creates
    /// fail (the OOM path).
    #[derive(Default)]
    struct MockVmm {
        next: AtomicU64,
        granularity: AtomicU64,
        reserves: AtomicU64,
        reserved_bytes: AtomicU64,
        address_frees: AtomicU64,
        creates: AtomicU64,
        releases: AtomicU64,
        maps: AtomicU64,
        unmaps: AtomicU64,
        allocs: AtomicU64,
        frees: AtomicU64,
        fail_creates: AtomicI64,
        fail_reserves: AtomicI64,
        fail_maps: AtomicI64,
        fail_access: AtomicI64,
        pool: std::sync::Mutex<Vec<(u64, u64)>>,
    }

    impl VmmOps for MockVmm {
        fn granularity(&self) -> Result<u64> {
            Ok(self.granularity.load(Ordering::SeqCst).max(16))
        }
        fn reserve(&self, bytes: u64) -> Result<u64> {
            if self.fail_reserves.load(Ordering::SeqCst) > 0
                && self.fail_reserves.fetch_sub(1, Ordering::SeqCst) == 1
            {
                return Err(RuntimeError::Oom("mock VA reservation failure".into()));
            }
            self.reserves.fetch_add(1, Ordering::SeqCst);
            self.reserved_bytes.fetch_add(bytes, Ordering::SeqCst);
            Ok(self.next.fetch_add(1 << 32, Ordering::SeqCst) + (1 << 32))
        }
        fn address_free(&self, _va: u64, _bytes: u64) {
            self.address_frees.fetch_add(1, Ordering::SeqCst);
        }
        fn create(&self, _bytes: u64) -> Result<u64> {
            if self.fail_creates.fetch_sub(1, Ordering::SeqCst) > 0 {
                return Err(RuntimeError::Oom("mock OOM".into()));
            }
            self.fail_creates.fetch_add(1, Ordering::SeqCst); // clamp at <=0
            self.creates.fetch_add(1, Ordering::SeqCst);
            Ok(self.next.fetch_add(1, Ordering::SeqCst))
        }
        fn release(&self, _handle: u64) {
            self.releases.fetch_add(1, Ordering::SeqCst);
        }
        fn map(&self, _va: u64, _bytes: u64, _handle: u64) -> Result<()> {
            // `fail_maps = k` makes the k-th upcoming map call fail, one-shot
            // (earlier calls succeed) — exercises PARTIALLY-mapped unwinds.
            if self.fail_maps.load(Ordering::SeqCst) > 0
                && self.fail_maps.fetch_sub(1, Ordering::SeqCst) == 1
            {
                return Err(RuntimeError::Device("mock map failure".into()));
            }
            self.maps.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn unmap(&self, _va: u64, _bytes: u64) {
            self.unmaps.fetch_add(1, Ordering::SeqCst);
        }
        fn set_access(&self, _va: u64, _bytes: u64) -> Result<()> {
            if self.fail_access.load(Ordering::SeqCst) > 0
                && self.fail_access.fetch_sub(1, Ordering::SeqCst) == 1
            {
                return Err(RuntimeError::Device("mock set-access failure".into()));
            }
            Ok(())
        }
        fn alloc(&self, _bytes: u64) -> Result<u64> {
            self.allocs.fetch_add(1, Ordering::SeqCst);
            Ok(self.next.fetch_add(1, Ordering::SeqCst))
        }
        fn free(&self, _va: u64) {
            self.frees.fetch_add(1, Ordering::SeqCst);
        }
        fn copy_dtod(&self, _dst: u64, _src: u64, _bytes: u64) -> Result<()> {
            Ok(())
        }
        fn pool_take(&self) -> Vec<(u64, u64)> {
            std::mem::take(&mut *self.pool.lock().unwrap())
        }
        fn pool_put(&self, mut chunks: Vec<(u64, u64)>) {
            self.pool.lock().unwrap().append(&mut chunks);
        }
    }

    /// 1 full layer, 1 kv head, hd 4 bf16 (8 B rows), 32-row context,
    /// 64 B blocks → block_rows = 8, 4 blocks/head, 2 tracks (K, V).
    fn pool(ops: Arc<MockVmm>) -> VmmKv {
        pool_with_cap(ops, 0)
    }

    fn pool_with_cap(ops: Arc<MockVmm>, cache_cap: u64) -> VmmKv {
        let geo = VmmGeometry {
            full_layers: vec![3],
            kvh_full: 1,
            hd_full: 4,
            slide_layers: vec![0, 1, 2],
            kvh_slide: 2,
            hd_slide: 4,
            window: 4,
            elem: 2,
            elem_slide: 2,
            max_ctx: 32,
            batch: 2,
        };
        VmmKv::new(ops, geo, 64, cache_cap).expect("pool")
    }

    #[test]
    fn gemma_128k_live_kv_reserves_logical_windows_and_commits_on_demand() {
        const GRAN: u64 = 2 << 20;
        let ops = Arc::new(MockVmm::default());
        ops.granularity.store(GRAN, Ordering::SeqCst);
        let geometry = VmmGeometry {
            full_layers: (0..8).collect(),
            kvh_full: 1,
            hd_full: 512,
            slide_layers: (0..40).collect(),
            kvh_slide: 8,
            hd_slide: 256,
            window: 1024,
            elem: 2,
            elem_slide: 2,
            max_ctx: 131_072,
            batch: 16,
        };
        let full = VmmKv::new_live(ops.clone(), geometry, GRAN).unwrap();
        assert_eq!(ops.reserves.load(Ordering::SeqCst), 16);
        assert_eq!(ops.reserved_bytes.load(Ordering::SeqCst), 32 << 30);
        assert_eq!(ops.creates.load(Ordering::SeqCst), 0);
        assert_eq!(full.stats().blocks_live, 0);

        full.ensure_rows(0, 1).unwrap();
        assert_eq!(full.stats().blocks_live * GRAN, 32 << 20);
        full.ensure_rows(0, 4096).unwrap();
        assert_eq!(full.stats().blocks_live * GRAN, 64 << 20);

        let ring_slot_bytes = 8 * 2048 * 256 * 2;
        let tensors: Vec<_> = (0..80)
            .map(|tensor| LiveRingTensor {
                tensor,
                slot_bytes: ring_slot_bytes,
            })
            .collect();
        let ring_ops = Arc::new(MockVmm::default());
        ring_ops.granularity.store(GRAN, Ordering::SeqCst);
        let mut rings = VmmRings::new(ring_ops.clone(), &tensors, 16).unwrap();
        assert_eq!(rings.stats().reserved_bytes, 10 << 30);
        assert_eq!(ring_ops.reserved_bytes.load(Ordering::SeqCst), 10 << 30);
        assert_eq!(rings.stats().resident_bytes, 0);
        assert_eq!(ring_ops.creates.load(Ordering::SeqCst), 0);
        rings.ensure_slot(0).unwrap();
        assert_eq!(rings.stats().resident_bytes, 640 << 20);
        assert_eq!(
            ring_ops.creates.load(Ordering::SeqCst) * ring_slot_bytes,
            640 << 20
        );
        rings.release_slot(0);
        assert_eq!(rings.stats().resident_bytes, 0);
        assert_eq!(ring_ops.releases.load(Ordering::SeqCst), 80);
        rings.ensure_slot(0).unwrap();
        assert_eq!(rings.stats().resident_bytes, 640 << 20);
        assert_eq!(ring_ops.creates.load(Ordering::SeqCst), 160);
    }

    fn prompt(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| i * 7 + 3).collect()
    }

    #[test]
    fn cache_track_reservation_failure_releases_prior_windows() {
        let geometry = pool(Arc::new(MockVmm::default())).geometry().clone();
        for failure in 1..=3 {
            let ops = Arc::new(MockVmm::default());
            ops.fail_reserves.store(failure, Ordering::SeqCst);
            assert!(VmmKv::new_tensors(ops.clone(), geometry.clone(), 64, 0,
                &[(3, 0), (3, 1), (3, 2)]).is_err());
            assert_eq!(ops.reserves.load(Ordering::SeqCst), (failure - 1) as u64);
            assert_eq!(ops.address_frees.load(Ordering::SeqCst), (failure - 1) as u64);
            assert_eq!(ops.creates.load(Ordering::SeqCst), 0);
        }
    }

    #[test]
    fn unpaired_cache_tracks_share_prefix_and_allocate_private_suffix() {
        let ops = Arc::new(MockVmm::default());
        let geometry = VmmGeometry {
            full_layers: vec![0, 1, 2],
            kvh_full: 1,
            hd_full: 4,
            slide_layers: vec![],
            kvh_slide: 0,
            hd_slide: 0,
            window: 0,
            elem: 2,
            elem_slide: 2,
            max_ctx: 32,
            batch: 2,
        };
        for tensors in [&[][..], &[(0, 0), (0, 0)][..], &[(3, 0)][..]] {
            assert!(VmmKv::new_tensors(ops.clone(), geometry.clone(), 64, 0, tensors).is_err());
        }
        let p = VmmKv::new_tensors(
            ops.clone(), geometry, 64, 0, &[(0, 0), (1, 0), (2, 0)],
        ).unwrap();
        for layer in 0..3 {
            assert!(p.tensor_va(layer, 0).is_some());
            assert!(p.tensor_va(layer, 1).is_none());
        }
        let tokens = prompt(25);
        assert!(p.try_attach(0, &tokens).unwrap().is_none());
        p.ensure_rows(0, 24).unwrap();
        p.publish_at(0, &tokens, 16, 16, |_| Ok(())).unwrap();
        assert_eq!(ops.creates.load(Ordering::SeqCst), 9);
        assert_eq!(p.try_attach(1, &tokens).unwrap().unwrap().rows, 16);
        assert_eq!(p.stats().blocks_shared_mapped, 6);
        assert_eq!(ops.creates.load(Ordering::SeqCst), 9);
        p.finish_attach(1);
        p.ensure_rows(1, 17).unwrap();
        assert_eq!(ops.creates.load(Ordering::SeqCst), 12);
        p.begin_seq(0);
        assert_eq!(p.mapped_rows(1), 24);
        p.begin_seq(1);
        assert_eq!(p.stats().blocks_live, 6);
        drop(p);
        assert_eq!(ops.creates.load(Ordering::SeqCst), ops.releases.load(Ordering::SeqCst));
        assert_eq!(ops.maps.load(Ordering::SeqCst), ops.unmaps.load(Ordering::SeqCst));
        assert_eq!(ops.allocs.load(Ordering::SeqCst), ops.frees.load(Ordering::SeqCst));
    }

    /// Uniform full-attention geometry (Qwen-family): no rings, window 0.
    fn uniform_pool(ops: Arc<MockVmm>) -> VmmKv {
        let geo = VmmGeometry {
            full_layers: vec![0, 1],
            kvh_full: 1,
            hd_full: 4,
            slide_layers: vec![],
            kvh_slide: 1,
            hd_slide: 4,
            window: 0,
            elem: 2,
            elem_slide: 2,
            max_ctx: 32,
            batch: 2,
        };
        VmmKv::new(ops, geo, 64, 0).expect("uniform pool")
    }

    #[test]
    fn uniform_full_attention_attaches_without_snapshots() {
        // Qwen-family: every layer full attention. Publish gates on
        // window.max(block) = block only; the snapshot is a 4-byte stub.
        let ops = Arc::new(MockVmm::default());
        let p = uniform_pool(ops.clone());
        let pr = prompt(17);
        assert!(p.try_attach(0, &pr).unwrap().is_none());
        p.ensure_rows(0, 17).unwrap();
        p.publish(0, &pr, 4, |_va| Ok(())).unwrap();
        p.ensure_rows(1, 1).unwrap();
        let a = p.try_attach(1, &pr).unwrap().expect("published boundary");
        assert_eq!(a.rows, 16);
        assert_eq!(a.snap_bytes, 4);
    }

    /// Attach-then-abort cycles (client disconnects mid-prefill) must return
    /// every block: live count settles back to the cache-held baseline no
    /// matter how many times the storm spins.
    #[test]
    fn cancel_storm_leaks_nothing() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        let pr = prompt(17);
        assert!(p.try_attach(0, &pr).unwrap().is_none());
        p.ensure_rows(0, 17).unwrap();
        p.publish(0, &pr, 128, |_va| Ok(())).unwrap();
        p.begin_seq(0);
        let baseline = p.stats().blocks_live;
        for i in 0..50usize {
            let seq = i % 2;
            p.ensure_rows(seq, 1).unwrap(); // begin_slot's row-0 pre-map
            p.try_attach(seq, &pr).unwrap().expect("cached prefix");
            p.begin_seq(seq); // abort before prefill completes
        }
        assert_eq!(
            p.stats().blocks_live,
            baseline,
            "attach/abort cycles leaked physical blocks"
        );
    }

    /// A driver map failure mid-attach (window 2 of 4 mappings in) must
    /// unwind the borrower to an EMPTY window — a half-mapped window that
    /// still claims `rows` would read unmapped VA on the next decode step —
    /// and must not poison the cache: the same attach succeeds on retry.
    #[test]
    fn attach_map_failure_unwinds_to_empty_window() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        let pr = prompt(17);
        assert!(p.try_attach(0, &pr).unwrap().is_none());
        p.ensure_rows(0, 17).unwrap();
        p.publish(0, &pr, 128, |_va| Ok(())).unwrap();

        p.ensure_rows(1, 1).unwrap();
        let live = p.stats().blocks_live;
        let unmaps_before = ops.unmaps.load(Ordering::SeqCst);
        ops.fail_maps.store(3, Ordering::SeqCst); // 2 maps land, the 3rd fails
        assert!(p.try_attach(1, &pr).is_err());
        assert_eq!(
            p.mapped_rows(1),
            0,
            "failed attach must leave an empty window"
        );
        assert_eq!(
            p.stats().blocks_live,
            live - 2,
            "only the displaced row-0 private blocks may go"
        );
        // Exact-range contract (device/cuda.rs): unmap ONLY what was mapped —
        // the 2 displaced row-0 blocks plus the 2 maps that landed, never the
        // 2 slots whose maps failed.
        assert_eq!(
            ops.unmaps.load(Ordering::SeqCst) - unmaps_before,
            4,
            "unwind must unmap exactly the displaced + successfully-mapped ranges"
        );

        p.ensure_rows(1, 1).unwrap();
        let a = p.try_attach(1, &pr).unwrap().expect("retry must hit");
        assert_eq!(a.rows, 16);
        assert_eq!(p.mapped_rows(1), 16);
    }

    /// Multi-turn: after decode extends a sequence past its prompt, a second
    /// publish with prompt+generated tokens must make the GENERATED blocks
    /// attachable — the follow-up turn embedding this turn's output pays
    /// only for its new input.
    #[test]
    fn tail_publish_extends_the_attachable_prefix() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        let pr = prompt(17); // prompt: 2 whole blocks
        assert!(p.try_attach(0, &pr).unwrap().is_none());
        p.ensure_rows(0, 17).unwrap();
        p.publish(0, &pr, 128, |_va| Ok(())).unwrap();

        // Decode extends the same slot to 24 rows (3 whole blocks); the tail
        // publish at sequence end covers prompt + generated.
        let full = prompt(24); // prompt(n) streams are prefix-consistent
        p.ensure_rows(0, 24).unwrap();
        p.publish(0, &full, 128, |_va| Ok(())).unwrap();
        p.begin_seq(0);

        // Turn 2: the next prompt embeds the whole first turn. All 3 blocks
        // attach — including the generated one.
        let mut pr2 = prompt(24);
        pr2.push(999);
        p.ensure_rows(1, 1).unwrap();
        let a = p
            .try_attach(1, &pr2)
            .unwrap()
            .expect("generated blocks attach");
        assert_eq!(a.rows, 24);
    }

    #[test]
    fn from_config_uniform_layers_fallback() {
        // No layer_types (Qwen/Llama-family config): every layer is full
        // attention, window 0. With layer_types (Gemma), sliding_window is
        // still required.
        let dir = std::env::temp_dir().join(format!("plow-vmm-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            br#"{"num_hidden_layers": 3, "num_key_value_heads": 8, "head_dim": 128}"#,
        )
        .unwrap();
        let g = VmmGeometry::from_config(&dir, 4096, 2).expect("uniform fallback");
        assert_eq!(g.full_layers, vec![0, 1, 2]);
        assert!(g.slide_layers.is_empty());
        assert_eq!(g.window, 0);
        assert_eq!((g.kvh_full, g.hd_full), (8, 128));

        // Gemma-style still needs sliding_window.
        std::fs::write(
            dir.join("config.json"),
            br#"{"layer_types": ["full_attention", "sliding_attention"],
                 "num_key_value_heads": 4, "head_dim": 256}"#,
        )
        .unwrap();
        assert!(
            VmmGeometry::from_config(&dir, 4096, 2).is_none(),
            "sliding layers without sliding_window must refuse"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn frontier_growth_maps_whole_blocks() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        p.ensure_rows(0, 20).unwrap(); // ceil(20/8) = 3 blocks × 2 tracks
        assert_eq!(p.mapped_rows(0), 24);
        assert_eq!(ops.creates.load(Ordering::SeqCst), 6);
        assert_eq!(p.stats().blocks_live, 6);
        // Idempotent below the frontier.
        p.ensure_rows(0, 10).unwrap();
        assert_eq!(ops.creates.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn stale_premap_hint_cannot_recreate_a_released_window() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        p.ensure_rows(0, 20).unwrap();
        let generation = p.shared.generation[0].load(Ordering::Acquire);
        p.begin_seq(0);
        assert_eq!(p.mapped_rows(0), 0);
        assert_eq!(p.stats().blocks_live, 0);
        ensure_rows(&p.shared, 0, 32, Some(generation)).unwrap();
        assert_eq!(p.mapped_rows(0), 0);
        assert_eq!(p.stats().blocks_live, 0);
        p.ensure_rows(0, 1).unwrap();
        ensure_rows(&p.shared, 0, 32, Some(generation)).unwrap();
        assert_eq!(p.mapped_rows(0), 8);
        let generation = p.shared.generation[0].load(Ordering::Acquire);
        ensure_rows(&p.shared, 0, 16, Some(generation)).unwrap();
        assert_eq!(p.mapped_rows(0), 16);
    }

    #[test]
    fn incomplete_mapping_column_can_be_retried_without_remapping_live_heads() {
        let ops = Arc::new(MockVmm::default());
        let p = uniform_pool(ops.clone());
        ops.fail_maps.store(2, Ordering::SeqCst);
        assert!(p.ensure_rows(0, 1).is_err());
        assert_eq!(p.mapped_rows(0), 0);
        assert_eq!(p.stats().blocks_live, 1);
        p.ensure_rows(0, 1).unwrap();
        assert_eq!(p.mapped_rows(0), 8);
        assert_eq!(p.stats().blocks_live, 4);
        assert_eq!(ops.maps.load(Ordering::SeqCst), 4);
        p.begin_seq(0);
        assert_eq!(p.stats().blocks_live, 0);
        assert_eq!(ops.maps.load(Ordering::SeqCst), ops.unmaps.load(Ordering::SeqCst));
    }

    #[test]
    fn ensure_rows_unwinds_map_and_access_failures() {
        for fail_access in [false, true] {
            let ops = Arc::new(MockVmm::default());
            let p = uniform_pool(ops.clone());
            if fail_access {
                ops.fail_access.store(1, Ordering::SeqCst);
            } else {
                ops.fail_maps.store(1, Ordering::SeqCst);
            }

            assert!(p.ensure_rows(0, 1).is_err());
            assert_eq!(p.mapped_rows(0), 0);
            assert_eq!(p.stats().blocks_live, 0);
            assert_eq!(
                ops.maps.load(Ordering::SeqCst),
                ops.unmaps.load(Ordering::SeqCst),
                "every successful mapping must be unwound"
            );
            assert_eq!(
                ops.creates.load(Ordering::SeqCst),
                ops.releases.load(Ordering::SeqCst),
                "every created block must be released"
            );
        }
    }

    #[test]
    fn attach_shares_blocks_without_new_creates() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        let pr = prompt(17); // 2 whole blocks + 1-token tail

        // Owner: no cache yet, prefills privately, publishes 2 blocks.
        assert!(p.try_attach(0, &pr).unwrap().is_none());
        p.ensure_rows(0, 17).unwrap(); // 3 blocks × 2 tracks = 6 creates
        p.publish(0, &pr, 128, |_va| Ok(())).unwrap();
        assert_eq!(ops.creates.load(Ordering::SeqCst), 6);
        assert_eq!(p.stats().cache_blocks, 4); // 2 blocks × 2 tracks

        // Borrower: identical prompt attaches the 2 published blocks —
        // zero new physical blocks for the shared span (the dedup). The
        // engine pre-maps row 0 at begin_slot (idle-row garbage writes);
        // attach must displace that private block, not double-map over it.
        p.ensure_rows(1, 1).unwrap();
        let created_before = ops.creates.load(Ordering::SeqCst);
        let a = p.try_attach(1, &pr).unwrap().expect("published boundary");
        assert_eq!(
            ops.releases.load(Ordering::SeqCst),
            2,
            "attach must release the displaced row-0 private blocks"
        );
        assert_eq!(a.rows, 16);
        assert_eq!(p.mapped_rows(1), 16);
        assert_eq!(ops.creates.load(Ordering::SeqCst), created_before);
        assert_eq!(p.stats().blocks_shared_mapped, 4);
        // The borrower's tail block is private again.
        p.ensure_rows(1, 17).unwrap();
        assert_eq!(ops.creates.load(Ordering::SeqCst), created_before + 2);
    }

    #[test]
    fn divergent_prompt_attaches_shared_prefix_only() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        let a = prompt(25); // 3 whole blocks
        p.try_attach(0, &a).unwrap();
        p.ensure_rows(0, 25).unwrap();
        p.publish(0, &a, 64, |_| Ok(())).unwrap();

        // Same first 2 blocks, divergent third → published boundary exists
        // only at 3 (the full publish), so LCP 2 has no snapshot → miss.
        let mut b = a.clone();
        b[20] ^= 1;
        assert!(p.try_attach(1, &b).unwrap().is_none());
        // Exact-prefix extension DOES attach at the published boundary.
        p.begin_seq(1);
        let mut c = a.clone();
        c.push(999);
        let at = p.try_attach(1, &c).unwrap().expect("boundary at 3 blocks");
        assert_eq!(at.rows, 24);
    }

    #[test]
    fn release_then_oom_evicts_cache_and_reuses_hbm() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        let pr = prompt(16); // exactly 2 blocks; attach limit keeps 1 shareable
        p.try_attach(0, &pr).unwrap();
        p.ensure_rows(0, 16).unwrap();
        p.publish(0, &pr, 32, |_| Ok(())).unwrap();
        // Owner sequence ends: windows unmapped, cache keeps the blocks.
        p.begin_seq(0);
        assert_eq!(
            ops.releases.load(Ordering::SeqCst),
            0,
            "cache must pin blocks"
        );
        assert_eq!(p.stats().blocks_live, 4);

        // Driver OOM on the next creates → LRU eviction frees cached blocks,
        // then the create succeeds.
        ops.fail_creates.store(2, Ordering::SeqCst);
        p.ensure_rows(1, 8).unwrap();
        assert!(
            ops.releases.load(Ordering::SeqCst) > 0,
            "eviction must free"
        );
        assert!(p.stats().nodes_evicted > 0);
        assert_eq!(p.mapped_rows(1), 8);
    }

    #[test]
    fn cache_budget_includes_boundary_snapshots() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops.clone(), 512);
        let a = prompt(17);
        p.try_attach(0, &a).unwrap();
        p.ensure_rows(0, 17).unwrap();
        p.publish(0, &a, 192, |_| Ok(())).unwrap();
        p.begin_seq(0);
        assert_eq!(p.stats().cache_bytes, 448);

        let mut b = a.clone();
        b[0] ^= 1;
        p.try_attach(1, &b).unwrap();
        p.ensure_rows(1, 17).unwrap();
        p.publish(1, &b, 192, |_| Ok(())).unwrap();
        let stats = p.stats();
        assert!(stats.cache_bytes <= 512);
        assert_eq!(stats.snapshot_bytes, 192);
        assert_eq!(ops.frees.load(Ordering::SeqCst), 1);
        assert!(p.try_attach(0, &a).unwrap().is_none());
    }

    #[test]
    fn cache_budget_trims_after_active_radix_leases_release() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops.clone(), 448);
        let a = prompt(17);
        p.try_attach(0, &a).unwrap();
        p.ensure_rows(0, 17).unwrap();
        p.publish(0, &a, 192, |_| Ok(())).unwrap();
        p.try_attach(1, &a).unwrap().expect("shared prefix");
        p.ensure_rows(0, 25).unwrap();
        p.publish(0, &prompt(25), 64, |_| Ok(())).unwrap();
        assert_eq!(p.stats().cache_bytes, 576);
        assert_eq!(p.stats().snapshot_bytes, 192);
        assert_eq!(ops.frees.load(Ordering::SeqCst), 1);

        p.finish_attach(1);
        assert_eq!(p.stats().cache_bytes, 576);
        assert_eq!(p.stats().snapshot_bytes, 192);
        p.begin_seq(0);
        assert!(p.stats().cache_bytes <= 448);
        assert_eq!(p.stats().snapshot_bytes, 192);
        assert_eq!(ops.frees.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn idle_slot_releases_cache_holds_without_unmapping_writable_kv() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops.clone(), 448);
        let tokens = prompt(17);
        p.try_attach(0, &tokens).unwrap();
        p.ensure_rows(0, 17).unwrap();
        p.publish(0, &tokens, 192, |_| Ok(())).unwrap();
        p.try_attach(1, &tokens).unwrap().expect("shared prefix");
        let mapped = p.mapped_rows(0);

        p.release_prefix(0);
        assert_eq!(p.stats().snapshot_bytes, 192);
        p.release_prefix(1);
        {
            let mut inner = p.shared.inner.lock();
            while evict_one(&p.shared, &mut inner, false) {}
        }
        assert_eq!(p.stats().snapshot_bytes, 0);
        assert_eq!(p.stats().cache_bytes, 0);
        assert_eq!(p.mapped_rows(0), mapped);
        assert!(p.stats().blocks_live > 0);
        assert_eq!(ops.releases.load(Ordering::SeqCst), 0);
        p.release_prefix(0);
        p.begin_seq(0);
        p.begin_seq(1);
        assert_eq!(p.mapped_rows(0), 0);
        assert_eq!(p.mapped_rows(1), 0);
        let stats = p.stats();
        assert_eq!(stats.blocks_live, stats.cache_blocks);
    }

    #[test]
    fn short_boundaries_match_tokens_and_leave_a_token_to_recompute() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops, 256);
        let tokens = prompt(7);
        p.try_attach(0, &tokens).unwrap();
        p.ensure_rows(0, 7).unwrap();
        p.publish_at(0, &tokens, 4, 32, |_| Ok(())).unwrap();
        p.publish_at(0, &tokens, 6, 48, |_| Ok(())).unwrap();
        p.begin_seq(0);

        assert_eq!(p.try_attach(1, &tokens).unwrap().unwrap().rows, 6);
        assert_eq!(p.mapped_rows(1), 0);
        assert_eq!(p.stats().blocks_shared_mapped, 0);
        p.finish_attach(1);
        p.ensure_rows(1, 7).unwrap();
        assert_eq!(p.mapped_rows(1), 8);
        p.begin_seq(1);
        assert_eq!(p.try_attach(1, &tokens[..6]).unwrap().unwrap().rows, 4);
        p.begin_seq(1);
        let mut changed = tokens.clone();
        changed[5] ^= 1;
        assert_eq!(p.try_attach(1, &changed).unwrap().unwrap().rows, 4);
        p.begin_seq(1);
        changed[0] ^= 1;
        assert!(p.try_attach(1, &changed).unwrap().is_none());
    }

    #[test]
    fn partial_boundary_shares_only_complete_blocks() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        let tokens = prompt(25);
        p.try_attach(0, &tokens).unwrap();
        p.ensure_rows(0, 25).unwrap();
        p.publish_at(0, &tokens, 16, 64, |_| Ok(())).unwrap();
        p.publish_at(0, &tokens, 21, 96, |_| Ok(())).unwrap();
        p.begin_seq(0);
        assert_eq!(p.try_attach(0, &tokens[..21]).unwrap().unwrap().rows, 16);
        let hit = p.try_attach(1, &tokens).unwrap().unwrap();
        assert_eq!(hit.rows, 21);
        assert_eq!(p.mapped_rows(1), 16);
        assert_eq!(p.stats().blocks_shared_mapped, 8);
        ops.fail_creates.store(1, Ordering::SeqCst);
        assert!(p.ensure_rows(1, 22).is_err());
        p.ensure_rows(1, 22).unwrap();
        assert_eq!(p.mapped_rows(1), 24);
        p.finish_attach(1);
        p.begin_seq(1);
        p.begin_seq(0);
        let mut changed = tokens.clone();
        changed[20] ^= 1;
        assert_eq!(p.try_attach(1, &changed).unwrap().unwrap().rows, 16);
    }

    #[test]
    fn hot_short_snapshot_survives_publishers_holding_full_blocks() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops, 192);
        let prime = prompt(7);
        p.try_attach(0, &prime).unwrap();
        p.ensure_rows(0, 7).unwrap();
        p.publish_at(0, &prime, 6, 48, |_| Ok(())).unwrap();
        p.begin_seq(0);
        let request = prompt(9);
        assert_eq!(p.try_attach(1, &request).unwrap().unwrap().rows, 6);
        p.ensure_rows(1, 9).unwrap();
        p.finish_attach(1);
        p.publish_at(1, &request, 8, 48, |_| Ok(())).unwrap();

        assert_eq!(p.stats().cache_bytes, 176);
        assert_eq!(p.stats().snapshot_bytes, 48);
        assert_eq!(p.stats().snapshots_evicted, 1);
        let mut next = request;
        next[6] ^= 1;
        assert_eq!(p.try_attach(0, &next).unwrap().unwrap().rows, 6);
    }

    #[test]
    fn reused_short_prefix_survives_a_burst_of_unique_tails() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops, 96);
        let publish = |first| {
            let mut tokens = prompt(7);
            tokens[0] = first;
            p.begin_seq(0);
            p.try_attach(0, &tokens).unwrap();
            p.ensure_rows(0, 7).unwrap();
            p.publish_at(0, &tokens, 6, 48, |_| Ok(())).unwrap();
            tokens
        };
        let a = publish(10);
        let hit = p.try_attach(1, &a).unwrap().unwrap();
        p.finish_attach(1);
        p.begin_seq(1);
        let b = publish(20);
        for first in 100..116 {
            publish(first);
        }
        let c = publish(30);
        assert_eq!(p.try_attach(1, &a).unwrap().unwrap().snap_va, hit.snap_va);
        p.begin_seq(1);
        assert!(p.try_attach(1, &b).unwrap().is_none());
        p.begin_seq(1);
        assert_eq!(p.try_attach(1, &c).unwrap().unwrap().rows, 6);
        assert!(p.stats().cache_bytes <= 96);
        assert_eq!(p.stats().snapshots_evicted, 17);
    }

    #[test]
    fn reused_snapshots_leave_room_for_a_new_prefix() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops, 96);
        for first in [10, 20, 30] {
            let mut tokens = prompt(7);
            tokens[0] = first;
            p.begin_seq(0);
            p.try_attach(0, &tokens).unwrap();
            p.ensure_rows(0, 7).unwrap();
            p.publish_at(0, &tokens, 6, 48, |_| Ok(())).unwrap();
            p.begin_seq(1);
            assert_eq!(p.try_attach(1, &tokens).unwrap().unwrap().rows, 6);
            p.finish_attach(1);
        }
        p.begin_seq(1);
        let mut oldest = prompt(7);
        oldest[0] = 10;
        assert!(p.try_attach(1, &oldest).unwrap().is_none());
        assert_eq!(p.stats().cache_bytes, 96);
        assert_eq!(p.stats().snapshots_evicted, 1);
    }

    #[test]
    fn output_snapshots_leave_room_for_reusable_prompt_boundaries() {
        let p = pool_with_cap(Arc::new(MockVmm::default()), 400);
        let mut short = prompt(7);
        short[0] = 99;
        p.try_attach(0, &short).unwrap();
        p.ensure_rows(0, 7).unwrap();
        p.publish_at(0, &short, 6, 48, |_| Ok(())).unwrap();
        p.begin_seq(0);

        let long = prompt(17);
        p.try_attach(0, &long).unwrap();
        p.ensure_rows(0, 21).unwrap();
        p.publish_at(0, &long, 16, 48, |_| Ok(())).unwrap();
        p.try_attach(1, &short).unwrap().unwrap();
        p.finish_attach(1);

        let generated = prompt(21);
        p.publish_at(0, &generated, 18, 48, |_| Ok(())).unwrap();
        p.publish_at(0, &generated, 20, 48, |_| Ok(())).unwrap();
        p.begin_seq(1);
        let attached = p.try_attach(1, &long).unwrap().expect("prompt boundary retained");
        assert_eq!(attached.rows, 16);
        assert!(p.stats().cache_bytes <= 400);
        assert_eq!(p.stats().snapshots_evicted, 1);
    }

    #[test]
    fn output_snapshot_reused_by_a_followup_gets_prompt_priority() {
        let p = pool_with_cap(Arc::new(MockVmm::default()), 96);
        let tokens = prompt(7);
        p.try_attach(0, &tokens[..3]).unwrap();
        p.ensure_rows(0, 7).unwrap();
        p.publish_at(0, &tokens[..3], 2, 48, |_| Ok(())).unwrap();
        p.publish_at(0, &tokens[..5], 4, 48, |_| Ok(())).unwrap();
        assert_eq!(p.try_attach(1, &tokens[..5]).unwrap().unwrap().rows, 4);
        p.finish_attach(1);
        p.begin_seq(1);
        p.publish_at(0, &tokens, 6, 48, |_| Ok(())).unwrap();
        assert_eq!(p.try_attach(1, &tokens).unwrap().unwrap().rows, 4);
        assert_eq!(p.stats().cache_bytes, 96);
        assert_eq!(p.stats().snapshots_evicted, 1);
    }

    #[test]
    fn new_long_prompt_snapshot_survives_its_output_tail() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops, 400);
        for first in [10, 20] {
            let mut tokens = prompt(7);
            tokens[0] = first;
            p.begin_seq(0);
            p.try_attach(0, &tokens).unwrap();
            p.ensure_rows(0, 7).unwrap();
            p.publish_at(0, &tokens, 6, 48, |_| Ok(())).unwrap();
            p.begin_seq(1);
            p.try_attach(1, &tokens).unwrap().unwrap();
            p.finish_attach(1);
        }
        let tokens = prompt(18);
        p.begin_seq(0);
        p.try_attach(0, &tokens).unwrap();
        p.ensure_rows(0, 18).unwrap();
        p.publish_at(0, &tokens, 16, 48, |_| Ok(())).unwrap();
        p.publish_at(0, &tokens, 17, 48, |_| Ok(())).unwrap();
        p.begin_seq(0);
        p.begin_seq(1);
        assert_eq!(p.try_attach(1, &tokens[..17]).unwrap().unwrap().rows, 16);
        assert_eq!(p.stats().cache_bytes, 400);
        assert_eq!(p.stats().snapshots_evicted, 1);
    }

    #[test]
    fn active_radix_leases_do_not_flush_the_hot_snapshot_at_the_soft_cap() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops, 176);
        let short = prompt(7);
        p.try_attach(0, &short).unwrap();
        p.ensure_rows(0, 7).unwrap();
        p.publish_at(0, &short, 6, 48, |_| Ok(())).unwrap();
        p.begin_seq(0);
        p.try_attach(1, &short).unwrap().unwrap();
        p.finish_attach(1);
        p.begin_seq(1);
        let long = prompt(17);
        p.try_attach(0, &long).unwrap().unwrap();
        p.finish_attach(0);
        p.ensure_rows(0, 17).unwrap();
        p.publish_at(0, &long, 16, 48, |_| Ok(())).unwrap();
        assert_eq!(p.stats().cache_bytes, 304);
        assert_eq!(p.try_attach(1, &short).unwrap().unwrap().rows, 6);
        p.finish_attach(1);
        p.begin_seq(1);
        p.begin_seq(0);
        assert!(p.stats().cache_bytes <= 176);
    }

    #[test]
    fn oom_reclamation_can_evict_the_last_reused_snapshot() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops, 176);
        let tokens = prompt(7);
        p.try_attach(0, &tokens).unwrap();
        p.ensure_rows(0, 7).unwrap();
        p.publish_at(0, &tokens, 6, 48, |_| Ok(())).unwrap();
        p.begin_seq(0);
        p.try_attach(1, &tokens).unwrap().unwrap();
        p.finish_attach(1);
        p.begin_seq(1);
        let long = prompt(17);
        p.try_attach(0, &long).unwrap().unwrap();
        p.finish_attach(0);
        p.ensure_rows(0, 17).unwrap();
        p.publish_at(0, &long, 16, 48, |_| Ok(())).unwrap();
        assert_eq!(p.stats().cache_bytes, 304);
        let mut inner = p.shared.inner.lock();
        assert!(evict_one(&p.shared, &mut inner, false));
        assert_eq!(inner.stats.snapshot_bytes, 0);
    }

    #[test]
    fn short_snapshot_stays_pinned_until_restore_finishes() {
        let ops = Arc::new(MockVmm::default());
        let p = pool_with_cap(ops.clone(), 48);
        let a = prompt(7);
        p.try_attach(0, &a).unwrap();
        p.ensure_rows(0, 7).unwrap();
        p.publish_at(0, &a, 6, 48, |_| Ok(())).unwrap();
        let hit = p.try_attach(1, &a).unwrap().unwrap();
        p.begin_seq(0);
        let mut b = a.clone();
        b[0] ^= 1;
        p.try_attach(0, &b).unwrap();
        p.ensure_rows(0, 7).unwrap();
        p.publish_at(0, &b, 6, 48, |_| Ok(())).unwrap();
        let inner = p.shared.inner.lock();
        assert!(inner.published[&None].iter().any(|snap| snap.va == hit.snap_va && snap.users == 1));
        drop(inner);
        p.finish_attach(1);
        p.release_prefix(1);
        assert_eq!(p.stats().cache_bytes, 48);
        assert_eq!(ops.frees.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn partial_publication_rejects_changed_or_uncomputed_tokens() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        let tokens = prompt(17);
        p.try_attach(0, &tokens).unwrap();
        p.ensure_rows(0, 8).unwrap();
        assert!(p.publish_at(0, &tokens, 15, 48, |_| Ok(())).is_err());
        p.ensure_rows(0, 17).unwrap();
        let mut changed = tokens.clone();
        changed[0] ^= 1;
        assert!(p.publish_at(0, &changed, 15, 48, |_| Ok(())).is_err());
        assert!(p.publish_at(0, &tokens, 15, 48, |_| {
            Err(RuntimeError::Device("snapshot copy failed".into()))
        }).is_err());
        assert_eq!(p.stats().snapshot_bytes, 0);
        assert_eq!(ops.frees.load(Ordering::SeqCst), 1);
        p.begin_seq(0);
        assert!(p.try_attach(1, &tokens).unwrap().is_none());
    }

    #[test]
    fn snapshot_fill_runs_without_the_pool_lock() {
        let p = pool(Arc::new(MockVmm::default()));
        let tokens = prompt(17);
        p.try_attach(0, &tokens).unwrap();
        p.ensure_rows(0, 17).unwrap();
        p.publish_at(0, &tokens, 15, 48, |_| {
            assert!(p.shared.inner.try_lock().is_some());
            Ok(())
        }).unwrap();
    }

    #[test]
    fn concurrent_snapshot_publish_keeps_one_buffer() {
        let ops = Arc::new(MockVmm::default());
        let p = pool(ops.clone());
        let tokens = prompt(17);
        p.try_attach(0, &tokens).unwrap();
        p.ensure_rows(0, 17).unwrap();
        let barrier = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    p.publish_at(0, &tokens, 15, 48, |_| {
                        barrier.wait();
                        Ok(())
                    }).unwrap();
                });
            }
        });
        assert_eq!(p.stats().snapshot_bytes, 48);
        assert_eq!(ops.allocs.load(Ordering::SeqCst), 2);
        assert_eq!(ops.frees.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn drop_returns_everything() {
        let ops = Arc::new(MockVmm::default());
        {
            let p = pool(ops.clone());
            let pr = prompt(17);
            p.try_attach(0, &pr).unwrap();
            p.ensure_rows(0, 17).unwrap();
            p.publish(0, &pr, 64, |_| Ok(())).unwrap();
            p.try_attach(1, &pr).unwrap().expect("attach");
            p.ensure_rows(1, 17).unwrap();
        }
        // Every created block released, every snapshot freed, maps unmapped.
        assert_eq!(
            ops.creates.load(Ordering::SeqCst),
            ops.releases.load(Ordering::SeqCst),
            "physical blocks leaked"
        );
        assert_eq!(
            ops.allocs.load(Ordering::SeqCst),
            ops.frees.load(Ordering::SeqCst),
            "snapshots leaked"
        );
    }

    /// The slab maps everything (granularity-rounded), the watermark wait
    /// returns, and Drop releases exactly what was created.
    #[test]
    fn slab_maps_all_and_tears_down() {
        let ops = Arc::new(MockVmm::default());
        // gran 16: 100 B → reserved 112; chunk hint 30 → 32-B chunks
        // (32, 32, 32, 16).
        let slab = VmmSlab::new(ops.clone() as Arc<dyn VmmOps>, 100, 30).expect("slab");
        slab.wait_mapped(100).expect("mapped");
        assert_eq!(ops.creates.load(Ordering::SeqCst), 4);
        assert_eq!(ops.maps.load(Ordering::SeqCst), 4);
        drop(slab);
        assert_eq!(ops.unmaps.load(Ordering::SeqCst), 4);
        assert_eq!(ops.releases.load(Ordering::SeqCst), 4);
    }

    /// A commit failure mid-slab reaches the waiter as an error instead of a
    /// hang, and Drop still balances create/release for the chunks that DID
    /// commit.
    #[test]
    fn slab_commit_error_reaches_waiter() {
        let ops = Arc::new(MockVmm::default());
        ops.fail_creates.store(1, Ordering::SeqCst);
        let slab = VmmSlab::new(ops.clone() as Arc<dyn VmmOps>, 64, 16).expect("slab");
        assert!(slab.wait_mapped(64).is_err(), "commit error must propagate");
        drop(slab);
        assert_eq!(
            ops.creates.load(Ordering::SeqCst),
            ops.releases.load(Ordering::SeqCst),
            "created chunks leaked"
        );
    }

    /// Pooled chunks with matching sizes are re-mapped instead of created:
    /// 100 B at chunk 32 needs (32, 32, 32, 16); a pool holding one 32 and
    /// one 16 leaves exactly two creates. The pool is drained either way.
    #[test]
    fn slab_reuses_pooled_chunks() {
        let ops = Arc::new(MockVmm::default());
        ops.pool
            .lock()
            .unwrap()
            .extend([(9001u64, 32u64), (9002, 16)]);
        let slab = VmmSlab::new(ops.clone() as Arc<dyn VmmOps>, 100, 30).expect("slab");
        slab.wait_mapped(100).expect("mapped");
        assert_eq!(
            ops.creates.load(Ordering::SeqCst),
            2,
            "two chunks missing from pool"
        );
        assert_eq!(ops.maps.load(Ordering::SeqCst), 4, "all four chunks mapped");
        assert!(ops.pool.lock().unwrap().is_empty(), "pool drained");
        drop(slab); // PLOW_SLAB_KEEP unset → all four released
        assert_eq!(ops.releases.load(Ordering::SeqCst), 4);
    }

    /// Spin until the pre-creator has parked `want` blocks (it runs on its
    /// own thread; the tests need a settled pool before asserting counters).
    fn wait_pooled(p: &VmmKv, want: u64) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while p.stats().blocks_pooled < want {
            assert!(
                std::time::Instant::now() < deadline,
                "pre-creator never reached {want} pooled blocks"
            );
            std::thread::yield_now();
        }
    }

    /// `enable_block_pool`: the pre-creator fills the pool off the request
    /// path, `ensure_rows` draws from it instead of the driver, zero-ref
    /// blocks park back in the pool, and Drop releases every parked handle.
    #[test]
    fn kv_block_pool_recycles_and_precreates() {
        let ops = Arc::new(MockVmm::default());
        // uniform_pool: 2 full layers × K/V = 4 tracks, 1 kv head, 64 B
        // blocks → pre-create target = 4 × 1 × 2 = 8 blocks.
        let mut p = uniform_pool(ops.clone());
        p.enable_block_pool(8 * 64);
        wait_pooled(&p, 8);
        assert_eq!(ops.creates.load(Ordering::SeqCst), 8);

        // One window column (8 rows) = 4 blocks — all served from the pool.
        p.ensure_rows(0, 8).unwrap();
        assert_eq!(ops.creates.load(Ordering::SeqCst), 8, "no driver create");
        let st = p.stats();
        assert_eq!(st.blocks_reused, 4);
        assert_eq!(st.blocks_pooled, 4);
        assert_eq!(st.blocks_live, 4);

        // Window reset derefs to zero refs → blocks park, none released.
        p.begin_seq(0);
        p.ensure_rows(0, 8).unwrap();
        assert_eq!(
            ops.creates.load(Ordering::SeqCst),
            8,
            "recycled, not created"
        );
        p.begin_seq(0);
        assert_eq!(p.stats().blocks_pooled, 8);
        assert_eq!(ops.releases.load(Ordering::SeqCst), 0);

        drop(p);
        assert_eq!(
            ops.releases.load(Ordering::SeqCst),
            8,
            "every parked handle released at drop"
        );
    }

    #[test]
    fn live_kv_recycles_blocks_without_load_time_commit() {
        let ops = Arc::new(MockVmm::default());
        let geo = uniform_pool(ops.clone()).geometry().clone();
        let mut p = VmmKv::new_live(ops.clone(), geo, 64).unwrap();
        p.enable_block_recycling(8 * 64);
        assert_eq!(ops.creates.load(Ordering::SeqCst), 0);
        assert_eq!(p.stats().blocks_pooled, 0);
        assert_eq!(p.stats().blocks_live, 0);

        p.ensure_rows(0, 8).unwrap();
        assert_eq!(ops.creates.load(Ordering::SeqCst), 4);
        p.begin_seq(0);
        assert_eq!(p.stats().blocks_pooled, 4);
        p.ensure_rows(0, 8).unwrap();
        assert_eq!(ops.creates.load(Ordering::SeqCst), 4);
        assert_eq!(p.stats().blocks_reused, 4);

        p.begin_seq(0);
        drop(p);
        assert_eq!(ops.releases.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn live_kv_reuses_blocks_across_sequences() {
        let ops = Arc::new(MockVmm::default());
        let geo = uniform_pool(ops.clone()).geometry().clone();
        let mut p = VmmKv::new_live(ops.clone(), geo, 64).unwrap();
        p.enable_block_pool(8 * 64);
        wait_pooled(&p, 8);
        let created = ops.creates.load(Ordering::SeqCst);

        p.ensure_rows(0, 1).unwrap();
        p.begin_seq(0);
        p.ensure_rows(0, 1).unwrap();

        assert_eq!(
            ops.creates.load(Ordering::SeqCst),
            created,
            "live sequence reset must reuse pooled blocks"
        );
    }

    /// The pool cap bounds parked VRAM: overflowing zero-ref blocks release
    /// to the driver instead of parking.
    #[test]
    fn kv_block_pool_cap_bounds_parked_blocks() {
        let ops = Arc::new(MockVmm::default());
        let mut p = uniform_pool(ops.clone());
        // Cap = 2 blocks; pre-create target = min(2, 8) = 2.
        p.enable_block_pool(2 * 64);
        wait_pooled(&p, 2);

        // Full window: 4 columns × 4 tracks = 16 blocks (2 reused, 14 fresh).
        p.ensure_rows(0, 32).unwrap();
        assert_eq!(ops.creates.load(Ordering::SeqCst), 16);
        assert_eq!(p.stats().blocks_reused, 2);

        // All 16 hit zero refs: 2 park (cap), 14 release.
        p.begin_seq(0);
        let st = p.stats();
        assert_eq!(st.blocks_pooled, 2);
        assert_eq!(ops.releases.load(Ordering::SeqCst), 14);
    }

    /// The OOM path with pooling on: eviction's zero-ref blocks PARK in the
    /// pool (they don't free VRAM), so `create_block`'s retry loop must draw
    /// the pool on every iteration. Checking it only on entry spun
    /// create-fail/evict until the cache ran dry and reported OOM with
    /// reusable handles in hand.
    #[test]
    fn oom_eviction_feeds_the_pool_not_the_driver() {
        let ops = Arc::new(MockVmm::default());
        let mut p = uniform_pool(ops.clone());
        p.enable_block_pool(2 * 64); // cap 2 blocks; pre-creates 2
        wait_pooled(&p, 2);

        // Two cache-published columns on seq 0 (8 blocks held by the radix
        // cache after the window drops), then release the window.
        let pr = prompt(17);
        p.ensure_rows(0, 17).unwrap();
        p.publish(0, &pr, 4, |_| Ok(())).unwrap();
        p.begin_seq(0);

        // Every driver create now fails. Growing seq 1 by one column (4
        // blocks) must succeed anyway: 2 from the pool, then an eviction
        // parks its zero-ref blocks and the loop reuses them.
        ops.fail_creates.store(1 << 20, Ordering::SeqCst);
        let created_before = ops.creates.load(Ordering::SeqCst);
        p.ensure_rows(1, 8).unwrap();
        assert_eq!(
            ops.creates.load(Ordering::SeqCst),
            created_before,
            "no driver create can have succeeded"
        );
        assert!(
            p.stats().blocks_reused >= 4,
            "growth must have been served from pooled handles"
        );
    }

    #[test]
    fn hash_blocks_is_chained_and_positional() {
        let a = hash_blocks(&prompt(32), 8);
        assert_eq!(a.len(), 4);
        // Same block content at a different position must hash differently
        // (chained), so the radix can never alias positions.
        let mut two = prompt(8);
        two.extend(prompt(8));
        let b = hash_blocks(&two, 8);
        assert_eq!(b[0], a[0]);
        assert_ne!(b[1], a[1]);
    }
}

#[cfg(test)]
#[path = "vmm_ring_tests.rs"]
mod ring_tests;
