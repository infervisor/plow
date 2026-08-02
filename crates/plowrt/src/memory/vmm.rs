//! §L2 VMM-backed prefix sharing for the full-attention KV cache
//! (plans/rtx-09-prefix-headmajor.md, "Implementation V1").
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
//! node → physical block ids, and published-boundary → sliding-window
//! snapshot. Attach happens only at a **published boundary** (a whole-block
//! prefix a finished prefill published), because the sliding-layer rings are
//! not VMM-shared — their last `window` rows are restored from the boundary
//! snapshot (plan §8), and the sub-block prompt tail is recomputed by normal
//! prefill from the block-aligned `c0` (so tail writes land in a fresh
//! private block, never a shared one).
//!
//! ## Eviction (leak-audit finding #9)
//!
//! `cuMemCreate` OOM and the `cache_cap_bytes` soft cap both drive
//! `PrefixCache::evict_lru` until satisfied; physical blocks are released at
//! refcount 0 and boundary snapshots freed with their node.

use std::sync::atomic::{AtomicU32, Ordering};
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
}

/// KV geometry of the model the engine loaded, resolved from the checkpoint's
/// `config.json` (the blob carries only names+bytes) and validated against
/// the blob's tensor sizes before VMM is enabled.
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
        if full_layers.is_empty() || kvh_full == 0 || hd_full == 0 {
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

/// Result of a successful prefix attach.
#[derive(Clone, Copy, Debug)]
pub struct Attach {
    /// Rows now shared-mapped: the block-aligned prefix length. The engine
    /// resumes prefill at this frontier.
    pub rows: u32,
    /// Device VA of the sliding-window snapshot taken at `rows` — the engine
    /// restores it into the borrower's rings (its own layout, written by the
    /// `publish` fill closure).
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
}

/// One physical sharing block: driver handle + mapping/cache refcount.
/// The driver refcounts mappings itself (probe [2]) — this host count is
/// policy: it decides when WE release the handle.
struct Block {
    handle: u64,
    refs: u32,
}

/// Per-(full layer, K|V) VA reservation and its mapping table:
/// `slots[(seq·kvh + head)·bph + k]` = the block mapped at that window slot.
struct Track {
    layer: u32,
    /// 0 = K, 1 = V.
    tensor: u32,
    va: u64,
    slots: Vec<Option<u32>>,
}

/// Per-sequence-slot radix bookkeeping: the prompt's block hashes, the
/// block-aligned token prefix they were computed from, and how many leading
/// path nodes this sequence holds a reference on.
///
/// # `tokens` IS WRITTEN AND NEVER READ — the collision check is not implemented
///
/// This doc comment used to assert that "radix verification compares tokens on
/// every hash match — collision safety". It does not: `tokens` is populated at
/// three sites (each a `to_vec()` of the aligned prefix) and there is no reader
/// anywhere in the crate. So a `BlockHash` collision between two different
/// prefixes would be accepted as a cache hit and the second sequence would
/// attach to the first's KV blocks — fluent output from the wrong context, with
/// nothing to say so.
///
/// The field is kept rather than deleted precisely so the gap stays visible and
/// the data is already there when the comparison is written. Deleting it would
/// remove three allocations and the last trace of a safety property the code
/// claims to have.
#[derive(Default)]
struct SlotSeq {
    hashes: Vec<BlockHash>,
    #[allow(dead_code)]
    tokens: Vec<u32>,
    held: usize,
}

/// A published boundary's sliding-window snapshot buffer.
struct Snap {
    va: u64,
    bytes: u64,
}

struct Inner {
    tracks: Vec<Track>,
    blocks: Vec<Block>,
    free_ids: Vec<u32>,
    /// Whole blocks mapped per sequence (uniform across tracks/heads).
    seq_blocks: Vec<u32>,
    cache: PrefixCache,
    /// Radix node identity → the block ids backing it, ordered [track][head].
    node_blocks: FxHashMap<(u32, u32), Vec<u32>>,
    /// Published boundary node → sliding-window snapshot.
    published: FxHashMap<(u32, u32), Snap>,
    /// Monotonic publish id, used as the radix `owner_seq` so slot reuse can
    /// never collide two nodes on the same key.
    next_pub: u32,
    seqs: Vec<SlotSeq>,
    stats: VmmStats,
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
    inner: Mutex<Inner>,
    /// Per-seq mapped-row frontier, readable lock-free on the decode path.
    frontier: Vec<AtomicU32>,
}

/// The VMM-backed KV pool + prefix cache. One per engine; owns the VA
/// reservations, the physical block slab, the radix cache and the pre-mapper
/// thread. Everything is reclaimed on Drop (the lifecycle-test contract).
pub struct VmmKv {
    shared: Arc<Shared>,
    premap_tx: Option<std::sync::mpsc::Sender<(u32, u32)>>,
    premap_join: Option<std::thread::JoinHandle<()>>,
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
        let gran = ops.granularity()?;
        let row_bytes = geo.row_bytes();
        let head_span = geo.max_ctx as u64 * row_bytes;
        if head_span % gran != 0 {
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
        let block_rows = (block_bytes / row_bytes) as u32;
        let bph = (head_span / block_bytes) as u32;

        let span = geo.batch as u64 * geo.kvh_full as u64 * head_span;
        let mut tracks = Vec::with_capacity(geo.full_layers.len() * 2);
        let nslots = (geo.batch * geo.kvh_full * bph) as usize;
        for &layer in &geo.full_layers {
            for tensor in 0..2u32 {
                let va = ops.reserve(span)?;
                tracks.push(Track {
                    layer,
                    tensor,
                    va,
                    slots: vec![None; nslots],
                });
            }
        }

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
            inner: Mutex::new(Inner {
                tracks,
                blocks: Vec::new(),
                free_ids: Vec::new(),
                seq_blocks: vec![0; batch],
                cache,
                node_blocks: FxHashMap::default(),
                published: FxHashMap::default(),
                next_pub: 0,
                seqs: (0..batch).map(|_| SlotSeq::default()).collect(),
                stats: VmmStats::default(),
            }),
            frontier: (0..batch).map(|_| AtomicU32::new(0)).collect(),
        });

        // Pre-mapper: keeps the NEXT block mapped ahead of decode growth so
        // the 2048-token boundary never stalls a step (plan verdict §4 —
        // ~6.8 ms if synchronous, free when overlapped). `advise` feeds it;
        // `ensure_rows` in the step path is the correctness backstop.
        let (tx, rx) = std::sync::mpsc::channel::<(u32, u32)>();
        let premap_shared = Arc::clone(&shared);
        let join = std::thread::Builder::new()
            .name("vmm-premap".into())
            .spawn(move || {
                while let Ok((seq, pos)) = rx.recv() {
                    let s = &premap_shared;
                    let target = ((pos / s.block_rows) + 2)
                        .saturating_mul(s.block_rows)
                        .min(s.geo.max_ctx);
                    if s.frontier[seq as usize].load(Ordering::Acquire) < target {
                        if let Err(e) = ensure_rows(s, seq as usize, target) {
                            // Non-fatal here: the synchronous backstop in the
                            // step path surfaces the real error.
                            tracing::warn!(error = %e, seq, target, "vmm pre-map failed");
                        }
                    }
                }
            })
            .map_err(|e| RuntimeError::Device(format!("vmm premap thread: {e}")))?;

        tracing::info!(
            full_layers = shared.geo.full_layers.len(),
            kv_elem = shared.geo.elem,
            kv_elem_slide = shared.geo.elem_slide,
            block_mib = block_bytes >> 20,
            block_rows,
            bph,
            va_gib = (span * shared.geo.full_layers.len() as u64 * 2) as f64 / (1u64 << 30) as f64,
            "vmm kv pool up (full layers VMM-backed, sliding on cudaMalloc)"
        );
        Ok(VmmKv {
            shared,
            premap_tx: Some(tx),
            premap_join: Some(join),
        })
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
        ensure_rows(&self.shared, seq, rows)
    }

    /// Decode-growth hint: ask the pre-mapper to keep the next block mapped
    /// beyond `pos`. Never blocks; drops silently after shutdown.
    pub fn advise(&self, seq: usize, pos: u32) {
        if let Some(tx) = &self.premap_tx {
            let _ = tx.send((seq as u32, pos));
        }
    }

    /// Start a new sequence in `seq`: release the previous sequence's radix
    /// references and unmap+deref its window (cached blocks survive through
    /// the cache's own references).
    pub fn begin_seq(&self, seq: usize) {
        let s = &self.shared;
        let mut inner = s.inner.lock();
        let held = std::mem::take(&mut inner.seqs[seq]);
        if held.held > 0 {
            inner.cache.release(&held.hashes, held.held);
        }
        release_window(s, &mut inner, seq);
    }

    /// Try to attach a cached prefix of `prompt` into `seq`'s windows:
    /// longest radix match, clipped to the longest **published boundary**
    /// (sliding snapshot available) and to `< prompt.len()` (the tail —
    /// at least the last token — is recomputed by prefill, which also
    /// regenerates the boundary block's sliding rows). On a hit the shared
    /// blocks are multi-mapped (refcounted) and the snapshot handle returned.
    /// On miss the prompt's hashes are still recorded for `publish`.
    pub fn try_attach(&self, seq: usize, prompt: &[u32]) -> Result<Option<Attach>> {
        let s = &self.shared;
        let hashes = hash_blocks(prompt, s.block_rows);
        let aligned = &prompt[..hashes.len() * s.block_rows as usize];
        let mut inner = s.inner.lock();

        let m = inner.cache.lookup(&hashes, aligned);
        let limit = ((prompt.len() as u64).saturating_sub(1) / s.block_rows as u64) as usize;
        let mut pick = 0usize;
        for i in (1..=m.blocks.min(limit)).rev() {
            if inner.published.contains_key(&m.placed[i - 1]) {
                pick = i;
                break;
            }
        }
        if pick == 0 {
            inner.stats.attach_misses += 1;
            inner.cache.release(&hashes, m.blocks);
            inner.seqs[seq] = SlotSeq {
                hashes,
                tokens: aligned.to_vec(),
                held: 0,
            };
            return Ok(None);
        }
        inner.stats.attach_hits += 1;
        inner.stats.tokens_attached += pick as u64 * s.block_rows as u64;
        if pick < m.blocks {
            // Keep references only on the attached prefix of the path.
            inner.cache.release(&hashes, m.blocks);
            let again = inner.cache.lookup(&hashes[..pick], aligned);
            debug_assert_eq!(again.blocks, pick);
        }
        let placed: Vec<(u32, u32)> = m.placed[..pick].to_vec();
        let snap_key = placed[pick - 1];
        inner.seqs[seq] = SlotSeq {
            hashes,
            tokens: aligned.to_vec(),
            held: pick,
        };

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
                            let b = &mut inner.blocks[id as usize];
                            b.refs -= 1;
                            if b.refs == 0 {
                                frees.push(b.handle);
                                inner.stats.blocks_live -= 1;
                                inner.free_ids.push(id);
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
        let snap = &inner.published[&snap_key];
        let attach = Attach {
            rows: pick as u32 * s.block_rows,
            snap_va: snap.va,
            snap_bytes: snap.bytes,
        };
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
                            let blk = &mut inner.blocks[id as usize];
                            blk.refs -= 1;
                            if blk.refs == 0 {
                                orphans.push(blk.handle);
                                inner.stats.blocks_live -= 1;
                                inner.free_ids.push(id);
                            }
                        }
                    }
                }
            }
            inner.seq_blocks[seq] = 0;
            s.frontier[seq].store(0, Ordering::Release);
            let held = std::mem::take(&mut inner.seqs[seq]);
            if held.held > 0 {
                inner.cache.release(&held.hashes, held.held);
            }
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
        let s = &self.shared;
        let hashes = hash_blocks(tokens, s.block_rows);
        let aligned = &tokens[..hashes.len() * s.block_rows as usize];
        let mut inner = s.inner.lock();
        let n_pub = hashes.len().min(inner.seq_blocks[seq] as usize);
        if n_pub == 0 {
            return Ok(());
        }
        let held_old = inner.seqs[seq].held;
        let tokens = aligned;

        let m = inner.cache.lookup(&hashes[..n_pub], &tokens);
        let pid = inner.next_pub;
        inner.next_pub += 1;
        let n_ok = inner.cache.insert(&hashes[..n_pub], &tokens, pid, m.blocks);
        // A collision (or stale path) stopped the insert early: blocks past
        // n_ok have no tree node — referencing them would leak their handles
        // behind a key no lookup can ever reach.
        let n_pub = n_pub.min(n_ok);
        if n_pub == 0 {
            inner.cache.release(&hashes, held_old);
            inner.seqs[seq].held = 0;
            return Ok(());
        }

        // Hand the cache a reference on every newly-published block.
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
            inner.node_blocks.insert((pid, idx as u32), ids);
        }
        // Path references: replace the admission-time holds with one hold per
        // published node (released at the next begin_seq), and record the
        // published stream so a later (longer) publish extends it.
        inner.cache.release(&hashes, held_old);
        inner.seqs[seq] = SlotSeq {
            tokens: tokens.to_vec(),
            hashes,
            held: n_pub,
        };

        // Boundary snapshot, keyed by the boundary node's identity.
        let bkey = if n_pub <= m.blocks {
            m.placed[n_pub - 1]
        } else {
            (pid, n_pub as u32 - 1)
        };
        if !inner.published.contains_key(&bkey) {
            let va = s.ops.alloc(snap_bytes)?;
            if let Err(e) = fill(va) {
                s.ops.free(va);
                return Err(e);
            }
            inner.published.insert(
                bkey,
                Snap {
                    va,
                    bytes: snap_bytes,
                },
            );
        }

        // Soft cap: shed cold cache until under budget (finding #9).
        if s.cache_cap > 0 {
            while inner.stats.cache_blocks * s.block_bytes > s.cache_cap {
                if !evict_one(s, &mut inner) {
                    break;
                }
            }
        }
        Ok(())
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
        for (_, snap) in std::mem::take(&mut inner.published) {
            s.ops.free(snap.va);
        }
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

fn ensure_rows(s: &Shared, seq: usize, rows: u32) -> Result<()> {
    let rows = rows.min(s.geo.max_ctx);
    if s.frontier[seq].load(Ordering::Acquire) >= rows {
        return Ok(());
    }
    let mut inner = s.inner.lock();
    let target = rows.div_ceil(s.block_rows);
    let kvh = s.geo.kvh_full;
    for k in inner.seq_blocks[seq]..target {
        for t in 0..inner.tracks.len() {
            for h in 0..kvh {
                let id = create_block(s, &mut inner)?;
                let va = slot_va(s, &inner.tracks[t], seq, h, k);
                let handle = inner.blocks[id as usize].handle;
                s.ops.map(va, s.block_bytes, handle)?;
                s.ops.set_access(va, s.block_bytes)?;
                let slot = slot_index(s, seq, h, k);
                debug_assert!(inner.tracks[t].slots[slot].is_none());
                inner.tracks[t].slots[slot] = Some(id);
            }
        }
        inner.seq_blocks[seq] = k + 1;
        s.frontier[seq].store((k + 1) * s.block_rows, Ordering::Release);
    }
    Ok(())
}

/// Create one physical block, evicting cache LRU on allocation failure
/// (the OOM half of finding #9's auto-eviction).
fn create_block(s: &Shared, inner: &mut Inner) -> Result<u32> {
    loop {
        match s.ops.create(s.block_bytes) {
            Ok(handle) => {
                inner.stats.blocks_created += 1;
                inner.stats.blocks_live += 1;
                let block = Block { handle, refs: 1 };
                return Ok(match inner.free_ids.pop() {
                    Some(id) => {
                        inner.blocks[id as usize] = block;
                        id
                    }
                    None => {
                        inner.blocks.push(block);
                        (inner.blocks.len() - 1) as u32
                    }
                });
            }
            Err(e) => {
                if !evict_one(s, inner) {
                    return Err(RuntimeError::Oom(format!("vmm kv block: {e}")));
                }
            }
        }
    }
}

/// Evict the LRU zero-ref radix leaf, dereferencing its blocks and freeing
/// its boundary snapshot. `false` when nothing is evictable.
fn evict_one(s: &Shared, inner: &mut Inner) -> bool {
    let Some(key) = inner.cache.evict_lru() else {
        return false;
    };
    if let Some(ids) = inner.node_blocks.remove(&key) {
        inner.stats.cache_blocks -= ids.len() as u64;
        for id in ids {
            deref_block(s, inner, id);
        }
    }
    if let Some(snap) = inner.published.remove(&key) {
        s.ops.free(snap.va);
    }
    inner.stats.nodes_evicted += 1;
    true
}

fn deref_block(s: &Shared, inner: &mut Inner, id: u32) {
    let b = &mut inner.blocks[id as usize];
    b.refs -= 1;
    if b.refs == 0 {
        s.ops.release(b.handle);
        inner.stats.blocks_live -= 1;
        inner.free_ids.push(id);
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

    /// Records every driver call; `fail_creates` makes the next N creates
    /// fail (the OOM path).
    #[derive(Default)]
    struct MockVmm {
        next: AtomicU64,
        creates: AtomicU64,
        releases: AtomicU64,
        maps: AtomicU64,
        unmaps: AtomicU64,
        allocs: AtomicU64,
        frees: AtomicU64,
        fail_creates: AtomicI64,
        fail_maps: AtomicI64,
    }

    impl VmmOps for MockVmm {
        fn granularity(&self) -> Result<u64> {
            Ok(16)
        }
        fn reserve(&self, _bytes: u64) -> Result<u64> {
            Ok(self.next.fetch_add(1 << 32, Ordering::SeqCst) + (1 << 32))
        }
        fn address_free(&self, _va: u64, _bytes: u64) {}
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
    }

    /// 1 full layer, 1 kv head, hd 4 bf16 (8 B rows), 32-row context,
    /// 64 B blocks → block_rows = 8, 4 blocks/head, 2 tracks (K, V).
    fn pool(ops: Arc<MockVmm>) -> VmmKv {
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
        VmmKv::new(ops, geo, 64, 0).expect("pool")
    }

    fn prompt(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| i * 7 + 3).collect()
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
