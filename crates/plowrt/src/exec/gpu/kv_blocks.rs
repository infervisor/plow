//! The KV block store on the CUDA engine (`docs/arch/26-kv-block-store.md` Part B, B1).
//!
//! A boundary at `p_a` rows is a set of blocks: the sliding windows' last `window` rows and
//! their fp8 ring scales in 32-row **window** blocks, every full-layer row `[0, p_a)` with its
//! fp8 scales in 32-row **prefix** blocks (`Class::Tail`), and for a recurrent model its
//! state in one **state** block. Every block is keyed by the chained hash of the tokens up to
//! its last row, so two boundaries of one conversation, or two conversations on one system
//! prompt, share every block their rows have in common: a turn publishes only its new rows.
//! The store is the only storage of a boundary: the pool's radix tree indexes boundaries and
//! multi-maps nothing (`VmmKv::enable_block_store(.., true)`); an attach maps the slot's own
//! blocks (`ensure_rows`) and gathers every row into them.
//!
//! The copies are block descriptors (block address, slot row, slot) over a static per-class
//! layout of (tensor, head) runs, executed by one `plow_kv_blockcopy` launch per class on the
//! engine stream with the descriptors in the launch parameters (PTX embedded, driver-JIT at
//! load); a backend without the kernel expands them into a memcpy loop. Model kernels see only
//! the slot's rings and tracks, exactly as with the contiguous snapshots this replaces.

use super::prefix::VmmServe;
use super::GpuEngine;
use crate::device::cuda::{CudaBackend, CudaEvent, KernelFn};
use crate::memory::kv_store::{
    chain_hashes, BlockId, ClassSpec, FillToken, Hold, KvStore, StoreStats,
};
use crate::memory::vmm::{SnapBlocks, VmmOps};
use crate::{Result, RuntimeError};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const ROWS_PER_BLOCK: u32 = 32;
const CHUNK_BYTES: u64 = 8 << 20;
/// Device headroom the spill-ahead thread keeps for publishes when a host tier is armed
/// (`VmmKv::spawn_spill_ahead`): about a second of boundaries at the production rate.
pub const SPILL_AHEAD_BYTES: u64 = 2 << 30;
/// Longest a publish waits in all for queued spills to free device slots.
const SPILL_WAIT: std::time::Duration = std::time::Duration::from_millis(50);
const SEED_CHAIN: u64 = 0x9e37_79b9_7f4a_7c15;
const SEED_CHECK: u64 = 0xc2b2_ae3d_27d4_eb4f;
const SEGCOPY_PTX: &str = include_str!("../../../../../runtime/nvidia/kv_segcopy.ptx");
const SEGCOPY_ENTRY: &str = "plow_kv_blockcopy";
const SEGCOPY_BLOCK: u32 = 256;
/// Descriptors per launch (`PLOW_KV_DESCS` in `kv_segcopy.cu`).
const DESCS: usize = 2032;
/// State class runs are cut into chunks of this size (one CTA each).
const STATE_CHUNK: u64 = 64 << 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Window,
    Tail,
    Scale,
    /// Recurrent state (GDN, conv history) captured at the boundary row: one block per
    /// boundary, every `state.*` tensor's per-slot bytes back to back.
    State,
}

impl Class {
    fn index(self) -> usize {
        self as usize
    }
}

/// One (tensor, head) run of a class (`plow_kv_layout` in `kv_segcopy.cu`): slot `s`, slot row
/// `r` at `dev + s * slot_stride + r * row_bytes`; in the block at `block_off`, `bytes` long.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    dev: u64,
    slot_stride: u64,
    row_bytes: u32,
    block_off: u32,
    bytes: u32,
    pad: u32,
}

/// One block of a copy (`plow_kv_desc`): its bytes, its first slot row, the slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct Desc {
    block: u64,
    row: u32,
    slot: u32,
}

/// `plow_kv_blockcopy_args`: the launch parameters, copied by the driver at launch.
#[repr(C)]
struct CopyArgs {
    layout: u64,
    to_block: u32,
    n: u32,
    d: [Desc; DESCS],
}

/// Layout ranges past the four classes: the state class without the tensors the first packed
/// prefill reads from the block itself ([`PendingState`]), and that read (block or slot state
/// into the contiguous initial-state scratch, one descriptor per scanned request).
const STATE_REST: usize = 4;
const STATE_INIT: usize = 5;
const RANGES: usize = 6;

/// Every range's runs, uploaded once (the slot tensors never move); `ranges[r]` =
/// `(first entry, entries)`.
struct Layouts {
    dev: Option<TableDev>,
    host: Vec<Layout>,
    ranges: [(u32, u32); RANGES],
}

/// Engine-stream work not known complete: once `done` retires, `ids` (deferred fills) seal
/// and `holds` drop (every reader of those blocks is on the engine stream).
struct PendingSeal {
    done: CudaEvent,
    ids: Vec<BlockId>,
    holds: Vec<Hold>,
}

/// An attach whose recurrent state was not copied into the slot: the slot's first packed
/// prefill reads each `left` tensor `(devp index, offset in the block)` straight from the
/// held block into its initial-state scratch; anything else that would read the slot's state
/// first copies the remaining tensors in ([`GpuEngine::state_materialize`]).
struct PendingState {
    hold: Hold,
    va: u64,
    left: Vec<(usize, u64)>,
}

/// A boundary's blocks: `(class, first row, block)` in the order the segments are built.
type Handle = Vec<(Class, u32, BlockId)>;

/// The dependencies a publish took on its boundary's blocks while the publish is in flight:
/// released when dropped (an early return, a failed copy), kept once [`Self::into_handle`]
/// records the boundary. Dropped by an unwind the dependencies stay (a copy submitted before
/// the panic may still write the blocks). Declared before the fill tokens of the same publish,
/// so a drop fails the blocks first and releases them after.
#[must_use = "dropping the set releases the dependencies"]
struct Deps<'s> {
    store: &'s KvStore,
    handle: Handle,
}

impl<'s> Deps<'s> {
    fn new(store: &'s KvStore, cap: usize) -> Self {
        Deps {
            store,
            handle: Vec::with_capacity(cap),
        }
    }

    fn push(&mut self, class: Class, row0: u32, id: BlockId) {
        self.handle.push((class, row0, id));
    }

    /// The boundary is recorded: its handle owns the dependencies from here on.
    fn into_handle(self) -> Handle {
        let mut this = std::mem::ManuallyDrop::new(self);
        std::mem::take(&mut this.handle)
    }
}

impl Drop for Deps<'_> {
    fn drop(&mut self) {
        if !self.handle.is_empty() && !std::thread::panicking() {
            let ids: Vec<BlockId> = self.handle.iter().map(|&(_, _, id)| id).collect();
            self.store.release(&ids);
        }
    }
}

/// The layout's device copy, allocated and freed through the pool's `VmmOps` (its `Oom` mapping
/// and provenance record, which `Backend::alloc` lacks); freed once, when dropped.
struct TableDev {
    be: Arc<CudaBackend>,
    va: u64,
}

impl TableDev {
    fn alloc(be: &Arc<CudaBackend>, bytes: u64) -> Result<Self> {
        let va = VmmOps::alloc(&**be, bytes)?;
        Ok(TableDev {
            be: Arc::clone(be),
            va,
        })
    }
}

impl Drop for TableDev {
    fn drop(&mut self) {
        VmmOps::free(&*self.be, self.va);
    }
}

#[derive(Clone, Debug, Default)]
pub struct BlockServeStats {
    pub store: StoreStats,
    pub publishes: u64,
    pub published_blocks: u64,
    pub published_new_blocks: u64,
    pub gathers: u64,
    pub gathered_blocks: u64,
    pub segments: u64,
    pub launches: u64,
    pub memcpy_segments: u64,
    /// Engine-thread wall time in publishes and gathers, and the part spent draining the
    /// stream (the rest is hashing, acquires, table building and upload).
    pub publish_us: u64,
    pub publish_sync_us: u64,
    pub gather_us: u64,
    pub gather_sync_us: u64,
    /// The longest single publish and gather.
    pub publish_max_us: u64,
    pub gather_max_us: u64,
    /// Boundary restores started ahead of an attach (prefetch or attach), the blocks they
    /// queued and their time; gathered blocks whose copy the engine stream waited on.
    pub restores: u64,
    pub restore_blocks: u64,
    pub restore_us: u64,
    pub gather_fenced_blocks: u64,
    /// Publishes whose seal waited for the scatter's event instead of a stream drain.
    pub seals_deferred: u64,
}

pub struct BlockServe {
    store: Arc<KvStore>,
    /// Store class index per row class (`None` = the packet has no such rows).
    window: Option<usize>,
    tail: Option<usize>,
    scale: Option<usize>,
    state: Option<usize>,
    /// `(devp index, per-slot bytes)` of the recurrent state tensors (`State` class).
    state_tensors: Vec<(usize, u64)>,
    block_rows: u32,
    handles: Mutex<FxHashMap<u64, Handle>>,
    next_handle: AtomicU64,
    kernel: Option<KernelFn>,
    layouts: std::sync::OnceLock<Layouts>,
    args: Mutex<Box<CopyArgs>>,
    seals: Mutex<Vec<PendingSeal>>,
    spare_events: Mutex<Vec<CudaEvent>>,
    /// The state tensors a first packed prefill can read from a block (`(devp index, offset in
    /// the block)`), set once by the engine; empty = every attach copies the whole state.
    state_defer: std::sync::OnceLock<Vec<(usize, u64)>>,
    pending_state: Mutex<Vec<Option<PendingState>>>,
    /// Transfer holds a launch-window gather keeps until the step's own drain
    /// ([`GpuEngine::flush_deferred_attach`]).
    held: Mutex<Vec<Vec<Hold>>>,
    stats: Mutex<BlockServeStats>,
    _grower_stop: WorkerStop,
}

/// Stops the store's worker threads when dropped: with the `BlockServe`, or on a failed
/// construction after a worker was already started.
struct WorkerStop(Arc<std::sync::atomic::AtomicBool>);

impl Drop for WorkerStop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

impl BlockServe {
    /// `device_cap` / `host_cap`: bytes the store may hold per tier; `state_tensors` = the
    /// engine's recurrent state tensors (`(devp index, per-slot bytes)`), empty for a model
    /// without one.
    pub(super) fn new(
        be: &Arc<CudaBackend>,
        v: &VmmServe,
        device_cap: u64,
        host_cap: u64,
        state_tensors: &[(usize, u64)],
    ) -> Result<Arc<Self>> {
        let kv = v.kv.as_ref().expect("block store needs the prefix pool");
        let g = kv.geometry();
        let block_rows = kv.block_rows();
        if block_rows % ROWS_PER_BLOCK != 0 {
            return Err(RuntimeError::Rejected(format!(
                "kv store: pool block of {block_rows} rows is not a multiple of {ROWS_PER_BLOCK}"
            )));
        }
        let hd_slide = u64::from(g.hd_slide * g.elem_slide);
        let hd_full = u64::from(g.hd_full * g.elem);
        let window_row = v.slide.len() as u64 * 2 * u64::from(g.kvh_slide) * hd_slide
            + v.slide_scale.len() as u64 * 2 * u64::from(g.kvh_slide) * 4;
        let tail_row = v.tensor_tracks.len() as u64 * u64::from(g.kvh_full) * hd_full
            + v.full_scale.len() as u64 * 2 * u64::from(g.kvh_full) * 4;
        let scale_row = v.full_scale.len() as u64 * 2 * u64::from(g.kvh_full) * 4;
        let state_row: u64 = state_tensors.iter().map(|&(_, bytes)| bytes).sum();
        let mut specs = Vec::new();
        let mut class = |name: &'static str, row_bytes: u64, rows: u32| -> Option<usize> {
            (row_bytes > 0).then(|| {
                specs.push(ClassSpec {
                    name,
                    row_bytes,
                    rows_per_block: rows,
                });
                specs.len() - 1
            })
        };
        let window = class("window", window_row, ROWS_PER_BLOCK);
        let tail = class("prefix", tail_row, ROWS_PER_BLOCK);
        let scale = None;
        let state = class("state", state_row, 1);
        if specs.is_empty() {
            return Err(RuntimeError::Rejected(
                "kv store: the packet has no boundary rows to store".into(),
            ));
        }
        // Every segment is a whole run of 32 (or `block_rows`) rows of one head: 16-byte
        // vectors need those runs 16-byte sized.
        let vector_ok = [
            ROWS_PER_BLOCK as u64 * hd_slide,
            ROWS_PER_BLOCK as u64 * hd_full,
        ]
        .iter()
        .chain(state_tensors.iter().map(|(_, bytes)| bytes))
        .all(|b| b % 16 == 0);
        let kernel = if vector_ok {
            match Self::load_kernel(be) {
                Ok(f) => Some(f),
                Err(e) => {
                    tracing::warn!(error = %e, "kv store: segment-copy kernel unavailable, memcpy loop");
                    None
                }
            }
        } else {
            None
        };
        let store = Arc::new(KvStore::new(
            Arc::clone(be) as Arc<dyn crate::memory::kv_store::StoreOps>,
            specs,
            CHUNK_BYTES,
            device_cap,
            host_cap,
        ));
        let batch = g.batch as usize;
        // Pinned chunks are allocated ahead on their own thread (`host_grow_step`), never
        // under the pool lock, and released device chunks are freed on another (a device free
        // synchronises the context: behind the spill backlog that held the engine thread
        // 150 ms per shrink); both threads end with the store.
        let grower_stop = WorkerStop(Arc::new(std::sync::atomic::AtomicBool::new(false)));
        {
            store.defer_frees();
            let store = Arc::clone(&store);
            let stop = Arc::clone(&grower_stop.0);
            std::thread::Builder::new()
                .name("kv-store-free".into())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        if !store.free_step() {
                            std::thread::sleep(std::time::Duration::from_millis(2));
                        }
                    }
                })
                .map_err(|e| RuntimeError::Rejected(format!("kv store: free worker: {e}")))?;
        }
        if host_cap > 0 {
            let store = Arc::clone(&store);
            let stop = Arc::clone(&grower_stop.0);
            std::thread::Builder::new()
                .name("kv-store-host-grower".into())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        if !store.host_grow_step() {
                            std::thread::sleep(std::time::Duration::from_millis(2));
                        }
                    }
                })
                .map_err(|e| RuntimeError::Rejected(format!("kv store: host grower: {e}")))?;
        }
        tracing::info!(
            window_row,
            tail_row,
            scale_row,
            state_row,
            device_cap_mib = device_cap >> 20,
            host_cap_mib = host_cap >> 20,
            kernel = kernel.is_some(),
            "kv store: armed"
        );
        Ok(Arc::new(BlockServe {
            store,
            window,
            tail,
            scale,
            state,
            state_tensors: state_tensors.to_vec(),
            block_rows,
            handles: Mutex::new(FxHashMap::default()),
            next_handle: AtomicU64::new(1),
            kernel,
            layouts: std::sync::OnceLock::new(),
            args: Mutex::new(Box::new(CopyArgs {
                layout: 0,
                to_block: 0,
                n: 0,
                d: [Desc::default(); DESCS],
            })),
            seals: Mutex::new(Vec::new()),
            spare_events: Mutex::new(Vec::new()),
            state_defer: std::sync::OnceLock::new(),
            pending_state: Mutex::new((0..batch).map(|_| None).collect()),
            held: Mutex::new((0..batch).map(|_| Vec::new()).collect()),
            _grower_stop: grower_stop,
            stats: Mutex::new(BlockServeStats::default()),
        }))
    }

    fn load_kernel(be: &Arc<CudaBackend>) -> Result<KernelFn> {
        let mut image = SEGCOPY_PTX.as_bytes().to_vec();
        image.push(0);
        let module = crate::device::Backend::module_load(&**be, &image)?;
        be.get_function(&module, SEGCOPY_ENTRY)
    }

    pub fn stats(&self) -> BlockServeStats {
        let mut s = self.stats.lock().clone();
        s.store = self.store.stats();
        s
    }

    fn unhold_slot(&self, b: usize) {
        // Taken out first: the guards' drop takes the store lock, not the stash lock.
        let holds = std::mem::take(&mut self.held.lock()[b]);
        drop(holds);
    }

    /// The blocks a boundary at `p_a` rows is made of, with the chained key index of each.
    fn plan(&self, p_a: u32, window: u32) -> Vec<(Class, u32, usize)> {
        plan_blocks(
            [
                self.window.is_some(),
                self.tail.is_some(),
                self.scale.is_some(),
                self.state.is_some(),
            ],
            self.block_rows,
            p_a,
            window,
        )
    }

    /// Whether boundaries carry recurrent state: then a boundary is publishable only while the
    /// slot stands exactly at it ([`GpuEngine::state_capture_cap`]).
    pub(super) fn captures_state(&self) -> bool {
        self.state.is_some()
    }

    fn class_index(&self, class: Class) -> usize {
        match class {
            Class::Window => self.window,
            Class::Tail => self.tail,
            Class::Scale => self.scale,
            Class::State => self.state,
        }
        .expect("planned class exists")
    }
}

impl Drop for BlockServe {
    fn drop(&mut self) {
        // Workers stop before the tables go (the field's own drop would run after).
        self._grower_stop.0.store(true, Ordering::Relaxed);
        // Holds a failed drain left behind stay with the store: their copies' completion is
        // unproven.
        for hold in self.held.get_mut().drain(..).flatten() {
            hold.abandon();
        }
        for p in self.pending_state.get_mut().drain(..).flatten() {
            p.hold.abandon();
        }
        for p in self.seals.get_mut().drain(..) {
            for hold in p.holds {
                hold.abandon();
            }
        }
    }
}

impl SnapBlocks for BlockServe {
    fn release(&self, handle: u64) {
        if let Some(h) = self.handles.lock().remove(&handle) {
            let ids: Vec<BlockId> = h.iter().map(|&(_, _, id)| id).collect();
            self.store.release(&ids);
        }
    }

    fn spill(&self, handle: u64) -> bool {
        let ids: Vec<BlockId> = match self.handles.lock().get(&handle) {
            Some(h) => h.iter().map(|&(_, _, id)| id).collect(),
            None => return false,
        };
        self.store.spill(&ids)
    }

    /// With the copy kernel the host blocks start back to device slots on the copy stream
    /// (a gather waits on the copies where they have not landed; a block no slot could be made
    /// for stays on the host and is read in place); the memcpy fallback copies them back first.
    fn restore(&self, handle: u64) -> Result<()> {
        let ids: Vec<BlockId> = match self.handles.lock().get(&handle) {
            Some(h) => h.iter().map(|&(_, _, id)| id).collect(),
            None => return Err(RuntimeError::Rejected("kv store: unknown boundary".into())),
        };
        if self.kernel.is_some() {
            let t0 = std::time::Instant::now();
            let mut queued = 0u64;
            for id in ids {
                if matches!(self.store.restore_async(id), Ok(true)) {
                    queued += 1;
                }
            }
            if queued > 0 {
                let mut st = self.stats.lock();
                st.restores += 1;
                st.restore_blocks += queued;
                st.restore_us += t0.elapsed().as_micros() as u64;
            }
            return Ok(());
        }
        for id in ids {
            self.store.ensure_device(id)?;
        }
        Ok(())
    }

    fn restoring(&self, handle: u64) -> bool {
        let ids: Vec<BlockId> = match self.handles.lock().get(&handle) {
            Some(h) => h.iter().map(|&(_, _, id)| id).collect(),
            None => return false,
        };
        self.store.off_device_any(&ids)
    }

    fn reap(&self) {
        self.store.reap();
    }

    fn pending_spill_bytes(&self) -> u64 {
        self.store.pending_spill_bytes()
    }

    fn captures_state(&self) -> bool {
        self.state.is_some()
    }

    fn shrink(&self, bytes: u64) -> u64 {
        self.store.shrink(bytes)
    }
}

impl BlockServe {
    /// The device budget the pool's pressure regime leaves the store (set once free memory
    /// after load is known).
    pub fn set_device_cap(&self, bytes: u64) {
        self.store.set_device_cap(bytes);
        tracing::info!(device_cap_mib = bytes >> 20, "kv store: device cap");
    }

    pub fn set_host_reserve(&self, bytes: u64) {
        self.store.set_host_reserve(bytes);
    }

    pub fn store(&self) -> Arc<KvStore> {
        Arc::clone(&self.store)
    }

    pub fn set_device_floor(&self, bytes: u64) {
        self.store.set_device_floor(bytes);
    }

    pub fn device_reserved_bytes(&self) -> u64 {
        self.store.device_reserved_bytes()
    }

    pub fn host_cap(&self) -> u64 {
        self.store.host_cap()
    }
}

impl GpuEngine {
    pub(super) fn kv_blocks(&self) -> Option<&Arc<BlockServe>> {
        self.vmm.as_ref().and_then(|v| v.blocks.as_ref())
    }

    /// `/metrics`: the store's counters, engine-lock-free.
    pub fn kv_store_handle(&self) -> Option<Arc<BlockServe>> {
        self.kv_blocks().cloned()
    }

    /// Publish slot `b`'s boundary at `p_a` into the store: acquire its blocks (a boundary
    /// the pool evicts on a full store makes room), scatter the new ones from the rings and
    /// tracks. The new blocks seal once the scatter's event retires ([`Self::blocks_reap_seals`]):
    /// until then only the engine stream, which orders every later gather behind the scatter,
    /// can read them. Returns the handle and the bytes the new blocks added.
    pub(super) fn blocks_publish(&self, b: usize, p_a: u32) -> Result<(u64, u64)> {
        let t0 = std::time::Instant::now();
        let mut sync_us = 0u64;
        self.blocks_reap_seals();
        self.state_materialize(b, None)?;
        let (v, kv) = self.vmm_prefix().expect("block publish without prefix vmm");
        let bs = v.blocks.as_ref().expect("block publish without store");
        let toks = &self.seq_tokens[b][..p_a as usize];
        let chain = chain_hashes::<rustc_hash::FxHasher>(toks, ROWS_PER_BLOCK, SEED_CHAIN);
        let check = chain_hashes::<std::collections::hash_map::DefaultHasher>(
            toks,
            ROWS_PER_BLOCK,
            SEED_CHECK,
        );
        let plan = bs.plan(p_a, kv.geometry().window);
        let spill_deadline = t0 + SPILL_WAIT;
        // On an error return `fills` drops first (the blocks fail), `deps` after (released).
        let mut deps = Deps::new(&bs.store, plan.len());
        let mut fills: Vec<FillToken> = Vec::new();
        let mut descs: [Vec<Desc>; RANGES] = Default::default();
        let mut new_bytes = 0u64;
        for (class, row0, key_index) in plan {
            let ci = bs.class_index(class);
            let key = crate::memory::kv_store::BlockKey {
                namespace: 0,
                chain: chain[key_index],
                check: check[key_index],
            };
            let acquired = loop {
                match bs.store.acquire(ci, key) {
                    Ok(a) => break a,
                    // Retired spills free device slots (queued ones are waited for, up to one
                    // deadline per publish); else the pool gives up a boundary, unless enough
                    // spills are queued already (past the deadline the publish is refused rather
                    // than queueing a burst on the copy engine the engine's readbacks share).
                    Err(RuntimeError::Oom(_))
                        if bs.store.reap_wait(
                            spill_deadline.saturating_duration_since(std::time::Instant::now()),
                        ) || (bs.store.pending_spill_bytes()
                            < crate::memory::vmm::SPILL_AHEAD_INFLIGHT_BYTES
                            && kv.reclaim_one()) =>
                    {
                        continue
                    }
                    Err(e) => return Err(e),
                }
            };
            deps.push(class, row0, acquired.id);
            if let Some(token) = acquired.fill {
                fills.push(token);
                new_bytes += bs.store.class_spec(ci).block_bytes();
                descs[class.index()].push(self.block_desc(b, class, row0, acquired.va));
            }
        }
        let new_blocks = fills.len() as u64;
        if !fills.is_empty() {
            let queued = self
                .run_blocks(&descs, true)
                .and_then(|()| self.blocks_seal_after(&fills));
            match queued {
                Ok(()) => {
                    for token in fills {
                        token.defer();
                    }
                }
                Err(e) => {
                    // Drain even a partially submitted copy: the blocks fail and are released
                    // after the engine stream is done with them.
                    let ts = std::time::Instant::now();
                    let _ = self.be.stream_synchronize(&self.stream);
                    sync_us = ts.elapsed().as_micros() as u64;
                    drop(fills);
                    drop(deps);
                    bs.stats.lock().publish_sync_us += sync_us;
                    return Err(e);
                }
            }
        }
        let handle = deps.into_handle();
        let id = bs.next_handle.fetch_add(1, Ordering::Relaxed);
        {
            let mut st = bs.stats.lock();
            st.publishes += 1;
            st.published_blocks += handle.len() as u64;
            st.published_new_blocks += new_blocks;
            let us = t0.elapsed().as_micros() as u64;
            st.publish_us += us;
            st.publish_max_us = st.publish_max_us.max(us);
            st.publish_sync_us += sync_us;
            if new_blocks > 0 {
                st.seals_deferred += 1;
            }
        }
        bs.handles.lock().insert(id, handle);
        Ok((id, new_bytes))
    }

    /// Record the event the fills of a just-queued scatter seal behind.
    fn blocks_seal_after(&self, fills: &[FillToken]) -> Result<()> {
        let bs = self.kv_blocks().expect("seal without store");
        let done = match bs.spare_events.lock().pop() {
            Some(e) => e,
            None => self.be.event_create(false)?,
        };
        if let Err(e) = self.be.event_record(&done, &self.stream) {
            bs.spare_events.lock().push(done);
            return Err(e);
        }
        bs.seals.lock().push(PendingSeal {
            done,
            ids: fills.iter().map(FillToken::id).collect(),
            holds: Vec::new(),
        });
        Ok(())
    }

    /// Drop `hold` once the engine stream passes this point (its reads were queued there).
    fn blocks_release_after(&self, hold: Hold) {
        let bs = self.kv_blocks().expect("release without store");
        let done = match bs.spare_events.lock().pop() {
            Some(e) => Ok(e),
            None => self.be.event_create(false),
        };
        let recorded = done.and_then(|done| match self.be.event_record(&done, &self.stream) {
            Ok(()) => Ok(done),
            Err(e) => {
                bs.spare_events.lock().push(done);
                Err(e)
            }
        });
        match recorded {
            Ok(done) => bs.seals.lock().push(PendingSeal {
                done,
                ids: Vec::new(),
                holds: vec![hold],
            }),
            Err(_) => {
                // Completion unproven: drain, or keep the block pinned.
                if self.be.stream_synchronize(&self.stream).is_ok() {
                    drop(hold);
                } else {
                    hold.abandon();
                }
            }
        }
    }

    /// Seal the deferred fills whose scatter retired (in queue order: one stream). A failed
    /// query fails the remaining fills (their bytes are never served; the slots stay pinned as
    /// `Failed` blocks the next acquire refills).
    pub(crate) fn blocks_reap_seals(&self) {
        let Some(bs) = self.kv_blocks() else {
            return;
        };
        let finished: Vec<(PendingSeal, bool)> = {
            let mut seals = bs.seals.lock();
            let mut out = Vec::new();
            while let Some(p) = seals.first() {
                let ok = match self.be.event_query(&p.done) {
                    Ok(true) => true,
                    Ok(false) => break,
                    Err(e) => {
                        tracing::warn!(error = %e, "kv store: copy event failed");
                        false
                    }
                };
                out.push((seals.remove(0), ok));
            }
            out
        };
        for (p, ok) in finished {
            if ok {
                bs.store.seal_ids(&p.ids);
                drop(p.holds);
            } else {
                // Never served; the blocks a failed copy may still touch stay pinned.
                bs.store.fail_ids(&p.ids);
                for hold in p.holds {
                    hold.abandon();
                }
            }
            bs.spare_events.lock().push(p.done);
        }
    }

    /// Gather the boundary `handle` into slot `b`'s rings and tracks. Inside a launch window
    /// the copy is drained by the step and the blocks stay held until then.
    pub(super) fn blocks_gather(&self, b: usize, handle: u64) -> Result<()> {
        let t0 = std::time::Instant::now();
        self.blocks_reap_seals();
        let (v, _) = self.vmm_prefix().expect("block gather without prefix vmm");
        let bs = v.blocks.as_ref().expect("block gather without store");
        let plan = bs
            .handles
            .lock()
            .get(&handle)
            .cloned()
            .ok_or_else(|| RuntimeError::Rejected("kv store: boundary vanished".into()))?;
        let ids: Vec<BlockId> = plan.iter().map(|&(_, _, id)| id).collect();
        let mut srcs = Vec::with_capacity(ids.len());
        let hold = bs.store.hold_sources(ids, &mut srcs)?;
        let mut fenced = 0u64;
        // The recurrent state a first packed prefill reads from the block itself.
        let defer = bs
            .state_defer
            .get()
            .filter(|d| !d.is_empty() && bs.kernel.is_some());
        let mut deferred: Option<(BlockId, u64)> = None;
        let copied = (|| {
            let mut descs: [Vec<Desc>; RANGES] = Default::default();
            for (&(class, row0, id), src) in plan.iter().zip(&srcs) {
                // The kernel reads a host-resident block in place (device-mapped tier) and a
                // restoring one behind its copy fence; the memcpy fallback needs it on the
                // device.
                let va = if bs.kernel.is_some() {
                    if let Some(fence) = src.fence {
                        self.be.fence_wait_on(fence, &self.stream)?;
                        fenced += 1;
                    }
                    src.va
                } else {
                    bs.store.ensure_device(id)?
                };
                let range = if class == Class::State && defer.is_some() && !src.from_host {
                    deferred = Some((id, va));
                    STATE_REST
                } else {
                    class.index()
                };
                descs[range].push(self.block_desc(b, class, row0, va));
            }
            self.run_blocks(&descs, false)
        })();
        // Inside a launch window a successful gather is drained by the step
        // (`flush_deferred_attach` drops the holds); outside one the holds retire behind an
        // event (consumers read the slot in stream order); a failure drains here.
        let mut sync_us = 0u64;
        let copied = if copied.is_ok() {
            copied
        } else {
            let ts = std::time::Instant::now();
            let r = copied.and(self.be.stream_synchronize(&self.stream));
            sync_us = ts.elapsed().as_micros() as u64;
            r
        };
        if let (Ok(()), Some((id, va)), Some(left)) = (&copied, deferred, defer) {
            let hold = bs.store.hold(vec![id]);
            let old = bs.pending_state.lock()[b].replace(PendingState {
                hold,
                va,
                left: left.clone(),
            });
            if let Some(old) = old {
                self.blocks_release_after(old.hold);
            }
        }
        match copied {
            Ok(()) if self.window_open => bs.held.lock()[b].push(hold),
            Ok(()) => self.blocks_release_after(hold),
            Err(_) => drop(hold),
        }
        copied?;
        let mut st = bs.stats.lock();
        st.gathers += 1;
        let us = t0.elapsed().as_micros() as u64;
        st.gather_us += us;
        st.gather_max_us = st.gather_max_us.max(us);
        st.gather_sync_us += sync_us;
        st.gathered_blocks += plan.len() as u64;
        st.gather_fenced_blocks += fenced;
        Ok(())
    }

    /// Whether the boundary `handle` names still has blocks restoring from the host tier.
    pub(super) fn blocks_restoring(&self, handle: u64) -> bool {
        let Some(bs) = self.kv_blocks() else {
            return false;
        };
        let Some(plan) = bs.handles.lock().get(&handle).cloned() else {
            return false;
        };
        let ids: Vec<BlockId> = plan.iter().map(|&(_, _, id)| id).collect();
        bs.store.restoring_any(&ids)
    }

    /// A slot's rings and tracks are about to be released or unmapped (`begin_seq` retires its
    /// VMM pages on the host, outside stream order): the store copies still queued against them
    /// (a deferred publish scatter, a gather's held sources) complete first.
    pub(super) fn blocks_quiesce(&self) -> Result<()> {
        let Some(bs) = self.kv_blocks() else {
            return Ok(());
        };
        self.blocks_reap_seals();
        let ts = std::time::Instant::now();
        // Seals are recorded in engine-stream order: the newest covers every earlier copy, and
        // not the steps queued after it.
        let r = match bs.seals.lock().last() {
            Some(p) => self.be.event_synchronize(&p.done),
            None => return Ok(()),
        };
        bs.stats.lock().publish_sync_us += ts.elapsed().as_micros() as u64;
        self.blocks_reap_seals();
        r
    }

    pub(super) fn blocks_unhold(&self, b: usize) {
        if let Some(bs) = self.kv_blocks() {
            bs.unhold_slot(b);
        }
    }

    /// The descriptor of one block of slot `b` at `va` (`class`, rows from `row0`).
    fn block_desc(&self, b: usize, class: Class, row0: u32, va: u64) -> Desc {
        let row = match class {
            Class::Window => {
                let (v, _) = self.vmm_prefix().expect("window block without prefix vmm");
                (u64::from(row0) % v.ring) as u32
            }
            Class::Tail | Class::Scale => row0,
            Class::State => 0,
        };
        Desc {
            block: va,
            row,
            slot: b as u32,
        }
    }

    /// The runs of each class between a slot's cache tensors and one block, in the block's
    /// byte order: window = `[slide layer][K,V][head][32 rows][hd]` then
    /// `[slide layer][Ks,Vs][head][32] f32`; tail = `[track][head][32 rows][hd]` then
    /// `[full layer][Ks,Vs][head][32] f32`; scale = `[full layer][Ks,Vs][head][block_rows] f32`;
    /// state = every state tensor's slot bytes, in `STATE_CHUNK` pieces.
    fn block_layouts(&self) -> Vec<Vec<Layout>> {
        let (v, kv) = self.vmm_prefix().expect("block layouts without prefix vmm");
        let g = kv.geometry();
        let rows = u64::from(ROWS_PER_BLOCK);
        let mut out: Vec<Vec<Layout>> = vec![Vec::new(); RANGES];
        let push =
            |list: &mut Vec<Layout>, dev: u64, slot_stride: u64, row_bytes: u64, bytes: u64| {
                let block_off = list
                    .last()
                    .map_or(0, |l: &Layout| u64::from(l.block_off) + u64::from(l.bytes));
                let fit =
                    |v: u64| u32::try_from(v).expect("kv store: a block layout exceeds 4 GiB");
                list.push(Layout {
                    dev,
                    slot_stride,
                    row_bytes: fit(row_bytes),
                    block_off: fit(block_off),
                    bytes: fit(bytes),
                    pad: 0,
                });
            };
        {
            let list = &mut out[Class::Window.index()];
            let hd_b = u64::from(g.hd_slide * g.elem_slide);
            let ring = v.ring;
            for &(ik, iv, stride) in &v.slide {
                for idx in [ik, iv] {
                    for h in 0..u64::from(g.kvh_slide) {
                        push(
                            list,
                            self.devp[idx].base + h * ring * hd_b,
                            stride,
                            hd_b,
                            rows * hd_b,
                        );
                    }
                }
            }
            let scale_stride = u64::from(g.kvh_slide) * ring * 4;
            for &(sk, sv) in &v.slide_scale {
                for idx in [sk, sv] {
                    for h in 0..u64::from(g.kvh_slide) {
                        push(
                            list,
                            self.devp[idx].base + h * ring * 4,
                            scale_stride,
                            4,
                            rows * 4,
                        );
                    }
                }
            }
        }
        let ctx = u64::from(g.max_ctx);
        {
            let list = &mut out[Class::Tail.index()];
            let row_bytes = u64::from(g.hd_full * g.elem);
            let stride = u64::from(g.kvh_full) * ctx * row_bytes;
            for &(index, _, _) in &v.tensor_tracks {
                for h in 0..u64::from(g.kvh_full) {
                    push(
                        list,
                        self.devp[index].base + h * ctx * row_bytes,
                        stride,
                        row_bytes,
                        rows * row_bytes,
                    );
                }
            }
            for &(sk, sv) in &v.full_scale {
                for idx in [sk, sv] {
                    for h in 0..u64::from(g.kvh_full) {
                        push(
                            list,
                            self.devp[idx].base + h * ctx * 4,
                            u64::from(g.kvh_full) * ctx * 4,
                            4,
                            rows * 4,
                        );
                    }
                }
            }
        }
        {
            let list = &mut out[Class::Scale.index()];
            let n = u64::from(kv.block_rows());
            for &(sk, sv) in &v.full_scale {
                for idx in [sk, sv] {
                    for h in 0..u64::from(g.kvh_full) {
                        push(
                            list,
                            self.devp[idx].base + h * ctx * 4,
                            u64::from(g.kvh_full) * ctx * 4,
                            4,
                            n * 4,
                        );
                    }
                }
            }
        }
        if let Some(bs) = v.blocks.as_ref() {
            let defer = bs.state_defer.get().map_or(&[][..], Vec::as_slice);
            for &(index, per_slot) in &bs.state_tensors {
                let mut off = 0;
                while off < per_slot {
                    let bytes = STATE_CHUNK.min(per_slot - off);
                    push(
                        &mut out[Class::State.index()],
                        self.devp[index].base + off,
                        per_slot,
                        0,
                        bytes,
                    );
                    let entry = *out[Class::State.index()].last().expect("just pushed");
                    if !defer.iter().any(|&(i, _)| i == index) {
                        out[STATE_REST].push(entry);
                    }
                    off += bytes;
                }
            }
            if let Some(init) = self.recurrent_initstate.filter(|_| !defer.is_empty()) {
                // Request `i` (descriptor slot) of the scratch from its source state.
                let stride = crate::device::cuda::qwen_gdn::STATE_BYTES;
                let list = &mut out[STATE_INIT];
                let mut off = 0;
                while off < stride {
                    let bytes = STATE_CHUNK.min(stride - off);
                    push(list, self.devp[init].base + off, stride, 0, bytes);
                    off += bytes;
                }
            }
        }
        out
    }

    /// Recurrent models: the chunk from `c0` must end exactly at the prompt-end boundary so
    /// its state can be captured there (`State` blocks hold the state after the boundary's
    /// rows, nothing else). Returns the chunk cap to use; `cap` when no alignment applies
    /// (no state class, boundary passed, or fewer rows left to it than the smallest bucket).
    pub(super) fn state_capture_cap(
        &self,
        b: usize,
        prompt: &[u32],
        c0: usize,
        cap: usize,
    ) -> usize {
        let cut = self.state_capture_cut(b, prompt, c0);
        let smallest = self.prefill.first().map_or(usize::MAX, |p| p.t as usize);
        if cut < smallest {
            return cap;
        }
        cap.min(cut)
    }

    /// Rows from `c0` to the next capture boundary of `prompt` on a recurrent model whose state
    /// the store captures; `usize::MAX` when no cut applies (no state class, attach off for the
    /// slot, prompt-end boundary passed). Packed chunks take `min` with it so the chunk ends
    /// exactly at the boundary and [`Self::state_capture_publish`] can capture there.
    pub fn state_capture_cut(&self, b: usize, prompt: &[u32], c0: usize) -> usize {
        self.state_capture_stop(b, prompt, c0)
            .map_or(usize::MAX, |(rows, _)| rows as usize - c0)
    }

    /// The next row past `c0` where slot `b`'s recurrent state must be captured, the only rows
    /// another request can attach to: a whole-block checkpoint a request waits on
    /// (`VmmKv::inflight_prefix`), the shared-prefix end (`VmmKv::share_rows`: a system prompt
    /// another prompt also sent), then the prompt-end boundary.
    fn state_capture_stop(
        &self,
        b: usize,
        prompt: &[u32],
        c0: usize,
    ) -> Option<(u32, crate::memory::vmm::Kind)> {
        use crate::memory::vmm::Kind;
        self.kv_blocks().filter(|bs| bs.captures_state())?;
        if self.prefix_attach_off[b] {
            return None;
        }
        let p_a = super::prefix::prompt_end_boundary(prompt.len() as u32);
        if c0 >= p_a as usize {
            return None;
        }
        let Some((_, kv)) = self.vmm_prefix() else {
            return Some((p_a, Kind::PROMPT_END));
        };
        let mut stop = (p_a, Kind::PROMPT_END);
        let share = kv.share_rows(b);
        if share as usize > c0 && share < stop.0 {
            stop = (share, Kind::SHARED_END);
        }
        let br = kv.block_rows();
        let mut p = (c0 as u32 / br + 1) * br;
        while p < stop.0 {
            if kv.checkpoint_awaited(prompt, p) {
                stop = (p, Kind::CHECKPOINT);
                break;
            }
            p += br;
        }
        Some(stop)
    }

    /// After a chunk that ended exactly at a capture boundary of a recurrent model
    /// ([`Self::state_capture_stop`]): publish it now, while the state tensors hold the state at
    /// that row. The slot's row history must already cover the chunk.
    pub(super) fn state_capture_publish(&mut self, b: usize, prompt_len: usize) {
        use crate::memory::vmm::Kind;
        if !self.kv_blocks().is_some_and(|bs| bs.captures_state()) || self.prefix_attach_off[b] {
            return;
        }
        let pos = self.pos[b];
        let Some((_, kv)) = self.vmm_prefix() else {
            return;
        };
        if pos == 0 || self.seq_tokens[b].len() < pos as usize {
            return;
        }
        let p_a = super::prefix::prompt_end_boundary(prompt_len as u32);
        let kind = if pos == p_a {
            Kind::PROMPT_END
        } else if pos > p_a {
            return;
        } else if pos == kv.share_rows(b) {
            Kind::SHARED_END
        } else if pos % kv.block_rows() == 0 && kv.checkpoint_awaited(&self.seq_tokens[b], pos) {
            Kind::CHECKPOINT
        } else {
            return;
        };
        self.seq_tokens[b].truncate(pos as usize);
        self.publish_boundary(b, pos, kind);
    }

    /// The state tensors a packed GDN prefill gathers into its initial-state scratch (the
    /// declared state operands of the chain's external GDN segments), when the chain packs; their
    /// offsets in a state block. Set once, before the layouts are built.
    fn init_state_defer(&self) {
        let Some(bs) = self.kv_blocks() else {
            return;
        };
        bs.state_defer.get_or_init(|| {
            let packs = self.qwen_prefill.as_ref().is_some_and(|g| g.varlen())
                && self.recurrent_initstate.is_some();
            if !packs || bs.state.is_none() {
                return Vec::new();
            }
            let operands: Vec<usize> = self
                .prefill
                .iter()
                .flat_map(|p| p.qwen_segments.iter().flatten())
                .map(|inst| inst.t[6] as usize)
                .collect();
            let mut off = 0u64;
            let mut out = Vec::new();
            for &(index, per_slot) in &bs.state_tensors {
                if per_slot == crate::device::cuda::qwen_gdn::STATE_BYTES
                    && operands.contains(&index)
                {
                    out.push((index, off));
                }
                off += per_slot;
            }
            out
        });
    }

    /// Where slot `b`'s first packed prefill reads `tensor`'s initial state from: the held
    /// block, when the attach left it there.
    pub(super) fn state_source(&self, b: usize, tensor: usize) -> Option<u64> {
        let bs = self.kv_blocks()?;
        let pending = bs.pending_state.lock();
        let p = pending[b].as_ref()?;
        p.left
            .iter()
            .find(|&&(i, _)| i == tensor)
            .map(|&(_, off)| p.va + off)
    }

    /// Slot `b`'s first packed prefill read `tensor` from the block (queued on the engine
    /// stream): the slot holds its state from here on.
    pub(super) fn state_consumed(&self, b: usize, tensor: usize) {
        let Some(bs) = self.kv_blocks() else {
            return;
        };
        let done = {
            let mut pending = bs.pending_state.lock();
            let Some(p) = pending[b].as_mut() else {
                return;
            };
            p.left.retain(|&(i, _)| i != tensor);
            if p.left.is_empty() {
                pending[b].take()
            } else {
                None
            }
        };
        if let Some(p) = done {
            self.blocks_release_after(p.hold);
        }
    }

    /// Copy the state tensors slot `b` still reads from its attached block (`tensor`, or every
    /// one) into the slot: whatever reads the slot's state other than the packed prefill's
    /// gather (an unpacked chunk, a rider step, a publish) reads it there.
    pub(crate) fn state_materialize(&self, b: usize, tensor: Option<usize>) -> Result<()> {
        let Some(bs) = self.kv_blocks() else {
            return Ok(());
        };
        let mut pending = bs.pending_state.lock();
        let Some(p) = pending[b].as_mut() else {
            return Ok(());
        };
        for &(index, off) in &p.left {
            if tensor.is_some_and(|t| t != index) {
                continue;
            }
            let per_slot = bs
                .state_tensors
                .iter()
                .find(|&&(i, _)| i == index)
                .map(|&(_, bytes)| bytes)
                .expect("deferred tensor is a state tensor");
            self.be.memcpy_dtod_async(
                self.devp[index].base + b as u64 * per_slot,
                p.va + off,
                per_slot,
                &self.stream,
            )?;
        }
        p.left.retain(|&(i, _)| tensor.is_some_and(|t| t != i));
        let done = if p.left.is_empty() {
            pending[b].take()
        } else {
            None
        };
        drop(pending);
        if let Some(p) = done {
            self.blocks_release_after(p.hold);
        }
        Ok(())
    }

    /// Slot `b` starts a new sequence: its attached block is no longer read.
    pub(crate) fn state_drop(&self, b: usize) {
        let Some(bs) = self.kv_blocks() else {
            return;
        };
        let p = bs.pending_state.lock()[b].take();
        if let Some(p) = p {
            self.blocks_release_after(p.hold);
        }
    }

    /// The packed prefill's initial-state gather when a scanned request reads its state from
    /// an attached block: request `i` of the contiguous scratch from `srcs[i]` (a block or a
    /// slot state), one launch.
    pub(super) fn state_init_gather(&self, srcs: &[u64]) -> Result<()> {
        let mut descs: [Vec<Desc>; RANGES] = Default::default();
        descs[STATE_INIT] = srcs
            .iter()
            .enumerate()
            .map(|(i, &block)| Desc {
                block,
                row: 0,
                slot: i as u32,
            })
            .collect();
        self.run_blocks(&descs, false)
    }

    /// Run the descriptors of each class on the engine stream: one `plow_kv_blockcopy` launch
    /// per class and `DESCS` descriptors (the driver copies the parameters at launch, so nothing
    /// outlives the call), or the memcpy loop without the kernel. Drained by the caller.
    fn run_blocks(&self, descs: &[Vec<Desc>; RANGES], to_block: bool) -> Result<()> {
        let bs = self.kv_blocks().expect("block copy without store");
        let layouts = match bs.layouts.get() {
            Some(l) => l,
            None => {
                self.init_state_defer();
                let lists = self.block_layouts();
                let mut host = Vec::new();
                let mut ranges = [(0u32, 0u32); RANGES];
                for (c, list) in lists.into_iter().enumerate() {
                    ranges[c] = (host.len() as u32, list.len() as u32);
                    host.extend(list);
                }
                let dev = match bs.kernel {
                    Some(_) if !host.is_empty() => {
                        let bytes = std::mem::size_of_val(host.as_slice());
                        let dev = TableDev::alloc(&self.be, bytes as u64)?;
                        // SAFETY: `Layout` is plain `repr(C)` data, `bytes` long.
                        let raw = unsafe {
                            std::slice::from_raw_parts(host.as_ptr() as *const u8, bytes)
                        };
                        self.be.memcpy_htod(dev.va, raw)?;
                        Some(dev)
                    }
                    _ => None,
                };
                let _ = bs.layouts.set(Layouts { dev, host, ranges });
                bs.layouts.get().expect("just set")
            }
        };
        let mut segments = 0u64;
        let mut launches = 0u64;
        for (range, d) in descs.iter().enumerate() {
            let (first, count) = layouts.ranges[range];
            if d.is_empty() || count == 0 {
                continue;
            }
            segments += d.len() as u64 * u64::from(count);
            match (bs.kernel, layouts.dev.as_ref()) {
                (Some(f), Some(dev)) => {
                    let mut args = bs.args.lock();
                    for chunk in d.chunks(DESCS) {
                        args.layout =
                            dev.va + u64::from(first) * std::mem::size_of::<Layout>() as u64;
                        args.to_block = u32::from(to_block);
                        args.n = chunk.len() as u32;
                        args.d[..chunk.len()].copy_from_slice(chunk);
                        let mut params = [&mut **args as *mut CopyArgs as *mut c_void];
                        self.be.launch_kernel_grid(
                            f,
                            [count, chunk.len() as u32, 1],
                            SEGCOPY_BLOCK,
                            0,
                            &mut params,
                            Some(&self.stream),
                        )?;
                        launches += 1;
                    }
                }
                _ => {
                    let runs = &layouts.host[first as usize..(first + count) as usize];
                    for desc in d {
                        for l in runs {
                            let dev = l.dev
                                + u64::from(desc.slot) * l.slot_stride
                                + u64::from(desc.row) * u64::from(l.row_bytes);
                            let blk = desc.block + u64::from(l.block_off);
                            let (dst, src) = if to_block { (blk, dev) } else { (dev, blk) };
                            self.be.memcpy_dtod_async(
                                dst,
                                src,
                                u64::from(l.bytes),
                                &self.stream,
                            )?;
                        }
                    }
                    bs.stats.lock().memcpy_segments += d.len() as u64 * u64::from(count);
                }
            }
        }
        let mut st = bs.stats.lock();
        st.segments += segments;
        st.launches += launches;
        Ok(())
    }
}

/// The blocks of a boundary at `p_a` rows (a multiple of 32): `(class, first row, key index)`
/// where key index `i` names the chained hash of `tokens[..(i + 1) * 32]`. `classes` =
/// `[window, tail, scale]` present.
fn plan_blocks(
    classes: [bool; 4],
    block_rows: u32,
    p_a: u32,
    window: u32,
) -> Vec<(Class, u32, usize)> {
    let key = |end: u32| (end / ROWS_PER_BLOCK - 1) as usize;
    let mut out = Vec::new();
    if classes[0] {
        let w = window.min(p_a) / ROWS_PER_BLOCK * ROWS_PER_BLOCK;
        for r in (p_a - w..p_a).step_by(ROWS_PER_BLOCK as usize) {
            out.push((Class::Window, r, key(r + ROWS_PER_BLOCK)));
        }
    }
    if classes[1] {
        // Every full-layer row of the boundary (the store is the only storage of them).
        for r in (0..p_a).step_by(ROWS_PER_BLOCK as usize) {
            out.push((Class::Tail, r, key(r + ROWS_PER_BLOCK)));
        }
    }
    if classes[2] {
        for k in 0..p_a / block_rows {
            let r = k * block_rows;
            out.push((Class::Scale, r, key(r + block_rows)));
        }
    }
    if classes[3] && p_a > 0 {
        out.push((Class::State, p_a, key(p_a)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_boundary_is_its_window_and_every_full_layer_row() {
        let p = plan_blocks([true, true, false, false], 4096, 4096 + 96, 1024);
        let window: Vec<u32> = p
            .iter()
            .filter(|x| x.0 == Class::Window)
            .map(|x| x.1)
            .collect();
        assert_eq!(window.len(), 32);
        assert_eq!((window[0], window[31]), (4096 + 96 - 1024, 4096 + 64));
        let prefix: Vec<u32> = p
            .iter()
            .filter(|x| x.0 == Class::Tail)
            .map(|x| x.1)
            .collect();
        assert_eq!(prefix.len(), (4096 + 96) / 32);
        assert_eq!((prefix[0], prefix[130]), (0, 4160));
        // The key index of a 32-row block is the chain entry ending at its last row.
        assert_eq!(p[0].2, (4096 + 96 - 1024 + 32) / 32 - 1);
        assert_eq!(p.last().unwrap().2, (4096 + 96) / 32 - 1);
    }

    #[test]
    fn a_short_prompt_has_a_short_window_and_no_whole_blocks() {
        let p = plan_blocks([true, true, false, false], 4096, 224, 1024);
        let window = p.iter().filter(|x| x.0 == Class::Window).count();
        let tail = p.iter().filter(|x| x.0 == Class::Tail).count();
        assert_eq!((window, tail), (7, 7));
        let p = plan_blocks([true, false, false, false], 4096, 224, 1024);
        assert_eq!(p.len(), 7);
    }

    #[test]
    fn two_boundaries_of_one_conversation_share_their_common_window_blocks() {
        let a = plan_blocks([true, true, false, false], 4096, 2048, 1024);
        let b = plan_blocks([true, true, false, false], 4096, 2048 + 384, 1024);
        let keys = |p: &[(Class, u32, usize)]| -> Vec<(Class, usize)> {
            p.iter().map(|x| (x.0, x.2)).collect()
        };
        let (ka, kb) = (keys(&a), keys(&b));
        let shared = kb.iter().filter(|k| ka.contains(k)).count();
        // 1024-row windows 384 rows apart share 20 of 32 window blocks; the tail shares all
        // of a's 64 blocks (same partial block, same rows).
        assert_eq!(shared, 20 + 64);
    }

    #[test]
    fn a_recurrent_model_adds_one_state_block_keyed_at_the_boundary() {
        let p = plan_blocks([false, true, false, true], 4096, 4096 + 96, 1024);
        let state: Vec<(u32, usize)> = p
            .iter()
            .filter(|x| x.0 == Class::State)
            .map(|x| (x.1, x.2))
            .collect();
        assert_eq!(state, vec![(4096 + 96, (4096 + 96) / 32 - 1)]);
        assert!(plan_blocks([false, false, false, true], 4096, 0, 1024).is_empty());
    }

    #[test]
    fn copy_parameters_match_the_kernel_and_fit_the_launch_limit() {
        assert_eq!(std::mem::size_of::<Layout>(), 32);
        assert_eq!(std::mem::size_of::<Desc>(), 16);
        assert_eq!(std::mem::size_of::<CopyArgs>(), 16 + 16 * DESCS);
        // CUDA 12.1+ kernel parameter limit.
        assert!(std::mem::size_of::<CopyArgs>() <= 32764);
        assert!(SEGCOPY_PTX.contains(&format!(
            "plow_kv_blockcopy_param_0[{}]",
            std::mem::size_of::<CopyArgs>()
        )));
    }

    /// Device slots only: enough for the dependency set.
    struct SlotOps(AtomicU64);

    impl crate::memory::kv_store::StoreOps for SlotOps {
        fn alloc(&self, bytes: u64) -> Result<u64> {
            Ok(self.0.fetch_add(bytes, Ordering::SeqCst))
        }
        fn free(&self, _va: u64) {}
        fn host_alloc(&self, _bytes: u64) -> Result<u64> {
            unreachable!("no host tier")
        }
        fn host_free(&self, _ptr: u64, _bytes: u64) {}
        fn copy_dtoh_async(&self, _host: u64, _va: u64, _bytes: u64) -> Result<u64> {
            unreachable!("no host tier")
        }
        fn fence_done(&self, _fence: u64) -> Result<bool> {
            unreachable!("no copies")
        }
        fn copy_htod(&self, _va: u64, _host: u64, _bytes: u64) -> Result<()> {
            unreachable!("no host tier")
        }
        fn copy_htod_async(&self, _va: u64, _host: u64, _bytes: u64) -> Result<u64> {
            unreachable!("no host tier")
        }
        fn copy_dtod_batch(&self, _copies: &[(u64, u64, u64)]) -> Result<()> {
            unreachable!("no compaction")
        }
        fn free_bytes(&self) -> Option<u64> {
            None
        }
    }

    fn two_block_store() -> KvStore {
        KvStore::new(
            Arc::new(SlotOps(AtomicU64::new(1 << 20))),
            vec![ClassSpec {
                name: "window",
                row_bytes: 8,
                rows_per_block: 4,
            }],
            32,
            2 * 32,
            0,
        )
    }

    fn key(i: u64) -> crate::memory::kv_store::BlockKey {
        crate::memory::kv_store::BlockKey {
            namespace: 0,
            chain: i,
            check: i,
        }
    }

    /// A dependency set dropped before the boundary is recorded releases its blocks (the slot
    /// is reclaimable); `into_handle` keeps them for the boundary.
    #[test]
    fn deps_release_on_drop_and_into_handle_keeps_them() {
        let s = two_block_store();
        let a = {
            let mut deps = Deps::new(&s, 1);
            let a = s.acquire(0, key(1)).unwrap();
            a.fill.expect("new block").seal();
            deps.push(Class::Window, 0, a.id);
            a.id
        };
        s.acquire(0, key(2))
            .unwrap()
            .fill
            .expect("new block")
            .seal();
        assert!(s.acquire(0, key(3)).is_ok(), "a's slot was reclaimable");
        assert_eq!(s.state(a), None, "a was released and reclaimed");

        let s = two_block_store();
        let mut deps = Deps::new(&s, 1);
        let a = s.acquire(0, key(1)).unwrap();
        a.fill.expect("new block").seal();
        deps.push(Class::Window, 0, a.id);
        let handle = deps.into_handle();
        assert_eq!(handle.len(), 1);
        s.acquire(0, key(2))
            .unwrap()
            .fill
            .expect("new block")
            .seal();
        assert!(s.acquire(0, key(3)).is_err(), "both blocks referenced");
        let ids: Vec<BlockId> = handle.iter().map(|&(_, _, id)| id).collect();
        s.release(&ids);
        assert!(s.acquire(0, key(3)).is_ok());
    }
}
