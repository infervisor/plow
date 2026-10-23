//! The KV block store on the CUDA engine (`docs/arch/26-kv-block-store.md` Part B, B1).
//!
//! A boundary at `p_a` rows is a set of blocks: the sliding windows' last `window` rows and
//! their fp8 ring scales in 32-row **window** blocks, the full layers' partial-block rows (and
//! their fp8 scales) in 32-row **tail** blocks, and the full layers' whole-block fp8 scale
//! prefix in `block_rows`-row **scale** blocks (the scale tensors are flat per slot, so only a
//! copy survives slot reuse). Every block is keyed by the chained hash of the tokens up to its
//! last row, so two boundaries of one conversation, or two conversations on one system prompt,
//! share every block their rows have in common: a turn publishes only its new rows.
//!
//! The copies are segment tables (one 16-byte-aligned run per tensor and head per block)
//! executed by one `plow_kv_segcopy` launch on the engine stream (PTX embedded, driver-JIT at
//! load); a backend without the kernel runs the table as a memcpy loop. Model kernels see only
//! the slot's rings and tracks, exactly as with the contiguous snapshots this replaces.

use super::prefix::VmmServe;
use super::GpuEngine;
use crate::device::cuda::{CudaBackend, KernelFn, PinnedHost};
use crate::memory::kv_store::{chain_hashes, BlockId, ClassSpec, KvStore, StoreStats};
use crate::memory::vmm::{SnapBlocks, VmmOps};
use crate::{Result, RuntimeError};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const ROWS_PER_BLOCK: u32 = 32;
const CHUNK_BYTES: u64 = 64 << 20;
const SEED_CHAIN: u64 = 0x9e37_79b9_7f4a_7c15;
const SEED_CHECK: u64 = 0xc2b2_ae3d_27d4_eb4f;
const SEGCOPY_PTX: &str = include_str!("../../../../../runtime/nvidia/kv_segcopy.ptx");
const SEGCOPY_ENTRY: &str = "plow_kv_segcopy";
const SEGCOPY_BLOCK: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Window,
    Tail,
    Scale,
    /// Recurrent state (GDN, conv history) captured at the boundary row: one block per
    /// boundary, every `state.*` tensor's per-slot bytes back to back.
    State,
}

/// One segment of a gather/scatter table (`plow_kv_seg` in `kv_segcopy.cu`).
#[repr(C)]
#[derive(Clone, Copy)]
struct Seg {
    src: u64,
    dst: u64,
    bytes: u32,
    pad: u32,
}

/// A boundary's blocks: `(class, first row, block)` in the order the segments are built.
type Handle = Vec<(Class, u32, BlockId)>;

struct Table {
    host: PinnedHost,
    dev: u64,
    cap: usize,
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
}

pub struct BlockServe {
    be: Arc<CudaBackend>,
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
    tables: Mutex<Vec<Option<Table>>>,
    /// Transfer holds a launch-window gather keeps until the step's own drain
    /// ([`GpuEngine::flush_deferred_attach`]).
    held: Mutex<Vec<Vec<BlockId>>>,
    stats: Mutex<BlockServeStats>,
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
        let tail = class("tail", tail_row, ROWS_PER_BLOCK);
        let scale = class("scale", scale_row, block_rows);
        let state = class("state", state_row, 1);
        if specs.is_empty() {
            return Err(RuntimeError::Rejected(
                "kv store: the packet has no boundary rows to store".into(),
            ));
        }
        // Every segment is a whole run of 32 (or `block_rows`) rows of one head: 16-byte
        // vectors need those runs 16-byte sized.
        let vector_ok = [ROWS_PER_BLOCK as u64 * hd_slide, ROWS_PER_BLOCK as u64 * hd_full]
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
            be: Arc::clone(be),
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
            tables: Mutex::new((0..batch).map(|_| None).collect()),
            held: Mutex::new(vec![Vec::new(); batch]),
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
        let ids = std::mem::take(&mut self.held.lock()[b]);
        if !ids.is_empty() {
            self.store.unhold(&ids);
        }
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
        for t in self.tables.get_mut().iter_mut().flatten() {
            VmmOps::free(&*self.be, t.dev);
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

    fn restore(&self, handle: u64) -> Result<()> {
        let ids: Vec<BlockId> = match self.handles.lock().get(&handle) {
            Some(h) => h.iter().map(|&(_, _, id)| id).collect(),
            None => return Err(RuntimeError::Rejected("kv store: unknown boundary".into())),
        };
        for id in ids {
            self.store.ensure_device(id)?;
        }
        Ok(())
    }

    fn reap(&self) {
        self.store.reap();
    }

    fn shrink(&self, bytes: u64) -> u64 {
        self.store.shrink(bytes)
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
    /// tracks, seal them. Returns the handle and the bytes the new blocks added.
    pub(super) fn blocks_publish(&self, b: usize, p_a: u32) -> Result<(u64, u64)> {
        let (v, kv) = self.vmm_prefix().expect("block publish without prefix vmm");
        let bs = v.blocks.as_ref().expect("block publish without store");
        let toks = &self.seq_tokens[b][..p_a as usize];
        let chain = chain_hashes::<rustc_hash::FxHasher>(toks, ROWS_PER_BLOCK, SEED_CHAIN);
        let check =
            chain_hashes::<std::collections::hash_map::DefaultHasher>(toks, ROWS_PER_BLOCK, SEED_CHECK);
        let plan = bs.plan(p_a, kv.geometry().window);
        let mut handle: Handle = Vec::with_capacity(plan.len());
        let mut fills: Vec<BlockId> = Vec::new();
        let mut segs: Vec<Seg> = Vec::new();
        let mut new_bytes = 0u64;
        let abort = |handle: &Handle, fills: &[BlockId]| {
            for &id in fills {
                bs.store.fail(id);
            }
            let ids: Vec<BlockId> = handle.iter().map(|&(_, _, id)| id).collect();
            bs.store.release(&ids);
        };
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
                    // Retired spills free device slots; else the pool gives up a boundary.
                    Err(RuntimeError::Oom(_)) if {
                        bs.store.reap();
                        kv.reclaim_one()
                    } =>
                    {
                        continue
                    }
                    Err(e) => {
                        abort(&handle, &fills);
                        return Err(e);
                    }
                }
            };
            handle.push((class, row0, acquired.id));
            if acquired.fill {
                fills.push(acquired.id);
                new_bytes += bs.store.class_spec(ci).block_bytes();
                self.block_segments(b, class, row0, acquired.va, true, &mut segs);
            }
        }
        if !segs.is_empty() {
            // Drain even a partially submitted table: the blocks are sealed or released after
            // the engine stream is done with them.
            let copied = self.run_segments(b, &segs);
            let drained = self.be.stream_synchronize(&self.stream);
            if let Err(e) = copied.and(drained) {
                abort(&handle, &fills);
                return Err(e);
            }
        }
        for &id in &fills {
            bs.store.seal(id);
        }
        let id = bs.next_handle.fetch_add(1, Ordering::Relaxed);
        {
            let mut st = bs.stats.lock();
            st.publishes += 1;
            st.published_blocks += handle.len() as u64;
            st.published_new_blocks += fills.len() as u64;
        }
        bs.handles.lock().insert(id, handle);
        Ok((id, new_bytes))
    }

    /// Gather the boundary `handle` into slot `b`'s rings and tracks. Inside a launch window
    /// the copy is drained by the step and the blocks stay held until then.
    pub(super) fn blocks_gather(&self, b: usize, handle: u64) -> Result<()> {
        let (v, _) = self.vmm_prefix().expect("block gather without prefix vmm");
        let bs = v.blocks.as_ref().expect("block gather without store");
        let plan = bs
            .handles
            .lock()
            .get(&handle)
            .cloned()
            .ok_or_else(|| RuntimeError::Rejected("kv store: boundary vanished".into()))?;
        let ids: Vec<BlockId> = plan.iter().map(|&(_, _, id)| id).collect();
        bs.store.hold(&ids);
        let copied = (|| {
            let mut segs = Vec::new();
            for &(class, row0, id) in &plan {
                let va = bs.store.ensure_device(id)?;
                self.block_segments(b, class, row0, va, false, &mut segs);
            }
            self.run_segments(b, &segs)
        })();
        // Inside a launch window a successful gather is drained by the step
        // (`flush_deferred_attach` drops the holds); every other outcome drains here.
        let copied = if copied.is_ok() && self.window_open {
            copied
        } else {
            copied.and(self.be.stream_synchronize(&self.stream))
        };
        match copied {
            Ok(()) if self.window_open => bs.held.lock()[b].extend_from_slice(&ids),
            _ => bs.store.unhold(&ids),
        }
        copied?;
        let mut st = bs.stats.lock();
        st.gathers += 1;
        st.gathered_blocks += plan.len() as u64;
        Ok(())
    }

    pub(super) fn blocks_unhold(&self, b: usize) {
        if let Some(bs) = self.kv_blocks() {
            bs.unhold_slot(b);
        }
    }

    /// The segments between slot `b`'s cache tensors and one block (`class`, rows from
    /// `row0`) at `va`; `to_block` picks the direction. Layout per class, in this order:
    /// window = `[slide layer][K,V][head][32 rows][hd]` then `[slide layer][Ks,Vs][head][32] f32`;
    /// tail = `[track][head][32 rows][hd]` then `[full layer][Ks,Vs][head][32] f32`;
    /// scale = `[full layer][Ks,Vs][head][block_rows] f32`.
    fn block_segments(
        &self,
        b: usize,
        class: Class,
        row0: u32,
        va: u64,
        to_block: bool,
        out: &mut Vec<Seg>,
    ) {
        let (v, kv) = self.vmm_prefix().expect("block segments without prefix vmm");
        let g = kv.geometry();
        let mut off = va;
        let mut seg = |dev: u64, bytes: u64| {
            let (src, dst) = if to_block { (dev, off) } else { (off, dev) };
            out.push(Seg {
                src,
                dst,
                bytes: bytes as u32,
                pad: 0,
            });
            off += bytes;
        };
        let rows = u64::from(ROWS_PER_BLOCK);
        let r0 = u64::from(row0);
        match class {
            Class::Window => {
                let hd_b = u64::from(g.hd_slide * g.elem_slide);
                let ring = v.ring;
                let slot_row = r0 % ring;
                for &(ik, iv, stride) in &v.slide {
                    for idx in [ik, iv] {
                        let base = self.devp[idx].base + b as u64 * stride;
                        for h in 0..u64::from(g.kvh_slide) {
                            seg(base + (h * ring + slot_row) * hd_b, rows * hd_b);
                        }
                    }
                }
                for &(sk, sv) in &v.slide_scale {
                    for idx in [sk, sv] {
                        let base =
                            self.devp[idx].base + b as u64 * u64::from(g.kvh_slide) * ring * 4;
                        for h in 0..u64::from(g.kvh_slide) {
                            seg(base + (h * ring + slot_row) * 4, rows * 4);
                        }
                    }
                }
            }
            Class::Tail => {
                let row_bytes = u64::from(g.hd_full * g.elem);
                let ctx = u64::from(g.max_ctx);
                for &(index, _, _) in &v.tensor_tracks {
                    for h in 0..u64::from(g.kvh_full) {
                        let dev = self.devp[index].base
                            + ((b as u64 * u64::from(g.kvh_full) + h) * ctx + r0) * row_bytes;
                        seg(dev, rows * row_bytes);
                    }
                }
                for &(sk, sv) in &v.full_scale {
                    for idx in [sk, sv] {
                        let base = self.devp[idx].base + b as u64 * u64::from(g.kvh_full) * ctx * 4;
                        for h in 0..u64::from(g.kvh_full) {
                            seg(base + (h * ctx + r0) * 4, rows * 4);
                        }
                    }
                }
            }
            Class::Scale => {
                let ctx = u64::from(g.max_ctx);
                let n = u64::from(kv.block_rows());
                for &(sk, sv) in &v.full_scale {
                    for idx in [sk, sv] {
                        let base = self.devp[idx].base + b as u64 * u64::from(g.kvh_full) * ctx * 4;
                        for h in 0..u64::from(g.kvh_full) {
                            seg(base + (h * ctx + r0) * 4, n * 4);
                        }
                    }
                }
            }
            Class::State => {
                let bs = v.blocks.as_ref().expect("state class needs the store");
                for &(index, per_slot) in &bs.state_tensors {
                    seg(self.devp[index].base + b as u64 * per_slot, per_slot);
                }
            }
        }
    }

    /// Recurrent models: the chunk from `c0` must end exactly at the prompt-end boundary so
    /// its state can be captured there (`State` blocks hold the state after the boundary's
    /// rows, nothing else). Returns the chunk cap to use; `cap` when no alignment applies
    /// (no state class, boundary passed, or fewer rows left to it than the smallest bucket).
    pub(super) fn state_capture_cap(&self, b: usize, prompt: &[u32], c0: usize, cap: usize) -> usize {
        let Some(bs) = self.kv_blocks().filter(|bs| bs.captures_state()) else {
            return cap;
        };
        if self.prefix_attach_off[b] {
            return cap;
        }
        let p_a = super::prefix::prompt_end_boundary(prompt.len() as u32) as usize;
        let smallest = self.prefill.first().map_or(usize::MAX, |p| p.t as usize);
        if c0 >= p_a || p_a - c0 < smallest {
            return cap;
        }
        cap.min(p_a - c0)
    }

    /// After a chunk that ended exactly at the prompt-end boundary of a recurrent model:
    /// publish it now, while the state tensors hold the state at that row.
    pub(super) fn state_capture_publish(&mut self, b: usize, prompt: &[u32]) {
        if !self.kv_blocks().is_some_and(|bs| bs.captures_state()) || self.prefix_attach_off[b] {
            return;
        }
        let p_a = super::prefix::prompt_end_boundary(prompt.len() as u32);
        if p_a == 0 || self.pos[b] != p_a {
            return;
        }
        self.seq_tokens[b].clear();
        self.seq_tokens[b].extend_from_slice(&prompt[..p_a as usize]);
        self.publish_boundary(b, p_a, crate::memory::vmm::Kind::PROMPT_END);
    }

    /// Run a segment table on the engine stream: one kernel launch over slot `b`'s table
    /// buffer, or a memcpy loop without the kernel. Drained by the caller.
    fn run_segments(&self, b: usize, segs: &[Seg]) -> Result<()> {
        let bs = self.kv_blocks().expect("segments without store");
        {
            let mut st = bs.stats.lock();
            st.segments += segs.len() as u64;
        }
        let Some(f) = bs.kernel else {
            for s in segs {
                self.be
                    .memcpy_dtod_async(s.dst, s.src, u64::from(s.bytes), &self.stream)?;
            }
            bs.stats.lock().memcpy_segments += segs.len() as u64;
            return Ok(());
        };
        let bytes = std::mem::size_of_val(segs);
        let mut tables = bs.tables.lock();
        let table = match tables[b].as_mut() {
            Some(t) if t.cap >= segs.len() => t,
            _ => {
                let cap = segs.len().next_power_of_two().max(1024);
                let host = self.be.host_alloc_pinned(cap * std::mem::size_of::<Seg>())?;
                let dev = VmmOps::alloc(&*self.be, (cap * std::mem::size_of::<Seg>()) as u64)?;
                if let Some(old) = tables[b].replace(Table { host, dev, cap }) {
                    VmmOps::free(&*self.be, old.dev);
                }
                tables[b].as_mut().expect("just set")
            }
        };
        // SAFETY: `Seg` is plain `repr(C)` data and the pinned buffer holds `cap >= len` of it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                segs.as_ptr() as *const u8,
                table.host.as_mut_slice().as_mut_ptr(),
                bytes,
            );
        }
        VmmOps::copy_htod(&*self.be, table.dev, table.host.as_ptr() as u64, bytes as u64)?;
        let mut table_ptr = table.dev;
        let mut n = segs.len() as u32;
        let mut params = [
            &mut table_ptr as *mut u64 as *mut c_void,
            &mut n as *mut u32 as *mut c_void,
        ];
        self.be
            .launch_kernel(f, n, SEGCOPY_BLOCK, 0, &mut params, Some(&self.stream))?;
        bs.stats.lock().launches += 1;
        Ok(())
    }
}

/// The blocks of a boundary at `p_a` rows (a multiple of 32): `(class, first row, key index)`
/// where key index `i` names the chained hash of `tokens[..(i + 1) * 32]`. `classes` =
/// `[window, tail, scale]` present.
fn plan_blocks(classes: [bool; 4], block_rows: u32, p_a: u32, window: u32) -> Vec<(Class, u32, usize)> {
    let key = |end: u32| (end / ROWS_PER_BLOCK - 1) as usize;
    let mut out = Vec::new();
    if classes[0] {
        let w = window.min(p_a) / ROWS_PER_BLOCK * ROWS_PER_BLOCK;
        for r in (p_a - w..p_a).step_by(ROWS_PER_BLOCK as usize) {
            out.push((Class::Window, r, key(r + ROWS_PER_BLOCK)));
        }
    }
    if classes[1] {
        let r0 = p_a - p_a % block_rows;
        for r in (r0..p_a).step_by(ROWS_PER_BLOCK as usize) {
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
    fn a_boundary_is_its_window_its_partial_block_and_its_whole_block_scales() {
        let p = plan_blocks([true, true, true, false], 4096, 4096 + 96, 1024);
        let window: Vec<u32> = p.iter().filter(|x| x.0 == Class::Window).map(|x| x.1).collect();
        assert_eq!(window.len(), 32);
        assert_eq!((window[0], window[31]), (4096 + 96 - 1024, 4096 + 64));
        let tail: Vec<u32> = p.iter().filter(|x| x.0 == Class::Tail).map(|x| x.1).collect();
        assert_eq!(tail, vec![4096, 4128, 4160]);
        let scale: Vec<(u32, usize)> = p
            .iter()
            .filter(|x| x.0 == Class::Scale)
            .map(|x| (x.1, x.2))
            .collect();
        assert_eq!(scale, vec![(0, 127)], "one whole block, keyed at its last 32 rows");
        // The key index of a 32-row block is the chain entry ending at its last row.
        assert_eq!(p[0].2, (4096 + 96 - 1024 + 32) / 32 - 1);
    }

    #[test]
    fn a_short_prompt_has_a_short_window_and_no_whole_blocks() {
        let p = plan_blocks([true, true, true, false], 4096, 224, 1024);
        let window = p.iter().filter(|x| x.0 == Class::Window).count();
        let tail = p.iter().filter(|x| x.0 == Class::Tail).count();
        assert_eq!((window, tail), (7, 7));
        assert!(!p.iter().any(|x| x.0 == Class::Scale));
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
    fn segment_layout_is_repr_c_24_bytes() {
        assert_eq!(std::mem::size_of::<Seg>(), 24);
    }
}
