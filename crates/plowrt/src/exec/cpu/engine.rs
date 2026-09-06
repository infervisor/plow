//! CPU engine, host side: a loaded model = the device blob's tensor table
//! materialised in host memory + the kernels resolved per program.
//!
//! This is the CPU twin of the tensor-binding loop in `exec/gpu.rs` /
//! `exec/amd.rs` with the vendor plumbing (slabs, VMM, pinned pipes, peer
//! views) removed: on the CPU a tensor handle is a host pointer, so binding is
//! allocate + copy. Names are the contract — a blob tensor name IS the
//! checkpoint name (`packet::names`), `init` ranges come from the blob, and
//! RoPE tables are `GenTensor` recipes generated here.

use std::alloc::Layout;
use std::ffi::c_void;
use std::path::Path;
use std::time::Instant;

use packet::dev::DevInst64;

use crate::asset::checkpoint::Checkpoint;
use crate::asset::devblob::{DevBlob, DevProg};
use crate::exec::cpu::ffi::{self, KernelTable};
use crate::{Result, RuntimeError};

/// Allocation alignment. 64 B keeps every tensor cache-line aligned for the
/// AVX-512/AMX loads; weights ≥ 2 MiB take a 2 MiB alignment so the kernel can
/// back them with transparent huge pages (`madvise` below).
const ALIGN: usize = 64;
const HUGE: usize = 2 << 20;

/// One host tensor. Owns its allocation; freed on drop.
pub struct HostTensor {
    ptr: *mut u8,
    layout: Layout,
    pub bytes: usize,
}

// SAFETY: plain heap memory; concurrent access is disjoint by the schedule,
// exactly as `CpuArena` documents.
unsafe impl Send for HostTensor {}
unsafe impl Sync for HostTensor {}

impl HostTensor {
    fn alloc(bytes: usize, zeroed: bool) -> Result<HostTensor> {
        // Tensors from 256 KiB up are rounded to whole huge pages: the prefill A operands
        // (e.g. 128 x 3840 bf16 = 960 KiB) sit below 2 MiB yet are tile-loaded at a multi-KiB
        // row stride, where 4 KiB pages cost a TLB miss per tile row. Slack <= 2 MiB each.
        let huge = bytes >= HUGE / 8;
        let align = if huge { HUGE } else { ALIGN };
        let size = bytes.max(1).next_multiple_of(align);
        let layout = Layout::from_size_align(size, align)
            .map_err(|e| RuntimeError::Oom(format!("tensor layout {bytes} B: {e}")))?;
        // SAFETY: non-zero size layout.
        let ptr = unsafe {
            if zeroed {
                std::alloc::alloc_zeroed(layout)
            } else {
                std::alloc::alloc(layout)
            }
        };
        if ptr.is_null() {
            return Err(RuntimeError::Oom(format!("host tensor {bytes} B")));
        }
        #[cfg(target_os = "linux")]
        if huge {
            // Best effort; a refusal costs TLB misses, not correctness.
            // SAFETY: ptr/size describe our own mapping.
            unsafe { libc::madvise(ptr as *mut c_void, size, libc::MADV_HUGEPAGE) };
        }
        Ok(HostTensor { ptr, layout, bytes })
    }

    #[inline]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// # Safety
    /// No concurrent writer to this tensor (quiescent point).
    pub unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.ptr, self.bytes)
    }

    /// # Safety
    /// No concurrent reader or writer to this tensor (quiescent point).
    pub unsafe fn as_mut_slice(&self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.ptr, self.bytes)
    }
}

impl Drop for HostTensor {
    fn drop(&mut self) {
        // SAFETY: allocated with exactly this layout in `alloc`.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

/// The flat host-pointer table every kernel indexes by tensor handle, shared by
/// the model (which owns the allocations) and the worker pool's `KernelExec`.
///
/// Entries change only through [`CpuModel::kv_rebase`], and only while no run is
/// in flight — the host is the sole writer and workers read it only inside a run
/// (the `RUN` command's Release/Acquire handoff orders the writes), so plain
/// cells suffice; no per-packet atomic on the hot path.
pub struct TensorTable {
    cells: Box<[std::cell::UnsafeCell<*mut c_void>]>,
}

// SAFETY: see the type doc — written only at quiescent points by one thread.
unsafe impl Send for TensorTable {}
unsafe impl Sync for TensorTable {}

impl TensorTable {
    pub fn new(ptrs: Vec<*mut c_void>) -> Self {
        TensorTable {
            cells: ptrs.into_iter().map(std::cell::UnsafeCell::new).collect(),
        }
    }

    /// Base of the `*mut c_void[]` kernels receive (`UnsafeCell<T>` is `repr(transparent)`).
    #[inline]
    pub fn as_ptr(&self) -> *const *mut c_void {
        self.cells.as_ptr() as *const *mut c_void
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    #[inline]
    pub fn get(&self, h: usize) -> *mut c_void {
        // SAFETY: quiescent read (no concurrent `set`); see the type doc.
        unsafe { *self.cells[h].get() }
    }

    /// # Safety
    /// No run in flight (no worker may be reading the table).
    pub unsafe fn set(&self, h: usize, p: *mut c_void) {
        *self.cells[h].get() = p;
    }
}

/// Index of the narrowest rung (ascending widths) covering `rows` sequences,
/// or the widest when none does — the AMD ladder rule (`decode_prog_for`).
pub fn rung_for(rungs: &[u32], rows: usize) -> usize {
    rungs
        .iter()
        .position(|&t| t as usize >= rows)
        .unwrap_or(rungs.len().saturating_sub(1))
}

/// Well-known runtime tensors the step protocol writes/reads by name.
#[derive(Clone, Copy, Debug, Default)]
pub struct Wellknown {
    pub ids: Option<usize>,
    pub pos: Option<usize>,
    pub kvlen: Option<usize>,
    pub logits: Option<usize>,
}

/// A device blob bound to host memory.
pub struct CpuModel {
    pub blob: DevBlob,
    tensors: Vec<HostTensor>,
    /// `tensors[h].ptr` as the flat table every kernel indexes by handle; shared
    /// with the worker pool so [`CpuModel::kv_rebase`] is visible to kernels.
    table: Arc<TensorTable>,
    pub names: Vec<String>,
    /// Decode sequence slots (`in.kvlen` entries). Per-slot KV blocks are
    /// `[batch][...]` in every `kv.*` tensor; the prefill program is single-
    /// sequence and reaches slot `s` by [`CpuModel::kv_rebase`].
    pub batch: usize,
    /// `(handle, per-slot bytes)` for every per-slot KV tensor (empty at batch 1).
    kv_slot_stride: Vec<(usize, u64)>,
    /// Slot the KV pointer table is currently rebased onto (0 = base).
    kv_slot: usize,
    pub wk: Wellknown,
    /// Index of the first decode program (`packet::devbuild::decode_rung_lo`).
    pub dec_ix: usize,
    /// Decode-program instruction indices whose `i[3]` is the KV write row.
    pub kvrow: Vec<u32>,
    /// Resolved kernels, one table per program (indexed like `blob.progs`).
    pub kernels: Vec<KernelTable>,
    pub weight_bytes: u64,
    pub load_ms: f64,
}

// SAFETY: raw pointers are into `tensors`' own allocations.
unsafe impl Send for CpuModel {}
unsafe impl Sync for CpuModel {}

impl CpuModel {
    /// Parse `blob_path`, allocate every tensor in host memory, bind checkpoint
    /// weights / blob init data / generated tables, and resolve kernels for
    /// every program. `ffi::init` must have run.
    pub fn load(blob_path: &Path, checkpoint: &Path) -> Result<CpuModel> {
        let t0 = Instant::now();
        let raw = std::fs::read(blob_path)
            .map_err(|e| RuntimeError::Device(format!("read {}: {e}", blob_path.display())))?;
        // L2-domain placement is accepted: the CPU interpreter dispatches per
        // domain window itself (`exec::cpu::interp`), so the mis-dispatch the
        // flag guards against cannot happen here.
        let blob = DevBlob::parse_l2(&raw, true)?;
        // PLOW_FP8_DIR (the `--fp8-dir` runtime flag) names the fp8 weight-twin directory.
        // PLOW_MXFP4_DIR names the mxfp4 twin (`mxfp4/<name>` e2m1 + `_scale` E8M0 rows,
        // perf-data/tools/quantize_mxfp4.py); the two axes are exclusive at emit time.
        // `--fp8-dir` already exists runtime-wide (AmdRuntimeConfig owns that clap id and its
        // PLOW_FP8_DIR env); only the mxfp4 twin is CPU-specific. Declaring a second `fp8_dir`
        // field here shadowed the first, and the twin then silently never loaded.
        let rt = crate::config::RuntimeConfig::get();
        let twin = [rt.amd.fp8_dir.as_deref(), rt.cpu.mxfp4_dir.as_deref()]
            .into_iter()
            .flatten()
            .find(|d| !d.is_empty())
            .map(std::path::PathBuf::from);
        let ckpt = Checkpoint::open_with_twin(checkpoint, twin.as_deref())?;

        // Kernels first: a missing op is a cheap, loud failure — before 20 GiB of copies.
        let mut kernels = Vec::with_capacity(blob.progs.len());
        for (pi, p) in blob.progs.iter().enumerate() {
            let table = KernelTable::resolve(p.insts.iter().map(|d| d.op)).map_err(|missing| {
                let names: Vec<String> = missing
                    .iter()
                    .map(|&op| {
                        packet::dev::DevOp::from_u16(op)
                            .map(|o| o.c_name().to_string())
                            .unwrap_or_else(|| format!("op {op}"))
                    })
                    .collect();
                RuntimeError::Device(format!(
                    "program {pi} (T={}) uses {} device ops without a CPU kernel: {}",
                    p.t,
                    missing.len(),
                    names.join(", ")
                ))
            })?;
            kernels.push(table);
        }

        let gen_of: rustc_hash::FxHashMap<u32, &packet::rope::GenTensor> =
            blob.gen.iter().map(|g| (g.tensor, g)).collect();

        let mut tensors = Vec::with_capacity(blob.tensors.len());
        let mut names = Vec::with_capacity(blob.tensors.len());
        let mut wk = Wellknown::default();
        let mut weight_bytes = 0u64;
        for (h, td) in blob.tensors.iter().enumerate() {
            let bytes = td.bytes as usize;
            match td.name.as_str() {
                "in.ids" => wk.ids = Some(h),
                "in.pos" => wk.pos = Some(h),
                "in.kvlen" => wk.kvlen = Some(h),
                "act.logits" => wk.logits = Some(h),
                _ => {}
            }
            // Gemma's fused-expert pointer tables (`moe.ewt.<l>` / `moe.est.<l>`) are runtime
            // tensors filled below from the bound expert tensors; the MLA/GLM per-projection
            // tables are not wired yet.
            let gemma_table = td.name.starts_with("moe.ewt.") || td.name.starts_with("moe.est.");
            if !gemma_table && packet::names::is_host_filled_table(&td.name) {
                return Err(RuntimeError::Device(format!(
                    "host-filled expert table `{}` is not supported by the CPU engine yet",
                    td.name
                )));
            }
            let t = if packet::names::is_checkpoint_weight(&td.name) {
                let src = ckpt
                    .tensor(&td.name)
                    .ok_or_else(|| RuntimeError::Device(format!("MISSING WEIGHT: {}", td.name)))?;
                if src.len() != bytes {
                    return Err(RuntimeError::Device(format!(
                        "SIZE MISMATCH {} (blob {} B, checkpoint {} B)",
                        td.name,
                        bytes,
                        src.len()
                    )));
                }
                let t = HostTensor::alloc(bytes, false)?;
                // SAFETY: fresh allocation of `bytes`, no other reference yet.
                unsafe { t.as_mut_slice().copy_from_slice(src) };
                weight_bytes += td.bytes;
                t
            } else if let Some(r) = &td.init {
                let src = &blob.init[r.clone()];
                if src.len() != bytes {
                    return Err(RuntimeError::Device(format!(
                        "init size mismatch {} (blob {} B, init {} B)",
                        td.name,
                        bytes,
                        src.len()
                    )));
                }
                let t = HostTensor::alloc(bytes, false)?;
                unsafe { t.as_mut_slice().copy_from_slice(src) };
                t
            } else if let Some(g) = gen_of.get(&(h as u32)) {
                let data = g.generate().ok_or_else(|| {
                    RuntimeError::Device(format!("unknown gen-tensor kind {} for {}", g.kind, td.name))
                })?;
                if data.len() != bytes {
                    return Err(RuntimeError::Device(format!(
                        "gen-tensor size mismatch {} (blob {} B, generated {} B)",
                        td.name,
                        bytes,
                        data.len()
                    )));
                }
                let t = HostTensor::alloc(bytes, false)?;
                unsafe { t.as_mut_slice().copy_from_slice(&data) };
                t
            } else {
                // Runtime tensor (activations, KV, inputs): zeroed.
                HostTensor::alloc(bytes, true)?
            };
            tensors.push(t);
            names.push(td.name.clone());
        }
        // Fused-expert pointer tables: ewt[e*2] = gate_up rows of expert e, ewt[e*2+1] = its down
        // rows (host addresses, u64), stride = tensor bytes / E — the CPU twin of exec::gpu's
        // `build_fused_expert_table`. fp8 packets also get `moe.est.<l>` with the scale rows.
        for h in 0..names.len() {
            let Some(layer) = names[h].strip_prefix("moe.ewt.").map(str::to_string) else {
                continue;
            };
            let find = |suf: &str| names.iter().rposition(|n| n.ends_with(suf));
            let gu = find(&format!("layers.{layer}.experts.gate_up_proj"));
            let dn = find(&format!("layers.{layer}.experts.down_proj"));
            let (Some(gu), Some(dn)) = (gu, dn) else {
                return Err(RuntimeError::Device(format!(
                    "MoE: layer {layer} missing fused expert tensor(s) for moe.ewt"
                )));
            };
            let e = tensors[h].bytes / 16;
            if e == 0 || tensors[h].bytes % 16 != 0 {
                return Err(RuntimeError::Device(format!("moe.ewt.{layer}: bad size {}", tensors[h].bytes)));
            }
            let fill = |dst: &HostTensor, a: &HostTensor, b: &HostTensor| {
                let (sa, sb) = (a.bytes / e, b.bytes / e);
                // SAFETY: dst is a fresh runtime tensor of e*16 bytes; a/b are live allocations.
                let out = unsafe { std::slice::from_raw_parts_mut(dst.as_ptr() as *mut u64, e * 2) };
                for i in 0..e {
                    out[2 * i] = a.as_ptr() as u64 + (i * sa) as u64;
                    out[2 * i + 1] = b.as_ptr() as u64 + (i * sb) as u64;
                }
            };
            fill(&tensors[h], &tensors[gu], &tensors[dn]);
            if names[gu].starts_with("fp8/") {
                let est = names.iter().position(|n| *n == format!("moe.est.{layer}"));
                let gs = find(&format!("layers.{layer}.experts.gate_up_proj_scale"));
                let ds = find(&format!("layers.{layer}.experts.down_proj_scale"));
                let (Some(est), Some(gs), Some(ds)) = (est, gs, ds) else {
                    return Err(RuntimeError::Device(format!(
                        "MoE fp8: layer {layer} missing expert scale tensor/table"
                    )));
                };
                fill(&tensors[est], &tensors[gs], &tensors[ds]);
            }
            tracing::debug!(layer, experts = e, "moe: fused expert pointer table filled");
        }
        let table = Arc::new(TensorTable::new(
            tensors.iter().map(|t| t.as_ptr() as *mut c_void).collect(),
        ));
        // Mirrors `exec::amd`: the batch is the `in.kvlen` width, and every `kv.*`
        // tensor (except block-residual scratch) is `[batch]` slot blocks.
        let batch = wk
            .kvlen
            .map(|h| (tensors[h].bytes / 4).max(1))
            .unwrap_or(1);
        let mut kv_slot_stride = Vec::new();
        if batch > 1 {
            if let Some(t) = names
                .iter()
                .find(|n| n.starts_with("kv.") && n.contains("state"))
            {
                return Err(RuntimeError::Device(format!(
                    "batch {batch} with recurrent-state tensor `{t}`: per-slot carried state \
                     is not supported by the CPU engine yet"
                )));
            }
            kv_slot_stride = names
                .iter()
                .enumerate()
                .filter(|(_, n)| n.starts_with("kv.") && !n.contains("blkres"))
                .map(|(h, _)| (h, tensors[h].bytes as u64 / batch as u64))
                .collect();
        }

        let dec_ix = {
            let pt: Vec<u32> = blob.progs.iter().map(|p| p.t).collect();
            packet::devbuild::decode_rung_lo(&pt)
        };
        if blob.kvrow.is_empty() {
            // MLA-style packets declare no sites and need `exec::amd::derive_kvrow`'s
            // rule; not ported yet, so refuse rather than write every token to row 0.
            return Err(RuntimeError::Device(
                "blob declares no KV-append sites (n_kvrow = 0); CPU engine cannot derive them yet"
                    .into(),
            ));
        }
        let kvrow = blob.kvrow.clone();

        let load_ms = t0.elapsed().as_secs_f64() * 1e3;
        tracing::info!(
            tensors = tensors.len(),
            programs = blob.progs.len(),
            n_cu = blob.n_cu,
            weight_gib = format_args!("{:.2}", weight_bytes as f64 / (1u64 << 30) as f64),
            load_ms = format_args!("{load_ms:.0}"),
            isa = ?ffi::isa(),
            "CPU model loaded"
        );
        Ok(CpuModel {
            blob,
            tensors,
            table,
            names,
            batch,
            kv_slot_stride,
            kv_slot: 0,
            wk,
            dec_ix,
            kvrow,
            kernels,
            weight_bytes,
            load_ms,
        })
    }

    /// The flat host pointer table kernels index by handle.
    #[inline]
    pub fn tensor_table(&self) -> &Arc<TensorTable> {
        &self.table
    }

    /// Point every per-slot KV tensor at slot `slot`'s block, so the single-
    /// sequence prefill program writes that slot. The decode programs derive
    /// each sequence's block themselves (`i[6] = n_batch_kv`) and must run at
    /// slot 0 — callers restore the base before any decode ([`CpuEngine::prefill_slot`]).
    ///
    /// Must be called between runs: the table is read by kernels during a run.
    pub fn kv_rebase(&mut self, slot: usize) -> Result<()> {
        if self.kv_slot == slot || self.kv_slot_stride.is_empty() {
            return Ok(());
        }
        if slot >= self.batch {
            return Err(RuntimeError::Device(format!(
                "kv_rebase to slot {slot} past batch {}",
                self.batch
            )));
        }
        for &(h, stride) in &self.kv_slot_stride {
            let base = self.tensors[h].as_ptr() as usize + (stride as usize) * slot;
            // SAFETY: quiescent point (caller contract); `base` stays inside `tensors[h]`.
            unsafe { self.table.set(h, base as *mut c_void) };
        }
        self.kv_slot = slot;
        Ok(())
    }

    pub fn kv_slot(&self) -> usize {
        self.kv_slot
    }

    /// Decode rungs (sequence widths), ascending, one per decode program.
    pub fn decode_rungs(&self) -> Vec<u32> {
        self.blob.progs[self.dec_ix..].iter().map(|p| p.t).collect()
    }

    /// The narrowest decode program covering `rows` sequence slots (the widest
    /// when none does).
    pub fn decode_prog_for(&self, rows: usize) -> usize {
        let rungs = self.decode_rungs();
        self.dec_ix + rung_for(&rungs, rows)
    }

    pub fn tensor(&self, h: usize) -> &HostTensor {
        &self.tensors[h]
    }

    pub fn tensor_by_name(&self, name: &str) -> Option<&HostTensor> {
        self.names.iter().position(|n| n == name).map(|h| &self.tensors[h])
    }

    pub fn decode_prog(&self) -> &DevProg {
        &self.blob.progs[self.dec_ix]
    }

    /// Prefill programs, in blob order (ascending T).
    pub fn prefill_progs(&self) -> &[DevProg] {
        &self.blob.progs[..self.dec_ix]
    }

    /// Patch the KV-append row into every declared site of program `dp`
    /// (`i[3]`, the one field Gemma-class packets use). Host memory: a store.
    pub fn patch_kvrow(&mut self, dp: usize, row: u32) -> Result<()> {
        let n = self.blob.progs[dp].insts.len();
        for &i in &self.kvrow {
            let inst: &mut DevInst64 = self.blob.progs[dp]
                .insts
                .get_mut(i as usize)
                .ok_or_else(|| {
                    RuntimeError::Device(format!("kvrow site {i} past program {dp}'s {n} instructions"))
                })?;
            inst.i[3] = row;
        }
        Ok(())
    }

    /// Write a little-endian `u32` scalar tensor (`in.pos`, `in.kvlen`, ...).
    pub fn write_u32(&self, h: usize, v: u32) {
        // SAFETY: called between steps (no worker runs), tensor is ≥ 4 B by
        // construction of the blob.
        unsafe {
            let s = self.tensors[h].as_mut_slice();
            s[..4].copy_from_slice(&v.to_le_bytes());
        }
    }

    pub fn read_u32(&self, h: usize) -> u32 {
        // SAFETY: as `write_u32`.
        unsafe {
            let s = self.tensors[h].as_slice();
            u32::from_le_bytes(s[..4].try_into().expect("4 bytes"))
        }
    }
}

// ---------------------------------------------------------------------------
// Step driver
// ---------------------------------------------------------------------------

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use packet::dev::DevOp;

use crate::exec::counters::CounterPool;
use crate::exec::cpu::control::unpack_fault;
use crate::exec::cpu::ffi::{Isa, KernelFn, PlowCpuCtx};
use crate::exec::cpu::interp::{Exec, LoadedProgram, WorkerCtx};
use crate::exec::cpu::topology::{NumaMode, Topology};
use crate::exec::cpu::workers::WorkerPool;
use crate::exec::kvrow::{place_lm_head_row, rebase_chunk_rows};

/// One worker's kernel context: the C `PlowCpuCtx` plus its scratch arena.
/// `thread_init` (AMX tile config) must run ON the worker thread, so it is done
/// lazily at the worker's first packet.
struct WorkerSlot {
    ctx: UnsafeCell<PlowCpuCtx>,
    _scratch: HostTensor,
    inited: AtomicBool,
}

// SAFETY: each slot is touched only by its own worker thread.
unsafe impl Send for WorkerSlot {}
unsafe impl Sync for WorkerSlot {}

/// [`Exec`] over the C kernel library: resolves `inst.op` in a flat table and
/// calls it with the model's host pointer table.
pub struct KernelExec {
    table: Vec<Option<KernelFn>>,
    tensors: Arc<TensorTable>,
    slots: Vec<WorkerSlot>,
}

// SAFETY: `tensors` holds pointers into the model's allocations, which outlive
// the pool (the engine drops the pool first).
unsafe impl Send for KernelExec {}
unsafe impl Sync for KernelExec {}

impl KernelExec {
    fn new(model: &CpuModel, workers: usize, worker_node: impl Fn(usize) -> u32) -> Result<Self> {
        let mut table: Vec<Option<KernelFn>> = vec![None; ffi::DOP_TABLE];
        for p in &model.blob.progs {
            for d in &p.insts {
                let op = d.op as usize;
                if table[op].is_none() {
                    table[op] = ffi::kernel(d.op);
                }
            }
        }
        let scratch_bytes = ffi::scratch_bytes().max(64) as usize;
        let mut slots = Vec::with_capacity(workers);
        for w in 0..workers {
            let scratch = HostTensor::alloc(scratch_bytes, true)?;
            let mut ctx = PlowCpuCtx::new(w as u32, worker_node(w));
            ctx.scratch = scratch.as_ptr() as *mut c_void;
            ctx.scratch_bytes = scratch_bytes as u32;
            slots.push(WorkerSlot {
                ctx: UnsafeCell::new(ctx),
                _scratch: scratch,
                inited: AtomicBool::new(false),
            });
        }
        Ok(KernelExec {
            table,
            tensors: Arc::clone(&model.table),
            slots,
        })
    }
}

impl Exec for KernelExec {
    #[inline]
    fn exec(&self, inst: &DevInst64, slice: u32, nblk: u32, w: &WorkerCtx) {
        let slot = &self.slots[w.worker as usize];
        // SAFETY: only this worker thread touches its slot.
        let ctx = unsafe { &mut *slot.ctx.get() };
        if !slot.inited.load(Ordering::Relaxed) {
            ffi::thread_init(ctx).expect("cpu kernel thread init");
            slot.inited.store(true, Ordering::Relaxed);
        }
        let f = self.table[inst.op as usize].unwrap_or_else(|| {
            panic!(
                "no CPU kernel for {} (op {})",
                DevOp::from_u16(inst.op).map(|o| o.c_name()).unwrap_or("?"),
                inst.op
            )
        });
        // SAFETY: handles were validated at load (< n_tensors or NONE); the
        // kernel contract is the interpreter's (slice of nblk, disjoint work).
        unsafe { f(inst, slice, nblk, self.tensors.as_ptr(), ctx) };
    }
}

/// Copy a blob program into the interpreter's form. Static per-cu streams; the
/// global-queue windows are wired in P6.
/// Is this a dense (non-MoE) single-row decode program? Those are pure weight streaming, the one
/// shape that wants the SMT siblings rather than one worker per core.
fn dense_row_decode(p: &DevProg) -> bool {
    use packet::dev::DevOp;
    if p.t != 1 {
        return false;
    }
    !p.insts.iter().any(|d| {
        matches!(
            DevOp::from_u16(d.op),
            Some(
                DevOp::MoeGluMx
                    | DevOp::MoeDownMx
                    | DevOp::MoeGluMxPf
                    | DevOp::MoeDownMxPf
                    | DevOp::MoeExpertGluNormGemma
                    | DevOp::MoeExpertDownGemma
                    | DevOp::MoeGroupGluGemmaPf
                    | DevOp::MoeGroupDownGemmaPf
            )
        )
    })
}

fn loaded(p: &DevProg, n_cu: u32, per_node: &[Vec<u32>], nodes: usize, physical: usize, logical: usize) -> LoadedProgram {
    let n_seg = p.stream.iter().map(|e| e.seg as u32).max().map_or(1, |m| m + 1);
    let seg_ofs = if n_seg > 1 {
        packet::devbuild::static_seg_ofs(&p.stream, &p.stream_ofs, &p.stream_len, n_seg).ok()
    } else {
        None
    };
    // Global-queue mode (opt-in, `PLOW_CPU_GQ=1`): the blob's op-major `GQ01` stream windowed by
    // `[segment][l2 domain]`; workers claim from their domain's window and steal from the others,
    // so a slow slice no longer stalls its whole static stream. Static streams stay the default
    // until it measures faster.
    let gq_on = crate::config::RuntimeConfig::get().cpu.gq_opt_in;
    let gq = if gq_on && !p.gq_stream.is_empty() {
        let domains = p.l2_domains.max(1);
        if p.gq_seg_ofs.len() as u32 == n_seg * domains + 1 && p.gq_stream.len() == p.stream.len() {
            Some(crate::exec::cpu::interp::GlobalQueue {
                stream: p.gq_stream.clone(),
                seg_ofs: p.gq_seg_ofs.clone(),
                domains,
            })
        } else {
            tracing::warn!(
                windows = p.gq_seg_ofs.len(),
                n_seg,
                domains,
                "PLOW_CPU_GQ: blob GQ appendix does not match the program; using static streams"
            );
            None
        }
    } else {
        None
    };
    // No per-program narrowing: an idle worker still polls on the SMT sibling of a busy core, which
    // cost more than the narrowing gained (see the module note on worker width). The pool is sized
    // once for the model and every program uses all of it.
    let _ = (per_node, nodes, physical, logical);
    LoadedProgram {
        cus_of: None,
        insts: p.insts.clone(),
        stream: p.stream.clone(),
        stream_ofs: p.stream_ofs.clone(),
        stream_len: p.stream_len.clone(),
        waits: p.waits.clone(),
        succs: p.succs.clone(),
        n_cu,
        n_seg,
        seg_ofs,
        gq,
    }
}

/// Worker-pool knobs (`CpuRuntimeConfig` resolved).
#[derive(Clone, Debug)]
pub struct CpuEngineOpts {
    /// 0 = one per online logical cpu.
    pub threads: usize,
    pub numa: NumaMode,
    pub isa: Isa,
    pub spin_us: u32,
}

impl Default for CpuEngineOpts {
    fn default() -> Self {
        CpuEngineOpts {
            threads: 0,
            numa: NumaMode::Auto,
            isa: Isa::Amx,
            spin_us: 2000,
        }
    }
}

/// One chunk of a prefill: program index, first absolute row, real rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub prog: usize,
    pub c0: u32,
    pub clen: u32,
}

/// Greedy chunk plan over the compiled prefill buckets: the smallest bucket
/// that holds the remainder, else the largest, repeated.
pub fn plan_chunks(buckets: &[(usize, u32)], n_prompt: u32) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut c0 = 0u32;
    while c0 < n_prompt {
        let ch = next_chunk(buckets, n_prompt, c0, u32::MAX);
        c0 += ch.clen;
        out.push(ch);
    }
    out
}

/// The next chunk of an `n_prompt`-token prefill whose rows `[0, c0)` are done, with at most
/// `cap` rows this call: among the buckets not wider than `cap` (all of them if `cap` is
/// below the narrowest — a chunk must fit SOME compiled program), the smallest that holds the
/// remainder, else the widest. `cap == u32::MAX` reproduces [`plan_chunks`]'s steps.
pub fn next_chunk(buckets: &[(usize, u32)], n_prompt: u32, c0: u32, cap: u32) -> Chunk {
    let rem = n_prompt - c0;
    let pick = |allowed: &dyn Fn(u32) -> bool| {
        buckets
            .iter()
            .filter(|(_, t)| allowed(*t) && *t >= rem)
            .min_by_key(|(_, t)| *t)
            .or_else(|| buckets.iter().filter(|(_, t)| allowed(*t)).max_by_key(|(_, t)| *t))
            .copied()
    };
    let (prog, t) = pick(&|t| t <= cap)
        .or_else(|| pick(&|_| true))
        .expect("at least one prefill bucket");
    Chunk { prog, c0, clen: rem.min(t) }
}

/// A loaded model + its persistent worker pool: single sequence, greedy
/// on-device sampling, the `exec::amd` step protocol on host memory.
pub struct CpuEngine {
    model: CpuModel,
    pool: WorkerPool,
    progs: Vec<Arc<LoadedProgram>>,
    counters: Vec<Arc<CounterPool>>,
    max_ctx: usize,
    pub isa: Isa,
    pub threads: usize,
    /// Wall time of the last `run_prog`, for step telemetry.
    pub last_run_us: f64,
}

impl CpuEngine {
    pub fn load(blob: &Path, checkpoint: &Path, opts: &CpuEngineOpts) -> Result<CpuEngine> {
        let isa = ffi::init(opts.isa)?;
        let model = CpuModel::load(blob, checkpoint)?;
        let n_cu = model.blob.n_cu;
        let topo = Topology::detect();
        // The pool needs the Exec before it exists; build the exec against the
        // node placement the pool will use (same rule: round-robin over nodes).
        let nodes = topo.select_nodes(&opts.numa);
        let (physical_w, logical_w);
        let threads = if opts.threads == 0 {
            // Mirrors WorkerPool::spawn's placement list, which orders physical cores first and
            // their SMT siblings after, so `k` threads pin to `k` distinct logical cpus.
            let logical: usize = nodes
                .iter()
                .map(|&n| topo.cores_on_node(n).map(|c| c.siblings.len().max(1)).sum::<usize>())
                .sum();
            let physical: usize = nodes.iter().map(|&n| topo.cores_on_node(n).count()).sum();
            // ONE WORKER PER PHYSICAL CORE, not per logical cpu. The wide execution resources are
            // per core — both SMT siblings issue into the same TMUL and the same pair of 512-bit
            // FMA ports — so a second thread on a core buys contention, not throughput, for
            // anything compute-bound. Measured on this 8-core / 16-thread Sapphire Rapids, 8
            // threads vs 16:
            //
            //   GPT-OSS MXFP4   prefill 512 tok  445 vs 399 tok/s   decode 24.5 vs 25.5 ms
            //   same, AVX-512   prefill 512 tok  442 vs 394 tok/s   decode 24.3 vs 25.3 ms
            //   Gemma-12B bf16  prefill 512 tok  259 vs 185 tok/s   decode 235 vs 230 ms
            //   Gemma-26B MXFP4 prefill 512 tok  208 vs 207 tok/s   decode 37.3 vs 38.4 ms
            //
            // Prefill wants it badly (up to +40%) and every quantized decode prefers it; the one
            // case that favours the siblings is pure bf16 decode, which is weight-bandwidth-bound
            // and gains 2.4% from the extra outstanding loads. That is a poor trade against 40%,
            // so the rule is unconditional and `--cpu-threads` remains for hosts that disagree.
            let _ = isa;
            if physical > 0 {
                physical
            } else {
                logical.max(1)
            }
        } else {
            opts.threads
        };
        // An explicit --cpu-threads pins both widths; otherwise physical for compute-bound programs
        // and logical for a dense single-row decode.
        if opts.threads == 0 {
            physical_w = nodes.iter().map(|&n| topo.cores_on_node(n).count()).sum::<usize>().max(1);
            logical_w = nodes
                .iter()
                .map(|&n| topo.cores_on_node(n).map(|c| c.siblings.len().max(1)).sum::<usize>())
                .sum::<usize>()
                .max(1);
        } else {
            physical_w = threads;
            logical_w = threads;
        }
        // WORKER WIDTH IS PER MODEL. The wide execution resources are per core (both SMT siblings
        // issue into the same TMUL and the same pair of 512-bit FMA ports), so compute-bound work
        // wants one worker per physical core; a dense single-row decode is pure weight streaming and
        // wants the siblings for their extra outstanding loads. Measured through serve on
        // Gemma-12B fp8, chat_short c=1, TTFT / TPOT in ms:
        //
        //   physical (8)  410 / 147      logical (16)  474 / 132
        //
        // For a 64-token reply that is 9.7 s against 8.8 s, so a dense model takes logical. GPT-OSS
        // measured better on physical for BOTH phases (prefill 445 vs 399 tok/s, decode 24.5 vs
        // 25.5 ms), so anything MoE takes physical.
        //
        // Deliberately NOT per program, though prefill and decode do want different widths: a worker
        // with no cus for the running program still polls (200 us re-park) on the sibling of a busy
        // core, and that tax exceeded the gain — the same fp8 cell measured 626 / 132 with a logical
        // pool whose prefill was narrowed to 8. Fixing that needs the idle worker to stop polling,
        // which is the real prerequisite for per-phase widths.
        let wants_logical = model.blob.progs.iter().any(dense_row_decode);
        let threads = threads.max(if wants_logical { logical_w } else { physical_w });
        tracing::info!(
            threads,
            ?isa,
            physical_cores = topo.physical_cores(),
            "cpu: worker count"
        );
        let exec = Arc::new(KernelExec::new(&model, threads, |w| {
            // Mirrors WorkerPool::spawn's placement; only informational for kernels.
            nodes[w % nodes.len()]
        })?);
        let pool = WorkerPool::spawn(&topo, threads, &opts.numa, opts.spin_us, n_cu, exec);
        let progs: Vec<Arc<LoadedProgram>> =
            model.blob.progs
                .iter()
                .map(|p| Arc::new(loaded(p, n_cu, pool.per_node(), nodes.len(), physical_w, logical_w)))
                .collect();
        let counters = model
            .blob
            .progs
            .iter()
            .map(|p| Arc::new(CounterPool::with_len(p.n_counter as usize)))
            .collect();
        let max_ctx = model
            .wk
            .pos
            .map(|h| model.tensor(h).bytes / 4)
            .unwrap_or(0);
        tracing::info!(
            threads = pool.threads(),
            n_cu,
            ?isa,
            max_ctx,
            "CPU engine ready"
        );
        Ok(CpuEngine {
            model,
            pool,
            progs,
            counters,
            max_ctx,
            isa,
            threads,
            last_run_us: 0.0,
        })
    }

    pub fn model(&self) -> &CpuModel {
        &self.model
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    fn need(&self, h: Option<usize>, what: &str) -> Result<usize> {
        h.ok_or_else(|| RuntimeError::Device(format!("blob declares no `{what}` tensor")))
    }

    /// Zero the program's counters and run every segment to completion.
    fn run_prog(&mut self, p: usize) -> Result<()> {
        let t0 = Instant::now();
        let ctr = &self.counters[p];
        ctr.reset_all();
        let prog = &self.progs[p];
        for seg in 0..prog.n_seg() {
            let gen = self.pool.run(prog, seg, ctr);
            if let Some(f) = self.pool.wait_done(gen) {
                let (op, inst, worker) = unpack_fault(f).unwrap_or((0, 0, 0));
                return Err(RuntimeError::Device(format!(
                    "CPU worker {worker} faulted in program {p} seg {seg}: op {op} ({}) inst {inst}",
                    DevOp::from_u16(op).map(|o| o.c_name()).unwrap_or("?")
                )));
            }
        }
        self.last_run_us = t0.elapsed().as_secs_f64() * 1e6;
        Ok(())
    }

    /// Prefill `prompt` into KV rows `[0, len)`; returns the greedy next token
    /// (which the device also leaves in `in.ids[0]` for the first decode step).
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
        if prompt.is_empty() {
            return Err(RuntimeError::Device("prefill of an empty prompt".into()));
        }
        if prompt.len() > self.max_ctx {
            return Err(RuntimeError::Device(format!(
                "prompt of {} tokens exceeds max_ctx {}",
                prompt.len(),
                self.max_ctx
            )));
        }
        let buckets = self.prefill_buckets();
        if buckets.is_empty() {
            return Err(RuntimeError::Device("blob has no prefill program".into()));
        }
        let plan = plan_chunks(&buckets, prompt.len() as u32);
        tracing::info!(tokens = prompt.len(), chunks = ?plan, "prefill plan");
        for ch in plan {
            self.prefill_chunk(prompt, ch)?;
        }
        self.last_token()
    }

    /// The compiled prefill buckets as `(program, rows)`.
    pub fn prefill_buckets(&self) -> Vec<(usize, u32)> {
        (0..self.model.dec_ix)
            .map(|i| (i, self.model.blob.progs[i].t))
            .collect()
    }

    /// The argmax the last prefill chunk (or decode step) left in `in.ids[0]`.
    pub fn last_token(&self) -> Result<u32> {
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        Ok(self.model.read_u32(t_ids))
    }

    /// Run ONE prefill chunk: rows `[ch.c0, ch.c0 + ch.clen)` of `prompt` on program `ch.prog`
    /// (KV rows written at their absolute positions, flash window `[0, c0 + clen)`). After the
    /// chunk that covers the last token, [`Self::last_token`] is the first generated token.
    pub fn prefill_chunk(&mut self, prompt: &[u32], ch: Chunk) -> Result<()> {
        if ch.prog >= self.model.dec_ix || (ch.c0 + ch.clen) as usize > prompt.len() || ch.clen == 0 {
            return Err(RuntimeError::Device(format!("bad prefill chunk {ch:?}")));
        }
        let (t_ids, t_pos, t_kvlen) = (
            self.need(self.model.wk.ids, "in.ids")?,
            self.need(self.model.wk.pos, "in.pos")?,
            self.need(self.model.wk.kvlen, "in.kvlen")?,
        );
        {
            let t = self.model.blob.progs[ch.prog].t;
            // Inputs: ids (padded), positions, kv length after this chunk.
            {
                // SAFETY: no run in flight.
                let ids = unsafe { self.model.tensor(t_ids).as_mut_slice() };
                let pos = unsafe { self.model.tensor(t_pos).as_mut_slice() };
                for i in 0..t as usize {
                    let id = if (i as u32) < ch.clen {
                        prompt[(ch.c0 + i as u32) as usize]
                    } else {
                        0
                    };
                    ids[i * 4..i * 4 + 4].copy_from_slice(&id.to_le_bytes());
                    pos[i * 4..i * 4 + 4].copy_from_slice(&(ch.c0 + i as u32).to_le_bytes());
                }
            }
            self.model.write_u32(t_kvlen, ch.c0 + ch.clen);
            // Rebase the program from its pristine copy: KV write rows at c0,
            // flash window [c0, c0+clen), row counts for a partial chunk.
            let lp = Arc::make_mut(&mut self.progs[ch.prog]);
            lp.insts.copy_from_slice(&self.model.blob.progs[ch.prog].insts);
            rebase_chunk_rows(&mut lp.insts, &self.model.names, ch.c0, ch.clen, t, Some(t));
            if place_lm_head_row(&mut lp.insts, self.model.wk.logits, ch.clen - 1).is_none()
                && self.model.wk.logits.is_some()
            {
                tracing::warn!(prog = ch.prog, "act.logits declared but no matmul writes it");
            }
            self.run_prog(ch.prog)?;
        }
        Ok(())
    }

    /// Decode sequence slots.
    pub fn batch(&self) -> usize {
        self.model.batch
    }

    pub fn decode_rungs(&self) -> Vec<u32> {
        self.model.decode_rungs()
    }

    /// Prefill `prompt` into slot `slot`'s KV block (the `exec::amd` invariant:
    /// rebase, run the single-sequence prefill, restore the base — even on
    /// failure, since a table left on slot `s` would fold every decode into it).
    pub fn prefill_slot(&mut self, slot: usize, prompt: &[u32]) -> Result<u32> {
        self.model.kv_rebase(slot)?;
        let r = self.prefill(prompt);
        self.model.kv_rebase(0)?;
        r
    }

    /// One chunk of a slot prefill (same rebase/restore discipline as [`Self::prefill_slot`]).
    pub fn prefill_slot_chunk(&mut self, slot: usize, prompt: &[u32], ch: Chunk) -> Result<()> {
        self.model.kv_rebase(slot)?;
        let r = self.prefill_chunk(prompt, ch);
        self.model.kv_rebase(0)?;
        r
    }

    /// One decode step for `pos.len()` sequence slots on the narrowest rung
    /// covering the highest slot the caller marks live. `pos`/`kvlen`/`ids` are
    /// per slot and may be ragged; idle slots carry `(0, 1, any id)` like the AMD
    /// path. Returns every slot's sampled token (slots past the rung's rows read 0).
    pub fn decode_step_batched(&mut self, pos: &[u32], kvlen: &[u32], ids: &[u32]) -> Result<Vec<u32>> {
        let rows = pos.len();
        let dp = self.model.decode_prog_for(rows);
        self.decode_step_batched_at(pos, kvlen, ids, dp)
    }

    /// [`Self::decode_step_batched`] on a named decode program.
    pub fn decode_step_batched_at(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        ids: &[u32],
        dp: usize,
    ) -> Result<Vec<u32>> {
        let b = self.model.batch;
        if pos.len() != b || kvlen.len() != b || ids.len() != b {
            return Err(RuntimeError::Device(format!(
                "decode_step_batched wants {b} pos/kvlen/ids, got {}/{}/{}",
                pos.len(),
                kvlen.len(),
                ids.len()
            )));
        }
        if let Some(&p) = pos.iter().find(|&&p| p as usize >= self.max_ctx) {
            return Err(RuntimeError::Device(format!(
                "position {p} past max_ctx {}",
                self.max_ctx
            )));
        }
        if self.model.kv_slot != 0 {
            return Err(RuntimeError::Device(format!(
                "decode with the KV table rebased onto slot {} — prefill_slot must restore it",
                self.model.kv_slot
            )));
        }
        if dp < self.model.dec_ix || dp >= self.model.blob.progs.len() {
            return Err(RuntimeError::Device(format!("program {dp} is not a decode rung")));
        }
        let (t_ids, t_pos, t_kvlen) = (
            self.need(self.model.wk.ids, "in.ids")?,
            self.need(self.model.wk.pos, "in.pos")?,
            self.need(self.model.wk.kvlen, "in.kvlen")?,
        );
        // Only a batch-1 program takes the KV write row from the host-patched
        // `i[3]`; laddered / batched blobs arm `i[6] = n_batch_kv` and read `pos[t]`.
        if b == 1 {
            let lp = Arc::make_mut(&mut self.progs[dp]);
            for &i in &self.model.kvrow {
                lp.insts[i as usize].i[3] = pos[0];
            }
        }
        // SAFETY: no run in flight (host-only window between steps).
        unsafe {
            let s_ids = self.model.tensor(t_ids).as_mut_slice();
            let s_pos = self.model.tensor(t_pos).as_mut_slice();
            let s_kv = self.model.tensor(t_kvlen).as_mut_slice();
            for i in 0..b {
                s_ids[i * 4..i * 4 + 4].copy_from_slice(&ids[i].to_le_bytes());
                s_pos[i * 4..i * 4 + 4].copy_from_slice(&pos[i].to_le_bytes());
                s_kv[i * 4..i * 4 + 4].copy_from_slice(&kvlen[i].to_le_bytes());
            }
        }
        self.run_prog(dp)?;
        let rows = (self.model.blob.progs[dp].t as usize).min(b);
        let mut out = self.read_ids(rows);
        out.resize(b, 0);
        Ok(out)
    }

    /// One decode step of slot 0: the token in `in.ids[0]` (the previous sample)
    /// is embedded at `pos`, attends over `kvlen` rows, and the greedy next token
    /// is written back to `in.ids[0]` and returned. Other slots idle.
    pub fn decode_step(&mut self, pos: u32, kvlen: u32) -> Result<u32> {
        let b = self.model.batch;
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        let mut ids = self.read_ids(b);
        ids.resize(b, 0);
        let mut ps = vec![0u32; b];
        let mut ks = vec![1u32; b];
        ps[0] = pos;
        ks[0] = kvlen;
        let dp = self.model.decode_prog_for(1);
        let out = self.decode_step_batched_at(&ps, &ks, &ids, dp)?;
        debug_assert_eq!(out[0], self.model.read_u32(t_ids));
        Ok(out[0])
    }

    /// Seed `in.ids[0]` (e.g. to decode from a given token without a prefill).
    pub fn set_token(&self, id: u32) -> Result<()> {
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        self.model.write_u32(t_ids, id);
        Ok(())
    }

    /// The first `n` entries of `in.ids` (the device-sampled tokens per slot).
    pub fn read_ids(&self, n: usize) -> Vec<u32> {
        let Some(h) = self.model.wk.ids else {
            return vec![0; n];
        };
        // SAFETY: quiescent point.
        let s = unsafe { self.model.tensor(h).as_slice() };
        s.chunks_exact(4)
            .take(n)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
