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

use packet::dev::{DevInst64, DevOp, StreamEnt, SE_DOMAIN_MASK, SE_DOMAIN_SHIFT, TENSOR_NONE16};

use crate::asset::checkpoint::Checkpoint;
use crate::asset::devblob::{DevBlob, DevProg};
use crate::exec::cpu::ffi::{self, KernelTable};
use crate::{Result, RuntimeError};

/// Allocation alignment. 64 B keeps every tensor cache-line aligned for the
/// AVX-512/AMX loads; weights ≥ 2 MiB take a 2 MiB alignment so the kernel can
/// back them with transparent huge pages (`madvise` below).
const ALIGN: usize = 64;
const HUGE: usize = 2 << 20;

#[cfg(test)]
mod allocation_tests {
    use super::*;

    #[test]
    fn tensor_allocation_checks_overflow_and_zeroes_padding() {
        assert!(HostTensor::alloc(usize::MAX, false).is_err());
        for bytes in [0, 65, HUGE / 8, HUGE + 1] {
            let t = HostTensor::alloc(bytes, true).unwrap();
            assert_eq!(t.as_ptr() as usize % t.layout.align(), 0);
            let data = unsafe { std::slice::from_raw_parts(t.as_ptr(), t.layout.size()) };
            assert!(data.iter().all(|&v| v == 0));
        }
    }

    #[test]
    #[ignore = "requires Linux mbind permission; validates actual NUMA VMA policy"]
    #[cfg(target_os = "linux")]
    fn live_numa_policy() {
        let topo = Topology::detect();
        for nodes in [&topo.nodes[..1], &topo.nodes[..]] {
            let t = HostTensor::alloc_on_nodes(HUGE * 2 * nodes.len(), true, nodes, true).unwrap();
            let maps = std::fs::read_to_string("/proc/self/numa_maps").unwrap();
            let addr = format!("{:x} ", t.as_ptr() as usize);
            let entry = maps
                .lines()
                .find(|l| l.starts_with(&addr))
                .expect("NUMA VMA entry");
            assert!(
                entry.contains(if nodes.len() == 1 {
                    "bind:"
                } else {
                    "interleave:"
                }),
                "{entry}"
            );
            let policy = entry.split_whitespace().nth(1).unwrap();
            let requested: Vec<u32> = policy
                .split_once(':')
                .unwrap()
                .1
                .split(',')
                .flat_map(|part| {
                    let (lo, hi) = part.split_once('-').unwrap_or((part, part));
                    lo.parse::<u32>().unwrap()..=hi.parse::<u32>().unwrap()
                })
                .collect();
            assert_eq!(requested, nodes);
            let resident: Vec<(u32, usize)> = entry
                .split_whitespace()
                .filter_map(|part| part.strip_prefix('N'))
                .map(|part| {
                    let (node, pages) = part.split_once('=').unwrap();
                    (node.parse().unwrap(), pages.parse().unwrap())
                })
                .collect();
            assert_eq!(
                resident.iter().map(|(_, pages)| pages).sum::<usize>() * 4096,
                t.layout.size()
            );
            if nodes.len() == 1 {
                assert!(
                    resident.iter().all(|(node, _)| *node == nodes[0]),
                    "{entry}"
                );
            }
            let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
            let header = format!("{:x}-", t.as_ptr() as usize);
            let mapping = smaps.split_once(&header).expect("tensor mapping").1;
            let flags = mapping.lines().find(|l| l.starts_with("VmFlags:")).unwrap();
            let huge = crate::config::RuntimeConfig::get()
                .cpu
                .huge_pages
                .unwrap_or(nodes.len() <= 1);
            assert!(
                flags
                    .split_whitespace()
                    .any(|f| f == if huge { "hg" } else { "nh" }),
                "{flags}"
            );
            eprintln!("{entry}");
        }
    }
}

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
        Self::alloc_on_nodes(bytes, zeroed, &[], false)
    }

    fn alloc_on_nodes(
        bytes: usize,
        zeroed: bool,
        nodes: &[u32],
        strict: bool,
    ) -> Result<HostTensor> {
        // Tensors from 256 KiB up are rounded to whole huge pages: the prefill A operands
        // (e.g. 128 x 3840 bf16 = 960 KiB) sit below 2 MiB yet are tile-loaded at a multi-KiB
        // row stride, where 4 KiB pages cost a TLB miss per tile row. Slack <= 2 MiB each.
        let huge = bytes >= HUGE / 8;
        // macOS: every tensor is page-aligned (16 KiB) and page-sized so the Metal engine can wrap
        // it as an `MTLBuffer` without a copy (`newBufferWithBytesNoCopy` requires both).
        #[cfg(target_os = "macos")]
        let align = if huge { HUGE } else { 16384 };
        #[cfg(not(target_os = "macos"))]
        let align = if huge { HUGE } else { ALIGN };
        let size = bytes
            .max(1)
            .checked_next_multiple_of(align)
            .ok_or_else(|| RuntimeError::Oom(format!("tensor size overflow: {bytes}")))?;
        let layout = Layout::from_size_align(size, align)
            .map_err(|e| RuntimeError::Oom(format!("tensor layout {bytes} B: {e}")))?;
        // SAFETY: non-zero size layout.
        let heap_alloc = || unsafe {
            if zeroed && !huge {
                std::alloc::alloc_zeroed(layout)
            } else {
                std::alloc::alloc(layout)
            }
        };
        #[cfg(target_os = "linux")]
        let ptr = if huge {
            let reserve = size
                .checked_add(align)
                .ok_or_else(|| RuntimeError::Oom(format!("tensor mapping overflow: {bytes}")))?;
            // A fresh mapping lets mbind establish placement before the first page fault,
            // and prevents allocator reuse from inheriting a previous tensor's policy.
            unsafe {
                let raw = libc::mmap(
                    std::ptr::null_mut(),
                    reserve,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                );
                if raw == libc::MAP_FAILED {
                    return Err(RuntimeError::Oom(format!(
                        "host mapping {bytes} B: {}",
                        std::io::Error::last_os_error()
                    )));
                }
                let addr = (raw as usize).next_multiple_of(align);
                let prefix = addr - raw as usize;
                if prefix > 0 {
                    libc::munmap(raw, prefix);
                }
                let suffix = reserve - prefix - size;
                if suffix > 0 {
                    libc::munmap((addr + size) as *mut c_void, suffix);
                }
                addr as *mut u8
            }
        } else {
            heap_alloc()
        };
        #[cfg(not(target_os = "linux"))]
        let ptr = heap_alloc();
        if ptr.is_null() {
            return Err(RuntimeError::Oom(format!("host tensor {bytes} B")));
        }
        #[cfg(target_os = "linux")]
        if huge {
            let use_huge = crate::config::RuntimeConfig::get()
                .cpu
                .huge_pages
                .unwrap_or(nodes.len() <= 1);
            // THP allocation can fall back to a different node instead of reclaiming
            // base pages on the interleave target, concentrating shared weights.
            let advice = if use_huge {
                libc::MADV_HUGEPAGE
            } else {
                libc::MADV_NOHUGEPAGE
            };
            // Best effort; this does not change the allocation's size or alignment.
            // SAFETY: ptr/size describe our own mapping.
            unsafe { libc::madvise(ptr as *mut c_void, size, advice) };
        }
        let tensor = HostTensor { ptr, layout, bytes };
        #[cfg(target_os = "linux")]
        if huge && !nodes.is_empty() {
            let maxnode = nodes.iter().copied().max().unwrap() as usize + 1;
            let bits = libc::c_ulong::BITS as usize;
            let mut mask = vec![0 as libc::c_ulong; maxnode.div_ceil(bits)];
            for &node in nodes {
                mask[node as usize / bits] |= 1 << (node as usize % bits);
            }
            // Linux's nodemask ABI consumes maxnode - 1 bits, including for node zero.
            let rc = unsafe {
                libc::syscall(
                    libc::SYS_mbind,
                    ptr,
                    size,
                    if nodes.len() == 1 {
                        libc::MPOL_BIND
                    } else {
                        libc::MPOL_INTERLEAVE
                    },
                    mask.as_ptr(),
                    mask.len() * bits + 1,
                    0u32,
                )
            };
            if rc != 0 {
                let error = std::io::Error::last_os_error();
                if strict {
                    return Err(RuntimeError::Device(format!(
                        "NUMA placement on {nodes:?}: {error}"
                    )));
                }
                static WARNED: AtomicBool = AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(%error, ?nodes, "NUMA placement unavailable; retaining OS memory policy");
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        if strict && !nodes.is_empty() {
            return Err(RuntimeError::Device(
                "explicit NUMA placement requires Linux".into(),
            ));
        }
        if huge && zeroed {
            unsafe { std::ptr::write_bytes(ptr, 0, size) };
        }
        Ok(tensor)
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
}

impl Drop for HostTensor {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        if self.layout.align() == HUGE {
            unsafe { libc::munmap(self.ptr.cast(), self.layout.size()) };
            return;
        }
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

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
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

fn validate_stream_entry(
    p: &DevProg,
    pi: usize,
    label: &str,
    ei: usize,
    e: &StreamEnt,
) -> Result<()> {
    let inst = p.insts.get(e.inst as usize).ok_or_else(|| {
        RuntimeError::Device(format!(
            "program {pi} {label} entry {ei} references instruction {} of {}",
            e.inst,
            p.insts.len()
        ))
    })?;
    if inst.blocks == 0 || e.slice >= inst.blocks as u32 {
        return Err(RuntimeError::Device(format!(
            "program {pi} {label} entry {ei} has slice {} for {} blocks",
            e.slice, inst.blocks
        )));
    }
    let check_range = |what: &str, ofs: u32, len: u16, total: usize| -> Result<()> {
        let start = ofs as usize;
        let end = start.checked_add(len as usize).filter(|&end| end <= total);
        if end.is_none() {
            return Err(RuntimeError::Device(format!(
                "program {pi} {label} entry {ei} {what} range {start}+{len} exceeds {total}"
            )));
        }
        Ok(())
    };
    check_range("wait", e.wait_ofs, e.wait_len, p.waits.len())?;
    check_range("successor", e.succ_ofs, e.succ_len, p.succs.len())?;
    for wait in &p.waits[e.wait_ofs as usize..e.wait_ofs as usize + e.wait_len as usize] {
        if wait.id >= p.n_counter {
            return Err(RuntimeError::Device(format!(
                "program {pi} {label} entry {ei} waits on counter {} of {}",
                wait.id, p.n_counter
            )));
        }
    }
    for &counter in &p.succs[e.succ_ofs as usize..e.succ_ofs as usize + e.succ_len as usize] {
        if counter >= p.n_counter {
            return Err(RuntimeError::Device(format!(
                "program {pi} {label} entry {ei} increments counter {counter} of {}",
                p.n_counter
            )));
        }
    }
    Ok(())
}

fn validate_cpu_blob(blob: &DevBlob) -> Result<()> {
    if blob.n_cu == 0 {
        return Err(RuntimeError::Device(
            "CPU blob declares no compute units".into(),
        ));
    }
    if blob.progs.is_empty() {
        return Err(RuntimeError::Device("CPU blob declares no programs".into()));
    }
    if blob.tensors.len() > TENSOR_NONE16 as usize {
        return Err(RuntimeError::Device(format!(
            "CPU blob declares {} tensors; the u16 instruction format supports at most {}",
            blob.tensors.len(),
            TENSOR_NONE16
        )));
    }

    for (pi, p) in blob.progs.iter().enumerate() {
        if p.t == 0 {
            return Err(RuntimeError::Device(format!("program {pi} has T=0")));
        }
        for (ii, inst) in p.insts.iter().enumerate() {
            if inst.op as usize >= ffi::DOP_TABLE {
                return Err(RuntimeError::Device(format!(
                    "program {pi} instruction {ii} has opcode {} beyond the CPU dispatch table",
                    inst.op
                )));
            }
            for (slot, &handle) in inst.t.iter().enumerate() {
                if handle != TENSOR_NONE16 && handle as usize >= blob.tensors.len() {
                    return Err(RuntimeError::Device(format!(
                        "program {pi} instruction {ii} tensor slot {slot} has handle {handle} of {}",
                        blob.tensors.len()
                    )));
                }
            }
            if inst.op == DevOp::RmsNorm as u16
                && (inst.t[3] != TENSOR_NONE16 || inst.t[4] != TENSOR_NONE16)
            {
                return Err(RuntimeError::Device(
                    "CPU RMSNorm FP8 fusion is unsupported; compile without --qnorm-fuse".into(),
                ));
            }
            if inst.op == DevOp::QuantFp8 as u16 {
                let fused = inst.t[3] != TENSOR_NONE16;
                if fused != (inst.t[4] != TENSOR_NONE16) {
                    return Err(RuntimeError::Device(
                        "CPU QUANT_FP8 requires both gate and up tensors".into(),
                    ));
                }
                let elements = u64::from(inst.i[0]) * u64::from(inst.i[1]);
                let bf16_bytes = elements.checked_mul(2).ok_or_else(|| {
                    RuntimeError::Device("CPU QUANT_FP8 dimensions overflow".into())
                })?;
                let sizes = [
                    elements,
                    bf16_bytes,
                    u64::from(inst.i[0]) * 4,
                    bf16_bytes,
                    bf16_bytes,
                ];
                for (slot, &bytes) in sizes[..if fused { 5 } else { 3 }].iter().enumerate() {
                    let handle = inst.t[slot];
                    if handle == TENSOR_NONE16 || blob.tensors[handle as usize].bytes < bytes {
                        return Err(RuntimeError::Device(format!(
                            "CPU QUANT_FP8 tensor slot {slot} is missing or smaller than {bytes} bytes"
                        )));
                    }
                }
            }
            if inst.op == DevOp::GemvQkv as u16 {
                for slot in 5..8 {
                    let handle = inst.i[slot] as usize;
                    if handle != 0 && handle >= blob.tensors.len() {
                        return Err(RuntimeError::Device(format!(
                            "program {pi} instruction {ii} bias slot i{slot} has handle {handle} of {}",
                            blob.tensors.len()
                        )));
                    }
                }
            }
        }
        if p.stream_ofs.len() != blob.n_cu as usize || p.stream_len.len() != blob.n_cu as usize {
            return Err(RuntimeError::Device(format!(
                "program {pi} has {}/{} stream offsets/lengths for {} compute units",
                p.stream_ofs.len(),
                p.stream_len.len(),
                blob.n_cu
            )));
        }
        for cu in 0..blob.n_cu as usize {
            let start = p.stream_ofs[cu] as usize;
            let len = p.stream_len[cu] as usize;
            if start
                .checked_add(len)
                .is_none_or(|end| end > p.stream.len())
            {
                return Err(RuntimeError::Device(format!(
                    "program {pi} compute-unit {cu} stream {start}+{len} exceeds {} entries",
                    p.stream.len()
                )));
            }
        }
        for (ei, e) in p.stream.iter().enumerate() {
            validate_stream_entry(p, pi, "static", ei, e)?;
        }
        if !p.gq_stream.is_empty() || !p.gq_seg_ofs.is_empty() {
            if p.gq_stream.len() != p.stream.len() {
                return Err(RuntimeError::Device(format!(
                    "program {pi} global queue has {} entries for a {}-entry static stream",
                    p.gq_stream.len(),
                    p.stream.len()
                )));
            }
            if p.gq_seg_ofs.first().copied() != Some(0)
                || p.gq_seg_ofs.last().copied() != Some(p.gq_stream.len() as u32)
                || p.gq_seg_ofs.windows(2).any(|w| w[0] > w[1])
            {
                return Err(RuntimeError::Device(format!(
                    "program {pi} has invalid global-queue window offsets"
                )));
            }
            for (ei, e) in p.gq_stream.iter().enumerate() {
                validate_stream_entry(p, pi, "global-queue", ei, e)?;
            }
        }
    }

    let dec_ix = blob.decode_rung_lo();
    if dec_ix >= blob.progs.len() {
        return Err(RuntimeError::Device(
            "CPU blob has no decode program".into(),
        ));
    }
    let word_len = |name: &str| -> Result<Option<usize>> {
        let Some(tensor) = blob.tensors.iter().find(|tensor| tensor.name == name) else {
            return Ok(None);
        };
        if tensor.bytes < 4 || !tensor.bytes.is_multiple_of(4) {
            return Err(RuntimeError::Device(format!(
                "tensor `{name}` has {} bytes; expected a non-empty u32 array",
                tensor.bytes
            )));
        }
        Ok(Some(tensor.bytes as usize / 4))
    };
    let batch = word_len("in.kvlen")?.unwrap_or(1);
    let max_rows = blob.progs.iter().map(|p| p.t as usize).max().unwrap_or(1);
    for name in ["in.ids", "in.pos"] {
        if let Some(words) = word_len(name)? {
            if words < max_rows.max(batch) {
                return Err(RuntimeError::Device(format!(
                    "tensor `{name}` has {words} rows; programs require {}",
                    max_rows.max(batch)
                )));
            }
        }
    }
    for (pi, p) in blob.progs.iter().enumerate().skip(dec_ix) {
        if p.t as usize > batch {
            return Err(RuntimeError::Device(format!(
                "decode program {pi} has T={}, larger than batch {batch}",
                p.t
            )));
        }
        for &site in &blob.kvrow {
            if site as usize >= p.insts.len() {
                return Err(RuntimeError::Device(format!(
                    "KV-row site {site} exceeds decode program {pi}'s {} instructions",
                    p.insts.len()
                )));
            }
        }
    }
    Ok(())
}

impl CpuModel {
    /// Parse `blob_path`, allocate every tensor in host memory, bind checkpoint
    /// weights / blob init data / generated tables, and resolve kernels for
    /// every program. `ffi::init` must have run.
    pub fn load(blob_path: &Path, checkpoint: &Path) -> Result<CpuModel> {
        Self::load_on_nodes(blob_path, checkpoint, &[], false)
    }

    fn load_on_nodes(
        blob_path: &Path,
        checkpoint: &Path,
        nodes: &[u32],
        strict: bool,
    ) -> Result<CpuModel> {
        let t0 = Instant::now();
        let raw = std::fs::read(blob_path)
            .map_err(|e| RuntimeError::Device(format!("read {}: {e}", blob_path.display())))?;
        // L2-domain placement is accepted: the CPU interpreter dispatches per
        // domain window itself (`exec::cpu::interp`), so the mis-dispatch the
        // flag guards against cannot happen here.
        let blob = DevBlob::parse_l2(&raw, true)?;
        validate_cpu_blob(&blob)?;
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
                let t = HostTensor::alloc_on_nodes(bytes, false, nodes, strict)?;
                // SAFETY: fresh allocation of `bytes`, no other reference yet.
                unsafe { std::slice::from_raw_parts_mut(t.as_ptr(), t.bytes).copy_from_slice(src) };
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
                let t = HostTensor::alloc_on_nodes(bytes, false, nodes, strict)?;
                unsafe { std::slice::from_raw_parts_mut(t.as_ptr(), t.bytes).copy_from_slice(src) };
                t
            } else if let Some(g) = gen_of.get(&(h as u32)) {
                let data = g.generate().ok_or_else(|| {
                    RuntimeError::Device(format!(
                        "unknown gen-tensor kind {} for {}",
                        g.kind, td.name
                    ))
                })?;
                if data.len() != bytes {
                    return Err(RuntimeError::Device(format!(
                        "gen-tensor size mismatch {} (blob {} B, generated {} B)",
                        td.name,
                        bytes,
                        data.len()
                    )));
                }
                let t = HostTensor::alloc_on_nodes(bytes, false, nodes, strict)?;
                unsafe {
                    std::slice::from_raw_parts_mut(t.as_ptr(), t.bytes).copy_from_slice(&data)
                };
                t
            } else {
                // Runtime tensor (activations, KV, inputs): zeroed.
                HostTensor::alloc_on_nodes(bytes, true, nodes, strict)?
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
                return Err(RuntimeError::Device(format!(
                    "moe.ewt.{layer}: bad size {}",
                    tensors[h].bytes
                )));
            }
            let fill = |dst: &HostTensor, a: &HostTensor, b: &HostTensor| -> Result<()> {
                if !a.bytes.is_multiple_of(e) || !b.bytes.is_multiple_of(e) {
                    return Err(RuntimeError::Device(format!(
                        "moe.ewt.{layer}: expert tensors do not divide into {e} experts"
                    )));
                }
                let (sa, sb) = (a.bytes / e, b.bytes / e);
                // SAFETY: dst is a fresh runtime tensor of e*16 bytes; a/b are live allocations.
                let out =
                    unsafe { std::slice::from_raw_parts_mut(dst.as_ptr() as *mut u64, e * 2) };
                for i in 0..e {
                    out[2 * i] = a.as_ptr() as u64 + (i * sa) as u64;
                    out[2 * i + 1] = b.as_ptr() as u64 + (i * sb) as u64;
                }
                Ok(())
            };
            fill(&tensors[h], &tensors[gu], &tensors[dn])?;
            if names[gu].starts_with("fp8/") {
                let est = names.iter().position(|n| *n == format!("moe.est.{layer}"));
                let gs = find(&format!("layers.{layer}.experts.gate_up_proj_scale"));
                let ds = find(&format!("layers.{layer}.experts.down_proj_scale"));
                let (Some(est), Some(gs), Some(ds)) = (est, gs, ds) else {
                    return Err(RuntimeError::Device(format!(
                        "MoE fp8: layer {layer} missing expert scale tensor/table"
                    )));
                };
                fill(&tensors[est], &tensors[gs], &tensors[ds])?;
            }
            tracing::debug!(layer, experts = e, "moe: fused expert pointer table filled");
        }
        let table = Arc::new(TensorTable::new(
            tensors.iter().map(|t| t.as_ptr() as *mut c_void).collect(),
        ));
        // Mirrors `exec::amd`: the batch is the `in.kvlen` width, and every `kv.*`
        // tensor (except block-residual scratch) is `[batch]` slot blocks.
        let batch = wk.kvlen.map(|h| (tensors[h].bytes / 4).max(1)).unwrap_or(1);
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
            for (h, name) in names
                .iter()
                .enumerate()
                .filter(|(_, n)| n.starts_with("kv.") && !n.contains("blkres"))
            {
                if !tensors[h].bytes.is_multiple_of(batch) {
                    return Err(RuntimeError::Device(format!(
                        "KV tensor `{name}` has {} bytes, not divisible by batch {batch}",
                        tensors[h].bytes
                    )));
                }
                kv_slot_stride.push((h, tensors[h].bytes as u64 / batch as u64));
            }
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
        if slot >= self.batch {
            return Err(RuntimeError::Device(format!(
                "kv_rebase to slot {slot} past batch {}",
                self.batch
            )));
        }
        if self.kv_slot == slot || self.kv_slot_stride.is_empty() {
            return Ok(());
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

    /// The pointer kernels currently receive for handle `h` (the tensor base, or the slot-rebased
    /// KV block after [`Self::kv_rebase`]).
    pub fn table_ptr(&self, h: usize) -> *mut c_void {
        // SAFETY: `h` indexes the table (validated at load); quiescent point.
        unsafe { *self.table.as_ptr().add(h) }
    }

    /// Read-only view of `len` bytes at `off` inside tensor `handle`.
    ///
    /// The head prefill's KV rows are already host memory, so a handoff reads
    /// its source straight out of the tensor rather than staging a copy of it.
    ///
    /// Addresses the tensor's BASE allocation, NOT [`Self::table_ptr`], which
    /// may be slot-rebased by [`Self::kv_rebase`]. A handoff plan already
    /// carries the slot in its offset, so reading through the rebased pointer
    /// would apply the slot twice.
    ///
    /// Bounds-checked because the caller's offsets come from a plan derived
    /// from the OTHER packet's declarations, and the whole point of the KV
    /// contract is that the two agreeing is checked rather than assumed.
    pub fn tensor_range(&self, handle: usize, off: u64, len: u64) -> Result<&[u8]> {
        let t = self
            .tensors
            .get(handle)
            .ok_or_else(|| RuntimeError::Rejected(format!("tensor handle {handle} out of range")))?;
        let end = off.checked_add(len).ok_or_else(|| {
            RuntimeError::Rejected(format!("tensor {handle} range {off}+{len} overflows"))
        })?;
        if end > t.bytes as u64 {
            return Err(RuntimeError::Rejected(format!(
                "tensor {handle} range {off}+{len} past its {} bytes",
                t.bytes
            )));
        }
        // SAFETY: `handle` indexes this model's own allocation, the range is
        // inside it, and `&self` keeps it alive and unwritten for the borrow.
        Ok(unsafe { std::slice::from_raw_parts(t.as_ptr().add(off as usize), len as usize) })
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
        self.names
            .iter()
            .position(|n| n == name)
            .map(|h| &self.tensors[h])
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
            let inst: &mut DevInst64 =
                self.blob.progs[dp]
                    .insts
                    .get_mut(i as usize)
                    .ok_or_else(|| {
                        RuntimeError::Device(format!(
                            "kvrow site {i} past program {dp}'s {n} instructions"
                        ))
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
            let tensor = &self.tensors[h];
            let dst = std::slice::from_raw_parts_mut(tensor.as_ptr(), tensor.bytes);
            dst[..4].copy_from_slice(&v.to_le_bytes());
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
            let scratch = HostTensor::alloc(scratch_bytes, false)?;
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
            unsafe {
                std::ptr::write_bytes(ctx.scratch.cast::<u8>(), 0, ctx.scratch_bytes as usize)
            };
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

/// cu -> L2 locality domain, plus the domain count, recovered from the bits the emitter tags onto
/// every stream entry (`SE_DOMAIN_MASK`). `L2Layout::domain_of` is a pure function of the workgroup
/// index and one layout serves a whole build, so each cu's entries all carry the same value and the
/// placed programs agree; both are checked rather than assumed.
///
/// `None` keeps placement on the topology-only round-robin, which is the right answer whenever the
/// blob expresses no locality: no `PLOW_L2_PLACE` at compile time, the legacy layout that encoded
/// the domain in `seg` rather than in flags (every domain bit reads zero), a single domain, or
/// placed programs that disagree. The domain is a RELATIVE hint about which slices share producers
/// — never a node id — so the mapping onto real NUMA nodes happens here, at load, where the host
/// topology is known. That is what keeps one blob portable across hosts with different node counts.
pub fn cu_domains(progs: &[DevProg], n_cu: u32) -> Option<(Vec<u32>, u32)> {
    let mut dom = vec![u32::MAX; n_cu as usize];
    let mut domains = 0u32;
    for p in progs.iter().filter(|p| p.l2_domains != 0) {
        if domains != 0 && domains != p.l2_domains {
            return None;
        }
        // `validate_cpu_blob` has already sized these, but this stays total for any caller.
        if p.stream_ofs.len() < n_cu as usize || p.stream_len.len() < n_cu as usize {
            return None;
        }
        domains = p.l2_domains;
        for (cu, slot) in dom.iter_mut().enumerate() {
            let start = p.stream_ofs[cu] as usize;
            // Slicing directly would panic on a blob that never went through `validate_cpu_blob`,
            // which a caller outside the engine (the l2_probe example) does not run.
            let entries = start
                .checked_add(p.stream_len[cu] as usize)
                .and_then(|end| p.stream.get(start..end))?;
            for e in entries {
                let d = ((e.flags & SE_DOMAIN_MASK) >> SE_DOMAIN_SHIFT) as u32;
                if d >= domains {
                    return None;
                }
                if *slot == u32::MAX {
                    *slot = d;
                } else if *slot != d {
                    return None;
                }
            }
        }
    }
    if domains < 2 || dom.iter().all(|&d| d == u32::MAX) {
        return None;
    }
    // A cu no placed program gave work to carries no hint; spread those so they stay balanced.
    for (cu, d) in dom.iter_mut().enumerate() {
        if *d == u32::MAX {
            *d = cu as u32 % domains;
        }
    }
    Some((dom, domains))
}

/// Workers to spawn: the width the topology and model want, but never more than the packet has
/// executors.
///
/// `cu_map` deals cus `0..n_cu` over the pool, so a worker past `n_cu` owns nothing in EVERY
/// program. It cannot be rescued by any later narrowing: it wakes on each run, finds an empty
/// stream, returns to the control ring, and spins `--cpu-spin-us` before parking — on the cores the
/// working set needs. This is NOT the per-program narrowing that measured worse (see the note at
/// the call site); those workers were idle for one program and busy for another, while these are
/// unusable for the whole model, so there is no width tradeoff to lose.
///
/// Measured on Gemma-4-31B dense, `n_cu = 144`, 8-node EPYC 9654, means of 3 (TTFT / decode step):
///
/// | workers | TTFT | decode step |
/// |---|---|---|
/// | 384 (all logical, the old default) | 20.50 s | 3459 ms |
/// | 192 (all physical) | 16.24 s | 1196 ms |
/// | 144 (`n_cu`) | 14.08 s | 415 ms |
///
/// The decode penalty tracks the idle count — 240 idle is 8.3x, 48 idle is 2.9x, 0 is baseline.
///
/// The global queue is exempt: there every worker claims from the shared window, so workers past
/// `n_cu` do useful work rather than idling. An explicit `--cpu-threads` is always honoured.
fn worker_width(explicit: usize, want: usize, n_cu: u32, gq: bool) -> usize {
    if explicit != 0 {
        return explicit;
    }
    if gq {
        return want.max(1);
    }
    want.min(n_cu as usize).max(1)
}

/// Stream entries each cu runs, ONE VECTOR PER PROGRAM — the work unit the static walk executes.
///
/// Kept per program rather than summed because programs are ALTERNATIVES: a prefill bucket or the
/// decode program is chosen per dispatch, and their loads never overlap. Summing them lets a plan
/// that ruins one program hide behind another that leans the other way — two programs at 200/2 and
/// 2/200 across a pair of nodes total 202/202 and look perfectly balanced, while each one on its
/// own is a 100x spread.
pub fn cu_work(progs: &[DevProg], n_cu: u32) -> Vec<Vec<u64>> {
    progs
        .iter()
        .map(|p| {
            (0..n_cu as usize)
                .map(|cu| *p.stream_len.get(cu).unwrap_or(&0) as u64)
                .collect()
        })
        .collect()
}

/// Busiest node under `plan` and under the `cu % nodes` round-robin, for one program's work.
fn peak_loads(plan: &[u32], work: &[u64], nodes: usize) -> (u64, u64) {
    let (mut placed, mut rr) = (vec![0u64; nodes], vec![0u64; nodes]);
    for (cu, &w) in work.iter().enumerate().take(plan.len()) {
        placed[plan[cu] as usize % nodes] += w;
        rr[cu % nodes] += w;
    }
    (
        placed.into_iter().max().unwrap_or(0),
        rr.into_iter().max().unwrap_or(0),
    )
}

/// Node position for every cu, from the packet's locality domains.
///
/// Requires the domains to divide over the nodes AND the resulting split to leave no node busier
/// than the round-robin would, in EVERY program. That second test is the one that matters, and it
/// is measured, not assumed: a domain is a GPU L2 partition, and nothing makes its slices equal in
/// COST. The blocked map (`cu / sms_per_partition`) is the worst case — low-numbered cus carry every
/// op that is sliced narrowly, so grouping them puts a third of the packet on one node. On a real
/// H100-mapped Gemma-4 blob that is a 3.0-4.1x work spread across nodes against round-robin's
/// 1.04-1.12x, and it measured 1.5x slower end to end. The makespan is set by the busiest node, so
/// comparing peaks is the right test; `None` falls back to the round-robin.
pub fn node_plan(
    cu_dom: &[u32],
    domains: u32,
    nodes: usize,
    work: &[Vec<u64>],
) -> Option<Vec<u32>> {
    if nodes < 2 || (domains as usize) < nodes || !(domains as usize).is_multiple_of(nodes) {
        return None;
    }
    // No work to judge means nothing establishes the plan is safe; `all()` on an empty slice is
    // vacuously true, which would turn "unknown" into "approved".
    if work.is_empty() {
        return None;
    }
    let per = domains as usize / nodes;
    let plan: Vec<u32> = cu_dom.iter().map(|&d| (d as usize / per) as u32).collect();
    work.iter()
        .all(|w| {
            let (placed, rr) = peak_loads(&plan, w, nodes);
            placed <= rr
        })
        .then_some(plan)
}

#[cfg(test)]
mod placement_tests {
    use super::*;

    fn prog(n_cu: u32, l2_domains: u32, dom_of: impl Fn(u32) -> u32) -> DevProg {
        let (mut stream, mut ofs, mut len) = (Vec::new(), Vec::new(), Vec::new());
        for cu in 0..n_cu {
            ofs.push(stream.len() as u32);
            len.push(2);
            for _ in 0..2 {
                stream.push(StreamEnt {
                    flags: (dom_of(cu) as u16) << SE_DOMAIN_SHIFT,
                    ..Default::default()
                });
            }
        }
        DevProg {
            t: 1,
            role: packet::devbuild::ProgramRole::DecodeRung { rows: 1 },
            n_counter: 0,
            insts: Vec::new(),
            stream,
            stream_ofs: ofs,
            stream_len: len,
            waits: Vec::new(),
            succs: Vec::new(),
            gq_stream: Vec::new(),
            gq_seg_ofs: Vec::new(),
            l2_domains,
        }
    }

    /// Both hardware maps the emitter can have used: AMD CDNA round-robin and NVIDIA blocked.
    /// The recovery reads the tagged bits, so it does not need to know which.
    #[test]
    fn recovers_either_hardware_domain_map() {
        for map in [(|cu: u32| cu % 8) as fn(u32) -> u32, |cu: u32| cu / 8] {
            let (dom, domains) = cu_domains(&[prog(64, 8, map)], 64).expect("placed blob");
            assert_eq!(domains, 8);
            assert_eq!(dom, (0..64).map(map).collect::<Vec<_>>());
        }
    }

    /// A blob that expresses no locality must leave placement alone. The legacy layout that put
    /// the domain in `seg` reads as all-zero domain bits, which is the `l2_domains == 0` case.
    #[test]
    fn unplaced_or_single_domain_blob_has_no_plan() {
        assert!(cu_domains(&[prog(64, 0, |cu| cu % 8)], 64).is_none());
        assert!(cu_domains(&[prog(64, 1, |_| 0)], 64).is_none());
        assert!(cu_domains(&[], 64).is_none());
    }

    /// One layout serves a whole build, so placed programs must agree; a blob that disagrees has
    /// no single map to honour. An unplaced program alongside a placed one is fine -- it expresses
    /// no locality, so it is indifferent to where its cus land.
    #[test]
    fn disagreeing_programs_have_no_plan() {
        let progs = vec![prog(64, 8, |cu| cu % 8), prog(64, 8, |cu| cu / 8)];
        assert!(cu_domains(&progs, 64).is_none());
        let mixed = vec![prog(64, 8, |cu| cu % 8), prog(64, 0, |_| 0)];
        assert!(cu_domains(&mixed, 64).is_some());
    }

    #[test]
    fn domain_beyond_the_declared_count_is_rejected() {
        assert!(cu_domains(&[prog(64, 4, |cu| cu % 8)], 64).is_none());
    }

    /// Same-domain cus share a node, and every node keeps an equal share.
    #[test]
    fn node_plan_groups_domains_onto_nodes() {
        let (dom, domains) = cu_domains(&[prog(64, 8, |cu| cu % 8)], 64).unwrap();
        let flat = vec![vec![1u64; 64]];
        let plan = node_plan(&dom, domains, 2, &flat).expect("8 domains divide over 2 nodes");
        for cu in 0..64u32 {
            assert_eq!(plan[cu as usize], dom[cu as usize] / 4);
        }
        for node in 0..2u32 {
            assert_eq!(plan.iter().filter(|&&p| p == node).count(), 32);
        }
    }

    /// The AMD round-robin map is `cu % domains`, so at `domains == nodes` the plan reduces to
    /// `cu % nodes` — exactly the placement it replaces. A real gfx950 blob lands here on an
    /// 8-node host, which is why an A/B of the two placements only says anything at other node
    /// counts; the plan diverges as soon as the counts differ.
    #[test]
    fn an_amd_map_matching_the_node_count_reproduces_the_round_robin() {
        let (dom, domains) = cu_domains(&[prog(304, 8, |cu| cu % 8)], 304).unwrap();
        let flat = vec![vec![1u64; 304]];
        let same = node_plan(&dom, domains, 8, &flat).unwrap();
        assert!(
            (0..304).all(|cu| same[cu] == cu as u32 % 8),
            "identical at 8 nodes"
        );
        let differs = node_plan(&dom, domains, 2, &flat).unwrap();
        assert!(
            (0..304).any(|cu| differs[cu] != cu as u32 % 2),
            "and diverges at 2 nodes, where the A/B is meaningful"
        );
    }

    /// Spreading over every node is the larger measured effect, so a split with fewer domains than
    /// nodes, or one that does not divide, declines rather than trade it away.
    #[test]
    fn node_plan_declines_when_it_cannot_stay_balanced() {
        let (dom, domains) = cu_domains(&[prog(64, 8, |cu| cu % 8)], 64).unwrap();
        let flat = vec![vec![1u64; 64]];
        assert!(
            node_plan(&dom, domains, 3, &flat).is_none(),
            "8 domains, 3 nodes"
        );
        assert!(node_plan(&dom, domains, 1, &flat).is_none(), "single node");
        assert!(
            node_plan(&dom, domains, 16, &flat).is_none(),
            "domains < nodes"
        );
        assert!(
            node_plan(&dom, domains, 8, &flat).is_some(),
            "one domain per node"
        );
    }

    /// The measured failure, reproduced: the blocked map puts every narrowly-sliced op's cus on
    /// one node. Equal cu COUNTS per node say nothing about equal cost, so the guard compares the
    /// busiest node's work against what the round-robin would give it. This shape ran 1.5x slower
    /// end to end on the real blob, and must be declined.
    #[test]
    fn node_plan_declines_a_plan_that_makes_the_busiest_node_worse() {
        // 132 cus, blocked over 8 domains of 18: domain 7 gets 6 cus, and low cus carry more work.
        let (dom, domains) = cu_domains(&[prog(132, 8, |cu| (cu / 18).min(7))], 132).unwrap();
        let skewed = vec![(0..132).map(|cu| 132 - cu as u64).collect::<Vec<u64>>()];
        assert!(
            node_plan(&dom, domains, 8, &skewed).is_none(),
            "declines the skew"
        );
        // The same domains with flat per-cu cost still balance, so the guard is not blanket-off.
        let flat = vec![vec![1u64; 132]];
        let n = node_plan(&dom, domains, 8, &flat);
        assert!(
            n.is_none(),
            "18/18/../6 cus per node is already worse than round-robin"
        );
    }

    /// Programs are ALTERNATIVES, so the balance test has to hold for each separately. Two that
    /// lean opposite ways sum to a perfectly balanced total while each on its own is ruinous —
    /// summing first would approve exactly the placement the guard exists to reject.
    #[test]
    fn a_program_cannot_hide_its_imbalance_behind_another() {
        // 4 cus, 2 domains, 2 nodes: cus 0,1 -> node 0 and cus 2,3 -> node 1 under the plan,
        // against round-robin's 0,2 -> node 0 and 1,3 -> node 1.
        let (dom, domains) = cu_domains(&[prog(4, 2, |cu| cu / 2)], 4).unwrap();
        let a = vec![100u64, 100, 1, 1]; // plan 200/2, round-robin 101/101
        let b = vec![1u64, 1, 100, 100]; // plan 2/200, round-robin 101/101
        for one in [&a, &b] {
            assert!(
                node_plan(&dom, domains, 2, std::slice::from_ref(one)).is_none(),
                "each program alone is a 200-vs-2 split and must be declined"
            );
        }
        assert!(
            node_plan(&dom, domains, 2, &[a.clone(), b.clone()]).is_none(),
            "and together too: their sum balances, but neither program ever runs as the sum"
        );
        // The hole this closes, made explicit: summing first yields a flat 101 per cu, which the
        // peak test then waves through. That was the earlier `cu_work`, and it is why the work is
        // now carried one row per program.
        let summed: Vec<u64> = (0..4).map(|i| a[i] + b[i]).collect();
        assert!(summed.iter().all(|&x| x == 101));
        assert!(
            node_plan(&dom, domains, 2, &[summed]).is_some(),
            "the summed view approves the very placement each program rejects"
        );
        // Sanity that this shape is otherwise acceptable, so the rejection is the imbalance.
        let flat = vec![vec![1u64; 4]];
        assert!(node_plan(&dom, domains, 2, &flat).is_some());
    }

    /// No work rows means nothing established the plan is safe. `all()` over an empty slice is
    /// vacuously true, so this has to be rejected explicitly or "unknown" reads as "approved".
    #[test]
    fn node_plan_declines_with_no_work_to_judge() {
        let (dom, domains) = cu_domains(&[prog(64, 8, |cu| cu % 8)], 64).unwrap();
        assert!(node_plan(&dom, domains, 8, &[]).is_none());
    }

    /// `cu_domains` is `pub` and the probe calls it on a blob that never saw `validate_cpu_blob`,
    /// so a stream window past the end has to decline rather than panic.
    #[test]
    fn cu_domains_declines_a_stream_window_past_the_end() {
        let mut p = prog(8, 2, |cu| cu / 4);
        p.stream_len[3] = 9_999;
        assert!(cu_domains(&[p], 8).is_none());
        let mut q = prog(8, 2, |cu| cu / 4);
        q.stream_ofs[5] = u32::MAX;
        assert!(cu_domains(&[q], 8).is_none());
    }

    /// A worker past `n_cu` owns nothing in any program and only costs its spin, so the auto width
    /// is capped. Measured 8.3x on decode; the explicit override and the global queue are exempt.
    #[test]
    fn worker_width_never_exceeds_the_packet() {
        assert_eq!(worker_width(0, 384, 144, false), 144, "capped at n_cu");
        assert_eq!(
            worker_width(0, 96, 144, false),
            96,
            "no cap needed below n_cu"
        );
        assert_eq!(
            worker_width(0, 384, 144, true),
            384,
            "global queue uses every worker"
        );
        assert_eq!(
            worker_width(192, 384, 144, false),
            192,
            "explicit --cpu-threads wins"
        );
        assert_eq!(worker_width(512, 384, 144, false), 512, "even above n_cu");
        assert_eq!(worker_width(0, 8, 0, false), 1, "never zero workers");
    }

    /// `cu_work` keeps one row per program rather than collapsing them.
    #[test]
    fn cu_work_is_per_program() {
        let progs = vec![prog(4, 2, |cu| cu / 2), prog(4, 2, |cu| cu / 2)];
        let w = cu_work(&progs, 4);
        assert_eq!(w.len(), 2, "one row per program, not one summed row");
        assert!(w.iter().all(|r| r.len() == 4 && r.iter().all(|&x| x == 2)));
    }
}

fn loaded(
    p: &DevProg,
    n_cu: u32,
    per_node: &[Vec<u32>],
    nodes: usize,
    physical: usize,
    logical: usize,
) -> LoadedProgram {
    let n_seg = p
        .stream
        .iter()
        .map(|e| e.seg as u32)
        .max()
        .map_or(1, |m| m + 1);
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
    /// 0 = model-selected topology width (physical cores for MoE, logical CPUs
    /// for dense single-row decode).
    pub threads: usize,
    pub numa: NumaMode,
    pub isa: Isa,
    pub spin_us: u32,
    /// Cores this engine's workers may use. `None` = the live topology.
    ///
    /// Set by the prefill-head pool to its reservation, so `WorkerPool` places
    /// heads inside it with the machinery it already has and the serving path's
    /// cores are simply not in the set it can see.
    pub topology: Option<Topology>,
}

impl Default for CpuEngineOpts {
    fn default() -> Self {
        CpuEngineOpts {
            threads: 0,
            numa: NumaMode::Auto,
            isa: Isa::Amx,
            spin_us: 2000,
            topology: None,
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
            .or_else(|| {
                buckets
                    .iter()
                    .filter(|(_, t)| allowed(*t))
                    .max_by_key(|(_, t)| *t)
            })
            .copied()
    };
    let (prog, t) = pick(&|t| t <= cap)
        .or_else(|| pick(&|_| true))
        .expect("at least one prefill bucket");
    Chunk {
        prog,
        c0,
        clen: rem.min(t),
    }
}

/// A loaded model + its persistent worker pool: single sequence, greedy
/// on-device sampling, the `exec::amd` step protocol on host memory.
pub struct CpuEngine {
    // Field order is the drop order: workers must stop before model allocations.
    pool: WorkerPool,
    model: CpuModel,
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
        let topo = opts.topology.clone().unwrap_or_else(Topology::detect);
        if let NumaMode::Nodes(requested) = &opts.numa {
            if requested.is_empty() || requested.iter().any(|n| !topo.nodes.contains(n)) {
                return Err(RuntimeError::Device(format!(
                    "NUMA nodes {requested:?} unavailable in allowed nodes {:?}",
                    topo.nodes
                )));
            }
        }
        let memory_nodes = match &opts.numa {
            NumaMode::Off => Vec::new(),
            _ => topo.select_nodes(&opts.numa),
        };
        let model = CpuModel::load_on_nodes(
            blob,
            checkpoint,
            &memory_nodes,
            matches!(opts.numa, NumaMode::Nodes(_)),
        )?;
        let n_cu = model.blob.n_cu;
        // The pool needs the Exec before it exists; build the exec against the
        // node placement the pool will use (same rule: round-robin over nodes).
        let nodes = topo.select_nodes(&opts.numa);
        let (physical_w, logical_w);
        let threads = if opts.threads == 0 {
            // Mirrors WorkerPool::spawn's placement list, which orders physical cores first and
            // their SMT siblings after, so `k` threads pin to `k` distinct logical cpus.
            let logical: usize = nodes
                .iter()
                .map(|&n| {
                    topo.cores_on_node(n)
                        .map(|c| c.siblings.len().max(1))
                        .sum::<usize>()
                })
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
            physical_w = nodes
                .iter()
                .map(|&n| topo.cores_on_node(n).count())
                .sum::<usize>()
                .max(1);
            logical_w = nodes
                .iter()
                .map(|&n| {
                    topo.cores_on_node(n)
                        .map(|c| c.siblings.len().max(1))
                        .sum::<usize>()
                })
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
        let threads = worker_width(
            opts.threads,
            threads.max(if wants_logical { logical_w } else { physical_w }),
            n_cu,
            crate::config::RuntimeConfig::get().cpu.gq_opt_in,
        );
        tracing::info!(
            threads,
            ?isa,
            physical_cores = topo.physical_cores(),
            "cpu: worker count"
        );
        let placement = topo.worker_cpus(&nodes);
        let exec = Arc::new(KernelExec::new(&model, threads, |w| {
            placement[w % placement.len()].1
        })?);
        // The packet's locality hint, mapped onto this host's nodes. Absent for an unplaced blob,
        // one node, or domains that do not divide over the nodes: placement stays cu % nodes.
        let cu_dom = crate::config::RuntimeConfig::get()
            .cpu
            .l2_place
            .then(|| cu_domains(&model.blob.progs, n_cu))
            .flatten();
        let plan = cu_dom.as_ref().and_then(|(d, n)| {
            let work = cu_work(&model.blob.progs, n_cu);
            node_plan(d, *n, nodes.len(), &work).map(|p| (p, *n))
        });
        match &plan {
            Some((_, domains)) => tracing::info!(
                domains,
                nodes = nodes.len(),
                "CPU NUMA placement follows the packet's L2 locality domains"
            ),
            None => tracing::debug!(
                nodes = nodes.len(),
                l2_domains = cu_dom.as_ref().map(|(_, n)| *n).unwrap_or(0),
                "CPU NUMA placement is round-robin; no usable packet locality plan"
            ),
        }
        let pool = WorkerPool::spawn(
            &topo,
            threads,
            &opts.numa,
            opts.spin_us,
            n_cu,
            plan.as_ref().map(|(p, n)| (p.as_slice(), *n)),
            exec,
        );
        let progs: Vec<Arc<LoadedProgram>> = model
            .blob
            .progs
            .iter()
            .map(|p| {
                Arc::new(loaded(
                    p,
                    n_cu,
                    pool.per_node(),
                    nodes.len(),
                    physical_w,
                    logical_w,
                ))
            })
            .collect();
        let counters = model
            .blob
            .progs
            .iter()
            .map(|p| Arc::new(CounterPool::with_len(p.n_counter as usize)))
            .collect();
        let max_ctx = model.wk.pos.map(|h| model.tensor(h).bytes / 4).unwrap_or(0);
        tracing::info!(
            threads = pool.threads(),
            n_cu,
            ?isa,
            max_ctx,
            "CPU engine ready"
        );
        Ok(CpuEngine {
            pool,
            model,
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

    /// Prefill buckets as `(program, rows)`, falling back to single-token decode.
    pub fn prefill_buckets(&self) -> Vec<(usize, u32)> {
        if self.model.dec_ix == 0 {
            return vec![(0, 1)];
        }
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
        let end = ch.c0.checked_add(ch.clen);
        let decode_prefill = self.model.dec_ix == 0 && ch.prog == 0 && ch.clen == 1;
        if (ch.prog >= self.model.dec_ix && !decode_prefill)
            || end.is_none_or(|end| end as usize > prompt.len() || end as usize > self.max_ctx)
            || ch.clen == 0
        {
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
                let ids_tensor = self.model.tensor(t_ids);
                let pos_tensor = self.model.tensor(t_pos);
                let ids = unsafe {
                    std::slice::from_raw_parts_mut(ids_tensor.as_ptr(), ids_tensor.bytes)
                };
                let pos = unsafe {
                    std::slice::from_raw_parts_mut(pos_tensor.as_ptr(), pos_tensor.bytes)
                };
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
            if decode_prefill {
                // Decode reads the absolute position from in.pos; its KV pointers are
                // already rebased to the prefill slot by the caller.
                return self.run_prog(ch.prog);
            }
            // Rebase the program from its pristine copy: KV write rows at c0,
            // flash window [c0, c0+clen), row counts for a partial chunk.
            let lp = Arc::make_mut(&mut self.progs[ch.prog]);
            lp.insts
                .copy_from_slice(&self.model.blob.progs[ch.prog].insts);
            rebase_chunk_rows(&mut lp.insts, &self.model.names, ch.c0, ch.clen, t, Some(t));
            if place_lm_head_row(&mut lp.insts, self.model.wk.logits, ch.clen - 1).is_none()
                && self.model.wk.logits.is_some()
            {
                tracing::warn!(
                    prog = ch.prog,
                    "act.logits declared but no matmul writes it"
                );
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
    pub fn decode_step_batched(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        ids: &[u32],
    ) -> Result<Vec<u32>> {
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
        if let Some(&k) = kvlen.iter().find(|&&k| k == 0 || k as usize > self.max_ctx) {
            return Err(RuntimeError::Device(format!(
                "KV length {k} outside 1..={}",
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
            return Err(RuntimeError::Device(format!(
                "program {dp} is not a decode rung"
            )));
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
            let ids_tensor = self.model.tensor(t_ids);
            let pos_tensor = self.model.tensor(t_pos);
            let kv_tensor = self.model.tensor(t_kvlen);
            let s_ids = std::slice::from_raw_parts_mut(ids_tensor.as_ptr(), ids_tensor.bytes);
            let s_pos = std::slice::from_raw_parts_mut(pos_tensor.as_ptr(), pos_tensor.bytes);
            let s_kv = std::slice::from_raw_parts_mut(kv_tensor.as_ptr(), kv_tensor.bytes);
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
