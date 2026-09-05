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
        let align = if bytes >= HUGE { HUGE } else { ALIGN };
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
        if bytes >= HUGE {
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
    /// `tensors[h].ptr` as the flat table every kernel indexes by handle.
    ptrs: Vec<*mut c_void>,
    pub names: Vec<String>,
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
        let ckpt = Checkpoint::open(checkpoint)?;

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
            if packet::names::is_host_filled_table(&td.name) {
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
        let ptrs: Vec<*mut c_void> = tensors.iter().map(|t| t.as_ptr() as *mut c_void).collect();

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
            ptrs,
            names,
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
    pub fn tensor_table(&self) -> &[*mut c_void] {
        &self.ptrs
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
