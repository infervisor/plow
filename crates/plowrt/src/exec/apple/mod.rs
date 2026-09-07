//! Apple Silicon GPU engine: the device-ISA interpreter as a Metal compute kernel
//! (`runtime/apple/interp.metal`), driven the way `exec::amd` drives gfx950 — n_seg
//! dispatches per program, one command buffer per program run, host sync at the end.
//! See `plans/apple-silicon-backend.md` §4.2 and §9 (probe results that fixed this shape).
//!
//! Unified memory does the heavy lifting: the model is loaded by the CPU engine's loader
//! ([`CpuModel`]) into page-aligned host tensors, and every tensor is wrapped as an
//! `MTLBuffer` with `newBufferWithBytesNoCopy` — the host tensors ARE the device buffers.
//! Kernels reach tensors through a table of GPU addresses indexed by handle, exactly like
//! the CPU kernels' pointer table. Programs (instructions, streams, wait/succ tables) are
//! uploaded once; the per-step dynamic surface (KV rows, chunk rebases) is patched into the
//! shared instruction buffer before each dispatch.
//!
//! Cross-unit sync is Event mode (probe (a)): the host never spins on GPU state mid-kernel;
//! it reads the sampled token after `waitUntilCompleted`.

use std::ffi::c_void;
use std::path::Path;
use std::ptr::NonNull;
use std::time::Instant;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AllocAnyThread;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode,
    MTLResidencySet, MTLResidencySetDescriptor, MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use packet::dev::DevInst64;

use crate::exec::cpu::engine::{plan_chunks, Chunk, CpuModel};
use crate::exec::cpu::ffi::{self, Isa};
use crate::exec::kvrow::{place_lm_head_row, rebase_chunk_rows};
use crate::{Result, RuntimeError};

const MSL: &str = include_str!("../../../../../runtime/apple/interp.metal");
const THREADS: usize = 256;
const PAGE: usize = 16384;

type Buf = Retained<ProtocolObject<dyn MTLBuffer>>;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    seg: u32,
    n_cu: u32,
    spin_max: u32,
    inst_lo: u32,
    inst_hi: u32,
}

/// One instruction of one program delegated to the Apple Neural Engine (rung 4): a prefill
/// GEMM whose weights were baked into a CoreML program at load. Runs on the host thread at a
/// segment boundary of the GPU walk; see `run_prog`.
#[cfg(feature = "ane")]
struct AneSlot {
    prog: usize,
    inst: usize,
    gemm: crate::exec::ane::AneGemm,
    x: Vec<f32>,
    y: Vec<f32>,
    /// Successor counters of every stream entry of `inst`, bumped by the host after the ANE run.
    succ_bumps: Vec<u32>,
    pub last_ms: f64,
}

struct ProgGpu {
    /// Host copy the step driver patches (`rebase_chunk_rows`, KV rows); copied into
    /// `insts` before every dispatch.
    insts_host: Vec<DevInst64>,
    insts: Buf,
    stream: Buf,
    ofs: Buf,
    len: Buf,
    waits: Buf,
    succs: Buf,
    ctr: Buf,
    n_counter: usize,
    n_seg: u32,
}

pub struct MetalEngine {
    pub model: CpuModel,
    _device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_single: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// `PLOW_METAL_SERIAL=1`: one instruction per dispatch (diagnostic, see `plow_single`).
    serial: bool,
    resset: Option<Retained<ProtocolObject<dyn MTLResidencySet>>>,
    /// One buffer per tensor handle: a no-copy view of the host tensor, or (when the
    /// allocation could not be wrapped) a shared copy the host reaches through [`Self::host_ptr`].
    bufs: Vec<Buf>,
    copied: Vec<bool>,
    tab: Buf,
    fault: Buf,
    progs: Vec<ProgGpu>,
    max_ctx: usize,
    spin_max: u32,
    pub last_run_us: f64,
    pub gpu_name: String,
    /// ANE-delegated instructions, ascending by (prog, inst).
    #[cfg(feature = "ane")]
    ane: Vec<AneSlot>,
    /// `(ops run on the ANE, summed ANE ms)` for the last program run that used it.
    pub last_ane: Option<(usize, f64)>,
}

// SAFETY: Metal and CoreML objects are thread-safe per Apple's documentation, and the serve
// layer serializes every call through one lock; the engine is moved to the engine thread once.
unsafe impl Send for MetalEngine {}

impl crate::serve::cpu_serve::SlotEngine for MetalEngine {
    fn prefill_slot(&mut self, slot: usize, prompt: &[u32]) -> Result<u32> {
        MetalEngine::prefill_slot(self, slot, prompt)
    }
    fn prefill_slot_chunk(&mut self, slot: usize, prompt: &[u32], ch: Chunk) -> Result<()> {
        MetalEngine::prefill_slot_chunk(self, slot, prompt, ch)
    }
    fn last_token(&self) -> Result<u32> {
        MetalEngine::last_token(self)
    }
    fn decode_step_batched_at(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        ids: &[u32],
        dp: usize,
    ) -> Result<Vec<u32>> {
        MetalEngine::decode_step_batched_at(self, pos, kvlen, ids, dp)
    }
    fn model(&self) -> &CpuModel {
        &self.model
    }
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    fn describe(&self) -> String {
        format!("metal gpu={} n_cu={}", self.gpu_name, self.model.blob.n_cu)
    }
}

fn err<E: std::fmt::Display>(what: &str, e: E) -> RuntimeError {
    RuntimeError::Device(format!("metal: {what}: {e}"))
}

impl MetalEngine {
    pub fn load(blob: &Path, checkpoint: &Path) -> Result<MetalEngine> {
        // The loader resolves CPU kernels per program (an ABI check); the table must exist.
        ffi::init(Isa::Amx)?;
        let model = CpuModel::load(blob, checkpoint)?;
        let t0 = Instant::now();
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| RuntimeError::Device("metal: no default device".into()))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| RuntimeError::Device("metal: no command queue".into()))?;
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Safe);
        // The default language version follows the SDK the binary was linked against (the nix
        // SDK is older); `atomic_thread_fence` with a device scope needs MSL 3.2.
        opts.setLanguageVersion(MTLLanguageVersion::Version3_2);
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(MSL), Some(&opts))
            .map_err(|e| err("MSL compile", e))?;
        let func = lib
            .newFunctionWithName(&NSString::from_str("plow_interp"))
            .ok_or_else(|| RuntimeError::Device("metal: plow_interp missing".into()))?;
        let pso = device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| err("pipeline", e))?;
        let func_single = lib
            .newFunctionWithName(&NSString::from_str("plow_single"))
            .ok_or_else(|| RuntimeError::Device("metal: plow_single missing".into()))?;
        let pso_single = device
            .newComputePipelineStateWithFunction_error(&func_single)
            .map_err(|e| err("pipeline single", e))?;
        let serial = std::env::var("PLOW_METAL_SERIAL").map_or(false, |v| v == "1");

        // Tensors: wrap the host allocations; fall back to a shared copy.
        let n = model.names.len();
        let mut bufs = Vec::with_capacity(n);
        let mut copied = vec![false; n];
        let mut wrapped = 0usize;
        for h in 0..n {
            let t = model.tensor(h);
            let ptr = t.as_ptr();
            let len = t.bytes.max(1).next_multiple_of(PAGE);
            let nocopy = if (ptr as usize) % PAGE == 0 {
                // SAFETY: the allocation is page-aligned and at least `len` bytes (HostTensor
                // rounds to whole pages on macOS); it outlives the buffer (`model` is owned).
                unsafe {
                    device.newBufferWithBytesNoCopy_length_options_deallocator(
                        NonNull::new(ptr as *mut c_void).expect("tensor ptr"),
                        len,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
                }
            } else {
                None
            };
            let b = match nocopy {
                Some(b) => {
                    wrapped += 1;
                    b
                }
                None => {
                    let b = device
                        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
                        .ok_or_else(|| RuntimeError::Oom(format!("metal buffer {len} B")))?;
                    // SAFETY: both regions are at least `bytes` long.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            ptr,
                            b.contents().as_ptr() as *mut u8,
                            t.bytes,
                        )
                    };
                    copied[h] = true;
                    b
                }
            };
            bufs.push(b);
        }
        let tab_host: Vec<u64> = bufs.iter().map(|b| b.gpuAddress()).collect();
        let tab = shared_from(&device, bytes_of(&tab_host))?;
        let fault = shared_from(&device, &[0u8; 64])?;

        // Programs.
        let mut progs = Vec::with_capacity(model.blob.progs.len());
        for p in &model.blob.progs {
            let n_seg = p
                .stream
                .iter()
                .map(|e| e.seg as u32)
                .max()
                .map_or(1, |m| m + 1);
            progs.push(ProgGpu {
                insts_host: p.insts.clone(),
                insts: shared_from(&device, bytes_of(&p.insts))?,
                stream: shared_from(&device, bytes_of(&p.stream))?,
                ofs: shared_from(&device, bytes_of(&p.stream_ofs))?,
                len: shared_from(&device, bytes_of(&p.stream_len))?,
                waits: shared_from(&device, bytes_of(&p.waits))?,
                succs: shared_from(&device, bytes_of(&p.succs))?,
                ctr: shared_from(&device, &vec![0u8; (p.n_counter as usize).max(1) * 4])?,
                n_counter: p.n_counter as usize,
                n_seg,
            });
        }

        // Residency: every buffer the kernels reach by address, once, on the queue.
        let resset = {
            let desc = MTLResidencySetDescriptor::init(MTLResidencySetDescriptor::alloc());
            match device.newResidencySetWithDescriptor_error(&desc) {
                Ok(set) => {
                    for b in &bufs {
                        set.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&**b));
                    }
                    for pg in &progs {
                        for b in [
                            &pg.insts, &pg.stream, &pg.ofs, &pg.len, &pg.waits, &pg.succs, &pg.ctr,
                        ] {
                            set.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&**b));
                        }
                    }
                    set.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*tab));
                    set.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&*fault));
                    set.commit();
                    set.requestResidency();
                    queue.addResidencySet(&set);
                    Some(set)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "metal: no residency set; using per-encoder useResource");
                    None
                }
            }
        };

        let max_ctx = model.wk.pos.map(|h| model.tensor(h).bytes / 4).unwrap_or(0);
        let gpu_name = device.name().to_string();
        // The blob's executor count must equal this part's GPU cores (§9: a larger grid only adds
        // spinning threadgroups). The canonical hwspec names equal `MTLDevice.name`.
        match hwspec::registry::lookup(&gpu_name) {
            Some(spec) if spec.sm_count != model.blob.n_cu => tracing::warn!(
                gpu = %gpu_name, cores = spec.sm_count, n_cu = model.blob.n_cu,
                "blob executor count differs from this GPU's core count; emit with --gpu {:?}",
                spec.name
            ),
            Some(_) => {}
            None => {
                tracing::warn!(gpu = %gpu_name, "no hwspec entry for this GPU; add one under crates/hwspec/src/apple")
            }
        }
        tracing::info!(
            gpu = %gpu_name,
            tensors = n,
            wrapped,
            copied = n - wrapped,
            programs = progs.len(),
            n_cu = model.blob.n_cu,
            max_ctx,
            setup_ms = format!("{:.0}", t0.elapsed().as_secs_f64() * 1e3).as_str(),
            "metal engine ready"
        );
        #[cfg(feature = "ane")]
        let ane = Self::ane_slots(&model)?;
        Ok(MetalEngine {
            #[cfg(feature = "ane")]
            ane,
            last_ane: None,
            model,
            _device: device,
            queue,
            pso,
            pso_single,
            serial,
            resset,
            bufs,
            copied,
            tab,
            fault,
            progs,
            max_ctx,
            spin_max: 1 << 26,
            last_run_us: 0.0,
            gpu_name,
        })
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// `PLOW_ANE=1|<n>|all`: delegate the first n (or every) eligible GEMM of the 128-row
    /// prefill bucket to the Neural Engine — fp8 (op 33, no activation scale) or bf16 (op 8, no
    /// bias). `PLOW_ANE=<prog>:<inst>` names one instruction. Weights are dequantized to fp16
    /// once at load (option B of §2.3); each op is its own CoreML program, cached on disk.
    #[cfg(feature = "ane")]
    fn ane_slots(model: &CpuModel) -> Result<Vec<AneSlot>> {
        use crate::exec::ane::{f32_to_f16, AneGemm};
        use packet::dev::TENSOR_NONE16;
        let Ok(spec) = std::env::var("PLOW_ANE") else {
            return Ok(Vec::new());
        };
        if spec.is_empty() || spec == "0" {
            return Ok(Vec::new());
        }
        let eligible = |d: &DevInst64| {
            (d.op == 33 && d.t[3] == TENSOR_NONE16 && d.t[4] != TENSOR_NONE16)
                || (d.op == 8 && d.t[7] == TENSOR_NONE16)
        };
        let mut picks: Vec<(usize, usize)> = Vec::new();
        if let Some((a, b)) = spec.split_once(':') {
            picks.push((
                a.parse::<usize>()
                    .map_err(|e| RuntimeError::Device(format!("PLOW_ANE: {e}")))?,
                b.parse::<usize>()
                    .map_err(|e| RuntimeError::Device(format!("PLOW_ANE: {e}")))?,
            ));
        } else {
            let limit = if spec == "all" {
                usize::MAX
            } else {
                spec.parse::<usize>()
                    .map_err(|e| RuntimeError::Device(format!("PLOW_ANE: {e}")))?
            };
            // The narrowest prefill bucket: what a short prompt runs.
            let p = 0usize;
            picks.extend(
                model.blob.progs[p]
                    .insts
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| eligible(d))
                    .map(|(i, _)| (p, i))
                    .take(limit),
            );
        }
        let e4m3 = |c: u8| -> f32 {
            let e = (c >> 3) & 15;
            let m = c & 7;
            let v = if e == 0 {
                m as f32 * 0.001953125
            } else {
                f32::from_bits((((e as u32) + 120) << 23) | ((m as u32) << 20))
            };
            if c & 0x80 != 0 {
                -v
            } else {
                v
            }
        };
        let dir = std::env::temp_dir().join("plow-ane");
        let t0 = Instant::now();
        let mut slots = Vec::with_capacity(picks.len());
        for (prog, inst) in picks {
            let d = model.blob.progs[prog].insts[inst];
            let (t, n, k) = (
                model.blob.progs[prog].t as usize,
                d.i[1] as usize,
                d.i[2] as usize,
            );
            let wbytes = unsafe { model.tensor(d.t[2] as usize).as_slice() };
            let mut w16 = vec![0u16; n * k];
            match d.op {
                33 => {
                    let ws = unsafe { model.tensor(d.t[4] as usize).as_slice() };
                    for r in 0..n {
                        let sc = f32::from_le_bytes(ws[r * 4..r * 4 + 4].try_into().unwrap());
                        for c in 0..k {
                            w16[r * k + c] = f32_to_f16(e4m3(wbytes[r * k + c]) * sc);
                        }
                    }
                }
                8 => {
                    for i in 0..n * k {
                        let b = f32::from_bits(
                            (u16::from_le_bytes([wbytes[2 * i], wbytes[2 * i + 1]]) as u32) << 16,
                        );
                        w16[i] = f32_to_f16(b);
                    }
                }
                other => {
                    return Err(RuntimeError::Device(format!(
                        "PLOW_ANE: instruction {inst} is op {other}, not a GEMM"
                    )))
                }
            }
            let name = format!(
                "{}_p{prog}_i{inst}_{t}x{k}x{n}",
                model.names[d.t[2] as usize].replace(['/', '.'], "_")
            );
            let gemm = AneGemm::new(
                &dir,
                &name,
                t,
                k,
                n,
                &w16,
                objc2_core_ml::MLComputeUnits::CPUAndNeuralEngine,
            )?;
            let p = &model.blob.progs[prog];
            let mut succ_bumps = Vec::new();
            for e in p.stream.iter().filter(|e| e.inst as usize == inst) {
                succ_bumps.extend_from_slice(
                    &p.succs[e.succ_ofs as usize..e.succ_ofs as usize + e.succ_len as usize],
                );
            }
            slots.push(AneSlot {
                prog,
                inst,
                gemm,
                x: vec![0.0; t * k],
                y: vec![0.0; t * n],
                succ_bumps,
                last_ms: 0.0,
            });
        }
        slots.sort_by_key(|s| (s.prog, s.inst));
        tracing::info!(
            ops = slots.len(),
            load_ms = format!("{:.0}", t0.elapsed().as_secs_f64() * 1e3).as_str(),
            "ANE slots ready"
        );
        Ok(slots)
    }

    /// Run the ANE slot's GEMM on the current instruction operands and publish its counters.
    #[cfg(feature = "ane")]
    fn run_ane(&mut self, p: usize, si: usize) -> Result<()> {
        let inst = self.ane[si].inst;
        let d = self.progs[p].insts_host[inst];
        let (n, k) = (d.i[1] as usize, d.i[2] as usize);
        let a_row0 = d.i[4] as usize;
        let c_row0 = d.i[5] as usize;
        let (ha, hc) = (d.t[1] as usize, d.t[0] as usize);
        let slot = &mut self.ane[si];
        let t = slot.gemm.t;
        // SAFETY: quiescent between command buffers; the tensors hold (a_row0 + t) x K and (c_row0 + t) x N.
        unsafe {
            let a = self.model.tensor(ha).as_ptr().add(a_row0 * k * 2) as *const u16;
            for i in 0..t * k {
                slot.x[i] = f32::from_bits((std::ptr::read_unaligned(a.add(i)) as u32) << 16);
            }
        }
        slot.gemm.run(&slot.x, &mut slot.y)?;
        unsafe {
            let c = self.model.tensor(hc).as_ptr().add(c_row0 * n * 2) as *mut u16;
            for i in 0..t * n {
                let f = slot.y[i];
                let u = f.to_bits();
                let bf = if (u & 0x7f80_0000) == 0x7f80_0000 {
                    ((u >> 16) | if u & 0xffff != 0 { 0x40 } else { 0 }) as u16
                } else {
                    ((u.wrapping_add(0x7fff + ((u >> 16) & 1))) >> 16) as u16
                };
                std::ptr::write_unaligned(c.add(i), bf);
            }
            let ctr = self.progs[p].ctr.contents().as_ptr() as *mut u32;
            for &s in &slot.succ_bumps {
                *ctr.add(s as usize) += 1;
            }
        }
        slot.last_ms = slot.gemm.last_ms;
        let ms = slot.last_ms;
        self.last_ane = Some(match self.last_ane {
            Some((c, acc)) if c != usize::MAX => (c + 1, acc + ms),
            _ => (1, ms),
        });
        Ok(())
    }

    /// Host view of tensor `h` (the shared buffer's memory: identical to the host tensor
    /// unless the allocation had to be copied).
    pub fn host_ptr(&self, h: usize) -> *mut u8 {
        if self.copied[h] {
            self.bufs[h].contents().as_ptr() as *mut u8
        } else {
            self.model.tensor(h).as_ptr()
        }
    }

    /// Host view of tensor `h` as bytes (quiescent between runs).
    pub fn tensor_bytes(&self, h: usize) -> &[u8] {
        // SAFETY: the buffer is at least `bytes` long and no run is in flight.
        unsafe { std::slice::from_raw_parts(self.host_ptr(h), self.model.tensor(h).bytes) }
    }

    pub fn write_u32s(&self, h: usize, at: usize, vals: &[u32]) {
        let p = self.host_ptr(h);
        // SAFETY: quiescent between runs; the tensor holds at least (at + len) u32s by blob construction.
        unsafe {
            for (i, v) in vals.iter().enumerate() {
                std::ptr::write_unaligned(p.add((at + i) * 4) as *mut u32, *v);
            }
        }
    }

    pub fn read_u32(&self, h: usize, at: usize) -> u32 {
        // SAFETY: as `write_u32s`.
        unsafe { std::ptr::read_unaligned(self.host_ptr(h).add(at * 4) as *const u32) }
    }

    fn need(&self, h: Option<usize>, what: &str) -> Result<usize> {
        h.ok_or_else(|| RuntimeError::Device(format!("blob declares no `{what}` tensor")))
    }

    /// Copy the patched instructions in, zero the counters, dispatch every segment, wait.
    fn run_prog(&mut self, p: usize) -> Result<()> {
        let t0 = Instant::now();
        let pg = &self.progs[p];
        // SAFETY: shared buffers sized at load for exactly these tables; no run in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(
                pg.insts_host.as_ptr() as *const u8,
                pg.insts.contents().as_ptr() as *mut u8,
                std::mem::size_of_val(pg.insts_host.as_slice()),
            );
            std::ptr::write_bytes(
                pg.ctr.contents().as_ptr() as *mut u8,
                0,
                pg.n_counter.max(1) * 4,
            );
            std::ptr::write_bytes(self.fault.contents().as_ptr() as *mut u8, 0, 64);
        }
        #[cfg(feature = "ane")]
        {
            let mine: Vec<usize> = (0..self.ane.len())
                .filter(|&i| self.ane[i].prog == p)
                .collect();
            if !mine.is_empty() && !self.serial {
                // Event mode: GPU segments between consecutive ANE instructions, the host runs
                // each ANE op and publishes its counters in between.
                self.last_ane = None;
                let n = self.progs[p].insts_host.len() as u32;
                let mut lo = 0u32;
                for si in mine {
                    let i = self.ane[si].inst as u32;
                    self.dispatch_range(p, lo, i)?;
                    self.run_ane(p, si)?;
                    lo = i + 1;
                }
                self.dispatch_range(p, lo, n)?;
                self.last_run_us = t0.elapsed().as_secs_f64() * 1e6;
                return self.check_fault(p);
            }
        }
        self.dispatch_range(p, 0, u32::MAX)?;
        self.last_run_us = t0.elapsed().as_secs_f64() * 1e6;
        self.check_fault(p)
    }

    /// Read the fault word left by the kernels of program `p`'s last command buffer.
    fn check_fault(&self, p: usize) -> Result<()> {
        // SAFETY: 64-byte shared buffer written by the kernel before completion.
        let f = unsafe { std::ptr::read_volatile(self.fault.contents().as_ptr() as *const u32) };
        if f != 0 {
            let inst = f & 0x3FFF_FFFF;
            let op = self.progs[p]
                .insts_host
                .get(inst as usize)
                .map(|d| d.op)
                .unwrap_or(0);
            let why = if f & 0x8000_0000 != 0 {
                "wait timed out"
            } else {
                "unimplemented op"
            };
            return Err(RuntimeError::Device(format!(
                "metal: program {p}: {why} at instruction {inst} (op {op} {})",
                packet::dev::DevOp::from_u16(op)
                    .map(|o| o.c_name())
                    .unwrap_or("?")
            )));
        }
        Ok(())
    }

    /// One command buffer running every segment of program `p` restricted to instructions
    /// `[inst_lo, inst_hi)`; waits for completion.
    fn dispatch_range(&mut self, p: usize, inst_lo: u32, inst_hi: u32) -> Result<()> {
        let pg = &self.progs[p];
        let n_cu = self.model.blob.n_cu as usize;
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| RuntimeError::Device("metal: no command buffer".into()))?;
        if self.serial {
            // Topological instruction order (the builder appends ops in dependency order).
            for (ii, d) in pg.insts_host.iter().enumerate() {
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or_else(|| RuntimeError::Device("metal: no encoder".into()))?;
                enc.setComputePipelineState(&self.pso_single);
                let sp = ii as u32;
                // SAFETY: bindings match `plow_single`; `sp` outlives the call.
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&pg.insts), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.tab), 0, 7);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&sp as *const u32 as *mut c_void).unwrap(),
                        4,
                        8,
                    );
                    enc.setBuffer_offset_atIndex(Some(&self.fault), 0, 9);
                }
                if self.resset.is_none() {
                    for b in &self.bufs {
                        enc.useResource_usage(
                            ProtocolObject::from_ref(&**b),
                            MTLResourceUsage::Read | MTLResourceUsage::Write,
                        );
                    }
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: d.blocks as usize,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: THREADS,
                        height: 1,
                        depth: 1,
                    },
                );
                enc.endEncoding();
            }
        }
        for seg in 0..(if self.serial { 0 } else { pg.n_seg }) {
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| RuntimeError::Device("metal: no encoder".into()))?;
            enc.setComputePipelineState(&self.pso);
            let bind = [
                &pg.insts, &pg.stream, &pg.ofs, &pg.len, &pg.waits, &pg.succs, &pg.ctr, &self.tab,
            ];
            for (i, b) in bind.iter().enumerate() {
                // SAFETY: buffer bindings 0..8 match the kernel signature.
                unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, i) };
            }
            let params = Params {
                seg,
                n_cu: n_cu as u32,
                spin_max: self.spin_max,
                inst_lo,
                inst_hi,
            };
            // SAFETY: `params` outlives the call; Metal copies the bytes.
            unsafe {
                enc.setBytes_length_atIndex(
                    NonNull::new(&params as *const Params as *mut c_void).unwrap(),
                    std::mem::size_of::<Params>(),
                    8,
                );
                enc.setBuffer_offset_atIndex(Some(&self.fault), 0, 9);
            }
            if self.resset.is_none() {
                for b in &self.bufs {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&**b),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                }
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: n_cu,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: THREADS,
                    height: 1,
                    depth: 1,
                },
            );
            enc.endEncoding();
        }
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() != MTLCommandBufferStatus::Completed {
            let e = cb.error().map(|e| e.to_string()).unwrap_or_default();
            return Err(RuntimeError::Device(format!(
                "metal: program {p} command buffer status {:?}: {e}",
                cb.status()
            )));
        }
        Ok(())
    }

    /// The compiled prefill buckets as `(program, rows)`.
    pub fn prefill_buckets(&self) -> Vec<(usize, u32)> {
        (0..self.model.dec_ix)
            .map(|i| (i, self.model.blob.progs[i].t))
            .collect()
    }

    /// Prefill `prompt` into KV rows `[0, len)`; returns the greedy next token.
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
        if prompt.is_empty() || prompt.len() > self.max_ctx {
            return Err(RuntimeError::Device(format!(
                "prompt of {} tokens outside 1..={}",
                prompt.len(),
                self.max_ctx
            )));
        }
        let buckets = self.prefill_buckets();
        if buckets.is_empty() {
            return Err(RuntimeError::Device("blob has no prefill program".into()));
        }
        for ch in plan_chunks(&buckets, prompt.len() as u32) {
            self.prefill_chunk(prompt, ch)?;
        }
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        Ok(self.read_u32(t_ids, 0))
    }

    /// Stage a prefill chunk (inputs + instruction rebases) without running it.
    pub fn prepare_prefill_chunk(&mut self, prompt: &[u32], ch: Chunk) -> Result<()> {
        let end = ch.c0.checked_add(ch.clen);
        if ch.prog >= self.model.dec_ix
            || end.is_none_or(|e| e as usize > prompt.len())
            || ch.clen == 0
        {
            return Err(RuntimeError::Device(format!("bad prefill chunk {ch:?}")));
        }
        let (t_ids, t_pos, t_kvlen) = (
            self.need(self.model.wk.ids, "in.ids")?,
            self.need(self.model.wk.pos, "in.pos")?,
            self.need(self.model.wk.kvlen, "in.kvlen")?,
        );
        let t = self.model.blob.progs[ch.prog].t;
        let ids: Vec<u32> = (0..t)
            .map(|i| {
                if i < ch.clen {
                    prompt[(ch.c0 + i) as usize]
                } else {
                    0
                }
            })
            .collect();
        let pos: Vec<u32> = (0..t).map(|i| ch.c0 + i).collect();
        self.write_u32s(t_ids, 0, &ids);
        self.write_u32s(t_pos, 0, &pos);
        self.write_u32s(t_kvlen, 0, &[ch.c0 + ch.clen]);
        let pristine = self.model.blob.progs[ch.prog].insts.clone();
        let logits = self.model.wk.logits;
        let names = self.model.names.clone();
        let pg = &mut self.progs[ch.prog];
        pg.insts_host.copy_from_slice(&pristine);
        rebase_chunk_rows(&mut pg.insts_host, &names, ch.c0, ch.clen, t, Some(t));
        if place_lm_head_row(&mut pg.insts_host, logits, ch.clen - 1).is_none() && logits.is_some()
        {
            tracing::warn!(
                prog = ch.prog,
                "act.logits declared but no matmul writes it"
            );
        }
        Ok(())
    }

    pub fn prefill_chunk(&mut self, prompt: &[u32], ch: Chunk) -> Result<()> {
        self.prepare_prefill_chunk(prompt, ch)?;
        self.run_prog(ch.prog)
    }

    /// Stage a decode step's inputs and instruction patches without running it; returns the
    /// decode program index.
    pub fn prepare_decode(&mut self, pos: u32, kvlen: u32, id: u32) -> Result<usize> {
        let b = self.model.batch;
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        let t_pos = self.need(self.model.wk.pos, "in.pos")?;
        let t_kvlen = self.need(self.model.wk.kvlen, "in.kvlen")?;
        let dp = self.model.decode_prog_for(1);
        let mut ps = vec![0u32; b];
        let mut ks = vec![1u32; b];
        ps[0] = pos;
        ks[0] = kvlen;
        self.write_u32s(t_ids, 0, &[id]);
        self.write_u32s(t_pos, 0, &ps);
        self.write_u32s(t_kvlen, 0, &ks);
        if b == 1 {
            let kvrow = self.model.kvrow.clone();
            let pg = &mut self.progs[dp];
            for &i in &kvrow {
                pg.insts_host[i as usize].i[3] = pos;
            }
        }
        Ok(dp)
    }

    /// The (patched) instructions of program `p`.
    pub fn insts_host(&self, p: usize) -> &[DevInst64] {
        &self.progs[p].insts_host
    }

    /// Diagnostic: run ONE instruction of program `p` (all its slices) as its own dispatch.
    pub fn run_inst(&mut self, p: usize, i: usize) -> Result<()> {
        let pg = &self.progs[p];
        let d = pg.insts_host[i];
        // SAFETY: shared buffers sized at load; no run in flight.
        unsafe {
            std::ptr::copy_nonoverlapping(
                pg.insts_host.as_ptr() as *const u8,
                pg.insts.contents().as_ptr() as *mut u8,
                std::mem::size_of_val(pg.insts_host.as_slice()),
            );
            std::ptr::write_bytes(self.fault.contents().as_ptr() as *mut u8, 0, 64);
        }
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| RuntimeError::Device("metal: no command buffer".into()))?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or_else(|| RuntimeError::Device("metal: no encoder".into()))?;
        enc.setComputePipelineState(&self.pso_single);
        let sp = i as u32;
        // SAFETY: bindings match `plow_single`; `sp` outlives the call.
        unsafe {
            enc.setBuffer_offset_atIndex(Some(&pg.insts), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&self.tab), 0, 7);
            enc.setBytes_length_atIndex(
                NonNull::new(&sp as *const u32 as *mut c_void).unwrap(),
                4,
                8,
            );
            enc.setBuffer_offset_atIndex(Some(&self.fault), 0, 9);
        }
        if self.resset.is_none() {
            for b in &self.bufs {
                enc.useResource_usage(
                    ProtocolObject::from_ref(&**b),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
        }
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (d.blocks as usize).max(1),
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        if cb.status() != MTLCommandBufferStatus::Completed {
            return Err(RuntimeError::Device(format!(
                "metal: run_inst {p}/{i}: status {:?}",
                cb.status()
            )));
        }
        let f = unsafe { std::ptr::read_volatile(self.fault.contents().as_ptr() as *const u32) };
        if f != 0 {
            return Err(RuntimeError::Device(format!(
                "metal: run_inst {p}/{i}: unimplemented op {}",
                d.op
            )));
        }
        Ok(())
    }

    /// One greedy decode step of slot 0: embed the token in `in.ids[0]` at `pos`, attend over
    /// `kvlen` rows; returns the next token (also left in `in.ids[0]`).
    pub fn decode_step(&mut self, pos: u32, kvlen: u32) -> Result<u32> {
        let b = self.model.batch;
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        let t_pos = self.need(self.model.wk.pos, "in.pos")?;
        let t_kvlen = self.need(self.model.wk.kvlen, "in.kvlen")?;
        if pos as usize >= self.max_ctx || kvlen == 0 || kvlen as usize > self.max_ctx {
            return Err(RuntimeError::Device(format!(
                "decode pos {pos} / kvlen {kvlen} outside max_ctx {}",
                self.max_ctx
            )));
        }
        let dp = self.model.decode_prog_for(1);
        let mut ps = vec![0u32; b];
        let mut ks = vec![1u32; b];
        ps[0] = pos;
        ks[0] = kvlen;
        self.write_u32s(t_pos, 0, &ps);
        self.write_u32s(t_kvlen, 0, &ks);
        if b == 1 {
            let kvrow = self.model.kvrow.clone();
            let pg = &mut self.progs[dp];
            for &i in &kvrow {
                pg.insts_host[i as usize].i[3] = pos;
            }
        }
        self.run_prog(dp)?;
        Ok(self.read_u32(t_ids, 0))
    }

    /// Re-derive the GPU address table from the model's (possibly slot-rebased) pointer table:
    /// every entry is the tensor's buffer address plus the same offset the CPU kernels see.
    fn refresh_tab(&mut self) {
        let n = self.bufs.len();
        // SAFETY: `tab` holds `n` u64s; quiescent between runs.
        unsafe {
            let t = self.tab.contents().as_ptr() as *mut u64;
            for h in 0..n {
                let base = self.model.tensor(h).as_ptr() as usize;
                let cur = self.model.table_ptr(h) as usize;
                *t.add(h) = self.bufs[h].gpuAddress() + (cur.wrapping_sub(base)) as u64;
            }
        }
    }

    /// Prefill `prompt` into slot `slot`'s KV block (rebase, run, restore — as the CPU engine).
    pub fn prefill_slot(&mut self, slot: usize, prompt: &[u32]) -> Result<u32> {
        self.model.kv_rebase(slot)?;
        self.refresh_tab();
        let r = self.prefill(prompt);
        self.model.kv_rebase(0)?;
        self.refresh_tab();
        r
    }

    pub fn prefill_slot_chunk(&mut self, slot: usize, prompt: &[u32], ch: Chunk) -> Result<()> {
        self.model.kv_rebase(slot)?;
        self.refresh_tab();
        let r = self.prefill_chunk(prompt, ch);
        self.model.kv_rebase(0)?;
        self.refresh_tab();
        r
    }

    pub fn last_token(&self) -> Result<u32> {
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        Ok(self.read_u32(t_ids, 0))
    }

    /// One decode step for every slot on decode program `dp` (the CPU engine's contract:
    /// per-slot `pos`/`kvlen`/`ids`, idle slots `(0, 1, any)`).
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
        if self.model.kv_slot() != 0 {
            return Err(RuntimeError::Device(
                "decode with the KV table rebased".into(),
            ));
        }
        if dp < self.model.dec_ix || dp >= self.model.blob.progs.len() {
            return Err(RuntimeError::Device(format!(
                "program {dp} is not a decode rung"
            )));
        }
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        let t_pos = self.need(self.model.wk.pos, "in.pos")?;
        let t_kvlen = self.need(self.model.wk.kvlen, "in.kvlen")?;
        self.write_u32s(t_ids, 0, ids);
        self.write_u32s(t_pos, 0, pos);
        self.write_u32s(t_kvlen, 0, kvlen);
        if b == 1 {
            let kvrow = self.model.kvrow.clone();
            let pg = &mut self.progs[dp];
            for &i in &kvrow {
                pg.insts_host[i as usize].i[3] = pos[0];
            }
        }
        self.run_prog(dp)?;
        let rows = (self.model.blob.progs[dp].t as usize).min(b);
        let mut out: Vec<u32> = (0..rows).map(|i| self.read_u32(t_ids, i)).collect();
        out.resize(b, 0);
        Ok(out)
    }

    pub fn set_token(&self, id: u32) -> Result<()> {
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        self.write_u32s(t_ids, 0, &[id]);
        Ok(())
    }
}

fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data tables (repr(C) integers/structs), viewed as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn shared_from(device: &ProtocolObject<dyn MTLDevice>, bytes: &[u8]) -> Result<Buf> {
    let len = bytes.len().max(16);
    let b = device
        .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| RuntimeError::Oom(format!("metal buffer {len} B")))?;
    // SAFETY: fresh buffer of `len >= bytes.len()` bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            b.contents().as_ptr() as *mut u8,
            bytes.len(),
        )
    };
    Ok(b)
}
