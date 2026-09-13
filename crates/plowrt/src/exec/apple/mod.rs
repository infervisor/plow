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

pub mod asr;
pub mod asr_conformer;
pub mod asr_subsampling;
#[cfg(feature = "ane")]
pub mod channel;
mod conformer_packet;
pub mod hetero;
pub mod q8;
mod rnnt_packet;

const MSL: &str = include_str!("../../../../../runtime/apple/interp.metal");
const THREADS: usize = 1024;
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
    four_row_mx4: bool,
}

/// A decode/prefill instruction whose column range is split between the GPU (the walk, on
/// columns `[0, n_gpu)`) and the CPU worker threads (NEON kernels on `[n_gpu, N)`), joined at a
/// segment boundary — rung 3's runtime half. `PLOW_CPU_SHARE=<pct>[:<n ops>]`.
struct CpuSlot {
    prog: usize,
    inst: usize,
    n_gpu: u32,
    threads: usize,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct ExecutionProfile {
    pub command_buffers: usize,
    pub gpu_device_ms: f64,
    pub gpu_wait_ms: f64,
}

pub struct MetalEngine {
    pub model: CpuModel,
    _device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pso: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_four_rows: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pso_single: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_mx4: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pso_mx4_prefill: Option<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    conformer: Option<conformer_packet::ConformerPipelines>,
    rnnt: Option<rnnt_packet::RnntPipelines>,
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
    embedding_rows: usize,
    spin_max: u32,
    pub last_run_us: f64,
    pub gpu_name: String,
    /// ANE-delegated instructions, ascending by (prog, inst).
    #[cfg(feature = "ane")]
    ane: Vec<AneSlot>,
    /// `(ops run on the ANE, summed ANE ms)` for the last program run that used it.
    pub last_ane: Option<(usize, f64)>,
    cpu_slots: Vec<CpuSlot>,
    /// `(ops split with the CPU, summed CPU-side ms)` for the last program run that used it.
    pub last_cpu: Option<(usize, f64)>,
    /// Compiler-planned heterogeneous prefill (`hetero.json` beside the blob).
    pub hetero: Option<hetero::Hetero>,
    #[cfg(feature = "ane")]
    pub channel: Option<channel::Channel>,
    pub profile: Option<ExecutionProfile>,
}

// SAFETY: Metal and CoreML objects are thread-safe per Apple's documentation, and the serve
// layer serializes every call through one lock; the engine is moved to the engine thread once.
unsafe impl Send for MetalEngine {}

impl crate::serve::cpu_serve::SlotEngine for MetalEngine {
    fn prefill_buckets(&self) -> Vec<(usize, u32)> {
        MetalEngine::prefill_buckets(self)
    }
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

#[derive(Default)]
pub(crate) struct DecodeTuning {
    pub qkv_dot4: bool,
    pub single_head_work: bool,
}

impl MetalEngine {
    pub fn load(blob: &Path, checkpoint: &Path) -> Result<MetalEngine> {
        Self::load_with_decode_tuning(blob, checkpoint, DecodeTuning::default())
    }

    pub(crate) fn load_with_decode_tuning(
        blob: &Path,
        checkpoint: &Path,
        tuning: DecodeTuning,
    ) -> Result<MetalEngine> {
        // The loader resolves CPU kernels per program (an ABI check); the table must exist.
        ffi::init(Isa::Amx)?;
        let model = CpuModel::load(blob, checkpoint)?;
        Self::from_model(model, blob, tuning)
    }

    /// Load a self-contained, model-independent packet asset. Runtime dispatch depends only on
    /// its opcodes and tensor table; no tokenizer, model family or KV protocol is selected.
    pub fn load_packet(blob: &Path) -> Result<MetalEngine> {
        ffi::init(Isa::Amx)?;
        let model = CpuModel::load_embedded(blob)?;
        Self::from_model(model, blob, DecodeTuning::default())
    }

    fn from_model(model: CpuModel, blob: &Path, tuning: DecodeTuning) -> Result<MetalEngine> {
        let embedding_rows = model
            .blob
            .progs
            .iter()
            .flat_map(|p| &p.insts)
            .filter(|d| {
                matches!(
                    packet::dev::DevOp::from_u16(d.op),
                    Some(packet::dev::DevOp::Embed | packet::dev::DevOp::EmbedOverlayBf16)
                ) && d.i[1] != 0
            })
            .map(|d| model.tensor(d.t[1] as usize).bytes / 2 / d.i[1] as usize)
            .min()
            .unwrap_or(0);
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
        #[cfg(feature = "ane")]
        let channel_enabled = crate::config::RuntimeConfig::get().apple.ane_mlp;
        #[cfg(not(feature = "ane"))]
        if crate::config::RuntimeConfig::get().apple.ane_mlp {
            return Err(err("channel MLP", "build with --features ane"));
        }
        #[cfg(feature = "ane")]
        let channel_source = channel_enabled.then(|| format!("{MSL}\n{}", channel::MSL));
        #[cfg(feature = "ane")]
        let source = channel_source.as_deref().unwrap_or(MSL);
        #[cfg(not(feature = "ane"))]
        let source = MSL;
        let apple_config = &crate::config::RuntimeConfig::get().apple;
        let qkv_source = apple_config
            .qkv_dot4
            .unwrap_or(tuning.qkv_dot4)
            .then(|| format!("#define PLOW_QKV_DOT4 1\n{source}"));
        let source = qkv_source.as_deref().unwrap_or(source);
        let heads_source = apple_config
            .decode_heads
            .unwrap_or(tuning.single_head_work)
            .then(|| format!("#define PLOW_DECODE_HEADS 1\n{source}"));
        let source = heads_source.as_deref().unwrap_or(source);
        let glu_source = apple_config
            .glu_pair
            .then(|| format!("#define PLOW_GLU_PAIR 1\n{source}"));
        let source = glu_source.as_deref().unwrap_or(source);
        let four_rows = model.batch == 4
            && model
                .blob
                .progs
                .iter()
                .flat_map(|program| &program.insts)
                .any(|inst| matches!(inst.op, 91 | 92) && inst.i[0] == 4);
        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&opts))
            .map_err(|e| err("MSL compile", e))?;
        let has_conformer = model
            .blob
            .progs
            .iter()
            .any(|program| conformer_packet::supports(&program.insts));
        let conformer = if has_conformer {
            match conformer_packet::ConformerPipelines::load(&device, &opts) {
                Ok(pipelines) => Some(pipelines),
                Err(error) => {
                    tracing::warn!(%error, "metal: Conformer fast kernels unavailable; using packet interpreter");
                    None
                }
            }
        } else {
            None
        };
        let has_rnnt = model
            .blob
            .progs
            .iter()
            .any(|program| rnnt_packet::supports(&program.insts));
        let rnnt = if has_rnnt {
            match rnnt_packet::RnntPipelines::load(&device, &opts) {
                Ok(pipelines) => Some(pipelines),
                Err(error) => {
                    tracing::warn!(%error, "metal: RNNT fast kernels unavailable; using packet interpreter");
                    None
                }
            }
        } else {
            None
        };
        let func = lib
            .newFunctionWithName(&NSString::from_str("plow_interp"))
            .ok_or_else(|| RuntimeError::Device("metal: plow_interp missing".into()))?;
        let pso = device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| err("pipeline", e))?;
        let four_rows_lib = if four_rows {
            let source = format!("#define PLOW_MX4_FOUR_ROWS 1\n{source}");
            Some(
                device
                    .newLibraryWithSource_options_error(&NSString::from_str(&source), Some(&opts))
                    .map_err(|e| err("four-row MSL compile", e))?,
            )
        } else {
            None
        };
        let specialized_lib = four_rows_lib.as_ref().unwrap_or(&lib);
        let pso_four_rows = four_rows_lib
            .as_ref()
            .map(|lib| {
                let function = lib
                    .newFunctionWithName(&NSString::from_str("plow_interp"))
                    .ok_or_else(|| err("four-row pipeline", "plow_interp missing"))?;
                device
                    .newComputePipelineStateWithFunction_error(&function)
                    .map_err(|e| err("four-row pipeline", e))
            })
            .transpose()?;
        let func_single = specialized_lib
            .newFunctionWithName(&NSString::from_str("plow_single"))
            .ok_or_else(|| RuntimeError::Device("metal: plow_single missing".into()))?;
        let pso_single = device
            .newComputePipelineStateWithFunction_error(&func_single)
            .map_err(|e| err("pipeline single", e))?;
        for pipeline in [&pso, &pso_single].into_iter().chain(pso_four_rows.iter()) {
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < THREADS
                || pipeline.staticThreadgroupMemoryLength() > device.maxThreadgroupMemoryLength()
            {
                return Err(err(
                    "pipeline capability",
                    "requires 32-lane SIMD groups, 1024 threads and sufficient threadgroup memory",
                ));
            }
        }
        let has_mx4 = model
            .blob
            .progs
            .iter()
            .flat_map(|program| &program.insts)
            .any(|instruction| matches!(instruction.op, 91 | 92 | 93 | 96 | 97 | 98));
        let pso_mx4 = if apple_config.mx4_dedicated || has_mx4 {
            let function = specialized_lib
                .newFunctionWithName(&NSString::from_str("plow_mx4_dedicated"))
                .ok_or_else(|| err("MXFP4 dedicated", "missing kernel"))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| err("MXFP4 dedicated", e))?;
            if pipeline.threadExecutionWidth() != 32
                || pipeline.maxTotalThreadsPerThreadgroup() < 64
            {
                return Err(err(
                    "MXFP4 dedicated",
                    "requires 32-lane SIMD and 64 threads",
                ));
            }
            Some(pipeline)
        } else {
            None
        };
        let pso_mx4_prefill = pso_mx4.as_ref().and_then(|_| {
            let function =
                specialized_lib.newFunctionWithName(&NSString::from_str("plow_mx4_prefill"))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .ok()?;
            (pipeline.threadExecutionWidth() == 32
                && pipeline.maxTotalThreadsPerThreadgroup() >= THREADS
                && pipeline.staticThreadgroupMemoryLength() <= device.maxThreadgroupMemoryLength())
            .then_some(pipeline)
        });
        let gpu_name = device.name().to_string();
        let matching_geometry = hwspec::registry::lookup(&gpu_name)
            .is_some_and(|spec| spec.sm_count == model.blob.n_cu);
        let serial = crate::config::RuntimeConfig::get().apple.serial || !matching_geometry;

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
                four_row_mx4: p
                    .insts
                    .iter()
                    .any(|inst| matches!(inst.op, 91 | 92) && inst.i[0] == 4),
            });
        }

        // Tensor buffers are reached indirectly through the GPU address table. Buffers bound
        // directly on each encoder are tracked by Metal and do not belong in the residency set.
        let resset = {
            let desc = MTLResidencySetDescriptor::init(MTLResidencySetDescriptor::alloc());
            match device.newResidencySetWithDescriptor_error(&desc) {
                Ok(set) => {
                    for b in &bufs {
                        set.addAllocation(ProtocolObject::<dyn MTLAllocation>::from_ref(&**b));
                    }
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
        if !matching_geometry {
            tracing::info!(gpu = %gpu_name, n_cu = model.blob.n_cu,
                "using instruction-ordered Metal dispatch for unmatched tuning geometry");
        }
        tracing::info!(
            gpu = %gpu_name,
            tensors = n,
            wrapped,
            copied = n - wrapped,
            programs = progs.len(),
            n_cu = model.blob.n_cu,
            max_ctx,
            embedding_rows,
            setup_ms = format!("{:.0}", t0.elapsed().as_secs_f64() * 1e3).as_str(),
            "metal engine ready"
        );
        #[cfg(feature = "ane")]
        let ane = Self::ane_slots(&model)?;
        let cpu_slots = Self::cpu_slots(&model)?;
        let hetero = hetero::Hetero::load(&model, blob)?;
        if hetero.is_some() && serial {
            return Err(RuntimeError::Device(
                "metal: a heterogeneous blob needs the persistent walk (unset PLOW_METAL_SERIAL)"
                    .into(),
            ));
        }
        #[cfg(feature = "ane")]
        if channel_enabled
            && (serial || !ane.is_empty() || !cpu_slots.is_empty() || hetero.is_some())
        {
            return Err(err(
                "channel MLP",
                "cannot combine serial, row, per-op ANE or CPU offload",
            ));
        }
        let mut engine = MetalEngine {
            #[cfg(feature = "ane")]
            ane,
            last_ane: None,
            cpu_slots,
            last_cpu: None,
            hetero,
            #[cfg(feature = "ane")]
            channel: None,
            model,
            _device: device,
            queue,
            pso,
            pso_four_rows,
            pso_single,
            pso_mx4,
            pso_mx4_prefill,
            conformer,
            rnnt,
            serial,
            resset,
            bufs,
            copied,
            tab,
            fault,
            progs,
            max_ctx,
            embedding_rows,
            spin_max: crate::config::RuntimeConfig::get()
                .apple
                .spin_max
                .unwrap_or(1 << 22),
            last_run_us: 0.0,
            profile: None,
            gpu_name,
        };
        #[cfg(feature = "ane")]
        if channel_enabled {
            match objc2::rc::autoreleasepool(|_| channel::Channel::load(&engine, &lib, blob)) {
                Ok(channel) => engine.channel = Some(channel),
                Err(e) => {
                    tracing::warn!(error = %e, "channel MLP load rejected; retaining unsplit GPU")
                }
            }
        }
        Ok(engine)
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn packet_tensor(&self, name: &str) -> Option<usize> {
        self.model
            .names
            .iter()
            .position(|candidate| candidate == name)
    }

    pub(crate) fn embedding_table_handle(&self) -> Result<usize> {
        self.model
            .blob
            .progs
            .iter()
            .flat_map(|program| &program.insts)
            .find(|instruction| {
                matches!(
                    packet::dev::DevOp::from_u16(instruction.op),
                    Some(packet::dev::DevOp::Embed | packet::dev::DevOp::EmbedOverlayBf16)
                )
            })
            .map(|instruction| instruction.t[1] as usize)
            .ok_or_else(|| RuntimeError::Rejected("packet has no token embedding operation".into()))
    }

    pub fn write_packet_f32(&self, tensor: usize, values: &[f32]) -> Result<()> {
        let declaration = self.model.names.get(tensor).ok_or_else(|| {
            RuntimeError::Device(format!("packet tensor handle {tensor} is missing"))
        })?;
        let bytes = values
            .len()
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::Device("packet input size overflows".into()))?;
        if self.model.tensor(tensor).bytes != bytes {
            return Err(RuntimeError::Device(format!(
                "packet tensor {} has {} bytes, input has {bytes}",
                declaration,
                self.model.tensor(tensor).bytes
            )));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                self.host_ptr(tensor),
                bytes,
            )
        };
        Ok(())
    }

    pub fn read_packet_f32(&self, tensor: usize) -> Result<Vec<f32>> {
        self.model.names.get(tensor).ok_or_else(|| {
            RuntimeError::Device(format!("packet tensor handle {tensor} is missing"))
        })?;
        let len = self.model.tensor(tensor).bytes / 4;
        Ok(
            unsafe {
                std::slice::from_raw_parts(self.host_ptr(tensor).cast::<f32>(), len).to_vec()
            },
        )
    }

    pub fn run_packet(&mut self, program: usize) -> Result<()> {
        if program >= self.progs.len() {
            return Err(RuntimeError::Device(format!(
                "packet program {program} is missing"
            )));
        }
        self.run_prog(program)
    }

    /// Select compiler-ordered dispatch for packet pipelines that require progress without
    /// cross-threadgroup spinning. The packet contract chooses this; model identity is irrelevant.
    pub fn set_ordered_dispatch(&mut self, enabled: bool) -> Result<()> {
        if enabled && self.hetero.is_some() {
            return Err(RuntimeError::Rejected(
                "ordered packet dispatch cannot use a heterogeneous lane plan".into(),
            ));
        }
        self.serial = crate::config::RuntimeConfig::get().apple.serial
            || enabled
            || !hwspec::registry::lookup(&self.gpu_name)
                .is_some_and(|spec| spec.sm_count == self.model.blob.n_cu);
        Ok(())
    }

    pub fn run_packet_sequence(&mut self, programs: &[usize]) -> Result<()> {
        for &program in programs {
            if program >= self.progs.len() {
                return Err(RuntimeError::Device(format!(
                    "packet program {program} is missing"
                )));
            }
        }
        if programs.is_empty() {
            return Ok(());
        }
        if self.serial || programs.iter().any(|&program| !self.gpu_only_program(program)) {
            for &program in programs {
                self.run_prog(program)?;
            }
            return Ok(());
        }

        let started = Instant::now();
        for programs in programs.chunks(32) {
            for &program in programs {
                self.reset_program(program);
            }
            let cb = self
                .queue
                .commandBuffer()
                .ok_or_else(|| RuntimeError::Device("metal: no command buffer".into()))?;
            for &program in programs {
                self.encode_into(
                    program,
                    0,
                    u32::MAX,
                    0,
                    self.progs[program].n_seg,
                    &cb,
                    None,
                )?;
            }
            cb.commit();
            let wait = Instant::now();
            cb.waitUntilCompleted();
            if let Some(profile) = &mut self.profile {
                profile.command_buffers += 1;
                profile.gpu_wait_ms += wait.elapsed().as_secs_f64() * 1e3;
                profile.gpu_device_ms += (cb.GPUEndTime() - cb.GPUStartTime()).max(0.0) * 1e3;
            }
            if cb.status() != MTLCommandBufferStatus::Completed {
                let error = cb.error().map(|error| error.to_string()).unwrap_or_default();
                return Err(RuntimeError::Device(format!(
                    "metal: packet sequence command buffer status {:?}: {error}",
                    cb.status()
                )));
            }
            let fault = unsafe {
                std::ptr::read_volatile(self.fault.contents().as_ptr() as *const u32)
            };
            if fault != 0 {
                return Err(RuntimeError::Device(format!(
                    "metal: packet sequence fault {fault:#010x}"
                )));
            }
        }
        self.last_run_us = started.elapsed().as_secs_f64() * 1e6;
        Ok(())
    }

    /// `PLOW_CPU_SHARE=<pct>[:<n ops>]`: give the CPU `pct`% of the columns of the first n (default
    /// all) GEMV-family instructions of the batch-1 decode program. The GPU walk stops after
    /// each such instruction; the two halves run concurrently and join at the boundary.
    fn cpu_slots(model: &CpuModel) -> Result<Vec<CpuSlot>> {
        let Some(spec) = &crate::config::RuntimeConfig::get().apple.cpu_share else {
            return Ok(Vec::new());
        };
        if spec.is_empty() || spec == "0" {
            return Ok(Vec::new());
        }
        let (pct, limit) = match spec.split_once(':') {
            Some((a, b)) => (
                a.parse::<u32>()
                    .map_err(|e| RuntimeError::Device(format!("PLOW_CPU_SHARE: {e}")))?,
                b.parse::<usize>()
                    .map_err(|e| RuntimeError::Device(format!("PLOW_CPU_SHARE: {e}")))?,
            ),
            None => (
                spec.parse::<u32>()
                    .map_err(|e| RuntimeError::Device(format!("PLOW_CPU_SHARE: {e}")))?,
                usize::MAX,
            ),
        };
        let threads = match crate::config::RuntimeConfig::get().cpu.threads {
            0 => 8,
            n => n as usize,
        };
        let dp = model.decode_prog_for(1);
        let mut out = Vec::new();
        for (i, d) in model.blob.progs[dp].insts.iter().enumerate() {
            if !matches!(d.op, 10 | 19 | 30 | 31) || out.len() >= limit {
                continue;
            }
            // bf16 GEMV norm folds (rms/gamma) are per row, fine; a bias would need a column offset.
            if d.op == 10 && d.t[7] != packet::dev::TENSOR_NONE16 {
                continue;
            }
            let n = d.i[1];
            let n_cpu = ((n * pct / 100) / 4) * 4;
            if n_cpu == 0 || n_cpu >= n {
                continue;
            }
            out.push(CpuSlot {
                prog: dp,
                inst: i,
                n_gpu: n - n_cpu,
                threads,
            });
        }
        tracing::info!(ops = out.len(), pct, threads, "CPU share slots ready");
        Ok(out)
    }

    /// Run the CPU half of slot `si` on `threads` worker threads: the same NEON kernel over the
    /// instruction's tail columns, with the weight / scale / output tables offset by `n_gpu`.
    fn run_cpu_slot(&self, si: usize) -> Result<()> {
        let slot = &self.cpu_slots[si];
        let mut d = self.progs[slot.prog].insts_host[slot.inst];
        let n_gpu = slot.n_gpu as usize;
        let k = d.i[2] as usize;
        let n_cpu = d.i[1] as usize - n_gpu;
        let n_tensors = self.model.names.len();
        let mut table: Vec<*mut c_void> = (0..n_tensors)
            .map(|h| self.host_ptr(h) as *mut c_void)
            .collect();
        let fp8 = d.op == 30 || d.op == 31;
        let w_bytes = if fp8 { k } else { k * 2 };
        let (w_slots, s_slots): (&[usize], &[usize]) = match d.op {
            30 => (&[2], &[5]),
            31 => (&[2, 5], &[3, 4]),
            19 => (&[2, 5], &[6, 7]), // bf16 GLU biases are per column too
            _ => (&[2], &[]),
        };
        let off = |table: &mut Vec<*mut c_void>, slot: usize, bytes: usize| {
            let h = d.t[slot];
            if h != packet::dev::TENSOR_NONE16 {
                // SAFETY: an in-bounds column offset of the tensor.
                table[h as usize] =
                    unsafe { (table[h as usize] as *mut u8).add(bytes) } as *mut c_void;
            }
        };
        for &w in w_slots {
            off(&mut table, w, n_gpu * w_bytes);
        }
        for &sc in s_slots {
            off(&mut table, sc, n_gpu * if d.op == 19 { 2 } else { 4 });
        }
        off(&mut table, 0, n_gpu * 2);
        d.i[1] = n_cpu as u32;
        let f = ffi::kernel(d.op)
            .ok_or_else(|| RuntimeError::Device(format!("no CPU kernel for op {}", d.op)))?;
        let scratch_bytes = ffi::scratch_bytes().max(64) as usize;
        let table_ptr = table.as_ptr() as usize;
        let threads = slot.threads;
        std::thread::scope(|sc| {
            for w in 0..threads {
                sc.spawn(move || {
                    let mut ctx = ffi::PlowCpuCtx::new(w as u32, 0);
                    let mut scratch = vec![0u64; scratch_bytes / 8 + 8];
                    ctx.scratch = scratch.as_mut_ptr() as *mut c_void;
                    ctx.scratch_bytes = scratch_bytes as u32;
                    let _ = ffi::thread_init(&mut ctx);
                    // SAFETY: disjoint column slices of one op on tensors the GPU half does not write.
                    unsafe {
                        f(
                            &d,
                            w as u32,
                            threads as u32,
                            table_ptr as *const *mut c_void,
                            &mut ctx,
                        )
                    };
                });
            }
        });
        Ok(())
    }

    /// `PLOW_ANE=1|<n>|all`: delegate the first n (or every) eligible GEMM of the 128-row
    /// prefill bucket to the Neural Engine — fp8 (op 33, no activation scale) or bf16 (op 8, no
    /// bias). `PLOW_ANE=<prog>:<inst>` names one instruction. Weights are dequantized to fp16
    /// once at load (option B of §2.3); each op is its own CoreML program, cached on disk.
    #[cfg(feature = "ane")]
    fn ane_slots(model: &CpuModel) -> Result<Vec<AneSlot>> {
        use crate::exec::ane::{f32_to_f16, AneGemm};
        use packet::dev::TENSOR_NONE16;
        let Some(spec) = &crate::config::RuntimeConfig::get().apple.ane else {
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
        let a = self.host_ptr(ha);
        let c = self.host_ptr(hc);
        let slot = &mut self.ane[si];
        let t = slot.gemm.t;
        // SAFETY: quiescent between command buffers; the tensors hold (a_row0 + t) x K and (c_row0 + t) x N.
        unsafe {
            let a = a.add(a_row0 * k * 2) as *const u16;
            for i in 0..t * k {
                slot.x[i] = f32::from_bits((std::ptr::read_unaligned(a.add(i)) as u32) << 16);
            }
        }
        slot.gemm.run(&slot.x, &mut slot.y)?;
        unsafe {
            let c = c.add(c_row0 * n * 2) as *mut u16;
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

    fn reset_program(&mut self, p: usize) {
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
    }

    fn gpu_only_program(&self, p: usize) -> bool {
        if self.hetero.as_ref().is_some_and(|h| h.has_active_lanes(p))
            || self.cpu_slots.iter().any(|s| s.prog == p)
        {
            return false;
        }
        #[cfg(feature = "ane")]
        if self.ane.iter().any(|s| s.prog == p)
            || self.channel.as_ref().is_some_and(|c| c.eligible(p))
        {
            return false;
        }
        true
    }

    /// Copy the patched instructions in, zero the counters, dispatch every segment, wait.
    fn run_prog(&mut self, p: usize) -> Result<()> {
        let t0 = Instant::now();
        self.reset_program(p);
        #[cfg(feature = "ane")]
        if self.channel.as_ref().is_some_and(|c| c.eligible(p)) {
            let mut channel = self.channel.take().unwrap();
            let result = channel.run(self);
            if result.is_err() {
                channel.stats.disabled = true;
            }
            self.channel = Some(channel);
            self.last_run_us = t0.elapsed().as_secs_f64() * 1e6;
            return result;
        }
        if self.hetero.as_ref().is_some_and(|h| h.has_active_lanes(p)) {
            self.run_prog_hetero(p)?;
            self.last_run_us = t0.elapsed().as_secs_f64() * 1e6;
            return self.check_fault(p);
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
        let mine: Vec<usize> = (0..self.cpu_slots.len())
            .filter(|&i| self.cpu_slots[i].prog == p)
            .collect();
        if !mine.is_empty() && !self.serial {
            self.last_cpu = None;
            let n = self.progs[p].insts_host.len() as u32;
            let mut lo = 0u32;
            let mut cpu_ms = 0.0;
            for si in mine {
                let (i, n_gpu) = (self.cpu_slots[si].inst as u32, self.cpu_slots[si].n_gpu);
                // Producers of instruction i must be complete before the CPU half reads them:
                // sync the walk up to i, then run the GPU's column share of i and the CPU's tail
                // concurrently. Two boundaries per split op — the cost §2.2 pairs joins to avoid.
                self.dispatch_range(p, lo, i)?;
                let full_n = self.progs[p].insts_host[i as usize].i[1];
                self.progs[p].insts_host[i as usize].i[1] = n_gpu;
                let cb = self.commit_range(p, i, i + 1)?;
                self.progs[p].insts_host[i as usize].i[1] = full_n;
                let t = Instant::now();
                self.run_cpu_slot(si)?;
                cpu_ms += t.elapsed().as_secs_f64() * 1e3;
                cb.waitUntilCompleted();
                if cb.status() != MTLCommandBufferStatus::Completed {
                    return Err(RuntimeError::Device(format!(
                        "metal: program {p} command buffer status {:?}",
                        cb.status()
                    )));
                }
                lo = i + 1;
                self.last_cpu = Some(match self.last_cpu {
                    Some((c, _)) => (c + 1, cpu_ms),
                    None => (1, cpu_ms),
                });
            }
            self.dispatch_range(p, lo, n)?;
            self.last_run_us = t0.elapsed().as_secs_f64() * 1e6;
            return self.check_fault(p);
        }
        self.dispatch_range(p, 0, u32::MAX)?;
        self.last_run_us = t0.elapsed().as_secs_f64() * 1e6;
        self.check_fault(p)
    }

    /// The three-lane segment loop (`hetero.rs`): per host segment, the GPU's command buffer
    /// runs asynchronously while the CPU pool and the ANE take their row blocks; the join is
    /// waiting for all three.
    fn run_prog_hetero(&mut self, p: usize) -> Result<()> {
        let n_seg = self.progs[p].n_seg;
        let plan = self.hetero.as_ref().unwrap().prog_plan(p).unwrap().clone();
        if plan.segments.iter().any(|s| s.seg >= n_seg) {
            return Err(RuntimeError::Device(format!(
                "hetero: program {p} plan names segment past the blob's {n_seg}"
            )));
        }
        let n = self.model.names.len();
        let table: Vec<*mut u8> = (0..n).map(|h| self.host_ptr(h)).collect();
        let mut si = 0usize;
        for seg in 0..n_seg {
            let cb = self.commit(p, 0, u32::MAX, seg, seg + 1)?;
            let sp = plan.segments.get(si).filter(|s| s.seg == seg);
            let mut started = false;
            if let Some(sp) = sp {
                si += 1;
                let insts = std::mem::take(&mut self.progs[p].insts_host);
                let het = self.hetero.as_mut().unwrap();
                let r = het.run_lanes(p, sp, &insts, &table);
                self.progs[p].insts_host = insts;
                started = r?;
                if started {
                    self.hetero.as_mut().unwrap().wait_cpu();
                }
            }
            let t = Instant::now();
            cb.waitUntilCompleted();
            self.record_gpu_profile(&cb, t);
            if let Some(h) = self.hetero.as_mut() {
                h.stats.segs += 1;
                if sp.is_some() {
                    h.stats.gpu_wait_ms += t.elapsed().as_secs_f64() * 1e3;
                }
            }
            let _ = started;
            if cb.status() != MTLCommandBufferStatus::Completed {
                let e = cb.error().map(|e| e.to_string()).unwrap_or_default();
                return Err(RuntimeError::Device(format!(
                    "metal: program {p} segment {seg} command buffer status {:?}: {e}",
                    cb.status()
                )));
            }
        }
        Ok(())
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
        let cb = self.commit_range(p, inst_lo, inst_hi)?;
        let start = self.profile.map(|_| Instant::now());
        cb.waitUntilCompleted();
        if let Some(start) = start {
            self.record_gpu_profile(&cb, start);
        }
        if cb.status() != MTLCommandBufferStatus::Completed {
            let e = cb.error().map(|e| e.to_string()).unwrap_or_default();
            return Err(RuntimeError::Device(format!(
                "metal: program {p} command buffer status {:?}: {e}",
                cb.status()
            )));
        }
        Ok(())
    }

    fn record_gpu_profile(&mut self, cb: &ProtocolObject<dyn MTLCommandBuffer>, start: Instant) {
        if let Some(profile) = &mut self.profile {
            profile.command_buffers += 1;
            profile.gpu_wait_ms += start.elapsed().as_secs_f64() * 1e3;
            profile.gpu_device_ms += (cb.GPUEndTime() - cb.GPUStartTime()).max(0.0) * 1e3;
        }
    }

    pub fn set_profiling(&mut self, enabled: bool) {
        self.profile = enabled.then_some(ExecutionProfile::default());
        if let Some(h) = &mut self.hetero {
            h.profile_enabled = enabled;
            h.reset_stats();
        }
    }

    /// As [`Self::dispatch_range`] but returns the committed command buffer without waiting, so
    /// host-side work (a CPU column share) can overlap it.
    fn commit_range(
        &mut self,
        p: usize,
        inst_lo: u32,
        inst_hi: u32,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let n_seg = self.progs[p].n_seg;
        self.commit(p, inst_lo, inst_hi, 0, n_seg)
    }

    /// One command buffer over segments `[seg_lo, seg_hi)` of program `p` restricted to
    /// instructions `[inst_lo, inst_hi)`; committed, not waited.
    fn commit(
        &mut self,
        p: usize,
        inst_lo: u32,
        inst_hi: u32,
        seg_lo: u32,
        seg_hi: u32,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| RuntimeError::Device("metal: no command buffer".into()))?;
        self.commit_into(p, inst_lo, inst_hi, seg_lo, seg_hi, cb, None)
    }

    fn commit_into(
        &mut self,
        p: usize,
        inst_lo: u32,
        inst_hi: u32,
        seg_lo: u32,
        seg_hi: u32,
        cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        staged_input: Option<&Buf>,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
        self.encode_into(p, inst_lo, inst_hi, seg_lo, seg_hi, &cb, staged_input)?;
        cb.commit();
        Ok(cb)
    }

    fn encode_into(
        &mut self,
        p: usize,
        inst_lo: u32,
        inst_hi: u32,
        seg_lo: u32,
        seg_hi: u32,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        staged_input: Option<&Buf>,
    ) -> Result<()> {
        // The instruction table may have been patched since the last copy.
        {
            let pg = &self.progs[p];
            // SAFETY: shared buffer sized at load for this table; no run in flight on it.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pg.insts_host.as_ptr() as *const u8,
                    pg.insts.contents().as_ptr() as *mut u8,
                    std::mem::size_of_val(pg.insts_host.as_slice()),
                );
            }
        }
        let pg = &self.progs[p];
        if !self.serial {
            if let Some(conformer) = self
                .conformer
                .as_ref()
                .filter(|_| conformer_packet::supports(&pg.insts_host))
            {
                conformer.encode(&pg.insts_host, &self.bufs, cb)?;
                return Ok(());
            }
            if let Some(rnnt) = self
                .rnnt
                .as_ref()
                .filter(|_| rnnt_packet::supports(&pg.insts_host))
            {
                rnnt.encode(&pg.insts_host, &self.bufs, cb)?;
                return Ok(());
            }
        }
        let n_cu = self.model.blob.n_cu as usize;
        if self.serial {
            // Topological instruction order (the builder appends ops in dependency order).
            for (ii, d) in pg.insts_host.iter().enumerate() {
                let enc = cb
                    .computeCommandEncoder()
                    .ok_or_else(|| RuntimeError::Device("metal: no encoder".into()))?;
                let dedicated = self.pso_mx4.as_ref().filter(|_| {
                    matches!(d.op, 91 | 92)
                        && d.i[0] > 0
                        && d.i[1] > 0
                        && d.i[1] % 8 == 0
                        && d.i[2] > 0
                        && d.i[2] % 32 == 0
                });
                let prefill = self
                    .pso_mx4_prefill
                    .as_ref()
                    .filter(|_| matches!(d.op, 93 | 96 | 97 | 98));
                enc.setComputePipelineState(dedicated.or(prefill).unwrap_or(&self.pso_single));
                if let Some(input) = staged_input {
                    enc.useResource_usage(
                        ProtocolObject::from_ref(&**input),
                        MTLResourceUsage::Read | MTLResourceUsage::Write,
                    );
                }
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
                        width: if dedicated.is_some() {
                            d.i[1] as usize / if d.op == 92 { 2 } else { 8 }
                        } else {
                            d.blocks as usize
                        },
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: if dedicated.is_some() { 64 } else { THREADS },
                        height: 1,
                        depth: 1,
                    },
                );
                enc.endEncoding();
            }
        }
        for seg in (if self.serial { 0 } else { seg_lo })..(if self.serial {
            0
        } else {
            seg_hi.min(pg.n_seg)
        }) {
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| RuntimeError::Device("metal: no encoder".into()))?;
            let pipeline = self
                .pso_four_rows
                .as_ref()
                .filter(|_| pg.four_row_mx4)
                .unwrap_or(&self.pso);
            enc.setComputePipelineState(pipeline);
            if let Some(input) = staged_input {
                enc.useResource_usage(
                    ProtocolObject::from_ref(&**input),
                    MTLResourceUsage::Read | MTLResourceUsage::Write,
                );
            }
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

    pub fn prefill_embeddings(&mut self, prompt: &[u32], embeddings: &[u16]) -> Result<u32> {
        self.prefill_embeddings_staged(
            prompt,
            embeddings.len(),
            false,
            |_, output, _, ch, hidden, _| {
                let source =
                    &embeddings[ch.c0 as usize * hidden..(ch.c0 + ch.clen) as usize * hidden];
                // The previous command buffer completed; the destination is idle.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        source.as_ptr().cast::<u8>(),
                        output.contents().as_ptr().cast::<u8>(),
                        source.len() * 2,
                    );
                }
                Ok(())
            },
        )
    }

    // With a command buffer, stage only encodes. Otherwise staging must complete before returning.
    pub(crate) fn prefill_embeddings_staged(
        &mut self,
        prompt: &[u32],
        elements: usize,
        device_stage: bool,
        mut stage: impl FnMut(
            Option<&ProtocolObject<dyn MTLCommandBuffer>>,
            &Buf,
            &Buf,
            Chunk,
            usize,
            usize,
        ) -> Result<()>,
    ) -> Result<u32> {
        let x = self
            .model
            .names
            .iter()
            .position(|n| n == "act.x")
            .ok_or_else(|| err("ASR", "missing act.x"))?;
        let buckets = self.prefill_buckets();
        let first = *buckets
            .first()
            .ok_or_else(|| err("ASR", "missing prefill program"))?;
        let embed = self.model.blob.progs[first.0]
            .insts
            .iter()
            .find(|d| {
                matches!(
                    packet::dev::DevOp::from_u16(d.op),
                    Some(packet::dev::DevOp::Embed | packet::dev::DevOp::EmbedOverlayBf16)
                ) && d.t[0] as usize == x
            })
            .ok_or_else(|| err("ASR", "missing embedding instruction"))?;
        let hidden = embed.i[1] as usize;
        if hidden == 0
            || prompt.is_empty()
            || prompt.len() >= self.max_ctx
            || elements != prompt.len() * hidden
        {
            return Err(err("ASR", "invalid prompt embedding dimensions"));
        }
        for ch in plan_chunks(&buckets, prompt.len() as u32) {
            self.prepare_prefill_chunk(prompt, ch)?;
            let inst = self.progs[ch.prog]
                .insts_host
                .iter_mut()
                .find(|d| {
                    matches!(
                        packet::dev::DevOp::from_u16(d.op),
                        Some(packet::dev::DevOp::Embed | packet::dev::DevOp::EmbedOverlayBf16)
                    ) && d.t[0] as usize == x
                })
                .ok_or_else(|| err("ASR", "missing chunk embedding instruction"))?;
            let table = inst.t[1] as usize;
            inst.op = packet::dev::DevOp::Nop as u16;
            if ch.clen as usize * hidden * 2 > self.model.tensor(x).bytes {
                return Err(err("ASR", "embedding buffer too small"));
            }
            let cb = if device_stage && self.gpu_only_program(ch.prog) {
                Some(
                    self.queue
                        .commandBuffer()
                        .ok_or_else(|| err("prefill", "command buffer"))?,
                )
            } else {
                None
            };
            stage(
                cb.as_deref(),
                &self.bufs[x],
                &self.bufs[table],
                ch,
                hidden,
                self.model.tensor(table).bytes / (hidden * 2),
            )?;
            if let Some(cb) = cb {
                let start = Instant::now();
                self.reset_program(ch.prog);
                let input = self.bufs[x].clone();
                let cb = self.commit_into(
                    ch.prog,
                    0,
                    u32::MAX,
                    0,
                    self.progs[ch.prog].n_seg,
                    cb,
                    Some(&input),
                )?;
                let wait = Instant::now();
                cb.waitUntilCompleted();
                self.record_gpu_profile(&cb, wait);
                self.last_run_us = start.elapsed().as_secs_f64() * 1e6;
                if cb.status() != MTLCommandBufferStatus::Completed {
                    return Err(err("prefill staging", format!("{:?}", cb.error())));
                }
                self.check_fault(ch.prog)?;
            } else {
                self.run_prog(ch.prog)?;
            }
        }
        self.last_token()
    }

    /// Stage a prefill chunk (inputs + instruction rebases) without running it.
    pub fn prepare_prefill_chunk(&mut self, prompt: &[u32], ch: Chunk) -> Result<()> {
        let end = ch.c0.checked_add(ch.clen);
        if ch.prog >= self.model.dec_ix
            || end.is_none_or(|e| e as usize > prompt.len())
            || ch.clen == 0
            || end.is_some_and(|e| e as usize > self.max_ctx)
            || self
                .model
                .blob
                .progs
                .get(ch.prog)
                .is_some_and(|p| ch.clen > p.t)
        {
            return Err(RuntimeError::Device(format!("bad prefill chunk {ch:?}")));
        }
        self.validate_ids(&prompt[ch.c0 as usize..end.unwrap() as usize])?;
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
        #[cfg(feature = "ane")]
        if let Some(channel) = &mut self.channel {
            channel.clen = ch.clen;
        }
        if let Some(h) = self.hetero.as_mut() {
            h.prepare_chunk(ch.prog, &mut pg.insts_host, &pristine, ch.clen);
        }
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
        self.validate_ids(&[id])?;
        if pos as usize >= self.max_ctx || kvlen == 0 || kvlen as usize > self.max_ctx {
            return Err(err(
                "decode",
                "position or KV length outside context allocation",
            ));
        }
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

    /// Mutable view for calibration experiments (a patched instruction is copied to the device
    /// by the next `run_inst`/`run_prog`).
    pub fn insts_host_mut(&mut self, p: usize) -> &mut [DevInst64] {
        &mut self.progs[p].insts_host
    }

    /// Diagnostic: run ONE instruction of program `p` (all its slices) as its own dispatch.
    /// As [`Self::run_inst`] but committed without waiting: the caller overlaps host work with it
    /// and then waits on the returned command buffer (`check_fault` reads the fault word).
    pub fn run_inst_async(
        &mut self,
        p: usize,
        i: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>> {
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
        Ok(cb)
    }

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

    pub fn prefill_slot_embeddings(
        &mut self,
        slot: usize,
        prompt: &[u32],
        embeddings: &[u16],
    ) -> Result<u32> {
        self.model.kv_rebase(slot)?;
        self.refresh_tab();
        let result = self.prefill_embeddings(prompt, embeddings);
        self.model.kv_rebase(0)?;
        self.refresh_tab();
        result
    }

    pub fn last_token(&self) -> Result<u32> {
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        Ok(self.read_u32(t_ids, 0))
    }

    /// A supplied command buffer must only be encoded, not committed. Without one,
    /// staging must finish before returning. Operand buffers must remain alive until retirement.
    /// Errors restore the table base, but do not roll back KV writes from completed chunks.
    pub fn prefill_slot_embeddings_staged(
        &mut self,
        slot: usize,
        prompt: &[u32],
        elements: usize,
        device_stage: bool,
        stage: impl FnMut(
            Option<&ProtocolObject<dyn MTLCommandBuffer>>,
            &Buf,
            &Buf,
            Chunk,
            usize,
            usize,
        ) -> Result<()>,
    ) -> Result<u32> {
        self.model.kv_rebase(slot)?;
        self.refresh_tab();
        let result = self.prefill_embeddings_staged(prompt, elements, device_stage, stage);
        self.model.kv_rebase(0)?;
        self.refresh_tab();
        result
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
        self.validate_ids(ids)?;
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
        self.validate_ids(&[id])?;
        let t_ids = self.need(self.model.wk.ids, "in.ids")?;
        self.write_u32s(t_ids, 0, &[id]);
        Ok(())
    }

    fn validate_ids(&self, ids: &[u32]) -> Result<()> {
        if let Some(id) = ids.iter().find(|&&id| id as usize >= self.embedding_rows) {
            return Err(err(
                "token",
                format!("ID {id} outside embedding rows {}", self.embedding_rows),
            ));
        }
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

impl crate::exec::packet_runtime::PacketRuntime for MetalEngine {
    fn begin_execution(&mut self) -> Result<()> {
        if let Some(set) = &self.resset {
            set.requestResidency();
        }
        Ok(())
    }

    fn end_execution(&mut self) -> Result<()> {
        if let Some(set) = &self.resset {
            set.endResidency();
        }
        Ok(())
    }

    fn tensor(&self, name: &str) -> Option<crate::exec::packet_runtime::PacketTensor> {
        let handle = self.packet_tensor(name)?;
        Some(crate::exec::packet_runtime::PacketTensor {
            handle,
            bytes: self.model.tensor(handle).bytes,
        })
    }

    fn write_tensor(
        &mut self,
        tensor: crate::exec::packet_runtime::PacketTensor,
        bytes: &[u8],
    ) -> Result<()> {
        self.model.names.get(tensor.handle).ok_or_else(|| {
            RuntimeError::Device(format!("packet tensor handle {} is missing", tensor.handle))
        })?;
        crate::exec::packet_runtime::check_transfer(
            tensor,
            tensor.handle,
            self.model.tensor(tensor.handle).bytes,
            bytes.len(),
        )?;
        // SAFETY: the engine is exclusively borrowed, no command is in flight, and sizes match.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.host_ptr(tensor.handle), bytes.len())
        };
        Ok(())
    }

    fn read_tensor(
        &self,
        tensor: crate::exec::packet_runtime::PacketTensor,
        bytes: &mut [u8],
    ) -> Result<()> {
        self.model.names.get(tensor.handle).ok_or_else(|| {
            RuntimeError::Device(format!("packet tensor handle {} is missing", tensor.handle))
        })?;
        crate::exec::packet_runtime::check_transfer(
            tensor,
            tensor.handle,
            self.model.tensor(tensor.handle).bytes,
            bytes.len(),
        )?;
        // SAFETY: no command mutates the shared buffer while `self` is borrowed and sizes match.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.host_ptr(tensor.handle),
                bytes.as_mut_ptr(),
                bytes.len(),
            )
        };
        Ok(())
    }

    fn copy_tensor(
        &mut self,
        source: crate::exec::packet_runtime::PacketTensor,
        source_offset: usize,
        target: crate::exec::packet_runtime::PacketTensor,
        target_offset: usize,
        bytes: usize,
    ) -> Result<()> {
        self.model.names.get(source.handle).ok_or_else(|| {
            RuntimeError::Device(format!("packet tensor handle {} is missing", source.handle))
        })?;
        self.model.names.get(target.handle).ok_or_else(|| {
            RuntimeError::Device(format!("packet tensor handle {} is missing", target.handle))
        })?;
        crate::exec::packet_runtime::check_copy(
            source,
            self.model.tensor(source.handle).bytes,
            source_offset,
            target,
            self.model.tensor(target.handle).bytes,
            target_offset,
            bytes,
        )?;
        // SAFETY: both unified-memory ranges were checked and no command is in flight.
        unsafe {
            std::ptr::copy(
                self.host_ptr(source.handle).add(source_offset),
                self.host_ptr(target.handle).add(target_offset),
                bytes,
            )
        };
        Ok(())
    }

    fn run(&mut self, program: usize) -> Result<()> {
        self.run_packet(program)
    }

    fn run_sequence(&mut self, programs: &[usize]) -> Result<()> {
        self.run_packet_sequence(programs)
    }

    fn last_run_us(&self) -> f64 {
        self.last_run_us
    }
}

#[cfg(test)]
mod source_tests {
    use super::*;

    #[test]
    fn packet_interpreter_source_compiles() {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device");
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setLanguageVersion(MTLLanguageVersion::Version3_2);
        device
            .newLibraryWithSource_options_error(&NSString::from_str(MSL), Some(&options))
            .expect("compile packet interpreter");
    }
}
