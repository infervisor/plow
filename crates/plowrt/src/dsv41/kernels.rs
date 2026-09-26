//! Typed launches of the DeepSeek-V4.1 sm_90a cubin (`runtime/nvidia/dsv41/dsv41_all.cu`).
//!
//! One [`Kernels`] per device. Every launch goes on the stage's stream. Argument order and meaning
//! mirror the `extern "C"` kernel signatures exactly; [`ABI_VERSION`] must match the cubin's
//! `dsv41_abi_version`, bumped whenever a signature changes, so a stale cubin is refused at load
//! instead of reading its arguments wrong.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use crate::device::cuda::{CudaBackend, CudaEvent, CudaStream, KernelFn};
use crate::device::Backend;
use crate::error::{Result, RuntimeError};

/// Must equal `dsv41_abi_version` in `runtime/nvidia/dsv41/dsv41_common.cuh`.
pub const ABI_VERSION: u32 = 3;

/// One kernel argument, held by value until the launch copies it.
#[derive(Clone, Copy)]
pub enum A {
    /// Device pointer (0 = null).
    P(u64),
    I(i32),
    L(i64),
    F(f32),
}

pub const IX_SMEM: u32 = ((4 * 32 * (128 + 8) + 64 * (128 + 8)) * 2 + 2 * 4 * 64 * 4) as u32;
/// `G_SMEM(K, BM)` in dsv41_moe.cu: the grouped fp4 GEMM's stage ring plus both scale grids.
pub const fn moe_smem(k: usize, bm: usize) -> u32 {
    (3 * bm * 160 + 3 * 128 * 80 + (128 + bm) * (k / 32) + bm * 4 + 16) as u32
}
/// The grouped fp4 GEMM entry for a tile height.
pub const fn moe_gemm_name(bm: usize) -> &'static str {
    match bm {
        16 => "dsv_moe_gemm_fp4_m16",
        32 => "dsv_moe_gemm_fp4_m32",
        _ => "dsv_moe_gemm_fp4",
    }
}
/// The largest K the engine launches it with (the hidden size, 5120) sets the attribute.
pub const MOE_SMEM_MAX_K: usize = 8192;
/// `W8_SMEM` in dsv41_gemm.cu: the W8A8 GEMM's three 128-K stages of A (64 rows) and W (128 rows).
pub const W8_SMEM: u32 = 3 * (64 + 128) * 144;
pub const SA_SMEM: u32 = ((64 * 520 + 64 * 520 + 64 * 72) * 2 + 4 * 64 * 4 * 2 + 64 * 4) as u32;

const MAX_ARGS: usize = 24;

const NAMES: &[&str] = &[
    "dsv_rmsnorm",
    "dsv_act_quant_fp8",
    "dsv_fp4_fakequant",
    "dsv_gemm_w8a8",
    "dsv_gemm_bf16w",
    "dsv_splitk_reduce",
    "dsv_gemm_f32",
    "dsv_gemm_f32_dot",
    "dsv_gemm_f32_rows",
    "dsv_rope",
    "dsv_sparse_attn",
    "dsv_attn_index",
    "dsv_compress_pool_prefill",
    "dsv_compress_pool_decode",
    "dsv_scatter_rows",
    "dsv_copy_rows",
    "dsv_row_rsqrt",
    "dsv_hc_sinkhorn",
    "dsv_hc_mix_partial",
    "dsv_hc_mix_finish",
    "dsv_hc_pre",
    "dsv_hc_post",
    "dsv_engram_gate",
    "dsv_embed_hc",
    "dsv_argmax",
    "dsv_gumbel",
    "dsv_bf16_to_f32",
    "dsv_scale_bf16",
    "dsv_index_score",
    "dsv_topk_select",
    "dsv_cand_block_scores",
    "dsv_keep_from_idx",
    "dsv_moe_route",
    "dsv_moe_count",
    "dsv_moe_offsets",
    "dsv_moe_fill",
    "dsv_moe_gemm_fp4",
    "dsv_moe_gemm_fp4_m32",
    "dsv_moe_gemm_fp4_m16",
    "dsv_moe_gemv_fp4",
    "dsv_swiglu_quant",
    "dsv_moe_combine",
    "dsv_gather_rows",
];

/// Which hardware ceiling bounds a kernel's arithmetic, for the roofline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Peak {
    /// e4m3 tensor-core MMA (the W8A8 and fp4-expert GEMMs).
    Fp8Mma,
    /// bf16 tensor-core MMA (bf16 GEMMs, attention, the indexer score).
    Bf16Mma,
    /// fp32 CUDA-core FMA.
    Fp32,
    /// No meaningful arithmetic: bounded by memory traffic alone.
    Mem,
}

/// H200 SXM ceilings (dense): HBM3e bandwidth and tensor/CUDA-core throughput.
pub const HBM_BYTES_PER_S: f64 = 4.8e12;
pub fn peak_flops(p: Peak) -> f64 {
    match p {
        Peak::Fp8Mma => 1979e12,
        Peak::Bf16Mma => 989e12,
        Peak::Fp32 => 67e12,
        Peak::Mem => f64::INFINITY,
    }
}

/// The work one launch does, for the roofline: useful FLOPs and the compulsory DRAM bytes (each
/// operand read once, each output written once; re-reads a better kernel could avoid are not
/// counted, so the roofline is the floor, not what the current kernel could reach).
#[derive(Clone, Copy, Debug)]
pub struct Cost {
    pub flops: f64,
    pub bytes: f64,
    pub peak: Peak,
}

impl Cost {
    pub fn mem(bytes: f64) -> Cost {
        Cost { flops: 0.0, bytes, peak: Peak::Mem }
    }
    /// Roofline time floor, seconds.
    pub fn floor_s(&self) -> f64 {
        (self.bytes / HBM_BYTES_PER_S).max(self.flops / peak_flops(self.peak))
    }
}

/// Per-kernel totals: calls, measured ms, and summed cost.
#[derive(Default, Clone, Copy, Debug)]
pub struct KTotal {
    pub calls: u64,
    pub ms: f64,
    pub flops: f64,
    pub bytes: f64,
    pub floor_ms: f64,
    pub peak: Option<Peak>,
}

/// Per-kernel GPU time (`PLOW_DSV41_PROFILE=2`): an event pair around every launch, plus the
/// launch's declared [`Cost`] when the call site gave one.
pub struct KernelProf {
    pending: Vec<(&'static str, CudaEvent, CudaEvent, Option<Cost>)>,
    pub totals: HashMap<&'static str, KTotal>,
    /// Off: launches run without the event pair (for an unperturbed step time).
    pub on: bool,
}

impl Default for KernelProf {
    fn default() -> Self {
        KernelProf { pending: Vec::new(), totals: HashMap::new(), on: true }
    }
}

pub struct Kernels {
    pub dev: Arc<CudaBackend>,
    _module: crate::device::Module,
    fns: HashMap<&'static str, KernelFn>,
    pub prof: Option<RefCell<KernelProf>>,
    /// The cost the next launch declares (consumed by it).
    next_cost: Cell<Option<Cost>>,
}

impl Kernels {
    pub fn load(dev: Arc<CudaBackend>, cubin: &[u8], profile: bool) -> Result<Self> {
        let module = dev.module_load(cubin)?;
        match dev.module_global_u32(&module, "dsv41_abi_version")? {
            Some(v) if v == ABI_VERSION => {}
            got => {
                return Err(RuntimeError::Device(format!(
                    "dsv41: cubin ABI {got:?}, engine expects {ABI_VERSION}; rebuild with scripts/build_dsv41_sm90a.sh"
                )))
            }
        }
        let mut fns = HashMap::new();
        for &n in NAMES {
            fns.insert(n, dev.get_function(&module, n)?);
        }
        dev.set_max_dynamic_smem(fns["dsv_sparse_attn"], SA_SMEM)?;
        dev.set_max_dynamic_smem(fns["dsv_index_score"], IX_SMEM)?;
        for bm in [16, 32, 64] {
            dev.set_max_dynamic_smem(fns[moe_gemm_name(bm)], moe_smem(MOE_SMEM_MAX_K, bm))?;
        }
        dev.set_max_dynamic_smem(fns["dsv_gemm_w8a8"], W8_SMEM)?;
        Ok(Kernels { dev, _module: module, fns, prof: profile.then(|| RefCell::new(KernelProf::default())), next_cost: Cell::new(None) })
    }

    /// Declare the work of the next launch (for the roofline report; free when not profiling).
    pub fn cost(&self, c: Cost) {
        if self.prof.is_some() {
            self.next_cost.set(Some(c));
        }
    }

    pub fn launch(&self, name: &str, grid: [u32; 3], block: u32, smem: u32, args: &[A], s: &CudaStream) -> Result<()> {
        let cost = self.next_cost.take();
        let (&key, &f) = self
            .fns
            .get_key_value(name)
            .ok_or_else(|| RuntimeError::Device(format!("dsv41: no kernel {name}")))?;
        if args.len() > MAX_ARGS {
            return Err(RuntimeError::Device(format!("dsv41: {name} has {} args (> {MAX_ARGS})", args.len())));
        }
        let mut vals = [[0u8; 8]; MAX_ARGS];
        for (v, a) in vals.iter_mut().zip(args) {
            match *a {
                A::P(x) => *v = x.to_ne_bytes(),
                A::L(x) => *v = x.to_ne_bytes(),
                A::I(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
                A::F(x) => v[..4].copy_from_slice(&x.to_ne_bytes()),
            }
        }
        let mut ptrs = [std::ptr::null_mut::<c_void>(); MAX_ARGS];
        for (p, v) in ptrs.iter_mut().zip(vals.iter_mut()) {
            *p = v.as_mut_ptr() as *mut c_void;
        }
        let Some(prof) = self.prof.as_ref().filter(|p| p.borrow().on) else {
            return self.dev.launch_kernel_grid(f, grid, block, smem, &mut ptrs[..args.len()], Some(s));
        };
        let (e0, e1) = (self.dev.event_create(true)?, self.dev.event_create(true)?);
        self.dev.event_record(&e0, s)?;
        self.dev.launch_kernel_grid(f, grid, block, smem, &mut ptrs[..args.len()], Some(s))?;
        self.dev.event_record(&e1, s)?;
        prof.borrow_mut().pending.push((key, e0, e1, cost));
        Ok(())
    }

    /// Fold the finished launches' times into the totals (call after the stream is synced).
    pub fn prof_collect(&self) -> Result<()> {
        if let Some(prof) = &self.prof {
            let mut p = prof.borrow_mut();
            let pending = std::mem::take(&mut p.pending);
            for (name, e0, e1, cost) in pending {
                let ms = self.dev.event_elapsed_ms(&e0, &e1)? as f64;
                let t = p.totals.entry(name).or_default();
                t.calls += 1;
                t.ms += ms;
                if let Some(c) = cost {
                    t.flops += c.flops;
                    t.bytes += c.bytes;
                    t.floor_ms += c.floor_s() * 1e3;
                    t.peak = Some(c.peak);
                }
            }
        }
        Ok(())
    }

    /// C[m][n] (f32) = A[m][k] . W[n][k]^T, A/W bf16 or f32. Small output grids (the mHC mixes are
    /// 24 x 20480; the router and head at decode batch) take the one-block-per-output dot form, and
    /// few outputs over many rows the row form; everything else the 64x64 tiles (the dot form re-reads
    /// each A row once per output, which at prefill M is hundreds of reads).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32(&self, c: u64, a: u64, w: u64, m: usize, n: usize, k: usize, lda: usize, a_bf16: bool, w_bf16: bool, s: &CudaStream) -> Result<()> {
        let args = [A::P(c), A::P(a), A::P(w), A::I(m as i32), A::I(n as i32), A::I(k as i32), A::L(lda as i64), A::L(n as i64), A::I(a_bf16 as i32), A::I(w_bf16 as i32)];
        let (ab, wb) = (if a_bf16 { 2.0 } else { 4.0 }, if w_bf16 { 2.0 } else { 4.0 });
        let (mf, nf, kf) = (m as f64, n as f64, k as f64);
        self.cost(Cost { flops: 2.0 * mf * nf * kf, bytes: mf * kf * ab + nf * kf * wb + mf * nf * 4.0, peak: Peak::Fp32 });
        if n <= 32 && m > 64 {
            self.launch("dsv_gemm_f32_rows", [cdiv(m as u64, 4), 1, 1], 256, 0, &args, s)
        } else if m <= 16 {
            self.launch("dsv_gemm_f32_dot", [n as u32, m as u32, 1], 256, 0, &args, s)
        } else {
            self.launch("dsv_gemm_f32", [cdiv(n as u64, 64), cdiv(m as u64, 64), 1], 256, 0, &args, s)
        }
    }
}

/// `ceil(a / b)` as a grid dimension.
pub fn cdiv(a: u64, b: u64) -> u32 {
    a.div_ceil(b) as u32
}
