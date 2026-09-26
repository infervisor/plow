//! Typed launches of the DeepSeek-V4.1 sm_90a cubin (`runtime/nvidia/dsv41/dsv41_all.cu`).
//!
//! One [`Kernels`] per device. Every launch goes on the stage's stream. Argument order and meaning
//! mirror the `extern "C"` kernel signatures exactly; see the .cu files for the math.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use crate::device::cuda::{CudaBackend, CudaStream, KernelFn};
use crate::device::Backend;
use crate::error::{Result, RuntimeError};

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
pub const SA_SMEM: u32 = ((64 * 520 + 64 * 520 + 64 * 72) * 2 + 4 * 64 * 4 * 2 + 64 * 4) as u32;

const NAMES: &[&str] = &[
    "dsv_rmsnorm",
    "dsv_act_quant_fp8",
    "dsv_fp4_fakequant",
    "dsv_gemm_w8a8",
    "dsv_gemm_bf16w",
    "dsv_gemm_f32",
    "dsv_gemm_f32_dot",
    "dsv_rope",
    "dsv_sparse_attn",
    "dsv_attn_index",
    "dsv_compress_pool_prefill",
    "dsv_compress_pool_decode",
    "dsv_scatter_rows",
    "dsv_copy_rows",
    "dsv_row_rsqrt",
    "dsv_hc_sinkhorn",
    "dsv_hc_pre",
    "dsv_hc_post",
    "dsv_engram_gate",
    "dsv_embed_hc",
    "dsv_argmax",
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
    "dsv_swiglu_quant",
    "dsv_moe_combine",
    "dsv_gather_rows",
];

pub struct Kernels {
    pub dev: Arc<CudaBackend>,
    _module: crate::device::Module,
    fns: HashMap<&'static str, KernelFn>,
}

impl Kernels {
    pub fn load(dev: Arc<CudaBackend>, cubin: &[u8]) -> Result<Self> {
        let module = dev.module_load(cubin)?;
        let mut fns = HashMap::new();
        for &n in NAMES {
            fns.insert(n, dev.get_function(&module, n)?);
        }
        dev.set_max_dynamic_smem(fns["dsv_sparse_attn"], SA_SMEM)?;
        dev.set_max_dynamic_smem(fns["dsv_index_score"], IX_SMEM)?;
        Ok(Kernels { dev, _module: module, fns })
    }

    pub fn launch(&self, name: &str, grid: [u32; 3], block: u32, smem: u32, args: &[A], s: &CudaStream) -> Result<()> {
        let f = *self
            .fns
            .get(name)
            .ok_or_else(|| RuntimeError::Device(format!("dsv41: no kernel {name}")))?;
        let mut vals: Vec<[u8; 8]> = args
            .iter()
            .map(|a| match *a {
                A::P(v) => v.to_ne_bytes(),
                A::L(v) => v.to_ne_bytes(),
                A::I(v) => {
                    let mut b = [0u8; 8];
                    b[..4].copy_from_slice(&v.to_ne_bytes());
                    b
                }
                A::F(v) => {
                    let mut b = [0u8; 8];
                    b[..4].copy_from_slice(&v.to_ne_bytes());
                    b
                }
            })
            .collect();
        let mut ptrs: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr() as *mut c_void).collect();
        self.dev.launch_kernel_grid(f, grid, block, smem, &mut ptrs, Some(s))
    }
}

impl Kernels {
    /// C[m][n] (f32) = A[m][k] . W[n][k]^T, A/W bf16 or f32. Small output grids (the mHC mixes are
    /// 24 x 20480; the router and head at small batch) take the one-block-per-output dot form: the
    /// 64x64-tiled kernel would put a single block on the whole K loop.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32(&self, c: u64, a: u64, w: u64, m: usize, n: usize, k: usize, lda: usize, a_bf16: bool, w_bf16: bool, s: &CudaStream) -> Result<()> {
        let args = [A::P(c), A::P(a), A::P(w), A::I(m as i32), A::I(n as i32), A::I(k as i32), A::L(lda as i64), A::L(n as i64), A::I(a_bf16 as i32), A::I(w_bf16 as i32)];
        if m * n <= 1 << 20 {
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
