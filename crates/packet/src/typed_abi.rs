//! Strongly typed instruction builder for the Plow persistent device ISA.
//!
//! Eliminates untyped `i[0..7]` and `t[0..7]` indexing errors by providing
//! semantically validated argument structs and builder methods for all standard
//! operations executed by the device interpreter.

use crate::dev::{DevInst, DevOp, RESIDUAL_F32_FLAG, TENSOR_NONE};

/// Builder interface for typed instruction assembly.
#[derive(Clone, Copy, Debug, Default)]
pub struct InstructionBuilder;

/// Strongly typed arguments for FlashPrefill operations.
#[derive(Clone, Debug, PartialEq)]
pub struct FlashPrefillArgs {
    pub out: u32,
    pub q: u32,
    pub k: u32,
    pub v: u32,
    pub kv_cache: Option<u32>,
    pub lse: Option<u32>,
    pub seq_len: u32,
    pub batch_size: u32,
    pub num_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub max_seq_len: u32,
    pub sliding_window: Option<u32>,
    pub is_causal: bool,
    pub stride_kv: u32,
}

impl FlashPrefillArgs {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.out == TENSOR_NONE
            || self.q == TENSOR_NONE
            || self.k == TENSOR_NONE
            || self.v == TENSOR_NONE
        {
            return Err("FlashPrefill requires valid out, q, k, v tensor handles");
        }
        if self.head_dim == 0 || self.head_dim % 8 != 0 {
            return Err("FlashPrefill head_dim must be non-zero and divisible by 8");
        }
        if self.num_heads == 0 || self.kv_heads == 0 || self.num_heads % self.kv_heads != 0 {
            return Err("FlashPrefill num_heads must be divisible by kv_heads");
        }
        Ok(())
    }

    pub fn apply(&self, inst: &mut DevInst) {
        inst.op = DevOp::FlashPrefill as u16;
        inst.t[0] = self.out;
        inst.t[1] = self.q;
        inst.t[2] = self.k;
        inst.t[3] = self.v;
        inst.t[4] = self.kv_cache.unwrap_or(TENSOR_NONE);
        inst.t[5] = self.lse.unwrap_or(TENSOR_NONE);
        inst.t[6] = TENSOR_NONE;
        inst.t[7] = TENSOR_NONE;

        inst.i[0] = self.seq_len;
        inst.i[1] = self.batch_size;
        inst.i[2] = self.num_heads;
        inst.i[3] = self.kv_heads;
        inst.i[4] = self.max_seq_len;
        inst.i[5] = self.sliding_window.unwrap_or(0);
        inst.i[6] = self.head_dim;
        inst.i[7] = if self.is_causal { 1 } else { 0 };

        inst.j[0] = self.stride_kv;
    }

    pub fn build(self, blocks: u16) -> DevInst {
        let mut inst = DevInst {
            op: DevOp::FlashPrefill as u16,
            blocks,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [TENSOR_NONE; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        self.apply(&mut inst);
        inst
    }
}

/// Strongly typed arguments for FlashMerge operations (multi-split attention reduction).
#[derive(Clone, Debug, PartialEq)]
pub struct FlashMergeArgs {
    pub out: u32,
    pub partial: u32,
    pub lse: u32,
    pub seq_len: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub splits: u32,
}

impl FlashMergeArgs {
    pub fn apply(&self, inst: &mut DevInst) {
        inst.op = DevOp::FlashMerge as u16;
        inst.t[0] = self.out;
        inst.t[1] = self.partial;
        inst.t[2] = self.lse;
        inst.i[0] = self.seq_len;
        inst.i[1] = self.num_heads;
        inst.i[2] = self.splits;
        inst.i[3] = self.head_dim;
    }

    pub fn build(self, blocks: u16) -> DevInst {
        let mut inst = DevInst {
            op: DevOp::FlashMerge as u16,
            blocks,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [TENSOR_NONE; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        self.apply(&mut inst);
        inst
    }
}

/// Strongly typed arguments for Standard GEMM.
#[derive(Clone, Debug, PartialEq)]
pub struct GemmArgs {
    pub op: DevOp,
    pub out: u32,
    pub x: u32,
    pub wt: u32,
    pub bias: Option<u32>,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub row0: u32,
    pub c_row0: u32,
}

impl GemmArgs {
    pub fn apply(&self, inst: &mut DevInst) {
        inst.op = self.op as u16;
        inst.t[0] = self.out;
        inst.t[1] = self.x;
        inst.t[2] = self.wt;
        inst.t[3] = self.bias.unwrap_or(TENSOR_NONE);
        inst.i[0] = self.m;
        inst.i[1] = self.n;
        inst.i[2] = self.k;
        inst.i[4] = self.row0;
        inst.i[5] = self.c_row0;
    }

    pub fn build(self, blocks: u16) -> DevInst {
        let mut inst = DevInst {
            op: self.op as u16,
            blocks,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [TENSOR_NONE; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        self.apply(&mut inst);
        inst
    }
}

/// Strongly typed arguments for FP8 GEMM with block scales.
#[derive(Clone, Debug, PartialEq)]
pub struct GemmFp8MxArgs {
    pub out: u32,
    pub x: u32,
    pub wt: u32,
    pub scale: u32,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub row0: u32,
    pub c_row0: u32,
}

impl GemmFp8MxArgs {
    pub fn apply(&self, inst: &mut DevInst) {
        inst.op = DevOp::GemmFp8Mx as u16;
        inst.t[0] = self.out;
        inst.t[1] = self.x;
        inst.t[2] = self.wt;
        inst.t[3] = self.scale;
        inst.i[0] = self.m;
        inst.i[1] = self.n;
        inst.i[2] = self.k;
        inst.i[4] = self.row0;
        inst.i[5] = self.c_row0;
    }

    pub fn build(self, blocks: u16) -> DevInst {
        let mut inst = DevInst {
            op: DevOp::GemmFp8Mx as u16,
            blocks,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [TENSOR_NONE; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        self.apply(&mut inst);
        inst
    }
}

/// Strongly typed arguments for RMSNorm and LayerNorm operations.
#[derive(Clone, Debug, PartialEq)]
pub struct RmsNormArgs {
    pub op: DevOp,
    pub out: u32,
    pub x: u32,
    pub weight: u32,
    pub residual_in: Option<u32>,
    pub residual_out: Option<u32>,
    pub rows: u32,
    pub hidden: u32,
    pub eps: f32,
    pub residual_f32: bool,
}

impl RmsNormArgs {
    pub fn simple(out: u32, x: u32, weight: u32, rows: u32, hidden: u32, eps: f32) -> Self {
        Self {
            op: DevOp::RmsNorm,
            out,
            x,
            weight,
            residual_in: None,
            residual_out: None,
            rows,
            hidden,
            eps,
            residual_f32: false,
        }
    }

    pub fn apply(&self, inst: &mut DevInst) {
        inst.op = self.op as u16;
        inst.t[0] = self.out;
        inst.t[1] = self.x;
        inst.t[2] = self.weight;
        inst.t[3] = self.residual_in.unwrap_or(TENSOR_NONE);
        inst.t[4] = self.residual_out.unwrap_or(TENSOR_NONE);

        inst.i[0] = self.rows;
        inst.i[1] = self.hidden;
        if self.residual_f32 {
            inst.i[6] |= RESIDUAL_F32_FLAG;
        }
        inst.f[0] = self.eps;
    }

    pub fn build(self, blocks: u16) -> DevInst {
        let mut inst = DevInst {
            op: self.op as u16,
            blocks,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [TENSOR_NONE; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        self.apply(&mut inst);
        inst
    }
}

/// Strongly typed arguments for HeadNormRope rotary embeddings.
#[derive(Clone, Debug, PartialEq)]
pub struct HeadNormRopeArgs {
    pub out: u32,
    pub x: u32,
    pub gamma: u32,
    pub cos: u32,
    pub sin: u32,
    pub pos: u32,
    pub ntok: u32,
    pub nhead: u32,
    pub hd: u32,
    pub out_row0: u32,
    pub flags: u32,
    pub pair_mode: u32,
    pub n_batch_kv: u32,
    pub eps: f32,
    pub out_stride: u32,
    pub kv_mask: u32,
}

pub type RopeArgs = HeadNormRopeArgs;

impl HeadNormRopeArgs {
    pub fn simple_rope(
        out: u32,
        x: u32,
        cos: u32,
        sin: u32,
        pos: u32,
        ntok: u32,
        nhead: u32,
        hd: u32,
    ) -> Self {
        Self {
            out,
            x,
            gamma: TENSOR_NONE,
            cos,
            sin,
            pos,
            ntok,
            nhead,
            hd,
            out_row0: 0,
            flags: 1, // skip_norm
            pair_mode: 0,
            n_batch_kv: 0,
            eps: 0.0,
            out_stride: 0,
            kv_mask: 0,
        }
    }

    pub fn apply(&self, inst: &mut DevInst) {
        inst.op = DevOp::HeadNormRope as u16;
        inst.t[0] = self.out;
        inst.t[1] = self.x;
        inst.t[2] = self.gamma;
        inst.t[3] = self.cos;
        inst.t[4] = self.sin;
        inst.t[5] = self.pos;
        inst.i[0] = self.ntok;
        inst.i[1] = self.nhead;
        inst.i[2] = self.hd;
        inst.i[3] = self.out_row0;
        inst.i[4] = self.flags;
        inst.i[5] = self.pair_mode;
        inst.i[6] = self.n_batch_kv;
        inst.f[0] = self.eps;
        inst.j[0] = self.out_stride;
        inst.j[1] = self.kv_mask;
    }

    pub fn build(self, blocks: u16) -> DevInst {
        let mut inst = DevInst {
            op: DevOp::HeadNormRope as u16,
            blocks,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [TENSOR_NONE; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        self.apply(&mut inst);
        inst
    }
}

/// Strongly typed arguments for SwiGLU / Gated MLP activation operations.
#[derive(Clone, Debug, PartialEq)]
pub struct SwiGluArgs {
    pub out: u32,
    pub gate: u32,
    pub up: u32,
    pub rows: u32,
    pub hidden: u32,
}

impl SwiGluArgs {
    pub fn apply(&self, inst: &mut DevInst) {
        inst.op = DevOp::Glu as u16;
        inst.t[0] = self.out;
        inst.t[1] = self.gate;
        inst.t[2] = self.up;
        inst.i[0] = self.rows;
        inst.i[1] = self.hidden;
    }

    pub fn build(self, blocks: u16) -> DevInst {
        let mut inst = DevInst {
            op: DevOp::Glu as u16,
            blocks,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [TENSOR_NONE; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        self.apply(&mut inst);
        inst
    }
}

/// Strongly typed arguments for MoE Router / Top-K Dispatch operations.
#[derive(Clone, Debug, PartialEq)]
pub struct MoeRouterArgs {
    pub out_tokens: u32,
    pub x: u32,
    pub router_weight: u32,
    pub score_bias: Option<u32>,
    pub tokens: u32,
    pub hidden: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub route_scale: f32,
}

impl MoeRouterArgs {
    pub fn apply(&self, inst: &mut DevInst) {
        inst.op = DevOp::GemmF32 as u16;
        inst.t[0] = self.out_tokens;
        inst.t[1] = self.x;
        inst.t[2] = self.router_weight;
        inst.t[3] = self.score_bias.unwrap_or(TENSOR_NONE);
        inst.i[0] = self.tokens;
        inst.i[1] = self.num_experts;
        inst.i[2] = self.hidden;
        inst.i[3] = self.top_k;
        inst.f[0] = self.route_scale;
    }

    pub fn build(self, blocks: u16) -> DevInst {
        let mut inst = DevInst {
            op: DevOp::GemmF32 as u16,
            blocks,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [TENSOR_NONE; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        self.apply(&mut inst);
        inst
    }
}

/// Strongly typed arguments for KDA chunk weight-update recurrence (DevOp::KdaChunkWu).
#[derive(Clone, Debug, PartialEq)]
pub struct KdaChunkWuArgs {
    pub cw: u32,
    pub cu: u32,
    pub ainv: u32,
    pub mix1: u32,
    pub mix2: u32,
    pub gate: u32,
    pub beta: u32,
    pub mix0: u32,
    pub t: u32,
    pub nh: u32,
    pub hd: u32,
    pub flags: u32,
}

pub type KdaScanArgs = KdaChunkWuArgs;

impl KdaChunkWuArgs {
    pub fn apply(&self, inst: &mut DevInst) {
        inst.op = DevOp::KdaChunkWu as u16;
        inst.t[0] = self.cw;
        inst.t[1] = self.cu;
        inst.t[2] = self.ainv;
        inst.t[3] = self.mix1;
        inst.t[4] = self.mix2;
        inst.t[5] = self.gate;
        inst.t[6] = self.beta;
        inst.t[7] = self.mix0;
        inst.i[0] = self.t;
        inst.i[1] = self.nh;
        inst.i[2] = self.hd;
        inst.i[4] = self.flags;
    }

    pub fn build(self, blocks: u16) -> DevInst {
        let mut inst = DevInst {
            op: DevOp::KdaChunkWu as u16,
            blocks,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [TENSOR_NONE; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        self.apply(&mut inst);
        inst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flash_prefill_args_packing() {
        let args = FlashPrefillArgs {
            out: 1,
            q: 2,
            k: 3,
            v: 4,
            kv_cache: Some(5),
            lse: Some(6),
            seq_len: 1024,
            batch_size: 1,
            num_heads: 32,
            kv_heads: 8,
            head_dim: 128,
            max_seq_len: 4096,
            sliding_window: Some(512),
            is_causal: true,
            stride_kv: 64,
        };
        assert!(args.validate().is_ok());
        let inst = args.build(110);
        assert_eq!(inst.op, DevOp::FlashPrefill as u16);
        assert_eq!(inst.blocks, 110);
        assert_eq!(inst.t[0], 1);
        assert_eq!(inst.t[4], 5);
        assert_eq!(inst.t[5], 6);
        assert_eq!(inst.i[0], 1024);
        assert_eq!(inst.i[2], 32);
        assert_eq!(inst.i[3], 8);
        assert_eq!(inst.i[5], 512);
        assert_eq!(inst.i[6], 128);
        assert_eq!(inst.i[7], 1);
        assert_eq!(inst.j[0], 64);
    }

    #[test]
    fn rms_norm_args_packing() {
        let args = RmsNormArgs::simple(10, 11, 12, 128, 4096, 1e-6);
        let inst = args.build(64);
        assert_eq!(inst.op, DevOp::RmsNorm as u16);
        assert_eq!(inst.blocks, 64);
        assert_eq!(inst.t[0], 10);
        assert_eq!(inst.t[1], 11);
        assert_eq!(inst.t[2], 12);
        assert_eq!(inst.i[0], 128);
        assert_eq!(inst.i[1], 4096);
        assert_eq!(inst.f[0], 1e-6);
    }

    #[test]
    fn gemm_fp8_mx_args_packing() {
        let args = GemmFp8MxArgs {
            out: 100,
            x: 101,
            wt: 102,
            scale: 103,
            m: 256,
            n: 4096,
            k: 4096,
            row0: 0,
            c_row0: 0,
        };
        let inst = args.build(80);
        assert_eq!(inst.op, DevOp::GemmFp8Mx as u16);
        assert_eq!(inst.t[0], 100);
        assert_eq!(inst.t[3], 103);
        assert_eq!(inst.i[0], 256);
        assert_eq!(inst.i[1], 4096);
        assert_eq!(inst.i[2], 4096);
    }
}
