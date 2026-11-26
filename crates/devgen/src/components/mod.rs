//! Emitter components (devgen 2.0): one block family each, configured from model values and
//! checked against the target's interpreter arms (`kernelcaps::arms`) when built, so a new
//! checkpoint of a known family is configuration, and one the interpreter cannot run is refused
//! by naming the op and the arm it lacks.
//!
//! A component emits into a [`Program`] (one prefill bucket or decode rung) and names its
//! checkpoint tensors from the module prefix it is handed; weight-quantization checks and the
//! projection route stay with the model, passed in as the projection callback.
pub(crate) mod fp8_proj;
pub(crate) mod gqa_attention;
pub(crate) mod mamba2_mixer;
pub(crate) mod nvfp4_moe;

use packet::devbuild::Builder;
use packet::typed_abi::TypedOp;

/// A projection `out[rows, n] = src[rows, k] . module.weight^T` the model routes (FP8 per-tensor,
/// BF16, ...): `(program, out, src, module, n, k, dep) -> instruction`.
pub(crate) type Proj<'a> = dyn FnMut(&mut Program, u32, u32, &str, u32, u32, u32) -> u32 + 'a;

/// One program's emit state: the builder and what every component reads about the program.
pub(crate) struct Program {
    pub b: Builder,
    pub prefill: bool,
    /// Rows of this program: the bucket (prefill) or the rung (decode).
    pub rows: u32,
    /// Decode slots (the widest rung): slot-major state is sized by it.
    pub slots: u32,
    pub ctx: u32,
    /// `in.active` (decode) or the all-ones `in.prefill_active` (prefill).
    pub active: u32,
    pub pos: u32,
    pub kvlen: u32,
    /// The live activation quant of the FP8 projections ([`fp8_proj`]).
    pub quant: Option<fp8_proj::Quant>,
    /// The target ISA: its cuBLASLt FP8 scale modes pick the prefill projection route.
    pub isa: Option<hwspec::IsaLevel>,
    /// A packed-prefill bucket (`PLOW_EMIT_PACKED_PREFILL`): up to `slots` requests share the
    /// launch (per-request scratch, attention in its own segment).
    pub packed: bool,
}

impl Program {
    /// The bf16 activation `act.<name>` of this program's rows.
    pub fn act(&mut self, name: &str, width: u32) -> u32 {
        self.b.tensor(
            &format!("act.{name}"),
            u64::from(self.rows) * u64::from(width) * 2,
        )
    }

    /// Emits `args` on every block.
    pub fn emit<A: TypedOp>(&mut self, args: A, deps: &[u32]) -> u32 {
        let cus = self.b.all();
        args.emit(&mut self.b, cus, deps)
    }
}

/// A bf16 linear (`[n, k]` weight tensor `weight`): GEMV in decode, the bf16 GEMM in prefill.
pub(crate) fn bf16_proj(
    p: &mut Program,
    out: u32,
    src: u32,
    weight: &str,
    n: u32,
    k: u32,
    dep: u32,
) -> u32 {
    let w = p.b.tensor(weight, u64::from(n) * u64::from(k) * 2);
    let rows = p.rows;
    let cus = p.b.all();
    if p.prefill {
        let args = packet::typed_abi::GemmArgs {
            C: out,
            A: src,
            B: w,
            M: rows,
            N: n,
            K: k,
            ..Default::default()
        };
        args.emit(&mut p.b, cus, &[dep])
    } else {
        let args = packet::typed_abi::GemvArgs {
            C: out,
            x: src,
            W: w,
            M: rows,
            N: n,
            K: k,
            ..Default::default()
        };
        args.emit(&mut p.b, cus, &[dep])
    }
}
