//! Typed operand views of [`DevInst`]: one struct per op, one field per named slot.
//!
//! A field is named exactly as [`crate::slots`] names its slot (case included, so `K` the
//! reduction width and `k` the top-k stay distinct), and `slots`' own drift test ties those names
//! to the [`DevOp`] doc comments. [`tests::fields_are_the_slot_table`] checks every struct here
//! against the slot table, so a typed builder cannot write an operand the interpreter reads under
//! another name — the defect `FlashPrefillArgs` carried when it predated the split-K epilogue
//! (`t0 = out`, `i7 = is_causal` against the live `t0 = Opart`, `i7 = nsplit`).
//!
//! Defaults are the builder's: tensors [`TENSOR_NONE`], integers and floats zero. `apply` writes
//! every field, so `Args { .., ..Default::default() }` inside [`crate::devbuild::Builder::emit`]
//! produces the same bytes as assigning the named slots by hand. Undocumented immediates (bit
//! flags some ops take in otherwise unnamed slots) stay the caller's to set after `apply`.
#![allow(non_snake_case)]

use crate::dev::{DevInst, DevOp, TENSOR_NONE};
use crate::devbuild::Builder;

/// A slot of [`DevInst`]: `t[n]`, `i[n]`, `f[n]` or `j[n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    T(usize),
    I(usize),
    F(usize),
    J(usize),
}

/// One op's typed operands.
pub trait TypedOp: Sized {
    const OP: DevOp;
    /// `(field, slot)` for every field, in declaration order.
    const SLOTS: &'static [(&'static str, Slot)];
    /// Writes every field into its slot (the opcode is the builder's).
    fn apply(&self, d: &mut DevInst);

    /// Emits the op on `cus` after `deps`.
    fn emit(self, b: &mut Builder, cus: Vec<u32>, deps: &[u32]) -> u32 {
        b.emit(Self::OP, cus, deps, |d| self.apply(d))
    }

    /// Emits as `op`, a tile or arm twin sharing this op's operands (`Gemm` -> `GemmSmall`).
    fn emit_as(self, op: DevOp, b: &mut Builder, cus: Vec<u32>, deps: &[u32]) -> u32 {
        b.emit(op, cus, deps, |d| self.apply(d))
    }

    /// A free-standing instruction over `blocks` slices.
    fn build(self, blocks: u16) -> DevInst {
        let mut inst = DevInst {
            op: Self::OP as u16,
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

macro_rules! typed_op {
    ($(#[$meta:meta])* $name:ident = $op:ident {
        $(t $tn:ident = $ti:literal,)*
        $(i $iname:ident = $ii:literal,)*
        $(f $fname:ident = $fi:literal,)*
        $(j $jn:ident = $ji:literal,)*
    }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name {
            $(pub $tn: u32,)*
            $(pub $iname: u32,)*
            $(pub $fname: f32,)*
            $(pub $jn: u32,)*
        }

        impl Default for $name {
            fn default() -> Self {
                Self {
                    $($tn: TENSOR_NONE,)*
                    $($iname: 0,)*
                    $($fname: 0.0,)*
                    $($jn: 0,)*
                }
            }
        }

        impl TypedOp for $name {
            const OP: DevOp = DevOp::$op;
            const SLOTS: &'static [(&'static str, Slot)] = &[
                $((stringify!($tn), Slot::T($ti)),)*
                $((stringify!($iname), Slot::I($ii)),)*
                $((stringify!($fname), Slot::F($fi)),)*
                $((stringify!($jn), Slot::J($ji)),)*
            ];
            fn apply(&self, d: &mut DevInst) {
                $(d.t[$ti] = self.$tn;)*
                $(d.i[$ii] = self.$iname;)*
                $(d.f[$fi] = self.$fname;)*
                $(d.j[$ji] = self.$jn;)*
            }
        }
    };
}

/// Builder namespace kept for the re-export; the typed structs carry their own `emit`.
#[derive(Clone, Copy, Debug, Default)]
pub struct InstructionBuilder;

// ===== Attention =====

typed_op! {
    /// Split-K flash prefill; `O_final` present = fused one-split bf16 epilogue (no merge).
    FlashPrefillArgs = FlashPrefill {
        t Opart = 0, t mlpart = 1, t Q = 2, t K = 3, t V = 4, t O_final = 5,
        i n_q = 0, i n_kv = 1, i n_head = 2, i n_kv_head = 3, i q_pos0 = 4, i window = 5,
        i hd = 6, i nsplit = 7,
        f scale = 0,
        j kv_stride = 0, j kv_mask = 1,
    }
}

typed_op! {
    FlashDecodeArgs = FlashDecode {
        t Opart = 0, t mlpart = 1, t Q = 2, t K = 3, t V = 4, t kv_len = 5, t decode_slot = 6,
        t kv_scale = 7,
        i n_batch = 0, i n_head = 1, i n_kv_head = 2, i kv_stride = 3, i window = 4,
        i nsplit = 5, i hd = 6, i kv_mask = 7,
        f scale = 0,
    }
}

typed_op! {
    FlashMergeArgs = FlashMerge {
        t O = 0, t Opart = 1, t mlpart = 2, t sinks = 3,
        i n_batch = 0, i n_head = 1, i nsplit = 2, i hd = 3,
    }
}

typed_op! {
    HeadNormRopeArgs = HeadNormRope {
        t out = 0, t x = 1, t gamma = 2, t cos = 3, t sin = 4, t pos = 5,
        i ntok = 0, i nhead = 1, i hd = 2, i out_row0 = 3, i flags = 4, i pair_mode = 5,
        i n_batch_kv = 6,
        f eps = 0,
        j out_stride = 0, j kv_mask = 1,
    }
}

pub type RopeArgs = HeadNormRopeArgs;

impl HeadNormRopeArgs {
    /// RoPE only (`flags` bit 0 = skip the norm).
    #[allow(clippy::too_many_arguments)]
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
            cos,
            sin,
            pos,
            ntok,
            nhead,
            hd,
            flags: 1,
            ..Default::default()
        }
    }
}

typed_op! {
    /// The sm_120 / Qwen head norm + rotary + KV store (`ctx != 0`: a cache writer).
    QwenHeadNormRopeArgs = QwenHeadNormRope {
        t out = 0, t x = 1, t gamma = 2, t cos = 3, t sin = 4, t pos = 5, t active = 6,
        t rowslot = 7,
        i H = 0, i D = 1, i rotary = 2, i rows = 3, i ctx = 4, i normalize = 5, i prefill = 6,
        i layout = 7,
        f eps = 0, f gamma_offset = 1,
        j source = 1,
    }
}

// ===== Dense projections =====

typed_op! {
    GemmArgs = Gemm {
        t C = 0, t A = 1, t B = 2, t bias = 7,
        i M = 0, i N = 1, i K = 2,
    }
}

typed_op! {
    GemvArgs = Gemv {
        t C = 0, t x = 1, t W = 2, t rms = 3, t gamma = 4, t bias = 7,
        i M = 0, i N = 1, i K = 2, i norm = 3,
        f eps = 0,
    }
}

typed_op! {
    /// W8A16 decode GEMV: e4m3 weight rows, per-channel `w_scale`.
    GemvFp8Args = GemvFp8 {
        t C = 0, t x = 1, t W = 2, t resid_out = 3, t b = 4, t w_scale = 5, t gamma_b = 6,
        t gamma_n = 7,
        i M = 0, i N = 1, i K = 2, i nrn = 3, i a_row0 = 4,
        f eps = 0, f scale = 1,
    }
}

typed_op! {
    /// W8A8 GEMM; `scale_mode` per [`crate::dev::FP8_TENSOR_SCALE`] / `FP8_UNIT_SCALE`.
    GemmFp8Args = GemmFp8 {
        t C = 0, t A = 1, t B = 2, t a_scale = 3, t w_scale = 4,
        i M = 0, i N = 1, i K = 2, i a_row0 = 4, i scale_mode = 5,
    }
}

typed_op! {
    QuantFp8Args = QuantFp8 {
        t xq = 0, t x = 1, t a_scale = 2, t gate = 3, t up = 4,
        i M = 0, i K = 1, i act = 2, i scale_mode = 5,
    }
}

typed_op! {
    /// Outer scaling of a unit-scale FP8 GEMM's output.
    RowColScaleArgs = RowColScale {
        t out = 0, t x = 1, t row = 2, t col = 3,
        i M = 0, i N = 1,
    }
}

typed_op! {
    GemmFp8MxArgs = GemmFp8Mx {
        t out = 0, t x = 1, t w = 2, t scale = 3,
        i T = 0, i N = 1, i K = 2,
    }
}

// ===== Norms and pointwise =====

typed_op! {
    RmsNormArgs = RmsNorm {
        t out = 0, t x = 1, t gamma = 2, t xq = 3, t ascale = 4,
        i rows = 0, i feat = 1,
        f eps = 0,
    }
}

typed_op! {
    QwenRmsNormArgs = QwenRmsNorm {
        t out = 0, t x = 1, t gamma = 2, t active = 3, t radd = 4,
        i width = 0, i B = 1,
        f eps = 0, f gamma_offset = 1,
    }
}

typed_op! {
    ResidualArgs = Residual {
        t out = 0, t a = 1, t b = 2, t pre = 3,
        i n = 0,
        f scale = 0,
    }
}

typed_op! {
    GluArgs = Glu {
        t out = 0, t gate = 1, t up = 2,
        i n = 0, i act = 1,
        f alpha = 0, f limit = 1,
    }
}

// ===== Mamba-2 =====

typed_op! {
    Mamba2ConvArgs = Mamba2Conv {
        t out = 0, t x = 1, t weight = 2, t bias = 3, t history = 4, t active = 5,
        t rowslot = 6, t rec = 7,
        i C = 0, i W = 1, i rows = 2, i T = 3, i prefill = 4, i depth = 5,
        j source = 1,
    }
}

typed_op! {
    /// Span verify ORs its operands into the high halves of `state_bytes`, `seed_word`, `layer`.
    Mamba2StepArgs = Mamba2Step {
        t y = 0, t xbc = 1, t proj = 2, t A_log = 3, t D = 4, t dt_bias = 5, t state = 6,
        t active = 7,
        i H = 0, i P = 1, i N = 2, i G = 3, i B = 4, i state_bytes = 5, i seed_word = 6,
        i layer = 7,
        f dt_lo = 0, f dt_hi = 1,
        j dt_source = 1,
    }
}

typed_op! {
    Mamba2GatedNormArgs = Mamba2GatedNorm {
        t out = 0, t y = 1, t z = 2, t gamma = 3, t active = 4, t in_scale = 5,
        i groups = 0, i group = 1, i rows = 2, i fp8_out = 3,
        f eps = 0,
        j source = 1,
    }
}

typed_op! {
    Mamba2ChunkStateArgs = Mamba2ChunkState {
        t cstate = 0, t xbc = 1, t proj = 2, t A_log = 3, t dt_bias = 4, t dtv = 5, t cum = 6,
        i H = 0, i P = 1, i R = 2, i T = 3, i N = 4, i G = 5,
        f dt_lo = 0, f dt_hi = 1,
        j dt_source = 1,
    }
}

typed_op! {
    Mamba2ChunkPassArgs = Mamba2ChunkPass {
        t cstate = 0, t cum = 1, t state = 2,
        i H = 0, i P = 1, i R = 2, i T = 3, i N = 4, i state_bytes = 5,
    }
}

typed_op! {
    Mamba2ChunkOutArgs = Mamba2ChunkOut {
        t y = 0, t xbc = 1, t cstate = 2, t cum = 3, t dtv = 4, t D = 5,
        i H = 0, i P = 1, i R = 2, i T = 3, i N = 4, i G = 5,
    }
}

// ===== MoE =====

typed_op! {
    MoeRouterArgs = MoeRouter {
        i H = 0, i n_exp = 1, i k = 2, i flags = 3,
        f route_scale = 0,
    }
}

typed_op! {
    MoeRouterLogitsF32Args = MoeRouterLogitsF32 {
        t logits = 0, t x = 1, t W = 2,
        i T = 0, i E = 1, i H = 2, i KS = 3,
    }
}

typed_op! {
    MoeRouterSigmoidTopkArgs = MoeRouterSigmoidTopk {
        t table = 0, t logits = 1, t bias = 2, t active = 3,
        i T = 0, i E = 1, i k = 2, i KS = 3, i flags = 4,
        f scale = 0,
    }
}

typed_op! {
    MoeUpNvfp4Args = MoeUpNvfp4 {
        t fu = 0, t x = 1, t table = 2, t W = 3, t S = 4, t gs = 5,
        i T = 0, i k = 1, i I = 2, i K = 3, i E = 4,
    }
}

typed_op! {
    MoeDownNvfp4Args = MoeDownNvfp4 {
        t out = 0, t fu = 1, t table = 2, t W = 3, t S = 4, t gs = 5,
        i T = 0, i k = 1, i H = 2, i I = 3, i E = 4, i mode = 5,
    }
}

typed_op! {
    MoeAlignNvfp4Args = MoeAlignNvfp4 {
        t meta = 0, t table = 1, t row_token = 2, t row_partidx = 3, t row_gate = 4,
        i T = 0, i E = 1, i k = 2,
    }
}

typed_op! {
    MoeUpNvfp4PfArgs = MoeUpNvfp4Pf {
        t fu_g = 0, t x = 1, t meta = 2, t W = 3, t S = 4, t gs = 5, t row_token = 6, t s2 = 7,
        i T = 0, i k = 1, i I = 2, i K = 3, i E = 4, i w4a4 = 5, i n2 = 6,
    }
}

typed_op! {
    MoeDownNvfp4PfArgs = MoeDownNvfp4Pf {
        t out = 0, t fu_g = 1, t meta = 2, t W = 3, t S = 4, t gs = 5, t row_partidx = 6,
        t row_gate = 7,
        i T = 0, i k = 1, i H = 2, i I = 3, i E = 4, i mode = 5, i w4a4 = 6,
    }
}

typed_op! {
    MoeQuantNvfp4Args = MoeQuantNvfp4 {
        t q = 0, t x = 1, t in_scale = 2,
        i R = 0, i K = 1, i n = 2,
    }
}

typed_op! {
    MoeCombinePfArgs = MoeCombinePf {
        t out = 0, t residual = 1, t shared = 2, t part = 3, t table = 4,
        i H = 0, i k = 1, i T = 2, i t_row0 = 3, i det = 4, i bf16 = 5,
    }
}

// ===== KDA =====

typed_op! {
    KdaChunkWuArgs = KdaChunkWu {
        t W = 0, t U = 1, t Ainv = 2, t k = 3, t v = 4, t g_prefix = 5, t beta = 6, t q = 7,
        i T = 0, i H = 1, i D = 2, i V = 3, i qpre = 4,
        f scale = 0,
    }
}

pub type KdaScanArgs = KdaChunkWuArgs;

/// Every typed op: `(op, slots)`.
pub const ALL: &[(DevOp, &[(&str, Slot)])] = &[
    (FlashPrefillArgs::OP, FlashPrefillArgs::SLOTS),
    (FlashDecodeArgs::OP, FlashDecodeArgs::SLOTS),
    (FlashMergeArgs::OP, FlashMergeArgs::SLOTS),
    (HeadNormRopeArgs::OP, HeadNormRopeArgs::SLOTS),
    (QwenHeadNormRopeArgs::OP, QwenHeadNormRopeArgs::SLOTS),
    (GemmArgs::OP, GemmArgs::SLOTS),
    (GemvArgs::OP, GemvArgs::SLOTS),
    (GemvFp8Args::OP, GemvFp8Args::SLOTS),
    (GemmFp8Args::OP, GemmFp8Args::SLOTS),
    (QuantFp8Args::OP, QuantFp8Args::SLOTS),
    (RowColScaleArgs::OP, RowColScaleArgs::SLOTS),
    (GemmFp8MxArgs::OP, GemmFp8MxArgs::SLOTS),
    (RmsNormArgs::OP, RmsNormArgs::SLOTS),
    (QwenRmsNormArgs::OP, QwenRmsNormArgs::SLOTS),
    (ResidualArgs::OP, ResidualArgs::SLOTS),
    (GluArgs::OP, GluArgs::SLOTS),
    (Mamba2ConvArgs::OP, Mamba2ConvArgs::SLOTS),
    (Mamba2StepArgs::OP, Mamba2StepArgs::SLOTS),
    (Mamba2GatedNormArgs::OP, Mamba2GatedNormArgs::SLOTS),
    (Mamba2ChunkStateArgs::OP, Mamba2ChunkStateArgs::SLOTS),
    (Mamba2ChunkPassArgs::OP, Mamba2ChunkPassArgs::SLOTS),
    (Mamba2ChunkOutArgs::OP, Mamba2ChunkOutArgs::SLOTS),
    (MoeRouterArgs::OP, MoeRouterArgs::SLOTS),
    (MoeRouterLogitsF32Args::OP, MoeRouterLogitsF32Args::SLOTS),
    (
        MoeRouterSigmoidTopkArgs::OP,
        MoeRouterSigmoidTopkArgs::SLOTS,
    ),
    (MoeUpNvfp4Args::OP, MoeUpNvfp4Args::SLOTS),
    (MoeDownNvfp4Args::OP, MoeDownNvfp4Args::SLOTS),
    (MoeAlignNvfp4Args::OP, MoeAlignNvfp4Args::SLOTS),
    (MoeUpNvfp4PfArgs::OP, MoeUpNvfp4PfArgs::SLOTS),
    (MoeDownNvfp4PfArgs::OP, MoeDownNvfp4PfArgs::SLOTS),
    (MoeQuantNvfp4Args::OP, MoeQuantNvfp4Args::SLOTS),
    (MoeCombinePfArgs::OP, MoeCombinePfArgs::SLOTS),
    (KdaChunkWuArgs::OP, KdaChunkWuArgs::SLOTS),
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slots::slots_for;

    /// Every field names its slot exactly as the slot table does, and every named slot of the
    /// table has its field: a typed struct is the slot table's row, nothing more or less.
    #[test]
    fn fields_are_the_slot_table() {
        let mut wrong = Vec::new();
        for &(op, fields) in ALL {
            let s = slots_for(op);
            let mut want: Vec<(String, Slot)> = Vec::new();
            for (n, t) in s.t.iter().enumerate() {
                want.extend(t.map(|t| (t.name.to_string(), Slot::T(n))));
            }
            // Integer and float names keep their optional `?` in the table.
            let name = |n: &str| n.trim_end_matches('?').to_string();
            for (n, i) in s.i.iter().enumerate() {
                want.extend(i.map(|i| (name(i), Slot::I(n))));
            }
            want.extend(s.f0.map(|f| (name(f), Slot::F(0))));
            want.extend(s.f1.map(|f| (name(f), Slot::F(1))));
            want.extend(s.j0.map(|j| (name(j), Slot::J(0))));
            want.extend(s.j1.map(|j| (name(j), Slot::J(1))));
            let mut have: Vec<(String, Slot)> =
                fields.iter().map(|&(n, s)| (n.to_string(), s)).collect();
            let key = |x: &(String, Slot)| format!("{:?}", x.1);
            want.sort_by_key(key);
            have.sort_by_key(key);
            if want != have {
                wrong.push(format!("{op:?}:\n   slots: {want:?}\n   typed: {have:?}"));
            }
        }
        assert!(
            wrong.is_empty(),
            "typed_abi disagrees with packet::slots:\n  {}",
            wrong.join("\n  ")
        );
    }

    #[test]
    fn flash_prefill_is_the_split_k_layout() {
        let inst = FlashPrefillArgs {
            Opart: 1,
            mlpart: 2,
            Q: 3,
            K: 4,
            V: 5,
            O_final: 6,
            n_q: 128,
            n_kv: 128,
            n_head: 32,
            n_kv_head: 2,
            hd: 128,
            nsplit: 1,
            scale: 0.125,
            kv_stride: 4096,
            kv_mask: u32::MAX,
            ..Default::default()
        }
        .build(48);
        assert_eq!(inst.op, DevOp::FlashPrefill as u16);
        assert_eq!(inst.t[..6], [1, 2, 3, 4, 5, 6]);
        assert_eq!(inst.i, [128, 128, 32, 2, 0, 0, 128, 1]);
        assert_eq!((inst.f[0], inst.j), (0.125, [4096, u32::MAX]));
    }

    #[test]
    fn defaults_are_the_builders() {
        let inst = Mamba2StepArgs::default().build(1);
        assert_eq!(inst.t, [TENSOR_NONE; 8]);
        assert_eq!((inst.i, inst.f, inst.j), ([0; 8], [0.0; 2], [0; 2]));
    }
}
