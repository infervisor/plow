//! Operand slot names for [`DevOp`] — what turns a `DevInst64` into a readable
//! disassembly.
//!
//! A raw instruction is eight tensor handles, eight `u32`s and three overlaid
//! words. Naming them is the whole difference between
//!
//! ```text
//! op=9 t=[41,12,7,65535,88] i=[1,6144,2048,0]
//! ```
//!
//! and
//!
//! ```text
//! GemmNorm  C<-act.hn  A<-act.x  B<-blk3.attn.q.w  gamma<-blk3.attn.q_norm.g  [M=1 N=6144 K=2048]
//! ```
//!
//! # Why a table, and not generated from the doc comments
//!
//! The names already exist: [`DevOp`]'s doc comments carry them in a consistent
//! grammar (`` `t0=C t1=A t2=B` · `i0=M i1=N i2=K` ``). Generating this table
//! from them at build time would remove the duplication, but it would make a
//! doc comment load-bearing — a reflow becomes a build break, and rustdoc
//! formatting is not a contract anyone agreed to.
//!
//! So the table is written out, and [a test](self::tests::table_matches_doc_comments)
//! parses the doc comments out of `dev.rs` and asserts they agree. Drift is
//! caught in CI rather than silently producing a confident, wrong disassembly —
//! which is worse than no disassembly at all.
//!
//! That test earned its keep immediately: it found [`DevOp::FlashPrefill`]'s
//! spec stale by the entire split-K epilogue (see the note on that variant).
//!
//! # The `fj` overlay
//!
//! [`DevInst64::fj`] is not three floats. `fj[0]` is `f[0]`; `fj[1]` is `f[1]`
//! **or** the integer `j[0]`, mutually exclusive and asserted in
//! [`DevInst::pack`]; `fj[2]` is `j[1]`. A disassembler that assumes float
//! prints `1.0e-45` where the value is the integer `7`, so which one `fj[1]`
//! holds is recorded per op ([`OpSlots::fj1_is_float`]) rather than guessed.
//!
//! # Authority
//!
//! These names describe the *interpreter's* reading of an instruction, and
//! `runtime/amd/interp.hip` is the ultimate authority. The table is a
//! convenience over it. Nothing here can corrupt data: a disassembler must emit
//! the raw slots alongside the named view, so a wrong name is visible rather
//! than load-bearing.

use crate::dev::DevOp;

/// One op's slot spec, positionally indexed. `""` marks an unused slot in the
/// middle of a run (several ops skip `t3`/`t4`), a `?` suffix marks an operand
/// that may be [`crate::dev::TENSOR_NONE16`].
#[derive(Clone, Copy, Debug)]
struct S {
    op: DevOp,
    t: &'static [&'static str],
    i: &'static [&'static str],
    /// `[f0, f1]`. A present `f1` is what makes `fj[1]` a float.
    f: &'static [&'static str],
    /// `[j0, j1]` — the integer reading of `fj[1]`/`fj[2]`.
    j: &'static [&'static str],
}

/// A named tensor operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub name: &'static str,
    /// The op tolerates [`crate::dev::TENSOR_NONE16`] here.
    pub optional: bool,
}

/// Where an op's names came from — reported so a reader can weigh them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Provenance {
    /// A spec in the op's own doc comment.
    Documented,
    /// Inherited from another op ("As [`DevOp::Gemm`], 64x128 tile"), with any
    /// overrides applied.
    Inherited(DevOp),
    /// Defined for ABI stability, body not built. No operands to name.
    Reserved,
    /// No spec anywhere. Renders raw.
    Undocumented,
}

/// Resolved slot names for one op.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpSlots {
    pub t: [Option<Slot>; 8],
    pub i: [Option<&'static str>; 8],
    pub f0: Option<&'static str>,
    pub f1: Option<&'static str>,
    pub j0: Option<&'static str>,
    pub j1: Option<&'static str>,
    pub provenance: Provenance,
}

impl OpSlots {
    /// Whether `fj[1]` holds an `f32` (`f1`) rather than the integer `j0`.
    /// See the module note on the overlay.
    pub fn fj1_is_float(&self) -> bool {
        self.f1.is_some()
    }
}

fn slot(raw: &'static str) -> Option<Slot> {
    match raw {
        "" => None,
        s => Some(Slot { name: s.trim_end_matches('?'), optional: s.ends_with('?') }),
    }
}

/// Ops whose doc comment carries an explicit spec. Generated once from those
/// comments and checked against them by `table_matches_doc_comments`.
#[rustfmt::skip]
const DOC: &[S] = &[
    S { op: DevOp::RmsNorm, t: &["out", "x", "gamma?"], i: &["rows", "feat"], f: &["eps"], j: &[] },
    S { op: DevOp::RowRms, t: &["rms", "x"], i: &["rows", "feat"], f: &["eps"], j: &[] },
    S { op: DevOp::HeadNormRope, t: &["out", "x", "gamma?", "cos?", "sin?", "pos"], i: &["ntok", "nhead", "hd", "out_row0"], f: &["eps"], j: &[] },
    S { op: DevOp::Residual, t: &["out", "a", "b"], i: &["n"], f: &["scale"], j: &[] },
    S { op: DevOp::Glu, t: &["out", "gate", "up"], i: &["n", "act"], f: &[], j: &[] },
    S { op: DevOp::Embed, t: &["out", "table", "ids"], i: &["ntok", "hidden"], f: &["scale"], j: &[] },
    S { op: DevOp::SoftCap, t: &["out", "x"], i: &["n"], f: &["cap"], j: &[] },
    S { op: DevOp::Gemm, t: &["C", "A", "B"], i: &["M", "N", "K"], f: &[], j: &[] },
    S { op: DevOp::Gemv, t: &["C", "x", "W", "rms?", "gamma?"], i: &["M", "N", "K", "norm"], f: &[], j: &[] },
    S { op: DevOp::FlashPrefill, t: &["Opart", "mlpart", "Q", "K", "V", "O_final"], i: &["n_q", "n_kv", "n_head", "n_kv_head", "q_pos0", "window", "hd", "nsplit"], f: &["scale"], j: &["kv_stride", "kv_mask"] },
    S { op: DevOp::FlashDecode, t: &["Opart", "mlpart", "Q", "K", "V", "kv_len"], i: &["n_batch", "n_head", "n_kv_head", "kv_stride", "window", "nsplit", "hd"], f: &["scale"], j: &[] },
    S { op: DevOp::FlashMerge, t: &["O", "Opart", "mlpart"], i: &["n_batch", "n_head", "nsplit", "hd"], f: &[], j: &[] },
    S { op: DevOp::NormResidual, t: &["out", "a", "b", "gamma?"], i: &["rows", "feat"], f: &["eps", "scale"], j: &[] },
    S { op: DevOp::AddNorm, t: &["out", "resid", "a", "b", "gamma?"], i: &["rows", "feat"], f: &["eps"], j: &[] },
    S { op: DevOp::Argmax, t: &["part", "x"], i: &["n"], f: &[], j: &[] },
    S { op: DevOp::ArgmaxFin, t: &["ids", "part"], i: &["blocks"], f: &[], j: &[] },
    S { op: DevOp::GemvGlu, t: &["fu", "x", "W_gate", "", "", "W_up"], i: &["M", "N", "K", "", "", "act"], f: &[], j: &[] },
    S { op: DevOp::GemmGlu, t: &["fu", "x", "W_gate", "", "", "W_up"], i: &["M", "N", "K", "", "", "act"], f: &[], j: &[] },
    S { op: DevOp::GemvQkv, t: &["q_out", "x", "W_q", "k_out", "W_k", "v_out", "W_v"], i: &["M", "Nq", "K", "Nk", "Nv"], f: &[], j: &[] },
    S { op: DevOp::GemvFp8, t: &["C", "x", "W", "", "", "w_scale"], i: &["M", "N", "K", "", "a_row0"], f: &[], j: &[] },
    S { op: DevOp::GemvGluFp8, t: &["fu", "x", "W_gate", "gate_scale", "up_scale", "W_up"], i: &["M", "N", "K", "", "", "act"], f: &[], j: &[] },
    S { op: DevOp::QuantFp8, t: &["xq", "x", "a_scale"], i: &["M", "K"], f: &[], j: &[] },
    S { op: DevOp::GemmFp8, t: &["C", "A", "B", "a_scale", "w_scale"], i: &["M", "N", "K", "", "a_row0"], f: &[], j: &[] },
    S { op: DevOp::GemmGluFp8, t: &["fu", "A", "Wg", "a_scale", "g_scale", "Wu", "u_scale"], i: &["M", "N", "K", "", "", "act"], f: &[], j: &[] },
    S { op: DevOp::NormResidualNorm, t: &["out", "resid", "a", "b", "gamma_b?", "gamma_n?"], i: &["rows", "feat"], f: &["eps", "scale"], j: &[] },
    S { op: DevOp::XReduce, t: &["out"], i: &["H", "n_gpu", "slot"], f: &[], j: &[] },
    S { op: DevOp::XArgmaxFin, t: &["ids", "local_part"], i: &["n_gpu", "", "slot"], f: &[], j: &[] },
    S { op: DevOp::XReduceTwoShot, t: &["out"], i: &["n", "n_gpu", "slot", "gate_rs", "gate_ag"], f: &[], j: &[] },
    S { op: DevOp::HeadNormRopeFp8, t: &["out", "", "", "", "", "", "scale"], i: &[], f: &[], j: &[] },
    S { op: DevOp::MoeRouter, t: &[], i: &["H", "n_exp", "k", "flags"], f: &["route_scale"], j: &[] },
    S { op: DevOp::MoeExpertGlu, t: &["fu", "x", "routing_table", "expert_weight_table"], i: &["slot", "I_moe", "H", "n_exp", "", "act"], f: &[], j: &[] },
    S { op: DevOp::MoeExpertDown, t: &["part", "fu", "routing_table", "expert_weight_table"], i: &["slot", "H", "I_moe", "n_exp"], f: &[], j: &[] },
    S { op: DevOp::MoeCombine, t: &["out", "residual?", "shared?", "part_base"], i: &["H", "k"], f: &[], j: &[] },
    S { op: DevOp::GemvFp8Blk, t: &["C", "x", "W", "", "", "w_scale"], i: &["M", "N", "K", "", "a_row0"], f: &[], j: &[] },
    S { op: DevOp::MoeExpertGluFp8Blk, t: &["fu", "x", "routing_table", "expert_weight_table", "expert_scale_table"], i: &["slot", "I_moe", "H", "n_exp", "", "act", "enc"], f: &["situ_beta", "situ_linear_beta"], j: &[] },
    S { op: DevOp::MoeExpertDownFp8Blk, t: &["part", "fu", "routing_table", "expert_weight_table", "expert_scale_table"], i: &["slot", "H", "I_moe", "n_exp"], f: &[], j: &[] },
    S { op: DevOp::DenseGluFp8Blk, t: &["fu", "x", "Wg", "Sg", "Su", "Wu"], i: &["N", "K", "", "", "", "act"], f: &[], j: &[] },
    S { op: DevOp::MoeGroupGluFp8Blk, t: &["fu", "x", "routing_table", "expert_weight_table", "expert_scale_table"], i: &["k", "I_moe", "H", "n_exp", "", "act", "enc"], f: &["situ_beta", "situ_linear_beta"], j: &[] },
    S { op: DevOp::MoeGroupDownFp8Blk, t: &["part", "fu", "routing_table", "expert_weight_table", "expert_scale_table"], i: &["k", "H", "I_moe", "n_exp"], f: &[], j: &[] },
    S { op: DevOp::FlashMlaDecode, t: &["Opart", "mlpart", "Qabs", "Qrope", "Ckv", "Krope", "kv_len"], i: &["n_batch", "n_head", "kv_stride", "window", "nsplit", "kv_mask"], f: &["scale"], j: &[] },
    S { op: DevOp::OUvFold, t: &["O", "Olat", "Wuv"], i: &["n_batch", "n_head", "V"], f: &[], j: &[] },
    S { op: DevOp::FlashGatherDecode, t: &["Opart", "mlpart", "Qabs", "Qrope", "Ckv", "Krope", "kv_len", "idx"], i: &["n_batch", "n_head", "kv_stride", "", "nsplit", "kv_mask", "top_k"], f: &["scale"], j: &[] },
    S { op: DevOp::MoeRouterTopk, t: &["table", "logit", "", "bias"], i: &["", "n_exp", "k", "flags"], f: &["route_scale"], j: &[] },
    S { op: DevOp::MlaMergeFold, t: &["O", "Opart", "mlpart", "Wuv"], i: &["n_batch", "n_head", "V", "", "nsplit"], f: &[], j: &[] },
    S { op: DevOp::IndexScore, t: &["Score", "Qidx", "Kidx", "W", "kv_len"], i: &["n_batch", "index_heads", "kv_stride", "index_head_dim"], f: &["scale"], j: &[] },
    S { op: DevOp::IndexSelect, t: &["idx", "Score", "gHist", "gCtl"], i: &["len", "top_k"], f: &[], j: &[] },
    S { op: DevOp::LayerNorm, t: &["out", "x", "gamma", "beta"], i: &["rows", "feat", "", "out_row0"], f: &["eps"], j: &[] },
    S { op: DevOp::MoeRouterGemma, t: &["table", "resid", "proj", "scale", "per_expert_scale"], i: &["H", "n_exp", "k"], f: &["root", "eps"], j: &[] },
    S { op: DevOp::MoeExpertGluGemma, t: &["fu", "x", "table", "ewt"], i: &[], f: &[], j: &[] },
    S { op: DevOp::MoeExpertDownGemma, t: &["part", "fu", "table", "ewt"], i: &["k", "H", "I_moe", "n_exp"], f: &[], j: &[] },
    S { op: DevOp::MoeCombineGemma, t: &["moe", "part"], i: &["H", "k"], f: &[], j: &[] },
    S { op: DevOp::MoeExpertGluGemmaFp8, t: &["fu", "x", "table", "ewt", "est"], i: &["k", "I", "H", "n_exp"], f: &[], j: &[] },
    S { op: DevOp::MoeExpertDownGemmaFp8, t: &["part", "fu", "table", "ewt", "est"], i: &["k", "H", "I", "n_exp"], f: &[], j: &[] },
    S { op: DevOp::MoeRouterGemmaScore, t: &["score", "resid", "proj", "scale"], i: &["H", "n_exp"], f: &["root", "eps"], j: &[] },
    S { op: DevOp::MoeRouterGemmaTopk, t: &["table", "score", "per_expert_scale"], i: &["", "n_exp", "k"], f: &[], j: &[] },
    S { op: DevOp::MoeCombineNormGemma, t: &["out", "part", "resid", "gamma"], i: &["H", "k"], f: &["eps"], j: &[] },
    S { op: DevOp::MoeExpertGluNormGemma, t: &["fu", "resid", "table", "ewt", "gamma"], i: &["k", "I", "H", "n_exp"], f: &["eps"], j: &[] },
    S { op: DevOp::MoeCombineResidNormGemma, t: &["hn", "x", "part", "h1", "g_pf2", "g_po", "gn"], i: &["H", "k"], f: &["eps", "layer_scalar"], j: &[] },
    S { op: DevOp::MoeRouterGemmaPf, t: &["table", "resid", "proj", "scale", "per_expert_scale"], i: &["H", "n_exp", "k", "T"], f: &["root", "eps"], j: &[] },
    S { op: DevOp::MoeAlignGemmaPf, t: &["meta", "table", "row_token", "row_partidx", "row_gate"], i: &["T", "n_exp", "k"], f: &[], j: &[] },
    S { op: DevOp::MoeGroupGluGemmaPf, t: &["fu_g", "xn2", "ewt", "meta", "row_token"], i: &["I_moe", "H", "n_exp", "", "", "act"], f: &[], j: &[] },
    S { op: DevOp::MoeGroupDownGemmaPf, t: &["part", "fu_g", "ewt", "meta", "row_partidx", "row_gate"], i: &["H", "I_moe", "n_exp"], f: &[], j: &[] },
    S { op: DevOp::MoeCombineNormGemmaPf, t: &["out", "part", "h1", "gamma"], i: &["H", "k", "T"], f: &["eps"], j: &[] },
    S { op: DevOp::GemvSz, t: &["C", "x", "blob"], i: &["M", "N", "K"], f: &[], j: &[] },
    S { op: DevOp::GemvGluSz, t: &["fu", "x", "gblob", "ublob"], i: &["M", "N", "K", "", "", "act"], f: &[], j: &[] },
    S { op: DevOp::GemvArgmax, t: &["C", "x", "W", "part"], i: &["1", "N", "K", "", "a_row0"], f: &["cap"], j: &[] },
    S { op: DevOp::MoeGroupGluGemmaPfW8a8, t: &["fu", "xq8", "ewt", "meta", "row_token", "ascale", "est"], i: &["I_moe", "H", "n_exp", "", "", "act"], f: &[], j: &[] },
    S { op: DevOp::MoeGroupDownGemmaPfW8a8, t: &["part", "fu8", "ewt", "meta", "row_partidx", "row_gate", "est", "fscale"], i: &["H", "I_moe", "n_exp"], f: &[], j: &[] },
    S { op: DevOp::MoeRouterTopkPf, t: &["table", "logit", "", "bias"], i: &["", "n_exp", "k", "flags", "T"], f: &["route_scale"], j: &[] },
    S { op: DevOp::MoeAlignPf, t: &["meta", "table", "row_token", "row_partidx", "row_gate"], i: &["T", "n_exp", "k"], f: &[], j: &[] },
    S { op: DevOp::MoeGroupGluPf, t: &["fu_g", "xn2", "expert_weight_table", "expert_scale_table", "meta", "row_token"], i: &["I_moe", "H", "n_exp", "fp8", "", "act"], f: &[], j: &[] },
    S { op: DevOp::MoeGroupDownPf, t: &["part", "fu_g", "expert_weight_table", "expert_scale_table", "meta", "", "row_partidx", "row_gate"], i: &["H", "I_moe", "n_exp", "fp8"], f: &[], j: &[] },
    S { op: DevOp::MoeCombinePf, t: &["out", "residual", "shared?", "part"], i: &["H", "k", "T"], f: &[], j: &[] },
    S { op: DevOp::KdaConv, t: &["out", "x", "w", "conv_state"], i: &["T", "conv_dim", "W", "act"], f: &[], j: &[] },
    S { op: DevOp::KdaGate, t: &["g", "beta", "g_raw", "beta_raw", "A_log", "dt_bias"], i: &["T", "H", "D", "gate_mode"], f: &["lower_bound"], j: &[] },
    S { op: DevOp::GemvMxfp4, t: &["C", "x", "W", "S"], i: &["M", "N", "K"], f: &[], j: &[] },
    S { op: DevOp::GemvGluMxfp4, t: &["C", "x", "Wg", "Sg", "Su", "Wu"], i: &["M", "N", "K", "", "", "act"], f: &[], j: &[] },
    S { op: DevOp::GemmMxfp4, t: &["C", "A", "W", "wscale"], i: &["M", "N", "K"], f: &[], j: &[] },
    S { op: DevOp::KdaStateStep, t: &[], i: &["T", "H", "D", "BV", "flags"], f: &["scale"], j: &[] },
    S { op: DevOp::KdaGatedNorm, t: &["y", "o", "norm_w", "g_raw"], i: &["T", "H", "D"], f: &["eps"], j: &[] },
    S { op: DevOp::AttnRes, t: &["out", "prefix_sum", "block_residual", "score_w"], i: &["T", "H", "nb"], f: &["eps"], j: &[] },
    S { op: DevOp::SituGlu, t: &["out", "gate", "up"], i: &[], f: &[], j: &[] },
    S { op: DevOp::MlaOutGate, t: &["out", "a", "b"], i: &[], f: &[], j: &[] },
    S { op: DevOp::GemmFp8Blk, t: &["C", "A", "W", "weight_scale_inv"], i: &["M", "N", "K"], f: &[], j: &[] },
];

/// Ops that say "As [`DevOp::X`]" / "twin of [`DevOp::X`]" / "Same operands as
/// [`DevOp::X`]" rather than restating a spec. `(op, base, overrides)`; an
/// override slot is applied only where it is non-empty, so a tile twin that
/// changes nothing carries an empty override.
#[rustfmt::skip]
const INHERIT: &[(DevOp, DevOp, S)] = &[
    // "with RMSNorm folded into the A-operand load. `t3=rms(f32) t4=gamma`"
    (DevOp::GemmNorm, DevOp::Gemm,
     S { op: DevOp::GemmNorm, t: &["", "", "", "rms", "gamma"], i: &[], f: &[], j: &[] }),
    // Pure tile-size twins: same operands, different MFMA tile.
    (DevOp::GemmSmall,  DevOp::Gemm, NONE),
    (DevOp::GemmMed,    DevOp::Gemm, NONE),
    (DevOp::GemmWide,   DevOp::Gemm, NONE),
    (DevOp::GemmC5,     DevOp::Gemm, NONE),
    (DevOp::GemmSmallFp8, DevOp::GemmFp8, NONE),
    (DevOp::GemmMedFp8,   DevOp::GemmFp8, NONE),
    (DevOp::GemmWideFp8,  DevOp::GemmFp8, NONE),
    (DevOp::GemmC5Fp8,    DevOp::GemmFp8, NONE),
    (DevOp::GemmSmallMxfp4, DevOp::GemmMxfp4, NONE),
    (DevOp::GemmMedMxfp4,   DevOp::GemmMxfp4, NONE),
    (DevOp::GemmWideMxfp4,  DevOp::GemmMxfp4, NONE),
    (DevOp::GemmC5Mxfp4,    DevOp::GemmMxfp4, NONE),
    // "`t3=K(fp8) t4=V(fp8) t6=k_scale t7=v_scale`; else as [`DevOp::Flash*`]"
    (DevOp::FlashDecodeFp8, DevOp::FlashDecode,
     S { op: DevOp::FlashDecodeFp8, t: &["", "", "", "K", "V", "", "k_scale", "v_scale"],
         i: &[], f: &[], j: &[] }),
    (DevOp::FlashPrefillFp8, DevOp::FlashPrefill,
     S { op: DevOp::FlashPrefillFp8, t: &["", "", "", "K", "V", "", "k_scale", "v_scale"],
         i: &[], f: &[], j: &[] }),
    // "Operands are identical to opcode 67."
    (DevOp::MoeRouterGemmaScoreFast, DevOp::MoeRouterGemmaScore, NONE),
    // "ABI mirrors [`DevOp::FlashMerge`] with `t1..` in peer_scratch + xctr gates."
    (DevOp::XFlashMerge, DevOp::FlashMerge, NONE),
];

const NONE: S = S { op: DevOp::Nop, t: &[], i: &[], f: &[], j: &[] };

/// Defined so the ABI is stable, body not built — there is nothing to name.
/// Keeping these explicit is what lets [`Provenance::Undocumented`] mean "an
/// oversight" rather than "probably fine".
const RESERVED: &[DevOp] = &[
    DevOp::Nop,
    DevOp::XReduceScatter,
    DevOp::XAllGather,
    DevOp::FlashMlaPrefill,
    DevOp::AttnSelect,
    DevOp::FlashGatherPrefill,
];

fn find(op: DevOp) -> Option<&'static S> {
    DOC.iter().find(|s| s.op == op)
}

fn expand(s: &S, provenance: Provenance) -> OpSlots {
    let mut out = OpSlots {
        t: [None; 8],
        i: [None; 8],
        f0: None,
        f1: None,
        j0: None,
        j1: None,
        provenance,
    };
    for (k, raw) in s.t.iter().take(8).enumerate() {
        out.t[k] = slot(raw);
    }
    for (k, raw) in s.i.iter().take(8).enumerate() {
        out.i[k] = if raw.is_empty() { None } else { Some(*raw) };
    }
    let g = |v: &'static [&'static str], k: usize| -> Option<&'static str> {
        v.get(k).copied().filter(|x| !x.is_empty())
    };
    out.f0 = g(s.f, 0);
    out.f1 = g(s.f, 1);
    out.j0 = g(s.j, 0);
    out.j1 = g(s.j, 1);
    out
}

/// Resolve an op's operand names, following one level of inheritance.
///
/// Returns [`Provenance::Undocumented`] with every slot `None` for an op that
/// has no spec — the caller renders those raw rather than guessing.
pub fn slots_for(op: DevOp) -> OpSlots {
    if let Some(s) = find(op) {
        return expand(s, Provenance::Documented);
    }
    if let Some((_, base, over)) = INHERIT.iter().find(|(o, _, _)| *o == op) {
        // One level only: every base in INHERIT is itself in DOC, asserted by
        // `inheritance_bases_are_documented`.
        let mut out = find(*base).map(|s| expand(s, Provenance::Inherited(*base))).unwrap_or(
            OpSlots {
                t: [None; 8], i: [None; 8], f0: None, f1: None, j0: None, j1: None,
                provenance: Provenance::Inherited(*base),
            },
        );
        let ov = expand(over, Provenance::Inherited(*base));
        for k in 0..8 {
            if ov.t[k].is_some() {
                out.t[k] = ov.t[k];
            }
            if ov.i[k].is_some() {
                out.i[k] = ov.i[k];
            }
        }
        out.f0 = ov.f0.or(out.f0);
        out.f1 = ov.f1.or(out.f1);
        out.j0 = ov.j0.or(out.j0);
        out.j1 = ov.j1.or(out.j1);
        return out;
    }
    let provenance = if RESERVED.contains(&op) { Provenance::Reserved } else { Provenance::Undocumented };
    OpSlots { t: [None; 8], i: [None; 8], f0: None, f1: None, j0: None, j1: None, provenance }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `dev.rs` read at compile time. Parsing the doc comments here — rather
    /// than generating the table from them — is the trade explained in the
    /// module header: drift becomes a test failure instead of a build break.
    const DEV_RS: &str = include_str!("dev.rs");

    #[derive(Debug, Default, PartialEq, Eq)]
    struct Spec {
        t: [Option<String>; 8],
        i: [Option<String>; 8],
        f: [Option<String>; 2],
        j: [Option<String>; 2],
    }

    /// Scan one backtick group as a run of `<letter><digit>=<value>` tokens.
    /// `None` if anything else is in it, which is how prose groups
    /// (`` `window = 0` is full causal ``) are rejected.
    fn scan(group: &str) -> Option<Vec<(char, usize, String)>> {
        let b: Vec<char> = group.chars().collect();
        let mut p = 0usize;
        let mut out = Vec::new();
        loop {
            while p < b.len() && b[p].is_whitespace() {
                p += 1;
            }
            if p >= b.len() {
                return if out.is_empty() { None } else { Some(out) };
            }
            let letter = b[p];
            if !matches!(letter, 't' | 'i' | 'f' | 'j') || p + 2 >= b.len() {
                return None;
            }
            let idx = b[p + 1].to_digit(10)? as usize;
            if b[p + 2] != '=' {
                return None;
            }
            p += 3;
            // Value: to the next whitespace, but bracketed runs may contain any.
            let start = p;
            while p < b.len() && !b[p].is_whitespace() {
                match b[p] {
                    '(' => {
                        while p < b.len() && b[p] != ')' {
                            p += 1;
                        }
                    }
                    '[' => {
                        while p < b.len() && b[p] != ']' {
                            p += 1;
                        }
                    }
                    _ => {}
                }
                p += 1;
            }
            if p == start {
                return None;
            }
            out.push((letter, idx, b[start..p.min(b.len())].iter().collect()));
        }
    }

    /// `Opart(f32)` -> `Opart`, `part(u64[blocks])` -> `part`, `gamma?` ->
    /// `gamma?`.
    ///
    /// `|NONE` is the second spelling of optional — `t2=shared[T,H]|NONE` on
    /// [`DevOp::MoeCombineResidNormGemma`] means the same as a `?` suffix, and
    /// normalizing it to one form is what keeps the table's `optional` flag
    /// honest for that op.
    fn normalize(raw: &str) -> String {
        let optional = raw.ends_with('?') || raw.contains("|NONE");
        let head: String =
            raw.chars().take_while(|c| *c != '(' && *c != '[' && *c != '|').collect();
        let head = head.trim_end_matches('?');
        if optional {
            format!("{head}?")
        } else {
            head.to_string()
        }
    }

    /// The FIRST maximal run of slot-only groups separated by nothing but
    /// whitespace and `·`, anchored on the group declaring `t0=`.
    ///
    /// Anchoring matters in both directions: [`DevOp::XReduce`] puts prose
    /// before its spec, and [`DevOp::FlashPrefill`] carries a second, historical
    /// `t0=...` inside a later note. A naive "first backticked group" or "every
    /// backticked group" rule gets one of those two wrong.
    fn spec_of(doc: &str) -> Option<Spec> {
        let mut groups: Vec<(usize, usize, &str)> = Vec::new();
        let bytes = doc.as_bytes();
        let mut k = 0usize;
        while k < bytes.len() {
            if bytes[k] == b'`' {
                if let Some(rel) = doc[k + 1..].find('`') {
                    groups.push((k, k + 1 + rel, &doc[k + 1..k + 1 + rel]));
                    k = k + rel + 2;
                    continue;
                }
                break;
            }
            k += 1;
        }
        let ok: Vec<bool> = groups.iter().map(|(_, _, g)| scan(g).is_some()).collect();
        let anchor = (0..groups.len())
            .find(|&n| ok[n] && groups[n].2.contains("t0="))
            .or_else(|| (0..groups.len()).find(|&n| ok[n]))?;

        let joined = |a: usize, b: usize| -> bool {
            doc[groups[a].1 + 1..groups[b].0].chars().all(|c| c.is_whitespace() || c == '·')
        };
        let (mut lo, mut hi) = (anchor, anchor);
        while hi + 1 < groups.len() && ok[hi + 1] && joined(hi, hi + 1) {
            hi += 1;
        }
        while lo > 0 && ok[lo - 1] && joined(lo - 1, lo) {
            lo -= 1;
        }

        let mut spec = Spec::default();
        for (_, _, g) in &groups[lo..=hi] {
            for (letter, idx, raw) in scan(g).unwrap() {
                let name = normalize(&raw);
                match letter {
                    't' if idx < 8 => spec.t[idx] = Some(name),
                    'i' if idx < 8 => spec.i[idx] = Some(name),
                    'f' if idx < 2 => spec.f[idx] = Some(name),
                    'j' if idx < 2 => spec.j[idx] = Some(name),
                    _ => {}
                }
            }
        }
        Some(spec)
    }

    /// Every `DevOp` variant with its doc comment, in declaration order.
    fn variants() -> Vec<(String, Option<Spec>)> {
        let mut out = Vec::new();
        let mut doc = String::new();
        let mut in_enum = false;
        for line in DEV_RS.lines() {
            let s = line.trim();
            if !in_enum {
                in_enum = s.starts_with("pub enum DevOp");
                continue;
            }
            if s == "}" {
                break;
            }
            if let Some(rest) = s.strip_prefix("///") {
                doc.push_str(rest.trim());
                doc.push(' ');
                continue;
            }
            if s.starts_with("//") || s.starts_with("#[") || s.is_empty() {
                continue;
            }
            // `Name = 12,`
            let name: String = s.chars().take_while(|c| c.is_alphanumeric()).collect();
            let is_variant = !name.is_empty()
                && name.starts_with(char::is_uppercase)
                && s[name.len()..].trim_start().starts_with('=');
            if is_variant {
                out.push((name, spec_of(&doc)));
            }
            doc.clear();
        }
        out
    }

    fn table_spec(s: &OpSlots) -> Spec {
        let mut out = Spec::default();
        for k in 0..8 {
            out.t[k] = s.t[k].map(|x| {
                if x.optional { format!("{}?", x.name) } else { x.name.to_string() }
            });
            out.i[k] = s.i[k].map(str::to_string);
        }
        out.f[0] = s.f0.map(str::to_string);
        out.f[1] = s.f1.map(str::to_string);
        out.j[0] = s.j0.map(str::to_string);
        out.j[1] = s.j1.map(str::to_string);
        out
    }

    /// THE drift gate. Every op whose doc comment carries a spec must have a
    /// table row saying exactly the same thing.
    ///
    /// An inheriting op is compared against its OVERRIDE, not its resolution:
    /// three of them (`GemmNorm`, `FlashDecodeFp8`, `FlashPrefillFp8`) state
    /// only the slots they change — "`t3=rms(f32) t4=gamma`" on top of
    /// [`DevOp::Gemm`] — so the doc spec is partial by design and comparing it
    /// to the merged result would fail on every operand it inherited.
    #[test]
    fn table_matches_doc_comments() {
        let mut missing = Vec::new();
        let mut wrong = Vec::new();
        for (name, spec) in variants() {
            let Some(spec) = spec else { continue };
            let row = DOC
                .iter()
                .find(|s| format!("{:?}", s.op) == name)
                .or_else(|| INHERIT.iter().find(|(o, _, _)| format!("{o:?}") == name).map(|(_, _, o)| o));
            let Some(row) = row else {
                missing.push(name);
                continue;
            };
            let have = table_spec(&expand(row, Provenance::Documented));
            if have != spec {
                wrong.push(format!("{name}:\n     doc: {spec:?}\n   table: {have:?}"));
            }
        }
        assert!(
            missing.is_empty() && wrong.is_empty(),
            "slot table has drifted from the DevOp doc comments.\n\
             missing rows: {missing:?}\n\
             disagreeing:\n  {}",
            wrong.join("\n  ")
        );
    }

    /// A row whose op lost its doc spec would silently become unverifiable —
    /// `table_matches_doc_comments` only walks ops that still have one — so the
    /// reverse direction is checked too.
    #[test]
    fn every_table_row_has_a_documented_spec() {
        let documented: Vec<String> =
            variants().into_iter().filter(|(_, s)| s.is_some()).map(|(n, _)| n).collect();
        let orphans: Vec<String> = DOC
            .iter()
            .map(|s| format!("{:?}", s.op))
            .filter(|n| !documented.contains(n))
            .collect();
        assert!(orphans.is_empty(), "table rows with no doc spec: {orphans:?}");
    }

    /// An inheriting op must not also have a `DOC` row: `slots_for` checks `DOC`
    /// first, so the row would shadow the inheritance and the op would resolve
    /// to its overrides alone, silently losing every inherited operand.
    #[test]
    fn no_op_is_in_both_tables() {
        for (op, _, _) in INHERIT {
            assert!(find(*op).is_none(), "{op:?} is in both DOC and INHERIT");
        }
    }

    /// `slots_for` follows exactly one level, so a base must resolve directly.
    #[test]
    fn inheritance_bases_are_documented() {
        for (op, base, _) in INHERIT {
            assert!(
                find(*base).is_some(),
                "{op:?} inherits {base:?}, which has no DOC row — \
                 slots_for does not chain, so this would silently resolve to nothing"
            );
        }
    }

    #[test]
    fn inherited_ops_resolve_to_their_base() {
        let g = slots_for(DevOp::GemmSmall);
        assert_eq!(g.provenance, Provenance::Inherited(DevOp::Gemm));
        assert_eq!(g.t[0].unwrap().name, "C");
        assert_eq!(g.i[2], Some("K"));

        // Override applied on top of the base.
        let n = slots_for(DevOp::GemmNorm);
        assert_eq!(n.t[0].unwrap().name, "C", "base operand survives");
        assert_eq!(n.t[3].unwrap().name, "rms", "override lands");
        assert_eq!(n.t[4].unwrap().name, "gamma");
    }

    /// The overlay the module header warns about, pinned on one op of each kind.
    #[test]
    fn fj1_overlay_is_recorded_per_op() {
        // NormResidual documents `f0=eps f1=scale` -> fj[1] is an f32.
        let nr = slots_for(DevOp::NormResidual);
        assert!(nr.fj1_is_float(), "NormResidual sets f1");
        assert_eq!(nr.f1, Some("scale"));

        // FlashPrefill uses the integer half: fj[1] is j0 = kv_stride.
        let fp = slots_for(DevOp::FlashPrefill);
        assert!(!fp.fj1_is_float(), "FlashPrefill uses j0, not f1");
        assert_eq!(fp.j0, Some("kv_stride"));
        assert_eq!(fp.j1, Some("kv_mask"));
    }

    /// Guards the bug this table was born from: `FlashPrefill`'s spec was stale
    /// by the split-K epilogue. Ground truth is `exec_flash_prefill`
    /// (`runtime/amd/interp.hip`), which passes
    /// `TEN(0), TEN(1), TEN(5), TEN(2), TEN(3), TEN(4)` into
    /// `d_flash_prefill(Opart, mlpart, O_final, Q, K, V)`.
    #[test]
    fn flash_prefill_matches_the_interpreter() {
        let s = slots_for(DevOp::FlashPrefill);
        let names: Vec<&str> = (0..6).map(|k| s.t[k].unwrap().name).collect();
        assert_eq!(names, ["Opart", "mlpart", "Q", "K", "V", "O_final"]);
        assert_eq!(s.i[7], Some("nsplit"));
    }

    /// Not an assertion about correctness — a visible inventory, so the ops with
    /// no spec stay a known list rather than a discovery.
    #[test]
    fn unspecified_ops_are_the_expected_set() {
        let undoc: Vec<String> = variants()
            .into_iter()
            .filter(|(name, spec)| {
                spec.is_none()
                    && !INHERIT.iter().any(|(o, _, _)| format!("{o:?}") == *name)
                    && !RESERVED.iter().any(|o| format!("{o:?}") == *name)
            })
            .map(|(n, _)| n)
            .collect();
        assert_eq!(
            undoc,
            ["Mamba2Scan"],
            "the set of ops with no operand spec changed; name them in RESERVED \
             (body not built) or give them a spec"
        );
    }
}
