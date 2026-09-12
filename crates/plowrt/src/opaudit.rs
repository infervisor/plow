//! Operator row-identity audit — Phase 1d of the unified-token-batch plan.
//!
//! Packing rows from several requests into one activation matrix is safe for an
//! operator exactly when the operator cannot confuse one request's rows for
//! another's. This module answers that question for every opcode in the ISA, and
//! for every instruction in an emitted program.
//!
//! # Why this is a table and not a heuristic
//!
//! The plan's central hazard is an operator that derives each row's absolute
//! position from a packet scalar. Under packing it computes a plausible number
//! for every row of every span but the last. Nothing traps: on AMD the
//! interpreter's dispatch `default:` writes nothing and does not fault
//! (`runtime/amd/interp.hip`, many comments to that effect), so a missing arm and
//! a mis-packed operand read the same way — a slightly wrong answer.
//!
//! So the rule is inverted: an opcode with no classification is class C and is
//! refused. [`classify`] is an exhaustive `match` over [`DevOp`] precisely so a
//! new opcode is a compile error here rather than an unclassified one at load,
//! and [`classify_wire`] refuses a discriminant no variant claims.
//!
//! # Where this table should eventually live
//!
//! In `crates/packet`, beside `slots.rs`, so `devgen` can stamp the declared
//! classes into a blob's auxiliary metadata at emit time and a loader can check
//! them without re-deriving. It is here because the audit had to exist before the
//! ISA could own it, and because `packet` is being changed concurrently. Nothing
//! below reads anything `packet` does not already export.
//!
//! # Classification basis, and where it disagrees with the ISA's own metadata
//!
//! Every row below rests on the operand contract in `packet::dev`'s doc comments,
//! which `packet::slots` mirrors and a drift test keeps honest. A handful of
//! opcodes are classified from `runtime/amd/interp.hip` instead, because the
//! ISA's spec for them is absent or stale; each says so in its `note`, and the
//! one with no spec anywhere is reported unclassified. See
//! `docs/arch/17-unified-token-batch.md (Part III)`.

use std::collections::BTreeMap;

use packet::dev::{DevInst64, DevOp};
use packet::disasm::op_name;
use serde::Serialize;

use crate::asset::devblob::{DevBlob, DevProg};

/// How an operator learns a row's request identity and absolute position.
///
/// The four classes of the plan's §3, unchanged. [`RowExtent`] is a separate
/// axis this module adds — see that type's own note.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RowClass {
    /// It does not. Work is per row or per element, no cross-row coupling, no
    /// position. Packing changes nothing but the live row count.
    A,
    /// An explicit per-row table already selects slot, state or validity. The
    /// descriptor fills the table the operator already has.
    B,
    /// One immediate or scalar tensor defines the base position and each row's
    /// position is *derived* from its index. **Silently wrong under packing.**
    C,
    /// State is an operand whose shape has no request axis at all. Cannot
    /// express more than one request per launch.
    D,
}

impl RowClass {
    pub fn as_str(self) -> &'static str {
        match self {
            RowClass::A => "A",
            RowClass::B => "B",
            RowClass::C => "C",
            RowClass::D => "D",
        }
    }
}

/// How many rows one packet can describe — an axis §3 does not have.
///
/// §3's class A says "live `M` is the only new input", which assumes the operator
/// *has* a live row count. Many do not: [`DevOp::MoeCombine`] is `i0=H i1=k` with
/// no token axis at all, and [`DevOp::GemvArgmax`] pins `i0=1`. Their arithmetic
/// is row-agnostic — genuinely class A — but a packed batch cannot be expressed
/// in one packet, and raising `M` is not available because there is no `M`. Those
/// operators have a `*Pf`/T-row twin that does carry one, and the emitter must
/// select it. Recording that separately keeps class A meaning what §3 says while
/// still refusing the packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RowExtent {
    /// No rows at all (`Nop`, a collective with no built body).
    None,
    /// The count enters as a flat element count (`i0=n`); packing scales it.
    Elementwise,
    /// An explicit row/`M`/`T`/`n_batch` operand.
    RowCounted,
    /// No row-count operand: the packet describes exactly one token.
    SingleRow,
}

impl RowExtent {
    pub fn as_str(self) -> &'static str {
        match self {
            RowExtent::None => "none",
            RowExtent::Elementwise => "elementwise",
            RowExtent::RowCounted => "row-counted",
            RowExtent::SingleRow => "single-row",
        }
    }
}

/// What the unified-token-batch route must do about this operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Disposition {
    /// Packs unchanged once the live row count is set.
    Ready,
    /// Class B: the descriptor fills the per-row table the operator already has.
    DescriptorFills,
    /// Class A arithmetic with no row-count operand — emit the T-row twin.
    UseRowForm,
    /// Class C: needs a per-row position source, or a launch per span, measured.
    NeedsConversion,
    /// Class D: at most one span per launch until a batched variable-length form
    /// exists.
    PerSpanLaunch,
    /// Not packable and not convertible from here: no ISA classification, or no
    /// built kernel body.
    Refuse,
}

impl Disposition {
    pub fn as_str(self) -> &'static str {
        match self {
            Disposition::Ready => "ready",
            Disposition::DescriptorFills => "descriptor-fills",
            Disposition::UseRowForm => "use-row-form",
            Disposition::NeedsConversion => "needs-conversion",
            Disposition::PerSpanLaunch => "per-span-launch",
            Disposition::Refuse => "refuse",
        }
    }

    /// Whether a program containing this disposition can be admitted to the
    /// packed route as it stands. Only A-with-a-row-count and B qualify.
    pub fn packable(self) -> bool {
        matches!(self, Disposition::Ready | Disposition::DescriptorFills)
    }
}

/// One opcode's row-identity classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct OpClass {
    pub class: RowClass,
    pub extent: RowExtent,
    pub disposition: Disposition,
    /// The operand(s) the classification rests on, named as the ISA names them.
    pub identity: &'static str,
    /// Non-empty when the class is conditional on an immediate, when the ISA
    /// spec and the interpreter disagree, or when there is no spec at all.
    pub note: &'static str,
    /// False when the ISA carries no usable operand spec for this opcode. The
    /// plan's rule then applies: treat as C and refuse.
    pub classified: bool,
}

const fn k(
    class: RowClass,
    extent: RowExtent,
    disposition: Disposition,
    identity: &'static str,
) -> OpClass {
    OpClass {
        class,
        extent,
        disposition,
        identity,
        note: "",
        classified: true,
    }
}

const fn note(mut o: OpClass, n: &'static str) -> OpClass {
    o.note = n;
    o
}

/// Row-agnostic with an explicit row/M/T count: the plan's class A proper.
const fn a_rows(identity: &'static str) -> OpClass {
    k(
        RowClass::A,
        RowExtent::RowCounted,
        Disposition::Ready,
        identity,
    )
}

/// Row-agnostic over a flat element count.
const fn a_elem(identity: &'static str) -> OpClass {
    k(
        RowClass::A,
        RowExtent::Elementwise,
        Disposition::Ready,
        identity,
    )
}

/// Row-agnostic arithmetic, one token per packet: emit the T-row twin instead.
const fn a_one(identity: &'static str) -> OpClass {
    k(
        RowClass::A,
        RowExtent::SingleRow,
        Disposition::UseRowForm,
        identity,
    )
}

/// Per-row indirection already present.
const fn b(identity: &'static str) -> OpClass {
    k(
        RowClass::B,
        RowExtent::RowCounted,
        Disposition::DescriptorFills,
        identity,
    )
}

/// Packet-scalar position base.
const fn cls_c(identity: &'static str) -> OpClass {
    k(
        RowClass::C,
        RowExtent::RowCounted,
        Disposition::NeedsConversion,
        identity,
    )
}

/// Per-sequence carried state.
const fn d(identity: &'static str) -> OpClass {
    k(
        RowClass::D,
        RowExtent::RowCounted,
        Disposition::PerSpanLaunch,
        identity,
    )
}

/// No usable ISA classification. §3's rule: class C, refused.
const fn unclassified(n: &'static str) -> OpClass {
    OpClass {
        class: RowClass::C,
        extent: RowExtent::None,
        disposition: Disposition::Refuse,
        identity: "(none)",
        note: n,
        classified: false,
    }
}

/// A body that is not built. Refused for a different reason than an
/// unclassified opcode, and reported separately.
const fn no_body(identity: &'static str, n: &'static str) -> OpClass {
    OpClass {
        class: RowClass::A,
        extent: RowExtent::None,
        disposition: Disposition::Refuse,
        identity,
        note: n,
        classified: true,
    }
}

/// The static, worst-case-over-operand-modes class of an opcode.
///
/// This is the form a **load-time capability check** must use: it decides from
/// the opcode alone, before any instruction is read. Where an operand mode can
/// promote the class (`HeadNormRope`'s batched ring, `KdaConv3`'s independent
/// sequences), this returns the *stricter* reading and [`refine`] recovers the
/// looser one from the actual instruction.
///
/// The `match` is exhaustive on purpose: adding an opcode to `packet::dev` breaks
/// this build until someone classifies it, which is the only mechanism that
/// keeps "unclassified is refused" from degrading into "unclassified is absent".
pub fn classify(op: DevOp) -> OpClass {
    match op {
        // ---- structural -------------------------------------------------
        DevOp::Nop => k(RowClass::A, RowExtent::None, Disposition::Ready, "(no operands)"),

        // ---- norms, elementwise, embedding ------------------------------
        DevOp::RmsNorm => a_rows("i0=rows"),
        DevOp::RowRms => a_rows("i0=rows"),
        DevOp::Residual => a_elem("i0=n"),
        DevOp::Glu => a_elem("i0=n"),
        DevOp::Embed => a_rows("i0=ntok, t2=ids per row"),
        DevOp::SoftCap => a_elem("i0=n"),
        DevOp::NormResidual => a_rows("i0=rows"),
        DevOp::AddNorm => a_rows("i0=rows"),
        DevOp::NormResidualNorm => a_rows("i0=rows"),
        DevOp::PerLayerInput => a_rows("i0=T, row-local per-layer input block"),
        DevOp::SituGlu => a_elem("i0=n"),
        DevOp::MlaOutGate => a_elem("i0=n"),
        DevOp::ZeroF32 => a_rows("i0=M"),
        DevOp::CastF32Bf16 => a_rows("i0=M"),
        DevOp::QuantFp8 => a_rows("i0=M, per-row a_scale"),
        DevOp::Q8GemmF32 => a_rows("i0=M, dense FP32 rows"),
        DevOp::LayerNormF32 => a_rows("i0=rows"),
        DevOp::ScaledAddF32 => a_elem("i0=n"),
        DevOp::GluF32 => a_rows("i0=rows"),
        DevOp::SiluF32 => a_elem("i0=n"),
        DevOp::DenseGemmF32 => a_rows("i0=M, dense FP32 rows"),
        DevOp::EmbedF16F32 => a_rows("single explicit token row"),
        DevOp::EmbedOverlayBf16 => a_rows("i0=rows, token gather with explicit row overlay"),
        DevOp::LstmCellF32 => a_elem("i0=width, explicit state tensors"),
        DevOp::ArgmaxF32 => a_rows("i0=rows"),
        DevOp::ReluF32 => a_elem("i0=n"),
        DevOp::BroadcastAddF32 => a_rows("i0=rows"),
        DevOp::Conv2dF32 => note(
            cls_c("2D convolution couples neighboring frame rows"),
            "run once per request span until a span descriptor is bound",
        ),
        DevOp::PackNcfwRowsF32 => note(
            cls_c("row packing couples convolution layout dimensions"),
            "run once per request span until a span descriptor is bound",
        ),
        DevOp::CausalDepthwiseConv1dF32 => note(
            cls_c("causal rows belong to one sequence"),
            "run once per request span until a span descriptor is bound",
        ),
        DevOp::RelativeAttentionF32 | DevOp::GroupedAttentionF32 => note(
            cls_c("attention rows belong to one sequence"),
            "run once per request span until a span descriptor is bound",
        ),

        // `i3=out_row0` is a packet-scalar ROW base into the DSA indexer's
        // [ctx][DI] key cache: row j lands at out_row0 + j. Zero (the plain
        // norm) has no cache write and is class A.
        DevOp::LayerNorm => note(
            cls_c("i3=out_row0 (index-key cache write base)"),
            "class A when i3 == 0 (no cache write); refine() reads the immediate",
        ),

        // ---- RoPE / KV writers ------------------------------------------
        // Two addressing modes in one opcode. `i6=n_batch_kv != 0` makes row t
        // sequence t and writes its OWN ring at pos[t] — per-row, class B. Zero
        // is the legacy `out_row0 + t` form, which takes ONE host-patched
        // position per step and cannot express B sequences at B positions.
        DevOp::HeadNormRope => note(
            cls_c("i3=out_row0 with i6=n_batch_kv == 0"),
            "class B when i6 != 0 (batch-major ring at t5=pos[t]); refine() reads the immediate",
        ),
        DevOp::HeadNormRopeFp8 => note(
            cls_c("i3=out_row0 with i6=n_batch_kv == 0"),
            "fp8-KV twin of HeadNormRope; same two modes",
        ),
        // prefill=1 writes ONE selected KV slot for every row; decode uses one
        // slot per row with an optional per-row pos and park mask.
        DevOp::QwenHeadNormRope => note(
            cls_c("i6=prefill selects one KV slot for all rows"),
            "class B when i6 == 0 (one slot per row, t5=pos, t6=active); refine() reads the immediate",
        ),

        // ---- dense matmul families --------------------------------------
        DevOp::Gemm | DevOp::GemmSmall | DevOp::GemmMed | DevOp::GemmWide | DevOp::GemmC5
        | DevOp::GemvAffineQ4 | DevOp::GemmAffineQ4 => {
            a_rows("i0=M")
        }
        DevOp::GemmNorm => a_rows("i0=M"),
        DevOp::GemmGlu => a_rows("i0=M"),
        DevOp::GemmSplitK => a_rows("i0=M"),
        DevOp::GemmFp8
        | DevOp::GemmMedFp8
        | DevOp::GemmSmallFp8
        | DevOp::GemmWideFp8
        | DevOp::GemmC5Fp8 => a_rows("i0=M, i4=a_row0 (contiguous row band)"),
        DevOp::GemmGluFp8 => a_rows("i0=M"),
        DevOp::GemmFp8Blk => a_rows("i0=M"),
        DevOp::GemmMxfp4
        | DevOp::GemmMedMxfp4
        | DevOp::GemmSmallMxfp4
        | DevOp::GemmWideMxfp4
        | DevOp::GemmC5Mxfp4 => a_rows("i0=M"),
        DevOp::GemmGluMxfp4 => a_rows("i0=M"),

        DevOp::Gemv => a_rows("i0=M"),
        DevOp::GemvGlu => a_rows("i0=M"),
        DevOp::GemvQkv => a_rows("i0=M"),
        DevOp::GemvQkvg => a_rows("i0=M"),
        DevOp::GemvFp8 => a_rows("i0=M, i4=a_row0 (contiguous row band)"),
        DevOp::GemvGluFp8 => a_rows("i0=M"),
        DevOp::GemvFp8Blk => a_rows("i0=M, i4=a_row0 (contiguous row band)"),
        DevOp::GemvMxfp4 | DevOp::GemvGluMxfp4 => a_rows("i0=M"),
        DevOp::GemvQkvMxfp4 | DevOp::GemvQkvFp8 => a_rows("i0=M"),
        DevOp::GemvSz | DevOp::GemvGluSz => a_rows("i0=M"),
        DevOp::GemvF32 => a_rows("i0=M"),
        DevOp::DenseGluFp8Blk => a_one("i0=N i1=K, no token axis"),
        // `i0=1` is baked: the fused head+argmax describes exactly one row.
        DevOp::GemvArgmax => note(
            a_one("i0=1 (pinned)"),
            "the terminal segment selects the head by S; this opcode cannot carry S > 1",
        ),

        // ---- attention: GQA ---------------------------------------------
        DevOp::FlashPrefill => cls_c("i4=q_pos0 (and the causal bound built on it)"),
        DevOp::FlashPrefillFp8 => cls_c("i4=q_pos0, inherited from FlashPrefill"),
        DevOp::FlashDecode => b("t5=kv_len[b], t6=decode_slot[b]"),
        DevOp::FlashDecodeFp8 => b("t5=kv_len[b], t6=decode_slot[b]"),
        DevOp::FlashMerge => note(
            a_rows("i0=n_batch"),
            "t3=sinks is one unscaled logit per HEAD with no value row: row-independent, \
             composes with packing unchanged at every nsplit including 1",
        ),

        // ---- attention: MLA ---------------------------------------------
        DevOp::FlashMlaDecode => b("t6=kv_len[b] (qpos = kv_len[b] - 1)"),
        DevOp::FlashMlaDecodeFp8 => b("t6=kv_len[b]"),
        // Per-request kv_len[b], but ONE shared n_tok immediate:
        // q_pos0 = kv_len[b] - n_tok (op_attention.h:2878, :3625). Correct only
        // while every request in the packet contributes the same chunk length.
        DevOp::FlashMlaPrefill => note(
            cls_c("i4=n_tok shared across rows; q_pos0 = kv_len[b] - n_tok"),
            "ISA marks this Reserved/not-built; runtime/amd/interp.hip:3745 dispatches it. \
             Classified from the interpreter",
        ),
        DevOp::FlashMlaPrefillFp8 => {
            cls_c("i4=n_tok shared across rows; q_pos0 = kv_len[b] - n_tok")
        }
        DevOp::FlashGatherDecode => b("t6=kv_len[b], t7=idx per row"),
        DevOp::FlashGatherPrefill => note(
            cls_c("shares exec_flash_mla_prefill's q_pos0 = kv_len[b] - n_tok"),
            "ISA marks this Reserved/not-built; runtime/amd/interp.hip:3748 dispatches it. \
             Classified from the interpreter",
        ),
        DevOp::OUvFold => a_rows("i0=n_batch"),
        DevOp::MlaMergeFold => a_rows("i0=n_batch"),
        DevOp::MlaMaterializePack => a_rows("i0=T (B,T operand axes)"),
        // No kv_len and no position operand at all: causality is the row index
        // inside the packet, so two spans in one block would cross-attend.
        DevOp::FlashMlaMaterializedPrefill => note(
            cls_c("i0=T with no kv_len or position operand; causal bound is the packet row index"),
            "external opus kernel; the packet carries no prefix length, so it can only \
             describe one span starting at position 0",
        ),
        DevOp::AttnSelect => note(
            unclassified(
                "ISA marks op 53 Reserved with no operand names; \
                 runtime/amd/interp.hip:4021 dispatches d_attn_select",
            ),
            "the interpreter reads t3 as a per-batch kv_len, which would be class B — \
             but the ISA spec is absent, so the plan's rule applies",
        ),

        // ---- attention: DSA indexer -------------------------------------
        // op 58 reads kv_len[b] per row (op_attention.h:4558). It is class B.
        // The `s <= q_pos0 + t` derivation the plan attributes to this opcode is
        // in op 117, its prefill twin.
        DevOp::IndexScore => note(
            b("t4=kv_len[b], i0=n_batch"),
            "plan §3/§11 name this opcode for the `s <= q_pos0 + t` derivation; that is \
             IndexScorePf (op 117). op 58 is per-row and class B",
        ),
        DevOp::IndexScoreKpool => b("t6=kv_len[b] (i32[n_batch], token-granular)"),
        DevOp::IndexSelect => a_one("i0=len_max, one cooperative launch, one query row"),
        DevOp::IndexScorePf => cls_c("q_pos0 = kv_len[0] - n_tok (op_attention.h:5173)"),
        DevOp::IndexSelectPf => cls_c("q_pos0 = kv_len[0] - n_tok; row bound q_pos0 + t + 1"),
        DevOp::IndexUnionPf => cls_c("q_pos0 = kv_len[0] - n_tok (op_attention.h:5474)"),
        DevOp::DsaQQuant => a_rows("i0=n_rows (flattened token x index-head)"),
        DevOp::DsaPoolCompress => {
            cls_c("t5=pos read at index 0; boundary from pos[0] / pool_size")
        }
        DevOp::DsaPoolExpand => cls_c("t2=kv_len SCALAR; q_pos0 = kv_len[0] - rows"),
        // One ring per launch, shape [pool_size, head_dim], no request axis.
        DevOp::DsaPoolStash => d("t0/t1 ring [pool_size, head_dim], no batch axis"),

        // ---- MoE: decode-shaped (one token per packet) -------------------
        DevOp::MoeRouter => a_one("i0=H i1=n_exp i2=k, no token axis"),
        DevOp::MoeRouterTopk => a_one("i1=n_exp i2=k, no token axis"),
        DevOp::MoeExpertGlu | DevOp::MoeExpertDown => a_one("i0=slot, one token"),
        DevOp::MoeCombine => a_one("i0=H i1=k, no token axis"),
        DevOp::MoeExpertGluFp8Blk | DevOp::MoeExpertDownFp8Blk => a_one("i0=slot, one token"),
        DevOp::MoeGroupGluFp8Blk | DevOp::MoeGroupDownFp8Blk => {
            a_one("i0=k loops the top-k slots of ONE token")
        }
        DevOp::MoeRouterGemma => a_one("i0=H i1=n_exp i2=k, ONE block, one token"),
        DevOp::MoeRouterGemmaScore | DevOp::MoeRouterGemmaScoreFast => {
            a_one("i0=H i1=n_exp, one token")
        }
        DevOp::MoeRouterGemmaTopk => a_one("i1=n_exp i2=k, ONE block, one token"),
        DevOp::MoeExpertGluGemma | DevOp::MoeExpertDownGemma => a_one("i0=k, one token"),
        DevOp::MoeExpertGluGemmaFp8 | DevOp::MoeExpertDownGemmaFp8 => a_one("i0=k, one token"),
        DevOp::MoeCombineGemma => a_one("i0=H i1=k, one token"),
        DevOp::MoeCombineNormGemma => a_one("i0=H i1=k, one token"),
        DevOp::MoeExpertGluNormGemma => a_one("i0=k, one token"),
        DevOp::MoeCombineResidNormGemma => a_one("i0=H i1=k, one token"),
        DevOp::MoeGluMx => a_rows("i6=n_batch; slot s reads x row s/k"),
        DevOp::MoeDownMx => a_rows("i6=n_batch"),

        // ---- MoE: T-row / grouped ---------------------------------------
        DevOp::MoeRouterTopkPf => a_rows("i4=T"),
        DevOp::MoeRouterGemmaPf => a_rows("i3=T"),
        DevOp::MoeAlignPf => b("i0=T; builds row_token / row_partidx / row_gate maps"),
        DevOp::MoeAlignGemmaPf => b("i0=T; builds row_token / row_partidx / row_gate maps"),
        DevOp::MoeGroupGluPf => b("t5=row_token gathers the source token per gathered row"),
        DevOp::MoeGroupDownPf => b("t6=row_partidx / t7=row_gate scatter per gathered row"),
        DevOp::MoeGroupGluGemmaPf => b("t4=row_token"),
        DevOp::MoeGroupDownGemmaPf => b("t4=row_partidx / t5=row_gate"),
        DevOp::MoeGroupGluGemmaPfW8a8 => b("t4=row_token, t5=ascale[T]"),
        DevOp::MoeGroupDownGemmaPfW8a8 => b("t4=row_partidx / t5=row_gate, t7=fscale[pad]"),
        DevOp::MoeGluMxPf => b("t5=row_token"),
        DevOp::MoeDownMxPf => b("t6=row_partidx / t7=row_gate"),
        DevOp::MoeCombinePf => a_rows("i2=T, i3=t_row0 (contiguous row band)"),
        DevOp::MoeCombineNormGemmaPf => a_rows("i2=T"),

        // ---- selection ---------------------------------------------------
        DevOp::Argmax => a_rows("i1=n_batch"),
        DevOp::ArgmaxFin => a_rows("i1=n_batch"),

        // ---- cross-GPU collectives --------------------------------------
        DevOp::XReduce => a_elem("i0=H (host sets the live element count)"),
        DevOp::XAllGather => a_elem("i0/i1/i2 = per-array element counts"),
        DevOp::XReduceTwoShot => a_elem("i0=n (= t*hidden)"),
        DevOp::XReduceAddNorm => a_one("i0=feat, one row, must fit one workgroup"),
        // interp.hip:5421 documents i0=nparts i1=n_batch i2=vocab_l i3=gate
        // i4=val_slot, and takes n_gpu from the kernarg. dev.rs/slots.rs say
        // `i0=n_gpu i2=slot`.
        DevOp::XArgmaxFin => note(
            a_rows("i1=n_batch (capped at PLOW_XAMAX_MAX_BATCH = 128)"),
            "ISA spec says `i0=n_gpu i2=slot`; runtime/amd/interp.hip:5421-5423 reads \
             i0=nparts i1=n_batch i2=vocab_l i3=gate i4=val_slot and takes n_gpu from the \
             kernarg. Classified from the interpreter",
        ),
        DevOp::XReduceScatter => no_body(
            "i0=n",
            "class A by operand shape, but op_collective.h has no arm; a packet carrying one \
             must be refused at load (PLOW_SEQ_PAR_SEAMS)",
        ),
        DevOp::XFlashMerge => no_body(
            "mirrors FlashMerge",
            "STUB: runtime/amd/interp.hip:5432 dispatches a no-op; body deferred to the CP phase",
        ),

        // ---- hyper-connections (GLM5-Next) ------------------------------
        DevOp::HyperConnPre => a_rows("i0=T"),
        DevOp::HyperConnPost => a_rows("i0=T"),
        DevOp::AttnRes => note(
            a_rows("i0=T; the mix is over one token's nb ring rows"),
            "the softmax couples the block-residual ring rows of a single token, not tokens",
        ),

        // ---- recurrent: GDN (Qwen3.5) -----------------------------------
        DevOp::QwenGdnConv => b("t3=history[B,C,W-1], t4=active[B]"),
        DevOp::QwenGdnStep => b("t6=state[B,HV,V,K], t7=active[B]"),
        DevOp::QwenGatedNorm => b("t4=active[B], i2=B"),
        DevOp::QwenQGateSplit => b("t3=active[B], i2=B"),
        DevOp::QwenSigmoidGate => b("t3=active[B], i1=B"),
        DevOp::QwenRmsNorm => b("t3=active[B], i1=B"),
        DevOp::QwenGdnQkvPrep => a_rows("i4=T"),
        DevOp::QwenGdnGatePrep => a_rows("i1=T"),
        DevOp::QwenGdnConvPrefill => d("t3=history[1,C,W-1] for exactly T valid tokens"),
        DevOp::QwenGdnPrefill => d("t6=state / t7=outstate [1,HV,V,K]"),

        // ---- recurrent: KDA (Kimi K3) -----------------------------------
        DevOp::KdaConv => d("t3=conv_state[3*H*D,W], no batch axis"),
        DevOp::KdaGate => a_rows("i0=T (pure elementwise)"),
        DevOp::KdaGatedNorm => a_rows("i0=T"),
        DevOp::KdaStateStep => d("t6=state[H,D,D], serial over T"),
        // op_kda.h:1041 — `bstride != 0` is the INDEPENDENT-SEQUENCE PATH, with
        // a per-row `parked` mask. Zero is the serial single-sequence form.
        DevOp::KdaConv3 => note(
            d("j0=bstride == 0 is the serial single-sequence form"),
            "class B when j0=bstride != 0: op_kda.h:1041 documents that as the \
             independent-sequence path, with j1=parked as the per-row write mask",
        ),
        DevOp::KdaStateStepG => note(
            d("state[H,D,D] with a serial T axis on the unpacked arm"),
            "class B on a packed_kda object: op_kda.h:1412 takes bstride + parked and \
             op_kda.h:1473 makes the row axis parallel. runtime/amd/interp.hip:2752 passes \
             bstride=0/parked=null on the ordinary arm and sources both from program \
             metadata on the packed one — so the class is NOT decidable from the packet",
        ),
        DevOp::KdaConvStateStepG => b("t7=descriptor carries in.pos and an optional parked mask"),
        DevOp::KdaDecodeFused => b("i0=rows; t7=descriptor carries in.pos and parked"),
        DevOp::KdaChunkPrepare
        | DevOp::KdaChunkIntra
        | DevOp::KdaChunkWu
        | DevOp::KdaChunkCarry => {
            d("documented dense single-sequence; the chunked scan is deliberately absent")
        }

        // ---- the token batch's own row selector ---------------------------
        // `t2=rows` is the explicit per-output-row source table, so each output row's
        // identity is READ, never derived from a base. That is class B by definition, and
        // it is the shape the descriptor exists to fill: `sample_input_rows` IS this table.
        // Note this arm was added because the exhaustive match refused to compile once the
        // opcode landed — which is the mechanism that keeps "unclassified is refused" from
        // decaying into "unclassified is absent".
        DevOp::RowGather => b("t2=rows[S], the explicit source row per output row"),

        // ---- unclassifiable ---------------------------------------------
        DevOp::Mamba2Scan => unclassified(
            "op 90 has no doc comment in packet::dev and no entry in packet::slots \
             (Provenance::Undocumented) — the only such opcode in the ISA",
        ),
    }
}

/// The class of one *instruction*, reading the immediates that select an
/// operand mode.
///
/// Only the opcodes whose [`classify`] note says so are affected; every other
/// opcode returns its static class unchanged. Use this to audit an emitted
/// program; use [`classify`] for a capability check that has no instruction.
pub fn refine(op: DevOp, inst: &DevInst64) -> OpClass {
    match op {
        DevOp::HeadNormRope | DevOp::HeadNormRopeFp8 if inst.i[6] != 0 => note(
            b("i6=n_batch_kv != 0: row t is sequence t, written at t5=pos[t]"),
            "batched ring mode",
        ),
        DevOp::QwenHeadNormRope if inst.i[6] == 0 => note(
            b("i6=prefill == 0: one KV slot per row, t5=pos, t6=active"),
            "decode mode",
        ),
        DevOp::LayerNorm if inst.i[3] == 0 => note(
            a_rows("i0=rows, i3=out_row0 == 0 (no cache write)"),
            "plain norm; no index-key cache write",
        ),
        // slots.rs: KdaConv3 carries `j0=bstride j1=parked`, and `j0` is the
        // integer reading of the `fj[1]` overlay.
        DevOp::KdaConv3 if inst.fj[1] != 0 => note(
            b("j0=bstride != 0: the T rows are B independent sequences, j1=parked masks writes"),
            "independent-sequence path (op_kda.h:1041)",
        ),
        _ => classify(op),
    }
}

/// Classify a raw wire discriminant.
///
/// A value no [`DevOp`] claims is refused as class C — a blob newer than this
/// build, or a corrupt one. On AMD it would otherwise reach the interpreter's
/// dispatch `default:`, which writes nothing and does not trap.
pub fn classify_wire(op: u16) -> OpClass {
    match DevOp::from_u16(op) {
        Some(o) => classify(o),
        None => unclassified("no DevOp claims this wire discriminant"),
    }
}

// ===== report =============================================================

/// One opcode's row in a program's audit.
#[derive(Clone, Debug, Serialize)]
pub struct OpRow {
    pub opcode: u16,
    /// `None` for a discriminant no `DevOp` claims.
    pub name: Option<&'static str>,
    pub count: usize,
    pub class: RowClass,
    pub extent: RowExtent,
    pub disposition: Disposition,
    pub classified: bool,
    pub identity: &'static str,
    #[serde(skip_serializing_if = "str::is_empty")]
    pub note: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProgramAudit {
    /// The `T` this program was compiled for; decode is 1.
    pub t: u32,
    pub n_inst: usize,
    /// Distinct opcodes, in numeric order.
    pub ops: Vec<OpRow>,
    /// Counts by class, over instructions (not distinct opcodes).
    pub by_class: BTreeMap<&'static str, usize>,
    /// True when every opcode is `ready` or `descriptor-fills`.
    pub packable: bool,
    /// Every opcode that is not, named with why — this is the refusal text a
    /// load-time capability check owes the caller.
    pub blockers: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BlobAudit {
    pub blob: String,
    pub target: u32,
    pub n_gpu: u32,
    pub programs: Vec<ProgramAudit>,
    /// True when every program is packable.
    pub packable: bool,
}

/// Audit one program's instruction stream.
pub fn audit_program(prog: &DevProg) -> ProgramAudit {
    let mut counts: BTreeMap<u16, usize> = BTreeMap::new();
    // Worst reading per opcode: an operand-conditional opcode is only clear if
    // EVERY site in the program is clear.
    let mut worst: BTreeMap<u16, OpClass> = BTreeMap::new();
    for inst in &prog.insts {
        *counts.entry(inst.op).or_insert(0) += 1;
        let cls = match DevOp::from_u16(inst.op) {
            Some(o) => refine(o, inst),
            None => classify_wire(inst.op),
        };
        worst
            .entry(inst.op)
            .and_modify(|prev| {
                if prev.disposition.packable() && !cls.disposition.packable() {
                    *prev = cls;
                }
            })
            .or_insert(cls);
    }

    let mut by_class: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut ops = Vec::with_capacity(counts.len());
    let mut blockers = Vec::new();
    for (&opcode, &count) in &counts {
        let cls = worst[&opcode];
        *by_class.entry(cls.class.as_str()).or_insert(0) += count;
        let name = DevOp::from_u16(opcode).map(op_name);
        if !cls.disposition.packable() {
            blockers.push(format!(
                "{} (op {opcode}) x{count}: class {} / {} — {}",
                name.unwrap_or("<unknown>"),
                cls.class.as_str(),
                cls.disposition.as_str(),
                if cls.classified {
                    cls.identity
                } else {
                    "unclassified, refused per plan §3"
                }
            ));
        }
        ops.push(OpRow {
            opcode,
            name,
            count,
            class: cls.class,
            extent: cls.extent,
            disposition: cls.disposition,
            classified: cls.classified,
            identity: cls.identity,
            note: cls.note,
        });
    }

    ProgramAudit {
        t: prog.t,
        n_inst: prog.insts.len(),
        ops,
        by_class,
        packable: blockers.is_empty(),
        blockers,
    }
}

/// Audit every program in a blob, or one selected by its compiled `T`.
pub fn audit(blob: &DevBlob, path: &str, only_t: Option<u32>) -> BlobAudit {
    let programs: Vec<ProgramAudit> = blob
        .progs
        .iter()
        .filter(|p| only_t.is_none_or(|t| p.t == t))
        .map(audit_program)
        .collect();
    BlobAudit {
        blob: path.to_string(),
        target: blob.target,
        n_gpu: blob.tp.as_ref().map_or(1, |t| t.n_gpu),
        packable: programs.iter().all(|p| p.packable),
        programs,
    }
}

/// The audit as text: one line per opcode, greppable.
pub fn text(rep: &BlobAudit) -> String {
    let mut s = format!(
        "{}  target=0x{:08x} n_gpu={} programs={}\n",
        rep.blob,
        rep.target,
        rep.n_gpu,
        rep.programs.len()
    );
    for p in &rep.programs {
        s.push_str(&format!(
            "\nT={} {} insts, {} distinct opcodes  [{}]\n",
            p.t,
            p.n_inst,
            p.ops.len(),
            if p.packable { "PACKABLE" } else { "REFUSED" }
        ));
        let by = |c: &str| p.by_class.get(c).copied().unwrap_or(0);
        s.push_str(&format!(
            "  instructions by class: A={} B={} C={} D={}\n",
            by("A"),
            by("B"),
            by("C"),
            by("D")
        ));
        for r in &p.ops {
            s.push_str(&format!(
                "  {:>3}  {:<30} x{:<5} {}  {:<12} {:<16} {}\n",
                r.opcode,
                r.name.unwrap_or("<unknown>"),
                r.count,
                r.class.as_str(),
                r.extent.as_str(),
                r.disposition.as_str(),
                r.identity
            ));
            if !r.note.is_empty() {
                s.push_str(&format!("       note: {}\n", r.note));
            }
        }
        if !p.blockers.is_empty() {
            s.push_str("  refused:\n");
            for b in &p.blockers {
                s.push_str(&format!("    - {b}\n"));
            }
        }
    }
    s
}

/// Every opcode in the ISA with its static class — the plan's §3 table itself,
/// not an audit of any program.
pub fn table() -> Vec<OpRow> {
    DevOp::ALL
        .iter()
        .map(|&op| {
            let cls = classify(op);
            OpRow {
                opcode: op as u16,
                name: Some(op_name(op)),
                count: 0,
                class: cls.class,
                extent: cls.extent,
                disposition: cls.disposition,
                classified: cls.classified,
                identity: cls.identity,
                note: cls.note,
            }
        })
        .collect()
}

/// The full-ISA table as text.
pub fn table_text(rows: &[OpRow]) -> String {
    let mut s = format!("{} opcodes\n", rows.len());
    for r in rows {
        s.push_str(&format!(
            "{:>3}  {:<32} {}  {:<12} {:<16} {}\n",
            r.opcode,
            r.name.unwrap_or("<unknown>"),
            r.class.as_str(),
            r.extent.as_str(),
            r.disposition.as_str(),
            r.identity
        ));
        if !r.note.is_empty() {
            s.push_str(&format!("     note: {}\n", r.note));
        }
    }
    s
}

#[cfg(test)]
#[path = "opaudit_tests.rs"]
mod tests;
