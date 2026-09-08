//! Operator row-identity classes — how an opcode learns a row's request identity and its
//! absolute position, and therefore whether packing rows from several requests into one matrix
//! changes its meaning.
//!
//! This is a property of the ISA, not of a checkpoint, which is what makes the unified token
//! batch model-general: classify the opcodes a program contains and the answer for the family
//! follows. See `docs/arch/17-unified-token-batch.md` §3 (and the plan it was written from) for
//! the argument; this module is the machine-readable form of that table, and the operator audit
//! and the load-time capability check both read it rather than restating it.
//!
//! # Why the match is exhaustive
//!
//! A new opcode with no entry here would default to something. Whichever default is chosen is
//! wrong for some opcode, and the failure mode is asymmetric: defaulting to [`RowClass::A`]
//! lets a position-deriving operator through a packed batch, and on AMD the interpreter's
//! dispatch `default:` neither writes nor traps, so the result is a fluent wrong answer. The
//! match therefore has no `_` arm — adding an opcode is a compile error until somebody
//! classifies it. Callers that must tolerate an unknown numeric opcode (a blob from a newer
//! compiler) go through [`class_of_op`], which reports [`RowClass::C`] for anything it cannot
//! name, because C is the class that gets refused.
//!
//! # Granularity
//!
//! The class is per OPCODE, not per instruction. Several opcodes have both a per-row form and a
//! packet-scalar form selected by an immediate ([`DevOp::HeadNormRope`]'s `n_batch_kv`,
//! [`DevOp::QwenHeadNormRope`]'s `prefill`). Those are classified `C`: the class drives a
//! refusal, and refusing an instruction that happened to be in its per-row form costs a
//! conversion, while admitting one in its scalar form costs a wrong token.

use crate::dev::DevOp;

/// How an operator learns a row's identity and position, and what packing does to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RowClass {
    /// **Row-agnostic.** No cross-row coupling and no position; work is per row or per element.
    /// Packing changes nothing but the live `M`. This is most of the FLOPs in every family,
    /// which is why the packing is worth doing at all.
    A,
    /// **Per-row indirection already present.** An explicit per-row table already selects the
    /// slot, the carried state or the row's validity. Packing changes nothing once the
    /// descriptor fills the table the operator already reads — do not invent a parallel
    /// mechanism beside it.
    B,
    /// **Packet-scalar position base.** One immediate or scalar tensor defines the base, and
    /// each row's position (or slot) is *derived* from its index. Under packing, rows of
    /// earlier spans get another request's sequence length. **Silently wrong**, with no trap.
    /// Either the operator gets a descriptor-aware form that reads `positions[]`, or it is
    /// launched once per span and that cost is measured, or the program is refused.
    C,
    /// **Per-sequence carried state.** State is an operand whose shape has no request axis at
    /// all. Cannot express more than one request per launch, so a step carries at most one
    /// span per D-class layer until a batched variable-length form exists. This is a real
    /// limit on the GDN/KDA families and is stated, not hidden: the dense and MoE parts of
    /// those models still pack.
    D,
}

impl RowClass {
    /// `true` when packing rows from several requests into one launch preserves the operator's
    /// meaning with nothing more than a correctly filled descriptor.
    pub fn packs(self) -> bool {
        matches!(self, RowClass::A | RowClass::B)
    }

    /// Single-letter name, as the audit prints it.
    pub fn letter(self) -> char {
        match self {
            RowClass::A => 'A',
            RowClass::B => 'B',
            RowClass::C => 'C',
            RowClass::D => 'D',
        }
    }
}

/// Class of a numeric opcode. An opcode this build cannot name is reported as
/// [`RowClass::C`] — the class that gets refused — because an unclassified operator is exactly
/// the case where nothing is known about how it reads a row's position.
pub fn class_of_op(op: u16) -> RowClass {
    match DevOp::from_u16(op) {
        Some(op) => class_of(op),
        None => RowClass::C,
    }
}

/// Class of `op`. See the module docs for why this match has no wildcard arm.
pub fn class_of(op: DevOp) -> RowClass {
    use DevOp::*;
    match op {
        // ---- A: row-agnostic ---------------------------------------------------------------
        // Nothing here reads a position or couples two rows. Dense projections, norms,
        // elementwise, quantization, argmax, the collectives, and the whole MoE routing /
        // grouping / combine chain, whose row maps are built from the batch it is given.
        Nop | RmsNorm | RowRms | Residual | Glu | SituGlu | SoftCap | LayerNorm | NormResidual
        | AddNorm | NormResidualNorm | QuantFp8 | ZeroF32 | CastF32Bf16 | MlaOutGate
        | KdaGatedNorm | QwenGatedNorm | QwenQGateSplit | QwenSigmoidGate | QwenRmsNorm => {
            RowClass::A
        }
        // `Embed` gathers rows of the EMBEDDING TABLE by token id — one id per row, no
        // position, no cross-row coupling. It is not a hidden-row gather; `RowGather` is.
        Embed => RowClass::A,
        Gemm | GemmSmall | GemmMed | GemmWide | GemmC5 | GemmNorm | GemmGlu | GemmSplitK
        | GemmFp8 | GemmMedFp8 | GemmSmallFp8 | GemmGluFp8 | GemmWideFp8 | GemmC5Fp8
        | GemmFp8Blk | GemmMxfp4 | GemmMedMxfp4 | GemmSmallMxfp4 | GemmWideMxfp4 | GemmC5Mxfp4
        | GemmGluMxfp4 | DenseGluFp8Blk => RowClass::A,
        Gemv | GemvSz | GemvGlu | GemvGluSz | GemvArgmax | GemvQkv | GemvQkvg | GemvF32
        | GemvFp8 | GemvGluFp8 | GemvFp8Blk | GemvQkvFp8 | GemvMxfp4 | GemvGluMxfp4
        | GemvQkvMxfp4 => RowClass::A,
        Argmax | ArgmaxFin => RowClass::A,
        // Collectives reduce whole tensors at live row extents identical on every rank; the
        // element count is an input, the row identity is not.
        XReduce | XReduceScatter | XAllGather | XFlashMerge | XArgmaxFin | XReduceTwoShot
        | XReduceAddNorm => RowClass::A,
        // MoE. Routing, grouping, scatter and combine are the row-grouping contract shared
        // across families: each expert receives its rows from any request or phase, and the
        // maps are built from the batch. `MoeCombinePf`'s `i3 = t_row0` is a band OFFSET into
        // the packed rows, not a sequence position.
        MoeRouter | MoeRouterTopk | MoeRouterTopkPf | MoeAlignPf | MoeExpertGlu | MoeExpertDown
        | MoeCombine | MoeCombinePf | MoeGroupGluPf | MoeGroupDownPf | MoeExpertGluFp8Blk
        | MoeExpertDownFp8Blk | MoeGroupGluFp8Blk | MoeGroupDownFp8Blk | MoeGluMx | MoeDownMx
        | MoeGluMxPf | MoeDownMxPf => RowClass::A,
        MoeRouterGemma
        | MoeRouterGemmaScore
        | MoeRouterGemmaScoreFast
        | MoeRouterGemmaTopk
        | MoeRouterGemmaPf
        | MoeAlignGemmaPf
        | MoeExpertGluGemma
        | MoeExpertDownGemma
        | MoeExpertGluGemmaFp8
        | MoeExpertDownGemmaFp8
        | MoeExpertGluNormGemma
        | MoeCombineGemma
        | MoeCombineNormGemma
        | MoeCombineResidNormGemma
        | MoeGroupGluGemmaPf
        | MoeGroupDownGemmaPf
        | MoeCombineNormGemmaPf
        | MoeGroupGluGemmaPfW8a8
        | MoeGroupDownGemmaPfW8a8 => RowClass::A,
        // Attention epilogues: per (row, head) folds of partials. No position, no coupling —
        // including GPT-OSS's sink logit, which is one unscaled value per HEAD with no value
        // row, so it composes with packing unchanged.
        FlashMerge | MlaMergeFold | OUvFold | AttnSelect => RowClass::A,
        // Layout/prep ops over `[T, ...]`: reshapes, unpacks and elementwise gate preparation.
        MlaMaterializePack | QwenGdnQkvPrep | QwenGdnGatePrep | KdaGate => RowClass::A,
        // Per-token hyper-connection mix/push and the attention-residual ring: the ring is per
        // token and the workgroups partition tokens.
        HyperConnPre | HyperConnPost | AttnRes => RowClass::A,
        // Per-row Hadamard + fp8 quant of the indexer queries. `n_rows` is the only axis; it
        // has no pool, no `ape`, and no position, unlike the rest of the DSA chain.
        DsaQQuant => RowClass::A,

        // ---- B: per-row indirection already present ---------------------------------------
        // `FlashDecode.t6 = decode_slot[b]` already selects the physical KV slot per compact
        // row; the descriptor fills it from `span_of(row).slot`. The gathered and fp8-KV
        // decode arms carry the same per-row `kv_len`/slot operands.
        FlashDecode | FlashDecodeFp8 | FlashMlaDecode | FlashMlaDecodeFp8 | FlashGatherDecode => {
            RowClass::B
        }
        // The GDN decode family: `state[B, ...]` / `history[B, ...]` with an optional
        // `active[B]` write mask. That mask IS the ISA's name for "this row is padding, write
        // nothing" — the park mask extends it rather than replacing it.
        QwenGdnConv | QwenGdnStep => RowClass::B,
        // `t4 = pos[rows]` selects each row's ring slot (`pos % pool_size`); the position is
        // read per row rather than derived from a chunk end.
        DsaPoolStash => RowClass::B,
        // The terminal segment's own gather: an explicit `[S]` row-index table, which is
        // exactly the table the descriptor fills. Bounds-checked against live `M`.
        RowGather => RowClass::B,

        // ---- C: packet-scalar position base ------------------------------------------------
        // `i4 = q_pos0` plus the causal bound built on it. Every row's sequence length is
        // derived from one scalar, which is correct for one request's contiguous chunk and
        // silently wrong for a packed batch. This is the one conversion the cheapest family
        // (dense GQA) needs, and the reason the class exists.
        FlashPrefill | FlashPrefillFp8 => RowClass::C,
        // MLA prefill carries the same per-packet query base.
        FlashMlaPrefill | FlashMlaPrefillFp8 | FlashMlaMaterializedPrefill | FlashGatherPrefill => {
            RowClass::C
        }
        // RoPE + KV write. `HeadNormRope` has a per-row form (`n_batch_kv != 0`, row `t` at
        // `pos[t]`) and a legacy form that takes ONE host-patched `out_row0` for the whole
        // packet; `QwenHeadNormRope`'s `prefill = 1` writes every row into a single selected
        // KV slot. The scalar forms are the hazard, and the class is per opcode, so both are C.
        HeadNormRope | HeadNormRopeFp8 | QwenHeadNormRope => RowClass::C,
        // The DSA lightning-indexer chain: the hardest C-class case and the reason the class
        // is named after a derivation rather than an operand. Nothing in the prefill DSA chain
        // ever produced a per-row array, so `q_pos0 = kv_len[0] - rows` and
        // `seq_len = q_pos0 + row + 1` are how a row learns its length. Under packing that is
        // wrong for every row outside the last span, with no trap — and DSA already needs its
        // own capability marker because a legal packet otherwise runs dense and ignores `t7`.
        IndexScore | IndexScorePf | IndexScoreKpool | IndexSelect | IndexSelectPf
        | IndexUnionPf | DsaPoolExpand | DsaPoolCompress => RowClass::C,

        // ---- D: per-sequence carried state -------------------------------------------------
        // Operand shapes with no request axis at all: `state`/`outstate` `[1, HV, V, K]`,
        // `history[1, C, W-1]` for exactly `T` valid tokens, and KDA's chunk pipeline, which
        // is documented dense single-sequence with the chunked scan deliberately absent.
        // A step carries at most one span per D-class layer, or launches per span; the dense
        // and MoE parts of these models still pack.
        QwenGdnPrefill | QwenGdnConvPrefill => RowClass::D,
        // The KDA family. Its conv and state-step opcodes carry BOTH forms behind one
        // immediate: `bstride != 0` makes the `T` rows `B` independent sequences with per-row
        // carried state, and `bstride == 0` makes them consecutive tokens of ONE sequence
        // (`op_kda.h`, "INDEPENDENT-SEQUENCE PATH"). The class is per opcode, so the
        // single-sequence form decides — and here D is the safe direction rather than the
        // strict one: D BOUNDS a step to one span per layer, it does not refuse the program,
        // so a false D costs a scheduling limit while a false B costs a wrong answer. The
        // chunked scan is deliberately absent and the chunk ops are documented dense
        // single-sequence.
        KdaConv | KdaConv3 | KdaStateStep | KdaStateStepG | KdaConvStateStepG | KdaDecodeFused
        | KdaChunkPrepare | KdaChunkIntra | KdaChunkWu | KdaChunkCarry => RowClass::D,
        Mamba2Scan => RowClass::D,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every opcode is classified, and the classification is total: the match above has no
    /// wildcard, so this also proves `ALL` and the enum agree on membership.
    #[test]
    fn every_opcode_has_a_class() {
        for &op in DevOp::ALL {
            let class = class_of(op);
            assert_eq!(
                class,
                class_of_op(op as u16),
                "{}: numeric and typed lookups disagree",
                op.c_name()
            );
            assert!(matches!(
                class,
                RowClass::A | RowClass::B | RowClass::C | RowClass::D
            ));
        }
    }

    /// An opcode value this build cannot name is class C — the class that gets refused.
    #[test]
    fn unknown_opcodes_are_refused_as_c() {
        assert_eq!(class_of_op(DevOp::COUNT), RowClass::C);
        assert_eq!(class_of_op(u16::MAX), RowClass::C);
    }

    /// The §3 examples, spelled out. These are the operators the design argument names, so a
    /// reclassification of any of them is a change to the argument and must be deliberate.
    #[test]
    fn the_named_examples_keep_their_documented_class() {
        for op in [
            DevOp::Gemm,
            DevOp::GemmSmall,
            DevOp::GemmMed,
            DevOp::GemmWide,
            DevOp::Gemv,
            DevOp::RmsNorm,
            DevOp::NormResidual,
            DevOp::Residual,
            DevOp::MoeRouter,
            DevOp::MoeRouterTopk,
            DevOp::MoeCombine,
            DevOp::FlashMerge,
        ] {
            assert_eq!(class_of(op), RowClass::A, "{}", op.c_name());
        }
        for op in [
            DevOp::FlashDecode,
            DevOp::QwenGdnConv,
            DevOp::QwenGdnStep,
            DevOp::QwenGatedNorm,
        ] {
            // QwenGatedNorm carries `active[B]` but has no slot/state indirection, so it is
            // A by the stronger test (row-agnostic); the mask makes it *safe*, not indirect.
            assert!(class_of(op).packs(), "{}", op.c_name());
        }
        assert_eq!(class_of(DevOp::FlashDecode), RowClass::B);
        for op in [
            DevOp::FlashPrefill,
            DevOp::IndexScore,
            DevOp::DsaPoolExpand,
            DevOp::IndexSelectPf,
        ] {
            assert_eq!(class_of(op), RowClass::C, "{}", op.c_name());
        }
        for op in [
            DevOp::QwenGdnPrefill,
            DevOp::QwenGdnConvPrefill,
            DevOp::KdaChunkPrepare,
            DevOp::KdaChunkCarry,
            DevOp::KdaStateStep,
        ] {
            assert_eq!(class_of(op), RowClass::D, "{}", op.c_name());
        }
    }

    /// Only A and B pack. C and D are the work and the limit respectively, and a caller that
    /// treats "not A" as "not packable" would refuse the whole decode side of GDN.
    #[test]
    fn packs_is_exactly_a_and_b() {
        assert!(RowClass::A.packs() && RowClass::B.packs());
        assert!(!RowClass::C.packs() && !RowClass::D.packs());
    }
}
