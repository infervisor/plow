//! What an opcode does, in the few classes a knob's scope is written in (checkpoint S).
//!
//! The collective set is the one `plowrt`'s TP audit patches (`is_tp_collective`) plus the
//! all-to-all; every other class is read off the opcode's family. Exhaustive over `DevOp`, so a
//! new opcode does not compile until it has a class.

use crate::dev::DevOp;

pub const CLASSES: &[&str] = &[
    "collective",
    "attention",
    "indexer",
    "moe",
    "gemm",
    "gemv",
    "norm",
    "kv_write",
    "native_route",
    "recurrent",
    "elementwise",
    "sample",
];

/// The classes of `op`: one functional class, plus `native_route` for host-launched native
/// kernels (AITER, hipBLASLt, the TP indexer adapter).
pub fn op_classes(op: DevOp) -> &'static [&'static str] {
    use DevOp::*;
    match op {
        XReduce | XReduceScatter | XAllGather | XArgmaxFin | XReduceTwoShot | XReduceAddNorm
        | XAllToAllHeads => &["collective"],
        XFlashMerge => &["attention"],

        FlashPrefill
        | FlashDecode
        | FlashMerge
        | FlashDecodeFp8
        | FlashPrefillFp8
        | FlashMlaDecode
        | FlashMlaPrefill
        | AttnSelect
        | FlashGatherDecode
        | FlashGatherPrefill
        | MlaMergeFold
        | OUvFold
        | FlashMlaDecodeFp8
        | AttnRes
        | MlaOutGate
        | MlaMaterializePack
        | FlashMlaMaterializedPrefill
        | DsaPoolCompress
        | DsaPoolExpand
        | DsaPoolStash
        | DsaQQuant => &["attention"],
        FlashMlaPrefillFp8 => &["attention", "native_route"],

        IndexScore | IndexSelect | IndexScorePf | IndexSelectPf | IndexUnionPf
        | IndexScoreKpool => &["indexer"],
        IndexTpPf => &["indexer", "native_route"],

        HeadNormRope | HeadNormRopeFp8 | QwenHeadNormRope => &["kv_write"],

        MoeRouter
        | MoeExpertGlu
        | MoeExpertDown
        | MoeCombine
        | MoeExpertGluFp8Blk
        | MoeExpertDownFp8Blk
        | MoeGroupGluFp8Blk
        | MoeGroupDownFp8Blk
        | MoeRouterTopk
        | MoeRouterGemma
        | MoeExpertGluGemma
        | MoeExpertDownGemma
        | MoeCombineGemma
        | MoeExpertGluGemmaFp8
        | MoeExpertDownGemmaFp8
        | MoeRouterGemmaScore
        | MoeRouterGemmaTopk
        | MoeRouterGemmaScoreFast
        | MoeCombineNormGemma
        | MoeExpertGluNormGemma
        | MoeCombineResidNormGemma
        | MoeRouterGemmaPf
        | MoeAlignGemmaPf
        | MoeGroupGluGemmaPf
        | MoeGroupDownGemmaPf
        | MoeCombineNormGemmaPf
        | MoeGroupGluGemmaPfW8a8
        | MoeGroupDownGemmaPfW8a8
        | MoeRouterTopkPf
        | MoeAlignPf
        | MoeGroupGluPf
        | MoeGroupDownPf
        | MoeCombinePf
        | MoeGluMx
        | MoeDownMx
        | MoeGluMxPf
        | MoeDownMxPf => &["moe"],
        MoeAiterFp8Pf => &["moe", "native_route"],

        Gemm | GemmNorm | GemmSmall | GemmMed | GemmGlu | GemmFp8 | GemmMedFp8 | GemmSmallFp8
        | GemmGluFp8 | GemmMxfp4 | GemmWide | GemmC5 | GemmMedMxfp4 | GemmSmallMxfp4
        | GemmWideMxfp4 | GemmC5Mxfp4 | GemmWideFp8 | GemmC5Fp8 | GemmFp8Blk | GemmGluMxfp4
        | GemmSplitK | DenseGluFp8Blk => &["gemm"],
        GemmLtPf | GemmBlkPf => &["gemm", "native_route"],

        Gemv | GemvGlu | GemvQkv | GemvFp8 | GemvGluFp8 | GemvFp8Blk | GemvSz | GemvGluSz
        | GemvMxfp4 | GemvGluMxfp4 | GemvQkvg | GemvQkvMxfp4 | GemvQkvFp8 | GemvF32 => &["gemv"],

        RmsNorm | RowRms | NormResidual | AddNorm | NormResidualNorm | LayerNorm | QwenRmsNorm
        | QwenGatedNorm | KdaGatedNorm => &["norm"],

        KdaConv | KdaGate | Mamba2Scan | KdaStateStep | KdaConv3 | KdaStateStepG
        | KdaConvStateStepG | KdaChunkPrepare | KdaChunkIntra | KdaChunkWu | KdaChunkCarry
        | KdaDecodeFused | QwenGdnConv | QwenGdnStep | QwenQGateSplit | QwenSigmoidGate
        | QwenGdnConvPrefill | QwenGdnQkvPrep | QwenGdnGatePrep | QwenGdnPrefill | HyperConnPre
        | HyperConnPost => &["recurrent"],

        Argmax | ArgmaxFin | GemvArgmax | RowGather => &["sample"],

        Nop | Residual | Glu | Embed | SoftCap | QuantFp8 | SituGlu | ZeroF32 | CastF32Bf16
        | PerLayerInput => &["elementwise"],
    }
}

/// `(opcode, classes)` for every `DevOp`: the class table checkpoint S is sent.
pub fn class_table() -> Vec<(u16, &'static [&'static str])> {
    (0..=u16::MAX)
        .filter_map(DevOp::from_u16)
        .map(|op| (op as u16, op_classes(op)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_collective_class_is_the_tp_audit_set_plus_all_to_all() {
        let collectives: Vec<u16> = class_table()
            .into_iter()
            .filter(|(_, cs)| cs.contains(&"collective"))
            .map(|(op, _)| op)
            .collect();
        assert_eq!(collectives, [24, 25, 26, 28, 29, 116, 160]);
    }

    #[test]
    fn every_class_is_declared() {
        for (op, cs) in class_table() {
            assert!(!cs.is_empty(), "op {op}");
            for c in cs {
                assert!(CLASSES.contains(c), "op {op}: undeclared class {c}");
            }
        }
    }
}
