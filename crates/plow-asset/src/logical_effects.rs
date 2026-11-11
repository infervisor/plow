//! Packet-derived whole-tensor effects. Physical aliases, kernel bounds and
//! counter fence implementations are separate obligations, not established here.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use packet::dev::{DevInst64, DevOp, FP8_BLOCK_GROUP, TENSOR_NONE16, TENSOR_NONE_I};
use packet::devbuild::ProgramRole;
use serde_json::{json, Value};

use crate::program::{Packet, Program};

pub(crate) const CONV1D_SPLITK: u32 = 1 << 20;

#[cfg(test)]
#[test]
fn ring_effects_require_positions_and_write_only_the_declared_destination() {
    let mut d = DevInst64 {
        op: DevOp::StreamTailF32 as u16,
        t: [0; 8],
        ..Default::default()
    };
    for (mode, written) in [(2, &[2][..]), (3, &[0][..])] {
        d.i[7] = mode;
        assert_eq!(outputs(&d).unwrap(), written);
        d.t[5] = TENSOR_NONE16;
        assert!(outputs(&d).is_err());
        d.t[5] = 0;
    }
    d.op = DevOp::AttentionF32 as u16;
    d.i[6] = 0x109 | (250 << 16);
    assert_eq!(outputs(&d).unwrap(), &[0]);
    for slot in 0..6 {
        d.t[slot] = TENSOR_NONE16;
        assert!(outputs(&d).is_err());
        d.t[slot] = 0;
    }
}

pub(crate) fn outputs(d: &DevInst64) -> Result<&'static [usize], String> {
    use DevOp::*;
    let op = DevOp::from_u16(d.op).ok_or("unknown kernel effects")?;
    if op == StreamTailF32 {
        // Gather writes z from src + state; store writes state from z alone.
        let (required, writes): (&[usize], &'static [usize]) = match d.i[7] {
            0 => (&[0, 1, 2, 3, 4, 5], &[0]),
            1 => (&[1, 2, 3, 4], &[2]),
            2 => (&[1, 2, 3, 4, 5], &[2]),
            3 => (&[0, 1, 2, 3, 4, 5], &[0]),
            _ => return Err("unknown stream-tail mode".into()),
        };
        if required.iter().any(|&slot| d.t[slot] == TENSOR_NONE16) {
            return Err("missing required effect operand".into());
        }
        return Ok(writes);
    }
    if op == AttentionF32 && crate::speech_flag_layout::is_ring_attention(d.i[6]) {
        if d.t[..6].contains(&TENSOR_NONE16) {
            return Err("missing ring-attention operand".into());
        }
        return Ok(&[0]);
    }
    if op == Conv1dF32 && d.fj[2] & CONV1D_SPLITK != 0 {
        // SpConvArgs::splitk uses t7 for tickets and partials, rather than a row scale.
        if [0, 1, 2, 7].iter().any(|&slot| d.t[slot] == TENSOR_NONE16) {
            return Err("split-K convolution missing required effect operand".into());
        }
        return Ok(&[0, 7]);
    }
    if let Some(writes) = llm_outputs(op, d) {
        return writes;
    }
    let required = match op {
        Nop => 0,
        ZeroF32 => 1,
        CastF32Bf16 => 2,
        Residual | Glu | GluStrided | NormResidual | Gemm | QuantFp8Block128 => 3,
        AddNorm | NormResidualNorm | MlaBmmFp8 => 4,
        GemmFp8Block128 => 5,
        // Generic speech FP32 ops (runtime/nvidia/op_speech_f32.cuh): out = t0, inputs after it.
        Q8GemmF32
        | LayerNormF32
        | ScaledAddF32
        | GluF32
        | CausalDepthwiseConv1dF32
        | RelativeAttentionF32
        | SiluF32
        | DenseGemmF32
        | EmbedF16F32
        | ArgmaxF32
        | ReluF32
        | BroadcastAddF32
        | Conv2dF32
        | PackNcfwRowsF32
        | GroupedAttentionF32
        | EmbedOverlayBf16
        | GemmF32
        | GatherRowsF32
        | CopyColsF32
        | Conv1dF32
        | ConvTranspose1dF32
        | UnaryF32
        | BinaryF32
        | CumSumF64
        | ReduceRowsF32
        | LengthAffineU32
        | RandF32
        | AttentionF32
        | RowStatsF32
        | RmsNormF32
        | RopeAxialF32 => 2,
        ChunkAttentionF32 => 5,
        LstmCellF32 => 3,
        TrimSilenceF32 => 4,
        _ => return Err(format!("unaudited logical effects for {op:?}")),
    };
    if d.t[..required].contains(&TENSOR_NONE16) {
        return Err("missing required effect operand".into());
    }
    match op {
        Nop => Ok(&[]),
        Residual | Glu | GluStrided | NormResidual | Gemm | ZeroF32 | CastF32Bf16 => Ok(&[0]),
        AddNorm | NormResidualNorm => Ok(&[0, 1]),
        QuantFp8Block128 => Ok(&[0, 2]),
        // The multi-output QB/split variants require their own footprint audit.
        GemmFp8Block128 if d.i[4] == 0 && d.t[5..].iter().all(|&h| h == TENSOR_NONE16) => Ok(&[0]),
        MlaBmmFp8 if d.i[4] == 0 => Ok(&[0]),
        // Flag 128 (reference split): t4 is the read-only per-row reference table and t5 the packed
        // split-K workspace. Otherwise t4 is the split-K workspace (tickets + partials), written by
        // every DenseGemmF32 sharing it.
        DenseGemmF32 if d.i[7] & 128 != 0 && d.t[5] != TENSOR_NONE16 => Ok(&[0, 5]),
        DenseGemmF32 if d.i[7] & 128 != 0 => Ok(&[0]),
        DenseGemmF32 if d.t[4] != TENSOR_NONE16 => Ok(&[0, 4]),
        LstmCellF32 => Ok(&[0, 1]),
        TrimSilenceF32 => Ok(&[0, 3]),
        // Flags bit 20: t7 is the split-K scratch (tickets and partial tiles), written and
        // re-armed by every split conv sharing it.
        Conv1dF32 if (d.fj[2] >> 20) & 1 == 1 && d.t[7] != TENSOR_NONE16 => Ok(&[0, 7]),
        Q8GemmF32
        | LayerNormF32
        | ScaledAddF32
        | GluF32
        | CausalDepthwiseConv1dF32
        | RelativeAttentionF32
        | SiluF32
        | DenseGemmF32
        | EmbedF16F32
        | ArgmaxF32
        | ReluF32
        | BroadcastAddF32
        | Conv2dF32
        | PackNcfwRowsF32
        | GroupedAttentionF32
        | EmbedOverlayBf16
        | GemmF32
        | GatherRowsF32
        | CopyColsF32
        | Conv1dF32
        | ConvTranspose1dF32
        | UnaryF32
        | BinaryF32
        | CumSumF64
        | ReduceRowsF32
        | LengthAffineU32
        | RandF32
        | AttentionF32
        | RowStatsF32
        | RmsNormF32
        | RopeAxialF32
        | ChunkAttentionF32 => Ok(&[0]),
        op => Err(format!("unaudited logical effects for {op:?}")),
    }
}

/// The LLM decoder ops (attention, RoPE/KV writes, GEMV/GEMM, quantization, MoE, Qwen GDN,
/// sampling): `None` for every other op. Only the audited variants are `Ok`.
fn llm_outputs(op: DevOp, d: &DevInst64) -> Option<Result<&'static [usize], String>> {
    use DevOp::*;
    let has = |s: usize| d.t[s] != TENSOR_NONE16;
    let all = |s: &[usize]| s.iter().all(|&k| has(k));
    let none = |s: &[usize]| s.iter().all(|&k| !has(k));
    let handle = |h: u32| h != 0 && h < TENSOR_NONE_I;
    let writes: Option<&'static [usize]> = match op {
        Embed if all(&[0, 1, 2]) => Some(&[0]),
        // op_elementwise.cuh d_embed_frame writes out only, including its in-place RMS norm.
        EmbedFrameBf16
            if all(&[0, 1, 2, 3, 4])
                && none(&[6, 7])
                && d.i[7] & !3 == 0
                && (d.i[7] & 2 == 0 || has(5)) =>
        {
            Some(&[0])
        }
        SoftCap | Argmax | ArgmaxFin if all(&[0, 1]) => Some(&[0]),
        GemvArgmax if all(&[0, 1, 2, 3]) => Some(&[0, 3]),
        FlashDecode if all(&[0, 1, 2, 3, 4, 5]) && none(&[6, 7]) => Some(&[0, 1]),
        FlashDecodeFp8 if all(&[0, 1, 2, 3, 4, 5, 6, 7]) => Some(&[0, 1]),
        FlashMerge if d.i[4] == 0 && all(&[0, 1, 2]) && none(&[3]) => Some(&[0]),
        FlashPrefill
            if d.i[7] == 1 && d.i[4] >> 31 == 0 && all(&[0, 1, 2, 3, 4, 5]) && none(&[6]) =>
        {
            Some(&[1, 5])
        }
        FlashPrefillFp8 if d.i[7] == 1 && d.i[4] >> 31 == 0 && all(&[0, 1, 2, 3, 4, 5, 6, 7]) => {
            Some(&[5])
        }
        HeadNormRope if d.i[7] == 0 && all(&[0, 1, 5]) && has(3) == has(4) => Some(&[0]),
        HeadNormRopeFp8 if d.i[7] == 0 && d.i[5] == 0 && all(&[0, 1, 5, 6]) && has(3) == has(4) => {
            Some(&[0, 6])
        }
        QwenHeadNormRope if all(&[0, 1]) && (d.i[2] == 0 || all(&[3, 4, 5])) => Some(&[0]),
        GemmFp8
            if all(&[0, 1, 2, 3, 4])
                && none(&[5, 6, 7])
                && d.i[3] == 0
                && (d.i[5] == 0 || d.i[5] == FP8_BLOCK_GROUP) =>
        {
            Some(&[0])
        }
        Gemv if d.i[3] == 0 && all(&[0, 1, 2]) => Some(&[0]),
        Gemv if d.i[3] == 4
            && d.i[4] == 0
            && handle(d.i[5])
            && handle(d.i[7])
            && all(&[0, 1, 2])
            && none(&[4, 5, 6, 7]) =>
        {
            Some(if has(3) { &[0, 1, 3] } else { &[0, 1] })
        }
        GemvFp8 if d.i[3] == 0 && all(&[0, 1, 2, 5]) && none(&[3, 4, 6, 7]) => Some(&[0]),
        GemvGlu if d.fj[2] == 0 && d.i[3] == 0 && all(&[0, 1, 2, 5]) && none(&[3, 4, 6, 7]) => {
            Some(&[0])
        }
        GemvGlu
            if d.fj[2] == 0
                && d.i[3] == 4
                && handle(d.i[4])
                && handle(d.i[7])
                && all(&[0, 1, 2, 5])
                && none(&[4, 6, 7]) =>
        {
            Some(if has(3) { &[0, 1, 3] } else { &[0, 1] })
        }
        GemvGluFp8 if d.fj[2] == 0 && d.i[3] == 0 && all(&[0, 1, 2, 3, 4, 5]) && none(&[6, 7]) => {
            Some(&[0])
        }
        GemvQkv
            if all(&[0, 1, 2, 3, 4])
                && none(&[7])
                && d.i[5..8] == [0, 0, 0]
                && has(5) == has(6)
                && has(5) == (d.i[4] != 0) =>
        {
            Some(if has(5) { &[0, 3, 5] } else { &[0, 3] })
        }
        QuantFp8 if d.i[3] == 0 && all(&[0, 1, 2]) && none(&[3, 4, 5, 6, 7]) => Some(&[0, 2]),
        QuantFp8 if d.i[3] == 0 && all(&[0, 1, 2, 3, 4]) && none(&[5, 6, 7]) => Some(&[0, 1, 2]),
        QuantFp8 if d.i[3] == FP8_BLOCK_GROUP && all(&[0, 1, 2]) && none(&[3, 4, 5, 6, 7]) => {
            Some(&[0, 2])
        }
        RmsNorm if all(&[0, 1]) && none(&[3, 4]) => Some(&[0]),
        RmsNorm if all(&[0, 1, 3, 4]) && none(&[5, 6, 7]) => Some(&[0, 3, 4]),
        GluPitched if all(&[0, 1]) => Some(&[0]),
        MoeRouterGemmaTopk if all(&[0, 1, 2]) => Some(&[0]),
        MoeCombineNormGemma | MoeCombineNormGemmaPf | MoeRouterGemmaScoreFast
            if all(&[0, 1, 2, 3]) =>
        {
            Some(&[0])
        }
        MoeRouterGemmaPf | MoeGroupGluGemmaPf | MoeExpertGluGemmaFp8 | MoeExpertDownGemmaFp8
            if all(&[0, 1, 2, 3, 4]) =>
        {
            Some(&[0])
        }
        MoeAlignGemmaPf if all(&[0, 1, 2, 3, 4]) && (d.i[3] == 0 || d.i[0] >= d.i[3]) => {
            Some(&[0, 2, 3, 4])
        }
        MoeGroupDownGemmaPf if all(&[0, 1, 2, 3, 4, 5]) => Some(&[0]),
        MoeGroupGluGemmaPfW8a8 if all(&[0, 1, 2, 3, 4, 5, 6]) => Some(&[0]),
        MoeGroupDownGemmaPfW8a8 if all(&[0, 1, 2, 3, 4, 5, 6, 7]) => Some(&[0]),
        MoeExpertDownGemma if all(&[0, 1, 2, 3]) && (d.i[6] == 0 || all(&[5, 6, 7])) => Some(&[0]),
        MoeExpertGluNormGemma if all(&[0, 1, 2, 3, 4, 5, 6, 7]) => Some(&[0, 5]),
        QwenGatedNorm if all(&[0, 1, 2, 3]) && none(&[5, 6, 7]) => Some(&[0]),
        QwenGdnConv | QwenGdnConvPrefill if all(&[0, 1, 2, 3]) && none(&[5, 6, 7]) => Some(&[0, 3]),
        QwenGdnGatePrep if all(&[0, 1, 2, 3, 4, 5]) && none(&[6, 7]) => Some(&[0, 1]),
        QwenGdnQkvPrep if all(&[0, 1, 2, 3]) && none(&[4, 5, 6, 7]) => Some(&[0, 1, 2]),
        QwenGdnStep if all(&[0, 1, 2, 3, 4, 5, 6]) => Some(&[0, 6]),
        QwenGdnPrefill if all(&[0, 1, 2, 3, 4, 5, 6, 7]) => Some(&[0, 6, 7]),
        QwenQGateSplit if all(&[0, 1, 2]) && none(&[4, 5, 6, 7]) => Some(&[0, 1]),
        QwenRmsNorm if all(&[0, 1, 2]) && none(&[5, 6, 7]) => {
            Some(if has(4) { &[0, 1] } else { &[0] })
        }
        QwenSigmoidGate if all(&[0, 1, 2]) && none(&[4, 5, 6, 7]) => Some(&[0]),
        Embed
        | EmbedFrameBf16
        | SoftCap
        | Argmax
        | ArgmaxFin
        | GemvArgmax
        | FlashDecode
        | FlashDecodeFp8
        | FlashMerge
        | FlashPrefill
        | FlashPrefillFp8
        | HeadNormRope
        | HeadNormRopeFp8
        | QwenHeadNormRope
        | GemmFp8
        | Gemv
        | GemvFp8
        | GemvGlu
        | GemvGluFp8
        | GemvQkv
        | QuantFp8
        | RmsNorm
        | GluPitched
        | MoeRouterGemmaTopk
        | MoeCombineNormGemma
        | MoeCombineNormGemmaPf
        | MoeRouterGemmaScoreFast
        | MoeRouterGemmaPf
        | MoeGroupGluGemmaPf
        | MoeExpertGluGemmaFp8
        | MoeExpertDownGemmaFp8
        | MoeAlignGemmaPf
        | MoeGroupDownGemmaPf
        | MoeGroupGluGemmaPfW8a8
        | MoeGroupDownGemmaPfW8a8
        | MoeExpertDownGemma
        | MoeExpertGluNormGemma
        | QwenGatedNorm
        | QwenGdnConv
        | QwenGdnConvPrefill
        | QwenGdnGatePrep
        | QwenGdnQkvPrep
        | QwenGdnStep
        | QwenGdnPrefill
        | QwenQGateSplit
        | QwenRmsNorm
        | QwenSigmoidGate => None,
        _ => return None,
    };
    Some(writes.ok_or_else(|| format!("unaudited logical effects variant of {op:?}")))
}

/// A decode-rung align below its row threshold (`i0 < i3`) returns before touching an operand on
/// NVIDIA; the prefill arm and AMD (`i3` = phase) do not, so elsewhere [`outputs`] fails it closed.
fn inert(d: &DevInst64, nvidia_decode: bool) -> bool {
    nvidia_decode && d.op == DevOp::MoeAlignGemmaPf as u16 && d.i[3] != 0 && d.i[0] < d.i[3]
}

/// Whether a blob's target fingerprint names an NVIDIA part (canonical name or alias, as given or
/// lower-cased). An unknown spelling or 0 is not NVIDIA.
fn nvidia_target(target: u32) -> bool {
    use hwspec::registry::{ALIASES, ALL};
    let nvidia = |name: &str| {
        ALL.iter()
            .any(|s| s.vendor == hwspec::Vendor::Nvidia && s.name.eq_ignore_ascii_case(name))
    };
    let names = ALL.iter().map(|s| s.name).filter(|&n| nvidia(n)).chain(
        ALIASES
            .iter()
            .filter(|(_, canonical)| nvidia(canonical))
            .flat_map(|&(alias, canonical)| [alias, canonical]),
    );
    target != 0
        && names
            .flat_map(|n| [n.to_string(), n.to_lowercase()])
            .any(|n| packet::devbuild::gpu_fingerprint(&n) == target)
}

/// Tensors an instruction reaches through a handle in `i`/`fj` instead of a `t` slot, with
/// whether it writes them. A `TEN_I` operand is absent at `0` or `0xFFFF`; a raw `T[h]` one (fold,
/// row slot, TMA descriptor) only at `0`.
fn handle_accesses(d: &DevInst64) -> Result<Vec<(u32, bool)>, String> {
    let present = |h: u32| h != 0 && h != TENSOR_NONE_I;
    let reads = |hs: &[u32]| {
        hs.iter()
            .filter(|&&h| present(h))
            .map(|&h| (h, false))
            .collect()
    };
    let fold = d.fj[2];
    let accesses: Vec<(u32, bool)> = match DevOp::from_u16(d.op) {
        Some(DevOp::FlashDecode) if fold != 0 => vec![(fold & 0xFFFF, true), (fold >> 16, true)],
        Some(DevOp::FlashDecodeFp8) if fold != 0 => vec![(fold, false)],
        Some(DevOp::Gemv) if d.i[3] == 4 => reads(&[d.i[5], d.i[6], d.i[7]]),
        Some(DevOp::GemvGlu) if d.i[3] == 4 && fold == 0 => reads(&[d.i[4], d.i[6], d.i[7]]),
        // sm_90a TMA descriptors over A and B (dev_isa.h GEMM: 0 = absent).
        Some(DevOp::Gemm | DevOp::GemmFp8) => [d.i[6], d.i[7]]
            .into_iter()
            .filter(|&h| h != 0)
            .map(|h| (h, false))
            .collect(),
        _ => Vec::new(),
    };
    if accesses.iter().any(|&(h, _)| h >= TENSOR_NONE_I) {
        return Err("invalid handle-encoded effect operand".into());
    }
    Ok(accesses)
}

/// Tensor-map operands read their target through the map: each must be a generated map over the
/// instruction's own operand (GEMM i6 = A t1, i7 = B t2; flash-prefill t7 = K t3 / V t4).
fn tmaps_bound(packet: &Packet<'_>, d: &DevInst64) -> Result<(), String> {
    use packet::rope::{GEN_TMAP_BF16, GEN_TMAP_E4M3, GEN_TMAP_KV_PAIR};
    let gen = |h: u32| packet.generated.iter().find(|g| g.tensor == h);
    let ok = match DevOp::from_u16(d.op) {
        Some(DevOp::Gemm | DevOp::GemmFp8) => [(d.i[6], d.t[1]), (d.i[7], d.t[2])]
            .into_iter()
            .filter(|&(h, _)| h != 0)
            .all(|(h, target)| {
                gen(h).is_some_and(|g| {
                    matches!(g.kind, GEN_TMAP_BF16 | GEN_TMAP_E4M3) && g.aux == u32::from(target)
                })
            }),
        Some(DevOp::FlashPrefill) if d.t[7] != TENSOR_NONE16 => {
            gen(u32::from(d.t[7])).is_some_and(|g| {
                g.kind == GEN_TMAP_KV_PAIR
                    && g.aux == u32::from(d.t[3])
                    && g.scale == u32::from(d.t[4])
            })
        }
        _ => true,
    };
    ok.then_some(())
        .ok_or_else(|| "tensor map not bound to its instruction operand".into())
}

/// A generated tensor map is a load-time descriptor whose target `tmaps_bound` checks; a write
/// would let a reader reach a target outside its declared effects.
fn is_tmap(packet: &Packet<'_>, handle: u32) -> bool {
    use packet::rope::{GEN_TMAP_BF16, GEN_TMAP_E4M3, GEN_TMAP_KV_PAIR};
    packet.generated.iter().any(|g| {
        g.tensor == handle && matches!(g.kind, GEN_TMAP_BF16 | GEN_TMAP_E4M3 | GEN_TMAP_KV_PAIR)
    })
}

/// Operand slots the MoE expert kernels dereference as pointer tables (devgen MoE emit).
fn expert_table_slots(d: &DevInst64) -> &'static [(usize, &'static str)] {
    const EWT: &str = "moe.ewt.";
    const EST: &str = "moe.est.";
    match DevOp::from_u16(d.op) {
        Some(
            DevOp::MoeExpertGluGemmaFp8 | DevOp::MoeExpertDownGemmaFp8 | DevOp::MoeExpertDownGemma,
        ) => &[(3, EWT), (4, EST)],
        Some(DevOp::MoeExpertGluNormGemma) => &[(3, EWT)],
        Some(DevOp::MoeGroupGluGemmaPf | DevOp::MoeGroupDownGemmaPf) => &[(2, EWT)],
        Some(DevOp::MoeGroupGluGemmaPfW8a8 | DevOp::MoeGroupDownGemmaPfW8a8) => {
            &[(2, EWT), (6, EST)]
        }
        _ => &[],
    }
}

/// The tensors a `moe.ewt.{l}` / `moe.est.{l}` pointer table points into, resolved as the
/// runtime fills it at load (last tensor named `layers.{l}.experts.{gate_up,down}_proj[_scale]`).
fn expert_table_targets(packet: &Packet<'_>, handle: u16) -> Result<Option<Vec<u16>>, String> {
    let Some(tensor) = packet.tensors.get(handle as usize) else {
        return Ok(None);
    };
    let (layer, scale) = match (
        tensor.name.strip_prefix("moe.ewt."),
        tensor.name.strip_prefix("moe.est."),
    ) {
        (Some(l), _) => (l, ""),
        (_, Some(l)) => (l, "_scale"),
        _ => return Ok(None),
    };
    ["gate_up_proj", "down_proj"]
        .map(|w| format!("layers.{layer}.experts.{w}{scale}"))
        .iter()
        .map(|suffix| {
            packet
                .tensors
                .iter()
                .rposition(|t| t.name.ends_with(suffix.as_str()))
                .map(|i| i as u16)
                .ok_or_else(|| format!("expert pointer table {} has no target", tensor.name))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

/// Residue band of a strided f32 access: every element index `e` it touches has
/// `e % stride` in `[lo, hi)` (Plow.Speech.band_elem_residue). `None` = whole tensor.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Band {
    stride: u32,
    lo: u32,
    hi: u32,
}

fn band(d: &DevInst64, slot: usize) -> Option<Band> {
    let make = |stride: u32, offset: u32, cols: u32, item_stride: u32| {
        let hi = offset.checked_add(cols)?;
        (stride > 0 && cols > 0 && hi <= stride && item_stride % stride == 0).then_some(Band {
            stride,
            lo: offset,
            hi,
        })
    };
    match (DevOp::from_u16(d.op)?, slot) {
        // out[b*j1 + r*i5 + i6 + c], x[b*j0 + r*i3 + i4 + c], c < i2 (the prefix format, i7
        // bit 0, writes whole item blocks). With a device group (i7 bit 1, t2 read) the output
        // item is g*i0 + b: still a multiple of j1, so the same band.
        (DevOp::CopyColsF32, 0) if d.i[7] & 1 == 0 => make(d.i[5], d.i[6], d.i[2], d.fj[2]),
        (DevOp::CopyColsF32, 1) => make(d.i[3], d.i[4], d.i[2], d.fj[1]),
        // out[r*(j0 or width) + j1 + c], c < width.
        (DevOp::GatherRowsF32, 0) => make(
            if d.fj[1] == 0 { d.i[1] } else { d.fj[1] },
            d.fj[2],
            d.i[1],
            0,
        ),
        _ => None,
    }
}

fn coarse_edges(program: &Program<'_>) -> Result<BTreeSet<(usize, usize)>, String> {
    let n = program.insts.len();
    if n == 0 || program.n_counter as usize != n || program.l2_domains != 0 {
        return Err("logical effects require ordinary coarse counters".into());
    }
    let mut seen = BTreeSet::new();
    let mut dependencies = vec![None; n];
    for entry in program.stream {
        let pc = entry.inst as usize;
        let inst = program
            .insts
            .get(pc)
            .ok_or("effect instruction out of range")?;
        if entry.flags != 0
            || entry.slice >= u32::from(inst.blocks)
            || !seen.insert((pc, entry.slice))
        {
            return Err("fine, duplicate or out-of-range effect slice".into());
        }
        let succ = program
            .succs
            .get(entry.succ_ofs as usize..entry.succ_ofs as usize + entry.succ_len as usize)
            .ok_or("invalid effect successor range")?;
        if succ != [pc as u32] {
            return Err("effect counter is not complete op retirement".into());
        }
        let waits = program
            .waits
            .get(entry.wait_ofs as usize..entry.wait_ofs as usize + entry.wait_len as usize)
            .ok_or("invalid effect wait range")?;
        let mut deps = BTreeSet::new();
        for wait in waits {
            let source = wait.id as usize;
            if source >= pc
                || wait.threshold != u32::from(program.insts[source].blocks)
                || !deps.insert(source)
            {
                return Err("effect wait lacks full earlier-producer completion".into());
            }
        }
        match &dependencies[pc] {
            Some(prior) if *prior != deps => {
                return Err("effect slices have different dependencies".into())
            }
            _ => dependencies[pc] = Some(deps),
        }
    }
    let mut edges = BTreeSet::new();
    for (pc, inst) in program.insts.iter().enumerate() {
        if inst.blocks == 0 || !(0..u32::from(inst.blocks)).all(|slice| seen.contains(&(pc, slice)))
        {
            return Err("missing effect producer slices".into());
        }
        for source in dependencies[pc]
            .as_ref()
            .ok_or("missing effect instruction")?
        {
            edges.insert((*source, pc));
        }
    }
    // The loaded GQ must execute exactly the slices whose counters were reconstructed.
    let canonical: BTreeMap<_, _> = program
        .stream
        .iter()
        .map(|e| ((e.inst, e.slice), e))
        .collect();
    let mut queued = BTreeSet::new();
    for entry in program.gq_stream {
        let original = canonical
            .get(&(entry.inst, entry.slice))
            .ok_or("unknown effect GQ slice")?;
        if !queued.insert((entry.inst as usize, entry.slice))
            || entry.wait_ofs != original.wait_ofs
            || entry.wait_len != original.wait_len
            || entry.succ_ofs != original.succ_ofs
            || entry.succ_len != original.succ_len
            || entry.flags != original.flags
            || entry.seg != original.seg
        {
            return Err("effect GQ gates differ from static stream".into());
        }
    }
    if queued != seen {
        return Err("missing effect GQ slices".into());
    }
    Ok(edges)
}

/// Normalize verified full-operation wire counters to one completion per task.
pub fn coarse_protocol(program: &Program<'_>) -> Result<Value, String> {
    let edges = coarse_edges(program)?;
    let n = program.insts.len();
    let mut waits = vec![Vec::new(); n];
    for (source, target) in edges {
        waits[target].push(source);
    }
    Ok(
        json!({"waits":waits,"succs":(0..n).map(|i|vec![i]).collect::<Vec<_>>(),
        "resource":(0..n).collect::<Vec<_>>(),"stream_idx":vec![0;n],"threshold":{}}),
    )
}

/// One task per complete instruction, plus entry/exit sentinels. Each tensor
/// handle has its own logical pool; no physical non-aliasing claim is made.
pub fn obligation(packet: &Packet<'_>, index: usize) -> Result<Value, String> {
    let program = packet.programs.get(index).ok_or("missing effect program")?;
    if packet.tp
        || program.role.is_packed_sibling()
        || program.role.is_token_batch_body()
        || program.role.is_rowsplit_sibling()
    {
        return Err("peer/packed logical effects not qualified".into());
    }
    let edges = coarse_edges(program)?;
    let count = program.insts.len();
    let exit = count + 1;
    let n = count + 2;
    let mut accesses = Vec::new();
    let mut users: BTreeMap<u16, BTreeMap<usize, (bool, Option<Band>)>> = BTreeMap::new();
    let nvidia_decode =
        matches!(program.role, ProgramRole::DecodeRung { .. }) && nvidia_target(packet.target);
    for (pc, inst) in program.insts.iter().enumerate() {
        if inst.op == DevOp::Nop as u16 || inert(inst, nvidia_decode) {
            continue;
        }
        let writes = outputs(inst)?;
        tmaps_bound(packet, inst)?;
        for &slot in writes {
            if inst.t[slot] == TENSOR_NONE16 {
                return Err("missing effect output".into());
            }
        }
        let slots = inst.t.iter().enumerate().map(|(slot, &handle)| {
            let write = writes.contains(&slot);
            (handle, write, band(inst, slot))
        });
        let handles = handle_accesses(inst)?
            .into_iter()
            .map(|(handle, write)| (handle as u16, write, None));
        for &(slot, prefix) in expert_table_slots(inst) {
            let handle = inst.t[slot];
            if handle != TENSOR_NONE16
                && !packet
                    .tensors
                    .get(handle as usize)
                    .is_some_and(|t| t.name.starts_with(prefix))
            {
                return Err("expert pointer-table operand is not a pointer table".into());
            }
        }
        let mut direct: Vec<_> = slots.chain(handles).collect();
        let mut indirect = Vec::new();
        for &(handle, write, _) in &direct {
            if let Some(targets) = expert_table_targets(packet, handle)? {
                if write {
                    return Err("instruction writes an expert pointer table".into());
                }
                indirect.extend(targets.into_iter().map(|t| (t, false, None)));
            }
        }
        direct.extend(indirect);
        for (handle, write, band) in direct {
            if handle == TENSOR_NONE16 {
                continue;
            }
            let tensor = packet
                .tensors
                .get(handle as usize)
                .ok_or("effect tensor out of range")?;
            if tensor.bytes == 0 || tensor.name.contains("@") {
                return Err("empty or aliased effect tensor".into());
            }
            if write && is_tmap(packet, u32::from(handle)) {
                return Err("instruction writes a tensor map".into());
            }
            users
                .entry(handle)
                .or_default()
                .entry(pc + 1)
                .and_modify(|prior| {
                    *prior = (prior.0 | write, prior.1.filter(|b| Some(*b) == band))
                })
                .or_insert((write, band));
        }
    }
    if users.is_empty() {
        return Err("empty logical effect scope".into());
    }
    let mut waits = vec![Vec::new(); n];
    for task in 1..=count {
        waits[task].push(0usize);
    }
    for &(source, target) in &edges {
        waits[target + 1].push(source + 1);
    }
    waits[exit] = (1..=count).collect();
    let mut graph = vec![Vec::new(); n];
    for (target, sources) in waits.iter().enumerate() {
        for &source in sources {
            graph[source].push(target);
        }
    }
    let mut needed = BTreeMap::<usize, BTreeSet<usize>>::new();
    let mut leases = Vec::new();
    let mut address_map = Vec::new();
    let mut logical_offset = 0u64;
    for (&handle, tasks) in &users {
        let tensor = &packet.tensors[handle as usize];
        // Bands are comparable only under one stride, and a whole-tensor access covers every
        // residue only when the stride fits the tensor.
        let mut strides = tasks.values().filter_map(|(_, b)| b.map(|b| b.stride));
        let banded = strides
            .next()
            .filter(|&s| strides.all(|t| t == s) && u64::from(s) * 4 <= tensor.bytes);
        let extent = |band: Option<Band>| match band.filter(|_| banded.is_some()) {
            Some(b) => (u64::from(b.lo) * 4, u64::from(b.hi - b.lo) * 4),
            None => (0, tensor.bytes),
        };
        leases.push(
            json!({"pool":handle,"allocation":handle,"generation":0,"owner":0,
            "offset":0,"size":tensor.bytes,"acquire":0,"retire":exit,"cancel":null}),
        );
        address_map.push(json!({"name":format!("tensor:{handle}"),"cls":"Persistent",
            "offset":logical_offset,"size":tensor.bytes,
            "writers":tasks.iter().filter_map(|(&task,&(write,_))| write.then_some(task)).collect::<Vec<_>>(),
            "readers":tasks.keys().copied().collect::<Vec<_>>() }));
        logical_offset = logical_offset
            .checked_add(tensor.bytes)
            .ok_or("logical effect extent overflow")?;
        for (&task, &(write, band)) in tasks {
            let (offset, size) = extent(band);
            accesses.push(
                json!({"task":task,"pool":handle,"allocation":handle,"generation":0,
                "owner":0,"offset":offset,"size":size,"write":write}),
            );
            needed.entry(0).or_default().insert(task);
            needed.entry(task).or_default().insert(exit);
            for (&other, &(other_write, other_band)) in tasks.range((task + 1)..) {
                let (other_offset, other_size) = extent(other_band);
                if (write || other_write)
                    && offset < other_offset + other_size
                    && other_offset < offset + size
                {
                    needed.entry(task).or_default().insert(other);
                }
            }
        }
    }
    needed.entry(0).or_default().insert(exit);
    // One parent-pointer tree per source (Lean `treeBefore`), pruned to the chains of the
    // needed targets. Every edge runs from a lower to a higher task, so parent < node.
    let mut trees = Vec::new();
    for (source, targets) in needed {
        let mut parent = vec![usize::MAX; n];
        let mut queue = VecDeque::from([source]);
        parent[source] = source;
        while let Some(node) = queue.pop_front() {
            for &next in &graph[node] {
                if parent[next] == usize::MAX {
                    parent[next] = node;
                    queue.push_back(next);
                }
            }
        }
        let mut kept: Vec<Option<usize>> = vec![None; n];
        for target in targets {
            // An unreached target stays uncovered and Lean rejects the conflict.
            let mut node = target;
            while node != source && parent[node] != usize::MAX && kept[node].is_none() {
                kept[node] = Some(parent[node]);
                node = parent[node];
            }
        }
        trees.push(json!({"source":source,"parent":kept}));
    }
    let dependencies: Vec<_> = edges.iter().map(|&(s, t)| [s + 1, t + 1]).collect();
    Ok(json!({
        "logical_effect_scope": "whole_declared_tensors, strided f32 copies/gathers as stride-residue bands; distinct logical pools; entry/exit lifetime; kernel bounds, physical aliases, runtime retirement and fence implementation external",
        "task_graph":{"n":n,"edges":dependencies},
        "protocol":{"waits":waits,"succs":(0..n).map(|id|vec![id]).collect::<Vec<_>>(),
            "threshold":(0..n).map(|id|(id.to_string(),1)).collect::<BTreeMap<_,_>>(),
            "resource":(0..n).collect::<Vec<_>>(),"stream_idx":vec![0;n]},
        "dependency_paths":vec![Vec::<usize>::new();edges.len()],
        "address_map":address_map,"address_trees":trees,
        "memory_effects":{"schema":1,"leases":leases,"accesses":accesses,
            "fence_counters":(0..n).collect::<Vec<_>>()}
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::{Builder, Model};

    fn model() -> Model {
        let mut b = Builder::new(1);
        let x = b.tensor("x", 16);
        let y = b.tensor("y", 16);
        let a = b.emit(DevOp::Residual, vec![0], &[], |d| {
            d.t[..3].copy_from_slice(&[y, x, x]);
            d.i[0] = 8;
        });
        b.emit(DevOp::Residual, vec![0], &[a], |d| {
            d.t[..3].copy_from_slice(&[x, y, y]);
            d.i[0] = 8;
        });
        let p = b.finish();
        Model {
            n_cu: 1,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        }
    }

    #[test]
    fn derives_real_operands_counters_and_logical_lifetimes() {
        let model = model();
        crate::program::with_model(&model, |packet| {
            let value = obligation(packet, 0).unwrap();
            assert_eq!(value["task_graph"], json!({"n":4,"edges":[[1,2]]}));
            assert_eq!(
                value["memory_effects"]["leases"].as_array().unwrap().len(),
                2
            );
            assert_eq!(
                value["memory_effects"]["accesses"]
                    .as_array()
                    .unwrap()
                    .len(),
                4
            );
            assert_eq!(value["address_map"].as_array().unwrap().len(), 2);
            assert_eq!(value["protocol"]["waits"][2], json!([0, 1]));
        });
    }

    #[test]
    fn stream_tail_store_is_a_state_write_and_gather_reads_state() {
        let mut b = Builder::new(1);
        let [src, z, state, slot, n, pos] =
            ["src", "z", "state", "slot", "n", "pos"].map(|name| b.tensor(name, 64));
        let gather = b.emit(DevOp::StreamTailF32, vec![0], &[], |d| {
            d.t[..6].copy_from_slice(&[z, src, state, slot, n, pos]);
            d.i = [1, 2, 2, 4, 4, 0, 4, 0];
            d.j[0] = 1;
        });
        b.emit(DevOp::StreamTailF32, vec![0], &[gather], |d| {
            d.t[1..5].copy_from_slice(&[z, state, slot, n]);
            d.i = [1, 2, 2, 4, 4, 0, 0, 1];
            d.j[0] = 1;
        });
        let p = b.finish();
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        };
        crate::program::with_model(&model, |packet| {
            let insts = packet.programs[0].insts;
            assert_eq!(outputs(&insts[0]).unwrap(), &[0]);
            assert_eq!(outputs(&insts[1]).unwrap(), &[2]);
            let value = obligation(packet, 0).unwrap();
            let entry = |handle: u32| {
                value["address_map"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|e| e["name"] == format!("tensor:{handle}"))
                    .unwrap()
                    .clone()
            };
            assert_eq!(entry(state)["writers"], json!([2]));
            assert_eq!(entry(state)["readers"], json!([1, 2]));
            assert_eq!(entry(z)["writers"], json!([1]));
            assert_eq!(entry(z)["readers"], json!([1, 2]));
            let none = TENSOR_NONE16;
            let (mut no_pos, mut no_z, mut bad_mode) = (insts[0], insts[1], insts[1]);
            no_pos.t[5] = none;
            no_z.t[1] = none;
            bad_mode.i[7] = 2;
            for inst in [no_pos, no_z, bad_mode] {
                assert!(outputs(&inst).is_err());
            }
        });
    }

    #[test]
    fn copy_group_and_attention_prefix_rows_are_reads() {
        let mut b = Builder::new(1);
        let [x, out, g, ctx, q, rows, idx] =
            ["x", "out", "g", "ctx", "q", "rows", "idx"].map(|name| b.tensor(name, 1024));
        let copy = b.emit(DevOp::CopyColsF32, vec![0], &[], |d| {
            d.t[..3].copy_from_slice(&[out, x, g]);
            d.i = [1, 1, 4, 4, 0, 4, 0, 2 | 4 << 8];
            d.j = [0, 4];
        });
        b.emit(DevOp::AttentionF32, vec![0], &[copy], |d| {
            d.t = [ctx, q, q, q, packet::dev::TENSOR_NONE, rows, out, idx];
            d.i = [1, 1, 1, 1, 64, 0, 32, 1];
        });
        let p = b.finish();
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        };
        crate::program::with_model(&model, |packet| {
            let value = obligation(packet, 0).unwrap();
            let entry = |handle: u32| {
                value["address_map"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|e| e["name"] == format!("tensor:{handle}"))
                    .unwrap()
                    .clone()
            };
            assert_eq!(entry(g)["readers"], json!([1]));
            assert_eq!(entry(g)["writers"], json!([]));
            assert_eq!(entry(rows)["readers"], json!([2]));
            assert_eq!(entry(out)["writers"], json!([1]));
        });
    }

    #[test]
    fn dense_reference_split_writes_its_packed_workspace_not_the_row_table() {
        let mut d = DevInst64 {
            op: DevOp::DenseGemmF32 as u16,
            blocks: 1,
            fj: [0; 3],
            t: [0, 1, 2, TENSOR_NONE16, 4, 5, TENSOR_NONE16, TENSOR_NONE16],
            i: [0, 0, 0, 0, 0, 0, 0, 128 | 8 | 4],
        };
        assert_eq!(outputs(&d).unwrap(), &[0, 5]);
        d.t[5] = TENSOR_NONE16;
        assert_eq!(outputs(&d).unwrap(), &[0]);
        d.i[7] = 8 | 4;
        assert_eq!(outputs(&d).unwrap(), &[0, 4]);
    }

    #[test]
    fn frame_embedding_effects_require_each_input_and_the_norm_weight_when_used() {
        for mode in 0..4 {
            let mut d = DevInst64 {
                op: DevOp::EmbedFrameBf16 as u16,
                blocks: 1,
                fj: [0; 3],
                t: [0, 1, 2, 3, 4, TENSOR_NONE16, TENSOR_NONE16, TENSOR_NONE16],
                i: [1, 8, 16, 2, 16, 0, 16, mode],
            };
            if mode & 2 != 0 {
                assert!(outputs(&d).is_err());
                d.t[5] = 5;
            }
            assert_eq!(outputs(&d).unwrap(), &[0]);
            for slot in 0..5 {
                let mut missing = d;
                missing.t[slot] = TENSOR_NONE16;
                assert!(outputs(&missing).is_err());
            }
            for slot in [6, 7] {
                let mut extra = d;
                extra.t[slot] = 6;
                assert!(outputs(&extra).is_err());
            }
            for unknown in [4, 8, u32::MAX] {
                let mut extra = d;
                extra.i[7] = unknown;
                assert!(outputs(&extra).is_err());
            }
        }
    }

    #[test]
    fn splitk_convolution_writes_scratch_while_row_scale_is_read_only() {
        let mut d = DevInst64 {
            op: DevOp::Conv1dF32 as u16,
            blocks: 1,
            fj: [0, 0, CONV1D_SPLITK],
            t: [
                0,
                1,
                2,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                TENSOR_NONE16,
                3,
            ],
            i: [1, 8, 4, 4, 1, 1, 1, 1],
        };
        assert_eq!(outputs(&d).unwrap(), &[0, 7]);
        for slot in [0, 1, 2, 7] {
            let mut missing = d;
            missing.t[slot] = TENSOR_NONE16;
            assert!(outputs(&missing).is_err());
        }
        d.fj[2] = 0;
        assert_eq!(outputs(&d).unwrap(), &[0]);
    }

    /// One audited LLM entry: op, present t-slots, `i` overrides, `fj[2]`, the expected write set
    /// (`None` = fail closed) and the kernel store it was derived from.
    type Case = (
        DevOp,
        &'static [usize],
        &'static [(usize, u32)],
        u32,
        Option<&'static [usize]>,
        &'static str,
    );

    #[rustfmt::skip]
    const LLM_CASES: &[Case] = &[
        (DevOp::Embed, &[0, 1, 2], &[], 0, Some(&[0]), "op_elementwise.cuh:20 d_embed out"),
        (DevOp::Embed, &[0, 1], &[], 0, None, "ids required"),
        (DevOp::SoftCap, &[0, 1], &[], 0, Some(&[0]), "op_elementwise.cuh:806 st_glob8(out)"),
        (DevOp::Argmax, &[0, 1], &[], 0, Some(&[0]), "op_elementwise.cuh:1095 part[]"),
        (DevOp::ArgmaxFin, &[0, 1], &[], 0, Some(&[0]), "op_elementwise.cuh:1112 ids[b] (in.ids)"),
        (DevOp::GemvArgmax, &[0, 1, 2, 3], &[], 0, Some(&[0, 3]), "op_gemm.cuh:817 C, :828 part"),
        (DevOp::FlashDecode, &[0, 1, 2, 3, 4, 5], &[], 0, Some(&[0, 1]), "op_attention.cuh:965-968 Opart/mlpart"),
        (DevOp::FlashDecode, &[0, 1, 2, 3, 4, 5, 7], &[], 0, None, "t7 HeadNormRope-fold descriptor"),
        (DevOp::FlashDecodeFp8, &[0, 1, 2, 3, 4, 5, 6, 7], &[], 0, Some(&[0, 1]), "interp_sm120.cu:2832 same stores, scales read"),
        (DevOp::FlashMerge, &[0, 1, 2], &[], 0, Some(&[0]), "op_attention.cuh:3250 O"),
        (DevOp::FlashMerge, &[0, 1, 2], &[(4, 1024)], 0, None, "padded MLA layout producer"),
        (DevOp::FlashPrefill, &[0, 1, 2, 3, 4, 5, 7], &[(7, 1)], 0, Some(&[1, 5]), "op_attention.cuh:3816 O; sm90 v3 :5460 fa_ws"),
        (DevOp::FlashPrefill, &[0, 1, 2, 3, 4, 5, 7], &[(7, 2)], 0, None, "split-KV partials"),
        (DevOp::FlashPrefill, &[0, 1, 2, 3, 4, 5, 7], &[(7, 1), (4, 1 << 31)], 0, None, "packed request table"),
        (DevOp::FlashPrefillFp8, &[0, 1, 2, 3, 4, 5, 6, 7], &[(7, 1)], 0, Some(&[5]), "gen_attn_pf_hd256_sliding_fp8kv.cu:763 O"),
        (DevOp::HeadNormRope, &[0, 1, 2, 3, 4, 5], &[], 0, Some(&[0]), "op_norm.cuh:1295 out (q / KV row)"),
        (DevOp::HeadNormRope, &[0, 1, 5], &[], 0, Some(&[0]), "op_norm.cuh:1295 v row"),
        (DevOp::HeadNormRope, &[0, 1, 2, 3, 5], &[], 0, None, "op_norm.cuh:1274 sin read with cos"),
        (DevOp::HeadNormRope, &[0, 1, 2, 3, 4, 5, 6], &[(7, 1)], 0, None, "fused KV pair writes t6"),
        (DevOp::HeadNormRopeFp8, &[0, 1, 2, 3, 4, 5, 6, 7], &[], 0, Some(&[0, 6]), "op_norm.cuh:1356 e4m3 row, :1474 scale"),
        (DevOp::HeadNormRopeFp8, &[0, 1, 5, 6], &[(5, 1)], 0, None, "unaudited i5"),
        (DevOp::QwenHeadNormRope, &[0, 1, 2, 3, 4, 5, 6], &[(2, 64)], 0, Some(&[0]), "op_qwen_gdn.cuh:840 out"),
        (DevOp::QwenHeadNormRope, &[0, 1, 5, 6], &[], 0, Some(&[0]), "op_qwen_gdn.cuh:840 v row, no rotary"),
        (DevOp::QwenHeadNormRope, &[0, 1, 5, 6], &[(2, 64)], 0, None, "op_qwen_gdn.cuh:825 rotary reads cos/sin"),
        (DevOp::GemmFp8, &[0, 1, 2, 3, 4], &[], 0, Some(&[0]), "interp_sm120.cu:1588 d_gemm_w8a8 C"),
        (DevOp::GemmFp8, &[0, 1, 2, 3, 4], &[(5, 128)], 0, Some(&[0]), "block-128 scales, same C"),
        (DevOp::GemmFp8, &[0, 1, 2, 3, 4, 5], &[], 0, None, "split-K / multi-output"),
        (DevOp::Gemv, &[0, 1, 2], &[], 0, Some(&[0]), "interp_sm120.cu:2425 C"),
        (DevOp::Gemv, &[0, 1, 2, 3], &[(3, 4), (5, 9), (7, 10)], 0, Some(&[0, 1, 3]), "op_qwen_gdn.cuh:743 fold norm writes x, residual"),
        (DevOp::Gemv, &[0, 1, 2, 3, 4, 5], &[(3, 8)], 0, None, "AddNorm fold producer"),
        (DevOp::Gemv, &[0, 1, 2, 3], &[(3, 4), (7, 10)], 0, None, "qwen_fold_residual8 needs residual in"),
        (DevOp::GemvFp8, &[0, 1, 2, 5], &[], 0, Some(&[0]), "interp_sm120.cu:2683"),
        (DevOp::GemvFp8, &[0, 1, 2, 5], &[(3, 4)], 0, None, "AMD NRN fold"),
        (DevOp::GemvGlu, &[0, 1, 2, 5], &[], 0, Some(&[0]), "interp_sm120.cu:2662"),
        (DevOp::GemvGlu, &[0, 1, 2, 3, 5], &[(3, 4), (4, 7), (7, 10)], 0, Some(&[0, 1, 3]), "interp_sm120.cu:2601 fold norm"),
        (DevOp::GemvGlu, &[0, 1, 2, 5], &[], 3, None, "d_gemv_glu_nrn writes TEN_I(3)"),
        (DevOp::GemvGluFp8, &[0, 1, 2, 3, 4, 5], &[], 0, Some(&[0]), "interp_sm120.cu:2697"),
        (DevOp::GemvGluFp8, &[0, 1, 2, 3, 4, 5], &[], 7, None, "AMD NRN fold writes through i3"),
        (DevOp::GemvQkv, &[0, 1, 2, 3, 4, 5, 6], &[(4, 2048)], 0, Some(&[0, 3, 5]), "interp_sm120.cu:2570 q, k, v"),
        (DevOp::GemvQkv, &[0, 1, 2, 3, 4], &[], 0, Some(&[0, 3]), "interp_sm120.cu:2570 k, v"),
        (DevOp::GemvQkv, &[0, 1, 2, 3, 4], &[(5, 1)], 0, None, "bias handles"),
        (DevOp::QuantFp8, &[0, 1, 2], &[], 0, Some(&[0, 2]), "interp_sm120.cu:1627 e4m3, scale"),
        (DevOp::QuantFp8, &[0, 1, 2, 3, 4], &[], 0, Some(&[0, 1, 2]), "op_gemm.cuh:2851/2861 fused GLU stores x"),
        (DevOp::QuantFp8, &[0, 1, 2], &[(3, 128)], 0, Some(&[0, 2]), "op_gemm.cuh:2771 group-128"),
        (DevOp::QuantFp8, &[0, 1, 2], &[(3, 64)], 0, None, "unaudited group"),
        (DevOp::RmsNorm, &[0, 1, 2], &[], 0, Some(&[0]), "interp_sm120.cu:1435"),
        (DevOp::RmsNorm, &[0, 1, 2, 3, 4], &[], 0, Some(&[0, 3, 4]), "op_norm.cuh:211/220 fused quant"),
        (DevOp::RmsNorm, &[0, 1, 2, 3], &[], 0, None, "half the fused quant pair"),
        (DevOp::GluPitched, &[0, 1], &[], 0, Some(&[0]), "op_elementwise.cuh:945"),
        (DevOp::MoeRouterGemmaTopk, &[0, 1, 2], &[], 0, Some(&[0]), "op_moe.cuh:1352 table"),
        (DevOp::MoeCombineNormGemma, &[0, 1, 2, 3], &[], 0, Some(&[0]), "op_moe.cuh:2382"),
        (DevOp::MoeCombineNormGemmaPf, &[0, 1, 2, 3], &[], 0, Some(&[0]), "op_moe.cuh:3249"),
        (DevOp::MoeRouterGemmaScoreFast, &[0, 1, 2, 3], &[], 0, Some(&[0]), "op_moe.cuh:906"),
        (DevOp::MoeRouterGemmaPf, &[0, 1, 2, 3, 4], &[], 0, Some(&[0]), "op_moe.cuh:2984 table"),
        (DevOp::MoeGroupGluGemmaPf, &[0, 1, 2, 3, 4], &[], 0, Some(&[0]), "op_moe.cuh:3379"),
        (DevOp::MoeExpertGluGemmaFp8, &[0, 1, 2, 3, 4], &[], 0, Some(&[0]), "op_moe.cuh:2119"),
        (DevOp::MoeExpertDownGemmaFp8, &[0, 1, 2, 3, 4], &[], 0, Some(&[0]), "op_moe.cuh:2256"),
        (DevOp::MoeAlignGemmaPf, &[0, 1, 2, 3, 4], &[], 0, Some(&[0, 2, 3, 4]), "op_moe.cuh:3065/3153 meta, rowtok, rowpart, rowgate"),
        (DevOp::MoeAlignGemmaPf, &[0, 1, 2, 3, 4], &[(0, 8), (3, 4)], 0, Some(&[0, 2, 3, 4]), "interp_sm120.cu:3191 i0 >= i3 runs"),
        (DevOp::MoeAlignGemmaPf, &[0, 1, 2, 3, 4], &[(0, 2), (3, 4)], 0, None, "interp_sm120.cu:1669 prefill arm ignores i3"),
        (DevOp::MoeGroupDownGemmaPf, &[0, 1, 2, 3, 4, 5], &[], 0, Some(&[0]), "op_moe.cuh:3477 part"),
        (DevOp::MoeGroupGluGemmaPfW8a8, &[0, 1, 2, 3, 4, 5, 6], &[], 0, Some(&[0]), "op_moe_sm90.cuh:657"),
        (DevOp::MoeGroupDownGemmaPfW8a8, &[0, 1, 2, 3, 4, 5, 6, 7], &[], 0, Some(&[0]), "op_moe_sm90.cuh:742"),
        (DevOp::MoeExpertDownGemma, &[0, 1, 2, 3, 5, 6, 7], &[(6, 4)], 0, Some(&[0]), "op_moe.cuh:1498 part (grouped arm)"),
        (DevOp::MoeExpertDownGemma, &[0, 1, 2, 3, 5, 7], &[(6, 4)], 0, None, "grouped arm reads t6"),
        (DevOp::MoeExpertGluNormGemma, &[0, 1, 2, 3, 4, 5, 6, 7], &[], 0, Some(&[0, 5]), "op_moe.cuh:2835, xn stage :2796"),
        (DevOp::MoeExpertGluNormGemma, &[0, 1, 2, 3, 4, 6, 7], &[], 0, None, "xn scratch required"),
        (DevOp::QwenGatedNorm, &[0, 1, 2, 3, 4], &[], 0, Some(&[0]), "op_qwen_gdn.cuh:365"),
        (DevOp::QwenGdnConv, &[0, 1, 2, 3, 4], &[], 0, Some(&[0, 3]), "op_qwen_gdn.cuh:134 history in place"),
        (DevOp::QwenGdnConvPrefill, &[0, 1, 2, 3, 4], &[], 0, Some(&[0, 3]), "op_qwen_gdn.cuh:962"),
        (DevOp::QwenGdnGatePrep, &[0, 1, 2, 3, 4, 5], &[], 0, Some(&[0, 1]), "op_qwen_gdn.cuh:1296"),
        (DevOp::QwenGdnQkvPrep, &[0, 1, 2, 3], &[], 0, Some(&[0, 1, 2]), "op_qwen_gdn.cuh:1217"),
        (DevOp::QwenGdnStep, &[0, 1, 2, 3, 4, 5, 6, 7], &[], 0, Some(&[0, 6]), "op_qwen_gdn.cuh:308/353 core, state"),
        (DevOp::QwenGdnPrefill, &[0, 1, 2, 3, 4, 5, 6, 7], &[], 0, Some(&[0, 6, 7]), "qwen_gdn.rs:321-331 core, outstate, state"),
        (DevOp::QwenGdnPrefill, &[0, 1, 2, 3, 4, 5, 6], &[], 0, None, "outstate required"),
        (DevOp::QwenQGateSplit, &[0, 1, 2, 3], &[], 0, Some(&[0, 1]), "op_qwen_gdn.cuh:450"),
        (DevOp::QwenRmsNorm, &[0, 1, 2, 3], &[], 0, Some(&[0]), "op_qwen_gdn.cuh:575"),
        (DevOp::QwenRmsNorm, &[0, 1, 2, 3, 4], &[], 0, Some(&[0, 1]), "op_qwen_gdn.cuh:532 residual in place"),
        (DevOp::QwenSigmoidGate, &[0, 1, 2, 3], &[], 0, Some(&[0]), "op_qwen_gdn.cuh:486"),
    ];

    #[test]
    fn llm_ops_write_exactly_their_audited_slots_and_fail_closed_otherwise() {
        for &(op, present, ints, fold, expected, source) in LLM_CASES {
            let mut d = DevInst64 {
                op: op as u16,
                blocks: 1,
                fj: [0, 0, fold],
                t: [TENSOR_NONE16; 8],
                i: [0; 8],
            };
            for &slot in present {
                d.t[slot] = slot as u16;
            }
            for &(k, v) in ints {
                d.i[k] = v;
            }
            match expected {
                Some(writes) => assert_eq!(outputs(&d), Ok(writes), "{op:?} ({source})"),
                None => assert!(outputs(&d).is_err(), "{op:?} must fail closed ({source})"),
            }
        }
    }

    #[test]
    fn handle_encoded_operands_are_effects() {
        let mut b = Builder::new(1);
        let [o, ml, q, k, v, len, at, ctr, x, w, res, delta, gamma] = [
            "o", "ml", "q", "k", "v", "len", "at", "ctr", "x", "w", "res", "delta", "gamma",
        ]
        .map(|name| b.tensor(name, 64));
        let flash = b.emit(DevOp::FlashDecode, vec![0], &[], |d| {
            d.t[..6].copy_from_slice(&[o, ml, q, k, v, len]);
            d.j[1] = at | ctr << 16;
        });
        b.emit(DevOp::Gemv, vec![0], &[flash], |d| {
            d.t[..4].copy_from_slice(&[o, x, w, res]);
            d.i[3] = 4;
            d.i[5] = at;
            d.i[6] = delta;
            d.i[7] = gamma;
        });
        let p = b.finish();
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![],
        };
        crate::program::with_model(&model, |packet| {
            let value = obligation(packet, 0).unwrap();
            let entry = |handle: u32| {
                value["address_map"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|e| e["name"] == format!("tensor:{handle}"))
                    .unwrap()
                    .clone()
            };
            assert_eq!(entry(at)["writers"], json!([1]));
            assert_eq!(entry(at)["readers"], json!([1, 2]));
            assert_eq!(entry(ctr)["writers"], json!([1]));
            assert_eq!(entry(delta)["readers"], json!([2]));
            assert_eq!(entry(gamma)["readers"], json!([2]));
            assert_eq!(entry(x)["writers"], json!([2]));
            let mut bad = packet.programs[0].insts[0];
            bad.fj[2] = TENSOR_NONE_I | ctr << 16;
            assert!(handle_accesses(&bad).is_err());
            let mut align = packet.programs[0].insts[0];
            align.op = DevOp::MoeAlignGemmaPf as u16;
            align.i[0] = 2;
            align.i[3] = 4;
            assert!(inert(&align, true) && !inert(&align, false));
            let h100 = packet::devbuild::gpu_fingerprint("H100 SXM5");
            assert!(
                nvidia_target(h100) && nvidia_target(packet::devbuild::gpu_fingerprint("h100"))
            );
            assert!(
                !nvidia_target(0) && !nvidia_target(packet::devbuild::gpu_fingerprint("MI300X"))
            );
            align.i[0] = 4;
            assert!(!inert(&align, true));
        });
    }

    #[test]
    fn tensor_maps_must_be_bound_to_their_own_operands() {
        let mut b = Builder::new(1);
        let [c, a, w, map_a, map_w] = ["c", "a", "w", "map_a", "map_w"].map(|n| b.tensor(n, 128));
        b.emit(DevOp::Gemm, vec![0], &[], |d| {
            d.t[..3].copy_from_slice(&[c, a, w]);
            d.i[6] = map_a;
            d.i[7] = map_w;
        });
        let p = b.finish();
        let map = |tensor: u32, aux: u32| packet::rope::GenTensor {
            tensor,
            kind: packet::rope::GEN_TMAP_BF16,
            aux,
            ..Default::default()
        };
        let mut model = Model {
            n_cu: 1,
            target: 0,
            tensors: p.tensors.clone(),
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![1],
            gen: vec![map(map_a, a), map(map_w, w)],
        };
        crate::program::with_model(&model, |packet| {
            let value = obligation(packet, 0).unwrap();
            let readers = |h: u32| {
                value["address_map"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|e| e["name"] == format!("tensor:{h}"))
                    .unwrap()["readers"]
                    .clone()
            };
            assert_eq!(readers(map_a), json!([1]));
        });
        let mut b = Builder::new(1);
        let [c, a, w, map_a, map_w] = ["c", "a", "w", "map_a", "map_w"].map(|n| b.tensor(n, 128));
        let copy = b.emit(DevOp::CopyColsF32, vec![0], &[], |d| {
            d.t[..3].copy_from_slice(&[map_a, map_w, packet::dev::TENSOR_NONE]);
            d.i = [1, 1, 4, 4, 0, 4, 0, 0];
        });
        b.emit(DevOp::Gemm, vec![0], &[copy], |d| {
            d.t[..3].copy_from_slice(&[c, a, w]);
            d.i[6] = map_a;
            d.i[7] = map_w;
        });
        let replaced = b.finish();
        let bound = std::mem::replace(&mut model.progs[0], replaced);
        crate::program::with_model(&model, |packet| {
            assert_eq!(
                obligation(packet, 0).unwrap_err(),
                "instruction writes a tensor map"
            )
        });
        model.progs[0] = bound;
        model.gen[1].aux = c;
        crate::program::with_model(&model, |packet| assert!(obligation(packet, 0).is_err()));
        model.gen.pop();
        crate::program::with_model(&model, |packet| assert!(obligation(packet, 0).is_err()));
    }

    #[test]
    fn expert_pointer_tables_read_their_targets() {
        let build = |write_table: bool| {
            let mut b = Builder::new(1);
            let [o, x, r, ewt, est, gu, dn, gus, dns] = [
                "o",
                "x",
                "r",
                "moe.ewt.0",
                "moe.est.0",
                "fp8/model.layers.0.experts.gate_up_proj",
                "fp8/model.layers.0.experts.down_proj",
                "fp8/model.layers.0.experts.gate_up_proj_scale",
                "fp8/model.layers.0.experts.down_proj_scale",
            ]
            .map(|n| b.tensor(n, 128));
            b.emit(DevOp::MoeExpertGluGemmaFp8, vec![0], &[], |d| {
                d.t[..5].copy_from_slice(&[o, x, r, ewt, est]);
            });
            if write_table {
                b.emit(DevOp::Residual, vec![0], &[], |d| {
                    d.t[..3].copy_from_slice(&[ewt, x, x]);
                    d.i[0] = 1;
                });
            }
            let p = b.finish();
            (
                Model {
                    n_cu: 1,
                    target: 0,
                    tensors: p.tensors.clone(),
                    progs: vec![p],
                    kv_row_insts: vec![],
                    prog_t: vec![1],
                    gen: vec![],
                },
                [gu, dn, gus, dns],
            )
        };
        let (mut model, targets) = build(false);
        crate::program::with_model(&model, |packet| {
            let value = obligation(packet, 0).unwrap();
            for h in targets {
                let entry = value["address_map"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|e| e["name"] == format!("tensor:{h}"))
                    .unwrap()
                    .clone();
                assert_eq!(entry["readers"], json!([1]));
            }
        });
        model.tensors[model.progs[0].insts[0].t[3] as usize].name = "scratch".into();
        crate::program::with_model(&model, |packet| {
            assert_eq!(
                obligation(packet, 0).unwrap_err(),
                "expert pointer-table operand is not a pointer table"
            )
        });
        model.tensors[model.progs[0].insts[0].t[3] as usize].name = "moe.ewt.0".into();
        model.tensors[targets[1] as usize].name = "unrelated".into();
        crate::program::with_model(&model, |packet| assert!(obligation(packet, 0).is_err()));
        let (model, _) = build(true);
        crate::program::with_model(&model, |packet| {
            assert_eq!(
                obligation(packet, 0).unwrap_err(),
                "instruction writes an expert pointer table"
            )
        });
    }

    #[test]
    fn rejects_uncovered_kernels_aliases_and_partial_completion() {
        for mutation in 0..5 {
            let mut model = model();
            match mutation {
                0 => model.progs[0].insts[0].op = DevOp::KdaStateStep as u16,
                1 => model.tensors[0].name = "x@band16".into(),
                2 => model.progs[0].waits[0].threshold = 0,
                3 => model.progs[0].gq_stream[0].succ_len = 0,
                _ => model.progs[0].insts[0].t[0] = packet::dev::TENSOR_NONE,
            }
            crate::program::with_model(&model, |packet| {
                assert!(obligation(packet, 0).is_err(), "mutation {mutation}")
            });
        }
    }
}
