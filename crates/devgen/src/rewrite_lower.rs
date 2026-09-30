//! `PLOW_EMIT_REWRITE`: the egglog rewrite's extracted fused graph decides the emitter fusions that
//! lower it. plowc extracts the complete graph (`rewrite::fused_sites_for_config`) and passes fused
//! kind → anchor checkpoint weight names; an emitter asks with the tensor handle it would fuse on.

use std::sync::atomic::{AtomicPtr, Ordering};

use packet::devbuild::Builder;

use crate::{emit_config, RewriteSites};

/// A devgen fused opcode and the rewrite fused kinds it lowers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lowering {
    /// `AddNorm`. A residual3 site's outer add and norm lower here too; its inner (combine) add
    /// stays with the MoE combine, which under TP precedes the collective.
    AddNorm,
    /// `NormResidualNorm`, with or without the layer scalar.
    NormResidualNorm,
    /// `NormLinear`: fusion of preceding RMSNorm or LayerNorm into linear projections (Q/K/V/gate/up/lm_head).
    NormLinear,
    /// `GatedMlp`: activation(gate) * up fused into SwiGLU / GeGLU.
    GatedMlp,
    /// `NormRope`: Q/K norm fused into RoPE or RoPE + Head scale.
    NormRope,
    /// `Residual3Norm`: 3-way residual combine + norm (MoE block boundary).
    Residual3Norm,
    /// `Conv1dF32` / `ConvTranspose1dF32` input activation (codec and vocoder convolutions).
    ConvInputAct,
    /// `Conv1dF32` / `ConvTranspose1dF32` output activation; a linear is a 1x1 conv there.
    ConvOutputAct,
    /// `Conv1dF32` / `ConvTranspose1dF32` `residual`.
    ConvResidual,
}

impl Lowering {
    pub(crate) fn kinds(self) -> &'static [&'static str] {
        match self {
            Lowering::AddNorm => &["FusedResidualNorm", "FusedResidual3Norm"],
            Lowering::NormResidualNorm => &["FusedNormResidualNorm", "FusedNormResidualScaleNorm"],
            Lowering::NormLinear => &[
                "FusedNormLinear",
                "FusedZeroCenteredNormLinear",
                "FusedNormLinearBias",
                "FusedLayerNormLinear",
            ],
            Lowering::GatedMlp => &["SwiGLU"],
            Lowering::NormRope => &[
                "FusedNormRope",
                "FusedZeroCenteredNormRope",
                "FusedNormRopeScale",
            ],
            Lowering::Residual3Norm => &["FusedResidual3Norm"],
            Lowering::ConvInputAct => &[
                "FusedActConv1d",
                "FusedParamActConv1d",
                "FusedActConv1dResidual",
                "FusedParamActConv1dResidual",
                "FusedParamActConv1dAct",
            ],
            Lowering::ConvOutputAct => &[
                "FusedConv1dAct",
                "FusedParamActConv1dAct",
                "FusedLinearAct",
                "FusedLinearBiasAct",
                "FusedLayerNormLinearBiasAct",
            ],
            Lowering::ConvResidual => &[
                "FusedConv1dResidual",
                "FusedActConv1dResidual",
                "FusedParamActConv1dResidual",
                "FusedLinearBiasResidual",
            ],
        }
    }

    pub(crate) fn egg_rules(self) -> &'static [&'static str] {
        match self {
            Lowering::AddNorm => &[
                "residual-rmsnorm-fuse",
                "residual-zero-centered-rmsnorm-fuse",
                "residual-layernorm-fuse",
            ],
            Lowering::NormResidualNorm => &[
                "norm-residual-rmsnorm-fuse",
                "norm-residual-scale-rmsnorm-fuse",
            ],
            Lowering::NormLinear => &[
                "rmsnorm-linear-fuse",
                "zero-centered-rmsnorm-linear-fuse",
                "rmsnorm-linearbias-fuse",
                "layernorm-linear-fuse",
            ],
            Lowering::GatedMlp => &["gated-mlp-fuse"],
            Lowering::NormRope => &[
                "rmsnorm-rope-fuse",
                "zero-centered-rmsnorm-rope-fuse",
                "rmsnorm-rope-scale-fuse",
            ],
            Lowering::Residual3Norm => &["residual3-rmsnorm-fuse"],
            Lowering::ConvInputAct => &[
                "act-conv1d-fuse",
                "param-act-conv1d-fuse",
                "act-conv1d-residual-fuse",
                "param-act-conv1d-residual-fuse",
                "param-act-conv1d-act-fuse",
            ],
            Lowering::ConvOutputAct => &[
                "conv1d-act-fuse",
                "param-act-conv1d-act-fuse",
                "linear-act-fuse",
                "linearbias-act-fuse",
                "layernorm-linearbias-act-fuse",
            ],
            Lowering::ConvResidual => &[
                "conv1d-residual-fuse",
                "act-conv1d-residual-fuse",
                "param-act-conv1d-residual-fuse",
                "linearbias-residual-fuse",
            ],
        }
    }
}

/// The fusion decisions of one audio network (`codec.pkt`, `s3gen.pkt`): the fused sites of its
/// own graph (`rewrite::fused_sites_for_codec_config`), keyed by the export's tensor names (the
/// packet's `w.`-prefixed ones minus the prefix). Without sites (`PLOW_EMIT_REWRITE=0`, a direct
/// devgen caller, no rewrite for the export) every query answers the lowering's hand choice.
#[derive(Clone, Copy)]
pub(crate) struct ConvFusions<'a>(pub(crate) Option<&'a RewriteSites>);

/// Where an activation between a producing and a consuming convolution runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActAt {
    ProducerOutput,
    ConsumerInput,
    /// Its own `UnaryF32` pass.
    Separate,
}

impl ConvFusions<'_> {
    fn has(self, lowering: Lowering, weight: &str) -> Option<bool> {
        let weight = weight.strip_prefix("w.").unwrap_or(weight);
        self.0.map(|sites| lowered(sites, lowering, weight))
    }

    /// Whether the conv on `weight` (kernel `kernel`) applies its input activation itself.
    ///
    /// Cost-model override: a pointwise conv is a plain GEMM that applies a fused input
    /// activation to its A tile once per output-column tile, so a wide one recomputes it several
    /// times; a separate pass computes it once and is measured faster on SNAC's pointwise convs.
    /// Pointwise convs therefore never fuse their input activation, whatever the rewrite says.
    pub(crate) fn input_act(self, weight: &str, kernel: u32, hand: bool) -> bool {
        kernel != 1 && self.has(Lowering::ConvInputAct, weight).unwrap_or(hand)
    }

    pub(crate) fn output_act(self, weight: &str, hand: bool) -> bool {
        self.has(Lowering::ConvOutputAct, weight).unwrap_or(hand)
    }

    pub(crate) fn residual(self, weight: &str, hand: bool) -> bool {
        self.has(Lowering::ConvResidual, weight).unwrap_or(hand)
    }

    /// An activation between two convolutions: the rewrite fuses it into one of them (the two
    /// forms tie at extraction, see rules.egg), and a consumer placement still passes
    /// [`Self::input_act`]'s cost model.
    pub(crate) fn between(self, producer: &str, consumer: &str, consumer_kernel: u32, hand: ActAt) -> ActAt {
        if self.0.is_none() {
            return hand;
        }
        if self.output_act(producer, false) {
            ActAt::ProducerOutput
        } else if self.input_act(consumer, consumer_kernel, false) {
            ActAt::ConsumerInput
        } else {
            ActAt::Separate
        }
    }
}

static SITES: AtomicPtr<RewriteSites> = AtomicPtr::new(std::ptr::null_mut());

/// After `emit_config::install`. With the knob off, or on by default without sites (a direct devgen
/// caller, or a checkpoint plowc has no rewrite for), nothing is installed and every query answers
/// `None`: the hand fusions. Only an explicit `PLOW_EMIT_REWRITE=1` without sites refuses.
pub(crate) fn install(sites: Option<RewriteSites>) {
    let on = emit_config::active().emit_rewrite;
    let ptr = match sites {
        Some(sites) if on => Box::into_raw(Box::new(sites)),
        None if on && emit_config::explicitly_set("emit_rewrite") => {
            panic!("PLOW_EMIT_REWRITE=1 needs the compiler's rewrite sites; emit through plowc")
        }
        _ => std::ptr::null_mut(),
    };
    SITES.store(ptr, Ordering::Release);
}

/// `None` with the knob off: the emitter keeps its own decision. Otherwise whether the extracted
/// graph holds a fused node `lowering` covers, anchored on `gamma`'s checkpoint name.
pub(crate) fn fused(b: &Builder, lowering: Lowering, gamma: u32) -> Option<bool> {
    let ptr = SITES.load(Ordering::Acquire);
    if ptr.is_null() {
        return None;
    }
    if gamma as usize >= b.n_tensors() {
        return Some(false);
    }
    // SAFETY: `install` stores Box::into_raw and never frees it, one snapshot per compile, like
    // `install_whole_graph_fusions`.
    let sites = unsafe { &*ptr };
    Some(lowered(sites, lowering, b.tensor_name(gamma)))
}

fn lowered(sites: &RewriteSites, lowering: Lowering, weight: &str) -> bool {
    lowering
        .kinds()
        .iter()
        .any(|kind| sites.get(*kind).is_some_and(|w| w.contains(weight)))
}

/// Check if any rewrite sites were installed.
#[allow(dead_code)]
pub(crate) fn has_sites() -> bool {
    !SITES.load(Ordering::Acquire).is_null()
}

/// Retrieve the active rewrite sites if installed.
pub(crate) fn active_sites() -> Option<RewriteSites> {
    let ptr = SITES.load(Ordering::Acquire);
    if ptr.is_null() {
        None
    } else {
        unsafe { Some((*ptr).clone()) }
    }
}

/// Query which egglog rules corresponding to `lowering` were matched for `anchor`.
#[allow(dead_code)]
pub(crate) fn query_lowering_rules(b: &Builder, lowering: Lowering, anchor: u32) -> Vec<String> {
    if let Some(true) = fused(b, lowering, anchor) {
        lowering.egg_rules().iter().map(|&s| s.to_string()).collect()
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULES: &str = include_str!("../../rewrite/src/egl/rules.egg");

    fn sites(kv: &[(&str, &[&str])]) -> RewriteSites {
        kv.iter()
            .map(|(k, w)| (k.to_string(), w.iter().map(|s| s.to_string()).collect()))
            .collect()
    }

    #[test]
    fn every_lowered_kind_is_a_rule_target() {
        for lowering in [
            Lowering::AddNorm,
            Lowering::NormResidualNorm,
            Lowering::NormLinear,
            Lowering::GatedMlp,
            Lowering::NormRope,
            Lowering::Residual3Norm,
            Lowering::ConvInputAct,
            Lowering::ConvOutputAct,
            Lowering::ConvResidual,
        ] {
            for kind in lowering.kinds() {
                assert!(
                    RULES.contains(&format!("\n         ({kind} ")),
                    "{lowering:?} lowers {kind}, which no rewrite in rules.egg produces"
                );
            }
        }
    }

    #[test]
    fn conv_fusions_follow_sites_under_the_pointwise_override() {
        let s = sites(&[
            ("FusedParamActConv1d", &["blk0.up.w"]),
            ("FusedParamActConv1dResidual", &["blk0.ru0.pw.w"]),
            ("FusedConv1dAct", &["hift.pre.w"]),
        ]);
        let f = ConvFusions(Some(&s));
        assert!(f.input_act("w.blk0.up.w", 16, false));
        assert!(!f.input_act("w.blk0.ru0.pw.w", 1, true), "pointwise input activation stays a pass");
        assert!(f.residual("w.blk0.ru0.pw.w", false));
        assert!(!f.residual("w.blk0.up.w", true));
        assert_eq!(
            f.between("w.hift.pre.w", "w.hift.up0.w", 16, ActAt::ConsumerInput),
            ActAt::ProducerOutput
        );
        assert_eq!(f.between("w.a.w", "w.b.w", 3, ActAt::ConsumerInput), ActAt::Separate);
        let hand = ConvFusions(None);
        assert!(hand.residual("w.a.w", true) && !hand.output_act("w.a.w", false));
        assert_eq!(hand.between("w.a.w", "w.b.w", 3, ActAt::ConsumerInput), ActAt::ConsumerInput);
    }

    #[test]
    fn add_norm_lowers_residual_and_residual3_sites_only() {
        let s = sites(&[
            (
                "FusedResidualNorm",
                &["model.layers.0.post_attention_layernorm.weight"],
            ),
            (
                "FusedResidual3Norm",
                &["model.layers.4.input_layernorm.weight"],
            ),
            (
                "FusedNormResidualNorm",
                &["model.layers.0.pre_feedforward_layernorm.weight"],
            ),
            (
                "FusedNormLinear",
                &["model.layers.0.input_layernorm.weight"],
            ),
        ]);
        assert!(lowered(
            &s,
            Lowering::AddNorm,
            "model.layers.0.post_attention_layernorm.weight"
        ));
        assert!(lowered(
            &s,
            Lowering::AddNorm,
            "model.layers.4.input_layernorm.weight"
        ));
        assert!(!lowered(
            &s,
            Lowering::AddNorm,
            "model.layers.0.pre_feedforward_layernorm.weight"
        ));
        assert!(!lowered(
            &s,
            Lowering::AddNorm,
            "model.layers.0.input_layernorm.weight"
        ));
        assert!(!lowered(
            &s,
            Lowering::AddNorm,
            "model.layers.1.post_attention_layernorm.weight"
        ));
    }

    #[test]
    fn norm_residual_norm_lowers_both_sandwich_forms_only() {
        let s = sites(&[
            (
                "FusedNormResidualNorm",
                &["m.layers.0.pre_feedforward_layernorm.weight"],
            ),
            (
                "FusedNormResidualScaleNorm",
                &["m.layers.1.input_layernorm.weight"],
            ),
            ("FusedResidualNorm", &["m.norm.weight"]),
        ]);
        assert!(lowered(
            &s,
            Lowering::NormResidualNorm,
            "m.layers.0.pre_feedforward_layernorm.weight"
        ));
        assert!(lowered(
            &s,
            Lowering::NormResidualNorm,
            "m.layers.1.input_layernorm.weight"
        ));
        assert!(!lowered(&s, Lowering::NormResidualNorm, "m.norm.weight"));
        assert!(!lowered(
            &s,
            Lowering::NormResidualNorm,
            "m.layers.0.pre_feedforward_layernorm"
        ));
    }
}
