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
}

impl Lowering {
    fn kinds(self) -> &'static [&'static str] {
        match self {
            Lowering::AddNorm => &["FusedResidualNorm", "FusedResidual3Norm"],
            Lowering::NormResidualNorm => &["FusedNormResidualNorm", "FusedNormResidualScaleNorm"],
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
        for lowering in [Lowering::AddNorm, Lowering::NormResidualNorm] {
            for kind in lowering.kinds() {
                assert!(
                    RULES.contains(&format!("\n         ({kind} ")),
                    "{lowering:?} lowers {kind}, which no rewrite in rules.egg produces"
                );
            }
        }
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
