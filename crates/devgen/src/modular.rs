//! Modular building-block packet compiler support.
//!
//! Provides modular pipeline emission, tracking of egglog rewrite rules,
//! and integration with Lean formal verification certificates.

use std::collections::BTreeSet;

use packet::devbuild::{SectionData, SECT_METADATA};
use plow_asset::{
    ModularBlockKind, ModularBlockProg, ModularPhase, ModularPipelineManifest,
    MODULAR_MANIFEST_SECTION,
};

use crate::rewrite_lower::{self, Lowering};

/// Determine the egglog rewrite rules applicable to a given block kind.
pub fn block_egg_rules(
    kind: ModularBlockKind,
    has_qk_norm: bool,
    is_sandwich: bool,
    is_moe: bool,
) -> Vec<String> {
    let mut rules = BTreeSet::new();

    // Query active rewrite sites if installed, otherwise collect canonical rules
    let sites = rewrite_lower::active_sites();

    match kind {
        ModularBlockKind::Embedding => {
            // Norm / linear fusion if per-layer inputs or post-embedding norm exists
            rules.insert("rmsnorm-linear-fuse".to_string());
        }
        ModularBlockKind::DenseAttention => {
            // 1. Input RMSNorm fused into Q/K/V linear projections
            rules.insert("rmsnorm-linear-fuse".to_string());
            // 2. Q/K norm fused into RoPE
            if has_qk_norm {
                rules.insert("rmsnorm-rope-fuse".to_string());
                rules.insert("rmsnorm-rope-scale-fuse".to_string());
            }
            // 3. Post-attention residual seam
            if is_sandwich {
                rules.insert("norm-residual-rmsnorm-fuse".to_string());
            } else {
                rules.insert("residual-rmsnorm-fuse".to_string());
            }
        }
        ModularBlockKind::DenseFfn => {
            // 1. Post-attention RMSNorm fused into gate/up projections
            rules.insert("rmsnorm-linear-fuse".to_string());
            // 2. Activation(gate) * up fused into SwiGLU / GeGLU
            rules.insert("gated-mlp-fuse".to_string());
            // 3. Post-FFN residual seam
            if is_sandwich {
                rules.insert("norm-residual-rmsnorm-fuse".to_string());
            } else {
                rules.insert("residual-rmsnorm-fuse".to_string());
            }
        }
        ModularBlockKind::Moe => {
            // Router linear
            rules.insert("rmsnorm-linear-fuse".to_string());
            if is_moe {
                // 3-way residual combine + norm
                rules.insert("residual3-rmsnorm-fuse".to_string());
            }
        }
        ModularBlockKind::FinalNormVocab => {
            // Final RMSNorm fused into LM head linear projection
            rules.insert("rmsnorm-linear-fuse".to_string());
        }
    }

    if let Some(s) = sites {
        // If specific sites are active, filter or annotate according to lowered kinds
        for lowering in [
            Lowering::AddNorm,
            Lowering::NormResidualNorm,
            Lowering::NormLinear,
            Lowering::GatedMlp,
            Lowering::NormRope,
            Lowering::Residual3Norm,
        ] {
            for kind_name in lowering.kinds() {
                if s.contains_key(*kind_name) {
                    for r in lowering.egg_rules() {
                        rules.insert(r.to_string());
                    }
                }
            }
        }
    }

    rules.into_iter().collect()
}

/// Construct a `ModularBlockProg` entry for a compiled program.
pub fn create_modular_block_prog(
    kind: ModularBlockKind,
    phase: ModularPhase,
    program_idx: u32,
    width: u32,
    layer: Option<u32>,
    has_qk_norm: bool,
    is_sandwich: bool,
    is_moe: bool,
) -> ModularBlockProg {
    let name = match (phase, kind) {
        (ModularPhase::Prefill, ModularBlockKind::Embedding) => format!("embed_t{width}"),
        (ModularPhase::Prefill, ModularBlockKind::DenseAttention) => format!("attn_t{width}"),
        (ModularPhase::Prefill, ModularBlockKind::DenseFfn) => format!("ffn_t{width}"),
        (ModularPhase::Prefill, ModularBlockKind::Moe) => format!("moe_t{width}"),
        (ModularPhase::Prefill, ModularBlockKind::FinalNormVocab) => format!("vocab_t{width}"),
        (ModularPhase::Decode, ModularBlockKind::Embedding) => format!("embed_b{width}"),
        (ModularPhase::Decode, ModularBlockKind::DenseAttention) => format!("attn_b{width}"),
        (ModularPhase::Decode, ModularBlockKind::DenseFfn) => format!("ffn_b{width}"),
        (ModularPhase::Decode, ModularBlockKind::Moe) => format!("moe_b{width}"),
        (ModularPhase::Decode, ModularBlockKind::FinalNormVocab) => format!("vocab_b{width}"),
    };
    let egg_rules = block_egg_rules(kind, has_qk_norm, is_sandwich, is_moe);
    ModularBlockProg {
        name,
        kind,
        phase,
        width,
        layer,
        program_idx,
        egg_rules,
        lean_correctness: None,
        lean_performance: None,
    }
}

/// Build a `ModularPipelineManifest` aggregating all modular blocks and applied egg rules.
pub fn build_modular_manifest(
    num_layers: u32,
    blocks: Vec<ModularBlockProg>,
    prefill_rungs: &[u32],
    decode_rungs: &[u32],
) -> ModularPipelineManifest {
    let mut manifest = ModularPipelineManifest::new(num_layers);
    manifest.prefill_rungs = prefill_rungs.to_vec();
    manifest.decode_rungs = decode_rungs.to_vec();

    for block in blocks {
        manifest.add_block(block);
    }

    manifest
}

/// Create a packet `SectionData` holding the serialized `ModularPipelineManifest`.
pub fn modular_pipeline_section(manifest: &ModularPipelineManifest) -> SectionData {
    SectionData {
        kind: SECT_METADATA,
        name: MODULAR_MANIFEST_SECTION.to_string(),
        data: serde_json::to_vec_pretty(manifest)
            .expect("serialization of modular pipeline manifest failed"),
    }
}

/// Update sections with Lean certificates and summary from a `LeanReport`.
pub fn update_section_with_lean(
    sections: &mut Vec<SectionData>,
    lean: &crate::LeanReport,
) {
    if let Some(pos) = sections
        .iter()
        .position(|s| s.kind == SECT_METADATA && s.name == MODULAR_MANIFEST_SECTION)
    {
        if let Ok(mut manifest) = serde_json::from_slice::<ModularPipelineManifest>(&sections[pos].data) {
            manifest.lean_summary = lean.modular_summary.clone();
            for certified_block in &lean.modular_blocks {
                if let Some(target) = manifest
                    .blocks
                    .iter_mut()
                    .find(|p| p.program_idx == certified_block.program_idx)
                {
                    target.lean_correctness = certified_block.lean_correctness.clone();
                    target.lean_performance = certified_block.lean_performance.clone();
                }
            }
            sections[pos].data = serde_json::to_vec_pretty(&manifest)
                .expect("re-serialization of modular pipeline manifest failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_egg_rules_presence() {
        let attn_rules = block_egg_rules(ModularBlockKind::DenseAttention, true, true, false);
        assert!(attn_rules.contains(&"rmsnorm-linear-fuse".to_string()));
        assert!(attn_rules.contains(&"rmsnorm-rope-fuse".to_string()));
        assert!(attn_rules.contains(&"rmsnorm-rope-scale-fuse".to_string()));
        assert!(attn_rules.contains(&"norm-residual-rmsnorm-fuse".to_string()));

        let ffn_rules = block_egg_rules(ModularBlockKind::DenseFfn, false, false, false);
        assert!(ffn_rules.contains(&"rmsnorm-linear-fuse".to_string()));
        assert!(ffn_rules.contains(&"gated-mlp-fuse".to_string()));
        assert!(ffn_rules.contains(&"residual-rmsnorm-fuse".to_string()));

        let moe_rules = block_egg_rules(ModularBlockKind::Moe, false, false, true);
        assert!(moe_rules.contains(&"rmsnorm-linear-fuse".to_string()));
        assert!(moe_rules.contains(&"residual3-rmsnorm-fuse".to_string()));
    }

    #[test]
    fn test_modular_manifest_construction_and_section() {
        let b1 = create_modular_block_prog(
            ModularBlockKind::DenseAttention,
            ModularPhase::Prefill,
            0,
            128,
            None,
            true,
            true,
            false,
        );
        let b2 = create_modular_block_prog(
            ModularBlockKind::DenseFfn,
            ModularPhase::Prefill,
            1,
            128,
            None,
            false,
            true,
            false,
        );

        let manifest = build_modular_manifest(40, vec![b1, b2], &[128], &[1]);
        assert_eq!(manifest.blocks.len(), 2);
        assert!(manifest.egg_rules_applied.contains(&"rmsnorm-linear-fuse".to_string()));
        assert!(manifest.egg_rules_applied.contains(&"gated-mlp-fuse".to_string()));

        let sect = modular_pipeline_section(&manifest);
        assert_eq!(sect.name, MODULAR_MANIFEST_SECTION);
        assert_eq!(sect.kind, SECT_METADATA);

        let parsed: ModularPipelineManifest = serde_json::from_slice(&sect.data).unwrap();
        assert_eq!(parsed.num_layers, 40);
        assert_eq!(parsed.blocks.len(), 2);
    }
}
