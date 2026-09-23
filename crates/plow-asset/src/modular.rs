use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

/// The architectural building blocks of a transformer model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModularBlockKind {
    /// Token embedding (plus optional per-layer embeddings / PLE).
    Embedding,
    /// Dense attention block: Input RMSNorm + Q/K/V projections + RoPE/HeadNorm + Flash attention + O projection + residual.
    DenseAttention,
    /// Dense FFN block: Post-attention RMSNorm + Gate/Up GEMM/GEMV + GeGLU/SwiGLU + Down GEMM/GEMV + residual.
    DenseFfn,
    /// Routed or shared MoE block: Post-attention RMSNorm + router + expert GEMMs + combine + residual.
    Moe,
    /// Final RMSNorm + lm_head projection + optional argmax/sampling.
    FinalNormVocab,
}

impl ModularBlockKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::DenseAttention => "dense_attention",
            Self::DenseFfn => "dense_ffn",
            Self::Moe => "moe",
            Self::FinalNormVocab => "final_norm_vocab",
        }
    }
}

/// Execution phase for a modular block: prefill or decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModularPhase {
    Prefill,
    Decode,
}

impl ModularPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

/// Scoped abstract checks, not a floating-point or machine-code implementation proof.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModularLeanCorrectness {
    /// Checkpoint D: GQ stream order is topological over counter edges (deadlock-free).
    pub ordering_verified: bool,
    /// Checkpoint G: Staged LDS arena bounds verified (all staged GEMV/GEMM fits in arena).
    pub lds_fit_verified: bool,
    /// Checkpoint A: Rewrite-name catalog checked against Lean syntax theorems.
    #[serde(alias = "rewrite_soundness_verified")]
    pub rewrite_name_catalog_checked: bool,
    /// Optional failure or skip diagnostic reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Conditional structural estimate; never an empirical performance qualification.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModularLeanPerformance {
    /// Calibrated critical path cycles; zero when inputs are unavailable.
    pub critical_path_cycles: u64,
    /// Memory/HBM bandwidth lower bound in cycles.
    pub bw_bound_cycles: u64,
    /// Conditional cycle lower bound; zero when inputs are unavailable.
    pub lower_bound_cycles: u64,
    /// Conditional latency bound; zero when inputs are unavailable.
    pub lower_bound_us: f64,
    /// Hardware binding bottleneck ("bandwidth" vs "latency").
    pub binding_constraint: String,
    /// Referenced allocation capacity; not a measurement of streamed/HBM bytes.
    pub touched_bytes: u64,
    /// Selected-op work and explicit missing inputs, kept separate from physical traffic.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_inputs: Option<serde_json::Value>,
    /// Whether this lower bound carries a formal Lean certificate.
    #[serde(alias = "certified")]
    pub structural_bound_certified: bool,
}

/// Summary of Lean formal verification and performance certificates across all modular blocks.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ModularLeanSummary {
    /// True if all modular blocks passed Checkpoint D (ordering), Checkpoint G (LDS fit), and Checkpoint A (rewrites).
    #[serde(alias = "all_correctness_verified")]
    pub all_abstract_checks_passed: bool,
    /// True if all modular blocks have certified Lean performance bounds.
    #[serde(alias = "all_performance_certified")]
    pub all_structural_bounds_certified: bool,
    /// Sum of critical path cycles across all primary blocks in a forward pass.
    pub total_critical_path_cycles: u64,
    /// Sum of certified microsecond lower bounds across all primary blocks in a forward pass.
    pub total_lower_bound_us: f64,
    /// Number of modular blocks verified.
    pub blocks_verified: usize,
    /// Total number of modular blocks in the manifest.
    pub total_blocks: usize,
}

/// Descriptor of a single modular block program inside the packet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModularBlockProg {
    /// Symbolic name (e.g. "embed_pf", "attn_t128", "ffn_t128", "attn_b1_decode", "ffn_b1_decode", "vocab_pf").
    pub name: String,
    /// Kind of architectural block.
    pub kind: ModularBlockKind,
    /// Execution phase (prefill or decode).
    pub phase: ModularPhase,
    /// Token count T for prefill, or batch size B for decode.
    pub width: u32,
    /// If specialized to a specific layer or variant (e.g. sliding vs full attention), the layer index.
    /// None if universally reusable across all layers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,
    /// Index in the DevBlob / Model program table.
    pub program_idx: u32,
    /// Egglog equality saturation rules applied within this modular block.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub egg_rules: Vec<String>,
    /// Lean correctness verification certificates (ordering, LDS fit, rewrite soundness).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lean_correctness: Option<ModularLeanCorrectness>,
    /// Lean performance verification certificates (critical path, memory bandwidth bound, cycle floor).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lean_performance: Option<ModularLeanPerformance>,
}

/// Full modular pipeline manifest describing how modular blocks are chained to form full forward passes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModularPipelineManifest {
    pub version: u32,
    pub num_layers: u32,
    pub blocks: Vec<ModularBlockProg>,
    /// Prefill sequence of rungs available (e.g. [64, 128, 192, 256, 384, 512, ...]).
    pub prefill_rungs: Vec<u32>,
    /// Decode batch rungs available (e.g. [1, 2, 4, 8, 16]).
    pub decode_rungs: Vec<u32>,
    /// Whether single-block microbenchmarks / isolated execution are supported.
    pub isolated_execution: bool,
    /// Mapping of (phase, kind, width) -> program_idx for fast runtime dispatch.
    #[serde(default)]
    pub dispatch_table: BTreeMap<String, u32>,
    /// All egglog equality saturation rules applied across the modular pipeline.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub egg_rules_applied: Vec<String>,
    /// Overall Lean verification summary across all modular blocks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lean_summary: Option<ModularLeanSummary>,
}

pub const MODULAR_MANIFEST_SECTION: &str = "modular_pipeline.json";
pub const MODULAR_MANIFEST_VERSION: u32 = 1;

impl ModularPipelineManifest {
    pub fn new(num_layers: u32) -> Self {
        Self {
            version: MODULAR_MANIFEST_VERSION,
            num_layers,
            blocks: Vec::new(),
            prefill_rungs: Vec::new(),
            decode_rungs: Vec::new(),
            isolated_execution: true,
            dispatch_table: BTreeMap::new(),
            egg_rules_applied: Vec::new(),
            lean_summary: None,
        }
    }

    pub fn add_block(&mut self, block: ModularBlockProg) {
        let key = Self::dispatch_key(block.phase, block.kind, block.width, block.layer);
        self.dispatch_table.insert(key, block.program_idx);
        for rule in &block.egg_rules {
            if !self.egg_rules_applied.contains(rule) {
                self.egg_rules_applied.push(rule.clone());
            }
        }
        self.blocks.push(block);
    }

    pub fn dispatch_key(
        phase: ModularPhase,
        kind: ModularBlockKind,
        width: u32,
        layer: Option<u32>,
    ) -> String {
        match layer {
            Some(l) => format!("{}:{}:{}:l{}", phase.as_str(), kind.as_str(), width, l),
            None => format!("{}:{}:{}", phase.as_str(), kind.as_str(), width),
        }
    }

    pub fn find_block(
        &self,
        phase: ModularPhase,
        kind: ModularBlockKind,
        width: u32,
        layer: Option<u32>,
    ) -> Option<u32> {
        // First try layer-specific key if layer is specified
        if let Some(l) = layer {
            let key = Self::dispatch_key(phase, kind, width, Some(l));
            if let Some(&idx) = self.dispatch_table.get(&key) {
                return Some(idx);
            }
        }
        // Fall back to layer-agnostic key
        let key = Self::dispatch_key(phase, kind, width, None);
        self.dispatch_table.get(&key).copied()
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;

    #[test]
    fn abstract_and_structural_checks_do_not_serialize_as_full_qualification() {
        let correctness = ModularLeanCorrectness {
            ordering_verified: true, lds_fit_verified: true,
            rewrite_name_catalog_checked: true, reason: None,
        };
        let value = serde_json::to_value(correctness).unwrap();
        assert_eq!(value["rewrite_name_catalog_checked"], true);
        assert!(value.get("rewrite_soundness_verified").is_none());
        let summary = ModularLeanSummary { all_abstract_checks_passed: true,
            all_structural_bounds_certified: true, ..Default::default() };
        let value = serde_json::to_value(summary).unwrap();
        assert!(value.get("all_correctness_verified").is_none());
        assert!(value.get("all_performance_certified").is_none());
        let value = serde_json::to_value(ModularLeanPerformance::default()).unwrap();
        assert!(value.get("certified").is_none());
        assert_eq!(value["structural_bound_certified"], false);
    }
}
