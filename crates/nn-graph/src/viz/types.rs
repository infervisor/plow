//! Shared Rust data types for the network visualizer.
//!
//! Provides a unified schema for single graphs, multi-network pipelines,
//! pipeline stages, tensor dataflow, and architectural summary metrics.

use crate::dtype::DType;
use crate::graph::{NodeId, Origin, TensorId};
use serde::Serialize;
use std::collections::BTreeMap;

/// Op family, used for colour and the composition summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpCategory {
    /// Linear, matmul, embedding lookup.
    Linear,
    /// Attention, RoPE, softmax.
    Attention,
    Convolution,
    Normalization,
    /// Activations, residual adds, scaling, gates.
    Elementwise,
    /// MoE router/experts, sparse-attention indexer.
    Routing,
    /// Reshape, transpose, slice, concat, broadcast, reduce.
    Layout,
}

impl OpCategory {
    pub const ALL: [OpCategory; 7] = [
        OpCategory::Linear,
        OpCategory::Attention,
        OpCategory::Convolution,
        OpCategory::Normalization,
        OpCategory::Elementwise,
        OpCategory::Routing,
        OpCategory::Layout,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            OpCategory::Linear => "Linear",
            OpCategory::Attention => "Attention",
            OpCategory::Convolution => "Convolution",
            OpCategory::Normalization => "Normalization",
            OpCategory::Elementwise => "Elementwise",
            OpCategory::Routing => "Routing",
            OpCategory::Layout => "Layout",
        }
    }

    /// Accent colour as `(dark theme, light theme)`, in the infervisor OKLCH palette.
    pub fn colors(&self) -> (&'static str, &'static str) {
        match self {
            OpCategory::Linear => ("oklch(69% 0.12 245)", "oklch(52% 0.13 247)"),
            OpCategory::Attention => ("oklch(81% 0.085 237)", "oklch(60% 0.12 230)"),
            OpCategory::Convolution => ("oklch(72% 0.1 290)", "oklch(52% 0.13 290)"),
            OpCategory::Normalization => ("oklch(62% 0.012 250)", "oklch(58% 0.012 250)"),
            OpCategory::Elementwise => ("oklch(76% 0.11 165)", "oklch(52% 0.11 165)"),
            OpCategory::Routing => ("oklch(64% 0.11 70)", "oklch(56% 0.12 70)"),
            OpCategory::Layout => ("oklch(46% 0.012 250)", "oklch(76% 0.012 250)"),
        }
    }
}

/// Primary modality of the model or pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelModality {
    Llm,
    Asr,
    Vad,
    Tts,
    Multimodal,
    Vision,
    AudioCodec,
    /// Text-to-image diffusion (DiT, VAE).
    Image,
}

impl ModelModality {
    pub fn badge(&self) -> &'static str {
        match self {
            ModelModality::Llm => "LLM",
            ModelModality::Asr => "ASR",
            ModelModality::Vad => "VAD",
            ModelModality::Tts => "TTS",
            ModelModality::Multimodal => "Multimodal",
            ModelModality::Vision => "Vision",
            ModelModality::AudioCodec => "Codec",
            ModelModality::Image => "Image",
        }
    }
}

/// A contiguous step of a network (e.g. audio encoder, text decoder). Every op belongs to exactly
/// one stage; params and boundary shapes are computed from the graph.
#[derive(Debug, Clone, Serialize)]
pub struct VizStage {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Checkpoint module prefixes of the stage's weights, numeric indices folded to `*`.
    pub modules: Vec<String>,
    pub block_count: u32,
    pub param_count: u64,
    pub input_shapes: Vec<String>,
    pub output_shapes: Vec<String>,
}

/// Serialized tensor information.
#[derive(Debug, Clone, Serialize)]
pub struct VizTensor {
    pub id: TensorId,
    pub name: String,
    pub dtype: DType,
    pub shape: String,
    pub elements: Option<u64>,
    pub bytes: Option<u64>,
    pub origin: Origin,
    /// Checkpoint tensors this weight stands for. MoE graphs build one representative routed
    /// expert (`experts.0`) for all `num_experts`; params and bytes scale by this.
    #[serde(skip_serializing_if = "is_one")]
    pub copies: u32,
}

fn is_one(n: &u32) -> bool {
    *n == 1
}

/// Serialized operation node.
#[derive(Debug, Clone, Serialize)]
pub struct VizNode {
    pub id: NodeId,
    pub op: &'static str,
    pub category: OpCategory,
    pub detail: serde_json::Value,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    pub block: Option<String>,
    pub stage: Option<String>,
    pub param_count: u64,
    pub weight_bytes: u64,
}

/// A block / layer grouping (e.g. transformer layer, conformer block, upsampling stage).
#[derive(Debug, Clone, Serialize)]
pub struct VizBlock {
    pub id: String,
    pub stage: Option<String>,
    pub node_ids: Vec<NodeId>,
    pub param_count: u64,
    pub weight_bytes: u64,
}

/// Totals for one network.
#[derive(Debug, Clone, Serialize)]
pub struct VizSummary {
    pub total_params: u64,
    pub total_weight_bytes: u64,
    pub num_blocks: u32,
    pub category_counts: BTreeMap<OpCategory, usize>,
    pub total_nodes: usize,
    pub total_tensors: usize,
}

/// One graph of the model (the full model, or a part that also runs on its own).
#[derive(Debug, Clone, Serialize)]
pub struct VizNetwork {
    pub kind: &'static str,
    pub id: String,
    pub title: String,
    pub description: String,
    pub stages: Vec<VizStage>,
    pub graph: VizGraph,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint: Option<VizCheckpoint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct CheckpointTensor {
    pub dtype: DType,
    pub shape: Vec<i64>,
    pub bytes: u64,
}

/// Checks tensor storage and shapes, not the graph's numerical semantics.
#[derive(Debug, Clone, Serialize)]
pub struct VizCheckpoint {
    pub tensor_count: usize,
    pub stored_bytes: u64,
    pub matched_tensors: usize,
    pub unrepresented_tensors: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct VizProvenance {
    pub revision: Option<String>,
    pub source_kind: &'static str,
    pub source_sha256: String,
    pub generator_version: String,
    pub bindings: BTreeMap<String, i64>,
}

/// Op/tensor graph of one network.
#[derive(Debug, Clone, Serialize)]
pub struct VizGraph {
    pub tensors: Vec<VizTensor>,
    pub nodes: Vec<VizNode>,
    pub blocks: Vec<VizBlock>,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub summary: VizSummary,
}

/// Everything the viewer shows for one model. `networks[0]` is the full model.
#[derive(Debug, Clone, Serialize)]
pub struct VizModelBundle {
    pub schema: &'static str,
    pub title: String,
    pub source: String,
    pub modality: ModelModality,
    pub default_network: String,
    pub networks: Vec<VizNetwork>,
    /// What the viewer cannot show for this model and why.
    pub notes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provenance: Option<VizProvenance>,
}

impl VizModelBundle {
    pub fn default_network(&self) -> &VizNetwork {
        self.networks
            .iter()
            .find(|n| n.id == self.default_network)
            .unwrap_or(&self.networks[0])
    }

    pub fn default_graph(&self) -> &VizGraph {
        &self.default_network().graph
    }
}
