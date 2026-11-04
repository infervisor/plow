//! Shared Rust data types for the network visualizer.
//!
//! Provides a unified schema for single graphs, multi-network pipelines,
//! pipeline stages, tensor dataflow, and architectural summary metrics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Op classification category for styling, grouping, and metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpCategory {
    Compute,       // Linear, MatMul, GEMM
    Attention,     // Multi-head Attention, GQA, Linear Attention, DSA Attention
    Convolution,   // Conv1d, Conv2d, Conv3d, ConvTranspose1d
    Normalization, // RMSNorm, LayerNorm, GroupNorm
    Activation,    // SiLU, GeLU, Snake, Tanh, ReLU, etc.
    Routing,       // MoE Router, MoE Experts, DSA Indexer
    Structural,    // Reshape, Transpose, Slice, Concat, Broadcast
    AudioCodec,    // RVQ Codebook, Pitch, Vocoder
    Other,
}

impl OpCategory {
    pub fn label(&self) -> &'static str {
        match self {
            OpCategory::Compute => "Compute",
            OpCategory::Attention => "Attention",
            OpCategory::Convolution => "Convolution",
            OpCategory::Normalization => "Normalization",
            OpCategory::Activation => "Activation",
            OpCategory::Routing => "MoE / Routing",
            OpCategory::Structural => "Structural",
            OpCategory::AudioCodec => "Audio / Codec",
            OpCategory::Other => "Other",
        }
    }

    pub fn color_palette(&self) -> (&'static str, &'static str) {
        // (background, border)
        match self {
            OpCategory::Compute => ("#0e1820", "#3d6e9c"),
            OpCategory::Attention => ("#132c47", "#8fcaf4"),
            OpCategory::Convolution => ("#10233a", "#5a9fd4"),
            OpCategory::Normalization => ("#14171c", "#8b929c"),
            OpCategory::Activation => ("#0f1d17", "#6cc0a0"),
            OpCategory::Routing => ("#231708", "#b8813a"),
            OpCategory::Structural => ("#101318", "#3a424c"),
            OpCategory::AudioCodec => ("#1e1424", "#b07a9a"),
            OpCategory::Other => ("#101318", "#3a424c"),
        }
    }
}

/// Primary modality of the model or pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelModality {
    Llm,
    Asr,
    Tts,
    Multimodal,
    Vision,
    AudioCodec,
}

impl ModelModality {
    pub fn badge(&self) -> &'static str {
        match self {
            ModelModality::Llm => "LLM",
            ModelModality::Asr => "ASR",
            ModelModality::Tts => "TTS",
            ModelModality::Multimodal => "Multimodal",
            ModelModality::Vision => "Vision",
            ModelModality::AudioCodec => "Codec",
        }
    }
}

/// A stage in an end-to-end multi-stage pipeline (e.g. ASR: Mel -> Audio Enc -> Proj -> Thinker LM;
/// TTS: Text Enc -> CFM -> Vocoder).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStage {
    pub id: String,
    pub name: String,
    pub description: String,
    pub modality: String,
    pub layer_count: Option<u32>,
    pub param_count: u64,
    pub input_shapes: Vec<String>,
    pub output_shapes: Vec<String>,
    pub subnetwork_name: Option<String>,
}

/// Where a tensor originates in the dataflow DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum VizOrigin {
    Input,
    Weight,
    Activation { node: u32 },
}

/// Serialized tensor information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizTensor {
    pub id: u32,
    pub name: String,
    pub dtype: String,
    pub shape: String,
    pub elements: Option<u64>,
    pub bytes: Option<u64>,
    pub origin: VizOrigin,
}

/// Serialized operation node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizNode {
    pub id: u32,
    pub op: String,
    pub category: OpCategory,
    pub detail: serde_json::Value,
    pub inputs: Vec<u32>,
    pub output: u32,
    pub block: Option<String>,
    pub stage: Option<String>,
    pub param_count: u64,
    pub weight_bytes: u64,
}

/// A block / layer grouping (e.g. transformer layer, conformer block, upsampling stage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizBlock {
    pub id: String,
    pub label: String,
    pub stage: Option<String>,
    pub node_ids: Vec<u32>,
    pub param_count: u64,
    pub weight_bytes: u64,
}

/// Summary metrics and architectural metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizSummary {
    pub title: String,
    pub model_type: String,
    pub modality: ModelModality,
    pub total_params: u64,
    pub total_weight_bytes: u64,
    pub num_layers: u32,
    pub hidden_size: Option<i64>,
    pub num_heads: Option<u32>,
    pub num_kv_heads: Option<u32>,
    pub vocab_size: Option<i64>,
    pub category_counts: HashMap<String, usize>,
    pub total_nodes: usize,
    pub total_tensors: usize,
}

/// A sub-network within a multi-network architecture (e.g. S3Gen's vocoder or Qwen3-ASR's audio encoder).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizSubNetwork {
    pub name: String,
    pub title: String,
    pub description: String,
    pub graph: VizGraph,
}

/// Complete graph visualization structure for a single network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizGraph {
    pub tensors: Vec<VizTensor>,
    pub nodes: Vec<VizNode>,
    pub blocks: Vec<VizBlock>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub summary: VizSummary,
}

/// Top-level model bundle holding one or more sub-networks, pipeline stages, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VizModelBundle {
    pub title: String,
    pub source: String,
    pub modality: ModelModality,
    pub default_network: String,
    pub stages: Vec<PipelineStage>,
    pub networks: HashMap<String, VizSubNetwork>,
    pub summary: VizSummary,
}

impl VizModelBundle {
    pub fn default_graph(&self) -> &VizGraph {
        self.networks
            .get(&self.default_network)
            .map(|sub| &sub.graph)
            .or_else(|| self.networks.values().next().map(|sub| &sub.graph))
            .expect("VizModelBundle must contain at least one network")
    }
}
