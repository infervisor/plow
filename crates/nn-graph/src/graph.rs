//! The operator graph: a `Vec`-indexed DAG of tensors and nodes.
//!
//! This is the Stage-1 frontend IR. It is deliberately a plain indexed graph
//! rather than `petgraph`: it is small (one transformer layer, replicated),
//! single-output per node, and built once then consumed by shape inference and
//! later lowering. `petgraph` is reserved for the tile dependency graph.

use crate::dim::SymbolTable;
use crate::dtype::DType;
use crate::op::Op;
use crate::shape::Shape;
use std::collections::BTreeSet;

/// Index of a tensor (graph value).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TensorId(pub u32);

/// Index of a node (operation).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(pub u32);

/// Where a tensor comes from.
#[derive(Clone, Debug)]
pub enum Origin {
    /// External graph input (activations: token ids, pixel values).
    Input,
    /// A model weight / parameter loaded from the checkpoint.
    Weight,
    /// Produced by a node.
    Node(NodeId),
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub name: Option<String>,
    pub dtype: DType,
    /// `None` until shape inference fills it (for `Node`-origin tensors).
    pub shape: Option<Shape>,
    pub origin: Origin,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub op: Op,
    pub inputs: Vec<TensorId>,
    pub output: TensorId,
    /// Index into [`Graph::blocks`] of the layer/block this node belongs to
    /// (e.g. one transformer layer). `None` for graph-level nodes outside any
    /// block (embedding, final norm, lm_head).
    pub block: Option<u32>,
}

#[derive(Debug)]
pub struct Graph {
    pub syms: SymbolTable,
    pub tensors: Vec<TensorInfo>,
    pub nodes: Vec<Node>,
    /// Labels of the structural blocks (transformer layers, encoder blocks),
    /// in build order. Nodes reference these by index via [`Node::block`].
    pub blocks: Vec<String>,
    /// Graph-level inputs and outputs, in declaration order.
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    /// Storage metadata needed to dequantize blockwise FP8 weights. These are
    /// checkpoint/load bindings, not extra logical operands of `Op::Linear`.
    pub fp8_scale_bindings: Vec<Fp8ScaleBinding>,
    /// Exhaustive per-layer routed-expert checkpoint bindings. The compute DAG
    /// carries one data-dependent dispatch op per layer; this manifest keeps
    /// every expert weight and scale loadable without cloning 3*E GEMMs into
    /// the graph.
    pub expert_bindings: Vec<ExpertLayerBinding>,
}

impl Graph {
    pub fn tensor(&self, id: TensorId) -> &TensorInfo {
        &self.tensors[id.0 as usize]
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Count nodes whose op matches a predicate (handy for tests/inspection).
    pub fn count_ops(&self, pred: impl Fn(&Op) -> bool) -> usize {
        self.nodes.iter().filter(|n| pred(&n.op)).count()
    }

    /// The weight (parameter) inputs consumed by a node, in input order. The
    /// loader uses this to associate each op with the checkpoint tensors it
    /// needs at later (weight-parsing) stages.
    pub fn op_weights(&self, node: NodeId) -> Vec<&TensorInfo> {
        self.node(node)
            .inputs
            .iter()
            .map(|id| self.tensor(*id))
            .filter(|t| matches!(t.origin, Origin::Weight))
            .collect()
    }

    /// Flat manifest of every weight the graph requires: which op owns it, its
    /// name, dtype, and (inferred) shape. This is the contract a weight loader
    /// parses against, and what [`crate::viz`] annotates op nodes with.
    pub fn weight_manifest(&self) -> Vec<WeightSpec<'_>> {
        let mut out = Vec::new();
        for (ni, node) in self.nodes.iter().enumerate() {
            for id in &node.inputs {
                let t = self.tensor(*id);
                if let Origin::Weight = t.origin {
                    out.push(WeightSpec {
                        node: NodeId(ni as u32),
                        op: node.op.name(),
                        name: t.name.as_deref().unwrap_or("<unnamed>"),
                        dtype: t.dtype,
                        shape: t.shape.as_ref(),
                    });
                }
            }
        }
        out
    }

    /// Every tensor that must be present in the checkpoint, including
    /// quantization metadata that is loaded alongside an operator's weight but
    /// is not a logical graph operand.
    pub fn checkpoint_manifest(&self) -> Vec<CheckpointWeightSpec<'_>> {
        let mut seen = BTreeSet::new();
        self.tensors
            .iter()
            .filter(|t| matches!(t.origin, Origin::Weight))
            .filter_map(|t| {
                let name = t.name.as_deref()?;
                seen.insert(name).then_some(CheckpointWeightSpec {
                    name,
                    dtype: t.dtype,
                    shape: t.shape.as_ref(),
                })
            })
            .collect()
    }

    /// Exact logical bytes required by the compiled checkpoint manifest.
    /// Names are deduplicated by [`Self::checkpoint_manifest`], so tied weights
    /// are charged once while block-FP8 scale grids remain explicit F32 data.
    pub fn checkpoint_storage_bytes(&self) -> Option<u64> {
        self.checkpoint_manifest()
            .into_iter()
            .try_fold(0u64, |sum, weight| {
                let elements = weight
                    .shape?
                    .dims()
                    .iter()
                    .map(|dim| dim.as_static())
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .try_fold(1u64, |count, dim| {
                        (dim >= 0).then(|| count.saturating_mul(dim as u64))
                    })?;
                Some(sum.saturating_add(weight.dtype.tile_bytes(elements)))
            })
    }
}

impl Graph {
    /// Label of block `idx` (e.g. `"layers.0"`).
    pub fn block_label(&self, idx: u32) -> Option<&str> {
        self.blocks.get(idx as usize).map(|s| s.as_str())
    }

    /// `(NodeId, &Node)` pairs belonging to block `idx`, in build order.
    pub fn block_nodes(&self, idx: u32) -> impl Iterator<Item = (NodeId, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(move |(_, n)| n.block == Some(idx))
            .map(|(i, n)| (NodeId(i as u32), n))
    }
}

/// One weight required by the graph, tied to the op that consumes it.
#[derive(Clone, Copy, Debug)]
pub struct WeightSpec<'a> {
    pub node: NodeId,
    pub op: &'static str,
    pub name: &'a str,
    pub dtype: DType,
    pub shape: Option<&'a Shape>,
}

#[derive(Clone, Copy, Debug)]
pub struct CheckpointWeightSpec<'a> {
    pub name: &'a str,
    pub dtype: DType,
    pub shape: Option<&'a Shape>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fp8ScaleBinding {
    pub weight: String,
    pub scale: String,
    pub block_shape: [i64; 2],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertProjectionBinding {
    pub weight: String,
    pub scale: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutedExpertBinding {
    pub gate: ExpertProjectionBinding,
    pub up: ExpertProjectionBinding,
    pub down: ExpertProjectionBinding,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertLayerBinding {
    pub block: u32,
    pub layer_label: String,
    pub num_experts: u32,
    pub top_k: u32,
    pub scoring_func: String,
    pub norm_topk: bool,
    pub route_scale: f32,
    pub n_group: u32,
    pub topk_group: u32,
    pub correction_bias: Option<String>,
    pub routed_experts: Vec<RoutedExpertBinding>,
}

/// Mutable builder for an operator graph. The architecture builders in
/// [`crate::models`] drive this through [`crate::Nn`]'s ergonomic helpers; the
/// IR stays minimal.
pub struct GraphBuilder {
    syms: SymbolTable,
    tensors: Vec<TensorInfo>,
    nodes: Vec<Node>,
    blocks: Vec<String>,
    current_block: Option<u32>,
    inputs: Vec<TensorId>,
    outputs: Vec<TensorId>,
    fp8_scale_bindings: Vec<Fp8ScaleBinding>,
    expert_bindings: Vec<ExpertLayerBinding>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        GraphBuilder {
            syms: SymbolTable::new(),
            tensors: Vec::new(),
            nodes: Vec::new(),
            blocks: Vec::new(),
            current_block: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            fp8_scale_bindings: Vec::new(),
            expert_bindings: Vec::new(),
        }
    }

    /// Open a structural block (a transformer layer / encoder block). Nodes
    /// emitted until [`GraphBuilder::end_block`] are tagged with it.
    pub fn begin_block(&mut self, label: &str) {
        let idx = self.blocks.len() as u32;
        self.blocks.push(label.to_string());
        self.current_block = Some(idx);
    }

    /// Close the current block; subsequent nodes are block-less again.
    pub fn end_block(&mut self) {
        self.current_block = None;
    }

    /// Intern a symbolic size variable (e.g. `"B"`, `"S"`).
    pub fn symbol(&mut self, name: &str) -> crate::dim::SymId {
        self.syms.intern(name)
    }

    fn push_tensor(&mut self, info: TensorInfo) -> TensorId {
        let id = TensorId(self.tensors.len() as u32);
        self.tensors.push(info);
        id
    }

    /// Declare an external input with a known shape.
    pub fn input(&mut self, name: &str, shape: Shape, dtype: DType) -> TensorId {
        let id = self.push_tensor(TensorInfo {
            name: Some(name.to_string()),
            dtype,
            shape: Some(shape),
            origin: Origin::Input,
        });
        self.inputs.push(id);
        id
    }

    /// Declare a model weight with a known shape.
    pub fn weight(&mut self, name: &str, shape: Shape, dtype: DType) -> TensorId {
        self.push_tensor(TensorInfo {
            name: Some(name.to_string()),
            dtype,
            shape: Some(shape),
            origin: Origin::Weight,
        })
    }

    pub fn fp8_scale_binding(
        &mut self,
        weight: &str,
        scale: &str,
        scale_shape: Shape,
        block_shape: [i64; 2],
    ) -> TensorId {
        let id = self.weight(scale, scale_shape, DType::F32);
        self.fp8_scale_bindings.push(Fp8ScaleBinding {
            weight: weight.to_string(),
            scale: scale.to_string(),
            block_shape,
        });
        id
    }

    pub fn expert_binding(&mut self, binding: ExpertLayerBinding) {
        self.expert_bindings.push(binding);
    }

    /// Add an operation node. The result tensor's shape is left `None` for the
    /// inference pass to fill. `out_dtype` is the declared output element type.
    pub fn op(&mut self, op: Op, inputs: Vec<TensorId>, out_dtype: DType) -> TensorId {
        let node_id = NodeId(self.nodes.len() as u32);
        let out = self.push_tensor(TensorInfo {
            name: None,
            dtype: out_dtype,
            shape: None,
            origin: Origin::Node(node_id),
        });
        self.nodes.push(Node {
            op,
            inputs,
            output: out,
            block: self.current_block,
        });
        out
    }

    /// Mark a tensor as a graph output.
    pub fn output(&mut self, id: TensorId) {
        self.outputs.push(id);
    }

    pub fn syms(&self) -> &SymbolTable {
        &self.syms
    }

    pub fn tensor(&self, id: TensorId) -> &TensorInfo {
        &self.tensors[id.0 as usize]
    }

    pub fn finish(self) -> Graph {
        Graph {
            syms: self.syms,
            tensors: self.tensors,
            nodes: self.nodes,
            blocks: self.blocks,
            inputs: self.inputs,
            outputs: self.outputs,
            fp8_scale_bindings: self.fp8_scale_bindings,
            expert_bindings: self.expert_bindings,
        }
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        GraphBuilder::new()
    }
}
