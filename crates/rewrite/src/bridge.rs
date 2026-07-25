//! Bridge: an `nn_graph` block → a tiling [`LayerPlan`].
//!
//! This is the join between the two IRs (the design's audit §4.3). `nn_graph`
//! carries the *logical* dataflow and per-op access semantics (the typed [`Op`]
//! enum); tiling needs the *physical* problem shape of each op — concrete
//! `m/n/k`, row counts, attention extents — bound to a shape bucket. This walks
//! one [`Graph`] block in program order and classifies each node's [`Op`] into a
//! [`tilegraph::OpKind`] with those shapes, naming operands so [`assemble`]'s
//! producer→consumer matching links the block up.
//!
//! Shapes must be fully inferred and static (the sequence/batch symbols bound to
//! a bucket) — a dynamic dim is an error here, not at tiling time. The op
//! vocabulary handled is the transformer-block set (norm / linear / rope / act /
//! elementwise / attention / layout); other ops are rejected.
//!
//! [`assemble`]: crate::assemble

use crate::extract::{Arg, FNode, FusedGraph};
use crate::tilegraph::{LayerPlan, LayoutSpec, OpKind, OpSpec, LAYOUT_RANK};
use costmodel::{AttnShape, GemmShape, RowShape};
use nn_graph::{Graph, Node, NodeId, Op, TensorId};
use std::collections::HashMap;

#[derive(thiserror::Error, Debug)]
pub enum BridgeError {
    #[error("node {node} ({op}): shape not inferred — run infer_shapes first")]
    MissingShape { node: usize, op: &'static str },
    #[error("node {node} ({op}): dynamic dim `{dim}` — bind the shape bucket first")]
    Dynamic {
        node: usize,
        op: &'static str,
        dim: String,
    },
    #[error("node {node}: op `{op}` is not supported in a transformer block")]
    Unsupported { node: usize, op: &'static str },
    #[error("fused node {node}: op `{op}` is not supported in a transformer block")]
    UnsupportedFused { node: usize, op: String },
    #[error("fused leaf `{name}` has no static-shaped tensor in the source graph")]
    Unbound { name: String },
    #[error("fused node {node} ({op}): could not read shape token `{token}`")]
    BadToken {
        node: usize,
        op: String,
        token: String,
    },
    #[error("fused node {node} ({op}): axis {axis} out of range for rank {rank}")]
    Axis {
        node: usize,
        op: String,
        axis: i64,
        rank: i64,
    },
    #[error("fused node {node} ({op}): unexpected rank {rank} ({expected})")]
    Rank {
        node: usize,
        op: String,
        rank: usize,
        expected: &'static str,
    },
}

/// Lower one block (`block` index into [`Graph::blocks`]) of a shape-inferred
/// graph to a [`LayerPlan`], in program order.
pub fn plan_from_block(g: &Graph, block: u32) -> Result<LayerPlan, BridgeError> {
    let mut ops = Vec::new();
    for (id, node) in g.block_nodes(block) {
        let kind = op_kind(g, id, node)?;
        let (weight_dtype, compute_dtype) = node_dtypes(g, node);
        ops.push(OpSpec {
            name: format!("{}{}", node.op.name(), id.0),
            inputs: node.inputs.iter().map(|t| tname(g, *t)).collect(),
            output: tname(g, node.output),
            kind,
            weight_dtype,
            compute_dtype,
        });
    }
    Ok(LayerPlan { ops })
}

/// Lower **every** block of a shape-inferred graph into one [`LayerPlan`], in
/// build order (block 0's ops first, then block 1's, ...). Blocks share
/// producer→consumer tensor names via the nn-graph naming, so `assemble` will
/// chain fine-grained tile-dependencies across every block boundary — the
/// consumer's first-op tiles unblock as their specific producer tiles finish,
/// with no whole-tensor barrier between blocks.
///
/// This is the entry point that unlocks **cross-block tile pipelining** and
/// supports **heterogeneous block types** (Gemma 4's mixed sliding/full
/// attention, DeepSeek's dense-then-MoE, DiT stages) — each block lives under
/// its own `block` index, but the plan is agnostic to the block type.
/// Implicit barriers only remain between distinct compiled *programs* (e.g.,
/// the voice pipeline `ASR → LLM → TTS`, or successive DiT denoising steps).
pub fn plan_from_all_blocks(g: &Graph) -> Result<LayerPlan, BridgeError> {
    let mut ops = Vec::new();
    for block in 0..(g.blocks.len() as u32) {
        for (id, node) in g.block_nodes(block) {
            let kind = op_kind(g, id, node)?;
            let (weight_dtype, compute_dtype) = node_dtypes(g, node);
            ops.push(OpSpec {
                // Suffix the block index into the op name so cross-block
                // counter ids and packet debug output are decipherable.
                name: format!("{}{}_L{}", node.op.name(), id.0, block),
                inputs: node.inputs.iter().map(|t| tname(g, *t)).collect(),
                output: tname(g, node.output),
                kind,
                weight_dtype,
                compute_dtype,
            });
        }
    }
    Ok(LayerPlan { ops })
}

/// A stable string name for a tensor: its declared name (inputs/weights) or a
/// synthesized `t{id}` for unnamed node outputs. `assemble` couples a producer
/// to a consumer by string equality of output/input names, so this just has to
/// be a stable, collision-free function of the tensor id.
fn tname(g: &Graph, id: TensorId) -> String {
    g.tensor(id)
        .name
        .clone()
        .unwrap_or_else(|| format!("t{}", id.0))
}

/// Extract (weight_dtype, compute_dtype) from a node's inputs.
///
/// Scans the node's input tensors for any with `Origin::Weight`; the first
/// weight found provides `weight_dtype`. The `compute_dtype` is the weight's
/// dequant target (BF16 for block-quant, same as weight for standard types).
/// If no weight input is found, both default to BF16.
fn node_dtypes(g: &Graph, node: &nn_graph::Node) -> (nn_graph::DType, nn_graph::DType) {
    let weight_dt = node
        .inputs
        .iter()
        .map(|id| g.tensor(*id))
        .find(|t| matches!(t.origin, nn_graph::Origin::Weight))
        .map(|t| t.dtype)
        .unwrap_or(nn_graph::DType::BF16);
    let compute_dt = weight_dt.dequant_target();
    (weight_dt, compute_dt)
}

/// Static dims of a tensor, or an error if missing/dynamic.
fn static_dims(
    g: &Graph,
    id: TensorId,
    node: usize,
    op: &'static str,
) -> Result<Vec<i64>, BridgeError> {
    let shape = g
        .tensor(id)
        .shape
        .as_ref()
        .ok_or(BridgeError::MissingShape { node, op })?;
    shape
        .dims()
        .iter()
        .map(|d| {
            d.as_static().ok_or_else(|| BridgeError::Dynamic {
                node,
                op,
                dim: d.to_string(),
            })
        })
        .collect()
}

/// Split a tensor's dims into (rows = product of all but last, feat = last) —
/// the row-op view of any tensor (its token/row axis flattened).
fn rows_feat(dims: &[i64]) -> (i64, i64) {
    match dims.split_last() {
        Some((&feat, lead)) => (lead.iter().product::<i64>().max(1), feat),
        None => (1, 1),
    }
}

/// Row-major (C-contiguous) element strides for a shape, indexed by axis.
fn row_strides(dims: &[i64]) -> [u32; LAYOUT_RANK] {
    let mut s = [0u32; LAYOUT_RANK];
    let mut acc: i64 = 1;
    for d in (0..dims.len().min(LAYOUT_RANK)).rev() {
        s[d] = acc as u32;
        acc *= dims[d].max(1);
    }
    s
}

fn bytes_of(dims: &[i64], elem: u64) -> u64 {
    dims.iter().product::<i64>().max(0) as u64 * elem
}

/// Build the strided LAYOUT descriptor for a single-input layout op. Returns a
/// kind-0 copy (`LayoutSpec::copy`) for anything not expressible within
/// `LAYOUT_RANK` (the runtime then does a flat byte copy).
fn layout_spec(op: &Op, in_dims: &[i64], out: &[i64], elem: u64) -> LayoutSpec {
    let out_bytes = bytes_of(out, elem);
    let copy = LayoutSpec::copy(out_bytes);
    if in_dims.len() > LAYOUT_RANK || out.len() > LAYOUT_RANK {
        return copy;
    }
    let mut shape = [0u32; LAYOUT_RANK];
    let mut in_stride = [0u32; LAYOUT_RANK];
    let out_stride = row_strides(out);
    let mk = |shape, in_stride, in_base| LayoutSpec {
        bytes: out_bytes,
        kind: 1,
        rank: out.len() as u8,
        elem_size: elem as u8,
        shape,
        in_stride,
        out_stride,
        in_base,
        out_base: 0,
        alias: false,
    };
    match op {
        // Identity byte order — a zero-copy view (Phase C aliases it away).
        Op::Reshape { .. } => LayoutSpec::reshape(out_bytes),
        Op::Transpose { perm } => {
            if perm.len() != out.len() {
                return copy;
            }
            let in_str = row_strides(in_dims);
            for d in 0..out.len() {
                let p = perm[d] as usize;
                if p >= in_dims.len() {
                    return copy;
                }
                shape[d] = out[d] as u32;
                in_stride[d] = in_str[p]; // read in[.. idx[d] along input axis perm[d] ..]
            }
            mk(shape, in_stride, 0)
        }
        Op::Broadcast { .. } => {
            // Right-align the input shape under the (≥-rank) output.
            if in_dims.len() > out.len() {
                return copy;
            }
            let pad = out.len() - in_dims.len();
            let in_str = row_strides(in_dims);
            let mut any = false;
            for d in 0..out.len() {
                shape[d] = out[d] as u32;
                if d < pad {
                    in_stride[d] = 0; // new leading axis ⇒ broadcast
                    any = true;
                } else {
                    let id = d - pad;
                    if in_dims[id] == 1 && out[d] > 1 {
                        in_stride[d] = 0;
                        any = true;
                    } else {
                        in_stride[d] = in_str[id];
                    }
                }
            }
            if any {
                mk(shape, in_stride, 0)
            } else {
                copy
            }
        }
        Op::Slice { axis, start, len } => {
            let rank = in_dims.len() as i64;
            let ax = if *axis < 0 { *axis as i64 + rank } else { *axis as i64 };
            let (Some(start), Some(_len)) = (start.as_static(), len.as_static()) else {
                return copy; // symbolic slice → fall back to a copy of `out_bytes`
            };
            if ax < 0 || ax as usize >= in_dims.len() {
                return copy;
            }
            let in_str = row_strides(in_dims);
            for d in 0..out.len() {
                shape[d] = out[d] as u32;
                in_stride[d] = in_str[d]; // strides of the full input; only the extent shrank
            }
            mk(shape, in_stride, (start * in_str[ax as usize] as i64) as u32)
        }
        // Concat is multi-source; handled by `concat_spec` (kind 2), not here.
        _ => copy,
    }
}

/// Binary-concat descriptor (kind 2): scatter input 0 into the output region
/// `[0, split)` along `axis` and input 1 into `[split, end)`, where `split` is
/// input 0's extent along `axis`. The runtime reads both sources from the binding
/// (in0/in1). Reuses `in_base` to carry the axis and `out_base` the split count.
/// n-ary concat is already lowered to a binary chain upstream (`lower.rs`).
fn concat_spec(in0: &[i64], axis: i32, out: &[i64], elem: u64) -> LayoutSpec {
    let out_bytes = bytes_of(out, elem);
    let copy = LayoutSpec::copy(out_bytes);
    if out.len() > LAYOUT_RANK || in0.len() != out.len() {
        return copy;
    }
    let rank = out.len() as i64;
    let ax = if axis < 0 { axis as i64 + rank } else { axis as i64 };
    if ax < 0 || ax as usize >= out.len() {
        return copy;
    }
    let mut shape = [0u32; LAYOUT_RANK];
    for d in 0..out.len() {
        shape[d] = out[d] as u32;
    }
    LayoutSpec {
        bytes: out_bytes,
        kind: 2,
        rank: out.len() as u8,
        elem_size: elem as u8,
        shape,
        in_stride: [0; LAYOUT_RANK], // unused; the kernel derives per-piece strides
        out_stride: row_strides(out),
        in_base: ax as u32,                // axis
        out_base: in0[ax as usize] as u32, // split = input 0's extent along axis
        alias: false,
    }
}

fn op_kind(g: &Graph, id: NodeId, node: &Node) -> Result<OpKind, BridgeError> {
    let ni = id.0 as usize;
    let name = node.op.name();
    let dims = |idx: usize| static_dims(g, node.inputs[idx], ni, name);
    let out = static_dims(g, node.output, ni, name)?;
    const ELEM: u64 = 2; // bf16 operands, matching the cost model.

    let row = |operands: i64, reduce: bool| {
        let (rows, feat) = rows_feat(&out);
        OpKind::Row(RowShape {
            rows,
            feat,
            operands,
            reduce,
        })
    };

    Ok(match &node.op {
        Op::Linear { out_features, .. } => {
            let x = dims(0)?;
            let (m, k) = rows_feat(&x); // rows × contraction
            OpKind::Gemm(GemmShape {
                m,
                n: *out_features,
                k,
            })
        }
        Op::MatMul => {
            let a = dims(0)?;
            let b = dims(1)?;
            let (m, k) = rows_feat(&a);
            OpKind::Gemm(GemmShape {
                m,
                n: *b.last().unwrap_or(&1),
                k,
            })
        }
        // Row-wise, with a reduction sweep (mean/var or normalizing sum).
        Op::RmsNorm { .. } | Op::LayerNorm { .. } => row(node.inputs.len() as i64, true),
        Op::Softmax { .. } => row(1, true),
        // Row-wise, single pass (no cross-row reduction).
        Op::Act(_) | Op::Scale(_) | Op::Rope { .. } => row(1, false),
        Op::Elementwise(_) => row(2, false),
        Op::Attention {
            num_heads,
            head_dim,
            ..
        } => {
            // Convention: q/k are token-major ([seq, .., head_dim]), so the
            // token axis is tensor axis 0.
            let q = dims(0)?;
            let k = dims(1)?;
            OpKind::Flash(AttnShape {
                heads: *num_heads as i64,
                seq_q: *q.first().unwrap_or(&1),
                seq_kv: *k.first().unwrap_or(&1),
                head_dim: *head_dim as i64,
            })
        }
        // Pure layout / data movement → a strided descriptor (or a copy fallback).
        Op::Reshape { .. }
        | Op::Transpose { .. }
        | Op::Broadcast { .. }
        | Op::Slice { .. } => {
            let in0 = dims(0)?;
            OpKind::Layout(layout_spec(&node.op, &in0, &out, ELEM))
        }
        Op::Concat { axis } => {
            let in0 = dims(0)?;
            OpKind::Layout(concat_spec(&in0, *axis, &out, ELEM))
        }
        // MoE router: softmax(x·W) → topk. Row-wise reduction (2 inputs: x + weight).
        Op::MoeRouter { .. } => row(2, true),
        _ => return Err(BridgeError::Unsupported { node: ni, op: name }),
    })
}

// --- fused-graph bridge ------------------------------------------------------
//
// The post-fusion path: `nn_graph::Graph` → `rewrite_graph` → [`FusedGraph`] →
// here. The fused DAG is shape-free (egglog fusion does not reason about
// shapes), so we recover each node's shape with a small inference pass seeded
// from the source graph's leaf (input/weight) shapes plus the attributes the
// lowering preserved (`out_features`, reshape tokens). A fused norm+linear
// collapses to a single GEMM task — the norm→linear hand-off is internalized by
// fusion — while genuinely cross-op edges (residual→projection, attention,
// SwiGLU) survive as tile dependencies.

/// Lower a fused graph (whole-graph, rooted at the source graph's output) to a
/// [`LayerPlan`], recovering shapes from `graph`'s leaves.
pub fn plan_from_fused(fused: &FusedGraph, graph: &Graph) -> Result<LayerPlan, BridgeError> {
    let leaves = leaf_shapes(graph);
    let leaf_dt = leaf_dtypes(graph);
    // Forward pass: children are interned before parents, so 0..len is
    // topological and every reference points to a lower (already-shaped) index.
    let mut shapes: Vec<Vec<i64>> = Vec::with_capacity(fused.nodes.len());
    for i in 0..fused.nodes.len() {
        shapes.push(fused_shape(fused, &leaves, &shapes, i)?);
    }

    let mut ops = Vec::new();
    for (i, n) in fused.nodes.iter().enumerate() {
        if is_leaf(&n.op) {
            continue;
        }
        let kind = fused_op_kind(fused, &shapes, i)?;
        let inputs = node_args(n)
            .iter()
            .map(|&c| operand_name(fused, c))
            .collect();
        // SwiGLU carries its activation kind (silu → SwiGLU, gelu_tanh → GeGLU);
        // OpKind::Row has no act field, so surface it in the op description name
        // for kernel selection. Keeps the `SwiGLU` prefix intact.
        let name = match n.op.as_str() {
            "SwiGLU" => format!("SwiGLU_{}{i}", str_arg(n).unwrap_or("silu")),
            _ => format!("{}{i}", n.op),
        };
        let (weight_dtype, compute_dtype) = fused_node_dtypes(fused, &leaf_dt, i);
        ops.push(OpSpec {
            name,
            inputs,
            output: format!("f{i}"),
            kind,
            weight_dtype,
            compute_dtype,
        });
    }
    Ok(LayerPlan { ops })
}

fn is_leaf(op: &str) -> bool {
    op == "Input" || op == "Weight"
}

/// Named tensors of the source graph with a fully-static shape.
fn leaf_shapes(g: &Graph) -> HashMap<String, Vec<i64>> {
    let mut m = HashMap::new();
    for t in &g.tensors {
        if let (Some(name), Some(shape)) = (&t.name, &t.shape) {
            if let Some(dims) = shape
                .dims()
                .iter()
                .map(|d| d.as_static())
                .collect::<Option<Vec<_>>>()
            {
                m.insert(name.clone(), dims);
            }
        }
    }
    m
}

/// Named weight tensors → their dtype (from the source graph).
fn leaf_dtypes(g: &Graph) -> HashMap<String, nn_graph::DType> {
    let mut m = HashMap::new();
    for t in &g.tensors {
        if let (Some(name), nn_graph::Origin::Weight) = (&t.name, &t.origin) {
            m.insert(name.clone(), t.dtype);
        }
    }
    m
}

/// Determine (weight_dtype, compute_dtype) for a fused node by scanning its
/// child args for "Weight" leaves and looking up their dtype.
fn fused_node_dtypes(
    fused: &FusedGraph,
    leaf_dt: &HashMap<String, nn_graph::DType>,
    idx: usize,
) -> (nn_graph::DType, nn_graph::DType) {
    let n = &fused.nodes[idx];
    // Walk child nodes: if any is a Weight leaf, its name is the operand name.
    for &child in node_args(n).iter() {
        let cn = &fused.nodes[child];
        if cn.op == "Weight" {
            let wname = operand_name(fused, child);
            if let Some(&dt) = leaf_dt.get(&wname) {
                return (dt, dt.dequant_target());
            }
        }
    }
    (nn_graph::DType::BF16, nn_graph::DType::BF16)
}

/// The `Arg::Node` children of a fused node, in order (the operands; attribute
/// args — ints/floats/strings — are dropped). By schema the activation Expr is
/// always the first child, so `inputs[0]` is the activation.
fn node_args(n: &FNode) -> Vec<usize> {
    n.args
        .iter()
        .filter_map(|a| if let Arg::Node(i) = a { Some(*i) } else { None })
        .collect()
}

fn last_int(n: &FNode) -> Option<i64> {
    n.args
        .iter()
        .rev()
        .find_map(|a| if let Arg::Int(v) = a { Some(*v) } else { None })
}

fn int_args(n: &FNode) -> Vec<i64> {
    n.args
        .iter()
        .filter_map(|a| if let Arg::Int(v) = a { Some(*v) } else { None })
        .collect()
}

fn str_arg(n: &FNode) -> Option<&str> {
    n.args.iter().find_map(|a| {
        if let Arg::Str(s) = a {
            Some(s.as_str())
        } else {
            None
        }
    })
}

fn str_args(n: &FNode) -> Vec<&str> {
    n.args
        .iter()
        .filter_map(|a| {
            if let Arg::Str(s) = a {
                Some(s.as_str())
            } else {
                None
            }
        })
        .collect()
}

/// Normalize a possibly-negative axis against `rank`, erroring out of range.
fn norm_axis(node: usize, op: &str, axis: i64, rank: i64) -> Result<usize, BridgeError> {
    let ax = if axis < 0 { axis + rank } else { axis };
    if ax < 0 || ax >= rank {
        return Err(BridgeError::Axis {
            node,
            op: op.to_string(),
            axis,
            rank,
        });
    }
    Ok(ax as usize)
}

/// The operand name a later node uses to reference child `c`: the leaf's own
/// name (so weights/inputs dedup), else the synthesized node-output name `f{c}`.
fn operand_name(fused: &FusedGraph, c: usize) -> String {
    let n = &fused.nodes[c];
    match n.op.as_str() {
        "Input" | "Weight" => str_arg(n).unwrap_or("?").to_string(),
        _ => format!("f{c}"),
    }
}

fn replace_last(dims: &[i64], last: i64) -> Vec<i64> {
    let mut v = dims.to_vec();
    if let Some(slot) = v.last_mut() {
        *slot = last;
    }
    v
}

fn parse_shape_token(tok: &str) -> Option<Vec<i64>> {
    let inner = tok.trim().strip_prefix('[')?.strip_suffix(']')?;
    if inner.trim().is_empty() {
        return Some(vec![]);
    }
    inner
        .split(',')
        .map(|p| p.trim().parse::<i64>().ok())
        .collect()
}

/// Recover the static shape of fused node `i` from its already-shaped children.
fn fused_shape(
    fused: &FusedGraph,
    leaves: &HashMap<String, Vec<i64>>,
    shapes: &[Vec<i64>],
    i: usize,
) -> Result<Vec<i64>, BridgeError> {
    let n = &fused.nodes[i];
    let na = node_args(n);
    let child = |k: usize| shapes[na[k]].clone();
    let bad = || BridgeError::BadToken {
        node: i,
        op: n.op.clone(),
        token: str_arg(n).unwrap_or("").into(),
    };

    Ok(match n.op.as_str() {
        "Input" | "Weight" => {
            let name = str_arg(n).unwrap_or("?");
            leaves
                .get(name)
                .cloned()
                .ok_or_else(|| BridgeError::Unbound { name: name.into() })?
        }
        // [..rows, out_features]
        "Linear"
        | "LinearBias"
        | "FusedNormLinear"
        | "FusedNormLinearBias"
        | "FusedLayerNormLinear"
        | "FusedLayerNormLinearBias"
        | "FusedLinearAct"
        | "FusedLinearBiasAct" => replace_last(&child(0), last_int(n).ok_or_else(bad)?),
        "MatMul" => replace_last(&child(0), *child(1).last().unwrap_or(&1)),
        // output takes the query's axes but the value's head dim (q, k, v)
        "Attention" => replace_last(&child(0), *child(2).last().unwrap_or(&1)),
        // shape-preserving (activation is the first child)
        "RmsNorm"
        | "LayerNorm"
        | "Rope"
        | "Act"
        | "Scale"
        | "Softmax"
        | "FusedNormRope"
        | "FusedNormRopeScale"
        | "SwiGLU"
        | "FusedAdaLN"
        | "FusedGatedResidual"
        | "FusedResidualNorm"
        | "FusedResidualLayerNorm"
        | "FusedGroupNormAct" => child(0),
        // Embedding: output is [..ids_shape, hidden] where hidden = table's last dim.
        "Embedding" | "FusedEmbeddingScale" => {
            let mut d = child(0);
            let table = child(1);
            d.push(*table.last().unwrap_or(&1));
            d
        }
        // GroupNorm+Act+Conv3d: output shape derived from conv weight + stride/padding.
        // Conv weight is [out_c, in_c, kd, kh, kw]; output channels = weight dim 0.
        // For simplicity, we use the same logic as Conv3d shape inference.
        "FusedGroupNormActConv3d" | "FusedGroupNormActConv3dBias" => {
            // child(0) is the input x; the conv weight carries the output channels.
            // Find the conv weight child (node arg after the groupnorm params).
            // Schema: x, gnW, gnB, (ints/floats: groups, eps, act), convW, ...
            // In node_args order: [x, gnW, gnB, convW, ...] (only Expr children)
            let x_shape = child(0);
            let conv_w_shape = child(3); // convW is 4th Expr arg
                                         // output channels from conv weight dim 0
            let out_c = conv_w_shape[0];
            // For stride/pad we'd need the tokens; approximate: same spatial as input
            // (stride=1, pad=same is the dominant VAE pattern).
            let mut out_shape = x_shape;
            out_shape[1] = out_c; // NCDHW: channel axis
            out_shape
        }
        // broadcasting elementwise: take the higher-rank operand
        "Ew" => {
            let (a, b) = (child(0), child(1));
            if b.len() > a.len() {
                b
            } else {
                a
            }
        }
        "Reshape" | "Broadcast" => parse_shape_token(str_arg(n).unwrap_or("")).ok_or_else(bad)?,
        "Transpose" => {
            let perm: Option<Vec<usize>> = str_arg(n)
                .unwrap_or("")
                .split(',')
                .map(|p| p.trim().parse().ok())
                .collect();
            let (x, perm) = (child(0), perm.ok_or_else(bad)?);
            perm.iter().map(|&p| x[p]).collect()
        }
        "Concat" => {
            let mut d = child(0);
            let axis = int_args(n).first().copied().unwrap_or(0);
            let ax = norm_axis(i, &n.op, axis, d.len() as i64)?;
            let b = child(1);
            let extent = *b.get(ax).ok_or(BridgeError::Axis {
                node: i,
                op: n.op.clone(),
                axis,
                rank: b.len() as i64,
            })?;
            d[ax] += extent;
            d
        }
        "Slice" => {
            // (Slice x axis start-tok len-tok)
            let mut d = child(0);
            let axis = int_args(n).first().copied().unwrap_or(0);
            let ax = norm_axis(i, &n.op, axis, d.len() as i64)?;
            // len is the second string token; a symbolic (non-integer) len has
            // no static shape to recover here.
            let len_tok = str_args(n).get(1).copied().unwrap_or("").to_string();
            let len: i64 = len_tok.trim().parse().map_err(|_| BridgeError::BadToken {
                node: i,
                op: n.op.clone(),
                token: len_tok.clone(),
            })?;
            d[ax] = len;
            d
        }
        _ => {
            return Err(BridgeError::UnsupportedFused {
                node: i,
                op: n.op.clone(),
            })
        }
    })
}

fn fused_op_kind(fused: &FusedGraph, shapes: &[Vec<i64>], i: usize) -> Result<OpKind, BridgeError> {
    let n = &fused.nodes[i];
    let na = node_args(n);
    let out = &shapes[i];
    const ELEM: u64 = 2;
    let row = |operands: i64, reduce: bool| {
        let (rows, feat) = rows_feat(out);
        OpKind::Row(RowShape {
            rows,
            feat,
            operands,
            reduce,
        })
    };

    Ok(match n.op.as_str() {
        "Linear"
        | "LinearBias"
        | "FusedNormLinear"
        | "FusedNormLinearBias"
        | "FusedLayerNormLinear"
        | "FusedLayerNormLinearBias"
        | "FusedLinearAct"
        | "FusedLinearBiasAct" => {
            let (m, k) = rows_feat(&shapes[na[0]]);
            OpKind::Gemm(GemmShape {
                m,
                n: last_int(n).unwrap_or(*out.last().unwrap_or(&1)),
                k,
            })
        }
        "MatMul" => {
            let (m, k) = rows_feat(&shapes[na[0]]);
            OpKind::Gemm(GemmShape {
                m,
                n: *shapes[na[1]].last().unwrap_or(&1),
                k,
            })
        }
        "RmsNorm" | "LayerNorm" => row(na.len() as i64, true),
        // GroupNorm+Act without a following conv: a norm kernel — row-wise with
        // a per-group reduction sweep over (x, w, b).
        "FusedGroupNormAct" => row(na.len() as i64, true),
        // Residual+norm: 3 operands (a, b, normw), reduction sweep.
        "FusedResidualNorm" => row(3, true),
        "FusedResidualLayerNorm" => row(4, true),
        "Softmax" => row(1, true),
        "Act" | "Scale" | "Rope" => row(1, false),
        // Embedding/FusedEmbeddingScale: memory-bound row-wise lookup.
        "Embedding" | "FusedEmbeddingScale" => row(1, false),
        // norm+rope keeps the norm's reduction; elementwise/gated combines do not.
        "FusedNormRope" | "FusedNormRopeScale" => row(2, true),
        "Ew" | "SwiGLU" | "FusedAdaLN" | "FusedGatedResidual" => row(na.len() as i64, false),
        // GroupNorm+Act+Conv3d: model as layout (conv-dominated, same cost as Conv3d).
        "FusedGroupNormActConv3d" | "FusedGroupNormActConv3dBias" => {
            OpKind::Layout(LayoutSpec::copy(out.iter().product::<i64>().max(0) as u64 * ELEM))
        }
        "Attention" => {
            // heads / head_dim come from the opaque config token the lowering
            // serialized (`heads=..;kv=..;hd=..;..`) — no shape guessing.
            let q = &shapes[na[0]];
            let k = &shapes[na[1]];
            let tok = str_arg(n).unwrap_or("");
            let field = |key: &str| -> Option<i64> {
                tok.split(';')
                    .find_map(|kv| kv.strip_prefix(key)?.strip_prefix('=')?.parse().ok())
            };
            let (Some(heads), Some(head_dim)) = (field("heads"), field("hd")) else {
                return Err(BridgeError::BadToken {
                    node: i,
                    op: n.op.clone(),
                    token: tok.into(),
                });
            };
            // Convention: q/k are token-major ([seq, ..., head_dim]); anything
            // deeper than rank 3 would misread the token axis — error instead.
            if q.len() > 3 || k.len() > 3 {
                return Err(BridgeError::Rank {
                    node: i,
                    op: n.op.clone(),
                    rank: q.len().max(k.len()),
                    expected: "token-major q/k of rank <= 3 ([seq, heads, head_dim])",
                });
            }
            OpKind::Flash(AttnShape {
                heads,
                seq_q: *q.first().unwrap_or(&1),
                seq_kv: *k.first().unwrap_or(&1),
                head_dim,
            })
        }
        "Reshape" | "Transpose" | "Broadcast" | "Concat" | "Slice" => {
            // Fused-graph path: shapes are known but the structured op attributes
            // (perm/axis) live in tokens; keep a contiguous copy here and rely on
            // the structured `op_kind` path for real descriptors.
            OpKind::Layout(LayoutSpec::copy(out.iter().product::<i64>().max(0) as u64 * ELEM))
        }
        _ => {
            return Err(BridgeError::UnsupportedFused {
                node: i,
                op: n.op.clone(),
            })
        }
    })
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    use nn_graph::{Op, Shape};

    #[test]
    fn transpose_2x3_to_3x2_descriptor() {
        // out axis d reads input axis perm[d]; input (2,3) row-major strides (3,1).
        let op = Op::Transpose { perm: vec![1, 0] };
        let s = layout_spec(&op, &[2, 3], &[3, 2], 1);
        assert_eq!(s.kind, 1);
        assert_eq!(s.rank, 2);
        assert_eq!(&s.shape[..2], &[3, 2]);
        assert_eq!(&s.in_stride[..2], &[1, 3]); // perm[0]=1→stride1, perm[1]=0→stride3
        assert_eq!(&s.out_stride[..2], &[2, 1]); // out (3,2) row-major
        assert_eq!(s.in_base, 0);
    }

    #[test]
    fn broadcast_size1_axis_gets_zero_stride() {
        // input (1,4) → out (3,4): axis 0 broadcasts (stride 0), axis 1 keeps stride 1.
        let op = Op::Broadcast { shape: Shape::new([3i64.into(), 4i64.into()]) };
        let s = layout_spec(&op, &[1, 4], &[3, 4], 1);
        assert_eq!(s.kind, 1);
        assert_eq!(&s.in_stride[..2], &[0, 1]);
        assert_eq!(&s.shape[..2], &[3, 4]);
    }

    #[test]
    fn slice_outer_axis_offsets_in_base() {
        // slice axis 0, start 2, len 3 of input (5,4): in_base = 2 * stride0(=4) = 8.
        let op = Op::Slice { axis: 0, start: 2i64.into(), len: 3i64.into() };
        let s = layout_spec(&op, &[5, 4], &[3, 4], 1);
        assert_eq!(s.kind, 1);
        assert_eq!(s.in_base, 8);
        assert_eq!(&s.in_stride[..2], &[4, 1]); // strides of the full input
        assert_eq!(&s.shape[..2], &[3, 4]);
    }

    #[test]
    fn reshape_is_a_contiguous_copy() {
        let op = Op::Reshape { shape: Shape::new([6i64.into()]) };
        let s = layout_spec(&op, &[2, 3], &[6], 2);
        assert_eq!(s.kind, 0);
        assert_eq!(s.bytes, 6 * 2);
    }

    #[test]
    fn concat_axis1_descriptor() {
        // concat (2,2)+(2,2) along axis 1 → (2,4): kind 2, axis=1, split=2.
        let s = concat_spec(&[2, 2], 1, &[2, 4], 1);
        assert_eq!(s.kind, 2);
        assert_eq!(s.in_base, 1); // axis
        assert_eq!(s.out_base, 2); // split = input 0 extent along axis
        assert_eq!(&s.out_stride[..2], &[4, 1]);
        assert_eq!(&s.shape[..2], &[2, 4]);
    }

    #[test]
    fn concat_negative_axis_normalizes() {
        // axis -1 on rank-2 output → axis 1.
        let s = concat_spec(&[2, 2], -1, &[2, 4], 1);
        assert_eq!(s.kind, 2);
        assert_eq!(s.in_base, 1);
    }

    #[test]
    fn over_rank_falls_back_to_copy() {
        let op = Op::Transpose { perm: vec![6, 5, 4, 3, 2, 1, 0] };
        let s = layout_spec(&op, &[2, 2, 2, 2, 2, 2, 2], &[2, 2, 2, 2, 2, 2, 2], 1);
        assert_eq!(s.kind, 0, "rank > LAYOUT_RANK → copy fallback");
    }
}
