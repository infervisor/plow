//! Symbolic shape inference.
//!
//! Walks nodes in construction order (the builder adds producers before
//! consumers, so this is already a topological order) and fills each
//! node-output tensor's shape by applying the op's shape rule to its inputs'
//! shapes. All arithmetic is symbolic via [`Dim`].

use crate::dim::Dim;
use crate::graph::{Graph, TensorId};
use crate::op::Op;
use crate::shape::Shape;
use smallvec::SmallVec;

#[derive(thiserror::Error, Debug)]
pub enum InferError {
    #[error("node {node} ({op}): expected {expected} inputs, got {got}")]
    Arity {
        node: usize,
        op: &'static str,
        expected: usize,
        got: usize,
    },

    #[error("node {node} ({op}): input {idx} has no shape (not yet inferred)")]
    MissingShape {
        node: usize,
        op: &'static str,
        idx: usize,
    },

    #[error("node {node} ({op}): {msg}")]
    Bad {
        node: usize,
        op: &'static str,
        msg: String,
    },
}

/// Run inference over the whole graph, mutating tensor shapes in place.
pub fn infer_shapes(graph: &mut Graph) -> Result<(), InferError> {
    for ni in 0..graph.nodes.len() {
        let (op, inputs, output) = {
            let n = &graph.nodes[ni];
            (n.op.clone(), n.inputs.clone(), n.output)
        };
        let cx = Ctx { graph, ni, op: &op };
        let shape = cx.infer(&inputs)?;
        set_shape(graph, output, shape);
    }
    Ok(())
}

fn set_shape(graph: &mut Graph, id: TensorId, shape: Shape) {
    graph.tensors[id.0 as usize].shape = Some(shape);
}

struct Ctx<'a> {
    graph: &'a Graph,
    ni: usize,
    op: &'a Op,
}

impl Ctx<'_> {
    fn err(&self, msg: impl Into<String>) -> InferError {
        InferError::Bad {
            node: self.ni,
            op: self.op.name(),
            msg: msg.into(),
        }
    }

    fn input_shape(&self, inputs: &[TensorId], idx: usize) -> Result<Shape, InferError> {
        let id = inputs.get(idx).ok_or(InferError::Arity {
            node: self.ni,
            op: self.op.name(),
            expected: idx + 1,
            got: inputs.len(),
        })?;
        self.graph
            .tensor(*id)
            .shape
            .clone()
            .ok_or(InferError::MissingShape {
                node: self.ni,
                op: self.op.name(),
                idx,
            })
    }

    fn expect_arity(&self, inputs: &[TensorId], n: usize) -> Result<(), InferError> {
        if inputs.len() != n {
            return Err(InferError::Arity {
                node: self.ni,
                op: self.op.name(),
                expected: n,
                got: inputs.len(),
            });
        }
        Ok(())
    }

    fn infer(&self, inputs: &[TensorId]) -> Result<Shape, InferError> {
        match self.op {
            Op::Linear { out_features, bias } => {
                self.expect_arity(inputs, if *bias { 3 } else { 2 })?;
                let x = self.input_shape(inputs, 0)?;
                if x.rank() == 0 {
                    return Err(self.err("linear input must have rank >= 1"));
                }
                Ok(replace_last(&x, Dim::stat(*out_features)))
            }

            Op::MatMul => {
                self.expect_arity(inputs, 2)?;
                let a = self.input_shape(inputs, 0)?;
                let b = self.input_shape(inputs, 1)?;
                if a.rank() < 2 || b.rank() < 2 {
                    return Err(self.err("matmul operands must have rank >= 2"));
                }
                let m = a.dim(a.rank() - 2).clone();
                let n = b.dim(b.rank() - 1).clone();
                // Broadcast batch dims (all but last two).
                let batch = broadcast_prefix(self, &a, &b, 2)?;
                let mut dims = batch;
                dims.push(m);
                dims.push(n);
                Ok(Shape::new(dims))
            }

            // Shape-preserving unary ops.
            Op::Act(_) | Op::Scale(_) | Op::Softmax { .. } => {
                self.expect_arity(inputs, 1)?;
                self.input_shape(inputs, 0)
            }

            // Shape-preserving ops with weight inputs.
            Op::RmsNorm { .. } => {
                self.expect_arity_range(inputs, 1, 2)?;
                self.input_shape(inputs, 0)
            }
            Op::RmsNormZeroCentered { .. } => {
                self.expect_arity(inputs, 2)?;
                self.input_shape(inputs, 0)
            }
            Op::LayerNorm { .. } => {
                self.expect_arity(inputs, 3)?;
                self.input_shape(inputs, 0)
            }
            Op::Rope { .. } => {
                self.expect_arity(inputs, 1)?;
                self.input_shape(inputs, 0)
            }

            Op::Attention { .. } => {
                // Output has the query's batch/seq/head axes but the *value's*
                // head dim (these differ for MLA, where qk_head_dim != v_head_dim).
                self.expect_arity_range(inputs, 3, 4)?;
                let q = self.input_shape(inputs, 0)?;
                let k = self.input_shape(inputs, 1)?;
                let v = self.input_shape(inputs, 2)?;
                if q.rank() < 2 || k.rank() < 2 || v.rank() < 2 {
                    return Err(self.err("attention q/k/v must have rank >= 2"));
                }
                // Q and K share the qk head dim (head counts may differ: GQA).
                let (qd, kd) = (q.dim(q.rank() - 1), k.dim(k.rank() - 1));
                if qd.provably_ne(kd) {
                    return Err(
                        self.err(format!("attention q/k head dim mismatch: {} vs {}", qd, kd))
                    );
                }
                Ok(replace_last(&q, v.dim(v.rank() - 1).clone()))
            }

            Op::Elementwise(_) => {
                self.expect_arity(inputs, 2)?;
                let a = self.input_shape(inputs, 0)?;
                let b = self.input_shape(inputs, 1)?;
                broadcast(self, &a, &b)
            }

            Op::Embedding => {
                self.expect_arity(inputs, 2)?;
                let ids = self.input_shape(inputs, 0)?;
                let table = self.input_shape(inputs, 1)?;
                if table.rank() != 2 {
                    return Err(self.err("embedding table must be rank-2 [vocab, hidden]"));
                }
                let mut dims: SmallVec<[Dim; 4]> = ids.dims().iter().cloned().collect();
                dims.push(table.dim(1).clone());
                Ok(Shape::new(dims))
            }

            Op::Conv2d { stride, padding } => {
                self.expect_arity_range(inputs, 2, 3)?;
                let x = self.input_shape(inputs, 0)?;
                let w = self.input_shape(inputs, 1)?;
                if x.rank() != 4 || w.rank() != 4 {
                    return Err(self.err("conv2d expects NCHW input and OIHW weight"));
                }
                let out_c = w.dim(0).clone();
                let (kh, kw) = (
                    req_static(self, w.dim(2), "kernel h")?,
                    req_static(self, w.dim(3), "kernel w")?,
                );
                let h = req_static(self, x.dim(2), "input h")?;
                let wd = req_static(self, x.dim(3), "input w")?;
                let oh = (h + 2 * padding.0 as i64 - kh) / stride.0 as i64 + 1;
                let ow = (wd + 2 * padding.1 as i64 - kw) / stride.1 as i64 + 1;
                Ok(Shape::new([
                    x.dim(0).clone(),
                    out_c,
                    Dim::stat(oh),
                    Dim::stat(ow),
                ]))
            }

            Op::Conv3d { stride, padding } => {
                self.expect_arity_range(inputs, 2, 3)?;
                let x = self.input_shape(inputs, 0)?;
                let w = self.input_shape(inputs, 1)?;
                if x.rank() != 5 || w.rank() != 5 {
                    return Err(self.err("conv3d expects NCDHW input and OIDHW weight"));
                }
                let out_c = w.dim(0).clone();
                let k = [
                    req_static(self, w.dim(2), "kernel d")?,
                    req_static(self, w.dim(3), "kernel h")?,
                    req_static(self, w.dim(4), "kernel w")?,
                ];
                let inp = [
                    req_static(self, x.dim(2), "input d")?,
                    req_static(self, x.dim(3), "input h")?,
                    req_static(self, x.dim(4), "input w")?,
                ];
                let st = [stride.0 as i64, stride.1 as i64, stride.2 as i64];
                let pad = [padding.0 as i64, padding.1 as i64, padding.2 as i64];
                let out: Vec<Dim> = (0..3)
                    .map(|i| Dim::stat((inp[i] + 2 * pad[i] - k[i]) / st[i] + 1))
                    .collect();
                Ok(Shape::new([
                    x.dim(0).clone(),
                    out_c,
                    out[0].clone(),
                    out[1].clone(),
                    out[2].clone(),
                ]))
            }

            // GroupNorm is shape-preserving.
            Op::GroupNorm { .. } => self.input_shape(inputs, 0),

            Op::Reshape { shape } => {
                // Trust the explicit target; reject when the element counts
                // are provably unequal (covers static mismatches and symbolic
                // ones like B*256 -> B*512; B*256 vs a static target is not
                // provable and is trusted).
                let x = self.input_shape(inputs, 0)?;
                let (a, b) = (x.numel(), shape.numel());
                if a.provably_ne(&b) {
                    return Err(self.err(format!("reshape changes element count: {} -> {}", a, b)));
                }
                Ok(shape.clone())
            }

            Op::Broadcast { shape } => {
                let x = self.input_shape(inputs, 0)?;
                if x.rank() != shape.rank() {
                    return Err(self.err("broadcast target rank != input rank"));
                }
                for i in 0..x.rank() {
                    let (xi, ti) = (x.dim(i), shape.dim(i));
                    if xi != ti && xi.as_static() != Some(1) {
                        return Err(
                            self.err(format!("axis {} not broadcastable: {} -> {}", i, xi, ti))
                        );
                    }
                }
                Ok(shape.clone())
            }

            Op::Transpose { perm } => {
                let x = self.input_shape(inputs, 0)?;
                if perm.len() != x.rank() {
                    return Err(self.err("transpose perm length != input rank"));
                }
                let dims: SmallVec<[Dim; 4]> =
                    perm.iter().map(|&p| x.dim(p as usize).clone()).collect();
                Ok(Shape::new(dims))
            }

            Op::Concat { axis } => {
                if inputs.len() < 2 {
                    return Err(self.err("concat needs >= 2 inputs"));
                }
                let first = self.input_shape(inputs, 0)?;
                let ax = normalize_axis(self, *axis, first.rank())?;
                let mut dims: SmallVec<[Dim; 4]> = first.dims().iter().cloned().collect();
                for i in 1..inputs.len() {
                    let s = self.input_shape(inputs, i)?;
                    if s.rank() != first.rank() {
                        return Err(self.err("concat inputs differ in rank"));
                    }
                    for d in 0..first.rank() {
                        if d != ax && first.dim(d).provably_ne(s.dim(d)) {
                            return Err(self.err(format!(
                                "concat input {} axis {} mismatch: {} vs {}",
                                i,
                                d,
                                first.dim(d),
                                s.dim(d)
                            )));
                        }
                    }
                    dims[ax] = dims[ax].add(s.dim(ax));
                }
                Ok(Shape::new(dims))
            }

            Op::Slice { axis, start, len } => {
                let x = self.input_shape(inputs, 0)?;
                let ax = normalize_axis(self, *axis, x.rank())?;
                // Bounds-check when everything is static (symbolic slices are
                // validated at bind time by the consumer).
                if let (Some(st), Some(ln), Some(d)) =
                    (start.as_static(), len.as_static(), x.dim(ax).as_static())
                {
                    if st < 0 || ln < 0 || st + ln > d {
                        return Err(self.err(format!(
                            "slice out of bounds: [{}, {}) on axis {} of size {}",
                            st,
                            st + ln,
                            ax,
                            d
                        )));
                    }
                }
                let mut dims: SmallVec<[Dim; 4]> = x.dims().iter().cloned().collect();
                dims[ax] = len.clone();
                Ok(Shape::new(dims))
            }

            Op::Reduce {
                kind: _,
                axis,
                keepdim,
            } => {
                let x = self.input_shape(inputs, 0)?;
                let ax = normalize_axis(self, *axis, x.rank())?;
                let mut dims: SmallVec<[Dim; 4]> = x.dims().iter().cloned().collect();
                if *keepdim {
                    dims[ax] = Dim::stat(1);
                } else {
                    dims.remove(ax);
                }
                Ok(Shape::new(dims))
            }

            Op::MoeRouter {
                num_experts,
                group,
                correction_bias,
                ..
            } => {
                self.expect_arity(inputs, if *correction_bias { 3 } else { 2 })?;
                let x = self.input_shape(inputs, 0)?;
                if x.rank() == 0 {
                    return Err(self.err("moe_router input must have rank >= 1"));
                }
                // The kernel partitions the experts into CONTIGUOUS equal groups
                // (`gsz = num_experts / n_group`) and a remainder would put some
                // experts in no group at all — kept or dropped depending on
                // uninitialised scratch. Caught here, where the config is still
                // in hand, rather than on device.
                if let Some(g) = group {
                    if g.n_group > 1 && *num_experts % g.n_group != 0 {
                        return Err(self.err(format!(
                            "moe_router: num_experts = {num_experts} is not divisible by \
                             n_group = {}; group-limited routing partitions the experts into \
                             contiguous equal groups and the remainder would belong to none",
                            g.n_group
                        )));
                    }
                    if g.topk_group == 0 || g.topk_group > g.n_group.max(1) {
                        return Err(self.err(format!(
                            "moe_router: topk_group = {} must be in 1..=n_group = {}",
                            g.topk_group, g.n_group
                        )));
                    }
                }
                Ok(replace_last(&x, Dim::stat(*num_experts as i64)))
            }

            Op::MoeExperts { .. } => {
                self.expect_arity(inputs, 2)?;
                self.input_shape(inputs, 0)
            }

            Op::DsaIndexer {
                num_heads,
                head_dim,
                rope_dim,
                top_k,
                ..
            } => {
                self.expect_arity(inputs, 7)?;
                let hidden = self.input_shape(inputs, 0)?;
                if hidden.rank() != 3 {
                    return Err(self.err("dsa_indexer hidden input must be [B,S,H]"));
                }
                if *num_heads == 0 || *head_dim == 0 || *top_k == 0 || *rope_dim > *head_dim {
                    return Err(self.err("dsa_indexer has invalid head/rope/top-k geometry"));
                }
                Ok(replace_last(&hidden, Dim::stat(*top_k as i64)))
            }

            Op::DsaAttention { top_k, .. } => {
                self.expect_arity(inputs, 4)?;
                let q = self.input_shape(inputs, 0)?;
                let k = self.input_shape(inputs, 1)?;
                let v = self.input_shape(inputs, 2)?;
                let indices = self.input_shape(inputs, 3)?;
                if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 || indices.rank() != 3 {
                    return Err(self.err("dsa_attention expects q/k/v rank 4 and indices rank 3"));
                }
                if indices.dim(2).as_static() != Some(*top_k as i64) {
                    return Err(self.err("dsa_attention index width does not match top_k"));
                }
                Ok(v.clone())
            }

            Op::Conv1dDepthwise { kernel } => {
                self.expect_arity_range(inputs, 2, 3)?;
                let x = self.input_shape(inputs, 0)?;
                let w = self.input_shape(inputs, 1)?;
                if x.rank() != 3 {
                    return Err(self.err("conv1d_depthwise expects [B, S, C]"));
                }
                if w.rank() != 3 {
                    return Err(self.err("conv1d_depthwise weight must be [C, 1, kernel]"));
                }
                if req_static(self, w.dim(1), "input channels per group")? != 1 {
                    return Err(self.err(format!(
                        "conv1d_depthwise input channels per group is {}, expected 1",
                        w.dim(1)
                    )));
                }
                let kw = req_static(self, w.dim(2), "kernel")?;
                if kw != *kernel as i64 {
                    return Err(self.err(format!(
                        "conv1d_depthwise weight kernel is {kw}, op says {kernel}"
                    )));
                }
                // Depthwise: one filter per channel, so the weight's rows must be
                // the channel count. A mismatch here is a mis-bound tensor, which
                // otherwise shows up as wrong output rather than a load error.
                let (c, wc) = (x.dim(2), w.dim(0));
                if c.provably_ne(wc) {
                    return Err(self.err(format!(
                        "conv1d_depthwise channel mismatch: input {c} vs weight {wc}"
                    )));
                }
                // Causal left-pad of `kernel - 1` ⇒ length preserved.
                Ok(x)
            }

            Op::LinearAttention {
                kind,
                num_heads,
                head_dim,
            } => {
                self.expect_arity(inputs, 7)?;
                let q = self.input_shape(inputs, 0)?;
                let k = self.input_shape(inputs, 1)?;
                let v = self.input_shape(inputs, 2)?;
                let gate = self.input_shape(inputs, 3)?;
                let beta = self.input_shape(inputs, 4)?;
                let a_log = self.input_shape(inputs, 5)?;
                let dt_bias = self.input_shape(inputs, 6)?;
                if q.rank() != 4 {
                    return Err(self.err("linear_attention q must be [B, S, heads, head_dim]"));
                }
                for (name, s) in [("k", &k), ("v", &v)] {
                    if s.rank() != 4 {
                        return Err(
                            self.err(format!("linear_attention {name} must be rank-4 like q"))
                        );
                    }
                    if s.dim(3).provably_ne(q.dim(3)) {
                        return Err(self.err(format!(
                            "linear_attention {name} head dim {} != q head dim {}",
                            s.dim(3),
                            q.dim(3)
                        )));
                    }
                    for axis in 0..3 {
                        if s.dim(axis).provably_ne(q.dim(axis)) {
                            return Err(self.err(format!(
                                "linear_attention {name} axis {axis} differs from q"
                            )));
                        }
                    }
                }
                let gate_rank = match kind {
                    crate::op::LinearAttnKind::KimiDelta => 4,
                    crate::op::LinearAttnKind::QwenGatedDelta => 3,
                };
                if gate.rank() != gate_rank {
                    return Err(self.err(format!(
                        "linear_attention gate must be rank {gate_rank} for {kind:?}"
                    )));
                }
                for axis in 0..gate_rank {
                    if gate.dim(axis).provably_ne(q.dim(axis)) {
                        return Err(
                            self.err(format!("linear_attention gate axis {axis} differs from q"))
                        );
                    }
                }
                // beta is ONE scalar per (token, head) — the delta-rule write
                // strength. A [B,S,heads,head_dim] beta would be a per-channel
                // gate, a different model.
                if beta.rank() != 3 {
                    return Err(self.err("linear_attention beta must be [B, S, heads]"));
                }
                for axis in 0..3 {
                    if beta.dim(axis).provably_ne(q.dim(axis)) {
                        return Err(
                            self.err(format!("linear_attention beta axis {axis} differs from q"))
                        );
                    }
                }
                let hd = req_static(self, q.dim(3), "head_dim")?;
                if hd != *head_dim as i64 {
                    return Err(self.err(format!(
                        "linear_attention head_dim is {hd} in the tensors, {head_dim} on the op"
                    )));
                }
                let nh = req_static(self, q.dim(2), "num_heads")?;
                if nh != *num_heads as i64 {
                    return Err(self.err(format!(
                        "linear_attention num_heads is {nh} in the tensors, {num_heads} on the op"
                    )));
                }
                if a_log.rank() != 1 || a_log.dim(0).provably_ne(q.dim(2)) {
                    return Err(self.err(format!(
                        "linear_attention A_log shape {a_log} must be [num_heads = {}]",
                        q.dim(2)
                    )));
                }
                let dt_width = match kind {
                    crate::op::LinearAttnKind::KimiDelta => nh * hd,
                    crate::op::LinearAttnKind::QwenGatedDelta => nh,
                };
                if dt_bias.rank() != 1
                    || req_static(self, dt_bias.dim(0), "dt_bias width")? != dt_width
                {
                    return Err(self.err(format!(
                        "linear_attention dt_bias shape {dt_bias} must have width {dt_width}"
                    )));
                }
                // The KDA state is square ([head_dim, head_dim]), so the output
                // carries the VALUE head dim — read off v rather than assumed.
                Ok(replace_last(&q, v.dim(3).clone()))
            }

            Op::SituGlu { .. } => {
                self.expect_arity(inputs, 2)?;
                let gate = self.input_shape(inputs, 0)?;
                let up = self.input_shape(inputs, 1)?;
                broadcast(self, &gate, &up)
            }

            Op::BlockResidual { max_snapshots } => {
                // [prefix, snapshot_0.., norm_weight, proj_weight].
                self.expect_arity_range(inputs, 3, 3 + *max_snapshots as usize)?;
                let prefix = self.input_shape(inputs, 0)?;
                if prefix.rank() == 0 {
                    return Err(self.err("block_residual prefix must have rank >= 1"));
                }
                // Every snapshot is an earlier value of the same running sum, so
                // it has to match the prefix exactly; a mismatch means the
                // snapshot stack was pushed from the wrong place.
                for i in 1..inputs.len() - 2 {
                    let s = self.input_shape(inputs, i)?;
                    if s.rank() != prefix.rank()
                        || s.dim(s.rank() - 1)
                            .provably_ne(prefix.dim(prefix.rank() - 1))
                    {
                        return Err(self.err(format!(
                            "block_residual snapshot {i} has shape {s}, expected the prefix shape {prefix}"
                        )));
                    }
                }
                let hidden = prefix.dim(prefix.rank() - 1);
                let norm = self.input_shape(inputs, inputs.len() - 2)?;
                if norm.rank() != 1 || norm.dim(0).provably_ne(hidden) {
                    return Err(self.err(format!(
                        "block_residual norm weight has shape {norm}, expected [{hidden}]"
                    )));
                }
                let proj = self.input_shape(inputs, inputs.len() - 1)?;
                if proj.rank() != 2
                    || req_static(self, proj.dim(0), "projection rows")? != 1
                    || proj.dim(1).provably_ne(hidden)
                {
                    return Err(self.err(format!(
                        "block_residual projection weight has shape {proj}, expected [1, {hidden}]"
                    )));
                }
                Ok(prefix)
            }
        }
    }

    fn expect_arity_range(
        &self,
        inputs: &[TensorId],
        lo: usize,
        hi: usize,
    ) -> Result<(), InferError> {
        if inputs.len() < lo || inputs.len() > hi {
            return Err(InferError::Bad {
                node: self.ni,
                op: self.op.name(),
                msg: format!(
                    "expected between {} and {} inputs, got {}",
                    lo,
                    hi,
                    inputs.len()
                ),
            });
        }
        Ok(())
    }
}

fn replace_last(shape: &Shape, last: Dim) -> Shape {
    let mut dims: SmallVec<[Dim; 4]> = shape.dims().iter().cloned().collect();
    let n = dims.len();
    dims[n - 1] = last;
    Shape::new(dims)
}

fn req_static(cx: &Ctx, d: &Dim, what: &str) -> Result<i64, InferError> {
    d.as_static()
        .ok_or_else(|| cx.err(format!("{} must be statically known, got {}", what, d)))
}

fn normalize_axis(cx: &Ctx, axis: i32, rank: usize) -> Result<usize, InferError> {
    let a = if axis < 0 { axis + rank as i32 } else { axis };
    if a < 0 || a as usize >= rank {
        return Err(cx.err(format!("axis {} out of range for rank {}", axis, rank)));
    }
    Ok(a as usize)
}

/// Broadcast two full shapes (numpy rules), symbolic-aware.
fn broadcast(cx: &Ctx, a: &Shape, b: &Shape) -> Result<Shape, InferError> {
    let rank = a.rank().max(b.rank());
    let mut dims: Vec<Dim> = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = axis_from_right(a, i);
        let db = axis_from_right(b, i);
        dims.push(broadcast_dim(cx, da, db)?);
    }
    dims.reverse();
    Ok(Shape::new(dims))
}

/// Broadcast only the leading `prefix`-excluded batch dims (all but the last
/// `tail` dims), used by matmul.
fn broadcast_prefix(
    cx: &Ctx,
    a: &Shape,
    b: &Shape,
    tail: usize,
) -> Result<SmallVec<[Dim; 4]>, InferError> {
    let ba = &a.dims()[..a.rank() - tail];
    let bb = &b.dims()[..b.rank() - tail];
    let rank = ba.len().max(bb.len());
    let mut out: Vec<Dim> = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = ba.len().checked_sub(i + 1).map(|j| &ba[j]);
        let db = bb.len().checked_sub(i + 1).map(|j| &bb[j]);
        out.push(broadcast_dim(cx, da, db)?);
    }
    out.reverse();
    Ok(out.into_iter().collect())
}

fn axis_from_right(s: &Shape, i: usize) -> Option<&Dim> {
    s.rank().checked_sub(i + 1).map(|j| s.dim(j))
}

fn broadcast_dim(cx: &Ctx, a: Option<&Dim>, b: Option<&Dim>) -> Result<Dim, InferError> {
    match (a, b) {
        (Some(a), Some(b)) => {
            if a == b {
                Ok(a.clone())
            } else if a.as_static() == Some(1) {
                Ok(b.clone())
            } else if b.as_static() == Some(1) {
                Ok(a.clone())
            } else {
                Err(cx.err(format!("cannot broadcast dims {} and {}", a, b)))
            }
        }
        (Some(d), None) | (None, Some(d)) => Ok(d.clone()),
        (None, None) => unreachable!("broadcast index past both shapes"),
    }
}
