//! Ergonomic layer-builder over [`crate::GraphBuilder`].
//!
//! Architecture builders (gemma, deepseek, siglip, qwen) all drive this one
//! helper so the op-emission and weight-naming logic is written once. Every
//! method returns the [`TensorId`] of its result, so layers compose left to
//! right.

use crate::dim::SymId;
use crate::op::{ActKind, EwKind, Op, ReduceKind};
use crate::{DType, Dim, Graph, GraphBuilder, Shape, TensorId};

pub struct Nn {
    g: GraphBuilder,
    /// dtype for activations produced by ops.
    act_dtype: DType,
    /// dtype for declared weights.
    weight_dtype: DType,
}

impl Nn {
    pub fn new(act_dtype: DType, weight_dtype: DType) -> Self {
        Nn {
            g: GraphBuilder::new(),
            act_dtype,
            weight_dtype,
        }
    }

    // ---- dims / symbols ----------------------------------------------------

    pub fn sym(&mut self, name: &str) -> Dim {
        Dim::sym(self.g.symbol(name))
    }

    pub fn sym_id(&mut self, name: &str) -> SymId {
        self.g.symbol(name)
    }

    pub fn shape(&self, dims: impl IntoIterator<Item = Dim>) -> Shape {
        Shape::new(dims)
    }

    // ---- sources -----------------------------------------------------------

    pub fn input(&mut self, name: &str, shape: Shape, dtype: DType) -> TensorId {
        self.g.input(name, shape, dtype)
    }

    /// A weight tensor with the configured weight dtype.
    pub fn param(&mut self, name: &str, dims: impl IntoIterator<Item = Dim>) -> TensorId {
        self.g.weight(name, Shape::new(dims), self.weight_dtype)
    }

    /// A checkpoint parameter whose storage dtype differs from the model's
    /// configured weight dtype.
    pub fn param_dtype(
        &mut self,
        name: &str,
        dims: impl IntoIterator<Item = Dim>,
        dtype: DType,
    ) -> TensorId {
        self.g.weight(name, Shape::new(dims), dtype)
    }

    fn emit(&mut self, op: Op, inputs: Vec<TensorId>) -> TensorId {
        self.g.op(op, inputs, self.act_dtype)
    }

    // ---- blocks ------------------------------------------------------------

    /// Open a structural block (one transformer layer / encoder block); all ops
    /// emitted until [`Nn::end_block`] are tagged with it.
    pub fn begin_block(&mut self, label: &str) {
        self.g.begin_block(label);
    }

    pub fn end_block(&mut self) {
        self.g.end_block();
    }

    // ---- layers ------------------------------------------------------------

    /// `y = x · Wᵀ (+ b)`, with `W` named `{name}.weight` of shape `[out, in]`.
    pub fn linear(
        &mut self,
        name: &str,
        x: TensorId,
        in_f: i64,
        out_f: i64,
        bias: bool,
    ) -> TensorId {
        let w = self.param(
            &format!("{name}.weight"),
            [Dim::stat(out_f), Dim::stat(in_f)],
        );
        let mut inputs = vec![x, w];
        if bias {
            inputs.push(self.param(&format!("{name}.bias"), [Dim::stat(out_f)]));
        }
        self.emit(
            Op::Linear {
                out_features: out_f,
                bias,
            },
            inputs,
        )
    }

    /// `linear` whose weight carries an explicit checkpoint dtype (a quantized
    /// projection in an otherwise bf16 graph).
    pub fn linear_dtype(
        &mut self,
        name: &str,
        x: TensorId,
        in_f: i64,
        out_f: i64,
        bias: bool,
        dtype: DType,
    ) -> TensorId {
        let w = self.param_dtype(
            &format!("{name}.weight"),
            [Dim::stat(out_f), Dim::stat(in_f)],
            dtype,
        );
        let mut inputs = vec![x, w];
        if bias {
            inputs.push(self.param(&format!("{name}.bias"), [Dim::stat(out_f)]));
        }
        self.emit(
            Op::Linear {
                out_features: out_f,
                bias,
            },
            inputs,
        )
    }

    pub fn rmsnorm(&mut self, name: &str, x: TensorId, hidden: i64, eps: f32) -> TensorId {
        let w = self.param(&format!("{name}.weight"), [Dim::stat(hidden)]);
        self.emit(Op::RmsNorm { eps }, vec![x, w])
    }

    /// RMSNorm with no learned gain (DeepSeek-V4 rescales its query heads this
    /// way). Separate from [`Nn::rmsnorm`] because the difference is a
    /// checkpoint tensor that either exists or does not.
    pub fn rmsnorm_weightless(&mut self, x: TensorId, eps: f32) -> TensorId {
        self.emit(Op::RmsNorm { eps }, vec![x])
    }

    pub fn rmsnorm_dtype(
        &mut self,
        name: &str,
        x: TensorId,
        hidden: i64,
        eps: f32,
        dtype: DType,
    ) -> TensorId {
        let w = self.param_dtype(&format!("{name}.weight"), [Dim::stat(hidden)], dtype);
        self.emit(Op::RmsNorm { eps }, vec![x, w])
    }

    pub fn layernorm(&mut self, name: &str, x: TensorId, hidden: i64, eps: f32) -> TensorId {
        let w = self.param(&format!("{name}.weight"), [Dim::stat(hidden)]);
        let b = self.param(&format!("{name}.bias"), [Dim::stat(hidden)]);
        self.emit(Op::LayerNorm { eps }, vec![x, w, b])
    }

    pub fn embedding(&mut self, name: &str, ids: TensorId, vocab: i64, hidden: i64) -> TensorId {
        let table = self.param(
            &format!("{name}.weight"),
            [Dim::stat(vocab), Dim::stat(hidden)],
        );
        self.emit(Op::Embedding, vec![ids, table])
    }

    pub fn act(&mut self, kind: ActKind, x: TensorId) -> TensorId {
        self.emit(Op::Act(kind), vec![x])
    }

    pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        self.emit(Op::Elementwise(EwKind::Add), vec![a, b])
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        self.emit(Op::Elementwise(EwKind::Mul), vec![a, b])
    }

    pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        self.emit(Op::Elementwise(EwKind::Sub), vec![a, b])
    }

    pub fn div(&mut self, a: TensorId, b: TensorId) -> TensorId {
        self.emit(Op::Elementwise(EwKind::Div), vec![a, b])
    }

    pub fn scale(&mut self, x: TensorId, factor: f32) -> TensorId {
        self.emit(Op::Scale(factor), vec![x])
    }

    pub fn rope(&mut self, x: TensorId, dim: u32, theta: f32) -> TensorId {
        self.emit(
            Op::Rope {
                dim,
                theta,
                interleave: false,
                inverse: false,
            },
            vec![x],
        )
    }

    pub fn rope_interleaved(&mut self, x: TensorId, dim: u32, theta: f32) -> TensorId {
        self.emit(
            Op::Rope {
                dim,
                theta,
                interleave: true,
                inverse: false,
            },
            vec![x],
        )
    }

    /// De-rotation by the conjugate angle, as DeepSeek-V4 applies to the rope
    /// lanes of the attention output.
    pub fn rope_inverse(&mut self, x: TensorId, dim: u32, theta: f32) -> TensorId {
        self.emit(
            Op::Rope {
                dim,
                theta,
                interleave: false,
                inverse: true,
            },
            vec![x],
        )
    }

    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        self.emit(Op::MatMul, vec![a, b])
    }

    pub fn softmax(&mut self, x: TensorId, axis: i32) -> TensorId {
        self.emit(Op::Softmax { axis }, vec![x])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &mut self,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        causal: bool,
        sliding_window: Option<u32>,
        logit_softcap: Option<f32>,
    ) -> TensorId {
        self.emit(
            Op::Attention {
                num_heads,
                num_kv_heads,
                head_dim,
                causal,
                sliding_window,
                logit_softcap,
                attn_sink: false,
            },
            vec![q, k, v],
        )
    }

    /// Attention with a learned per-head sink logit in the softmax denominator.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_sink(
        &mut self,
        name: &str,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        causal: bool,
        sliding_window: Option<u32>,
    ) -> TensorId {
        let sink = self.param_dtype(name, [Dim::stat(num_heads as i64)], DType::F32);
        self.emit(
            Op::Attention {
                num_heads,
                num_kv_heads,
                head_dim,
                causal,
                sliding_window,
                logit_softcap: None,
                attn_sink: true,
            },
            vec![q, k, v, sink],
        )
    }

    pub fn reshape(&mut self, x: TensorId, dims: impl IntoIterator<Item = Dim>) -> TensorId {
        self.emit(
            Op::Reshape {
                shape: Shape::new(dims),
            },
            vec![x],
        )
    }

    pub fn transpose(&mut self, x: TensorId, perm: Vec<u32>) -> TensorId {
        self.emit(Op::Transpose { perm }, vec![x])
    }

    pub fn broadcast(&mut self, x: TensorId, dims: impl IntoIterator<Item = Dim>) -> TensorId {
        self.emit(
            Op::Broadcast {
                shape: Shape::new(dims),
            },
            vec![x],
        )
    }

    pub fn concat(&mut self, axis: i32, xs: Vec<TensorId>) -> TensorId {
        self.emit(Op::Concat { axis }, xs)
    }

    /// Static slice (common case).
    pub fn slice(&mut self, x: TensorId, axis: i32, start: i64, len: i64) -> TensorId {
        self.emit(
            Op::Slice {
                axis,
                start: Dim::stat(start),
                len: Dim::stat(len),
            },
            vec![x],
        )
    }

    /// Slice with symbolic start/length (e.g. splitting a joint sequence).
    pub fn slice_dim(&mut self, x: TensorId, axis: i32, start: Dim, len: Dim) -> TensorId {
        self.emit(Op::Slice { axis, start, len }, vec![x])
    }

    pub fn reduce(&mut self, x: TensorId, kind: ReduceKind, axis: i32, keepdim: bool) -> TensorId {
        self.emit(
            Op::Reduce {
                kind,
                axis,
                keepdim,
            },
            vec![x],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &mut self,
        name: &str,
        x: TensorId,
        in_c: i64,
        out_c: i64,
        kernel: (i64, i64),
        stride: (u32, u32),
        padding: (u32, u32),
        bias: bool,
    ) -> TensorId {
        let w = self.param(
            &format!("{name}.weight"),
            [
                Dim::stat(out_c),
                Dim::stat(in_c),
                Dim::stat(kernel.0),
                Dim::stat(kernel.1),
            ],
        );
        let mut inputs = vec![x, w];
        if bias {
            inputs.push(self.param(&format!("{name}.bias"), [Dim::stat(out_c)]));
        }
        self.emit(Op::Conv2d { stride, padding }, inputs)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn conv3d(
        &mut self,
        name: &str,
        x: TensorId,
        in_c: i64,
        out_c: i64,
        kernel: (i64, i64, i64),
        stride: (u32, u32, u32),
        padding: (u32, u32, u32),
        bias: bool,
    ) -> TensorId {
        let w = self.param(
            &format!("{name}.weight"),
            [
                Dim::stat(out_c),
                Dim::stat(in_c),
                Dim::stat(kernel.0),
                Dim::stat(kernel.1),
                Dim::stat(kernel.2),
            ],
        );
        let mut inputs = vec![x, w];
        if bias {
            inputs.push(self.param(&format!("{name}.bias"), [Dim::stat(out_c)]));
        }
        self.emit(Op::Conv3d { stride, padding }, inputs)
    }

    pub fn groupnorm(
        &mut self,
        name: &str,
        x: TensorId,
        channels: i64,
        groups: u32,
        eps: f32,
    ) -> TensorId {
        let w = self.param(&format!("{name}.weight"), [Dim::stat(channels)]);
        let b = self.param(&format!("{name}.bias"), [Dim::stat(channels)]);
        self.emit(Op::GroupNorm { groups, eps }, vec![x, w, b])
    }

    /// MoE gate producing per-token expert logits `[.., num_experts]`.
    pub fn moe_router(
        &mut self,
        name: &str,
        x: TensorId,
        hidden: i64,
        num_experts: u32,
        top_k: u32,
    ) -> TensorId {
        let w = self.param(
            &format!("{name}.weight"),
            [Dim::stat(num_experts as i64), Dim::stat(hidden)],
        );
        self.emit(
            Op::MoeRouter {
                num_experts,
                top_k,
                group: None,
                hash: false,
                select_bias: false,
            },
            vec![x, w],
        )
    }

    /// Flat top-k gate with DeepSeek's per-expert selection bias.
    pub fn moe_router_select_bias(
        &mut self,
        name: &str,
        x: TensorId,
        hidden: i64,
        num_experts: u32,
        top_k: u32,
    ) -> TensorId {
        let w = self.param(
            &format!("{name}.weight"),
            [Dim::stat(num_experts as i64), Dim::stat(hidden)],
        );
        let bias = self.param_dtype(
            &format!("{name}.bias"),
            [Dim::stat(num_experts as i64)],
            DType::F32,
        );
        self.emit(
            Op::MoeRouter {
                num_experts,
                top_k,
                group: None,
                hash: false,
                select_bias: true,
            },
            vec![x, w, bias],
        )
    }

    /// MoE gate whose expert set comes from a frozen `[vocab, top_k]` token-id
    /// table rather than from the scores (DeepSeek-V4's first `n_hash_layers`).
    ///
    /// Its own constructor for the same reason as [`Nn::moe_router_grouped`]:
    /// the scores still produce the combine weights, so a hash layer built as a
    /// scored one looks right and routes every token somewhere else.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_router_hashed(
        &mut self,
        name: &str,
        x: TensorId,
        ids: TensorId,
        hidden: i64,
        vocab: i64,
        num_experts: u32,
        top_k: u32,
    ) -> TensorId {
        let w = self.param(
            &format!("{name}.weight"),
            [Dim::stat(num_experts as i64), Dim::stat(hidden)],
        );
        let table = self.param_dtype(
            &format!("{name}.tid2eid"),
            [Dim::stat(vocab), Dim::stat(top_k as i64)],
            DType::I64,
        );
        self.emit(
            Op::MoeRouter {
                num_experts,
                top_k,
                group: None,
                hash: true,
                select_bias: false,
            },
            vec![x, w, ids, table],
        )
    }

    /// MoE gate with DeepSeek `noaux_tc` group-limited routing.
    ///
    /// Separate from [`Nn::moe_router`] rather than an `Option` parameter on it:
    /// flat top-k and group-limited top-k select DIFFERENT expert sets, so a
    /// caller has to say which one the checkpoint was trained with. An optional
    /// argument invites passing `None` by default, which is how a K3 or
    /// DeepSeek-V3 graph silently becomes a flat-routing model.
    pub fn moe_router_grouped(
        &mut self,
        name: &str,
        x: TensorId,
        hidden: i64,
        num_experts: u32,
        top_k: u32,
        group: crate::op::MoeGroups,
    ) -> TensorId {
        let w = self.param(
            &format!("{name}.weight"),
            [Dim::stat(num_experts as i64), Dim::stat(hidden)],
        );
        self.emit(
            Op::MoeRouter {
                num_experts,
                top_k,
                group: Some(group),
                hash: false,
                select_bias: false,
            },
            vec![x, w],
        )
    }

    /// Depthwise causal 1-D conv over the sequence axis of `[B, S, channels]`.
    pub fn conv1d_depthwise(
        &mut self,
        name: &str,
        x: TensorId,
        channels: i64,
        kernel: u32,
        weight_dtype: DType,
    ) -> TensorId {
        let w = self.param_dtype(
            &format!("{name}.weight"),
            [Dim::stat(channels), Dim::stat(1), Dim::stat(kernel as i64)],
            weight_dtype,
        );
        self.emit(Op::Conv1dDepthwise { kernel }, vec![x, w])
    }

    /// Linear attention with a carried recurrent state (Kimi delta rule).
    ///
    /// `beta` is `[B, S, heads]` — one write-strength scalar per token and head.
    pub fn linear_attention(
        &mut self,
        kind: crate::op::LinearAttnKind,
        q: TensorId,
        k: TensorId,
        v: TensorId,
        gate: TensorId,
        beta: TensorId,
        a_log: TensorId,
        dt_bias: TensorId,
        num_heads: u32,
        head_dim: u32,
    ) -> TensorId {
        self.emit(
            Op::LinearAttention {
                kind,
                num_heads,
                head_dim,
            },
            vec![q, k, v, gate, beta, a_log, dt_bias],
        )
    }

    /// Kimi-K3's `situ` GLU: `situ_a(gate) * softclip(up)`.
    pub fn situ_glu(
        &mut self,
        gate: TensorId,
        up: TensorId,
        beta: f32,
        linear_beta: f32,
    ) -> TensorId {
        self.emit(Op::SituGlu { beta, linear_beta }, vec![gate, up])
    }

    /// Kimi-K3's `AttnRes` block residual: a softmax mix over the running
    /// prefix sum and the snapshot stack, in place of a plain residual add.
    pub fn block_residual(
        &mut self,
        name: &str,
        prefix: TensorId,
        snapshots: &[TensorId],
        hidden: i64,
        max_snapshots: u32,
    ) -> TensorId {
        let norm = self.param(&format!("{name}_norm.weight"), [Dim::stat(hidden)]);
        let proj = self.param(
            &format!("{name}_proj.weight"),
            [Dim::stat(1), Dim::stat(hidden)],
        );
        let mut ins = Vec::with_capacity(snapshots.len() + 3);
        ins.push(prefix);
        ins.extend_from_slice(snapshots);
        ins.push(norm);
        ins.push(proj);
        self.emit(Op::BlockResidual { max_snapshots }, ins)
    }

    // ---- DeepSeek-V4 hyper-connections -------------------------------------

    /// The three weights a hyper-connection mix reads. `rows` is the projection
    /// height: `(2 + hc_mult) * hc_mult` per layer, `hc_mult` at the head.
    fn hc_params(
        &mut self,
        name: &str,
        rows: i64,
        hc_mult: u32,
        hidden: i64,
        scale_len: i64,
    ) -> [TensorId; 3] {
        let f = self.param_dtype(
            &format!("{name}_fn"),
            [Dim::stat(rows), Dim::stat(hc_mult as i64 * hidden)],
            DType::F32,
        );
        let scale = self.param_dtype(&format!("{name}_scale"), [Dim::stat(scale_len)], DType::F32);
        let base = self.param_dtype(&format!("{name}_base"), [Dim::stat(rows)], DType::F32);
        [f, scale, base]
    }

    /// Collapse `[.., hc_mult, hidden]` residual streams to `[.., hidden]`.
    pub fn hc_reduce(
        &mut self,
        name: &str,
        x: TensorId,
        hc_mult: u32,
        hidden: i64,
        sinkhorn_iters: u32,
        eps: f32,
    ) -> TensorId {
        let rows = (2 + hc_mult as i64) * hc_mult as i64;
        let [f, scale, base] = self.hc_params(name, rows, hc_mult, hidden, 3);
        self.emit(
            Op::HcReduce {
                hc_mult,
                mode: crate::op::HcMode::Sinkhorn {
                    iters: sinkhorn_iters,
                },
                eps,
            },
            vec![x, f, scale, base],
        )
    }

    /// The final, un-normalized reduce that feeds the lm_head.
    pub fn hc_reduce_head(
        &mut self,
        name: &str,
        x: TensorId,
        hc_mult: u32,
        hidden: i64,
        eps: f32,
    ) -> TensorId {
        let [f, scale, base] = self.hc_params(name, hc_mult as i64, hc_mult, hidden, 1);
        self.emit(
            Op::HcReduce {
                hc_mult,
                mode: crate::op::HcMode::SigmoidGate,
                eps,
            },
            vec![x, f, scale, base],
        )
    }

    /// Write `branch` back into the `residual` streams. `name` must match the
    /// [`Nn::hc_reduce`] it pairs with — both read the same three weights.
    #[allow(clippy::too_many_arguments)]
    pub fn hc_expand(
        &mut self,
        name: &str,
        branch: TensorId,
        residual: TensorId,
        hc_mult: u32,
        hidden: i64,
        sinkhorn_iters: u32,
        eps: f32,
    ) -> TensorId {
        let rows = (2 + hc_mult as i64) * hc_mult as i64;
        let [f, scale, base] = self.hc_params(name, rows, hc_mult, hidden, 3);
        self.emit(
            Op::HcExpand {
                hc_mult,
                sinkhorn_iters,
                eps,
            },
            vec![branch, residual, f, scale, base],
        )
    }

    // ---- DeepSeek-V4 attention/FFN pieces -----------------------------------

    /// Block-diagonal projection over the group axis, from one stacked tensor.
    pub fn grouped_linear(
        &mut self,
        name: &str,
        x: TensorId,
        groups: u32,
        in_f: i64,
        out_f: i64,
        dtype: DType,
    ) -> TensorId {
        let w = self.param_dtype(
            &format!("{name}.weight"),
            [Dim::stat(groups as i64 * out_f), Dim::stat(in_f)],
            dtype,
        );
        self.emit(
            Op::GroupedLinear {
                groups,
                out_features: out_f,
            },
            vec![x, w],
        )
    }

    pub fn clamped_swiglu(&mut self, gate: TensorId, up: TensorId, limit: f32) -> TensorId {
        self.emit(Op::ClampedSwiGlu { limit }, vec![gate, up])
    }

    /// Learned KV pooling over `ratio` consecutive tokens.
    ///
    /// `overlap` layers project twice the width, because each window also
    /// carries the previous window's half.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_compress(
        &mut self,
        name: &str,
        x: TensorId,
        hidden: i64,
        ratio: u32,
        out: i64,
        overlap: bool,
        out_seq: Dim,
    ) -> TensorId {
        let wide = if overlap { 2 * out } else { out };
        let wkv = self.param(
            &format!("{name}.wkv.weight"),
            [Dim::stat(wide), Dim::stat(hidden)],
        );
        let wgate = self.param(
            &format!("{name}.wgate.weight"),
            [Dim::stat(wide), Dim::stat(hidden)],
        );
        let ape = self.param_dtype(
            &format!("{name}.ape"),
            [Dim::stat(ratio as i64), Dim::stat(wide)],
            DType::F32,
        );
        let norm = self.param(&format!("{name}.norm.weight"), [Dim::stat(out)]);
        self.emit(
            Op::KvCompress {
                ratio,
                overlap,
                out_seq,
            },
            vec![x, wkv, wgate, ape, norm],
        )
    }

    // ---- finish ------------------------------------------------------------

    pub fn mark_output(&mut self, id: TensorId) {
        self.g.output(id);
    }

    pub fn finish(self) -> Graph {
        self.g.finish()
    }
}
