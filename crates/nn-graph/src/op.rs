//! Operator definitions for the frontend IR.
//!
//! Granularity is operator-level (a whole GEMM, a whole attention), matching
//! the design doc's Stage-1 frontend IR. Weights are explicit input tensors
//! (not attributes) so they become DMA-in nodes when the graph is later lowered
//! to tiles. Shape-inference rules for each op live in [`crate::infer`].

/// Pointwise activation functions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActKind {
    Silu,
    Gelu,
    GeluTanh,
    Relu,
    Sigmoid,
    QuickGelu,
}

/// Binary elementwise operations (numpy broadcasting).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum EwKind {
    Add,
    Sub,
    Mul,
    Div,
}

/// Reductions over a single axis.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReduceKind {
    Mean,
    Sum,
    Max,
}

/// The operators of the graph. Each is single-output; multi-output structures
/// (e.g. separate Q/K/V projections) are expressed as separate nodes.
#[derive(Clone, Debug)]
pub enum Op {
    /// `[.., in] · W[out, in] (+ bias[out]) -> [.., out]`.
    /// Inputs: `[x, weight]` or `[x, weight, bias]`.
    Linear { out_features: i64, bias: bool },

    /// Batched matmul `[.., m, k] · [.., k, n] -> [.., m, n]`. Inputs `[a, b]`.
    /// Used for attention score/context products where both operands are
    /// activations.
    MatMul,

    /// RMSNorm over the last axis. Inputs `[x, weight]`. Shape-preserving.
    RmsNorm { eps: f32 },

    /// LayerNorm over the last axis. Inputs `[x, weight, bias]`. Shape-preserving.
    LayerNorm { eps: f32 },

    /// Rotary position embedding over `[.., heads, head_dim]`. Inputs `[x]`.
    /// Shape-preserving. `dim` rotated may be < head_dim (partial RoPE).
    /// `interleave`: when true, pairs (x[0],x[1]), (x[2],x[3])... (GLM-style);
    /// when false, pairs (x[0],x[d/2]), (x[1],x[d/2+1])... (Llama-style).
    /// `inverse`: rotate by the conjugate angle. DeepSeek-V4 rotates the query
    /// and key, then de-rotates the attention OUTPUT's rope lanes by the same
    /// positions, so the two directions must be distinguishable.
    Rope {
        dim: u32,
        theta: f32,
        interleave: bool,
        inverse: bool,
    },

    /// Pointwise activation. Inputs `[x]`. Shape-preserving.
    Act(ActKind),

    /// Binary elementwise with broadcasting. Inputs `[a, b]`.
    Elementwise(EwKind),

    /// Multiply by a compile-time scalar (e.g. Gemma's `sqrt(hidden)` embedding
    /// scale, attention `1/sqrt(d)`). Inputs `[x]`. Shape-preserving.
    Scale(f32),

    /// Softmax along `axis`. Inputs `[x]`. Shape-preserving.
    Softmax { axis: i32 },

    /// Scaled-dot-product / flash attention. Inputs `[q, k, v]`, then the
    /// optional mask, then the optional per-head sink — each present only when
    /// its attribute says so, and always in that order. Output shape is Q with
    /// its last dim replaced by V's last dim (these differ for MLA where
    /// `qk_head_dim != v_head_dim`).
    /// Attributes drive lowering and document the variant.
    Attention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        causal: bool,
        /// Sliding-window width for local attention layers (Gemma), else None.
        sliding_window: Option<u32>,
        /// Tanh logit soft-cap (Gemma2). Documentational for shape inference.
        logit_softcap: Option<f32>,
        /// A learned per-head logit that joins the softmax denominator without
        /// contributing a value row (DeepSeek-V4's `attn_sink`). When true the
        /// last input is that `[heads]` weight. It shifts every probability in
        /// the row, so a graph that dropped it would be plausible and wrong.
        attn_sink: bool,
    },

    /// Token-embedding lookup. Inputs `[ids, table]` with `table: [vocab, H]`.
    /// Output: `ids.shape ++ [H]`.
    Embedding,

    /// 2D convolution, NCHW. Inputs `[x, weight]` or `[x, weight, bias]` with
    /// `weight: [out_c, in_c, kh, kw]`. Used for ViT patch embedding.
    Conv2d {
        stride: (u32, u32),
        padding: (u32, u32),
    },

    /// 3D convolution, NCDHW. Inputs `[x, weight]` or `[x, weight, bias]` with
    /// `weight: [out_c, in_c, kd, kh, kw]`. Used by the Qwen-Image 3D causal
    /// VAE. Spatial dims must be statically known (per shape bucket).
    Conv3d {
        stride: (u32, u32, u32),
        padding: (u32, u32, u32),
    },

    /// GroupNorm over the channel axis (axis 1, NCHW/NCDHW). Inputs
    /// `[x, weight, bias]`. Shape-preserving.
    GroupNorm { groups: u32, eps: f32 },

    /// Reshape to an explicit (possibly symbolic) target shape. Inputs `[x]`.
    Reshape { shape: crate::Shape },

    /// Permute axes. Inputs `[x]`.
    Transpose { perm: Vec<u32> },

    /// Broadcast size-1 axes up to an explicit target shape (numpy `expand`).
    /// Inputs `[x]`. Each input dim must be 1 or already equal the target.
    /// Used e.g. to share MLA's rotary key across attention heads.
    Broadcast { shape: crate::Shape },

    /// Concatenate along `axis`. Inputs `[a, b, ...]` (>= 2).
    Concat { axis: i32 },

    /// Slice `len` elements starting at `start` along `axis`. Inputs `[x]`.
    /// `start`/`len` are symbolic dims (MMDiT splits a joint sequence of
    /// symbolic length back into its text/image parts).
    Slice {
        axis: i32,
        start: crate::Dim,
        len: crate::Dim,
    },

    /// Axis reduction. Inputs `[x]`.
    Reduce {
        kind: ReduceKind,
        axis: i32,
        keepdim: bool,
    },

    /// MoE routing gate: `[.., H] · W[E, H] -> [.., E]` then top-k selection.
    /// Modeled as producing per-token expert logits `[.., num_experts]`.
    /// Inputs `[x, weight]`. The actual dispatch/combine is data-dependent and
    /// handled at runtime via indirection; for the static graph the routed
    /// expert FFN output is shape-equal to its input.
    ///
    /// `group` is DeepSeek's `noaux_tc` group-limited routing, which Kimi-K3
    /// and DeepSeek-V3 use and which is NOT the same expert set as a flat
    /// top-k. See [`MoeGroups`]; `None` is flat top-k over all experts.
    ///
    /// `hash` layers do not select experts from the scores at all: the expert
    /// set is looked up from the token id through a frozen `[vocab, top_k]`
    /// table, and the scores only supply the combine weights. Inputs are then
    /// `[x, weight, ids, tid2eid]`. DeepSeek-V4's first `n_hash_layers` route
    /// this way; scoring them like the rest picks a different expert set for
    /// every token, which is exactly the silent-wrong-model failure the
    /// `group` split above exists to prevent.
    ///
    /// `select_bias` is DeepSeek's per-expert selection bias: it is added to the
    /// scores for the top-k comparison and then dropped, so it changes WHICH
    /// experts run without changing their combine weights. When true it is the
    /// input right after `weight`.
    MoeRouter {
        num_experts: u32,
        top_k: u32,
        group: Option<MoeGroups>,
        hash: bool,
        select_bias: bool,
    },

    /// Depthwise causal 1-D convolution over the sequence axis, `[B, S, C]`.
    /// Inputs `[x, weight]` or `[x, weight, bias]` with
    /// `weight: [C, 1, kernel]`.
    /// Shape-preserving: the reference left-pads by `kernel - 1` so output
    /// length equals input length.
    ///
    /// Kimi-K3's KDA applies one of these to each of q/k/v with `kernel = 4`.
    /// Depthwise (one filter per channel) is the only form in use, so it is the
    /// only form modeled — a dense 1-D conv would need an `[out_c, in_c, k]`
    /// weight and a different rule, and inventing it unused is how a wrong
    /// rule ships untested.
    ///
    /// # The carried conv state is not an input here
    ///
    /// At decode the kernel needs the previous `kernel - 1` positions, which a
    /// server keeps in a per-sequence conv-state buffer. That buffer is a
    /// RUNTIME resource, exactly like the KV cache behind [`Op::Attention`],
    /// and for the same reason it is not a graph edge: it is not part of the
    /// symbolic dataflow and modeling it would make every prefill graph carry
    /// an input no prefill ever reads.
    Conv1dDepthwise { kernel: u32 },

    /// Linear (sub-quadratic) attention with a carried recurrent state.
    ///
    /// Inputs `[q, k, v, gate, beta, A_log, dt_bias]`: q/k/v/gate are rank-4
    /// `[B, S, heads, head_dim]`, beta is `[B, S, heads]`, A_log is `[heads]`,
    /// and dt_bias is `[heads * head_dim]`. Output has q's shape.
    ///
    /// # Why the state is not an input, and why this is not [`Op::Attention`]
    ///
    /// The `[heads, head_dim, head_dim]` recurrent state is the linear-attention
    /// analogue of the KV cache: a runtime resource carried across steps, not a
    /// symbolic graph edge. Modeling it as an edge would also force a second
    /// output (the updated state) and break this enum's single-output
    /// invariant, which is documented above and which the whole shape-inference
    /// pass relies on.
    ///
    /// It is a distinct op rather than an `Attention` variant because there is
    /// no softmax, no score matrix and no causal mask — the causality is in the
    /// recurrence itself. Folding it into `Attention` would make every consumer
    /// that matches on `Attention` silently wrong for a model it has never
    /// seen.
    LinearAttention {
        kind: LinearAttnKind,
        num_heads: u32,
        head_dim: u32,
    },

    /// Kimi-K3's `situ` gated linear unit. Inputs `[gate, up]`, shape-preserving.
    ///
    /// Reference (`modeling_kimi_linear.py`, registered as `ACT2FN["situ"]`):
    ///
    /// ```text
    /// situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    /// up'    = linear_beta * tanh(up / linear_beta)   // skipped when linear_beta <= 0
    /// out    = situ_a * up'
    /// ```
    ///
    /// # Why this is one op and not `Act(..)` followed by `Mul`
    ///
    /// Every other GLU in this IR is `mul(act(gate), up)`, and `situ` is not
    /// that shape: it transforms the UP branch too. Expressing it as an
    /// [`ActKind`] plus a multiply would apply the gate transform and leave
    /// `up` unclipped — at `|up| < linear_beta` that is a small error that
    /// grows with the tail, i.e. plausible output and the wrong model. The
    /// reference registers `situ` as a single activation for the same reason.
    SituGlu { beta: f32, linear_beta: f32 },

    /// Kimi-K3's block residual (`AttnRes`): a softmax mix over the running
    /// prefix sum and up to `max_snapshots` earlier snapshots of it, applied in
    /// place of the plain residual add.
    ///
    /// Inputs `[prefix, snapshot_0, .., snapshot_n, norm_weight, proj_weight]`.
    /// Output is shape-equal to `prefix`.
    /// `attn_res_block_size` layers apart, the block pushes a snapshot and
    /// resets the prefix, so this is what makes a K3 layer structurally unlike
    /// `residual + attn; residual + mlp`. Modeled explicitly because a plain
    /// add is numerically indistinguishable from it at a non-snapshot layer
    /// (measured 3.0e-3 at the block output against 8.1e-1 at the mix itself),
    /// so a graph that used `Add` here would look right and be wrong.
    BlockResidual { max_snapshots: u32 },

    /// DeepSeek-V4 hyper-connection **reduce**: collapse the `hc_mult` parallel
    /// residual streams `[B, S, hc, D]` down to the single `[B, S, D]` a
    /// sub-layer consumes.
    ///
    /// Inputs `[x, hc_fn, hc_scale, hc_base]`; `hc_fn` is `[mix, hc*D]` with
    /// `mix = (2 + hc_mult) * hc_mult` for [`HcMode::Sinkhorn`] and `hc` for
    /// [`HcMode::SigmoidGate`]. The mixing coefficients come from an
    /// RMS-scaled projection of the flattened streams, so the weights are read
    /// by both this op and the matching [`Op::HcExpand`].
    HcReduce {
        hc_mult: u32,
        mode: HcMode,
        eps: f32,
    },

    /// DeepSeek-V4 hyper-connection **expand**: write a sub-layer's output back
    /// into the `hc_mult` residual streams, in place of a residual add.
    ///
    /// Inputs `[branch, residual, hc_fn, hc_scale, hc_base]` where `residual`
    /// is the same `[B, S, hc, D]` the paired [`Op::HcReduce`] consumed; output
    /// is shape-equal to `residual`. The post/combine coefficients are a pure
    /// function of `residual` and these weights, which is why recomputing them
    /// here keeps this enum's single-output invariant without changing what is
    /// computed.
    ///
    /// # Why this is not `Add`
    ///
    /// The stream update is `post ⊗ branch + comb · residual` with a learned,
    /// input-dependent, Sinkhorn-normalized `comb` mixing every stream into
    /// every other. A plain residual add is the special case `hc_mult = 1,
    /// post = comb = 1`; using it for a 4-stream model produces a graph that
    /// type-checks and models a different network — the same reasoning that
    /// made [`Op::BlockResidual`] its own op.
    HcExpand {
        hc_mult: u32,
        sinkhorn_iters: u32,
        eps: f32,
    },

    /// Block-diagonal linear: `groups` independent projections applied to the
    /// matching slice of the input, from ONE checkpoint tensor.
    ///
    /// Inputs `[x, weight]` with `x: [.., groups, in]` and
    /// `weight: [groups * out_features, in]`, viewed as
    /// `[groups, out_features, in]`; output is `[.., groups, out_features]`.
    ///
    /// DeepSeek-V4's `wo_a` is this: 8 groups of attention heads each projected
    /// by their own `[o_lora_rank, in]` block. It is not [`Op::Linear`] — that
    /// is a dense `[groups*out, groups*in]` map, which mixes groups the
    /// reference keeps separate — and it cannot be `groups` separate `Linear`s
    /// either, because the manifest is the loader's contract and the checkpoint
    /// ships exactly one tensor here.
    GroupedLinear { groups: u32, out_features: i64 },

    /// SwiGLU whose two branches are clamped before the product. Inputs
    /// `[gate, up]`, shape-preserving:
    ///
    /// ```text
    /// out = silu(min(gate, limit)) * clamp(up, -limit, limit)
    /// ```
    ///
    /// # Why this is not `Act(Silu)` + `Mul`
    ///
    /// The clamp is one-sided on the gate and two-sided on the up branch, and
    /// it binds only in the tail. Dropping it leaves a graph that agrees with
    /// the reference on almost every token and diverges exactly where the
    /// activation blows up — the failure mode [`Op::SituGlu`] was carved out
    /// for. DeepSeek-V4 trains with `swiglu_limit = 10`.
    ClampedSwiGlu { limit: f32 },

    /// DeepSeek-V4's learned KV compressor: pool every `ratio` consecutive
    /// tokens into one compressed KV entry, `[B, S, D] -> [B, S/ratio, out]`.
    ///
    /// Inputs `[x, wkv, wgate, ape, norm_weight]`. Within a window the gate
    /// scores (plus the per-offset `ape` bias) are softmaxed and used to average
    /// the projected KV; the result is RMS-normed. When `overlap` (the
    /// `ratio == 4` layers) each window also carries the previous window's
    /// half, which is why those layers' `wkv`/`wgate`/`ape` are twice as wide.
    ///
    /// # Why this is its own op
    ///
    /// It is the only op in the IR that changes the sequence rate, and the
    /// overlapped form builds its windows by value-padding (`0` for KV, `-inf`
    /// for the scores) — there is no pad op to express that with, and modelling
    /// an overlapped layer as the non-overlapped one silently halves its
    /// receptive field.
    ///
    /// `out_seq` names the compressed length. [`crate::Dim`] is a polynomial
    /// with integer coefficients, so `S / ratio` is not a dim it can hold; the
    /// builder passes a symbol (`Sc4`, `Sc128`) instead, one per rate, and
    /// inference checks the relation whenever the input length is static.
    KvCompress {
        ratio: u32,
        overlap: bool,
        out_seq: crate::Dim,
    },
}

/// How an [`Op::HcReduce`] turns its projection into per-stream weights.
///
/// The two modes are not a parameter of one formula: the per-layer reduce
/// Sinkhorn-normalizes a `(2 + hc_mult) * hc_mult` block (which also yields the
/// post/combine halves [`Op::HcExpand`] needs), while the final head reduce is
/// a bare sigmoid gate over `hc_mult` coefficients with no normalization and no
/// expand partner.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HcMode {
    /// Per-layer: `iters` rounds of Sinkhorn row/column normalization.
    Sinkhorn { iters: u32 },
    /// Final head: `sigmoid(mix * scale + base) + eps`, no normalization.
    SigmoidGate,
}

/// DeepSeek `noaux_tc` group-limited expert routing.
///
/// The experts are partitioned into `n_group` contiguous equal groups, each
/// scored by the sum of its top-2 biased scores; only the best `topk_group`
/// groups are kept and the top-k runs inside those. `topk_group >= n_group` is
/// the identity, and so is `n_group <= 1` — which is why GLM-5.2 (one group)
/// matches a flat-top-k reference and Kimi-K3 (eight) does not.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoeGroups {
    pub n_group: u32,
    pub topk_group: u32,
}

/// Which linear-attention recurrence [`Op::LinearAttention`] carries.
///
/// One variant today. It exists as an enum rather than being implied by the op
/// because the recurrences in this family (delta rule, gated delta, plain
/// linear attention) differ in what they do with `beta` and `gate`, and a
/// consumer that assumed the wrong one would produce fluent, wrong output —
/// the failure mode this IR is meant to make impossible to reach by accident.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LinearAttnKind {
    /// Kimi Delta Attention: a gated delta rule with a low-rank forget gate.
    /// `state <- state * decay(gate) + beta * (v - state·k) ⊗ k`.
    KimiDelta,
}

impl Op {
    /// Human-readable opcode name for dumps/visualization.
    pub fn name(&self) -> &'static str {
        match self {
            Op::Linear { .. } => "linear",
            Op::MatMul => "matmul",
            Op::RmsNorm { .. } => "rmsnorm",
            Op::LayerNorm { .. } => "layernorm",
            Op::Rope { .. } => "rope",
            Op::Act(_) => "act",
            Op::Elementwise(_) => "elementwise",
            Op::Scale(_) => "scale",
            Op::Softmax { .. } => "softmax",
            Op::Attention { .. } => "attention",
            Op::Embedding => "embedding",
            Op::Conv2d { .. } => "conv2d",
            Op::Conv3d { .. } => "conv3d",
            Op::GroupNorm { .. } => "groupnorm",
            Op::Reshape { .. } => "reshape",
            Op::Transpose { .. } => "transpose",
            Op::Broadcast { .. } => "broadcast",
            Op::Concat { .. } => "concat",
            Op::Slice { .. } => "slice",
            Op::Reduce { .. } => "reduce",
            Op::MoeRouter { .. } => "moe_router",
            Op::Conv1dDepthwise { .. } => "conv1d_depthwise",
            Op::LinearAttention { .. } => "linear_attention",
            Op::SituGlu { .. } => "situ_glu",
            Op::BlockResidual { .. } => "block_residual",
            Op::HcReduce { .. } => "hc_reduce",
            Op::HcExpand { .. } => "hc_expand",
            Op::GroupedLinear { .. } => "grouped_linear",
            Op::ClampedSwiGlu { .. } => "clamped_swiglu",
            Op::KvCompress { .. } => "kv_compress",
        }
    }
}
