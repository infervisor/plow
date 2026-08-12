//! Lower an `nn_graph::Graph` into egglog source: one `(let nN …)` per node,
//! leaves inlined as `(Input "name")` / `(Weight "name")`. Returns the body and
//! the root variable to extract.

use nn_graph::op::{ActKind, EwKind, ReduceKind};
use nn_graph::{Graph, Op, Origin, TensorId};
use std::collections::HashMap;
use std::fmt::Write;

#[derive(thiserror::Error, Debug)]
pub enum LowerError {
    #[error("graph has no output to extract")]
    NoOutput,
    #[error("tensor `{name}` (id {id}) is node-produced but its producer was not lowered")]
    UnmappedTensor { id: u32, name: String },
    /// The op has no term in the egglog signature, so no rule can match it.
    ///
    /// Returned rather than lowered to an opaque placeholder: a term the
    /// ruleset does not know would sit in the e-graph looking rewritable, and
    /// any rule that matched its *inputs* could rewrite across it. Refusing to
    /// lower the graph at all is the honest answer, and the caller
    /// (`report_devblob_egglog`) is advisory and warn-only.
    #[error("op `{op}` has no egglog term; the rewrite pass cannot represent this graph")]
    Unsupported { op: &'static str },
}

/// Returns `(let-bindings, root_var)`.
pub fn lower(g: &Graph) -> Result<(String, String), LowerError> {
    let mut body = String::new();
    let mut var: HashMap<TensorId, String> = HashMap::new();

    for (i, node) in g.nodes.iter().enumerate() {
        let v = format!("n{i}");
        let term = term_for(g, &node.op, &node.inputs, &var)?;
        let _ = writeln!(body, "(let {v} {term})");
        var.insert(node.output, v);
    }

    let out = *g.outputs.last().ok_or(LowerError::NoOutput)?;
    let root = expr_of(g, out, &var)?;
    Ok((body, root))
}

/// Egglog expression for a tensor: a leaf constructor, or the node's `let` var.
fn expr_of(g: &Graph, id: TensorId, var: &HashMap<TensorId, String>) -> Result<String, LowerError> {
    let t = g.tensor(id);
    let name = t.name.as_deref().unwrap_or("?");
    Ok(match t.origin {
        Origin::Input => format!("(Input {})", quote(name)),
        Origin::Weight => format!("(Weight {})", quote(name)),
        Origin::Node(_) => var.get(&id).cloned().ok_or_else(|| {
            // No leaf constructor may stand in here: `(Input "?")` would alias
            // every unmapped tensor to one e-node. Fail loudly instead.
            LowerError::UnmappedTensor {
                id: id.0,
                name: t.name.clone().unwrap_or_else(|| format!("t{}", id.0)),
            }
        })?,
    })
}

fn term_for(
    g: &Graph,
    op: &Op,
    inputs: &[TensorId],
    var: &HashMap<TensorId, String>,
) -> Result<String, LowerError> {
    let e = |i: usize| expr_of(g, inputs[i], var);
    Ok(match op {
        Op::Embedding => format!("(Embedding {} {})", e(0)?, e(1)?),
        Op::Scale(f) => format!("(Scale {} {})", e(0)?, f64lit(*f)),
        // Two inputs is the gained form; one is V4's per-head query rescale,
        // which has no checkpoint tensor to name.
        Op::RmsNorm { eps } if inputs.len() == 1 => {
            format!("(RmsNormNoGain {} {})", e(0)?, f64lit(*eps))
        }
        Op::RmsNorm { eps } => format!("(RmsNorm {} {} {})", e(0)?, e(1)?, f64lit(*eps)),
        Op::LayerNorm { eps } => {
            format!("(LayerNorm {} {} {} {})", e(0)?, e(1)?, e(2)?, f64lit(*eps))
        }
        Op::Linear { out_features, bias } => {
            if *bias {
                format!(
                    "(LinearBias {} {} {} {})",
                    e(0)?,
                    e(1)?,
                    e(2)?,
                    out_features
                )
            } else {
                format!("(Linear {} {} {})", e(0)?, e(1)?, out_features)
            }
        }
        Op::MatMul => format!("(MatMul {} {})", e(0)?, e(1)?),
        Op::Reshape { shape } => format!("(Reshape {} {})", e(0)?, quote(&shape.to_string())),
        Op::Transpose { perm } => {
            let tok = perm
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(",");
            format!("(Transpose {} {})", e(0)?, quote(&tok))
        }
        // A de-rotation is not the rotation: they must not share an e-node, so
        // the direction rides in the term rather than being dropped.
        Op::Rope {
            dim,
            theta,
            inverse: false,
            ..
        } => format!("(Rope {} {} {})", e(0)?, *dim, f64lit(*theta)),
        Op::Rope {
            dim,
            theta,
            inverse: true,
            ..
        } => format!("(RopeInverse {} {} {})", e(0)?, *dim, f64lit(*theta)),
        Op::Attention {
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            sliding_window,
            logit_softcap,
            attn_sink,
        } => {
            // Serialize the attention config into one deterministic opaque
            // token so differently-configured attentions stay distinct e-nodes
            // (schema.egg: attributes ride along as opaque tokens).
            let cfg = format!(
                "heads={num_heads};kv={num_kv_heads};hd={head_dim};causal={};win={};cap={};sink={}",
                *causal as u8,
                sliding_window.map(|w| w.to_string()).unwrap_or_default(),
                logit_softcap.map(|c| f64lit(c)).unwrap_or_default(),
                *attn_sink as u8,
            );
            // The sink is a weight leaf and the mask is a whole subgraph;
            // neither fits in the config token, so each combination is its own
            // constructor. Dropping either would orphan its operands.
            match (*attn_sink, inputs.len()) {
                (false, 3) => {
                    format!("(Attention {} {} {} {})", e(0)?, e(1)?, e(2)?, quote(&cfg))
                }
                (true, 4) => format!(
                    "(AttentionSink {} {} {} {} {})",
                    e(0)?,
                    e(1)?,
                    e(2)?,
                    e(3)?,
                    quote(&cfg)
                ),
                (true, 5) => format!(
                    "(AttentionSinkMask {} {} {} {} {} {})",
                    e(0)?,
                    e(1)?,
                    e(2)?,
                    e(3)?,
                    e(4)?,
                    quote(&cfg)
                ),
                // A masked attention with no sink has no term yet, and no model
                // in the tree builds one. Refused rather than silently dropped.
                _ => {
                    return Err(LowerError::Unsupported {
                        op: "attention (masked, no sink)",
                    })
                }
            }
        }
        Op::Elementwise(k) => format!("(Ew {} {} {})", quote(ew(*k)), e(0)?, e(1)?),
        Op::Act(k) => format!("(Act {} {})", quote(act(*k)), e(0)?),
        Op::Softmax { axis } => format!("(Softmax {} {})", e(0)?, axis),
        Op::Concat { axis } => {
            // n-ary concat → binary chain: (Concat axis a (Concat axis b c))
            let n = inputs.len();
            assert!(n >= 2);
            let mut term = e(n - 1)?;
            for i in (0..n - 1).rev() {
                term = format!("(Concat {} {} {})", axis, e(i)?, term);
            }
            term
        }
        Op::Slice { axis, start, len } => {
            // Start/len are string tokens (mirroring Reshape's shape-token):
            // `Dim`'s canonical Display gives concrete dims their integer text
            // and distinct symbolic dims distinct tokens, so two different
            // symbolic slices of one tensor never hash-cons together.
            format!(
                "(Slice {} {} {} {})",
                e(0)?,
                axis,
                quote(&start.to_string()),
                quote(&len.to_string())
            )
        }
        Op::Broadcast { shape } => {
            format!("(Broadcast {} {})", e(0)?, quote(&shape.to_string()))
        }
        Op::Reduce { kind, axis, .. } => {
            format!("(Reduce {} {} {})", quote(reduce(*kind)), e(0)?, axis)
        }
        Op::Conv2d { stride, padding } => {
            let s_tok = format!("{},{}", stride.0, stride.1);
            let p_tok = format!("{},{}", padding.0, padding.1);
            if inputs.len() >= 3 {
                format!(
                    "(Conv2dBias {} {} {} {} {})",
                    e(0)?,
                    e(1)?,
                    e(2)?,
                    quote(&s_tok),
                    quote(&p_tok)
                )
            } else {
                format!(
                    "(Conv2d {} {} {} {})",
                    e(0)?,
                    e(1)?,
                    quote(&s_tok),
                    quote(&p_tok)
                )
            }
        }
        Op::Conv3d { stride, padding } => {
            let s_tok = format!("{},{},{}", stride.0, stride.1, stride.2);
            let p_tok = format!("{},{},{}", padding.0, padding.1, padding.2);
            if inputs.len() >= 3 {
                format!(
                    "(Conv3dBias {} {} {} {} {})",
                    e(0)?,
                    e(1)?,
                    e(2)?,
                    quote(&s_tok),
                    quote(&p_tok)
                )
            } else {
                format!(
                    "(Conv3d {} {} {} {})",
                    e(0)?,
                    e(1)?,
                    quote(&s_tok),
                    quote(&p_tok)
                )
            }
        }
        Op::GroupNorm { groups, eps } => {
            format!(
                "(GroupNorm {} {} {} {} {})",
                e(0)?,
                e(1)?,
                e(2)?,
                groups,
                f64lit(*eps)
            )
        }
        // Group-limited routing selects a DIFFERENT expert set than flat top-k,
        // and the egglog term carries only `num_experts`/`top_k` — so a grouped
        // router lowered through it would be indistinguishable from a flat one.
        // Refused rather than approximated.
        Op::MoeRouter {
            num_experts,
            top_k,
            group: None,
            hash: false,
            select_bias: false,
        } => {
            format!("(MoeRouter {} {} {} {})", e(0)?, e(1)?, num_experts, top_k)
        }
        Op::MoeRouter { group: Some(_), .. } => {
            return Err(LowerError::Unsupported {
                op: "moe_router (group-limited)",
            })
        }
        // Distinct constructors, not extra operands on `MoeRouter`: both pick a
        // different expert set, so sharing an e-node with flat top-k would let a
        // rule rewrite one into the other.
        Op::MoeRouter {
            num_experts,
            top_k,
            hash: true,
            ..
        } => format!(
            "(MoeRouterHashed {} {} {} {} {} {})",
            e(0)?,
            e(1)?,
            e(2)?,
            e(3)?,
            num_experts,
            top_k
        ),
        Op::MoeRouter {
            num_experts,
            top_k,
            select_bias: true,
            ..
        } => format!(
            "(MoeRouterBias {} {} {} {} {})",
            e(0)?,
            e(1)?,
            e(2)?,
            num_experts,
            top_k
        ),
        // --- Kimi-K3 ---
        Op::Conv1dDepthwise { kernel } => {
            format!("(Conv1dDepthwise {} {} {})", e(0)?, e(1)?, kernel)
        }
        Op::SituGlu { beta, linear_beta } => format!(
            "(SituGlu {} {} {} {})",
            e(0)?,
            e(1)?,
            f64lit(*beta),
            f64lit(*linear_beta)
        ),
        Op::LinearAttention {
            kind,
            num_heads,
            head_dim,
        } => {
            // The recurrent state is a runtime resource, not an edge — the same
            // convention `Op::Attention` uses for the KV cache. The kind rides as
            // a token so a future second recurrence stays a distinct e-node.
            let k = match kind {
                nn_graph::op::LinearAttnKind::KimiDelta => "kimi_delta",
            };
            format!(
                "(LinearAttention {} {} {} {} {} {} {} {} {} {})",
                e(0)?,
                e(1)?,
                e(2)?,
                e(3)?,
                e(4)?,
                e(5)?,
                e(6)?,
                quote(k),
                num_heads,
                head_dim
            )
        }
        // --- DeepSeek-V4 ---
        Op::HcReduce { hc_mult, mode, eps } => {
            // The mode decides both the projection height and whether a Sinkhorn
            // normalization runs, so it rides as a token rather than being
            // flattened into the iteration count.
            let m = match mode {
                nn_graph::op::HcMode::Sinkhorn { iters } => format!("sinkhorn:{iters}"),
                nn_graph::op::HcMode::SigmoidGate => "sigmoid".to_string(),
            };
            format!(
                "(HcReduce {} {} {} {} {} {} {})",
                e(0)?,
                e(1)?,
                e(2)?,
                e(3)?,
                hc_mult,
                quote(&m),
                f64lit(*eps)
            )
        }
        Op::HcExpand {
            hc_mult,
            sinkhorn_iters,
            eps,
        } => format!(
            "(HcExpand {} {} {} {} {} {} {} {})",
            e(0)?,
            e(1)?,
            e(2)?,
            e(3)?,
            e(4)?,
            hc_mult,
            sinkhorn_iters,
            f64lit(*eps)
        ),
        Op::GroupedLinear {
            groups,
            out_features,
        } => format!(
            "(GroupedLinear {} {} {} {})",
            e(0)?,
            e(1)?,
            groups,
            out_features
        ),
        Op::ClampedSwiGlu { limit } => {
            format!("(ClampedSwiGlu {} {} {})", e(0)?, e(1)?, f64lit(*limit))
        }
        Op::KvCompress {
            ratio,
            overlap,
            out_seq,
        } => format!(
            "(KvCompress {} {} {} {} {} {} {} {})",
            e(0)?,
            e(1)?,
            e(2)?,
            e(3)?,
            e(4)?,
            ratio,
            quote(&format!("{overlap}")),
            quote(&format!("{out_seq}"))
        ),
        // Variable snapshots lower to a cons chain; both checkpoint weights remain leaves.
        Op::BlockResidual { max_snapshots } => {
            let norm = inputs.len() - 2;
            let proj = inputs.len() - 1;
            let mut chain = String::from("(SnapNil)");
            for i in (1..norm).rev() {
                chain = format!("(SnapCons {} {})", expr_of(g, inputs[i], var)?, chain);
            }
            format!(
                "(BlockResidual {} {} {} {} {})",
                e(0)?,
                chain,
                expr_of(g, inputs[norm], var)?,
                expr_of(g, inputs[proj], var)?,
                max_snapshots
            )
        }
    })
}

fn ew(k: EwKind) -> &'static str {
    match k {
        EwKind::Add => "add",
        EwKind::Sub => "sub",
        EwKind::Mul => "mul",
        EwKind::Div => "div",
    }
}

fn act(k: ActKind) -> &'static str {
    match k {
        ActKind::Silu => "silu",
        ActKind::Gelu => "gelu",
        ActKind::GeluTanh => "gelu_tanh",
        ActKind::Relu => "relu",
        ActKind::Sigmoid => "sigmoid",
        ActKind::QuickGelu => "quick_gelu",
    }
}

fn reduce(k: ReduceKind) -> &'static str {
    match k {
        ReduceKind::Mean => "mean",
        ReduceKind::Sum => "sum",
        ReduceKind::Max => "max",
    }
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "'"))
}

/// Format an f32 as a valid egglog `f64` literal (always with a decimal point).
fn f64lit(x: f32) -> String {
    let mut s = format!("{}", x as f64);
    if !s.contains('.') && !s.contains('e') && !s.contains('E') {
        s.push_str(".0");
    }
    s
}
