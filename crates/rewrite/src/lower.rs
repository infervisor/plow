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
    let (body, var) = lower_nodes(g)?;
    let out = *g.outputs.last().ok_or(LowerError::NoOutput)?;
    let root = expr_of(g, out, &var)?;
    Ok((body, root))
}

/// [`lower`] with one root per graph output, in output order.
pub fn lower_outputs(g: &Graph) -> Result<(String, Vec<String>), LowerError> {
    let (body, var) = lower_nodes(g)?;
    if g.outputs.is_empty() {
        return Err(LowerError::NoOutput);
    }
    let roots = g
        .outputs
        .iter()
        .map(|&out| expr_of(g, out, &var))
        .collect::<Result<_, _>>()?;
    Ok((body, roots))
}

fn lower_nodes(g: &Graph) -> Result<(String, HashMap<TensorId, String>), LowerError> {
    let mut body = String::new();
    let mut var: HashMap<TensorId, String> = HashMap::new();

    for (i, node) in g.nodes.iter().enumerate() {
        let v = format!("n{i}");
        let term = term_for(g, &node.op, &node.inputs, &var)?;
        let _ = writeln!(body, "(let {v} {term})");
        var.insert(node.output, v);
    }
    Ok((body, var))
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
        Op::RmsNorm { eps } if inputs.len() == 1 => {
            format!("(UnitRmsNorm {} {})", e(0)?, f64lit(*eps))
        }
        Op::RmsNorm { eps } => format!("(RmsNorm {} {} {})", e(0)?, e(1)?, f64lit(*eps)),
        Op::RmsNormZeroCentered { eps } => {
            format!("(ZeroCenteredRmsNorm {} {} {})", e(0)?, e(1)?, f64lit(*eps))
        }
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
        Op::Rope {
            dim,
            theta,
            frequency_dim,
            ..
        } if frequency_dim != dim => format!(
            "(ProportionalRope {} {} {} {})",
            e(0)?,
            *dim,
            f64lit(*theta),
            *frequency_dim
        ),
        Op::Rope { dim, theta, .. } => format!("(Rope {} {} {})", e(0)?, *dim, f64lit(*theta)),
        Op::Attention {
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            sliding_window,
            logit_softcap,
        } => {
            // Serialize the attention config into one deterministic opaque
            // token so differently-configured attentions stay distinct e-nodes
            // (schema.egg: attributes ride along as opaque tokens).
            let cfg = format!(
                "heads={num_heads};kv={num_kv_heads};hd={head_dim};causal={};win={};cap={}",
                *causal as u8,
                sliding_window.map(|w| w.to_string()).unwrap_or_default(),
                logit_softcap.map(|c| f64lit(c)).unwrap_or_default(),
            );
            format!("(Attention {} {} {} {})", e(0)?, e(1)?, e(2)?, quote(&cfg))
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
        Op::MoeRouter {
            num_experts,
            top_k,
            group: None,
            ..
        } => {
            format!("(MoeRouter {} {} {} {})", e(0)?, e(1)?, num_experts, top_k)
        }
        // Every grouped router WITH a correction bias lowers here, whatever it
        // scores with. Matching only `Sigmoid` sent the others to
        // `MoeRouterGrouped`, which takes no bias operand — so `e(2)`, the
        // `e_score_correction_bias` weight leaf, was dropped from the term and
        // with it from the manifest.
        Op::MoeRouter {
            num_experts,
            top_k,
            group: Some(group),
            scoring,
            norm_topk,
            route_scale,
            correction_bias: true,
        } => format!(
            "(MoeRouterNoAux {} {} {} {} {} {} {} {} {} {})",
            e(0)?,
            e(1)?,
            e(2)?,
            quote(moe_scoring(*scoring)),
            num_experts,
            top_k,
            group.n_group,
            group.topk_group,
            i64::from(*norm_topk),
            f64lit(*route_scale),
        ),
        Op::MoeRouter {
            num_experts,
            top_k,
            group: Some(group),
            ..
        } => format!(
            "(MoeRouterGrouped {} {} {} {} {} {})",
            e(0)?,
            e(1)?,
            num_experts,
            top_k,
            group.n_group,
            group.topk_group
        ),
        Op::MoeExperts {
            num_experts,
            top_k,
            intermediate_size,
            block_fp8,
        } => format!(
            "(MoeExperts {} {} {} {} {} {})",
            e(0)?,
            e(1)?,
            num_experts,
            top_k,
            intermediate_size,
            i64::from(*block_fp8),
        ),
        Op::DsaIndexer {
            num_heads,
            head_dim,
            rope_dim,
            top_k,
            theta,
        } => format!(
            "(DsaIndexer {} {} {} {} {} {} {} {})",
            e(0)?,
            e(1)?,
            e(2)?,
            e(3)?,
            e(4)?,
            e(5)?,
            e(6)?,
            quote(&format!(
                "heads={num_heads};hd={head_dim};rope={rope_dim};topk={top_k};theta={}",
                f64lit(*theta)
            ))
        ),
        Op::DsaAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            top_k,
        } => format!(
            "(DsaAttention {} {} {} {} {})",
            e(0)?,
            e(1)?,
            e(2)?,
            e(3)?,
            quote(&format!(
                "heads={num_heads};kv={num_kv_heads};hd={head_dim};topk={top_k}"
            ))
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
                nn_graph::op::LinearAttnKind::QwenGatedDelta => "qwen_gated_delta",
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
        // Variable snapshots lower to a cons chain; both checkpoint weights remain leaves.
        Op::HcMixes {
            hc_mult,
            sinkhorn_iters,
            eps,
        } => format!(
            "(HcMixes {} {} {} {} {} {} {})",
            e(0)?,
            e(1)?,
            e(2)?,
            e(3)?,
            hc_mult,
            sinkhorn_iters,
            f64lit(*eps)
        ),
        Op::HcPre { hc_mult } => format!("(HcPre {} {} {})", e(0)?, e(1)?, hc_mult),
        Op::HcPost { hc_mult } => format!(
            "(HcPost {} {} {} {} {})",
            e(0)?,
            e(1)?,
            e(2)?,
            e(3)?,
            hc_mult
        ),
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
        ActKind::Tanh => "tanh",
        ActKind::Relu => "relu",
        ActKind::Sigmoid => "sigmoid",
        ActKind::QuickGelu => "quick_gelu",
    }
}

fn moe_scoring(s: nn_graph::op::MoeScoring) -> &'static str {
    match s {
        nn_graph::op::MoeScoring::Softmax => "softmax",
        nn_graph::op::MoeScoring::Sigmoid => "sigmoid",
        nn_graph::op::MoeScoring::SqrtSoftplus => "sqrtsoftplus",
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

#[cfg(test)]
mod tests {
    use nn_graph::op::{MoeGroups, MoeScoring};
    use nn_graph::{DType, Nn};

    /// Lower a one-router graph scored with `scoring`.
    fn router_term(scoring: MoeScoring) -> String {
        let mut nn = Nn::new(DType::BF16, DType::BF16);
        let b = nn.sym("B");
        let s = nn.sym("S");
        let ids = nn.input("input_ids", nn.shape([b, s]), DType::I32);
        let x = nn.embedding("model.embed_tokens", ids, 128, 64);
        let routes = nn.moe_router_noaux(
            "model.layers.0.mlp.gate",
            x,
            64,
            8,
            2,
            MoeGroups {
                n_group: 1,
                topk_group: 1,
            },
            true,
            1.5,
            scoring,
        );
        nn.mark_output(routes);
        let g = nn.finish();
        super::lower(&g).expect("lower").0
    }

    /// Three scoring functions, three distinct terms. A sqrtsoftplus router did
    /// not reach `MoeRouterNoAux` at all before it carried the scoring: the arm
    /// matched `Sigmoid` alone.
    #[test]
    fn scoring_is_not_sigmoid_by_definition() {
        let softmax = router_term(MoeScoring::Softmax);
        let sigmoid = router_term(MoeScoring::Sigmoid);
        let sqrt = router_term(MoeScoring::SqrtSoftplus);

        assert!(sigmoid.contains("\"sigmoid\""), "{sigmoid}");
        assert!(sqrt.contains("\"sqrtsoftplus\""), "{sqrt}");
        assert!(softmax.contains("\"softmax\""), "{softmax}");
        assert_ne!(softmax, sigmoid);
        assert_ne!(sigmoid, sqrt);
        assert_ne!(softmax, sqrt);
    }

    /// Every scoring keeps the correction-bias leaf. A grouped router that
    /// missed the `Sigmoid` arm used to fall through to `MoeRouterGrouped`,
    /// which has no bias operand -- dropping `e_score_correction_bias` from the
    /// term and with it from the weight manifest.
    #[test]
    fn correction_bias_survives_every_scoring() {
        for scoring in [
            MoeScoring::Softmax,
            MoeScoring::Sigmoid,
            MoeScoring::SqrtSoftplus,
        ] {
            let term = router_term(scoring);
            assert!(term.contains("MoeRouterNoAux"), "{scoring:?}: {term}");
            assert!(
                term.contains("e_score_correction_bias"),
                "{scoring:?} dropped the correction bias: {term}"
            );
        }
    }
}
