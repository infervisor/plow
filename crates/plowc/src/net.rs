//! A small, plow-native network definition (the `--net <file.json>` front-end).
//!
//! HuggingFace models come in through `nn_graph::hub`; this is the
//! offline, hand-written alternative: a JSON list of ops the driver chains into
//! a [`LayerPlan`]. Each op consumes the previous op's output plus its own
//! weight, so authoring a network is just listing layers in order. The plan is
//! rebuilt per shape bucket — `M` (rows) and the attention sequence come from
//! the bucket, everything else from the config — so one `--net` file compiles
//! across the whole bucket ladder, exactly like a real model.

use costmodel::{AttnShape, GemmShape, RowShape};
use rewrite::{LayerPlan, OpKind, OpSpec};
use schedule::ShapeBucket;
use serde::Deserialize;

/// A simple feed-forward network: a starting feature width and a list of ops.
#[derive(Deserialize, Clone, Debug)]
pub struct NetConfig {
    pub name: String,
    /// Feature width fed to the first op (the model's hidden size).
    pub hidden: i64,
    pub ops: Vec<NetOp>,
}

/// One op in a [`NetConfig`]. The tag is the `"op"` field in JSON.
#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum NetOp {
    /// Row reduction (RMSNorm-style) over `feat` (defaults to the running width).
    Norm { feat: Option<i64> },
    /// Linear / matmul to `n` output features from `k` (defaults to the running width).
    Gemm { n: i64, k: Option<i64> },
    /// Pointwise activation over the running width.
    Act,
    /// Attention with `heads` heads of `head_dim` (sequence comes from the bucket).
    Flash { heads: i64, head_dim: i64 },
}

impl NetOp {
    fn label(&self) -> &'static str {
        match self {
            NetOp::Norm { .. } => "norm",
            NetOp::Gemm { .. } => "gemm",
            NetOp::Act => "act",
            NetOp::Flash { .. } => "flash",
        }
    }
}

impl NetConfig {
    /// Reject non-positive dimensions before they become degenerate ops (a 0-wide
    /// GEMM, a 0-head attention) that would panic or divide-by-zero deep in the
    /// cost model / scheduler. Returns a human-readable reason on the first bad op.
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden <= 0 {
            return Err(format!(
                "network `{}`: hidden must be > 0, got {}",
                self.name, self.hidden
            ));
        }
        for (i, op) in self.ops.iter().enumerate() {
            let bad = |what: &str, v: i64| {
                format!(
                    "network `{}` op {i} ({}): {what} must be > 0, got {v}",
                    self.name,
                    op.label()
                )
            };
            match op {
                NetOp::Norm { feat: Some(f) } if *f <= 0 => return Err(bad("feat", *f)),
                NetOp::Gemm { n, k } => {
                    if *n <= 0 {
                        return Err(bad("n", *n));
                    }
                    if let Some(k) = k {
                        if *k <= 0 {
                            return Err(bad("k", *k));
                        }
                    }
                }
                NetOp::Flash { heads, head_dim } => {
                    if *heads <= 0 {
                        return Err(bad("heads", *heads));
                    }
                    if *head_dim <= 0 {
                        return Err(bad("head_dim", *head_dim));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Lower this network to a [`LayerPlan`] specialized to `bucket`.
    pub fn build_plan(&self, bucket: &ShapeBucket) -> LayerPlan {
        let rows = bucket.rows();
        let seq = bucket.attn_seq();
        let mut feat = self.hidden; // running feature width
        let mut prev = "x".to_string(); // running activation tensor name
        let mut ops = Vec::with_capacity(self.ops.len());
        for (i, op) in self.ops.iter().enumerate() {
            let output = format!("t{i}");
            let (inputs, kind) = match op {
                NetOp::Norm { feat: f } => {
                    let fe = f.unwrap_or(feat);
                    feat = fe;
                    (
                        vec![prev.clone(), format!("nw{i}")],
                        OpKind::Row(RowShape {
                            rows,
                            feat: fe,
                            operands: 2,
                            reduce: true,
                        }),
                    )
                }
                NetOp::Gemm { n, k } => {
                    let kk = k.unwrap_or(feat);
                    feat = *n;
                    (
                        vec![prev.clone(), format!("w{i}")],
                        OpKind::Gemm(GemmShape {
                            m: rows,
                            n: *n,
                            k: kk,
                        }),
                    )
                }
                NetOp::Act => (
                    vec![prev.clone()],
                    OpKind::Row(RowShape {
                        rows,
                        feat,
                        operands: 1,
                        reduce: false,
                    }),
                ),
                NetOp::Flash { heads, head_dim } => (
                    vec![prev.clone()],
                    OpKind::Flash(AttnShape {
                        heads: *heads,
                        seq_q: seq,
                        seq_kv: seq,
                        head_dim: *head_dim,
                    }),
                ),
            };
            ops.push(OpSpec {
                name: format!("{}{i}", op.label()),
                inputs,
                output: output.clone(),
                kind,
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            });
            prev = output;
        }
        LayerPlan { ops }
    }
}
