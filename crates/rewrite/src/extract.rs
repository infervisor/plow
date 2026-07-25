//! Run egglog on the lowered program and parse the extracted term into a
//! hash-consed `FusedGraph` (a small Vec-indexed DAG owned by this crate — the
//! fused-op vocabulary stays out of `nn_graph`).

#[derive(thiserror::Error, Debug)]
pub enum ExtractError {
    #[error("egglog error: {0}")]
    Egglog(String),
    #[error("could not parse extracted term: {0}")]
    Parse(String),
}

/// One node of the post-rewrite operator DAG.
#[derive(Clone, Debug)]
pub struct FNode {
    pub op: String,
    pub args: Vec<Arg>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Node(usize),
    Str(String),
    Int(i64),
    Float(String),
}

/// Hash-consed DAG extracted from egglog (shared subterms deduplicated).
#[derive(Default, Debug)]
pub struct FusedGraph {
    pub nodes: Vec<FNode>,
    pub root: usize,
}

impl FusedGraph {
    /// Operation nodes only (excludes `Input`/`Weight` leaves).
    pub fn op_count(&self) -> usize {
        self.nodes.iter().filter(|n| !is_leaf(&n.op)).count()
    }

    /// Number of fused nodes produced by the rewrite rules.
    pub fn fused_count(&self) -> usize {
        self.nodes.iter().filter(|n| is_fused(&n.op)).count()
    }

    pub fn contains(&self, op: &str) -> bool {
        self.nodes.iter().any(|n| n.op == op)
    }
}

fn is_leaf(op: &str) -> bool {
    op == "Input" || op == "Weight"
}

pub(crate) fn is_fused(op: &str) -> bool {
    matches!(
        op,
        "FusedNormLinear"
            | "FusedNormLinearBias"
            | "FusedLayerNormLinear"
            | "FusedLayerNormLinearBias"
            | "SwiGLU"
            | "FusedGroupNormAct"
            | "FusedAdaLN"
            | "FusedGatedResidual"
            | "FusedNormRope"
            | "FusedNormRopeScale"
            | "FusedResidualNorm"
            | "FusedResidualLayerNorm"
            | "FusedGroupNormActConv3d"
            | "FusedGroupNormActConv3dBias"
            | "FusedLinearAct"
            | "FusedLinearBiasAct"
            | "FusedEmbeddingScale"
    )
}

/// Build the full egglog program, run it, and extract the fused graph.
pub fn run(schema: &str, rules: &str, lets: &str, root: &str) -> Result<FusedGraph, ExtractError> {
    // Two deliberate choices, both load-bearing for memory:
    //
    // * `(run-schedule (saturate (run)))` instead of the old `(run 100)`: the
    //   fusion ruleset is directed one-way rewrites (no birewrite, nothing
    //   generative), so the e-graph reaches fixpoint after the first
    //   iteration and saturate stops there.
    // * NO `(extract …)` command. Its result comes back as a printed
    //   s-expression, and printing un-shares the term DAG: every layer of a
    //   residual-stream model references its hidden state twice, so the
    //   printed tree is ~2^layers nodes even though the DAG is tiny. On the
    //   48-layer Gemma-4-12B unroll that string OOM-killed the compile
    //   (>150 GiB RSS) while the e-graph itself held only ~4k nodes.
    //   Extraction goes through the TermDag API below, which is hash-consed
    //   end to end.
    let program = format!("{schema}\n{rules}\n{lets}\n(run-schedule (saturate (run)))\n");

    let mut egraph = egglog::EGraph::default();
    egraph
        .parse_and_run_program(None, &program)
        .map_err(|e| ExtractError::Egglog(e.to_string()))?;

    let (sort, value) = egraph
        .eval_expr(&egglog::prelude::exprs::var(root))
        .map_err(|e| ExtractError::Egglog(e.to_string()))?;
    let (termdag, tid, _cost) = egraph
        .extract_value(&sort, value)
        .map_err(|e| ExtractError::Egglog(e.to_string()))?;
    term_to_graph(&termdag, tid)
}

// --- TermDag → hash-consed FusedGraph ---------------------------------------

fn term_to_graph(td: &egglog::TermDag, root: egglog::TermId) -> Result<FusedGraph, ExtractError> {
    use egglog::{ast::Literal, Term};

    let mut g = FusedGraph::default();
    let mut intern: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut memo: std::collections::HashMap<egglog::TermId, Arg> = std::collections::HashMap::new();
    // Iterative post-order: the unrolled model is a >1000-deep chain, too deep
    // to recurse over safely.
    let mut stack: Vec<(egglog::TermId, bool)> = vec![(root, false)];
    while let Some((id, ready)) = stack.pop() {
        if memo.contains_key(&id) {
            continue;
        }
        match td.get(id) {
            Term::Lit(l) => {
                let arg = match l {
                    Literal::Int(n) => Arg::Int(*n),
                    Literal::Float(f) => Arg::Float(f.to_string()),
                    Literal::String(s) => Arg::Str(s.clone()),
                    other => {
                        return Err(ExtractError::Parse(format!(
                            "unexpected literal in extracted term: {other:?}"
                        )))
                    }
                };
                memo.insert(id, arg);
            }
            Term::Var(v) => {
                return Err(ExtractError::Parse(format!(
                    "unexpected variable in extracted term: {v}"
                )))
            }
            Term::App(op, children) => {
                if !ready {
                    stack.push((id, true));
                    for &c in children.iter().rev() {
                        if !memo.contains_key(&c) {
                            stack.push((c, false));
                        }
                    }
                    continue;
                }
                let args: Vec<Arg> = children.iter().map(|c| memo[c].clone()).collect();
                let key = node_key(op, &args);
                let i = *intern.entry(key).or_insert_with(|| {
                    g.nodes.push(FNode { op: op.clone(), args });
                    g.nodes.len() - 1
                });
                memo.insert(id, Arg::Node(i));
            }
        }
    }
    match memo[&root] {
        Arg::Node(i) => {
            g.root = i;
            Ok(g)
        }
        ref other => Err(ExtractError::Parse(format!("root is not a node: {other:?}"))),
    }
}

fn node_key(op: &str, args: &[Arg]) -> String {
    let mut k = String::from(op);
    for a in args {
        k.push('|');
        match a {
            Arg::Node(i) => k.push_str(&format!("#{i}")),
            Arg::Str(s) => k.push_str(&format!("s:{s}")),
            Arg::Int(n) => k.push_str(&format!("i:{n}")),
            Arg::Float(f) => k.push_str(&format!("f:{f}")),
        }
    }
    k
}
