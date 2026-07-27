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

/// egglog's tree-additive cost, in **arbitrary precision** — the ONLY change
/// from the default is that the accumulator cannot overflow.
///
/// # The bug this fixes
///
/// `TreeAdditiveCostModel` sums its children, and [`egglog::extract::Cost`] for
/// every integer type combines with `saturating_add`
/// (`egglog-2.0.0/src/extract.rs:70`). Tree cost is not DAG cost: a residual
/// stream references its hidden state ~8× per layer (q/k/v read the normed
/// hidden; the sandwich norm and the residual add each read the stream), so the
/// *tree* unfolding of layer `L` costs ~8^L even though the DAG is linear in
/// `L`. That crosses `u64::MAX` around **layer 21**.
///
/// Past that point every e-class on the residual chain is pinned at
/// `u64::MAX`, so `new_cost < *e.get()` is never true again and Bellman-Ford's
/// `topo_rnk` stops advancing in step with the costs. `save_best_parent_edge`
/// then requires `target_topo_rnk > compute_topo_rnk_hyperedge(row)` to record
/// a parent edge; that test fails for *every* e-node feeding some e-class, and
/// reconstruction hits `parent_edge…get(&value).unwrap()` on `None`
/// (`extract.rs:471`). With `panic = "abort"` in the release profile that is
/// process death — which is why `explore_stats` exists as a saturate-only
/// analysis path and why the devblob emitter never calls `rewrite_graph`.
///
/// Bisected against the real Gemma-4-12B text config: extraction succeeds at
/// 1, 2, 3, 4, 6, 8, 12 and 16 layers and aborts at 24 and 48. It is a *scale*
/// bug, not a graph-shape bug — and 48 layers is the model plow serves.
///
/// # Why `BigInt` and not a different cost function
///
/// Deliberately surgical. A cheaper-to-compute bounded cost (e.g. critical-path
/// depth, `max` instead of `+`) also cannot overflow, but it *re-ranks the
/// fusion space*: measured, it flips the `pre_feedforward_norm` site from
/// `FusedResidualNorm` to two `FusedNormLinear`s, because depth cost cannot see
/// that the latter recomputes the norm once per consumer. That is a design
/// change to the rewrite's objective, not a bug fix, and it broke six
/// `fuse_all_models` expectations. `BigInt` reproduces the default's extraction
/// decisions *exactly* wherever the default did not overflow, so every existing
/// test keeps passing and the only behavioural delta is that 24+ layer models
/// now extract instead of aborting.
///
/// Exactness note: `schema.egg` declares no `:cost` on any constructor
/// (verified — `grep -c :cost` is 0 across all three `egl/*.egg` files), so
/// `TreeAdditiveCostModel::enode_cost`'s `func.decl.cost.unwrap_or(1)` is
/// uniformly `1`. `Function::decl` is private in egglog 2.0.0, so a head weight
/// of `1` here is not an approximation — it is the same number.
///
/// Cost: ~143 bits (3 limbs) at 48 layers over a ~4k-node e-graph. Immaterial.
struct BigTreeAdditiveCost;

impl egglog::extract::CostModel<num::BigInt> for BigTreeAdditiveCost {
    fn fold(&self, _head: &str, children: &[num::BigInt], head_cost: num::BigInt) -> num::BigInt {
        children.iter().fold(head_cost, |s, c| s + c)
    }

    fn enode_cost(
        &self,
        _egraph: &egglog::EGraph,
        _func: &egglog::Function,
        _row: &egglog::FunctionRow,
    ) -> num::BigInt {
        num::BigInt::from(1)
    }
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
    // Extract under `BigTreeAdditiveCost` rather than egglog's default. Same
    // cost function, arbitrary precision: see [`BigTreeAdditiveCost`] — the
    // default saturates `u64` on any residual model past ~21 layers and then
    // aborts the process during reconstruction.
    let extractor = egglog::extract::Extractor::<num::BigInt>::compute_costs_from_rootsorts(
        Some(vec![sort.clone()]),
        &egraph,
        BigTreeAdditiveCost,
    );
    let mut termdag = egglog::TermDag::default();
    let (_cost, tid) = extractor
        .extract_best_with_sort(&egraph, &mut termdag, value, sort)
        .ok_or_else(|| ExtractError::Egglog("no extractable term for the graph root".into()))?;
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
