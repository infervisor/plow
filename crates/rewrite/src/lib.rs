//! `rewrite` — the rewriting half of the Infervisor JIT.
//!
//! Takes a shape-inferred `nn_graph::Graph` (the Stage-1 frontend IR), lowers it
//! into egglog, runs Stage-2 operator-fusion rules to equality saturation, and
//! extracts the lowest-cost equivalent form as a [`FusedGraph`].
//!
//! egglog (e-graph + datalog) is used deliberately: the rewrite rules are
//! declarative, and the same engine can later host the scheduler's placement /
//! interval-conflict constraints (design §8.5). `nn_graph` stays egglog-agnostic
//! — the dependency lives only here.

pub mod bridge;
pub mod collapse;
pub mod explore;
mod extract;
pub mod footprint;
mod lower;
pub mod oracle;
pub mod rule_body;
pub mod tile;
pub mod tilegraph;

pub use bridge::{plan_from_all_blocks, plan_from_block, plan_from_fused, BridgeError};
pub use collapse::{collapse, try_collapse, CollapseError};
pub use explore::{
    best_chunk_count, best_chunk_count_egglog, chunk_prefill_cycles, explore_tiles, Choice,
    ChunkCostIn, GemmJob,
};
pub use extract::{Arg, FNode, FusedGraph, FusedSites};
pub use footprint::{footprints, Footprint, OpIo, TensorSlice, TileDomain};
pub use tile::{lower_gemm, TileNode, TileSeq};
/// Re-export of [`tilegraph::TileNode`] under a distinct name so downstream
/// users of [`TileGraph`] can name its element type without ambiguity with
/// [`tile::TileNode`].
pub use tilegraph::TileNode as GraphNode;
pub use tilegraph::{
    assemble, consumer_thresholds, materialize_tile_deps, AxisCouple, Compute, ConcatGroup,
    ConstraintSet, Handoff, HandoffKind, LayerPlan, LayoutSpec, LocalityReq, MatEdge, ModelOp,
    ModelOpKind, OpDesc, OpKind, OpSpec, RelaxableHandoff, TileDep, TileDependency, TileGraph,
    LAYOUT_RANK,
};

use nn_graph::Graph;

const SCHEMA: &str = include_str!("egl/schema.egg");
const RULES: &str = include_str!("egl/rules.egg");

/// The embedded egglog rewrite-rule source (`egl/rules.egg`) — the exact text
/// the engine runs. Exposed so plowc can parse the `; rule: <name>`
/// annotations and submit the live rule catalog to Lean checkpoint A instead
/// of a hardcoded copy.
pub fn rules_source() -> &'static str {
    RULES
}

#[derive(thiserror::Error, Debug)]
pub enum RewriteError {
    #[error(transparent)]
    Lower(#[from] lower::LowerError),
    #[error(transparent)]
    Extract(#[from] extract::ExtractError),
}

/// Op-count summary across the rewrite.
#[derive(Clone, Copy, Debug)]
pub struct RewriteStats {
    /// Operation nodes in the input graph.
    pub ops_before: usize,
    /// Operation nodes in the extracted fused graph (shared subterms deduped).
    pub ops_after: usize,
    /// Number of fused nodes the rules produced.
    pub fused: usize,
}

/// Rewrite a graph: lower → saturate with fusion rules → extract.
pub fn rewrite_graph(g: &Graph) -> Result<(FusedGraph, RewriteStats), RewriteError> {
    let (lets, root) = lower::lower(g)?;
    let fused = extract::run(SCHEMA, RULES, &lets, &[root])?;
    let stats = RewriteStats {
        ops_before: g.nodes.len(),
        ops_after: fused.op_count(),
        fused: fused.fused_count(),
    };
    Ok((fused, stats))
}

/// [`rewrite_graph`] extracting EVERY graph output from the same e-graph and extractor, so shared
/// paths extract identically. Extracting only the last output drops every fused site that feeds
/// only an earlier one — on GLM, whose last output is the MTP head, the base `model.norm`.
pub fn rewrite_graph_outputs(g: &Graph) -> Result<(FusedGraph, RewriteStats), RewriteError> {
    let (lets, roots) = lower::lower_outputs(g)?;
    let fused = extract::run(SCHEMA, RULES, &lets, &roots)?;
    let stats = RewriteStats {
        ops_before: g.nodes.len(),
        ops_after: fused.op_count(),
        fused: fused.fused_count(),
    };
    Ok((fused, stats))
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum SitesError {
    /// No graph for this checkpoint: no builder for its architecture, or a config it cannot parse.
    #[error("graph build failed: {0}")]
    Unavailable(String),
    /// The graph built, but lowering or extraction failed.
    #[error("rewrite failed: {0}")]
    Rewrite(String),
}

/// The fused sites devgen lowers under `PLOW_EMIT_REWRITE`: [`FusedGraph::sites`] of the
/// checkpoint's text-generation graph, built and bound as plowc's whole-graph coverage builds it.
pub fn fused_sites_for_config(config_json: &str) -> Result<FusedSites, SitesError> {
    let mut graph = nn_graph::models::build_text_generation_from_config_json_at(
        config_json,
        &nn_graph::models::ShapeBucket::default(),
    )
    .map_err(|e| SitesError::Unavailable(e.to_string()))?;
    graph.bind(&nn_graph::Bindings::new().set("B", 1).set("S", 8192));
    let (fused, _) =
        rewrite_graph_outputs(&graph).map_err(|e| SitesError::Rewrite(e.to_string()))?;
    Ok(fused.sites())
}

/// Saturate the fusion rules over `g` and report per-fused-op e-graph match
/// counts WITHOUT extracting a term. For analysis-only callers (the devblob
/// path's fusion report): extraction can trip an upstream egglog-2.0.0 panic
/// (`extract.rs:471`, unwrap on a costless e-class — e.g. the Qwen3 and
/// Gemma-MoE graphs), and the release profile's `panic = "abort"` turns that
/// into process death. Saturation + `(print-size)` never enters that path.
///
/// Returns `(graph_ops, [(fused_op_name, e-graph count)])` — counts are
/// discovered fusion *opportunities* in the saturated e-graph, not the
/// extracted operator count.
pub fn explore_stats(g: &Graph) -> Result<(usize, Vec<(String, usize)>), RewriteError> {
    let (lets, _root) = lower::lower(g)?;
    let program =
        format!("{SCHEMA}\n{RULES}\n{lets}\n(run-schedule (saturate (run)))\n(print-size)\n");
    let mut egraph = egglog::EGraph::default();
    let msgs = egraph
        .parse_and_run_program(None, &program)
        .map_err(|e| RewriteError::Extract(extract::ExtractError::Egglog(e.to_string())))?;
    let mut fused = Vec::new();
    for m in &msgs {
        for line in m.to_string().lines() {
            if let Some((name, count)) = line.rsplit_once(':') {
                let name = name.trim();
                if let Ok(count) = count.trim().parse::<usize>() {
                    if extract::is_fused(name) && count > 0 {
                        fused.push((name.to_string(), count));
                    }
                }
            }
        }
    }
    Ok((g.nodes.len(), fused))
}
