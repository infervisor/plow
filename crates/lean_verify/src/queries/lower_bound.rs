//! Query: schedule lower-bound target.
//!
//! Evaluates `max(E1, E2, E3)` on a validated DAG and caller-supplied costs.
//! The result is conditional on those costs and the channel model; it does not
//! certify machine-code execution time or physical HBM traffic.

use serde::{Deserialize, Serialize};

use crate::{query, QueryResult, VerifyError};

/// Request: compute the lower bound for a task graph.
#[derive(Clone, Debug, Serialize)]
pub struct LowerBoundRequest {
    /// Edges in the task graph (pairs of task indices).
    pub edges: Vec<(usize, usize)>,
    /// Per-task duration in cycles.
    pub durations: Vec<u64>,
    /// Total HBM bytes moved across all DMA tasks.
    pub total_hbm_bytes: u64,
    /// Peak HBM bandwidth in bytes per cycle.
    pub peak_bw_bytes_per_cycle: u64,
    /// Total FLOPs across all compute tasks.
    pub total_flops: u64,
    /// Peak throughput in FLOPs per cycle (all SMs combined).
    pub peak_flops_per_cycle: u64,
}

/// Which constraint binds the lower bound.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub enum BindingConstraint {
    #[serde(rename = "critical_path")]
    CriticalPath,
    #[serde(rename = "hbm_bandwidth")]
    HbmBandwidth,
    #[serde(rename = "compute_throughput")]
    ComputeThroughput,
}

/// Response: the conditional model lower bound.
#[derive(Clone, Debug, Deserialize)]
pub struct LowerBoundResult {
    /// The model lower bound: `max(critical_path, bw_bound, compute_bound)`.
    pub lower_bound: u64,
    /// Which of the three constraints is the binding one.
    pub binding_constraint: BindingConstraint,
    /// Individual bound values for diagnostics.
    pub critical_path: u64,
    pub bw_bound: u64,
    pub compute_bound: u64,
    /// Human-readable certificate.
    #[serde(default)]
    pub certificate: Option<String>,
}

/// Ask the Lean oracle for the certified schedule lower bound.
///
/// Returns `Err` if the binary isn't available or the query fails. The caller
/// can fall back to computing the bound in Rust (same arithmetic, no proof).
pub fn query_lower_bound(req: &LowerBoundRequest) -> Result<LowerBoundResult, VerifyError> {
    let payload = serde_json::to_value(req).map_err(VerifyError::SerializeRequest)?;
    let result: QueryResult = query("lower_bound", payload)?;
    let mut lb: LowerBoundResult = serde_json::from_value(result.answer)
        .map_err(|e| VerifyError::DeserializeQueryResult(e, "lower_bound answer".into()))?;
    lb.certificate = result.certificate;
    Ok(lb)
}
