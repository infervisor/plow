//! Query: counter granularity oracle.
//!
//! Asks Lean whether fine-grained (per-slice) counters can improve makespan
//! on each cross-op edge. Backed by `Plow.CounterGranularity.fineCanPay`.
//!
//! The Lean side checks whether the consumer slices' work is non-uniform —
//! the only condition under which fine counters can beat coarse (proved by
//! the `collapse` theorem).

use serde::{Deserialize, Serialize};

use crate::{query, QueryResult, VerifyError};

/// One edge to evaluate: a producer→consumer boundary in the tile graph.
#[derive(Clone, Debug, Serialize)]
pub struct EdgeQuery {
    /// Stable identifier for this edge (returned in the response).
    pub id: u64,
    /// Consumer slice indices (workgroup IDs on the consumer side).
    pub consumer_slices: Vec<u64>,
    /// Per-slice work cost (cycles) — same length as `consumer_slices`.
    pub work: Vec<u64>,
}

/// Request: evaluate counter granularity for a batch of edges.
#[derive(Clone, Debug, Serialize)]
pub struct CounterGranularityRequest {
    pub edges: Vec<EdgeQuery>,
}

/// Per-edge decision from the oracle.
#[derive(Clone, Debug, Deserialize)]
pub struct EdgeDecision {
    /// Matches the `id` from the request.
    pub id: u64,
    /// `true` = emit fine (per-slice) counters; `false` = coarse suffices.
    pub use_fine: bool,
    /// Human-readable reason (e.g. "uniform work: collapse theorem applies").
    #[serde(default)]
    pub reason: Option<String>,
}

/// Response: per-edge fine/coarse decisions.
#[derive(Clone, Debug, Deserialize)]
pub struct CounterGranularityResult {
    pub decisions: Vec<EdgeDecision>,
}

/// Ask the Lean oracle for per-edge counter granularity decisions.
///
/// Returns `Err` if the binary isn't available or the query fails. Callers
/// should fall back to the Rust heuristic on error.
pub fn query_counter_granularity(
    req: &CounterGranularityRequest,
) -> Result<CounterGranularityResult, VerifyError> {
    let payload = serde_json::to_value(req).map_err(VerifyError::SerializeRequest)?;
    let result: QueryResult = query("counter_granularity", payload)?;
    let decisions: CounterGranularityResult = serde_json::from_value(result.answer)
        .map_err(|e| VerifyError::DeserializeQueryResult(e, "counter_granularity answer".into()))?;
    Ok(decisions)
}
