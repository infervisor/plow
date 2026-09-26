//! Checkpoint D — Schedule (counter protocol + reclamation).
//!
//! Serializes a `(TaskGraph, CounterProtocol, schedule_order, AddressMap)`
//! bundle to the JSON schema `Plow.CLI.Payload` expects.

use serde::{Deserialize, Serialize};

use crate::{call, Certificate, VerifyError};

/// One entry in the address map — the reader/writer task sets are what the
/// reclamation check consumes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddrEntry {
    pub name: String,
    pub offset: u64,
    pub size: u64,
    pub cls: String, // "Persistent" | "RequestIo" | "Scratch" | "Growable"
    pub writers: Vec<usize>,
    pub readers: Vec<usize>,
}

/// Full payload for checkpoint D.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleRequest {
    pub task_graph: TaskGraphView,
    pub protocol: ProtocolView,
    pub schedule_order: Vec<u64>,
    pub address_map: Vec<AddrEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskGraphView {
    pub n: usize,
    pub edges: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolView {
    pub waits: Vec<Vec<u64>>,
    pub succs: Vec<Vec<u64>>,
    /// counter id (as string) → threshold. Lean's Json uses string keys.
    pub threshold: std::collections::BTreeMap<String, u64>,
    pub resource: Vec<u64>,
    pub stream_idx: Vec<u64>,
}

/// Verify the schedule + address map. Returns the certificate on success or
/// a `VerifyError::Rejected` on failure.
pub fn check_schedule(req: &ScheduleRequest) -> Result<Certificate, VerifyError> {
    let payload = crate::paths::payload(req)?;
    call("D", payload)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionPath {
    pub source: usize,
    pub target: usize,
    pub via: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AllocationLease {
    pub pool: u64,
    pub allocation: u64,
    pub generation: u64,
    pub owner: u64,
    pub offset: u64,
    pub size: u64,
    pub acquire: usize,
    pub retire: usize,
    pub cancel: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryAccess {
    pub task: usize,
    pub pool: u64,
    pub allocation: u64,
    pub generation: u64,
    pub owner: u64,
    pub offset: u64,
    pub size: u64,
    pub write: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryEffects {
    pub schema: u32,
    pub leases: Vec<AllocationLease>,
    pub accesses: Vec<MemoryAccess>,
    pub fence_counters: Vec<u64>,
}

/// Conditional on complete access declarations and the implementation of the
/// declared completion fences. Ordinary resource issue order is not sufficient.
pub fn check_memory_effects(
    req: &ScheduleRequest,
    paths: &[CompletionPath],
    effects: &MemoryEffects,
) -> Result<Certificate, VerifyError> {
    let mut payload = serde_json::to_value(req).map_err(VerifyError::SerializeRequest)?;
    payload["address_paths"] = serde_json::to_value(paths).map_err(VerifyError::SerializeRequest)?;
    payload["memory_effects"] = serde_json::to_value(effects).map_err(VerifyError::SerializeRequest)?;
    call("D", payload)
}
