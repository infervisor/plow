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
    let payload = serde_json::to_value(req).map_err(VerifyError::SerializeRequest)?;
    call("D", payload)
}
