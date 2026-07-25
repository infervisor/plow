//! Checkpoint B — Tile partition + cost bounds.
//!
//! Backed by `Plow.TilePartition.tile_partition_covers`. The caller submits
//! per-candidate `(gemm, tile, cost_bound)` triples from the extractor's
//! candidate table; the Lean side verifies partition validity + cost bound.

use serde::{Deserialize, Serialize};

use crate::{call, Certificate, VerifyError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmShapeJ {
    pub m: u64,
    pub n: u64,
    pub k: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileShapeJ {
    pub bm: u64,
    pub bn: u64,
    pub bk: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileCandidate {
    pub gemm: GemmShapeJ,
    pub tile: TileShapeJ,
    /// Upper bound on tile-work the extractor is willing to accept for this
    /// candidate. `tileCount · bm · bn · bk` must not exceed this.
    pub cost_bound: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilePartitionRequest {
    pub candidates: Vec<TileCandidate>,
}

/// Verify every tile candidate against partition validity + cost bound.
pub fn check_tile_partition(req: &TilePartitionRequest) -> Result<Certificate, VerifyError> {
    let payload = serde_json::to_value(req).map_err(VerifyError::SerializeRequest)?;
    call("B", payload)
}
