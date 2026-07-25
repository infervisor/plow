//! Checkpoint C — SRAM temporal fit.
//!
//! The Rust `sram_fit::analyze_temporal_fit` pass identifies hand-offs that
//! could be promoted from HBM back to SramSameSm without over-subscribing the
//! per-SM page pool. This bridge asks the Lean verifier to re-check every
//! candidate against `Plow.Sram.temporalFitSafe`, backed by the universal
//! theorem `occupancy_le_of_temporal_fit`.

use serde::{Deserialize, Serialize};

use crate::{call, Certificate, VerifyError};

/// One hand-off descriptor: producer and consumer page footprints plus the
/// scheduled cycle windows in which each holds its pages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Handoff {
    pub producer_pages: u64,
    pub consumer_pages: u64,
    /// Last cycle the producer holds its pages (`[0, producer_release)`).
    pub producer_release: u64,
    /// First cycle the consumer holds its pages.
    pub consumer_acquire: u64,
    /// Last cycle the consumer holds its pages (`[acquire, release]`).
    pub consumer_release: u64,
}

/// Full payload for checkpoint C.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SramFitRequest {
    /// Page budget shared by producer + consumer on the target SM.
    pub budget: u64,
    /// Every promotion candidate the caller wants proven safe.
    pub handoffs: Vec<Handoff>,
}

/// Verify the SRAM promotion set. Returns the certificate on success or a
/// `VerifyError::Rejected` on failure.
pub fn check_sram_fit(req: &SramFitRequest) -> Result<Certificate, VerifyError> {
    let payload = serde_json::to_value(req).map_err(VerifyError::SerializeRequest)?;
    call("C", payload)
}
