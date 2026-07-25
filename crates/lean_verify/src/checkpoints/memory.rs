//! Checkpoint F — Memory / allocation safety.
//!
//! Shares the JSON payload shape and Lean verifier with checkpoint D, but
//! the caller intent is different (F is the post-emit "the map itself is
//! safe" check; D is the pre-emit "the schedule enforces every edge" check).
//! Callers hand the exact same bundle: [`crate::checkpoints::schedule::ScheduleRequest`].

use crate::{call, Certificate, VerifyError};

pub use super::schedule::ScheduleRequest;

/// Verify the emitted address map. Returns the certificate on success.
pub fn check_address_map(req: &ScheduleRequest) -> Result<Certificate, VerifyError> {
    let payload = serde_json::to_value(req).map_err(VerifyError::SerializeRequest)?;
    call("F", payload)
}
