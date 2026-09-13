//! Checkpoint K — knob consistency.
//!
//! The payload is `plow_asset::knob::registry_json` plus `target`, `sources` and, from `plowc`,
//! the emitter's `recorded` values. `plow_verify` resolves the sources (cli > env > production
//! default > static), checks every constraint (`Plow.Knobs.checkK_sound`), and checks the registry
//! itself against its declared targets. A `cases` array switches to batch mode for differential
//! testing: one verdict per case, `ok`, `wf` or the first violated constraint's id.

use crate::{call, Certificate, VerifyError};

pub fn check_knobs(payload: &serde_json::Value) -> Result<Certificate, VerifyError> {
    call("K", payload.clone())
}

/// Batch mode: the verdict of every case, in order.
pub fn verdicts(payload: &serde_json::Value) -> Result<Vec<String>, VerifyError> {
    let cert = call("K", payload.clone())?;
    let notes = cert.notes.unwrap_or_default();
    match notes.strip_prefix("verdicts=") {
        Some("") if cert.ok => Ok(Vec::new()),
        Some(v) if cert.ok => Ok(v.split(';').map(String::from).collect()),
        _ => Err(VerifyError::Rejected(cert.reason.unwrap_or(notes))),
    }
}
