//! Checkpoint S — knob scope.
//!
//! The payload is `plowrt::knob_scope::payload`: base/variant program pairs aligned by key, the
//! route steps, the knob delta, its declared scope and the op class table. `plow_verify` accepts
//! when every difference is inside the scope (`Plow.Knobs.Scope.checkS_sound`); with no delta it
//! accepts only identical packets (`off_identity`). The notes carry the difference counts and the
//! `empty_effect` / `scope_slack` warnings.

use crate::{call, Certificate, VerifyError};

pub fn check_scope(payload: &serde_json::Value) -> Result<Certificate, VerifyError> {
    call("S", payload.clone())
}
