//! Checkpoint A — legacy rewrite-name catalog check.
//!
//! Definitional syntax theorems exist for catalog entries, but this wrapper
//! does not bind actual rule bodies or floating-point kernels. It accepts rule names
//! (from the egglog engine's per-bucket "rules fired" report) and confirms
//! every rule is in the sound-rules table.
//!
//! plowc's actual-body producer additionally sends engine-parsed terms for
//! `Plow.RewriteBody.check_sound` through the same endpoint.

use serde::{Deserialize, Serialize};

use crate::{call, Certificate, VerifyError};

/// Full payload for checkpoint A.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewriteRulesRequest {
    /// Every rule name the egglog engine reports as having fired.
    pub rules: Vec<String>,
}

/// Verify all rules are in the sound-rules table.
pub fn check_rewrite_rules(req: &RewriteRulesRequest) -> Result<Certificate, VerifyError> {
    let payload = serde_json::to_value(req).map_err(VerifyError::SerializeRequest)?;
    call("A", payload)
}
