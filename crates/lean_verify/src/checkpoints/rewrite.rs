//! Checkpoint A — Rewrite rule soundness.
//!
//! The Lean side proves each rewrite rule's LHS and RHS denote the same tree
//! (`Plow.Rewrite.rule_*`). The dispatcher accepts a list of rule names
//! (from the egglog engine's per-bucket "rules fired" report) and confirms
//! every rule is in the sound-rules table.
//!
//! Adding a new rewrite rule requires updating both the Rust egglog side and
//! `Plow.Rewrite.soundRules` with a proof — the CLI enforces that link.

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
