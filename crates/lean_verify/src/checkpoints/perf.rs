//! Checkpoint P — performance certificate for a default flip.
//!
//! The payload names the ledger entries it rests on (`ledger`), the touched rungs with their
//! treatment entry ids, the untouched rungs with their base and variant digests (from checkpoint
//! S), the tier-4 serving treatments, whether the scope changes numerics, and the correctness facts.
//! `plow_verify` accepts only when every rung decides `accept` (`Plow.Knobs.Ledger.checkP_sound`);
//! a floor it cannot compute is `insufficient_evidence` (`insufficient_blocks`). The reason starts
//! with `flip rejected` or `insufficient_evidence`.

use crate::{call, Certificate, VerifyError};

pub fn check_perf(payload: &serde_json::Value) -> Result<Certificate, VerifyError> {
    call("P", payload.clone())
}
