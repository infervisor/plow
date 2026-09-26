use serde::{Deserialize, Serialize};

use crate::{call, Certificate, VerifyError};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredCandidate {
    /// Exact specialization/identity domain supplied by the AOT planner.
    pub domain: String,
    pub key: String,
    /// Positive cost in one common unit. Lean compares the JSON decimal exactly.
    pub cost: serde_json::Number,
    pub qualified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyChoice {
    pub domain: String,
    pub key: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasuredPolicyRequest {
    pub required: Vec<String>,
    pub candidates: Vec<MeasuredCandidate>,
    pub choices: Vec<PolicyChoice>,
}

/// Certifies coverage and minimization of supplied costs, not measurement authenticity.
pub fn check_measured_policy(request: &MeasuredPolicyRequest) -> Result<Certificate, VerifyError> {
    call(
        "R",
        serde_json::to_value(request).map_err(VerifyError::SerializeRequest)?,
    )
}
