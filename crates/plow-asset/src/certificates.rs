use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

pub fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RungDomain {
    pub phase: String,
    pub rows: u32,
    pub n: u32,
    pub k: u32,
    pub heads: u32,
    pub n_tail: u32,
    pub k_tail: u32,
    pub context_capacity: u32,
    pub min_live: u32,
    pub max_live: u32,
    pub inactive_rows: bool,
}

impl RungDomain {
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.phase.as_str(), "decode" | "prefill")
            || self.rows == 0
            || self.n == 0
            || self.k == 0
            || self.heads == 0
            || self.n_tail > self.n
            || self.k_tail > self.k
            || self.min_live > self.max_live
            || self.max_live > self.context_capacity
            || self.context_capacity == 0
            || (!self.inactive_rows && self.min_live == 0)
        {
            return Err("invalid rung specialization domain".into());
        }
        Ok(())
    }

    /// The token loop only checks these numeric guards; artifact checks run at load.
    pub fn accepts(&self, rows: u32, min_live: u32, max_live: u32, inactive: bool) -> bool {
        rows == self.rows
            && min_live <= max_live
            && min_live >= self.min_live
            && max_live <= self.max_live
            && (!inactive || self.inactive_rows)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjectBinding {
    pub file: String,
    pub sha256: String,
    pub entry: String,
    pub abi_sha256: String,
    pub kernarg_bytes: u32,
    pub grid: [u32; 3],
    pub threads: u32,
    pub wave_size: u32,
    pub vgpr: u32,
    pub agpr: u32,
    pub sgpr: u32,
    pub private_bytes: u32,
    pub static_lds_bytes: u32,
    pub dynamic_lds_bytes: u32,
    pub occupancy: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionIdentity {
    pub model: String,
    pub revision: String,
    // Includes all storage/compute/accumulator dtypes, scales, rounding and table generation.
    pub precision_contract_sha256: String,
    pub checkpoint_sha256: String,
    pub packet_sha256: String,
    pub segment_dag_sha256: String,
    pub objects: Vec<ObjectBinding>,
    pub rungs: Vec<RungDomain>,
    pub hardware: String,
    pub topology_sha256: String,
    pub runtime_sha256: String,
    pub runtime_config_sha256: String,
    pub compiler_sha256: String,
    pub oracle_sha256: String,
}

impl ExecutionIdentity {
    pub fn validate(&self) -> Result<(), String> {
        if self.model.is_empty()
            || self.revision.is_empty()
            || self.hardware.is_empty()
            || self.objects.is_empty()
            || self.rungs.is_empty()
        {
            return Err("incomplete execution identity".into());
        }
        for digest in [
            &self.precision_contract_sha256,
            &self.checkpoint_sha256,
            &self.packet_sha256,
            &self.segment_dag_sha256,
            &self.topology_sha256,
            &self.runtime_sha256,
            &self.runtime_config_sha256,
            &self.compiler_sha256,
            &self.oracle_sha256,
        ] {
            if !is_sha256(digest) {
                return Err("invalid execution identity digest".into());
            }
        }
        let mut objects = BTreeSet::new();
        for object in &self.objects {
            let mut path = std::path::Path::new(&object.file).components();
            if !matches!(path.next(), Some(std::path::Component::Normal(_)))
                || path.next().is_some()
                || !objects.insert((&object.file, &object.entry))
                || object.entry.is_empty()
                || object.entry.as_bytes().contains(&0)
                || !is_sha256(&object.sha256)
                || !is_sha256(&object.abi_sha256)
                || object.grid.contains(&0)
                || object.threads == 0
                || object.threads > 1024
                || !matches!(object.wave_size, 32 | 64)
                || object.threads % object.wave_size != 0
                || object.occupancy == 0
                || object.kernarg_bytes == 0
            {
                return Err("invalid or duplicate object/launch binding".into());
            }
        }
        for (i, rung) in self.rungs.iter().enumerate() {
            rung.validate()?;
            if self.rungs[..i].contains(rung) {
                return Err("duplicate rung domain".into());
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        Ok(crate::decode_objects::image_sha256(&bytes))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemanticScope {
    RewriteNameCatalog,
    RewriteBodyExpansion,
    CoarseDependencyPreservation,
    AddressReuse,
    PrecisionTransformation,
    MeasuredPolicy,
    SelectedGemmPolicy,
    LayoutMapping,
    LogicalTensorEffects,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticCertificate {
    pub subject_sha256: String,
    pub scope: SemanticScope,
    pub request_sha256: String,
    pub verifier_sha256: String,
    pub theorem: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralCostCertificate {
    pub subject_sha256: String,
    pub request_sha256: String,
    pub verifier_sha256: String,
    pub logical_read_bytes: u64,
    pub logical_write_bytes: u64,
    pub flops: u64,
    pub launches: u32,
    pub critical_path_ns: u64,
    pub physical_hbm_bytes: Option<u64>,
    pub physical_traffic_assumptions: Vec<String>,
    pub duration_measurements_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignArm {
    pub id: String,
    pub samples: u32,
    pub median: f64,
    pub mad: f64,
    pub raw_samples_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmpiricalPerformanceCertificate {
    pub subject_sha256: String,
    pub control_subject_sha256: String,
    pub protocol_sha256: String,
    pub verifier_sha256: String,
    pub request_sha256: String,
    pub metric: String,
    pub lower_is_better: bool,
    // Protocol order: control, treatment, repeat control, repeat treatment.
    pub arms: [CampaignArm; 4],
    pub numerical_gate_sha256: String,
    pub artifact_gate_sha256: String,
    pub serving_gate_sha256: String,
}

impl EmpiricalPerformanceCertificate {
    pub fn improvement_beyond_floor(&self) -> Result<f64, String> {
        for value in [
            &self.subject_sha256,
            &self.control_subject_sha256,
            &self.protocol_sha256,
            &self.verifier_sha256,
            &self.request_sha256,
            &self.numerical_gate_sha256,
            &self.artifact_gate_sha256,
            &self.serving_gate_sha256,
        ] {
            if !is_sha256(value) {
                return Err("missing empirical evidence binding".into());
            }
        }
        let mut ids = BTreeSet::new();
        for arm in &self.arms {
            if arm.id.is_empty()
                || !ids.insert(&arm.id)
                || arm.samples < 30
                || !arm.median.is_finite()
                || arm.median <= 0.0
                || !arm.mad.is_finite()
                || arm.mad < 0.0
                || !is_sha256(&arm.raw_samples_sha256)
            {
                return Err("invalid four-arm campaign evidence".into());
            }
        }
        if self.metric.is_empty() {
            return Err("missing metric".into());
        }
        let [c, t, c2, t2] = &self.arms;
        let drift = (c.median - c2.median).abs();
        let spread = (t.median - t2.median).abs();
        if spread > 3.0 * drift {
            return Err("unstable treatment; rerun campaign".into());
        }
        let floor = drift.max(spread) + 2.0 * self.arms.iter().map(|a| a.mad).fold(0.0, f64::max);
        let improvement = ((c.median + c2.median) - (t.median + t2.median))
            * 0.5
            * if self.lower_is_better { 1.0 } else { -1.0 };
        if !improvement.is_finite() || improvement <= floor {
            return Err("improvement does not exceed four-arm floor".into());
        }
        Ok(improvement - floor)
    }
}

/// Binding validation does not execute the proofs or authenticate the evidence files.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundCertificates {
    pub identity: ExecutionIdentity,
    pub semantic: Vec<SemanticCertificate>,
    pub structural_cost: Option<StructuralCostCertificate>,
    pub empirical_performance: Option<EmpiricalPerformanceCertificate>,
}

pub const PACKET_CHECKS_FILE: &str = "lean-checks.json";

/// Replayable compiler checks, not full execution-identity or performance admission.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompileCheckReceipt {
    pub program: Option<usize>,
    pub scope: SemanticScope,
    pub checkpoint: String,
    pub request_sha256: String,
    pub verifier_sha256: String,
    pub request: serde_json::Value,
    pub response: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PacketCheckReceipts {
    pub schema: u32,
    pub packet_sha256: String,
    pub compiler_sha256: String,
    pub checks: Vec<CompileCheckReceipt>,
}

impl PacketCheckReceipts {
    pub fn validate_packet(&self, packet: &[u8]) -> Result<(), String> {
        if self.schema != 1
            || !is_sha256(&self.compiler_sha256)
            || self.packet_sha256 != crate::decode_objects::image_sha256(packet)
        {
            return Err("compiler check receipt does not match loaded packet".into());
        }
        let mut obligations = BTreeSet::new();
        for check in &self.checks {
            let supported = match check.scope {
                SemanticScope::RewriteBodyExpansion => check.checkpoint == "A"
                    && check.program.is_none()
                    && check.request.get("bodies").and_then(serde_json::Value::as_array)
                        .is_some_and(|bodies| !bodies.is_empty())
                    && check.request.get("source_sha256").and_then(serde_json::Value::as_str)
                        .is_some_and(is_sha256),
                SemanticScope::CoarseDependencyPreservation => {
                    check.checkpoint == "D"
                        && check.program.is_some()
                        && check
                            .request
                            .get("dependency_paths")
                            .and_then(serde_json::Value::as_array)
                            .is_some()
                }
                SemanticScope::MeasuredPolicy => {
                    check.checkpoint == "R"
                        && check.program.is_none()
                        && check
                            .request
                            .get("required")
                            .and_then(serde_json::Value::as_array)
                            .is_some_and(|r| !r.is_empty())
                }
                SemanticScope::SelectedGemmPolicy => check.checkpoint == "R"
                    && check.program.is_some() && check.request.get("wire_binding").is_some(),
                SemanticScope::LayoutMapping => check.checkpoint == "L" && check.program.is_some(),
                SemanticScope::LogicalTensorEffects => check.checkpoint == "D"
                    && check.program.is_some() && check.request.get("memory_effects").is_some(),
                _ => false,
            };
            if !supported
                || !obligations.insert((&check.checkpoint, &check.request_sha256, check.program))
                || !is_sha256(&check.verifier_sha256)
                || check.request_sha256
                    != crate::decode_objects::image_sha256(
                        &serde_json::to_vec(&check.request).map_err(|e| e.to_string())?,
                    )
                || check
                    .response
                    .get("ok")
                    .and_then(serde_json::Value::as_bool)
                    != Some(true)
                || check
                    .response
                    .get("checkpoint")
                    .and_then(serde_json::Value::as_str)
                    != Some(&check.checkpoint)
            {
                return Err("invalid, unsupported or duplicated compiler check receipt".into());
            }
        }
        Ok(())
    }
}

pub fn mla_layout_obligation(
    producer_instruction: usize,
    producer: &packet::dev::DevInst64,
    consumer: &packet::dev::DevInst64,
    capacity_bytes: u64,
) -> Result<serde_json::Value, String> {
    use packet::dev::DevOp;
    if producer.op != DevOp::FlashMerge as u16
        || producer.i[4] != 1024
        || consumer.op != DevOp::MlaBmmFp8 as u16
        || consumer.i[4..] != [0, 1024, 0, 0]
        || producer.t[0] != consumer.t[1]
        || producer.i[0] != consumer.i[0]
        || producer.i[1] != consumer.i[1]
        || producer.i[3] != consumer.i[3]
        || capacity_bytes % 2 != 0
    {
        return Err("MLA layout witness does not match producer/consumer ABI".into());
    }
    Ok(serde_json::json!({
        "producer_instruction": producer_instruction,
        "consumer_instruction": producer_instruction + 1,
        "rows": consumer.i[0], "heads": consumer.i[1], "n": consumer.i[2], "k": consumer.i[3],
        "head_stride": consumer.i[5], "capacity_elements": capacity_bytes / 2,
        "boundary": "bf16_rne", "weight_scale": "scalar_after_group_sum",
        "activation_group": 128, "inactive_rows": false,
    }))
}

impl BoundCertificates {
    pub fn validate_binding(
        &self,
        expected: &ExecutionIdentity,
        required: &[RungDomain],
    ) -> Result<(), String> {
        let subject = self.identity.digest()?;
        expected.validate()?;
        if &self.identity != expected {
            return Err("execution identity mismatch".into());
        }
        if required.is_empty() || required.iter().any(|r| !self.identity.rungs.contains(r)) {
            return Err("incomplete rung coverage".into());
        }
        for cert in &self.semantic {
            if cert.subject_sha256 != subject
                || !is_sha256(&cert.request_sha256)
                || !is_sha256(&cert.verifier_sha256)
                || cert.theorem.is_empty()
            {
                return Err("invalid semantic binding".into());
            }
        }
        if let Some(cert) = &self.structural_cost {
            if cert.subject_sha256 != subject
                || !is_sha256(&cert.request_sha256)
                || !is_sha256(&cert.verifier_sha256)
                || cert.physical_hbm_bytes.is_some() && cert.physical_traffic_assumptions.is_empty()
                || cert
                    .duration_measurements_sha256
                    .as_ref()
                    .is_some_and(|s| !is_sha256(s))
            {
                return Err("invalid structural-cost binding/assumptions".into());
            }
        }
        if let Some(cert) = &self.empirical_performance {
            if cert.subject_sha256 != subject {
                return Err("invalid empirical subject".into());
            }
            cert.improvement_beyond_floor()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> ExecutionIdentity {
        let sha = "a".repeat(64);
        ExecutionIdentity {
            model: "model".into(),
            revision: "revision".into(),
            hardware: "gfx950/256CU".into(),
            precision_contract_sha256: sha.clone(),
            checkpoint_sha256: sha.clone(),
            packet_sha256: sha.clone(),
            segment_dag_sha256: sha.clone(),
            topology_sha256: sha.clone(),
            runtime_sha256: sha.clone(),
            runtime_config_sha256: sha.clone(),
            compiler_sha256: sha.clone(),
            oracle_sha256: sha.clone(),
            objects: vec![ObjectBinding {
                file: "kernel.elf".into(),
                sha256: sha.clone(),
                entry: "kernel".into(),
                abi_sha256: sha,
                kernarg_bytes: 64,
                grid: [256, 1, 1],
                threads: 256,
                wave_size: 64,
                vgpr: 80,
                agpr: 0,
                sgpr: 64,
                private_bytes: 0,
                static_lds_bytes: 0,
                dynamic_lds_bytes: 0,
                occupancy: 4,
            }],
            rungs: vec![RungDomain {
                phase: "decode".into(),
                rows: 16,
                n: 256,
                k: 512,
                heads: 8,
                n_tail: 0,
                k_tail: 0,
                context_capacity: 8192,
                min_live: 1,
                max_live: 8192,
                inactive_rows: false,
            }],
        }
    }

    #[test]
    fn binding_rejects_each_identity_dimension_and_uncovered_rungs() {
        let expected = identity();
        let bundle = BoundCertificates {
            identity: expected.clone(),
            semantic: vec![],
            structural_cost: None,
            empirical_performance: None,
        };
        assert!(bundle.validate_binding(&expected, &expected.rungs).is_ok());
        let value = serde_json::to_value(&expected).unwrap();
        for key in value.as_object().unwrap().keys() {
            let mut changed = value.clone();
            match key.as_str() {
                "objects" => changed[key][0]["private_bytes"] = 4.into(),
                "rungs" => changed[key][0]["max_live"] = 4096.into(),
                _ if key.ends_with("sha256") => changed[key] = "b".repeat(64).into(),
                _ => changed[key] = "other".into(),
            }
            let changed = serde_json::from_value(changed).unwrap();
            assert!(
                bundle.validate_binding(&changed, &expected.rungs).is_err(),
                "{key}"
            );
        }
        let mut missing = expected.rungs.clone();
        missing[0].inactive_rows = true;
        assert!(bundle.validate_binding(&expected, &missing).is_err());
        assert!(bundle.validate_binding(&expected, &[]).is_err());
        let mut invalid = expected;
        invalid.objects[0].file = "../kernel.elf".into();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn structural_assumptions_and_semantic_scope_are_not_empirical_qualification() {
        let identity = identity();
        let subject = identity.digest().unwrap();
        let sha = "a".repeat(64);
        let mut bundle = BoundCertificates {
            identity: identity.clone(),
            semantic: vec![SemanticCertificate {
                subject_sha256: subject.clone(),
                scope: SemanticScope::RewriteNameCatalog,
                request_sha256: sha.clone(),
                verifier_sha256: sha.clone(),
                theorem: "catalog".into(),
            }],
            structural_cost: Some(StructuralCostCertificate {
                subject_sha256: subject,
                request_sha256: sha.clone(),
                verifier_sha256: sha,
                logical_read_bytes: 1024,
                logical_write_bytes: 32,
                flops: 512,
                launches: 3,
                critical_path_ns: 0,
                physical_hbm_bytes: Some(1056),
                physical_traffic_assumptions: vec![],
                duration_measurements_sha256: None,
            }),
            empirical_performance: None,
        };
        assert!(bundle.validate_binding(&identity, &identity.rungs).is_err());
        bundle.structural_cost.as_mut().unwrap().physical_hbm_bytes = None;
        assert!(bundle.validate_binding(&identity, &identity.rungs).is_ok());
        assert!(bundle.empirical_performance.is_none());
    }

    #[test]
    fn rung_guard_rejects_uncovered_rows_tails_and_inactive_entries() {
        let domain = RungDomain {
            phase: "decode".into(),
            rows: 16,
            n: 256,
            k: 512,
            heads: 8,
            n_tail: 0,
            k_tail: 0,
            context_capacity: 8192,
            min_live: 1,
            max_live: 8192,
            inactive_rows: false,
        };
        assert!(domain.validate().is_ok());
        assert!(domain.accepts(16, 10, 8192, false));
        assert!(!domain.accepts(32, 10, 8192, false));
        assert!(!domain.accepts(16, 0, 8192, true));
        assert!(!domain.accepts(16, 2, 8193, false));
        let mut invalid = domain;
        invalid.n_tail = 257;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn empirical_floor_requires_four_distinct_complete_arms() {
        let sha = "a".repeat(64);
        let mut cert = EmpiricalPerformanceCertificate {
            subject_sha256: sha.clone(),
            control_subject_sha256: sha.clone(),
            protocol_sha256: sha.clone(),
            verifier_sha256: sha.clone(),
            request_sha256: sha.clone(),
            metric: "latency_ns".into(),
            lower_is_better: true,
            arms: [100., 80., 102., 81.].map(|median| CampaignArm {
                id: median.to_string(),
                samples: 30,
                median,
                mad: 0.5,
                raw_samples_sha256: sha.clone(),
            }),
            numerical_gate_sha256: sha.clone(),
            artifact_gate_sha256: sha.clone(),
            serving_gate_sha256: sha,
        };
        assert_eq!(cert.improvement_beyond_floor().unwrap(), 17.5);
        cert.arms[3].samples = 29;
        assert!(cert.improvement_beyond_floor().is_err());
        cert.arms[3].samples = 30;
        cert.arms[3].median = 89.;
        assert!(cert.improvement_beyond_floor().is_err());
        cert.arms[3].median = 81.;
        cert.arms[3].id = cert.arms[0].id.clone();
        assert!(cert.improvement_beyond_floor().is_err());
    }
}
