use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

pub fn request_sha256(request: &serde_json::Value) -> Result<String, String> {
    // Cargo feature unification can select different JSON map orders in plowc and plowrt.
    let mut canonical = request.clone();
    canonical.sort_all_objects();
    let bytes = serde_json::to_vec(&canonical).map_err(|error| error.to_string())?;
    Ok(crate::decode_objects::image_sha256(&bytes))
}

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
    pub wave_size: u32,
    pub vgpr: Option<u32>,
    pub agpr: Option<u32>,
    pub sgpr: Option<u32>,
    pub private_bytes: u32,
    pub static_lds_bytes: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchBinding {
    pub program: u32,
    pub segment: u32,
    pub dispatch: u32,
    pub domain: u32,
    pub object_sha256: String,
    pub entry: String,
    pub abi_sha256: String,
    pub grid: [u32; 3],
    pub threads: u32,
    pub dynamic_lds_bytes: u32,
    pub occupancy: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionIdentity {
    pub schema: u32,
    pub model: String,
    pub revision: String,
    // Includes all storage/compute/accumulator dtypes, scales, rounding and table generation.
    pub precision_contract_sha256: String,
    pub checkpoint_sha256: String,
    pub packet_sha256: String,
    pub segment_dag_sha256: String,
    pub objects: Vec<ObjectBinding>,
    pub launches: Vec<LaunchBinding>,
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
        if self.schema != 2
            || self.model.is_empty()
            || self.revision.is_empty()
            || self.hardware.is_empty()
            || self.objects.is_empty()
            || self.launches.is_empty()
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
        let mut files = BTreeMap::new();
        for object in &self.objects {
            let mut path = std::path::Path::new(&object.file).components();
            if !matches!(path.next(), Some(std::path::Component::Normal(_)))
                || path.next().is_some()
                || object.file.as_bytes().contains(&0)
                || files.insert(&object.file, &object.sha256).is_some_and(|sha| sha != &object.sha256)
                || !objects.insert((&object.sha256, &object.entry))
                || object.entry.is_empty()
                || object.entry.as_bytes().contains(&0)
                || !is_sha256(&object.sha256)
                || !is_sha256(&object.abi_sha256)
                || !matches!(object.wave_size, 32 | 64)
                || object.vgpr.is_none()
                || object.agpr.is_none()
                || object.sgpr.is_none()
                || object.kernarg_bytes == 0
            {
                return Err("invalid, incomplete or duplicate immutable object binding".into());
            }
        }
        for (i, rung) in self.rungs.iter().enumerate() {
            rung.validate()?;
            if self.rungs[..i].contains(rung) {
                return Err("duplicate rung domain".into());
            }
        }
        let mut sites = BTreeSet::new();
        let mut covered = BTreeSet::new();
        for launch in &self.launches {
            let object = self.objects.iter().find(|object|
                object.sha256 == launch.object_sha256 && object.entry == launch.entry)
                .ok_or("launch references an unknown object/symbol")?;
            if launch.abi_sha256 != object.abi_sha256
                || launch.domain as usize >= self.rungs.len()
                || !sites.insert((launch.program, launch.segment, launch.dispatch, launch.domain))
                || launch.grid.contains(&0)
                || launch.threads == 0 || launch.threads > 1024
                || launch.threads % object.wave_size != 0
                || launch.occupancy.is_none_or(|n| n == 0)
                || object.static_lds_bytes.checked_add(launch.dynamic_lds_bytes).is_none()
            {
                return Err("invalid, incomplete or duplicate launch/domain/ABI binding".into());
            }
            covered.insert(launch.domain as usize);
        }
        if covered.len() != self.rungs.len() {
            return Err("rung domain has no selected launch binding".into());
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        Ok(crate::decode_objects::image_sha256(&bytes))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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
    /// `media_geometry.v1`: speech/multimodal contract geometry of a bundle, checked at
    /// qualification (`plowrt qualify`), not carried as a compiler receipt.
    MediaGeometry,
    /// `kv_ring.v1`: packed prefill writes never overwrite a ring row their own queries read,
    /// checked at qualification from the packet's `live_kv`/`packed_prefill` sections.
    KvRing,
    /// `speech_fusion.v1`: the emitted operands of fused speech sites (LayerNorm prologue,
    /// Conv1dF32 row_scale) meet `Plow.Speech`'s preconditions, checked at qualification.
    SpeechFusion,
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

/// `<stem>.lean-checks.json` beside a sidecar packet (`encoder.pkt`, `codec.pkt`, `s3gen.pkt`):
/// one logical-tensor-effects check per DISTINCT obligation (capacity programs repeat them), and
/// `program_checks[p]` = the check whose request is program `p`'s obligation (`None` = gap).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarCheckReceipts {
    pub schema: u32,
    pub packet_sha256: String,
    pub compiler_sha256: String,
    pub checks: Vec<CompileCheckReceipt>,
    pub program_checks: Vec<Option<usize>>,
}

pub fn sidecar_checks_file(packet: &std::path::Path) -> std::path::PathBuf {
    let stem = packet.file_stem().and_then(|s| s.to_str()).unwrap_or("packet");
    packet.with_file_name(format!("{stem}.{PACKET_CHECKS_FILE}"))
}

impl SidecarCheckReceipts {
    pub fn validate_packet(&self, packet: &[u8]) -> Result<(), String> {
        if self.schema != 1
            || !is_sha256(&self.compiler_sha256)
            || self.packet_sha256 != crate::decode_objects::image_sha256(packet)
        {
            return Err("sidecar check receipt does not match loaded packet".into());
        }
        let mut digests = BTreeSet::new();
        for (index, check) in self.checks.iter().enumerate() {
            let first = self.program_checks.iter().position(|c| *c == Some(index));
            if check.scope != SemanticScope::LogicalTensorEffects
                || check.checkpoint != "D"
                || check.program != first
                || check.program.is_none()
                || check.request.get("memory_effects").is_none()
                || !is_sha256(&check.verifier_sha256)
                || check.request_sha256 != request_sha256(&check.request)?
                || !digests.insert(&check.request_sha256)
                || check.response.get("ok").and_then(serde_json::Value::as_bool) != Some(true)
                || check.response.get("checkpoint").and_then(serde_json::Value::as_str) != Some("D")
            {
                return Err("invalid, unreferenced or duplicated sidecar check receipt".into());
            }
        }
        if self.program_checks.iter().flatten().any(|&i| i >= self.checks.len()) {
            return Err("sidecar program check out of range".into());
        }
        Ok(())
    }
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
                || check.request_sha256 != request_sha256(&check.request)?
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

/// How load and artifact qualification treat compiler receipts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerificationPolicy {
    /// Pre-policy behavior: absent receipts or an unavailable/changed verifier load unverified.
    Off,
    /// Derive the full obligation set and report every gap; reject only what `Off` rejects.
    Report,
    /// Any gap rejects: missing/unrunnable/unapproved verifier, absent, unused, duplicated or
    /// wrong-program receipts, and uncovered required obligations.
    Strict,
}

impl VerificationPolicy {
    pub const VALUES: &'static [&'static str] = &["off", "report", "strict"];

    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "off" => Ok(Self::Off),
            "report" => Ok(Self::Report),
            "strict" => Ok(Self::Strict),
            other => Err(format!("unknown verification policy {other:?} (off, report, strict)")),
        }
    }
}

/// One verifier executable approved for receipts: the sha256 of the immutable `plow_verify` image
/// built from Lean sources whose digest is `lean_sources_sha256`. `commit` is traceability only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovedVerifier {
    pub sha256: String,
    pub lean_sources_sha256: String,
    pub toolchain: String,
    pub commit: String,
    pub note: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApprovedVerifiers {
    pub schema: u32,
    pub verifiers: Vec<ApprovedVerifier>,
}

pub const APPROVED_VERIFIERS_JSON: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../../lean-plow/approved-verifiers.json"));

pub fn approved_verifiers() -> Result<ApprovedVerifiers, String> {
    let list: ApprovedVerifiers =
        serde_json::from_str(APPROVED_VERIFIERS_JSON).map_err(|error| error.to_string())?;
    if list.schema != 1
        || list.verifiers.is_empty()
        || list.verifiers.iter().any(|v| {
            !is_sha256(&v.sha256) || !is_sha256(&v.lean_sources_sha256) || v.toolchain.is_empty()
        })
    {
        return Err("invalid approved verifier list".into());
    }
    Ok(list)
}

pub fn is_approved_verifier(sha256: &str) -> bool {
    approved_verifiers().is_ok_and(|list| list.verifiers.iter().any(|v| v.sha256 == sha256))
}

/// One required check: a scope, its endpoint and the packet subject it covers.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Obligation {
    pub scope: SemanticScope,
    pub checkpoint: &'static str,
    pub program: Option<usize>,
    /// The producing instruction, for per-site scopes (layout).
    pub site: Option<usize>,
}

/// Completeness of one packet's receipts against the obligations derived from its programs.
/// It does not run the verifier; replay results are added by the caller as gaps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Qualification {
    pub packet_sha256: String,
    pub programs: usize,
    pub required: BTreeSet<Obligation>,
    pub satisfied: BTreeSet<Obligation>,
    /// Measured/GEMM policy receipts: bound to the wire when present, never required.
    pub policy_receipts: usize,
    pub receipt_verifiers: BTreeSet<String>,
    pub gaps: Vec<String>,
}

impl Qualification {
    fn new(packet: &[u8], programs: usize, required: BTreeSet<Obligation>) -> Self {
        Self {
            packet_sha256: crate::decode_objects::image_sha256(packet),
            programs,
            required,
            satisfied: BTreeSet::new(),
            policy_receipts: 0,
            receipt_verifiers: BTreeSet::new(),
            gaps: Vec::new(),
        }
    }

    pub fn qualified(&self) -> bool {
        self.gaps.is_empty() && self.satisfied == self.required
    }

    /// Digest of the required scope set; replay caches key on it so a verdict for one scope set
    /// is never reused for another.
    pub fn scope_set_sha256(&self) -> String {
        let bytes = serde_json::to_vec(&self.required).expect("obligations serialize");
        crate::decode_objects::image_sha256(&bytes)
    }

    fn satisfy(&mut self, obligation: Obligation, what: &str) {
        if !self.required.contains(&obligation) {
            self.gaps.push(format!("{what}: unused receipt {obligation:?}"));
        } else if !self.satisfied.insert(obligation.clone()) {
            self.gaps.push(format!("{what}: duplicate receipt {obligation:?}"));
        }
    }

    fn missing(&mut self, packet: &crate::program::Packet<'_>) {
        let mut by_scope: BTreeMap<SemanticScope, Vec<Option<usize>>> = BTreeMap::new();
        for obligation in self.required.difference(&self.satisfied) {
            by_scope.entry(obligation.scope).or_default().push(obligation.program);
        }
        for (scope, programs) in by_scope {
            let reason = match (scope, programs.iter().flatten().next()) {
                (SemanticScope::LogicalTensorEffects, Some(&p)) => {
                    crate::logical_effects::obligation(packet, p).err()
                }
                _ => None,
            };
            let programs: Vec<_> = programs.iter().map(|p| p.map_or(-1, |p| p as i64)).collect();
            self.gaps.push(format!(
                "{scope:?} missing for {} of {} programs {programs:?}{}",
                programs.len(),
                self.programs,
                reason.map(|r| format!(" (first: {r})")).unwrap_or_default()
            ));
        }
    }
}

fn layout_sites(program: &crate::program::Program<'_>) -> Vec<usize> {
    use packet::dev::DevOp;
    program
        .insts
        .iter()
        .enumerate()
        .filter(|(_, d)| d.op == DevOp::FlashMerge as u16 && d.i[4] == 1024)
        .map(|(i, _)| i)
        .collect()
}

/// Every obligation a compiler (`plowc` devblob) packet must carry: rewrite bodies once; coarse
/// dependency preservation and logical tensor effects for every program; the padded layout
/// mapping for every padded MLA producer.
pub fn required_packet_obligations(packet: &crate::program::Packet<'_>) -> BTreeSet<Obligation> {
    let mut required = BTreeSet::from([Obligation {
        scope: SemanticScope::RewriteBodyExpansion,
        checkpoint: "A",
        program: None,
        site: None,
    }]);
    for (p, program) in packet.programs.iter().enumerate() {
        for (scope, checkpoint) in [
            (SemanticScope::CoarseDependencyPreservation, "D"),
            (SemanticScope::LogicalTensorEffects, "D"),
        ] {
            required.insert(Obligation { scope, checkpoint, program: Some(p), site: None });
        }
        for site in layout_sites(program) {
            required.insert(Obligation {
                scope: SemanticScope::LayoutMapping,
                checkpoint: "L",
                program: Some(p),
                site: Some(site),
            });
        }
    }
    required
}

/// Reconstruct a receipt's obligation from the loaded packet. `Err` = the receipt does not
/// describe these bytes.
pub fn bind_packet_receipt(
    packet: &crate::program::Packet<'_>,
    check: &CompileCheckReceipt,
) -> Result<Option<Obligation>, String> {
    let program = match check.program {
        None => None,
        Some(index) => Some((index, packet.programs.get(index).ok_or("receipt program is absent from packet")?)),
    };
    let obligation = |scope, checkpoint, site| {
        Ok(Some(Obligation { scope, checkpoint, program: check.program, site }))
    };
    match (check.scope, program) {
        (SemanticScope::RewriteBodyExpansion, None) => obligation(check.scope, "A", None),
        (SemanticScope::MeasuredPolicy, None) => Ok(None),
        (SemanticScope::SelectedGemmPolicy, Some((index, _))) => {
            let expected = crate::gemm_policy::binding(packet, index, &check.request)?;
            if check.request.get("wire_binding") != Some(&expected) {
                return Err("GEMM policy differs from loaded instruction/placement".into());
            }
            Ok(None)
        }
        (SemanticScope::CoarseDependencyPreservation, Some((_, program))) => {
            let expected = crate::logical_effects::coarse_protocol(program)?;
            if check.request.get("protocol") != Some(&expected)
                || check.request.pointer("/task_graph/n").and_then(serde_json::Value::as_u64)
                    != Some(program.insts.len() as u64)
            {
                return Err("dependency receipt differs from loaded wire counters".into());
            }
            obligation(check.scope, "D", None)
        }
        (SemanticScope::LogicalTensorEffects, Some((index, _))) => {
            if check.request != crate::logical_effects::obligation(packet, index)? {
                return Err("logical effects differ from loaded operands/counters".into());
            }
            obligation(check.scope, "D", None)
        }
        (SemanticScope::LayoutMapping, Some((_, program))) => {
            let site = check
                .request
                .get("producer_instruction")
                .and_then(serde_json::Value::as_u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or("layout producer index missing")?;
            let producer = program.insts.get(site).ok_or("layout producer absent")?;
            let consumer = site
                .checked_add(1)
                .and_then(|i| program.insts.get(i))
                .ok_or("layout consumer absent")?;
            let capacity = packet
                .tensors
                .get(producer.t[0] as usize)
                .ok_or("layout output absent")?
                .bytes;
            if check.request != mla_layout_obligation(site, producer, consumer, capacity)? {
                return Err("layout obligation differs from loaded instructions".into());
            }
            obligation(check.scope, "L", Some(site))
        }
        (scope, _) => Err(format!("unsupported receipt scope {scope:?} for this program binding")),
    }
}

/// Completeness of a compiler packet's receipts. Binding failures are gaps here; the runtime's
/// legacy path reports them as rejections.
pub fn qualify_packet(
    receipts: Option<&PacketCheckReceipts>,
    raw: &[u8],
    packet: &crate::program::Packet<'_>,
) -> Qualification {
    let mut q = Qualification::new(raw, packet.programs.len(), required_packet_obligations(packet));
    let Some(receipts) = receipts else {
        q.gaps.push("no compiler check receipts".into());
        q.missing(packet);
        return q;
    };
    if let Err(error) = receipts.validate_packet(raw) {
        q.gaps.push(error);
        q.missing(packet);
        return q;
    }
    for (index, check) in receipts.checks.iter().enumerate() {
        let what = format!("receipt {index} ({:?}, program {:?})", check.scope, check.program);
        q.receipt_verifiers.insert(check.verifier_sha256.clone());
        if !is_approved_verifier(&check.verifier_sha256) {
            q.gaps.push(format!("{what}: verifier {} is not approved", check.verifier_sha256));
        }
        match bind_packet_receipt(packet, check) {
            Ok(Some(obligation)) => q.satisfy(obligation, &what),
            Ok(None) => q.policy_receipts += 1,
            Err(error) => q.gaps.push(format!("{what}: {error}")),
        }
    }
    q.missing(packet);
    q
}

/// Completeness of a sidecar packet's receipts: every program maps to a check whose request is
/// that program's reconstructed logical-effects obligation.
pub fn qualify_sidecar(
    receipts: Option<&SidecarCheckReceipts>,
    raw: &[u8],
    packet: &crate::program::Packet<'_>,
) -> Qualification {
    let required = (0..packet.programs.len())
        .map(|p| Obligation {
            scope: SemanticScope::LogicalTensorEffects,
            checkpoint: "D",
            program: Some(p),
            site: None,
        })
        .collect();
    let mut q = Qualification::new(raw, packet.programs.len(), required);
    let Some(receipts) = receipts else {
        q.gaps.push("no sidecar check receipts".into());
        q.missing(packet);
        return q;
    };
    if let Err(error) = receipts.validate_packet(raw) {
        q.gaps.push(error);
        q.missing(packet);
        return q;
    }
    if receipts.program_checks.len() != packet.programs.len() {
        q.gaps.push(format!(
            "program_checks covers {} of {} programs",
            receipts.program_checks.len(),
            packet.programs.len()
        ));
    }
    for check in &receipts.checks {
        q.receipt_verifiers.insert(check.verifier_sha256.clone());
        if !is_approved_verifier(&check.verifier_sha256) {
            q.gaps.push(format!("sidecar verifier {} is not approved", check.verifier_sha256));
        }
    }
    for (p, entry) in receipts.program_checks.iter().enumerate().take(packet.programs.len()) {
        let Some(index) = *entry else { continue };
        let Some(check) = receipts.checks.get(index) else { continue };
        match crate::logical_effects::obligation(packet, p) {
            Ok(request) if request == check.request => q.satisfy(
                Obligation {
                    scope: SemanticScope::LogicalTensorEffects,
                    checkpoint: "D",
                    program: Some(p),
                    site: None,
                },
                &format!("program {p}"),
            ),
            Ok(_) => q.gaps.push(format!("program {p}: check {index} is not this program's obligation")),
            Err(error) => q.gaps.push(format!("program {p}: {error}")),
        }
    }
    q.missing(packet);
    q
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
    #[test]
    fn request_digest_is_independent_of_json_map_order() {
        let first: serde_json::Value = serde_json::from_str(
            r#"{"z":[{"b":2,"a":1}],"a":{"d":4,"c":3}}"#,
        ).unwrap();
        let second: serde_json::Value = serde_json::from_str(
            r#"{"a":{"c":3,"d":4},"z":[{"a":1,"b":2}]}"#,
        ).unwrap();
        let expected = crate::decode_objects::image_sha256(
            br#"{"a":{"c":3,"d":4},"z":[{"a":1,"b":2}]}"#,
        );
        assert_eq!(super::request_sha256(&first).unwrap(), expected);
        assert_eq!(super::request_sha256(&second).unwrap(), expected);
        let mut changed = second;
        changed["z"][0]["b"] = serde_json::json!(3);
        assert_ne!(super::request_sha256(&changed).unwrap(), expected);
    }

    use super::*;

    fn identity() -> ExecutionIdentity {
        let sha = "a".repeat(64);
        ExecutionIdentity {
            schema: 2,
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
                abi_sha256: sha.clone(),
                kernarg_bytes: 64,
                wave_size: 64,
                vgpr: Some(80),
                agpr: Some(0),
                sgpr: Some(64),
                private_bytes: 0,
                static_lds_bytes: 0,
            }],
            launches: vec![LaunchBinding {
                program: 0, segment: 0, dispatch: 0, domain: 0,
                object_sha256: sha.clone(), entry: "kernel".into(), abi_sha256: sha,
                grid: [256, 1, 1], threads: 256,
                dynamic_lds_bytes: 0,
                occupancy: Some(4),
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
                "schema" => changed[key] = 3.into(),
                "objects" => changed[key][0]["private_bytes"] = 4.into(),
                "launches" => changed[key][0]["grid"][0] = 128.into(),
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
    fn one_immutable_entry_can_bind_multiple_launch_sites_and_domains() {
        let mut id = identity();
        let mut domain = id.rungs[0].clone();
        domain.rows = 32;
        id.rungs.push(domain);
        let mut launch = id.launches[0].clone();
        launch.program = 1;
        launch.domain = 1;
        launch.grid = [128, 2, 1];
        launch.threads = 512;
        launch.dynamic_lds_bytes = 1024;
        launch.occupancy = Some(2);
        id.launches.push(launch.clone());
        launch.dispatch = 1;
        id.launches.push(launch);
        assert_eq!(id.objects.len(), 1);
        assert!(id.validate().is_ok());
        let json = serde_json::to_value(&id).unwrap();
        assert!(json["objects"][0].get("grid").is_none());
        assert_eq!(serde_json::from_value::<ExecutionIdentity>(json).unwrap(), id);
        id.launches.push(id.launches[1].clone());
        assert!(id.validate().is_err());
    }

    #[test]
    fn launch_references_require_exact_object_symbol_abi_and_domain() {
        for mutation in 0..7 {
            let mut id = identity();
            match mutation {
                0 => id.launches[0].object_sha256 = "b".repeat(64),
                1 => id.launches[0].entry = "other".into(),
                2 => id.launches[0].abi_sha256 = "b".repeat(64),
                3 => id.launches[0].domain = 1,
                4 => id.launches[0].threads = 33,
                5 => id.launches[0].grid[1] = 0,
                _ => { let mut d = id.rungs[0].clone(); d.rows = 32; id.rungs.push(d); }
            }
            assert!(id.validate().is_err(), "mutation {mutation}");
        }
    }

    #[test]
    fn missing_resources_are_unknown_not_synthesized_zero() {
        for resource in ["vgpr", "agpr", "sgpr"] {
            let mut json = serde_json::to_value(identity()).unwrap();
            json["objects"][0].as_object_mut().unwrap().remove(resource);
            let id: ExecutionIdentity = serde_json::from_value(json).unwrap();
            assert!(id.validate().is_err(), "{resource}");
            assert!(serde_json::to_value(id).unwrap()["objects"][0][resource].is_null());
        }
        let mut id = identity();
        id.launches[0].occupancy = None;
        assert!(id.validate().is_err());
        id.launches[0].occupancy = Some(0);
        assert!(id.validate().is_err());
        let mut old = serde_json::to_value(identity()).unwrap();
        old.as_object_mut().unwrap().remove("launches");
        assert!(serde_json::from_value::<ExecutionIdentity>(old).is_err());
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

    fn residual_program(extra: bool) -> packet::devbuild::Program {
        use packet::dev::DevOp;
        let mut b = packet::devbuild::Builder::new(1);
        let x = b.tensor("x", 16);
        let y = b.tensor("y", 16);
        let first = b.emit(DevOp::Residual, vec![0], &[], |d| {
            d.t[..3].copy_from_slice(&[y, x, x]);
            d.i[0] = 8;
        });
        let second = b.emit(DevOp::Residual, vec![0], &[first], |d| {
            d.t[..3].copy_from_slice(&[x, y, y]);
            d.i[0] = 8;
        });
        if extra {
            b.emit(DevOp::Residual, vec![0], &[second], |d| {
                d.t[..3].copy_from_slice(&[y, x, x]);
                d.i[0] = 8;
            });
        }
        b.finish()
    }

    fn two_program_model() -> packet::devbuild::Model {
        let first = residual_program(false);
        packet::devbuild::Model {
            n_cu: 1,
            target: 0,
            tensors: first.tensors.clone(),
            progs: vec![first, residual_program(true)],
            kv_row_insts: vec![],
            prog_t: vec![1, 1],
            gen: vec![],
        }
    }

    fn approved() -> String {
        approved_verifiers().unwrap().verifiers[0].sha256.clone()
    }

    fn receipt(program: Option<usize>, scope: SemanticScope, checkpoint: &str, request: serde_json::Value) -> CompileCheckReceipt {
        CompileCheckReceipt {
            program,
            scope,
            checkpoint: checkpoint.into(),
            request_sha256: request_sha256(&request).unwrap(),
            verifier_sha256: approved(),
            request,
            response: serde_json::json!({"ok": true, "checkpoint": checkpoint}),
        }
    }

    fn complete_receipts(model: &packet::devbuild::Model) -> (Vec<u8>, PacketCheckReceipts) {
        let raw = model.to_blob();
        let rewrite = serde_json::json!({"rules": ["r"], "source_sha256": "c".repeat(64),
            "bodies": [{"name": "r", "lhs": [], "rhs": []}]});
        let mut checks = vec![receipt(None, SemanticScope::RewriteBodyExpansion, "A", rewrite)];
        crate::program::with_model(model, |packet| {
            for (p, program) in packet.programs.iter().enumerate() {
                let coarse = serde_json::json!({"task_graph": {"n": program.insts.len(), "edges": []},
                    "protocol": crate::logical_effects::coarse_protocol(program).unwrap(),
                    "dependency_paths": [], "address_map": []});
                checks.push(receipt(Some(p), SemanticScope::CoarseDependencyPreservation, "D", coarse));
                let effects = crate::logical_effects::obligation(packet, p).unwrap();
                checks.push(receipt(Some(p), SemanticScope::LogicalTensorEffects, "D", effects));
            }
        });
        let receipts = PacketCheckReceipts {
            schema: 1,
            packet_sha256: crate::decode_objects::image_sha256(&raw),
            compiler_sha256: "a".repeat(64),
            checks,
        };
        (raw, receipts)
    }

    fn qualify(model: &packet::devbuild::Model, raw: &[u8], receipts: Option<&PacketCheckReceipts>) -> Qualification {
        crate::program::with_model(model, |packet| qualify_packet(receipts, raw, packet))
    }

    #[test]
    fn strict_qualification_requires_every_obligation_once_from_an_approved_verifier() {
        let model = two_program_model();
        let (raw, receipts) = complete_receipts(&model);
        let q = qualify(&model, &raw, Some(&receipts));
        assert!(q.qualified(), "{:?}", q.gaps);
        assert_eq!(q.required.len(), 1 + 2 * 2);
        assert!(!qualify(&model, &raw, None).qualified());
        for mutation in 0..8 {
            let mut bad = receipts.clone();
            match mutation {
                0 => { bad.checks.pop(); }
                1 => bad.checks[2].program = Some(7),
                2 => bad.checks[1].verifier_sha256 = "b".repeat(64),
                3 => {
                    bad.checks[2].request["accesses"] = serde_json::json!([]);
                    bad.checks[2].request_sha256 = request_sha256(&bad.checks[2].request).unwrap();
                }
                4 => {
                    // Same obligation twice under a different request digest.
                    let mut copy = bad.checks[1].clone();
                    copy.request["address_map"] = serde_json::json!([[]]);
                    copy.request_sha256 = request_sha256(&copy.request).unwrap();
                    bad.checks.push(copy);
                }
                5 => bad.checks[1].program = None,
                6 => bad.packet_sha256 = "b".repeat(64),
                _ => {
                    // A program's effects receipt relabelled as the other program's.
                    bad.checks[2].program = Some(1);
                    bad.checks.remove(4);
                }
            }
            assert!(!qualify(&model, &raw, Some(&bad)).qualified(), "mutation {mutation}");
        }
        let mut changed = two_program_model();
        changed.progs[1].insts[2].t[0] = changed.progs[1].insts[2].t[1];
        let raw = changed.to_blob();
        let mut rebound = receipts;
        rebound.packet_sha256 = crate::decode_objects::image_sha256(&raw);
        assert!(!qualify(&changed, &raw, Some(&rebound)).qualified(),
            "an updated packet hash cannot reuse receipts for changed operands");
    }

    #[test]
    fn sidecar_qualification_maps_every_program_to_its_own_obligation() {
        let model = two_program_model();
        let raw = model.to_blob();
        let checks = crate::program::with_model(&model, |packet| {
            (0..2).map(|p| receipt(Some(p), SemanticScope::LogicalTensorEffects, "D",
                crate::logical_effects::obligation(packet, p).unwrap())).collect::<Vec<_>>()
        });
        let receipts = SidecarCheckReceipts {
            schema: 1,
            packet_sha256: crate::decode_objects::image_sha256(&raw),
            compiler_sha256: "a".repeat(64),
            checks,
            program_checks: vec![Some(0), Some(1)],
        };
        let q = |r: Option<&SidecarCheckReceipts>| {
            crate::program::with_model(&model, |packet| qualify_sidecar(r, &raw, packet))
        };
        assert!(q(Some(&receipts)).qualified(), "{:?}", q(Some(&receipts)).gaps);
        assert!(!q(None).qualified());
        for mutation in 0..4 {
            let mut bad = receipts.clone();
            match mutation {
                0 => bad.program_checks[1] = None,
                1 => { bad.program_checks.pop(); }
                2 => bad.program_checks = vec![Some(1), Some(0)],
                _ => bad.checks[1].verifier_sha256 = "b".repeat(64),
            }
            assert!(!q(Some(&bad)).qualified(), "mutation {mutation}");
        }
    }

    #[test]
    fn scope_set_digest_changes_with_the_required_obligations() {
        let model = two_program_model();
        let (raw, receipts) = complete_receipts(&model);
        let two = qualify(&model, &raw, Some(&receipts)).scope_set_sha256();
        let mut one = two_program_model();
        one.progs.truncate(1);
        one.prog_t.truncate(1);
        let raw = one.to_blob();
        assert_ne!(two, qualify(&one, &raw, None).scope_set_sha256());
    }

    #[test]
    fn approved_verifier_list_is_well_formed() {
        let list = approved_verifiers().unwrap();
        assert!(list.verifiers.iter().all(|v| is_approved_verifier(&v.sha256)));
        assert!(!is_approved_verifier(&"0".repeat(64)));
        assert_eq!(VerificationPolicy::parse("strict"), Ok(VerificationPolicy::Strict));
        assert!(VerificationPolicy::parse("on").is_err());
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
