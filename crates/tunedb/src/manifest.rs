//! Network block definitions and deterministic tune manifest generation.
//!
//! Bridges the structural block representation of a model
//! ([`NetworkBlockDefinition`]) into a canonicalized, execution-weighted
//! [`TuneManifest`]. Structurally identical layers collapse into a single
//! [`BlockDefinition`] weighted by `occurrences`, duplicate operators across
//! blocks collapse into unique [`OpCase`] records weighted by total dynamic
//! execution count, and both can be resolved against a [`TuneStore`] to surface
//! qualified kernels, fallback rankings, staleness reasons, and block
//! coverage.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::Path;

use kernelcaps::QuantScheme;
use serde::{Deserialize, Serialize};

use crate::record::{
    blockers_for, BlockDefinition, BlockMeasurement, Correctness, Digests, KernelMeasurement,
    OpCase, RecordState, Selection,
};
use crate::store::{StoreError, TuneStore};

/// Current wire/file format version emitted by [`TuneManifest`].
pub const MANIFEST_VERSION: u32 = 1;

/// Complete block-level structural definition of a network for tuning.
///
/// Captures every block/layer instance in the topology together with its
/// structural kind, distinguishing attributes, and ordered operator instances.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkBlockDefinition {
    /// Human-readable model identifier (e.g. `"gemma-4-31b"` or `"glm-5"`).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    /// Model family identifier (e.g. `"gemma4"`, `"llama"`, `"glm5"`).
    pub model_family: String,
    /// Block/layer instances making up the network before deduplication.
    pub blocks: Vec<NetworkBlock>,
}

impl NetworkBlockDefinition {
    /// Construct a network definition from a model family and block instances.
    pub fn new(model_family: impl Into<String>, blocks: Vec<NetworkBlock>) -> Self {
        Self {
            model: String::new(),
            model_family: model_family.into(),
            blocks,
        }
    }

    /// Attach a concrete model identifier to this definition.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Deserialize a [`NetworkBlockDefinition`] from JSON.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Serialize this definition as pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Canonicalize blocks and operators into a priority-ordered [`TuneManifest`].
    pub fn compile(&self) -> TuneManifest {
        let mut op_registry: BTreeMap<String, OpCase> = BTreeMap::new();
        let mut folded: BTreeMap<String, BlockAccumulator> = BTreeMap::new();

        for block in &self.blocks {
            let mut ordered_op_hashes = Vec::with_capacity(block.ops.len());
            for op in &block.ops {
                let op_case = op.to_op_case();
                let hash = op_case.hash.clone();
                op_registry.entry(hash.clone()).or_insert(op_case);
                ordered_op_hashes.push(hash);
            }

            let blk_hash = block_definition_hash(
                &self.model_family,
                &block.block_kind,
                &block.attributes,
                &ordered_op_hashes,
            );

            let acc = folded.entry(blk_hash.clone()).or_insert_with(|| {
                BlockAccumulator::new(
                    blk_hash,
                    &self.model_family,
                    &block.block_kind,
                    &block.attributes,
                    &ordered_op_hashes,
                )
            });
            acc.record_layer(block.layer_id);
        }

        let mut blocks: Vec<ManifestBlock> =
            folded.into_values().map(BlockAccumulator::finish).collect();
        blocks.sort_by(|a, b| {
            b.definition
                .occurrences
                .cmp(&a.definition.occurrences)
                .then_with(|| a.definition.hash.cmp(&b.definition.hash))
        });

        let op_cases = build_op_cases(&blocks, &op_registry);
        let total_op_instances = blocks
            .iter()
            .map(|b| {
                let per_block: u32 = b.op_counts.iter().map(|c| c.count).sum();
                per_block.saturating_mul(b.definition.occurrences)
            })
            .sum();

        let network_hash = network_hash(&self.model_family, &blocks);

        TuneManifest {
            version: MANIFEST_VERSION,
            network_hash,
            model: self.model.clone(),
            model_family: self.model_family.clone(),
            total_block_instances: u32::try_from(self.blocks.len()).unwrap_or(u32::MAX),
            unique_blocks: u32::try_from(blocks.len()).unwrap_or(u32::MAX),
            total_op_instances,
            unique_op_cases: u32::try_from(op_cases.len()).unwrap_or(u32::MAX),
            blocks,
            op_cases,
            coverage: None,
        }
    }

    /// Canonicalize into a [`TuneManifest`] and resolve against `store` for
    /// `hardware` under `want` build digests.
    pub fn compile_for(
        &self,
        store: &TuneStore,
        hardware: &str,
        want: &Digests,
    ) -> Result<TuneManifest, StoreError> {
        self.compile().resolve(store, hardware, want)
    }
}

/// One concrete block/layer instance before canonical folding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkBlock {
    /// Layer index in the source topology.
    pub layer_id: u32,
    /// Structural kind (e.g. `"attention_sliding"`, `"attention_full"`, `"moe"`).
    pub block_kind: String,
    /// Distinguishing structural attributes not captured by op shapes alone
    /// (e.g. `sliding_window=1024`, `attention=full`).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, String>,
    /// Ordered operator instances contained in this block.
    pub ops: Vec<NetworkOp>,
}

impl NetworkBlock {
    /// Construct a block instance with empty structural attributes.
    pub fn new(layer_id: u32, block_kind: impl Into<String>, ops: Vec<NetworkOp>) -> Self {
        Self {
            layer_id,
            block_kind: block_kind.into(),
            attributes: BTreeMap::new(),
            ops,
        }
    }

    /// Attach a key/value structural attribute.
    pub fn with_attr(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.attributes.insert(key.into(), value.into());
        self
    }

    /// Canonical structural hash of this block within `model_family`.
    pub fn hash(&self, model_family: &str) -> String {
        let op_hashes: Vec<String> = self.ops.iter().map(NetworkOp::op_case_hash).collect();
        block_definition_hash(model_family, &self.block_kind, &self.attributes, &op_hashes)
    }
}

/// One operator instance inside a [`NetworkBlock`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkOp {
    /// Operator class (e.g. `"gemm"`, `"gemv"`, `"attention"`, `"moe_decode"`).
    pub semantic: String,
    /// Inference phase or bucket label (e.g. `"prefill"`, `"decode"`).
    pub phase: String,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    /// Quantization / dtype scheme string matching `tunedb` store conventions.
    pub quant: String,
}

impl NetworkOp {
    /// Construct an operator instance explicitly.
    pub fn new(
        semantic: impl Into<String>,
        phase: impl Into<String>,
        m: i64,
        n: i64,
        k: i64,
        quant: impl Into<String>,
    ) -> Self {
        Self {
            semantic: semantic.into(),
            phase: phase.into(),
            m,
            n,
            k,
            quant: quant.into(),
        }
    }

    /// Construct a GEMM operator instance matching [`crate::gemm::gemm_op_case`].
    pub fn gemm(phase: impl Into<String>, m: i64, n: i64, k: i64, quant: QuantScheme) -> Self {
        Self::new("gemm", phase, m, n, k, format!("{quant:?}"))
    }

    /// Construct a GEMV operator instance matching [`crate::gemv::gemv_op_case`].
    pub fn gemv(
        family: impl Into<String>,
        phase: impl Into<String>,
        m: i64,
        n: i64,
        k: i64,
        quant: impl Into<String>,
    ) -> Self {
        Self::new(family, phase, m, n, k, quant)
    }

    /// Canonical content hash identifying this operator shape/phase/quantization.
    pub fn op_case_hash(&self) -> String {
        op_case_hash(
            &self.semantic,
            &self.phase,
            self.m,
            self.n,
            self.k,
            &self.quant,
        )
    }

    /// Convert to a content-addressed [`OpCase`].
    pub fn to_op_case(&self) -> OpCase {
        OpCase {
            hash: self.op_case_hash(),
            semantic: self.semantic.clone(),
            phase: self.phase.clone(),
            m: self.m,
            n: self.n,
            k: self.k,
            quant: self.quant.clone(),
        }
    }
}

/// Deterministic, execution-weighted tuning manifest produced from a
/// [`NetworkBlockDefinition`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TuneManifest {
    pub version: u32,
    pub network_hash: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    pub model_family: String,
    pub total_block_instances: u32,
    pub unique_blocks: u32,
    pub total_op_instances: u32,
    pub unique_op_cases: u32,
    /// Canonicalized block definitions sorted by `occurrences` descending.
    pub blocks: Vec<ManifestBlock>,
    /// Deduplicated operator cases sorted by `total_occurrences` descending.
    pub op_cases: Vec<ManifestOpCase>,
    /// Populated when resolved against a [`TuneStore`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<ManifestCoverage>,
}

impl TuneManifest {
    /// Resolve all block definitions and operator cases against `store` for
    /// `hardware` using current `want` digests.
    pub fn resolve(
        &self,
        store: &TuneStore,
        hardware: &str,
        want: &Digests,
    ) -> Result<TuneManifest, StoreError> {
        let kernels = store.load_kernels(hardware)?;
        let selections = store.load_selections(hardware)?;
        let block_measurements = store.load_block_measurements(hardware)?;

        let mut kernels_by_case: BTreeMap<&str, Vec<&KernelMeasurement>> = BTreeMap::new();
        for rec in &kernels {
            kernels_by_case
                .entry(rec.op_case.as_str())
                .or_default()
                .push(rec);
        }

        let mut selections_by_case: BTreeMap<&str, Vec<&Selection>> = BTreeMap::new();
        for sel in &selections {
            selections_by_case
                .entry(sel.op_case.as_str())
                .or_default()
                .push(sel);
        }

        let mut blocks_by_hash: BTreeMap<&str, Vec<&BlockMeasurement>> = BTreeMap::new();
        for bm in &block_measurements {
            blocks_by_hash
                .entry(bm.block.as_str())
                .or_default()
                .push(bm);
        }

        let mut resolved_ops = Vec::with_capacity(self.op_cases.len());
        let mut op_status_by_hash: BTreeMap<String, CoverageState> = BTreeMap::new();
        let mut coverage = ManifestCoverage {
            hardware: hardware.to_string(),
            ..Default::default()
        };

        for entry in &self.op_cases {
            let resolution =
                resolve_op_entry(entry, hardware, &kernels_by_case, &selections_by_case, want);
            op_status_by_hash.insert(entry.op_case.hash.clone(), resolution.status);
            coverage.record_op(resolution.status, entry.total_occurrences);

            let mut updated = entry.clone();
            updated.resolution = Some(resolution);
            resolved_ops.push(updated);
        }

        let mut resolved_blocks = Vec::with_capacity(self.blocks.len());
        for blk in &self.blocks {
            let block_cov = resolve_block_entry(
                blk,
                hardware,
                blocks_by_hash
                    .get(blk.definition.hash.as_str())
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
                &op_status_by_hash,
                want,
            );
            coverage.record_block(&block_cov);

            let mut updated = blk.clone();
            updated.coverage = Some(block_cov);
            resolved_blocks.push(updated);
        }

        let mut out = self.clone();
        out.blocks = resolved_blocks;
        out.op_cases = resolved_ops;
        out.coverage = Some(coverage);
        Ok(out)
    }

    /// Return prioritized operator cases that still require tuning or refresh
    /// (every status other than [`CoverageState::Qualified`], or all cases if
    /// the manifest has not been resolved).
    pub fn actionable_op_cases(&self) -> Vec<&ManifestOpCase> {
        self.op_cases
            .iter()
            .filter(|op| {
                op.resolution
                    .as_ref()
                    .is_none_or(|r| r.status != CoverageState::Qualified)
            })
            .collect()
    }

    /// Synthesize [`Selection`] records for every [`ManifestOpCase`] that has a
    /// current qualified winning kernel in its resolution.
    pub fn derive_selections(&self, objective: &str) -> Vec<Selection> {
        self.op_cases
            .iter()
            .filter_map(|op| {
                let res = op.resolution.as_ref()?;
                if res.status != CoverageState::Qualified {
                    return None;
                }
                let best = res.best_kernel.as_ref()?;
                Some(Selection {
                    op_case: op.op_case.hash.clone(),
                    hardware: best.hardware.clone(),
                    profile: best.profile.clone(),
                    objective: objective.to_string(),
                    kernel_id: best.kernel_id,
                    kernel_name: best.kernel_name.clone(),
                    fallbacks: res.fallback_kernels.clone(),
                    tier: "measured".to_string(),
                    digests: best.digests.clone(),
                    campaign: best.campaign.clone(),
                })
            })
            .collect()
    }

    /// Return the resolved winning kernel id (if qualified) aligned with each
    /// unique `op_case` in `block_hash`'s [`BlockDefinition`].
    pub fn block_kernel_ids(&self, block_hash: &str) -> Option<Vec<Option<u16>>> {
        let blk = self
            .blocks
            .iter()
            .find(|b| b.definition.hash == block_hash)?;
        let by_op: BTreeMap<&str, Option<u16>> = self
            .op_cases
            .iter()
            .map(|op| {
                let winner = op
                    .resolution
                    .as_ref()
                    .and_then(|r| r.best_kernel.as_ref())
                    .map(|k| k.kernel_id);
                (op.op_case.hash.as_str(), winner)
            })
            .collect();
        Some(
            blk.definition
                .op_cases
                .iter()
                .map(|h| by_op.get(h.as_str()).copied().flatten())
                .collect(),
        )
    }

    /// Serialize the manifest as pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a [`TuneManifest`] from JSON.
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Write the manifest atomically to `path` as pretty JSON.
    pub fn save_json(&self, path: &Path) -> Result<(), StoreError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = self.to_json_pretty()?;
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        std::fs::write(&tmp, body)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load a [`TuneManifest`] from a JSON file at `path`.
    pub fn load_json(path: &Path) -> Result<Self, StoreError> {
        let body = std::fs::read_to_string(path)?;
        Ok(Self::from_json_str(&body)?)
    }
}

/// Canonicalized block entry inside [`TuneManifest`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestBlock {
    pub definition: BlockDefinition,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub attributes: BTreeMap<String, String>,
    /// Multiplicity of each unique `OpCase` inside a single instance of this block.
    pub op_counts: Vec<BlockOpCount>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub coverage: Option<BlockCoverage>,
}

impl ManifestBlock {
    /// First layer index carrying this canonical block (usable as `--block` representative).
    pub fn representative_layer(&self) -> Option<u32> {
        self.definition.layer_ids.first().copied()
    }
}

/// Count of how many times an `OpCase` runs within one instance of a block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockOpCount {
    pub op_case: String,
    pub count: u32,
}

/// Deduplicated, execution-weighted operator entry inside [`TuneManifest`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestOpCase {
    pub op_case: OpCase,
    /// Store lookup key compatible with `gemm_op_case` / `gemv_op_case`.
    pub store_key: String,
    /// Total dynamic execution count across all blocks in the network.
    pub total_occurrences: u32,
    /// Canonical block hashes referencing this operator case.
    pub blocks: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<OpResolution>,
}

/// Qualification / freshness state of a manifest entry against a [`TuneStore`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageState {
    Qualified,
    Stale,
    Provisional,
    Missing,
}

/// Store resolution details attached to a [`ManifestOpCase`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpResolution {
    pub status: CoverageState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_kernel: Option<KernelMeasurement>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallback_kernels: Vec<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<Selection>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stale_reasons: Vec<String>,
}

/// Store resolution summary attached to a [`ManifestBlock`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockCoverage {
    /// Status of whole-block [`BlockMeasurement`] records for this block hash.
    pub block_status: CoverageState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_measurement: Option<BlockMeasurement>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stale_reasons: Vec<String>,
    pub qualified_ops: u32,
    pub stale_ops: u32,
    pub provisional_ops: u32,
    pub missing_ops: u32,
}

/// Aggregate unique and execution-weighted coverage across a resolved manifest.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestCoverage {
    pub hardware: String,
    pub unique_ops_qualified: u32,
    pub unique_ops_stale: u32,
    pub unique_ops_provisional: u32,
    pub unique_ops_missing: u32,
    pub weighted_ops_qualified: u32,
    pub weighted_ops_stale: u32,
    pub weighted_ops_provisional: u32,
    pub weighted_ops_missing: u32,
    pub blocks_with_qualified_measurement: u32,
    pub blocks_fully_op_covered: u32,
}

impl ManifestCoverage {
    fn record_op(&mut self, status: CoverageState, weight: u32) {
        match status {
            CoverageState::Qualified => {
                self.unique_ops_qualified += 1;
                self.weighted_ops_qualified = self.weighted_ops_qualified.saturating_add(weight);
            }
            CoverageState::Stale => {
                self.unique_ops_stale += 1;
                self.weighted_ops_stale = self.weighted_ops_stale.saturating_add(weight);
            }
            CoverageState::Provisional => {
                self.unique_ops_provisional += 1;
                self.weighted_ops_provisional =
                    self.weighted_ops_provisional.saturating_add(weight);
            }
            CoverageState::Missing => {
                self.unique_ops_missing += 1;
                self.weighted_ops_missing = self.weighted_ops_missing.saturating_add(weight);
            }
        }
    }

    fn record_block(&mut self, cov: &BlockCoverage) {
        if cov.block_status == CoverageState::Qualified {
            self.blocks_with_qualified_measurement += 1;
        }
        if cov.stale_ops == 0 && cov.provisional_ops == 0 && cov.missing_ops == 0 {
            self.blocks_fully_op_covered += 1;
        }
    }
}

/// Compute the deterministic content hash for an operator specification.
pub fn op_case_hash(semantic: &str, phase: &str, m: i64, n: i64, k: i64, quant: &str) -> String {
    let mut enc = CanonicalEncoder::new("tunedb.op_case.v1");
    enc.push_str(semantic);
    enc.push_str(phase);
    enc.push_i64(m);
    enc.push_i64(n);
    enc.push_i64(k);
    enc.push_str(quant);
    enc.finish("op")
}

/// Canonical `TuneStore` op-case key (`"{semantic}/{m}x{n}x{k}/{quant}"`),
/// matching [`crate::gemm::gemm_op_case`] and [`crate::gemv::gemv_op_case`].
pub fn store_op_case_key(op: &OpCase) -> String {
    format!("{}/{}x{}x{}/{}", op.semantic, op.m, op.n, op.k, op.quant)
}

/// Compute the deterministic structural hash for a network block definition.
pub fn block_definition_hash(
    model_family: &str,
    block_kind: &str,
    attributes: &BTreeMap<String, String>,
    ordered_op_hashes: &[String],
) -> String {
    let mut enc = CanonicalEncoder::new("tunedb.block_definition.v1");
    enc.push_str(model_family);
    enc.push_str(block_kind);
    enc.push_usize(attributes.len());
    for (k, v) in attributes {
        enc.push_str(k);
        enc.push_str(v);
    }
    enc.push_usize(ordered_op_hashes.len());
    for h in ordered_op_hashes {
        enc.push_str(h);
    }
    enc.finish("blk")
}

fn network_hash(model_family: &str, canonical_blocks: &[ManifestBlock]) -> String {
    let mut enc = CanonicalEncoder::new("tunedb.network_manifest.v1");
    enc.push_str(model_family);
    enc.push_usize(canonical_blocks.len());
    for blk in canonical_blocks {
        enc.push_str(&blk.definition.hash);
        enc.push_u32(blk.definition.occurrences);
        enc.push_usize(blk.definition.layer_ids.len());
        for &id in &blk.definition.layer_ids {
            enc.push_u32(id);
        }
    }
    enc.finish("net")
}

struct CanonicalEncoder(Vec<u8>);

impl CanonicalEncoder {
    fn new(domain: &str) -> Self {
        let mut enc = Self(Vec::new());
        enc.push_str(domain);
        enc
    }

    fn push_usize(&mut self, v: usize) {
        self.0.extend_from_slice(&(v as u64).to_le_bytes());
    }

    fn push_u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn push_i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }

    fn push_str(&mut self, s: &str) {
        self.push_usize(s.len());
        self.0.extend_from_slice(s.as_bytes());
    }

    fn finish(self, prefix: &str) -> String {
        format!("{prefix}-{}", kernelcaps::build::digest(&self.0))
    }
}

struct BlockAccumulator {
    hash: String,
    model_family: String,
    block_kind: String,
    attributes: BTreeMap<String, String>,
    layer_ids: Vec<u32>,
    occurrences: u32,
    unique_ops: Vec<String>,
    op_counts: Vec<BlockOpCount>,
}

impl BlockAccumulator {
    fn new(
        hash: String,
        model_family: &str,
        block_kind: &str,
        attributes: &BTreeMap<String, String>,
        ordered_op_hashes: &[String],
    ) -> Self {
        let mut index_by_op: BTreeMap<&str, usize> = BTreeMap::new();
        let mut unique_ops = Vec::new();
        let mut op_counts: Vec<BlockOpCount> = Vec::new();

        for h in ordered_op_hashes {
            if let Some(&idx) = index_by_op.get(h.as_str()) {
                op_counts[idx].count += 1;
            } else {
                let idx = unique_ops.len();
                index_by_op.insert(h.as_str(), idx);
                unique_ops.push(h.clone());
                op_counts.push(BlockOpCount {
                    op_case: h.clone(),
                    count: 1,
                });
            }
        }

        Self {
            hash,
            model_family: model_family.to_string(),
            block_kind: block_kind.to_string(),
            attributes: attributes.clone(),
            layer_ids: Vec::new(),
            occurrences: 0,
            unique_ops,
            op_counts,
        }
    }

    fn record_layer(&mut self, layer_id: u32) {
        self.layer_ids.push(layer_id);
        self.occurrences = self.occurrences.saturating_add(1);
    }

    fn finish(mut self) -> ManifestBlock {
        self.layer_ids.sort_unstable();
        self.layer_ids.dedup();
        ManifestBlock {
            definition: BlockDefinition {
                hash: self.hash,
                model_family: self.model_family,
                block_kind: self.block_kind,
                layer_ids: self.layer_ids,
                occurrences: self.occurrences,
                op_cases: self.unique_ops,
            },
            attributes: self.attributes,
            op_counts: self.op_counts,
            coverage: None,
        }
    }
}

fn build_op_cases(
    sorted_blocks: &[ManifestBlock],
    registry: &BTreeMap<String, OpCase>,
) -> Vec<ManifestOpCase> {
    struct OpAcc {
        total_occurrences: u32,
        blocks: Vec<String>,
    }

    let mut accs: BTreeMap<String, OpAcc> = BTreeMap::new();
    for blk in sorted_blocks {
        for oc in &blk.op_counts {
            let entry = accs.entry(oc.op_case.clone()).or_insert_with(|| OpAcc {
                total_occurrences: 0,
                blocks: Vec::new(),
            });
            let weighted = blk.definition.occurrences.saturating_mul(oc.count);
            entry.total_occurrences = entry.total_occurrences.saturating_add(weighted);
            entry.blocks.push(blk.definition.hash.clone());
        }
    }

    let mut out: Vec<ManifestOpCase> = accs
        .into_iter()
        .filter_map(|(hash, acc)| {
            let op_case = registry.get(&hash)?.clone();
            let store_key = store_op_case_key(&op_case);
            Some(ManifestOpCase {
                op_case,
                store_key,
                total_occurrences: acc.total_occurrences,
                blocks: acc.blocks,
                resolution: None,
            })
        })
        .collect();

    out.sort_by(|a, b| {
        b.total_occurrences
            .cmp(&a.total_occurrences)
            .then_with(|| a.op_case.hash.cmp(&b.op_case.hash))
    });
    out
}

fn resolve_op_entry(
    entry: &ManifestOpCase,
    hardware: &str,
    kernels_by_case: &BTreeMap<&str, Vec<&KernelMeasurement>>,
    selections_by_case: &BTreeMap<&str, Vec<&Selection>>,
    want: &Digests,
) -> OpResolution {
    let mut candidates: Vec<&KernelMeasurement> = Vec::new();
    if let Some(recs) = kernels_by_case.get(entry.op_case.hash.as_str()) {
        candidates.extend(recs.iter().copied());
    }
    if entry.store_key != entry.op_case.hash {
        if let Some(recs) = kernels_by_case.get(entry.store_key.as_str()) {
            candidates.extend(recs.iter().copied());
        }
    }

    let usable_selection = [entry.op_case.hash.as_str(), entry.store_key.as_str()]
        .into_iter()
        .flat_map(|key| {
            selections_by_case
                .get(key)
                .map(Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .copied()
        })
        .rfind(|sel| sel.hardware == hardware && sel.digests.stale_against(want).is_empty())
        .cloned();

    let (mut qualified_current, fallback_status, stale_reasons) = classify_records(
        candidates
            .into_iter()
            .filter(|rec| rec.hardware == hardware && rec.qualification_blockers().is_empty())
            .map(|rec| (rec, &rec.state, &rec.digests)),
        want,
    );

    if !qualified_current.is_empty() {
        qualified_current.sort_by(|a, b| {
            a.stats
                .median_ns
                .total_cmp(&b.stats.median_ns)
                .then_with(|| a.kernel_id.cmp(&b.kernel_id))
        });
        let best = qualified_current[0].clone();
        let mut fallback_kernels = Vec::new();
        for rec in &qualified_current[1..] {
            if rec.kernel_id != best.kernel_id && !fallback_kernels.contains(&rec.kernel_id) {
                fallback_kernels.push(rec.kernel_id);
            }
        }
        return OpResolution {
            status: CoverageState::Qualified,
            best_kernel: Some(best),
            fallback_kernels,
            selection: usable_selection,
            stale_reasons: Vec::new(),
        };
    }

    OpResolution {
        status: fallback_status,
        best_kernel: None,
        fallback_kernels: Vec::new(),
        selection: usable_selection,
        stale_reasons,
    }
}

fn resolve_block_entry(
    blk: &ManifestBlock,
    hardware: &str,
    measurements: &[&BlockMeasurement],
    op_status_by_hash: &BTreeMap<String, CoverageState>,
    want: &Digests,
) -> BlockCoverage {
    let (mut qualified_current, fallback_status, stale_reasons) = classify_records(
        measurements
            .iter()
            .copied()
            .filter(|bm| {
                bm.hardware == hardware
                    && blockers_for(&Correctness::Pass, bm.stats.samples).is_empty()
            })
            .map(|bm| (bm, &bm.state, &bm.digests)),
        want,
    );

    let (block_status, best_measurement, final_stale_reasons) = if !qualified_current.is_empty() {
        qualified_current.sort_by(|a, b| a.stats.median_ns.total_cmp(&b.stats.median_ns));
        (
            CoverageState::Qualified,
            Some(qualified_current[0].clone()),
            Vec::new(),
        )
    } else {
        (fallback_status, None, stale_reasons)
    };

    let mut qualified_ops = 0;
    let mut stale_ops = 0;
    let mut provisional_ops = 0;
    let mut missing_ops = 0;
    for op_hash in &blk.definition.op_cases {
        match op_status_by_hash
            .get(op_hash)
            .copied()
            .unwrap_or(CoverageState::Missing)
        {
            CoverageState::Qualified => qualified_ops += 1,
            CoverageState::Stale => stale_ops += 1,
            CoverageState::Provisional => provisional_ops += 1,
            CoverageState::Missing => missing_ops += 1,
        }
    }

    BlockCoverage {
        block_status,
        best_measurement,
        stale_reasons: final_stale_reasons,
        qualified_ops,
        stale_ops,
        provisional_ops,
        missing_ops,
    }
}

fn classify_records<'a, T>(
    records: impl Iterator<Item = (T, &'a RecordState, &'a Digests)>,
    want: &Digests,
) -> (Vec<T>, CoverageState, Vec<String>) {
    let mut qualified_current = Vec::new();
    let mut stale_reasons: BTreeSet<String> = BTreeSet::new();
    let mut has_stale = false;
    let mut has_provisional = false;

    for (rec, state, digests) in records {
        let moved = digests.stale_against(want);
        match state {
            RecordState::Qualified if moved.is_empty() => qualified_current.push(rec),
            RecordState::Qualified => {
                has_stale = true;
                for c in moved {
                    stale_reasons.insert(c.to_string());
                }
            }
            RecordState::Stale { changed } => {
                has_stale = true;
                for c in changed {
                    stale_reasons.insert(c.clone());
                }
                for c in moved {
                    stale_reasons.insert(c.to_string());
                }
            }
            RecordState::Provisional if moved.is_empty() => has_provisional = true,
            RecordState::Provisional | RecordState::Rejected { .. } => {}
        }
    }

    let fallback = if has_stale {
        CoverageState::Stale
    } else if has_provisional {
        CoverageState::Provisional
    } else {
        CoverageState::Missing
    };
    (
        qualified_current,
        fallback,
        stale_reasons.into_iter().collect(),
    )
}

/// Fail when a resolved [`TuneManifest`] has any missing, stale, or provisional
/// operator case (used by `--strict` checks in CLIs and CI gates).
pub fn check_strict(manifest: &TuneManifest, cell: &str) -> Result<(), String> {
    let Some(cov) = &manifest.coverage else {
        return Err(format!(
            "tune manifest check failed (--strict) for {cell}: manifest has not been resolved"
        ));
    };
    let uncovered_unique =
        cov.unique_ops_missing + cov.unique_ops_stale + cov.unique_ops_provisional;
    if uncovered_unique > 0 {
        let uncovered_weighted =
            cov.weighted_ops_missing + cov.weighted_ops_stale + cov.weighted_ops_provisional;
        return Err(format!(
            "tune manifest check failed (--strict) for {cell}: {} missing, {} stale, \
             {} provisional out of {} unique op cases ({} / {} weighted executions uncovered)",
            cov.unique_ops_missing,
            cov.unique_ops_stale,
            cov.unique_ops_provisional,
            manifest.unique_op_cases,
            uncovered_weighted,
            manifest.total_op_instances,
        ));
    }
    Ok(())
}

/// Pick existing build digests recorded in `store` for `cell` (preferring the
/// most recent selectable record with no qualification blockers), or `None` if
/// `cell` has no kernel records.
pub fn infer_digests_for_cell(
    store: &TuneStore,
    cell: &str,
) -> Result<Option<Digests>, StoreError> {
    let kernels = store.load_kernels(cell)?;
    Ok(kernels
        .iter()
        .rev()
        .find(|k| {
            k.hardware == cell && k.state.is_selectable() && k.qualification_blockers().is_empty()
        })
        .or_else(|| {
            kernels
                .iter()
                .rev()
                .find(|k| k.hardware == cell && k.state.is_selectable())
        })
        .or_else(|| kernels.iter().rev().find(|k| k.hardware == cell))
        .map(|k| k.digests.clone()))
}

/// Format a human-readable coverage & priority report for a resolved [`TuneManifest`].
pub fn format_report(
    manifest: &TuneManifest,
    cell: &str,
    digest_note: &str,
    out_path: Option<&Path>,
) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "model       : {} ({})",
        manifest.model, manifest.model_family
    );
    let _ = writeln!(s, "network hash: {}", manifest.network_hash);
    let _ = writeln!(s, "cell        : {cell}");
    let _ = writeln!(s, "digests     : {digest_note}");
    if let Some(p) = out_path {
        let _ = writeln!(s, "wrote       : {}", p.display());
    }
    let _ = writeln!(
        s,
        "blocks      : {} total -> {} canonical",
        manifest.total_block_instances, manifest.unique_blocks
    );
    let _ = writeln!(
        s,
        "op cases    : {} dynamic -> {} unique",
        manifest.total_op_instances, manifest.unique_op_cases
    );
    let _ = writeln!(s);

    let _ = writeln!(s, "Canonical blocks:");
    for blk in &manifest.blocks {
        let rep = blk
            .representative_layer()
            .map(|id| id.to_string())
            .unwrap_or_else(|| "-".into());
        let blk_cov = blk
            .coverage
            .as_ref()
            .map(|c| {
                format!(
                    "blk:{:?} ops:{}/{}/{}/{} (Q/S/P/M)",
                    c.block_status, c.qualified_ops, c.stale_ops, c.provisional_ops, c.missing_ops
                )
            })
            .unwrap_or_default();
        let _ = writeln!(
            s,
            "  {}  x{:<3}  rep={:<3}  kind={:<18}  ops={:<3}  {}",
            blk.definition.hash,
            blk.definition.occurrences,
            rep,
            blk.definition.block_kind,
            blk.definition.op_cases.len(),
            blk_cov
        );
    }

    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "Prioritized op cases (top {} shown):",
        manifest.op_cases.len().min(30)
    );
    for op in manifest.op_cases.iter().take(30) {
        let status = match &op.resolution {
            None => "UNRESOLVED".to_string(),
            Some(r) => match r.status {
                CoverageState::Qualified => {
                    let k = r.best_kernel.as_ref().unwrap();
                    format!(
                        "QUALIFIED ({}#{}, {:.1} us)",
                        k.kernel_name,
                        k.kernel_id,
                        k.stats.median_ns / 1000.0
                    )
                }
                CoverageState::Stale => {
                    format!("STALE ({})", r.stale_reasons.join(","))
                }
                CoverageState::Provisional => "PROVISIONAL".to_string(),
                CoverageState::Missing => "MISSING".to_string(),
            },
        };
        let _ = writeln!(
            s,
            "  x{:<4}  {:<7}  {:<36}  {}",
            op.total_occurrences, op.op_case.phase, op.store_key, status
        );
    }

    if let Some(cov) = &manifest.coverage {
        let _ = writeln!(s);
        let _ = writeln!(
            s,
            "Coverage    : unique {}/{}/{}/{} (Q/S/P/M) | weighted {}/{}/{}/{} (Q/S/P/M)",
            cov.unique_ops_qualified,
            cov.unique_ops_stale,
            cov.unique_ops_provisional,
            cov.unique_ops_missing,
            cov.weighted_ops_qualified,
            cov.weighted_ops_stale,
            cov.weighted_ops_provisional,
            cov.weighted_ops_missing,
        );
    }
    s
}

#[cfg(feature = "models")]
mod nn_bridge {
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

    use kernelcaps::QuantScheme;
    use nn_graph::{ActKind, DType, Graph, Op, SymId, TensorInfo};
    use plow_asset::Phase;

    use super::{NetworkBlock, NetworkBlockDefinition, NetworkOp};

    /// Build a [`NetworkBlockDefinition`] from a HuggingFace `config.json` string
    /// evaluated across `batches`, `seqs`, and `phases` without checkpoint weights.
    pub fn from_config_json(
        config_json: &str,
        model_name: &str,
        batches: &[i64],
        seqs: &[i64],
        phases: &[Phase],
        image_size: i64,
    ) -> Result<NetworkBlockDefinition, Box<dyn std::error::Error>> {
        let model_family = model_family_from_json(config_json);
        let bucket = nn_graph::models::ShapeBucket::square(image_size.max(1));
        let graph =
            nn_graph::models::build_text_generation_from_config_json_at(config_json, &bucket)
                .or_else(|_| nn_graph::models::build_from_config_json_at(config_json, &bucket))?;
        Ok(from_graph(
            &graph,
            model_name,
            &model_family,
            batches,
            seqs,
            phases,
        ))
    }

    /// Lower a `nn_graph::Graph` into a [`NetworkBlockDefinition`] evaluated across
    /// the requested batch/sequence buckets and phases.
    pub fn from_graph(
        graph: &Graph,
        model_name: &str,
        model_family: &str,
        batches: &[i64],
        seqs: &[i64],
        phases: &[Phase],
    ) -> NetworkBlockDefinition {
        let fp8_weights: HashSet<&str> = graph
            .fp8_scale_bindings
            .iter()
            .map(|b| b.weight.as_str())
            .collect();
        let slots = expand_slots(batches, seqs, phases);

        let bound_slots: Vec<(&str, HashMap<SymId, i64>)> = slots
            .iter()
            .map(|&(label, b, s)| {
                let mut bindings = HashMap::new();
                if let Some(id) = graph.syms.id_of("B") {
                    bindings.insert(id, b);
                }
                if let Some(id) = graph.syms.id_of("S") {
                    bindings.insert(id, s);
                }
                (label, bindings)
            })
            .collect();

        let mut blocks = Vec::with_capacity(graph.blocks.len() + 1);
        for blk_idx in 0..graph.blocks.len() {
            let label = graph.block_label(blk_idx as u32).unwrap_or("block");
            let kind = canonical_block_kind(label);
            let mut attrs = BTreeMap::new();
            let mut ops = Vec::new();

            for (phase_label, bindings) in &bound_slots {
                for (_, node) in graph.block_nodes(blk_idx as u32) {
                    collect_node_attrs(node, &mut attrs);
                    if let Some(op) = lower_node(graph, node, phase_label, &fp8_weights, bindings) {
                        ops.push(op);
                    }
                }
            }

            let mut nb = NetworkBlock::new(blk_idx as u32, kind, ops);
            nb.attributes = attrs;
            blocks.push(nb);
        }

        let mut top_attrs = BTreeMap::new();
        let mut top_ops = Vec::new();
        for (phase_label, bindings) in &bound_slots {
            for node in &graph.nodes {
                if node.block.is_none() {
                    collect_node_attrs(node, &mut top_attrs);
                    if let Some(op) = lower_node(graph, node, phase_label, &fp8_weights, bindings) {
                        top_ops.push(op);
                    }
                }
            }
        }
        if !top_ops.is_empty() {
            let mut nb = NetworkBlock::new(graph.blocks.len() as u32, "graph_head_tail", top_ops);
            nb.attributes = top_attrs;
            blocks.push(nb);
        }

        NetworkBlockDefinition::new(model_family, blocks).with_model(model_name)
    }

    fn expand_slots(
        batches: &[i64],
        seqs: &[i64],
        phases: &[Phase],
    ) -> Vec<(&'static str, i64, i64)> {
        let bs: Vec<i64> = if batches.is_empty() {
            vec![1]
        } else {
            batches.to_vec()
        };
        let ss: Vec<i64> = if seqs.is_empty() {
            vec![512]
        } else {
            seqs.to_vec()
        };
        let ps: Vec<Phase> = if phases.is_empty() {
            vec![Phase::Prefill, Phase::Decode]
        } else {
            phases.to_vec()
        };

        let mut seen = BTreeSet::new();
        let mut out = Vec::new();
        for phase in ps {
            match phase {
                Phase::Prefill => {
                    for &b in &bs {
                        for &s in &ss {
                            if seen.insert(("prefill", b, s)) {
                                out.push(("prefill", b, s));
                            }
                        }
                    }
                }
                Phase::Decode => {
                    for &b in &bs {
                        if seen.insert(("decode", b, 1)) {
                            out.push(("decode", b, 1));
                        }
                    }
                }
            }
        }
        out
    }

    fn lower_node(
        g: &Graph,
        node: &nn_graph::Node,
        phase: &str,
        fp8_weights: &HashSet<&str>,
        bindings: &HashMap<SymId, i64>,
    ) -> Option<NetworkOp> {
        let in0 = node.inputs.first().map(|&id| g.tensor(id));
        match &node.op {
            Op::Linear { out_features, .. } => {
                let x = in0?;
                let w = node.inputs.get(1).map(|&id| g.tensor(id))?;
                let (m, k) = split_leading_and_last(x, bindings)?;
                let quant = weight_quant_scheme(w, fp8_weights);
                Some(NetworkOp::gemm(phase, m, *out_features, k, quant))
            }
            Op::MatMul => {
                let a = in0?;
                let b = node.inputs.get(1).map(|&id| g.tensor(id))?;
                let (bm, k) = split_leading_and_last(a, bindings)?;
                let n = static_last(b, bindings)?;
                Some(NetworkOp::new(
                    "matmul",
                    phase,
                    bm,
                    n,
                    k,
                    dtype_label(a.dtype),
                ))
            }
            Op::Attention {
                num_heads,
                head_dim,
                ..
            } => {
                let q = in0?;
                let m = split_leading_and_last(q, bindings)
                    .map(|(t, _)| t)
                    .unwrap_or(0);
                Some(NetworkOp::new(
                    "attention",
                    phase,
                    m,
                    (*num_heads as i64) * (*head_dim as i64),
                    *head_dim as i64,
                    dtype_label(q.dtype),
                ))
            }
            Op::MoeRouter { num_experts, .. } => {
                let x = in0?;
                let (m, k) = split_leading_and_last(x, bindings)?;
                Some(NetworkOp::new(
                    "moe_router",
                    phase,
                    m,
                    *num_experts as i64,
                    k,
                    dtype_label(x.dtype),
                ))
            }
            Op::MoeExperts {
                intermediate_size,
                block_fp8,
                ..
            } => {
                let x = in0?;
                let (m, k) = split_leading_and_last(x, bindings)?;
                let quant = if *block_fp8 {
                    "BlockFp8"
                } else {
                    dtype_label(x.dtype)
                };
                Some(NetworkOp::new(
                    "moe_experts",
                    phase,
                    m,
                    *intermediate_size as i64,
                    k,
                    quant,
                ))
            }
            Op::LinearAttention {
                num_heads,
                head_dim,
                ..
            }
            | Op::DsaIndexer {
                num_heads,
                head_dim,
                ..
            }
            | Op::DsaAttention {
                num_heads,
                head_dim,
                ..
            } => {
                let x = in0?;
                let (m, _) = split_leading_and_last(x, bindings).unwrap_or((0, 0));
                Some(NetworkOp::new(
                    node.op.name(),
                    phase,
                    m,
                    (*num_heads as i64) * (*head_dim as i64),
                    *head_dim as i64,
                    dtype_label(x.dtype),
                ))
            }
            Op::RmsNorm { .. }
            | Op::RmsNormZeroCentered { .. }
            | Op::RmsNormInputScaled { .. }
            | Op::LayerNorm { .. }
            | Op::Rope { .. }
            | Op::Act(_)
            | Op::Elementwise(_)
            | Op::Softmax { .. }
            | Op::SituGlu { .. }
            | Op::BlockResidual { .. }
            | Op::ResidualGate { .. }
            | Op::Conv1dDepthwise { .. }
            | Op::Conv1d { .. }
            | Op::ConvTranspose1d { .. }
            | Op::Conv2d { .. }
            | Op::Conv3d { .. }
            | Op::GroupNorm { .. } => {
                let x = in0?;
                let out_t = g.tensor(node.output);
                let (m, k) = split_leading_and_last(x, bindings).unwrap_or((0, 0));
                let n = static_last(out_t, bindings).unwrap_or(k);
                Some(NetworkOp::new(
                    node.op.name(),
                    phase,
                    m,
                    n,
                    k,
                    dtype_label(x.dtype),
                ))
            }
            Op::Reshape { .. }
            | Op::Transpose { .. }
            | Op::Broadcast { .. }
            | Op::Concat { .. }
            | Op::Slice { .. }
            | Op::Reduce { .. }
            | Op::Scale(_)
            | Op::Embedding => None,
        }
    }

    fn collect_node_attrs(node: &nn_graph::Node, attrs: &mut BTreeMap<String, String>) {
        match &node.op {
            Op::Attention {
                num_heads,
                num_kv_heads,
                head_dim,
                causal,
                sliding_window,
                logit_softcap,
            } => {
                attrs.insert(
                    "attention".into(),
                    if let Some(w) = sliding_window {
                        format!("sliding:{w}")
                    } else {
                        "full".into()
                    },
                );
                attrs.insert(
                    "attn_geom".into(),
                    format!("h={num_heads},kv={num_kv_heads},hd={head_dim},causal={causal}"),
                );
                if let Some(cap) = logit_softcap {
                    attrs.insert("logit_softcap".into(), format!("{cap}"));
                }
            }
            Op::MoeRouter {
                num_experts,
                top_k,
                group,
                scoring,
                norm_topk,
                route_scale,
                correction_bias,
                logit_scale,
            } => {
                attrs.insert(
                    "moe_router".into(),
                    format!(
                        "e={num_experts},k={top_k},group={group:?},scoring={scoring:?},\
                         norm={norm_topk},scale={route_scale},bias={correction_bias},lscale={logit_scale}"
                    ),
                );
            }
            Op::MoeExperts {
                num_experts,
                top_k,
                intermediate_size,
                block_fp8,
                post_expert_norm,
            } => {
                attrs.insert(
                    "moe".into(),
                    format!(
                        "e={num_experts},k={top_k},inter={intermediate_size},fp8={block_fp8},post_norm={}",
                        post_expert_norm.is_some()
                    ),
                );
            }
            Op::LinearAttention {
                kind,
                num_heads,
                head_dim,
            } => {
                attrs.insert(
                    "linear_attention".into(),
                    format!("{kind:?}:h={num_heads},hd={head_dim}"),
                );
            }
            Op::DsaAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                top_k,
            } => {
                attrs.insert(
                    "dsa_attention".into(),
                    format!("h={num_heads},kv={num_kv_heads},hd={head_dim},topk={top_k}"),
                );
            }
            Op::Rope {
                dim,
                theta,
                interleave,
                frequency_dim,
            } => {
                attrs.insert(
                    "rope".into(),
                    format!("d={dim},theta={theta},interleave={interleave},freq={frequency_dim}"),
                );
            }
            Op::Conv1dDepthwise { kernel } => {
                attrs.insert("conv1d_dw".into(), format!("k={kernel}"));
            }
            Op::Conv1d {
                stride,
                dilation,
                groups,
                padding,
                pad_mode,
            } => {
                attrs.insert(
                    "conv1d".into(),
                    format!(
                        "s={stride},d={dilation},g={groups},pad={},{},{pad_mode:?}",
                        padding.0, padding.1
                    ),
                );
            }
            Op::ConvTranspose1d {
                stride,
                groups,
                crop,
                output_padding,
            } => {
                attrs.insert(
                    "conv_t1d".into(),
                    format!(
                        "s={stride},g={groups},crop={},{},opad={output_padding}",
                        crop.0, crop.1
                    ),
                );
            }
            Op::Conv2d { stride, padding } => {
                attrs.insert(
                    "conv2d".into(),
                    format!("s={}x{},p={}x{}", stride.0, stride.1, padding.0, padding.1),
                );
            }
            Op::Conv3d { stride, padding } => {
                attrs.insert(
                    "conv3d".into(),
                    format!(
                        "s={}x{}x{},p={}x{}x{}",
                        stride.0, stride.1, stride.2, padding.0, padding.1, padding.2
                    ),
                );
            }
            Op::GroupNorm { groups, .. } => {
                attrs.insert("groupnorm".into(), format!("g={groups}"));
            }
            Op::Act(ActKind::Snake) => {
                attrs.insert("act_snake".into(), "true".into());
            }
            Op::BlockResidual { max_snapshots } => {
                attrs.insert("block_residual".into(), format!("{max_snapshots}"));
            }
            Op::ResidualGate { .. } => {
                attrs.insert("residual_gate".into(), "true".into());
            }
            _ => {}
        }
    }

    fn split_leading_and_last(
        t: &TensorInfo,
        bindings: &HashMap<SymId, i64>,
    ) -> Option<(i64, i64)> {
        let dims = t.shape.as_ref()?.dims();
        if dims.is_empty() {
            return None;
        }
        let last = dims.last()?.eval(bindings)?;
        let mut lead: i64 = 1;
        for d in &dims[..dims.len() - 1] {
            lead = lead.saturating_mul(d.eval(bindings)?);
        }
        Some((lead, last))
    }

    fn static_last(t: &TensorInfo, bindings: &HashMap<SymId, i64>) -> Option<i64> {
        t.shape.as_ref()?.last()?.eval(bindings)
    }

    fn weight_quant_scheme(w: &TensorInfo, fp8_weights: &HashSet<&str>) -> QuantScheme {
        if let Some(name) = w.name.as_deref() {
            if fp8_weights.contains(name) {
                return QuantScheme::BlockFp8;
            }
        }
        match w.dtype {
            DType::F8E4M3 | DType::F8E5M2 => QuantScheme::W8A8,
            DType::F4 => QuantScheme::Mxfp4,
            _ => QuantScheme::None,
        }
    }

    fn dtype_label(dtype: DType) -> &'static str {
        match dtype {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F8E4M3 => "f8e4m3",
            DType::F8E5M2 => "f8e5m2",
            DType::F4 => "mxfp4",
            DType::I32 => "i32",
            DType::I64 => "i64",
            DType::I8 => "i8",
            DType::U8 => "u8",
            DType::Bool => "bool",
            DType::Q4_0 => "q4_0",
            DType::Q4_1 => "q4_1",
            DType::Q5_0 => "q5_0",
            DType::Q5_1 => "q5_1",
            DType::Q8_0 => "q8_0",
            DType::Q8_1 => "q8_1",
            DType::Q2_K => "q2_k",
            DType::Q3_K => "q3_k",
            DType::Q4_K => "q4_k",
            DType::Q5_K => "q5_k",
            DType::Q6_K => "q6_k",
            DType::IQ4_NL => "iq4_nl",
        }
    }

    fn canonical_block_kind(label: &str) -> String {
        let dot_norm: Vec<&str> = label
            .split('.')
            .map(|seg| {
                if !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit()) {
                    "*"
                } else {
                    seg
                }
            })
            .collect();
        let joined = dot_norm.join(".");
        if let Some((prefix, tail)) = joined.rsplit_once('_') {
            if !tail.is_empty() && tail.bytes().all(|b| b.is_ascii_digit()) {
                return format!("{prefix}_*");
            }
        }
        joined
    }

    fn model_family_from_json(json: &str) -> String {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
            return "unknown".to_string();
        };
        v.get("model_type")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                v.get("text_config")
                    .and_then(|t| t.get("model_type"))
                    .and_then(serde_json::Value::as_str)
            })
            .or_else(|| {
                v.get("architectures")
                    .and_then(serde_json::Value::as_array)
                    .and_then(|a| a.first())
                    .and_then(serde_json::Value::as_str)
            })
            .unwrap_or("unknown")
            .to_string()
    }
}

#[cfg(feature = "models")]
pub use nn_bridge::{from_config_json, from_graph};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::Stats;
    use crate::store::passed;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    const HW: &str = "nvidia/sm_90a/h100-nvl";

    fn tmpdir(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "tunedb-manifest-test-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    fn digests() -> Digests {
        Digests {
            execution: None,
            implementation: "impl-v1".into(),
            interpreter: "interp-v1".into(),
            toolchain: "nvcc-12.8".into(),
            oracle: "gemm-v1".into(),
        }
    }

    fn kernel_meas(
        op_case: &str,
        id: u16,
        name: &str,
        median_ns: f64,
        digests: Digests,
    ) -> KernelMeasurement {
        KernelMeasurement {
            op_case: op_case.into(),
            kernel_id: id,
            kernel_name: name.into(),
            profile: "prefill_dense".into(),
            hardware: HW.into(),
            sku: "H100".into(),
            digests,
            stats: Stats::from_samples(vec![median_ns; 5]).unwrap(),
            correctness: Correctness::Pass,
            registers: Some(128),
            state: RecordState::Provisional,
            campaign: "unit-test".into(),
        }
    }

    #[test]
    fn identical_layers_fold_into_single_weighted_block() {
        let qkv = NetworkOp::gemm("prefill", 512, 6144, 4096, QuantScheme::None);
        let o_proj = NetworkOp::gemm("prefill", 512, 4096, 4096, QuantScheme::None);
        let gate_up = NetworkOp::gemm("prefill", 512, 14336, 4096, QuantScheme::None);
        let down = NetworkOp::gemm("prefill", 512, 4096, 14336, QuantScheme::None);

        let blocks = (0..32)
            .map(|layer| {
                NetworkBlock::new(
                    layer,
                    "llama_decoder",
                    // gate and up share the exact same GEMM shape inside each block
                    vec![
                        qkv.clone(),
                        o_proj.clone(),
                        gate_up.clone(),
                        gate_up.clone(),
                        down.clone(),
                    ],
                )
            })
            .collect();

        let net = NetworkBlockDefinition::new("llama", blocks).with_model("llama-3-8b");
        let manifest = net.compile();

        assert_eq!(manifest.version, MANIFEST_VERSION);
        assert_eq!(manifest.model, "llama-3-8b");
        assert_eq!(manifest.total_block_instances, 32);
        assert_eq!(manifest.unique_blocks, 1);
        assert_eq!(manifest.unique_op_cases, 4);
        assert_eq!(manifest.total_op_instances, 32 * 5);

        let blk = &manifest.blocks[0];
        assert_eq!(blk.definition.occurrences, 32);
        assert_eq!(blk.definition.layer_ids, (0..32).collect::<Vec<_>>());
        assert_eq!(blk.definition.op_cases.len(), 4);

        // gate_up ran twice per block -> 64 total occurrences and sorts first.
        assert_eq!(manifest.op_cases[0].op_case.hash, gate_up.op_case_hash());
        assert_eq!(manifest.op_cases[0].total_occurrences, 64);
        assert_eq!(manifest.op_cases[1].total_occurrences, 32);
    }

    #[test]
    fn attributes_and_op_order_distinguish_blocks_while_layer_order_is_invariant() {
        let shared_qkv = NetworkOp::gemm("prefill", 512, 6144, 4096, QuantScheme::None);
        let shared_down = NetworkOp::gemm("prefill", 512, 4096, 14336, QuantScheme::None);

        // Same GEMM dims, but sliding vs full attention attributes must stay distinct.
        let mut blocks = Vec::new();
        for layer in 0..10 {
            let mut b = NetworkBlock::new(
                layer,
                "gemma4_decoder",
                vec![shared_qkv.clone(), shared_down.clone()],
            );
            if layer % 5 == 4 {
                b = b.with_attr("attention", "full");
            } else {
                b = b
                    .with_attr("attention", "sliding")
                    .with_attr("window", "1024");
            }
            blocks.push(b);
        }

        let net_forward = NetworkBlockDefinition::new("gemma4", blocks.clone());
        blocks.reverse();
        let net_reversed = NetworkBlockDefinition::new("gemma4", blocks);

        let m1 = net_forward.compile();
        let m2 = net_reversed.compile();

        assert_eq!(m1.network_hash, m2.network_hash);
        assert_eq!(m1.blocks, m2.blocks);
        assert_eq!(m1.op_cases, m2.op_cases);

        assert_eq!(m1.unique_blocks, 2);
        assert_eq!(m1.blocks[0].definition.occurrences, 8);
        assert_eq!(
            m1.blocks[0].definition.layer_ids,
            vec![0, 1, 2, 3, 5, 6, 7, 8]
        );
        assert_eq!(m1.blocks[1].definition.occurrences, 2);
        assert_eq!(m1.blocks[1].definition.layer_ids, vec![4, 9]);

        // Reordering ops inside a block changes its block hash.
        let reordered = NetworkBlock::new(0, "gemma4_decoder", vec![shared_down, shared_qkv])
            .with_attr("attention", "full");
        assert_ne!(reordered.hash("gemma4"), m1.blocks[1].definition.hash);
    }

    #[test]
    fn length_prefixed_hashing_avoids_concatenation_collisions() {
        assert_ne!(
            op_case_hash("ab", "c", 1, 2, 3, "None"),
            op_case_hash("a", "bc", 1, 2, 3, "None")
        );
    }

    #[test]
    fn store_resolution_ranks_kernels_reports_staleness_and_roundtrips_selections() {
        let root = tmpdir("resolve");
        let store = TuneStore::new(&root);

        let qkv = NetworkOp::gemm("prefill", 512, 6144, 4096, QuantScheme::None);
        let o_proj = NetworkOp::gemm("prefill", 512, 4096, 4096, QuantScheme::None);
        let missing_op = NetworkOp::gemv("gemv_bf16", "decode", 1, 4096, 4096, "bf16");

        let net = NetworkBlockDefinition::new(
            "glm5",
            vec![
                NetworkBlock::new(
                    0,
                    "dense",
                    vec![qkv.clone(), o_proj.clone(), missing_op.clone()],
                ),
                NetworkBlock::new(
                    1,
                    "dense",
                    vec![qkv.clone(), o_proj.clone(), missing_op.clone()],
                ),
            ],
        );
        let base_manifest = net.compile();
        let qkv_store_key = store_op_case_key(&qkv.to_op_case());
        assert_eq!(qkv_store_key, "gemm/512x6144x4096/None");

        let current = digests();
        let mut stale_d = current.clone();
        stale_d.implementation = "impl-v0".into();

        // QKV has two qualified kernels under the legacy `gemm/MxNxK/Quant` store key.
        store
            .publish(
                HW,
                vec![
                    passed(kernel_meas(
                        &qkv_store_key,
                        10,
                        "SLOW",
                        4000.0,
                        current.clone(),
                    )),
                    passed(kernel_meas(
                        &qkv_store_key,
                        11,
                        "FAST",
                        2000.0,
                        current.clone(),
                    )),
                    passed(kernel_meas(
                        &qkv_store_key,
                        12,
                        "ALT",
                        3000.0,
                        current.clone(),
                    )),
                ],
            )
            .unwrap();

        // o_proj only has a stale qualified record keyed by op_case hash.
        store
            .publish(
                HW,
                vec![passed(kernel_meas(
                    &o_proj.op_case_hash(),
                    20,
                    "STALE_K",
                    1500.0,
                    stale_d,
                ))],
            )
            .unwrap();

        // Publish a qualified whole-block measurement.
        let blk_hash = base_manifest.blocks[0].definition.hash.clone();
        store
            .publish_block_measurements(
                HW,
                vec![BlockMeasurement {
                    block: blk_hash.clone(),
                    hardware: HW.into(),
                    profile: "prefill_dense".into(),
                    digests: current.clone(),
                    stats: Stats::from_samples(vec![9500.0; 5]).unwrap(),
                    kernels: vec![11, 20],
                    peak_bytes: Some(1 << 20),
                    state: RecordState::Qualified,
                    campaign: "block-sweep".into(),
                }],
            )
            .unwrap();

        let resolved = base_manifest.resolve(&store, HW, &current).unwrap();
        let cov = resolved.coverage.as_ref().unwrap();
        assert_eq!(cov.hardware, HW);
        assert_eq!(cov.unique_ops_qualified, 1);
        assert_eq!(cov.unique_ops_stale, 1);
        assert_eq!(cov.unique_ops_missing, 1);
        assert_eq!(cov.weighted_ops_qualified, 2);
        assert_eq!(cov.weighted_ops_stale, 2);
        assert_eq!(cov.weighted_ops_missing, 2);
        assert_eq!(cov.blocks_with_qualified_measurement, 1);
        assert_eq!(cov.blocks_fully_op_covered, 0);

        let qkv_res = resolved
            .op_cases
            .iter()
            .find(|o| o.op_case.hash == qkv.op_case_hash())
            .unwrap()
            .resolution
            .as_ref()
            .unwrap();
        assert_eq!(qkv_res.status, CoverageState::Qualified);
        assert_eq!(qkv_res.best_kernel.as_ref().unwrap().kernel_id, 11);
        assert_eq!(qkv_res.fallback_kernels, vec![12, 10]);

        let o_res = resolved
            .op_cases
            .iter()
            .find(|o| o.op_case.hash == o_proj.op_case_hash())
            .unwrap()
            .resolution
            .as_ref()
            .unwrap();
        assert_eq!(o_res.status, CoverageState::Stale);
        assert_eq!(o_res.stale_reasons, vec!["implementation".to_string()]);

        // Synthesize selections, publish to store, and re-resolve to verify closure.
        let selections = resolved.derive_selections("min_median_ns");
        assert_eq!(selections.len(), 1);
        assert_eq!(selections[0].kernel_id, 11);
        assert_eq!(selections[0].fallbacks, vec![12, 10]);
        store.publish_selections(HW, selections).unwrap();

        let resolved_again = net.compile_for(&store, HW, &current).unwrap();
        let qkv_again = resolved_again
            .op_cases
            .iter()
            .find(|o| o.op_case.hash == qkv.op_case_hash())
            .unwrap()
            .resolution
            .as_ref()
            .unwrap();
        assert_eq!(qkv_again.selection.as_ref().unwrap().kernel_id, 11);
        assert_eq!(resolved_again.actionable_op_cases().len(), 2);
        assert_eq!(
            resolved_again.block_kernel_ids(&blk_hash).unwrap(),
            vec![Some(11), None, None]
        );

        // Rejected record stays unusable (Missing), while a staged Provisional
        // record surfaces as Provisional (never Qualified) and different hardware
        // cells remain isolated.
        let rejected = kernel_meas(&missing_op.op_case_hash(), 29, "REJ", 50.0, current.clone());
        store
            .record_rejected(HW, vec![rejected], "oracle-mismatch")
            .unwrap();
        let still_missing = net.compile_for(&store, HW, &current).unwrap();
        let miss_res = still_missing
            .op_cases
            .iter()
            .find(|o| o.op_case.hash == missing_op.op_case_hash())
            .unwrap()
            .resolution
            .as_ref()
            .unwrap();
        assert_eq!(miss_res.status, CoverageState::Missing);

        let provisional = kernel_meas(
            &missing_op.op_case_hash(),
            30,
            "PROV",
            100.0,
            current.clone(),
        );
        let kernel_path = root.join(HW).join("kernel_measurement.jsonl");
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&kernel_path)
                .unwrap();
            writeln!(f, "{}", serde_json::to_string(&provisional).unwrap()).unwrap();
        }
        let with_prov = net.compile_for(&store, HW, &current).unwrap();
        let prov_res = with_prov
            .op_cases
            .iter()
            .find(|o| o.op_case.hash == missing_op.op_case_hash())
            .unwrap()
            .resolution
            .as_ref()
            .unwrap();
        assert_eq!(prov_res.status, CoverageState::Provisional);

        let other_hw = net
            .compile_for(&store, "amd/gfx950/mi350x", &current)
            .unwrap();
        assert_eq!(other_hw.coverage.as_ref().unwrap().unique_ops_missing, 3);

        // JSON persistence round-trip.
        let manifest_path = root.join("manifest.json");
        resolved_again.save_json(&manifest_path).unwrap();
        let loaded = TuneManifest::load_json(&manifest_path).unwrap();
        assert_eq!(loaded, resolved_again);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[cfg(feature = "models")]
    #[test]
    fn nn_graph_config_lowers_folds_and_enforces_strict_mode() {
        let llama_cfg = r#"{
            "model_type": "llama",
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 2048,
            "rms_norm_eps": 1e-6,
            "torch_dtype": "bfloat16"
        }"#;
        let def = from_config_json(
            llama_cfg,
            "tiny-llama",
            &[1],
            &[512],
            &[plow_asset::Phase::Prefill, plow_asset::Phase::Decode],
            1024,
        )
        .unwrap();
        let manifest = def.compile();

        assert_eq!(manifest.model_family, "llama");
        assert_eq!(manifest.total_block_instances, 7);
        assert_eq!(manifest.unique_blocks, 2);
        assert_eq!(manifest.blocks[0].definition.occurrences, 6);
        assert_eq!(manifest.blocks[0].definition.block_kind, "model.layers.*");

        let gate_up = manifest
            .op_cases
            .iter()
            .find(|op| op.store_key == "gemm/512x1024x512/None")
            .expect("gate/up GEMM present");
        assert_eq!(gate_up.total_occurrences, 12);

        // Uncovered model resolved against an empty store fails --strict clearly.
        let empty_root = tmpdir("strict-empty");
        let resolved = manifest
            .resolve(&TuneStore::new(&empty_root), HW, &digests())
            .unwrap();
        let err = check_strict(&resolved, HW).unwrap_err();
        assert!(err.contains("--strict"), "unexpected error: {err}");
        let _ = std::fs::remove_dir_all(&empty_root);
    }

    #[cfg(feature = "models")]
    #[test]
    fn resolves_model_backed_gemms_against_checked_in_mi350x_store() {
        let tuning_dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../tuning"));
        let cell = "amd/gfx950/mi350x";
        let store = TuneStore::new(tuning_dir);
        if !tuning_dir
            .join(cell)
            .join("kernel_measurement.jsonl")
            .is_file()
        {
            return;
        }
        let want_latest = infer_digests_for_cell(&store, cell)
            .unwrap()
            .expect("mi350x store has digests");
        let kernels = store.load_kernels(cell).unwrap();
        let want_gemm = kernels
            .iter()
            .rev()
            .find(|k| k.op_case == "gemm/512x6144x512/None" && k.state.is_selectable())
            .expect("qualified gemm/512x6144x512/None exists in mi350x store")
            .digests
            .clone();

        // Build a Llama-style config whose MLP (`hidden=512, intermediate=6144`) emits
        // `gemm/512x6144x512/None` present in the checked-in mi350x store.
        let cfg = r#"{
            "model_type": "llama",
            "hidden_size": 512,
            "intermediate_size": 6144,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "vocab_size": 2048,
            "rms_norm_eps": 1e-6,
            "torch_dtype": "bfloat16"
        }"#;
        let def = from_config_json(
            cfg,
            "mi350x-test-llama",
            &[1],
            &[512],
            &[plow_asset::Phase::Prefill],
            1024,
        )
        .unwrap();
        let resolved_gemm = def.compile_for(&store, cell, &want_gemm).unwrap();
        let hit = resolved_gemm
            .op_cases
            .iter()
            .find(|op| op.store_key == "gemm/512x6144x512/None")
            .expect("MLP gate/up GEMM present");
        let res = hit.resolution.as_ref().unwrap();
        assert_eq!(res.status, CoverageState::Qualified);
        assert!(res.best_kernel.is_some());

        // Under the cell's latest digests (from a newer build), the older record is Stale.
        if want_latest != want_gemm {
            let resolved_latest = def.compile_for(&store, cell, &want_latest).unwrap();
            let stale_hit = resolved_latest
                .op_cases
                .iter()
                .find(|op| op.store_key == "gemm/512x6144x512/None")
                .expect("MLP gate/up GEMM present");
            assert_eq!(
                stale_hit.resolution.as_ref().unwrap().status,
                CoverageState::Stale
            );
        }
    }

    #[cfg(feature = "models")]
    #[test]
    fn heterogeneous_nn_graph_folds_by_layer_type_and_preserves_op_attrs() {
        // Hybrid linear_attention (x3) + full_attention (x1) like Qwen3.5.
        let qwen35_cfg = r#"{
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "vocab_size": 500,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "layer_types": [
                    "linear_attention", "linear_attention",
                    "linear_attention", "full_attention"
                ],
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 16,
                "linear_num_key_heads": 2,
                "linear_num_value_heads": 6,
                "linear_value_head_dim": 16,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "rope_theta": 10000000.0,
                    "partial_rotary_factor": 0.25,
                    "rope_type": "default",
                    "mrope_interleaved": true
                },
                "attention_bias": false,
                "attn_output_gate": true,
                "hidden_act": "silu",
                "mamba_ssm_dtype": "float32",
                "output_gate_type": "swish",
                "tie_word_embeddings": false,
                "dtype": "bfloat16"
            }
        }"#;

        let def = from_config_json(
            qwen35_cfg,
            "tiny-qwen35",
            &[1],
            &[128],
            &[plow_asset::Phase::Prefill],
            1024,
        )
        .unwrap();
        let manifest = def.compile();

        assert_eq!(manifest.model_family, "qwen3_5");
        assert_eq!(manifest.total_block_instances, 5);
        // 3 linear_attention layers fold -> 1 block (x3), 1 full_attention -> 1 block (x1), head/tail -> 1 (x1).
        assert_eq!(manifest.unique_blocks, 3);

        let linear_blk = manifest
            .blocks
            .iter()
            .find(|b| b.attributes.contains_key("linear_attention"))
            .expect("linear_attention canonical block present");
        assert_eq!(linear_blk.definition.occurrences, 3);
        assert_eq!(linear_blk.definition.layer_ids, vec![0, 1, 2]);
        assert!(linear_blk.attributes.contains_key("conv1d_dw"));

        let full_blk = manifest
            .blocks
            .iter()
            .find(|b| b.attributes.contains_key("attention"))
            .expect("full attention canonical block present");
        assert_eq!(full_blk.definition.occurrences, 1);
        assert_eq!(full_blk.definition.layer_ids, vec![3]);
        assert!(full_blk.attributes.contains_key("rope"));
    }
}
