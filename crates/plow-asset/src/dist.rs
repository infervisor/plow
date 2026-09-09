//! Distribution schemas: the model index, the bundle manifest, and the objset
//! manifest — plus the rule that picks a variant for a live machine.
//!
//! The rule lives here rather than in `plowrt` because both sides need it: the
//! packer refuses to publish a variant the rule could never select, and the
//! runtime uses it to choose one. A second implementation could disagree, which
//! is the hazard `exec/gpu.rs`'s pairing check already refuses on principle.
//!
//! `LiveTarget` is a plain description of a probed machine so this crate needs
//! no `hwspec` dependency; `plowrt` fills it from a `HardwareFingerprint`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const INDEX_SCHEMA: &str = "plow.dist.model.v1";
pub const BUNDLE_SCHEMA: &str = "plow.dist.bundle.v1";
pub const OBJSET_SCHEMA: &str = "plow.dist.objset.v1";

/// How far a variant has been taken. Ranked, so an unmeasured build can be
/// published without being presented as equivalent to a measured one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Status {
    /// `plowc` refuses this point; the recipe records why. Never selectable.
    Refused,
    /// Emits and loads, but carries no measurement.
    Emits,
    /// Emits, gated, and measured.
    Validated,
}

/// What changed relative to `supersedes`, and therefore what a client must
/// fetch to upgrade.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Delta {
    /// Same packet, new code objects. Cheap: only the objset blobs move.
    Objset,
    /// New packet, and therefore a new pairing hash and a new objset.
    Packet,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub vendor: String,
    pub isa: String,
    pub sku: String,
    pub units: u32,
    pub mem_bytes: u64,
}

/// How work is spread across GPUs. An object, not a string, so a composite
/// (`tp4pp2`) is expressible the day `plowc` stops refusing `--parallel pp`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Parallel {
    pub mode: String,
    pub n: u32,
}

impl Parallel {
    pub fn label(&self) -> String {
        format!("{}{}", self.mode, self.n)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Build {
    pub plow_git: String,
    pub plowc: String,
    pub objset_id: String,
    /// `build.json`'s `pairing.hash`, or `None` for a GENERAL objset (every arm
    /// compiled, no `plow_packet_hash_{lo,hi}` stamp) which pairs with any packet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing_hash: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measured {
    pub tok_s: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub p50_tpot_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub variant_id: String,
    pub label: String,
    pub generation: u32,
    pub status: Status,
    pub target: Target,
    pub parallel: Parallel,
    pub max_ctx: u32,
    /// `build.json`'s feature flags. A map rather than a struct so a new flag
    /// does not invalidate every published index.
    #[serde(default)]
    pub features: BTreeMap<String, bool>,
    pub build: Build,
    pub manifest: String,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub released: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<Delta>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measured: Option<Measured>,
    /// Why `plowc` refuses this point. Required when `status` is `Refused`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

/// `v1/<namespace>/<name>/index.json`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIndex {
    pub schema: String,
    pub namespace: String,
    pub name: String,
    /// `<org>/<repo>` on HuggingFace. The model's identity root.
    pub hf: String,
    pub revision: String,
    /// The API slug the runtime registers under (`weights.json`'s `network`).
    pub network: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub checkpoint: CheckpointRef,
    pub variants: Vec<Variant>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointRef {
    pub shards: u32,
    pub layout: String,
    pub bytes: u64,
}

/// Where a bundle file came from. The distribution ships only what plow
/// produced; anything the HF snapshot already carries is linked at prepare time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Provenance {
    /// Present in the HF snapshot. NOT distributed; `prepare` links it and
    /// refuses when the snapshot has none.
    Checkpoint,
    /// Generated by plow (e.g. `scripts/kimi_k3_tokenizer.py` reconstructing a
    /// `tokenizer.json` for a checkpoint that ships only `tiktoken.model`).
    /// Distributed.
    Derived,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileRef {
    pub role: String,
    pub name: String,
    pub sha256: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerRef {
    pub source: Provenance,
    pub file: String,
    /// Set only when `source` is `Derived` — a checkpoint-sourced file is
    /// whatever the snapshot holds and is not pinned by the bundle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator: Option<String>,
    #[serde(default)]
    pub verified: bool,
}

/// `v1/<ns>/<name>/manifests/<label>@g<n>` — one variant's manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub schema: String,
    pub namespace: String,
    pub name: String,
    pub label: String,
    pub generation: u32,
    pub variant_id: String,
    pub plow_git: String,
    pub network: String,
    pub target: Target,
    pub parallel: Parallel,
    pub max_ctx: u32,
    pub files: Vec<FileRef>,
    pub tokenizer: TokenizerRef,
    pub checkpoint: BundleCheckpoint,
    pub objset: BundleObjset,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing_hash: Option<String>,
    /// The `PLOW_*` settings this bundle was validated with. `serve` applies
    /// them as defaults and logs every operator override.
    #[serde(default)]
    pub runtime_env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleCheckpoint {
    /// `hf:<org>/<repo>`.
    pub source: String,
    pub revision: String,
    pub shards: u32,
    pub layout: String,
    pub bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BundleObjset {
    pub objset_id: String,
    pub manifest: String,
    pub sha256: String,
    /// Narrow-rung decode overrides, replacing `PLOW_HSACO_LOWRUNG`'s absolute
    /// paths, which cannot survive leaving the build host.
    #[serde(default)]
    pub lowrung: Vec<LowRung>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LowRung {
    pub max: u32,
    pub objset_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjSetObject {
    pub name: String,
    pub sha256: String,
    pub bytes: u64,
    /// Marker symbols read out of `.symtab` at pack time, so the arm check can
    /// run BEFORE any object is opened and name the missing symbol.
    #[serde(default)]
    pub arms: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packet_hash: Option<String>,
}

/// `v1/objsets/<vendor>/<isa>/<sku>/<objset-id>.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObjSet {
    pub schema: String,
    pub objset_id: String,
    pub target: Target,
    pub toolchain: String,
    pub plow_git: String,
    pub script: String,
    /// The env the build script was driven with — the recipe's `[objects].env`.
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Verbatim `build_defines.json`: stem → the exact `-D` string compiled.
    #[serde(default)]
    pub defines: BTreeMap<String, String>,
    pub objects: Vec<ObjSetObject>,
}

// --- selection ---------------------------------------------------------------

/// A probed machine. Filled by `plowrt` from a live `HardwareFingerprint`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveTarget {
    pub vendor: String,
    pub isa: String,
    pub sku: Option<String>,
    pub units: u32,
    pub mem_bytes: u64,
    pub gpus: u32,
    pub toolchain: Option<String>,
}

/// Caller-supplied narrowing. `modes` is what this build actually implements —
/// `plowc` parses `dp`/`pp`/`ep` and then refuses them, so shipping an index
/// that names one must not produce a selection.
#[derive(Clone, Debug, Default)]
pub struct Constraints {
    pub label: Option<String>,
    pub generation: Option<u32>,
    pub parallel_n: Option<u32>,
    pub max_ctx: Option<u32>,
    pub features: Vec<String>,
    /// `PLOW_OVERSUB`: permit a variant whose `units` is an exact multiple of
    /// the live count, mirroring `exec/amd.rs`'s own oversubscription rule.
    pub oversub: bool,
}

/// Why a variant cannot serve this machine. Every arm names both sides so a
/// refusal is actionable without a second command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Reject {
    Status,
    Vendor { want: String, live: String },
    Isa { want: String, live: String },
    Units { want: u32, live: u32 },
    ParallelMode { mode: String },
    ParallelN { want: u32, live: u32 },
    Mem { want: u64, live: u64 },
    MaxCtx { want: u32, has: u32 },
    Feature { name: String },
    Label { want: String },
    Generation { want: u32 },
}

impl std::fmt::Display for Reject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reject::Status => write!(f, "status is `refused` — plowc cannot emit this point"),
            Reject::Vendor { want, live } => write!(f, "built for vendor {want}, this is {live}"),
            Reject::Isa { want, live } => write!(f, "built for {want}, this is {live}"),
            Reject::Units { want, live } => write!(
                f,
                "built for {want} CUs/SMs, this has {live} (tile and grid choices are baked in; \
                 set PLOW_OVERSUB=1 only for an exact multiple)"
            ),
            Reject::ParallelMode { mode } => {
                write!(f, "parallel mode `{mode}` is not implemented by this build")
            }
            Reject::ParallelN { want, live } => {
                write!(f, "needs {want} GPUs, {live} visible")
            }
            Reject::Mem { want, live } => {
                write!(f, "needs {want} bytes of device memory, this has {live}")
            }
            Reject::MaxCtx { want, has } => write!(f, "max_ctx {has} < requested {want}"),
            Reject::Feature { name } => write!(f, "does not carry requested feature `{name}`"),
            Reject::Label { want } => write!(f, "label is not `{want}`"),
            Reject::Generation { want } => write!(f, "generation is not {want}"),
        }
    }
}

/// One variant judged against a machine. `warns` are recorded, never fatal.
#[derive(Clone, Debug)]
pub struct Assessment<'a> {
    pub variant: &'a Variant,
    pub reject: Option<Reject>,
    pub warns: Vec<String>,
}

impl Assessment<'_> {
    pub fn ok(&self) -> bool {
        self.reject.is_none()
    }
}

/// Modes `plowc` actually wires today. `Dp`/`Pp`/`Ep` are declared and then
/// refused, so a variant naming one can be published but never selected.
pub const IMPLEMENTED_MODES: &[&str] = &["tp"];

fn assess_one<'a>(v: &'a Variant, live: &LiveTarget, c: &Constraints) -> Assessment<'a> {
    let mut warns = Vec::new();
    let reject = (|| {
        if let Some(want) = &c.label {
            if &v.label != want {
                return Some(Reject::Label { want: want.clone() });
            }
        }
        if let Some(want) = c.generation {
            if v.generation != want {
                return Some(Reject::Generation { want });
            }
        }
        if v.status == Status::Refused {
            return Some(Reject::Status);
        }
        if v.target.vendor != live.vendor {
            return Some(Reject::Vendor {
                want: v.target.vendor.clone(),
                live: live.vendor.clone(),
            });
        }
        if v.target.isa != live.isa {
            return Some(Reject::Isa {
                want: v.target.isa.clone(),
                live: live.isa.clone(),
            });
        }
        // Mirrors the late check at `exec/amd.rs`: equal, or an exact multiple
        // under oversubscription. `stream_ofs`/`stream_len` are `[n_cu]` tables
        // indexed by workgroup, so anything else silently drops streams.
        let units_ok = v.target.units == live.units
            || (c.oversub
                && live.units != 0
                && v.target.units > live.units
                && v.target.units % live.units == 0);
        if !units_ok {
            return Some(Reject::Units {
                want: v.target.units,
                live: live.units,
            });
        }
        if !IMPLEMENTED_MODES.contains(&v.parallel.mode.as_str()) {
            return Some(Reject::ParallelMode {
                mode: v.parallel.mode.clone(),
            });
        }
        if v.parallel.n > live.gpus {
            return Some(Reject::ParallelN {
                want: v.parallel.n,
                live: live.gpus,
            });
        }
        if let Some(want) = c.parallel_n {
            if v.parallel.n != want {
                return Some(Reject::ParallelN {
                    want,
                    live: v.parallel.n,
                });
            }
        }
        if v.target.mem_bytes > live.mem_bytes {
            return Some(Reject::Mem {
                want: v.target.mem_bytes,
                live: live.mem_bytes,
            });
        }
        if let Some(want) = c.max_ctx {
            if v.max_ctx < want {
                return Some(Reject::MaxCtx {
                    want,
                    has: v.max_ctx,
                });
            }
        }
        for name in &c.features {
            if !v.features.get(name).copied().unwrap_or(false) {
                return Some(Reject::Feature { name: name.clone() });
            }
        }
        None
    })();

    if reject.is_none() {
        match (&live.sku, &v.target.sku) {
            (Some(l), w) if l != w => warns.push(format!(
                "built for SKU {w}, this is {l} — same ISA and unit count, so it loads, \
                 but the tuning was measured elsewhere"
            )),
            _ => {}
        }
        if v.status == Status::Emits {
            warns.push("emits and loads, but carries no measurement".into());
        }
    }
    Assessment {
        variant: v,
        reject,
        warns,
    }
}

/// Judge every variant. Order is preserved, so a caller can print the whole
/// table — which is what `plowrt show` does and what a failed `load` reports.
pub fn assess<'a>(
    variants: &'a [Variant],
    live: &LiveTarget,
    c: &Constraints,
) -> Vec<Assessment<'a>> {
    variants.iter().map(|v| assess_one(v, live, c)).collect()
}

/// Rank key, best first. Generation leads: a newer generation of one label is
/// the same model on the same hardware, only built better.
fn rank(v: &Variant, live: &LiveTarget) -> (u32, Status, bool, u32, i64) {
    let exact_sku = live.sku.as_deref() == Some(v.target.sku.as_str());
    let tok_s = v
        .measured
        .as_ref()
        .map(|m| (m.tok_s * 1000.0) as i64)
        .unwrap_or(0);
    (v.generation, v.status, exact_sku, v.parallel.n, tok_s)
}

/// Pick the best variant for this machine, or explain why none fits.
///
/// A tie between two survivors is an error, not a coin flip: two builds that
/// rank identically differ in something the ranking does not model, and picking
/// one silently would make the choice unreproducible.
pub fn select<'a>(
    variants: &'a [Variant],
    live: &LiveTarget,
    c: &Constraints,
) -> Result<&'a Variant, String> {
    let table = assess(variants, live, c);
    let mut ok: Vec<&Assessment> = table.iter().filter(|a| a.ok()).collect();
    if ok.is_empty() {
        return Err(explain(&table, live));
    }
    ok.sort_by_key(|a| std::cmp::Reverse(rank(a.variant, live)));
    if ok.len() > 1 && rank(ok[0].variant, live) == rank(ok[1].variant, live) {
        return Err(format!(
            "ambiguous: {} and {} rank identically (generation {}, status {:?}, {} GPUs). \
             Name one explicitly with `:<label>@g<n>`.",
            ok[0].variant.label,
            ok[1].variant.label,
            ok[0].variant.generation,
            ok[0].variant.status,
            ok[0].variant.parallel.n
        ));
    }
    Ok(ok[0].variant)
}

/// The refusal table: every candidate and the rule that rejected it.
pub fn explain(table: &[Assessment], live: &LiveTarget) -> String {
    use std::fmt::Write;
    let mut s = format!(
        "no published variant fits this machine ({} {}, {} units, {} GPUs, {} bytes",
        live.vendor, live.isa, live.units, live.gpus, live.mem_bytes
    );
    match &live.sku {
        Some(sku) => {
            let _ = write!(s, ", {sku})");
        }
        None => s.push(')'),
    }
    if table.is_empty() {
        s.push_str("\n  (this model publishes no variants)");
        return s;
    }
    for a in table {
        let _ = write!(
            s,
            "\n  {}@g{}  {}",
            a.variant.label,
            a.variant.generation,
            match &a.reject {
                Some(r) => r.to_string(),
                None => "ok".into(),
            }
        );
    }
    s
}

// --- validation --------------------------------------------------------------

fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// One path component, no separators, no `..`. Same hardening as
/// [`crate::decode_objects::DecodeObject::validate`].
fn is_plain_name(s: &str) -> bool {
    let mut c = std::path::Path::new(s).components();
    matches!(c.next(), Some(std::path::Component::Normal(_))) && c.next().is_none()
}

impl Variant {
    pub fn validate(&self) -> Result<(), String> {
        if self.label.is_empty() || self.variant_id.is_empty() || !is_sha256(&self.sha256) {
            return Err(format!("variant {}: bad identity", self.label));
        }
        if self.parallel.n == 0 || self.target.units == 0 || self.max_ctx == 0 {
            return Err(format!("variant {}: zero-valued key axis", self.label));
        }
        if (self.status == Status::Refused) != self.refusal.is_some() {
            return Err(format!(
                "variant {}: `refusal` must be present exactly when status is `refused`",
                self.label
            ));
        }
        if self.status == Status::Validated && self.measured.is_none() {
            return Err(format!(
                "variant {}: status `validated` requires a `measured` record",
                self.label
            ));
        }
        if self.supersedes.is_some() != self.delta.is_some() {
            return Err(format!(
                "variant {}: `supersedes` and `delta` travel together",
                self.label
            ));
        }
        Ok(())
    }
}

impl ModelIndex {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != INDEX_SCHEMA {
            return Err(format!("index: unsupported schema {}", self.schema));
        }
        if self.namespace.is_empty() || self.name.is_empty() || self.network.is_empty() {
            return Err("index: empty namespace, name or network".into());
        }
        if self.hf.split('/').count() != 2 {
            return Err(format!("index: `hf` must be <org>/<repo>, got {}", self.hf));
        }
        let mut seen = std::collections::BTreeSet::new();
        for v in &self.variants {
            v.validate()?;
            if !seen.insert((v.label.as_str(), v.generation)) {
                return Err(format!(
                    "index: duplicate variant {}@g{}",
                    v.label, v.generation
                ));
            }
        }
        Ok(())
    }
}

impl ObjSet {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != OBJSET_SCHEMA {
            return Err(format!("objset: unsupported schema {}", self.schema));
        }
        if self.objset_id.is_empty() || self.objects.is_empty() {
            return Err("objset: empty identity or object list".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for o in &self.objects {
            if !is_plain_name(&o.name) || !is_sha256(&o.sha256) || o.bytes == 0 {
                return Err(format!("objset: bad object entry {}", o.name));
            }
            if !seen.insert(o.name.as_str()) {
                return Err(format!("objset: duplicate object {}", o.name));
            }
        }
        Ok(())
    }

    /// A GENERAL objset stamps no packet hash and pairs with any packet; a
    /// specialised one carries `plow_packet_hash_{lo,hi}` and pairs only with
    /// the packet that produced it. Publishing a mismatch would turn into a
    /// confusing refusal deep in the loader, so the packer checks it here.
    pub fn pairs_with(&self, packet_pairing_hash: Option<&str>) -> Result<(), String> {
        for o in &self.objects {
            match (&o.packet_hash, packet_pairing_hash) {
                (None, _) => {}
                (Some(stamp), Some(want)) if stamp == want => {}
                (Some(stamp), want) => {
                    return Err(format!(
                        "objset {}: object {} is stamped for packet {stamp} but the bundle's \
                         packet is {}",
                        self.objset_id,
                        o.name,
                        want.unwrap_or("<unstamped>")
                    ))
                }
            }
        }
        Ok(())
    }
}

impl Bundle {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != BUNDLE_SCHEMA {
            return Err(format!("bundle: unsupported schema {}", self.schema));
        }
        if self.files.is_empty() {
            return Err("bundle: no files".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for f in &self.files {
            if !is_plain_name(&f.name) || !is_sha256(&f.sha256) {
                return Err(format!("bundle: bad file entry {}", f.name));
            }
            if !seen.insert(f.name.as_str()) {
                return Err(format!("bundle: duplicate file {}", f.name));
            }
        }
        if !self.files.iter().any(|f| f.role == "packet") {
            return Err("bundle: no file with role `packet`".into());
        }
        // A derived tokenizer is a shipped artifact and must be pinned; a
        // checkpoint-sourced one is whatever the snapshot holds.
        match self.tokenizer.source {
            Provenance::Derived => {
                if !self.tokenizer.sha256.as_deref().is_some_and(is_sha256)
                    || self.tokenizer.generator.is_none()
                {
                    return Err(
                        "bundle: a derived tokenizer needs both `sha256` and `generator`".into(),
                    );
                }
                if !self.files.iter().any(|f| f.name == self.tokenizer.file) {
                    return Err(format!(
                        "bundle: derived tokenizer {} is not in `files`",
                        self.tokenizer.file
                    ));
                }
            }
            Provenance::Checkpoint => {
                if self.tokenizer.sha256.is_some() || self.tokenizer.generator.is_some() {
                    return Err("bundle: a checkpoint tokenizer is not pinned by the bundle".into());
                }
                if self.files.iter().any(|f| f.name == self.tokenizer.file) {
                    return Err(format!(
                        "bundle: {} is sourced from the checkpoint and must not be shipped",
                        self.tokenizer.file
                    ));
                }
            }
        }
        if !self.checkpoint.source.starts_with("hf:") {
            return Err(format!(
                "bundle: checkpoint source must be `hf:<org>/<repo>`, got {}",
                self.checkpoint.source
            ));
        }
        let mut rungs = std::collections::BTreeSet::new();
        for l in &self.objset.lowrung {
            if l.max == 0 || !rungs.insert(l.max) {
                return Err(format!("bundle: bad or duplicate lowrung max {}", l.max));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "dist_tests.rs"]
mod tests;
