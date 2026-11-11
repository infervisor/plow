//! Generate `bundle.json` and `objset.json` from a recipe and a built
//! directory. The only producer there is: the two Python packers this replaced
//! were deleted with it, because a second path that still works can still
//! drift, and twice it did — a published bundle dropped a `[serve] env` key
//! and recorded a commit its recipe did not pin.
//!
//! The archive contract is unchanged; what moves here is the step that decides
//! IDENTITY: the digests, the object discovery order, the role of each file,
//! and the objset id. That step hand-matched the `deny_unknown_fields` schemas
//! in `plow_asset::dist` from another language, so a field added in Rust left
//! Python emitting a shape the registry refuses — or silently accepts as wrong.
//! Built here, the manifests ARE those structs, and the compiler checks them.
//!
//! Identity comes from the recipe because that is the only place it is written
//! down: a build's `build.json` records what was compiled (features, pairing
//! hash, `runtime_requires`) and nothing about which registry variant it is.
//! The retired Python release script already validated a manifest against
//! its recipe, so the recipe was already the authority — it just was not the
//! source.
//!
//! Byte-for-byte compatible with the retired Python packers, deliberately:
//! `v1/objsets/<id>` is keyed by a digest they computed, and a published generation is
//! immutable. A different canonicalisation here would write a NEW id for
//! identical content and orphan the published manifest. `canonical_json` and
//! `objset_id` below are therefore ports, not re-derivations.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use plow_asset::dist::{
    Bundle, BundleCheckpoint, BundleObjset, CheckpointFile, FileRef, LowRung, ObjSet, ObjSetObject,
    Parallel, Provenance, Target, TokenizerRef, OBJSET_SCHEMA,
};

use crate::{Result, RuntimeError};

fn bad(msg: impl Into<String>) -> RuntimeError {
    RuntimeError::Dist(msg.into())
}

/// Reserved names in an asset directory, by role. A hand-copy of constants the
/// emit already knows (`devgen::tts::CODEC_PACKET`,
/// `plow_asset::certificates::PACKET_CHECKS_FILE`, ...) and the reason the
/// build should declare its own inventory instead; until it does, this table is
/// what the Python packer had, ported unchanged so that nothing moved in the
/// same step that moved the producer.
const ROLES: &[(&str, &str)] = &[
    ("model.pkt", "packet"),
    ("build.json", "build_manifest"),
    ("weights.json", "asset_manifest"),
    ("plow_config.h", "config_header"),
    // A speech bundle carries two more: the codec lowered to its own packet,
    // and the cuBLASLt algorithm table the recipe's `PLOW_LT_RUNG_ALGOS`
    // timed per rung. Dropping either publishes an IMMUTABLE generation that
    // still loads — no codec means no audio, no table means the tuning is
    // gone — so they are roles, not sidecars.
    ("codec.pkt", "codec_packet"),
    ("cublaslt_algos.jsonl", "algo_table"),
    // Lean compile receipts, read from beside the packet they bind by
    // `plowc qualify`. A bundle that drops them can never be qualified for as
    // long as it exists.
    ("lean-checks.json", "packet_checks"),
    // The host-side GDN prefill library a Qwen3.5-family packet's prefill
    // loads from beside it (`exec/gpu.rs`, `NativeGdn::load`): built per
    // target by `scripts/build_sm90a_qwen_gdn.py`, so it is part of the
    // build, not an object, and a bundle without it cannot prefill.
    ("libplow_gdn_prefill.so", "host_library"),
];

/// Suffixes that belong to the objset, not the bundle: exactly the three globs
/// [`discover_objects`] collects. `.so` is deliberately absent — `campaign.py`
/// counts it as an object but no glob takes it, so one sitting in an asset
/// directory is claimed by nothing and must be reported.
const OBJSET_SUFFIXES: &[&str] = &[".cubin", ".elf", ".co", LIBRARY_OBJECT_SUFFIX];

/// A library object container (`device/cuda/aot.rs`: FA4 attention, GDN prefill) that NVIDIA
/// serving loads from the asset directory by the name its packet declares: an objset object.
const LIBRARY_OBJECT_SUFFIX: &str = ".aot";

/// The GDN prefill library (`device::cuda::qwen_gdn::OBJECT`), loaded from the asset directory
/// whenever a prefill program runs `QwenGdnPrefill`; the packet names it by that opcode, not in
/// `segment_roles`.
const GDN_PREFILL_OBJECT: &str = "plow_gdn_prefill.aot";

/// Shares the directory and belongs to something else: the tokenizer, which is
/// classified by provenance rather than by name.
const SKIP_QUIETLY: &[&str] = &["tokenizer.json"];

/// Vendors whose kernels compile on the serving machine from the packet, so a
/// bundle for them carries no objset. Mirrors `plow_asset::dist::runtime_compiled`.
fn runtime_compiled(vendor: &str) -> bool {
    plow_asset::dist::runtime_compiled(vendor)
}

// --- canonical bytes ---------------------------------------------------------

/// `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`,
/// byte for byte.
///
/// Not a stylistic choice. `bundle.objset.sha256` and every rung's `sha256`
/// name the bytes of an objset manifest canonicalised exactly this way by
/// the retired Python packers, and `objset_id` hashes two digests built from
/// it. `serde_json` agrees on the separators but preserves field order and
/// emits non-ASCII raw, so the walk below sorts keys and escapes like
/// `ensure_ascii=True`.
pub fn canonical_json(value: &impl serde::Serialize) -> Result<Vec<u8>> {
    let v = serde_json::to_value(value).map_err(|e| bad(format!("serialize: {e}")))?;
    let mut out = Vec::new();
    write_canonical(&v, &mut out);
    Ok(out)
}

fn write_canonical(v: &serde_json::Value, out: &mut Vec<u8>) {
    use serde_json::Value;
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => out.extend_from_slice(n.to_string().as_bytes()),
        Value::String(s) => write_json_string(s, out),
        Value::Array(a) => {
            out.push(b'[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(e, out);
            }
            out.push(b']');
        }
        Value::Object(m) => {
            // `sort_keys=True`. UTF-8 byte order is code-point order, so this
            // is Python's `str` comparison.
            let mut keys: Vec<&str> = m.keys().map(|k| k.as_str()).collect();
            keys.sort_unstable();
            out.push(b'{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_json_string(k, out);
                out.push(b':');
                write_canonical(&m[*k], out);
            }
            out.push(b'}');
        }
    }
}

fn write_json_string(s: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for c in s.chars() {
        match c {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\u{8}' => out.extend_from_slice(b"\\b"),
            '\u{c}' => out.extend_from_slice(b"\\f"),
            // Python's `ESCAPE_ASCII` passes through `[ -~]` and escapes
            // everything else — including 0x7f (DEL), which IS ascii and would
            // otherwise slip out raw and change the digest.
            ' '..='~' => out.push(c as u8),
            // `ensure_ascii=True`: one `\uXXXX` per UTF-16 unit, so a scalar
            // above the BMP becomes a surrogate PAIR, as Python writes it, and
            // a control character below 0x20 becomes `\u00xx`.
            c => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.extend_from_slice(format!("\\u{u:04x}").as_bytes());
                }
            }
        }
    }
    out.push(b'"');
}

fn sha256_bytes(b: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(b))
}

fn sha256_file(p: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(p).map_err(|e| bad(format!("{}: {e}", p.display())))?;
    let (mut h, mut buf) = (Sha256::new(), vec![0u8; 1 << 20]);
    loop {
        let k = f
            .read(&mut buf)
            .map_err(|e| bad(format!("{}: {e}", p.display())))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
    }
    Ok(format!("{:x}", h.finalize()))
}

/// A 12-hex identity over the parts that define an artifact, as the retired
/// Python packers formed one.
fn short_id(parts: &[&str]) -> String {
    sha256_bytes(parts.join("\u{1f}").as_bytes())[..12].to_string()
}

/// A published manifest must not reference the build host: it is consumed on
/// machines with no nix and no `/nix/store`, where a leaked path resolves to
/// nothing.
pub(crate) fn no_build_host_paths(bytes: &[u8], where_: &str) -> Result<()> {
    let blob = String::from_utf8_lossy(bytes);
    for bad_path in ["/nix/store/", "/home/", "/root/"] {
        if blob.contains(bad_path) {
            return Err(bad(format!(
                "{where_}: contains a build-host path ({bad_path}). Published manifests must be \
                 portable; record a reference, not a local path."
            )));
        }
    }
    Ok(())
}

// --- the recipe --------------------------------------------------------------

#[derive(Debug, serde::Deserialize)]
struct Recipe {
    namespace: Option<String>,
    name: Option<String>,
    label: Option<String>,
    plow_git: Option<String>,
    /// `refused` names a point `plowc` cannot emit; it never packs, so its
    /// checkpoint is not held to the pinning rule.
    status: Option<String>,
    #[serde(default)]
    target: RecipeTarget,
    #[serde(default)]
    checkpoint: RecipeCheckpoint,
    #[serde(default)]
    cell: RecipeCell,
    #[serde(default)]
    serve: RecipeServe,
    objects: Option<RecipeObjects>,
    #[serde(default)]
    sub_emit: Vec<RecipeSubEmit>,
    build: Option<RecipeBuild>,
    speech_tags: Option<plow_asset::speech_tag_policy::SpeechTagPolicy>,
    bench: Option<serde_json::Value>,
    gates: Option<serde_json::Value>,
}

#[derive(Debug, serde::Deserialize)]
struct RecipeSubEmit {
    name: String,
    hf_dir: String,
}

/// `[build]`: a packet built by a script rather than a `plowc` emit.
#[derive(Debug, Default, serde::Deserialize)]
struct RecipeBuild {
    /// The packet the build writes; the bundle's `packet`.
    output: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct RecipeTarget {
    vendor: Option<String>,
    isa: Option<String>,
    sku: Option<String>,
    units: Option<u32>,
    mem_bytes: Option<u64>,
    num_gpus: Option<u32>,
    parallel: Option<String>,
    toolchain: Option<String>,
    max_ctx: Option<u32>,
    executor: Option<String>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct RecipeCheckpoint {
    source: Option<String>,
    revision: Option<String>,
    /// A speech model's codec, lowered to its own `codec.pkt`.
    codec: Option<String>,
    layout: Option<String>,
    shards: Option<u32>,
    bytes: Option<u64>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct RecipeCell {
    max_ctx: Option<u32>,
    hf_dir: Option<String>,
}

/// A checkpoint the recipe's own prep steps write under the build's out dir: plow-produced,
/// so the bundle carries it as a view of the snapshot.
fn prepared(hf_dir: &str) -> bool {
    hf_dir.starts_with("{out}/")
}

#[derive(Debug, Default, serde::Deserialize)]
struct RecipeServe {
    #[serde(default)]
    env: BTreeMap<String, String>,
}

#[derive(Debug, Default, serde::Deserialize)]
struct RecipeObjects {
    script: Option<String>,
    #[serde(default)]
    env: BTreeMap<String, String>,
}

/// What a recipe does not say, and a build host does. Both are recorded in the
/// manifests, so both must be reproducible: pass them to regenerate a manifest
/// whose recipe has since moved, and otherwise leave them out and let the
/// recipe answer.
#[derive(Debug, Default, Clone, Copy)]
pub struct Overrides<'a> {
    /// The commit that actually built these bytes, when it is not the commit
    /// the recipe pins. A published build records the recipe's pin, so this
    /// is for reproducing an existing manifest, not for publishing a new one.
    pub plow_git: Option<&'a str>,
    /// The compiler the objects were built with, for a recipe whose `[target]`
    /// does not record one.
    pub toolchain: Option<&'a str>,
}

/// Everything the manifests take from the recipe, resolved.
#[derive(Debug, Clone)]
pub struct Identity {
    pub namespace: String,
    pub name: String,
    /// `None` derives it from the target — see [`default_label`].
    pub label: Option<String>,
    pub plow_git: String,
    pub target: Target,
    pub parallel: Parallel,
    pub max_ctx: u32,
    pub toolchain: String,
    /// The objset's `script`: the build script for a recipe that has one,
    /// otherwise the recipe itself, which is what drove the emit.
    pub script: String,
    /// `[objects] env` — the env the build script was driven with.
    pub objects_env: BTreeMap<String, String>,
    /// Files the recipe says the build serves with, which the bundle must
    /// therefore carry: `codec.pkt` for a `[checkpoint] codec`,
    /// `cublaslt_algos.jsonl` for a timed `PLOW_LT_RUNG_ALGOS`. A published
    /// generation is immutable, so a missing one is refused at pack time
    /// rather than found by the first client that loads it.
    pub required: Vec<String>,
    /// `[serve] env` — the settings this variant was validated with.
    pub runtime_env: BTreeMap<String, String>,
    pub checkpoint_source: String,
    pub checkpoint_revision: String,
    pub checkpoint_layout: String,
    pub checkpoint_shards: u32,
    pub checkpoint_bytes: u64,
    /// The recipe's path, recorded verbatim.
    pub recipe: String,
    /// The bundle directories (`checkpoint`, `<sub>/checkpoint`) whose checkpoint the recipe's
    /// prep built: each is published as a view of the snapshot.
    pub views: Vec<String>,
    /// A `[build]` recipe's packet (`[build] output`): its packets carry every weight, so the
    /// bundle names its checkpoint for provenance only ([`plow_asset::dist::PACKET_LAYOUT`]).
    pub packet: Option<String>,
    /// `[target] executor = "host"`: plowrt runs the packet on the host CPU; no objset.
    pub host: bool,
}

/// Read a variant's identity out of its recipe.
pub fn read_recipe(path: &Path, over: Overrides<'_>) -> Result<Identity> {
    let text =
        std::fs::read_to_string(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
    let r: Recipe = toml::from_str(&text)
        .map_err(|e| bad(format!("{}: not a valid recipe: {e}", path.display())))?;
    let speech = r
        .bench
        .as_ref()
        .and_then(|v| v.get("kind"))
        .and_then(|v| v.as_str())
        == Some("speech")
        || r.gates
            .as_ref()
            .and_then(|v| v.as_object())
            .is_some_and(|g| {
                g.keys()
                    .any(|k| k.starts_with("tts_cer") || k.starts_with("tts_tags"))
            });
    if speech && r.speech_tags.is_none() {
        return Err(bad(
            "speech_tags: missing policy (packing requires an emitted map or measured waiver)",
        ));
    }
    if let Some(policy) = &r.speech_tags {
        policy.validate().map_err(bad)?;
        if r.status.as_deref() == Some("validated")
            && policy
                .map
                .as_ref()
                .is_some_and(|m| m.tags.values().any(|e| e.text.is_some()))
            && policy.native_sweep.is_none()
        {
            return Err(bad(
                "speech_tags: a validated native map requires its measured native_sweep report",
            ));
        }
    }
    let at = |what: &str| format!("{}: {what}", path.display());

    let name = r
        .name
        .ok_or_else(|| bad(at("no `name` — that is the model's name in the registry")))?;
    let plow_git = match over.plow_git.map(str::to_string).or(r.plow_git) {
        Some(g)
            if g.len() == 40
                && g.chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()) =>
        {
            g
        }
        Some(g) => {
            return Err(bad(at(&format!(
                "`plow_git` must be a full 40-hex commit, got {g:?}"
            ))))
        }
        None => {
            return Err(bad(at(
                "no `plow_git`: an asset is tied to the compiler source that built it",
            )))
        }
    };

    let t = r.target;
    let isa = t.isa.ok_or_else(|| bad(at("[target] has no `isa`")))?;
    let sku = t.sku.ok_or_else(|| bad(at("[target] has no `sku`")))?;
    let units = t.units.ok_or_else(|| bad(at("[target] has no `units`")))?;
    // The SKU's memory, not the build host's: the published veena bundle
    // records 80 GiB for an `H100 SXM5` although it was built on an H200,
    // because the number describes the SKU a variant is published FOR. A probe
    // would have written the host's memory and made the variant unselectable on
    // real H100s. `hwspec` already holds the figure per part — and it is what
    // separates MI300X from MI325X, identical at gfx942/304 CUs — so a recipe
    // need not carry it, and 40 of the 41 do not.
    let spec = hwspec::registry::lookup(&sku);
    let mem_bytes = match (t.mem_bytes, spec) {
        // A stated figure wins: on a unified-memory part `hwspec` records the
        // SMALLEST shipping configuration (M4 Max: 36 GiB of a possible 128),
        // and a CPU's DRAM is configurable too, so there the registry is a
        // floor rather than this target's memory.
        (Some(m), _) => m,
        (None, Some(s)) => s.mem.capacity.0,
        (None, None) => {
            return Err(bad(at(&format!(
                "[target] has no `mem_bytes` and sku {sku:?} is not a part `hwspec` describes: \
                 state the memory of the SKU this variant is published for, or add the part \
                 under `crates/hwspec/src`"
            ))))
        }
    };
    // Enabled SMs/CUs are fixed by the part, never configured, so a recipe that
    // disagrees with the registry is a typo and not a narrower deployment.
    if let Some(s) = spec.filter(|s| units != s.sm_count) {
        return Err(bad(at(&format!(
            "[target] units = {units} but {sku} has {}",
            s.sm_count
        ))));
    }
    // The recipe declares the vendor; the Python packer derived it from the isa
    // prefix. Prefer the recipe and refuse a disagreement rather than silently
    // publishing under a vendor nothing will match.
    let derived = if isa.starts_with("gfx") {
        "amd"
    } else if isa.starts_with("sm_") {
        "nvidia"
    } else {
        "cpu"
    };
    let vendor = match t.vendor {
        // Only a GPU isa names its vendor, so only there is a disagreement a
        // mistake. A CPU isa does not: `amx` is Intel's and `neon` is Arm's, so
        // `vendor = "intel"` beside `isa = "amx"` is the right declaration and
        // not a conflict — refusing it left the five Xeon recipes unpackable.
        Some(v) if derived != "cpu" && v != derived && !runtime_compiled(&v) => {
            return Err(bad(at(&format!(
                "[target] vendor = {v:?} but isa {isa:?} is a {derived} isa"
            ))))
        }
        Some(v) => v,
        None => derived.to_string(),
    };

    // A runtime-compiled target (Apple) ships no objset, so nothing records a
    // compiler: the shaders are built on the serving machine. NVIDIA cubins are
    // pinned by their bytes in the objset content digest and run on any driver
    // new enough for them, so a CUDA toolchain is recorded when the recipe names
    // one but never required.
    let toolchain =
        match over.toolchain.map(str::to_string).or(t.toolchain) {
            Some(v) => v,
            None if runtime_compiled(&vendor) || vendor == "nvidia" => String::new(),
            None => return Err(bad(at(
                "[target] has no `toolchain`: the objset records the compiler its objects were \
                 built with, and the recipe checker verifies it",
            ))),
        };

    // A script-built packet carries every weight it binds: no plowc emit, no checkpoint to bind.
    let packet = match (&r.build, &r.cell.hf_dir) {
        (Some(b), None) => Some(b.output.clone().ok_or_else(|| {
            bad(at(
                "[build] names no `output`: a script-built bundle ships the packet it names",
            ))
        })?),
        _ => None,
    };
    let host = match t.executor.as_deref() {
        None => false,
        Some(plow_asset::dist::HOST_EXECUTOR) if packet.is_some() => true,
        Some(e) => {
            return Err(bad(at(&format!(
                "[target] executor = {e:?}: only a `[build]` packet runs on `{}`",
                plow_asset::dist::HOST_EXECUTOR
            ))))
        }
    };
    let max_ctx = t
        .max_ctx
        .or(r.cell.max_ctx)
        .ok_or_else(|| bad(at("neither [target] nor [cell] declares `max_ctx`")))?;

    let c = r.checkpoint;
    let checkpoint_source = c
        .source
        .ok_or_else(|| bad(at("[checkpoint] has no `source`")))?;
    // A bundle names the bytes it was validated against. A Hugging Face
    // repository is referenced by commit, never by a branch that can move, and
    // the farm links the snapshot of exactly that commit into the bundle.
    let provenance_only = packet.is_some() && !checkpoint_source.starts_with("hf:");
    if checkpoint_source != plow_asset::dist::DIST_SOURCE
        && r.status.as_deref() != Some("refused")
        && !provenance_only
    {
        let repo = checkpoint_source.strip_prefix("hf:").ok_or_else(|| {
            bad(at(&format!(
                "[checkpoint] source {checkpoint_source:?} is not `hf:<org>/<repo>` or `dist`; a \
                 Hugging Face checkpoint is written `hf:<org>/<repo>` with `revision` set to its \
                 commit"
            )))
        })?;
        if repo.split('/').count() != 2 || repo.contains("..") {
            return Err(bad(at(&format!(
                "[checkpoint] source {checkpoint_source:?}: expected `hf:<org>/<repo>`"
            ))));
        }
        match c.revision.as_deref() {
            Some(rev) if super::weights::is_commit(rev) => {}
            Some(rev) => {
                return Err(bad(at(&format!(
                    "[checkpoint] revision {rev:?} is not a 40-hex commit. A branch can move after \
                     the build and hand a client other weights; pin the commit the build used \
                     (the `snapshots/<commit>` directory name in the HF cache)"
                ))))
            }
            None => {
                return Err(bad(at(
                    "[checkpoint] has no `revision`: a Hugging Face checkpoint is pinned to the \
                     40-hex commit the build used",
                )))
            }
        }
    }

    let mut required = Vec::new();
    if c.codec.is_some() {
        required.push("codec.pkt".to_string());
    }
    if r.serve
        .env
        .get("PLOW_LT_RUNG_ALGOS")
        .is_some_and(|v| matches!(v.as_str(), "1" | "true"))
    {
        required.push("cublaslt_algos.jsonl".to_string());
    }

    let mut views = Vec::new();
    if r.cell.hf_dir.as_deref().is_some_and(prepared) {
        views.push("checkpoint".to_string());
    }
    for sub in &r.sub_emit {
        if prepared(&sub.hf_dir) {
            views.push(format!("{}/checkpoint", sub.name));
        }
    }

    let objects = r.objects.unwrap_or_default();
    Ok(Identity {
        namespace: r.namespace.unwrap_or_else(|| "infervisor".into()),
        name,
        label: r.label,
        plow_git,
        target: Target {
            vendor,
            isa,
            sku,
            units,
            mem_bytes,
        },
        parallel: Parallel {
            mode: t.parallel.unwrap_or_else(|| "tp".into()),
            n: t.num_gpus.unwrap_or(1),
        },
        max_ctx,
        toolchain,
        script: objects
            .script
            .unwrap_or_else(|| path.to_string_lossy().into_owned()),
        objects_env: objects.env,
        required,
        runtime_env: r.serve.env,
        checkpoint_source,
        checkpoint_revision: c.revision.unwrap_or_default(),
        checkpoint_layout: match &packet {
            Some(_) => plow_asset::dist::PACKET_LAYOUT.to_string(),
            None => c.layout.unwrap_or_else(|| "hf-safetensors".into()),
        },
        checkpoint_shards: c.shards.unwrap_or(0),
        checkpoint_bytes: c.bytes.unwrap_or(0),
        recipe: path.to_string_lossy().into_owned(),
        views,
        packet,
        host,
    })
}

/// `<isa>-<sku>-<parallel>-<ctx>[-<features>]`, lowercased.
///
/// Derived rather than typed so two builds of the same point cannot be given
/// different names by accident.
pub fn default_label(
    target: &Target,
    parallel: &Parallel,
    max_ctx: u32,
    features: &BTreeMap<String, bool>,
) -> String {
    let on = ["fp8_kv", "mxfp4_weights", "w8a8", "fp8_weights"]
        .into_iter()
        .filter(|k| features.get(*k).copied().unwrap_or(false));
    let ctx = if max_ctx % 1024 == 0 {
        format!("{}k", max_ctx / 1024)
    } else {
        max_ctx.to_string()
    };
    let sku = target
        .sku
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-");
    let mut parts = vec![target.isa.clone(), sku, format!("tp{}", parallel.n), ctx];
    parts.extend(on.map(str::to_string));
    parts.join("-")
}

// --- objects -----------------------------------------------------------------

/// Reads an object's marker symbols. A trait so a test can assert a `.cubin`
/// never reaches it: a cubin's identity is its digest, and running `readelf`
/// over one would be a silent waste at best and a wrong `arms` list at worst.
pub trait SymbolReader {
    fn symbols(&self, path: &Path) -> Result<Vec<String>>;
}

/// `readelf -sW`, as the retired Python packer ran it: `$PLOW_READELF` (what
/// `nix develop` exports and `build_gfx942.sh` itself requires), then
/// `llvm-readelf`, then `readelf`.
///
/// Still a subprocess, deliberately. An ELF crate would be a new dependency
/// and a new way for the AMD arm set to differ from the one the build script
/// and the doctor read.
pub struct Readelf;

impl SymbolReader for Readelf {
    fn symbols(&self, path: &Path) -> Result<Vec<String>> {
        let mut tried = Vec::new();
        for bin in crate::config::RuntimeConfig::readelf_env()
            .into_iter()
            .chain(["llvm-readelf".to_string(), "readelf".to_string()])
        {
            match std::process::Command::new(&bin)
                .arg("-sW")
                .arg(path)
                .output()
            {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    tried.push(bin);
                    continue;
                }
                Err(e) => return Err(bad(format!("{bin} {}: {e}", path.display()))),
                Ok(out) if !out.status.success() => {
                    return Err(bad(format!(
                        "{}: {bin} failed: {}",
                        path.display(),
                        String::from_utf8_lossy(&out.stderr).trim()
                    )))
                }
                Ok(out) => return Ok(markers(&String::from_utf8_lossy(&out.stdout))),
            }
        }
        Err(bad(format!(
            "no readelf available (tried {}). Set PLOW_READELF, or run inside `nix develop`, \
             which exports it pointing at the toolchain's llvm-readelf.",
            tried.join(", ")
        )))
    }
}

/// The `plow_*` marker symbols in `readelf -sW` output, sorted and deduped.
///
/// Only the markers matter to the loader; recording every symbol would bloat
/// the manifest without adding a check. Ported from the retired Python packer's
/// `\b(plow_[A-Za-z0-9_]+)\b`: a maximal run of word characters that starts
/// with `plow_` at a non-word boundary.
fn markers(text: &str) -> Vec<String> {
    fn word(b: u8) -> bool {
        b.is_ascii_alphanumeric() || b == b'_'
    }
    let b = text.as_bytes();
    let mut found = std::collections::BTreeSet::new();
    let mut i = 0;
    while i < b.len() {
        if !word(b[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < b.len() && word(b[i]) {
            i += 1;
        }
        let run = &text[start..i];
        if run.starts_with("plow_") && run.len() > "plow_".len() {
            found.insert(run.to_string());
        }
    }
    found.into_iter().collect()
}

/// A specialised object stamps the packet it pairs with.
///
/// The value is encoded in the symbol NAME (`plow_packet_hash_lo_<hex>`),
/// because plowrt reads `.symtab` before the object is on a device and a value
/// would cost a round trip. A GENERAL object — every arm compiled — carries no
/// stamp and pairs with any packet.
fn packet_hash(syms: &[String]) -> Result<Option<String>> {
    let mut lo = None;
    let mut hi = None;
    for s in syms {
        if let Some(h) = s.strip_prefix("plow_packet_hash_lo_") {
            lo = Some(h);
        } else if let Some(h) = s.strip_prefix("plow_packet_hash_hi_") {
            hi = Some(h);
        }
    }
    match (lo, hi) {
        (None, None) => Ok(None),
        (Some(lo), Some(hi)) => {
            let parse = |s: &str| {
                u64::from_str_radix(s, 16)
                    .map_err(|_| bad(format!("packet-hash stamp {s:?} is not hex")))
            };
            Ok(Some(format!("{:#018x}", parse(hi)? << 32 | parse(lo)?)))
        }
        _ => Err(bad(
            "object carries half a packet-hash stamp; that is a broken build",
        )),
    }
}

/// The built objects in a directory, in the ONE order the objset id depends on.
///
/// `.elf`, then `.co`, then `.cubin`, each sorted within its class.
/// `content_digest` hashes `objects` in order, so the order is part of the
/// identity: cubins come last specifically so a directory that gains one cannot
/// re-id an objset that already exists. Merging the three globs into a single
/// sorted pass would silently change published ids.
pub fn discover_objects(dir: &Path, syms: &dyn SymbolReader) -> Result<Vec<ObjSetObject>> {
    if !dir.is_dir() {
        return Err(bad(format!("{} is not a directory", dir.display())));
    }
    let mut by_suffix: BTreeMap<&str, Vec<PathBuf>> = BTreeMap::new();
    for entry in std::fs::read_dir(dir).map_err(|e| bad(format!("{}: {e}", dir.display())))? {
        let p = entry
            .map_err(|e| bad(format!("{}: {e}", dir.display())))?
            .path();
        if !p.is_file() {
            continue;
        }
        let name = p
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        for suffix in [".elf", ".co", ".cubin"] {
            if name.ends_with(suffix) {
                by_suffix.entry(suffix).or_default().push(p);
                break;
            }
        }
    }
    for paths in by_suffix.values_mut() {
        paths.sort();
    }
    if by_suffix.values().all(Vec::is_empty) {
        return Err(bad(format!(
            "{}: no *.elf, *.cubin or *.co found",
            dir.display()
        )));
    }

    let mut objects = Vec::new();
    let mut stamps = std::collections::BTreeSet::new();
    // `.elf` only: the AMD path, where an extension's objects MUST be stamped
    // (`check_object_stamp` refuses an unstamped one) and the loader reads the
    // arms before the object is on a device.
    for p in by_suffix.get(".elf").into_iter().flatten() {
        let found = syms.symbols(p)?;
        let stamp = packet_hash(&found)?;
        if let Some(s) = &stamp {
            stamps.insert(s.clone());
        }
        objects.push(ObjSetObject {
            name: file_name(p),
            sha256: sha256_file(p)?,
            bytes: size(p)?,
            arms: found
                .into_iter()
                .filter(|s| !s.starts_with("plow_packet_hash_"))
                .collect(),
            packet_hash: stamp,
        });
    }
    // Pinned vendor code objects (AITER fmoe / MLA `.co`) are loaded by plow's
    // own adapters and are part of the set: a set without the 64-row fmoe
    // object serves the 32x256 MoE tile. They carry no markers and no stamp
    // (each adapter script pins its own sha256), so identity is the digest
    // alone — and NVIDIA cubins are recorded the same way, for the same reason.
    for suffix in [".co", ".cubin"] {
        for p in by_suffix.get(suffix).into_iter().flatten() {
            objects.push(ObjSetObject {
                name: file_name(p),
                sha256: sha256_file(p)?,
                bytes: size(p)?,
                arms: Vec::new(),
                packet_hash: None,
            });
        }
    }
    if stamps.len() > 1 {
        return Err(bad(format!(
            "objects in this directory are stamped for different packets: {:?}. \
             That is a mixed build, not an objset.",
            stamps
        )));
    }
    Ok(objects)
}

/// NVIDIA: every object serving resolves for this asset directory, as the build left it — the
/// directory's own `*.cubin` and `*.aot` (interpreter, generated, sampler and library images
/// are read from beside the packet), then `objects/*` named `objects/<file>`
/// (`asset::serve::objects_dir`, the segmented-prefill set). A name in both places names two
/// different builds and serving reads each from its own place, so both ship.
///
/// Identity order: the top-level cubins exactly as [`discover_objects`] orders them, so a
/// directory without library objects or segment objects keeps the objset id it always had; then
/// `.aot`; then `objects/`, each sorted.
pub fn served_objects(assets: &Path) -> Result<Vec<ObjSetObject>> {
    // A packet that embeds its interpreters may have no object beside it but its library or
    // segment objects; only an empty set as a whole is an error.
    let mut objects = if sorted_files(assets, &[".elf", ".co", ".cubin"])?.is_empty() {
        Vec::new()
    } else {
        discover_objects(assets, &Readelf)?
    };
    for p in sorted_files(assets, &[LIBRARY_OBJECT_SUFFIX])? {
        objects.push(digest_object(&p, file_name(&p))?);
    }
    let segment_dir = assets.join(plow_asset::dist::SEGMENT_OBJECTS_DIR);
    if segment_dir.is_dir() {
        let mut stray = Vec::new();
        for p in sorted_files(&segment_dir, &[])? {
            let name = file_name(&p);
            if [".cubin", LIBRARY_OBJECT_SUFFIX]
                .iter()
                .any(|s| name.ends_with(s))
            {
                let path = format!("{}/{name}", plow_asset::dist::SEGMENT_OBJECTS_DIR);
                objects.push(digest_object(&p, path)?);
            } else {
                stray.push(name);
            }
        }
        if !stray.is_empty() {
            eprintln!(
                "warning: {} file(s) in {} are not objects and are NOT in this bundle: {}",
                stray.len(),
                segment_dir.display(),
                stray.join(", ")
            );
        }
    }
    if objects.is_empty() {
        return Err(bad(format!(
            "{}: no *.cubin, *.aot or {}/ objects found",
            assets.display(),
            plow_asset::dist::SEGMENT_OBJECTS_DIR
        )));
    }
    Ok(objects)
}

/// Regular files directly in `dir`, sorted; only those ending in one of `suffixes` unless it is
/// empty.
fn sorted_files(dir: &Path, suffixes: &[&str]) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| bad(format!("{}: {e}", dir.display())))? {
        let p = entry
            .map_err(|e| bad(format!("{}: {e}", dir.display())))?
            .path();
        let name = file_name(&p);
        if p.is_file() && (suffixes.is_empty() || suffixes.iter().any(|s| name.ends_with(s))) {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

fn digest_object(p: &Path, name: String) -> Result<ObjSetObject> {
    Ok(ObjSetObject {
        name,
        sha256: sha256_file(p)?,
        bytes: size(p)?,
        arms: Vec::new(),
        packet_hash: None,
    })
}

/// The objects a packet loads from its asset directory by name, with the digest it pins when
/// it pins one: every `segment_roles` object, and the GDN prefill library when a
/// prefill program runs `QwenGdnPrefill`. Empty for a file that is not a PLOWDEV packet.
pub fn declared_objects(packet: &Path) -> Result<Vec<(String, Option<String>)>> {
    use crate::asset::devblob::DevBlob;
    let buf = std::fs::read(packet).map_err(|e| bad(format!("{}: {e}", packet.display())))?;
    let magic: Option<&[u8; 8]> = buf.get(..8).and_then(|s| s.try_into().ok());
    if !magic.is_some_and(packet::devbuild::is_blob_magic) {
        return Ok(Vec::new());
    }
    let blob =
        DevBlob::parse_l2(&buf, true).map_err(|e| bad(format!("{}: {e}", packet.display())))?;
    let mut out = Vec::new();
    let section = blob
        .reserved_metadata(&buf, plow_asset::segment_roles::SECTION)
        .map_err(|e| bad(format!("{}: {e}", packet.display())))?;
    if let Some(bytes) = section {
        let v: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|e| bad(format!("{}: segment_roles section: {e}", packet.display())))?;
        // Every declared object names the file a role loads from beside the packet.
        for object in v["objects"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(_, o)| o)
        {
            if let Some(file) = object["file"].as_str().filter(|f| !f.is_empty()) {
                let sha = object["sha256"].as_str().map(str::to_string);
                out.push((file.to_string(), sha));
            }
        }
    }
    let gdn = blob
        .progs
        .iter()
        .flat_map(|p| &p.insts)
        .any(|i| i.op == packet::dev::DevOp::QwenGdnPrefill as u16);
    if gdn {
        out.push((GDN_PREFILL_OBJECT.to_string(), None));
    }
    Ok(out)
}

/// Refuse an objset that does not carry, byte for byte, an object the packet loads by name.
fn check_declared(packet: &Path, objects: &[ObjSetObject]) -> Result<()> {
    let mut problems = Vec::new();
    for (file, sha) in declared_objects(packet)? {
        match objects.iter().find(|o| o.name == file) {
            None => problems.push(format!("{file} is not in the asset directory")),
            Some(o) if sha.as_deref().is_some_and(|s| s != o.sha256) => problems.push(format!(
                "{file} hashes to {}, the packet declares {}",
                o.sha256,
                sha.unwrap_or_default()
            )),
            Some(_) => {}
        }
    }
    if problems.is_empty() {
        return Ok(());
    }
    Err(bad(format!(
        "{}: the objects this packet loads by name do not match what would ship: {}",
        packet.display(),
        problems.join("; ")
    )))
}

/// A `--objects` directory beside an NVIDIA asset directory must be the very set serving
/// resolves there (`<assets>/objects`): the objset ships what qualification served, never
/// another copy of it.
fn check_named_objects(assets: &Path, named: &Path, objects: &[ObjSetObject]) -> Result<()> {
    let same = |a: &Path, b: &Path| matches!((a.canonicalize(), b.canonicalize()), (Ok(x), Ok(y)) if x == y);
    let segment_dir = assets.join(plow_asset::dist::SEGMENT_OBJECTS_DIR);
    if same(named, assets) || same(named, &segment_dir) {
        return Ok(());
    }
    let prefix = format!("{}/", plow_asset::dist::SEGMENT_OBJECTS_DIR);
    let mut problems = Vec::new();
    let named_files = sorted_files(named, &[])?;
    for p in &named_files {
        let name = file_name(p);
        if !OBJSET_SUFFIXES.iter().any(|s| name.ends_with(s)) {
            continue;
        }
        let want = sha256_file(p)?;
        match objects.iter().find(|o| o.name == format!("{prefix}{name}")) {
            None => problems.push(format!("{name} is not in {}", segment_dir.display())),
            Some(o) if o.sha256 != want => {
                problems.push(format!("{name} differs from {}", segment_dir.display()))
            }
            Some(_) => {}
        }
    }
    for o in objects.iter().filter(|o| o.name.starts_with(&prefix)) {
        let name = &o.name[prefix.len()..];
        if !named_files.iter().any(|p| file_name(p) == name) {
            problems.push(format!(
                "{name} is served from {} but not in {}",
                segment_dir.display(),
                named.display()
            ));
        }
    }
    if problems.is_empty() {
        return Ok(());
    }
    Err(bad(format!(
        "{}: NVIDIA serving reads segment objects from {}, and this directory is not that set: {}. \
         Pack the asset directory the build served from",
        named.display(),
        segment_dir.display(),
        problems.join("; ")
    )))
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned()
}

fn size(p: &Path) -> Result<u64> {
    Ok(std::fs::metadata(p)
        .map_err(|e| bad(format!("{}: {e}", p.display())))?
        .len())
}

/// Assemble the objset and derive its id.
///
/// ```text
/// defines_digest = sha256(canonical_json(build_defines.json))
/// content_digest = sha256(canonical_json([[name, sha256], ...]))   # ORDERED
/// objset_id      = sha256("\x1f".join([isa, sku, toolchain,
///                                      defines_digest, content_digest,
///                                      plow_git]))[:12]
/// ```
///
/// The CONTENT is part of the identity, not just the build inputs. Keying on
/// inputs alone collides whenever they do not fully determine the output — a
/// `PLOW_ROWS_ONLY` partial directory and a full build share their defines, and
/// a directory with no `build_defines.json` shares them with every other such
/// directory. Two objsets with one id would overwrite each other at
/// `v1/objsets/<id>.json`.
pub fn build_objset(
    id: &Identity,
    objects_dir: &Path,
    objects: Vec<ObjSetObject>,
) -> Result<ObjSet> {
    let defines_path = objects_dir.join("build_defines.json");
    let defines: BTreeMap<String, String> = if defines_path.exists() {
        let raw = std::fs::read(&defines_path)
            .map_err(|e| bad(format!("{}: {e}", defines_path.display())))?;
        serde_json::from_slice(&raw)
            .map_err(|e| bad(format!("{}: not valid JSON: {e}", defines_path.display())))?
    } else {
        // Every CUDA recipe emits cubins straight into the assets directory
        // with no defines file, which is why this warns rather than refuses —
        // and why `content_digest` exists.
        eprintln!(
            "warning: {} is absent, so this objset records no -D set and the recipe checker \
             cannot verify it",
            defines_path.display()
        );
        BTreeMap::new()
    };

    let defines_digest = sha256_bytes(&canonical_json(&defines)?);
    let content: Vec<[&str; 2]> = objects
        .iter()
        .map(|o| [o.name.as_str(), o.sha256.as_str()])
        .collect();
    let content_digest = sha256_bytes(&canonical_json(&content)?);
    let objset_id = short_id(&[
        &id.target.isa,
        &id.target.sku,
        &id.toolchain,
        &defines_digest,
        &content_digest,
        &id.plow_git,
    ]);

    Ok(ObjSet {
        schema: OBJSET_SCHEMA.to_string(),
        objset_id,
        target: id.target.clone(),
        toolchain: id.toolchain.clone(),
        plow_git: id.plow_git.clone(),
        script: id.script.clone(),
        env: id.objects_env.clone(),
        defines,
        objects,
    })
}

// --- the bundle --------------------------------------------------------------

/// The tokenizer's provenance, which must be declared.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub enum Tokenizer {
    /// The checkpoint's own file: NOT distributed. `plowc` symlinks it, and
    /// shipping it would redistribute HF content and risk it diverging from the
    /// checkpoint it must match.
    #[default]
    Checkpoint,
    /// Generated by plow (`scripts/kimi_k3_tokenizer.py` reconstructing a
    /// `tokenizer.json` for a checkpoint that ships only `tiktoken.model`).
    /// Distributed, and pinned by digest.
    Derived { generator: String, verified: bool },
}

/// What a bundle needs that a recipe cannot know.
#[derive(Debug, Default)]
pub struct BundleArgs<'a> {
    pub tokenizer: Tokenizer,
    /// `<max>:<objset.json>` narrow-rung decode overrides, already parsed.
    pub lowrung: Vec<(u32, PathBuf)>,
    /// Carry the checkpoint in the distribution (source `dist`) instead of
    /// naming `hf:`: every file under this directory becomes a blob.
    pub checkpoint_dir: Option<&'a Path>,
}

/// A specialised object pairs only with the packet that produced it (`<dir>/build.json`'s
/// pairing hash). Catching a mismatch here turns a refusal twenty frames into the loader into an
/// obvious one.
fn check_stamps<'a>(
    dir: &Path,
    pairing: Option<&str>,
    objects: impl Iterator<Item = &'a ObjSetObject>,
) -> Result<()> {
    for o in objects {
        let Some(stamp) = o.packet_hash.as_deref() else {
            continue;
        };
        match pairing {
            None => {
                return Err(bad(format!(
                    "{} is stamped for packet {stamp}, but {} records no pairing hash. \
                     Publishing this pair would be refused at load.",
                    o.name,
                    dir.join("build.json").display()
                )))
            }
            Some(p) if p != stamp => {
                return Err(bad(format!(
                    "{} is stamped for packet {stamp}, but this packet is {p}. The objset was \
                     built against a different emit; rebuild one of them.",
                    o.name
                )))
            }
            Some(_) => {}
        }
    }
    Ok(())
}

/// Assemble the bundle from a built asset directory.
pub fn build_bundle(
    assets: &Path,
    id: &Identity,
    objset: Option<&ObjSet>,
    args: &BundleArgs<'_>,
) -> Result<Bundle> {
    if !assets.is_dir() {
        return Err(bad(format!("{} is not a directory", assets.display())));
    }
    match (objset, runtime_compiled(&id.target.vendor)) {
        // Apple's Metal shaders compile on the serving machine from the packet;
        // its bundle carries no objset and `serve` compiles on load.
        (Some(_), true) => {
            return Err(bad(format!(
                "vendor {} compiles its kernels at run time and ships no objset",
                id.target.vendor
            )))
        }
        (None, false) if !id.host => {
            return Err(bad(format!(
                "vendor {} ships device objects: the bundle needs an objset",
                id.target.vendor
            )))
        }
        _ => {}
    }

    let build_json: serde_json::Value = read_json_if_present(&assets.join("build.json"))?;
    let pairing = objset.and(
        build_json
            .get("pairing")
            .and_then(|p| p.get("hash"))
            .and_then(|h| h.as_str()),
    );

    // A sub-bundle's objects were checked against its own packet when the objset was built.
    let root_objects = objset
        .map(|s| s.objects.as_slice())
        .unwrap_or_default()
        .iter()
        .filter(|o| !plow_asset::dist::in_sub_bundle(&o.name));
    check_stamps(assets, pairing, root_objects)?;

    let mut files = Vec::new();
    let mut unclaimed = Vec::new();
    let mut names: Vec<(String, PathBuf)> = sorted_files(assets, &[])?
        .into_iter()
        .map(|p| (file_name(&p), p))
        .collect();
    // A sub-bundle (Breeze's `text_encoder/`) is its own emit beside the packet: its files ship
    // under its name, classified like the top level's.
    for sub in sub_bundles(assets)? {
        for p in sorted_files(&assets.join(&sub), &[])? {
            names.push((format!("{sub}/{}", file_name(&p)), p));
        }
    }
    for (name, p) in &names {
        let base = name.rsplit('/').next().unwrap_or(name);
        let role = match &id.packet {
            Some(packet) if name == packet => Some("packet"),
            _ if name.contains('/') && base == "model.pkt" => Some("sub_packet"),
            _ => classify(base),
        };
        match role {
            Some(role) => files.push(FileRef {
                role: role.to_string(),
                name: name.clone(),
                sha256: sha256_file(p)?,
                bytes: size(p)?,
            }),
            // Say so. An unclassified asset used to vanish in silence, and the
            // result is a published generation that looks complete and serves
            // wrongly — a published generation is IMMUTABLE.
            None => {
                if !SKIP_QUIETLY.contains(&base)
                    && !OBJSET_SUFFIXES.iter().any(|s| base.ends_with(s))
                {
                    unclaimed.push(name.clone());
                }
            }
        }
    }
    for need in &id.required {
        if !files.iter().any(|f| f.name == *need) {
            return Err(bad(format!(
                "{}: no {need}, which the recipe says this model serves with; the build is \
                 incomplete, and a published generation is immutable",
                assets.display()
            )));
        }
    }
    // Without a table plowrt times every BF16 cuBLASLt plan at load, and the pick (a split-K
    // reduction among them) re-rolls per server start: greedy output then differs between starts.
    let lt_shapes = lt_bf16_shapes(&build_json);
    if lt_shapes > 0 && !files.iter().any(|f| f.name == "cublaslt_algos.jsonl") {
        return Err(bad(format!(
            "{}: build.json plans {lt_shapes} BF16 cuBLASLt shape(s) but there is no \
             cublaslt_algos.jsonl; serving would time them at every load and its numerics would \
             vary per start. Run the campaign probe step to record the table",
            assets.display()
        )));
    }
    if !unclaimed.is_empty() {
        // stderr: an `--out`-less run prints the manifest to stdout.
        eprintln!(
            "warning: {} file(s) in {} carry no role and are NOT in this bundle: {}. A published \
             generation is immutable, so give them a role before publishing if the runtime needs \
             them.",
            unclaimed.len(),
            assets.display(),
            unclaimed.join(", ")
        );
    }
    if !files.iter().any(|f| f.role == "packet") {
        return Err(bad(format!(
            "{}: no {} — that is not a compiled asset directory",
            assets.display(),
            id.packet.as_deref().unwrap_or("model.pkt")
        )));
    }

    let tok_path = assets.join("tokenizer.json");
    let tokenizer = match args.tokenizer {
        Tokenizer::Checkpoint => TokenizerRef {
            source: Provenance::Checkpoint,
            file: "tokenizer.json".into(),
            sha256: None,
            generator: None,
            verified: false,
        },
        Tokenizer::Derived {
            ref generator,
            verified,
        } => {
            let meta = std::fs::symlink_metadata(&tok_path)
                .map_err(|e| bad(format!("{}: {e}", tok_path.display())))?;
            if !meta.is_file() {
                return Err(bad(format!(
                    "a derived tokenizer needs a REAL {} in the asset directory (a symlink means \
                     it came from the checkpoint)",
                    tok_path.display()
                )));
            }
            let sha256 = sha256_file(&tok_path)?;
            files.push(FileRef {
                role: "tokenizer".into(),
                name: "tokenizer.json".into(),
                sha256: sha256.clone(),
                bytes: meta.len(),
            });
            TokenizerRef {
                source: Provenance::Derived,
                file: "tokenizer.json".into(),
                sha256: Some(sha256),
                generator: Some(generator.clone()),
                verified,
            }
        }
    };

    let weights: serde_json::Value = read_json_if_present(&assets.join("weights.json"))?;
    // `weights.json`'s `network`, not the recipe's: the recipe's names the
    // ARCHITECTURE family (veena's says `llama`) while the bundle's is the
    // registry key `plowrt serve` registers under (`veena`). A script-built
    // packet writes no `weights.json` and serves under the model's name.
    let network = weights
        .get("network")
        .and_then(|n| n.as_str())
        .or(id.packet.as_ref().map(|_| id.name.as_str()))
        .ok_or_else(|| {
            bad(format!(
                "{}: no `network` — it is the registry key `plowrt serve` registers under",
                assets.join("weights.json").display()
            ))
        })?
        .to_string();

    let features: BTreeMap<String, bool> = build_json
        .get("features")
        .and_then(|f| f.as_object())
        .map(|f| {
            f.iter()
                .filter_map(|(k, v)| v.as_bool().map(|b| (k.clone(), b)))
                .collect()
        })
        .unwrap_or_default();
    let label = id
        .label
        .clone()
        .unwrap_or_else(|| default_label(&id.target, &id.parallel, id.max_ctx, &features));

    let mut views = Vec::new();
    for dir in &id.views {
        let bound = assets.join(dir);
        if !bound.is_dir() {
            return Err(bad(format!(
                "{}: the recipe's prep built this checkpoint, but it is not there to publish",
                bound.display()
            )));
        }
        let snapshot = super::view::local_snapshot(&id.checkpoint_source, &id.checkpoint_revision)?;
        views.push(super::view::compute(dir, &bound, &snapshot)?);
    }

    let mut bundle = Bundle {
        schema: plow_asset::dist::bundle_schema(&id.target.vendor).to_string(),
        namespace: id.namespace.clone(),
        name: id.name.clone(),
        label,
        // Both assigned at publish, against the registry's live index.
        generation: 0,
        variant_id: String::new(),
        plow_git: id.plow_git.clone(),
        network,
        target: id.target.clone(),
        parallel: id.parallel.clone(),
        max_ctx: id.max_ctx,
        files,
        tokenizer,
        checkpoint: plow_asset::dist::BundleCheckpoint {
            views,
            ..build_checkpoint(id, args.checkpoint_dir)?
        },
        objset: None,
        pairing_hash: pairing.map(str::to_string),
        runtime_env: id.runtime_env.clone(),
        recipe: Some(id.recipe.clone()),
        executor: id.host.then(|| plow_asset::dist::HOST_EXECUTOR.to_string()),
    };
    if bundle.needs_views_schema() {
        bundle.schema = plow_asset::dist::BUNDLE_SCHEMA_VIEWS.to_string();
    }
    if let Some(objset) = objset {
        bundle.objset = Some(BundleObjset {
            objset_id: objset.objset_id.clone(),
            manifest: format!("v1/objsets/{}.json", objset.objset_id),
            sha256: sha256_bytes(&canonical_json(objset)?),
            lowrung: build_lowrung(&args.lowrung)?,
        });
    } else if !args.lowrung.is_empty() {
        return Err(bad(
            "rung overrides name objsets; a runtime-compiled target has none to override",
        ));
    }
    Ok(bundle)
}

/// The sub-bundles of an asset directory: each subdirectory holding its own `model.pkt` (not a
/// link: `checkpoint` is the farm; not hidden: `.cubin-build` is build state).
fn sub_bundles(assets: &Path) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(assets).map_err(|e| bad(format!("{}: {e}", assets.display())))? {
        let p = e
            .map_err(|e| bad(format!("{}: {e}", assets.display())))?
            .path();
        let name = file_name(&p);
        let real_dir = std::fs::symlink_metadata(&p).is_ok_and(|m| m.is_dir());
        if real_dir
            && !name.starts_with('.')
            && name != plow_asset::dist::SEGMENT_OBJECTS_DIR
            && p.join("model.pkt").is_file()
        {
            out.push(name);
        }
    }
    out.sort();
    Ok(out)
}

fn classify(name: &str) -> Option<&'static str> {
    if let Some((_, role)) = ROLES.iter().find(|(n, _)| *n == name) {
        return Some(role);
    }
    if name.ends_with(".safetensors") {
        return Some("derived_shard");
    }
    // `codec.lean-checks.json`, `encoder.lean-checks.json`, ... — one per
    // sidecar packet, named after its stem, so this is a suffix rule.
    if name.ends_with(".lean-checks.json") {
        return Some("sidecar_checks");
    }
    // The sidecar packets themselves: `encoder.pkt` (an ASR frontend),
    // `s3gen.pkt`, ... — loaded by the packet runtimes beside `model.pkt`
    // by their own names. `model.pkt` and `codec.pkt` matched the table
    // above; anything else ending in `.pkt` is a sidecar, and dropping it
    // publishes an ASR bundle that transcribes nothing.
    if name.ends_with(".pkt") {
        return Some("sidecar_packet");
    }
    None
}

/// `build.json` `cublaslt.shapes` with dtype `bf16`: the plans plowrt selects by load-time timing
/// unless the bundle's table pins them.
fn lt_bf16_shapes(build_json: &serde_json::Value) -> usize {
    build_json
        .pointer("/cublaslt/shapes")
        .and_then(|s| s.as_array())
        .map_or(0, |s| {
            s.iter()
                .filter(|x| x.get("dtype").and_then(|d| d.as_str()) == Some("bf16"))
                .count()
        })
}

fn read_json_if_present(p: &Path) -> Result<serde_json::Value> {
    if !p.exists() {
        return Ok(serde_json::Value::Null);
    }
    let raw = std::fs::read(p).map_err(|e| bad(format!("{}: {e}", p.display())))?;
    serde_json::from_slice(&raw).map_err(|e| bad(format!("{}: not valid JSON: {e}", p.display())))
}

/// The checkpoint the bundle is joined to: an `hf:` repository the node
/// fetches, or — with a checkpoint directory — a `dist` checkpoint whose files
/// travel as blobs beside the bundle's, for a private or air-gapped model.
fn build_checkpoint(id: &Identity, dir: Option<&Path>) -> Result<BundleCheckpoint> {
    let Some(root) = dir else {
        return Ok(BundleCheckpoint {
            source: id.checkpoint_source.clone(),
            revision: id.checkpoint_revision.clone(),
            shards: id.checkpoint_shards,
            layout: id.checkpoint_layout.clone(),
            bytes: id.checkpoint_bytes,
            files: Vec::new(),
            views: Vec::new(),
        });
    };
    if !root.is_dir() {
        return Err(bad(format!("{} is not a directory", root.display())));
    }
    // The revision names the snapshot directory on the node: one path component.
    let rev = &id.checkpoint_revision;
    if rev.is_empty() || rev.contains('/') || rev == "." || rev == ".." {
        return Err(bad(
            "a carried checkpoint needs `[checkpoint] revision` usable as a directory name",
        ));
    }
    let mut files = Vec::new();
    walk_checkpoint(root, root, &mut files)?;
    files.sort_by(|a, b| a.name.cmp(&b.name));
    // `prepare` globs `*.safetensors` in the snapshot's top level only.
    let shards = files
        .iter()
        .filter(|f| !f.name.contains('/') && f.name.ends_with(".safetensors"))
        .count() as u32;
    if shards == 0 {
        return Err(bad(format!(
            "{}: no top-level *.safetensors — that is not a checkpoint",
            root.display()
        )));
    }
    if id.checkpoint_shards != 0 && id.checkpoint_shards != shards {
        return Err(bad(format!(
            "{}: {shards} top-level safetensors shards, the recipe says {}",
            root.display(),
            id.checkpoint_shards
        )));
    }
    Ok(BundleCheckpoint {
        source: plow_asset::dist::DIST_SOURCE.to_string(),
        revision: rev.clone(),
        shards,
        layout: id.checkpoint_layout.clone(),
        bytes: files.iter().map(|f| f.bytes).sum(),
        files,
        views: Vec::new(),
    })
}

fn walk_checkpoint(root: &Path, dir: &Path, out: &mut Vec<CheckpointFile>) -> Result<()> {
    for entry in std::fs::read_dir(dir).map_err(|e| bad(format!("{}: {e}", dir.display())))? {
        let p = entry
            .map_err(|e| bad(format!("{}: {e}", dir.display())))?
            .path();
        let name = file_name(&p);
        // Hidden entries are tool state (`.cache/huggingface/`), not the
        // checkpoint.
        if name.starts_with('.') {
            continue;
        }
        if p.is_dir() {
            walk_checkpoint(root, &p, out)?;
            continue;
        }
        if !p.is_file() {
            continue;
        }
        let rel = p
            .strip_prefix(root)
            .map_err(|_| bad(format!("{} is not under {}", p.display(), root.display())))?;
        out.push(CheckpointFile {
            name: rel.to_string_lossy().replace('\\', "/"),
            sha256: sha256_file(&p)?,
            bytes: size(&p)?,
        });
    }
    Ok(())
}

/// Narrow-rung decode overrides, each pinned by its manifest DIGEST.
///
/// Not just an id: the objects inside a rung manifest are verified against
/// digests that manifest itself names, so an unverified one could serve
/// arbitrary decode objects — and the rungs are the decode hot path.
fn build_lowrung(specs: &[(u32, PathBuf)]) -> Result<Vec<LowRung>> {
    let mut out: Vec<LowRung> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for (max, path) in specs {
        if *max == 0 || !seen.insert(*max) {
            return Err(bad(format!("bad or duplicate rung width {max}")));
        }
        let raw = std::fs::read(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        let rung: ObjSet =
            serde_json::from_slice(&raw).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        rung.validate()
            .map_err(|e| bad(format!("{}: {e}", path.display())))?;
        // The digest covers the manifest as written, every field included:
        // the typed parse drops fields this plowrt does not know, and a hash
        // of that would not match the bytes the registry serves.
        let whole: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        out.push(LowRung {
            max: *max,
            objset_id: rung.objset_id.clone(),
            sha256: sha256_bytes(&canonical_json(&whole)?),
        });
    }
    out.sort_by_key(|r| r.max);
    Ok(out)
}

// --- the one entry point -----------------------------------------------------

/// Both manifests for a built variant, as the bytes that go in the archive.
#[derive(Debug)]
pub struct Generated {
    pub bundle: Bundle,
    pub bundle_bytes: Vec<u8>,
    pub objset: Option<(ObjSet, Vec<u8>)>,
    /// The directory the objset's object names are relative to.
    pub objects_root: Option<PathBuf>,
}

/// Generate `bundle.json` and `objset.json` for a built variant.
///
/// `objects` is where the device objects live. Omitted, it IS `assets`: a
/// recipe with no `[objects]` step — every CUDA recipe today — emits its cubins
/// straight in beside `model.pkt`, which is what makes this one command. A
/// recipe with a build script writes them elsewhere and must say where.
pub fn generate(
    recipe: &Path,
    assets: &Path,
    objects: Option<&Path>,
    over: Overrides<'_>,
    args: &BundleArgs<'_>,
) -> Result<Generated> {
    let id = read_recipe(recipe, over)?;
    let objset = match (runtime_compiled(&id.target.vendor), objects) {
        (true, Some(d)) => {
            return Err(bad(format!(
                "{}: vendor {} compiles its kernels at run time; it has no object directory",
                d.display(),
                id.target.vendor
            )))
        }
        (true, None) => None,
        (false, _) if id.host => {
            let objects = sorted_files(assets, OBJSET_SUFFIXES)?;
            if let Some(o) = objects.first() {
                return Err(bad(format!(
                    "{}: a host-executed packet ships no device object",
                    o.display()
                )));
            }
            None
        }
        // NVIDIA serves its objects from the asset directory, so that is what ships.
        (false, named) if id.target.vendor == "nvidia" => {
            let mut objects = served_objects(assets)?;
            if let Some(named) = named {
                check_named_objects(assets, named, &objects)?;
            }
            for packet in sorted_files(assets, &[".pkt"])? {
                check_declared(&packet, &objects)?;
            }
            // A sub-bundle serves its own objects from beside its packet; appended, so a
            // directory without one keeps the objset id it always had.
            for sub in sub_bundles(assets)? {
                let dir = assets.join(&sub);
                let own = served_objects(&dir)?;
                for packet in sorted_files(&dir, &[".pkt"])? {
                    check_declared(&packet, &own)?;
                }
                let build: serde_json::Value = read_json_if_present(&dir.join("build.json"))?;
                let pairing = build
                    .get("pairing")
                    .and_then(|p| p.get("hash"))
                    .and_then(|h| h.as_str());
                check_stamps(&dir, pairing, own.iter())?;
                objects.extend(own.into_iter().map(|o| ObjSetObject {
                    name: format!("{sub}/{}", o.name),
                    ..o
                }));
            }
            let defines_dir = named
                .filter(|d| d.join("build_defines.json").exists())
                .unwrap_or(assets);
            Some((
                build_objset(&id, defines_dir, objects)?,
                assets.to_path_buf(),
            ))
        }
        (false, dir) => {
            let dir = dir.unwrap_or(assets);
            let objects = discover_objects(dir, &Readelf)?;
            Some((build_objset(&id, dir, objects)?, dir.to_path_buf()))
        }
    };
    let (objset, objects_root) = match objset {
        Some((o, root)) => (Some(o), Some(root)),
        None => (None, None),
    };
    let bundle = build_bundle(assets, &id, objset.as_ref(), args)?;

    let bundle_bytes = canonical_json(&bundle)?;
    no_build_host_paths(&bundle_bytes, "bundle.json")?;
    let objset = match objset {
        Some(o) => {
            let bytes = canonical_json(&o)?;
            no_build_host_paths(&bytes, "objset.json")?;
            Some((o, bytes))
        }
        None => None,
    };
    Ok(Generated {
        bundle,
        bundle_bytes,
        objset,
        objects_root,
    })
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
pub(crate) mod tests;
