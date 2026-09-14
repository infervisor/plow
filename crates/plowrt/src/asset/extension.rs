//! Finding extensions beside a serving directory and turning them into the facts the contract
//! checks — docs/arch/19, phases 2 and 3.
//!
//! The RULES live in `plow_asset::extension`, deliberately: phase 4's `plowc extend` applies
//! the same six to refuse an unusable extension at emit rather than at load. This module is
//! only the part that needs a filesystem and a parsed container.
//!
//! Nothing here runs for a serving directory with no extensions. [`discover`] returns an empty
//! list, [`load`] does no I/O, and `merge` with no extensions is the identity on the parent's
//! ladder — so a packet that ships today behaves exactly as it does today.

use std::path::{Path, PathBuf};

use plow_asset::extension::{
    self, Arenas, Budget, ConfigAxes, ExtensionFacts, KvRing, ParentFacts, Refusal, Requires,
    BLOB_AXIS_HIDDEN, BLOB_AXIS_L2_DOMAINS, BLOB_AXIS_L2_SMS, BLOB_AXIS_N_CU, BLOB_AXIS_TARGET,
    BLOB_AXIS_TP_DEGREE, BLOB_AXIS_TP_SLOT_BYTES,
};

use crate::asset::devblob::DevBlob;
use crate::{Result, RuntimeError};

/// Extension directories are searched for as `<assets>.ext/*`, a SIBLING of the serving
/// directory rather than a subdirectory of it: `DevBlob::find_in_dir` walks the assets dir and
/// reports two containers as ambiguous, and an extension is not the model.
pub const EXT_DIR_SUFFIX: &str = ".ext";

/// Colon-separated extension directories, overriding discovery. Naming them explicitly is how
/// a bisect serves the parent alone, or one extension of several.
pub const EXT_ENV: &str = "PLOW_EXTENSIONS";

/// Extension directories to merge, in a deterministic order.
///
/// `PLOW_EXTENSIONS` wins when set (including when set to empty, which means "serve the parent
/// alone"). Otherwise every subdirectory of `<assets>.ext` that holds an `extension.pkt`,
/// sorted by name so two loads of the same directory merge in the same order and refuse the
/// same extension first.
pub fn discover(assets: &Path) -> Result<Vec<PathBuf>> {
    match std::env::var(EXT_ENV) {
        Ok(v) => Ok(explicit(&v)),
        Err(_) => discover_in(assets),
    }
}

/// Parse a `PLOW_EXTENSIONS` value. Empty means "serve the parent alone", which is how a
/// bisect turns the mechanism off without moving a directory.
pub fn explicit(list: &str) -> Vec<PathBuf> {
    list.split(':')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect()
}

/// Discovery proper, with no environment read — see [`discover`].
pub fn discover_in(assets: &Path) -> Result<Vec<PathBuf>> {
    let mut root = assets.as_os_str().to_os_string();
    root.push(EXT_DIR_SUFFIX);
    let root = PathBuf::from(root);
    if !root.is_dir() {
        return Ok(Vec::new());
    }
    let entries = std::fs::read_dir(&root).map_err(|source| RuntimeError::Io {
        path: root.clone(),
        source,
    })?;
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join(extension::EXTENSION_FILE).is_file())
        .collect();
    found.sort();
    Ok(found)
}

/// One extension read off disk: its container image, the parsed blob, and its `requires.json`.
pub struct LoadedExtension {
    pub dir: PathBuf,
    pub image: Vec<u8>,
    pub blob: DevBlob,
    pub requires: Requires,
    pub facts: ExtensionFacts,
}

fn refuse(dir: &Path, rule: extension::Rule, detail: impl Into<String>) -> RuntimeError {
    RuntimeError::Rejected(
        Refusal {
            extension: dir.display().to_string(),
            rule,
            detail: detail.into(),
        }
        .to_string(),
    )
}

fn read_config(dir: &Path) -> Result<String> {
    let path = dir.join(extension::CONFIG_FILE);
    std::fs::read_to_string(&path).map_err(|source| RuntimeError::Io { path, source })
}

/// The shared axes the CONTAINER states, as opposed to the ones `plow_config.h` does.
///
/// These are facts about state the runtime allocates once and every program then addresses:
/// how many CUs the stream is windowed for, the TP degree and the peer region it implies, the
/// L2 placement the interpreter dispatches by, and the GPU the packet was compiled for. An
/// extension that disagrees on any of them describes a different machine.
fn blob_axes(blob: &DevBlob, axes: &mut ConfigAxes) {
    axes.set(BLOB_AXIS_N_CU, blob.n_cu as i64)
        .set(BLOB_AXIS_TARGET, blob.target as i64)
        .set(BLOB_AXIS_TP_DEGREE, blob.tp_degree() as i64)
        .set(BLOB_AXIS_HIDDEN, blob.tp.map_or(0, |t| t.hidden) as i64)
        .set(
            BLOB_AXIS_TP_SLOT_BYTES,
            blob.tp.map_or(0, |t| t.slot_bytes) as i64,
        )
        .set(
            BLOB_AXIS_L2_DOMAINS,
            blob.progs.iter().map(|p| p.l2_domains).max().unwrap_or(0) as i64,
        )
        // The header's SMs-per-partition, which the placed interpreter reads; carried beside
        // the domain count because a packet placed for a different partition size dispatches
        // its queues to the wrong physical SMs.
        .set(BLOB_AXIS_L2_SMS, 0);
}

/// Read the parent's `build.json`, if it has one.
///
/// Absent is not an error for a parent loaded without extensions: assets shipped before the
/// manifest existed have none, and the loader has always accepted them. It IS an error the
/// moment an extension is present, because `build.json` is where the parent states the facts
/// rules 4 and 6 need, and guessing past a missing one is how a check stops checking.
fn parent_manifest(assets: &Path) -> Result<Option<serde_json::Value>> {
    let path = assets.join("build.json");
    match std::fs::read(&path) {
        Ok(raw) => serde_json::from_slice(&raw)
            .map(Some)
            .map_err(|source| RuntimeError::Json { path, source }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(source) => Err(RuntimeError::Io { path, source }),
    }
}

fn pairing_hash(manifest: Option<&serde_json::Value>) -> Option<u64> {
    manifest
        .and_then(|m| m.pointer("/pairing/hash"))
        .and_then(|v| v.as_str())
        .and_then(extension::parse_pairing_hash)
}

/// The KV geometry rule 4 needs.
///
/// `shapes.kv_window` / `shapes.kv_ring_rows` are what phase 4 must make `plowc` record.
/// `shapes.kv_window == 0` is an all-global model and therefore full-causal; a packet whose
/// manifest states neither is [`KvRing::Unstated`], which refuses only a WIDER bucket.
fn kv_ring(manifest: Option<&serde_json::Value>) -> KvRing {
    let get = |k: &str| {
        manifest
            .and_then(|m| m.pointer(&format!("/shapes/{k}")))
            .and_then(serde_json::Value::as_u64)
    };
    match (get("kv_window"), get("kv_ring_rows")) {
        (Some(0), _) => KvRing::FullCausal,
        (Some(window), Some(ring_rows)) => KvRing::Windowed {
            window: window as u32,
            ring_rows: ring_rows as u32,
        },
        _ => KvRing::Unstated,
    }
}

/// Every program's scratch-arena high-water mark, from `weights.json`.
fn workspace_bytes(assets: &Path) -> u64 {
    crate::asset::read_json::<plow_asset::Manifest>(&assets.join("weights.json"))
        .map(|m| m.buckets.iter().map(|b| b.arena_bytes).max().unwrap_or(0))
        .unwrap_or(0)
}

/// The facts the contract needs about the loaded `model.pkt`.
pub fn parent_facts(assets: &Path, image: &[u8], blob: &DevBlob) -> Result<ParentFacts> {
    let manifest = parent_manifest(assets)?;
    let mut config = ConfigAxes::from_config_header(&read_config(assets)?);
    blob_axes(blob, &mut config);

    let decode_batch = blob
        .decode_rungs()
        .into_iter()
        .max()
        .unwrap_or(packet::devbuild::DECODE_RUNG_MAX);
    let reserved = Budget {
        workspace_bytes: workspace_bytes(assets),
        ..blob.budget()
    };
    Ok(ParentFacts {
        hash: extension::packet_hash(image),
        tensor_count: blob.tensors.len() as u32,
        tensor_digest: blob.tensor_table_digest(),
        config,
        kv: kv_ring(manifest.as_ref()),
        programs: blob.program_roles(),
        decode_batch,
        arenas: Arenas {
            reserved,
            // Mirrors the ceiling in `DevProg::seg_classes_with`: the segment class table is
            // not a pool and does not grow.
            segment_ceiling: 2048,
            // Instruction streams, counter banks and scratch all come out of the VMM pools the
            // engine allocates per program at load, so a wider program simply allocates more.
            growable: true,
        },
        pairing_hash: pairing_hash(manifest.as_ref()),
    })
}

/// Read one extension directory. Refuses a malformed container or `requires.json` here, before
/// the contract sees it: those are not rule violations, they are not an extension at all.
pub fn load_one(dir: &Path, l2_dispatch_ok: bool) -> Result<LoadedExtension> {
    let pkt = dir.join(extension::EXTENSION_FILE);
    let image = std::fs::read(&pkt).map_err(|source| RuntimeError::Io {
        path: pkt.clone(),
        source,
    })?;
    let blob = DevBlob::parse_extension(&image, l2_dispatch_ok)?;
    let parent = blob.parent.clone().ok_or_else(|| {
        RuntimeError::Packet {
            path: pkt.clone(),
            reason: "extension container carries no parent reference".into(),
        }
    })?;

    let requires: Requires = crate::asset::read_json(&dir.join(extension::REQUIRES_FILE))?;
    requires
        .validate()
        .map_err(|e| refuse(dir, extension::Rule::ConfigCompatibility, e))?;

    let config_src = read_config(dir)?;
    let mut config = ConfigAxes::from_config_header(&config_src);
    blob_axes(&blob, &mut config);

    // Phase 3: the extension's `requires.json` must pin the EXTENSION's own packet hash. If it
    // named the parent's, an object built for the parent would pair with this bucket and
    // dispatch arms it does not have — which on AMD writes nothing rather than trapping.
    let own = config_pairing_hash(&config_src).ok_or_else(|| {
        refuse(
            dir,
            extension::Rule::ConfigCompatibility,
            "plow_config.h carries no PLOW_PACKET_HASH — not a plowc-written header",
        )
    })?;
    let declared = extension::parse_pairing_hash(&requires.pairing_hash).unwrap_or(0);
    if declared != own {
        return Err(refuse(
            dir,
            extension::Rule::ConfigCompatibility,
            format!(
                "requires.json pins packet 0x{declared:016x} but this extension's \
                 plow_config.h is 0x{own:016x} — the objects would pair with a different \
                 artifact"
            ),
        ));
    }

    // The chunk a new bucket runs at IS its width: a bucket is the largest chunk the runtime
    // submits to it. Rule 4 sizes the ring from this.
    let chunk = blob
        .program_roles()
        .iter()
        .filter_map(|r| match r {
            extension::ProgramRole::PrefillBucket { rows } => Some(*rows),
            _ => None,
        })
        .max()
        .unwrap_or(0);

    let facts = ExtensionFacts {
        name: dir.display().to_string(),
        parent: parent.clone(),
        config,
        programs: blob.program_roles(),
        chunk,
        budget: blob.budget(),
    };
    Ok(LoadedExtension {
        dir: dir.to_path_buf(),
        image,
        blob,
        requires,
        facts,
    })
}

/// `PLOW_PACKET_HASH` out of a `plow_config.h`. The same one line
/// `scripts/build_gfx942.sh`'s `cfg_get` reads.
pub fn config_pairing_hash(src: &str) -> Option<u64> {
    src.lines()
        .filter_map(|l| l.trim_start().strip_prefix("#define PLOW_PACKET_HASH "))
        .find_map(extension::parse_pairing_hash)
}

/// Load the parent's extensions and merge them, refusing by name and rule.
///
/// Returns the merged ladder and the loaded extensions, in merge order. With no extensions the
/// merge is the identity and nothing is read from disk beyond the parent's own sidecars.
pub fn load(
    assets: &Path,
    image: &[u8],
    blob: &DevBlob,
    l2_dispatch_ok: bool,
) -> Result<(extension::Merged, Vec<LoadedExtension>)> {
    let dirs = discover(assets)?;
    let parent = parent_facts(assets, image, blob)?;
    if dirs.is_empty() {
        let merged = extension::merge(&parent, &[]).map_err(|r| RuntimeError::Rejected(r.to_string()))?;
        return Ok((merged, Vec::new()));
    }
    let loaded: Vec<LoadedExtension> = dirs
        .iter()
        .map(|d| load_one(d, l2_dispatch_ok))
        .collect::<Result<_>>()?;
    let facts: Vec<ExtensionFacts> = loaded.iter().map(|l| l.facts.clone()).collect();
    let merged =
        extension::merge(&parent, &facts).map_err(|r| RuntimeError::Rejected(r.to_string()))?;
    for line in &merged.growth.grew {
        tracing::info!("packet extensions: {line}");
    }
    tracing::info!(
        extensions = loaded.len(),
        programs = merged.programs.len(),
        prefill = ?merged.prefill_widths(),
        decode = ?merged.decode_rungs(),
        "merged packet extensions"
    );
    Ok((merged, loaded))
}

#[cfg(test)]
#[path = "extension_tests.rs"]
mod tests;
