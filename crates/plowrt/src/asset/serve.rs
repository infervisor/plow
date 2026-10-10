//! A model's serve manifest (`plow_asset::serve_manifest`): the packet's `serve.json`, or — for a
//! packet emitted before the section existed — the same values read from the checkpoint's HF
//! files, which is what the runtime used to do at each use site.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use plow_asset::serve_manifest::{self, ServeManifest};

use crate::{Result, RuntimeError};

#[derive(Debug)]
pub struct ServeInfo {
    pub manifest: ServeManifest,
    /// The packet's `plow.multimodal.v1` contract: which media parts it serves and how.
    pub multimodal: Option<plow_asset::multimodal::MmContract>,
    /// The packet carried `serve.json`. A legacy packet keeps every legacy fallback (built-in
    /// chat builders, `<think>` splitting); a packet that carries the section is taken at its word.
    pub from_packet: bool,
}

/// The packet's `serve.json` bytes, read from the section directory without loading the packet.
pub fn read_section(pkt: &Path) -> Result<Option<Vec<u8>>> {
    read_named_section(pkt, serve_manifest::SECTION)
}

/// The packet's multimodal contract, when it carries one.
pub fn read_multimodal(pkt: &Path) -> Result<Option<plow_asset::multimodal::MmContract>> {
    let Some(bytes) = read_named_section(pkt, plow_asset::multimodal::SECTION)? else { return Ok(None) };
    let contract: plow_asset::multimodal::MmContract = serde_json::from_slice(&bytes)
        .map_err(|e| RuntimeError::Device(format!("{}: multimodal contract: {e}", pkt.display())))?;
    contract.validate().map_err(|e| RuntimeError::Device(format!("{}: {e}", pkt.display())))?;
    Ok(Some(contract))
}

/// A metadata section's bytes, read from the section directory without loading the packet.
pub fn read_named_section(pkt: &Path, section: &str) -> Result<Option<Vec<u8>>> {
    let io = |source| RuntimeError::Io { path: pkt.to_path_buf(), source };
    let bad = |what: &str| RuntimeError::Device(format!("{}: {what}", pkt.display()));
    let mut f = std::fs::File::open(pkt).map_err(io)?;
    let len = f.metadata().map_err(io)?.len();
    let mut hdr = [0u8; 64];
    f.read_exact(&mut hdr).map_err(io)?;
    let dir = u64::from_le_bytes(hdr[40..48].try_into().expect("8 bytes"));
    if dir == 0 {
        return Ok(None);
    }
    if dir.checked_add(8).is_none_or(|end| end > len) {
        return Err(bad("section directory past the end of the packet"));
    }
    f.seek(SeekFrom::Start(dir)).map_err(io)?;
    let mut head = [0u8; 8];
    f.read_exact(&mut head).map_err(io)?;
    if &head[..4] != packet::devbuild::SECT_MAGIC {
        return Err(bad("bad section directory magic"));
    }
    let n = u32::from_le_bytes(head[4..8].try_into().expect("4 bytes")) as u64;
    if n > 4096 || dir + 8 + n * 48 > len {
        return Err(bad("section directory truncated"));
    }
    let mut entries = vec![0u8; (n * 48) as usize];
    f.read_exact(&mut entries).map_err(io)?;
    let mut found = None;
    for e in entries.chunks_exact(48) {
        let name = &e[24..48];
        let name = &name[..name.iter().position(|&b| b == 0).unwrap_or(24)];
        let kind = u32::from_le_bytes(e[0..4].try_into().expect("4 bytes"));
        if name == section.as_bytes() && kind == packet::devbuild::SECT_METADATA {
            if found.is_some() {
                return Err(bad(&format!("two {section} sections")));
            }
            let off = u64::from_le_bytes(e[8..16].try_into().expect("8 bytes"));
            let size = u64::from_le_bytes(e[16..24].try_into().expect("8 bytes"));
            if size > 16 << 20 || off.checked_add(size).is_none_or(|end| end > len) {
                return Err(bad(&format!("{section} section range outside the packet")));
            }
            found = Some((off, size));
        }
    }
    let Some((off, size)) = found else { return Ok(None) };
    f.seek(SeekFrom::Start(off)).map_err(io)?;
    let mut data = vec![0u8; size as usize];
    f.read_exact(&mut data).map_err(io)?;
    Ok(Some(data))
}

/// Resolve the serve manifest of the bundle in `asset_dir` served against `checkpoint_dir`, and
/// check the checkpoint against the packet's weight pins. A mismatch is fatal and names the
/// bundle, the checkpoint and each file that differs.
pub fn resolve(asset_dir: &Path, checkpoint_dir: &Path) -> Result<ServeInfo> {
    let pkt = crate::asset::devblob::DevBlob::find_in_dir(asset_dir)?;
    let section = match &pkt {
        Some(p) => read_section(p)?,
        None => None,
    };
    let multimodal = match &pkt {
        Some(p) => read_multimodal(p)?,
        None => None,
    };
    let Some(bytes) = section else {
        return Ok(ServeInfo { manifest: ServeManifest::from_checkpoint(asset_dir, checkpoint_dir), multimodal, from_packet: false });
    };
    let manifest = ServeManifest::parse(&bytes).map_err(RuntimeError::Device)?;
    if !manifest.weights.is_empty() {
        serve_manifest::verify_pins(checkpoint_dir, &manifest.weights).map_err(|e| {
            RuntimeError::Device(format!(
                "model {}: {e} (checkpoint from {}; pass `--assets {},checkpoint=<dir>` with the \
                 checkpoint the packet was emitted against)",
                asset_dir.display(),
                checkpoint_source(asset_dir).1,
                asset_dir.display()
            ))
        })?;
        tracing::info!(model = %asset_dir.display(), checkpoint = %checkpoint_dir.display(),
            shards = manifest.weights.len(), "checkpoint matches the packet's weight pins");
    } else if checkpoint_dir.is_dir() {
        tracing::warn!(model = %asset_dir.display(), checkpoint = %checkpoint_dir.display(),
            "the packet records no weight pins: checkpoint identity is checked only by tensor byte \
             sizes at bind");
    }
    Ok(ServeInfo { manifest, multimodal, from_packet: true })
}

/// Where a bundle's checkpoint came from (see [`checkpoint_dir`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointSource {
    /// `--assets <dir>,checkpoint=<path>` (or a control-plane load naming one).
    Explicit,
    /// `<assets>/checkpoint`.
    Bundled,
    /// The process-wide `PLOW_CHECKPOINT` / `--rt-checkpoint`.
    Process,
    /// None of the above exists; `<assets>/checkpoint` is reported.
    Missing,
}

impl std::fmt::Display for CheckpointSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Explicit => "--assets ...,checkpoint=",
            Self::Bundled => "<assets>/checkpoint",
            Self::Process => "PLOW_CHECKPOINT",
            Self::Missing => "nowhere: no checkpoint=, no <assets>/checkpoint, no PLOW_CHECKPOINT",
        })
    }
}

type CheckpointMap = parking_lot::RwLock<std::collections::HashMap<PathBuf, PathBuf>>;

fn explicit_checkpoints() -> &'static CheckpointMap {
    static MAP: std::sync::OnceLock<CheckpointMap> = std::sync::OnceLock::new();
    MAP.get_or_init(Default::default)
}

fn map_key(asset_dir: &Path) -> PathBuf {
    std::fs::canonicalize(asset_dir).unwrap_or_else(|_| asset_dir.to_path_buf())
}

/// Pair the bundle in `asset_dir` with an explicit checkpoint for the rest of the process. A
/// second, different pairing for the same bundle is refused.
pub fn set_checkpoint(asset_dir: &Path, checkpoint: &Path) -> Result<()> {
    let mut map = explicit_checkpoints().write();
    let key = map_key(asset_dir);
    match map.get(&key) {
        Some(have) if map_key(have) != map_key(checkpoint) => Err(RuntimeError::Rejected(format!(
            "{}: already paired with checkpoint {}, not {}",
            asset_dir.display(),
            have.display(),
            checkpoint.display()
        ))),
        _ => {
            map.insert(key, checkpoint.to_path_buf());
            Ok(())
        }
    }
}

/// Split one `--assets` value, `<dir>[,checkpoint=<path>]`, and record the pairing. A value that
/// names an existing path is taken literally (a directory name may contain a comma).
pub fn take_asset_arg(arg: &Path) -> Result<PathBuf> {
    let (dir, checkpoint) = split_asset_arg(arg)?;
    if let Some(checkpoint) = checkpoint {
        set_checkpoint(&dir, &checkpoint)?;
    }
    Ok(dir)
}

fn split_asset_arg(arg: &Path) -> Result<(PathBuf, Option<PathBuf>)> {
    let Some(text) = arg.to_str().filter(|_| !arg.exists()) else {
        return Ok((arg.to_path_buf(), None));
    };
    let mut fields = text.split(',');
    let dir = PathBuf::from(fields.next().unwrap_or_default());
    let mut checkpoint = None;
    for field in fields {
        match field.split_once('=') {
            Some(("checkpoint", path)) if !path.is_empty() && checkpoint.is_none() => {
                checkpoint = Some(PathBuf::from(path))
            }
            _ => {
                return Err(RuntimeError::Rejected(format!(
                    "--assets {text:?}: expected DIR[,checkpoint=PATH]"
                )))
            }
        }
    }
    Ok((dir, checkpoint))
}

/// The checkpoint a bundle serves against: the explicit per-bundle pairing
/// (`--assets <dir>,checkpoint=<path>`), else `<assets>/checkpoint`, else the process-wide
/// `PLOW_CHECKPOINT`. Every consumer of a bundle's HF files (weights, tokenizer, generation and
/// processor configs, chat template) resolves through here.
pub fn checkpoint_dir(asset_dir: &Path) -> PathBuf {
    checkpoint_source(asset_dir).0
}

pub fn checkpoint_source(asset_dir: &Path) -> (PathBuf, CheckpointSource) {
    let explicit = explicit_checkpoints().read().get(&map_key(asset_dir)).cloned();
    let process = crate::config::RuntimeConfig::get().checkpoint.clone();
    checkpoint_with(explicit, asset_dir, process.as_deref())
}

fn checkpoint_with(explicit: Option<PathBuf>, asset_dir: &Path, process: Option<&str>) -> (PathBuf, CheckpointSource) {
    if let Some(path) = explicit {
        return (path, CheckpointSource::Explicit);
    }
    let bundled = asset_dir.join("checkpoint");
    if bundled.is_dir() {
        return (bundled, CheckpointSource::Bundled);
    }
    match process.filter(|p| !p.is_empty()) {
        Some(path) => (PathBuf::from(path), CheckpointSource::Process),
        None => (bundled, CheckpointSource::Missing),
    }
}

/// Log where each bundle's checkpoint comes from, and warn where the legacy process-wide
/// `PLOW_CHECKPOINT` is ambiguous: applied to several bundles, or shadowed by a bundle's own.
pub fn log_checkpoints(asset_dirs: &[PathBuf]) {
    let process = crate::config::RuntimeConfig::get().checkpoint.clone();
    let mut via_process = Vec::new();
    for dir in asset_dirs {
        let (path, source) = checkpoint_source(dir);
        tracing::info!(model = %dir.display(), checkpoint = %path.display(), %source, "bundle checkpoint");
        match source {
            CheckpointSource::Process => via_process.push(dir.display().to_string()),
            CheckpointSource::Bundled if process.as_deref().is_some_and(|p| !p.is_empty()) => tracing::warn!(
                model = %dir.display(),
                "PLOW_CHECKPOINT is ignored for this bundle: its own checkpoint/ wins (pass \
                 `--assets DIR,checkpoint=PATH` to pair it explicitly)"
            ),
            _ => {}
        }
    }
    if via_process.len() > 1 {
        tracing::warn!(models = ?via_process,
            "the process-wide PLOW_CHECKPOINT serves more than one bundle; pair each with \
             `--assets DIR,checkpoint=PATH`");
    }
}

/// The object directory of one bundle: an explicit `PLOW_PF_SEG_DIR` / `--pf-seg-dir` (which
/// applies to every bundle of the process), else the bundle's own `objects/`. Never derived
/// from another bundle, so co-served bundles each pair with their own specialised objects.
pub fn objects_dir(assets_dir: &Path) -> Option<PathBuf> {
    objects_dir_with(crate::config::RuntimeConfig::get().nv.pf_seg_dir.as_deref(), assets_dir)
}

fn objects_dir_with(explicit: Option<&str>, assets_dir: &Path) -> Option<PathBuf> {
    explicit.filter(|dir| !dir.is_empty()).map(PathBuf::from).or_else(|| {
        let own = assets_dir.join("objects");
        own.is_dir().then_some(own)
    })
}

/// `serve.json` serve-knob defaults of the bundles in `asset_dirs`, plus the cuBLASLt table a
/// self-contained bundle carries (`cublaslt_algos.jsonl` for `PLOW_LT_ALGOS`), as
/// `(knob, value, source)`. A bundle's `objects/` is not exported: each bundle resolves it
/// itself ([`objects_dir`]). A knob the environment already sets is left out, so an explicit
/// setting always wins. Every packet-carried key must be a registered runtime knob.
pub fn asset_env_defaults(
    asset_dirs: &[PathBuf],
    env: impl Fn(&str) -> Option<String>,
) -> Result<Vec<(String, String, String)>> {
    let mut out: Vec<(String, String, String)> = Vec::new();
    let mut put = |key: &str, value: String, source: String, out: &mut Vec<(String, String, String)>| -> Result<()> {
        if env(key).is_some() {
            return Ok(());
        }
        if let Some((_, have, from)) = out.iter().find(|(k, _, _)| k == key) {
            if *have != value {
                return Err(RuntimeError::Rejected(format!(
                    "{key}: {from} asks for {have} and {source} for {value}; set {key} explicitly to serve both"
                )));
            }
            return Ok(());
        }
        out.push((key.to_string(), value, source));
        Ok(())
    };
    for dir in asset_dirs {
        if let Some(pkt) = crate::asset::devblob::DevBlob::find_in_dir(dir)? {
            if let Some(bytes) = read_section(&pkt)? {
                let m = ServeManifest::parse(&bytes).map_err(RuntimeError::Device)?;
                for (key, value) in &m.serve_defaults {
                    if !crate::knob_spec::is_runtime_env(key) {
                        return Err(RuntimeError::Rejected(format!(
                            "{}: serve default {key} is not a registered plowrt runtime knob",
                            pkt.display()
                        )));
                    }
                    put(key, value.clone(), pkt.display().to_string(), &mut out)?;
                }
            }
        }
        let lt = dir.join("cublaslt_algos.jsonl");
        if lt.is_file() && env("PLOW_LT_ALGOS_WRITE").is_none() {
            put("PLOW_LT_ALGOS", lt.display().to_string(), dir.display().to_string(), &mut out)?;
        }
    }
    Ok(out)
}

/// The runtime contract this plowrt implements. A bundle whose `build.json`
/// `runtime_requires.plowrt_contract` is higher needs a newer plowrt.
pub const RUNTIME_CONTRACT: u32 = plow_asset::RUNTIME_CONTRACT;

/// The cuBLASLt release grouped matmul (`cublasLtGroupedMatrixLayoutCreate`) needs on Hopper.
const GROUPED_LT: usize = 130400;

/// What a bundle needs from the runtime, read from its emit-time `build.json`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RuntimeRequires {
    /// Minimum `cublasLtGetVersion` (13.4.0 = 130400) and why.
    pub cublaslt: Option<(usize, String)>,
    /// Minimum [`RUNTIME_CONTRACT`].
    pub plowrt_contract: u32,
}

/// `build.json` `runtime_requires` (`{"cublaslt": "13.4", "plowrt_contract": 1}`) when the
/// bundle declares it; otherwise derived from the emit knobs the packet was built with: a grouped
/// cuBLASLt MoE route (`emit.moe_pf_lt` / `emit.moe_dec_lt`) needs cuBLASLt 13.4. A bundle without
/// `build.json` requires nothing.
pub fn runtime_requires(asset_dir: &Path) -> Result<RuntimeRequires> {
    let path = asset_dir.join("build.json");
    let Ok(bytes) = std::fs::read(&path) else { return Ok(RuntimeRequires::default()) };
    let build: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|e| RuntimeError::Rejected(format!("{}: {e}", path.display())))?;
    runtime_requires_from(&build).map_err(|e| RuntimeError::Rejected(format!("{}: {e}", path.display())))
}

fn runtime_requires_from(build: &serde_json::Value) -> std::result::Result<RuntimeRequires, String> {
    let mut out = RuntimeRequires::default();
    if let Some(declared) = build.get("runtime_requires") {
        if let Some(v) = declared.get("cublaslt") {
            let text = v.as_str().ok_or("runtime_requires.cublaslt must be a version string")?;
            out.cublaslt = Some((parse_version(text)?, "declared in build.json".into()));
        }
        if let Some(v) = declared.get("plowrt_contract") {
            out.plowrt_contract = v
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("runtime_requires.plowrt_contract must be an integer")?;
        }
        return Ok(out);
    }
    let knob = |k: &str| build.pointer(&format!("/knobs/values/{}", k.replace('/', "~1"))) == Some(&true.into());
    let grouped: Vec<&str> = ["emit.moe_pf_lt", "emit.moe_dec_lt"].into_iter().filter(|k| knob(k)).collect();
    if !grouped.is_empty() {
        out.cublaslt = Some((GROUPED_LT, format!("grouped cuBLASLt MoE route ({})", grouped.join(", "))));
    }
    Ok(out)
}

/// `13`, `13.4` or `13.4.2` as `cublasLtGetVersion` encodes it.
fn parse_version(text: &str) -> std::result::Result<usize, String> {
    let parts: Vec<usize> = text
        .trim()
        .split('.')
        .map(|p| p.parse().map_err(|_| format!("bad cuBLASLt version {text:?}")))
        .collect::<std::result::Result<_, _>>()?;
    match parts[..] {
        [major] => Ok(major * 10000),
        [major, minor] if minor < 100 => Ok(major * 10000 + minor * 100),
        [major, minor, patch] if minor < 100 && patch < 100 => Ok(major * 10000 + minor * 100 + patch),
        _ => Err(format!("bad cuBLASLt version {text:?}")),
    }
}

fn show_version(v: usize) -> String {
    format!("{}.{}.{}", v / 10000, v / 100 % 100, v % 100)
}

/// Fail closed when this runtime cannot serve the bundle: an older runtime contract than it
/// declares, or a cuBLASLt (`lt`: the library this process binds, or why none loads) older than
/// it needs.
pub fn check_runtime(
    asset_dir: &Path,
    req: &RuntimeRequires,
    lt: &std::result::Result<(PathBuf, usize), String>,
) -> Result<()> {
    if req.plowrt_contract > RUNTIME_CONTRACT {
        return Err(RuntimeError::Rejected(format!(
            "model {} needs plowrt runtime contract {}; this plowrt implements {RUNTIME_CONTRACT}. \
             Deploy the plowrt this bundle was released with (or newer).",
            asset_dir.display(),
            req.plowrt_contract
        )));
    }
    let Some((need, why)) = &req.cublaslt else { return Ok(()) };
    let have = match lt {
        Ok((path, v)) if v >= need => {
            tracing::info!(model = %asset_dir.display(), cublaslt = %path.display(), version = %show_version(*v),
                need = %show_version(*need), "runtime requirement met");
            return Ok(());
        }
        Ok((path, v)) => format!("{} ({})", path.display(), show_version(*v)),
        Err(e) => e.clone(),
    };
    Err(RuntimeError::Rejected(format!(
        "model {} needs cuBLASLt >= {need} ({why}); this runtime loads {have}. Ship a libcublasLt \
         of cuBLAS {need} or newer beside plowrt.",
        asset_dir.display(),
        need = show_version(*need)
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::{Model, SectionData};

    #[test]
    fn asset_args_pair_a_bundle_with_its_checkpoint() {
        let dir = std::env::temp_dir().join(format!("plowrt-assetarg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let arg = format!("{},checkpoint=/hf/model@abc", dir.display());
        assert_eq!(split_asset_arg(Path::new(&arg)).unwrap(), (dir.clone(), Some(PathBuf::from("/hf/model@abc"))));
        assert_eq!(split_asset_arg(&dir).unwrap(), (dir.clone(), None));
        for bad in [",checkpoint=", ",ckpt=/x", ",checkpoint=/a,checkpoint=/b", ","] {
            assert!(split_asset_arg(Path::new(&format!("{}{bad}", dir.display()))).is_err(), "{bad}");
        }
        // An existing path is literal even with a comma in its name.
        let comma = dir.join("a,checkpoint=b");
        std::fs::create_dir_all(&comma).unwrap();
        assert_eq!(split_asset_arg(&comma).unwrap(), (comma.clone(), None));

        assert_eq!(take_asset_arg(Path::new(&arg)).unwrap(), dir);
        assert_eq!(checkpoint_source(&dir), (PathBuf::from("/hf/model@abc"), CheckpointSource::Explicit));
        assert!(set_checkpoint(&dir, Path::new("/hf/model@abc")).is_ok());
        assert!(set_checkpoint(&dir, Path::new("/hf/other")).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn checkpoint_precedence_is_explicit_then_bundled_then_process() {
        let dir = std::env::temp_dir().join(format!("plowrt-ckptorder-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let explicit = Some(PathBuf::from("/hf/x"));
        assert_eq!(checkpoint_with(None, &dir, None), (dir.join("checkpoint"), CheckpointSource::Missing));
        assert_eq!(checkpoint_with(None, &dir, Some("/p")), (PathBuf::from("/p"), CheckpointSource::Process));
        std::fs::create_dir_all(dir.join("checkpoint")).unwrap();
        assert_eq!(checkpoint_with(None, &dir, Some("/p")), (dir.join("checkpoint"), CheckpointSource::Bundled));
        assert_eq!(checkpoint_with(explicit, &dir, Some("/p")), (PathBuf::from("/hf/x"), CheckpointSource::Explicit));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn runtime_requirements_come_from_build_json() {
        let none = runtime_requires_from(&serde_json::json!({"knobs": {"values": {"emit.moe_pf_lt": false}}})).unwrap();
        assert_eq!(none, RuntimeRequires::default());
        let moe = runtime_requires_from(&serde_json::json!({"knobs": {"values": {"emit.moe_pf_lt": true}}})).unwrap();
        assert_eq!(moe.cublaslt.as_ref().map(|c| c.0), Some(130400));
        let declared = serde_json::json!({"runtime_requires": {"cublaslt": "13.4.2", "plowrt_contract": 1},
            "knobs": {"values": {"emit.moe_pf_lt": true}}});
        let declared = runtime_requires_from(&declared).unwrap();
        assert_eq!((declared.cublaslt.map(|c| c.0), declared.plowrt_contract), (Some(130402), 1));
        assert!(runtime_requires_from(&serde_json::json!({"runtime_requires": {"cublaslt": 13}})).is_err());
        assert!(runtime_requires_from(&serde_json::json!({"runtime_requires": {"cublaslt": "13.x"}})).is_err());
        assert_eq!(parse_version("12").unwrap(), 120000);
    }

    #[test]
    fn runtime_check_fails_closed() {
        let dir = Path::new("/m/gemma-26b");
        let req = RuntimeRequires { cublaslt: Some((130400, "grouped".into())), plowrt_contract: 1 };
        assert!(check_runtime(dir, &req, &Ok((PathBuf::from("/rt/libcublasLt.so.13"), 130402))).is_ok());
        let old = check_runtime(dir, &req, &Ok((PathBuf::from("/rt/libcublasLt.so.12"), 120901))).unwrap_err().to_string();
        assert!(old.contains("/m/gemma-26b") && old.contains("13.4.0") && old.contains("12.9.1"), "{old}");
        assert!(check_runtime(dir, &req, &Err("no loadable cuBLASLt".into())).is_err());
        assert!(check_runtime(dir, &RuntimeRequires::default(), &Err("none".into())).is_ok());
        let newer = RuntimeRequires { plowrt_contract: RUNTIME_CONTRACT + 1, ..Default::default() };
        assert!(check_runtime(dir, &newer, &Ok((PathBuf::new(), 130402))).is_err());
    }

    fn bundle(name: &str, serve: Option<&ServeManifest>) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("plowrt-serve-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut sections = vec![SectionData { kind: packet::devbuild::SECT_METADATA, name: "other".into(), data: b"{}".to_vec() }];
        if let Some(m) = serve {
            sections.push(SectionData {
                kind: packet::devbuild::SECT_METADATA,
                name: serve_manifest::SECTION.into(),
                data: m.to_json(),
            });
        }
        let model = Model {
            n_cu: 1,
            target: 0,
            tensors: Vec::new(),
            progs: Vec::new(),
            kv_row_insts: Vec::new(),
            prog_t: Vec::new(),
            gen: Vec::new(),
        };
        std::fs::write(dir.join("model.pkt"), model.to_blob_v6(&sections)).unwrap();
        dir
    }

    #[test]
    fn packet_section_wins_and_a_legacy_packet_reads_the_checkpoint() {
        let m = ServeManifest { version: 1, stop_token_ids: vec![7], ..Default::default() };
        let dir = bundle("section", Some(&m));
        let info = resolve(&dir, &dir.join("checkpoint")).unwrap();
        assert!(info.from_packet);
        assert_eq!(info.manifest.stop_token_ids, [7]);

        let legacy = bundle("legacy", None);
        std::fs::create_dir_all(legacy.join("checkpoint")).unwrap();
        std::fs::write(legacy.join("checkpoint/generation_config.json"), r#"{"eos_token_id":[3,2]}"#).unwrap();
        let info = resolve(&legacy, &legacy.join("checkpoint")).unwrap();
        assert!(!info.from_packet);
        assert_eq!(info.manifest.stop_token_ids, [2, 3]);
        std::fs::remove_dir_all(dir).unwrap();
        std::fs::remove_dir_all(legacy).unwrap();
    }

    #[test]
    fn weight_pins_are_checked_at_resolve() {
        let pin = plow_asset::serve_manifest::WeightPin { file: "w.safetensors".into(), bytes: 1, header_sha256: "x".into() };
        let m = ServeManifest { version: 1, weights: vec![pin], ..Default::default() };
        let dir = bundle("pins", Some(&m));
        let err = resolve(&dir, &dir).unwrap_err().to_string();
        assert!(err.contains("does not match the packet"), "{err}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn each_bundle_resolves_its_own_objects_dir() {
        let with = bundle("objects-with", None);
        std::fs::create_dir_all(with.join("objects")).unwrap();
        let without = bundle("objects-without", None);
        assert_eq!(objects_dir_with(None, &with), Some(with.join("objects")));
        assert_eq!(objects_dir_with(None, &without), None);
        assert_eq!(objects_dir_with(Some(""), &without), None);
        // An explicit directory overrides every bundle's own.
        assert_eq!(objects_dir_with(Some("/explicit"), &with), Some(PathBuf::from("/explicit")));
        assert_eq!(objects_dir_with(Some("/explicit"), &without), Some(PathBuf::from("/explicit")));
        // Co-serving does not leak one bundle's objects/ into the process environment.
        let got = asset_env_defaults(&[with.clone(), without.clone()], |_| None).unwrap();
        assert!(!got.iter().any(|(k, _, _)| k == "PLOW_PF_SEG_DIR"));
        std::fs::remove_dir_all(with).unwrap();
        std::fs::remove_dir_all(without).unwrap();
    }

    #[test]
    fn serve_defaults_yield_to_the_environment_and_must_be_runtime_knobs() {
        let mut m = ServeManifest { version: 1, ..Default::default() };
        m.serve_defaults.insert("PLOW_PF_INTERLEAVE".into(), "2048".into());
        m.serve_defaults.insert("PLOW_MULTISTEP".into(), "0".into());
        let dir = bundle("defaults", Some(&m));
        std::fs::create_dir_all(dir.join("objects")).unwrap();
        let got = asset_env_defaults(&[dir.clone()], |k| (k == "PLOW_MULTISTEP").then(|| "4".into())).unwrap();
        let keys: Vec<_> = got.iter().map(|(k, v, _)| (k.as_str(), v.clone())).collect();
        assert!(keys.contains(&("PLOW_PF_INTERLEAVE", "2048".into())));
        assert!(!keys.iter().any(|(k, _)| *k == "PLOW_PF_SEG_DIR"));
        assert!(!keys.iter().any(|(k, _)| *k == "PLOW_MULTISTEP"));

        m.serve_defaults.insert("PLOW_NOT_A_KNOB".into(), "1".into());
        let bad = bundle("defaults-bad", Some(&m));
        assert!(asset_env_defaults(&[bad.clone()], |_| None).is_err());
        std::fs::remove_dir_all(dir).unwrap();
        std::fs::remove_dir_all(bad).unwrap();
    }
}
