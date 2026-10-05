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
    /// The packet carried `serve.json`. A legacy packet keeps every legacy fallback (built-in
    /// chat builders, `<think>` splitting); a packet that carries the section is taken at its word.
    pub from_packet: bool,
}

/// The packet's `serve.json` bytes, read from the section directory without loading the packet.
pub fn read_section(pkt: &Path) -> Result<Option<Vec<u8>>> {
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
        if name == serve_manifest::SECTION.as_bytes() && kind == packet::devbuild::SECT_METADATA {
            if found.is_some() {
                return Err(bad("two serve.json sections"));
            }
            let off = u64::from_le_bytes(e[8..16].try_into().expect("8 bytes"));
            let size = u64::from_le_bytes(e[16..24].try_into().expect("8 bytes"));
            if size > 16 << 20 || off.checked_add(size).is_none_or(|end| end > len) {
                return Err(bad("serve.json section range outside the packet"));
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
/// check the checkpoint against the packet's weight pins.
pub fn resolve(asset_dir: &Path, checkpoint_dir: &Path) -> Result<ServeInfo> {
    let pkt = crate::asset::devblob::DevBlob::find_in_dir(asset_dir)?;
    let section = match &pkt {
        Some(p) => read_section(p)?,
        None => None,
    };
    let Some(bytes) = section else {
        return Ok(ServeInfo { manifest: ServeManifest::from_checkpoint(asset_dir, checkpoint_dir), from_packet: false });
    };
    let manifest = ServeManifest::parse(&bytes).map_err(RuntimeError::Device)?;
    if !manifest.weights.is_empty() {
        serve_manifest::verify_pins(checkpoint_dir, &manifest.weights).map_err(RuntimeError::Device)?;
    }
    Ok(ServeInfo { manifest, from_packet: true })
}

/// The checkpoint a bundle serves against: `PLOW_CHECKPOINT`, else `<assets>/checkpoint`.
pub fn checkpoint_dir(asset_dir: &Path) -> PathBuf {
    crate::config::RuntimeConfig::get()
        .checkpoint
        .clone()
        .map(PathBuf::from)
        .unwrap_or_else(|| asset_dir.join("checkpoint"))
}

/// `serve.json` serve-knob defaults of the bundles in `asset_dirs`, plus the files a
/// self-contained bundle carries (`objects/` for `PLOW_PF_SEG_DIR`, `cublaslt_algos.jsonl` for
/// `PLOW_LT_ALGOS`), as `(knob, value, source)`. A knob the environment already sets is left
/// out, so an explicit setting always wins. Every packet-carried key must be a registered
/// runtime knob.
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
        let objects = dir.join("objects");
        if objects.is_dir() {
            put("PLOW_PF_SEG_DIR", objects.display().to_string(), dir.display().to_string(), &mut out)?;
        }
        let lt = dir.join("cublaslt_algos.jsonl");
        if lt.is_file() && env("PLOW_LT_ALGOS_WRITE").is_none() {
            put("PLOW_LT_ALGOS", lt.display().to_string(), dir.display().to_string(), &mut out)?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::devbuild::{Model, SectionData};

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
    fn serve_defaults_yield_to_the_environment_and_must_be_runtime_knobs() {
        let mut m = ServeManifest { version: 1, ..Default::default() };
        m.serve_defaults.insert("PLOW_PF_INTERLEAVE".into(), "2048".into());
        m.serve_defaults.insert("PLOW_MULTISTEP".into(), "0".into());
        let dir = bundle("defaults", Some(&m));
        std::fs::create_dir_all(dir.join("objects")).unwrap();
        let got = asset_env_defaults(&[dir.clone()], |k| (k == "PLOW_MULTISTEP").then(|| "4".into())).unwrap();
        let keys: Vec<_> = got.iter().map(|(k, v, _)| (k.as_str(), v.clone())).collect();
        assert!(keys.contains(&("PLOW_PF_INTERLEAVE", "2048".into())));
        assert!(keys.iter().any(|(k, _)| *k == "PLOW_PF_SEG_DIR"));
        assert!(!keys.iter().any(|(k, _)| *k == "PLOW_MULTISTEP"));

        m.serve_defaults.insert("PLOW_NOT_A_KNOB".into(), "1".into());
        let bad = bundle("defaults-bad", Some(&m));
        assert!(asset_env_defaults(&[bad.clone()], |_| None).is_err());
        std::fs::remove_dir_all(dir).unwrap();
        std::fs::remove_dir_all(bad).unwrap();
    }
}
