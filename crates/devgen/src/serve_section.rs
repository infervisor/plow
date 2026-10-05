//! The packet's `serve.json` (`plow_asset::serve_manifest`): the checkpoint's chat template, stop
//! set, sampling defaults and KV geometry, the recipe's serve-knob defaults and the shard pins, so
//! `plowrt serve` needs nothing from the checkpoint's HF files at serve time.

use std::path::Path;

use packet::devbuild::{SectionData, SECT_METADATA};
use plow_asset::serve_manifest::{self, ServeManifest};

pub(crate) fn manifest(dir: &Path, serve_defaults: Option<&str>) -> Result<ServeManifest, String> {
    let mut m = ServeManifest::from_checkpoint(dir, dir);
    // The template's file name, not the emitting host's path.
    if let Some(chat) = m.chat.as_mut() {
        chat.source = chat.source.as_deref().and_then(|s| Path::new(s).file_name()).map(|n| n.to_string_lossy().into_owned());
    }
    for item in serve_defaults.unwrap_or("").split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let (key, value) = item
            .split_once('=')
            .filter(|(k, _)| k.starts_with("PLOW_"))
            .ok_or_else(|| format!("PLOW_EMIT_SERVE_DEFAULTS: `{item}` is not PLOW_KEY=VALUE"))?;
        m.serve_defaults.insert(key.trim().to_string(), value.trim().to_string());
    }
    m.weights = serve_manifest::pin_checkpoint(dir)?;
    Ok(m)
}

pub(crate) fn section(dir: &Path, serve_defaults: Option<&str>) -> Result<SectionData, String> {
    Ok(SectionData {
        kind: SECT_METADATA,
        name: serve_manifest::SECTION.into(),
        data: manifest(dir, serve_defaults)?.to_json(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carries_serve_defaults_and_refuses_malformed_ones() {
        let d = std::env::temp_dir().join(format!("devgen-serve-section-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("generation_config.json"), r#"{"eos_token_id":1}"#).unwrap();
        let m = manifest(&d, Some("PLOW_PF_INTERLEAVE=2048, PLOW_MULTISTEP=0")).unwrap();
        assert_eq!(m.serve_defaults["PLOW_PF_INTERLEAVE"], "2048");
        assert_eq!(m.serve_defaults["PLOW_MULTISTEP"], "0");
        assert_eq!(m.stop_token_ids, [1]);
        assert!(m.weights.is_empty());
        assert!(manifest(&d, Some("PF_INTERLEAVE=1")).is_err());
        assert!(manifest(&d, Some("PLOW_X")).is_err());
        let s = section(&d, None).unwrap();
        assert_eq!(s.name, "serve.json");
        assert_eq!(ServeManifest::parse(&s.data).unwrap().version, serve_manifest::VERSION);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
