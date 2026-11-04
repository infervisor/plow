//! The packet's `serve.json` (`plow_asset::serve_manifest`): the checkpoint's chat template, stop
//! set, sampling defaults and KV geometry, the recipe's serve-knob defaults and the shard pins, so
//! `plowrt serve` needs nothing from the checkpoint's HF files at serve time.

use std::path::Path;

use packet::devbuild::{SectionData, SECT_METADATA};
use plow_asset::serve_manifest::{self, ServeManifest};

/// `drafter_kv_layers`: full-attention KV layers the packet adds after the target's own (a block
/// drafter's context, `spec.context.json`); the KV layout pages and snapshots them like the target's.
pub(crate) fn manifest(
    dir: &Path,
    serve_defaults: Option<&str>,
    drafter_kv_layers: &[u32],
) -> Result<ServeManifest, String> {
    let mut m = ServeManifest::emit(dir, dir)?;
    if !drafter_kv_layers.is_empty() {
        m.kv.as_mut()
            .ok_or("drafter KV layers need the checkpoint's KV geometry")?
            .full_layers
            .extend_from_slice(drafter_kv_layers);
    }
    // The template's file name, not the emitting host's path.
    if let Some(chat) = m.chat.as_mut() {
        chat.source = chat
            .source
            .as_deref()
            .and_then(|s| Path::new(s).file_name())
            .map(|n| n.to_string_lossy().into_owned());
    }
    for item in serve_defaults
        .unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let (key, value) = item
            .split_once('=')
            .filter(|(k, _)| k.starts_with("PLOW_"))
            .ok_or_else(|| format!("PLOW_EMIT_SERVE_DEFAULTS: `{item}` is not PLOW_KEY=VALUE"))?;
        m.serve_defaults
            .insert(key.trim().to_string(), value.trim().to_string());
    }
    m.weights = serve_manifest::pin_checkpoint(dir)?;
    Ok(m)
}

pub(crate) fn section(
    dir: &Path,
    serve_defaults: Option<&str>,
    drafter_kv_layers: &[u32],
) -> Result<SectionData, String> {
    section_with_tp(dir, serve_defaults, drafter_kv_layers, 1)
}

pub(crate) fn section_with_tp(
    dir: &Path,
    serve_defaults: Option<&str>,
    drafter_kv_layers: &[u32],
    tp: u32,
) -> Result<SectionData, String> {
    if tp == 0 {
        return Err("serve KV geometry requires a positive TP degree".into());
    }
    let mut m = manifest(dir, serve_defaults, drafter_kv_layers)?;
    if tp > 1 {
        if let Some(kv) = m.kv.as_mut() {
            shard_kv_geometry(kv, tp)?;
        }
    }
    Ok(SectionData {
        kind: SECT_METADATA,
        name: serve_manifest::SECTION.into(),
        data: m.to_json(),
    })
}

fn shard_kv_geometry(kv: &mut serve_manifest::KvGeometry, tp: u32) -> Result<(), String> {
    let local = |heads: u32, layers: &[u32]| {
        if layers.is_empty() {
            return Ok(heads);
        }
        if tp == 0 || heads == 0 || (heads % tp != 0 && tp % heads != 0) {
            return Err(format!("serve KV heads {heads} cannot shard over TP={tp}"));
        }
        Ok(crate::kvh_local(heads, tp, layers[0]))
    };
    let full = local(kv.kv_heads_full, &kv.full_layers)?;
    let slide = local(kv.kv_heads_slide, &kv.slide_layers)?;
    kv.kv_heads_full = full;
    kv.kv_heads_slide = slide;
    Ok(())
}

/// [`section`] for the full-model emitters outside the dense path, with the emit config's
/// `PLOW_EMIT_SERVE_DEFAULTS`; a checkpoint the manifest cannot be built from fails the emit.
pub(crate) fn section_for(dir: &Path) -> SectionData {
    section(
        dir,
        crate::emit_config::active().serve_defaults.as_deref(),
        &[],
    )
    .unwrap_or_else(|error| panic!("serve manifest: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tp_fixture(name: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("devgen-serve-tp-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"layer_types":["sliding_attention","full_attention"],"num_hidden_layers":2,
                "num_key_value_heads":8,"head_dim":256,"num_global_key_value_heads":4,
                "global_head_dim":512,"sliding_window":1024,"eos_token_id":1}"#,
        )
        .unwrap();
        dir
    }

    #[test]
    fn tp1_serve_bytes_match_the_checkpoint_manifest() {
        let dir = tp_fixture("identity");
        let expected = manifest(&dir, Some("PLOW_MULTISTEP=0"), &[]).unwrap();
        let actual = section_with_tp(&dir, Some("PLOW_MULTISTEP=0"), &[], 1).unwrap();
        assert_eq!(actual.data, expected.to_json());
        assert_eq!(
            section(&dir, Some("PLOW_MULTISTEP=0"), &[]).unwrap().data,
            actual.data
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tp2_geometry_splits_full_and_sliding_heads() {
        let dir = tp_fixture("split");
        let expected = manifest(&dir, None, &[]).unwrap();
        let section = section_with_tp(&dir, None, &[], 2).unwrap();
        let actual = ServeManifest::parse(&section.data).unwrap();
        let mut geometry = expected.kv.clone().unwrap();
        geometry.kv_heads_full = 2;
        geometry.kv_heads_slide = 4;
        let mut expected = expected;
        expected.kv = Some(geometry);
        assert_eq!(actual, expected);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn tp_geometry_keeps_replicated_kv_heads() {
        let dir = tp_fixture("replicate");
        for tp in [8, 16] {
            let section = section_with_tp(&dir, None, &[], tp).unwrap();
            let kv = ServeManifest::parse(&section.data).unwrap().kv.unwrap();
            assert_eq!((kv.kv_heads_full, kv.kv_heads_slide), (1, 1));
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn missing_cache_classes_keep_their_unused_geometry() {
        let mut kv = serve_manifest::KvGeometry {
            slide_layers: vec![0],
            kv_heads_slide: 8,
            head_dim_slide: 256,
            window: 1024,
            ..Default::default()
        };
        shard_kv_geometry(&mut kv, 2).unwrap();
        assert_eq!((kv.kv_heads_full, kv.kv_heads_slide), (0, 4));
        kv.full_layers = vec![1];
        kv.kv_heads_full = 4;
        kv.slide_layers.clear();
        kv.kv_heads_slide = 0;
        shard_kv_geometry(&mut kv, 2).unwrap();
        assert_eq!((kv.kv_heads_full, kv.kv_heads_slide), (2, 0));
    }

    #[test]
    fn zero_tp_degree_is_refused() {
        let dir = tp_fixture("zero");
        assert!(section_with_tp(&dir, None, &[], 0).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn incompatible_or_zero_head_counts_are_refused_without_mutation() {
        for heads in [0, 3, 6] {
            let mut kv = serve_manifest::KvGeometry {
                full_layers: vec![0],
                kv_heads_full: 4,
                slide_layers: vec![1],
                kv_heads_slide: heads,
                ..Default::default()
            };
            let before = kv.clone();
            assert!(shard_kv_geometry(&mut kv, 4).is_err());
            assert_eq!(kv, before);
        }
    }

    #[test]
    fn drafter_layers_use_the_same_local_full_head_geometry() {
        let dir = tp_fixture("drafter");
        let section = section_with_tp(&dir, None, &[2, 3], 2).unwrap();
        let kv = ServeManifest::parse(&section.data).unwrap().kv.unwrap();
        assert_eq!(kv.full_layers, [1, 2, 3]);
        assert_eq!(kv.slide_layers, [0]);
        assert_eq!((kv.kv_heads_full, kv.kv_heads_slide), (2, 4));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn carries_serve_defaults_and_refuses_malformed_ones() {
        let d = std::env::temp_dir().join(format!("devgen-serve-section-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("generation_config.json"), r#"{"eos_token_id":1}"#).unwrap();
        // No template and no vocabulary: no chat route, completions only.
        assert_eq!(manifest(&d, None, &[]).unwrap().chat, None);
        std::fs::write(d.join("chat_template.jinja"), "{{ messages[0].content }}").unwrap();
        let m = manifest(&d, Some("PLOW_PF_INTERLEAVE=2048, PLOW_MULTISTEP=0"), &[]).unwrap();
        assert_eq!(m.version, serve_manifest::VERSION);
        let chat = m.chat.as_ref().unwrap();
        assert_eq!(chat.source.as_deref(), Some("chat_template.jinja"));
        assert_eq!(chat.response.as_ref().unwrap().tools, "none");
        assert_eq!(m.serve_defaults["PLOW_PF_INTERLEAVE"], "2048");
        assert_eq!(m.serve_defaults["PLOW_MULTISTEP"], "0");
        assert_eq!(m.stop_token_ids, [1]);
        assert!(m.weights.is_empty());
        assert!(manifest(&d, Some("PF_INTERLEAVE=1"), &[]).is_err());
        assert!(manifest(&d, Some("PLOW_X"), &[]).is_err());
        assert!(
            manifest(&d, None, &[48]).is_err(),
            "no KV geometry to extend"
        );
        let s = section(&d, None, &[]).unwrap();
        assert_eq!(s.name, "serve.json");
        assert_eq!(
            ServeManifest::parse(&s.data).unwrap().version,
            serve_manifest::VERSION
        );
        std::fs::write(
            d.join("config.json"),
            r#"{"layer_types":["sliding_attention","full_attention"],"num_hidden_layers":2,
                "num_key_value_heads":8,"head_dim":256,"num_global_key_value_heads":1,
                "global_head_dim":512,"sliding_window":1024}"#,
        )
        .unwrap();
        let kv = |extra: &[u32]| manifest(&d, None, extra).unwrap().kv.unwrap();
        assert_eq!(kv(&[]).full_layers, [1]);
        assert_eq!(kv(&[2, 3]).full_layers, [1, 2, 3]);
        assert_eq!(kv(&[2, 3]).slide_layers, [0]);
        std::fs::remove_dir_all(&d).unwrap();
    }
}
