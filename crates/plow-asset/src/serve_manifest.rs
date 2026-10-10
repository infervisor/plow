//! `serve.json`: what the host side of `plowrt serve` needs about a model besides its device
//! programs — chat template, stop set, sampling defaults, KV geometry, serve-knob defaults and
//! weight pins. `plowc` writes it into `model.pkt` from the checkpoint at emit time, so a release
//! bundle serves from the packet alone. A packet without the section (emitted before it existed)
//! gets the same values synthesized from the checkpoint's HF files by [`ServeManifest::from_checkpoint`],
//! the reads the runtime used to do itself.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SECTION: &str = "serve.json";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServeManifest {
    pub version: u32,
    /// Absent: the model serves no chat route (completions only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat: Option<ChatSpec>,
    /// Token ids that end a generation: the checkpoint's eos set plus the ids a chat turn
    /// closes on before it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_token_ids: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampling: Option<SamplingDefaults>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv: Option<KvGeometry>,
    /// Defaults for registered `PLOW_*` serve knobs (the qualified recipe's serve settings);
    /// an explicit environment value overrides each.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub serve_defaults: BTreeMap<String, String>,
    /// The checkpoint shards the packet was emitted against.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub weights: Vec<WeightPin>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatSpec {
    /// The checkpoint's Jinja chat template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// Where the template came from, for logs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// A built-in prompt builder, for checkpoints that ship no template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builtin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bos_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eos_token: Option<String>,
    /// Markers framing a reasoning trace in the generation; absent: no trace is split out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningTags>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningTags {
    pub open: String,
    pub close: String,
}

impl ReasoningTags {
    pub fn think() -> Self {
        ReasoningTags { open: "<think>".into(), close: "</think>".into() }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingDefaults {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// 0 = off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
}

/// Attention cache layout per layer class. Full-attention layers own `[max_ctx]` rows, sliding
/// layers `window` rows; layers that read another layer's cache own none and are not listed.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvGeometry {
    pub full_layers: Vec<u32>,
    pub kv_heads_full: u32,
    pub head_dim_full: u32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slide_layers: Vec<u32>,
    #[serde(default)]
    pub kv_heads_slide: u32,
    #[serde(default)]
    pub head_dim_slide: u32,
    /// 0 when there are no sliding layers.
    #[serde(default)]
    pub window: u32,
}

/// A safetensors shard identified by size and the sha256 of its header (dtypes, shapes and
/// offsets of every tensor): cheap to check at load, and a shard with the same tensor names and
/// byte sizes but another layout or dtype no longer passes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightPin {
    pub file: String,
    pub bytes: u64,
    pub header_sha256: String,
}

impl ServeManifest {
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let m: ServeManifest = serde_json::from_slice(bytes).map_err(|e| format!("{SECTION}: {e}"))?;
        if m.version == 0 || m.version > VERSION {
            return Err(format!("{SECTION}: version {} is not one this reader knows (<= {VERSION})", m.version));
        }
        Ok(m)
    }

    pub fn to_json(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("serve manifest serializes")
    }

    /// What the HF files beside the weights say: `asset_dir` (and its `checkpoint/`), then
    /// `checkpoint_dir`, for the chat template and sampling defaults; `checkpoint_dir` for the stop
    /// set and KV geometry. No serve defaults and no weight pins.
    pub fn from_checkpoint(asset_dir: &Path, checkpoint_dir: &Path) -> Self {
        let chat = find_chat_template(asset_dir).or_else(|| find_chat_template(checkpoint_dir)).map(|t| ChatSpec {
            reasoning: think_tags_in(asset_dir, checkpoint_dir).then(ReasoningTags::think),
            template: Some(t.text),
            source: Some(t.source),
            builtin: None,
            bos_token: t.bos_token,
            eos_token: t.eos_token,
        });
        let mut stop_token_ids = read_eos_ids(checkpoint_dir);
        stop_token_ids.extend(chat_stop_ids(checkpoint_dir, &stop_token_ids));
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();
        ServeManifest {
            version: VERSION,
            chat,
            stop_token_ids,
            sampling: read_sampling(asset_dir).or_else(|| read_sampling(checkpoint_dir)),
            kv: KvGeometry::from_config(checkpoint_dir),
            serve_defaults: BTreeMap::new(),
            weights: Vec::new(),
        }
    }
}

/// Whether the tokenizer declares `<think>` and `</think>` as added tokens: the checkpoint frames
/// reasoning with them (Qwen3, GLM, DeepSeek-R1).
fn think_tags_in(asset_dir: &Path, checkpoint_dir: &Path) -> bool {
    [asset_dir.join("tokenizer.json"), asset_dir.join("checkpoint").join("tokenizer.json"), checkpoint_dir.join("tokenizer.json")]
        .iter()
        .find_map(|p| std::fs::read(p).ok())
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            let added = v.get("added_tokens")?.as_array()?;
            let has = |t: &str| added.iter().any(|e| e.get("content").and_then(|c| c.as_str()) == Some(t));
            Some(has("<think>") && has("</think>"))
        })
        .unwrap_or(false)
}

fn read_json(p: &Path) -> Option<serde_json::Value> {
    std::fs::read(p).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

/// `generation_config.json` from `dir` or `dir/checkpoint`.
pub fn read_sampling(dir: &Path) -> Option<SamplingDefaults> {
    let v = [dir.join("generation_config.json"), dir.join("checkpoint").join("generation_config.json")]
        .iter()
        .find_map(|p| read_json(p))?;
    let f = |k: &str| v.get(k).and_then(serde_json::Value::as_f64).map(|x| x as f32);
    let s = SamplingDefaults {
        temperature: f("temperature"),
        top_p: f("top_p"),
        top_k: v.get("top_k").and_then(serde_json::Value::as_u64).map(|k| k as u32),
        repetition_penalty: f("repetition_penalty"),
    };
    (s != SamplingDefaults::default()).then_some(s)
}

/// The checkpoint's stop-token set: `generation_config.json` `eos_token_id` (int or list),
/// falling back to `config.json`, falling back to empty.
pub fn read_eos_ids(dir: &Path) -> Vec<u32> {
    for file in ["generation_config.json", "config.json"] {
        let Some(v) = read_json(&dir.join(file)) else {
            continue;
        };
        match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                if let Some(id) = n.as_u64() {
                    return vec![id as u32];
                }
            }
            Some(serde_json::Value::Array(a)) => {
                let ids: Vec<u32> = a.iter().filter_map(|x| x.as_u64().map(|v| v as u32)).collect();
                if !ids.is_empty() {
                    return ids;
                }
            }
            _ => {}
        }
    }
    Vec::new()
}

/// Stop ids a served chat turn needs beyond `eos_token_id`: a checkpoint whose eos is
/// `<|end_of_msg|>` and that ships `<|close|>` (the Kimi-K3 turn structure) closes its answer on
/// `<|close|>` before the sequence eos.
pub fn chat_stop_ids(dir: &Path, eos: &[u32]) -> Vec<u32> {
    let Some(v) = read_json(&dir.join("tokenizer_config.json")) else {
        return Vec::new();
    };
    let Some(added) = v.get("added_tokens_decoder").and_then(|a| a.as_object()) else {
        return Vec::new();
    };
    let id_of = |want: &str| -> Option<u32> {
        added.iter().find_map(|(k, e)| {
            (e.get("content").and_then(|c| c.as_str()) == Some(want))
                .then(|| k.parse::<u32>().ok())
                .flatten()
        })
    };
    match (id_of("<|end_of_msg|>"), id_of("<|close|>")) {
        (Some(eom), Some(close)) if eos.contains(&eom) => vec![close],
        _ => Vec::new(),
    }
}

impl KvGeometry {
    /// The checkpoint's `config.json` (`text_config` or top level). `layer_types` splits full and
    /// sliding layers and then `sliding_window` is required; without it every
    /// `num_hidden_layers` layer is full attention. The trailing `num_kv_shared_layers` read an
    /// earlier layer's cache and own none. `None` when the shape is not there.
    pub fn from_config(checkpoint_dir: &Path) -> Option<Self> {
        let v = read_json(&checkpoint_dir.join("config.json"))?;
        let t = v.get("text_config").unwrap_or(&v);
        let mut full_layers = Vec::new();
        let mut slide_layers = Vec::new();
        match t.get("layer_types").and_then(|x| x.as_array()) {
            Some(layer_types) => {
                for (l, ty) in layer_types.iter().enumerate() {
                    match ty.as_str()? {
                        "full_attention" => full_layers.push(l as u32),
                        "sliding_attention" => slide_layers.push(l as u32),
                        _ => return None,
                    }
                }
            }
            None => {
                let n = t.get("num_hidden_layers")?.as_u64()? as u32;
                full_layers = (0..n).collect();
            }
        }
        let u = |k: &str| t.get(k).and_then(|x| x.as_u64()).map(|x| x as u32);
        if let (Some(n), Some(shared)) = (u("num_hidden_layers"), u("num_kv_shared_layers")) {
            let own = n.saturating_sub(shared);
            full_layers.retain(|&l| l < own);
            slide_layers.retain(|&l| l < own);
        }
        let kv_heads_slide = u("num_key_value_heads")?;
        let head_dim_slide = u("head_dim")?;
        let window = match slide_layers.is_empty() {
            true => u("sliding_window").unwrap_or(0),
            false => u("sliding_window")?,
        };
        Some(KvGeometry {
            full_layers,
            kv_heads_full: u("num_global_key_value_heads").unwrap_or(kv_heads_slide),
            head_dim_full: u("global_head_dim").unwrap_or(head_dim_slide),
            slide_layers,
            kv_heads_slide,
            head_dim_slide,
            window,
        })
    }
}

/// A chat template found beside the weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoundTemplate {
    pub text: String,
    pub source: String,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

fn non_empty(p: &Path) -> Option<String> {
    std::fs::read_to_string(p).ok().filter(|s| !s.trim().is_empty())
}

/// The template text plus the special tokens, from `dir` or `dir/checkpoint`. HF puts the template
/// in four places and all four are in the wild: `chat_template.jinja`, `chat_template.json`, a
/// `chat_template` key in `tokenizer_config.json` (a string or a list of named templates) and a
/// `chat_templates/` directory. Ordered as `transformers` orders them.
pub fn find_chat_template(dir: &Path) -> Option<FoundTemplate> {
    for base in [dir.to_path_buf(), dir.join("checkpoint")] {
        let jinja = base.join("chat_template.jinja");
        let cfg_path = base.join("tokenizer_config.json");
        let cfg = read_json(&cfg_path);
        let tok = |k: &str| -> Option<String> {
            match cfg.as_ref()?.get(k)? {
                serde_json::Value::String(s) => Some(s.clone()),
                serde_json::Value::Object(o) => o.get("content")?.as_str().map(str::to_string),
                _ => None,
            }
        };
        let found = |text: String, source: String| FoundTemplate {
            text,
            source,
            bos_token: tok("bos_token"),
            eos_token: tok("eos_token"),
        };
        if let Some(text) = non_empty(&jinja) {
            return Some(found(text, jinja.display().to_string()));
        }
        let json_path = base.join("chat_template.json");
        if let Some(text) = non_empty(&json_path)
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("chat_template")?.as_str().map(str::to_owned))
            .filter(|s| !s.trim().is_empty())
        {
            return Some(found(text, json_path.display().to_string()));
        }
        let entry = cfg.as_ref().and_then(|c| c.get("chat_template"));
        if let Some(text) = entry.and_then(|v| v.as_str()).filter(|s| !s.trim().is_empty()) {
            return Some(found(text.to_string(), format!("{}#chat_template", cfg_path.display())));
        }
        if let Some((name, text)) = entry.and_then(|v| v.as_array()).and_then(|entries| {
            let pick = entries
                .iter()
                .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
                .or_else(|| entries.first())?;
            let text = pick.get("template")?.as_str()?.to_owned();
            let name = pick.get("name").and_then(|n| n.as_str()).unwrap_or("default").to_owned();
            (!text.trim().is_empty()).then_some((name, text))
        }) {
            return Some(found(text, format!("{}#chat_template[{name}]", cfg_path.display())));
        }
        let tdir = base.join("chat_templates");
        let named = tdir.join("default.jinja");
        let from_dir = non_empty(&named).map(|t| (named.clone(), t)).or_else(|| {
            let mut entries: Vec<_> = std::fs::read_dir(&tdir)
                .ok()?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "jinja"))
                .collect();
            entries.sort();
            let p = entries.into_iter().next()?;
            non_empty(&p).map(|t| (p, t))
        });
        if let Some((path, text)) = from_dir {
            return Some(found(text, path.display().to_string()));
        }
    }
    None
}

/// Size and header digest of one safetensors shard.
pub fn pin_shard(path: &Path) -> Result<WeightPin, String> {
    use std::io::Read;
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut f = std::fs::File::open(path).map_err(err)?;
    let bytes = f.metadata().map_err(err)?.len();
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(err)?;
    let n = u64::from_le_bytes(len);
    if n > bytes.saturating_sub(8) || n > 512 << 20 {
        return Err(format!("{}: safetensors header length {n} is out of range", path.display()));
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header).map_err(err)?;
    let mut h = Sha256::new();
    h.update(len);
    h.update(&header);
    Ok(WeightPin {
        file: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        bytes,
        header_sha256: format!("{:x}", h.finalize()),
    })
}

/// Every `*.safetensors` shard in `dir`, sorted by name.
pub fn pin_checkpoint(dir: &Path) -> Result<Vec<WeightPin>, String> {
    let mut shards: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    shards.sort();
    shards.iter().map(|p| pin_shard(p)).collect()
}

/// Check `pins` against the shards in `dir`. Every pinned shard must exist with the same size and
/// header; the error names each one that does not.
pub fn verify_pins(dir: &Path, pins: &[WeightPin]) -> Result<(), String> {
    let mut bad = Vec::new();
    for pin in pins {
        match pin_shard(&dir.join(&pin.file)) {
            Ok(got) if got == *pin => {}
            Ok(got) if got.bytes != pin.bytes => {
                bad.push(format!("{}: {} bytes, packet pins {}", pin.file, got.bytes, pin.bytes))
            }
            Ok(_) => bad.push(format!("{}: tensor header differs from the one the packet was emitted against", pin.file)),
            Err(e) => bad.push(e),
        }
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!("checkpoint {} does not match the packet: {}", dir.display(), bad.join("; ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("plow-serve-manifest-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn round_trips_and_refuses_newer_versions() {
        let m = ServeManifest {
            version: VERSION,
            chat: Some(ChatSpec { template: Some("{{ x }}".into()), reasoning: Some(ReasoningTags::think()), ..Default::default() }),
            stop_token_ids: vec![1, 106],
            sampling: Some(SamplingDefaults { temperature: Some(0.6), top_k: Some(20), ..Default::default() }),
            kv: Some(KvGeometry { full_layers: vec![5], kv_heads_full: 1, head_dim_full: 512, ..Default::default() }),
            serve_defaults: [("PLOW_PF_INTERLEAVE".to_string(), "2048".to_string())].into(),
            weights: vec![WeightPin { file: "a.safetensors".into(), bytes: 9, header_sha256: "00".into() }],
        };
        assert_eq!(ServeManifest::parse(&m.to_json()).unwrap(), m);
        assert_eq!(ServeManifest::parse(br#"{"version":1}"#).unwrap().chat, None);
        assert!(ServeManifest::parse(br#"{"version":2}"#).is_err());
        assert!(ServeManifest::parse(br#"{"version":1,"later":true}"#).is_err());
    }

    #[test]
    fn synthesizes_the_legacy_reads() {
        let d = tmp("legacy");
        std::fs::write(d.join("chat_template.jinja"), "{{ bos_token }}hi").unwrap();
        std::fs::write(d.join("tokenizer_config.json"), r#"{"bos_token":{"content":"<bos>"},"eos_token":"<eos>"}"#).unwrap();
        std::fs::write(d.join("generation_config.json"), r#"{"eos_token_id":[106,1],"temperature":0.6,"top_k":20}"#).unwrap();
        std::fs::write(
            d.join("config.json"),
            r#"{"text_config":{"layer_types":["sliding_attention","full_attention","sliding_attention"],
                "num_hidden_layers":3,"num_kv_shared_layers":1,"num_key_value_heads":8,"head_dim":256,
                "num_global_key_value_heads":1,"global_head_dim":512,"sliding_window":1024}}"#,
        )
        .unwrap();
        let m = ServeManifest::from_checkpoint(&d, &d);
        let chat = m.chat.unwrap();
        assert_eq!(chat.template.as_deref(), Some("{{ bos_token }}hi"));
        assert_eq!(chat.bos_token.as_deref(), Some("<bos>"));
        assert_eq!(chat.reasoning, None);
        assert_eq!(m.stop_token_ids, [1, 106]);
        assert_eq!(m.sampling.unwrap().top_k, Some(20));
        let kv = m.kv.unwrap();
        assert_eq!((kv.full_layers, kv.slide_layers, kv.window), (vec![1], vec![0], 1024));
        assert_eq!((kv.kv_heads_full, kv.head_dim_full, kv.kv_heads_slide, kv.head_dim_slide), (1, 512, 8, 256));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn k3_closes_its_turn_before_eos() {
        let d = tmp("k3");
        std::fs::write(
            d.join("tokenizer_config.json"),
            r#"{"added_tokens_decoder":{"163586":{"content":"<|end_of_msg|>"},"163590":{"content":"<|close|>"}}}"#,
        )
        .unwrap();
        assert_eq!(chat_stop_ids(&d, &[163586]), [163590]);
        assert!(chat_stop_ids(&d, &[2]).is_empty());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn weight_pins_catch_a_changed_header() {
        let d = tmp("pins");
        let shard = |hdr: &str| {
            let mut b = (hdr.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(hdr.as_bytes());
            b.extend_from_slice(&[0u8; 4]);
            b
        };
        std::fs::write(d.join("m.safetensors"), shard(r#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#)).unwrap();
        let pins = pin_checkpoint(&d).unwrap();
        assert!(verify_pins(&d, &pins).is_ok());
        std::fs::write(d.join("m.safetensors"), shard(r#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,4]}}"#)).unwrap();
        assert!(verify_pins(&d, &pins).unwrap_err().contains("header differs"));
        std::fs::remove_file(d.join("m.safetensors")).unwrap();
        assert!(verify_pins(&d, &pins).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
