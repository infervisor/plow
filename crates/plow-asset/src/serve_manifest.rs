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
/// The version `plowc` emits. Version 2 DECLARES what the model answers in ([`ResponseSpec`]) and
/// which built-in builder a template-less checkpoint gets; the server reads nothing from the
/// template text or the vocabulary for it. Version 1 (and a legacy packet without the section)
/// left those to the server's own probes.
pub const VERSION: u32 = 2;
/// The version [`ServeManifest::from_checkpoint`] synthesizes for a legacy packet.
pub const LEGACY_VERSION: u32 = 1;

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
    /// Version 1: markers framing a reasoning trace in the generation; absent: the server reads
    /// the template for them. Version 2 declares them in `response` instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningTags>,
    /// Version 2: what the model answers in. Required when the manifest has a `chat` section.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<ResponseSpec>,
}

/// What a version-2 packet's model answers in, fixed at emit from the template and the
/// vocabulary. Explicit `none` states, so an emitter that forgot a field fails validation
/// instead of being read as "no tools" or "no trace".
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseSpec {
    /// `"none"` (the template never renders `tools`), `"unparsed"` (it does, in a syntax the
    /// server has no parser for) or the syntax's id (`ToolFormat::name`).
    pub tools: String,
    pub reasoning: ReasoningDecl,
}

/// Whether, and how, the model frames a reasoning trace in its generation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningDecl {
    None,
    Tags { open: String, close: String },
}

impl ReasoningDecl {
    pub fn markers(&self) -> Option<(&str, &str)> {
        match self {
            ReasoningDecl::None => None,
            ReasoningDecl::Tags { open, close } => Some((open, close)),
        }
    }
}

impl ResponseSpec {
    pub const TOOLS_NONE: &'static str = "none";
    pub const TOOLS_UNPARSED: &'static str = "unparsed";

    /// The declared tool support, when the id is one this build knows.
    pub fn tool_support(&self) -> Option<crate::chat::ToolSupport> {
        use crate::chat::{ToolFormat, ToolSupport};
        Some(match self.tools.as_str() {
            Self::TOOLS_NONE => ToolSupport::None,
            Self::TOOLS_UNPARSED => ToolSupport::Unparsed,
            id => ToolSupport::Format(ToolFormat::from_name(id)?),
        })
    }

    fn from_template(t: &crate::chat::ChatTemplate, think_tokens: bool) -> Self {
        use crate::chat::ToolSupport;
        let tools = match t.tools {
            ToolSupport::None => Self::TOOLS_NONE.to_string(),
            ToolSupport::Unparsed => Self::TOOLS_UNPARSED.to_string(),
            ToolSupport::Format(f) => f.name().to_string(),
        };
        ResponseSpec {
            tools,
            reasoning: reasoning_decl(t.reasoning, think_tokens),
        }
    }
}

/// The version-1 server's precedence, now fixed at emit: `<think>` when the tokenizer's added
/// tokens have the pair (the version-1 manifest declared it, and a declaration beat the template);
/// else a trace channel of the template's own (Gemma 4); else none. A template that mentions
/// `<think>` without the tokens is a trace the model only ever closes.
fn reasoning_decl(template: Option<(&str, &str)>, think_tokens: bool) -> ReasoningDecl {
    let think = ReasoningTags::think();
    if think_tokens {
        return ReasoningDecl::Tags {
            open: think.open,
            close: think.close,
        };
    }
    match template {
        Some((open, close)) if (open, close) != (think.open.as_str(), think.close.as_str()) => {
            ReasoningDecl::Tags {
                open: open.into(),
                close: close.into(),
            }
        }
        _ => ReasoningDecl::None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReasoningTags {
    pub open: String,
    pub close: String,
}

impl ReasoningTags {
    pub fn think() -> Self {
        ReasoningTags {
            open: "<think>".into(),
            close: "</think>".into(),
        }
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
        let m: ServeManifest =
            serde_json::from_slice(bytes).map_err(|e| format!("{SECTION}: {e}"))?;
        if m.version == 0 || m.version > VERSION {
            return Err(format!(
                "{SECTION}: version {} is not one this reader knows (<= {VERSION})",
                m.version
            ));
        }
        m.validate()?;
        Ok(m)
    }

    /// Whether the packet declares its serving choices (version 2) rather than leaving them to
    /// the server's probes.
    pub fn declares_response(&self) -> bool {
        self.version >= 2
    }

    /// The version's invariants: version 1 carries no version-2 fields; version 2 declares the
    /// response protocol with known ids, names a template that compiles or a known built-in
    /// builder, and no version-1 reasoning field.
    fn validate(&self) -> Result<(), String> {
        let bad = |what: String| Err(format!("{SECTION} version {}: {what}", self.version));
        let Some(chat) = &self.chat else {
            return Ok(());
        };
        if !self.declares_response() {
            if chat.response.is_some() {
                return bad("`chat.response` is a version-2 field".into());
            }
            return Ok(());
        }
        if chat.reasoning.is_some() {
            return bad(
                "`chat.reasoning` is a version-1 field; declare `chat.response.reasoning`".into(),
            );
        }
        let Some(response) = &chat.response else {
            return bad("`chat.response` is required".into());
        };
        if response.tool_support().is_none() {
            return bad(format!(
                "`chat.response.tools` {:?} is not a tool syntax this build parses",
                response.tools
            ));
        }
        if let ReasoningDecl::Tags { open, close } = &response.reasoning {
            if open.is_empty() || close.is_empty() {
                return bad("`chat.response.reasoning` tags must be non-empty".into());
            }
        }
        if let Some(id) = &chat.builtin {
            if !crate::chat::BUILTIN_IDS.contains(&id.as_str()) {
                return bad(format!(
                    "`chat.builtin` {id:?} is not a built-in builder this build has"
                ));
            }
        }
        match &chat.template {
            Some(text) => {
                crate::chat::ChatTemplate::check(
                    text,
                    chat.source.clone().unwrap_or_else(|| SECTION.into()),
                    chat.bos_token.clone(),
                    chat.eos_token.clone(),
                )
                .map_err(|e| {
                    format!(
                        "{SECTION} version {}: chat template does not compile: {e}",
                        self.version
                    )
                })?;
            }
            None if chat.builtin.is_none() => {
                return bad("`chat` names neither a template nor a built-in builder".into());
            }
            // The built-in builders render no tools, so a packet served by one cannot declare a
            // tool syntax the request planner would never see.
            None if response.tools != ResponseSpec::TOOLS_NONE => {
                return bad(format!(
                    "`chat.response.tools` {:?} with a built-in builder and no template",
                    response.tools
                ));
            }
            None => {}
        }
        Ok(())
    }

    pub fn to_json(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("serve manifest serializes")
    }

    /// What the HF files beside the weights say: `asset_dir` (and its `checkpoint/`), then
    /// `checkpoint_dir`, for the chat template and sampling defaults; `checkpoint_dir` for the stop
    /// set and KV geometry. No serve defaults and no weight pins.
    pub fn from_checkpoint(asset_dir: &Path, checkpoint_dir: &Path) -> Self {
        let chat = find_chat_template(asset_dir)
            .or_else(|| find_chat_template(checkpoint_dir))
            .map(|t| ChatSpec {
                reasoning: think_tags_in(asset_dir, checkpoint_dir).then(ReasoningTags::think),
                template: Some(t.text),
                source: Some(t.source),
                builtin: None,
                bos_token: t.bos_token,
                eos_token: t.eos_token,
                response: None,
            });
        let mut stop_token_ids = read_eos_ids(checkpoint_dir);
        stop_token_ids.extend(chat_stop_ids(checkpoint_dir, &stop_token_ids));
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();
        ServeManifest {
            version: LEGACY_VERSION,
            chat,
            stop_token_ids,
            sampling: read_sampling(asset_dir).or_else(|| read_sampling(checkpoint_dir)),
            kv: KvGeometry::from_config(checkpoint_dir),
            serve_defaults: BTreeMap::new(),
            weights: Vec::new(),
        }
    }

    /// The manifest `plowc` emits (version [`VERSION`]): [`Self::from_checkpoint`]'s reads, with
    /// the response protocol declared from the template (compiled here, so one that does not
    /// compile fails the emit) and the vocabulary, and the built-in builder named for a
    /// checkpoint that ships no template. A checkpoint with neither a template nor a vocabulary
    /// naming a builder serves no chat route (`chat: None`): the version-1 server fell back to
    /// Gemma's builder for it, logging that the model almost certainly does not use that format,
    /// and a deliberately wrong prompt is not a chat route. Still no serve defaults and no weight
    /// pins.
    pub fn emit(asset_dir: &Path, checkpoint_dir: &Path) -> Result<Self, String> {
        let mut m = Self::from_checkpoint(asset_dir, checkpoint_dir);
        m.version = VERSION;
        let think = think_tags_in(asset_dir, checkpoint_dir);
        m.chat = match m.chat.take() {
            Some(mut chat) => {
                let text = chat
                    .template
                    .clone()
                    .expect("from_checkpoint fills the template");
                let t = crate::chat::ChatTemplate::compile(
                    text,
                    chat.source.clone().unwrap_or_else(|| SECTION.into()),
                    chat.bos_token.clone(),
                    chat.eos_token.clone(),
                )?;
                chat.reasoning = None;
                chat.response = Some(ResponseSpec::from_template(&t, think));
                Some(chat)
            }
            None => {
                let one = |marker: &str| added_token_in(asset_dir, checkpoint_dir, marker);
                crate::chat::builtin_for(one).map(|builtin| ChatSpec {
                    builtin: Some(builtin.into()),
                    response: Some(ResponseSpec {
                        tools: ResponseSpec::TOOLS_NONE.into(),
                        reasoning: reasoning_decl(None, think),
                    }),
                    ..Default::default()
                })
            }
        };
        m.validate()?;
        Ok(m)
    }
}

/// The tokenizer's `added_tokens` (`asset_dir`, its `checkpoint/`, then `checkpoint_dir`: the
/// order the server loads `tokenizer.json` in, so compiler and server read the same file). The
/// version-2 builder contract is bounded by this file: a marker is "one token" when it is an
/// added token. A marker that the base vocabulary happens to encode as one id without being an
/// added token names no builder here, and a checkpoint without `tokenizer.json` (served by the
/// byte-fallback tokenizer, which the GPU engines refuse) has none. Agreement with the server's
/// own encode probe is verified by the serve goldens on the production vocabularies.
pub fn added_tokens(asset_dir: &Path, checkpoint_dir: &Path) -> Vec<String> {
    [
        asset_dir.join("tokenizer.json"),
        asset_dir.join("checkpoint").join("tokenizer.json"),
        checkpoint_dir.join("tokenizer.json"),
    ]
    .iter()
    .find_map(|p| std::fs::read(p).ok())
    .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
    .and_then(|v| {
        Some(
            v.get("added_tokens")?
                .as_array()?
                .iter()
                .filter_map(|e| e.get("content")?.as_str().map(str::to_string))
                .collect(),
        )
    })
    .unwrap_or_default()
}

/// Whether `marker` is one added token of the tokenizer: the server's "one id in its own vocab"
/// probe, as the tokenizer files state it.
fn added_token_in(asset_dir: &Path, checkpoint_dir: &Path, marker: &str) -> bool {
    added_tokens(asset_dir, checkpoint_dir)
        .iter()
        .any(|t| t == marker)
}

/// Whether the tokenizer declares `<think>` and `</think>` as added tokens: the checkpoint frames
/// reasoning with them (Qwen3, GLM, DeepSeek-R1).
fn think_tags_in(asset_dir: &Path, checkpoint_dir: &Path) -> bool {
    let added = added_tokens(asset_dir, checkpoint_dir);
    let has = |t: &str| added.iter().any(|a| a == t);
    has("<think>") && has("</think>")
}

fn read_json(p: &Path) -> Option<serde_json::Value> {
    std::fs::read(p)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
}

/// `generation_config.json` from `dir` or `dir/checkpoint`.
pub fn read_sampling(dir: &Path) -> Option<SamplingDefaults> {
    let v = [
        dir.join("generation_config.json"),
        dir.join("checkpoint").join("generation_config.json"),
    ]
    .iter()
    .find_map(|p| read_json(p))?;
    let f = |k: &str| {
        v.get(k)
            .and_then(serde_json::Value::as_f64)
            .map(|x| x as f32)
    };
    let s = SamplingDefaults {
        temperature: f("temperature"),
        top_p: f("top_p"),
        top_k: v
            .get("top_k")
            .and_then(serde_json::Value::as_u64)
            .map(|k| k as u32),
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
                let ids: Vec<u32> = a
                    .iter()
                    .filter_map(|x| x.as_u64().map(|v| v as u32))
                    .collect();
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
    /// earlier layer's cache and own none. `linear_attention` layers (Qwen3-Next style gated
    /// DeltaNet) carry recurrent state instead of KV rows and own no cache rows here. `None` when
    /// the shape is not there.
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
                        "linear_attention" => {}
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
    std::fs::read_to_string(p)
        .ok()
        .filter(|s| !s.trim().is_empty())
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
        if let Some(text) = entry
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
        {
            return Some(found(
                text.to_string(),
                format!("{}#chat_template", cfg_path.display()),
            ));
        }
        if let Some((name, text)) = entry.and_then(|v| v.as_array()).and_then(|entries| {
            let pick = entries
                .iter()
                .find(|e| e.get("name").and_then(|n| n.as_str()) == Some("default"))
                .or_else(|| entries.first())?;
            let text = pick.get("template")?.as_str()?.to_owned();
            let name = pick
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("default")
                .to_owned();
            (!text.trim().is_empty()).then_some((name, text))
        }) {
            return Some(found(
                text,
                format!("{}#chat_template[{name}]", cfg_path.display()),
            ));
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
        return Err(format!(
            "{}: safetensors header length {n} is out of range",
            path.display()
        ));
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header).map_err(err)?;
    let mut h = Sha256::new();
    h.update(len);
    h.update(&header);
    Ok(WeightPin {
        file: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
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
            Ok(got) if got.bytes != pin.bytes => bad.push(format!(
                "{}: {} bytes, packet pins {}",
                pin.file, got.bytes, pin.bytes
            )),
            Ok(_) => bad.push(format!(
                "{}: tensor header differs from the one the packet was emitted against",
                pin.file
            )),
            Err(e) => bad.push(e),
        }
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "checkpoint {} does not match the packet: {}",
            dir.display(),
            bad.join("; ")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d =
            std::env::temp_dir().join(format!("plow-serve-manifest-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn round_trips_and_refuses_newer_versions() {
        let m = ServeManifest {
            version: LEGACY_VERSION,
            chat: Some(ChatSpec {
                template: Some("{{ x }}".into()),
                reasoning: Some(ReasoningTags::think()),
                ..Default::default()
            }),
            stop_token_ids: vec![1, 106],
            sampling: Some(SamplingDefaults {
                temperature: Some(0.6),
                top_k: Some(20),
                ..Default::default()
            }),
            kv: Some(KvGeometry {
                full_layers: vec![5],
                kv_heads_full: 1,
                head_dim_full: 512,
                ..Default::default()
            }),
            serve_defaults: [("PLOW_PF_INTERLEAVE".to_string(), "2048".to_string())].into(),
            weights: vec![WeightPin {
                file: "a.safetensors".into(),
                bytes: 9,
                header_sha256: "00".into(),
            }],
        };
        assert_eq!(ServeManifest::parse(&m.to_json()).unwrap(), m);
        assert_eq!(
            ServeManifest::parse(br#"{"version":1}"#).unwrap().chat,
            None
        );
        assert!(ServeManifest::parse(br#"{"version":3}"#).is_err());
        assert!(ServeManifest::parse(br#"{"version":1,"later":true}"#).is_err());
    }

    fn v2(chat: serde_json::Value) -> Result<ServeManifest, String> {
        ServeManifest::parse(
            serde_json::json!({"version": 2, "chat": chat})
                .to_string()
                .as_bytes(),
        )
    }

    #[test]
    fn version_2_declares_the_response_or_is_refused() {
        let ok = v2(serde_json::json!({
            "template": "{{ messages[0].content }}",
            "response": {"tools": "hermes", "reasoning": {"kind": "tags", "open": "<think>", "close": "</think>"}}
        }))
        .unwrap();
        assert!(ok.declares_response());
        let response = ok.chat.as_ref().unwrap().response.as_ref().unwrap();
        assert_eq!(
            response.tool_support(),
            Some(crate::chat::ToolSupport::Format(
                crate::chat::ToolFormat::Hermes
            ))
        );
        assert_eq!(response.reasoning.markers(), Some(("<think>", "</think>")));
        assert_eq!(ServeManifest::parse(&ok.to_json()).unwrap(), ok);
        // No `chat` at all is a completions-only packet, on either version.
        assert!(ServeManifest::parse(br#"{"version":2}"#)
            .unwrap()
            .chat
            .is_none());
        let builtin = v2(serde_json::json!({
            "builtin": "k3", "response": {"tools": "none", "reasoning": {"kind": "none"}}
        }))
        .unwrap();
        assert_eq!(builtin.chat.unwrap().builtin.as_deref(), Some("k3"));

        let refused = |chat: serde_json::Value, what: &str| {
            let err = v2(chat).expect_err(what);
            assert!(err.contains(what), "{err}");
        };
        refused(
            serde_json::json!({"template": "{{ x }}"}),
            "`chat.response` is required",
        );
        refused(
            serde_json::json!({"template": "{{ x }}", "reasoning": {"open": "<think>", "close": "</think>"},
                "response": {"tools": "none", "reasoning": {"kind": "none"}}}),
            "version-1 field",
        );
        refused(
            serde_json::json!({"template": "{{ x }}", "response": {"tools": "xml2", "reasoning": {"kind": "none"}}}),
            "not a tool syntax",
        );
        refused(
            serde_json::json!({"template": "{{ x }}", "response": {"tools": "none",
                "reasoning": {"kind": "tags", "open": "", "close": "</think>"}}}),
            "must be non-empty",
        );
        refused(
            serde_json::json!({"template": "{{ x }}", "response": {"tools": "none", "reasoning": {"kind": "other"}}}),
            "serve.json",
        );
        refused(
            serde_json::json!({"builtin": "mistral", "response": {"tools": "none", "reasoning": {"kind": "none"}}}),
            "not a built-in builder",
        );
        refused(
            serde_json::json!({"builtin": "harmony", "response": {"tools": "harmony", "reasoning": {"kind": "none"}}}),
            "with a built-in builder and no template",
        );
        refused(
            serde_json::json!({"response": {"tools": "none", "reasoning": {"kind": "none"}}}),
            "neither a template nor a built-in builder",
        );
        refused(
            serde_json::json!({"template": "{% if %}", "response": {"tools": "none", "reasoning": {"kind": "none"}}}),
            "does not compile",
        );
        // Version 1 never carries the declaration.
        let err = ServeManifest::parse(
            br#"{"version":1,"chat":{"template":"x","response":{"tools":"none","reasoning":{"kind":"none"}}}}"#,
        )
        .unwrap_err();
        assert!(err.contains("version-2 field"), "{err}");
    }

    #[test]
    fn emit_declares_from_the_template_and_the_vocabulary() {
        let d = tmp("emit");
        std::fs::write(
            d.join("chat_template.jinja"),
            "{% for m in messages %}{{ m.content }}{% endfor %}{% if tools %}<tool_call>{{ tools | tojson }}{% endif %}<think>",
        )
        .unwrap();
        std::fs::write(d.join("generation_config.json"), r#"{"eos_token_id":1}"#).unwrap();
        // No tokenizer: the template's `<think>` alone does not declare a trace.
        let m = ServeManifest::emit(&d, &d).unwrap();
        assert_eq!(m.version, VERSION);
        let chat = m.chat.as_ref().unwrap();
        assert_eq!(chat.reasoning, None);
        let response = chat.response.as_ref().unwrap();
        assert_eq!(response.tools, "hermes");
        assert_eq!(response.reasoning, ReasoningDecl::None);
        assert_eq!(ServeManifest::parse(&m.to_json()).unwrap(), m);
        // The tokenizer's `<think>` tokens do.
        std::fs::write(
            d.join("tokenizer.json"),
            r#"{"added_tokens":[{"content":"<think>"},{"content":"</think>"},{"content":"<|end_of_msg|>"}]}"#,
        )
        .unwrap();
        let m = ServeManifest::emit(&d, &d).unwrap();
        let response = m.chat.unwrap().response.unwrap();
        assert_eq!(response.reasoning.markers(), Some(("<think>", "</think>")));
        // The tokenizer's `<think>` tokens win over a template's own channel (the version-1
        // manifest declared `<think>` from the tokens, and a declaration beat the template);
        // without the tokens the channel is declared.
        std::fs::write(
            d.join("chat_template.jinja"),
            "{{ messages[0].content }}<|channel>thought
<channel|>",
        )
        .unwrap();
        let m = ServeManifest::emit(&d, &d).unwrap();
        let response = m.chat.unwrap().response.unwrap();
        assert_eq!(response.tools, "none");
        assert_eq!(response.reasoning.markers(), Some(("<think>", "</think>")));
        std::fs::write(
            d.join("tokenizer.json"),
            r#"{"added_tokens":[{"content":"<|end_of_msg|>"}]}"#,
        )
        .unwrap();
        let m = ServeManifest::emit(&d, &d).unwrap();
        let response = m.chat.unwrap().response.unwrap();
        assert_eq!(
            response.reasoning.markers(),
            Some(("<|channel>thought", "<channel|>"))
        );
        // No template: the vocabulary names the builder (K3 here, whose `<think>` tokens were
        // removed above, so no trace is declared and no tool syntax).
        std::fs::remove_file(d.join("chat_template.jinja")).unwrap();
        let m = ServeManifest::emit(&d, &d).unwrap();
        let chat = m.chat.unwrap();
        assert_eq!(chat.builtin.as_deref(), Some("k3"));
        assert_eq!(chat.template, None);
        let response = chat.response.unwrap();
        assert_eq!(response.tools, "none");
        assert_eq!(response.reasoning, ReasoningDecl::None);
        // Neither: no chat route. (The version-1 server served Gemma's format here, with an error
        // log; a wrong prompt is not a chat route.)
        std::fs::write(d.join("tokenizer.json"), r#"{"added_tokens":[]}"#).unwrap();
        let m = ServeManifest::emit(&d, &d).unwrap();
        assert_eq!(m.chat, None);
        assert_eq!(ServeManifest::parse(&m.to_json()).unwrap(), m);
        // A template that does not compile fails the emit, never falls back.
        std::fs::write(d.join("chat_template.jinja"), "{% if %}").unwrap();
        assert!(ServeManifest::emit(&d, &d).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn the_asset_tokenizer_names_the_builder_before_the_checkpoints() {
        // The server loads the asset directory's tokenizer first; the compiler reads the same file
        // first, so a checkpoint whose vocabulary says otherwise does not change the builder.
        let d = tmp("builder-precedence");
        std::fs::create_dir_all(d.join("checkpoint")).unwrap();
        let other = tmp("builder-precedence-ckpt");
        std::fs::write(d.join("generation_config.json"), r#"{"eos_token_id":1}"#).unwrap();
        std::fs::write(
            other.join("tokenizer.json"),
            r#"{"added_tokens":[{"content":"<|end_of_msg|>"}]}"#,
        )
        .unwrap();
        assert_eq!(
            ServeManifest::emit(&d, &other)
                .unwrap()
                .chat
                .unwrap()
                .builtin
                .as_deref(),
            Some("k3")
        );
        std::fs::write(
            d.join("checkpoint/tokenizer.json"),
            r#"{"added_tokens":[{"content":"<|assistant|>"}]}"#,
        )
        .unwrap();
        assert_eq!(
            ServeManifest::emit(&d, &other)
                .unwrap()
                .chat
                .unwrap()
                .builtin
                .as_deref(),
            Some("glm")
        );
        std::fs::write(
            d.join("tokenizer.json"),
            r#"{"added_tokens":[{"content":"<|turn>"}]}"#,
        )
        .unwrap();
        assert_eq!(
            ServeManifest::emit(&d, &other)
                .unwrap()
                .chat
                .unwrap()
                .builtin
                .as_deref(),
            Some("gemma")
        );
        std::fs::remove_dir_all(&d).unwrap();
        std::fs::remove_dir_all(&other).unwrap();
    }

    #[test]
    fn synthesizes_the_legacy_reads() {
        let d = tmp("legacy");
        std::fs::write(d.join("chat_template.jinja"), "{{ bos_token }}hi").unwrap();
        std::fs::write(
            d.join("tokenizer_config.json"),
            r#"{"bos_token":{"content":"<bos>"},"eos_token":"<eos>"}"#,
        )
        .unwrap();
        std::fs::write(
            d.join("generation_config.json"),
            r#"{"eos_token_id":[106,1],"temperature":0.6,"top_k":20}"#,
        )
        .unwrap();
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
        assert_eq!(
            (kv.full_layers, kv.slide_layers, kv.window),
            (vec![1], vec![0], 1024)
        );
        assert_eq!(
            (
                kv.kv_heads_full,
                kv.head_dim_full,
                kv.kv_heads_slide,
                kv.head_dim_slide
            ),
            (1, 512, 8, 256)
        );
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn linear_attention_layers_own_no_kv_rows() {
        let d = tmp("gdn");
        std::fs::write(
            d.join("config.json"),
            r#"{"layer_types":["linear_attention","linear_attention","full_attention","linear_attention"],
                "num_hidden_layers":4,"num_key_value_heads":4,"head_dim":256,
                "linear_num_value_heads":48,"linear_key_head_dim":128}"#,
        )
        .unwrap();
        let kv = KvGeometry::from_config(&d).unwrap();
        assert_eq!(
            (kv.full_layers, kv.slide_layers, kv.window),
            (vec![2], vec![], 0)
        );
        assert_eq!((kv.kv_heads_full, kv.head_dim_full), (4, 256));
        std::fs::write(
            d.join("config.json"),
            r#"{"layer_types":["mamba"],"num_hidden_layers":1}"#,
        )
        .unwrap();
        assert!(KvGeometry::from_config(&d).is_none());
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
        std::fs::write(
            d.join("m.safetensors"),
            shard(r#"{"w":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#),
        )
        .unwrap();
        let pins = pin_checkpoint(&d).unwrap();
        assert!(verify_pins(&d, &pins).is_ok());
        std::fs::write(
            d.join("m.safetensors"),
            shard(r#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,4]}}"#),
        )
        .unwrap();
        assert!(verify_pins(&d, &pins)
            .unwrap_err()
            .contains("header differs"));
        std::fs::remove_file(d.join("m.safetensors")).unwrap();
        assert!(verify_pins(&d, &pins).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }
}
