//! Per-model serving configuration: what the CHECKPOINT says about how it
//! wants to be sampled and how it frames a reasoning trace.
//!
//! WHY THIS EXISTS. Every request was built from one process-wide
//! `SamplingParams::default()` — temperature 1.0, top_p 1.0, top_k off — for
//! every model, while `generation_config.json` (which this crate already reads,
//! for the eos set) carries the values the checkpoint's authors chose. Qwen3
//! ships 0.6/0.95/20; serving it at 1.0/1.0 is a quality regression against
//! vLLM, which applies the checkpoint's generation config by default, and it is
//! per-model by nature so there is nowhere else to put it.

use std::path::Path;

use crate::text::sample::SamplingParams;

/// Everything per-model the request path needs that is not the weights.
#[derive(Clone, Debug)]
pub struct ServingConfig {
    /// The checkpoint's own sampling defaults. A request field overrides one of
    /// these; a request that omits it gets the checkpoint's value, NOT the
    /// process-wide default.
    pub default_sampling: SamplingParams,
    /// The deployment's request defaults for this model (`--chat-defaults`), set once at startup.
    pub chat: std::sync::OnceLock<ChatDefaults>,
}

impl Default for ServingConfig {
    fn default() -> Self {
        ServingConfig {
            default_sampling: SamplingParams::default(),
            chat: std::sync::OnceLock::new(),
        }
    }
}

impl ServingConfig {
    /// The checkpoint's sampling defaults over the stock ones; a field it omits keeps the stock
    /// value.
    pub fn from_defaults(d: Option<&plow_asset::serve_manifest::SamplingDefaults>) -> Self {
        let mut cfg = ServingConfig::default();
        let Some(d) = d else { return cfg };
        let p = &mut cfg.default_sampling;
        if let Some(t) = d.temperature {
            p.temperature = t;
        }
        if let Some(t) = d.top_p {
            p.top_p = t;
        }
        if let Some(k) = d.top_k {
            p.top_k = k as usize;
        }
        if let Some(r) = d.repetition_penalty {
            p.repetition_penalty = r;
        }
        cfg
    }

    /// Read `generation_config.json` from `dir` or `dir/checkpoint` — the same
    /// two places [`crate::asset::checkpoint`] already looks for the eos set.
    ///
    /// Absent file, absent key or unparseable value each leave the
    /// corresponding default alone; a checkpoint that says nothing about
    /// sampling is served exactly as before.
    pub fn load(dir: &Path) -> Self {
        Self::from_defaults(plow_asset::serve_manifest::read_sampling(dir).as_ref())
    }
}

/// Per-model request defaults from `--chat-defaults` (`PLOW_CHAT_DEFAULTS`): a JSON object keyed
/// by served model name, the model-default layer for request fields `generation_config.json` does
/// not carry. A request field overrides its default.
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatDefaults {
    pub max_tokens: Option<u32>,
    pub thinking_token_budget: Option<i64>,
    pub max_reply_tokens: Option<u32>,
    /// Committed in place of the model's tokens once a trace block's budget is spent (a bridge
    /// sentence, then the close marker).
    pub reasoning_end: Option<String>,
    /// Appended after the generation prompt when thinking is on: the open marker and seed text.
    pub thought_prefix: Option<String>,
    /// Trace tokens allowed past the budget while waiting for a line end.
    #[serde(default)]
    pub newline_grace: u32,
    /// Under the request's own.
    pub chat_template_kwargs: Option<std::collections::BTreeMap<String, serde_json::Value>>,
    /// Move planning lines out of the answer (`serve::reply_guard`).
    pub reply_guard: Option<crate::serve::reply_guard::GuardMode>,
    /// The guard's fallback lines for an answer left empty.
    pub reply_guard_lines: Option<crate::serve::reply_guard::GuardLines>,
    /// A blank line after the trace ends the request (`finish_reason: "stop"`): a second
    /// paragraph of a voice reply is narration.
    #[serde(default)]
    pub reply_stop_blank_line: bool,
    /// `POST /v1/requests/{id}/reasoning_end` may end a thinking request's trace while it runs
    /// (a caller barged in). Caps that request's device-chained decode at
    /// [`crate::serve::think::LIVE_CHAIN`] tokens.
    #[serde(default)]
    pub live_reasoning_end: bool,
    /// Host-logit steering (CUDA engine; `serve::think::ThinkBias`). Biases by token text on the
    /// first two reply tokens, each text a single token.
    #[serde(default)]
    pub reply_first_bias: std::collections::BTreeMap<String, f32>,
    /// −100 on the open marker while replying.
    #[serde(default)]
    pub ban_reopen: bool,
    /// A rising bias on the close marker at line ends over a block's last `tokens` budget tokens.
    pub close_ramp: Option<CloseRamp>,
    /// Close a block at a line end when the close marker's logit ranks below this.
    pub early_exit_rank: Option<u32>,
    /// With `reply_guard`: a turn left with nothing to say is generated once more from its own
    /// trace, closed, before the fallback line is used.
    #[serde(default)]
    pub reply_regenerate: bool,
    /// With `PLOW_SPEC`: a forced `reasoning_end` rides speculative verifies as the row's
    /// drafts, several forced ids per tick instead of one.
    #[serde(default)]
    pub spec_forced_end: bool,
    /// llama.cpp `reasoning_budget_message`: the bridge before the close marker.
    pub reasoning_budget_message: Option<String>,
    /// llama.cpp `reasoning_format` (`none` keeps the trace in `content`).
    pub reasoning_format: Option<String>,
    /// OpenAI `reasoning_effort` (`none` turns thinking off).
    pub reasoning_effort: Option<String>,
    /// vLLM `include_reasoning`.
    pub include_reasoning: Option<bool>,
    /// vLLM `--override-generation-config`: sampling defaults and `max_new_tokens`.
    pub generation: Option<GenerationDefaults>,
    /// vLLM `--generation-config vllm`: the checkpoint's sampling defaults are not used.
    #[serde(skip)]
    pub stock_sampling: bool,
}

/// vLLM's `--override-generation-config` keys (HF `generation_config.json` names).
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub min_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    /// A default `max_tokens`.
    pub max_new_tokens: Option<u32>,
}

impl GenerationDefaults {
    fn over(self, base: Self) -> Self {
        GenerationDefaults {
            temperature: self.temperature.or(base.temperature),
            top_p: self.top_p.or(base.top_p),
            top_k: self.top_k.or(base.top_k),
            min_p: self.min_p.or(base.min_p),
            repetition_penalty: self.repetition_penalty.or(base.repetition_penalty),
            presence_penalty: self.presence_penalty.or(base.presence_penalty),
            frequency_penalty: self.frequency_penalty.or(base.frequency_penalty),
            max_new_tokens: self.max_new_tokens.or(base.max_new_tokens),
        }
    }

    fn apply(&self, p: &mut SamplingParams) {
        if let Some(v) = self.temperature {
            p.temperature = v;
        }
        if let Some(v) = self.top_p {
            p.top_p = v;
        }
        if let Some(v) = self.top_k {
            p.top_k = v as usize;
        }
        if let Some(v) = self.min_p {
            p.min_p = v;
        }
        if let Some(v) = self.repetition_penalty {
            p.repetition_penalty = v;
        }
        if let Some(v) = self.presence_penalty {
            p.presence_penalty = v;
        }
        if let Some(v) = self.frequency_penalty {
            p.frequency_penalty = v;
        }
    }
}

/// The sampling a request starts from: the checkpoint's defaults (or the stock ones under
/// `--generation-config vllm`), then the generation overrides.
pub fn base_sampling(bundle: &crate::asset::ModelBundle) -> SamplingParams {
    let d = bundle.serving().chat.get();
    let mut p = if d.is_some_and(|d| d.stock_sampling) {
        SamplingParams::default()
    } else {
        bundle.serving().default_sampling.clone()
    };
    if let Some(g) = d.and_then(|d| d.generation) {
        g.apply(&mut p);
    }
    p
}

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CloseRamp {
    pub tokens: u32,
    pub bias: f32,
}

impl ChatDefaults {
    /// Any host-logit steering rule is set.
    pub fn steers(&self) -> bool {
        !self.reply_first_bias.is_empty()
            || self.ban_reopen
            || self.close_ramp.is_some()
            || self.early_exit_rank.is_some()
    }

    /// The steering rules over the request's own `logit_bias`; `None` if a text is not one token.
    pub fn bias(
        &self,
        tok: &dyn crate::text::tokenizer::Tokenize,
        base: &[(u32, f32)],
    ) -> Result<crate::serve::think::ThinkBias, String> {
        let reply_first = self
            .reply_first_bias
            .iter()
            .map(
                |(text, &b)| match tok.encode_with_special_tokens(text, false)[..] {
                    [id] => Ok((id, b)),
                    ref ids => Err(format!("{text:?} is not a single token ({ids:?})")),
                },
            )
            .collect::<Result<_, _>>()?;
        Ok(crate::serve::think::ThinkBias {
            base: base.to_vec(),
            reply_first,
            ban_reopen: self.ban_reopen,
            close_ramp: self.close_ramp.map(|r| (r.tokens, r.bias)),
            early_exit_rank: self.early_exit_rank,
        })
    }
}

/// Why a thinking-control value cannot apply to a model with these markers.
pub fn check_thinking(
    markers: Option<(&str, &str)>,
    budget: Option<i64>,
    reasoning_end: Option<&str>,
) -> Result<(), (&'static str, String)> {
    if budget.is_some_and(|b| b < -1) {
        return Err((
            "thinking_token_budget",
            "`thinking_token_budget` must be -1 (off), 0 (thinking off) or a token count".into(),
        ));
    }
    if let Some(end) = reasoning_end.filter(|e| !e.is_empty()) {
        match markers {
            Some((_, close)) if end.ends_with(close) => {}
            _ => {
                return Err((
                    "reasoning_end",
                    format!(
                        "`reasoning_end` must end with the model's close marker {:?}",
                        markers.map_or("", |m| m.1)
                    ),
                ))
            }
        }
    }
    Ok(())
}

impl ChatDefaults {
    /// `self` (a model's file entry) over `base` (the server flags), field by field; template
    /// kwargs merge key by key.
    pub fn over(self, base: &ChatDefaults) -> ChatDefaults {
        let b = base.clone();
        let chat_template_kwargs = match (self.chat_template_kwargs, b.chat_template_kwargs) {
            (Some(mut mine), Some(theirs)) => {
                for (k, v) in theirs {
                    mine.entry(k).or_insert(v);
                }
                Some(mine)
            }
            (mine, theirs) => mine.or(theirs),
        };
        let generation = match (self.generation, b.generation) {
            (Some(mine), Some(theirs)) => Some(mine.over(theirs)),
            (mine, theirs) => mine.or(theirs),
        };
        ChatDefaults {
            max_tokens: self.max_tokens.or(b.max_tokens),
            thinking_token_budget: self.thinking_token_budget.or(b.thinking_token_budget),
            max_reply_tokens: self.max_reply_tokens.or(b.max_reply_tokens),
            reasoning_end: self.reasoning_end.or(b.reasoning_end),
            thought_prefix: self.thought_prefix.or(b.thought_prefix),
            newline_grace: if self.newline_grace > 0 {
                self.newline_grace
            } else {
                b.newline_grace
            },
            chat_template_kwargs,
            reply_guard: self.reply_guard.or(b.reply_guard),
            reply_guard_lines: self.reply_guard_lines.or(b.reply_guard_lines),
            reply_stop_blank_line: self.reply_stop_blank_line || b.reply_stop_blank_line,
            live_reasoning_end: self.live_reasoning_end || b.live_reasoning_end,
            reply_first_bias: if self.reply_first_bias.is_empty() {
                b.reply_first_bias
            } else {
                self.reply_first_bias
            },
            ban_reopen: self.ban_reopen || b.ban_reopen,
            close_ramp: self.close_ramp.or(b.close_ramp),
            early_exit_rank: self.early_exit_rank.or(b.early_exit_rank),
            reply_regenerate: self.reply_regenerate || b.reply_regenerate,
            spec_forced_end: self.spec_forced_end || b.spec_forced_end,
            reasoning_budget_message: self.reasoning_budget_message.or(b.reasoning_budget_message),
            reasoning_format: self.reasoning_format.or(b.reasoning_format),
            reasoning_effort: self.reasoning_effort.or(b.reasoning_effort),
            include_reasoning: self.include_reasoning.or(b.include_reasoning),
            generation,
            stock_sampling: self.stock_sampling || b.stock_sampling,
        }
    }
}

/// vLLM `--reasoning-config`.
#[derive(Clone, Debug, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReasoningConfig {
    #[serde(default)]
    reasoning_parser: String,
    #[serde(default)]
    reasoning_start_str: String,
    #[serde(default)]
    reasoning_end_str: String,
}

/// The server-wide reasoning settings, from vLLM's, SGLang's and llama.cpp's own flag names.
#[derive(Clone, Debug, Default)]
pub struct ServerReasoning {
    /// The defaults every model starts from (its `--chat-defaults` entry goes on top).
    pub defaults: ChatDefaults,
    /// vLLM/SGLang `--reasoning-parser` (or `reasoning_config.reasoning_parser`).
    pub parser: Option<String>,
    /// vLLM `reasoning_config.reasoning_start_str`.
    pub start_str: Option<String>,
    /// vLLM `--chat-template`: a file, or the template itself.
    pub chat_template: Option<String>,
}

fn json_flag<T: serde::de::DeserializeOwned>(flag: &str, v: &str) -> Result<T, String> {
    serde_json::from_str(v).map_err(|e| format!("--{flag}: {e}"))
}

impl ServerReasoning {
    pub fn from_runtime(c: &crate::config::RuntimeConfig) -> Result<Self, String> {
        let mut d = ChatDefaults::default();
        let rc: ReasoningConfig = match &c.reasoning_config {
            Some(v) => json_flag("reasoning-config", v)?,
            None => ReasoningConfig::default(),
        };
        if let Some(v) = &c.default_chat_template_kwargs {
            d.chat_template_kwargs = Some(json_flag("default-chat-template-kwargs", v)?);
        }
        match c.reasoning.as_deref() {
            None | Some("auto") => {}
            Some(on @ ("on" | "off")) => {
                d.chat_template_kwargs
                    .get_or_insert_with(Default::default)
                    .entry("enable_thinking".into())
                    .or_insert((on == "on").into());
            }
            Some(other) => return Err(format!("--reasoning: {other:?} is not on, off or auto")),
        }
        let mut generation = match c.generation_config.as_deref() {
            None | Some("auto") => None,
            Some("vllm") => {
                d.stock_sampling = true;
                None
            }
            Some(dir) => {
                let p = Path::new(dir);
                let g = plow_asset::serve_manifest::read_sampling(p).ok_or_else(|| {
                    format!("--generation-config: no generation_config.json in {dir}")
                })?;
                d.stock_sampling = true;
                let max_new = std::fs::read_to_string(p.join("generation_config.json"))
                    .ok()
                    .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
                    .and_then(|v| v.get("max_new_tokens").and_then(serde_json::Value::as_u64));
                Some(GenerationDefaults {
                    temperature: g.temperature,
                    top_p: g.top_p,
                    top_k: g.top_k,
                    repetition_penalty: g.repetition_penalty,
                    max_new_tokens: max_new.map(|m| m as u32),
                    ..Default::default()
                })
            }
        };
        if let Some(v) = &c.override_generation_config {
            let o: GenerationDefaults = json_flag("override-generation-config", v)?;
            generation = Some(generation.map_or(o, |g| o.over(g)));
        }
        d.generation = generation;
        d.thinking_token_budget = c.reasoning_budget;
        d.reasoning_budget_message = c.reasoning_budget_message.clone();
        d.reasoning_end = c
            .reasoning_end
            .clone()
            .or((!rc.reasoning_end_str.is_empty()).then(|| rc.reasoning_end_str.clone()));
        d.reasoning_format = c.reasoning_format.clone();
        d.reasoning_effort = c.reasoning_effort.clone();
        d.max_reply_tokens = c.max_reply_tokens;
        if let Some(f) = d.reasoning_format.as_deref() {
            if !matches!(f, "none" | "auto" | "deepseek") {
                return Err(format!(
                    "--reasoning-format: {f:?} is not none, auto or deepseek"
                ));
            }
        }
        let parser = c
            .reasoning_parser
            .clone()
            .or((!rc.reasoning_parser.is_empty()).then_some(rc.reasoning_parser));
        if c.enable_reasoning {
            tracing::info!("--enable-reasoning (vLLM < 0.10): reasoning is always split here; the flag changes nothing");
        }
        Ok(ServerReasoning {
            defaults: d,
            parser,
            start_str: (!rc.reasoning_start_str.is_empty()).then_some(rc.reasoning_start_str),
            chat_template: c.chat_template.clone(),
        })
    }
}

/// The marker pair a vLLM / SGLang reasoning parser splits on, for the parsers whose framing is
/// checked against the model's template. Unlisted parsers are accepted as named: the split uses
/// the model's own markers either way.
fn parser_markers(name: &str) -> Option<(&'static str, &'static str)> {
    Some(match name {
        "deepseek_r1" | "deepseek-r1" | "deepseek_v3" | "deepseek-v3" | "deepseek_v4"
        | "deepseek_v41" | "qwen3" | "qwen3-thinking" | "glm45" | "glm47" | "kimi_k2" | "kimi"
        | "kimi_k3" | "minimax_m2" | "minimax_m3" | "minimax" | "step3" | "step3p5" | "olmo3"
        | "mimo" | "ling3" | "nemotron_v3" => ("<think>", "</think>"),
        "gemma4" => ("<|channel>", "<channel|>"),
        "mistral" => ("[THINK]", "[/THINK]"),
        "seed_oss" => ("<seed:think>", "</seed:think>"),
        _ => return None,
    })
}

/// Why `parser` / `start_str` contradict how `bundle`'s template frames its trace.
fn check_framing(
    bundle: &crate::asset::ModelBundle,
    parser: Option<&str>,
    start_str: Option<&str>,
) -> Result<(), String> {
    let markers = bundle.reasoning().markers();
    let harmony = bundle.chat_template().is_some_and(|t| {
        t.tools
            == crate::serve::tools::ToolSupport::Format(crate::serve::tools::ToolFormat::Harmony)
    });
    if let Some(p) = parser {
        match (parser_markers(p), markers) {
            _ if p == "openai_gptoss" || p == "gpt-oss" => {
                if !harmony {
                    return Err(format!("--reasoning-parser {p}: the model's template is not gpt-oss harmony"));
                }
            }
            (Some((open, close)), Some((mo, mc))) => {
                if !(mo.starts_with(open) && mc == close) {
                    return Err(format!(
                        "--reasoning-parser {p} splits on {open}…{close}; the model's template frames its trace with {mo}…{mc}"
                    ));
                }
            }
            (Some((open, close)), None) => {
                return Err(format!(
                    "--reasoning-parser {p} splits on {open}…{close}; the model's template frames no reasoning trace"
                ))
            }
            (None, _) => tracing::info!(
                parser = p,
                markers = ?markers,
                "--reasoning-parser: not checked against the template; the model's own markers split the trace"
            ),
        }
    }
    if let Some(s) = start_str {
        if !markers.is_some_and(|(open, _)| open.starts_with(s) || s.starts_with(open)) {
            return Err(format!(
                "--reasoning-config reasoning_start_str {s:?} is not the model's open marker {:?}",
                markers.map_or("", |m| m.0)
            ));
        }
    }
    Ok(())
}

/// Why `d` cannot apply to `bundle`; `bad(field, why)` names the source.
fn check_defaults(
    d: &ChatDefaults,
    bundle: &crate::asset::ModelBundle,
    bad: &dyn Fn(&str, String) -> String,
) -> Result<(), String> {
    let markers = bundle.reasoning().markers();
    check_thinking(markers, d.thinking_token_budget, d.reasoning_end.as_deref())
        .map_err(|(f, e)| bad(f, e))?;
    if let Some(p) = &d.thought_prefix {
        if !markers.is_some_and(|(open, _)| p.starts_with(open)) {
            return Err(bad(
                "thought_prefix",
                format!(
                    "must start with the open marker {:?}",
                    markers.map_or("", |m| m.0)
                ),
            ));
        }
    }
    if d.reply_regenerate && d.reply_guard.is_none() {
        return Err(bad("reply_regenerate", "set without `reply_guard`".into()));
    }
    if d.reply_guard_lines.is_some() && d.reply_guard.is_none() {
        return Err(bad("reply_guard_lines", "set without `reply_guard`".into()));
    }
    if let Some(f) = d.reasoning_format.as_deref() {
        if !matches!(f, "none" | "auto" | "deepseek") {
            return Err(bad(
                "reasoning_format",
                format!("{f:?} is not none, auto or deepseek"),
            ));
        }
    }
    let need_ids = [
        ("reply_stop_blank_line", d.reply_stop_blank_line),
        ("live_reasoning_end", d.live_reasoning_end),
        (
            "thinking_token_budget",
            d.thinking_token_budget.is_some_and(|b| b > 0),
        ),
        ("max_reply_tokens", d.max_reply_tokens.is_some()),
    ];
    for (field, on) in need_ids {
        if on {
            bundle.think_markers().map_err(|e| bad(field, e))?;
        }
    }
    if d.steers() {
        if !cfg!(feature = "cuda") {
            return Err(bad(
                "reply_first_bias",
                "host-logit steering needs the CUDA engine (other engines never hand the host a \
                 logit row)"
                    .into(),
            ));
        }
        bundle
            .think_markers()
            .map_err(|e| bad("reply_first_bias", e))?;
        d.bias(bundle.tokenizer().as_ref(), &[])
            .map_err(|e| bad("reply_first_bias", e))?;
        if d.close_ramp.is_some() && d.thinking_token_budget.is_none_or(|b| b <= 0) {
            return Err(bad(
                "close_ramp",
                "needs a `thinking_token_budget` to ramp toward".into(),
            ));
        }
    }
    if d.max_reply_tokens == Some(0) {
        return Err(bad("max_reply_tokens", "must be at least 1".into()));
    }
    Ok(())
}

/// Give every served model its request defaults at startup: the server flags (`server`), then its
/// `--chat-defaults` entry (`file`) on top; and check the reasoning parser, start marker and
/// template override against each model. `models` lists the served models (canonical name and
/// bundle); `resolve` maps a file key (a served name or alias) to one. An unknown key, model or
/// contradiction is an error, not a no-op.
pub fn install_chat_defaults(
    server: &ServerReasoning,
    file: Option<&Path>,
    models: &[(String, std::sync::Arc<crate::asset::ModelBundle>)],
    resolve: impl Fn(&str) -> Option<(String, std::sync::Arc<crate::asset::ModelBundle>)>,
) -> Result<(), String> {
    let mut entries = std::collections::HashMap::new();
    if let Some(path) = file {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let parsed: std::collections::HashMap<String, ChatDefaults> =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        for (name, d) in parsed {
            let (canonical, _) = resolve(&name)
                .ok_or_else(|| format!("{}: no served model is named {name:?}", path.display()))?;
            if entries.insert(canonical.clone(), (name, d)).is_some() {
                return Err(format!(
                    "{}: {canonical} has defaults under two names",
                    path.display()
                ));
            }
        }
    }
    for (canonical, bundle) in models {
        check_framing(
            bundle,
            server.parser.as_deref(),
            server.start_str.as_deref(),
        )
        .map_err(|e| format!("{canonical}: {e}"))?;
        if let Some(t) = &server.chat_template {
            bundle
                .override_chat_template(t)
                .map_err(|e| format!("--chat-template: {canonical}: {e}"))?;
        }
        let (d, source) = match entries.remove(canonical) {
            Some((name, d)) => (
                d.over(&server.defaults),
                format!(
                    "{}: {name}",
                    file.map_or(String::new(), |p| p.display().to_string())
                ),
            ),
            None => (server.defaults.clone(), "server flags".to_string()),
        };
        check_defaults(&d, bundle, &|field, why| {
            format!("{source}: {field}: {why}")
        })?;
        tracing::info!(
            model = %canonical,
            from = %source,
            max_tokens = ?d.max_tokens.or(d.generation.and_then(|g| g.max_new_tokens)),
            thinking_token_budget = ?d.thinking_token_budget,
            max_reply_tokens = ?d.max_reply_tokens,
            newline_grace = d.newline_grace,
            reasoning_end = ?d.reasoning_end,
            reasoning_budget_message = ?d.reasoning_budget_message,
            reasoning_format = ?d.reasoning_format,
            reasoning_effort = ?d.reasoning_effort,
            include_reasoning = ?d.include_reasoning,
            chat_template_kwargs = ?d.chat_template_kwargs,
            generation = ?d.generation,
            stock_sampling = d.stock_sampling,
            reply_guard = ?d.reply_guard,
            thought_prefix = ?d.thought_prefix,
            "chat defaults"
        );
        let _ = bundle.serving().chat.set(d);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime(args: &[&str]) -> crate::config::RuntimeConfig {
        #[derive(clap::Parser)]
        struct Cli {
            #[command(flatten)]
            rt: crate::config::RuntimeConfig,
        }
        <Cli as clap::Parser>::try_parse_from(std::iter::once("plowrt").chain(args.iter().copied()))
            .unwrap()
            .rt
    }

    /// vLLM's reasoning and generation flags and llama.cpp's reasoning flags parse as written in
    /// those engines' command lines and become the server-wide defaults.
    #[test]
    fn vllm_and_llamacpp_reasoning_flags_become_server_defaults() {
        let vllm = runtime(&[
            "--reasoning-parser",
            "qwen3",
            "--reasoning-config",
            r#"{"reasoning_start_str": "<think>", "reasoning_end_str": "Answering now.</think>"}"#,
            "--default-chat-template-kwargs",
            r#"{"enable_thinking": true}"#,
            "--override-generation-config",
            r#"{"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 512}"#,
            "--generation-config",
            "vllm",
            "--enable-reasoning",
        ]);
        let s = ServerReasoning::from_runtime(&vllm).unwrap();
        assert_eq!(s.parser.as_deref(), Some("qwen3"));
        assert_eq!(s.start_str.as_deref(), Some("<think>"));
        assert_eq!(
            s.defaults.reasoning_end.as_deref(),
            Some("Answering now.</think>")
        );
        assert_eq!(
            s.defaults.chat_template_kwargs.as_ref().unwrap()["enable_thinking"],
            true
        );
        let g = s.defaults.generation.unwrap();
        assert_eq!(
            (g.temperature, g.top_p, g.max_new_tokens),
            (Some(0.6), Some(0.95), Some(512))
        );
        assert!(s.defaults.stock_sampling);

        let llama = runtime(&[
            "--reasoning-budget",
            "128",
            "--reasoning-budget-message",
            "Answering now.",
            "--reasoning-format",
            "deepseek",
            "--reasoning",
            "off",
            "--chat-template-kwargs",
            r#"{"foo": 1}"#,
            "--reasoning-effort",
            "low",
            "--max-reply-tokens",
            "64",
        ]);
        let d = ServerReasoning::from_runtime(&llama).unwrap().defaults;
        assert_eq!(d.thinking_token_budget, Some(128));
        assert_eq!(
            d.reasoning_budget_message.as_deref(),
            Some("Answering now.")
        );
        assert_eq!(d.reasoning_format.as_deref(), Some("deepseek"));
        assert_eq!(d.reasoning_effort.as_deref(), Some("low"));
        assert_eq!(d.max_reply_tokens, Some(64));
        let kw = d.chat_template_kwargs.unwrap();
        assert_eq!(
            (kw["enable_thinking"].clone(), kw["foo"].clone()),
            (false.into(), 1.into())
        );
        // -1 is llama.cpp's "unrestricted"; the plow alias parses the same
        let d = ServerReasoning::from_runtime(&runtime(&["--thinking-token-budget", "-1"]))
            .unwrap()
            .defaults;
        assert_eq!(d.thinking_token_budget, Some(-1));
        for bad in [
            &["--reasoning", "maybe"][..],
            &["--reasoning-format", "deepseek-legacy"],
            &[
                "--reasoning-config",
                r#"{"reasoning_parser": "qwen3", "bogus": 1}"#,
            ],
            &["--override-generation-config", r#"{"do_sample": true}"#],
        ] {
            assert!(
                ServerReasoning::from_runtime(&runtime(bad)).is_err(),
                "{bad:?}"
            );
        }
    }

    /// A model's file entry goes over the server flags field by field; kwargs merge by key.
    #[test]
    fn file_defaults_go_over_server_flags() {
        let server = ChatDefaults {
            max_tokens: Some(100),
            thinking_token_budget: Some(64),
            chat_template_kwargs: Some(
                [
                    ("enable_thinking".to_string(), true.into()),
                    ("a".to_string(), 1.into()),
                ]
                .into(),
            ),
            generation: Some(GenerationDefaults {
                temperature: Some(0.6),
                top_p: Some(0.9),
                ..Default::default()
            }),
            ..Default::default()
        };
        let file = ChatDefaults {
            max_tokens: Some(450),
            chat_template_kwargs: Some([("a".to_string(), 2.into())].into()),
            generation: Some(GenerationDefaults {
                temperature: Some(1.0),
                ..Default::default()
            }),
            ..Default::default()
        };
        let d = file.over(&server);
        assert_eq!(
            (d.max_tokens, d.thinking_token_budget),
            (Some(450), Some(64))
        );
        let kw = d.chat_template_kwargs.unwrap();
        assert_eq!(
            (kw["a"].clone(), kw["enable_thinking"].clone()),
            (2.into(), true.into())
        );
        let g = d.generation.unwrap();
        assert_eq!((g.temperature, g.top_p), (Some(1.0), Some(0.9)));
    }

    #[test]
    fn a_checkpoint_without_a_generation_config_keeps_the_stock_defaults() {
        let cfg = ServingConfig::load(Path::new("/nonexistent"));
        assert_eq!(cfg.default_sampling.temperature, 1.0);
        assert_eq!(cfg.default_sampling.top_p, 1.0);
    }

    #[test]
    fn the_checkpoints_sampling_defaults_are_read() {
        let dir = std::env::temp_dir().join("plowrt-servingconfig-test");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(
            dir.join("generation_config.json"),
            r#"{"temperature": 0.6, "top_p": 0.95, "top_k": 20}"#,
        )
        .expect("write");
        let cfg = ServingConfig::load(&dir);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(cfg.default_sampling.temperature, 0.6);
        assert_eq!(cfg.default_sampling.top_p, 0.95);
        assert_eq!(cfg.default_sampling.top_k, 20);
    }
}
