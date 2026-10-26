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

/// Read the defaults file once at startup and give each model its entry (`resolve` maps a served
/// name or alias to the model's canonical name and bundle). An unknown key or model is an error,
/// not a no-op.
pub fn load_chat_defaults(
    path: &Path,
    resolve: impl Fn(&str) -> Option<(String, std::sync::Arc<crate::asset::ModelBundle>)>,
) -> Result<(), String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let file: std::collections::HashMap<String, ChatDefaults> =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    for (name, d) in file {
        let (canonical, bundle) = resolve(&name)
            .ok_or_else(|| format!("{}: no served model is named {name:?}", path.display()))?;
        let markers = bundle.reasoning().markers();
        let bad = |field: &str, why: String| format!("{}: {name}.{field}: {why}", path.display());
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
        if d.reply_stop_blank_line {
            bundle
                .think_markers()
                .map_err(|e| bad("reply_stop_blank_line", e))?;
        }
        if d.steers() {
            if !cfg!(feature = "cuda") {
                return Err(bad(
                    "reply_first_bias",
                    "host-logit steering needs the CUDA engine (other engines never hand the \
                     host a logit row)"
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
        if d.live_reasoning_end {
            bundle
                .think_markers()
                .map_err(|e| bad("live_reasoning_end", e))?;
        }
        if d.max_reply_tokens == Some(0) {
            return Err(bad("max_reply_tokens", "must be at least 1".into()));
        }
        if d.thinking_token_budget.is_some_and(|b| b > 0) || d.max_reply_tokens.is_some() {
            bundle
                .think_markers()
                .map_err(|e| bad("thinking_token_budget", e))?;
        }
        tracing::info!(
            model = %canonical,
            max_tokens = ?d.max_tokens,
            thinking_token_budget = ?d.thinking_token_budget,
            max_reply_tokens = ?d.max_reply_tokens,
            newline_grace = d.newline_grace,
            reasoning_end = ?d.reasoning_end,
            reply_guard = ?d.reply_guard,
            reply_stop_blank_line = d.reply_stop_blank_line,
            thought_prefix = ?d.thought_prefix,
            chat_template_kwargs = ?d.chat_template_kwargs,
            "chat defaults"
        );
        bundle.serving().chat.set(d).map_err(|_| {
            format!(
                "{}: {canonical} has defaults under two names",
                path.display()
            )
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
