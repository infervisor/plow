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
//!
//! The reasoning mode is here for the same reason: whether a generation prompt
//! opens a thinking trace is a property of the checkpoint's template, not of
//! the prompt bytes, and deciding it by searching the rendered prompt let USER
//! TEXT flip it.

use std::path::Path;

use crate::text::sample::SamplingParams;

/// How a model frames a reasoning trace, so the server can put the trace in
/// `reasoning_content` and the answer in `content`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReasoningMode {
    /// No separable trace. Everything generated is the answer.
    #[default]
    None,
    /// The generation prompt leaves `<think>` OPEN and the model closes it.
    /// GLM-5.2/5.3, Qwen3 and DeepSeek-R1 all frame it this way.
    ThinkTag,
}

impl ReasoningMode {
    /// The marker that ENDS the trace.
    pub fn close_marker(self) -> Option<&'static str> {
        match self {
            ReasoningMode::None => None,
            ReasoningMode::ThinkTag => Some("</think>"),
        }
    }

    /// The framing implied by a rendered prompt — i.e. does generation begin
    /// INSIDE a trace, and if so which marker closes it.
    ///
    /// Decided from the prompt's SUFFIX, never a search of it.
    /// `add_generation_prompt` puts the generation prompt last, so only a
    /// marker at the very end is the model's own. The previous probe was
    /// `prompt.rfind("<think>")` over the whole rendered prompt — which
    /// contains USER TEXT — so a user message carrying `<think>` with no later
    /// `</think>` routed the entire answer into `reasoning_content` and
    /// returned an EMPTY `content`, on any model, reasoning or not.
    ///
    /// Per REQUEST, not per model, and deliberately: it is the rendered prompt
    /// that decides, so this is right for the checkpoint's own template, for
    /// the built-in builders when a checkpoint ships none, and for a request
    /// that turned thinking off with `chat_template_kwargs` (which renders the
    /// pair CLOSED, `<think></think>`, and so reads as `None` here).
    pub fn detect(prompt: &str) -> Self {
        if prompt.ends_with("<think>") {
            ReasoningMode::ThinkTag
        } else {
            ReasoningMode::None
        }
    }

    /// Whether generation begins inside a trace.
    pub fn opens(self) -> bool {
        self != ReasoningMode::None
    }
}

/// Everything per-model the request path needs that is not the weights.
#[derive(Clone, Debug)]
pub struct ServingConfig {
    /// The checkpoint's own sampling defaults. A request field overrides one of
    /// these; a request that omits it gets the checkpoint's value, NOT the
    /// process-wide default.
    pub default_sampling: SamplingParams,
}

impl Default for ServingConfig {
    fn default() -> Self {
        ServingConfig {
            default_sampling: SamplingParams::default(),
        }
    }
}

impl ServingConfig {
    /// Read `generation_config.json` from `dir` or `dir/checkpoint` — the same
    /// two places [`crate::asset::checkpoint`] already looks for the eos set.
    ///
    /// Absent file, absent key or unparseable value each leave the
    /// corresponding default alone; a checkpoint that says nothing about
    /// sampling is served exactly as before.
    pub fn load(dir: &Path) -> Self {
        let mut cfg = ServingConfig::default();
        let Some(v) = [dir.join("generation_config.json"), dir.join("checkpoint").join("generation_config.json")]
            .iter()
            .find_map(|p| {
                std::fs::read_to_string(p)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            })
        else {
            return cfg;
        };
        let p = &mut cfg.default_sampling;
        if let Some(t) = v.get("temperature").and_then(serde_json::Value::as_f64) {
            p.temperature = t as f32;
        }
        if let Some(t) = v.get("top_p").and_then(serde_json::Value::as_f64) {
            p.top_p = t as f32;
        }
        // HF spells "no top-k" as 0 or as a missing key; the sampler spells it 0.
        if let Some(k) = v.get("top_k").and_then(serde_json::Value::as_u64) {
            p.top_k = k as usize;
        }
        if let Some(r) = v
            .get("repetition_penalty")
            .and_then(serde_json::Value::as_f64)
        {
            p.repetition_penalty = r as f32;
        }
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE BUG THIS EXISTS TO KILL. A user message mentioning `<think>` used to
    /// make the server treat the model's whole answer as a reasoning trace and
    /// return an empty `content`.
    #[test]
    fn user_text_carrying_the_marker_cannot_open_a_trace() {
        let prompt = "<|user|>what does <think> do?<|assistant|>";
        assert_eq!(ReasoningMode::detect(prompt), ReasoningMode::None);
        assert!(!ReasoningMode::detect(prompt).opens());
    }

    /// GLM's real generation prompt leaves the block open and must still be
    /// recognised — the fix must not trade one silent failure for another.
    #[test]
    fn a_generation_prompt_that_really_opens_one_is_recognised() {
        let p = "<|user|>hi<|assistant|><think>";
        assert_eq!(ReasoningMode::detect(p), ReasoningMode::ThinkTag);
        assert_eq!(ReasoningMode::detect(p).close_marker(), Some("</think>"));
    }

    /// `enable_thinking: false` renders the pair CLOSED, and that is the whole
    /// reason the mode is decided per request rather than per model.
    #[test]
    fn thinking_turned_off_renders_as_no_trace() {
        assert_eq!(
            ReasoningMode::detect("<|user|>hi<|assistant|><think></think>"),
            ReasoningMode::None
        );
    }

    /// Non-reasoning families end on an ordinary assistant turn.
    #[test]
    fn ordinary_generation_prompts_have_no_trace() {
        assert_eq!(
            ReasoningMode::detect("<start_of_turn>model\n"),
            ReasoningMode::None
        );
        // gpt-oss's built-in builder pins the FINAL channel: reasoning is off,
        // and nothing here may claim otherwise.
        assert_eq!(
            ReasoningMode::detect("<|start|>assistant<|channel|>final<|message|>"),
            ReasoningMode::None
        );
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
