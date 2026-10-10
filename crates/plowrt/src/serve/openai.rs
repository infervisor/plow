//! §G OpenAI-compatible request/response DTOs.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The sampling knobs both endpoints accept, in their OpenAI/vLLM wire form.
///
/// These are `flatten`ed into both request bodies rather than declared twice.
/// Every one of them was previously dropped by serde as an unknown field: the
/// host sampler has implemented `top_k`/`min_p`/`repetition_penalty`/
/// `logit_bias` all along, and a request asking for them was answered under
/// different sampling than it requested, with a 200 and no warning. That is the
/// same silent-wrong-answer class the `tools`/`n` refusals already guard.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct SamplingFields {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// vLLM spells "disabled" as `-1`; this server's sampler spells it `0`.
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// OpenAI `logit_bias`: token id (as a STRING key, per the schema) -> bias.
    #[serde(default)]
    pub logit_bias: Option<BTreeMap<String, f32>>,
    /// vLLM extension: refuse to stop before this many tokens.
    #[serde(default)]
    pub min_tokens: Option<u32>,
    /// vLLM extension: extra ids that end generation.
    #[serde(default)]
    pub stop_token_ids: Option<Vec<u32>>,
}

/// A rejected parameter: the field name and why.
pub struct ParamError {
    pub field: &'static str,
    pub message: String,
}

fn range(
    v: Option<f32>,
    field: &'static str,
    lo: f32,
    hi: f32,
    lo_open: bool,
) -> Result<(), ParamError> {
    let Some(v) = v else { return Ok(()) };
    let low_ok = if lo_open { v > lo } else { v >= lo };
    if !v.is_finite() || !low_ok || v > hi {
        return Err(ParamError {
            field,
            message: format!(
                "`{field}` must be in {}{lo}, {hi}]; got {v}",
                if lo_open { "(" } else { "[" }
            ),
        });
    }
    Ok(())
}

impl SamplingFields {
    /// Range-check every field. OpenAI answers 400 for an out-of-range value;
    /// this server used to accept them and sample from the result — `top_p: 0`
    /// truncates the candidate set to nothing, and a negative `temperature`
    /// falls through the greedy branch into a scaled softmax with an inverted
    /// sign.
    pub fn validate(&self) -> Result<(), ParamError> {
        range(self.temperature, "temperature", 0.0, 2.0, false)?;
        range(self.top_p, "top_p", 0.0, 1.0, true)?;
        range(self.min_p, "min_p", 0.0, 1.0, false)?;
        range(self.repetition_penalty, "repetition_penalty", 0.0, 2.0, true)?;
        range(self.presence_penalty, "presence_penalty", -2.0, 2.0, false)?;
        range(self.frequency_penalty, "frequency_penalty", -2.0, 2.0, false)?;
        if let Some(k) = self.top_k {
            if k < -1 {
                return Err(ParamError {
                    field: "top_k",
                    message: format!("`top_k` must be -1 (disabled) or >= 0; got {k}"),
                });
            }
        }
        if self.stop_token_ids.as_ref().is_some_and(|ids| ids.len() > MAX_STOP_TOKEN_IDS) {
            return Err(ParamError {
                field: "stop_token_ids",
                message: format!("at most {MAX_STOP_TOKEN_IDS} `stop_token_ids` are accepted"),
            });
        }
        for (k, v) in self.logit_bias.iter().flatten() {
            if k.parse::<u32>().is_err() {
                return Err(ParamError {
                    field: "logit_bias",
                    message: format!("`logit_bias` keys are token ids; `{k}` is not one"),
                });
            }
            if !v.is_finite() || !(-100.0..=100.0).contains(v) {
                return Err(ParamError {
                    field: "logit_bias",
                    message: format!("`logit_bias` values must be in [-100, 100]; got {v}"),
                });
            }
        }
        Ok(())
    }

    /// Overlay the request's knobs onto `params`, which arrives carrying the
    /// MODEL's defaults. A field the request omits keeps the model default —
    /// that ordering is the whole point, and is why this takes `&mut` rather
    /// than building a fresh `SamplingParams`.
    pub fn apply(&self, params: &mut crate::text::sample::SamplingParams) {
        if let Some(v) = self.temperature {
            params.temperature = v;
        }
        if let Some(v) = self.top_p {
            params.top_p = v;
        }
        if let Some(v) = self.top_k {
            params.top_k = v.max(0) as usize;
        }
        if let Some(v) = self.min_p {
            params.min_p = v;
        }
        if let Some(v) = self.repetition_penalty {
            params.repetition_penalty = v;
        }
        if let Some(v) = self.presence_penalty {
            params.presence_penalty = v;
        }
        if let Some(v) = self.frequency_penalty {
            params.frequency_penalty = v;
        }
        if let Some(b) = &self.logit_bias {
            params.logit_bias = b
                .iter()
                .filter_map(|(k, v)| k.parse::<u32>().ok().map(|t| (t, *v)))
                .collect();
        }
    }
}

/// `POST /v1/chat/completions` request body (subset).
#[derive(Clone, Debug, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    /// `session_id` / `prompt_cache_key` / `metadata` / `trace` ([`crate::serve::session::RequestIds::apply_body`]).
    #[serde(flatten)]
    pub route: crate::serve::session::RouteFields,
    #[serde(default)]
    pub stream: bool,
    /// OpenAI renamed this field to `max_completion_tokens` for chat
    /// completions and deprecated `max_tokens`; current clients (including
    /// `vllm bench serve --backend openai-chat`) send only the new name.
    /// Accept both, or the cap is silently ignored and generation runs to EOS.
    #[serde(default, alias = "max_completion_tokens")]
    pub max_tokens: Option<u32>,
    #[serde(flatten)]
    pub sampling: SamplingFields,
    /// vLLM extension: run to `max_tokens` instead of stopping at eos. Sent by
    /// `vllm bench serve` for the synthetic datasets; ignoring it makes every
    /// benchmark against plowrt under-report throughput. See `GenParams`.
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// OpenAI `stop`: a string or up to four strings. APPLIED — generation ends
    /// at the first match and the matched text is withheld from the output.
    #[serde(default)]
    pub stop: Option<StopSpec>,
    /// Mixed into the sampling RNG so a run is reproducible on request.
    #[serde(default)]
    pub seed: Option<u64>,
    /// vLLM's escape hatch into the checkpoint's own template. `{"enable_thinking":
    /// false}` is how every Qwen3/GLM client turns reasoning off; dropping it
    /// meant plowrt served a thinking trace where the same request to vLLM did
    /// not — which is exactly how `scripts/bench_vllm_rocm.sh` drives vLLM, so
    /// the two sides of that A/B were not running the same prompt.
    #[serde(default)]
    pub chat_template_kwargs: Option<BTreeMap<String, serde_json::Value>>,
    /// OpenAI's reasoning knob. Passed into the template as `reasoning_effort`,
    /// which is the name the checkpoints that read it use.
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    /// vLLM extension: render the final assistant turn as a prefix to continue
    /// rather than closing it and opening a new one.
    #[serde(default)]
    pub continue_final_message: Option<bool>,
    /// Per-request template override, for checkpoints that ship none.
    #[serde(default)]
    pub chat_template: Option<String>,
    // The rest are parsed ONLY so the handler can REFUSE them. Serde drops
    // unknown fields silently, and a silently dropped `tools` or `n` is a
    // wrong answer that scores as a successful request — the failure mode the
    // image refusal already guards against.
    /// OpenAI `logprobs` (bool) / `top_logprobs` (0..=20); see `serve::logprobs`.
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
    #[serde(default)]
    pub top_logprobs: Option<serde_json::Value>,
    /// vLLM names: `"raw_logprobs"` (default) or `"raw_logits"`; per request here.
    #[serde(default)]
    pub logprobs_mode: Option<String>,
    #[serde(default)]
    pub return_tokens_as_token_ids: Option<bool>,
    #[serde(default)]
    pub n: Option<u32>,
    /// OpenAI tool calling; validated and mapped by `serve::tools::request`.
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Deprecated OpenAI function calling: refused in favour of `tools`.
    #[serde(default)]
    pub functions: Option<serde_json::Value>,
    #[serde(default)]
    pub function_call: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// vLLM's `include_reasoning`: `false` leaves the trace out of the response (it is still
    /// split from `content` and counted in `reasoning_tokens`).
    #[serde(default)]
    pub include_reasoning: Option<bool>,
}

/// Stop matching runs on the serialized dispatcher for every generated token, in time
/// proportional to the number and length of the stop strings, so a single request with
/// unbounded ones stalls decoding for every request on its model.
pub const MAX_STOP_STRINGS: usize = 64;
pub const MAX_STOP_STRING_BYTES: usize = 1024;
pub const MAX_STOP_TOKEN_IDS: usize = 1024;

/// Request limits shared by chat and completions: a zero token budget and stop sets the
/// dispatcher cannot afford to scan per token are refused with a 400.
pub fn validate_limits(max_tokens: Option<u32>, stop: Option<&StopSpec>) -> Result<(), ParamError> {
    if max_tokens == Some(0) {
        return Err(ParamError { field: "max_tokens", message: "`max_tokens` must be at least 1".into() });
    }
    let stops: &[String] = match stop {
        Some(StopSpec::One(s)) => std::slice::from_ref(s),
        Some(StopSpec::Many(v)) => v,
        None => &[],
    };
    if stops.len() > MAX_STOP_STRINGS {
        return Err(ParamError { field: "stop", message: format!("at most {MAX_STOP_STRINGS} `stop` strings are accepted") });
    }
    if stops.iter().any(|s| s.len() > MAX_STOP_STRING_BYTES) {
        return Err(ParamError {
            field: "stop",
            message: format!("each `stop` string must be at most {MAX_STOP_STRING_BYTES} bytes"),
        });
    }
    Ok(())
}

/// OpenAI `stop`: the wire form is a bare string or an array of them.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StopSpec {
    One(String),
    Many(Vec<String>),
}

impl StopSpec {
    /// The stop strings, empty ones dropped (an empty stop would halt at once).
    pub fn list(&self) -> Vec<String> {
        match self {
            StopSpec::One(s) => vec![s.clone()],
            StopSpec::Many(v) => v.clone(),
        }
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect()
    }
}

/// `POST /v1/completions` request body (supported subset).
#[derive(Clone, Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    #[serde(flatten)]
    pub route: crate::serve::session::RouteFields,
    /// OpenAI allows `str | [str] | [int] | [[int]]`. A bare `String` here made
    /// `client.completions.create(prompt=[...])` fail with a 422 before any
    /// handler ran, so accept the array forms and refuse BATCHES explicitly.
    pub prompt: PromptSpec,
    #[serde(default = "default_add_special_tokens")]
    pub add_special_tokens: bool,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(flatten)]
    pub sampling: SamplingFields,
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Plow parity extension. Available only for non-streaming requests.
    #[serde(default)]
    pub return_token_ids: bool,
    #[serde(default)]
    pub stop: Option<StopSpec>,
    #[serde(default)]
    pub seed: Option<u64>,
    // Parsed to be REFUSED, as on the chat request.
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub best_of: Option<u32>,
    #[serde(default)]
    pub echo: Option<bool>,
    /// OpenAI `logprobs`: alternatives per position, 0..=20; see `serve::logprobs`.
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
    #[serde(default)]
    pub logprobs_mode: Option<String>,
    #[serde(default)]
    pub return_tokens_as_token_ids: Option<bool>,
    #[serde(default)]
    pub suffix: Option<String>,
}

/// A `/v1/completions` prompt in any of OpenAI's four wire forms.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PromptSpec {
    Text(String),
    Tokens(Vec<u32>),
    Batch(Vec<String>),
    TokenBatch(Vec<Vec<u32>>),
}

fn default_add_special_tokens() -> bool {
    true
}

/// OpenAI `stream_options`: `include_usage` opts the stream into a final
/// usage-only chunk (empty `choices`) before `[DONE]`.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// A request message.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct Message {
    pub role: String,
    /// OPTIONAL. An assistant turn that carried a tool call has `content: null`,
    /// and a non-`Option` field here made replaying such a conversation fail
    /// with a 422 inside the extractor, before the handler could say why.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    /// The model's thinking trace, split out of the raw generation so that
    /// `content` carries the ANSWER alone. GLM's generation prompt leaves
    /// `<think>` open and `</think>` is not a special token, so without this
    /// the whole trace landed in `content` as literal text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// An assistant turn's calls, validated by `serve::tools::request`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// A `role: "tool"` result's call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// The assistant message of a non-streamed response. `content` is `null` (not absent) when the
/// turn is only tool calls, as OpenAI returns it.
#[derive(Clone, Debug, Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionCall,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FunctionCall {
    pub name: String,
    /// A JSON object, serialized.
    pub arguments: String,
}

/// One streamed `delta.tool_calls` entry.
#[derive(Clone, Debug, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    pub function: FunctionDelta,
}

#[derive(Clone, Debug, Serialize)]
pub struct FunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

impl Message {
    /// The message's text, empty when absent.
    pub fn text(&self) -> String {
        self.content.as_ref().map(Content::as_text).unwrap_or_default()
    }

    /// Whether this message carries an image part.
    pub fn has_image(&self) -> bool {
        self.content.as_ref().is_some_and(Content::has_image)
    }

    /// Whether this message carries an image or audio part.
    pub fn has_media(&self) -> bool {
        self.content.as_ref().is_some_and(Content::has_media)
    }

    /// The content as the chat template sees it: the text, or, when the message carries media,
    /// its parts with each image as `{"type": "image"}` and each audio clip as `{"type": "audio"}`
    /// (the template renders the model's placeholder for each, in order).
    pub fn template_content(&self) -> serde_json::Value {
        match &self.content {
            Some(c @ Content::Parts(parts)) if c.has_media() => serde_json::Value::Array(
                parts
                    .iter()
                    .map(|p| match p.media_kind() {
                        Some(kind) => serde_json::json!({ "type": kind }),
                        None => serde_json::json!({ "type": "text", "text": p.text().unwrap_or_default() }),
                    })
                    .collect(),
            ),
            _ => serde_json::Value::String(self.text()),
        }
    }
}

/// Message content: a plain string or multimodal parts (`text` / `image_url`).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Content {
    /// Concatenate text parts (drops non-text for the text pipeline).
    pub fn as_text(&self) -> String {
        match self {
            Content::Text(s) => s.clone(),
            Content::Parts(parts) => parts
                .iter()
                .filter_map(ContentPart::text)
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Whether any part is an image (routes through the vision stage).
    pub fn has_image(&self) -> bool {
        matches!(self, Content::Parts(parts) if parts.iter().any(|p| p.media_kind() == Some("image")))
    }

    /// Whether any part is an image or audio clip.
    pub fn has_media(&self) -> bool {
        matches!(self, Content::Parts(parts) if parts.iter().any(|p| p.media_kind().is_some()))
    }
}

/// A chat content part. Images: Chat Completions `image_url` (`{"url": ...}`) and Responses
/// `input_image` (`"image_url": "..."`); audio: `input_audio` (`{"data": base64, "format"}`) and
/// `audio_url` (`{"url": ...}`).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    InputText { text: String },
    ImageUrl { image_url: ImageUrl },
    InputImage { image_url: String },
    InputAudio { input_audio: InputAudio },
    AudioUrl { audio_url: ImageUrl },
}

impl ContentPart {
    pub fn text(&self) -> Option<&str> {
        match self {
            ContentPart::Text { text } | ContentPart::InputText { text } => Some(text),
            _ => None,
        }
    }

    /// `image` / `audio` for a media part.
    pub fn media_kind(&self) -> Option<&'static str> {
        match self {
            ContentPart::ImageUrl { .. } | ContentPart::InputImage { .. } => Some("image"),
            ContentPart::InputAudio { .. } | ContentPart::AudioUrl { .. } => Some("audio"),
            ContentPart::Text { .. } | ContentPart::InputText { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct InputAudio {
    pub data: String,
    #[serde(default)]
    pub format: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_sets_and_zero_budget_are_bounded() {
        assert!(validate_limits(Some(1), Some(&StopSpec::Many(vec!["a".into(); MAX_STOP_STRINGS]))).is_ok());
        assert_eq!(validate_limits(Some(0), None).err().map(|e| e.field), Some("max_tokens"));
        let many = StopSpec::Many(vec!["a".into(); MAX_STOP_STRINGS + 1]);
        assert_eq!(validate_limits(None, Some(&many)).err().map(|e| e.field), Some("stop"));
        let long = StopSpec::One("x".repeat(MAX_STOP_STRING_BYTES + 1));
        assert_eq!(validate_limits(None, Some(&long)).err().map(|e| e.field), Some("stop"));
        let ids = SamplingFields { stop_token_ids: Some(vec![1; MAX_STOP_TOKEN_IDS + 1]), ..Default::default() };
        assert_eq!(ids.validate().err().map(|e| e.field), Some("stop_token_ids"));
    }

    /// THE POINT OF `SamplingFields`, asserted end to end: the wire form has to
    /// DESERIALIZE and then actually reach `SamplingParams`. A test that only
    /// checks the request is accepted passes just as happily when every value
    /// is thrown away, which is the bug this replaced.
    #[test]
    fn every_sampling_knob_survives_the_wire_and_reaches_the_sampler() {
        let req: ChatRequest = serde_json::from_str(
            r#"{
                "model": "m",
                "messages": [],
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.05,
                "repetition_penalty": 1.1,
                "presence_penalty": 0.25,
                "frequency_penalty": 0.5,
                "logit_bias": {"7": -3.5},
                "min_tokens": 3,
                "stop_token_ids": [11, 22]
            }"#,
        )
        .expect("the flattened sampling block deserializes");

        let mut p = crate::text::sample::SamplingParams::default();
        req.sampling.apply(&mut p);
        assert_eq!(p.temperature, 0.6);
        assert_eq!(p.top_p, 0.95);
        assert_eq!(p.top_k, 20);
        assert_eq!(p.min_p, 0.05);
        assert_eq!(p.repetition_penalty, 1.1);
        assert_eq!(p.presence_penalty, 0.25);
        assert_eq!(p.frequency_penalty, 0.5);
        assert_eq!(p.logit_bias, vec![(7u32, -3.5f32)]);
        assert_eq!(req.sampling.min_tokens, Some(3));
        assert_eq!(req.sampling.stop_token_ids.as_deref(), Some(&[11u32, 22][..]));
    }

    /// `flatten` changes how serde drives the WHOLE struct, so the fields that
    /// are not part of the flattened block have to keep working — `seed` is a
    /// u64 and `max_completion_tokens` is an alias, both easy to lose.
    #[test]
    fn flattening_the_sampling_block_does_not_break_the_other_fields() {
        let req: ChatRequest = serde_json::from_str(
            r#"{"model":"m","messages":[],"seed":18446744073709551615,
                "max_completion_tokens":128,"stream":true,
                "stream_options":{"include_usage":true}}"#,
        )
        .expect("deserializes");
        assert_eq!(req.seed, Some(u64::MAX), "a full-width u64 seed survived");
        assert_eq!(req.max_tokens, Some(128), "the OpenAI alias still applies");
        assert!(req.stream);
        assert!(req.stream_options.expect("present").include_usage);
    }

    /// A request that sets nothing must leave the MODEL's defaults untouched —
    /// that is what makes `request > model default > server default` work.
    #[test]
    fn an_empty_sampling_block_overrides_nothing() {
        let req: ChatRequest =
            serde_json::from_str(r#"{"model":"m","messages":[]}"#).expect("deserializes");
        let mut p = crate::text::sample::SamplingParams {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            ..Default::default()
        };
        req.sampling.apply(&mut p);
        assert_eq!((p.temperature, p.top_p, p.top_k), (0.6, 0.95, 20));
    }

    /// vLLM spells "no top-k" as -1; this sampler spells it 0.
    #[test]
    fn vllms_disabled_top_k_maps_to_this_samplers_spelling() {
        let f = SamplingFields {
            top_k: Some(-1),
            ..Default::default()
        };
        let mut p = crate::text::sample::SamplingParams::default();
        f.apply(&mut p);
        assert_eq!(p.top_k, 0);
        assert!(f.validate().is_ok());
    }

    use super::{Content, ContentPart, ImageUrl};

    #[test]
    fn multipart_text_preserves_exact_boundaries() {
        let content = Content::Parts(vec![
            ContentPart::Text {
                text: "line one\n".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "image".into(),
                    detail: None,
                },
            },
            ContentPart::Text {
                text: "line two".into(),
            },
        ]);
        assert_eq!(content.as_text(), "line one\nline two");
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

// --- responses ---

#[derive(Clone, Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: &'static str,
    /// Unix seconds. REQUIRED by the OpenAI schema and previously absent from
    /// every response this server produced, which made strict clients reject it
    /// and lenient ones surface an undefined value.
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// OpenAI `usage` block. `prompt_tokens_details.cached_tokens` reports what
/// the prefix cache served (mirrors OpenAI's own prompt-caching field, so
/// standard clients pick it up unchanged).
#[derive(Clone, Copy, Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub prompt_tokens_details: PromptTokensDetails,
    /// Present only for a model that frames a reasoning trace. Without it a
    /// client cannot tell how much of `completion_tokens` was the trace rather
    /// than the answer, which is what it is billed and budgeted on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: u64,
}

impl From<crate::serve::stream::TokenUsage> for Usage {
    fn from(u: crate::serve::stream::TokenUsage) -> Self {
        Usage {
            prompt_tokens: u.prompt_tokens as u64,
            completion_tokens: u.completion_tokens as u64,
            total_tokens: (u.prompt_tokens + u.completion_tokens) as u64,
            prompt_tokens_details: PromptTokensDetails {
                cached_tokens: u.cached_tokens as u64,
            },
            completion_tokens_details: None,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<crate::serve::logprobs::ChatLogprobs>,
    pub finish_reason: Option<&'static str>,
    /// Present only when the wire `finish_reason` had to be widened to fit the
    /// OpenAI vocabulary — currently a preemption reported as `"length"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_plow_finish_reason: Option<&'static str>,
}

/// One streamed `chat.completion.chunk`.
#[derive(Clone, Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
    /// Stamped once per request and repeated on every chunk of that stream.
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Set on the final frame only (OpenAI stream-usage shape).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<crate::serve::logprobs::ChatLogprobs>,
    pub finish_reason: Option<&'static str>,
    /// Same widening note as [`Choice::x_plow_finish_reason`]. The streamed
    /// path used to omit it, so an operator-forced stop reached a streaming
    /// client as an ordinary `"length"` and was indistinguishable from the
    /// model simply hitting `max_tokens`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_plow_finish_reason: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Streamed thinking trace, before the answer starts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<CompletionTokenIds>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompletionTokenIds {
    pub prompt: Vec<u32>,
    pub completion: Vec<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub logprobs: Option<serde_json::Value>,
    pub finish_reason: Option<&'static str>,
    /// Same widening note as [`Choice::x_plow_finish_reason`]. `/v1/completions`
    /// carried no such field at all, so a preemption there was reported purely
    /// as `"length"` with no way for a caller to tell the two apart.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub x_plow_finish_reason: Option<&'static str>,
}

/// `GET /v1/models` response.
#[derive(Clone, Debug, Serialize)]
pub struct ModelList {
    pub object: &'static str,
    pub data: Vec<ModelCard>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelCard {
    pub x_plow_endpoints: Vec<&'static str>,
    /// Chat input modalities: `text`, plus `image` / `audio` when the packet carries encoders.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub x_plow_modalities: Vec<String>,
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub owned_by: &'static str,
    /// `root` is the id a derived model was served from; with no adapters or
    /// aliases it is the model's own id, which is what vLLM reports too.
    pub root: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// The compiled context length. LiteLLM, OpenWebUI and vLLM's own clients
    /// read this to size a request; a card without it makes them guess.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
    /// Empty, but PRESENT: the OpenAI schema declares it and typed clients
    /// index into it.
    pub permission: Vec<serde_json::Value>,
    /// `"device_argmax"` when this model's backend IGNORES the request's
    /// sampling parameters (the gfx950 engine samples greedily on device).
    /// Absent when sampling is applied normally. A vendor-prefixed field, so a
    /// typed OpenAI client ignores it and a plow-aware one can check it before
    /// sending a request whose `temperature` would be discarded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub x_plow_sampling: Option<&'static str>,
}

/// Unix seconds, for the `created` field every OpenAI object carries.
pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The OpenAI error envelope: `{"error": {"message", "type", "code"}}`.
/// Every error this server returns goes through here. It used to emit a bare
/// `{"error": "<string>"}`, which openai-python cannot read — it looks up
/// `body["error"]["message"]` — so a client got a useless message and no code
/// to branch on.
#[derive(Clone, Debug, Serialize)]
pub struct ApiErrorBody {
    pub error: ApiError,
}

#[derive(Clone, Debug, Serialize)]
pub struct ApiError {
    pub message: String,
    pub r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
}

impl ApiErrorBody {
    pub fn new(
        message: impl Into<String>,
        r#type: &'static str,
        code: Option<&'static str>,
        param: Option<String>,
    ) -> Self {
        ApiErrorBody {
            error: ApiError {
                message: message.into(),
                r#type,
                code,
                param,
            },
        }
    }
}

#[cfg(test)]
mod image_refusal_tests {
    use super::{Content, ContentPart, ImageUrl};

    /// `has_image` must see an image part ANYWHERE in a multipart message, and
    /// `text()` must be shown to drop it — the two together are why
    /// `chat_completions` refuses rather than serving a fluent answer to a
    /// question it never saw.
    #[test]
    fn an_image_part_is_detected_and_would_otherwise_be_dropped() {
        let c = Content::Parts(vec![
            ContentPart::Text {
                text: "what is in this ".into(),
            },
            ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,AAAA".into(),
                    detail: None,
                },
            },
            ContentPart::Text {
                text: "picture?".into(),
            },
        ]);
        assert!(c.has_image(), "an image part must be visible to the guard");
        // THE HAZARD, stated as an assertion: flattening silently discards it
        // and leaves a question that reads as complete.
        assert_eq!(c.as_text(), "what is in this picture?");
    }

    #[test]
    fn plain_text_is_never_refused() {
        assert!(!Content::Text("hello".into()).has_image());
        assert!(!Content::Parts(vec![ContentPart::Text {
            text: "hello".into()
        }])
        .has_image());
        // An empty multipart message is text-only, not an image.
        assert!(!Content::Parts(vec![]).has_image());
    }
}
