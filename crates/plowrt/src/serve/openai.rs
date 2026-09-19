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
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
    #[serde(default)]
    pub top_logprobs: Option<serde_json::Value>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub functions: Option<serde_json::Value>,
    #[serde(default)]
    pub function_call: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
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
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
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

#[derive(Clone, Debug, Deserialize, Serialize)]
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
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Whether any part is an image (routes through the vision stage).
    pub fn has_image(&self) -> bool {
        matches!(self, Content::Parts(parts)
            if parts.iter().any(|p| matches!(p, ContentPart::ImageUrl { .. })))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[cfg(test)]
mod tests {
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
    pub message: Message,
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
