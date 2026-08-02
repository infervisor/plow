//! §G OpenAI-compatible request/response DTOs.

use serde::{Deserialize, Serialize};

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
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// vLLM extension: run to `max_tokens` instead of stopping at eos. Sent by
    /// `vllm bench serve` for the synthetic datasets; ignoring it makes every
    /// benchmark against plowrt under-report throughput. See `GenParams`.
    #[serde(default)]
    pub ignore_eos: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
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
    pub content: Content,
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
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct PromptTokensDetails {
    pub cached_tokens: u64,
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
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: &'static str,
}

/// One streamed `chat.completion.chunk`.
#[derive(Clone, Debug, Serialize)]
pub struct ChatChunk {
    pub id: String,
    pub object: &'static str,
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
}

#[derive(Clone, Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
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
    pub owned_by: &'static str,
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
