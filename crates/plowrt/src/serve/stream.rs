//! §G SSE helpers + the mux→handler streaming chunk.
//!
//! `StreamChunk` is the wire between the muxer (a token producer) and the
//! HTTP handler (an OpenAI-shape formatter). The mux emits one `Token` per
//! produced token (with an incremental detokenized `text` delta) and a final
//! `Done` on stop; the handler decides whether to buffer them into one JSON
//! reply (non-streaming) or forward each as a `chat.completion.chunk` frame
//! (SSE). Errors ride the same channel so a mid-generation failure closes the
//! stream deterministically.
//!
//! Cancellation is implicit: the handler drops the `mpsc::Receiver` when the
//! client disconnects; the mux's `send()` returns Err on the next token and
//! the slot is freed.

use axum::body::Bytes;
use tokio::sync::mpsc;

use crate::RuntimeError;

/// The OpenAI stream terminator.
pub const DONE: &str = "[DONE]";

/// The id of a text-only [`StreamChunk::Token`]: no token was produced, so it is not a completion
/// token id.
pub const TEXT_ONLY: u32 = u32::MAX;

/// One event from the muxer to the request handler.
#[derive(Debug)]
pub enum StreamChunk {
    /// One newly-produced token, plus the incremental decoded string delta
    /// (may be empty when the tokenizer's decode of the running id vec did
    /// not yield a new visible segment, e.g. a partial UTF-8 sequence).
    /// `logprobs` is set when the request asked for them (OpenAI `logprobs`).
    /// `id` is [`TEXT_ONLY`] for a chunk that only releases held bytes at a stop token.
    Token {
        id: u32,
        text: String,
        logprobs: Option<Box<crate::text::logprobs::TokenLogprobs>>,
    },
    /// Terminal event: stop condition met. `executed` is the aggregate packet
    /// count for the whole request (feeds observability, not the wire).
    Done {
        executed: usize,
        reason: FinishReason,
        usage: TokenUsage,
    },
    /// Terminal event: generation failed. The handler maps this to an HTTP
    /// status when the request is still buffering, or closes the SSE stream
    /// otherwise.
    Err(RuntimeError),
}

/// Final token accounting for one request, carried on `Done` and rendered
/// as OpenAI `usage` (with `prompt_tokens_details.cached_tokens` when the
/// prefix cache served part of the prompt).
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    /// Prompt tokens attached from the prefix cache (KV not recomputed).
    pub cached_tokens: usize,
    pub completion_tokens: usize,
}

/// OpenAI-compatible finish reason.
#[derive(Debug, Clone, Copy)]
pub enum FinishReason {
    Stop,
    Length,
    /// The serve manager reclaimed the engine mid-generation (S1 switch with
    /// preemptive drain) — the stream carries everything generated so far and
    /// the client should retry for the remainder.
    Preempted,
}

impl FinishReason {
    /// The internal name, for logs and metrics. NOT for the wire.
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Preempted => "preempted",
        }
    }

    /// The value that goes on the wire. `Preempted` is NOT an OpenAI
    /// `finish_reason`, and a client typed against
    /// `Literal["stop","length","tool_calls","content_filter","function_call"]`
    /// rejects the whole response when it sees one. It maps to `"length"` —
    /// which is honest, the answer really was cut short — and the true cause
    /// stays visible in `x_plow_finish_reason` and in the server log.
    pub fn as_openai(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length | FinishReason::Preempted => "length",
        }
    }

    /// Whether the wire value hides a plow-specific cause worth reporting.
    pub fn is_vendor_specific(self) -> bool {
        matches!(self, FinishReason::Preempted)
    }
}

/// Sender/receiver pair for one request's stream.
pub type ChunkSender = mpsc::Sender<StreamChunk>;
pub type ChunkReceiver = mpsc::Receiver<StreamChunk>;

pub fn channel() -> (ChunkSender, ChunkReceiver) {
    // Up to 32 queued tokens plus a reserved terminal event.
    mpsc::channel(33)
}

/// Serialize a chunk to its SSE `data:` payload.
pub fn chunk_data(chunk: &impl serde::Serialize) -> String {
    serde_json::to_string(chunk).unwrap_or_default()
}

/// The bytes `axum::response::sse::Event::default().data(data)` puts on the wire: one
/// `data: <line>` field per `\n`-separated line, then the blank line. Panics on `\r`, as it does.
pub fn sse_data(data: &str) -> Bytes {
    let mut buf = Vec::with_capacity(data.len() + 8);
    for line in data.as_bytes().split(|&b| b == b'\n') {
        sse_field(&mut buf, b"data", line);
    }
    buf.push(b'\n');
    Bytes::from(buf)
}

/// The bytes of `axum::response::sse::Event::default().comment(text)`.
pub fn sse_comment(text: &str) -> Bytes {
    let mut buf = Vec::with_capacity(text.len() + 4);
    sse_field(&mut buf, b"", text.as_bytes());
    buf.push(b'\n');
    Bytes::from(buf)
}

fn sse_field(buf: &mut Vec<u8>, name: &[u8], value: &[u8]) {
    assert!(
        !value.iter().any(|&b| b == b'\r' || b == b'\n'),
        "SSE field value cannot contain newlines or carriage returns"
    );
    buf.extend_from_slice(name);
    buf.extend_from_slice(b": ");
    buf.extend_from_slice(value);
    buf.push(b'\n');
}

/// An SSE response over pre-framed events: the headers and framing of `axum::response::sse::Sse`
/// (no keep-alive), without an `Event` buffer per frame.
pub fn sse_response<S>(frames: S) -> axum::response::Response
where
    S: futures::Stream<Item = Bytes> + Send + 'static,
{
    use axum::response::IntoResponse;
    use futures::StreamExt;
    (
        [
            (axum::http::header::CONTENT_TYPE, "text/event-stream"),
            (axum::http::header::CACHE_CONTROL, "no-cache"),
        ],
        axum::body::Body::from_stream(frames.map(Ok::<_, std::convert::Infallible>)),
    )
        .into_response()
}

/// The fixed head of every chunk of one stream, `{"id":..,"object":..,"created":..,"model":..,
/// "choices":[`, serialized once. [`Self::frame`] appends one choice and closes the object: byte
/// for byte the serde output of the whole chunk with that single choice and no `usage`.
pub struct FrameHead(Vec<u8>);

impl FrameHead {
    pub fn new(id: &str, object: &str, created: u64, model: &str) -> Self {
        let mut head = b"data: {\"id\":".to_vec();
        let _ = serde_json::to_writer(&mut head, id);
        head.extend_from_slice(b",\"object\":");
        let _ = serde_json::to_writer(&mut head, object);
        head.extend_from_slice(b",\"created\":");
        head.extend_from_slice(created.to_string().as_bytes());
        head.extend_from_slice(b",\"model\":");
        let _ = serde_json::to_writer(&mut head, model);
        head.extend_from_slice(b",\"choices\":[");
        FrameHead(head)
    }

    /// One SSE event carrying the chunk with `choice` as its only choice.
    pub fn frame(&self, choice: &impl serde::Serialize) -> Bytes {
        let mut buf = Vec::with_capacity(self.0.len() + 160);
        buf.extend_from_slice(&self.0);
        let _ = serde_json::to_writer(&mut buf, choice);
        buf.extend_from_slice(b"]}\n\n");
        Bytes::from(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::sse::{Event, Sse};
    use axum::response::IntoResponse;
    use http_body_util::BodyExt;

    async fn collect(r: axum::response::Response) -> (String, axum::http::HeaderMap) {
        let headers = r.headers().clone();
        let body = r.into_body().collect().await.unwrap().to_bytes();
        (String::from_utf8(body.to_vec()).unwrap(), headers)
    }

    fn texts() -> Vec<String> {
        let mut v: Vec<String> = ["", " the", "a\nb", "\n", "\"q\"\\", "tab\t", "\u{1}\u{1f}", "é日本\u{FFFD}", "</s>"]
            .map(String::from)
            .to_vec();
        v.push((0..300u32).map(|i| char::from_u32(32 + (i * 37) % 900).unwrap_or('x')).collect());
        v
    }

    #[tokio::test]
    async fn frames_match_axum_events_byte_for_byte() {
        use crate::serve::openai::{ChatChunk, ChunkChoice, CompletionChoice, CompletionResponse, Delta};
        let head = FrameHead::new("cmpl-\"1\"", "text_completion", 1_789_920_673, "gemma\n4");
        let chat_head = FrameHead::new("chatcmpl-9", "chat.completion.chunk", 7, "m");
        let (mut want, mut got) = (Vec::new(), Vec::new());
        for text in texts() {
            let choice = CompletionChoice { index: 0, text: text.clone(), logprobs: None, finish_reason: None, x_plow_finish_reason: None };
            let full = CompletionResponse {
                id: "cmpl-\"1\"".into(),
                object: "text_completion",
                created: 1_789_920_673,
                model: "gemma\n4".into(),
                choices: vec![choice.clone()],
                usage: None,
                token_ids: None,
            };
            want.push(Event::default().data(chunk_data(&full)));
            got.push(head.frame(&choice));
            let c = ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant"), content: Some(text.clone()), reasoning_content: None, tool_calls: None },
                logprobs: None,
                finish_reason: None,
                x_plow_finish_reason: None,
            };
            let full = ChatChunk { id: "chatcmpl-9".into(), object: "chat.completion.chunk", created: 7, model: "m".into(), choices: vec![c.clone()], usage: None };
            want.push(Event::default().data(chunk_data(&full)));
            got.push(chat_head.frame(&c));
            want.push(Event::default().data(&text));
            got.push(sse_data(&text));
        }
        want.push(Event::default().comment("server-timing ttft;dur=1.5"));
        got.push(sse_comment("server-timing ttft;dur=1.5"));
        want.push(Event::default().data(DONE));
        got.push(sse_data(DONE));
        let axum = collect(Sse::new(futures::stream::iter(want.into_iter().map(Ok::<_, std::convert::Infallible>))).into_response()).await;
        assert_eq!(collect(sse_response(futures::stream::iter(got))).await, axum);
    }
}
