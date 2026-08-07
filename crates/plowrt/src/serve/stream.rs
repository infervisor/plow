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

use tokio::sync::mpsc;

use crate::serve::openai::ChatChunk;
use crate::RuntimeError;

/// The OpenAI stream terminator.
pub const DONE: &str = "[DONE]";

/// One event from the muxer to the request handler.
#[derive(Debug)]
pub enum StreamChunk {
    /// One newly-produced token, plus the incremental decoded string delta
    /// (may be empty when the tokenizer's decode of the running id vec did
    /// not yield a new visible segment, e.g. a partial UTF-8 sequence).
    Token { id: u32, text: String },
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
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Preempted => "preempted",
        }
    }
}

/// Sender/receiver pair for one request's stream.
pub type ChunkSender = mpsc::Sender<StreamChunk>;
pub type ChunkReceiver = mpsc::Receiver<StreamChunk>;

pub fn channel() -> (ChunkSender, ChunkReceiver) {
    mpsc::channel(32)
}

/// Serialize a chunk to its SSE `data:` payload.
pub fn chunk_data(chunk: &ChatChunk) -> String {
    serde_json::to_string(chunk).unwrap_or_default()
}
