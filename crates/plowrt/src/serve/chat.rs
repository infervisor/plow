//! §G `/v1/chat/completions` — streaming (SSE) and non-streaming.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, Stream, StreamExt};

use crate::serve::openai::*;
use crate::serve::stream::{self as stream_mod, FinishReason, StreamChunk};
use crate::serve::{status_for, AppState};

static REQ_SEQ: AtomicU64 = AtomicU64::new(0);

fn request_id() -> String {
    format!("chatcmpl-{:016x}", REQ_SEQ.fetch_add(1, Ordering::Relaxed))
}

/// Handler: dispatches to the streaming or non-streaming path.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    // S1 multi-model: a request for a managed, non-resident model triggers
    // the switch HERE (evict LRU + load), before the prompt is built — so the
    // template choice below sees the engine. Resident models pass through on
    // the manager's lock-free fast path. A switch that cannot fit sheds with
    // 503 + Retry-After (the client should back off, not hammer the planner).
    #[cfg(feature = "cuda")]
    if let Some(mgr) = state.manager() {
        if mgr.manages(&req.model) {
            use crate::serve::manager::EnsureError;
            if let Err(e) = mgr.ensure_resident(&req.model).await {
                return match e {
                    EnsureError::WontFit { .. } => (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
                        [("retry-after", "30")],
                        Json(serde_json::json!({"error": e.to_string()})),
                    )
                        .into_response(),
                    EnsureError::Load(err) => (
                        status_for(&err),
                        Json(serde_json::json!({"error": err.to_string()})),
                    )
                        .into_response(),
                };
            }
        }
    }

    // Models served by the GPU engine get their real chat template (the
    // tokenizer resolves the markers through `added_tokens`); the CPU
    // reference path keeps the simple role-prefix flatten — its logits are a
    // stand-in, so a template would be costume jewelry there.
    let prompt = if state.has_gpu_engine(&req.model) {
        gemma_chat_prompt(&req.messages)
    } else {
        let mut prompt = String::new();
        for m in &req.messages {
            if !prompt.is_empty() {
                prompt.push_str("\n\n");
            }
            prompt.push_str(&m.role);
            prompt.push_str(":\n");
            prompt.push_str(&m.content.as_text());
        }
        prompt
    };

    // Build generation controls from the request.
    let mut gen = crate::serve::GenParams::default();
    if let Some(m) = req.max_tokens {
        gen.max_tokens = m as usize;
    }
    if let Some(t) = req.temperature {
        gen.params.temperature = t;
    }
    if let Some(p) = req.top_p {
        gen.params.top_p = p;
    }
    if let Some(ignore) = req.ignore_eos {
        gen.ignore_eos = ignore;
    }

    // Route to the per-model muxer. Tokens stream back as `StreamChunk`s over
    // an mpsc — the muxer produces one per generated token, ending with `Done`.
    crate::obs::Metrics::inc(&state.metrics.requests);
    let (Some(mux), Ok(bundle)) = (state.mux(&req.model), state.registry.get(&req.model)) else {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("no model registered for '{}'.", req.model)})),
        )
            .into_response();
    };
    // Tokenize HERE, on the handler task — the dispatcher loop is the
    // serialized decode critical path and must never encode a long prompt.
    let prompt_ids = bundle.tokenizer().encode(&prompt);
    let (tx, rx) = stream_mod::channel();
    let job = crate::serve::mux::Job {
        prompt_ids,
        gen,
        arrived: std::time::Instant::now(),
        respond: tx,
    };
    if mux.submit(job).is_err() {
        return (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "model dispatcher unavailable"})),
        )
            .into_response();
    }

    let request_id = request_id();
    if req.stream {
        let include_usage = req.stream_options.map(|o| o.include_usage).unwrap_or(false);
        sse_response(request_id, req.model, rx, include_usage).into_response()
    } else {
        buffer_and_reply(request_id, req.model, rx).await
    }
}

/// The Gemma-4 canonical chat format (the checkpoint's
/// `chat_template.jinja`, text-only subset): `<bos>`, one
/// `<|turn>{role}\n…<turn|>\n` block per message (`assistant` → `model`, a
/// leading `system` message becomes a system turn), then the generation
/// prompt `<|turn>model\n<|channel>thought\n<channel|>` (thinking disabled —
/// the closed empty thought channel, exactly what the template emits).
/// The marker strings tokenize to their special ids via `added_tokens`.
fn gemma_chat_prompt(messages: &[Message]) -> String {
    let mut p = String::from("<bos>");
    for m in messages {
        let role = match m.role.as_str() {
            "assistant" => "model",
            "system" | "developer" => "system",
            _ => "user",
        };
        p.push_str("<|turn>");
        p.push_str(role);
        p.push('\n');
        p.push_str(m.content.as_text().trim());
        p.push_str("<turn|>\n");
    }
    p.push_str("<|turn>model\n<|channel>thought\n<channel|>");
    p
}

/// Non-streaming path: consume every chunk until `Done`/`Err`, concatenating
/// `Token.text` deltas into one response body. Errors before the first token
/// map to an HTTP status; errors mid-generation return what was produced so
/// far and the caller sees a partial (matches OpenAI behavior for `finish_reason`).
async fn buffer_and_reply(
    request_id: String,
    model: String,
    mut rx: stream_mod::ChunkReceiver,
) -> Response {
    let mut text = String::new();
    let mut finish = "stop";
    let mut usage = None;
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { text: delta, .. } => text.push_str(&delta),
            StreamChunk::Done { reason, usage: u, .. } => {
                finish = reason.as_str();
                usage = Some(u.into());
                break;
            }
            StreamChunk::Err(e) => {
                // Explicit, never a silent `finish: stop`: an error before any
                // content is a plain HTTP error; mid-generation it is still an
                // HTTP error carrying the partial text — the client must see
                // that the stream was cut (shed, engine fault), not a clean
                // completion.
                tracing::warn!(%model, error = %e, partial_chars = text.len(), "chat: stream error");
                return (
                    status_for(&e),
                    Json(serde_json::json!({
                        "error": e.to_string(),
                        "partial": text,
                    })),
                )
                    .into_response();
            }
        }
    }
    Json(ChatResponse {
        id: request_id.clone(),
        object: "chat.completion",
        model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".into(),
                content: Content::Text(text),
            },
            finish_reason: match finish {
                "length" => "length",
                _ => "stop",
            },
        }],
        usage,
    })
    .into_response()
}

/// Streaming path: one SSE `chat.completion.chunk` frame per produced token,
/// terminated by a final chunk carrying `finish_reason` and then `[DONE]`.
/// Empty-delta tokens (partial UTF-8) are still emitted so the client can
/// keep an accurate token count if it wishes — the `delta.content` field is
/// simply the empty string for those. With `stream_options.include_usage`
/// an extra usage-only chunk (empty `choices` — the OpenAI stream-usage
/// shape) precedes `[DONE]`; without it no chunk carries usage.
fn sse_response(
    request_id: String,
    model: String,
    rx: stream_mod::ChunkReceiver,
    include_usage: bool,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let role_chunk = ChatChunk {
        id: request_id.clone(),
        object: "chat.completion.chunk",
        model: model.clone(),
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant"),
                content: None,
            },
            finish_reason: None,
        }],
        usage: None,
    };
    let role_frame = stream::once(async move {
        Ok::<_, std::convert::Infallible>(
            Event::default().data(stream_mod::chunk_data(&role_chunk)),
        )
    });

    // State threaded through the unfold: the receiver, and the tail frames
    // (optional usage-only chunk, then [DONE]) drained one per poll.
    struct SseState {
        rx: stream_mod::ChunkReceiver,
        done: bool,
        pending: std::collections::VecDeque<Event>,
    }
    let body = stream::unfold(
        SseState {
            rx,
            done: false,
            pending: std::collections::VecDeque::new(),
        },
        move |mut st| {
            let model = model.clone();
            let request_id = request_id.clone();
            async move {
                if st.done {
                    return None;
                }
                if let Some(ev) = st.pending.pop_front() {
                    st.done = st.pending.is_empty();
                    return Some((Ok(ev), st));
                }
                let chunk = st.rx.recv().await?;
                let (frame, terminate) = match chunk {
                    StreamChunk::Token { text, .. } => {
                        let ch = ChatChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk",
                            model,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: Some(text),
                                },
                                finish_reason: None,
                            }],
                            usage: None,
                        };
                        (Event::default().data(stream_mod::chunk_data(&ch)), false)
                    }
                    StreamChunk::Done { reason, usage, .. } => {
                        let ch = ChatChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk",
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                },
                                finish_reason: Some(match reason {
                                    FinishReason::Length => "length",
                                    FinishReason::Stop => "stop",
                                }),
                            }],
                            usage: None,
                        };
                        if include_usage {
                            // OpenAI stream-usage shape: a separate chunk with
                            // EMPTY choices carries usage, never the finish chunk.
                            let uch = ChatChunk {
                                id: request_id.clone(),
                                object: "chat.completion.chunk",
                                model,
                                choices: Vec::new(),
                                usage: Some(usage.into()),
                            };
                            st.pending
                                .push_back(Event::default().data(stream_mod::chunk_data(&uch)));
                        }
                        (Event::default().data(stream_mod::chunk_data(&ch)), true)
                    }
                    StreamChunk::Err(e) => {
                        // Explicit in-band error marker (SSE has no status to
                        // change mid-stream) — and logged, never silent.
                        tracing::warn!(%model, error = %e, "chat: SSE stream error");
                        let ch = ChatChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk",
                            model,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: Some(format!("[error: {e}]")),
                                },
                                finish_reason: Some("stop"),
                            }],
                            usage: None,
                        };
                        (Event::default().data(stream_mod::chunk_data(&ch)), true)
                    }
                };
                if terminate {
                    st.pending
                        .push_back(Event::default().data(stream_mod::DONE));
                }
                Some((Ok(frame), st))
            }
        },
    );

    Sse::new(role_frame.chain(body))
}

#[cfg(test)]
mod tests {
    use super::request_id;

    #[test]
    fn request_ids_are_unique() {
        assert_ne!(request_id(), request_id());
    }
}
