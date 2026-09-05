//! `/v1/completions` — raw-prompt streaming and non-streaming generation.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{self, Stream};

use crate::serve::openai::*;
use crate::serve::stream::{self as stream_mod, StreamChunk};
use crate::serve::{status_for, AppState};

static REQ_SEQ: AtomicU64 = AtomicU64::new(0);

fn validate_return_token_ids(stream: bool, return_token_ids: bool) -> Result<(), &'static str> {
    if stream && return_token_ids {
        Err("return_token_ids is supported only for non-streaming completions")
    } else {
        Ok(())
    }
}

fn request_id() -> String {
    format!("cmpl-{:016x}", REQ_SEQ.fetch_add(1, Ordering::Relaxed))
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    if let Err(error) = validate_return_token_ids(req.stream, req.return_token_ids) {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": error})),
        )
            .into_response();
    }
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

    let t_arrive = std::time::Instant::now();
    crate::obs::ttft::reset();

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

    crate::obs::Metrics::inc(&state.metrics.requests);
    let (Some(mux), Ok(bundle)) = (state.mux(&req.model), state.registry.get(&req.model)) else {
        return (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("no model registered for '{}'.", req.model)})),
        )
            .into_response();
    };
    let prompt_ids = crate::obs::ttft::timed(&crate::obs::ttft::ENCODE, || {
        bundle
            .tokenizer()
            .encode_with_special_tokens(&req.prompt, req.add_special_tokens)
    });
    if prompt_ids.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(serde_json::json!({"error": "prompt encodes to zero tokens"})),
        )
            .into_response();
    }
    let n_prompt = prompt_ids.len();
    let (tx, rx) = stream_mod::channel();
    let response_prompt_ids = req.return_token_ids.then(|| prompt_ids.clone());
    let job = crate::serve::mux::Job {
        prompt_ids,
        gen,
        arrived: std::time::Instant::now(),
        respond: tx,
    };
    if let Err(err) = mux.submit(job) {
        return match err {
            crate::serve::mux::SubmitError::Full(_) => {
                crate::obs::Metrics::inc(&state.metrics.rejected);
                (
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    Json(serde_json::json!({"error": "model request queue full"})),
                )
                    .into_response()
            }
            crate::serve::mux::SubmitError::Closed(_) => (
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({"error": "model dispatcher unavailable"})),
            )
                .into_response(),
        };
    }

    let id = request_id();
    if req.stream {
        let include_usage = req.stream_options.map(|o| o.include_usage).unwrap_or(false);
        sse_response(id, req.model, rx, include_usage, t_arrive, n_prompt).into_response()
    } else {
        buffer_and_reply(id, req.model, rx, response_prompt_ids).await
    }
}

async fn buffer_and_reply(
    request_id: String,
    model: String,
    mut rx: stream_mod::ChunkReceiver,
    prompt_token_ids: Option<Vec<u32>>,
) -> Response {
    let mut text = String::new();
    let mut completion_token_ids = Vec::new();
    let mut finish = None;
    let mut usage = None;
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { id, text: delta } => {
                text.push_str(&delta);
                if prompt_token_ids.is_some() {
                    completion_token_ids.push(id);
                }
            }
            StreamChunk::Done {
                reason, usage: u, ..
            } => {
                finish = Some(reason.as_str());
                usage = Some(u.into());
                break;
            }
            StreamChunk::Err(e) => {
                tracing::warn!(%model, error = %e, partial_chars = text.len(), "completion stream error");
                return (
                    status_for(&e),
                    Json(serde_json::json!({"error": e.to_string(), "partial": text})),
                )
                    .into_response();
            }
        }
    }
    let Some(finish) = finish else {
        tracing::warn!(%model, partial_chars = text.len(), "completion stream ended without terminal chunk");
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "generation stream ended without a finish reason (slot cut)",
                "partial": text,
            })),
        )
            .into_response();
    };
    Json(CompletionResponse {
        id: request_id,
        object: "text_completion",
        model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            logprobs: None,
            finish_reason: Some(finish),
        }],
        usage,
        token_ids: prompt_token_ids.map(|prompt| CompletionTokenIds {
            prompt,
            completion: completion_token_ids,
        }),
    })
    .into_response()
}

fn sse_response(
    request_id: String,
    model: String,
    rx: stream_mod::ChunkReceiver,
    include_usage: bool,
    t_arrive: std::time::Instant,
    n_prompt: usize,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    struct SseState {
        rx: stream_mod::ChunkReceiver,
        first: bool,
        done: bool,
        pending: std::collections::VecDeque<Event>,
    }
    let body = stream::unfold(
        SseState {
            rx,
            first: true,
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
                let chunk = match st.rx.recv().await {
                    Some(c) => c,
                    None => {
                        tracing::warn!(%model, "completion SSE stream ended without terminal chunk");
                        return None;
                    }
                };
                let (choice, tail_usage, terminate) = match chunk {
                    StreamChunk::Token { text, .. } => {
                        if st.first {
                            st.first = false;
                            crate::obs::ttft::dump(t_arrive.elapsed().as_nanos() as u64, n_prompt);
                            crate::obs::pfx::report();
                        }
                        (
                            vec![CompletionChoice {
                                index: 0,
                                text,
                                logprobs: None,
                                finish_reason: None,
                            }],
                            None,
                            false,
                        )
                    }
                    StreamChunk::Done { reason, usage, .. } => (
                        vec![CompletionChoice {
                            index: 0,
                            text: String::new(),
                            logprobs: None,
                            finish_reason: Some(reason.as_str()),
                        }],
                        include_usage.then(|| usage.into()),
                        true,
                    ),
                    StreamChunk::Err(e) => {
                        tracing::warn!(%model, error = %e, "completion SSE stream error");
                        (
                            vec![CompletionChoice {
                                index: 0,
                                text: format!("[error: {e}]"),
                                logprobs: None,
                                finish_reason: Some("stop"),
                            }],
                            None,
                            true,
                        )
                    }
                };
                let frame = CompletionResponse {
                    id: request_id,
                    object: "text_completion",
                    model: model.clone(),
                    choices: choice,
                    usage: None,
                    token_ids: None,
                };
                if let Some(usage) = tail_usage {
                    let usage_frame = CompletionResponse {
                        id: frame.id.clone(),
                        object: "text_completion",
                        model,
                        choices: Vec::new(),
                        usage: Some(usage),
                        token_ids: None,
                    };
                    st.pending
                        .push_back(Event::default().data(stream_mod::chunk_data(&usage_frame)));
                }
                if terminate {
                    st.pending
                        .push_back(Event::default().data(stream_mod::DONE));
                }
                Some((
                    Ok(Event::default().data(stream_mod::chunk_data(&frame))),
                    st,
                ))
            }
        },
    );
    Sse::new(body)
}

#[cfg(test)]
mod tests {
    use super::{request_id, validate_return_token_ids};
    use crate::serve::openai::{
        CompletionChoice, CompletionRequest, CompletionResponse, CompletionTokenIds,
    };

    #[test]
    fn request_ids_are_unique() {
        assert_ne!(request_id(), request_id());
    }

    #[test]
    fn vllm_completion_request_fields_deserialize() {
        let req: CompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "model",
            "prompt": "raw prompt",
            "stream": true,
            "max_tokens": 1024,
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": true,
            "stream_options": {"include_usage": true},
            "return_token_ids": true,
            "best_of": 1
        }))
        .unwrap();
        assert_eq!(req.prompt, "raw prompt");
        assert_eq!(req.max_tokens, Some(1024));
        assert_eq!(req.ignore_eos, Some(true));
        assert!(req.stream_options.unwrap().include_usage);
        assert!(req.return_token_ids);
    }

    #[test]
    fn completion_token_ids_are_opt_in_at_response_root() {
        let response = CompletionResponse {
            id: "cmpl-test".into(),
            object: "text_completion",
            model: "model".into(),
            choices: vec![CompletionChoice {
                index: 0,
                text: "x".into(),
                logprobs: None,
                finish_reason: Some("length"),
            }],
            usage: None,
            token_ids: Some(CompletionTokenIds {
                prompt: vec![1, 2],
                completion: vec![3],
            }),
        };
        let json = serde_json::to_value(response).unwrap();
        assert_eq!(json["token_ids"]["prompt"], serde_json::json!([1, 2]));
        assert_eq!(json["token_ids"]["completion"], serde_json::json!([3]));
    }

    #[test]
    fn streaming_rejects_return_token_ids() {
        assert!(validate_return_token_ids(true, true).is_err());
        assert!(validate_return_token_ids(true, false).is_ok());
        assert!(validate_return_token_ids(false, true).is_ok());
    }
}
