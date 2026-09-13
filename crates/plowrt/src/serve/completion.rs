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

/// Seeded per PROCESS, not from zero. A counter starting at 0 made the first
/// response of every server exactly `cmpl-0000000000000000` and made ids
/// collide across restarts and replicas, which breaks any log correlation or
/// dedup keyed on `id`.
static REQ_SEQ: std::sync::LazyLock<AtomicU64> = std::sync::LazyLock::new(|| {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
        ^ (std::process::id() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    AtomicU64::new(nonce)
});

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
    // `Result<Json<..>, JsonRejection>` rather than `Json<..>`: axum's default
    // rejection is a PLAIN-TEXT 400/415/422, and a client that calls
    // `resp.json()` on a 4xx — every OpenAI SDK does — raises a decode error
    // instead of showing the user what was wrong with their request.
    req: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let Json(req) = match req {
        Ok(r) => r,
        Err(e) => {
            return crate::serve::api_error(
                e.status(),
                e.body_text(),
                "invalid_request_error",
                Some("invalid_json"),
                None,
            )
        }
    };
    if let Err(error) = validate_return_token_ids(req.stream, req.return_token_ids) {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            error,
            "invalid_request_error",
            Some("unsupported_parameter"),
            Some("return_token_ids".into()),
        );
    }
    // Refuse rather than drop, as on the chat endpoint.
    if req.n.is_some_and(|n| n != 1) || req.best_of.is_some_and(|b| b != 1) {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "this server returns exactly one choice; n and best_of must be 1 if present",
            "invalid_request_error",
            Some("unsupported_parameter"),
            Some("n".into()),
        );
    }
    if req.echo == Some(true) || req.suffix.is_some() {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "`echo` and `suffix` are not implemented by this server",
            "invalid_request_error",
            Some("unsupported_parameter"),
            Some("echo".into()),
        );
    }
    if req.logprobs.as_ref().is_some_and(|v| !v.is_null()) {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "`logprobs` is not implemented; it is refused rather than returned as null",
            "invalid_request_error",
            Some("unsupported_parameter"),
            Some("logprobs".into()),
        );
    }
    #[cfg(feature = "cuda")]
    if let Some(mgr) = state.manager_for(&req.model) {
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
                    // No `retry-after`: only an explicit load brings it back.
                    EnsureError::Unloaded => (
                        axum::http::StatusCode::SERVICE_UNAVAILABLE,
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
    if let Some(stop) = &req.stop {
        gen.stop = stop.list();
    }
    gen.seed = req.seed;

    let (Some(mux), Ok(bundle)) = (state.mux(&req.model), state.registry.get(&req.model)) else {
        return crate::serve::api_error(
            axum::http::StatusCode::NOT_FOUND,
            format!("no model registered for '{}'.", req.model),
            "invalid_request_error",
            Some("model_not_found"),
            Some("model".into()),
        );
    };
    // OpenAI's four prompt forms. Token-id prompts skip the tokenizer entirely;
    // batches are refused explicitly rather than silently serving element 0.
    let batch_refusal = || {
        crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "a batched prompt is not supported; send one prompt per request",
            "invalid_request_error",
            Some("unsupported_parameter"),
            Some("prompt".into()),
        )
    };
    let encode = |text: &str| {
        crate::obs::ttft::timed(&crate::obs::ttft::ENCODE, || {
            bundle
                .tokenizer()
                .encode_with_special_tokens(text, req.add_special_tokens)
        })
    };
    let prompt_ids = match &req.prompt {
        PromptSpec::Text(text) => encode(text),
        PromptSpec::Tokens(ids) => ids.clone(),
        PromptSpec::Batch(v) => match v.as_slice() {
            [one] => encode(one),
            _ => return batch_refusal(),
        },
        PromptSpec::TokenBatch(v) => match v.as_slice() {
            [one] => one.clone(),
            _ => return batch_refusal(),
        },
    };
    // A caller-supplied id is DATA, not a promise. `PromptSpec::Tokens`/`TokenBatch` hand these
    // straight to the embedding gather, where the kernel indexes the table with a SIGNED offset:
    // 4294967295 arrives as -1 and reads BEFORE the embedding table, and any id past the vocabulary
    // reads past its end. `bench.rs`'s `validate_input` has always checked this; the serving path
    // did not, so the same malformed request was a 400 through one door and an out-of-bounds device
    // read through the other.
    let vocab = bundle.tokenizer().vocab_size();
    if vocab > 0 {
        if let Some(&id) = prompt_ids.iter().find(|&&id| id as usize >= vocab) {
            return crate::serve::api_error(
                axum::http::StatusCode::BAD_REQUEST,
                format!("prompt token id {id} is outside the vocabulary size {vocab}"),
                "invalid_request_error",
                Some("invalid_prompt"),
                Some("prompt".into()),
            );
        }
    }
    if prompt_ids.is_empty() {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "prompt encodes to zero tokens",
            "invalid_request_error",
            Some("invalid_prompt"),
            Some("prompt".into()),
        );
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
    if let Err(err) = mux.submit_arrived(job, t_arrive) {
        return match err {
            crate::serve::mux::SubmitError::Full(_) => {
                crate::serve::api_error(
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    "model request queue full",
                    "rate_limit_error",
                    Some("server_overloaded"),
                    None,
                )
            }
            crate::serve::mux::SubmitError::Closed(_) => crate::serve::api_error(
                axum::http::StatusCode::SERVICE_UNAVAILABLE,
                "model dispatcher unavailable",
                "server_error",
                None,
                None,
            ),
        };
    }

    let id = request_id();
    let created = now_secs();
    if req.stream {
        let include_usage = req.stream_options.map(|o| o.include_usage).unwrap_or(false);
        sse_response(
            id,
            req.model,
            rx,
            include_usage,
            t_arrive,
            n_prompt,
            created,
        )
        .into_response()
    } else {
        buffer_and_reply(id, req.model, rx, response_prompt_ids, created).await
    }
}

async fn buffer_and_reply(
    request_id: String,
    model: String,
    mut rx: stream_mod::ChunkReceiver,
    prompt_token_ids: Option<Vec<u32>>,
    created: u64,
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
                finish = Some(reason);
                usage = Some(u.into());
                break;
            }
            StreamChunk::Err(e) => {
                tracing::warn!(%model, error = %e, partial_chars = text.len(), "completion stream error");
                return crate::serve::api_error_for(&e);
            }
        }
    }
    let Some(finish) = finish else {
        tracing::warn!(%model, partial_chars = text.len(), "completion stream ended without terminal chunk");
        return crate::serve::api_error(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "generation stream ended without a finish reason (slot cut)",
            "server_error",
            None,
            None,
        );
    };
    Json(CompletionResponse {
        id: request_id,
        object: "text_completion",
        created,
        model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            logprobs: None,
            finish_reason: Some(finish.as_openai()),
            // `Preempted` widens to "length" on the wire; without this the
            // caller cannot tell an operator-forced stop from max_tokens.
            x_plow_finish_reason: finish.is_vendor_specific().then(|| finish.as_str()),
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
    created: u64,
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
                                x_plow_finish_reason: None,
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
                            finish_reason: Some(reason.as_openai()),
                            x_plow_finish_reason: reason
                                .is_vendor_specific()
                                .then(|| reason.as_str()),
                        }],
                        include_usage.then(|| usage.into()),
                        true,
                    ),
                    StreamChunk::Err(e) => {
                        // An error object in its own frame and NO `[DONE]`, not
                        // the error text dressed as generated output with
                        // `finish_reason: "stop"` — see the matching comment on
                        // the chat endpoint for why that scored as success.
                        tracing::warn!(%model, error = %e, "completion SSE stream error");
                        let body = crate::serve::openai::ApiErrorBody::new(
                            e.to_string(),
                            "server_error",
                            None,
                            None,
                        );
                        let data = serde_json::to_string(&body).unwrap_or_else(|_| {
                            "{\"error\":{\"message\":\"stream error\",\"type\":\"server_error\"}}"
                                .to_string()
                        });
                        st.done = true;
                        return Some((Ok(Event::default().data(data)), st));
                    }
                };
                let frame = CompletionResponse {
                    id: request_id,
                    object: "text_completion",
                    created,
                    model: model.clone(),
                    choices: choice,
                    usage: None,
                    token_ids: None,
                };
                if let Some(usage) = tail_usage {
                    let usage_frame = CompletionResponse {
                        id: frame.id.clone(),
                        object: "text_completion",
                        created,
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
        CompletionChoice, CompletionRequest, CompletionResponse, CompletionTokenIds, PromptSpec,
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
        assert!(matches!(&req.prompt, PromptSpec::Text(t) if t == "raw prompt"));
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
            created: 0,
            model: "model".into(),
            choices: vec![CompletionChoice {
                index: 0,
                text: "x".into(),
                logprobs: None,
                finish_reason: Some("length"),
                x_plow_finish_reason: None,
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
