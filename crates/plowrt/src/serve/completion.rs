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

/// `X-Request-Id` / `X-Session-Id`: echoed on the response; a session's requests resume the rows
/// its previous request retained.
pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    req: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let mut ids = match crate::serve::session::RequestIds::from_headers(&headers) {
        Ok(ids) => ids,
        Err(e) => {
            return crate::serve::api_error(axum::http::StatusCode::BAD_REQUEST, e, "invalid_request_error", Some("invalid_value"), None)
        }
    };
    if let Err(e) = req.as_ref().map_or(Ok(()), |Json(r)| ids.apply_body(&r.route)) {
        return crate::serve::api_error(axum::http::StatusCode::BAD_REQUEST, e, "invalid_request_error", Some("invalid_value"), Some("session_id".into()));
    }
    if let Some(r) = crate::serve::overload::gate(&ids) {
        return r;
    }
    let mut response = completions_with(state, req, &ids).await;
    ids.stamp(&mut response);
    response
}

async fn completions_with(
    state: Arc<AppState>,
    // `Result<Json<..>, JsonRejection>` rather than `Json<..>`: axum's default
    // rejection is a PLAIN-TEXT 400/415/422, and a client that calls
    // `resp.json()` on a 4xx — every OpenAI SDK does — raises a decode error
    // instead of showing the user what was wrong with their request.
    req: Result<Json<CompletionRequest>, axum::extract::rejection::JsonRejection>,
    ids: &crate::serve::session::RequestIds,
) -> Response {
    let Json(mut req) = match req {
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

    // Aliases, resolved once before any slug-keyed lookup — see `chat`.
    let requested_model = req.model.clone();
    if let Some(canonical) = state.registry.resolve(&req.model) {
        req.model = canonical;
    }

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
    let logprobs = match crate::serve::logprobs::parse_completion(req.logprobs.as_ref(), req.logprobs_mode.as_deref()) {
        Ok(lp) => lp,
        Err((msg, param)) => {
            return crate::serve::api_error(
                axum::http::StatusCode::BAD_REQUEST,
                msg,
                "invalid_request_error",
                Some("invalid_value"),
                Some(param.into()),
            )
        }
    };
    if logprobs.is_some() && !cfg!(any(feature = "cuda", feature = "cpu")) {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "`logprobs` is served by the CUDA and CPU engines only",
            "invalid_request_error",
            Some("unsupported_parameter"),
            Some("logprobs".into()),
        );
    }
    if !state.residency(&req.model).admits() {
        return crate::serve::api_error(axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "Model is explicitly unloaded or unloading", "server_error", Some("model_unloaded"), Some("model".into()));
    }
    #[cfg(feature = "cuda")]
    {
        use crate::serve::manager::EnsureError;
        let ensured = match state.dp_set(&req.model) {
            Some(set) => Some(state.dp_admit(set).await),
            None => match state.manager_for(&req.model) {
                Some(mgr) if mgr.manages(&req.model) => Some(mgr.ensure_resident(&req.model).await),
                _ => None,
            },
        };
        {
            if let Some(Err(e)) = ensured {
                return match e {
                    EnsureError::WontFit { .. } | EnsureError::SwitchTimeout(_) => (
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

    if let Err(e) = req
        .sampling
        .validate()
        .and_then(|()| crate::serve::openai::validate_limits(req.max_tokens, req.stop.as_ref()))
    {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            e.message,
            "invalid_request_error",
            Some("invalid_value"),
            Some(e.field.into()),
        );
    }

    let t_arrive = std::time::Instant::now();
    crate::obs::ttft::reset();

    // Model defaults first, request on top — same resolution order as chat.
    let mut gen = crate::serve::GenParams {
        params: state
            .registry
            .get(&req.model)
            .map(|b| b.serving().default_sampling.clone())
            .unwrap_or_default(),
        ..Default::default()
    };
    if let Some(m) = req.max_tokens {
        gen.max_tokens = m as usize;
    }
    req.sampling.apply(&mut gen.params);
    if let Some(ignore) = req.ignore_eos {
        gen.ignore_eos = ignore;
    }
    if let Some(stop) = &req.stop {
        gen.stop = stop.list();
    }
    gen.seed = req.seed;
    gen.params.logprobs = logprobs;
    gen.min_tokens = req.sampling.min_tokens.unwrap_or(0) as usize;
    gen.stop_token_ids = req.sampling.stop_token_ids.clone().unwrap_or_default();
    if gen.min_tokens > gen.max_tokens {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            format!(
                "`min_tokens` ({}) exceeds `max_tokens` ({}); the request could never finish",
                gen.min_tokens, gen.max_tokens
            ),
            "invalid_request_error",
            Some("invalid_value"),
            Some("min_tokens".into()),
        );
    }

    let dp = state.dp_set(&req.model);
    let direct = if dp.is_some() { None } else { state.mux(&req.model) };
    let (true, Ok(bundle)) = (dp.is_some() || direct.is_some(), state.registry.get(&req.model)) else {
        return crate::serve::api_error(
            axum::http::StatusCode::NOT_FOUND,
            format!("no model registered for '{}'.", req.model),
            "invalid_request_error",
            Some("model_not_found"),
            Some("model".into()),
        );
    };
    let ingress = direct.as_ref().map(|m| m.ingress());
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
            crate::serve::encode_prompt(text, |t| {
                bundle.tokenizer().encode_with_special_tokens(t, req.add_special_tokens)
            })
        })
    };
    let text_bytes = match &req.prompt {
        PromptSpec::Text(text) => text.len(),
        PromptSpec::Batch(v) => v.iter().map(String::len).sum(),
        PromptSpec::Tokens(_) | PromptSpec::TokenBatch(_) => 0,
    };
    if let Some(e) = crate::serve::prompt_bytes_overflow(state.max_ctx(&req.model), bundle.tokenizer().max_token_bytes(), text_bytes) {
        return crate::serve::api_error_for(&e);
    }
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
    if req.max_tokens.is_none() {
        gen.max_tokens = crate::serve::default_max_tokens(gen.max_tokens, state.max_ctx(&req.model), n_prompt);
    }
    if let Some(e) = crate::serve::context_overflow(state.max_ctx(&req.model), n_prompt, gen.max_tokens) {
        return crate::serve::api_error_for(&e);
    }
    let routed;
    let (mut rank, mut prefix, mut _pick) = (0, None, None);
    let mux = match (&direct, dp) {
        (Some(m), _) => m,
        (None, Some(set)) => match state.dp_route(set, ids.session.as_deref(), Some(&prompt_ids), 0) {
            Some((r, m, key, pick)) => {
                (rank, prefix, _pick) = (r, key, Some(pick));
                routed = m;
                &routed
            }
            None => return crate::serve::chat::dp_unavailable(&req.model),
        },
        (None, None) => unreachable!("checked at lookup"),
    };
    let key = dp.map_or(req.model.as_str(), |set| set.ranks[rank].key.as_str());
    let (tx, rx) = stream_mod::channel();
    let response_prompt_ids = req.return_token_ids.then(|| prompt_ids.clone());
    let lp_fmt = logprobs.map(|_| crate::serve::logprobs::TokenText {
        tok: bundle.tokenizer().clone(),
        as_ids: req.return_tokens_as_token_ids.unwrap_or(false),
    });
    let Some(in_flight) = ids.begin(&req.model) else {
        return crate::serve::api_error(
            axum::http::StatusCode::CONFLICT,
            format!("request {} is already in flight in this session", ids.request),
            "invalid_request_error",
            Some("duplicate_request_id"),
            None,
        );
    };
    let mut run = crate::serve::turns::StageRun::start(
        ids,
        crate::serve::turns::Kind::Llm,
        &req.model,
        Some(state.model_metrics(key)),
        t_arrive,
        true,
    );
    let (report, report_rx) = ids.report();
    let session = ids.session.as_ref().and_then(|_| ids.ticket(crate::serve::session::row_keys(&prompt_ids, &[], &[]), report));
    let job = crate::serve::mux::Job {
        prompt_ids,
        gen,
        arrived: std::time::Instant::now(),
        respond: tx,
        opts: crate::serve::mux::JobOpts {
            session,
            turn: run.key(),
            continuing: run.continuing(),
            prefix: prefix.take(),
            ..Default::default()
        },
    };
    if crate::obs::host::on() {
        crate::obs::host::submitted(n_prompt, t_arrive.elapsed());
    }
    if let Err(err) = state.submit_routed(dp.map(|s| (&**s, rank)), ids.session.as_deref(), mux, job, t_arrive, ingress) {
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
    let cache = crate::serve::session::CacheOutcome::received(report_rx).await;
    run.admitted(cache.and_then(|c| c.at));
    let mut response = if req.stream {
        let include_usage = req.stream_options.map(|o| o.include_usage).unwrap_or(false);
        let stamped = run.headers();
        let sse = sse_response(
            id,
            requested_model.clone(),
            rx,
            include_usage,
            t_arrive,
            n_prompt,
            created,
            lp_fmt,
            run,
        );
        let mut response = crate::serve::session::hold_until_sent(sse.into_response(), in_flight);
        response.headers_mut().extend(stamped);
        response
    } else {
        buffer_and_reply(id, requested_model, rx, response_prompt_ids, created, lp_fmt, run).await
    };
    if let Some(cache) = cache {
        cache.stamp(&mut response);
    }
    response
}

async fn buffer_and_reply(
    request_id: String,
    model: String,
    mut rx: stream_mod::ChunkReceiver,
    prompt_token_ids: Option<Vec<u32>>,
    created: u64,
    lp_fmt: Option<crate::serve::logprobs::TokenText>,
    mut run: crate::serve::turns::StageRun,
) -> Response {
    let mut text = String::new();
    let mut lps = crate::serve::logprobs::CompletionLogprobs::default();
    let mut completion_token_ids = Vec::new();
    let mut finish = None;
    let mut usage = None;
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { id, text: delta, logprobs } => {
                run.first();
                if let (Some(fmt), Some(lp)) = (&lp_fmt, &logprobs) {
                    lps.push(fmt, id, lp, text.len());
                }
                text.push_str(&delta);
                if prompt_token_ids.is_some() && id != crate::serve::stream::TEXT_ONLY {
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
    run.done();
    let mut response = Json(CompletionResponse {
        id: request_id,
        object: "text_completion",
        created,
        model,
        choices: vec![CompletionChoice {
            index: 0,
            text,
            logprobs: lp_fmt.map(|_| serde_json::to_value(lps).unwrap_or_default()),
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
    .into_response();
    run.stamp(&mut response);
    response
}

fn sse_response(
    request_id: String,
    model: String,
    rx: stream_mod::ChunkReceiver,
    include_usage: bool,
    t_arrive: std::time::Instant,
    n_prompt: usize,
    created: u64,
    lp_fmt: Option<crate::serve::logprobs::TokenText>,
    run: crate::serve::turns::StageRun,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    struct SseState {
        rx: stream_mod::ChunkReceiver,
        /// Streamed text length so far (`text_offset`).
        offset: usize,
        first: bool,
        done: bool,
        pending: std::collections::VecDeque<Event>,
        run: crate::serve::turns::StageRun,
    }
    let body = stream::unfold(
        SseState {
            rx,
            offset: 0,
            first: true,
            done: false,
            pending: std::collections::VecDeque::new(),
            run,
        },
        move |mut st| {
            let model = model.clone();
            let request_id = request_id.clone();
            let lp_fmt = lp_fmt.clone();
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
                    StreamChunk::Token { id, text, logprobs } => {
                        let logprobs = lp_fmt.as_ref().map(|fmt| {
                            let mut lps = crate::serve::logprobs::CompletionLogprobs::default();
                            if let Some(lp) = &logprobs {
                                lps.push(fmt, id, lp, st.offset);
                            }
                            serde_json::to_value(lps).unwrap_or_default()
                        });
                        st.offset += text.len();
                        if st.first {
                            st.first = false;
                            st.run.first();
                            crate::obs::ttft::dump(t_arrive.elapsed().as_nanos() as u64, n_prompt);
                            crate::obs::pfx::report();
                            if crate::obs::host::on() {
                                crate::obs::host::first_frame(n_prompt, t_arrive.elapsed());
                            }
                        }
                        (
                            vec![CompletionChoice {
                                index: 0,
                                text,
                                logprobs,
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
                    st.run.done();
                    st.pending.push_back(st.run.sse_comment());
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
            "best_of": 1,
            "session_id": "s",
            "metadata": {"turn_id": "t"}
        }))
        .unwrap();
        assert_eq!(req.route.session_id.as_deref(), Some("s"));
        assert!(req.route.metadata.as_ref().is_some_and(|m| m["turn_id"] == "t"));
        assert_eq!(req.sampling.temperature, Some(0.0));
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
