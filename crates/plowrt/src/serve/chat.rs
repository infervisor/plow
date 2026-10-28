//! §G `/v1/chat/completions` — streaming (SSE) and non-streaming.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream;

use crate::serve::openai::*;
use crate::serve::stream::{self as stream_mod, StreamChunk};
use crate::serve::{status_for, AppState};

/// Seeded per PROCESS, not from zero. A counter starting at 0 made the first
/// response of every server exactly `chatcmpl-0000000000000000` and made ids
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

fn request_id() -> String {
    format!("chatcmpl-{:016x}", REQ_SEQ.fetch_add(1, Ordering::Relaxed))
}

/// Handler: dispatches to the streaming or non-streaming path.
/// `X-Request-Id` / `X-Session-Id`: echoed on the response; a session's requests resume the rows
/// its previous request retained.
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: axum::http::HeaderMap,
    req: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let mut ids = match crate::serve::session::RequestIds::from_headers(&headers) {
        Ok(ids) => ids,
        Err(e) => {
            return crate::serve::api_error(
                axum::http::StatusCode::BAD_REQUEST,
                e,
                "invalid_request_error",
                Some("invalid_value"),
                None,
            )
        }
    };
    if let Err(e) = req
        .as_ref()
        .map_or(Ok(()), |Json(r)| ids.apply_body(&r.route))
    {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            e,
            "invalid_request_error",
            Some("invalid_value"),
            Some("session_id".into()),
        );
    }
    if let Some(r) = crate::serve::overload::gate(&ids) {
        return r;
    }
    let mut response = chat_completions_with(state, req, &ids).await;
    ids.stamp(&mut response);
    response
}

async fn chat_completions_with(
    state: Arc<AppState>,
    // `Result<Json<..>, JsonRejection>` rather than `Json<..>`: axum's default
    // rejection is a PLAIN-TEXT 400/415/422, and a client that calls
    // `resp.json()` on a 4xx — every OpenAI SDK does — raises a decode error
    // instead of showing the user what was wrong with their request.
    req: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
    ids: &crate::serve::session::RequestIds,
) -> Response {
    let Json(mut req) = match req {
        Ok(r) => r,
        Err(e) => return crate::serve::json_rejection(e),
    };

    // ALIASES. `model` may be a name this bundle merely answers to (vLLM's
    // `--served-model-name`). Resolve it ONCE, here, before anything keyed by
    // slug runs — residency, the manager, the mux, the metrics map — so an
    // alias never grows a second key for the same model. The response still
    // echoes the name the client sent, as vLLM does.
    req.normalize();
    let requested_model = req.model.clone();
    if let Some(canonical) = state.registry.resolve(&req.model) {
        req.model = canonical;
    }
    // Per-model request defaults (`--chat-defaults`), under the request's own fields.
    let defaults_of = state.registry.get(&req.model).ok();
    let defaults = defaults_of.as_deref().and_then(|b| b.serving().chat.get());
    if let Some(d) = defaults {
        req.max_tokens = req
            .max_tokens
            .or(d.max_tokens)
            .or(d.generation.and_then(|g| g.max_new_tokens));
        if let Some(kw) = &d.chat_template_kwargs {
            let k = req
                .chat_template_kwargs
                .get_or_insert_with(Default::default);
            for (key, v) in kw {
                k.entry(key.clone()).or_insert_with(|| v.clone());
            }
        }
        req.reasoning_effort = req.reasoning_effort.take().or(d.reasoning_effort.clone());
        req.reasoning_format = req.reasoning_format.take().or(d.reasoning_format.clone());
        req.include_reasoning = req.include_reasoning.or(d.include_reasoning);
        req.reasoning_budget_message = req
            .reasoning_budget_message
            .take()
            .or(d.reasoning_budget_message.clone());
        // A default `reasoning_effort` maps to the template switch the way the request's does,
        // under the request's own kwargs.
        req.normalize();
    }
    let think_budget = req
        .thinking_token_budget
        .or(defaults.and_then(|d| d.thinking_token_budget));
    if think_budget == Some(0) {
        // Truncated thinking does worse than none at small budgets, so a zero budget renders the
        // template's thinking-off prompt rather than forcing the close at once.
        req.chat_template_kwargs
            .get_or_insert_with(Default::default)
            .insert("enable_thinking".into(), false.into());
    }

    // S1 multi-model: a request for a managed, non-resident model triggers
    // the switch HERE (evict LRU + load), before the prompt is built — so the
    // template choice below sees the engine. Resident models pass through on
    // the manager's lock-free fast path. A switch that cannot fit sheds with
    // 503 + Retry-After (the client should back off, not hammer the planner).
    if let Some(r) = crate::serve::models::refuse_unserved(&state, &req.model, "chat/completions") {
        return r;
    }
    if !state.residency(&req.model).admits() {
        return crate::serve::api_error(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "Model is explicitly unloaded or unloading",
            "server_error",
            Some("model_unloaded"),
            Some("model".into()),
        );
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
                    // No `retry-after`: retrying cannot help. The model is
                    // resident-capable but an operator took it down, and only
                    // an explicit load brings it back.
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
    // Again once a switch has bound the model: a non-resident one read as text above.
    if let Some(r) = crate::serve::models::refuse_unserved(&state, &req.model, "chat/completions") {
        return r;
    }

    // REFUSE WHAT THIS SERVER CANNOT HONOR, rather than dropping it.
    //
    // serde has no `deny_unknown_fields` on these DTOs, so every parameter the
    // handler did not read was silently discarded and the request came back
    // 200 with a confidently wrong answer. `tools` was the worst of them: a
    // client that called `bind_tools()` got fluent prose and `finish_reason:
    // "stop"`, which scores as success everywhere. This is the same reasoning
    // as the image refusal below, applied to the rest of the surface.
    if let Some(n) = req.n {
        if n != 1 {
            return crate::serve::api_error(
                axum::http::StatusCode::BAD_REQUEST,
                format!("n={n} is not supported; this server returns exactly one choice"),
                "invalid_request_error",
                Some("unsupported_parameter"),
                Some("n".into()),
            );
        }
    }
    // `{"type": "text"}` is the default format and asks for nothing; SDK wrappers send it.
    if req
        .response_format
        .as_ref()
        .is_some_and(|v| !v.is_null() && v.get("type").and_then(|t| t.as_str()) != Some("text"))
    {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "`response_format` is not implemented by this server; it is refused rather than \
             ignored, because ignoring it returns a confidently wrong answer",
            "invalid_request_error",
            Some("unsupported_parameter"),
            Some("response_format".into()),
        );
    }
    // TOOLS: served when the model's own template renders them and its call syntax is one
    // `serve::tools` parses; refused otherwise (see `tools::request::plan`).
    let tool_support = if state.chat_templated(&req.model) {
        state
            .registry
            .get(&req.model)
            .ok()
            .and_then(|b| b.chat_template().map(|t| t.tools))
    } else {
        None
    };
    let tool_plan = match crate::serve::tools::request::plan(&req, tool_support) {
        Ok(p) => p,
        Err(r) => {
            return crate::serve::api_error(
                axum::http::StatusCode::BAD_REQUEST,
                r.message,
                "invalid_request_error",
                Some(r.code),
                Some(r.param.into()),
            )
        }
    };

    // OpenAI's schema requires at least one message, and answering an empty
    // conversation means generating from a bare generation prompt: the model
    // invents a question and answers it, with a 200 and a plausible body.
    // `/v1/completions` already refuses its empty-prompt equivalent.
    if req.messages.is_empty() {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "`messages` must contain at least one message",
            "invalid_request_error",
            Some("invalid_prompt"),
            Some("messages".into()),
        );
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
    if req
        .reasoning_format
        .as_deref()
        .is_some_and(|f| !matches!(f, "none" | "auto" | "deepseek"))
    {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "`reasoning_format` must be \"none\" (the trace stays in `content`), \"auto\" or \"deepseek\"",
            "invalid_request_error",
            Some("invalid_value"),
            Some("reasoning_format".into()),
        );
    }
    if let Err((field, message)) = crate::serve::config::check_thinking(
        defaults_of.as_deref().and_then(|b| b.reasoning().markers()),
        req.thinking_token_budget,
        req.reasoning_end.as_deref(),
    )
    .and_then(|()| match req.max_reply_tokens {
        Some(0) => Err((
            "max_reply_tokens",
            "`max_reply_tokens` must be at least 1".into(),
        )),
        _ => Ok(()),
    }) {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            message,
            "invalid_request_error",
            Some("invalid_value"),
            Some(field.into()),
        );
    }

    // §TTFT: the clock the breakdown is measured against starts HERE, at the
    // first line of the handler that has the request body — everything before
    // it (accept, read, JSON decode) is axum's and shows up as UNACCOUNTED.
    let t_arrive = std::time::Instant::now();
    crate::obs::ttft::reset();

    // Models served by the GPU engine get their real chat template (the
    // tokenizer resolves the markers through `added_tokens`); the CPU
    // reference path keeps the simple role-prefix flatten — its logits are a
    // stand-in, so a template would be costume jewelry there.
    let mut template_error: Option<String> = None;
    let mut request_error: Option<crate::serve::tools::request::Refusal> = None;
    let mut no_template: Option<String> = None;
    let render_opts = crate::serve::template::RenderOpts {
        kwargs: req.chat_template_kwargs.clone().unwrap_or_default(),
        reasoning_effort: req.reasoning_effort.clone(),
        continue_final_message: req.continue_final_message.unwrap_or(false),
        no_generation_prompt: req.add_generation_prompt == Some(false),
        tools: tool_plan.template_tools.clone(),
    };
    let mut prompt = crate::obs::ttft::timed(&crate::obs::ttft::TEMPLATE, || {
        if state.chat_templated(&req.model) {
            let tok = state.registry.get(&req.model).ok();
            match chat_prompt(tok.as_deref(), &req.messages, &render_opts) {
                Ok(p) => p,
                Err(e) => {
                    match e {
                        PromptError::Template(e) => template_error = Some(e),
                        PromptError::Request(r) => request_error = Some(r),
                        PromptError::NoTemplate(e) => no_template = Some(e),
                    }
                    String::new()
                }
            }
        } else {
            let mut prompt = String::new();
            for m in &req.messages {
                if !prompt.is_empty() {
                    prompt.push_str("\n\n");
                }
                prompt.push_str(&m.role);
                prompt.push_str(":\n");
                prompt.push_str(&m.text());
            }
            prompt
        }
    });

    if let Some(e) = no_template {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            format!("{e}; use /v1/completions"),
            "invalid_request_error",
            Some("chat_template_missing"),
            Some("messages".into()),
        );
    }
    if let Some(r) = request_error {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            r.message,
            "invalid_request_error",
            Some(r.code),
            Some(r.param.into()),
        );
    }
    if let Some(e) = template_error {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            format!("the model's chat template rejected this conversation: {e}"),
            "invalid_request_error",
            Some("invalid_message_sequence"),
            Some("messages".into()),
        );
    }

    // Build generation controls: the CHECKPOINT's defaults first, then the
    // request's own fields on top. Resolution order is
    // `request > model default > server default` — before this, the model
    // default step did not exist and every checkpoint was served at the stock
    // temperature 1.0 / top_p 1.0 whatever its `generation_config.json` said.
    let mut gen = crate::serve::GenParams {
        params: state
            .registry
            .get(&req.model)
            .map(|b| crate::serve::config::base_sampling(&b))
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
    gen.params.logprobs = match crate::serve::logprobs::parse_chat(
        req.logprobs.as_ref(),
        req.top_logprobs.as_ref(),
        req.logprobs_mode.as_deref(),
    ) {
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
    if gen.params.logprobs.is_some()
        && !cfg!(any(feature = "cuda", feature = "cpu", feature = "hsa"))
    {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "`logprobs` is served by the CUDA, HSA and CPU engines only",
            "invalid_request_error",
            Some("unsupported_parameter"),
            Some("logprobs".into()),
        );
    }
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

    // MEDIA PARTS (`serve::mm`): image and audio parts are encoded by the encoders the model's
    // packet carries and replace the placeholders its own template renders. A part the model
    // has no encoder for is refused before it becomes a token: dropping it (`Content::as_text`
    // keeps only text) would answer about the wrong question, fluently.
    // Encoding waits for the rank: its encoders and slab are that engine's own.
    let mm_contract = match crate::serve::mm::media_parts(&req.messages).as_slice() {
        [] => None,
        parts => {
            let contract = state
                .registry
                .get(&req.model)
                .ok()
                .filter(|_| state.has_gpu_engine(&req.model))
                .and_then(|b| b.serve().multimodal.clone());
            let missing = parts
                .iter()
                .filter_map(|p| p.media_kind())
                .find(|k| contract.as_ref().is_none_or(|c| c.modality(k).is_none()));
            if let Some(kind) = missing {
                return crate::serve::api_error(
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("unsupported content type for this model: {kind}"),
                    "invalid_request_error",
                    Some("unsupported_parameter"),
                    Some("messages[].content".into()),
                );
            }
            contract
        }
    };

    // Route to the per-model muxer. Tokens stream back as `StreamChunk`s over
    // an mpsc — the muxer produces one per generated token, ending with `Done`.
    // A data-parallel model picks its rank after tokenizing (the prefix probe needs the ids).
    let dp = state.dp_set(&req.model);
    let direct = if dp.is_some() {
        None
    } else {
        state.mux(&req.model)
    };
    let (true, Ok(bundle)) = (
        dp.is_some() || direct.is_some(),
        state.registry.get(&req.model),
    ) else {
        return crate::serve::api_error(
            axum::http::StatusCode::NOT_FOUND,
            format!("no model registered for '{}'.", req.model),
            "invalid_request_error",
            Some("model_not_found"),
            Some("model".into()),
        );
    };
    let ingress = direct.as_ref().map(|m| m.ingress());
    // Tokenize HERE, on the handler task — the dispatcher loop is the
    // serialized decode critical path and must never encode a long prompt.
    // The packet's reasoning markers (`<think>` for a legacy packet): the splitter's own
    // `Deciding` state settles after the first few bytes of generation, so a model that never
    // reasons costs at most a marker-length delay on its first chunk. What the PROMPT decides is
    // only whether the trace is already open (GLM leaves it dangling; Qwen3 and DeepSeek-R1 emit
    // the marker themselves).
    let reasoning_mode = crate::serve::reasoning::ReasoningMode::for_bundle(&bundle);
    let mut reasoning_open = reasoning_mode.prompt_opens(&prompt);
    // Reasoning on in the rendered prompt: the trace is left open, or the template's own switch
    // (Gemma 4 / Qwen3 `enable_thinking`) is on for this request.
    let thinking = reasoning_mode.markers().is_some()
        && (reasoning_open
            || (state.chat_templated(&req.model)
                && bundle
                    .chat_template()
                    .is_some_and(|t| t.thinking_requested(&render_opts.kwargs))));
    // A forced `tool_choice` writes the call's opener for the model, so the generation starts
    // inside the call (after closing a trace the prompt left open), and is validated at the end.
    let opener = tool_plan.parse.as_ref().and_then(|s| s.opener());
    // A seeded trace: plow writes the trace's opening itself, so the split knows it is open.
    // After a tool result the template already opened it and only the seed text is added.
    if let (true, None, Some(prefix), Some((open, _))) = (
        thinking,
        &opener,
        defaults.and_then(|d| d.thought_prefix.as_deref()),
        reasoning_mode.markers(),
    ) {
        if reasoning_open {
            let ws = prompt[prompt.trim_end().len()..].to_string();
            let seed = &prefix[open.len()..];
            prompt.push_str(seed.strip_prefix(ws.as_str()).unwrap_or(seed));
        } else {
            prompt.push_str(prefix);
            reasoning_open = true;
        }
    }
    if let Some(o) = &opener {
        if reasoning_open {
            prompt.push_str(reasoning_mode.close_marker().unwrap_or_default());
            reasoning_open = false;
        }
        prompt.push_str(o);
    }
    // llama.cpp's `reasoning_format: "none"`: the trace is not split out of `content`.
    let split_mode = if req.reasoning_format.as_deref() == Some("none") {
        crate::serve::reasoning::ReasoningMode::None
    } else {
        reasoning_mode
    };
    // Markers that are one token each (Qwen3's `<think>`, Gemma 4's special-led channel) cannot
    // come from prose: an open marker after the answer began opens another trace block, and a
    // stray close marker is framing, not answer text.
    let reenter = split_mode != crate::serve::reasoning::ReasoningMode::None
        && bundle.think_markers().is_ok();
    let tool_stream = response_stream(
        &tool_plan,
        tool_support,
        split_mode,
        reasoning_open,
        reenter,
        &bundle,
        req.stream,
        opener.as_deref(),
    );
    gen.keep_special_tokens = tool_stream.is_some();
    let include_reasoning = req.include_reasoning.unwrap_or(true);
    if let Some(e) = crate::serve::prompt_bytes_overflow(
        state.max_ctx(&req.model),
        bundle.tokenizer().max_token_bytes(),
        prompt.len(),
    ) {
        return crate::serve::api_error_for(&e);
    }
    let mut prompt_ids = crate::obs::ttft::timed(&crate::obs::ttft::ENCODE, || {
        crate::serve::encode_prompt(&prompt, |p| bundle.tokenizer().encode(p))
    });
    let mm_pending = match &mm_contract {
        Some(c) => match crate::serve::mm::expand(
            c,
            &req.messages,
            &mut prompt_ids,
            crate::serve::mm::Limits::from_env(),
        )
        .await
        {
            Ok(p) => p,
            Err(e) => return mm_error(e),
        },
        None => None,
    };
    let n_prompt = prompt_ids.len();
    if prompt_ids.is_empty() {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "the conversation renders to zero prompt tokens",
            "invalid_request_error",
            Some("invalid_prompt"),
            Some("messages".into()),
        );
    }
    if req.max_tokens.is_none() {
        gen.max_tokens =
            crate::serve::default_max_tokens(gen.max_tokens, state.max_ctx(&req.model), n_prompt);
    }
    if let Some(e) =
        crate::serve::context_overflow(state.max_ctx(&req.model), n_prompt, gen.max_tokens)
    {
        return crate::serve::api_error_for(&e);
    }
    // THINKING CONTROL: a budget per trace block (thinking on only) and a cap on the reply. The
    // mux forces `reasoning_end` once a block's budget is spent.
    let reply_cap = req
        .max_reply_tokens
        .or(defaults.and_then(|d| d.max_reply_tokens));
    let asked = think_budget.filter(|&b| b > 0);
    let stop_blank = defaults.is_some_and(|d| d.reply_stop_blank_line);
    let live = thinking && defaults.is_some_and(|d| d.live_reasoning_end);
    let steer = thinking && defaults.is_some_and(|d| d.steers());
    if asked.is_some() || reply_cap.is_some() || stop_blank || live || steer {
        let field = if asked.is_some() {
            "thinking_token_budget"
        } else {
            "max_reply_tokens"
        };
        let refuse = |message: String, field: &str| {
            crate::serve::api_error(
                axum::http::StatusCode::BAD_REQUEST,
                message,
                "invalid_request_error",
                Some("invalid_value"),
                Some(field.into()),
            )
        };
        if req
            .max_reply_tokens
            .is_some_and(|c| c as usize > gen.max_tokens)
        {
            return refuse(
                format!(
                    "`max_reply_tokens` exceeds `max_tokens` ({})",
                    gen.max_tokens
                ),
                "max_reply_tokens",
            );
        }
        let markers = match bundle.think_markers() {
            Ok(m) => m,
            Err(e) => {
                return refuse(
                    format!("`{field}` is not available for this model: {e}"),
                    field,
                )
            }
        };
        let budget = asked.filter(|_| thinking);
        // Thinking off for this request leaves a budget nothing to bound.
        if budget.is_some() || reply_cap.is_some() || stop_blank || live || steer {
            // llama.cpp's `reasoning_budget_message` is the bridge without the close marker.
            let message_end = req
                .reasoning_budget_message
                .as_ref()
                .map(|m| format!("{m}{}", reasoning_mode.close_marker().unwrap_or_default()));
            // Unset everywhere: a bridge sentence before the close marker. A bare close leaves the
            // model mid-thought and it carries on deliberating in the answer (Qwen3-0.6B, 20
            // prompts, budget 32: 7 answers vs none with the bridge). `reasoning_end: ""` is the
            // bare marker.
            let default_end = format!(
                "{}{}",
                crate::serve::think::DEFAULT_BRIDGE,
                reasoning_mode.close_marker().unwrap_or_default()
            );
            let end = req
                .reasoning_end
                .as_deref()
                .or(message_end.as_deref())
                .or(defaults.and_then(|d| d.reasoning_end.as_deref()))
                .unwrap_or(&default_end);
            let mut forced_end = bundle.tokenizer().encode_with_special_tokens(end, false);
            if forced_end.last() != Some(&markers.close) {
                forced_end.push(markers.close);
            }
            // The trace and the reply share `max_tokens`: a budget that leaves the reply no room
            // would end every turn inside its trace (`finish_reason: length`, no answer). Room
            // counted for the first block: its open marker and label (2), the budget, the newline
            // grace, the forced end, then the reply (its cap, else one token).
            if let Some(b) = budget {
                let grace = defaults.map_or(0, |d| d.newline_grace) as usize;
                let reply = reply_cap.map_or(1, |c| c as usize);
                let need = 2 + b as usize + grace + forced_end.len() + reply;
                if need > gen.max_tokens {
                    return refuse(
                        format!(
                            "`thinking_token_budget` ({b}) + newline grace ({grace}) + the forced end \
                             ({} tokens) + the reply ({reply}) need {need} tokens, more than \
                             `max_tokens` ({}): the answer would never be reached",
                            forced_end.len(),
                            gen.max_tokens
                        ),
                        "thinking_token_budget",
                    );
                }
            }
            gen.think = Some(Box::new(crate::serve::think::ThinkCtl::new(
                markers,
                crate::serve::think::ThinkLimits {
                    budget: budget.map(|b| b.min(i64::from(u32::MAX)) as u32),
                    reply_cap,
                    forced_end: forced_end.into(),
                    grace: defaults.map_or(0, |d| d.newline_grace),
                    stop_blank_line: stop_blank,
                    live: live.then(|| crate::serve::think::live_register(&ids.request)),
                    spec_forced: defaults.is_some_and(|d| d.spec_forced_end),
                    // Validated at startup: every steering text is one token.
                    bias: match defaults.filter(|_| steer) {
                        Some(d) => d
                            .bias(bundle.tokenizer().as_ref(), &gen.params.logit_bias)
                            .unwrap_or_default(),
                        None => Default::default(),
                    },
                },
                reasoning_open,
            )));
        }
    }
    let lp_fmt = gen
        .params
        .logprobs
        .map(|_| crate::serve::logprobs::TokenText {
            tok: bundle.tokenizer().clone(),
            as_ids: req.return_tokens_as_token_ids.unwrap_or(false),
        });
    let routed;
    let (mut rank, mut prefix, mut _pick) = (0, None, None);
    let mux = match (&direct, dp) {
        (Some(m), _) => m,
        (None, Some(set)) => {
            match state.dp_route(set, ids.session.as_deref(), Some(&prompt_ids), 0) {
                Some((r, m, key, pick)) => {
                    (rank, prefix, _pick) = (r, key, Some(pick));
                    routed = m;
                    &routed
                }
                None => return dp_unavailable(&req.model),
            }
        }
        (None, None) => unreachable!("checked at lookup"),
    };
    if prefix.is_none() {
        prefix = mux.prefix_key(&prompt_ids);
    }
    let key = dp.map_or(req.model.as_str(), |set| set.ranks[rank].key.as_str());
    let mm_job = match mm_pending {
        Some(p) => {
            let Some(mm) = state.mm_model(key) else {
                return mm_error(crate::serve::mm::MmError {
                    status: 503,
                    message: "model is not loaded; retry".into(),
                });
            };
            match crate::serve::mm::encode(&mm, p).await {
                Ok(job) => Some(job),
                Err(e) => return mm_error(e),
            }
        }
        None => None,
    };
    let (tx, rx) = stream_mod::channel();
    let Some(in_flight) = ids.begin(&req.model) else {
        return crate::serve::api_error(
            axum::http::StatusCode::CONFLICT,
            format!(
                "request {} is already in flight in this session",
                ids.request
            ),
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
    let session = ids.session.as_ref().and_then(|_| {
        ids.ticket(
            crate::serve::session::row_keys(&prompt_ids, &[], &[]),
            report,
        )
    });
    let regen = defaults
        .filter(|d| d.reply_regenerate && d.reply_guard.is_some() && mm_contract.is_none())
        .and_then(|_| {
            let close = reasoning_mode.close_marker()?;
            let mut gen = crate::serve::GenParams {
                max_tokens: reply_cap.map_or(gen.max_tokens, |c| c as usize),
                params: gen.params.clone(),
                stop: gen.stop.clone(),
                seed: gen.seed,
                stop_token_ids: gen.stop_token_ids.clone(),
                keep_special_tokens: true,
                ..Default::default()
            };
            gen.params.logprobs = None;
            Some(Regen {
                mux: mux.clone(),
                bundle: Arc::clone(&bundle),
                prompt: prompt.clone(),
                close,
                gen,
            })
        });
    let job = crate::serve::mux::Job {
        prompt_ids,
        gen,
        arrived: std::time::Instant::now(),
        respond: tx,
        opts: crate::serve::mux::JobOpts {
            session,
            turn: run.key(),
            continuing: run.continuing(),
            tenant: ids.tenant,
            session_end: ids.session_end,
            prefix: prefix.take(),
            mm: mm_job,
            ..Default::default()
        },
    };
    if crate::obs::host::on() {
        crate::obs::host::submitted(n_prompt, t_arrive.elapsed());
    }
    if let Err(err) = state.submit_routed(
        dp.map(|s| (&**s, rank)),
        ids.session.as_deref(),
        mux,
        job,
        t_arrive,
        ingress,
    ) {
        return match err {
            crate::serve::mux::SubmitError::Full(_) => {
                let mut r = crate::serve::api_error(
                    axum::http::StatusCode::TOO_MANY_REQUESTS,
                    "model request queue full",
                    "rate_limit_error",
                    Some("server_overloaded"),
                    None,
                );
                if crate::config::RuntimeConfig::get().shed_queue_depth > 0 {
                    r.headers_mut().insert(
                        axum::http::header::RETRY_AFTER,
                        axum::http::HeaderValue::from_static("1"),
                    );
                }
                r
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

    let guard = defaults.filter(|d| d.reply_guard.is_some()).map(|d| {
        let caller = req
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.text())
            .unwrap_or_default();
        crate::serve::reply_guard::ReplyGuard::new(
            d.reply_guard_lines.as_ref(),
            crate::serve::reply_guard::caller_script(&caller),
        )
    });
    let tally =
        crate::obs::serving::ReasoningTally::admitted(state.model_metrics(key), thinking, t_arrive);
    let request_id = request_id();
    // Stamped ONCE and repeated on every chunk of a stream, as OpenAI does.
    let created = now_secs();
    let cache = crate::serve::session::CacheOutcome::received(report_rx).await;
    run.admitted(cache.and_then(|c| c.at));
    let mut response = if req.stream {
        let include_usage = req.stream_options.map(|o| o.include_usage).unwrap_or(false);
        let stamped = run.headers();
        let sse = sse_response(
            request_id,
            requested_model,
            rx,
            include_usage,
            t_arrive,
            n_prompt,
            created,
            split_mode,
            reasoning_open,
            reenter,
            include_reasoning,
            lp_fmt,
            tool_stream,
            guard,
            regen,
            run,
            tally,
        );
        let mut response = crate::serve::session::hold_until_sent(sse.into_response(), in_flight);
        response.headers_mut().extend(stamped);
        response
    } else {
        buffer_and_reply(
            request_id,
            requested_model,
            rx,
            created,
            split_mode,
            reasoning_open,
            reenter,
            include_reasoning,
            lp_fmt,
            tool_stream,
            guard,
            regen,
            run,
            tally,
        )
        .await
    };
    if let Some(cache) = cache {
        cache.stamp(&mut response);
    }
    response
}

/// `POST /v1/requests/{id}/reasoning_end`: end the trace of the in-flight request `id` (its
/// `X-Request-Id`) now, as when a voice caller barges in. Served for models whose
/// `--chat-defaults` set `live_reasoning_end`; the forced `reasoning_end` follows within
/// [`crate::serve::think::LIVE_CHAIN`] tokens.
pub async fn reasoning_end(axum::extract::Path(id): axum::extract::Path<String>) -> Response {
    if crate::serve::think::live_end(&id) {
        Json(serde_json::json!({"id": id, "reasoning_end": true})).into_response()
    } else {
        crate::serve::api_error(
            axum::http::StatusCode::NOT_FOUND,
            format!("no request {id} with live reasoning control is in flight"),
            "invalid_request_error",
            Some("request_not_found"),
            None,
        )
    }
}

/// The response-side state machine when the plain reasoning split is not enough: the request
/// declares tools, the format frames every answer (harmony), or the reasoning markers are special
/// tokens (Gemma 4's thought channel) and must be decoded to be seen.
pub(crate) fn response_stream(
    plan: &crate::serve::tools::request::Plan,
    support: Option<crate::serve::tools::ToolSupport>,
    mode: crate::serve::reasoning::ReasoningMode,
    reasoning_open: bool,
    reenter: bool,
    bundle: &crate::asset::ModelBundle,
    stream: bool,
    opener: Option<&str>,
) -> Option<crate::serve::tools::stream::ToolStream> {
    use crate::serve::tools::{request::ParseSpec, ToolFormat, ToolSupport};
    let spec = plan.parse.clone().or(match support {
        Some(ToolSupport::Format(f @ ToolFormat::Harmony)) => Some(ParseSpec::plain(f)),
        _ => None,
    });
    if spec.is_none() && !bundle.reasoning_in_specials() {
        return None;
    }
    let mut ts = crate::serve::tools::stream::ToolStream::new(
        spec.as_ref(),
        mode,
        reasoning_open,
        bundle.special_text().clone(),
        stream,
    );
    if reenter {
        ts.split.allow_reentry();
    }
    if let Some(o) = opener {
        ts.prime(o);
    }
    Some(ts)
}

/// The model's tool call failed validation (`strict`, or a forced `tool_choice`).
fn tool_call_error(message: &str) -> crate::serve::openai::ApiErrorBody {
    crate::serve::openai::ApiErrorBody::new(
        message,
        "server_error",
        Some("invalid_tool_call"),
        None,
    )
}

fn mm_error(e: crate::serve::mm::MmError) -> Response {
    let mut r = crate::serve::api_error(
        axum::http::StatusCode::from_u16(e.status).unwrap_or(axum::http::StatusCode::BAD_REQUEST),
        e.message,
        if e.status == 400 {
            "invalid_request_error"
        } else {
            "server_error"
        },
        (e.status == 400).then_some("invalid_value"),
        Some("messages[].content".into()),
    );
    // Transient refusals (full slab, encoder restarting): rows free as in-flight requests finish.
    if e.status == 503 {
        r.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            axum::http::HeaderValue::from_static("1"),
        );
    }
    r
}

/// No rank of a data-parallel model is serving (all unloading, or each refused the request).
pub(crate) fn dp_unavailable(model: &str) -> Response {
    crate::serve::api_error(
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        format!("no data-parallel rank of '{model}' is serving; retry"),
        "server_error",
        None,
        None,
    )
}

/// Pick the chat template a GPU-served model wants.
///
/// Selected by PROBING the bundle's own tokenizer for the family's turn
/// marker: a marker is ONE special id in its own vocab (it is in that
/// checkpoint's `added_tokens`) and several ordinary pieces in anyone else's.
/// So the assets decide, which is what kept this honest when GLM-5.2 became the
/// second GPU-served family — before this, every GPU model got Gemma's markers,
/// which another checkpoint's tokenizer spells out as literal text.
///
/// Unknown family → Gemma's format, the previous behavior.
/// Why a chat conversation has no prompt.
#[derive(Debug)]
pub(crate) enum PromptError {
    /// The template REFUSED this conversation (HF templates call `raise_exception` for shapes
    /// they cannot render, e.g. an out-of-order role sequence); a 400, never another prompt.
    Template(String),
    Request(crate::serve::tools::request::Refusal),
    /// The packet names a built-in builder this server does not have.
    NoTemplate(String),
}

/// The prompt a templated (GPU-served) model gets for `messages`: THE CHECKPOINT'S OWN TEMPLATE
/// FIRST. The built-in per-family builders are the fallback for checkpoints that ship none — they
/// are an approximation of a file the weights already carry, and every divergence between the two
/// is a wrong prompt. The one place this choice is made; the serve golden test replays it.
pub(crate) fn chat_prompt(
    bundle: Option<&crate::asset::ModelBundle>,
    messages: &[Message],
    render_opts: &crate::serve::template::RenderOpts,
) -> Result<String, PromptError> {
    if let Some(t) = bundle.and_then(|b| b.chat_template()) {
        use crate::serve::tools::request::{render, RenderError};
        return render(t, messages, render_opts).map_err(|e| match e {
            RenderError::Template(e) => PromptError::Template(e),
            RenderError::Request(r) => PromptError::Request(r),
        });
    }
    Ok(match bundle.map(|b| b.serve()) {
        // A packet that names a built-in builder gets exactly it. A version-1 packet that
        // carries neither a usable template nor a builder (checkpoints without a template, e.g.
        // Kimi-K3) keeps the vocabulary-probed builders it was served with before serve.json; a
        // version-2 packet declared its builder at emit, so nothing is probed for it.
        Some(serve) if serve.from_packet => {
            match serve
                .manifest
                .chat
                .as_ref()
                .and_then(|c| c.builtin.as_deref())
            {
                Some(id) => builtin_chat_prompt(id, messages).ok_or_else(|| {
                    PromptError::NoTemplate(format!("unknown built-in chat format {id:?}"))
                })?,
                None if serve.manifest.declares_response() => {
                    return Err(PromptError::NoTemplate(
                        "the packet declares no chat template and no built-in chat format".into(),
                    ))
                }
                None => gpu_chat_prompt(bundle, messages),
            }
        }
        _ => gpu_chat_prompt(bundle, messages),
    })
}

/// The built-in prompt builder a packet names (`serve.json` `chat.builtin`), for checkpoints that
/// ship no template.
fn builtin_chat_prompt(id: &str, messages: &[Message]) -> Option<String> {
    Some(match id {
        "k3" => k3_chat_prompt(messages),
        "harmony" => harmony_chat_prompt(messages),
        "glm" => glm_chat_prompt(messages),
        "llama3" => llama3_chat_prompt(messages),
        "gemma" => gemma_chat_prompt(messages),
        _ => return None,
    })
}

fn gpu_chat_prompt(bundle: Option<&crate::asset::ModelBundle>, messages: &[Message]) -> String {
    let one = |m: &str| {
        bundle
            .map(|b| {
                let t = b.tokenizer();
                !t.is_byte_fallback() && t.encode(m).len() == 1
            })
            .unwrap_or(false)
    };
    // Probed BEFORE GLM's marker, not after: the two are disjoint in practice, but K3's
    // `<|end_of_msg|>` is also its `eos_token_id` (163586, generation_config.json), which makes it
    // the least ambiguous thing in the vocab to key on.
    if one("<|end_of_msg|>") {
        k3_chat_prompt(messages)
    } else if one("<|start|>") && one("<|channel|>") {
        harmony_chat_prompt(messages)
    } else if one("<|assistant|>") {
        glm_chat_prompt(messages)
    } else if one("<|start_header_id|>") && one("<|eot_id|>") {
        llama3_chat_prompt(messages)
    } else {
        // GEMMA IS NOW PROBED FOR, NOT ASSUMED. It used to be the silent
        // fallback for anything unrecognised, so a new family whose markers
        // matched no probe was served GEMMA'S format — which its own tokenizer
        // spells out as literal text, producing a fluent answer to a prompt the
        // model never really saw. There is no Jinja engine here to fall back
        // on, so the least this can do is SAY so.
        if !one("<|turn>") {
            tracing::error!(
                "no chat template matches this tokenizer (probed K3, harmony, GLM, Gemma); \
                 falling back to Gemma's format, which this model almost certainly does not \
                 use — the prompt markers will be tokenized as ordinary text. Add a template \
                 arm for this family before trusting any output."
            );
        }
        gemma_chat_prompt(messages)
    }
}

/// Llama-3.x chat format (the checkpoint's `chat_template.jinja`, text-only, no tools):
/// `<|begin_of_text|>`, one `<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>`
/// block per message, then the generation prompt for the assistant. The markers are single
/// ids via `added_tokens` (128000, 128006, 128007, 128009); `<|eot_id|>` is in the stop set
/// through `generation_config.json`.
fn llama3_chat_prompt(messages: &[Message]) -> String {
    let mut p = String::from("<|begin_of_text|>");
    for m in messages {
        let role = match m.role.as_str() {
            "assistant" => "assistant",
            "system" | "developer" => "system",
            _ => "user",
        };
        p.push_str("<|start_header_id|>");
        p.push_str(role);
        p.push_str("<|end_header_id|>\n\n");
        p.push_str(m.text().trim());
        p.push_str("<|eot_id|>");
    }
    p.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
    p
}

/// OpenAI harmony format (gpt-oss), text-only, no tools, reasoning DISABLED: one
/// `<|start|>{role}<|message|>{text}<|end|>` block per message, then the generation prompt
/// `<|start|>assistant<|channel|>final<|message|>` — opening the FINAL channel directly, so
/// the served tokens are the answer and not an analysis trace (the same choice the K3/GLM
/// arms make). `<|start|>` 200006, `<|message|>` 200008, `<|end|>` 200007, `<|channel|>`
/// 200005 are single ids via `added_tokens`; the stop set (`<|return|>` 200002, `<|call|>`
/// 200012) comes from `generation_config.json` through `read_eos_ids`.
fn harmony_chat_prompt(messages: &[Message]) -> String {
    let mut p = String::new();
    for m in messages {
        let role = match m.role.as_str() {
            "assistant" => "assistant",
            "system" | "developer" => "developer",
            _ => "user",
        };
        p.push_str("<|start|>");
        p.push_str(role);
        p.push_str("<|message|>");
        p.push_str(m.text().trim());
        p.push_str("<|end|>");
    }
    p.push_str("<|start|>assistant<|channel|>final<|message|>");
    p
}

/// Kimi-K3's chat format — text-only, no tools, thinking DISABLED.
///
/// K3 ships NO `chat_template.jinja` and no `chat_template` in
/// `tokenizer_config.json`; the format lives in Python, in
/// `encoding_k3.py::build_chat_segments`, as an "XTML" segment structure. This
/// is that function's own output for the text-only case, captured by running
/// it against the real checkpoint rather than read off the source:
///
/// ```text
/// <|open|>message role="user"<|sep|>Hello!<|close|>message<|sep|><|end_of_msg|>
/// <|open|>message role="assistant"<|sep|><|open|>response<|sep|>
/// ```
///
/// THINKING DISABLED is the same deliberate choice the GLM arm makes, and it
/// changes the generation prompt rather than merely dropping a line: with
/// thinking on, K3 opens `<|open|>think<|sep|>` and additionally prepends a
/// `role="system" type="thinking-effort"` message explaining the effort knob.
/// With it off the assistant turn opens the RESPONSE channel directly. A served
/// benchmark measures answer tokens, not a reasoning trace whose length nobody
/// controls — and the `think` channel would otherwise have to be parsed back
/// out of the stream.
///
/// The four markers are single ids via `added_tokens` — `<|open|>` 163587,
/// `<|close|>` 163588, `<|sep|>` 163589, `<|end_of_msg|>` 163586 — which is
/// also what the probe above keys on. Note `<|open|>`/`<|close|>`/`<|sep|>`
/// carry `"special": false` in the checkpoint's `added_tokens_decoder`; that
/// affects `skip_special_tokens` on DECODE, not whether they encode as one id.
fn k3_chat_prompt(messages: &[Message]) -> String {
    let mut p = String::new();
    for m in messages {
        let role = match m.role.as_str() {
            "assistant" => "assistant",
            "system" | "developer" => "system",
            _ => "user",
        };
        p.push_str("<|open|>message role=\"");
        p.push_str(role);
        p.push_str("\"<|sep|>");
        p.push_str(m.text().trim());
        p.push_str("<|close|>message<|sep|><|end_of_msg|>");
    }
    p.push_str("<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>");
    p
}

/// GLM's chat format, rendered to match the checkpoint's own
/// `chat_template.jinja` rather than approximating it.
///
/// WHAT THIS USED TO GET WRONG, all four verified against the rendered
/// template for `zai-org/GLM-5.3`:
///
/// 1. It emitted the generation prompt as `<|assistant|><think></think>`,
///    closing the thinking block. That was GLM-5.2's trick. **5.3's template
///    has no disable branch at all** — it always ends `<|assistant|><think>` —
///    so the closed form put every request off-distribution.
/// 2. It dropped the `<|system|>Reasoning Effort: Max` line. The template
///    emits that on EVERY render (the default is `max`, not none), so the model
///    never saw a system line it was always trained with.
/// 3. Assistant turns in the HISTORY were written as `<|assistant|>{content}`.
///    The template writes `<|assistant|><think></think>{content}`, so every
///    multi-turn conversation was malformed.
/// 4. `role: "tool"` fell through to the `_ =>` arm and was rendered as a USER
///    turn. The template has its own `<|observation|><tool_response>` block.
///
/// Thinking is left OPEN, as the template does. [`split_reasoning`] separates
/// the trace from the answer afterwards so the OpenAI `content` still carries
/// the answer alone.
fn glm_chat_prompt(messages: &[Message]) -> String {
    let mut p = String::from("[gMASK]<sop>");
    p.push_str("<|system|>Reasoning Effort: Max");
    for m in messages {
        let text = m.text();
        match m.role.as_str() {
            // The template wraps a tool result in its own observation block;
            // rendering it as a user turn made the model read a tool payload as
            // something the human said.
            "tool" | "observation" => {
                p.push_str("<|observation|><tool_response>");
                p.push_str(text.trim());
                p.push_str("</tool_response>");
            }
            "assistant" => {
                p.push_str("<|assistant|><think></think>");
                p.push_str(text.trim());
            }
            "system" | "developer" => {
                p.push_str("<|system|>");
                p.push_str(text.trim());
            }
            _ => {
                p.push_str("<|user|>");
                p.push_str(text.trim());
            }
        }
    }
    // OPEN, exactly as the template ends. Not `<think></think>`.
    p.push_str("<|assistant|><think>");
    p
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
        p.push_str(m.text().trim());
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
    created: u64,
    reasoning_mode: crate::serve::reasoning::ReasoningMode,
    reasoning_open: bool,
    reenter: bool,
    include_reasoning: bool,
    lp_fmt: Option<crate::serve::logprobs::TokenText>,
    mut tools: Option<crate::serve::tools::stream::ToolStream>,
    mut guard: Option<crate::serve::reply_guard::ReplyGuard>,
    regen: Option<Regen>,
    mut run: crate::serve::turns::StageRun,
    mut tally: crate::obs::serving::ReasoningTally,
) -> Response {
    let mut calls: Vec<crate::serve::openai::ToolCall> = Vec::new();
    let mut text = String::new();
    let mut lp_content = Vec::new();
    // Driven PER TOKEN, exactly as the streamed path drives it — same type,
    // same order — so the two cannot disagree, and `trace_tokens` is a real
    // token count rather than a count of calls.
    let mut split = crate::serve::reasoning::ReasoningSplit::new(reasoning_mode, reasoning_open);
    if reenter {
        split.allow_reentry();
    }
    let (mut reasoning_buf, mut answer_buf) = (String::new(), String::new());
    // Tokens that fell inside the trace, for `completion_tokens_details`.
    // Counted from CHUNKS (one per generated token) rather than derived from
    // the split text — the split is over characters and cannot be converted
    // back into a token count.
    // `None` until a terminal chunk arrives. It must NOT default to "stop": the
    // mux frees a slot on ANY `try_send` failure, and a bounded-channel `Full`
    // (serve/stream.rs caps the stream at 32 chunks) drops the sender without
    // sending `Done` or `Err`. Defaulting to "stop" turned that into a 200 with
    // a fluent, truncated answer that claimed to be complete — invisible to a
    // benchmark client, which counts it as a successful request.
    let mut finish: Option<stream_mod::FinishReason> = None;
    let mut usage: Option<Usage> = None;
    let mut think = None;
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token {
                id,
                text: delta,
                logprobs,
                ..
            } => {
                run.first();
                if let (Some(fmt), Some(lp)) = (&lp_fmt, &logprobs) {
                    lp_content.push(crate::serve::logprobs::chat_entry(fmt, id, lp));
                }
                text.push_str(&delta);
                match tools.as_mut() {
                    Some(ts) => {
                        let st = ts.push(&delta);
                        reasoning_buf.push_str(st.reasoning.unwrap_or_default());
                        if st.content.is_some_and(|c| !c.trim().is_empty()) {
                            tally.reply_started();
                        }
                        answer_buf.push_str(st.content.unwrap_or_default());
                        if let Some(e) = ts.error() {
                            return (
                                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                                Json(tool_call_error(e)),
                            )
                                .into_response();
                        }
                    }
                    None => {
                        let (r, c) = split.push(&delta);
                        reasoning_buf.push_str(&r.unwrap_or_default());
                        if c.as_deref().is_some_and(|c| !c.trim().is_empty()) {
                            tally.reply_started();
                        }
                        answer_buf.push_str(&c.unwrap_or_default());
                    }
                }
            }
            StreamChunk::Done {
                reason, usage: u, ..
            } => {
                finish = Some(reason);
                think = u.think;
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
                return crate::serve::api_error_for(&e);
            }
        }
    }
    let Some(finish) = finish else {
        // The channel closed with no terminal chunk. The generation was cut
        // (mux slot freed on a send failure, or a dispatcher tick panic) and
        // there is no honest completion to return — say so rather than ship the
        // partial as a finished answer.
        tracing::warn!(
            %model,
            partial_chars = text.len(),
            "chat: stream ended with no terminal chunk — reporting as truncated"
        );
        return crate::serve::api_error(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "generation stream ended without a finish reason (slot cut)",
            "server_error",
            None,
            None,
        );
    };
    // The generation prompt leaves `<think>` open on GLM, so the raw text is
    // "<trace></think><answer>". Split it: `content` carries the answer alone
    // and the trace goes to `reasoning` (vLLM's name), which is where a reasoning
    // model's clients look for it. A model that never opened a trace is
    // unaffected — `split_reasoning` returns the whole string as the answer.
    match tools.as_mut() {
        Some(ts) => {
            let st = ts.finish(!matches!(finish, stream_mod::FinishReason::Stop));
            reasoning_buf.push_str(st.reasoning.unwrap_or_default());
            answer_buf.push_str(st.content.unwrap_or_default());
            if let Some(e) = ts.error() {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    Json(tool_call_error(e)),
                )
                    .into_response();
            }
            calls = ts.take_calls();
        }
        None => {
            let (r, c) = split.finish();
            reasoning_buf.push_str(&r.unwrap_or_default());
            answer_buf.push_str(&c.unwrap_or_default());
        }
    }
    // The guard reads the answer by its lines alone, so one pass over the whole of it is what
    // the streamed path produces piece by piece.
    if let Some(g) = guard.as_mut() {
        let answer = std::mem::take(&mut answer_buf);
        if !reasoning_buf.trim().is_empty() {
            g.note_reasoning();
        }
        g.push(&answer, &mut reasoning_buf, &mut answer_buf);
        if g.silent(!calls.is_empty()) {
            if let Some((more, n)) = match &regen {
                Some(r) => r.reply(&text).await,
                None => None,
            } {
                g.push(&more, &mut reasoning_buf, &mut answer_buf);
                if let Some(u) = usage.as_mut() {
                    u.completion_tokens += n as u64;
                    u.total_tokens += n as u64;
                }
            }
        }
        let fallback = g.finish(!calls.is_empty(), &mut reasoning_buf, &mut answer_buf);
        tally.guarded(g, fallback);
    }
    let think = guarded_report(think, guard.as_ref());
    let reasoning = {
        let t = reasoning_buf.trim();
        (include_reasoning && !t.is_empty()).then(|| t.to_string())
    };
    let trace_tokens = tools
        .as_ref()
        .map_or(split.trace_tokens, |ts| ts.split.trace_tokens);
    // A turn that is only calls, or only a trace, answers `content: null`, as vLLM does.
    let answer = ((calls.is_empty() && trace_tokens == 0) || !answer_buf.trim().is_empty())
        .then_some(answer_buf);
    tally.finished(trace_tokens, think);
    if trace_tokens > 0 {
        if let Some(u) = usage.as_mut() {
            u.completion_tokens_details = Some(CompletionTokensDetails {
                reasoning_tokens: trace_tokens,
            });
        }
    }
    // `tool_calls` only for a turn that ended on its own: a cut-off turn stays "length".
    let finish_reason = match finish {
        stream_mod::FinishReason::Stop if !calls.is_empty() => "tool_calls",
        f => f.as_openai(),
    };
    run.done();
    let mut response = Json(ChatResponse {
        id: request_id.clone(),
        object: "chat.completion",
        created,
        model,
        system_fingerprint: crate::serve::openai::SYSTEM_FINGERPRINT,
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant",
                content: answer,
                refusal: None,
                reasoning_content: crate::serve::openai::Reasoning(reasoning),
                tool_calls: (!calls.is_empty()).then_some(calls),
            },
            logprobs: lp_fmt.map(|_| crate::serve::logprobs::ChatLogprobs {
                content: lp_content,
            }),
            // The WIRE value, which is not always the internal one: a
            // preemption is reported as "length" because "preempted" is not an
            // OpenAI finish_reason and a typed client rejects the response on
            // sight. Collapsing it to "stop" would be worse still — a
            // truncated answer claiming to be complete — so the real cause
            // rides along in `x_plow_finish_reason`.
            finish_reason: Some(finish_reason),
            x_plow_finish_reason: finish.is_vendor_specific().then(|| finish.as_str()),
            x_plow_thinking: think,
        }],
        usage,
    })
    .into_response();
    run.stamp(&mut response);
    response
}

/// 7.9: one more generation for a guarded turn left with nothing to say: the prompt, the turn's
/// trace up to its first close (the close appended when it never closed), and a reply capped at
/// `max_reply_tokens`.
struct Regen {
    mux: crate::serve::mux::ModelMux,
    bundle: Arc<crate::asset::ModelBundle>,
    prompt: String,
    close: &'static str,
    gen: crate::serve::GenParams,
}

impl Regen {
    /// The new reply's answer text and its token count; `None` when the dispatcher refused it.
    async fn reply(&self, generated: &str) -> Option<(String, usize)> {
        let mut prompt = self.prompt.clone();
        match generated.find(self.close) {
            Some(i) => prompt.push_str(&generated[..i + self.close.len()]),
            None => {
                prompt.push_str(generated);
                prompt.push_str(self.close);
            }
        }
        let (tx, mut rx) = stream_mod::channel();
        let job = crate::serve::mux::Job {
            prompt_ids: self.bundle.tokenizer().encode(&prompt),
            gen: self.gen.clone(),
            arrived: std::time::Instant::now(),
            respond: tx,
            opts: Default::default(),
        };
        self.mux.submit_wait(job).await.ok()?;
        let mut text = String::new();
        let mut n = 0;
        while let Some(chunk) = rx.recv().await {
            match chunk {
                StreamChunk::Token { text: t, .. } => {
                    text.push_str(&t);
                    n += 1;
                }
                StreamChunk::Done { .. } => break,
                StreamChunk::Err(_) => return None,
            }
        }
        // The reply may open another trace: split it off, then drop the remaining framing.
        let mut split =
            crate::serve::reasoning::ReasoningSplit::new(self.bundle.reasoning(), false);
        if self.bundle.reasoning_in_specials() {
            split.allow_reentry();
        }
        let (_, mut answer) = (String::new(), String::new());
        let mut trace = String::new();
        split.push_into(&text, &mut trace, &mut answer);
        split.finish_into(&mut trace, &mut answer);
        Some((self.bundle.special_text().strip(&answer), n))
    }
}

/// One response step through the reply guard; `r` / `c` hold the step's reasoning and content.
fn guard_step<'a>(
    g: &mut crate::serve::reply_guard::ReplyGuard,
    s: crate::serve::tools::stream::Step<'a>,
    r: &'a mut String,
    c: &'a mut String,
) -> crate::serve::tools::stream::Step<'a> {
    r.clear();
    c.clear();
    if let Some(x) = s.reasoning {
        r.push_str(x);
        g.note_reasoning();
    }
    if let Some(x) = s.content {
        g.push(x, r, c);
    }
    let (r, c): (&'a String, &'a String) = (r, c);
    crate::serve::tools::stream::Step {
        reasoning: (!r.is_empty()).then_some(r.as_str()),
        content: (!c.is_empty()).then_some(c.as_str()),
        tool_calls: s.tool_calls,
    }
}

/// [`guard_step`] over the plain split's owned pieces.
fn guard_owned(
    g: &mut crate::serve::reply_guard::ReplyGuard,
    reasoning: Option<String>,
    content: Option<String>,
) -> (Option<String>, Option<String>) {
    let mut r = reasoning.unwrap_or_default();
    if !r.is_empty() {
        g.note_reasoning();
    }
    let mut c = String::new();
    if let Some(x) = content {
        g.push(&x, &mut r, &mut c);
    }
    ((!r.is_empty()).then_some(r), (!c.is_empty()).then_some(c))
}

/// `x_plow_thinking` with the guard's outcome folded in.
fn guarded_report(
    think: Option<crate::serve::think::ThinkReport>,
    guard: Option<&crate::serve::reply_guard::ReplyGuard>,
) -> Option<crate::serve::think::ThinkReport> {
    match guard {
        Some(g) => Some(crate::serve::think::ThinkReport {
            guard_moved: g.moved,
            ..think.unwrap_or_default()
        }),
        None => think,
    }
}

/// An error object in its own SSE frame; the stream then ends without `[DONE]`.
fn sse_error(body: &crate::serve::openai::ApiErrorBody) -> Bytes {
    let data = serde_json::to_string(body).unwrap_or_else(|_| {
        "{\"error\":{\"message\":\"stream error\",\"type\":\"server_error\"}}".to_string()
    });
    stream_mod::sse_data(&data)
}

/// Streaming path: one SSE `chat.completion.chunk` frame per produced token,
/// terminated by a final chunk carrying `finish_reason` and then `[DONE]`.
///
/// The `role: "assistant"` delta rides the FIRST token's chunk; it is NOT a
/// leading frame of its own. That is a measurement fix, not a cosmetic one:
/// `vllm bench serve`'s chat backend stamps TTFT on the first chunk carrying a
/// `choices` array **whatever its content**
/// (`vllm/benchmarks/backend_request_func.py`), so a role frame emitted at
/// request-arrival time made plowrt report TTFT = one HTTP round trip. Measured
/// on gfx950 with a 7013-token prompt: role frame at 7.1 ms, first real token at
/// 1322 ms. It also poisoned TPOT and mean ITL, since the whole prefill then
/// landed in the first inter-token gap. vLLM's own server sends nothing before
/// its first token (measured TTFT 208 ms on a 1024-token prefill), so this is
/// also what makes the two servers comparable under one client.
///
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
    t_arrive: std::time::Instant,
    n_prompt: usize,
    created: u64,
    reasoning_mode: crate::serve::reasoning::ReasoningMode,
    reasoning_open: bool,
    reenter: bool,
    include_reasoning: bool,
    lp_fmt: Option<crate::serve::logprobs::TokenText>,
    tools: Option<crate::serve::tools::stream::ToolStream>,
    guard: Option<crate::serve::reply_guard::ReplyGuard>,
    regen: Option<Regen>,
    run: crate::serve::turns::StageRun,
    tally: crate::obs::serving::ReasoningTally,
) -> Response {
    // State threaded through the unfold: the receiver, and the tail frames
    // (optional usage-only chunk, then [DONE]) drained one per poll.
    struct SseState {
        rx: stream_mod::ChunkReceiver,
        done: bool,
        /// Frames the finish produced beyond its own, sent before the next receive.
        queued: std::collections::VecDeque<Bytes>,
        /// Set when the response needs more than the reasoning split (tools, harmony, special
        /// reasoning markers); replaces `split`.
        tools: Option<crate::serve::tools::stream::ToolStream>,
        include_reasoning: bool,
        /// The `role` delta has not been sent yet — it rides the FIRST token.
        role_pending: bool,
        /// Routes each delta to `reasoning_content` or `content`. The SAME
        /// type the buffered path uses, so the two cannot disagree about one
        /// generation.
        split: crate::serve::reasoning::ReasoningSplit,
        pending: std::collections::VecDeque<Bytes>,
        run: crate::serve::turns::StageRun,
        head: stream_mod::FrameHead,
        request_id: String,
        model: String,
        lp_fmt: Option<crate::serve::logprobs::TokenText>,
        tally: crate::obs::serving::ReasoningTally,
        guard: Option<crate::serve::reply_guard::ReplyGuard>,
        /// The guard's reasoning and content for the current step.
        gr: String,
        gc: String,
        regen: Option<Regen>,
        /// The generation's text, kept only for `regen`.
        raw: String,
    }
    let body = stream::unfold(
        SseState {
            rx,
            done: false,
            queued: std::collections::VecDeque::new(),
            tools,
            include_reasoning,
            role_pending: true,
            split: {
                let mut split =
                    crate::serve::reasoning::ReasoningSplit::new(reasoning_mode, reasoning_open);
                if reenter {
                    split.allow_reentry();
                }
                split
            },
            pending: std::collections::VecDeque::new(),
            run,
            head: stream_mod::FrameHead::new(&request_id, "chat.completion.chunk", created, &model),
            request_id,
            model,
            lp_fmt,
            tally,
            guard,
            gr: String::new(),
            gc: String::new(),
            regen,
            raw: String::new(),
        },
        move |mut st| {
            async move {
                if st.done {
                    return None;
                }
                if let Some(ev) = st.queued.pop_front() {
                    return Some((ev, st));
                }
                if let Some(ev) = st.pending.pop_front() {
                    st.done = st.pending.is_empty();
                    return Some((ev, st));
                }
                let chunk = match st.rx.recv().await {
                    Some(c) => c,
                    None => {
                        // No terminal chunk: the mux freed the slot on a
                        // `try_send` failure (the bounded 32-chunk stream
                        // filling counts, not just the client dropping) or a
                        // tick panicked. The wire shape is left alone — the
                        // stream ends without `[DONE]`, which is the only
                        // signal SSE has — but it must not be SILENT
                        // server-side, because a bench client scores this as a
                        // successful request with fewer output tokens.
                        tracing::warn!(
                            model = %st.model,
                            "chat: SSE stream ended with no terminal chunk (slot cut) \
                             — no finish_reason, no [DONE]"
                        );
                        return None;
                    }
                };
                let (frame, terminate) = match chunk {
                    StreamChunk::Token {
                        id, text, logprobs, ..
                    } => {
                        let logprobs =
                            st.lp_fmt
                                .as_ref()
                                .map(|fmt| crate::serve::logprobs::ChatLogprobs {
                                    content: logprobs
                                        .iter()
                                        .map(|lp| crate::serve::logprobs::chat_entry(fmt, id, lp))
                                        .collect(),
                                });
                        let role = st.role_pending.then(|| {
                            st.role_pending = false;
                            "assistant"
                        });
                        // §TTFT: this frame is the one `vllm bench serve` stamps.
                        if role.is_some() {
                            st.run.first();
                            crate::obs::ttft::dump(t_arrive.elapsed().as_nanos() as u64, n_prompt);
                            crate::obs::pfx::report();
                            if crate::obs::host::on() {
                                crate::obs::host::first_frame(n_prompt, t_arrive.elapsed());
                            }
                        }
                        if st.regen.is_some() {
                            st.raw.push_str(&text);
                        }
                        let frame = match st.tools.as_mut() {
                            Some(ts) => {
                                let step = ts.push(&text);
                                let step = match st.guard.as_mut() {
                                    Some(g) => guard_step(g, step, &mut st.gr, &mut st.gc),
                                    None => step,
                                };
                                if step.content.is_some_and(|c| !c.trim().is_empty()) {
                                    st.tally.reply_started();
                                }
                                let f = step.frame(
                                    &st.head,
                                    role,
                                    logprobs.as_ref(),
                                    st.include_reasoning,
                                    None,
                                );
                                if let Some(e) = ts.error() {
                                    tracing::warn!(model = %st.model, error = %e, "chat: tool call failed validation");
                                    st.done = true;
                                    return Some((sse_error(&tool_call_error(e)), st));
                                }
                                f
                            }
                            None => {
                                let (reasoning, content) = st.split.push(&text);
                                let (reasoning, content) = match st.guard.as_mut() {
                                    Some(g) => guard_owned(g, reasoning, content),
                                    None => (reasoning, content),
                                };
                                if content.as_deref().is_some_and(|c| !c.trim().is_empty()) {
                                    st.tally.reply_started();
                                }
                                st.head.frame(&ChunkChoice {
                                    index: 0,
                                    delta: Delta {
                                        role,
                                        content,
                                        reasoning_content: crate::serve::openai::Reasoning(
                                            reasoning.filter(|_| st.include_reasoning),
                                        ),
                                        tool_calls: None,
                                    },
                                    logprobs,
                                    finish_reason: None,
                                    x_plow_finish_reason: None,
                                    x_plow_thinking: None,
                                })
                            }
                        };
                        (frame, false)
                    }
                    StreamChunk::Done {
                        reason, mut usage, ..
                    } if st.tools.is_some() => {
                        let ts = st.tools.as_mut().expect("checked");
                        let tail = {
                            let s = ts.finish(!matches!(reason, stream_mod::FinishReason::Stop));
                            let s = match st.guard.as_mut() {
                                Some(g) => guard_step(g, s, &mut st.gr, &mut st.gc),
                                None => s,
                            };
                            (!s.is_empty(st.include_reasoning))
                                .then(|| s.frame(&st.head, None, None, st.include_reasoning, None))
                        };
                        if let Some(e) = ts.error() {
                            tracing::warn!(model = %st.model, error = %e, "chat: tool call failed validation");
                            st.done = true;
                            return Some((sse_error(&tool_call_error(e)), st));
                        }
                        let finish = match reason {
                            stream_mod::FinishReason::Stop if ts.calls() > 0 => "tool_calls",
                            r => r.as_openai(),
                        };
                        let trace_tokens = ts.split.trace_tokens;
                        st.tally.finished(trace_tokens, usage.think);
                        let mut frames: Vec<Bytes> = tail.into_iter().collect();
                        if let Some(g) = st.guard.as_mut() {
                            let (mut r, mut c) = (String::new(), String::new());
                            if g.silent(ts.calls() > 0) {
                                if let Some((more, n)) = match &st.regen {
                                    Some(re) => re.reply(&st.raw).await,
                                    None => None,
                                } {
                                    g.push(&more, &mut r, &mut c);
                                    usage.completion_tokens += n;
                                }
                            }
                            let fallback = g.finish(ts.calls() > 0, &mut r, &mut c);
                            st.tally.guarded(g, fallback);
                            if !c.is_empty() || (st.include_reasoning && !r.is_empty()) {
                                frames.push(st.head.frame(&ChunkChoice {
                                    index: 0,
                                    delta: Delta {
                                        role: None,
                                        content: (!c.is_empty()).then_some(c),
                                        reasoning_content: crate::serve::openai::Reasoning(
                                            (st.include_reasoning && !r.is_empty()).then_some(r),
                                        ),
                                        tool_calls: None,
                                    },
                                    logprobs: None,
                                    finish_reason: None,
                                    x_plow_finish_reason: None,
                                    x_plow_thinking: None,
                                }));
                            }
                        }
                        let think = guarded_report(usage.think, st.guard.as_ref());
                        let mut last = ChatChunk {
                            id: st.request_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model: st.model.clone(),
                            system_fingerprint: crate::serve::openai::SYSTEM_FINGERPRINT,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    reasoning_content: crate::serve::openai::Reasoning(None),
                                    tool_calls: None,
                                },
                                logprobs: None,
                                finish_reason: Some(finish),
                                x_plow_finish_reason: reason
                                    .is_vendor_specific()
                                    .then(|| reason.as_str()),
                                x_plow_thinking: think,
                            }],
                            usage: None,
                        };
                        frames.push(stream_mod::sse_data(&stream_mod::chunk_data(&last)));
                        if include_usage {
                            let mut u: Usage = usage.into();
                            if trace_tokens > 0 {
                                u.completion_tokens_details = Some(CompletionTokensDetails {
                                    reasoning_tokens: trace_tokens,
                                });
                            }
                            last.choices.clear();
                            last.usage = Some(u);
                            st.pending
                                .push_back(stream_mod::sse_data(&stream_mod::chunk_data(&last)));
                        }
                        let mut frames = frames.into_iter();
                        let first = frames.next().expect("the finish frame");
                        st.queued.extend(frames);
                        (first, true)
                    }
                    StreamChunk::Done {
                        reason, mut usage, ..
                    } => {
                        // FLUSH. Whatever the splitter still holds belongs to
                        // this generation; abandoning it truncated the streamed
                        // `reasoning_content` against the buffered path.
                        let (mut flushed_r, mut flushed_c) = st.split.finish();
                        st.tally.finished(st.split.trace_tokens, usage.think);
                        if let Some(g) = st.guard.as_mut() {
                            let (r, c) = guard_owned(g, flushed_r, flushed_c);
                            let (mut r, mut c) = (r.unwrap_or_default(), c.unwrap_or_default());
                            if g.silent(false) {
                                if let Some((more, n)) = match &st.regen {
                                    Some(re) => re.reply(&st.raw).await,
                                    None => None,
                                } {
                                    g.push(&more, &mut r, &mut c);
                                    usage.completion_tokens += n;
                                }
                            }
                            let fallback = g.finish(false, &mut r, &mut c);
                            st.tally.guarded(g, fallback);
                            flushed_r = (!r.is_empty()).then_some(r);
                            flushed_c = (!c.is_empty()).then_some(c);
                        }
                        let ch = ChunkChoice {
                            index: 0,
                            delta: Delta {
                                role: None,
                                content: flushed_c,
                                reasoning_content: crate::serve::openai::Reasoning(
                                    flushed_r.filter(|_| st.include_reasoning),
                                ),
                                tool_calls: None,
                            },
                            logprobs: None,
                            // The wire value: "preempted" is not an OpenAI
                            // finish_reason and a typed client rejects it.
                            finish_reason: Some(reason.as_openai()),
                            // ...so the true cause rides alongside it, as
                            // it already does on the buffered path.
                            x_plow_finish_reason: reason
                                .is_vendor_specific()
                                .then(|| reason.as_str()),
                            x_plow_thinking: guarded_report(usage.think, st.guard.as_ref()),
                        };
                        if include_usage {
                            // OpenAI stream-usage shape: a separate chunk with
                            // EMPTY choices carries usage, never the finish chunk.
                            let uch = ChatChunk {
                                id: st.request_id.clone(),
                                object: "chat.completion.chunk",
                                created,
                                model: st.model.clone(),
                                system_fingerprint: crate::serve::openai::SYSTEM_FINGERPRINT,
                                choices: Vec::new(),
                                usage: Some({
                                    let mut u: Usage = usage.into();
                                    if st.split.trace_tokens > 0 {
                                        u.completion_tokens_details =
                                            Some(CompletionTokensDetails {
                                                reasoning_tokens: st.split.trace_tokens,
                                            });
                                    }
                                    u
                                }),
                            };
                            st.pending
                                .push_back(stream_mod::sse_data(&stream_mod::chunk_data(&uch)));
                        }
                        (st.head.frame(&ch), true)
                    }
                    StreamChunk::Err(e) => {
                        // AN ERROR IS AN ERROR OBJECT, NOT ASSISTANT TEXT.
                        //
                        // This used to emit the error as `delta.content =
                        // "[error: ...]"` with `finish_reason: "stop"`, i.e. a
                        // 200 whose generated text contained the failure. Every
                        // standard client — openai-python, LangChain, `vllm
                        // bench serve` — scores that as a SUCCESSFUL request
                        // and hands the error string to the user as if the
                        // model had said it. The reachable causes are ordinary
                        // (no free slot, arrival-rate shed, KV OOM, context
                        // overflow, device fault), so this was not a corner.
                        //
                        // SSE cannot change the HTTP status once the body has
                        // started, so the honest wire form is an error OBJECT
                        // in its own frame, and NO `[DONE]` — a client that
                        // sees the stream end without `[DONE]` knows it was
                        // cut. This is the shape vLLM uses for the same case.
                        tracing::warn!(model = %st.model, error = %e, "chat: SSE stream error");
                        let body = crate::serve::openai::ApiErrorBody::new(
                            e.to_string(),
                            "server_error",
                            None,
                            None,
                        );
                        st.done = true;
                        return Some((sse_error(&body), st));
                    }
                };
                if terminate {
                    st.run.done();
                    st.pending.push_back(st.run.sse_comment_frame());
                    st.pending.push_back(stream_mod::sse_data(stream_mod::DONE));
                }
                Some((frame, st))
            }
        },
    );

    stream_mod::sse_response(body)
}

#[cfg(test)]
mod tests {
    use super::{gemma_chat_prompt, k3_chat_prompt, request_id, Message};

    #[test]
    fn request_ids_are_unique() {
        assert_ne!(request_id(), request_id());
    }

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.into(),
            content: Some(crate::serve::openai::Content::Text(text.into())),
            ..Default::default()
        }
    }

    /// The Gemma-4 built-in builder AGREES with the checkpoint's own template,
    /// byte for byte, on a text-only conversation.
    ///
    /// This is the reassuring half of [`gemma_chat_prompt`]'s "text-only
    /// subset" caveat, and it is worth pinning because the caveat reads as
    /// alarming without it: the fallback is not an approximation of the plain
    /// chat case, it is exact there. What the template carries and the builder
    /// does not is tool calling, multimodal parts and thinking-content
    /// ordering — none of which a text-only request reaches.
    ///
    /// So a checkpoint whose `chat_template.jinja` went unlinked still served
    /// ordinary chat correctly. Linking it matters for the other three.
    ///
    /// Skips where the checkpoint is absent, like the GLM test below.
    #[test]
    fn the_gemma_builder_matches_the_checkpoint_template_on_text() {
        let dir = std::path::Path::new("/app/plow/build-gemma31/checkpoint");
        if !dir.join("chat_template.jinja").exists() {
            eprintln!("skipped: no Gemma-4 checkpoint on this host");
            return;
        }
        let t = crate::serve::template::load(dir).expect("template compiles");
        for convo in [
            vec![("system", "You are helpful."), ("user", "Hi there")],
            vec![("user", "one"), ("assistant", "two"), ("user", "three")],
        ] {
            let built =
                gemma_chat_prompt(&convo.iter().map(|&(r, c)| msg(r, c)).collect::<Vec<_>>());
            let rendered = t
                .render(
                    &convo
                        .iter()
                        .map(|&(r, c)| serde_json::json!({"role": r, "content": c}))
                        .collect::<Vec<_>>(),
                )
                .expect("renders");
            assert_eq!(built, rendered, "convo {convo:?}");
        }
    }

    /// The exact Gemma-4 generation prompt, so a template or builder change has
    /// to restate it. Note `<|turn>`/`<turn|>` and the open `<|channel>thought`
    /// — NOT Gemma-3's `<start_of_turn>`/`<end_of_turn>`.
    #[test]
    fn the_gemma_generation_prompt_is_pinned() {
        assert_eq!(
            gemma_chat_prompt(&[msg("system", "You are helpful."), msg("user", "Hi there")]),
            "<bos><|turn>system\nYou are helpful.<turn|>\n\
             <|turn>user\nHi there<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>"
        );
    }

    /// PINNED AGAINST THE CHECKPOINT'S OWN TEMPLATE, rendered with jinja2 from
    /// `/workspace/models/GLM-5.3-FP8/chat_template.jinja` for
    /// `[system, user]` with `add_generation_prompt=True`. Every part of this
    /// string was wrong before: the reasoning-effort line was missing and the
    /// thinking block was closed.
    #[test]
    fn the_glm_prompt_is_what_the_checkpoint_template_renders() {
        assert_eq!(
            super::glm_chat_prompt(&[msg("system", "You are helpful."), msg("user", "Hi there")]),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>You are helpful.\
             <|user|>Hi there<|assistant|><think>"
        );
    }

    /// The template writes an assistant HISTORY turn as
    /// `<|assistant|><think></think>{content}`. Writing it without the think
    /// block put every multi-turn conversation off-distribution.
    #[test]
    fn a_glm_assistant_history_turn_carries_a_closed_think_block() {
        let p = super::glm_chat_prompt(&[
            msg("user", "one"),
            msg("assistant", "two"),
            msg("user", "three"),
        ]);
        assert!(
            p.contains("<|assistant|><think></think>two"),
            "assistant history turn is missing its think block: {p}"
        );
        assert!(p.ends_with("<|assistant|><think>"));
    }

    /// A tool result has its own observation block. It used to fall through to
    /// the `_ =>` arm and be rendered as something the HUMAN said.
    #[test]
    fn a_glm_tool_result_is_an_observation_not_a_user_turn() {
        let p = super::glm_chat_prompt(&[msg("tool", "{\"temp\": 12}")]);
        assert!(
            p.contains("<|observation|><tool_response>{\"temp\": 12}</tool_response>"),
            "tool turn was not rendered as an observation: {p}"
        );
        assert!(!p.contains("<|user|>"));
    }

    /// Pinned against `encoding_k3.py::build_chat_segments` RUN on the real
    /// moonshotai/Kimi-K3 snapshot with `thinking=False, add_generation_prompt=True`,
    /// not against a reading of it. K3 ships no jinja template, so this string is
    /// the only executable statement of the format we have.
    #[test]
    fn the_k3_prompt_is_what_encoding_k3_renders() {
        assert_eq!(
            k3_chat_prompt(&[msg("user", "Hello!")]),
            "<|open|>message role=\"user\"<|sep|>Hello!<|close|>message<|sep|><|end_of_msg|>\
             <|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>"
        );
    }

    /// `developer` folds to `system` and `assistant` keeps its own role, so a
    /// multi-turn conversation round-trips into the same segment shape.
    #[test]
    fn the_k3_prompt_maps_every_role() {
        let p = k3_chat_prompt(&[
            msg("developer", "Be terse."),
            msg("user", "hi"),
            msg("assistant", "hello"),
            msg("user", "bye"),
        ]);
        assert!(p.starts_with("<|open|>message role=\"system\"<|sep|>Be terse.<|close|>"));
        assert_eq!(p.matches("<|end_of_msg|>").count(), 4);
        assert_eq!(p.matches("role=\"assistant\"").count(), 2); // one turn + the prompt
        assert!(p.ends_with("<|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>"));
    }
}
