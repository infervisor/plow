//! §G `/v1/chat/completions` — streaming (SSE) and non-streaming.

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
pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    // `Result<Json<..>, JsonRejection>` rather than `Json<..>`: axum's default
    // rejection is a PLAIN-TEXT 400/415/422, and a client that calls
    // `resp.json()` on a 4xx — every OpenAI SDK does — raises a decode error
    // instead of showing the user what was wrong with their request.
    req: Result<Json<ChatRequest>, axum::extract::rejection::JsonRejection>,
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
    for (val, field) in [
        (&req.tools, "tools"),
        (&req.tool_choice, "tool_choice"),
        (&req.functions, "functions"),
        (&req.function_call, "function_call"),
        (&req.response_format, "response_format"),
        (&req.logprobs, "logprobs"),
        (&req.top_logprobs, "top_logprobs"),
    ] {
        if val.as_ref().is_some_and(|v| !v.is_null()) {
            return crate::serve::api_error(
                axum::http::StatusCode::BAD_REQUEST,
                format!(
                    "`{field}` is not implemented by this server; it is refused rather than \
                     ignored, because ignoring it returns a confidently wrong answer"
                ),
                "invalid_request_error",
                Some("unsupported_parameter"),
                Some(field.into()),
            );
        }
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
    let prompt = crate::obs::ttft::timed(&crate::obs::ttft::TEMPLATE, || {
        if state.has_gpu_engine(&req.model) {
            let tok = state.registry.get(&req.model).ok();
            // THE CHECKPOINT'S OWN TEMPLATE FIRST. The built-in per-family
            // builders are the fallback for checkpoints that ship none — they
            // are an approximation of a file the weights already carry, and
            // every divergence between the two is a wrong prompt.
            if let Some(t) = tok.as_deref().and_then(|b| b.chat_template()) {
                let msgs: Vec<serde_json::Value> = req
                    .messages
                    .iter()
                    .map(|m| {
                        serde_json::json!({
                            "role": m.role,
                            "content": m.text(),
                        })
                    })
                    .collect();
                match t.render(&msgs) {
                    Ok(p) => return p,
                    Err(e) => {
                        // The template REFUSED this conversation (HF templates
                        // call `raise_exception` for shapes they cannot render,
                        // e.g. an out-of-order role sequence). Surface it as a
                        // 400 rather than silently rendering something else.
                        template_error = Some(e);
                        return String::new();
                    }
                }
            }
            gpu_chat_prompt(tok.as_deref(), &req.messages)
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

    if let Some(e) = template_error {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            format!("the model's chat template rejected this conversation: {e}"),
            "invalid_request_error",
            Some("invalid_message_sequence"),
            Some("messages".into()),
        );
    }

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
    if let Some(stop) = &req.stop {
        gen.stop = stop.list();
    }
    gen.seed = req.seed;

    // NO VISION PATH EXISTS. `Content::as_text()` flattens a multipart message by
    // keeping the `Text` parts and dropping everything else, so an `image_url`
    // part is discarded and the model answers from the surrounding text alone —
    // fluently, and about the wrong question. `Content::has_image` was written
    // for exactly this check and had NO CALLERS, so the drop was silent.
    //
    // This is the same failure the compile-time refusals guard against
    // (`nn-graph`'s kimi_k3 builder and `plowc`'s `synth_kimi_k3` both refuse a
    // `vision_config` rather than build a text tower that is "silently wrong on
    // every image prompt"). Those cover the COMPILER. Nothing covered the
    // SERVER, so a text-tower build of any multimodal checkpoint — Kimi-K3,
    // Gemma-4, Qwen-VL — would take an image request and quietly ignore it.
    //
    // Refused rather than errored later: the request never becomes a token, so
    // it is not counted, not tokenized and not admitted.
    if req.messages.iter().any(|m| m.has_image()) {
        return crate::serve::api_error(
            axum::http::StatusCode::BAD_REQUEST,
            "this build serves the TEXT tower only and has no vision stage; an image part \
             would be silently dropped, so the request is refused",
            "invalid_request_error",
            Some("unsupported_parameter"),
            Some("messages[].content".into()),
        );
    }

    // Route to the per-model muxer. Tokens stream back as `StreamChunk`s over
    // an mpsc — the muxer produces one per generated token, ending with `Done`.
    crate::obs::Metrics::inc(&state.metrics.requests);
    let (Some(mux), Ok(bundle)) = (state.mux(&req.model), state.registry.get(&req.model)) else {
        return crate::serve::api_error(
            axum::http::StatusCode::NOT_FOUND,
            format!("no model registered for '{}'.", req.model),
            "invalid_request_error",
            Some("model_not_found"),
            Some("model".into()),
        );
    };
    // Tokenize HERE, on the handler task — the dispatcher loop is the
    // serialized decode critical path and must never encode a long prompt.
    let reasoning_open = opens_reasoning(&prompt);
    let prompt_ids = crate::obs::ttft::timed(&crate::obs::ttft::ENCODE, || {
        bundle.tokenizer().encode(&prompt)
    });
    let n_prompt = prompt_ids.len();
    let (tx, rx) = stream_mod::channel();
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

    let request_id = request_id();
    // Stamped ONCE and repeated on every chunk of a stream, as OpenAI does.
    let created = now_secs();
    if req.stream {
        let include_usage = req.stream_options.map(|o| o.include_usage).unwrap_or(false);
        sse_response(
            request_id,
            req.model,
            rx,
            include_usage,
            t_arrive,
            n_prompt,
            created,
            reasoning_open,
        )
        .into_response()
    } else {
        buffer_and_reply(request_id, req.model, rx, created).await
    }
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

/// Split a GLM answer into (reasoning, answer) at the first `</think>`.
///
/// The generation prompt leaves `<think>` open, so the model emits its trace
/// and then closes it. `</think>` is `special: false` in GLM's added tokens, so
/// `skip_special_tokens` does NOT remove it and it would otherwise land in the
/// user-visible `content` as literal text along with the whole trace. Returns
/// `(None, whole)` until the marker is seen, which is also the correct answer
/// for a model that never opened a trace.
pub fn split_reasoning(text: &str) -> (Option<String>, String) {
    const CLOSE: &str = "</think>";
    match text.find(CLOSE) {
        Some(i) => {
            let reasoning = text[..i].trim().to_string();
            let answer = text[i + CLOSE.len()..].trim_start().to_string();
            (
                (!reasoning.is_empty()).then_some(reasoning),
                answer,
            )
        }
        None => (None, text.to_string()),
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
) -> Response {
    let mut text = String::new();
    // `None` until a terminal chunk arrives. It must NOT default to "stop": the
    // mux frees a slot on ANY `try_send` failure, and a bounded-channel `Full`
    // (serve/stream.rs caps the stream at 32 chunks) drops the sender without
    // sending `Done` or `Err`. Defaulting to "stop" turned that into a 200 with
    // a fluent, truncated answer that claimed to be complete — invisible to a
    // benchmark client, which counts it as a successful request.
    let mut finish: Option<stream_mod::FinishReason> = None;
    let mut usage = None;
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { text: delta, .. } => text.push_str(&delta),
            StreamChunk::Done {
                reason, usage: u, ..
            } => {
                finish = Some(reason);
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
    // and the trace goes to `reasoning_content`, which is where a reasoning
    // model's clients look for it. A model that never opened a trace is
    // unaffected — `split_reasoning` returns the whole string as the answer.
    let (reasoning, answer) = split_reasoning(&text);
    Json(ChatResponse {
        id: request_id.clone(),
        object: "chat.completion",
        created,
        model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".into(),
                content: Some(Content::Text(answer)),
                reasoning_content: reasoning,
            },
            // The WIRE value, which is not always the internal one: a
            // preemption is reported as "length" because "preempted" is not an
            // OpenAI finish_reason and a typed client rejects the response on
            // sight. Collapsing it to "stop" would be worse still — a
            // truncated answer claiming to be complete — so the real cause
            // rides along in `x_plow_finish_reason`.
            finish_reason: Some(finish.as_openai()),
            x_plow_finish_reason: finish.is_vendor_specific().then(|| finish.as_str()),
        }],
        usage,
    })
    .into_response()
}

/// Does the prompt hand generation an OPEN `<think>` block?
///
/// Only then do the first tokens belong in `reasoning_content`. The GLM
/// template ends its generation prompt with `<|assistant|><think>`; Gemma,
/// Llama and Qwen templates end with an ordinary assistant turn. Assuming
/// reasoning unconditionally routed the first token of EVERY model into
/// `reasoning_content` and left `delta.content` absent on the chunk that
/// stamps the client's TTFT — the same measurement hazard the role delta
/// documents below, reached by a different path.
fn opens_reasoning(prompt: &str) -> bool {
    match prompt.rfind("<think>") {
        None => false,
        Some(open) => prompt.rfind("</think>").is_none_or(|close| close < open),
    }
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
    reasoning_open: bool,
) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    // State threaded through the unfold: the receiver, and the tail frames
    // (optional usage-only chunk, then [DONE]) drained one per poll.
    struct SseState {
        rx: stream_mod::ChunkReceiver,
        done: bool,
        /// The `role` delta has not been sent yet — it rides the FIRST token.
        role_pending: bool,
        /// Still inside the model's `<think>` trace: deltas route to
        /// `reasoning_content` until `</think>` arrives. The marker can be
        /// split across token boundaries, so a small tail is held back.
        in_reasoning: bool,
        hold: String,
        pending: std::collections::VecDeque<Event>,
    }
    let body = stream::unfold(
        SseState {
            rx,
            done: false,
            role_pending: true,
            in_reasoning: reasoning_open,
            hold: String::new(),
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
                        // No terminal chunk: the mux freed the slot on a
                        // `try_send` failure (the bounded 32-chunk stream
                        // filling counts, not just the client dropping) or a
                        // tick panicked. The wire shape is left alone — the
                        // stream ends without `[DONE]`, which is the only
                        // signal SSE has — but it must not be SILENT
                        // server-side, because a bench client scores this as a
                        // successful request with fewer output tokens.
                        tracing::warn!(
                            %model,
                            "chat: SSE stream ended with no terminal chunk (slot cut) \
                             — no finish_reason, no [DONE]"
                        );
                        return None;
                    }
                };
                let (frame, terminate) = match chunk {
                    StreamChunk::Token { text, .. } => {
                        let role = st.role_pending.then(|| {
                            st.role_pending = false;
                            "assistant"
                        });
                        // §TTFT: this frame is the one `vllm bench serve` stamps.
                        if role.is_some() {
                            crate::obs::ttft::dump(t_arrive.elapsed().as_nanos() as u64, n_prompt);
                            crate::obs::pfx::report();
                        }
                        // Route the delta to `reasoning_content` until the
                        // trace closes. `</think>` is NOT a special token, so
                        // without this the whole trace and the literal marker
                        // land in the user-visible `content`.
                        const CLOSE: &str = "</think>";
                        let (reasoning, content) = if st.in_reasoning {
                            st.hold.push_str(&text);
                            match st.hold.find(CLOSE) {
                                Some(i) => {
                                    let before = st.hold[..i].to_string();
                                    let after = st.hold[i + CLOSE.len()..].to_string();
                                    st.hold.clear();
                                    st.in_reasoning = false;
                                    (
                                        (!before.is_empty()).then_some(before),
                                        Some(after),
                                    )
                                }
                                None => {
                                    // Hold back only as much as could still be
                                    // a prefix of the marker; emit the rest.
                                    let keep = CLOSE.len().min(st.hold.len());
                                    let cut = (0..=keep)
                                        .rev()
                                        .map(|k| st.hold.len() - k)
                                        .find(|&c| st.hold.is_char_boundary(c))
                                        .unwrap_or(st.hold.len());
                                    let emit = st.hold[..cut].to_string();
                                    st.hold = st.hold[cut..].to_string();
                                    ((!emit.is_empty()).then_some(emit), None)
                                }
                            }
                        } else {
                            (None, Some(text))
                        };
                        let ch = ChatChunk {
                            id: request_id.clone(),
                            object: "chat.completion.chunk",
                            created,
                            model,
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role,
                                    content,
                                    reasoning_content: reasoning,
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
                            created,
                            model: model.clone(),
                            choices: vec![ChunkChoice {
                                index: 0,
                                delta: Delta {
                                    role: None,
                                    content: None,
                                    reasoning_content: None,
                                },
                                // The wire value: "preempted" is not an OpenAI
                                // finish_reason and a typed client rejects it.
                                finish_reason: Some(reason.as_openai()),
                            }],
                            usage: None,
                        };
                        if include_usage {
                            // OpenAI stream-usage shape: a separate chunk with
                            // EMPTY choices carries usage, never the finish chunk.
                            let uch = ChatChunk {
                                id: request_id.clone(),
                                object: "chat.completion.chunk",
                                created,
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
                        tracing::warn!(%model, error = %e, "chat: SSE stream error");
                        let body = crate::serve::openai::ApiErrorBody::new(
                            e.to_string(),
                            "server_error",
                            None,
                            None,
                        );
                        let data = serde_json::to_string(&body)
                            .unwrap_or_else(|_| {
                                "{\"error\":{\"message\":\"stream error\",\"type\":\"server_error\"}}"
                                    .to_string()
                            });
                        st.done = true;
                        return Some((Ok(Event::default().data(data)), st));
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

    Sse::new(body)
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
            reasoning_content: None,
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
        let t = crate::serve::template::ChatTemplate::load(dir).expect("template compiles");
        for convo in [
            vec![("system", "You are helpful."), ("user", "Hi there")],
            vec![("user", "one"), ("assistant", "two"), ("user", "three")],
        ] {
            let built = gemma_chat_prompt(
                &convo
                    .iter()
                    .map(|&(r, c)| msg(r, c))
                    .collect::<Vec<_>>(),
            );
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

    /// `</think>` is `special: false` in GLM's added tokens, so it survives
    /// `skip_special_tokens` and would otherwise reach the user as literal
    /// text along with the whole trace.
    #[test]
    fn the_reasoning_trace_is_split_out_of_the_answer() {
        let (r, a) = super::split_reasoning("weighing it up</think>The answer is 4.");
        assert_eq!(r.as_deref(), Some("weighing it up"));
        assert_eq!(a, "The answer is 4.");
    }

    /// A model that never opened a trace must come back unchanged, not empty.
    #[test]
    fn text_without_a_think_marker_is_all_answer() {
        let (r, a) = super::split_reasoning("just an answer");
        assert!(r.is_none());
        assert_eq!(a, "just an answer");
    }

    #[test]
    fn only_an_open_think_block_starts_the_reasoning_router() {
        // GLM's generation prompt hands generation an open trace.
        assert!(super::opens_reasoning("<|assistant|><think>"));
        // A closed trace in the HISTORY does not.
        assert!(!super::opens_reasoning(
            "<|assistant|><think></think>hi<|user|>again<|assistant|>"
        ));
        // Gemma, Llama and Qwen never mention it.
        assert!(!super::opens_reasoning("<start_of_turn>model\n"));
        // `</think>` must not read as an opening marker.
        assert!(!super::opens_reasoning("done</think>"));
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
