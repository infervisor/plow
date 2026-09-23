//! OpenAI API surface: models, chat/raw completions, and tokenizer alignment
//! over the CPU backend.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use plowrt::device::cpu::CpuBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::mux::{self, MuxConfig};
use plowrt::serve::{app, AppState};
use tower::ServiceExt;

mod common;

use std::sync::atomic::{AtomicU32, Ordering};
static DIR_SEQ: AtomicU32 = AtomicU32::new(0);

fn make_app() -> axum::Router {
    make_app_with_tokenizer(None)
}

fn make_app_with_tokenizer(tokenizer_json: Option<&str>) -> axum::Router {
    // Unique dir per call — the tests run in parallel and must not share assets.
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_api_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle(&dir, "api-model");
    if let Some(json) = tokenizer_json {
        std::fs::write(dir.join("tokenizer.json"), json).unwrap();
    }

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let registry = Registry::new();
    registry.load(&dir, None).unwrap();
    let state = Arc::new(AppState::new(registry, execset));

    // Match the production shape: install a bucket muxer per registered slug.
    let slugs: Vec<String> = state.registry.slugs();
    for slug in slugs {
        let bundle = state.registry.get(&slug).unwrap();
        let m = mux::spawn(
            slug.clone(),
            bundle,
            Arc::clone(&state),
            MuxConfig::default(),
        );
        state.install_mux(slug, m);
    }
    app(state)
}

async fn body_string(resp: axum::response::Response) -> String {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

async fn token_prompt(app: &axum::Router, prompt: serde_json::Value) -> (StatusCode, String) {
    let body = serde_json::json!({"model": "api-model", "prompt": prompt, "max_tokens": 1});
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    (resp.status(), body_string(resp).await)
}

#[tokio::test]
async fn token_id_prompts_outside_the_vocabulary_are_refused() {
    let app = make_app();
    for prompt in [serde_json::json!([65, 256]), serde_json::json!([[4294967295u32]])] {
        let (status, body) = token_prompt(&app, prompt).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("outside the vocabulary size 256"), "{body}");
        assert!(body.contains("invalid_prompt"), "{body}");
    }
    assert_eq!(token_prompt(&app, serde_json::json!([65, 255])).await.0, StatusCode::OK);
}

#[cfg(feature = "hf-tokenizer")]
#[tokio::test]
async fn hf_tokenizer_vocab_bounds_token_id_prompts() {
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,
        "model":{"type":"WordLevel","vocab":{"hello":0,"world":1,"[UNK]":2},"unk_token":"[UNK]"}}"#;
    let app = make_app_with_tokenizer(Some(json));
    for _ in 0..2 {
        let (status, body) = token_prompt(&app, serde_json::json!([0, 3])).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body.contains("prompt token id 3 is outside the vocabulary size 3"), "{body}");
        assert_eq!(token_prompt(&app, serde_json::json!([0, 2])).await.0, StatusCode::OK);
    }
}

#[tokio::test]
async fn lists_models() {
    let resp = make_app()
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    assert!(body.contains("api-model"));
    assert!(body.contains("\"object\":\"list\""));
}

#[tokio::test]
async fn non_stream_completion() {
    let req_body = serde_json::json!({
        "model": "api-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 4
    });
    let resp = make_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    assert!(body.contains("\"object\":\"chat.completion\""));
    assert!(body.contains("api-model"));
}

#[tokio::test]
async fn stream_completion_terminates() {
    let req_body = serde_json::json!({
        "model": "api-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "max_tokens": 4
    });
    let resp = make_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    assert!(body.contains("chat.completion.chunk"));
    assert!(body.contains("[DONE]"));

    let frames: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    // The `role` delta RIDES THE FIRST TOKEN — it is not a frame of its own.
    //
    // This assertion used to be the exact opposite (`content` absent from frame
    // 0), which pinned a measurement artefact as if it were the API contract.
    // `vllm bench serve` stamps TTFT on the first chunk carrying a `choices`
    // array, whatever is in it (`backend_request_func.py`), so a role-only frame
    // emitted at request arrival stamps TTFT at arrival. Measured on gfx950 with
    // a 7013-token prompt: role frame at 7.1 ms, first real token at 1322 ms —
    // a 188x understatement, and plow-specific, because vLLM sends nothing
    // before its first token. Removed in 63f9957; this test is what stops it
    // coming back.
    let first_delta = &frames[0]["choices"][0]["delta"];
    assert_eq!(first_delta["role"], "assistant");
    assert!(
        first_delta["content"].is_string(),
        "the first streamed chunk must carry a REAL TOKEN, not just the role: a \
         chunk with a `choices` array and no content still stamps the client's \
         TTFT. Got {first_delta}"
    );
    let second_delta = &frames[1]["choices"][0]["delta"];
    assert!(second_delta.get("role").is_none());
    assert!(second_delta["content"].is_string());
    let request_id = frames[0]["id"].as_str().unwrap();
    assert!(frames.iter().all(|frame| frame["id"] == request_id));
}

#[tokio::test]
async fn raw_completion_uses_text_completion_shape() {
    let req_body = serde_json::json!({
        "model": "api-model",
        "prompt": "hello",
        "stream": false,
        "max_tokens": 4,
        "temperature": 0.0,
        "ignore_eos": true
    });
    let resp = make_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert_eq!(body["object"], "text_completion");
    assert!(body["choices"][0]["text"].is_string());
    assert!(body["choices"][0]["finish_reason"].is_string());
    assert_eq!(body["usage"]["prompt_tokens"], 5);
}

#[tokio::test]
async fn raw_completion_stream_has_no_empty_leading_choice() {
    let req_body = serde_json::json!({
        "model": "api-model",
        "prompt": "hello",
        "stream": true,
        "max_tokens": 4,
        "ignore_eos": true,
        "stream_options": {"include_usage": true}
    });
    let resp = make_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    assert!(body.contains("[DONE]"));
    let frames: Vec<serde_json::Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert_eq!(frames[0]["object"], "text_completion");
    assert!(frames[0]["choices"][0]["text"].is_string());
    assert!(frames[0]["choices"][0]["finish_reason"].is_null());
    let usage = frames.last().unwrap();
    assert_eq!(usage["choices"].as_array().unwrap().len(), 0);
    assert!(usage["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn tokenizer_alignment_round_trips() {
    let app = make_app();
    let resp = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/tokenize")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "api-model",
                        "prompt": "hello",
                        "add_special_tokens": false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let tokenized: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert_eq!(
        tokenized["tokens"],
        serde_json::json!([104, 101, 108, 108, 111])
    );

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/detokenize")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({"model": "api-model", "tokens": tokenized["tokens"]})
                        .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let detokenized: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
    assert_eq!(detokenized["prompt"], "hello");
}

#[tokio::test]
async fn unknown_model_404() {
    let req_body = serde_json::json!({
        "model": "does-not-exist",
        "messages": [{"role": "user", "content": "hi"}]
    });
    let resp = make_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(req_body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[cfg(feature = "hf-tokenizer")]
#[tokio::test]
async fn completion_and_tokenize_apply_requested_special_tokens() {
    use plowrt::text::tokenizer::{HfTokenizer, Tokenize};
    use tokenizers::{models::wordlevel::WordLevel, processors::template::TemplateProcessing};

    let model = WordLevel::builder()
        .vocab(
            [
                ("hello".to_string(), 0),
                ("[UNK]".to_string(), 1),
                ("<bos>".to_string(), 2),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".to_string())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("<bos> $A")
            .unwrap()
            .special_tokens(vec![("<bos>", 2)])
            .build()
            .unwrap(),
    ));
    let json = tokenizer.to_string(false).unwrap();
    let app = make_app_with_tokenizer(Some(&json));
    for (special, expected) in [
        (None, vec![2, 0]),
        (Some(true), vec![2, 0]),
        (Some(false), vec![0]),
    ] {
        let mut body = serde_json::json!({"model":"api-model", "prompt":"hello",
            "max_tokens":1, "return_token_ids":true});
        if let Some(value) = special {
            body["add_special_tokens"] = value.into();
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let result: serde_json::Value = serde_json::from_str(&body_string(response).await).unwrap();
        assert_eq!(result["token_ids"]["prompt"], serde_json::json!(expected));
        assert_eq!(result["usage"]["prompt_tokens"], expected.len());
    }
    // `/tokenize` defaults `add_special_tokens` to true, like `/v1/completions` and vLLM.
    for (special, expected) in [
        (None, vec![2, 0]),
        (Some(true), vec![2, 0]),
        (Some(false), vec![0]),
    ] {
        let mut body = serde_json::json!({"model":"api-model", "prompt":"hello"});
        if let Some(value) = special {
            body["add_special_tokens"] = value.into();
        }
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/tokenize")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let result: serde_json::Value = serde_json::from_str(&body_string(response).await).unwrap();
        assert_eq!(result["tokens"], serde_json::json!(expected));
    }
    // Chat renders its own special tokens and continues to use plain encode.
    let dir = std::env::temp_dir().join(format!("plowrt_api_plain_encode_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("tokenizer.json"), json).unwrap();
    let tokenizer = HfTokenizer::from_file(&dir.join("tokenizer.json")).unwrap();
    assert_eq!(tokenizer.encode("hello"), vec![0]);
    assert_eq!(
        tokenizer.encode_with_special_tokens("hello", true),
        vec![2, 0]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

/// Helper: POST a chat body and return the status plus the body text.
async fn chat(body: serde_json::Value) -> (StatusCode, String) {
    let resp = make_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    (status, body_string(resp).await)
}

fn base_chat() -> serde_json::Value {
    serde_json::json!({
        "model": "api-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4
    })
}

/// Routers fetch one card to read `max_model_len` before sizing a request.
/// The route did not exist, so they got a 404 from a server serving the model.
#[tokio::test]
async fn a_single_model_card_is_fetchable() {
    let app = make_app();
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/models/api-model")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    assert!(body.contains("\"id\":\"api-model\""), "{body}");
    assert!(body.contains("\"object\":\"model\""), "{body}");
    assert!(body.contains("\"root\":\"api-model\""), "{body}");
}

#[tokio::test]
async fn an_unknown_model_card_is_a_404_not_a_panic() {
    let resp = make_app()
        .oneshot(
            Request::builder()
                .uri("/v1/models/nope")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body = body_string(resp).await;
    assert!(body.contains("model_not_found"), "{body}");
}

/// `created` was `now_secs()` evaluated per card, so it changed on every
/// scrape and a client diffing the catalogue saw every model as new.
#[tokio::test]
async fn the_model_card_created_stamp_is_stable_across_scrapes() {
    let app = make_app();
    let first = {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        body_string(resp).await
    };
    // A second later in wall-clock terms would change a per-call `now_secs()`.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let second = {
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        body_string(resp).await
    };
    assert_eq!(first, second, "the catalogue must not change between scrapes");
}

/// The host sampler implements these; serde used to drop them, so the request
/// was answered under different sampling than it asked for, with a 200.
#[tokio::test]
async fn the_sampler_knobs_are_accepted_not_dropped() {
    let mut body = base_chat();
    body["top_k"] = serde_json::json!(20);
    body["min_p"] = serde_json::json!(0.05);
    body["repetition_penalty"] = serde_json::json!(1.1);
    body["presence_penalty"] = serde_json::json!(0.5);
    body["frequency_penalty"] = serde_json::json!(0.5);
    body["logit_bias"] = serde_json::json!({"5": 1.5});
    body["min_tokens"] = serde_json::json!(1);
    body["stop_token_ids"] = serde_json::json!([9999]);
    let (status, text) = chat(body).await;
    assert_eq!(status, StatusCode::OK, "{text}");
}

/// vLLM's escape hatch into the template. Dropping it meant `enable_thinking:
/// false` did nothing here while it worked against vLLM.
#[tokio::test]
async fn chat_template_kwargs_are_accepted() {
    let mut body = base_chat();
    body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});
    body["reasoning_effort"] = serde_json::json!("low");
    let (status, text) = chat(body).await;
    assert_eq!(status, StatusCode::OK, "{text}");
}

/// `top_p: 0` truncates the candidate set to nothing and a negative
/// temperature falls through the greedy branch. OpenAI answers 400; this
/// server used to accept them and sample from the result.
#[tokio::test]
async fn out_of_range_sampling_is_a_400() {
    for (field, value) in [
        ("top_p", serde_json::json!(0.0)),
        ("temperature", serde_json::json!(-1.0)),
        ("min_p", serde_json::json!(2.0)),
        ("presence_penalty", serde_json::json!(9.0)),
        ("frequency_penalty", serde_json::json!(-9.0)),
        ("repetition_penalty", serde_json::json!(0.0)),
        ("top_k", serde_json::json!(-5)),
    ] {
        let mut body = base_chat();
        body[field] = value.clone();
        let (status, text) = chat(body).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{field}={value} should be refused, got {status}: {text}"
        );
        assert!(text.contains(field), "the error must name the field: {text}");
    }
}

/// A non-integer key is not a token id, and a bias outside OpenAI's documented
/// range is a client bug worth naming rather than silently clamping.
#[tokio::test]
async fn a_malformed_logit_bias_is_a_400() {
    let mut body = base_chat();
    body["logit_bias"] = serde_json::json!({"not-an-id": 1.0});
    let (status, text) = chat(body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");

    let mut body = base_chat();
    body["logit_bias"] = serde_json::json!({"5": 500.0});
    let (status, text) = chat(body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
}

/// `min_tokens` above `max_tokens` can never finish; saying so beats holding
/// a slot open until the cap cuts it.
#[tokio::test]
async fn min_tokens_above_max_tokens_is_a_400() {
    let mut body = base_chat();
    body["min_tokens"] = serde_json::json!(99);
    let (status, text) = chat(body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("min_tokens"), "{text}");
}

/// The same surface on /v1/completions, which shares `SamplingFields`.
#[tokio::test]
async fn the_completions_endpoint_validates_the_same_way() {
    let body = serde_json::json!({
        "model": "api-model",
        "prompt": "hi",
        "max_tokens": 4,
        "top_p": 0.0
    });
    let resp = make_app()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

/// Same shape as `make_app`, plus an extra served name for the one model.
fn make_app_with_alias(alias: &str) -> axum::Router {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_api_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle(&dir, "api-model");

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let registry = Registry::new();
    registry.load(&dir, None).unwrap();
    registry.add_alias(alias.to_string(), "api-model").unwrap();
    let state = Arc::new(AppState::new(registry, execset));
    for slug in state.registry.slugs() {
        let bundle = state.registry.get(&slug).unwrap();
        let m = mux::spawn(
            slug.clone(),
            bundle,
            Arc::clone(&state),
            MuxConfig::default(),
        );
        state.install_mux(slug, m);
    }
    app(state)
}

/// A client that hardcodes a model name it cannot change must be servable
/// without renaming the bundle (which would change every metric label with it).
#[tokio::test]
async fn a_request_for_an_alias_is_served_and_echoes_the_requested_name() {
    let body = serde_json::json!({
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4
    });
    let resp = make_app_with_alias("gpt-3.5-turbo")
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_string(resp).await;
    // The RESPONSE echoes what the client asked for, as vLLM does — not the
    // canonical slug it was resolved to.
    assert!(text.contains("\"model\":\"gpt-3.5-turbo\""), "{text}");
}

/// An alias must be discoverable, or a client cannot learn the name works.
#[tokio::test]
async fn aliases_appear_in_the_catalogue_pointing_at_their_target() {
    let resp = make_app_with_alias("gpt-3.5-turbo")
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let text = body_string(resp).await;
    assert!(text.contains("\"id\":\"api-model\""), "{text}");
    assert!(text.contains("\"id\":\"gpt-3.5-turbo\""), "{text}");
    assert!(text.contains("\"parent\":\"api-model\""), "{text}");
}

#[tokio::test]
async fn a_single_card_is_fetchable_by_alias() {
    let resp = make_app_with_alias("gpt-3.5-turbo")
        .oneshot(
            Request::builder()
                .uri("/v1/models/gpt-3.5-turbo")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let text = body_string(resp).await;
    assert!(text.contains("\"root\":\"api-model\""), "{text}");
}

/// An empty conversation used to be answered with a 200: the template renders
/// a bare generation prompt and the model invents a question and answers it.
#[tokio::test]
async fn an_empty_messages_array_is_a_400() {
    let (status, text) = chat(serde_json::json!({
        "model": "api-model",
        "messages": [],
        "max_tokens": 8
    }))
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{text}");
    assert!(text.contains("messages"), "{text}");
}
