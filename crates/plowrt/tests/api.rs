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
    let mut registry = Registry::new();
    registry.load(&dir, None).unwrap();
    let state = Arc::new(AppState::new(registry, execset));

    // Match the production shape: install a bucket muxer per registered slug.
    let slugs: Vec<String> = state.registry.slugs().map(str::to_string).collect();
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
    for (special, expected) in [
        (None, vec![0]),
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
