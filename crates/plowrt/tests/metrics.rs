use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use plowrt::device::{cpu::CpuBackend, Backend};
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::mux::{self, MuxConfig};
use plowrt::serve::{app, AppState, Residency};
use tower::ServiceExt;

mod common;

fn setup() -> Arc<AppState> {
    let registry = Registry::new();
    let dir = std::env::temp_dir().join(format!("plowrt-metrics-{}", std::process::id()));
    common::write_bundle_with_batches(&dir, "same-network", &[1, 4]);
    // Two aliases of one network must remain distinct model instances.
    registry.load(&dir, Some("alpha".into())).unwrap();
    registry.load(&dir, Some("beta".into())).unwrap();
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let state = Arc::new(AppState::new(
        registry,
        Arc::new(ExecutorSet::bringup(backend).unwrap()),
    ));
    for slug in state.registry.slugs() {
        start(&state, &slug);
    }
    state
}

fn start(state: &Arc<AppState>, slug: &str) {
    let mux = mux::spawn(
        slug.into(),
        state.registry.get(slug).unwrap(),
        state.clone(),
        MuxConfig {
            max_hold_ms: 0.0,
            ..Default::default()
        },
    );
    state.install_mux(slug.into(), mux);
}

async fn scrape(state: &Arc<AppState>) -> String {
    let response = app(state.clone())
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()["content-type"],
        "text/plain; version=0.0.4; charset=utf-8"
    );
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

async fn complete(state: &Arc<AppState>, model: &str, output: usize, stream: bool) {
    let body = serde_json::json!({"model":model,"prompt":"abc","max_tokens":output,"ignore_eos":true,"stream":stream});
    let response = app(state.clone())
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
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    if stream {
        assert!(String::from_utf8_lossy(&bytes).contains("[DONE]"));
    } else {
        let response: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(response["usage"]["completion_tokens"], output);
    }
}

fn sample(text: &str, metric: &str, model: &str) -> f64 {
    let prefix = format!("{metric}{{model_name=\"{model}\",engine=\"0\"}} ");
    text.lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("missing {prefix}"))
        .parse()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn endpoint_tracks_models_independently_and_survives_residency_reload() {
    let state = setup();
    tokio::join!(
        complete(&state, "alpha", 3, true),
        complete(&state, "beta", 7, false)
    );
    let text = scrape(&state).await;
    for (model, output) in [("alpha", 3.0), ("beta", 7.0)] {
        assert_eq!(sample(&text, "vllm:generation_tokens_total", model), output);
        assert_eq!(sample(&text, "vllm:num_requests_running", model), 0.0);
        assert_eq!(sample(&text, "vllm:num_requests_waiting", model), 0.0);
        assert_eq!(
            sample(&text, "vllm:time_to_first_token_seconds_count", model),
            1.0
        );
        assert_eq!(
            sample(&text, "vllm:inter_token_latency_seconds_count", model),
            output - 1.0
        );
        assert_eq!(
            sample(&text, "vllm:request_generation_tokens_sum", model),
            output
        );
        assert_eq!(sample(&text, "vllm:request_prompt_tokens_sum", model), 3.0);
        assert_eq!(sample(&text, "plowrt_model_requests_total", model), 1.0);
        assert!(sample(&text, "plowrt_tick_batch_size_count", model) > 0.0);
    }
    let mut types = std::collections::HashSet::new();
    let mut samples = std::collections::HashSet::new();
    for line in text.lines() {
        if let Some(declaration) = line.strip_prefix("# TYPE ") {
            assert!(types.insert(declaration.split_whitespace().next().unwrap()));
        } else if !line.starts_with('#') {
            assert!(
                samples.insert(line.rsplit_once(' ').unwrap().0),
                "duplicate {line}"
            );
        }
    }

    state.set_residency("alpha", Residency::Unloading);
    let old = state.remove_mux("alpha").unwrap();
    old.drain().await;
    drop(old);
    state.set_residency("alpha", Residency::Unloaded);
    assert_eq!(
        sample(&scrape(&state).await, "plowrt_model_ready", "alpha"),
        0.0
    );
    state.set_residency("alpha", Residency::Auto);
    start(&state, "alpha");
    complete(&state, "alpha", 2, false).await;
    let text = scrape(&state).await;
    assert_eq!(sample(&text, "vllm:generation_tokens_total", "alpha"), 5.0);
    assert_eq!(sample(&text, "vllm:generation_tokens_total", "beta"), 7.0);
    assert_eq!(sample(&text, "plowrt_model_ready", "alpha"), 1.0);

    for slug in state.registry.slugs() {
        let mux = state.remove_mux(&slug).unwrap();
        mux.drain().await;
        drop(mux);
    }
    state.registry.unload("alpha").unwrap();
    // Drain completion precedes the dispatcher dropping its final metric Arc.
    for _ in 0..100 {
        let text = scrape(&state).await;
        if !text.contains("model_name=\"alpha\"") {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("unregistered model retained metric series");
}
