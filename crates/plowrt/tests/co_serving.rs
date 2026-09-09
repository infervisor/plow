use plowrt::device::{cpu::CpuBackend, Backend};
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::mux::{self, Job, ModelMux, MuxConfig};
use plowrt::serve::stream::{self, StreamChunk};
use plowrt::serve::{AppState, GenParams};
use std::sync::Arc;
use std::time::{Duration, Instant};
mod common;

fn setup(label: &str) -> (Arc<AppState>, Vec<ModelMux>) {
    setup_groups(label, 1)
}

fn setup_groups(label: &str, groups: usize) -> (Arc<AppState>, Vec<ModelMux>) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        let mut cfg = plowrt::config::RuntimeConfig::get().clone();
        cfg.co_sched = plowrt::serve::cosched::CoSched::Rr;
        cfg.co_sched_quantum = 4;
        plowrt::config::RuntimeConfig::init(cfg);
    });
    let registry = Registry::new();
    for name in ["a", "b"] {
        let slug = format!("{label}-{name}");
        let dir = std::env::temp_dir().join(format!("review-{}-{slug}", std::process::id()));
        common::write_bundle_with_batches(&dir, &slug, &[1, 4]);
        registry.load(dir, None).unwrap();
    }
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let state = Arc::new(AppState::new(
        registry,
        Arc::new(ExecutorSet::bringup(backend).unwrap()),
    ));
    state.install_device_turns(groups);
    assert!(state.device_turn("a").unwrap().ordered());
    let muxes = state
        .registry
        .slugs()
        .into_iter()
        .enumerate()
        .map(|(i, slug)| {
            state.set_slug_group(&slug, i % groups);
            let cfg = MuxConfig {
                max_hold_ms: 0.0,
                ..Default::default()
            };
            let m = mux::spawn(
                slug.clone(),
                state.registry.get(&slug).unwrap(),
                state.clone(),
                cfg,
            );
            state.install_mux(slug, m.clone());
            m
        })
        .collect();
    (state, muxes)
}

fn submit(m: &ModelMux, prompt: u32, n: usize) -> stream::ChunkReceiver {
    let (tx, rx) = stream::channel();
    m.submit(Job {
        prompt_ids: vec![prompt, 42, 99],
        gen: GenParams {
            max_tokens: n,
            ignore_eos: true,
            ..Default::default()
        },
        arrived: Instant::now(),
        respond: tx,
    })
    .unwrap_or_else(|_| panic!("submit failed"));
    rx
}

async fn collect(mut rx: stream::ChunkReceiver) -> (Vec<u32>, bool) {
    let mut tokens = Vec::new();
    let mut done = false;
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .unwrap()
    {
        match chunk {
            StreamChunk::Token { id, .. } => tokens.push(id),
            StreamChunk::Done { usage, .. } => {
                assert_eq!(usage.completion_tokens, tokens.len());
                done = true;
            }
            StreamChunk::Err(e) => panic!("stream error: {e}"),
        }
    }
    (tokens, done)
}

#[tokio::test]
async fn concurrent_models_keep_response_channels_and_reused_slots_isolated() {
    let (_state, muxes) = setup("isolation");
    let mut expected = Vec::new();
    for i in 0..8 {
        let result = collect(submit(&muxes[i % 2], 65 + i as u32, 12)).await;
        assert!(result.1);
        expected.push(result.0);
    }
    for _ in 0..3 {
        let tasks: Vec<_> = (0..8)
            .map(|i| tokio::spawn(collect(submit(&muxes[i % 2], 65 + i as u32, 12))))
            .collect();
        for (i, t) in tasks.into_iter().enumerate() {
            let (tokens, done) = t.await.unwrap();
            assert!(done);
            assert_eq!(
                tokens, expected[i],
                "response {i} mixed with another slot/model"
            );
        }
    }
    for m in muxes {
        m.preempt().await;
    }
}

#[tokio::test]
async fn preempt_waiting_model_does_not_need_another_models_device_turn() {
    let (state, muxes) = setup("waiting");
    let guard = state.device_turn("waiting-a").unwrap().acquire().await;
    let rx = submit(&muxes[0], 65, 12);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let m = muxes[0].clone();
    let mut stop = tokio::spawn(async move { m.preempt().await });
    let completed = tokio::time::timeout(Duration::from_millis(100), &mut stop)
        .await
        .is_ok();
    drop(guard);
    if !completed {
        tokio::time::timeout(Duration::from_secs(2), stop)
            .await
            .unwrap()
            .unwrap();
    }
    let (tokens, done) = collect(rx).await;
    muxes[1].preempt().await;
    assert!(completed, "preempt blocked behind an unrelated model's turn; produced {} tokens after stop was requested (terminal={done})", tokens.len());
}

#[tokio::test]
async fn final_token_filling_channel_still_delivers_terminal() {
    let (_state, muxes) = setup("terminal");
    let rx = submit(&muxes[0], 65, 32);
    tokio::time::sleep(Duration::from_millis(200)).await;
    let (tokens, done) = collect(rx).await;
    for m in muxes {
        m.preempt().await;
    }
    assert_eq!(tokens.len(), 32);
    assert!(
        done,
        "all 32 tokens arrived, but the normal completion terminal was dropped"
    );
}

#[tokio::test]
async fn busy_device_group_does_not_stall_another_groups_response() {
    let (state, muxes) = setup_groups("different-gpus", 2);
    let guard = state
        .device_turn("different-gpus-a")
        .unwrap()
        .acquire()
        .await;
    let a = submit(&muxes[0], 65, 12);
    let b = submit(&muxes[1], 66, 12);
    let b = tokio::time::timeout(Duration::from_millis(200), collect(b)).await;
    drop(guard);
    assert!(b.is_ok(), "GPU 0 turn blocked GPU 1 response");
    assert!(b.unwrap().1);
    assert!(collect(a).await.1);
    for m in muxes {
        m.preempt().await;
    }
}

#[tokio::test]
async fn a_slow_consumer_gets_an_error_while_another_model_completes() {
    let (_state, muxes) = setup("slow-consumer");
    let mut slow = submit(&muxes[0], 65, 64);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert!(collect(submit(&muxes[1], 66, 12)).await.1);
    let mut terminal = false;
    while let Some(chunk) = slow.recv().await {
        match chunk {
            StreamChunk::Err(e) => {
                assert!(e.to_string().contains("consumer is too slow"));
                terminal = true;
            }
            StreamChunk::Done { .. } => panic!("slow stream claimed normal completion"),
            _ => {}
        }
    }
    assert!(terminal);
    for m in muxes {
        m.preempt().await;
    }
}

async fn http_completion(
    router: axum::Router,
    model: String,
    prompt: String,
    stream: bool,
) -> (String, String) {
    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "stream_options": {"include_usage": true},
        "temperature": 0.0, "ignore_eos": true, "max_tokens": 12
    });
    let response = router
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
    assert_eq!(response.status(), axum::http::StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    if !stream {
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["model"], model);
        assert_eq!(v["usage"]["completion_tokens"], 12);
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        return (
            v["id"].as_str().unwrap().into(),
            v["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .into(),
        );
    }
    let text = std::str::from_utf8(&bytes).unwrap();
    assert_eq!(text.matches("data: [DONE]").count(), 1);
    let frames: Vec<serde_json::Value> = text
        .lines()
        .filter_map(|s| s.strip_prefix("data: "))
        .filter(|s| *s != "[DONE]")
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let id = frames[0]["id"].as_str().unwrap();
    let mut content = String::new();
    let mut finished = 0;
    for v in &frames {
        assert_eq!(v["id"], id);
        assert_eq!(v["model"], model);
        if let Some(delta) = v["choices"][0]["delta"]["content"].as_str() {
            content.push_str(delta);
        }
        if let Some(reason) = v["choices"][0]["finish_reason"].as_str() {
            assert_eq!(reason, "length");
            finished += 1;
        }
    }
    assert_eq!(finished, 1);
    assert_eq!(frames.last().unwrap()["usage"]["completion_tokens"], 12);
    (id.into(), content)
}

#[tokio::test]
async fn mixed_streaming_and_buffered_http_responses_keep_model_ids_and_content() {
    let (state, muxes) = setup("http");
    let slugs = state.registry.slugs();
    let router = plowrt::serve::app(state);
    let mut expected = Vec::new();
    for i in 0..8 {
        expected.push(
            http_completion(
                router.clone(),
                slugs[i % 2].clone(),
                format!("prompt-{i}"),
                false,
            )
            .await
            .1,
        );
    }
    let mut tasks = Vec::new();
    for i in 0..8 {
        tasks.push(tokio::spawn(http_completion(
            router.clone(),
            slugs[i % 2].clone(),
            format!("prompt-{i}"),
            i % 3 != 0,
        )));
    }
    let mut ids = std::collections::HashSet::new();
    for (i, t) in tasks.into_iter().enumerate() {
        let (id, content) = t.await.unwrap();
        assert!(ids.insert(id), "request IDs reused across responses");
        assert_eq!(content, expected[i]);
    }
    for m in muxes {
        m.preempt().await;
    }
}
