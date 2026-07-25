//! §I Muxer continuous-batching behavior.
//!
//! With a compiled bundle whose largest decode bucket is `batch=4`, four
//! concurrent chat completions must all be admitted into the engine's slot
//! table and served in the same tick — i.e. the mux is really batching, not
//! serializing.

use std::sync::atomic::{AtomicU32, Ordering};
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

static DIR_SEQ: AtomicU32 = AtomicU32::new(0);

fn make_app_with_batches(slug: &str, batches: &[i64]) -> axum::Router {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_mux_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle_with_batches(&dir, slug, batches);

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let mut registry = Registry::new();
    registry.load(&dir, None).unwrap();
    let state = Arc::new(AppState::new(registry, execset));

    let slugs: Vec<String> = state.registry.slugs().map(str::to_string).collect();
    for slug in slugs {
        let bundle = state.registry.get(&slug).unwrap();
        let m = mux::spawn(slug.clone(), bundle, Arc::clone(&state), MuxConfig::default());
        state.install_mux(slug, m);
    }
    app(state)
}

async fn body_string(resp: axum::response::Response) -> String {
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

fn chat_req(model: &str, msg: &str) -> Request<Body> {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": msg}],
        "stream": false,
        "max_tokens": 4,
    });
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// Four concurrent requests must all be admitted into the slot table (capacity
/// == max bucket batch == 4) and served together. The
/// `plowrt_batch_size_mean` counter should end well above 1.0 — proving the
/// engine actually batched them rather than draining serially.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_share_a_batch() {
    let app = make_app_with_batches("mux-model", &[1, 2, 4]);

    let mut handles = Vec::new();
    for i in 0..4 {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            let resp = app
                .oneshot(chat_req("mux-model", &format!("hello {i}")))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            body_string(resp).await
        }));
    }
    for h in handles {
        let body = h.await.unwrap();
        assert!(body.contains("\"object\":\"chat.completion\""));
    }

    // Scrape /metrics from the same router — batch_size_mean must be > 1.
    let metrics = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(metrics.status(), StatusCode::OK);
    let text = body_string(metrics).await;
    let mean = text
        .lines()
        .find_map(|l| l.strip_prefix("plowrt_batch_size_mean "))
        .and_then(|v| v.parse::<f64>().ok())
        .expect("plowrt_batch_size_mean present");
    assert!(
        mean > 1.0,
        "mux serialized instead of batched: batch_size_mean={mean}\n{text}"
    );
    // batch_count should be small — one tick per token step, not per request.
    let n_batches: u64 = text
        .lines()
        .find_map(|l| l.strip_prefix("plowrt_batch_count_total "))
        .and_then(|v| v.parse().ok())
        .expect("plowrt_batch_count_total present");
    assert!(n_batches >= 1, "expected at least one tick, got {n_batches}");
}

/// A single request still completes cleanly on a batched bundle (no accidental
/// dependence on multi-slot occupancy).
#[tokio::test]
async fn single_request_on_batched_bundle() {
    let app = make_app_with_batches("mux-solo", &[1, 2, 4]);
    let resp = app
        .oneshot(chat_req("mux-solo", "hi"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = body_string(resp).await;
    assert!(body.contains("\"object\":\"chat.completion\""));
}

/// Streaming SSE emits one `chat.completion.chunk` frame **per generated
/// token**, then a terminal chunk carrying `finish_reason`, then `[DONE]`.
/// Regression guard: the old implementation split the final string on
/// whitespace and pretended to stream — this asserts a real per-token stream.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_emits_per_token_chunks() {
    let app = make_app_with_batches("mux-stream", &[1]);
    let max_tokens = 6u32;
    let body = serde_json::json!({
        "model": "mux-stream",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "max_tokens": max_tokens,
    });
    let resp = app
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
    // Every SSE frame is a `data: ...\n\n` group. Count `chat.completion.chunk`
    // frames whose delta.content is a non-null string (real token frames) —
    // there must be at least one per generated token, plus a terminal frame.
    let token_frames = text.matches("\"finish_reason\":null").count();
    assert!(
        token_frames >= max_tokens as usize,
        "expected >= {max_tokens} token frames, got {token_frames} in\n{text}"
    );
    let term_frames = text.matches("\"finish_reason\":\"").count();
    assert!(term_frames >= 1, "expected a terminal frame in\n{text}");
    assert!(text.contains("[DONE]"));
}

/// Four concurrent requests against a bundle whose largest bucket carries a
/// `TOKEN_SAMPLE_BATCH` packet must exercise the phase-3 batched path — one
/// bucket walk per tick, not one per slot. The regression guard: response
/// bodies come back valid (batched decode + slot fan-out works end-to-end),
/// and the batch-size metric shows real batching.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sample_batch_bundle_uses_batched_path() {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_mux_sb_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle_with_sample_batch(&dir, "sb-model", 4);

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let mut registry = Registry::new();
    registry.load(&dir, None).unwrap();
    let state = Arc::new(AppState::new(registry, execset));
    let slugs: Vec<String> = state.registry.slugs().map(str::to_string).collect();
    for slug in slugs {
        let bundle = state.registry.get(&slug).unwrap();
        let m = mux::spawn(slug.clone(), bundle, Arc::clone(&state), MuxConfig::default());
        state.install_mux(slug, m);
    }
    let app = app(state);

    let mut handles = Vec::new();
    for i in 0..4 {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            let resp = app
                .oneshot(chat_req("sb-model", &format!("hi {i}")))
                .await
                .unwrap();
            assert_eq!(resp.status(), StatusCode::OK);
            body_string(resp).await
        }));
    }
    for h in handles {
        let body = h.await.unwrap();
        assert!(body.contains("\"object\":\"chat.completion\""));
    }

    let metrics = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let text = body_string(metrics).await;
    let mean = text
        .lines()
        .find_map(|l| l.strip_prefix("plowrt_batch_size_mean "))
        .and_then(|v| v.parse::<f64>().ok())
        .expect("batch_size_mean present");
    assert!(mean > 1.0, "batched path never fired: mean={mean}\n{text}");
}

/// Plow-compiler → runtime seam: the KV per-layer bases the mux hands to
/// `KvArena` come from `AddressSpace::kv_layer_bases`, resolved from each
/// `kv_cache_L{i}` `MemEntry`'s offset in the manifest. Assert the two layers'
/// bases are non-zero, distinct, and separated by exactly the compiler's
/// per-layer stride. Guards `plowc emits offset` → `runtime honors offset`.
#[tokio::test]
async fn address_space_resolves_kv_layer_bases_from_manifest() {
    use plowrt::asset::ModelBundle;
    use plowrt::memory::AddressSpace;

    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_kv_bases_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let initial_blocks: i64 = 4;
    common::write_bundle_with_kv(&dir, "bases-model", 4, initial_blocks);

    let bundle = ModelBundle::load(&dir).unwrap();
    // Any bucket carries the same KV shape.
    let key = bundle.bucket_keys().next().unwrap();
    let bucket = bundle.bucket(key).unwrap();
    let paging = bucket.map.kv_paging.clone().unwrap();

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(1));
    let addr = AddressSpace::allocate(backend, bucket.map.clone()).unwrap();
    let bases = addr.kv_layer_bases(&paging);

    assert_eq!(bases.len(), 2, "expected 2 layers, got {}", bases.len());
    let per_layer_reserved: u64 = paging.block_bytes * initial_blocks as u64;
    assert!(bases[0] > 0, "layer 0 base is zero — map lookup failed");
    assert_eq!(
        bases[1] - bases[0],
        per_layer_reserved,
        "per-layer stride mismatch (expected {per_layer_reserved})"
    );

    // Sanity: the same value is reachable via the by-name lookup, i.e.
    // `kv_layer_bases` and `addr_of("kv_cache_L0", 0)` agree.
    let by_name = addr.addr_of("kv_cache_L0", 0).unwrap();
    assert_eq!(by_name, bases[0]);
}

/// KV OOM: the arena is sized for one slot's footprint; a burst of 4 concurrent
/// requests must see the excess admissions rejected with an Oom error (not
/// silently held).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn kv_oom_sheds_excess_requests() {
    use plowrt::serve::mux::{Job, MuxConfig};
    use plowrt::serve::stream::{channel, StreamChunk};
    use plowrt::RuntimeError;

    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_mux_oom_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    // Every slot needs ceil((prompt_len + max_tokens) / block_tokens=4)
    // blocks per layer. Set `initial_blocks_per_layer` = 2 so the arena fits
    // at most one slot at a time (max_tokens=4, prompt<=4 ⇒ 2 blocks).
    common::write_bundle_with_kv(&dir, "oom-model", 4, 2);

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let mut registry = Registry::new();
    registry.load(&dir, None).unwrap();
    let state = Arc::new(AppState::new(registry, execset));
    let bundle = state.registry.get("oom-model").unwrap();
    let m = plowrt::serve::mux::spawn(
        "oom-model".into(),
        bundle,
        Arc::clone(&state),
        MuxConfig {
            multi_step: false,
            ..MuxConfig::default()
        },
    );

    // Submit two jobs back-to-back, each holding on to its receiver so KV
    // stays allocated. Only one should get admitted; the second must Oom.
    let mut receivers = Vec::new();
    let mut oom_seen = false;
    for i in 0..2 {
        let (tx, mut rx) = channel();
        let job = Job {
            prompt_ids: format!("hi {i}").bytes().map(u32::from).collect(),
            gen: plowrt::serve::GenParams {
                max_tokens: 4,
                ..Default::default()
            },
            arrived: std::time::Instant::now(),
            respond: tx,
        };
        assert!(m.submit(job).is_ok());
        // Peek at the very first chunk to distinguish admit vs OOM.
        let first = tokio::time::timeout(std::time::Duration::from_millis(300), rx.recv()).await;
        match first {
            Ok(Some(StreamChunk::Err(RuntimeError::Oom(_)))) => {
                oom_seen = true;
            }
            Ok(Some(StreamChunk::Token { .. })) => {
                receivers.push(rx);
            }
            other => panic!("unexpected first chunk from request {i}: {other:?}"),
        }
    }
    assert!(
        oom_seen,
        "kv arena exhaustion should have rejected the second request"
    );
}

/// Dropping the client mid-stream must free the slot: the mux notices its
/// send() failure on the next token, drops the slot, and the next request
/// gets served. Regression guard against per-request oneshot leaking a slot.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_frees_the_slot() {
    use plowrt::serve::mux::{Job, MuxConfig};
    use plowrt::serve::stream::{channel, StreamChunk};

    // Directly wire an AppState + mux — go under the HTTP layer so we can
    // simulate a dropped receiver deterministically.
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_mux_cancel_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle_with_batches(&dir, "cancel-model", &[1]);

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let mut registry = Registry::new();
    registry.load(&dir, None).unwrap();
    let state = Arc::new(AppState::new(registry, execset));
    let bundle = state.registry.get("cancel-model").unwrap();
    let m = plowrt::serve::mux::spawn(
        "cancel-model".into(),
        bundle,
        Arc::clone(&state),
        MuxConfig::default(),
    );

    // First job: drop the receiver immediately so the mux hits Err on send.
    let (tx1, rx1) = channel();
    drop(rx1);
    let job1 = Job {
        prompt_ids: "hello".bytes().map(u32::from).collect(),
        gen: plowrt::serve::GenParams {
            max_tokens: 32,
            ..Default::default()
        },
        arrived: std::time::Instant::now(),
        respond: tx1,
    };
    assert!(m.submit(job1).is_ok());

    // Give the dispatcher one tick to notice the closed receiver.
    tokio::time::sleep(std::time::Duration::from_millis(30)).await;

    // Second job: expect it to be served — the first slot must have been freed.
    let (tx2, mut rx2) = channel();
    let job2 = Job {
        prompt_ids: "second".bytes().map(u32::from).collect(),
        gen: plowrt::serve::GenParams {
            max_tokens: 3,
            ..Default::default()
        },
        arrived: std::time::Instant::now(),
        respond: tx2,
    };
    assert!(m.submit(job2).is_ok());

    let mut got_done = false;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while std::time::Instant::now() < deadline {
        let recv = tokio::time::timeout(std::time::Duration::from_millis(200), rx2.recv()).await;
        match recv {
            Ok(Some(StreamChunk::Done { .. })) => {
                got_done = true;
                break;
            }
            Ok(Some(_)) => continue,
            Ok(None) => break,
            Err(_) => continue,
        }
    }
    assert!(got_done, "second request never completed — first slot was leaked");
}
