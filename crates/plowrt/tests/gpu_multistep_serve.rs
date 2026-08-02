//! Bounded multi-step through the full serve path (plan stage 5, mux wiring).
//! Loads one GPU model with `PLOW_MULTISTEP=8`, submits a greedy request
//! through the mux (the serve path minus HTTP), and asserts the reply is
//! coherent ("Paris") — proving the mux's multi-step branch (K tokens/tick,
//! mid-quantum stop via handle_produced_token) streams correct greedy output.
//! Engine-level token identity is covered separately by `gpu_multistep`.
//!
//! Gated on `PLOW_GPU_TEST=1` + a serve-asset dir (`PLOW_GPU_ASSETS`, default
//! the b4 dir). Skips silently otherwise.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::manager::ModelManager;
use plowrt::serve::mux::MuxConfig;
use plowrt::serve::stream::{self, StreamChunk};
use plowrt::serve::{AppState, GenParams};

async fn greedy_reply(state: &Arc<AppState>, slug: &str, prompt_ids: Vec<u32>) -> String {
    let mux = state.mux(slug).expect("mux");
    let gen = GenParams {
        max_tokens: 24,
        ..GenParams::default()
    }; // temperature 0 = greedy
    let (tx, mut rx) = stream::channel();
    mux.submit(plowrt::serve::mux::Job {
        prompt_ids,
        gen,
        arrived: std::time::Instant::now(),
        respond: tx,
    })
    .map_err(|_| ())
    .expect("submit");
    let mut text = String::new();
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { text: d, .. } => text.push_str(&d),
            StreamChunk::Done { .. } => break,
            StreamChunk::Err(e) => panic!("stream error: {e}"),
        }
    }
    text
}

#[tokio::test]
async fn multi_step_serve_greedy_paris() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + assets)");
        return;
    }
    let assets = PathBuf::from(
        std::env::var("PLOW_GPU_ASSETS").unwrap_or_else(|_| "/root/gpu-assets-b4/b4".into()),
    );
    assert!(assets.is_dir(), "assets dir {} missing", assets.display());
    // Multi-step must be armed at engine load (needs the sampler cubin for
    // plow_advance); the caller sets PLOW_MULTISTEP + PLOW_NV_CUBIN_SAMPLE.
    assert_eq!(
        std::env::var("PLOW_MULTISTEP").ok().as_deref(),
        Some("8"),
        "run with PLOW_MULTISTEP=8"
    );

    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let mut registry = Registry::new();
    let slug = registry.load(assets.clone(), None).expect("load");
    let backend: Arc<dyn Backend> = Arc::clone(&be) as Arc<dyn Backend>;
    let execset = Arc::new(ExecutorSet::bringup(backend).expect("execset"));
    let state = Arc::new(AppState::with_trace(registry, execset, false));

    let mgr = ModelManager::new(
        Arc::clone(&be),
        &state,
        MuxConfig::default(),
        vec![(slug.clone(), assets.clone(), assets.join("checkpoint"))],
        None,
    )
    .expect("manager");
    mgr.ensure_resident(&slug).await.expect("ensure_resident");

    let prompt = "<bos><|turn>user\nWhat is the capital of France? Answer in one word.<turn|>\n\
                  <|turn>model\n<|channel>thought\n<channel|>";
    let ids = state
        .registry
        .get(&slug)
        .expect("bundle")
        .tokenizer()
        .encode(prompt);
    let reply = greedy_reply(&state, &slug, ids).await;
    eprintln!("multi-step serve reply: {reply:?}");
    assert!(
        reply.contains("Paris"),
        "expected Paris in multi-step serve reply {reply:?}"
    );
}
