//! Serve-path acceptance for a REAL device fault: the production symptom is a
//! served request answering `{"error":"device error: cuStreamSynchronize:
//! CUDA_ERROR_LAUNCH_FAILED","partial":""}` forever, one identical failure
//! per request (observed live: Xid 43, old build). This test drives the full
//! serve stack minus HTTP — manager, mux, dispatcher, stream — through a real
//! kernel trap and asserts the overhauled behavior: the faulted request gets
//! the TYPED fatal fault (which `status_for` maps to 503 — unit-tested
//! in-module), the dispatcher goes Dead, the next request is rejected up
//! front with the same fault, and the backend logs zero per-call warns.
//!
//! Gated on `PLOW_GPU_TEST=1` + PLOW_GPU_ASSETS. The trap poisons the primary
//! context for the whole process — this file must stay single-test.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::manager::ModelManager;
use plowrt::serve::mux::MuxConfig;
use plowrt::serve::stream::{self, StreamChunk};
use plowrt::serve::{AppState, GenParams};

/// Same trap kernel as `gpu_fault.rs`: a store through null — the driver
/// reports a fatal status at the next sync and the context is dead.
const TRAP_PTX: &[u8] = b".version 7.0
.target sm_70
.address_size 64

.visible .entry plow_trap()
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    mov.u64 %rd1, 0;
    mov.u32 %r1, 1;
    st.global.u32 [%rd1], %r1;
    ret;
}
\0";

/// WARN counter over the CUDA backend target (see `gpu_fault.rs`): the
/// contract is one poisoning `error!`, zero per-call warns.
struct CudaWarnCount(Arc<AtomicUsize>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CudaWarnCount {
    fn on_event(&self, ev: &tracing::Event<'_>, _cx: tracing_subscriber::layer::Context<'_, S>) {
        let md = ev.metadata();
        if *md.level() == tracing::Level::WARN && md.target().starts_with("plowrt::device::cuda") {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Run one greedy request through the mux; return (text, first stream error).
async fn request(state: &Arc<AppState>, slug: &str, ids: Vec<u32>) -> (String, Option<String>) {
    let mux = state.mux(slug).expect("mux");
    let gen = GenParams {
        max_tokens: 16,
        ..GenParams::default()
    };
    let (tx, mut rx) = stream::channel();
    mux.submit(plowrt::serve::mux::Job {
        prompt_ids: ids,
        gen,
        arrived: std::time::Instant::now(),
        respond: tx,
    })
    .map_err(|_| ())
    .expect("submit");
    let mut text = String::new();
    let mut err = None;
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { text: d, .. } => text.push_str(&d),
            StreamChunk::Done { .. } => break,
            StreamChunk::Err(e) => {
                // The typed-fault contract the HTTP layer depends on: fatal
                // device faults must arrive as DeviceFault (503 via
                // status_for), never stringified.
                assert!(
                    e.is_fatal(),
                    "post-trap stream error must be a fatal DeviceFault, got: {e}"
                );
                err = Some(e.to_string());
                break;
            }
        }
    }
    (text, err)
}

#[tokio::test]
async fn device_fault_kills_engine_and_rejects_typed() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + assets)");
        return;
    }
    let assets = PathBuf::from(std::env::var("PLOW_GPU_ASSETS").expect("set PLOW_GPU_ASSETS"));
    assert!(assets.is_dir(), "assets dir {} missing", assets.display());

    let cuda_warns = Arc::new(AtomicUsize::new(0));
    {
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .with(CudaWarnCount(Arc::clone(&cuda_warns)))
            .with(tracing_subscriber::fmt::layer())
            .try_init()
            .ok();
    }

    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let mut registry = Registry::new();
    let slug = registry.load(assets.clone(), None).expect("load");
    let backend: Arc<dyn Backend> = Arc::clone(&be) as Arc<dyn Backend>;
    let execset = Arc::new(ExecutorSet::bringup(backend).expect("execset"));
    let state = Arc::new(AppState::with_trace(registry, execset, false));
    let mgr = Arc::new(
        ModelManager::new(
            Arc::clone(&be),
            &state,
            MuxConfig::default(),
            vec![(slug.clone(), assets.clone(), assets.join("checkpoint"))],
            None,
        )
        .expect("manager"),
    );
    mgr.ensure_resident(&slug).await.expect("ensure_resident");

    let ids = state
        .registry
        .get(&slug)
        .expect("bundle")
        .tokenizer()
        .encode(
            "<bos><|turn>user\nWhat is the capital of France? Answer in one word.<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>",
        );

    // Phase 1: healthy serve.
    let (reply, err) = request(&state, &slug, ids.clone()).await;
    eprintln!("phase 1 (healthy): {reply:?}");
    assert!(err.is_none(), "healthy request errored: {err:?}");
    assert!(reply.contains("Paris"), "expected Paris, got {reply:?}");

    // Phase 2: the trap — same process, same primary context as the engine.
    let m = be.module_load(TRAP_PTX).expect("trap PTX load");
    let f = be.get_function(&m, "plow_trap").expect("plow_trap symbol");
    be.launch_kernel(f, 1, 1, 0, &mut [], None).expect("launch");
    let fault = be.synchronize().expect_err("trap must fault");
    eprintln!("phase 2: trap surfaced as: {fault}");
    assert!(fault.is_fatal() && be.is_poisoned());

    // Phase 3: the request in flight against the dead context gets the typed
    // fatal fault (503 at the HTTP layer) — not an eternal per-request retry.
    let (_, err) = request(&state, &slug, ids.clone()).await;
    let err = err.expect("post-trap request must error");
    eprintln!("phase 3: faulted request answered: {err}");
    assert!(err.contains("device fault:"), "typed display: {err}");

    // Phase 4: the dispatcher is Dead — the next request is rejected up
    // front with the same fault (admission gate, no dispatch attempted).
    let (_, err) = request(&state, &slug, ids).await;
    let err = err.expect("request to a dead engine must error");
    eprintln!("phase 4: dead-engine request rejected: {err}");
    assert!(err.contains("device fault:"), "typed display: {err}");

    // The no-flood contract across serve + fault + rejections.
    let warns = cuda_warns.load(Ordering::Relaxed);
    assert_eq!(
        warns, 0,
        "{warns} device::cuda WARNs — poisoned path must not flood"
    );
}
