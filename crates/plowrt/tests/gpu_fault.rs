//! Real-fault acceptance test for the typed device-fault path: launch a
//! kernel that traps (null store), and verify the chain the error-handling
//! overhaul promises — the driver status surfaces as a fatal `DeviceFault`,
//! the backend latches poisoned, every later call short-circuits with the
//! recorded fault instead of touching the driver (no log flood, no hang),
//! engine ops fail typed-and-fatal, a fresh engine load on the dead context
//! fails the same way (the S1 manager's switch path maps that to 503), and
//! engine drop stays quiet.
//!
//! One test fn, strict order: the trap poisons the process's PRIMARY context
//! (`cuDevicePrimaryCtxRetain`), so everything healthy must run before it.
//!
//! Gated on `PLOW_GPU_TEST=1`; the engine phases additionally need
//! PLOW_GPU_ASSETS (serve-assets layout). Run under a GPU lease:
//!   gpulease fault cargo test -p plowrt --features cuda --test gpu_fault -- --nocapture

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;
use plowrt::exec::gpu::GpuEngine;
use plowrt::text::tokenizer::load_tokenizer;

/// Counts WARN events from the CUDA backend. The whole point of the poison
/// latch is that a dead context produces ONE error! and no per-call warns —
/// teardown of a full engine frees hundreds of allocations, every one of
/// which fails with the sticky status.
struct CudaWarnCount(Arc<AtomicUsize>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CudaWarnCount {
    fn on_event(&self, ev: &tracing::Event<'_>, _cx: tracing_subscriber::layer::Context<'_, S>) {
        let md = ev.metadata();
        if *md.level() == tracing::Level::WARN && md.target().starts_with("plowrt::device::cuda") {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// A kernel whose only act is a store through a null pointer — the driver
/// reports `CUDA_ERROR_ILLEGAL_ADDRESS` (700) at the next sync point and the
/// context is dead. PTX so the driver JIT builds it for any device; the
/// trailing NUL is required by `cuModuleLoadData`.
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

#[test]
fn kernel_trap_poisons_the_context_and_short_circuits() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU)");
        return;
    }
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
    assert!(!be.is_poisoned());

    // Phase 1 (healthy, optional): a real engine serves before the trap, so
    // the fault assertions below are about the fault, not a broken setup.
    let assets = std::env::var("PLOW_GPU_ASSETS").map(PathBuf::from).ok();
    let mut engine = assets.as_ref().filter(|a| a.is_dir()).map(|assets| {
        let ckpt = assets.join("checkpoint");
        let tok = load_tokenizer(assets);
        let mut e = GpuEngine::load(Arc::clone(&be), assets, &ckpt).expect("engine load");
        let ids = tok.encode("<bos><|turn>user\nSay OK.<turn|>\n<|turn>model\n");
        e.begin_slot(0, ids.len() + 4).expect("begin_slot");
        let mut toks = Vec::new();
        for &id in &ids[..ids.len().min(4)] {
            e.step_slots(&[(0, id)], &mut toks).expect("healthy step");
        }
        eprintln!("phase 1: healthy engine stepped {} prompt tokens", 4);
        e
    });
    if engine.is_none() {
        eprintln!("phase 1 skipped: no PLOW_GPU_ASSETS — backend-only run");
    }

    // Phase 2: the trap. The launch is async — the fault surfaces at the
    // sync that follows.
    let m = be.module_load(TRAP_PTX).expect("trap PTX load");
    let f = be.get_function(&m, "plow_trap").expect("plow_trap symbol");
    let launched = be.launch_kernel(f, 1, 1, 0, &mut [], None);
    let fault = match launched.and_then(|()| be.synchronize()) {
        Err(e) => e,
        Ok(()) => panic!("trap kernel completed without a fault"),
    };
    eprintln!("phase 2: trap surfaced as: {fault}");
    assert!(fault.is_fatal(), "trap must classify fatal: {fault}");
    let code = fault.device_code().expect("typed fault");
    assert!(
        [700, 719].contains(&code),
        "expected ILLEGAL_ADDRESS/LAUNCH_FAILED, got {code}"
    );
    assert!(be.is_poisoned(), "fatal status must latch the backend");

    // Phase 3: short-circuit. Every entry point fails immediately with the
    // RECORDED fault (same op/code) — the driver is no longer consulted, so a
    // request flood cannot become a log flood, and sync cannot hang.
    let sc = be.synchronize().expect_err("poisoned sync must error");
    assert!(sc.is_fatal());
    assert_eq!(sc.device_code(), Some(code), "recorded fault is replayed");
    let alloc = match be.alloc(0, 4096) {
        Err(e) => e,
        Ok(_) => panic!("poisoned alloc must error"),
    };
    assert_eq!(alloc.device_code(), Some(code));

    // Phase 4 (with assets): live engine ops now fail typed-and-fatal — this
    // is what run_one_tick's note_fault sees, driving EngineHealth to Dead
    // and the 503 mapping.
    if let Some(e) = engine.as_mut() {
        let mut toks = Vec::new();
        let step = e
            .step_slots(&[(0, 1)], &mut toks)
            .expect_err("step on dead context must error");
        eprintln!("phase 4: engine step failed as: {step}");
        assert!(step.is_fatal(), "engine step fault must stay fatal: {step}");
    }

    // Phase 5 (with assets): the S1 switch path — a fresh engine load against
    // the poisoned context fails fast with the typed fatal fault (the manager
    // surfaces it as EnsureError::Load, which chat maps to 503).
    if let Some(assets) = assets.as_ref().filter(|a| a.is_dir()) {
        let ckpt = assets.join("checkpoint");
        let reload = match GpuEngine::load(Arc::clone(&be), assets, &ckpt) {
            Err(e) => e,
            Ok(_) => panic!("engine load on a poisoned context must fail"),
        };
        eprintln!("phase 5: reload on dead context failed as: {reload}");
        assert!(reload.is_fatal(), "reload fault must stay fatal: {reload}");
        assert_eq!(reload.device_code(), Some(code));
    }

    // Phase 6: drop the engine on the poisoned context — expected-to-fail
    // teardown reports at debug, and nothing panics.
    drop(engine);
    eprintln!("phase 6: engine dropped on poisoned context without panic");

    // The no-flood contract: across trap + short-circuits + full teardown,
    // the backend emitted its one poisoning `error!` and ZERO warns — every
    // expected-to-fail teardown call reported at debug.
    let warns = cuda_warns.load(Ordering::Relaxed);
    assert_eq!(
        warns, 0,
        "poisoned context must not warn per call ({warns} device::cuda WARNs)"
    );
}
