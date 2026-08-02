//! S1 multi-model serving acceptance (wave-4 item 7): switch correctness
//! cycle A→B→A with token identity, VRAM back-to-plan at every state, and
//! honest switch-cost timing at REAL contexts (live KV at 4k/32k on the
//! outgoing model) — the 5e880f6 "≈0" was residency-switch only at ctx=512;
//! S1 pays the full unload + weight-upload + first-token cost, measured here.
//!
//! Gated on `PLOW_GPU_TEST=1` plus real serve assets for TWO models:
//!   PLOW_MM_ASSETS_A (default /root/gpu-assets-s6/b1  — 12B ctx132k)
//!   PLOW_MM_ASSETS_B (default /root/gpu-assets-26b/b1 — 26B ctx132k)
//! Run under a GPU lease, single-threaded (two tests share the device):
//!   gpulease multimodel cargo test -p plowrt --features cuda \
//!     --test gpu_multimodel -- --nocapture --test-threads=1
//! Skips silently otherwise.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::manager::ModelManager;
use plowrt::serve::mux::MuxConfig;
use plowrt::serve::stream::{self, StreamChunk};
use plowrt::serve::{AppState, GenParams};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn gated() -> bool {
    std::env::var("PLOW_GPU_TEST").as_deref() == Ok("1")
}

fn assets_a() -> PathBuf {
    PathBuf::from(
        std::env::var("PLOW_MM_ASSETS_A").unwrap_or_else(|_| "/root/gpu-assets-s6/b1".into()),
    )
}

fn assets_b() -> PathBuf {
    PathBuf::from(
        std::env::var("PLOW_MM_ASSETS_B").unwrap_or_else(|_| "/root/gpu-assets-26b/b1".into()),
    )
}

fn used(be: &CudaBackend) -> u64 {
    let (free, total) = be.mem_info().expect("cuMemGetInfo");
    total - free
}

/// Registry + execset + AppState over one CUDA backend, models A then B.
fn build_state(be: &Arc<CudaBackend>) -> (Arc<AppState>, String, String) {
    let mut registry = Registry::new();
    let slug_a = registry.load(assets_a(), None).expect("load A");
    let slug_b = registry.load(assets_b(), None).expect("load B");
    assert_ne!(
        slug_a, slug_b,
        "the two assets dirs must be different models"
    );
    let backend: Arc<dyn Backend> = Arc::clone(be) as Arc<dyn Backend>;
    let execset = Arc::new(ExecutorSet::bringup(backend).expect("execset"));
    (
        Arc::new(AppState::with_trace(registry, execset, false)),
        slug_a,
        slug_b,
    )
}

/// Submit one request through the model's mux (the serve path minus HTTP):
/// greedy decode, returns `(token ids, text, ttft_ms)`.
async fn request(
    state: &Arc<AppState>,
    slug: &str,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
) -> (Vec<u32>, String, f64) {
    let mux = state
        .mux(slug)
        .unwrap_or_else(|| panic!("no mux for {slug}"));
    let mut gen = GenParams {
        max_tokens,
        ..GenParams::default()
    };
    gen.params.temperature = 0.0; // greedy — token identity across reloads
    let (tx, mut rx) = stream::channel();
    let t0 = Instant::now();
    mux.submit(plowrt::serve::mux::Job {
        prompt_ids,
        gen,
        arrived: Instant::now(),
        respond: tx,
    })
    .map_err(|_| ())
    .expect("submit");

    let (mut ids, mut text, mut ttft_ms) = (Vec::new(), String::new(), f64::NAN);
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { id, text: delta } => {
                if ids.is_empty() {
                    ttft_ms = t0.elapsed().as_secs_f64() * 1000.0;
                }
                ids.push(id);
                text.push_str(&delta);
            }
            StreamChunk::Done { .. } => break,
            StreamChunk::Err(e) => panic!("{slug}: stream error: {e}"),
        }
    }
    (ids, text, ttft_ms)
}

/// The exact server chat-template Paris prompt.
fn paris_ids(state: &Arc<AppState>, slug: &str) -> Vec<u32> {
    let prompt = "<bos><|turn>user\nWhat is the capital of France? Answer in one word.<turn|>\n\
                  <|turn>model\n<|channel>thought\n<channel|>";
    state
        .registry
        .get(slug)
        .expect("bundle")
        .tokenizer()
        .encode(prompt)
}

/// A ~`n`-token priming prompt (real KV content; the values don't matter).
fn long_ids(state: &Arc<AppState>, slug: &str, n: usize) -> Vec<u32> {
    let base = state
        .registry
        .get(slug)
        .expect("bundle")
        .tokenizer()
        .encode("<bos>The quick brown fox jumps over the lazy dog. ");
    let mut ids = Vec::with_capacity(n);
    while ids.len() < n {
        ids.extend_from_slice(&base[ids.len().min(1)..]); // keep one <bos> only
        if base.len() <= 1 {
            panic!("tokenizer produced no ids");
        }
    }
    ids.truncate(n);
    ids
}

/// VRAM must sit at the resident set's plan: above the resident weights sum
/// (the load really happened), below required-sum + slack (nothing leaked).
fn assert_vram_at_plan(be: &CudaBackend, mgr: &ModelManager, residents: &[&str], what: &str) {
    let u = used(be);
    let mut lo = 0u64;
    let mut hi = 2 * GIB; // context + driver + allocator slack
    for s in residents {
        lo += mgr.plan(s).expect("plan").weights_bytes;
        hi += mgr.required(s).expect("required");
    }
    eprintln!(
        "[vram] {what}: used {} MiB (plan bounds {}..{} MiB, residents {residents:?})",
        u / MIB,
        lo / MIB,
        hi / MIB
    );
    assert!(
        u >= lo,
        "{what}: used {} MiB below resident weights {} MiB",
        u / MIB,
        lo / MIB
    );
    assert!(
        u <= hi,
        "{what}: used {} MiB above plan {} MiB — leak",
        u / MIB,
        hi / MIB
    );
}

/// Time one S1 switch: ensure_resident(target) (evict + load) then the Paris
/// first token on the target. Prints the phase breakdown from the manager.
async fn timed_switch(
    state: &Arc<AppState>,
    mgr: &Arc<ModelManager>,
    target: &str,
    label: &str,
) -> (Vec<u32>, String) {
    let t0 = Instant::now();
    mgr.ensure_resident(target).await.expect("ensure_resident");
    let ensure_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (ids, text, ttft_ms) = request(state, target, paris_ids(state, target), 24).await;
    let report = mgr.last_switch.lock().clone().expect("switch report");
    assert_eq!(report.target, target);
    eprintln!(
        "[switch {label}] -> {target}: evicted {:?}  load {:.0} ms  ensure {:.0} ms  \
         first-token {:.0} ms  TOTAL {:.0} ms",
        report
            .evicted
            .iter()
            .map(|(s, d, u)| format!("{s} (drain {d:.1} ms, unload {u:.0} ms)"))
            .collect::<Vec<_>>(),
        report.load_ms,
        ensure_ms,
        ttft_ms,
        ensure_ms + ttft_ms,
    );
    (ids, text)
}

/// Gate 5: the A→B→A correctness cycle with VRAM ledger + switch timing at
/// live-KV ctx 4k and 32k on the outgoing model. Budget-capped so the pair
/// cannot co-reside (S1 switching is forced even on 96 GB).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s1_switch_cycle_token_identity_and_timing() {
    if !gated() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + two-model assets)");
        return;
    }
    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let (state, slug_a, slug_b) = build_state(&be);

    // Probe the plans (no budget), then cap the card so only ONE model fits.
    let probe = ModelManager::new(
        Arc::clone(&be),
        &state,
        MuxConfig::default(),
        vec![
            (slug_a.clone(), assets_a(), assets_a().join("checkpoint")),
            (slug_b.clone(), assets_b(), assets_b().join("checkpoint")),
        ],
        None,
    )
    .expect("probe manager");
    let req_a = probe.required(&slug_a).expect("req A");
    let req_b = probe.required(&slug_b).expect("req B");
    drop(probe);
    let budget = req_a.max(req_b) + used(&be) + 2 * GIB;
    eprintln!(
        "[plan] A {} MiB, B {} MiB, budget {} MiB (forcing single residency)",
        req_a / MIB,
        req_b / MIB,
        budget / MIB
    );

    let mgr = Arc::new(
        ModelManager::new(
            Arc::clone(&be),
            &state,
            MuxConfig::default(),
            vec![
                (slug_a.clone(), assets_a(), assets_a().join("checkpoint")),
                (slug_b.clone(), assets_b(), assets_b().join("checkpoint")),
            ],
            Some(budget),
        )
        .expect("manager"),
    );
    state.install_manager(Arc::clone(&mgr));
    mgr.load_initial().await.expect("initial load");
    assert!(mgr.is_resident(&slug_a), "A resident at startup");
    assert!(
        !mgr.is_resident(&slug_b),
        "B must not co-reside under the budget"
    );
    assert_vram_at_plan(&be, &mgr, &[&slug_a], "A resident");

    // First run on A — the reference tokens for the identity gate.
    let (ids_a1, text_a1, _) = request(&state, &slug_a, paris_ids(&state, &slug_a), 24).await;
    eprintln!("[A run 1] {text_a1:?}");
    assert!(
        text_a1.contains("Paris"),
        "A: expected Paris in {text_a1:?}"
    );

    for &ctx in &[4096usize, 32768] {
        // Live KV at `ctx` on the outgoing model (A), then switch to B.
        let t0 = Instant::now();
        let (_, _, _) = request(&state, &slug_a, long_ids(&state, &slug_a, ctx), 4).await;
        eprintln!(
            "[prime] A at ctx {ctx}: {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        let (_, text_b) = timed_switch(&state, &mgr, &slug_b, &format!("A@{ctx}k-out")).await;
        assert!(text_b.contains("Paris"), "B: expected Paris in {text_b:?}");
        assert!(!mgr.is_resident(&slug_a), "A evicted");
        assert_vram_at_plan(&be, &mgr, &[&slug_b], &format!("B resident (ctx {ctx})"));

        // Live KV at `ctx` on the outgoing model (B), then switch back to A.
        let t0 = Instant::now();
        let (_, _, _) = request(&state, &slug_b, long_ids(&state, &slug_b, ctx), 4).await;
        eprintln!(
            "[prime] B at ctx {ctx}: {:.1} s",
            t0.elapsed().as_secs_f64()
        );
        let (ids_a2, text_a2) = timed_switch(&state, &mgr, &slug_a, &format!("B@{ctx}k-out")).await;
        assert!(!mgr.is_resident(&slug_b), "B evicted");
        assert_vram_at_plan(
            &be,
            &mgr,
            &[&slug_a],
            &format!("A resident again (ctx {ctx})"),
        );

        // Token identity: A after reload == A's first run, byte for byte.
        assert_eq!(
            ids_a1, ids_a2,
            "A not token-identical after switch-back (ctx {ctx}): {text_a1:?} vs {text_a2:?}"
        );
        eprintln!("[identity ctx {ctx}] A tokens identical after switch-back");
    }
}

/// Admission-shed regression gate: a long-prompt request through the mux at
/// the default `--slo-ms 250` must NOT shed itself or concurrent decode
/// streams (prefill ticks are excluded from the decode-service EWMA). Uses a
/// batch>1 engine so a decode stream is live WHILE the long prompt prefills.
///   PLOW_MM_ASSETS_C (default /root/gpu-assets-b4/b4 — 12B ctx8k batch 4).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn long_prefill_does_not_shed_decode_streams() {
    if !gated() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + model assets)");
        return;
    }
    let assets = PathBuf::from(
        std::env::var("PLOW_MM_ASSETS_C").unwrap_or_else(|_| "/root/gpu-assets-b4/b4".into()),
    );
    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let mut registry = Registry::new();
    let slug = registry.load(&assets, None).expect("load C");
    let backend: Arc<dyn Backend> = Arc::clone(&be) as Arc<dyn Backend>;
    let execset = Arc::new(ExecutorSet::bringup(backend).expect("execset"));
    let state = Arc::new(AppState::with_trace(registry, execset, false));
    let mgr = Arc::new(
        ModelManager::new(
            Arc::clone(&be),
            &state,
            MuxConfig::default(), // slo_ms 250 — the regression trigger
            vec![(slug.clone(), assets.clone(), assets.join("checkpoint"))],
            None,
        )
        .expect("manager"),
    );
    state.install_manager(Arc::clone(&mgr));
    mgr.load_initial().await.expect("initial load");

    // A live decode stream (long generation) + a ~6k-token prompt arriving
    // mid-decode. Both must complete; the old EWMA would shed both.
    let story = state
        .registry
        .get(&slug)
        .expect("bundle")
        .tokenizer()
        .encode(
            "<bos><|turn>user\nTell me a very long story about a fox.<turn|>\n\
             <|turn>model\n<|channel>thought\n<channel|>",
        );
    let decode = request(&state, &slug, story, 64);
    let state2 = Arc::clone(&state);
    let slug2 = slug.clone();
    let long = async move {
        // Let the decode stream get live slots first.
        tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        let ids = long_ids(&state2, &slug2, 6000);
        request(&state2, &slug2, ids, 8).await
    };
    let (dec, lng) = tokio::join!(decode, long);
    eprintln!(
        "[no-shed] decode stream: {} tokens ({:?}...)  long-prompt: {} tokens",
        dec.0.len(),
        &dec.1.chars().take(40).collect::<String>(),
        lng.0.len()
    );
    assert!(
        dec.0.len() >= 32,
        "decode stream cut short: {} tokens",
        dec.0.len()
    );
    assert!(!lng.0.is_empty(), "long-prompt request produced no tokens");
}

/// Co-residency: with no budget cap, both models load at startup if the
/// planner fits them, and interleaved requests stream from both without a
/// switch. Skips (honestly) when the pair does not fit this card.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn co_resident_pair_interleaves() {
    if !gated() {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + two-model assets)");
        return;
    }
    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let (state, slug_a, slug_b) = build_state(&be);
    let mgr = Arc::new(
        ModelManager::new(
            Arc::clone(&be),
            &state,
            MuxConfig::default(),
            vec![
                (slug_a.clone(), assets_a(), assets_a().join("checkpoint")),
                (slug_b.clone(), assets_b(), assets_b().join("checkpoint")),
            ],
            None,
        )
        .expect("manager"),
    );
    state.install_manager(Arc::clone(&mgr));
    mgr.load_initial().await.expect("initial load");
    if !(mgr.is_resident(&slug_a) && mgr.is_resident(&slug_b)) {
        eprintln!(
            "skipped: pair does not co-fit ({} + {} MiB on this card)",
            mgr.required(&slug_a).unwrap() / MIB,
            mgr.required(&slug_b).unwrap() / MIB
        );
        return;
    }
    assert_vram_at_plan(&be, &mgr, &[&slug_a, &slug_b], "both resident");

    // Concurrent requests to both resident models — no switch, both stream.
    let (ra, rb) = tokio::join!(
        request(&state, &slug_a, paris_ids(&state, &slug_a), 24),
        request(&state, &slug_b, paris_ids(&state, &slug_b), 24),
    );
    eprintln!("[co-resident] A: {:?}  B: {:?}", ra.1, rb.1);
    assert!(ra.1.contains("Paris"), "A: {:?}", ra.1);
    assert!(rb.1.contains("Paris"), "B: {:?}", rb.1);
    assert!(
        mgr.last_switch.lock().is_none(),
        "co-resident requests must not trigger a switch"
    );
}
