//! S1 multi-model switch exercise — the serve-path proof for the multi-model
//! memory work: default-on slab chunk pool (manager credits + trims it),
//! KV block pool, bounded drain via preempt, and token identity across a
//! switch cycle. Runs against TWO real asset bundles on a possibly-shared
//! card (all VRAM accounting is delta-based, unlike `gpu_multimodel`'s
//! absolute ledger).
//!
//!   cargo run --release -p plowrt --features cuda --example s1_switch_bench \
//!       -- <assets-A> <assets-B>
//!
//! Phases:
//!   1. budget-capped manager (single residency forced), A resident
//!   2. request A (greedy Paris) — reference tokens
//!   3. switch A→B (timed; A's chunks pool, B's slab re-maps them)
//!   4. switch B→A (timed) and assert token identity with phase 2
//!   5. preempt drill: long generation live on A, PLOW_DRAIN_TIMEOUT_MS=250,
//!      switch to B — the stream must close `finish_reason: "preempted"`
//!      and the drain phase must be bounded (not O(max_tokens)).

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use plowrt::device::cuda::CudaBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::memory::vmm::VmmOps;
use plowrt::orch::Registry;
use plowrt::serve::manager::ModelManager;
use plowrt::serve::mux::MuxConfig;
use plowrt::serve::stream::{self, FinishReason, StreamChunk};
use plowrt::serve::{AppState, GenParams};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;

fn used(be: &CudaBackend) -> u64 {
    let (free, total) = be.mem_info().expect("cuMemGetInfo");
    total - free
}

/// Greedy request through the model's mux; returns (ids, text, finish).
async fn request(
    state: &Arc<AppState>,
    slug: &str,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
) -> (Vec<u32>, String, Option<FinishReason>) {
    request_opts(state, slug, prompt_ids, max_tokens, false).await
}

async fn request_opts(
    state: &Arc<AppState>,
    slug: &str,
    prompt_ids: Vec<u32>,
    max_tokens: usize,
    ignore_eos: bool,
) -> (Vec<u32>, String, Option<FinishReason>) {
    let mux = state
        .mux(slug)
        .unwrap_or_else(|| panic!("no mux for {slug}"));
    let mut gen = GenParams {
        max_tokens,
        ignore_eos,
        ..GenParams::default()
    };
    gen.params.temperature = 0.0;
    let (tx, mut rx) = stream::channel();
    mux.submit(plowrt::serve::mux::Job {
        prompt_ids,
        gen,
        arrived: Instant::now(),
        respond: tx,
    })
    .map_err(|_| ())
    .expect("submit");
    let (mut ids, mut text, mut fin) = (Vec::new(), String::new(), None);
    while let Some(chunk) = rx.recv().await {
        match chunk {
            StreamChunk::Token { id, text: delta } => {
                ids.push(id);
                text.push_str(&delta);
            }
            StreamChunk::Done { reason, .. } => {
                fin = Some(reason);
                break;
            }
            StreamChunk::Err(e) => panic!("{slug}: stream error: {e}"),
        }
    }
    (ids, text, fin)
}

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

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "plowrt=info".into()))
        .init();
    let mut args = std::env::args().skip(1);
    let dir_a = PathBuf::from(
        args.next()
            .expect("usage: s1_switch_bench <assets-A> <assets-B>"),
    );
    let dir_b = PathBuf::from(
        args.next()
            .expect("usage: s1_switch_bench <assets-A> <assets-B>"),
    );

    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let mut registry = Registry::new();
    let slug_a = registry.load(&dir_a, None).expect("load A");
    let slug_b = registry.load(&dir_b, None).expect("load B");
    assert_ne!(slug_a, slug_b, "need two distinct networks");
    let backend: Arc<dyn Backend> = Arc::clone(&be) as Arc<dyn Backend>;
    let execset = Arc::new(ExecutorSet::bringup(backend).expect("execset"));
    let state = Arc::new(AppState::with_trace(registry, execset, false));

    // Budget cap: exactly one model resident at a time, even on a big card.
    let probe = ModelManager::new(
        Arc::clone(&be),
        &state,
        MuxConfig::default(),
        vec![
            (slug_a.clone(), dir_a.clone(), dir_a.join("checkpoint")),
            (slug_b.clone(), dir_b.clone(), dir_b.join("checkpoint")),
        ],
        None,
    )
    .expect("probe manager");
    let req_a = probe.required(&slug_a).expect("req A");
    let req_b = probe.required(&slug_b).expect("req B");
    drop(probe);
    let budget = req_a.max(req_b) + used(&be) + 2 * GIB;
    eprintln!(
        "[plan] A {} MiB, B {} MiB, budget {} MiB (single residency forced)",
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
                (slug_a.clone(), dir_a.clone(), dir_a.join("checkpoint")),
                (slug_b.clone(), dir_b.clone(), dir_b.join("checkpoint")),
            ],
            Some(budget),
        )
        .expect("manager"),
    );
    state.install_manager(Arc::clone(&mgr));
    mgr.load_initial().await.expect("initial load");
    assert!(mgr.is_resident(&slug_a), "A resident at startup");
    assert!(!mgr.is_resident(&slug_b), "B must not co-reside");

    // Phase 2: reference run on A.
    let (ids_a1, text_a1, _) = request(&state, &slug_a, paris_ids(&state, &slug_a), 24).await;
    eprintln!("[A run 1] {text_a1:?}");
    assert!(
        text_a1.contains("Paris"),
        "A: expected Paris in {text_a1:?}"
    );

    // Phase 3: A -> B. A's slab chunks pool on evict; B re-maps them.
    let pool_pre = VmmOps::pool_bytes(&*be);
    let t0 = Instant::now();
    mgr.ensure_resident(&slug_b).await.expect("switch to B");
    let sw_ab = t0.elapsed().as_secs_f64();
    let rep = mgr.last_switch.lock().clone().expect("report");
    eprintln!(
        "[switch A->B] {:.2} s (evicted {:?}, load {:.0} ms); pool before switch {} MiB, after {} MiB",
        sw_ab,
        rep.evicted
            .iter()
            .map(|(s, d, u)| format!("{s} drain {d:.0} ms unload {u:.0} ms"))
            .collect::<Vec<_>>(),
        rep.load_ms,
        pool_pre / MIB,
        VmmOps::pool_bytes(&*be) / MIB,
    );
    let (_, text_b, _) = request(&state, &slug_b, paris_ids(&state, &slug_b), 24).await;
    assert!(text_b.contains("Paris"), "B: expected Paris in {text_b:?}");

    // Phase 4: B -> A, token identity.
    let t0 = Instant::now();
    mgr.ensure_resident(&slug_a).await.expect("switch to A");
    let sw_ba = t0.elapsed().as_secs_f64();
    let rep = mgr.last_switch.lock().clone().expect("report");
    eprintln!("[switch B->A] {:.2} s (load {:.0} ms)", sw_ba, rep.load_ms);
    let (ids_a2, text_a2, _) = request(&state, &slug_a, paris_ids(&state, &slug_a), 24).await;
    assert_eq!(
        ids_a1, ids_a2,
        "A not token-identical after switch cycle: {text_a1:?} vs {text_a2:?}"
    );
    eprintln!("[identity] A tokens identical after A->B->A");

    // Phase 5: preempt drill. A long generation is live on A; the switch to B
    // gives it 250 ms then preempts. Total drain must be bounded (seconds,
    // not the ~minutes a 512-token greedy run would take).
    std::env::set_var("PLOW_DRAIN_TIMEOUT_MS", "250");
    let state2 = Arc::clone(&state);
    let slug_a2 = slug_a.clone();
    let long = tokio::spawn(async move {
        // ignore_eos: the generation MUST still be live when the switch
        // lands, whatever the model wants to say.
        let ids = paris_ids(&state2, &slug_a2);
        request_opts(&state2, &slug_a2, ids, 512, true).await
    });
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
    let t0 = Instant::now();
    mgr.ensure_resident(&slug_b).await.expect("preempt switch");
    let sw_pre = t0.elapsed().as_secs_f64();
    let (ids_long, _, fin) = long.await.expect("long request task");
    let rep = mgr.last_switch.lock().clone().expect("report");
    let drain_ms = rep.evicted.first().map(|(_, d, _)| *d).unwrap_or(f64::NAN);
    eprintln!(
        "[preempt] switch {:.2} s, drain {:.0} ms, long stream got {} tokens, finish {:?}",
        sw_pre,
        drain_ms,
        ids_long.len(),
        fin
    );
    assert!(
        matches!(fin, Some(FinishReason::Preempted)),
        "long stream must close as preempted, got {fin:?}"
    );
    assert!(
        drain_ms < 5_000.0,
        "drain must be bounded by the timeout, took {drain_ms:.0} ms"
    );
    assert!(
        !ids_long.is_empty(),
        "preempted stream should carry the tokens generated so far"
    );
    eprintln!("s1_switch_bench: ALL PHASES OK");
}
