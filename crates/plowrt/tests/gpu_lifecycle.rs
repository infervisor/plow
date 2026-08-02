//! Device-memory lifecycle acceptance test (leak-audit D1/D2 closure):
//! load a real engine, serve a request, DROP it in-process, verify VRAM
//! returns to baseline, then LOAD AGAIN and serve again — the reload proves
//! no double-free (the d_ctr/d_gq_cursor aliased views) and no leaked
//! module/allocation/context from the first life.
//!
//! Gated on `PLOW_GPU_TEST=1` plus real model assets:
//!   PLOW_GPU_ASSETS (default /root/gpu-assets-b4/b4) — pkt + cubins +
//!   tokenizer.json + checkpoint/ (the serve-assets layout).
//! Run under a GPU lease: `gpulease devmem cargo test -p plowrt --features
//! cuda --test gpu_lifecycle -- --nocapture`. Skips silently otherwise.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Arc;

use plowrt::device::cuda::CudaBackend;
use plowrt::exec::gpu::GpuEngine;
use plowrt::text::tokenizer::{load_tokenizer, Tokenize};

const MIB: u64 = 1 << 20;
/// Allowed VRAM drift after unload (driver-internal caches, allocator
/// metadata). The engine itself is ~15 GiB — leaks show up far above this.
const TOLERANCE: u64 = 64 * MIB;

/// One served request: the exact chat-template prompt the server builds
/// (`serve::chat::gemma_chat_prompt`), greedy decode, stop on the
/// checkpoint's eos set.
fn serve_paris(e: &mut GpuEngine, tok: &Arc<dyn Tokenize>) -> String {
    let prompt = "<bos><|turn>user\nWhat is the capital of France? Answer in one word.<turn|>\n\
                  <|turn>model\n<|channel>thought\n<channel|>";
    let ids = tok.encode(prompt);
    assert!(!ids.is_empty(), "tokenizer produced no ids");
    let max_new = 24usize;
    e.begin_slot(0, ids.len() + max_new).expect("begin_slot");

    let mut toks = Vec::new();
    let mut t = if e.has_prefill() {
        e.prefill_slot(0, &ids).expect("prefill_slot")
    } else {
        let mut t = 0u32;
        for &id in &ids {
            e.step_slots(&[(0, id)], &mut toks).expect("step (prompt)");
            t = toks[0];
        }
        t
    };
    let stop = Arc::clone(e.stop_ids());
    let mut out = Vec::new();
    for _ in 0..max_new {
        if stop.contains(&t) {
            break;
        }
        out.push(t);
        e.step_slots(&[(0, t)], &mut toks).expect("step (gen)");
        t = toks[0];
    }
    tok.decode(&out)
}

/// Device memory in use, from the driver's own ledger (`cuMemGetInfo`).
fn used(be: &CudaBackend) -> u64 {
    let (free, total) = be.mem_info().expect("cuMemGetInfo");
    total - free
}

#[test]
fn load_serve_unload_reload_cycle() {
    if std::env::var("PLOW_GPU_TEST").as_deref() != Ok("1") {
        eprintln!("skipped: set PLOW_GPU_TEST=1 (needs GPU + model assets)");
        return;
    }
    // Surface the engine's load/step tracing under --nocapture (which cubin
    // features armed, batch width, step timing) — silent otherwise.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .try_init()
        .ok();
    let assets = PathBuf::from(
        std::env::var("PLOW_GPU_ASSETS").unwrap_or_else(|_| "/root/gpu-assets-b4/b4".into()),
    );
    assert!(
        assets.is_dir(),
        "assets dir {} missing (set PLOW_GPU_ASSETS)",
        assets.display()
    );
    let ckpt = assets.join("checkpoint");
    let tok = load_tokenizer(&assets);

    let be = Arc::new(CudaBackend::new(0).expect("CUDA backend"));
    let baseline = used(&be);
    eprintln!("VRAM baseline (ctx retained): {} MiB", baseline / MIB);

    let mut loaded = [0u64; 2];
    let mut after = [0u64; 2];
    for cycle in 0..2usize {
        let mut e = GpuEngine::load(Arc::clone(&be), &assets, &ckpt).expect("engine load");
        loaded[cycle] = used(&be);
        let reply = serve_paris(&mut e, &tok);
        eprintln!(
            "cycle {cycle}: loaded {} MiB (+{} MiB), reply: {reply:?}",
            loaded[cycle] / MIB,
            (loaded[cycle] - baseline) / MIB
        );
        assert!(
            reply.contains("Paris"),
            "cycle {cycle}: expected \"Paris\" in reply {reply:?}"
        );

        drop(e); // <- the unload under test: modules + every owned DeviceMem
        after[cycle] = used(&be);
        eprintln!(
            "cycle {cycle}: after unload {} MiB (baseline {} MiB)",
            after[cycle] / MIB,
            baseline / MIB
        );
        assert!(
            after[cycle] <= baseline + TOLERANCE,
            "cycle {cycle}: VRAM did not return to baseline: {} MiB used vs \
             baseline {} MiB (+{} MiB tolerance)",
            after[cycle] / MIB,
            baseline / MIB,
            TOLERANCE / MIB
        );
        // The load itself must be substantial or `used` isn't measuring the
        // engine (a 12B fp8/bf16 engine is many GiB).
        assert!(
            loaded[cycle] - baseline > 1024 * MIB,
            "engine footprint implausibly small: +{} MiB",
            (loaded[cycle] - baseline) / MIB
        );
    }
    eprintln!(
        "VRAM curve (MiB): baseline {} -> load {} -> unload {} -> reload {} -> unload {}",
        baseline / MIB,
        loaded[0] / MIB,
        after[0] / MIB,
        loaded[1] / MIB,
        after[1] / MIB
    );
}
