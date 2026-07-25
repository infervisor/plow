//! Parametric load sweep: batch × context × concurrency.
//!
//! Drives the mux with varying parameters and measures tokens/sec. Not a
//! pass/fail correctness test — produces structured output for comparison with
//! vLLM benchmarks. Run with `--nocapture` to see the table:
//!
//! ```sh
//! cargo test -p plowrt --test sweep -- --nocapture
//! ```

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use plowrt::device::cpu::CpuBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::mux::{Job, MuxConfig};
use plowrt::serve::stream::{self, StreamChunk};
use plowrt::serve::{AppState, GenParams};

mod common;

static DIR_SEQ: AtomicU32 = AtomicU32::new(0);

/// Create a temp bundle directory with a model named `slug` and given batches.
fn setup_sweep(slug: &str, batches: &[i64]) -> Arc<AppState> {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_sweep_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle_with_batches(&dir, slug, batches);

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let mut registry = Registry::new();
    registry.load(&dir, None).unwrap();
    Arc::new(AppState::new(registry, execset))
}

/// Run a sweep point: submit `concurrency` requests, each with a prompt of
/// `ctx_len` tokens generating `max_tokens` tokens. Returns aggregate tok/sec.
async fn run_sweep_point(
    state: &Arc<AppState>,
    slug: &str,
    ctx_len: usize,
    max_tokens: usize,
    concurrency: usize,
) -> f64 {
    let bundle = state.registry.get(slug).unwrap();
    let mux = plowrt::serve::mux::spawn(
        slug.into(),
        bundle,
        Arc::clone(state),
        MuxConfig::default(),
    );

    let start = Instant::now();
    let mut handles = Vec::new();

    for i in 0..concurrency {
        let mux_c = mux.clone();
        let handle = tokio::spawn(async move {
            let (tx, mut rx) = stream::channel();
            // Fake prompt: ctx_len bytes of varying content.
            let prompt_ids: Vec<u32> =
                (0..ctx_len).map(|j| ((i * 1000 + j) % 256) as u32).collect();
            let job = Job {
                prompt_ids,
                gen: GenParams {
                    max_tokens,
                    ..Default::default()
                },
                arrived: Instant::now(),
                respond: tx,
            };
            let _ = mux_c.submit(job);

            let mut tokens = 0usize;
            while let Some(chunk) = rx.recv().await {
                match chunk {
                    StreamChunk::Token { .. } => tokens += 1,
                    StreamChunk::Done { .. } | StreamChunk::Err(_) => break,
                }
            }
            tokens
        });
        handles.push(handle);
    }

    let mut total_tokens = 0usize;
    for h in handles {
        total_tokens += h.await.unwrap_or(0);
    }
    let elapsed = start.elapsed().as_secs_f64();
    if elapsed > 0.0 {
        total_tokens as f64 / elapsed
    } else {
        0.0
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sweep_matrix() {
    let batches: Vec<i64> = vec![1, 4, 8];
    let state = setup_sweep("sweep-model", &batches);

    let contexts = [32, 128, 512];
    let max_tokens_list = [16, 64];
    let concurrencies = [1, 4, 8];

    println!(
        "\n{:<8} {:<10} {:<6} {:<12}",
        "ctx", "max_tok", "conc", "tok/s"
    );
    println!("{}", "-".repeat(40));

    for &ctx in &contexts {
        for &max_tok in &max_tokens_list {
            for &conc in &concurrencies {
                let tps = run_sweep_point(&state, "sweep-model", ctx, max_tok, conc).await;
                println!("{:<8} {:<10} {:<6} {:<12.1}", ctx, max_tok, conc, tps);
            }
        }
    }
}
