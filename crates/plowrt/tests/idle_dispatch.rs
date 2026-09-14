//! Idle dispatch: a cold-start hold ends once the burst it waits for has arrived, instead of
//! running to its deadline.

use std::sync::Arc;
use std::time::{Duration, Instant};

use plowrt::device::cpu::CpuBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::mux::{self, Job, MuxConfig};
use plowrt::serve::stream::{channel, ChunkSender, StreamChunk};
use plowrt::serve::AppState;

mod common;

fn job(respond: ChunkSender) -> Job {
    Job {
        prompt_ids: "hello".bytes().map(u32::from).collect(),
        gen: plowrt::serve::GenParams {
            max_tokens: 2,
            ..Default::default()
        },
        arrived: Instant::now(),
        respond,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_burst_of_three_ends_the_hold_when_the_third_arrives() {
    let dir = std::env::temp_dir().join(format!("plowrt_idle_dispatch_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle_with_batches(&dir, "idle-model", &[1, 20]);
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let registry = Registry::new();
    registry.load(&dir, None).unwrap();
    let state = Arc::new(AppState::new(registry, Arc::new(ExecutorSet::bringup(backend).unwrap())));
    let m = mux::spawn(
        "idle-model".into(),
        state.registry.get("idle-model").unwrap(),
        Arc::clone(&state),
        MuxConfig {
            max_hold_ms: 2000.0,
            idle_dispatch: true,
            ..MuxConfig::default()
        },
    );

    // Two peers still tokenizing when the first job reaches the dispatcher: it must hold.
    let (g2, g3) = (m.ingress(), m.ingress());
    let t0 = Instant::now();
    let (tx1, mut rx1) = channel();
    assert!(m.submit(job(tx1)).is_ok());
    tokio::time::sleep(Duration::from_millis(20)).await;
    let (tx2, _rx2) = channel();
    assert!(m.submit_arrived(job(tx2), Instant::now(), Some(g2)).is_ok());
    tokio::time::sleep(Duration::from_millis(30)).await;
    let (tx3, _rx3) = channel();
    let third = t0.elapsed();
    assert!(m.submit_arrived(job(tx3), Instant::now(), Some(g3)).is_ok());

    let first = loop {
        match tokio::time::timeout(Duration::from_secs(5), rx1.recv()).await {
            Ok(Some(StreamChunk::Err(e))) => panic!("job 1 failed: {e}"),
            Ok(Some(_)) => break t0.elapsed(),
            other => panic!("job 1 produced nothing: {:?}", other.is_err()),
        }
    };
    assert!(first >= third, "dispatched before the burst arrived: {first:?} < {third:?}");
    assert!(first < Duration::from_millis(1000), "held to the 2000 ms deadline: {first:?}");
    std::fs::remove_dir_all(&dir).ok();
}
