//! §I.5 stop-and-flush: what in-flight work sees when a model is preempted.
//!
//! This is the mechanism an operator `unload` rides on, so the contract is
//! worth pinning:
//!
//! * a live generation is TERMINATED, not left hanging, and it ends with
//!   `Preempted` plus the tokens produced so far — a stream that stops with no
//!   terminal chunk is indistinguishable from a crash;
//! * a job that was queued but never admitted is answered too, with a
//!   retryable rejection rather than being dropped;
//! * the wire value stays inside the OpenAI `finish_reason` vocabulary while
//!   the real cause stays reportable.
//!
//! Note what is NOT asserted here: a client that stops reading altogether has
//! its slot freed by the token path itself (`mux.rs` treats a full channel as a
//! gone consumer, deliberately and with a comment), so it never survives to be
//! preempted. That is pre-existing behaviour, unrelated to unload.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use plowrt::device::cpu::CpuBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::mux::{self, Job, MuxConfig};
use plowrt::serve::stream::{channel, ChunkReceiver, ChunkSender, FinishReason, StreamChunk};
use plowrt::serve::{AppState, GenParams};

mod common;

static DIR_SEQ: AtomicU32 = AtomicU32::new(0);

fn spawn_mux(slug: &str) -> (Arc<AppState>, mux::ModelMux) {
    let n = DIR_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("plowrt_preempt_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    common::write_bundle_with_batches(&dir, slug, &[1]);

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());
    let registry = Registry::new();
    registry.load(&dir, None).unwrap();
    let state = Arc::new(AppState::new(registry, execset));
    let bundle = state.registry.get(slug).unwrap();
    let m = mux::spawn(
        slug.into(),
        bundle,
        Arc::clone(&state),
        MuxConfig::default(),
    );
    (state, m)
}

fn job(respond: ChunkSender, max_tokens: usize) -> Job {
    Job {
        prompt_ids: "hello".bytes().map(u32::from).collect(),
        gen: GenParams {
            max_tokens,
            // Run to the cap rather than stopping at the reference path's
            // newline heuristic, so the generation is reliably still live when
            // the preempt lands.
            ignore_eos: true,
            ..Default::default()
        },
        arrived: std::time::Instant::now(),
        respond,
    }
}

/// Read to the terminal event, consuming tokens as they arrive (a well-behaved
/// client). `None` means the channel closed with no terminal at all.
async fn terminal(rx: &mut ChunkReceiver) -> Option<StreamChunk> {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
            Ok(Some(StreamChunk::Token { .. })) => continue,
            Ok(Some(other)) => return Some(other),
            Ok(None) => return None,
            Err(_) => continue,
        }
    }
    None
}

#[tokio::test]
async fn a_live_generation_is_closed_as_preempted_with_its_partial_output() {
    let (_state, m) = spawn_mux("preempt-live");
    let (tx, mut rx) = channel();
    assert!(m.submit(job(tx, 100_000)).is_ok());

    // Preempt from a second task while this one keeps draining, which is what
    // a real client does. Preempting before the slot is admitted would pass
    // without exercising the partial-output path at all.
    let pre = m.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(80)).await;
        pre.preempt().await;
    });

    match terminal(&mut rx).await {
        Some(StreamChunk::Done { reason, usage, .. }) => {
            assert!(
                matches!(reason, FinishReason::Preempted),
                "expected Preempted, got {reason:?}"
            );
            assert!(
                usage.prompt_tokens > 0,
                "usage must report the prompt that was consumed"
            );
        }
        Some(other) => panic!("expected a Done terminal, got {other:?}"),
        None => {
            panic!("stream ended with NO terminal chunk — a client cannot tell this from a crash")
        }
    }
}

/// A job that was accepted but never got a slot must be answered, not dropped.
/// The preempt flag bypasses channel order, so unlike a message-initiated drain
/// these jobs never had their chance to be admitted.
#[tokio::test]
async fn a_queued_job_is_rejected_rather_than_dropped() {
    let (_state, m) = spawn_mux("preempt-queued");

    // Capacity is one slot, so the first job occupies the table and the second
    // waits in the channel. Drain the first from its own task so it is not
    // killed by backpressure before the preempt lands.
    let (tx1, mut rx1) = channel();
    assert!(m.submit(job(tx1, 100_000)).is_ok());
    tokio::spawn(async move { while rx1.recv().await.is_some() {} });
    tokio::time::sleep(Duration::from_millis(60)).await;

    let (tx2, mut rx2) = channel();
    assert!(m.submit(job(tx2, 32)).is_ok());

    m.preempt().await;

    match terminal(&mut rx2).await {
        // Either terminal is acceptable — what must not happen is a silent end.
        Some(StreamChunk::Err(_)) | Some(StreamChunk::Done { .. }) => {}
        Some(StreamChunk::Token { .. }) => unreachable!("terminal() consumes tokens"),
        None => panic!("a queued job's stream ended with no terminal chunk"),
    }
}

/// The wire value must stay inside the OpenAI `finish_reason` vocabulary — a
/// client typed against it rejects the whole response otherwise — while the
/// true cause stays reportable in `x_plow_finish_reason`.
#[test]
fn preempted_widens_to_length_on_the_wire_but_keeps_its_real_cause() {
    assert_eq!(FinishReason::Preempted.as_openai(), "length");
    assert_eq!(FinishReason::Preempted.as_str(), "preempted");
    assert!(FinishReason::Preempted.is_vendor_specific());
    // An ordinary length stop must NOT claim a vendor cause, or a client would
    // see "preempted" every time a generation simply hit max_tokens.
    assert!(!FinishReason::Length.is_vendor_specific());
    assert!(!FinishReason::Stop.is_vendor_specific());
}
