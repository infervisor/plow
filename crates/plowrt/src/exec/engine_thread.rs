//! Dedicated engine/submission thread (stage 1): every GPU tick for one model
//! runs on ONE persistent OS thread
//! instead of a rotating tokio blocking-pool thread. The CUDA context is then
//! bound to that thread once (the backend's per-call `cuCtxSetCurrent` elides
//! on a context hit), the tick never pays blocking-pool dispatch, and later
//! stages have a natural home for asynchronous command submission and
//! completion draining.
//!
//! The thread owns nothing — it executes closures the dispatcher sends, so
//! engine ownership (the registry's `Arc<Mutex<GpuEngine>>`) is unchanged. A
//! panicking tick is caught and reported like `spawn_blocking`'s `JoinError`;
//! the thread survives it. Dropping the handle closes the channel and the
//! thread exits.

use std::panic::AssertUnwindSafe;

type Job = Box<dyn FnOnce() + Send>;

/// Handle to one persistent worker thread executing submitted closures FIFO.
pub struct EngineThread {
    tx: std::sync::mpsc::Sender<Job>,
}

impl EngineThread {
    /// Spawn the named worker thread. It exits when the handle drops.
    pub fn spawn(name: String) -> EngineThread {
        let (tx, rx) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name(name)
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    job();
                }
            })
            .expect("spawn engine thread");
        EngineThread { tx }
    }

    /// Run `f` on the engine thread and await its result without blocking the
    /// calling task. `Err` carries the panic/teardown description — the same
    /// contract as `spawn_blocking`'s `JoinError`, so callers handle both
    /// paths uniformly.
    pub async fn run<T, F>(&self, f: F) -> std::result::Result<T, String>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let job: Job = Box::new(move || {
            let out = std::panic::catch_unwind(AssertUnwindSafe(f));
            let _ = tx.send(out);
        });
        if self.tx.send(job).is_err() {
            return Err("engine thread exited".into());
        }
        match rx.await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(p)) => Err(panic_message(p.as_ref())),
            Err(_) => Err("engine tick dropped without a result".into()),
        }
    }
}

/// Best-effort panic payload → text (panics carry `&str` or `String`).
fn panic_message(p: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        format!("tick panicked: {s}")
    } else if let Some(s) = p.downcast_ref::<String>() {
        format!("tick panicked: {s}")
    } else {
        "tick panicked".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn runs_in_order_on_one_thread() {
        let t = EngineThread::spawn("test-engine".into());
        let first = t.run(|| std::thread::current().id()).await.unwrap();
        for i in 0..8 {
            let id = t
                .run(move || (std::thread::current().id(), i))
                .await
                .unwrap();
            assert_eq!(id.0, first, "every tick runs on the same thread");
            assert_eq!(id.1, i);
        }
    }

    #[tokio::test]
    async fn panic_is_reported_and_thread_survives() {
        let t = EngineThread::spawn("test-engine-panic".into());
        let err = t.run(|| panic!("boom")).await.unwrap_err();
        assert!(err.contains("boom"), "panic message preserved: {err}");
        // The thread is still serving.
        assert_eq!(t.run(|| 7u32).await.unwrap(), 7);
    }
}
