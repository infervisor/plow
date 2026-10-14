//! Persistent per-rank host workers (`PLOW_TP_RANK_POOL`).
//!
//! `std::thread::scope` spawns and joins one OS thread per rank on every call: ~140 us for eight
//! ranks, paid once per decode token on the TP chain fill. These workers live for the process and
//! take one job each per call.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Mutex, OnceLock};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct Pool {
    tx: Vec<Sender<Job>>,
    done: Receiver<bool>,
}

static POOL: OnceLock<Mutex<Pool>> = OnceLock::new();

const WORKERS: usize = 16;

fn pool() -> &'static Mutex<Pool> {
    POOL.get_or_init(|| {
        let (done_tx, done) = channel::<bool>();
        let tx = (0..WORKERS)
            .map(|i| {
                let (tx, rx) = channel::<Job>();
                let done_tx = done_tx.clone();
                std::thread::Builder::new()
                    .name(format!("plow-rank-{i}"))
                    .spawn(move || {
                        while let Ok(job) = rx.recv() {
                            let ok =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)).is_ok();
                            if done_tx.send(ok).is_err() {
                                break;
                            }
                        }
                    })
                    .expect("spawn rank pool worker");
                tx
            })
            .collect();
        Mutex::new(Pool { tx, done })
    })
}

pub(crate) fn enabled() -> bool {
    crate::config::RuntimeConfig::get().amd.tp_rank_pool
}

/// Run every job on its own worker and wait for all of them. `false` if any job panicked.
pub(crate) fn run<'a>(jobs: Vec<Box<dyn FnOnce() + Send + 'a>>) -> bool {
    assert!(jobs.len() <= WORKERS, "rank pool has {WORKERS} workers");
    let p = pool().lock().unwrap_or_else(|e| e.into_inner());
    let n = jobs.len();
    for (tx, job) in p.tx.iter().zip(jobs) {
        // SAFETY: every job sent here is waited for below before this function returns, so the
        // borrows it captures (lifetime 'a) outlive its execution.
        let job: Job = unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'a>, Job>(job) };
        tx.send(job).expect("rank pool worker exited");
    }
    let mut ok = true;
    for _ in 0..n {
        ok &= p.done.recv().expect("rank pool worker exited");
    }
    ok
}
