//! §I.5 Co-tenant scheduling — whose turn it is when models share a GPU.
//!
//! Two models resident on one device do not execute concurrently, whatever the
//! host does. The interpreter grid is `occupancy × sm_count` — the whole device
//! — and `cuLaunchCooperativeKernel` is all-blocks-co-resident-or-fail, so a
//! second model's grid is admitted only once the first VACATES. What co-tenancy
//! actually buys is (a) no switch cost, and (b) the device changing hands at
//! every launch boundary instead of every model switch. Segmented dispatch is
//! what makes (b) fine-grained: each segment is a separate admission point, so
//! the window a co-tenant can be blocked for is one segment rather than one
//! prompt.
//!
//! [`CoSched::Free`] leaves that to the driver — each engine keeps its own
//! `CU_STREAM_NON_BLOCKING` stream, host work and copies overlap the co-tenant's
//! compute, and whoever is ready when the device drains gets it. It is the
//! default and it is the fastest thing available.
//!
//! [`CoSched::Rr`] adds a per-device turn instead, handed round in arrival
//! order. It costs overlap and buys two things Free cannot give: a starvation
//! bound, and a launch order that repeats run to run — which is what makes a
//! co-residency benchmark mean anything.
//!
//! The quantum is not 1 for a measured reason: two models with different
//! dynamic shared-memory requests force an SM carveout reconfiguration on every
//! alternation (~150–300 µs, `exec/gpu.rs`). Handing a model several
//! consecutive ticks amortises that; handing it too many is just Free with
//! extra steps.

use std::sync::Arc;

use tokio::sync::{Mutex, OwnedMutexGuard};

/// How co-resident models take the device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoSched {
    /// No host-side ordering: private streams, driver admission. Default.
    #[default]
    Free,
    /// Round-robin: one model's tick at a time per device, in arrival order.
    Rr,
}

impl std::str::FromStr for CoSched {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "free" => Ok(CoSched::Free),
            "rr" | "round-robin" => Ok(CoSched::Rr),
            other => Err(format!(
                "unknown co-tenant scheduler {other:?} (expected free or rr)"
            )),
        }
    }
}

/// The turn-taking state for one device group.
///
/// Fairness comes from `tokio::sync::Mutex`, which queues waiters in the order
/// they arrived rather than letting whoever wakes first win. That is the whole
/// mechanism: with two models ticking, strict alternation falls out of FIFO
/// acquisition, and the quantum is how many ticks a holder keeps before it
/// releases and goes to the back of the queue.
#[derive(Debug)]
pub struct DeviceTurn {
    mode: CoSched,
    quantum: u32,
    turn: Arc<Mutex<()>>,
}

impl DeviceTurn {
    pub fn new(mode: CoSched, quantum: u32) -> DeviceTurn {
        DeviceTurn {
            mode,
            // A zero quantum would release the turn without ever using it and
            // spin the queue; one tick is the smallest meaningful share.
            quantum: quantum.max(1),
            turn: Arc::new(Mutex::new(())),
        }
    }

    pub fn mode(&self) -> CoSched {
        self.mode
    }

    pub fn quantum(&self) -> u32 {
        self.quantum
    }

    /// Whether a caller must hold a turn before ticking.
    pub fn ordered(&self) -> bool {
        self.mode == CoSched::Rr
    }

    /// Take the device turn, waiting behind anyone already queued.
    pub async fn acquire(&self) -> OwnedMutexGuard<()> {
        Arc::clone(&self.turn).lock_owned().await
    }
}

/// A model's hold on its group's turn, across consecutive ticks.
///
/// Kept by the dispatcher between iterations so a quantum can span ticks. It is
/// a no-op in [`CoSched::Free`], and dropping it releases the device — which is
/// why the dispatcher must drop it before parking on an empty slot table, or an
/// idle model would hold a GPU its co-tenant is waiting for.
#[derive(Default)]
pub struct Turn {
    guard: Option<OwnedMutexGuard<()>>,
    used: u32,
}

impl Turn {
    /// Ensure this model holds the turn, acquiring it if the quantum expired or
    /// it was never held. No-op when the group is unordered.
    pub async fn take(&mut self, turn: &DeviceTurn) {
        if !turn.ordered() {
            return;
        }
        if self.used >= turn.quantum() {
            self.release();
        }
        if self.guard.is_none() {
            self.guard = Some(turn.acquire().await);
            self.used = 0;
        }
        self.used += 1;
    }

    /// Give the device back now, whatever is left of the quantum.
    pub fn release(&mut self) {
        self.guard = None;
        self.used = 0;
    }

    /// Whether the turn is currently held (tests, assertions).
    pub fn held(&self) -> bool {
        self.guard.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn parses_its_modes_and_rejects_others() {
        assert_eq!("free".parse::<CoSched>().unwrap(), CoSched::Free);
        assert_eq!("rr".parse::<CoSched>().unwrap(), CoSched::Rr);
        assert_eq!("round-robin".parse::<CoSched>().unwrap(), CoSched::Rr);
        assert!("fair".parse::<CoSched>().is_err());
    }

    /// Free must never serialize anything — that is the entire difference
    /// between the two modes.
    #[tokio::test]
    async fn free_never_takes_the_turn() {
        let dt = DeviceTurn::new(CoSched::Free, 4);
        let mut a = Turn::default();
        let mut b = Turn::default();
        a.take(&dt).await;
        b.take(&dt).await;
        assert!(!a.held());
        assert!(!b.held());
    }

    #[tokio::test]
    async fn rr_holds_the_turn_and_excludes_a_co_tenant() {
        let dt = Arc::new(DeviceTurn::new(CoSched::Rr, 4));
        let mut a = Turn::default();
        a.take(&dt).await;
        assert!(a.held());

        // While A holds it, B cannot get in.
        let dt2 = Arc::clone(&dt);
        let blocked = tokio::spawn(async move {
            let mut b = Turn::default();
            b.take(&dt2).await;
            b.held()
        });
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(!blocked.is_finished(), "B entered while A held the turn");

        a.release();
        assert!(blocked.await.unwrap(), "B never got the turn after A released");
    }

    /// A quantum of N means N consecutive ticks, then the turn goes back to the
    /// queue. Without this a model with a long prefill could hold the device
    /// indefinitely, which is the starvation `rr` exists to bound.
    #[tokio::test]
    async fn a_holder_yields_after_its_quantum() {
        let dt = Arc::new(DeviceTurn::new(CoSched::Rr, 3));
        let mut a = Turn::default();
        for _ in 0..3 {
            a.take(&dt).await;
            assert!(a.held());
        }

        // A's next take must go through a fresh acquisition, so a waiter that
        // queued in the meantime is served first.
        let entered = Arc::new(AtomicU32::new(0));
        let dt2 = Arc::clone(&dt);
        let seen = Arc::clone(&entered);
        let waiter = tokio::spawn(async move {
            let mut b = Turn::default();
            b.take(&dt2).await;
            seen.fetch_add(1, Ordering::SeqCst);
            // Hold it so the assertion below is about ordering, not timing.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            b.release();
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;

        a.take(&dt).await; // quantum spent -> releases, queues behind B
        assert_eq!(
            entered.load(Ordering::SeqCst),
            1,
            "the waiting model did not get the device after the quantum expired"
        );
        waiter.await.unwrap();
    }

    /// Two models alternate rather than one running to completion — the
    /// property that makes the mode worth having.
    #[tokio::test]
    async fn two_models_interleave_under_rr() {
        let dt = Arc::new(DeviceTurn::new(CoSched::Rr, 1));
        let order = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for name in ['a', 'b'] {
            let dt = Arc::clone(&dt);
            let order = Arc::clone(&order);
            tasks.push(tokio::spawn(async move {
                let mut t = Turn::default();
                for _ in 0..4 {
                    t.take(&dt).await;
                    order.lock().push(name);
                    tokio::task::yield_now().await;
                    t.release();
                }
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        let order = order.lock();
        assert_eq!(order.len(), 8);
        // Neither model monopolised: each got its four turns, and no model ran
        // all of its ticks before the other started.
        assert_eq!(order.iter().filter(|&&c| c == 'a').count(), 4);
        assert!(
            order[..4].contains(&'a') && order[..4].contains(&'b'),
            "one model ran to completion before the other started: {order:?}"
        );
    }
}
