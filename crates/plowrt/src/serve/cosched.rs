//! Per-device-group scheduling for co-resident models.
//!
//! Free leaves launch admission to each backend. Round-robin holds a FIFO
//! async turn for a configured number of mux ticks. Separate groups have
//! separate turns; waiting consumes no engine submission thread.
//!
//! A tick can include a prefill chunk and bounded multistep decode. The quantum
//! counts ticks, not tokens, kernel segments, or elapsed time. Round-robin also
//! bounds cold prefill work so an entire prompt cannot hide inside one tick.
//! The default quantum of four amortizes CUDA shared-memory carveout changes.
//!
//! # Why there is nothing to overlap
//!
//! Free is not "parallel" and Rr is not "serializing something that was
//! concurrent". On CUDA the interpreter grid is `occupancy × sm_count` — the
//! whole device — and `cuLaunchCooperativeKernel` is
//! all-blocks-co-resident-or-fail, so a second model's grid is admitted only
//! once the first VACATES. Two co-resident models never execute at the same
//! time whatever the host does.
//!
//! What co-tenancy buys is therefore (a) no switch cost, and (b) a device that
//! changes hands at every launch boundary instead of every model switch.
//! Segmented dispatch is what makes (b) fine-grained: each segment is its own
//! admission point, so the window a co-tenant can be blocked for is one segment
//! rather than one prompt — and it is why bounding a tick's prefill work
//! matters at all. Free leaves the choice of who goes next to the backend; Rr
//! makes it FIFO, which costs nothing in overlap (there was none) and buys a
//! starvation bound and an order that repeats run to run.
//!
//! The quantum is not 1 for a measured reason: two models with different
//! dynamic shared-memory requests force an SM carveout reconfiguration on every
//! alternation (~150-300 us, `exec/gpu.rs`). Several consecutive ticks amortise
//! that; too many is Free with extra steps.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{oneshot, Mutex, OwnedMutexGuard};

/// How co-resident models take the device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoSched {
    /// No host-side ordering: private streams, driver admission. Default.
    #[default]
    Free,
    /// Round-robin: one model's tick at a time per device, in arrival order.
    Rr,
    /// One model's tick at a time, most urgent first ([`Urgency`]): a model holding deadline work
    /// (an ASR final, a speech stream, a prompt waiting for its first token) takes the device
    /// ahead of decode throughput, and throughput ahead of revisable partial results. A waiter
    /// gains one class per [`AGE`] it waits, so nothing starves.
    Deadline,
}

impl std::str::FromStr for CoSched {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "free" => Ok(CoSched::Free),
            "rr" | "round-robin" => Ok(CoSched::Rr),
            "deadline" => Ok(CoSched::Deadline),
            other => Err(format!(
                "unknown co-tenant scheduler {other:?} (expected free, rr or deadline)"
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
    prio: Arc<parking_lot::Mutex<PrioState>>,
}

/// How soon a model needs the device, most urgent first ([`CoSched::Deadline`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Urgency {
    /// A whole short result a user waits on (an ASR final): its every step is on the deadline.
    Final = 0,
    /// A user-facing deadline is running: a speech stream's first audio, a prompt waiting for
    /// its first token.
    Deadline = 1,
    /// Decode throughput.
    Normal = 2,
    /// Revisable work only (partial transcripts).
    Bulk = 3,
}

/// A waiter moves up one [`Urgency`] class per `AGE` waited: bulk work gets the device within
/// two of them however much deadline work keeps arriving.
pub const AGE: Duration = Duration::from_millis(100);

/// How long a holder keeps the device while a waiter of its own class is queued. Ticks differ by
/// 20x across models (a 5 ms ASR step, a 100 ms multistep LLM quantum), so the share is time.
pub const DEADLINE_QUANTUM: Duration = Duration::from_millis(20);

#[derive(Debug)]
struct Waiter {
    urgency: Urgency,
    since: Instant,
    seq: u64,
    wake: oneshot::Sender<()>,
}

impl Waiter {
    fn rank(&self, now: Instant) -> u32 {
        let aged = (now.saturating_duration_since(self.since).as_millis() / AGE.as_millis()) as u32;
        (self.urgency as u32).saturating_sub(aged)
    }
}

#[derive(Debug, Default)]
struct PrioState {
    held: bool,
    seq: u64,
    waiters: Vec<Waiter>,
}

impl PrioState {
    /// Give the device to the best-ranked waiter, or mark it free.
    fn hand_off(&mut self) {
        let now = Instant::now();
        while let Some(i) = (0..self.waiters.len()).min_by_key(|&i| (self.waiters[i].rank(now), self.waiters[i].seq)) {
            if self.waiters.swap_remove(i).wake.send(()).is_ok() {
                return;
            }
        }
        self.held = false;
    }

    /// Whether a holder of class `mine`, holding since `since`, should hand the device on.
    fn should_yield(&self, mine: Urgency, since: Instant, now: Instant) -> bool {
        let quantum_spent = now.saturating_duration_since(since) >= DEADLINE_QUANTUM;
        self.waiters.iter().any(|w| {
            let r = w.rank(now);
            r < mine as u32 || (quantum_spent && r <= mine as u32)
        })
    }
}

/// The device held under [`CoSched::Deadline`]; dropping it hands the device on.
struct PrioHold(Arc<parking_lot::Mutex<PrioState>>);

impl Drop for PrioHold {
    fn drop(&mut self) {
        self.0.lock().hand_off();
    }
}

impl PrioHold {
    async fn acquire(state: &Arc<parking_lot::Mutex<PrioState>>, urgency: Urgency) -> PrioHold {
        let (seq, rx) = {
            let mut s = state.lock();
            if !s.held && s.waiters.is_empty() {
                s.held = true;
                return PrioHold(Arc::clone(state));
            }
            let (wake, rx) = oneshot::channel();
            s.seq += 1;
            let seq = s.seq;
            s.waiters.push(Waiter { urgency, since: Instant::now(), seq, wake });
            (seq, rx)
        };
        // A cancelled wait (the dispatcher's preempt select) must not strand a hand-off: leave
        // the queue, or pass on the device if it already arrived.
        struct Pending(Arc<parking_lot::Mutex<PrioState>>, u64, bool);
        impl Drop for Pending {
            fn drop(&mut self) {
                if self.2 {
                    return;
                }
                let mut s = self.0.lock();
                match s.waiters.iter().position(|w| w.seq == self.1) {
                    Some(i) => drop(s.waiters.swap_remove(i)),
                    None => s.hand_off(),
                }
            }
        }
        let mut pending = Pending(Arc::clone(state), seq, false);
        let _ = rx.await;
        pending.2 = true;
        PrioHold(Arc::clone(state))
    }
}

impl DeviceTurn {
    pub fn new(mode: CoSched, quantum: u32) -> DeviceTurn {
        DeviceTurn {
            mode,
            // A zero quantum would release the turn without ever using it and
            // spin the queue; one tick is the smallest meaningful share.
            quantum: quantum.max(1),
            turn: Arc::new(Mutex::new(())),
            prio: Default::default(),
        }
    }

    /// A turn in `mode` with the serving quantum: consecutive ticks one model keeps the device.
    /// Not 1: models with different dynamic shared-memory requests force an SM carveout
    /// reconfiguration on every alternation (~150-300us).
    pub fn serving(mode: CoSched) -> DeviceTurn {
        DeviceTurn::new(mode, 4)
    }

    pub fn mode(&self) -> CoSched {
        self.mode
    }

    pub fn quantum(&self) -> u32 {
        self.quantum
    }

    /// Whether a caller must hold a turn before ticking.
    pub fn ordered(&self) -> bool {
        self.mode != CoSched::Free
    }

    /// Whether a model of class `mine` is outranked by a waiting co-tenant right now: it should
    /// keep its next tick short.
    pub fn outranked(&self, mine: Urgency) -> bool {
        let now = Instant::now();
        self.mode == CoSched::Deadline && self.prio.lock().waiters.iter().any(|w| w.rank(now) < mine as u32)
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
    prio: Option<PrioHold>,
    since: Option<Instant>,
    used: u32,
}

impl Turn {
    /// Ensure this model holds the turn, acquiring it if the quantum expired or
    /// it was never held. No-op when the group is unordered.
    pub async fn take(&mut self, turn: &DeviceTurn) {
        self.take_at(turn, Urgency::Normal).await
    }

    /// [`Self::take`] for a model whose most urgent work is `urgency` ([`CoSched::Deadline`]).
    pub async fn take_at(&mut self, turn: &DeviceTurn, urgency: Urgency) {
        if turn.mode == CoSched::Deadline {
            if let (Some(hold), Some(since)) = (&self.prio, self.since) {
                if !hold.0.lock().should_yield(urgency, since, Instant::now()) {
                    self.used += 1;
                    return;
                }
                self.release();
            }
            self.prio = Some(PrioHold::acquire(&turn.prio, urgency).await);
            self.since = Some(Instant::now());
            self.used = 1;
            return;
        }
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
        self.prio = None;
        self.since = None;
        self.used = 0;
    }

    /// Whether the turn is currently held (tests, assertions).
    pub fn held(&self) -> bool {
        self.guard.is_some() || self.prio.is_some()
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
        assert_eq!("deadline".parse::<CoSched>().unwrap(), CoSched::Deadline);
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
        assert!(
            blocked.await.unwrap(),
            "B never got the turn after A released"
        );
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

    /// A deadline waiter goes ahead of a decode waiter that queued first, and the holder hands the
    /// device over at its next tick because it is outranked.
    #[tokio::test]
    async fn deadline_work_goes_first() {
        let dt = Arc::new(DeviceTurn::new(CoSched::Deadline, 4));
        let mut holder = Turn::default();
        holder.take_at(&dt, Urgency::Normal).await;
        assert!(holder.held());
        let order = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for (name, urgency) in [('n', Urgency::Normal), ('d', Urgency::Deadline)] {
            let (dt, order) = (Arc::clone(&dt), Arc::clone(&order));
            tasks.push(tokio::spawn(async move {
                let mut t = Turn::default();
                t.take_at(&dt, urgency).await;
                order.lock().push(name);
                t.release();
            }));
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(dt.outranked(Urgency::Normal));
        holder.take_at(&dt, Urgency::Normal).await; // outranked: yields, then queues again
        order.lock().push('h');
        holder.release();
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(*order.lock(), vec!['d', 'n', 'h']);
    }

    /// Bulk work ages into the deadline class, so a stream of deadline ticks cannot starve it.
    #[tokio::test]
    async fn bulk_waiters_age_in() {
        let dt = Arc::new(DeviceTurn::new(CoSched::Deadline, 4));
        let mut holder = Turn::default();
        holder.take_at(&dt, Urgency::Deadline).await;
        let dt2 = Arc::clone(&dt);
        let bulk = tokio::spawn(async move {
            let mut t = Turn::default();
            t.take_at(&dt2, Urgency::Bulk).await;
        });
        let t0 = Instant::now();
        while !bulk.is_finished() {
            holder.take_at(&dt, Urgency::Deadline).await;
            tokio::time::sleep(Duration::from_millis(5)).await;
            assert!(t0.elapsed() < 3 * AGE, "bulk waiter starved");
        }
        assert!(t0.elapsed() >= AGE);
    }

    /// A waiter cancelled after the device was handed to it passes the device on.
    #[tokio::test]
    async fn a_cancelled_waiter_does_not_strand_the_device() {
        let dt = Arc::new(DeviceTurn::new(CoSched::Deadline, 4));
        let mut holder = Turn::default();
        holder.take_at(&dt, Urgency::Normal).await;
        let dt2 = Arc::clone(&dt);
        let waiter = tokio::spawn(async move {
            let mut t = Turn::default();
            t.take_at(&dt2, Urgency::Normal).await;
            std::future::pending::<()>().await;
        });
        tokio::time::sleep(Duration::from_millis(5)).await;
        waiter.abort();
        let _ = waiter.await;
        holder.release();
        let mut next = Turn::default();
        tokio::time::timeout(Duration::from_secs(1), next.take_at(&dt, Urgency::Bulk))
            .await
            .expect("the device was stranded");
    }

}
