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
    /// One model's tick at a time by [`Due::rank`]: first outputs (an ASR final, a prompt waiting
    /// for its first token, a speech stream's first audio) ahead of decode and stream windows,
    /// ahead of partials; least slack within a band, and work about to miss (a stream about to
    /// underrun) ahead of all. A waiter is due by [`MAX_WAIT`] after it queued, so nothing starves.
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

/// The starvation bound: a waiter queued this long is due now whatever its own deadline
/// (`PLOW_COSCHED_MAX_WAIT_MS` overrides).
pub const MAX_WAIT: Duration = Duration::from_millis(2000);

pub(crate) fn max_wait() -> Duration {
    static WAIT: std::sync::OnceLock<Duration> = std::sync::OnceLock::new();
    *WAIT.get_or_init(|| {
        crate::config::RuntimeConfig::get()
            .cosched_max_wait_ms
            .map_or(MAX_WAIT, |ms| Duration::from_millis(ms.max(1) as u64))
    })
}

/// How long a holder keeps the device while a waiter of the same slack is queued. Ticks differ by
/// 20x across models (a 5 ms ASR step, a 100 ms multistep LLM quantum), so the share is time.
pub const DEADLINE_QUANTUM: Duration = Duration::from_millis(20);

/// Slacks closer than this are a tie (shared by [`DEADLINE_QUANTUM`]); a holder hands over only to
/// a waiter tighter by more, so near-equal deadlines don't swap the device every tick.
const YIELD_MARGIN: i64 = DEADLINE_QUANTUM.as_nanos() as i64;

/// Priority band of a [`Due`], most urgent first. Least slack wins only within a band: by slack
/// alone a decode tick (`last token + TBT`) always beats a first output due hundreds of ms out,
/// which meets first-output deadlines late instead of minimizing them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Band {
    /// About to miss ([`URGENT_SLACK`]) or at the starvation bound; assigned by [`Due::rank`].
    Urgent = 0,
    /// An ASR final: a few ms of device work upstream of every other stage of its turn.
    Final = 1,
    /// A first output on a turn's critical path: LLM first token, TTS first audio.
    First = 2,
    /// Decode and stream windows.
    Stream = 3,
    /// Revisable work (partial transcripts).
    Bulk = 4,
}

impl From<Urgency> for Band {
    fn from(u: Urgency) -> Band {
        match u {
            Urgency::Final => Band::Final,
            Urgency::Deadline => Band::First,
            Urgency::Normal => Band::Stream,
            Urgency::Bulk => Band::Bulk,
        }
    }
}

/// Work with at most this slack, or twice its own cost, ranks [`Band::Urgent`].
pub const URGENT_SLACK: Duration = Duration::from_millis(30);

/// When work must be done and what it will cost on the device ([`CoSched::Deadline`]).
#[derive(Clone, Copy, Debug)]
pub struct Due {
    pub deadline: Instant,
    pub cost: Duration,
    pub band: Band,
}

impl Due {
    /// Time to spare after the work's cost, ns; negative once it can no longer make it.
    pub fn slack(&self, now: Instant) -> i64 {
        let ahead = match self.deadline.checked_duration_since(now) {
            Some(d) => d.as_nanos() as i64,
            None => -(now.duration_since(self.deadline).as_nanos() as i64),
        };
        ahead - self.cost.as_nanos() as i64
    }

    /// `(band, slack)`, lower runs first. Work at or under its urgent slack ranks
    /// [`Band::Urgent`] and then by slack alone, except [`Band::Bulk`] (its deadline is soft, the
    /// starvation bound promotes it instead) and [`Band::Final`], which turns urgent only once
    /// past its deadline.
    ///
    /// ASR finals ahead of urgent work measured worse under load: at 200 calls the speech
    /// pipeline stalled (TTFA p50 5.5-7.3 s vs 1.8 s; LLM TTFT p95 8-12 s vs 2.4-2.8 s), with or
    /// without partials deferred while finals wait.
    pub fn rank(&self, now: Instant) -> (Band, i64) {
        let slack = self.slack(now);
        let urgent = match self.band {
            Band::Bulk => i64::MIN,
            Band::Final => 0,
            _ => (2 * self.cost).max(URGENT_SLACK).as_nanos() as i64,
        };
        (if slack <= urgent { Band::Urgent } else { self.band }, slack)
    }

    /// Whether `self`, waiting, should take the device from a holder due `mine`: a lower band; or
    /// the same band and tighter by more than `margin_ns`, or tighter and going negative during
    /// the holder's next tick (`mine.cost`).
    pub fn outranks(&self, mine: &Due, now: Instant, margin_ns: i64) -> bool {
        let ((b, s), (mb, my)) = (self.rank(now), mine.rank(now));
        b < mb || (b == mb && (s < my.saturating_sub(margin_ns) || (s < my && s < mine.cost.as_nanos() as i64)))
    }

    /// The class callers' deadlines, for work pending since `since`. Final and Deadline are the
    /// old classes' budgets (an ASR final's every step; a first token or first audio); Normal is
    /// the starvation bound itself; Bulk lies past it. The band keeps the class order.
    pub fn from_urgency(u: Urgency, since: Instant) -> Due {
        let after = match u {
            Urgency::Final => Duration::from_millis(100),
            Urgency::Deadline => Duration::from_millis(300),
            Urgency::Normal => max_wait(),
            Urgency::Bulk => 4 * max_wait(),
        };
        Due { deadline: since + after, cost: Duration::ZERO, band: u.into() }
    }
}

#[derive(Debug)]
struct Waiter {
    due: Due,
    since: Instant,
    seq: u64,
    wake: oneshot::Sender<()>,
}

impl Waiter {
    /// The waiter's `Due`; once it has waited `max_wait`, urgent with its deadline pulled in to
    /// `since + max_wait`. Only then: capping at enqueue would flatten every deadline past the
    /// bound into one.
    fn due(&self, now: Instant, max_wait: Duration) -> Due {
        let cap = self.since + max_wait;
        if now >= cap {
            Due { deadline: self.due.deadline.min(cap), band: Band::Urgent, ..self.due }
        } else {
            self.due
        }
    }
}

#[derive(Debug)]
struct PrioState {
    held: bool,
    seq: u64,
    max_wait: Duration,
    waiters: Vec<Waiter>,
}

impl PrioState {
    fn new() -> PrioState {
        PrioState { held: false, seq: 0, max_wait: max_wait(), waiters: Vec::new() }
    }

    fn tightest(&self, now: Instant) -> Option<usize> {
        (0..self.waiters.len()).min_by_key(|&i| (self.waiters[i].due(now, self.max_wait).rank(now), self.waiters[i].seq))
    }

    /// Give the device to the best-ranked waiter, or mark it free.
    fn hand_off(&mut self) {
        let now = Instant::now();
        while let Some(i) = self.tightest(now) {
            if self.waiters.swap_remove(i).wake.send(()).is_ok() {
                return;
            }
        }
        self.held = false;
    }

    fn outranks(&self, w: &Waiter, mine: Due, now: Instant) -> bool {
        w.due(now, self.max_wait).outranks(&mine, now, YIELD_MARGIN)
    }

    /// Whether a holder due `mine`, holding since `since`, should hand the device on: outranked,
    /// or tied (same band, slack within the margin) after a [`DEADLINE_QUANTUM`].
    fn should_yield(&self, mine: Due, since: Instant, now: Instant) -> bool {
        let quantum_spent = now.saturating_duration_since(since) >= DEADLINE_QUANTUM;
        let (mb, my) = mine.rank(now);
        self.waiters.iter().any(|w| {
            self.outranks(w, mine, now) || {
                let (b, s) = w.due(now, self.max_wait).rank(now);
                quantum_spent && b == mb && s <= my + YIELD_MARGIN
            }
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
    async fn acquire(state: &Arc<parking_lot::Mutex<PrioState>>, due: Due) -> PrioHold {
        let (seq, rx) = {
            let mut s = state.lock();
            if !s.held && s.waiters.is_empty() {
                s.held = true;
                return PrioHold(Arc::clone(state));
            }
            let (wake, rx) = oneshot::channel();
            s.seq += 1;
            let seq = s.seq;
            s.waiters.push(Waiter { due, since: Instant::now(), seq, wake });
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
            prio: Arc::new(parking_lot::Mutex::new(PrioState::new())),
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

    /// [`Self::outranked_due`] for a model whose most urgent work, starting now, is `mine`.
    pub fn outranked(&self, mine: Urgency) -> bool {
        self.outranked_due(Due::from_urgency(mine, Instant::now()))
    }

    /// Whether a model due `mine` is outranked by a waiting co-tenant right now: it should keep
    /// its next tick short.
    pub fn outranked_due(&self, mine: Due) -> bool {
        if self.mode != CoSched::Deadline {
            return false;
        }
        let now = Instant::now();
        let s = self.prio.lock();
        s.waiters.iter().any(|w| s.outranks(w, mine, now))
    }

    /// The best-ranked waiter's `Due` (starvation bound applied), if anyone waits.
    pub fn tightest_waiter(&self) -> Option<Due> {
        if self.mode != CoSched::Deadline {
            return None;
        }
        let now = Instant::now();
        let s = self.prio.lock();
        s.tightest(now).map(|i| s.waiters[i].due(now, s.max_wait))
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

    /// [`Self::take`] for a model whose most urgent work, starting now, is `urgency`.
    pub async fn take_at(&mut self, turn: &DeviceTurn, urgency: Urgency) {
        self.take_due(turn, Due::from_urgency(urgency, Instant::now())).await
    }

    /// [`Self::take`] for a model whose most urgent work is `due` ([`CoSched::Deadline`]).
    pub async fn take_due(&mut self, turn: &DeviceTurn, due: Due) {
        if turn.mode == CoSched::Deadline {
            if let (Some(hold), Some(since)) = (&self.prio, self.since) {
                if !hold.0.lock().should_yield(due, since, Instant::now()) {
                    self.used += 1;
                    return;
                }
                self.release();
            }
            self.prio = Some(PrioHold::acquire(&turn.prio, due).await);
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

    /// Bulk work is due once it has waited the starvation bound, so a stream of deadline ticks
    /// cannot starve it.
    #[tokio::test]
    async fn bulk_waiters_age_in() {
        const BOUND: Duration = Duration::from_millis(50);
        let dt = Arc::new(DeviceTurn::new(CoSched::Deadline, 4));
        dt.prio.lock().max_wait = BOUND;
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
            assert!(t0.elapsed() < 6 * BOUND, "bulk waiter starved");
        }
        assert!(t0.elapsed() >= BOUND);
    }

    fn queue(s: &mut PrioState, due: Due, since: Instant) {
        s.seq += 1;
        let seq = s.seq;
        s.waiters.push(Waiter { due, since, seq, wake: oneshot::channel().0 });
    }

    fn due_in(now: Instant, ms: u64, cost_ms: u64) -> Due {
        Due { deadline: now + Duration::from_millis(ms), cost: Duration::from_millis(cost_ms), band: Band::Stream }
    }

    fn tightest_seq(s: &PrioState, now: Instant) -> u64 {
        s.waiters[s.tightest(now).unwrap()].seq
    }

    #[test]
    fn slack_counts_cost_and_goes_negative() {
        let now = Instant::now();
        assert_eq!(due_in(now, 100, 30).slack(now), 70_000_000);
        let late = Due { deadline: now, ..due_in(now, 0, 5) };
        assert_eq!(late.slack(now + Duration::from_millis(10)), -15_000_000);
    }

    /// Waiters are served least slack first, cost included, whatever order they queued in.
    #[test]
    fn least_slack_goes_first_and_ties_keep_arrival_order() {
        let now = Instant::now();
        let mut s = PrioState::new();
        queue(&mut s, due_in(now, 500, 0), now); // 1
        queue(&mut s, due_in(now, 300, 250), now); // 2: slack 50
        queue(&mut s, due_in(now, 100, 0), now); // 3: slack 100
        queue(&mut s, due_in(now, 100, 0), now); // 4: tie with 3
        let mut order = Vec::new();
        while let Some(i) = s.tightest(now) {
            order.push(s.waiters.swap_remove(i).seq);
        }
        assert_eq!(order, vec![2, 3, 4, 1]);
    }

    /// Classes arriving together keep today's order: Final, Deadline, Normal, Bulk.
    #[test]
    fn from_urgency_keeps_class_order_for_same_time_arrivals() {
        let now = Instant::now();
        let mut s = PrioState::new();
        for u in [Urgency::Bulk, Urgency::Normal, Urgency::Deadline, Urgency::Final] {
            queue(&mut s, Due::from_urgency(u, now), now);
        }
        let mut order = Vec::new();
        while let Some(i) = s.tightest(now) {
            order.push(s.waiters.swap_remove(i).seq);
        }
        assert_eq!(order, vec![4, 3, 2, 1]);
    }

    /// A Normal waiter queued `MAX_WAIT` ago beats a fresh Deadline one; so does Bulk, whose own
    /// deadline is past the bound, once it has waited the bound, and not before.
    #[test]
    fn starvation_bound_makes_old_waiters_due() {
        let now = Instant::now();
        let ago = |d: Duration| now.checked_sub(d).unwrap();
        let mut s = PrioState::new();
        let old = ago(s.max_wait + Duration::from_millis(10));
        queue(&mut s, Due::from_urgency(Urgency::Normal, old), old); // 1
        queue(&mut s, Due::from_urgency(Urgency::Deadline, now), now); // 2
        assert_eq!(tightest_seq(&s, now), 1);

        let mut s = PrioState::new();
        let half = ago(s.max_wait / 2);
        queue(&mut s, Due::from_urgency(Urgency::Bulk, half), half); // 1
        queue(&mut s, Due::from_urgency(Urgency::Deadline, now), now); // 2
        assert_eq!(tightest_seq(&s, now), 2);
        queue(&mut s, Due::from_urgency(Urgency::Bulk, old), old); // 3
        assert_eq!(tightest_seq(&s, now), 3);
        assert!(s.waiters[2].due(now, s.max_wait).deadline <= old + s.max_wait);
    }

    /// The holder yields to a waiter tighter by more than the margin, or to one that would go
    /// negative during the holder's next tick; a tie waits out the quantum.
    #[test]
    fn holder_yield_rule() {
        let now = Instant::now();
        let fresh = now;
        let spent = now.checked_sub(DEADLINE_QUANTUM).unwrap();
        let mine = due_in(now, 100, 0);

        let mut s = PrioState::new();
        queue(&mut s, due_in(now, 50, 0), now);
        assert!(s.should_yield(mine, fresh, now), "tighter by more than the margin");

        let mut s = PrioState::new();
        queue(&mut s, due_in(now, 90, 0), now);
        assert!(!s.should_yield(mine, fresh, now), "a tie keeps the device within the quantum");
        assert!(s.should_yield(mine, spent, now), "a tie hands over after the quantum");

        let mut s = PrioState::new();
        queue(&mut s, due_in(now, 60, 0), now);
        assert!(!s.should_yield(due_in(now, 70, 0), fresh, now));
        // Both urgent (slack under twice the cost): within the band, the slack rule.
        let mut s = PrioState::new();
        queue(&mut s, due_in(now, 60, 20), now); // slack 40
        assert!(s.should_yield(due_in(now, 130, 80), fresh, now), "would go negative during my tick");
        assert!(!s.should_yield(due_in(now, 110, 80), fresh, now), "the holder is tighter still");
    }

    fn due_band(now: Instant, ms: u64, cost_ms: u64, band: Band) -> Due {
        Due { band, ..due_in(now, ms, cost_ms) }
    }

    /// A first output goes ahead of decode however much tighter the decode's deadline, and a
    /// holder ticking decode hands the device to it.
    #[test]
    fn first_outputs_outrank_decode_whatever_the_slack() {
        let now = Instant::now();
        let mut s = PrioState::new();
        queue(&mut s, due_band(now, 90, 10, Band::Stream), now); // 1: decode, slack 80
        queue(&mut s, due_band(now, 800, 100, Band::First), now); // 2: TTS first audio, slack 700
        queue(&mut s, due_band(now, 400, 20, Band::First), now); // 3: LLM first token, slack 380
        queue(&mut s, due_band(now, 500, 20, Band::Final), now); // 4: ASR final, slack 480
        queue(&mut s, due_band(now, 60, 5, Band::Bulk), now); // 5: partial, slack 55
        let mut order = Vec::new();
        while let Some(i) = s.tightest(now) {
            order.push(s.waiters.swap_remove(i).seq);
        }
        assert_eq!(order, vec![4, 3, 2, 1, 5], "ASR final, first outputs by slack, decode, partial");

        let mut s = PrioState::new();
        queue(&mut s, due_band(now, 800, 100, Band::First), now);
        assert!(s.should_yield(due_band(now, 90, 10, Band::Stream), now, now));
        assert!(!s.should_yield(due_band(now, 500, 20, Band::First), now, now), "tighter first output keeps it");
        // An ASR final takes the device from a tighter first output (a render boosted to First).
        let mut s = PrioState::new();
        queue(&mut s, due_band(now, 500, 20, Band::Final), now);
        assert!(s.should_yield(due_band(now, 450, 100, Band::First), now, now));
        assert!(!s.should_yield(due_band(now, 100, 60, Band::First), now, now), "an urgent holder keeps it");
    }

    /// A stream about to underrun (slack under twice its render) preempts a first output, and so
    /// does decode that has waited out its TBT; a partial is not promoted by its soft deadline.
    #[test]
    fn near_miss_ranks_urgent_over_first_outputs() {
        let now = Instant::now();
        let first = due_band(now, 500, 20, Band::First);
        let stream = due_band(now, 150, 60, Band::Stream); // slack 90 <= 2 * 60
        assert_eq!(stream.rank(now).0, Band::Urgent);
        assert_eq!(first.rank(now).0, Band::First);
        assert!(stream.outranks(&first, now, YIELD_MARGIN));
        assert!(!first.outranks(&stream, now, YIELD_MARGIN));
        let mut s = PrioState::new();
        queue(&mut s, first, now);
        queue(&mut s, stream, now);
        assert_eq!(tightest_seq(&s, now), 2);

        let decode = due_band(now, 100, 10, Band::Stream);
        assert_eq!(decode.rank(now).0, Band::Stream);
        assert_eq!(decode.rank(now + Duration::from_millis(75)).0, Band::Urgent);
        let late = due_band(now, 10, 5, Band::Bulk);
        assert_eq!(late.rank(now + Duration::from_millis(50)).0, Band::Bulk);
        // A first output past its deadline is urgent too.
        assert_eq!(first.rank(now + Duration::from_millis(600)).0, Band::Urgent);
        // Everything late (overload): urgent work by slack alone, an ASR final included.
        let dry = due_band(now, 0, 300, Band::Stream); // slack -300
        let final_ = due_band(now, 0, 20, Band::Final); // slack -20
        assert!(dry.outranks(&final_, now, YIELD_MARGIN) && !final_.outranks(&dry, now, YIELD_MARGIN));
        // A final turns urgent only past its deadline: until then its band is ahead of the rest.
        assert_eq!(due_band(now, 50, 40, Band::Final).rank(now).0, Band::Final);
        assert_eq!(due_band(now, 30, 40, Band::Final).rank(now).0, Band::Urgent);
    }

    #[test]
    fn free_and_rr_never_report_waiters() {
        for mode in [CoSched::Free, CoSched::Rr] {
            let dt = DeviceTurn::new(mode, 4);
            queue(&mut dt.prio.lock(), due_in(Instant::now(), 0, 0), Instant::now());
            assert!(dt.tightest_waiter().is_none());
            assert!(!dt.outranked_due(due_in(Instant::now(), 1000, 0)));
        }
    }

    /// The device goes to the least-slack waiter through real turns, and `tightest_waiter` and
    /// `outranked_due` see it.
    #[tokio::test]
    async fn take_due_serves_least_slack() {
        let dt = Arc::new(DeviceTurn::new(CoSched::Deadline, 4));
        let mut holder = Turn::default();
        let now = Instant::now();
        holder.take_due(&dt, due_in(now, 5000, 0)).await;
        let order = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let mut tasks = Vec::new();
        for ms in [900u64, 200, 600] {
            let (dt, order) = (Arc::clone(&dt), Arc::clone(&order));
            tasks.push(tokio::spawn(async move {
                let mut t = Turn::default();
                t.take_due(&dt, due_in(now, ms, 0)).await;
                order.lock().push(ms);
                t.release();
            }));
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(dt.tightest_waiter().map(|d| d.deadline), Some(now + Duration::from_millis(200)));
        assert!(dt.outranked_due(due_in(now, 5000, 0)));
        assert!(!dt.outranked_due(due_in(now, 100, 0)));
        holder.release();
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(*order.lock(), vec![200, 600, 900]);
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
