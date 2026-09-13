//! SLO-aware per-tick prefill budget (`PLOW_TBT_SLO_MS`, `PLOW_TTFT_SLO_MS`).
//!
//! Every tick still runs every live decode row. Under an inter-token (TBT) target the prefill
//! the tick carries is sized in MILLISECONDS, not rows: candidates are taken in order and each
//! gets the largest chunk whose launch keeps the predicted tick (prefill launches + the decode
//! pass + host remainder) at or under the target. Several requests may each get a launch.
//! Under a TTFT target candidates are ordered by deadline slack (EDF): prompts this tick
//! completes first, then by slack, and requests that can no longer make their deadline last.
//!
//! **Progress rule.** When even the smallest chunk would push the tick over the target, the
//! tick runs decode only; after `K - 1` such ticks one chunk runs, where
//! `K = ceil(p_min / s)`, `s = target - (decode + host)` and `p_min` is the predicted cost of the
//! smallest chunk. That chunk is sized to fill `K * s`, so over the `K` ticks the mean
//! inter-token time is `decode + host + cost / K <= target`. `K` is capped at [`MAX_SKIP_TICKS`]:
//! a target at or below the decode pass itself cannot be met, and the cap keeps prompts moving
//! (one smallest chunk every `MAX_SKIP_TICKS` ticks) instead of starving them.
//!
//! A shrunk chunk runs in whatever bucket the engine plans for `[c0, c0 + n)`, which
//! [`Ladder::bucket_for`] mirrors: the smallest rung that holds `n`, moved to the sparse rung at
//! its real row count when the prior context reaches `amd_tail_sparse_ctx`.
//!
//! Costs come from [`TickCost`], an online model updated from every observed tick. The target is
//! met on an upper quantile, not the mean: each tick's prediction is inflated by the running p90
//! of actual / predicted for its [`TickClass`] (plain, completing a prompt, starting one).
//! Unset targets never reach this module: [`plan_tick`] returns [`step::plan`] verbatim.
//!
//! Under a target the plan uses isolated launches only; cross-request packing is not used
//! because its cost is not modelled.

use super::step::{self, Backend, Candidate, Launch, Plan, Tick};

/// Longest run of decode-only ticks the progress rule allows while prefill waits.
pub const MAX_SKIP_TICKS: u32 = 8;

/// Latency targets. Both `None` is the throughput schedule.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Targets {
    pub tbt_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
}

impl Targets {
    pub fn active(&self) -> bool {
        self.tbt_ms.is_some() || self.ttft_ms.is_some()
    }
}

/// The compiled prefill rungs as `(rows, sparse)`, ascending, plus the tail-sparse floor.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ladder {
    pub rungs: Vec<(u32, bool)>,
    pub tail_sparse_ctx: Option<u32>,
}

impl Ladder {
    /// The bucket the engine plans for a chunk of `rows` whose first row sits at `prior`: the
    /// smallest rung that holds it, or the widest sparse rung when that rung is dense and
    /// `prior` reaches the tail-sparse floor (`serve::engine::retarget_dense_tail`).
    pub fn bucket_for(&self, rows: u32, prior: u32) -> Option<u32> {
        let (width, sparse) = *self.rungs.iter().find(|&&(w, _)| w >= rows)?;
        let wider_sparse = self
            .rungs
            .iter()
            .filter(|&&(w, s)| s && w > width)
            .map(|&(w, _)| w)
            .max();
        match (self.tail_sparse_ctx, wider_sparse) {
            (Some(floor), Some(sparse_w)) if !sparse && prior >= floor => Some(sparse_w),
            _ => Some(width),
        }
    }

    fn widest(&self) -> u32 {
        self.rungs.last().map_or(0, |r| r.0)
    }

    fn granule(&self) -> u32 {
        self.rungs.first().map_or(1, |r| r.0.max(1))
    }
}

/// What the planner needs to price a tick.
pub trait CostModel {
    /// One isolated prefill launch of `rows` real rows in `bucket`, first row at `prior`;
    /// `fresh` = the request's first launch (its cursor is built in the same call).
    fn launch_ms(&self, bucket: u32, rows: u32, prior: u32, fresh: bool) -> f64;
    /// The decode pass over `rows` slots; `after_prefill` = it follows a prefill launch.
    fn decode_ms(&self, rows: u32, after_prefill: bool) -> f64;
    /// Tick time outside the launches and the decode pass.
    fn host_ms(&self) -> f64;
    /// What a tick of `class` is planned against: an upper quantile of actual / predicted.
    fn margin(&self, _class: TickClass) -> f64 {
        1.0
    }
}

/// A tick by its riskiest launch, for the planning margin. The tail of the model's error is
/// concentrated in ticks that start a request (its cursor build) or complete one (the token read
/// and the prefix publish); ordinary middle chunks are within a few percent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum TickClass {
    #[default]
    Plain,
    Completing,
    Fresh,
}

/// Streaming quantile with multiplicative, scale-free steps: at equilibrium `P(x > q) = 1 - tau`.
#[derive(Clone, Copy, Debug)]
pub struct Quantile {
    q: f64,
    tau: f64,
    eta: f64,
}

impl Quantile {
    pub fn new(q: f64, tau: f64, eta: f64) -> Self {
        Self { q, tau, eta }
    }

    pub fn observe(&mut self, x: f64) {
        if !(x.is_finite() && x > 0.0) {
            return;
        }
        self.q *= if x > self.q { (self.eta * self.tau).exp() } else { (-self.eta * (1.0 - self.tau)).exp() };
    }

    pub fn get(&self) -> f64 {
        self.q
    }
}

/// Recursive least squares with exponential forgetting. The prior is `theta`; the covariance
/// is kept within its initial trace so directions the data never excite (every chunk the same
/// width, say) do not wind up and swing on the first sample that does.
#[derive(Clone, Debug)]
struct Rls<const N: usize> {
    theta: [f64; N],
    p: [[f64; N]; N],
    trace0: f64,
    lambda: f64,
    samples: u64,
}

impl<const N: usize> Rls<N> {
    /// `prior_sd` per coefficient and `noise_sd` of one observation, both in ms.
    fn new(theta: [f64; N], prior_sd: [f64; N], noise_sd: f64, lambda: f64) -> Self {
        let mut p = [[0.0; N]; N];
        for i in 0..N {
            p[i][i] = (prior_sd[i] / noise_sd).powi(2);
        }
        let trace0 = (0..N).map(|i| p[i][i]).sum();
        Self { theta, p, trace0, lambda, samples: 0 }
    }

    fn predict(&self, x: &[f64; N]) -> f64 {
        self.theta.iter().zip(x).map(|(t, x)| t * x).sum()
    }

    fn update(&mut self, x: &[f64; N], y: f64) {
        let mut px = [0.0; N];
        for i in 0..N {
            px[i] = (0..N).map(|j| self.p[i][j] * x[j]).sum();
        }
        let denom = self.lambda + x.iter().zip(&px).map(|(a, b)| a * b).sum::<f64>();
        if !(denom.is_finite() && denom > 0.0 && y.is_finite()) {
            return;
        }
        let err = y - self.predict(x);
        let mut trace = 0.0;
        for i in 0..N {
            let k = px[i] / denom;
            self.theta[i] += k * err;
            for j in 0..N {
                self.p[i][j] = (self.p[i][j] - k * px[j]) / self.lambda;
            }
            trace += self.p[i][i];
        }
        if trace > self.trace0 {
            let s = self.trace0 / trace;
            self.p.iter_mut().flatten().for_each(|v| *v *= s);
        }
        self.samples += 1;
    }

    fn scaled(&self, s: f64) -> Self {
        let mut out = self.clone();
        out.theta.iter_mut().for_each(|t| *t *= s);
        out
    }
}

const LAUNCH_DIM: usize = 5;
const DECODE_DIM: usize = 2;

fn launch_x(rows: u32, prior: u32) -> [f64; LAUNCH_DIM] {
    let r = f64::from(rows) / 8192.0;
    let p = f64::from(prior) / 65536.0;
    [1.0, r, p, r * p, if prior == 0 { 1.0 } else { 0.0 }]
}

fn decode_x(rows: u32) -> [f64; DECODE_DIM] {
    [1.0, f64::from(rows) / 20.0]
}

/// Priors, in ms, over `launch_x`: GLM-5.3 TP8 FP8, 78 layers, MI300X, from the 2026-09-11/12
/// `PLOW_TICK_LOG` server logs. Online updates replace them within a few samples.
fn launch_prior(bucket: u32) -> [f64; LAUNCH_DIM] {
    match bucket {
        128 => [100.0, 100.0, 650.0, 0.0, 0.0],
        512 => [120.0, 200.0, 1700.0, 0.0, 0.0],
        2048 => [356.0, 106.0, 1056.0, 160.0, 0.0],
        8192 => [150.0, 760.0, 40.0, 0.0, 90.0],
        _ => [60.0, 800.0, 400.0, 0.0, 0.0],
    }
}
const LAUNCH_PRIOR_SD: [f64; LAUNCH_DIM] = [200.0, 600.0, 600.0, 600.0, 300.0];
const DECODE_PRIOR: [[f64; DECODE_DIM]; 2] = [[92.8, 7.7], [114.6, 4.3]];
const DECODE_PRIOR_SD: [f64; DECODE_DIM] = [50.0, 50.0];

/// Online per-tick cost predictor: one RLS model per prefill bucket over
/// `[1, rows, prior, rows * prior, first chunk]`, one per decode pass kind (after a prefill
/// launch, or decode-only) over `[1, rows]`, an EWMA of a fresh request's cursor overhead,
/// and a winsorised EWMA of the host remainder.
///
/// A model with no samples yet predicts its prior scaled by how fast this engine's decode-only
/// pass is relative to the prior's (`speed`), so a faster engine or a truncated packet starts
/// near its own scale; its first sample starts from that scaled prior.
#[derive(Clone, Debug)]
pub struct TickCost {
    launches: Vec<(u32, Rls<LAUNCH_DIM>)>,
    decode: [Rls<DECODE_DIM>; 2],
    fresh_ms: f64,
    host_ms: f64,
    /// p90 of actual / predicted tick, per [`TickClass`]; fresh ticks are rare, so step faster.
    margins: [Quantile; 3],
}

impl Default for TickCost {
    fn default() -> Self {
        let decode = |i: usize| Rls::new(DECODE_PRIOR[i], DECODE_PRIOR_SD, 10.0, 0.98);
        let q = |eta| Quantile::new(1.0, 0.9, eta);
        Self {
            launches: Vec::new(),
            decode: [decode(0), decode(1)],
            fresh_ms: 0.0,
            host_ms: 0.0,
            margins: [q(0.1), q(0.1), q(0.3)],
        }
    }
}

impl TickCost {
    /// This engine's decode pass relative to the prior's, from whichever pass kind has samples.
    fn speed(&self) -> f64 {
        let Some(i) = [0, 1].into_iter().find(|&i| self.decode[i].samples > 0) else {
            return 1.0;
        };
        let x = decode_x(20);
        let prior = DECODE_PRIOR[i][0] * x[0] + DECODE_PRIOR[i][1] * x[1];
        (self.decode[i].predict(&x) / prior).clamp(0.01, 100.0)
    }

    fn launch_model(&self, bucket: u32) -> Option<&Rls<LAUNCH_DIM>> {
        self.launches.iter().find(|(b, _)| *b == bucket).map(|(_, m)| m)
    }

    pub fn observe_launch(&mut self, bucket: u32, rows: u32, prior: u32, fresh: bool, ms: f64) {
        if fresh {
            let chunk = self.launch_ms(bucket, rows, prior, false);
            let extra = (ms - chunk).max(0.0);
            self.fresh_ms = 0.8 * self.fresh_ms + 0.2 * extra;
            return;
        }
        let speed = self.speed();
        let i = match self.launches.iter().position(|(b, _)| *b == bucket) {
            Some(i) => i,
            None => {
                let prior_model = Rls::new(launch_prior(bucket), LAUNCH_PRIOR_SD, 50.0, 0.99);
                self.launches.push((bucket, prior_model.scaled(speed)));
                self.launches.len() - 1
            }
        };
        self.launches[i].1.update(&launch_x(rows, prior), ms);
    }

    pub fn observe_decode(&mut self, rows: u32, after_prefill: bool, ms: f64) {
        let i = usize::from(after_prefill);
        if after_prefill && self.decode[1].samples == 0 {
            self.decode[1] = Rls::new(DECODE_PRIOR[1], DECODE_PRIOR_SD, 10.0, 0.98).scaled(self.speed());
        }
        self.decode[i].update(&decode_x(rows), ms);
    }

    /// One planned tick of `class` ran `ratio` times its (raw) prediction.
    pub fn observe_margin(&mut self, class: TickClass, ratio: f64) {
        self.margins[class as usize].observe(ratio);
    }

    pub fn observe_host(&mut self, ms: f64) {
        let capped = ms.max(0.0).min(3.0 * self.host_ms + 5.0);
        self.host_ms = 0.9 * self.host_ms + 0.1 * capped;
    }
}

impl CostModel for TickCost {
    fn launch_ms(&self, bucket: u32, rows: u32, prior: u32, fresh: bool) -> f64 {
        let x = launch_x(rows, prior);
        let chunk = match self.launch_model(bucket) {
            Some(m) => m.predict(&x),
            None => {
                let p = launch_prior(bucket);
                self.speed() * p.iter().zip(&x).map(|(a, b)| a * b).sum::<f64>()
            }
        };
        chunk.max(0.0) + if fresh { self.fresh_ms } else { 0.0 }
    }

    fn decode_ms(&self, rows: u32, after_prefill: bool) -> f64 {
        let i = usize::from(after_prefill);
        let x = decode_x(rows);
        let m = &self.decode[i];
        if i == 1 && m.samples == 0 {
            return m.predict(&x) * self.speed();
        }
        m.predict(&x).max(0.0)
    }

    fn host_ms(&self) -> f64 {
        self.host_ms
    }

    fn margin(&self, class: TickClass) -> f64 {
        self.margins[class as usize].get().clamp(0.8, 4.0)
    }
}

/// Per-dispatcher SLO state carried across ticks.
#[derive(Clone, Debug, Default)]
pub struct SloState {
    pub cost: TickCost,
    /// Consecutive decode-only ticks the progress rule has spent while prefill waited.
    pub skipped: u32,
    /// The engine's prefill ladder, read once.
    pub ladder: Option<Ladder>,
    /// The last plan's outcome, until its tick is observed.
    pub last: Option<SloOutcome>,
}

/// A prefill candidate as the SLO planner sees it.
#[derive(Clone, Copy, Debug)]
pub struct SloCandidate {
    pub base: Candidate,
    /// Absolute position of its next row (its cursor frontier; 0 for a fresh request).
    pub prior: u32,
    /// Prompt rows it still has to prefill.
    pub remaining: u32,
    /// Time since it arrived.
    pub age_ms: f64,
}

impl SloCandidate {
    fn fresh(&self) -> bool {
        !self.base.planned
    }

    fn class(&self, rows: u32) -> TickClass {
        if self.fresh() {
            TickClass::Fresh
        } else if rows >= self.remaining {
            TickClass::Completing
        } else {
            TickClass::Plain
        }
    }

    /// The most rows this tick may give it: its planned chunk, or for a fresh request what the
    /// engine's first plan against the full budget would hold.
    fn max_rows(&self, full_budget: u32) -> u32 {
        let natural = if self.base.planned { self.base.span.n_rows } else { self.remaining };
        natural.min(full_budget)
    }
}

/// The chunk an isolated launch actually ran: its bucket's rows, first row, real rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ran {
    pub bucket: u32,
    pub c0: u32,
    pub clen: u32,
}

/// Why a tick's prefill looks the way it does (for the tick log).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SloOutcome {
    /// Prefill budget the target left this tick (ms); infinite when no target binds.
    pub budget_ms: f64,
    /// Predicted tick: launches + decode pass + host (ms).
    pub predicted_ms: f64,
    /// The progress rule ran this tick's chunk.
    pub progress: bool,
    /// The progress rule's `K` when it applied this tick.
    pub k: u32,
    /// The tick's class and the margin it was planned against.
    pub class: TickClass,
    pub margin: f64,
}

/// Plan one tick. Unset targets return [`step::plan`] verbatim.
#[allow(clippy::too_many_arguments)]
pub fn plan_tick(
    targets: Targets,
    backend: Backend,
    tick: Tick,
    decodes: impl IntoIterator<Item = u32>,
    candidates: &[SloCandidate],
    ladder: &Ladder,
    cost: &impl CostModel,
    skipped: &mut u32,
    program_rows: impl Fn(u32) -> Option<u32>,
    program_span_limit: impl Fn(u32) -> u32,
) -> (Plan, SloOutcome) {
    if !targets.active() {
        let base: Vec<Candidate> = candidates.iter().map(|c| c.base).collect();
        let plan = step::plan(backend, tick, decodes, &base, program_rows, program_span_limit);
        return (plan, SloOutcome::default());
    }
    let decodes: Vec<u32> = decodes.into_iter().collect();
    let full_budget = tick.cap_rows.min(backend.step_budget);
    let decode_rows = u32::try_from(decodes.len()).unwrap_or(u32::MAX);
    let fixed_ms = |after_prefill: bool| {
        if decodes.is_empty() {
            cost.host_ms()
        } else {
            cost.decode_ms(decode_rows, after_prefill) + cost.host_ms()
        }
    };
    let binding = !decodes.is_empty() && targets.tbt_ms.is_some();
    let target = targets.tbt_ms.unwrap_or(f64::INFINITY);
    // Raw prefill ms a tick of `class` may still spend: the target holds on the class's upper
    // quantile of actual / predicted, not on the mean prediction.
    let room = |class: TickClass, spent: f64| {
        if binding {
            target / cost.margin(class) - fixed_ms(true) - spent
        } else {
            f64::INFINITY
        }
    };
    let mut outcome = SloOutcome { budget_ms: room(TickClass::Plain, 0.0), ..SloOutcome::default() };
    let order = order(targets, tick, candidates, ladder, cost, full_budget);
    let mut launches = Vec::new();
    let mut class = TickClass::Plain;
    let mut left_rows = full_budget;
    let mut spent_ms = 0.0;
    for &i in &order {
        let c = &candidates[i];
        if left_rows == 0 {
            break;
        }
        let whole = c.max_rows(full_budget);
        let rows = if binding {
            let max_rows = whole.min(left_rows);
            rows_within(c, max_rows, room(class.max(c.class(max_rows)), spent_ms), ladder, cost)
        } else {
            // No target binds: a planned chunk runs whole or waits, as in `step::plan`.
            (whole <= left_rows && whole > 0).then_some(whole)
        };
        let Some(rows) = rows else { continue };
        spent_ms += launch_cost(c, rows, ladder, cost);
        left_rows -= rows;
        class = class.max(c.class(rows));
        launches.push(launch(c, rows));
    }
    if binding && launches.is_empty() {
        if let Some(&first) = order.first() {
            let c = &candidates[first];
            let max_rows = c.max_rows(full_budget);
            let min_rows = ladder.granule().min(max_rows).max(1);
            let p_min = launch_cost(c, min_rows, ladder, cost);
            let slack = room(c.class(min_rows), 0.0);
            let k = progress_k(p_min, slack);
            outcome.k = k;
            if *skipped + 1 >= k {
                let budget = (f64::from(k) * slack).max(p_min);
                let rows = rows_within(c, max_rows, budget, ladder, cost).unwrap_or(min_rows);
                spent_ms = launch_cost(c, rows, ladder, cost);
                class = c.class(rows);
                launches.push(launch(c, rows));
                outcome.progress = true;
            }
        }
    }
    if launches.is_empty() && binding && !candidates.is_empty() {
        *skipped += 1;
    } else {
        *skipped = 0;
    }
    outcome.class = class;
    outcome.margin = cost.margin(class);
    outcome.predicted_ms = spent_ms + fixed_ms(!launches.is_empty());
    let prefill: u32 = launches.iter().map(Launch::rows).sum();
    (
        Plan {
            decodes,
            launches,
            budget_left: full_budget.saturating_sub(prefill),
        },
        outcome,
    )
}

/// `K` of the progress rule: the fewest ticks whose slack pays for the smallest chunk.
pub fn progress_k(p_min_ms: f64, slack_ms: f64) -> u32 {
    if slack_ms <= 0.0 || !slack_ms.is_finite() {
        return if slack_ms.is_infinite() && slack_ms > 0.0 { 1 } else { MAX_SKIP_TICKS };
    }
    let k = (p_min_ms / slack_ms).ceil();
    if k.is_finite() {
        (k.max(1.0) as u32).min(MAX_SKIP_TICKS)
    } else {
        MAX_SKIP_TICKS
    }
}

fn launch(c: &SloCandidate, rows: u32) -> Launch {
    let mut span = c.base.span;
    span.row0 = 0;
    span.n_rows = rows;
    span.kv_len = span.kv_row0 + rows;
    Launch { spans: vec![span] }
}

fn launch_cost(c: &SloCandidate, rows: u32, ladder: &Ladder, cost: &impl CostModel) -> f64 {
    let bucket = ladder.bucket_for(rows, c.prior).unwrap_or(rows);
    cost.launch_ms(bucket, rows, c.prior, c.fresh())
}

/// The largest chunk of at most `max_rows` whose launch fits `budget_ms`: `max_rows` itself or
/// a multiple of the smallest rung. `None` when nothing fits.
fn rows_within(
    c: &SloCandidate,
    max_rows: u32,
    budget_ms: f64,
    ladder: &Ladder,
    cost: &impl CostModel,
) -> Option<u32> {
    if max_rows == 0 {
        return None;
    }
    if launch_cost(c, max_rows, ladder, cost) <= budget_ms {
        return Some(max_rows);
    }
    let granule = ladder.granule();
    let mut rows = (max_rows - 1) / granule * granule;
    while rows > 0 {
        if launch_cost(c, rows, ladder, cost) <= budget_ms {
            return Some(rows);
        }
        rows -= granule;
    }
    None
}

/// Candidate order. With a TTFT target: prompts completing this tick with slack left, then
/// the rest with slack left by slack, then the requests past saving by arrival. Otherwise the
/// arrival (or rotation) order of `step::plan`.
fn order(
    targets: Targets,
    tick: Tick,
    candidates: &[SloCandidate],
    ladder: &Ladder,
    cost: &impl CostModel,
    full_budget: u32,
) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..candidates.len())
        .filter(|&i| {
            let c = &candidates[i];
            c.base.span.n_rows > 0 && (c.base.span.slot as usize) < tick.slots
        })
        .collect();
    let slots = tick.slots.max(1);
    let fifo = |c: &SloCandidate| {
        let slot = c.base.span.slot as usize;
        if tick.rotate {
            ((slot + slots - tick.turn % slots) % slots, 0, slot)
        } else {
            (0, c.base.arrival, slot)
        }
    };
    let Some(ttft) = targets.ttft_ms else {
        idx.sort_by_key(|&i| fifo(&candidates[i]));
        return idx;
    };
    let widest = ladder.widest().max(1);
    let key = |c: &SloCandidate| {
        let per_row = cost.launch_ms(ladder.bucket_for(widest, c.prior).unwrap_or(widest), widest, c.prior, false)
            / f64::from(widest);
        let slack = ttft - c.age_ms - f64::from(c.remaining) * per_row;
        let completes = c.remaining <= c.max_rows(full_budget);
        let tier = match (slack >= 0.0, completes) {
            (true, true) => 0u8,
            (true, false) => 1,
            (false, _) => 2,
        };
        (tier, if tier == 2 { 0.0 } else { slack })
    };
    idx.sort_by(|&a, &b| {
        let (ca, cb) = (&candidates[a], &candidates[b]);
        let (ka, kb) = (key(ca), key(cb));
        ka.0.cmp(&kb.0)
            .then(ka.1.total_cmp(&kb.1))
            .then(fifo(ca).cmp(&fifo(cb)))
    });
    idx
}

/// Whether one request met each target: TTFT, and its mean inter-token time (TPOT) against the
/// TBT target. An unset target is met.
pub fn attained(targets: Targets, ttft_ms: f64, tpot_ms: f64) -> (bool, bool) {
    (
        targets.ttft_ms.is_none_or(|t| ttft_ms <= t),
        targets.tbt_ms.is_none_or(|t| tpot_ms <= t),
    )
}

/// Seconds a per-launch and per-decode-pass saving removes from a recorded run: the yardstick
/// the review log applies by hand. `ticks` = `(prefill launches, decode rows)` per recorded tick.
pub fn seconds_removed(ticks: &[(u32, u32)], per_launch_ms: f64, per_decode_ms: f64) -> f64 {
    ticks
        .iter()
        .map(|&(launches, decode_rows)| {
            f64::from(launches) * per_launch_ms + if decode_rows > 0 { per_decode_ms } else { 0.0 }
        })
        .sum::<f64>()
        / 1e3
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::PrefillSpan;

    /// Linear fake: a launch costs `fixed + per_row * rows + deep` where `deep` is added at a
    /// prior of 16384 or more in a dense bucket; decode `decode` ms; no host time.
    struct Fake {
        fixed: f64,
        per_row: f64,
        dense_deep: f64,
        decode: f64,
    }

    impl CostModel for Fake {
        fn launch_ms(&self, bucket: u32, rows: u32, prior: u32, _fresh: bool) -> f64 {
            let deep = if bucket < 8192 && prior >= 16384 { self.dense_deep } else { 0.0 };
            self.fixed + self.per_row * f64::from(rows) + deep
        }
        fn decode_ms(&self, _rows: u32, _after: bool) -> f64 {
            self.decode
        }
        fn host_ms(&self) -> f64 {
            0.0
        }
    }

    fn glm() -> Fake {
        Fake { fixed: 150.0, per_row: 0.09, dense_deep: 1200.0, decode: 110.0 }
    }

    fn ladder() -> Ladder {
        Ladder {
            rungs: vec![(128, false), (512, false), (2048, false), (8192, true)],
            tail_sparse_ctx: Some(16384),
        }
    }

    fn amd() -> Backend {
        Backend { step_budget: 8192, packing: true, split_spans: false, decode_rows_join_prefill: false }
    }

    fn tick() -> Tick {
        Tick { cap_rows: u32::MAX, packing: true, rotate: false, turn: 0, slots: 32 }
    }

    fn cand(slot: u32, arrival: u64, prior: u32, rows: u32, remaining: u32, age_ms: f64) -> SloCandidate {
        SloCandidate {
            base: Candidate {
                span: PrefillSpan {
                    row0: 0,
                    n_rows: rows,
                    slot,
                    flags: 0,
                    kv_row0: 0,
                    kv_len: rows,
                    state_slot: slot,
                    program: 0,
                },
                arrival,
                packable: false,
                planned: true,
            },
            prior,
            remaining,
            age_ms,
        }
    }

    fn run(targets: Targets, cands: &[SloCandidate], decodes: u32, state: &mut SloState) -> (Plan, SloOutcome) {
        plan_tick(targets, amd(), tick(), 0..decodes, cands, &ladder(), &glm(), &mut state.skipped, |_| None, |_| u32::MAX)
    }

    fn rows(plan: &Plan) -> Vec<(u32, u32)> {
        plan.launches.iter().map(|l| (l.spans[0].slot, l.spans[0].n_rows)).collect()
    }

    const TBT500: Targets = Targets { tbt_ms: Some(500.0), ttft_ms: None };

    #[test]
    fn the_bucket_mirror_moves_a_shrunk_deep_chunk_to_the_sparse_rung_at_real_rows() {
        let l = ladder();
        assert_eq!(l.bucket_for(1500, 4096), Some(2048));
        assert_eq!(l.bucket_for(1500, 65536), Some(8192));
        assert_eq!(l.bucket_for(3000, 0), Some(8192));
        assert_eq!(l.bucket_for(100, 16383), Some(128));
        assert_eq!(l.bucket_for(9000, 0), None);
        let off = Ladder { tail_sparse_ctx: None, ..ladder() };
        assert_eq!(off.bucket_for(1500, 65536), Some(2048));
    }

    /// 500 ms target, 110 ms decode: 390 ms of prefill = 150 + 0.09 n, so n = 2666 -> the
    /// 128-row granule below it, 2560. Decode rows are all admitted.
    #[test]
    fn the_budget_is_the_largest_chunk_that_keeps_the_tick_under_target() {
        let mut s = SloState::default();
        let (plan, out) = run(TBT500, &[cand(3, 0, 65536, 8192, 20000, 0.0)], 19, &mut s);
        assert_eq!(plan.decodes.len(), 19);
        assert_eq!(rows(&plan), [(3, 2560)]);
        assert!(out.predicted_ms <= 500.0 && out.predicted_ms > 490.0, "{out:?}");
        assert!(!out.progress);
    }

    /// Several requests share one tick's budget, each its own launch paying its own fixed cost.
    #[test]
    fn several_requests_fill_the_budget_in_order() {
        let mut s = SloState::default();
        let cands = [cand(0, 1, 65536, 8192, 20000, 0.0), cand(1, 2, 65536, 8192, 20000, 0.0)];
        let t = Targets { tbt_ms: Some(1000.0), ttft_ms: None };
        let (plan, out) = run(t, &cands, 5, &mut s);
        // 890 ms: slot 0 takes its whole 8192 chunk? 150 + 737 = 887 fits; slot 1 gets nothing.
        assert_eq!(rows(&plan), [(0, 8192)]);
        assert!(out.predicted_ms <= 1000.0);
        let short = [cand(0, 1, 65536, 1000, 1000, 0.0), cand(1, 2, 65536, 8192, 20000, 0.0)];
        let (plan, _) = run(t, &short, 5, &mut s);
        // slot 0: 150 + 90 = 240; slot 1 gets 890 - 240 = 650 -> 150 + 0.09 n <= 650 -> 5504.
        assert_eq!(rows(&plan), [(0, 1000), (1, 5504)]);
    }

    /// No decoders: nothing binds and planned chunks run whole, as the throughput schedule does.
    #[test]
    fn without_decoders_the_target_does_not_bind() {
        let mut s = SloState::default();
        let (plan, out) = run(TBT500, &[cand(0, 0, 65536, 8192, 20000, 0.0)], 0, &mut s);
        assert_eq!(rows(&plan), [(0, 8192)]);
        assert!(out.budget_ms.is_infinite());
    }

    /// 250 ms target, 110 ms decode: slack 140 < the 161.5 ms smallest chunk. K = 2: one tick
    /// decodes alone, the next runs a chunk sized to 2 * 140 = 280 ms (1444 rows -> 1408), and
    /// the mean tick over the pair is (110 + 110 + 276.7) / 2 <= 250.
    #[test]
    fn the_progress_rule_runs_one_chunk_every_k_ticks_and_the_mean_meets_the_target() {
        let t = Targets { tbt_ms: Some(250.0), ttft_ms: None };
        let c = [cand(0, 0, 65536, 8192, 60000, 0.0)];
        let mut s = SloState::default();
        let mut ticks = Vec::new();
        let mut prefilled = 0;
        for _ in 0..100 {
            let (plan, out) = run(t, &c, 20, &mut s);
            assert_eq!(plan.decodes.len(), 20);
            assert_eq!(out.k, 2);
            ticks.push(out.predicted_ms);
            prefilled += plan.prefill_rows();
            if out.progress {
                assert_eq!(rows(&plan), [(0, 1408)]);
            }
        }
        assert_eq!(prefilled, 50 * 1408);
        let mean = ticks.iter().sum::<f64>() / ticks.len() as f64;
        assert!(mean <= 250.0, "mean {mean}");
        assert!(ticks.iter().copied().fold(0.0, f64::max) > 250.0);
    }

    #[test]
    fn k_is_capped_when_the_decode_pass_alone_breaks_the_target() {
        assert_eq!(progress_k(160.0, 140.0), 2);
        assert_eq!(progress_k(100.0, 140.0), 1);
        assert_eq!(progress_k(160.0, -5.0), MAX_SKIP_TICKS);
        assert_eq!(progress_k(10_000.0, 1.0), MAX_SKIP_TICKS);
        let t = Targets { tbt_ms: Some(100.0), ttft_ms: None };
        let mut s = SloState::default();
        let c = [cand(0, 0, 65536, 8192, 60000, 0.0)];
        let runs: Vec<bool> = (0..3 * MAX_SKIP_TICKS).map(|_| run(t, &c, 20, &mut s).1.progress).collect();
        assert_eq!(runs.iter().filter(|&&p| p).count(), 3);
        assert!(runs[MAX_SKIP_TICKS as usize - 1]);
    }

    /// EDF: a prompt finishing this tick first, then by slack, the doomed last by arrival.
    #[test]
    fn a_ttft_target_orders_by_slack_with_completing_prompts_first_and_doomed_last() {
        let t = Targets { tbt_ms: None, ttft_ms: Some(10_000.0) };
        let cands = [
            // arrival order would pick slot 0 first
            cand(0, 0, 8192, 8192, 50000, 9_000.0), // needs ~5.5 s, 1 s left: doomed
            cand(1, 1, 8192, 8192, 30000, 2_000.0),  // needs ~3.4 s, slack ~4.6 s
            cand(2, 2, 8192, 8192, 60000, 1_000.0),  // needs ~6.7 s, slack ~2.3 s
            cand(3, 3, 65536, 700, 700, 500.0),      // a prefix-hit suffix: completes now
        ];
        let o = order(t, tick(), &cands, &ladder(), &glm(), 8192);
        assert_eq!(o, [3, 2, 1, 0]);
        let fifo = order(Targets { tbt_ms: Some(500.0), ttft_ms: None }, tick(), &cands, &ladder(), &glm(), 8192);
        assert_eq!(fifo, [0, 1, 2, 3]);
    }

    /// Unset targets hand back `step::plan` verbatim, over the same ragged mixes the step
    /// planner's accounting test uses.
    #[test]
    fn unset_targets_plan_exactly_what_the_step_planner_plans() {
        for seed in 0..300u64 {
            let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mut next = move || {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            };
            let d = (next() % 21) as u32;
            let n = (next() % 8) as u32;
            let cands: Vec<SloCandidate> = (0..n)
                .map(|i| {
                    let rows = [128, 512, 1024, 2048, 8192][(next() % 5) as usize];
                    let planned = next() % 4 != 0;
                    let mut c = cand(d + i, next() % 50, 4096, if planned { rows } else { u32::MAX }, 9000, 0.0);
                    c.base.planned = planned;
                    c.base.packable = next() % 2 == 0;
                    c.base.span.program = 3;
                    c
                })
                .collect();
            let cap = [u32::MAX, 8192, 4096, 2048][(next() % 4) as usize];
            let t = Tick { cap_rows: cap, packing: true, rotate: next() % 2 == 0, turn: (next() % 20) as usize, slots: 32 };
            let rungs = |p: u32| (p == 3).then_some(2048);
            let base: Vec<Candidate> = cands.iter().map(|c| c.base).collect();
            let want = step::plan(amd(), t, 0..d, &base, rungs, |_| u32::MAX);
            let mut s = SloState::default();
            let (got, out) = plan_tick(Targets::default(), amd(), t, 0..d, &cands, &ladder(), &glm(), &mut s.skipped, rungs, |_| u32::MAX);
            assert_eq!(got, want, "seed={seed}");
            assert_eq!(out, SloOutcome::default());
            assert_eq!(s.skipped, 0);
        }
    }

    /// Under a target nothing is packed: packable candidates run as isolated launches, so a
    /// shrunk (possibly retargeted) chunk never meets `packed_span_admissible` at all.
    #[test]
    fn a_target_plans_isolated_launches_even_for_packable_candidates() {
        let mut cands: Vec<SloCandidate> = (0..4).map(|s| cand(s, u64::from(s), 4096, 512, 5000, 0.0)).collect();
        for c in &mut cands {
            c.base.packable = true;
            c.base.span.program = 3;
        }
        let mut s = SloState::default();
        let t = Targets { tbt_ms: Some(10_000.0), ttft_ms: None };
        let (plan, _) = plan_tick(t, amd(), tick(), 0..2, &cands, &ladder(), &glm(), &mut s.skipped, |p| (p == 3).then_some(2048), |_| u32::MAX);
        assert_eq!(plan.launches.len(), 4);
        assert!(plan.launches.iter().all(|l| !l.is_pack()));
        let base: Vec<Candidate> = cands.iter().map(|c| c.base).collect();
        let unset = step::plan(amd(), tick(), 0..2, &base, |p| (p == 3).then_some(2048), |_| u32::MAX);
        assert!(unset.launches[0].is_pack(), "the same candidates do pack without a target");
    }

    #[test]
    fn the_upper_quantile_tracks_p90_and_a_margin_shrinks_the_chunk() {
        let mut q = Quantile::new(1.0, 0.9, 0.02);
        let mut x = 12345u64;
        let mut above = 0;
        for i in 0..60_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let sample = 1.0 + (x >> 11) as f64 / (1u64 << 53) as f64;
            if i >= 10_000 && sample > q.get() {
                above += 1;
            }
            q.observe(sample);
        }
        // U[1, 2): p90 = 1.9. The tracker jitters by about eta * q around it; what the planner
        // relies on is the coverage, one sample in ten above the margin.
        assert!((q.get() - 1.9).abs() < 0.06, "p90 of U[1,2) is 1.9, got {}", q.get());
        let share = f64::from(above) / 50_000.0;
        assert!((share - 0.10).abs() < 0.01, "share above the tracked p90: {share}");

        struct Wide(Fake, f64);
        impl CostModel for Wide {
            fn launch_ms(&self, b: u32, r: u32, p: u32, f: bool) -> f64 {
                self.0.launch_ms(b, r, p, f)
            }
            fn decode_ms(&self, r: u32, a: bool) -> f64 {
                self.0.decode_ms(r, a)
            }
            fn host_ms(&self) -> f64 {
                0.0
            }
            fn margin(&self, _: TickClass) -> f64 {
                self.1
            }
        }
        let mut s = SloState::default();
        let c = [cand(3, 0, 65536, 8192, 20000, 0.0)];
        let wide = Wide(glm(), 1.25);
        let (plan, out) = plan_tick(TBT500, amd(), tick(), 0..19, &c, &ladder(), &wide, &mut s.skipped, |_| None, |_| u32::MAX);
        // 500 / 1.25 = 400: 290 ms of prefill -> 150 + 0.09 n <= 290 -> n = 1555 -> 1536.
        assert_eq!(rows(&plan), [(3, 1536)]);
        assert_eq!((out.class, out.margin), (TickClass::Plain, 1.25));
        assert!(out.predicted_ms * out.margin <= 500.0);
    }

    #[test]
    fn a_tick_is_classed_by_its_riskiest_launch() {
        let mut s = SloState::default();
        let t = Targets { tbt_ms: Some(2000.0), ttft_ms: None };
        let mut fresh = cand(1, 2, 0, u32::MAX, 9000, 0.0);
        fresh.base.planned = false;
        let cands = [cand(0, 1, 65536, 700, 700, 0.0), fresh];
        let (plan, out) = run(t, &cands, 5, &mut s);
        assert_eq!(plan.launches.len(), 2);
        assert_eq!(out.class, TickClass::Fresh);
        let (_, out) = run(t, &cands[..1], 5, &mut s);
        assert_eq!(out.class, TickClass::Completing);
    }

    #[test]
    fn rls_learns_a_new_machine_from_the_prior() {
        let mut c = TickCost::default();
        // A machine 10x faster than the prior at decode and prefill.
        for _ in 0..20 {
            c.observe_decode(20, false, 10.05);
        }
        let first = c.launch_ms(8192, 8192, 32768, false);
        assert!((first - 93.0).abs() < 15.0, "speed-scaled prior {first}");
        for i in 0..200u32 {
            let rows = 1024 + (i * 997) % 7168;
            let prior = (i * 4099) % 65536 + 1;
            let truth = 15.0 + 0.009 * f64::from(rows) + 0.0001 * f64::from(prior);
            c.observe_launch(8192, rows, prior, false, truth);
        }
        let got = c.launch_ms(8192, 3000, 40000, false);
        let truth = 15.0 + 27.0 + 4.0;
        assert!((got - truth).abs() / truth < 0.03, "{got} vs {truth}");
        assert!((c.decode_ms(20, false) - 10.05).abs() < 0.5);
    }

    /// Held-out validation of [`TickCost`] on recorded `PLOW_TICK_LOG` server logs:
    /// `PLOW_SLO_REPLAY_LOGS=a.log:b.log cargo test -p plowrt --lib replay -- --ignored --nocapture`.
    /// A fresh model per log; every tick is predicted before it is observed, the way the
    /// planner uses it. Reports |error| of ticks carrying a prefill launch and decode rows.
    #[test]
    #[ignore]
    fn replay_tick_logs() {
        let Ok(paths) = std::env::var("PLOW_SLO_REPLAY_LOGS") else { return };
        let kv = |s: &str| -> std::collections::HashMap<String, String> {
            s.split_whitespace()
                .filter_map(|w| w.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        };
        let f = |m: &std::collections::HashMap<String, String>, k: &str| -> f64 {
            m.get(k).and_then(|v| v.parse().ok()).unwrap_or(0.0)
        };
        let pct = |v: &mut Vec<f64>, q: f64| -> f64 {
            v.sort_by(f64::total_cmp);
            v.get(((v.len().max(1) - 1) as f64 * q).round() as usize).copied().unwrap_or(f64::NAN)
        };
        let (mut all_mixed, mut all_decode) = (Vec::new(), Vec::new());
        let (mut covered, mut over) = (Vec::<f64>::new(), Vec::new());
        let mut by_class = [(0u32, 0u32); 3];
        let mut margins: Vec<[f64; 3]> = Vec::new();
        for path in paths.split(':') {
            let text = std::fs::read_to_string(path).expect("log");
            let mut cost = TickCost::default();
            let mut launches: Vec<(u32, u32, u32, f64, bool)> = Vec::new();
            let mut mixed = Vec::new();
            for line in text.lines() {
                if let Some(i) = line.find("PFCHUNK ") {
                    let m = kv(&line[i..]);
                    let last = m.get("last").is_some_and(|v| v == "true");
                    launches.push((f(&m, "bucket") as u32, f(&m, "clen") as u32, f(&m, "c0") as u32, f(&m, "total"), last));
                } else if let Some(i) = line.find("TICK n=") {
                    let m = kv(&line[i..]);
                    let (total, pf, dec, rows) = (f(&m, "total"), f(&m, "pf"), f(&m, "dec"), f(&m, "dec_rows") as u32);
                    let after = !launches.is_empty();
                    if rows > 0 {
                        let pred = launches
                            .iter()
                            .map(|&(b, clen, c0, _, _)| cost.launch_ms(b, clen, c0, c0 == 0))
                            .sum::<f64>()
                            + cost.decode_ms(rows, after)
                            + cost.host_ms();
                        let err = ((pred - total) / total).abs();
                        if after {
                            let class = if launches.iter().any(|l| l.2 == 0) {
                                TickClass::Fresh
                            } else if launches.iter().any(|l| l.4) {
                                TickClass::Completing
                            } else {
                                TickClass::Plain
                            };
                            let planned = pred * cost.margin(class);
                            covered.push(f64::from(u8::from(total <= planned)));
                            by_class[class as usize].0 += 1;
                            by_class[class as usize].1 += u32::from(total <= planned);
                            over.push(((total - planned) / planned).max(0.0));
                            cost.observe_margin(class, total / pred);
                            mixed.push(err)
                        } else {
                            all_decode.push(err)
                        }
                    }
                    for &(b, clen, c0, ms, _) in &launches {
                        cost.observe_launch(b, clen, c0, c0 == 0, ms);
                    }
                    if rows > 0 {
                        cost.observe_decode(rows, after, dec);
                    }
                    cost.observe_host(total - pf - dec);
                    launches.clear();
                }
            }
            margins.push([TickClass::Plain, TickClass::Completing, TickClass::Fresh].map(|c| cost.margin(c)));
            let n = mixed.len();
            println!("{path}: {n} prefill+decode ticks, |err| p50 {:.1}% p90 {:.1}%", 100.0 * pct(&mut mixed, 0.5), 100.0 * pct(&mut mixed, 0.9));
            all_mixed.extend(mixed);
        }
        let (nm, nd) = (all_mixed.len(), all_decode.len());
        println!(
            "held out: {nm} prefill+decode ticks |err| p50 {:.1}% p90 {:.1}% p99 {:.1}%; {nd} decode-only ticks p50 {:.1}% p90 {:.1}%",
            100.0 * pct(&mut all_mixed, 0.5),
            100.0 * pct(&mut all_mixed, 0.9),
            100.0 * pct(&mut all_mixed, 0.99),
            100.0 * pct(&mut all_decode, 0.5),
            100.0 * pct(&mut all_decode, 0.9),
        );
        let n = covered.len().max(1) as f64;
        println!(
            "planned against the per-class p90 margin: {:.1}% of prefill+decode ticks ran at or under the plan; \
             overrun of the rest p50 {:.1}% p99 {:.1}%",
            100.0 * covered.iter().sum::<f64>() / n,
            100.0 * pct(&mut over.iter().copied().filter(|&o| o > 0.0).collect(), 0.5),
            100.0 * pct(&mut over.iter().copied().filter(|&o| o > 0.0).collect(), 0.99),
        );
        for (i, name) in ["plain", "completing", "fresh"].iter().enumerate() {
            let (n, ok) = by_class[i];
            let mut end: Vec<f64> = margins.iter().map(|m| m[i]).collect();
            println!(
                "  {name}: {n} ticks, {:.1}% at or under plan; end-of-log margin p50 {:.3} max {:.3}",
                100.0 * f64::from(ok) / f64::from(n.max(1)),
                pct(&mut end, 0.5),
                end.iter().copied().fold(0.0, f64::max),
            );
        }
        assert!(pct(&mut all_mixed, 0.5) < 0.10);
    }

    /// A model projection, not a measurement: C20, 20 prompts of 70000 +-14% (seeded), 700 output
    /// tokens each, all arriving at t = 0, served by `plan_tick` on the GLM ladder with "truth"
    /// costs from a `TickCost` that has replayed `PLOW_SLO_SIM_LOGS` (colon-separated TICK logs;
    /// priors alone when unset). The planner sees the truth exactly (no model error, margin 1),
    /// so the SLO arms are an optimistic bound. Targets from `PLOW_SLO_SIM_TARGETS` (ms, 0 = none).
    /// `cargo test -p plowrt --lib simulate_c20 -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn simulate_c20() {
        let mut truth = TickCost::default();
        if let Ok(paths) = std::env::var("PLOW_SLO_SIM_LOGS") {
            for path in paths.split(':') {
                let text = std::fs::read_to_string(path).expect("log");
                let mut launches: Vec<(u32, u32, u32, f64)> = Vec::new();
                for line in text.lines() {
                    let kv = |s: &str, k: &str| -> f64 {
                        s.split_whitespace()
                            .find_map(|w| w.strip_prefix(&format!("{k}=")[..]))
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0.0)
                    };
                    if let Some(i) = line.find("PFCHUNK ") {
                        let l = &line[i..];
                        launches.push((kv(l, "bucket") as u32, kv(l, "clen") as u32, kv(l, "c0") as u32, kv(l, "total")));
                    } else if let Some(i) = line.find("TICK n=") {
                        let l = &line[i..];
                        let rows = kv(l, "dec_rows") as u32;
                        for &(b, clen, c0, ms) in &launches {
                            truth.observe_launch(b, clen, c0, c0 == 0, ms);
                        }
                        if rows > 0 {
                            truth.observe_decode(rows, !launches.is_empty(), kv(l, "dec"));
                        }
                        launches.clear();
                    }
                }
            }
        }
        let targets: Vec<f64> = std::env::var("PLOW_SLO_SIM_TARGETS")
            .unwrap_or_else(|_| "0,500,250".into())
            .split(',')
            .map(|t| t.parse().unwrap())
            .collect();
        let mut x = 7u64;
        let lens: Vec<u32> = (0..20)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                60_200 + ((x >> 33) % 19_601) as u32
            })
            .collect();
        const OUT: u32 = 700;
        let l = ladder();
        for &target in &targets {
            let t = Targets { tbt_ms: (target > 0.0).then_some(target), ttft_ms: None };
            // (frontier, planned step end, planned, tokens out, first token time, itls)
            let mut reqs: Vec<(u32, u32, bool, u32, f64, Vec<f64>)> =
                lens.iter().map(|_| (0, 0, false, 0, 0.0, Vec::new())).collect();
            let (mut now, mut skipped, mut ticks) = (0.0f64, 0u32, 0u32);
            while reqs.iter().any(|r| r.3 < OUT) {
                ticks += 1;
                assert!(ticks < 2_000_000, "simulation did not finish");
                let decodes: Vec<u32> =
                    (0..20).filter(|&i| reqs[i].0 == lens[i] && reqs[i].3 < OUT).map(|i| i as u32).collect();
                let cands: Vec<SloCandidate> = (0..20)
                    .filter(|&i| reqs[i].0 < lens[i])
                    .map(|i| {
                        let r = &reqs[i];
                        let rows = if r.2 { r.1 - r.0 } else { u32::MAX };
                        let mut c = cand(i as u32, i as u64, r.0, rows, lens[i] - r.0, now);
                        c.base.planned = r.2;
                        c
                    })
                    .collect();
                let (plan, _) = plan_tick(t, amd(), tick(), decodes.iter().copied(), &cands, &l, &truth, &mut skipped, |_| None, |_| u32::MAX);
                let mut dt = 0.0;
                let mut finished = Vec::new();
                for launch in &plan.launches {
                    let i = launch.spans[0].slot as usize;
                    let r = &mut reqs[i];
                    let fresh = !r.2;
                    if fresh {
                        r.2 = true;
                        r.1 = (r.0 + 8192).min(lens[i]);
                    }
                    let rows = launch.spans[0].n_rows.min(r.1 - r.0);
                    let bucket = l.bucket_for(rows, r.0).unwrap_or(8192);
                    dt += truth.launch_ms(bucket, rows, r.0, fresh);
                    r.0 += rows;
                    if r.0 == r.1 && r.0 < lens[i] {
                        r.1 = (r.0 + 8192).min(lens[i]);
                    }
                    if r.0 == lens[i] {
                        finished.push(i);
                    }
                }
                if !decodes.is_empty() {
                    dt += truth.decode_ms(decodes.len() as u32, !plan.launches.is_empty());
                }
                now += dt;
                for &i in &decodes {
                    reqs[i as usize].3 += 1;
                    reqs[i as usize].5.push(dt);
                }
                for i in finished {
                    reqs[i].3 = 1;
                    reqs[i].4 = now;
                }
            }
            let tok_s = f64::from(20 * OUT) / (now / 1e3);
            let mut ttft: Vec<f64> = reqs.iter().map(|r| r.4 / 1e3).collect();
            let mut itl: Vec<f64> = reqs.iter().flat_map(|r| r.5.iter().copied()).collect();
            let met = reqs
                .iter()
                .filter(|r| {
                    let mut v = r.5.clone();
                    target <= 0.0 || v.is_empty() || {
                        v.sort_by(f64::total_cmp);
                        v[((v.len() - 1) as f64 * 0.99) as usize] <= target
                    }
                })
                .count();
            let pct = |v: &mut Vec<f64>, q: f64| {
                v.sort_by(f64::total_cmp);
                v[((v.len() - 1) as f64 * q) as usize]
            };
            println!(
                "target {target:>5.0} ms: {:.1} s, {tok_s:.2} out tok/s, TTFT p50 {:.1} s p99 {:.1} s, ITL p50 {:.0} p99 {:.0} ms, requests with p99 ITL under target {met}/20, ticks {ticks}",
                now / 1e3,
                pct(&mut ttft, 0.5),
                pct(&mut ttft, 0.99),
                pct(&mut itl, 0.5),
                pct(&mut itl, 0.99),
            );
        }
    }

    #[test]
    fn the_yardstick_counts_launches_and_decode_passes() {
        let ticks = [(1, 19), (1, 0), (0, 20), (2, 20)];
        assert!((seconds_removed(&ticks, 25.0, 10.0) - (4.0 * 25.0 + 3.0 * 10.0) / 1e3).abs() < 1e-12);
    }
}
