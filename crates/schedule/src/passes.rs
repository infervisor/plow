//! Scheduler passes A–G over the expanded [`TaskGraph`] (design §4, §7, §8).
//!
//! Placement (A) is folded into the list schedule (C): when a ready task is
//! popped, it is assigned to the resource of its class giving the earliest
//! feasible start. Liveness (B) falls out of start/duration. Counters (D/E) are
//! clustered per producer→consumer op boundary and scoped by placement.
//! Prefetch (F) is emergent — the dependency-respecting schedule already issues
//! a DMA-in before its consumer (§7.6). Packets (G) are the per-resource lowering.

use crate::expand::{Task, TaskGraph, TaskId, TaskKind};
use crate::interval::Cycle;
use crate::machine::Machine;
use crate::resource::{PagePool, ResourceId, ResourceState};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet};

pub use packet::Scope;

/// A clustered dependency counter.
#[derive(Clone, Debug)]
pub struct Counter {
    pub id: usize,
    pub threshold: u32,
    pub scope: Scope,
    pub producer_node: usize,
    pub consumer_node: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketKind {
    Compute,
    TmaIn,
    TmaOut,
    Rdma,
    HostCoord,
}

/// A runtime packet: one scheduled task lowered for its resource (Pass G).
#[derive(Clone, Debug)]
pub struct Packet {
    pub task: TaskId,
    pub op: String,
    pub kind: PacketKind,
    pub start: Cycle,
    /// Counters this packet waits on before issuing.
    pub wait: Vec<usize>,
    /// Counters this packet increments on completion.
    pub successors: Vec<usize>,
}

/// The scheduler's output.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub streams: HashMap<ResourceId, Vec<(TaskId, Cycle)>>,
    pub packets: HashMap<ResourceId, Vec<Packet>>,
    pub counters: Vec<Counter>,
    pub placement: HashMap<TaskId, ResourceId>,
    pub starts: Vec<Cycle>,
    /// Per-tile SRAM page-slot assignment (which pages of its SM it holds over
    /// its live interval). Absent ⇒ spilled (didn't fit the per-SM page pool).
    pub sram_slots: HashMap<TaskId, Vec<usize>>,
    /// Number of tiles that could not be assigned pages (spills).
    pub spills: usize,
    /// Per-tile TMEM accumulator column assignment (Blackwell `tcgen05`). Empty
    /// on architectures without TMEM.
    pub tmem_slots: HashMap<TaskId, Vec<usize>>,
    /// Tiles whose accumulator did not fit TMEM (a tile-shape-too-big signal).
    pub tmem_spills: usize,
    pub makespan: Cycle,
}

/// Ready-queue entry. `Ord` encodes the priority + deterministic tie-breakers:
/// critical-path length, then out-degree (mobility), then shorter duration, then
/// lowest task id.
struct Prioritized {
    cp: u64,
    succ: usize,
    dur: Cycle,
    task: TaskId,
}
impl PartialEq for Prioritized {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Prioritized {}
impl Ord for Prioritized {
    fn cmp(&self, o: &Self) -> Ordering {
        self.cp
            .cmp(&o.cp)
            .then(self.succ.cmp(&o.succ))
            .then(o.dur.cmp(&self.dur)) // shorter duration first
            .then(o.task.cmp(&self.task)) // lower id first
    }
}
impl PartialOrd for Prioritized {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Kahn topological order of the task graph.
fn topo(tg: &TaskGraph, succ: &[Vec<usize>], indeg0: &[u32]) -> Vec<TaskId> {
    let mut indeg = indeg0.to_vec();
    let mut q: Vec<usize> = (0..tg.tasks.len()).filter(|&i| indeg[i] == 0).collect();
    let mut out = Vec::with_capacity(tg.tasks.len());
    while let Some(t) = q.pop() {
        out.push(t);
        for &s in &succ[t] {
            indeg[s] -= 1;
            if indeg[s] == 0 {
                q.push(s);
            }
        }
    }
    out
}

/// Longest-path-to-sink (critical path) per task, weighted by duration.
fn critical_path(tg: &TaskGraph, succ: &[Vec<usize>], order: &[TaskId]) -> Vec<u64> {
    let mut cp = vec![0u64; tg.tasks.len()];
    for &t in order.iter().rev() {
        let best = succ[t].iter().map(|&s| cp[s]).max().unwrap_or(0);
        cp[t] = tg.tasks[t].dur + best;
    }
    cp
}

/// Pass A+C — place every task and order it on its resource.
pub fn list_schedule(
    machine: &Machine,
    tg: &TaskGraph,
    colocated_nodes: &HashSet<usize>,
    colo_pin: &HashMap<usize, ColoPin>,
    domain_pin: &HashMap<usize, usize>,
    counters: &[Counter],
    wait_of: &[Vec<usize>],
    succ_ctr_of: &[Vec<usize>],
) -> Schedule {
    let n = tg.tasks.len();
    let mut succ = vec![Vec::new(); n];
    let mut preds = vec![Vec::new(); n];
    let mut indeg = vec![0u32; n];
    for &(a, b) in &tg.edges {
        succ[a].push(b);
        preds[b].push(a);
        indeg[b] += 1;
    }
    let order = topo(tg, &succ, &indeg);
    let cp = critical_path(tg, &succ, &order);

    let mut res = ResourceState::new(machine);
    let mut start = vec![0u64; n];
    let mut placement: HashMap<TaskId, ResourceId> = HashMap::new();
    let mut streams: HashMap<ResourceId, Vec<(TaskId, Cycle)>> = HashMap::new();
    // Per-task locality-domain assignment (XCD on MI300X, GPC on H100/Blackwell).
    let mut task_domain: Vec<Option<usize>> = vec![None; n];

    let mut indeg_rt = indeg.clone();
    let mut ready: BinaryHeap<Prioritized> = (0..n)
        .filter(|&i| indeg_rt[i] == 0)
        .map(|i| Prioritized {
            cp: cp[i],
            succ: succ[i].len(),
            dur: tg.tasks[i].dur,
            task: i,
        })
        .collect();

    while let Some(p) = ready.pop() {
        let t = p.task;
        let task = &tg.tasks[t];
        let dep_ready = preds[t]
            .iter()
            .map(|&pr| start[pr] + tg.tasks[pr].dur)
            .max()
            .unwrap_or(0);
        // Compute the preferred locality domain from predecessors (majority vote).
        let preferred_domain = pred_locality_hint(machine, &preds[t], &task_domain, task);
        let (r, s) = choose_resource(
            machine,
            &res,
            task,
            dep_ready,
            colocated_nodes,
            colo_pin,
            domain_pin,
            preferred_domain,
        );
        reserve(&mut res, r, task, s);
        start[t] = s;
        placement.insert(t, r);
        // Track locality domain assignment for successor affinity.
        if let ResourceId::Sm(u, sm) = r {
            task_domain[t] = Some(machine.locality_domain_of(u, sm));
        }
        streams.entry(r).or_default().push((t, s));
        for &c in &succ[t] {
            indeg_rt[c] -= 1;
            if indeg_rt[c] == 0 {
                ready.push(Prioritized {
                    cp: cp[c],
                    succ: succ[c].len(),
                    dur: tg.tasks[c].dur,
                    task: c,
                });
            }
        }
    }

    // Pass F (prefetch is emergent) — keep each stream in issue (start) order.
    for v in streams.values_mut() {
        v.sort_by_key(|&(task, st)| (st, task));
    }
    let makespan = (0..n)
        .map(|t| start[t] + tg.tasks[t].dur)
        .max()
        .unwrap_or(0);

    // Pass B — per-slot on-chip-memory allocation. SRAM pages over the resident
    // output's live interval; TMEM accumulator columns over the compute interval.
    let (sram_slots, spills) = allocate_sram(tg, &succ, &start, &placement);
    let (tmem_slots, tmem_spills) = allocate_tmem(tg, &start, &placement);

    let packets = build_packets(&streams, tg, wait_of, succ_ctr_of);
    Schedule {
        streams,
        packets,
        counters: counters.to_vec(),
        placement,
        starts: start,
        sram_slots,
        spills,
        tmem_slots,
        tmem_spills,
        makespan,
    }
}

/// Pass B (TMEM) — assign each matmul tile its MMA-accumulator columns in its
/// SM's Tensor Memory, held over the compute interval (the accumulator lives
/// from the first MMA through the epilogue). No-op where TMEM is absent.
///
/// Capacities come from `tg.tmem_cols_per_sm` (captured at expansion) so
/// post-schedule passes can re-run this without the `Machine`.
pub(crate) fn allocate_tmem(
    tg: &TaskGraph,
    start: &[Cycle],
    placement: &HashMap<TaskId, ResourceId>,
) -> (HashMap<TaskId, Vec<usize>>, usize) {
    let mut by_sm: HashMap<(usize, usize), Vec<TaskId>> = HashMap::new();
    for (&t, &r) in placement {
        if let ResourceId::Sm(u, sm) = r {
            if tg.tasks[t].tmem_cols > 0 {
                by_sm.entry((u, sm)).or_default().push(t);
            }
        }
    }
    let mut slots = HashMap::new();
    let mut spills = 0;
    for ((u, _sm), mut tasks) in by_sm {
        tasks.sort_by_key(|&t| (start[t], t));
        let mut pool = PagePool::new(tg.tmem_cols_per_sm.get(u).copied().unwrap_or(0));
        for t in tasks {
            let s = start[t];
            let e = s + tg.tasks[t].dur.max(1);
            match pool.allocate(s, e, tg.tasks[t].tmem_cols) {
                Some(sl) => {
                    slots.insert(t, sl);
                }
                None => spills += 1,
            }
        }
    }
    (slots, spills)
}

/// Pass B — assign each compute tile concrete SRAM page slots on its SM over its
/// live interval (production → last consumer). Linear-scan per SM page pool
/// (design §8.4); a tile that doesn't fit is a spill.
///
/// **Invariant:** Every data dependency must be represented as an edge in
/// `tg.edges`. The live interval `[start, last_consumer_end)` is computed
/// from successor edges — if a dependency exists only via a counter (not a
/// graph edge), the page may be freed prematurely.
///
/// Capacities come from `tg.pages_per_sm` (captured at expansion) so
/// post-schedule passes can re-run this without the `Machine`.
pub(crate) fn allocate_sram(
    tg: &TaskGraph,
    succ: &[Vec<usize>],
    start: &[Cycle],
    placement: &HashMap<TaskId, ResourceId>,
) -> (HashMap<TaskId, Vec<usize>>, usize) {
    // Compute tiles grouped by the SM they run on.
    let mut by_sm: HashMap<(usize, usize), Vec<TaskId>> = HashMap::new();
    for (&t, &r) in placement {
        if let ResourceId::Sm(u, sm) = r {
            if tg.tasks[t].kind == TaskKind::Compute
                && (tg.tasks[t].out_pages > 0 || tg.tasks[t].sram_pages > 0)
            {
                by_sm.entry((u, sm)).or_default().push(t);
            }
        }
    }
    let mut slots = HashMap::new();
    let mut spills = 0;
    for ((u, _sm), mut tasks) in by_sm {
        tasks.sort_by_key(|&t| (start[t], t));
        let mut pool = PagePool::new(tg.pages_per_sm.get(u).copied().unwrap_or(0));
        for t in tasks {
            let s = start[t];
            // Output pages live until the last consumer *ends* — a consumer
            // reads its input throughout [start, start+dur), so freeing at its
            // start would let the page be reallocated mid-read.
            let last = succ[t]
                .iter()
                .map(|&c| start[c] + tg.tasks[c].dur)
                .max()
                .unwrap_or(0);
            let end_live = last.max(s + tg.tasks[t].dur);
            // Working-set (A/B staging) pages are transient — only live during
            // the compute interval [s, s+dur).
            let end_compute = s + tg.tasks[t].dur;
            match pool.allocate_with_working_set(
                s,
                end_compute,
                tg.tasks[t].sram_pages,
                end_live,
                tg.tasks[t].out_pages,
            ) {
                Some(sl) => {
                    slots.insert(t, sl);
                }
                None => spills += 1,
            }
        }
    }
    (slots, spills)
}

/// How a colocated tile pins within its unit: the coupled row axis, that axis's
/// block size, and the colocation group's common band (largest block in the
/// group). Built from `ConstraintSet::domains`.
#[derive(Clone, Copy, Debug)]
pub struct ColoPin {
    pub axis: usize,
    pub block: i64,
    pub band: i64,
}

/// SM index a colocated tile pins to, so a producer tile and the consumer
/// tile(s) reading it share one SM (the SRAM hand-off).
///
/// With a [`ColoPin`], pin by the tile's ABSOLUTE position along its coupled row
/// axis (`coord[axis] * block`) bucketed into `band`-sized groups. This lands a
/// producer and the several smaller consumer tiles covering the same rows on one
/// SM even when their block sizes differ or the coupled axis is 1 (Flash).
/// Without a pin (no domain info), fall back to the legacy `coord[0]` keying.
fn colocated_sm(task: &Task, sm_count: usize, pin: Option<&ColoPin>) -> usize {
    let smc = sm_count.max(1);
    match pin {
        Some(p) => {
            let idx = task.coord.get(p.axis).copied().unwrap_or(0).max(0);
            let abs_row = idx.saturating_mul(p.block.max(1));
            ((abs_row / p.band.max(1)) as usize) % smc
        }
        None => {
            let c0 = task.coord.first().copied().unwrap_or(0).max(0) as usize;
            c0 % smc
        }
    }
}

/// Maximum cycles of slack allowed to prefer the same locality domain (XCD /
/// GPC) over the absolute-earliest SM. Keeps communicating tiles on the same
/// L2 partition without significantly hurting latency.
const LOCALITY_SLACK: Cycle = 50;

/// Derive the preferred locality domain from a task's predecessors. Returns
/// `None` when the unit has only one domain (no preference benefit) or no
/// predecessor has been placed on an SM yet.
///
/// "Locality domain" is the unified concept covering:
/// - **MI300X**: XCD (8 chiplets × 38 CUs sharing 4 MiB L2 each)
/// - **H100/Blackwell**: GPC (DSM domain — SMs sharing L2 partition slices)
/// - **Ada (monolithic, no DSM)**: single trivial domain (returns `None`)
fn pred_locality_hint(
    machine: &Machine,
    preds: &[TaskId],
    task_domain: &[Option<usize>],
    task: &Task,
) -> Option<usize> {
    if task.kind != TaskKind::Compute {
        return None;
    }
    let u = task.unit;
    let domain_count = machine.locality_domain_count(u);
    if domain_count <= 1 {
        return None;
    }
    // Majority vote: count how many predecessors landed on each domain.
    let mut counts = [0u32; 64]; // supports up to 64 domains
    let max_domain = domain_count.min(64);
    let mut total = 0u32;
    for &p in preds {
        if let Some(d) = task_domain[p] {
            if d < max_domain {
                counts[d] += 1;
                total += 1;
            }
        }
    }
    if total == 0 {
        return None;
    }
    let best = counts[..max_domain]
        .iter()
        .enumerate()
        .max_by_key(|&(_, &cnt)| cnt)
        .map(|(i, _)| i)
        .unwrap();
    Some(best)
}

/// Candidate resources for a task's class, and the earliest feasible start +
/// chosen resource (Pass A routing + the list scheduler's resource query).
///
/// `preferred_domain` is the soft locality-domain hint from predecessor
/// placement; when `Some(d)`, the scheduler prefers SMs on domain `d` within
/// `LOCALITY_SLACK` cycles of the absolute-earliest start.
fn choose_resource(
    machine: &Machine,
    res: &ResourceState,
    task: &Task,
    after: Cycle,
    colocated: &HashSet<usize>,
    colo_pin: &HashMap<usize, ColoPin>,
    domain_pin: &HashMap<usize, usize>,
    preferred_domain: Option<usize>,
) -> (ResourceId, Cycle) {
    let candidates: Vec<ResourceId> = match task.kind {
        TaskKind::Compute => {
            let u = task.unit;
            let smc = machine.unit(u).sm_count.max(1);
            if colocated.contains(&task.node) {
                // MustColocate: one SM, by coupled-row band.
                vec![ResourceId::Sm(u, colocated_sm(task, smc, colo_pin.get(&task.node)))]
            } else if let Some(&d) = domain_pin.get(&task.node) {
                // SameDomain (DSM) or SameL2Partition — both pin to a locality
                // domain (GPC on H100, XCD on MI300). Use the unified helper
                // that respects whichever partitioning the arch exposes.
                machine
                    .locality_domain_sms(u, d)
                    .map(|s| ResourceId::Sm(u, s))
                    .collect()
            } else {
                (0..smc).map(|s| ResourceId::Sm(u, s)).collect()
            }
        }
        TaskKind::DmaIn | TaskKind::DmaOut if task.cross_unit => {
            (0..machine.dpu_engines).map(ResourceId::Dpu).collect()
        }
        TaskKind::DmaIn | TaskKind::DmaOut => (0..machine.unit(task.unit).dma_engines)
            .map(|e| ResourceId::Dma(task.unit, e))
            .collect(),
        TaskKind::Host => (0..machine.host_threads).map(ResourceId::Host).collect(),
    };

    // Compute (resource, feasible_start) for each candidate.
    let mut scored: Vec<(ResourceId, Cycle)> = candidates
        .into_iter()
        .map(|r| (r, feasible_start(machine, res, r, task, after)))
        .collect();
    scored.sort_by_key(|&(r, s)| (s, resource_key(r)));

    // Without a locality hint, pick the absolute earliest (original behavior).
    let Some(pref) = preferred_domain else {
        return scored[0];
    };

    let best_start = scored[0].1;
    // Among candidates within LOCALITY_SLACK of the best, prefer same domain.
    scored
        .iter()
        .filter(|&&(_, s)| s <= best_start + LOCALITY_SLACK)
        .min_by_key(|&&(r, s)| {
            let on_pref = match r {
                ResourceId::Sm(u, sm) => machine.locality_domain_of(u, sm) == pref,
                _ => false,
            };
            (!on_pref, s, resource_key(r))
        })
        .copied()
        .unwrap_or(scored[0])
}

fn resource_key(r: ResourceId) -> (u8, usize, usize) {
    match r {
        ResourceId::Sm(u, s) => (0, u, s),
        ResourceId::Dma(u, e) => (1, u, e),
        ResourceId::Dpu(i) => (2, 0, i),
        ResourceId::Host(i) => (3, 0, i),
    }
}

/// Earliest start ≥ `after` where the exclusive hold fits and any capacity
/// (HBM / interconnect) stays under limit.
fn feasible_start(
    machine: &Machine,
    res: &ResourceState,
    r: ResourceId,
    task: &Task,
    after: Cycle,
) -> Cycle {
    let dur = task.dur.max(1);
    let mut s = res.earliest_free(r, after, dur);
    if task.bytes == 0 {
        return s;
    }
    let w = task.bytes as f64 / dur as f64;
    let limit = match r {
        ResourceId::Dma(u, _) => machine.unit(u).hbm_bytes_per_cycle,
        ResourceId::Dpu(_) => machine.link_bytes_per_cycle,
        // Compute tasks carry folded transfer bytes (dma-fold prologue loads and
        // collapsed-mode boundary stores; see `expand.rs`) the kernel issues via
        // TMA. Those draw the unit's HBM controller during the compute window, so
        // bound them against the same per-unit HBM budget as DMA-engine transfers
        // — otherwise the schedule silently oversubscribes HBM and understates the
        // makespan (F5). The simulator honors the resulting stalls via the
        // scheduled-start floor.
        ResourceId::Sm(u, _) => machine.unit(u).hbm_bytes_per_cycle,
        // No capacity constraint on this resource class.
        _ => return s,
    };
    if w > limit + 1e-9 {
        // The task alone exceeds the capacity — no start is ever feasible
        // (duration/limit model mismatch). Place it at the earliest exclusive
        // slot, but say so.
        eprintln!(
            "[schedule] warning: task '{}' needs {w:.2} B/cycle on {r:?} but the \
             capacity limit is {limit:.2}; reserving over capacity at cycle {s}",
            task.op
        );
        return s;
    }
    let cap_ok = |s: Cycle| match r {
        ResourceId::Dma(u, _) | ResourceId::Sm(u, _) => res.hbm_ok(u, s, s + dur, w, limit),
        ResourceId::Dpu(_) => res.link_ok(s, s + dur, w, limit),
        _ => true,
    };
    // Fine probes: step to the next exclusive-free slot after each congested
    // window.
    for _ in 0..64 {
        if cap_ok(s) {
            return s;
        }
        s = res.earliest_free(r, s + dur, dur);
    }
    // Coarse probes: exponentially widening strides. Capacity reservations are
    // finite, so past the last one the window is empty; 32 doublings put the
    // bound at `dur << 32` cycles — beyond any real makespan horizon.
    let mut stride = dur;
    for _ in 0..32 {
        s = res.earliest_free(r, s.saturating_add(stride), dur);
        if cap_ok(s) {
            return s;
        }
        stride = stride.saturating_mul(2);
    }
    eprintln!(
        "[schedule] warning: no bandwidth-feasible start found for task '{}' on \
         {r:?}; reserving over capacity at cycle {s}",
        task.op
    );
    s
}

fn reserve(res: &mut ResourceState, r: ResourceId, task: &Task, s: Cycle) {
    let dur = task.dur.max(1);
    res.reserve(r, s, dur);
    // SRAM pages are assigned per-slot in a later pass (`allocate_sram`); here we
    // only reserve the exclusive resource + any bandwidth capacity.
    match r {
        // Folded compute-task bytes share the unit's HBM budget with DMA-engine
        // transfers (see `feasible_start`), so reserve them the same way.
        ResourceId::Dma(u, _) | ResourceId::Sm(u, _) if task.bytes > 0 => {
            res.reserve_hbm(u, s, s + dur, task.bytes as f64 / dur as f64)
        }
        ResourceId::Dpu(_) if task.bytes > 0 => {
            res.reserve_link(s, s + dur, task.bytes as f64 / dur as f64)
        }
        _ => {}
    }
}

/// Per-unit peak HBM bandwidth demand of a finished schedule.
#[derive(Clone, Debug, Default)]
pub struct HbmAudit {
    /// `(unit, peak_bytes_per_cycle, capacity_bytes_per_cycle)`, sorted by unit.
    pub per_unit: Vec<(usize, f64, f64)>,
}

impl HbmAudit {
    /// Units whose peak concurrent HBM demand exceeds capacity.
    pub fn oversubscribed(&self) -> Vec<(usize, f64, f64)> {
        self.per_unit
            .iter()
            .copied()
            .filter(|&(_, peak, cap)| peak > cap + 1e-9)
            .collect()
    }
}

/// Audit a schedule's peak per-unit HBM bandwidth demand, **including the folded
/// transfer bytes carried on compute tasks** (dma-fold prologue loads and
/// collapsed-mode boundary stores; see `expand.rs`). The list scheduler reserves
/// HBM capacity only for separate `DmaIn`/`DmaOut` tasks (see `reserve`), so
/// folded compute bytes escape its capacity check; this post-schedule sweep
/// accounts for them so an oversubscribed schedule is surfaced rather than
/// silently emitted.
///
/// Sweep-line per unit over each HBM-touching task's active interval
/// `[start, start+dur)` at rate `bytes/dur`, taking the peak concurrent sum.
/// Purely diagnostic — it does not change task timing (a bandwidth-accurate
/// makespan needs the bandwidth-aware simulator; see `sim.rs`).
pub fn hbm_bandwidth_audit(tasks: &TaskGraph, sched: &Schedule, machine: &Machine) -> HbmAudit {
    // unit -> (cycle, delta_rate) events. Half-open intervals: at a shared cycle,
    // ends (negative deltas sort first) retire before starts, so a load ending at
    // cycle c does not count as concurrent with one starting at c.
    let mut events: HashMap<usize, Vec<(Cycle, f64)>> = HashMap::new();
    for (tid, task) in tasks.tasks.iter().enumerate() {
        // Cross-unit transfers ride the interconnect (DPU/link), not HBM.
        if task.bytes == 0 || task.cross_unit {
            continue;
        }
        if !matches!(
            task.kind,
            TaskKind::DmaIn | TaskKind::DmaOut | TaskKind::Compute
        ) {
            continue;
        }
        let dur = task.dur.max(1);
        let start = sched.starts.get(tid).copied().unwrap_or(0);
        let rate = task.bytes as f64 / dur as f64;
        let ev = events.entry(task.unit).or_default();
        ev.push((start, rate));
        ev.push((start + dur, -rate));
    }
    let mut per_unit = Vec::new();
    for (unit, mut evs) in events {
        evs.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        let mut cur = 0.0f64;
        let mut peak = 0.0f64;
        for (_, d) in evs {
            cur += d;
            if cur > peak {
                peak = cur;
            }
        }
        per_unit.push((unit, peak, machine.unit(unit).hbm_bytes_per_cycle));
    }
    per_unit.sort_by_key(|&(u, _, _)| u);
    HbmAudit { per_unit }
}

// --- Pass D/E: clustered, placement-scoped counters --------------------------

/// Build dependency counters from the task-graph edges. Returns the counters
/// plus per-task wait / successor counter lists.
///
/// ### `ClusterMode::Coarse`
/// One counter per cross-op `(producer_node, consumer_node)` boundary,
/// threshold = producer tile count. Simple but creates all-wait-all semantics
/// that can deadlock when consumers and producers share an SM.
///
/// ### `ClusterMode::Fine`
/// One counter **per consumer tile** on cross-op boundaries: each consumer tile
/// waits only on its specific producer tiles (threshold = that tile's in-degree
/// from the task graph's fine edges). Falls back to coarse for boundaries whose
/// edges are all-to-all (no fine structure). Maximum pipelining and deadlock-free.
///
/// ### Intra-op boundaries (`pn == cn`)
/// Always per-producer-task (both modes) — the op's staging chain
/// (`DMA-in → compute → DMA-out`) shares a node id, so coarse would self-depend.
///
/// `colocated` is the set of compute-node ids pinned to a single SM (via
/// colocation groups). Only intra-node counters whose node is colocated are
/// scoped `IntraSm`; other same-unit intra-node counters are `IntraGpu`.
pub fn build_counters(
    tg: &TaskGraph,
    units: &HashMap<usize, costmodel::UnitId>,
    colocated: &HashSet<usize>,
    mode: crate::config::ClusterMode,
) -> (Vec<Counter>, Vec<Vec<usize>>, Vec<Vec<usize>>) {
    // Group edges by (producer node, consumer node), preserving the edges so the
    // intra-node case can split per producer task.
    let mut groups: HashMap<(usize, usize), Vec<(TaskId, TaskId)>> = HashMap::new();
    for &(a, b) in &tg.edges {
        let (pn, cn) = (tg.tasks[a].node, tg.tasks[b].node);
        groups.entry((pn, cn)).or_default().push((a, b));
    }
    let mut keys: Vec<_> = groups.keys().copied().collect();
    keys.sort_unstable();

    let mut counters = Vec::new();
    let mut wait_of = vec![Vec::new(); tg.tasks.len()];
    let mut succ_of = vec![Vec::new(); tg.tasks.len()];

    // Emit one counter: `producers` increment it, `consumers` wait on it.
    let emit = |counters: &mut Vec<Counter>,
                    wait_of: &mut [Vec<usize>],
                    succ_of: &mut [Vec<usize>],
                    producers: &[TaskId],
                    consumers: &[TaskId],
                    pn: usize,
                    cn: usize,
                    scope: Scope| {
        let id = counters.len();
        counters.push(Counter {
            id,
            threshold: producers.len() as u32,
            scope,
            producer_node: pn,
            consumer_node: cn,
        });
        for &p in producers {
            succ_of[p].push(id);
        }
        for &c in consumers {
            wait_of[c].push(id);
        }
    };

    let use_fine = matches!(mode, crate::config::ClusterMode::Fine);

    for (pn, cn) in keys {
        let edges = &groups[&(pn, cn)];
        let scope = scope_of(pn, cn, units, colocated);
        if pn != cn {
            if use_fine {
                // Fine mode: one counter per consumer tile.
                // Group edges by consumer task — each consumer gets its own counter
                // with threshold = the number of distinct producer tiles feeding it.
                let mut by_consumer: BTreeMap<TaskId, Vec<TaskId>> = BTreeMap::new();
                for &(a, b) in edges {
                    by_consumer.entry(b).or_default().push(a);
                }

                // Detect all-to-all pattern: every consumer has the same full
                // producer set ⇒ fine doesn't help; fall back to coarse to avoid
                // counter explosion while still being correct (the scheduler
                // awareness check handles the placement constraint).
                let all_producers: HashSet<TaskId> =
                    edges.iter().map(|&(a, _)| a).collect();
                let is_all_to_all = by_consumer.len() > 1
                    && by_consumer.values().all(|ps| {
                        let pset: HashSet<TaskId> = ps.iter().copied().collect();
                        pset == all_producers
                    });

                if is_all_to_all {
                    // Coarse fallback for all-to-all boundaries.
                    let mut producers: Vec<TaskId> = all_producers.into_iter().collect();
                    producers.sort_unstable();
                    let mut consumers: Vec<TaskId> =
                        by_consumer.keys().copied().collect();
                    consumers.sort_unstable();
                    emit(
                        &mut counters, &mut wait_of, &mut succ_of,
                        &producers, &consumers, pn, cn, scope,
                    );
                } else {
                    // Fine: one counter per consumer tile.
                    for (c, mut producers) in by_consumer {
                        producers.sort_unstable();
                        producers.dedup();
                        emit(
                            &mut counters, &mut wait_of, &mut succ_of,
                            &producers, &[c], pn, cn, scope,
                        );
                    }
                }
            } else {
                // Coarse mode: one counter for the whole boundary.
                let mut producers: Vec<TaskId> = edges.iter().map(|&(a, _)| a).collect();
                let mut consumers: Vec<TaskId> = edges.iter().map(|&(_, b)| b).collect();
                producers.sort_unstable();
                producers.dedup();
                consumers.sort_unstable();
                consumers.dedup();
                emit(
                    &mut counters, &mut wait_of, &mut succ_of,
                    &producers, &consumers, pn, cn, scope,
                );
            }
        } else {
            // Intra-op staging: one counter per producer task (wait/inc disjoint).
            let mut by_producer: BTreeMap<TaskId, Vec<TaskId>> = BTreeMap::new();
            for &(a, b) in edges {
                by_producer.entry(a).or_default().push(b);
            }
            for (p, mut consumers) in by_producer {
                consumers.sort_unstable();
                consumers.dedup();
                emit(&mut counters, &mut wait_of, &mut succ_of, &[p], &consumers, pn, cn, scope);
            }
        }
    }
    (counters, wait_of, succ_of)
}

fn scope_of(
    pn: usize,
    cn: usize,
    units: &HashMap<usize, costmodel::UnitId>,
    colocated: &HashSet<usize>,
) -> Scope {
    match (units.get(&pn), units.get(&cn)) {
        (Some(a), Some(b)) if a != b => Scope::CrossUnit,
        // IntraSm only when both producer and consumer are colocated (pinned to
        // one SM). Same-node but non-colocated tiles spread across many SMs.
        (Some(_), Some(_)) if pn == cn && colocated.contains(&pn) => Scope::IntraSm,
        _ => Scope::IntraGpu,
    }
}

// --- Pass G: packets ---------------------------------------------------------

fn build_packets(
    streams: &HashMap<ResourceId, Vec<(TaskId, Cycle)>>,
    tg: &TaskGraph,
    wait_of: &[Vec<usize>],
    succ_of: &[Vec<usize>],
) -> HashMap<ResourceId, Vec<Packet>> {
    let mut out: HashMap<ResourceId, Vec<Packet>> = HashMap::new();
    for (&r, items) in streams {
        let packets = items
            .iter()
            .map(|&(task, start)| Packet {
                task,
                op: tg.tasks[task].op.clone(),
                kind: packet_kind(r, tg.tasks[task].kind),
                start,
                wait: wait_of[task].clone(),
                successors: succ_of[task].clone(),
            })
            .collect();
        out.insert(r, packets);
    }
    out
}

fn packet_kind(r: ResourceId, k: TaskKind) -> PacketKind {
    match (r, k) {
        (ResourceId::Sm(..), _) => PacketKind::Compute,
        (ResourceId::Dpu(_), _) => PacketKind::Rdma,
        (ResourceId::Host(_), _) => PacketKind::HostCoord,
        (ResourceId::Dma(..), TaskKind::DmaOut) => PacketKind::TmaOut,
        (ResourceId::Dma(..), _) => PacketKind::TmaIn,
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;
    use crate::expand::Task;
    use crate::machine::UnitHw;

    fn task(kind: TaskKind, unit: usize, dur: Cycle, bytes: u64) -> Task {
        Task {
            node: 0,
            op: "t".into(),
            unit,
            kind,
            coord: vec![],
            dur,
            bytes,
            tensor_bytes: 0,
            sram_pages: 0,
            out_pages: 0,
            tmem_cols: 0,
            tensor: None,
            cross_unit: false,
        }
    }

    fn one_unit_machine(hbm_bytes_per_cycle: f64) -> Machine {
        Machine {
            units: vec![UnitHw {
                id: 0,
                sm_count: 1,
                pages_per_sm: 0,
                tmem_cols_per_sm: 0,
                dsm_domains: 1,
                sms_per_domain: 1,
                chiplet_count: 1,
                sms_per_chiplet: 1,
                l2_partitions: 1,
                sms_per_l2_partition: 1,
                dma_engines: 1,
                hbm_bytes_per_cycle,
            }],
            dpu_engines: 1,
            host_threads: 1,
            unified_memory: false,
            has_fast_interconnect: false,
            link_bytes_per_cycle: 1.0,
        }
    }

    fn sched_with_starts(starts: Vec<Cycle>) -> Schedule {
        Schedule {
            streams: HashMap::new(),
            packets: HashMap::new(),
            counters: vec![],
            placement: HashMap::new(),
            starts,
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 0,
        }
    }

    // The bug F5 targets: folded bytes on a compute task are real HBM traffic the
    // scheduler's `reserve` omits. Concurrent with a separate DMA-in they push the
    // unit over capacity, which the audit must surface.
    #[test]
    fn folded_compute_bytes_counted_against_hbm() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            task(TaskKind::Compute, 0, 10, 100), // 10 B/cyc folded
            task(TaskKind::DmaIn, 0, 10, 100),   // 10 B/cyc separate
        ];
        let sched = sched_with_starts(vec![0, 0]); // overlap [0,10)
        let audit = hbm_bandwidth_audit(&tg, &sched, &one_unit_machine(15.0));
        let over = audit.oversubscribed();
        assert_eq!(over.len(), 1);
        assert_eq!(over[0].0, 0);
        assert!((over[0].1 - 20.0).abs() < 1e-9, "peak = {}", over[0].1);
    }

    // Half-open intervals: a load ending at cycle c is not concurrent with one
    // starting at c, so back-to-back loads stay within capacity.
    #[test]
    fn non_overlapping_loads_within_capacity() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            task(TaskKind::Compute, 0, 10, 100),
            task(TaskKind::DmaIn, 0, 10, 100),
        ];
        let sched = sched_with_starts(vec![0, 10]);
        let audit = hbm_bandwidth_audit(&tg, &sched, &one_unit_machine(15.0));
        assert!(audit.oversubscribed().is_empty());
    }

    // Cross-unit transfers ride the interconnect, and zero-byte tasks move no HBM.
    #[test]
    fn cross_unit_and_zero_byte_ignored() {
        let mut tg = TaskGraph::default();
        let mut xfer = task(TaskKind::DmaIn, 0, 10, 1000);
        xfer.cross_unit = true;
        tg.tasks = vec![task(TaskKind::Compute, 0, 10, 0), xfer];
        let sched = sched_with_starts(vec![0, 0]);
        let audit = hbm_bandwidth_audit(&tg, &sched, &one_unit_machine(1.0));
        assert!(audit.oversubscribed().is_empty());
    }

    fn at(kind: TaskKind, coord: Vec<i64>) -> Task {
        let mut t = task(kind, 0, 1, 0);
        t.coord = coord;
        t
    }

    // F4: a producer GEMM tile (block 128, axis 0) and the two smaller consumer
    // Row tiles (block 64, axis 0) covering the same 128 rows must land on one SM.
    // band = max(128, 64) = 128.
    #[test]
    fn coupled_tiles_with_differing_blocks_share_sm() {
        let smc = 8;
        let prod = ColoPin { axis: 0, block: 128, band: 128 };
        let cons = ColoPin { axis: 0, block: 64, band: 128 };
        // Producer tile 1 covers rows [128,256); consumers 2 and 3 cover
        // [128,192) and [192,256) — all in producer 1's band → SM 1.
        let p1 = colocated_sm(&at(TaskKind::Compute, vec![1]), smc, Some(&prod));
        let c2 = colocated_sm(&at(TaskKind::Compute, vec![2]), smc, Some(&cons));
        let c3 = colocated_sm(&at(TaskKind::Compute, vec![3]), smc, Some(&cons));
        assert_eq!(p1, 1);
        assert_eq!(c2, 1);
        assert_eq!(c3, 1);
    }

    // The coupled axis is 1 for Flash — keying off coord[0] (the head axis) would
    // scatter tiles that should pair. With axis=1 the q-band drives the SM.
    #[test]
    fn flash_pins_on_axis_1_not_head() {
        let smc = 4;
        let pin = ColoPin { axis: 1, block: 64, band: 64 };
        // Two tiles, different heads (coord[0]) but same q-band (coord[1]=2) →
        // same SM. Legacy coord[0] keying would give head0→0, head3→3.
        let a = colocated_sm(&at(TaskKind::Compute, vec![0, 2]), smc, Some(&pin));
        let b = colocated_sm(&at(TaskKind::Compute, vec![3, 2]), smc, Some(&pin));
        assert_eq!(a, 2);
        assert_eq!(b, 2);
    }

    // No pin (untiled/layout domain) falls back to the legacy coord[0] keying.
    #[test]
    fn no_pin_falls_back_to_coord0() {
        let smc = 4;
        assert_eq!(colocated_sm(&at(TaskKind::Compute, vec![5, 9]), smc, None), 1);
    }
}
