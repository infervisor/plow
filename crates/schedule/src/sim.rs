//! Cycle simulator — replays a [`Schedule`] to a real makespan and validates it.
//!
//! The list scheduler emits an *event-ordered* schedule, not a cycle-locked one
//! (design §4.5): the runtime times tasks dynamically via counters. This module
//! replays that, two ways, as longest-path over a DAG of:
//! * **resource-order edges** — consecutive tasks on one SM/DMA/DPU/host run
//!   serially in the scheduler's chosen order (a resource is exclusive); and
//! * the dependency model:
//!   - **ideal** (`ideal_makespan`): direct data-dependency edges — perfect
//!     pipelining, a consumer starts as soon as *its* producers finish.
//!   - **counter** (`makespan`): the real runtime — a consumer waits on its
//!     *clustered counters*, so coarse clustering makes it wait for the whole
//!     producer op. The gap `makespan − ideal_makespan` is the pipelining lost
//!     to clustering.
//!
//! Consistency (`consistent`) checks that the schedule replays to exactly the
//! makespan it was costed at: a dependency + resource-order longest path,
//! **floored by the scheduler's chosen start times**, must equal
//! `schedule.makespan`. The floor lets the replay honor stalls the scheduler
//! inserted for HBM bandwidth (folded compute loads compete with DMA for the
//! unit's HBM budget); a dependency-infeasible schedule replays strictly later
//! than its costed makespan and is flagged. `ideal_makespan` stays the unfloored
//! perfect-pipelining lower bound, so `makespan − ideal_makespan` still reports
//! the pipelining + bandwidth overhead. Peak HBM oversubscription is checked
//! separately by [`crate::hbm_bandwidth_audit`].

use crate::expand::{TaskGraph, TaskId};
use crate::interval::Cycle;
use crate::passes::Schedule;
use crate::resource::ResourceId;
use std::collections::HashMap;

/// Result of replaying a schedule.
#[derive(Clone, Debug)]
pub struct SimResult {
    /// Real (counter-gated) makespan.
    pub makespan: Cycle,
    /// Makespan with perfect per-tile pipelining (dependency-gated).
    pub ideal_makespan: Cycle,
    /// Busy cycles per resource (for utilization).
    pub busy: HashMap<ResourceId, Cycle>,
    /// `ideal_makespan == schedule.makespan` — the static schedule replays to
    /// the same time it was costed at.
    pub consistent: bool,
    /// The replay's dependency graph has a cycle — the schedule would **deadlock**
    /// at runtime, and both makespans above are only lower bounds over the acyclic
    /// part. Always `false` for a correct schedule; the counter verifier
    /// ([`crate::verify`]) pinpoints the offending counters.
    pub cyclic: bool,
}

impl SimResult {
    /// Cycles lost to coarse counter clustering (vs. ideal pipelining).
    pub fn clustering_overhead(&self) -> Cycle {
        self.makespan.saturating_sub(self.ideal_makespan)
    }
    /// Fraction of the makespan a resource is busy (0..=1).
    pub fn utilization(&self, r: ResourceId) -> f64 {
        if self.makespan == 0 {
            return 0.0;
        }
        self.busy.get(&r).copied().unwrap_or(0) as f64 / self.makespan as f64
    }
}

/// Replay `sched` (over `tasks`) to a real makespan.
pub fn simulate(tasks: &TaskGraph, sched: &Schedule) -> SimResult {
    let n = tasks.tasks.len();
    let dur: Vec<Cycle> = tasks.tasks.iter().map(|t| t.dur.max(1)).collect();

    // Resource-order edges: consecutive tasks in each resource's issue stream.
    let mut res_edges: Vec<(TaskId, TaskId)> = Vec::new();
    for items in sched.streams.values() {
        let mut ordered: Vec<(TaskId, Cycle)> = items.clone();
        ordered.sort_by_key(|&(t, s)| (s, t));
        for w in ordered.windows(2) {
            res_edges.push((w[0].0, w[1].0));
        }
    }

    // Counter-gated edges: a task waits on its counters; a counter is satisfied
    // when *every* producer that increments it has finished (threshold = the
    // producer count), so each such producer is an edge into the consumer.
    let mut wait: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut producers: HashMap<usize, Vec<TaskId>> = HashMap::new();
    for packets in sched.packets.values() {
        for p in packets {
            wait[p.task] = p.wait.clone();
            for &c in &p.successors {
                producers.entry(c).or_default().push(p.task);
            }
        }
    }
    let mut counter_edges = res_edges.clone();
    for (t, counters) in wait.iter().enumerate() {
        for c in counters {
            for &p in producers.get(c).into_iter().flatten() {
                counter_edges.push((p, t));
            }
        }
    }

    // Dependency-gated edges: direct data edges + resource order.
    let mut dep_edges = res_edges;
    dep_edges.extend(tasks.edges.iter().copied());

    // `ideal`/`makespan` are unfloored structural lower bounds (perfect
    // pipelining, and counter-gated). `costed` floors the dependency +
    // resource-order replay by the scheduler's start times, so it reproduces
    // exactly `schedule.makespan` when the schedule is dependency-feasible —
    // honoring the HBM-bandwidth stalls the scheduler inserted, which the
    // unfloored `ideal` does not see.
    let zero = vec![0u64; n];
    let (finish_dep, ideal, dep_acyclic) = longest_path(n, &dep_edges, &dur, &zero);
    let (_finish_ctr, makespan, ctr_acyclic) = longest_path(n, &counter_edges, &dur, &zero);
    let floor: &[Cycle] = if sched.starts.len() == n { &sched.starts } else { &zero };
    let (_finish_costed, costed, _costed_acyclic) = longest_path(n, &dep_edges, &dur, floor);

    let mut busy: HashMap<ResourceId, Cycle> = HashMap::new();
    for (&t, &r) in &sched.placement {
        *busy.entry(r).or_insert(0) += dur[t];
    }

    let _ = finish_dep;
    SimResult {
        makespan,
        ideal_makespan: ideal,
        busy,
        consistent: costed == sched.makespan,
        cyclic: !dep_acyclic || !ctr_acyclic,
    }
}

/// Longest path (critical path) finishing time per node over a DAG, and the
/// overall makespan. `finish[t] = dur[t] + max(floor[t], max(finish[pred]))`.
/// `floor` is a per-task earliest-start lower bound (pass all-zero for the pure
/// structural critical path); the schedule's chosen starts as `floor` make the
/// replay honor bandwidth stalls baked into those starts.
fn longest_path(
    n: usize,
    edges: &[(TaskId, TaskId)],
    dur: &[Cycle],
    floor: &[Cycle],
) -> (Vec<Cycle>, Cycle, bool) {
    let mut preds = vec![Vec::new(); n];
    let mut indeg = vec![0u32; n];
    let mut adj = vec![Vec::new(); n];
    for &(a, b) in edges {
        preds[b].push(a);
        adj[a].push(b);
        indeg[b] += 1;
    }
    // Kahn topological order.
    let mut queue: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    let mut indeg_rt = indeg;
    while let Some(t) = queue.pop() {
        order.push(t);
        for &s in &adj[t] {
            indeg_rt[s] -= 1;
            if indeg_rt[s] == 0 {
                queue.push(s);
            }
        }
    }
    // A complete topo order ⇒ the graph is a DAG. If it isn't, Kahn drops the
    // cycle's nodes and the makespan below is only a lower bound over the acyclic
    // remainder; `acyclic = false` lets the caller flag the estimate as unreliable
    // (a cyclic counter graph is a real deadlock — see `SimResult::cyclic`).
    let acyclic = order.len() == n;
    let mut finish = vec![0u64; n];
    for &t in &order {
        let base = preds[t].iter().map(|&p| finish[p]).max().unwrap_or(0);
        finish[t] = base.max(floor.get(t).copied().unwrap_or(0)) + dur[t];
    }
    let makespan = finish.iter().copied().max().unwrap_or(0);
    (finish, makespan, acyclic)
}

/// A human-readable dump of the scheduled packet streams (the runtime artifact).
pub fn dump_packets(sched: &Schedule) -> String {
    let mut resources: Vec<&ResourceId> = sched.packets.keys().collect();
    resources.sort_by_key(|r| format!("{r:?}"));
    let mut out = String::new();
    for r in resources {
        out.push_str(&format!("{r:?}:\n"));
        let mut pkts = sched.packets[r].clone();
        pkts.sort_by_key(|p| (p.start, p.task));
        for p in &pkts {
            out.push_str(&format!(
                "  @{:>8}  {:?}  {}  wait={:?} inc={:?}\n",
                p.start, p.kind, p.op, p.wait, p.successors
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cyclic dependency graph is an invalid schedule — `longest_path` reports
    /// `acyclic = false` instead of silently returning a wrong makespan.
    #[test]
    fn longest_path_flags_cycles() {
        // 0 → 1 → 0: Kahn can order neither node.
        let (_finish, _makespan, acyclic) = longest_path(2, &[(0, 1), (1, 0)], &[1, 1], &[0, 0]);
        assert!(!acyclic, "a cycle must be reported");
    }

    /// An acyclic chain produces the expected critical-path makespan.
    #[test]
    fn longest_path_acyclic_chain() {
        let (_finish, makespan, acyclic) = longest_path(3, &[(0, 1), (1, 2)], &[2, 3, 4], &[0, 0, 0]);
        assert!(acyclic);
        assert_eq!(makespan, 9);
    }

    /// A per-task floor (e.g. a bandwidth stall the scheduler inserted) pushes a
    /// task's start past its structural earliest, lengthening the makespan.
    #[test]
    fn longest_path_honors_floor() {
        // Chain 0→1: durs 2,3 → structural makespan 5. Floor task 1 at cycle 10
        // (a stall) → it can't finish before 13.
        let (_f, makespan, acyclic) = longest_path(2, &[(0, 1)], &[2, 3], &[0, 10]);
        assert!(acyclic);
        assert_eq!(makespan, 13);
    }
}
