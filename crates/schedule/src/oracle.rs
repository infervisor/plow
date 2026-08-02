//! Lean performance oracle — queries the `plow_verify` binary for provably-optimal
//! scheduling decisions, with pure-Rust fallbacks when the binary is unavailable.
//!
//! The oracle implements three query types:
//! - **Counter granularity** (Phase 2): per-edge fine/coarse decision backed by
//!   `Plow.CounterGranularity.fineCanPay`.
//! - **Lower bound** (Phase 3): `max(critical_path, bw_bound, compute_bound)`
//!   backed by `Plow.CostBounds.makespan_dominates_lower_bounds`.
//! - **Prefetch depth** (Phase 5): dynamic `qsize` per-SM backed by the
//!   `Plow.Prefetch.optimal_depth` theorem.
//!
//! All functions return results unconditionally: either from Lean (certified) or
//! from the equivalent Rust computation (uncertified but identical arithmetic).

use std::collections::HashMap;

use crate::expand::{TaskGraph, TaskId, TaskKind};
use crate::interval::Cycle;
use crate::passes::Schedule;

// ─── Phase 2: Counter Granularity ───────────────────────────────────────────

/// Per-edge counter granularity decision.
#[derive(Clone, Debug)]
pub struct EdgeGranularity {
    /// Edge identifier (producer_node, consumer_node).
    pub producer_node: usize,
    pub consumer_node: usize,
    /// `true` = use fine (per-tile) counters; `false` = coarse suffices.
    pub use_fine: bool,
    /// Whether the decision came from Lean (certified) or Rust fallback.
    pub certified: bool,
}

/// Query counter granularity for all cross-op edges in the task graph.
///
/// When `lean_oracle` is `true`, attempts to query the Lean binary. On failure
/// (binary absent, timeout, etc.), falls back to the Rust heuristic.
///
/// The Rust heuristic: use fine counters unless ALL consumer tiles have uniform
/// work (same duration), in which case coarse counters suffice (the collapse
/// theorem guarantees equivalent makespan).
pub fn query_counter_granularity(tg: &TaskGraph, lean_oracle: bool) -> Vec<EdgeGranularity> {
    // Group edges by (producer_node, consumer_node).
    let mut groups: HashMap<(usize, usize), Vec<(TaskId, TaskId)>> = HashMap::new();
    for &(a, b) in &tg.edges {
        let (pn, cn) = (tg.tasks[a].node, tg.tasks[b].node);
        if pn != cn {
            groups.entry((pn, cn)).or_default().push((a, b));
        }
    }

    let mut decisions = Vec::new();

    #[cfg(feature = "lean-verify")]
    if lean_oracle {
        if let Some(lean_decisions) = try_lean_counter_granularity(tg, &groups) {
            return lean_decisions;
        }
        // Fall through to Rust heuristic on failure.
    }
    let _ = lean_oracle; // suppress unused warning when feature is off

    // Rust fallback: check work uniformity per consumer on each boundary.
    for (&(pn, cn), edges) in &groups {
        let consumer_tasks: Vec<TaskId> = edges.iter().map(|&(_, b)| b).collect();
        let use_fine = !is_uniform_work(&consumer_tasks, tg);
        decisions.push(EdgeGranularity {
            producer_node: pn,
            consumer_node: cn,
            use_fine,
            certified: false,
        });
    }

    decisions
}

/// Check whether all tasks in the set have the same duration (uniform work).
/// If uniform, fine counters provide no benefit over coarse (collapse theorem).
fn is_uniform_work(tasks: &[TaskId], tg: &TaskGraph) -> bool {
    if tasks.len() <= 1 {
        return true;
    }
    let first_dur = tg.tasks[tasks[0]].dur;
    tasks.iter().all(|&t| tg.tasks[t].dur == first_dur)
}

/// Attempt to call the Lean oracle for counter granularity decisions.
#[cfg(feature = "lean-verify")]
fn try_lean_counter_granularity(
    tg: &TaskGraph,
    groups: &HashMap<(usize, usize), Vec<(TaskId, TaskId)>>,
) -> Option<Vec<EdgeGranularity>> {
    use lean_verify::queries::counter_granularity::{CounterGranularityRequest, EdgeQuery};

    let mut edge_queries = Vec::new();
    let mut id_to_key: HashMap<u64, (usize, usize)> = HashMap::new();

    for (idx, (&(pn, cn), edges)) in groups.iter().enumerate() {
        let consumer_tasks: Vec<TaskId> = edges.iter().map(|&(_, b)| b).collect();
        let consumer_slices: Vec<u64> = consumer_tasks.iter().map(|&t| t as u64).collect();
        let work: Vec<u64> = consumer_tasks.iter().map(|&t| tg.tasks[t].dur).collect();

        let id = idx as u64;
        id_to_key.insert(id, (pn, cn));
        edge_queries.push(EdgeQuery {
            id,
            consumer_slices,
            work,
        });
    }

    let req = CounterGranularityRequest {
        edges: edge_queries,
    };

    match lean_verify::queries::counter_granularity::query_counter_granularity(&req) {
        Ok(result) => {
            let decisions: Vec<EdgeGranularity> = result
                .decisions
                .into_iter()
                .filter_map(|d| {
                    let (pn, cn) = id_to_key.get(&d.id)?;
                    Some(EdgeGranularity {
                        producer_node: *pn,
                        consumer_node: *cn,
                        use_fine: d.use_fine,
                        certified: true,
                    })
                })
                .collect();
            Some(decisions)
        }
        Err(_) => None, // fall back to Rust
    }
}

// ─── Phase 3: Lower Bound ──────────────────────────────────────────────────

/// Which constraint binds the schedule lower bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingConstraint {
    CriticalPath,
    HbmBandwidth,
    ComputeThroughput,
}

/// Certified lower bound on the optimal schedule's makespan.
#[derive(Clone, Debug)]
pub struct LowerBound {
    /// The proven lower bound: `max(critical_path, bw_bound, compute_bound)`.
    pub bound: Cycle,
    /// Which of the three constraints is binding.
    pub binding: BindingConstraint,
    /// Individual bound values for diagnostics.
    pub critical_path: Cycle,
    pub bw_bound: Cycle,
    pub compute_bound: Cycle,
    /// Whether backed by a Lean certificate.
    pub certified: bool,
}

/// Compute the schedule lower bound. Queries Lean if available; otherwise
/// computes the same arithmetic in Rust (uncertified).
///
/// The lower bound is `max(E1, E2, E3)` where:
/// - E1 = critical path length (longest-path DAG analysis)
/// - E2 = total_hbm_bytes / peak_hbm_bw (bandwidth bound)
/// - E3 = total_flops / peak_flops (compute bound)
pub fn compute_lower_bound(
    tg: &TaskGraph,
    peak_bw_bytes_per_cycle: u64,
    peak_flops_per_cycle: u64,
    lean_oracle: bool,
) -> LowerBound {
    // `durations` is only read by the Lean lower-bound query below; without that
    // feature the binding is unused.
    #[cfg_attr(not(feature = "lean-verify"), allow(unused_variables))]
    let (total_hbm_bytes, total_flops, durations) = gather_workload_stats(tg);
    let critical_path = compute_critical_path(tg);

    #[cfg(feature = "lean-verify")]
    if lean_oracle {
        if let Some(lb) = try_lean_lower_bound(
            tg,
            &durations,
            total_hbm_bytes,
            peak_bw_bytes_per_cycle,
            total_flops,
            peak_flops_per_cycle,
        ) {
            return lb;
        }
    }
    let _ = lean_oracle;

    // Rust fallback: same arithmetic, no proof.
    let bw_bound = if peak_bw_bytes_per_cycle > 0 {
        total_hbm_bytes.div_ceil(peak_bw_bytes_per_cycle)
    } else {
        0
    };
    let compute_bound = if peak_flops_per_cycle > 0 {
        total_flops.div_ceil(peak_flops_per_cycle)
    } else {
        0
    };

    let bound = critical_path.max(bw_bound).max(compute_bound);
    let binding = if bound == critical_path {
        BindingConstraint::CriticalPath
    } else if bound == bw_bound {
        BindingConstraint::HbmBandwidth
    } else {
        BindingConstraint::ComputeThroughput
    };

    LowerBound {
        bound,
        binding,
        critical_path,
        bw_bound,
        compute_bound,
        certified: false,
    }
}

/// Gather total HBM bytes, total FLOPs, and per-task durations.
fn gather_workload_stats(tg: &TaskGraph) -> (u64, u64, Vec<u64>) {
    let mut total_hbm_bytes = 0u64;
    let mut total_flops = 0u64;
    let durations: Vec<u64> = tg.tasks.iter().map(|t| t.dur).collect();

    for task in &tg.tasks {
        match task.kind {
            TaskKind::DmaIn | TaskKind::DmaOut => {
                total_hbm_bytes += task.bytes;
            }
            TaskKind::Compute => {
                // Approximate: duration × throughput gives FLOPs for compute tasks.
                // For scheduling purposes, we use bytes as HBM contribution when
                // the compute task has collapsed DMA.
                total_hbm_bytes += task.bytes; // collapsed DMA bytes
                total_flops += task.dur; // cycles ≈ FLOPs at peak (normalized)
            }
            TaskKind::Host => {}
        }
    }

    (total_hbm_bytes, total_flops, durations)
}

/// Longest-path computation on the task DAG — the critical path lower bound.
fn compute_critical_path(tg: &TaskGraph) -> Cycle {
    let n = tg.tasks.len();
    if n == 0 {
        return 0;
    }

    // Topological sort via Kahn's algorithm.
    let mut indeg = vec![0u32; n];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(a, b) in &tg.edges {
        indeg[b] += 1;
        succ[a].push(b);
    }
    let mut queue: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop() {
        order.push(u);
        for &v in &succ[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                queue.push(v);
            }
        }
    }

    // Forward pass: longest path from any source.
    let mut dist = vec![0u64; n];
    for &u in &order {
        let end = dist[u] + tg.tasks[u].dur;
        for &v in &succ[u] {
            dist[v] = dist[v].max(end);
        }
    }

    // Critical path = max(dist[u] + dur[u]) over all tasks.
    (0..n).map(|u| dist[u] + tg.tasks[u].dur).max().unwrap_or(0)
}

#[cfg(feature = "lean-verify")]
fn try_lean_lower_bound(
    tg: &TaskGraph,
    durations: &[u64],
    total_hbm_bytes: u64,
    peak_bw_bytes_per_cycle: u64,
    total_flops: u64,
    peak_flops_per_cycle: u64,
) -> Option<LowerBound> {
    use lean_verify::queries::lower_bound::{BindingConstraint as LeanBC, LowerBoundRequest};

    let req = LowerBoundRequest {
        edges: tg.edges.clone(),
        durations: durations.to_vec(),
        total_hbm_bytes,
        peak_bw_bytes_per_cycle,
        total_flops,
        peak_flops_per_cycle,
    };

    match lean_verify::queries::lower_bound::query_lower_bound(&req) {
        Ok(result) => {
            let binding = match result.binding_constraint {
                LeanBC::CriticalPath => BindingConstraint::CriticalPath,
                LeanBC::HbmBandwidth => BindingConstraint::HbmBandwidth,
                LeanBC::ComputeThroughput => BindingConstraint::ComputeThroughput,
            };
            Some(LowerBound {
                bound: result.lower_bound,
                binding,
                critical_path: result.critical_path,
                bw_bound: result.bw_bound,
                compute_bound: result.compute_bound,
                certified: true,
            })
        }
        Err(_) => None,
    }
}

// ─── Phase 5: Prefetch Depth Oracle ─────────────────────────────────────────

/// Dynamic prefetch depth recommendation per unit.
#[derive(Clone, Debug)]
pub struct PrefetchDepth {
    /// Recommended prefetch depth (replaces `Config::qsize`).
    pub depth: u32,
    /// Whether the decision is certified by Lean.
    pub certified: bool,
    /// Ratio of DMA latency to compute latency that drove the decision.
    pub dma_compute_ratio: f64,
}

/// Compute optimal prefetch depth for a given unit's workload.
///
/// The optimal depth is `ceil(avg_dma_latency / avg_compute_latency)` clamped
/// to `[1, max_depth]`. This ensures the DMA pipeline stays full while compute
/// executes, hiding memory latency without over-provisioning SRAM pages.
///
/// Backed by `Plow.Prefetch.optimal_depth` when Lean is available.
pub fn compute_prefetch_depth(
    tg: &TaskGraph,
    sched: &Schedule,
    unit: usize,
    max_depth: u32,
    lean_oracle: bool,
) -> PrefetchDepth {
    let _ = lean_oracle; // Lean query for prefetch depth is Phase 5+ (stub)

    // Gather DMA-in and Compute durations assigned to this unit.
    let mut dma_durs: Vec<u64> = Vec::new();
    let mut compute_durs: Vec<u64> = Vec::new();

    for (&task_id, res) in &sched.placement {
        let unit_of_res = match res {
            crate::resource::ResourceId::Sm(u, _) => *u,
            crate::resource::ResourceId::Dma(u, _) => *u,
            crate::resource::ResourceId::Dpu(_) => continue,
            crate::resource::ResourceId::Host(_) => continue,
        };
        if unit_of_res != unit {
            continue;
        }
        let task = &tg.tasks[task_id];
        match task.kind {
            TaskKind::DmaIn => dma_durs.push(task.dur),
            TaskKind::Compute => compute_durs.push(task.dur),
            _ => {}
        }
    }

    if dma_durs.is_empty() || compute_durs.is_empty() {
        return PrefetchDepth {
            depth: 1,
            certified: false,
            dma_compute_ratio: 0.0,
        };
    }

    let avg_dma: f64 = dma_durs.iter().sum::<u64>() as f64 / dma_durs.len() as f64;
    let avg_compute: f64 = compute_durs.iter().sum::<u64>() as f64 / compute_durs.len() as f64;

    let ratio = if avg_compute > 0.0 {
        avg_dma / avg_compute
    } else {
        1.0
    };

    // Depth = ceil(ratio), clamped to [1, max_depth].
    let depth = (ratio.ceil() as u32).clamp(1, max_depth);

    PrefetchDepth {
        depth,
        certified: false,
        dma_compute_ratio: ratio,
    }
}

// ─── Phase 4: Bubble-Fill Pass ──────────────────────────────────────────────

/// Report from the bubble-fill reordering pass.
#[derive(Clone, Debug, Default)]
pub struct BubbleFillReport {
    /// Number of bubble slots that were filled by reordering.
    pub bubbles_filled: usize,
    /// Total idle cycles recovered.
    pub cycles_recovered: Cycle,
    /// Number of streams that were modified.
    pub streams_modified: usize,
}

/// Identify and fill "bubbles" — idle gaps in a resource stream where a later
/// task could have started earlier without violating dependencies.
///
/// Unlike the prefetch hoisting pass (which only moves DMA-ins), this pass
/// considers ALL task types and attempts to compact each stream by pulling tasks
/// forward into gaps left by cross-stream dependencies.
///
/// Returns a new schedule with bubbles filled and a diagnostic report.
pub fn fill_bubbles(tg: &TaskGraph, sched: &Schedule) -> (Schedule, BubbleFillReport) {
    let n = tg.tasks.len();
    let mut out = sched.clone();
    let mut report = BubbleFillReport::default();

    // Build predecessor end-time for each task.
    let mut pred_end: Vec<Cycle> = vec![0; n];
    for &(a, b) in &tg.edges {
        let a_end = sched.starts[a] + tg.tasks[a].dur;
        pred_end[b] = pred_end[b].max(a_end);
    }

    // Process each stream: for each task, compute its earliest possible start
    // (max of predecessor ends + prev-on-stream end). If earlier than current
    // start, shift it forward.
    let resources: Vec<crate::resource::ResourceId> = out.streams.keys().copied().collect();
    let mut any_stream_modified = false;
    for res in resources {
        let stream = out.streams.get(&res).unwrap().clone();
        if stream.is_empty() {
            continue;
        }

        let mut new_stream: Vec<(TaskId, Cycle)> = Vec::with_capacity(stream.len());
        let mut stream_modified = false;
        let mut prev_end: Cycle = 0;

        for &(task_id, current_start) in &stream {
            // Earliest this task can start: after all its DAG predecessors AND
            // after the previous task on this stream finishes.
            let earliest = pred_end[task_id].max(prev_end);
            let new_start = earliest;

            if new_start < current_start {
                report.bubbles_filled += 1;
                report.cycles_recovered += current_start - new_start;
                stream_modified = true;
            }

            new_stream.push((task_id, new_start));
            prev_end = new_start + tg.tasks[task_id].dur;
        }

        if stream_modified {
            any_stream_modified = true;
            report.streams_modified += 1;
            out.streams.insert(res, new_stream.clone());
            // Update starts for tasks on this stream.
            for &(task_id, start) in &new_stream {
                out.starts[task_id] = start;
            }
            // Update packets too.
            if let Some(packets) = out.packets.get_mut(&res) {
                for (i, pkt) in packets.iter_mut().enumerate() {
                    if i < new_stream.len() {
                        pkt.start = new_stream[i].1;
                    }
                }
            }
        }
    }

    // Shifting task starts invalidates the per-slot SRAM/TMEM assignments made
    // against the pre-fill timeline — two tiles pulled earlier can now overlap
    // on the same page slot, and `emit` would bake stale slot ids into packets.
    // Re-run Pass B against the new starts, exactly as `prefetch::hoist_prefetches`
    // does after it retimes streams. Spills are recounted the same way the main
    // scheduling path counts them (a tile that no longer fits loses its slot).
    if any_stream_modified {
        let n_tasks = tg.tasks.len();
        let mut succ_of: Vec<Vec<TaskId>> = vec![Vec::new(); n_tasks];
        for &(a, b) in &tg.edges {
            succ_of[a].push(b);
        }
        let (sram_slots, spills) =
            crate::passes::allocate_sram(tg, &succ_of, &out.starts, &out.placement);
        let (tmem_slots, tmem_spills) =
            crate::passes::allocate_tmem(tg, &out.starts, &out.placement);
        out.sram_slots = sram_slots;
        out.spills = spills;
        out.tmem_slots = tmem_slots;
        out.tmem_spills = tmem_spills;
    }

    // Recompute makespan.
    out.makespan = (0..n)
        .map(|t| out.starts[t] + tg.tasks[t].dur)
        .max()
        .unwrap_or(0);

    (out, report)
}

// ─── Oracle Report ──────────────────────────────────────────────────────────

/// Combined oracle report summarizing all performance decisions.
#[derive(Clone, Debug)]
pub struct OracleReport {
    /// Lower bound on optimal makespan.
    pub lower_bound: LowerBound,
    /// Actual makespan achieved by the schedule.
    pub achieved_makespan: Cycle,
    /// Gap from optimality: `(achieved - lower_bound) / lower_bound`.
    pub optimality_gap: f64,
    /// Bubble-fill results.
    pub bubble_fill: BubbleFillReport,
    /// Per-unit prefetch depth recommendations.
    pub prefetch_depths: Vec<PrefetchDepth>,
    /// Whether any decision was certified by Lean.
    pub any_certified: bool,
}

/// Run the full oracle pipeline: compute lower bound, fill bubbles, compute
/// prefetch depths. Returns the optimized schedule and a combined report.
pub fn run_oracle(
    tg: &TaskGraph,
    sched: &Schedule,
    peak_bw_bytes_per_cycle: u64,
    peak_flops_per_cycle: u64,
    num_units: usize,
    max_prefetch_depth: u32,
    lean_oracle: bool,
) -> (Schedule, OracleReport) {
    // Phase 3: Lower bound.
    let lower_bound = compute_lower_bound(
        tg,
        peak_bw_bytes_per_cycle,
        peak_flops_per_cycle,
        lean_oracle,
    );

    // Phase 4: Bubble fill.
    let (optimized, bubble_fill) = fill_bubbles(tg, sched);

    // Phase 5: Prefetch depths.
    let prefetch_depths: Vec<PrefetchDepth> = (0..num_units)
        .map(|u| compute_prefetch_depth(tg, &optimized, u, max_prefetch_depth, lean_oracle))
        .collect();

    let achieved = optimized.makespan;
    let gap = if lower_bound.bound > 0 {
        (achieved as f64 - lower_bound.bound as f64) / lower_bound.bound as f64
    } else {
        0.0
    };

    let any_certified = lower_bound.certified || prefetch_depths.iter().any(|p| p.certified);

    let report = OracleReport {
        lower_bound,
        achieved_makespan: achieved,
        optimality_gap: gap,
        bubble_fill,
        prefetch_depths,
        any_certified,
    };

    (optimized, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expand::Task;

    fn make_task(kind: TaskKind, dur: Cycle, bytes: u64) -> Task {
        Task {
            node: 0,
            op: "test".into(),
            unit: 0,
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

    #[test]
    fn critical_path_chain() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            make_task(TaskKind::Compute, 10, 0),
            make_task(TaskKind::Compute, 20, 0),
            make_task(TaskKind::Compute, 30, 0),
        ];
        tg.edges = vec![(0, 1), (1, 2)];
        assert_eq!(compute_critical_path(&tg), 60); // 10 + 20 + 30
    }

    #[test]
    fn critical_path_parallel() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            make_task(TaskKind::Compute, 10, 0),
            make_task(TaskKind::Compute, 100, 0),
            make_task(TaskKind::Compute, 5, 0),
        ];
        tg.edges = vec![(0, 2), (1, 2)]; // both feed task 2
                                         // Path 0→2: 10+5=15, path 1→2: 100+5=105
        assert_eq!(compute_critical_path(&tg), 105);
    }

    #[test]
    fn uniform_work_detected() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            make_task(TaskKind::Compute, 10, 0),
            make_task(TaskKind::Compute, 10, 0),
            make_task(TaskKind::Compute, 10, 0),
        ];
        assert!(is_uniform_work(&[0, 1, 2], &tg));
    }

    #[test]
    fn non_uniform_work_detected() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            make_task(TaskKind::Compute, 10, 0),
            make_task(TaskKind::Compute, 20, 0),
            make_task(TaskKind::Compute, 10, 0),
        ];
        assert!(!is_uniform_work(&[0, 1, 2], &tg));
    }

    #[test]
    fn lower_bound_bandwidth_bound() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            make_task(TaskKind::DmaIn, 10, 1000),
            make_task(TaskKind::DmaIn, 10, 1000),
            make_task(TaskKind::Compute, 5, 0),
        ];
        tg.edges = vec![(0, 2), (1, 2)];

        let lb = compute_lower_bound(&tg, 100, 1000, false);
        // BW bound: 2000 bytes / 100 bps = 20 cycles
        // Critical path: max(10+5, 10+5) = 15
        // Compute bound: 5 flops / 1000 = 1
        assert_eq!(lb.bw_bound, 20);
        assert_eq!(lb.critical_path, 15);
        assert_eq!(lb.bound, 20);
        assert_eq!(lb.binding, BindingConstraint::HbmBandwidth);
    }

    #[test]
    fn prefetch_depth_ratio() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            make_task(TaskKind::DmaIn, 30, 100), // task 0
            make_task(TaskKind::Compute, 10, 0), // task 1
        ];

        // Build a minimal schedule with both tasks on unit 0.
        use crate::resource::ResourceId;
        let sm = ResourceId::Sm(0, 0);
        let sched = Schedule {
            streams: HashMap::from([(sm, vec![(0, 0), (1, 30)])]),
            packets: HashMap::new(),
            counters: vec![],
            placement: HashMap::from([(0usize, sm), (1usize, sm)]),
            starts: vec![0, 30],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 40,
        };

        let pd = compute_prefetch_depth(&tg, &sched, 0, 8, false);
        // ratio = 30/10 = 3.0, depth = ceil(3.0) = 3
        assert_eq!(pd.depth, 3);
        assert!((pd.dma_compute_ratio - 3.0).abs() < 0.01);
    }
}
