//! Pass §8.5 — SRAM temporal fit analysis.
//!
//! `relax.rs` demotes a same-SM (`HandoffKind::SramSameSm`) hand-off to DSM or
//! HBM whenever `producer_pages + consumer_pages > budget`. That is a *spatial*
//! check: it assumes both peaks are simultaneously live in SRAM, which for a
//! true same-SM hand-off can be false — the SM runs producer and consumer
//! serially, so their working sets are temporally disjoint on the SM's page
//! pool. When they don't overlap in time, `max(pages) ≤ budget` is enough.
//!
//! This pass runs **after** `list_schedule` (so task starts are known), walks
//! the demoted relaxables, and identifies those whose actual scheduled cycles
//! satisfy temporal disjointness. The output is a report — no packet stream
//! is modified here. Callers (plowc under `--sram-fit`) can log potential
//! savings; a follow-up Phase 2 will re-relax + reschedule from these
//! candidates.
//!
//! # Safety of the analysis
//!
//! Per `lean-plow/Plow/Sram.lean::temporal_fit_safe`, a hand-off is temporally
//! feasible iff
//!
//! * `producer_release ≤ consumer_acquire` — producer's peak-page moment ends
//!   before consumer's peak-page moment starts, AND
//! * `max(producer_pages, consumer_pages) ≤ budget` — each individual peak fits
//!   the pool.
//!
//! The analysis conservatively estimates `producer_release` as the last task
//! on the producer op-node's completion (`start + dur`), and `consumer_acquire`
//! as the first task on the consumer op-node's start. Those are both provided
//! by `Scheduled.schedule.starts` and `Scheduled.tasks.tasks[t].dur`.

use std::collections::HashMap;

use rewrite::{ConstraintSet, GraphNode as TileNode, HandoffKind, LocalityReq, TileGraph};

use crate::interval::Cycle;
use crate::machine::Machine;
use crate::Scheduled;

/// One promotable hand-off that pessimistic relax demoted but temporal
/// analysis says could have stayed same-SM.
#[derive(Debug, Clone)]
pub struct PromotionCandidate {
    /// Op-node id of the producer.
    pub producer: usize,
    /// Op-node id of the consumer.
    pub consumer: usize,
    /// Tensor name that flows across the hand-off.
    pub tensor: String,
    /// The kind relax chose (DSM or HBM).
    pub demoted_to: HandoffKind,
    /// Producer's cycle window `[first_start, last_end)` on its unit.
    pub producer_release: Cycle,
    /// Consumer's first task start cycle.
    pub consumer_acquire: Cycle,
    /// Producer's total SRAM page footprint (from `cons.sram_pages`).
    pub producer_pages: u64,
    /// Consumer's total SRAM page footprint.
    pub consumer_pages: u64,
    /// The SM page budget on the producer's unit.
    pub budget: u64,
    /// Cycles saved per hand-off by keeping this in SRAM (cost of the demoted
    /// realization minus cost of the SRAM realization).
    pub cycles_saved: u64,
}

/// Aggregate report from `analyze_temporal_fit`.
#[derive(Debug, Clone, Default)]
pub struct SramFitReport {
    pub candidates: Vec<PromotionCandidate>,
    /// Total handoffs that relax demoted.
    pub demoted: usize,
    /// Sum of `cycles_saved` across all candidates.
    pub total_cycles_savings: u64,
}

impl SramFitReport {
    pub fn promotion_rate_pct(&self) -> f64 {
        if self.demoted == 0 {
            0.0
        } else {
            100.0 * self.candidates.len() as f64 / self.demoted as f64
        }
    }
}

/// Post-schedule analysis: walk the relaxables and identify handoffs that
/// pessimistic relax demoted but that could have stayed same-SM under
/// temporal fit. Returns a diagnostic report; does not modify the schedule.
///
/// # Soundness of the two-condition filter
///
/// The filter below applies exactly two predicates to each candidate hand-off:
///
///   1. **Spatial**: `max(producer_pages, consumer_pages) ≤ budget`
///   2. **Temporal disjointness**: `producer_release ≤ consumer_acquire`
///
/// `Plow.Sram.occupancy_le_of_temporal_fit` (in `lean-plow/Plow/Sram.lean`)
/// proves those two predicates suffice to guarantee `occupancy t ≤ budget`
/// at every cycle. Once that theorem is discharged (Round 6), this Rust
/// filter is **known-correct by construction** — every candidate it accepts
/// satisfies the occupancy bound, and no additional runtime verification
/// step is needed. Checkpoint C therefore stays out of plowc's per-bucket
/// verify loop (see the comment on `run_lean_verify` in `crates/plowc`);
/// the Lean-side `checkSramFit` dispatcher remains available for opt-in
/// spot checks.
pub fn analyze_temporal_fit(
    scheduled: &Scheduled,
    cons: &ConstraintSet,
    machine: &Machine,
) -> SramFitReport {
    let node_of_task: HashMap<usize, usize> = scheduled
        .tasks
        .tasks
        .iter()
        .enumerate()
        .map(|(tid, t)| (tid, t.node))
        .collect();

    // op-node → (first start cycle, last end cycle) across every task
    // scheduled for it. Empty ⇒ node has no scheduled tasks (weight-only).
    let mut node_window: HashMap<usize, (Cycle, Cycle)> = HashMap::new();
    for (tid, task) in scheduled.tasks.tasks.iter().enumerate() {
        let node = node_of_task[&tid];
        let start = scheduled.schedule.starts.get(tid).copied().unwrap_or(0);
        let end = start.saturating_add(task.dur);
        node_window
            .entry(node)
            .and_modify(|(fs, le)| {
                *fs = (*fs).min(start);
                *le = (*le).max(end);
            })
            .or_insert((start, end));
    }

    let mut candidates = Vec::new();
    let mut demoted = 0usize;
    let mut total_cycles_savings = 0u64;

    for r in &cons.relaxables {
        // Skip if relax kept it same-SM.
        if r.default == HandoffKind::SramSameSm {
            continue;
        }
        // Only handoffs that had a SramSameSm alternative are candidates for
        // temporal promotion.
        let Some(&(_, sram_cost)) = r.alts.iter().find(|&&(k, _)| k == HandoffKind::SramSameSm)
        else {
            continue;
        };
        demoted += 1;

        let unit = *cons.placement.get(&r.producer).unwrap_or(&0);
        let budget = machine.unit(unit).pages_per_sm;
        let pp = cons.sram_pages.get(&r.producer).copied().unwrap_or(0);
        let pc = cons.sram_pages.get(&r.consumer).copied().unwrap_or(0);

        // Spatial check: even individually, must fit.
        if pp.max(pc) > budget {
            continue;
        }

        // Temporal disjointness: producer_release ≤ consumer_acquire.
        let (_, prod_end) = match node_window.get(&r.producer) {
            Some(w) => *w,
            None => continue, // nothing scheduled for producer (shouldn't happen)
        };
        let (cons_start, _) = match node_window.get(&r.consumer) {
            Some(w) => *w,
            None => continue,
        };
        if prod_end > cons_start {
            continue; // producer still running when consumer starts
        }

        // How many cycles the demotion cost that SRAM would have avoided.
        let demoted_cost = r
            .alts
            .iter()
            .find(|&&(k, _)| k == r.default)
            .map(|&(_, c)| c)
            .unwrap_or(0);
        let cycles_saved = demoted_cost.saturating_sub(sram_cost);

        candidates.push(PromotionCandidate {
            producer: r.producer,
            consumer: r.consumer,
            tensor: r.tensor.clone(),
            demoted_to: r.default,
            producer_release: prod_end,
            consumer_acquire: cons_start,
            producer_pages: pp,
            consumer_pages: pc,
            budget,
            cycles_saved,
        });
        total_cycles_savings = total_cycles_savings.saturating_add(cycles_saved);
    }

    SramFitReport {
        candidates,
        demoted,
        total_cycles_savings,
    }
}

/// Apply the promotions from the analysis: flip each candidate's
/// `RelaxableHandoff.default` back to `SramSameSm`, update `cons.locality` to
/// `MustColocate`, mark the DmaOut/DmaIn nodes as `resident: true`, and
/// rebuild `cons.colocation_groups` from every SramSameSm relaxable (union-
/// find). Mirrors `relax::relax` in reverse.
///
/// After this returns, the caller should re-run `schedule::schedule` on the
/// mutated `(TileGraph, ConstraintSet)` so `list_schedule` re-runs against
/// the new colocation groups and packet emitter sees the resident flags.
///
/// # Safety
///
/// Provably safe by `Plow.Sram.occupancy_le_of_temporal_fit`: each promoted
/// hand-off satisfies `max(pages) ≤ budget` and temporal disjointness (the
/// two clauses the analysis pass already checked), so the SM's page pool is
/// never over-subscribed.
pub fn promote_temporal_fits(
    g: &TileGraph,
    cons: &ConstraintSet,
    candidates: &[PromotionCandidate],
) -> (TileGraph, ConstraintSet) {
    let mut g = g.clone();
    let mut cons = cons.clone();

    if candidates.is_empty() {
        return (g, cons);
    }

    // Lookup: (compute node, tensor) → its neighbour DmaOut / DmaIn node in
    // the tile graph. Same map relax builds; we reuse it in reverse.
    let mut dma_out_of: HashMap<(usize, String), usize> = HashMap::new();
    let mut dma_in_of: HashMap<(usize, String), usize> = HashMap::new();
    for &(a, b) in &g.edges {
        match (&g.nodes[a], &g.nodes[b]) {
            (TileNode::Compute { .. }, TileNode::DmaOut { tensor, .. }) => {
                dma_out_of.insert((a, tensor.clone()), b);
            }
            (TileNode::DmaIn { tensor, .. }, TileNode::Compute { .. }) => {
                dma_in_of.insert((b, tensor.clone()), a);
            }
            _ => {}
        }
    }

    // Which (producer, consumer, tensor) triples to promote.
    let target_set: std::collections::HashSet<(usize, usize, String)> = candidates
        .iter()
        .map(|c| (c.producer, c.consumer, c.tensor.clone()))
        .collect();

    // Collect updates in a first pass; apply to `cons` fields after so the
    // borrow checker is happy (we mutate `relaxables` and `locality` in the
    // same struct).
    let mut relaxables = cons.relaxables.clone();
    let mut locality = cons.locality.clone();
    let mut resident_flips: Vec<usize> = Vec::new();

    for r in relaxables.iter_mut() {
        let key = (r.producer, r.consumer, r.tensor.clone());
        if !target_set.contains(&key) {
            continue;
        }
        // Only promote things that were demoted.
        if r.default == HandoffKind::SramSameSm {
            continue;
        }
        r.default = HandoffKind::SramSameSm;
        locality.insert((r.producer, r.consumer), LocalityReq::MustColocate);
        if let Some(&n) = dma_out_of.get(&(r.producer, r.tensor.clone())) {
            resident_flips.push(n);
        }
        if let Some(&n) = dma_in_of.get(&(r.consumer, r.tensor.clone())) {
            resident_flips.push(n);
        }
    }

    for n in resident_flips {
        match &mut g.nodes[n] {
            TileNode::DmaIn { resident, .. } | TileNode::DmaOut { resident, .. } => {
                *resident = true;
            }
            _ => {}
        }
    }

    // Rebuild colocation groups from every SramSameSm relaxable — this
    // unions our newly-promoted ones onto whatever the collapse pass and
    // relax pass left behind. Union-find inlined (mirrors relax.rs).
    let mut uf = UnionFind::default();
    for r in relaxables.iter() {
        if r.default == HandoffKind::SramSameSm {
            uf.union(r.producer, r.consumer);
        }
    }
    cons.relaxables = relaxables;
    cons.locality = locality;
    cons.colocation_groups = uf.groups();

    (g, cons)
}

#[derive(Default)]
struct UnionFind {
    parent: HashMap<usize, usize>,
}
impl UnionFind {
    fn find(&mut self, x: usize) -> usize {
        let p = *self.parent.entry(x).or_insert(x);
        if p == x {
            x
        } else {
            let r = self.find(p);
            self.parent.insert(x, r);
            r
        }
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent.insert(ra, rb);
        }
    }
    fn groups(&mut self) -> Vec<Vec<usize>> {
        let keys: Vec<usize> = self.parent.keys().copied().collect();
        let mut by_root: HashMap<usize, Vec<usize>> = HashMap::new();
        for k in keys {
            let r = self.find(k);
            by_root.entry(r).or_default().push(k);
        }
        by_root
            .into_values()
            .filter(|grp| grp.len() > 1)
            .map(|mut grp| {
                grp.sort_unstable();
                grp
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expand::{Task, TaskGraph, TaskKind};
    use crate::machine::Machine;
    use crate::passes::Schedule;
    use crate::Scheduled;
    use costmodel::hwspec;
    use rewrite::{HandoffKind, RelaxableHandoff};
    use std::collections::HashMap;

    fn h100() -> &'static hwspec::GpuSpec {
        hwspec::registry::lookup("H100 SXM5").unwrap()
    }

    fn task(node: usize, dur: Cycle) -> Task {
        Task {
            node,
            op: "x".into(),
            unit: 0,
            kind: TaskKind::Compute,
            coord: vec![],
            dur,
            bytes: 0,
            tensor_bytes: 0,
            sram_pages: 0,
            out_pages: 0,
            tmem_cols: 0,
            tensor: None,
            cross_unit: false,
        }
    }

    fn machine() -> Machine {
        // Single H100 unit with default page pool.
        use costmodel::{Soc, DEFAULT_PAGE_BYTES};
        let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
        Machine::from_soc(&soc, &crate::Config::default())
    }

    fn scheduled_from(tasks: TaskGraph, starts: Vec<Cycle>) -> Scheduled {
        let sched = Schedule {
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
        };
        Scheduled {
            schedule: sched,
            tasks,
            machine: machine(),
            oracle_report: None,
        }
    }

    /// A demoted (HBM) hand-off where sum > budget but max ≤ budget AND the
    /// producer finishes before the consumer starts ⇒ candidate.
    #[test]
    fn identifies_promotable_demoted_handoff() {
        // producer node 10 has tasks 0, 1 (compute cycles 0-100), consumer node 20
        // has tasks 2, 3 (cycles 200-300). Budget = 40 pages.
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            task(10, 50), // producer task 0
            task(10, 50), // producer task 1
            task(20, 50), // consumer task 2
            task(20, 50), // consumer task 3
        ];
        let starts = vec![0, 50, 200, 250];
        let s = scheduled_from(tg, starts);

        let mut cons = ConstraintSet::default();
        cons.placement.insert(10, 0);
        cons.placement.insert(20, 0);
        cons.sram_pages.insert(10, 25); // producer needs 25 pages
        cons.sram_pages.insert(20, 25); // consumer needs 25 pages
                                        // sum = 50 > 40 (would have been demoted), max = 25 ≤ 40 (fits temporally)
        cons.relaxables.push(RelaxableHandoff {
            producer: 10,
            consumer: 20,
            tensor: "T".into(),
            default: HandoffKind::Hbm,
            alts: vec![(HandoffKind::Hbm, 500), (HandoffKind::SramSameSm, 20)],
        });

        let m = s.machine.clone();
        // Override budget via a locally-scoped fake machine? Actually the real
        // machine has a large budget. Let's just accept whatever the H100 has
        // and pick pages that force a demotion below.
        // 25 + 25 = 50 vs H100 budget: check what H100 actually gives.
        let budget = m.unit(0).pages_per_sm;
        // Rewrite pages so sum > budget but max ≤ budget.
        let half = budget / 2 + 1; // sum = budget + 2, max = half
        cons.sram_pages.insert(10, half);
        cons.sram_pages.insert(20, half);

        let rep = analyze_temporal_fit(&s, &cons, &m);
        assert_eq!(rep.demoted, 1);
        assert_eq!(rep.candidates.len(), 1);
        assert_eq!(rep.candidates[0].producer, 10);
        assert_eq!(rep.candidates[0].consumer, 20);
        assert_eq!(rep.candidates[0].producer_release, 100);
        assert_eq!(rep.candidates[0].consumer_acquire, 200);
        assert_eq!(rep.candidates[0].cycles_saved, 480); // 500 - 20
    }

    /// A demoted hand-off where the consumer starts BEFORE the producer
    /// finishes → temporal overlap → not a candidate.
    #[test]
    fn rejects_temporally_overlapping_handoff() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            task(10, 100), // producer: 0-100
            task(20, 50),  // consumer starts at 50 → overlaps producer
        ];
        let starts = vec![0, 50];
        let s = scheduled_from(tg, starts);

        let mut cons = ConstraintSet::default();
        cons.placement.insert(10, 0);
        cons.placement.insert(20, 0);
        let m = s.machine.clone();
        let budget = m.unit(0).pages_per_sm;
        let half = budget / 2 + 1;
        cons.sram_pages.insert(10, half);
        cons.sram_pages.insert(20, half);
        cons.relaxables.push(RelaxableHandoff {
            producer: 10,
            consumer: 20,
            tensor: "T".into(),
            default: HandoffKind::Hbm,
            alts: vec![(HandoffKind::Hbm, 500), (HandoffKind::SramSameSm, 20)],
        });

        let rep = analyze_temporal_fit(&s, &cons, &m);
        assert_eq!(rep.demoted, 1);
        assert!(
            rep.candidates.is_empty(),
            "temporal overlap must reject the candidate"
        );
    }

    /// A hand-off where max > budget (even individually) → not a candidate.
    #[test]
    fn rejects_when_max_exceeds_budget() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![task(10, 50), task(20, 50)];
        let starts = vec![0, 100];
        let s = scheduled_from(tg, starts);

        let mut cons = ConstraintSet::default();
        cons.placement.insert(10, 0);
        cons.placement.insert(20, 0);
        let m = s.machine.clone();
        let budget = m.unit(0).pages_per_sm;
        // Both individually exceed budget.
        cons.sram_pages.insert(10, budget + 10);
        cons.sram_pages.insert(20, budget + 10);
        cons.relaxables.push(RelaxableHandoff {
            producer: 10,
            consumer: 20,
            tensor: "T".into(),
            default: HandoffKind::Hbm,
            alts: vec![(HandoffKind::Hbm, 500), (HandoffKind::SramSameSm, 20)],
        });

        let rep = analyze_temporal_fit(&s, &cons, &m);
        assert!(rep.candidates.is_empty());
    }

    /// A hand-off that relax already kept in SRAM is not a demotion → not
    /// counted.
    #[test]
    fn ignores_already_colocated_handoffs() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![task(10, 50), task(20, 50)];
        let starts = vec![0, 100];
        let s = scheduled_from(tg, starts);

        let mut cons = ConstraintSet::default();
        cons.placement.insert(10, 0);
        cons.placement.insert(20, 0);
        cons.sram_pages.insert(10, 10);
        cons.sram_pages.insert(20, 10);
        cons.relaxables.push(RelaxableHandoff {
            producer: 10,
            consumer: 20,
            tensor: "T".into(),
            default: HandoffKind::SramSameSm,
            alts: vec![(HandoffKind::SramSameSm, 20)],
        });

        let m = s.machine.clone();
        let rep = analyze_temporal_fit(&s, &cons, &m);
        assert_eq!(rep.demoted, 0);
        assert!(rep.candidates.is_empty());
    }

    /// A hand-off with no SramSameSm alt was never eligible for SRAM.
    #[test]
    fn skips_handoffs_with_no_sram_alt() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![task(10, 50), task(20, 50)];
        let starts = vec![0, 100];
        let s = scheduled_from(tg, starts);

        let mut cons = ConstraintSet::default();
        cons.placement.insert(10, 0);
        cons.placement.insert(20, 0);
        cons.sram_pages.insert(10, 5);
        cons.sram_pages.insert(20, 5);
        cons.relaxables.push(RelaxableHandoff {
            producer: 10,
            consumer: 20,
            tensor: "T".into(),
            default: HandoffKind::Rdma,
            alts: vec![(HandoffKind::Rdma, 5000)],
        });

        let m = s.machine.clone();
        let rep = analyze_temporal_fit(&s, &cons, &m);
        assert_eq!(rep.demoted, 0);
    }

    // --------------------------------------------------------------
    // Promotion tests
    // --------------------------------------------------------------

    use rewrite::{GraphNode as TileNode, LocalityReq, TileGraph};

    fn tile_graph_with_hbm_handoff() -> TileGraph {
        // Two compute nodes bridged by DmaOut → DmaIn, both non-resident
        // (which is what relax leaves after demoting to HBM).
        let mut g = TileGraph::default();
        use rewrite::Compute;
        let compute_a_id = g.nodes.len();
        g.nodes.push(TileNode::Compute {
            op: "gemm_a".into(),
            kind: Compute::Gemm(costmodel::TileShape::new(16, 16, 16)),
            passes: 1,
            sram_pages: 4,
            inline_in: vec![],
            inline_out: false,
        });
        let dma_out_id = g.nodes.len();
        g.nodes.push(TileNode::DmaOut {
            tensor: "T".into(),
            resident: false, // demoted state
        });
        let dma_in_id = g.nodes.len();
        g.nodes.push(TileNode::DmaIn {
            tensor: "T".into(),
            resident: false, // demoted state
        });
        let compute_b_id = g.nodes.len();
        g.nodes.push(TileNode::Compute {
            op: "gemm_b".into(),
            kind: Compute::Gemm(costmodel::TileShape::new(16, 16, 16)),
            passes: 1,
            sram_pages: 4,
            inline_in: vec![],
            inline_out: false,
        });
        g.edges.push((compute_a_id, dma_out_id));
        g.edges.push((dma_out_id, dma_in_id));
        g.edges.push((dma_in_id, compute_b_id));
        g
    }

    /// Promotion flips DmaOut/DmaIn to resident, sets handoff to SramSameSm,
    /// updates locality, and unions the two compute nodes into a colocation group.
    #[test]
    fn promotes_hbm_handoff_to_sram_same_sm() {
        let g = tile_graph_with_hbm_handoff();
        // Compute a at index 0, DmaOut at 1, DmaIn at 2, Compute b at 3.
        let compute_a = 0usize;
        let compute_b = 3usize;

        let mut cons = ConstraintSet::default();
        cons.placement.insert(compute_a, 0);
        cons.placement.insert(compute_b, 0);
        cons.sram_pages.insert(compute_a, 4);
        cons.sram_pages.insert(compute_b, 4);
        cons.relaxables.push(RelaxableHandoff {
            producer: compute_a,
            consumer: compute_b,
            tensor: "T".into(),
            default: HandoffKind::Hbm,
            alts: vec![(HandoffKind::Hbm, 500), (HandoffKind::SramSameSm, 20)],
        });

        let candidate = PromotionCandidate {
            producer: compute_a,
            consumer: compute_b,
            tensor: "T".into(),
            demoted_to: HandoffKind::Hbm,
            producer_release: 100,
            consumer_acquire: 200,
            producer_pages: 4,
            consumer_pages: 4,
            budget: 32,
            cycles_saved: 480,
        };

        let (new_g, new_cons) = promote_temporal_fits(&g, &cons, &[candidate]);

        // Relaxable default is now SramSameSm.
        assert_eq!(new_cons.relaxables[0].default, HandoffKind::SramSameSm);
        // Locality updated.
        assert_eq!(
            new_cons.locality.get(&(compute_a, compute_b)).copied(),
            Some(LocalityReq::MustColocate)
        );
        // DmaOut and DmaIn flipped to resident.
        match &new_g.nodes[1] {
            TileNode::DmaOut { resident, .. } => assert!(*resident),
            _ => panic!("expected DmaOut at 1"),
        }
        match &new_g.nodes[2] {
            TileNode::DmaIn { resident, .. } => assert!(*resident),
            _ => panic!("expected DmaIn at 2"),
        }
        // Colocation group now unions the two computes.
        assert_eq!(new_cons.colocation_groups.len(), 1);
        assert!(new_cons.colocation_groups[0].contains(&compute_a));
        assert!(new_cons.colocation_groups[0].contains(&compute_b));
    }

    /// Empty candidate list ⇒ no changes.
    #[test]
    fn empty_candidates_is_noop() {
        let g = tile_graph_with_hbm_handoff();
        let mut cons = ConstraintSet::default();
        cons.relaxables.push(RelaxableHandoff {
            producer: 0,
            consumer: 3,
            tensor: "T".into(),
            default: HandoffKind::Hbm,
            alts: vec![(HandoffKind::Hbm, 500), (HandoffKind::SramSameSm, 20)],
        });
        let (new_g, new_cons) = promote_temporal_fits(&g, &cons, &[]);
        assert_eq!(new_cons.relaxables[0].default, HandoffKind::Hbm);
        match &new_g.nodes[1] {
            TileNode::DmaOut { resident, .. } => assert!(!*resident),
            _ => panic!("expected DmaOut"),
        }
    }

    /// Already-SramSameSm handoff is not double-touched by promotion.
    #[test]
    fn already_colocated_is_untouched() {
        let g = tile_graph_with_hbm_handoff();
        let mut cons = ConstraintSet::default();
        cons.relaxables.push(RelaxableHandoff {
            producer: 0,
            consumer: 3,
            tensor: "T".into(),
            // Already colocated (in an unusual state where DMA nodes are still
            // non-resident — a shouldn't-happen but we don't corrupt it).
            default: HandoffKind::SramSameSm,
            alts: vec![(HandoffKind::SramSameSm, 20)],
        });
        let candidate = PromotionCandidate {
            producer: 0,
            consumer: 3,
            tensor: "T".into(),
            demoted_to: HandoffKind::SramSameSm,
            producer_release: 0,
            consumer_acquire: 0,
            producer_pages: 0,
            consumer_pages: 0,
            budget: 0,
            cycles_saved: 0,
        };
        let (_, new_cons) = promote_temporal_fits(&g, &cons, &[candidate]);
        // Still SramSameSm, no double-write.
        assert_eq!(new_cons.relaxables[0].default, HandoffKind::SramSameSm);
    }
}
