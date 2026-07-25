//! Pass §8.3 — DMA-in prefetch hoisting.
//!
//! `list_schedule` produces per-resource FIFO streams. On memory-bound
//! workloads, DMA-in tasks that could logically issue much earlier are pinned
//! behind unrelated tasks in stream order, closing the compute/DMA overlap
//! gap between `sim::simulate::makespan` and `sim::simulate::ideal_makespan`.
//!
//! This pass reorders each stream so every `TaskKind::DmaIn` sits right after
//! its **last** stream-local predecessor — the last task on the same stream
//! that it actually depends on (via `TaskGraph.edges`). Tasks with no
//! stream-local predecessor are hoisted to position 0.
//!
//! # Safety
//!
//! A DMA-in `d` is only moved earlier past tasks it does **not** depend on
//! (per `TaskGraph.edges`). The DAG happens-before is preserved by
//! construction — the counter waits `d` carries (via `Packet.wait`) travel
//! with `d`, so any counter-gated dependency remains enforced at runtime.
//!
//! Correspondingly, the Lean spec's `protocol_covers_deps` still holds on the
//! reduced schedule: every data-dep edge remains covered by either the
//! (unchanged) counter graph or the new resource order. `--lean-verify` in
//! plowc gives a per-bucket witness of this.

use std::collections::{HashMap, HashSet};

use crate::expand::{TaskGraph, TaskId, TaskKind};
use crate::interval::Cycle;
use crate::passes::{Packet, Schedule};
use crate::resource::ResourceId;

/// Per-bucket summary of what got hoisted.
#[derive(Debug, Clone, Default)]
pub struct PrefetchReport {
    /// The DMA-in tasks whose stream position moved earlier.
    pub hoisted: Vec<TaskId>,
    /// Number of DMA-in tasks that were already at their earliest legal
    /// position (nothing to do).
    pub already_optimal: usize,
    /// Sum of (old_position - new_position) over hoisted tasks. High values
    /// mean bigger overlap wins.
    pub total_slot_advance: usize,
}

impl PrefetchReport {
    pub fn avg_slot_advance(&self) -> f64 {
        if self.hoisted.is_empty() {
            0.0
        } else {
            self.total_slot_advance as f64 / self.hoisted.len() as f64
        }
    }
}

/// Hoist every DMA-in in every stream to the earliest legal position it can
/// take. Returns a re-consistent [`Schedule`] with `streams`, `packets`, and
/// `starts` all updated together.
pub fn hoist_prefetches(tasks: &TaskGraph, sched: &Schedule) -> (Schedule, PrefetchReport) {
    // Predecessors from data edges: p must complete before b starts.
    let n = tasks.tasks.len();
    let mut preds_of: Vec<HashSet<TaskId>> = vec![HashSet::new(); n];
    for &(a, b) in &tasks.edges {
        preds_of[b].insert(a);
    }

    let mut out = sched.clone();
    let mut hoisted = Vec::new();
    let mut already_optimal = 0usize;
    let mut total_slot_advance = 0usize;

    let resources: Vec<ResourceId> = out.streams.keys().copied().collect();
    for res in resources {
        let (new_stream, per_res_hoisted, per_res_already, per_res_advance) =
            reorder_stream_dma_ins(tasks, &out.streams[&res], &preds_of);
        // Only rebuild the resource if anything actually moved.
        if per_res_hoisted.is_empty() {
            already_optimal += per_res_already;
            continue;
        }
        rebuild_resource(&mut out, res, new_stream);
        hoisted.extend(per_res_hoisted);
        already_optimal += per_res_already;
        total_slot_advance += per_res_advance;
    }

    // Once positions have moved on possibly several streams, cycles across
    // streams may have shifted. Recompute `starts` from the new order using
    // a fixed-point sweep bounded by task count (a DAG has ≤ n depth).
    recompute_starts(tasks, &mut out);

    // The shifted starts invalidate the per-slot SRAM/TMEM assignments made
    // against the pre-hoist timeline (two tasks could otherwise hold the same
    // page slot over overlapping windows, and emit would write stale slot ids
    // into packets). Re-run Pass B against the new starts; spills are
    // recounted the same way the original scheduling path counts them (a
    // task that no longer fits simply loses its slot assignment).
    let mut succ_of: Vec<Vec<TaskId>> = vec![Vec::new(); n];
    for &(a, b) in &tasks.edges {
        succ_of[a].push(b);
    }
    let (sram_slots, spills) =
        crate::passes::allocate_sram(tasks, &succ_of, &out.starts, &out.placement);
    let (tmem_slots, tmem_spills) =
        crate::passes::allocate_tmem(tasks, &out.starts, &out.placement);
    out.sram_slots = sram_slots;
    out.spills = spills;
    out.tmem_slots = tmem_slots;
    out.tmem_spills = tmem_spills;

    (
        out,
        PrefetchReport {
            hoisted,
            already_optimal,
            total_slot_advance,
        },
    )
}

/// Rebuild `stream` by pulling every `DmaIn` back to the position right after
/// its last stream-local predecessor. Returns the new order (task list only —
/// cycles are recomputed later) plus stats.
///
/// Preserves the relative order of non-DMA-in tasks and of DMA-in tasks whose
/// earliest legal position collides (they land at the same anchor).
fn reorder_stream_dma_ins(
    tasks: &TaskGraph,
    stream: &[(TaskId, Cycle)],
    preds_of: &[HashSet<TaskId>],
) -> (Vec<TaskId>, Vec<TaskId>, usize, usize) {
    let ids: Vec<TaskId> = stream.iter().map(|(t, _)| *t).collect();
    let ids_set: HashSet<TaskId> = ids.iter().copied().collect();

    // Non-DMA-in tasks anchor the stream — collect them in original order.
    // For each DMA-in, find the LAST non-DMA-in-in-stream position it depends
    // on. Insert DMA-ins right after their anchors, preserving their original
    // relative order.
    let non_dma: Vec<TaskId> = ids
        .iter()
        .copied()
        .filter(|&t| tasks.tasks[t].kind != TaskKind::DmaIn)
        .collect();
    let non_dma_pos: HashMap<TaskId, usize> = non_dma
        .iter()
        .enumerate()
        .map(|(i, &t)| (t, i))
        .collect();

    // For each DMA-in, its "anchor" is the index into `non_dma` of its last
    // in-stream predecessor. `-1` == no anchor ⇒ goes at the front.
    #[derive(Clone, Copy)]
    struct Insertion {
        dma: TaskId,
        anchor: i64,   // -1 for "before everything"
        original_idx: usize,
    }
    let mut inserts = Vec::new();
    for (orig_idx, &t) in ids.iter().enumerate() {
        if tasks.tasks[t].kind != TaskKind::DmaIn {
            continue;
        }
        let mut anchor: i64 = -1;
        for &p in &preds_of[t] {
            if !ids_set.contains(&p) {
                continue;
            }
            // We only anchor on non-DMA-in preds. A DMA-in pred is another
            // hoistable, and we resolve those in a second pass below.
            if tasks.tasks[p].kind == TaskKind::DmaIn {
                continue;
            }
            if let Some(&pos) = non_dma_pos.get(&p) {
                anchor = anchor.max(pos as i64);
            }
        }
        inserts.push(Insertion { dma: t, anchor, original_idx: orig_idx });
    }

    // Also account for DMA-in → DMA-in edges: a DMA-in that depends on another
    // DMA-in must come after it. Bump anchor to the max anchor of its DMA-in
    // preds (which have themselves been placed against the same non-DMA-in
    // grid). One relaxation sweep suffices because the DMA-in dep sub-graph
    // is acyclic (inherited from `TaskGraph`).
    let mut anchor_of: HashMap<TaskId, i64> =
        inserts.iter().map(|i| (i.dma, i.anchor)).collect();
    let mut changed = true;
    let mut iters = 0;
    while changed && iters < ids.len() + 1 {
        changed = false;
        iters += 1;
        for ins in inserts.iter_mut() {
            let dma = ins.dma;
            for &p in &preds_of[dma] {
                if tasks.tasks.get(p).map(|t| t.kind) != Some(TaskKind::DmaIn) {
                    continue;
                }
                if let Some(&pa) = anchor_of.get(&p) {
                    if pa > ins.anchor {
                        ins.anchor = pa;
                        anchor_of.insert(dma, pa);
                        changed = true;
                    }
                }
            }
        }
    }

    // Group insertions by anchor and stably order within each group by their
    // original stream index.
    inserts.sort_by_key(|i| (i.anchor, i.original_idx));

    // Weave the new stream: for each non-DMA-in slot, prepend any DMA-in whose
    // anchor is `slot - 1`, then the slot's own task.
    let mut new_order: Vec<TaskId> = Vec::with_capacity(ids.len());
    let mut ins_cursor = 0usize;
    // First bucket: anchor == -1.
    while ins_cursor < inserts.len() && inserts[ins_cursor].anchor < 0 {
        new_order.push(inserts[ins_cursor].dma);
        ins_cursor += 1;
    }
    for (slot, &nd) in non_dma.iter().enumerate() {
        new_order.push(nd);
        while ins_cursor < inserts.len() && inserts[ins_cursor].anchor == slot as i64 {
            new_order.push(inserts[ins_cursor].dma);
            ins_cursor += 1;
        }
    }
    // Any leftover DMA-ins whose anchor exceeds the non_dma range (shouldn't
    // happen, but defensively keep them appended).
    while ins_cursor < inserts.len() {
        new_order.push(inserts[ins_cursor].dma);
        ins_cursor += 1;
    }

    debug_assert_eq!(new_order.len(), ids.len(), "no task lost during reorder");

    // Compute stats.
    let old_pos: HashMap<TaskId, usize> =
        ids.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let new_pos: HashMap<TaskId, usize> =
        new_order.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let mut hoisted = Vec::new();
    let mut already_optimal = 0usize;
    let mut total_slot_advance = 0usize;
    for ins in &inserts {
        let op = old_pos[&ins.dma];
        let np = new_pos[&ins.dma];
        if np < op {
            hoisted.push(ins.dma);
            total_slot_advance += op - np;
        } else {
            already_optimal += 1;
        }
    }

    (new_order, hoisted, already_optimal, total_slot_advance)
}

/// Rewrite `Schedule.streams[res]` and `Schedule.packets[res]` in the new
/// order. Cycles are placeholders here (0); `recompute_starts` fills them.
fn rebuild_resource(out: &mut Schedule, res: ResourceId, new_order: Vec<TaskId>) {
    let old_packets = out.packets.remove(&res).unwrap_or_default();
    let pkt_by_task: HashMap<TaskId, Packet> =
        old_packets.into_iter().map(|p| (p.task, p)).collect();

    let mut new_stream: Vec<(TaskId, Cycle)> = Vec::with_capacity(new_order.len());
    let mut new_packets: Vec<Packet> = Vec::with_capacity(new_order.len());
    for t in new_order {
        let pkt = pkt_by_task
            .get(&t)
            .cloned()
            .expect("every reordered task must have a packet");
        new_stream.push((t, pkt.start));
        new_packets.push(pkt);
    }
    out.streams.insert(res, new_stream);
    out.packets.insert(res, new_packets);
}

/// Fixed-point recompute of `Schedule.starts`, `Schedule.streams[*]` cycles,
/// and `Schedule.packets[*]` start cycles from the current stream orderings
/// and `TaskGraph.edges`. Bounded by `tasks.len()` iterations (DAG depth).
fn recompute_starts(tasks: &TaskGraph, out: &mut Schedule) {
    let n = tasks.tasks.len();

    // preds_of[t] = tasks that must complete before t starts.
    let mut preds_of: Vec<Vec<TaskId>> = vec![Vec::new(); n];
    for &(a, b) in &tasks.edges {
        preds_of[b].push(a);
    }

    // Cache the per-resource task order and each task's prev-on-stream, if any.
    let mut prev_on_stream: HashMap<TaskId, Option<TaskId>> = HashMap::new();
    for stream in out.streams.values() {
        let mut prev: Option<TaskId> = None;
        for &(t, _) in stream {
            prev_on_stream.insert(t, prev);
            prev = Some(t);
        }
    }

    let mut starts: Vec<Cycle> = out.starts.clone();
    if starts.len() < n {
        starts.resize(n, 0);
    }
    for _ in 0..(n + 1) {
        let mut changed = false;
        for t in 0..n {
            let dur = tasks.tasks[t].dur;
            let mut earliest: Cycle = 0;
            for &p in &preds_of[t] {
                earliest = earliest.max(starts[p].saturating_add(tasks.tasks[p].dur));
            }
            if let Some(prev) = prev_on_stream.get(&t).copied().flatten() {
                earliest = earliest
                    .max(starts[prev].saturating_add(tasks.tasks[prev].dur));
            }
            if earliest != starts[t] {
                starts[t] = earliest;
                changed = true;
            }
            let _ = dur;
        }
        if !changed {
            break;
        }
    }

    // Write back to streams & packets so all three views stay consistent.
    for stream in out.streams.values_mut() {
        for (t, c) in stream.iter_mut() {
            *c = starts[*t];
        }
    }
    for stream in out.packets.values_mut() {
        for pkt in stream.iter_mut() {
            pkt.start = starts[pkt.task];
        }
    }
    out.starts = starts;
    // makespan is the max end-cycle across all tasks.
    out.makespan = (0..n)
        .map(|t| out.starts[t].saturating_add(tasks.tasks[t].dur))
        .max()
        .unwrap_or(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expand::Task;
    use crate::passes::{Counter, PacketKind, Scope};
    use std::collections::HashMap;

    fn task(id: usize, kind: TaskKind, dur: Cycle) -> Task {
        Task {
            node: id,
            op: "x".into(),
            unit: 0,
            kind,
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

    fn packet(t: TaskId, start: Cycle, wait: &[usize], succs: &[usize], kind: PacketKind) -> Packet {
        Packet {
            task: t,
            op: "x".into(),
            kind,
            start,
            wait: wait.to_vec(),
            successors: succs.to_vec(),
        }
    }

    /// The consistency check every test post-condition must satisfy: for each
    /// resource, `streams[r]` and `packets[r]` agree on task order and cycle,
    /// and `starts[t]` matches wherever `t` appears.
    fn assert_streams_packets_starts_consistent(sched: &Schedule) {
        for (r, stream) in &sched.streams {
            let packets = sched.packets.get(r).expect("packets mirror streams");
            assert_eq!(stream.len(), packets.len(),
                "streams and packets differ in length for {:?}", r);
            for (i, ((t, c), p)) in stream.iter().zip(packets.iter()).enumerate() {
                assert_eq!(*t, p.task,
                    "position {i} in {:?}: stream task {t} vs packet task {}", r, p.task);
                assert_eq!(*c, p.start,
                    "position {i} in {:?}: stream cycle {c} vs packet start {}", r, p.start);
                assert_eq!(*c, sched.starts[*t],
                    "task {t} on {:?}: stream cycle {c} vs starts[{t}] {}",
                    r, sched.starts[*t]);
            }
        }
    }

    /// Data-dep respect: every edge `(a, b)` in `tg.edges` must satisfy
    /// `starts[a] + dur[a] <= starts[b]` after the pass.
    fn assert_data_deps_respected(tg: &TaskGraph, sched: &Schedule) {
        for &(a, b) in &tg.edges {
            let end_a = sched.starts[a] + tg.tasks[a].dur;
            assert!(end_a <= sched.starts[b],
                "edge ({a}, {b}) violated: a ends at {end_a}, b starts at {}",
                sched.starts[b]);
        }
    }

    /// A single stream: [compute_0, compute_1, dma_in_2, compute_3]. Only edge
    /// is (compute_0 → dma_in_2). `dma_in_2` should be hoisted to right after
    /// compute_0.
    #[test]
    fn hoists_dma_in_past_unrelated_compute() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            task(0, TaskKind::Compute, 10),
            task(1, TaskKind::Compute, 10),  // unrelated to dma_in
            task(2, TaskKind::DmaIn,   10),
            task(3, TaskKind::Compute, 10),
        ];
        tg.edges = vec![(0, 2)]; // only dep: dma_in_2 needs compute_0's data

        let sm = ResourceId::Sm(0, 0);
        let sched = Schedule {
            streams: HashMap::from([(sm, vec![
                (0usize, 0), (1, 10), (2, 20), (3, 30),
            ])]),
            packets: HashMap::from([(sm, vec![
                packet(0, 0,  &[], &[], PacketKind::Compute),
                packet(1, 10, &[], &[], PacketKind::Compute),
                packet(2, 20, &[], &[], PacketKind::TmaIn),
                packet(3, 30, &[], &[], PacketKind::Compute),
            ])]),
            counters: vec![],
            placement: HashMap::from([(0usize, sm), (1, sm), (2, sm), (3, sm)]),
            starts: vec![0, 10, 20, 30],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 40,
        };

        let (out, rep) = hoist_prefetches(&tg, &sched);
        assert_streams_packets_starts_consistent(&out);
        assert_data_deps_respected(&tg, &out);
        assert_eq!(rep.hoisted, vec![2]);
        // Task 2 moved from position 2 to position 1.
        let order: Vec<TaskId> = out.streams[&sm].iter().map(|(t, _)| *t).collect();
        assert_eq!(order, vec![0, 2, 1, 3]);
        assert_eq!(rep.total_slot_advance, 1);
        // dma_in_2 now starts at cycle 10 (right after compute_0).
        assert_eq!(out.starts[2], 10);
        // Task 1 got pushed back one slot (now after dma_in_2).
        assert_eq!(out.starts[1], 20);
        assert_eq!(out.starts[3], 30);
    }

    /// A DMA-in with no stream-local predecessor: goes to position 0.
    #[test]
    fn dma_in_with_no_local_pred_goes_to_position_zero() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            task(0, TaskKind::Compute, 10),
            task(1, TaskKind::Compute, 10),
            task(2, TaskKind::DmaIn,   10),
        ];
        // No edges — the DMA-in is fully independent.
        let sm = ResourceId::Sm(0, 0);
        let sched = Schedule {
            streams: HashMap::from([(sm, vec![
                (0usize, 0), (1, 10), (2, 20),
            ])]),
            packets: HashMap::from([(sm, vec![
                packet(0, 0,  &[], &[], PacketKind::Compute),
                packet(1, 10, &[], &[], PacketKind::Compute),
                packet(2, 20, &[], &[], PacketKind::TmaIn),
            ])]),
            counters: vec![],
            placement: HashMap::from([(0usize, sm), (1, sm), (2, sm)]),
            starts: vec![0, 10, 20],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 30,
        };
        let (out, rep) = hoist_prefetches(&tg, &sched);
        assert_streams_packets_starts_consistent(&out);
        assert_data_deps_respected(&tg, &out);
        assert_eq!(rep.hoisted, vec![2]);
        let order: Vec<TaskId> = out.streams[&sm].iter().map(|(t, _)| *t).collect();
        assert_eq!(order, vec![2, 0, 1], "no anchor ⇒ front");
        assert_eq!(out.starts[2], 0);
    }

    /// DMA-in already at its earliest position: nothing to do.
    #[test]
    fn already_optimal_dma_in_untouched() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            task(0, TaskKind::Compute, 10),
            task(1, TaskKind::DmaIn,   10),   // depends on compute_0, right after ⇒ optimal
            task(2, TaskKind::Compute, 10),
        ];
        tg.edges = vec![(0, 1)];
        let sm = ResourceId::Sm(0, 0);
        let sched = Schedule {
            streams: HashMap::from([(sm, vec![
                (0usize, 0), (1, 10), (2, 20),
            ])]),
            packets: HashMap::from([(sm, vec![
                packet(0, 0,  &[], &[], PacketKind::Compute),
                packet(1, 10, &[], &[], PacketKind::TmaIn),
                packet(2, 20, &[], &[], PacketKind::Compute),
            ])]),
            counters: vec![],
            placement: HashMap::from([(0usize, sm), (1, sm), (2, sm)]),
            starts: vec![0, 10, 20],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 30,
        };
        let (out, rep) = hoist_prefetches(&tg, &sched);
        assert_streams_packets_starts_consistent(&out);
        assert_data_deps_respected(&tg, &out);
        assert!(rep.hoisted.is_empty(), "no hoist needed");
        assert_eq!(rep.already_optimal, 1);
        let order: Vec<TaskId> = out.streams[&sm].iter().map(|(t, _)| *t).collect();
        assert_eq!(order, vec![0, 1, 2]);
    }

    /// Two DMA-ins where the second depends on the first. Both should hoist,
    /// preserving the dep-imposed relative order.
    #[test]
    fn two_dma_ins_with_dep_preserve_relative_order() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            task(0, TaskKind::Compute, 10),  // unrelated
            task(1, TaskKind::DmaIn,   10),  // no deps
            task(2, TaskKind::DmaIn,   10),  // depends on task 1
            task(3, TaskKind::Compute, 10),  // consumes both, gates the end
        ];
        tg.edges = vec![(1, 2), (1, 3), (2, 3)];
        let sm = ResourceId::Sm(0, 0);
        let sched = Schedule {
            streams: HashMap::from([(sm, vec![
                (0usize, 0), (1, 10), (2, 20), (3, 30),
            ])]),
            packets: HashMap::from([(sm, vec![
                packet(0, 0,  &[], &[], PacketKind::Compute),
                packet(1, 10, &[], &[], PacketKind::TmaIn),
                packet(2, 20, &[], &[], PacketKind::TmaIn),
                packet(3, 30, &[], &[], PacketKind::Compute),
            ])]),
            counters: vec![],
            placement: HashMap::from([(0usize, sm), (1, sm), (2, sm), (3, sm)]),
            starts: vec![0, 10, 20, 30],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 40,
        };
        let (out, rep) = hoist_prefetches(&tg, &sched);
        assert_streams_packets_starts_consistent(&out);
        assert_data_deps_respected(&tg, &out);
        // Both DMA-ins hoisted to the front (no non-DMA-in preds); task 2's
        // dep on task 1 keeps them in that order.
        let order: Vec<TaskId> = out.streams[&sm].iter().map(|(t, _)| *t).collect();
        assert!(order.starts_with(&[1, 2]),
            "DMA-ins must land at front in dep order; got {:?}", order);
        assert!(rep.hoisted.contains(&1));
        assert!(rep.hoisted.contains(&2));
    }

    /// Counter waits carried on the hoisted packet stay attached (byte-for-byte
    /// equality of `wait` / `successors` after the reorder).
    #[test]
    fn counter_waits_travel_with_the_hoisted_packet() {
        let mut tg = TaskGraph::default();
        tg.tasks = vec![
            task(0, TaskKind::Compute, 10),
            task(1, TaskKind::Compute, 10),  // gates dma_in_2 via counter, but no data edge
            task(2, TaskKind::DmaIn,   10),
        ];
        tg.edges = vec![]; // no data edges — the counter is the only ordering
        let sm = ResourceId::Sm(0, 0);
        let counter = Counter {
            id: 42,
            threshold: 1,
            scope: Scope::IntraGpu,
            producer_node: 1,
            consumer_node: 2,
        };
        let sched = Schedule {
            streams: HashMap::from([(sm, vec![
                (0usize, 0), (1, 10), (2, 20),
            ])]),
            packets: HashMap::from([(sm, vec![
                packet(0, 0,  &[],   &[],   PacketKind::Compute),
                packet(1, 10, &[],   &[42], PacketKind::Compute),
                packet(2, 20, &[42], &[],   PacketKind::TmaIn),
            ])]),
            counters: vec![counter],
            placement: HashMap::from([(0usize, sm), (1, sm), (2, sm)]),
            starts: vec![0, 10, 20],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 30,
        };
        let (out, _rep) = hoist_prefetches(&tg, &sched);
        assert_streams_packets_starts_consistent(&out);
        // The DMA-in still holds `wait: [42]` — the counter travels with it,
        // so runtime still respects the counter gate even if positions shift.
        let dma_pkt = out
            .packets
            .get(&sm)
            .unwrap()
            .iter()
            .find(|p| p.task == 2)
            .unwrap();
        assert_eq!(dma_pkt.wait, vec![42]);
    }
}
