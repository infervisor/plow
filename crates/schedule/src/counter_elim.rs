//! Pass §8.1 — counter elimination.
//!
//! Drops counters that are redundant with resource-order. Provably safe by the
//! universal lemma `resourceOrdered ⊆ happensBefore` proven in
//! `lean-plow/Plow/Protocol.lean` (specifically `WellFormed.resForward` and the
//! `.resource` constructor of `happensBefore`).
//!
//! A counter is redundant iff:
//!
//! 1. Every producer and every consumer sit on the same [`ResourceId`], AND
//! 2. `max(producer.stream_idx) < min(consumer.stream_idx)` — i.e. every
//!    producer issues strictly before every consumer in that stream's FIFO
//!    order.
//!
//! If those hold, every edge the counter represents is already ordered by the
//! stream. Dropping the counter frees a runtime atomic per edge without
//! weakening the schedule's happens-before.
//!
//! Emit-side effect: the counter's id is removed from `counters` and from
//! every packet's `wait` / `successors` list (the runtime never sees it).
//! We do *not* renumber remaining counter ids; the emitter tolerates gaps and
//! renumbering would just make debugging harder.

use std::collections::{HashMap, HashSet};

use crate::expand::TaskId;
use crate::passes::{Packet, Schedule};
use crate::resource::ResourceId;

/// Which counters got dropped and why. Populated by
/// [`eliminate_redundant_counters`].
#[derive(Debug, Clone, Default)]
pub struct EliminationReport {
    /// Counter ids that were removed.
    pub eliminated: Vec<usize>,
    /// Counter count that survived.
    pub kept: usize,
    /// Total counters before the pass.
    pub before: usize,
}

impl EliminationReport {
    pub fn savings_pct(&self) -> f64 {
        if self.before == 0 {
            0.0
        } else {
            100.0 * self.eliminated.len() as f64 / self.before as f64
        }
    }
}

/// Build (`stream_idx`, `resource`) lookups from the schedule's stream lists.
/// Every scheduled task appears in exactly one resource stream.
fn task_stream_pos(sched: &Schedule) -> HashMap<TaskId, (ResourceId, usize)> {
    let mut out = HashMap::new();
    for (&res, stream) in &sched.streams {
        for (idx, &(task, _cycle)) in stream.iter().enumerate() {
            out.insert(task, (res, idx));
        }
    }
    out
}

/// Producers of each counter (tasks that increment it) and consumers (tasks
/// that wait on it), scanned across every resource's packet list.
fn producers_consumers(sched: &Schedule) -> (HashMap<usize, Vec<TaskId>>, HashMap<usize, Vec<TaskId>>) {
    let mut producers: HashMap<usize, Vec<TaskId>> = HashMap::new();
    let mut consumers: HashMap<usize, Vec<TaskId>> = HashMap::new();
    for stream in sched.packets.values() {
        for Packet { task, wait, successors, .. } in stream {
            for &c in successors {
                producers.entry(c).or_default().push(*task);
            }
            for &c in wait {
                consumers.entry(c).or_default().push(*task);
            }
        }
    }
    // Dedup — the same task can appear multiple times if the emitter listed a
    // counter more than once.
    for v in producers.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    for v in consumers.values_mut() {
        v.sort_unstable();
        v.dedup();
    }
    (producers, consumers)
}

/// Decide whether a counter is redundant with resource-order.
fn is_redundant(
    producers: &[TaskId],
    consumers: &[TaskId],
    pos: &HashMap<TaskId, (ResourceId, usize)>,
) -> bool {
    if producers.is_empty() || consumers.is_empty() {
        // A counter with no producer or no consumer is broken (satisfiability
        // was expected to catch it). Don't touch it — let downstream fail
        // rather than mask the bug here.
        return false;
    }
    let mut resources: HashSet<ResourceId> = HashSet::new();
    let mut max_p: Option<usize> = None;
    let mut min_c: Option<usize> = None;
    for &t in producers {
        let (r, idx) = pos.get(&t).copied().unwrap_or_else(|| return_default());
        resources.insert(r);
        max_p = Some(max_p.map_or(idx, |m| m.max(idx)));
    }
    for &t in consumers {
        let (r, idx) = pos.get(&t).copied().unwrap_or_else(|| return_default());
        resources.insert(r);
        min_c = Some(min_c.map_or(idx, |m| m.min(idx)));
    }
    // All producers + consumers pinned to one resource; every producer strictly
    // before every consumer in stream order ⇒ redundant.
    resources.len() == 1
        && max_p.zip(min_c).is_some_and(|(mp, mc)| mp < mc)
}

/// Fallback when a task isn't in any stream — treat as its own resource so
/// nothing gets falsely eliminated. Uses u64::MAX / usize::MAX as sentinels.
fn return_default() -> (ResourceId, usize) {
    (ResourceId::Host(usize::MAX), usize::MAX)
}

/// Apply the pass to `sched` and return a modified schedule + report. The
/// input is left untouched; caller decides whether to install the result.
pub fn eliminate_redundant_counters(sched: &Schedule) -> (Schedule, EliminationReport) {
    let pos = task_stream_pos(sched);
    let (producers, consumers) = producers_consumers(sched);

    let mut eliminated = Vec::new();
    for counter in &sched.counters {
        let ps = producers.get(&counter.id).cloned().unwrap_or_default();
        let cs = consumers.get(&counter.id).cloned().unwrap_or_default();
        if is_redundant(&ps, &cs, &pos) {
            eliminated.push(counter.id);
        }
    }
    let dropped: HashSet<usize> = eliminated.iter().copied().collect();

    let mut out = sched.clone();
    out.counters.retain(|c| !dropped.contains(&c.id));
    for stream in out.packets.values_mut() {
        for pkt in stream.iter_mut() {
            pkt.wait.retain(|c| !dropped.contains(c));
            pkt.successors.retain(|c| !dropped.contains(c));
        }
    }

    let report = EliminationReport {
        eliminated,
        kept: out.counters.len(),
        before: sched.counters.len(),
    };
    (out, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::{Counter, PacketKind, Scope};
    use std::collections::HashMap;

    fn stream_of(pairs: &[(TaskId, u64)]) -> Vec<(TaskId, u64)> {
        pairs.to_vec()
    }

    fn packet(task: TaskId, wait: &[usize], succs: &[usize]) -> Packet {
        Packet {
            task,
            op: "x".into(),
            kind: PacketKind::Compute,
            start: 0,
            wait: wait.to_vec(),
            successors: succs.to_vec(),
        }
    }

    /// Task 0 (SM 0:0 pos 0) → task 1 (SM 0:0 pos 1). Counter 0 gates the dep,
    /// but resource-order already covers it — the pass must eliminate it.
    #[test]
    fn drops_intra_stream_counter() {
        let res = ResourceId::Sm(0, 0);
        let sched = Schedule {
            streams: HashMap::from([(res, stream_of(&[(0, 0), (1, 10)]))]),
            packets: HashMap::from([(
                res,
                vec![packet(0, &[], &[0]), packet(1, &[0], &[])],
            )]),
            counters: vec![Counter {
                id: 0,
                threshold: 1,
                scope: Scope::IntraSm,
                producer_node: 0,
                consumer_node: 1,
            }],
            placement: HashMap::from([(0usize, res), (1usize, res)]),
            starts: vec![0, 10],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 20,
        };
        let (out, rep) = eliminate_redundant_counters(&sched);
        assert_eq!(rep.eliminated, vec![0]);
        assert_eq!(rep.kept, 0);
        assert!(out.counters.is_empty());
        assert!(out.packets[&res][0].successors.is_empty());
        assert!(out.packets[&res][1].wait.is_empty());
    }

    /// Cross-resource counter (SM producer → DMA consumer): resource-order
    /// can't cover this, so the pass must keep the counter.
    #[test]
    fn keeps_cross_resource_counter() {
        let r0 = ResourceId::Sm(0, 0);
        let r1 = ResourceId::Dma(0, 0);
        let sched = Schedule {
            streams: HashMap::from([
                (r0, stream_of(&[(0, 0)])),
                (r1, stream_of(&[(1, 10)])),
            ]),
            packets: HashMap::from([
                (r0, vec![packet(0, &[], &[0])]),
                (r1, vec![packet(1, &[0], &[])]),
            ]),
            counters: vec![Counter {
                id: 0,
                threshold: 1,
                scope: Scope::IntraGpu,
                producer_node: 0,
                consumer_node: 1,
            }],
            placement: HashMap::from([(0usize, r0), (1usize, r1)]),
            starts: vec![0, 10],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 20,
        };
        let (out, rep) = eliminate_redundant_counters(&sched);
        assert!(rep.eliminated.is_empty());
        assert_eq!(rep.kept, 1);
        assert_eq!(out.counters.len(), 1);
        assert_eq!(out.packets[&r0][0].successors, vec![0]);
        assert_eq!(out.packets[&r1][0].wait, vec![0]);
    }

    /// Same resource but producer's stream position is AFTER the consumer's:
    /// resource-order actually flows the wrong way, so the counter is needed.
    #[test]
    fn keeps_counter_when_stream_order_reversed() {
        // Consumer at pos 0, producer at pos 1 on the same stream (unusual but
        // legal — e.g. when the scheduler picked this order for a specific
        // reason). The counter is the only thing making the wait correct.
        let res = ResourceId::Sm(0, 0);
        let sched = Schedule {
            streams: HashMap::from([(res, stream_of(&[(1, 0), (0, 10)]))]),
            packets: HashMap::from([(
                res,
                vec![packet(1, &[0], &[]), packet(0, &[], &[0])],
            )]),
            counters: vec![Counter {
                id: 0,
                threshold: 1,
                scope: Scope::IntraSm,
                producer_node: 0,
                consumer_node: 1,
            }],
            placement: HashMap::from([(0usize, res), (1usize, res)]),
            starts: vec![10, 0],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 20,
        };
        let (_, rep) = eliminate_redundant_counters(&sched);
        assert!(rep.eliminated.is_empty(), "counter is only proof of ordering");
    }

    /// Multi-producer coarse-mode counter: all producers on one resource,
    /// all consumers on the same resource, every producer.pos < every
    /// consumer.pos ⇒ redundant.
    #[test]
    fn drops_multi_producer_all_same_resource_ordered() {
        let res = ResourceId::Sm(0, 0);
        let sched = Schedule {
            streams: HashMap::from([(
                res,
                stream_of(&[(0, 0), (1, 5), (2, 10), (3, 15)]),
            )]),
            packets: HashMap::from([(
                res,
                vec![
                    packet(0, &[], &[0]),
                    packet(1, &[], &[0]),
                    packet(2, &[0], &[]),
                    packet(3, &[0], &[]),
                ],
            )]),
            counters: vec![Counter {
                id: 0,
                threshold: 2,
                scope: Scope::IntraSm,
                producer_node: 0,
                consumer_node: 1,
            }],
            placement: HashMap::from([
                (0usize, res),
                (1usize, res),
                (2usize, res),
                (3usize, res),
            ]),
            starts: vec![0, 5, 10, 15],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 20,
        };
        let (_, rep) = eliminate_redundant_counters(&sched);
        assert_eq!(rep.eliminated, vec![0]);
    }
}
