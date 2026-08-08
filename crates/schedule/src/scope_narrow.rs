//! Pass §8.2 — scope narrowing.
//!
//! Post-schedule pass. `build_counters` picks `Scope::IntraSm` only when the
//! producer/consumer nodes are in the *pre-schedule* `colocated` set. That
//! decision is conservative: after `list_schedule` runs, we know the actual
//! per-task SM placement, and some `IntraGpu` counters turn out to have every
//! producer and consumer on the same SM. Downgrading those to `IntraSm`
//! swaps a ~40-cycle L2 atomic for a single-cycle SM-local barrier at runtime
//! (§8.2 of the design notes).
//!
//! # Safety
//!
//! `Scope` is a **runtime memory-visibility attribute**, not part of the
//! happens-before semantics the DAG proofs (`Plow.Protocol.protocol_covers_deps`)
//! reason about. Narrowing a counter's scope only weakens the memory barrier;
//! it never changes which tasks wait on which counters. The narrowing is safe
//! iff every producer and every consumer of the counter actually run on the
//! same SM at runtime — which is precisely what this pass checks against
//! `Schedule.placement`.

use std::collections::{HashMap, HashSet};

use crate::expand::TaskId;
use crate::passes::{Packet, Schedule, Scope};
use crate::resource::ResourceId;

/// How the pass changed the counter list. `narrowed` are counter ids that got
/// downgraded from `IntraGpu` to `IntraSm`; nothing else is touched.
#[derive(Debug, Clone, Default)]
pub struct ScopeNarrowReport {
    pub narrowed: Vec<usize>,
    pub before_intra_gpu: usize,
    pub before_intra_sm: usize,
    pub before_cross_unit: usize,
}

impl ScopeNarrowReport {
    pub fn narrowed_pct(&self) -> f64 {
        let denom = self.before_intra_gpu.max(1);
        100.0 * self.narrowed.len() as f64 / denom as f64
    }
}

fn sm_of(sched: &Schedule, task: TaskId) -> Option<(usize, usize)> {
    match sched.placement.get(&task) {
        Some(ResourceId::Sm(u, s)) => Some((*u, *s)),
        _ => None,
    }
}

/// Walk `sched.packets` to build `counter_id → (producers, consumers)` — the
/// same shape `counter_elim` uses; kept local so the two passes stay
/// independent.
fn producers_consumers(
    sched: &Schedule,
) -> (HashMap<usize, Vec<TaskId>>, HashMap<usize, Vec<TaskId>>) {
    let mut producers: HashMap<usize, Vec<TaskId>> = HashMap::new();
    let mut consumers: HashMap<usize, Vec<TaskId>> = HashMap::new();
    for stream in sched.packets.values() {
        for Packet {
            task,
            wait,
            successors,
            ..
        } in stream
        {
            for &c in successors {
                producers.entry(c).or_default().push(*task);
            }
            for &c in wait {
                consumers.entry(c).or_default().push(*task);
            }
        }
    }
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

/// True iff every producer and every consumer is placed on the *same* SM.
fn all_on_one_sm(producers: &[TaskId], consumers: &[TaskId], sched: &Schedule) -> bool {
    if producers.is_empty() || consumers.is_empty() {
        return false;
    }
    let mut sm_set: HashSet<(usize, usize)> = HashSet::new();
    for &t in producers.iter().chain(consumers.iter()) {
        match sm_of(sched, t) {
            Some(sm) => {
                sm_set.insert(sm);
                if sm_set.len() > 1 {
                    return false;
                }
            }
            None => return false, // non-SM placement (DMA/DPU/Host) — not narrowable
        }
    }
    sm_set.len() == 1
}

/// Downgrade every `IntraGpu` counter whose actual placement is same-SM to
/// `IntraSm`. Returns the modified schedule + a report.
pub fn narrow_scopes(sched: &Schedule) -> (Schedule, ScopeNarrowReport) {
    let (producers, consumers) = producers_consumers(sched);
    let mut narrowed = Vec::new();
    let mut before_intra_gpu = 0;
    let mut before_intra_sm = 0;
    let mut before_cross_unit = 0;
    for c in &sched.counters {
        match c.scope {
            Scope::IntraGpu => before_intra_gpu += 1,
            Scope::IntraSm => before_intra_sm += 1,
            Scope::CrossUnit => before_cross_unit += 1,
        }
        if c.scope != Scope::IntraGpu {
            continue;
        }
        let ps = producers.get(&c.id).cloned().unwrap_or_default();
        let cs = consumers.get(&c.id).cloned().unwrap_or_default();
        if all_on_one_sm(&ps, &cs, sched) {
            narrowed.push(c.id);
        }
    }
    let to_narrow: HashSet<usize> = narrowed.iter().copied().collect();

    let mut out = sched.clone();
    for c in out.counters.iter_mut() {
        if to_narrow.contains(&c.id) {
            c.scope = Scope::IntraSm;
        }
    }

    let report = ScopeNarrowReport {
        narrowed,
        before_intra_gpu,
        before_intra_sm,
        before_cross_unit,
    };
    (out, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::{Counter, PacketKind};
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

    /// Counter with all producers + consumers on SM(0,0): must narrow.
    #[test]
    fn narrows_when_all_on_same_sm() {
        let sm = ResourceId::Sm(0, 3);
        let sched = Schedule {
            streams: HashMap::from([(sm, stream_of(&[(0, 0), (1, 10)]))]),
            packets: HashMap::from([(sm, vec![packet(0, &[], &[0]), packet(1, &[0], &[])])]),
            counters: vec![Counter {
                id: 0,
                threshold: 1,
                scope: Scope::IntraGpu,
                producer_node: 0,
                consumer_node: 1,
            }],
            placement: HashMap::from([(0usize, sm), (1usize, sm)]),
            starts: vec![0, 10],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 20,
        };
        let (out, rep) = narrow_scopes(&sched);
        assert_eq!(rep.narrowed, vec![0]);
        assert_eq!(out.counters[0].scope, Scope::IntraSm);
    }

    /// Producers and consumers on two different SMs of the same unit:
    /// same-node but different SMs ⇒ can't narrow.
    #[test]
    fn keeps_scope_when_across_sms_same_unit() {
        let sm0 = ResourceId::Sm(0, 0);
        let sm1 = ResourceId::Sm(0, 1);
        let sched = Schedule {
            streams: HashMap::from([(sm0, stream_of(&[(0, 0)])), (sm1, stream_of(&[(1, 10)]))]),
            packets: HashMap::from([
                (sm0, vec![packet(0, &[], &[0])]),
                (sm1, vec![packet(1, &[0], &[])]),
            ]),
            counters: vec![Counter {
                id: 0,
                threshold: 1,
                scope: Scope::IntraGpu,
                producer_node: 0,
                consumer_node: 1,
            }],
            placement: HashMap::from([(0usize, sm0), (1usize, sm1)]),
            starts: vec![0, 10],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 20,
        };
        let (out, rep) = narrow_scopes(&sched);
        assert!(rep.narrowed.is_empty());
        assert_eq!(out.counters[0].scope, Scope::IntraGpu);
    }

    /// Producer is SM but consumer is a DMA engine ⇒ non-SM endpoint ⇒ keep.
    #[test]
    fn keeps_scope_when_endpoint_is_dma() {
        let sm = ResourceId::Sm(0, 0);
        let dma = ResourceId::Dma(0, 0);
        let sched = Schedule {
            streams: HashMap::from([(sm, stream_of(&[(0, 0)])), (dma, stream_of(&[(1, 10)]))]),
            packets: HashMap::from([
                (sm, vec![packet(0, &[], &[0])]),
                (dma, vec![packet(1, &[0], &[])]),
            ]),
            counters: vec![Counter {
                id: 0,
                threshold: 1,
                scope: Scope::IntraGpu,
                producer_node: 0,
                consumer_node: 1,
            }],
            placement: HashMap::from([(0usize, sm), (1usize, dma)]),
            starts: vec![0, 10],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 20,
        };
        let (_, rep) = narrow_scopes(&sched);
        assert!(rep.narrowed.is_empty());
    }

    /// IntraSm and CrossUnit counters are never touched, only IntraGpu.
    #[test]
    fn does_not_touch_intra_sm_or_cross_unit() {
        let sm = ResourceId::Sm(0, 3);
        let sched = Schedule {
            streams: HashMap::from([(sm, stream_of(&[(0, 0), (1, 5)]))]),
            packets: HashMap::from([(sm, vec![packet(0, &[], &[0, 1]), packet(1, &[0, 1], &[])])]),
            counters: vec![
                Counter {
                    id: 0,
                    threshold: 1,
                    scope: Scope::IntraSm,
                    producer_node: 0,
                    consumer_node: 1,
                },
                Counter {
                    id: 1,
                    threshold: 1,
                    scope: Scope::CrossUnit,
                    producer_node: 0,
                    consumer_node: 1,
                },
            ],
            placement: HashMap::from([(0usize, sm), (1usize, sm)]),
            starts: vec![0, 5],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 10,
        };
        let (out, rep) = narrow_scopes(&sched);
        assert!(rep.narrowed.is_empty(), "IntraSm/CrossUnit not eligible");
        assert_eq!(out.counters[0].scope, Scope::IntraSm);
        assert_eq!(out.counters[1].scope, Scope::CrossUnit);
    }
}
