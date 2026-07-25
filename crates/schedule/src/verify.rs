//! Verification of the emitted packet stream's synchronization (design §7 — the
//! runtime has no global clock, so correctness *is* the ordering protocol).
//!
//! A tile begins only when **both** conditions hold:
//! * **counters** — every counter it waits on has reached its threshold (a
//!   counter is incremented by each producer on completion, threshold = the
//!   producer count, so "satisfied" means *every* producer finished); and
//! * **in-order issue** — its resource (one SM / DMA engine / DPU / host thread)
//!   runs a sequential queue, so the previous packet on that resource has
//!   completed.
//!
//! Counters carry the *cross-resource* dependencies; in-order issue carries the
//! *same-resource* ones. Together they are the runtime's happens-before. This
//! module proves that relation — as encoded in the [`packet::Program`] byte
//! stream the runtime actually receives — covers every data dependency in the
//! [`TaskGraph`]: no tile can start before *all* of its parents finish.
//!
//! Three layered checks, static then dynamic:
//!
//! 1. **Counter satisfiability.** A counter's threshold must equal the number of
//!    instructions that increment it. Equal ⇒ reaching it means *every* producer
//!    finished (the "all parents" guarantee). Above the producer count it can
//!    never be reached (deadlock); below it, a waiter is released when only
//!    *some* producers are done (a race).
//!
//! 2. **Edge gating.** Every data edge `a → b` must be enforced — either a
//!    counter that `a` increments and `b` waits on, or `a` ordered before `b` on
//!    the same resource. Otherwise `b` could start before `a` completes.
//!
//! 3. **Eager replay.** Execute the protocol, firing every instruction the
//!    instant its waits *and* its resource predecessor allow — the most
//!    aggressive order, the worst case for races. Assert no instruction fires
//!    before all of its data parents have, and that the program runs to
//!    completion (no deadlock). This is the dynamic witness for (1)+(2).
//!
//! The checks run against the **decoded** stream (`Program::decode`), so they
//! validate the wire bytes — not the in-memory builder form.

use crate::emit::{emit_program, issue_order};
use crate::expand::{TaskGraph, TaskId};
use crate::passes::Schedule;
use packet::{Inst, Program};
use rewrite::{ConstraintSet, TileGraph};
use std::collections::{HashMap, HashSet};

/// A clean verification result: what was proven, and the protocol's shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifyReport {
    pub insts: usize,
    pub counters: usize,
    /// Data edges proven to be enforced (by a counter or by resource order).
    pub edges_checked: usize,
    /// Replay depth — rounds of fully-parallel issue (critical-path length of
    /// the happens-before relation).
    pub rounds: usize,
    /// Widest set of instructions ready at once (max achievable parallelism).
    pub max_ready: usize,
}

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum VerifyError {
    #[error("stream has {insts} instructions but the issue order has {order} tasks")]
    LengthMismatch { insts: usize, order: usize },
    #[error("counter {id}: threshold {threshold} exceeds its {producers} producers — unreachable (deadlock)")]
    Unreachable {
        id: u32,
        threshold: u32,
        producers: u32,
    },
    #[error("counter {id}: threshold {threshold} below its {producers} producers — releases on *any*, not *all* (race)")]
    LooseThreshold {
        id: u32,
        threshold: u32,
        producers: u32,
    },
    #[error("instruction references counter {0}, which is not in the counter table")]
    UnknownCounter(u32),
    #[error("task {task} waits on counter {counter} that it also increments — it can never fire (deadlock)")]
    SelfDependentCounter { task: TaskId, counter: u32 },
    #[error("data dependency task {parent} → task {child} is enforced by neither a counter nor resource order (child can start early)")]
    UnguardedEdge { parent: TaskId, child: TaskId },
    #[error("replay: task {child} fires before its parent task {parent} completes")]
    PrematureFire { parent: TaskId, child: TaskId },
    #[error("replay: deadlock — {0} instructions never become ready")]
    Deadlock(usize),
}

/// The resource a record runs on — `(kind, unit, slot)`. Equal keys ⇒ one
/// sequential queue, so stream order is issue order on that resource.
fn resource_key(ins: &Inst) -> (u8, u8, u16) {
    (ins.resource as u8, ins.unit, ins.index)
}

/// Per-instruction resource predecessor: the previous record on the same
/// resource (which in-order issue forces to complete first), or `None`.
fn resource_preds(prog: &Program) -> Vec<Option<usize>> {
    let mut pred = vec![None; prog.insts.len()];
    let mut last: HashMap<(u8, u8, u16), usize> = HashMap::new();
    for (i, ins) in prog.insts.iter().enumerate() {
        let k = resource_key(ins);
        if let Some(&p) = last.get(&k) {
            pred[i] = Some(p);
        }
        last.insert(k, i);
    }
    pred
}

/// Emit `sched`'s program, round-trip it through the wire format, and verify the
/// decoded stream enforces every dependency in `tasks`. The end-to-end entry.
pub fn verify_schedule(
    g: &TileGraph,
    cons: &ConstraintSet,
    tasks: &TaskGraph,
    sched: &Schedule,
) -> Result<VerifyReport, VerifyError> {
    let prog = emit_program(g, cons, tasks, sched);
    let decoded = Program::decode(&prog.to_bytes()).expect("emitted stream must decode");
    verify(&decoded, &issue_order(sched), tasks)
}

/// Verify that `prog` enforces every data edge in `tasks`, where `order[i]` is
/// the task that `prog.insts[i]` was emitted for.
pub fn verify(
    prog: &Program,
    order: &[TaskId],
    tasks: &TaskGraph,
) -> Result<VerifyReport, VerifyError> {
    if prog.insts.len() != order.len() {
        return Err(VerifyError::LengthMismatch {
            insts: prog.insts.len(),
            order: order.len(),
        });
    }

    let threshold: HashMap<u32, u32> = prog.counters.iter().map(|c| (c.id, c.threshold)).collect();

    // Producers of each counter = the instructions that increment it.
    let mut producers: HashMap<u32, u32> = HashMap::new();
    for ins in &prog.insts {
        for &c in &ins.succ {
            *producers.entry(c).or_insert(0) += 1;
        }
    }

    // Every referenced counter must exist.
    for ins in &prog.insts {
        for &c in ins.wait.iter().chain(&ins.succ) {
            if !threshold.contains_key(&c) {
                return Err(VerifyError::UnknownCounter(c));
            }
        }
    }

    // (1) Satisfiability: threshold == producer count (reachable, and "all").
    for c in &prog.counters {
        let p = producers.get(&c.id).copied().unwrap_or(0);
        if c.threshold > p {
            return Err(VerifyError::Unreachable {
                id: c.id,
                threshold: c.threshold,
                producers: p,
            });
        }
        if c.threshold < p {
            return Err(VerifyError::LooseThreshold {
                id: c.id,
                threshold: c.threshold,
                producers: p,
            });
        }
    }

    let inst_of: HashMap<TaskId, usize> = order.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let keys: Vec<(u8, u8, u16)> = prog.insts.iter().map(resource_key).collect();
    let succ_set: Vec<HashSet<u32>> = prog
        .insts
        .iter()
        .map(|i| i.succ.iter().copied().collect())
        .collect();
    let wait_set: Vec<HashSet<u32>> = prog
        .insts
        .iter()
        .map(|i| i.wait.iter().copied().collect())
        .collect();

    // (1b) No tile may wait on a counter it also increments — its own completion
    // would be a precondition for it to start. (Coarse clustering can produce
    // this when a node both produces into and consumes from one boundary.)
    for (i, &t) in order.iter().enumerate() {
        if let Some(&c) = wait_set[i].intersection(&succ_set[i]).next() {
            return Err(VerifyError::SelfDependentCounter {
                task: t,
                counter: c,
            });
        }
    }

    // (2) Edge gating: each data edge is carried by a counter (cross-resource) or
    // by resource order (same resource, parent issued earlier).
    for &(a, b) in &tasks.edges {
        let (ia, ib) = (inst_of[&a], inst_of[&b]);
        let counter_gated = !succ_set[ia].is_disjoint(&wait_set[ib]);
        let order_gated = keys[ia] == keys[ib] && ia < ib;
        if !counter_gated && !order_gated {
            return Err(VerifyError::UnguardedEdge {
                parent: a,
                child: b,
            });
        }
    }

    // (3) Eager replay: dynamic witness — no premature fire, no deadlock.
    let res_pred = resource_preds(prog);
    let (rounds, max_ready) = replay(prog, order, tasks, &threshold, &res_pred)?;

    Ok(VerifyReport {
        insts: prog.insts.len(),
        counters: prog.counters.len(),
        edges_checked: tasks.edges.len(),
        rounds,
        max_ready,
    })
}

/// Replay the protocol, firing every ready instruction each round (the earliest
/// any tile could possibly start). A tile is ready when its wait counters are
/// met *and* its resource predecessor has completed. Before firing, assert each
/// tile's data parents have already completed; if nothing is ready but tiles
/// remain, the program deadlocks. Returns `(rounds, max_ready_set)`.
fn replay(
    prog: &Program,
    order: &[TaskId],
    tasks: &TaskGraph,
    threshold: &HashMap<u32, u32>,
    res_pred: &[Option<usize>],
) -> Result<(usize, usize), VerifyError> {
    let n = prog.insts.len();
    let inst_of: HashMap<TaskId, usize> = order.iter().enumerate().map(|(i, &t)| (t, i)).collect();

    // Data parents of each instruction, in instruction-index space.
    let mut parents = vec![Vec::new(); n];
    for &(a, b) in &tasks.edges {
        parents[inst_of[&b]].push(inst_of[&a]);
    }

    let mut val: HashMap<u32, u32> = HashMap::new();
    let mut done = vec![false; n];
    let mut remaining = n;
    let (mut rounds, mut max_ready) = (0, 0);

    while remaining > 0 {
        let ready: Vec<usize> = (0..n)
            .filter(|&i| {
                !done[i]
                    && res_pred[i].is_none_or(|p| done[p])
                    && prog.insts[i]
                        .wait
                        .iter()
                        .all(|w| val.get(w).copied().unwrap_or(0) >= threshold[w])
            })
            .collect();
        if ready.is_empty() {
            return Err(VerifyError::Deadlock(remaining));
        }
        max_ready = max_ready.max(ready.len());

        // Worst case: each tile is checked against the snapshot *before* this
        // round fires, so a parent that would only fire in the same round counts
        // as not-yet-done — exactly the premature-start condition.
        for &i in &ready {
            for &p in &parents[i] {
                if !done[p] {
                    return Err(VerifyError::PrematureFire {
                        parent: order[p],
                        child: order[i],
                    });
                }
            }
        }
        // Complete the batch and apply its counter increments.
        for &i in &ready {
            done[i] = true;
            for &c in &prog.insts[i].succ {
                *val.entry(c).or_insert(0) += 1;
            }
        }
        remaining -= ready.len();
        rounds += 1;
    }
    Ok((rounds, max_ready))
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::{Body, Counter, Inst, ResourceKind};

    fn inst(resource: ResourceKind, index: u16, wait: Vec<u32>, succ: Vec<u32>) -> Inst {
        Inst {
            resource,
            unit: 0,
            index,
            body: Body::Host,
            wait,
            succ,
        }
    }

    /// `replay` flags a tile that can fire while a *cross-resource* data parent
    /// is still pending and no counter gates the edge — the dynamic counterpart
    /// of `UnguardedEdge`.
    #[test]
    fn replay_catches_premature_start() {
        // Two ungated instructions on *different* resources (so resource order
        // does not save the edge), with a data edge 0 → 1. Both fire in round 0.
        let prog = Program {
            insts: vec![
                inst(ResourceKind::Host, 0, vec![], vec![]),
                inst(ResourceKind::Sm, 0, vec![], vec![]),
            ],
            counters: vec![],
            ..Default::default()
        };
        let tasks = TaskGraph {
            tasks: vec![],
            edges: vec![(0, 1)],
            ..Default::default()
        };
        let err = replay(&prog, &[0, 1], &tasks, &HashMap::new(), &[None, None]).unwrap_err();
        assert_eq!(
            err,
            VerifyError::PrematureFire {
                parent: 0,
                child: 1
            }
        );
    }

    /// In-order issue: when the two share a resource, the predecessor must
    /// complete first, so the same edge is enforced and replay succeeds.
    #[test]
    fn replay_respects_resource_order() {
        let prog = Program {
            insts: vec![
                inst(ResourceKind::Sm, 0, vec![], vec![]),
                inst(ResourceKind::Sm, 0, vec![], vec![]),
            ],
            counters: vec![],
            ..Default::default()
        };
        let tasks = TaskGraph {
            tasks: vec![],
            edges: vec![(0, 1)],
            ..Default::default()
        };
        let (rounds, _) =
            replay(&prog, &[0, 1], &tasks, &HashMap::new(), &[None, Some(0)]).unwrap();
        assert_eq!(
            rounds, 2,
            "the predecessor must fire in its own round first"
        );
    }

    /// `replay` flags a counter wait cycle across resources: each waits on a
    /// counter the other increments, so neither ever becomes ready.
    #[test]
    fn replay_catches_deadlock() {
        let prog = Program {
            insts: vec![
                inst(ResourceKind::Host, 0, vec![1], vec![0]),
                inst(ResourceKind::Sm, 0, vec![0], vec![1]),
            ],
            counters: vec![
                Counter {
                    id: 0,
                    threshold: 1,
                    scope: 1,
                    _pad: [0; 3],
                },
                Counter {
                    id: 1,
                    threshold: 1,
                    scope: 1,
                    _pad: [0; 3],
                },
            ],
            ..Default::default()
        };
        let tasks = TaskGraph {
            tasks: vec![],
            edges: vec![],
            ..Default::default()
        };
        let threshold = [(0u32, 1u32), (1, 1)].into_iter().collect();
        assert_eq!(
            replay(&prog, &[0, 1], &tasks, &threshold, &[None, None]),
            Err(VerifyError::Deadlock(2))
        );
    }
}
