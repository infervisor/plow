//! Bridge from the concrete `(TaskGraph, Schedule, ConstraintSet, AddressMap)`
//! tuple that `plowc` produces into the JSON payload the Lean verifier
//! (`plow_verify` CLI) consumes.
//!
//! Kept in `crates/schedule` so it lives close to the source types; no direct
//! Lean dependency — just serialization + a `lean_verify` crate call.
//!
//! Design notes (§5.10 D+F):
//!
//! * `resource` — a stable `u64` per `ResourceId` (interned in the order
//!   resources first appear in the schedule's placement map). Only identity
//!   matters to the verifier, not the numeric value.
//! * `stream_idx` — position within the resource's stream (0-based).
//! * `schedule_order` — the cycle each task starts. This is the topological
//!   witness the Lean side needs to close `happensBefore_acyclic`: cycles are
//!   monotone in happens-before by construction of `list_schedule`.

use std::collections::{BTreeMap, HashMap};

use lean_verify::checkpoints::schedule::{
    AddrEntry as LeanAddrEntry, ProtocolView, ScheduleRequest, TaskGraphView,
};

use crate::expand::{TaskGraph, TaskId};
use crate::memory::{plan_from_schedule_with_task_sets, AddressMap, BufClass};
use crate::passes::{Packet, Schedule};
use crate::resource::ResourceId;

/// Assign each `ResourceId` a stable `u64` — deterministic within a call, but
/// the values themselves are opaque to the verifier (it only checks equality).
fn intern_resources(sched: &Schedule) -> HashMap<ResourceId, u64> {
    let mut idx = HashMap::new();
    // Iterate in placement order (task id ascending) so successive calls on
    // the same schedule produce identical resource ids.
    let mut placements: Vec<(TaskId, ResourceId)> =
        sched.placement.iter().map(|(&t, &r)| (t, r)).collect();
    placements.sort_by_key(|(t, _)| *t);
    for (_, r) in placements {
        let next = idx.len() as u64;
        idx.entry(r).or_insert(next);
    }
    idx
}

/// Build the per-task wait/succ counter lists by scanning `Schedule.packets`
/// across every resource. Returns `(waits[t], succs[t])` for `t ∈ [0, n)`.
fn scan_wait_succ(sched: &Schedule, n: usize) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
    let mut waits = vec![Vec::<u64>::new(); n];
    let mut succs = vec![Vec::<u64>::new(); n];
    for stream in sched.packets.values() {
        for pkt in stream {
            let Packet {
                task,
                wait,
                successors,
                ..
            } = pkt;
            if *task < n {
                waits[*task].extend(wait.iter().map(|&c| c as u64));
                succs[*task].extend(successors.iter().map(|&c| c as u64));
            }
        }
    }
    (waits, succs)
}

/// Build the per-task stream index by walking each stream in order.
fn stream_indices(sched: &Schedule, n: usize) -> Vec<u64> {
    let mut out = vec![0u64; n];
    for stream in sched.streams.values() {
        for (i, (task, _cycle)) in stream.iter().enumerate() {
            if *task < n {
                out[*task] = i as u64;
            }
        }
    }
    out
}

fn class_str(cls: BufClass) -> String {
    match cls {
        BufClass::Persistent => "Persistent",
        // The Lean verifier treats Static identically to Persistent: same
        // lifetime (whole-program, no writers), same reader-set semantics.
        BufClass::Static => "Persistent",
        BufClass::RequestIo => "RequestIo",
        BufClass::Scratch => "Scratch",
        BufClass::Growable => "Growable",
    }
    .into()
}

/// Assemble the full request bundle. `address_map` should be the map returned
/// by [`crate::memory::plan_from_schedule_with_task_sets`]; the task-set map
/// keys must match entry names.
pub fn build_schedule_request(
    tasks: &TaskGraph,
    sched: &Schedule,
    address_map: &AddressMap,
    task_sets: &crate::memory::TensorTaskSets,
) -> ScheduleRequest {
    let n = tasks.tasks.len();
    let resource_ids = intern_resources(sched);

    let resource: Vec<u64> = (0..n)
        .map(|t| {
            sched
                .placement
                .get(&t)
                .and_then(|r| resource_ids.get(r).copied())
                .unwrap_or(u64::MAX) // unplaced tasks land in a lone bucket
        })
        .collect();

    let stream_idx = stream_indices(sched, n);
    let (waits, succs) = scan_wait_succ(sched, n);

    let mut threshold = BTreeMap::new();
    for c in &sched.counters {
        threshold.insert(c.id.to_string(), c.threshold as u64);
    }

    let edges: Vec<(usize, usize)> = tasks.edges.iter().map(|&(a, b)| (a, b)).collect();

    // `starts` is a Vec<Cycle> indexed by task id.
    let schedule_order: Vec<u64> = (0..n)
        .map(|t| sched.starts.get(t).copied().unwrap_or(0) as u64)
        .collect();

    let entries: Vec<LeanAddrEntry> = address_map
        .entries
        .iter()
        .map(|e| {
            let (writers, readers) = task_sets
                .get(&e.name)
                .cloned()
                .unwrap_or_else(|| (Vec::new(), Vec::new()));
            LeanAddrEntry {
                name: e.name.clone(),
                offset: e.offset,
                size: e.reserved,
                cls: class_str(e.class),
                writers,
                readers,
            }
        })
        .collect();

    ScheduleRequest {
        task_graph: TaskGraphView { n, edges },
        protocol: ProtocolView {
            waits,
            succs,
            threshold,
            resource,
            stream_idx,
        },
        schedule_order,
        address_map: entries,
    }
}

/// One-stop helper: build the request from the raw pipeline outputs. This is
/// what plowc calls per bucket.
pub fn request_for_bucket(
    tasks: &TaskGraph,
    sched: &Schedule,
    cons: &rewrite::ConstraintSet,
) -> ScheduleRequest {
    let (map, task_sets) = plan_from_schedule_with_task_sets(tasks, sched, cons);
    build_schedule_request(tasks, sched, &map, &task_sets)
}
