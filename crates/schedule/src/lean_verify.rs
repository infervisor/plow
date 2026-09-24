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

use std::collections::{BTreeMap, BTreeSet, HashMap};

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

fn validate_sources(
    tasks: &TaskGraph,
    sched: &Schedule,
    map: &AddressMap,
    task_sets: &crate::memory::TensorTaskSets,
    additional_effects: &crate::memory::TensorTaskSets,
) -> Result<(), String> {
    let n = tasks.tasks.len();
    if sched.starts.len() != n
        || sched.placement.len() != n
        || tasks.edges.iter().any(|&(a, b)| a >= n || b >= n || a == b)
    {
        return Err("incomplete schedule placement/start/edge domain".into());
    }
    let mut seen = BTreeSet::new();
    for (resource, stream) in &sched.streams {
        let packets = sched
            .packets
            .get(resource)
            .ok_or("resource has no packet stream")?;
        if packets.len() != stream.len() {
            return Err("packet/stream task count differs".into());
        }
        for ((task, start), packet) in stream.iter().zip(packets) {
            if *task >= n
                || !seen.insert(*task)
                || sched.placement.get(task) != Some(resource)
                || sched.starts[*task] != *start
                || packet.task != *task
                || packet.start != *start
                || packet.op != tasks.tasks[*task].op
                || packet.kind != crate::passes::packet_kind(*resource, tasks.tasks[*task].kind)
            {
                return Err("task placement, start or packet/stream order differs".into());
            }
        }
    }
    if seen.len() != n || sched.packets.keys().any(|r| !sched.streams.contains_key(r)) {
        return Err("missing or extra scheduled task stream".into());
    }
    let mut counters = BTreeSet::new();
    if sched
        .counters
        .iter()
        .any(|c| !counters.insert(c.id) || c.threshold == 0)
        || sched.packets.values().flatten().any(|p| {
            p.wait
                .iter()
                .chain(&p.successors)
                .any(|id| !counters.contains(id))
        })
    {
        return Err("missing, duplicated or zero-threshold counter".into());
    }
    let mut expected = crate::memory::TensorTaskSets::new();
    let mut bytes = HashMap::<&str, u64>::new();
    for (id, task) in tasks.tasks.iter().enumerate() {
        let Some(name) = &task.tensor else { continue };
        let (writers, readers) = expected.entry(name.clone()).or_default();
        let size = bytes.entry(name).or_default();
        *size = (*size).max(task.bytes.max(task.tensor_bytes));
        match task.kind {
            crate::expand::TaskKind::Compute | crate::expand::TaskKind::DmaOut => writers.push(id),
            crate::expand::TaskKind::DmaIn => readers.push(id),
            crate::expand::TaskKind::Host => {}
        }
    }
    for (name, (writers, readers)) in additional_effects {
        if expected.contains_key(name)
            || (writers.is_empty() && readers.is_empty())
            || writers.iter().chain(readers).any(|&id| id >= n)
            || writers.iter().collect::<BTreeSet<_>>().len() != writers.len()
            || readers.iter().collect::<BTreeSet<_>>().len() != readers.len()
        {
            return Err("invalid supplemental task effects".into());
        }
        expected.insert(name.clone(), (writers.clone(), readers.clone()));
        // Footprint size is the supplemental producer's obligation; no task.tensor
        // byte extent exists for implicit state such as a flash kernel's KV cache.
        bytes.insert(name, 0);
    }
    if task_sets != &expected {
        return Err("tensor task sets differ from expanded task effects".into());
    }
    let mut entries = BTreeSet::new();
    let mut names = BTreeSet::new();
    for entry in &map.entries {
        names.insert(&entry.name);
        if !entries.insert((&entry.name, entry.device))
            || !bytes
                .get(entry.name.as_str())
                .is_some_and(|size| *size <= entry.reserved)
            || entry
                .offset
                .checked_add(entry.reserved)
                .is_none_or(|end| end > map.arena_bytes)
        {
            return Err("invalid, undersized or unaccounted address-map entry".into());
        }
    }
    if expected.keys().any(|name| !names.contains(name)) {
        return Err("expanded task tensor is missing from the address map".into());
    }
    Ok(())
}

/// Assemble the full request bundle. `address_map` should be the map returned
/// by [`crate::memory::plan_from_schedule_with_task_sets`]; the task-set map
/// keys must match entry names.
pub fn build_schedule_request(
    tasks: &TaskGraph,
    sched: &Schedule,
    address_map: &AddressMap,
    task_sets: &crate::memory::TensorTaskSets,
) -> Result<ScheduleRequest, String> {
    build_schedule_request_with_effects(tasks, sched, address_map, task_sets, &HashMap::new())
}

/// Supplemental effects must come from an audited operation producer, not from
/// missing entries in `task_sets`. Their kernel footprints remain external.
pub fn build_schedule_request_with_effects(
    tasks: &TaskGraph,
    sched: &Schedule,
    address_map: &AddressMap,
    task_sets: &crate::memory::TensorTaskSets,
    additional_effects: &crate::memory::TensorTaskSets,
) -> Result<ScheduleRequest, String> {
    validate_sources(tasks, sched, address_map, task_sets, additional_effects)?;
    let n = tasks.tasks.len();
    let resource_ids = intern_resources(sched);

    let resource: Vec<u64> = (0..n)
        .map(|t| {
            sched
                .placement
                .get(&t)
                .and_then(|r| resource_ids.get(r).copied())
                .expect("validated task placement")
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
    let schedule_order: Vec<u64> = (0..n).map(|t| sched.starts[t] as u64).collect();

    let entries: Vec<LeanAddrEntry> = address_map
        .entries
        .iter()
        .map(|e| {
            let (writers, readers) = task_sets
                .get(&e.name)
                .cloned()
                .expect("validated tensor task sets");
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

    Ok(ScheduleRequest {
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
    })
}

/// One-stop helper: build the request from the raw pipeline outputs. This is
/// what plowc calls per bucket.
pub fn request_for_bucket(
    tasks: &TaskGraph,
    sched: &Schedule,
    cons: &rewrite::ConstraintSet,
) -> Result<ScheduleRequest, String> {
    let (map, task_sets) = plan_from_schedule_with_task_sets(tasks, sched, cons);
    build_schedule_request(tasks, sched, &map, &task_sets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expand::{Task, TaskKind};
    use crate::memory::{allocate, BufReq, TensorTaskSets};
    use crate::passes::{Counter, PacketKind, Scope};

    fn fixture() -> (TaskGraph, Schedule, AddressMap, TensorTaskSets) {
        let task = |name: &str, kind| Task {
            node: 0,
            op: name.into(),
            unit: 0,
            kind,
            coord: vec![],
            dur: 1,
            bytes: 64,
            tensor_bytes: 64,
            sram_pages: 0,
            out_pages: 0,
            tmem_cols: 0,
            tensor: Some(name.into()),
            cross_unit: false,
        };
        let tasks = TaskGraph {
            tasks: vec![
                task("input", TaskKind::DmaIn),
                task("output", TaskKind::Compute),
            ],
            edges: vec![(0, 1)],
            ..Default::default()
        };
        let dma = ResourceId::Dma(0, 0);
        let sm = ResourceId::Sm(0, 0);
        let sched = Schedule {
            streams: HashMap::from([(dma, vec![(0, 0)]), (sm, vec![(1, 1)])]),
            packets: HashMap::from([
                (
                    dma,
                    vec![Packet {
                        task: 0,
                        op: "input".into(),
                        kind: PacketKind::TmaIn,
                        start: 0,
                        wait: vec![],
                        successors: vec![0],
                    }],
                ),
                (
                    sm,
                    vec![Packet {
                        task: 1,
                        op: "output".into(),
                        kind: PacketKind::Compute,
                        start: 1,
                        wait: vec![0],
                        successors: vec![],
                    }],
                ),
            ]),
            counters: vec![Counter {
                id: 0,
                threshold: 1,
                scope: Scope::IntraGpu,
                producer_node: 0,
                consumer_node: 0,
            }],
            placement: HashMap::from([(0, dma), (1, sm)]),
            starts: vec![0, 1],
            sram_slots: HashMap::new(),
            spills: 0,
            tmem_slots: HashMap::new(),
            tmem_spills: 0,
            makespan: 2,
        };
        let map = allocate(&[
            BufReq::new("input", 64, BufClass::Persistent),
            BufReq::new("output", 64, BufClass::Scratch).with_live(1, 2),
        ]);
        let sets = HashMap::from([
            ("input".into(), (vec![], vec![0])),
            ("output".into(), (vec![1], vec![])),
        ]);
        (tasks, sched, map, sets)
    }

    #[test]
    fn source_completeness_cannot_be_replaced_with_empty_effects_or_default_placement() {
        let (tasks, sched, map, sets) = fixture();
        let request = build_schedule_request(&tasks, &sched, &map, &sets).unwrap();
        assert_eq!(request.address_map.len(), 2);
        assert_eq!(request.protocol.waits[1], vec![0]);
        for mutation in 0..15 {
            let (mut tasks, mut sched, mut map, mut sets) = fixture();
            match mutation {
                0 => {
                    sched.placement.remove(&0);
                }
                1 => {
                    sched.starts.pop();
                }
                2 => {
                    sched.packets.remove(&ResourceId::Dma(0, 0));
                }
                3 => {
                    sets.remove("input");
                }
                4 => sets.get_mut("input").unwrap().1.clear(),
                5 => map.entries[0].reserved = 32,
                6 => {
                    map.entries.pop();
                }
                7 => sched.counters.push(sched.counters[0].clone()),
                8 => sched.counters[0].threshold = 0,
                9 => sched.packets.get_mut(&ResourceId::Sm(0, 0)).unwrap()[0].wait = vec![99],
                10 => sched.packets.get_mut(&ResourceId::Sm(0, 0)).unwrap()[0].start = 0,
                11 => tasks.edges.push((0, 2)),
                12 => map.entries[0].offset = u64::MAX,
                13 => {
                    sched.packets.get_mut(&ResourceId::Sm(0, 0)).unwrap()[0].kind =
                        PacketKind::TmaIn
                }
                _ => {
                    sched.packets.get_mut(&ResourceId::Sm(0, 0)).unwrap()[0].op =
                        "wrong operation".into()
                }
            }
            assert!(
                build_schedule_request(&tasks, &sched, &map, &sets).is_err(),
                "mutation {mutation}"
            );
        }
    }

    #[test]
    fn implicit_state_requires_explicit_nonempty_supplemental_effects() {
        let (tasks, sched, mut map, mut sets) = fixture();
        let mut extra = HashMap::from([("state".into(), (vec![1], vec![1]))]);
        sets.extend(extra.clone());
        let mut state = map.entries[0].clone();
        state.name = "state".into();
        state.offset = map.arena_bytes;
        map.arena_bytes += state.reserved;
        map.entries.push(state);
        assert!(build_schedule_request(&tasks, &sched, &map, &sets).is_err());
        assert!(build_schedule_request_with_effects(&tasks, &sched, &map, &sets, &extra).is_ok());
        for mutation in 0..4 {
            let mut bad = extra.clone();
            match mutation {
                0 => bad.get_mut("state").unwrap().0.clear(),
                1 => bad.get_mut("state").unwrap().1 = vec![99],
                2 => {
                    bad.insert("input".into(), (vec![], vec![0]));
                }
                _ => bad.get_mut("state").unwrap().0 = vec![1, 1],
            }
            assert!(
                build_schedule_request_with_effects(&tasks, &sched, &map, &sets, &bad).is_err()
            );
        }
        extra.insert("state".into(), (vec![], vec![]));
        sets.insert("state".into(), (vec![], vec![]));
        assert!(build_schedule_request_with_effects(&tasks, &sched, &map, &sets, &extra).is_err());
    }
}
