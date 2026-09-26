use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::checkpoints::schedule::{CompletionPath, ScheduleRequest};
use crate::VerifyError;

/// Witness generation is untrusted: Lean checks every used path independently.
pub(crate) fn payload(request: &ScheduleRequest) -> Result<serde_json::Value, VerifyError> {
    let mut value = serde_json::to_value(request).map_err(VerifyError::SerializeRequest)?;
    let n = request.task_graph.n;
    let protocol = &request.protocol;
    if [
        protocol.waits.len(),
        protocol.succs.len(),
        protocol.resource.len(),
        protocol.stream_idx.len(),
    ]
    .iter()
    .any(|&len| len != n)
    {
        return Ok(value);
    }
    let mut graph = vec![Vec::new(); n];
    let mut producers: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    let mut streams: BTreeMap<u64, BTreeMap<u64, Vec<usize>>> = BTreeMap::new();
    for task in 0..n {
        for &counter in &protocol.succs[task] {
            producers.entry(counter).or_default().push(task);
        }
        streams
            .entry(protocol.resource[task])
            .or_default()
            .entry(protocol.stream_idx[task])
            .or_default()
            .push(task);
    }
    for (target, waits) in protocol.waits.iter().enumerate() {
        for counter in waits {
            for &source in producers.get(counter).into_iter().flatten() {
                graph[source].push(target);
            }
        }
    }
    for stream in streams.values() {
        let groups: Vec<_> = stream.values().collect();
        for pair in groups.windows(2) {
            for &source in pair[0] {
                graph[source].extend(pair[1]);
            }
        }
    }
    for targets in &mut graph {
        targets.sort_unstable();
        targets.dedup();
    }
    let mut needed: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for &(source, target) in &request.task_graph.edges {
        needed.entry(source).or_default().insert(target);
    }
    let mut address_pairs = BTreeSet::new();
    for (i, a) in request.address_map.iter().enumerate() {
        for b in &request.address_map[i + 1..] {
            let overlaps = u128::from(a.offset) < u128::from(b.offset) + u128::from(b.size)
                && u128::from(b.offset) < u128::from(a.offset) + u128::from(a.size);
            if a.name == b.name || !overlaps {
                continue;
            }
            for (readers, writers) in [(&a.readers, &b.writers), (&b.readers, &a.writers)] {
                for &source in readers {
                    for &target in writers {
                        needed.entry(source).or_default().insert(target);
                        address_pairs.insert((source, target));
                    }
                }
            }
        }
    }
    let mut paths = BTreeMap::new();
    for (source, targets) in needed {
        if source >= n {
            continue;
        }
        let mut parent = vec![usize::MAX; n];
        let mut queue = VecDeque::from([source]);
        parent[source] = source;
        while let Some(node) = queue.pop_front() {
            for &next in &graph[node] {
                if parent[next] == usize::MAX {
                    parent[next] = node;
                    queue.push_back(next);
                }
            }
        }
        for target in targets {
            if target >= n || target == source || parent[target] == usize::MAX {
                continue;
            }
            let mut via = Vec::new();
            let mut cursor = parent[target];
            while cursor != source {
                via.push(cursor);
                cursor = parent[cursor];
            }
            via.reverse();
            paths.insert((source, target), via);
        }
    }
    value["dependency_paths"] = serde_json::json!(request
        .task_graph
        .edges
        .iter()
        .map(|edge| paths.get(edge).cloned().unwrap_or_default())
        .collect::<Vec<_>>());
    let address_paths: Vec<_> = address_pairs
        .into_iter()
        .filter_map(|(source, target)| {
            Some(CompletionPath {
                source,
                target,
                via: paths.get(&(source, target))?.clone(),
            })
        })
        .collect();
    value["address_paths"] =
        serde_json::to_value(address_paths).map_err(VerifyError::SerializeRequest)?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoints::schedule::{AddrEntry, ProtocolView, TaskGraphView};

    fn request() -> ScheduleRequest {
        ScheduleRequest {
            task_graph: TaskGraphView {
                n: 3,
                edges: vec![(0, 2)],
            },
            protocol: ProtocolView {
                waits: vec![vec![], vec![0], vec![1]],
                succs: vec![vec![0], vec![1], vec![]],
                resource: vec![0, 1, 2],
                stream_idx: vec![0; 3],
                threshold: BTreeMap::new(),
            },
            schedule_order: vec![0, 1, 2],
            address_map: vec![
                AddrEntry {
                    name: "old".into(),
                    offset: 0,
                    size: 8,
                    cls: "Scratch".into(),
                    readers: vec![0],
                    writers: vec![],
                },
                AddrEntry {
                    name: "new".into(),
                    offset: 0,
                    size: 8,
                    cls: "Scratch".into(),
                    readers: vec![],
                    writers: vec![2],
                },
            ],
        }
    }

    #[test]
    fn builds_transitive_dependency_and_reclamation_witnesses() {
        let mut req = request();
        let value = payload(&req).unwrap();
        assert_eq!(value["dependency_paths"], serde_json::json!([[1]]));
        assert_eq!(
            value["address_paths"],
            serde_json::json!([{"source":0,"target":2,"via":[1]}])
        );
        req.protocol.waits[2].clear();
        assert_eq!(
            payload(&req).unwrap()["address_paths"],
            serde_json::json!([])
        );
    }

    #[test]
    fn stream_ties_do_not_create_false_order_and_invalid_indices_do_not_panic() {
        let mut req = request();
        req.protocol.waits = vec![vec![]; 3];
        req.protocol.resource = vec![0; 3];
        req.protocol.stream_idx = vec![0, 0, 1];
        assert_eq!(
            payload(&req).unwrap()["dependency_paths"],
            serde_json::json!([[]])
        );
        req.task_graph.edges = vec![(0, 1), (3, 0)];
        req.address_map[0].readers.push(3);
        assert!(payload(&req).is_ok());
    }
}
