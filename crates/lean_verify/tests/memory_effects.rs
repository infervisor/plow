use lean_verify::checkpoints::schedule::{
    check_memory_effects, AllocationLease, CompletionPath, MemoryAccess, MemoryEffects,
    ProtocolView, ScheduleRequest, TaskGraphView,
};

fn fixture() -> (ScheduleRequest, Vec<CompletionPath>, MemoryEffects) {
    let n = 12;
    let edges: Vec<_> = (1..n).map(|i| (i - 1, i)).collect();
    let req = ScheduleRequest {
        task_graph: TaskGraphView { n, edges },
        protocol: ProtocolView {
            waits: (0..n)
                .map(|i| if i == 0 { vec![] } else { vec![(i - 1) as u64] })
                .collect(),
            succs: (0..n).map(|i| vec![i as u64]).collect(),
            threshold: (0..n).map(|i| (i.to_string(), 1)).collect(),
            resource: (0..n as u64).collect(),
            stream_idx: vec![0; n],
        },
        schedule_order: (0..n as u64).collect(),
        address_map: vec![],
    };
    let paths = (0..n)
        .flat_map(|source| {
            ((source + 1)..n).map(move |target| CompletionPath {
                source,
                target,
                via: ((source + 1)..target).collect(),
            })
        })
        .collect();
    let leases = vec![
        AllocationLease {
            pool: 0,
            allocation: 0,
            generation: 1,
            owner: 10,
            offset: 0,
            size: 256,
            acquire: 0,
            retire: 5,
            cancel: Some(3),
        },
        AllocationLease {
            pool: 0,
            allocation: 0,
            generation: 2,
            owner: 20,
            offset: 0,
            size: 256,
            acquire: 6,
            retire: 11,
            cancel: None,
        },
    ];
    let accesses = [
        (1, 1, 10, true),
        (2, 1, 10, false),
        (4, 1, 10, false),
        (7, 2, 20, true),
        (8, 2, 20, false),
    ]
    .into_iter()
    .map(|(task, generation, owner, write)| MemoryAccess {
        task,
        pool: 0,
        allocation: 0,
        generation,
        owner,
        offset: 32,
        size: 64,
        write,
    })
    .collect();
    let effects = MemoryEffects {
        schema: 1,
        leases,
        accesses,
        fence_counters: (0..n as u64).collect(),
    };
    (req, paths, effects)
}

#[test]
#[ignore = "requires plow_verify binary"]
fn generations_reuse_only_after_retirement_including_queued_work_after_cancel() {
    let (req, paths, effects) = fixture();
    let cert = check_memory_effects(&req, &paths, &effects).unwrap();
    assert!(cert.ok, "{cert:?}");
    assert!(cert.notes.unwrap().contains("cancellation_retired"));
    for mutation in 0..7 {
        let mut bad = effects.clone();
        match mutation {
            0 => bad.accesses[0].generation = 0,
            1 => bad.accesses[0].owner = 20,
            2 => bad.accesses[0].size = 256,
            3 => bad.leases[0].cancel = Some(11),
            4 => bad.leases[1].acquire = 4,
            5 => bad.leases.push(bad.leases[0].clone()),
            _ => bad.accesses[0].task = 5,
        }
        let cert = check_memory_effects(&req, &paths, &bad).unwrap();
        assert!(!cert.ok, "mutation {mutation}: {cert:?}");
    }
}

#[test]
#[ignore = "requires plow_verify binary"]
fn issue_order_partial_thresholds_and_missing_fences_cannot_certify_effects() {
    let (req, paths, effects) = fixture();
    for mutation in 0..5 {
        let mut req = req.clone();
        let mut effects = effects.clone();
        match mutation {
            0 => {
                req.protocol.resource.fill(0);
                req.protocol.stream_idx = req.schedule_order.clone();
            }
            1 => {
                req.protocol.threshold.insert("0".into(), 0);
            }
            2 => effects.fence_counters.clear(),
            3 => effects.accesses.clear(),
            _ => req.protocol.succs[0].push(0),
        }
        assert!(
            !check_memory_effects(&req, &paths, &effects).unwrap().ok,
            "mutation {mutation}"
        );
    }
}

#[test]
#[ignore = "requires plow_verify binary"]
fn raw_war_waw_require_completion_order_but_read_read_does_not() {
    let (mut req, _, mut effects) = fixture();
    // Tasks 1 and 2 are concurrent, but both retire after task 3.
    req.task_graph.edges.retain(|edge| *edge != (1, 2));
    req.task_graph.edges.extend([(0, 2), (1, 3)]);
    req.protocol.waits[2] = vec![0];
    req.protocol.waits[3] = vec![1, 2];
    let mut paths = Vec::new();
    for source in 0..req.task_graph.n {
        let mut queue = std::collections::VecDeque::from([(source, Vec::new())]);
        let mut visited = std::collections::BTreeSet::from([source]);
        while let Some((current, via)) = queue.pop_front() {
            for &(from, target) in &req.task_graph.edges {
                if from != current || !visited.insert(target) {
                    continue;
                }
                paths.push(CompletionPath {
                    source,
                    target,
                    via: via.clone(),
                });
                let mut next = via.clone();
                next.push(target);
                queue.push_back((target, next));
            }
        }
    }
    for (write_a, write_b) in [(true, false), (false, true), (true, true), (false, false)] {
        effects.accesses[0].write = write_a;
        effects.accesses[1].write = write_b;
        let cert = check_memory_effects(&req, &paths, &effects).unwrap();
        assert_eq!(
            cert.ok,
            !write_a && !write_b,
            "{write_a}/{write_b}: {cert:?}"
        );
        if !cert.ok {
            assert!(
                cert.reason.as_deref().unwrap().contains("RAW/WAR/WAW"),
                "{cert:?}"
            );
        }
    }
}
