//! End-to-end on the CPU backend: write a tiny compiled bundle to disk, load it
//! through the registry, and run a request through the full
//! resolve→tokenize→bucket-select→interpret path.

use std::sync::Arc;

use packet::{Body, Counter, Inst, Program, ResourceKind};
use plowrt::device::cpu::CpuBackend;
use plowrt::device::Backend;
use plowrt::exec::ExecutorSet;
use plowrt::orch::Registry;
use plowrt::serve::AppState;

/// A minimal decode-bucket program: one producer + one gated consumer.
fn tiny_program() -> Program {
    Program {
        insts: vec![
            Inst {
                resource: ResourceKind::Sm,
                unit: 0,
                index: 0,
                body: Body::Host,
                wait: vec![],
                succ: vec![0],
            },
            Inst {
                resource: ResourceKind::Sm,
                unit: 0,
                index: 1,
                body: Body::Host,
                wait: vec![0],
                succ: vec![],
            },
        ],
        counters: vec![Counter {
            id: 0,
            threshold: 1,
            scope: 1,
            _pad: [0; 3],
        }],
        bucket_id: 0,
        plan_gen: 0,
        flags: 0,
    }
}

fn write_bundle(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).unwrap();
    let prog = tiny_program();
    let pkt = prog.to_bytes();
    std::fs::write(dir.join("decode_b1_s8.pkt"), &pkt).unwrap();

    let map = serde_json::json!({
        "arena_bytes": 64,
        "growable_base": 64,
        "segments": [
            {"device": 0, "global_base": 0, "size": 64, "growable_base": 64}
        ],
        "entries": [
            {"slot": 0, "name": "tokens", "class": "request_io",
             "offset": 0, "reserved": 8, "growable": false, "device": 0}
        ]
    });
    std::fs::write(
        dir.join("decode_b1_s8.map.json"),
        serde_json::to_vec_pretty(&map).unwrap(),
    )
    .unwrap();

    let manifest = serde_json::json!({
        "network": "tiny-test-model",
        "gpu": "H100 SXM5",
        "num_gpus": 1,
        "parallel": "tp",
        "weight_shared": true,
        "buckets": [{
            "phase": "decode", "batch": 1, "seq": 8,
            "packet_file": "decode_b1_s8.pkt", "packet_bytes": pkt.len(),
            "instructions": 2, "tile_nodes": 2, "tasks": 2,
            "makespan": 100, "ideal_makespan": 80, "arena_bytes": 64,
            "memory_file": "decode_b1_s8.map.json"
        }]
    });
    std::fs::write(
        dir.join("weights.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn load_and_generate() {
    let dir =
        std::env::temp_dir().join(format!("plowrt_bundle_{}_{}", std::process::id(), "t1"));
    let _ = std::fs::remove_dir_all(&dir);
    write_bundle(&dir);

    let backend: Arc<dyn Backend> = Arc::new(CpuBackend::new(4));
    let execset = Arc::new(ExecutorSet::bringup(backend).unwrap());

    let mut registry = Registry::new();
    registry.load(&dir, None).unwrap();
    assert_eq!(registry.len(), 1);
    assert!(registry.slugs().any(|s| s == "tiny-test-model"));

    let state = AppState::new(registry, execset);
    let gen = plowrt::serve::GenParams {
        max_tokens: 4,
        ..Default::default()
    };
    let (text, executed) = state.generate("tiny-test-model", "hi there", &gen).unwrap();
    // 2-instruction bucket run once per generated token.
    assert!(executed > 0, "the schedule ran");
    assert!(!text.is_empty(), "produced detokenized output");

    // Unknown model → error, not panic.
    assert!(state.generate("nope", "x", &gen).is_err());

    let _ = std::fs::remove_dir_all(&dir);
}
