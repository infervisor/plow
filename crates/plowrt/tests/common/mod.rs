//! Shared test fixtures.
// Different integration-test binaries use different subsets of these helpers.
#![allow(dead_code)]

use packet::{Body, Counter, Inst, Opcode, Program, ResourceKind};

/// A minimal decode-bucket program: one producer + one counter-gated consumer.
pub fn tiny_program() -> Program {
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

/// A decode-bucket program that carries a real `TOKEN_SAMPLE_BATCH` packet
/// with batch width `b` and `vocab = 256` (matches the byte tokenizer). The
/// program is a single Host op that produces counter 0 → a Token packet gated
/// on it: the mux fills the B×vocab logits tile before firing.
pub fn sample_batch_program(b: u32) -> Program {
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
                resource: ResourceKind::Host,
                unit: 0,
                index: 1,
                body: Body::Token {
                    in_slot: 0,
                    out_slot: 1,
                    kind: Opcode::TOKEN_SAMPLE_BATCH,
                    vocab: 256,
                    arg: b,
                },
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

/// Write a bundle whose largest decode rung carries a `SAMPLE_BATCH` packet.
/// Smaller rungs still use `tiny_program`; the mux picks the covering rung
/// based on live-slot count, so the batched path only triggers when there's
/// really a batch to fire.
pub fn write_bundle_with_sample_batch(dir: &std::path::Path, slug: &str, max_batch: u32) {
    std::fs::create_dir_all(dir).unwrap();
    let tiny = tiny_program().to_bytes();
    let batched = sample_batch_program(max_batch).to_bytes();
    std::fs::write(dir.join("decode_tiny.pkt"), &tiny).unwrap();
    std::fs::write(dir.join("decode_batched.pkt"), &batched).unwrap();

    let map = serde_json::json!({
        "arena_bytes": 64,
        "growable_base": 64,
        "segments": [{"device": 0, "global_base": 0, "size": 64, "growable_base": 64}],
        "entries": [{"slot": 0, "name": "tokens", "class": "request_io",
                     "offset": 0, "reserved": 8, "growable": false, "device": 0}]
    });
    std::fs::write(
        dir.join("decode_tiny.map.json"),
        serde_json::to_vec_pretty(&map).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("decode_batched.map.json"),
        serde_json::to_vec_pretty(&map).unwrap(),
    )
    .unwrap();

    let mut buckets = vec![];
    // Small rungs use tiny (no SAMPLE_BATCH — fallback path).
    for b in 1..(max_batch as i64) {
        buckets.push(serde_json::json!({
            "phase": "decode", "batch": b, "seq": 8,
            "packet_file": "decode_tiny.pkt", "packet_bytes": tiny.len(),
            "instructions": 2, "tile_nodes": 2, "tasks": 2,
            "makespan": 100, "ideal_makespan": 80, "arena_bytes": 64,
            "memory_file": "decode_tiny.map.json"
        }));
    }
    // Largest rung carries SAMPLE_BATCH.
    buckets.push(serde_json::json!({
        "phase": "decode", "batch": max_batch as i64, "seq": 8,
        "packet_file": "decode_batched.pkt", "packet_bytes": batched.len(),
        "instructions": 2, "tile_nodes": 2, "tasks": 2,
        "makespan": 100, "ideal_makespan": 80, "arena_bytes": 64,
        "memory_file": "decode_batched.map.json"
    }));

    let manifest = serde_json::json!({
        "network": slug,
        "gpu": "H100 SXM5",
        "num_gpus": 1,
        "parallel": "tp",
        "weight_shared": true,
        "buckets": buckets,
    });
    std::fs::write(
        dir.join("weights.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

/// Same as `write_bundle_with_sample_batch` but attaches `KvPaging` to the
/// bucket map so the mux's `KvArena` gets built. `initial_blocks_per_layer`
/// caps the total live-slot KV footprint (the KV OOM test uses a tiny value).
///
/// Matches what plowc emits: per-layer `kv_cache_L{i}` `MemEntry` records
/// with distinct offsets past `growable_base`, and a segment sized to hold
/// them. `AddressSpace::kv_layer_bases` then resolves real per-layer bases.
pub fn write_bundle_with_kv(
    dir: &std::path::Path,
    slug: &str,
    max_batch: u32,
    initial_blocks_per_layer: i64,
) {
    write_bundle_with_sample_batch(dir, slug, max_batch);

    // Per-layer sizing: 2 layers × block_bytes(64) × initial_blocks each.
    let block_bytes: u64 = 64;
    let n_layers: u64 = 2;
    let per_layer_reserved = block_bytes * initial_blocks_per_layer.max(0) as u64;
    // Segment covers the request_io region + every layer's growable band.
    let growable_base: u64 = 64;
    let total_size = growable_base + per_layer_reserved * n_layers;

    let mut entries = vec![serde_json::json!({
        "slot": 0, "name": "tokens", "class": "request_io",
        "offset": 0, "reserved": 8, "growable": false, "device": 0
    })];
    for i in 0..n_layers {
        entries.push(serde_json::json!({
            "slot": 1 + i,
            "name": format!("kv_cache_L{i}"),
            "class": "growable",
            "offset": growable_base + i * per_layer_reserved,
            "reserved": per_layer_reserved,
            "growable": true,
            "device": 0
        }));
    }

    // Rewrite the two map files to embed KvPaging + KV MemEntries.
    let map = serde_json::json!({
        "arena_bytes": total_size,
        "growable_base": growable_base,
        "segments": [{
            "device": 0, "global_base": 0, "size": total_size,
            "growable_base": growable_base
        }],
        "entries": entries,
        "kv_paging": {
            "block_tokens": 4,
            "block_bytes": block_bytes,
            "kv_heads": 2,
            "head_dim": 8,
            "per_layer": [
                {"layer_idx": 0, "buffer_name": "kv_cache_L0",
                 "initial_blocks": initial_blocks_per_layer},
                {"layer_idx": 1, "buffer_name": "kv_cache_L1",
                 "initial_blocks": initial_blocks_per_layer}
            ]
        }
    });
    let bytes = serde_json::to_vec_pretty(&map).unwrap();
    std::fs::write(dir.join("decode_tiny.map.json"), &bytes).unwrap();
    std::fs::write(dir.join("decode_batched.map.json"), &bytes).unwrap();
}

/// Write a loadable multi-rung compiled model under `dir` with API slug `slug`.
/// Every rung in `batches` points to the same tiny program (same `.pkt`); the
/// distinct bucket keys give the mux a real ladder to round up. `seq = 8` for
/// each rung. Slot capacity in the mux == `batches.iter().max()`.
pub fn write_bundle_with_batches(dir: &std::path::Path, slug: &str, batches: &[i64]) {
    std::fs::create_dir_all(dir).unwrap();
    let pkt = tiny_program().to_bytes();
    // One shared packet + memory-map file, referenced by every rung.
    let pkt_name = "decode_shared.pkt";
    let map_name = "decode_shared.map.json";
    std::fs::write(dir.join(pkt_name), &pkt).unwrap();

    let map = serde_json::json!({
        "arena_bytes": 64,
        "growable_base": 64,
        "segments": [{"device": 0, "global_base": 0, "size": 64, "growable_base": 64}],
        "entries": [{"slot": 0, "name": "tokens", "class": "request_io",
                     "offset": 0, "reserved": 8, "growable": false, "device": 0}]
    });
    std::fs::write(dir.join(map_name), serde_json::to_vec_pretty(&map).unwrap()).unwrap();

    let buckets: Vec<serde_json::Value> = batches
        .iter()
        .map(|b| {
            serde_json::json!({
                "phase": "decode", "batch": b, "seq": 8,
                "packet_file": pkt_name, "packet_bytes": pkt.len(),
                "instructions": 2, "tile_nodes": 2, "tasks": 2,
                "makespan": 100, "ideal_makespan": 80, "arena_bytes": 64,
                "memory_file": map_name
            })
        })
        .collect();

    let manifest = serde_json::json!({
        "network": slug,
        "gpu": "H100 SXM5",
        "num_gpus": 1,
        "parallel": "tp",
        "weight_shared": true,
        "buckets": buckets,
    });
    std::fs::write(
        dir.join("weights.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

/// Write a loadable one-bucket compiled model under `dir` with API slug `slug`.
pub fn write_bundle(dir: &std::path::Path, slug: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let pkt = tiny_program().to_bytes();
    std::fs::write(dir.join("decode_b1_s8.pkt"), &pkt).unwrap();

    let map = serde_json::json!({
        "arena_bytes": 64,
        "growable_base": 64,
        "segments": [{"device": 0, "global_base": 0, "size": 64, "growable_base": 64}],
        "entries": [{"slot": 0, "name": "tokens", "class": "request_io",
                     "offset": 0, "reserved": 8, "growable": false, "device": 0}]
    });
    std::fs::write(
        dir.join("decode_b1_s8.map.json"),
        serde_json::to_vec_pretty(&map).unwrap(),
    )
    .unwrap();

    let manifest = serde_json::json!({
        "network": slug,
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
