//! `kv_ring.v1`: the packed prefill launches the runtime planner binds for the production
//! sliding-ring shapes (4de53bf6 H100 bundles) are accepted; a ring too short for the rows a
//! launch writes, an unstaged or unmasked plan on a staged ring, a split slot and a context
//! overrun are rejected.

use plow_asset::kv_ring::{launch, launches, Ring, Shape};
use serde_json::{json, Value};

const RUNGS: &[u32] = &[512, 1024, 2048, 4096, 8192];

fn shape(max_request_rows: Option<u32>, stage_rows: Option<u32>, batch: usize, max_ctx: usize) -> Shape<'static> {
    Shape { rungs: RUNGS, max_request_rows, stage_rows, batch, max_ctx }
}

fn payload(launches: Vec<Value>) -> Value {
    json!({"schema": 1, "launches": launches})
}

#[test]
#[ignore = "requires built plow_verify; CPU-only"]
fn production_ring_launches_hold_and_each_mutation_rejects() {
    let staged = |ctx| Ring { ring_log: 11, window: 1024, capacity: ctx };
    let e4b = Ring { ring_log: 12, window: 512, capacity: 8192 };
    let build = |ring: Ring, s: Shape<'_>| payload(launches(&[ring], &s).unwrap());
    let mut cases = vec![
        ("12b", build(staged(16384), shape(Some(4096), Some(1024), 128, 16384)), true),
        ("26b", build(staged(131072), shape(Some(4096), Some(1024), 128, 131072)), true),
        ("31b", build(staged(16384), shape(Some(4096), Some(1024), 8, 16384)), true),
        ("e4b", build(e4b, shape(Some(2048), None, 128, 8192)), true),
        ("12b unstaged", build(staged(16384), shape(Some(4096), None, 128, 16384)), false),
        ("12b unmasked", build(staged(16384), shape(None, None, 128, 16384)), false),
        ("e4b half ring", build(Ring { ring_log: 11, ..e4b }, shape(Some(2048), None, 128, 8192)), false),
        ("e4b wider window", build(Ring { window: 2050, ..e4b }, shape(Some(2048), None, 128, 8192)), false),
    ];
    let ring = staged(16384);
    let split = launch(ring, &[0, 0, 1, 0], &[0, 1, 0, 2]).unwrap();
    cases.push(("split slot", payload(vec![split]), false));
    let overrun = launch(Ring { capacity: 100, ..ring }, &[0, 0], &[99, 100]).unwrap();
    cases.push(("context overrun", payload(vec![overrun]), false));
    let ok = launch(ring, &[0, 0, -1, 3], &[99, 100, 0, 7]).unwrap();
    cases.push(("hand launch", payload(vec![ok]), true));

    let requests: Vec<_> = cases.iter().map(|(_, p, _)| ("kv_ring.v1", p.clone())).collect();
    let certs = lean_verify::call_batch(&requests).unwrap();
    for ((what, _, ok), cert) in cases.iter().zip(certs) {
        assert_eq!(cert.ok, *ok, "{what}: {:?}", cert.reason);
    }
}
