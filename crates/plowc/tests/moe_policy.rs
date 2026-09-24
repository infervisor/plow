#![cfg(feature = "lean-verify")]

use tunedb::moe_decode::*;
use tunedb::{Correctness, Digests, RecordState, Stats};

#[test]
#[ignore = "requires built plow_verify; CPU-only"]
fn real_moe_policy_producer_checks_handoff_adjusted_selection() {
    let cell = MoeDecodeCell {
        hardware: "amd/gfx950/test".into(),
        n_cu: 256,
        decode_rung: 16,
        topk: 8,
        hidden: 6144,
        inter_local: 512,
        experts: 256,
        weight_enc: "fp8".into(),
    };
    let want = Digests {
        execution: None,
        implementation: "body".into(),
        interpreter: "object".into(),
        toolchain: "compiler".into(),
        oracle: MOE_DECODE_ORACLE.into(),
    };
    let record = |route, ns| MoeDecodeMeasurement {
        cell: cell.clone(),
        route,
        digests: want.clone(),
        stats: Stats::from_samples(vec![ns; 5]).unwrap(),
        correctness: Correctness::Pass,
        state: RecordState::Qualified,
        campaign: "test".into(),
    };
    let mut records = vec![
        record(MoeDecodeRoute::Interpreter, 100.0),
        record(MoeDecodeRoute::Standalone, 40.0),
        record(MoeDecodeRoute::Standalone, 60.0),
    ];
    let mut unqualified = record(MoeDecodeRoute::Standalone, 1.0);
    unqualified.stats.samples = 1;
    records.push(unqualified);
    for (handoff, expected) in [
        (10.0, MoeDecodeRoute::Standalone),
        (80.0, MoeDecodeRoute::Interpreter),
    ] {
        let selected = select_moe_decode_route(&records, &cell, &want, handoff, 0.1);
        assert_eq!(selected.route, expected);
        let witness = policy_witness(&records, &cell, &want, handoff, 0.1, selected).unwrap();
        assert_eq!(witness["candidates"].as_array().unwrap().len(), 3);
        assert!(lean_verify::call("R", witness.clone()).unwrap().ok);
        let opposite = MoeDecodeSelection {
            route: match expected {
                MoeDecodeRoute::Interpreter => MoeDecodeRoute::Standalone,
                MoeDecodeRoute::Standalone => MoeDecodeRoute::Interpreter,
            },
            ..selected
        };
        let bad = policy_witness(&records, &cell, &want, handoff, 0.1, opposite).unwrap();
        assert!(!lean_verify::call("R", bad).unwrap().ok);
    }
}
