//! Integration test for checkpoint B — Tile partition + cost bounds.
//!
//! Backed by `Plow.TilePartition.tile_partition_covers`. The dispatcher
//! rejects candidates whose tile shape doesn't partition the GEMM, or whose
//! tile-work exceeds the caller's cost bound.

#![cfg(feature = "lean-verify")]

use lean_verify::checkpoints::tile_partition::{
    check_tile_partition, GemmShapeJ, TileCandidate, TilePartitionRequest, TileShapeJ,
};

fn simple_candidate() -> TileCandidate {
    // 128×128×64 GEMM, tile 64×64×32 → 2×2×2 = 8 tiles, work = 8·64·64·32 = 1,048,576.
    TileCandidate {
        gemm: GemmShapeJ {
            m: 128,
            n: 128,
            k: 64,
        },
        tile: TileShapeJ {
            bm: 64,
            bn: 64,
            bk: 32,
        },
        cost_bound: 1_048_576,
    }
}

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_divisor_partition_within_bound() {
    let req = TilePartitionRequest {
        candidates: vec![simple_candidate()],
    };
    let cert = check_tile_partition(&req).expect("verifier call");
    assert!(cert.ok, "safe candidate rejected: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_empty_list() {
    let req = TilePartitionRequest { candidates: vec![] };
    let cert = check_tile_partition(&req).expect("verifier call");
    assert!(cert.ok, "empty list rejected: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_zero_bm() {
    let mut c = simple_candidate();
    c.tile.bm = 0;
    let req = TilePartitionRequest {
        candidates: vec![c],
    };
    let cert = check_tile_partition(&req).expect("verifier call");
    assert!(!cert.ok, "bm=0 accepted: {cert:?}");
    let reason = cert.reason.expect("rejection reason");
    assert!(reason.contains("bm"), "unexpected reason: {reason}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_tile_larger_than_gemm_dim() {
    // The extractor deliberately emits over-sized tiles for small ops
    // (e.g. an MoE router with n=8 fits under bn=16 with a single
    // masked tile). ValidPartition allows this — completeness still holds
    // via ceilDiv m bm = 1, and the tile-work bound just wastes some FLOPs.
    let mut c = simple_candidate();
    c.tile.bm = 256; // > m = 128 — one over-sized tile covers the whole M axis.
                     // Bump cost bound to absorb the wasted work: tile-work grows because
                     // the tile occupies more space than needed.
    c.cost_bound = 4_194_304; // 4 × the original bound is more than enough.
    let req = TilePartitionRequest {
        candidates: vec![c],
    };
    let cert = check_tile_partition(&req).expect("verifier call");
    assert!(cert.ok, "over-sized tile rejected: {cert:?}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn rejects_cost_bound_exceeded() {
    let mut c = simple_candidate();
    c.cost_bound = 1_000; // tile-work 1M ≫ 1000
    let req = TilePartitionRequest {
        candidates: vec![c],
    };
    let cert = check_tile_partition(&req).expect("verifier call");
    assert!(!cert.ok, "over-budget accepted: {cert:?}");
    let reason = cert.reason.expect("rejection reason");
    assert!(reason.contains("cost_bound"), "unexpected reason: {reason}");
}

#[test]
#[ignore = "requires plow_verify binary"]
fn accepts_non_divisor_partition_with_slack_bound() {
    // 100×100×64 with 32×32×32 tiles: ⌈100/32⌉² · ⌈64/32⌉ = 4·4·2 = 32 tiles,
    // work = 32 · 32³ = 1_048_576. Give exactly that bound.
    let c = TileCandidate {
        gemm: GemmShapeJ {
            m: 100,
            n: 100,
            k: 64,
        },
        tile: TileShapeJ {
            bm: 32,
            bn: 32,
            bk: 32,
        },
        cost_bound: 1_048_576,
    };
    let req = TilePartitionRequest {
        candidates: vec![c],
    };
    let cert = check_tile_partition(&req).expect("verifier call");
    assert!(cert.ok, "non-divisor partition rejected: {cert:?}");
}
