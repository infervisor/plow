//! End-to-end: prove the oracle actually changes what the compiler produces.
//!
//! The reviewer's P2 #2: without this, every compiler test could be silently
//! running in analytical mode — the oracle plumbing present but inert, so a
//! regression that made it a no-op would pass unnoticed.
//!
//! These drive `rewrite::tilegraph::assemble_tuned` directly with hand-built
//! oracles, because that is the seam where tile selection happens and it needs
//! no GPU, no toolchain, and no probe. Two claims:
//!
//! 1. a `Buildable` inventory forces a tile the analytical model would not pick;
//! 2. a qualified measurement flips the winner between two buildable tiles.

use costmodel::tile::{GemmShape, TileShape};
use costmodel::{Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use rewrite::oracle::{GemmQuery, KernelOracle, TileAdvice};
use rewrite::tilegraph::{assemble_tuned, Compute, OpSpec, TileNode};
use rewrite::{LayerPlan, OpKind};

/// One dense bf16 GEMM, the minimal plan that exercises tile selection.
fn one_gemm_plan(m: i64, n: i64, k: i64) -> LayerPlan {
    LayerPlan {
        ops: vec![OpSpec::bf16(
            "gemm".into(),
            vec!["x".into(), "w".into()],
            "y".into(),
            OpKind::Gemm(GemmShape { m, n, k }),
        )],
    }
}

fn soc() -> Soc<'static> {
    let spec = costmodel::hwspec::registry::lookup("H100 NVL").expect("H100 NVL");
    Soc::single(spec, DEFAULT_PAGE_BYTES)
}

/// The tile chosen for the single GEMM in a plan.
fn chosen_tile(oracle: &dyn KernelOracle) -> TileShape {
    let soc = soc();
    let plan = one_gemm_plan(4096, 4096, 4096);
    let (g, _cons) =
        assemble_tuned(&soc, &plan, SramPolicy::Stream, None, oracle).expect("assemble");
    for node in &g.nodes {
        if let TileNode::Compute {
            kind: Compute::Gemm(t),
            ..
        } = node
        {
            return *t;
        }
    }
    panic!("no GEMM compute node in the assembled graph");
}

/// Baseline: the analytical model's own choice, with no oracle.
fn analytical_tile() -> TileShape {
    chosen_tile(&rewrite::oracle::NoOracle)
}

/// An oracle that offers exactly one buildable tile — a stand-in for a probed
/// interpreter whose `d_gemm` is macro-fixed.
struct FixedTile(TileShape);
impl KernelOracle for FixedTile {
    fn gemm_tiles(&self, _q: &GemmQuery) -> TileAdvice {
        TileAdvice::Buildable(vec![self.0])
    }
}

/// An oracle offering two buildable tiles, with a measurement making the
/// larger one win — even though its analytical cost is worse.
struct MeasuredTwo {
    a: TileShape,
    b: TileShape,
    /// ns for (a, b); lower wins.
    ns: (u64, u64),
}
impl KernelOracle for MeasuredTwo {
    fn gemm_tiles(&self, _q: &GemmQuery) -> TileAdvice {
        TileAdvice::Buildable(vec![self.a, self.b])
    }
    fn measured_gemm(&self, _q: &GemmQuery, tiles: &[TileShape]) -> Option<Vec<u64>> {
        Some(
            tiles
                .iter()
                .map(|t| if *t == self.a { self.ns.0 } else { self.ns.1 })
                .collect(),
        )
    }
}

/// A buildable inventory forces its tile, even when the analytical model would
/// have chosen a different one. If this ever equals the analytical tile by
/// accident, pick a tile the model would not.
#[test]
fn a_buildable_inventory_changes_the_selected_tile() {
    let analytical = analytical_tile();

    // A deliberately odd but legal tile the analytical model does not pick.
    let forced = TileShape {
        bm: 16,
        bn: 64,
        bk: 32,
        split_k: 1,
    };
    assert_ne!(
        analytical, forced,
        "precondition: the forced tile is not the analytical one"
    );

    let got = chosen_tile(&FixedTile(forced));
    assert_eq!(
        got, forced,
        "the compiler must emit the tile the inventory carries"
    );
    assert_ne!(
        got, analytical,
        "and it must differ from the analytical choice"
    );
}

/// A qualified measurement flips the winner between two buildable tiles. The
/// measurement makes `b` faster; without it, analytical cost would prefer `a`.
#[test]
fn a_measurement_changes_the_winner() {
    let a = TileShape {
        bm: 64,
        bn: 64,
        bk: 32,
        split_k: 1,
    };
    let b = TileShape {
        bm: 128,
        bn: 128,
        bk: 32,
        split_k: 1,
    };

    // b wins on measurement (lower ns), regardless of analytical order.
    let measured = MeasuredTwo {
        a,
        b,
        ns: (900, 100),
    };
    assert_eq!(chosen_tile(&measured), b, "the measured-fast tile must win");

    // Flip the measurement; the winner flips too. This proves the measurement
    // is what decided it, not some fixed tie-break.
    let flipped = MeasuredTwo {
        a,
        b,
        ns: (100, 900),
    };
    assert_eq!(
        chosen_tile(&flipped),
        a,
        "flipping the measurement flips the winner"
    );
}

/// The plumbing is not inert: NoOracle and a Buildable oracle genuinely produce
/// different graphs for the same plan. Guards against a refactor that drops the
/// oracle on the floor.
#[test]
fn the_oracle_is_not_ignored() {
    let forced = TileShape {
        bm: 16,
        bn: 64,
        bk: 32,
        split_k: 1,
    };
    assert_ne!(
        chosen_tile(&rewrite::oracle::NoOracle),
        chosen_tile(&FixedTile(forced))
    );
}
