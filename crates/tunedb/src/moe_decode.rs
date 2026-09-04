//! Model-neutral grouped-MoE decode route records.
//!
//! An adjacent grouped GLU+DOWN pair at decode can run inside the ordinary
//! interpreter packet chain or as two ordered raw launches in their own segment.
//! The standalone pair is a faster body but pays one measured segment handoff
//! per layer, so the choice is a profitability question on exact geometry, not a
//! model predicate. Selection needs a qualified, current record for BOTH routes
//! of the same cell; missing evidence keeps the interpreter route.

use serde::{Deserialize, Serialize};

use crate::{Correctness, Digests, RecordState, Stats};

pub const MOE_DECODE_ORACLE: &str = "moe-decode-pair-bitexact-v1";

/// Measured per-layer cost of leaving and re-entering the ordinary decode
/// interpreter around one standalone segment (ordered AQL dispatch plus
/// all-workgroup convergence), gfx950, nanoseconds. From the exact TP8 network
/// gate: control charges 3.216 ms/token to 92 grouped pairs against the
/// isolated 1.543 ms chain, leaving 0.940-0.964 ms/token of handoff
/// (`perf-data/kimi-k3-mi355x-decode-grouped-moe-20260904.md`).
pub const GFX950_SEGMENT_HANDOFF_NS: f64 = 10_300.0;

/// Do not reroute for less than this fraction of the interpreter pair body.
/// Below it the network gain is inside fold noise and not worth 276 launches.
pub const MIN_GAIN_FRACTION: f64 = 0.10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeDecodeRoute {
    /// Both bodies inside the ordinary decode interpreter packet chain.
    Interpreter,
    /// Two ordered raw launches in one isolated segment.
    Standalone,
}

/// Exact grouped-MoE decode geometry of one rank. `weight_enc` is the expert
/// weight encoding name (`mxfp4`, ...), never a model identifier.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MoeDecodeCell {
    pub hardware: String,
    pub n_cu: u32,
    pub decode_rung: u32,
    pub topk: u32,
    pub hidden: u32,
    pub inter_local: u32,
    pub experts: u32,
    pub weight_enc: String,
}

impl MoeDecodeCell {
    pub fn key(&self) -> String {
        format!(
            "{}|ncu{}|b{}|k{}/h{}/i{}/e{}/{}",
            self.hardware,
            self.n_cu,
            self.decode_rung,
            self.topk,
            self.hidden,
            self.inter_local,
            self.experts,
            self.weight_enc
        )
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MoeDecodeMeasurement {
    pub cell: MoeDecodeCell,
    pub route: MoeDecodeRoute,
    pub digests: Digests,
    /// Per-layer GLU+DOWN pair body, nanoseconds. For the standalone route the
    /// stats cover both ordered launches; the segment handoff is charged at
    /// selection time, not stored here.
    pub stats: Stats,
    pub correctness: Correctness,
    pub state: RecordState,
    pub campaign: String,
}

impl MoeDecodeMeasurement {
    pub fn qualification_blockers(&self) -> Vec<String> {
        crate::blockers_for(&self.correctness, self.stats.samples)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeDecodeSource {
    /// No usable pair of records; the interpreter route is the fixed fallback.
    FixedFallback,
    /// Both routes measured, qualified, and current for this exact cell.
    Qualified,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeDecodeSelection {
    pub route: MoeDecodeRoute,
    pub source: MoeDecodeSource,
    /// Interpreter median minus (standalone median + handoff), nanoseconds per
    /// layer. Positive favours the standalone route. Zero without evidence.
    pub projected_gain_ns: f64,
}

/// Choose the standalone route only when qualified, current records for both
/// routes of this exact cell show the standalone pair plus one handoff beating
/// the interpreter pair by at least `min_gain_fraction`. Anything less keeps the
/// interpreter route. There is deliberately no cross-cell interpolation.
pub fn select_moe_decode_route(
    records: &[MoeDecodeMeasurement],
    cell: &MoeDecodeCell,
    want: &Digests,
    handoff_ns: f64,
    min_gain_fraction: f64,
) -> MoeDecodeSelection {
    let usable = |route: MoeDecodeRoute| {
        records
            .iter()
            .filter(|r| {
                r.cell == *cell
                    && r.route == route
                    && r.state.is_selectable()
                    && matches!(r.correctness, Correctness::Pass)
                    && r.digests.stale_against(want).is_empty()
            })
            .map(|r| r.stats.median_ns)
            .min_by(|a, b| a.total_cmp(b))
    };
    let (Some(interp), Some(standalone)) = (
        usable(MoeDecodeRoute::Interpreter),
        usable(MoeDecodeRoute::Standalone),
    ) else {
        return MoeDecodeSelection {
            route: MoeDecodeRoute::Interpreter,
            source: MoeDecodeSource::FixedFallback,
            projected_gain_ns: 0.0,
        };
    };
    let gain = interp - (standalone + handoff_ns.max(0.0));
    let route = if gain >= interp * min_gain_fraction.max(0.0) {
        MoeDecodeRoute::Standalone
    } else {
        MoeDecodeRoute::Interpreter
    };
    MoeDecodeSelection {
        route,
        source: MoeDecodeSource::Qualified,
        projected_gain_ns: gain,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digests(tag: &str) -> Digests {
        Digests {
            implementation: tag.into(),
            interpreter: tag.into(),
            toolchain: "rocm-7.14.0-nix".into(),
            oracle: MOE_DECODE_ORACLE.into(),
        }
    }

    fn cell() -> MoeDecodeCell {
        MoeDecodeCell {
            hardware: "amd/gfx950/mi350x".into(),
            n_cu: 256,
            decode_rung: 1,
            topk: 16,
            hidden: 3584,
            inter_local: 384,
            experts: 896,
            weight_enc: "mxfp4".into(),
        }
    }

    fn rec(route: MoeDecodeRoute, us: f64, tag: &str) -> MoeDecodeMeasurement {
        MoeDecodeMeasurement {
            cell: cell(),
            route,
            digests: digests(tag),
            stats: Stats::from_samples(vec![us * 1000.0; 5]).unwrap(),
            correctness: Correctness::Pass,
            state: RecordState::Qualified,
            campaign: "test".into(),
        }
    }

    #[test]
    fn standalone_wins_when_body_plus_handoff_clears_the_margin() {
        let records = [
            rec(MoeDecodeRoute::Interpreter, 34.96, "a"),
            rec(MoeDecodeRoute::Standalone, 16.78, "a"),
        ];
        let sel = select_moe_decode_route(
            &records,
            &cell(),
            &digests("a"),
            GFX950_SEGMENT_HANDOFF_NS,
            MIN_GAIN_FRACTION,
        );
        assert_eq!(sel.route, MoeDecodeRoute::Standalone);
        assert_eq!(sel.source, MoeDecodeSource::Qualified);
        assert!((sel.projected_gain_ns - 7_880.0).abs() < 1.0);
    }

    #[test]
    fn handoff_can_erase_an_isolated_win() {
        let records = [
            rec(MoeDecodeRoute::Interpreter, 20.0, "a"),
            rec(MoeDecodeRoute::Standalone, 12.0, "a"),
        ];
        let sel = select_moe_decode_route(&records, &cell(), &digests("a"), 10_300.0, 0.10);
        assert_eq!(sel.route, MoeDecodeRoute::Interpreter);
        assert_eq!(sel.source, MoeDecodeSource::Qualified);
        assert!(sel.projected_gain_ns < 0.0);
    }

    #[test]
    fn missing_or_stale_evidence_keeps_the_interpreter_route() {
        let only_standalone = [rec(MoeDecodeRoute::Standalone, 16.78, "a")];
        let sel = select_moe_decode_route(&only_standalone, &cell(), &digests("a"), 0.0, 0.0);
        assert_eq!(sel.source, MoeDecodeSource::FixedFallback);
        assert_eq!(sel.route, MoeDecodeRoute::Interpreter);

        let stale = [
            rec(MoeDecodeRoute::Interpreter, 34.96, "old"),
            rec(MoeDecodeRoute::Standalone, 16.78, "old"),
        ];
        let sel = select_moe_decode_route(&stale, &cell(), &digests("new"), 0.0, 0.0);
        assert_eq!(sel.source, MoeDecodeSource::FixedFallback);

        let other_cell = MoeDecodeCell {
            inter_local: 768,
            ..cell()
        };
        let sel = select_moe_decode_route(&stale, &other_cell, &digests("old"), 0.0, 0.0);
        assert_eq!(sel.source, MoeDecodeSource::FixedFallback);
    }

    #[test]
    fn unqualified_or_failed_records_are_ignored() {
        let mut bad = rec(MoeDecodeRoute::Standalone, 1.0, "a");
        bad.state = RecordState::Provisional;
        let mut wrong = rec(MoeDecodeRoute::Standalone, 1.0, "a");
        wrong.correctness = Correctness::Fail {
            detail: "mismatch".into(),
        };
        let records = [rec(MoeDecodeRoute::Interpreter, 34.96, "a"), bad, wrong];
        let sel = select_moe_decode_route(&records, &cell(), &digests("a"), 0.0, 0.0);
        assert_eq!(sel.source, MoeDecodeSource::FixedFallback);
    }

    #[test]
    fn published_jsonl_line_round_trips() {
        // Exactly what scripts/tune_moe_decode_publish.py writes.
        let line = r#"{"cell":{"hardware":"amd/gfx950/mi350x","n_cu":256,"decode_rung":1,"topk":16,"hidden":3584,"inter_local":384,"experts":896,"weight_enc":"mxfp4"},"route":"standalone","digests":{"implementation":"gfx950-870078e93f2c92f0","interpreter":"gfx950-870078e93f2c92f0","toolchain":"rocm-7.14.0-nix","oracle":"moe-decode-pair-bitexact-v1"},"stats":{"median_ns":16800.0,"p10_ns":16700.0,"p90_ns":16900.0,"min_ns":16700.0,"samples":5},"correctness":"pass","state":{"state":"qualified"},"campaign":"k3-moe-decode-20260904"}"#;
        let rec: MoeDecodeMeasurement = serde_json::from_str(line).unwrap();
        assert_eq!(rec.route, MoeDecodeRoute::Standalone);
        assert_eq!(rec.cell, cell());
        assert!(rec.state.is_selectable());
        assert!(rec.qualification_blockers().is_empty());
        let back: MoeDecodeMeasurement =
            serde_json::from_str(&serde_json::to_string(&rec).unwrap()).unwrap();
        assert_eq!(back, rec);
    }

    #[test]
    fn cell_key_is_geometry_only() {
        assert_eq!(cell().key(), "amd/gfx950/mi350x|ncu256|b1|k16/h3584/i384/e896/mxfp4");
    }
}
