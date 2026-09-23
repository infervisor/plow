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
/// (`perf-data/kimi-k3-mi355x-campaign-summary-20260904.md`, grouped-MoE standalone route).
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
        let mut blockers = crate::blockers_for(&self.correctness, self.stats.samples);
        if [self.stats.min_ns, self.stats.p10_ns, self.stats.median_ns, self.stats.p90_ns]
            .iter().any(|v| !v.is_finite() || *v <= 0.0)
            || self.stats.min_ns > self.stats.p10_ns
            || self.stats.p10_ns > self.stats.median_ns
            || self.stats.median_ns > self.stats.p90_ns
        {
            blockers.push("invalid MoE timing distribution".into());
        }
        if self.cell.hardware.is_empty() || self.cell.weight_enc.is_empty()
            || [self.cell.n_cu, self.cell.decode_rung, self.cell.topk,
                self.cell.hidden, self.cell.inter_local, self.cell.experts].contains(&0)
            || self.cell.topk > self.cell.experts
            || self.campaign.is_empty()
            || self.digests.implementation.is_empty() || self.digests.interpreter.is_empty()
            || self.digests.toolchain.is_empty() || self.digests.oracle != MOE_DECODE_ORACLE
        {
            blockers.push("incomplete MoE geometry/measurement identity".into());
        }
        if let Some(identity) = &self.digests.execution {
            if identity.validate().is_err() || identity.hardware != self.cell.hardware {
                blockers.push("invalid MoE execution identity".into());
            }
        }
        blockers
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

fn eligible(record: &MoeDecodeMeasurement, cell: &MoeDecodeCell, want: &Digests) -> bool {
    record.cell == *cell && record.state.is_selectable()
        && record.qualification_blockers().is_empty()
        && record.digests.stale_against(want).is_empty()
}

fn valid_policy(handoff_ns: f64, min_gain_fraction: f64) -> bool {
    handoff_ns.is_finite() && handoff_ns >= 0.0
        && min_gain_fraction.is_finite() && (0.0..=1.0).contains(&min_gain_fraction)
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
    let fallback = MoeDecodeSelection {
        route: MoeDecodeRoute::Interpreter,
        source: MoeDecodeSource::FixedFallback,
        projected_gain_ns: 0.0,
    };
    if !valid_policy(handoff_ns, min_gain_fraction) { return fallback; }
    let usable = |route: MoeDecodeRoute| {
        records
            .iter()
            .filter(|r| {
                r.route == route && eligible(r, cell, want)
            })
            .map(|r| r.stats.median_ns)
            .min_by(|a, b| a.total_cmp(b))
    };
    let (Some(interp), Some(standalone)) = (
        usable(MoeDecodeRoute::Interpreter),
        usable(MoeDecodeRoute::Standalone),
    ) else {
        return fallback;
    };
    let cost = standalone + (handoff_ns + interp * min_gain_fraction);
    if !cost.is_finite() { return fallback; }
    let gain = interp - (standalone + handoff_ns);
    let route = if cost <= interp {
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

/// Exact minimum of supplied body costs plus explicit handoff/margin assumptions.
/// Neither the handoff model nor these kernel-store records establish serving speedup.
pub fn policy_witness(records: &[MoeDecodeMeasurement], cell: &MoeDecodeCell,
    want: &Digests, handoff_ns: f64, min_gain_fraction: f64,
    selected: MoeDecodeSelection) -> Option<serde_json::Value> {
    if selected.source != MoeDecodeSource::Qualified || !valid_policy(handoff_ns, min_gain_fraction) {
        return None;
    }
    let candidates: std::collections::BTreeMap<_, _> = records.iter()
        .filter(|record| eligible(record, cell, want))
        .map(|record| (plow_asset::decode_objects::image_sha256(
            &serde_json::to_vec(record).unwrap()), record)).collect();
    let baseline = candidates.values().filter(|r| r.route == MoeDecodeRoute::Interpreter)
        .map(|r| r.stats.median_ns).min_by(f64::total_cmp)?;
    if !candidates.values().any(|r| r.route == MoeDecodeRoute::Standalone) { return None; }
    let cost = |r: &MoeDecodeMeasurement| r.stats.median_ns + match r.route {
        MoeDecodeRoute::Interpreter => 0.0,
        MoeDecodeRoute::Standalone => handoff_ns + baseline * min_gain_fraction,
    };
    if candidates.values().any(|r| !cost(r).is_finite()) { return None; }
    let (key, _) = candidates.iter().filter(|(_, r)| r.route == selected.route)
        .min_by(|(_, a), (_, b)| a.stats.median_ns.total_cmp(&b.stats.median_ns))?;
    let domain = plow_asset::decode_objects::image_sha256(&serde_json::to_vec(
        &(cell, want, handoff_ns, min_gain_fraction)).ok()?);
    Some(serde_json::json!({
        "required":[domain],
        "candidates":candidates.iter().map(|(key,r)|serde_json::json!({
            "domain":domain,"key":key,"cost":cost(r),"qualified":true
        })).collect::<Vec<_>>(),
        "choices":[{"domain":domain,"key":key}],
        "cost_scope":"supplied measured pair medians + external handoff + policy margin; not end-to-end performance",
        "handoff_ns":handoff_ns,"min_gain_fraction":min_gain_fraction,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digests(tag: &str) -> Digests {
        Digests {
            execution: None,
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
        let witness = policy_witness(&records, &cell(), &digests("a"),
            GFX950_SEGMENT_HANDOFF_NS, MIN_GAIN_FRACTION, sel).unwrap();
        let chosen = &witness["choices"][0]["key"];
        let candidates = witness["candidates"].as_array().unwrap();
        let cost = candidates.iter().find(|c| &c["key"] == chosen).unwrap()["cost"].as_f64().unwrap();
        assert!(candidates.iter().all(|c| c["cost"].as_f64().unwrap() >= cost));
        let changed = policy_witness(&records, &cell(), &digests("a"),
            GFX950_SEGMENT_HANDOFF_NS + 1.0, MIN_GAIN_FRACTION, sel).unwrap();
        assert_ne!(witness["required"], changed["required"]);
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
    fn forged_qualified_summaries_and_invalid_policy_inputs_cannot_select() {
        for mutation in 0..9 {
            let mut bad = rec(MoeDecodeRoute::Standalone, 1.0, "a");
            match mutation {
                0 => bad.stats.samples = 1,
                1 => bad.stats.median_ns = f64::NAN,
                2 => bad.stats.min_ns = -1.0,
                3 => bad.stats.p90_ns = 0.0,
                4 => bad.stats.p10_ns = 2000.0,
                5 => bad.campaign.clear(),
                6 => bad.cell.topk = bad.cell.experts + 1,
                7 => bad.digests.oracle = "unverified".into(),
                _ => bad.digests.toolchain.clear(),
            }
            assert!(!bad.qualification_blockers().is_empty(), "mutation {mutation}");
            let records = [rec(MoeDecodeRoute::Interpreter, 34.96, "a"), bad];
            let selection = select_moe_decode_route(&records, &cell(), &digests("a"), 0.0, 0.0);
            assert_eq!(selection.source, MoeDecodeSource::FixedFallback, "mutation {mutation}");
            assert!(policy_witness(&records, &cell(), &digests("a"), 0.0, 0.0, selection).is_none());
        }
        let records = [rec(MoeDecodeRoute::Interpreter, 34.96, "a"),
            rec(MoeDecodeRoute::Standalone, 1.0, "a")];
        for (handoff, margin) in [(f64::NAN, 0.0), (f64::INFINITY, 0.0), (-1.0, 0.0),
            (0.0, f64::NAN), (0.0, -0.1), (0.0, 1.1), (f64::MAX, 1.0)] {
            let selection = select_moe_decode_route(&records, &cell(), &digests("a"), handoff, margin);
            // Finite MAX still yields a finite, losing model cost at these tiny body durations.
            if handoff == f64::MAX { assert_eq!(selection.route, MoeDecodeRoute::Interpreter); }
            else { assert_eq!(selection.source, MoeDecodeSource::FixedFallback); }
        }
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
        assert_eq!(
            cell().key(),
            "amd/gfx950/mi350x|ncu256|b1|k16/h3584/i384/e896/mxfp4"
        );
    }
}
