//! `build.json`'s `dispatch_audit` — what shape of machine each matmul was actually
//! handed, and how much of it the work fills.
//!
//! ## The bug class this exists to make visible
//!
//! Both of the biggest emit defects found on Gemma-4-31B/MI300X were COMPILED CEILINGS: a
//! compile-time constant that does not match the shape the runtime presents. Both were worth
//! 20-40%, both were invisible in every log, manifest and test, and both were found only by
//! disassembling the built object.
//!
//!   * `PLOW_GEMV_MM` — the decode object compiles ONE `gemv_rows` body serving every
//!     `M <= MM` by predicating each activation row (`live = (m < M)`) and computing its dot
//!     product anyway. A shipping object built `MM=4` against a blob carrying T=1/2/4 decode
//!     programs ran FOUR dot chains per batch-1 token and discarded three. Fixed by
//!     batch-width-matched decode objects: +22-37% throughput, -28% TPOT at concurrency 1.
//!   * `pick_tile`'s CU budget — it ranks candidates by `rounds x per-tile cost` with
//!     `rounds = ceil(tiles / n_units)` and was handed the GLOBAL 304 CUs while `split3`/
//!     `split2` give q/k/v and gate/up DISJOINT sets of 76 or 152. Fixed at
//!     `crates/devgen/src/lib.rs`'s `let budget =`: -5.2%/-4.6% TTFT at 128/512.
//!
//! Neither needed new information. `cus.len()` was in scope at the emit site; the GEMV
//! ceiling is `tuning.gv_mm_max`, which this manifest has always written. What was missing
//! was that nobody wrote the two numbers down NEXT TO EACH OTHER, where a diff would show
//! one moving. That is all this module does.
//!
//! ## Derived from the emitted instruction stream, like everything else here
//!
//! Same doctrine as [`crate::manifest`]: nothing below reads the `EmitConfig`, the env, or the
//! emitter's intent. `M`/`N`/`K` are instruction immediates (`i[0..3]`, uniform across the
//! matmul family), the workgroup count is [`packet::dev::DevInst::blocks`] (which
//! `Builder::emit_dep` sets from `cus.len()`), and the CU set is recovered from
//! `Program::stream` / `stream_ofs`, which is the per-CU dispatch the blob on disk actually
//! carries. An audit derived from the emitter's intent would reintroduce, one level up,
//! exactly the drift it exists to catch.
//!
//! ## What is deliberately NOT scored
//!
//! Occupancy needs a tile, and a tile is only knowable here for the rungs THIS compiler
//! chose from [`crate::gemm_tile_of`]. The GLU-fused GEMMs take theirs from `GM_BM`/`GM_BN`,
//! which are `-D` defines of the object (192 or 256 depending on the build), and the grouped
//! MoE ops carry no token count in the packet at all. Those get a dispatch row with
//! `tiles: null` and no occupancy. A guessed tile would put a fabricated percentage in the
//! manifest, which is strictly worse than an absent one: the whole value of this section is
//! that a number in it can be trusted enough to act on.

use std::collections::BTreeMap;

use packet::dev::DevOp;
use packet::devbuild::Model;
use serde_json::{json, Value};

/// Occupancy below this fraction is reported as a finding. Overridable with
/// `PLOW_AUDIT_OCC_FLOOR` (percent, integer).
///
/// 50% is where a dispatch stops being explicable as tile quantization and starts being a
/// half-idle machine: the measured `o_proj`/`down_proj` case is 84 tiles over 304 CUs
/// (27.6%, a flat 18.1-18.5 GB/s per CU), and the two `pick_tile` budget cases that were
/// worth 5% each sit at 55% and 84%. A floor at 50 names the first and not the other two,
/// which is the intended sensitivity — the tile-budget defect is a RATIO between two rows,
/// not a low absolute occupancy, and `worst_by_cost` is what surfaces it.
pub const DEFAULT_OCCUPANCY_FLOOR_PCT: u32 = 50;

/// Fraction of GEMV row work that may be spent on dead rows before it is a finding.
/// Overridable with `PLOW_AUDIT_GEMV_WASTE_MAX` (percent, integer).
///
/// 25% is one wasted row in four — i.e. it admits a T=3 program on an `MM=4` object (the
/// honest cost of a power-of-two ceiling) and refuses the T=1-on-`MM=4` case that cost
/// 22-37%, which computes three dead rows out of four.
pub const DEFAULT_GEMV_WASTE_MAX_PCT: u32 = 25;

/// One matmul dispatch shape, deduplicated across the layers that repeat it.
///
/// The dedup key is everything except `count`: two rows differ only where the machine
/// treats them differently, so a 62-layer model produces a table a person can read and a
/// diff a person can review.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
struct Row {
    /// Sort key first, so `BTreeMap` ordering IS the emitted order and no later sort can
    /// disagree with it. `kind` is spelled "0prefill"/"1decode" to keep prefill first
    /// without a second comparator.
    kind: &'static str,
    t: u32,
    op: String,
    m: u32,
    n: u32,
    k: u32,
    /// Distinct CUs carrying a stream entry for this instruction — the `cus.len()` the emit
    /// site had, recovered from the dispatch rather than from the emitter.
    cus: u32,
    /// `DevInst::blocks`. Equal to `cus` except under `PLOW_GEMV_SPLIT`, which repeats CU
    /// ids so one CU runs several slices; keeping both is what makes that case legible.
    workgroups: u32,
    /// `(bm, bn)` when this compiler chose the tile, else `None` — see the module note.
    tile: Option<(u32, u32)>,
    /// Half-bytes per weight element (bf16 = 4, fp8 = 2, mxfp4 = 1). Half-bytes so the cost
    /// ordering below is exact integer arithmetic: a float sort key would make two emits of
    /// one blob differ in the last digit and defeat the point of the section.
    half_bytes: u32,
    /// Weight streams this op reads: 1 plain, 2 for the GLU pair, 3 for a fused QKV.
    streams: u32,
}

impl Row {
    fn tiles(&self) -> Option<u32> {
        let (bm, bn) = self.tile?;
        Some(self.m.div_ceil(bm) * self.n.div_ceil(bn))
    }

    /// `ceil(tiles / cus)` — `tile_cost`'s own arithmetic, which is the point: this is the
    /// term `pick_tile` ranks on, so a row here is directly comparable with the decision
    /// that produced it.
    fn rounds(&self) -> Option<u32> {
        let tiles = self.tiles()?;
        Some(tiles.div_ceil(self.cus.max(1)))
    }

    /// Fraction of the dispatched slots that carry a tile, over the whole multi-round run.
    ///
    /// `tiles / (rounds * cus)`, which for the single-round case IS `tiles / cus` — the
    /// 84/304 = 27.6% of the measured `o_proj`. Defining it over `rounds * cus` rather than
    /// clamping `tiles / cus` at 1 is what keeps it informative once `tiles > cus`: gate/up
    /// at T=128 is 168 tiles on 152 CUs, which clamps to a meaningless 1.0 and is really
    /// 55.3% of two rounds.
    fn occupancy(&self) -> Option<f64> {
        let (tiles, rounds) = (self.tiles()?, self.rounds()?);
        let slots = u64::from(rounds) * u64::from(self.cus.max(1));
        Some(round4(f64::from(tiles) / slots as f64))
    }

    /// Bytes this dispatch moves, weights dominating. Integer throughout, and an ESTIMATE:
    /// it is a weight for ordering findings by what they cost, not a roofline.
    fn bytes(&self) -> u64 {
        let (m, n, k) = (u64::from(self.m), u64::from(self.n), u64::from(self.k));
        let w = u64::from(self.streams) * n * k * u64::from(self.half_bytes) / 2;
        w + m * k * 2 + m * n * 2
    }

    /// `(1 - occupancy) x bytes`, in bytes, by exact integer arithmetic.
    ///
    /// The list is ordered by this and not by occupancy because the two disagree about what
    /// matters: a 5% occupancy on a 2 MiB op is noise, and a 55% occupancy on a 300 MiB op
    /// is the whole regression. Ordering by "how bad the ratio looks" is how a real cost
    /// stays buried under a page of harmless ones.
    fn waste_bytes(&self) -> u64 {
        let (Some(tiles), Some(rounds)) = (self.tiles(), self.rounds()) else {
            return 0;
        };
        let slots = u64::from(rounds) * u64::from(self.cus.max(1));
        let bytes = self.bytes();
        bytes - bytes * u64::from(tiles) / slots.max(1)
    }

    fn label(&self) -> String {
        format!("{} {}x{}x{} @t{}", self.op, self.m, self.n, self.k, self.t)
    }
}

/// Four decimals is finer than any decision made on this number and coarse enough that the
/// digits are the same on every host — the section has to diff cleanly or it is not doing
/// its job.
fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// Is this opcode a GEMV, i.e. does it run under the object's compiled `GV_MM_MAX` ceiling?
fn is_gemv(op: DevOp) -> bool {
    matches!(
        op,
        DevOp::Gemv
            | DevOp::GemvGlu
            | DevOp::GemvQkv
            | DevOp::GemvQkvg
            | DevOp::GemvFp8
            | DevOp::GemvGluFp8
            | DevOp::GemvFp8Blk
            | DevOp::GemvMxfp4
            | DevOp::GemvGluMxfp4
            | DevOp::GemvQkvMxfp4
            | DevOp::GemvQkvFp8
            | DevOp::GemvSz
            | DevOp::GemvGluSz
            | DevOp::GemvArgmax
            | DevOp::GemvF32
    )
}

/// Weight streams and encoding for the ops whose tile this compiler did not choose, so a
/// GEMV or a fused GLU still gets an honest byte estimate for the ordering.
///
/// `None` = not a matmul; the audit ignores it entirely.
fn stream_shape(op: DevOp) -> Option<(u32, u32)> {
    use DevOp::*;
    // (weight streams, half-bytes per weight element)
    Some(match op {
        Gemv | GemvSz | GemvArgmax | GemvF32 | GemmNorm => (1, 4),
        GemvQkv | GemvQkvg => (3, 4),
        GemvGlu | GemvGluSz | GemmGlu => (2, 4),
        GemvFp8 | GemvFp8Blk => (1, 2),
        GemvQkvFp8 => (3, 2),
        GemvGluFp8 | GemmGluFp8 => (2, 2),
        GemvMxfp4 => (1, 1),
        GemvQkvMxfp4 => (3, 1),
        GemvGluMxfp4 | GemmGluMxfp4 => (2, 1),
        _ => return None,
    })
}

/// `N` summed over the streams the instruction actually sweeps.
///
/// The fused QKV ops concatenate three output column ranges (`i1=Nq i3=Nk i4=Nv`, with
/// `Nv = 0` the legal two-stream form), so reading `i1` alone would under-report their
/// weight traffic by up to 3x and sort them below ops they dominate.
fn total_n(op: DevOp, i: &[u32; 8]) -> u32 {
    match op {
        DevOp::GemvQkv | DevOp::GemvQkvg | DevOp::GemvQkvFp8 | DevOp::GemvQkvMxfp4 => {
            i[1].saturating_add(i[3]).saturating_add(i[4])
        }
        _ => i[1],
    }
}

/// Distinct CUs carrying a stream entry for each instruction of one program.
///
/// `stream_ofs`/`stream_len` are indexed by CU, so walking them in CU order lets a
/// "last CU seen" marker count distinct CUs in one pass without a set per instruction.
fn cus_per_inst(p: &packet::devbuild::Program) -> Vec<u32> {
    let mut count = vec![0u32; p.insts.len()];
    let mut last = vec![u32::MAX; p.insts.len()];
    for cu in 0..p.stream_ofs.len().min(p.stream_len.len()) {
        let (a, b) = (
            p.stream_ofs[cu] as usize,
            (p.stream_ofs[cu] as usize).saturating_add(p.stream_len[cu] as usize),
        );
        for e in &p.stream[a.min(p.stream.len())..b.min(p.stream.len())] {
            let j = e.inst as usize;
            if j < count.len() && last[j] != cu as u32 {
                last[j] = cu as u32;
                count[j] += 1;
            }
        }
    }
    count
}

/// Collect one row per distinct matmul dispatch shape, with the number of instructions that
/// share it.
fn rows(m: &Model) -> BTreeMap<Row, u32> {
    let dec_lo = packet::devbuild::decode_rung_lo(&m.prog_t);
    let mut out: BTreeMap<Row, u32> = BTreeMap::new();
    for (pi, p) in m.progs.iter().enumerate() {
        let encoded_t = m.prog_t.get(pi).copied().unwrap_or(0);
        let kind = if pi >= dec_lo {
            "1decode"
        } else if packet::devbuild::is_token_batch_program(encoded_t) {
            "2token_batch"
        } else {
            "0prefill"
        };
        let t = packet::devbuild::program_rows(encoded_t);
        let cus = cus_per_inst(p);
        for (j, d) in p.insts.iter().enumerate() {
            let Some(op) = DevOp::from_u16(d.op) else {
                continue;
            };
            // The tile is authoritative where this compiler picked it; `stream_shape` covers
            // the rest of the family for the byte estimate only.
            let tiled = crate::gemm_tile_of(op, d.i[7]);
            let Some((streams, half_bytes)) =
                tiled.map(|(_, _, hb)| (1, hb)).or_else(|| stream_shape(op))
            else {
                continue;
            };
            let row = Row {
                kind,
                t,
                op: format!("{op:?}"),
                m: d.i[0],
                n: total_n(op, &d.i),
                k: d.i[2],
                cus: cus.get(j).copied().unwrap_or(0),
                workgroups: u32::from(d.blocks),
                tile: tiled.map(|(bm, bn, _)| (bm, bn)),
                half_bytes,
                streams,
            };
            *out.entry(row).or_insert(0) += 1;
        }
    }
    out
}

fn row_json(r: &Row, count: u32) -> Value {
    json!({
        "op": r.op,
        "kind": &r.kind[1..],
        "t": r.t,
        "m": r.m,
        "n": r.n,
        "k": r.k,
        "cus": r.cus,
        "workgroups": r.workgroups,
        "tile": r.tile.map(|(bm, bn)| format!("{bm}x{bn}")),
        "tiles": r.tiles(),
        "rounds": r.rounds(),
        "occupancy": r.occupancy(),
        "bytes": r.bytes(),
        "waste_bytes": r.waste_bytes(),
        "insts": count,
    })
}

/// The audit section, plus the findings the threshold check reads back.
///
/// `gv_mm_max` is `tuning.gv_mm_max` — the `next_pow2(decode_batch)` ceiling the object is
/// built with. It is passed in rather than recomputed so the audit and the `-D` the backend
/// section asks for can never disagree about what the object compiles.
pub fn section(m: &Model, gv_mm_max: u32, floor_pct: u32, waste_max_pct: u32) -> Value {
    let rows = rows(m);

    let ops: Vec<Value> = rows.iter().map(|(r, c)| row_json(r, *c)).collect();

    // Ordered by what the shortfall COSTS, then by the row's own key so ties are stable.
    let mut ranked: Vec<(&Row, u32)> = rows
        .iter()
        .filter(|(r, _)| r.tiles().is_some())
        .map(|(r, c)| (r, *c))
        .collect();
    ranked.sort_by(|a, b| {
        (b.0.waste_bytes() * u64::from(b.1))
            .cmp(&(a.0.waste_bytes() * u64::from(a.1)))
            .then_with(|| a.0.cmp(b.0))
    });
    let worst: Vec<Value> = ranked
        .iter()
        .take(16)
        .map(|(r, c)| {
            let mut v = row_json(r, *c);
            v["waste_bytes_total"] = json!(r.waste_bytes() * u64::from(*c));
            v
        })
        .collect();

    // GEMV ceiling: computed rows vs live rows. `gv_mm_max` is ONE constant for the whole
    // object, so a blob whose decode ladder spans T=1..4 necessarily runs its T=1 rung at a
    // 4x ratio — which is the defect, stated as a number, exactly as it was measured.
    let ceiling = gv_mm_max.max(1);
    let mut gemv: Vec<Value> = rows
        .iter()
        .filter(|(r, _)| is_gemv_name(&r.op))
        .map(|(r, c)| {
            let live = r.m.clamp(1, ceiling);
            json!({
                "op": r.op,
                "kind": &r.kind[1..],
                "t": r.t,
                "live_rows": r.m,
                "compiled_m": ceiling,
                "computed_per_useful": round4(f64::from(ceiling) / f64::from(live)),
                "wasted_row_fraction": round4(1.0 - f64::from(live) / f64::from(ceiling)),
                "bytes": r.bytes(),
                "insts": c,
            })
        })
        .collect();
    gemv.sort_by(|a, b| {
        b["wasted_row_fraction"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&a["wasted_row_fraction"].as_f64().unwrap_or(0.0))
            .then_with(|| a.to_string().cmp(&b.to_string()))
    });

    let floor = f64::from(floor_pct) / 100.0;
    let waste_max = f64::from(waste_max_pct) / 100.0;

    let mut findings: Vec<Value> = Vec::new();
    for (r, c) in &ranked {
        let Some(occ) = r.occupancy() else { continue };
        if occ < floor {
            findings.push(json!({
                "kind": "occupancy",
                "op": r.label(),
                "occupancy": occ,
                "tiles": r.tiles(),
                "cus": r.cus,
                "rounds": r.rounds(),
                "waste_bytes_total": r.waste_bytes() * u64::from(*c),
            }));
        }
    }
    let worst_gemv = gemv
        .iter()
        .find(|g| g["wasted_row_fraction"].as_f64().unwrap_or(0.0) > waste_max);
    if let Some(g) = worst_gemv {
        findings.push(json!({
            "kind": "gemv_ceiling",
            "op": format!("{} @t{}", g["op"].as_str().unwrap_or(""), g["t"]),
            "live_rows": g["live_rows"],
            "compiled_m": ceiling,
            "computed_per_useful": g["computed_per_useful"],
            "wasted_row_fraction": g["wasted_row_fraction"],
        }));
    }

    json!({
        "note": "derived from the emitted instruction stream: M/N/K are i[0..3], workgroups \
                 are DevInst::blocks, and `cus` is the distinct CU count carrying a stream \
                 entry for the instruction. `tiles`/`rounds`/`occupancy` are present only \
                 where this compiler chose the tile (the pick_tile rungs); the GLU-fused and \
                 grouped-MoE bodies take theirs from the object's -D defines and are reported \
                 without a guessed one.",
        "occupancy_definition": "tiles / (rounds * cus); equals tiles / cus in the \
                                 single-round case",
        "thresholds": {
            "occupancy_floor_pct": floor_pct,
            "occupancy_floor_env": "PLOW_AUDIT_OCC_FLOOR",
            "gemv_waste_max_pct": waste_max_pct,
            "gemv_waste_max_env": "PLOW_AUDIT_GEMV_WASTE_MAX",
            "promote_to_refusal_env": "PLOW_AUDIT_STRICT",
            "policy": "warn",
        },
        "gemv_ceiling": {
            "compiled_m": ceiling,
            "source": "tuning.gv_mm_max = next_pow2(decode_batch)",
            "rows": gemv,
        },
        "worst_by_cost": worst,
        "findings": findings,
        "ops": ops,
    })
}

/// `Row::op` is the `Debug` spelling, so the GEMV test is on the name rather than a second
/// `DevOp` round trip through a string.
fn is_gemv_name(op: &str) -> bool {
    DevOp::ALL
        .iter()
        .find(|o| format!("{o:?}") == op)
        .is_some_and(|o| is_gemv(*o))
}

/// The threshold check. `Ok(())` = clean; `Err(report)` = at least one finding.
///
/// Separate from [`section`] so the check is testable against a hand-built manifest and so
/// the emit path decides the POLICY (warn or refuse) in one place rather than this module
/// deciding it for every caller.
pub fn check(section: &Value) -> Result<(), String> {
    let findings = section
        .get("findings")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if findings.is_empty() {
        return Ok(());
    }
    let mut out = String::new();
    for f in findings.iter().take(8) {
        let line = match f["kind"].as_str() {
            Some("gemv_ceiling") => format!(
                "  GEMV ceiling: {} runs {} live row(s) under a compiled M={} — {}x computed \
                 per useful row, {:.1}% of the row work discarded.\n",
                f["op"].as_str().unwrap_or("?"),
                f["live_rows"],
                f["compiled_m"],
                f["computed_per_useful"],
                f["wasted_row_fraction"].as_f64().unwrap_or(0.0) * 100.0,
            ),
            _ => format!(
                "  occupancy: {} fills {:.1}% of its dispatch — {} tile(s) over {} CU(s) in {} \
                 round(s), ~{} MiB moved at that fill.\n",
                f["op"].as_str().unwrap_or("?"),
                f["occupancy"].as_f64().unwrap_or(0.0) * 100.0,
                f["tiles"],
                f["cus"],
                f["rounds"],
                f["waste_bytes_total"].as_u64().unwrap_or(0) / (1 << 20),
            ),
        };
        out.push_str(&line);
    }
    if findings.len() > 8 {
        out.push_str(&format!(
            "  ... and {} more (see build.json dispatch_audit.findings).\n",
            findings.len() - 8
        ));
    }
    Err(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{DevInst, StreamEnt};
    use packet::devbuild::Program;

    /// One program, one instruction, dispatched across `cus` CUs. Fields spelled out
    /// rather than defaulted, as `manifest.rs`'s own fixtures do — `Program` has no
    /// `Default`, and a new field should make this fail to compile rather than be silently
    /// zero in a test that pins numbers.
    fn prog(op: DevOp, i: [u32; 8], cus: u32) -> Program {
        Program {
            hier_base: 0,
            n_cu: cus,
            n_counter: 0,
            insts: vec![DevInst {
                op: op as u16,
                blocks: cus as u16,
                i,
                ..Default::default()
            }],
            stream: (0..cus)
                .map(|s| StreamEnt {
                    inst: 0,
                    slice: s,
                    ..Default::default()
                })
                .collect(),
            stream_ofs: (0..cus).collect(),
            stream_len: vec![1; cus as usize],
            waits: vec![],
            succs: vec![],
            tensors: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_sms: 0,
            l2_domains: 0,
        }
    }

    fn model(progs: Vec<Program>, prog_t: Vec<u32>) -> Model {
        Model {
            n_cu: 304,
            target: 0,
            tensors: vec![],
            progs,
            kv_row_insts: vec![],
            prog_t,
            gen: vec![],
        }
    }

    /// THE MEASURED CASE: `o_proj` at T=128 on Gemma-4-31B is 84 GemmSmall tiles handed the
    /// whole 304-CU machine, and runs at 27.6% occupancy and a flat 18.1-18.5 GB/s per CU.
    /// Nothing in the build output said so before this section existed.
    #[test]
    fn o_proj_at_t128_reports_27_6_percent() {
        let m = model(
            vec![prog(
                DevOp::GemmSmall,
                [128, 5376, 5376, 0, 0, 0, 0, 0],
                304,
            )],
            vec![128],
        );
        let s = section(
            &m,
            1,
            DEFAULT_OCCUPANCY_FLOOR_PCT,
            DEFAULT_GEMV_WASTE_MAX_PCT,
        );
        let row = &s["ops"][0];
        assert_eq!(row["tiles"], 84, "2 x ceil(5376/128) = 84");
        assert_eq!(row["cus"], 304);
        assert_eq!(row["rounds"], 1);
        assert_eq!(row["occupancy"], 0.2763);
        // And it is loud, not merely recorded.
        assert!(check(&s).is_err());
    }

    /// The `pick_tile` budget defect, both halves, as the numbers the fix was measured on:
    /// gate/up at T=128 is 168 tiles on 152 CUs (two rounds), k/v at T=512 is 256 tiles on
    /// 76 (four rounds). Clamping `tiles / cus` at 1 would report both as full.
    #[test]
    fn disjoint_cu_sets_report_their_real_fill() {
        let m = model(
            vec![
                prog(DevOp::GemmMed, [128, 21504, 5376, 0, 0, 0, 0, 0], 152),
                prog(DevOp::GemmSmall, [512, 4096, 5376, 0, 0, 0, 0, 0], 76),
            ],
            vec![128, 512],
        );
        let s = section(
            &m,
            1,
            DEFAULT_OCCUPANCY_FLOOR_PCT,
            DEFAULT_GEMV_WASTE_MAX_PCT,
        );
        let gate = s["ops"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["t"] == 128)
            .unwrap();
        assert_eq!(gate["tiles"], 168);
        assert_eq!(gate["rounds"], 2);
        assert_eq!(gate["occupancy"], 0.5526);
        let kv = s["ops"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["t"] == 512)
            .unwrap();
        assert_eq!(kv["tiles"], 256);
        assert_eq!(kv["rounds"], 4);
        assert_eq!(kv["occupancy"], 0.8421);
    }

    /// THE `PLOW_GEMV_MM` CASE: a T=1 decode program against an object compiled `MM=4`
    /// computes four dot chains per token and discards three.
    #[test]
    fn a_t1_program_under_a_compiled_m4_is_flagged() {
        let m = model(
            vec![prog(DevOp::Gemv, [1, 5376, 5376, 0, 0, 0, 0, 0], 304)],
            vec![1],
        );
        let s = section(
            &m,
            4,
            DEFAULT_OCCUPANCY_FLOOR_PCT,
            DEFAULT_GEMV_WASTE_MAX_PCT,
        );
        let g = &s["gemv_ceiling"]["rows"][0];
        assert_eq!(g["compiled_m"], 4);
        assert_eq!(g["live_rows"], 1);
        assert_eq!(g["computed_per_useful"], 4.0);
        assert_eq!(g["wasted_row_fraction"], 0.75);
        let report = check(&s).expect_err("a 4x ceiling must be a finding");
        assert!(report.contains("GEMV ceiling"), "{report}");
    }

    /// The matched configuration — the one the batch-width-matched decode objects produce —
    /// must be silent, or the check is a permanent warning nobody reads.
    #[test]
    fn a_width_matched_decode_object_is_clean() {
        let m = model(
            vec![prog(DevOp::Gemv, [1, 5376, 5376, 0, 0, 0, 0, 0], 304)],
            vec![1],
        );
        let s = section(
            &m,
            1,
            DEFAULT_OCCUPANCY_FLOOR_PCT,
            DEFAULT_GEMV_WASTE_MAX_PCT,
        );
        assert_eq!(s["gemv_ceiling"]["rows"][0]["wasted_row_fraction"], 0.0);
        assert_eq!(check(&s), Ok(()));
    }

    /// A well-filled GEMM must not trip the floor.
    #[test]
    fn a_full_dispatch_is_clean() {
        let m = model(
            vec![prog(DevOp::Gemm, [2048, 21504, 5376, 0, 0, 0, 0, 0], 304)],
            vec![2048],
        );
        let s = section(
            &m,
            1,
            DEFAULT_OCCUPANCY_FLOOR_PCT,
            DEFAULT_GEMV_WASTE_MAX_PCT,
        );
        assert_eq!(check(&s), Ok(()));
    }

    /// The section has to diff cleanly or a regression does not show up as one.
    #[test]
    fn two_emits_of_one_model_are_byte_identical() {
        let build = || {
            let m = model(
                vec![
                    prog(DevOp::GemmMed, [128, 21504, 5376, 0, 0, 0, 0, 0], 152),
                    prog(DevOp::GemmSmall, [128, 5376, 5376, 0, 0, 0, 0, 0], 304),
                    prog(DevOp::Gemv, [1, 5376, 5376, 0, 0, 0, 0, 0], 304),
                ],
                vec![128, 128, 1],
            );
            serde_json::to_vec_pretty(&section(
                &m,
                4,
                DEFAULT_OCCUPANCY_FLOOR_PCT,
                DEFAULT_GEMV_WASTE_MAX_PCT,
            ))
            .expect("audit section")
        };
        assert_eq!(build(), build());
    }

    /// Ordering is by what the shortfall costs, not by how bad the ratio looks: a tiny op at
    /// 3% must not outrank a large one at 55%.
    #[test]
    fn worst_by_cost_ranks_by_bytes_not_by_ratio() {
        let m = model(
            vec![
                // 1 tile on 304 CUs — a terrible ratio on almost no bytes.
                prog(DevOp::GemmSmall, [1, 128, 128, 0, 0, 0, 0, 0], 304),
                // 168 tiles on 152 CUs — 55%, on a 200 MiB weight.
                prog(DevOp::GemmMed, [128, 21504, 5376, 0, 0, 0, 0, 0], 152),
            ],
            vec![128, 128],
        );
        let s = section(
            &m,
            1,
            DEFAULT_OCCUPANCY_FLOOR_PCT,
            DEFAULT_GEMV_WASTE_MAX_PCT,
        );
        assert_eq!(s["worst_by_cost"][0]["op"], "GemmMed");
    }
}
