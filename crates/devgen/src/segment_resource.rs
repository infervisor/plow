//! `build.json`'s `segment_resource` — what each wave-class segment reserves, and the one
//! segmentation defect whose failure mode is corruption rather than degradation.
//!
//! ## The bug class this exists to make visible
//!
//! The host relaunches the interpreter once per segment, on the code object matching that
//! segment's wave class, at that class's thread count ([`docs/arch/06-runtime.md`], "Segmented
//! dispatch"). The blob does not carry the class: the runtime RE-DERIVES it from the emitted
//! stream, and **both GPU backends implement the same default rule** —
//! `plowrt::exec::amd::derive_segments` and `plowrt::asset::devblob::seg_classes_with`:
//!
//! > a segment is class 4 (256 threads) iff ANY stream entry in it points at a flash-prefill
//! > instruction; everything else is class 8.
//!
//! `ANY`, not `every`. So a segment holding one `FlashPrefill` beside anything else sends the
//! WHOLE segment to the four-wave flash object, which dispatches only the ops it carries and
//! silently drops the rest. `crates/devgen/src/lib.rs`'s knob table records that exact outcome
//! twice, in production, both times as **zero logits**: `PLOW_L2_PLACE` "formerly overwrote the
//! wave-class tag on a MULTI-SEGMENT program", and `PLOW_UNISEG` "destroyed the wave-class split
//! → zero logits, 8.7 ms 'prefill'". Both were found by serving a model, not by emitting one.
//!
//! NVIDIA carries one purity refusal already — a segment mixing exact HD256/GQA2 attention with
//! another operator is rejected BY NAME at load — which is the shape the rest of the
//! classification lacks, on both backends. See [`ClassRule`] for the per-target scope.
//!
//! The emitter's own segmentation (`packet::devbuild::Builder`'s `wave_class`) and the runtime's
//! re-derivation are two different functions over the same data, written in two crates, agreeing
//! by convention. Four modules then rewrite `StreamEnt::seg` AFTER the Builder has cut it —
//! `attention_prefill_role`, `dense_cublaslt`, `fp8_m1_role`, `gemv_decode_role` — and each
//! checks only that its own instructions stayed contiguous, never that the result is still pure.
//! Nothing, anywhere, checks the two ends against each other.
//!
//! That is all this module does: apply the CONSUMER's rule to the emitted blob, at emit time,
//! and write the answer down next to the segmentation that produced it.
//!
//! ## Derived from the emitted instruction stream, like everything else here
//!
//! Same doctrine as [`crate::dispatch_audit`], for the same reason: nothing below reads the
//! [`crate::emit_config`], the environment, or the emitter's intent. Segment ids come from
//! [`packet::dev::StreamEnt::seg`], opcodes from the instruction table, and the CU set from
//! `Program::stream` / `stream_ofs` / `stream_len` — the per-CU dispatch the blob on disk
//! actually carries. Deriving the class from the emitter's inputs instead would reproduce, one
//! level up, exactly the drift this exists to catch: `PLOW_UNISEG` corrupted the tag while every
//! emitter-side input still said the split was requested.
//!
//! ## What is deliberately NOT modelled
//!
//! **The launch geometry.** A segment's thread count is a property of the OBJECT the host picks,
//! and the packet records neither the object nor the occupancy. So [`co_resident`] can decide CU
//! disjointness and derived-class equality and nothing else; the remaining precondition — that
//! two segments launch at the same occupancy — is not checkable from this artifact. That gap is
//! the thing a multi-queue lowering has to close first, and naming it is more useful than
//! guessing at it.
//!
//! **The other purity rules.** `derive_segments` also has `mla_pure`, `xr_wave_pure` and the KDA
//! predicates. Those are written to DEGRADE on a mismatch (their comments say so: "either
//! mismatch degrades, never corrupts"), and a degrade-safe rule does not need an emit-time
//! tripwire. The flash rule is the one that is unconditional and impure-tolerant, so it is the
//! one checked here.

use std::collections::{BTreeMap, BTreeSet};

use packet::dev::DevOp;
use packet::devbuild::{Model, Program};
use serde_json::{json, Value};

/// One wave-class segment's resource footprint, derived from the emitted dispatch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentResource {
    /// Index into `Model::progs`.
    pub program: usize,
    /// Rows the program runs, decoded from `Model::prog_t` — the same `t` the dispatch audit
    /// labels its rows with, so the two sections join on it.
    pub t: u32,
    /// `0prefill` / `1decode` / `2token_batch`, ordered so a lexical sort groups them.
    pub kind: &'static str,
    /// [`packet::dev::StreamEnt::seg`].
    pub seg: u16,
    /// Distinct instructions reachable from this segment's stream entries.
    pub insts: u32,
    /// Distinct CU ids carrying at least one entry of this segment — the reservation, recovered
    /// from the dispatch rather than from the emit site.
    pub cus: u32,
    /// Largest [`packet::dev::DevInst::blocks`] over this segment's instructions.
    pub workgroups: u32,
    /// Opcodes present, so a mixed segment is legible in the manifest without a second lookup.
    pub ops: BTreeSet<u16>,
}

/// Which consumer re-derives the wave class for this target, and therefore whose rule this
/// section is entitled to apply.
///
/// Both GPU backends implement the SAME default rule — `plowrt::exec::amd::derive_segments` and
/// `plowrt::asset::devblob::seg_classes_with` each set class 4 when ANY entry of a segment is a
/// flash-prefill op — so one predicate serves both. The CPU and Apple engines honour
/// `StreamEnt::seg` (they filter the stream by it) but derive no wave class and select no
/// per-class object, so there is nothing to be impure with respect to.
///
/// A target whose rule is not modelled here gets resource facts and NO class claim. That follows
/// [`crate::dispatch_audit`]'s rule for an unknowable tile: "a guessed tile would put a
/// fabricated percentage in the manifest, which is strictly worse than an absent one", and the
/// `resource_contract` in [`crate::manifest`], which is already `null` off AMD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClassRule {
    /// `plowrt::exec::amd::derive_segments`.
    Amd,
    /// `plowrt::asset::devblob::seg_classes_with` at its DEFAULT policy.
    ///
    /// NVIDIA's classification additionally has `pure_mode` (1/2/3), `fa512_mode` (1/2) and
    /// `fa256_gqa2`, taken from `PLOW_PF_SEG_PURE` / `PLOW_PF_SEG_FA512` /
    /// `PLOW_PF_SEG_FA256_GQA2` at SERVE time, or inferred from the packet's isolated segments.
    /// Under `pure_mode` the polarity inverts — a segment is GEMM-class only if every op in it
    /// is a mapped GEMM — and under `fa512_mode` hd512 flash becomes class 2. None of those are
    /// decidable from the blob alone, so this section asserts the default policy only. The
    /// `fa256_gqa2` arm needs no tripwire from here: it already refuses a mixed segment by name
    /// at load.
    Nvidia,
    /// CPU / Apple: segment-aware, but no wave class and no per-class object.
    NoWaveClass,
}

impl ClassRule {
    fn of(arch: &str) -> Self {
        if arch.starts_with("gfx") {
            Self::Amd
        } else if arch.starts_with("sm_") {
            Self::Nvidia
        } else {
            Self::NoWaveClass
        }
    }

    fn classifies(self) -> bool {
        !matches!(self, Self::NoWaveClass)
    }

    fn name(self) -> &'static str {
        match self {
            Self::Amd => "amd/derive_segments",
            Self::Nvidia => "nvidia/seg_classes_with@default-policy",
            Self::NoWaveClass => "none",
        }
    }
}

/// The flash-interpreter predicate both GPU backends implement, reproduced here against the
/// emitted instruction rather than the emitter's intent.
fn is_flash_prefill(op: u16) -> bool {
    op == DevOp::FlashPrefill as u16 || op == DevOp::FlashPrefillFp8 as u16
}

impl SegmentResource {
    /// This segment would be relaunched on the four-wave flash object.
    pub fn flash_class(&self) -> bool {
        self.ops.iter().copied().any(is_flash_prefill)
    }

    /// It would be relaunched there while carrying work that object does not dispatch.
    ///
    /// This is the corrupting case, not a slow one: the flash object skips every op it does not
    /// carry, so the non-flash instructions in this segment do not run at all and nothing
    /// reports it.
    pub fn flash_impure(&self) -> bool {
        self.flash_class() && self.ops.iter().copied().any(|op| !is_flash_prefill(op))
    }

    /// Opcodes that would be dropped by the flash object, named for the finding.
    fn stranded(&self) -> Vec<String> {
        self.ops
            .iter()
            .copied()
            .filter(|&op| !is_flash_prefill(op))
            .map(name_of)
            .collect()
    }
}

fn name_of(op: u16) -> String {
    DevOp::from_u16(op).map_or_else(|| format!("op{op}"), |op| format!("{op:?}"))
}

/// Two segments may occupy the device at once — as far as the PACKET can tell.
///
/// Necessary, not sufficient. Co-residency additionally requires that both launch at the same
/// occupancy, which is a property of the objects the host selects and is absent from the blob
/// (see the module note). Nothing in this crate schedules on the answer today; it exists so the
/// facts a concurrent lowering needs are derived in one place, with their limits stated.
pub fn co_resident(a: &SegmentResource, b: &SegmentResource, cus_a: &[u32], cus_b: &[u32]) -> bool {
    a.flash_class() == b.flash_class()
        && cus_a
            .iter()
            .collect::<BTreeSet<_>>()
            .is_disjoint(&cus_b.iter().collect())
}

/// Per-segment CU id sets for one program, indexed by segment.
fn cu_sets(p: &Program) -> BTreeMap<u16, BTreeSet<u32>> {
    let mut out: BTreeMap<u16, BTreeSet<u32>> = BTreeMap::new();
    for cu in 0..p.stream_ofs.len().min(p.stream_len.len()) {
        let a = p.stream_ofs[cu] as usize;
        let b = a.saturating_add(p.stream_len[cu] as usize);
        for e in &p.stream[a.min(p.stream.len())..b.min(p.stream.len())] {
            out.entry(e.seg).or_default().insert(cu as u32);
        }
    }
    out
}

/// Every segment of one program.
pub fn segments(p: &Program, program: usize, t: u32, kind: &'static str) -> Vec<SegmentResource> {
    let cus = cu_sets(p);
    let mut insts: BTreeMap<u16, BTreeSet<u32>> = BTreeMap::new();
    for e in &p.stream {
        insts.entry(e.seg).or_default().insert(e.inst);
    }
    insts
        .into_iter()
        .map(|(seg, inst_ids)| {
            let mut ops = BTreeSet::new();
            let mut workgroups = 0u32;
            for &j in &inst_ids {
                if let Some(d) = p.insts.get(j as usize) {
                    ops.insert(d.op);
                    workgroups = workgroups.max(u32::from(d.blocks));
                }
            }
            SegmentResource {
                program,
                t,
                kind,
                seg,
                insts: inst_ids.len() as u32,
                cus: cus.get(&seg).map_or(0, BTreeSet::len) as u32,
                workgroups,
                ops,
            }
        })
        .collect()
}

/// Every segment of every program in the blob.
pub fn table(m: &Model) -> Vec<SegmentResource> {
    let dec_lo = packet::devbuild::decode_rung_lo(&m.prog_t);
    let mut out = Vec::new();
    for (pi, p) in m.progs.iter().enumerate() {
        let encoded_t = m.prog_t.get(pi).copied().unwrap_or(0);
        let kind = if pi >= dec_lo {
            "1decode"
        } else if packet::devbuild::is_token_batch_program(encoded_t) {
            "2token_batch"
        } else {
            "0prefill"
        };
        out.extend(segments(
            p,
            pi,
            packet::devbuild::program_rows(encoded_t),
            kind,
        ));
    }
    out
}

fn row_json(r: &SegmentResource, rule: ClassRule) -> Value {
    json!({
        "program": r.program,
        "kind": &r.kind[1..],
        "t": r.t,
        "seg": r.seg,
        "insts": r.insts,
        "cus": r.cus,
        "workgroups": r.workgroups,
        // Absent, not guessed, where no consumer rule is modelled for this target.
        "flash_class": if rule.classifies() {
            json!(r.flash_class())
        } else {
            Value::Null
        },
        "ops": r.ops.iter().copied().map(name_of).collect::<Vec<_>>(),
    })
}

/// The `segment_resource` section of `build.json`.
pub fn section(m: &Model, arch: &str) -> Value {
    let rule = ClassRule::of(arch);
    let rows = table(m);
    let findings: Vec<Value> = rows
        .iter()
        .filter(|r| rule.classifies() && r.flash_impure())
        .map(|r| {
            json!({
                "kind": "flash_segment_impure",
                "rule": rule.name(),
                "program": r.program,
                "t": r.t,
                "seg": r.seg,
                "cus": r.cus,
                "stranded_ops": r.stranded(),
            })
        })
        .collect();
    json!({
        "note": "Per-segment resource footprint, derived from Program::stream / stream_ofs / \
                 stream_len and the instruction table — never from the EmitConfig or the \
                 environment. `cus` is the distinct CU ids carrying an entry of the segment; \
                 `workgroups` is the largest DevInst::blocks in it.",
        "class_rule": rule.name(),
        "class_rule_note": "`flash_class` applies the CONSUMER's rule: any flash-prefill entry \
                            sends the whole segment to the four-wave flash object. AMD and \
                            NVIDIA implement the same default rule \
                            (exec::amd::derive_segments, asset::devblob::seg_classes_with). \
                            NVIDIA additionally has pure_mode / fa512_mode / fa256_gqa2, taken \
                            from PLOW_PF_SEG_PURE / PLOW_PF_SEG_FA512 / PLOW_PF_SEG_FA256_GQA2 \
                            at SERVE time; under those the polarity inverts or hd512 flash \
                            becomes class 2, so this section asserts the DEFAULT policy only \
                            (the fa256_gqa2 arm already refuses a mixed segment by name at \
                            load). CPU and Apple honour `seg` but derive no wave class, so \
                            `flash_class` is null and no finding is raised for them.",
        "tile_provenance": "NOT joined per segment, deliberately. `tuning.tile_source` is \
                            blob-global and decision-based (tune_demand::note_decision), which \
                            is the only correct source: the shape-keyed lookup log is what \
                            reported 48/48 covered while the packet was byte-identical to a \
                            --no-tuning build, because the ingest rung table maps tiles to \
                            opcodes by geometry. Per-segment provenance needs note_decision to \
                            carry the shape it decided; joining these rows to the lookup log \
                            would reproduce that exact false positive.",
        "co_residency": "The packet records no launch geometry, so segment co-residency is only \
                         decidable here up to CU disjointness and derived-class equality. Equal \
                         occupancy is an object property and is not checkable from this blob.",
        "segments": rows.iter().map(|r| row_json(r, rule)).collect::<Vec<_>>(),
        "findings": findings,
    })
}

/// `Err` with a rendered report when the blob carries a segment the flash object would strand.
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
        out.push_str(&format!(
            "  impure flash segment: program {} (t={}) segment {} holds {} beside a \
             flash-prefill op — the host relaunches the whole segment on the four-wave flash \
             object, which does not dispatch {}.\n",
            f["program"],
            f["t"],
            f["seg"],
            f["stranded_ops"].as_array().map_or_else(String::new, |v| v
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")),
            if f["stranded_ops"].as_array().map_or(0, Vec::len) == 1 {
                "it"
            } else {
                "them"
            },
        ));
    }
    if findings.len() > 8 {
        out.push_str(&format!("  ... and {} more.\n", findings.len() - 8));
    }
    Err(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{DevInst, StreamEnt};

    fn inst(op: DevOp, blocks: u16) -> DevInst {
        DevInst {
            op: op as u16,
            blocks,
            ..Default::default()
        }
    }

    fn ent(inst: u32, seg: u16) -> StreamEnt {
        StreamEnt {
            inst,
            seg,
            ..Default::default()
        }
    }

    /// Fields spelled out rather than defaulted, as `dispatch_audit`'s fixtures do — `Program`
    /// has no `Default`, and a new field should fail this to compile rather than be silently
    /// zero in a test that pins numbers.
    fn program(insts: Vec<DevInst>, stream: Vec<StreamEnt>, per_cu: usize) -> Program {
        let cus = if per_cu == 0 {
            0
        } else {
            stream.len() / per_cu
        };
        Program {
            hier_base: 0,
            n_cu: cus as u32,
            n_counter: 0,
            insts,
            stream,
            stream_ofs: (0..cus).map(|cu| (cu * per_cu) as u32).collect(),
            stream_len: vec![per_cu as u32; cus],
            waits: vec![],
            succs: vec![],
            tensors: vec![],
            gq_stream: vec![],
            gq_seg_ofs: vec![],
            l2_sms: 0,
            l2_domains: 0,
        }
    }

    /// Two CUs, two segments: segment 0 is pure flash, segment 1 is pure GEMM. Each CU carries
    /// one entry of each segment.
    fn split_program() -> Program {
        program(
            vec![inst(DevOp::FlashPrefill, 2), inst(DevOp::GemmSmall, 2)],
            vec![ent(0, 0), ent(1, 1), ent(0, 0), ent(1, 1)],
            2,
        )
    }

    #[test]
    fn pure_segments_carry_one_family_and_their_cu_reservation() {
        let segs = segments(&split_program(), 0, 128, "0prefill");
        assert_eq!(segs.len(), 2);
        assert!(segs[0].flash_class());
        assert!(!segs[0].flash_impure());
        assert!(!segs[1].flash_class());
        assert_eq!(segs[0].cus, 2);
        assert_eq!(segs[0].workgroups, 2);
        assert_eq!(segs[0].insts, 1);
    }

    /// The negative control: the two ops collapse into ONE segment, which is what
    /// `PLOW_UNISEG` did to a multi-segment program.
    #[test]
    fn collapsed_segment_strands_the_non_flash_op() {
        let mut p = split_program();
        for e in &mut p.stream {
            e.seg = 0;
        }
        let segs = segments(&p, 0, 128, "0prefill");
        assert_eq!(segs.len(), 1);
        assert!(segs[0].flash_class());
        assert!(segs[0].flash_impure());
        assert_eq!(segs[0].stranded(), vec!["GemmSmall".to_string()]);
    }

    #[test]
    fn check_names_the_stranded_op_and_passes_on_a_split_blob() {
        let mut p = split_program();
        let clean = json!({ "findings": [] });
        assert!(check(&clean).is_ok());

        for e in &mut p.stream {
            e.seg = 0;
        }
        let segs = segments(&p, 0, 128, "0prefill");
        let findings: Vec<Value> = segs
            .iter()
            .filter(|r| r.flash_impure())
            .map(|r| {
                json!({
                    "kind": "flash_segment_impure",
                    "program": r.program,
                    "t": r.t,
                    "seg": r.seg,
                    "cus": r.cus,
                    "stranded_ops": r.stranded(),
                })
            })
            .collect();
        let report = check(&json!({ "findings": findings })).unwrap_err();
        assert!(report.contains("GemmSmall"), "{report}");
        assert!(report.contains("four-wave flash object"), "{report}");
    }

    #[test]
    fn co_residency_needs_the_same_class_and_disjoint_cus() {
        let segs = segments(&split_program(), 0, 128, "0prefill");
        // Same program, different classes — never co-resident whatever the CUs say.
        assert!(!co_resident(&segs[0], &segs[1], &[0], &[1]));

        let flash = &segs[0];
        assert!(co_resident(flash, flash, &[0, 1], &[2, 3]));
        assert!(!co_resident(flash, flash, &[0, 1], &[1, 2]));
    }

    /// Both GPU backends re-derive the same default rule, so both get a class and a finding.
    /// CPU/Apple derive no wave class, so the row carries `null` and raises nothing — absent
    /// rather than fabricated.
    #[test]
    fn the_class_claim_follows_the_target_that_re_derives_it() {
        assert_eq!(ClassRule::of("gfx942"), ClassRule::Amd);
        assert_eq!(ClassRule::of("sm_90a"), ClassRule::Nvidia);
        assert_eq!(ClassRule::of("sm_120a"), ClassRule::Nvidia);
        assert_eq!(ClassRule::of("cpu"), ClassRule::NoWaveClass);
        assert_eq!(ClassRule::of("apple"), ClassRule::NoWaveClass);

        let segs = segments(&split_program(), 0, 128, "0prefill");
        for rule in [ClassRule::Amd, ClassRule::Nvidia] {
            assert_eq!(row_json(&segs[0], rule)["flash_class"], json!(true));
        }
        assert_eq!(
            row_json(&segs[0], ClassRule::NoWaveClass)["flash_class"],
            Value::Null
        );
    }

    /// The impurity finding is raised for both GPU targets and suppressed where no consumer
    /// re-derives a class — on the SAME collapsed program, so the suppression is the variable.
    #[test]
    fn findings_are_raised_per_target_rule_not_per_packet() {
        let mut p = split_program();
        for e in &mut p.stream {
            e.seg = 0;
        }
        let m = Model {
            n_cu: 2,
            target: 0,
            tensors: vec![],
            progs: vec![p],
            kv_row_insts: vec![],
            prog_t: vec![128],
            gen: vec![],
        };
        for arch in ["gfx942", "sm_90a"] {
            let s = section(&m, arch);
            assert_eq!(
                s["findings"].as_array().map(Vec::len),
                Some(1),
                "{arch}: {s}"
            );
            assert_eq!(s["findings"][0]["rule"], json!(ClassRule::of(arch).name()));
        }
        for arch in ["cpu", "apple"] {
            let s = section(&m, arch);
            assert_eq!(
                s["findings"].as_array().map(Vec::len),
                Some(0),
                "{arch}: {s}"
            );
            assert_eq!(s["class_rule"], json!("none"));
            assert_eq!(s["segments"][0]["flash_class"], Value::Null);
        }
    }

    #[test]
    fn an_empty_program_has_no_segments() {
        assert!(segments(&program(vec![], vec![], 0), 0, 0, "0prefill").is_empty());
    }
}
