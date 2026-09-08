use super::*;

fn inst(op: DevOp) -> DevInst64 {
    DevInst64 {
        op: op as u16,
        blocks: 1,
        fj: [0; 3],
        t: [packet::dev::TENSOR_NONE16; 8],
        i: [0; 8],
    }
}

fn prog(t: u32, ops: &[DevInst64]) -> DevProg {
    DevProg {
        t,
        packed_prefill_only: false,
        n_counter: 0,
        insts: ops.to_vec(),
        stream: Vec::new(),
        stream_ofs: Vec::new(),
        stream_len: Vec::new(),
        waits: Vec::new(),
        succs: Vec::new(),
        gq_stream: Vec::new(),
        gq_seg_ofs: Vec::new(),
        l2_domains: 0,
    }
}

/// The rule the plan states, and the mechanism that keeps it true: every opcode
/// the ISA defines has a classification, and a discriminant no opcode claims is
/// refused rather than ignored.
#[test]
fn every_isa_opcode_is_classified_and_unknown_ones_are_refused() {
    assert_eq!(DevOp::ALL.len(), table().len());
    for &op in DevOp::ALL {
        let cls = classify(op);
        assert!(
            !cls.identity.is_empty(),
            "{} has no named identity operand",
            op_name(op)
        );
        if !cls.classified {
            assert_eq!(
                cls.class,
                RowClass::C,
                "{} unclassified but not C",
                op_name(op)
            );
            assert_eq!(cls.disposition, Disposition::Refuse);
            assert!(!cls.note.is_empty(), "{} must say why", op_name(op));
        }
    }
    // 0xFFFF is not, and will not be, a DevOp: the wire slot is u16 and the ISA
    // is 154 opcodes.
    let unknown = classify_wire(0xFFFF);
    assert_eq!(unknown.class, RowClass::C);
    assert_eq!(unknown.disposition, Disposition::Refuse);
    assert!(!unknown.classified);
}

/// Exactly two opcodes have no usable operand spec in the ISA. `Mamba2Scan` has
/// no doc comment in `packet::dev` and no `packet::slots` entry at all
/// (`Provenance::Undocumented`); `AttnSelect` is listed `Provenance::Reserved`
/// ("body not built") while `runtime/amd/interp.hip:4021` dispatches it against a
/// passing hardware test. §3's rule makes both class C and refuses them.
///
/// This is the list the sibling ISA work should shrink. When it does, this test
/// says so rather than letting a stale refusal survive as folklore.
#[test]
fn only_two_opcodes_have_no_usable_isa_spec() {
    let unclassified: Vec<&'static str> = DevOp::ALL
        .iter()
        .filter(|&&op| !classify(op).classified)
        .map(|&op| op_name(op))
        .collect();
    assert_eq!(unclassified, vec!["AttnSelect", "Mamba2Scan"]);
}

/// The three §3 exemplars, exactly as the plan names them.
#[test]
fn the_named_class_c_hazards_are_class_c() {
    for op in [
        DevOp::FlashPrefill,
        DevOp::DsaPoolExpand,
        DevOp::IndexSelectPf,
        DevOp::IndexScorePf,
    ] {
        let cls = classify(op);
        assert_eq!(cls.class, RowClass::C, "{}", op_name(op));
        assert_eq!(cls.disposition, Disposition::NeedsConversion);
        assert!(!cls.disposition.packable());
    }
}

/// §3 and §11 name `IndexScore` (op 58) for the `s <= q_pos0 + t` derivation.
/// `d_index_score` (op_attention.h:4558) reads `kv_len[b]` per row; the
/// derivation is in `IndexScorePf` (op 117). This test pins the correction so a
/// later edit cannot quietly restore the plan's reading.
#[test]
fn index_score_is_per_row_and_its_prefill_twin_is_not() {
    assert_eq!(classify(DevOp::IndexScore).class, RowClass::B);
    assert_eq!(classify(DevOp::IndexScorePf).class, RowClass::C);
}

/// §3's class-B exemplars.
#[test]
fn the_named_class_b_operators_are_class_b() {
    for op in [
        DevOp::FlashDecode,
        DevOp::QwenGdnConv,
        DevOp::QwenGdnStep,
        DevOp::QwenGatedNorm,
        DevOp::QwenRmsNorm,
        DevOp::QwenSigmoidGate,
        DevOp::QwenQGateSplit,
    ] {
        let cls = classify(op);
        assert_eq!(cls.class, RowClass::B, "{}", op_name(op));
        assert_eq!(cls.disposition, Disposition::DescriptorFills);
        assert!(cls.disposition.packable());
    }
}

/// §3's class-D exemplars, and the boundary: the *bare* KDA state step and the
/// chunk ops are D; the gated/decode variants carry `bstride`+`parked` and are
/// not.
#[test]
fn the_named_class_d_operators_are_class_d() {
    for op in [
        DevOp::QwenGdnPrefill,
        DevOp::QwenGdnConvPrefill,
        DevOp::KdaStateStep,
        DevOp::KdaChunkPrepare,
        DevOp::KdaChunkIntra,
        DevOp::KdaChunkWu,
        DevOp::KdaChunkCarry,
    ] {
        let cls = classify(op);
        assert_eq!(cls.class, RowClass::D, "{}", op_name(op));
        assert_eq!(cls.disposition, Disposition::PerSpanLaunch);
    }
    // The decode-side KDA boundary ops already index state per row.
    for op in [DevOp::KdaConvStateStepG, DevOp::KdaDecodeFused] {
        assert_eq!(classify(op).class, RowClass::B, "{}", op_name(op));
    }
}

/// §3's class-A rule is "live `M` is the only new input". Operators with no row
/// count cannot express a packed batch at all, so they are class A and still
/// refused — the distinction §3 does not draw.
#[test]
fn class_a_without_a_row_count_is_refused_and_names_its_row_form() {
    let combine = classify(DevOp::MoeCombine);
    assert_eq!(combine.class, RowClass::A);
    assert_eq!(combine.extent, RowExtent::SingleRow);
    assert_eq!(combine.disposition, Disposition::UseRowForm);
    assert!(!combine.disposition.packable());

    // Its T-row twin is the one an emitter must select.
    let combine_pf = classify(DevOp::MoeCombinePf);
    assert_eq!(combine_pf.extent, RowExtent::RowCounted);
    assert!(combine_pf.disposition.packable());
}

/// Sinks are row-independent, so `FlashMerge` stays class A under packing —
/// the plan's claim, pinned.
#[test]
fn flash_merge_with_sinks_is_row_independent() {
    let cls = classify(DevOp::FlashMerge);
    assert_eq!(cls.class, RowClass::A);
    assert!(cls.disposition.packable());
    assert!(cls.note.contains("row-independent"));
}

/// Two addressing modes in one opcode. The static class is the strict one (a
/// capability check has no instruction to read); `refine` recovers the batched
/// ring from `i6`.
#[test]
fn head_norm_rope_is_class_c_only_in_its_legacy_addressing_mode() {
    assert_eq!(classify(DevOp::HeadNormRope).class, RowClass::C);

    let legacy = inst(DevOp::HeadNormRope);
    assert_eq!(refine(DevOp::HeadNormRope, &legacy).class, RowClass::C);

    let mut batched = inst(DevOp::HeadNormRope);
    batched.i[6] = 4; // n_batch_kv
    let cls = refine(DevOp::HeadNormRope, &batched);
    assert_eq!(cls.class, RowClass::B);
    assert!(cls.disposition.packable());
}

/// The same shape, inverted: `QwenHeadNormRope`'s `i6=prefill` is the *unsafe*
/// setting, so the refinement has to read it the other way round.
#[test]
fn qwen_head_norm_rope_is_class_c_in_prefill_mode() {
    let decode = inst(DevOp::QwenHeadNormRope);
    assert_eq!(refine(DevOp::QwenHeadNormRope, &decode).class, RowClass::B);

    let mut prefill = inst(DevOp::QwenHeadNormRope);
    prefill.i[6] = 1;
    assert_eq!(refine(DevOp::QwenHeadNormRope, &prefill).class, RowClass::C);
}

/// `LayerNorm`'s `i3=out_row0` is a packet-scalar row base into the DSA
/// index-key cache. Zero is a plain norm.
#[test]
fn layer_norm_is_class_c_only_when_it_writes_the_index_key_cache() {
    let plain = inst(DevOp::LayerNorm);
    assert_eq!(refine(DevOp::LayerNorm, &plain).class, RowClass::A);

    let mut cached = inst(DevOp::LayerNorm);
    cached.i[3] = 7;
    assert_eq!(refine(DevOp::LayerNorm, &cached).class, RowClass::C);
}

/// `KdaConv3`'s `j0=bstride` is the ISA's own name for "the T rows are B
/// independent sequences" (op_kda.h:1041) — the same idea as `active[B]`, under
/// a different name. The plan's §5.4 does not mention it.
#[test]
fn kda_conv3_is_class_b_on_the_independent_sequence_path() {
    let serial = inst(DevOp::KdaConv3);
    assert_eq!(refine(DevOp::KdaConv3, &serial).class, RowClass::D);

    let mut packed = inst(DevOp::KdaConv3);
    packed.fj[1] = 64; // j0 = bstride
    let cls = refine(DevOp::KdaConv3, &packed);
    assert_eq!(cls.class, RowClass::B);
    assert!(cls.disposition.packable());
}

/// An operand-conditional opcode is only clear if EVERY site in the program is
/// clear: one legacy `HeadNormRope` refuses the whole program.
#[test]
fn one_unsafe_site_refuses_the_opcode_for_the_whole_program() {
    let mut batched = inst(DevOp::HeadNormRope);
    batched.i[6] = 4;
    let legacy = inst(DevOp::HeadNormRope);

    let clear = audit_program(&prog(128, &[batched, batched]));
    assert!(clear.packable, "{:?}", clear.blockers);

    let mixed = audit_program(&prog(128, &[batched, legacy]));
    assert!(!mixed.packable);
    assert_eq!(mixed.ops.len(), 1, "one distinct opcode");
    assert_eq!(mixed.ops[0].count, 2);
    assert_eq!(mixed.ops[0].class, RowClass::C);
}

/// A program made only of class-A row-counted work packs; adding the one
/// class-C attention operator refuses it and names it.
#[test]
fn a_program_verdict_names_every_blocker() {
    let dense = prog(
        128,
        &[
            inst(DevOp::Embed),
            inst(DevOp::RmsNorm),
            inst(DevOp::Gemm),
            inst(DevOp::FlashDecode),
            inst(DevOp::FlashMerge),
            inst(DevOp::Argmax),
            inst(DevOp::ArgmaxFin),
        ],
    );
    let ok = audit_program(&dense);
    assert!(ok.packable, "{:?}", ok.blockers);
    assert_eq!(ok.by_class.get("A"), Some(&6));
    assert_eq!(ok.by_class.get("B"), Some(&1));

    let mut with_prefill = dense.insts.clone();
    with_prefill.push(inst(DevOp::FlashPrefill));
    let blocked = audit_program(&prog(128, &with_prefill));
    assert!(!blocked.packable);
    assert_eq!(blocked.blockers.len(), 1);
    assert!(blocked.blockers[0].contains("FlashPrefill"));
    assert!(blocked.blockers[0].contains("q_pos0"));
}

/// An opcode the build does not know is a wire discriminant, not a `DevOp`, and
/// the audit must still produce a row for it — refused.
#[test]
fn an_unknown_wire_opcode_is_reported_and_refused() {
    let mut unknown = inst(DevOp::Nop);
    unknown.op = 0xFFFE;
    let rep = audit_program(&prog(1, &[unknown]));
    assert!(!rep.packable);
    assert_eq!(rep.ops[0].name, None);
    assert!(rep.blockers[0].contains("<unknown>"));
}

/// The text renderer must not panic on any classification, and must name the
/// verdict — this is what a human reads.
#[test]
fn the_table_and_the_text_renderers_cover_every_opcode() {
    let rows = table();
    let text = table_text(&rows);
    assert!(text.starts_with("154 opcodes\n"), "{}", &text[..40]);
    for &op in DevOp::ALL {
        assert!(
            text.contains(op_name(op)),
            "{} missing from the table",
            op_name(op)
        );
    }

    let rep = BlobAudit {
        blob: "t".into(),
        target: 0,
        n_gpu: 1,
        packable: false,
        programs: vec![audit_program(&prog(1, &[inst(DevOp::FlashPrefill)]))],
    };
    let out = text_of(&rep);
    assert!(out.contains("REFUSED"));
    assert!(out.contains("FlashPrefill"));
}

fn text_of(rep: &BlobAudit) -> String {
    super::text(rep)
}

/// Run the classifier over what the shipped runtime fusion ("mixed step v1")
/// actually packs.
///
/// v1 synthesizes its packed program at model load by rewriting the blob's
/// ordinary prefill program against a hand-written opcode whitelist
/// (`exec/mixed_program.rs`'s match, re-checked by `exec/amd_mixed_step.rs`'s
/// 12-opcode `validate_program`). This audits the result — the actual
/// instruction stream v1 would upload — rather than reading the whitelist.
///
/// The finding this pins: **every operator v1 packs is class A or class B, and
/// its two class-C operators are the two it converts by hand.** `FlashPrefill`
/// keeps `i4=q_pos0` and is legal only because v1 admits exactly one prefill
/// span; `HeadNormRope` is promoted to class B not by an operand but by the
/// mixed object's compile-time `PLOW_MIXED_STEP` arm, which reads the row's slot
/// from `plow_mixed_row` (`runtime/amd/op_norm.h`). No IndexScore, no
/// DsaPoolExpand, no MoE, no recurrent state can appear: those opcodes have no
/// arm in v1's match and the synthesis refuses the whole model.
///
/// Skipped where the measured blob is absent, so the suite stays hermetic.
/// `exec::mixed_program` is `#[cfg(feature = "hsa")]` — v1 is AMD-only.
#[cfg(feature = "hsa")]
#[test]
fn mixed_step_v1_packs_only_class_a_and_class_b_plus_two_hand_converted_c() {
    let path = std::path::Path::new("/app/plow/build-gemma31/assets-final-plain/model.pkt");
    let Ok(buf) = std::fs::read(path) else {
        eprintln!("skip: no gemma31 blob on this host");
        return;
    };
    let blob = crate::asset::devblob::DevBlob::parse_l2(&buf, true).unwrap();
    let synth = crate::exec::mixed_program::synthesize(&blob, 4).expect("v1 synthesis");
    assert!(!synth.programs.is_empty());

    let mut seen: std::collections::BTreeSet<&'static str> = Default::default();
    for spec in &synth.programs {
        let audit = audit_program(&prog(spec.decode_rows, &spec.program.insts));
        // Printed under `--nocapture`: this is the table the fusion verdict in
        // docs/arch/17-operator-row-identity-classes.md quotes.
        for row in &audit.ops {
            eprintln!(
                "v1 decode_rows={} {:>3} {:<16} x{:<4} {} {:<12} {}",
                spec.decode_rows,
                row.opcode,
                row.name.unwrap_or("?"),
                row.count,
                row.class.as_str(),
                row.extent.as_str(),
                row.disposition.as_str()
            );
        }
        for row in &audit.ops {
            let name = row.name.expect("v1 emits no unknown opcode");
            seen.insert(name);
            assert!(row.classified, "v1 packs an unclassified opcode: {name}");
            // Everything v1 packs is A or B, EXCEPT the two it converts inside
            // the fusion itself.
            if row.class == RowClass::C {
                assert!(
                    matches!(name, "FlashPrefill" | "HeadNormRope"),
                    "v1 packs an unconverted class-C operator: {name}"
                );
            }
            assert_ne!(row.class, RowClass::D, "v1 packs recurrent state: {name}");
            // And nothing v1 packs is single-row: the whole rewrite exists to
            // give every packed operator a live row count.
            assert_ne!(
                row.extent,
                RowExtent::SingleRow,
                "v1 packs a single-row operator: {name}"
            );
        }
    }

    // The 12 opcodes `amd_mixed_step::validate_program` admits are a superset of
    // what synthesis actually emits; nothing outside it can appear.
    for name in &seen {
        assert!(
            matches!(
                *name,
                "Embed"
                    | "RmsNorm"
                    | "HeadNormRope"
                    | "Gemm"
                    | "GemmGlu"
                    | "FlashDecode"
                    | "FlashPrefill"
                    | "FlashMerge"
                    | "Argmax"
                    | "ArgmaxFin"
                    | "SoftCap"
                    | "NormResidual"
            ),
            "{name} is outside v1's device whitelist"
        );
    }
    // The two class-C conversions are load-bearing: if either stopped appearing,
    // v1 changed shape and this verdict has to be re-derived.
    assert!(seen.contains("FlashPrefill") && seen.contains("HeadNormRope"));
    assert!(
        seen.contains("FlashDecode"),
        "v1 always lifts a decode attention"
    );
}
