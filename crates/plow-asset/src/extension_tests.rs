//! The six rules, one refusal and one admission each, plus the backward-compatible
//! no-extension case. There is no extension to load yet, so these synthetic parent/extension
//! pairs ARE the contract: phase 4's emitter is written against them.

use super::*;
use packet::devbuild::{
    decode_rung_program_t, derive_roles, packed_prefill_program_t, token_batch_program_t,
    RoleSource,
};

// --- fixtures ----------------------------------------------------------------

fn tensors() -> Vec<TensorIdentity> {
    vec![
        TensorIdentity::new("tok_emb", 1 << 20, false),
        TensorIdentity::new("rope_cos", 4096, true),
        TensorIdentity::new("kv_cache_L0", 1 << 24, false),
    ]
}

const PARENT_CONFIG: &str = "\
#define PLOW_PACKET_HASH 0x0123456789abcdefull

#define PLOW_PACKET_GQA 8
#define PLOW_PACKET_DECODE_BATCH 32
#define PLOW_PACKET_LINEAR_BIAS 0
#define PLOW_PACKET_ROPE_HALF_HD64 0
#define PLOW_PACKET_ATTENTION_SINKS 0
#define PLOW_PACKET_HAS_FLASH_DECODE_FP8 1
#define PLOW_PACKET_HAS_FLASH_MLA_PREFILL_FP8 1
#ifndef PLOW_HAS_FLASH_HD64
#define PLOW_HAS_FLASH_HD64 0
#endif
#ifndef PLOW_HAS_FLASH_HD128
#define PLOW_HAS_FLASH_HD128 1
#endif
#ifndef PLOW_HAS_FLASH_HD256
#define PLOW_HAS_FLASH_HD256 0
#endif
#ifndef PLOW_HAS_FLASH_HD512
#define PLOW_HAS_FLASH_HD512 1
#endif
/* program-local: an extension may differ here */
#ifndef GM_SM_BK
#define GM_SM_BK 128
#endif
#define PLOW_PACKET_OBJECT_ARCH \"gfx942\"
";

fn axes() -> ConfigAxes {
    let mut a = ConfigAxes::from_config_header(PARENT_CONFIG);
    a.set(BLOB_AXIS_N_CU, 304)
        .set(BLOB_AXIS_TP_DEGREE, 8)
        .set(BLOB_AXIS_HIDDEN, 5120);
    a
}

fn parent() -> ParentFacts {
    ParentFacts {
        hash: packet_hash(b"the parent model.pkt image"),
        tensor_count: tensors().len() as u32,
        tensor_digest: tensor_table_digest(&tensors(), 8),
        config: axes(),
        kv: KvRing::FullCausal,
        programs: derive_roles(
            &[512, 2048, 8192, packed_prefill_program_t(8192), 1, 8, 32],
            RoleSource::Positional,
            |_| 0,
        ),
        decode_batch: 32,
        arenas: Arenas {
            reserved: Budget {
                inst_stream_bytes: 4 << 20,
                counters: 65_536,
                segments: 1024,
                workspace_bytes: 2 << 30,
            },
            segment_ceiling: 2048,
            growable: true,
        },
        pairing_hash: Some(0x0123_4567_89ab_cdef),
    }
}

/// A well-formed extension: one 4096-row prefill bucket between 2048 and 8192.
fn ext(p: &ParentFacts) -> ExtensionFacts {
    ExtensionFacts {
        name: "assets.ext/bucket-4096".into(),
        parent: ParentRef {
            parent_hash: p.hash,
            tensor_count: p.tensor_count,
            tensor_digest: p.tensor_digest,
        },
        config: axes(),
        programs: vec![ProgramRole::PrefillBucket { rows: 4096 }],
        chunk: 4096,
        budget: Budget {
            inst_stream_bytes: 1 << 20,
            counters: 20_000,
            segments: 600,
            workspace_bytes: 1 << 30,
        },
    }
}

fn refusal(r: Result<Merged, Refusal>) -> Refusal {
    r.expect_err("expected a refusal")
}

// --- the backward-compatible case --------------------------------------------

#[test]
fn a_parent_with_no_extensions_is_the_identity() {
    let p = parent();
    let m = merge(&p, &[]).unwrap();
    assert_eq!(m.programs.len(), p.programs.len());
    assert!(m.programs.iter().all(|(o, _)| *o == Origin::Parent));
    assert_eq!(
        m.programs.iter().map(|(_, r)| *r).collect::<Vec<_>>(),
        p.programs
    );
    assert!(m.growth.grew.is_empty(), "nothing grows for no extensions");
    assert_eq!(m.prefill_widths(), vec![512, 2048, 8192]);
    assert_eq!(m.decode_rungs(), vec![1, 8, 32]);
}

#[test]
fn the_two_role_sources_split_a_parent_and_an_extension_the_way_each_needs() {
    let p = parent();
    // The packed sibling at 8192 sits INSIDE the prefill range and is not a bucket; the three
    // trailing widths are the decode ladder. This is what `decode_rung_lo` plus the two markers
    // said before roles existed, and it is what `RoleSource::Positional` keeps saying.
    assert_eq!(
        p.programs,
        vec![
            ProgramRole::PrefillBucket { rows: 512 },
            ProgramRole::PrefillBucket { rows: 2048 },
            ProgramRole::PrefillBucket { rows: 8192 },
            ProgramRole::PackedSibling { of_rows: 8192 },
            ProgramRole::DecodeRung { rows: 1 },
            ProgramRole::DecodeRung { rows: 8 },
            ProgramRole::DecodeRung { rows: 32 },
        ]
    );
    let tb = derive_roles(
        &[512, token_batch_program_t(256), 1],
        RoleSource::Positional,
        |_| 0,
    );
    assert_eq!(tb[1], ProgramRole::TokenBatchBody { band: 0, rows: 256 });

    // An extension has no position to read a role out of, so it states them: an unmarked
    // program is a prefill bucket and a marked one is a rung, whatever its index.
    assert_eq!(
        derive_roles(&[4096], RoleSource::Stated, |_| 0),
        vec![ProgramRole::PrefillBucket { rows: 4096 }]
    );
    assert_eq!(
        derive_roles(&[decode_rung_program_t(12)], RoleSource::Stated, |_| 0),
        vec![ProgramRole::DecodeRung { rows: 12 }]
    );
    assert_eq!(
        derive_roles(&[4096], RoleSource::Positional, |_| 0),
        vec![ProgramRole::DecodeRung { rows: 4096 }],
        "the positional rule alone reads a lone program as a rung — the defect \
         `RoleSource::Stated` exists for"
    );
    // A stated bit wins in either container, so the two sources agree wherever both speak.
    assert_eq!(
        derive_roles(&[decode_rung_program_t(12)], RoleSource::Positional, |_| 0),
        derive_roles(&[decode_rung_program_t(12)], RoleSource::Stated, |_| 0)
    );
}

#[test]
fn a_well_formed_extension_is_admitted() {
    let p = parent();
    let m = merge(&p, &[ext(&p)]).unwrap();
    assert_eq!(m.prefill_widths(), vec![512, 2048, 4096, 8192]);
    assert_eq!(m.decode_rungs(), vec![1, 8, 32]);
    assert_eq!(
        m.programs.last().unwrap().0,
        Origin::Extension("assets.ext/bucket-4096".into())
    );
    assert!(m.growth.grew.is_empty());
}

// --- rule 1: parent identity -------------------------------------------------

#[test]
fn rule1_refuses_an_extension_for_a_different_packet() {
    let p = parent();
    let mut e = ext(&p);
    e.parent.parent_hash = packet_hash(b"some other model.pkt");
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::ParentIdentity);
    assert_eq!(r.extension, "assets.ext/bucket-4096");
    assert!(r.detail.contains("declares parent"));
    assert!(r.to_string().contains("rule 1 (parent identity)"));
}

// --- rule 2: tensor-table digest ---------------------------------------------

#[test]
fn rule2_refuses_a_reemitted_parent_with_the_same_hash_shape() {
    let p = parent();
    // A parent re-emitted with one extra tensor: every index past it now means something else.
    let mut moved = tensors();
    moved.insert(1, TensorIdentity::new("new_bias", 512, false));
    let mut e = ext(&p);
    e.parent.tensor_count = moved.len() as u32;
    e.parent.tensor_digest = tensor_table_digest(&moved, 8);
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::TensorTableDigest);
    assert!(r.detail.contains("4-tensor table"), "{}", r.detail);
}

#[test]
fn rule2_refuses_a_same_count_table_whose_contents_moved() {
    let p = parent();
    let mut swapped = tensors();
    swapped.swap(0, 1);
    let mut e = ext(&p);
    e.parent.tensor_digest = tensor_table_digest(&swapped, 8);
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::TensorTableDigest);
    assert!(r.detail.contains("would mean something else"));
}

#[test]
fn the_tensor_digest_separates_order_size_init_and_tp() {
    let base = tensor_table_digest(&tensors(), 8);
    assert_eq!(base, tensor_table_digest(&tensors(), 8), "not stable");

    let mut reordered = tensors();
    reordered.swap(0, 2);
    assert_ne!(base, tensor_table_digest(&reordered, 8), "order");

    let mut resized = tensors();
    resized[0].bytes += 1;
    assert_ne!(base, tensor_table_digest(&resized, 8), "bytes (dtype/shape)");

    let mut reinit = tensors();
    reinit[1].initialized = false;
    assert_ne!(base, tensor_table_digest(&reinit, 8), "initialized");

    let mut renamed = tensors();
    renamed[0].name = "tok_emb2".into();
    assert_ne!(base, tensor_table_digest(&renamed, 8), "name");

    assert_ne!(base, tensor_table_digest(&tensors(), 4), "tp degree");

    // Name concatenation must not alias: ("ab","") and ("a","b") are different tables.
    let a = vec![
        TensorIdentity::new("ab", 4, false),
        TensorIdentity::new("", 4, false),
    ];
    let b = vec![
        TensorIdentity::new("a", 4, false),
        TensorIdentity::new("b", 4, false),
    ];
    assert_ne!(tensor_table_digest(&a, 1), tensor_table_digest(&b, 1));
}

// --- rule 3: config compatibility --------------------------------------------

#[test]
fn rule3_reads_only_the_shared_axes_out_of_a_config_header() {
    let a = ConfigAxes::from_config_header(PARENT_CONFIG);
    assert_eq!(a.get("PLOW_PACKET_GQA"), Some(8));
    assert_eq!(a.get("PLOW_PACKET_DECODE_BATCH"), Some(32));
    assert_eq!(a.get("PLOW_HAS_FLASH_HD512"), Some(1));
    assert_eq!(a.get("PLOW_HAS_FLASH_HD64"), Some(0));
    // Program-local and string-valued macros are deliberately not axes.
    assert_eq!(a.get("GM_SM_BK"), None);
    assert_eq!(a.get("PLOW_PACKET_OBJECT_ARCH"), None);
    assert_eq!(a.get("PLOW_PACKET_HASH"), None);
}

#[test]
fn rule3_admits_a_program_local_difference() {
    let p = parent();
    let mut e = ext(&p);
    // A different tile choice is the entire point of an extension.
    let local = PARENT_CONFIG.replace("#define GM_SM_BK 128", "#define GM_SM_BK 64");
    e.config = ConfigAxes::from_config_header(&local);
    e.config
        .set(BLOB_AXIS_N_CU, 304)
        .set(BLOB_AXIS_TP_DEGREE, 8)
        .set(BLOB_AXIS_HIDDEN, 5120);
    merge(&p, &[e]).expect("a tile-choice difference must be admitted");
}

#[test]
fn rule3_refuses_a_shared_axis_difference_by_name() {
    let p = parent();
    for (macro_name, from, to) in [
        ("PLOW_PACKET_DECODE_BATCH", "32", "16"),
        ("PLOW_PACKET_GQA", "8", "4"),
        ("PLOW_HAS_FLASH_HD512", "1", "0"),
        ("PLOW_PACKET_HAS_FLASH_DECODE_FP8", "1", "0"),
    ] {
        let mut e = ext(&p);
        let src = PARENT_CONFIG.replace(
            &format!("#define {macro_name} {from}"),
            &format!("#define {macro_name} {to}"),
        );
        assert_ne!(src, PARENT_CONFIG, "{macro_name} substitution missed");
        e.config = ConfigAxes::from_config_header(&src);
        e.config
            .set(BLOB_AXIS_N_CU, 304)
            .set(BLOB_AXIS_TP_DEGREE, 8)
            .set(BLOB_AXIS_HIDDEN, 5120);
        let r = refusal(merge(&p, &[e]));
        assert_eq!(r.rule, Rule::ConfigCompatibility);
        assert!(r.detail.contains(macro_name), "{}", r.detail);
    }
}

#[test]
fn rule3_refuses_a_blob_axis_difference() {
    let p = parent();
    for (axis, v) in [
        (BLOB_AXIS_TP_DEGREE, 4),
        (BLOB_AXIS_N_CU, 256),
        (BLOB_AXIS_HIDDEN, 4096),
    ] {
        let mut e = ext(&p);
        e.config.set(axis, v);
        let r = refusal(merge(&p, &[e]));
        assert_eq!(r.rule, Rule::ConfigCompatibility);
        assert!(r.detail.contains(axis), "{}", r.detail);
    }
}

#[test]
fn rule3_refuses_an_axis_the_extension_does_not_state_at_all() {
    let p = parent();
    let mut e = ext(&p);
    e.config.0.remove("PLOW_PACKET_GQA");
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::ConfigCompatibility);
    assert!(r.detail.contains("absent"), "{}", r.detail);
}

// --- rule 4: the KV ring invariant -------------------------------------------

#[test]
fn rule4_full_causal_mla_is_unaffected() {
    let p = parent();
    assert_eq!(p.kv, KvRing::FullCausal);
    let mut e = ext(&p);
    // Wider than the parent's widest, at a chunk no ring could ever hold — and still fine,
    // because a full-causal model returns (ctx, MASK_NONE) and has no ring to overrun.
    e.programs = vec![ProgramRole::PrefillBucket { rows: 65_536 }];
    e.chunk = 65_536;
    merge(&p, &[e]).expect("full-causal MLA must be unaffected by rule 4");
}

#[test]
fn rule4_refuses_a_windowed_model_by_name() {
    let mut p = parent();
    p.kv = KvRing::Windowed {
        window: 1024,
        ring_rows: kv_ring_rows(1024, 8192), // 16384, the shipped Gemma-4 ring
    };
    let mut e = ext(&p);
    e.programs = vec![ProgramRole::PrefillBucket { rows: 16_384 }];
    e.chunk = 16_384;
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::KvRingInvariant);
    assert_eq!(r.extension, "assets.ext/bucket-4096");
    assert!(r.detail.contains("16384-row prefill bucket"), "{}", r.detail);
    assert!(r.detail.contains("32768"), "names the ring it needs: {}", r.detail);
}

#[test]
fn rule4_admits_a_windowed_bucket_the_parent_ring_already_covers() {
    let mut p = parent();
    p.kv = KvRing::Windowed {
        window: 1024,
        ring_rows: kv_ring_rows(1024, 8192),
    };
    let mut e = ext(&p);
    e.programs = vec![ProgramRole::PrefillBucket { rows: 4096 }];
    e.chunk = 4096;
    // 4096 is narrower than the parent's widest bucket, so the rule does not even arm; and at
    // chunk 8192 the ring would still be exactly right.
    merge(&p, &[e]).unwrap();

    let mut wide = ext(&p);
    wide.programs = vec![ProgramRole::PrefillBucket { rows: 16_384 }];
    wide.chunk = 8192; // a wider bucket that still chunks at the parent's chunk
    merge(&p, &[wide]).expect("ring >= window + chunk - 1 holds");
}

#[test]
fn rule4_refuses_a_wider_bucket_when_the_parent_states_no_kv_geometry() {
    let mut p = parent();
    p.kv = KvRing::Unstated;
    let mut e = ext(&p);
    e.programs = vec![ProgramRole::PrefillBucket { rows: 16_384 }];
    e.chunk = 16_384;
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::KvRingInvariant);
    assert!(r.detail.contains("states no KV window"), "{}", r.detail);
    assert!(r.detail.contains("shapes.kv_window"), "{}", r.detail);
}

#[test]
fn rule4_does_not_arm_on_anything_but_a_wider_bucket() {
    let mut p = parent();
    p.kv = KvRing::Unstated;
    // The motivating extension — a 4096 bucket between 2048 and 8192 — is narrower than the
    // parent's widest, so an unstated ring never blocks it. Nor do rungs, siblings or bodies.
    for programs in [
        vec![ProgramRole::PrefillBucket { rows: 4096 }],
        vec![ProgramRole::DecodeRung { rows: 12 }],
        vec![ProgramRole::PackedSibling { of_rows: 2048 }],
        vec![ProgramRole::TokenBatchBody { band: 32, rows: 512 }],
    ] {
        let mut e = ext(&p);
        e.programs = programs;
        merge(&p, &[e]).expect("rule 4 armed on a program that adds no wider bucket");
    }
}

#[test]
fn kv_ring_rows_is_the_dev_isa_invariant() {
    assert_eq!(kv_ring_rows(1024, 8192), 16_384);
    assert_eq!(kv_ring_rows(1024, 1024), 2048);
    assert_eq!(kv_ring_rows(4096, 4096), 8192);
    for (w, c) in [(1u32, 1u32), (1024, 8192), (7, 9), (4096, 512)] {
        assert!(kv_ring_rows(w, c) >= w + c - 1);
        assert!(kv_ring_rows(w, c).is_power_of_two());
    }
}

// --- rule 5: ladder well-formedness ------------------------------------------

#[test]
fn rule5_refuses_a_duplicate_prefill_bucket() {
    let p = parent();
    let mut e = ext(&p);
    e.programs = vec![ProgramRole::PrefillBucket { rows: 2048 }];
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::LadderWellFormedness);
    assert!(r.detail.contains("duplicate prefill bucket at 2048"), "{}", r.detail);
    assert!(r.detail.contains("may only ADD"));
}

#[test]
fn rule5_refuses_a_duplicate_decode_rung() {
    let p = parent();
    let mut e = ext(&p);
    e.programs = vec![ProgramRole::DecodeRung { rows: 8 }];
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::LadderWellFormedness);
    assert!(r.detail.contains("duplicate decode rung at 8"), "{}", r.detail);
}

#[test]
fn rule5_refuses_a_decode_rung_above_the_band() {
    let p = parent();
    let mut e = ext(&p);
    e.programs = vec![ProgramRole::DecodeRung { rows: 64 }];
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::LadderWellFormedness);
    assert!(r.detail.contains("outside the decode band 1..=32"), "{}", r.detail);
}

#[test]
fn rule5_admits_a_new_decode_rung_inside_the_band() {
    let p = parent();
    let mut e = ext(&p);
    e.programs = vec![ProgramRole::DecodeRung { rows: 12 }];
    let m = merge(&p, &[e]).unwrap();
    assert_eq!(m.decode_rungs(), vec![1, 8, 12, 32]);
}

#[test]
fn rule5_refuses_a_sibling_or_body_naming_a_bucket_nobody_has() {
    let p = parent();
    for role in [
        ProgramRole::PackedSibling { of_rows: 4096 },
        ProgramRole::TokenBatchBody { band: 32, rows: 4096 },
    ] {
        let mut e = ext(&p);
        e.programs = vec![role];
        let r = refusal(merge(&p, &[e]));
        assert_eq!(r.rule, Rule::LadderWellFormedness);
        assert!(r.detail.contains("4096-row prefill bucket"), "{}", r.detail);
        assert!(r.detail.contains("union does not have"), "{}", r.detail);
    }
}

#[test]
fn rule5_admits_a_sibling_and_body_that_come_with_their_bucket() {
    let p = parent();
    let mut e = ext(&p);
    e.programs = vec![
        ProgramRole::PrefillBucket { rows: 4096 },
        ProgramRole::PackedSibling { of_rows: 4096 },
        ProgramRole::TokenBatchBody { band: 32, rows: 4096 },
    ];
    let m = merge(&p, &[e]).unwrap();
    assert_eq!(m.prefill_widths(), vec![512, 2048, 4096, 8192]);
    assert_eq!(m.programs.len(), p.programs.len() + 3);
}

#[test]
fn rule5_names_the_extension_that_broke_the_ladder_not_the_last_one() {
    let p = parent();
    let good = ext(&p);
    let mut bad = ext(&p);
    bad.name = "assets.ext/bucket-4096-again".into();
    // Collides with the FIRST extension, not with the parent.
    bad.programs = vec![ProgramRole::PrefillBucket { rows: 4096 }];
    let mut trailing = ext(&p);
    trailing.name = "assets.ext/rung-12".into();
    trailing.programs = vec![ProgramRole::DecodeRung { rows: 12 }];
    let r = refusal(merge(&p, &[good, bad, trailing]));
    assert_eq!(r.rule, Rule::LadderWellFormedness);
    assert_eq!(r.extension, "assets.ext/bucket-4096-again");
    assert!(r.detail.contains("assets.ext/bucket-4096 already has one"), "{}", r.detail);
}

#[test]
fn rule5_refuses_a_parent_whose_own_ladder_is_malformed() {
    let mut p = parent();
    p.programs.push(ProgramRole::DecodeRung { rows: 8 });
    let r = refusal(merge(&p, &[]));
    assert_eq!(r.rule, Rule::LadderWellFormedness);
    assert_eq!(r.extension, "model.pkt");
}

// --- rule 6: budget ----------------------------------------------------------

#[test]
fn rule6_admits_what_fits_without_growing_anything() {
    let p = parent();
    let m = merge(&p, &[ext(&p)]).unwrap();
    assert!(m.growth.grew.is_empty());
}

#[test]
fn rule6_grows_the_vmm_pools_and_says_so() {
    let p = parent();
    let mut e = ext(&p);
    e.budget.inst_stream_bytes = 8 << 20;
    e.budget.counters = 100_000;
    let m = merge(&p, &[e]).unwrap();
    assert_eq!(m.growth.grew.len(), 2);
    assert!(m.growth.grew[0].contains("instruction-stream bytes from 4194304 to 8388608"));
    assert!(m.growth.grew[1].contains("counters from 65536 to 100000"));
}

#[test]
fn rule6_refuses_when_the_arenas_cannot_grow() {
    let mut p = parent();
    p.arenas.growable = false;
    let mut e = ext(&p);
    e.budget.workspace_bytes = 4 << 30;
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::Budget);
    assert!(r.detail.contains("workspace bytes"), "{}", r.detail);
    assert!(r.detail.contains("cannot grow them"));
}

#[test]
fn rule6_refuses_over_the_fixed_segment_ceiling_even_when_growable() {
    let p = parent();
    assert!(p.arenas.growable);
    let mut e = ext(&p);
    e.budget.segments = 4096;
    let r = refusal(merge(&p, &[e]));
    assert_eq!(r.rule, Rule::Budget);
    assert!(r.detail.contains("4096 ordered segments"), "{}", r.detail);
    assert!(r.detail.contains("not growable"));
}

#[test]
fn rule6_names_the_extension_that_demanded_the_most() {
    let mut p = parent();
    p.arenas.growable = false;
    let mut small = ext(&p);
    small.name = "assets.ext/small".into();
    small.budget.counters = 70_000;
    let mut big = ext(&p);
    big.name = "assets.ext/big".into();
    big.programs = vec![ProgramRole::DecodeRung { rows: 12 }];
    big.budget.counters = 200_000;
    let r = refusal(merge(&p, &[small, big]));
    assert_eq!(r.rule, Rule::Budget);
    assert_eq!(r.extension, "assets.ext/big");
}

// --- ordering ----------------------------------------------------------------

#[test]
fn the_rules_run_in_order_so_the_first_actionable_message_wins() {
    let p = parent();
    // Wrong parent AND a duplicate bucket AND over budget: rule 1 is what a human can act on.
    let mut e = ext(&p);
    e.parent.parent_hash = packet_hash(b"elsewhere");
    e.parent.tensor_digest = [0u8; 32];
    e.programs = vec![ProgramRole::PrefillBucket { rows: 512 }];
    e.budget.segments = 9999;
    assert_eq!(refusal(merge(&p, &[e])).rule, Rule::ParentIdentity);
}

// --- phase 3: requires.json and the object stamp -----------------------------

fn requires_json() -> Requires {
    Requires {
        version: 1,
        arch: "gfx942".into(),
        pairing_hash: "0xfeedfacecafebeef".into(),
        objects: vec![
            RequiredObject {
                stem: "interp_prefill".into(),
                sha256: "a".repeat(64),
                rows: Some(4096),
            },
            RequiredObject {
                stem: "interp_flash".into(),
                sha256: "b".repeat(64),
                rows: None,
            },
        ],
    }
}

#[test]
fn requires_json_round_trips_and_yields_the_rows_to_build() {
    let r = requires_json();
    r.validate().unwrap();
    let s = serde_json::to_string(&r).unwrap();
    assert_eq!(serde_json::from_str::<Requires>(&s).unwrap(), r);
    assert_eq!(r.rows(), vec![4096]);
    assert_eq!(parse_pairing_hash(&r.pairing_hash), Some(0xfeed_face_cafe_beef));
}

#[test]
fn requires_json_refuses_a_stem_that_escapes_the_hsaco_directory() {
    for stem in ["../interp_prefill", "a/b", "", "/abs"] {
        let mut r = requires_json();
        r.objects[0].stem = stem.into();
        assert!(r.validate().is_err(), "accepted stem `{stem}`");
    }
}

#[test]
fn requires_json_refuses_junk() {
    let mut v = requires_json();
    v.version = 2;
    assert!(v.validate().unwrap_err().contains("version 2"));

    let mut h = requires_json();
    h.pairing_hash = "not-a-hash".into();
    assert!(h.validate().unwrap_err().contains("pairing_hash"));

    let mut s = requires_json();
    s.objects[0].sha256 = "short".into();
    assert!(s.validate().unwrap_err().contains("sha256"));

    let mut d = requires_json();
    d.objects[1].stem = "interp_prefill".into();
    assert!(d.validate().unwrap_err().contains("twice"));

    let mut e = requires_json();
    e.objects.clear();
    assert!(e.validate().unwrap_err().contains("no objects"));
}

#[test]
fn the_object_stamp_must_name_the_extension_not_the_parent() {
    let ext_hash = 0xfeed_face_cafe_beef_u64;
    let parent_hash = 0x0123_4567_89ab_cdef_u64;

    check_object_stamp("ext", "interp_prefill", Some(ext_hash), ext_hash, Some(parent_hash))
        .expect("the extension's own stamp pairs");

    let r = check_object_stamp(
        "ext",
        "interp_prefill",
        Some(parent_hash),
        ext_hash,
        Some(parent_hash),
    )
    .unwrap_err();
    assert_eq!(r.rule, Rule::ConfigCompatibility);
    assert!(r.detail.contains("stamps the PARENT packet"), "{}", r.detail);
    assert!(r.detail.contains("none of this extension's arms"));

    let r = check_object_stamp("ext", "interp_prefill", Some(7), ext_hash, Some(parent_hash))
        .unwrap_err();
    assert!(r.detail.contains("stamps packet 0x0000000000000007"), "{}", r.detail);

    // An UNSTAMPED object is accepted beside a parent (it is a general object) but never
    // beside an extension: nothing then shows it has the arms this bucket needs.
    let r = check_object_stamp("ext", "interp_prefill", None, ext_hash, Some(parent_hash))
        .unwrap_err();
    assert!(r.detail.contains("carries no packet-pairing stamp"), "{}", r.detail);
}

// --- misc --------------------------------------------------------------------

#[test]
fn hex_and_packet_hash_are_stable_and_content_addressed() {
    assert_eq!(packet_hash(b"abc"), packet_hash(b"abc"));
    assert_ne!(packet_hash(b"abc"), packet_hash(b"abd"));
    assert_eq!(hex(&packet_hash(b"")).len(), 64);
    assert_eq!(
        hex(&packet_hash(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn the_parent_ref_wire_type_converts_to_the_owned_one() {
    let p = parent();
    let wire = packet::ext::BlobParentRef::new(p.hash, p.tensor_digest, p.tensor_count);
    let owned: ParentRef = (&wire).into();
    assert_eq!(owned.parent_hash, p.hash);
    assert_eq!(owned.tensor_digest, p.tensor_digest);
    assert_eq!(owned.tensor_count, p.tensor_count);
}
