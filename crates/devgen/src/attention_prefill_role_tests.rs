use super::*;
use packet::devbuild::Builder;
use std::sync::atomic::{AtomicU64, Ordering};

fn fixture(hd: u32, with_map: bool, fused: bool) -> Model {
    let mut builder = Builder::new(132);
    let q = builder.tensor("q", 128 * 32 * u64::from(hd) * 2);
    let k = builder.tensor("k", 4 * 65536 * u64::from(hd) * 2);
    let v = builder.tensor("v", 4 * 65536 * u64::from(hd) * 2);
    let partial = builder.tensor("partial", 128 * 32 * u64::from(hd) * 4);
    let ml = builder.tensor("ml", 128 * 32 * 8);
    let out = builder.tensor("out", 128 * 32 * u64::from(hd) * 2);
    let map = builder.tensor("map", 256);
    let before = builder.emit(DevOp::Nop, builder.all(), &[], |_| {});
    let flash = builder.emit(DevOp::FlashPrefill, builder.all(), &[before], |op| {
        op.t[..5].copy_from_slice(&[partial, ml, q, k, v]);
        op.t[5] = if fused { out } else { TENSOR_NONE };
        op.t[7] = if with_map { map } else { TENSOR_NONE };
        op.i = [128, 128, 32, 4, 0, 0, hd, 1];
        op.j[0] = 65536;
        op.j[1] = u32::MAX;
        op.f[0] = 1.0;
    });
    let after = if fused {
        flash
    } else {
        builder.emit(DevOp::FlashMerge, builder.all(), &[flash], |op| {
            op.t[..3].copy_from_slice(&[out, partial, ml]);
            op.i[..4].copy_from_slice(&[128, 32, 1, hd]);
        })
    };
    builder.emit(DevOp::Nop, builder.all(), &[after], |_| {});
    let prefill = builder.finish();
    let mut decode = Builder::new(132);
    decode.adopt_tensors(prefill.tensors.clone());
    decode.emit(DevOp::Nop, decode.all(), &[], |_| {});
    Model {
        n_cu: 132,
        target: 0,
        tensors: prefill.tensors.clone(),
        progs: vec![prefill, decode.finish()],
        kv_row_insts: vec![],
        prog_t: vec![128, 1],
        gen: vec![],
    }
}

fn selection() -> Selection {
    Selection::from_image("attention.cubin".into(), b"cubin", false)
}

fn apply_output(
    model: &mut Model,
    sections: &mut Vec<SectionData>,
    profile: &str,
    output: &Path,
) -> Result<bool, String> {
    apply_output_object(model, sections, profile, output, "h100", 1024, false, None)
}

#[test]
fn wgmma_object_selects_its_tile_and_rejects_partial_output() {
    let directory = output_dir("wgmma");
    let output = directory.join("model.pkt");
    let mut globals = OBJECT_GLOBALS;
    globals[2].1 = 64;
    globals[3].1 = 32;
    std::fs::write(directory.join(OBJECT_FILE), object_image(&globals)).unwrap();
    let mut model = fixture(512, true, true);
    let mut sections = Vec::new();
    assert!(apply_output(&mut model, &mut sections, "sm90a", &output).unwrap());
    let roles = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    assert_eq!(
        roles.objects[&PREFILL_ATTENTION_HD512_WG32].attention,
        Some(capability(true))
    );

    let mut partial = fixture(512, true, false);
    let before = partial.to_blob();
    let mut sections = Vec::new();
    assert!(apply_output(&mut partial, &mut sections, "sm90a", &output).is_err());
    assert_eq!(partial.to_blob(), before);
    assert!(sections.is_empty());

    globals[3].1 = 64;
    std::fs::write(directory.join(OBJECT_FILE), object_image(&globals)).unwrap();
    let mut wide = fixture(512, true, true);
    assert!(apply_output(&mut wide, &mut Vec::new(), "sm90a", &output).is_err());
    globals[6].1 = 205_824;
    std::fs::write(directory.join(OBJECT_FILE), object_image(&globals)).unwrap();
    let mut sections = Vec::new();
    assert!(apply_output(&mut wide, &mut sections, "sm90a", &output).unwrap());
    let roles = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    let mut expected = capability(true);
    expected.kv_tile = 64;
    assert_eq!(
        roles.objects[&PREFILL_ATTENTION_HD512_WG32].attention,
        Some(expected)
    );

    let mut n_split = globals.to_vec();
    n_split.push(("plow_attention_score_partitions", 2));
    n_split[6].1 = 206_848;
    std::fs::write(directory.join(OBJECT_FILE), object_image(&n_split)).unwrap();
    let mut split = fixture(512, true, true);
    assert!(apply_output(&mut split, &mut Vec::new(), "sm90a", &output).unwrap());
    n_split[6].1 = 205_824;
    std::fs::write(directory.join(OBJECT_FILE), object_image(&n_split)).unwrap();
    assert!(apply_output(&mut split, &mut Vec::new(), "sm90a", &output).is_err());

    globals[3].1 = 16;
    std::fs::write(directory.join(OBJECT_FILE), object_image(&globals)).unwrap();
    assert!(apply_output(&mut partial, &mut sections, "sm90a", &output).is_err());
    std::fs::remove_dir_all(directory).unwrap();
}

fn output_dir(label: &str) -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "plow-attention-role-{label}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn object_image(globals: &[(&str, u32)]) -> Vec<u8> {
    plow_asset::cubin::synthetic_elf(OBJECT_ENTRY, globals, 90)
}

fn hd256_fixture() -> Model {
    let mut model = fixture(256, true, true);
    let flash = &mut model.progs[0].insts[1];
    flash.i[2] = 16;
    flash.i[3] = 8;
    flash.i[5] = 1024;
    model.tensors[flash.t[5] as usize].bytes = 128 * 16 * 256 * 2;
    model
}

fn h100_hardware() -> String {
    let spec = hwspec::registry::lookup("h100").unwrap();
    kernelcaps::HardwareFingerprint::from_spec(spec)
        .unwrap()
        .tuning_path()
}

fn hd256_image() -> Vec<u8> {
    plow_asset::cubin::synthetic_elf(HD256_OBJECT_ENTRY, &HD256_OBJECT_GLOBALS, 90)
}

fn hd256_record(model: &Model, image: &[u8]) -> tunedb::AttentionRoleMeasurement {
    let program_sha256 = plow_asset::program::with_model(model, |packet| {
        plow_asset::live_kv::program_digest(&packet.programs[0])
    });
    let object_sha256 = plow_asset::decode_objects::image_sha256(image);
    tunedb::AttentionRoleMeasurement {
        cell: tunedb::AttentionRoleCell {
            hardware: h100_hardware(),
            n_cu: model.n_cu,
            arch: "sm90a".into(),
            dtype: "bf16".into(),
            kv_dtype: "bf16".into(),
            head_dim: 256,
            gqa: 2,
            window: 1024,
            m_rung: model.prog_t[0],
            live_kv_bucket: tunedb::KvBucket::K1,
            topology: tunedb::AttentionTopology::Single,
        },
        role: PREFILL_ATTENTION_HD256_BKV32,
        object_file: HD256_OBJECT_FILE.into(),
        object_sha256: object_sha256.clone(),
        program_sha256,
        config: tunedb::AttentionRoleConfig {
            query_tile: 64,
            kv_tile: 32,
            warps: 8,
            stages: 2,
            nsplit: 1,
            group_factor: 2,
        },
        stats: tunedb::Stats::from_samples(vec![50.0; 5]).unwrap(),
        baseline: tunedb::Stats::from_samples(vec![100.0; 5]).unwrap(),
        digests: tunedb::Digests {
            implementation: hd256_implementation(),
            interpreter: object_sha256,
            toolchain: kernelcaps::toolchain_label(hwspec::IsaLevel::Sm90a),
            oracle: tunedb::ATTENTION_ROLE_ORACLE.into(),
        },
        correctness: tunedb::Correctness::Pass,
        state: tunedb::RecordState::Qualified,
        campaign: "test".into(),
    }
}

#[test]
fn exact_hd256_bkv32_object_preserves_existing_packet_segments() {
    let directory = output_dir("hd256-bkv32");
    let output = directory.join("model.pkt");
    let image = hd256_image();
    std::fs::write(directory.join(HD256_OBJECT_FILE), &image).unwrap();
    let mut model = hd256_fixture();
    let original_insts = model.progs[0].insts.clone();
    let store_root = directory.join("tuning");
    tunedb::TuneStore::new(&store_root)
        .publish_attention_roles(&h100_hardware(), vec![hd256_record(&model, &image)])
        .unwrap();
    let mut sections = Vec::new();
    assert!(apply_output_object(
        &mut model,
        &mut sections,
        "sm90a",
        &output,
        "h100",
        1024,
        false,
        store_root.to_str(),
    )
    .unwrap());
    assert_eq!(model.progs[0].insts, original_insts);
    let roles = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    assert_eq!(roles.programs[0].roles, [0, 11, 0]);
    assert_eq!(model.progs[0].gq_seg_ofs.len(), 4);
    let object = &roles.objects[&PREFILL_ATTENTION_HD256_BKV32];
    assert_eq!(object.abi, PREFILL_ATTENTION_HD256_BKV32_ABI);
    assert_eq!(object.attention.as_ref().unwrap().kv_tile, 32);
    assert_eq!(object.sha256.as_deref().map(str::len), Some(64));
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn hd256_qualification_can_select_a_subset_of_prefill_rungs() {
    let directory = output_dir("hd256-rung-subset");
    let output = directory.join("model.pkt");
    let image = hd256_image();
    std::fs::write(directory.join(HD256_OBJECT_FILE), &image).unwrap();
    let mut model = hd256_fixture();
    let mut second = hd256_fixture().progs.remove(0);
    second.insts[1].i[0] = 1024;
    second.insts[1].i[1] = 1024;
    model.progs.insert(1, second);
    model.prog_t.insert(1, 1024);
    let record = hd256_record(&model, &image);
    let store_root = directory.join("tuning");
    tunedb::TuneStore::new(&store_root)
        .publish_attention_roles(&h100_hardware(), vec![record])
        .unwrap();
    let mut sections = Vec::new();
    assert!(apply_output_object(
        &mut model,
        &mut sections,
        "sm90a",
        &output,
        "h100",
        1024,
        false,
        store_root.to_str(),
    )
    .unwrap());
    let roles = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    assert_eq!(roles.programs.len(), 1);
    assert_eq!(roles.programs[0].index, 0);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn hd256_object_presence_without_qualified_record_is_byte_identical() {
    let directory = output_dir("hd256-unqualified");
    let output = directory.join("model.pkt");
    std::fs::write(directory.join(HD256_OBJECT_FILE), hd256_image()).unwrap();
    let mut model = hd256_fixture();
    let before = model.to_blob();
    let mut sections = Vec::new();
    assert!(!apply_output(&mut model, &mut sections, "sm90a", &output).unwrap());
    assert_eq!(model.to_blob(), before);
    assert!(sections.is_empty());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn inexact_hd256_record_falls_back_byte_identically() {
    let directory = output_dir("hd256-inexact");
    let output = directory.join("model.pkt");
    let image = hd256_image();
    std::fs::write(directory.join(HD256_OBJECT_FILE), &image).unwrap();
    let mut model = hd256_fixture();
    let mut record = hd256_record(&model, &image);
    record.cell.m_rung /= 2;
    let store_root = directory.join("tuning");
    tunedb::TuneStore::new(&store_root)
        .publish_attention_roles(&h100_hardware(), vec![record])
        .unwrap();
    let before = model.to_blob();
    let mut sections = Vec::new();
    assert!(!apply_output_object(
        &mut model,
        &mut sections,
        "sm90a",
        &output,
        "h100",
        1024,
        false,
        store_root.to_str(),
    )
    .unwrap());
    assert_eq!(model.to_blob(), before);
    assert!(sections.is_empty());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn selected_hd256_object_hash_drift_fails_before_packet_mutation() {
    let directory = output_dir("hd256-hash-drift");
    let output = directory.join("model.pkt");
    let image = hd256_image();
    let mut model = hd256_fixture();
    let store_root = directory.join("tuning");
    tunedb::TuneStore::new(&store_root)
        .publish_attention_roles(&h100_hardware(), vec![hd256_record(&model, &image)])
        .unwrap();
    std::fs::write(directory.join(HD256_OBJECT_FILE), b"drifted object").unwrap();
    let before = model.to_blob();
    let mut sections = Vec::new();
    assert!(apply_output_object(
        &mut model,
        &mut sections,
        "sm90a",
        &output,
        "h100",
        1024,
        false,
        store_root.to_str(),
    )
    .unwrap_err()
    .contains("differs from qualified SHA256"));
    assert_eq!(model.to_blob(), before);
    assert!(sections.is_empty());
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn live_kv_coverage_has_each_reachable_bucket_once() {
    assert_eq!(live_kv_buckets(0), Vec::<tunedb::KvBucket>::new());
    assert_eq!(live_kv_buckets(2047), vec![tunedb::KvBucket::K1]);
    assert_eq!(
        live_kv_buckets(98304),
        vec![
            tunedb::KvBucket::K1,
            tunedb::KvBucket::K4,
            tunedb::KvBucket::K8,
            tunedb::KvBucket::K16,
            tunedb::KvBucket::K32,
            tunedb::KvBucket::K64,
            tunedb::KvBucket::K128,
        ]
    );
}

#[test]
fn required_cells_match_scheduler_reachability_through_16k() {
    use tunedb::{AttentionTopology as Topology, KvBucket};

    let rungs = [
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
    ];
    let buckets = live_kv_buckets(16384);
    let mut excluded = Vec::new();
    let mut required = 0;
    for m_rung in rungs {
        let cells = required_cells(16384, true, m_rung);
        assert_eq!(cells.iter().collect::<BTreeSet<_>>().len(), cells.len());
        required += cells.len();
        for &bucket in &buckets {
            for topology in [Topology::Single, Topology::PackedHomogeneous] {
                let reachable = match topology {
                    Topology::Single => KvBucket::of(m_rung) <= bucket,
                    Topology::PackedHomogeneous => m_rung > 1,
                    Topology::PackedRagged => unreachable!(),
                };
                assert_eq!(cells.contains(&(bucket, topology)), reachable);
                if !reachable {
                    excluded.push((m_rung, bucket, topology));
                }
            }
            assert_eq!(
                cells.contains(&(bucket, Topology::PackedRagged)),
                m_rung > 1
            );
        }
    }
    assert_eq!(required, 156);
    assert_eq!(
        excluded,
        vec![
            (1, KvBucket::K1, Topology::PackedHomogeneous),
            (1, KvBucket::K4, Topology::PackedHomogeneous),
            (1, KvBucket::K8, Topology::PackedHomogeneous),
            (1, KvBucket::K16, Topology::PackedHomogeneous),
            (2048, KvBucket::K1, Topology::Single),
            (4096, KvBucket::K1, Topology::Single),
            (8192, KvBucket::K1, Topology::Single),
            (8192, KvBucket::K4, Topology::Single),
        ]
    );
}

#[test]
fn isolates_only_compatible_hd512_instructions_and_binds_hash() {
    let mut model = fixture(512, true, false);
    let original_insts = model.progs[0].insts.clone();
    let mut sections = Vec::new();
    apply(&mut model, &mut sections, &selection(), "sm90a", None).unwrap();
    assert_eq!(model.progs[0].insts, original_insts);
    let roles = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    let program = &roles.programs[0];
    assert_eq!(program.roles, [0, 6, 0]);
    assert_eq!(model.progs[0].gq_seg_ofs.len(), program.roles.len() + 1);
    let object = &roles.objects[&PREFILL_ATTENTION_HD512_WG32];
    assert_eq!(
        object.sha256.as_deref(),
        Some(plow_asset::decode_objects::image_sha256(b"cubin").as_str())
    );
    assert_eq!(object.attention.as_ref(), Some(&capability(false)));
}

#[test]
fn stays_inert_without_apply_and_rejects_incompatible_geometry() {
    let model = fixture(512, true, false);
    assert_eq!(model.to_blob(), model.to_blob());
    for mut model in [fixture(256, true, false), fixture(512, false, false)] {
        assert!(apply(&mut model, &mut Vec::new(), &selection(), "sm90a", None).is_err());
    }
    let mut model = fixture(512, true, false);
    model.tensors.last_mut().unwrap().bytes = 128;
    assert!(apply(&mut model, &mut Vec::new(), &selection(), "sm90a", None).is_err());
    let mut model = fixture(512, true, false);
    assert!(apply(&mut model, &mut Vec::new(), &selection(), "gfx950", None).is_err());
}

#[test]
fn output_object_is_explicit_inert_and_validated_before_mutation() {
    let directory = output_dir("selection");
    let output = directory.join("model.pkt");
    let mut model = fixture(512, true, false);
    let before = model.to_blob();
    let mut sections = Vec::new();
    assert!(!apply_output(&mut model, &mut sections, "sm90a", &output).unwrap());
    assert_eq!(model.to_blob(), before);
    assert!(sections.is_empty());

    std::fs::write(directory.join(OBJECT_FILE), b"not a cubin").unwrap();
    assert!(apply_output(&mut model, &mut sections, "sm90a", &output).is_err());
    assert_eq!(model.to_blob(), before);
    assert!(sections.is_empty());

    let mut stale = OBJECT_GLOBALS;
    stale[3].1 = 32;
    std::fs::write(directory.join(OBJECT_FILE), object_image(&stale)).unwrap();
    assert!(apply_output(&mut model, &mut sections, "sm90a", &output).is_err());
    assert_eq!(model.to_blob(), before);
    assert!(sections.is_empty());

    std::fs::write(directory.join(OBJECT_FILE), object_image(&OBJECT_GLOBALS)).unwrap();
    assert!(apply_output(&mut model, &mut sections, "sm90a", &output).unwrap());
    assert_eq!(sections.len(), 1);
    let metadata = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    assert_eq!(
        metadata.objects[&PREFILL_ATTENTION_HD512_WG32].file,
        OBJECT_FILE
    );
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn composes_with_existing_role_objects_and_preserves_their_segments() {
    let mut model = fixture(512, true, false);
    let program = &mut model.progs[0];
    let split = program
        .gq_stream
        .iter()
        .position(|entry| entry.inst != 0)
        .unwrap();
    for entry in program.stream.iter_mut().chain(&mut program.gq_stream) {
        entry.seg = u16::from(entry.inst != 0);
    }
    program.gq_seg_ofs = vec![0, split as u32, program.gq_stream.len() as u32];
    let mut sections = vec![SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: br#"{"version":1,"objects":{"1":{"abi":"fp8_gemm_tma128_v1","file":"existing.cubin"}},"programs":[{"index":0,"roles":[1,0]}]}"#.to_vec(),
    }];
    apply(&mut model, &mut sections, &selection(), "sm90a", None).unwrap();
    let metadata = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    let existing = &metadata.objects[&plow_asset::segment_roles::FP8_PREFILL_GEMM];
    assert_eq!(existing.abi, "fp8_gemm_tma128_v1");
    assert_eq!(existing.file, "existing.cubin");
    assert!(existing.sha256.is_none() && existing.attention.is_none());
    let record = metadata
        .programs
        .iter()
        .find(|program| program.index == 0)
        .unwrap();
    let roles_for_inst: Vec<_> = model.progs[0]
        .insts
        .iter()
        .enumerate()
        .map(|(inst, _)| {
            let segment = model.progs[0]
                .gq_stream
                .iter()
                .find(|entry| entry.inst as usize == inst)
                .unwrap()
                .seg as usize;
            record.roles[segment]
        })
        .collect();
    assert_eq!(roles_for_inst, [1, 6, 0, 0]);
}

#[test]
fn accepts_fused_output_and_rejects_unsafe_fused_contracts() {
    let mut model = fixture(512, true, true);
    let mut sections = Vec::new();
    apply(&mut model, &mut sections, &selection(), "sm90a", None).unwrap();
    let metadata = SegmentRoles::from_bytes(&sections[0].data).unwrap();
    assert_eq!(metadata.programs[0].roles, [0, 6, 0]);

    let mut short = fixture(512, true, true);
    let output = short.progs[0]
        .insts
        .iter()
        .find(|op| op.op == DevOp::FlashPrefill as u16)
        .unwrap()
        .t[5] as usize;
    short.tensors[output].bytes -= 1;
    assert!(apply(&mut short, &mut Vec::new(), &selection(), "sm90a", None).is_err());

    let mut split = fixture(512, true, true);
    split.progs[0]
        .insts
        .iter_mut()
        .find(|op| op.op == DevOp::FlashPrefill as u16)
        .unwrap()
        .i[7] = 2;
    assert!(apply(&mut split, &mut Vec::new(), &selection(), "sm90a", None).is_err());
}
