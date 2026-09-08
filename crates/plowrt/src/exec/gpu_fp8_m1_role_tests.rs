use super::*;
fn fixture() -> packet::devbuild::Model {
    use packet::dev::{DevOp, TENSOR_NONE};
    use packet::devbuild::{Builder, Model};
    let mut b = Builder::new(132);
    b.force_uniseg();
    let out = b.tensor("act.out", 20480);
    let a = b.tensor("act.fp8", 5120);
    let w = b.tensor("fp8/weight", 10240 * 5120);
    let scale = b.tensor("act.scale", 4);
    let ws = b.tensor("fp8/weight_scale", 10240 * 4);
    let norm = b.tensor("act.norm", 10240);
    let input = b.tensor("act.input", 10240);
    let map = b.tensor_gen(
        "tmap.weight",
        128,
        packet::rope::GenTensor::tmap_e4m3(w, 10240, 5120, 64),
    );
    let z = b.emit(DevOp::Nop, vec![0], &[], |_| {});
    let n = b.emit(DevOp::QwenRmsNorm, vec![0], &[z], |d| {
        d.t[0] = norm;
        d.t[1] = input;
        d.i[0] = 1;
        d.i[1] = 5120;
    });
    let q = b.emit(DevOp::QuantFp8, vec![0], &[n], |d| {
        d.t = [
            a,
            norm,
            scale,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
        ];
        d.i[0] = 1;
        d.i[1] = 5120;
    });
    let g = b.emit(DevOp::GemmFp8, b.all(), &[q], |d| {
        d.t[..5].copy_from_slice(&[out, a, w, scale, ws]);
        d.i = [1, 10240, 5120, 0, 0, 0, 0, map];
    });
    b.emit(DevOp::Nop, vec![0], &[g], |_| {});
    let gen = b.gen_tensors();
    let p = b.finish();
    Model {
        n_cu: 132,
        target: 0,
        tensors: p.tensors.clone(),
        progs: vec![p],
        kv_row_insts: vec![],
        prog_t: vec![1],
        gen,
    }
}

fn raw() -> Vec<u8> {
    let mut m = fixture();
    let p = &mut m.progs[0];
    for e in p.stream.iter_mut().chain(&mut p.gq_stream) {
        e.seg = if e.inst < 3 {
            0
        } else if e.inst == 3 {
            1
        } else {
            2
        };
    }
    p.gq_seg_ofs = vec![0, 3, 135, 136];
    m.to_blob_v6(&[packet::devbuild::SectionData{kind:packet::devbuild::SECT_METADATA,name:"segment_roles.json".into(),data:format!(r#"{{"version":1,"objects":{{"4":{{"abi":"fp8_gemm_m1_tma_v1","file":"role.cubin","sha256":"{}","promote_k512":1}}}},"programs":[{{"index":0,"roles":[0,4,0]}}]}}"#,"a".repeat(64)).into_bytes()}])
}
fn parse(raw: &[u8]) -> Result<SegmentRoles> {
    let b = DevBlob::parse(raw)?;
    segment_role_metadata(&b, raw)?
        .ok_or_else(|| RuntimeError::Rejected("missing role metadata".into()))
}
#[test]
fn complete_role4_metadata_and_duplicate_rejection() {
    let raw = raw();
    assert!(parse(&raw).is_ok());
    let b = DevBlob::parse(&raw).unwrap();
    let bytes = b
        .section_data_named(&raw, packet::devbuild::SECT_METADATA, "segment_roles.json")
        .unwrap();
    let mut v: serde_json::Value = serde_json::from_slice(bytes).unwrap();
    for field in ["sha256", "promote_k512"] {
        let mut bad = v.clone();
        bad["objects"]["4"].as_object_mut().unwrap().remove(field);
        assert!(SegmentRoles::parse(&serde_json::to_vec(&bad).unwrap(), &b).is_err());
    }
    for (field, value) in [
        ("file", serde_json::json!("../bad")),
        ("sha256", serde_json::json!("a")),
        ("promote_k512", serde_json::json!(2)),
        ("abi", serde_json::json!("gemv_sm90_cta512_v1")),
    ] {
        let mut bad = v.clone();
        bad["objects"]["4"][field] = value;
        assert!(SegmentRoles::parse(&serde_json::to_vec(&bad).unwrap(), &b).is_err());
    }
    let obj = serde_json::to_string(&v["objects"]["4"]).unwrap();
    for key in ["4", "04"] {
        let dup = format!(
            r#"{{"version":1,"objects":{{"4":{obj},"{key}":{obj}}},"programs":[{{"index":0,"roles":[0,4,0]}}]}}"#
        );
        assert!(SegmentRoles::parse(dup.as_bytes(), &b).is_err());
    }
    v["programs"][0]["roles"] = serde_json::json!([0, 4]);
    assert!(SegmentRoles::parse(&serde_json::to_vec(&v).unwrap(), &b).is_err());
}
#[test]
fn image_hash_and_isa_reject_without_device() {
    let object = SegmentObject {
        abi: plow_asset::fp8_m1_role::ABI.into(),
        file: "role.cubin".into(),
        sha256: Some("a".repeat(64)),
        promote_k512: Some(1),
        attention: None,
    };
    assert!(check_image(&[0; 128], &object, "sm90a").is_err());
}

#[test]
fn checkpoint_dtype_shape_contract() {
    use safetensors::Dtype::*;
    let shape = [10240, 5120];
    assert!(checkpoint_fields(
        Some(F8_E4M3),
        Some(&shape),
        Some(F32),
        Some(&shape[..1]),
        shape
    )
    .is_ok());
    for dtype in [BF16, F16, U8, F32] {
        assert!(checkpoint_fields(
            Some(dtype),
            Some(&shape),
            Some(F32),
            Some(&shape[..1]),
            shape
        )
        .is_err());
    }
    assert!(checkpoint_fields(
        Some(F8_E4M3),
        Some(&[5120, 10240]),
        Some(F32),
        Some(&shape[..1]),
        shape
    )
    .is_err());
    assert!(checkpoint_fields(
        Some(F8_E4M3),
        Some(&shape),
        Some(BF16),
        Some(&shape[..1]),
        shape
    )
    .is_err());
    assert!(checkpoint_fields(Some(F8_E4M3), Some(&shape), Some(F32), Some(&[1]), shape).is_err());
}
#[test]
#[ignore = "CPU frozen actual object/packet proof: TEST_FP8_ROLE_IMAGE and TEST_FP8_ROLE_PACKET"]
fn actual_frozen_role_image_and_packet() {
    let path = std::env::var("TEST_FP8_ROLE_IMAGE").unwrap();
    let image = std::fs::read(path).unwrap();
    let mut object = SegmentObject {
        abi: plow_asset::fp8_m1_role::ABI.into(),
        file: "role.cubin".into(),
        sha256: Some(plow_asset::decode_objects::image_sha256(&image)),
        promote_k512: Some(1),
        attention: None,
    };
    check_image(&image, &object, "sm90a").unwrap();
    assert!(check_image(&image, &object, "sm120").is_err());
    object.sha256 = Some("a".repeat(64));
    assert!(check_image(&image, &object, "sm90a").is_err());
    let mut wrong = image.clone();
    wrong[49] = 120;
    object.sha256 = Some(plow_asset::decode_objects::image_sha256(&wrong));
    assert!(check_image(&wrong, &object, "sm90a").is_err());
    let packet = std::fs::read(std::env::var("TEST_FP8_ROLE_PACKET").unwrap()).unwrap();
    let b = DevBlob::parse(&packet).unwrap();
    let roles = parse(&packet).unwrap();
    assert_eq!(roles.programs.last().unwrap().roles, [0, 4, 0]);
    let mut malformed = b;
    malformed
        .gen
        .iter_mut()
        .find(|g| g.tensor == 79)
        .unwrap()
        .aux = 16;
    let bytes = malformed
        .section_data_named(
            &packet,
            packet::devbuild::SECT_METADATA,
            "segment_roles.json",
        )
        .unwrap();
    assert!(SegmentRoles::parse(bytes, &malformed).is_err());
}

#[test]
fn reserved_role_section_kind_count_range_and_absence() {
    let raw = raw();
    let mut b = DevBlob::parse(&raw).unwrap();
    assert!(segment_role_metadata(&b, &raw).unwrap().is_some());
    let ix = b
        .sections
        .iter()
        .position(|s| s.name == plow_asset::segment_roles::SECTION)
        .unwrap();
    let original = || {
        let mut packet = DevBlob::parse(&raw).unwrap();
        packet.sections.remove(ix)
    };
    b.sections[ix].kind = 123;
    assert!(segment_role_metadata(&b, &raw).is_err());
    b.sections[ix] = original();
    b.sections.push(original());
    assert!(segment_role_metadata(&b, &raw).is_err());
    b.sections.last_mut().unwrap().kind = 123;
    assert!(segment_role_metadata(&b, &raw).is_err());
    b.sections.pop();
    b.sections[ix].offset = usize::MAX;
    assert!(segment_role_metadata(&b, &raw).is_err());
    b.sections[ix].offset = raw.len();
    assert!(segment_role_metadata(&b, &raw).is_err());
    b.sections.remove(ix);
    assert!(segment_role_metadata(&b, &raw).unwrap().is_none());
}
