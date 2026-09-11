use super::*;

#[test]
fn segmented_grid_requires_no_ordinary_prefill_fallback() {
    use plow_asset::segment_roles::{GEMV_CTA512, INTERPRETER};

    assert!(uses_segmented_prefill(true, false, 2, &[]));
    assert!(uses_segmented_prefill(true, false, 1, &[GEMV_CTA512]));
    assert!(!uses_segmented_prefill(true, false, 1, &[INTERPRETER]));
    assert!(!uses_segmented_prefill(true, false, 1, &[]));
    assert!(!uses_segmented_prefill(false, false, 2, &[]));
    assert!(!uses_segmented_prefill(true, true, 2, &[GEMV_CTA512]));
}

#[test]
#[ignore = "CPU asset inspection; set TEST_PACKED_FP8_ASSETS to compiled H100 assets"]
fn fp8_packed_assets_match_packet_and_reject_bf16_objects() {
    let assets = PathBuf::from(std::env::var_os("TEST_PACKED_FP8_ASSETS").unwrap());
    let raw = std::fs::read(assets.join("model.pkt")).unwrap();
    let blob = DevBlob::parse(&raw).unwrap();
    let live = crate::memory::vmm::LiveKvLayout::manifest(&blob, &raw)
        .unwrap()
        .unwrap();
    let pack: plow_asset::packed_prefill::Manifest = serde_json::from_slice(
        blob.reserved_metadata(&raw, plow_asset::packed_prefill::SECTION)
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(live.version, 2);
    assert_eq!(pack.version, 2);
    assert!(!live.caches.is_empty());
    assert!(live.caches.iter().all(|c| c.scales.is_some()));
    blob.with_packet_view(|p| pack.validate(p, &live)).unwrap();
    assert_eq!(
        blob.decode_progs().iter().map(|p| p.t).collect::<Vec<_>>(),
        [1, 2, 4, 8, 16]
    );
    assert_eq!(
        blob.prefill_progs().iter().map(|p| p.t).collect::<Vec<_>>(),
        [128, 512, 1024]
    );
    let build: serde_json::Value =
        serde_json::from_slice(&std::fs::read(assets.join("build.json")).unwrap()).unwrap();
    let hash = build["pairing"]["hash"].as_str().unwrap();
    let hash = u64::from_str_radix(hash.trim_start_matches("0x"), 16).unwrap();
    for role in ["pfpackedseg", "pfpackedgemm"] {
        let name = format!("interp_sm90a_{role}");
        let image = std::fs::read(assets.join(format!("{name}_fp8kv.cubin"))).unwrap();
        pack.validate_object(|n| cubin::global_u32(&image, n))
            .unwrap();
        let info = cubin::inspect(&image).unwrap();
        assert_eq!(info.sm, 90);
        assert!(info
            .entries
            .contains(&format!("_Z{}{name}11PlowProgram", name.len())));
        let lo = cubin::global_u32(&image, &format!("plow_packet_hash_lo_{role}")).unwrap();
        let hi = cubin::global_u32(&image, &format!("plow_packet_hash_hi_{role}")).unwrap();
        assert_eq!((u64::from(hi) << 32) | u64::from(lo), hash);
        let bf16 = std::fs::read(assets.join(format!("{name}.cubin"))).unwrap();
        assert!(pack
            .validate_object(|n| cubin::global_u32(&bf16, n))
            .is_err());
    }
}

fn prefill_images() -> (InterpreterProfile, Vec<u8>, Vec<u8>) {
    let profile = interpreter_profile((9, 0)).unwrap();
    let ordinary = plow_asset::cubin::synthetic_elf(profile.prefill_symbol, &[], 90);
    let mixed = plow_asset::cubin::synthetic_elf(
        profile.prefill_symbol,
        &[(
            plow_asset::mixed_step::OBJECT_CAPABILITY,
            plow_asset::mixed_step::VERSION,
        )],
        90,
    );
    (profile, ordinary, mixed)
}

#[test]
fn selects_native_hopper_for_h100_and_h200() {
    let p = interpreter_profile((9, 0)).unwrap();
    assert_eq!(p.tag, "sm90a");
    assert_eq!(p.decode_file, "interp_sm90a.cubin");
    assert_eq!(p.decode_symbol, "_Z12interp_sm90a11PlowProgram");
}

#[test]
fn preserves_sm120_profile_and_rejects_unknown_arches() {
    let p = interpreter_profile((12, 0)).unwrap();
    assert_eq!(p.prefill_file, "interp_sm120_pf.cubin");
    assert_eq!(p.prefill_symbol, "_Z15interp_sm120_pf11PlowProgram");
    assert!(interpreter_profile((8, 9)).is_none());
    assert!(interpreter_profile((10, 0)).is_none());
}

#[test]
fn auxiliary_mixed_object_cannot_win_ordinary_prefill_discovery() {
    let (profile, ordinary, mixed) = prefill_images();
    assert_eq!(
        interp_candidate(&ordinary, &profile, 90, Role::Prefill).unwrap(),
        profile.prefill_symbol
    );
    assert!(interp_candidate(&mixed, &profile, 90, Role::Prefill)
        .unwrap_err()
        .contains("mixed-step auxiliary object"));
}

#[test]
fn auxiliary_packed_object_cannot_win_ordinary_prefill_discovery() {
    let (profile, ordinary, _) = prefill_images();
    let packed = plow_asset::cubin::synthetic_elf(
        profile.prefill_symbol,
        &[(
            plow_asset::packed_prefill::CAPABILITY,
            plow_asset::packed_prefill::CAPABILITY_VALUE,
        )],
        90,
    );
    assert_eq!(
        interp_candidate(&ordinary, &profile, 90, Role::Prefill).unwrap(),
        profile.prefill_symbol
    );
    assert!(interp_candidate(&packed, &profile, 90, Role::Prefill)
        .unwrap_err()
        .contains("packed-prefill auxiliary object"));
}

#[test]
fn embedded_mixed_object_is_skipped_for_ordinary_prefill() {
    use crate::asset::devblob::DevSection;

    let (profile, ordinary, mixed) = prefill_images();
    let mut raw = mixed.clone();
    raw.extend_from_slice(&ordinary);
    let blob = DevBlob {
        n_cu: 0,
        flags: 0,
        target: 0,
        tensors: Vec::new(),
        init: Vec::new(),
        kvrow: Vec::new(),
        progs: Vec::new(),
        sections: vec![
            DevSection {
                kind: packet::devbuild::SECT_CUBIN,
                name: "mixed".into(),
                offset: 0,
                size: mixed.len(),
            },
            DevSection {
                kind: packet::devbuild::SECT_CUBIN,
                name: "ordinary".into(),
                offset: mixed.len(),
                size: ordinary.len(),
            },
        ],
        gen: Vec::new(),
        tp: None,
        parent: None,
    };
    let mut rejected = Vec::new();
    let selected =
        embedded_interp_image(&blob, &raw, &profile, 90, Role::Prefill, &mut rejected).unwrap();
    assert_eq!(selected.image, ordinary);
    assert_eq!(selected.source, "embedded section 'ordinary'");
    assert!(rejected[0].contains("mixed-step auxiliary object"));
}

#[test]
fn filesystem_mixed_object_is_skipped_for_ordinary_prefill() {
    let (profile, ordinary, mixed) = prefill_images();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "plow-interp-discovery-{}-{unique}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(dir.join(profile.prefill_file), mixed).unwrap();
    std::fs::write(dir.join("ordinary.cubin"), &ordinary).unwrap();

    let mut rejected = Vec::new();
    let selected =
        filesystem_interp_image(&dir, &profile, 90, Role::Prefill, &mut rejected).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(selected.image, ordinary);
    assert_eq!(selected.source, "ordinary.cubin");
    assert!(rejected[0].contains("mixed-step auxiliary object"));
}
