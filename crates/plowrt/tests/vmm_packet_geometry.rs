use plowrt::memory::vmm::LiveKvLayout;

#[test]
#[ignore = "CPU-only actual packet check; set PLOW_VMM_PACKET_TEST to a blob path"]
fn live_geometry_actual_packet() {
    let path = std::env::var("PLOW_VMM_PACKET_TEST").expect("packet path");
    let bytes = std::fs::read(&path).unwrap();
    let blob = plowrt::asset::devblob::DevBlob::parse(&bytes).unwrap();
    let layout = LiveKvLayout::from_blob(&blob).unwrap();
    eprintln!(
        "{}: {:?}; full tensor pairs {:?}",
        path, layout.geometry, layout.full_tensors
    );
}

#[test]
#[ignore = "CPU-only batched TMA packet check; set PLOW_VMM_PACKET_TEST to a blob path"]
fn live_geometry_accepts_actual_batched_tma_packet() {
    let path = std::env::var("PLOW_VMM_PACKET_TEST").expect("packet path");
    let bytes = std::fs::read(&path).unwrap();
    let blob = plowrt::asset::devblob::DevBlob::parse(&bytes).unwrap();
    assert!(blob.decode_prog().unwrap().t > 1);
    assert!(blob
        .prefill_progs()
        .iter()
        .flat_map(|p| &p.insts)
        .any(|inst| {
            inst.op == packet::dev::DevOp::FlashPrefill as u16
                && blob.gen.iter().any(|g| {
                    g.kind == packet::rope::GEN_TMAP_KV_PAIR && g.tensor == u32::from(inst.t[7])
                })
        }));
    let layout = LiveKvLayout::from_blob(&blob).unwrap();
    assert_eq!(layout.geometry.batch, blob.decode_prog().unwrap().t);
    eprintln!(
        "{path}: {:?}; full pairs {:?}",
        layout.geometry, layout.full_tensors
    );
}

#[test]
#[ignore = "CPU-only compiled packed-request contract; TEST_PACKED_PREFILL_PACKET"]
fn packed_prefill_compiled_packet_contract() {
    use plow_asset::packed_prefill::{Manifest, SECTION};
    let path = std::env::var("TEST_PACKED_PREFILL_PACKET").unwrap();
    let bytes = std::fs::read(path).unwrap();
    let blob = plowrt::asset::devblob::DevBlob::parse(&bytes).unwrap();
    let live = LiveKvLayout::manifest(&blob, &bytes).unwrap().unwrap();
    let data = blob
        .section_data_named(&bytes, packet::devbuild::SECT_METADATA, SECTION)
        .unwrap();
    let metadata: Manifest = serde_json::from_slice(data).unwrap();
    blob.with_packet_view(|p| metadata.validate(p, &live))
        .unwrap();
    for kind in 0..4 {
        let mut bad = metadata.clone();
        match kind {
            0 => bad.version = 0,
            1 => bad.request = bad.slot,
            2 => {
                bad.programs.pop();
            }
            _ => bad.programs[0].push('0'),
        }
        assert!(blob.with_packet_view(|p| bad.validate(p, &live)).is_err());
    }
    if !metadata.maps.is_empty() {
        let mut bad = metadata.clone();
        bad.maps[0].slots = metadata.slot;
        assert!(blob.with_packet_view(|p| bad.validate(p, &live)).is_err());
        let mut bad = metadata.clone();
        bad.maps.clear();
        assert!(blob.with_packet_view(|p| bad.validate(p, &live)).is_err());
    }
    blob.with_packet_view(|packet| {
        for case in 0..6 {
            let mut programs = packet.programs.to_vec();
            let mut insts = programs[0].insts.to_vec();
            let mut stream = programs[0].gq_stream.to_vec();
            let pc = insts
                .iter()
                .position(|d| d.op == packet::dev::DevOp::FlashPrefill as u16)
                .unwrap();
            match case {
                0 => insts[pc].i[7] = u32::MAX,
                1 => insts[pc].t[0] = insts[pc].t[2],
                2 => {
                    let merge = insts
                        .iter_mut()
                        .find(|d| d.op == packet::dev::DevOp::FlashMerge as u16)
                        .unwrap();
                    merge.i[2] += 1;
                }
                3 => {
                    let e = stream
                        .iter_mut()
                        .find(|e| e.inst as usize == pc && e.slice == 1)
                        .unwrap();
                    e.slice = 0;
                }
                4 => insts[pc].op = packet::dev::DevOp::FlashPrefillFp8 as u16,
                _ => insts[pc].op = packet::dev::DevOp::KdaConv as u16,
            }
            programs[0].insts = &insts;
            programs[0].gq_stream = &stream;
            let altered = plow_asset::program::Packet {
                programs: &programs,
                ..*packet
            };
            let mut altered_live = live.clone();
            let mut altered_metadata = metadata.clone();
            let digest = plow_asset::live_kv::program_digest(&programs[0]);
            altered_live.programs[0] = digest.clone();
            altered_metadata.programs[0] = digest;
            assert!(
                altered_metadata.validate(&altered, &altered_live).is_err(),
                "forged valid identity case {case}"
            );
        }
    });
    if let Ok(control) = std::env::var("TEST_PACKED_PREFILL_CONTROL") {
        let baseline =
            plowrt::asset::devblob::DevBlob::parse(&std::fs::read(control).unwrap()).unwrap();
        assert_eq!(blob.progs.len(), baseline.progs.len());
        for (candidate, original) in blob.progs.iter().zip(&baseline.progs) {
            assert_eq!(
                candidate.insts, original.insts,
                "original instruction operands/counters"
            );
            assert_eq!(
                candidate
                    .waits
                    .iter()
                    .map(|w| (w.id, w.threshold))
                    .collect::<Vec<_>>(),
                original
                    .waits
                    .iter()
                    .map(|w| (w.id, w.threshold))
                    .collect::<Vec<_>>(),
                "original waits"
            );
            assert_eq!(
                candidate.succs, original.succs,
                "original successor counters"
            );
        }
    }
    for p in blob.prefill_progs() {
        p.check_gq_topological().unwrap();
    }
    eprintln!(
        "packed contract PASS {}buckets {}descriptor tables",
        blob.prefill_progs().len(),
        metadata.maps.len()
    );
}
