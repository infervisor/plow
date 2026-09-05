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
