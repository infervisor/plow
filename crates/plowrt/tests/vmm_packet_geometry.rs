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
#[ignore = "CPU-only negative packet check; set PLOW_VMM_PACKET_TEST to a batched TMA blob"]
fn live_geometry_rejects_actual_batched_tma_packet() {
    let path = std::env::var("PLOW_VMM_PACKET_TEST").expect("packet path");
    let bytes = std::fs::read(&path).unwrap();
    let blob = plowrt::asset::devblob::DevBlob::parse(&bytes).unwrap();
    let error = LiveKvLayout::from_blob(&blob)
        .err()
        .expect("unsupported batched TMA");
    assert!(
        error.to_string().contains("per-slot descriptors"),
        "{error}"
    );
    eprintln!("{path}: {error}");
}
