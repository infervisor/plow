use super::*;

/// `PLOW_TEST_BLOB=<model.pkt>` of a Gemma-4 H100 BF16 recipe: the MoE routes the 26B recipe
/// pinned serve-side (`PLOW_MOE_PF_LT=1`, `PLOW_MOE_DEC_LT=4`) are the packet defaults; a dense
/// packet declares neither segment kind.
#[test]
#[ignore = "set PLOW_TEST_BLOB"]
fn recipe_moe_routes_are_the_packet_defaults() {
    let path = std::env::var("PLOW_TEST_BLOB").unwrap();
    let raw = std::fs::read(&path).unwrap();
    let blob = DevBlob::parse(&raw).unwrap();
    let roles = segment_role_metadata(&blob, &raw).unwrap();
    let declares = |role: u8| {
        roles
            .as_ref()
            .is_some_and(|r| r.programs.iter().any(|p| p.roles.contains(&role)))
    };
    let prefill = declares(plow_asset::segment_roles::MOE_PREFILL_CUBLASLT);
    let decode_min = moe_lt_decode_packet_min(&blob, roles.as_ref());
    eprintln!("{path}: MOE_PREFILL_CUBLASLT {prefill}, MOE_DECODE_CUBLASLT from {decode_min:?}");
    assert_eq!(decode_min.is_some(), prefill);
    assert!(decode_min.is_none_or(|rows| rows == 4));
}

/// `PLOW_TEST_BLOB=<model.pkt>`: every prefill program's declared segment roles pass the load-time
/// validator, and each `MOE_PREFILL_CUBLASLT` segment is one the route takes (CPU only; the check
/// `load_prefill` runs before any launch). Prints the router-prologue segment count.
#[test]
#[ignore = "set PLOW_TEST_BLOB"]
fn packet_moe_prefill_segments_load() {
    let path = std::env::var("PLOW_TEST_BLOB").unwrap();
    let raw = std::fs::read(&path).unwrap();
    let blob = DevBlob::parse(&raw).unwrap();
    let roles = segment_role_metadata(&blob, &raw)
        .unwrap()
        .expect("segment roles");
    let (mut routed, mut prologues) = (0, 0);
    for (index, g) in blob.prefill_progs().iter().enumerate() {
        let Some(program) = roles.program(index) else {
            continue;
        };
        let selected =
            packet_role_segments_with(g, &program.roles, &blob.tensors, &roles.objects).unwrap();
        let segments = moe_lt::segments(g, &blob.tensors, &selected).unwrap();
        routed += segments.iter().flatten().count();
        prologues += moe_lt::prologue_count(&segments);
    }
    eprintln!("{path}: {routed} MoE prefill segments routed, {prologues} with the router prologue");
    assert!(routed > 0);
}
