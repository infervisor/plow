//! Shared `--block` extraction helpers: layer-range parsing + the `block.json`
//! descriptor `SECT_METADATA` section. Used by BOTH the dense and MLA emit paths.
//! Split out of `lib.rs` (module breakdown).

/// Parse a `--block` spec (`l` or `l..r`) into a bounds-checked half-open layer
/// range. Shared by the gemma and GLM emit paths.
pub(crate) fn parse_block(spec: &str, layers: usize) -> std::ops::Range<usize> {
    let r = if let Some((a, b)) = spec.split_once("..") {
        let lo: usize = a.trim().parse().expect("--block l..r: bad l");
        let hi: usize = b.trim().parse().expect("--block l..r: bad r");
        lo..hi
    } else {
        let l: usize = spec.trim().parse().expect("--block l: bad l");
        l..l + 1
    };
    assert!(
        r.start < r.end && r.end <= layers,
        "--block {r:?} out of range for a {layers}-layer model"
    );
    r
}

/// Serialize a block descriptor to pretty JSON, write a sibling `block.json` next to
/// `out`, and return the `SECT_METADATA` section that mirrors it into the blob.
/// Shared by the gemma and GLM `--block` emit paths.
pub(crate) fn write_block_descriptor(
    out: &str,
    desc: &plow_asset::BlockDescriptor,
) -> packet::devbuild::SectionData {
    let json = serde_json::to_vec_pretty(desc).expect("serialize block.json");
    let sib = std::path::Path::new(out).with_file_name("block.json");
    std::fs::write(&sib, &json).expect("write sibling block.json");
    packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: "block.json".into(),
        data: json,
    }
}
