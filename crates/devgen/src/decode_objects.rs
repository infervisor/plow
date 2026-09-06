use packet::dev::{DevProgram, SE_FINE, SE_XCTR};
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use plow_asset::decode_coverage::DenseBf16;
use plow_asset::decode_objects::{DecodeObjects, SECTION};
use std::path::Path;

fn validate(model: &Model, metadata: &DecodeObjects, arch: &str) -> Result<(), String> {
    let profile = match arch {
        "sm_90a" => "sm90a",
        "sm_120" => "sm120",
        _ => return Err("unsupported CUDA profile".into()),
    };
    let start = model
        .prog_t
        .iter()
        .position(|&rows| rows == 1)
        .ok_or("decode ladder must start at one row")?;
    let programs: Vec<_> = model
        .prog_t
        .iter()
        .enumerate()
        .skip(start)
        .map(|(i, &rows)| (i, rows))
        .collect();
    metadata.validate(&programs, model.n_cu, std::mem::size_of::<DevProgram>())?;
    for (index, _) in programs {
        let p = &model.progs[index];
        if p.n_cu != model.n_cu
            || p.hier_base != 0
            || p.l2_domains != 0
            || p.gq_seg_ofs.len() != 2
            || p.gq_seg_ofs.first() != Some(&0)
            || p.gq_seg_ofs.last().copied() != Some(p.gq_stream.len() as u32)
            || p.gq_stream.is_empty()
            || p.stream
                .iter()
                .chain(&p.gq_stream)
                .any(|e| e.seg != 0 || e.flags & (SE_FINE | SE_XCTR) != 0)
        {
            return Err("only plain cooperative decode programs can be bound".into());
        }
    }
    for spec in metadata.objects.values() {
        if spec.profile != profile
            || spec.entry != format!("_Z12interp_{profile}11PlowProgram")
            || spec.threads != 256
        {
            return Err("object does not match the CUDA broad-interpreter ABI".into());
        }
    }
    Ok(())
}

pub(crate) fn append(
    model: &Model,
    sections: &mut Vec<SectionData>,
    path: Option<&Path>,
    arch: &str,
    output: &Path,
) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    let raw = std::fs::read(path).map_err(|e| e.to_string())?;
    let metadata: DecodeObjects = serde_json::from_slice(&raw).map_err(|e| e.to_string())?;
    append_metadata(model, sections, &metadata, arch, output)
}

pub(crate) fn append_metadata(
    model: &Model,
    sections: &mut Vec<SectionData>,
    metadata: &DecodeObjects,
    arch: &str,
    output: &Path,
) -> Result<(), String> {
    validate(model, metadata, arch)?;
    if sections
        .iter()
        .any(|s| s.kind == SECT_METADATA && s.name == SECTION)
    {
        return Err("duplicate decode object metadata".into());
    }
    let dir = output.parent().ok_or("output directory missing")?;
    for (&id, spec) in &metadata.objects {
        let path = dir.join(&spec.file);
        let size = std::fs::metadata(&path).map_err(|e| e.to_string())?.len();
        if size == 0 || size > 64 * 1024 * 1024 {
            return Err("invalid object size".into());
        }
        let image = std::fs::read(path).map_err(|e| e.to_string())?;
        if !spec.matches_image(&image) {
            return Err("object SHA256 mismatch".into());
        }
        let info = plow_asset::cubin::inspect(&image).ok_or("invalid object ELF")?;
        let sm = match arch {
            "sm_90a" => 90,
            "sm_120" => 120,
            _ => unreachable!(),
        };
        if info.sm != sm || !info.entries.contains(&spec.entry) {
            return Err("object ISA/entry mismatch".into());
        }
        let coverage = DenseBf16::from_image(&image)?;
        if coverage.0[5] != spec.arena_bytes {
            return Err("object arena differs from declared resources".into());
        }
        for binding in metadata.programs.iter().filter(|p| p.object == id) {
            let splitk = plow_asset::cubin::global_u32(&image, "plow_gemm_splitk_abi");
            plow_asset::program::with_model(model, |packet| {
                coverage.program(packet, binding.index, splitk)
            })?;
        }
    }
    sections.push(SectionData {
        kind: SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::DevOp;
    use packet::devbuild::Builder;
    use plow_asset::decode_objects::{image_sha256, DecodeObject, DecodeProgramObject};
    use std::collections::BTreeMap;
    fn capability_elf() -> Vec<u8> {
        let names = [
            "_Z12interp_sm90a11PlowProgram",
            "plow_decode_bf16_abi",
            "plow_decode_gf256",
            "plow_decode_gf512",
            "plow_decode_staging_bytes",
            "plow_gemv_mm_cap",
            "plow_arena_bytes",
        ];
        let constants = [1u32, 2, 8, 16384, 16, 16384];
        let mut strings = vec![0u8];
        let mut symbols = Vec::new();
        for (index, name) in names.iter().enumerate() {
            let mut symbol = [0u8; 24];
            symbol[..4].copy_from_slice(&(strings.len() as u32).to_le_bytes());
            symbol[4] = if index == 0 { 0x12 } else { 0x11 };
            symbol[6..8].copy_from_slice(&3u16.to_le_bytes());
            symbol[8..16].copy_from_slice(&((index.saturating_sub(1) * 4) as u64).to_le_bytes());
            symbol[16..24].copy_from_slice(&4u64.to_le_bytes());
            strings.extend_from_slice(name.as_bytes());
            strings.push(0);
            symbols.extend_from_slice(&symbol);
        }
        let mut image = vec![0u8; 320];
        image[..6].copy_from_slice(b"\x7fELF\x02\x01");
        image[40..48].copy_from_slice(&64u64.to_le_bytes());
        image[48..52].copy_from_slice(&(90u32 << 8).to_le_bytes());
        image[58..60].copy_from_slice(&64u16.to_le_bytes());
        image[60..62].copy_from_slice(&4u16.to_le_bytes());
        for (index, kind, data) in [
            (1usize, 2u32, symbols),
            (2, 3, strings),
            (
                3,
                1,
                constants.into_iter().flat_map(u32::to_le_bytes).collect(),
            ),
        ] {
            let header = 64 + index * 64;
            image[header + 4..header + 8].copy_from_slice(&kind.to_le_bytes());
            let offset = image.len() as u64;
            image[header + 24..header + 32].copy_from_slice(&offset.to_le_bytes());
            image[header + 32..header + 40].copy_from_slice(&(data.len() as u64).to_le_bytes());
            image.extend_from_slice(&data);
        }
        image[168..172].copy_from_slice(&2u32.to_le_bytes());
        image[184..192].copy_from_slice(&24u64.to_le_bytes());
        image
    }
    fn model() -> Model {
        let progs = [128, 1, 2]
            .iter()
            .map(|_| {
                let mut b = Builder::new(7);
                b.force_uniseg();
                b.emit(DevOp::Nop, b.all(), &[], |_| {});
                b.finish()
            })
            .collect();
        Model {
            n_cu: 7,
            target: 0,
            tensors: vec![],
            progs,
            kv_row_insts: vec![],
            prog_t: vec![128, 1, 2],
            gen: vec![],
        }
    }
    fn metadata() -> DecodeObjects {
        DecodeObjects {
            version: 1,
            kernarg_bytes: std::mem::size_of::<DevProgram>(),
            objects: BTreeMap::from([(
                0,
                DecodeObject {
                    file: "old.cubin".into(),
                    sha256: image_sha256(&capability_elf()),
                    profile: "sm90a".into(),
                    entry: "_Z12interp_sm90a11PlowProgram".into(),
                    threads: 256,
                    arena_bytes: 16384,
                    grid: 7,
                },
            )]),
            programs: vec![
                DecodeProgramObject {
                    index: 1,
                    rows: 1,
                    object: 0,
                },
                DecodeProgramObject {
                    index: 2,
                    rows: 2,
                    object: 0,
                },
            ],
        }
    }
    #[test]
    fn absent_configuration_preserves_packet_bytes_without_io() {
        let m = model();
        let old = m.to_blob();
        let mut sections = vec![];
        append(
            &m,
            &mut sections,
            None,
            "unsupported",
            Path::new("/missing/output"),
        )
        .unwrap();
        assert!(sections.is_empty());
        assert_eq!(m.to_blob(), old);
    }
    #[test]
    fn target_geometry_coverage_and_coarse_scheduler_are_required() {
        let m = model();
        let spec = metadata();
        assert!(validate(&m, &spec, "sm_90a").is_ok());
        assert!(validate(&m, &spec, "gfx950").is_err());
        assert!(validate(&m, &spec, "sm_120").is_err());
        let mut other = spec.clone();
        let obj = other.objects.get_mut(&0).unwrap();
        obj.profile = "sm120".into();
        obj.entry = "_Z12interp_sm12011PlowProgram".into();
        assert!(validate(&m, &other, "sm_120").is_ok());
        let mut missing = spec.clone();
        missing.programs.pop();
        assert!(validate(&m, &missing, "sm_90a").is_err());
        let mut segmented = model();
        segmented.progs[1].gq_seg_ofs.push(1);
        assert!(validate(&segmented, &spec, "sm_90a").is_err());
        let mut flagged = model();
        flagged.progs[1].stream[0].flags |= SE_XCTR;
        assert!(validate(&flagged, &spec, "sm_90a").is_err());
    }
    #[test]
    fn emission_binds_existing_object_content_and_rejects_change() {
        let dir = std::env::temp_dir().join(format!("plow-decode-objects-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("objects.json");
        std::fs::write(&path, serde_json::to_vec(&metadata()).unwrap()).unwrap();
        std::fs::write(dir.join("old.cubin"), capability_elf()).unwrap();
        let m = model();
        let old = m.to_blob();
        let mut sections = vec![];
        append(
            &m,
            &mut sections,
            Some(&path),
            "sm_90a",
            &dir.join("model.pkt"),
        )
        .unwrap();
        assert_eq!(m.to_blob(), old);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].name, SECTION);
        assert_eq!(
            serde_json::from_slice::<DecodeObjects>(&sections[0].data)
                .unwrap()
                .programs
                .len(),
            2
        );
        assert!(append(
            &m,
            &mut sections,
            Some(&path),
            "sm_90a",
            &dir.join("model.pkt")
        )
        .is_err());
        let mut unsupported = model();
        unsupported.progs[1].insts[0].op = DevOp::FlashDecodeFp8 as u16;
        assert!(append(
            &unsupported,
            &mut vec![],
            Some(&path),
            "sm_90a",
            &dir.join("model.pkt")
        )
        .is_err());
        let mut pruned = capability_elf();
        let end = pruned.len();
        pruned[end - 24..end - 20].fill(0);
        let mut matching_hash = metadata();
        matching_hash.objects.get_mut(&0).unwrap().sha256 = image_sha256(&pruned);
        std::fs::write(&path, serde_json::to_vec(&matching_hash).unwrap()).unwrap();
        std::fs::write(dir.join("old.cubin"), pruned).unwrap();
        assert!(append(
            &m,
            &mut vec![],
            Some(&path),
            "sm_90a",
            &dir.join("model.pkt")
        )
        .is_err());
        std::fs::write(dir.join("old.cubin"), b"changed").unwrap();
        assert!(append(
            &m,
            &mut vec![],
            Some(&path),
            "sm_90a",
            &dir.join("model.pkt")
        )
        .is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
