//! Library prefill attention routes (`PLOW_EMIT_ATTN_LIBRARY`, per rung): the packet declares,
//! per prefill program, which lone causal global `FlashPrefill(Fp8)` segments the library
//! object (`plow_fa4_hd512.aot` or `plow_fa4_hd256.aot` in the out dir, scripts/fa4_aot/build.py)
//! serves. The segments keep their native role as the
//! declared fallback; the runtime executes exactly this declaration.

use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::Model;
use plow_asset::segment_roles::{
    LibraryAbi, LibraryKvLayout, ProgramRoles, SegmentObject, SegmentRoles,
    LIBRARY_PREFILL_ATTENTION, SECTION,
};
use std::path::Path;

/// Library objects the emitter looks for in the out dir, in order (at most one may be present).
pub(crate) const FILES: [&str; 2] = ["plow_fa4_hd512.aot", "plow_fa4_hd256.aot"];

/// The object's ABI string, from its container manifest (`PLOWAOT1`, u32 length, JSON).
fn library_abi(image: &[u8]) -> Result<String, String> {
    if image.len() < 12 || &image[..8] != b"PLOWAOT1" {
        return Err("not a PLOWAOT1 kernel container".into());
    }
    let n = u32::from_le_bytes(image[8..12].try_into().expect("4 bytes")) as usize;
    let manifest: serde_json::Value =
        serde_json::from_slice(image.get(12..12 + n).ok_or("manifest past the end")?)
            .map_err(|e| format!("manifest: {e}"))?;
    let abi = manifest["abi"].as_str().ok_or("manifest names no abi")?;
    LibraryAbi::parse(abi)
        .map(|_| abi.to_string())
        .ok_or_else(|| format!("library ABI {abi:?} unsupported"))
}

/// The runtime's site contract (plowrt `lib_attention::sites`): causal global attention of the
/// library's geometry on a linear cache, fused output, no split. FP8 KV only through the hd512
/// head-major gather.
fn is_site(op: &packet::dev::DevInst, abi: &LibraryAbi) -> bool {
    let fp8 = op.op == DevOp::FlashPrefillFp8 as u16;
    (op.op == DevOp::FlashPrefill as u16
        || (fp8 && abi.head_dim == 512 && abi.layout == LibraryKvLayout::HeadMajor))
        && op.i[2] == abi.heads
        && abi.kv_heads.contains(&op.i[3])
        && op.i[5] == 0
        && op.i[6] == abi.head_dim
        && packet::dev::flash_nsplit(op.i[7]) == 1
        && op.j[0] > 0
        && op.j[0] % 64 == 0
        && op.j[1] == u32::MAX
        && op.t[2..6].iter().all(|&t| t != TENSOR_NONE)
        && (!fp8 || op.t[6..8].iter().all(|&t| t != TENSOR_NONE))
}

/// Attention the library's geometry claims: every such site of a requested rung is served.
fn is_claimed(op: &packet::dev::DevInst, abi: &LibraryAbi) -> bool {
    (op.op == DevOp::FlashPrefill as u16 || op.op == DevOp::FlashPrefillFp8 as u16)
        && op.i[6] == abi.head_dim
        && op.i[5] == 0
}

/// Declares the route on every prefill program whose rung enables it. Returns the segment count.
pub(crate) fn apply_output_object(
    model: &Model,
    sections: &mut [packet::devbuild::SectionData],
    output: &Path,
) -> Result<usize, String> {
    let prefill_count = packet::devbuild::decode_rung_lo(&model.prog_t);
    let rung = |index: usize| {
        crate::emit_config::rung::at(crate::emit_config::rung::Rung::prefill(model.prog_t[index]))
    };
    let requested: Vec<usize> = (0..prefill_count)
        .filter(|&index| rung(index).attn_library)
        .collect();
    if requested.is_empty() {
        return Ok(0);
    }
    let directory = output.parent().unwrap_or_else(|| Path::new("."));
    let present: Vec<&str> = FILES
        .into_iter()
        .filter(|f| directory.join(f).exists())
        .collect();
    let [file] = present[..] else {
        return Err(format!(
            "PLOW_EMIT_ATTN_LIBRARY needs exactly one of {FILES:?} in {}",
            directory.display()
        ));
    };
    let path = directory.join(file);
    let image = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let abi_string = library_abi(&image).map_err(|e| format!("{}: {e}", path.display()))?;
    let abi = LibraryAbi::parse(&abi_string).expect("checked");
    let position = sections
        .iter()
        .position(|section| section.name == SECTION)
        .ok_or("library attention needs the packet's segment roles")?;
    let mut metadata = SegmentRoles::from_bytes(&sections[position].data)?;
    if metadata.objects.contains_key(&LIBRARY_PREFILL_ATTENTION) {
        return Err("library attention object already declared".into());
    }
    let mut total = 0;
    for index in requested {
        let program = &model.progs[index];
        let mut library = Vec::new();
        for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
            let entries = &program.gq_stream[bounds[0] as usize..bounds[1] as usize];
            let Some(first) = entries.first().map(|e| e.inst as usize) else {
                continue;
            };
            if entries.iter().all(|e| e.inst as usize == first)
                && program
                    .stream
                    .iter()
                    .all(|e| (e.inst as usize == first) == (e.seg as usize == segment))
                && is_site(&program.insts[first], &abi)
            {
                library.push(segment as u32);
            }
        }
        let claimed = program
            .insts
            .iter()
            .filter(|op| is_claimed(op, &abi))
            .count();
        if claimed == 0 {
            continue;
        }
        // Every claimed site of a requested rung is served, or the request is refused.
        if library.len() != claimed {
            return Err(format!(
                "prefill program {index} (rows {}): {} of {claimed} hd{} global attention sites \
                 are lone library-compatible segments",
                model.prog_t[index],
                library.len(),
                abi.head_dim
            ));
        }
        let record = match metadata.programs.iter_mut().find(|p| p.index == index) {
            Some(record) => record,
            None => {
                metadata.programs.push(ProgramRoles {
                    index,
                    roles: vec![0; program.gq_seg_ofs.len() - 1],
                    library: Vec::new(),
                });
                metadata.programs.last_mut().expect("pushed")
            }
        };
        if record.roles.len() + 1 != program.gq_seg_ofs.len() {
            return Err(format!("prefill program {index}: role window coverage"));
        }
        total += library.len();
        record.library = library;
    }
    if total == 0 {
        return Err(format!(
            "PLOW_EMIT_ATTN_LIBRARY: no requested rung has hd{} global attention",
            abi.head_dim
        ));
    }
    metadata.objects.insert(
        LIBRARY_PREFILL_ATTENTION,
        SegmentObject {
            abi: abi_string,
            file: file.into(),
            sha256: Some(plow_asset::decode_objects::image_sha256(&image)),
            promote_k512: None,
            attention: None,
            gemm: None,
            decode_plan: None,
        },
    );
    metadata.validate_schema()?;
    sections[position].data = serde_json::to_vec(&metadata).map_err(|e| e.to_string())?;
    Ok(total)
}

#[cfg(test)]
mod tests {
    #[test]
    fn container_manifest_names_the_abi() {
        let abi = "lib_flash_prefill_v2:fa4:hd=256:heads=24:kv_heads=4:layout=ilv:dtype=bf16";
        let manifest = format!(r#"{{"abi":"{abi}","entries":[]}}"#);
        let mut image = b"PLOWAOT1".to_vec();
        image.extend_from_slice(&(manifest.len() as u32).to_le_bytes());
        image.extend_from_slice(manifest.as_bytes());
        assert_eq!(super::library_abi(&image).as_deref(), Ok(abi));
        assert!(super::library_abi(b"\x7fELF plow_fa4_caps:abi=3:dtype=0\0").is_err());
        image[12 + manifest.find("v2").unwrap() + 1] = b'1';
        assert!(super::library_abi(&image).is_err());
    }
}
