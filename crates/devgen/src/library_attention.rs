//! Library prefill attention routes (`PLOW_EMIT_ATTN_LIBRARY`, per rung): the packet declares,
//! per prefill program, which lone global hd512 `FlashPrefill(Fp8)` segments the library
//! (`libplow_fa4_hd512.so` in the out dir) serves. The segments keep their native role as the
//! declared fallback; the runtime executes exactly this declaration.

use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::Model;
use plow_asset::segment_roles::{
    ProgramRoles, SegmentObject, SegmentRoles, LIBRARY_FA4_HD512_BF16_ABI,
    LIBRARY_FA4_HD512_F16_ABI, LIBRARY_PREFILL_ATTENTION, SECTION,
};
use std::path::Path;

pub(crate) const FILE: &str = "libplow_fa4_hd512.so";
/// `plow_fa4_caps` in runtime/nvidia/fa4_hd512.cpp, followed by the dtype digit.
const CAPS: &[u8] = b"plow_fa4_caps:abi=3:dtype=";

/// The library's ABI string, from the capability marker it embeds.
fn library_abi(image: &[u8]) -> Result<&'static str, String> {
    let at = image
        .windows(CAPS.len())
        .position(|w| w == CAPS)
        .ok_or_else(|| format!("{FILE} carries no plow_fa4 ABI 3 capability marker"))?;
    match image.get(at + CAPS.len()) {
        Some(b'0') => Ok(LIBRARY_FA4_HD512_BF16_ABI),
        Some(b'1') => Ok(LIBRARY_FA4_HD512_F16_ABI),
        _ => Err(format!("{FILE}: unknown plow_fa4 dtype")),
    }
}

/// The runtime's site contract (plowrt `lib_attention::sites`): causal global hd512, 16 query
/// heads over 1 or 2 KV heads, a linear head-major cache, fused output, no split.
fn is_site(op: &packet::dev::DevInst) -> bool {
    let fp8 = op.op == DevOp::FlashPrefillFp8 as u16;
    (fp8 || op.op == DevOp::FlashPrefill as u16)
        && op.i[2] == 16
        && matches!(op.i[3], 1 | 2)
        && op.i[5] == 0
        && op.i[6] == 512
        && packet::dev::flash_nsplit(op.i[7]) == 1
        && op.j[0] > 0
        && op.j[0] % 64 == 0
        && op.j[1] == u32::MAX
        && op.t[2..6].iter().all(|&t| t != TENSOR_NONE)
        && (!fp8 || op.t[6..8].iter().all(|&t| t != TENSOR_NONE))
}

fn is_hd512_flash(op: &packet::dev::DevInst) -> bool {
    (op.op == DevOp::FlashPrefill as u16 || op.op == DevOp::FlashPrefillFp8 as u16)
        && op.i[6] == 512
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
    let path = directory.join(FILE);
    let image = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let abi = library_abi(&image)?;
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
                && is_site(&program.insts[first])
            {
                library.push(segment as u32);
            }
        }
        let hd512 = program.insts.iter().filter(|op| is_hd512_flash(op)).count();
        if hd512 == 0 {
            continue;
        }
        // Every hd512 site of a requested rung is served, or the request is refused.
        if library.len() != hd512 {
            return Err(format!(
                "prefill program {index} (rows {}): {} of {hd512} hd512 attention sites are lone \
                 library-compatible segments",
                model.prog_t[index],
                library.len()
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
        return Err("PLOW_EMIT_ATTN_LIBRARY: no requested rung has hd512 attention".into());
    }
    metadata.objects.insert(
        LIBRARY_PREFILL_ATTENTION,
        SegmentObject {
            abi: abi.into(),
            file: FILE.into(),
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
    fn caps_marker_names_the_dtype() {
        let mut image = b"\x7fELF....".to_vec();
        image.extend_from_slice(b"plow_fa4_caps:abi=3:dtype=1\0");
        assert_eq!(
            super::library_abi(&image),
            Ok(super::LIBRARY_FA4_HD512_F16_ABI)
        );
        assert!(super::library_abi(b"plow_fa4_caps:abi=2:dtype=0").is_err());
    }
}
