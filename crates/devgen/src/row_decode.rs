use packet::dev::DevProgram;
use packet::devbuild::{Model, SectionData, SECT_METADATA};
use plow_asset::decode_objects::{DecodeObject, DecodeObjects, DecodeProgramObject};
use std::collections::BTreeMap;
use std::path::Path;

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
    let object: DecodeObject =
        serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let profile = match arch {
        "sm_90a" => "sm90a",
        _ => return Err("row decode requires Hopper".into()),
    };
    object.validate(model.n_cu)?;
    if object.profile != profile
        || object.entry != format!("_Z12interp_{profile}11PlowProgram")
        || object.threads != 256
    {
        return Err("row-decode object ABI mismatch".into());
    }
    let metadata = plow_asset::program::with_model(model, |packet| {
        let metadata = DecodeObjects {
            version: 1,
            kernarg_bytes: std::mem::size_of::<DevProgram>(),
            objects: BTreeMap::from([(0, object.clone())]),
            programs: plow_asset::row_decode::programs(packet)
                .into_iter()
                .map(|(index, rows)| DecodeProgramObject {
                    index,
                    rows,
                    object: 0,
                })
                .collect(),
        };
        plow_asset::row_decode::validate(packet, &metadata, std::mem::size_of::<DevProgram>())?;
        Ok::<_, String>(metadata)
    })?;
    let source = path
        .parent()
        .ok_or("row-decode object directory missing")?
        .join(&object.file);
    let image = std::fs::read(&source).map_err(|e| e.to_string())?;
    let elf = plow_asset::cubin::inspect(&image).ok_or("invalid row-decode object")?;
    if !object.matches_image(&image)
        || elf.sm != 90
        || !elf.entries.contains(&object.entry)
        || plow_asset::cubin::global_u32(&image, plow_asset::row_decode::CAPABILITY) != Some(1)
        || plow_asset::cubin::global_u32(&image, "plow_arena_bytes") != Some(object.arena_bytes)
    {
        return Err("row-decode object identity or capabilities mismatch".into());
    }
    let target = output
        .parent()
        .ok_or("row-decode output directory missing")?
        .join(&object.file);
    if source != target {
        std::fs::write(target, image).map_err(|e| e.to_string())?;
    }
    sections.push(SectionData {
        kind: SECT_METADATA,
        name: plow_asset::row_decode::SECTION.into(),
        data: serde_json::to_vec(&metadata).map_err(|e| e.to_string())?,
    });
    Ok(())
}
