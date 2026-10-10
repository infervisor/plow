//! Generated-kernel catalog roles (`plow_asset::segment_roles::GeneratedAbi`): one loader for
//! every entry. The object is hash-pinned and its cubin geometry must equal the packet's
//! declaration; the launch is the packet grid, one CTA per SM.
use super::*;
use plow_asset::segment_roles::{
    GeneratedAbi, GENERATED_FLASH_PREFILL_ABI, GENERATED_FLASH_PREFILL_FP8KV_ABI,
};

pub(super) fn load_generated_role(
    be: &Arc<CudaBackend>,
    dir: &std::path::Path,
    object: &SegmentObject,
    profile: &str,
    n_cu: u32,
    packed: Option<&plow_asset::packed_prefill::Manifest>,
) -> Result<PacketRole> {
    let reject =
        |why: &str| RuntimeError::Rejected(format!("generated role {}: {why}", object.abi));
    let abi = GeneratedAbi::parse(&object.abi).ok_or_else(|| reject("unparsable ABI"))?;
    let attention = object
        .attention
        .as_ref()
        .ok_or_else(|| reject("no capability"))?;
    if profile != "sm90a"
        || (abi.family != GENERATED_FLASH_PREFILL_ABI
            && abi.family != GENERATED_FLASH_PREFILL_FP8KV_ABI)
    {
        return Err(reject("requires SM90 and a known family"));
    }
    let path = dir.join(&object.file);
    let image = std::fs::read(&path)
        .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
    if object.sha256.as_deref() != Some(plow_asset::decode_objects::image_sha256(&image).as_str()) {
        return Err(reject("object hash mismatch"));
    }
    if let Some(pack) = packed {
        pack.validate_object(|name| cubin::global_u32(&image, name))
            .map_err(RuntimeError::Rejected)?;
    }
    let module = DecodeModule::load(be, &image)?;
    for (name, want) in [
        ("plow_gen_flash_prefill_abi", if abi.fp8_kv() { 2 } else { 1 }),
        ("plow_gen_block", abi.block),
        ("plow_gen_arena_bytes", abi.smem),
        ("plow_attention_head_dim", attention.head_dim),
        ("plow_attention_query_tile", attention.query_tile),
        ("plow_attention_kv_tile", attention.kv_tile),
        ("plow_attention_warps", attention.warps),
    ] {
        if be.module_global_u32(&module, name)? != Some(want) {
            return Err(reject("cubin geometry differs from the packet declaration"));
        }
    }
    let function = be.get_function(&module, "plow_gen_flash_prefill")?;
    let direct = be.get_function(&module, "plow_gen_flash_prefill_direct")?;
    let media_span = be.module_global_u32(&module, "plow_attention_media_span")? == Some(1);
    for f in [function, direct] {
        be.set_max_dynamic_smem(f, abi.smem)?;
        if be.occupancy_blocks_per_sm(f, abi.block, abi.smem as usize)? * be.sm_count() != n_cu {
            return Err(reject("occupancy must equal the packet grid"));
        }
    }
    tracing::info!(abi = %object.abi, object = %path.display(), "generated role object loaded");
    Ok(PacketRole {
        function,
        direct_gen: Some((direct, attention.head_dim, abi.fp8_kv())),
        media_span,
        direct_hd512: None,
        direct_hd256_gqa2: None,
        direct_w8a8_glu: None,
        grid: n_cu,
        smem: abi.smem,
        block: abi.block,
        _module: module,
    })
}
