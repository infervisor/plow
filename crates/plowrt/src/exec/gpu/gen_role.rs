//! Generated-kernel catalog roles (`plow_asset::segment_roles::GeneratedAbi`): one loader for
//! every entry. The object is hash-pinned and its cubin geometry must equal the packet's
//! declaration; the launch is the packet grid, one CTA per SM.
use super::*;
use plow_asset::segment_roles::{
    GeneratedAbi, GENERATED_FLASH_PREFILL_ABI, GENERATED_FLASH_PREFILL_FP8KV_ABI,
};

fn pinned_image(
    dir: &std::path::Path,
    object: &SegmentObject,
    reject: &dyn Fn(&str) -> RuntimeError,
) -> Result<(std::path::PathBuf, Vec<u8>)> {
    let path = dir.join(&object.file);
    let image = std::fs::read(&path)
        .map_err(|e| RuntimeError::Device(format!("{}: {e}", path.display())))?;
    if object.sha256.as_deref() != Some(plow_asset::decode_objects::image_sha256(&image).as_str()) {
        return Err(reject("object hash mismatch"));
    }
    Ok((path, image))
}

pub(super) fn load_generated_role(
    be: &Arc<CudaBackend>,
    dir: &std::path::Path,
    object: &SegmentObject,
    profile: &str,
    n_cu: u32,
    packed: Option<&plow_asset::packed_prefill::Manifest>,
    kv_layout: u32,
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
    let (path, image) = pinned_image(dir, object, &reject)?;
    if let Some(pack) = packed {
        pack.validate_object(|name| cubin::global_u32(&image, name))
            .map_err(RuntimeError::Rejected)?;
    }
    let module = DecodeModule::load(be, &image)?;
    for (name, want) in [
        (
            "plow_gen_flash_prefill_abi",
            if abi.fp8_kv() { 2 } else { 1 },
        ),
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
    if be
        .module_global_u32(&module, "plow_gen_kv_layout")?
        .unwrap_or(0)
        != kv_layout
    {
        return Err(reject("KV layout differs from the packet's cache writers"));
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
        direct_gen: Some((direct, attention.head_dim, abi.fp8_kv(), kv_layout)),
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

/// A generated flash-decode object (`GENERATED_FLASH_DECODE_ABI`, `gen_flash_decode.cu`): the
/// runtime launches it on a (kv heads x nsplit, rows) grid with its own argument struct
/// (`riders.rs` `GenDecodeArgs`), so no packet segment binds it.
pub(super) struct GenDecode {
    pub(super) function: KernelFn,
    pub(super) block: u32,
    pub(super) smem: u32,
    pub(super) head_dim: u32,
    /// Query heads per KV head the body is compiled for.
    pub(super) gqa: u32,
    pub(super) kv_tile: u32,
    pub(super) nsplit_max: u32,
    /// The launch's `pad`: the object traps unless it is the layout it was compiled for.
    pub(super) kv_layout: u32,
    _module: Arc<DecodeModule>,
}

/// `PlowGenFlashDecode` (runtime/nvidia/gen_flash_decode.cu): one launch's arguments, built by
/// the runtime for riders (`riders.rs`) and routed decode steps (`gen_decode.rs`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct GenDecodeArgs {
    pub(super) opart: u64,
    pub(super) mlpart: u64,
    pub(super) q: u64,
    pub(super) k: u64,
    pub(super) v: u64,
    pub(super) kv_len: u64,
    pub(super) slot: u64,
    pub(super) out: u64,
    pub(super) ctr: u64,
    pub(super) rows: u32,
    pub(super) n_head: u32,
    pub(super) n_kv_head: u32,
    pub(super) kv_stride: u32,
    pub(super) window: u32,
    pub(super) nsplit: u32,
    pub(super) kv_mask: u32,
    pub(super) hd: u32,
    pub(super) scale: f32,
    pub(super) pad: u32,
}

pub(super) fn load_generated_decode(
    be: &Arc<CudaBackend>,
    dir: &std::path::Path,
    object: &SegmentObject,
    profile: &str,
    kv_layout: u32,
) -> Result<GenDecode> {
    let reject =
        |why: &str| RuntimeError::Rejected(format!("generated decode role {}: {why}", object.abi));
    let abi = GeneratedAbi::parse(&object.abi).ok_or_else(|| reject("unparsable ABI"))?;
    let attention = object
        .attention
        .as_ref()
        .ok_or_else(|| reject("no capability"))?;
    if profile != "sm90a" || !abi.decode() {
        return Err(reject("requires SM90 and the decode family"));
    }
    let (path, image) = pinned_image(dir, object, &reject)?;
    let module = DecodeModule::load(be, &image)?;
    for (name, want) in [
        ("plow_gen_flash_decode_abi", 1),
        ("plow_gen_block", abi.block),
        ("plow_gen_arena_bytes", abi.smem),
        ("plow_attention_head_dim", attention.head_dim),
        ("plow_gen_decode_gqa", attention.query_tile),
        ("plow_attention_kv_tile", attention.kv_tile),
        ("plow_attention_warps", attention.warps),
    ] {
        if be.module_global_u32(&module, name)? != Some(want) {
            return Err(reject("cubin geometry differs from the packet declaration"));
        }
    }
    let nsplit_max = be
        .module_global_u32(&module, "plow_gen_decode_nsplit_max")?
        .filter(|&n| n > 0)
        .ok_or_else(|| reject("no split width"))?;
    if be
        .module_global_u32(&module, "plow_gen_kv_layout")?
        .unwrap_or(0)
        != kv_layout
    {
        return Err(reject("KV layout differs from the packet's cache writers"));
    }
    let function = be.get_function(&module, "plow_gen_flash_decode")?;
    be.set_max_dynamic_smem(function, abi.smem)?;
    if be.occupancy_blocks_per_sm(function, abi.block, abi.smem as usize)? == 0 {
        return Err(reject("does not fit one SM"));
    }
    tracing::info!(abi = %object.abi, object = %path.display(), "generated decode object loaded");
    Ok(GenDecode {
        function,
        block: abi.block,
        smem: abi.smem,
        head_dim: attention.head_dim,
        gqa: attention.query_tile,
        kv_tile: attention.kv_tile,
        nsplit_max,
        kv_layout,
        _module: module,
    })
}
