use super::*;
use plow_asset::fp8_m1_role::{capability, ARENA, ENTRY};

fn check_image(image: &[u8], object: &SegmentObject, profile: &str) -> Result<()> {
    let info = plow_asset::cubin::inspect(image)
        .ok_or_else(|| RuntimeError::Rejected("invalid FP8 role ELF".into()))?;
    if profile != "sm90a"
        || info.sm != 90
        || !info.entries.iter().any(|e| e == ENTRY)
        || object.sha256.as_deref()
            != Some(plow_asset::decode_objects::image_sha256(image).as_str())
    {
        return Err(RuntimeError::Rejected(
            "FP8 role object hash/ISA/entry mismatch".into(),
        ));
    }
    Ok(())
}

pub(super) fn load_fp8_m1_role(
    be: &Arc<CudaBackend>,
    dir: &std::path::Path,
    object: &SegmentObject,
    profile: &str,
    grid: u32,
) -> Result<PacketRole> {
    let path = dir.join(&object.file);
    let image = std::fs::read(&path)
        .map_err(|e| RuntimeError::Rejected(format!("{}: {e}", path.display())))?;
    check_image(&image, object, profile)?;
    let module = DecodeModule::load(be, &image)?;
    let loaded = (|| {
        let symbols = [
            "plow_fp8_gemm_m1_tma_abi",
            "plow_segment_gq_abi_fp8m1",
            "plow_block_fp8m1",
            "plow_arena_bytes_fp8m1",
            "plow_fp8_m1_max_k",
            "plow_fp8_m1_k_multiple",
            "plow_fp8_m1_promote_k512",
        ];
        let mut values = [None; 7];
        for (v, s) in values.iter_mut().zip(symbols) {
            *v = be.module_global_u32(&module, s)?;
        }
        for name in ["plow_packet_hash_lo_fp8m1", "plow_packet_hash_hi_fp8m1"] {
            if be.module_global_u32(&module, name)?.is_some() {
                return Err(RuntimeError::Rejected(
                    "stamped FP8 role unsupported".into(),
                ));
            }
        }
        let promote = object
            .promote_k512
            .ok_or_else(|| RuntimeError::Rejected("missing FP8 accumulator mode".into()))?;
        capability(profile, values, promote, Some(grid), grid).map_err(RuntimeError::Rejected)?;
        let function = be.get_function(&module, ENTRY)?;
        be.set_max_dynamic_smem(function, ARENA)?;
        let capacity = be
            .occupancy_blocks_per_sm(function, 256, ARENA as usize)?
            .checked_mul(be.sm_count());
        capability(profile, values, promote, capacity, grid).map_err(RuntimeError::Rejected)?;
        tracing::info!(role=4,object=%path.display(),sha256=?object.sha256,block=256,smem=ARENA,?capacity,grid,promote,"FP8 M1 segment object loaded");
        Ok((function, ARENA))
    })();
    match loaded {
        Ok((function, smem)) => Ok(PacketRole {
            function,
            grid,
            smem,
            block: 256,
            _module: module,
        }),
        Err(e) => Err(e),
    }
}

pub(super) fn validate_fp8_role_checkpoint(
    roles: &SegmentRoles,
    blob: &DevBlob,
    ckpt: &Checkpoint,
) -> Result<()> {
    for p in &roles.programs {
        let g = &blob.progs[p.index];
        for (seg, &role) in p.roles.iter().enumerate() {
            if role != plow_asset::segment_roles::FP8_M1 {
                continue;
            }
            let d = &g.insts[g.gq_stream[g.gq_seg_ofs[seg] as usize].inst as usize];
            let weight = &blob.tensors[d.t[2] as usize].name;
            let scale = &blob.tensors[d.t[4] as usize].name;
            let shape = [d.i[1] as usize, d.i[2] as usize];
            checkpoint_fields(
                ckpt.dtype(weight),
                ckpt.tensor_ex(weight).map(|(_, s)| s),
                ckpt.dtype(scale),
                ckpt.tensor_ex(scale).map(|(_, s)| s),
                shape,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "gpu_fp8_m1_role_tests.rs"]
mod tests;

fn checkpoint_fields(
    weight: Option<safetensors::Dtype>,
    weight_shape: Option<&[usize]>,
    scale: Option<safetensors::Dtype>,
    scale_shape: Option<&[usize]>,
    shape: [usize; 2],
) -> Result<()> {
    if weight != Some(safetensors::Dtype::F8_E4M3)
        || weight_shape != Some(shape.as_slice())
        || scale != Some(safetensors::Dtype::F32)
        || scale_shape != Some(&shape[..1])
    {
        return Err(RuntimeError::Rejected(
            "FP8 M1 checkpoint dtype/shape mismatch".into(),
        ));
    }
    Ok(())
}
