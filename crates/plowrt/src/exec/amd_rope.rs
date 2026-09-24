use std::path::Path;

use packet::rope::{GenTensor, GEN_AMD_ROPE_BF16_COS, GEN_AMD_ROPE_BF16_SIN,
    GEN_AMD_ROPE_IDX_BF16_COS, GEN_AMD_ROPE_IDX_BF16_SIN};

use crate::asset::devblob::DevTensor;
use crate::device::{hsa::HsaBackend, DeviceMem, Module};
use crate::exec::device_api::EngineDevice;
use crate::{Result, RuntimeError};

fn invalid(message: &str) -> RuntimeError {
    RuntimeError::Device(format!("AMD BF16 RoPE: {message}"))
}

fn plan(gen: &[GenTensor], tensors: &[DevTensor]) -> Result<Option<Vec<GenTensor>>> {
    let recipes: Vec<_> = gen
        .iter()
        .copied()
        .filter(|g| matches!(g.kind, GEN_AMD_ROPE_BF16_COS | GEN_AMD_ROPE_BF16_SIN
            | GEN_AMD_ROPE_IDX_BF16_COS | GEN_AMD_ROPE_IDX_BF16_SIN))
        .collect();
    if recipes.is_empty() {
        return Ok(None);
    }
    if !matches!(recipes.len(), 2 | 4) {
        return Err(invalid("requires main pair and optional complete indexer pair"));
    }
    let mut ordered = Vec::with_capacity(recipes.len());
    for (kind, name) in [
        (GEN_AMD_ROPE_BF16_COS, "in.cos"), (GEN_AMD_ROPE_BF16_SIN, "in.sin"),
        (GEN_AMD_ROPE_IDX_BF16_COS, "in.icos"), (GEN_AMD_ROPE_IDX_BF16_SIN, "in.isin"),
    ].into_iter().take(recipes.len()) {
        let g = *recipes.iter().find(|g| g.kind == kind)
            .ok_or_else(|| invalid("missing recipe kind"))?;
        if !g.amd_rope_bf16() || g.ctx != recipes[0].ctx {
            return Err(invalid("invalid plain GLM rotary64/head64-or-128/theta8e6 recipe"));
        }
        if gen.iter().filter(|r| r.tensor == g.tensor).count() != 1
            || tensors
                .get(g.tensor as usize)
                .is_none_or(|t| t.name != name || t.bytes != g.byte_len() || t.init.is_some())
        {
            return Err(invalid("recipe ownership, name or capacity"));
        }
        ordered.push(g);
    }
    Ok(Some(ordered))
}

pub(super) fn bind(
    be: &HsaBackend,
    dir: &Path,
    gen: &[GenTensor],
    tensors: &[DevTensor],
    memory: &[DeviceMem],
    modules: &mut Vec<Module>,
) -> Result<()> {
    let Some(recipes) = plan(gen, tensors)? else {
        return Ok(());
    };
    if be.arch() != "gfx950" {
        return Err(invalid("requires gfx950"));
    }
    let mut pointers = Vec::with_capacity(recipes.len());
    for g in &recipes {
        let mem = memory
            .get(g.tensor as usize)
            .ok_or_else(|| invalid("missing device tensor"))?;
        if mem.len != g.byte_len() {
            return Err(invalid("device capacity"));
        }
        let end = mem.base.checked_add(mem.len).ok_or_else(|| invalid("device address overflow"))?;
        for (&ptr, prior) in pointers.iter().zip(&recipes) {
            if mem.base < ptr + prior.byte_len() && ptr < end {
                return Err(invalid("aliased tables"));
            }
        }
        pointers.push(mem.base);
    }
    let path = dir.join("rope_cache_bf16_gfx950.elf");
    let image = std::fs::read(&path).map_err(|e| invalid(&format!("{}: {e}", path.display())))?;
    if plow_asset::decode_objects::image_sha256(&image)
        != "989f9a41382f3d04de69ac9a0820e38baf93454857e6dcb6e15c4b4c21d9f58d"
    {
        return Err(invalid("unqualified code object"));
    }
    let module = EngineDevice::module_load(be, &image)?;
    let kernel = EngineDevice::get_function(be, &module, "glm_rope_cache_bf16")?;
    if kernel.kernarg_size() != 280
        || kernel.group_segment_size() != 0
        || kernel.private_segment_size() != 0
    {
        return Err(invalid("kernel ABI or resource envelope"));
    }
    modules.push(module);
    let args = [
        pointers[0],
        pointers[1],
        u64::from(recipes[0].ctx) | (u64::from(8000000f32.to_bits()) << 32),
    ];
    be.launch(kernel, 256, 256, 0, bytemuck::cast_slice(&args))?;
    be.synchronize()?;
    if recipes.len() == 4 {
        let path = dir.join("rope_indexer_cache_bf16_gfx950.elf");
        let image = std::fs::read(&path).map_err(|e| invalid(&format!("{}: {e}", path.display())))?;
        if plow_asset::decode_objects::image_sha256(&image)
            != "4b00f93b1b78b21f4b81168f054526672c14ac6581acb26c3285ec8944bcaad8" {
            return Err(invalid("unqualified indexer code object"));
        }
        let module = EngineDevice::module_load(be, &image)?;
        let kernel = EngineDevice::get_function(be, &module, "glm_rope_indexer_cache_bf16")?;
        if kernel.kernarg_size() != 296 || kernel.group_segment_size() != 0
            || kernel.private_segment_size() != 0 {
            return Err(invalid("indexer kernel ABI or resource envelope"));
        }
        modules.push(module);
        let args = [pointers[0], pointers[1], pointers[2], pointers[3], u64::from(recipes[0].ctx)];
        be.launch(kernel, 256, 256, 0, bytemuck::cast_slice(&args))?;
        be.synchronize()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::rope::RopeScale;

    #[test]
    #[ignore = "requires queued gfx950 GPU and PLOW_TEST_AITER_DIR with object/reference cache"]
    fn amd_rope_bind_matches_pinned_cache() {
        let dir = std::env::var("PLOW_TEST_AITER_DIR").unwrap();
        let dir = Path::new(&dir);
        let reference = std::fs::read(dir.join("reference/rope-cache.bf16")).unwrap();
        assert_eq!(
            plow_asset::decode_objects::image_sha256(&reference),
            "0d16c27c9681a59375a933a58bf9afc20e86bfb72d84cd8fd294846d67cfdb96"
        );
        let be = HsaBackend::new(0).unwrap();
        let mut modules = Vec::new();
        for ctx in [512, 8192, 71680, 131072] {
            let mut gen = GenTensor::rope_pair(ctx, 64, 8000000.0, 1.0, RopeScale::None).to_vec();
            gen[0].kind = GEN_AMD_ROPE_BF16_COS;
            gen[1].kind = GEN_AMD_ROPE_BF16_SIN;
            gen[1].tensor = 1;
            let mut idx = GenTensor::rope_idx_pair(ctx, 64, 128, 8000000.0);
            idx[0].kind = GEN_AMD_ROPE_IDX_BF16_COS;
            idx[1].kind = GEN_AMD_ROPE_IDX_BF16_SIN;
            idx[0].tensor = 2;
            idx[1].tensor = 3;
            gen.extend(idx);
            let tensors: Vec<_> = ["in.cos", "in.sin", "in.icos", "in.isin"]
                .into_iter()
                .zip(&gen)
                .map(|(name, g)| DevTensor {
                    name: name.into(),
                    bytes: g.byte_len(),
                    init: None,
                })
                .collect();
            let memory: Vec<_> = tensors
                .iter()
                .map(|t| EngineDevice::alloc(&be, t.bytes).unwrap())
                .collect();
            for _ in 0..2 {
                for mem in &memory {
                    be.memcpy_htod(mem.base, &vec![255; mem.len as usize]).unwrap();
                }
                bind(&be, dir, &gen, &tensors, &memory, &mut modules).unwrap();
                for (part, mem) in memory.iter().enumerate() {
                    let mut actual = vec![0; mem.len as usize];
                    EngineDevice::download(&be, mem, 0, &mut actual).unwrap();
                    let width = if part < 2 { 32 } else { 64 };
                    for (i, word) in actual.chunks_exact(4).enumerate() {
                        let expected = if i % width < 32 {
                            let offset = ((i / width) * 64 + (part % 2) * 32 + i % width) * 2;
                            u32::from(u16::from_le_bytes(
                                reference[offset..offset + 2].try_into().unwrap(),
                            )) << 16
                        } else if part == 2 { 1f32.to_bits() } else { 0 };
                        assert_eq!(
                            u32::from_le_bytes(word.try_into().unwrap()),
                            expected,
                            "ctx={ctx} part={part} element={i}"
                        );
                    }
                }
            }
            println!("AMD RoPE bind bitwise PASS ctx={ctx}");
        }
    }

    #[test]
    fn amd_rope_pair_rejects_invalid_ownership_and_geometry() {
        let mut gen = GenTensor::rope_pair(131072, 64, 8000000.0, 1.0, RopeScale::None);
        gen[0].kind = GEN_AMD_ROPE_BF16_COS;
        gen[1].kind = GEN_AMD_ROPE_BF16_SIN;
        gen[1].tensor = 1;
        let mut tensors = vec![
            DevTensor {
                name: "in.cos".into(),
                bytes: gen[0].byte_len(),
                init: None,
            },
            DevTensor {
                name: "in.sin".into(),
                bytes: gen[1].byte_len(),
                init: None,
            },
        ];
        assert!(plan(&gen, &tensors).unwrap().is_some());
        assert!(plan(&[gen[1], gen[0]], &tensors).unwrap().is_some());
        assert!(plan(&[], &tensors).unwrap().is_none());
        assert!(plan(&gen[..1], &tensors).is_err());
        assert!(plan(&[gen[0], gen[1], gen[0]], &tensors).is_err());
        for field in 0..5 {
            let mut bad = gen;
            match field {
                0 => bad[1].tensor = 0,
                1 => bad[1].ctx = 8192,
                2 => bad[0].theta = 10000.0,
                3 => bad[0].hd = 128,
                _ => bad[1].kind = GEN_AMD_ROPE_BF16_COS,
            }
            assert!(plan(&bad, &tensors).is_err());
        }
        tensors[1].bytes -= 4;
        assert!(plan(&gen, &tensors).is_err());
        tensors[1].bytes += 4;
        tensors[1].init = Some(0..1);
        assert!(plan(&gen, &tensors).is_err());
        tensors[1].init = None;
        let mut all = gen.to_vec();
        let mut idx = GenTensor::rope_idx_pair(131072, 64, 128, 8000000.0);
        for (i, (kind, name)) in [(GEN_AMD_ROPE_IDX_BF16_COS, "in.icos"),
            (GEN_AMD_ROPE_IDX_BF16_SIN, "in.isin")].into_iter().enumerate() {
            idx[i].kind = kind;
            idx[i].tensor = (i + 2) as u32;
            tensors.push(DevTensor { name: name.into(), bytes: idx[i].byte_len(), init: None });
        }
        all.extend(idx);
        assert_eq!(plan(&all, &tensors).unwrap().unwrap(), all);
        let reversed: Vec<_> = all.iter().copied().rev().collect();
        assert_eq!(plan(&reversed, &tensors).unwrap().unwrap(), all);
        assert!(plan(&all[2..], &tensors).is_err());
        assert!(plan(&all[..3], &tensors).is_err());
        for field in 0..6 {
            let mut bad = all.clone();
            match field {
                0 => bad[2].tensor = 0,
                1 => { bad[2].ctx = 8192; bad[3].ctx = 8192; },
                2 => bad[2].aux = 0,
                3 => bad[2].hd = 64,
                4 => bad[3].kind = GEN_AMD_ROPE_IDX_BF16_COS,
                _ => { let mut duplicate = bad[2]; duplicate.kind = 2; bad.push(duplicate); },
            }
            assert!(plan(&bad, &tensors).is_err());
        }
        tensors[3].name = "in.sin".into();
        assert!(plan(&all, &tensors).is_err());
    }
}
