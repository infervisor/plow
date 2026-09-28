//! Routed-expert pointer tables on the CUDA engine: the `[E][3]` `{gate, up, down}` u64 tables
//! (`<pfx>expert_weight_table` / `<pfx>expert_scale_table`) the grouped prefill MoE reads, filled
//! from the checkpoint's per-expert tensors the way `exec::amd::bind_packed_experts` does: every
//! expert, packed into one slab per table, the table entry for `(expert, proj)` the address of its
//! slot. Under tensor parallelism each expert is this rank's `crate::asset::shard::slice_for` slice
//! (gate/up over their `I_moe` rows, down over its `I_moe` columns).
use crate::asset::checkpoint::Checkpoint;
use crate::device::DeviceMem;
use crate::orch::moe::{check_expert_geometry, packed_expert_table, resolve_expert_names};
use crate::{Result, RuntimeError};
use packet::dev::DevOp;
use crate::asset::devblob::DevBlob;

/// Is `name` a table this binder fills (the plain pair; the EP / shared-fold / preshuffled
/// companions are not wired on CUDA).
pub(crate) fn binds(name: &str) -> bool {
    name.ends_with("expert_weight_table") || name.ends_with("expert_scale_table")
}

/// Bytes ahead of the weight slab's first matrix: a `CUtensorMap` over the slab as
/// `[E * 3][I][H / 2]` bytes (gate / up rows, box `[1][32][256]`) that the sm_90a decode MoE GLU
/// streams with one TMA copy per chunk (`op_moe_decode_v41.cuh`, `wtab[0] - EXPERT_TMAP_BYTES`).
pub(crate) const EXPERT_TMAP_BYTES: u64 = 4096;

/// Fills every plain expert table pair in `blob`. Returns the slabs, which must outlive the engine.
/// `tmap(base, dims, strides, box)` encodes a rank-3 byte tensor map.
pub(crate) fn bind_packed_experts(
    blob: &DevBlob,
    ckpt: &Checkpoint,
    devp: &[DeviceMem],
    alloc: impl Fn(u64) -> Result<DeviceMem>,
    upload: impl Fn(&DeviceMem, u64, &[u8]) -> Result<()>,
    tmap: impl Fn(u64, [u64; 3], [u64; 2], [u32; 3]) -> Result<[u8; 128]>,
    rank: u32,
    n_gpu: u32,
) -> Result<Vec<DeviceMem>> {
    let slice = |name: &str| -> Result<std::borrow::Cow<'_, [u8]>> {
        let (bytes, shape) = ckpt.tensor_ex(name).ok_or_else(|| RuntimeError::Device(format!("MISSING EXPERT TENSOR: {name}")))?;
        if bytes.len() as u64 % n_gpu as u64 != 0 {
            return Err(RuntimeError::Device(format!("{name}: {} B does not split {n_gpu} ways", bytes.len())));
        }
        crate::asset::shard::slice_for(name, bytes, shape, bytes.len() as u64 / n_gpu as u64, rank, n_gpu)
    };
    let mut slabs = Vec::new();
    for (i_ewt, td) in blob.tensors.iter().enumerate() {
        let Some(pfx) = td.name.strip_suffix("expert_weight_table") else { continue };
        let est_name = format!("{pfx}expert_scale_table");
        let i_est = blob
            .tensors
            .iter()
            .position(|t| t.name == est_name)
            .ok_or_else(|| RuntimeError::Device(format!("{} has no matching {est_name}", td.name)))?;
        if td.bytes == 0 || td.bytes % 24 != 0 || blob.tensors[i_est].bytes != td.bytes {
            return Err(RuntimeError::Device(format!("{}: not a pair of [E][3] u64 tables", td.name)));
        }
        let n_exp = (td.bytes / 24) as u32;
        let i_moe = blob
            .progs
            .iter()
            .flat_map(|p| &p.insts)
            .find(|d| d.op == DevOp::MoeGroupGluPf as u16 && d.t[2] as usize == i_ewt)
            .map(|d| d.i[0] as usize)
            .ok_or_else(|| RuntimeError::Device(format!("{}: no MoeGroupGluPf streams it", td.name)))?;
        let en = resolve_expert_names(ckpt, pfx)?;
        check_expert_geometry(ckpt, &en)?;
        let probe = en.weight_of(0, 0, false);
        let (_, shape0) = ckpt.tensor_ex(&probe).ok_or_else(|| RuntimeError::Device(format!("MISSING EXPERT WEIGHT: {probe}")))?;
        if shape0.first() != Some(&(i_moe * n_gpu as usize)) {
            return Err(RuntimeError::Device(format!(
                "{pfx}: the packet streams I_moe={i_moe} per rank at tp={n_gpu} but the checkpoint's experts are \
                 {shape0:?} (expert-parallel placement is not ported to CUDA)"
            )));
        }
        let w_stride = slice(&probe)?.len() as u64;
        let s_stride = slice(&en.scale_of(0, 0, false))?.len() as u64;
        let d_w = alloc(EXPERT_TMAP_BYTES + n_exp as u64 * 3 * w_stride)?;
        let d_s = alloc(n_exp as u64 * 3 * s_stride)?;
        let w_base = d_w.base + EXPERT_TMAP_BYTES;
        let rows = (shape0[0] as u64) / n_gpu as u64;
        let row_bytes = w_stride / rows;
        if shape0.len() == 2 && rows * row_bytes == w_stride && row_bytes % 256 == 0 {
            let m = tmap(w_base, [row_bytes, rows, n_exp as u64 * 3], [row_bytes, w_stride], [256, 32, 1])?;
            upload(&d_w, 0, &m)?;
        }
        for e in 0..n_exp {
            for j in 0..3 {
                for (name, dst, stride) in [(en.weight_of(e, j, false), &d_w, w_stride), (en.scale_of(e, j, false), &d_s, s_stride)] {
                    let bytes = slice(&name)?;
                    if bytes.len() as u64 != stride {
                        return Err(RuntimeError::Device(format!("{name}: {} B, expert 0's is {stride} B", bytes.len())));
                    }
                    let off = if std::ptr::eq(dst, &d_w) { EXPERT_TMAP_BYTES } else { 0 };
                    upload(dst, off + (e as u64 * 3 + j as u64) * stride, &bytes)?;
                }
            }
        }
        let wtab = packed_expert_table(w_base, w_stride, n_exp, 0..n_exp);
        let stab = packed_expert_table(d_s.base, s_stride, n_exp, 0..n_exp);
        upload(&devp[i_ewt], 0, bytemuck::cast_slice(&wtab))?;
        upload(&devp[i_est], 0, bytemuck::cast_slice(&stab))?;
        tracing::info!(table = %td.name, n_exp, rank, n_gpu, w_stride, s_stride, layout = %format!("{}{{gate,up,down}}{}+{}", en.ns, en.payload, en.scale), "bound routed experts");
        slabs.push(d_w);
        slabs.push(d_s);
    }
    Ok(slabs)
}
