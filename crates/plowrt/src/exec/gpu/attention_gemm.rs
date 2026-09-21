//! Vendor-GEMM prefill attention (`PLOW_PF_ATTN_GEMM`) for full-attention layers with ONE KV
//! head: per query-row tile, `S = Q.K^T` (cuBLASLt), an in-place causal softmax
//! (`attn_softmax_sm90a.cubin`), `O = P.V` (cuBLASLt). Runtime-side: the packet is unchanged,
//! the `FlashPrefill` segment is simply not launched.
//!
//! One KV head makes both products plain 2-D GEMMs over the packet's own layouts: Q and the
//! output are `[row][head][hd]`, a slot's K/V are `[row][hd]`.

use super::*;
use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::cuda::lt::{AttentionGemm as Gemm, Lt, Plan};

pub(super) const SOFTMAX_OBJECT: &str = "attn_softmax_sm90a.cubin";
const SOFTMAX_ENTRY: &str = "plow_attn_softmax";
const SOFTMAX_ENTRY_F32: &str = "plow_attn_softmax_f32";
/// Scores are produced in the log2 domain so the kernel's exp is the hardware exp2.
const LOG2_E: f32 = std::f32::consts::LOG2_E;
/// Score rows are padded to this many columns (the kernel loads whole 8-element vectors).
const PITCH: u32 = 64;
const PLAN_CACHE: usize = 4096;

/// One `FlashPrefill` site served by the route. Addresses are the slot-0 tensor bases.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Site {
    pub(super) instruction: usize,
    q: u64,
    k: u64,
    v: u64,
    output: u64,
    pub(super) heads: u32,
    head_dim: u32,
    slot_bytes: u64,
    scale: f32,
}

/// `[q0, qlen, slot, kvlen]`: the packed request table's row, also built for a serialized chunk.
pub(super) type Request = [u32; 4];

#[repr(C)]
struct SoftmaxArgs {
    scores: u64,
    rows: u32,
    cols: u32,
    pitch: u32,
    heads: u32,
    first: u32,
    pad: u32,
}

/// Segments whose single instruction is a full-attention, one-KV-head BF16 `FlashPrefill` with
/// the fused output. Anything else keeps its native launch.
pub(super) fn sites(
    program: &DevProg,
    tensors: &[DevTensor],
    devp: &[DeviceMem],
    batch: usize,
) -> Vec<Option<Site>> {
    let count = program.gq_seg_ofs.len().saturating_sub(1);
    let mut out = vec![None; count];
    for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
        let Some(entries) = program.gq_stream.get(bounds[0] as usize..bounds[1] as usize) else {
            continue;
        };
        let Some(instruction) = entries.first().map(|e| e.inst as usize) else {
            continue;
        };
        let Some(op) = program.insts.get(instruction) else {
            continue;
        };
        let [rows, _, heads, kv_heads, _, window, head_dim, nsplit] = op.i;
        let kv_stride = op.fj[1];
        if op.op != DevOp::FlashPrefill as u16
            || kv_heads != 1
            || window != 0
            || nsplit != 1
            || heads == 0
            || head_dim == 0
            || head_dim % 8 != 0
            || kv_stride == 0
            || op.fj[2] != u32::MAX
            || op.t[..6].contains(&TENSOR_NONE16)
            || entries.iter().any(|e| e.inst as usize != instruction)
            || program
                .stream
                .iter()
                .any(|e| (e.inst as usize == instruction) != (e.seg as usize == segment))
        {
            continue;
        }
        let row_bytes = u64::from(heads) * u64::from(head_dim) * 2;
        let slot_bytes = u64::from(kv_stride) * u64::from(head_dim) * 2;
        let fits = |handle: u16, bytes: u64| {
            tensors.get(handle as usize).is_some_and(|t| t.bytes >= bytes)
                && devp.get(handle as usize).is_some()
        };
        if !fits(op.t[2], u64::from(rows) * row_bytes)
            || !fits(op.t[5], u64::from(rows) * row_bytes)
            || !fits(op.t[3], batch as u64 * slot_bytes)
            || !fits(op.t[4], batch as u64 * slot_bytes)
            || tensors[op.t[3] as usize].bytes / batch as u64 != slot_bytes
            || tensors[op.t[4] as usize].bytes / batch as u64 != slot_bytes
        {
            continue;
        }
        out[segment] = Some(Site {
            instruction,
            q: devp[op.t[2] as usize].base,
            k: devp[op.t[3] as usize].base,
            v: devp[op.t[4] as usize].base,
            output: devp[op.t[5] as usize].base,
            heads,
            head_dim,
            slot_bytes,
            scale: f32::from_bits(op.fj[0]),
        });
    }
    out
}

pub(super) struct AttentionGemm {
    be: Arc<CudaBackend>,
    lt: Arc<Lt>,
    module: Module,
    softmax: KernelFn,
    grid: u32,
    tile_rows: u32,
    /// Scores land in f32 (P stays bf16, at the start of each f32 score row).
    scores_f32: bool,
    scratch: DeviceMem,
    plans: std::collections::HashMap<(Gemm, u32, u32, u32), Plan>,
}

impl AttentionGemm {
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        lt: Arc<Lt>,
        object: &Path,
        max_heads: u32,
        max_ctx: usize,
    ) -> Result<Self> {
        let config = &crate::config::RuntimeConfig::get().nv;
        let image = std::fs::read(object).map_err(|e| {
            RuntimeError::Rejected(format!(
                "PLOW_PF_ATTN_GEMM needs {}: {e}",
                object.display()
            ))
        })?;
        let module = be.module_load(&image)?;
        if be.module_global_u32(&module, "plow_attn_softmax_abi")? != Some(1)
            || be.module_global_u32(&module, "plow_block_attn_softmax")? != Some(BLOCK)
        {
            return Err(RuntimeError::Rejected(
                "incompatible attention softmax object".into(),
            ));
        }
        let scores_f32 = config.pf_attn_gemm_s32;
        let softmax = be.get_function(
            &module,
            if scores_f32 { SOFTMAX_ENTRY_F32 } else { SOFTMAX_ENTRY },
        )?;
        let tile_rows = config.pf_attn_gemm_tile.max(1);
        let pitch = (max_ctx as u64).next_multiple_of(u64::from(PITCH));
        let element = if scores_f32 { 4 } else { 2 };
        let scratch = be.alloc(0, u64::from(tile_rows) * u64::from(max_heads) * pitch * element)?;
        tracing::info!(
            object = %object.display(),
            tile_rows,
            scores_f32,
            scratch_mib = scratch.len >> 20,
            "vendor-GEMM prefill attention loaded"
        );
        Ok(Self {
            be: Arc::clone(be),
            lt,
            module,
            softmax,
            grid: be.sm_count() * config.pf_attn_gemm_grid.max(1),
            tile_rows,
            scores_f32,
            scratch,
            plans: std::collections::HashMap::new(),
        })
    }

    pub(super) fn unload(self) -> Result<()> {
        self.be.module_unload(&self.module)
    }

    fn plan(&mut self, kind: Gemm, m: u32, n: u32, pitch: u32, head_dim: u32) -> Result<&Plan> {
        if self.plans.len() >= PLAN_CACHE {
            self.plans.clear();
        }
        // P is bf16 at the start of each score row: its pitch counts the f32 row in bf16 units.
        let pitch = if self.scores_f32 && kind == Gemm::Values { 2 * pitch } else { pitch };
        match self.plans.entry((kind, m, n, head_dim)) {
            std::collections::hash_map::Entry::Occupied(e) => Ok(e.into_mut()),
            std::collections::hash_map::Entry::Vacant(e) => Ok(e.insert(self.lt.attention_plan(
                kind,
                self.scores_f32,
                m,
                n,
                pitch,
                head_dim,
            )?)),
        }
    }

    /// Every request of one launch, in table order; rows past the last request become zero.
    pub(super) fn run(
        &mut self,
        site: &Site,
        requests: &[Request],
        rows: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        let row_bytes = u64::from(site.heads) * u64::from(site.head_dim) * 2;
        let mut real = 0;
        for &[q0, qlen, slot, kvlen] in requests {
            let past = kvlen.checked_sub(qlen).ok_or_else(|| {
                RuntimeError::Rejected("attention request is longer than its KV".into())
            })?;
            if u64::from(kvlen) * u64::from(site.head_dim) * 2 > site.slot_bytes
                || q0.checked_add(qlen).is_none_or(|end| end > rows)
            {
                return Err(RuntimeError::Rejected(
                    "attention request exceeds its tensors".into(),
                ));
            }
            let k = site.k + u64::from(slot) * site.slot_bytes;
            let v = site.v + u64::from(slot) * site.slot_bytes;
            let mut done = 0;
            while done < qlen {
                let tile = self.tile_rows.min(qlen - done);
                let m = tile * site.heads;
                let n = past + done + tile;
                let pitch = n.next_multiple_of(PITCH);
                let element = if self.scores_f32 { 4 } else { 2 };
                if u64::from(m) * u64::from(pitch) * element > self.scratch.len {
                    return Err(RuntimeError::Rejected(
                        "attention score tile exceeds its scratch".into(),
                    ));
                }
                let offset = u64::from(q0 + done) * row_bytes;
                let scratch = self.scratch.base;
                self.plan(Gemm::Scores, m, n, pitch, site.head_dim)?.matmul(
                    site.scale * LOG2_E,
                    k,
                    site.q + offset,
                    scratch,
                    stream,
                )?;
                let mut args = SoftmaxArgs {
                    scores: scratch,
                    rows: m,
                    cols: n,
                    pitch,
                    heads: site.heads,
                    first: past + done + 1,
                    pad: 0,
                };
                let mut params = [&mut args as *mut SoftmaxArgs as *mut std::ffi::c_void];
                self.be
                    .launch_kernel(self.softmax, self.grid, BLOCK, 0, &mut params, Some(stream))?;
                self.plan(Gemm::Values, m, n, pitch, site.head_dim)?.matmul(
                    1.0,
                    v,
                    scratch,
                    site.output + offset,
                    stream,
                )?;
                done += tile;
            }
            real = real.max(q0 + qlen);
        }
        if real < rows {
            self.be.memset_d8_async(
                site.output + u64::from(real) * row_bytes,
                0,
                (u64::from(rows - real) * row_bytes) as usize,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{DevInst64, StreamEnt};

    fn fixture() -> (DevProg, Vec<DevTensor>, Vec<DeviceMem>) {
        let mut flash = DevInst64 {
            op: DevOp::FlashPrefill as u16,
            blocks: 1,
            ..Default::default()
        };
        flash.t = [0, 1, 2, 3, 4, 5, TENSOR_NONE16, TENSOR_NONE16];
        flash.i = [128, 128, 16, 1, 0, 0, 512, 1];
        flash.fj = [1.0f32.to_bits(), 1024, u32::MAX];
        let nop = DevInst64 {
            op: DevOp::Nop as u16,
            blocks: 1,
            ..Default::default()
        };
        let stream: Vec<_> = (0..2)
            .map(|index| StreamEnt {
                inst: index,
                seg: index as u16,
                ..Default::default()
            })
            .collect();
        let rows = 128 * 16 * 512 * 2;
        let cache = 4 * 1024 * 512 * 2;
        let tensors: Vec<_> = [rows, rows, rows, cache, cache, rows]
            .into_iter()
            .enumerate()
            .map(|(index, bytes)| DevTensor {
                name: format!("t{index}"),
                bytes,
                init: None,
            })
            .collect();
        let devp = tensors
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                DeviceMem::view(0x1000_0000 * (index as u64 + 1), tensor.bytes)
            })
            .collect();
        (
            DevProg {
                t: 128,
                role: packet::devbuild::ProgramRole::PrefillBucket { rows: 128 },
                n_counter: 0,
                insts: vec![nop, flash],
                stream: stream.clone(),
                stream_ofs: vec![0],
                stream_len: vec![2],
                waits: vec![],
                succs: vec![],
                gq_stream: stream,
                gq_seg_ofs: vec![0, 1, 2],
                l2_domains: 0,
            },
            tensors,
            devp,
        )
    }

    #[test]
    fn selects_one_kv_head_full_attention_only() {
        let (program, tensors, devp) = fixture();
        let found = sites(&program, &tensors, &devp, 4);
        assert!(found[0].is_none());
        let site = found[1].expect("global attention site");
        assert_eq!((site.instruction, site.heads, site.head_dim), (1, 16, 512));
        assert_eq!(site.slot_bytes, 1024 * 512 * 2);

        for patch in [
            (|op: &mut DevInst64| op.i[3] = 2) as fn(&mut DevInst64),
            |op| op.i[5] = 1024,
            |op| op.i[7] = 2,
            |op| op.t[5] = TENSOR_NONE16,
            |op| op.fj[2] = 1023,
            |op| op.op = DevOp::FlashPrefillFp8 as u16,
        ] {
            let (mut program, tensors, devp) = fixture();
            patch(&mut program.insts[1]);
            assert!(sites(&program, &tensors, &devp, 4)[1].is_none());
        }
        // A KV tensor whose slot pitch is not the instruction's stride is not a linear cache.
        assert!(sites(&program, &tensors, &devp, 2)[1].is_none());
    }
}
