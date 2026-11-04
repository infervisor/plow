//! Vendor-GEMM prefill attention (`PLOW_PF_ATTN_GEMM`) for full-attention layers: per
//! query-row tile, `S = Q.K^T` (cuBLASLt), an in-place causal softmax
//! (`attn_softmax_sm90a.cubin`), `O = P.V` (cuBLASLt). Runtime-side: the packet is unchanged,
//! the `FlashPrefill` segment is simply not launched.
//!
//! One KV head makes both products plain 2-D GEMMs over the packet's own layouts: Q and the
//! output are `[row][head][hd]`, a slot's K/V are `[row][hd]`. With several KV heads (a slot is
//! `[kv head][row][hd]`) each KV head's query heads are one strided-batch GEMM; the score tile
//! keeps the same `[row][head][pitch]` layout, so the softmax is shared.

use super::*;
use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::cuda::lt::{AttentionGemm as Gemm, Lt, Plan};

pub(super) const SOFTMAX_OBJECT: &str = "attn_softmax_sm90a.cubin";
/// ABI 2 adds the grouped entries; an ABI-1 object serves the per-request route only.
const SOFTMAX_ABI: u32 = 2;
/// The warp-per-row entries (coalesced); the thread-per-row ones stay in the object for A/Bs.
const SOFTMAX_ENTRY: &str = "plow_attn_softmax_w";
const SOFTMAX_ENTRY_F32: &str = "plow_attn_softmax_w_f32";
/// Scores are produced in the log2 domain so the kernel's exp is the hardware exp2.
const LOG2_E: f32 = std::f32::consts::LOG2_E;
/// Score rows are padded to this many columns: the tail GEMM below reads P in 128-column
/// pieces, and the kernel loads whole 8-element vectors.
const PITCH: u32 = 128;
/// cuBLASLt's BF16 kernels want the KV extent 8-aligned (measured 2.6x slower otherwise). V
/// rows past a request's `kvlen` may hold anything, so the last tile's `P.V` stops at a
/// 128-aligned column and finishes on a zero-padded copy of the remaining rows, this many.
const TAIL_ROWS: u32 = 128;
const PLAN_CACHE: usize = 4096;

/// The softmax object the route loads (the bundle's object dir, then the asset dir), or `None`
/// when the route is off: `PLOW_PF_ATTN_GEMM=0`, or unset and the packet carries no object.
/// `=1` names the object whether or not it exists, so a missing one fails the load.
pub(super) fn object(assets_dir: &Path) -> Option<std::path::PathBuf> {
    let config = &crate::config::RuntimeConfig::get().nv;
    let path = crate::asset::serve::objects_dir(assets_dir)
        .map(|dir| dir.join(SOFTMAX_OBJECT))
        .filter(|path| path.exists())
        .unwrap_or_else(|| assets_dir.join(SOFTMAX_OBJECT));
    match config.pf_attn_gemm {
        Some(false) => None,
        Some(true) => Some(path),
        None => path.exists().then_some(path),
    }
}

/// Query rows per score tile: the knob, capped at the widest routed bucket (no tile is longer
/// than its request).
fn tile_rows(max_rows: u32) -> u32 {
    crate::config::RuntimeConfig::get()
        .nv
        .pf_attn_gemm_tile
        .clamp(1, max_rows.max(1))
}

/// KV columns the score scratch is sized for. A longer extent runs proportionally shorter tiles
/// (`AttentionGemm::tile`) instead of growing the scratch with the context: f32 scores at a
/// 262144 context would take 16 GiB.
const SCRATCH_COLS: usize = 16384;

/// The context the scratch holds whole tiles of.
pub(super) fn scratch_ctx(max_ctx: usize) -> usize {
    max_ctx.min(SCRATCH_COLS)
}

/// The score scratch `load` allocates: one tile of every head at the widest KV pitch it holds.
pub(super) fn scratch_bytes(max_heads: u32, max_ctx: usize, max_rows: u32) -> u64 {
    let element = if crate::config::RuntimeConfig::get().nv.pf_attn_gemm_s32 {
        4
    } else {
        2
    };
    let pitch = (scratch_ctx(max_ctx) as u64).next_multiple_of(u64::from(PITCH));
    u64::from(tile_rows(max_rows)) * u64::from(max_heads) * pitch * element
}

/// One `FlashPrefill` site served by the route. Addresses are the slot-0 tensor bases.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Site {
    pub(super) instruction: usize,
    q: u64,
    k: u64,
    v: u64,
    output: u64,
    pub(super) heads: u32,
    pub(super) head_dim: u32,
    kv_heads: u32,
    /// One KV head's `[kv_stride][hd]` block; a slot holds `kv_heads` of them.
    head_bytes: u64,
    slot_bytes: u64,
    scale: f32,
}

impl Site {
    fn row_bytes(&self) -> u64 {
        u64::from(self.heads) * u64::from(self.head_dim) * 2
    }

    fn kv_row_bytes(&self) -> u64 {
        u64::from(self.head_dim) * 2
    }

    fn slot_rows(&self) -> u32 {
        (self.head_bytes / self.kv_row_bytes()) as u32
    }

    /// Query heads per KV head.
    fn group(&self) -> u32 {
        self.heads / self.kv_heads
    }
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

/// Segments whose single instruction is a full-attention BF16 `FlashPrefill` with the fused
/// output. Anything else keeps its native launch.
pub(super) fn sites(
    program: &DevProg,
    tensors: &[DevTensor],
    devp: &[DeviceMem],
    batch: usize,
) -> Vec<Option<Site>> {
    let count = program.gq_seg_ofs.len().saturating_sub(1);
    let mut out = vec![None; count];
    for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
        let Some(entries) = program
            .gq_stream
            .get(bounds[0] as usize..bounds[1] as usize)
        else {
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
            || kv_heads == 0
            || heads % kv_heads != 0
            || window != 0
            || nsplit != 1
            || heads == 0
            || head_dim == 0
            || head_dim % 8 != 0
            || kv_stride == 0
            || kv_stride % 8 != 0
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
        let head_bytes = u64::from(kv_stride) * u64::from(head_dim) * 2;
        let slot_bytes = u64::from(kv_heads) * head_bytes;
        let fits = |handle: u16, bytes: u64| {
            tensors
                .get(handle as usize)
                .is_some_and(|t| t.bytes >= bytes)
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
            kv_heads,
            head_bytes,
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
    /// `TAIL_ROWS` KV rows for the last tile's remainder GEMM.
    vtail: DeviceMem,
    /// (kind, m, n, pitch, head dim, query heads per GEMM batch, heads).
    plans: std::collections::HashMap<(Gemm, u32, u32, u32, u32, u32, u32), Plan>,
}

impl AttentionGemm {
    /// `max_rows` the largest routed bucket.
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        lt: Arc<Lt>,
        object: &Path,
        max_heads: u32,
        max_head_dim: u32,
        max_ctx: usize,
        max_rows: u32,
    ) -> Result<Self> {
        let config = &crate::config::RuntimeConfig::get().nv;
        let image = std::fs::read(object).map_err(|e| {
            RuntimeError::Rejected(format!("PLOW_PF_ATTN_GEMM needs {}: {e}", object.display()))
        })?;
        let module = be.module_load(&image)?;
        let abi = be.module_global_u32(&module, "plow_attn_softmax_abi")?;
        if !matches!(abi, Some(1) | Some(SOFTMAX_ABI))
            || be.module_global_u32(&module, "plow_block_attn_softmax")? != Some(BLOCK)
        {
            return Err(RuntimeError::Rejected(format!(
                "incompatible attention softmax object (ABI {SOFTMAX_ABI} expected)"
            )));
        }
        let scores_f32 = config.pf_attn_gemm_s32;
        let softmax = be.get_function(
            &module,
            if scores_f32 {
                SOFTMAX_ENTRY_F32
            } else {
                SOFTMAX_ENTRY
            },
        )?;
        let tile_rows = tile_rows(max_rows);
        let scratch = be.alloc(0, scratch_bytes(max_heads, max_ctx, max_rows))?;
        let vtail = be.alloc(0, u64::from(TAIL_ROWS) * u64::from(max_head_dim) * 2)?;
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
            vtail,
            plans: std::collections::HashMap::new(),
        })
    }

    pub(super) fn unload(self) -> Result<()> {
        self.be.module_unload(&self.module)
    }

    /// `m` counts rows of one GEMM batch: every head's rows (one KV head, `group == 1`) or one
    /// query head's (a batch of `group` query heads).
    #[allow(clippy::too_many_arguments)]
    /// Query rows of the next tile after `past` KV rows: the knob's tile, shortened while its
    /// score rows would not fit the scratch.
    fn tile(&self, heads: u32, past: u32, left: u32, element: u64) -> u32 {
        let mut tile = self.tile_rows.min(left);
        loop {
            let pitch = (past + tile).next_multiple_of(8).next_multiple_of(PITCH);
            let fit = self.scratch.len / (u64::from(heads) * u64::from(pitch) * element);
            if u64::from(tile) <= fit || tile == 1 {
                return tile;
            }
            tile = (fit as u32).max(1);
        }
    }

    fn plan(
        &mut self,
        kind: Gemm,
        m: u32,
        n: u32,
        pitch: u32,
        head_dim: u32,
        group: u32,
        heads: u32,
    ) -> Result<&Plan> {
        if self.plans.len() >= PLAN_CACHE {
            self.plans.clear();
        }
        // P is bf16 at the start of each score row: its pitch counts the f32 row in bf16 units.
        let pitch = if self.scores_f32 && kind == Gemm::Values {
            2 * pitch
        } else {
            pitch
        };
        match self
            .plans
            .entry((kind, m, n, pitch, head_dim, group, heads))
        {
            std::collections::hash_map::Entry::Occupied(e) => Ok(e.into_mut()),
            std::collections::hash_map::Entry::Vacant(e) => Ok(e.insert(self.lt.attention_plan(
                kind,
                self.scores_f32,
                m,
                n,
                pitch,
                head_dim,
                group,
                heads,
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
        let row_bytes = site.row_bytes();
        let kv_row_bytes = site.kv_row_bytes();
        let slot_rows = site.slot_rows();
        let element = if self.scores_f32 { 4 } else { 2 };
        let group = site.group();
        let mut real = 0;
        for &[q0, qlen, slot, kvlen] in requests {
            let past = kvlen.checked_sub(qlen).ok_or_else(|| {
                RuntimeError::Rejected("attention request is longer than its KV".into())
            })?;
            if kvlen > slot_rows
                || u64::from(TAIL_ROWS) * kv_row_bytes > self.vtail.len
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
                let tile = self.tile(site.heads, past + done, qlen - done, element);
                let m = tile * site.heads;
                // One GEMM batch: every head's rows, or one query head's of a KV head's group.
                let (gm, batch) = if site.kv_heads == 1 {
                    (m, 1)
                } else {
                    (tile, group)
                };
                let n = past + done + tile;
                // The padded score columns are masked: their K rows are either the request's own
                // rows below kvlen or whatever lies past it (slot_rows is 8-aligned), never read
                // back. P is zero from each row's causal limit to the pitch.
                let n8 = n.next_multiple_of(8);
                let pitch = n8.next_multiple_of(PITCH);
                if u64::from(m) * u64::from(pitch) * element > self.scratch.len {
                    return Err(RuntimeError::Rejected(
                        "attention score tile exceeds its scratch".into(),
                    ));
                }
                let offset = u64::from(q0 + done) * row_bytes;
                let scratch = self.scratch.base;
                // KV head `h`'s query heads start `h * group` heads into Q, S and O rows.
                let q_head = u64::from(group) * u64::from(site.head_dim) * 2;
                let s_head = u64::from(group) * u64::from(pitch) * element;
                for h in 0..u64::from(site.kv_heads) {
                    self.plan(
                        Gemm::Scores,
                        gm,
                        n8,
                        pitch,
                        site.head_dim,
                        batch,
                        site.heads,
                    )?
                    .matmul(
                        site.scale * LOG2_E,
                        k + h * site.head_bytes,
                        site.q + offset + h * q_head,
                        scratch + h * s_head,
                        stream,
                    )?;
                }
                let mut args = SoftmaxArgs {
                    scores: scratch,
                    rows: m,
                    cols: pitch,
                    pitch,
                    heads: site.heads,
                    first: past + done + 1,
                    pad: 0,
                };
                let mut params = [&mut args as *mut SoftmaxArgs as *mut std::ffi::c_void];
                self.be.launch_kernel(
                    self.softmax,
                    self.grid,
                    BLOCK,
                    0,
                    &mut params,
                    Some(stream),
                )?;
                // V rows below kvlen are the request's own; past it they may hold anything.
                let aligned = if n8 > kvlen { n & !(TAIL_ROWS - 1) } else { n8 };
                for h in 0..u64::from(site.kv_heads) {
                    let v = v + h * site.head_bytes;
                    let p = scratch + h * s_head;
                    let out = site.output + offset + h * q_head;
                    let hd = site.head_dim;
                    if aligned > 0 {
                        self.plan(Gemm::Values, gm, aligned, pitch, hd, batch, site.heads)?
                            .matmul(1.0, v, p, out, stream)?;
                    }
                    if aligned < n8 {
                        let rem = kvlen - aligned;
                        let vtail = self.vtail.base;
                        self.be.memset_d8_async(
                            vtail + u64::from(rem) * kv_row_bytes,
                            0,
                            (u64::from(TAIL_ROWS - rem) * kv_row_bytes) as usize,
                            stream,
                        )?;
                        self.be.memcpy_dtod_async(
                            vtail,
                            v + u64::from(aligned) * kv_row_bytes,
                            u64::from(rem) * kv_row_bytes,
                            stream,
                        )?;
                        // P is bf16 whatever the score type: column `aligned` is 2*aligned B in.
                        self.plan(Gemm::Values, gm, TAIL_ROWS, pitch, hd, batch, site.heads)?
                            .matmul_beta(
                                1.0,
                                if aligned > 0 { 1.0 } else { 0.0 },
                                vtail,
                                p + u64::from(aligned) * 2,
                                out,
                                stream,
                            )?;
                    }
                }
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
            .map(|(index, tensor)| DeviceMem::view(0x1000_0000 * (index as u64 + 1), tensor.bytes))
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
    fn selects_full_attention_only() {
        let (program, tensors, devp) = fixture();
        let found = sites(&program, &tensors, &devp, 4);
        assert!(found[0].is_none());
        let site = found[1].expect("global attention site");
        assert_eq!((site.instruction, site.heads, site.head_dim), (1, 16, 512));
        assert_eq!(site.slot_bytes, 1024 * 512 * 2);
        assert_eq!(
            (site.row_bytes(), site.kv_row_bytes(), site.slot_rows()),
            (16384, 1024, 1024)
        );
        assert_eq!((site.kv_heads, site.group()), (1, 16));

        // Two KV heads: a slot holds both heads' `[kv_stride][hd]` blocks.
        {
            let (mut program, tensors, devp) = fixture();
            program.insts[1].i[3] = 2;
            let site = sites(&program, &tensors, &devp, 2)[1].expect("GQA attention site");
            assert_eq!(
                (site.kv_heads, site.group(), site.slot_rows()),
                (2, 8, 1024)
            );
            assert_eq!(
                (site.head_bytes, site.slot_bytes),
                (1024 * 512 * 2, 2 * 1024 * 512 * 2)
            );
            assert!(sites(&program, &tensors, &devp, 4)[1].is_none());
        }

        for patch in [
            (|op: &mut DevInst64| op.i[3] = 3) as fn(&mut DevInst64),
            |op| op.i[3] = 0,
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

    #[test]
    fn scratch_is_one_tile_of_the_widest_routed_bucket() {
        // Default 2048-row tile, bf16 scores: the 26B's 16-head full layers at a 16k context.
        assert_eq!(scratch_bytes(16, 16384, 4224), 1 << 30);
        // No request of a 1024-row bucket runs a longer tile.
        assert_eq!(scratch_bytes(16, 15000, 1024), 1024 * 16 * 15104 * 2);
        // A longer context keeps the 16k scratch and runs shorter tiles.
        assert_eq!(scratch_bytes(16, 262144, 4224), 1 << 30);
    }
}
