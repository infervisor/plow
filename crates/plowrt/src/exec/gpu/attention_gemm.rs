//! Vendor-GEMM prefill attention (`PLOW_PF_ATTN_GEMM`) for full-attention layers with ONE KV
//! head: per query-row tile, `S = Q.K^T` (cuBLASLt), an in-place causal softmax
//! (`attn_softmax_sm90a.cubin`), `O = P.V` (cuBLASLt). Runtime-side: the packet is unchanged,
//! the `FlashPrefill` segment is simply not launched.
//!
//! One KV head makes both products plain 2-D GEMMs over the packet's own layouts: Q and the
//! output are `[row][head][hd]`, a slot's K/V are `[row][hd]`.
//!
//! A launch's tiles (every request of the pack, every tile of each) form GROUPS. The grouped
//! path (`begin_launch` / `run_site`) uploads one table per launch — the groups' shapes, the
//! per-site matrix pointers, and the byte ranges to zero — and serves each routed segment with
//! one grouped `Q.K^T`, one grouped softmax and one grouped `P.V` per wave (a wave is the
//! prefix of groups whose score tiles fit the scratch together). The shapes live in device
//! arrays a grouped layout binds at creation, so the plans never depend on the launch.
//! `run` is the per-request form, kept for launches the grouped path cannot take.

use super::*;
use crate::asset::devblob::{DevProg, DevTensor};
use crate::device::cuda::lt::{AttentionGemm as Gemm, GroupedAttention, GroupedPlan, Lt, Plan};

pub(super) const SOFTMAX_OBJECT: &str = "attn_softmax_sm90a.cubin";
/// ABI 2 adds the grouped entries; an ABI-1 object serves the per-request route only.
const SOFTMAX_ABI: u32 = 2;
/// The warp-per-row entries (coalesced); the thread-per-row ones stay in the object for A/Bs.
const SOFTMAX_ENTRY: &str = "plow_attn_softmax_w";
const SOFTMAX_ENTRY_F32: &str = "plow_attn_softmax_w_f32";
const SOFTMAX_GROUPED_ENTRY: &str = "plow_attn_softmax_gw";
const SOFTMAX_GROUPED_ENTRY_F32: &str = "plow_attn_softmax_gw_f32";
const ZERO_ENTRY: &str = "plow_attn_zero";
/// Scores are produced in the log2 domain so the kernel's exp is the hardware exp2.
const LOG2_E: f32 = std::f32::consts::LOG2_E;
/// Score rows are padded to this many columns: the tail GEMM below reads P in 128-column
/// pieces, and the kernel loads whole 8-element vectors.
const PITCH: u32 = 128;
/// cuBLASLt's BF16 kernels want the KV extent 8-aligned (measured 2.6x slower otherwise). V
/// rows past a request's `kvlen` may hold anything, so the last tile's `P.V` stops at a
/// 128-aligned column and finishes on a zero-padded copy of the remaining rows, this many.
/// (The grouped path zeroes those V rows instead.)
const TAIL_ROWS: u32 = 128;
const PLAN_CACHE: usize = 4096;
/// A grouped layout binds its shape arrays at creation, so each wave index owns a fixed
/// block of the launch table; a launch needing more waves takes the per-request path.
const WAVES: usize = 8;
/// Pinned staging buffers a launch table upload rotates through.
const STAGING: usize = 4;
/// Shape arrays per wave block: hd, m, n8, ld_s, ld_p.
const DIMS: usize = 5;

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
        (self.slot_bytes / self.kv_row_bytes()) as u32
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

#[repr(C)]
struct SoftmaxGroupedArgs {
    scores: u64,
    rows: u64,
    cols: u64,
    ld: u64,
    first: u64,
    row_start: u64,
    groups: u32,
    total_rows: u32,
    heads: u32,
    pad: u32,
}

#[repr(C)]
struct ZeroArgs {
    ptr: u64,
    bytes: u64,
    count: u32,
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

/// One score tile of one request: query rows `[row0, row0 + tile)` of the launch.
#[derive(Clone, Copy)]
struct Group {
    row0: u32,
    slot: u32,
    m: u32,
    n8: u32,
    pitch: u32,
    first: u32,
    scratch: u64,
}

struct Wave {
    groups: std::ops::Range<usize>,
    rows: u32,
    m_bucket: u32,
    n_bucket: u32,
    /// Table offsets of the wave's score pointers, causal starts and row prefix sums.
    s_off: u64,
    first_off: u64,
    row_start_off: u64,
}

struct Launch {
    sites: Vec<Site>,
    waves: Vec<Wave>,
    /// Table offset of each site's per-wave `[q | k | v | o]` pointer arrays.
    site_off: Vec<Vec<u64>>,
}

struct Staging {
    host: PinnedHost,
    done: CudaEvent,
    pending: bool,
}

/// Little-endian writer over the pinned staging buffer, mirroring the device table.
struct Table<'a> {
    bytes: &'a mut [u8],
    end: usize,
}

impl Table<'_> {
    fn put_i32(&mut self, at: usize, values: impl IntoIterator<Item = i32>) -> Result<usize> {
        let mut at = at;
        for v in values {
            self.bytes
                .get_mut(at..at + 4)
                .ok_or_else(|| RuntimeError::Rejected("attention launch table overflow".into()))?
                .copy_from_slice(&v.to_le_bytes());
            at += 4;
        }
        self.end = self.end.max(at);
        Ok(at)
    }

    fn put_u64(&mut self, at: usize, values: impl IntoIterator<Item = u64>) -> Result<usize> {
        let mut at = at;
        for v in values {
            self.bytes
                .get_mut(at..at + 8)
                .ok_or_else(|| RuntimeError::Rejected("attention launch table overflow".into()))?
                .copy_from_slice(&v.to_le_bytes());
            at += 8;
        }
        self.end = self.end.max(at);
        Ok(at)
    }
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
    plans: std::collections::HashMap<(Gemm, u32, u32, u32, u32), Plan>,
    /// Grouped path (`PLOW_PF_ATTN_GEMM_GROUPED`, grouped cuBLASLt matmuls, an ABI-2 object):
    /// its softmax and zero entries.
    grouped: Option<[KernelFn; 2]>,
    max_groups: usize,
    /// `[WAVES x DIMS x max_groups] i32` shape blocks, then the per-launch pointer tables.
    table: DeviceMem,
    dims_bytes: usize,
    staging: Vec<Staging>,
    staging_next: usize,
    grouped_plans: std::collections::HashMap<GroupedKey, GroupedPlan>,
    launch: Option<Launch>,
    /// `PLOW_PF_SEG_TIME`: one event per phase boundary of every grouped wave run.
    seg_time: bool,
    phases: Vec<[CudaEvent; 4]>,
}

/// (kind, wave, m bucket, n bucket, head dim, alpha bits): the wave fixes the shape arrays,
/// the buckets steer the heuristic.
type GroupedKey = (Gemm, usize, u32, u32, u32, u32);

impl AttentionGemm {
    /// `max_sites` routed segments per launch, `batch` slots, `max_rows` the largest bucket.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        lt: Arc<Lt>,
        object: &Path,
        max_heads: u32,
        max_head_dim: u32,
        max_ctx: usize,
        max_sites: usize,
        batch: usize,
        max_rows: u32,
    ) -> Result<Self> {
        let config = &crate::config::RuntimeConfig::get().nv;
        let image = std::fs::read(object).map_err(|e| {
            RuntimeError::Rejected(format!(
                "PLOW_PF_ATTN_GEMM needs {}: {e}",
                object.display()
            ))
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
        let kernel_groups = be
            .module_global_u32(&module, "plow_attn_softmax_max_groups")?
            .unwrap_or(0) as usize;
        let scores_f32 = config.pf_attn_gemm_s32;
        let softmax = be.get_function(
            &module,
            if scores_f32 { SOFTMAX_ENTRY_F32 } else { SOFTMAX_ENTRY },
        )?;
        let tile_rows = config.pf_attn_gemm_tile.max(1);
        let pitch = (max_ctx as u64).next_multiple_of(u64::from(PITCH));
        let element = if scores_f32 { 4 } else { 2 };
        let scratch = be.alloc(0, u64::from(tile_rows) * u64::from(max_heads) * pitch * element)?;
        let vtail = be.alloc(0, u64::from(TAIL_ROWS) * u64::from(max_head_dim) * 2)?;
        let grouped = if config.pf_attn_gemm_grouped && lt.has_grouped() && kernel_groups > 0 {
            let entry = if scores_f32 {
                SOFTMAX_GROUPED_ENTRY_F32
            } else {
                SOFTMAX_GROUPED_ENTRY
            };
            Some([be.get_function(&module, entry)?, be.get_function(&module, ZERO_ENTRY)?])
        } else {
            None
        };
        if config.pf_attn_gemm_grouped && grouped.is_none() {
            tracing::warn!(
                "PLOW_PF_ATTN_GEMM_GROUPED needs grouped cuBLASLt matmuls and softmax object ABI \
                 {SOFTMAX_ABI}; per-request launches"
            );
        }
        // Every request adds at most one partial tile to the launch's whole tiles.
        let max_groups = (batch + max_rows.div_ceil(tile_rows) as usize)
            .clamp(1, kernel_groups.max(1));
        let dims_bytes = WAVES * DIMS * max_groups * 4;
        let pointer_bytes = WAVES * (max_groups * 16 + 16)
            + max_sites * WAVES * max_groups * 32
            + max_sites * (batch + 1) * 16;
        let table = be.alloc(0, (dims_bytes + pointer_bytes) as u64)?;
        let mut staging = Vec::with_capacity(STAGING);
        for _ in 0..STAGING {
            staging.push(Staging {
                host: be.host_alloc_pinned(dims_bytes + pointer_bytes)?,
                done: be.event_create(false)?,
                pending: false,
            });
        }
        tracing::info!(
            object = %object.display(),
            tile_rows,
            scores_f32,
            grouped = grouped.is_some(),
            max_groups,
            table_kib = (dims_bytes + pointer_bytes) >> 10,
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
            grouped,
            max_groups,
            table,
            dims_bytes,
            staging,
            staging_next: 0,
            grouped_plans: std::collections::HashMap::new(),
            launch: None,
            seg_time: config.pf_seg_time,
            phases: Vec::new(),
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
        match self.plans.entry((kind, m, n, pitch, head_dim)) {
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
        let row_bytes = site.row_bytes();
        let kv_row_bytes = site.kv_row_bytes();
        let slot_rows = site.slot_rows();
        let element = if self.scores_f32 { 4 } else { 2 };
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
                let tile = self.tile_rows.min(qlen - done);
                let m = tile * site.heads;
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
                self.plan(Gemm::Scores, m, n8, pitch, site.head_dim)?.matmul(
                    site.scale * LOG2_E,
                    k,
                    site.q + offset,
                    scratch,
                    stream,
                )?;
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
                self.be
                    .launch_kernel(self.softmax, self.grid, BLOCK, 0, &mut params, Some(stream))?;
                let out = site.output + offset;
                // V rows below kvlen are the request's own; past it they may hold anything.
                let aligned = if n8 > kvlen { n & !(TAIL_ROWS - 1) } else { n8 };
                if aligned > 0 {
                    self.plan(Gemm::Values, m, aligned, pitch, site.head_dim)?.matmul(
                        1.0,
                        v,
                        scratch,
                        out,
                        stream,
                    )?;
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
                    // P is bf16 whatever the score type: column `aligned` sits 2*aligned bytes in.
                    self.plan(Gemm::Values, m, TAIL_ROWS, pitch, site.head_dim)?.matmul_beta(
                        1.0,
                        if aligned > 0 { 1.0 } else { 0.0 },
                        vtail,
                        scratch + u64::from(aligned) * 2,
                        out,
                        stream,
                    )?;
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

    /// The launch's tiles in request order, and the tensor rows every request must fit.
    fn groups(
        &self,
        site: &Site,
        requests: &[Request],
        rows: u32,
    ) -> Result<Option<(Vec<Group>, u32)>> {
        let element = if self.scores_f32 { 4 } else { 2 };
        let slot_rows = site.slot_rows();
        let mut groups = Vec::new();
        let mut real = 0;
        for &[q0, qlen, slot, kvlen] in requests {
            let past = kvlen.checked_sub(qlen).ok_or_else(|| {
                RuntimeError::Rejected("attention request is longer than its KV".into())
            })?;
            if kvlen > slot_rows || q0.checked_add(qlen).is_none_or(|end| end > rows) {
                return Err(RuntimeError::Rejected(
                    "attention request exceeds its tensors".into(),
                ));
            }
            let mut done = 0;
            while done < qlen {
                let tile = self.tile_rows.min(qlen - done);
                let m = tile * site.heads;
                let n8 = (past + done + tile).next_multiple_of(8);
                let pitch = n8.next_multiple_of(PITCH);
                if u64::from(m) * u64::from(pitch) * element > self.scratch.len {
                    return Err(RuntimeError::Rejected(
                        "attention score tile exceeds its scratch".into(),
                    ));
                }
                if groups.len() == self.max_groups {
                    return Ok(None);
                }
                groups.push(Group {
                    row0: q0 + done,
                    slot,
                    m,
                    n8,
                    pitch,
                    first: past + done + 1,
                    scratch: 0,
                });
                done += tile;
            }
            real = real.max(q0 + qlen);
        }
        Ok(Some((groups, real)))
    }

    /// Stage the launch table for `sites` (in segment order) and zero the KV rows every
    /// group's `P.V` reads past its request, then the output rows past the last request.
    /// `false` = the launch is not groupable; `run` serves its sites instead.
    pub(super) fn begin_launch(
        &mut self,
        sites: &[Site],
        requests: &[Request],
        rows: u32,
        stream: &CudaStream,
    ) -> Result<bool> {
        self.launch = None;
        let Some(first) = sites.first() else {
            return Ok(false);
        };
        let geometry = |s: &Site| (s.heads, s.head_dim, s.slot_bytes);
        let Some([_, zero]) = self.grouped else {
            return Ok(false);
        };
        if requests.is_empty()
            || sites.iter().any(|s| geometry(s) != geometry(first))
        {
            return Ok(false);
        }
        let Some((mut groups, real)) = self.groups(first, requests, rows)? else {
            return Ok(false);
        };
        let element = if self.scores_f32 { 4 } else { 2 };
        // Waves: prefixes of groups whose score tiles fit the scratch together.
        let mut waves: Vec<Wave> = Vec::new();
        let mut start = 0;
        let mut used = 0u64;
        for index in 0..groups.len() {
            let bytes = u64::from(groups[index].m) * u64::from(groups[index].pitch) * element;
            if used + bytes > self.scratch.len {
                waves.push(Wave {
                    groups: start..index,
                    rows: 0,
                    m_bucket: 0,
                    n_bucket: 0,
                    s_off: 0,
                    first_off: 0,
                    row_start_off: 0,
                });
                start = index;
                used = 0;
            }
            groups[index].scratch = self.scratch.base + used;
            used += bytes;
        }
        waves.push(Wave {
            groups: start..groups.len(),
            rows: 0,
            m_bucket: 0,
            n_bucket: 0,
            s_off: 0,
            first_off: 0,
            row_start_off: 0,
        });
        if waves.len() > WAVES {
            return Ok(false);
        }

        let slot = self.staging_next;
        self.staging_next = (slot + 1) % STAGING;
        let staging = &mut self.staging[slot];
        if std::mem::take(&mut staging.pending) {
            self.be.event_synchronize(&staging.done)?;
        }
        let mut table = Table {
            bytes: staging.host.as_mut_slice(),
            end: 0,
        };
        let head_dim = first.head_dim as i32;
        // P is bf16 at the start of each score row: its pitch counts an f32 row in bf16 units.
        let ld_scale: u32 = if self.scores_f32 { 2 } else { 1 };
        let maxg = self.max_groups;
        for (w, wave) in waves.iter_mut().enumerate() {
            let block = w * DIMS * maxg * 4;
            let members = &groups[wave.groups.clone()];
            let column = |f: &dyn Fn(&Group) -> i32, pad: i32| {
                members
                    .iter()
                    .map(f)
                    .chain(std::iter::repeat(pad))
                    .take(maxg)
                    .collect::<Vec<_>>()
            };
            // Groups past the launch's are empty (m = 0) with valid, 8-row operands.
            table.put_i32(block, std::iter::repeat_n(head_dim, maxg))?;
            table.put_i32(block + maxg * 4, column(&|g| g.m as i32, 0))?;
            table.put_i32(block + 2 * maxg * 4, column(&|g| g.n8 as i32, 8))?;
            table.put_i32(block + 3 * maxg * 4, column(&|g| g.pitch as i32, 8))?;
            table.put_i32(
                block + 4 * maxg * 4,
                column(&|g| (g.pitch * ld_scale) as i32, 8),
            )?;
            wave.rows = members.iter().map(|g| g.m).sum();
            wave.m_bucket = (wave.rows / members.len().max(1) as u32).next_power_of_two();
            wave.n_bucket = members
                .iter()
                .map(|g| g.n8)
                .max()
                .unwrap_or(8)
                .next_power_of_two();
        }
        let mut at = self.dims_bytes;
        let spare = self.scratch.base;
        // Every pointer array has `max_groups` entries: the empty groups point at the scratch.
        let padded = |values: Vec<u64>| {
            values
                .into_iter()
                .chain(std::iter::repeat(spare))
                .take(maxg)
        };
        for wave in &mut waves {
            let members = &groups[wave.groups.clone()];
            wave.s_off = at as u64;
            at = table.put_u64(at, padded(members.iter().map(|g| g.scratch).collect()))?;
            wave.first_off = at as u64;
            at = table.put_i32(at, members.iter().map(|g| g.first as i32))?;
            wave.row_start_off = at as u64;
            let mut sum = 0i32;
            at = table.put_i32(
                at,
                std::iter::once(0).chain(members.iter().map(|g| {
                    sum += g.m as i32;
                    sum
                })),
            )?;
            at = at.next_multiple_of(8);
        }
        let mut site_off = Vec::with_capacity(sites.len());
        for site in sites {
            let mut offsets = Vec::with_capacity(waves.len());
            for wave in &waves {
                let members = &groups[wave.groups.clone()];
                offsets.push(at as u64);
                let pointers: [Vec<u64>; 4] = [
                    members.iter().map(|g| site.q + u64::from(g.row0) * site.row_bytes()).collect(),
                    members.iter().map(|g| site.k + u64::from(g.slot) * site.slot_bytes).collect(),
                    members.iter().map(|g| site.v + u64::from(g.slot) * site.slot_bytes).collect(),
                    members
                        .iter()
                        .map(|g| site.output + u64::from(g.row0) * site.row_bytes())
                        .collect(),
                ];
                for values in pointers {
                    at = table.put_u64(at, padded(values))?;
                }
            }
            site_off.push(offsets);
        }
        // Byte ranges to zero: V rows [kvlen, kvlen8) of every request (read by the last tile's
        // P.V against P = 0; the pad rows of a launch overwrite the last request's with finite
        // values before its attention runs), and the output rows past the last request.
        let mut ranges: Vec<(u64, i32)> = Vec::new();
        for site in sites {
            let kv_row_bytes = site.kv_row_bytes();
            let slot_rows = site.slot_rows();
            for &[_, _, slot, kvlen] in requests {
                let kvlen8 = kvlen.next_multiple_of(8).min(slot_rows);
                if kvlen8 > kvlen {
                    let v = site.v + u64::from(slot) * site.slot_bytes;
                    ranges.push((
                        v + u64::from(kvlen) * kv_row_bytes,
                        (u64::from(kvlen8 - kvlen) * kv_row_bytes) as i32,
                    ));
                }
            }
            if real < rows {
                ranges.push((
                    site.output + u64::from(real) * site.row_bytes(),
                    (u64::from(rows - real) * site.row_bytes()) as i32,
                ));
            }
        }
        let zero_ptr = at as u64;
        at = table.put_u64(at, ranges.iter().map(|r| r.0))?;
        let zero_bytes = at as u64;
        at = table.put_i32(at, ranges.iter().map(|r| r.1))?;
        let end = at.next_multiple_of(8);
        table.end = table.end.max(end);
        let bytes = table.end;

        // SAFETY: the pinned buffer stays allocated for the engine's lifetime and is not
        // rewritten before `done` (recorded below) has completed.
        unsafe {
            self.be.memcpy_htod_async(
                self.table.base,
                &staging.host.as_slice()[..bytes],
                stream,
            )?;
        }
        self.be.event_record(&staging.done, stream)?;
        staging.pending = true;
        if !ranges.is_empty() {
            let mut args = ZeroArgs {
                ptr: self.table.base + zero_ptr,
                bytes: self.table.base + zero_bytes,
                count: ranges.len() as u32,
                pad: 0,
            };
            let mut params = [&mut args as *mut ZeroArgs as *mut std::ffi::c_void];
            self.be.launch_kernel(
                zero,
                self.be.sm_count() * 2,
                BLOCK,
                0,
                &mut params,
                Some(stream),
            )?;
        }
        self.launch = Some(Launch {
            sites: sites.to_vec(),
            waves,
            site_off,
        });
        Ok(true)
    }

    fn grouped_plan(&mut self, key: GroupedKey) -> Result<()> {
        if self.grouped_plans.len() >= PLAN_CACHE {
            self.grouped_plans.clear();
        }
        if self.grouped_plans.contains_key(&key) {
            return Ok(());
        }
        let (kind, wave, m_bucket, n_bucket, head_dim, alpha) = key;
        let maxg = self.max_groups as u64;
        let block = self.table.base + (wave * DIMS * self.max_groups * 4) as u64;
        let plan = self.lt.grouped_attention_plan(
            kind,
            self.scores_f32,
            f32::from_bits(alpha),
            &GroupedAttention {
                groups: self.max_groups as u32,
                head_dim,
                hd: block,
                m: block + maxg * 4,
                n: block + 2 * maxg * 4,
                ld_s: block + 3 * maxg * 4,
                ld_p: block + 4 * maxg * 4,
                average_m: m_bucket,
                average_n: n_bucket,
            },
        )?;
        self.grouped_plans.insert(key, plan);
        Ok(())
    }

    /// The grouped GEMM of `key` on the staged wave. Its first run times every heuristic
    /// candidate on the real operands (the shapes are a bucket, not an exact size, so the
    /// heuristic's first pick is a guess: measured 15-20% slower than the exact-shape kernel)
    /// and keeps the fastest; the buffers then hold that kernel's result.
    fn grouped_run(
        &mut self,
        key: GroupedKey,
        a: u64,
        w: u64,
        c: u64,
        stream: &CudaStream,
    ) -> Result<()> {
        self.grouped_plan(key)?;
        let plan = self.grouped_plans.get_mut(&key).expect("cached above");
        if plan.candidates() == 0 {
            return plan.run(a, w, c, stream);
        }
        let start = self.be.event_create(true)?;
        let end = self.be.event_create(true)?;
        let mut times = Vec::with_capacity(plan.candidates());
        let mut best: Option<(f32, usize)> = None;
        for index in 0..plan.candidates() {
            let mut ms = f32::INFINITY;
            for _ in 0..2 {
                self.be.event_record(&start, stream)?;
                if let Err(e) = plan.run_candidate(index, a, w, c, stream) {
                    self.be.stream_synchronize(stream)?;
                    if e.is_fatal() {
                        return Err(e);
                    }
                    tracing::warn!(error = %e, index, "grouped attention candidate rejected");
                    break;
                }
                self.be.event_record(&end, stream)?;
                self.be.event_synchronize(&end)?;
                ms = ms.min(self.be.event_elapsed_ms(&start, &end)?);
            }
            times.push(ms);
            if best.is_none_or(|(b, _)| ms < b) {
                best = Some((ms, index));
            }
        }
        let (ms, index) = best.filter(|(ms, _)| ms.is_finite()).ok_or_else(|| {
            RuntimeError::Device("no runnable grouped attention candidate".into())
        })?;
        plan.select(index);
        tracing::info!(
            ?key,
            index,
            ms = format!("{ms:.3}").as_str(),
            candidates = ?times,
            "grouped attention algorithm measured"
        );
        plan.run(a, w, c, stream)
    }

    /// Phase times `[qk, softmax, pv]` ms of every wave run since the last call (the caller
    /// has synchronized the stream).
    pub(super) fn phase_report(&mut self) -> Result<Vec<[f32; 3]>> {
        let mut out = Vec::with_capacity(self.phases.len());
        for events in self.phases.drain(..) {
            out.push([
                self.be.event_elapsed_ms(&events[0], &events[1])?,
                self.be.event_elapsed_ms(&events[1], &events[2])?,
                self.be.event_elapsed_ms(&events[2], &events[3])?,
            ]);
        }
        Ok(out)
    }

    /// Routed segment `index` (the order `begin_launch` saw its sites) of the staged launch.
    pub(super) fn run_site(&mut self, index: usize, stream: &CudaStream) -> Result<()> {
        let launch = self.launch.take().ok_or_else(|| {
            RuntimeError::Rejected("attention route: no staged launch".into())
        })?;
        let result = self.run_site_of(&launch, index, stream);
        self.launch = Some(launch);
        result
    }

    fn run_site_of(&mut self, launch: &Launch, index: usize, stream: &CudaStream) -> Result<()> {
        let [softmax_grouped, _] = self.grouped.expect("a staged launch is grouped");
        let site = launch.sites[index];
        let maxg = self.max_groups as u64;
        let table = self.table.base;
        for (w, wave) in launch.waves.iter().enumerate() {
            let groups = wave.groups.len() as u64;
            let scores = (
                Gemm::Scores,
                w,
                wave.m_bucket,
                wave.n_bucket,
                site.head_dim,
                (site.scale * LOG2_E).to_bits(),
            );
            let values = (
                Gemm::Values,
                w,
                wave.m_bucket,
                wave.n_bucket,
                site.head_dim,
                1.0f32.to_bits(),
            );
            let block = table + (w * DIMS * self.max_groups * 4) as u64;
            let q = table + launch.site_off[index][w];
            let k = q + maxg * 8;
            let v = k + maxg * 8;
            let o = v + maxg * 8;
            let s = table + wave.s_off;
            let timed = if self.seg_time {
                let events = [
                    self.be.event_create(true)?,
                    self.be.event_create(true)?,
                    self.be.event_create(true)?,
                    self.be.event_create(true)?,
                ];
                self.be.event_record(&events[0], stream)?;
                Some(events)
            } else {
                None
            };
            self.grouped_run(scores, q, k, s, stream)?;
            if let Some(events) = &timed {
                self.be.event_record(&events[1], stream)?;
            }
            let mut args = SoftmaxGroupedArgs {
                scores: s,
                rows: block + maxg * 4,
                cols: block + 2 * maxg * 4,
                ld: block + 3 * maxg * 4,
                first: table + wave.first_off,
                row_start: table + wave.row_start_off,
                groups: groups as u32,
                total_rows: wave.rows,
                heads: site.heads,
                pad: 0,
            };
            let mut params = [&mut args as *mut SoftmaxGroupedArgs as *mut std::ffi::c_void];
            self.be.launch_kernel(
                softmax_grouped,
                self.grid,
                BLOCK,
                0,
                &mut params,
                Some(stream),
            )?;
            if let Some(events) = &timed {
                self.be.event_record(&events[2], stream)?;
            }
            self.grouped_run(values, s, v, o, stream)?;
            if let Some(events) = timed {
                self.be.event_record(&events[3], stream)?;
                self.phases.push(events);
            }
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
        assert_eq!((site.row_bytes(), site.kv_row_bytes(), site.slot_rows()), (16384, 1024, 1024));

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

    #[test]
    fn launch_table_writer_is_bounded() {
        let mut bytes = [0u8; 32];
        let mut table = Table {
            bytes: &mut bytes,
            end: 0,
        };
        assert_eq!(table.put_i32(0, [1, -2]).unwrap(), 8);
        assert_eq!(table.put_u64(8, [3]).unwrap(), 16);
        assert_eq!(table.end, 16);
        assert!(table.put_u64(24, [1, 2]).is_err());
        assert_eq!(&bytes[..16], &[1, 0, 0, 0, 254, 255, 255, 255, 3, 0, 0, 0, 0, 0, 0, 0]);
    }
}
