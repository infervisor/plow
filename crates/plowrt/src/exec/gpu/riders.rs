//! Split attention for the unified token batch on sm_90a.
//!
//! Decode rows riding a packed prefill launch share its GEMMs and norms, but in the prefill
//! FlashAttention each was a one-row request scanning its whole KV on a few CTAs (0.53-0.87 ms
//! per rider on Gemma-4-12B, about a standalone decode row). Here the launch's prefill attention
//! runs only the prefill requests, and after each attention segment the riders (packed rows
//! `[0, rows)`) attend through the decode object's split-KV flash decode over a slot map and a
//! merge into the same output rows (`plow_<arch>_rider_*`, `runtime/nvidia/interp_sm120.cu`).
//! The AMD route does the same with `FlashDecode` + `FlashMerge` (Part II of
//! docs/arch/17-unified-token-batch.md).
use super::*;

/// `PlowRiderAttn` (interp_sm120.cu).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RiderArgs {
    opart: u64,
    mlpart: u64,
    q: u64,
    k: u64,
    v: u64,
    k_scale: u64,
    v_scale: u64,
    kv_len: u64,
    slot: u64,
    out: u64,
    /// Non-zero: `[rows, nsplit hd256, nsplit hd512]` on the device (a captured launch).
    dynamic: u64,
    rows: u32,
    n_head: u32,
    n_kv_head: u32,
    kv_stride: u32,
    window: u32,
    nsplit: u32,
    kv_mask: u32,
    hd: u32,
    scale: f32,
    pad: u32,
}

/// One attention layer of a prefill bucket: its rider launches follow segment `after`.
#[derive(Clone, Copy)]
struct Site {
    after: usize,
    args: RiderArgs,
}

/// Partials scratch: this many splits per slot at the widest batch; fewer riders split wider.
const SPLITS_PER_SLOT: u64 = 16;
const NSPLIT_CAP: u64 = 64;

pub(super) struct Riders {
    flash256: (KernelFn, u32),
    flash512: (KernelFn, u32),
    merge: KernelFn,
    gf512: u32,
    /// Per prefill bucket, in segment order; empty = the bucket keeps riders in its prefill FA.
    sites: Vec<Vec<Site>>,
    opart: DeviceMem,
    mlpart: DeviceMem,
    /// `kv_len[batch]` then `slot[batch]`, then the captured launches' `[rows, nsplit x 2]`.
    tables: DeviceMem,
    host: Vec<i32>,
    batch: usize,
    sms: u32,
    /// Rows riding the launch being enqueued (0 = none) and their longest KV.
    rows: u32,
    max_kv: u32,
}

fn rider_functions(be: &CudaBackend, module: &Module, arch: &str, fp8: bool) -> Option<[(KernelFn, u32); 3]> {
    let global = |name: &str| be.module_global_u32(module, name).ok().flatten();
    if global("plow_rider_fp8_kv") != Some(u32::from(fp8)) {
        return None;
    }
    let f = |name: &str| be.get_function(module, &format!("plow_{arch}_{name}")).ok();
    let (s256, s512) = (global("plow_rider_smem256")?, global("plow_rider_smem512")?);
    let (f256, f512, merge) = (f("rider_flash256")?, f("rider_flash512")?, f("rider_merge")?);
    be.set_max_dynamic_smem(f256, s256).ok()?;
    be.set_max_dynamic_smem(f512, s512).ok()?;
    Some([(f256, s256), (f512, s512), (merge, 0)])
}

impl Riders {
    /// The route for `e`'s packed prefill buckets, when its decode object carries the rider
    /// kernels and every attention site's operands match the widest decode rung's (same cache,
    /// window, ring and scale: the rows riders attend are exactly what a decode step would read).
    pub(super) fn load(e: &GpuEngine, blob: &DevBlob) -> Result<Option<Self>> {
        if !RuntimeConfig::get().nv.tb_split_attn || e.packed_prefill.is_none() || e.seg_pf.is_none() {
            return Ok(None);
        }
        let Some(decode) = blob.decode_progs().last().copied() else {
            return Ok(None);
        };
        let flash_ops = [DevOp::FlashDecode as u16, DevOp::FlashDecodeFp8 as u16];
        let decode_sites: Vec<&DevInst64> =
            decode.insts.iter().filter(|d| flash_ops.contains(&d.op)).collect();
        let Some(first) = decode_sites.first() else {
            return Ok(None);
        };
        let fp8 = first.op == DevOp::FlashDecodeFp8 as u16;
        let Some(profile) = interpreter_profile(e.be.compute_capability()) else {
            return Ok(None);
        };
        let Some([flash256, flash512, (merge, _)]) =
            rider_functions(&e.be, &e.module, profile.tag, fp8)
        else {
            tracing::info!(route = "token-batch-split-attention", ready = false, "decode object has no rider kernels");
            return Ok(None);
        };
        let gf512 = e
            .be
            .module_global_u32(&e.module, "plow_rider_gf512")?
            .unwrap_or(1)
            .max(1);
        let ptr = |h: u16| (h != TENSOR_NONE16).then(|| e.devp[h as usize].base).unwrap_or(0);
        let (mut max_heads, mut max_hd) = (0u64, 0u64);
        let mut sites = Vec::with_capacity(e.prefill.len());
        for bucket in &e.prefill {
            let Some(bucket_sites) = (|| {
                if !uses_segmented_prefill(true, false, bucket.seg_class.len(), &bucket.packet_segment_roles)
                    || !bucket.qwen_segments.is_empty()
                    || bucket.flash_sites.len() != decode_sites.len()
                {
                    return None;
                }
                let segment_of = |pc: usize| {
                    bucket.segment_sites.iter().position(|s| s.iter().any(|&(i, _)| i == pc))
                };
                let mut out = Vec::with_capacity(bucket.flash_sites.len());
                for (&pc, d) in bucket.flash_sites.iter().zip(&decode_sites) {
                    let p = &bucket.h_inst[pc];
                    let hd = p.i[6];
                    let seg = segment_of(pc)?;
                    let consumer = bucket.h_inst[pc + 1..]
                        .iter()
                        .position(|n| n.t.contains(&p.t[5]))
                        .and_then(|k| segment_of(pc + 1 + k));
                    // Riders read Q and write O after this segment: nothing later in it may touch them.
                    let shares_segment = bucket.segment_sites[seg]
                        .iter()
                        .any(|&(i, _)| i > pc && bucket.h_inst[i].t.iter().any(|&t| t == p.t[2] || t == p.t[5]));
                    let pfp8 = p.op == DevOp::FlashPrefillFp8 as u16;
                    if pfp8 != fp8
                        || d.op != first.op
                        || !matches!(hd, 256 | 512)
                        || shares_segment
                        || consumer.is_none_or(|c| c <= seg)
                        || p.t[5] == TENSOR_NONE16
                        || p.i[7] != 1
                        || (p.t[2], p.t[3], p.t[4]) != (d.t[2], d.t[3], d.t[4])
                        || (fp8 && (p.t[6], p.t[7]) != (d.t[6], d.t[7]))
                        || (p.i[2], p.i[3], p.i[5], hd) != (d.i[1], d.i[2], d.i[4], d.i[6])
                        || (p.fj[1], p.fj[2], p.fj[0]) != (d.i[3], d.i[7], d.fj[0])
                        || (hd == 512 && (d.i[1] / d.i[2].max(1)) % gf512 != 0)
                        || (hd == 256 && (d.i[1] / d.i[2].max(1)) % 2 != 0)
                    {
                        return None;
                    }
                    max_heads = max_heads.max(u64::from(d.i[1]));
                    max_hd = max_hd.max(u64::from(hd));
                    out.push(Site {
                        after: seg,
                        args: RiderArgs {
                            q: ptr(p.t[2]),
                            k: ptr(d.t[3]),
                            v: ptr(d.t[4]),
                            k_scale: if fp8 { ptr(d.t[6]) } else { 0 },
                            v_scale: if fp8 { ptr(d.t[7]) } else { 0 },
                            out: ptr(p.t[5]),
                            n_head: d.i[1],
                            n_kv_head: d.i[2],
                            kv_stride: d.i[3],
                            window: d.i[4],
                            kv_mask: d.i[7],
                            hd,
                            scale: f32::from_bits(d.fj[0]),
                            ..Default::default()
                        },
                    });
                }
                Some(out)
            })() else {
                sites.push(Vec::new());
                continue;
            };
            sites.push(bucket_sites);
        }
        if sites.iter().all(Vec::is_empty) {
            tracing::info!(route = "token-batch-split-attention", ready = false, "no bucket's attention sites qualify");
            return Ok(None);
        }
        let batch = e.batch;
        let partials = batch as u64 * SPLITS_PER_SLOT * max_heads;
        let opart = e.be.alloc(0, partials * max_hd * 4)?;
        // An hd256 site may split twice as wide in the same opart bytes.
        let mlpart = e.be.alloc(0, partials * (max_hd / 256).max(1) * 2 * 4)?;
        let tables = e.be.alloc(0, (batch * 2 * 4 + 16) as u64)?;
        tracing::info!(
            route = "token-batch-split-attention",
            ready = true,
            fp8_kv = fp8,
            buckets = sites.iter().filter(|s| !s.is_empty()).count(),
            sites = sites.iter().map(Vec::len).max().unwrap_or(0),
            "decode rows riding packed prefill launches attend through split-KV flash decode"
        );
        Ok(Some(Self {
            flash256,
            flash512,
            merge,
            gf512,
            sites,
            opart,
            mlpart,
            tables,
            host: vec![0; batch * 2 + 3],
            batch,
            sms: e.be.sm_count().max(1),
            rows: 0,
            max_kv: 0,
        }))
    }

    pub(super) fn serves(&self, bucket: usize) -> bool {
        self.sites.get(bucket).is_some_and(|s| !s.is_empty())
    }

    pub(super) fn rows(&self) -> u32 {
        self.rows
    }

    /// Arm the next launch for `riders` = `(slot, kv_len)` of packed rows `0..` and stage their
    /// tables on `stream`. `riders` empty disarms.
    pub(super) fn arm(
        &mut self,
        be: &CudaBackend,
        riders: impl ExactSizeIterator<Item = (usize, u32)>,
        stream: &CudaStream,
    ) -> Result<()> {
        let n = riders.len();
        self.rows = 0;
        self.max_kv = 0;
        if n == 0 {
            return Ok(());
        }
        if n > self.batch {
            return Err(RuntimeError::Rejected("token-batch riders exceed the slot count".into()));
        }
        for (i, (slot, kv_len)) in riders.enumerate() {
            self.host[i] = kv_len as i32;
            self.host[self.batch + i] = slot as i32;
            self.max_kv = self.max_kv.max(kv_len);
        }
        // SAFETY: `host` lives on `self` and is not resized; pageable copies are staged before
        // the call returns. Both ranges lie inside `tables`.
        self.rows = n as u32;
        let split = |hd: u32| {
            self.sites
                .iter()
                .flatten()
                .find(|s| s.args.hd == hd)
                .map_or(1, |s| self.nsplit(&s.args) as i32)
        };
        let dynamic = [n as i32, split(256), split(512)];
        self.host[2 * self.batch..].copy_from_slice(&dynamic);
        unsafe {
            be.memcpy_htod_async(self.tables.base, bytemuck::cast_slice(&self.host[..n]), stream)?;
            be.memcpy_htod_async(
                self.tables.base + (self.batch * 4) as u64,
                bytemuck::cast_slice(&self.host[self.batch..self.batch + n]),
                stream,
            )?;
            be.memcpy_htod_async(
                self.tables.base + (self.batch * 8) as u64,
                bytemuck::cast_slice(&self.host[2 * self.batch..]),
                stream,
            )?;
        }
        Ok(())
    }

    pub(super) fn disarm(&mut self) {
        self.rows = 0;
        self.max_kv = 0;
    }

    /// Splits per (row, head group) for the armed rows: enough items to fill the device, at
    /// least one 256-row tile each, within the partials scratch.
    fn nsplit(&self, a: &RiderArgs) -> u32 {
        let gf = if a.hd == 256 { 2 } else { self.gf512 };
        let groups = u64::from(self.rows) * u64::from((a.n_head / gf).max(1));
        let span = if a.window > 0 { self.max_kv.min(a.window) } else { self.max_kv };
        let row_bytes = u64::from(self.rows) * u64::from(a.n_head) * u64::from(a.hd) * 4;
        (4 * u64::from(self.sms))
            .div_ceil(groups.max(1))
            .min(u64::from(span.div_ceil(256)).max(1))
            .min(self.opart.len / row_bytes.max(1))
            .clamp(1, NSPLIT_CAP) as u32
    }

    /// As [`Self::launch_after`] for a graph capture: every size is read on the device from the
    /// launch's `arm`, so one graph serves any rider count.
    pub(super) fn launch_captured(
        &self,
        be: &CudaBackend,
        bucket: usize,
        end: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let Some(site) = self.sites[bucket].iter().find(|s| s.after + 1 == end) else {
            return Ok(());
        };
        let mut args = RiderArgs {
            opart: self.opart.base,
            mlpart: self.mlpart.base,
            kv_len: self.tables.base,
            slot: self.tables.base + (self.batch * 4) as u64,
            dynamic: self.tables.base + (self.batch * 8) as u64,
            ..site.args
        };
        let (function, smem) = if site.args.hd == 256 { self.flash256 } else { self.flash512 };
        let grid = self.sms * 4;
        let mut params = [&mut args as *mut RiderArgs as *mut std::ffi::c_void];
        be.launch_kernel(function, grid, BLOCK, smem, &mut params, Some(stream))?;
        be.launch_kernel(self.merge, grid, BLOCK, 0, &mut params, Some(stream))
    }

    /// The graph pieces of bucket `bucket`'s segments `0..segments`: each ends at a rider site.
    pub(super) fn pieces(&self, bucket: usize, segments: usize) -> Vec<(usize, usize)> {
        split_after(self.sites[bucket].iter().map(|s| s.after), segments)
    }

    /// Segments `start..end` (a graph piece between routed attention segments) cut after each
    /// rider site inside it.
    pub(super) fn split(&self, bucket: usize, start: usize, end: usize) -> Vec<(usize, usize)> {
        let afters = self.sites[bucket].iter().map(|s| s.after).filter(|&a| a >= start && a + 1 < end);
        split_after(afters.map(|a| a - start), end - start)
            .into_iter()
            .map(|(a, b)| (a + start, b + start))
            .collect()
    }

    /// The rider attention of the site whose segment ends at `end`, if any.
    pub(super) fn launch_after(
        &self,
        be: &CudaBackend,
        bucket: usize,
        end: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let Some(site) = self.sites[bucket].iter().find(|s| s.after + 1 == end) else {
            return Ok(());
        };
        let rows = self.rows;
        let a = &site.args;
        let gf = if a.hd == 256 { 2 } else { self.gf512 };
        let groups = u64::from(rows) * u64::from((a.n_head / gf).max(1));
        let nsplit = self.nsplit(a);
        let mut args = RiderArgs {
            opart: self.opart.base,
            mlpart: self.mlpart.base,
            kv_len: self.tables.base,
            slot: self.tables.base + (self.batch * 4) as u64,
            rows,
            nsplit,
            ..*a
        };
        let (function, smem) = if a.hd == 256 { self.flash256 } else { self.flash512 };
        let items = groups * u64::from(nsplit);
        let grid = items.min(u64::from(self.sms) * 4) as u32;
        let mut params = [&mut args as *mut RiderArgs as *mut std::ffi::c_void];
        be.launch_kernel(function, grid, BLOCK, smem, &mut params, Some(stream))?;
        let merges = u64::from(rows) * u64::from(a.n_head);
        let grid = merges.min(u64::from(self.sms) * 4) as u32;
        be.launch_kernel(self.merge, grid, BLOCK, 0, &mut params, Some(stream))
    }
}

/// Segments `0..segments` cut after each of `afters` (ascending): the graph pieces between which
/// the riders attend.
fn split_after(afters: impl Iterator<Item = usize>, segments: usize) -> Vec<(usize, usize)> {
    let mut pieces = Vec::new();
    let mut start = 0;
    for after in afters {
        pieces.push((start, after + 1));
        start = after + 1;
    }
    if start < segments {
        pieces.push((start, segments));
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::split_after;

    #[test]
    fn pieces_end_at_each_attention_segment_and_cover_the_chain() {
        assert_eq!(split_after([2, 5].into_iter(), 9), [(0, 3), (3, 6), (6, 9)]);
        assert_eq!(split_after([0, 8].into_iter(), 9), [(0, 1), (1, 9)]);
        assert_eq!(split_after(std::iter::empty(), 4), [(0, 4)]);
    }

    #[test]
    fn rider_args_match_the_device_layout() {
        assert_eq!(std::mem::size_of::<super::RiderArgs>(), 11 * 8 + 10 * 4);
    }
}
