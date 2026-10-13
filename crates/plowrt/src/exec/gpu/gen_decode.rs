//! The packet's generated flash-decode object on the ordinary decode step. devgen isolates a
//! decode program's `FlashDecode` + `FlashMerge` pair into two single-instruction segments and
//! records the generated decode role on both (`gen_kernels::bind_decode`); the runtime replaces
//! the pair with one launch of the object between the interpreter windows (split-KV partials
//! merged in-kernel, the split width chosen on the host from the step's widest kv length), or
//! runs the untouched interpreter pair as one merged window when the rung is not routed.
use super::gen_role::{GenDecode, GenDecodeArgs};
use super::*;
use bytemuck::cast_slice;
use plow_asset::segment_roles::GeneratedAbi;
use std::sync::atomic::{AtomicU32, Ordering};

/// A decode program's generated attention segments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GenSeg {
    /// The attention instruction and the merge instruction the object folds in.
    Decode { inst: usize, merge: usize },
    /// The merge's own segment: nothing to launch.
    Merge(usize),
    /// A lone `QwenGdnStep` (devgen PLOW_QWEN_GDN_ISOLATE): the decode object's GDN step kernel
    /// at its own occupancy grid instead of the interpreter window's.
    Gdn(usize),
}

/// The one instruction segment `seg` holds, if it holds exactly one and all of it.
fn lone(g: &DevProg, seg: usize) -> Option<usize> {
    let inst = g
        .stream
        .iter()
        .chain(&g.gq_stream)
        .find(|e| e.seg as usize == seg)?
        .inst;
    g.stream
        .iter()
        .chain(&g.gq_stream)
        .all(|e| (e.seg as usize == seg) == (e.inst == inst))
        .then_some(inst as usize)
}

pub(super) fn is_generated_decode(
    role: u8,
    objects: &std::collections::BTreeMap<u8, SegmentObject>,
) -> bool {
    plow_asset::segment_roles::is_generated(role)
        && objects
            .get(&role)
            .and_then(|object| GeneratedAbi::parse(&object.abi))
            .is_some_and(|abi| abi.decode())
}

/// The program's generated decode segments: every role segment is a lone `FlashDecode` at the
/// rung's rows over a linear bf16 KV (no slot table), followed by the lone `FlashMerge` of its
/// partials.
pub(super) fn decode_segments(
    g: &DevProg,
    roles: &[u8],
    objects: &std::collections::BTreeMap<u8, SegmentObject>,
) -> Result<Vec<Option<GenSeg>>> {
    let reject = |why: &str| RuntimeError::Rejected(format!("generated decode segments: {why}"));
    let count = g
        .gq_seg_ofs
        .len()
        .checked_sub(1)
        .ok_or_else(|| reject("no segment table"))?;
    if roles.len() != count {
        return Err(reject("role count differs from the segment table"));
    }
    let mut out = vec![None; count];
    let mut pending: Option<(usize, usize)> = None;
    for (seg, &role) in roles.iter().enumerate() {
        if !is_generated_decode(role, objects) {
            continue;
        }
        let inst =
            lone(g, seg).ok_or_else(|| reject("role segment holds more than one instruction"))?;
        let d = &g.insts[inst];
        match (DevOp::from_u16(d.op), pending) {
            (Some(DevOp::FlashDecode), None) => {
                if d.i[0] != g.t
                    || d.i[2] == 0
                    || d.i[1] % d.i[2] != 0
                    || d.t[5] == TENSOR_NONE16
                    || d.t[6] != TENSOR_NONE16
                    || d.t[7] != TENSOR_NONE16
                {
                    return Err(reject("FlashDecode operands"));
                }
                pending = Some((seg, inst));
            }
            (Some(DevOp::FlashMerge), Some((decode_seg, decode))) => {
                let a = &g.insts[decode];
                if d.t[1] != a.t[0]
                    || d.t[2] != a.t[1]
                    || d.i[0] != a.i[0]
                    || d.i[1] != a.i[1]
                    || d.i[3] != a.i[6]
                {
                    return Err(reject(
                        "FlashMerge does not merge the preceding FlashDecode",
                    ));
                }
                out[decode_seg] = Some(GenSeg::Decode {
                    inst: decode,
                    merge: inst,
                });
                out[seg] = Some(GenSeg::Merge(inst));
                pending = None;
            }
            (op, pending) => {
                return Err(reject(&format!(
                    "role segments must pair a FlashDecode with its FlashMerge (segment {seg} \
                     holds {op:?} (op {}), pending decode segment {:?})",
                    d.op,
                    pending.map(|p| p.0)
                )))
            }
        }
    }
    if pending.is_some() {
        return Err(reject("FlashDecode without its FlashMerge"));
    }
    Ok(out)
}

/// Marks every lone `QwenGdnStep` segment at the rung's rows (row b steps slot b, gated by
/// `in.active`) for the GDN step kernel.
pub(super) fn mark_gdn_steps(g: &DevProg, segments: &mut [Option<GenSeg>]) {
    for (seg, site) in segments.iter_mut().enumerate() {
        if site.is_some() {
            continue;
        }
        let Some(inst) = lone(g, seg) else { continue };
        let d = &g.insts[inst];
        if DevOp::from_u16(d.op) == Some(DevOp::QwenGdnStep)
            && d.i[4] == g.t
            && d.t[7] != TENSOR_NONE16
        {
            *site = Some(GenSeg::Gdn(inst));
        }
    }
}

/// Mirrors `PlowGdnRide` (interp_sm120.cu).
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct GdnStepArgs {
    pub(super) out: u64,
    pub(super) qkv: u64,
    pub(super) a: u64,
    pub(super) b: u64,
    pub(super) alog: u64,
    pub(super) dt_bias: u64,
    pub(super) state: u64,
    pub(super) table: u64,
    pub(super) n: u32,
    pub(super) tstride: u32,
    pub(super) hk: u32,
    pub(super) hv: u32,
    pub(super) kdim: u32,
    pub(super) vdim: u32,
    pub(super) alog_f32: u32,
    pub(super) pad: u32,
    pub(super) scale: f32,
    pub(super) eps: f32,
    pub(super) active: u64,
}

#[derive(Clone, Copy)]
pub(super) struct GdnStepKernel {
    pub(super) function: KernelFn,
    pub(super) grid: u32,
    pub(super) block: u32,
    pub(super) smem: u32,
}

/// One isolated decode GDN step.
pub(super) struct GdnStepRoute {
    be: Arc<CudaBackend>,
    kernel: GdnStepKernel,
    args: GdnStepArgs,
}

impl GdnStepRoute {
    pub(super) fn run(&self, stream: &CudaStream) -> Result<()> {
        let mut args = self.args;
        let mut params = [&mut args as *mut GdnStepArgs as *mut std::ffi::c_void];
        self.be.launch_kernel_grid(
            self.kernel.function,
            [self.kernel.grid, 1, 1],
            self.kernel.block,
            self.kernel.smem,
            &mut params,
            Some(stream),
        )
    }

    pub(super) fn function(&self) -> KernelFn {
        self.kernel.function
    }
}

/// `(segment, instruction)` of every instruction the routes execute (for `ordered_waits_for`).
pub(super) fn instructions(segments: &[Option<GenSeg>]) -> Vec<(usize, usize)> {
    segments
        .iter()
        .enumerate()
        .filter_map(|(seg, site)| {
            site.map(|site| match site {
                GenSeg::Decode { inst, .. } | GenSeg::Merge(inst) | GenSeg::Gdn(inst) => {
                    (seg, inst)
                }
            })
        })
        .collect()
}

/// Shared by every routed rung: partial buffers for the widest rung at the widest split, the
/// arrival counters (the kernel leaves them at zero) and the identity slot table (decode row b
/// holds KV slot b).
pub(super) struct GenDecodeLaunch {
    function: KernelFn,
    block: u32,
    smem: u32,
    head_dim: u32,
    gqa: u32,
    kv_tile: u32,
    nsplit_max: u32,
    pub(super) kv_layout: u32,
    sms: u32,
    opart: DeviceMem,
    mlpart: DeviceMem,
    ctr: DeviceMem,
    slots: DeviceMem,
    /// The step's widest fed kv length, stored before the launch (`patch_nv_nsplit`).
    pub(super) max_kv: AtomicU32,
    /// `PLOW_GEN_DECODE_PIN_NSPLIT`: the split count ignores `max_kv` (static launch arguments).
    pin_nsplit: bool,
}

impl GenDecodeLaunch {
    pub(super) fn new(
        be: &Arc<CudaBackend>,
        g: &GenDecode,
        rows: u32,
        n_head: u32,
        n_kv_head: u32,
    ) -> Result<Self> {
        let splits = u64::from(rows) * u64::from(n_head) * u64::from(g.nsplit_max);
        let ctr_bytes = u64::from(rows) * u64::from(n_kv_head) * 4;
        let ctr = be.alloc(0, ctr_bytes)?;
        be.memset_d8(ctr.base, 0, ctr_bytes as usize)?;
        let slots = be.alloc(0, u64::from(rows) * 4)?;
        let table: Vec<i32> = (0..rows as i32).collect();
        be.memcpy_htod(slots.base, cast_slice(&table))?;
        Ok(Self {
            function: g.function,
            block: g.block,
            smem: g.smem,
            head_dim: g.head_dim,
            gqa: g.gqa,
            kv_tile: g.kv_tile,
            nsplit_max: g.nsplit_max,
            kv_layout: g.kv_layout,
            sms: be.sm_count().max(1),
            opart: be.alloc(0, splits * u64::from(g.head_dim) * 4)?,
            mlpart: be.alloc(0, splits * 2 * 4)?,
            ctr,
            slots,
            max_kv: AtomicU32::new(1),
            pin_nsplit: crate::config::RuntimeConfig::get().nv.gen_decode_pin_nsplit,
        })
    }

    /// The split count is static (`PLOW_GEN_DECODE_PIN_NSPLIT`): a routed rung can be captured.
    pub(super) fn pinned(&self) -> bool {
        self.pin_nsplit
    }

    /// Splits per (row, kv head): cover the SMs, at most two KV tiles per split (the policy the
    /// catalog classes were tuned with, build_catalog.py `pick_nsplit`; riders use the same).
    /// Pinned: the SM cover alone (a split past the tiles writes an empty partial, weight 0).
    fn nsplit(&self, rows: u32, n_kv_head: u32, window: u32) -> u32 {
        let cover = self
            .sms
            .div_ceil((rows * n_kv_head).max(1))
            .clamp(1, self.nsplit_max);
        if self.pin_nsplit {
            return cover;
        }
        let max_kv = self.max_kv.load(Ordering::Relaxed).max(1);
        let span = if window > 0 {
            max_kv.min(window)
        } else {
            max_kv
        };
        let tiles = span.div_ceil(self.kv_tile.max(1)).max(1);
        cover.min((tiles / 2).max(1))
    }
}

/// One routed rung's launch at one attention site.
pub(super) struct GenDecodeRoute {
    be: Arc<CudaBackend>,
    launch: Arc<GenDecodeLaunch>,
    args: GenDecodeArgs,
}

impl GenDecodeRoute {
    pub(super) fn run(&self, stream: &CudaStream) -> Result<()> {
        let mut args = self.args;
        args.nsplit = self.launch.nsplit(args.rows, args.n_kv_head, args.window);
        let mut params = [&mut args as *mut GenDecodeArgs as *mut std::ffi::c_void];
        self.be.launch_kernel_grid(
            self.launch.function,
            [args.n_kv_head * args.nsplit, args.rows, 1],
            self.launch.block,
            self.launch.smem,
            &mut params,
            Some(stream),
        )
    }

    pub(super) fn function(&self) -> KernelFn {
        self.launch.function
    }
}

/// The routes of one decode program: the object on each FlashDecode segment (writing the
/// merge's output), a skip on each FlashMerge segment.
pub(super) fn routes(
    be: &Arc<CudaBackend>,
    launch: &Arc<GenDecodeLaunch>,
    g: &DevProg,
    segments: &[Option<GenSeg>],
    devp: &[DeviceMem],
    gdn: Option<GdnStepKernel>,
) -> Result<Vec<Option<LibraryRoute>>> {
    let ptr = |t: u16| devp[t as usize].base;
    segments
        .iter()
        .map(|site| match *site {
            None => Ok(None),
            Some(GenSeg::Merge(_)) => Ok(Some(LibraryRoute::Skip)),
            Some(GenSeg::Gdn(inst)) => {
                let kernel = gdn.ok_or_else(|| {
                    RuntimeError::Rejected("GDN step segment without the step kernel".into())
                })?;
                let d = &g.insts[inst];
                Ok(Some(LibraryRoute::GdnStep(GdnStepRoute {
                    be: Arc::clone(be),
                    kernel,
                    args: GdnStepArgs {
                        out: ptr(d.t[0]),
                        qkv: ptr(d.t[1]),
                        a: ptr(d.t[2]),
                        b: ptr(d.t[3]),
                        alog: ptr(d.t[4]),
                        dt_bias: ptr(d.t[5]),
                        state: ptr(d.t[6]),
                        table: 0,
                        n: d.i[4],
                        tstride: 0,
                        hk: d.i[0],
                        hv: d.i[1],
                        kdim: d.i[2],
                        vdim: d.i[3],
                        alog_f32: d.i[5],
                        pad: 0,
                        scale: f32::from_bits(d.fj[0]),
                        eps: f32::from_bits(d.fj[1]),
                        active: ptr(d.t[7]),
                    },
                })))
            }
            Some(GenSeg::Decode { inst, merge }) => {
                let d = &g.insts[inst];
                if d.i[6] != launch.head_dim || d.i[1] != d.i[2] * launch.gqa {
                    return Err(RuntimeError::Rejected(
                        "generated decode route: site geometry differs from the object".into(),
                    ));
                }
                // The body stages whole KV tiles (rows past kv_len are masked after the load):
                // a slot stride the tile divides keeps the last tile inside the slot's rows.
                if d.i[3] % launch.kv_tile.max(1) != 0 {
                    return Err(RuntimeError::Rejected(
                        "generated decode route: KV stride is not a multiple of the object's KV tile"
                            .into(),
                    ));
                }
                Ok(Some(LibraryRoute::Gen(GenDecodeRoute {
                    be: Arc::clone(be),
                    launch: Arc::clone(launch),
                    args: GenDecodeArgs {
                        opart: launch.opart.base,
                        mlpart: launch.mlpart.base,
                        q: ptr(d.t[2]),
                        k: ptr(d.t[3]),
                        v: ptr(d.t[4]),
                        kv_len: ptr(d.t[5]),
                        slot: launch.slots.base,
                        out: ptr(g.insts[merge].t[0]),
                        ctr: launch.ctr.base,
                        rows: d.i[0],
                        n_head: d.i[1],
                        n_kv_head: d.i[2],
                        kv_stride: d.i[3],
                        window: d.i[4],
                        nsplit: 1,
                        kv_mask: d.i[7],
                        hd: d.i[6],
                        scale: f32::from_bits(d.fj[0]),
                        pad: launch.kv_layout,
                    },
                })))
            }
        })
        .collect()
}
