use super::*;
use crate::asset::devblob::{DevProg, DevTensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DecodeSegment {
    pub(super) instruction: usize,
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) output_bytes: u64,
    pub(super) input_bytes: u64,
    pub(super) weight_bytes: u64,
}

pub(super) struct CublasLtDecodeRoute {
    plan: Arc<ProjectionPlan>,
    input: u64,
    weight: u64,
    output: u64,
    tail: Option<HeadTail>,
}

/// The light head kernel and its scratch: an lm_head whose N is not a multiple of 16 runs its
/// first `N & !15` columns on cuBLASLt into the scratch (aligned pitch), then the kernel copies
/// them into the logits and computes the rest.
#[derive(Clone)]
pub(super) struct HeadKernel {
    pub(super) be: Arc<CudaBackend>,
    pub(super) function: KernelFn,
    pub(super) scratch: Arc<DeviceMem>,
    /// Zeroed per-row argmax state for a folded Argmax/ArgmaxFin: `best` u64 then `ctr` u32,
    /// `CUBLASLT_DECODE_MAX_ROWS` each. The kernel re-zeroes what it uses.
    pub(super) amax: Option<Arc<DeviceMem>>,
}

struct HeadTail {
    kernel: HeadKernel,
    rows: u32,
    n: u32,
    k: u32,
    n0: u32,
    /// The folded ArgmaxFin's ids, or 0.
    ids: u64,
}

/// A decode segment holding only the lm_head's greedy Argmax + ArgmaxFin, computed by the head
/// kernel instead (`plow_<arch>_light_head` with ids): the segment launches nothing.
#[derive(Clone, Copy)]
pub(super) struct ArgmaxFold {
    segment: usize,
    head_segment: usize,
    argmax: usize,
    fin: usize,
    ids: u64,
}

/// An lm_head `segment` that cuBLASLt serves through the head kernel (see HeadTail).
fn head_tail(segment: &DecodeSegment, backend: &ProjectionBackend, head: Option<&HeadKernel>) -> Option<HeadKernel> {
    head.filter(|_| segment.n % 16 != 0 && segment.n > 16 && matches!(backend, ProjectionBackend::Lt(_)))
        .filter(|h| u64::from(segment.m) * u64::from(segment.n & !15) * 2 <= h.scratch.len)
        .cloned()
}

/// The Argmax/ArgmaxFin segments of `g` that directly follow a head-kernel lm_head and read
/// exactly its logits.
pub(super) fn argmax_folds(
    g: &DevProg,
    segments: &[Option<DecodeSegment>],
    devp: &[DeviceMem],
    backend: &ProjectionBackend,
    head: Option<&HeadKernel>,
) -> Vec<ArgmaxFold> {
    if !crate::config::RuntimeConfig::get().nv.decode_head_argmax || head.is_none_or(|h| h.amax.is_none()) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        let Some(segment) = segment else { continue };
        if head_tail(segment, backend, head).is_none()
            || segment.m > plow_asset::segment_roles::CUBLASLT_DECODE_MAX_ROWS
            || segments.get(index + 1).copied().flatten().is_some()
            || index + 2 >= g.gq_seg_ofs.len()
        {
            continue;
        }
        let seg = index + 1;
        let mut insts: Vec<usize> = Vec::new();
        for e in &g.gq_stream[g.gq_seg_ofs[seg] as usize..g.gq_seg_ofs[seg + 1] as usize] {
            if !insts.contains(&(e.inst as usize)) {
                insts.push(e.inst as usize);
            }
        }
        let [a, f] = insts[..] else { continue };
        let (h, da, df) = (&g.insts[segment.instruction], &g.insts[a], &g.insts[f]);
        if da.op == DevOp::Argmax as u16
            && df.op == DevOp::ArgmaxFin as u16
            && da.t[1] == h.t[0]
            && da.i[0] == segment.n
            && da.i[1] == segment.m
            && df.t[1] == da.t[0]
            && df.i[1] == segment.m
        {
            out.push(ArgmaxFold { segment: seg, head_segment: index, argmax: a, fin: f, ids: devp[df.t[0] as usize].base });
        }
    }
    out
}

/// `(segment, instruction)` of every instruction a folded argmax segment skips.
pub(super) fn fold_instructions(folds: &[ArgmaxFold]) -> Vec<(usize, usize)> {
    folds.iter().flat_map(|f| [(f.segment, f.argmax), (f.segment, f.fin)]).collect()
}

impl CublasLtDecodeRoute {
    pub(super) fn run(&self, stream: &CudaStream) -> Result<()> {
        let Some(tail) = &self.tail else {
            return self.plan.run(self.input, self.weight, self.output, stream);
        };
        let scratch = tail.kernel.scratch.base;
        self.plan.run(self.input, self.weight, scratch, stream)?;
        let (mut c, mut x, mut w, mut src) = (self.output, self.input, self.weight, scratch);
        let (mut n, mut k, mut n0) = (tail.n, tail.k, tail.n0);
        let (mut ids, mut best, mut ctr) = match (&tail.kernel.amax, tail.ids) {
            (Some(amax), ids) if ids != 0 => (
                ids,
                amax.base,
                amax.base + 8 * u64::from(plow_asset::segment_roles::CUBLASLT_DECODE_MAX_ROWS),
            ),
            _ => (0, 0, 0),
        };
        let mut params = [
            &mut c as *mut u64 as *mut std::ffi::c_void,
            &mut x as *mut u64 as *mut std::ffi::c_void,
            &mut w as *mut u64 as *mut std::ffi::c_void,
            &mut src as *mut u64 as *mut std::ffi::c_void,
            &mut n as *mut u32 as *mut std::ffi::c_void,
            &mut k as *mut u32 as *mut std::ffi::c_void,
            &mut n0 as *mut u32 as *mut std::ffi::c_void,
            &mut ids as *mut u64 as *mut std::ffi::c_void,
            &mut best as *mut u64 as *mut std::ffi::c_void,
            &mut ctr as *mut u64 as *mut std::ffi::c_void,
        ];
        tail.kernel
            .be
            .launch_kernel(tail.kernel.function, tail.rows * 8, BLOCK, 0, &mut params, Some(stream))
    }

    /// Executed by the previous segment's paired call.
    pub(super) fn folded(&self) -> bool {
        matches!(*self.plan, ProjectionPlan::Folded)
    }
}

/// A host library call that replaces one decode segment.
pub(super) enum LibraryRoute {
    Projection(CublasLtDecodeRoute),
    Moe(moe_lt::MoeLtRoute),
    Light(LightRoute),
}

impl LibraryRoute {
    pub(super) fn run(&self, stream: &CudaStream) -> Result<()> {
        match self {
            Self::Projection(route) => route.run(stream),
            Self::Moe(route) => route.run(stream),
            Self::Light(route) => route.run(stream),
        }
    }
}

/// An interpreter segment run as ordinary launches of the decode object's light kernels: one
/// per level of independent instructions, one block per packet slice, no claim loop or gates.
pub(super) struct LightRoute {
    be: Arc<CudaBackend>,
    kernarg: DevProgram,
    launches: Vec<LightLaunch>,
    _scratch: Option<Arc<DeviceMem>>,
}

/// `PlowLightX` (interp_sm120.cu): HeadNormRope `inst[j]` reads x at `base + col[j]`, row
/// pitch `row` elements (a fused q|k|v projection's output); `base` 0 = no override.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct LightX {
    base: u64,
    row: u32,
    inst: [u32; 3],
    col: [u32; 3],
}

/// Which light kernel a launch runs, with its extra parameters.
#[derive(Clone, Copy)]
enum LightKind {
    /// The one-instruction `plow_<arch>_light`.
    Single,
    NormQuant,
    /// `light_attn` over this many instructions.
    Attn(u32),
    /// `light_flash`: the q, k, v HeadNormRope instructions it folds in, or all `!0`.
    Flash([u32; 4]),
}

/// `PlowLightOp` (light ABI 2): an instruction with its tensor pointers resolved at load.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LightOp {
    d: DevInst64,
    t: [u64; 8],
    fold: [u64; 2],
}

/// `PlowLightSpan` (light ABI 2): the `light_attn` instructions of one launch.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LightSpan {
    count: u32,
    x_row: [u32; 4],
    fused: u32,
    op: [LightOp; 4],
}

impl LightOp {
    fn resolve(d: &DevInst64, devp: &[DeviceMem]) -> Self {
        let at = |h: u16| {
            if h == packet::dev::TENSOR_NONE16 {
                0
            } else {
                devp[h as usize].base
            }
        };
        // `fj[2]` names the merge fold's tensors on FlashDecode only.
        let fold = if d.op == DevOp::FlashDecode as u16 { d.fj[2] } else { 0 };
        Self {
            d: *d,
            t: d.t.map(at),
            fold: if fold == 0 {
                [0; 2]
            } else {
                [at(fold as u16), at((fold >> 16) as u16)]
            },
        }
    }
}

struct LightLaunch {
    function: KernelFn,
    xs: LightX,
    kind: LightKind,
    /// Light ABI 2: the launch's instructions by value (`Single` = `op[0]`).
    direct: Option<Box<LightSpan>>,
    instruction: u32,
    blocks: u32,
    block: u32,
    smem: u32,
}

impl LightRoute {
    pub(super) fn run(&self, stream: &CudaStream) -> Result<()> {
        for launch in &self.launches {
            if let Some(span) = &launch.direct {
                let mut span = **span;
                let mut op = span.op[0];
                let mut params = [match launch.kind {
                    LightKind::Single => &mut op as *mut LightOp as *mut std::ffi::c_void,
                    _ => &mut span as *mut LightSpan as *mut std::ffi::c_void,
                }];
                self.be.launch_kernel(
                    launch.function,
                    launch.blocks,
                    launch.block,
                    launch.smem,
                    &mut params,
                    Some(stream),
                )?;
                continue;
            }
            let mut arg = self.kernarg;
            let mut instruction = launch.instruction;
            let (mut count, mut hnr) = match launch.kind {
                LightKind::Attn(n) => (n, [0; 4]),
                LightKind::Flash(h) => (0, h),
                LightKind::Single | LightKind::NormQuant => (0, [0; 4]),
            };
            let mut xs = launch.xs;
            let mut params = [
                &mut arg as *mut DevProgram as *mut std::ffi::c_void,
                &mut instruction as *mut u32 as *mut std::ffi::c_void,
                match launch.kind {
                    LightKind::Flash(_) => &mut hnr as *mut [u32; 4] as *mut std::ffi::c_void,
                    _ => &mut count as *mut u32 as *mut std::ffi::c_void,
                },
                &mut xs as *mut LightX as *mut std::ffi::c_void,
            ];
            let params = match launch.kind {
                LightKind::Single | LightKind::NormQuant => &mut params[..2],
                _ => &mut params[..],
            };
            self.be.launch_kernel(
                launch.function,
                launch.blocks,
                launch.block,
                launch.smem,
                params,
                Some(stream),
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct LightFunctions {
    single: KernelFn,
    norm_quant: Option<KernelFn>,
    /// `plow_light_gemma`: the single kernel also runs NormResidual(Norm) and GluStrided.
    gemma: bool,
    fp8_attn: bool,
    /// `plow_light_abi` 2: instructions and tensor pointers are passed by value.
    direct: bool,
    /// `plow_<arch>_light_tail`: SoftCap / Argmax / ArgmaxFin (light ABI 2).
    tail: Option<KernelFn>,
    /// `plow_<arch>_light_capmax`: SoftCap and the batched Argmax in one pass.
    capmax: Option<KernelFn>,
    /// `plow_<arch>_light_attn`, its arena and the head dims it carries.
    attn: Option<(KernelFn, u32, [u32; 2])>,
    /// `plow_<arch>_light_attn_s`: the hd256 attention alone, two blocks per SM.
    attn_s: Option<KernelFn>,
    fp8_flash256: Option<(KernelFn, u32)>,
    /// `plow_<arch>_light_head` (unaligned lm_head tail).
    pub(super) head: Option<KernelFn>,
    /// `plow_<arch>_light_flash` (streamed hd128 FlashDecode) and its smem.
    flash: Option<(KernelFn, u32)>,
    /// `plow_<arch>_light_pf`: a prefill bucket's lone RmsNorm / Residual / SiLU Glu.
    pub(super) prefill: Option<KernelFn>,
}

/// `FA_ST_THREADS`: eight row-group warps and the producer warp.
const FLASH_BLOCK: u32 = BLOCK + 32;

/// The light kernels of the decode object `module` (`<stem>` = `interp_<arch>`), when it has
/// them and `PLOW_DECODE_LIGHT` is on. `smem` = the object's arena, which `light_attn` takes.
pub(super) fn light_functions(
    be: &CudaBackend,
    module: &Module,
    stem: &str,
    smem: u32,
) -> Result<Option<LightFunctions>> {
    let abi = be.module_global_u32(module, "plow_light_abi")?;
    if !crate::config::RuntimeConfig::get().nv.decode_light || !matches!(abi, Some(1 | 2)) {
        return Ok(None);
    }
    let arch = stem.trim_start_matches("interp_");
    let single = be.get_function(module, &format!("plow_{arch}_light"))?;
    let norm_quant = be.get_function(module, &format!("plow_{arch}_light_norm_quant")).ok();
    let attn = match be.module_global_u32(module, "plow_light_attn_hd")? {
        Some(hd) => {
            let f = be.get_function(module, &format!("plow_{arch}_light_attn"))?;
            be.set_max_dynamic_smem(f, smem)?;
            let hd2 = be.module_global_u32(module, "plow_light_attn_hd2")?.unwrap_or(hd);
            Some((f, smem, [hd, hd2]))
        }
        _ => None,
    };
    let gemma = be.module_global_u32(module, "plow_light_gemma")? == Some(1);
    let fp8_attn = abi == Some(2) && be.module_global_u32(module, "plow_light_fp8_attn")? == Some(1);
    let fp8_flash256 = match be.module_global_u32(module, "plow_light_fp8_flash256_smem")? {
        Some(bytes) if fp8_attn => {
            let f = be.get_function(module, &format!("plow_{arch}_light_fp8_flash256"))?;
            be.set_max_dynamic_smem(f, bytes)?;
            Some((f, bytes))
        }
        _ => None,
    };
    let head = be.get_function(module, &format!("plow_{arch}_light_head")).ok();
    let flash = match be.module_global_u32(module, "plow_light_flash_smem")? {
        Some(bytes) if crate::config::RuntimeConfig::get().nv.decode_light_flash => {
            let f = be.get_function(module, &format!("plow_{arch}_light_flash"))?;
            be.set_max_dynamic_smem(f, bytes)?;
            Some((f, bytes))
        }
        _ => None,
    };
    let prefill = be
        .get_function(module, &format!("plow_{arch}_light_pf"))
        .ok()
        .filter(|_| crate::config::RuntimeConfig::get().nv.prefill_light);
    let tail = if abi == Some(2) {
        be.get_function(module, &format!("plow_{arch}_light_tail")).ok()
    } else {
        None
    };
    let capmax = if abi == Some(2) {
        be.get_function(module, &format!("plow_{arch}_light_capmax")).ok()
    } else {
        None
    };
    let attn_s = match be.get_function(module, &format!("plow_{arch}_light_attn_s")) {
        Ok(f) if crate::config::RuntimeConfig::get().nv.decode_light_attn_s => {
            be.set_max_dynamic_smem(f, ATTN_S_SMEM)?;
            Some(f)
        }
        _ => None,
    };
    Ok(Some(LightFunctions { single, norm_quant, gemma, fp8_attn, direct: abi == Some(2), tail, capmax, attn, attn_s, fp8_flash256, head, flash, prefill }))
}

fn fp8_attention_tail(d: &packet::dev::DevInst64) -> bool {
    let none = packet::dev::TENSOR_NONE16;
    match DevOp::from_u16(d.op) {
        Some(DevOp::QuantFp8) => d.t[3] == none && d.t[4] == none,
        Some(DevOp::FlashMerge) => matches!(d.i[3], 256 | 512) && d.t[3] == none && d.t[7] == none,
        _ => false,
    }
}

fn norm_quant_pair(g: &DevProg, entries: &[packet::dev::StreamEnt]) -> Option<(usize, usize)> {
    let first = entries.first()?;
    let n = first.inst as usize;
    let q = n + 1;
    let (Some(norm), Some(quant)) = (g.insts.get(n), g.insts.get(q)) else { return None };
    let blocks = usize::from(norm.blocks);
    if blocks == 0 || entries.len() != 2 * blocks {
        return None;
    }
    let none = packet::dev::TENSOR_NONE16;
    let grouped = entries[..blocks].iter().enumerate().all(|(slice, e)| {
        e.inst as usize == n && e.slice as usize == slice
    }) && entries[blocks..].iter().enumerate().all(|(slice, e)| {
        e.inst as usize == q && e.slice as usize == slice
    });
    let interleaved = entries.chunks_exact(2).enumerate().all(|(slice, pair)| {
        pair[0].inst as usize == n && pair[1].inst as usize == q
            && pair[0].slice as usize == slice && pair[1].slice as usize == slice
    });
    (quant.blocks == norm.blocks
        && norm.op == DevOp::NormResidualNorm as u16
        && quant.op == DevOp::QuantFp8 as u16
        && norm.i[0] == u32::from(norm.blocks)
        && norm.i[0] == quant.i[0]
        && norm.i[1] == quant.i[1]
        && norm.i[1] == 3840
        && norm.t[..4].iter().all(|&t| t != none)
        && quant.t[..3].iter().all(|&t| t != none)
        && quant.t[1] == norm.t[0]
        && quant.t[3] == none
        && quant.t[4] == none
        && (grouped || interleaved))
    .then_some((n, q))
}

/// Each light-routable interpreter segment of `g`: `(segment, levels)`, a level being a
/// contiguous instruction range whose members do not wait on each other. A lone AddNorm or Glu
/// takes the one-instruction kernel; a segment of HeadNormRope / FlashDecode at the object's
/// head dim takes `light_attn`.
pub(super) fn light_segments(
    g: &DevProg,
    library: &[Option<DecodeSegment>],
    functions: &LightFunctions,
) -> Vec<(usize, Vec<(usize, usize)>)> {
    let waits_of = |inst: usize| -> Vec<u32> {
        g.gq_stream
            .iter()
            .filter(|e| e.inst as usize == inst)
            .flat_map(|e| &g.waits[e.wait_ofs as usize..e.wait_ofs as usize + e.wait_len as usize])
            .map(|w| w.id)
            .collect()
    };
    (0..g.gq_seg_ofs.len().saturating_sub(1))
        .filter(|&seg| library.get(seg).copied().flatten().is_none())
        .filter_map(|seg| {
            let entries = &g.gq_stream[g.gq_seg_ofs[seg] as usize..g.gq_seg_ofs[seg + 1] as usize];
            if g.t == 128 && functions.direct && functions.norm_quant.is_some() {
                if let Some((n, q)) = norm_quant_pair(g, entries) {
                    return Some((seg, vec![(n, 1), (q, 1)]));
                }
            }
            let mut insts: Vec<usize> = Vec::new();
            for e in entries {
                if insts.last() != Some(&(e.inst as usize)) {
                    insts.push(e.inst as usize);
                }
            }
            // Every instruction's entries are contiguous and cover its slices once, in order.
            let whole = insts.iter().all(|&inst| {
                let d = &g.insts[inst];
                let own: Vec<_> = entries.iter().filter(|e| e.inst as usize == inst).collect();
                d.blocks != 0
                    && own.len() == d.blocks as usize
                    && own.iter().enumerate().all(|(i, e)| e.slice as usize == i)
            });
            let distinct = insts.iter().collect::<std::collections::BTreeSet<_>>().len() == insts.len();
            if !whole || !distinct {
                return None;
            }
            let op = |inst: usize| DevOp::from_u16(g.insts[inst].op);
            // The logits tail (SoftCap -> Argmax -> ArgmaxFin), one launch each.
            if functions.gemma
                && functions.tail.is_some()
                && insts.iter().all(|&i| {
                    matches!(op(i), Some(DevOp::SoftCap | DevOp::Argmax | DevOp::ArgmaxFin))
                })
            {
                return Some((seg, insts.iter().map(|&i| (i, 0)).collect()));
            }
            if let [inst] = insts[..] {
                let gemma_op = matches!(
                    op(inst),
                    Some(DevOp::NormResidual | DevOp::NormResidualNorm | DevOp::GluStrided)
                );
                if matches!(op(inst), Some(DevOp::AddNorm | DevOp::Glu))
                    || (functions.gemma && gemma_op)
                {
                    return Some((seg, vec![(inst, 0)]));
                }
            }
            let attn_hd = |hd: u32| functions.attn.is_some_and(|(_, _, hds)| hds.contains(&hd));
            let attn = functions.attn.is_some()
                && insts.iter().all(|&inst| {
                    let d = &g.insts[inst];
                    match op(inst) {
                        Some(DevOp::HeadNormRope) => attn_hd(d.i[2]) && d.i[5] == 0,
                        Some(DevOp::HeadNormRopeFp8) => {
                            functions.fp8_attn && attn_hd(d.i[2]) && d.i[5] == 0
                                && d.t[6] != packet::dev::TENSOR_NONE16
                                && d.t[7] == packet::dev::TENSOR_NONE16
                        }
                        Some(DevOp::FlashDecode) => attn_hd(d.i[6]),
                        Some(DevOp::FlashDecodeFp8) => {
                            functions.fp8_attn && attn_hd(d.i[6])
                                && d.t[6] != packet::dev::TENSOR_NONE16
                                && d.t[7] != packet::dev::TENSOR_NONE16
                        }
                        _ => functions.fp8_attn && fp8_attention_tail(d),
                    }
                });
            if !attn
                || (insts.iter().any(|&i| fp8_attention_tail(&g.insts[i]))
                    && !insts.iter().any(|&i| matches!(op(i), Some(DevOp::FlashDecode | DevOp::FlashDecodeFp8))))
            {
                return None;
            }
            let split_fp8_attention = functions.fp8_attn
                && insts.iter().any(|&i| fp8_attention_tail(&g.insts[i]));
            let mut levels: Vec<(usize, usize)> = Vec::new();
            for &inst in &insts {
                if fp8_attention_tail(&g.insts[inst]) {
                    levels.push((inst, 0));
                    continue;
                }
                if split_fp8_attention {
                    levels.push((inst, 1));
                    continue;
                }
                // A by-value span (light ABI 2) carries at most 4 instructions.
                let joins = levels.last().is_some_and(|&(lo, n)| {
                    lo + n == inst
                        && !(functions.direct && n >= 4)
                        && waits_of(inst).iter().all(|&w| !(lo..lo + n).contains(&(w as usize)))
                });
                match levels.last_mut() {
                    Some((_, n)) if joins => *n += 1,
                    _ => levels.push((inst, 1)),
                }
            }
            Some((seg, levels))
        })
        .collect()
}

/// A prefill bucket's interpreter segments of at most three RmsNorm / Residual / SiLU Glu
/// instructions (`plow_<arch>_light_pf`, one launch each in program order), skipping `library`
/// segments.
pub(super) fn prefill_light_segments(g: &DevProg, library: &[Option<usize>]) -> Vec<(usize, Vec<usize>)> {
    (0..g.gq_seg_ofs.len().saturating_sub(1))
        .filter(|&seg| library.get(seg).copied().flatten().is_none())
        .filter_map(|seg| {
            let entries = &g.gq_stream[g.gq_seg_ofs[seg] as usize..g.gq_seg_ofs[seg + 1] as usize];
            let mut insts: Vec<usize> = Vec::new();
            for e in entries {
                if insts.last() != Some(&(e.inst as usize)) {
                    insts.push(e.inst as usize);
                }
            }
            let none = packet::dev::TENSOR_NONE16;
            let ok = (1..=3).contains(&insts.len())
                && insts.windows(2).all(|w| w[0] < w[1])
                && insts.iter().all(|&inst| {
                    let d = &g.insts[inst];
                    let own: Vec<_> = entries.iter().filter(|e| e.inst as usize == inst).collect();
                    d.blocks != 0
                        && own.len() == d.blocks as usize
                        && own.iter().enumerate().all(|(i, e)| e.slice as usize == i)
                        && match DevOp::from_u16(d.op) {
                            Some(DevOp::RmsNorm) => d.t[3] == none && d.t[4] == none,
                            Some(DevOp::Residual) => true,
                            Some(DevOp::Glu) => d.i[1] == 1,
                            _ => false,
                        }
                });
            ok.then_some((seg, insts))
        })
        .collect()
}

pub(super) fn prefill_glu_quant_segments(g: &DevProg) -> Vec<(usize, Vec<usize>)> {
    g.gq_seg_ofs
        .windows(2)
        .enumerate()
        .filter_map(|(seg, w)| {
            let entries = g.gq_stream.get(w[0] as usize..w[1] as usize)?;
            let inst = entries.first()?.inst as usize;
            let d = g.insts.get(inst)?;
            let matches = d.op == DevOp::QuantFp8 as u16
                && d.i[0] >= 32
                && d.i[1] == 15360
                && d.i[2] == 0
                && d.t[..5].iter().all(|&t| t != packet::dev::TENSOR_NONE16)
                && d.blocks != 0
                && entries.len() == usize::from(d.blocks)
                && entries
                    .iter()
                    .enumerate()
                    .all(|(slice, e)| e.inst as usize == inst && e.slice as usize == slice);
            matches.then_some((seg, vec![inst]))
        })
        .collect()
}

/// Segments that are exactly one plain (no GLU) QuantFp8 over a 4096- or 8192-wide row, all
/// slices: `plow_quant_cached` runs it one row per CTA.
pub(super) fn prefill_plain_quant_segments(g: &DevProg) -> Vec<(usize, Vec<usize>)> {
    let none = packet::dev::TENSOR_NONE16;
    g.gq_seg_ofs
        .windows(2)
        .enumerate()
        .filter_map(|(seg, w)| {
            let entries = g.gq_stream.get(w[0] as usize..w[1] as usize)?;
            let inst = entries.first()?.inst as usize;
            let d = g.insts.get(inst)?;
            let matches = d.op == DevOp::QuantFp8 as u16
                && d.i[0] >= 32
                && matches!(d.i[1], 4096 | 8192)
                && d.i[2] == 0
                && d.t[..3].iter().all(|&t| t != none)
                && d.t[3] == none
                && d.t[4] == none
                && d.blocks != 0
                && entries.len() == usize::from(d.blocks)
                && entries
                    .iter()
                    .enumerate()
                    .all(|(slice, e)| e.inst as usize == inst && e.slice as usize == slice);
            matches.then_some((seg, vec![inst]))
        })
        .collect()
}

/// One light route running `insts` of a prefill bucket, in order.
pub(super) fn prefill_light_route(
    be: &Arc<CudaBackend>,
    function: KernelFn,
    kernarg: DevProgram,
    g: &DevProg,
    insts: &[usize],
) -> LightRoute {
    LightRoute {
        be: Arc::clone(be),
        kernarg,
        launches: insts
            .iter()
            .map(|&inst| LightLaunch {
                function,
                xs: LightX::default(),
                kind: LightKind::Single,
                direct: None,
                instruction: inst as u32,
                blocks: u32::from(g.insts[inst].blocks),
                block: BLOCK,
                smem: 0,
            })
            .collect(),
        _scratch: None,
    }
}

/// The cached GLU quant kernel strides rows by `gridDim`: one CTA per row instead of the
/// instruction's one per SM, which walked ~31 rows of a 4096-row chunk with its load and store
/// phases never overlapping. Per-row work is unchanged, so the outputs are bit-identical.
/// `quant_only` (the object's no-store twin) serves an instruction whose bf16 GLU output (t[1])
/// no other instruction of the program names: its FP8 bytes and scales are the same.
pub(super) fn prefill_glu_quant_route(
    be: &Arc<CudaBackend>,
    function: KernelFn,
    quant_only: Option<KernelFn>,
    kernarg: DevProgram,
    g: &DevProg,
    insts: &[usize],
) -> LightRoute {
    let mut route = prefill_light_route(be, function, kernarg, g, insts);
    for launch in &mut route.launches {
        let inst = launch.instruction as usize;
        launch.blocks = g.insts[inst].i[0];
        if let Some(q) = quant_only.filter(|_| glu_output_unread(g, inst)) {
            launch.function = q;
        }
    }
    route
}

fn glu_output_unread(g: &DevProg, inst: usize) -> bool {
    let output = g.insts[inst].t[1];
    output != packet::dev::TENSOR_NONE16
        && g
            .insts
            .iter()
            .enumerate()
            .all(|(i, d)| i == inst || !d.t.contains(&output))
}

/// Segments that are exactly a prefill NormResidual and the RmsNorm reading its output (the next
/// instruction), each over all of its slices: `plow_norm_rms_pf` runs the pair at any grid.
pub(super) fn prefill_norm_rms_segments(g: &DevProg) -> Vec<(usize, Vec<usize>)> {
    g.gq_seg_ofs
        .windows(2)
        .enumerate()
        .filter_map(|(seg, w)| {
            let entries = g.gq_stream.get(w[0] as usize..w[1] as usize)?;
            let n = entries.first()?.inst as usize;
            let (nr, rn) = (g.insts.get(n)?, g.insts.get(n + 1)?);
            let complete = |inst: usize, blocks: u16| {
                let own = entries.iter().filter(|e| e.inst as usize == inst);
                blocks != 0
                    && own.clone().count() == usize::from(blocks)
                    && own.enumerate().all(|(slice, e)| e.slice as usize == slice)
            };
            let none = packet::dev::TENSOR_NONE16;
            let matches = nr.op == DevOp::NormResidual as u16
                && rn.op == DevOp::RmsNorm as u16
                && entries.iter().all(|e| e.inst as usize == n || e.inst as usize == n + 1)
                && complete(n, nr.blocks)
                && complete(n + 1, rn.blocks)
                && rn.i[..3] == [nr.i[0], nr.i[1], 0]
                && nr.t[0] != none
                && rn.t[1] == nr.t[0];
            matches.then_some((seg, vec![n, n + 1]))
        })
        .collect()
}

/// The NormResidual+RmsNorm pair at one warp per row (`PLOW_NV_WARPS` = 8 rows per block).
pub(super) fn prefill_norm_rms_route(
    be: &Arc<CudaBackend>,
    function: KernelFn,
    kernarg: DevProgram,
    g: &DevProg,
    insts: &[usize],
) -> LightRoute {
    let mut route = prefill_light_route(be, function, kernarg, g, &insts[..1]);
    for launch in &mut route.launches {
        launch.blocks = g.insts[launch.instruction as usize].i[0].div_ceil(8);
    }
    route
}

/// `(segment, instruction)` of every instruction a light route executes.
pub(super) fn light_instructions(light: &[(usize, Vec<(usize, usize)>)]) -> Vec<(usize, usize)> {
    light
        .iter()
        .flat_map(|(seg, levels)| {
            levels
                .iter()
                .flat_map(move |&(lo, n)| (lo..lo + n.max(1)).map(move |inst| (*seg, inst)))
        })
        .collect()
}

/// A q|k|v projection triple run as one cuBLASLt matmul into a scratch row of
/// `n_total` columns, read in place by the next light attention launch's HeadNormRope.
#[derive(Clone, Copy)]
pub(super) struct QkvFusion {
    /// Segment of the q projection; k and v follow it.
    pub(super) segment: usize,
    pub(super) n_total: u32,
    xs: LightX,
}

/// The q|k|v triples of `g` that can run as one matmul: three adjacent projection segments
/// over one input and K, their weights contiguous (q, k, v), each output read only by one
/// unfused HeadNormRope in the first launch of the light attention segment that follows.
pub(super) fn qkv_fusions(
    g: &DevProg,
    segments: &[Option<DecodeSegment>],
    devp: &[DeviceMem],
    light: &[(usize, Vec<(usize, usize)>)],
) -> Vec<QkvFusion> {
    if !crate::config::RuntimeConfig::get().nv.decode_lt_qkv {
        return Vec::new();
    }
    let t = |inst: usize, slot: usize| g.insts[inst].t[slot] as usize;
    let mut out = Vec::new();
    for (seg, window) in segments.windows(3).enumerate() {
        let [Some(q), Some(k), Some(v)] = window else { continue };
        let Some((_, levels)) = light.iter().find(|(s, _)| *s == seg + 3) else { continue };
        let Some(&(lo, n)) = levels.first() else { continue };
        let [iq, ik, iv] = [q.instruction, k.instruction, v.instruction];
        let weight = |i: usize| devp[t(i, 2)].base;
        let input = |i: usize| devp[t(i, 1)].base;
        if q.m != k.m
            || k.m != v.m
            || q.k != k.k
            || k.k != v.k
            || k.n != v.n
            || input(iq) != input(ik)
            || input(ik) != input(iv)
            || weight(iq) + q.weight_bytes != weight(ik)
            || weight(ik) + k.weight_bytes != weight(iv)
        {
            continue;
        }
        let hnr = |producer: usize| {
            (lo..lo + n).find(|&i| {
                g.insts[i].op == DevOp::HeadNormRope as u16
                    && t(i, 1) == t(producer, 0)
                    && g.insts[i].i[7] == 0
            })
        };
        let (Some(hq), Some(hk), Some(hv)) = (hnr(iq), hnr(ik), hnr(iv)) else { continue };
        // Until the next write of the producer's output (activations are reused per layer).
        let sole_reader = |producer: usize, reader: usize| {
            let out = t(producer, 0);
            g.insts[producer + 1..]
                .iter()
                .enumerate()
                .map(|(i, d)| (producer + 1 + i, d))
                .take_while(|(_, d)| d.t[0] as usize != out)
                .all(|(i, d)| i == reader || !d.t[1..].iter().any(|&h| h as usize == out))
        };
        if !(sole_reader(iq, hq) && sole_reader(ik, hk) && sole_reader(iv, hv)) {
            continue;
        }
        let n_total = q.n + k.n + v.n;
        out.push(QkvFusion {
            segment: seg,
            n_total,
            xs: LightX {
                base: 0,
                row: n_total,
                inst: [hq as u32, hk as u32, hv as u32],
                col: [0, q.n, q.n + k.n],
            },
        });
    }
    out
}

/// A merge-folded hd128 FlashDecode the streamed `light_flash` body serves: nsplit 1, no window,
/// no ring wrap.
fn streamed_flash(d: &DevInst64) -> bool {
    d.op == DevOp::FlashDecode as u16
        && d.i[6] == 128
        && d.i[5] == 1
        && d.i[4] == 0
        && d.i[7] == u32::MAX
        && d.fj[2] != 0
}

/// A light attention segment `light_flash` runs whole: one level of the q, k and v HeadNormRope
/// (no norm, no gamma, hd128, per-sequence KV rows at `pos`), then a streamed FlashDecode that
/// reads exactly their outputs and is the only reader of the roped q. Returns the flash
/// instruction and the q, k, v instructions.
fn folded_hnr(g: &DevProg, levels: &[(usize, usize)]) -> Option<(usize, [u32; 4])> {
    let [(lo, 3), (fi, 1)] = levels[..] else { return None };
    let f = &g.insts[fi];
    let none = packet::dev::TENSOR_NONE16;
    if !streamed_flash(f) || f.t[6] != none {
        return None;
    }
    let (mut q, mut k, mut v) = (None, None, None);
    for i in lo..lo + 3 {
        let d = &g.insts[i];
        if d.op != DevOp::HeadNormRope as u16
            || d.i[2] != 128
            || d.i[4] != 1
            || d.i[5] != 0
            || d.i[7] != 0
            || d.t[2] != none
            || d.i[0] != f.i[0]
        {
            return None;
        }
        match (d.fj[1] != 0, d.t[3] != none) {
            (false, true) => q = Some(i),
            (true, true) => k = Some(i),
            (true, false) => v = Some(i),
            _ => return None,
        }
    }
    let (q, k, v) = (q?, k?, v?);
    let [dq, dk, dv] = [&g.insts[q], &g.insts[k], &g.insts[v]];
    let kv_ok = |d: &DevInst64| {
        d.i[1] == f.i[2] && d.i[6] == f.i[0] && d.fj[1] == f.i[3] && d.fj[2] == u32::MAX && d.t[5] == dq.t[5]
    };
    let q_out = dq.t[0];
    // Nothing but the flash reads the roped q before it is next written.
    let sole = g.insts[q + 1..]
        .iter()
        .enumerate()
        .map(|(i, d)| (q + 1 + i, d))
        .take_while(|(_, d)| d.t[0] != q_out)
        .all(|(i, d)| i == fi || !d.t[1..].contains(&q_out));
    let ok = dq.t[0] == f.t[2]
        && dk.t[0] == f.t[3]
        && dv.t[0] == f.t[4]
        && dq.i[1] == f.i[1]
        && dq.i[3] == 0
        && dq.fj[1] == 0
        && dk.t[3..6] == dq.t[3..6]
        && kv_ok(dk)
        && kv_ok(dv)
        && sole;
    ok.then_some((fi, [q as u32, k as u32, v as u32, 0]))
}

/// Replace each light segment's (empty) route with its launches.
pub(super) fn add_light_routes(
    routes: &mut Vec<Option<LibraryRoute>>,
    be: &Arc<CudaBackend>,
    functions: &LightFunctions,
    kernarg: DevProgram,
    g: &DevProg,
    light: &[(usize, Vec<(usize, usize)>)],
    fusions: &[QkvFusion],
    scratch: Option<&Arc<DeviceMem>>,
    folds: &[ArgmaxFold],
    devp: &[DeviceMem],
) {
    if functions.fp8_attn {
        let attention_segments = light.iter().filter(|(_, levels)| {
            levels.iter().any(|&(inst, _)| g.insts[inst].op == DevOp::FlashDecodeFp8 as u16)
        }).count();
        tracing::info!(attention_segments, "FP8 attention light routes prepared");
    }
    for fold in folds {
        if routes.len() <= fold.segment {
            routes.resize_with(fold.segment + 1, || None);
        }
        routes[fold.segment] = Some(LibraryRoute::Light(LightRoute {
            be: Arc::clone(be),
            kernarg,
            launches: Vec::new(),
            _scratch: None,
        }));
    }
    for (seg, levels) in light {
        if let (Some(function), [(n, 1), (q, 1)]) = (functions.norm_quant, levels.as_slice()) {
            if g.insts[*n].op == DevOp::NormResidualNorm as u16 {
                let mut span = LightSpan { count: 2, ..Default::default() };
                span.op[0] = LightOp::resolve(&g.insts[*n], devp);
                span.op[1] = LightOp::resolve(&g.insts[*q], devp);
                if routes.len() <= *seg {
                    routes.resize_with(seg + 1, || None);
                }
                routes[*seg] = Some(LibraryRoute::Light(LightRoute {
                    be: Arc::clone(be),
                    kernarg,
                    launches: vec![LightLaunch {
                        function,
                        xs: LightX::default(),
                        kind: LightKind::NormQuant,
                        direct: Some(Box::new(span)),
                        instruction: *n as u32,
                        blocks: u32::from(g.insts[*n].blocks),
                        block: BLOCK,
                        smem: 0,
                    }],
                    _scratch: None,
                }));
                continue;
            }
        }
        let fused = fusions.iter().find(|f| f.segment + 3 == *seg).zip(scratch).map(|(f, s)| {
            LightX {
                base: s.base,
                ..f.xs
            }
        });
        if routes.len() <= *seg {
            routes.resize_with(seg + 1, || None);
        }
        let flash = |inst: usize, xs: LightX, hnr: [u32; 4]| {
            functions.flash.filter(|_| streamed_flash(&g.insts[inst])).map(|(function, smem)| LightLaunch {
                function,
                xs,
                kind: LightKind::Flash(hnr),
                instruction: inst as u32,
                blocks: u32::from(g.insts[inst].blocks),
                block: FLASH_BLOCK,
                smem,
                direct: None,
            })
        };
        let folded = folded_hnr(g, levels)
            .and_then(|(inst, hnr)| flash(inst, fused.unwrap_or_default(), hnr));
        let launches = match folded {
            Some(launch) => vec![launch],
            None => levels
                .iter()
                .enumerate()
                .map(|(level, &(inst, n))| {
                    let xs = fused.filter(|_| level == 0).unwrap_or_default();
                    let blocks = (inst..inst + n.max(1))
                        .map(|i| u32::from(g.insts[i].blocks))
                        .max()
                        .unwrap_or(1);
                    if let Some(launch) = flash(inst, xs, [!0; 4]).filter(|_| n == 1) {
                        return launch;
                    }
                    let direct = functions.direct.then(|| {
                        let mut span = LightSpan { count: n as u32, ..Default::default() };
                        for (j, i) in (inst..inst + n.max(1)).enumerate() {
                            span.op[j] = LightOp::resolve(&g.insts[i], devp);
                            if let Some(k) = (0..3).find(|&k| xs.base != 0 && xs.inst[k] == i as u32) {
                                span.op[j].t[1] = xs.base + u64::from(xs.col[k]) * 2;
                                span.x_row[j] = xs.row;
                            }
                        }
                        Box::new(span)
                    });
                    match (n, functions.attn) {
                        (n, Some((function, smem, _))) if n > 0 => LightLaunch {
                            function,
                            xs,
                            kind: LightKind::Attn(n as u32),
                            instruction: inst as u32,
                            blocks,
                            block: BLOCK,
                            smem,
                            direct,
                        },
                        _ => LightLaunch {
                            function: match (functions.tail, DevOp::from_u16(g.insts[inst].op)) {
                                (Some(tail), Some(DevOp::SoftCap | DevOp::Argmax | DevOp::ArgmaxFin)) => {
                                    tail
                                }
                                _ => functions.single,
                            },
                            xs,
                            kind: LightKind::Single,
                            instruction: inst as u32,
                            blocks,
                            block: BLOCK,
                            smem: 0,
                            direct,
                        },
                    }
                })
                .collect::<Vec<LightLaunch>>(),
        };
        let launches = match launches.as_slice() {
            [hnr, flash] if fusable(hnr, flash) => {
                let (h, f) = (hnr.direct.as_ref().unwrap(), flash.direct.as_ref().unwrap());
                let n = h.count as usize;
                let mut span = **h;
                span.count = n as u32 + 1;
                span.fused = 1;
                span.op[n] = f.op[0];
                vec![LightLaunch {
                    function: flash.function,
                    xs: LightX::default(),
                    kind: LightKind::Attn(n as u32 + 1),
                    instruction: flash.instruction,
                    blocks: flash.blocks,
                    block: flash.block,
                    smem: flash.smem,
                    direct: Some(Box::new(span)),
                }]
            }
            [cap, amax, fin] => match capmax(functions, cap, amax) {
                Some(launch) => vec![launch, clone_launch(fin)],
                None => launches,
            },
            _ => launches,
        };
        if functions.fp8_attn && light.first().is_some_and(|(first, _)| first == seg) {
            let spans: Vec<_> = launches.iter().filter_map(|l| l.direct.as_ref()).map(|s| {
                (s.count, s.fused, s.op.iter().map(|o| (o.d.op, o.d.blocks, o.d.i[0], o.d.i[1])).collect::<Vec<_>>())
            }).collect();
            tracing::info!(segment = seg, ?levels, ?spans, "FP8 attention first light launch spans");
        }
        let launches = launches.into_iter().map(|l| sliding(functions, l)).collect();
        routes[*seg] = Some(LibraryRoute::Light(LightRoute {
            be: Arc::clone(be),
            kernarg,
            launches,
            _scratch: fused.and(scratch.cloned()),
        }));
    }
}

/// `light_attn_s` arena: the hd256 row-group fold (8 groups x 256 f32 + m/l).
const ATTN_S_SMEM: u32 = 16 << 10;

fn fp8_flash256_blocks(span: &LightSpan, original_blocks: u32) -> Option<u32> {
    let d = &span.op[0].d;
    if span.count != 1 || span.fused != 0 || d.op != DevOp::FlashDecodeFp8 as u16
        || d.i[6] != 256 || d.i[0] != 128 || d.i[1] != 16 || d.i[2] != 8
        || d.i[5] != 1 || d.blocks == 0 || original_blocks == 0
    {
        return None;
    }
    let work = d.i[0] * (d.i[1] / 2) * d.i[5];
    Some(work.min(original_blocks.saturating_mul(8)).min(u32::from(u16::MAX)))
}

/// An hd256 attention launch on `light_attn_s`, at twice the blocks (two per SM).
fn sliding(functions: &LightFunctions, launch: LightLaunch) -> LightLaunch {
    if let (Some((function, smem)), LightKind::Attn(1), Some(span)) =
        (functions.fp8_flash256, launch.kind, launch.direct.as_ref())
    {
        if let Some(blocks) = fp8_flash256_blocks(span, launch.blocks) {
            let mut span = **span;
            span.op[0].d.blocks = blocks as u16;
            return LightLaunch { function, blocks, smem, direct: Some(Box::new(span)), ..launch };
        }
    }
    let (Some(function), LightKind::Attn(_), Some(span)) = (functions.attn_s, launch.kind, launch.direct.as_ref())
    else {
        return launch;
    };
    let ops = &span.op[..span.count as usize];
    let hd256 = ops.iter().all(|o| {
        let d = &o.d;
        d.op == DevOp::HeadNormRope as u16 && d.i[2] == 256 || d.op == DevOp::FlashDecode as u16 && d.i[6] == 256
    });
    let Some(blocks) = launch.blocks.checked_mul(2).filter(|&b| b <= u32::from(u16::MAX)) else {
        return launch;
    };
    if !hd256 || ops.iter().any(|o| o.d.blocks == 0) {
        return launch;
    }
    let mut span = **span;
    for o in &mut span.op[..span.count as usize] {
        o.d.blocks = o.d.blocks.saturating_mul(2);
    }
    LightLaunch {
        function,
        blocks,
        smem: ATTN_S_SMEM,
        direct: Some(Box::new(span)),
        ..launch
    }
}

fn clone_launch(l: &LightLaunch) -> LightLaunch {
    LightLaunch { direct: l.direct.clone(), ..*l }
}

/// An in-place SoftCap on the logits followed by the batched Argmax over them, as one
/// `light_capmax` launch of `rows x G` blocks (G chunks per row, at most the part stride).
fn capmax(functions: &LightFunctions, cap: &LightLaunch, amax: &LightLaunch) -> Option<LightLaunch> {
    let function = functions.capmax?;
    let (c, a) = (&cap.direct.as_ref()?.op[0], &amax.direct.as_ref()?.op[0]);
    let (n, rows, parts) = (a.d.i[0], a.d.i[1], u32::from(a.d.blocks));
    let ok = c.d.op == DevOp::SoftCap as u16
        && a.d.op == DevOp::Argmax as u16
        && c.t[0] == c.t[1]
        && a.t[1] == c.t[0]
        && rows >= 2
        && n % 8 == 0
        && u64::from(c.d.i[0]) == u64::from(n) * u64::from(rows);
    if !ok {
        return None;
    }
    let chunks = (512 / rows).clamp(1, parts);
    let mut span = LightSpan { count: 2, ..Default::default() };
    span.op[0] = *c;
    span.op[1] = *a;
    Some(LightLaunch {
        function,
        xs: LightX::default(),
        kind: LightKind::Attn(2),
        instruction: amax.instruction,
        blocks: rows * chunks,
        block: BLOCK,
        smem: 0,
        direct: Some(Box::new(span)),
    })
}

/// A light ABI 2 HeadNormRope level whose every instruction writes the next FlashDecode's
/// q, k or v (per-batch KV ring), run as that flash launch's per-block prologue.
fn fusable(hnr: &LightLaunch, flash: &LightLaunch) -> bool {
    let (Some(h), Some(f), LightKind::Attn(_), LightKind::Attn(1)) =
        (&hnr.direct, &flash.direct, hnr.kind, flash.kind)
    else {
        return false;
    };
    let fl = &f.op[0].d;
    if f.count != 1 || fl.op != DevOp::FlashDecode as u16 || f.op[0].t[6] != 0 || fl.i[2] == 0 {
        return false;
    }
    let [q, k, v] = [f.op[0].t[2], f.op[0].t[3], f.op[0].t[4]];
    let gqa = fl.i[1] / fl.i[2];
    let ops = &h.op[..h.count as usize];
    let tasks: u32 = ops.iter().map(|o| if o.t[0] == q { gqa } else { 1 }).sum();
    !ops.is_empty()
        && tasks as usize <= BLOCK as usize / 32
        && ops.iter().all(|o| {
            let d = &o.d;
            let kv = o.t[0] == k || o.t[0] == v;
            d.op == DevOp::HeadNormRope as u16
                && matches!(d.i[2], 256 | 512)
                && d.i[5] == 0
                && (o.t[0] == q && d.fj[1] == 0 || kv && d.fj[1] != 0 && d.i[6] != 0)
                && (d.i[7] == 0 || kv && o.t[6] == v)
        })
}

/// Every host-launched kernel of `routes` that leaves cuBLASLt's workspace alone.
fn glue_kernels<'a>(routes: impl Iterator<Item = &'a LibraryRoute>) -> Vec<KernelFn> {
    routes
        .filter_map(|route| match route {
            LibraryRoute::Moe(route) => Some(route.glue().collect::<Vec<_>>()),
            LibraryRoute::Light(route) => Some(route.launches.iter().map(|l| l.function).collect()),
            LibraryRoute::Projection(_) => None,
        })
        .flatten()
        .collect()
}

pub(super) fn library_routes(routes: Vec<Option<CublasLtDecodeRoute>>) -> Vec<Option<LibraryRoute>> {
    routes
        .into_iter()
        .map(|route| route.map(LibraryRoute::Projection))
        .collect()
}

pub(super) enum ProjectionBackend {
    Lt(Arc<crate::device::cuda::lt::Lt>),
    Native(Arc<native_decode::Native>),
}

enum ProjectionPlan {
    Lt(Arc<crate::device::cuda::lt::Plan>),
    Native(native_decode::Plan),
    /// Operands are bound into the plan's Params blob.
    Cutlass(crate::device::cuda::cutlass_fp8::Plan),
    Folded,
}

impl ProjectionPlan {
    fn run(&self, input: u64, weight: u64, output: u64, stream: &CudaStream) -> Result<()> {
        match self {
            Self::Lt(p) => p.run(input, weight, output, stream),
            Self::Native(p) => p.run(input, weight, output, stream),
            Self::Cutlass(p) => p.run(stream),
            Self::Folded => Ok(()),
        }
    }
}


/// `[output, input, weight]` of a routed projection, checked aligned and nonaliasing.
fn operands(segment: &DecodeSegment, insts: &[DevInst64], devp: &[DeviceMem]) -> Result<[u64; 3]> {
    let d = &insts[segment.instruction];
    let [output, input, weight] = [
        devp[d.t[0] as usize].base,
        devp[d.t[1] as usize].base,
        devp[d.t[2] as usize].base,
    ];
    let end = |base: u64, bytes: u64| {
        base.checked_add(bytes).ok_or_else(|| {
            RuntimeError::Rejected("cuBLASLt decode tensor address range overflow".into())
        })
    };
    let output_end = end(output, segment.output_bytes)?;
    let input_end = end(input, segment.input_bytes)?;
    let weight_end = end(weight, segment.weight_bytes)?;
    let overlaps_output = (output < input_end && input < output_end)
        || (output < weight_end && weight < output_end);
    if [input, weight, output].iter().any(|p| p % 16 != 0) || overlaps_output {
        return Err(RuntimeError::Rejected(
            "cuBLASLt decode requires aligned, nonaliasing tensors".into(),
        ));
    }
    Ok([output, input, weight])
}

/// Batch 0 `(weight, output)` and the pair strides when segment `b` (adjacent to `a`) is the
/// same GEMM shape over the same input into a disjoint output.
fn pair_operands(
    a: &DecodeSegment,
    [out_a, in_a, w_a]: [u64; 3],
    b: &DecodeSegment,
    [out_b, in_b, w_b]: [u64; 3],
    cold: bool,
) -> Option<(u64, u64, crate::device::cuda::lt::Pair)> {
    if (a.m, a.n, a.k) != (b.m, b.n, b.k) || in_a != in_b || a.output_bytes != b.output_bytes {
        return None;
    }
    let ((w0, o0), (w1, o1)) = if w_a < w_b && out_a < out_b {
        ((w_a, out_a), (w_b, out_b))
    } else if w_b < w_a && out_b < out_a {
        ((w_b, out_b), (w_a, out_a))
    } else {
        return None;
    };
    if o1 - o0 < a.output_bytes || w1 - w0 < a.weight_bytes {
        return None;
    }
    let pair = crate::device::cuda::lt::Pair {
        w_stride: i64::try_from((w1 - w0) / 2).ok()?,
        c_stride: i64::try_from((o1 - o0) / 2).ok()?,
        input: in_a,
        output: o0,
        cold,
    };
    Some((w0, o0, pair))
}

/// `extra` names further `(segment, instruction)` pairs a library route executes.
pub(super) fn ordered_waits(
    g: &DevProg,
    segments: &[Option<DecodeSegment>],
    extra: &[(usize, usize)],
) -> Result<Vec<packet::dev::Wait>> {
    let library: Vec<_> = segments
        .iter()
        .map(|segment| segment.map(|s| s.instruction))
        .collect();
    ordered_waits_for(g, &library, extra)
}

/// `segments[seg]` names the one instruction of a segment that a host library call replaces.
pub(super) fn ordered_waits_for(
    g: &DevProg,
    segments: &[Option<usize>],
    extra: &[(usize, usize)],
) -> Result<Vec<packet::dev::Wait>> {
    validate_segment_windows(g)?;
    let reject = || RuntimeError::Rejected("cuBLASLt requires ordered coarse dependencies".into());
    if g.n_counter as usize != g.insts.len() || segments.len() + 1 != g.gq_seg_ofs.len() {
        return Err(reject());
    }
    let mut placement = vec![None; g.insts.len()];
    for e in &g.stream {
        let slot = placement.get_mut(e.inst as usize).ok_or_else(reject)?;
        if slot.is_some_and(|seg| seg != e.seg)
            || e.flags != 0
            || g.succs
                .get(e.succ_ofs as usize..e.succ_ofs as usize + e.succ_len as usize)
                != Some(&[e.inst][..])
        {
            return Err(reject());
        }
        *slot = Some(e.seg);
    }
    if placement.iter().any(Option::is_none) {
        return Err(reject());
    }
    for e in &g.stream {
        let waits = g
            .waits
            .get(e.wait_ofs as usize..e.wait_ofs as usize + e.wait_len as usize)
            .ok_or_else(reject)?;
        for w in waits {
            if w.id >= e.inst
                || g.insts
                    .get(w.id as usize)
                    .is_none_or(|d| w.threshold != u32::from(d.blocks))
                || placement
                    .get(w.id as usize)
                    .copied()
                    .flatten()
                    .is_none_or(|seg| seg > e.seg)
            {
                return Err(reject());
            }
        }
    }
    let mut library = vec![false; g.insts.len()];
    let routed = segments
        .iter()
        .enumerate()
        .filter_map(|(seg, instruction)| instruction.map(|i| (seg, i)));
    for (seg, instruction) in routed.chain(extra.iter().copied()) {
        if placement.get(instruction) != Some(&Some(seg as u16)) {
            return Err(reject());
        }
        library[instruction] = true;
    }
    let mut waits = g.waits.clone();
    for w in &mut waits {
        if library.get(w.id as usize).copied().ok_or_else(reject)? {
            // The consumer's launch follows the complete library call on the same stream.
            w.threshold = 0;
        }
    }
    Ok(waits)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_routes(
    backend: &ProjectionBackend,
    segments: Vec<Option<DecodeSegment>>,
    insts: &mut [DevInst64],
    devp: &[DeviceMem],
    templates: Option<&[Option<LibraryRoute>]>,
    pair: bool,
    decode: bool,
    fusions: &[QkvFusion],
    scratch: Option<&DeviceMem>,
    head: Option<&HeadKernel>,
    folds: &[ArgmaxFold],
    rows: &[u32],
) -> Result<Vec<Option<CublasLtDecodeRoute>>> {
    let mut routes: Vec<Option<CublasLtDecodeRoute>> = Vec::new();
    if segments.is_empty() {
        return Ok(routes);
    }
    let pair_lt = match backend {
        ProjectionBackend::Lt(lt) if pair => Some(lt),
        _ => None,
    };
    // A rung pairs exactly where its template (the widest rung) paired. It pins the template's
    // algorithms only with PLOW_LT_RUNG_ALGOS off; otherwise it times its own.
    let pin = !crate::config::RuntimeConfig::get().nv.lt_rung_algos;
    let template_pair = |index: usize| -> Result<Option<Option<&crate::device::cuda::lt::Plan>>> {
        let Some(routes) = templates else {
            return Ok(Some(None));
        };
        match (routes.get(index), routes.get(index + 1)) {
            (
                Some(Some(LibraryRoute::Projection(route))),
                Some(Some(LibraryRoute::Projection(next))),
            ) if next.folded() => match route.plan.as_ref() {
                ProjectionPlan::Lt(_) if !pin => Ok(Some(None)),
                ProjectionPlan::Lt(plan) => Ok(Some(Some(plan.as_ref()))),
                _ => Err(RuntimeError::Rejected("decode projection backend changed".into())),
            },
            _ => Ok(None),
        }
    };
    let mut plans = std::collections::HashMap::new();
    let mut pair_plans = std::collections::HashMap::new();
    let mut fp8_plans = 0usize;
    let mut cutlass_plans = 0usize;
    let mut pairs = 0usize;
    let mut index = 0;
    while index < segments.len() {
        let Some(segment) = segments[index] else {
            routes.push(None);
            index += 1;
            continue;
        };
        // An lm_head with an unaligned vocab: cuBLASLt serves its aligned columns (see HeadTail).
        let tail = head_tail(&segment, backend, head);
        let fold = folds.iter().find(|f| f.head_segment == index);
        if fold.is_some() && tail.is_none() {
            return Err(RuntimeError::Rejected("folded argmax without its head kernel".into()));
        }
        let key = (
            segment.m,
            if tail.is_some() { segment.n & !15 } else { segment.n },
            segment.k,
        );
        let ops = operands(&segment, insts, devp)?;
        let [output, input, weight] = ops;
        let op = &insts[segment.instruction];
        if matches!(DevOp::from_u16(op.op), Some(DevOp::GemmFp8 | DevOp::GemmMedFp8 | DevOp::GemmSmallFp8)) {
            let ProjectionBackend::Lt(lt) = backend else {
                return Err(RuntimeError::Rejected("FP8 projection requires cuBLASLt".into()));
            };
            if tail.is_some() || fold.is_some() {
                return Err(RuntimeError::Rejected("FP8 projection cannot fold an output head".into()));
            }
            let mut scales = [0; 2];
            for (index, (handle, elements)) in [(op.t[3], segment.m), (op.t[4], segment.n)].into_iter().enumerate() {
                let mem = devp.get(handle as usize).ok_or_else(|| RuntimeError::Rejected("missing FP8 scale vector".into()))?;
                let bytes = u64::from(elements) * 4;
                let end = mem.base.checked_add(bytes).ok_or_else(|| RuntimeError::Rejected("FP8 scale range overflow".into()))?;
                if mem.base == 0 || mem.base % 16 != 0 || mem.len < bytes
                    || (mem.base < output + segment.output_bytes && output < end) {
                    return Err(RuntimeError::Rejected("invalid FP8 scale vector".into()));
                }
                scales[index] = mem.base;
            }
            // Scale pointers belong to this layer; shape-only BF16 plan sharing is invalid here.
            let fast_k = crate::config::RuntimeConfig::get()
                .nv
                .lt_fp8_fast_accum_max_k
                .is_some_and(|max_k| segment.k <= max_k);
            let cutlass = match lt.cutlass_fp8().filter(|_| decode) {
                Some(c) => c.plan(
                    segment.m,
                    segment.n,
                    segment.k,
                    [input, weight, output, scales[0], scales[1]],
                    fast_k,
                )?,
                None => None,
            };
            let plan = match cutlass {
                Some(plan) => {
                    cutlass_plans += 1;
                    ProjectionPlan::Cutlass(plan)
                }
                None => ProjectionPlan::Lt(lt.fp8_plan(
                    segment.m,
                    segment.n,
                    segment.k,
                    scales[1],
                    scales[0],
                    !decode && fast_k,
                )?),
            };
            fp8_plans += 1;
            insts[segment.instruction].op = DevOp::Nop as u16;
            routes.push(Some(CublasLtDecodeRoute {
                plan: Arc::new(plan), input, weight, output, tail: None,
            }));
            index += 1;
            continue;
        }
        insts[segment.instruction].op = DevOp::Nop as u16;
        if let (Some(fusion), Some(scratch), ProjectionBackend::Lt(lt)) = (
            fusions.iter().find(|f| f.segment == index),
            scratch,
            backend,
        ) {
            let template = templates
                .map(|routes| match routes.get(index) {
                    Some(Some(LibraryRoute::Projection(route))) => match route.plan.as_ref() {
                        ProjectionPlan::Lt(p) => Ok(p.as_ref()),
                        _ => Err(RuntimeError::Rejected("decode projection backend changed".into())),
                    },
                    _ => Err(RuntimeError::Rejected("cuBLASLt rung template route missing".into())),
                })
                .transpose()?;
            if u64::from(segment.m) * u64::from(fusion.n_total) * 2 > scratch.len {
                return Err(RuntimeError::Rejected("fused q|k|v scratch too small".into()));
            }
            let plan = lt.plan(segment.m, fusion.n_total, segment.k, weight, template.filter(|_| pin), rows)?;
            for next in segments[index + 1..index + 3].iter().flatten() {
                insts[next.instruction].op = DevOp::Nop as u16;
            }
            routes.push(Some(CublasLtDecodeRoute {
                plan: Arc::new(ProjectionPlan::Lt(plan)),
                input,
                weight,
                output: scratch.base,
                tail: None,
            }));
            for _ in 0..2 {
                routes.push(Some(CublasLtDecodeRoute {
                    plan: Arc::new(ProjectionPlan::Folded),
                    input: 0,
                    weight: 0,
                    output: 0,
                    tail: None,
                }));
            }
            index += 3;
            continue;
        }
        let pair_template = match pair_lt {
            Some(_) => template_pair(index)?,
            None => None,
        };
        if let (Some(lt), Some(pair_template)) = (pair_lt, pair_template) {
            let next = segments.get(index + 1).copied().flatten();
            let paired = match next {
                Some(next) => {
                    let next_ops = operands(&next, insts, devp)?;
                    pair_operands(&segment, ops, &next, next_ops, decode).map(|p| (next, p))
                }
                None => None,
            };
            if let Some((next, (w0, o0, p))) = paired {
                let plan = match pair_plans.entry((key, p.w_stride, p.c_stride)) {
                    std::collections::hash_map::Entry::Occupied(e) => Arc::clone(e.get()),
                    std::collections::hash_map::Entry::Vacant(e) => Arc::clone(e.insert(Arc::new(
                        ProjectionPlan::Lt(lt.pair_plan(key.0, key.1, key.2, w0, p, pair_template, rows)?),
                    ))),
                };
                insts[next.instruction].op = DevOp::Nop as u16;
                routes.push(Some(CublasLtDecodeRoute {
                    plan,
                    input,
                    weight: w0,
                    output: o0,
                    tail: None,
                }));
                routes.push(Some(CublasLtDecodeRoute {
                    plan: Arc::new(ProjectionPlan::Folded),
                    input: 0,
                    weight: 0,
                    output: 0,
                    tail: None,
                }));
                pairs += 1;
                index += 2;
                continue;
            }
        }
        let plan = match plans.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => Arc::clone(e.get()),
            std::collections::hash_map::Entry::Vacant(e) => {
                let template = templates
                    .map(|routes| match routes.get(index) {
                        Some(Some(LibraryRoute::Projection(route))) => Ok(route),
                        _ => Err(RuntimeError::Rejected(
                            "cuBLASLt rung template route missing".into(),
                        )),
                    })
                    .transpose()?;
                let plan = match backend {
                    ProjectionBackend::Lt(lt) => {
                        let template = template
                            .map(|r| match r.plan.as_ref() {
                                ProjectionPlan::Lt(p) => Ok(p.as_ref()),
                                _ => Err(RuntimeError::Rejected(
                                    "decode projection backend changed".into(),
                                )),
                            })
                            .transpose()?;
                        ProjectionPlan::Lt(lt.plan(key.0, key.1, key.2, weight, template.filter(|_| pin), rows)?)
                    }
                    ProjectionBackend::Native(native) => {
                        let template = template
                            .map(|r| match r.plan.as_ref() {
                                ProjectionPlan::Native(p) => Ok(p),
                                _ => Err(RuntimeError::Rejected(
                                    "decode projection backend changed".into(),
                                )),
                            })
                            .transpose()?;
                        ProjectionPlan::Native(native.plan(key.0, key.1, key.2, template)?)
                    }
                };
                Arc::clone(e.insert(Arc::new(plan)))
            }
        };
        routes.push(Some(CublasLtDecodeRoute {
            plan,
            input,
            weight,
            output,
            tail: tail.map(|kernel| HeadTail {
                kernel,
                rows: segment.m,
                n: segment.n,
                k: segment.k,
                n0: key.1,
                ids: fold.map_or(0, |f| f.ids),
            }),
        }));
        index += 1;
    }
    tracing::info!(
        segments = routes.len(),
        projections = routes.iter().flatten().count(),
        plans = plans.len(),
        fp8_plans,
        cutlass_plans,
        pairs,
        native = matches!(backend, ProjectionBackend::Native(_)),
        "projection routes prepared"
    );
    Ok(routes)
}

pub(super) struct CublasLtDecodeGraph {
    be: Arc<CudaBackend>,
    graph: Option<crate::device::cuda::GraphExec>,
    _routes: Vec<Option<LibraryRoute>>,
}

impl CublasLtDecodeGraph {
    pub(super) fn capture(
        be: &Arc<CudaBackend>,
        stream: &CudaStream,
        base: DevProgram,
        function: KernelFn,
        grid: u32,
        smem: u32,
        routes: Vec<Option<LibraryRoute>>,
    ) -> Result<Self> {
        let untouched = glue_kernels(routes.iter().flatten());
        let graph = be.graph_capture_hoisting(stream, &untouched, || {
            for (seg, route) in routes.iter().enumerate() {
                if let Some(route) = route {
                    route.run(stream)?;
                    continue;
                }
                let mut arg = base;
                segment_window(&mut arg, &base, seg, false);
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                be.launch_cooperative(function, grid, BLOCK, smem, &mut params, Some(stream))?;
            }
            Ok(())
        })?;
        Ok(Self {
            be: Arc::clone(be),
            graph: Some(graph),
            _routes: routes,
        })
    }

    pub(super) fn launch(&self, stream: &CudaStream) -> Result<()> {
        self.be
            .graph_launch(self.graph.as_ref().expect("captured decode graph"), stream)
    }
}

impl Drop for CublasLtDecodeGraph {
    fn drop(&mut self) {
        if let Some(graph) = self.graph.take() {
            self.be.graph_destroy(graph);
        }
    }
}

impl GpuEngine {
    #[cfg(test)]
    pub(super) fn capture_library_reference_graph(
        &self,
        waits: u64,
    ) -> Result<crate::device::cuda::GraphExec> {
        self.be.graph_capture(&self.stream, || {
            for (seg, route) in self.cublaslt_decode.iter().enumerate() {
                if let Some(route) = route {
                    route.run(&self.stream)?;
                }
                let mut arg = self.kernarg;
                arg.waits = waits;
                segment_window(&mut arg, &self.kernarg, seg, false);
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                self.be.launch_cooperative(
                    self.f,
                    self.grid,
                    BLOCK,
                    self.smem,
                    &mut params,
                    Some(&self.stream),
                )?;
            }
            Ok(())
        })
    }

    pub(super) fn capture_decode_graph(&mut self) -> Result<()> {
        let be = Arc::clone(&self.be);
        let untouched = glue_kernels(self.cublaslt_decode.iter().flatten());
        self.cublaslt_decode_graph =
            Some(be.graph_capture_hoisting(&self.stream, &untouched, || {
                self.enqueue_decode_chain()
            })?);
        Ok(())
    }

    /// Capture a partial widest-rung decode for tensor inspection. Later outputs are stale.
    pub fn capture_debug_decode_prefix(&mut self, segments: usize) -> Result<()> {
        let total = self.cublaslt_decode.len().max(self.decode_packet_roles.len());
        if segments == 0 || segments > total || !self.cublaslt_decode_capture
            || self.decode_contexts.is_some()
        {
            return Err(RuntimeError::Rejected(
                "partial segment capture requires a captured decode chain and a valid segment count".into(),
            ));
        }
        self.be.stream_synchronize(&self.stream)?;
        let graph = self.be.graph_capture(&self.stream, || {
            self.enqueue_decode_segments(segments)
        })?;
        if let Some(old) = self.cublaslt_decode_graph.replace(graph) {
            self.be.graph_destroy(old);
        }
        Ok(())
    }

    fn enqueue_decode_chain(&self) -> Result<()> {
        self.enqueue_decode_segments(
            self.cublaslt_decode.len().max(self.decode_packet_roles.len()).max(1),
        )
    }

    fn enqueue_decode_segments(&self, segments: usize) -> Result<()> {
        for seg in 0..segments {
            if let Some(Some(route)) = self.cublaslt_decode.get(seg) {
                route.run(&self.stream)?;
                continue;
            }
            let mut arg = self.kernarg;
            let role = self
                .decode_packet_roles
                .get(seg)
                .copied()
                .filter(|&id| id != plow_asset::segment_roles::INTERPRETER)
                .and_then(|id| self.packet_roles[id as usize - 1].as_ref());
            if !self.decode_packet_roles.is_empty() {
                segment_window(&mut arg, &self.kernarg, seg, role.is_some());
            } else if !self.cublaslt_decode.is_empty() {
                arg.cur_seg = 0;
                arg.gq_seg_ofs += (seg * 4) as u64;
                arg.gq_cursor += (seg * CTR_STRIDE as usize * 4) as u64;
            }
            let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
            // A MoE-routed chain runs no grouped-arm body: its launches take the narrow arena.
            let (function, smem) = match &self.routed_decode {
                Some(routed) => (routed.function, routed.smem),
                None if self.moe_lt_decode => (self.f, self.smem_narrow),
                None => self
                    .gemv_wide
                    .as_ref()
                    .map_or((self.f, self.smem), |o| (o.function, o.smem)),
            };
            self.be.launch_cooperative(
                role.map_or(function, |r| r.function),
                role.map_or(self.grid, |r| r.grid),
                role.map_or(BLOCK, |r| r.block),
                role.map_or(smem, |r| r.smem),
                &mut params,
                Some(&self.stream),
            )?;
        }
        Ok(())
    }

    pub(super) fn launch_decode(&mut self) -> Result<()> {
        if (self.cublaslt_decode.is_empty() && self.decode_packet_roles.is_empty())
            || !self.cublaslt_decode_capture
        {
            return self.enqueue_decode_chain();
        }
        if self.cublaslt_decode_graph.is_none() {
            self.capture_decode_graph()?;
        }
        self.be.graph_launch(
            self.cublaslt_decode_graph.as_ref().expect("captured"),
            &self.stream,
        )
    }
}

#[derive(Clone, Copy)]
enum ProjectionPhase<'a> {
    Decode,
    Prefill(&'a str),
}

pub(super) fn decode_segments(
    program: &DevProg,
    tensors: &[DevTensor],
    roles: &[u8],
) -> Result<Vec<Option<DecodeSegment>>> {
    projection_segments(program, tensors, roles, ProjectionPhase::Decode)
}

pub(super) fn prefill_segments(
    program: &DevProg,
    tensors: &[DevTensor],
    roles: &[u8],
    profile: &str,
) -> Result<Vec<Option<DecodeSegment>>> {
    projection_segments(program, tensors, roles, ProjectionPhase::Prefill(profile))
}

fn projection_segments(
    program: &DevProg,
    tensors: &[DevTensor],
    roles: &[u8],
    phase: ProjectionPhase<'_>,
) -> Result<Vec<Option<DecodeSegment>>> {
    let fail = || RuntimeError::Rejected("invalid packet-declared projection segments".into());
    let rows = packet::devbuild::program_rows(program.t);
    if match phase {
        ProjectionPhase::Decode => {
            !(1..=plow_asset::segment_roles::CUBLASLT_DECODE_MAX_ROWS).contains(&rows)
        }
        ProjectionPhase::Prefill(_) => rows > plow_asset::segment_roles::CUBLASLT_PREFILL_MAX_ROWS,
    } || program.l2_domains != 0
        || program.gq_stream.is_empty()
    {
        return Err(fail());
    }
    let count = program.gq_seg_ofs.len().checked_sub(1).ok_or_else(fail)?;
    if count == 0
        || roles.len() != count
        || !roles.iter().copied().any(|role| match phase {
            ProjectionPhase::Decode => plow_asset::segment_roles::is_projection(role),
            ProjectionPhase::Prefill(_) => role == plow_asset::segment_roles::CUBLASLT,
        })
        || program.gq_seg_ofs.first() != Some(&0)
        || program.gq_seg_ofs.last().copied() != Some(program.gq_stream.len() as u32)
    {
        return Err(fail());
    }
    let mut routes = vec![None; count];
    for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
        let entries = program
            .gq_stream
            .get(bounds[0] as usize..bounds[1] as usize)
            .ok_or_else(fail)?;
        if entries.is_empty() || entries.iter().any(|entry| entry.seg as usize != segment) {
            return Err(fail());
        }
        let selected = match phase {
            ProjectionPhase::Decode => plow_asset::segment_roles::is_projection(roles[segment]),
            ProjectionPhase::Prefill(_) => roles[segment] == plow_asset::segment_roles::CUBLASLT,
        };
        if !selected {
            continue;
        }
        let instruction = entries[0].inst as usize;
        let op = program.insts.get(instruction).ok_or_else(fail)?;
        let fp8 = matches!(DevOp::from_u16(op.op), Some(DevOp::GemmFp8 | DevOp::GemmMedFp8 | DevOp::GemmSmallFp8));
        let valid_op = match phase {
            ProjectionPhase::Decode => op.op == DevOp::Gemv as u16
                || (fp8 && roles[segment] == plow_asset::segment_roles::CUBLASLT),
            ProjectionPhase::Prefill(profile) => {
                if fp8 {
                    plow_asset::segment_roles::cublaslt_prefill_fp8(profile, op.i[0], op.i[1], op.i[2])
                } else { matches!(
                    DevOp::from_u16(op.op),
                    Some(DevOp::Gemm | DevOp::GemmMed | DevOp::GemmSmall)
                ) && plow_asset::segment_roles::cublaslt_prefill_bf16(
                    profile, op.i[0], op.i[1], op.i[2],
                ) }
            }
        };
        let valid_immediates = match phase {
            ProjectionPhase::Decode => op.i[3..].iter().all(|&value| value == 0),
            ProjectionPhase::Prefill(_) => op.i[3..6].iter().all(|&value| value == 0),
        };
        if !valid_op
            || op.t[if fp8 { 5 } else { 3 }..].iter().any(|&t| t != packet::dev::TENSOR_NONE16)
            || op.i[0] != rows
            || op.i[1] == 0
            || op.i[2] == 0
            || !valid_immediates
            || entries
                .iter()
                .any(|entry| entry.inst as usize != instruction)
            || program.stream.iter().any(|entry| {
                (entry.inst as usize == instruction) != (entry.seg as usize == segment)
            })
        {
            return Err(fail());
        }
        let [m, n, k] = [op.i[0], op.i[1], op.i[2]];
        let bytes = |a: u32, b: u32| {
            u64::from(a)
                .checked_mul(u64::from(b))
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(fail)
        };
        let output_bytes = bytes(m, n)?;
        let input_bytes = bytes(m, k)? / if fp8 { 2 } else { 1 };
        let weight_bytes = bytes(n, k)? / if fp8 { 2 } else { 1 };
        if fp8 {
            if op.t[..5].iter().copied().collect::<std::collections::BTreeSet<_>>().len() != 5 {
                return Err(fail());
            }
            for (handle, required) in [(op.t[3], u64::from(m) * 4), (op.t[4], u64::from(n) * 4)] {
                if tensors.get(handle as usize).is_none_or(|tensor| tensor.bytes < required) {
                    return Err(fail());
                }
            }
        }
        for (handle, required) in [
            (op.t[0], output_bytes),
            (op.t[1], input_bytes),
            (op.t[2], weight_bytes),
        ] {
            if tensors
                .get(handle as usize)
                .is_none_or(|tensor| tensor.bytes < required)
            {
                return Err(fail());
            }
        }
        if op.t[0] == op.t[1] || op.t[0] == op.t[2] {
            return Err(fail());
        }
        routes[segment] = Some(DecodeSegment {
            instruction,
            m,
            n,
            k,
            output_bytes,
            input_bytes,
            weight_bytes,
        });
    }
    Ok(routes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::{DevInst64, StreamEnt};

    #[test]
    fn norm_quant_pair_requires_complete_interleaved_slices_and_shared_output() {
        let (mut g, _) = fixture(4);
        let none = packet::dev::TENSOR_NONE16;
        let norm = &mut g.insts[1];
        norm.op = DevOp::NormResidualNorm as u16;
        norm.blocks = 4;
        norm.i[..2].copy_from_slice(&[4, 3840]);
        norm.t.fill(none);
        norm.t[..4].copy_from_slice(&[0, 1, 1, 2]);
        let quant = &mut g.insts[2];
        quant.op = DevOp::QuantFp8 as u16;
        quant.blocks = 4;
        quant.i[..2].copy_from_slice(&[4, 3840]);
        quant.t.fill(none);
        quant.t[..3].copy_from_slice(&[3, 0, 4]);
        let entries: Vec<_> = (0..4)
            .flat_map(|slice| [1, 2].map(move |inst| StreamEnt { inst, slice, ..Default::default() }))
            .collect();
        assert_eq!(norm_quant_pair(&g, &entries), Some((1, 2)));
        let grouped: Vec<_> = [1, 2]
            .into_iter()
            .flat_map(|inst| (0..4).map(move |slice| StreamEnt { inst, slice, ..Default::default() }))
            .collect();
        assert_eq!(norm_quant_pair(&g, &grouped), Some((1, 2)));
        let mut wrong_grouped = grouped.clone();
        wrong_grouped.swap(0, 4);
        assert_eq!(norm_quant_pair(&g, &wrong_grouped), None);
        let mut wrong_order = entries.clone();
        wrong_order.swap(1, 2);
        assert_eq!(norm_quant_pair(&g, &wrong_order), None);
        assert_eq!(norm_quant_pair(&g, &entries[..6]), None);
        g.insts[2].t[1] = 5;
        assert_eq!(norm_quant_pair(&g, &entries), None);
        g.insts[2].t[1] = 0;
        g.insts[2].t[3] = 6;
        assert_eq!(norm_quant_pair(&g, &entries), None);
        g.insts[2].t[3] = none;
        g.insts[1].i[0] = 8;
        g.insts[2].i[0] = 8;
        assert_eq!(norm_quant_pair(&g, &entries), None);
        g.insts[1].i[0] = 4;
        g.insts[2].i[0] = 4;
        g.insts[1].i[1] = 4096;
        g.insts[2].i[1] = 4096;
        assert_eq!(norm_quant_pair(&g, &entries), None);
    }

    #[test]
    fn fp8_attention_tail_rejects_fused_glu_and_packed_merge() {
        let mut d = DevInst64 { op: DevOp::QuantFp8 as u16, ..Default::default() };
        d.t.fill(packet::dev::TENSOR_NONE16);
        assert!(fp8_attention_tail(&d));
        d.t[3] = 0;
        assert!(!fp8_attention_tail(&d));
        d.t[3] = packet::dev::TENSOR_NONE16;
        d.op = DevOp::FlashMerge as u16;
        for hd in [256, 512] {
            d.i[3] = hd;
            assert!(fp8_attention_tail(&d));
        }
        d.i[3] = 128;
        assert!(!fp8_attention_tail(&d));
        d.i[3] = 256;
        d.t[7] = 0;
        assert!(!fp8_attention_tail(&d));
    }

    #[test]
    fn fp8_flash256_grid_stays_within_work_and_route() {
        let mut span = LightSpan { count: 1, ..Default::default() };
        let d = &mut span.op[0].d;
        d.op = DevOp::FlashDecodeFp8 as u16;
        d.blocks = 132;
        d.i[0] = 128;
        d.i[1] = 16;
        d.i[2] = 8;
        d.i[5] = 1;
        d.i[6] = 256;
        assert_eq!(fp8_flash256_blocks(&span, 132), Some(1024));
        span.fused = 1;
        assert_eq!(fp8_flash256_blocks(&span, 132), None);
        span.fused = 0;
        span.op[0].d.i[0] = 64;
        assert_eq!(fp8_flash256_blocks(&span, 132), None);
    }

    fn fixture(batch: u32) -> (DevProg, Vec<DevTensor>) {
        let ordinary = DevInst64 {
            op: DevOp::Nop as u16,
            blocks: 1,
            ..Default::default()
        };
        let mut gemv = ordinary;
        gemv.op = DevOp::Gemv as u16;
        gemv.t.fill(packet::dev::TENSOR_NONE16);
        gemv.t[..3].copy_from_slice(&[0, 1, 2]);
        gemv.i[..3].copy_from_slice(&[batch, 48, 5120]);
        let stream: Vec<_> = (0..3)
            .map(|index| StreamEnt {
                inst: index,
                seg: index as u16,
                ..Default::default()
            })
            .collect();
        let tensors = [
            ("act.out", batch as u64 * 48 * 2),
            ("act.in", batch as u64 * 5120 * 2),
            ("model.layers.0.projection.weight", 48 * 5120 * 2),
        ]
        .into_iter()
        .map(|(name, bytes)| DevTensor {
            name: name.into(),
            bytes,
            init: None,
        })
        .collect();
        (
            DevProg {
                t: batch,
                role: packet::devbuild::ProgramRole::PrefillBucket { rows: batch },
                n_counter: 0,
                insts: vec![ordinary, gemv, ordinary],
                stream: stream.clone(),
                stream_ofs: vec![0],
                stream_len: vec![3],
                waits: vec![],
                succs: vec![],
                gq_stream: stream,
                gq_seg_ofs: vec![0, 1, 2, 3],
                l2_domains: 0,
            },
            tensors,
        )
    }

    #[test]
    fn prefill_glu_quant_requires_exact_shape_and_complete_segment() {
        let make = || {
            let (mut g, _) = fixture(128);
            g.insts[1].op = DevOp::QuantFp8 as u16;
            g.insts[1].i[..3].copy_from_slice(&[128, 15360, 0]);
            g.insts[1].t[..5].copy_from_slice(&[0, 1, 2, 3, 4]);
            g
        };
        let mut g = make();
        assert_eq!(prefill_glu_quant_segments(&g), vec![(1, vec![1])]);
        for field in [0, 1, 2] {
            let mut bad = make();
            bad.insts[1].i[field] = [1, 4096, 1][field];
            assert!(prefill_glu_quant_segments(&bad).is_empty());
        }
        for tensor in 0..5 {
            let mut bad = make();
            bad.insts[1].t[tensor] = packet::dev::TENSOR_NONE16;
            assert!(prefill_glu_quant_segments(&bad).is_empty());
        }
        g.insts[1].blocks = 2;
        assert!(prefill_glu_quant_segments(&g).is_empty());
        g.insts[1].blocks = 1;
        g.gq_stream[1].slice = 1;
        assert!(prefill_glu_quant_segments(&g).is_empty());
        g.gq_stream[1].slice = 0;
        g.gq_seg_ofs = vec![0, 1, 3];
        assert!(prefill_glu_quant_segments(&g).is_empty());
    }

    #[test]
    fn prefill_norm_rms_pairs_a_norm_residual_with_the_rmsnorm_reading_it() {
        let make = || {
            let (mut g, _) = fixture(128);
            let none = packet::dev::TENSOR_NONE16;
            g.insts[1].op = DevOp::NormResidual as u16;
            g.insts[1].i[..2].copy_from_slice(&[128, 3840]);
            g.insts[1].t = [0, 0, 1, 2, none, none, none, none];
            g.insts[2].op = DevOp::RmsNorm as u16;
            g.insts[2].i[..3].copy_from_slice(&[128, 3840, 0]);
            g.insts[2].t = [3, 0, 4, 5, 6, none, none, none];
            g.gq_seg_ofs = vec![0, 1, 3];
            g
        };
        assert_eq!(prefill_norm_rms_segments(&make()), vec![(1, vec![1, 2])]);
        let mut bad = make();
        bad.insts[2].t[1] = 7;
        assert!(prefill_norm_rms_segments(&bad).is_empty(), "the RmsNorm reads another tensor");
        let mut bad = make();
        bad.insts[2].i[2] = 64;
        assert!(prefill_norm_rms_segments(&bad).is_empty(), "output row offset");
        let mut bad = make();
        bad.insts[2].op = DevOp::QuantFp8 as u16;
        assert!(prefill_norm_rms_segments(&bad).is_empty());
        let mut bad = make();
        bad.insts[2].blocks = 2;
        assert!(prefill_norm_rms_segments(&bad).is_empty(), "a slice runs elsewhere");
        let mut bad = make();
        bad.gq_seg_ofs = vec![0, 1, 2, 3];
        assert!(prefill_norm_rms_segments(&bad).is_empty(), "the pair spans two segments");
    }

    fn roles() -> [u8; 3] {
        [
            plow_asset::segment_roles::INTERPRETER,
            plow_asset::segment_roles::CUBLASLT,
            plow_asset::segment_roles::INTERPRETER,
        ]
    }

    fn prefill_fixture(rows: u32, n: u32, k: u32) -> (DevProg, Vec<DevTensor>) {
        let (mut program, mut tensors) = fixture(rows);
        let op = &mut program.insts[1];
        op.op = DevOp::Gemm as u16;
        op.i[..].copy_from_slice(&[rows, n, k, 0, 0, 0, 17, 18]);
        tensors[0].bytes = u64::from(rows) * u64::from(n) * 2;
        tensors[1].bytes = u64::from(rows) * u64::from(k) * 2;
        tensors[2].bytes = u64::from(n) * u64::from(k) * 2;
        (program, tensors)
    }

    #[test]
    fn stream_order_replaces_only_library_counter_waits() {
        let (mut g, tensors) = fixture(4);
        g.n_counter = 3;
        g.succs = vec![0, 1, 2];
        g.waits = vec![
            packet::dev::Wait {
                id: 0,
                threshold: 1,
            },
            packet::dev::Wait {
                id: 1,
                threshold: 1,
            },
        ];
        for e in &mut g.stream {
            e.succ_ofs = e.inst;
            e.succ_len = 1;
            e.wait_ofs = e.inst.saturating_sub(1);
            e.wait_len = u16::from(e.inst != 0);
        }
        g.gq_stream = g.stream.clone();
        let routes = decode_segments(&g, &tensors, &roles()).unwrap();
        let waits = ordered_waits(&g, &routes, &[]).unwrap();
        assert_eq!(waits[0], g.waits[0]);
        assert_eq!(
            waits[1],
            packet::dev::Wait {
                id: 1,
                threshold: 0
            }
        );
        assert_eq!(g.waits[1].threshold, 1);

        g.succs[0] = 1;
        assert!(ordered_waits(&g, &routes, &[]).is_err());
        g.succs[0] = 0;
        g.waits[0].id = 2;
        assert!(ordered_waits(&g, &routes, &[]).is_err());
        g.waits[0].id = 0;
        g.waits[1].id = 3;
        assert!(ordered_waits(&g, &routes, &[]).is_err());
        g.waits[1].id = 1;
        g.waits[1].threshold = 2;
        assert!(ordered_waits(&g, &routes, &[]).is_err());
        g.waits[1].threshold = 1;
        g.stream[1].flags = packet::dev::SE_FINE;
        g.gq_stream = g.stream.clone();
        assert!(ordered_waits(&g, &routes, &[]).is_err());
    }

    #[test]
    fn accepts_packet_selected_bf16_projections() {
        for batch in [1, 4] {
            let (program, tensors) = fixture(batch);
            assert_eq!(
                decode_segments(&program, &tensors, &roles()).unwrap(),
                vec![
                    None,
                    Some(DecodeSegment {
                        instruction: 1,
                        m: batch,
                        n: 48,
                        k: 5120,
                        output_bytes: u64::from(batch) * 48 * 2,
                        input_bytes: u64::from(batch) * 5120 * 2,
                        weight_bytes: 48 * 5120 * 2,
                    }),
                    None,
                ]
            );
        }
    }

    #[test]
    fn accepts_only_measured_sm90_bf16_prefill_cells() {
        use plow_asset::segment_roles::{
            CUBLASLT_PREFILL_GEMMA4_26B_SHAPES, CUBLASLT_PREFILL_GEMMA4_SHAPES,
            CUBLASLT_PREFILL_ROWS, CUBLASLT_PREFILL_SPEECH_ROWS, CUBLASLT_PREFILL_WIDE_ROWS,
        };
        for &rows in CUBLASLT_PREFILL_ROWS
            .iter()
            .chain(&CUBLASLT_PREFILL_WIDE_ROWS)
            .chain(&CUBLASLT_PREFILL_SPEECH_ROWS)
        {
            for &(n, k) in CUBLASLT_PREFILL_GEMMA4_SHAPES
                .iter()
                .chain(&CUBLASLT_PREFILL_GEMMA4_26B_SHAPES)
            {
                let (program, tensors) = prefill_fixture(rows, n, k);
                let routes = prefill_segments(&program, &tensors, &roles(), "sm90a").unwrap();
                let route = routes[1].expect("measured projection route");
                assert_eq!((route.m, route.n, route.k), (rows, n, k));
            }
        }

        for (profile, rows, n, k) in [
            ("sm120", 128, 3840, 15360),
            ("sm90a", 48, 3840, 15360),
            ("sm90a", 1000, 3840, 15360),
            ("sm90a", 32768, 3840, 15360),
            ("sm90a", 128, 3840, 3840),
            ("sm90a", 128, 2816, 3840),
            ("sm90a", 1024, 3840, 3840),
        ] {
            let (program, tensors) = prefill_fixture(rows, n, k);
            assert!(prefill_segments(&program, &tensors, &roles(), profile).is_err());
        }
    }

    #[test]
    fn prefill_projection_rejects_bias_fp8_and_nonisolated_packets() {
        let (mut program, tensors) = prefill_fixture(128, 3840, 15360);
        program.insts[1].t[7] = 0;
        assert!(prefill_segments(&program, &tensors, &roles(), "sm90a").is_err());
        program.insts[1].t[7] = packet::dev::TENSOR_NONE16;
        program.insts[1].op = DevOp::GemmFp8 as u16;
        assert!(prefill_segments(&program, &tensors, &roles(), "sm90a").is_err());
        program.insts[1].op = DevOp::Gemm as u16;
        program.gq_stream[0].seg = 1;
        assert!(prefill_segments(&program, &tensors, &roles(), "sm90a").is_err());
    }

    #[test]
    fn fp8_prefill_validates_scale_contract_and_native_exceptions() {
        for rows in [128, 256, 512, 1024, 2048, 4096, 8192] {
            for &(n, k) in &plow_asset::segment_roles::CUBLASLT_PREFILL_GEMMA4_SHAPES {
                let (mut program, mut tensors) = prefill_fixture(rows, n, k);
                program.insts[1].op = DevOp::GemmFp8 as u16;
                program.insts[1].t[3..5].copy_from_slice(&[3, 4]);
                tensors[1].bytes /= 2;
                tensors[2].bytes /= 2;
                for (name, bytes) in [
                    ("input.scale", u64::from(rows) * 4),
                    ("weight.scale", u64::from(n) * 4),
                ] {
                    tensors.push(DevTensor {
                        name: name.into(),
                        bytes,
                        init: None,
                    });
                }
                let result = prefill_segments(&program, &tensors, &roles(), "sm90a");
                if (n, k) == (3840, 15360) && rows >= 2048 {
                    assert!(result.is_err());
                    continue;
                }
                let route = result.unwrap()[1].unwrap();
                assert_eq!(route.input_bytes, u64::from(rows) * u64::from(k));
                assert_eq!(route.weight_bytes, u64::from(n) * u64::from(k));
                for handle in 0..5 {
                    let saved = tensors[handle].bytes;
                    tensors[handle].bytes -= 1;
                    assert!(prefill_segments(&program, &tensors, &roles(), "sm90a").is_err());
                    tensors[handle].bytes = saved;
                }
                for slot in 3..5 {
                    let saved = program.insts[1].t[slot];
                    for bad in [packet::dev::TENSOR_NONE16, 0, 1, 2] {
                        program.insts[1].t[slot] = bad;
                        assert!(prefill_segments(&program, &tensors, &roles(), "sm90a").is_err());
                    }
                    program.insts[1].t[slot] = saved;
                }
                program.insts[1].i[4] = 1;
                assert!(prefill_segments(&program, &tensors, &roles(), "sm90a").is_err());
            }
        }
    }

    #[test]
    fn fp8_decode_requires_independent_scales_and_library_role() {
        for rows in [32, 64, 128] {
            let (mut program, mut tensors) = prefill_fixture(rows, 3840, 15360);
            let op = &mut program.insts[1];
            op.op = DevOp::GemmFp8 as u16;
            op.i[6..].fill(0);
            op.t[3..5].copy_from_slice(&[3, 4]);
            tensors[1].bytes /= 2;
            tensors[2].bytes /= 2;
            for (name, bytes) in [("input.scale", u64::from(rows) * 4), ("weight.scale", 3840 * 4)] {
                tensors.push(DevTensor { name: name.into(), bytes, init: None });
            }
            let route = decode_segments(&program, &tensors, &roles()).unwrap()[1].unwrap();
            assert_eq!(route.input_bytes, u64::from(rows) * 15360);
            assert_eq!(route.output_bytes, u64::from(rows) * 3840 * 2);
            let mut native_roles = roles();
            native_roles[1] = plow_asset::segment_roles::NATIVE_DECODE_TC;
            assert!(decode_segments(&program, &tensors, &native_roles).is_err());
            for handle in 0..5 {
                tensors[handle].bytes -= 1;
                assert!(decode_segments(&program, &tensors, &roles()).is_err());
                tensors[handle].bytes += 1;
            }
            for slot in 3..5 {
                let saved = program.insts[1].t[slot];
                for bad in [packet::dev::TENSOR_NONE16, 0, 1, 2, 7 - slot as u16] {
                    program.insts[1].t[slot] = bad;
                    assert!(decode_segments(&program, &tensors, &roles()).is_err());
                }
                program.insts[1].t[slot] = saved;
            }
            program.insts[1].i[6] = 1;
            assert!(decode_segments(&program, &tensors, &roles()).is_err());
        }
    }

    #[test]
    fn native_projection_reuses_geometry_and_rejects_bias() {
        let (mut program, tensors) = fixture(4);
        let native_roles = [0, plow_asset::segment_roles::NATIVE_DECODE_TC, 0];
        let routes = decode_segments(&program, &tensors, &native_roles).unwrap();
        assert_eq!(
            routes,
            decode_segments(&program, &tensors, &roles()).unwrap()
        );
        program.insts[1].t[7] = 0;
        assert!(decode_segments(&program, &tensors, &native_roles).is_err());
    }

    #[test]
    fn rejects_invalid_precision_geometry_and_extents() {
        let (program, tensors) = fixture(4);
        let (mut bad, _) = fixture(program.t);
        bad.insts[1].op = DevOp::GemvFp8 as u16;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.insts[1].i[4] = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.insts[1].i[0] = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.insts[1].t[0] = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (_, mut small) = fixture(program.t);
        small[0].bytes -= 1;
        assert!(decode_segments(&program, &small, &roles()).is_err());
        let (mut overflow, _) = fixture(program.t);
        overflow.insts[1].i[1] = u32::MAX;
        overflow.insts[1].i[2] = u32::MAX;
        assert!(decode_segments(&overflow, &tensors, &roles()).is_err());
    }

    #[test]
    fn rejects_invalid_roles_and_queue_windows() {
        let (program, tensors) = fixture(1);
        assert!(decode_segments(&program, &tensors, &[0, 0, 0]).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.gq_stream[0].seg = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.stream[0].seg = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.gq_seg_ofs[1] = 4;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
        let (mut bad, _) = fixture(program.t);
        bad.l2_domains = 1;
        assert!(decode_segments(&bad, &tensors, &roles()).is_err());
    }
}
