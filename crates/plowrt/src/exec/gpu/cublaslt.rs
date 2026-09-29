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
}

struct HeadTail {
    kernel: HeadKernel,
    rows: u32,
    n: u32,
    k: u32,
    n0: u32,
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
        let mut params = [
            &mut c as *mut u64 as *mut std::ffi::c_void,
            &mut x as *mut u64 as *mut std::ffi::c_void,
            &mut w as *mut u64 as *mut std::ffi::c_void,
            &mut src as *mut u64 as *mut std::ffi::c_void,
            &mut n as *mut u32 as *mut std::ffi::c_void,
            &mut k as *mut u32 as *mut std::ffi::c_void,
            &mut n0 as *mut u32 as *mut std::ffi::c_void,
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

struct LightLaunch {
    function: KernelFn,
    xs: LightX,
    /// `None`: the one-instruction `plow_<arch>_light` kernel; `Some(n)`: `light_attn` over
    /// `n` instructions from `instruction`.
    count: Option<u32>,
    instruction: u32,
    blocks: u32,
    smem: u32,
}

impl LightRoute {
    fn run(&self, stream: &CudaStream) -> Result<()> {
        for launch in &self.launches {
            let mut arg = self.kernarg;
            let mut instruction = launch.instruction;
            let mut count = launch.count.unwrap_or(0);
            let mut xs = launch.xs;
            let mut params = [
                &mut arg as *mut DevProgram as *mut std::ffi::c_void,
                &mut instruction as *mut u32 as *mut std::ffi::c_void,
                &mut count as *mut u32 as *mut std::ffi::c_void,
                &mut xs as *mut LightX as *mut std::ffi::c_void,
            ];
            let params = if launch.count.is_some() { &mut params[..] } else { &mut params[..2] };
            self.be.launch_kernel(
                launch.function,
                launch.blocks,
                BLOCK,
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
    /// `plow_<arch>_light_attn`, its arena and the head dim it carries.
    attn: Option<(KernelFn, u32, u32)>,
    /// `plow_<arch>_light_head` (unaligned lm_head tail).
    pub(super) head: Option<KernelFn>,
}

/// The light kernels of the decode object `module` (`<stem>` = `interp_<arch>`), when it has
/// them and `PLOW_DECODE_LIGHT` is on. `smem` = the object's arena, which `light_attn` takes.
pub(super) fn light_functions(
    be: &CudaBackend,
    module: &Module,
    stem: &str,
    smem: u32,
) -> Result<Option<LightFunctions>> {
    if !crate::config::RuntimeConfig::get().nv.decode_light
        || be.module_global_u32(module, "plow_light_abi")? != Some(1)
    {
        return Ok(None);
    }
    let arch = stem.trim_start_matches("interp_");
    let single = be.get_function(module, &format!("plow_{arch}_light"))?;
    let attn = match be.module_global_u32(module, "plow_light_attn_hd")? {
        Some(hd) => {
            let f = be.get_function(module, &format!("plow_{arch}_light_attn"))?;
            be.set_max_dynamic_smem(f, smem)?;
            Some((f, smem, hd))
        }
        _ => None,
    };
    let head = be.get_function(module, &format!("plow_{arch}_light_head")).ok();
    Ok(Some(LightFunctions { single, attn, head }))
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
            if let [inst] = insts[..] {
                if matches!(op(inst), Some(DevOp::AddNorm | DevOp::Glu)) {
                    return Some((seg, vec![(inst, 0)]));
                }
            }
            let attn_hd = functions.attn.map(|(_, _, hd)| hd);
            let attn = attn_hd.is_some()
                && insts.iter().all(|&inst| {
                    let d = &g.insts[inst];
                    match op(inst) {
                        Some(DevOp::HeadNormRope) => Some(d.i[2]) == attn_hd && d.i[5] == 0,
                        Some(DevOp::FlashDecode) => Some(d.i[6]) == attn_hd,
                        _ => false,
                    }
                });
            if !attn {
                return None;
            }
            let mut levels: Vec<(usize, usize)> = Vec::new();
            for &inst in &insts {
                let joins = levels.last().is_some_and(|&(lo, n)| {
                    lo + n == inst
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
) {
    for (seg, levels) in light {
        let fused = fusions.iter().find(|f| f.segment + 3 == *seg).zip(scratch).map(|(f, s)| {
            LightX {
                base: s.base,
                ..f.xs
            }
        });
        if routes.len() <= *seg {
            routes.resize_with(seg + 1, || None);
        }
        let launches = levels
            .iter()
            .enumerate()
            .map(|(level, &(inst, n))| {
                let xs = fused.filter(|_| level == 0).unwrap_or_default();
                let blocks = (inst..inst + n.max(1))
                    .map(|i| u32::from(g.insts[i].blocks))
                    .max()
                    .unwrap_or(1);
                match (n, functions.attn) {
                    (n, Some((function, smem, _))) if n > 0 => LightLaunch {
                        function,
                        xs,
                        count: Some(n as u32),
                        instruction: inst as u32,
                        blocks,
                        smem,
                    },
                    _ => LightLaunch {
                        function: functions.single,
                        xs,
                        count: None,
                        instruction: inst as u32,
                        blocks,
                        smem: 0,
                    },
                }
            })
            .collect();
        routes[*seg] = Some(LibraryRoute::Light(LightRoute {
            be: Arc::clone(be),
            kernarg,
            launches,
            _scratch: fused.and(scratch.cloned()),
        }));
    }
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
    Folded,
}

impl ProjectionPlan {
    fn run(&self, input: u64, weight: u64, output: u64, stream: &CudaStream) -> Result<()> {
        match self {
            Self::Lt(p) => p.run(input, weight, output, stream),
            Self::Native(p) => p.run(input, weight, output, stream),
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
) -> Result<Vec<Option<CublasLtDecodeRoute>>> {
    let mut routes: Vec<Option<CublasLtDecodeRoute>> = Vec::new();
    if segments.is_empty() {
        return Ok(routes);
    }
    let pair_lt = match backend {
        ProjectionBackend::Lt(lt) if pair => Some(lt),
        _ => None,
    };
    // A rung pairs exactly where its template (the widest rung) paired, and pins that pair's
    // algorithm.
    let template_pair = |index: usize| -> Result<Option<Option<&crate::device::cuda::lt::Plan>>> {
        let Some(routes) = templates else {
            return Ok(Some(None));
        };
        match (routes.get(index), routes.get(index + 1)) {
            (
                Some(Some(LibraryRoute::Projection(route))),
                Some(Some(LibraryRoute::Projection(next))),
            ) if next.folded() => match route.plan.as_ref() {
                ProjectionPlan::Lt(plan) => Ok(Some(Some(plan.as_ref()))),
                _ => Err(RuntimeError::Rejected("decode projection backend changed".into())),
            },
            _ => Ok(None),
        }
    };
    let mut plans = std::collections::HashMap::new();
    let mut pair_plans = std::collections::HashMap::new();
    let mut pairs = 0usize;
    let mut index = 0;
    while index < segments.len() {
        let Some(segment) = segments[index] else {
            routes.push(None);
            index += 1;
            continue;
        };
        // An lm_head with an unaligned vocab: cuBLASLt serves its aligned columns (see HeadTail).
        let tail = head
            .filter(|_| segment.n % 16 != 0 && segment.n > 16 && matches!(backend, ProjectionBackend::Lt(_)))
            .filter(|h| u64::from(segment.m) * u64::from(segment.n & !15) * 2 <= h.scratch.len)
            .cloned();
        let key = (
            segment.m,
            if tail.is_some() { segment.n & !15 } else { segment.n },
            segment.k,
        );
        let ops = operands(&segment, insts, devp)?;
        let [output, input, weight] = ops;
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
            let plan = lt.plan(segment.m, fusion.n_total, segment.k, weight, template)?;
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
                        ProjectionPlan::Lt(lt.pair_plan(key.0, key.1, key.2, w0, p, pair_template)?),
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
                        ProjectionPlan::Lt(lt.plan(key.0, key.1, key.2, weight, template)?)
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
            }),
        }));
        index += 1;
    }
    tracing::info!(
        segments = routes.len(),
        projections = routes.iter().flatten().count(),
        plans = plans.len(),
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

    fn enqueue_decode_chain(&self) -> Result<()> {
        let segments = self
            .cublaslt_decode
            .len()
            .max(self.decode_packet_roles.len())
            .max(1);
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
        let valid_op = match phase {
            ProjectionPhase::Decode => op.op == DevOp::Gemv as u16,
            ProjectionPhase::Prefill(profile) => {
                matches!(
                    DevOp::from_u16(op.op),
                    Some(DevOp::Gemm | DevOp::GemmMed | DevOp::GemmSmall)
                ) && plow_asset::segment_roles::cublaslt_prefill_bf16(
                    profile, op.i[0], op.i[1], op.i[2],
                )
            }
        };
        let valid_immediates = match phase {
            ProjectionPhase::Decode => op.i[3..].iter().all(|&value| value == 0),
            ProjectionPhase::Prefill(_) => op.i[3..6].iter().all(|&value| value == 0),
        };
        if !valid_op
            || op.t[3..].iter().any(|&t| t != packet::dev::TENSOR_NONE16)
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
        let input_bytes = bytes(m, k)?;
        let weight_bytes = bytes(n, k)?;
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
