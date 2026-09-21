//! cuBLASLt grouped-GEMM route for the Gemma-4 MoE prefill experts (`PLOW_MOE_PF_LT`).
//!
//! A packet emitted with `PLOW_EMIT_MOE_PF_LT` isolates each layer's `MoeGroupGluGemmaPf` +
//! `MoeGroupDownGemmaPf` pair in a `MOE_PREFILL_CUBLASLT` segment. The route serves that segment
//! with two grouped matmuls whose per-expert row counts and matrix pointers live on the device
//! (no host sync, graph-capturable), plus four glue kernels (`runtime/nvidia/moe_lt_sm90.cu`).
//! `MoeAlignGemmaPf` still runs in the interpreter and its tables are the route's input, so the
//! gate-scaled f32 `part[token*k+slot]` contract and the combine op are unchanged.

use super::*;
use crate::asset::devblob::DevTensor;
use crate::device::cuda::lt::{GroupedDims, GroupedPlan, Lt};

pub(super) const OBJECT_FILE: &str = "interp_sm90a_moe_lt.cubin";
const GLUE_THREADS: u32 = 256;

/// One validated `MOE_PREFILL_CUBLASLT` segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MoeLtSegment {
    pub(super) glu: usize,
    pub(super) down: usize,
    hidden: u32,
    inter: u32,
    experts: u32,
    act: u32,
    /// Gathered-row capacity of the align tables and `fu_g`.
    capacity: u32,
    average_rows: u32,
}

pub(super) fn segments(
    program: &DevProg,
    tensors: &[DevTensor],
    roles: &[u8],
) -> Result<Vec<Option<MoeLtSegment>>> {
    let fail = || RuntimeError::Rejected("invalid packet-declared MoE prefill segment".into());
    let count = program.gq_seg_ofs.len().checked_sub(1).ok_or_else(fail)?;
    if roles.len() != count {
        return Err(fail());
    }
    let mut routes = vec![None; count];
    for (segment, bounds) in program.gq_seg_ofs.windows(2).enumerate() {
        if roles[segment] != plow_asset::segment_roles::MOE_PREFILL_CUBLASLT {
            continue;
        }
        let entries = program
            .gq_stream
            .get(bounds[0] as usize..bounds[1] as usize)
            .ok_or_else(fail)?;
        let glu = entries.first().ok_or_else(fail)?.inst as usize;
        let down = glu + 1;
        let (Some(align), Some(g), Some(d)) = (
            glu.checked_sub(1).and_then(|i| program.insts.get(i)),
            program.insts.get(glu),
            program.insts.get(down),
        ) else {
            return Err(fail());
        };
        let [inter, hidden, experts] = [g.i[0], g.i[1], g.i[2]];
        let bytes = |handle: u16| tensors.get(handle as usize).map_or(0, |t| t.bytes);
        let capacity = bytes(g.t[0]) / (u64::from(inter.max(1)) * 2);
        let none = packet::dev::TENSOR_NONE16;
        if align.op != DevOp::MoeAlignGemmaPf as u16
            || g.op != DevOp::MoeGroupGluGemmaPf as u16
            || d.op != DevOp::MoeGroupDownGemmaPf as u16
            || entries
                .iter()
                .any(|e| e.inst as usize != glu && e.inst as usize != down)
            || program.stream.iter().any(|e| {
                (e.inst as usize == glu || e.inst as usize == down) != (e.seg as usize == segment)
            })
            || g.t[5..].iter().chain(&d.t[6..]).any(|&t| t != none)
            || [g.t[3], g.t[4]] != [align.t[0], align.t[2]]
            || [d.t[1], d.t[2], d.t[3], d.t[4], d.t[5]]
                != [g.t[0], g.t[2], align.t[0], align.t[3], align.t[4]]
            || [d.i[0], d.i[1], d.i[2]] != [hidden, inter, experts]
            || align.i[1] != experts
            || experts == 0
            || inter == 0
            || hidden == 0
            || inter % 8 != 0
            || hidden % 8 != 0
            || g.i[5] > 1
            || capacity == 0
            || capacity > u64::from(u32::MAX)
            || bytes(g.t[2]) != u64::from(experts) * 16
            || bytes(g.t[3]) < (3 * u64::from(experts) + 1) * 4
            || [g.t[4], d.t[4], d.t[5]].iter().any(|&t| bytes(t) < capacity * 4)
        {
            return Err(fail());
        }
        routes[segment] = Some(MoeLtSegment {
            glu,
            down,
            hidden,
            inter,
            experts,
            act: g.i[5],
            capacity: capacity as u32,
            average_rows: (u64::from(program.t) * u64::from(align.i[2]) / u64::from(experts))
                as u32,
        });
    }
    Ok(routes)
}

/// Glue kernels, scratch and device-side group tables shared by every route of an engine.
pub(super) struct MoeLt {
    be: Arc<CudaBackend>,
    lt: Arc<Lt>,
    _module: Arc<DecodeModule>,
    setup: KernelFn,
    gather: KernelFn,
    glu: KernelFn,
    scatter: KernelFn,
    blocks: u32,
    hidden: u32,
    inter: u32,
    experts: u32,
    /// `xs` (the gathered input; reused as the down output once gate|up consumed it) and `gu`.
    xs: DeviceMem,
    gu: DeviceMem,
    /// i32 `[rows | 2I | H | I] x experts`, then u64 `[xs | gu | fu | dn] x experts`.
    tables: DeviceMem,
    /// Per expert-weight table: device `[gate_up | down] x experts` matrix pointers.
    weights: std::sync::Mutex<std::collections::HashMap<u64, Arc<DeviceMem>>>,
    plans: std::sync::Mutex<std::collections::HashMap<u32, [Arc<GroupedPlan>; 2]>>,
}

pub(super) struct MoeLtRoute {
    owner: Arc<MoeLt>,
    plans: [Arc<GroupedPlan>; 2],
    weights: Arc<DeviceMem>,
    meta: u64,
    row_token: u64,
    row_partidx: u64,
    row_gate: u64,
    xn2: u64,
    fu: u64,
    part: u64,
    act: u32,
}

impl MoeLt {
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        lt: &Arc<Lt>,
        directories: &[&Path],
        profile: &str,
        shape: &MoeLtSegment,
    ) -> Result<Arc<Self>> {
        let image = directories
            .iter()
            .find_map(|dir| std::fs::read(dir.join(OBJECT_FILE)).ok())
            .ok_or_else(|| {
                RuntimeError::Rejected(format!(
                    "PLOW_MOE_PF_LT needs {OBJECT_FILE} next to the prefill objects"
                ))
            })?;
        if profile != "sm90a"
            || plow_asset::cubin::inspect(&image).is_none_or(|i| i.sm != 90)
            || plow_asset::cubin::global_u32(&image, "plow_moe_lt_abi") != Some(1)
        {
            return Err(RuntimeError::Rejected(format!(
                "{OBJECT_FILE}: not an sm90a MoE cuBLASLt glue object of ABI 1"
            )));
        }
        let module = DecodeModule::load(be, &image)?;
        let function = |name: &str| be.get_function(&module, name);
        let (hidden, inter, experts) = (shape.hidden, shape.inter, shape.experts);
        let capacity = u64::from(shape.capacity);
        let xs = be.alloc(0, capacity * u64::from(hidden) * 2)?;
        let gu = be.alloc(0, capacity * u64::from(inter) * 4)?;
        let tables = be.alloc(0, u64::from(experts) * (4 * 4 + 4 * 8))?;
        let constants: Vec<i32> = [0, 2 * inter, hidden, inter]
            .into_iter()
            .flat_map(|value| std::iter::repeat_n(value as i32, experts as usize))
            .collect();
        be.upload(&tables, 0, bytemuck::cast_slice(&constants))?;
        let gather = function("plow_moe_lt_gather")?;
        let blocks = be.occupancy_blocks_per_sm(gather, GLUE_THREADS, 0)?.max(1) * be.sm_count();
        tracing::info!(
            scratch_mib = (xs.len + gu.len) >> 20,
            capacity,
            blocks,
            "MoE prefill cuBLASLt grouped route loaded"
        );
        Ok(Arc::new(Self {
            be: Arc::clone(be),
            lt: Arc::clone(lt),
            setup: function("plow_moe_lt_setup")?,
            gather,
            glu: function("plow_moe_lt_glu")?,
            scatter: function("plow_moe_lt_scatter")?,
            _module: module,
            blocks,
            hidden,
            inter,
            experts,
            xs,
            gu,
            tables,
            weights: Default::default(),
            plans: Default::default(),
        }))
    }

    fn table(&self, index: u64) -> u64 {
        self.tables.base + index * u64::from(self.experts) * 4
    }

    fn pointers(&self, index: u64) -> u64 {
        self.table(4) + index * u64::from(self.experts) * 8
    }

    fn plans(self: &Arc<Self>, average_rows: u32) -> Result<[Arc<GroupedPlan>; 2]> {
        let mut plans = self.plans.lock().expect("MoE Lt plans");
        if let Some(found) = plans.get(&average_rows) {
            return Ok(found.clone());
        }
        let plan = |n: u32, n_table: u64, k: u32, k_table: u64| {
            self.lt.grouped_plan(&GroupedDims {
                groups: self.experts,
                n,
                k,
                rows: self.table(0),
                n_array: self.table(n_table),
                k_array: self.table(k_table),
                average_rows,
            })
        };
        let pair = [
            plan(2 * self.inter, 1, self.hidden, 2)?,
            plan(self.hidden, 2, self.inter, 3)?,
        ];
        plans.insert(average_rows, pair.clone());
        Ok(pair)
    }

    /// Split the engine's `[gate_up, down] x experts` table into the two pointer arrays a
    /// grouped matmul takes.
    fn weights(&self, table: &DeviceMem) -> Result<Arc<DeviceMem>> {
        let mut cache = self.weights.lock().expect("MoE Lt weights");
        if let Some(found) = cache.get(&table.base) {
            return Ok(Arc::clone(found));
        }
        let experts = self.experts as usize;
        let mut interleaved = vec![0u64; experts * 2];
        self.be
            .download(table, 0, bytemuck::cast_slice_mut(&mut interleaved))?;
        if interleaved.iter().any(|&p| p == 0 || p % 16 != 0) {
            return Err(RuntimeError::Rejected(
                "MoE cuBLASLt route needs bound, 16-byte aligned BF16 expert tensors".into(),
            ));
        }
        let split: Vec<u64> = (0..2)
            .flat_map(|which| interleaved.iter().skip(which).step_by(2).copied())
            .collect();
        let device = Arc::new(self.be.alloc(0, (experts * 16) as u64)?);
        self.be.upload(&device, 0, bytemuck::cast_slice(&split))?;
        cache.insert(table.base, Arc::clone(&device));
        Ok(device)
    }

    pub(super) fn route(
        self: &Arc<Self>,
        segment: &MoeLtSegment,
        insts: &mut [DevInst64],
        devp: &[DeviceMem],
    ) -> Result<MoeLtRoute> {
        let (glu, down) = (insts[segment.glu], insts[segment.down]);
        if [segment.hidden, segment.inter, segment.experts]
            != [self.hidden, self.inter, self.experts]
            || u64::from(segment.capacity) * u64::from(self.hidden) * 2 > self.xs.len
        {
            return Err(RuntimeError::Rejected(
                "MoE cuBLASLt segments disagree on the expert geometry".into(),
            ));
        }
        let base = |handle: u16| devp[handle as usize].base;
        let route = MoeLtRoute {
            owner: Arc::clone(self),
            plans: self.plans(segment.average_rows)?,
            weights: self.weights(&devp[glu.t[2] as usize])?,
            meta: base(glu.t[3]),
            row_token: base(glu.t[4]),
            row_partidx: base(down.t[4]),
            row_gate: base(down.t[5]),
            xn2: base(glu.t[1]),
            fu: base(glu.t[0]),
            part: base(down.t[0]),
            act: segment.act,
        };
        insts[segment.glu].op = DevOp::Nop as u16;
        insts[segment.down].op = DevOp::Nop as u16;
        Ok(route)
    }
}

impl MoeLtRoute {
    pub(super) fn run(&self, stream: &CudaStream) -> Result<()> {
        let o = &*self.owner;
        let arg = |value: &mut u64| (value as *mut u64).cast::<std::ffi::c_void>();
        let arg32 = |value: &mut u32| (value as *mut u32).cast::<std::ffi::c_void>();
        let (mut meta, mut xn2, mut fu, mut part) = (self.meta, self.xn2, self.fu, self.part);
        let (mut row_token, mut row_partidx, mut row_gate) =
            (self.row_token, self.row_partidx, self.row_gate);
        let (mut xs, mut gu) = (o.xs.base, o.gu.base);
        let (mut rows, mut pointers) = (o.table(0), o.pointers(0));
        let (mut experts, mut hidden, mut inter) = (o.experts, o.hidden, o.inter);
        let (mut act, mut blocks) = (self.act, o.blocks);
        let mut dn = xs;

        let mut params = [
            arg(&mut rows),
            arg(&mut pointers),
            arg(&mut meta),
            arg(&mut xs),
            arg(&mut gu),
            arg(&mut fu),
            arg(&mut dn),
            arg32(&mut experts),
            arg32(&mut hidden),
            arg32(&mut inter),
        ];
        o.be.launch_kernel(o.setup, 1, GLUE_THREADS, 0, &mut params, Some(stream))?;
        let mut params = [
            arg(&mut xs),
            arg(&mut xn2),
            arg(&mut row_token),
            arg(&mut meta),
            arg32(&mut experts),
            arg32(&mut hidden),
            arg32(&mut blocks),
        ];
        o.be.launch_kernel(o.gather, o.blocks, GLUE_THREADS, 0, &mut params, Some(stream))?;
        self.plans[0].run(o.pointers(0), self.weights.base, o.pointers(1), stream)?;
        let mut params = [
            arg(&mut fu),
            arg(&mut gu),
            arg(&mut meta),
            arg32(&mut experts),
            arg32(&mut inter),
            arg32(&mut act),
            arg32(&mut blocks),
        ];
        o.be.launch_kernel(o.glu, o.blocks, GLUE_THREADS, 0, &mut params, Some(stream))?;
        self.plans[1].run(
            o.pointers(2),
            self.weights.base + u64::from(o.experts) * 8,
            o.pointers(3),
            stream,
        )?;
        let mut params = [
            arg(&mut part),
            arg(&mut dn),
            arg(&mut row_partidx),
            arg(&mut row_gate),
            arg(&mut meta),
            arg32(&mut experts),
            arg32(&mut hidden),
            arg32(&mut blocks),
        ];
        o.be.launch_kernel(o.scatter, o.blocks, GLUE_THREADS, 0, &mut params, Some(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use packet::dev::StreamEnt;

    const ROLE: u8 = plow_asset::segment_roles::MOE_PREFILL_CUBLASLT;
    const CAPACITY: u64 = 1024 * 8 + 128 * 128;

    fn fixture() -> (DevProg, Vec<DevTensor>) {
        let mut insts = vec![
            DevInst64 {
                op: DevOp::Nop as u16,
                blocks: 1,
                ..Default::default()
            };
            4
        ];
        for inst in &mut insts {
            inst.t.fill(packet::dev::TENSOR_NONE16);
        }
        // Handles: 0 fug, 1 xn2, 2 ewt, 3 meta, 4 rowtok, 5 part, 6 rowpart, 7 rowgate, 8 table.
        insts[0].op = DevOp::MoeAlignGemmaPf as u16;
        insts[0].t[..5].copy_from_slice(&[3, 8, 4, 6, 7]);
        insts[0].i[..3].copy_from_slice(&[1024, 128, 8]);
        insts[1].op = DevOp::MoeGroupGluGemmaPf as u16;
        insts[1].t[..5].copy_from_slice(&[0, 1, 2, 3, 4]);
        insts[1].i[..3].copy_from_slice(&[704, 2816, 128]);
        insts[2].op = DevOp::MoeGroupDownGemmaPf as u16;
        insts[2].t[..6].copy_from_slice(&[5, 0, 2, 3, 6, 7]);
        insts[2].i[..3].copy_from_slice(&[2816, 704, 128]);
        let stream: Vec<_> = [(0, 0), (1, 1), (2, 1), (3, 2)]
            .into_iter()
            .map(|(inst, seg)| StreamEnt {
                inst,
                seg,
                ..Default::default()
            })
            .collect();
        let tensors = [
            CAPACITY * 704 * 2,
            1024 * 2816 * 2,
            128 * 16,
            (3 * 128 + 2) * 4,
            CAPACITY * 4,
            1024 * 8 * 2816 * 4,
            CAPACITY * 4,
            CAPACITY * 4,
            1024 * 8 * 8,
        ]
        .into_iter()
        .map(|bytes| DevTensor {
            name: "act".into(),
            bytes,
            init: None,
        })
        .collect();
        (
            DevProg {
                t: 1024,
                role: packet::devbuild::ProgramRole::PrefillBucket { rows: 1024 },
                n_counter: 0,
                insts,
                stream: stream.clone(),
                stream_ofs: vec![0],
                stream_len: vec![4],
                waits: vec![],
                succs: vec![],
                gq_stream: stream,
                gq_seg_ofs: vec![0, 1, 3, 4],
                l2_domains: 0,
            },
            tensors,
        )
    }

    #[test]
    fn accepts_the_emitted_glu_down_pair() {
        let (program, tensors) = fixture();
        let routes = segments(&program, &tensors, &[0, ROLE, 0]).unwrap();
        let route = routes[1].expect("routed pair");
        assert_eq!((route.glu, route.down), (1, 2));
        assert_eq!(
            (route.hidden, route.inter, route.experts, route.average_rows),
            (2816, 704, 128, 64)
        );
        assert_eq!(u64::from(route.capacity), CAPACITY);
        assert!(routes[0].is_none() && routes[2].is_none());
        assert!(segments(&program, &tensors, &[0, 0, 0])
            .unwrap()
            .iter()
            .all(Option::is_none));
        assert!(segments(&program, &tensors, &[0, ROLE]).is_err());
    }

    #[test]
    fn packet_role_validator_takes_the_two_instruction_library_segment() {
        let (program, tensors) = fixture();
        assert_eq!(
            packet_role_segments(&program, &[0, ROLE, 0], &tensors).unwrap(),
            [0, ROLE, 0]
        );
        // A segment of one instruction (or of three) is not this role.
        for (bounds, segs) in [([0, 1, 2, 4], [0, 1, 2, 2]), ([0, 1, 4, 4], [0, 1, 1, 1])] {
            let (mut split, tensors) = fixture();
            split.gq_seg_ofs = bounds.to_vec();
            for entry in split.stream.iter_mut().chain(&mut split.gq_stream) {
                entry.seg = segs[entry.inst as usize];
            }
            assert!(packet_role_segments(&split, &[0, ROLE, 0], &tensors).is_err());
        }
    }

    #[test]
    fn rejects_foreign_operands_geometry_and_short_tables() {
        let roles = [0, ROLE, 0];
        let edits: [fn(&mut DevProg, &mut Vec<DevTensor>); 6] = [
            |p, _| p.insts[2].t[1] = 1,
            |p, _| p.insts[2].i[1] = 712,
            |p, _| {
                p.insts[1].i[0] = 700;
                p.insts[2].i[1] = 700;
            },
            |p, _| p.insts[0].op = DevOp::Nop as u16,
            |p, _| {
                p.stream[3].seg = 1;
                p.gq_stream[3].seg = 1;
                p.gq_seg_ofs = vec![0, 1, 4, 4];
            },
            |_, t| t[6].bytes -= 4,
        ];
        for edit in edits {
            let (mut program, mut tensors) = fixture();
            edit(&mut program, &mut tensors);
            assert!(segments(&program, &tensors, &roles).is_err());
        }
    }
}
