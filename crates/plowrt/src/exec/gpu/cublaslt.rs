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
}

impl CublasLtDecodeRoute {
    pub(super) fn run(&self, stream: &CudaStream) -> Result<()> {
        self.plan.run(self.input, self.weight, self.output, stream)
    }
}

pub(super) enum ProjectionBackend {
    Lt(Arc<crate::device::cuda::lt::Lt>),
    Native(Arc<native_decode::Native>),
}

enum ProjectionPlan {
    Lt(Arc<crate::device::cuda::lt::Plan>),
    Native(native_decode::Plan),
}

impl ProjectionPlan {
    fn run(&self, input: u64, weight: u64, output: u64, stream: &CudaStream) -> Result<()> {
        match self {
            Self::Lt(p) => p.run(input, weight, output, stream),
            Self::Native(p) => p.run(input, weight, output, stream),
        }
    }
}

pub(super) fn ordered_waits(
    g: &DevProg,
    segments: &[Option<DecodeSegment>],
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
    for (seg, route) in segments.iter().enumerate() {
        if let Some(route) = route {
            if placement.get(route.instruction) != Some(&Some(seg as u16)) {
                return Err(reject());
            }
            library[route.instruction] = true;
        }
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

pub(super) fn prepare_routes(
    backend: &ProjectionBackend,
    segments: Vec<Option<DecodeSegment>>,
    insts: &mut [DevInst64],
    devp: &[DeviceMem],
    templates: Option<&[Option<CublasLtDecodeRoute>]>,
) -> Result<Vec<Option<CublasLtDecodeRoute>>> {
    let mut routes = Vec::new();
    if segments.is_empty() {
        return Ok(routes);
    }
    let mut plans = std::collections::HashMap::new();
    for (index, segment) in segments.into_iter().enumerate() {
        let route = if let Some(segment) = segment {
            let d = &mut insts[segment.instruction];
            let key = (segment.m, segment.n, segment.k);
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
            let plan = match plans.entry(key) {
                std::collections::hash_map::Entry::Occupied(e) => Arc::clone(e.get()),
                std::collections::hash_map::Entry::Vacant(e) => {
                    let template = templates
                        .map(|routes| {
                            routes.get(index).and_then(Option::as_ref).ok_or_else(|| {
                                RuntimeError::Rejected(
                                    "cuBLASLt rung template route missing".into(),
                                )
                            })
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
            d.op = DevOp::Nop as u16;
            Some(CublasLtDecodeRoute {
                plan,
                input,
                weight,
                output,
            })
        } else {
            None
        };
        routes.push(route);
    }
    tracing::info!(
        segments = routes.len(),
        projections = routes.iter().flatten().count(),
        plans = plans.len(),
        native = matches!(backend, ProjectionBackend::Native(_)),
        "projection routes prepared"
    );
    Ok(routes)
}

pub(super) struct CublasLtDecodeGraph {
    be: Arc<CudaBackend>,
    graph: Option<crate::device::cuda::GraphExec>,
    _routes: Vec<Option<CublasLtDecodeRoute>>,
}

impl CublasLtDecodeGraph {
    pub(super) fn capture(
        be: &Arc<CudaBackend>,
        stream: &CudaStream,
        base: DevProgram,
        function: KernelFn,
        grid: u32,
        smem: u32,
        routes: Vec<Option<CublasLtDecodeRoute>>,
    ) -> Result<Self> {
        let graph = be.graph_capture(stream, || {
            for (seg, route) in routes.iter().enumerate() {
                if let Some(route) = route {
                    route
                        .plan
                        .run(route.input, route.weight, route.output, stream)?;
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
                    route
                        .plan
                        .run(route.input, route.weight, route.output, &self.stream)?;
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
        self.cublaslt_decode_graph =
            Some(be.graph_capture(&self.stream, || self.enqueue_decode_chain())?);
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
                route
                    .plan
                    .run(route.input, route.weight, route.output, &self.stream)?;
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
            self.be.launch_cooperative(
                role.map_or(self.f, |r| r.function),
                role.map_or(self.grid, |r| r.grid),
                role.map_or(BLOCK, |r| r.block),
                role.map_or(self.smem, |r| r.smem),
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
        ProjectionPhase::Decode => !(1..=32).contains(&rows),
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
                packed_prefill_only: false,
                token_batch_body: false,
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

    fn prefill_fixture(rows: u32, k: u32) -> (DevProg, Vec<DevTensor>) {
        let (mut program, mut tensors) = fixture(rows);
        let op = &mut program.insts[1];
        op.op = DevOp::Gemm as u16;
        op.i[..].copy_from_slice(&[rows, 3840, k, 0, 0, 0, 17, 18]);
        tensors[0].bytes = u64::from(rows) * 3840 * 2;
        tensors[1].bytes = u64::from(rows) * u64::from(k) * 2;
        tensors[2].bytes = 3840 * u64::from(k) * 2;
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
        let waits = ordered_waits(&g, &routes).unwrap();
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
        assert!(ordered_waits(&g, &routes).is_err());
        g.succs[0] = 0;
        g.waits[0].id = 2;
        assert!(ordered_waits(&g, &routes).is_err());
        g.waits[0].id = 0;
        g.waits[1].id = 3;
        assert!(ordered_waits(&g, &routes).is_err());
        g.waits[1].id = 1;
        g.waits[1].threshold = 2;
        assert!(ordered_waits(&g, &routes).is_err());
        g.waits[1].threshold = 1;
        g.stream[1].flags = packet::dev::SE_FINE;
        g.gq_stream = g.stream.clone();
        assert!(ordered_waits(&g, &routes).is_err());
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
        for rows in plow_asset::segment_roles::CUBLASLT_PREFILL_ROWS {
            for k in [8192, 15360] {
                let (program, tensors) = prefill_fixture(rows, k);
                let routes = prefill_segments(&program, &tensors, &roles(), "sm90a").unwrap();
                let route = routes[1].expect("measured projection route");
                assert_eq!((route.m, route.n, route.k), (rows, 3840, k));
            }
        }

        for (profile, rows, n, k) in [
            ("sm120", 128, 3840, 15360),
            ("sm90a", 64, 3840, 15360),
            ("sm90a", 1024, 3840, 15360),
            ("sm90a", 128, 4096, 3840),
            ("sm90a", 128, 3840, 4096),
        ] {
            let (mut program, tensors) = prefill_fixture(rows, k);
            program.insts[1].i[1] = n;
            assert!(prefill_segments(&program, &tensors, &roles(), profile).is_err());
        }
    }

    #[test]
    fn prefill_projection_rejects_bias_fp8_and_nonisolated_packets() {
        let (mut program, tensors) = prefill_fixture(128, 15360);
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
