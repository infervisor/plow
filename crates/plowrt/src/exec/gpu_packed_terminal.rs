use super::*;
use packet::dev::{StreamEnt, Wait};

pub(super) struct PackedTerminal {
    template: [DevInst64; 5],
    capacity: usize,
    rows: DeviceMem,
    _selected: DeviceMem,
    _normalized: DeviceMem,
    _tensors: DeviceMem,
    program: plow_asset::aux_program::Program,
    arg: DevProgram,
    counter_bytes: usize,
    _tables: Vec<DeviceMem>,
    host_rows: Vec<u32>,
    host_ids: Vec<u32>,
}

fn layout(insts: &[DevInst64], rows: u32, logits: usize, ids: usize) -> Option<[DevInst64; 5]> {
    let tail: [DevInst64; 5] = insts.get(insts.len().checked_sub(5)?..)?.try_into().ok()?;
    let [norm, head, cap, max, fin] = tail;
    if rows == 0
        || [norm.op, head.op, cap.op, max.op, fin.op]
            != [
                DevOp::RmsNorm,
                DevOp::Gemm,
                DevOp::SoftCap,
                DevOp::Argmax,
                DevOp::ArgmaxFin,
            ]
            .map(|op| op as u16)
        || norm.i[0] != rows
        || norm.i[1] == 0
        || head.i[0] != 1
        || head.i[1] == 0
        || head.i[2] != norm.i[1]
        || head.i[4] != rows - 1
        || head.i[6] != 0
        || head.i[7] != 0
        || head.t[1] != norm.t[0]
        || head.t[0] as usize != logits
        || cap.t[0] != head.t[0]
        || cap.t[1] != head.t[0]
        || cap.i[0] != head.i[1]
        || max.t[1] != head.t[0]
        || max.i[0] != head.i[1]
        || max.i[1] > 1
        || fin.t[1] != max.t[0]
        || fin.t[0] as usize != ids
        || fin.i[0] != max.blocks as u32
        || fin.i[1] > 1
        || norm.t[1] == norm.t[0]
        || norm.t[1] == head.t[0]
        || norm.t[1] == max.t[0]
    {
        return None;
    }
    Some(tail)
}

fn chain(insts: Vec<DevInst64>, grid: u32, rows: u32) -> plow_asset::aux_program::Program {
    let mut per_cu = vec![Vec::new(); grid as usize];
    let mut queue = Vec::new();
    let mut waits = Vec::new();
    let mut succs = Vec::new();
    for (i, inst) in insts.iter().enumerate() {
        let wait_ofs = waits.len() as u32;
        if i > 0 {
            waits.push(Wait {
                id: i as u32 - 1,
                threshold: insts[i - 1].blocks as u32,
            });
        }
        succs.push(i as u32);
        for slice in 0..inst.blocks as u32 {
            let entry = StreamEnt {
                inst: i as u32,
                slice,
                wait_ofs,
                succ_ofs: i as u32,
                wait_len: u16::from(i > 0),
                succ_len: 1,
                flags: 0,
                seg: 0,
            };
            queue.push(entry);
            per_cu[slice as usize % grid as usize].push(entry);
        }
    }
    let mut stream = Vec::new();
    let mut stream_ofs = Vec::new();
    let mut stream_len = Vec::new();
    for entries in per_cu {
        stream_ofs.push(stream.len() as u32);
        stream_len.push(entries.len() as u32);
        stream.extend(entries);
    }
    plow_asset::aux_program::Program {
        rows,
        n_counter: insts.len() as u32,
        insts,
        stream,
        stream_ofs,
        stream_len,
        waits,
        succs,
        gq_seg_ofs: vec![0, queue.len() as u32],
        gq_stream: queue,
    }
}

impl PackedTerminal {
    pub(super) fn load(e: &GpuEngine) -> Result<Option<Self>> {
        if !e.vmm_prefix_enabled() || e.packed_prefill.is_none() || e.pf_batch.is_none() {
            return Ok(None);
        }
        let Some(first) = e
            .prefill
            .first()
            .and_then(|b| layout(&b.h_inst, b.t, e.t_logits, e.t_ids))
        else {
            return Ok(None);
        };
        for bucket in &e.prefill {
            let Some(mut tail) = layout(&bucket.h_inst, bucket.t, e.t_logits, e.t_ids) else {
                return Ok(None);
            };
            tail[0].i[0] = first[0].i[0];
            tail[0].blocks = first[0].blocks;
            tail[1].i[4] = first[1].i[4];
            if tail != first {
                return Ok(None);
            }
        }
        let module = e.module_pf.as_ref().ok_or_else(|| {
            RuntimeError::Rejected("compact terminal requires a prefill object".into())
        })?;
        if e.be.module_global_u32(module, "plow_row_gather_1")? != Some(1) {
            return Ok(None);
        }
        let [norm, head, _, max, _] = first;
        let source = &e.devp[norm.t[1] as usize];
        for id in [norm.t[0], head.t[0], max.t[0], first[4].t[0]] {
            let destination = &e.devp[id as usize];
            if source.base < destination.base + destination.len
                && destination.base < source.base + source.len
            {
                return Ok(None);
            }
        }
        let hidden = norm.i[1];
        let capacity = e.batch;
        let blocks = u16::try_from(capacity)
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| {
                RuntimeError::Rejected("compact terminal capacity out of range".into())
            })?;
        let elements = (capacity as u32).checked_mul(head.i[1]).ok_or_else(|| {
            RuntimeError::Rejected("compact terminal logits extent overflow".into())
        })?;
        let selected = e.be.alloc(0, capacity as u64 * hidden as u64 * 2)?;
        let normalized = e.be.alloc(0, selected.len)?;
        let rows = e.be.alloc(0, capacity as u64 * 4)?;
        let mut pointers: Vec<u64> = e.devp.iter().map(|mem| mem.base).collect();
        let handle = u16::try_from(pointers.len())
            .ok()
            .filter(|&h| h < TENSOR_NONE16 - 2)
            .ok_or_else(|| {
                RuntimeError::Rejected("compact terminal tensor handles exhausted".into())
            })?;
        pointers.extend([selected.base, normalized.base, rows.base]);
        for (id, bytes) in [
            (norm.t[1], e.pf_max_rows() as u64 * hidden as u64 * 2),
            (norm.t[2], hidden as u64 * 2),
            (head.t[2], head.i[1] as u64 * hidden as u64 * 2),
            (head.t[0], capacity as u64 * head.i[1] as u64 * 2),
            (max.t[0], capacity as u64 * max.blocks as u64 * 8),
            (e.t_ids as u16, capacity as u64 * 4),
        ] {
            if e.devp.get(id as usize).is_none_or(|m| m.len < bytes) {
                return Err(RuntimeError::Rejected(format!(
                    "compact terminal tensor {id} needs {bytes} bytes"
                )));
            }
        }
        let tensors = e.be.alloc(0, pointers.len() as u64 * 8)?;
        e.be.upload(&tensors, 0, pod_bytes(&pointers))?;
        let mut gather = DevInst64 {
            op: DevOp::RowGather as u16,
            blocks,
            t: [TENSOR_NONE16; 8],
            ..Default::default()
        };
        gather.t[..3].copy_from_slice(&[handle, norm.t[1], handle + 2]);
        gather.i[..3].copy_from_slice(&[capacity as u32, hidden, e.pf_max_rows() as u32]);
        let mut insts = vec![gather];
        insts.extend(first);
        insts[1].t[0] = handle + 1;
        insts[1].t[1] = handle;
        insts[1].i[0] = capacity as u32;
        insts[1].blocks = blocks;
        insts[2].t[1] = handle + 1;
        insts[2].i[0] = capacity as u32;
        insts[2].i[4] = 0;
        insts[3].i[0] = elements;
        insts[4].i[1] = capacity as u32;
        insts[5].i[1] = capacity as u32;
        let program = chain(insts, e.grid, capacity as u32);
        plow_asset::aux_program::Section {
            n_cu: e.grid,
            programs: vec![program.clone()],
        }
        .validate(pointers.len())
        .map_err(RuntimeError::Rejected)?;
        let (arg, _, counter_bytes, tables) =
            super::gpu_mixed_step::upload_program(&e.be, &program, tensors.base, 0, 0)?;
        Ok(Some(Self {
            template: first,
            capacity,
            rows,
            _selected: selected,
            _normalized: normalized,
            _tensors: tensors,
            program,
            arg,
            counter_bytes,
            _tables: tables,
            host_rows: Vec::with_capacity(capacity),
            host_ids: vec![0; capacity],
        }))
    }

    pub(super) fn run(&mut self, e: &GpuEngine, live: usize) -> Result<()> {
        if self.host_rows.is_empty() {
            return Ok(());
        }
        if self.host_rows.len() > self.capacity
            || self.host_rows.iter().any(|&r| r as usize >= live)
        {
            return Err(RuntimeError::Rejected(
                "compact terminal sample rows out of bounds".into(),
            ));
        }
        let count = self.host_rows.len() as u32;
        self.program.insts[0].i[0] = count;
        self.program.insts[0].i[2] = live as u32;
        self.program.insts[1].i[0] = count;
        self.program.insts[2].i[0] = count;
        self.program.insts[3].i[0] = count * self.template[1].i[1];
        self.program.insts[4].i[1] = count;
        self.program.insts[5].i[1] = count;
        let launched = (|| {
            // Both upload sources are owned here until the stream is drained.
            unsafe {
                e.be.memcpy_htod_async(self.rows.base, pod_bytes(&self.host_rows), &e.stream)?;
                e.be.memcpy_htod_async(self.arg.insts, pod_bytes(&self.program.insts), &e.stream)?;
            }
            e.be.memset_d8_async(self.arg.counters, 0, self.counter_bytes, &e.stream)?;
            let mut arg = self.arg;
            let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
            e.be.launch_cooperative(
                e.f_pf.unwrap(),
                e.grid,
                BLOCK,
                e.smem_pf,
                &mut params,
                Some(&e.stream),
            )?;
            e.be.stream_synchronize(&e.stream)
        })();
        if let Err(error) = launched {
            let _ = e.be.stream_synchronize(&e.stream);
            return Err(error);
        }
        e.be.download(
            &e.devp[e.t_ids],
            0,
            bytemuck::cast_slice_mut(&mut self.host_ids[..self.host_rows.len()]),
        )?;
        if self.host_ids[..self.host_rows.len()]
            .iter()
            .any(|&id| id as usize >= e.vocab)
        {
            return Err(RuntimeError::Device(
                "compact terminal produced an invalid token".into(),
            ));
        }
        Ok(())
    }
}

impl GpuEngine {
    pub fn has_packed_terminal(&self) -> bool {
        self.packed_terminal.is_some()
    }

    pub fn prefill_batched_complete(
        &mut self,
        reqs: &[PfBatchReq<'_>],
        out: &mut Vec<(usize, u32)>,
    ) -> Result<()> {
        if self.packed_terminal.is_none() {
            return Err(RuntimeError::Rejected(
                "compact packed terminal unavailable".into(),
            ));
        }
        out.clear();
        self.prefill_batched(reqs)?;
        let mut terminal = self.packed_terminal.take().unwrap();
        terminal.host_rows.clear();
        let mut live = 0;
        for req in reqs {
            live += req.len;
            if req.c0 + req.len == req.prompt.len() {
                terminal.host_rows.push((live - 1) as u32);
            }
        }
        let result = terminal.run(self, live);
        if result.is_ok() {
            out.extend(
                reqs.iter()
                    .filter(|r| r.c0 + r.len == r.prompt.len())
                    .map(|r| r.slot)
                    .zip(terminal.host_ids.iter().copied()),
            );
        }
        self.packed_terminal = Some(terminal);
        result?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tail() -> Vec<DevInst64> {
        let mut ops: Vec<_> = [
            DevOp::RmsNorm,
            DevOp::Gemm,
            DevOp::SoftCap,
            DevOp::Argmax,
            DevOp::ArgmaxFin,
        ]
        .into_iter()
        .map(|op| DevInst64 {
            op: op as u16,
            blocks: 1,
            t: [TENSOR_NONE16; 8],
            ..Default::default()
        })
        .collect();
        ops[0].t[..3].copy_from_slice(&[1, 2, 3]);
        ops[0].i[..2].copy_from_slice(&[128, 32]);
        ops[0].blocks = 128;
        ops[1].t[..3].copy_from_slice(&[4, 1, 5]);
        ops[1].i[..3].copy_from_slice(&[1, 64, 32]);
        ops[1].i[4] = 127;
        ops[1].blocks = 8;
        ops[2].t[..2].copy_from_slice(&[4, 4]);
        ops[2].i[0] = 64;
        ops[3].t[..2].copy_from_slice(&[6, 4]);
        ops[3].i[0] = 64;
        ops[3].blocks = 4;
        ops[4].t[..2].copy_from_slice(&[0, 6]);
        ops[4].i[0] = 4;
        ops
    }

    #[test]
    fn compact_tail_refuses_changed_math_and_broken_dependencies() {
        assert!(layout(&tail(), 128, 4, 0).is_some());
        for mutate in [
            |ops: &mut Vec<DevInst64>| ops[1].op = DevOp::Gemv as u16,
            |ops: &mut Vec<DevInst64>| ops[1].i[6] = 1,
            |ops: &mut Vec<DevInst64>| ops[1].i[4] = 126,
            |ops: &mut Vec<DevInst64>| ops[1].t[1] = 2,
            |ops: &mut Vec<DevInst64>| ops[4].i[0] = 3,
            |ops: &mut Vec<DevInst64>| ops[0].t[1] = 4,
        ] {
            let mut ops = tail();
            mutate(&mut ops);
            assert!(layout(&ops, 128, 4, 0).is_none());
        }
        assert!(layout(&[], 0, 4, 0).is_none());
    }

    #[test]
    fn terminal_chain_covers_each_slice_and_waits_for_all_producer_slices() {
        let program = chain(tail(), 8, 1);
        plow_asset::aux_program::Section {
            n_cu: 8,
            programs: vec![program.clone()],
        }
        .validate(7)
        .unwrap();
        for (i, inst) in program.insts.iter().enumerate() {
            let entries: Vec<_> = program
                .gq_stream
                .iter()
                .filter(|e| e.inst == i as u32)
                .collect();
            assert_eq!(entries.len(), inst.blocks as usize);
            for (slice, entry) in entries.into_iter().enumerate() {
                assert_eq!(entry.slice, slice as u32);
                if i > 0 {
                    let wait = &program.waits[entry.wait_ofs as usize];
                    assert_eq!(wait.id, i as u32 - 1);
                    assert_eq!(wait.threshold, program.insts[i - 1].blocks as u32);
                }
            }
        }
    }

    #[test]
    #[ignore = "requires H100 packed-prefix assets and ordinary prefix_logits reference outputs"]
    fn packed_terminal_sparse_slots_match_full_ordinary_logits() {
        let assets = PathBuf::from(std::env::var("PACKED_TERMINAL_TEST_ASSETS").unwrap());
        let reference = PathBuf::from(std::env::var("PACKED_TERMINAL_TEST_REFERENCE").unwrap());
        let records: Vec<serde_json::Value> = (0..2)
            .map(|i| {
                serde_json::from_slice(&std::fs::read(reference.join(format!("{i}.json"))).unwrap())
                    .unwrap()
            })
            .collect();
        let prompts: Vec<Vec<u32>> = records
            .iter()
            .map(|r| serde_json::from_value(r["prompt_ids"].clone()).unwrap())
            .collect();
        let feeds: Vec<Vec<u32>> = records
            .iter()
            .map(|r| serde_json::from_value(r["selected_tokens"].clone()).unwrap())
            .collect();
        let expected: Vec<Vec<f32>> = (0..2)
            .map(|i| {
                std::fs::read(reference.join(format!("{i}.f32")))
                    .unwrap()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect()
            })
            .collect();
        let mut e = GpuEngine::load(
            Arc::new(CudaBackend::new(0).unwrap()),
            &assets,
            &assets.join("checkpoint"),
        )
        .unwrap();
        assert!(e.has_packed_terminal() && e.batch >= 8);
        let slots = [7, 3];
        let ends = [prompts[0].len() - 3, prompts[1].len() - 5];
        let mut pos = [0, 0];
        for i in 0..2 {
            e.begin_slot(slots[i], prompts[i].len() + 64).unwrap();
        }
        let mut output = Vec::new();
        while pos != ends {
            let active = (0..2).filter(|&i| pos[i] < ends[i]).count();
            let quota = e.pf_max_rows() / active;
            let requests: Vec<_> = (0..2)
                .filter(|&i| pos[i] < ends[i])
                .map(|i| PfBatchReq {
                    slot: slots[i],
                    prompt: &prompts[i],
                    c0: pos[i],
                    len: (ends[i] - pos[i]).min(quota),
                })
                .collect();
            e.prefill_batched_complete(&requests, &mut output).unwrap();
            assert!(output.is_empty());
            for r in requests {
                pos[slots.iter().position(|&s| s == r.slot).unwrap()] += r.len;
            }
        }
        let requests: Vec<_> = [1, 0]
            .into_iter()
            .map(|i| PfBatchReq {
                slot: slots[i],
                prompt: &prompts[i],
                c0: pos[i],
                len: prompts[i].len() - pos[i],
            })
            .collect();
        e.prefill_batched_complete(&requests, &mut output).unwrap();
        assert_eq!(output, vec![(3, feeds[1][0]), (7, feeds[0][0])]);
        let mut logits = Vec::new();
        for (row, i) in [1, 0].into_iter().enumerate() {
            e.logits_row(row, &mut logits).unwrap();
            assert_eq!(
                logits
                    .iter()
                    .zip(&expected[i][..e.vocab])
                    .position(|(a, b)| a.to_bits() != b.to_bits()),
                None,
                "terminal case {i}"
            );
        }
        let mut ids = Vec::new();
        for step in 1..64 {
            e.step_slots(
                &[
                    (slots[0], feeds[0][step - 1]),
                    (slots[1], feeds[1][step - 1]),
                ],
                &mut ids,
            )
            .unwrap();
            for i in 0..2 {
                e.logits_row(slots[i], &mut logits).unwrap();
                assert_eq!(
                    logits
                        .iter()
                        .zip(&expected[i][step * e.vocab..(step + 1) * e.vocab])
                        .position(|(a, b)| a.to_bits() != b.to_bits()),
                    None,
                    "case {i} step {step}"
                );
            }
        }
        eprintln!("PASS compact terminal: 128 full-logit snapshots, sparse slots, reversed request order, sample rows 4/7");
    }
}
