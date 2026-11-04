//! Bring-up qualification through the packet's real interpreters, before model loading.

use std::time::{Duration, Instant};

use packet::dev::{StreamEnt, Wait};

use super::*;
use crate::device::cuda::ipc::IpcRegion;
use crate::exec::proc_group::ProcGroup;
use crate::exec::tp::PeerLayout;

const PROBE_BLOCKS: u32 = 4;
const PROBE_TIMEOUT: Duration = Duration::from_secs(30);

struct Object {
    module: Module,
    function: KernelFn,
    smem: u32,
}

struct Storage {
    objects: Vec<Object>,
    buffers: Vec<DeviceMem>,
    stream: CudaStream,
}

pub(crate) struct Probe {
    be: Arc<CudaBackend>,
    group: Arc<ProcGroup>,
    storage: Option<Storage>,
    args: DevProgram,
    output: u64,
    output_stride: u64,
    own: u64,
    peers: Vec<u64>,
    layout: PeerLayout,
    unsafe_to_free: bool,
}

impl Probe {
    pub fn prepare(
        be: Arc<CudaBackend>,
        group: Arc<ProcGroup>,
        assets: &Path,
        regions: &[IpcRegion],
    ) -> Result<Self> {
        be.unload_modules_on_err(|| Self::prepare_inner(be.clone(), group, assets, regions))
    }

    fn prepare_inner(
        be: Arc<CudaBackend>,
        group: Arc<ProcGroup>,
        assets: &Path,
        regions: &[IpcRegion],
    ) -> Result<Self> {
        let layout = group
            .layout()
            .ok_or_else(|| RuntimeError::Rejected("P2P probe needs a layout".into()))?;
        if group.tp() < 2
            || group.tp() > 31
            || layout.n_xctr < 6
            || layout.hidden < 64
            || layout.partial_off(1)? > u32::MAX as u64
        {
            return Err(RuntimeError::Rejected(
                "P2P probe needs 2..31 ranks, six gates and u32 slot offsets".into(),
            ));
        }
        let pkt = DevBlob::find_in_dir(assets)?
            .ok_or_else(|| RuntimeError::Rejected("P2P probe: no packet".into()))?;
        let raw = std::fs::read(&pkt).map_err(|source| RuntimeError::Io { path: pkt, source })?;
        let blob = DevBlob::parse(&raw)?;
        let cc = be.compute_capability();
        let profile = interpreter_profile(cc)
            .ok_or_else(|| RuntimeError::Rejected("P2P probe: unsupported interpreter".into()))?;
        let partial_f32 = rank::validate_partial_f32(&blob)?;
        if partial_f32 && cc != (12, 0) {
            return Err(RuntimeError::Rejected(
                "FP32 peer partials require sm_120a".into(),
            ));
        }
        let resolve = |role| {
            resolve_interp_image(assets, &blob, &raw, &profile, cc.0 * 10 + cc.1, role)?.ok_or_else(
                || RuntimeError::Rejected("P2P probe needs decode and prefill interpreters".into()),
            )
        };
        let decode = resolve(Role::Decode)?;
        let prefill = resolve(Role::Prefill)?;
        let roles = if partial_f32 {
            segment_role_metadata(&blob, &raw)?
        } else {
            None
        };
        let mut objects = Vec::new();
        for (pf, image) in [(false, &decode), (true, &prefill)] {
            let module = be.module_load(&image.image)?;
            if partial_f32 && !pf {
                if let Some(reason) = xr_capability_unavailable(
                    &be,
                    &module,
                    Some(&prefill),
                    assets,
                    &blob,
                    &raw,
                    cc.0 * 10 + cc.1,
                    roles.as_ref(),
                    rank::PARTIAL_F32_MARKER,
                )? {
                    return Err(RuntimeError::Rejected(format!(
                        "FP32 peer partials: {reason}"
                    )));
                }
            }
            let cfg = &RuntimeConfig::get().nv;
            let name = if pf {
                cfg.kernel_pf.as_deref()
            } else {
                cfg.kernel.as_deref()
            }
            .unwrap_or(&image.entry);
            let function = be.get_function(&module, name)?;
            let smem = if pf { cfg.smem_pf } else { cfg.smem }.unwrap_or(
                be.module_global_u32(
                    &module,
                    if pf {
                        "plow_arena_bytes_pf"
                    } else {
                        "plow_arena_bytes"
                    },
                )?
                .unwrap_or(if pf { SMEM_PF } else { 12352 }),
            );
            be.set_max_dynamic_smem(function, smem)?;
            if be.occupancy_blocks_per_sm(function, BLOCK, smem as usize)? * be.sm_count()
                < PROBE_BLOCKS
            {
                return Err(RuntimeError::Rejected(
                    "P2P probe: insufficient cooperative occupancy".into(),
                ));
            }
            objects.push(Object {
                module,
                function,
                smem,
            });
        }
        let mut buffers = Vec::new();
        let mut upload = |bytes: &[u8]| -> Result<u64> {
            let mem = be.alloc(0, bytes.len().max(4) as u64)?;
            be.upload(&mem, 0, bytes)?;
            let base = mem.base;
            buffers.push(mem);
            Ok(base)
        };
        let peers: Vec<_> = regions.iter().map(|r| r.base).collect();
        let mut table = peers.clone();
        table.push(0); // P2P is always unstaged, independent of object markers.
        let peer_scratch = upload(pod_bytes(&table))?;
        let output_stride = u64::from(probe_elements(layout, true)) * 2;
        let output = upload(&vec![0u8; output_stride as usize * 4])?;
        let tensors = upload(pod_bytes(
            &(0..4)
                .map(|i| output + i * output_stride)
                .collect::<Vec<_>>(),
        ))?;
        let insts = upload(pod_bytes(&[DevInst64::default(); 4]))?;
        let entry = |inst: u32, slice: u32| StreamEnt {
            inst,
            slice,
            wait_ofs: inst.saturating_sub(1),
            wait_len: u16::from(inst > 0),
            succ_ofs: inst,
            succ_len: 1,
            ..StreamEnt::default()
        };
        let entries: Vec<_> = (0..PROBE_BLOCKS)
            .flat_map(|slice| (0..4).map(move |inst| entry(inst, slice)))
            .collect();
        let queue: Vec<_> = (0..4)
            .flat_map(|inst| (0..PROBE_BLOCKS).map(move |slice| entry(inst, slice)))
            .collect();
        let stream = upload(pod_bytes(&entries))?;
        let gq_stream = upload(pod_bytes(&queue))?;
        let stream_ofs = upload(pod_bytes(
            &(0..PROBE_BLOCKS).map(|b| b * 4).collect::<Vec<_>>(),
        ))?;
        let stream_len = upload(pod_bytes(&[4u32; PROBE_BLOCKS as usize]))?;
        let gq_seg_ofs = upload(pod_bytes(&[0u32, PROBE_BLOCKS * 4]))?;
        let waits = upload(pod_bytes(
            &(0..3)
                .map(|id| Wait {
                    id,
                    threshold: PROBE_BLOCKS,
                })
                .collect::<Vec<_>>(),
        ))?;
        let succs = upload(pod_bytes(&[0u32, 1, 2, 3]))?;
        let counters = upload(&[0u8; 4 * 128])?;
        let gq_cursor = upload(&[0u8; 128])?;
        let own = peers[group.rank() as usize % group.tp() as usize];
        let args = DevProgram {
            insts,
            stream,
            stream_ofs,
            stream_len,
            waits,
            succs,
            counters,
            tensors,
            trace: 0,
            cur_seg: 0,
            l2_domains: 0,
            n_seg: 1,
            hier_base: 0,
            gq_stream,
            gq_seg_ofs,
            gq_cursor,
            xctr: own + layout.xctr_off(),
            peer_scratch,
            rank: group.rank() % group.tp(),
            n_gpu: group.tp(),
            seg_ofs: 0,
            prefill_spans: 0,
            prefill_parked: 0,
            n_prefill_spans: 0,
            n_prefill_rows: 0,
            token_batch: 0,
        };
        let stream = be.stream_create()?;
        Ok(Self {
            be,
            group,
            storage: Some(Storage {
                objects,
                buffers,
                stream,
            }),
            args,
            output,
            output_stride,
            own,
            peers,
            layout,
            unsafe_to_free: false,
        })
    }

    /// Recoverable data/flag failures are returned only after all launch units drain.
    /// Driver errors abort the barriers and make the whole bring-up fatal.
    pub fn run(&mut self) -> Result<(Option<String>, f64)> {
        let result = self.run_inner();
        if result.is_err() {
            self.group.abort();
        }
        result
    }

    pub fn addressability(&mut self) -> Result<(Option<String>, f64)> {
        let result = self.readback_inner();
        if result.is_err() {
            self.group.abort();
        }
        result
    }

    fn readback_inner(&mut self) -> Result<(Option<String>, f64)> {
        let mut failure = None;
        let mut rtt = 0.0f64;
        let storage = self.storage.as_ref().expect("probe storage");
        // Copy-engine addressability is a preliminary check, never the P2P qualification.
        recover(
            &self.be,
            &mut failure,
            self.be.memset_d8(self.own, 0, self.layout.bytes() as usize),
        )?;
        let pattern = (0..64)
            .map(|i| (i + self.args.rank * 67) as u8)
            .collect::<Vec<_>>();
        recover(
            &self.be,
            &mut failure,
            self.be.memcpy_htod(self.own, &pattern),
        )?;
        self.group.barrier(PROBE_TIMEOUT)?;
        for (r, &peer) in self.peers.iter().enumerate() {
            let mut back = [0u8; 64];
            recover(&self.be, &mut failure, self.be.memcpy_dtoh(&mut back, peer))?;
            if back
                .iter()
                .enumerate()
                .any(|(i, &v)| v != (i as u32 + r as u32 * 67) as u8)
            {
                failure.get_or_insert_with(|| format!("peer {r} readback mismatch"));
            }
            recover(
                &self.be,
                &mut failure,
                self.be.memcpy_htod(
                    peer + 128 + u64::from(self.args.rank) * 4,
                    &(self.args.rank + 1).to_le_bytes(),
                ),
            )?;
        }
        self.group.barrier(PROBE_TIMEOUT)?;
        for r in 0..self.args.n_gpu {
            let mut back = [0u8; 4];
            recover(
                &self.be,
                &mut failure,
                self.be
                    .memcpy_dtoh(&mut back, self.own + 128 + u64::from(r) * 4),
            )?;
            if u32::from_le_bytes(back) != r + 1 {
                failure.get_or_insert_with(|| format!("peer {r} store mismatch"));
            }
        }
        // Every directed pair gets a flag request/ack. Polling is bounded without faulting
        // or poisoning a context when a platform silently drops peer stores.
        for src in 0..self.args.n_gpu {
            for dst in 0..self.args.n_gpu {
                if src == dst {
                    continue;
                }
                let flag = self.layout.xstatus_off();
                recover(
                    &self.be,
                    &mut failure,
                    self.be.memset_d8(self.own + flag, 0, 4),
                )?;
                self.group.barrier(PROBE_TIMEOUT)?;
                let start = Instant::now();
                if self.args.rank == src {
                    recover(
                        &self.be,
                        &mut failure,
                        self.be.stream_write_value32(
                            &storage.stream,
                            self.peers[dst as usize] + flag,
                            1,
                        ),
                    )?;
                    if !recover(&self.be, &mut failure, poll(&self.be, self.own + flag, 2))?
                        .unwrap_or(false)
                    {
                        failure.get_or_insert_with(|| format!("flag timeout {src}->{dst}->{src}"));
                    }
                    rtt = rtt.max(start.elapsed().as_secs_f64() * 1e6);
                } else if self.args.rank == dst {
                    if recover(&self.be, &mut failure, poll(&self.be, self.own + flag, 1))?
                        .unwrap_or(false)
                    {
                        recover(
                            &self.be,
                            &mut failure,
                            self.be.stream_write_value32(
                                &storage.stream,
                                self.peers[src as usize] + flag,
                                2,
                            ),
                        )?;
                    } else {
                        failure.get_or_insert_with(|| format!("flag timeout {src}->{dst}"));
                    }
                }
                if let Err(e) = self.be.stream_synchronize(&storage.stream) {
                    self.unsafe_to_free = true;
                    return Err(e);
                }
                self.group.barrier(PROBE_TIMEOUT)?;
            }
        }
        Ok((failure, rtt))
    }

    fn run_inner(&mut self) -> Result<(Option<String>, f64)> {
        let mut failure = None;
        let storage = self.storage.as_ref().expect("probe storage");
        for iteration in 0..4 {
            let prefill = iteration % 2 == 1;
            let n = probe_elements(self.layout, prefill);
            let input = probe_input(self.args.rank, iteration, n);
            self.be.memcpy_htod(self.own, &input)?;
            self.be.memcpy_htod(
                self.own + self.layout.partial_off(1)?,
                &probe_input(self.args.rank, iteration + 7, n),
            )?;
            let insts = probe_program(self.layout, self.args.n_gpu, prefill);
            self.be.memcpy_htod(self.args.insts, pod_bytes(&insts))?;
            self.be.memset_d8_async(
                self.output,
                0xff,
                self.output_stride as usize * 4,
                &storage.stream,
            )?;
            self.be
                .memset_d8_async(self.args.counters, 0, 4 * 128, &storage.stream)?;
            self.be
                .memset_d8_async(self.args.gq_cursor, 0, 128, &storage.stream)?;
            self.be.memset_d8_async(
                self.args.xctr,
                0,
                self.layout.xstate_bytes() as usize,
                &storage.stream,
            )?;
            if let Err(e) = self.be.stream_synchronize(&storage.stream) {
                self.unsafe_to_free = true;
                return Err(e);
            }
            self.group.barrier(PROBE_TIMEOUT)?;
            let object = &storage.objects[usize::from(prefill)];
            let mut arg = self.args;
            let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
            self.be.launch_cooperative(
                object.function,
                PROBE_BLOCKS,
                BLOCK,
                object.smem,
                &mut params,
                Some(&storage.stream),
            )?;
            if let Err(e) = self.be.stream_synchronize(&storage.stream) {
                self.unsafe_to_free = true;
                return Err(e);
            }
            let mut status = [0u8; 4];
            self.be
                .memcpy_dtoh(&mut status, self.own + self.layout.xstatus_off())?;
            if status != [0; 4] {
                failure.get_or_insert_with(|| {
                    format!(
                        "interpreter status {:#x} in probe {iteration}",
                        u32::from_le_bytes(status)
                    )
                });
            }
            let mut output = vec![0u8; n as usize * 2];
            for op in 0..4 {
                self.be
                    .memcpy_dtoh(&mut output, self.output + op as u64 * self.output_stride)?;
                if output != probe_expected(self.args.n_gpu, iteration, n, op) {
                    failure.get_or_insert_with(|| format!("interpreter output mismatch in probe {iteration}, op {op} (prefill={prefill})"));
                }
            }
            // All remote readers must retire before slot 0 is overwritten on the next unit.
            self.group.barrier(PROBE_TIMEOUT)?;
        }
        Ok((failure, 0.0))
    }
}

fn recover<T>(
    be: &CudaBackend,
    failure: &mut Option<String>,
    result: Result<T>,
) -> Result<Option<T>> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(e) if be.ipc_must_retain() => Err(e),
        Err(e) => {
            failure.get_or_insert_with(|| e.to_string());
            Ok(None)
        }
    }
}

fn poll(be: &CudaBackend, address: u64, want: u32) -> Result<bool> {
    let start = Instant::now();
    let mut back = [0u8; 4];
    while start.elapsed() < Duration::from_millis(250) {
        be.memcpy_dtoh(&mut back, address)?;
        if u32::from_le_bytes(back) == want {
            return Ok(true);
        }
        std::thread::yield_now();
    }
    Ok(false)
}

fn probe_elements(layout: PeerLayout, two_shot: bool) -> u32 {
    layout.hidden
        * if two_shot {
            layout.max_tokens.min(16)
        } else {
            1
        }
}

fn probe_program(layout: PeerLayout, tp: u32, prefill: bool) -> [DevInst64; 4] {
    // A, B, A, B: A's second rendezvous retires every rank's readers of B before
    // B's reduced slices are overwritten. Unique gates exercise reuse within one launch.
    let gates = [0, 1, 3, 4];
    std::array::from_fn(|op| {
        let two_shot = op % 2 == 1;
        let mut inst = DevInst64 {
            op: if two_shot {
                DevOp::XReduceTwoShot
            } else {
                DevOp::XReduce
            } as u16,
            blocks: PROBE_BLOCKS as u16,
            t: [TENSOR_NONE16; 8],
            ..DevInst64::default()
        };
        inst.t[0] = op as u16;
        inst.i[0] = probe_elements(layout, prefill);
        inst.i[1] = tp;
        inst.i[2] = if two_shot {
            layout.partial_off(1).expect("slot B") as u32
        } else {
            0
        };
        inst.i[3] = gates[op];
        inst.i[4] = gates[op] + 1;
        inst.fj[2] = layout.n_xctr + 1;
        inst
    })
}

fn probe_value(rank: u32, iteration: u32, element: u32) -> f32 {
    // Small integers are exact in bf16, including the FP32 sum on up to 31 ranks.
    ((element.wrapping_mul(17) + rank * 7 + iteration * 11) % 9) as f32 - 4.0
}

fn probe_input(rank: u32, iteration: u32, n: u32) -> Vec<u8> {
    (0..n)
        .flat_map(|i| ((probe_value(rank, iteration, i).to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn probe_expected(tp: u32, iteration: u32, n: u32, op: u32) -> Vec<u8> {
    let iteration = iteration + if op % 2 == 1 { 7 } else { 0 };
    (0..n)
        .flat_map(|i| {
            let mut sum: f32 = (0..tp).map(|r| probe_value(r, iteration, i)).sum();
            if op == 3 {
                // The first two-shot replaced exactly one rank's partial at each element.
                let owner = ((u64::from(i + 1) * u64::from(tp) - 1) / u64::from(n)) as u32;
                sum = 2.0 * sum - probe_value(owner, iteration, i);
            }
            ((sum.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

impl Drop for Probe {
    fn drop(&mut self) {
        let Some(storage) = self.storage.take() else {
            return;
        };
        if self.unsafe_to_free
            || self.be.ipc_must_retain()
            || self.be.stream_synchronize(&storage.stream).is_err()
        {
            std::mem::forget(storage);
            return;
        }
        for object in &storage.objects {
            let _ = self.be.module_unload(&object.module);
        }
        drop(storage.buffers);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_two_shot_oracle_matches_in_place_shards() {
        for tp in [2, 3, 8, 31] {
            for n in [64, 65, 2560] {
                let mut partials: Vec<Vec<f32>> = (0..tp)
                    .map(|rank| (0..n).map(|i| probe_value(rank, 7, i)).collect())
                    .collect();
                for op in [1, 3] {
                    let reduced: Vec<f32> = (0..n as usize)
                        .map(|i| partials.iter().map(|p| p[i]).sum())
                        .collect();
                    for (rank, partial) in partials.iter_mut().enumerate() {
                        let lo = n as usize * rank / tp as usize;
                        let hi = n as usize * (rank + 1) / tp as usize;
                        partial[lo..hi].copy_from_slice(&reduced[lo..hi]);
                    }
                    let bytes: Vec<u8> = reduced
                        .iter()
                        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                        .collect();
                    assert_eq!(probe_expected(tp, 0, n, op), bytes);
                }
            }
        }
    }

    #[test]
    fn probe_covers_both_collectives_and_reuses_slot_with_new_inputs() {
        let layout = PeerLayout::new(2560, 4096, 8).unwrap();
        for two_shot in [false, true] {
            let program = probe_program(layout, 2, two_shot);
            assert_eq!(program[0].op, DevOp::XReduce as u16);
            assert_eq!(program[1].op, DevOp::XReduceTwoShot as u16);
            assert_eq!(program[1].i[2], program[3].i[2]);
            assert_ne!(program[1].i[3], program[3].i[3]);
            for inst in program {
                assert_eq!(inst.blocks, 4);
                assert_eq!(inst.fj[2], 9);
                assert!(inst.i[0] >= 2560);
            }
            assert_ne!(probe_input(0, 0, 64), probe_input(1, 0, 64));
            assert_ne!(probe_input(0, 0, 64), probe_input(0, 2, 64));
            assert_ne!(probe_expected(2, 0, 64, 0), probe_input(0, 0, 64));
        }
    }
}
