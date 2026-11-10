use super::fish_depth::FishRow;
use super::packet_exec::CudaPacketRuntime;
use crate::device::cuda::{CudaBackend, CudaEvent, CudaStream, KernelFn, PinnedHost};
use crate::device::{Backend, DeviceMem};
use crate::exec::packet_runtime::{check_region, PacketTensor, PacketWrite};
use crate::{Result, RuntimeError};
use std::sync::Arc;

fn reject(message: &str) -> RuntimeError {
    RuntimeError::Rejected(message.into())
}

pub struct CodecTickState {
    be: Arc<CudaBackend>,
    slots: DeviceMem,
    packet: DeviceMem,
    slot_stage: Arc<PinnedHost>,
    packet_stage: Arc<PinnedHost>,
    codes_kernel: KernelFn,
    pack_kernel: KernelFn,
    lengths_kernel: KernelFn,
    capacity: usize,
    codebooks: usize,
    samples: usize,
    stride: usize,
    failed: bool,
    depth_events: (CudaEvent, CudaEvent),
}

pub struct CodecTick<'a> {
    pub state: &'a mut CodecTickState,
    pub runtime: &'a mut CudaPacketRuntime,
    pub programs: &'a [usize],
    pub writes: &'a [PacketWrite<'a>],
    pub code_input: PacketTensor,
    pub pcm_output: PacketTensor,
    pub lengths: &'a [(PacketTensor, u32)],
    pub pcm: &'a mut Vec<i16>,
}

impl CodecTickState {
    pub fn new(
        runtime: &CudaPacketRuntime,
        capacity: usize,
        codebooks: usize,
        frame_samples: usize,
    ) -> Result<Self> {
        if capacity == 0 || codebooks == 0 || frame_samples == 0 {
            return Err(reject("empty codec tick geometry"));
        }
        let stride = codebooks
            .checked_mul(4)
            .and_then(|n| frame_samples.checked_mul(2)?.checked_add(n))
            .and_then(|n| n.checked_add(3))
            .map(|n| n & !3)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| reject("codec tick row is too large"))?;
        let bytes = capacity
            .checked_mul(stride)
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| reject("codec tick capacity is too large"))?;
        let be = runtime.backend().clone();
        let (codes_kernel, pack_kernel, lengths_kernel) = runtime.tick_kernels()?;
        Ok(Self {
            slots: be.alloc(0, (capacity * 4) as u64)?,
            packet: be.alloc(0, bytes as u64)?,
            slot_stage: Arc::new(be.host_alloc_pinned(capacity * 4)?),
            packet_stage: Arc::new(be.host_alloc_pinned(bytes)?),
            depth_events: (be.event_create(true)?, be.event_create(true)?),
            be,
            codes_kernel,
            pack_kernel,
            lengths_kernel,
            capacity,
            codebooks,
            samples: frame_samples,
            stride,
            failed: false,
        })
    }
}

impl CodecTick<'_> {
    pub(super) fn begin_depth(&self, stream: &CudaStream) -> Result<()> {
        self.state
            .be
            .event_record(&self.state.depth_events.0, stream)
    }
    pub(super) fn end_depth(&self, stream: &CudaStream) -> Result<()> {
        self.state
            .be
            .event_record(&self.state.depth_events.1, stream)
    }
    pub(super) fn depth_us(&self) -> Result<f32> {
        self.state
            .be
            .event_elapsed_ms(&self.state.depth_events.0, &self.state.depth_events.1)
            .map(|ms| ms * 1000.)
    }

    pub(super) fn validate(
        &self,
        be: &Arc<CudaBackend>,
        rows: &[FishRow],
        codebooks: usize,
        source_bytes: u64,
        stage_bytes: usize,
    ) -> Result<()> {
        let s = &*self.state;
        if s.failed || !be.shares_context(&s.be) || !be.shares_context(self.runtime.backend()) {
            return Err(reject("codec tick requires one live CUDA backend"));
        }
        if rows.is_empty() || rows.len() > s.capacity || codebooks != s.codebooks {
            return Err(reject("codec tick row geometry mismatch"));
        }
        for (i, row) in rows.iter().enumerate() {
            let end = row
                .slot
                .checked_add(1)
                .and_then(|n| n.checked_mul(codebooks * 4))
                .ok_or_else(|| reject("codec tick slot overflow"))?;
            if end > stage_bytes
                || end as u64 > source_bytes
                || rows[..i].iter().any(|r| r.slot == row.slot)
            {
                return Err(reject("invalid codec tick slot map"));
            }
        }
        check_region(self.code_input, 0, s.capacity * codebooks * 4)?;
        check_region(self.pcm_output, 0, s.capacity * s.samples * 4)?;
        if self.code_input.handle == self.pcm_output.handle {
            return Err(reject("codec tick codes and PCM alias"));
        }
        if self.lengths.is_empty() {
            return Err(reject("codec tick needs packet length controls"));
        }
        for &(tensor, scale) in self.lengths {
            check_region(tensor, 0, s.capacity * 4)?;
            self.runtime.checked_device_ptr(tensor)?;
            if scale == 0
                || tensor.handle == self.code_input.handle
                || tensor.handle == self.pcm_output.handle
            {
                return Err(reject("invalid codec tick length control"));
            }
        }
        self.runtime.checked_device_ptr(self.code_input)?;
        self.runtime.checked_device_ptr(self.pcm_output)?;
        self.runtime.check_sequence(self.programs)?;
        self.runtime.check_chained_writes(self.writes)?;
        if self
            .writes
            .iter()
            .any(|w| w.tensor.handle == self.code_input.handle)
        {
            return Err(reject("codec tick controls overwrite generated codes"));
        }
        Ok(())
    }

    pub(super) fn complete(
        &mut self,
        be: &Arc<CudaBackend>,
        stream: &CudaStream,
        source: u64,
        rows: &[FishRow],
        stop: u32,
        stage: &mut [u8],
    ) -> Result<()> {
        let s = &mut *self.state;
        let mut slots = s.slots.base;
        let mut source = source;
        let mut target = self.runtime.checked_device_ptr(self.code_input)?;
        let mut pcm = self.runtime.checked_device_ptr(self.pcm_output)?;
        let mut packet = s.packet.base;
        let mut n = rows.len() as u32;
        let mut capacity = s.capacity as u32;
        let mut k = s.codebooks as u32;
        let mut samples = s.samples as u32;
        let mut stride = s.stride as u32;
        let mut stop = stop;
        let slot_stage = Arc::get_mut(&mut s.slot_stage)
            .ok_or_else(|| reject("codec tick staging is quarantined"))?;
        for (i, row) in rows.iter().enumerate() {
            slot_stage.as_mut_slice()[i * 4..i * 4 + 4]
                .copy_from_slice(&(row.slot as u32).to_le_bytes());
        }
        let queued = (|| {
            // SAFETY: both runtimes and all staging remain borrowed until the drain below.
            unsafe {
                self.runtime
                    .enqueue_chained(stream, self.programs, self.writes)?;
                be.memcpy_htod_async(slots, &slot_stage.as_slice()[..rows.len() * 4], stream)?;
            }
            let mut args = [
                &mut source as *mut u64 as *mut _,
                &mut slots as *mut u64 as *mut _,
                &mut target as *mut u64 as *mut _,
                &mut n as *mut u32 as *mut _,
                &mut capacity as *mut u32 as *mut _,
                &mut k as *mut u32 as *mut _,
                &mut stop as *mut u32 as *mut _,
            ];
            be.launch_kernel(
                s.codes_kernel,
                (capacity * k).div_ceil(256),
                256,
                0,
                &mut args,
                Some(stream),
            )?;
            for &(tensor, mut scale) in self.lengths {
                let mut length = self.runtime.checked_device_ptr(tensor)?;
                let mut args = [
                    &mut source as *mut u64 as *mut _,
                    &mut slots as *mut u64 as *mut _,
                    &mut length as *mut u64 as *mut _,
                    &mut n as *mut u32 as *mut _,
                    &mut capacity as *mut u32 as *mut _,
                    &mut k as *mut u32 as *mut _,
                    &mut scale as *mut u32 as *mut _,
                    &mut stop as *mut u32 as *mut _,
                ];
                be.launch_kernel(
                    s.lengths_kernel,
                    capacity.div_ceil(256),
                    256,
                    0,
                    &mut args,
                    Some(stream),
                )?;
            }
            // SAFETY: the same stream orders controls, generated codes, and codec programs.
            unsafe {
                self.runtime
                    .enqueue_chained_programs(stream, self.programs)?;
            }
            let mut args = [
                &mut source as *mut u64 as *mut _,
                &mut slots as *mut u64 as *mut _,
                &mut pcm as *mut u64 as *mut _,
                &mut packet as *mut u64 as *mut _,
                &mut n as *mut u32 as *mut _,
                &mut k as *mut u32 as *mut _,
                &mut samples as *mut u32 as *mut _,
                &mut stride as *mut u32 as *mut _,
                &mut stop as *mut u32 as *mut _,
            ];
            be.launch_kernel(s.pack_kernel, n, 256, 0, &mut args, Some(stream))?;
            let output = Arc::get_mut(&mut s.packet_stage)
                .ok_or_else(|| reject("codec tick output is quarantined"))?;
            // SAFETY: persistent pinned destination is retained through the drain below.
            unsafe {
                be.memcpy_dtoh_async(
                    &mut output.as_mut_slice()[..rows.len() * s.stride],
                    packet,
                    stream,
                )
            }
        })();
        if let Err(error) = be.stream_synchronize(stream) {
            s.failed = true;
            s.slots.quarantine();
            s.packet.quarantine();
            std::mem::forget(s.slot_stage.clone());
            std::mem::forget(s.packet_stage.clone());
            self.runtime.quarantine_chain();
            return Err(error);
        }
        queued?;
        let bytes = s.packet_stage.as_slice();
        self.pcm.clear();
        self.pcm.reserve(rows.len() * s.samples);
        for (i, row) in rows.iter().enumerate() {
            let base = i * s.stride;
            let codes = &bytes[base..base + s.codebooks * 4];
            stage[row.slot * s.codebooks * 4..(row.slot + 1) * s.codebooks * 4]
                .copy_from_slice(codes);
            let audio = &bytes[base + s.codebooks * 4..base + s.codebooks * 4 + s.samples * 2];
            #[cfg(target_endian = "little")]
            if let Ok(samples) = bytemuck::try_cast_slice::<u8, i16>(audio) {
                self.pcm.extend_from_slice(samples);
                continue;
            }
            self.pcm.extend(
                audio
                    .chunks_exact(2)
                    .map(|b| i16::from_le_bytes([b[0], b[1]])),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::packet_runtime::{PacketAsset, PacketData, PacketRuntime};

    fn single_layer_fixture(assets: &std::path::Path, cap: usize) -> std::path::PathBuf {
        use packet::dev::DevOp;
        use packet::devbuild::{Builder, Model, SectionData, SECT_METADATA};
        use plow_asset::packet_pipeline::{
            PacketPipeline, PacketPipelines, PipelineDType, PipelineTensor, SECTION,
        };
        use std::collections::BTreeMap;
        let dir = assets.parent().unwrap().join("codec-tick-layer-probe");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(
            assets.join("interp_sm90a_speech.cubin"),
            dir.join("interp_sm90a_speech.cubin"),
        )
        .unwrap();
        let (k, samples, vocab) = (32usize, 1537usize, 64usize);
        let mut b = Builder::new(132);
        b.set_tensor_dedup(true);
        let input = b.tensor("codes", (cap * k * 4) as u64);
        let pcm = b.tensor("pcm", (cap * samples * 4) as u64);
        let mut tensors = BTreeMap::new();
        for (role, name, dtype, n) in [
            ("codes", "codes", PipelineDType::U32, cap * k),
            ("pcm", "pcm", PipelineDType::F32, cap * samples),
            ("stream.slot", "slot", PipelineDType::U32, cap),
            ("stream.pos", "position", PipelineDType::U32, cap),
            ("stream.rope_pos", "rope", PipelineDType::U32, cap),
            ("stream.n.0", "length", PipelineDType::U32, cap),
        ] {
            b.tensor(name, (n * 4) as u64);
            tensors.insert(
                role.to_owned(),
                PipelineTensor {
                    name: name.into(),
                    dtype,
                    shape: vec![n as u64],
                },
            );
        }
        let table: Vec<f32> = (0..vocab * samples)
            .map(|i| ((i * 37 % 10001) as f32 - 5000.) / 4000.)
            .collect();
        let weights = b.tensor_init("table", bytemuck::cast_slice(&table).to_vec());
        b.emit(DevOp::GatherRowsF32, b.all(), &[], |d| {
            d.t[..3].copy_from_slice(&[pcm, weights, input]);
            d.i = [
                cap as u32,
                samples as u32,
                vocab as u32,
                1,
                1,
                k as u32,
                0,
                0,
            ];
            d.j = [samples as u32, 0];
        });
        let tensors_model = b.tensors();
        let program = b.finish();
        let model = Model {
            n_cu: 132,
            target: 0,
            tensors: tensors_model,
            progs: vec![program],
            prog_t: vec![1],
            kv_row_insts: vec![],
            gen: vec![],
        };
        let pipelines = PacketPipelines {
            version: 1,
            pipelines: vec![PacketPipeline {
                name: "single_embedding_layer".into(),
                driver: "codec.v1".into(),
                programs: BTreeMap::from([(format!("step.b{cap}.f1"), 0)]),
                tensors,
                parameters: BTreeMap::from([
                    ("codec.frame_codes".into(), k as u64),
                    ("codec.frame_samples".into(), samples as u64),
                    ("codec.codebook".into(), vocab as u64),
                    ("stream.slots".into(), 512),
                    ("lengths.count".into(), 1),
                    ("lengths.0.rows_per_frame".into(), 1),
                ]),
                strings: BTreeMap::new(),
            }],
        };
        let path = dir.join(format!("single-layer.b{cap}.pkt"));
        std::fs::write(
            &path,
            model.to_blob_v6(&[SectionData {
                kind: SECT_METADATA,
                name: SECTION.into(),
                data: serde_json::to_vec(&pipelines).unwrap(),
            }]),
        )
        .unwrap();
        path
    }

    #[test]
    #[ignore = "CUDA; CODEC_TICK_TEST_ASSETS supplies the static speech object for a synthetic single-layer packet"]
    fn codec_tick_matches_completed_single_layer() {
        let assets = std::path::PathBuf::from(std::env::var("CODEC_TICK_TEST_ASSETS").unwrap());
        let be = Arc::new(CudaBackend::new(0).unwrap());
        for n in [32usize, 64, 128, 200, 224, 300] {
            let path = single_layer_fixture(&assets, n.next_multiple_of(32));
            let asset = PacketAsset::load(&path).unwrap();
            let mut baseline = CudaPacketRuntime::load_on(be.clone(), &path).unwrap();
            let mut chained = CudaPacketRuntime::load_on(be.clone(), &path).unwrap();
            let p = asset.bind_driver("codec.v1", &baseline).unwrap();
            let cap = p
                .programs()
                .filter_map(|(r, _)| {
                    r.strip_prefix("step.b")?
                        .strip_suffix(".f1")?
                        .parse::<usize>()
                        .ok()
                })
                .filter(|&b| b >= n)
                .min()
                .unwrap();
            let programs = vec![p.program(&format!("step.b{cap}.f1")).unwrap()];
            let codes = p.tensor("codes").unwrap();
            let pcm = p.tensor("pcm").unwrap();
            let k = p.parameter("codec.frame_codes").unwrap() as usize;
            let samples = p.parameter("codec.frame_samples").unwrap() as usize;
            let slots = p.parameter("stream.slots").unwrap() as usize;
            let stop = p.parameter("codec.codebook").unwrap() as u32;
            let rpf = p
                .optional_parameter("stream.transformer_rows_per_frame")
                .unwrap_or(1) as usize;
            let rows: Vec<_> = (0..n)
                .map(|r| FishRow {
                    slot: (r * 73) % slots,
                    hidden_row: 0,
                    seed: 0,
                    step: 0,
                    temperature: 1.,
                    top_p: 1.,
                    top_k: 1,
                    ras_temperature: 1.,
                    ras_top_p: 1.,
                    repetition_penalty: 1.,
                    cfg: None,
                })
                .collect();
            let mut raw: Vec<u32> = (0..slots * k)
                .map(|i| (i as u32 * 139 + 7) % stop)
                .collect();
            for row in &rows {
                if row.slot % 11 == 0 {
                    raw[row.slot * k..(row.slot + 1) * k].fill(0);
                    raw[row.slot * k] = stop;
                }
            }
            let lengths: Vec<_> = (0..p.parameter("lengths.count").unwrap())
                .map(|i| {
                    (
                        p.tensor(&format!("stream.n.{i}")).unwrap(),
                        p.parameter(&format!("lengths.{i}.rows_per_frame")).unwrap() as u32,
                    )
                })
                .collect();
            let mut controls: Vec<(PacketTensor, Vec<u32>)> = [
                "stream.slot",
                "stream.pos",
                "stream.rope_pos",
                "stream.row_pos",
            ]
            .into_iter()
            .filter_map(|name| p.optional_tensor(name))
            .map(|t| (t, vec![0; t.bytes / 4]))
            .collect();
            let mut spare = (0..slots).filter(|&s| !rows.iter().any(|r| r.slot == s));
            for r in 0..cap {
                controls[0].1[r] =
                    rows.get(r)
                        .map(|r| r.slot)
                        .unwrap_or_else(|| spare.next().unwrap()) as u32;
                for q in 0..rpf {
                    controls[2].1[r * rpf + q] = q as u32;
                }
            }
            for &(tensor, scale) in &lengths {
                let mut values = vec![0; tensor.bytes / 4];
                for (r, row) in rows.iter().enumerate() {
                    values[r] = if raw[row.slot * k] == stop { 0 } else { scale };
                }
                controls.push((tensor, values));
            }
            let mut packed = vec![0; cap * k];
            for (r, row) in rows.iter().enumerate() {
                if raw[row.slot * k] != stop {
                    packed[r * k..(r + 1) * k]
                        .copy_from_slice(&raw[row.slot * k..(row.slot + 1) * k]);
                }
            }
            let writes: Vec<_> = controls
                .iter()
                .map(|(tensor, values)| PacketWrite {
                    tensor: *tensor,
                    offset: 0,
                    data: PacketData::Host(bytemuck::cast_slice(values)),
                })
                .collect();
            let mut all: Vec<_> = writes
                .iter()
                .map(|w| PacketWrite {
                    tensor: w.tensor,
                    offset: w.offset,
                    data: w.data,
                })
                .collect();
            all.push(PacketWrite {
                tensor: codes,
                offset: 0,
                data: PacketData::Host(bytemuck::cast_slice(&packed)),
            });
            let tensors: Vec<_> = all.iter().map(|w| w.tensor).collect();
            for rt in [&mut baseline, &mut chained] {
                rt.prepare_writes(&tensors).unwrap();
                for _ in 0..2 {
                    rt.write_tensors(&all).unwrap();
                    rt.run_sequence(&programs).unwrap();
                }
            }
            let source = be.alloc(0, (raw.len() * 4) as u64).unwrap();
            be.upload(&source, 0, bytemuck::cast_slice(&raw)).unwrap();
            let stream = be.stream_create().unwrap();
            let mut state = CodecTickState::new(&chained, cap, k, samples).unwrap();
            let mut audio = Vec::new();
            let mut stage = vec![0x5a; raw.len() * 4];
            let mut expected = vec![0f32; cap * samples];
            for repeat in 0..2 {
                let started = std::time::Instant::now();
                baseline.write_tensors(&all).unwrap();
                baseline.run_sequence(&programs).unwrap();
                baseline
                    .read_tensor_at(pcm, 0, bytemuck::cast_slice_mut(&mut expected))
                    .unwrap();
                let base_us = started.elapsed().as_secs_f64() * 1e6;
                let mut tick = CodecTick {
                    state: &mut state,
                    runtime: &mut chained,
                    programs: &programs,
                    writes: &writes,
                    code_input: codes,
                    pcm_output: pcm,
                    lengths: &lengths,
                    pcm: &mut audio,
                };
                tick.validate(&be, &rows, k, source.len, stage.len())
                    .unwrap();
                let started = std::time::Instant::now();
                tick.complete(&be, &stream, source.base, &rows, stop, &mut stage)
                    .unwrap();
                let chain_us = started.elapsed().as_secs_f64() * 1e6;
                for &(tensor, scale) in &lengths {
                    let mut values = vec![0u32; cap];
                    chained
                        .read_tensor_at(tensor, 0, bytemuck::cast_slice_mut(&mut values))
                        .unwrap();
                    for r in 0..cap {
                        assert_eq!(
                            values[r],
                            if r < n && raw[rows[r].slot * k] != stop {
                                scale
                            } else {
                                0
                            }
                        );
                    }
                }
                for (r, row) in rows.iter().enumerate() {
                    assert_eq!(
                        &stage[row.slot * k * 4..(row.slot + 1) * k * 4],
                        bytemuck::cast_slice::<u32, u8>(&raw[row.slot * k..(row.slot + 1) * k])
                    );
                    for s in 0..samples {
                        let x = expected[r * samples + s];
                        let q = if raw[row.slot * k] == stop {
                            0
                        } else {
                            (x.clamp(-1., 1.) * 32767.).round() as i16
                        };
                        assert_eq!(audio[r * samples + s], q, "B{n} row{r} sample{s}");
                    }
                }
                println!("CODEC_CHAIN B={n} cap={cap} repeat={repeat} base_us={base_us:.3} chain_us={chain_us:.3}");
            }
            let tick = CodecTick {
                state: &mut state,
                runtime: &mut chained,
                programs: &programs,
                writes: &writes,
                code_input: codes,
                pcm_output: pcm,
                lengths: &lengths,
                pcm: &mut audio,
            };
            let mut bad = rows.clone();
            bad[1] = bad[0];
            assert!(tick
                .validate(&be, &bad, k, source.len, stage.len())
                .is_err());
            assert!(tick.validate(&be, &rows, k, source.len, 4).is_err());
            let before = stage.clone();
            for r in 0..slots {
                if !rows.iter().any(|row| row.slot == r) {
                    assert!(stage[r * k * 4..(r + 1) * k * 4].iter().all(|&b| b == 0x5a));
                }
            }
            assert_eq!(stage, before);
        }
        println!("PASS chained codec: codes, stops, exact int16 PCM, sparse slots, padded rows, duplicate/bounds refusal");
    }
}
