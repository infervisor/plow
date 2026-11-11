use super::*;
use crate::exec::mixed_step_staging::{fill_token_batch_words, TokenBatchLayout};
use plow_asset::decode_objects::DecodeObjects;

struct Program {
    index: usize,
    rows: usize,
    layout: TokenBatchLayout,
    arg: DevProgram,
    counters: u64,
    counter_bytes: usize,
    object: Arc<BoundDecodeObject>,
    _tables: Vec<DeviceMem>,
}

pub(super) struct RowDecode {
    programs: Vec<Program>,
    host: Arc<PinnedHost>,
    metadata: DeviceMem,
    descriptor_offset: usize,
}

impl RowDecode {
    pub(super) fn load(
        e: &GpuEngine,
        blob: &DevBlob,
        raw: &[u8],
        assets: &Path,
    ) -> Result<Option<Self>> {
        if !RuntimeConfig::get().row_decode {
            return Ok(None);
        }
        let Some(metadata_bytes) = blob.reserved_metadata(raw, plow_asset::row_decode::SECTION)?
        else {
            if blob.progs.iter().any(|p| p.role.is_row_decode_body()) {
                return Err(RuntimeError::Rejected(
                    "row-decode programs lack object bindings".into(),
                ));
            }
            return Ok(None);
        };
        let metadata: DecodeObjects = serde_json::from_slice(metadata_bytes)
            .map_err(|err| RuntimeError::Rejected(format!("row decode: {err}")))?;
        let block = blob.reserved_metadata(raw, "block.json")?.is_some();
        if !block && (!token_batch::CudaTokenBatch::eligible(e) || !e.has_packed_terminal()) {
            return Err(RuntimeError::Rejected(
                "row decode requires the unified token-batch route".into(),
            ));
        }
        blob.with_packet_view(|packet| {
            plow_asset::row_decode::validate(packet, &metadata, std::mem::size_of::<DevProgram>())
        })
        .map_err(RuntimeError::Rejected)?;
        if e.recurrent.is_some() || e.shared_tail.is_some() || e.be.compute_capability() != (9, 0) {
            return Err(RuntimeError::Rejected(
                "unsupported row-decode device or recurrent/shared-tail state".into(),
            ));
        }
        let mut objects = std::collections::BTreeMap::new();
        let profile = interpreter_profile(e.be.compute_capability()).expect("validated profile");
        for (&id, spec) in &metadata.objects {
            let path = assets.join(&spec.file);
            let image = std::fs::read(&path).map_err(|source| RuntimeError::Io { path, source })?;
            let elf = cubin::inspect(&image)
                .ok_or_else(|| RuntimeError::Rejected("invalid row-decode object".into()))?;
            if !spec.matches_image(&image)
                || spec.profile != profile.tag
                || spec.entry != profile.decode_symbol
                || elf.sm != 90
                || !elf.entries.contains(&spec.entry)
                || cubin::global_u32(&image, plow_asset::row_decode::CAPABILITY) != Some(1)
                || cubin::global_u32(&image, "plow_kv_layout").unwrap_or(0)
                    != packet_kv_layout(blob)
            {
                return Err(RuntimeError::Rejected(
                    "row-decode object identity, layout or capability mismatch".into(),
                ));
            }
            let module = decode_object::DecodeModule::load(&e.be, &image)?;
            GpuEngine::check_packet_pairing(&e.be, &module, assets)?;
            check_norm_weight_offset(&e.be, &module, blob)?;
            let layout = blob.speech_flag_layout(raw)?;
            packet_exec::check_speech_flag_layout(
                &e.be,
                &module,
                &assets.join(&spec.file),
                layout.declared,
                layout.regular,
            )?;
            objects.insert(id, decode_object::bind_module(spec, &e.be, module)?);
        }
        let max_rows = metadata.programs.last().expect("validated bindings").rows as usize;
        let max_layout = TokenBatchLayout::new(max_rows, e.batch, e.batch)?;
        let descriptor_offset = max_layout
            .words()
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(7))
            .map(|bytes| bytes & !7)
            .ok_or_else(|| RuntimeError::Rejected("row-decode staging overflow".into()))?;
        let bytes = descriptor_offset + std::mem::size_of::<packet::dev::TokenBatch>();
        let mut host = e.be.host_alloc_pinned(bytes)?;
        host.as_mut_slice().fill(0);
        let host = Arc::new(host);
        let device = e.be.alloc(0, bytes as u64)?;
        let mut programs = Vec::new();
        for binding in &metadata.programs {
            let p = &blob.progs[binding.index];
            let owned = plow_asset::aux_program::Program {
                rows: p.t,
                n_counter: p.n_counter,
                insts: p.insts.clone(),
                stream: p.stream.clone(),
                stream_ofs: p.stream_ofs.clone(),
                stream_len: p.stream_len.clone(),
                waits: p.waits.clone(),
                succs: p.succs.clone(),
                gq_stream: p.gq_stream.clone(),
                gq_seg_ofs: p.gq_seg_ofs.clone(),
            };
            let (mut arg, counters, counter_bytes, tables) =
                mixed_step::upload_program(&e.be, &owned, e.d_tens.base, 0, 0)?;
            arg.token_batch = device.base + descriptor_offset as u64;
            programs.push(Program {
                index: binding.index,
                rows: binding.rows as usize,
                layout: TokenBatchLayout::new(binding.rows as usize, e.batch, e.batch)?,
                arg,
                counters,
                counter_bytes,
                object: Arc::clone(&objects[&binding.object]),
                _tables: tables,
            });
        }
        Ok(Some(Self {
            programs,
            host,
            metadata: device,
            descriptor_offset,
        }))
    }

    pub(super) fn covering(&self, rows: usize) -> Option<(usize, u32)> {
        self.programs
            .iter()
            .find(|p| p.rows >= rows)
            .map(|p| (p.index, p.rows as u32))
    }

    fn enqueue(
        &mut self,
        e: &mut GpuEngine,
        plan: &plow_asset::token_batch::Plan,
        chunks: &[PackedTokenReq<'_>],
    ) -> Result<()> {
        let program = self
            .programs
            .iter()
            .find(|p| p.rows == plan.row_capacity as usize)
            .ok_or_else(|| RuntimeError::Rejected("row-decode capacity unavailable".into()))?;
        if let Some(pipe) = e.pipe.as_ref() {
            e.be.event_synchronize(&pipe.body_ev)?;
        }
        let bytes = program.layout.words() * 4;
        let row_bytes = u64::from(plan.row_capacity) * 4;
        if row_bytes > e.devp[e.t_ids].len || row_bytes > e.devp[e.t_pos].len {
            return Err(RuntimeError::Rejected(
                "row-decode input tensor capacity exceeded".into(),
            ));
        }
        fill_token_batch_words(
            &program.layout,
            bytemuck::cast_slice_mut(
                &mut Arc::get_mut(&mut self.host)
                    .expect("row-decode staging is exclusive")
                    .as_mut_slice()[..bytes],
            ),
            plan,
        )?;
        let words: &mut [u32] = bytemuck::cast_slice_mut(
            &mut Arc::get_mut(&mut self.host)
                .expect("row-decode staging is exclusive")
                .as_mut_slice()[..bytes],
        );
        for (span, chunk) in plan.spans.iter().zip(chunks) {
            if let Some(token) = chunk.embedding_token {
                words[program.layout.input_ids.start + span.row0 as usize] = token;
            }
        }
        let address =
            |range: &std::ops::Range<usize>| self.metadata.base + (range.start * 4) as u64;
        let descriptor = packet::dev::TokenBatch {
            version: packet::dev::TOKEN_BATCH_VERSION,
            row_capacity: plan.row_capacity,
            real_rows: plan.real_rows,
            sample_rows: plan.sample_rows,
            n_spans: plan.spans.len() as u32,
            flags: 0,
            _pad0: 0,
            _pad1: 0,
            spans: address(&program.layout.span_table),
            input_ids: address(&program.layout.input_ids),
            positions: address(&program.layout.positions),
            active: address(&program.layout.active),
            sample_rows_idx: address(&program.layout.sample_rows),
        };
        Arc::get_mut(&mut self.host)
            .expect("row-decode staging is exclusive")
            .as_mut_slice()[self.descriptor_offset..]
            .copy_from_slice(pod_bytes(&[descriptor]));
        if let Some(vmm) = &mut e.vmm {
            for pending in &plan.pending {
                let slot = pending.slot as usize;
                if let Some(rings) = &mut vmm.rings {
                    rings.ensure_slot(slot)?;
                }
                vmm.ensure_rows(slot, pending.new_frontier)?;
                vmm.advise(slot, pending.new_frontier);
            }
        }
        unsafe {
            e.be.memcpy_htod_async(self.metadata.base, self.host.as_slice(), &e.stream)?;
        }
        e.be.memcpy_dtod_async(
            e.devp[e.t_ids].base,
            descriptor.input_ids,
            u64::from(plan.row_capacity) * 4,
            &e.stream,
        )?;
        e.be.memcpy_dtod_async(
            e.devp[e.t_pos].base,
            descriptor.positions,
            u64::from(plan.row_capacity) * 4,
            &e.stream,
        )?;
        e.be.memset_d8_async(program.counters, 0, program.counter_bytes, &e.stream)?;
        let mut arg = program.arg;
        let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
        e.tb_launched = true;
        e.be.launch_cooperative(
            program.object.function,
            program.object.grid,
            program.object.block,
            program.object.smem,
            &mut params,
            Some(&e.stream),
        )?;
        if let Some(pipe) = e.pipe.as_ref() {
            e.be.event_record(&pipe.body_ev, &e.stream)?;
        }
        Ok(())
    }
}

impl GpuEngine {
    pub fn row_decode_rungs(&self) -> impl DoubleEndedIterator<Item = usize> + '_ {
        self.row_decode
            .iter()
            .flat_map(|row| row.programs.iter().map(|p| p.rows))
    }

    pub(super) fn row_decode_covering(&self, rows: usize) -> Option<(usize, u32)> {
        self.row_decode.as_ref()?.covering(rows)
    }

    pub(super) fn enqueue_row_decode(
        &mut self,
        plan: &plow_asset::token_batch::Plan,
        chunks: &[PackedTokenReq<'_>],
    ) -> Result<()> {
        let mut row = self
            .row_decode
            .take()
            .ok_or_else(|| RuntimeError::Rejected("row decode unavailable".into()))?;
        let result = row.enqueue(self, plan, chunks);
        self.row_decode = Some(row);
        if result.is_err() {
            if let Err(error) = self.be.stream_synchronize(&self.stream) {
                return Err(self.quarantine_row_decode(error));
            }
        }
        result
    }

    pub(super) fn quarantine_row_decode(&mut self, error: RuntimeError) -> RuntimeError {
        if let Some(row) = self.row_decode.take() {
            std::mem::forget(row);
            std::mem::forget(self.be.clone());
        }
        #[cfg(target_os = "linux")]
        return self.be.quarantine_staging(&error);
        #[cfg(not(target_os = "linux"))]
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plow_asset::token_batch::{Phase, Request, Selection};

    #[test]
    #[ignore = "requires TEST_ROW_BLOCK_ASSETS and PLOW_ROW_DECODE=1; run through gpuq"]
    fn row_block_matches_decode_and_causal_prefill() {
        let assets = std::path::PathBuf::from(std::env::var("TEST_ROW_BLOCK_ASSETS").unwrap());
        let desc: plow_asset::BlockDescriptor =
            serde_json::from_slice(&std::fs::read(assets.join("block.json")).unwrap()).unwrap();
        assert_eq!(desc.kind.len(), 2, "this gate runs exactly one layer");
        let hidden = desc.hidden as usize;
        let be = Arc::new(CudaBackend::new(0).unwrap());
        let mut e = GpuEngine::load(be, &assets, &assets.join("checkpoint")).unwrap();
        assert!(e.row_decode.is_some());
        let synth = |rows: usize, salt: usize| -> Vec<f32> {
            (0..rows * hidden)
                .map(|i| (((i * 17 + salt * 31) % 101) as f32 - 50.) / 50.)
                .collect()
        };
        let output = &desc.outputs[0].name;
        let prefix: Vec<u32> = (100..356).collect();
        for slot in 0..130 {
            e.begin_slot(slot, 300).unwrap();
            if slot < 128 {
                e.upload_activation("act.x", &synth(256, slot)).unwrap();
                e.prefill_slot(slot, &prefix).unwrap();
            }
        }
        let input = synth(176, 77);
        let feeds: Vec<_> = (0..128).map(|slot| (slot, 100)).collect();
        let mut decode_times = Vec::new();
        for iteration in 0..12 {
            for slot in 0..128 {
                e.rewind_slot(slot, 256).unwrap();
            }
            e.upload_activation("act.x", &input[..128 * hidden])
                .unwrap();
            let start = std::time::Instant::now();
            e.step_slots(&feeds, &mut Vec::new()).unwrap();
            if iteration >= 2 {
                decode_times.push(start.elapsed().as_secs_f64() * 1e3);
            }
        }
        let mut reference = e.download_activation(output).unwrap()[..128 * hidden].to_vec();
        let prompt: Vec<u32> = (400..424).collect();
        for slot in 128..130 {
            let offset = (128 + (slot - 128) * 24) * hidden;
            e.upload_activation("act.x", &input[offset..offset + 24 * hidden])
                .unwrap();
            e.prefill_slot(slot, &prompt).unwrap();
            reference.extend_from_slice(&e.download_activation(output).unwrap()[..24 * hidden]);
        }
        for slot in 0..130 {
            e.rewind_slot(slot, if slot < 128 { 256 } else { 0 })
                .unwrap();
        }
        let token = [100];
        let requests: Vec<_> = (0..130)
            .map(|slot| Request {
                id: slot as u32,
                slot: slot as u32,
                state_slot: slot as u32,
                generation: e.slot_generations[slot],
                phase: if slot < 128 {
                    Phase::Decode
                } else {
                    Phase::Prefill
                },
                tokens: if slot < 128 { &token } else { &prompt },
                prompt_len: if slot < 128 { 256 } else { 24 },
                selection: Selection::default(),
            })
            .collect();
        let (program, capacity) = e.row_decode_covering(176).unwrap();
        let plan = plow_asset::token_batch::plan(
            &requests,
            &e.pos,
            &e.slot_generations,
            capacity,
            e.max_ctx as u32,
            program as u32,
        )
        .unwrap();
        assert_eq!(plan.sample_rows, 130);
        let chunks: Vec<_> = requests
            .iter()
            .map(|r| PackedTokenReq {
                slot: r.slot as usize,
                tokens: r.tokens,
                c0: e.pos[r.slot as usize] as usize,
                prompt_len: r.prompt_len as usize,
                embedding_token: None,
            })
            .collect();
        let mut times = Vec::new();
        let mut candidate = Vec::new();
        for iteration in 0..12 {
            e.upload_activation("act.x", &input).unwrap();
            let start = std::time::Instant::now();
            e.enqueue_row_decode(&plan, &chunks).unwrap();
            e.be.stream_synchronize(&e.stream).unwrap();
            if iteration >= 2 {
                times.push(start.elapsed().as_secs_f64() * 1e3);
            }
            candidate = e.download_activation(output).unwrap()[..176 * hidden].to_vec();
        }
        for (label, range) in [
            ("decode", 0..128 * hidden),
            ("arrivals", 128 * hidden..176 * hidden),
        ] {
            let mut err2 = 0f64;
            let mut ref2 = 0f64;
            let mut max_abs = 0f32;
            for (&a, &b) in candidate[range.clone()].iter().zip(&reference[range]) {
                assert!(a.is_finite() && b.is_finite());
                err2 += f64::from(a - b).powi(2);
                ref2 += f64::from(b).powi(2);
                max_abs = max_abs.max((a - b).abs());
            }
            let rel_l2 = (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
            println!("ROW_BLOCK_ORACLE {label} rel_l2={rel_l2} max_abs={max_abs}");
            assert!(rel_l2 <= 0.006 && max_abs <= 0.1, "{label} mismatch");
        }
        times.sort_by(f64::total_cmp);
        decode_times.sort_by(f64::total_cmp);
        println!(
            "ROW_BLOCK_CONTROL decode=128 ctx=256 median_ms={}",
            decode_times[decode_times.len() / 2]
        );
        println!("ROW_BLOCK_TIMING decode=128 arrivals=2 prompt_rows=24 rung={capacity} ctx=256 median_ms={} max_ms={}", times[times.len()/2], times[times.len()-1]);
    }
}
