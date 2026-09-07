use super::*;
use crate::exec::mixed_packet::LoadedMixedPacket;
use crate::exec::mixed_step_staging::MixedStepStaging;
use packet::dev::PrefillSpan;
use plow_asset::mixed_step::{DecodeRequest, PrefillRequest};
use std::collections::BTreeMap;

const SPAN_WORDS: usize = std::mem::size_of::<PrefillSpan>() / 4;

struct MixedObject {
    function: KernelFn,
    block: u32,
    smem: u32,
    _module: Arc<DecodeModule>,
}

struct MixedProgram {
    rows: u32,
    decode_rows: u32,
    program_index: u32,
    grid: u32,
    decode_slot: usize,
    sample_outputs: bool,
    object: Arc<MixedObject>,
    kernarg: DevProgram,
    counter_base: u64,
    counter_bytes: usize,
    _tables: Vec<DeviceMem>,
}

#[derive(Clone)]
struct HostLayout {
    rows: usize,
    decode: usize,
    spans: usize,
    ids: std::ops::Range<usize>,
    pos: std::ops::Range<usize>,
    kvlen: std::ops::Range<usize>,
    decode_slot: std::ops::Range<usize>,
    parked: std::ops::Range<usize>,
    prefill_spans: std::ops::Range<usize>,
}

impl HostLayout {
    fn new(rows: usize, decode: usize, spans: usize) -> Result<Self> {
        let mut next = 0usize;
        let mut take = |count: usize| -> Result<std::ops::Range<usize>> {
            let start = next;
            next = next
                .checked_add(count)
                .ok_or_else(|| RuntimeError::Rejected("mixed step staging overflow".into()))?;
            Ok(start..next)
        };
        let ids = take(rows)?;
        let pos = take(rows)?;
        let kvlen = take(rows)?;
        let decode_slot = take(decode)?;
        let parked = take(rows)?;
        let prefill_spans =
            take(spans.checked_mul(SPAN_WORDS).ok_or_else(|| {
                RuntimeError::Rejected("mixed step span staging overflow".into())
            })?)?;
        Ok(Self {
            rows,
            decode,
            spans,
            ids,
            pos,
            kvlen,
            decode_slot,
            parked,
            prefill_spans,
        })
    }

    fn words(&self) -> usize {
        self.prefill_spans.end
    }
}

struct MixedHostStage {
    slab: PinnedHost,
    layout: HostLayout,
}

impl MixedHostStage {
    fn new(be: &CudaBackend, rows: usize, decode: usize, spans: usize) -> Result<Self> {
        let layout = HostLayout::new(rows, decode, spans)?;
        let bytes = layout
            .words()
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::Rejected("mixed step staging bytes overflow".into()))?;
        Ok(Self {
            slab: be.host_alloc_pinned(bytes.max(4))?,
            layout,
        })
    }

    fn words(&self) -> &[u32] {
        bytemuck::cast_slice(self.slab.as_slice())
    }

    fn words_mut(&mut self) -> &mut [u32] {
        bytemuck::cast_slice_mut(self.slab.as_mut_slice())
    }

    fn fill(&mut self, plan: &plow_asset::mixed_step::Plan) -> Result<()> {
        let layout = self.layout.clone();
        fill_words(&layout, self.words_mut(), plan)
    }

    fn bytes(&self, range: std::ops::Range<usize>, count: usize) -> &[u8] {
        bytemuck::cast_slice(&self.words()[range.start..range.start + count])
    }

    fn bytes_mut(&mut self, range: std::ops::Range<usize>, count: usize) -> &mut [u8] {
        let start = range.start * 4;
        &mut self.slab.as_mut_slice()[start..start + count * 4]
    }

    fn decode_tokens(&self, count: usize) -> &[u32] {
        &self.words()[self.layout.ids.start..self.layout.ids.start + count]
    }
}

fn fill_words(
    layout: &HostLayout,
    words: &mut [u32],
    plan: &plow_asset::mixed_step::Plan,
) -> Result<()> {
    if words.len() < layout.words()
        || plan.rows.len() > layout.rows
        || plan.decode_slots.len() > layout.decode
        || plan.prefill_spans.len() > layout.spans
    {
        return Err(RuntimeError::Rejected(
            "mixed step exceeds preallocated staging".into(),
        ));
    }
    for (index, row) in plan.rows.iter().enumerate() {
        words[layout.ids.start + index] = row.token;
        words[layout.pos.start + index] = row.position;
        words[layout.kvlen.start + index] = row.kv_len;
    }
    for (index, &slot) in plan.decode_slots.iter().enumerate() {
        words[layout.decode_slot.start + index] = slot as u32;
    }
    words[layout.parked.start..layout.parked.start + plan.parked.len()]
        .copy_from_slice(&plan.parked);
    for (index, span) in plan.prefill_spans.iter().enumerate() {
        let at = layout.prefill_spans.start + index * SPAN_WORDS;
        words[at..at + SPAN_WORDS].copy_from_slice(&[
            span.row0,
            span.n_rows,
            span.slot,
            span.flags,
            span.kv_row0,
            span.kv_len,
            span.state_slot,
            span.program,
        ]);
    }
    Ok(())
}

pub(super) struct MixedCudaStep {
    programs: Vec<MixedProgram>,
    staging: MixedStepStaging,
    host: MixedHostStage,
    _device_metadata: DeviceMem,
    span_base: u64,
    parked_base: u64,
    ids_base: u64,
    pos_base: u64,
    kvlen_base: u64,
    slot_capacity: u32,
}

impl MixedCudaStep {
    pub(super) fn load(
        be: &Arc<CudaBackend>,
        packet: LoadedMixedPacket<'_>,
        blob: &DevBlob,
        devp: &[DeviceMem],
        tensor_table: u64,
        batch: usize,
    ) -> Result<Self> {
        if packet.backend() != plow_asset::mixed_step::PayloadKind::Cubin
            || packet.physical_slot_capacity() as usize != batch
            || packet.max_active_requests() as usize > batch
        {
            return Err(RuntimeError::Rejected(
                "mixed CUDA packet slot capacity does not match the engine".into(),
            ));
        }
        let variants: Vec<_> = packet.variants().collect();
        let max_rows = variants.iter().map(|v| v.rows()).max().unwrap_or(0) as usize;
        let max_decode = variants.iter().map(|v| v.decode_rows()).max().unwrap_or(0) as usize;
        let max_spans = packet.max_active_requests() as usize;
        let tensor = |name: &str, need: usize| -> Result<(usize, u64)> {
            let index = blob
                .tensors
                .iter()
                .position(|tensor| tensor.name == name)
                .ok_or_else(|| RuntimeError::Rejected(format!("mixed step missing {name}")))?;
            let declared = &blob.tensors[index];
            if declared.init.is_some()
                || declared.bytes < need as u64
                || devp[index].len < need as u64
            {
                return Err(RuntimeError::Rejected(format!(
                    "mixed step {name} has {} bytes, needs {need}",
                    declared.bytes
                )));
            }
            Ok((index, devp[index].base))
        };
        let (_, ids_base) = tensor("in.ids", max_rows * 4)?;
        let (_, pos_base) = tensor("in.pos", max_rows * 4)?;
        let (_, kvlen_base) = tensor("in.kvlen", max_rows * 4)?;

        let span_bytes = max_spans
            .checked_mul(std::mem::size_of::<PrefillSpan>())
            .ok_or_else(|| RuntimeError::Rejected("mixed step span bytes overflow".into()))?;
        let parked_bytes = max_rows
            .checked_mul(4)
            .ok_or_else(|| RuntimeError::Rejected("mixed step parked bytes overflow".into()))?;
        let metadata_bytes = span_bytes
            .checked_add(parked_bytes)
            .ok_or_else(|| RuntimeError::Rejected("mixed step metadata bytes overflow".into()))?;
        let device_metadata = be.alloc(0, metadata_bytes.max(4) as u64)?;
        let span_base = device_metadata.base;
        let parked_base = span_base + span_bytes as u64;
        let host = MixedHostStage::new(be, max_rows, max_decode, max_spans)?;
        let staging = MixedStepStaging::with_capacity(max_rows, max_spans, max_spans);

        let mut objects = BTreeMap::<String, Arc<MixedObject>>::new();
        let mut programs = Vec::with_capacity(variants.len());
        for selected in variants {
            let object_section = selected.object();
            let object = if let Some(object) = objects.get(object_section.name) {
                Arc::clone(object)
            } else {
                let info = cubin::inspect(object_section.bytes).ok_or_else(|| {
                    RuntimeError::Rejected("mixed CUDA object is not a valid cubin".into())
                })?;
                let want_sm = be.compute_capability().0 * 10 + be.compute_capability().1;
                let entry = info
                    .interp_entry(Role::Prefill)
                    .filter(|_| info.sm == want_sm)
                    .ok_or_else(|| {
                        RuntimeError::Rejected(
                            "mixed CUDA object has no matching prefill interpreter".into(),
                        )
                    })?;
                let module = DecodeModule::load(be, object_section.bytes)?;
                if be.module_global_u32(&module, plow_asset::mixed_step::OBJECT_CAPABILITY)?
                    != Some(plow_asset::mixed_step::VERSION)
                {
                    return Err(RuntimeError::Rejected(
                        "mixed CUDA object capability mismatch".into(),
                    ));
                }
                let function = be.get_function(&module, entry)?;
                let block = be
                    .module_global_u32(&module, "plow_block_pf")?
                    .ok_or_else(|| {
                        RuntimeError::Rejected(
                            "mixed CUDA object has no block-size contract".into(),
                        )
                    })?;
                let smem = be
                    .module_global_u32(&module, "plow_arena_bytes_pf")?
                    .ok_or_else(|| {
                        RuntimeError::Rejected(
                            "mixed CUDA object has no shared-memory contract".into(),
                        )
                    })?;
                if block == 0 {
                    return Err(RuntimeError::Rejected(
                        "mixed CUDA object has zero block size".into(),
                    ));
                }
                be.set_max_dynamic_smem(function, smem)?;
                let capacity = be
                    .occupancy_blocks_per_sm(function, block, smem as usize)?
                    .checked_mul(be.sm_count())
                    .ok_or_else(|| RuntimeError::Rejected("mixed CUDA grid overflow".into()))?;
                if packet.n_cu() == 0 || packet.n_cu() > capacity {
                    return Err(RuntimeError::Rejected(format!(
                        "mixed CUDA grid {} exceeds cooperative capacity {capacity}",
                        packet.n_cu()
                    )));
                }
                let object = Arc::new(MixedObject {
                    function,
                    block,
                    smem,
                    _module: module,
                });
                objects.insert(object_section.name.to_owned(), Arc::clone(&object));
                object
            };

            let decode_slot = selected.decode_slot().ok_or_else(|| {
                RuntimeError::Rejected("mixed CUDA program has no decode-slot operand".into())
            })? as usize;
            let need = selected.decode_rows() as usize * 4;
            if blob.tensors.get(decode_slot).is_none_or(|tensor| {
                tensor.name != plow_asset::mixed_step::DECODE_SLOT_TENSOR
                    || tensor.init.is_some()
                    || tensor.bytes < need as u64
            }) {
                return Err(RuntimeError::Rejected(
                    "mixed CUDA decode-slot tensor mismatch".into(),
                ));
            }
            let (kernarg, counter_base, counter_bytes, tables) =
                upload_program(be, selected.program(), tensor_table, span_base, parked_base)?;
            programs.push(MixedProgram {
                rows: selected.rows(),
                decode_rows: selected.decode_rows(),
                program_index: selected.program_index(),
                grid: packet.n_cu(),
                decode_slot,
                sample_outputs: selected
                    .program()
                    .insts
                    .iter()
                    .any(|inst| inst.op == DevOp::ArgmaxFin as u16),
                object,
                kernarg,
                counter_base,
                counter_bytes,
                _tables: tables,
            });
        }

        Ok(Self {
            programs,
            staging,
            host,
            _device_metadata: device_metadata,
            span_base,
            parked_base,
            ids_base,
            pos_base,
            kvlen_base,
            slot_capacity: packet.physical_slot_capacity(),
        })
    }

    fn select(&self, rows: u32, decode_rows: u32) -> Result<usize> {
        self.programs
            .iter()
            .position(|program| program.rows == rows && program.decode_rows == decode_rows)
            .ok_or_else(|| {
                RuntimeError::Rejected(format!(
                    "mixed step has no exact {rows}:{decode_rows} variant"
                ))
            })
    }

    fn row_capacity(&self, decode_rows: u32, prefill_rows: u32) -> Option<u32> {
        select_row_capacity(
            self.programs
                .iter()
                .map(|program| (program.rows, program.decode_rows, program.sample_outputs)),
            decode_rows,
            prefill_rows,
        )
    }
}

fn select_row_capacity(
    variants: impl Iterator<Item = (u32, u32, bool)>,
    decode_rows: u32,
    prefill_rows: u32,
) -> Option<u32> {
    if decode_rows == 0 || prefill_rows == 0 {
        return None;
    }
    variants
        .filter(|&(rows, decode, samples)| samples && decode == decode_rows && rows > decode)
        .min_by_key(|&(rows, decode, _)| {
            let capacity = rows - decode;
            (
                prefill_rows.saturating_sub(capacity),
                capacity.abs_diff(prefill_rows),
            )
        })
        .map(|(rows, _, _)| rows)
}

fn upload_program(
    be: &Arc<CudaBackend>,
    program: &plow_asset::aux_program::Program,
    tensor_table: u64,
    span_base: u64,
    parked_base: u64,
) -> Result<(DevProgram, u64, usize, Vec<DeviceMem>)> {
    if program.gq_seg_ofs.len() != 2 {
        return Err(RuntimeError::Rejected(
            "mixed CUDA program must have one queue segment".into(),
        ));
    }
    let upload = |bytes: &[u8]| -> Result<DeviceMem> {
        let mem = be.alloc(0, bytes.len().max(4) as u64)?;
        if !bytes.is_empty() {
            be.upload(&mem, 0, bytes)?;
        }
        Ok(mem)
    };
    let d_inst = upload(pod_bytes(&program.insts))?;
    let d_stream = upload(pod_bytes(&program.stream))?;
    let d_sofs = upload(pod_bytes(&program.stream_ofs))?;
    let d_slen = upload(pod_bytes(&program.stream_len))?;
    let d_waits = upload(pod_bytes(&program.waits))?;
    let d_succs = upload(pod_bytes(&program.succs))?;
    let d_gq_stream = upload(pod_bytes(&program.gq_stream))?;
    let d_gq_seg = upload(pod_bytes(&program.gq_seg_ofs))?;
    let counter_only = program.n_counter as usize * CTR_STRIDE as usize * 4;
    let cursor_offset = counter_only.max(4);
    let counter_bytes = cursor_offset + CTR_STRIDE as usize * 4;
    let d_counter = be.alloc(0, counter_bytes as u64)?;
    let kernarg = DevProgram {
        insts: d_inst.base,
        stream: d_stream.base,
        stream_ofs: d_sofs.base,
        stream_len: d_slen.base,
        waits: d_waits.base,
        succs: d_succs.base,
        counters: d_counter.base,
        tensors: tensor_table,
        trace: 0,
        cur_seg: 0,
        l2_domains: 0,
        hier_base: 0,
        n_seg: 1,
        gq_stream: d_gq_stream.base,
        gq_seg_ofs: d_gq_seg.base,
        gq_cursor: d_counter.base + cursor_offset as u64,
        xctr: 0,
        peer_scratch: 0,
        rank: 0,
        n_gpu: 1,
        seg_ofs: 0,
        prefill_spans: span_base,
        prefill_parked: parked_base,
        n_prefill_spans: 0,
        n_prefill_rows: program.rows,
    };
    let counter_base = d_counter.base;
    Ok((
        kernarg,
        counter_base,
        counter_bytes,
        vec![
            d_inst,
            d_stream,
            d_sofs,
            d_slen,
            d_waits,
            d_succs,
            d_gq_stream,
            d_gq_seg,
            d_counter,
        ],
    ))
}

impl GpuEngine {
    /// Select the packet-declared mixed variant that covers the most waiting
    /// prefill rows for this decode width, preferring less padding on ties.
    pub fn mixed_step_rows(&self, decode_rows: usize, prefill_rows: usize) -> Option<u32> {
        self.mixed_step.as_ref()?.row_capacity(
            u32::try_from(decode_rows).ok()?,
            u32::try_from(prefill_rows).ok()?,
        )
    }

    /// Execute one compiler-emitted mixed dense block. `rows` selects an
    /// exact packet variant; decode outputs retain the input request order.
    pub fn mixed_step(
        &mut self,
        rows: u32,
        decode: &[DecodeRequest],
        prefill: &[PrefillRequest<'_>],
        output: &mut [u32],
    ) -> Result<()> {
        let mut mixed = self.mixed_step.take().ok_or_else(|| {
            RuntimeError::Rejected("packet has no NVIDIA mixed-step program".into())
        })?;
        let result = (|| -> Result<()> {
            if decode.is_empty() || prefill.is_empty() {
                return Err(RuntimeError::Rejected(
                    "mixed step requires decode and prefill work".into(),
                ));
            }
            let variant_index = mixed.select(rows, decode.len() as u32)?;
            let program = &mixed.programs[variant_index];
            let output_rows = usize::from(program.sample_outputs) * decode.len();
            if output.len() != output_rows {
                return Err(RuntimeError::Rejected(format!(
                    "mixed step output has {} rows, expected {}",
                    output.len(),
                    output_rows
                )));
            }
            let plan = mixed
                .staging
                .stage(
                    decode,
                    prefill,
                    &self.pos,
                    rows,
                    self.max_ctx as u32,
                    program.program_index,
                )
                .map_err(|error| RuntimeError::Rejected(error.to_string()))?;
            if let Err(error) = program_validate_slots(program, plan, mixed.slot_capacity) {
                mixed.staging.discard();
                return Err(error);
            }
            if let Some(vmm) = &mut self.vmm {
                for &(slot, end) in &plan.mapped_ends {
                    if let Some(rings) = &mut vmm.rings {
                        rings.ensure_slot(slot as usize)?;
                    }
                    vmm.kv.ensure_rows(slot as usize, end)?;
                }
            }
            mixed.host.fill(plan)?;
            let layout = mixed.host.layout.clone();
            let submitted = (|| -> Result<()> {
                unsafe {
                    self.be.memcpy_htod_async(
                        mixed.ids_base,
                        mixed.host.bytes(layout.ids.clone(), rows as usize),
                        &self.stream,
                    )?;
                    self.be.memcpy_htod_async(
                        mixed.pos_base,
                        mixed.host.bytes(layout.pos.clone(), rows as usize),
                        &self.stream,
                    )?;
                    self.be.memcpy_htod_async(
                        mixed.kvlen_base,
                        mixed.host.bytes(layout.kvlen.clone(), rows as usize),
                        &self.stream,
                    )?;
                    self.be.memcpy_htod_async(
                        self.devp[program.decode_slot].base,
                        mixed.host.bytes(layout.decode_slot.clone(), decode.len()),
                        &self.stream,
                    )?;
                    self.be.memcpy_htod_async(
                        mixed.span_base,
                        mixed
                            .host
                            .bytes(layout.prefill_spans.clone(), prefill.len() * SPAN_WORDS),
                        &self.stream,
                    )?;
                    self.be.memcpy_htod_async(
                        mixed.parked_base,
                        mixed.host.bytes(layout.parked.clone(), rows as usize),
                        &self.stream,
                    )?;
                }
                self.be.memset_d8_async(
                    program.counter_base,
                    0,
                    program.counter_bytes,
                    &self.stream,
                )?;
                let mut arg = program.kernarg;
                arg.n_prefill_spans = prefill.len() as u32;
                let mut params = [&mut arg as *mut DevProgram as *mut std::ffi::c_void];
                self.be.launch_cooperative(
                    program.object.function,
                    program.grid,
                    program.object.block,
                    program.object.smem,
                    &mut params,
                    Some(&self.stream),
                )?;
                if program.sample_outputs {
                    unsafe {
                        self.be.memcpy_dtoh_async(
                            mixed.host.bytes_mut(layout.ids.clone(), decode.len()),
                            mixed.ids_base,
                            &self.stream,
                        )?;
                    }
                }
                Ok(())
            })();
            if let Err(error) = submitted {
                let _ = self.be.stream_synchronize(&self.stream);
                mixed.staging.discard();
                return Err(error);
            }
            if let Err(error) = self.be.stream_synchronize(&self.stream) {
                mixed.staging.discard();
                return Err(error);
            }
            if program.sample_outputs {
                mixed
                    .staging
                    .finish_after_device_success(
                        &mut self.pos,
                        mixed.host.decode_tokens(decode.len()),
                        output,
                    )
                    .map_err(|error| RuntimeError::Rejected(error.to_string()))?;
            } else {
                mixed
                    .staging
                    .commit_after_device_success(&mut self.pos)
                    .map_err(|error| RuntimeError::Rejected(error.to_string()))?;
            }
            if self.vmm_prefix_enabled() {
                for request in decode {
                    self.seq_tokens[request.slot as usize].push(request.token);
                }
                for request in prefill {
                    self.seq_tokens[request.slot as usize].extend_from_slice(request.tokens);
                }
            }
            if let Some(vmm) = &self.vmm {
                for request in decode {
                    vmm.kv
                        .advise(request.slot as usize, self.pos[request.slot as usize]);
                }
                for request in prefill {
                    vmm.kv
                        .advise(request.slot as usize, self.pos[request.slot as usize]);
                }
            }
            Ok(())
        })();
        if result.is_err() {
            mixed.staging.discard();
        }
        self.mixed_step = Some(mixed);
        result
    }
}

fn program_validate_slots(
    program: &MixedProgram,
    plan: &plow_asset::mixed_step::Plan,
    slot_capacity: u32,
) -> Result<()> {
    plow_asset::mixed_step::validate_decode_slot_binding(
        &plan.decode_slots,
        program.decode_rows,
        slot_capacity,
        Some(program.decode_slot as u16),
    )
    .map_err(RuntimeError::Rejected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_layout_is_fixed_disjoint_and_exact() {
        let layout = HostLayout::new(128, 16, 16).unwrap();
        let ranges = [
            layout.ids.clone(),
            layout.pos.clone(),
            layout.kvlen.clone(),
            layout.decode_slot.clone(),
            layout.parked.clone(),
            layout.prefill_spans.clone(),
        ];
        for pair in ranges.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        assert_eq!(layout.words(), 128 * 4 + 16 + 16 * SPAN_WORDS);
    }

    #[test]
    fn row_images_reuse_the_same_bounded_storage() {
        let layout = HostLayout::new(8, 2, 2).unwrap();
        let mut words = vec![0u32; layout.words()];
        let pointer = words.as_ptr();
        let capacity = words.capacity();
        let frontiers = [0, 4, 8, 12];
        for token in [7, 9] {
            let decode = [DecodeRequest {
                slot: 2,
                state_slot: 0,
                token,
            }];
            let prefill_tokens = [10, 11];
            let prefill = [PrefillRequest {
                slot: 1,
                state_slot: 3,
                start: 4,
                tokens: &prefill_tokens,
                prompt_len: 64,
            }];
            let plan =
                plow_asset::mixed_step::plan(&decode, &prefill, &frontiers, 8, 64, 5).unwrap();
            fill_words(&layout, &mut words, &plan).unwrap();
            assert_eq!(words[layout.ids.start], token);
            assert_eq!(words[layout.pos.start], 8);
            assert_eq!(words[layout.kvlen.start], 9);
            assert_eq!(words[layout.decode_slot.start], 2);
            assert_eq!(words[layout.prefill_spans.start], 1);
            assert_eq!(words.as_ptr(), pointer);
            assert_eq!(words.capacity(), capacity);
        }
    }

    #[test]
    fn mixed_row_capacity_prefers_a_cover_then_the_largest_partial() {
        let variants = [(64, 1, true), (128, 1, true), (256, 1, true)];
        assert_eq!(select_row_capacity(variants.into_iter(), 1, 70), Some(128));
        assert_eq!(select_row_capacity(variants.into_iter(), 1, 300), Some(256));
    }

    #[test]
    fn mixed_row_capacity_requires_sampled_matching_decode_width() {
        let variants = [(128, 1, false), (128, 2, true)];
        assert_eq!(select_row_capacity(variants.into_iter(), 1, 64), None);
        assert_eq!(select_row_capacity(variants.into_iter(), 2, 64), Some(128));
    }
}
