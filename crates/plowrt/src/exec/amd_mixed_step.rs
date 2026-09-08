use super::*;
use crate::exec::mixed_step_staging::{fill_words, HostLayout, MixedStepStaging, SPAN_WORDS};
use plow_asset::mixed_step::{DecodeRequest, PrefillRequest, SpanCover};

/// Which packed route an instance of [`MixedAmdStep`] is.
///
/// The two share a synthesized program, a host staging layout and a launch. They differ in the
/// code object, the kernel symbol, the span contract on the wire, and which prefill buckets
/// they can execute — so they are one type with a route rather than two copies of five hundred
/// lines that would drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum StepRoute {
    /// Mixed step v1 (`PLOW_FUSION`): a decode band ahead of the spans, `interp_mixed_gq.elf`.
    Mixed,
    /// The unified token batch (`plans/unified-token-batch.md`): spans cover `[0, M)`,
    /// `interp_tokbatch_gq.elf`.
    TokenBatch,
}

impl StepRoute {
    fn object(self) -> &'static str {
        match self {
            StepRoute::Mixed => "interp_mixed_gq.elf",
            StepRoute::TokenBatch => super::amd_token_batch::TOKEN_BATCH_OBJECT,
        }
    }

    fn kernel(self, arch: &str) -> String {
        match self {
            StepRoute::Mixed => format!("plow_interp_mixed_{arch}_gq"),
            StepRoute::TokenBatch => super::amd_token_batch::token_batch_kernel_symbol(arch),
        }
    }

    fn cover(self) -> SpanCover {
        match self {
            StepRoute::Mixed => SpanCover::DecodeBand,
            StepRoute::TokenBatch => SpanCover::PrefixFree,
        }
    }

    fn label(self) -> &'static str {
        match self {
            StepRoute::Mixed => "fusion",
            StepRoute::TokenBatch => "token batch",
        }
    }
}

struct MixedProgram {
    rows: u32,
    decode_rows: u32,
    program_index: u32,
    decode_slot: u64,
    kernel: HsaKernel,
    block: u32,
    arg: DevProgram,
    counter_bytes: usize,
    samples: bool,
    _tables: Vec<DeviceMem>,
}

pub(super) struct MixedAmdStep {
    pub(super) route: StepRoute,
    programs: Vec<MixedProgram>,
    staging: MixedStepStaging,
    layout: HostLayout,
    host: HsaPinned,
    zero: HsaPinned,
    metadata: DeviceMem,
    parked_base: u64,
    ids_base: u64,
    pos_base: u64,
    kvlen_base: u64,
    vocab: u32,
    _buffers: Vec<DeviceMem>,
    /// Unloaded by `Drop`. `Module` has no destructor of its own, so a plain field here
    /// leaks the HSA executable for the life of the process; the device buffers beside it
    /// are freed by `DeviceMem`'s own cleanup, which is what made the asymmetry easy to miss.
    module: Module,
    /// Held only so `Drop` can reach `module_unload`.
    be: Arc<HsaBackend>,
}

/// Unloads a freshly-loaded module unless the constructor completes.
///
/// `Module` has no destructor, and `MixedAmdStep::load` has nineteen error exits between
/// `module_load` and its `Ok(Self { .. })`. Cleanup that has to be spelled at each one is
/// cleanup that a future refusal will omit.
struct ModuleGuard<'a> {
    be: &'a Arc<HsaBackend>,
    module: Option<Module>,
}

impl Drop for ModuleGuard<'_> {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            if let Err(e) = EngineDevice::module_unload(&**self.be, &module) {
                tracing::warn!(error = %e, "fusion: unloading after a refusal failed");
            }
        }
    }
}

impl Drop for MixedAmdStep {
    fn drop(&mut self) {
        // Teardown, not a hot path: report and continue rather than panic in a destructor.
        if let Err(e) = EngineDevice::module_unload(&*self.be, &self.module) {
            tracing::warn!(error = %e, "fusion: unloading the mixed-step module failed");
        }
    }
}

impl MixedAmdStep {
    pub(super) fn load(
        be: &Arc<HsaBackend>,
        blob: &DevBlob,
        devp: &[DeviceMem],
        hsaco_dir: &Path,
        batch: usize,
        route: StepRoute,
    ) -> Result<Self> {
        let mut synthesized = crate::exec::mixed_program::synthesize(blob, batch)?;
        if route == StepRoute::TokenBatch {
            // The route is qualified at `nsplit == 1` with a fused flash epilogue, and
            // `runtime/amd/interp.hip`'s token-batch FlashPrefill arm TRAPS on anything else:
            // when the KV partition moves with a span boundary the merge is no longer
            // associative, and `exec_mixed_prefill_merge` still resolves rows through
            // `plow_mixed_prefill_span`, which requires the decode prefix this contract
            // removes. Drop those buckets here rather than admit a plan that would trap —
            // a refused bucket is a scheduling limit, a trapped wavefront is an outage.
            synthesized.programs.retain(|spec| {
                spec.program
                    .insts
                    .iter()
                    .filter(|i| i.op == DevOp::FlashPrefill as u16)
                    .all(|i| i.i[7] == 1 && i.t[5] != packet::dev::TENSOR_NONE16)
            });
            if synthesized.programs.is_empty() {
                return Err(RuntimeError::Rejected(
                    "capability `token_batch_unsplit_attention`: no prefill bucket in this blob \
                     has nsplit=1 with a fused flash epilogue"
                        .into(),
                ));
            }
        }
        let path = hsaco_dir.join(route.object());
        let object = std::fs::read(&path).map_err(|error| {
            RuntimeError::Rejected(format!(
                "{} object {}: {error}",
                route.label(),
                path.display()
            ))
        })?;
        let markers: &[&str] = match route {
            StepRoute::Mixed => &[
                "plow_mixed_dynamic_rows_1",
                "plow_mixed_step_bf16_1",
                "plow_mixed_gemm_glu_1",
                "plow_mixed_prefill_split_1",
            ],
            // The token-batch object is built ON the mixed object's shape, so it must answer
            // for that contract too; the four `plow_token_batch_*` names are what say the
            // descriptor-aware arms are actually present rather than merely compiled around.
            StepRoute::TokenBatch => &[
                "plow_mixed_dynamic_rows_1",
                "plow_mixed_step_bf16_1",
                "plow_mixed_gemm_glu_1",
                "plow_mixed_prefill_split_1",
                "plow_token_batch_1",
                "plow_token_batch_dense_gqa_1",
                "plow_token_batch_combined_m_1",
                "plow_token_batch_span_attn_1",
            ],
        };
        for marker in markers {
            if elf_symbol_u32(&object, marker) != Some(1) {
                return Err(RuntimeError::Rejected(format!(
                    "{} object lacks {marker}",
                    route.label()
                )));
            }
        }
        if elf_symbol_u32(&object, "plow_mixed_block") != Some(256)
            || blob.n_cu > EngineDevice::sm_count(&**be)
        {
            return Err(RuntimeError::Rejected(
                "fusion object launch geometry mismatch".into(),
            ));
        }
        let module = EngineDevice::module_load(&**be, &object)?;
        let symbol = route.kernel(&EngineDevice::arch(&**be).to_string());
        let kernel = EngineDevice::get_function(&**be, &module, &symbol)?;
        let abi = std::mem::size_of::<DevProgram>() as u32;
        // From here to the `Ok(Self { .. })` below the module is loaded and owned by nobody,
        // and NINETEEN error exits follow — mostly `?`, which no closure or explicit call can
        // cover. `guard` unloads on drop unless the constructor reaches the end and disarms
        // it, so a refusal added later cannot reintroduce the leak by forgetting to.
        let mut guard = ModuleGuard {
            be,
            module: Some(module),
        };
        if ![abi, abi + 256].contains(&kernel.kernarg_size()) || kernel.group_segment_size() > 65536
        {
            return Err(RuntimeError::Rejected(
                "fusion object ABI or LDS mismatch".into(),
            ));
        }
        let rows = synthesized
            .programs
            .iter()
            .map(|p| p.program.rows)
            .max()
            .unwrap_or(0) as usize;
        let decode = synthesized
            .programs
            .iter()
            .map(|p| p.decode_rows)
            .max()
            .unwrap_or(0) as usize;
        if rows == 0 || decode == 0 {
            return Err(RuntimeError::Rejected(format!(
                "packet has no supported {} buckets",
                route.label()
            )));
        }
        // PrefixFree turns each decode request into a span and splits a completing prompt into
        // two, so the span table is bounded by `2 * batch` rather than by `batch`.
        let span_capacity = if route == StepRoute::TokenBatch {
            batch * 2
        } else {
            batch
        };
        let count = synthesized
            .tensors
            .iter()
            .map(|t| t.handle as usize + 1)
            .max()
            .unwrap_or(devp.len())
            .max(devp.len());
        let mut addresses: Vec<u64> = devp.iter().map(|m| m.base).collect();
        addresses.resize(count, 0);
        let mut names: Vec<String> = blob.tensors.iter().map(|t| t.name.clone()).collect();
        names.resize(count, String::new());
        let mut sizes: Vec<u64> = blob.tensors.iter().map(|t| t.bytes).collect();
        sizes.resize(count, 0);
        let mut initialized: Vec<bool> = blob.tensors.iter().map(|t| t.init.is_some()).collect();
        initialized.resize(count, false);
        let mut buffers = Vec::with_capacity(synthesized.tensors.len() + 1);
        for tensor in &synthesized.tensors {
            let handle = tensor.handle as usize;
            if initialized[handle] {
                return Err(RuntimeError::Rejected(
                    "fusion cannot override initialized tensors".into(),
                ));
            }
            let mem = EngineDevice::alloc(&**be, tensor.bytes)?;
            addresses[handle] = mem.base;
            names[handle] = tensor.name.clone();
            sizes[handle] = tensor.bytes;
            buffers.push(mem);
        }
        let tensors: Vec<_> = names
            .iter()
            .enumerate()
            .map(|(i, name)| plow_asset::mixed_step::TensorContract {
                name,
                bytes: sizes[i],
                initialized: initialized[i],
            })
            .collect();
        let base = |name: &str| -> Result<u64> {
            names
                .iter()
                .position(|n| n == name)
                .map(|i| addresses[i])
                .ok_or_else(|| RuntimeError::Rejected(format!("fusion missing {name}")))
        };
        let ids_base = base("in.ids")?;
        let pos_base = base("in.pos")?;
        let kvlen_base = base("in.kvlen")?;
        let table = EngineDevice::alloc(&**be, (addresses.len() * 8) as u64)?;
        EngineDevice::upload(&**be, &table, 0, as_bytes(&addresses))?;
        let tensor_table = table.base;
        buffers.push(table);
        // Under PrefixFree the leading sampled rows are decode rows PLUS the terminal row of
        // every prompt completing this step, so the compact decode-slot image is `batch` long,
        // not `decode` (= batch - 1) long.
        let leading = if route == StepRoute::TokenBatch {
            batch
        } else {
            decode
        };
        let layout = HostLayout::new(rows, leading, span_capacity)?;
        let host = be.host_alloc_pinned(layout.words() * 4)?;
        let span_bytes = span_capacity * std::mem::size_of::<PrefillSpan>();
        let metadata = EngineDevice::alloc(&**be, (span_bytes + rows * 4) as u64)?;
        let parked_base = metadata.base + span_bytes as u64;
        let mut programs = Vec::with_capacity(synthesized.programs.len());
        let mut vocab = None;
        let glu_lds = elf_symbol_u32(&object, "plow_mixed_glu_lds_halves");
        for (index, spec) in synthesized.programs.iter().enumerate() {
            let program = &spec.program;
            validate_program(program)?;
            plow_asset::mixed_step::dense_amd_capacity_consumer_contract(
                program,
                spec.decode_rows,
                &tensors,
            )
            .map_err(|error| RuntimeError::Rejected(error.to_string()))?;
            for inst in &program.insts {
                if inst.op == DevOp::GemmGlu as u16
                    && !glu_lds.is_some_and(|halves| {
                        inst.i[2] > 0
                            && inst.i[2] <= halves
                            && halves <= kernel.group_segment_size() / 2
                    })
                {
                    return Err(RuntimeError::Rejected(
                        "fusion GLU exceeds object activation capacity".into(),
                    ));
                }
            }
            let embed = program
                .insts
                .iter()
                .find(|i| i.op == DevOp::Embed as u16)
                .ok_or_else(|| RuntimeError::Rejected("fusion missing embedding".into()))?;
            let row_bytes = u64::from(embed.i[1]) * 2;
            let bytes = sizes[embed.t[1] as usize];
            let count = (row_bytes > 0 && bytes % row_bytes == 0)
                .then(|| bytes / row_bytes)
                .and_then(|n| u32::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| {
                    RuntimeError::Rejected("fusion embedding geometry mismatch".into())
                })?;
            if vocab.is_some_and(|prior| prior != count) {
                return Err(RuntimeError::Rejected(
                    "fusion buckets disagree on vocabulary".into(),
                ));
            }
            vocab = Some(count);
            // THE OPERATOR AUDIT, as a load-time gate rather than a habit
            // (`plans/unified-token-batch.md` §3). Every opcode this program contains is
            // classified; anything class C without a declared conversion, class D, or with no
            // descriptor-aware arm is refused BY NAME. An opcode this build cannot name is
            // treated as C, because C is the class that gets refused.
            if route == StepRoute::TokenBatch {
                let caps = plow_asset::token_batch::Capabilities::amd_dense_gqa(
                    EngineDevice::arch(&**be).to_string(),
                    program.rows,
                    spec.decode_rows,
                );
                let audit = caps
                    .refuse_program(program.insts.iter().map(|i| i.op))
                    .map_err(|refusal| RuntimeError::Rejected(refusal.to_string()))?;
                tracing::debug!(
                    rows = program.rows,
                    sample_capacity = spec.decode_rows,
                    opcodes = audit.len(),
                    converted = audit.iter().filter(|e| e.converted).count(),
                    "token batch: operator audit admits every opcode in this bucket"
                );
            }
            let (arg, counter_bytes, tables) =
                upload_program(be, program, tensor_table, metadata.base, parked_base)?;
            programs.push(MixedProgram {
                rows: program.rows,
                decode_rows: spec.decode_rows,
                program_index: index as u32,
                decode_slot: addresses[spec.decode_slot as usize],
                kernel,
                block: 256,
                arg,
                counter_bytes,
                samples: program
                    .insts
                    .iter()
                    .any(|i| i.op == DevOp::ArgmaxFin as u16),
                _tables: tables,
            });
        }
        let mut zero =
            be.host_alloc_pinned(programs.iter().map(|p| p.counter_bytes).max().unwrap_or(4))?;
        zero.as_mut_slice().fill(0);
        Ok(Self {
            route,
            programs,
            staging: MixedStepStaging::with_capacity(rows, span_capacity, batch),
            layout,
            host,
            zero,
            metadata,
            parked_base,
            ids_base,
            pos_base,
            kvlen_base,
            vocab: vocab.unwrap(),
            _buffers: buffers,
            module: guard.module.take().expect("armed above, taken once"),
            be: Arc::clone(be),
        })
    }
}

fn validate_program(program: &plow_asset::aux_program::Program) -> Result<()> {
    if program.gq_seg_ofs.len() != 2 {
        return Err(RuntimeError::Rejected(
            "mixed AMD requires one queue segment".into(),
        ));
    }
    if program
        .stream
        .iter()
        .chain(&program.gq_stream)
        .any(|entry| entry.flags != 0)
    {
        return Err(RuntimeError::Rejected(
            "mixed AMD requires coarse stream dependencies".into(),
        ));
    }
    for inst in &program.insts {
        if !matches!(
            DevOp::from_u16(inst.op),
            Some(
                DevOp::Embed
                    | DevOp::RmsNorm
                    | DevOp::HeadNormRope
                    | DevOp::Gemm
                    | DevOp::GemmGlu
                    | DevOp::FlashDecode
                    | DevOp::FlashPrefill
                    | DevOp::FlashMerge
                    | DevOp::Argmax
                    | DevOp::ArgmaxFin
                    | DevOp::SoftCap
                    | DevOp::NormResidual
            )
        ) {
            return Err(RuntimeError::Rejected(format!(
                "mixed AMD unsupported opcode {}",
                inst.op
            )));
        }
    }
    Ok(())
}

impl AmdEngine {
    fn step_slot(&mut self, route: StepRoute) -> &mut Option<MixedAmdStep> {
        match route {
            StepRoute::Mixed => &mut self.mixed_step,
            StepRoute::TokenBatch => &mut self.token_batch_step,
        }
    }

    fn step_ref(&self, route: StepRoute) -> Option<&MixedAmdStep> {
        match route {
            StepRoute::Mixed => self.mixed_step.as_ref(),
            StepRoute::TokenBatch => self.token_batch_step.as_ref(),
        }
    }

    pub fn mixed_step_rows(&self, decode_rows: usize, prefill_rows: usize) -> Option<u32> {
        self.packed_step_rows(StepRoute::Mixed, decode_rows, prefill_rows)
    }

    /// Row capacity of the token-batch bucket that would carry `leading_rows` sampled rows
    /// (decode requests plus prompts completing this step) and `prefill_rows` body rows.
    ///
    /// `leading_rows == 0` is refused, not treated as one: `S = 0` means no output segment,
    /// and the legacy `n_batch == 0 means one row` convention would deliver a token to a
    /// request that asked for none.
    pub fn token_batch_rows(&self, leading_rows: usize, prefill_rows: usize) -> Option<u32> {
        self.packed_step_rows(StepRoute::TokenBatch, leading_rows, prefill_rows)
    }

    fn packed_step_rows(
        &self,
        route: StepRoute,
        decode_rows: usize,
        prefill_rows: usize,
    ) -> Option<u32> {
        // v1 needs BOTH sides: with no prefill it is an ordinary decode step, and with no
        // decode its selection stage has nothing to read. The token batch needs only a
        // non-empty selection: a step of one completing prompt and nothing else is legal and
        // is exactly the concurrency-1 case v1 has to pay a second pass for.
        if decode_rows == 0 || (route == StepRoute::Mixed && prefill_rows == 0) {
            return None;
        }
        let step = self.step_ref(route)?;
        // ATTRIBUTION KNOB, not a tuning one. This route is confined to buckets with
        // `nsplit == 1`, which on a blob whose small rungs split their attention means it runs
        // a DIFFERENT bucket from the ordinary route for the same prompt — so a greedy output
        // difference has two candidate causes at once, the packing and the bucket. Pinning the
        // bucket holds the packing fixed and moves only the reduction order, which is what
        // separates them. Unset in every measured configuration.
        let pinned = std::env::var("PLOW_TOKEN_BATCH_ROWS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        step.programs
            .iter()
            .filter(|p| pinned.is_none_or(|rows| p.rows == rows))
            .filter(|p| {
                p.samples && decode_rows <= p.decode_rows as usize && p.rows as usize > decode_rows
            })
            .min_by_key(|p| {
                let capacity = p.rows as usize - decode_rows;
                (
                    prefill_rows.saturating_sub(capacity),
                    capacity.abs_diff(prefill_rows),
                )
            })
            .map(|p| p.rows)
    }

    pub fn mixed_step(
        &mut self,
        rows: u32,
        decode: &[DecodeRequest],
        prefill: &[PrefillRequest<'_>],
        frontiers: &mut [u32],
        output: &mut [u32],
    ) -> Result<()> {
        self.packed_step(StepRoute::Mixed, rows, decode, prefill, frontiers, output)
    }

    /// One unified token-batch step. `output` receives one sampled id per LEADING row, in
    /// plan order: the decode requests first, then the prompts that complete in this step.
    pub fn token_batch_step(
        &mut self,
        rows: u32,
        decode: &[DecodeRequest],
        prefill: &[PrefillRequest<'_>],
        frontiers: &mut [u32],
        output: &mut [u32],
    ) -> Result<()> {
        self.packed_step(
            StepRoute::TokenBatch,
            rows,
            decode,
            prefill,
            frontiers,
            output,
        )
    }

    fn packed_step(
        &mut self,
        route: StepRoute,
        rows: u32,
        decode: &[DecodeRequest],
        prefill: &[PrefillRequest<'_>],
        frontiers: &mut [u32],
        output: &mut [u32],
    ) -> Result<()> {
        let token_batch = route == StepRoute::TokenBatch;
        // Sampled rows: decode requests, plus the terminal row of every prompt this step
        // completes. On the mixed route the second group does not exist — its terminal tokens
        // are replayed through a separate decode-shaped pass instead.
        let leading = decode.len()
            + if token_batch {
                prefill
                    .iter()
                    .filter(|r| {
                        !r.tokens.is_empty()
                            && r.start.saturating_add(r.tokens.len() as u32) == r.prompt_len
                    })
                    .count()
            } else {
                0
            };
        let mut mixed = self.step_slot(route).take().ok_or_else(|| {
            RuntimeError::Rejected(format!("packet has no AMD {} program", route.label()))
        })?;
        let result = (|| -> Result<()> {
            if self.packed_prefill.is_some()
                || leading == 0
                || (!token_batch && (decode.is_empty() || prefill.is_empty()))
                || frontiers.len() != self.batch
                || output.len() != leading
                || decode.len().saturating_add(prefill.len()) > self.batch
            {
                return Err(RuntimeError::Rejected(format!(
                    "{} AMD request state mismatch",
                    route.label()
                )));
            }
            let program = mixed
                .programs
                .iter()
                .find(|p| p.rows == rows && p.samples && leading <= p.decode_rows as usize)
                .ok_or_else(|| {
                    RuntimeError::Rejected(format!(
                        "{} AMD missing sampled variant {rows}:{leading}",
                        route.label()
                    ))
                })?;
            let plan = mixed
                .staging
                .stage_cover(
                    decode,
                    prefill,
                    frontiers,
                    rows,
                    self.max_ctx as u32,
                    program.program_index,
                    route.cover(),
                )
                .map_err(|e| RuntimeError::Rejected(e.to_string()))?;
            if plan.decode_rows as usize != leading || plan.prefill_spans.len() > mixed.layout.spans
            {
                return Err(RuntimeError::Rejected(format!(
                    "{} AMD plan shape mismatch",
                    route.label()
                )));
            }
            if token_batch {
                crate::exec::amd_packed::validate_token_batch_rows(
                    program.program_index,
                    rows,
                    self.batch,
                    leading,
                    &plan.prefill_spans,
                    &plan.parked,
                )
                .map_err(RuntimeError::Rejected)?;
            } else {
                crate::exec::amd_packed::validate_rows(
                    program.program_index,
                    plan.decode_rows,
                    rows,
                    self.batch,
                    &plan.prefill_spans,
                    &plan.parked,
                )
                .map_err(RuntimeError::Rejected)?;
            }
            if plan.rows.iter().any(|row| row.token >= mixed.vocab) {
                return Err(RuntimeError::Rejected(format!(
                    "{} AMD token outside vocabulary",
                    route.label()
                )));
            }
            for &(slot, end) in &plan.mapped_ends {
                self.vmm_ensure(slot as usize, end)?;
            }
            self.kv_rebase(0)?;
            fill_words(
                &mixed.layout,
                bytemuck::cast_slice_mut(mixed.host.as_mut_slice()),
                plan,
            )?;
            // ONE completion wait for the whole descriptor, not seven. Four of
            // these slices are a word per row — 32 bytes at concurrency 8 — so
            // issued singly they are seven signal lifecycles and seven blocked
            // waits sitting in front of the launch, and none of it overlaps.
            // `memcpy_htod_pinned_batch` defaults to the same loop, so a backend
            // that cannot batch stays correct.
            let slice = |range: &std::ops::Range<usize>, words: usize| -> &[u8] {
                &mixed.host.as_slice()[range.start * 4..(range.start + words) * 4]
            };
            let n_spans = plan.prefill_spans.len();
            let uploads: [(u64, &[u8]); 7] = [
                (mixed.ids_base, slice(&mixed.layout.ids, rows as usize)),
                (mixed.pos_base, slice(&mixed.layout.pos, rows as usize)),
                (mixed.kvlen_base, slice(&mixed.layout.kvlen, rows as usize)),
                (
                    program.decode_slot,
                    slice(&mixed.layout.decode_slot, leading),
                ),
                (
                    mixed.metadata.base,
                    slice(&mixed.layout.prefill_spans, n_spans * SPAN_WORDS),
                ),
                (mixed.parked_base, slice(&mixed.layout.parked, rows as usize)),
                (
                    program.arg.counters,
                    &mixed.zero.as_slice()[..program.counter_bytes],
                ),
            ];
            self.be.memcpy_htod_pinned_batch(&uploads)?;
            let mut arg = program.arg;
            arg.n_prefill_spans = n_spans as u32;
            let started = Instant::now();
            self.be.launch(
                program.kernel,
                self.n_cu,
                program.block,
                0,
                as_bytes(std::slice::from_ref(&arg)),
            )?;
            self.seg_launches += 1;
            self.drain()?;
            self.seg_drain_us += started.elapsed().as_secs_f64() * 1e6;
            let start = mixed.layout.ids.start * 4;
            self.be.memcpy_dtoh_pinned(
                &mut mixed.host.as_mut_slice()[start..start + leading * 4],
                mixed.ids_base,
            )?;
            let tokens = bytemuck::cast_slice(&mixed.host.as_slice()[start..start + leading * 4]);
            mixed
                .staging
                .finish_after_device_success(frontiers, tokens, output)
                .map_err(|e| RuntimeError::Rejected(e.to_string()))
        })();
        if result.is_err() {
            let _ = self.drain();
            mixed.staging.discard();
        }
        *self.step_slot(route) = Some(mixed);
        result
    }
}

fn upload_program(
    be: &Arc<HsaBackend>,
    program: &plow_asset::aux_program::Program,
    tensor_table: u64,
    span_base: u64,
    parked_base: u64,
) -> Result<(DevProgram, usize, Vec<DeviceMem>)> {
    if program.gq_seg_ofs.len() != 2 {
        return Err(RuntimeError::Rejected(
            "mixed AMD program must have one queue segment".into(),
        ));
    }
    let upload = |bytes: &[u8]| -> Result<DeviceMem> {
        let mem = EngineDevice::alloc(&**be, bytes.len().max(4) as u64)?;
        if !bytes.is_empty() {
            EngineDevice::upload(&**be, &mem, 0, bytes)?;
        }
        Ok(mem)
    };
    let d_inst = upload(as_bytes(&program.insts))?;
    let d_stream = upload(as_bytes(&program.stream))?;
    let d_sofs = upload(as_bytes(&program.stream_ofs))?;
    let d_slen = upload(as_bytes(&program.stream_len))?;
    let d_waits = upload(as_bytes(&program.waits))?;
    let d_succs = upload(as_bytes(&program.succs))?;
    let d_gq_stream = upload(as_bytes(&program.gq_stream))?;
    let d_gq_seg = upload(as_bytes(&program.gq_seg_ofs))?;
    let counter_only = program.n_counter as usize * CTR_STRIDE_U32 * 4;
    let cursor_offset = counter_only.max(4);
    let counter_bytes = cursor_offset + CTR_STRIDE_U32 * 4;
    let d_counter = EngineDevice::alloc(&**be, counter_bytes as u64)?;
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
        n_gpu: 0,
        seg_ofs: 0,
        prefill_spans: span_base,
        prefill_parked: parked_base,
        n_prefill_spans: 0,
        n_prefill_rows: program.rows,
        token_batch: 0,
    };
    Ok((
        kernarg,
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
