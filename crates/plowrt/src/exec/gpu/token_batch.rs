use super::*;
use crate::exec::mixed_step_staging::TokenBatchStaging;
use plow_asset::token_batch::{Phase, Request, Selection};

/// `PLOW_STEP_TIME=1` attribution for the UNIFIED TOKEN-BATCH route.
///
/// `StepTiming` (gpu.rs) instruments the `step_slots` path only, so a model that decodes
/// through this route — Gemma-4-26B-A4B does; the log says
/// `route="unified-token-batch" ... fires=true` — produces no timing at all under
/// `PLOW_STEP_TIME=1`, and the 26B campaign could not say whether its 48.8 ms TPOT was
/// host gap, submission, or device time. (Standalone benches put the MoE kernels at
/// 3.1 ms and all decode kernels at ~5.5 ms, so ~88 % was unattributed.)
///
/// Split reported: `enqueue` is `packed_token_body_enqueue` (host-side submission of the
/// interpreter launch, async) and `terminal` is the packed terminal, which must wait on the device
/// for the sampled ids — so `terminal` carries the device time the enqueue deferred.
mod steptime {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub(super) static STEPS: AtomicU64 = AtomicU64::new(0);
    pub(super) static ENQUEUE_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static TERMINAL_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static ROWS: AtomicU64 = AtomicU64::new(0);

    pub(super) fn add(enqueue_ns: u64, terminal_ns: u64, rows: u64) {
        STEPS.fetch_add(1, Ordering::Relaxed);
        ENQUEUE_NS.fetch_add(enqueue_ns, Ordering::Relaxed);
        TERMINAL_NS.fetch_add(terminal_ns, Ordering::Relaxed);
        ROWS.fetch_add(rows, Ordering::Relaxed);
    }

    /// Log a rolling mean every `n` steps. Means, not last-step values: a single decode
    /// step is noisy and the campaign protocol compares medians of many.
    pub(super) fn log_every(n: u64) {
        let steps = STEPS.load(Ordering::Relaxed);
        if steps == 0 || steps % n != 0 {
            return;
        }
        let per = |v: u64| v as f64 / steps as f64 / 1e6; // ns total -> ms mean
        let enq = per(ENQUEUE_NS.load(Ordering::Relaxed));
        let term = per(TERMINAL_NS.load(Ordering::Relaxed));
        tracing::info!(
            steps,
            rows_total = ROWS.load(Ordering::Relaxed),
            enqueue_ms = enq,
            terminal_ms = term,
            step_ms = enq + term,
            "token-batch step means (host clocks; terminal includes the device wait)"
        );
    }
}

pub(super) struct CudaTokenBatch {
    staging: TokenBatchStaging,
    fired: bool,
}

impl CudaTokenBatch {
    pub(super) fn load(e: &GpuEngine) -> Option<Self> {
        (!RuntimeConfig::get().fusion
            && e.be.compute_capability() == (9, 0)
            && e.packed_prefill.is_some()
            && e.has_packed_terminal()
            // A recurrent chain packs only through its per-request state gather / varlen scan /
            // scatter, so a one-row decode request is a continuing chunk of length 1; the split
            // attention riders never arm on it (`riders.rs`), the rows attend through the
            // prefill attention's request table.
            && (e.recurrent.is_none() || e.recurrent_packs)
            && e.mixed_step.is_none()
            // Host rows spliced by launch row (overlay) or a decode-only embedding (position
            // base): the batch embeds decode rows through the prefill program, in its own order.
            && e.handle_of("in.encoder_overlay").is_none()
            && e.handle_of("in.pos_base").is_none()
            // Decode rows ride as one-row requests of the packed prefill attention, which is
            // qualified only on the hd256/512 attention arms. At hd128 (Veena, H100, c64) its
            // ticks carried ~77 one-row requests and served 27.1 audio s/s against 33.8 for
            // packed prefill plus the ordinary decode launch.
            && e.prefill.iter().all(|b| {
                b.flash_sites.iter().all(|&pc| matches!(b.h_inst[pc].i[6], 256 | 512))
            }))
        .then(|| Self {
            staging: TokenBatchStaging::with_capacity(e.pf_max_rows(), e.batch),
            fired: false,
        })
    }
}

/// `PLOW_PF_PACKLOG=1`: one `PACKLOG STEP` line per unified token-batch launch. `enq` is the host
/// time from the first enqueue to the id read-back's enqueue, `window` the host work run while
/// the launch executes, `wait` the remaining block on the device, `post` commit + publication;
/// `gpu` is the device span (events around the launch), so `enq + window + wait + post - gpu`
/// is the time the device sat idle inside this step.
struct StepClock {
    t0: std::time::Instant,
    enq_us: u64,
    window_us: u64,
    wait_us: u64,
}

impl StepClock {
    fn start(e: &mut GpuEngine) -> Option<Self> {
        if !RuntimeConfig::get().pf_packlog {
            return None;
        }
        if e.step_events.is_none() {
            e.step_events = Some((e.be.event_create(true).ok()?, e.be.event_create(true).ok()?));
        }
        let (start, _) = e.step_events.as_ref()?;
        e.be.event_record(start, &e.stream).ok()?;
        Some(Self {
            t0: std::time::Instant::now(),
            enq_us: 0,
            window_us: 0,
            wait_us: 0,
        })
    }

    fn end(&mut self, e: &GpuEngine) {
        if let Some((_, end)) = e.step_events.as_ref() {
            let _ = e.be.event_record(end, &e.stream);
        }
        self.enq_us = self.t0.elapsed().as_micros() as u64;
    }

    fn log(self, e: &GpuEngine, rows: usize, bucket: u32) {
        let total = self.t0.elapsed().as_micros() as u64;
        let gpu_us = e
            .step_events
            .as_ref()
            .and_then(|(start, end)| e.be.event_elapsed_ms(start, end).ok())
            .map_or(0, |ms| (ms * 1e3) as u64);
        eprintln!(
            "PACKLOG STEP rows={rows} bucket={bucket} enq_us={} window_us={} wait_us={} post_us={} gpu_us={gpu_us}",
            self.enq_us,
            self.window_us,
            self.wait_us,
            total.saturating_sub(self.enq_us + self.window_us + self.wait_us),
        );
    }
}

impl GpuEngine {
    /// `(slot, drafts, verdict)` of every verify span the token batches since the last call
    /// committed.
    pub fn take_token_batch_verdicts(
        &mut self,
        out: &mut Vec<(usize, u32, crate::orch::speculative::SpecVerdict)>,
    ) {
        out.clear();
        out.append(&mut self.tb_verdicts);
    }

    pub fn token_batch_enabled(&self) -> bool {
        self.token_batch.is_some()
    }

    pub fn slot_generation(&self, slot: usize) -> Option<u32> {
        self.slot_generations.get(slot).copied()
    }

    pub fn token_batch_step(
        &mut self,
        requests: &[Request<'_>],
        output: &mut Vec<(u32, u32)>,
    ) -> Result<()> {
        let r = self.token_batch_step_inner(requests, output, None, None);
        self.note_step_result(r.is_err());
        r
    }

    /// As [`Self::token_batch_step`], with `spec(slot)` rows drawn on the device by `plow_sample`
    /// from their logits rows instead of returning the argmax. A row whose spec is `None` keeps
    /// the argmax; with no sampler loaded every row does.
    pub fn token_batch_step_sampled(
        &mut self,
        requests: &[Request<'_>],
        output: &mut Vec<(u32, u32)>,
        spec: &dyn Fn(u32, u32) -> Option<DevSample>,
    ) -> Result<()> {
        let r = self.token_batch_step_inner(requests, output, Some(spec), None);
        self.note_step_result(r.is_err());
        r
    }

    /// [`Self::token_batch_step`] (or `_sampled` with `spec`) that runs `window` on the host
    /// while the launch executes: after the body, terminal and id read-back are enqueued and
    /// before the wait. `window` may admit slots this launch does not touch
    /// ([`Self::admit_packed_slot`] defers their snapshot waits to the step's own drain).
    pub fn token_batch_step_window(
        &mut self,
        requests: &[Request<'_>],
        output: &mut Vec<(u32, u32)>,
        spec: Option<&dyn Fn(u32, u32) -> Option<DevSample>>,
        window: &mut dyn FnMut(&mut GpuEngine),
    ) -> Result<()> {
        let r = self.token_batch_step_inner(requests, output, spec, Some(window));
        self.note_step_result(r.is_err());
        r
    }

    fn token_batch_step_inner(
        &mut self,
        requests: &[Request<'_>],
        output: &mut Vec<(u32, u32)>,
        spec: Option<&dyn Fn(u32, u32) -> Option<DevSample>>,
        mut window: Option<&mut dyn FnMut(&mut GpuEngine)>,
    ) -> Result<()> {
        output.clear();
        self.tb_launched = false;
        let mut state = self.token_batch.take().ok_or_else(|| {
            RuntimeError::Rejected("CUDA unified token-batch capability unavailable".into())
        })?;
        let mut body_enqueued = false;
        let verdicts_before = self.tb_verdicts.len();
        // Multimodal soft-token rows (bit 31): the prefill buckets this batch runs replace them
        // from the slab (MmRowsBf16).
        let mm_rows = self
            .handle_of(plow_asset::multimodal::TABLE_TENSOR)
            .is_some();
        let result = (|| {
            // A verify span's rejected drafts roll back by frontier only: a recurrent state the
            // scan advanced past them cannot be taken back (`spec_enable` refuses such packets).
            if self.recurrent.is_some() && requests.iter().any(|r| r.phase == Phase::Verify) {
                return Err(RuntimeError::Rejected(
                    "CUDA token batch cannot verify on a recurrent chain".into(),
                ));
            }
            // Only a prefill chunk reads an attached state from its block; a decode or verify row
            // reads the slot.
            for request in requests.iter().filter(|r| r.phase != Phase::Prefill) {
                self.state_materialize(request.slot as usize, None)?;
            }
            if requests.iter().any(|request| {
                request.state_slot != request.slot
                    || request.selection != Selection::default()
                    || request.tokens.iter().any(|&token| {
                        token as usize >= self.vocab
                            && !(mm_rows && token & plow_asset::multimodal::ROW_ID_BIT != 0)
                    })
            }) {
                return Err(RuntimeError::Rejected(
                    "CUDA token batch requires direct slots, valid tokens and greedy selection"
                        .into(),
                ));
            }
            let rows = requests.iter().try_fold(0usize, |rows, request| {
                rows.checked_add(request.tokens.len())
                    .ok_or_else(|| RuntimeError::Rejected("CUDA token-batch row overflow".into()))
            })?;
            let (bucket, capacity) = self
                .prefill
                .iter()
                .enumerate()
                .find(|(_, bucket)| bucket.t as usize >= rows)
                .map(|(index, bucket)| (index, bucket.t))
                .ok_or_else(|| {
                    RuntimeError::Rejected("CUDA token-batch capacity exceeded".into())
                })?;
            let plan = state
                .staging
                .stage(
                    requests,
                    &self.pos,
                    &self.slot_generations,
                    capacity,
                    self.max_ctx as u32,
                    bucket as u32,
                )
                .map_err(|error| RuntimeError::Rejected(error.to_string()))?;
            if plan.sample_rows as usize > self.batch {
                return Err(RuntimeError::Rejected(
                    "CUDA token-batch output capacity exceeded".into(),
                ));
            }
            let chunks: smallvec::SmallVec<[_; 16]> = plan
                .pending
                .iter()
                .map(|pending| {
                    let request = requests
                        .iter()
                        .find(|r| r.id == pending.request)
                        .expect("planner retains request identity");
                    PackedTokenReq {
                        slot: pending.slot as usize,
                        tokens: request.tokens,
                        c0: pending.expected_frontier as usize,
                        // The chunk's own end bounds decode rows (their end is past the
                        // prompt); a prefill chunk carries the real prompt length so the
                        // KV pre-map hint can look one bucket ahead of this chunk.
                        prompt_len: (pending.new_frontier as usize)
                            .max(request.prompt_len as usize),
                    }
                })
                .collect();
            // Owned copies of what this step needs after the plan's borrow ends.
            let sample_rows: smallvec::SmallVec<[u32; 32]> =
                plan.sample_input_rows.iter().copied().collect();
            let real_rows = plan.real_rows as usize;
            // Verify spans (`Phase::Verify`, planned last): `(slot, first sample, rows, frontier)`.
            let verifies: smallvec::SmallVec<[(usize, u32, u32, u32); 8]> = plan
                .pending
                .iter()
                .zip(&plan.phases)
                .filter(|(_, phase)| **phase == Phase::Verify)
                .filter_map(|(p, _)| {
                    Some((
                        p.slot as usize,
                        p.sample_index?,
                        p.new_frontier - p.expected_frontier,
                        p.expected_frontier,
                    ))
                })
                .collect();
            let specs: Option<smallvec::SmallVec<[DevSample; 32]>> =
                spec.filter(|_| self.sampler.is_some()).and_then(|spec| {
                    // Row `j` of a verify span takes plain decode's draw for its output position.
                    let plain = verifies.first().map_or(usize::MAX, |v| v.1 as usize);
                    let specs: smallvec::SmallVec<[DevSample; 32]> = plan
                        .sample_owners
                        .iter()
                        .enumerate()
                        .map(|(i, owner)| {
                            let j = verifies
                                .iter()
                                .find(|v| v.0 == owner.slot as usize)
                                .filter(|_| i >= plain)
                                .map_or(0, |v| i as u32 - v.1);
                            spec(owner.slot, j).unwrap_or_else(DevSample::greedy)
                        })
                        .collect();
                    specs.iter().any(|s| s.temp > 0.0).then_some(specs)
                });
            let spec_slots: smallvec::SmallVec<[u32; 32]> =
                plan.sample_owners.iter().map(|owner| owner.slot).collect();
            let completed: smallvec::SmallVec<[_; 16]> = plan
                .pending
                .iter()
                .zip(&plan.phases)
                .filter(|(pending, phase)| pending.completes_prompt && **phase == Phase::Prefill)
                .map(|(pending, _)| pending.slot as usize)
                .collect();
            // Prefill members with their prompt length: a recurrent model's chunk that ends at
            // the prompt-end boundary publishes it right after the step (`state_capture_publish`).
            let prefill_members: smallvec::SmallVec<[(usize, usize); 16]> = plan
                .pending
                .iter()
                .zip(&plan.phases)
                .filter(|(_, phase)| **phase == Phase::Prefill)
                .filter_map(|(pending, _)| {
                    let request = requests.iter().find(|r| r.id == pending.request)?;
                    Some((pending.slot as usize, request.prompt_len as usize))
                })
                .collect();
            // A request waiting on this prompt's shared blocks attaches as soon as the chunk
            // that computed them lands, not after the owner's whole prompt.
            let chunk_ends: smallvec::SmallVec<[(usize, u32); 16]> = match RuntimeConfig::get()
                .prefix_chunk_publish()
                .then(|| {
                    self.vmm
                        .as_ref()
                        .and_then(|v| v.kv.as_ref())
                        .map(|kv| kv.block_rows())
                })
                .flatten()
            {
                Some(br) => plan
                    .pending
                    .iter()
                    .zip(&plan.phases)
                    .filter(|(pending, phase)| {
                        **phase == Phase::Prefill
                            && !pending.completes_prompt
                            && pending.new_frontier / br > pending.expected_frontier / br
                    })
                    .map(|(pending, _)| (pending.slot as usize, pending.new_frontier / br * br))
                    .collect(),
                None => smallvec::SmallVec::new(),
            };
            if self.vmm_prefix_enabled()
                && chunks
                    .iter()
                    .any(|chunk| self.seq_tokens[chunk.slot].len() != chunk.c0)
            {
                return Err(RuntimeError::Rejected(
                    "CUDA token-batch prefix history frontier mismatch".into(),
                ));
            }
            // PLOW_STEP_TIME=1 only: `Instant::now` is a syscall-class read, so it stays
            // behind the flag rather than running on every decode step.
            let timed = RuntimeConfig::get().nv.step_time;
            let t_enq = timed.then(std::time::Instant::now);
            let mut gpu = StepClock::start(self);
            // KV-shared tail: the body stops at the tail boundary and the tail runs the sampled
            // rows alone, which the terminal then finds at rows 0..n.
            let tail_rows = self.shared_tail_rows(&chunks, &sample_rows);
            self.pf_seg_prefix = tail_rows.is_some();
            // Leading decode rows ride as split-attention riders (`riders.rs`); a final prefill
            // chunk of one row is not one, whatever its length.
            self.pf_riders = plan
                .phases
                .iter()
                .zip(&chunks)
                .take_while(|(phase, chunk)| **phase == Phase::Decode && chunk.tokens.len() == 1)
                .count();
            let body = self.packed_token_body_enqueue(&chunks);
            self.pf_riders = 0;
            self.pf_seg_prefix = false;
            body?;
            body_enqueued = true;
            let (sample_rows, real_rows) = match &tail_rows {
                Some(rows) => {
                    self.shared_tail_enqueue(rows, real_rows)?;
                    ((0..rows.len() as u32).collect(), rows.len())
                }
                None => (sample_rows, real_rows),
            };
            let enqueue_ns = t_enq.map_or(0, |t| t.elapsed().as_nanos() as u64);
            let mut terminal = self
                .packed_terminal
                .take()
                .expect("capability checked at load");
            let t_term = timed.then(std::time::Instant::now);
            let snap = tail_rows.is_none() && self.spec_draft.is_some();
            let sampled = terminal
                .begin_rows(
                    self,
                    &sample_rows,
                    real_rows,
                    specs.as_deref().map(|s| (s, spec_slots.as_slice())),
                )
                .and_then(|i| {
                    if snap {
                        // The drafter's next input: each sample row's final-norm hidden.
                        let (src, pitch) = terminal.hidden_rows();
                        let owners = state.staging.pending_plan().map(|p| &p.sample_owners);
                        let plain = verifies.first().map_or(usize::MAX, |v| v.1 as usize);
                        self.spec_hidden_save_rows(
                            src,
                            pitch,
                            owners
                                .into_iter()
                                .flatten()
                                .take(plain)
                                .enumerate()
                                .map(|(row, o)| (o.slot as usize, row as u32)),
                        )?;
                    }
                    if let Some(gpu) = gpu.as_mut() {
                        gpu.end(self);
                    }
                    if let Some(window) = window.as_mut() {
                        if self.window_event.is_none() {
                            self.window_event = self.be.event_create(false).ok();
                        }
                        if let Some(done) = &self.window_event {
                            let _ = self.be.event_record(done, &self.stream);
                        }
                        let t = gpu.as_ref().map(|_| std::time::Instant::now());
                        self.window_open = true;
                        window(self);
                        self.window_open = false;
                        if let (Some(gpu), Some(t)) = (gpu.as_mut(), t) {
                            gpu.window_us = t.elapsed().as_micros() as u64;
                        }
                    }
                    let t = gpu.as_ref().map(|_| std::time::Instant::now());
                    let ids = terminal.finish_rows(self, i);
                    if let (Some(gpu), Some(t)) = (gpu.as_mut(), t) {
                        gpu.wait_us = t.elapsed().as_micros() as u64;
                    }
                    ids
                })
                .and_then(|ids| {
                    state
                        .staging
                        .deliver(ids, output)
                        .map_err(|error| RuntimeError::Rejected(error.to_string()))
                });
            let terminal_ns = t_term.map_or(0, |t| t.elapsed().as_nanos() as u64);
            let verdicts = sampled.and_then(|()| {
                if verifies.is_empty() {
                    return Ok(0);
                }
                // Accept each verify span from its sampled rows and keep only its emitted
                // inputs: rows past them are dead, their KV never read.
                let stop = Arc::clone(self.stop_ids());
                let plain = verifies[0].1 as usize;
                for &(slot, first, n, start) in &verifies {
                    let tokens = requests
                        .iter()
                        .find(|r| r.phase == Phase::Verify && r.slot as usize == slot)
                        .expect("planned")
                        .tokens;
                    let ids: smallvec::SmallVec<[u32; 8]> = output
                        [first as usize..(first + n) as usize]
                        .iter()
                        .map(|&(_, id)| id)
                        .collect();
                    let v = crate::orch::speculative::accept_greedy(
                        tokens,
                        &ids,
                        first,
                        self.slot_generations[slot],
                        &stop,
                    );
                    state
                        .staging
                        .shorten(slot as u32, start + u32::from(v.n_emit))
                        .map_err(|error| RuntimeError::Rejected(error.to_string()))?;
                    self.kv_hw[slot] = self.kv_hw[slot].max(start + n);
                    self.tb_verdicts.push((slot, n - 1, v));
                }
                output.truncate(plain);
                let (src, pitch) = terminal.hidden_rows();
                let rows: smallvec::SmallVec<[(usize, u32); 8]> = self.tb_verdicts
                    [self.tb_verdicts.len() - verifies.len()..]
                    .iter()
                    .map(|(slot, _, v)| (*slot, v.hrow))
                    .collect();
                if tail_rows.is_none() {
                    self.spec_hidden_save_rows(src, pitch, rows.into_iter())?;
                }
                Ok(verifies.len())
            });
            self.packed_terminal = Some(terminal);
            verdicts?;
            if timed {
                steptime::add(enqueue_ns, terminal_ns, rows as u64);
                steptime::log_every(8);
            }
            state
                .staging
                .commit_after_device_success(&mut self.pos, &self.slot_generations)
                .map_err(|error| RuntimeError::Rejected(error.to_string()))?;
            self.spec_hidden_commit(true);
            if self.vmm_prefix_enabled() {
                for chunk in &chunks {
                    let n = (self.pos[chunk.slot] as usize - chunk.c0).min(chunk.tokens.len());
                    self.seq_tokens[chunk.slot].extend_from_slice(&chunk.tokens[..n]);
                }
                for &(slot, prompt_len) in &prefill_members {
                    self.state_capture_publish(slot, prompt_len);
                }
                for chunk in &chunks {
                    self.vmm_publish_shared(chunk.slot, chunk.c0 as u32);
                }
                for slot in completed {
                    self.vmm_publish(slot, self.pos[slot].saturating_sub(1));
                    self.vmm_prefill_done(slot);
                }
                for (slot, rows) in chunk_ends {
                    self.vmm_publish(slot, rows);
                }
            }
            if let Some(gpu) = gpu {
                gpu.log(self, rows, capacity);
            }
            if !state.fired {
                tracing::info!(
                    route = "unified-token-batch",
                    backend = "cuda",
                    ready = true,
                    fires = true,
                    requests = requests.len(),
                    rows,
                    "token-batch route fired"
                );
                state.fired = true;
            }
            tracing::debug!(
                decode = requests.iter().filter(|r| r.phase == Phase::Decode).count(),
                prefill = requests
                    .iter()
                    .filter(|r| r.phase == Phase::Prefill)
                    .count(),
                rows,
                samples = output.len(),
                "unified token batch committed"
            );
            Ok(())
        })();
        if result.is_err() {
            self.tb_verdicts.truncate(verdicts_before);
            self.spec_hidden_commit(false);
            if body_enqueued {
                let _ = self.be.stream_synchronize(&self.stream);
            }
            state.staging.discard();
            output.clear();
        }
        self.token_batch = Some(state);
        let flushed = self.flush_deferred_attach();
        result.and(flushed)
    }

    /// Whether the last token batch stopped before its first device launch: no kernel touched
    /// a slot's KV or recurrent state and no frontier moved. Only a host refusal
    /// (`RuntimeError::Rejected`) from such a step may run its rows another way.
    pub fn token_batch_refused_before_launch(&self) -> bool {
        !self.tb_launched
    }

    /// Inside a launch window: whether the launch is still running on the device.
    pub(crate) fn launch_in_flight(&self) -> bool {
        self.window_open
            && self
                .window_event
                .as_ref()
                .is_some_and(|done| !self.be.event_query(done).unwrap_or(true))
    }

    /// Complete the restores a launch window deferred: drain the stream, then release their
    /// snapshot holds. A failed drain keeps the holds: the copies' completion is unproven.
    pub(crate) fn flush_deferred_attach(&mut self) -> Result<()> {
        self.blocks_reap_seals();
        if self.deferred_attach.is_empty() {
            return Ok(());
        }
        self.be.stream_synchronize(&self.stream)?;
        let kv = self.vmm.as_ref().and_then(|v| v.kv.as_ref());
        let slots: Vec<usize> = self.deferred_attach.drain(..).collect();
        for b in slots {
            self.blocks_unhold(b);
            if let Some(kv) = kv {
                kv.finish_attach(b);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request<'a>(
        e: &GpuEngine,
        id: u32,
        slot: usize,
        phase: Phase,
        tokens: &'a [u32],
        prompt_len: usize,
    ) -> Request<'a> {
        Request {
            id,
            slot: slot as u32,
            state_slot: slot as u32,
            generation: e.slot_generation(slot).unwrap(),
            phase,
            tokens,
            prompt_len: prompt_len as u32,
            selection: Selection::default(),
        }
    }

    #[test]
    #[ignore = "requires H100 packed-prefix assets and natural prompt reference metadata"]
    fn unified_cuda_rows_match_isolated_suffixes_and_commit_once() {
        check_unified_rows(false, false);
    }

    #[test]
    #[ignore = "requires H100 packed-prefix assets and natural prompt reference metadata"]
    fn unified_cuda_aligned_prompts_reuse_prefixes() {
        check_unified_rows(true, false);
    }

    #[test]
    #[ignore = "requires H100 FP8-KV packed-prefix assets and natural prompt reference metadata"]
    fn unified_cuda_fp8_kv_rows_and_prefixes_match_isolated() {
        check_unified_rows(true, true);
    }

    fn check_unified_rows(aligned: bool, require_fp8_kv: bool) {
        let assets = PathBuf::from(std::env::var("CUDA_TOKEN_BATCH_TEST_ASSETS").unwrap());
        if require_fp8_kv {
            let bytes = std::fs::read(assets.join("model.pkt")).unwrap();
            let blob = DevBlob::parse(&bytes).unwrap();
            let live = crate::memory::vmm::LiveKvLayout::manifest(&blob, &bytes)
                .unwrap()
                .unwrap();
            assert_eq!(live.version, 2);
            assert!(live.caches.iter().all(|c| c.scales.is_some()));
            assert_eq!(
                blob.decode_progs().iter().map(|p| p.t).collect::<Vec<_>>(),
                [1, 2, 4, 8, 16]
            );
            assert_eq!(
                blob.prefill_progs().iter().map(|p| p.t).collect::<Vec<_>>(),
                [128, 512, 1024]
            );
        }
        let reference = PathBuf::from(std::env::var("CUDA_TOKEN_BATCH_TEST_REFERENCE").unwrap());
        let mut prompts: Vec<Vec<u32>> = (0..2)
            .map(|i| {
                let record: serde_json::Value = serde_json::from_slice(
                    &std::fs::read(reference.join(format!("{i}.json"))).unwrap(),
                )
                .unwrap();
                serde_json::from_value(record["prompt_ids"].clone()).unwrap()
            })
            .collect();
        if aligned {
            prompts[0].truncate(1024);
            prompts[1].truncate(16384);
        }
        let mut e = GpuEngine::load(
            Arc::new(CudaBackend::new(0).unwrap()),
            &assets,
            &assets.join("checkpoint"),
        )
        .unwrap();
        // Bit-exact against isolated prefill holds only when decode rows run prefill attention;
        // split-attention riders are gated against the FP32 reference instead.
        e.riders = None;
        assert!(e.token_batch_enabled() && e.batch >= 8 && e.vmm_prefix_enabled());
        let slots = [e.batch - 1, e.batch / 2 - 1];
        let mut histories = prompts.clone();
        let mut next = Vec::new();
        for i in 0..2 {
            e.begin_slot(slots[i], prompts[i].len() + 32).unwrap();
            next.push(e.prefill_slot(slots[i], &prompts[i]).unwrap());
        }
        e.retire_slot(slots[1], false);
        e.begin_slot(slots[1], prompts[1].len() + 32).unwrap();
        let cached = e.attach_prompt(slots[1], &prompts[1]).unwrap();
        assert_eq!(cached, (prompts[1].len() - 1) / 32 * 32);
        let mut output = Vec::new();
        let mut logits = Vec::new();
        let mut ordinary_tail_max_abs = 0.0f32;
        let mut ordinary_tail_changed_frames = 0;
        for step in 0..16 {
            histories[0].push(next[0]);
            if step > 0 {
                histories[1].push(next[1]);
            }
            let frontiers = [e.pos[slots[0]] as usize, e.pos[slots[1]] as usize];
            let mut expected = Vec::new();
            for i in 0..2 {
                let before = e.pos[slots[i]];
                e.run_one_prefill_chunk(
                    e.f_pf.unwrap(),
                    slots[i],
                    &histories[i],
                    frontiers[i],
                    128,
                )
                .unwrap();
                assert_eq!(e.pos[slots[i]], before);
                e.logits_row(0, &mut logits).unwrap();
                let ordinary = logits.clone();
                let real = histories[i].len() - frontiers[i];
                // Match the compact tail's RMSNorm reduction. Ordinary prefill uses
                // the >=32-row reduction, which can differ by a BF16 rounding bit.
                let mut terminal = e.packed_terminal.take().unwrap();
                let i = terminal
                    .begin_rows(&mut e, &[(real - 1) as u32], real, None)
                    .unwrap();
                terminal.finish_rows(&e, i).unwrap();
                e.packed_terminal = Some(terminal);
                e.logits_row(0, &mut logits).unwrap();
                let delta = logits
                    .iter()
                    .zip(&ordinary)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                ordinary_tail_max_abs = ordinary_tail_max_abs.max(delta);
                ordinary_tail_changed_frames += usize::from(delta > 0.0);
                expected.push(logits.clone());
            }
            let requests = [
                request(
                    &e,
                    91,
                    slots[1],
                    if step == 0 {
                        Phase::Prefill
                    } else {
                        Phase::Decode
                    },
                    &histories[1][frontiers[1]..],
                    prompts[1].len(),
                ),
                request(
                    &e,
                    37,
                    slots[0],
                    Phase::Decode,
                    &histories[0][frontiers[0]..],
                    prompts[0].len(),
                ),
            ];
            e.token_batch_step(&requests, &mut output).unwrap();
            assert_eq!(
                output.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
                if step == 0 { [37, 91] } else { [91, 37] }
            );
            for (row, &(id, token)) in output.iter().enumerate() {
                let i = if id == 37 { 0 } else { 1 };
                e.logits_row(row, &mut logits).unwrap();
                assert_eq!(logits.len(), e.vocab);
                assert!(logits.iter().all(|v| v.is_finite()));
                let mismatches = logits
                    .iter()
                    .zip(&expected[i])
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                let max_abs = logits
                    .iter()
                    .zip(&expected[i])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    logits
                        .iter()
                        .zip(&expected[i])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "isolated suffix mismatch, step {step}, logical request {i}, frontier {}, mismatches {mismatches}, max_abs {max_abs}", frontiers[i]
                );
                assert_eq!(e.pos[slots[i]] as usize, histories[i].len());
                assert_eq!(e.seq_tokens[slots[i]], histories[i]);
                next[i] = token;
            }
        }
        eprintln!("PASS unified CUDA: 32 full-vocabulary frames, cached prefill/decode, sparse reversed owners, one commit per input");
        eprintln!("Ordinary versus compact isolated output tail: {ordinary_tail_changed_frames}/32 changed frames, max_abs {ordinary_tail_max_abs}");

        let slot = 0;
        e.begin_slot(slot, 257).unwrap();
        let prompt = &prompts[0][..256];
        let first = request(&e, 123, slot, Phase::Prefill, &prompt[..64], prompt.len());
        e.token_batch_step(&[first], &mut output).unwrap();
        assert!(output.is_empty());
        assert_eq!(e.pos[slot], 64);
        assert_eq!(e.seq_tokens[slot], prompt[..64]);
        let finish = request(&e, 123, slot, Phase::Prefill, &prompt[64..], prompt.len());
        e.token_batch_step(&[finish], &mut output).unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].0, 123);
        assert_eq!(e.pos[slot], 256);
        e.begin_slot(1, 257).unwrap();
        assert_eq!(e.attach_prompt(1, prompt).unwrap(), 224);
        e.retire_slot(1, false);

        let token = [output[0].1];
        let stale = request(&e, 123, slot, Phase::Decode, &token, 256);
        e.retire_slot(slot, false);
        e.begin_slot(slot, 257).unwrap();
        let before = e.pos.clone();
        assert!(e.token_batch_step(&[stale], &mut output).is_err());
        assert!(e.token_batch_refused_before_launch());
        assert!(output.is_empty());
        assert_eq!(e.pos, before);
        let duplicate = request(&e, 123, slot, Phase::Prefill, &token, 1);
        assert!(e
            .token_batch_step(&[duplicate, duplicate], &mut output)
            .is_err());
        assert_eq!(e.pos, before);
        eprintln!("PASS unified CUDA: zero-output intermediate, terminal without replay, stale generation and duplicate refusal");
    }
}
