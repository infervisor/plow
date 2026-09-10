use super::*;
use crate::exec::mixed_step_staging::TokenBatchStaging;
use plow_asset::token_batch::{Phase, Request, Selection};

pub(super) struct CudaTokenBatch {
    staging: TokenBatchStaging,
    fired: bool,
}

impl CudaTokenBatch {
    pub(super) fn load(e: &GpuEngine) -> Option<Self> {
        (RuntimeConfig::get().token_batch
            && !RuntimeConfig::get().fusion
            && e.be.compute_capability() == (9, 0)
            && e.packed_prefill.is_some()
            && e.has_packed_terminal()
            && e.recurrent.is_none()
            && e.mixed_step.is_none())
        .then(|| Self {
            staging: TokenBatchStaging::with_capacity(e.pf_max_rows(), e.batch),
            fired: false,
        })
    }
}

impl GpuEngine {
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
        output.clear();
        let mut state = self.token_batch.take().ok_or_else(|| {
            RuntimeError::Rejected("CUDA unified token-batch capability unavailable".into())
        })?;
        let result = (|| {
            if requests.iter().any(|request| {
                request.state_slot != request.slot
                    || request.selection != Selection::default()
                    || request
                        .tokens
                        .iter()
                        .any(|&token| token as usize >= self.vocab)
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
                        prompt_len: pending.new_frontier as usize,
                    }
                })
                .collect();
            let completed: smallvec::SmallVec<[_; 16]> = plan
                .pending
                .iter()
                .zip(&plan.phases)
                .filter(|(pending, phase)| pending.completes_prompt && **phase == Phase::Prefill)
                .map(|(pending, _)| pending.slot as usize)
                .collect();
            if self.vmm_prefix_enabled()
                && chunks
                    .iter()
                    .any(|chunk| self.seq_tokens[chunk.slot].len() != chunk.c0)
            {
                return Err(RuntimeError::Rejected(
                    "CUDA token-batch prefix history frontier mismatch".into(),
                ));
            }
            self.packed_token_body(&chunks)?;
            let mut terminal = self
                .packed_terminal
                .take()
                .expect("capability checked at load");
            let sampled = terminal
                .run_rows(self, &plan.sample_input_rows, plan.real_rows as usize)
                .and_then(|ids| {
                    state
                        .staging
                        .deliver(ids, output)
                        .map_err(|error| RuntimeError::Rejected(error.to_string()))
                });
            self.packed_terminal = Some(terminal);
            sampled?;
            state
                .staging
                .commit_after_device_success(&mut self.pos, &self.slot_generations)
                .map_err(|error| RuntimeError::Rejected(error.to_string()))?;
            if self.vmm_prefix_enabled() {
                for chunk in &chunks {
                    self.seq_tokens[chunk.slot].extend_from_slice(chunk.tokens);
                }
                for slot in completed {
                    self.vmm_publish(slot, self.pos[slot].saturating_sub(1));
                }
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
                prefill = requests.iter().filter(|r| r.phase == Phase::Prefill).count(),
                rows,
                samples = output.len(),
                "unified token batch committed"
            );
            Ok(())
        })();
        if result.is_err() {
            state.staging.discard();
            output.clear();
        }
        self.token_batch = Some(state);
        result
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
                terminal.run_rows(&e, &[(real - 1) as u32], real).unwrap();
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
