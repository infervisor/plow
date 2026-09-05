//! The CPU engine behind a served slug: one sequence slot, whole-prompt
//! prefill, greedy on-device argmax — the same shape as the single-GPU AMD
//! engine, so it rides the mux's [`SeqEngine`] tick unchanged.

use std::path::Path;
use std::sync::Arc;

use packet::dev::PrefillSpan;

use super::engine::SeqEngine;
use crate::exec::cpu::engine::{CpuEngine, CpuEngineOpts};
use crate::{Result, RuntimeError};

pub struct CpuServe {
    eng: CpuEngine,
    stop_ids: Arc<Vec<u32>>,
    decode_rungs: Box<[u32]>,
    /// KV rows written for slot 0; the next token embeds at this position.
    pos: u32,
    live: bool,
    max_ctx: usize,
}

impl CpuServe {
    pub fn load(blob: &Path, checkpoint: &Path, opts: &CpuEngineOpts) -> Result<Self> {
        let mut ids = crate::asset::checkpoint::read_eos_ids(checkpoint);
        ids.extend(crate::asset::checkpoint::chat_stop_ids(checkpoint, &ids));
        let eng = CpuEngine::load(blob, checkpoint, opts)?;
        let max_ctx = eng.max_ctx();
        tracing::info!(
            max_ctx,
            threads = eng.threads,
            isa = ?eng.isa,
            stop_ids = ?ids,
            "CPU serve engine ready"
        );
        Ok(CpuServe {
            eng,
            stop_ids: Arc::new(ids),
            decode_rungs: Box::new([1]),
            pos: 0,
            live: false,
            max_ctx,
        })
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn decode_rungs(&self) -> &[u32] {
        &self.decode_rungs
    }

    pub fn engine(&self) -> &CpuEngine {
        &self.eng
    }

    fn check_slot(&self, slot: usize) -> Result<()> {
        if slot != 0 {
            return Err(RuntimeError::Rejected(format!(
                "slot {slot} past engine batch 1"
            )));
        }
        Ok(())
    }

    /// Whole-prompt prefill into slot 0; returns the first generated token.
    pub fn prefill(&mut self, slot: usize, prompt: &[u32]) -> Result<u32> {
        self.check_slot(slot)?;
        if prompt.is_empty() {
            return Err(RuntimeError::Rejected("empty prompt".into()));
        }
        if prompt.len() >= self.max_ctx {
            return Err(RuntimeError::Rejected(format!(
                "prompt is {} tokens, max_ctx is {}",
                prompt.len(),
                self.max_ctx
            )));
        }
        let tok = self.eng.prefill(prompt)?;
        self.pos = prompt.len() as u32;
        self.live = true;
        Ok(tok)
    }

    /// Embed `id` at the slot's position and return the greedy next token.
    pub fn step(&mut self, slot: usize, id: u32) -> Result<u32> {
        self.check_slot(slot)?;
        if !self.live {
            return Err(RuntimeError::Rejected("step on a slot with no prefill".into()));
        }
        if self.pos as usize >= self.max_ctx {
            return Err(RuntimeError::Rejected(format!(
                "position {} past max_ctx {}",
                self.pos, self.max_ctx
            )));
        }
        // The mux feeds the token it streamed; the engine left its own sample in
        // `in.ids[0]`, which is the same id in the greedy case — write it anyway.
        self.eng.set_token(id)?;
        let tok = self.eng.decode_step(self.pos, self.pos + 1)?;
        self.pos += 1;
        Ok(tok)
    }

    pub fn release(&mut self, slot: usize) {
        if slot == 0 {
            self.live = false;
            self.pos = 0;
        }
    }
}

impl SeqEngine for CpuServe {
    fn stop_ids(&self) -> &Arc<Vec<u32>> {
        &self.stop_ids
    }

    fn batch(&self) -> usize {
        1
    }

    fn release(&mut self, slot: usize) {
        CpuServe::release(self, slot)
    }

    fn prefill_turn(&self) -> usize {
        0
    }

    fn advance_prefill_turn(&mut self, _slot: usize) {}

    fn prefill_prog_t(&self, _prog: usize) -> Option<u32> {
        None
    }

    fn packable_prefill_span(&self, _slot: usize, _max_rows: u32) -> Option<PrefillSpan> {
        None
    }

    fn advance_packed_prefill(&mut self, _members: &[(usize, &[u32])]) -> Result<()> {
        Err(RuntimeError::Rejected(
            "packed prefill is not supported by the CPU engine".into(),
        ))
    }

    fn prefill_frontier(&self, _slot: usize) -> Option<usize> {
        None
    }

    fn prefill_chunked_at_most(
        &mut self,
        slot: usize,
        prompt: &[u32],
        _tick_max_bucket: u32,
    ) -> Result<Option<u32>> {
        self.prefill(slot, prompt).map(Some)
    }

    fn multistep_quantum(&self, _feeds: &[(usize, u32)], _requested: usize) -> Option<usize> {
        None
    }

    fn multi_step(
        &mut self,
        _feeds: &[(usize, u32)],
        _quantum: usize,
        _out: &mut Vec<u32>,
    ) -> Result<usize> {
        Err(RuntimeError::Rejected(
            "multi-step is not supported by the CPU engine".into(),
        ))
    }

    fn step_batch(&mut self, feeds: &[(usize, u32)]) -> Result<Vec<(usize, u32)>> {
        let mut out = Vec::with_capacity(feeds.len());
        for &(slot, id) in feeds {
            out.push((slot, self.step(slot, id)?));
        }
        Ok(out)
    }
}
