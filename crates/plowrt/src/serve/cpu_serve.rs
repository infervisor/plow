//! The CPU engine behind a served slug: `batch` sequence slots, whole-prompt
//! prefill into a slot, one batched greedy decode step per tick — the same
//! shape as the single-GPU AMD engine, so it rides the mux's [`SeqEngine`] tick
//! unchanged. Slot `i` of the mux IS engine slot `i`.

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
    batch: usize,
    /// KV rows written per slot; the next token embeds at this position.
    pos: Vec<u32>,
    live: Vec<bool>,
    /// The token each slot embeds on its next step (the mux's last output).
    next_id: Vec<u32>,
    /// Staging for the batched step, reused across ticks.
    pos_stage: Vec<u32>,
    kvlen_stage: Vec<u32>,
    last_rung: u32,
    max_ctx: usize,
}

impl CpuServe {
    pub fn load(blob: &Path, checkpoint: &Path, opts: &CpuEngineOpts) -> Result<Self> {
        let mut ids = crate::asset::checkpoint::read_eos_ids(checkpoint);
        ids.extend(crate::asset::checkpoint::chat_stop_ids(checkpoint, &ids));
        let eng = CpuEngine::load(blob, checkpoint, opts)?;
        let max_ctx = eng.max_ctx();
        let batch = eng.batch();
        let decode_rungs = eng.decode_rungs().into_boxed_slice();
        tracing::info!(
            max_ctx,
            batch,
            rungs = ?decode_rungs,
            threads = eng.threads,
            isa = ?eng.isa,
            stop_ids = ?ids,
            "CPU serve engine ready"
        );
        Ok(CpuServe {
            eng,
            stop_ids: Arc::new(ids),
            decode_rungs,
            batch,
            pos: vec![0; batch],
            live: vec![false; batch],
            next_id: vec![0; batch],
            pos_stage: vec![0; batch],
            kvlen_stage: vec![1; batch],
            last_rung: 0,
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
        if slot >= self.batch {
            return Err(RuntimeError::Rejected(format!(
                "slot {slot} past engine batch {}",
                self.batch
            )));
        }
        Ok(())
    }

    /// Whole-prompt prefill into `slot`; returns the first generated token.
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
        let tok = self.eng.prefill_slot(slot, prompt)?;
        self.pos[slot] = prompt.len() as u32;
        self.live[slot] = true;
        self.next_id[slot] = tok;
        Ok(tok)
    }

    /// Embed `id` at the slot's position and return the greedy next token
    /// (single-slot convenience; the mux uses [`SeqEngine::step_batch`]).
    pub fn step(&mut self, slot: usize, id: u32) -> Result<u32> {
        let out = SeqEngine::step_batch(self, &[(slot, id)])?;
        Ok(out[0].1)
    }

    /// One batched step advancing `feeds`' slots. Every live slot is stepped
    /// (its KV row at `pos` is rewritten identically if it is not fed), idle
    /// slots carry `(pos 0, kvlen 1)`, and the rung is the narrowest covering
    /// the highest live slot — exactly the AMD `dispatch_all` protocol.
    fn dispatch(&mut self, feeds: &[(usize, u32)]) -> Result<Vec<(usize, u32)>> {
        for &(s, id) in feeds {
            self.check_slot(s)?;
            if !self.live[s] {
                return Err(RuntimeError::Rejected(format!(
                    "step on slot {s} with no prefill"
                )));
            }
            if self.pos[s] as usize >= self.max_ctx {
                return Err(RuntimeError::Rejected(format!(
                    "slot {s} position {} past max_ctx {}",
                    self.pos[s], self.max_ctx
                )));
            }
            self.next_id[s] = id;
        }
        for s in 0..self.batch {
            let (p, k) = if self.live[s] {
                (self.pos[s], self.pos[s] + 1)
            } else {
                (0, 1)
            };
            self.pos_stage[s] = p;
            self.kvlen_stage[s] = k;
        }
        let rows = (0..self.batch)
            .filter(|&s| self.live[s])
            .map(|s| s + 1)
            .max()
            .unwrap_or(1);
        let dp = self.eng.model().decode_prog_for(rows);
        let rung = self.eng.model().blob.progs[dp].t;
        if rung != self.last_rung {
            tracing::info!(rung, occupied = rows, "cpu: decode ladder rung");
            self.last_rung = rung;
        }
        let out = self
            .eng
            .decode_step_batched_at(&self.pos_stage, &self.kvlen_stage, &self.next_id, dp)?;
        for &(s, _) in feeds {
            self.pos[s] += 1;
            self.next_id[s] = out[s];
        }
        Ok(feeds.iter().map(|&(s, _)| (s, out[s])).collect())
    }

    /// Free a slot: the KV block is fixed and preallocated, so this only stops
    /// the slot being fed; the next request rewrites every row it reads.
    pub fn release(&mut self, slot: usize) {
        if slot < self.batch {
            self.live[slot] = false;
            self.pos[slot] = 0;
        }
    }
}

impl SeqEngine for CpuServe {
    fn stop_ids(&self) -> &Arc<Vec<u32>> {
        &self.stop_ids
    }

    fn batch(&self) -> usize {
        self.batch
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
        if feeds.is_empty() {
            return Ok(Vec::new());
        }
        self.dispatch(feeds)
    }
}
