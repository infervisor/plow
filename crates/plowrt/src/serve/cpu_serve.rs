//! The CPU engine behind a served slug: `batch` sequence slots, prefill into a
//! slot (whole prompt by default; `PLOW_CPU_PF_CHUNK=n` caps a tick to one
//! compiled bucket of <= n rows), one batched greedy decode step per tick —
//! the same shape as the single-GPU AMD engine, so it rides the mux's
//! [`SeqEngine`] tick unchanged. Slot `i` of the mux IS engine slot `i`.

use std::path::Path;
use std::sync::Arc;

use packet::dev::PrefillSpan;

use super::engine::SeqEngine;
use crate::exec::cpu::engine::{next_chunk, Chunk, CpuEngine, CpuEngineOpts, CpuModel};
use crate::{Result, RuntimeError};

/// The slot-engine surface this serve engine drives. Implemented by the CPU worker-pool engine
/// and the Metal engine: both load through [`CpuModel`] and share the program/slot semantics
/// (single-sequence prefill rebased onto a slot, one batched decode step on the narrowest rung),
/// so one serve engine and one mux tick body cover both units.
pub trait SlotEngine: Send {
    fn prefill_buckets(&self) -> Vec<(usize, u32)>;
    fn prefill_slot(&mut self, slot: usize, prompt: &[u32]) -> Result<u32>;
    fn prefill_slot_chunk(&mut self, slot: usize, prompt: &[u32], ch: Chunk) -> Result<()>;
    fn last_token(&self) -> Result<u32>;
    fn decode_step_batched_at(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        ids: &[u32],
        dp: usize,
    ) -> Result<Vec<u32>>;
    fn model(&self) -> &CpuModel;
    fn max_ctx(&self) -> usize;
    /// One line for the ready log (unit, threads, tier).
    fn describe(&self) -> String;
    fn packed_prefill(&self) -> bool {
        false
    }
    fn prefill_packed(
        &mut self,
        _prog: usize,
        _spans: &[PrefillSpan],
        _prompts: &[&[u32]],
    ) -> Result<Vec<u32>> {
        Err(RuntimeError::Rejected(
            "packed prefill is not supported by this slot engine".into(),
        ))
    }
}

impl SlotEngine for CpuEngine {
    fn prefill_buckets(&self) -> Vec<(usize, u32)> {
        CpuEngine::prefill_buckets(self)
    }
    fn prefill_slot(&mut self, slot: usize, prompt: &[u32]) -> Result<u32> {
        CpuEngine::prefill_slot(self, slot, prompt)
    }
    fn prefill_slot_chunk(&mut self, slot: usize, prompt: &[u32], ch: Chunk) -> Result<()> {
        CpuEngine::prefill_slot_chunk(self, slot, prompt, ch)
    }
    fn last_token(&self) -> Result<u32> {
        CpuEngine::last_token(self)
    }
    fn decode_step_batched_at(
        &mut self,
        pos: &[u32],
        kvlen: &[u32],
        ids: &[u32],
        dp: usize,
    ) -> Result<Vec<u32>> {
        CpuEngine::decode_step_batched_at(self, pos, kvlen, ids, dp)
    }
    fn model(&self) -> &CpuModel {
        CpuEngine::model(self)
    }
    fn max_ctx(&self) -> usize {
        CpuEngine::max_ctx(self)
    }
    fn describe(&self) -> String {
        format!("cpu threads={} isa={:?}", self.threads, self.isa)
    }
}

pub struct CpuServe {
    eng: Box<dyn SlotEngine>,
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
    /// Prompt rows already prefilled per slot (0 = no chunked prefill in flight).
    pf_pos: Vec<u32>,
    pf_len: Vec<u32>,
    pf_ready: Vec<Option<u32>>,
    /// Compiled prefill buckets `(program, rows)`.
    buckets: Vec<(usize, u32)>,
    /// Largest chunk one tick may prefill while other slots decode (`PLOW_CPU_PF_CHUNK`;
    /// 0 = whole prompt, the default). MEASURED OFF: at 256 the summarize c=8 cell went from
    /// TTFT 32 s / TPOT 1005 ms to 48 s / 1268 — every chunk re-streams all weights and a
    /// rung-8 decode step (~400 ms) runs between chunks, while live slots still stall for a
    /// whole chunk. Only faster prefill or packing slots into one program helps here.
    pf_chunk: u32,
}

impl CpuServe {
    pub fn load(blob: &Path, checkpoint: &Path, opts: &CpuEngineOpts) -> Result<Self> {
        let eng = CpuEngine::load(blob, checkpoint, opts)?;
        Self::from_engine(Box::new(eng), checkpoint)
    }

    /// Wrap an already-loaded slot engine (CPU or Metal).
    pub fn from_engine(eng: Box<dyn SlotEngine>, checkpoint: &Path) -> Result<Self> {
        let mut ids = crate::asset::checkpoint::read_eos_ids(checkpoint);
        ids.extend(crate::asset::checkpoint::chat_stop_ids(checkpoint, &ids));
        let max_ctx = eng.max_ctx();
        let batch = eng.model().batch;
        let decode_rungs = eng.model().decode_rungs().into_boxed_slice();
        let buckets = eng.prefill_buckets();
        if buckets.is_empty() {
            return Err(RuntimeError::Device(
                "CPU serve blob has no prefill program".into(),
            ));
        }
        let pf_chunk = crate::config::RuntimeConfig::get().cpu.prefill_chunk;
        tracing::info!(
            max_ctx,
            batch,
            rungs = ?decode_rungs,
            prefill_buckets = ?buckets,
            pf_chunk,
            engine = %eng.describe(),
            stop_ids = ?ids,
            "slot serve engine ready"
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
            pf_pos: vec![0; batch],
            pf_len: vec![0; batch],
            pf_ready: vec![None; batch],
            buckets,
            pf_chunk,
        })
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn decode_rungs(&self) -> &[u32] {
        &self.decode_rungs
    }

    /// Sequence slots the mux may admit concurrently (the blob's decode batch).
    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn engine(&self) -> &dyn SlotEngine {
        &*self.eng
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

    fn check_prompt(&self, slot: usize, prompt: &[u32]) -> Result<()> {
        self.check_slot(slot)?;
        if prompt.is_empty() {
            return Err(RuntimeError::Rejected("empty prompt".into()));
        }
        if prompt.len() >= self.max_ctx {
            return Err(RuntimeError::ContextLength(format!(
                "prompt is {} tokens, max_ctx is {}",
                prompt.len(),
                self.max_ctx
            )));
        }
        Ok(())
    }

    fn admit_prefilled(&mut self, slot: usize, prompt: &[u32], tok: u32) {
        self.pf_pos[slot] = 0;
        self.pf_len[slot] = 0;
        self.pos[slot] = prompt.len() as u32;
        self.live[slot] = true;
        self.next_id[slot] = tok;
    }

    /// Whole-prompt prefill into `slot`; returns the first generated token.
    pub fn prefill(&mut self, slot: usize, prompt: &[u32]) -> Result<u32> {
        self.check_prompt(slot, prompt)?;
        let tok = self.eng.prefill_slot(slot, prompt)?;
        self.admit_prefilled(slot, prompt, tok);
        Ok(tok)
    }

    /// One prefill chunk of at most `cap` rows into `slot`; `Ok(Some(tok))` once the prompt is
    /// covered. Between chunks the slot is NOT live: the batched step parks it on its frontier
    /// row (see `dispatch`), so a decode tick in between cannot touch a finished KV row.
    pub fn prefill_chunk(&mut self, slot: usize, prompt: &[u32], cap: u32) -> Result<Option<u32>> {
        self.check_prompt(slot, prompt)?;
        if let Some(tok) = self.pf_ready[slot].take() {
            self.admit_prefilled(slot, prompt, tok);
            return Ok(Some(tok));
        }
        let n = prompt.len() as u32;
        if self.pf_pos[slot] >= n && self.pf_pos[slot] != 0 {
            return Err(RuntimeError::Rejected(format!(
                "prefill frontier {} is past the {n}-token prompt",
                self.pf_pos[slot]
            )));
        }
        let ch = next_chunk(&self.buckets, n, self.pf_pos[slot], cap.max(1));
        if let Err(e) = self.eng.prefill_slot_chunk(slot, prompt, ch) {
            self.pf_pos[slot] = 0;
            self.pf_len[slot] = 0;
            self.pf_ready[slot] = None;
            return Err(e);
        }
        self.pf_pos[slot] += ch.clen;
        if self.pf_pos[slot] < n {
            return Ok(None);
        }
        let tok = self.eng.last_token()?;
        self.admit_prefilled(slot, prompt, tok);
        Ok(Some(tok))
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
            // A slot mid-prefill is parked on its frontier row: the batched step's KV write
            // for a non-fed slot lands on `pos`, and the frontier row is exactly the one the
            // next chunk rewrites — rows `[0, pf_pos)` stay intact. Idle slots park on row 0.
            let (p, k) = if self.live[s] {
                (self.pos[s], self.pos[s] + 1)
            } else if self.pf_pos[s] > 0 {
                (self.pf_pos[s], self.pf_pos[s] + 1)
            } else {
                (0, 1)
            };
            self.pos_stage[s] = p;
            self.kvlen_stage[s] = k;
        }
        let rows = crate::sched::rungs::occupied_extent(self.live.iter().copied()).max(1);
        let dp = self.eng.model().decode_prog_for(rows);
        let rung = self.eng.model().blob.progs[dp].t;
        if rung != self.last_rung {
            tracing::info!(rung, occupied = rows, "cpu: decode ladder rung");
            self.last_rung = rung;
        }
        let out = self.eng.decode_step_batched_at(
            &self.pos_stage,
            &self.kvlen_stage,
            &self.next_id,
            dp,
        )?;
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
            self.pf_pos[slot] = 0;
            self.pf_len[slot] = 0;
            self.pf_ready[slot] = None;
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

    fn prefill_prog_t(&self, prog: usize) -> Option<u32> {
        self.buckets
            .iter()
            .find_map(|&(p, rows)| (p == prog).then_some(rows))
    }

    fn step_backend(&self) -> crate::sched::step::Backend {
        if !self.eng.packed_prefill() {
            return crate::sched::step::Backend::default();
        }
        crate::sched::step::Backend {
            step_budget: self
                .buckets
                .iter()
                .map(|&(_, rows)| rows)
                .max()
                .unwrap_or(u32::MAX),
            packing: true,
            split_spans: false,
            decode_rows_join_prefill: false,
        }
    }

    fn packable_prefill_span(&self, slot: usize, max_rows: u32) -> Option<PrefillSpan> {
        if !self.eng.packed_prefill() || self.live.get(slot).copied()? {
            return None;
        }
        let rows = *self.pf_len.get(slot)?;
        if rows == 0 || !rows.is_multiple_of(32) || self.pf_pos.get(slot).copied()? != 0 {
            return None;
        }
        let peers = self
            .pf_len
            .iter()
            .zip(&self.pf_pos)
            .zip(&self.live)
            .filter(|&((&len, &pos), &live)| !live && pos == 0 && len == rows)
            .count();
        if peers < 4 {
            return None;
        }
        let wanted = rows
            .checked_mul(u32::try_from(peers).ok()?)?
            .min(max_rows);
        let program = self
            .buckets
            .iter()
            .find(|&&(_, width)| width >= wanted && width <= max_rows)?
            .0;
        Some(PrefillSpan {
            row0: 0,
            n_rows: rows,
            slot: slot as u32,
            flags: packet::dev::PREFILL_SPAN_RESET_STATE,
            kv_row0: 0,
            kv_len: rows,
            state_slot: slot as u32,
            program: program as u32,
        })
    }

    fn advance_packed_prefill(&mut self, members: &[(usize, &[u32])]) -> Result<()> {
        if members.len() < 2 || !self.eng.packed_prefill() {
            return Err(RuntimeError::Rejected(
                "packed prefill is unavailable".into(),
            ));
        }
        let total = members.iter().try_fold(0u32, |rows, &(slot, prompt)| {
            self.check_prompt(slot, prompt)?;
            if self.live[slot] || self.pf_pos[slot] != 0 || self.pf_len[slot] != prompt.len() as u32
            {
                return Err(RuntimeError::Rejected(
                    "invalid packed prefill member".into(),
                ));
            }
            rows.checked_add(prompt.len() as u32)
                .ok_or_else(|| RuntimeError::Rejected("packed prefill row overflow".into()))
        })?;
        let (prog, _) = self
            .buckets
            .iter()
            .copied()
            .find(|&(_, width)| width >= total)
            .ok_or_else(|| RuntimeError::Rejected("no packed prefill bucket".into()))?;
        let mut row0 = 0u32;
        let spans: Vec<_> = members
            .iter()
            .map(|&(slot, prompt)| {
                let rows = prompt.len() as u32;
                let span = PrefillSpan {
                    row0,
                    n_rows: rows,
                    slot: slot as u32,
                    flags: packet::dev::PREFILL_SPAN_RESET_STATE,
                    kv_row0: 0,
                    kv_len: rows,
                    state_slot: slot as u32,
                    program: prog as u32,
                };
                row0 += rows;
                span
            })
            .collect();
        let prompts: Vec<_> = members.iter().map(|&(_, prompt)| prompt).collect();
        let tokens = self.eng.prefill_packed(prog, &spans, &prompts)?;
        if tokens.len() != members.len() {
            return Err(RuntimeError::Device("packed prefill token count".into()));
        }
        for ((&(slot, prompt), token), span) in members.iter().zip(tokens).zip(&spans) {
            self.pf_pos[slot] = prompt.len() as u32;
            self.pf_ready[slot] = Some(token);
            debug_assert_eq!(span.slot as usize, slot);
        }
        Ok(())
    }

    fn prefill_frontier(&self, slot: usize) -> Option<usize> {
        (slot < self.batch).then(|| self.pf_pos[slot] as usize)
    }

    fn prepare_packed_prefill_slot(
        &mut self,
        slot: usize,
        prompt: &[u32],
        _max_rows: u32,
    ) -> Result<()> {
        if self.eng.packed_prefill() {
            self.check_prompt(slot, prompt)?;
            if !self.live[slot] && self.pf_pos[slot] == 0 {
                self.pf_len[slot] = prompt.len() as u32;
            }
        }
        Ok(())
    }

    fn terminal_prefill_ready(&self, slot: usize, prompt: &[u32]) -> bool {
        self.pf_ready.get(slot).is_some_and(Option::is_some)
            && self.pf_pos.get(slot).copied() == Some(prompt.len() as u32)
    }

    fn finish_prefill_batch(
        &mut self,
        feeds: &[(usize, u32)],
        members: &[(usize, &[u32])],
    ) -> Result<Vec<(usize, u32)>> {
        for &(slot, prompt) in members {
            self.check_prompt(slot, prompt)?;
            if !self.terminal_prefill_ready(slot, prompt) {
                return Err(RuntimeError::Rejected(
                    "packed prefill member is not ready".into(),
                ));
            }
        }
        let mut output = self.dispatch(feeds)?;
        for &(slot, prompt) in members {
            let token = self.pf_ready[slot]
                .take()
                .expect("terminal prefill was validated above");
            self.admit_prefilled(slot, prompt, token);
            output.push((slot, token));
        }
        Ok(output)
    }

    /// `tick_max_bucket` is the mux's interleave budget (u32::MAX when no slot decodes, so a
    /// lone prompt still prefills in one tick); `pf_chunk` caps it further on the CPU, where a
    /// 1105-token whole-prompt prefill stalled every live decode ~7 s (measured: c=8
    /// summarize TPOT 933 ms vs 250 at c=1).
    fn prefill_chunked_at_most(
        &mut self,
        slot: usize,
        prompt: &[u32],
        tick_max_bucket: u32,
    ) -> Result<Option<u32>> {
        let cap = if crate::config::RuntimeConfig::get().co_sched == super::cosched::CoSched::Rr {
            tick_max_bucket.min(if self.pf_chunk == 0 {
                u32::MAX
            } else {
                self.pf_chunk
            })
        } else if tick_max_bucket == u32::MAX || self.pf_chunk == 0 {
            u32::MAX
        } else {
            tick_max_bucket.min(self.pf_chunk)
        };
        self.prefill_chunk(slot, prompt, cap)
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
