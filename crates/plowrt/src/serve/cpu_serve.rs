//! The CPU engine behind a served slug: `batch` sequence slots, prefill into a
//! slot (whole prompt by default; `PLOW_CPU_PF_CHUNK=n` caps a tick to one
//! compiled bucket of <= n rows), one batched greedy decode step per tick —
//! the same shape as the single-GPU AMD engine, so it rides the mux's
//! [`SeqEngine`] tick unchanged. Slot `i` of the mux IS engine slot `i`.

use std::path::Path;
use std::sync::Arc;

use packet::dev::PrefillSpan;

use super::engine::SeqEngine;
use crate::exec::cpu::engine::{next_chunk, Chunk, CpuEngine, CpuEngineOpts, CpuModel, PackMember};
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
    /// Row `row` of the last program's softcapped logits as f32; `false` = not available.
    fn logits_row(&self, _row: usize, _out: &mut Vec<f32>) -> bool {
        false
    }
    /// [`CpuEngine::pack_rows`]: the widest packed launch, `None` = no packed prefill.
    fn pack_rows(&self) -> Option<u32> {
        None
    }
    /// [`CpuEngine::prefill_packed`].
    fn prefill_packed(&mut self, _members: &[PackMember<'_>]) -> Result<Vec<u32>> {
        Err(RuntimeError::Rejected("packed prefill is not supported by this engine".into()))
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
    fn logits_row(&self, row: usize, out: &mut Vec<f32>) -> bool {
        CpuEngine::logits_row(self, row, out)
    }
    fn pack_rows(&self) -> Option<u32> {
        CpuEngine::pack_rows(self)
    }
    fn prefill_packed(&mut self, members: &[PackMember<'_>]) -> Result<Vec<u32>> {
        CpuEngine::prefill_packed(self, members)
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
    /// Compiled prefill buckets `(program, rows)`.
    buckets: Vec<(usize, u32)>,
    /// Largest chunk one tick may prefill while other slots decode (`PLOW_CPU_PF_CHUNK`;
    /// 0 = whole prompt, the default). MEASURED OFF: at 256 the summarize c=8 cell went from
    /// TTFT 32 s / TPOT 1005 ms to 48 s / 1268 — every chunk re-streams all weights and a
    /// rung-8 decode step (~400 ms) runs between chunks, while live slots still stall for a
    /// whole chunk. Only faster prefill or packing slots into one program helps here.
    pf_chunk: u32,
    /// The logits row holding each slot's latest token (prefill: 0, decode: the slot).
    lp_row: Vec<usize>,
    /// Rows a released slot still holds (its last sequence's `pos`): a session resume keeps
    /// any prefix of them. Idle slots park on this row, the next one a resume rewrites.
    kept: Vec<u32>,
    /// Rows the current sequence resumed from (`cached_rows`).
    resumed: Vec<u32>,
    /// Leading rows of each slot that a prefill wrote (the rest a decode step did). Only these
    /// are reused: a prefill row is the same bits whatever chunk, pack or reused prefix wrote it,
    /// a decoded one is not, so a shared prefix stays bit-identical to a fresh prefill.
    pfilled: Vec<u32>,
    /// Smallest sliding ring's `stride - window`: a resume may drop at most this many tail
    /// rows, or the suffix prefill would read ring entries the dropped rows overwrote.
    ring_slack: u32,
    /// Cross-slot prefix share (`PLOW_CPU_PREFIX_SHARE`): every per-slot KV tensor as
    /// `(handle, heads, ring rows, bytes per row)`, head-major `[slot][head][row]`. Empty = off.
    share_kv: Vec<(usize, u32, u32, usize)>,
    /// A donor whose rows reach this has wrapped a ring and no longer holds `[0, rows)`.
    share_rows: u32,
    /// The tokens behind each slot's KV rows.
    hist: Vec<Vec<u32>>,
    /// Packed prefill (`PLOW_CPU_PACK_PREFILL`): the widest bucket `(program, rows)` a pack runs
    /// on, which a pack is offered against; `None` = off.
    pack: Option<(usize, u32)>,
    /// Prompt length a slot was prepared for packing with (0 = none). Its rows are not yet the
    /// prompt's, so it is never a prefix-share donor.
    pend: Vec<u32>,
    /// `(slot, token)` per prompt the last packed prefill completed.
    packed_tokens: Vec<(usize, u32)>,
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
        // No manifest: only an exact continuation resumes.
        let ring_slack = eng.model().blob.with_packet_view(plow_asset::live_kv::emit).map_or(0, |m| {
            m.caches.iter().filter(|c| c.window > 0).map(|c| c.stride.saturating_sub(c.window)).min().unwrap_or(u32::MAX)
        });
        let pack = eng
            .pack_rows()
            .filter(|_| crate::config::RuntimeConfig::get().cpu.pack_prefill && batch > 1)
            .and_then(|rows| buckets.iter().copied().find(|&(_, t)| t == rows));
        let (share_kv, share_rows) = if crate::config::RuntimeConfig::get().cpu.prefix_share && batch > 1 {
            prefix_share_layout(eng.model(), batch)
        } else {
            (Vec::new(), 0)
        };
        tracing::info!(
            max_ctx,
            batch,
            prefix_share_tensors = share_kv.len(),
            packed_prefill = pack.is_some(),
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
            buckets,
            pf_chunk,
            lp_row: (0..batch).collect(),
            kept: vec![0; batch],
            resumed: vec![0; batch],
            pfilled: vec![0; batch],
            ring_slack,
            share_kv,
            share_rows,
            hist: vec![Vec::new(); batch],
            pack,
            pend: vec![0; batch],
            packed_tokens: Vec::new(),
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
        self.pfilled[slot] = prompt.len() as u32;
        self.lp_row[slot] = 0;
        self.pf_pos[slot] = 0;
        self.pend[slot] = 0;
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
        let n = prompt.len() as u32;
        if self.pf_pos[slot] >= n && self.pf_pos[slot] != 0 {
            return Err(RuntimeError::Rejected(format!(
                "prefill frontier {} is past the {n}-token prompt",
                self.pf_pos[slot]
            )));
        }
        if !self.share_kv.is_empty() {
            if self.pf_pos[slot] == 0 {
                self.share_prefix(slot, prompt);
            }
            self.hist[slot].clear();
            self.hist[slot].extend_from_slice(prompt);
        }
        let ch = next_chunk(&self.buckets, n, self.pf_pos[slot], cap.max(1));
        if let Err(e) = self.eng.prefill_slot_chunk(slot, prompt, ch) {
            self.pf_pos[slot] = 0;
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
            // next chunk rewrites — rows `[0, pf_pos)` stay intact. Idle slots park the same way
            // on `kept`, so rows `[0, kept)` survive for a session resume. A parked slot's output
            // is discarded, so it attends over one row: a retained 16K session below the live
            // extent would otherwise stream its whole KV every step.
            let (p, k) = if self.live[s] {
                (self.pos[s], self.pos[s] + 1)
            } else if self.pf_pos[s] > 0 {
                (self.pf_pos[s], 1)
            } else {
                (self.kept[s], 1)
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
            if !self.share_kv.is_empty() {
                let h = &mut self.hist[s];
                h.truncate(self.pos[s] as usize);
                h.push(self.next_id[s]);
            }
            self.pos[s] += 1;
            self.next_id[s] = out[s];
            self.lp_row[s] = s;
        }
        Ok(feeds.iter().map(|&(s, _)| (s, out[s])).collect())
    }

    /// Free a slot: the KV block is fixed and preallocated, so this only stops the slot being
    /// fed. Its rows stay for a session resume; any other next request rewrites what it reads.
    pub fn release(&mut self, slot: usize) {
        if slot < self.batch {
            self.kept[slot] = if self.live[slot] {
                self.pos[slot].min(self.max_ctx as u32 - 1)
            } else {
                self.pfilled[slot] = self.pf_pos[slot];
                self.pf_pos[slot]
            };
            self.live[slot] = false;
            self.pos[slot] = 0;
            self.pf_pos[slot] = 0;
            self.pend[slot] = 0;
            self.resumed[slot] = 0;
        }
    }

    /// Rows `[0, n)` of `slot` that hold its `hist`, that a prefill wrote, and that no ring has
    /// overwritten.
    fn intact_rows(&self, slot: usize) -> usize {
        let rows = if self.live[slot] {
            self.pos[slot].min(self.pfilled[slot])
        } else if self.pf_pos[slot] > 0 {
            self.pf_pos[slot]
        } else {
            self.kept[slot].min(self.pfilled[slot])
        };
        // The parked step also writes row `rows`, so it must stay inside every ring too.
        if rows >= self.share_rows {
            return 0;
        }
        (rows as usize).min(self.hist[slot].len())
    }

    /// Start a fresh prefill of `slot` at the longest prefix of `prompt` some slot's KV already
    /// holds: copy those rows (one memcpy per tensor and head) and prefill only the rest. The
    /// last prompt row is always prefilled, so the chunk still yields logits.
    fn share_prefix(&mut self, slot: usize, prompt: &[u32]) {
        const MIN_ROWS: usize = 32;
        let cap = prompt.len() - 1;
        let mut best = (0usize, slot);
        for d in (0..self.batch).filter(|&d| d == slot || self.pend[d] == 0) {
            let n = self.intact_rows(d).min(cap);
            let l = self.hist[d][..n].iter().zip(prompt).take_while(|(a, b)| a == b).count();
            if l > best.0 || (l == best.0 && d == slot) {
                best = (l, d);
            }
        }
        let (rows, donor) = best;
        if rows < MIN_ROWS {
            return;
        }
        if donor != slot {
            let model = self.eng.model();
            for &(h, heads, stride, row) in &self.share_kv {
                let base = model.tensor(h).as_ptr();
                let block = heads as usize * stride as usize * row;
                for head in 0..heads as usize {
                    let off = head * stride as usize * row;
                    // SAFETY: `prefix_share_layout` checked each tensor is `batch` blocks of
                    // `block` bytes; donor != slot keeps the ranges disjoint; no program runs.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            base.add(donor * block + off),
                            base.add(slot * block + off),
                            rows * row,
                        );
                    }
                }
            }
        }
        tracing::debug!(slot, donor, rows, "cpu: prefix share");
        self.pf_pos[slot] = rows as u32;
        self.resumed[slot] = rows as u32;
    }

    /// Start `slot`'s next prefill over the rows its last sequence left: the first `rows`, at
    /// most those a prefill wrote. Returns the rows kept (0 = cold).
    pub fn resume_slot(&mut self, slot: usize, rows: usize) -> usize {
        let rows = rows.min(self.pfilled.get(slot).map_or(0, |&p| p as usize));
        let ok = slot < self.batch
            && !self.live[slot]
            && self.pf_pos[slot] == 0
            && rows > 0
            && rows <= self.kept[slot] as usize
            && rows < self.max_ctx
            && {
                let dropped = self.kept[slot] as usize - rows;
                dropped == 0 || dropped < self.ring_slack as usize
            };
        if !ok {
            return 0;
        }
        self.pf_pos[slot] = rows as u32;
        self.resumed[slot] = rows as u32;
        rows
    }
}

/// The live-KV manifest's per-slot BF16 caches as `(handle, heads, rows, row bytes)` plus the
/// smallest ring, or nothing when a cache is not a plain head-major block a row copy can move.
fn prefix_share_layout(model: &CpuModel, batch: usize) -> (Vec<(usize, u32, u32, usize)>, u32) {
    let Ok(m) = model.blob.with_packet_view(plow_asset::live_kv::emit) else {
        return (Vec::new(), 0);
    };
    let mut kv: Vec<(usize, u32, u32, usize)> = Vec::new();
    for c in &m.caches {
        let identity = c.mask == u32::MAX || (c.stride.is_power_of_two() && c.mask == c.stride - 1);
        if c.scales.is_some() || !identity {
            return (Vec::new(), 0);
        }
        let row = c.hd as usize * 2;
        for h in c.pair.map(usize::from) {
            if model.tensor(h).bytes != batch * c.heads as usize * c.stride as usize * row {
                return (Vec::new(), 0);
            }
            if !kv.iter().any(|e| e.0 == h) {
                kv.push((h, c.heads, c.stride, row));
            }
        }
    }
    let rows = m.caches.iter().map(|c| c.stride).min().unwrap_or(0);
    (kv, rows)
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
        self.buckets.iter().find(|&&(p, _)| p == prog).map(|&(_, t)| t)
    }

    fn step_backend(&self) -> crate::sched::step::Backend {
        crate::sched::step::Backend {
            packing: self.pack.is_some(),
            ..Default::default()
        }
    }

    fn resume_before_pack(&self) -> bool {
        self.pack.is_some()
    }

    /// Seeds the slot's packing cursor (its prompt length) and its prefix share, so a fresh
    /// prompt is a pack candidate in the tick it arrives.
    fn prepare_packed_prefill_slot(&mut self, slot: usize, prompt: &[u32], _max_rows: u32) -> Result<()> {
        if self.pack.is_none() || self.live.get(slot) != Some(&false) || self.pend[slot] != 0 {
            return Ok(());
        }
        self.check_prompt(slot, prompt)?;
        if !self.share_kv.is_empty() && self.pf_pos[slot] == 0 {
            self.share_prefix(slot, prompt);
        }
        self.pend[slot] = prompt.len() as u32;
        Ok(())
    }

    /// The slot's whole remaining prompt, offered against the widest packed bucket when it fits
    /// one: then prefilling it alone would be a single chunk too, which the pack reproduces bit
    /// for bit. [`Self::advance_packed_prefill`] runs the narrowest bucket holding the pack.
    fn packable_prefill_span(&self, slot: usize, max_rows: u32) -> Option<PrefillSpan> {
        let (prog, widest) = self.pack?;
        let (n, c0) = (*self.pend.get(slot)?, self.pf_pos[slot]);
        let rows = n.checked_sub(c0).filter(|&r| r > 0 && r <= widest)?;
        (!self.live[slot] && rows <= max_rows).then_some(PrefillSpan {
            row0: 0,
            n_rows: rows,
            slot: slot as u32,
            flags: 0,
            kv_row0: c0,
            kv_len: c0 + rows,
            state_slot: slot as u32,
            program: prog as u32,
        })
    }

    fn advance_packed_prefill(&mut self, members: &[(usize, &[u32])]) -> Result<()> {
        if self.pack.is_none() {
            return Err(RuntimeError::Rejected("packed prefill is off".into()));
        }
        self.packed_tokens.clear();
        let mut pm = Vec::with_capacity(members.len());
        for &(slot, prompt) in members {
            self.check_prompt(slot, prompt)?;
            if self.live[slot] || self.pend[slot] as usize != prompt.len() {
                return Err(RuntimeError::Rejected(format!(
                    "packed prefill slot {slot} was not prepared for this prompt"
                )));
            }
            let c0 = self.pf_pos[slot] as usize;
            pm.push(PackMember {
                slot,
                c0: c0 as u32,
                rows: &prompt[c0..],
                sample: true,
            });
        }
        if !self.share_kv.is_empty() {
            for &(slot, prompt) in members {
                self.hist[slot].clear();
                self.hist[slot].extend_from_slice(prompt);
            }
        }
        let toks = match self.eng.prefill_packed(&pm) {
            Ok(t) => t,
            Err(e) => {
                for &(slot, _) in members {
                    self.pf_pos[slot] = 0;
                }
                return Err(e);
            }
        };
        let mut toks = toks.into_iter().enumerate();
        for (mb, &(_, prompt)) in pm.iter().zip(members) {
            self.pf_pos[mb.slot] += mb.rows.len() as u32;
            if mb.sample {
                let (row, tok) = toks.next().expect("one token per sampled member");
                self.admit_prefilled(mb.slot, prompt, tok);
                self.lp_row[mb.slot] = row;
                self.packed_tokens.push((mb.slot, tok));
            }
        }
        Ok(())
    }

    fn take_packed_tokens(&mut self) -> Vec<(usize, u32)> {
        std::mem::take(&mut self.packed_tokens)
    }

    fn prefill_frontier(&self, slot: usize) -> Option<usize> {
        (slot < self.batch).then(|| self.pf_pos[slot] as usize)
    }

    fn resume_slot(&mut self, slot: usize, rows: usize) -> usize {
        CpuServe::resume_slot(self, slot, rows)
    }

    fn cached_rows(&self, slot: usize) -> usize {
        self.resumed.get(slot).map_or(0, |&r| r as usize)
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
        let cap = if crate::serve::policy::installed_co_sched() == super::cosched::CoSched::Rr {
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

    fn logits_row(&self, slot: usize, out: &mut Vec<f32>) -> bool {
        slot < self.batch && self.eng.logits_row(self.lp_row[slot], out)
    }

    fn step_batch(&mut self, feeds: &[(usize, u32)]) -> Result<Vec<(usize, u32)>> {
        if feeds.is_empty() {
            return Ok(Vec::new());
        }
        self.dispatch(feeds)
    }
}
