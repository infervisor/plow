//! "Lowering accepts what the planner emits", device-free.
//!
//! The single-sequence tick (`mux::run_one_tick`'s `SeqEngine` arm) lowers a
//! [`crate::sched::step::Plan`] as: a pack → `advance_packed_prefill`, a lone span →
//! `prefill_chunked_at_most`, the decodes → the batched dispatch. This module runs that same
//! lowering against a mock engine that declares a backend and REFUSES BY NAME anything outside
//! it, exactly as the CPU engine (`cpu_serve.rs`, no packing) and the AMD engine (packs of two
//! or more distinct, whole spans) do — so a plan the planner emits for a declared backend is,
//! by this test, one the backend can execute without narrowing it.

use super::engine::SeqEngine;
use crate::sched::step::{plan, Backend, Candidate, Plan, Tick};
use crate::{Result, RuntimeError};
use packet::dev::PrefillSpan;
use std::collections::BTreeMap;
use std::sync::Arc;

struct Mock {
    backend: Backend,
    stop: Arc<Vec<u32>>,
    /// slot -> (next chunk rows, packable)
    cursors: BTreeMap<usize, (u32, bool)>,
    packs: Vec<Vec<usize>>,
    chunks: Vec<(usize, u32)>,
}

impl SeqEngine for Mock {
    fn stop_ids(&self) -> &Arc<Vec<u32>> {
        &self.stop
    }
    fn batch(&self) -> usize {
        32
    }
    fn release(&mut self, _slot: usize) {}
    fn prefill_turn(&self) -> usize {
        0
    }
    fn advance_prefill_turn(&mut self, _slot: usize) {}
    fn prefill_prog_t(&self, prog: usize) -> Option<u32> {
        (prog == 1 && self.backend.packing).then_some(2048)
    }
    fn packable_prefill_span(&self, slot: usize, max_rows: u32) -> Option<PrefillSpan> {
        if !self.backend.packing {
            return None;
        }
        let &(rows, packable) = self.cursors.get(&slot)?;
        (packable && rows <= max_rows && rows <= 2048).then(|| PrefillSpan {
            row0: 0,
            n_rows: rows,
            slot: slot as u32,
            flags: 0,
            kv_row0: 4096,
            kv_len: 4096 + rows,
            state_slot: slot as u32,
            program: 1,
        })
    }
    fn advance_packed_prefill(&mut self, members: &[(usize, &[u32])]) -> Result<()> {
        if !self.backend.packing {
            return Err(RuntimeError::Rejected(
                "packed prefill is not supported by this engine".into(),
            ));
        }
        if members.len() < 2 {
            return Err(RuntimeError::Rejected(
                "packed prefill requires at least two cursors".into(),
            ));
        }
        let mut slots: Vec<usize> = members.iter().map(|m| m.0).collect();
        slots.sort_unstable();
        slots.dedup();
        if slots.len() != members.len() {
            return Err(RuntimeError::Rejected("packed prefill slot appears twice".into()));
        }
        let rows: u32 = members
            .iter()
            .map(|&(slot, _)| self.cursors.get(&slot).map_or(u32::MAX, |c| c.0))
            .sum();
        if rows > 2048 {
            return Err(RuntimeError::Rejected(format!(
                "no packed prefill rung covers {rows} rows"
            )));
        }
        self.packs.push(members.iter().map(|m| m.0).collect());
        Ok(())
    }
    fn prefill_frontier(&self, slot: usize) -> Option<usize> {
        self.cursors.contains_key(&slot).then_some(4096)
    }
    fn next_prefill_rows(&self, slot: usize) -> Option<u32> {
        self.cursors.get(&slot).map(|c| c.0)
    }
    fn step_backend(&self) -> Backend {
        self.backend
    }
    fn prefill_chunked_at_most(
        &mut self,
        slot: usize,
        _prompt: &[u32],
        tick_max_bucket: u32,
    ) -> Result<Option<u32>> {
        let rows = self.cursors.get(&slot).map_or(tick_max_bucket.min(8192), |c| c.0);
        if rows > tick_max_bucket {
            return Err(RuntimeError::Rejected(format!(
                "chunk of {rows} rows exceeds the tick cap {tick_max_bucket}"
            )));
        }
        self.chunks.push((slot, rows));
        Ok(None)
    }
    fn multistep_quantum(&self, _feeds: &[(usize, u32)], _requested: usize) -> Option<usize> {
        None
    }
    fn multi_step(&mut self, _f: &[(usize, u32)], _q: usize, _o: &mut Vec<u32>) -> Result<usize> {
        Err(RuntimeError::Rejected("multi-step unavailable".into()))
    }
    fn step_batch(&mut self, feeds: &[(usize, u32)]) -> Result<Vec<(usize, u32)>> {
        Ok(feeds.to_vec())
    }
}

/// The mux's lowering, verbatim in shape: pack → packed advance, span → one chunk against the
/// FULL tick cap, decodes → one batched step.
fn lower(e: &mut dyn SeqEngine, step: &Plan, tick_max: u32) -> Result<()> {
    let prompt = [0u32; 8];
    for launch in &step.launches {
        if launch.is_pack() {
            let members: Vec<(usize, &[u32])> = launch
                .spans
                .iter()
                .map(|span| (span.slot as usize, &prompt[..]))
                .collect();
            e.advance_packed_prefill(&members)?;
        } else {
            e.prefill_chunked_at_most(launch.spans[0].slot as usize, &prompt, tick_max)?;
        }
    }
    let feeds: Vec<(usize, u32)> = step.decodes.iter().map(|&s| (s as usize, 7)).collect();
    e.step_batch(&feeds)?;
    Ok(())
}

/// The mux's candidate construction, in shape.
fn candidates(e: &dyn SeqEngine, pending: &[(usize, u64)], full_budget: u32, pf_batch: bool) -> Vec<Candidate> {
    pending
        .iter()
        .map(|&(slot, arrival)| {
            if let Some(span) = pf_batch.then(|| e.packable_prefill_span(slot, full_budget)).flatten() {
                return Candidate { span, arrival, packable: true, planned: true };
            }
            let planned = e.next_prefill_rows(slot);
            let rows = planned.unwrap_or(u32::MAX);
            Candidate {
                span: PrefillSpan {
                    row0: 0,
                    n_rows: rows,
                    slot: slot as u32,
                    flags: 0,
                    kv_row0: 0,
                    kv_len: rows,
                    state_slot: slot as u32,
                    program: 0,
                },
                arrival,
                packable: false,
                planned: planned.is_some(),
            }
        })
        .collect()
}

fn mixes() -> impl Iterator<Item = (u32, BTreeMap<usize, (u32, bool)>, Vec<(usize, u64)>)> {
    (0..120u64).map(|seed| {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let d = (next() % 21) as u32;
        let n = (next() % 8) as usize;
        let mut cursors = BTreeMap::new();
        let mut pending = Vec::new();
        for i in 0..n {
            let slot = d as usize + i;
            pending.push((slot, next() % 100));
            if next() % 4 != 0 {
                let rows = [128, 512, 1024, 2048, 8192][(next() % 5) as usize];
                cursors.insert(slot, (rows, next() % 2 == 0));
            }
        }
        (d, cursors, pending)
    })
}

fn run(backend: Backend, pf_batch: bool, cap_rows: u32) -> (usize, usize) {
    let (mut packs, mut chunks) = (0, 0);
    for (d, cursors, pending) in mixes() {
        let mut e = Mock {
            backend,
            stop: Arc::new(Vec::new()),
            cursors,
            packs: Vec::new(),
            chunks: Vec::new(),
        };
        let full_budget = cap_rows.min(backend.step_budget);
        let cands = candidates(&e, &pending, full_budget, pf_batch);
        let step = plan(
            backend,
            Tick { cap_rows, packing: pf_batch, rotate: false, turn: 0, slots: 32 },
            0..d,
            &cands,
            |p| e.prefill_prog_t(p as usize),
            |_| u32::MAX,
        );
        lower(&mut e, &step, cap_rows).unwrap_or_else(|err| {
            panic!("backend {backend:?} refused a plan it was declared for: {err}\n{step:#?}")
        });
        assert_eq!(step.decodes.len(), d as usize);
        packs += e.packs.len();
        chunks += e.chunks.len();
    }
    (packs, chunks)
}

/// The CPU engine's declaration (`Backend::default()`): whole prompts, no packing, no ladder.
/// The planner never hands it a pack and never more than one launch per tick.
#[test]
fn a_whole_prompt_engine_is_never_asked_to_pack() {
    let (packs, chunks) = run(Backend::default(), true, u32::MAX);
    assert_eq!(packs, 0);
    assert!(chunks > 0);
    let (packs, _) = run(Backend::default(), false, 2048);
    assert_eq!(packs, 0);
}

/// The AMD engine's declaration: an 8192 budget, whole spans, packs of two or more distinct
/// spans within a 2048 packed rung. Every emitted pack is accepted and packs do form.
#[test]
fn a_packing_engine_accepts_every_pack_and_chunk_the_planner_emits() {
    let amd = Backend {
        step_budget: 8192,
        packing: true,
        split_spans: false,
        decode_rows_join_prefill: false,
    };
    let (packs, chunks) = run(amd, true, u32::MAX);
    assert!(packs > 0, "the ragged mixes contain packable pairs");
    assert!(chunks > 0);
    // Packing switched off at the runtime: the same engine is handed chunks only.
    let (packs, _) = run(amd, false, u32::MAX);
    assert_eq!(packs, 0);
    // A tick cap below the budget still yields only launches that fit it.
    run(amd, true, 2048);
}

/// A plan built for a packing backend is refused BY NAME by an engine that does not pack —
/// the contract that forbids a silent narrower plan on the wrong backend.
#[test]
fn a_pack_handed_to_a_non_packing_engine_is_refused_by_name() {
    let mut cpu = Mock {
        backend: Backend::default(),
        stop: Arc::new(Vec::new()),
        cursors: BTreeMap::from([(0, (512, true)), (1, (512, true))]),
        packs: Vec::new(),
        chunks: Vec::new(),
    };
    let span = |slot: u32| PrefillSpan {
        row0: 0,
        n_rows: 512,
        slot,
        flags: 0,
        kv_row0: 0,
        kv_len: 512,
        state_slot: slot,
        program: 1,
    };
    let foreign = Plan {
        decodes: Vec::new(),
        launches: vec![crate::sched::step::Launch { spans: vec![span(0), span(1)] }],
        budget_left: 0,
    };
    let err = lower(&mut cpu, &foreign, u32::MAX).unwrap_err().to_string();
    assert!(err.contains("packed prefill is not supported"), "{err}");
    assert!(cpu.packs.is_empty() && cpu.chunks.is_empty());
}
