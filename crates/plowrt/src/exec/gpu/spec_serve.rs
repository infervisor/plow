//! One speculative serving tick: draft, verify, commit (`PLOW_SPEC`).
//!
//! The scheduler names rows by slot, pending token and draft depth; which hidden rows (or tap
//! rows) a drafter consumes and how it is launched stay behind [`GpuEngine::spec_tick`].

use std::path::Path;
use std::time::Instant;

use crate::orch::speculative::SpecVerdict;
use crate::{Result, RuntimeError};

use super::spec::SpecSpan;
use super::GpuEngine;

/// What a speculative tick may ask of this engine.
#[derive(Clone, Debug)]
pub struct SpecLimits {
    /// Deepest draft per row (drafter depth and the sliding rings' slack).
    pub kmax: u32,
    /// Widest verify launch, in rows.
    pub max_rows: usize,
    /// Draft program row counts, ascending.
    pub draft_rows: Vec<usize>,
}

/// One row of a speculative tick: `k = 0` decodes the pending token plainly inside the verify.
#[derive(Clone, Copy, Debug)]
pub struct SpecRow {
    pub slot: usize,
    pub token: u32,
    pub k: u32,
}

/// A tick's outcome, per row in request order: the verdict and the verify row of its pending
/// token (where a host-sampled row reads its logits).
#[derive(Default)]
pub struct SpecTick {
    pub verdicts: Vec<SpecVerdict>,
    pub row0: Vec<u32>,
    pub rows: usize,
    pub draft_rows: usize,
    pub draft_k: u32,
    pub draft_ms: f64,
    pub verify_ms: f64,
    /// Rows committed: none means every slot is as it was (rows past `pos` are dead by design).
    pub committed: usize,
    drafts: Vec<u32>,
    tokens: Vec<u32>,
    stage: Vec<(usize, u32, u32)>,
}

fn reject(m: impl std::fmt::Display) -> RuntimeError {
    RuntimeError::Rejected(format!("spec tick: {m}"))
}

impl GpuEngine {
    /// Arm verify and load the packet's drafter for serving.
    pub fn spec_serve_enable(&mut self, assets: &Path) -> Result<SpecLimits> {
        let ring_k = self.spec_enable(assets)?;
        self.spec_draft_enable(assets)?;
        let limits = self.spec_serving().ok_or_else(|| reject("not armed"))?;
        if ring_k < limits.kmax as usize {
            tracing::info!(ring_k, "speculative depth capped by the sliding rings");
        }
        Ok(limits)
    }

    pub fn spec_serving(&self) -> Option<SpecLimits> {
        let (kmax, draft_rows) = self.spec_draft_shape()?;
        let ring_k = self
            .resume_ring_slack
            .map_or(u32::MAX, |s| s.saturating_sub(1));
        let max_rows = self.spec_max_rows();
        Some(SpecLimits {
            kmax: kmax.min(ring_k),
            max_rows,
            draft_rows: draft_rows.into_iter().filter(|&r| r <= max_rows).collect(),
        })
    }

    /// Draft `k` tokens for every row with `k > 0` (each needs [`Self::spec_draft_ready`]),
    /// verify every row's span in one launch, commit each verdict and save the hidden each new
    /// pending token came from. A failure before the verify leaves every slot as it was.
    pub fn spec_tick(&mut self, rows: &[SpecRow], stop: &[u32], out: &mut SpecTick) -> Result<()> {
        out.verdicts.clear();
        out.committed = 0;
        if self.pipe_busy() {
            return Err(reject("decode pipeline in flight"));
        }
        let k_draft = rows.iter().map(|r| r.k).max().unwrap_or(0);
        out.stage.clear();
        for r in rows.iter().filter(|r| r.k > 0) {
            let row = out.stage.len() as u32;
            out.stage.push((r.slot, r.token, row));
        }
        out.draft_rows = out.stage.len();
        out.draft_k = k_draft;
        out.draft_ms = 0.0;
        out.drafts.clear();
        if !out.stage.is_empty() {
            let t = Instant::now();
            for i in 0..out.stage.len() {
                let (slot, _, row) = out.stage[i];
                self.spec_hidden_restore(slot, row)?;
            }
            let stage = std::mem::take(&mut out.stage);
            let staged = self.spec_draft_stage(&stage, k_draft);
            out.stage = stage;
            staged?;
            self.spec_draft_run(&mut out.drafts)?;
            out.draft_ms = t.elapsed().as_secs_f64() * 1e3;
        }
        out.tokens.clear();
        let mut d = 0usize;
        for r in rows {
            out.tokens.push(r.token);
            if r.k > 0 {
                let base = d * k_draft as usize;
                out.tokens
                    .extend_from_slice(&out.drafts[base..base + r.k as usize]);
                d += 1;
            }
        }
        let mut spans = Vec::with_capacity(rows.len());
        let mut at = 0usize;
        out.row0.clear();
        for r in rows {
            let n = r.k as usize + 1;
            out.row0.push(at as u32);
            spans.push(SpecSpan {
                slot: r.slot,
                tokens: &out.tokens[at..at + n],
            });
            at += n;
        }
        out.rows = at;
        let t = Instant::now();
        self.spec_verify(&spans, stop, &mut out.verdicts)?;
        out.verify_ms = t.elapsed().as_secs_f64() * 1e3;
        for (sp, v) in spans.iter().zip(&out.verdicts) {
            let pos = self.pos[sp.slot] + u32::from(v.n_emit);
            self.spec_hidden_save(sp.slot, v.hrow, pos)?;
            self.spec_commit(sp, v)?;
            out.committed += 1;
        }
        Ok(())
    }
}
