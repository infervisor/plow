//! Speculative-decode DRAFT programs carried by a packet (`PLOW_EMIT_SPEC`).
//!
//! A draft program proposes `k` tokens for `rows` sequences in one launch on the decode
//! interpreter, reading the target's committed KV and the verify's final-norm rows. The packet
//! owns the model logic; the runtime stages the five input tensors, launches the
//! `(rows, k)` program from [`PROGRAMS_SECTION`] (an [`crate::aux_program`] section) and reads the
//! `k` output token tensors. A draft never writes the target KV cache.
use serde::{Deserialize, Serialize};

pub const SECTION: &str = "spec.json";
pub const PROGRAMS_SECTION: &str = "spec.draft.programs";
pub const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DraftProgram {
    pub rows: u32,
    pub k: u32,
    /// Index into the [`PROGRAMS_SECTION`] programs.
    pub index: u32,
}

/// Per-row input tensors (`[rows]` i32 each) and the token chain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DraftTensors {
    /// `tokens[0]` is the input token; `tokens[j]` (j >= 1) receives draft `j`.
    pub tokens: Vec<u16>,
    /// Row of `hidden` (the verify's final-norm activation) that produced `tokens[0]`.
    pub hidden_row: u16,
    /// Query position of the draft (the committed length minus one).
    pub position: u16,
    /// Committed KV length the draft attends.
    pub kv_len: u16,
    /// KV slot of the row's sequence.
    pub slot: u16,
    /// The target activation `hidden_row` indexes, and its row capacity.
    pub hidden: u16,
    pub hidden_rows: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    /// Draft family, e.g. `gemma4_assistant`.
    pub family: String,
    pub kmax: u32,
    pub programs: Vec<DraftProgram>,
    pub tensors: DraftTensors,
    /// sha256 of the draft weight shard the packet binds (`model-mtp.safetensors`).
    pub shard: String,
    pub shard_sha256: String,
}

impl Manifest {
    pub fn validate(&self, n_programs: usize, tensor_count: usize) -> Result<(), String> {
        let handle = |h: u16| usize::from(h) < tensor_count;
        let t = &self.tensors;
        if self.version != VERSION
            || self.kmax == 0
            || t.tokens.len() != self.kmax as usize + 1
            || !t.tokens.iter().all(|&h| handle(h))
            || ![t.hidden_row, t.position, t.kv_len, t.slot, t.hidden]
                .iter()
                .all(|&h| handle(h))
            || t.hidden_rows == 0
        {
            return Err("spec draft: manifest geometry".into());
        }
        for p in &self.programs {
            if p.rows == 0 || p.k == 0 || p.k > self.kmax || p.index as usize >= n_programs {
                return Err(format!("spec draft: program rows {} k {}", p.rows, p.k));
            }
        }
        Ok(())
    }

    /// The narrowest program covering `rows` sequences at exactly `k` drafts.
    pub fn select(&self, rows: usize, k: u32) -> Option<&DraftProgram> {
        self.programs
            .iter()
            .filter(|p| p.k == k && p.rows as usize >= rows)
            .min_by_key(|p| p.rows)
    }
}
