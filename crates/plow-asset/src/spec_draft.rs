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

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        Manifest {
            version: VERSION,
            family: "gemma4_assistant".into(),
            kmax: 2,
            programs: [(1, 1), (1, 2), (4, 1), (4, 2)]
                .iter()
                .enumerate()
                .map(|(index, &(rows, k))| DraftProgram {
                    rows,
                    k,
                    index: index as u32,
                })
                .collect(),
            tensors: DraftTensors {
                tokens: vec![1, 2, 3],
                hidden_row: 4,
                position: 5,
                kv_len: 6,
                slot: 7,
                hidden: 8,
                hidden_rows: 128,
            },
            shard: "model-mtp.safetensors".into(),
            shard_sha256: String::new(),
        }
    }

    #[test]
    fn selects_the_narrowest_covering_program() {
        let m = manifest();
        m.validate(4, 9).unwrap();
        assert_eq!(m.select(1, 2).map(|p| p.index), Some(1));
        assert_eq!(m.select(3, 1).map(|p| p.index), Some(2));
        assert!(m.select(5, 1).is_none());
        assert!(m.select(1, 3).is_none());
    }

    #[test]
    fn refuses_bad_handles_and_program_indices() {
        let m = manifest();
        assert!(m.validate(3, 9).is_err());
        assert!(m.validate(4, 8).is_err());
        let mut short = manifest();
        short.tensors.tokens.pop();
        assert!(short.validate(4, 9).is_err());
    }
}
