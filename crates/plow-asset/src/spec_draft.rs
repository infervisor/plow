//! Speculative-decode DRAFT programs carried by a packet (`PLOW_EMIT_SPEC`).
//!
//! A draft program proposes `k` tokens for `rows` sequences in one launch on the decode
//! interpreter. A chain drafter (`gemma4_assistant`) reads the target's committed KV and the
//! verify's final-norm rows; a block drafter ([`BlockDraft`], `gemma4_dspark`) runs one
//! `size`-row block per sequence over its own KV layers, which the target programs fill
//! (`crate::spec_context`). The packet owns the model logic; the runtime stages the input tensors,
//! launches the `(rows, k)` program from [`PROGRAMS_SECTION`] (an [`crate::aux_program`] section)
//! and reads the `k` output token tensors. A draft never writes the target KV cache.
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

/// Per-row input tensors (`[rows]` i32 each, `[size * rows]` for a block drafter's `position`,
/// `kv_len` and `slot`) and the token chain.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DraftTensors {
    /// `tokens[0]` is the input token; `tokens[j]` (j >= 1) receives draft `j`.
    pub tokens: Vec<u16>,
    /// Chain drafters: row of `hidden` (the verify's final-norm activation) that produced
    /// `tokens[0]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden_row: Option<u16>,
    /// Query position of the draft (chain: the committed length minus one; block: per block row).
    pub position: u16,
    /// KV length the draft attends (chain: committed; block: committed + `size`, every row).
    pub kv_len: u16,
    /// KV slot of the row's sequence (block: per block row, the slot its attention reads).
    pub slot: u16,
    /// Chain drafters: the target activation `hidden_row` indexes, and its row capacity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hidden: Option<u16>,
    #[serde(default)]
    pub hidden_rows: u32,
}

/// A block drafter: one non-causal block of `size` rows per sequence, ids
/// `[tokens[0], mask_token x (size - 1)]` at positions `c..c+size` (`c` = committed rows), its own
/// K/V stored at those positions before attending; `k` sequential Markov stages turn the block's
/// logits into `tokens[1..=k]`. Block rows are request-major (row `r * size + i`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockDraft {
    pub size: u32,
    pub mask_token: u32,
    /// `[size * rows]` i32 block ids.
    pub ids: u16,
    /// `[size * rows]` i32 KV store slot per block row; negative = no store (padding rows).
    pub store_slot: u16,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block: Option<BlockDraft>,
    /// sha256 of the draft weight shard the packet binds (`model-mtp.safetensors`).
    pub shard: String,
    pub shard_sha256: String,
}

impl Manifest {
    pub fn validate(&self, n_programs: usize, tensor_count: usize) -> Result<(), String> {
        let handle = |h: u16| usize::from(h) < tensor_count;
        let t = &self.tensors;
        let inputs_ok = match &self.block {
            None => {
                matches!((t.hidden_row, t.hidden), (Some(r), Some(h)) if handle(r) && handle(h))
                    && t.hidden_rows != 0
            }
            Some(b) => {
                t.hidden_row.is_none()
                    && t.hidden.is_none()
                    && b.size != 0
                    && self.kmax <= b.size
                    && handle(b.ids)
                    && handle(b.store_slot)
            }
        };
        if self.version != VERSION
            || self.kmax == 0
            || t.tokens.len() != self.kmax as usize + 1
            || !t.tokens.iter().all(|&h| handle(h))
            || ![t.position, t.kv_len, t.slot].iter().all(|&h| handle(h))
            || !inputs_ok
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
                hidden_row: Some(4),
                position: 5,
                kv_len: 6,
                slot: 7,
                hidden: Some(8),
                hidden_rows: 128,
            },
            block: None,
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

    #[test]
    fn chain_json_is_unchanged_and_block_drops_the_hidden() {
        let m = manifest();
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains(r#""hidden_row":4"#) && !json.contains("block"));
        assert_eq!(serde_json::from_str::<Manifest>(&json).unwrap(), m);
        let mut b = manifest();
        b.tensors.hidden_row = None;
        b.tensors.hidden = None;
        b.tensors.hidden_rows = 0;
        assert!(
            b.validate(4, 11).is_err(),
            "a chain drafter needs its hidden"
        );
        b.block = Some(BlockDraft {
            size: 7,
            mask_token: 4,
            ids: 9,
            store_slot: 10,
        });
        b.validate(4, 11).unwrap();
        assert!(b.validate(4, 10).is_err());
        b.block.as_mut().unwrap().size = 1;
        assert!(b.validate(4, 11).is_err(), "kmax exceeds the block");
    }
}
