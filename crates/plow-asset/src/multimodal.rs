//! `plow.multimodal.v1`: how a text packet takes encoder rows (images, audio) in place of
//! placeholder tokens. Written by the emitter from the checkpoint and its processor config; read
//! by the runtime, which names no model.
//!
//! A prompt row whose token id has bit 31 set is a soft-token row: the LM's `Embed` gives it the
//! pad token's embedding (and per-layer input) and `MmRowsBf16` then replaces its hidden row with
//! the slab row the id names in `in.mm_table`. The runtime derives those ids from the media
//! content, so prefix-cache keys over token ids distinguish different images with the same text.
//!
//! A modality whose LM attends bidirectionally within each item (`attention =
//! "bidirectional_span"`) also sets [`SPAN_ID_BIT`] on its ids. Prefill derives each row's span
//! extent from the ids (`MmSpanExtent`), and the attention sites the emitter marked let a query
//! see the rest of its own run of span ids. A chunk end or sliding stage boundary must never fall
//! strictly inside such a run ([`span_safe_rows`]): the rows before it would miss the item's keys.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const SECTION: &str = "plow.multimodal.v1";
pub const VERSION: u32 = 1;
/// Soft-token ids carry this bit; the low 31 bits are a content hash.
pub const ROW_ID_BIT: u32 = 0x8000_0000;
/// Soft-token ids of a [`MediaAttention::BidirectionalSpan`] modality also carry this bit (no
/// other modality's do); the low 30 bits are then the content hash.
pub const SPAN_ID_BIT: u32 = 0x4000_0000;
/// The LM tensors the soft-token rows travel through.
pub const SLAB_TENSOR: &str = "in.mm_slab";
pub const TABLE_TENSOR: &str = "in.mm_table";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmContract {
    pub version: u32,
    /// LM hidden width of one encoder row (bf16 in the slab).
    pub hidden: u32,
    /// Token id soft-token rows embed as before their row is replaced.
    pub pad_token: u32,
    /// Rows of `in.mm_slab` (encoder rows of in-flight prompts).
    pub slab_rows: u32,
    /// Entries of `in.mm_table` (`(id, slab row)` pairs, a power of two).
    pub table_capacity: u32,
    pub modalities: Vec<MmModality>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MmModality {
    /// `image` or `audio`: the request content parts this modality serves.
    pub kind: String,
    /// Encoder sidecar beside the text packet (`forward.v1` pipeline `mm.encode`).
    pub packet: String,
    /// The token the chat template renders per item; replaced by `begin, rows.., end`.
    pub placeholder: u32,
    pub begin: Option<u32>,
    pub end: Option<u32>,
    /// Host preprocessing algorithm, parameterized by `parameters` (`*_f32` keys are f32 bits).
    pub processor: String,
    pub parameters: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "MediaAttention::is_causal")]
    pub attention: MediaAttention,
}

/// How the LM's prompt attention treats one item's soft-token rows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MediaAttention {
    #[default]
    Causal,
    /// Rows of one item also attend to the item's later rows, on the attention sites the emitter
    /// marked. Items are wrapped by `begin`/`end`, so a run of consecutive span ids is one item.
    BidirectionalSpan,
}

impl MediaAttention {
    pub fn is_causal(&self) -> bool {
        *self == MediaAttention::Causal
    }
}

/// `true` for a soft-token id of a bidirectional-span modality.
pub fn is_span_id(id: u32) -> bool {
    id & (ROW_ID_BIT | SPAN_ID_BIT) == ROW_ID_BIT | SPAN_ID_BIT
}

/// The rows of `prompt[start..start + rows]` one prefill may take so that neither its end nor an
/// internal sliding stage boundary (every `stage` rows from `start`; 0 = unstaged) falls strictly
/// inside a run of span ids. Cuts back to the start of the first run that would be split: 0 when
/// that run begins at `start` (the request then needs a share of at least the run).
pub fn span_safe_rows(prompt: &[u32], start: usize, rows: usize, stage: usize) -> usize {
    let end = (start + rows).min(prompt.len());
    let mut i = start;
    while i < end {
        if !is_span_id(prompt[i]) {
            i += 1;
            continue;
        }
        let s = i;
        while i < prompt.len() && is_span_id(prompt[i]) {
            i += 1;
        }
        let split_end = end < i;
        let split_stage = stage > 0 && start + ((s - start) / stage + 1) * stage < i.min(end);
        if split_end || split_stage {
            return s - start;
        }
    }
    end - start
}

/// As [`span_safe_rows`], but a run that begins at `start` and does not fit is taken whole when
/// it fits `limit` rows (a request must make progress past its own span).
pub fn span_chunk_rows(prompt: &[u32], start: usize, rows: usize, limit: usize, stage: usize) -> usize {
    match span_safe_rows(prompt, start, rows, stage) {
        0 => {
            let run = prompt[start..].iter().take_while(|&&id| is_span_id(id)).count();
            if run <= limit { run } else { 0 }
        }
        safe => safe,
    }
}

impl MmModality {
    pub fn param(&self, key: &str) -> Option<u64> {
        self.parameters.get(key).copied()
    }

    pub fn param_f32(&self, key: &str) -> Option<f32> {
        self.param(key).and_then(|v| u32::try_from(v).ok()).map(f32::from_bits)
    }
}

impl MmContract {
    pub fn validate(&self) -> Result<(), String> {
        if self.version != VERSION
            || self.hidden == 0
            || self.slab_rows == 0
            || !self.table_capacity.is_power_of_two()
            || self.table_capacity < self.slab_rows
            || self.modalities.is_empty()
        {
            return Err("invalid multimodal contract".into());
        }
        for m in &self.modalities {
            if !matches!(m.kind.as_str(), "image" | "audio")
                || m.packet.is_empty()
                || m.placeholder & ROW_ID_BIT != 0
                || (!m.attention.is_causal() && (m.begin.is_none() || m.end.is_none()))
            {
                return Err(format!("invalid multimodal modality {:?}", m.kind));
            }
        }
        Ok(())
    }

    pub fn modality(&self, kind: &str) -> Option<&MmModality> {
        self.modalities.iter().find(|m| m.kind == kind)
    }

    pub fn has_spans(&self) -> bool {
        self.modalities.iter().any(|m| !m.attention.is_causal())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: u32 = ROW_ID_BIT | SPAN_ID_BIT | 7;
    const A: u32 = ROW_ID_BIT | 7;

    #[test]
    fn span_ids_are_tagged_runs() {
        assert!(is_span_id(S));
        assert!(!is_span_id(A));
        assert!(!is_span_id(SPAN_ID_BIT));
    }

    #[test]
    fn chunks_never_end_inside_a_span() {
        // text 0..4, span 4..8, text 8..10
        let p = [1, 2, 3, 4, S, S, S, S, 5, 6];
        assert_eq!(span_safe_rows(&p, 0, 10, 0), 10);
        assert_eq!(span_safe_rows(&p, 0, 6, 0), 4, "cut back to the span start");
        assert_eq!(span_safe_rows(&p, 0, 8, 0), 8, "ending at the span end is safe");
        assert_eq!(span_safe_rows(&p, 0, 4, 0), 4);
        assert_eq!(span_safe_rows(&p, 4, 2, 0), 0, "a span at the start needs the whole run");
        assert_eq!(span_safe_rows(&p, 5, 5, 0), 5, "a prefix-cache hit inside the span continues it");
        // Causal media ids never constrain.
        let q = [1, A, A, A, 2];
        assert_eq!(span_safe_rows(&q, 0, 2, 0), 2);
        // A span at the start is taken whole when the request may hold it.
        assert_eq!(span_chunk_rows(&p, 4, 2, 4096, 0), 4);
        assert_eq!(span_chunk_rows(&p, 4, 2, 3, 0), 0);
        assert_eq!(span_chunk_rows(&p, 0, 6, 4096, 0), 4);
    }

    #[test]
    fn stage_boundaries_never_split_a_span() {
        // stage 4: boundaries at 4 and 8; span 3..6 crosses 4.
        let p = [1, 2, 3, S, S, S, 4, 5, 6, 7, 8, 9];
        assert_eq!(span_safe_rows(&p, 0, 12, 4), 3);
        // A span starting on the boundary is safe.
        let q = [1, 2, 3, 4, S, S, 5, 6, 7, 8];
        assert_eq!(span_safe_rows(&q, 0, 10, 4), 10);
        // Stage boundaries count from the chunk start, not position 0.
        assert_eq!(span_safe_rows(&p, 3, 9, 4), 9);
        assert_eq!(span_safe_rows(&p, 1, 11, 4), 2, "boundary at 5 splits 3..6");
        // A span past the chunk end does not matter.
        let r = [1, 2, 3, 4, 5, 6, S, S];
        assert_eq!(span_safe_rows(&r, 0, 6, 4), 6);
    }

    #[test]
    fn causal_contracts_serialize_without_the_attention_field() {
        let m = MmModality {
            kind: "audio".into(),
            packet: "a.pkt".into(),
            placeholder: 3,
            begin: None,
            end: None,
            processor: "p".into(),
            parameters: BTreeMap::new(),
            attention: MediaAttention::Causal,
        };
        assert!(!serde_json::to_string(&m).unwrap().contains("attention"));
        let span = MmModality { attention: MediaAttention::BidirectionalSpan, ..m };
        assert!(serde_json::to_string(&span).unwrap().contains("\"attention\":\"bidirectional_span\""));
    }
}
