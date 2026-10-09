//! `plow.multimodal.v1`: how a text packet takes encoder rows (images, audio) in place of
//! placeholder tokens. Written by the emitter from the checkpoint and its processor config; read
//! by the runtime, which names no model.
//!
//! A prompt row whose token id has bit 31 set is a soft-token row: the LM's `Embed` gives it the
//! pad token's embedding (and per-layer input) and `MmRowsBf16` then replaces its hidden row with
//! the slab row the id names in `in.mm_table`. The runtime derives those ids from the media
//! content, so prefix-cache keys over token ids distinguish different images with the same text.
use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const SECTION: &str = "plow.multimodal.v1";
pub const VERSION: u32 = 1;
/// Soft-token ids carry this bit; the low 31 bits are a content hash.
pub const ROW_ID_BIT: u32 = 0x8000_0000;
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
            if !matches!(m.kind.as_str(), "image" | "audio") || m.packet.is_empty() || m.placeholder & ROW_ID_BIT != 0 {
                return Err(format!("invalid multimodal modality {:?}", m.kind));
            }
        }
        Ok(())
    }

    pub fn modality(&self, kind: &str) -> Option<&MmModality> {
        self.modalities.iter().find(|m| m.kind == kind)
    }
}
