//! Fish S2 depth-stage programs carried by a packet (`fish_depth.json` + an
//! [`crate::aux_program`] section, one program per decode rung width).
//!
//! After a backbone step the runtime stages the per-row sampling inputs, launches the program of
//! the rung's width, and reads `frame_codes[row][0..codebooks]`: column 0 is the backbone class
//! (`< semantic_tokens`: token `semantic_base + class`; `stop_class`: end of speech), columns
//! `1..` the depth codebooks. The program leaves the codes where the next step's frame embedding
//! reads them.
use serde::{Deserialize, Serialize};

pub const SECTION: &str = "fish_depth.json";
pub const PROGRAMS_SECTION: &str = "fish_depth.programs";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedDepth {
    pub version: u32,
    pub input_vocab: u32,
    pub backbone_logits: bool,
    /// The backbone's normalized hidden rows consumed by the depth programs.
    pub hidden: u16,
    /// A frame token used only for decode embeddings that drop the text-token row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_token: Option<u32>,
}

/// Per-row tensors the runtime stages (handles into the packet's tensor table).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tensors {
    /// f32 `[rows][4]`: temperature, top_p, RAS temperature, RAS top_p.
    pub params: u16,
    /// u32 `[rows]`; 0 leaves the row untouched (inactive).
    pub top_k: u16,
    /// u64 `[rows]`: the request's sampling seed.
    pub seed: u16,
    /// u32 `[rows]`: the frame index (RNG coordinate; RAS is off at frame 0).
    pub step: u16,
    /// u32 `[rows][ras_window]`: the request's RAS history (class + 1, 0 = empty).
    pub history: u16,
    /// u32 `[rows]`: the row of the backbone's final-norm activation that produced this frame.
    pub hidden_row: u16,
    /// u32 `[slots][codebooks]`: each slot's latest frame, read by the next decode step.
    pub frame_codes: u16,
    /// u32 `[ctx][codebooks]`: prompt frame rows the host writes before a prefill.
    pub prompt_codes: u16,
    /// u32 `[rows][codebooks]`: classes to force (u32::MAX = sample); Breeze only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub force: Option<u16>,
}

/// The sampling policy of the depth programs, which fixes how the runtime stages a row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Sampler {
    /// `SampleFishBf16`: params (T, top_p, RAS T, RAS top_p), `top_k` and `step` tensors,
    /// `history` = the RAS window.
    #[default]
    Fish,
    /// `SampleSmallF32`: params (T, top_p, repetition penalty, CFG scale), `top_k` = `step` = one
    /// `[rows][2]` (top_k, step) tensor, `history` = a `[rows][ras_window]` u32 class bitmap.
    Breeze,
    /// Maya 2 Native: the host samples cb0 from the backbone's logits (its controlled stop) and
    /// stages it in `history` (u32 `[rows]`); the programs sample codebooks `1..` like Breeze's
    /// depth levels (params, `[rows][2]` control) at [`Manifest::depth_sampling`].
    Maya2,
}

/// `top_k` word role bits of a [`Manifest::pair_cfg`] sampler.
pub const GUIDE: u32 = 1 << 31;
pub const FOLLOW: u32 = 1 << 30;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    /// Row width of each program, in [`PROGRAMS_SECTION`] order.
    pub rows: Vec<u32>,
    pub codebooks: u32,
    pub semantic_base: u32,
    pub semantic_tokens: u32,
    pub stop_class: u32,
    pub ras_window: u32,
    #[serde(default)]
    pub sampler: Sampler,
    /// The sampler takes CFG pairs: a guide row (`top_k` bit [`GUIDE`], scale in `params[3]`)
    /// samples `u + s (c - u)` over its own and the next row's logits and writes the code to both;
    /// the follower row (bit [`FOLLOW`]) samples nothing.
    #[serde(default)]
    pub pair_cfg: bool,
    /// Maya 2: the depth levels' fixed (temperature as f32 bits, top-k).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_sampling: Option<[u32; 2]>,
    /// Maya 2: each depth layer's per-slot memory constant, f32 `[slots][hidden]`, written when a
    /// request binds (the null memory's, or its speaker's).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slot_memory: Vec<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub packed: Option<PackedDepth>,
    /// The programs were emitted to be combined: a launch may cover its live rows with any of
    /// the declared capacities, several at once. Unset (packets emitted before it existed): each
    /// launch runs the one smallest program covering its highest live row.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub planned_rows: bool,
    pub tensors: Tensors,
}

impl Manifest {
    pub fn validate(&self, n_programs: usize, tensor_count: usize) -> Result<(), String> {
        let t = &self.tensors;
        let handles = [
            t.params,
            t.top_k,
            t.seed,
            t.step,
            t.history,
            t.hidden_row,
            t.frame_codes,
            t.prompt_codes,
        ];
        if self.version != VERSION {
            return Err(format!(
                "fish_depth.json version {} (want {VERSION})",
                self.version
            ));
        }
        if self.rows.len() != n_programs || self.rows.iter().any(|&r| r == 0) {
            return Err("fish_depth.json rows disagree with the programs".into());
        }
        if handles
            .iter()
            .chain(&t.force)
            .any(|&h| usize::from(h) >= tensor_count)
        {
            return Err("fish_depth.json names a tensor the packet lacks".into());
        }
        if self.codebooks == 0 || self.stop_class != self.semantic_tokens {
            return Err("fish_depth.json: codebooks > 0 and stop class = semantic_tokens".into());
        }
        if let Some(p) = &self.packed {
            if p.version != 1
                || p.input_vocab == 0
                || usize::from(p.hidden) >= tensor_count
                || p.decode_token.is_some_and(|token| {
                    token >= p.input_vocab
                        || token < self.semantic_base
                        || token - self.semantic_base >= self.semantic_tokens
                })
            {
                return Err("fish_depth.json: invalid packed depth contract".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        serde_json::from_str(
            r#"{
            "version":1,"rows":[1,8],"codebooks":8,"semantic_base":100,
            "semantic_tokens":10,"stop_class":10,"ras_window":0,
            "tensors":{"params":0,"top_k":1,"seed":2,"step":3,"history":4,
            "hidden_row":5,"frame_codes":6,"prompt_codes":7}
        }"#,
        )
        .unwrap()
    }

    #[test]
    fn packed_depth_contract_is_optional_and_rejects_unknown_or_invalid_mappings() {
        let mut m = manifest();
        assert_eq!(m.packed, None);
        assert!(!m.planned_rows, "a packet without the stamp is not planned");
        assert!(m.validate(2, 10).is_ok());
        m.packed = Some(PackedDepth {
            version: 1,
            input_vocab: 120,
            backbone_logits: true,
            hidden: 8,
            decode_token: Some(100),
        });
        assert!(m.validate(2, 10).is_ok());
        let encoded = serde_json::to_vec(&m).unwrap();
        assert_eq!(serde_json::from_slice::<Manifest>(&encoded).unwrap(), m);
        for p in [
            PackedDepth {
                version: 2,
                input_vocab: 120,
                backbone_logits: true,
                hidden: 8,
                decode_token: None,
            },
            PackedDepth {
                version: 1,
                input_vocab: 120,
                backbone_logits: true,
                hidden: 10,
                decode_token: None,
            },
            PackedDepth {
                version: 1,
                input_vocab: 120,
                backbone_logits: true,
                hidden: 8,
                decode_token: Some(99),
            },
            PackedDepth {
                version: 1,
                input_vocab: 120,
                backbone_logits: true,
                hidden: 8,
                decode_token: Some(110),
            },
        ] {
            m.packed = Some(p);
            assert!(m.validate(2, 10).is_err());
        }
    }
}
