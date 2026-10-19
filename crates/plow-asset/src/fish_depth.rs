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
}

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
            return Err(format!("fish_depth.json version {} (want {VERSION})", self.version));
        }
        if self.rows.len() != n_programs || self.rows.iter().any(|&r| r == 0) {
            return Err("fish_depth.json rows disagree with the programs".into());
        }
        if handles.iter().any(|&h| usize::from(h) >= tensor_count) {
            return Err("fish_depth.json names a tensor the packet lacks".into());
        }
        if self.codebooks == 0 || self.stop_class != self.semantic_tokens {
            return Err("fish_depth.json: codebooks > 0 and stop class = semantic_tokens".into());
        }
        Ok(())
    }
}
