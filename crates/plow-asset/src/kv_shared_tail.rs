//! KV-SHARED TAIL of a prefill program (Gemma-3n/4 E-series and any model whose trailing layers
//! read another layer's KV cache instead of writing their own).
//!
//! From the first layer of the trailing KV-shared run onwards, a prefill program writes no KV
//! cache: every op is row-local except attention, which reads caches the earlier layers already
//! wrote. A row's output there matters only if that row is SAMPLED, so a packed launch runs the
//! program up to `boundaries[program]`, gathers the sampled rows of the `carried` activations
//! into rows `0..n`, and runs the rest of a small bucket over those rows alone. The math per row
//! is unchanged.
//!
//! `boundaries[p]` is the first instruction of the tail in prefill program `p`; the runtime
//! requires it to open a segment. `carried` are the activations the tail reads that the prefix
//! wrote, each `[rows][row_bytes]` row-major.
use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
use serde::{Deserialize, Serialize};

pub const SECTION: &str = "kv_shared_tail";
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Carried {
    pub tensor: u16,
    pub row_bytes: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub version: u32,
    /// First layer of the trailing KV-shared run.
    pub first_layer: u32,
    /// Tail start instruction per prefill program, in program-table order.
    pub boundaries: Vec<u32>,
    pub carried: Vec<Carried>,
}

/// Tensor slots an op writes. Everything else it references is read.
fn written_slots(op: u16) -> &'static [usize] {
    match DevOp::from_u16(op) {
        Some(DevOp::FlashPrefill | DevOp::FlashPrefillFp8) => &[0, 1, 5],
        Some(DevOp::Nop) => &[],
        _ => &[0],
    }
}

/// The activations the tail `insts[boundary..]` reads before writing them, restricted to
/// `candidate` handles (row-major per-row activations). A tensor the tail first touches as a pure
/// write is scratch and is not carried.
pub fn carried_reads(
    insts: &[DevInst64],
    boundary: usize,
    candidate: impl Fn(u16) -> bool,
) -> Vec<u16> {
    let mut seen = std::collections::BTreeSet::new();
    let mut carried = Vec::new();
    for d in &insts[boundary..] {
        let written = written_slots(d.op);
        let mut first_touch = Vec::new();
        for (slot, &h) in d.t.iter().enumerate() {
            if h == TENSOR_NONE16 || !candidate(h) || seen.contains(&h) {
                continue;
            }
            let read = !written.contains(&slot)
                || d.t
                    .iter()
                    .enumerate()
                    .any(|(s, &o)| o == h && !written.contains(&s));
            first_touch.push((h, read));
        }
        for (h, read) in first_touch {
            if seen.insert(h) && read {
                carried.push(h);
            }
        }
    }
    carried.sort_unstable();
    carried
}

impl Manifest {
    pub fn validate(&self, prefill_programs: usize) -> Result<(), String> {
        if self.version != VERSION {
            return Err(format!("kv-shared tail: version {} unsupported", self.version));
        }
        if self.boundaries.len() != prefill_programs {
            return Err("kv-shared tail: one boundary per prefill program required".into());
        }
        if self.carried.is_empty() || self.carried.iter().any(|c| c.row_bytes == 0 || c.row_bytes % 16 != 0) {
            return Err("kv-shared tail: carried rows must be non-empty 16-byte multiples".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inst(op: DevOp, t: &[u16]) -> DevInst64 {
        let mut d = DevInst64 {
            op: op as u16,
            t: [TENSOR_NONE16; 8],
            ..Default::default()
        };
        d.t[..t.len()].copy_from_slice(t);
        d
    }

    #[test]
    fn scratch_is_not_carried_and_in_place_reads_are() {
        // 0: x, 1: hn, 2: qg (scratch), 3: at (flash output), 4: weight
        let insts = [
            inst(DevOp::RmsNorm, &[1, 0, 4]),
            inst(DevOp::Gemm, &[2, 1, 4]),
            inst(DevOp::FlashPrefill, &[5, 6, 2, 7, 8, 3]),
            inst(DevOp::Gemm, &[2, 3, 4]),
            inst(DevOp::NormResidual, &[0, 0, 2, 4]),
        ];
        let act = |h: u16| h <= 3;
        assert_eq!(carried_reads(&insts, 1, act), vec![0, 1]);
        assert_eq!(carried_reads(&insts, 2, act), vec![0, 2]);
    }
}
