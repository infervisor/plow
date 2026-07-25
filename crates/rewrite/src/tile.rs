//! Stage 3/4: tile decomposition + DMA/compute separation, driven by the
//! [`costmodel`].
//!
//! A fused GEMM/Linear node with concrete dims (M/N/K bound by the shape bucket)
//! is lowered to a tile: the cost model enumerates the architecture's legal tile
//! shapes ([`CostModel::candidates`]) and picks the lowest-cycle one
//! ([`CostModel::best_tile`] — the custom, hardware-aware extractor of design
//! §2.6), then it is split into DMA-in / compute / DMA-out nodes (Stage 4, §2.5)
//! annotated with the **paged-SRAM** footprint and the number of streaming
//! passes. The compute node and the operand DMA-ins form a **co-location group**
//! (they must land on the same SM — §6.1).
//!
//! Whether an over-budget tile is rejected or streamed is the caller's
//! [`SramPolicy`]; the budget itself can be kernel-dependent (the SM kernel
//! reserves SRAM — see [`CostModel::with_available`]).
//!
//! NB: this lowers a single *already-chosen* tile. Tile **exploration** — the
//! cost-driven choice among candidates — is [`crate::explore`], which keeps the
//! legality/cost in `costmodel` (Rust) but runs the argmin selection in egglog
//! datalog so it can later be made jointly with fusion / SRAM hand-off. For one
//! isolated GEMM the two agree; `lower_gemm` here uses the Rust argmin directly.

use costmodel::{CostModel, GemmShape, SramPolicy, TileShape};

#[derive(thiserror::Error, Debug)]
pub enum TileError {
    #[error("no tile candidate fits under the {0:?} SRAM policy")]
    NoCandidate(SramPolicy),
}

/// A Stage-4 tile node.
#[derive(Clone, Debug, PartialEq)]
pub enum TileNode {
    /// Stage an operand into SRAM (TMA / DMA engine).
    DmaIn { tensor: String },
    /// The matrix-engine tile-step (one `BM×BN×BK` mainloop body), streamed over
    /// `passes` SRAM passes.
    Compute { tile: TileShape, passes: u64 },
    /// Write the output tile back (TMA store / DMA).
    DmaOut { tensor: String },
}

/// The tile graph for one GEMM: an ordered node sequence plus the constraint
/// annotations the scheduler consumes (§3.1, §6.1).
#[derive(Clone, Debug)]
pub struct TileSeq {
    pub nodes: Vec<TileNode>,
    pub tile: TileShape,
    /// Estimated cycles for the whole GEMM with this tiling.
    pub cycles: u64,
    /// SRAM page footprint of the per-iteration working set.
    pub sram_pages: u64,
    /// Streaming passes (1 ⇒ working set is resident; >1 ⇒ K/BN-streamed).
    pub passes: u64,
    /// Node indices that must share an SM (the compute + its operand DMA-ins).
    pub colocation: Vec<usize>,
}

/// Lower one GEMM/Linear (`out = act · weightᵀ`, dims `g`) to a tile graph,
/// choosing the tile with the cost model under `policy`.
pub fn lower_gemm(
    cm: &CostModel,
    g: GemmShape,
    act: &str,
    weight: &str,
    out: &str,
    policy: SramPolicy,
) -> Result<TileSeq, TileError> {
    let (tile, cycles) = cm
        .best_tile(g, policy)
        .ok_or(TileError::NoCandidate(policy))?;

    let passes = cm.passes(tile);
    let sram_pages = cm.sram_pages(tile);

    // Stage-4 split: stage both operands, compute, write back.
    let nodes = vec![
        TileNode::DmaIn {
            tensor: weight.to_string(),
        },
        TileNode::DmaIn {
            tensor: act.to_string(),
        },
        TileNode::Compute { tile, passes },
        TileNode::DmaOut {
            tensor: out.to_string(),
        },
    ];
    // Operand DMA-ins (0,1) and the compute (2) co-locate on one SM.
    let colocation = vec![0, 1, 2];

    Ok(TileSeq {
        nodes,
        tile,
        cycles,
        sram_pages,
        passes,
        colocation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use costmodel::DEFAULT_PAGE_BYTES;

    fn h100() -> &'static costmodel::hwspec::GpuSpec {
        costmodel::hwspec::registry::lookup("H100 SXM5").unwrap()
    }

    #[test]
    fn tiles_a_gemm_into_dma_compute() {
        let spec = h100();
        let cm = CostModel::new(spec, DEFAULT_PAGE_BYTES);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };

        let seq = lower_gemm(&cm, g, "act", "w.weight", "out", SramPolicy::Stream).unwrap();

        // Cost model chose the lowest-cycle candidate.
        assert_eq!(
            (seq.tile, seq.cycles),
            cm.best_tile(g, SramPolicy::Stream).unwrap()
        );

        // Stage-4 structure: DmaIn(weight), DmaIn(act), Compute, DmaOut.
        assert_eq!(seq.nodes.len(), 4);
        assert!(matches!(seq.nodes[0], TileNode::DmaIn { .. }));
        assert!(matches!(seq.nodes[2], TileNode::Compute { .. }));
        assert!(matches!(seq.nodes[3], TileNode::DmaOut { .. }));

        // Paged-SRAM footprint recorded; co-location group = operands + compute.
        assert_eq!(seq.sram_pages, cm.sram_pages(seq.tile));
        assert_eq!(seq.colocation, vec![0, 1, 2]);
    }

    #[test]
    fn streaming_when_budget_is_tiny() {
        // Kernel reserves most SRAM: only 4 KiB / 1 page left for staging.
        let spec = h100();
        let cm = CostModel::with_available(spec, 4 * 1024, 4 * 1024);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };

        let seq = lower_gemm(&cm, g, "act", "w.weight", "out", SramPolicy::Stream).unwrap();
        assert!(seq.passes > 1, "tiny budget should force streaming passes");
    }

    #[test]
    fn filter_policy_can_exhaust_candidates() {
        // Budget too small for any candidate ⇒ Filter yields none, Stream still works.
        let spec = h100();
        let cm = CostModel::with_available(spec, 1024, 1024);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };

        assert!(lower_gemm(&cm, g, "a", "w", "o", SramPolicy::Filter).is_err());
        assert!(lower_gemm(&cm, g, "a", "w", "o", SramPolicy::Stream).is_ok());
    }
}
