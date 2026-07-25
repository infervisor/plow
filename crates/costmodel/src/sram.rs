//! Paged SRAM model.
//!
//! Per-SM shared memory is modeled as integer **pages** of `page_bytes`. A
//! tile's per-iteration working set (staged operands × buffering) rounds up to
//! whole pages. The budget is **kernel-dependent**: an SM compute kernel reserves
//! part of shared memory for its own buffers, so only `available` bytes remain
//! for operand staging — build the model with [`SramModel::with_available`].
//!
//! Whether an over-budget tile is rejected or merely streamed over multiple
//! passes is a caller policy ([`SramPolicy`]), not baked in here.

use crate::tile::TileShape;
use hwspec::GpuSpec;

/// How to treat a tile whose working set exceeds the SRAM page budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SramPolicy {
    /// Drop the candidate (hard fit constraint).
    Filter,
    /// Keep it; it streams over `loop_passes` passes (kernel self-coordinates).
    Stream,
}

#[derive(Clone, Copy, Debug)]
pub struct SramModel {
    pub page_bytes: u64,
    /// Usable pages for tile staging (after any kernel reservation).
    pub pages_per_sm: u64,
}

impl SramModel {
    /// Full per-SM shared memory available for staging.
    pub fn from_spec(spec: &GpuSpec, page_bytes: u64) -> SramModel {
        SramModel::with_available(spec.sm.shared_mem.0, page_bytes)
    }

    /// Kernel-dependent budget: only `available_bytes` of shared memory is usable
    /// for operand staging (the SM kernel reserves the rest).
    pub fn with_available(available_bytes: u64, page_bytes: u64) -> SramModel {
        SramModel {
            page_bytes,
            pages_per_sm: available_bytes / page_bytes,
        }
    }

    pub fn pages(&self, bytes: u64) -> u64 {
        bytes.div_ceil(self.page_bytes)
    }

    /// Bytes staged per mainloop iteration: A-tile + B-tile, times buffering
    /// (e.g. 2 for double-buffering). The fp32 accumulator lives in registers.
    pub fn working_set_bytes(tile: TileShape, elem_bytes: u64, buffering: u64) -> u64 {
        let a = (tile.bm * tile.bk) as u64;
        let b = (tile.bk * tile.bn) as u64;
        (a + b) * elem_bytes * buffering
    }

    pub fn working_set_pages(&self, tile: TileShape, elem_bytes: u64, buffering: u64) -> u64 {
        self.pages(Self::working_set_bytes(tile, elem_bytes, buffering))
    }

    /// Number of streaming passes a tile needs: 1 if its working set fits,
    /// otherwise it is split (chunk BN/BK) until each pass fits.
    pub fn loop_passes(&self, working_set_pages: u64) -> u64 {
        if working_set_pages <= self.pages_per_sm.max(1) {
            1
        } else {
            working_set_pages.div_ceil(self.pages_per_sm.max(1))
        }
    }

    /// Does the tile's working set fit a single pass?
    pub fn fits(&self, tile: TileShape, elem_bytes: u64, buffering: u64) -> bool {
        self.working_set_pages(tile, elem_bytes, buffering) <= self.pages_per_sm.max(1)
    }
}
