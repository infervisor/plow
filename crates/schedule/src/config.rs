//! Scheduler configuration knobs.
//!
//! Granularity and DMA handling are *configuration*, not separate code paths:
//! `Granularity::PerOp` is the degenerate "op not tiled / pinned to one SM"
//! case, and `DmaModel` selects separate DMA tasks vs. megakernel-style folding.

/// How finely a compute op is expanded onto SMs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Granularity {
    /// One task per tile coordinate (`TileDomain::coords()`), spread across SMs.
    #[default]
    PerTile,
    /// One task per op — the untiled / single-SM case.
    PerOp,
    /// One task per **row-axis chunk**: the op's tile grid is partitioned into
    /// `k` contiguous groups along its token/row axis (Gemm `M`, Flash `seq_q`,
    /// Row axis-0), each group becoming one coarse task whose duration/bytes are
    /// the per-tile cost × the tiles it covers. This is the granularity the
    /// double-buffered prefill kernel consumes: chunk `c`'s consumer op overlaps
    /// chunk `c+1`'s producer op on a disjoint SM set, with 1:1 producer→consumer
    /// counter edges (see [`crate::expand::group_by_row_axis`] and
    /// [`crate::expand::expand_prefill_chunks`]). `k` is chosen cost-model-driven
    /// by `rewrite::explore::best_chunk_count`.
    PerChunk(u32),
}

/// Whether DMA is a separate schedulable task or folded into the compute task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DmaModel {
    /// `DmaIn | Compute | DmaOut` as distinct nodes on DMA engines (design IR).
    #[default]
    Separate,
    /// Load+compute+store fused into the compute task (megakernel style); the
    /// DMA bytes reserve HBM bandwidth during the compute interval.
    Collapsed,
}

/// Counter clustering granularity (Pass D).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ClusterMode {
    /// One counter per producer-op boundary (threshold = its tile count).
    /// Safe for small shapes but deadlocks at realistic tile counts unless the
    /// scheduler explicitly avoids placing consumers before coarse-counter
    /// producers on the same resource.
    Coarse,
    /// One counter per consumer tile on cross-op boundaries (threshold = that
    /// tile's producer in-degree from [`materialize_tile_deps`]). Achieves max
    /// pipelining: a consumer fires as soon as *its* specific producers finish.
    /// Falls back to coarse for boundaries without fine tile deps (all-to-all).
    #[default]
    Fine,
}

/// Full scheduler configuration.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub granularity: Granularity,
    pub dma_model: DmaModel,
    pub cluster: ClusterMode,
    /// Prefetch buffer depth: in-flight DMA-in tiles allowed per SM (Pass F).
    pub qsize: u32,
    /// DMA engines per unit. `0` (default) ⇒ use the GPU spec's `copy_engines`;
    /// a non-zero value overrides the hardware spec.
    pub dma_engines: u32,
    /// Node-level DPU (RDMA / collective) engines.
    pub dpu_engines: u32,
    /// Node-level host CPU threads.
    pub host_threads: u32,
    /// Interconnect bandwidth in GB/s for cross-unit transfers. `0.0` (default)
    /// ⇒ use the GPU spec's `interconnect` bandwidth; a non-zero value overrides it.
    pub link_gbps: f64,
    /// Maximum tiles per op before the expander auto-falls-back to `PerOp`
    /// granularity for that op. Caps extreme stream sizes (e.g. very large
    /// batch × long-sequence prefills). 0 = no limit (default).
    pub max_tiles_per_op: u32,
    /// Enable the Lean performance oracle. When `true`, the scheduler queries
    /// the `plow_verify` binary for provably-optimal decisions (counter
    /// granularity, prefetch depth, lower-bound certificates). Falls back to
    /// Rust heuristics if the binary is unavailable.
    pub lean_oracle: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            granularity: Granularity::PerTile,
            dma_model: DmaModel::Separate,
            cluster: ClusterMode::Fine,
            qsize: 3,
            dma_engines: 0, // 0 ⇒ use the GPU spec's copy_engines
            dpu_engines: 1,
            host_threads: 4,
            link_gbps: 0.0, // 0 ⇒ use the GPU spec's interconnect bandwidth
            max_tiles_per_op: 0,
            lean_oracle: false, // opt-in: Lean binary required
        }
    }
}
