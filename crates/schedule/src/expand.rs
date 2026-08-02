//! Pass 0 — expand the op-level [`TileGraph`] into the scheduler's per-tile
//! [`TaskGraph`], honoring [`Granularity`] and [`DmaModel`].
//!
//! Reuses the rewrite crate's tiling vocabulary: `TileDomain::coords()` gives the
//! tile grid, [`footprints`] gives each tile's read/write byte volumes, and
//! [`materialize_tile_deps`] turns a compact `TileDep` into explicit
//! producer-tile → consumer-tile edges. Cross-op ordering that isn't row-coupled
//! falls back to a coarse all-tiles→all-tiles edge so the schedule stays correct.

use crate::config::{ClusterMode, Config, DmaModel, Granularity};
use crate::interval::Cycle;
use crate::passes::{build_counters, Counter};
use costmodel::{cost, MmaDtype, Soc, UnitId};
use rewrite::{
    footprints, materialize_tile_deps, Compute, ConstraintSet, GraphNode, HandoffKind, OpIo,
    OpKind, TensorSlice, TileDomain, TileGraph,
};
use std::collections::{HashMap, HashSet};

pub type TaskId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskKind {
    DmaIn,
    Compute,
    DmaOut,
    /// CPU-side coordination / routing / staging (host thread pool). Not emitted
    /// by the current expansion, but a first-class class the scheduler routes.
    Host,
}

/// One schedulable unit of work.
#[derive(Clone, Debug)]
pub struct Task {
    pub node: usize,
    pub op: String,
    pub unit: UnitId,
    pub kind: TaskKind,
    pub coord: Vec<i64>,
    pub dur: Cycle,
    pub bytes: u64,
    /// Whole-tensor byte size of `tensor` (op-level, all tiles), independent of
    /// this task's per-tile slice `bytes`. 0 when `tensor` is `None` or unknown.
    /// The memory planner sizes buffers from this, not per-tile `bytes`.
    pub tensor_bytes: u64,
    /// Compute working-set pages (transient, during the compute interval; SM is
    /// exclusive so only one is live at a time). Capped at the per-SM budget.
    pub sram_pages: u64,
    /// Resident output-tile pages, held from production until the last consumer
    /// (the live interval the per-page allocator packs).
    pub out_pages: u64,
    /// Tensor-Memory columns the tile's MMA accumulator occupies (Blackwell
    /// `tcgen05`; 0 on other archs or non-matmul tiles).
    pub tmem_cols: u64,
    pub tensor: Option<String>,
    /// A cross-unit transfer (routed to a DPU over the interconnect).
    pub cross_unit: bool,
}

/// The expanded per-tile dependency graph.
#[derive(Clone, Debug, Default)]
pub struct TaskGraph {
    pub tasks: Vec<Task>,
    pub edges: Vec<(TaskId, TaskId)>,
    /// Per-unit SRAM page-pool capacity (pages per SM), captured from the
    /// machine at expansion so post-schedule passes (e.g. prefetch hoisting)
    /// can re-run slot allocation without the `Machine`.
    pub pages_per_sm: Vec<u64>,
    /// Per-unit TMEM column capacity per SM (0 where TMEM is absent).
    pub tmem_cols_per_sm: Vec<u64>,
}

impl TaskGraph {
    fn push(&mut self, t: Task) -> TaskId {
        self.tasks.push(t);
        self.tasks.len() - 1
    }
}

/// Per-compute-node info gathered from the op-level graph.
struct NodeInfo<'a> {
    op: String,
    unit: UnitId,
    tile: Compute,
    passes: u64,
    sram_pages: u64,
    kind: OpKind,
    inputs: &'a [String],
    output: &'a str,
    /// Boundary DMAs folded into the kernel by [`rewrite::collapse`] — the kernel
    /// issues its own load/store, so no separate DMA task is emitted (their bytes
    /// fold into the compute's bandwidth reservation).
    inline_in: &'a [String],
    inline_out: bool,
    /// Bytes per activation element (compute precision, used for A-operand & outputs).
    activation_elem: u64,
    /// Bytes per weight element (may differ for block-quant, used for B-operand).
    weight_elem: u64,
}

pub fn expand(
    soc: &Soc,
    machine: &crate::machine::Machine,
    g: &TileGraph,
    cons: &ConstraintSet,
    cfg: &Config,
) -> TaskGraph {
    // --- maps over the op-level graph ---------------------------------------
    // producer compute of each DmaOut node, consumer compute of each DmaIn node.
    let mut producer_of: HashMap<usize, usize> = HashMap::new();
    let mut consumer_of: HashMap<usize, usize> = HashMap::new();
    // (compute node, input tensor) -> resident? (true = SRAM hand-off, no DRAM DMA)
    let mut resident_in: HashMap<(usize, String), bool> = HashMap::new();
    for &(a, b) in &g.edges {
        match (&g.nodes[a], &g.nodes[b]) {
            (GraphNode::Compute { .. }, GraphNode::DmaOut { .. }) => {
                producer_of.insert(b, a);
            }
            (GraphNode::DmaIn { tensor, resident }, GraphNode::Compute { .. }) => {
                consumer_of.insert(a, b);
                resident_in.insert((b, tensor.clone()), *resident);
            }
            _ => {}
        }
    }
    // resident flag of each compute node's output DmaOut.
    let mut output_resident: HashMap<usize, bool> = HashMap::new();
    for (dmaout, &c) in &producer_of {
        if let GraphNode::DmaOut { resident, .. } = &g.nodes[*dmaout] {
            output_resident.insert(c, *resident);
        }
    }

    // --- expand each compute node -------------------------------------------
    let mut tg = TaskGraph::default();
    tg.pages_per_sm = machine.units.iter().map(|u| u.pages_per_sm).collect();
    tg.tmem_cols_per_sm = machine.units.iter().map(|u| u.tmem_cols_per_sm).collect();
    let mut node_tasks: HashMap<usize, Vec<TaskId>> = HashMap::new(); // compute node -> its compute tasks
    let mut tile_task: HashMap<(usize, Vec<i64>), TaskId> = HashMap::new();
    // PerChunk only: compute task -> its row-element interval [start,end) on the
    // token axis, used to form 1:1 chunk-level cross-op edges by range overlap.
    let mut chunk_range: HashMap<TaskId, (i64, i64)> = HashMap::new();

    for (nid, n) in g.nodes.iter().enumerate() {
        let GraphNode::Compute {
            op,
            kind: tile,
            passes,
            sram_pages,
            inline_in,
            inline_out,
        } = n
        else {
            continue;
        };
        let desc = &cons.op_io[&nid];
        let info = NodeInfo {
            op: op.clone(),
            unit: cons.placement[&nid],
            tile: *tile,
            passes: *passes,
            sram_pages: *sram_pages,
            kind: desc.kind,
            inputs: &desc.inputs,
            output: &desc.output,
            inline_in,
            inline_out: *inline_out,
            activation_elem: desc.activation_elem,
            weight_elem: desc.weight_elem,
        };
        let domain = cons.domains[&nid];
        let all_coords = domain.coords();
        // Auto-fallback: if the tile count exceeds the threshold, treat this op
        // as PerOp to cap stream size (prevents multi-GB streams from extreme
        // batch × sequence shapes).
        let fallback_to_per_op =
            cfg.max_tiles_per_op > 0 && all_coords.len() > cfg.max_tiles_per_op as usize;
        let n_tiles = all_coords.len().max(1) as u64;
        let per_op = matches!(cfg.granularity, Granularity::PerOp) || fallback_to_per_op;

        // Work units to emit for this op. PerTile: one per coord (1 tile each).
        // PerOp: one for the whole op (n_tiles, op-level byte formulas). PerChunk:
        // one per row-axis chunk (tiles = tiles in that chunk, per-tile × tiles
        // bytes/duration).
        let work: Vec<WorkUnit> = if per_op {
            vec![WorkUnit {
                coord: all_coords.first().cloned().unwrap_or_default(),
                tiles: n_tiles,
                range: None,
            }]
        } else if let Granularity::PerChunk(k) = cfg.granularity {
            group_by_row_axis(&domain, &all_coords, k)
                .into_iter()
                .filter(|c| !c.coords.is_empty())
                .map(|c| WorkUnit {
                    coord: c.coords[0].clone(),
                    tiles: c.coords.len() as u64,
                    range: Some(c.range),
                })
                .collect()
        } else {
            all_coords
                .iter()
                .map(|c| WorkUnit {
                    coord: c.clone(),
                    tiles: 1,
                    range: None,
                })
                .collect()
        };

        for wu in &work {
            let coord = &wu.coord;
            // one task stands in for `wu.tiles` tiles: scale per-tile cost/bytes.
            let mult = wu.tiles;
            // compute task duration
            let dur = compute_cycles(soc, machine, &info, coord, info.activation_elem)
                .saturating_mul(mult);
            // footprints for this tile (PerTile) or the op (PerOp uses op-level bytes)
            let io = OpIo {
                inputs: info.inputs,
                output: info.output,
            };
            let fp = footprints(&info.kind, &info.tile, &io, coord);

            let mut collapsed_bytes = 0u64;
            let mut dmains: Vec<TaskId> = Vec::new();

            // non-resident inputs -> DRAM DMA-in (or folded if Collapsed)
            for (i, tname) in info.inputs.iter().enumerate() {
                let resident = resident_in
                    .get(&(nid, tname.clone()))
                    .copied()
                    .unwrap_or(false);
                if resident {
                    continue; // SRAM hand-off: ordering comes from a cross-op edge
                }
                let in_elem =
                    op_elem_for_input(&info.kind, i, info.activation_elem, info.weight_elem);
                let bytes = if per_op {
                    op_in_bytes(&info.kind, i, in_elem)
                } else {
                    fp.reads
                        .get(i)
                        .and_then(|s| slice_bytes(s, in_elem))
                        .map(|b| b.saturating_mul(mult))
                        .unwrap_or_else(|| op_in_bytes(&info.kind, i, in_elem))
                };
                if bytes == 0 {
                    continue;
                }
                // dma-fold: the kernel issues this load itself (no separate DMA
                // task); its bytes fold into the compute's bandwidth reservation.
                if info.inline_in.iter().any(|t| t == tname) {
                    collapsed_bytes += bytes;
                    continue;
                }
                match cfg.dma_model {
                    DmaModel::Collapsed => collapsed_bytes += bytes,
                    DmaModel::Separate => {
                        let d = tg.push(Task {
                            node: nid,
                            op: format!("{}.in[{i}]", info.op),
                            unit: info.unit,
                            kind: TaskKind::DmaIn,
                            coord: coord.clone(),
                            dur: machine.hbm_cycles(info.unit, bytes),
                            bytes,
                            tensor_bytes: op_in_bytes(&info.kind, i, in_elem),
                            sram_pages: 0,
                            out_pages: 0,
                            tmem_cols: 0,
                            tensor: Some(tname.clone()),
                            cross_unit: false,
                        });
                        dmains.push(d);
                    }
                }
            }

            // output-tile byte volume (and its resident page footprint).
            // Outputs are always in compute (activation) precision.
            let out_bytes = if per_op {
                op_out_bytes(&info.kind, info.activation_elem)
            } else {
                slice_bytes(&fp.write, info.activation_elem)
                    .map(|b| b.saturating_mul(mult))
                    .unwrap_or_else(|| op_out_bytes(&info.kind, info.activation_elem))
            };
            let page_bytes = soc.unit(info.unit).cm.sram.page_bytes.max(1);
            let out_pages = out_bytes.div_ceil(page_bytes);
            // MMA accumulator in TMEM (Blackwell only): a column holds 128 f32.
            let tmem_cols = if machine.unit(info.unit).tmem_cols_per_sm > 0 {
                accumulator_cols(&info.kind, &info.tile)
            } else {
                0
            };

            // In Collapsed mode, fold the output DMA bytes into the compute
            // task's bandwidth so the scheduler's capacity check is accurate.
            let out_resident = output_resident.get(&nid).copied().unwrap_or(false);
            // Fold the store into the kernel epilogue when collapsed-mode or when
            // the collapse stage marked this output inline.
            let fold_out = matches!(cfg.dma_model, DmaModel::Collapsed) || info.inline_out;
            if !out_resident && out_bytes > 0 && fold_out {
                collapsed_bytes += out_bytes;
            }

            // the compute task. A streamed tile's working set may exceed the
            // per-SM budget; its *resident* footprint is capped at the budget
            // (it streams in budget-sized chunks over `passes`).
            let resident_pages = info.sram_pages.min(machine.unit(info.unit).pages_per_sm);
            let ct = tg.push(Task {
                node: nid,
                op: info.op.clone(),
                unit: info.unit,
                kind: TaskKind::Compute,
                coord: coord.clone(),
                dur: dur.max(1),
                bytes: collapsed_bytes,
                tensor_bytes: 0,
                sram_pages: resident_pages,
                out_pages,
                tmem_cols,
                tensor: None,
                cross_unit: false,
            });
            for d in dmains {
                tg.edges.push((d, ct));
            }
            node_tasks.entry(nid).or_default().push(ct);
            tile_task.insert((nid, coord.clone()), ct);
            if let Some(r) = wu.range {
                chunk_range.insert(ct, r);
            }

            // output -> DMA-out (unless SRAM-resident for a consumer, or folded
            // into the kernel epilogue).
            if !out_resident
                && out_bytes > 0
                && matches!(cfg.dma_model, DmaModel::Separate)
                && !info.inline_out
            {
                let o = tg.push(Task {
                    node: nid,
                    op: format!("{}.out", info.op),
                    unit: info.unit,
                    kind: TaskKind::DmaOut,
                    coord: coord.clone(),
                    dur: machine.hbm_cycles(info.unit, out_bytes),
                    bytes: out_bytes,
                    tensor_bytes: op_out_bytes(&info.kind, info.activation_elem),
                    sram_pages: 0,
                    out_pages: 0,
                    tmem_cols: 0,
                    tensor: Some(info.output.to_string()),
                    cross_unit: false,
                });
                tg.edges.push((ct, o));
            }
        }
    }

    // --- cross-op edges from hand-offs --------------------------------------
    for h in &cons.handoffs {
        let Some(&pc) = producer_of.get(&h.producer_dma_out) else {
            continue;
        };
        let Some(&cc) = consumer_of.get(&h.consumer_dma_in) else {
            continue;
        };
        let prod_tiles = node_tasks.get(&pc).cloned().unwrap_or_default();
        let cons_tiles = node_tasks.get(&cc).cloned().unwrap_or_default();
        if prod_tiles.is_empty() || cons_tiles.is_empty() {
            continue;
        }

        if h.cross_unit {
            // A `Barrier` is a fence over unified memory: the consumer reads the
            // producer's bytes in place, so there is NO data-transfer task — only
            // an ordering edge producer→consumer. Emitting a DPU transfer here (as
            // `P2p`/`Rdma` need) would fabricate link traffic the fence never moves.
            if h.kind == HandoffKind::Barrier {
                for &p in &prod_tiles {
                    for &c in &cons_tiles {
                        tg.edges.push((p, c));
                    }
                }
                continue;
            }
            // P2P / RDMA: mediate with one interconnect transfer (routed to a DPU).
            let pd = &cons.op_io[&pc];
            let bytes = op_out_bytes(&pd.kind, pd.activation_elem);
            let t = tg.push(Task {
                node: pc,
                op: format!(
                    "{}->{} xfer",
                    cons.op_io[&pc].output, cons.op_io[&cc].output
                ),
                unit: cons.placement[&cc],
                kind: TaskKind::DmaIn,
                coord: vec![],
                dur: machine.link_cycles(bytes).max(1),
                bytes,
                tensor_bytes: bytes,
                sram_pages: 0,
                out_pages: 0,
                tmem_cols: 0,
                tensor: Some(cons.op_io[&pc].output.clone()),
                cross_unit: true,
            });
            for &p in &prod_tiles {
                tg.edges.push((p, t));
            }
            for &c in &cons_tiles {
                tg.edges.push((t, c));
            }
            continue;
        }

        // Same-unit: fine (row-coupled) edges if a TileDep exists, else coarse.
        let dep = cons
            .tile_deps
            .iter()
            .find(|d| d.producer == pc && d.consumer == cc);

        // PerChunk: producer chunk → consumer chunk 1:1 (or fan) by row-range
        // interval overlap on the shared token axis. This is the producer-consumer
        // pipeline the double-buffered kernel consumes. Only forms on a real
        // row-coupled boundary (a `TileDep`); all-to-all boundaries (e.g. a K
        // reduction with no row coupling) fall through to the coarse case, since
        // there chunks are not independently pipelinable.
        if matches!(cfg.granularity, Granularity::PerChunk(_)) && dep.is_some() {
            for &p in &prod_tiles {
                for &c in &cons_tiles {
                    if let (Some(&(a0, a1)), Some(&(b0, b1))) =
                        (chunk_range.get(&p), chunk_range.get(&c))
                    {
                        if a0 < b1 && b0 < a1 {
                            tg.edges.push((p, c));
                        }
                    }
                }
            }
            continue;
        }

        let fine = !matches!(cfg.granularity, Granularity::PerOp)
            && dep.is_some()
            && cons.domains.contains_key(&pc)
            && cons.domains.contains_key(&cc);
        if fine {
            let dep = dep.unwrap();
            for e in materialize_tile_deps(&cons.domains[&pc], &cons.domains[&cc], &dep.dep) {
                if let (Some(&p), Some(&c)) = (
                    tile_task.get(&(pc, e.producer_coord)),
                    tile_task.get(&(cc, e.consumer_coord)),
                ) {
                    tg.edges.push((p, c));
                }
            }
        } else {
            for &p in &prod_tiles {
                for &c in &cons_tiles {
                    tg.edges.push((p, c));
                }
            }
        }
    }

    tg
}

// --- PerChunk row-axis grouping ---------------------------------------------

/// One work unit the expander emits for a compute node.
struct WorkUnit {
    /// Representative tile coordinate (the first tile of the group).
    coord: Vec<i64>,
    /// How many tiles this unit stands for (scales per-tile cost/bytes).
    tiles: u64,
    /// PerChunk only: the row-element interval `[start,end)` on the token axis
    /// this chunk covers (for chunk-level cross-op edges).
    range: Option<(i64, i64)>,
}

/// A row-axis chunk: the tile coords whose row-axis grid index falls in one
/// contiguous partition of `[0, n_row_tiles)`, plus the row-element interval it
/// spans on the shared token axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RowChunk {
    pub coords: Vec<Vec<i64>>,
    /// Row-element interval `[start,end)` on the token axis (Gemm `M`, Flash
    /// `seq_q`, Row axis-0). `(0,0)` for an untiled (Layout) domain.
    pub range: (i64, i64),
}

/// The token/row extent of a tile domain (Gemm `M`, Flash `seq_q`, Row `rows`).
fn row_extent(d: &TileDomain) -> i64 {
    match *d {
        TileDomain::Row { rows, .. } => rows,
        TileDomain::Gemm { m, .. } => m,
        TileDomain::Flash { seq_q, .. } => seq_q,
        TileDomain::Layout => 0,
    }
}

/// Partition an op's tile `coords` into `n_chunks` contiguous groups along the
/// domain's **row axis** (the tensor's token axis: Gemm `M` = grid axis 0, Flash
/// `seq_q` = grid axis 1, Row = grid axis 0). Every tile sharing a row-axis grid
/// index lands in the same chunk (for a GEMM, this keeps all `N`-tiles of one
/// `M`-row-block together). Groups are the even contiguous split
/// `[b·n/k, (b+1)·n/k)` of the `n = ⌈rows/br⌉` row-tiles into `k` buckets, so
/// `n_chunks` above the row-tile count yields empty trailing chunks. An untiled
/// (Layout) domain has no row axis → one chunk holding all coords.
///
/// This is the *merge* the double-buffered prefill kernel consumes: chunk `c`'s
/// coords are the tiles the kernel runs on SM-set `c`, and the chunk's `range`
/// is what [`expand`] overlaps against the consumer op's chunk ranges to form the
/// 1:1 producer→consumer pipeline edges.
pub fn group_by_row_axis(domain: &TileDomain, coords: &[Vec<i64>], n_chunks: u32) -> Vec<RowChunk> {
    let Some((axis, block)) = domain.row_axis() else {
        return vec![RowChunk {
            coords: coords.to_vec(),
            range: (0, 0),
        }];
    };
    let n_rows = coords
        .iter()
        .map(|c| c.get(axis).copied().unwrap_or(0) + 1)
        .max()
        .unwrap_or(0);
    let k = (n_chunks.max(1) as i64).min(n_rows.max(1));
    let extent = row_extent(domain);
    let split = |b: i64| (b * n_rows) / k;
    (0..k)
        .map(|b| {
            let (rt0, rt1) = (split(b), split(b + 1));
            let cs: Vec<Vec<i64>> = coords
                .iter()
                .filter(|c| {
                    let r = c.get(axis).copied().unwrap_or(0);
                    r >= rt0 && r < rt1
                })
                .cloned()
                .collect();
            RowChunk {
                coords: cs,
                range: ((rt0 * block).min(extent), (rt1 * block).min(extent)),
            }
        })
        .collect()
}

// --- PerChunk pipeline API (consumed by the double-buffered prefill kernel) ---

/// How chunk tasks map to SM sets — the A/B knob the chunk-pipeline kernel tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChunkPlacementPolicy {
    /// **Static affinity**: chunk `c`'s *entire* producer→consumer op chain pins
    /// to the same SM set `[c·n_cu/k, (c+1)·n_cu/k)`. The producer's output stays
    /// resident in that partition's L2 slice, so the consumer reads it hot — no
    /// HBM round-trip across the op boundary. This is the chunk thesis: static +
    /// L2-locality, reversing the decode global-queue default.
    #[default]
    StaticColocated,
    /// **Global queue**: chunks may run on any SM (`[0, n_cu)`) and work-steal.
    /// No cross-op L2 residency guarantee — a scattered consumer re-reads the
    /// producer output from HBM. The A/B baseline.
    GlobalQueue,
}

/// One chunk task's SM-set placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkPlacement {
    /// Task index into [`PerChunkPlan::tasks`].
    pub task: TaskId,
    /// The compute node this task belongs to.
    pub node: usize,
    /// Chunk index `c` (row-axis partition), stable across the op chain.
    pub chunk: u32,
    /// SM (CU) half-open range `[lo, hi)` this chunk runs on.
    pub cu_range: (u32, u32),
}

/// The PerChunk task graph plus everything the double-buffered prefill kernel
/// needs: per-chunk SM-set placement and the 1:1 producer→consumer counter edges.
#[derive(Clone, Debug)]
pub struct PerChunkPlan {
    /// The expanded PerChunk(k) task graph (chunk tasks + chunk-level edges).
    pub tasks: TaskGraph,
    /// Per compute chunk task → its SM-set placement (see [`ChunkPlacementPolicy`]).
    pub placement: Vec<ChunkPlacement>,
    /// Fine (per-consumer-chunk) counters: threshold = the producer chunks
    /// feeding that consumer chunk (1 for matched chunk counts). **Never a global
    /// barrier** — a global barrier would serialize the SM sets and kill overlap.
    pub counters: Vec<Counter>,
    /// Per-task counters to wait on (index-aligned with `tasks.tasks`).
    pub wait_of: Vec<Vec<usize>>,
    /// Per-task counters to increment on completion.
    pub succ_of: Vec<Vec<usize>>,
    /// The compute→compute chunk edges (a subset of `tasks.edges`), exposed for
    /// the kernel's producer→consumer wiring.
    pub chunk_edges: Vec<(TaskId, TaskId)>,
    /// The chosen chunk count `k`.
    pub k: u32,
}

/// Build the PerChunk pipeline for a prefill op chain: expand at
/// `Granularity::PerChunk(k)`, assign each chunk task an SM set per `policy`, and
/// cluster fine (per-consumer-chunk, threshold-1) counters.
///
/// This is the single entry point the full-path prefill build calls to obtain
/// the chunk count + 1:1 edge structure the double-buffered kernel consumes.
/// `n_cu` is the compute-unit (SM) count to partition across the `k` chunks.
pub fn expand_prefill_chunks(
    soc: &Soc,
    machine: &crate::machine::Machine,
    g: &TileGraph,
    cons: &ConstraintSet,
    n_cu: u32,
    k: u32,
    policy: ChunkPlacementPolicy,
) -> PerChunkPlan {
    let k = k.max(1);
    let cfg = Config {
        granularity: Granularity::PerChunk(k),
        ..Config::default()
    };
    let tasks = expand(soc, machine, g, cons, &cfg);

    // Recover the chunk index of each compute task: within a node, chunk tasks
    // are emitted in row-axis order, so sorting by the row-axis coord recovers
    // the chunk index deterministically.
    let mut by_node: HashMap<usize, Vec<TaskId>> = HashMap::new();
    for (i, t) in tasks.tasks.iter().enumerate() {
        if t.kind == TaskKind::Compute {
            by_node.entry(t.node).or_default().push(i);
        }
    }
    let mut placement = Vec::new();
    let n_cu = n_cu.max(1);
    for (node, mut ts) in by_node {
        let axis = cons
            .domains
            .get(&node)
            .and_then(TileDomain::row_axis)
            .map(|(a, _)| a)
            .unwrap_or(0);
        ts.sort_by_key(|&t| tasks.tasks[t].coord.get(axis).copied().unwrap_or(0));
        for (ci, &t) in ts.iter().enumerate() {
            let c = ci as u32;
            let cu_range = match policy {
                // Static affinity: chunk c pins to its own SM set; the whole op
                // chain for chunk c shares it → producer output stays in that L2
                // slice for the consumer.
                ChunkPlacementPolicy::StaticColocated => ((c * n_cu) / k, ((c + 1) * n_cu) / k),
                // Global queue: every chunk may use the whole chip (work-steal).
                ChunkPlacementPolicy::GlobalQueue => (0, n_cu),
            };
            placement.push(ChunkPlacement {
                task: t,
                node,
                chunk: c,
                cu_range,
            });
        }
    }
    placement.sort_by_key(|p| (p.node, p.chunk));

    // Fine counters: one per consumer chunk, threshold = its producer-chunk
    // in-degree (1 for matched chunk counts). No colocation set → no IntraSm
    // scoping; the chunk edges are what gate the pipeline.
    let units = cons.placement.clone();
    let (counters, wait_of, succ_of) =
        build_counters(&tasks, &units, &HashSet::new(), ClusterMode::Fine);

    let chunk_edges: Vec<(TaskId, TaskId)> = tasks
        .edges
        .iter()
        .copied()
        .filter(|&(a, b)| {
            tasks.tasks[a].kind == TaskKind::Compute
                && tasks.tasks[b].kind == TaskKind::Compute
                && tasks.tasks[a].node != tasks.tasks[b].node
        })
        .collect();

    PerChunkPlan {
        tasks,
        placement,
        counters,
        wait_of,
        succ_of,
        chunk_edges,
        k,
    }
}

// --- duration + byte helpers -------------------------------------------------

fn compute_cycles(
    soc: &Soc,
    machine: &crate::machine::Machine,
    info: &NodeInfo,
    _coord: &[i64],
    elem: u64,
) -> Cycle {
    let spec = soc.unit(info.unit).cm.spec;
    match (info.kind, info.tile) {
        (OpKind::Gemm(_), Compute::Gemm(t)) => {
            cost::tile_compute_cycles(spec, t, MmaDtype::Bf16).saturating_mul(info.passes.max(1))
        }
        (OpKind::Flash(a), Compute::Flash(t)) => {
            let macs = 2 * (t.bq.max(1) * a.seq_kv.max(1) * a.head_dim.max(1)) as u64;
            cost::macs_cycles(spec, macs, MmaDtype::Bf16).max(1)
        }
        (OpKind::Row(r), Compute::Row(t)) => {
            let bytes = (t.br.max(1) * r.feat.max(1)) as u64 * r.operands.max(1) as u64 * elem;
            machine.hbm_cycles(info.unit, bytes)
        }
        (OpKind::Layout(s), _) => machine.hbm_cycles(
            info.unit,
            s.bytes / machine.unit(info.unit).sm_count.max(1) as u64,
        ),
        _ => 1,
    }
}

/// TMEM columns a tile's MMA accumulator occupies — one column holds 128 f32
/// (128 lanes). Only matmul-class tiles (GEMM / flash) accumulate in TMEM.
fn accumulator_cols(kind: &OpKind, tile: &Compute) -> u64 {
    let elems = match (kind, tile) {
        (OpKind::Gemm(_), Compute::Gemm(t)) => (t.bm.max(0) * t.bn.max(0)) as u64,
        (OpKind::Flash(a), Compute::Flash(t)) => (t.bq.max(0) * a.head_dim.max(0)) as u64,
        _ => 0,
    };
    elems.div_ceil(128)
}

/// Element bytes of a rectangular slice, or `None` if it spans the whole tensor
/// (empty ranges — a layout passthrough).
fn slice_bytes(s: &TensorSlice, elem: u64) -> Option<u64> {
    if s.ranges.is_empty() {
        return None;
    }
    let elems: i64 = s.ranges.iter().map(|r| (r.end - r.start).max(0)).product();
    Some(elems.max(0) as u64 * elem)
}

/// Choose the correct element byte size for a given input of an op.
/// For GEMMs: input 0 = activation (compute precision), input 1+ = weight.
/// All other ops use activation_elem uniformly.
fn op_elem_for_input(kind: &OpKind, idx: usize, activation_elem: u64, weight_elem: u64) -> u64 {
    match kind {
        OpKind::Gemm(_) if idx >= 1 => weight_elem,
        _ => activation_elem,
    }
}

/// Whole-tensor byte volume of op input `idx` (for `PerOp` granularity).
fn op_in_bytes(kind: &OpKind, idx: usize, elem: u64) -> u64 {
    match kind {
        OpKind::Gemm(g) => match idx {
            0 => (g.m * g.k) as u64 * elem, // activation
            1 => (g.n * g.k) as u64 * elem, // weight [N,K]
            _ => g.n.max(0) as u64 * elem,  // bias-ish
        },
        OpKind::Row(r) => (r.rows * r.feat) as u64 * elem,
        OpKind::Flash(a) => match idx {
            0 => (a.heads * a.seq_q * a.head_dim) as u64 * elem,
            _ => (a.heads * a.seq_kv * a.head_dim) as u64 * elem,
        },
        OpKind::Layout(s) => s.bytes,
    }
}

fn op_out_bytes(kind: &OpKind, elem: u64) -> u64 {
    match kind {
        OpKind::Gemm(g) => (g.m * g.n) as u64 * elem,
        OpKind::Row(r) => (r.rows * r.feat) as u64 * elem,
        OpKind::Flash(a) => (a.heads * a.seq_q * a.head_dim) as u64 * elem,
        OpKind::Layout(s) => s.bytes,
    }
}

#[cfg(test)]
mod chunk_tests {
    use super::*;

    // GEMM domain: M=2048, N=15360, bm=128, bn=256 → 16 M-tiles × 60 N-tiles.
    fn gemm_domain() -> TileDomain {
        TileDomain::Gemm {
            m: 2048,
            n: 15360,
            bm: 128,
            bn: 256,
        }
    }

    #[test]
    fn group_partitions_row_axis_evenly() {
        let d = gemm_domain();
        let coords = d.coords();
        let n_ntiles = 15360 / 256; // 60 tiles per M-row-block
        for k in [2u32, 4, 8] {
            let chunks = group_by_row_axis(&d, &coords, k);
            assert_eq!(chunks.len() as u32, k, "k={k}: chunk count");
            // Every tile lands in exactly one chunk; no loss, no duplication.
            let total: usize = chunks.iter().map(|c| c.coords.len()).sum();
            assert_eq!(total, coords.len(), "k={k}: partition covers all tiles");
            // 16 M-tiles split into k contiguous groups ⇒ (16/k) M-rows each, ×60
            // N-tiles per row.
            let per_chunk_mtiles = 16 / k as usize;
            for c in &chunks {
                assert_eq!(
                    c.coords.len(),
                    per_chunk_mtiles * n_ntiles as usize,
                    "k={k}: tiles per chunk"
                );
            }
            // Row ranges are contiguous and cover [0, M) exactly once.
            let mut ranges: Vec<(i64, i64)> = chunks.iter().map(|c| c.range).collect();
            ranges.sort();
            assert_eq!(ranges[0].0, 0);
            assert_eq!(ranges.last().unwrap().1, 2048);
            for w in ranges.windows(2) {
                assert_eq!(w[0].1, w[1].0, "k={k}: ranges contiguous, no gap/overlap");
            }
        }
    }

    #[test]
    fn group_row_axis_is_axis1_for_flash() {
        // Flash row axis = grid axis 1 (seq_q); coords are [head, q].
        let d = TileDomain::Flash {
            heads: 4,
            seq_q: 1024,
            bq: 128,
        }; // 8 q-tiles × 4 heads
        let coords = d.coords();
        let chunks = group_by_row_axis(&d, &coords, 4);
        assert_eq!(chunks.len(), 4);
        // Each chunk = 2 q-tiles × 4 heads = 8 coords, range spans 256 seq each.
        for c in &chunks {
            assert_eq!(c.coords.len(), 8);
            assert_eq!(c.range.1 - c.range.0, 256);
        }
    }

    #[test]
    fn group_caps_k_at_row_tile_count() {
        // k above the row-tile count ⇒ capped (no empty chunks emitted).
        let d = TileDomain::Row { rows: 512, br: 128 }; // 4 row-tiles
        let coords = d.coords();
        let chunks = group_by_row_axis(&d, &coords, 8);
        assert_eq!(chunks.len(), 4, "k capped at 4 row-tiles");
        assert!(chunks.iter().all(|c| c.coords.len() == 1));
    }

    #[test]
    fn layout_domain_is_one_chunk() {
        let d = TileDomain::Layout;
        let chunks = group_by_row_axis(&d, &d.coords(), 4);
        assert_eq!(chunks.len(), 1);
    }
}
