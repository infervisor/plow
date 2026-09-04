//! Per-layer tile graph + constraint set, over a heterogeneous SoC (design
//! §3.1, §6.1).
//!
//! [`assemble`] takes a [`LayerPlan`] — one transformer layer's ops with their
//! concrete (bucket-bound) shapes and named operands — plus a [`Soc`], and
//! lowers it to a [`TileGraph`] of DMA/compute nodes plus the [`ConstraintSet`]
//! the scheduler consumes.
//!
//! Tile *selection* for every (op, region) goes through a single
//! [`crate::explore::select`] call: `costmodel` enumerates the legal candidates
//! and costs them, egglog picks the per-task argmin in one pass.
//!
//! Heterogeneity: a GEMM is partitioned across the SoC's units
//! ([`Soc::partition_n`]) into one placed `Compute` per region — **uneven across
//! units** (sized to throughput), uniform within each (its own MMA tile) — plus
//! a [`Compute::Join`] concat. A single-unit SoC ⇒ one region = today's path.
//!
//! Three structural facts are recorded as constraints (§2.5):
//! * **dma-dedup** — an operand read by several tasks is staged from DRAM once.
//! * **sram-handoff** — a *same-unit* producer→consumer pair keeps the value in
//!   SRAM (no HBM round-trip) and is pinned to one SM (a colocation group).
//! * **cross-unit hand-off** — under [unified memory](MemoryModel::unified) a
//!   consumer on another unit reads with no extra copy (a barrier, not a
//!   colocation); under discrete memory it would be an interconnect transfer.

use crate::explore::{self, Candidate, ChoicePoint, ExploreError};
use crate::footprint::TileDomain;
use costmodel::{
    AttnShape, CostModel, CostParams, FlashTile, GemmShape, RowShape, RowTile, Soc, SramPolicy,
    TileShape, UnitId,
};
use std::collections::HashMap;

/// The compute kind of a task, with its chosen tile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Compute {
    Gemm(TileShape),
    Flash(FlashTile),
    Row(RowTile),
    /// Concat of a split GEMM's per-region partial outputs (layout / DMA only).
    Join,
    /// Layout-only (reshape/transpose/slice/…): no matrix engine, pure DMA.
    Layout,
}

/// Max tensor rank a layout descriptor addresses — re-exported from [`packet`].
pub const LAYOUT_RANK: usize = packet::LAYOUT_MAX_RANK;

/// A layout op's runtime descriptor: a strided block copy
/// `out[out_base + Σ idxₐ·out_strideₐ] = in[in_base + Σ idxₐ·in_strideₐ]` over
/// `shape`. `kind==0` is a plain contiguous copy; `kind==1` is the general
/// gather/scatter (transpose/broadcast/slice). Strides/bases are in elements.
#[derive(Clone, Copy, Debug)]
pub struct LayoutSpec {
    pub bytes: u64,
    pub kind: u8,
    pub rank: u8,
    pub elem_size: u8,
    pub shape: [u32; LAYOUT_RANK],
    pub in_stride: [u32; LAYOUT_RANK],
    pub out_stride: [u32; LAYOUT_RANK],
    pub in_base: u32,
    pub out_base: u32,
    /// The output is byte-identical to input 0 (a reshape): the allocator may
    /// alias the two buffers and the copy becomes a zero-byte no-op (Phase C).
    pub alias: bool,
}

impl LayoutSpec {
    /// A contiguous byte copy of `bytes` bytes (kind 0) — used for internal joins
    /// and any layout not lowered to a strided descriptor.
    pub fn copy(bytes: u64) -> Self {
        LayoutSpec {
            bytes,
            kind: 0,
            rank: 0,
            elem_size: 0,
            shape: [0; LAYOUT_RANK],
            in_stride: [0; LAYOUT_RANK],
            out_stride: [0; LAYOUT_RANK],
            in_base: 0,
            out_base: 0,
            alias: false,
        }
    }
    /// A zero-copy reshape: identical bytes to input 0, eligible for aliasing.
    pub fn reshape(bytes: u64) -> Self {
        LayoutSpec {
            alias: true,
            ..Self::copy(bytes)
        }
    }
}

/// What an op computes (problem shape bound by the bucket).
#[derive(Clone, Copy, Debug)]
pub enum OpKind {
    /// `out[M,N] = act[M,K] · weightᵀ`. Partitioned along N across the SoC.
    Gemm(GemmShape),
    Flash(AttnShape),
    Row(RowShape),
    /// A model operation whose mathematical identity must survive packet
    /// emission.  It is tiled like a row op, but is never emitted as the
    /// generic ROW opcode: `kind` and `args` are part of the packet ABI.
    Model(ModelOp),
    Layout(LayoutSpec),
}

/// Compiler-side metadata for operations which cannot be represented by the
/// generic GEMM/FLASH/ROW/LAYOUT families without losing model semantics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelOp {
    pub kind: ModelOpKind,
    pub rows: i64,
    pub feat: i64,
    pub operands: i64,
    /// Kind-specific integer payload. Floating-point attributes travel as
    /// their IEEE-754 bit pattern.
    pub args: [u32; 4],
    /// Exact whole-tensor input sizes. Model ops include irregular operands
    /// (embedding tables, depthwise kernels, scalar/head parameters) that
    /// cannot be inferred from the output row shape.
    pub input_bytes: [u64; 8],
}

impl ModelOp {
    pub fn row_shape(self) -> RowShape {
        RowShape {
            rows: self.rows,
            feat: self.feat,
            operands: self.operands,
            reduce: matches!(
                self.kind,
                ModelOpKind::RmsNorm | ModelOpKind::RmsNormZeroCentered
            ),
        }
    }
}

/// Stable offsets from `packet::Opcode::VARIANT_MODEL_BASE`.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelOpKind {
    Embedding = 1,
    RmsNormZeroCentered = 2,
    Rope = 3,
    Silu = 4,
    Sigmoid = 5,
    Add = 6,
    Sub = 7,
    Mul = 8,
    Div = 9,
    CausalDepthwiseConv1d = 10,
    QwenGatedDelta = 11,
    RmsNorm = 12,
}

/// One op of the layer. Convention: `inputs[0]` is the activation (shared, read
/// in full by every region); `inputs[1..]` are parameters (sliced along N when
/// the op is split).
#[derive(Clone, Debug)]
pub struct OpSpec {
    pub name: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub kind: OpKind,
    /// Weight operand dtype (e.g. Q4_K for block-quant, BF16 for standard).
    /// Determines SRAM staging size for the B-operand and kernel variant.
    pub weight_dtype: nn_graph::DType,
    /// Compute / activation dtype (what the matrix engine accumulates in).
    pub compute_dtype: nn_graph::DType,
}

impl OpSpec {
    /// Construct with default dtypes (BF16 activation + BF16 weight). Use this
    /// when the source doesn't carry per-tensor dtype info (e.g. the `--net`
    /// path or test fixtures). The bridge populates real dtypes from the graph.
    pub fn bf16(name: String, inputs: Vec<String>, output: String, kind: OpKind) -> Self {
        OpSpec {
            name,
            inputs,
            output,
            kind,
            weight_dtype: nn_graph::DType::BF16,
            compute_dtype: nn_graph::DType::BF16,
        }
    }
}

/// A whole layer's ops in program order.
#[derive(Clone, Debug, Default)]
pub struct LayerPlan {
    pub ops: Vec<OpSpec>,
}

/// A tile-graph node.
#[derive(Clone, Debug, PartialEq)]
pub enum TileNode {
    /// Stage an operand. `resident` ⇒ already in this unit's SRAM from a
    /// same-unit producer (no HBM read).
    DmaIn { tensor: String, resident: bool },
    /// A task's tile-step, streamed over `passes`, footprint `sram_pages`.
    /// `inline_in`/`inline_out` are boundary DMAs folded into the kernel (the
    /// kernel issues its own load/store) after [`crate::collapse`].
    Compute {
        op: String,
        kind: Compute,
        passes: u64,
        sram_pages: u64,
        inline_in: Vec<String>,
        inline_out: bool,
    },
    /// Write the output. `resident` ⇒ kept in SRAM for a same-unit consumer.
    DmaOut { tensor: String, resident: bool },
}

/// An sram-handoff: the consumer `dma_in` reads what the producer `dma_out`
/// wrote. `cross_unit` ⇒ the two run on different units (no colocation).
///
/// `kind` is the physical realization chosen by [`crate::collapse`] (`Hbm` until
/// then). Expansion branches on it — e.g. a cross-unit `Barrier` is a fence over
/// unified memory and emits no data-transfer task.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Handoff {
    pub producer_dma_out: usize,
    pub consumer_dma_in: usize,
    pub cross_unit: bool,
    pub kind: HandoffKind,
}

/// The chosen physical realization of a producer→consumer hand-off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandoffKind {
    /// DRAM round-trip (DmaOut→DmaIn). Producer/consumer free to parallelize.
    Hbm,
    /// SRAM-resident on one SM — no round-trip, but serializes the two.
    SramSameSm,
    /// Distributed shared memory: shared across SMs in one GPC domain.
    Dsm,
    /// L2-partition-resident: producer writes to an L2 slice, consumer reads
    /// from the same slice without an HBM round-trip. Available on both
    /// H100 (per-GPC L2) and MI300 (per-XCD L2). Fills the middle ground
    /// between `SramSameSm` (fastest, serialized) and `Hbm` (parallel,
    /// full round-trip). See the design notes.
    L2Local,
    /// Cross-unit under unified memory: a fence, no data movement.
    Barrier,
    /// Cross-unit direct read over the fast peer fabric (NVLink / Infinity),
    /// within one node-domain.
    P2p,
    /// Cross-unit DPU-routed transfer over a slow link / between nodes (RDMA).
    Rdma,
}

/// Placement requirement a hand-off realization imposes on the scheduler (§3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LocalityReq {
    /// Producer and consumer must share an SM (`SramSameSm`).
    MustColocate,
    /// They must share a DSM domain (a GPC), on different SMs (`Dsm`).
    SameDomain,
    /// They must share the same L2 partition (per-GPC on H100, per-XCD on
    /// MI300) — cheaper than `SameDomain` (no DSM messaging fabric needed),
    /// stronger than `SameNode` (must sit within one L2 slice).
    SameL2Partition,
    /// They must sit within one fast-fabric node-domain (`P2p` / `Barrier`).
    SameNode,
    /// No placement constraint (`Hbm` round-trip, or `Rdma`).
    NoConstraint,
}

/// A cost-driven default hand-off choice the scheduler may relax (§6.4): it
/// carries every realization's cost so placement can flip under SRAM/occupancy
/// pressure.
#[derive(Clone, Debug, PartialEq)]
pub struct RelaxableHandoff {
    pub producer: usize,
    pub consumer: usize,
    pub tensor: String,
    pub default: HandoffKind,
    pub alts: Vec<(HandoffKind, u64)>,
}

/// How one shared tensor axis couples a consumer's tile grid to a producer's:
/// consumer tile coord `c` reads `[c·consumer_block : +consumer_block]` of the
/// axis, which is produced by producer tiles whose `[p·producer_block : …]`
/// write-range overlaps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AxisCouple {
    pub consumer_axis: usize,
    pub producer_axis: usize,
    pub consumer_block: i64,
    pub producer_block: i64,
}

/// The compact, fine-grained cross-op dependency on one shared tensor (the
/// source of truth; [`materialize_tile_deps`] expands it to explicit tile edges).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileDependency {
    pub tensor: String,
    /// One entry per coupled axis (currently just the token/row axis).
    pub couple: Vec<AxisCouple>,
}

/// A producer→consumer dependency between two compute nodes, carrying the
/// tile-coordinate coupling.
#[derive(Clone, Debug)]
pub struct TileDep {
    pub producer: usize,
    pub consumer: usize,
    pub dep: TileDependency,
}

/// One materialized tile-to-tile edge: producer tile `producer_coord` must
/// complete before consumer tile `consumer_coord` may issue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatEdge {
    pub producer_coord: Vec<i64>,
    pub consumer_coord: Vec<i64>,
}

/// Scheduler-facing constraints recorded while assembling the graph (§6.1).
#[derive(Clone, Debug, Default)]
pub struct ConstraintSet {
    /// Compute nodes that must share an SM (same-unit hand-offs, union-find).
    pub colocation_groups: Vec<Vec<usize>>,
    /// Compute node → the unit it runs on.
    pub placement: HashMap<usize, UnitId>,
    /// Compute node → its SRAM page footprint.
    pub sram_pages: HashMap<usize, u64>,
    /// Operand staged once from DRAM → the compute nodes reusing it (dma-dedup).
    pub staged_inputs: Vec<(String, Vec<usize>)>,
    /// Producer→consumer hand-offs (same-unit resident or cross-unit).
    pub handoffs: Vec<Handoff>,
    /// Fine-grained cross-op tile dependencies (compact; expand with
    /// [`materialize_tile_deps`]).
    pub tile_deps: Vec<TileDep>,
    /// Compute node → its tile-coordinate domain.
    pub domains: HashMap<usize, TileDomain>,
    /// Cost-driven hand-off realizations the scheduler may relax ([`crate::collapse`]).
    pub relaxables: Vec<RelaxableHandoff>,
    /// Producer→consumer compute-edge → its placement requirement.
    pub locality: HashMap<(usize, usize), LocalityReq>,
    /// Compute node → its problem shape + operand names, so a downstream
    /// scheduler can recover per-tile footprints (via [`crate::footprints`]) and
    /// recompute per-tile/op durations from the cost model.
    pub op_io: HashMap<usize, OpDesc>,
    /// Whether the SoC's units share one coherent address space.
    pub unified_memory: bool,
    /// Contiguous (outer-axis) concats: the output tensor's bytes are exactly its
    /// inputs concatenated, so the memory allocator can place the inputs adjacently
    /// inside the output's storage (no fresh allocation for the output). Phase C3.
    pub concat_groups: Vec<ConcatGroup>,
}

/// A contiguous concat the allocator can realize by adjacency: `output`'s storage
/// holds `parts` laid end-to-end in order.
#[derive(Clone, Debug)]
pub struct ConcatGroup {
    pub output: String,
    pub parts: Vec<String>,
}

/// A compute node's problem shape and operand names (the scheduler's view of an
/// op: enough to rebuild footprints and costs). `inputs[0]` is the activation.
#[derive(Clone, Debug)]
pub struct OpDesc {
    pub kind: OpKind,
    pub inputs: Vec<String>,
    pub output: String,
    /// Original checkpoint weight dtype; retained for manifest bindings such
    /// as FP8 weight→scale associations.
    pub weight_dtype: nn_graph::DType,
    /// Bytes per activation element (compute precision).
    pub activation_elem: u64,
    /// Bytes per weight element (possibly block-quant amortized).
    pub weight_elem: u64,
    /// Whether the weight uses a block-quantized format (Q4_K, Q8_0, etc.).
    pub block_quant: bool,
    /// Whether the weight is MX FP4 (native Blackwell 4-bit tensor-core path).
    pub native_fp4: bool,
}

/// The assembled tile graph.
#[derive(Clone, Debug, Default)]
pub struct TileGraph {
    pub nodes: Vec<TileNode>,
    /// Data-flow edges (producer node → consumer node).
    pub edges: Vec<(usize, usize)>,
}

/// A candidate compute (tile) for a task, with the cost/footprint costmodel gives it.
struct Cand {
    kind: Compute,
    cost: u64,
    passes: u64,
    sram_pages: u64,
}

/// The problem shape behind a task, kept so its [`TileDomain`] can be built once
/// its tile is chosen.
#[derive(Clone, Copy, Debug)]
enum TaskShape {
    Gemm(GemmShape),
    Flash(AttnShape),
    Row(RowShape),
    Model(ModelOp),
    Layout,
}

/// One node to emit: a placed compute with its operands, output, and the index
/// of its candidate list in `choices`.
struct Task {
    op: String,
    unit: UnitId,
    inputs: Vec<String>,
    output: String,
    choice: usize,
    shape: TaskShape,
    kind: OpKind,
    activation_elem: u64,
    weight_elem: u64,
    block_quant: bool,
    native_fp4: bool,
    weight_dtype: nn_graph::DType,
}

/// The tile-coordinate domain of a task once its compute tile is chosen.
fn domain_of(shape: &TaskShape, kind: &Compute) -> TileDomain {
    match (shape, kind) {
        (TaskShape::Gemm(g), Compute::Gemm(t)) => TileDomain::Gemm {
            m: g.m,
            n: g.n,
            bm: t.bm,
            bn: t.bn,
        },
        (TaskShape::Row(r), Compute::Row(t)) => TileDomain::Row {
            rows: r.rows,
            br: t.br,
        },
        (TaskShape::Model(m), Compute::Row(t)) => TileDomain::Row {
            rows: m.rows,
            br: t.br,
        },
        (TaskShape::Flash(a), Compute::Flash(t)) => TileDomain::Flash {
            heads: a.heads,
            seq_q: a.seq_q,
            bq: t.bq,
        },
        _ => TileDomain::Layout,
    }
}

impl TaskShape {
    /// Does this op read its `idx`-th input along its token/row axis? Only then
    /// does a hand-off on that input couple the two tile grids.
    fn couples_input(&self, idx: usize) -> bool {
        match self {
            TaskShape::Gemm(_) | TaskShape::Flash(_) => idx == 0, // the activation / query
            TaskShape::Row(_) | TaskShape::Model(_) => true,      // every input is row-aligned
            TaskShape::Layout => false,
        }
    }
}

/// Lower a whole layer to its tile graph + constraint set on `soc`.
///
/// `weight_tiling` pins the GEMM weight tile `(BN, BK)` so a single weight layout
/// serves every shape bucket (design §10.2): only `BM` (the activation/`M`
/// tiling) then varies. `None` lets the cost model choose all of `(BM, BN, BK)`
/// freely (the default, single-shape path).
pub fn assemble(
    soc: &Soc,
    plan: &LayerPlan,
    policy: SramPolicy,
    weight_tiling: Option<(i64, i64)>,
) -> Result<(TileGraph, ConstraintSet), ExploreError> {
    assemble_tuned(soc, plan, policy, weight_tiling, &crate::oracle::NoOracle)
}

/// [`assemble`], with an oracle consulted about kernel availability and cost.
///
/// The oracle filters candidates the target cannot execute and may replace the
/// analytical cost with a measurement. With [`NoOracle`](crate::oracle::NoOracle)
/// this is exactly `assemble`, which is what keeps the plumbing safe to land
/// before any measurement exists.
pub fn assemble_tuned(
    soc: &Soc,
    plan: &LayerPlan,
    policy: SramPolicy,
    weight_tiling: Option<(i64, i64)>,
    oracle: &dyn crate::oracle::KernelOracle,
) -> Result<(TileGraph, ConstraintSet), ExploreError> {
    // --- Pass 1: expand ops into placed tasks, enumerate + cost candidates. ---
    let mut choices: Vec<Vec<Cand>> = Vec::new();
    let mut tasks: Vec<Task> = Vec::new();
    let mut push_choice = |cands: Vec<Cand>| -> usize {
        choices.push(cands);
        choices.len() - 1
    };

    for op in &plan.ops {
        let cost_params = CostParams::from_dtypes(op.weight_dtype, op.compute_dtype);
        match op.kind {
            OpKind::Gemm(g) => {
                let regions = soc.partition_n(g);
                let split = regions.len() > 1;
                let mut slice_outs = Vec::new();
                for r in &regions {
                    let cm = &soc.unit(r.unit).cm;
                    let (inputs, output) = if split {
                        // Slice the parameters and the output along N; share the activation.
                        let mut ins = vec![op.inputs[0].clone()];
                        for w in &op.inputs[1..] {
                            ins.push(slice_name(w, r.n_start));
                        }
                        (ins, slice_name(&op.output, r.n_start))
                    } else {
                        (op.inputs.clone(), op.output.clone())
                    };
                    slice_outs.push(output.clone());
                    let choice = push_choice(gemm_cands(
                        cm,
                        r.shape,
                        policy,
                        weight_tiling,
                        cost_params,
                        oracle,
                    ));
                    tasks.push(Task {
                        op: op.name.clone(),
                        unit: r.unit,
                        inputs,
                        output,
                        choice,
                        shape: TaskShape::Gemm(r.shape),
                        kind: OpKind::Gemm(r.shape),
                        activation_elem: cost_params.activation_elem,
                        weight_elem: cost_params.weight_elem,
                        block_quant: cost_params.block_quant,
                        native_fp4: cost_params.native_fp4,
                        weight_dtype: op.weight_dtype,
                    });
                }
                if split {
                    // Join the per-region slices back into the full output on unit 0.
                    let cm0 = &soc.unit(0).cm;
                    let bytes = (g.m * g.n) as u64 * cost_params.activation_elem;
                    let choice = push_choice(vec![Cand {
                        kind: Compute::Join,
                        cost: cm0.layout_cost(bytes, soc.memory.unified),
                        passes: 1,
                        sram_pages: 0,
                    }]);
                    tasks.push(Task {
                        op: format!("{}.join", op.name),
                        unit: 0,
                        inputs: slice_outs,
                        output: op.output.clone(),
                        choice,
                        shape: TaskShape::Layout,
                        kind: OpKind::Layout(LayoutSpec::copy(bytes)),
                        activation_elem: cost_params.activation_elem,
                        weight_elem: cost_params.weight_elem,
                        block_quant: false, // joins are always in compute precision
                        native_fp4: false,
                        weight_dtype: nn_graph::DType::BF16,
                    });
                }
            }
            OpKind::Flash(a) => {
                let cm = &soc.unit(0).cm;
                let choice = push_choice(flash_cands(cm, a, policy));
                tasks.push(Task {
                    op: op.name.clone(),
                    unit: 0,
                    inputs: op.inputs.clone(),
                    output: op.output.clone(),
                    choice,
                    shape: TaskShape::Flash(a),
                    kind: OpKind::Flash(a),
                    activation_elem: cost_params.activation_elem,
                    weight_elem: cost_params.weight_elem,
                    block_quant: cost_params.block_quant,
                    native_fp4: cost_params.native_fp4,
                    weight_dtype: op.weight_dtype,
                });
            }
            OpKind::Row(r) => {
                let cm = &soc.unit(0).cm;
                let choice = push_choice(row_cands(cm, r, policy));
                tasks.push(Task {
                    op: op.name.clone(),
                    unit: 0,
                    inputs: op.inputs.clone(),
                    output: op.output.clone(),
                    choice,
                    shape: TaskShape::Row(r),
                    kind: OpKind::Row(r),
                    activation_elem: cost_params.activation_elem,
                    weight_elem: cost_params.weight_elem,
                    block_quant: cost_params.block_quant,
                    native_fp4: cost_params.native_fp4,
                    weight_dtype: op.weight_dtype,
                });
            }
            OpKind::Model(m) => {
                let cm = &soc.unit(0).cm;
                let choice = push_choice(row_cands(cm, m.row_shape(), policy));
                tasks.push(Task {
                    op: op.name.clone(),
                    unit: 0,
                    inputs: op.inputs.clone(),
                    output: op.output.clone(),
                    choice,
                    shape: TaskShape::Model(m),
                    kind: OpKind::Model(m),
                    activation_elem: cost_params.activation_elem,
                    weight_elem: cost_params.weight_elem,
                    block_quant: cost_params.block_quant,
                    native_fp4: cost_params.native_fp4,
                    weight_dtype: op.weight_dtype,
                });
            }
            OpKind::Layout(spec) => {
                let cm = &soc.unit(0).cm;
                let choice = push_choice(vec![Cand {
                    kind: Compute::Layout,
                    cost: cm.layout_cost(spec.bytes, false),
                    passes: 1,
                    sram_pages: 0,
                }]);
                tasks.push(Task {
                    op: op.name.clone(),
                    unit: 0,
                    inputs: op.inputs.clone(),
                    output: op.output.clone(),
                    choice,
                    shape: TaskShape::Layout,
                    kind: OpKind::Layout(spec), // preserve the bridge-computed descriptor
                    activation_elem: cost_params.activation_elem,
                    weight_elem: cost_params.weight_elem,
                    block_quant: cost_params.block_quant,
                    native_fp4: cost_params.native_fp4,
                    weight_dtype: op.weight_dtype,
                });
            }
        }
    }

    // One egglog datalog pass picks the argmin candidate per task.
    let points: Vec<ChoicePoint> = choices
        .iter()
        .enumerate()
        .map(|(id, cands)| ChoicePoint {
            id: id as i64,
            candidates: cands
                .iter()
                .enumerate()
                .map(|(t, c)| Candidate {
                    tag: t as i64,
                    cost: c.cost,
                })
                .collect(),
        })
        .collect();
    let chosen = explore::select(&points)?;

    // --- Pass 2: build the graph, recording dedup + hand-off + placement. ---
    let mut g = TileGraph::default();
    let mut cons = ConstraintSet {
        unified_memory: soc.memory.unified,
        ..Default::default()
    };
    // tensor → its producers (compute node, DmaOut node, unit). Usually one, but a
    // tensor assembled from disjoint sub-regions (e.g. concat adjacency) has
    // several; a consumer waits on every producer of the region it reads. With one
    // whole-tensor producer this is identical to the prior single-writer model.
    let mut produced: HashMap<String, Vec<(usize, usize, UnitId)>> = HashMap::new();
    // operand already staged from DRAM → (its DmaIn node, index into staged_inputs).
    let mut staged: HashMap<String, (usize, usize)> = HashMap::new();
    let mut uf = UnionFind::default();

    for task in &tasks {
        let tag = chosen[&(task.choice as i64)] as usize;
        let cand = &choices[task.choice][tag];
        let domain = domain_of(&task.shape, &cand.kind);
        let consumer_row = domain.row_axis();

        let mut input_dma = Vec::new();
        let mut same_unit_producers = Vec::new();
        // Producers feeding a row-coupled input: (producer compute, shared tensor).
        let mut couplings: Vec<(usize, String)> = Vec::new();

        for (idx, inp) in task.inputs.iter().enumerate() {
            if let Some(prods) = produced.get(inp) {
                let prods = prods.clone(); // drop the borrow on `produced` before mutating g
                                           // SRAM-resident only when every producer is on this unit.
                let all_same = prods.iter().all(|&(_, _, u)| u == task.unit);
                let dma_in = push(
                    &mut g,
                    TileNode::DmaIn {
                        tensor: inp.clone(),
                        resident: all_same,
                    },
                );
                // Wait on every producer of the region this consumer reads.
                for (prod_compute, prod_dma_out, prod_unit) in prods {
                    let same = prod_unit == task.unit;
                    if same {
                        if let TileNode::DmaOut { resident, .. } = &mut g.nodes[prod_dma_out] {
                            *resident = true;
                        }
                        same_unit_producers.push(prod_compute);
                    }
                    cons.handoffs.push(Handoff {
                        producer_dma_out: prod_dma_out,
                        consumer_dma_in: dma_in,
                        cross_unit: !same,
                        // Realization is chosen later by `collapse`; default to
                        // the HBM round-trip until then.
                        kind: HandoffKind::Hbm,
                    });
                    // A fine-grained tile dependency exists when the consumer reads this
                    // input along its row axis and the producer tiles that axis too.
                    if task.shape.couples_input(idx)
                        && consumer_row.is_some()
                        && cons
                            .domains
                            .get(&prod_compute)
                            .and_then(TileDomain::row_axis)
                            .is_some()
                    {
                        couplings.push((prod_compute, inp.clone()));
                    }
                }
                input_dma.push(dma_in);
            } else if let Some(&(dma_in, _)) = staged.get(inp) {
                input_dma.push(dma_in); // dma-dedup: reuse the staged operand.
            } else {
                let dma_in = push(
                    &mut g,
                    TileNode::DmaIn {
                        tensor: inp.clone(),
                        resident: false,
                    },
                );
                input_dma.push(dma_in);
                cons.staged_inputs.push((inp.clone(), Vec::new()));
                staged.insert(inp.clone(), (dma_in, cons.staged_inputs.len() - 1));
            }
        }

        let compute = push(
            &mut g,
            TileNode::Compute {
                op: task.op.clone(),
                kind: cand.kind,
                passes: cand.passes,
                sram_pages: cand.sram_pages,
                inline_in: Vec::new(),
                inline_out: false,
            },
        );
        cons.placement.insert(compute, task.unit);
        cons.sram_pages.insert(compute, cand.sram_pages);
        cons.domains.insert(compute, domain);
        cons.op_io.insert(
            compute,
            OpDesc {
                kind: task.kind,
                inputs: task.inputs.clone(),
                output: task.output.clone(),
                weight_dtype: task.weight_dtype,
                activation_elem: task.activation_elem,
                weight_elem: task.weight_elem,
                block_quant: task.block_quant,
                native_fp4: task.native_fp4,
            },
        );
        // Contiguous (outer-axis) concat: the output's bytes are its inputs laid
        // end-to-end, so the allocator can place them adjacently inside the output
        // (Phase C3 — no separate allocation for the output).
        if let OpKind::Layout(s) = task.kind {
            if s.kind == 2 {
                let lead: u64 = (0..s.in_base as usize).map(|d| s.shape[d] as u64).product();
                if lead <= 1 && task.inputs.len() >= 2 {
                    cons.concat_groups.push(ConcatGroup {
                        output: task.output.clone(),
                        parts: task.inputs.clone(),
                    });
                }
            }
        }
        for &d in &input_dma {
            g.edges.push((d, compute));
        }
        for inp in &task.inputs {
            if let Some(&(_, si)) = staged.get(inp) {
                cons.staged_inputs[si].1.push(compute);
            }
        }
        for prod in same_unit_producers {
            uf.union(prod, compute);
        }
        // Record the fine-grained tile dependency for each coupled producer.
        if let Some((consumer_axis, consumer_block)) = consumer_row {
            for (prod, tensor) in couplings {
                let (producer_axis, producer_block) = cons.domains[&prod].row_axis().unwrap();
                cons.tile_deps.push(TileDep {
                    producer: prod,
                    consumer: compute,
                    dep: TileDependency {
                        tensor,
                        couple: vec![AxisCouple {
                            consumer_axis,
                            producer_axis,
                            consumer_block,
                            producer_block,
                        }],
                    },
                });
            }
        }

        let dma_out = push(
            &mut g,
            TileNode::DmaOut {
                tensor: task.output.clone(),
                resident: false,
            },
        );
        g.edges.push((compute, dma_out));
        produced
            .entry(task.output.clone())
            .or_default()
            .push((compute, dma_out, task.unit));
    }

    cons.colocation_groups = uf.groups();
    Ok((g, cons))
}

fn slice_name(base: &str, n_start: i64) -> String {
    format!("{base}#n{n_start}")
}

fn push(g: &mut TileGraph, n: TileNode) -> usize {
    g.nodes.push(n);
    g.nodes.len() - 1
}

/// Expand a compact [`TileDependency`] into explicit producer-tile → consumer-tile
/// edges: each consumer tile depends on the producer tiles whose write-range on
/// the coupled axis overlaps its read-range. Axes the producer does not tile
/// (e.g. a GEMM's N) are free — every such consumer tile shares the same
/// producer set.
pub fn materialize_tile_deps(
    producer: &TileDomain,
    consumer: &TileDomain,
    dep: &TileDependency,
) -> Vec<MatEdge> {
    let Some(c) = dep.couple.first() else {
        return Vec::new();
    };
    let producers = producer.coords();
    let mut out = Vec::new();
    for cc in consumer.coords() {
        let c0 = cc[c.consumer_axis] * c.consumer_block;
        let c1 = c0 + c.consumer_block;
        for pc in &producers {
            let p0 = pc[c.producer_axis] * c.producer_block;
            let p1 = p0 + c.producer_block;
            if p0 < c1 && c0 < p1 {
                out.push(MatEdge {
                    producer_coord: pc.clone(),
                    consumer_coord: cc.clone(),
                });
            }
        }
    }
    out
}

/// Per consumer-tile producer in-degree — the counter threshold the scheduler
/// waits on before issuing that tile's DMA/compute (design §6.4).
pub fn consumer_thresholds(
    producer: &TileDomain,
    consumer: &TileDomain,
    dep: &TileDependency,
) -> HashMap<Vec<i64>, u32> {
    let mut m = HashMap::new();
    for e in materialize_tile_deps(producer, consumer, dep) {
        *m.entry(e.consumer_coord).or_insert(0) += 1;
    }
    m
}

fn gemm_cands(
    cm: &CostModel,
    g: GemmShape,
    policy: SramPolicy,
    weight_tiling: Option<(i64, i64)>,
    params: CostParams,
    oracle: &dyn crate::oracle::KernelOracle,
) -> Vec<Cand> {
    use crate::oracle::{GemmQuery, TileAdvice};

    let query = GemmQuery {
        shape: g,
        weight_elem: params.weight_elem,
        activation_elem: params.activation_elem,
        block_quant: params.block_quant,
        native_fp4: params.native_fp4,
    };

    // Analytical candidates, pinned to the shared weight layout if one is set.
    let pin = |t: &TileShape| weight_tiling.is_none_or(|(bn, bk)| t.bn == bn && t.bk == bk);
    let analytical: Vec<TileShape> = cm
        .candidates_typed(g, policy, params)
        .into_iter()
        .filter(pin)
        .collect();

    // The tiles to actually price. The oracle decides whether these come from
    // the analytical model or from the target's real kernel set.
    let tiles: Vec<TileShape> = match oracle.gemm_tiles(&query) {
        // No inventory: analytical, exactly as before the oracle existed.
        TileAdvice::Analytical => analytical,
        // Inventory present but blind to this dtype: analytical, and recorded as
        // capability-unverified. Not silent, not a crash.
        TileAdvice::Unverified => {
            oracle.note_unverified(&query);
            analytical
        }
        // Authoritative: these are the ONLY tiles the target builds. Choosing
        // among synthesized shapes here is a category error -- on NVIDIA the
        // tile is a compile-time macro and the packet's tile fields are not even
        // read. So the compiler emits a tile that exists, never one that does
        // not, which is what closes the fail-open hole.
        TileAdvice::Buildable(built) => {
            let pinned: Vec<TileShape> = built.iter().copied().filter(pin).collect();
            if !pinned.is_empty() {
                pinned
            } else {
                // Every buildable tile was excluded by the shared weight layout:
                // the analytically-chosen (BN, BK) and the real kernels disagree.
                // Report it and fall back to analytical for this op only, rather
                // than emit a tile the target cannot run OR fail the build.
                oracle.note_pin_conflict(&query, &built);
                if built.is_empty() {
                    analytical
                } else {
                    built
                }
            }
        }
    };

    // Measurements replace the estimate for the whole set or none of it: ns and
    // cycles are different scales.
    let measured = oracle
        .measured_gemm(&query, &tiles)
        .filter(|m| m.len() == tiles.len());

    tiles
        .into_iter()
        .enumerate()
        .map(|(i, t)| Cand {
            kind: Compute::Gemm(t),
            cost: measured
                .as_ref()
                .map(|m| m[i])
                .unwrap_or_else(|| cm.gemm_cost_typed(g, t, params)),
            passes: cm.passes_typed(t, params),
            sram_pages: cm.sram_pages_typed(t, params),
        })
        .collect()
}

fn flash_cands(cm: &CostModel, a: AttnShape, policy: SramPolicy) -> Vec<Cand> {
    cm.flash_candidates(a, policy)
        .into_iter()
        .map(|t| Cand {
            kind: Compute::Flash(t),
            cost: cm.flash_cost(a, t),
            passes: cm.flash_passes(a, t),
            sram_pages: cm
                .sram
                .pages(t.working_set_bytes(a.head_dim, cm.elem_bytes, cm.buffering)),
        })
        .collect()
}

fn row_cands(cm: &CostModel, r: RowShape, policy: SramPolicy) -> Vec<Cand> {
    cm.row_candidates(r, policy)
        .into_iter()
        .map(|t| {
            let pages = cm
                .sram
                .pages(t.working_set_bytes(r, cm.elem_bytes, cm.buffering));
            Cand {
                kind: Compute::Row(t),
                cost: cm.row_cost(r),
                passes: cm.sram.loop_passes(pages),
                sram_pages: pages,
            }
        })
        .collect()
}

/// Minimal union-find over compute node indices (for colocation groups).
#[derive(Default)]
struct UnionFind {
    parent: HashMap<usize, usize>,
}
impl UnionFind {
    fn find(&mut self, x: usize) -> usize {
        let p = *self.parent.entry(x).or_insert(x);
        if p == x {
            x
        } else {
            let r = self.find(p);
            self.parent.insert(x, r);
            r
        }
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent.insert(ra, rb);
        }
    }
    fn groups(&mut self) -> Vec<Vec<usize>> {
        let keys: Vec<usize> = self.parent.keys().copied().collect();
        let mut by_root: HashMap<usize, Vec<usize>> = HashMap::new();
        for k in keys {
            let r = self.find(k);
            by_root.entry(r).or_default().push(k);
        }
        by_root
            .into_values()
            .filter(|grp| grp.len() > 1)
            .map(|mut grp| {
                grp.sort_unstable();
                grp
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use costmodel::DEFAULT_PAGE_BYTES;

    fn h100() -> &'static costmodel::hwspec::GpuSpec {
        costmodel::hwspec::registry::lookup("H100 SXM5").unwrap()
    }

    /// A tiny attention prefix: RmsNorm → {q_proj, k_proj}. The norm output is
    /// handed off to both projections.
    fn plan() -> LayerPlan {
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };
        LayerPlan {
            ops: vec![
                OpSpec {
                    name: "input_norm".into(),
                    inputs: vec!["x".into(), "norm.w".into()],
                    output: "h".into(),
                    kind: OpKind::Row(RowShape {
                        rows: 4096,
                        feat: 4096,
                        operands: 2,
                        reduce: true,
                    }),
                    weight_dtype: nn_graph::DType::BF16,
                    compute_dtype: nn_graph::DType::BF16,
                },
                OpSpec {
                    name: "q_proj".into(),
                    inputs: vec!["h".into(), "q.w".into()],
                    output: "q".into(),
                    kind: OpKind::Gemm(g),
                    weight_dtype: nn_graph::DType::BF16,
                    compute_dtype: nn_graph::DType::BF16,
                },
                OpSpec {
                    name: "k_proj".into(),
                    inputs: vec!["h".into(), "k.w".into()],
                    output: "k".into(),
                    kind: OpKind::Gemm(g),
                    weight_dtype: nn_graph::DType::BF16,
                    compute_dtype: nn_graph::DType::BF16,
                },
            ],
        }
    }

    #[test]
    fn single_unit_handoff_eliminates_round_trip_and_colocates() {
        let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
        let (g, cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();

        // `h` produced once, consumed twice ⇒ two same-unit hand-offs.
        assert_eq!(cons.handoffs.len(), 2);
        for hf in &cons.handoffs {
            assert!(!hf.cross_unit);
            assert!(matches!(
                g.nodes[hf.producer_dma_out],
                TileNode::DmaOut { resident: true, .. }
            ));
            assert!(matches!(
                g.nodes[hf.consumer_dma_in],
                TileNode::DmaIn { resident: true, .. }
            ));
        }
        // norm + q + k pinned to one SM.
        assert_eq!(cons.colocation_groups.len(), 1);
        assert_eq!(cons.colocation_groups[0].len(), 3);
        // Everything on unit 0.
        assert!(cons.placement.values().all(|&u| u == 0));
    }

    #[test]
    fn weights_are_staged_once() {
        let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
        let (g, _) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();
        let mut seen = std::collections::HashSet::new();
        for n in &g.nodes {
            if let TileNode::DmaIn {
                tensor,
                resident: false,
            } = n
            {
                assert!(seen.insert(tensor.clone()), "tensor {tensor} staged twice");
            }
        }
        // x, norm.w, q.w, k.w = 4 distinct DRAM stages.
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn gemm_splits_across_units_with_join() {
        // Two GPUs over unified memory: one Linear splits in two, then joins.
        let soc = Soc::homogeneous(h100(), 2, DEFAULT_PAGE_BYTES);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };
        let layer = LayerPlan {
            ops: vec![OpSpec {
                name: "o_proj".into(),
                inputs: vec!["act".into(), "o.w".into()],
                output: "out".into(),
                kind: OpKind::Gemm(g),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            }],
        };
        let (graph, cons) = assemble(&soc, &layer, SramPolicy::Stream, None).unwrap();

        // Two placed GEMM regions + one Join.
        let gemms: Vec<usize> = graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                matches!(
                    n,
                    TileNode::Compute {
                        kind: Compute::Gemm(_),
                        ..
                    }
                )
            })
            .map(|(i, _)| i)
            .collect();
        let joins = graph
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    n,
                    TileNode::Compute {
                        kind: Compute::Join,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(gemms.len(), 2);
        assert_eq!(joins, 1);

        // The two regions land on different units.
        let units: std::collections::HashSet<UnitId> =
            gemms.iter().map(|i| cons.placement[i]).collect();
        assert_eq!(units, [0, 1].into_iter().collect());

        // Activation staged once; the weight sliced into two distinct stages.
        let staged: Vec<&String> = graph
            .nodes
            .iter()
            .filter_map(|n| match n {
                TileNode::DmaIn {
                    tensor,
                    resident: false,
                } => Some(tensor),
                _ => None,
            })
            .collect();
        assert_eq!(staged.iter().filter(|t| **t == "act").count(), 1);
        assert!(staged.iter().any(|t| t.starts_with("o.w#n0")));
        assert!(staged.iter().any(|t| **t == slice_name("o.w", 2048)));

        // The Join reads both slices: one same-unit (resident) + one cross-unit.
        let mut resident = 0;
        let mut cross = 0;
        for hf in &cons.handoffs {
            if hf.cross_unit {
                cross += 1;
            } else {
                resident += 1;
            }
        }
        assert_eq!((resident, cross), (1, 1));
        assert!(cons.unified_memory);
    }

    /// Find the `TileDep` whose consumer compute node has op name `name`.
    fn dep_for<'a>(g: &TileGraph, cons: &'a ConstraintSet, name: &str) -> &'a TileDep {
        cons.tile_deps
            .iter()
            .find(|d| matches!(&g.nodes[d.consumer], TileNode::Compute { op, .. } if op == name))
            .expect("no tile dep for consumer")
    }

    #[test]
    fn rms_gemm_tile_coupling() {
        // norm → q_proj: the GEMM's M-tiles depend only on the norm row-blocks
        // covering their rows.
        let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
        let (g, cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();

        let td = dep_for(&g, &cons, "q_proj");
        let pd = cons.domains[&td.producer];
        let cd = cons.domains[&td.consumer];
        let (TileDomain::Row { br, .. }, TileDomain::Gemm { bm, .. }) = (pd, cd) else {
            panic!("expected Row→Gemm domains");
        };

        let edges = materialize_tile_deps(&pd, &cd, &td.dep);
        assert!(!edges.is_empty());
        // Every edge is a genuine M-range overlap between the two blockings.
        for e in &edges {
            let (i, r) = (e.consumer_coord[0], e.producer_coord[0]);
            let (c0, c1) = (i * bm, i * bm + bm);
            let (p0, p1) = (r * br, r * br + br);
            assert!(p0 < c1 && c0 < p1, "tile ({i},*) ⟂ row-block {r}");
        }

        // The counter threshold equals the materialized in-degree per consumer tile.
        let thresholds = consumer_thresholds(&pd, &cd, &td.dep);
        let mut counted: HashMap<Vec<i64>, u32> = HashMap::new();
        for e in &edges {
            *counted.entry(e.consumer_coord.clone()).or_insert(0) += 1;
        }
        assert_eq!(thresholds, counted);
    }

    #[test]
    fn gemm_n_axis_is_free() {
        // For a fixed M-tile i, every N-tile j depends on the same producer set:
        // the norm does not vary along the GEMM's output-feature axis.
        let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
        let (g, cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();
        let td = dep_for(&g, &cons, "q_proj");
        let (pd, cd) = (cons.domains[&td.producer], cons.domains[&td.consumer]);

        // producer-set(i, j) keyed by (i, j).
        let mut by_ij: HashMap<(i64, i64), Vec<i64>> = HashMap::new();
        for e in materialize_tile_deps(&pd, &cd, &td.dep) {
            by_ij
                .entry((e.consumer_coord[0], e.consumer_coord[1]))
                .or_default()
                .push(e.producer_coord[0]);
        }
        for v in by_ij.values_mut() {
            v.sort_unstable();
        }
        // Group by i; all j must share the identical producer set.
        let mut by_i: HashMap<i64, Vec<Vec<i64>>> = HashMap::new();
        for ((i, _), set) in by_ij {
            by_i.entry(i).or_default().push(set);
        }
        for sets in by_i.values() {
            assert!(
                sets.windows(2).all(|w| w[0] == w[1]),
                "N-tiles disagree on producers"
            );
        }
    }

    #[test]
    fn split_region_keeps_m_coupling() {
        // norm → a split GEMM (2 units). Each region still couples on the shared
        // activation's M axis; the N split does not touch the dependency.
        let soc = Soc::homogeneous(h100(), 2, DEFAULT_PAGE_BYTES);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };
        let layer = LayerPlan {
            ops: vec![
                OpSpec {
                    name: "input_norm".into(),
                    inputs: vec!["x".into(), "norm.w".into()],
                    output: "h".into(),
                    kind: OpKind::Row(RowShape {
                        rows: 4096,
                        feat: 4096,
                        operands: 2,
                        reduce: true,
                    }),
                    weight_dtype: nn_graph::DType::BF16,
                    compute_dtype: nn_graph::DType::BF16,
                },
                OpSpec {
                    name: "o_proj".into(),
                    inputs: vec!["h".into(), "o.w".into()],
                    output: "out".into(),
                    kind: OpKind::Gemm(g),
                    weight_dtype: nn_graph::DType::BF16,
                    compute_dtype: nn_graph::DType::BF16,
                },
            ],
        };
        let (_, cons) = assemble(&soc, &layer, SramPolicy::Stream, None).unwrap();

        // One tile dep per region (the join is layout — no coupling).
        let region_deps: Vec<&TileDep> = cons
            .tile_deps
            .iter()
            .filter(|d| d.dep.tensor == "h")
            .collect();
        assert_eq!(region_deps.len(), 2);
        for d in region_deps {
            assert_eq!(d.dep.couple[0].consumer_axis, 0); // M axis
            assert_eq!(d.dep.couple[0].producer_axis, 0); // norm row axis
        }
    }

    #[test]
    fn row_to_row_passthrough() {
        // norm → activation, both row-wise: each consumer row-block couples to the
        // producer row-blocks overlapping its rows (1:1 when the blocks align; the
        // memory-bound row cost ties, so the chosen block sizes may differ).
        let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
        let layer = LayerPlan {
            ops: vec![
                OpSpec {
                    name: "norm".into(),
                    inputs: vec!["x".into(), "norm.w".into()],
                    output: "h".into(),
                    kind: OpKind::Row(RowShape {
                        rows: 4096,
                        feat: 4096,
                        operands: 2,
                        reduce: true,
                    }),
                    weight_dtype: nn_graph::DType::BF16,
                    compute_dtype: nn_graph::DType::BF16,
                },
                OpSpec {
                    name: "act".into(),
                    inputs: vec!["h".into()],
                    output: "y".into(),
                    kind: OpKind::Row(RowShape {
                        rows: 4096,
                        feat: 4096,
                        operands: 1,
                        reduce: false,
                    }),
                    weight_dtype: nn_graph::DType::BF16,
                    compute_dtype: nn_graph::DType::BF16,
                },
            ],
        };
        let (g, cons) = assemble(&soc, &layer, SramPolicy::Stream, None).unwrap();
        let td = dep_for(&g, &cons, "act");
        let (pd, cd) = (cons.domains[&td.producer], cons.domains[&td.consumer]);
        assert!(matches!(pd, TileDomain::Row { .. }) && matches!(cd, TileDomain::Row { .. }));

        // Every materialized edge is a genuine row-range overlap, and the counter
        // threshold equals the per-consumer in-degree.
        let (TileDomain::Row { br: pb, .. }, TileDomain::Row { br: cb, .. }) = (pd, cd) else {
            unreachable!()
        };
        let edges = materialize_tile_deps(&pd, &cd, &td.dep);
        assert!(!edges.is_empty());
        for e in &edges {
            let (cr, pr) = (e.consumer_coord[0], e.producer_coord[0]);
            assert!(pr * pb < cr * cb + cb && cr * cb < pr * pb + pb);
        }
        let thresholds = consumer_thresholds(&pd, &cd, &td.dep);
        let mut counted: HashMap<Vec<i64>, u32> = HashMap::new();
        for e in &edges {
            *counted.entry(e.consumer_coord.clone()).or_insert(0) += 1;
        }
        assert_eq!(thresholds, counted);
        // When the blocks happen to align, the coupling is exactly 1:1.
        if pb == cb {
            assert!(thresholds.values().all(|&c| c == 1));
        }
    }
}
