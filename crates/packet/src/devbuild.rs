//! Build a device packet program and serialise it for the runtime.
//!
//! This is the compiler side of the device ISA in [`crate::dev`]. It turns a list of
//! ops into the flattened tables the persistent interpreter walks: one instruction
//! table, one stream per CU, and a counter graph.
//!
//! # Why CUs are a first-class argument
//!
//! plow exists to schedule a NETWORK, not to run one op fast. The interesting
//! decision is not "how fast is this GEMM" but "which ops can be in flight at the
//! same time, on which CUs". So [`Builder::emit`] takes the CU set the op runs on:
//! give two independent ops disjoint CU sets and they overlap; give them both the
//! whole machine and they serialise behind a counter. Both are expressible, and the
//! trace ([`crate::dev::TraceRec`]) shows which one you actually got.
//!
//! # The counter contract
//!
//! Each op gets one counter. Every workgroup that runs the op bumps it once on
//! completion, so the counter reaches `cus.len()` exactly when the op is done. A
//! consumer waits for `threshold == number of producing workgroups` — which is why
//! the threshold is derived from the producer's CU set and never hand-written.

use core::mem::size_of;
use std::collections::{BTreeSet, HashSet};

use crate::dev::{DevInst, DevOp, StreamEnt, Wait, SE_FINE, TENSOR_NONE, TENSOR_NONE_I};
use crate::rope::GenTensor;

/// Edges that survive transitive reduction: drop A→C when a path A→…→C of
/// length ≥ 2 exists. `edges` is `(producer, consumer)` over op indices.
///
/// Lives here rather than in `plowrt` because BOTH the emitter (which applies
/// the reduction) and `plowrt disasm --counters` (which reports what it would
/// save) must compute the same set — two implementations would drift, and the
/// disassembler's numbers appear in committed `perf-data/` write-ups.
///
/// For a DAG the transitive reduction is unique and has the same transitive
/// closure as the input, so removing every covered edge AT ONCE is safe even
/// when a justifying path's own edges are also removed: each is in turn covered
/// by a further path, and acyclicity makes the induction terminate.
pub fn transitive_reduction(n: usize, edges: &BTreeSet<(u32, u32)>) -> BTreeSet<(u32, u32)> {
    // Reachability by 2+ hops. The op DAG is emitted in topological order
    // (producer index < consumer index), so a single reverse sweep suffices.
    let mut succ: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    for &(a, b) in edges {
        succ[a as usize].insert(b);
    }
    // reach[a] = every node reachable from a (any distance ≥ 1).
    let mut reach: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    for a in (0..n).rev() {
        let mut r: HashSet<u32> = HashSet::new();
        for &b in &succ[a] {
            r.insert(b);
            for &c in &reach[b as usize] {
                r.insert(c);
            }
        }
        reach[a] = r;
    }
    let mut out = BTreeSet::new();
    for &(a, b) in edges {
        // Redundant iff some other direct successor of a reaches b.
        let redundant = succ[a as usize]
            .iter()
            .any(|&m| m != b && reach[m as usize].contains(&b));
        if !redundant {
            out.insert((a, b));
        }
    }
    out
}

/// The tuning-store key for one decode-GEMV shape.
///
/// Deliberately the same grammar as `tunedb::gemm_op_case` — `"<family>/<m>x<n>x<k>/<quant>"` —
/// so the two families sort together in one `kernel_measurement.jsonl` and a reader can tell
/// which phase a record belongs to from the key alone.
///
/// It lives in `packet` rather than in `tunedb` for the reason `tunedb`'s own `Cargo.toml`
/// gives for the GEMM twin, applied one crate lower: the thing that PRODUCES these keys is
/// [`Builder::emit_dep`], and `packet` has no dependencies by design. `tunedb` re-exports it,
/// so there is still exactly one definition.
pub fn gemv_op_case(family: &str, m: u32, n: u32, k: u32, quant: &str) -> String {
    format!("{family}/{m}x{n}x{k}/{quant}")
}

/// Op cases the GEMV tuning store can answer for, installed once by the emitter.
///
/// `packet` cannot read the store (no dependencies, and `tunedb` sits above it), so the
/// answer is pushed down instead of pulled up. Empty/unset means "no store" and every shape
/// reports `MISS`, which is the honest cold-start reading and the one the GEMM path took for
/// GLM-5.2 for the whole campaign before anyone looked.
static TUNED_GEMV_CASES: std::sync::OnceLock<std::collections::HashSet<String>> =
    std::sync::OnceLock::new();

/// Install the set of GEMV op cases the store holds qualified, non-stale records for.
///
/// Idempotent by construction ([`std::sync::OnceLock`]); a second call is ignored, because a
/// compile is a process and the store is read once.
pub fn set_tuned_gemv_cases(cases: std::collections::HashSet<String>) {
    let _ = TUNED_GEMV_CASES.set(cases);
}

/// `PLOW_TUNE_DUMP=1` -> one `TUNEDUMP_GEMV` line per emitted decode-GEMV op.
///
/// # This is the whole point, so it is stated here and not in a script
///
/// `scripts/rebench_tune_gemm.sh`'s shape list was hand-authored, and GLM-5.2 prefill was
/// therefore 100% unmeasured for as long as the tuner existed — every lookup missed, while
/// the differential test kept passing because SOME qualified record existed for SOME model.
/// It was invisible from outside: the calibration tier still read "measured". `PLOW_TUNE_DUMP`
/// on the GEMM path is what made it visible, and re-deriving the list from the dump is what
/// stops it recurring.
///
/// The GEMV path had no dump at all, no store, and no selection function — so its campaign
/// list would have had to be hand-authored from scratch, which is the same mistake with a
/// clean sheet of paper. This hook is placed in [`Builder::emit_dep`], the one function every
/// emitter in `devgen` funnels through, so the census is DERIVED: a GEMV emit site added
/// tomorrow appears in it without anyone remembering to instrument it.
///
/// Line format, chosen so `sort -u` over the dump IS the campaign list:
/// ```text
/// TUNEDUMP_GEMV <m> <n> <k> <quant> <PLOW_DOP_...> <HIT|MISS>
/// ```
fn tune_dump_gemv(op: DevOp, inst: &DevInst) {
    if std::env::var("PLOW_TUNE_DUMP").ok().as_deref() != Some("1") {
        return;
    }
    let Some((fam, m, n, k, quant)) = op.gemv_case(&inst.i) else {
        return;
    };
    let hit = TUNED_GEMV_CASES
        .get()
        .is_some_and(|s| s.contains(&gemv_op_case(fam, m, n, k, quant)));
    eprintln!(
        "TUNEDUMP_GEMV {m} {n} {k} {quant} {} {}",
        op.c_name(),
        if hit { "HIT" } else { "MISS" }
    );
}

/// A tensor the program refers to by handle. The runtime allocates it and fills the
/// device pointer table; the program only ever sees the handle.
///
/// `init` carries bytes the COMPILER computed and the runtime should just upload. RoPE
/// tables live here: Gemma's full-attention layers use a partial rotary where only the
/// first 64 of 256 angles are nonzero and the remaining 192 are NoPE (cos=1, sin=0), and
/// that subtlety belongs in exactly one place. Duplicating it in the runtime is how you
/// end up with two implementations that disagree and a model that is fluent but wrong.
#[derive(Clone, Debug)]
pub struct TensorDecl {
    pub name: String,
    pub bytes: u64,
    pub init: Option<Vec<u8>>,
}

/// A dependency on a producer op.
///
/// `Coarse` is the original contract: wait for **every** workgroup of the producer. It is
/// correct for any edge and is the only correct choice for a reduction, where each consumer
/// element genuinely depends on all of the producer's output (`o_proj` reduces over all of
/// K, `down_proj` over all of I).
///
/// `Fine` is for edges whose dependency is *structured* — a head, a column block, a KV
/// split. Then consumer slice `s` needs only the handful of producer slices in `map[s]`,
/// and making it wait for the other 250-odd is pure loss: measured on the real model, the
/// gate spends **2.63 ms of a 16.9 ms decode token** waiting for a straggler after half the
/// machine has finished, and the straggler is diffuse rather than one persistently slow CU.
///
/// Use `Fine` **only** where the dependency really is sparse. On an all-to-all edge it is
/// strictly worse: the producer would have to bump one counter per consumer slice
/// (256 × 256 atomics) to say what a single coarse counter says with one.
pub enum Dep {
    /// Wait for all `blocks` workgroups of the producer.
    Coarse(u32),
    /// Wait only for the producer slices that feed each of my slices.
    Fine {
        /// The producer's counter id, as returned by [`Builder::emit`].
        producer: u32,
        /// `map[my_slice]` = the producer slices that feed my slice.
        map: Vec<Vec<u32>>,
    },
}

impl Dep {
    fn producer(&self) -> u32 {
        match self {
            Dep::Coarse(c) => *c,
            Dep::Fine { producer, .. } => *producer,
        }
    }
}

/// Slice-level locality census — see [`Builder::locality_census_stats`].
#[derive(Clone, Debug)]
struct LocalityCensus {
    ops: usize,
    slices: u64,
    domains: usize,
    map_name: &'static str,
    /// Slice-level producer→consumer pairs in the whole program.
    pairs: u64,
    /// Of those, the ones on an edge where the consumer slice reads EVERY producer slice.
    all_to_all_pairs: u64,
    /// Same-domain pairs under the emitted mapping (`L2Layout::domain_of`).
    same_current: u64,
    /// Same-domain pairs under a greedy predecessor-affinity pass under a balance cap.
    same_greedy: u64,
    /// Same-domain pairs if each consumer slice could pick its own best domain with producers
    /// pinned and balance ignored — an unachievable upper bound, useful as a ceiling.
    same_ceiling: u64,
    moved_slices: u64,
    moved_ops: u64,
}

/// One op, before flattening.
struct Op {
    inst: DevInst,
    cus: Vec<u32>,
    deps: Vec<Dep>,
    counter: u32,   // the coarse counter this op bumps
    work: Vec<u32>, // per-slice cost, from the cost model. See `select_granularity`.
}

/// Dependency graph for one complete emitted program. Fusion discovery runs on this graph only
/// after every model block has been lowered; packet order is retained separately as a scheduling
/// constraint.
struct ProgramGraph {
    predecessors: Vec<Vec<usize>>,
    successors: Vec<Vec<usize>>,
}

impl ProgramGraph {
    fn from_ops(ops: &[Op]) -> Self {
        let mut predecessors = vec![Vec::new(); ops.len()];
        let mut successors = vec![Vec::new(); ops.len()];
        for (consumer, op) in ops.iter().enumerate() {
            for dep in &op.deps {
                let producer = dep.producer() as usize;
                assert!(
                    producer < consumer,
                    "program dependency {producer} -> {consumer} is not topological"
                );
                predecessors[consumer].push(producer);
                successors[producer].push(consumer);
            }
        }
        Self {
            predecessors,
            successors,
        }
    }
}

/// How the hardware maps a LOGICAL workgroup index to an L2 locality domain.
///
/// This is the single fact the whole placement feature rests on, and the two vendors do not
/// agree — so it is carried as data rather than assumed. `interp`'s `cu` is `blockIdx.x`, a
/// logical index; the domain a packet is placed in has to be the domain the hardware will
/// actually run that workgroup on, or placement destroys locality instead of creating it while
/// still emitting perfectly correct tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum L2Map {
    /// Workgroup *n* -> domain `n / sms_per_partition`. NVIDIA: consecutive blocks fill a GPC
    /// before moving to the next.
    Block,
    /// Workgroup *n* -> domain `n % partition_count`. AMD CDNA3/CDNA4: the hardware dispatcher
    /// assigns workgroups to XCDs round-robin.
    ///
    /// **Measured on MI355X** (`runtime/tests/xcd_map_gfx950_test.hip`, `HW_REG_XCC_ID` per
    /// workgroup): `n % 8` predicts the true XCC id for **100.0%** of workgroups across every
    /// geometry probed — 256x512 at occupancy 1 (the interpreter's decode grid), 512 blocks at
    /// occupancy 2, 64 blocks, 256 threads at occupancy 4 — and reproducibly across launches.
    /// `n / 32` scores 12.5%, which is just the coincidence `n/32 == n%8`, not partial
    /// correctness. At 256 blocks / occ 1 the 32 workgroups of each XCD sit on 32 DISTINCT
    /// physical CUs, so a domain really is a 32-CU / 4 MiB-L2 group.
    RoundRobin,
}

/// The L2 partition geometry a program is placed against, plus how workgroups reach it.
///
/// `sms` and `domains` come from `hwspec::GpuSpec::l2_partitioning` (XCD on MI300/MI350, GPC on
/// H100/B200) — not from an env-supplied constant. `map` comes from the target vendor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct L2Layout {
    /// SMs/CUs per L2 partition. Only the [`L2Map::Block`] formula divides by it; under
    /// [`L2Map::RoundRobin`] it is carried for reporting and for the runtime's own checks.
    pub sms: u32,
    /// Number of L2 partitions — the domain count, and the number of `gq_seg_ofs` windows.
    pub domains: u32,
    pub map: L2Map,
}

impl L2Layout {
    /// The locality domain of logical workgroup `cu`. Always in `0..domains`.
    #[inline]
    pub fn domain_of(&self, cu: u32) -> u32 {
        match self.map {
            L2Map::Block => cu / self.sms.max(1),
            L2Map::RoundRobin => cu % self.domains.max(1),
        }
    }
}

pub struct Builder {
    n_cu: u32,
    ops: Vec<Op>,
    tensors: Vec<TensorDecl>,
    gen: Vec<GenTensor>,
    /// L2-domain-aware placement (`PLOW_L2_PLACE`). `None` ⇒ off; the blob is
    /// byte-identical and `seg` keeps its wave-class meaning. When set,
    /// [`Builder::finish`] carries the locality domain independently in flags
    /// and groups `gq_stream` into one device queue per ordered segment and domain.
    /// The `cus` sets are NOT touched — placement is dynamic (cursor-claimed) at
    /// runtime, so it cannot regress disjoint `Builder::split` placements.
    /// See the design notes.
    place_l2: Option<L2Layout>,
    /// The target cannot honour `PLOW_UNISEG` — set by an emitter that knows what it is building
    /// for. See [`Builder::deny_uniseg`].
    uniseg_denied: bool,
    /// See [`Builder::force_uniseg`].
    uniseg_forced: bool,
    /// See [`Builder::set_gq_order_asap`]. Default from `PLOW_GQ_ORDER=asap`.
    gq_order_asap: bool,
    /// Split descriptor-consuming prefill families into independent wave classes.
    /// Callers must enable this only for prefill programs.
    packed_prefill_segments: bool,
    /// Isolate structurally compatible MXFP4 grouped-MoE stage-2 boundaries.
    lean_moe_stage2_segments: bool,
    /// Isolate structurally compatible MXFP4 grouped-MoE stage-1 packets.
    lean_moe_stage1_segments: bool,
    /// Isolate fixed-order f32 grouped-MoE prefill combines.
    lean_moe_combine_segments: bool,
    /// Rewrite eligible replicated-input grouped-MoE prefill boundaries to whole-expert/full-I.
    moe_prefill_ep_degree: Option<u32>,
    /// Isolate BT64/D128 chunk-KDA intra packets for a standalone gfx950 object.
    lean_kda_intra_segments: bool,
    /// Mark isolated BT64/D128 chunk-KDA intra packets for the wave-item object.
    kda_intra_wave_items_segments: bool,
    /// Isolate exact qpre BT64/D128 Wu->carry pairs for standalone gfx950 objects.
    lean_kda_key_factor_segments: bool,
    /// Isolate adjacent FlashMlaDecode+MlaMergeFold pairs for a gfx950 object.
    decode_mla_segments: bool,
    /// Isolate adjacent grouped decode GLU+DOWN pairs for ordered raw launches.
    decode_grouped_moe_segments: bool,
    /// Isolate XReduceTwoShot packets for the gfx950 wave-RS interpreter object.
    xreduce_wave_rs_segments: bool,
    /// Fold an eligible materialized Residual into its AttnRes consumer.
    fuse_materialized_residual_inputs: bool,
    /// Slices per machine-filling decode GEMV, as a multiple of `n_cu`. 1 (default) ⇒
    /// byte-identical. See [`Builder::set_gemv_split`].
    gemv_split: u32,
    /// Re-declaring a tensor name returns the existing handle instead of appending.
    /// `false` (default) ⇒ byte-identical. See [`Builder::set_tensor_dedup`].
    tensor_dedup: bool,
    /// Coarse dep edges removed by the transitive reduction in [`Builder::finish`].
    /// Reported so an emitter can log it; the reduction itself is unconditional.
    tr_dropped: usize,
}

fn mla_v2_segment(op: u16, n_tok: u32) -> bool {
    n_tok >= 2048 && (op == DevOp::FlashMlaPrefill as u16 || op == DevOp::FlashMlaPrefillFp8 as u16)
}

// Experimental packed-prefill segment classes. These values describe an operator
// family, not a model. The serving runtime independently re-derives and purity-checks
// them before it can select a family object.
fn packed_prefill_segment_class(op: u16) -> Option<u8> {
    if op == DevOp::RmsNorm as u16
        || op == DevOp::HeadNormRope as u16
        || op == DevOp::HeadNormRopeFp8 as u16
    {
        Some(5)
    } else if op == DevOp::FlashMlaPrefill as u16 || op == DevOp::FlashMlaPrefillFp8 as u16 {
        Some(6)
    } else if op == DevOp::KdaStateStep as u16
        || op == DevOp::KdaConv3 as u16
        || op == DevOp::KdaStateStepG as u16
        || op == DevOp::KdaChunkPrepare as u16
        || op == DevOp::KdaChunkIntra as u16
        || op == DevOp::KdaChunkWu as u16
        || op == DevOp::KdaChunkCarry as u16
    {
        Some(7)
    } else {
        None
    }
}

fn packed_prefill_segmenting_needed(
    uniseg: bool,
    enabled: bool,
    mut ops: impl Iterator<Item = u16>,
) -> bool {
    !uniseg && enabled && ops.any(|op| packed_prefill_segment_class(op).is_some())
}

fn lean_moe_stage2_inst(d: &DevInst) -> bool {
    d.op == DevOp::MoeGroupDownPf as u16
        && d.i[3] == 2
        && d.i[1] == 384
        && d.i[0] != 0
        && d.i[0].is_multiple_of(16)
        && d.i[2] != 0
        && d.i[4] == 0
        && d.i[5] == 0
        && d.t[5] != TENSOR_NONE
        && d.t[6] != TENSOR_NONE
        && d.t[7] != TENSOR_NONE
}

fn lean_moe_stage2_pair(ops: &[Op], i: usize) -> bool {
    let Some((d, c)) = ops.get(i).zip(ops.get(i + 1)) else {
        return false;
    };
    let (d, c) = (&d.inst, &c.inst);
    lean_moe_stage2_inst(d)
        && c.op == DevOp::MoeCombinePf as u16
        && c.t[1] == TENSOR_NONE
        && c.t[2] == TENSOR_NONE
        && c.t[3] == d.t[0]
        && c.i[0] == d.i[0]
        && c.i[1] != 0
        && c.i[2] != 0
        && c.i[3] == 0
        && c.i[4] == 0
        && c.i[7] == 0
}

fn lean_moe_stage1_inst(inst: &DevInst) -> bool {
    inst.op == DevOp::MoeGroupGluPf as u16
        && inst.i[0] == 384
        && inst.i[1] == 3584
        && inst.i[2] == 896
        && inst.i[3] == 2
        && inst.i[4] == 0
        && inst.i[5] <= 2
        && inst.i[6] == 0
        && inst.i[7] == 0
        && inst.t[..8].iter().all(|&t| t != TENSOR_NONE)
}

fn lean_moe_combine_inst(inst: &DevInst) -> bool {
    inst.op == DevOp::MoeCombinePf as u16
        && inst.t[0] != TENSOR_NONE
        && inst.t[3] != TENSOR_NONE
        && inst.i[0] != 0
        && inst.i[1] == 16
        && inst.i[2] != 0
        && inst.i[3..].iter().all(|&v| v == 0)
        && inst.f.iter().all(|&v| v.to_bits() == 0)
        && inst.j.iter().all(|&v| v == 0)
}

fn lean_kda_key_factor_pair(ops: &[Op], i: usize) -> bool {
    let Some((wu, carry)) = ops.get(i).zip(ops.get(i + 1)) else {
        return false;
    };
    let (w, c) = (&wu.inst, &carry.inst);
    w.op == DevOp::KdaChunkWu as u16
        && c.op == DevOp::KdaChunkCarry as u16
        && w.i[0] >= 512
        && w.i[0] == c.i[0]
        && w.i[1] != 0
        && w.i[1] == c.i[1]
        && w.i[2] == 128
        && w.i[2] == c.i[2]
        && w.i[3] == 128
        && w.i[3] == c.i[3]
        && w.i[4] == 1
        && c.i[4] == 1
        && w.i[5..].iter().all(|&v| v == 0)
        && c.i[5..].iter().all(|&v| v == 0)
        && w.t.iter().all(|&t| t != TENSOR_NONE)
        && c.t.iter().all(|&t| t != TENSOR_NONE)
        && c.t[2] == w.t[7]
        && c.t[3] == w.t[3]
        && c.t[4] == w.t[0]
        && c.t[5] == w.t[1]
        && c.t[7] == w.t[5]
        && w.f[0].to_bits() == c.f[0].to_bits()
        && w.f[0].is_finite()
        && w.f[0] > 0.0
        && w.f[1..].iter().all(|&v| v.to_bits() == 0)
        && c.f[1..].iter().all(|&v| v.to_bits() == 0)
        && w.j.iter().all(|&v| v == 0)
        && c.j.iter().all(|&v| v == 0)
        && carry.deps.iter().any(|d| d.producer() as usize == i)
}

impl Builder {
    pub fn new(n_cu: u32) -> Self {
        Self {
            n_cu,
            ops: Vec::new(),
            tensors: Vec::new(),
            gen: Vec::new(),
            place_l2: None,
            uniseg_denied: false,
            uniseg_forced: false,
            gq_order_asap: std::env::var("PLOW_GQ_ORDER").ok().as_deref() == Some("asap"),
            packed_prefill_segments: false,
            lean_moe_stage2_segments: false,
            lean_moe_stage1_segments: false,
            lean_moe_combine_segments: false,
            moe_prefill_ep_degree: None,
            lean_kda_intra_segments: false,
            kda_intra_wave_items_segments: false,
            lean_kda_key_factor_segments: false,
            decode_mla_segments: false,
            decode_grouped_moe_segments: false,
            xreduce_wave_rs_segments: false,
            fuse_materialized_residual_inputs: true,
            gemv_split: 1,
            tensor_dedup: false,
            tr_dropped: 0,
        }
    }

    /// Enable L2-domain-aware placement from `hwspec::GpuSpec::l2_partitioning` plus the
    /// target's workgroup->domain map. `None` (default) leaves the wave-class `seg` and a
    /// byte-identical blob.
    ///
    /// [`Builder::finish`] may still decline under [`L2Map::Block`] if
    /// `n_cu > domains·sms`, where `cu/sms` would orphan packets. Host segments
    /// and L2 domains are independent fields, so segmented programs remain placeable.
    pub fn set_l2_placement(&mut self, layout: Option<L2Layout>) {
        self.place_l2 = layout;
    }

    /// Refuse to honour `PLOW_UNISEG` for this program, whatever the environment says.
    ///
    /// `PLOW_UNISEG` collapses every op into one segment. That is harmless on sm_120 — its
    /// interpreter runs the whole program in one cooperative launch at a fixed block size and never
    /// reads a wave class — and DESTRUCTIVE on gfx950, where the host relaunches once per segment
    /// and reads the class back from `seg`. With one segment, the segment contains flash packets,
    /// the host dispatches the entire prefill program on the 4-wave flash object, and that object's
    /// body is `if (op == FLASH_PREFILL…)` with no switch: every GEMM, norm and lm_head is silently
    /// dropped and the logits come back zero.
    ///
    /// The POLICY lives with the caller because only the caller knows the target; this type just
    /// honours it. Same split as [`Builder::set_l2_placement`], and for the same reason: a flag
    /// whose meaning depends on the backend cannot be resolved in a backend-agnostic builder.
    pub fn deny_uniseg(&mut self) {
        self.uniseg_denied = true;
    }

    /// Order each global-queue window by EARLIEST START instead of emit order (`PLOW_GQ_ORDER=asap`).
    ///
    /// The global queue hands out entries in stream order and a workgroup that claims a gated
    /// entry SPINS on it. Emit order is topological but not ready-ordered: K3's decode emits the
    /// shared-expert `GemvGlu -> Gemv` (ready the moment `AttnRes` lands) AFTER the routed chain
    /// `MoeRouterTopk -> MoeGroupGlu -> MoeGroupDown -> MoeCombine -> XReduce -> Gemv`, so all 256
    /// workgroups claim the routed slices and spin through the 22 us router while the shared
    /// expert — which could have run entirely under that spin — waits for them, and the layer's
    /// closing `XReduce` waits for the shared expert. MEASURED (gfx950 TP8, one-token trace,
    /// `scripts/k3_trace_wg.py`): 24.2 us/layer x 92 MoE layers = 2.2 ms/token of critical path.
    ///
    /// Ranks are a unit-cost list schedule: `start(op) = max over producers (start + cost)`,
    /// cost 1 per op and 3 for a single-workgroup op (the b=1 router/AttnRes bodies are 2-3
    /// GEMV bodies long). Every window is STABLE-sorted by rank, so ties keep emit order and a
    /// consumer's rank is strictly above every producer's: the per-window order stays
    /// topological, which is what the queue's deadlock-freedom argument (interp.hip, the
    /// `gq_claim` note) needs. Nothing else moves — static streams, counters, and the segment
    /// windows are untouched.
    pub fn set_gq_order_asap(&mut self, on: bool) {
        self.gq_order_asap = on;
    }

    fn gq_asap_ranks(&self) -> Vec<u32> {
        let mut start = vec![0u32; self.ops.len()];
        for i in 0..self.ops.len() {
            let mut s = 0u32;
            for d in &self.ops[i].deps {
                let p = d.producer() as usize;
                if p < i {
                    let cost = if self.ops[p].inst.blocks <= 1 { 3 } else { 1 };
                    s = s.max(start[p] + cost);
                }
            }
            start[i] = s;
        }
        start
    }

    /// T18: force this program to ONE segment regardless of `PLOW_UNISEG`. Set by devgen on
    /// SMALL prefill buckets (`PLOW_UNISEG_MAX_T`): a tail chunk of ~50 tokens pays ~480
    /// segment launches (~40 ms measured) for ~5 ms of work — one launch on the full fat
    /// object wins outright. No-op if `deny_uniseg` was called (the AMD target reads `seg`).
    pub fn force_uniseg(&mut self) {
        self.uniseg_forced = true;
    }

    pub fn set_packed_prefill_segments(&mut self, enabled: bool) {
        self.packed_prefill_segments = enabled;
    }

    pub fn set_lean_moe_stage2_segments(&mut self, enabled: bool) {
        self.lean_moe_stage2_segments = enabled;
    }

    pub fn set_lean_moe_stage1_segments(&mut self, enabled: bool) {
        self.lean_moe_stage1_segments = enabled;
    }

    pub fn set_lean_moe_combine_segments(&mut self, enabled: bool) {
        self.lean_moe_combine_segments = enabled;
    }

    /// Enable the model-independent replicated-input expert-parallel rewrite.
    ///
    /// Eligibility is proven from the emitted graph: an MXFP4 align/GLU/down/combine chain
    /// followed by a TP reduction. The reduction proves that the combine output is a replicated
    /// tensor boundary. Unsupported graphs fail during `Builder::finish` instead of emitting a
    /// packet the ordinary interpreter would misread.
    pub fn set_moe_prefill_ep_degree(&mut self, degree: Option<u32>) {
        self.moe_prefill_ep_degree = degree.filter(|&n| n > 1);
    }

    pub fn set_lean_kda_intra_segments(&mut self, enabled: bool) {
        self.lean_kda_intra_segments = enabled;
    }

    pub fn set_kda_intra_wave_items_segments(&mut self, enabled: bool) {
        self.kda_intra_wave_items_segments = enabled;
    }

    pub fn set_lean_kda_key_factor_segments(&mut self, enabled: bool) {
        self.lean_kda_key_factor_segments = enabled;
    }

    pub fn set_decode_mla_segments(&mut self, enabled: bool) {
        self.decode_mla_segments = enabled;
    }

    pub fn set_decode_grouped_moe_segments(&mut self, enabled: bool) {
        self.decode_grouped_moe_segments = enabled;
    }

    pub fn set_xreduce_wave_rs_segments(&mut self, enabled: bool) {
        self.xreduce_wave_rs_segments = enabled;
    }

    pub fn set_fuse_materialized_residual_inputs(&mut self, enabled: bool) {
        self.fuse_materialized_residual_inputs = enabled;
    }

    /// Slice the machine-filling `Gemv` / `GemvGlu` / `GemvQkv` packets into `s * n_cu` shares
    /// instead of `n_cu`. 1 (default) ⇒ byte-identical.
    ///
    /// DECODE ONLY, and the caller decides — exactly as [`Builder::deny_uniseg`] does, because
    /// only the caller knows which program it is emitting. A `Gemv` also appears in a PREFILL
    /// bucket (the lm_head row), where it is one packet on a program whose other 895 are GEMMs,
    /// and re-slicing it there is measured to cost 162 → 1222 ms of prefill at `s = 4`.
    /// See `PLOW_GEMV_SPLIT` in `devgen` for the knob and the measurement.
    pub fn set_gemv_split(&mut self, s: u32) {
        self.gemv_split = s.clamp(1, 16);
    }

    /// Re-declaring a name returns the EXISTING handle and grows it to the larger byte count,
    /// instead of appending a second entry under the same name.
    ///
    /// OFF by default, so every emitter in the tree stays byte-identical: with it off `tensor`
    /// appends unconditionally, and a name collision is the caller's bug.
    ///
    /// # Why this is a mode and not the behaviour
    ///
    /// It exists for the STACKED emit — one tensor table, several programs (prefill buckets plus
    /// decode), the shape [`Builder::adopt_tensors`] serves. GLM gets there by hoisting every
    /// declaration into a `declare_glm_rows` that takes `max_rows`, so the emitters never declare
    /// anything. Kimi-K3's emitters declare their own scratch inline, per layer, at names that are
    /// identical across buckets and sizes that are not — and the two phases do not even declare the
    /// same SET (a prefill bucket has the grouped-MoE row maps, decode has none). Adopting the
    /// previous program's table and re-emitting then needs exactly this: same name ⇒ same handle,
    /// bytes ⇒ the max over the programs that asked.
    ///
    /// Building DECODE FIRST under this mode is what keeps a decode program byte-identical to a
    /// decode-only emit: it declares into an empty table, so its handles are its own, and every
    /// later bucket can only APPEND. The alternative — emitting each program against its own table
    /// and remapping handles afterwards — cannot be done safely here, because several ops in this
    /// family carry demoted tensor handles in `i[]` slots (`GemvQkvg`'s `i6`, `KdaConv3`'s
    /// `i4`/`i5`/`i6`/`i7`, `KdaStateStepG`'s `i5`) that no generic remap can see.
    pub fn set_tensor_dedup(&mut self, on: bool) {
        self.tensor_dedup = on;
    }

    /// Declare a tensor and get its handle. See [`Builder::set_tensor_dedup`] for the
    /// re-declaration rule.
    pub fn tensor(&mut self, name: &str, bytes: u64) -> u32 {
        if self.tensor_dedup {
            if let Some(i) = self.tensors.iter().position(|t| t.name == name) {
                self.tensors[i].bytes = self.tensors[i].bytes.max(bytes);
                return i as u32;
            }
        }
        self.tensors.push(TensorDecl {
            name: name.to_string(),
            bytes,
            init: None,
        });
        (self.tensors.len() - 1) as u32
    }

    /// Declare a tensor whose contents the compiler already knows (e.g. RoPE tables).
    pub fn tensor_init(&mut self, name: &str, init: Vec<u8>) -> u32 {
        self.tensors.push(TensorDecl {
            name: name.to_string(),
            bytes: init.len() as u64,
            init: Some(init),
        });
        (self.tensors.len() - 1) as u32
    }

    /// Declare a tensor the RUNTIME materialises from `recipe` at bind time —
    /// the same bytes [`tensor_init`](Self::tensor_init) would have expanded,
    /// without carrying them in the blob. `bytes` must equal what the recipe
    /// produces; [`Model::to_blob_v6`] asserts it.
    pub fn tensor_gen(&mut self, name: &str, bytes: u64, mut recipe: GenTensor) -> u32 {
        let h = self.tensor(name, bytes);
        recipe.tensor = h;
        // Under `set_tensor_dedup` a re-declared name gives back the SAME handle, so a second
        // recipe for it would be a duplicate the blob writer has to materialise twice. Keep the
        // first — the two are equal by construction (same name, same recipe) or the caller has a
        // bug the assert below names.
        if let Some(old) = self.gen.iter().find(|g| g.tensor == h) {
            assert_eq!(
                old.byte_len(),
                recipe.byte_len(),
                "tensor_gen: `{name}` re-declared with a recipe of a different length"
            );
            return h;
        }
        self.gen.push(recipe);
        h
    }

    /// The generated-tensor recipes declared so far.
    pub fn gen_tensors(&self) -> Vec<GenTensor> {
        self.gen.clone()
    }

    pub fn n_cu(&self) -> u32 {
        self.n_cu
    }

    /// The whole machine.
    pub fn all(&self) -> Vec<u32> {
        (0..self.n_cu).collect()
    }

    /// Split the machine into `parts` disjoint CU sets, so independent ops overlap.
    pub fn split(&self, parts: u32, i: u32) -> Vec<u32> {
        let per = self.n_cu / parts;
        let lo = i * per;
        let hi = if i + 1 == parts { self.n_cu } else { lo + per };
        (lo..hi).collect()
    }

    /// Emit an op onto `cus`, gated behind `deps`. Returns the counter it bumps,
    /// which is what a consumer passes back in as a dep.
    pub fn emit(
        &mut self,
        op: DevOp,
        cus: Vec<u32>,
        deps: &[u32],
        f: impl FnOnce(&mut DevInst),
    ) -> u32 {
        let d: Vec<Dep> = deps.iter().map(|&c| Dep::Coarse(c)).collect();
        self.emit_dep(op, cus, d, f)
    }

    /// As [`Builder::emit`], but the dependencies may be [`Dep::Fine`] — so a slice waits
    /// only on the producer slices that actually feed it, instead of on the whole op.
    pub fn emit_dep(
        &mut self,
        op: DevOp,
        cus: Vec<u32>,
        deps: Vec<Dep>,
        f: impl FnOnce(&mut DevInst),
    ) -> u32 {
        assert!(!cus.is_empty(), "an op must run at least one CU");
        for d in &deps {
            if let Dep::Fine { map, .. } = d {
                assert_eq!(
                    map.len(),
                    cus.len(),
                    "a Fine dep needs one producer-slice list per consumer slice"
                );
            }
        }
        let mut inst = DevInst {
            op: op as u16,
            blocks: cus.len() as u16,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [TENSOR_NONE; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        f(&mut inst);
        // AFTER the closure: the immediates do not exist until it has run.
        tune_dump_gemv(op, &inst);
        let counter = self.ops.len() as u32;
        // Uniform by default: an op that does not tell the builder its per-slice costs is
        // assumed balanced, which makes `select_granularity` fall back to coarse counters.
        let work = vec![1u32; cus.len()];
        self.ops.push(Op {
            inst,
            cus,
            deps,
            counter,
            work,
        });
        counter
    }

    /// As [`Builder::emit_dep`], but supplying the per-slice cost the cost model predicts.
    ///
    /// This is what lets [`Builder::select_granularity`] decide coarse-vs-fine on its own.
    /// Without it an op is assumed UNIFORM (every slice the same cost), which is the safe
    /// default: it makes the selector fall back to coarse counters.
    pub fn emit_dep_work(
        &mut self,
        op: DevOp,
        cus: Vec<u32>,
        deps: Vec<Dep>,
        work: Vec<u32>,
        f: impl FnOnce(&mut DevInst),
    ) -> u32 {
        assert_eq!(work.len(), cus.len(), "one work estimate per slice");
        let c = self.emit_dep(op, cus, deps, f);
        self.ops[c as usize].work = work;
        c
    }

    /// **Decide coarse vs fine, from the dataflow. This is the whole point of a compiler.**
    ///
    /// `plowc` declares [`Dep::Fine`] wherever the dataflow is *sparse* — a head, a column
    /// block, a KV split — and [`Dep::Coarse`] wherever it is a reduction. So the coarse edges
    /// ARE the barriers, and the fine edges partition the graph into barrier-to-barrier regions.
    ///
    /// The question is whether a region's fine gates buy anything, and that is settled — proved,
    /// in `lean-plow/Plow/CounterGranularity.lean`:
    ///
    /// > `collapse` : if every stage's producer map covers the previous stage, and the work is
    /// > UNIFORM across each stage's slices, then the fine schedule's makespan is *identical* to
    /// > the coarse one — for any producer maps whatsoever. The maps do not matter.
    ///
    /// The reason is that the barrier closing the region takes a `max` over every consumer
    /// slice, and `max_v (max_{u ∈ P v} finish u) = max_{u ∈ ⋃ P v} finish u = max_{all u}`.
    /// The union of the per-slice producer sets covers everything, so the global maximum is
    /// re-imposed however finely the gates upstream were cut.
    ///
    /// The contrapositive (`hetero_can_win`) is the opportunity: fine gates pay **only** when a
    /// straggling producer feeds a *cheap* consumer, so its slack is absorbed instead of
    /// reaching the barrier. That needs the consumers to do DIFFERENT amounts of work.
    ///
    /// So: keep the fine gates in a region iff some op in it has non-uniform per-slice work.
    ///
    /// * Transformer attention — every head is identical ⇒ uniform ⇒ **downgraded to coarse**.
    ///   Measured, and it is the right answer: fine gates cost 16.9 → 17.2 ms/token and returned
    ///   nothing, exactly as `collapse` says they must.
    /// * MoE — experts get different token counts by construction ⇒ heterogeneous ⇒ **kept**.
    fn select_granularity(&mut self) -> (usize, usize) {
        let n = self.ops.len();
        let mut uf: Vec<usize> = (0..n).collect();
        fn find(uf: &mut [usize], mut x: usize) -> usize {
            while uf[x] != x {
                uf[x] = uf[uf[x]];
                x = uf[x];
            }
            x
        }

        // A fine edge joins producer and consumer into one barrier-to-barrier region.
        // A coarse edge does not — it IS the barrier.
        for i in 0..n {
            let producers: Vec<u32> = self.ops[i]
                .deps
                .iter()
                .filter_map(|d| match d {
                    Dep::Fine { producer, .. } => Some(*producer),
                    Dep::Coarse(_) => None,
                })
                .collect();
            for p in producers {
                let (a, b) = (find(&mut uf, i), find(&mut uf, p as usize));
                if a != b {
                    uf[a] = b;
                }
            }
        }

        // Is any op in the region heterogeneous?
        let mut hetero = vec![false; n];
        for i in 0..n {
            let w = &self.ops[i].work;
            let h = w.first().is_some_and(|w0| w.iter().any(|x| x != w0));
            let r = find(&mut uf, i);
            hetero[r] |= h;
        }

        // Downgrade the fine gates in every homogeneous region — they are provably free of
        // benefit there, and they are NOT free of cost (an extra counter per producer slice,
        // an extra atomic per producer, and a wider wait list per consumer).
        //
        // SE_FINE straggler-recovery lever (PLOW_FINE_FORCE=1). The `collapse` theorem downgrades
        // every homogeneous region because per-slice work is MODELLED uniform, so on real hardware
        // the diffuse-straggler wait (dev.rs "Per-slice gates") is never recovered — the model is
        // right about the MODEL. This lever overrides that to MEASURE the real-hardware delta. It
        // keeps a fine edge iff it is genuinely SPARSE (some consumer slice waits on strictly fewer
        // than all producer slices); an all-to-all "fine" edge (e.g. a GEMV whose column map
        // collapses to full fan-in under wave-interleaving) is still downgraded, so the test
        // isolates the recoverable gates (headnorm->flash, flash->merge) and never pays the
        // 256x256-atomic all-to-all cost the Dep doc warns about. Default (unset) = byte-identical.
        let force = std::env::var("PLOW_FINE_FORCE").ok().as_deref() == Some("1");
        let blocks: Vec<usize> = self.ops.iter().map(|o| o.cus.len()).collect();
        let (mut kept, mut downgraded) = (0, 0);
        for i in 0..n {
            let r = find(&mut uf, i);
            let hetero_keep = hetero[r];
            for d in self.ops[i].deps.iter_mut() {
                if let Dep::Fine { producer, map } = d {
                    let sparse = map.iter().any(|s| s.len() < blocks[*producer as usize]);
                    let keep = hetero_keep || (force && sparse);
                    if keep {
                        kept += 1;
                    } else {
                        *d = Dep::Coarse(*producer);
                        downgraded += 1;
                    }
                }
            }
        }
        (kept, downgraded)
    }

    /// Slice-level locality census (`PLOW_PLACE_REPORT=1`). Diagnostic; changes nothing.
    ///
    /// **The question a locality-aware placement pass has to answer before it is written.**
    /// A same-domain placement can only pay where a consumer slice's producer set is SPARSE. If
    /// a consumer slice reads EVERY producer slice — which is what a surviving `Dep::Coarse`
    /// means, and what a reduction like a GEMV reading the whole activation vector *is* — then
    /// its reads are spread across every domain by construction, and no assignment beats the
    /// uniform `1/domains`. Concentrating the producer instead just moves the imbalance.
    ///
    /// Run after `select_granularity`, so the fine/coarse decisions are the final ones.
    ///
    /// Reported:
    /// * the slice-pair census split by all-to-all vs sparse edges — **placement-independent**,
    ///   so it bounds every possible pass, not just the one simulated here;
    /// * same-domain pairs under the CURRENT mapping (`L2Layout::domain_of(cus[slice])`);
    /// * same-domain pairs under a greedy predecessor-affinity pass — `passes.rs`'s
    ///   `pred_locality_hint` rule (majority domain over already-placed predecessors), under the
    ///   balance cap every dispatcher is subject to (`ceil(slices/domains)` per domain, because
    ///   each XCD runs its own fixed share of the grid and an unbalanced pass idles hardware);
    /// * how many slices and ops that pass MOVES.
    fn locality_census(&self, l2: Option<L2Layout>) {
        let c = self.locality_census_stats(l2);
        let pairs = c.pairs;
        let pct = |x: u64| {
            if pairs == 0 {
                0.0
            } else {
                100.0 * x as f64 / pairs as f64
            }
        };
        eprintln!(
            "  locality census ({} ops, {} slices, {} domains, map {}):",
            c.ops, c.slices, c.domains, c.map_name
        );
        eprintln!(
            "    slice-level producer->consumer pairs: {pairs}  \
             ({:.1}% on ALL-TO-ALL edges, where 1/{} = {:.1}% is the ceiling for ANY placement)",
            pct(c.all_to_all_pairs),
            c.domains,
            100.0 / c.domains as f64
        );
        eprintln!(
            "    same-domain pairs: current {:.2}%  |  greedy pred-affinity {:.2}%  |  \
             per-slice argmax ceiling {:.2}% (producers pinned as-is, balance ignored)",
            pct(c.same_current),
            pct(c.same_greedy),
            pct(c.same_ceiling)
        );
        eprintln!(
            "    greedy moves {}/{} slices in {}/{} ops",
            c.moved_slices, c.slices, c.moved_ops, c.ops
        );
    }

    /// The numbers behind [`Builder::locality_census`]. Split out so the invariant the census
    /// exists to expose is a unit test rather than a line of stderr.
    fn locality_census_stats(&self, l2: Option<L2Layout>) -> LocalityCensus {
        let layout = l2.unwrap_or(L2Layout {
            sms: 32,
            domains: 8,
            map: L2Map::RoundRobin,
        });
        let dc = layout.domains.max(1) as usize;
        let n = self.ops.len();

        // Current per-slice domain — what `finish` writes into `StreamEnt::seg`.
        let cur: Vec<Vec<usize>> = self
            .ops
            .iter()
            .map(|o| {
                o.cus
                    .iter()
                    .map(|&c| layout.domain_of(c) as usize)
                    .collect()
            })
            .collect();

        // Greedy predecessor-affinity, ops in emission order (which is topological).
        let mut greedy: Vec<Vec<usize>> = Vec::with_capacity(n);
        for op in &self.ops {
            let ns = op.cus.len();
            let cap = ns.div_ceil(dc);
            let mut load = vec![0usize; dc];
            let mut asg = vec![0usize; ns];
            let mut pref: Vec<Vec<u32>> = Vec::with_capacity(ns);
            for s in 0..ns {
                let mut counts = vec![0u32; dc];
                for dep in &op.deps {
                    let p = dep.producer() as usize;
                    match dep {
                        Dep::Coarse(_) => {
                            for &d in &greedy[p] {
                                counts[d] += 1;
                            }
                        }
                        Dep::Fine { map, .. } => {
                            for &ps in &map[s] {
                                counts[greedy[p][ps as usize]] += 1;
                            }
                        }
                    }
                }
                pref.push(counts);
            }
            // Strongest preference first, so a slice that really wants one domain claims it
            // before the cap fills with slices that were indifferent anyway.
            let mut order: Vec<usize> = (0..ns).collect();
            order.sort_by_key(|&s| {
                let c = &pref[s];
                let mx = c.iter().copied().max().unwrap_or(0) as u64;
                let tot: u64 = c.iter().map(|&x| x as u64).sum();
                (
                    std::cmp::Reverse(mx * dc as u64 - tot.min(mx * dc as u64)),
                    s,
                )
            });
            for s in order {
                let c = &pref[s];
                let best = (0..dc)
                    .filter(|&d| load[d] < cap)
                    .max_by_key(|&d| (c[d], std::cmp::Reverse(d)))
                    .unwrap_or(0);
                asg[s] = best;
                load[best] += 1;
            }
            greedy.push(asg);
        }

        let (mut pairs, mut a2a) = (0u64, 0u64);
        let (mut same_cur, mut same_greedy, mut ceil_pairs) = (0u64, 0u64, 0u64);
        let (mut moved_slices, mut moved_ops, mut slices) = (0u64, 0u64, 0u64);
        for (i, op) in self.ops.iter().enumerate() {
            slices += op.cus.len() as u64;
            let mut op_moved = false;
            for s in 0..op.cus.len() {
                if greedy[i][s] != cur[i][s] {
                    moved_slices += 1;
                    op_moved = true;
                }
                // Best a consumer slice could do with its producers held where they are.
                let mut by_dom = vec![0u64; dc];
                for dep in &op.deps {
                    let p = dep.producer() as usize;
                    let pn = self.ops[p].cus.len();
                    let sparse = match dep {
                        Dep::Coarse(_) => false,
                        Dep::Fine { map, .. } => map[s].len() < pn,
                    };
                    let plist: Vec<u32> = match dep {
                        Dep::Coarse(_) => (0..pn as u32).collect(),
                        Dep::Fine { map, .. } => map[s].clone(),
                    };
                    pairs += plist.len() as u64;
                    if !sparse {
                        a2a += plist.len() as u64;
                    }
                    for &ps in &plist {
                        by_dom[cur[p][ps as usize]] += 1;
                        if cur[p][ps as usize] == cur[i][s] {
                            same_cur += 1;
                        }
                        if greedy[p][ps as usize] == greedy[i][s] {
                            same_greedy += 1;
                        }
                    }
                }
                ceil_pairs += by_dom.iter().copied().max().unwrap_or(0);
            }
            if op_moved {
                moved_ops += 1;
            }
        }

        LocalityCensus {
            ops: n,
            slices,
            domains: dc,
            map_name: match layout.map {
                L2Map::Block => "block",
                L2Map::RoundRobin => "round-robin",
            },
            pairs,
            all_to_all_pairs: a2a,
            same_current: same_cur,
            same_greedy,
            same_ceiling: ceil_pairs,
            moved_slices,
            moved_ops,
        }
    }

    /// Flatten into the tables the interpreter walks.
    ///
    /// # Counter layout
    ///
    /// `[0, n_ops)` are the **coarse** counters, one per op: every slice bumps its op's
    /// counter, so it reaches `blocks` exactly when the op is done, and a coarse consumer
    /// waits for that threshold.
    ///
    /// An op that some consumer depends on *finely* additionally gets one counter **per
    /// slice**, based at `fine_base[op]`, each with threshold 1. A fine consumer waits on
    /// just the slices that feed it. A fine producer bumps both — its coarse counter (it may
    /// still have coarse consumers) and its own slice counter — which is two atomics instead
    /// of one, against a gate that was costing milliseconds.
    ///
    /// # Ordering, and why it is load-bearing
    ///
    /// Ops are appended in topological order, so for any dependency A → B every slice of A
    /// precedes every slice of B in every CU's stream. A fine list can only lower a threshold
    /// or narrow a wait set, never make a workgroup wait on something issued later in its own
    /// in-order stream — which is exactly the deadlock that
    /// the design notes document. **Do not reorder streams** without
    /// reading that file.
    /// Run whole-program fusion after every block has been emitted. Candidate discovery walks the
    /// complete dependency graph, never a model tag or layer index. The first rule retains packet
    /// adjacency as a scheduling-safety condition: the packet builder's coarse edges also encode
    /// ordering for tensor inputs, so moving a consumer across an intervening packet is not legal
    /// until those tensor edges are explicit. The fused consumer still materializes the residual
    /// output for every other graph consumer.
    fn fuse_materialized_residual_inputs(&mut self) -> usize {
        let graph = ProgramGraph::from_ops(&self.ops);
        let mut fuse_with = vec![None; self.ops.len()];
        for consumer_idx in 0..self.ops.len() {
            let consumer = &self.ops[consumer_idx];
            let [producer_idx] = graph.predecessors[consumer_idx].as_slice() else {
                continue;
            };
            let producer = &self.ops[*producer_idx];
            let n = consumer.inst.i[0].checked_mul(consumer.inst.i[1]);
            let compatible = producer.inst.op == DevOp::Residual as u16
                && consumer.inst.op == DevOp::AttnRes as u16
                && consumer_idx == *producer_idx + 1
                && matches!(consumer.deps.as_slice(), [Dep::Coarse(c)] if *c == *producer_idx as u32)
                && producer.deps.iter().all(|d| matches!(d, Dep::Coarse(_)))
                && producer.inst.t[0] == consumer.inst.t[1]
                && producer.inst.t[1] != TENSOR_NONE
                && producer.inst.t[2] != TENSOR_NONE
                && n == Some(producer.inst.i[0])
                && producer.inst.f[0].to_bits() == 1.0f32.to_bits()
                && consumer.inst.t[6] == TENSOR_NONE
                && consumer.inst.t[7] == TENSOR_NONE
                && consumer.inst.i[5] == TENSOR_NONE_I;
            if compatible {
                debug_assert!(graph.successors[*producer_idx].contains(&consumer_idx));
                fuse_with[*producer_idx] = Some(consumer_idx);
            }
        }

        let old = std::mem::take(&mut self.ops);
        let mut old_to_new = vec![u32::MAX; old.len()];
        let mut fused = 0usize;
        let mut next = old.into_iter().enumerate().peekable();

        let remap = |dep: Dep, map: &[u32]| match dep {
            Dep::Coarse(c) => Dep::Coarse(map[c as usize]),
            Dep::Fine {
                producer,
                map: fine,
            } => Dep::Fine {
                producer: map[producer as usize],
                map: fine,
            },
        };

        while let Some((idx, mut op)) = next.next() {
            let can_fuse = fuse_with[idx].is_some_and(|consumer_idx| {
                next.peek()
                    .is_some_and(|(next_idx, _)| *next_idx == consumer_idx)
            });

            if can_fuse {
                let (consumer_idx, mut consumer) = next.next().unwrap();
                let new_idx = self.ops.len() as u32;
                old_to_new[idx] = new_idx;
                old_to_new[consumer_idx] = new_idx;
                consumer.inst.t[6] = op.inst.t[1];
                consumer.inst.t[7] = op.inst.t[2];
                consumer.inst.i[5] = if op.inst.t[3] == TENSOR_NONE {
                    TENSOR_NONE_I
                } else {
                    op.inst.t[3]
                };
                consumer.deps = op.deps.drain(..).map(|d| remap(d, &old_to_new)).collect();
                consumer.counter = new_idx;
                self.ops.push(consumer);
                fused += 1;
                continue;
            }

            let new_idx = self.ops.len() as u32;
            old_to_new[idx] = new_idx;
            op.deps = op.deps.drain(..).map(|d| remap(d, &old_to_new)).collect();
            op.counter = new_idx;
            self.ops.push(op);
        }
        fused
    }

    /// Fold an eligible two-shot all-reduce into its AttnRes+RMSNorm consumer. The preceding
    /// residual pass leaves the rounded prefix materialization on `AttnRes`; this pass keeps that
    /// tensor live while moving the consumer into phase 2 of the collective.
    fn fuse_xreduce_attnres(&mut self) -> usize {
        let graph = ProgramGraph::from_ops(&self.ops);
        let mut fuse_with = vec![None; self.ops.len()];
        for consumer_idx in 0..self.ops.len() {
            let consumer = &self.ops[consumer_idx];
            let [producer_idx] = graph.predecessors[consumer_idx].as_slice() else {
                continue;
            };
            let producer = &self.ops[*producer_idx];
            let residual = if consumer.inst.t[6] == producer.inst.t[0]
                && consumer.inst.t[7] != TENSOR_NONE
            {
                Some(consumer.inst.t[7])
            } else if consumer.inst.t[7] == producer.inst.t[0] && consumer.inst.t[6] != TENSOR_NONE
            {
                Some(consumer.inst.t[6])
            } else if consumer.inst.t[6] == TENSOR_NONE
                && consumer.inst.t[7] == TENSOR_NONE
                && consumer.inst.t[1] == producer.inst.t[0]
            {
                Some(TENSOR_NONE)
            } else {
                None
            };
            let n = consumer.inst.i[0].checked_mul(consumer.inst.i[1]);
            let compatible = producer.inst.op == DevOp::XReduceTwoShot as u16
                && consumer.inst.op == DevOp::AttnRes as u16
                && consumer_idx == *producer_idx + 1
                && matches!(consumer.deps.as_slice(), [Dep::Coarse(c)] if *c == *producer_idx as u32)
                && matches!(graph.successors[*producer_idx].as_slice(), [c] if *c == consumer_idx)
                && producer.inst.t[1..].iter().all(|&t| t == TENSOR_NONE)
                && producer.inst.i[5..].iter().all(|&i| i == 0)
                && producer.inst.i[1] > 1
                && consumer.inst.i[0] % producer.inst.i[1] == 0
                && n == Some(producer.inst.i[0])
                && producer.cus == consumer.cus
                && consumer.inst.t[1] != TENSOR_NONE
                && consumer.inst.t[2] != TENSOR_NONE
                && consumer.inst.t[3] != TENSOR_NONE
                && consumer.inst.t[4] == TENSOR_NONE
                && consumer.inst.t[5] != TENSOR_NONE
                && consumer.inst.i[5] == TENSOR_NONE_I
                && residual.is_some();
            if compatible {
                fuse_with[*producer_idx] = Some((consumer_idx, residual.unwrap()));
            }
        }

        let old = std::mem::take(&mut self.ops);
        let mut old_to_new = vec![u32::MAX; old.len()];
        let mut fused = 0usize;
        let mut next = old.into_iter().enumerate().peekable();
        let remap = |dep: Dep, map: &[u32]| match dep {
            Dep::Coarse(c) => Dep::Coarse(map[c as usize]),
            Dep::Fine {
                producer,
                map: fine,
            } => Dep::Fine {
                producer: map[producer as usize],
                map: fine,
            },
        };

        while let Some((idx, mut op)) = next.next() {
            let candidate = fuse_with[idx].filter(|(consumer_idx, _)| {
                next.peek()
                    .is_some_and(|(next_idx, _)| *next_idx == *consumer_idx)
            });
            if let Some((consumer_idx, residual)) = candidate {
                let (_, consumer) = next.next().unwrap();
                let new_idx = self.ops.len() as u32;
                old_to_new[idx] = new_idx;
                old_to_new[consumer_idx] = new_idx;
                op.inst.t[1] = residual;
                op.inst.t[2] = consumer.inst.t[0];
                op.inst.t[3] = consumer.inst.t[2];
                op.inst.t[4] = consumer.inst.t[3];
                op.inst.t[5] = consumer.inst.t[5];
                op.inst.t[6] = consumer.inst.t[1];
                op.inst.i[5] = consumer.inst.i[1];
                op.inst.i[6] = consumer.inst.i[2];
                op.inst.i[7] = consumer.inst.i[4];
                op.inst.f[0] = consumer.inst.f[0];
                op.deps = op.deps.drain(..).map(|d| remap(d, &old_to_new)).collect();
                op.counter = new_idx;
                self.ops.push(op);
                fused += 1;
                continue;
            }

            let new_idx = self.ops.len() as u32;
            old_to_new[idx] = new_idx;
            op.deps = op.deps.drain(..).map(|d| remap(d, &old_to_new)).collect();
            op.counter = new_idx;
            self.ops.push(op);
        }
        fused
    }

    fn ep_companion_tensor(&mut self, handle: u32) -> u32 {
        let source = self
            .tensors
            .get(handle as usize)
            .expect("EP table handle is invalid")
            .clone();
        assert!(source.init.is_none(), "EP tables must be runtime-bound");
        let ep_name = format!("{}_ep", source.name);
        if let Some(i) = self.tensors.iter().position(|t| t.name == ep_name) {
            return i as u32;
        }
        let i = self.tensors.len();
        assert!(
            i < TENSOR_NONE as usize,
            "EP companion tensor table overflows u16"
        );
        self.tensors.push(TensorDecl {
            name: ep_name,
            bytes: source.bytes,
            init: None,
        });
        i as u32
    }

    fn rewrite_replicated_moe_prefill_ep(&mut self, degree: u32) -> usize {
        assert!(degree > 1, "EP degree must exceed one");
        let mut chains = Vec::new();
        for (glu, op) in self.ops.iter().enumerate() {
            let g = &op.inst;
            if g.op != DevOp::MoeGroupGluPf as u16 || g.i[3] != 2 || g.i[6] != 0 {
                continue;
            }
            let Some(down) = self.ops.iter().position(|candidate| {
                let d = &candidate.inst;
                d.op == DevOp::MoeGroupDownPf as u16
                    && d.t[1] == g.t[0]
                    && d.t[2] == g.t[2]
                    && d.t[3] == g.t[3]
                    && d.t[4] == g.t[4]
                    && d.i[0] == g.i[1]
                    && d.i[1] == g.i[0]
                    && d.i[2] == g.i[2]
                    && d.i[3] == g.i[3]
                    && d.i[4..].iter().all(|&v| v == 0)
            }) else {
                continue;
            };
            let d = &self.ops[down].inst;
            let Some(combine) = self.ops.iter().position(|candidate| {
                let c = &candidate.inst;
                c.op == DevOp::MoeCombinePf as u16
                    && c.t[3] == d.t[0]
                    && c.i[0] == d.i[0]
                    && c.i[1] == 16
                    && c.i[2] != 0
                    && c.i[3..].iter().all(|&v| v == 0)
            }) else {
                continue;
            };
            let reduced = self.ops.iter().any(|candidate| {
                matches!(
                    DevOp::from_u16(candidate.inst.op),
                    Some(DevOp::XReduce | DevOp::XReduceTwoShot)
                ) && candidate
                    .deps
                    .iter()
                    .any(|dep| dep.producer() as usize == combine)
            });
            if !reduced {
                continue;
            }
            let align: Vec<usize> = self
                .ops
                .iter()
                .enumerate()
                .filter_map(|(i, candidate)| {
                    let a = &candidate.inst;
                    (a.op == DevOp::MoeAlignPf as u16
                        && a.t[0] == g.t[4]
                        && a.i[0] == self.ops[combine].inst.i[2]
                        && a.i[1] == g.i[2]
                        && a.i[2] == 16)
                        .then_some(i)
                })
                .collect();
            assert!(
                !align.is_empty(),
                "EP boundary has no align producer for its declared metadata"
            );
            let full_i = g.i[0]
                .checked_mul(degree)
                .expect("EP full intermediate width overflows u32");
            assert!(
                g.i[0] > 0
                    && full_i.is_multiple_of(128)
                    && g.i[1].is_multiple_of(128)
                    && g.i[2] >= degree
                    && self.ops[combine].inst.i[2] > 1,
                "EP boundary geometry is unsupported"
            );
            chains.push((glu, down, combine, align, full_i));
        }

        for (glu, down, combine, align, full_i) in &chains {
            let weight = self.ops[*glu].inst.t[2];
            let scale = self.ops[*glu].inst.t[3];
            let ep_weight = self.ep_companion_tensor(weight);
            let ep_scale = self.ep_companion_tensor(scale);
            for handle in [weight, scale] {
                let source_name = self.tensors[handle as usize].name.clone();
                if let Some(companion) = self
                    .tensors
                    .iter()
                    .position(|t| t.name == format!("{source_name}_moe2"))
                {
                    self.ep_companion_tensor(companion as u32);
                }
            }
            self.ops[*glu].inst.t[2] = ep_weight;
            self.ops[*glu].inst.t[3] = ep_scale;
            self.ops[*down].inst.t[2] = ep_weight;
            self.ops[*down].inst.t[3] = ep_scale;
            let row_token_bytes = self
                .tensors
                .get(self.ops[*glu].inst.t[5] as usize)
                .expect("EP row-token tensor handle is invalid")
                .bytes;
            assert!(
                row_token_bytes.is_multiple_of(4),
                "EP row-token tensor is misaligned"
            );
            let rows = row_token_bytes / 4;
            let payload_bytes = rows
                .checked_mul(u64::from(*full_i) / 2)
                .expect("EP payload tensor size overflows u64");
            let scale_bytes = rows
                .checked_mul(u64::from(*full_i) / 32)
                .expect("EP scale tensor size overflows u64");
            for (handle, required) in [
                (self.ops[*glu].inst.t[0], payload_bytes),
                (self.ops[*glu].inst.t[7], scale_bytes),
            ] {
                let tensor = self
                    .tensors
                    .get_mut(handle as usize)
                    .expect("EP boundary tensor handle is invalid");
                tensor.bytes = tensor.bytes.max(required);
            }
            let experts = self.ops[*glu].inst.i[2];
            let meta = self.ops[align[0]].inst.t[0] as usize;
            let meta_words = u64::from(experts)
                .checked_mul(67)
                .and_then(|n| n.checked_add(1))
                .expect("EP metadata size overflows u64");
            let meta_bytes = meta_words * 4;
            self.tensors
                .get_mut(meta)
                .expect("EP align metadata handle is invalid")
                .bytes = meta_bytes;
            for &i in align {
                self.ops[i].inst.i[5] = degree;
            }
            self.ops[*glu].inst.i[0] = *full_i;
            self.ops[*glu].inst.i[6] = degree;
            self.ops[*down].inst.i[1] = *full_i;
            self.ops[*down].inst.i[6] = degree;
            self.ops[*combine].inst.t[4] = self.ops[align[0]].inst.t[1];
            self.ops[*combine].inst.i[5] = degree;
            self.ops[*combine].inst.i[6] = experts;
        }
        chains.len()
    }

    pub fn finish(mut self) -> Program {
        if let Some(degree) = self.moe_prefill_ep_degree {
            let rewritten = self.rewrite_replicated_moe_prefill_ep(degree);
            assert!(
                rewritten != 0,
                "replicated MoE EP requested at degree {degree}, but the complete graph has no eligible MXFP4 align/GLU/down/combine -> TP-reduction boundary"
            );
            eprintln!("  whole-graph placement: {rewritten} routed-MoE boundaries use EP{degree}");
        }
        if self.fuse_materialized_residual_inputs {
            let fused = self.fuse_materialized_residual_inputs();
            if fused != 0 {
                eprintln!("  whole-graph fusion: {fused} materialized residual inputs");
            }
        }
        if std::env::var("PLOW_FUSE_XR_ATTNRES").ok().as_deref() == Some("1") {
            let fused = self.fuse_xreduce_attnres();
            if fused != 0 {
                eprintln!("  whole-graph fusion: {fused} XReduceTwoShot+AttnRes consumers");
            }
        }
        let n_cu = self.n_cu as usize;
        let n_ops = self.ops.len();

        // Effective L2-domain placement (PLOW_L2_PLACE) — geometry half. The wave-class half
        // needs `cur_seg` and is applied below, once segmentation is known.
        //
        // The COVERAGE GUARD is specific to `L2Map::Block`: there a slice's domain is `cu / sms`,
        // which must be < P so it matches the runtime's `smid / sms` in [0, P). `n_cu > P·sms`
        // means occupancy>1 (n_cu = 2·sm_count) or a grid≠sm_count mismatch — placement would
        // emit domain windows the runtime never pulls (orphaned packets -> deadlock). Skip it and
        // fall back byte-identical.
        //
        // `L2Map::RoundRobin` needs no such guard: `cu % P` is in [0, P) by construction, and the
        // probe MEASURED round-robin still holding at occupancy 2 (512 blocks, 100.0%). Applying
        // the block guard there would have silently disabled placement on exactly the occ-2
        // configs it is safe for.
        let l2_place: Option<L2Layout> = self.place_l2.and_then(|l| {
            if l.sms == 0 || l.domains == 0 {
                None
            } else if l.map == L2Map::Block && self.n_cu > l.domains * l.sms {
                eprintln!(
                    "  l2 placement SKIPPED: n_cu {} > {} domains × {} SM = {} \
                     (occupancy>1 or grid≠sm_count, block map) — byte-identical",
                    self.n_cu,
                    l.domains,
                    l.sms,
                    l.domains * l.sms
                );
                None
            } else {
                Some(l)
            }
        });

        // Coarse or fine, decided from the dataflow — not from a flag. See the doc comment on
        // `select_granularity`, and the `collapse` theorem it implements.
        let (kept, downgraded) = self.select_granularity();
        if kept + downgraded > 0 {
            eprintln!(
                "  counter granularity: {kept} fine edges kept, {downgraded} downgraded to \
                 coarse (homogeneous region — see Plow/CounterGranularity.lean:collapse)"
            );
        }

        // CHAIN-BYPASS — a MEASUREMENT INSTRUMENT, numerically WRONG, never shipped.
        //
        // knob-contract §7a-REFINED says a serial packet on the decode chain costs ~5.3 us in the
        // ADDING direction; §7a-CHAIN measured the REMOVING direction with this knob at ~1.4 us.
        // §6b-i requires pricing a scheduling change with a cheap build that touches only the
        // schedule before building the real kernel.
        //
        // `PLOW_CHAIN_BYPASS=<op>[,<op>...]` (opcode numbers) splices the named ops OUT of the
        // dependency chain: every consumer that waits on op O instead waits on O's own
        // predecessors. The op is STILL EMITTED and still runs on the same workgroups, so
        // packet count, workgroup-packet count, and total memory traffic are IDENTICAL — the
        // ONLY delta is critical-path depth. That isolates chain length exactly the way §7a's
        // PLOW_NO_FUSE_QKV isolated gate count.
        //
        // It is also the STRICT UPPER BOUND on any partial-completion / tile-granular signalling
        // scheme on the same edge: bypass is the limit of "the consumer never waits at all".
        //
        // Consumers read the op's stale output, so tokens are garbage. That is intended: this
        // measures scheduling, and wrong numerics are a valid instrument for scheduling.
        if let Ok(spec) = std::env::var("PLOW_CHAIN_BYPASS") {
            let want: Vec<u16> = spec
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            // Predecessor sets first, so a run of bypassed ops collapses transitively. Only
            // all-coarse ops are liftable; the DECODE program is entirely coarse (graphstat:
            // SE_FINE == 0), the prefill programs carry Fine edges and are left untouched —
            // which is what we want, this prices decode only.
            let mut lifted: Vec<Option<Vec<u32>>> = vec![None; n_ops];
            for i in 0..n_ops {
                if !want.contains(&self.ops[i].inst.op) {
                    continue;
                }
                if self.ops[i]
                    .deps
                    .iter()
                    .any(|d| !matches!(d, Dep::Coarse(_)))
                {
                    continue;
                }
                let mut up: Vec<u32> = Vec::new();
                for d in &self.ops[i].deps {
                    let Dep::Coarse(c) = d else { unreachable!() };
                    match &lifted[*c as usize] {
                        Some(t) => up.extend(t.iter().copied()),
                        None => up.push(*c),
                    }
                }
                up.sort_unstable();
                up.dedup();
                lifted[i] = Some(up);
            }
            let mut spliced = 0usize;
            for i in 0..n_ops {
                if lifted[i].is_some() {
                    continue; // the bypassed op keeps its own deps: it must still RUN correctly
                }
                let old = std::mem::take(&mut self.ops[i].deps);
                let mut ds: Vec<Dep> = Vec::new();
                let mut seen: Vec<u32> = Vec::new();
                for d in old {
                    // A Fine consumer edge is left intact: rewriting it to coarse would change
                    // gate granularity, not just chain depth.
                    let liftable = match d {
                        Dep::Coarse(c) => lifted[c as usize].as_ref(),
                        Dep::Fine { .. } => None,
                    };
                    match liftable {
                        Some(up) => {
                            spliced += 1;
                            for &c in up {
                                if !seen.contains(&c) {
                                    seen.push(c);
                                    ds.push(Dep::Coarse(c));
                                }
                            }
                        }
                        None => {
                            if !seen.contains(&d.producer()) {
                                seen.push(d.producer());
                                ds.push(d);
                            }
                        }
                    }
                }
                self.ops[i].deps = ds;
            }
            eprintln!(
                "  CHAIN-BYPASS ops {want:?}: {spliced} consumer edges spliced past \
                 {} bypassed ops — NUMERICALLY WRONG, measurement only",
                lifted.iter().filter(|l| l.is_some()).count()
            );
        }

        // TRANSITIVE REDUCTION of the coarse counter DAG.
        //
        // A coarse dep A→C is redundant when a path A→…→C already exists through other coarse
        // deps: the gate it installs orders nothing the path does not already order. Dropping it
        // removes one wait entry from every one of C's slices, i.e. `blocks(C)` runtime polls per
        // token. Measured on the 93-layer K3 decode blob: 3038 edges → 2969, and 454,942 polls →
        // 401,950 (52,992 removed, 11.6%).
        //
        // SOUNDNESS, and why BOTH the dropped edge and every path edge must be Coarse:
        //   * `Dep::Coarse` means "every workgroup of the producer has bumped", so a chain of
        //     coarse edges A→B→C orders all-of-A before all-of-B before all-of-C — exactly the
        //     constraint the dropped A→C asserted.
        //   * `Dep::Fine` orders only the slices in its map, so it can neither be dropped this way
        //     nor justify a path. Fine deps are skipped in both roles: they stay in `deps`, and
        //     they are never inserted into `edges`. Ignoring them only COSTS reductions (an
        //     ordering we decline to exploit); it can never license an unsafe one.
        //
        // This does not weaken `happensBefore`: the dropped edge is still ordered, transitively.
        // It does mean a data edge need no longer be DIRECTLY counter-gated, which is why the
        // Lean side needed a coverage statement in terms of `happensBefore` rather than
        // `WellFormed.edgeCovered` — see `lean-plow/Plow/TransitiveReduction.lean`,
        // `tr_preserves_coverage`.
        {
            let mut edges: BTreeSet<(u32, u32)> = BTreeSet::new();
            for (i, op) in self.ops.iter().enumerate() {
                for d in &op.deps {
                    if let Dep::Coarse(c) = d {
                        edges.insert((*c, i as u32));
                    }
                }
            }
            let keep = transitive_reduction(n_ops, &edges);
            let dropped = edges.len() - keep.len();
            // Two independent removals, and they were once conflated in the disassembler's
            // `polls_removable`: the transitive reduction drops DISTINCT edges implied by a path
            // (measured 69 edges, 17,664 polls on the K3 decode blob), while `seen` drops REPEATED
            // waits on the SAME counter within one op's list (a further 35,328 polls, 7.8%). A
            // duplicate is redundant by definition — the second wait on a counter is satisfied at
            // exactly the moment the first is — and the edge set alone cannot see them, because it
            // is a set.
            let mut dup = 0usize;
            for (i, op) in self.ops.iter_mut().enumerate() {
                let mut seen: Vec<u32> = Vec::new();
                let before = op.deps.len();
                op.deps.retain(|d| match d {
                    Dep::Coarse(c) => {
                        if !keep.contains(&(*c, i as u32)) || seen.contains(c) {
                            false
                        } else {
                            seen.push(*c);
                            true
                        }
                    }
                    Dep::Fine { .. } => true,
                });
                dup += before - op.deps.len();
            }
            self.tr_dropped = dropped;
            if (dropped > 0 || dup > dropped) && std::env::var_os("PLOW_TR_QUIET").is_none() {
                eprintln!(
                    "  counter-graph reduction: {} of {} distinct coarse edges implied by a path, \
                     {} duplicate waits; {} wait entries removed",
                    dropped,
                    edges.len(),
                    dup - dropped,
                    dup
                );
            }
        }

        // Which ops does someone depend on FINELY? Those get per-slice counters.
        let mut fine_base = vec![u32::MAX; n_ops];
        let mut n_counter = n_ops as u32;
        for op in &self.ops {
            for d in &op.deps {
                if let Dep::Fine { producer, .. } = d {
                    let p = *producer as usize;
                    assert!(p < n_ops, "dep refers to an op that was never emitted");
                    if fine_base[p] == u32::MAX {
                        fine_base[p] = n_counter;
                        n_counter += self.ops[p].cus.len() as u32;
                    }
                }
            }
        }

        let mut insts = Vec::with_capacity(n_ops);
        let mut waits: Vec<Wait> = Vec::new();
        let mut succs: Vec<u32> = Vec::new();
        let mut streams: Vec<Vec<StreamEnt>> = vec![Vec::new(); n_cu];
        // GLOBAL-QUEUE (Experiment E1): the same stream entries in OP-MAJOR (topological) order —
        // every slice of op A before every slice of op B. The per-CU concatenation below is
        // CU-major and NOT globally topological, so a flat cursor over it would deadlock; this
        // op-major order is deadlock-free under a monotonic fetch-add cursor. Carried alongside
        // the per-CU streams so one pkt holds both layouts; the interp picks by build flag.
        let mut gq_stream: Vec<StreamEnt> = Vec::new();

        // WAVE-CLASS SEGMENTATION. FlashPrefill wants a 4-wave (FA_DC=256, 512-reg) launch; every
        // other op wants 8 waves (2 waves/SIMD latency hiding). Occupancy is a launch-time property,
        // so the host relaunches once per maximal same-class run of ops in this (topological) emit
        // order, with that run's wave count. A segment is that run; `seg_of[i]` is op i's segment.
        // The host reads the class back from the ops themselves. See the design notes.
        // PLOW_UNISEG=1 collapses every op into ONE segment. The wave-class split exists so an AMD
        // host can relaunch FlashPrefill at a 4-wave occupancy; the sm_120 persistent interpreter
        // runs EVERY op at a fixed 256-thread (8-warp) block and synchronises the whole program in
        // one cooperative launch under the counter protocol (exactly as the decode program does), so
        // the segment boundary is spurious there and would otherwise force a segmented relaunch path.
        // `deny_uniseg` wins over the environment: a target that cannot express one segment must
        // not be given one because a variable said so. See that method for the failure it prevents.
        let uniseg = !self.uniseg_denied
            && (self.uniseg_forced || std::env::var("PLOW_UNISEG").ok().as_deref() == Some("1"));
        // AMD L2-placed packets split FlashMlaPrefill (bf16, op 51) and its fp8-KV twin
        // (op 110) into their own wave-class-4
        // segments at T>=2048 so the AMD host can route them to the 4-wave flash object's V2
        // kernel (d_flash_mla_prefill_v2). Smaller buckets remain one 8-wave L2-placed launch.
        // Emit-time, because segments only form on wave_class
        // BOUNDARIES: reclassifying host-side would drag whatever ops share the segment onto
        // an object that silently skips them. PLOW_MLA_PF_V2=0 is the explicit opt-out; non-AMD
        // packets remain byte-identical. The host applies
        // its own purity + size guards (exec/amd.rs derive_segments), so an env mismatch in
        // either direction degrades to the 8-wave kernel rather than corrupting.
        let mla_v2 = !uniseg
            && match std::env::var("PLOW_MLA_PF_V2").ok().as_deref() {
                Some("0") => false,
                Some("1") => true,
                // A placed packet is an AMD production artifact. Isolating a pure MLA flash
                // segment is safe even when its optional lean object is absent: the host then
                // runs that ordered segment on the ordinary 8-wave interpreter.
                _ => self.place_l2.is_some(),
            };
        // Opt-in only: live packed serving remains disabled. Giving descriptor-consuming
        // families distinct classes lets a future runtime route them to lean objects without
        // putting their branches in the production megakernel. Unset preserves packet bytes.
        let packed_prefill_segments = packed_prefill_segmenting_needed(
            uniseg,
            self.packed_prefill_segments,
            self.ops.iter().map(|o| o.inst.op),
        );
        let lean_moe_stage2 = !uniseg
            && self.lean_moe_stage2_segments
            && (0..self.ops.len()).any(|i| lean_moe_stage2_pair(&self.ops, i));
        let lean_moe_stage1 = !uniseg
            && self.lean_moe_stage1_segments
            && self.ops.iter().any(|op| lean_moe_stage1_inst(&op.inst));
        let lean_moe_combine = !uniseg
            && self.lean_moe_combine_segments
            && self.ops.iter().any(|op| lean_moe_combine_inst(&op.inst));
        let moe_prefill_ep = self.moe_prefill_ep_degree.is_some();
        let kda_intra_wave_items = !uniseg && self.kda_intra_wave_items_segments;
        let lean_kda_intra = !uniseg
            && (self.lean_kda_intra_segments || kda_intra_wave_items)
            && self.ops.iter().any(|op| {
                op.inst.op == DevOp::KdaChunkIntra as u16
                    && op.inst.i[0] >= 512
                    && op.inst.i[1] != 0
                    && op.inst.i[2] == 128
            });
        let lean_kda_key_factor = !uniseg
            && self.lean_kda_key_factor_segments
            && (0..self.ops.len()).any(|i| lean_kda_key_factor_pair(&self.ops, i));
        let decode_mla_segments = !uniseg
            && self.decode_mla_segments
            && self.ops.windows(2).any(|pair| {
                pair[0].inst.op == DevOp::FlashMlaDecode as u16
                    && pair[1].inst.op == DevOp::MlaMergeFold as u16
            });
        let decode_grouped_moe = !uniseg
            && (self.decode_grouped_moe_segments
                || std::env::var("PLOW_MOE_DECODE_STANDALONE").ok().as_deref() == Some("1"))
            && self.ops.windows(2).any(|pair| {
                pair[0].inst.op == DevOp::MoeGroupGluFp8Blk as u16
                    && pair[1].inst.op == DevOp::MoeGroupDownFp8Blk as u16
            });
        let graph_phase_objects =
            std::env::var("PLOW_PHASE_OBJECTS").ok().as_deref() == Some("1");
        let xreduce_wave_rs = !uniseg
            && self.place_l2.is_some()
            && (self.xreduce_wave_rs_segments
                || std::env::var("PLOW_XR_WAVE_RS").ok().as_deref() == Some("1"))
            && self
                .ops
                .iter()
                .any(|op| op.inst.op == DevOp::XReduceTwoShot as u16);
        let isolate_xreduce = xreduce_wave_rs
            || (!uniseg
                && self.place_l2.is_some()
                && graph_phase_objects
                && self
                    .ops
                    .iter()
                    .any(|op| op.inst.op == DevOp::XReduceTwoShot as u16));
        // This encoding is understood only by its dedicated interpreter object. As with the
        // raw KDA boundary, keep it isolated even when PLOW_UNISEG was requested; otherwise the
        // ordinary XReduce arm would silently interpret the fused operand slots as its legacy
        // residual/gather contract.
        let xr_attnres = self
            .ops
            .iter()
            .any(|op| op.inst.op == DevOp::XReduceTwoShot as u16 && op.inst.t[3] != TENSOR_NONE);
        let mla_materialized = self.ops.iter().any(|op| {
            matches!(
                DevOp::from_u16(op.inst.op),
                Some(DevOp::MlaMaterializePack | DevOp::FlashMlaMaterializedPrefill)
            )
        });
        // PLOW_SEG_PURE_GEMM=1 (T11): class-8 segments carry ONLY GEMM-family ops; every light
        // op (norms, rope, quant, embed, softcap) joins the flash class. The point: the sm_90a
        // segmented launcher runs class-8 segments on the lean `_pfgemm` object, and a pure-GEMM
        // segment lets that object be compiled with the non-GEMM arms stripped
        // (PLOW_NV_GEMM_ONLY), which is what gives ptxas a probe-shaped TU. More segment
        // boundaries is the cost; only a prefill program pays it (gated on the program actually
        // containing a flash op, so decode — which has no class-4 op and runs single-launch —
        // emits byte-identical packets with the knob set).
        // "1" = every plain tiled GEMM is class-8 (pairs with PLOW_NV_GEMM_ONLY);
        // "fp8" = ONLY TMA-mapped fp8 GEMMs are class-8 (pairs with the ws-entry object,
        // whose sole arm is the warp-specialized w8a8 body — a bf16 or mapless packet
        // landing there would __trap()).
        let pure_env = std::env::var("PLOW_SEG_PURE_GEMM").ok();
        let pure_mode = match pure_env.as_deref() {
            Some("1") => 1u8,
            Some("fp8") => 2u8,
            _ => 0u8,
        };
        let pure_gemm = !uniseg
            && pure_mode != 0
            && self.ops.iter().any(|o| {
                o.inst.op == DevOp::FlashPrefill as u16
                    || o.inst.op == DevOp::FlashPrefillFp8 as u16
            });
        // PLOW_SEG_FA512=1 (T12): hd512 (full-attention) FlashPrefill packets get their OWN
        // class (2) so the host can launch them on the dedicated *_pffa flash object. hd is
        // carried in inst.i[6]. Requires the serve-side mirror PLOW_PF_SEG_FA512=1.
        // "1" = hd512 only; "all" = every FlashPrefill (needs the PLOW_NV_FA_ONLY_HD256 object).
        let fa512_env = std::env::var("PLOW_SEG_FA512").ok();
        let fa512_mode = match fa512_env.as_deref() {
            Some("1") if !uniseg => 1u8,
            Some("all") if !uniseg => 2u8,
            _ => 0u8,
        };
        // PLOW_SEG_V2=1 (T16, needs fa512=all + pure=fp8): rope and flash-merge join the FA
        // class (the *_pffa object carries their arms under PLOW_NV_FA_ROPE), and QuantFp8
        // joins the GEMM class (the uni256 object carries the quant arm) — the per-layer
        // [rope, flash, merge] and [gate/up, gluquant, down] chains become ONE launch each.
        // "1" = full v2 (rope/merge->FA + quant->GEMM; refuted on the 256-thread objects);
        // "q8" (T36) = quant->GEMM only — the ws384 object carries a consumer-warpgroup
        // quant arm, so the [gate/up, glu-quant, down] chain becomes one class-8 run.
        let v2_env = std::env::var("PLOW_SEG_V2").ok();
        let seg_v2 = v2_env.as_deref() == Some("1");
        let seg_q8 = seg_v2 || v2_env.as_deref() == Some("q8");
        let wave_class = |i: usize| -> u8 {
            let op = self.ops[i].inst.op;
            if op == DevOp::KdaDecodeFused as u16 {
                // A standalone raw-argument object owns this boundary. Keep its segment pure
                // even if PLOW_UNISEG was requested; runtime routing may then select by opcode.
                3
            } else if decode_grouped_moe
                && ((op == DevOp::MoeGroupGluFp8Blk as u16
                    && self
                        .ops
                        .get(i + 1)
                        .is_some_and(|next| next.inst.op == DevOp::MoeGroupDownFp8Blk as u16))
                    || (op == DevOp::MoeGroupDownFp8Blk as u16
                        && i > 0
                        && self.ops[i - 1].inst.op == DevOp::MoeGroupGluFp8Blk as u16))
            {
                20
            } else if op == DevOp::MlaMaterializePack as u16 {
                14
            } else if op == DevOp::FlashMlaMaterializedPrefill as u16 {
                15
            } else if xr_attnres
                && op == DevOp::XReduceTwoShot as u16
                && self.ops[i].inst.t[3] != TENSOR_NONE
            {
                12
            } else if lean_kda_intra
                && op == DevOp::KdaChunkIntra as u16
                && self.ops[i].inst.i[0] >= 512
                && self.ops[i].inst.i[1] != 0
                && self.ops[i].inst.i[2] == 128
            {
                11
            } else if lean_kda_key_factor && lean_kda_key_factor_pair(&self.ops, i) {
                16
            } else if lean_kda_key_factor && i > 0 && lean_kda_key_factor_pair(&self.ops, i - 1) {
                17
            } else if moe_prefill_ep && op == DevOp::MoeAlignPf as u16 && self.ops[i].inst.i[5] > 1
            {
                21
            } else if moe_prefill_ep
                && op == DevOp::MoeGroupGluPf as u16
                && self.ops[i].inst.i[6] > 1
            {
                22
            } else if moe_prefill_ep
                && op == DevOp::MoeGroupDownPf as u16
                && self.ops[i].inst.i[6] > 1
            {
                23
            } else if moe_prefill_ep
                && op == DevOp::MoeCombinePf as u16
                && self.ops[i].inst.i[5] > 1
            {
                24
            } else if lean_moe_stage2 && lean_moe_stage2_pair(&self.ops, i) {
                // The standalone gfx950 kernel owns exactly the deterministic Down scatter.
                // Combine stays in the following interpreter segment and preserves fixed-order
                // f32 accumulation.
                9
            } else if lean_moe_stage1 && lean_moe_stage1_inst(&self.ops[i].inst) {
                // The BK256 standalone object owns exactly one grouped gate/up packet.
                10
            } else if lean_moe_combine && lean_moe_combine_inst(&self.ops[i].inst) {
                // The standalone object preserves the interpreter's fixed slot order.
                13
            } else if decode_mla_segments
                && ((op == DevOp::FlashMlaDecode as u16
                    && self
                        .ops
                        .get(i + 1)
                        .is_some_and(|next| next.inst.op == DevOp::MlaMergeFold as u16))
                    || (op == DevOp::MlaMergeFold as u16
                        && i > 0
                        && self.ops[i - 1].inst.op == DevOp::FlashMlaDecode as u16))
            {
                18
            } else if isolate_xreduce && op == DevOp::XReduceTwoShot as u16 {
                19
            } else if uniseg {
                8
            } else if packed_prefill_segments && packed_prefill_segment_class(op).is_some() {
                packed_prefill_segment_class(op).unwrap()
            } else if op == DevOp::FlashPrefill as u16 || op == DevOp::FlashPrefillFp8 as u16 {
                // T37: the *_pffa object instantiates hd 256/512 only — other head dims
                // (Qwen/Llama hd128) stay on the fat object rather than trapping there.
                let hd = self.ops[i].inst.i[6];
                if (fa512_mode == 2 && (hd == 256 || hd == 512)) || (fa512_mode == 1 && hd == 512) {
                    2
                } else {
                    4
                }
            } else if mla_v2 && mla_v2_segment(op, self.ops[i].inst.i[4]) {
                4
            } else if seg_v2
                && pure_gemm // pure_gemm implies a prefill program — decode stays unsegmented
                && fa512_mode == 2
                && (op == DevOp::HeadNormRope as u16
                    || op == DevOp::HeadNormRopeFp8 as u16
                    || op == DevOp::FlashMerge as u16)
            {
                2
            } else if seg_q8 && pure_gemm && op == DevOp::QuantFp8 as u16 {
                8
            } else if pure_gemm {
                // Plain tiled GEMMs only. GemmGlu*/GemmNorm stay flash-class: the GEMM_ONLY
                // object strips their arms (fused-epilogue bodies would pollute its register
                // allocation), so they must run on the fat object. Pure-GEMM packets are
                // emitted UNFUSED (PLOW_NO_GLU_FUSE=1) so in practice none are emitted.
                const FP8_OPS: [DevOp; 3] =
                    [DevOp::GemmFp8, DevOp::GemmMedFp8, DevOp::GemmSmallFp8];
                const BF16_OPS: [DevOp; 3] = [DevOp::Gemm, DevOp::GemmSmall, DevOp::GemmMed];
                let fp8 = FP8_OPS.iter().any(|g| *g as u16 == op);
                let bf16 = BF16_OPS.iter().any(|g| *g as u16 == op);
                let mapped = self.ops[i].inst.i[6] != 0 && self.ops[i].inst.i[7] != 0;
                // T37 GENERALITY: BOTH modes require the TMA maps — the ws384/uni256 lean
                // objects trap on a mapless class-8 packet, and another model/emitter may
                // legitimately skip a mint. Unmapped GEMMs go to the fat object's cp.async
                // fallback instead (correct, just slower).
                let claimed = match pure_mode {
                    // T24: mapped bf16 GEMMs (lm_head) also class 8 — the uni256 lean object
                    // carries both precisions' n256 bodies, and the fat 128-reg object runs
                    // the bf16 tile spilled (measured 4.45 ms on the lm_head segment).
                    2 => (fp8 || bf16) && mapped,
                    _ => (fp8 || bf16) && mapped,
                };
                if claimed {
                    8
                } else {
                    4
                }
            } else {
                8
            }
        };
        // PLOW_SEG_PER_OP: one SEGMENT per op, i.e. host-side AQL chaining instead of the
        // persistent kernel's counter protocol. Measurement knob; see below for why it is not a
        // default.
        //
        // The runtime ALREADY chains segments with the AQL barrier bit and no host round-trip
        // (`exec/amd.rs`: "n_seg launches, ONE drain"), so this needs no new dispatch mechanism —
        // only `seg_of[i] = i`. What it buys, measured in `runtime/bench/aql_launch_floor.c` at
        // plow's exact shape (256 WG x 512 thr, 147.5 KB group segment): a fully ordered
        // agent-coherent AQL dispatch is 1.458 us/packet against the 5.72 us the counter protocol
        // spends, because the COMMAND PROCESSOR does the cache maintenance once per packet
        // (0.25 us) instead of 256 workgroups each issuing a cache-wide op. The barrier bit itself
        // is free (1.462 vs 1.458 with it cleared) — at 1 WG/CU a 256-workgroup packet cannot
        // overlap its successor anyway.
        //
        // WHY THIS IS A KNOB AND NOT THE DEFAULT: it is unproven at TP>=4 and there is a recorded
        // failure. `exec/amd_tp.rs` quotes `runtime/tests/tp_decode.c`: "Per-rank-all-segments let
        // the ranks desync — a lagging rank made peers time out and bail, giving a WRONG, 100x-slow
        // reduction at TP>=4." That is why PREFILL takes an all-rank host barrier per segment,
        // which costs the round-trip (measured 6.538 us/packet) and would make 2459 segments a
        // LOSS. Decode today is one dispatch per rank, and its ranks rendezvous device-side, so
        // per-op segmentation is only viable if the 278 `XReduce` collectives resync the ranks
        // often enough to bound the drift by themselves. That is an empirical question about drift,
        // not a design one, and this knob is how it gets answered.
        // Materialized so later passes (SEG_CLASS_SLICE mutates self.ops) can read it.
        let op_class: Vec<u8> = (0..self.ops.len()).map(wave_class).collect();
        let wave_class = |i: usize| -> u8 { op_class[i] };
        let seg_per_op = std::env::var("PLOW_SEG_PER_OP").ok().as_deref() == Some("1");
        let mut seg_of = vec![0u16; self.ops.len()];
        let mut cur_seg = 0u16;
        for i in 0..self.ops.len() {
            if seg_per_op {
                // `cur_seg` is not just a loop temporary — `n_seg` below is derived from it, so it
                // must track the highest seg actually emitted or the gq window table is sized 1
                // while entries carry seg up to n_ops-1.
                cur_seg = u16::try_from(i).expect("PLOW_SEG_PER_OP: >65535 ops does not fit seg");
                seg_of[i] = cur_seg;
                continue;
            }
            if i > 0 && wave_class(i) != wave_class(i - 1) {
                cur_seg += 1;
            }
            seg_of[i] = cur_seg;
        }
        // A standalone raw kernel cannot participate in the interpreter's counter protocol.
        // The HSA queue barrier between segment launches already orders every earlier segment
        // before every later one, so cross-segment counter edges are redundant. Keep all
        // same-segment edges unchanged; this applies only to programs carrying the raw boundary.
        let raw_segmented = self
            .ops
            .iter()
            .any(|op| op.inst.op == DevOp::KdaDecodeFused as u16)
            || lean_moe_stage2
            || lean_moe_stage1
            || lean_moe_combine
            || moe_prefill_ep
            || lean_kda_intra
            || lean_kda_key_factor
            || xr_attnres
            || mla_materialized
            || decode_mla_segments
            || decode_grouped_moe
            || isolate_xreduce;
        let same_segment_dep = |consumer: usize, dep: &Dep| {
            let producer = dep.producer() as usize;
            let raw_moe_pair_edge =
                decode_grouped_moe && wave_class(consumer) == 20 && wave_class(producer) == 20;
            !raw_moe_pair_edge && (!raw_segmented || seg_of[consumer] == seg_of[producer])
        };
        let mut same_segment_consumer = vec![false; self.ops.len()];
        let mut same_segment_fine_consumer = vec![false; self.ops.len()];
        for (consumer, op) in self.ops.iter().enumerate() {
            for dep in &op.deps {
                if same_segment_dep(consumer, dep) {
                    let producer = dep.producer() as usize;
                    same_segment_consumer[producer] = true;
                    if matches!(dep, Dep::Fine { .. }) {
                        same_segment_fine_consumer[producer] = true;
                    }
                }
            }
        }

        // PLOW_SEG_DUMP=1: report the segmentation this program actually got.
        //
        // The wave class is decided here, re-derived independently by the host (`exec/amd.rs`
        // `derive_segments`), and never written to the build manifest — so "what occupancy does
        // this model launch at?" was answerable only by reading two files and simulating the loop
        // in one's head. It is also the field whose corruption produces the all-zero-logits
        // failure described directly below, which is the strongest argument for being able to
        // print it. Diagnostic only: no packet bytes depend on this.
        if !seg_per_op && std::env::var("PLOW_SEG_DUMP").ok().as_deref() == Some("1") {
            let mut counts: Vec<(u8, usize)> = Vec::new();
            for i in 0..self.ops.len() {
                let cls = wave_class(i); // signature changed to index-based (T24/T37 classing)
                match counts.last_mut() {
                    Some((c, n)) if *c == cls && seg_of[i] == seg_of[i.saturating_sub(1)] => {
                        *n += 1
                    }
                    _ => counts.push((cls, 1)),
                }
            }
            let summary: Vec<String> = counts
                .iter()
                .enumerate()
                .map(|(s, (cls, n))| format!("seg{s}={cls}w x{n}"))
                .collect();
            eprintln!(
                "  wave-class segments: {} segment(s) over {} ops -- {}",
                cur_seg as usize + 1,
                self.ops.len(),
                summary.join(", ")
            );
        }

        // Locality census (`PLOW_PLACE_REPORT=1`). Diagnostic only — reads the op DAG, writes
        // nothing. Answers the question a locality-aware placement pass has to answer FIRST:
        // how much of this program's slice-level dataflow could same-domain placement capture?
        if std::env::var("PLOW_PLACE_REPORT").ok().as_deref() == Some("1") {
            self.locality_census(l2_place);
        }

        // SEGMENT-CLASS RE-SLICING (PLOW_SEG_CLASS_SLICE=1, T10). occ-2 makes 2 blocks/SM
        // resident = 2*n_cu resident blocks. A GEMM-class (wave-class-8) segment's
        // machine-filling ops must be sliced to 2*n_cu so BOTH resident blocks per SM get
        // GEMM work — otherwise half the grid idles at occ-2. Flash-class (wave-class-4)
        // segments keep n_cu (they run on the occ-1 object). We ONLY double ops that already
        // fill the machine (`cus.len() == n_cu`); smaller ops keep their slice count. The
        // counter thresholds follow AUTOMATICALLY because every wait threshold is derived from
        // `producer.cus.len()` below, so doubling `cus` doubles the threshold in lock-step —
        // the counter-DAG discipline holds by construction. Fine producers / fine consumers
        // are skipped: their per-slice map[] is built for the ORIGINAL slice count, so doubling
        // would desync the map (this only occurs in heterogeneous MoE regions; a dense prefill
        // has none after select_granularity downgrades). Unflagged ⇒ no-op ⇒ byte-identical.
        // "1" = classic (double filling class-8 GEMM ops, for the occ-2 ws object);
        // "light" (T25) = double ONLY class-4 light ops — the uni256 GEMM object is occ-1
        // (grid 132), and 264 slices there make every block run the full TMA-ring
        // prologue/drain TWICE per op (measured ~30% in-model loss vs the standalone probe).
        let slice_env = std::env::var("PLOW_SEG_CLASS_SLICE").ok();
        let slice_mode = match slice_env.as_deref() {
            Some("1") => 1u8,
            Some("light") => 2u8,
            _ => 0u8,
        };
        let seg_class_slice = !uniseg && slice_mode != 0;
        // PLOW_SEG_SLICE_ALL=1 (T14): also double the machine-filling FLASH-class (light) ops
        // — the FATLITE object runs them at occ-2, so both resident blocks need slices.
        // Class-2 (dedicated flash) ops keep n_cu: the FA object is occ-1.
        let seg_slice_all =
            seg_class_slice && std::env::var("PLOW_SEG_SLICE_ALL").ok().as_deref() == Some("1");
        if seg_class_slice {
            let n_cu_sz = self.n_cu as usize;
            // Ops that some other op depends on FINELY — skip these (map[] would desync).
            let mut fine_prod = vec![false; self.ops.len()];
            for op in &self.ops {
                for d in &op.deps {
                    if let Dep::Fine { producer, .. } = d {
                        fine_prod[*producer as usize] = true;
                    }
                }
            }
            for i in 0..self.ops.len() {
                let gemm_class = (slice_mode == 1 && wave_class(i) == 8)
                    || (seg_slice_all && wave_class(i) == 4);
                let fills = self.ops[i].cus.len() == n_cu_sz;
                let has_fine = self.ops[i]
                    .deps
                    .iter()
                    .any(|d| matches!(d, Dep::Fine { .. }));
                if gemm_class && fills && !has_fine && !fine_prod[i] {
                    let orig = self.ops[i].cus.clone();
                    let mut cus = orig.clone();
                    cus.extend_from_slice(&orig); // slices 0..2*n_cu-1, cu ids valid (repeated)
                    self.ops[i].cus = cus;
                    self.ops[i].inst.blocks = (2 * n_cu_sz) as u16;
                    self.ops[i].work = vec![1u32; 2 * n_cu_sz];
                }
            }
        }

        // FINER DECODE-GEMV SLICES ([`Builder::set_gemv_split`], the `PLOW_GEMV_SPLIT` knob).
        // Emit S*n_cu slices for the three wide decode GEMVs instead of n_cu, so a workgroup that
        // finishes early claims another slice off the global queue instead of waiting at the
        // barrier. The RATIONALE and the MEASUREMENT live with the knob in `devgen`; this is the
        // mechanism only.
        //
        // WHY THE THREE OPS AND ONLY THEM. `d_gemv`/`d_gemv_glu`/`d_gemv_qkv` are all GV_BLOCKED
        // and OUTPUT-STATIONARY: slice s owns the contiguous column run [s*per, s*per+per),
        // per = ceil(N/nblk), and reduces the whole of K inside one wave. Re-slicing therefore
        // moves which workgroup computes a column and NOTHING else — the K accumulation order is
        // unchanged, so the logits are BIT-IDENTICAL and the token stream is identical by
        // construction, not by luck (verified: same md5 over 64 greedy tokens at S=1/2/4).
        // Ops with a cross-slice epilogue must NOT be listed here: `GemvArgmax` writes one partial
        // per slice and `ArgmaxFin` folds exactly `all.len()` of them (see devgen `nparts`), so
        // re-slicing it would silently drop partials.
        //
        // Deadlock-freedom is preserved: the streams stay op-major topological and every wait
        // threshold below is derived from `producer.cus.len()`, so the threshold tracks the new
        // slice count in lock-step. Fine producers/consumers are skipped for the same reason as
        // `PLOW_SEG_CLASS_SLICE` above (their per-slice map[] is built for the original count).
        let gemv_split = self.gemv_split as usize;
        if gemv_split > 1 {
            let n_cu_sz = self.n_cu as usize;
            let mut fine_prod = vec![false; self.ops.len()];
            for op in &self.ops {
                for d in &op.deps {
                    if let Dep::Fine { producer, .. } = d {
                        fine_prod[*producer as usize] = true;
                    }
                }
            }
            let wide_gemv = |op: u16| {
                op == DevOp::Gemv as u16
                    || op == DevOp::GemvGlu as u16
                    || op == DevOp::GemvQkv as u16
            };
            let mut resliced = 0usize;
            for i in 0..self.ops.len() {
                let fills = self.ops[i].cus.len() == n_cu_sz;
                let has_fine = self.ops[i]
                    .deps
                    .iter()
                    .any(|d| matches!(d, Dep::Fine { .. }));
                if wide_gemv(self.ops[i].inst.op) && fills && !has_fine && !fine_prod[i] {
                    let orig = self.ops[i].cus.clone();
                    let mut cus = Vec::with_capacity(orig.len() * gemv_split);
                    for _ in 0..gemv_split {
                        cus.extend_from_slice(&orig); // cu ids repeat; placement is cursor-claimed
                    }
                    self.ops[i].inst.blocks = cus.len() as u16;
                    self.ops[i].work = vec![1u32; cus.len()];
                    self.ops[i].cus = cus;
                    resliced += 1;
                }
            }
            eprintln!(
                "  gemv split S={gemv_split}: {resliced} gemv/gemv_glu/gemv_qkv packets \
                 resliced {n_cu_sz} -> {} slices",
                n_cu_sz * gemv_split
            );
        }

        for (idx, op) in self.ops.iter().enumerate() {
            let mut inst = op.inst;

            // The op's COARSE lists, on the instruction. A dep's threshold is how the
            // PRODUCER was sliced — deriving it here is the whole reason the builder owns the
            // CU sets: a hand-written threshold is a deadlock.
            inst.wait_ofs = waits.len() as u32;
            for d in &op.deps {
                if !same_segment_dep(idx, d) {
                    continue;
                }
                let producer = &self.ops[d.producer() as usize];
                // A Fine dep still needs a coarse fallback entry only if we are NOT emitting
                // per-slice lists for this op — but we always are (see `fine` below), so a
                // Fine dep contributes nothing to the instruction's list.
                if let Dep::Coarse(c) = d {
                    waits.push(Wait {
                        id: *c,
                        threshold: producer.cus.len() as u32,
                    });
                }
            }
            inst.wait_len = (waits.len() as u32 - inst.wait_ofs) as u16;

            inst.succ_ofs = succs.len() as u32;
            if !raw_segmented || same_segment_consumer[idx] {
                succs.push(op.counter);
                inst.succ_len = 1;
            } else {
                inst.succ_len = 0;
            }

            let has_fine_dep = op
                .deps
                .iter()
                .any(|d| same_segment_dep(idx, d) && matches!(d, Dep::Fine { .. }));
            let is_fine_producer =
                fine_base[idx] != u32::MAX && (!raw_segmented || same_segment_fine_consumer[idx]);
            let fine = has_fine_dep || is_fine_producer;

            // `slice` is the op-local index of this workgroup, NOT the CU id: the op's
            // kernel splits its work into `blocks` shares and this is which share.
            for (slice, &cu) in op.cus.iter().enumerate() {
                let mut e = StreamEnt {
                    inst: idx as u32,
                    slice: slice as u32,
                    ..Default::default()
                };
                // Keep the ordered kernel-family segment even under L2 placement. The per-slice domain
                // is packed into flags below, so a pure family can run on a lean object while
                // every launch still has one independently drained GQ window per XCD.
                //
                // `domain_of` — NOT an inline `cu / sms`. `cu` here is a LOGICAL workgroup index
                // (`interp`'s `blockIdx.x`), and only NVIDIA fills a GPC with consecutive blocks.
                // AMD dispatches workgroups to XCDs round-robin (MEASURED: `n % 8` is 100.0% of
                // the true `HW_REG_XCC_ID` on MI355X, `n / 32` is 12.5%), so hard-coding the
                // block formula here would hand every domain-0 packet to workgroups 0..31 that
                // the hardware has scattered across all eight XCDs — destroying L2 locality
                // instead of creating it, and emitting perfectly correct tokens while doing it.
                e.seg = if l2_place.is_some() && !self.uniseg_denied {
                    0
                } else {
                    seg_of[idx]
                };
                if fine {
                    e.flags = SE_FINE;

                    if has_fine_dep {
                        e.wait_ofs = waits.len() as u32;
                        for d in &op.deps {
                            if !same_segment_dep(idx, d) {
                                continue;
                            }
                            match d {
                                Dep::Coarse(c) => {
                                    let p = &self.ops[*c as usize];
                                    waits.push(Wait {
                                        id: *c,
                                        threshold: p.cus.len() as u32,
                                    });
                                }
                                Dep::Fine { producer, map } => {
                                    let base = fine_base[*producer as usize];
                                    for &ps in &map[slice] {
                                        waits.push(Wait {
                                            id: base + ps,
                                            threshold: 1,
                                        });
                                    }
                                }
                            }
                        }
                        e.wait_len = (waits.len() as u32 - e.wait_ofs) as u16;
                    } else {
                        // A fine PRODUCER with only coarse deps of its own: its wait list is
                        // the instruction's, so point at it rather than duplicating it.
                        e.wait_ofs = inst.wait_ofs;
                        e.wait_len = inst.wait_len;
                    }

                    e.succ_ofs = succs.len() as u32;
                    succs.push(op.counter); // still bump the coarse counter: it may have coarse consumers
                    if is_fine_producer {
                        succs.push(fine_base[idx] + slice as u32);
                    }
                    e.succ_len = (succs.len() as u32 - e.succ_ofs) as u16;
                } else {
                    // COARSE entry: point at the instruction's lists. The 64-byte wire
                    // DevInst64 carries no wait/succ metadata, so the interpreter reads
                    // gates from the StreamEnt unconditionally — every entry must be valid.
                    e.wait_ofs = inst.wait_ofs;
                    e.wait_len = inst.wait_len;
                    e.succ_ofs = inst.succ_ofs;
                    e.succ_len = inst.succ_len;
                }
                if let Some(l) = l2_place {
                    assert!(
                        l.domains <= 8,
                        "StreamEnt flags hold at most eight L2 domains"
                    );
                    let domain = l.domain_of(cu) as u16;
                    e.flags |= domain << crate::dev::SE_DOMAIN_SHIFT;
                }
                if xreduce_wave_rs && inst.op == DevOp::XReduceTwoShot as u16 {
                    e.flags |= crate::dev::SE_XR_WAVE_RS;
                }
                if kda_intra_wave_items
                    && inst.op == DevOp::KdaChunkIntra as u16
                    && inst.i[0] >= 512
                    && inst.i[1] != 0
                    && inst.i[2] == 128
                {
                    e.flags |= crate::dev::SE_KDA_INTRA_WAVE_ITEMS;
                }
                streams[cu as usize].push(e);
                gq_stream.push(e); // op-major: outer loop is op order, inner is slice order
            }
            insts.push(inst);
        }

        let mut stream = Vec::new();
        let mut stream_ofs = Vec::with_capacity(n_cu);
        let mut stream_len = Vec::with_capacity(n_cu);
        for s in &streams {
            stream_ofs.push(stream.len() as u32);
            stream_len.push(s.len() as u32);
            stream.extend_from_slice(s);
        }

        // Group by ordered kernel-family segment, then L2 domain. A stable sort preserves
        // op-major order within each window; cross-window deps remain counter-gated.
        // With `gq_order_asap`, each window is ordered by earliest-start rank instead (see
        // `set_gq_order_asap`); ties keep op-major order and the order stays topological.
        let asap = if self.gq_order_asap {
            Some(self.gq_asap_ranks())
        } else {
            None
        };
        let rank = |e: &StreamEnt| asap.as_ref().map_or(0, |a| a[e.inst as usize]);
        if let Some(l) = l2_place {
            gq_stream.sort_by_key(|e| {
                let domain = (e.flags & crate::dev::SE_DOMAIN_MASK) >> crate::dev::SE_DOMAIN_SHIFT;
                (e.seg as u32 * l.domains + domain as u32, rank(e))
            });
        } else if asap.is_some() {
            gq_stream.sort_by_key(|e| (e.seg, rank(e)));
        }

        // PER-(PACKET, DOMAIN) SLICE COUNT, for the two-level cache-maintenance rendezvous
        // (PLOW_SE_NPER / PLOW_GATE_HIER). This is THE number that made HIER2 unbuildable under
        // the plain global queue: there a workgroup claims whatever entry is next, so which
        // slices of a packet run on which XCD is decided at run time. Under L2 placement the
        // domain is assigned HERE, so the count is a static constant — and it is only meaningful
        // once `gq_stream` has been sorted into its per-domain windows, which is why this sits
        // after the sort and not beside the `e.seg` assignment that produces it.
        //
        // Packed into the spare high bits of `flags`, which is only ever read through masks, so
        // the struct does not grow. A count of 1 is left as 0: a single-slice packet on a domain
        // has no followers to rendezvous with, and the interpreter reads 0 as "no hierarchy".
        if l2_place.is_some() {
            let mut per: std::collections::HashMap<(u32, u16), u32> =
                std::collections::HashMap::new();
            for e in &gq_stream {
                // Fine slices can carry different wait/successor lists. Sharing one
                // (instruction, domain) rendezvous would let the elected slice stand in for
                // dependencies or signals that only a follower carries.
                if e.flags & SE_FINE != 0 {
                    continue;
                }
                let domain = (e.flags & crate::dev::SE_DOMAIN_MASK) >> crate::dev::SE_DOMAIN_SHIFT;
                *per.entry((e.inst, domain)).or_insert(0) += 1;
            }
            let mut over = 0usize;
            for e in gq_stream.iter_mut() {
                if e.flags & SE_FINE != 0 {
                    continue;
                }
                let domain = (e.flags & crate::dev::SE_DOMAIN_MASK) >> crate::dev::SE_DOMAIN_SHIFT;
                let n = *per.get(&(e.inst, domain)).unwrap_or(&0);
                if n > 1 {
                    // 9 bits. A domain cannot hold more than n_cu slices of one packet, and
                    // n_cu > 511 would need a wider field rather than a silent truncation.
                    if n > 511 {
                        over += 1;
                        continue;
                    }
                    e.flags |= (n as u16) << crate::dev::SE_NPER_SHIFT;
                }
            }
            assert_eq!(
                over, 0,
                "PLOW_SE_NPER holds 9 bits; {over} (packet, domain) pairs exceed 511 slices"
            );
        }

        // Segment window bounds in gq_stream. With L2 placement every ordered kernel-family
        // segment has one window per domain, indexed `segment * domains + domain`.
        let n_seg = match l2_place {
            Some(l) => {
                let ordered_segments = if self.uniseg_denied {
                    cur_seg as usize + 1
                } else {
                    1
                };
                ordered_segments * l.domains as usize
            }
            None => cur_seg as usize + 1,
        };
        let mut gq_seg_ofs = vec![0u32; n_seg + 1];
        {
            let mut s = 0usize;
            for (i, e) in gq_stream.iter().enumerate() {
                let key = if let Some(l) = l2_place {
                    let domain =
                        (e.flags & crate::dev::SE_DOMAIN_MASK) >> crate::dev::SE_DOMAIN_SHIFT;
                    e.seg as usize * l.domains as usize + domain as usize
                } else {
                    e.seg as usize
                };
                while key > s {
                    s += 1;
                    gq_seg_ofs[s] = i as u32;
                }
            }
            gq_seg_ofs[n_seg] = gq_stream.len() as u32;
        }

        // Static allocation report (PLOW_L2_PLACE): packets (op-slices) per L2
        // domain window, and the skew a physical-SM interp would see across
        // partitions. Emitted here so a build surfaces the balance without a GPU.
        // The MAP is printed too, because a placed blob is only as good as that
        // formula matching the hardware, and it is not visible anywhere else.
        if let Some(l) = l2_place {
            let ordered_segments = n_seg / l.domains as usize;
            let per: Vec<u32> = (0..l.domains as usize)
                .map(|d| {
                    (0..ordered_segments)
                        .map(|s| {
                            let w = s * l.domains as usize + d;
                            gq_seg_ofs[w + 1] - gq_seg_ofs[w]
                        })
                        .sum()
                })
                .collect();
            let (lo, hi) = (
                per.iter().copied().min().unwrap_or(0),
                per.iter().copied().max().unwrap_or(0),
            );
            let skew = if hi > 0 {
                100.0 * (hi - lo) as f64 / hi as f64
            } else {
                0.0
            };
            let map = match l.map {
                L2Map::Block => "block (wg n -> dom n/sms)",
                L2Map::RoundRobin => "round-robin (wg n -> dom n%domains)",
            };
            eprintln!(
                "  l2 placement: {} domains × {} ordered segments × {} SM, map {map}, packets/domain {per:?}, \
                 skew {skew:.1}% (max {hi} vs min {lo})",
                l.domains,
                ordered_segments,
                l.sms
            );
        }

        // TWO-LEVEL MAINTENANCE SCRATCH, appended to the counter region rather than given its own
        // allocation and its own pointer. Three u32 per (packet, domain): publish arrivals,
        // observe election, observe release. The region is already allocated, already zeroed by
        // the host's per-token re-arm, and already reachable from the interpreter as a counter id
        // — so extending it costs one `u32` in an existing alignment pad instead of a tenth
        // kernarg pointer, which `AmdEngine::load`'s size check has caught going wrong before.
        //
        // Only when the program is L2-placed: without per-domain windows there is no `nper` to
        // rendezvous on (see the PLOW_SE_NPER note), so the scratch would be dead weight.
        let (hier_base, n_counter) = match l2_place {
            Some(l) => (n_counter, n_counter + 3 * n_ops as u32 * l.domains),
            None => (0, n_counter),
        };

        Program {
            n_cu: self.n_cu,
            n_counter,
            hier_base,
            insts,
            stream,
            stream_ofs,
            stream_len,
            waits,
            succs,
            tensors: self.tensors,
            gq_stream,
            gq_seg_ofs,
            l2_sms: l2_place.map(|l| l.sms).unwrap_or(0),
            l2_domains: l2_place.map(|l| l.domains).unwrap_or(0),
        }
    }
}

/// Per-(CU, segment) window bounds into each CU's OWN stream slice — the static
/// interpreter's analogue of [`Program::gq_seg_ofs`].
///
/// Returns `[n_cu][n_seg+1]` `u32` in row-major order: CU `cu`'s segment `s`
/// occupies entries `[row[s], row[s+1])` of `stream[stream_ofs[cu] ..
/// stream_ofs[cu]+stream_len[cu]]`, with `row = &out[cu*(n_seg+1)..]`. The
/// indices are RELATIVE to that CU's slice, which is exactly how the interpreter
/// indexes (`my = stream + stream_ofs[cu]`).
///
/// # Why this is well-defined
///
/// `Builder::finish` assigns `seg_of[i]` as a run-length encoding over ops that
/// only ever INCREMENTS, and pushes each CU's entries in OP ORDER. So every CU's
/// stream already holds its segments in contiguous, ascending runs, and a
/// `[lo, hi)` window is as valid here as `gq_seg_ofs` is on the op-major stream.
///
/// # Why it is DERIVED from the entries rather than from the wave-class shape
///
/// Under L2-domain placement (`PLOW_NV_PLACE`) `seg` is `cu / sms` — a per-slice
/// L2 domain, not a wave class — so every entry in one CU's stream carries the
/// SAME `seg`. Windowing stays correct (one full run, empty windows elsewhere)
/// only because this reads the entries. Deriving it from "one segment per
/// wave-class run" would be wrong there, which is the standing bug shape
/// (knob-contract §4: an arm that is correct for the shape it was written for
/// and silently wrong for the other one).
///
/// Errors if any CU's stream is NOT non-decreasing in `seg` (which would mean
/// the invariant above was broken upstream) or if an entry names a segment `>=
/// n_seg`. Both are load-time refusals rather than a silently truncated run.
pub fn static_seg_ofs(
    stream: &[StreamEnt],
    stream_ofs: &[u32],
    stream_len: &[u32],
    n_seg: u32,
) -> Result<Vec<u32>, String> {
    let n_cu = stream_ofs.len();
    if stream_len.len() != n_cu {
        return Err(format!(
            "stream_ofs has {n_cu} entries but stream_len has {}",
            stream_len.len()
        ));
    }
    let row = n_seg as usize + 1;
    let mut out = vec![0u32; n_cu * row];
    for cu in 0..n_cu {
        let (o, len) = (stream_ofs[cu] as usize, stream_len[cu] as usize);
        let slice = stream
            .get(o..o + len)
            .ok_or_else(|| format!("cu {cu} stream slice [{o}, {}) is out of bounds", o + len))?;
        let r = &mut out[cu * row..(cu + 1) * row];
        // Walk the runs. `s` trails the current entry's segment; every segment
        // the walk steps over gets an EMPTY window at the current index, which
        // is what makes a CU that carries only one domain (L2 placement) or
        // only some of the wave-class segments come out right.
        let mut s = 0usize;
        let mut prev = 0u16;
        for (i, e) in slice.iter().enumerate() {
            if (e.seg as u32) >= n_seg {
                return Err(format!(
                    "cu {cu} entry {i} names segment {} but the program has {n_seg}",
                    e.seg
                ));
            }
            if i > 0 && e.seg < prev {
                return Err(format!(
                    "cu {cu} stream is not monotonic in seg: entry {i} is seg {} after seg {prev} \
                     — per-(cu,seg) windows require contiguous runs",
                    e.seg
                ));
            }
            prev = e.seg;
            while e.seg as usize > s {
                s += 1;
                r[s] = i as u32;
            }
        }
        // Every segment ABOVE the highest one this CU carries closes at the end
        // of the slice, so its window is empty and the table still covers [0,len).
        r[s + 1..=n_seg as usize].fill(len as u32);
    }
    Ok(out)
}

pub struct Program {
    pub n_cu: u32,
    pub n_counter: u32,
    /// Base counter id of the two-level maintenance scratch; 0 = hierarchy off. See `DevProgram::hier_base`.
    pub hier_base: u32,
    pub insts: Vec<DevInst>,
    pub stream: Vec<StreamEnt>,
    pub stream_ofs: Vec<u32>,
    pub stream_len: Vec<u32>,
    pub waits: Vec<Wait>,
    pub succs: Vec<u32>,
    pub tensors: Vec<TensorDecl>,
    /// Op-major (topological) permutation of `stream` for the global-queue interpreter.
    pub gq_stream: Vec<StreamEnt>,
    /// Window bounds into `gq_stream`. Under L2 placement the windows are
    /// `[ordered_segment][domain]`; otherwise there is one per ordered segment.
    pub gq_seg_ofs: Vec<u32>,
    /// L2-domain placement (PLOW_L2_PLACE): SMs per partition, and the number of
    /// L2 domains per ordered kernel-family segment. `StreamEnt.seg` remains the ordered
    /// segment and flags carry the domain. The blob header carries
    /// [`PLOW_BLOB_F_L2DOM`] plus [`PLOW_BLOB_F_L2SEG`].
    pub l2_sms: u32,
    pub l2_domains: u32,
}

// \x07/\x08 = the 64-byte DevInst64 wire format (was \x05/\x06 at 104 bytes).
// Same container layout otherwise; old blobs must be recompiled with plowc.
pub const BLOB_MAGIC: &[u8; 8] = b"PLOWDEV\x07";
pub const BLOB_MAGIC_V6: &[u8; 8] = b"PLOWDEV\x08";
/// v7 = v6 plus a [`SECT_GEN_TENSORS`] directory: tensors the READER must
/// materialise (the RoPE tables), rather than data the writer expanded into the
/// init section.
///
/// This needs its own magic rather than riding v6, because the failure mode of
/// getting it wrong is silent. A v6-era reader handed a v7 blob would see
/// `init_off == INIT_NONE` on the RoPE tensors, fall through to its zero-fill
/// path, and serve a model with cos=sin=0 — fluent text, wrong text, no error.
/// Bumping the magic turns that into a load-time rejection.
pub const BLOB_MAGIC_V7: &[u8; 8] = b"PLOWDEV\x09";
/// Current L2 placement keeps ordered segments and physical domains independent.
/// The layout needs a new magic so older runtimes cannot silently read `seg` as a domain.
pub const BLOB_MAGIC_L2SEG: &[u8; 8] = b"PLOWDEV\x0a";
/// Generated-tensor v7 plus the independent segment/domain packet layout.
pub const BLOB_MAGIC_V7_L2SEG: &[u8; 8] = b"PLOWDEV\x0b";

/// Every container version this build can read.
///
/// Readers must go through this rather than spelling out their own list: the
/// runtime checks the magic in two places (parse, and the assets-dir sniff in
/// `DevBlob::find_in_dir`), and when v7 was added to only one of them a v7 blob
/// parsed correctly but was never *discovered* — `plowrt serve` just reported no
/// model. One list, one place.
pub const BLOB_MAGICS: [&[u8; 8]; 5] = [
    BLOB_MAGIC,
    BLOB_MAGIC_V6,
    BLOB_MAGIC_V7,
    BLOB_MAGIC_L2SEG,
    BLOB_MAGIC_V7_L2SEG,
];

/// Is `m` a container version this build understands?
pub fn is_blob_magic(m: &[u8; 8]) -> bool {
    BLOB_MAGICS.contains(&m)
}
pub const NAME_LEN: usize = 80;
pub const INIT_NONE: u64 = u64::MAX;

// --- decode batch ladder -----------------------------------------------------

/// Packed winners carried by one 128-byte XArgmaxFin counter line.
pub const XARGMAX_BATCH_PER_LINE: u32 = 16;
/// Widest cross-rank argmax batch supported by the packet/runtime contract.
pub const XARGMAX_MAX_BATCH: u32 = 128;

/// Number of consecutive peer-visible counter lines needed by XArgmaxFin.
pub fn xargmax_value_lines(n_batch: u32) -> Option<u32> {
    let n = n_batch.max(1);
    (n <= XARGMAX_MAX_BATCH).then(|| n.div_ceil(XARGMAX_BATCH_PER_LINE))
}

/// Widest decode rung any emit may declare. Individual operator families may impose a
/// narrower bound (for example Gemma MoE's 32-row per-CTA scratch), while walking AMD GEMV
/// objects handle widths above their 16-row compile-time bucket.
pub const DECODE_RUNG_MAX: u32 = 128;

/// [`BlobProgHeader::t`] bit marking a prefill program as a packed-dispatch-only
/// topology. The low bits remain the compiled row count, preserving the fixed
/// wire layout while allowing an ordinary and segmented program for one rung.
pub const PACKED_PREFILL_PROG: u32 = 1 << 31;

pub fn program_rows(t: u32) -> u32 {
    t & !PACKED_PREFILL_PROG
}

pub fn is_packed_prefill_program(t: u32) -> bool {
    t & PACKED_PREFILL_PROG != 0
}

pub fn packed_prefill_program_t(rows: u32) -> u32 {
    assert_eq!(
        rows & PACKED_PREFILL_PROG,
        0,
        "program row count exceeds 31 bits"
    );
    rows | PACKED_PREFILL_PROG
}

/// Index of the FIRST decode program in `prog_t`, i.e. the start of the decode
/// rung ladder. Everything before it is a prefill bucket.
///
/// THE RULE, and why it needs no new blob field. Programs are emitted
/// prefill-buckets-ascending then decode-rungs-ascending, and the two ranges are
/// ordered by construction: decode is a trailing strictly ascending run at widths no greater
/// than [`DECODE_RUNG_MAX`]. A width-128 prefill bucket may equal a width-128 decode rung, but
/// the strict comparison below cannot cross that equal-width boundary.
///
/// A blob with ONE decode program lands on `prog_t.len() - 1`, which is the
/// `progs.len() - 1` every caller used before the ladder existed.
pub fn decode_rung_lo(prog_t: &[u32]) -> usize {
    let mut lo = prog_t.len().saturating_sub(1);
    while lo > 0 && prog_t[lo - 1] <= DECODE_RUNG_MAX && prog_t[lo - 1] < prog_t[lo] {
        lo -= 1;
    }
    lo
}

// --- v6 section directory ----------------------------------------------------

pub const SECT_MAGIC: &[u8; 4] = b"SECT";

pub const SECT_PROGRAMS: u32 = 0;
pub const SECT_CUBIN: u32 = 1;
pub const SECT_HSACO: u32 = 2;
pub const SECT_WEIGHT_MAP: u32 = 3;
pub const SECT_METADATA: u32 = 4;
pub const SECT_STATIC_TENSORS: u32 = 5;
/// An array of [`crate::rope::GenTensor`] recipes. Present iff the blob is v7.
pub const SECT_GEN_TENSORS: u32 = 6;

pub const SECT_NAME_LEN: usize = 24;

/// One entry in the v6 section directory. Mirrors `PlowSectionEntry` in `dev_blob.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlobSectionEntry {
    pub kind: u32,
    pub _pad: u32,
    pub offset: u64,
    pub size: u64,
    pub name: [u8; SECT_NAME_LEN],
}

const _: () = assert!(size_of::<BlobSectionEntry>() == 48);

/// A section to embed in a v6 blob (compiler-side, not serialized directly).
pub struct SectionData {
    pub kind: u32,
    pub name: String,
    pub data: Vec<u8>,
}

/// Mirrors `PlowTensorDecl` in `runtime/common/dev_blob.h`. Locked by `tests/dev_abi.rs`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlobTensor {
    pub name: [u8; NAME_LEN],
    pub bytes: u64,
    pub init_off: u64,
}

/// Mirrors `PlowBlobHeader`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlobHeader {
    pub magic: [u8; 8],
    pub n_cu: u32,
    pub n_tensor: u32,
    pub n_prog: u32,
    pub n_kvrow: u32,
    /// Packet-stream type flags — see [`PLOW_BLOB_F_GQ`]. Lets the runtime tell a global-queue-capable
    /// blob from a static-only one at the header, without sniffing the trailing section.
    pub flags: u32,
    /// Target-GPU fingerprint ([`gpu_fingerprint`]) — the spec the blob was compiled for, so the
    /// runtime can warn when loaded on a different GPU (only `n_cu` was cross-checked before). `0`
    /// ⇒ unknown. Same byte offset as the former `_pad`, so the wire layout is unchanged. (Model
    /// arch tag + HF id ride the `SECT_METADATA` `block.json` descriptor, not this fixed header.)
    pub target: u32,
    pub init_bytes: u64,
    /// Reserved for future metadata. The header is fixed at 64 bytes (one cache line, 8-aligned) so new
    /// fields can be carved out of this block without moving the existing ones or the sections after it.
    pub reserved: [u64; 3],
}

/// [`BlobHeader::flags`] bit: the blob carries an op-major global-queue packet stream (the trailing
/// `gq_stream`/`gq_seg_ofs` appendix), so `PLOW_GLOBAL_QUEUE=1` can run it. Absent ⇒ static-only.
pub const PLOW_BLOB_F_GQ: u32 = 1;

/// [`BlobHeader::flags`] bit: the blob uses L2-domain packet placement. A runtime
/// without physical-domain dispatch must refuse it. Current blobs also carry
/// [`PLOW_BLOB_F_L2SEG`]; legacy blobs stored the domain in `StreamEnt.seg`.
pub const PLOW_BLOB_F_L2DOM: u32 = 2;

/// L2 placement keeps the ordered kernel-family segment in `StreamEnt.seg` and carries
/// the domain in `flags`. Its GQ appendix is windowed by `(segment, domain)`.
/// Older `PLOW_BLOB_F_L2DOM` blobs used `seg` itself as the domain.
pub const PLOW_BLOB_F_L2SEG: u32 = 4;

/// Stable 32-bit fingerprint of a target GPU spec name (e.g. `"H100 SXM5"`), stamped into
/// [`BlobHeader::target`] so the runtime can warn when a blob is loaded on a GPU it was not
/// compiled for (tile sizes, KV layout, RoPE, and L2 domains are all target-specific — today
/// only `n_cu` is cross-checked). `0` ⇒ unknown/unspecified (check skipped). FNV-1a; the
/// runtime resolves its device to the same canonical spec name and compares.
pub fn gpu_fingerprint(name: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in name.as_bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Mirrors `PlowProgHeader`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BlobProgHeader {
    pub n_inst: u32,
    pub n_stream: u32,
    pub n_wait: u32,
    pub n_succ: u32,
    pub n_counter: u32,
    pub t: u32,
}

const _: () = assert!(size_of::<BlobTensor>() == 96);
const _: () = assert!(size_of::<BlobHeader>() == 64);
const _: () = assert!(size_of::<BlobProgHeader>() == 24);

impl Program {
    /// Serialise to the blob the C runtime loads. Layout is little-endian and matches
    /// `runtime/common/dev_blob.h` field for field.
    pub fn to_blob(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(BLOB_MAGIC);
        for v in [
            self.n_cu,
            self.insts.len() as u32,
            self.stream.len() as u32,
            self.waits.len() as u32,
            self.succs.len() as u32,
            self.n_counter,
            self.tensors.len() as u32,
            0u32, // pad
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        // SAFETY-free: all of these are #[repr(C)] PODs whose layout the dev_abi test
        // pins against the C header, so a byte copy is exactly what the device reads.
        fn pod<T: Copy>(v: &[T], out: &mut Vec<u8>) {
            let n = std::mem::size_of_val(v);
            let p = v.as_ptr() as *const u8;
            out.extend_from_slice(unsafe { std::slice::from_raw_parts(p, n) });
        }
        let packed: Vec<crate::dev::DevInst64> = self.insts.iter().map(DevInst::pack).collect();
        pod(&packed, &mut b);
        pod(&self.stream, &mut b);
        pod(&self.stream_ofs, &mut b);
        pod(&self.stream_len, &mut b);
        pod(&self.waits, &mut b);
        pod(&self.succs, &mut b);
        // tensor table: name, bytes, and an offset into the init-data section (or NONE)
        let mut init_blob: Vec<u8> = Vec::new();
        let mut decls: Vec<u8> = Vec::new();
        for t in &self.tensors {
            let mut name = [0u8; NAME_LEN];
            let src = t.name.as_bytes();
            assert!(src.len() < NAME_LEN, "tensor name too long: {}", t.name);
            name[..src.len()].copy_from_slice(src);
            decls.extend_from_slice(&name);
            decls.extend_from_slice(&t.bytes.to_le_bytes());
            match &t.init {
                Some(d) => {
                    decls.extend_from_slice(&(init_blob.len() as u64).to_le_bytes());
                    init_blob.extend_from_slice(d);
                }
                None => decls.extend_from_slice(&u64::MAX.to_le_bytes()),
            }
        }
        b.extend_from_slice(&(init_blob.len() as u64).to_le_bytes());
        b.extend_from_slice(&decls);
        b.extend_from_slice(&init_blob);
        b
    }

    /// Total workgroup-packets — the number of trace records a run produces.
    pub fn n_trace(&self) -> usize {
        self.stream.len()
    }
}

/// A whole model: several programs (prefill, decode) over ONE shared tensor table.
///
/// They must share it. Prefill fills the KV cache and decode appends to it, so they have to
/// name the same device buffers — and the weights are 57 GiB, which nobody is loading twice.
/// The runtime binds tensors once, by name, and then just picks a program per step.
pub struct Model {
    pub n_cu: u32,
    /// Target-GPU fingerprint ([`gpu_fingerprint`]) written to [`BlobHeader::target`]. `0` ⇒
    /// unknown/unspecified (the runtime skips the GPU-match warning).
    pub target: u32,
    pub tensors: Vec<TensorDecl>,
    pub progs: Vec<Program>,
    /// Instruction indices in program 1 (decode) whose `i[3]` is the KV-cache write row.
    /// The runtime rewrites them to the current position each step. Everything else the
    /// decoder needs per step (`ids`, `pos`, `kv_len`) is already a TENSOR, and only this one
    /// operand is an immediate — so this is the entire dynamic surface of a decode step.
    pub kv_row_insts: Vec<u32>,
    /// The `T` each program was compiled for. The runtime picks the smallest prefill bucket
    /// that fits the prompt; the last entry is the decode program (T = 1).
    pub prog_t: Vec<u32>,
    /// Tensors the runtime materialises at bind time (the RoPE tables). Non-empty
    /// makes the blob v7 — see [`BLOB_MAGIC_V7`].
    pub gen: Vec<GenTensor>,
}

impl Model {
    /// Expand every [`GenTensor`] recipe back into the init section, so the blob
    /// stays v5/v6 with the tables carried inline.
    ///
    /// This is the escape hatch for readers that predate v7 — notably the C host
    /// harnesses under `runtime/tests/`, which are the shipping drivers on gfx950
    /// and sm_120 and parse the init section directly. They gain nothing from the
    /// recipe and would reject a v7 magic outright.
    pub fn bake_gen(&mut self) {
        for g in std::mem::take(&mut self.gen) {
            // A tensormap recipe CANNOT be baked: its bytes are a function of the target's
            // device address, which only exists at bind time. Baking the zero placeholder
            // would serve TMA a garbage descriptor with no error — fail at build instead.
            assert!(
                g.kind != crate::rope::GEN_TMAP_BF16
                    && g.kind != crate::rope::GEN_TMAP_E4M3
                    && g.kind != crate::rope::GEN_TMAP_KV_PAIR,
                "gen tensor {}: tensormap recipes cannot be baked into init (--no-rope-gen is \
                 incompatible with TMA packets)",
                g.tensor
            );
            let data = g
                .generate()
                .unwrap_or_else(|| panic!("gen tensor {}: unknown kind {}", g.tensor, g.kind));
            self.tensors[g.tensor as usize].init = Some(data);
        }
    }

    /// Serialise. Every field goes through the structs mirrored from `dev_blob.h`, so the
    /// layout cannot drift from what the runtime parses without `dev_abi` failing.
    ///
    /// Generated tensors need the v7 section directory, so a model that has any
    /// routes to [`Self::to_blob_v6`] rather than emitting a v5 container the
    /// recipes could not ride in.
    pub fn to_blob(&self) -> Vec<u8> {
        if !self.gen.is_empty() {
            return self.to_blob_v6(&[]);
        }
        fn pod<T: Copy>(v: &[T], out: &mut Vec<u8>) {
            let n = std::mem::size_of_val(v);
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, n)
            });
        }
        // pack the tensor decls and the init section together
        let mut init: Vec<u8> = Vec::new();
        let mut decls: Vec<BlobTensor> = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let mut name = [0u8; NAME_LEN];
            let src = t.name.as_bytes();
            assert!(src.len() < NAME_LEN, "tensor name too long: {}", t.name);
            name[..src.len()].copy_from_slice(src);
            let init_off = match &t.init {
                Some(d) => {
                    let o = init.len() as u64;
                    init.extend_from_slice(d);
                    o
                }
                None => INIT_NONE,
            };
            decls.push(BlobTensor {
                name,
                bytes: t.bytes,
                init_off,
            });
        }

        // L2-domain placement summary across programs (PLOW_L2_PLACE): all placed
        // programs share the target's (sms, domains); mark the header + carry them.
        let (l2_flag, l2_sms, l2_dom) = match self.progs.iter().find(|p| p.l2_domains > 0) {
            Some(p) => (
                PLOW_BLOB_F_L2DOM | PLOW_BLOB_F_L2SEG,
                p.l2_sms,
                p.l2_domains,
            ),
            None => (0, 0, 0),
        };
        let hdr = BlobHeader {
            magic: if l2_flag == 0 {
                *BLOB_MAGIC
            } else {
                *BLOB_MAGIC_L2SEG
            },
            n_cu: self.n_cu,
            n_tensor: self.tensors.len() as u32,
            n_prog: self.progs.len() as u32,
            n_kvrow: self.kv_row_insts.len() as u32,
            // Every program carries the op-major gq_stream appendix (emitted below), so mark the
            // stream global-queue-capable. The runtime reads this to allow PLOW_GLOBAL_QUEUE=1.
            // + F_L2DOM when any program is L2-domain-placed (PLOW_L2_PLACE); the runtime must
            // then use physical-SM domain dispatch or REFUSE the blob. reserved[1]=SMs/partition,
            // reserved[2]=domain count, so the interp reads them instead of a build define.
            flags: PLOW_BLOB_F_GQ | l2_flag,
            target: self.target,
            init_bytes: init.len() as u64,
            reserved: [0, l2_sms as u64, l2_dom as u64],
        };
        let mut b = Vec::new();
        pod(&[hdr], &mut b);
        pod(&decls, &mut b);
        b.extend_from_slice(&init);
        pod(&self.kv_row_insts, &mut b);
        for (i, p) in self.progs.iter().enumerate() {
            pod(
                &[BlobProgHeader {
                    n_inst: p.insts.len() as u32,
                    n_stream: p.stream.len() as u32,
                    n_wait: p.waits.len() as u32,
                    n_succ: p.succs.len() as u32,
                    n_counter: p.n_counter,
                    t: self.prog_t[i],
                }],
                &mut b,
            );
            let packed: Vec<crate::dev::DevInst64> = p.insts.iter().map(DevInst::pack).collect();
            pod(&packed, &mut b);
            pod(&p.stream, &mut b);
            pod(&p.stream_ofs, &mut b);
            pod(&p.stream_len, &mut b);
            pod(&p.waits, &mut b);
            pod(&p.succs, &mut b);
        }
        // GLOBAL-QUEUE appendix (Experiment E1). OPTIONAL trailing section: loaders that stop
        // after the n_prog programs never read it, so the static path and the other harnesses
        // (gemma4_prefill.c, net_gemma_block_test.c) are unaffected and no magic bump is needed.
        // Layout: "GQ01", then per program { n_seg:u32, gq_stream[n_stream], gq_seg_ofs[n_seg+1] }.
        // gq_stream length == that program's n_stream (it is a permutation), so it is not restated.
        b.extend_from_slice(b"GQ01");
        for p in &self.progs {
            let n_seg = p.gq_seg_ofs.len() as u32 - 1;
            b.extend_from_slice(&n_seg.to_le_bytes());
            pod(&p.gq_stream, &mut b);
            pod(&p.gq_seg_ofs, &mut b);
        }
        b
    }

    /// Serialise as a v6 blob with an appended section directory. The programs payload
    /// is byte-identical to v5; the section directory and section data follow the GQ01
    /// appendix. When `sections` is empty this still produces a valid v6 blob (the
    /// directory is present but has zero entries).
    ///
    /// If [`Self::gen`] is non-empty the recipes are prepended as a
    /// [`SECT_GEN_TENSORS`] section and the container becomes v7.
    pub fn to_blob_v6(&self, sections: &[SectionData]) -> Vec<u8> {
        fn pod<T: Copy>(v: &[T], out: &mut Vec<u8>) {
            let n = std::mem::size_of_val(v);
            out.extend_from_slice(unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, n)
            });
        }

        // Generated tensors ride in front of the caller's sections so a reader can
        // resolve every tensor before it touches anything optional.
        let mut all: Vec<SectionData> = Vec::with_capacity(sections.len() + 1);
        if !self.gen.is_empty() {
            let mut data = Vec::with_capacity(self.gen.len() * size_of::<GenTensor>());
            pod(&self.gen, &mut data);
            all.push(SectionData {
                kind: SECT_GEN_TENSORS,
                name: "rope".into(),
                data,
            });
        }
        all.extend(sections.iter().map(|s| SectionData {
            kind: s.kind,
            name: s.name.clone(),
            data: s.data.clone(),
        }));
        let sections: &[SectionData] = &all;

        // A generated tensor whose declared size disagrees with what the recipe
        // produces would leave the tail of the buffer uninitialised on device.
        for g in &self.gen {
            let want = self.tensors[g.tensor as usize].bytes;
            let got = g.generate().map(|d| d.len() as u64);
            assert_eq!(
                Some(want),
                got,
                "gen tensor {} ({}): recipe produces {:?} bytes, decl says {want}",
                g.tensor,
                self.tensors[g.tensor as usize].name,
                got
            );
        }

        // --- tensor decls + init (same as v5) ---
        let mut init: Vec<u8> = Vec::new();
        let mut decls: Vec<BlobTensor> = Vec::with_capacity(self.tensors.len());
        for t in &self.tensors {
            let mut name = [0u8; NAME_LEN];
            let src = t.name.as_bytes();
            assert!(src.len() < NAME_LEN, "tensor name too long: {}", t.name);
            name[..src.len()].copy_from_slice(src);
            let init_off = match &t.init {
                Some(d) => {
                    let o = init.len() as u64;
                    init.extend_from_slice(d);
                    o
                }
                None => INIT_NONE,
            };
            decls.push(BlobTensor {
                name,
                bytes: t.bytes,
                init_off,
            });
        }

        // L2-domain placement summary (PLOW_L2_PLACE) — see to_blob().
        let (l2_flag, l2_sms, l2_dom) = match self.progs.iter().find(|p| p.l2_domains > 0) {
            Some(p) => (
                PLOW_BLOB_F_L2DOM | PLOW_BLOB_F_L2SEG,
                p.l2_sms,
                p.l2_domains,
            ),
            None => (0, 0, 0),
        };
        // Header placeholder — sect_dir_offset patched after we know the full layout.
        let hdr = BlobHeader {
            magic: match (self.gen.is_empty(), l2_flag == 0) {
                (true, true) => *BLOB_MAGIC_V6,
                (false, true) => *BLOB_MAGIC_V7,
                (true, false) => *BLOB_MAGIC_L2SEG,
                (false, false) => *BLOB_MAGIC_V7_L2SEG,
            },
            n_cu: self.n_cu,
            n_tensor: self.tensors.len() as u32,
            n_prog: self.progs.len() as u32,
            n_kvrow: self.kv_row_insts.len() as u32,
            flags: PLOW_BLOB_F_GQ | l2_flag,
            target: self.target,
            init_bytes: init.len() as u64,
            // reserved[0] = sect_dir_offset (patched below); [1]=SMs/partition, [2]=domain count.
            reserved: [0, l2_sms as u64, l2_dom as u64],
        };
        let mut b = Vec::new();
        pod(&[hdr], &mut b);
        pod(&decls, &mut b);
        b.extend_from_slice(&init);
        pod(&self.kv_row_insts, &mut b);

        for (i, p) in self.progs.iter().enumerate() {
            pod(
                &[BlobProgHeader {
                    n_inst: p.insts.len() as u32,
                    n_stream: p.stream.len() as u32,
                    n_wait: p.waits.len() as u32,
                    n_succ: p.succs.len() as u32,
                    n_counter: p.n_counter,
                    t: self.prog_t[i],
                }],
                &mut b,
            );
            let packed: Vec<crate::dev::DevInst64> = p.insts.iter().map(DevInst::pack).collect();
            pod(&packed, &mut b);
            pod(&p.stream, &mut b);
            pod(&p.stream_ofs, &mut b);
            pod(&p.stream_len, &mut b);
            pod(&p.waits, &mut b);
            pod(&p.succs, &mut b);
        }

        // GQ01 appendix (same as v5)
        b.extend_from_slice(b"GQ01");
        for p in &self.progs {
            let n_seg = p.gq_seg_ofs.len() as u32 - 1;
            b.extend_from_slice(&n_seg.to_le_bytes());
            pod(&p.gq_stream, &mut b);
            pod(&p.gq_seg_ofs, &mut b);
        }

        // --- v6 section data (appended after GQ01) ---
        let mut sect_entries: Vec<BlobSectionEntry> = Vec::with_capacity(sections.len());
        for s in sections {
            let offset = b.len() as u64;
            b.extend_from_slice(&s.data);
            let mut name = [0u8; SECT_NAME_LEN];
            let src = s.name.as_bytes();
            let copy_len = src.len().min(SECT_NAME_LEN - 1);
            name[..copy_len].copy_from_slice(&src[..copy_len]);
            sect_entries.push(BlobSectionEntry {
                kind: s.kind,
                _pad: 0,
                offset,
                size: s.data.len() as u64,
                name,
            });
        }

        // Section directory
        let sect_dir_offset = b.len() as u64;
        b.extend_from_slice(SECT_MAGIC);
        b.extend_from_slice(&(sections.len() as u32).to_le_bytes());
        pod(&sect_entries, &mut b);

        // Patch reserved[0] in the header with the section directory offset.
        // BlobHeader layout: magic(8) + n_cu(4) + n_tensor(4) + n_prog(4) + n_kvrow(4)
        //                    + flags(4) + _pad(4) + init_bytes(8) = 40 bytes before reserved[0].
        let reserved0_off = 40;
        b[reserved0_off..reserved0_off + 8].copy_from_slice(&sect_dir_offset.to_le_bytes());

        b
    }
}

impl Builder {
    /// Hand the tensor table to another Builder so two programs address the same buffers.
    pub fn adopt_tensors(&mut self, tensors: Vec<TensorDecl>) {
        self.tensors = tensors;
    }
    pub fn tensors(&self) -> Vec<TensorDecl> {
        self.tensors.clone()
    }
}

#[cfg(test)]
mod moe_prefill_ep_tests {
    use super::*;

    fn graph(with_reduction: bool) -> Builder {
        let mut b = Builder::new(8);
        let routes = b.tensor("routes", 8192 * 16 * 4);
        let meta = b.tensor("meta", (3 * 896 + 1) * 4);
        let row_token = b.tensor("row_token", 8192 * 16 * 4);
        let row_partidx = b.tensor("row_partidx", 8192 * 16 * 4);
        let row_gate = b.tensor("row_gate", 8192 * 16 * 4);
        let act = b.tensor("act", 8192 * 3584 * 2);
        let up = b.tensor("up", 8192 * 16 * 384 / 2);
        let up_scale = b.tensor("up_scale", 8192 * 16 * 384 / 32);
        let part = b.tensor("part", 8192 * 16 * 3584 * 4);
        let out = b.tensor("out", 8192 * 3584 * 2);
        let up_weights = b.tensor("expert_weight_table", 896 * 8);
        let up_scales = b.tensor("expert_scale_table", 896 * 8);
        b.tensor("expert_weight_table_moe2", 896 * 8);
        b.tensor("expert_scale_table_moe2", 896 * 8);
        let all = b.all();
        let align = b.emit(DevOp::MoeAlignPf, all.clone(), &[], |d| {
            d.t[..5].copy_from_slice(&[meta, routes, row_token, row_partidx, row_gate]);
            d.i[..3].copy_from_slice(&[8192, 896, 16]);
        });
        let glu = b.emit(DevOp::MoeGroupGluPf, all.clone(), &[align], |d| {
            d.t.copy_from_slice(&[
                up,
                act,
                up_weights,
                up_scales,
                meta,
                row_token,
                row_partidx,
                up_scale,
            ]);
            d.i[..6].copy_from_slice(&[384, 3584, 896, 2, 0, 1]);
        });
        let down = b.emit(DevOp::MoeGroupDownPf, all.clone(), &[glu], |d| {
            d.t.copy_from_slice(&[
                part,
                up,
                up_weights,
                up_scales,
                meta,
                up_scale,
                row_partidx,
                row_gate,
            ]);
            d.i[..4].copy_from_slice(&[3584, 384, 896, 2]);
        });
        let combine = b.emit(DevOp::MoeCombinePf, all.clone(), &[down], |d| {
            d.t[0] = out;
            d.t[3] = part;
            d.i[..3].copy_from_slice(&[3584, 16, 8192]);
        });
        if with_reduction {
            b.emit(DevOp::XReduce, all, &[combine], |d| {
                d.t[0] = out;
                d.i[0] = 8192 * 3584;
                d.i[1] = 8;
            });
        }
        b
    }

    #[test]
    fn whole_graph_ep_rewrite_encodes_full_i_and_fixed_slot_ownership() {
        let mut b = graph(true);
        assert_eq!(b.rewrite_replicated_moe_prefill_ep(8), 1);
        let align = &b.ops[0].inst;
        let glu = &b.ops[1].inst;
        let down = &b.ops[2].inst;
        let combine = &b.ops[3].inst;
        assert_eq!(align.i[5], 8);
        assert_eq!((glu.i[0], glu.i[6]), (3072, 8));
        assert_eq!((glu.t[2], glu.t[3]), (14, 15));
        assert_eq!((down.i[1], down.i[6]), (3072, 8));
        assert_eq!((down.t[2], down.t[3]), (14, 15));
        assert_eq!((combine.t[4], combine.i[5], combine.i[6]), (0, 8, 896));
        assert_eq!(b.tensors[1].bytes, (67 * 896 + 1) * 4);
        assert_eq!(b.tensors[6].bytes, 8192 * 16 * 3072 / 2);
        assert_eq!(b.tensors[7].bytes, 8192 * 16 * 3072 / 32);
        assert_eq!(b.tensors[14].name, "expert_weight_table_ep");
        assert_eq!(b.tensors[15].name, "expert_scale_table_ep");
        assert_eq!(b.tensors[16].name, "expert_weight_table_moe2_ep");
        assert_eq!(b.tensors[17].name, "expert_scale_table_moe2_ep");

        // Shared tensor tables are adopted by every prefill rung. Rewriting an already widened
        // declaration must be idempotent, not multiply it by the TP degree again.
        assert_eq!(b.rewrite_replicated_moe_prefill_ep(8), 0);
        assert_eq!(b.tensors[6].bytes, 8192 * 16 * 3072 / 2);
    }

    #[test]
    fn whole_graph_ep_rewrite_requires_a_replicated_reduction_boundary() {
        let mut b = graph(false);
        assert_eq!(b.rewrite_replicated_moe_prefill_ep(8), 0);
        assert!(b.ops.iter().all(|op| op.inst.i[6] == 0));
    }
}

/// L2-domain placement: the workgroup->domain formula, and when placement declines.
///
/// The formula is the entire feature. A wrong one still emits correct tokens — it just puts a
/// domain's packets on workgroups the hardware runs somewhere else — so nothing downstream can
/// catch it and it has to be pinned here.
/// Why locality-aware placement has nothing to win on these programs, as a test rather than a
/// paragraph. See the design notes.
#[cfg(test)]
mod whole_graph_fusion_tests {
    use super::*;

    fn graph(scale: f32, pre: bool) -> Builder {
        let mut b = Builder::new(4);
        let out = b.tensor("out", 128);
        let a = b.tensor("a", 128);
        let rhs = b.tensor("b", 128);
        let outer = b.tensor("pre", 128);
        let ring = b.tensor("ring", 128);
        let score = b.tensor("score", 256);
        let seed = b.emit(DevOp::Nop, vec![0], &[], |_| {});
        let residual = b.emit(DevOp::Residual, vec![0, 1], &[seed], |d| {
            d.t[0] = out;
            d.t[1] = a;
            d.t[2] = rhs;
            d.t[3] = pre.then_some(outer).unwrap_or(TENSOR_NONE);
            d.i[0] = 64;
            d.f[0] = scale;
        });
        let consumer = b.emit(DevOp::AttnRes, vec![0], &[residual], |d| {
            d.t[0] = a;
            d.t[1] = out;
            d.t[2] = ring;
            d.t[3] = score;
            d.i[0] = 1;
            d.i[1] = 64;
            d.i[5] = TENSOR_NONE_I;
        });
        b.emit(DevOp::Nop, vec![0], &[consumer], |_| {});
        b
    }

    #[test]
    fn whole_graph_fusion_preserves_materialization_rounding_and_remaps_counters() {
        for pre in [false, true] {
            let mut b = graph(1.0, pre);
            assert_eq!(b.fuse_materialized_residual_inputs(), 1);
            assert_eq!(b.ops.len(), 3);
            let fused = &b.ops[1];
            assert_eq!(fused.inst.op, DevOp::AttnRes as u16);
            assert_eq!((fused.inst.t[6], fused.inst.t[7]), (1, 2));
            assert_eq!(fused.inst.i[5], if pre { 3 } else { TENSOR_NONE_I });
            assert!(matches!(fused.deps.as_slice(), [Dep::Coarse(0)]));
            assert_eq!(fused.counter, 1);
            assert!(matches!(b.ops[2].deps.as_slice(), [Dep::Coarse(1)]));
            assert_eq!(b.ops[2].counter, 2);
        }
    }

    #[test]
    fn materialized_residual_fusion_defaults_on_and_has_a_rollback() {
        let fused = graph(1.0, false).finish();
        assert_eq!(
            fused
                .insts
                .iter()
                .filter(|inst| inst.op == DevOp::Residual as u16)
                .count(),
            0
        );

        let mut rollback = graph(1.0, false);
        rollback.set_fuse_materialized_residual_inputs(false);
        assert_eq!(
            rollback
                .finish()
                .insts
                .iter()
                .filter(|inst| inst.op == DevOp::Residual as u16)
                .count(),
            1
        );
    }

    #[test]
    fn whole_graph_fusion_rejects_non_unit_residuals() {
        let mut b = graph(0.5, false);
        assert_eq!(b.fuse_materialized_residual_inputs(), 0);
        assert_eq!(b.ops.len(), 4);
    }

    #[test]
    fn whole_graph_fusion_preserves_later_fanout_consumers() {
        let mut b = graph(1.0, false);
        b.emit(DevOp::Nop, vec![0], &[1], |_| {});
        assert_eq!(b.fuse_materialized_residual_inputs(), 1);
        assert_eq!(b.ops.len(), 4);
        assert!(matches!(b.ops[3].deps.as_slice(), [Dep::Coarse(1)]));
    }

    fn xreduce_attnres_graph(extra_xr_consumer: bool) -> Builder {
        let mut b = Builder::new(8);
        let partial = b.tensor("partial", 2048);
        let prefix_in = b.tensor("prefix_in", 2048);
        let prefix = b.tensor("prefix", 2048);
        let mixed = b.tensor("mixed", 2048);
        let ring = b.tensor("ring", 8192);
        let score = b.tensor("score", 256);
        let gamma = b.tensor("gamma", 128);
        let all = b.all();
        let seed = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        let xr = b.emit(DevOp::XReduceTwoShot, all.clone(), &[seed], |d| {
            d.t[0] = partial;
            d.i[0] = 16 * 64;
            d.i[1] = 8;
            d.i[3] = 4;
            d.i[4] = 5;
        });
        let residual = b.emit(DevOp::Residual, all.clone(), &[xr], |d| {
            d.t[0] = prefix;
            d.t[1] = prefix_in;
            d.t[2] = partial;
            d.i[0] = 16 * 64;
            d.f[0] = 1.0;
        });
        let consumer = b.emit(DevOp::AttnRes, all.clone(), &[residual], |d| {
            d.t[0] = mixed;
            d.t[1] = prefix;
            d.t[2] = ring;
            d.t[3] = score;
            d.t[5] = gamma;
            d.i[0] = 16;
            d.i[1] = 64;
            d.i[2] = 2;
            d.i[4] = 4;
            d.i[5] = TENSOR_NONE_I;
            d.f[0] = 1e-5;
        });
        b.emit(DevOp::Nop, all.clone(), &[consumer], |_| {});
        if extra_xr_consumer {
            b.emit(DevOp::Nop, all, &[xr], |_| {});
        }
        b
    }

    #[test]
    fn xreduce_phase2_folds_exact_graph_contract_and_remaps_counters() {
        let mut b = xreduce_attnres_graph(false);
        assert_eq!(b.fuse_materialized_residual_inputs(), 1);
        assert_eq!(b.fuse_xreduce_attnres(), 1);
        assert_eq!(b.ops.len(), 3);
        let fused = &b.ops[1];
        assert_eq!(fused.inst.op, DevOp::XReduceTwoShot as u16);
        assert_eq!(
            fused.inst.t[0], 0,
            "ordinary reduced output remains materialized"
        );
        assert_eq!(fused.inst.t[1], 1, "residual addend");
        assert_eq!(fused.inst.t[2], 3, "AttnRes output");
        assert_eq!(fused.inst.t[3], 4, "AttnRes ring");
        assert_eq!(fused.inst.t[4], 5, "AttnRes score");
        assert_eq!(fused.inst.t[5], 6, "fused post-norm gamma");
        assert_eq!(fused.inst.t[6], 2, "rounded prefix remains materialized");
        assert_eq!(&fused.inst.i[5..], &[64, 2, 4]);
        assert_eq!(fused.inst.f[0].to_bits(), 1e-5f32.to_bits());
        assert!(matches!(b.ops[2].deps.as_slice(), [Dep::Coarse(1)]));
    }

    #[test]
    fn xreduce_phase2_rejects_a_collective_with_another_graph_consumer() {
        let mut b = xreduce_attnres_graph(true);
        assert_eq!(b.fuse_materialized_residual_inputs(), 1);
        assert_eq!(b.fuse_xreduce_attnres(), 0);
    }

    #[test]
    fn xreduce_phase2_rejects_flat_slices_that_split_rows() {
        let mut b = xreduce_attnres_graph(false);
        assert_eq!(b.fuse_materialized_residual_inputs(), 1);
        b.ops[1].inst.i[0] = 15 * 64;
        b.ops[2].inst.i[0] = 15;
        assert_eq!(b.fuse_xreduce_attnres(), 0);
    }

    #[test]
    fn xreduce_phase2_folds_a_direct_prefix_without_inventing_a_residual() {
        let mut b = Builder::new(8);
        let prefix = b.tensor("prefix", 2048);
        let mixed = b.tensor("mixed", 2048);
        let ring = b.tensor("ring", 8192);
        let score = b.tensor("score", 256);
        let gamma = b.tensor("gamma", 128);
        let all = b.all();
        let seed = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        let xr = b.emit(DevOp::XReduceTwoShot, all.clone(), &[seed], |d| {
            d.t[0] = prefix;
            d.i[0] = 16 * 64;
            d.i[1] = 8;
        });
        b.emit(DevOp::AttnRes, all, &[xr], |d| {
            d.t[0] = mixed;
            d.t[1] = prefix;
            d.t[2] = ring;
            d.t[3] = score;
            d.t[5] = gamma;
            d.i[0] = 16;
            d.i[1] = 64;
            d.i[2] = 2;
            d.i[4] = 4;
            d.i[5] = TENSOR_NONE_I;
            d.f[0] = 1e-5;
        });
        assert_eq!(b.fuse_xreduce_attnres(), 1);
        assert_eq!(b.ops[1].inst.t[1], TENSOR_NONE);
        assert_eq!(b.ops[1].inst.t[6], prefix);
        b.force_uniseg();
        let p = b.finish();
        let seg = |inst| {
            p.stream
                .iter()
                .find(|e| e.inst == inst)
                .expect("instruction has a stream entry")
                .seg
        };
        assert_ne!(
            seg(0),
            seg(1),
            "the fused op needs its spill-isolated object even under forced uniseg"
        );
        assert!(p
            .stream
            .iter()
            .filter(|e| e.seg == seg(1))
            .all(|e| e.inst == 1));
    }
}

#[cfg(test)]
mod locality_census_tests {
    use super::*;

    const D: u32 = 8;
    const LAYOUT: L2Layout = L2Layout {
        sms: 32,
        domains: D,
        map: L2Map::RoundRobin,
    };

    /// A chain of full-width ops joined by COARSE edges — the shape every dense decode and
    /// prefill program collapses to, because `select_granularity` downgrades a homogeneous
    /// region's fine edges (`CounterGranularity.collapse`) and a downgraded edge is all-to-all.
    fn coarse_chain(n_cu: u32, len: usize) -> Builder {
        let mut b = Builder::new(n_cu);
        let all: Vec<u32> = (0..n_cu).collect();
        let mut prev = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        for _ in 1..len {
            prev = b.emit(DevOp::Nop, all.clone(), &[prev], |_| {});
        }
        b
    }

    /// **The result that kills the lever.** On an all-to-all edge a consumer slice reads every
    /// producer slice, so its reads are spread over the producer's domains by construction: under
    /// ANY assignment that keeps the producer balanced, exactly `1/domains` of the pairs are
    /// same-domain. The emitted mapping already sits on that number, so a predecessor-affinity
    /// pass has nothing to add — it can only move slices around, which is what it does.
    #[test]
    fn an_all_to_all_edge_pins_same_domain_locality_at_one_over_domains() {
        let c = coarse_chain(256, 6).locality_census_stats(Some(LAYOUT));
        assert_eq!(c.pairs, c.all_to_all_pairs, "coarse edges are all-to-all");
        assert!(c.pairs > 0);
        let want = c.pairs / D as u64;
        assert_eq!(c.same_current, want, "emitted mapping = 1/{D} of pairs");
        assert_eq!(
            c.same_greedy, want,
            "greedy pred-affinity = the same 1/{D}, exactly"
        );
        assert_eq!(
            c.same_ceiling, want,
            "even the balance-free per-slice argmax cannot beat it"
        );
    }

    /// And it is not that the greedy pass is a no-op: it relocates most of the program and still
    /// gains nothing. That is the diff worth reporting — large in placement, zero in locality.
    #[test]
    fn the_greedy_pass_moves_most_slices_and_buys_nothing() {
        let c = coarse_chain(256, 6).locality_census_stats(Some(LAYOUT));
        assert!(
            c.moved_slices * 2 > c.slices,
            "greedy moved {}/{} slices — expected a majority",
            c.moved_slices,
            c.slices
        );
        assert_eq!(
            c.same_greedy, c.same_current,
            "…and changed the locality by zero"
        );
    }

    /// A SPARSE edge is the only place placement can pay: give each consumer slice exactly one
    /// producer slice and the greedy pass reaches 100%, well past the `1/domains` floor. This is
    /// the control that proves the census measures something real — without it the two tests
    /// above would also pass on a census that always returned `1/domains`.
    #[test]
    fn a_sparse_edge_is_where_placement_can_actually_pay() {
        let n_cu = 256u32;
        let mut b = Builder::new(n_cu);
        let all: Vec<u32> = (0..n_cu).collect();
        let p = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        // Slice s reads producer slice s, and the work is non-uniform so `select_granularity`
        // keeps the fine edge (`hetero_can_win`) instead of collapsing it.
        let map: Vec<Vec<u32>> = (0..n_cu).map(|s| vec![s]).collect();
        let work: Vec<u32> = (0..n_cu).map(|s| 1 + s).collect();
        b.emit_dep_work(
            DevOp::Nop,
            all,
            vec![Dep::Fine { producer: p, map }],
            work,
            |_| {},
        );
        b.select_granularity();
        let c = b.locality_census_stats(Some(LAYOUT));
        assert_eq!(c.all_to_all_pairs, 0, "a 1:1 map is sparse");
        assert_eq!(
            c.same_current, c.pairs,
            "slice s and slice s are the same workgroup index"
        );
        assert_eq!(c.same_greedy, c.pairs, "greedy holds it");
    }
}

#[cfg(test)]
mod l2_placement_tests {
    use super::*;

    fn domain(e: &StreamEnt) -> u32 {
        ((e.flags & crate::dev::SE_DOMAIN_MASK) >> crate::dev::SE_DOMAIN_SHIFT) as u32
    }

    /// One packet per op, each sliced across all `n_cu` workgroups. `reads_class` is the target
    /// fact `deny_uniseg` records: true for a host that relaunches per segment and reads the wave
    /// class out of `seg` (AMD), false for one cooperative launch that never looks (sm_120).
    fn build(n_cu: u32, ops: &[DevOp], layout: Option<L2Layout>, reads_class: bool) -> Program {
        let mut b = Builder::new(n_cu);
        b.set_l2_placement(layout);
        if reads_class {
            b.deny_uniseg();
        }
        for &op in ops {
            let all: Vec<u32> = (0..n_cu).collect();
            b.emit(op, all, &[], |_| {});
        }
        b.finish()
    }

    /// The AMD shape: the host relaunches per segment, so `seg` is read.
    fn placed(n_cu: u32, ops: &[DevOp], layout: Option<L2Layout>) -> Program {
        build(n_cu, ops, layout, true)
    }

    /// AMD. MEASURED on MI355X (`runtime/tests/xcd_map_gfx950_test.hip`): the hardware
    /// dispatcher assigns workgroup *n* to XCD `n % 8`, at 100.0% over every geometry probed.
    /// So a packet destined for domain d must be given to the workgroups where `n % 8 == d`.
    #[test]
    fn round_robin_places_workgroup_n_on_domain_n_mod_domains() {
        let l = L2Layout {
            sms: 32,
            domains: 8,
            map: L2Map::RoundRobin,
        };
        let p = placed(256, &[DevOp::Nop], Some(l));
        assert_eq!(p.l2_domains, 8, "placement must be active");
        for e in &p.stream {
            assert_eq!(
                domain(e),
                e.slice % 8,
                "slice {} must sit in domain {} (n % 8), got {}",
                e.slice,
                e.slice % 8,
                domain(e)
            );
            assert_eq!(e.seg, 0, "the ordered segment must remain independent");
        }
        // Every domain window is equally full — 8 domains × 32 of the 256 slices.
        let per: Vec<u32> = (0..8)
            .map(|d| p.gq_seg_ofs[d + 1] - p.gq_seg_ofs[d])
            .collect();
        assert_eq!(
            per,
            vec![32; 8],
            "round-robin over 256 slices must not skew"
        );
    }

    /// K3's MoE layer shape: a b=1 router chain emitted BEFORE an independent shared-expert pair
    /// that was ready earlier. ASAP ordering must hoist the pair ahead of the gated chain in
    /// every XCD window, keep every window topological, and leave the static streams alone.
    #[test]
    fn gq_asap_order_hoists_ready_packets_ahead_of_gated_ones() {
        let l = L2Layout {
            sms: 32,
            domains: 8,
            map: L2Map::RoundRobin,
        };
        let emit = |asap: bool| {
            let mut b = Builder::new(256);
            b.set_l2_placement(Some(l));
            b.deny_uniseg();
            b.set_gq_order_asap(asap);
            let all: Vec<u32> = (0..256).collect();
            let attn = b.emit(DevOp::AttnRes, vec![0], &[], |_| {});
            let router = b.emit(DevOp::MoeRouterTopk, vec![0], &[attn], |_| {});
            let glu = b.emit(DevOp::MoeGroupGluFp8Blk, all.clone(), &[router], |_| {});
            let down = b.emit(DevOp::MoeGroupDownFp8Blk, all.clone(), &[glu], |_| {});
            let sh_glu = b.emit(DevOp::GemvGlu, all.clone(), &[attn], |_| {});
            let sh_down = b.emit(DevOp::Gemv, all.clone(), &[sh_glu], |_| {});
            b.emit(DevOp::XReduce, all, &[down, sh_down], |_| {});
            (b.finish(), [attn, router, glu, down, sh_glu, sh_down])
        };
        let (base, _) = emit(false);
        let (p, ids) = emit(true);
        assert_eq!(p.stream, base.stream, "static streams must not move");
        assert_eq!(p.gq_seg_ofs, base.gq_seg_ofs, "windows must not move");
        let [_, _, glu, down, sh_glu, sh_down] = ids;
        for w in 0..p.gq_seg_ofs.len() - 1 {
            let win = &p.gq_stream[p.gq_seg_ofs[w] as usize..p.gq_seg_ofs[w + 1] as usize];
            let pos = |inst: u32| win.iter().position(|e| e.inst == inst);
            if let (Some(a), Some(g)) = (pos(sh_glu), pos(glu)) {
                assert!(a < g, "window {w}: shared GemvGlu must precede the gated MoeGroupGlu");
            }
            if let (Some(a), Some(g)) = (pos(sh_down), pos(glu)) {
                assert!(a < g, "window {w}: shared down Gemv must precede the gated MoeGroupGlu");
            }
            if let (Some(a), Some(g)) = (pos(sh_glu), pos(sh_down)) {
                assert!(a < g, "window {w}: producer before consumer");
            }
            if let (Some(a), Some(g)) = (pos(glu), pos(down)) {
                assert!(a < g, "window {w}: producer before consumer");
            }
        }
        // Same multiset of entries, just permuted.
        let mut x: Vec<_> = p.gq_stream.iter().map(|e| (e.inst, e.slice)).collect();
        let mut y: Vec<_> = base.gq_stream.iter().map(|e| (e.inst, e.slice)).collect();
        x.sort();
        y.sort();
        assert_eq!(x, y);
    }

    /// NVIDIA. Consecutive blocks fill a GPC, so the block formula stands — and this test is
    /// what stops the AMD fix from becoming a blanket rewrite of a shipped NVIDIA path.
    #[test]
    fn block_map_places_workgroup_n_on_domain_n_div_sms() {
        let l = L2Layout {
            sms: 32,
            domains: 8,
            map: L2Map::Block,
        };
        let p = placed(256, &[DevOp::Nop], Some(l));
        assert_eq!(p.l2_domains, 8);
        for e in &p.stream {
            assert_eq!(domain(e), e.slice / 32, "block map is n / sms");
            assert_eq!(e.seg, 0, "the ordered segment must remain independent");
        }
    }

    /// The two maps must actually DISAGREE on the interpreter's real grid. If they agreed,
    /// carrying the map would be ceremony; they overlap on only 32 of 256 workgroups
    /// (`n/32 == n%8`), which is the 12.5% the probe measured for the block formula.
    #[test]
    fn the_two_maps_disagree_on_the_real_decode_grid() {
        let rr = L2Layout {
            sms: 32,
            domains: 8,
            map: L2Map::RoundRobin,
        };
        let bl = L2Layout {
            sms: 32,
            domains: 8,
            map: L2Map::Block,
        };
        let agree = (0..256u32)
            .filter(|&n| rr.domain_of(n) == bl.domain_of(n))
            .count();
        assert_eq!(
            agree, 32,
            "the maps must coincide on exactly 32 of 256 workgroups"
        );
    }

    /// `n_cu > domains*sms` is occupancy>1. The block formula would then run off the end of the
    /// domain count and orphan packets, so it declines. Round-robin is in range by construction,
    /// and the probe MEASURED it still holding at occupancy 2 — so it must NOT decline, or
    /// placement would be silently off on exactly the occ-2 configs it is safe for.
    #[test]
    fn occupancy_two_declines_block_but_not_round_robin() {
        let bl = placed(
            512,
            &[DevOp::Nop],
            Some(L2Layout {
                sms: 32,
                domains: 8,
                map: L2Map::Block,
            }),
        );
        assert_eq!(bl.l2_domains, 0, "block map must decline at occupancy 2");
        let rr = placed(
            512,
            &[DevOp::Nop],
            Some(L2Layout {
                sms: 32,
                domains: 8,
                map: L2Map::RoundRobin,
            }),
        );
        assert_eq!(
            rr.l2_domains, 8,
            "round-robin is in range at occupancy 2 — measured"
        );
        for e in &rr.stream {
            assert_eq!(domain(e), e.slice % 8);
            assert_eq!(e.seg, 0);
        }
    }

    /// THE ZERO-LOGITS GUARD. Host family segments and device-side XCD queues must coexist.
    /// Collapsing either dimension dispatches packets on the wrong object or loses locality.
    #[test]
    fn a_multi_wave_class_program_keeps_segments_and_xcd_windows() {
        let l = L2Layout {
            sms: 32,
            domains: 8,
            map: L2Map::RoundRobin,
        };
        let p = placed(256, &[DevOp::Nop, DevOp::FlashPrefill, DevOp::Nop], Some(l));
        assert_eq!(p.l2_domains, 8, "placement must remain active");
        // Three ordered kernel-family segments, each with eight independently drained XCD windows.
        assert_eq!(
            p.gq_seg_ofs.len() - 1,
            24,
            "expected [ordered segment][XCD] queue windows"
        );
        assert!(
            p.stream.iter().any(|e| e.seg == 1),
            "the flash run keeps its own segment"
        );
        assert!(
            p.stream.iter().any(|e| domain(e) == 7),
            "packets must be emitted for every XCD"
        );
    }

    #[test]
    fn hierarchy_counts_exclude_fine_entries_with_slice_local_edges() {
        let mut b = Builder::new(8);
        b.set_l2_placement(Some(L2Layout {
            sms: 4,
            domains: 2,
            map: L2Map::RoundRobin,
        }));
        b.deny_uniseg();
        let cus: Vec<u32> = (0..8).collect();
        let producer = b.emit(DevOp::Nop, cus.clone(), &[], |_| {});
        let map: Vec<Vec<u32>> = (0..8).map(|slice| vec![slice]).collect();
        b.emit_dep_work(
            DevOp::Nop,
            cus,
            vec![Dep::Fine { producer, map }],
            (1..=8).collect(),
            |_| {},
        );
        let p = b.finish();

        let fine: Vec<_> = p
            .gq_stream
            .iter()
            .filter(|entry| entry.flags & SE_FINE != 0)
            .collect();
        assert!(
            !fine.is_empty(),
            "test must retain slice-local dependencies"
        );
        assert!(
            fine.iter()
                .all(|entry| entry.flags & crate::dev::SE_NPER_MASK == 0),
            "one fine slice must never rendezvous on behalf of another slice's waits/signals"
        );
    }

    #[test]
    fn raw_boundary_keeps_same_segment_counters_and_removes_cross_segment_edges() {
        let mut b = Builder::new(4);
        let cus = vec![0, 1, 2, 3];
        let coarse = b.emit(DevOp::Nop, cus.clone(), &[], |_| {});
        let fine_producer = b.emit(DevOp::Nop, cus.clone(), &[coarse], |_| {});
        let map: Vec<Vec<u32>> = (0..4).map(|slice| vec![slice]).collect();
        let before_raw = b.emit_dep_work(
            DevOp::Nop,
            cus.clone(),
            vec![Dep::Fine {
                producer: fine_producer,
                map: map.clone(),
            }],
            vec![1, 40, 3, 9],
            |_| {},
        );
        let raw = b.emit_dep_work(
            DevOp::KdaDecodeFused,
            cus.clone(),
            vec![Dep::Fine {
                producer: before_raw,
                map,
            }],
            vec![1, 40, 3, 9],
            |_| {},
        );
        b.emit(DevOp::Nop, cus, &[raw], |_| {});
        let p = b.finish();

        assert_ne!(p.insts[0].succ_len, 0, "same-segment coarse successor lost");
        assert_ne!(p.insts[1].wait_len, 0, "same-segment coarse wait lost");
        assert_ne!(p.insts[1].succ_len, 0, "same-segment fine successor lost");
        assert!(p
            .stream
            .iter()
            .filter(|entry| entry.inst == 2)
            .any(|entry| entry.flags & SE_FINE != 0 && entry.wait_len != 0));

        assert_eq!((p.insts[3].wait_len, p.insts[3].succ_len), (0, 0));
        assert_eq!(p.insts[4].wait_len, 0);
        assert!(p
            .stream
            .iter()
            .filter(|entry| entry.inst == 3)
            .all(|entry| entry.wait_len == 0 && entry.succ_len == 0 && entry.flags & SE_FINE == 0));
    }

    /// The other half: a SINGLE-wave-class program (decode has no `FlashPrefill`) has nothing in
    /// `seg` to destroy, so it is placed even on a target that reads the class. This is what
    /// makes AMD decode placement possible at all, and it is why the gate is per-program rather
    /// than per-target.
    #[test]
    fn a_single_wave_class_program_is_placed() {
        let l = L2Layout {
            sms: 32,
            domains: 8,
            map: L2Map::RoundRobin,
        };
        let p = placed(256, &[DevOp::Nop, DevOp::Gemv, DevOp::Nop], Some(l));
        assert_eq!(p.l2_domains, 8, "decode-shaped programs must be placed");
        assert_eq!(p.l2_sms, 32);
    }

    /// …and the skip is conditioned on the TARGET reading the class, not merely on the program
    /// having one. An sm_120 prefill program is wave-class-segmented too, but that interpreter
    /// runs the whole program in one cooperative launch and never reads `seg` — so placement over
    /// it is a shipped NVIDIA feature and must keep working. Without this condition the AMD fix
    /// would have silently disabled it.
    #[test]
    fn a_segmented_program_is_still_placed_when_the_target_ignores_the_class() {
        let l = L2Layout {
            sms: 32,
            domains: 8,
            map: L2Map::Block,
        };
        let p = build(
            256,
            &[DevOp::Nop, DevOp::FlashPrefill, DevOp::Nop],
            Some(l),
            false,
        );
        assert_eq!(
            p.l2_domains, 8,
            "sm_120 placement must survive the AMD gate"
        );
        assert_eq!(
            p.gq_seg_ofs.len() - 1,
            8,
            "a target that ignores ordered segments needs only eight domain queues"
        );
        assert!(p.stream.iter().all(|e| e.seg == 0));
    }

    /// Off ⇒ byte-identical. The whole feature has to be a no-op when unset, or every existing
    /// blob changes underneath the gates.
    #[test]
    fn placement_off_is_byte_identical() {
        let ops = [DevOp::Nop, DevOp::FlashPrefill, DevOp::Gemv];
        let a = placed(256, &ops, None);
        let b = placed(
            256,
            &ops,
            Some(L2Layout {
                sms: 0,
                domains: 0,
                map: L2Map::Block,
            }),
        );
        assert_eq!(a.stream, b.stream);
        assert_eq!(a.gq_stream, b.gq_stream);
        assert_eq!(a.gq_seg_ofs, b.gq_seg_ofs);
        assert_eq!(b.l2_domains, 0);
    }
}

#[cfg(test)]
mod granularity_tests {
    use super::*;

    /// Build `producer -> consumer` with a fine dep, and report how many fine edges survive
    /// `select_granularity`. `work` is the consumer's per-slice cost.
    fn survives(work: Vec<u32>) -> bool {
        let mut b = Builder::new(4);
        let p = b.emit(DevOp::Nop, vec![0, 1, 2, 3], &[], |_| {});
        // consumer slice s waits only on producer slice s
        let map: Vec<Vec<u32>> = (0..work.len() as u32).map(|s| vec![s]).collect();
        let cus: Vec<u32> = (0..work.len() as u32).collect();
        b.emit_dep_work(
            DevOp::Nop,
            cus,
            vec![Dep::Fine { producer: p, map }],
            work,
            |_| {},
        );
        // n_counter > n_ops exactly when a fine producer got per-slice counters.
        b.finish().n_counter > 2
    }

    /// A transformer's heads all do the same work, so the region is homogeneous and the
    /// `collapse` theorem (Plow/CounterGranularity.lean) says fine gates cannot win. The
    /// compiler must therefore NOT emit them — they cost counters and atomics for nothing.
    #[test]
    fn uniform_work_is_downgraded_to_coarse() {
        assert!(
            !survives(vec![10, 10, 10, 10]),
            "uniform region must fall back to coarse"
        );
    }

    /// MoE experts get different token counts by construction. Then a straggling producer can
    /// feed a CHEAP consumer and its slack is absorbed instead of reaching the barrier — which
    /// is exactly `hetero_can_win`. The compiler must keep the fine gates here.
    #[test]
    fn heterogeneous_work_keeps_fine_gates() {
        assert!(
            survives(vec![1, 40, 3, 9]),
            "imbalanced region must keep its fine gates"
        );
    }
}

#[cfg(test)]
mod seg_window_tests {
    use super::*;

    #[test]
    fn mla_v2_segments_only_machine_filling_prefill_buckets() {
        assert!(!mla_v2_segment(DevOp::FlashMlaPrefill as u16, 1024));
        assert!(!mla_v2_segment(DevOp::FlashMlaPrefillFp8 as u16, 1024));
        assert!(mla_v2_segment(DevOp::FlashMlaPrefill as u16, 2048));
        assert!(mla_v2_segment(DevOp::FlashMlaPrefillFp8 as u16, 2048));
        assert!(!mla_v2_segment(DevOp::Gemv as u16, 8192));
    }

    #[test]
    fn packed_prefill_classes_are_operator_families() {
        assert_eq!(packed_prefill_segment_class(DevOp::RmsNorm as u16), Some(5));
        assert_eq!(
            packed_prefill_segment_class(DevOp::HeadNormRope as u16),
            Some(5)
        );
        assert_eq!(
            packed_prefill_segment_class(DevOp::HeadNormRopeFp8 as u16),
            Some(5)
        );
        assert_eq!(
            packed_prefill_segment_class(DevOp::FlashMlaPrefill as u16),
            Some(6)
        );
        assert_eq!(
            packed_prefill_segment_class(DevOp::FlashMlaPrefillFp8 as u16),
            Some(6)
        );
        assert_eq!(
            packed_prefill_segment_class(DevOp::KdaConv3 as u16),
            Some(7)
        );
        assert_eq!(
            packed_prefill_segment_class(DevOp::KdaChunkIntra as u16),
            Some(7)
        );
        assert_eq!(packed_prefill_segment_class(DevOp::Gemv as u16), None);
    }

    #[test]
    fn kda_only_program_can_enable_packed_prefill_segments() {
        let ops = [DevOp::KdaChunkPrepare as u16, DevOp::Gemv as u16];
        assert!(packed_prefill_segmenting_needed(
            false,
            true,
            ops.into_iter()
        ));
        assert!(!packed_prefill_segmenting_needed(
            true,
            true,
            ops.into_iter()
        ));
        assert!(!packed_prefill_segmenting_needed(
            false,
            false,
            ops.into_iter()
        ));
        assert!(!packed_prefill_segmenting_needed(
            false,
            true,
            [DevOp::Gemv as u16].into_iter()
        ));
    }

    /// The number of segments a program's stream spans, derived the way every
    /// runtime derives it (`plowrt::exec::amd::derive_segments`,
    /// `gemma4_chat.c`): `max(seg) + 1`. There is no blob field.
    fn n_seg_of(p: &Program) -> u32 {
        p.stream.iter().map(|e| e.seg as u32 + 1).max().unwrap_or(1)
    }

    /// A program whose ops alternate wave class, so `Builder::finish` cuts it
    /// into several segments and every CU's stream carries several runs.
    fn segmented_program() -> Program {
        let mut b = Builder::new(8);
        b.deny_uniseg(); // PLOW_UNISEG in the ambient env must not flatten the fixture
        let all = b.all();
        let mut prev: Vec<u32> = Vec::new();
        // GEMM, flash, GEMM, flash, ... — the wave class flips every op, so
        // seg_of increments on every boundary.
        for i in 0..9u32 {
            let op = if i % 2 == 1 {
                DevOp::FlashPrefill
            } else {
                DevOp::Nop
            };
            // Vary the CU set so some CUs miss some ops entirely — that is the
            // case where a per-CU window is NOT just the segment's global range.
            let cus: Vec<u32> = match i % 3 {
                0 => all.clone(),
                1 => all.iter().copied().filter(|c| c % 2 == 0).collect(),
                _ => vec![0, 1, 2],
            };
            prev = vec![b.emit(op, cus, &prev, |_| {})];
        }
        b.finish()
    }

    /// THE INVARIANT the windows rest on: `seg_of` is a run-length encoding that
    /// only increments, and each CU's entries are pushed in op order — so every
    /// CU's stream is non-decreasing in `seg`. Asserted against the builder's own
    /// output, not against a literal, so a change to the segmentation rule breaks
    /// it here rather than on hardware.
    #[test]
    fn per_cu_stream_is_monotonic_in_seg() {
        let p = segmented_program();
        assert!(n_seg_of(&p) > 1, "fixture must actually be segmented");
        for cu in 0..p.n_cu as usize {
            let (o, len) = (p.stream_ofs[cu] as usize, p.stream_len[cu] as usize);
            let segs: Vec<u16> = p.stream[o..o + len].iter().map(|e| e.seg).collect();
            assert!(
                segs.windows(2).all(|w| w[0] <= w[1]),
                "cu {cu} stream is not monotonic in seg: {segs:?}"
            );
        }
    }

    /// The windows must select EXACTLY the entries the old scan-and-filter loop
    /// selected, for every (cu, seg) — that is the whole correctness claim.
    #[test]
    fn windows_select_exactly_what_the_filter_selected() {
        let p = segmented_program();
        let n_seg = n_seg_of(&p);
        let ofs = static_seg_ofs(&p.stream, &p.stream_ofs, &p.stream_len, n_seg).unwrap();
        let row = n_seg as usize + 1;
        for cu in 0..p.n_cu as usize {
            let (o, len) = (p.stream_ofs[cu] as usize, p.stream_len[cu] as usize);
            let my = &p.stream[o..o + len];
            let r = &ofs[cu * row..(cu + 1) * row];
            assert_eq!(r[0], 0, "cu {cu} window table must start at 0");
            assert_eq!(
                r[n_seg as usize], len as u32,
                "cu {cu} windows must cover the slice"
            );
            let mut covered = 0usize;
            for s in 0..n_seg {
                let (lo, hi) = (r[s as usize] as usize, r[s as usize + 1] as usize);
                assert!(
                    lo <= hi && hi <= len,
                    "cu {cu} seg {s}: bad window [{lo},{hi}) of {len}"
                );
                covered += hi - lo;
                // what the window yields
                let win: Vec<u32> = (lo..hi).map(|i| my[i].inst).collect();
                // what `if (e.seg != prog.cur_seg) continue;` yielded
                let filt: Vec<u32> = my
                    .iter()
                    .filter(|e| e.seg as u32 == s)
                    .map(|e| e.inst)
                    .collect();
                assert_eq!(win, filt, "cu {cu} seg {s}: window != filtered scan");
            }
            assert_eq!(covered, len, "cu {cu}: windows must partition the slice");
        }
    }

    /// Static per-CU streams retain ordered kernel-family segments. XCD assignment lives in flags and
    /// therefore cannot alter the segment windows used by the static fallback.
    #[test]
    fn l2_placement_keeps_static_ordered_segment_windows() {
        for map in [L2Map::Block, L2Map::RoundRobin] {
            let l = L2Layout {
                sms: 2,
                domains: 4,
                map,
            };
            let mut b = Builder::new(8);
            b.set_l2_placement(Some(l));
            b.deny_uniseg();
            let all = b.all();
            let a = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
            b.emit(DevOp::FlashPrefill, all, &[a], |_| {});
            let p = b.finish();
            let n_seg = 2;
            let ofs = static_seg_ofs(&p.stream, &p.stream_ofs, &p.stream_len, n_seg).unwrap();
            let row = n_seg as usize + 1;
            for cu in 0..p.n_cu as usize {
                let len = p.stream_len[cu] as usize;
                let r = &ofs[cu * row..(cu + 1) * row];
                for s in 0..n_seg {
                    let (lo, hi) = (r[s as usize], r[s as usize + 1]);
                    let want = 1;
                    assert_eq!(
                        hi - lo,
                        want,
                        "{map:?} cu {cu} seg {s}: expected {want} entries"
                    );
                }
                let o = p.stream_ofs[cu] as usize;
                for e in &p.stream[o..o + len] {
                    let got = ((e.flags & crate::dev::SE_DOMAIN_MASK)
                        >> crate::dev::SE_DOMAIN_SHIFT) as u32;
                    assert_eq!(got, l.domain_of(cu as u32));
                }
            }
        }
    }

    /// A stream whose segments are NOT contiguous must be REFUSED, not windowed
    /// into a silently short run. This is the failure the runtime must see at
    /// load time rather than as a model that speaks confident nonsense.
    #[test]
    fn non_monotonic_stream_is_refused() {
        let e = |seg: u16| StreamEnt {
            seg,
            ..Default::default()
        };
        let stream = vec![e(0), e(1), e(0)];
        let err = static_seg_ofs(&stream, &[0], &[3], 2).unwrap_err();
        assert!(err.contains("not monotonic"), "unexpected error: {err}");
    }
}

#[cfg(test)]
mod xreduce_wave_rs_segment_tests {
    use super::*;

    fn program(enabled: bool) -> Program {
        let mut b = Builder::new(8);
        b.deny_uniseg();
        b.set_l2_placement(Some(L2Layout {
            sms: 1,
            domains: 8,
            map: L2Map::RoundRobin,
        }));
        b.set_xreduce_wave_rs_segments(enabled);
        let all = b.all();
        let before = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        let xr = b.emit(DevOp::XReduceTwoShot, all.clone(), &[before], |d| {
            d.i[0] = 8192 * 7168;
            d.i[1] = 8;
        });
        b.emit(DevOp::Nop, all, &[xr], |_| {});
        b.finish()
    }

    #[test]
    fn opt_in_marks_one_pure_xreduce_segment() {
        let p = program(true);
        let xr_seg = p.stream.iter().find(|e| e.inst == 1).unwrap().seg;
        assert!(p
            .stream
            .iter()
            .filter(|e| e.seg == xr_seg)
            .all(|e| e.inst == 1 && e.flags & crate::dev::SE_XR_WAVE_RS != 0));
        assert_eq!(
            p.stream
                .iter()
                .map(|e| e.seg)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
        assert!(p.insts.iter().all(|d| d.wait_len == 0 && d.succ_len == 0));
    }

    #[test]
    fn default_does_not_mark_or_split_xreduce() {
        let p = program(false);
        assert!(p
            .stream
            .iter()
            .all(|e| e.flags & crate::dev::SE_XR_WAVE_RS == 0));
        assert_eq!(
            p.stream
                .iter()
                .map(|e| e.seg)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1
        );
    }
}

#[cfg(test)]
mod decode_grouped_moe_segment_tests {
    use super::*;

    fn program(enabled: bool) -> Program {
        let mut b = Builder::new(4);
        b.deny_uniseg();
        b.set_decode_grouped_moe_segments(enabled);
        let all = b.all();
        let before = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        let glu = b.emit(DevOp::MoeGroupGluFp8Blk, all.clone(), &[before], |d| {
            d.i = [16, 384, 3584, 896, 0, 2, 2, 0];
        });
        let down = b.emit(DevOp::MoeGroupDownFp8Blk, all.clone(), &[glu], |d| {
            d.i = [16, 3584, 384, 896, 0, 0, 2, 0];
        });
        b.emit(DevOp::Nop, all, &[down], |_| {});
        b.finish()
    }

    #[test]
    fn raw_pair_strips_internal_and_external_counters_but_keeps_ordered_segment() {
        let p = program(true);
        let glu_seg = p.stream.iter().find(|e| e.inst == 1).unwrap().seg;
        let down_seg = p.stream.iter().find(|e| e.inst == 2).unwrap().seg;
        assert_eq!(glu_seg, down_seg);
        assert!(p.stream.iter().filter(|e| e.seg == glu_seg).all(|e| {
            matches!(e.inst, 1 | 2)
                && e.wait_len == 0
                && e.succ_len == 0
                && e.flags & crate::dev::SE_XCTR == 0
        }));
        assert_eq!(
            p.stream
                .iter()
                .map(|e| e.seg)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
        assert!(p.insts.iter().all(|d| d.wait_len == 0 && d.succ_len == 0));
    }

    #[test]
    fn grouped_decode_segmentation_is_default_off() {
        let p = program(false);
        assert_eq!(
            p.stream
                .iter()
                .map(|e| e.seg)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1
        );
        assert_ne!(p.insts[1].succ_len, 0);
        assert_ne!(p.insts[2].wait_len, 0);
    }
}

#[cfg(test)]
mod lean_moe_stage2_tests {
    use super::*;

    fn program(enabled: bool, inter_dim: u32) -> Program {
        let mut b = Builder::new(4);
        b.deny_uniseg();
        b.set_lean_moe_stage2_segments(enabled);
        let tensors: Vec<_> = (0..8).map(|i| b.tensor(&format!("moe{i}"), 4096)).collect();
        let all = b.all();
        let before = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        let down = b.emit(DevOp::MoeGroupDownPf, all.clone(), &[before], |d| {
            d.t.copy_from_slice(&tensors);
            d.i = [3584, inter_dim, 896, 2, 0, 0, 0, 0];
        });
        let combine = b.emit(DevOp::MoeCombinePf, all.clone(), &[down], |d| {
            d.t[0] = tensors[0];
            d.t[1] = TENSOR_NONE;
            d.t[2] = TENSOR_NONE;
            d.t[3] = tensors[0];
            d.i = [3584, 16, 1024, 0, 0, 0, 0, 0];
        });
        b.emit(DevOp::Nop, all, &[combine], |_| {});
        b.finish()
    }

    #[test]
    fn eligible_stage2_down_gets_one_pure_segment() {
        let p = program(true, 384);
        let segments_for = |inst: u32| {
            p.stream
                .iter()
                .filter(|e| e.inst == inst)
                .map(|e| e.seg)
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_ne!(segments_for(1), segments_for(2));
        assert_ne!(segments_for(0), segments_for(1));
        assert_eq!(segments_for(2), segments_for(3));
        assert_eq!((p.insts[1].wait_len, p.insts[1].succ_len), (0, 0));
    }

    #[test]
    fn stage2_route_is_opt_in_and_shape_gated() {
        let disabled = program(false, 384);
        assert_eq!(
            disabled
                .stream
                .iter()
                .map(|e| e.seg)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1
        );
        let mut unsupported = Builder::new(1);
        let tensors: Vec<_> = (0..8)
            .map(|i| unsupported.tensor(&format!("moe{i}"), 4096))
            .collect();
        let all = unsupported.all();
        unsupported.emit(DevOp::MoeGroupDownPf, all.clone(), &[], |d| {
            d.t.copy_from_slice(&tensors);
            d.i = [3584, 512, 896, 2, 0, 0, 0, 0];
        });
        unsupported.emit(DevOp::MoeCombinePf, all, &[0], |d| {
            d.t[0] = tensors[0];
            d.t[1] = TENSOR_NONE;
            d.t[2] = TENSOR_NONE;
            d.t[3] = tensors[0];
            d.i = [3584, 16, 1024, 0, 0, 0, 0, 0];
        });
        assert!(!lean_moe_stage2_pair(&unsupported.ops, 0));
    }
}

#[cfg(test)]
mod lean_moe_combine_tests {
    use super::*;

    fn program(enabled: bool, topk: u32, part16: u32) -> Program {
        let mut b = Builder::new(4);
        b.deny_uniseg();
        b.set_lean_moe_combine_segments(enabled);
        let out = b.tensor("out", 4096);
        let residual = b.tensor("residual", 4096);
        let shared = b.tensor("shared", 4096);
        let part = b.tensor("part", 65536);
        let all = b.all();
        let before = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        let combine = b.emit(DevOp::MoeCombinePf, all.clone(), &[before], |d| {
            d.t = [
                out,
                residual,
                shared,
                part,
                TENSOR_NONE,
                TENSOR_NONE,
                TENSOR_NONE,
                TENSOR_NONE,
            ];
            d.i = [2816, topk, 1024, 0, 0, 0, 0, part16];
        });
        b.emit(DevOp::Nop, all, &[combine], |_| {});
        b.finish()
    }

    #[test]
    fn eligible_fixed_order_combine_gets_a_pure_segment() {
        let p = program(true, 16, 0);
        let segs: Vec<_> = p
            .stream
            .iter()
            .filter(|entry| entry.inst == 1)
            .map(|entry| entry.seg)
            .collect();
        assert!(!segs.is_empty());
        assert!(p
            .stream
            .iter()
            .filter(|entry| entry.seg == segs[0])
            .all(|entry| entry.inst == 1));
        assert_eq!((p.insts[1].wait_len, p.insts[1].succ_len), (0, 0));
    }

    #[test]
    fn combine_route_is_opt_in_and_contract_gated() {
        for p in [
            program(false, 16, 0),
            program(true, 8, 0),
            program(true, 16, 1),
        ] {
            assert_eq!(
                p.stream
                    .iter()
                    .map(|entry| entry.seg)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                1
            );
        }
    }
}

#[cfg(test)]
mod lean_kda_intra_tests {
    use super::*;

    fn program(enabled: bool, dim: u32) -> Program {
        let mut b = Builder::new(4);
        b.deny_uniseg();
        b.set_lean_kda_intra_segments(enabled);
        let tensors: Vec<_> = (0..6).map(|i| b.tensor(&format!("kda{i}"), 4096)).collect();
        let all = b.all();
        let before = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        let intra = b.emit(DevOp::KdaChunkIntra, all.clone(), &[before], |d| {
            d.t[..6].copy_from_slice(&tensors);
            d.i = [8192, 12, dim, 0, 0, 0, 0, 0];
            d.f[0] = 1.0 / (128.0f32).sqrt();
        });
        b.emit(DevOp::Nop, all, &[intra], |_| {});
        b.finish()
    }

    #[test]
    fn eligible_intra_gets_one_pure_raw_segment() {
        let p = program(true, 128);
        let seg = |inst| {
            p.stream
                .iter()
                .find(|e| e.inst == inst)
                .expect("instruction has a stream entry")
                .seg
        };
        assert_ne!(seg(0), seg(1));
        assert_ne!(seg(1), seg(2));
        assert_eq!((p.insts[1].wait_len, p.insts[1].succ_len), (0, 0));
    }

    #[test]
    fn intra_segmentation_is_opt_in_and_d128_only() {
        for p in [program(false, 128), program(true, 64)] {
            assert_eq!(
                p.stream
                    .iter()
                    .map(|e| e.seg)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                1
            );
        }
    }

    #[test]
    fn wave_items_marks_only_the_eligible_pure_segment() {
        let mut b = Builder::new(4);
        b.deny_uniseg();
        b.set_kda_intra_wave_items_segments(true);
        let tensors: Vec<_> = (0..6).map(|i| b.tensor(&format!("kda{i}"), 4096)).collect();
        let all = b.all();
        let before = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        let intra = b.emit(DevOp::KdaChunkIntra, all.clone(), &[before], |d| {
            d.t[..6].copy_from_slice(&tensors);
            d.i = [8192, 12, 128, 0, 0, 0, 0, 0];
            d.f[0] = 1.0 / (128.0f32).sqrt();
        });
        b.emit(DevOp::Nop, all, &[intra], |_| {});
        let p = b.finish();
        let marked: Vec<_> = p
            .stream
            .iter()
            .filter(|e| e.flags & crate::dev::SE_KDA_INTRA_WAVE_ITEMS != 0)
            .collect();
        assert!(!marked.is_empty());
        assert!(marked.iter().all(|e| e.inst == 1));
        assert!(p
            .stream
            .iter()
            .filter(|e| e.inst != 1)
            .all(|e| e.flags & crate::dev::SE_KDA_INTRA_WAVE_ITEMS == 0));
    }
}

#[cfg(test)]
mod decode_mla_segment_tests {
    use super::*;

    fn program(enabled: bool) -> Program {
        let mut b = Builder::new(4);
        b.deny_uniseg();
        b.set_decode_mla_segments(enabled);
        let all = b.all();
        let before = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        let flash = b.emit(DevOp::FlashMlaDecode, all.clone(), &[before], |_| {});
        let merge = b.emit(DevOp::MlaMergeFold, all.clone(), &[flash], |_| {});
        b.emit(DevOp::Gemv, all, &[merge], |_| {});
        b.finish()
    }

    #[test]
    fn exact_adjacent_pair_forms_one_pure_segment() {
        let p = program(true);
        let seg = |inst| {
            p.stream
                .iter()
                .find(|e| e.inst == inst)
                .expect("instruction has a stream entry")
                .seg
        };
        assert_ne!(seg(0), seg(1));
        assert_eq!(seg(1), seg(2));
        assert_ne!(seg(2), seg(3));
        assert!(p
            .stream
            .iter()
            .filter(|e| e.seg == seg(1))
            .all(|e| e.inst == 1 || e.inst == 2));
    }

    #[test]
    fn split_is_disabled_unless_the_target_enables_it() {
        assert_eq!(
            program(false)
                .stream
                .iter()
                .map(|e| e.seg)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1
        );
    }
}

#[cfg(test)]
mod lean_kda_key_factor_tests {
    use super::*;

    fn program(enabled: bool, dim: u32, qpre: u32) -> Program {
        let mut b = Builder::new(4);
        b.deny_uniseg();
        b.set_lean_kda_key_factor_segments(enabled);
        let tensors: Vec<_> = (0..12)
            .map(|i| b.tensor(&format!("kda{i}"), 4096))
            .collect();
        let all = b.all();
        let before = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        let wu = b.emit(DevOp::KdaChunkWu, all.clone(), &[before], |d| {
            d.t.copy_from_slice(&tensors[..8]);
            d.i = [8192, 12, dim, dim, qpre, 0, 0, 0];
            d.f[0] = 1.0 / (128.0f32).sqrt();
        });
        b.emit(DevOp::KdaChunkCarry, all.clone(), &[wu], |d| {
            d.t = [
                tensors[8],
                tensors[9],
                tensors[7],
                tensors[3],
                tensors[0],
                tensors[1],
                tensors[10],
                tensors[5],
            ];
            d.i = [8192, 12, dim, dim, qpre, 0, 0, 0];
            d.f[0] = 1.0 / (128.0f32).sqrt();
        });
        b.emit(DevOp::Nop, all, &[2], |_| {});
        b.finish()
    }

    #[test]
    fn eligible_pair_gets_two_pure_ordered_raw_segments() {
        let p = program(true, 128, 1);
        let seg = |inst| {
            p.stream
                .iter()
                .find(|e| e.inst == inst)
                .expect("instruction has a stream entry")
                .seg
        };
        assert_ne!(seg(0), seg(1));
        assert_ne!(seg(1), seg(2));
        assert_ne!(seg(2), seg(3));
        assert_eq!((p.insts[1].wait_len, p.insts[1].succ_len), (0, 0));
        assert_eq!((p.insts[2].wait_len, p.insts[2].succ_len), (0, 0));
    }

    #[test]
    fn segmentation_requires_exact_qpre_d128_pair() {
        for p in [
            program(false, 128, 1),
            program(true, 64, 1),
            program(true, 128, 0),
        ] {
            assert_eq!(
                p.stream
                    .iter()
                    .map(|e| e.seg)
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                1
            );
        }
    }
}

#[cfg(test)]
mod lean_moe_stage1_tests {
    use super::*;

    fn program(enabled: bool, inter_dim: u32) -> Program {
        let mut b = Builder::new(4);
        b.deny_uniseg();
        b.set_lean_moe_stage1_segments(enabled);
        let tensors: Vec<_> = (0..8).map(|i| b.tensor(&format!("moe{i}"), 4096)).collect();
        let all = b.all();
        let before = b.emit(DevOp::Nop, all.clone(), &[], |_| {});
        b.emit(DevOp::MoeGroupGluPf, all.clone(), &[before], |d| {
            d.t.copy_from_slice(&tensors);
            d.i = [inter_dim, 3584, 896, 2, 0, 2, 0, 0];
        });
        b.emit(DevOp::Nop, all, &[1], |_| {});
        b.finish()
    }

    #[test]
    fn eligible_stage1_packet_gets_one_pure_segment() {
        let p = program(true, 384);
        let segment_for = |inst: u32| {
            p.stream
                .iter()
                .find(|e| e.inst == inst)
                .map(|e| e.seg)
                .unwrap()
        };
        assert_ne!(segment_for(0), segment_for(1));
        assert_ne!(segment_for(1), segment_for(2));
    }

    #[test]
    fn stage1_route_is_opt_in_and_shape_gated() {
        let disabled = program(false, 384);
        assert_eq!(
            disabled
                .stream
                .iter()
                .map(|e| e.seg)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1
        );
        let unsupported = program(true, 512);
        assert_eq!(
            unsupported
                .stream
                .iter()
                .map(|e| e.seg)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            1
        );
    }
}

#[cfg(test)]
mod v6_tests {
    use super::*;
    use crate::dev::{DevInst, StreamEnt, Wait};

    fn tiny_model() -> Model {
        let inst = |op: u16| DevInst {
            op,
            blocks: 1,
            wait_len: 0,
            succ_len: 0,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [0; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        let se = |inst: u32, slice: u32| StreamEnt {
            inst,
            slice,
            wait_ofs: 0,
            succ_ofs: 0,
            wait_len: 0,
            succ_len: 0,
            flags: 0,
            seg: 0,
        };
        let prog = || Program {
            n_cu: 2,
            n_counter: 1,
            hier_base: 0,
            insts: vec![inst(6), inst(18)],
            stream: vec![se(0, 0), se(1, 0)],
            stream_ofs: vec![0, 1],
            stream_len: vec![1, 1],
            waits: vec![Wait {
                id: 0,
                threshold: 1,
            }],
            succs: vec![0],
            tensors: Vec::new(),
            gq_stream: vec![se(0, 0), se(1, 0)],
            gq_seg_ofs: vec![0, 2],
            l2_sms: 0,
            l2_domains: 0,
        };
        Model {
            n_cu: 2,
            target: 0,
            tensors: vec![TensorDecl {
                name: "buf".into(),
                bytes: 64,
                init: None,
            }],
            progs: vec![prog()],
            kv_row_insts: vec![],
            prog_t: vec![128],
            gen: Vec::new(),
        }
    }

    #[test]
    fn v6_blob_with_sections_round_trips() {
        let m = tiny_model();
        let cubin_data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04];
        let meta_data = b"{\"model\":\"gemma4\"}".to_vec();
        let sections = vec![
            SectionData {
                kind: SECT_CUBIN,
                name: "interp_sm120".into(),
                data: cubin_data.clone(),
            },
            SectionData {
                kind: SECT_METADATA,
                name: "meta".into(),
                data: meta_data.clone(),
            },
        ];
        let blob = m.to_blob_v6(&sections);

        // Magic is v6
        assert_eq!(&blob[..8], BLOB_MAGIC_V6);

        // sect_dir_offset is in reserved[0] at byte 40
        let dir_off = u64::from_le_bytes(blob[40..48].try_into().unwrap()) as usize;
        assert!(dir_off > 0);
        assert_eq!(&blob[dir_off..dir_off + 4], SECT_MAGIC);

        // n_sections
        let n = u32::from_le_bytes(blob[dir_off + 4..dir_off + 8].try_into().unwrap());
        assert_eq!(n, 2);

        // Read entries
        let ent_size = size_of::<BlobSectionEntry>();
        let e0: BlobSectionEntry = unsafe {
            core::ptr::read_unaligned(blob[dir_off + 8..].as_ptr() as *const BlobSectionEntry)
        };
        let e1: BlobSectionEntry = unsafe {
            core::ptr::read_unaligned(
                blob[dir_off + 8 + ent_size..].as_ptr() as *const BlobSectionEntry
            )
        };
        assert_eq!(e0.kind, SECT_CUBIN);
        assert_eq!(e1.kind, SECT_METADATA);
        assert_eq!(
            &blob[e0.offset as usize..e0.offset as usize + e0.size as usize],
            &cubin_data
        );
        assert_eq!(
            &blob[e1.offset as usize..e1.offset as usize + e1.size as usize],
            &meta_data
        );
    }

    #[test]
    fn v6_empty_sections_still_has_directory() {
        let m = tiny_model();
        let blob = m.to_blob_v6(&[]);
        assert_eq!(&blob[..8], BLOB_MAGIC_V6);
        let dir_off = u64::from_le_bytes(blob[40..48].try_into().unwrap()) as usize;
        assert_eq!(&blob[dir_off..dir_off + 4], SECT_MAGIC);
        let n = u32::from_le_bytes(blob[dir_off + 4..dir_off + 8].try_into().unwrap());
        assert_eq!(n, 0);
    }

    #[test]
    fn v5_blob_unchanged() {
        let m = tiny_model();
        let v5 = m.to_blob();
        assert_eq!(&v5[..8], BLOB_MAGIC);
        // reserved[0] should be 0 in v5
        let r0 = u64::from_le_bytes(v5[40..48].try_into().unwrap());
        assert_eq!(r0, 0);
    }

    #[test]
    fn independent_l2_segment_layout_bumps_container_magic() {
        let mut m = tiny_model();
        m.progs[0].l2_sms = 1;
        m.progs[0].l2_domains = 2;
        assert_eq!(&m.to_blob()[..8], BLOB_MAGIC_L2SEG);
        assert_eq!(&m.to_blob_v6(&[])[..8], BLOB_MAGIC_L2SEG);
    }

    #[test]
    fn multi_bucket_v6_roundtrip() {
        // Simulate a real model: 4 prefill buckets (T=128,512,1024,4096) + decode (T=1).
        let inst = |op: u16| DevInst {
            op,
            blocks: 2,
            wait_len: 0,
            succ_len: 1,
            wait_ofs: 0,
            succ_ofs: 0,
            t: [0; 8],
            i: [0; 8],
            f: [0.0; 2],
            j: [0; 2],
        };
        let se = |inst: u32, slice: u32| StreamEnt {
            inst,
            slice,
            wait_ofs: 0,
            succ_ofs: 0,
            wait_len: 0,
            succ_len: 0,
            flags: 0,
            seg: 0,
        };
        let make_prog = |n_inst: usize| Program {
            n_cu: 4,
            n_counter: n_inst as u32,
            hier_base: 0,
            insts: (0..n_inst).map(|i| inst(8 + i as u16)).collect(),
            stream: (0..n_inst * 2)
                .map(|i| se(i as u32 / 2, i as u32 % 2))
                .collect(),
            stream_ofs: vec![0, n_inst as u32, n_inst as u32 * 2, n_inst as u32 * 2],
            stream_len: vec![n_inst as u32, n_inst as u32, 0, 0],
            waits: (0..n_inst)
                .map(|i| Wait {
                    id: i as u32,
                    threshold: 2,
                })
                .collect(),
            succs: (0..n_inst).map(|i| i as u32).collect(),
            tensors: Vec::new(),
            gq_stream: (0..n_inst * 2)
                .map(|i| se(i as u32 / 2, i as u32 % 2))
                .collect(),
            gq_seg_ofs: vec![0, n_inst as u32 * 2],
            l2_sms: 0,
            l2_domains: 0,
        };

        let bucket_ts: Vec<u32> = vec![128, 512, 1024, 4096, 1]; // last = decode
        let progs: Vec<Program> = vec![
            make_prog(3),  // T=128
            make_prog(5),  // T=512
            make_prog(7),  // T=1024
            make_prog(10), // T=4096
            make_prog(4),  // T=1 (decode)
        ];

        let m = Model {
            n_cu: 4,
            target: 0,
            tensors: vec![
                TensorDecl {
                    name: "model.q".into(),
                    bytes: 1024,
                    init: None,
                },
                TensorDecl {
                    name: "model.k".into(),
                    bytes: 512,
                    init: None,
                },
                TensorDecl {
                    name: "kv.cache".into(),
                    bytes: 8192,
                    init: None,
                },
                TensorDecl {
                    name: "rope.cos".into(),
                    bytes: 16,
                    init: Some(vec![1; 16]),
                },
            ],
            progs,
            kv_row_insts: vec![2, 3],
            prog_t: bucket_ts.clone(),
            gen: Vec::new(),
        };

        // Pack as v6 with an embedded cubin
        let fake_cubin = vec![0xAA; 256];
        let sections = vec![SectionData {
            kind: SECT_CUBIN,
            name: "sm120".into(),
            data: fake_cubin.clone(),
        }];
        let blob = m.to_blob_v6(&sections);

        // Parse header
        assert_eq!(&blob[..8], BLOB_MAGIC_V6);
        let hdr_n_prog = u32::from_le_bytes(blob[16..20].try_into().unwrap());
        assert_eq!(hdr_n_prog, 5, "all 5 programs packed");

        // Verify via DevBlob parser (in plowrt) is tested separately; here we
        // manually verify the section directory and prog_t contract.
        let dir_off = u64::from_le_bytes(blob[40..48].try_into().unwrap()) as usize;
        assert_eq!(&blob[dir_off..dir_off + 4], SECT_MAGIC);
        let n_sect = u32::from_le_bytes(blob[dir_off + 4..dir_off + 8].try_into().unwrap());
        assert_eq!(n_sect, 1);

        // Verify the cubin section data
        let ent: BlobSectionEntry = unsafe {
            core::ptr::read_unaligned(blob[dir_off + 8..].as_ptr() as *const BlobSectionEntry)
        };
        assert_eq!(ent.kind, SECT_CUBIN);
        assert_eq!(ent.size, 256);
        assert_eq!(
            &blob[ent.offset as usize..ent.offset as usize + 256],
            &fake_cubin
        );

        // Byte-equality with v5 payload (minus magic + reserved[0])
        let v5 = m.to_blob();
        // Programs data should be identical between v5 and v6 (same byte offsets
        // for tensors, init, programs)
        let _payload_start = 64usize; // after BlobHeader
        let v5_gq_end = v5.len(); // v5 ends after GQ01
                                  // In v6, sections appear after the GQ01 appendix, then the directory
        assert!(blob.len() > v5_gq_end, "v6 is larger due to sections");
    }

    /// The DECODE BATCH LADDER is separated from the prefill bucket ladder by WIDTH alone
    /// (`decode_rung_lo`), so the separation is worth pinning: a misclassification hands a
    /// prefill bucket to a decode dispatch, which is a wrong answer and not a crash.
    #[test]
    fn decode_rung_lo_separates_the_two_ladders() {
        // No ladder: prefill buckets then ONE decode program. This is every blob emitted
        // before the ladder existed, and it must land on `len - 1`.
        assert_eq!(decode_rung_lo(&[128, 512, 1024, 1]), 3);
        assert_eq!(decode_rung_lo(&[128, 512, 1024, 16]), 3);
        // Decode-only (GLM-5.2 shape).
        assert_eq!(decode_rung_lo(&[1]), 0);
        // A full ladder behind a bucket ladder.
        assert_eq!(decode_rung_lo(&[128, 512, 1024, 1, 2, 4, 8, 16]), 3);
        // A ladder with no prefill at all.
        assert_eq!(decode_rung_lo(&[1, 2, 4, 8, 16]), 0);
        // Non-power-of-two rungs are just as separable — the rule is width, not shape.
        assert_eq!(decode_rung_lo(&[128, 1024, 1, 3, 6, 12]), 2);
        // The widest rung IS the last program: a descending tail is not a ladder, and the
        // scan must not walk into it.
        assert_eq!(decode_rung_lo(&[128, 16, 8]), 2);
        // Width 128 is legal for decode. The equal-width prefill bucket remains outside the
        // trailing ladder because the scan is strict.
        assert_eq!(decode_rung_lo(&[128, 512, 1, 16, 32, 64, 128]), 2);
        assert_eq!(decode_rung_lo(&[128, 128]), 1);
        // A packed-only copy of each prefill rung sits between the ordinary
        // ladder and decode. Its tag cannot be mistaken for a decode width.
        assert_eq!(
            decode_rung_lo(&[
                128,
                1024,
                packed_prefill_program_t(128),
                packed_prefill_program_t(1024),
                1,
                4,
                8,
            ]),
            4
        );
        assert_eq!(program_rows(packed_prefill_program_t(1024)), 1024);
        assert!(is_packed_prefill_program(packed_prefill_program_t(1024)));
    }

    #[test]
    fn xargmax_line_count_is_bounded_and_rounds_up() {
        assert_eq!(xargmax_value_lines(0), Some(1));
        assert_eq!(xargmax_value_lines(1), Some(1));
        assert_eq!(xargmax_value_lines(16), Some(1));
        assert_eq!(xargmax_value_lines(17), Some(2));
        assert_eq!(xargmax_value_lines(128), Some(8));
        assert_eq!(xargmax_value_lines(129), None);
    }
}
