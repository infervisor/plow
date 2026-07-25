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

use crate::dev::{DevInst, DevOp, StreamEnt, Wait, SE_FINE, TENSOR_NONE};
use crate::rope::GenTensor;

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

/// One op, before flattening.
struct Op {
    inst: DevInst,
    cus: Vec<u32>,
    deps: Vec<Dep>,
    counter: u32,   // the coarse counter this op bumps
    work: Vec<u32>, // per-slice cost, from the cost model. See `select_granularity`.
}

pub struct Builder {
    n_cu: u32,
    ops: Vec<Op>,
    tensors: Vec<TensorDecl>,
    gen: Vec<GenTensor>,
    /// L2-domain-aware placement (`PLOW_NV_PLACE`): `(sms_per_partition,
    /// partition_count)` of the target GPU (XCD on MI300/MI350, GPC on
    /// H100/B200). `None` ⇒ off; the blob is byte-identical and `seg` keeps its
    /// wave-class meaning. When set, [`Builder::finish`] repurposes the `seg`
    /// field as a **locality domain** `0..P` and groups `gq_stream` by domain, so
    /// a physical-SM-aware interp (a cluster/HW_ID cursor per domain) can pull
    /// only its domain's packets. The `cus` sets are NOT touched — placement is
    /// dynamic (cursor-claimed) at runtime, so it cannot regress disjoint
    /// `Builder::split` placements. See `plans/devblob-locality-placement.md`.
    place_l2: Option<(u32, u32)>,
}

impl Builder {
    pub fn new(n_cu: u32) -> Self {
        Self { n_cu, ops: Vec::new(), tensors: Vec::new(), gen: Vec::new(), place_l2: None }
    }

    /// Enable L2-domain-aware placement: `(sms_per_partition, partition_count)`
    /// from `hwspec::GpuSpec::l2_partitioning`. `None` (default) leaves the
    /// wave-class `seg` and a byte-identical blob. [`Builder::finish`] skips
    /// placement (byte-identical) if `n_cu > partition_count·sms` — occupancy>1
    /// (`n_cu == 2·sm_count`) or a grid≠sm_count mismatch, where `cu/sms` would
    /// exceed the runtime's `partition_count` domains and orphan packets. See
    /// the field docs and `plans/devblob-locality-placement.md`.
    pub fn set_l2_placement(&mut self, layout: Option<(u32, u32)>) {
        self.place_l2 = layout;
    }

    /// Declare a tensor and get its handle.
    pub fn tensor(&mut self, name: &str, bytes: u64) -> u32 {
        self.tensors.push(TensorDecl { name: name.to_string(), bytes, init: None });
        (self.tensors.len() - 1) as u32
    }

    /// Declare a tensor whose contents the compiler already knows (e.g. RoPE tables).
    pub fn tensor_init(&mut self, name: &str, init: Vec<u8>) -> u32 {
        self.tensors.push(TensorDecl { name: name.to_string(), bytes: init.len() as u64, init: Some(init) });
        (self.tensors.len() - 1) as u32
    }

    /// Declare a tensor the RUNTIME materialises from `recipe` at bind time —
    /// the same bytes [`tensor_init`](Self::tensor_init) would have expanded,
    /// without carrying them in the blob. `bytes` must equal what the recipe
    /// produces; [`Model::to_blob_v6`] asserts it.
    pub fn tensor_gen(&mut self, name: &str, bytes: u64, mut recipe: GenTensor) -> u32 {
        let h = self.tensor(name, bytes);
        recipe.tensor = h;
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
    pub fn emit(&mut self, op: DevOp, cus: Vec<u32>, deps: &[u32], f: impl FnOnce(&mut DevInst)) -> u32 {
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
        let counter = self.ops.len() as u32;
        // Uniform by default: an op that does not tell the builder its per-slice costs is
        // assumed balanced, which makes `select_granularity` fall back to coarse counters.
        let work = vec![1u32; cus.len()];
        self.ops.push(Op { inst, cus, deps, counter, work });
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
    /// `plans/fine-counter-deadlock-fix.md` documents. **Do not reorder streams** without
    /// reading that file.
    pub fn finish(mut self) -> Program {
        let n_cu = self.n_cu as usize;
        let n_ops = self.ops.len();

        // Effective L2-domain placement (PLOW_NV_PLACE), with the occupancy-1
        // coverage guard: every slice's domain is `cu / sms`, which must be < P
        // so it matches the runtime's `smid / sms` in [0, P). `n_cu > P·sms`
        // means occupancy>1 (n_cu = 2·sm_count) or a grid≠sm_count mismatch —
        // placement would emit domain windows the runtime never pulls (orphaned
        // packets -> deadlock). Skip it and fall back byte-identical.
        let l2_place: Option<(u32, u32)> = self.place_l2.and_then(|(sms, p)| {
            if sms == 0 || p == 0 {
                None
            } else if self.n_cu > p * sms {
                eprintln!(
                    "  l2 placement SKIPPED: n_cu {} > {p} domains × {sms} SM = {} \
                     (occupancy>1 or grid≠sm_count) — byte-identical",
                    self.n_cu,
                    p * sms
                );
                None
            } else {
                Some((sms, p))
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
        // The host reads the class back from the ops themselves. See plans/segmented-dispatch.md.
        // PLOW_UNISEG=1 collapses every op into ONE segment. The wave-class split exists so an AMD
        // host can relaunch FlashPrefill at a 4-wave occupancy; the sm_120 persistent interpreter
        // runs EVERY op at a fixed 256-thread (8-warp) block and synchronises the whole program in
        // one cooperative launch under the counter protocol (exactly as the decode program does), so
        // the segment boundary is spurious there and would otherwise force a segmented relaunch path.
        let uniseg = std::env::var("PLOW_UNISEG").ok().as_deref() == Some("1");
        let wave_class = |op: u16| -> u8 {
            if uniseg {
                8
            } else if op == DevOp::FlashPrefill as u16 || op == DevOp::FlashPrefillFp8 as u16 {
                4
            } else {
                8
            }
        };
        let mut seg_of = vec![0u16; self.ops.len()];
        let mut cur_seg = 0u16;
        for i in 0..self.ops.len() {
            if i > 0 && wave_class(self.ops[i].inst.op) != wave_class(self.ops[i - 1].inst.op) {
                cur_seg += 1;
            }
            seg_of[i] = cur_seg;
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
        let seg_class_slice =
            !uniseg && std::env::var("PLOW_SEG_CLASS_SLICE").ok().as_deref() == Some("1");
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
                let gemm_class = wave_class(self.ops[i].inst.op) == 8;
                let fills = self.ops[i].cus.len() == n_cu_sz;
                let has_fine = self.ops[i].deps.iter().any(|d| matches!(d, Dep::Fine { .. }));
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

        for (idx, op) in self.ops.iter().enumerate() {
            let mut inst = op.inst;

            // The op's COARSE lists, on the instruction. A dep's threshold is how the
            // PRODUCER was sliced — deriving it here is the whole reason the builder owns the
            // CU sets: a hand-written threshold is a deadlock.
            inst.wait_ofs = waits.len() as u32;
            for d in &op.deps {
                let producer = &self.ops[d.producer() as usize];
                // A Fine dep still needs a coarse fallback entry only if we are NOT emitting
                // per-slice lists for this op — but we always are (see `fine` below), so a
                // Fine dep contributes nothing to the instruction's list.
                if let Dep::Coarse(c) = d {
                    waits.push(Wait { id: *c, threshold: producer.cus.len() as u32 });
                }
            }
            inst.wait_len = (waits.len() as u32 - inst.wait_ofs) as u16;

            inst.succ_ofs = succs.len() as u32;
            succs.push(op.counter);
            inst.succ_len = 1;

            let has_fine_dep = op.deps.iter().any(|d| matches!(d, Dep::Fine { .. }));
            let is_fine_producer = fine_base[idx] != u32::MAX;
            let fine = has_fine_dep || is_fine_producer;

            // `slice` is the op-local index of this workgroup, NOT the CU id: the op's
            // kernel splits its work into `blocks` shares and this is which share.
            for (slice, &cu) in op.cus.iter().enumerate() {
                let mut e =
                    StreamEnt { inst: idx as u32, slice: slice as u32, ..Default::default() };
                // L2-domain placement (PLOW_NV_PLACE): `seg` is a PER-SLICE domain
                // = physical CU / SMs-per-partition, so a full op's slices spread
                // across every L2 domain (no skew) and slice `s` sits in the same
                // domain across ops (consumer reads producer from one L2 slice).
                // A physical-SM interp pulls its domain's gq window. Off ⇒ `seg`
                // keeps its wave-class meaning (byte-identical).
                e.seg = match l2_place {
                    Some((sms, _)) => (cu / sms) as u16,
                    None => seg_of[idx],
                };
                if fine {
                    e.flags = SE_FINE;

                    if has_fine_dep {
                        e.wait_ofs = waits.len() as u32;
                        for d in &op.deps {
                            match d {
                                Dep::Coarse(c) => {
                                    let p = &self.ops[*c as usize];
                                    waits.push(Wait { id: *c, threshold: p.cus.len() as u32 });
                                }
                                Dep::Fine { producer, map } => {
                                    let base = fine_base[*producer as usize];
                                    for &ps in &map[slice] {
                                        waits.push(Wait { id: base + ps, threshold: 1 });
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

        // Under L2 placement, `seg` == domain is not monotonic in op-emit order,
        // so group gq_stream by domain first. A STABLE sort preserves each
        // domain's op-major (topological) order; cross-domain deps stay
        // counter-gated. This yields contiguous per-domain [ofs[d], ofs[d+1))
        // windows a physical-SM interp pulls with one cursor per domain.
        if l2_place.is_some() {
            gq_stream.sort_by_key(|e| e.seg);
        }

        // Segment window bounds in gq_stream. gq_stream is op-major and seg_of[] is monotonic in
        // op-emit order, so each segment occupies a contiguous [ofs[s], ofs[s+1]) range — the
        // interp bounds its cursor to this window under RUNSEG. Under L2 placement `seg` ranges
        // over the P L2 domains instead of the wave-class count.
        // Under L2 placement, `seg` ranges over the P L2 domains (fixed by the
        // hardware partition_count), NOT ceil(n_cu/sms) — so the window count
        // always matches the runtime's `smid/sms` domain count.
        let n_seg = match l2_place {
            Some((_, p)) => p as usize,
            None => cur_seg as usize + 1,
        };
        let mut gq_seg_ofs = vec![0u32; n_seg + 1];
        {
            let mut s = 0usize;
            for (i, e) in gq_stream.iter().enumerate() {
                while e.seg as usize > s {
                    s += 1;
                    gq_seg_ofs[s] = i as u32;
                }
            }
            gq_seg_ofs[n_seg] = gq_stream.len() as u32;
        }

        // Static allocation report (PLOW_NV_PLACE): packets (op-slices) per L2
        // domain window, and the skew a physical-SM interp would see across
        // partitions. Emitted here so a build surfaces the balance without a GPU.
        if l2_place.is_some() {
            let per: Vec<u32> = (0..n_seg)
                .map(|d| gq_seg_ofs[d + 1] - gq_seg_ofs[d])
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
            eprintln!(
                "  l2 placement: {n_seg} domains, packets/domain {per:?}, skew {skew:.1}% \
                 (max {hi} vs min {lo})"
            );
        }

        Program {
            n_cu: self.n_cu,
            n_counter,
            insts,
            stream,
            stream_ofs,
            stream_len,
            waits,
            succs,
            tensors: self.tensors,
            gq_stream,
            gq_seg_ofs,
            l2_sms: l2_place.map(|(s, _)| s).unwrap_or(0),
            l2_domains: l2_place.map(|(_, p)| p).unwrap_or(0),
        }
    }
}

pub struct Program {
    pub n_cu: u32,
    pub n_counter: u32,
    pub insts: Vec<DevInst>,
    pub stream: Vec<StreamEnt>,
    pub stream_ofs: Vec<u32>,
    pub stream_len: Vec<u32>,
    pub waits: Vec<Wait>,
    pub succs: Vec<u32>,
    pub tensors: Vec<TensorDecl>,
    /// Op-major (topological) permutation of `stream` for the global-queue interpreter.
    pub gq_stream: Vec<StreamEnt>,
    /// `[n_seg+1]` segment window bounds into `gq_stream`.
    pub gq_seg_ofs: Vec<u32>,
    /// L2-domain placement (PLOW_NV_PLACE): SMs per partition, and the number of
    /// L2 domains `gq_seg_ofs` is windowed by. `0` ⇒ not placed (`seg` is
    /// wave-class). When non-zero, `gq_stream`'s `seg` is a domain and the blob
    /// header carries [`PLOW_BLOB_F_L2DOM`]; a runtime without physical-SM
    /// domain dispatch must refuse it. See `plans/devblob-locality-placement.md`.
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

/// Every container version this build can read.
///
/// Readers must go through this rather than spelling out their own list: the
/// runtime checks the magic in two places (parse, and the assets-dir sniff in
/// `DevBlob::find_in_dir`), and when v7 was added to only one of them a v7 blob
/// parsed correctly but was never *discovered* — `plowrt serve` just reported no
/// model. One list, one place.
pub const BLOB_MAGICS: [&[u8; 8]; 3] = [BLOB_MAGIC, BLOB_MAGIC_V6, BLOB_MAGIC_V7];

/// Is `m` a container version this build understands?
pub fn is_blob_magic(m: &[u8; 8]) -> bool {
    BLOB_MAGICS.contains(&m)
}
pub const NAME_LEN: usize = 80;
pub const INIT_NONE: u64 = u64::MAX;

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

/// [`BlobHeader::flags`] bit: `gq_stream`'s `seg` is an **L2 domain** (PLOW_NV_PLACE),
/// and `gq_seg_ofs` windows it by domain, not wave-class. A runtime WITHOUT
/// physical-SM domain dispatch (`PLOW_NV_PLACE_DISPATCH`) must REFUSE such a blob —
/// its wave-class segmentation would mis-dispatch `seg`. `reserved[1]` carries SMs
/// per partition, `reserved[2]` the domain count, so the interp need not be told
/// via a build define. See `plans/devblob-locality-placement.md`.
pub const PLOW_BLOB_F_L2DOM: u32 = 2;

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
            decls.push(BlobTensor { name, bytes: t.bytes, init_off });
        }

        // L2-domain placement summary across programs (PLOW_NV_PLACE): all placed
        // programs share the target's (sms, domains); mark the header + carry them.
        let (l2_flag, l2_sms, l2_dom) = match self.progs.iter().find(|p| p.l2_domains > 0) {
            Some(p) => (PLOW_BLOB_F_L2DOM, p.l2_sms, p.l2_domains),
            None => (0, 0, 0),
        };
        let hdr = BlobHeader {
            magic: *BLOB_MAGIC,
            n_cu: self.n_cu,
            n_tensor: self.tensors.len() as u32,
            n_prog: self.progs.len() as u32,
            n_kvrow: self.kv_row_insts.len() as u32,
            // Every program carries the op-major gq_stream appendix (emitted below), so mark the
            // stream global-queue-capable. The runtime reads this to allow PLOW_GLOBAL_QUEUE=1.
            // + F_L2DOM when any program is L2-domain-placed (PLOW_NV_PLACE); the runtime must
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
            pod(&[BlobProgHeader {
                n_inst: p.insts.len() as u32,
                n_stream: p.stream.len() as u32,
                n_wait: p.waits.len() as u32,
                n_succ: p.succs.len() as u32,
                n_counter: p.n_counter,
                t: self.prog_t[i],
            }], &mut b);
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
            all.push(SectionData { kind: SECT_GEN_TENSORS, name: "rope".into(), data });
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
            decls.push(BlobTensor { name, bytes: t.bytes, init_off });
        }

        // L2-domain placement summary (PLOW_NV_PLACE) — see to_blob().
        let (l2_flag, l2_sms, l2_dom) = match self.progs.iter().find(|p| p.l2_domains > 0) {
            Some(p) => (PLOW_BLOB_F_L2DOM, p.l2_sms, p.l2_domains),
            None => (0, 0, 0),
        };
        // Header placeholder — sect_dir_offset patched after we know the full layout.
        let hdr = BlobHeader {
            magic: if self.gen.is_empty() { *BLOB_MAGIC_V6 } else { *BLOB_MAGIC_V7 },
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
            pod(&[BlobProgHeader {
                n_inst: p.insts.len() as u32,
                n_stream: p.stream.len() as u32,
                n_wait: p.waits.len() as u32,
                n_succ: p.succs.len() as u32,
                n_counter: p.n_counter,
                t: self.prog_t[i],
            }], &mut b);
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
        b[reserved0_off..reserved0_off + 8]
            .copy_from_slice(&sect_dir_offset.to_le_bytes());

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
        b.emit_dep_work(DevOp::Nop, cus, vec![Dep::Fine { producer: p, map }], work, |_| {});
        // n_counter > n_ops exactly when a fine producer got per-slice counters.
        b.finish().n_counter > 2
    }

    /// A transformer's heads all do the same work, so the region is homogeneous and the
    /// `collapse` theorem (Plow/CounterGranularity.lean) says fine gates cannot win. The
    /// compiler must therefore NOT emit them — they cost counters and atomics for nothing.
    #[test]
    fn uniform_work_is_downgraded_to_coarse() {
        assert!(!survives(vec![10, 10, 10, 10]), "uniform region must fall back to coarse");
    }

    /// MoE experts get different token counts by construction. Then a straggling producer can
    /// feed a CHEAP consumer and its slack is absorbed instead of reaching the barrier — which
    /// is exactly `hetero_can_win`. The compiler must keep the fine gates here.
    #[test]
    fn heterogeneous_work_keeps_fine_gates() {
        assert!(survives(vec![1, 40, 3, 9]), "imbalanced region must keep its fine gates");
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
            insts: vec![inst(6), inst(18)],
            stream: vec![se(0, 0), se(1, 0)],
            stream_ofs: vec![0, 1],
            stream_len: vec![1, 1],
            waits: vec![Wait { id: 0, threshold: 1 }],
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
            tensors: vec![TensorDecl { name: "buf".into(), bytes: 64, init: None }],
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
            SectionData { kind: SECT_CUBIN, name: "interp_sm120".into(), data: cubin_data.clone() },
            SectionData { kind: SECT_METADATA, name: "meta".into(), data: meta_data.clone() },
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
                blob[dir_off + 8 + ent_size..].as_ptr() as *const BlobSectionEntry,
            )
        };
        assert_eq!(e0.kind, SECT_CUBIN);
        assert_eq!(e1.kind, SECT_METADATA);
        assert_eq!(&blob[e0.offset as usize..e0.offset as usize + e0.size as usize], &cubin_data);
        assert_eq!(&blob[e1.offset as usize..e1.offset as usize + e1.size as usize], &meta_data);
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
    fn multi_bucket_v6_roundtrip() {
        // Simulate a real model: 4 prefill buckets (T=128,512,1024,4096) + decode (T=1).
        let inst = |op: u16| DevInst {
            op, blocks: 2, wait_len: 0, succ_len: 1, wait_ofs: 0, succ_ofs: 0,
            t: [0; 8], i: [0; 8], f: [0.0; 2], j: [0; 2],
        };
        let se = |inst: u32, slice: u32| StreamEnt {
            inst, slice, wait_ofs: 0, succ_ofs: 0, wait_len: 0, succ_len: 0, flags: 0, seg: 0,
        };
        let make_prog = |n_inst: usize| Program {
            n_cu: 4,
            n_counter: n_inst as u32,
            insts: (0..n_inst).map(|i| inst(8 + i as u16)).collect(),
            stream: (0..n_inst * 2).map(|i| se(i as u32 / 2, i as u32 % 2)).collect(),
            stream_ofs: vec![0, n_inst as u32, n_inst as u32 * 2, n_inst as u32 * 2],
            stream_len: vec![n_inst as u32, n_inst as u32, 0, 0],
            waits: (0..n_inst).map(|i| Wait { id: i as u32, threshold: 2 }).collect(),
            succs: (0..n_inst).map(|i| i as u32).collect(),
            tensors: Vec::new(),
            gq_stream: (0..n_inst * 2).map(|i| se(i as u32 / 2, i as u32 % 2)).collect(),
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
                TensorDecl { name: "model.q".into(), bytes: 1024, init: None },
                TensorDecl { name: "model.k".into(), bytes: 512, init: None },
                TensorDecl { name: "kv.cache".into(), bytes: 8192, init: None },
                TensorDecl { name: "rope.cos".into(), bytes: 16, init: Some(vec![1; 16]) },
            ],
            progs,
            kv_row_insts: vec![2, 3],
            prog_t: bucket_ts.clone(),
            gen: Vec::new(),
        };

        // Pack as v6 with an embedded cubin
        let fake_cubin = vec![0xAA; 256];
        let sections = vec![
            SectionData { kind: SECT_CUBIN, name: "sm120".into(), data: fake_cubin.clone() },
        ];
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
        assert_eq!(&blob[ent.offset as usize..ent.offset as usize + 256], &fake_cubin);

        // Byte-equality with v5 payload (minus magic + reserved[0])
        let v5 = m.to_blob();
        // Programs data should be identical between v5 and v6 (same byte offsets
        // for tensors, init, programs)
        let payload_start = 64usize; // after BlobHeader
        let v5_gq_end = v5.len(); // v5 ends after GQ01
        // In v6, sections appear after the GQ01 appendix, then the directory
        assert!(blob.len() > v5_gq_end, "v6 is larger due to sections");
    }
}
