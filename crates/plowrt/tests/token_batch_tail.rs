//! §9 "Head", on the CPU: the compact terminal segment, run on the real CPU engine.
//!
//! ```text
//! body final residual X[M,H]
//!     -> RowGather(sample_input_rows[S])   -> selected_x[S,H]
//!     -> final RMSNorm                     -> selected_norm[S,H]
//!     -> LM head with M=S                  -> logits[S,V]
//!     -> Argmax -> ArgmaxFin               -> sampled_ids[S]
//! ```
//!
//! Built as a real `LoadedProgram` and run through `exec::cpu::interp` on the persistent worker
//! pool with the counter DAG, not by calling kernels in a loop — the point of doing this on CPU
//! first is that it makes the row-selection contract testable with no hardware, and a test that
//! bypasses the interpreter would not have tested the thing that will later run on a GPU.
//!
//! Two comparisons, both from §9:
//!
//! 1. **Isolated vs packed.** The same selected rows, run one at a time at `S = 1`, must give
//!    the same ids as the one compact `S`-row run. This is the mapping check, and it is the one
//!    that catches a gather that reads the right number of rows from the wrong places.
//! 2. **A host oracle.** An independent Rust implementation of the same chain over the same
//!    bf16 rounding. This is the value check.
//!
//! Plus the refusals the plan requires here: S = 0 must not run the selection stage at all
//! (legacy argmax reads zero as one), and a missing `RowGather` kernel must be refused BY NAME
//! before launch rather than dispatched into a table hole.
#![cfg(feature = "cpu")]

use std::ffi::c_void;
use std::sync::Arc;

use packet::dev::{DevInst64, DevOp, StreamEnt, Wait, TENSOR_NONE16};
use plow_asset::token_batch::{self, Phase, Request, Selection};
use plowrt::exec::counters::CounterPool;
use plowrt::exec::cpu::ffi::{self, KernelTable, PlowCpuCtx};
use plowrt::exec::cpu::interp::{Exec, LoadedProgram, WorkerCtx};
use plowrt::exec::cpu::topology::{NumaMode, Topology};
use plowrt::exec::cpu::workers::WorkerPool;

const H: usize = 64;
const VOCAB: usize = 96;

fn f2bf(f: f32) -> u16 {
    let u = f.to_bits();
    if (u & 0x7F80_0000) == 0x7F80_0000 {
        return ((u >> 16) | u32::from(u & 0xFFFF != 0) * 0x40) as u16;
    }
    let lsb = (u >> 16) & 1;
    ((u.wrapping_add(0x7FFF + lsb)) >> 16) as u16
}

fn bf2f(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// Deterministic, well-spread values — a body output whose rows differ enough that a gather
/// that picks the wrong row cannot coincidentally agree.
fn hidden(rows: usize) -> Vec<u16> {
    (0..rows * H)
        .map(|i| {
            let r = (i / H) as f32;
            let c = (i % H) as f32;
            f2bf(((r * 0.37 + c * 0.11).sin() + 0.25 * (r - c * 0.5).cos()) * 0.5)
        })
        .collect()
}

fn weights(n: usize, k: usize, seed: f32) -> Vec<u16> {
    (0..n * k)
        .map(|i| {
            let a = (i / k) as f32;
            let b = (i % k) as f32;
            f2bf(((a * 0.19 + b * 0.07 + seed).sin()) * 0.3)
        })
        .collect()
}

/// The host oracle: the same chain, in f32 with the same bf16 rounding points as the kernels.
fn oracle(x: &[u16], rows: &[u32], gamma: &[u16], head: &[u16], eps: f32) -> Vec<u32> {
    rows.iter()
        .map(|&row| {
            let src = &x[row as usize * H..][..H];
            let ss: f32 = src.iter().map(|&v| bf2f(v) * bf2f(v)).sum();
            let inv = 1.0 / (ss / H as f32 + eps).sqrt();
            let normed: Vec<u16> = (0..H)
                .map(|i| f2bf(bf2f(src[i]) * inv * bf2f(gamma[i])))
                .collect();
            let mut best = (f32::NEG_INFINITY, 0u32);
            for n in 0..VOCAB {
                let w = &head[n * H..][..H];
                let mut acc = 0.0f32;
                for k in 0..H {
                    acc += bf2f(normed[k]) * bf2f(w[k]);
                }
                // The device rounds the logit to bf16 before the argmax reads it, and the
                // greedy tie rule is LOWEST index. Both matter: comparing f32 accumulators
                // instead would disagree with the kernel on a tie the model can actually hit.
                let logit = bf2f(f2bf(acc));
                if logit > best.0 {
                    best = (logit, n as u32);
                }
            }
            best.1
        })
        .collect()
}

/// Host tensors, kept alive for the run; `table` is the pointer array kernels index.
struct Tensors {
    storage: Vec<Vec<u8>>,
    table: Vec<*mut c_void>,
}

impl Tensors {
    fn new() -> Self {
        Tensors {
            storage: Vec::new(),
            table: Vec::new(),
        }
    }
    fn push_bytes(&mut self, bytes: Vec<u8>) -> u16 {
        let handle = self.table.len() as u16;
        self.storage.push(bytes);
        self.table
            .push(self.storage.last_mut().unwrap().as_mut_ptr() as *mut c_void);
        handle
    }
    fn push_u16(&mut self, v: &[u16]) -> u16 {
        self.push_bytes(v.iter().flat_map(|x| x.to_ne_bytes()).collect())
    }
    fn push_u32(&mut self, v: &[u32]) -> u16 {
        self.push_bytes(v.iter().flat_map(|x| x.to_ne_bytes()).collect())
    }
    fn zeros(&mut self, bytes: usize) -> u16 {
        self.push_bytes(vec![0u8; bytes])
    }
    fn read_u32(&self, handle: u16, n: usize) -> Vec<u32> {
        self.storage[handle as usize][..n * 4]
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes(c.try_into().unwrap()))
            .collect()
    }
}

/// [`Exec`] over the real C kernel table, with the tensor pointers this test owns.
struct TailExec {
    table: KernelTable,
    tensors: *const *mut c_void,
    ctx: std::sync::Mutex<Vec<PlowCpuCtx>>,
}

// SAFETY: the pointer table outlives the pool (the test drops the pool first), and each worker
// touches only its own context, serialized here by the mutex because this test's `nblk` is 1.
unsafe impl Send for TailExec {}
unsafe impl Sync for TailExec {}

impl Exec for TailExec {
    fn exec(&self, inst: &DevInst64, slice: u32, nblk: u32, w: &WorkerCtx) {
        let mut ctxs = self.ctx.lock().unwrap();
        let ctx = &mut ctxs[w.worker as usize];
        let f = self
            .table
            .get(inst.op)
            .expect("kernel resolved at load, so this cannot be a table hole");
        // SAFETY: every handle the program names is a live tensor sized for the op's extent,
        // and `ctx` went through `thread_init` below.
        unsafe { f(inst, slice, nblk, self.tensors, ctx) };
    }
}

struct Op {
    inst: DevInst64,
    deps: Vec<usize>,
}

/// One instruction, one slice, gated on its producers — the terminal segment is a short serial
/// chain, so the DAG is the interesting part, not the parallelism.
fn program(ops: &[Op]) -> (LoadedProgram, Vec<packet::Counter>) {
    let mut insts = Vec::new();
    let mut waits = Vec::new();
    let mut succs = Vec::new();
    let mut stream = Vec::new();
    for (k, op) in ops.iter().enumerate() {
        let mut inst = op.inst;
        inst.blocks = 1;
        insts.push(inst);
        let wait_ofs = waits.len() as u32;
        for &d in &op.deps {
            waits.push(Wait {
                id: d as u32,
                threshold: 1,
            });
        }
        let succ_ofs = succs.len() as u32;
        succs.push(k as u32);
        stream.push(StreamEnt {
            inst: k as u32,
            slice: 0,
            wait_ofs,
            succ_ofs,
            wait_len: op.deps.len() as u16,
            succ_len: 1,
            flags: 0,
            seg: 0,
        });
    }
    let counters = (0..ops.len())
        .map(|k| packet::Counter {
            id: k as u32,
            threshold: 1,
            scope: 1,
            _pad: [0; 3],
        })
        .collect();
    (
        LoadedProgram {
            insts,
            stream,
            stream_ofs: vec![0],
            stream_len: vec![ops.len() as u32],
            waits,
            succs,
            n_cu: 1,
            n_seg: 1,
            seg_ofs: None,
            gq: None,
            cus_of: None,
        },
        counters,
    )
}

fn inst(op: DevOp, t: [u16; 8], i: [u32; 8], f0: f32) -> DevInst64 {
    let mut d = DevInst64 {
        op: op as u16,
        blocks: 1,
        t,
        i,
        ..DevInst64::default()
    };
    // `fj` is the raw overlay: fj[0] holds f[0]'s BIT PATTERN, not a float member.
    d.fj[0] = f0.to_bits();
    d
}

const NONE: u16 = TENSOR_NONE16;

/// Build and run the terminal segment for `sample_rows` over the body hidden `x`.
fn run_tail(x: &[u16], m: u32, sample_rows: &[u32], gamma: &[u16], head: &[u16]) -> Vec<u32> {
    let s = sample_rows.len() as u32;
    assert!(s > 0, "S = 0 must not reach the selection stage at all");
    let mut t = Tensors::new();
    let h_x = t.push_u16(x);
    let h_rows = t.push_u32(sample_rows);
    let h_sel = t.zeros(s as usize * H * 2);
    let h_norm = t.zeros(s as usize * H * 2);
    let h_gamma = t.push_u16(gamma);
    let h_head = t.push_u16(head);
    let h_logits = t.zeros(s as usize * VOCAB * 2);
    let h_part = t.zeros(s as usize * 8);
    let h_ids = t.zeros(s as usize * 4);

    let ops = vec![
        // RowGather: t0=out t1=x t2=rows, i0=S i1=H i2=M.
        Op {
            inst: inst(
                DevOp::RowGather,
                [h_sel, h_x, h_rows, NONE, NONE, NONE, NONE, NONE],
                [s, H as u32, m, 0, 0, 0, 0, 0],
                0.0,
            ),
            deps: vec![],
        },
        // The model's EXISTING final RMSNorm, at M = S. Gathering before it is valid because
        // final RMSNorm is per token, and it avoids normalizing rows nobody asked for.
        Op {
            inst: inst(
                DevOp::RmsNorm,
                [h_norm, h_sel, h_gamma, NONE, NONE, NONE, NONE, NONE],
                [s, H as u32, 0, 0, 0, 0, 0, 0],
                1e-6,
            ),
            deps: vec![0],
        },
        // LM head, selected by S rather than M: i0=M i1=N i2=K.
        Op {
            inst: inst(
                DevOp::Gemm,
                [h_logits, h_norm, h_head, NONE, NONE, NONE, NONE, NONE],
                [s, VOCAB as u32, H as u32, 0, 0, 0, 0, 0],
                0.0,
            ),
            deps: vec![1],
        },
        // Selection stage: replicated greedy, the AMD single-GPU / NVIDIA / CPU shape.
        Op {
            inst: inst(
                DevOp::Argmax,
                [h_part, h_logits, NONE, NONE, NONE, NONE, NONE, NONE],
                [VOCAB as u32, s, 0, 0, 0, 0, 0, 0],
                0.0,
            ),
            deps: vec![2],
        },
        Op {
            inst: inst(
                DevOp::ArgmaxFin,
                [h_ids, h_part, NONE, NONE, NONE, NONE, NONE, NONE],
                [1, s, 0, 0, 0, 0, 0, 0],
                0.0,
            ),
            deps: vec![3],
        },
    ];

    let (prog, counters) = program(&ops);
    // A missing arm must be refused BY NAME here, before launch — never dispatched into a
    // table hole. `KernelTable::resolve` is the CPU's `plow_cpu_has` probe.
    let table = KernelTable::resolve(prog.insts.iter().map(|d| d.op)).unwrap_or_else(|missing| {
        panic!(
            "no CPU kernel for {:?}",
            missing
                .iter()
                .map(|&op| DevOp::from_u16(op).map_or("?", |o| o.c_name()))
                .collect::<Vec<_>>()
        )
    });

    let scratch = ffi::scratch_bytes().max(64) as usize;
    let mut scratch_buf = vec![0u8; scratch];
    let mut ctx = PlowCpuCtx::new(0, 0);
    ctx.scratch = scratch_buf.as_mut_ptr() as *mut c_void;
    ctx.scratch_bytes = scratch as u32;
    ffi::thread_init(&mut ctx).unwrap();

    let exec = Arc::new(TailExec {
        table,
        tensors: t.table.as_ptr(),
        ctx: std::sync::Mutex::new(vec![ctx]),
    });
    let topo = Topology::detect();
    let pool = WorkerPool::spawn(&topo, 1, &NumaMode::Off, 20, 1, None, exec);
    let cp = Arc::new(CounterPool::from_counters(&counters));
    let prog = Arc::new(prog);
    cp.reset_all();
    let gen = pool.run(&prog, 0, &cp);
    assert_eq!(pool.wait_done(gen), None, "terminal segment faulted");
    drop(pool);
    t.read_u32(h_ids, s as usize)
}

/// The plan's §6.1 acceptance case, end to end on the CPU engine: the compact tail selects
/// exactly the rows the planner named, and produces the ids the host oracle does.
#[test]
fn the_compact_tail_matches_the_host_oracle_and_the_isolated_route() {
    ffi::init(ffi::Isa::Scalar).unwrap();
    let decode_a = [7u32];
    let decode_b = [9u32];
    let finishing: Vec<u32> = (0..50).collect();
    let intermediate: Vec<u32> = (0..80).collect();
    let sel = Selection::default();
    let requests = [
        Request {
            id: 10,
            slot: 0,
            state_slot: 0,
            generation: 1,
            phase: Phase::Decode,
            tokens: &decode_a,
            prompt_len: 100,
            selection: sel,
        },
        Request {
            id: 11,
            slot: 1,
            state_slot: 1,
            generation: 1,
            phase: Phase::Decode,
            tokens: &decode_b,
            prompt_len: 900,
            selection: sel,
        },
        Request {
            id: 12,
            slot: 2,
            state_slot: 2,
            generation: 1,
            phase: Phase::Prefill,
            tokens: &finishing,
            prompt_len: 120,
            selection: sel,
        },
        Request {
            id: 13,
            slot: 3,
            state_slot: 3,
            generation: 1,
            phase: Phase::Prefill,
            tokens: &intermediate,
            prompt_len: 4096,
            selection: sel,
        },
    ];
    let frontiers = [100, 900, 70, 0];
    let generations = [1, 1, 1, 1];
    let plan = token_batch::plan(&requests, &frontiers, &generations, 132, 8192, 0).unwrap();
    assert_eq!((plan.real_rows, plan.sample_rows), (132, 3));
    assert_eq!(plan.sample_input_rows, vec![0, 1, 51]);

    let x = hidden(plan.real_rows as usize);
    let gamma: Vec<u16> = (0..H).map(|i| f2bf(0.8 + 0.01 * i as f32)).collect();
    let head = weights(VOCAB, H, 1.5);

    let packed = run_tail(&x, plan.real_rows, &plan.sample_input_rows, &gamma, &head);
    assert_eq!(
        packed,
        oracle(&x, &plan.sample_input_rows, &gamma, &head, 1e-6),
        "compact tail disagrees with the host oracle",
    );

    // Isolated vs packed: the same rows, one at a time. This is what catches a gather that
    // reads the right COUNT of rows from the wrong PLACES — a bug the oracle comparison alone
    // would also catch, but only because the oracle is independent; running the same route at
    // S = 1 proves the mapping without relying on that.
    for (index, &row) in plan.sample_input_rows.iter().enumerate() {
        let one = run_tail(&x, plan.real_rows, &[row], &gamma, &head);
        assert_eq!(
            one,
            vec![packed[index]],
            "row {row} differs isolated vs packed"
        );
    }

    // The owners are the LOGICAL requests, in sample order — delivery must not depend on the
    // row number, because packed rows, physical slots and compact output rows are three
    // different numberings.
    assert_eq!(
        plan.sample_owners
            .iter()
            .map(|o| o.request)
            .collect::<Vec<_>>(),
        vec![10, 11, 12],
    );
}

/// Arbitrary row indices, including a reversed and a repeated selection, so the gather cannot
/// be passing by accident of monotone order. (A repeated index is not something the planner
/// emits — one sample row per request — but the OPCODE must still be exact, because a later
/// mode that relaxes §4.4 would be relying on it.)
#[test]
fn the_gather_honours_arbitrary_row_indices() {
    ffi::init(ffi::Isa::Scalar).unwrap();
    let m = 40u32;
    let x = hidden(m as usize);
    let gamma: Vec<u16> = (0..H).map(|i| f2bf(1.0 - 0.003 * i as f32)).collect();
    let head = weights(VOCAB, H, -0.7);
    for rows in [
        vec![0u32],
        vec![39],
        vec![39, 0, 17],
        vec![5, 5, 5],
        (0..8).map(|i| i * 5).collect::<Vec<u32>>(),
    ] {
        let got = run_tail(&x, m, &rows, &gamma, &head);
        assert_eq!(got, oracle(&x, &rows, &gamma, &head, 1e-6), "rows {rows:?}");
    }
}

/// S = 0 submits the body only. The selection stage is not run with zero rows — legacy argmax
/// reads zero as one and would deliver a token for a request that asked for none.
#[test]
fn s_zero_submits_no_terminal_segment() {
    ffi::init(ffi::Isa::Scalar).unwrap();
    let chunk: Vec<u32> = (0..16).collect();
    let sel = Selection::default();
    let requests = [Request {
        id: 1,
        slot: 0,
        state_slot: 0,
        generation: 1,
        phase: Phase::Prefill,
        tokens: &chunk,
        prompt_len: 64,
        selection: sel,
    }];
    let plan = token_batch::plan(&requests, &[0], &[1], 32, 4096, 0).unwrap();
    assert_eq!(plan.sample_rows, 0);
    assert!(plan.sample_input_rows.is_empty());
    // The tail helper refuses to build a zero-row segment, which is the contract: the caller
    // commits prefill progress after the body and submits nothing else.
    let built = std::panic::catch_unwind(|| {
        run_tail(
            &hidden(16),
            16,
            &[],
            &[f2bf(1.0); H],
            &weights(VOCAB, H, 0.0),
        )
    });
    assert!(
        built.is_err(),
        "a zero-row selection stage must not be built"
    );
}

/// The CPU's refusal probe sees the new opcode, and reports the tier it resolves to rather than
/// a silent fallback. `plow_cpu_has` IS the CPU's capability check; if it were false the tail
/// would be refused at load, which is the behaviour every backend owes for a missing arm.
#[test]
fn the_cpu_probe_reports_a_real_row_gather_kernel() {
    ffi::init(ffi::Isa::Scalar).unwrap();
    assert!(
        ffi::has(DevOp::RowGather as u16),
        "no CPU RowGather kernel: the terminal segment must be refused at load, not dispatched"
    );
    assert_eq!(
        ffi::tier_of(DevOp::RowGather as u16),
        Some(ffi::Isa::Scalar)
    );
    // And an opcode with no kernel is named, not silently skipped.
    let missing = KernelTable::resolve([DevOp::RowGather as u16, 250].into_iter()).unwrap_err();
    assert_eq!(missing, vec![250]);
}
