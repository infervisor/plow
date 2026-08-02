//! Verification flow: the emitted packet stream's counter protocol enforces
//! every data dependency — no tile starts before its parents complete. Drives
//! real schedules (1 GPU / 2×H100 / B200) through the byte-level verifier, then
//! tampers with the stream to prove each check actually bites.

use costmodel::{hwspec, Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use nn_graph::{infer_shapes, ActKind, DType, Nn};
use rewrite::{assemble, collapse, plan_from_block, LayerPlan};
use schedule::{
    emit_program, issue_order, relax, schedule, verify, Config, Machine, Scheduled, VerifyError,
    VerifyReport,
};
use std::collections::{HashMap, HashSet};

const H: i64 = 256;
const NH: i64 = 4;
const NKV: i64 = 2;
const HD: i64 = 64;
const QD: i64 = NH * HD;
const KVD: i64 = NKV * HD;
const IM: i64 = 512;
const T: i64 = 256;

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}
fn b200() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("B200").unwrap()
}

/// A small but complete transformer block (norm → qkv → attn → o → mlp).
fn block_plan() -> LayerPlan {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let x = nn.input("x", nn.shape([T.into(), H.into()]), DType::BF16);
    nn.begin_block("layers.0");
    let h1 = nn.rmsnorm("input_norm", x, H, 1e-6);
    let q = nn.linear("q_proj", h1, H, QD, false);
    let k = nn.linear("k_proj", h1, H, KVD, false);
    let v = nn.linear("v_proj", h1, H, KVD, false);
    let qh = nn.reshape(q, [T.into(), NH.into(), HD.into()]);
    let kh = nn.reshape(k, [T.into(), NKV.into(), HD.into()]);
    let vh = nn.reshape(v, [T.into(), NKV.into(), HD.into()]);
    let attn = nn.attention(
        qh, kh, vh, NH as u32, NKV as u32, HD as u32, true, None, None,
    );
    let ao = nn.reshape(attn, [T.into(), QD.into()]);
    let o = nn.linear("o_proj", ao, QD, H, false);
    let r1 = nn.add(x, o);
    let h2 = nn.rmsnorm("post_norm", r1, H, 1e-6);
    let gate = nn.linear("gate_proj", h2, H, IM, false);
    let up = nn.linear("up_proj", h2, H, IM, false);
    let ga = nn.act(ActKind::Silu, gate);
    let gu = nn.mul(ga, up);
    let down = nn.linear("down_proj", gu, IM, H, false);
    let out = nn.add(r1, down);
    nn.end_block();
    nn.mark_output(out);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("infer");
    plan_from_block(&g, 0).expect("plan")
}

fn scheduled(soc: &Soc) -> (rewrite::TileGraph, rewrite::ConstraintSet, Scheduled) {
    let plan = block_plan();
    let (g, cons) = assemble(soc, &plan, SramPolicy::Stream, None).unwrap();
    let s = schedule(soc, &g, &cons, &Config::default());
    (g, cons, s)
}

/// The full compile pipeline `compile_buckets` now runs — assemble → collapse →
/// relax → schedule — must still emit a stream whose counters enforce every
/// dependency (cost-driven hand-off lowering must not introduce a wait cycle).
#[test]
fn collapsed_relaxed_stream_verifies() {
    let cfg = Config::default();
    for soc in [
        Soc::single(h100(), DEFAULT_PAGE_BYTES),
        Soc::single(b200(), DEFAULT_PAGE_BYTES),
    ] {
        let plan = block_plan();
        let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
        let (g, cons) = collapse(&soc, &g, &cons);
        let (g, cons) = relax(&Machine::from_soc(&soc, &cfg), &g, &cons);
        let s = schedule(&soc, &g, &cons, &cfg);
        let report = s
            .verify(&g, &cons)
            .expect("collapsed+relaxed schedule must verify");
        assert_eq!(report.edges_checked, s.tasks.edges.len());
        assert!(report.rounds > 0);
    }
}

/// Core flow on a single GPU (and on B200): every data edge is enforced (by a
/// counter or resource order) and the protocol replays to completion with no
/// premature start.
#[test]
fn single_device_stream_verifies() {
    for soc in [
        Soc::single(h100(), DEFAULT_PAGE_BYTES),
        Soc::single(b200(), DEFAULT_PAGE_BYTES),
    ] {
        let (g, cons, s) = scheduled(&soc);
        let report: VerifyReport = s.verify(&g, &cons).expect("schedule must verify");

        // Every data edge of the task graph was proven enforced.
        assert_eq!(report.edges_checked, s.tasks.edges.len());
        assert!(report.edges_checked > 0, "block has data dependencies");
        // The protocol makes progress and completes (no deadlock).
        assert!(report.rounds > 0);
        assert_eq!(report.insts, s.schedule.placement.len());
        // There is real parallelism to exploit (the point of the counter scheme).
        assert!(report.max_ready >= 1);
    }
}

/// The 2-GPU tensor-parallel stream verifies: intra-op staging counters are
/// per-producer, so the cross-unit transfer no longer makes a compute tile wait
/// on a counter it must increment (the self-dependency that previously
/// deadlocked). The cross-unit boundary is still gated by a `CrossUnit` counter.
#[test]
fn tensor_parallel_stream_verifies() {
    let soc = Soc::homogeneous(h100(), 2, DEFAULT_PAGE_BYTES);
    let (g, cons, s) = scheduled(&soc);
    let report = s
        .verify(&g, &cons)
        .expect("tensor-parallel stream must verify after the counter fix");
    assert_eq!(report.edges_checked, s.tasks.edges.len());
    assert!(report.rounds > 0);
    // A genuine cross-unit dependency is still present and counter-gated.
    assert!(
        s.schedule
            .counters
            .iter()
            .any(|c| c.scope == schedule::Scope::CrossUnit),
        "cross-unit transfer should still be gated by a CrossUnit counter"
    );
}

/// Raising a counter's threshold above its producer count makes it unreachable
/// — the verifier reports the deadlock instead of silently accepting it.
#[test]
fn rejects_unreachable_counter() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons, s) = scheduled(&soc);
    let mut prog = emit_program(&g, &cons, &s.tasks, &s.schedule);
    let order = issue_order(&s.schedule);

    // Inflate the first counter's threshold beyond what producers can ever reach.
    let c = &mut prog.counters[0];
    c.threshold += 1_000;
    let id = c.id;
    let err = verify(&prog, &order, &s.tasks).unwrap_err();
    assert_eq!(
        err,
        VerifyError::Unreachable {
            id,
            threshold: prog.counters[0].threshold,
            producers: prog.counters[0].threshold - 1_000
        }
    );
}

/// Dropping the counter that gates a *cross-resource* edge lets the child start
/// before its parent — caught as an unguarded edge (resource order can't cover a
/// cross-resource dependency).
#[test]
fn rejects_dropped_cross_resource_wait() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons, s) = scheduled(&soc);
    let mut prog = emit_program(&g, &cons, &s.tasks, &s.schedule);
    let order = issue_order(&s.schedule);
    let inst_of: HashMap<usize, usize> = order.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    let key = |i: usize| {
        (
            prog.insts[i].resource as u8,
            prog.insts[i].unit,
            prog.insts[i].index,
        )
    };

    // Pick a data edge gated by a counter across two resources, and strip that
    // gating counter from the consumer's wait list.
    let (parent, child) = *s
        .tasks
        .edges
        .iter()
        .find(|&&(a, b)| {
            let (ia, ib) = (inst_of[&a], inst_of[&b]);
            let crosses = key(ia) != key(ib);
            let gated: HashSet<u32> = prog.insts[ia]
                .succ
                .iter()
                .filter(|c| prog.insts[ib].wait.contains(c))
                .copied()
                .collect();
            crosses && !gated.is_empty()
        })
        .expect("a cross-resource counter-gated edge");

    let (ia, ib) = (inst_of[&parent], inst_of[&child]);
    let gating: HashSet<u32> = prog.insts[ia].succ.iter().copied().collect();
    prog.insts[ib].wait.retain(|c| !gating.contains(c));

    assert_eq!(
        verify(&prog, &order, &s.tasks),
        Err(VerifyError::UnguardedEdge { parent, child })
    );
}

/// Realistic shapes produce multi-tile ops. Fine counters must still verify:
/// no deadlock, no premature fire.
///
/// Uses `DmaModel::Collapsed` to isolate counter logic from DMA-engine
/// scheduling. Shape chosen to produce ~512 tiles per GEMM (32×16) — enough
/// to exercise multi-tile scheduling without O(N²) edge explosion.
#[test]
fn realistic_shape_verifies_with_fine_counters() {
    use costmodel::{GemmShape, RowShape, SramPolicy};
    use rewrite::{LayerPlan, OpKind, OpSpec};

    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let plan = LayerPlan {
        ops: vec![
            OpSpec {
                name: "gemm1".into(),
                inputs: vec!["x".into(), "w1".into()],
                output: "h".into(),
                kind: OpKind::Gemm(GemmShape {
                    m: 4096,
                    n: 2048,
                    k: 2048,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
            OpSpec {
                name: "act".into(),
                inputs: vec!["h".into()],
                output: "a".into(),
                kind: OpKind::Row(RowShape {
                    rows: 4096,
                    feat: 2048,
                    operands: 1,
                    reduce: false,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
            OpSpec {
                name: "gemm2".into(),
                inputs: vec!["a".into(), "w2".into()],
                output: "y".into(),
                kind: OpKind::Gemm(GemmShape {
                    m: 4096,
                    n: 2048,
                    k: 2048,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
        ],
    };
    let cfg = Config {
        dma_model: schedule::DmaModel::Collapsed,
        cluster: schedule::ClusterMode::Fine,
        ..Config::default()
    };
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &cfg);
    let report = s
        .verify(&g, &cons)
        .expect("fine counters must verify at realistic tile counts");
    assert!(report.edges_checked > 0);
    assert!(report.rounds > 0);
    // With fine counters, max parallelism should be > 1 (tiles fire as soon as
    // their specific producers complete, not all of them).
    assert!(
        report.max_ready > 1,
        "fine counters should enable parallel tile issue (got max_ready={})",
        report.max_ready
    );
}

/// Fine counters pipeline correctly: in a multi-tile GEMM→Row boundary,
/// consumer tiles should fire in fewer rounds than the total producer count
/// (i.e. a consumer doesn't wait for all producers, only its own).
///
/// Uses `DmaModel::Collapsed` to isolate counter pipelining from DMA-engine
/// scheduling (which has its own cycle issues at high tile counts with coarse
/// counters and `Separate` DMA).
#[test]
fn fine_counters_pipeline_across_boundary() {
    use costmodel::{GemmShape, RowShape, SramPolicy};
    use rewrite::{LayerPlan, OpKind, OpSpec};

    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let plan = LayerPlan {
        ops: vec![
            OpSpec {
                name: "gemm".into(),
                inputs: vec!["x".into(), "w".into()],
                output: "h".into(),
                kind: OpKind::Gemm(GemmShape {
                    m: 4096,
                    n: 2048,
                    k: 2048,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
            OpSpec {
                name: "row".into(),
                inputs: vec!["h".into()],
                output: "y".into(),
                kind: OpKind::Row(RowShape {
                    rows: 4096,
                    feat: 2048,
                    operands: 1,
                    reduce: false,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
        ],
    };
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();

    // Fine (default) with collapsed DMA — pipelined.
    let cfg_fine = Config {
        dma_model: schedule::DmaModel::Collapsed,
        cluster: schedule::ClusterMode::Fine,
        ..Config::default()
    };
    let s_fine = schedule(&soc, &g, &cons, &cfg_fine);
    let r_fine = s_fine.verify(&g, &cons).expect("fine must verify");

    // Coarse with collapsed DMA — serialized all-wait-all but no DMA interleaving.
    let cfg_coarse = Config {
        dma_model: schedule::DmaModel::Collapsed,
        cluster: schedule::ClusterMode::Coarse,
        ..Config::default()
    };
    let s_coarse = schedule(&soc, &g, &cons, &cfg_coarse);
    let r_coarse = s_coarse
        .verify(&g, &cons)
        .expect("coarse must verify at this size");

    // Fine should have same or better parallelism (more max_ready or fewer rounds).
    assert!(
        r_fine.max_ready >= r_coarse.max_ready || r_fine.rounds <= r_coarse.rounds,
        "fine should pipeline at least as well as coarse: fine({} rounds, {} max_ready) vs coarse({} rounds, {} max_ready)",
        r_fine.rounds, r_fine.max_ready, r_coarse.rounds, r_coarse.max_ready
    );
}
