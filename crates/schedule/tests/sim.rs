//! The cycle simulator replays a schedule to a real makespan, stays consistent
//! with the analytic estimate, and surfaces clustering overhead + utilization.

use costmodel::{hwspec, GemmShape, RowShape, Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use rewrite::{assemble, collapse, LayerPlan, OpKind, OpSpec};
use schedule::{schedule, Config};

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}

/// norm → proj → act: cross-op hand-offs that produce clustered counters.
fn plan() -> LayerPlan {
    let row = |name: &str, ins: &[&str], out: &str, operands| OpSpec {
        name: name.into(),
        inputs: ins.iter().map(|s| s.to_string()).collect(),
        output: out.into(),
        kind: OpKind::Row(RowShape {
            rows: 512,
            feat: 512,
            operands,
            reduce: false,
        }),
        weight_dtype: nn_graph::DType::BF16,
        compute_dtype: nn_graph::DType::BF16,
    };
    LayerPlan {
        ops: vec![
            row("norm", &["x", "nw"], "h", 2),
            OpSpec {
                name: "proj".into(),
                inputs: vec!["h".into(), "w".into()],
                output: "y".into(),
                kind: OpKind::Gemm(GemmShape {
                    m: 512,
                    n: 512,
                    k: 512,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
            row("act", &["y"], "z", 1),
        ],
    }
}

#[test]
fn replay_is_consistent_and_bounded() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &Config::default());
    let sim = s.simulate();

    // The static schedule replays to the same time it was costed at.
    assert!(
        sim.consistent,
        "ideal replay {} != analytic {}",
        sim.ideal_makespan, s.schedule.makespan
    );
    // The real (counter-gated) makespan is never better than perfect pipelining,
    // and never below the longest single task.
    assert!(sim.makespan >= sim.ideal_makespan);
    let longest = s.tasks.tasks.iter().map(|t| t.dur).max().unwrap();
    assert!(sim.ideal_makespan >= longest && sim.makespan > 0);
    // No resource is busy longer than the makespan (utilization ≤ 1).
    for (&r, &b) in &sim.busy {
        assert!(
            b <= sim.makespan,
            "{r:?} busy {b} > makespan {}",
            sim.makespan
        );
        assert!(sim.utilization(r) <= 1.0 + 1e-9);
    }
    // Clustering overhead is well-defined and non-negative.
    assert_eq!(sim.clustering_overhead(), sim.makespan - sim.ideal_makespan);
}

#[test]
fn collapsed_graph_also_replays_consistently() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();
    let (g2, c2) = collapse(&soc, &g, &cons);
    let s = schedule(&soc, &g2, &c2, &Config::default());
    let sim = s.simulate();
    assert!(sim.consistent);
    assert!(sim.makespan >= sim.ideal_makespan && sim.makespan > 0);
}

#[test]
fn dma_fold_skips_separate_loads() {
    // After collapse, single-consumer DRAM loads are folded into their kernels —
    // no separate DMA-in task is emitted for them — and the schedule still
    // replays consistently.
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();
    // Baseline: separate DMA-in tasks for the weight "w" exist.
    let base = schedule(&soc, &g, &cons, &Config::default());
    let base_w = base
        .tasks
        .tasks
        .iter()
        .filter(|t| t.tensor.as_deref() == Some("w"))
        .count();
    assert!(
        base_w > 0,
        "expected a DMA-in for the weight before folding"
    );

    let (g2, c2) = collapse(&soc, &g, &cons);
    let s = schedule(&soc, &g2, &c2, &Config::default());
    let folded_w = s
        .tasks
        .tasks
        .iter()
        .filter(|t| t.tensor.as_deref() == Some("w"))
        .count();
    assert_eq!(
        folded_w, 0,
        "folded weight load should not be a separate DMA task"
    );
    // Still a valid, self-consistent schedule.
    assert!(s.simulate().consistent);
}

/// At realistic tile counts (multi-tile per op), the fine counter strategy
/// produces a non-cyclic dependency graph (no deadlock).
///
/// Uses `DmaModel::Collapsed` to isolate counter correctness from the orthogonal
/// DMA-engine scheduling problem. Without `collapse`+`relax` (which establish
/// fine `TileDep` coupling), all cross-op boundaries are all-to-all — the coarse
/// fallback fires. With collapsed DMA, all tasks live on SMs; the scheduler's
/// topological order guarantees producers before consumers on each SM, so no cycle.
#[test]
fn realistic_shape_is_acyclic() {
    use schedule::config::{ClusterMode, DmaModel};

    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    // 512 compute tiles per GEMM: (4096/128) × (2048/128) = 32 × 16 = 512.
    // Enough to exercise multi-tile scheduling without O(N²) edge explosion.
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
        dma_model: DmaModel::Collapsed,
        cluster: ClusterMode::Fine,
        ..Config::default()
    };
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &cfg);
    let sim = s.simulate();
    assert!(
        !sim.cyclic,
        "fine counters must produce a deadlock-free schedule at realistic tile counts"
    );
    assert!(sim.makespan > 0);
    assert!(sim.consistent);
}

#[test]
fn packet_dump_lists_streams() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &Config::default());
    let dump = s.dump_packets();
    assert!(!dump.is_empty());
    // Mentions at least one SM resource and a compute packet.
    assert!(dump.contains("Sm("));
    assert!(dump.contains("Compute"));
    // Every scheduled task appears as a packet line (one '@' per packet).
    let lines = dump.matches('@').count();
    let tasks = s.schedule.placement.len();
    assert_eq!(
        lines, tasks,
        "packet count {lines} != scheduled tasks {tasks}"
    );
}
