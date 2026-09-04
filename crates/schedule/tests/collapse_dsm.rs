//! The collapsed graph schedules feasibly, DSM domains are modeled, and the
//! relax pass demotes a colocated hand-off that can't fit one SM.

use costmodel::{hwspec, GemmShape, RowShape, Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use rewrite::{
    assemble, collapse, ConstraintSet, GraphNode, HandoffKind, LayerPlan, LocalityReq, OpDesc,
    OpKind, OpSpec, RelaxableHandoff, TileGraph,
};
use schedule::{relax, schedule, Config, Machine, ResourceId, TaskKind};

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}

fn plan() -> LayerPlan {
    LayerPlan {
        ops: vec![
            OpSpec {
                name: "norm".into(),
                inputs: vec!["x".into(), "nw".into()],
                output: "h".into(),
                kind: OpKind::Row(RowShape {
                    rows: 512,
                    feat: 512,
                    operands: 2,
                    reduce: true,
                }),
                weight_dtype: nn_graph::DType::BF16,
                compute_dtype: nn_graph::DType::BF16,
            },
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
        ],
    }
}

#[test]
fn machine_models_dsm_domains() {
    let m = Machine::from_soc(&Soc::single(h100(), DEFAULT_PAGE_BYTES), &Config::default());
    // H100 (Hopper) has GPC/DSM domains; the whole unit is not one trivial domain.
    assert!(m.unit(0).dsm_domains > 1);
    assert!(m.unit(0).sms_per_domain > 0 && m.unit(0).sms_per_domain < m.unit(0).sm_count);
    // domain_sms stays within the enabled SM count.
    let last = m.unit(0).dsm_domains - 1;
    assert!(m.domain_sms(0, last).end <= m.unit(0).sm_count);
}

#[test]
fn collapsed_graph_schedules_feasibly() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();
    let (g2, c2) = collapse(&soc, &g, &cons);
    let s = schedule(&soc, &g2, &c2, &Config::default());

    // Every compute lands on an SM; exclusive holds never overlap (feasible).
    for (i, t) in s.tasks.tasks.iter().enumerate() {
        if t.kind == TaskKind::Compute {
            assert!(matches!(s.schedule.placement[&i], ResourceId::Sm(0, _)));
        }
    }
    for items in s.schedule.streams.values() {
        let mut iv: Vec<(u64, u64)> = items
            .iter()
            .map(|&(t, st)| (st, st + s.tasks.tasks[t].dur.max(1)))
            .collect();
        iv.sort_unstable();
        for w in iv.windows(2) {
            assert!(w[0].1 <= w[1].0);
        }
    }
    assert!(s.schedule.makespan > 0);
}

#[test]
fn same_domain_handoff_places_in_one_gpc() {
    // Force a DSM hand-off and check both ends land in the same GPC domain.
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, mut cons) = assemble(&soc, &plan(), SramPolicy::Stream, None).unwrap();
    let (g2, mut c2) = collapse(&soc, &g, &cons);
    let _ = (&mut cons, &g2);
    // Pin the norm→proj hand-off to DSM regardless of the cost default.
    let (producer, consumer) = (c2.relaxables[0].producer, c2.relaxables[0].consumer);
    c2.locality.clear();
    c2.locality
        .insert((producer, consumer), LocalityReq::SameDomain);
    c2.colocation_groups.clear();

    let s = schedule(&soc, &g2, &c2, &Config::default());
    let machine = &s.machine;
    let dom = |i: usize| match s.schedule.placement[&i] {
        ResourceId::Sm(_, sm) => sm / machine.unit(0).sms_per_domain,
        _ => usize::MAX,
    };
    // Producer and consumer compute tiles share a DSM domain.
    let prod_tiles: Vec<usize> = s
        .tasks
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.node == producer && t.kind == TaskKind::Compute)
        .map(|(i, _)| i)
        .collect();
    let cons_tiles: Vec<usize> = s
        .tasks
        .tasks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.node == consumer && t.kind == TaskKind::Compute)
        .map(|(i, _)| i)
        .collect();
    assert!(!prod_tiles.is_empty() && !cons_tiles.is_empty());
    let d = dom(prod_tiles[0]);
    assert!(
        prod_tiles.iter().chain(&cons_tiles).all(|&i| dom(i) == d),
        "DSM pair split across domains"
    );
}

#[test]
fn relax_demotes_oversized_colocation() {
    // A hand-built same-SM hand-off whose two ends together exceed one SM's pages.
    let machine = Machine::from_soc(&Soc::single(h100(), DEFAULT_PAGE_BYTES), &Config::default());
    let budget = machine.unit(0).pages_per_sm;

    let g = TileGraph {
        nodes: vec![
            GraphNode::Compute {
                op: "p".into(),
                kind: rewrite::Compute::Layout,
                passes: 1,
                sram_pages: 0,
                inline_in: vec![],
                inline_out: false,
            },
            GraphNode::DmaOut {
                tensor: "y".into(),
                resident: true,
            },
            GraphNode::DmaIn {
                tensor: "y".into(),
                resident: true,
            },
            GraphNode::Compute {
                op: "c".into(),
                kind: rewrite::Compute::Layout,
                passes: 1,
                sram_pages: 0,
                inline_in: vec![],
                inline_out: false,
            },
        ],
        edges: vec![(0, 1), (2, 3)],
    };
    let mut cons = ConstraintSet::default();
    cons.placement.insert(0, 0);
    cons.placement.insert(3, 0);
    cons.sram_pages.insert(0, budget); // together (2×budget) overflow one SM
    cons.sram_pages.insert(3, budget);
    cons.op_io.insert(
        0,
        OpDesc {
            kind: OpKind::Layout(rewrite::LayoutSpec::copy(4096)),
            inputs: vec![],
            output: "y".into(),
            weight_dtype: nn_graph::DType::BF16,
            activation_elem: 2,
            weight_elem: 2,
            block_quant: false,
            native_fp4: false,
        },
    );
    cons.locality.insert((0, 3), LocalityReq::MustColocate);
    cons.colocation_groups.push(vec![0, 3]);
    cons.relaxables.push(RelaxableHandoff {
        producer: 0,
        consumer: 3,
        tensor: "y".into(),
        default: HandoffKind::SramSameSm,
        alts: vec![
            (HandoffKind::Hbm, 100),
            (HandoffKind::SramSameSm, 5),
            (HandoffKind::Dsm, 20),
        ],
    });

    let (g2, c2) = relax(&machine, &g, &cons);
    // The over-subscribed colocation is demoted to the cheapest alternative (DSM).
    assert_eq!(c2.relaxables[0].default, HandoffKind::Dsm);
    assert_eq!(c2.locality[&(0, 3)], LocalityReq::SameDomain);
    assert!(
        c2.colocation_groups.is_empty(),
        "demoted pair should not colocate"
    );
    // DSM keeps the value resident (no HBM round-trip).
    assert!(matches!(
        g2.nodes[1],
        GraphNode::DmaOut { resident: true, .. }
    ));
}
