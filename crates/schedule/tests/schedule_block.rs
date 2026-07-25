//! End-to-end: build a small transformer block, assemble its tile graph on
//! 1 GPU / 2×H100 / a heterogeneous SoC, schedule it, and assert the
//! resource-feasibility invariants (design §4–§9).

use costmodel::{hwspec, CostModel, DEFAULT_PAGE_BYTES, MemoryModel, Soc, SramPolicy, Unit, UnitKind};
use nn_graph::{infer_shapes, ActKind, DType, Nn};
use rewrite::{assemble, plan_from_block, LayerPlan};
use schedule::{
    build_counters, list_schedule, schedule, Config, DmaModel, Granularity, Machine, Packet,
    ResourceId, Scope, Task, TaskGraph, TaskKind,
};
use std::collections::HashMap;

// Tiny block dims (the scheduler doesn't need realistic sizes).
const H: i64 = 256;
const NH: i64 = 4;
const NKV: i64 = 2;
const HD: i64 = 64;
const QD: i64 = NH * HD; // 256
const KVD: i64 = NKV * HD; // 128
const IM: i64 = 512;
const T: i64 = 256;

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}
fn h100_pcie() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 PCIe").unwrap()
}
fn b200() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("B200").unwrap()
}

/// A heterogeneous SoC: H100 SXM5 (132 SMs) + H100 PCIe (114 SMs), unified.
fn heterogeneous() -> Soc<'static> {
    Soc {
        units: vec![
            Unit {
                id: 0,
                kind: UnitKind::Gpu,
                weight: 1.0,
                cm: CostModel::new(h100(), DEFAULT_PAGE_BYTES),
            },
            Unit {
                id: 1,
                kind: UnitKind::Gpu,
                weight: 0.7,
                cm: CostModel::new(h100_pcie(), DEFAULT_PAGE_BYTES),
            },
        ],
        memory: MemoryModel { unified: true },
    }
}

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
    let qn = nn.rmsnorm("q_norm", qh, HD, 1e-6);
    let kn = nn.rmsnorm("k_norm", kh, HD, 1e-6);
    let qr = nn.rope(qn, HD as u32, 1e6);
    let kr = nn.rope(kn, HD as u32, 1e6);
    let attn = nn.attention(
        qr, kr, vh, NH as u32, NKV as u32, HD as u32, true, None, None,
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

/// Assert the core feasibility invariants of a schedule.
fn assert_feasible(s: &schedule::Scheduled, machine: &Machine) {
    let sched = &s.schedule;
    // (a) every compute task on an SM of its own unit.
    for (i, t) in s.tasks.tasks.iter().enumerate() {
        if t.kind == TaskKind::Compute {
            match sched.placement[&i] {
                ResourceId::Sm(u, sm) => {
                    assert_eq!(u, t.unit, "compute on wrong unit");
                    assert!(sm < machine.unit(u).sm_count);
                    // (d) a tile's SRAM footprint fits the per-SM budget.
                    assert!(t.sram_pages <= machine.unit(u).pages_per_sm);
                }
                r => panic!("compute task not on an SM: {r:?}"),
            }
        }
    }
    // (c) no two exclusive reservations overlap on any single resource.
    for (r, items) in &sched.streams {
        let mut iv: Vec<(u64, u64)> = items
            .iter()
            .map(|&(t, st)| (st, st + s.tasks.tasks[t].dur.max(1)))
            .collect();
        iv.sort_unstable();
        for w in iv.windows(2) {
            assert!(w[0].1 <= w[1].0, "overlapping reservations on {r:?}");
        }
    }
    // (g) prefetch/ordering: every DMA-in lands before the consumer it feeds.
    for &(a, b) in &s.tasks.edges {
        if s.tasks.tasks[a].kind == TaskKind::DmaIn {
            assert!(
                sched.starts[a] + s.tasks.tasks[a].dur.max(1) <= sched.starts[b],
                "DMA-in not before its consumer"
            );
        }
    }
    // (h) makespan finite and ≥ the longest single task.
    let longest = s.tasks.tasks.iter().map(|t| t.dur).max().unwrap_or(0);
    assert!(sched.makespan >= longest && sched.makespan > 0);
    // (f) counters: positive thresholds.
    assert!(sched.counters.iter().all(|c| c.threshold >= 1));

    // (d) per-page SRAM. Each *assigned* tile holds exactly its page count, no
    // individual page slot is double-booked over overlapping live intervals, and
    // every resident tile is accounted for (assigned or counted as a spill — a
    // spill is a correct outcome, not a failure: that tile round-trips HBM).
    let mut succ = vec![Vec::new(); s.tasks.tasks.len()];
    for &(a, b) in &s.tasks.edges {
        succ[a].push(b);
    }
    let live = |t: usize| -> (u64, u64) {
        let st = sched.starts[t];
        let last = succ[t].iter().map(|&c| sched.starts[c]).max().unwrap_or(0);
        (st, last.max(st + s.tasks.tasks[t].dur.max(1)))
    };
    let mut slot_iv: HashMap<(ResourceId, usize), Vec<(u64, u64)>> = HashMap::new();
    for (&t, slots) in &sched.sram_slots {
        assert_eq!(
            slots.len() as u64,
            s.tasks.tasks[t].out_pages,
            "tile did not get its full page count"
        );
        // assigned page slots are within the SM's budget.
        if let ResourceId::Sm(u, _) = sched.placement[&t] {
            assert!(slots
                .iter()
                .all(|&p| (p as u64) < machine.unit(u).pages_per_sm));
        }
        let r = sched.placement[&t];
        let iv = live(t);
        for &sl in slots {
            slot_iv.entry((r, sl)).or_default().push(iv);
        }
    }
    for ivs in slot_iv.values_mut() {
        ivs.sort_unstable();
        for w in ivs.windows(2) {
            assert!(
                w[0].1 <= w[1].0,
                "SRAM page slot double-booked over overlapping live intervals"
            );
        }
    }
    // accounting: every resident compute tile is either assigned or spilled.
    let resident = s
        .tasks
        .tasks
        .iter()
        .enumerate()
        .filter(|(i, t)| {
            t.kind == TaskKind::Compute
                && t.out_pages > 0
                && matches!(sched.placement[i], ResourceId::Sm(..))
        })
        .count();
    assert_eq!(
        resident,
        sched.sram_slots.len() + sched.spills,
        "resident tiles not accounted for"
    );

    // (i) TMEM accumulators: each assigned matmul tile holds exactly its column
    // count within the SM budget, none double-booked over the compute interval,
    // and every TMEM-using tile is accounted for.
    let mut tmem_iv: HashMap<(ResourceId, usize), Vec<(u64, u64)>> = HashMap::new();
    for (&t, cols) in &sched.tmem_slots {
        assert_eq!(
            cols.len() as u64,
            s.tasks.tasks[t].tmem_cols,
            "tile did not get its accumulator columns"
        );
        if let ResourceId::Sm(u, _) = sched.placement[&t] {
            assert!(cols
                .iter()
                .all(|&c| (c as u64) < machine.unit(u).tmem_cols_per_sm));
        }
        let st = sched.starts[t];
        let iv = (st, st + s.tasks.tasks[t].dur.max(1));
        for &c in cols {
            tmem_iv
                .entry((sched.placement[&t], c))
                .or_default()
                .push(iv);
        }
    }
    for ivs in tmem_iv.values_mut() {
        ivs.sort_unstable();
        for w in ivs.windows(2) {
            assert!(w[0].1 <= w[1].0, "TMEM column double-booked");
        }
    }
    let tmem_users = s
        .tasks
        .tasks
        .iter()
        .enumerate()
        .filter(|(i, t)| t.tmem_cols > 0 && matches!(sched.placement[i], ResourceId::Sm(..)))
        .count();
    assert_eq!(
        tmem_users,
        sched.tmem_slots.len() + sched.tmem_spills,
        "TMEM tiles not accounted for"
    );
}

#[test]
fn single_gpu_schedule_is_feasible() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let plan = block_plan();
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &Config::default());
    assert_feasible(&s, &s.machine);

    // Single unit ⇒ no cross-unit work: nothing on a DPU, no CrossUnit counter.
    assert!(!s
        .schedule
        .placement
        .values()
        .any(|r| matches!(r, ResourceId::Dpu(_))));
    assert!(s
        .schedule
        .counters
        .iter()
        .all(|c| c.scope != Scope::CrossUnit));
    // (b) colocated nodes pin by row coordinate.
    assert_colocation(&s);
}

#[test]
fn two_h100_splits_and_uses_dpu() {
    let soc = Soc::homogeneous(h100(), 2, DEFAULT_PAGE_BYTES);
    let plan = block_plan();
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &Config::default());
    assert_feasible(&s, &s.machine);

    // The split GEMMs' joins are cross-unit ⇒ a DPU transfer + a CrossUnit counter.
    let on_dpu = s
        .schedule
        .placement
        .values()
        .filter(|r| matches!(r, ResourceId::Dpu(_)))
        .count();
    assert!(on_dpu > 0, "expected cross-unit transfers routed to a DPU");
    assert!(s
        .schedule
        .counters
        .iter()
        .any(|c| c.scope == Scope::CrossUnit));
    // Both units are actually used by compute.
    let units: std::collections::HashSet<_> = s
        .schedule
        .placement
        .iter()
        .filter_map(|(t, r)| match r {
            ResourceId::Sm(u, _) if s.tasks.tasks[*t].kind == TaskKind::Compute => Some(*u),
            _ => None,
        })
        .collect();
    assert_eq!(units, [0, 1].into_iter().collect());
}

#[test]
fn heterogeneous_soc_schedule_is_feasible() {
    let soc = heterogeneous();
    let plan = block_plan();
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &Config::default());
    assert_feasible(&s, &s.machine);
    // Heterogeneous SM counts carried through to the machine model.
    assert_eq!(s.machine.unit(0).sm_count, 132);
    assert_eq!(s.machine.unit(1).sm_count, 114);
    assert!(s
        .schedule
        .placement
        .values()
        .any(|r| matches!(r, ResourceId::Dpu(_))));
}

#[test]
fn b200_tracks_tmem_accumulators() {
    // On B200 the MMA accumulators live in TMEM — matmul tiles get TMEM columns.
    let soc = Soc::single(b200(), DEFAULT_PAGE_BYTES);
    let plan = block_plan();
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &Config::default());
    assert_feasible(&s, &s.machine);

    // The machine model carries TMEM: 256 KiB / 512 B-per-column = 512 columns.
    assert_eq!(s.machine.unit(0).tmem_cols_per_sm, 512);
    // Every GEMM / flash tile reserves an accumulator in TMEM; row/layout don't.
    let matmul_tiles = s
        .tasks
        .tasks
        .iter()
        .filter(|t| t.kind == TaskKind::Compute && t.tmem_cols > 0)
        .count();
    assert!(
        matmul_tiles > 0,
        "expected matmul tiles to use TMEM on B200"
    );
    assert!(
        !s.schedule.tmem_slots.is_empty(),
        "TMEM columns were not assigned"
    );
    // Accumulators of these tiles fit (no TMEM spill).
    assert_eq!(s.schedule.tmem_spills, 0);
}

#[test]
fn h100_has_no_tmem() {
    // Hopper has no TMEM — no tile reserves accumulator columns.
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let plan = block_plan();
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let s = schedule(&soc, &g, &cons, &Config::default());
    assert_eq!(s.machine.unit(0).tmem_cols_per_sm, 0);
    assert!(s.tasks.tasks.iter().all(|t| t.tmem_cols == 0));
    assert!(s.schedule.tmem_slots.is_empty());
    assert_eq!(s.schedule.tmem_spills, 0);
}

#[test]
fn config_matrix_all_valid() {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let plan = block_plan();
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    for gran in [Granularity::PerTile, Granularity::PerOp] {
        for dma in [DmaModel::Separate, DmaModel::Collapsed] {
            let cfg = Config {
                granularity: gran,
                dma_model: dma,
                ..Config::default()
            };
            let s = schedule(&soc, &g, &cons, &cfg);
            assert_feasible(&s, &s.machine);
        }
    }
    // PerOp ⇒ one compute task per op (covers op-level scheduling).
    let cfg = Config {
        granularity: Granularity::PerOp,
        ..Config::default()
    };
    let s = schedule(&soc, &g, &cons, &cfg);
    let compute_nodes: std::collections::HashSet<_> = s
        .tasks
        .tasks
        .iter()
        .filter(|t| t.kind == TaskKind::Compute)
        .map(|t| t.node)
        .collect();
    let compute_tasks = s
        .tasks
        .tasks
        .iter()
        .filter(|t| t.kind == TaskKind::Compute)
        .count();
    assert_eq!(
        compute_nodes.len(),
        compute_tasks,
        "PerOp should emit one compute task per op"
    );
}

#[test]
fn host_thread_routing() {
    // A hand-built task graph with a host-coordination task lands on a host thread.
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let machine = Machine::from_soc(&soc, &Config::default());
    let tg = TaskGraph {
        tasks: vec![Task {
            node: 0,
            op: "route".into(),
            unit: 0,
            kind: TaskKind::Host,
            coord: vec![],
            dur: 10,
            bytes: 0,
            tensor_bytes: 0,
            sram_pages: 0,
            out_pages: 0,
            tmem_cols: 0,
            tensor: None,
            cross_unit: false,
        }],
        edges: vec![],
        ..Default::default()
    };
    let (counters, wait, succ) = build_counters(&tg, &HashMap::new(), &Default::default(), schedule::ClusterMode::default());
    let s = list_schedule(
        &machine,
        &tg,
        &Default::default(),
        &Default::default(),
        &Default::default(),
        &counters,
        &wait,
        &succ,
    );
    assert!(matches!(s.placement[&0], ResourceId::Host(_)));
    // Pass G produced a host-coordination packet.
    let pkts: Vec<&Packet> = s.packets.values().flatten().collect();
    assert_eq!(pkts.len(), 1);
    assert_eq!(pkts[0].kind, schedule::PacketKind::HostCoord);
}

#[test]
fn machine_models_each_soc() {
    let one = Machine::from_soc(&Soc::single(h100(), DEFAULT_PAGE_BYTES), &Config::default());
    assert_eq!(one.units.len(), 1);
    assert_eq!(one.units[0].sm_count, 132);

    let two = Machine::from_soc(
        &Soc::homogeneous(h100(), 2, DEFAULT_PAGE_BYTES),
        &Config::default(),
    );
    assert_eq!(two.units.len(), 2);
    assert!(two.unit(0).hbm_bytes_per_cycle > 0.0);

    let het = Machine::from_soc(&heterogeneous(), &Config::default());
    assert_eq!(het.unit(0).sm_count, 132);
    assert_eq!(het.unit(1).sm_count, 114);
    // The PCIe part has lower HBM bandwidth than the SXM5 part.
    assert!(het.unit(1).hbm_bytes_per_cycle < het.unit(0).hbm_bytes_per_cycle);
}

/// Tasks of the same node with equal row coordinate must share one SM
/// (the deterministic colocation pinning).
fn assert_colocation(s: &schedule::Scheduled) {
    let mut by_key: HashMap<(usize, i64), ResourceId> = HashMap::new();
    for (i, t) in s.tasks.tasks.iter().enumerate() {
        if t.kind != TaskKind::Compute {
            continue;
        }
        let key = (t.node, t.coord.first().copied().unwrap_or(0));
        if let Some(prev) = by_key.insert(key, s.schedule.placement[&i]) {
            assert_eq!(
                prev, s.schedule.placement[&i],
                "same node+coord on different SMs"
            );
        }
    }
}
