//! CHUNK-2: the PerChunk prefill pipeline the double-buffered kernel consumes.
//!
//! Asserts the tile→chunk *merge* (`group_by_row_axis`) and the 1:1
//! producer→consumer chunk edges / counters that `expand_prefill_chunks`
//! produces, plus the static-affinity SM-set placement — and that PerOp/PerTile
//! expansion is unchanged by the PerChunk work.

use costmodel::{hwspec, Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use nn_graph::{infer_shapes, ActKind, DType, Nn};
use rewrite::{assemble, plan_from_block, LayerPlan, TileDomain};
use schedule::{
    expand, expand_prefill_chunks, ChunkPlacementPolicy, Config, Granularity, Machine, TaskKind,
};
use std::collections::{HashMap, HashSet};

const H: i64 = 512;
const IM: i64 = 1024;
const T: i64 = 2048; // prefill M

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}

/// A 2-layer MLP block: down_proj(silu(gate)·up). The proj GEMMs are row-coupled
/// on the M (token) axis, so the assembler records tile deps → chunk pipeline.
fn mlp_plan() -> LayerPlan {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let x = nn.input("x", nn.shape([T.into(), H.into()]), DType::BF16);
    nn.begin_block("layers.0");
    let h = nn.rmsnorm("norm", x, H, 1e-6);
    let gate = nn.linear("gate_proj", h, H, IM, false);
    let up = nn.linear("up_proj", h, H, IM, false);
    let ga = nn.act(ActKind::Silu, gate);
    let gu = nn.mul(ga, up);
    let down = nn.linear("down_proj", gu, IM, H, false);
    let out = nn.add(x, down);
    nn.end_block();
    nn.mark_output(out);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("infer");
    plan_from_block(&g, 0).expect("plan")
}

fn setup() -> (Soc<'static>, rewrite::TileGraph, rewrite::ConstraintSet) {
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let plan = mlp_plan();
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    (soc, g, cons)
}

#[test]
fn static_colocated_placement_tiles_the_chip_disjointly() {
    let (soc, g, cons) = setup();
    let cfg = Config {
        granularity: Granularity::PerChunk(4),
        ..Config::default()
    };
    let machine = Machine::from_soc(&soc, &cfg);
    let n_cu = 132;
    let plan = expand_prefill_chunks(
        &soc,
        &machine,
        &g,
        &cons,
        n_cu,
        4,
        ChunkPlacementPolicy::StaticColocated,
    );

    // Group placements by node; each node's chunk CU-ranges must be disjoint and
    // tile [0, n_cu) — chunk c pins to SM-set [c*n_cu/k, (c+1)*n_cu/k).
    let mut by_node: HashMap<usize, Vec<(u32, (u32, u32))>> = HashMap::new();
    for p in &plan.placement {
        by_node
            .entry(p.node)
            .or_default()
            .push((p.chunk, p.cu_range));
    }
    for (_node, mut v) in by_node {
        v.sort();
        assert_eq!(v[0].1 .0, 0, "first chunk starts at SM 0");
        assert_eq!(v.last().unwrap().1 .1, n_cu, "last chunk ends at n_cu");
        for w in v.windows(2) {
            assert_eq!(w[0].1 .1, w[1].1 .0, "SM-sets disjoint & contiguous");
        }
    }
}

#[test]
fn global_queue_placement_uses_whole_chip() {
    let (soc, g, cons) = setup();
    let cfg = Config {
        granularity: Granularity::PerChunk(4),
        ..Config::default()
    };
    let machine = Machine::from_soc(&soc, &cfg);
    let plan = expand_prefill_chunks(
        &soc,
        &machine,
        &g,
        &cons,
        132,
        4,
        ChunkPlacementPolicy::GlobalQueue,
    );
    assert!(
        plan.placement.iter().all(|p| p.cu_range == (0, 132)),
        "GQ: every chunk may use the whole chip"
    );
}

#[test]
fn chunk_edges_are_one_to_one_on_coupled_boundaries() {
    let (soc, g, cons) = setup();
    let k = 4u32;
    let cfg = Config {
        granularity: Granularity::PerChunk(k),
        ..Config::default()
    };
    let machine = Machine::from_soc(&soc, &cfg);
    let plan = expand_prefill_chunks(
        &soc,
        &machine,
        &g,
        &cons,
        132,
        k,
        ChunkPlacementPolicy::default(),
    );

    assert!(!plan.chunk_edges.is_empty(), "a chunk pipeline must exist");

    // node -> its compute chunk tasks.
    let mut node_tasks: HashMap<usize, HashSet<usize>> = HashMap::new();
    for p in &plan.placement {
        node_tasks.entry(p.node).or_default().insert(p.task);
    }

    // On every row-coupled boundary (a recorded tile dep) with equal chunk
    // counts, producer chunk c → consumer chunk c is 1:1: each consumer chunk has
    // exactly one producer chunk feeding it (threshold 1, never a global barrier).
    let mut checked = 0;
    for dep in &cons.tile_deps {
        let (pn, cn) = (dep.producer, dep.consumer);
        let (Some(pt), Some(ct)) = (node_tasks.get(&pn), node_tasks.get(&cn)) else {
            continue;
        };
        // Skip boundaries whose ops have no row axis (untiled).
        let row = |n: usize| cons.domains.get(&n).and_then(TileDomain::row_axis);
        if row(pn).is_none() || row(cn).is_none() {
            continue;
        }
        if pt.len() != ct.len() {
            continue; // matched-count boundaries only
        }
        // Count predecessors of each consumer chunk task among producer chunks.
        let mut indeg: HashMap<usize, usize> = ct.iter().map(|&t| (t, 0)).collect();
        for &(a, b) in &plan.chunk_edges {
            if pt.contains(&a) && ct.contains(&b) {
                *indeg.get_mut(&b).unwrap() += 1;
            }
        }
        for (&_c, &d) in &indeg {
            assert_eq!(
                d, 1,
                "consumer chunk must have exactly one producer chunk (1:1)"
            );
        }
        checked += 1;
    }
    assert!(
        checked > 0,
        "at least one coupled boundary was 1:1-verified"
    );

    // No cross-op counter is a global barrier: on a matched coupled boundary the
    // threshold is 1, not k.
    let mut fine_ones = 0;
    for ctr in &plan.counters {
        if ctr.producer_node != ctr.consumer_node {
            assert!(ctr.threshold >= 1);
            if ctr.threshold == 1 {
                fine_ones += 1;
            }
        }
    }
    assert!(
        fine_ones > 0,
        "the pipeline uses threshold-1 (fine) counters"
    );
}

#[test]
fn perop_pertile_expansion_unchanged() {
    // Regression: adding PerChunk must not perturb PerTile / PerOp expansion.
    let (soc, g, cons) = setup();

    let mk = |gran: Granularity| {
        let cfg = Config {
            granularity: gran,
            ..Config::default()
        };
        let machine = Machine::from_soc(&soc, &cfg);
        expand(&soc, &machine, &g, &cons, &cfg)
    };

    let per_tile = mk(Granularity::PerTile);
    let per_op = mk(Granularity::PerOp);

    // PerOp: exactly one compute task per compute node.
    let n_compute_nodes = cons.domains.len();
    let per_op_compute = per_op
        .tasks
        .iter()
        .filter(|t| t.kind == TaskKind::Compute)
        .count();
    assert_eq!(
        per_op_compute, n_compute_nodes,
        "PerOp: 1 compute task per node"
    );

    // PerTile: sum of every node's tile count.
    let expected_tiles: usize = cons.domains.values().map(|d| d.coords().len()).sum();
    let per_tile_compute = per_tile
        .tasks
        .iter()
        .filter(|t| t.kind == TaskKind::Compute)
        .count();
    assert_eq!(
        per_tile_compute, expected_tiles,
        "PerTile: 1 compute task per tile"
    );

    // PerChunk sits strictly between the two for a multi-tile op chain.
    let cfg4 = Config {
        granularity: Granularity::PerChunk(4),
        ..Config::default()
    };
    let m4 = Machine::from_soc(&soc, &cfg4);
    let per_chunk = expand(&soc, &m4, &g, &cons, &cfg4);
    let per_chunk_compute = per_chunk
        .tasks
        .iter()
        .filter(|t| t.kind == TaskKind::Compute)
        .count();
    assert!(per_chunk_compute >= per_op_compute);
    assert!(per_chunk_compute <= per_tile_compute);
}
