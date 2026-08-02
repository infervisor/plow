//! Cost-driven hand-off collapse: the tile-egglog rules enumerate HBM / SRAM /
//! DSM alternatives and the extractor picks a cost-driven default per hand-off.

use costmodel::{hwspec, GemmShape, RowShape, Soc, SramPolicy, DEFAULT_PAGE_BYTES};
use rewrite::{assemble, collapse, GraphNode, HandoffKind, LayerPlan, LocalityReq, OpKind, OpSpec};

fn h100() -> &'static hwspec::GpuSpec {
    hwspec::registry::lookup("H100 SXM5").unwrap()
}

fn row(name: &str, ins: &[&str], out: &str, rows: i64, feat: i64, operands: i64) -> OpSpec {
    OpSpec {
        name: name.into(),
        inputs: ins.iter().map(|s| s.to_string()).collect(),
        output: out.into(),
        kind: OpKind::Row(RowShape {
            rows,
            feat,
            operands,
            reduce: false,
        }),
        weight_dtype: nn_graph::DType::BF16,
        compute_dtype: nn_graph::DType::BF16,
    }
}
fn gemm(name: &str, ins: &[&str], out: &str, m: i64, n: i64, k: i64) -> OpSpec {
    OpSpec {
        name: name.into(),
        inputs: ins.iter().map(|s| s.to_string()).collect(),
        output: out.into(),
        kind: OpKind::Gemm(GemmShape { m, n, k }),
        weight_dtype: nn_graph::DType::BF16,
        compute_dtype: nn_graph::DType::BF16,
    }
}

/// Find the relaxable hand-off whose tensor is `t`.
fn relax<'a>(cons: &'a rewrite::ConstraintSet, t: &str) -> &'a rewrite::RelaxableHandoff {
    cons.relaxables
        .iter()
        .find(|r| r.tensor == t)
        .expect("relaxable for tensor")
}

#[test]
fn enumerates_three_alternatives_on_dsm_unit() {
    // norm -> proj: a same-unit hand-off on an H100 (which has a GPC/DSM domain).
    let plan = LayerPlan {
        ops: vec![
            row("norm", &["x", "nw"], "h", 512, 512, 2),
            gemm("proj", &["h", "w"], "y", 512, 512, 512),
        ],
    };
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let (_, c2) = collapse(&soc, &g, &cons);

    let r = relax(&c2, "h");
    // All three realizations are offered (DSM rule fired on the DSM unit).
    let kinds: Vec<HandoffKind> = r.alts.iter().map(|&(k, _)| k).collect();
    assert!(kinds.contains(&HandoffKind::Hbm));
    assert!(kinds.contains(&HandoffKind::SramSameSm));
    assert!(
        kinds.contains(&HandoffKind::Dsm),
        "DSM alternative missing on a DSM unit"
    );
    // The default is the min-cost alternative.
    let min = r.alts.iter().min_by_key(|&&(_, c)| c).unwrap().0;
    assert_eq!(r.default, min);
}

#[test]
fn chosen_kind_is_recorded_on_handoff() {
    // norm -> proj on an H100: the single same-unit hand-off's `kind` must be the
    // realization `collapse` picked (not the `Hbm` construction placeholder), so
    // expansion can branch on it (F3 plumbing).
    let plan = LayerPlan {
        ops: vec![
            row("norm", &["x", "nw"], "h", 512, 512, 2),
            gemm("proj", &["h", "w"], "y", 512, 512, 512),
        ],
    };
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let (_, c2) = collapse(&soc, &g, &cons);

    assert_eq!(c2.handoffs.len(), 1, "one norm->proj hand-off");
    // The cheapest realization on this DSM unit is not the HBM round-trip.
    let default = relax(&c2, "h").default;
    assert_ne!(default, HandoffKind::Hbm);
    assert_eq!(
        c2.handoffs[0].kind, default,
        "hand-off carries the chosen realization, not the Hbm placeholder"
    );
}

#[test]
fn default_sets_resident_and_locality_consistently() {
    let plan = LayerPlan {
        ops: vec![
            row("norm", &["x", "nw"], "h", 512, 512, 2),
            gemm("proj", &["h", "w"], "y", 512, 512, 512),
        ],
    };
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let (g2, c2) = collapse(&soc, &g, &cons);

    for r in &c2.relaxables {
        let req = c2.locality[&(r.producer, r.consumer)];
        match r.default {
            HandoffKind::Hbm => assert_eq!(req, LocalityReq::NoConstraint),
            HandoffKind::SramSameSm => assert_eq!(req, LocalityReq::MustColocate),
            HandoffKind::Dsm => assert_eq!(req, LocalityReq::SameDomain),
            HandoffKind::L2Local => assert_eq!(req, LocalityReq::SameL2Partition),
            // Cross-unit tiers don't arise on this single-unit SoC.
            HandoffKind::Barrier | HandoffKind::P2p => assert_eq!(req, LocalityReq::SameNode),
            HandoffKind::Rdma => assert_eq!(req, LocalityReq::NoConstraint),
        }
        // resident flag follows the choice (round-trip ⇒ not resident).
        let resident = matches!(r.default, HandoffKind::SramSameSm | HandoffKind::Dsm);
        for n in &g2.nodes {
            if let GraphNode::DmaOut {
                tensor,
                resident: res,
            } = n
            {
                if tensor == &r.tensor {
                    assert_eq!(*res, resident, "DmaOut resident mismatch for {}", r.tensor);
                }
            }
        }
    }
    // Only SRAM hand-offs co-locate; each colocation pair is a MustColocate edge.
    for grp in &c2.colocation_groups {
        for &a in grp {
            for &b in grp {
                if a != b {
                    let same = c2
                        .locality
                        .get(&(a, b))
                        .or_else(|| c2.locality.get(&(b, a)));
                    if let Some(req) = same {
                        assert_eq!(*req, LocalityReq::MustColocate);
                    }
                }
            }
        }
    }
}

#[test]
fn folds_single_consumer_dram_loads() {
    // norm reads x,nw (each one consumer); proj reads h (produced) + w (one
    // consumer). The DRAM loads x/nw/w fold into their kernels; h does not.
    let plan = LayerPlan {
        ops: vec![
            row("norm", &["x", "nw"], "h", 512, 512, 2),
            gemm("proj", &["h", "w"], "y", 512, 512, 512),
        ],
    };
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let (g2, _) = collapse(&soc, &g, &cons);

    let inline_of = |op: &str| -> Vec<String> {
        g2.nodes
            .iter()
            .find_map(|n| match n {
                GraphNode::Compute {
                    op: o, inline_in, ..
                } if o == op => Some(inline_in.clone()),
                _ => None,
            })
            .expect("compute node")
    };
    let proj = inline_of("proj");
    assert!(
        proj.contains(&"w".to_string()),
        "single-consumer weight should fold"
    );
    assert!(
        !proj.contains(&"h".to_string()),
        "produced hand-off must not fold"
    );
    let norm = inline_of("norm");
    assert!(norm.contains(&"x".to_string()) && norm.contains(&"nw".to_string()));
}

#[test]
fn cross_unit_handoff_enumerates_barrier_p2p_rdma() {
    // A Linear split across two GPUs over unified memory: the Join reads one
    // slice from the other unit — a cross-unit hand-off. The collapse should
    // enumerate the cross-node tiers and (under unified memory) default to the
    // cheap Barrier.
    let plan = LayerPlan {
        ops: vec![gemm("o_proj", &["act", "o.w"], "out", 4096, 4096, 4096)],
    };
    let soc = Soc::homogeneous(h100(), 2, DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let (_, c2) = collapse(&soc, &g, &cons);

    // The cross-unit relaxable (the Join's far-slice read).
    let xr = c2
        .relaxables
        .iter()
        .find(|r| {
            r.alts.iter().any(|&(k, _)| {
                matches!(
                    k,
                    HandoffKind::Rdma | HandoffKind::P2p | HandoffKind::Barrier
                )
            })
        })
        .expect("a cross-unit relaxable");
    let kinds: Vec<HandoffKind> = xr.alts.iter().map(|&(k, _)| k).collect();
    // RDMA is always the fallback; unified memory adds Barrier; same-die peers
    // (within domain_size) add P2P.
    assert!(kinds.contains(&HandoffKind::Rdma));
    assert!(kinds.contains(&HandoffKind::Barrier));
    assert!(kinds.contains(&HandoffKind::P2p));
    // Default is the cheapest — Barrier under unified memory.
    assert_eq!(xr.default, HandoffKind::Barrier);
    assert_eq!(
        c2.locality[&(xr.producer, xr.consumer)],
        LocalityReq::SameNode
    );
}

#[test]
fn expensive_producer_avoids_same_sm() {
    // A big GEMM feeding a row op: keeping it resident would serialize the whole
    // GEMM onto one SM (cost = its full compute) — far costlier than a round-trip,
    // so the cost-driven default is NOT a same-SM hand-off.
    let plan = LayerPlan {
        ops: vec![
            gemm("big", &["a", "w"], "y", 4096, 4096, 4096),
            row("act", &["y"], "z", 4096, 4096, 1),
        ],
    };
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let (_, c2) = collapse(&soc, &g, &cons);
    let r = relax(&c2, "y");
    assert_ne!(
        r.default,
        HandoffKind::SramSameSm,
        "expensive producer should not serialize on one SM"
    );
}

/// L2 alternative fires on H100 (per-GPC L2) — every same-unit relaxable
/// gains an `L2Local` option in addition to `Hbm` / `SramSameSm` / `Dsm`.
#[test]
fn l2_alternative_fires_on_partitioned_l2() {
    let plan = LayerPlan {
        ops: vec![
            row("norm", &["x", "nw"], "h", 512, 512, 2),
            gemm("proj", &["h", "w"], "y", 512, 512, 512),
        ],
    };
    let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let (_, c2) = collapse(&soc, &g, &cons);
    let r = relax(&c2, "h");
    let kinds: Vec<HandoffKind> = r.alts.iter().map(|&(k, _)| k).collect();
    assert!(
        kinds.contains(&HandoffKind::L2Local),
        "L2Local alternative missing on H100 (has per-GPC L2 partitioning)"
    );
}

/// On MI300X (no DSM but 8 chiplets with per-XCD L2), the L2 alternative
/// fires while DSM does not — the intended replacement for cross-CU on-chip
/// data reuse under CDNA.
#[test]
fn mi300_has_l2_but_no_dsm() {
    let mi300 = hwspec::registry::lookup("MI300X").expect("MI300X registered");
    let plan = LayerPlan {
        ops: vec![
            row("norm", &["x", "nw"], "h", 512, 512, 2),
            gemm("proj", &["h", "w"], "y", 512, 512, 512),
        ],
    };
    let soc = Soc::single(mi300, DEFAULT_PAGE_BYTES);
    let (g, cons) = assemble(&soc, &plan, SramPolicy::Stream, None).unwrap();
    let (g2, c2) = collapse(&soc, &g, &cons);
    let r = relax(&c2, "h");
    let kinds: Vec<HandoffKind> = r.alts.iter().map(|&(k, _)| k).collect();
    assert!(
        kinds.contains(&HandoffKind::L2Local),
        "L2Local should fire on MI300 (has per-XCD L2 partitioning)"
    );
    assert!(
        !kinds.contains(&HandoffKind::Dsm),
        "DSM should NOT fire on MI300 (CDNA has no cross-CU shared memory)"
    );

    // F3: every hand-off's DMA nodes must be resident iff its chosen realization
    // keeps the value on-chip (SRAM/DSM/L2Local) — i.e. an L2Local hand-off must
    // NOT round-trip HBM. Cross-check the graph's resident flags against the kind.
    let resident_of = |node: usize| match g2.nodes[node] {
        GraphNode::DmaIn { resident, .. } | GraphNode::DmaOut { resident, .. } => resident,
        _ => panic!("hand-off endpoint is not a DMA node"),
    };
    for hf in &c2.handoffs {
        let want_resident = matches!(
            hf.kind,
            HandoffKind::SramSameSm | HandoffKind::Dsm | HandoffKind::L2Local
        );
        assert_eq!(
            resident_of(hf.producer_dma_out),
            want_resident,
            "DmaOut resident flag disagrees with chosen kind {:?}",
            hf.kind
        );
        assert_eq!(
            resident_of(hf.consumer_dma_in),
            want_resident,
            "DmaIn resident flag disagrees with chosen kind {:?}",
            hf.kind
        );
    }
}
