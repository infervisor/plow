//! Cost-driven hand-off lowering (design §2.5) over the assembled tile graph.
//!
//! Each same-unit producer→consumer hand-off has three realizations — an HBM
//! round-trip, a same-SM SRAM hand-off (no round-trip but serialized), or a DSM
//! hand-off (cross-SM in one GPC, no round-trip and parallel). The tile-level
//! egglog rules ([`egl/tile.egg`]) enumerate the available alternatives
//! (`SramHandoff` always; `DsmHandoff` only on DSM units); a cost extractor here
//! picks the cheapest **default** with [`costmodel::handoff_costs`] and records
//! every option as a [`RelaxableHandoff`] the scheduler may flip (§6.4).
//!
//! The collapse is *flag-based*: it sets the hand-off's `resident` flags, rebuilds
//! `colocation_groups` (only SRAM hand-offs co-locate), and fills `locality` +
//! `relaxables`. The graph shape is unchanged, so the scheduler/`expand` (which
//! already honor resident hand-offs + colocation) consume it directly.

use crate::tilegraph::{
    Compute, ConstraintSet, HandoffKind, LocalityReq, OpKind, RelaxableHandoff, TileGraph, TileNode,
};
use costmodel::{cost, handoff_costs, xunit_handoff_costs, Cycles, Soc, UnitId};
use std::collections::{HashMap, HashSet};

const TILE_SCHEMA: &str = include_str!("egl/tile.egg");
const ELEM: u64 = 2;

#[derive(thiserror::Error, Debug)]
pub enum CollapseError {
    #[error("tile-egglog error: {0}")]
    Egglog(String),
}

/// Apply cost-driven hand-off lowering to `(g, cons)`, returning the rewritten
/// graph + constraints (`resident` flags, `colocation_groups`, `locality`,
/// `relaxables`).
///
/// Infallible wrapper around [`try_collapse`], kept for the existing callers:
/// a tile-egglog failure is a compiler bug, so it panics rather than silently
/// falling back to HBM/RDMA for every hand-off.
pub fn collapse(soc: &Soc, g: &TileGraph, cons: &ConstraintSet) -> (TileGraph, ConstraintSet) {
    try_collapse(soc, g, cons).expect("hand-off collapse failed")
}

/// Fallible [`collapse`]: surfaces tile-egglog errors instead of swallowing them.
pub fn try_collapse(
    soc: &Soc,
    g: &TileGraph,
    cons: &ConstraintSet,
) -> Result<(TileGraph, ConstraintSet), CollapseError> {
    let mut g = g.clone();
    let mut cons = cons.clone();

    // producer compute of each DmaOut node; consumer compute of each DmaIn node.
    let mut producer_of: HashMap<usize, usize> = HashMap::new();
    let mut consumer_of: HashMap<usize, usize> = HashMap::new();
    for &(a, b) in &g.edges {
        match (&g.nodes[a], &g.nodes[b]) {
            (TileNode::Compute { .. }, TileNode::DmaOut { .. }) => {
                producer_of.insert(b, a);
            }
            (TileNode::DmaIn { .. }, TileNode::Compute { .. }) => {
                consumer_of.insert(a, b);
            }
            _ => {}
        }
    }

    // A producer→consumer hand-off edge, same-unit or cross-unit.
    struct Ho {
        producer: usize,
        consumer: usize,
        dma_out: usize,
        dma_in: usize,
        tensor: String,
        /// Producer unit; for a same-unit hand-off the consumer shares it.
        unit: UnitId,
        /// Consumer unit (== `unit` for same-unit hand-offs).
        consumer_unit: UnitId,
        bytes: u64,
        prod_cycles: Cycles,
    }
    let (mut hos, mut xhos): (Vec<Ho>, Vec<Ho>) = (Vec::new(), Vec::new());
    for hf in &cons.handoffs {
        let (&producer, &consumer) = match (
            producer_of.get(&hf.producer_dma_out),
            consumer_of.get(&hf.consumer_dma_in),
        ) {
            (Some(p), Some(c)) => (p, c),
            _ => continue,
        };
        let unit = cons.placement[&producer];
        let consumer_unit = cons.placement.get(&consumer).copied().unwrap_or(unit);
        let desc = &cons.op_io[&producer];
        let bytes = output_bytes(&desc.kind);
        let prod_cycles = op_cycles(soc, unit, &desc.kind, compute_kind(&g, producer));
        let ho = Ho {
            producer,
            consumer,
            dma_out: hf.producer_dma_out,
            dma_in: hf.consumer_dma_in,
            tensor: desc.output.clone(),
            unit,
            consumer_unit,
            bytes,
            prod_cycles,
        };
        if hf.cross_unit {
            xhos.push(ho);
        } else {
            hos.push(ho);
        }
    }

    // Which producers got SRAM / DSM / L2Local alternatives, per the egglog rules.
    let same_unit_alts = discover_alternatives(
        soc,
        &hos.iter().map(|h| (h.producer, h.unit)).collect::<Vec<_>>(),
    )?;
    let sram = &same_unit_alts.sram;
    let dsm = &same_unit_alts.dsm;
    let l2 = &same_unit_alts.l2;
    // Which cross-unit (producer, consumer-unit) pairs got Barrier / P2P / RDMA
    // alternatives — keyed per pair, since link class depends on both ends.
    let xalts = discover_xunit_alternatives(
        soc,
        &xhos
            .iter()
            .map(|h| (h.producer, h.unit, h.consumer_unit))
            .collect::<Vec<_>>(),
    )?;

    // Cost-pick a default per hand-off and apply it.
    cons.relaxables.clear();
    cons.locality.clear();
    let mut colo = UnionFind::default();
    // Chosen realization per hand-off, keyed by (dma_out, dma_in), written back
    // onto `cons.handoffs` after the picks so expansion can branch on it.
    let mut chosen_kind: HashMap<(usize, usize), HandoffKind> = HashMap::new();

    // --- same-unit hand-offs (HBM / SRAM / DSM / L2Local) ---
    for h in &hos {
        let spec = soc.unit(h.unit).cm.spec;
        let c = handoff_costs(spec, h.prod_cycles, h.bytes);
        let label = h.producer.to_string();
        let mut alts = vec![(HandoffKind::Hbm, c.hbm)];
        if sram.contains(&label) {
            alts.push((HandoffKind::SramSameSm, c.sram_same_sm));
        }
        if dsm.contains(&label) {
            alts.push((HandoffKind::Dsm, c.dsm));
        }
        // L2Local: the egglog `L2Handoff` rule fires only when the unit has
        // `L2Partitioning` — that's the source of truth. See the design notes.
        if l2.contains(&label) {
            alts.push((HandoffKind::L2Local, c.l2_local));
        }
        let default = cheapest(&alts);
        let (resident, req) = realize(default);
        set_resident(&mut g, h.dma_out, resident);
        set_resident(&mut g, h.dma_in, resident);
        cons.locality.insert((h.producer, h.consumer), req);
        chosen_kind.insert((h.dma_out, h.dma_in), default);
        if default == HandoffKind::SramSameSm {
            colo.union(h.producer, h.consumer);
        }
        cons.relaxables.push(RelaxableHandoff {
            producer: h.producer,
            consumer: h.consumer,
            tensor: h.tensor.clone(),
            default,
            alts,
        });
    }

    // --- cross-unit (cross-node) hand-offs (Barrier / P2P / RDMA) ---
    for h in &xhos {
        let spec = soc.unit(h.unit).cm.spec;
        let c = xunit_handoff_costs(spec, h.bytes);
        // Availability depends on the (producer, consumer-unit) pair: a near
        // consumer may have P2P while a far one only has RDMA.
        let label = xunit_label(h.producer, h.consumer_unit);
        // RDMA is always available (the cross-node / slow-link fallback).
        let mut alts = vec![(HandoffKind::Rdma, c.rdma)];
        if xalts.p2p.contains(&label) && c.p2p != Cycles::MAX {
            alts.push((HandoffKind::P2p, c.p2p));
        }
        if xalts.barrier.contains(&label) {
            alts.push((HandoffKind::Barrier, c.barrier));
        }
        let default = cheapest(&alts);
        let (_resident, req) = realize(default);
        // Cross-unit data still flows through the DmaIn/DmaOut (the scheduler
        // routes it as P2P/RDMA); the locality req records the node constraint.
        cons.locality.insert((h.producer, h.consumer), req);
        chosen_kind.insert((h.dma_out, h.dma_in), default);
        cons.relaxables.push(RelaxableHandoff {
            producer: h.producer,
            consumer: h.consumer,
            tensor: h.tensor.clone(),
            default,
            alts,
        });
    }
    // Record the chosen realization on each hand-off so expansion can branch on
    // it (e.g. a `Barrier` fence moves no data). Untouched hand-offs keep `Hbm`.
    for hf in &mut cons.handoffs {
        if let Some(&k) = chosen_kind.get(&(hf.producer_dma_out, hf.consumer_dma_in)) {
            hf.kind = k;
        }
    }

    // Only SRAM hand-offs co-locate now (the cost-driven subset, not all same-unit).
    cons.colocation_groups = colo.groups();

    fold_boundary_loads(&mut g, &cons);
    Ok((g, cons))
}

/// The `Comp` label for one cross-unit hand-off: the producer keyed by the
/// consumer's unit, so each (producer, consumer-unit) pair gets its own term
/// and its own alternative set.
fn xunit_label(producer: usize, consumer_unit: UnitId) -> String {
    format!("{producer}@{consumer_unit}")
}

/// The cheapest realization in a non-empty alternatives list.
fn cheapest(alts: &[(HandoffKind, Cycles)]) -> HandoffKind {
    alts.iter()
        .min_by_key(|&&(_, cost)| cost)
        .map(|&(k, _)| k)
        .unwrap()
}

/// The `resident` flag and placement requirement a chosen realization imposes.
fn realize(kind: HandoffKind) -> (bool, LocalityReq) {
    match kind {
        HandoffKind::Hbm => (false, LocalityReq::NoConstraint),
        HandoffKind::SramSameSm => (true, LocalityReq::MustColocate),
        HandoffKind::Dsm => (true, LocalityReq::SameDomain),
        // L2Local: producer's L2 write stays in the partition for the consumer
        // to read — on-chip resident (no HBM round-trip), exactly like `Dsm` but
        // scoped to an L2 slice instead of a DSM domain. `resident = true` skips
        // the DmaOut/DmaIn round-trip; `SameL2Partition` (handled by the same
        // domain-pin path as `SameDomain`) keeps producer and consumer on one L2
        // slice. Marking it non-resident made expansion emit an ordinary HBM DMA,
        // contradicting the "no round-trip" the cost model prices it for.
        HandoffKind::L2Local => (true, LocalityReq::SameL2Partition),
        // Cross-unit: data still moves over the fabric (not SRAM-resident); the
        // requirement is which fabric tier the placement must keep them on.
        HandoffKind::Barrier | HandoffKind::P2p => (false, LocalityReq::SameNode),
        HandoffKind::Rdma => (false, LocalityReq::NoConstraint),
    }
}

/// dma-fold (prologue): a DRAM load (a weight / graph input — no producer) read
/// by exactly one compute is folded into that kernel, which issues its own TMA
/// load. Shared loads (fan-out > 1) stay separate so the value is staged once.
fn fold_boundary_loads(g: &mut TileGraph, cons: &ConstraintSet) {
    // Tensors produced by some compute (so NOT external DRAM), and consumer counts.
    let mut produced: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut consumers: HashMap<String, usize> = HashMap::new();
    for desc in cons.op_io.values() {
        produced.insert(desc.output.clone());
        for inp in &desc.inputs {
            *consumers.entry(inp.clone()).or_insert(0) += 1;
        }
    }
    for (nid, node) in g.nodes.iter_mut().enumerate() {
        let TileNode::Compute { inline_in, .. } = node else {
            continue;
        };
        let Some(desc) = cons.op_io.get(&nid) else {
            continue;
        };
        for inp in &desc.inputs {
            if !produced.contains(inp) && consumers.get(inp).copied().unwrap_or(0) == 1 {
                inline_in.push(inp.clone());
            }
        }
    }
}

fn set_resident(g: &mut TileGraph, node: usize, value: bool) {
    match &mut g.nodes[node] {
        TileNode::DmaIn { resident, .. } | TileNode::DmaOut { resident, .. } => *resident = value,
        _ => {}
    }
}

fn compute_kind(g: &TileGraph, node: usize) -> Compute {
    match &g.nodes[node] {
        TileNode::Compute { kind, .. } => *kind,
        _ => Compute::Layout,
    }
}

fn output_bytes(kind: &OpKind) -> u64 {
    let elems = match kind {
        OpKind::Gemm(g) => g.m * g.n,
        OpKind::Row(r) => r.rows * r.feat,
        OpKind::Model(m) => m.rows * m.feat,
        OpKind::Flash(a) => a.heads * a.seq_q * a.head_dim,
        OpKind::Layout(s) => return s.bytes,
    };
    elems.max(0) as u64 * ELEM
}

/// The producer op's compute cost (the serialization a same-SM hand-off imposes).
fn op_cycles(soc: &Soc, unit: UnitId, kind: &OpKind, tile: Compute) -> Cycles {
    let cm = &soc.unit(unit).cm;
    match (kind, tile) {
        (OpKind::Gemm(g), Compute::Gemm(t)) => cm.gemm_cost(*g, t),
        (OpKind::Flash(a), Compute::Flash(t)) => cm.flash_cost(*a, t),
        (OpKind::Row(r), _) => cm.row_cost(*r),
        (OpKind::Model(m), _) => cm.row_cost(m.row_shape()),
        (OpKind::Layout(s), _) => cm.layout_cost(s.bytes, false),
        _ => cost::macs_cycles(cm.spec, 1, costmodel::MmaDtype::Bf16),
    }
}

/// Run the tile-egglog rules and report which producers (by label) gained
/// each same-unit alternative: `SramHandoff`, `DsmHandoff`, `L2Handoff`.
/// DSM appears only for units asserted `DsmUnit` (spec has a DSM domain);
/// L2Handoff appears only for units asserted `L2Partitioned` (spec has L2
/// partitioning) — so the rules, not Rust, decide availability.
#[derive(Default)]
struct SameUnitAlts {
    sram: HashSet<String>,
    dsm: HashSet<String>,
    l2: HashSet<String>,
}

fn discover_alternatives(
    soc: &Soc,
    handoffs: &[(usize, UnitId)],
) -> Result<SameUnitAlts, CollapseError> {
    if handoffs.is_empty() {
        return Ok(SameUnitAlts::default());
    }
    let mut prog = String::from(TILE_SCHEMA);
    prog.push('\n');
    // Units with a DSM domain.
    for u in 0..soc.units.len() {
        if soc.unit(u).cm.spec.dsm.is_some() {
            prog.push_str(&format!("(DsmUnit {u})\n"));
        }
        if soc.unit(u).cm.spec.l2_partitioning.is_some() {
            prog.push_str(&format!("(L2Partitioned {u})\n"));
        }
    }
    // One round-trip term per (producer, unit); identical ones hash-cons.
    let mut seen = HashSet::new();
    for &(p, u) in handoffs {
        if seen.insert(p) {
            prog.push_str(&format!(
                "(let h{p} (Load (Store (Comp \"{p}\" {u}) {u}) {u}))\n"
            ));
        }
    }
    prog.push_str(
        "(run 5)\n\
         (print-function SramHandoff 1000000)\n\
         (print-function DsmHandoff 1000000)\n\
         (print-function L2Handoff 1000000)\n",
    );

    let mut egraph = egglog::EGraph::default();
    let msgs = egraph
        .parse_and_run_program(None, &prog)
        .map_err(|e| CollapseError::Egglog(e.to_string()))?;
    let mut alts = SameUnitAlts::default();
    for m in &msgs {
        let s = m.to_string();
        if s.contains("SramHandoff") {
            alts.sram.extend(comp_labels(&s));
        }
        if s.contains("DsmHandoff") {
            alts.dsm.extend(comp_labels(&s));
        }
        if s.contains("L2Handoff") {
            alts.l2.extend(comp_labels(&s));
        }
    }
    Ok(alts)
}

/// Which cross-unit producers (by label) gained the Barrier / P2P / RDMA
/// alternatives. RDMA fires for every cross-unit hand-off (the fallback); Barrier
/// only under [`Unified`](costmodel::MemoryModel) memory; P2P only between units
/// on the fast peer fabric within one node-domain (`≤ domain_size` apart). The
/// egglog rules, not Rust, decide availability from the asserted facts.
#[derive(Default)]
struct XunitAlts {
    barrier: HashSet<String>,
    p2p: HashSet<String>,
    rdma: HashSet<String>,
}

fn discover_xunit_alternatives(
    soc: &Soc,
    handoffs: &[(usize, UnitId, UnitId)],
) -> Result<XunitAlts, CollapseError> {
    let mut alts = XunitAlts::default();
    if handoffs.is_empty() {
        return Ok(alts);
    }
    let mut prog = String::from(TILE_SCHEMA);
    prog.push('\n');
    if soc.memory.unified {
        prog.push_str("(Unified)\n");
    }
    // FastLink for unit pairs on the fast fabric within one domain. With no real
    // topology, unit-id distance proxies for fabric proximity: a pair is fast iff
    // both ends have a peer interconnect and are `< domain_size` apart.
    for &(_, a, b) in handoffs {
        let sa = soc.unit(a).cm.spec;
        let sb = soc.unit(b).cm.spec;
        let domain = sa.interconnect.map(|ic| ic.domain_size).unwrap_or(0) as i64;
        let near = (a as i64 - b as i64).abs() < domain;
        if sa.interconnect.is_some() && sb.interconnect.is_some() && near {
            prog.push_str(&format!("(FastLink {a} {b})\n"));
        }
    }
    // One cross-unit base term per (producer, consumer-unit) pair — link class
    // (FastLink → P2P legality) depends on both ends, so a producer with a near
    // and a far consumer needs two distinctly-labeled terms.
    let mut seen = HashSet::new();
    for &(p, a, b) in handoffs {
        if seen.insert((p, b)) {
            let label = xunit_label(p, b);
            prog.push_str(&format!(
                "(let x{p}_{b} (CrossLoad (CrossStore (Comp \"{label}\" {a}) {a}) {a} {b}))\n"
            ));
        }
    }
    prog.push_str(
        "(run 5)\n\
         (print-function BarrierHandoff 1000000)\n\
         (print-function P2PHandoff 1000000)\n\
         (print-function RdmaHandoff 1000000)\n",
    );

    let mut egraph = egglog::EGraph::default();
    let msgs = egraph
        .parse_and_run_program(None, &prog)
        .map_err(|e| CollapseError::Egglog(e.to_string()))?;
    for m in &msgs {
        let s = m.to_string();
        if s.contains("BarrierHandoff") {
            alts.barrier.extend(comp_labels(&s));
        }
        if s.contains("P2PHandoff") {
            alts.p2p.extend(comp_labels(&s));
        }
        if s.contains("RdmaHandoff") {
            alts.rdma.extend(comp_labels(&s));
        }
    }
    Ok(alts)
}

/// Extract every `(Comp "LABEL" …)` label from an egglog dump.
fn comp_labels(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = s;
    while let Some(i) = rest.find("(Comp \"") {
        rest = &rest[i + "(Comp \"".len()..];
        if let Some(j) = rest.find('"') {
            out.push(rest[..j].to_string());
            rest = &rest[j + 1..];
        } else {
            break;
        }
    }
    out
}

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

    // The physical realization → (resident, locality) contract. L2Local must be
    // on-chip resident (no HBM round-trip), like Dsm, but scoped to an L2 slice
    // (F3): non-resident made expansion emit an ordinary HBM DMA.
    #[test]
    fn realize_residency_and_locality() {
        assert_eq!(
            realize(HandoffKind::Hbm),
            (false, LocalityReq::NoConstraint)
        );
        assert_eq!(
            realize(HandoffKind::SramSameSm),
            (true, LocalityReq::MustColocate)
        );
        assert_eq!(realize(HandoffKind::Dsm), (true, LocalityReq::SameDomain));
        // The fix: resident (no round-trip) + partition-scoped, mirroring Dsm.
        assert_eq!(
            realize(HandoffKind::L2Local),
            (true, LocalityReq::SameL2Partition)
        );
    }
}
