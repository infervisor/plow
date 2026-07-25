//! Relax pass (the hybrid half of §6.4): the collapse stage picked a cost-driven
//! default per hand-off, but a same-SM (`MustColocate`) choice can be infeasible
//! once placement is considered — if the producer's resident output plus the
//! consumer's working set exceed one SM's page budget, they can't both live on
//! that SM. Here we demote such hand-offs to the next-cheapest realization from
//! their `RelaxableHandoff.alts` (DSM if available, else an HBM round-trip),
//! updating `locality`, `colocation_groups`, and the `resident` flags.

use crate::machine::Machine;
use rewrite::{ConstraintSet, GraphNode as TileNode, HandoffKind, LocalityReq, TileGraph};
use std::collections::HashMap;

/// Demote colocated hand-offs that would over-subscribe their SM's page pool.
pub fn relax(machine: &Machine, g: &TileGraph, cons: &ConstraintSet) -> (TileGraph, ConstraintSet) {
    let mut g = g.clone();
    let mut cons = cons.clone();

    // (compute, tensor) → its DmaOut / DmaIn node, to flip resident flags.
    let mut dma_out: HashMap<(usize, String), usize> = HashMap::new();
    let mut dma_in: HashMap<(usize, String), usize> = HashMap::new();
    for &(a, b) in &g.edges {
        match (&g.nodes[a], &g.nodes[b]) {
            (TileNode::Compute { .. }, TileNode::DmaOut { tensor, .. }) => {
                dma_out.insert((a, tensor.clone()), b);
            }
            (TileNode::DmaIn { tensor, .. }, TileNode::Compute { .. }) => {
                dma_in.insert((b, tensor.clone()), a);
            }
            _ => {}
        }
    }

    let mut relaxables = cons.relaxables.clone();
    let mut locality = cons.locality.clone();
    let mut flips: Vec<(usize, bool)> = Vec::new(); // (dma node, resident)

    for r in &mut relaxables {
        if r.default != HandoffKind::SramSameSm {
            continue;
        }
        let unit = *cons.placement.get(&r.producer).unwrap_or(&0);
        let budget = machine.unit(unit).pages_per_sm;
        let need = cons
            .sram_pages
            .get(&r.producer)
            .copied()
            .unwrap_or(0)
            .saturating_add(cons.sram_pages.get(&r.consumer).copied().unwrap_or(0));
        if need <= budget {
            continue; // the colocated pair fits — keep the default.
        }
        // Demote to the cheapest non-same-SM alternative (DSM preferred over HBM).
        let next = r
            .alts
            .iter()
            .filter(|&&(k, _)| k != HandoffKind::SramSameSm)
            .min_by_key(|&&(_, c)| c)
            .map(|&(k, _)| k)
            .unwrap_or(HandoffKind::Hbm);
        r.default = next;
        let (resident, req) = match next {
            HandoffKind::Dsm => (true, LocalityReq::SameDomain),
            // L2Local is on-chip resident (no HBM round-trip), like Dsm but scoped
            // to an L2 slice — must match `collapse::realize`.
            HandoffKind::L2Local => (true, LocalityReq::SameL2Partition),
            _ => (false, LocalityReq::NoConstraint),
        };
        locality.insert((r.producer, r.consumer), req);
        if let Some(&n) = dma_out.get(&(r.producer, r.tensor.clone())) {
            flips.push((n, resident));
        }
        if let Some(&n) = dma_in.get(&(r.consumer, r.tensor.clone())) {
            flips.push((n, resident));
        }
    }

    for (n, resident) in flips {
        match &mut g.nodes[n] {
            TileNode::DmaIn { resident: res, .. } | TileNode::DmaOut { resident: res, .. } => {
                *res = resident
            }
            _ => {}
        }
    }
    // Rebuild colocation from the hand-offs that stayed same-SM.
    let mut uf = UnionFind::default();
    for r in &relaxables {
        if r.default == HandoffKind::SramSameSm {
            uf.union(r.producer, r.consumer);
        }
    }
    cons.relaxables = relaxables;
    cons.locality = locality;
    cons.colocation_groups = uf.groups();
    (g, cons)
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
