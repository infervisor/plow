//! Per-resource occupancy state (design §5.2): one exclusive [`IntervalSet`] per
//! SM / DMA engine / DPU / host thread, plus capacity [`BandwidthSet`]s for SRAM
//! pages (per SM), HBM (per unit) and the interconnect.

use crate::interval::{BandwidthSet, Cycle, IntervalSet};
use crate::machine::{Machine, SmId};
use costmodel::UnitId;

/// A handle to one schedulable resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResourceId {
    Sm(UnitId, SmId),
    Dma(UnitId, usize),
    Dpu(usize),
    Host(usize),
}

/// A single SM's SRAM, tracked **per page**: one exclusive timeline per page
/// slot (design §5.3). Allocating a tile's working set = assigning it specific
/// free slots over its live interval = linear-scan register allocation (§8.4).
#[derive(Clone, Debug)]
pub struct PagePool {
    slots: Vec<IntervalSet>,
}

impl PagePool {
    pub fn new(pages: u64) -> PagePool {
        PagePool {
            slots: vec![IntervalSet::new(); pages as usize],
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Assign `need` specific page slots, each free over `[start, end)`. Returns
    /// the chosen slot ids, or `None` if fewer than `need` are free (a spill).
    pub fn allocate(&mut self, start: Cycle, end: Cycle, need: u64) -> Option<Vec<usize>> {
        if need == 0 {
            return Some(Vec::new());
        }
        let end = end.max(start + 1);
        let mut chosen = Vec::with_capacity(need as usize);
        for (i, s) in self.slots.iter().enumerate() {
            if !s.overlaps(start, end) {
                chosen.push(i);
                if chosen.len() == need as usize {
                    break;
                }
            }
        }
        if chosen.len() < need as usize {
            return None;
        }
        for &i in &chosen {
            self.slots[i].reserve(start, end - start);
        }
        Some(chosen)
    }

    /// Allocate output pages over `[start, end_live)` and transient working-set
    /// pages over `[start, end_compute)`, both from distinct slots. Returns the
    /// output slot ids (working-set slots are transient and not tracked after the
    /// compute interval ends), or `None` if insufficient capacity (a spill).
    ///
    /// This guarantees that during `[start, end_compute)` the SM has enough room
    /// for *both* groups simultaneously — preventing the bug where working-set
    /// staging (A/B tiles) overflows when live output pages from earlier tiles
    /// consume most of the page pool.
    pub fn allocate_with_working_set(
        &mut self,
        start: Cycle,
        end_compute: Cycle,
        working_pages: u64,
        end_live: Cycle,
        out_pages: u64,
    ) -> Option<Vec<usize>> {
        if out_pages == 0 && working_pages == 0 {
            return Some(Vec::new());
        }
        let end_live = end_live.max(start + 1);
        let end_compute = end_compute.max(start + 1);
        // First pass: find slots free over the longer interval [start, end_live).
        // These hold the output pages that persist until the last consumer.
        let mut out_slots = Vec::with_capacity(out_pages as usize);
        let mut used = vec![false; self.slots.len()];
        if out_pages > 0 {
            for (i, s) in self.slots.iter().enumerate() {
                if !s.overlaps(start, end_live) {
                    out_slots.push(i);
                    used[i] = true;
                    if out_slots.len() == out_pages as usize {
                        break;
                    }
                }
            }
            if out_slots.len() < out_pages as usize {
                return None;
            }
        }
        // Second pass: find slots free during [start, end_compute) not already
        // claimed by the output group — these hold the transient A/B staging.
        let mut ws_slots = Vec::new();
        if working_pages > 0 {
            for (i, s) in self.slots.iter().enumerate() {
                if !used[i] && !s.overlaps(start, end_compute) {
                    ws_slots.push(i);
                    if ws_slots.len() == working_pages as usize {
                        break;
                    }
                }
            }
            if ws_slots.len() < working_pages as usize {
                return None;
            }
        }
        // Commit both groups atomically.
        for &i in &out_slots {
            self.slots[i].reserve(start, end_live - start);
        }
        for &i in &ws_slots {
            self.slots[i].reserve(start, end_compute - start);
        }
        Some(out_slots)
    }
}

/// All resource timelines for one node. The hierarchy is unit → SM → page:
/// `sm` is the SM-level exclusive timeline, `sram_pool` the per-page slots.
#[derive(Clone, Debug)]
pub struct ResourceState {
    sm: Vec<Vec<IntervalSet>>,  // [unit][sm]    exclusive (whole SM)
    dma: Vec<Vec<IntervalSet>>, // [unit][engine] exclusive
    dpu: Vec<IntervalSet>,      // node-level RDMA / collective engines
    host: Vec<IntervalSet>,     // node-level CPU thread pool
    hbm: Vec<BandwidthSet>,     // [unit]        capacity
    link: BandwidthSet,         // aggregate interconnect capacity
}

impl ResourceState {
    pub fn new(m: &Machine) -> ResourceState {
        ResourceState {
            sm: m
                .units
                .iter()
                .map(|u| vec![IntervalSet::new(); u.sm_count])
                .collect(),
            dma: m
                .units
                .iter()
                .map(|u| vec![IntervalSet::new(); u.dma_engines])
                .collect(),
            dpu: vec![IntervalSet::new(); m.dpu_engines],
            host: vec![IntervalSet::new(); m.host_threads],
            hbm: m.units.iter().map(|_| BandwidthSet::new()).collect(),
            link: BandwidthSet::new(),
        }
    }

    fn exclusive(&self, r: ResourceId) -> &IntervalSet {
        match r {
            ResourceId::Sm(u, s) => &self.sm[u][s],
            ResourceId::Dma(u, e) => &self.dma[u][e],
            ResourceId::Dpu(i) => &self.dpu[i],
            ResourceId::Host(i) => &self.host[i],
        }
    }

    fn exclusive_mut(&mut self, r: ResourceId) -> &mut IntervalSet {
        match r {
            ResourceId::Sm(u, s) => &mut self.sm[u][s],
            ResourceId::Dma(u, e) => &mut self.dma[u][e],
            ResourceId::Dpu(i) => &mut self.dpu[i],
            ResourceId::Host(i) => &mut self.host[i],
        }
    }

    /// Earliest start ≥ `after` for a `dur`-long exclusive hold on `r`.
    pub fn earliest_free(&self, r: ResourceId, after: Cycle, dur: Cycle) -> Cycle {
        self.exclusive(r).earliest_free(after, dur)
    }

    /// Reserve the exclusive hold (caller used [`Self::earliest_free`]).
    pub fn reserve(&mut self, r: ResourceId, start: Cycle, dur: Cycle) {
        self.exclusive_mut(r).reserve(start, dur);
    }

    /// Capacity check + reserve helpers for HBM / interconnect bandwidth.
    pub fn hbm_ok(&self, unit: UnitId, start: Cycle, end: Cycle, w: f64, limit: f64) -> bool {
        self.hbm[unit].capacity_ok(start, end, w, limit)
    }
    pub fn reserve_hbm(&mut self, unit: UnitId, start: Cycle, end: Cycle, w: f64) {
        self.hbm[unit].reserve(start, end, w);
    }
    pub fn link_ok(&self, start: Cycle, end: Cycle, w: f64, limit: f64) -> bool {
        self.link.capacity_ok(start, end, w, limit)
    }
    pub fn reserve_link(&mut self, start: Cycle, end: Cycle, w: f64) {
        self.link.reserve(start, end, w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_allocate_share_and_spill() {
        let mut pool = PagePool::new(4); // 4 page slots
                                         // Tile A holds 2 pages over [0,10).
        let a = pool.allocate(0, 10, 2).expect("A fits");
        assert_eq!(a.len(), 2);
        // Tile B holds 2 pages over [5,15): overlaps A, must take the *other* 2 slots.
        let b = pool.allocate(5, 15, 2).expect("B fits");
        assert!(
            a.iter().all(|p| !b.contains(p)),
            "overlapping tiles share a page"
        );
        // Tile C wants 2 pages over [6,8): pool is full while A and B are live → spill.
        assert!(pool.allocate(6, 8, 2).is_none(), "should spill when full");
        // Tile D after both A and B end reuses freed slots.
        let d = pool.allocate(20, 30, 4).expect("D fits after A,B free");
        assert_eq!(d.len(), 4);
    }
}
