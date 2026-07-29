//! The node the scheduler targets: the `Soc`'s compute units (GPUs/NPUs, each a
//! set of SMs with an SRAM budget and HBM bandwidth) plus node-level **DPU**
//! engines (cross-unit/cross-node RDMA + collectives) and a **host CPU thread**
//! pool (coordination, routing, staging). DPUs and host threads are not `Soc`
//! compute units — they are added here on top of the `Soc`.

use crate::config::Config;
use crate::interval::Cycle;
use costmodel::{Soc, UnitId, TMEM_COL_BYTES};

pub type SmId = usize;

/// Interconnect class between two units.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkClass {
    /// One coherent address space — a cross-unit read is a barrier, not a copy.
    Unified,
    /// Fast switched fabric (NVLink / Infinity Fabric).
    Fast,
    /// Slow link (PCIe / inter-node) — routed over a DPU.
    Slow,
}

/// Per-unit hardware the scheduler tracks.
#[derive(Clone, Debug)]
pub struct UnitHw {
    pub id: UnitId,
    pub sm_count: usize,
    pub pages_per_sm: u64,
    /// Tensor-Memory columns per SM (0 on architectures without TMEM). The MMA
    /// accumulator of a tile must fit here on Blackwell datacenter parts.
    pub tmem_cols_per_sm: u64,
    /// DSM (distributed-shared-memory) grouping: how many GPC domains the SMs
    /// split into, and how many SMs per domain. `(1, sm_count)` when there is no
    /// DSM (the whole unit is one trivial domain).
    pub dsm_domains: usize,
    pub sms_per_domain: usize,
    /// Chiplet / L2-domain grouping: how many chiplets the SMs split into, and
    /// how many SMs per chiplet. `(1, sm_count)` on monolithic dies.
    pub chiplet_count: usize,
    pub sms_per_chiplet: usize,
    /// L2 partitioning: how many L2 slices the die exposes, and how many SMs
    /// per slice. `(1, sm_count)` on unified-L2 archs.
    ///
    /// On H100 this coincides with `dsm_domains` (per-GPC L2 == GPC).
    /// On MI300 this coincides with `chiplet_count` (per-XCD L2 == chiplet).
    /// A future part could decouple all three.
    pub l2_partitions: usize,
    pub sms_per_l2_partition: usize,
    pub dma_engines: usize,
    /// HBM bandwidth as bytes movable per cycle (the capacity limit).
    pub hbm_bytes_per_cycle: f64,
}

/// The whole node.
#[derive(Clone, Debug)]
pub struct Machine {
    pub units: Vec<UnitHw>,
    pub dpu_engines: usize,
    pub host_threads: usize,
    pub unified_memory: bool,
    /// Whether units have a fast peer fabric (NVLink / Infinity Fabric). When
    /// false, a non-unified cross-unit transfer is a slow (PCIe) link.
    pub has_fast_interconnect: bool,
    /// Interconnect bandwidth (bytes/cycle) for cross-unit transfers.
    pub link_bytes_per_cycle: f64,
}

/// Bytes movable per cycle from a GBps figure and a clock.
fn bytes_per_cycle(gbps: f64, clock_hz: u64) -> f64 {
    if clock_hz == 0 {
        return f64::INFINITY;
    }
    gbps * 1.0e9 / clock_hz as f64
}

impl Machine {
    /// Build the node model from a `Soc` and the scheduler `Config`.
    pub fn from_soc(soc: &Soc, cfg: &Config) -> Machine {
        let units = soc
            .units
            .iter()
            .map(|u| {
                let spec = u.cm.spec;
                let (dsm_domains, sms_per_domain) = match spec.dsm {
                    Some(d) => (d.domain_count as usize, d.sms_per_domain as usize),
                    None => (1, spec.sm_count as usize),
                };
                let (chiplet_count, sms_per_chiplet) = match spec.chiplet {
                    Some(c) => (c.chiplet_count as usize, c.sms_per_chiplet as usize),
                    None => (1, spec.sm_count as usize),
                };
                let (l2_partitions, sms_per_l2_partition) = match spec.l2_partitioning {
                    Some(l) => (l.partition_count as usize, l.sms_per_partition as usize),
                    None => (1, spec.sm_count as usize),
                };
                UnitHw {
                    id: u.id,
                    sm_count: spec.sm_count as usize,
                    pages_per_sm: u.cm.sram.pages_per_sm,
                    tmem_cols_per_sm: spec.sm.tmem.0 / TMEM_COL_BYTES,
                    dsm_domains,
                    sms_per_domain,
                    chiplet_count,
                    sms_per_chiplet,
                    l2_partitions,
                    sms_per_l2_partition,
                    // Hardware copy-engine count from the spec; Config overrides
                    // when non-zero (0 ⇒ use the GPU spec).
                    dma_engines: if cfg.dma_engines > 0 {
                        cfg.dma_engines
                    } else {
                        spec.copy_engines
                    }
                    .max(1) as usize,
                    // Datasheet peak: the scheduler uses this to compare placements,
                    // where a constant derate cancels. A REPORTED absolute floor must
                    // use `spec.mem.bandwidth_for_bound()` — see `plowc --lean-oracle`.
                    hbm_bytes_per_cycle: bytes_per_cycle(spec.mem.bandwidth.0, spec.clock_boost.0),
                }
            })
            .collect();
        // Interconnect: the unit-0 spec is the node's fabric. Bandwidth comes
        // from its `interconnect` (Config overrides when `link_gbps > 0`); a
        // spec with no fast fabric and no override ⇒ 0 ⇒ the slow-link path.
        let unit0 = soc.units.first().map(|u| u.cm.spec);
        let clock = unit0.map(|s| s.clock_boost.0).unwrap_or(1);
        let fabric = unit0.and_then(|s| s.interconnect);
        let link_gbps = if cfg.link_gbps > 0.0 {
            cfg.link_gbps
        } else {
            fabric.map(|ic| ic.per_gpu_bandwidth.0).unwrap_or(0.0)
        };
        Machine {
            units,
            dpu_engines: cfg.dpu_engines.max(1) as usize,
            host_threads: cfg.host_threads.max(1) as usize,
            unified_memory: soc.memory.unified,
            has_fast_interconnect: fabric.is_some(),
            link_bytes_per_cycle: bytes_per_cycle(link_gbps, clock),
        }
    }

    pub fn unit(&self, id: UnitId) -> &UnitHw {
        debug_assert!(
            id < self.units.len(),
            "UnitId {id} out of bounds (have {})",
            self.units.len()
        );
        &self.units[id]
    }

    /// The SM indices belonging to DSM domain `domain` of `unit` (clamped to the
    /// enabled SM count).
    pub fn domain_sms(&self, unit: UnitId, domain: usize) -> std::ops::Range<usize> {
        let u = self.unit(unit);
        let start = (domain * u.sms_per_domain).min(u.sm_count);
        let end = (start + u.sms_per_domain).min(u.sm_count);
        start..end
    }

    /// Which chiplet (L2 domain) a given SM belongs to.
    pub fn chiplet_of(&self, unit: UnitId, sm: SmId) -> usize {
        let u = self.unit(unit);
        if u.chiplet_count <= 1 {
            return 0;
        }
        sm / u.sms_per_chiplet.max(1)
    }

    /// Which L2 partition a given SM's data lives in. On H100 this coincides
    /// with the GPC / DSM domain; on MI300 with the chiplet.
    pub fn l2_partition_of(&self, unit: UnitId, sm: SmId) -> usize {
        let u = self.unit(unit);
        if u.l2_partitions <= 1 {
            return 0;
        }
        sm / u.sms_per_l2_partition.max(1)
    }

    /// The SM index range belonging to L2 partition `partition` on `unit`.
    /// The list scheduler uses this to prefer placement within a pinned
    /// partition (SameL2Partition locality).
    pub fn l2_partition_sms(&self, unit: UnitId, partition: usize) -> std::ops::Range<usize> {
        let u = self.unit(unit);
        let start = (partition * u.sms_per_l2_partition).min(u.sm_count);
        let end = (start + u.sms_per_l2_partition).min(u.sm_count);
        start..end
    }

    /// The SM indices belonging to chiplet `chiplet` of `unit` (clamped to the
    /// enabled SM count).
    pub fn chiplet_sms(&self, unit: UnitId, chiplet: usize) -> std::ops::Range<usize> {
        let u = self.unit(unit);
        let start = (chiplet * u.sms_per_chiplet).min(u.sm_count);
        let end = (start + u.sms_per_chiplet).min(u.sm_count);
        start..end
    }

    /// Unified locality domain for soft scheduling affinity. Returns the
    /// locality domain id for an SM:
    ///
    /// - **Chiplet-based** (MI300X): the XCD the CU belongs to (8 domains × 38 CUs).
    /// - **DSM/GPC-based** (H100, Blackwell): the GPC the SM belongs to — SMs
    ///   within a GPC share L2 partition slices and are physically near.
    /// - **Monolithic without DSM** (Ada): single trivial domain (no preference).
    ///
    /// The scheduler uses this to break ties: prefer placing a consumer on the
    /// same locality domain as its producer, within a small slack window.
    pub fn locality_domain_of(&self, unit: UnitId, sm: SmId) -> usize {
        let u = self.unit(unit);
        // Prefer chiplet grouping (finer physical boundary) if available.
        if u.chiplet_count > 1 {
            return sm / u.sms_per_chiplet.max(1);
        }
        // Fall back to DSM/GPC domains — SMs in one GPC share L2 slices.
        if u.dsm_domains > 1 {
            return sm / u.sms_per_domain.max(1);
        }
        0
    }

    /// The SM index range belonging to `locality_domain_of`'s domain `d` on
    /// `unit`. Analogous to `domain_sms` (DSM) and `l2_partition_sms` (L2)
    /// but respects whichever partitioning `locality_domain_of` selected —
    /// chiplet on MI300, GPC on H100, trivial on monolithic non-DSM.
    ///
    /// The list scheduler uses this to restrict SM candidates when a task is
    /// pinned to a locality domain (`SameDomain` or `SameL2Partition`
    /// requests both collapse into this range).
    pub fn locality_domain_sms(&self, unit: UnitId, domain: usize) -> std::ops::Range<usize> {
        let u = self.unit(unit);
        let per = if u.chiplet_count > 1 {
            u.sms_per_chiplet.max(1)
        } else if u.dsm_domains > 1 {
            u.sms_per_domain.max(1)
        } else {
            u.sm_count.max(1)
        };
        let start = (domain * per).min(u.sm_count);
        let end = (start + per).min(u.sm_count);
        start..end
    }

    /// Number of locality domains on this unit (chiplets or GPC domains, whichever
    /// is the active grouping). Returns 1 on monolithic non-DSM dies.
    pub fn locality_domain_count(&self, unit: UnitId) -> usize {
        let u = self.unit(unit);
        if u.chiplet_count > 1 {
            return u.chiplet_count;
        }
        if u.dsm_domains > 1 {
            return u.dsm_domains;
        }
        1
    }

    /// The link class between two units (same unit ⇒ irrelevant; treated Fast).
    pub fn link(&self, a: UnitId, b: UnitId) -> LinkClass {
        if a == b {
            LinkClass::Fast
        } else if self.unified_memory {
            LinkClass::Unified
        } else if self.has_fast_interconnect {
            LinkClass::Fast
        } else {
            LinkClass::Slow
        }
    }

    /// Cycles to move `bytes` over a unit's HBM (DMA duration).
    pub fn hbm_cycles(&self, unit: UnitId, bytes: u64) -> Cycle {
        let bpc = self.unit(unit).hbm_bytes_per_cycle.max(1.0);
        (bytes as f64 / bpc).ceil() as Cycle
    }

    /// Cycles to move `bytes` over the interconnect (cross-unit transfer).
    pub fn link_cycles(&self, bytes: u64) -> Cycle {
        (bytes as f64 / self.link_bytes_per_cycle.max(1.0)).ceil() as Cycle
    }

    /// Peak aggregate HBM bandwidth across all units (bytes/cycle). Used by
    /// the oracle for lower-bound estimation (E2 = total_bytes / peak_bw).
    pub fn peak_hbm_bytes_per_cycle(&self) -> u64 {
        self.units
            .iter()
            .map(|u| u.hbm_bytes_per_cycle.ceil() as u64)
            .sum::<u64>()
            .max(1)
    }

    /// Peak aggregate compute throughput across all units, normalized to
    /// "cycles of work per wall-clock cycle". Since the scheduler's task
    /// durations are already in cycles-at-peak, this equals total SM count:
    /// with all SMs busy, total_compute_cycles / sm_count = wall-clock cycles.
    pub fn peak_flops_per_cycle(&self) -> u64 {
        self.units
            .iter()
            .map(|u| u.sm_count as u64)
            .sum::<u64>()
            .max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use costmodel::{Soc, DEFAULT_PAGE_BYTES};

    fn h100() -> &'static costmodel::hwspec::GpuSpec {
        costmodel::hwspec::registry::lookup("H100 SXM5").unwrap()
    }

    #[test]
    fn machine_reads_copy_engines_and_fabric_from_spec() {
        let soc = Soc::homogeneous(h100(), 2, DEFAULT_PAGE_BYTES);
        let m = Machine::from_soc(&soc, &Config::default());
        // copy_engines comes from the spec (H100 = 3), not a Config default.
        assert_eq!(m.unit(0).dma_engines, 3);
        // Fast fabric present ⇒ link class is Fast and bandwidth is nonzero.
        assert!(m.has_fast_interconnect);
        assert!(m.link_bytes_per_cycle > 0.0);
    }

    #[test]
    fn config_overrides_spec() {
        let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
        let cfg = Config {
            dma_engines: 5,
            ..Config::default()
        };
        let m = Machine::from_soc(&soc, &cfg);
        assert_eq!(m.unit(0).dma_engines, 5, "non-zero Config overrides the spec");
    }

    fn mi300x() -> &'static costmodel::hwspec::GpuSpec {
        costmodel::hwspec::registry::lookup("MI300X").unwrap()
    }

    #[test]
    fn mi300x_has_8_chiplets() {
        let soc = Soc::single(mi300x(), DEFAULT_PAGE_BYTES);
        let m = Machine::from_soc(&soc, &Config::default());
        let u = m.unit(0);
        assert_eq!(u.chiplet_count, 8);
        assert_eq!(u.sms_per_chiplet, 38);
    }

    #[test]
    fn chiplet_of_maps_sms_to_correct_xcd() {
        let soc = Soc::single(mi300x(), DEFAULT_PAGE_BYTES);
        let m = Machine::from_soc(&soc, &Config::default());
        // CU 0 → XCD 0, CU 37 → XCD 0, CU 38 → XCD 1, CU 303 → XCD 7
        assert_eq!(m.chiplet_of(0, 0), 0);
        assert_eq!(m.chiplet_of(0, 37), 0);
        assert_eq!(m.chiplet_of(0, 38), 1);
        assert_eq!(m.chiplet_of(0, 75), 1);
        assert_eq!(m.chiplet_of(0, 76), 2);
        assert_eq!(m.chiplet_of(0, 303), 7);
    }

    #[test]
    fn chiplet_sms_returns_correct_range() {
        let soc = Soc::single(mi300x(), DEFAULT_PAGE_BYTES);
        let m = Machine::from_soc(&soc, &Config::default());
        assert_eq!(m.chiplet_sms(0, 0), 0..38);
        assert_eq!(m.chiplet_sms(0, 1), 38..76);
        assert_eq!(m.chiplet_sms(0, 7), 266..304);
    }

    #[test]
    fn monolithic_die_has_trivial_chiplet() {
        let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
        let m = Machine::from_soc(&soc, &Config::default());
        let u = m.unit(0);
        assert_eq!(u.chiplet_count, 1);
        assert_eq!(u.sms_per_chiplet, 132);
        // All SMs map to chiplet 0.
        assert_eq!(m.chiplet_of(0, 0), 0);
        assert_eq!(m.chiplet_of(0, 131), 0);
    }

    #[test]
    fn h100_locality_domain_uses_gpc() {
        // H100 has no chiplets but 8 GPC domains — locality_domain_of should
        // use GPC for soft affinity (SMs sharing L2 partition slices).
        let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
        let m = Machine::from_soc(&soc, &Config::default());
        assert_eq!(m.locality_domain_count(0), 8, "H100 has 8 GPC domains");
        // SM 0 and SM 17 are in GPC 0 (18 SMs per domain).
        assert_eq!(m.locality_domain_of(0, 0), 0);
        assert_eq!(m.locality_domain_of(0, 17), 0);
        // SM 18 is in GPC 1.
        assert_eq!(m.locality_domain_of(0, 18), 1);
        // SM 131 is in GPC 7 (131 / 18 = 7).
        assert_eq!(m.locality_domain_of(0, 131), 7);
    }

    #[test]
    fn mi300x_locality_domain_uses_xcd() {
        // MI300X has chiplets but no DSM — locality_domain_of should use XCDs.
        let soc = Soc::single(mi300x(), DEFAULT_PAGE_BYTES);
        let m = Machine::from_soc(&soc, &Config::default());
        assert_eq!(m.locality_domain_count(0), 8, "MI300X has 8 XCDs");
        assert_eq!(m.locality_domain_of(0, 0), 0);
        assert_eq!(m.locality_domain_of(0, 37), 0);
        assert_eq!(m.locality_domain_of(0, 38), 1);
        assert_eq!(m.locality_domain_of(0, 303), 7);
    }
}
