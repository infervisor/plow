//! Heterogeneous compute units of an SoC, and partitioning an op across them.
//!
//! Today every unit is a GPU (`Soc::single` / `Soc::homogeneous`), so
//! [`Soc::partition_n`] degenerates to one region covering the whole op — the
//! existing single-device path. The abstraction is in place so that when NPUs /
//! CPUs land, a matmul can be split across units (sized to each unit's
//! throughput, tiled with each unit's own MMA shapes) and the same placement /
//! memory-domain constraints carry straight over.

use crate::mma;
use crate::{CostModel, GemmShape};
use hwspec::GpuSpec;

/// Stable identifier for a compute unit within an SoC (dense, `0..units.len()`).
pub type UnitId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitKind {
    Gpu,
    Npu,
    Cpu,
}

/// One compute unit: its cost model plus a relative throughput weight used to
/// size its share of a split op.
pub struct Unit<'a> {
    pub id: UnitId,
    pub kind: UnitKind,
    /// Relative throughput for load balancing (regions sized ∝ weight).
    pub weight: f64,
    pub cm: CostModel<'a>,
}

/// Memory model of the SoC.
#[derive(Clone, Copy, Debug)]
pub struct MemoryModel {
    /// All units share one coherent address space ⇒ a value produced by one
    /// unit is readable by another with no extra copy (only a barrier), and a
    /// shared operand is staged from DRAM once. Discrete memory would make
    /// cross-unit operands explicit interconnect transfers.
    pub unified: bool,
}

/// A heterogeneous System-on-Chip: a set of units over a memory model.
pub struct Soc<'a> {
    pub units: Vec<Unit<'a>>,
    pub memory: MemoryModel,
}

/// One unit's share of a partitioned GEMM (a slice of the N / output-feature axis).
#[derive(Clone, Copy, Debug)]
pub struct Region {
    pub unit: UnitId,
    pub shape: GemmShape,
    pub n_start: i64,
}

impl<'a> Soc<'a> {
    /// A single GPU (the common case today): no real split; trivially unified.
    pub fn single(spec: &'a GpuSpec, page_bytes: u64) -> Soc<'a> {
        Soc {
            units: vec![Unit {
                id: 0,
                kind: UnitKind::Gpu,
                weight: 1.0,
                cm: CostModel::new(spec, page_bytes),
            }],
            memory: MemoryModel { unified: true },
        }
    }

    /// `n` identical GPUs over unified memory (e.g. multi-die / MIG slices).
    pub fn homogeneous(spec: &'a GpuSpec, n: usize, page_bytes: u64) -> Soc<'a> {
        let units = (0..n)
            .map(|id| Unit {
                id,
                kind: UnitKind::Gpu,
                weight: 1.0,
                cm: CostModel::new(spec, page_bytes),
            })
            .collect();
        Soc {
            units,
            memory: MemoryModel { unified: true },
        }
    }

    pub fn unit(&self, id: UnitId) -> &Unit<'a> {
        &self.units[id]
    }

    fn total_weight(&self) -> f64 {
        self.units.iter().map(|u| u.weight).sum()
    }

    /// Partition `g` along N (output features) across the units, each region
    /// sized ∝ its unit's throughput weight and rounded down to that unit's
    /// MMA-N granularity so the slice stays tensor-engine-legal; the last unit
    /// absorbs the remainder. A single unit ⇒ one region covering all of N.
    pub fn partition_n(&self, g: GemmShape) -> Vec<Region> {
        if self.units.len() == 1 {
            return vec![Region {
                unit: self.units[0].id,
                shape: g,
                n_start: 0,
            }];
        }
        let total = self.total_weight();
        let mut regions = Vec::new();
        let mut start = 0i64;
        let last = self.units.len() - 1;
        for (i, u) in self.units.iter().enumerate() {
            let gran = mma::max_n(u.cm.spec.arch).max(1);
            let len = if i == last {
                // Round up to MMA-N granularity so the last region stays tile-legal
                // (the kernel masks padded output columns, same as M-remainder).
                let raw = g.n - start;
                ((raw + gran - 1) / gran * gran).max(gran)
            } else {
                let share = (g.n as f64 * u.weight / total).round() as i64;
                ((share / gran) * gran).clamp(0, g.n - start)
            };
            if len > 0 {
                regions.push(Region {
                    unit: u.id,
                    shape: GemmShape {
                        m: g.m,
                        n: len,
                        k: g.k,
                    },
                    n_start: start,
                });
                start += len;
            }
        }
        // INVARIANT: regions cover all of `g.n`. The last region rounds *up* to the
        // MMA-N granularity, so `start` may exceed `g.n` by `< gran` — that tail is
        // padding the kernel masks (it never under-covers). Hence `start >= g.n`,
        // not `== g.n`.
        debug_assert!(
            start >= g.n,
            "partition_n under-covered N: {start} < {}",
            g.n
        );
        regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SramPolicy, DEFAULT_PAGE_BYTES};

    fn h100() -> &'static GpuSpec {
        hwspec::registry::lookup("H100 SXM5").unwrap()
    }

    #[test]
    fn single_unit_is_one_whole_region() {
        let soc = Soc::single(h100(), DEFAULT_PAGE_BYTES);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };
        let regions = soc.partition_n(g);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].shape.n, 4096);
        assert_eq!(regions[0].n_start, 0);
    }

    #[test]
    fn homogeneous_split_covers_n_and_stays_legal() {
        let soc = Soc::homogeneous(h100(), 2, DEFAULT_PAGE_BYTES);
        let g = GemmShape {
            m: 4096,
            n: 4096,
            k: 4096,
        };
        let regions = soc.partition_n(g);
        assert_eq!(regions.len(), 2);
        // Disjoint, contiguous, exhaustive over N.
        assert_eq!(regions[0].n_start, 0);
        assert_eq!(regions[1].n_start, regions[0].shape.n);
        assert_eq!(regions.iter().map(|r| r.shape.n).sum::<i64>(), g.n);
        // Each region is still MMA-legal ⇒ has tile candidates on its unit.
        for r in &regions {
            let cm = &soc.unit(r.unit).cm;
            assert!(!cm.candidates(r.shape, SramPolicy::Stream).is_empty());
        }
    }
}
