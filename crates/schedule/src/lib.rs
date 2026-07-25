//! `schedule` — the scheduling half of the Infervisor JIT (design §4–§9).
//!
//! Consumes a `(TileGraph, ConstraintSet)` already device/unit-placed by
//! [`rewrite::assemble`] for a given [`costmodel::Soc`] (1 GPU, 2×H100, or a
//! heterogeneous SoC) and produces, via a resource-interval list scheduler,
//! per-resource ordered streams + a clustered counter graph + runtime packets.
//!
//! Resources are SMs, DMA engines, node-level **DPU** engines (cross-unit RDMA)
//! and a **host CPU thread** pool, each tracked by an [`interval::IntervalSet`];
//! HBM and the interconnect are capacity [`interval::BandwidthSet`]s.

pub mod bucket;
pub mod config;
pub mod counter_elim;
pub mod emit;
pub mod expand;
pub mod interval;
/// JSON-IPC bridge to the Lean verifier (`plow_verify` CLI). Compiled only
/// under the `lean-verify` Cargo feature.
#[cfg(feature = "lean-verify")]
pub mod lean_verify;
pub mod machine;
pub mod memory;
pub mod oracle;
pub mod passes;
pub mod prefetch;
pub mod relax;
pub mod resource;
pub mod scope_narrow;
pub mod sim;
pub mod sram_fit;
pub mod trace;
pub mod verify;

pub use bucket::{
    choose_buckets, choose_weight_tiling, choose_weight_tiling_tuned, compile_buckets, compile_buckets_tuned, BucketStream, Compiled, KvLayout, Phase,
    Request, ShapeBucket, WeightLayout,
};
pub use config::{ClusterMode, Config, DmaModel, Granularity};
pub use emit::{emit_program, emit_program_with_meta, issue_order};
pub use expand::{
    expand, expand_prefill_chunks, group_by_row_axis, ChunkPlacement, ChunkPlacementPolicy,
    PerChunkPlan, RowChunk, Task, TaskGraph, TaskId, TaskKind,
};
pub use interval::{BandwidthSet, Cycle, IntervalSet};
pub use memory::{allocate as allocate_memory, AddrEntry, AddressMap, BufClass, BufReq, Segment};
pub use machine::{LinkClass, Machine, SmId, UnitHw};
pub use oracle::{
    compute_lower_bound, compute_prefetch_depth, fill_bubbles, run_oracle, BubbleFillReport,
    LowerBound, OracleReport, PrefetchDepth,
};
/// Re-export of the standalone runtime ABI crate (kept separate so the runtime
/// can depend on just it).
pub use packet;
pub use passes::{
    build_counters, hbm_bandwidth_audit, list_schedule, ColoPin, Counter, HbmAudit, Packet,
    PacketKind, Schedule, Scope,
};
pub use relax::relax;
pub use resource::{PagePool, ResourceId, ResourceState};
pub use sim::{dump_packets, simulate, SimResult};
pub use verify::{verify, verify_schedule, VerifyError, VerifyReport};

use rewrite::{ConstraintSet, TileGraph};
use std::collections::{HashMap, HashSet};

/// A fully scheduled tile graph: the schedule plus the expanded task graph and
/// the machine model it was placed on (both handy for inspection / lowering).
pub struct Scheduled {
    pub schedule: Schedule,
    pub tasks: TaskGraph,
    pub machine: Machine,
    /// Oracle report, present when `Config::lean_oracle` is enabled or when
    /// `schedule_with_oracle` is called directly.
    pub oracle_report: Option<OracleReport>,
}

/// One SRAM spill event: a task whose page demand exceeded SM capacity.
#[derive(Clone, Debug)]
pub struct SpillInfo {
    /// Task index in the expanded TaskGraph.
    pub task: usize,
    /// Operation name.
    pub op: String,
    /// Working-set pages demanded (transient A/B staging).
    pub working_pages: u64,
    /// Output pages demanded (live until last consumer).
    pub out_pages: u64,
    /// Total capacity of the page pool on this SM.
    pub pool_capacity: u64,
    /// Unit that owns the SM.
    pub unit: usize,
}

/// Aggregated spill diagnostic.
#[derive(Clone, Debug)]
pub struct SpillReport {
    /// Individual spill events.
    pub sram_spills: Vec<SpillInfo>,
    /// Total SRAM page spills.
    pub sram_spill_count: usize,
    /// TMEM accumulator spills (Blackwell only).
    pub tmem_spill_count: usize,
}

impl Scheduled {
    /// Replay this schedule to a real (counter-gated) makespan + utilization.
    pub fn simulate(&self) -> SimResult {
        sim::simulate(&self.tasks, &self.schedule)
    }
    /// The runtime packet streams as readable text.
    pub fn dump_packets(&self) -> String {
        sim::dump_packets(&self.schedule)
    }
    /// Verify the emitted packet stream's counters enforce every data
    /// dependency (no tile starts before its parents complete).
    pub fn verify(&self, g: &TileGraph, cons: &ConstraintSet) -> Result<VerifyReport, VerifyError> {
        verify::verify_schedule(g, cons, &self.tasks, &self.schedule)
    }

    /// Produce a structured diagnostic of SRAM and TMEM spills. Each spill
    /// represents a tile that couldn't be assigned on-chip page slots (its
    /// combined working-set + output demand exceeded the SM's page pool at its
    /// scheduled time). This is the primary signal for tile-shape / SRAM budget
    /// mis-configuration.
    pub fn spill_report(&self) -> SpillReport {
        use crate::expand::TaskKind;
        use crate::resource::ResourceId;

        let mut sram_spills = Vec::new();
        for (i, task) in self.tasks.tasks.iter().enumerate() {
            if task.kind != TaskKind::Compute {
                continue;
            }
            if task.out_pages == 0 && task.sram_pages == 0 {
                continue;
            }
            // A task is spilled if it has page demand but no slot assignment.
            if self.schedule.sram_slots.contains_key(&i) {
                continue;
            }
            let unit = match self.schedule.placement.get(&i) {
                Some(ResourceId::Sm(u, _)) => *u,
                _ => 0,
            };
            sram_spills.push(SpillInfo {
                task: i,
                op: task.op.clone(),
                working_pages: task.sram_pages,
                out_pages: task.out_pages,
                pool_capacity: self.machine.unit(unit).pages_per_sm,
                unit,
            });
        }
        SpillReport {
            sram_spill_count: self.schedule.spills,
            tmem_spill_count: self.schedule.tmem_spills,
            sram_spills,
        }
    }
}

/// Schedule `g` onto `soc` under `cfg`: expand to tiles, cluster counters, then
/// list-schedule onto the node's resources.
pub fn schedule(
    soc: &costmodel::Soc,
    g: &TileGraph,
    cons: &ConstraintSet,
    cfg: &Config,
) -> Scheduled {
    let machine = Machine::from_soc(soc, cfg);
    let tasks = expand(soc, &machine, g, cons, cfg);

    // node → unit map (for counter scope), and the colocated-node set.
    let units = cons.placement.clone();
    let colocated: HashSet<usize> = cons.colocation_groups.iter().flatten().copied().collect();
    let (counters, wait_of, succ_of) = build_counters(&tasks, &units, &colocated, cfg.cluster);

    // Per-node colocation pin: pin each colocated tile by its coupled row band so
    // a producer and the consumer tile(s) that read its SRAM-resident output land
    // on one SM. `band` is the group's largest row block, so a producer covering
    // several smaller consumer tiles collides with all of them (see `ColoPin`).
    // Nodes whose domain has no row axis (untiled layout) get no pin and fall
    // back to the legacy `coord[0]` keying.
    let mut colo_pin: HashMap<usize, passes::ColoPin> = HashMap::new();
    for group in &cons.colocation_groups {
        let band = group
            .iter()
            .filter_map(|n| cons.domains.get(n).and_then(|d| d.row_axis()))
            .map(|(_, block)| block)
            .max()
            .unwrap_or(1)
            .max(1);
        for &n in group {
            if let Some((axis, block)) = cons.domains.get(&n).and_then(|d| d.row_axis()) {
                colo_pin.insert(n, passes::ColoPin { axis, block, band });
            }
        }
    }

    // SameDomain (DSM) hand-offs: assign each producer/consumer pair a
    // locality domain so both land in one DSM-reachable cluster (round-robin
    // across domains). SameL2Partition hand-offs share the same pinning map (a
    // partition IS a GPC on H100 and an XCD on MI300); the L2 case is
    // strictly weaker but the round-robin placement satisfies both.
    //
    // The modulus is `locality_domain_count` — the same partitioning
    // `choose_resource` resolves pins with (`locality_domain_sms`), so a pin
    // always names a real, distinct SM group (chiplet on MI300, GPC on H100).
    // Pairs are visited in sorted key order (the locality map is a HashMap)
    // and an already-pinned node keeps its pin, its partner joining it —
    // first assignment wins, deterministically.
    let mut domain_pin: HashMap<usize, usize> = HashMap::new();
    let mut pairs: Vec<(usize, usize)> = cons
        .locality
        .iter()
        .filter(|(_, req)| {
            matches!(
                req,
                rewrite::LocalityReq::SameDomain | rewrite::LocalityReq::SameL2Partition
            )
        })
        .map(|(&k, _)| k)
        .collect();
    pairs.sort_unstable();
    let mut next_domain = 0usize;
    for (producer, consumer) in pairs {
        let unit = *cons.placement.get(&producer).unwrap_or(&0);
        let domains = machine.locality_domain_count(unit).max(1);
        let d = match (domain_pin.get(&producer), domain_pin.get(&consumer)) {
            (Some(&d), _) => d,
            (None, Some(&d)) => d,
            (None, None) => {
                let d = next_domain % domains;
                next_domain += 1;
                d
            }
        };
        domain_pin.entry(producer).or_insert(d);
        domain_pin.entry(consumer).or_insert(d);
    }

    let sched = list_schedule(
        &machine,
        &tasks,
        &colocated,
        &colo_pin,
        &domain_pin,
        &counters,
        &wait_of,
        &succ_of,
    );

    // If lean_oracle is enabled, run the oracle pipeline (lower bound + bubble
    // fill + prefetch depth) on the freshly-produced schedule.
    if cfg.lean_oracle {
        let peak_bw = machine.peak_hbm_bytes_per_cycle();
        let peak_flops = machine.peak_flops_per_cycle();
        let num_units = machine.units.len();
        let max_depth = 8; // allow oracle to recommend up to 8-deep prefetch

        let (optimized, report) = oracle::run_oracle(
            &tasks,
            &sched,
            peak_bw,
            peak_flops,
            num_units,
            max_depth,
            cfg.lean_oracle,
        );

        Scheduled {
            schedule: optimized,
            tasks,
            machine,
            oracle_report: Some(report),
        }
    } else {
        Scheduled {
            schedule: sched,
            tasks,
            machine,
            oracle_report: None,
        }
    }
}

/// Schedule with the oracle pipeline explicitly, regardless of `cfg.lean_oracle`.
/// Useful for A/B comparison testing.
pub fn schedule_with_oracle(
    soc: &costmodel::Soc,
    g: &TileGraph,
    cons: &ConstraintSet,
    cfg: &Config,
) -> Scheduled {
    let mut cfg_with_oracle = *cfg;
    cfg_with_oracle.lean_oracle = true;
    schedule(soc, g, cons, &cfg_with_oracle)
}
