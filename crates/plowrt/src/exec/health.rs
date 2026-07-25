//! §J Bringup verification & health, incl. the optional counter-space monitor
//! (deadlock detector + progress tracker).

use packet::Program;

use crate::device::ExecutorTarget;
use crate::exec::counters::CounterPool;
use crate::{Result, RuntimeError};

/// Verify the launched executor set matches expectations before serving: the
/// count is right and each executor's opcode mask covers the families the loaded
/// program uses. A missing/mismatched executor fails startup loudly.
pub fn verify_bringup(
    targets: &[ExecutorTarget],
    expected: usize,
    required_families: u32,
) -> Result<()> {
    if targets.len() != expected {
        return Err(RuntimeError::Device(format!(
            "bringup: expected {expected} executors, {} reported ready",
            targets.len()
        )));
    }
    for t in targets {
        if t.opcode_mask & required_families != required_families {
            return Err(RuntimeError::Device(format!(
                "executor {} opcode mask {:#x} lacks required families {:#x}",
                t.instance_id, t.opcode_mask, required_families
            )));
        }
    }
    Ok(())
}

/// A precise deadlock report for a stuck counter.
#[derive(Clone, Debug)]
pub struct DeadlockReport {
    pub counter: u32,
    pub value: u64,
    pub threshold: u64,
    /// Max increments any run of this schedule could deliver to `counter`
    /// (number of instructions listing it as a successor).
    pub max_possible: u64,
    pub reason: &'static str,
}

/// The optional counter-space monitor. Read-only over the counter pool + the
/// static counter graph derived from the program — off by default, cheap when on.
pub struct CounterMonitor {
    /// For each counter id, how many instructions can ever increment it. Static;
    /// computed once from the program.
    max_increments: Vec<u64>,
    /// Sum of thresholds — denominator of the progress ratio.
    total_threshold: u64,
    last_progress: f64,
}

impl CounterMonitor {
    /// Build from a program's instruction stream + counter table.
    pub fn new(program: &Program) -> Self {
        let n = program
            .counters
            .iter()
            .map(|c| c.id as usize + 1)
            .max()
            .unwrap_or(0);
        let mut max_increments = vec![0u64; n];
        for inst in &program.insts {
            for &c in &inst.succ {
                if (c as usize) < n {
                    max_increments[c as usize] += 1;
                }
            }
        }
        let total_threshold = program.counters.iter().map(|c| c.threshold as u64).sum();
        CounterMonitor {
            max_increments,
            total_threshold,
            last_progress: 0.0,
        }
    }

    /// Completion ratio `Σ min(value, threshold) / Σ threshold` in `[0, 1]`.
    pub fn progress(&mut self, pool: &CounterPool) -> f64 {
        if self.total_threshold == 0 {
            self.last_progress = 1.0;
            return 1.0;
        }
        let mut done = 0u64;
        for id in 0..pool.len() as u32 {
            done += pool.load(id).min(pool.threshold(id));
        }
        let p = done as f64 / self.total_threshold as f64;
        self.last_progress = p;
        p
    }

    /// Static reachability check: any counter whose threshold exceeds the number
    /// of instructions that could ever increment it can never be satisfied — a
    /// dropped increment or mis-scoped atomic. This is what a fault-injected
    /// missing successor trips.
    pub fn unsatisfiable(&self, pool: &CounterPool) -> Vec<DeadlockReport> {
        let mut out = Vec::new();
        for id in 0..pool.len() as u32 {
            let threshold = pool.threshold(id);
            let max = self.max_increments.get(id as usize).copied().unwrap_or(0);
            if threshold > max {
                out.push(DeadlockReport {
                    counter: id,
                    value: pool.load(id),
                    threshold,
                    max_possible: max,
                    reason: "threshold exceeds total possible increments (dropped/mis-scoped)",
                });
            }
        }
        out
    }

    /// Runtime stall check: given a fresh progress reading and the previous one,
    /// report whether the schedule advanced. Callers treat `!advanced` across a
    /// time budget as a global stall (fire `CANCEL`).
    pub fn advanced_since(&self, current: f64, previous: f64) -> bool {
        current > previous + f64::EPSILON
    }

    pub fn last_progress(&self) -> f64 {
        self.last_progress
    }
}
