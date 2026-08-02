//! §O Simulator / dry-run mode.
//!
//! Loads the compiled packets and walks each one **without a device**, honoring
//! the counter protocol and ordering, logging what every packet *would* do. It
//! reuses the CPU reference interpreter's counter-gated walk
//! (`device::cpu::run_streams`) via a recording [`StepObserver`], so there is no
//! duplicated gating/ordering logic — the difference from a live run is only
//! whether numerics execute and that every packet is recorded + logged.

use packet::{Body, Inst, Opcode, ResourceKind};

use crate::device::cpu::{run_streams, InterpretStats, StepObserver};
use crate::exec::counters::CounterPool;
use crate::exec::health::{CounterMonitor, DeadlockReport};
use crate::obs::trace::{TaskSpan, Timeline};

/// Whether the simulator executes the golden op numerics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathMode {
    /// Pure dry run — decode + log every packet, run no math.
    DryRun,
    /// Also execute the CPU golden op bodies (checks numerics vs the reference).
    Golden,
}

impl MathMode {
    fn run_math(self) -> bool {
        matches!(self, MathMode::Golden)
    }
}

/// One simulated packet: everything needed for the per-packet log line and the
/// Chrome-trace span.
#[derive(Clone, Debug)]
pub struct SimEvent {
    pub seq: usize,
    pub packet_index: usize,
    pub resource: &'static str,
    pub index: u16,
    pub unit: u8,
    pub opcode: u16,
    pub name: &'static str,
    pub body_summary: String,
    pub waits: Vec<u32>,
    pub succs: Vec<u32>,
    pub t_start: u64,
    pub t_end: u64,
}

impl SimEvent {
    /// A clean single log line, e.g.
    /// `#42 SM3 GEMM m=128 n=4096 k=3072 tile=128x256x64 wait[c17] -> succ[c23] t=15320..15960cyc`.
    pub fn log_line(&self) -> String {
        let waits = if self.waits.is_empty() {
            "-".to_string()
        } else {
            self.waits
                .iter()
                .map(|c| format!("c{c}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let succs = if self.succs.is_empty() {
            "-".to_string()
        } else {
            self.succs
                .iter()
                .map(|c| format!("c{c}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        format!(
            "#{:<5} {}{:<3} {:<10} {:<32} wait[{}] -> succ[{}] t={}..{}cyc",
            self.seq,
            self.resource,
            self.index,
            self.name,
            self.body_summary,
            waits,
            succs,
            self.t_start,
            self.t_end,
        )
    }
}

/// The observer that records a [`SimEvent`] per fired packet.
struct Recorder {
    math: bool,
    events: Vec<SimEvent>,
}

impl StepObserver for Recorder {
    #[inline]
    fn run_math(&self) -> bool {
        self.math
    }

    fn on_fire(&mut self, packet_index: usize, inst: &Inst, t_start: u64, t_end: u64) {
        let (name, body_summary) = describe(&inst.body);
        self.events.push(SimEvent {
            seq: self.events.len(),
            packet_index,
            resource: resource_name(inst.resource),
            index: inst.index,
            unit: inst.unit,
            opcode: inst.body.opcode().0,
            name,
            body_summary,
            waits: inst.wait.clone(),
            succs: inst.succ.clone(),
            t_start,
            t_end,
        });
    }
}

/// A full simulation result.
#[derive(Debug)]
pub struct SimReport {
    pub events: Vec<SimEvent>,
    pub stats: InterpretStats,
    /// Static deadlock check (counters whose threshold exceeds total increments).
    pub unsatisfiable: Vec<DeadlockReport>,
    /// Compiler's estimate for comparison, when known (from the manifest).
    pub compiler_makespan: Option<u64>,
    pub compiler_ideal: Option<u64>,
}

impl SimReport {
    /// Packet counts grouped by opcode family name.
    pub fn by_family(&self) -> Vec<(&'static str, usize)> {
        let mut acc: Vec<(&'static str, usize)> = Vec::new();
        for e in &self.events {
            match acc.iter_mut().find(|(n, _)| *n == e.name) {
                Some((_, c)) => *c += 1,
                None => acc.push((e.name, 1)),
            }
        }
        acc.sort_by(|a, b| b.1.cmp(&a.1));
        acc
    }

    /// Build a Chrome-trace [`Timeline`] from the events (one span per packet).
    pub fn timeline(&self) -> Timeline {
        let mut tl = Timeline::new();
        for e in &self.events {
            tl.push(TaskSpan {
                exec: e.index as u32,
                task: e.packet_index as u32,
                opcode: e.opcode,
                t_start: e.t_start,
                t_end: e.t_end,
            });
        }
        tl
    }

    /// Chrome-trace JSON (chrome://tracing / Perfetto).
    pub fn to_chrome_json(&self) -> String {
        self.timeline().to_chrome_json()
    }

    /// A human summary block.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "packets: {} fired / {} total  ({})\n",
            self.stats.executed,
            self.stats.total,
            if self.stats.completed {
                "completed"
            } else {
                "INCOMPLETE — deadlock"
            }
        ));
        s.push_str(&format!("simulated makespan: {} cyc", self.stats.makespan));
        if let Some(m) = self.compiler_makespan {
            s.push_str(&format!("  (compiler makespan {m}"));
            if let Some(i) = self.compiler_ideal {
                s.push_str(&format!(", ideal {i}"));
            }
            s.push(')');
        }
        s.push('\n');
        s.push_str("by family:");
        for (name, count) in self.by_family() {
            s.push_str(&format!(" {name}={count}"));
        }
        s.push('\n');
        if !self.unsatisfiable.is_empty() {
            s.push_str("UNSATISFIABLE COUNTERS (dropped/mis-scoped):\n");
            for r in &self.unsatisfiable {
                s.push_str(&format!(
                    "  counter {} threshold {} > max increments {} — {}\n",
                    r.counter, r.threshold, r.max_possible, r.reason
                ));
            }
        }
        s
    }
}

/// The simulator.
pub struct Simulator {
    pub math: MathMode,
}

impl Simulator {
    pub fn new(math: MathMode) -> Self {
        Simulator { math }
    }

    /// Dry-run (or golden-run) `program` against a fresh counter pool, returning
    /// the full report. Also runs the static counter-graph check.
    pub fn run(&self, program: &packet::Program) -> SimReport {
        let pool = CounterPool::from_counters(&program.counters);
        pool.reset_all();
        let mut rec = Recorder {
            math: self.math.run_math(),
            events: Vec::with_capacity(program.insts.len()),
        };
        let stats = run_streams(program, &pool, &mut rec);
        let monitor = CounterMonitor::new(program);
        let unsatisfiable = monitor.unsatisfiable(&pool);
        SimReport {
            events: rec.events,
            stats,
            unsatisfiable,
            compiler_makespan: None,
            compiler_ideal: None,
        }
    }
}

/// Opcode-family display name from the body variant.
fn describe(body: &Body) -> (&'static str, String) {
    match body {
        Body::Gemm {
            m,
            n,
            k,
            bm,
            bn,
            bk,
            ..
        } => ("GEMM", format!("m={m} n={n} k={k} tile={bm}x{bn}x{bk}")),
        Body::Flash {
            seq_q,
            seq_kv,
            head_dim,
            heads,
            bq,
            bkv,
            ..
        } => (
            "FLASH",
            format!("sq={seq_q} skv={seq_kv} hd={head_dim} heads={heads} tile={bq}x{bkv}"),
        ),
        Body::Row {
            reduce,
            rows,
            feat,
            br,
            ..
        } => (
            if *reduce { "ROW_REDUCE" } else { "ROW_PW" },
            format!("rows={rows} feat={feat} br={br}"),
        ),
        Body::Layout { rank, shape, .. } => {
            let dims: Vec<String> = shape
                .iter()
                .take(*rank as usize)
                .map(|d| d.to_string())
                .collect();
            ("LAYOUT", format!("shape=[{}]", dims.join(",")))
        }
        Body::Dma {
            load, bytes, slot, ..
        } => (
            if *load { "TMA_LOAD" } else { "TMA_STORE" },
            format!("bytes={bytes} slot={slot}"),
        ),
        Body::Rdma {
            bytes,
            src_unit,
            dst_unit,
        } => ("RDMA", format!("bytes={bytes} u{src_unit}->u{dst_unit}")),
        Body::Token {
            in_slot,
            out_slot,
            kind,
            vocab,
            ..
        } => {
            let name = match *kind {
                packet::Opcode::TOKEN_SAMPLE_GREEDY => "SAMPLE",
                packet::Opcode::TOKEN_SAMPLE_STOCHASTIC => "SAMPLE_S",
                packet::Opcode::TOKEN_TOKENIZE => "TOKENIZE",
                packet::Opcode::TOKEN_DETOKENIZE => "DETOKENIZE",
                _ => "TOKEN",
            };
            (name, format!("in={in_slot} out={out_slot} vocab={vocab}"))
        }
        Body::Host => ("HOST", String::new()),
    }
}

fn resource_name(r: ResourceKind) -> &'static str {
    match r {
        ResourceKind::Sm => "SM",
        ResourceKind::Dma => "DMA",
        ResourceKind::Dpu => "DPU",
        ResourceKind::Host => "HOST",
    }
}

/// (kept for symmetry with the packet ABI — opcode value already stored)
#[allow(dead_code)]
fn opcode_value(op: Opcode) -> u16 {
    op.0
}
