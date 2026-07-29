//! Per-counter detail, layered over [`graph::Stats`].
//!
//! `graphstat` answers "how many counters are dead" with a number. That is the
//! right shape for a regression check and the wrong shape for doing anything
//! about it: `dead_ctr = 1` does not say *which* counter, what produces it, or
//! what removing it would buy. Everything here is addressable — a counter id, a
//! producing instruction, a cost.
//!
//! The aggregates are not recomputed. [`Report::of`] takes the `Stats` the
//! caller already has, so the two can never disagree.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use packet::dev::SE_FINE;

use super::graph::Stats;
use crate::asset::devblob::DevProg;

/// One counter, with everything needed to act on it.
#[derive(Clone, Debug)]
pub struct Counter {
    pub id: u32,
    /// The count waiters require. `None` when nothing waits on it — a dead
    /// counter has no threshold because no wait record mentions it.
    pub threshold: Option<u32>,
    /// Instruction that bumps it. Coarse counter ids are op indices, so this is
    /// the id itself whenever it names a real op.
    pub producer: Option<u32>,
    /// Instructions that wait on it.
    pub consumers: Vec<u32>,
    /// Nothing waits on it, yet every slice of the producer still bumps it.
    pub dead: bool,
    /// Polls attributable to this counter: one per consumer slice. This is what
    /// removing it would actually save.
    pub poll_cost: u64,
    /// Bumps attributable to it: one per producer slice. Wasted in full when
    /// `dead`.
    pub bump_cost: u64,
}

/// A transitively redundant edge, with the path that makes it redundant.
///
/// The witness is the point. `graphstat` reports that 78 edges are redundant;
/// reviewing them needs to see *why* each one is, and a path is the only form
/// of that argument anyone can check.
#[derive(Clone, Debug)]
pub struct RedundantEdge {
    pub from: u32,
    pub to: u32,
    /// A path `from -> … -> to` of length ≥ 2.
    pub via: Vec<u32>,
    /// Polls removing this edge would save.
    pub polls_saved: u64,
    /// Both endpoints sit on one CU in program order, so the gate is already
    /// implied by placement and deleting it buys the poll but no overlap.
    pub co_placed: bool,
}

/// Counter pressure across the program.
///
/// A counter is live from its producer to its last consumer. The peak bounds
/// how small a counter pool could be; it is not derivable from any of the
/// aggregates.
#[derive(Clone, Debug, Default)]
pub struct Liveness {
    pub max_concurrent: u32,
    /// Instruction index where the peak occurs.
    pub at_inst: u32,
    pub p50: u32,
    pub p99: u32,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub counters: Vec<Counter>,
    pub redundant: Vec<RedundantEdge>,
    /// `threshold -> how many counters use it`.
    pub threshold_histogram: BTreeMap<u32, u32>,
    pub liveness: Liveness,
    /// Entries carrying per-slice (`SE_FINE`) counters. Fine counters are not
    /// op-addressable, so they are counted rather than enumerated.
    pub fine_ents: usize,
}

impl Report {
    pub fn of(prog: &DevProg, s: &Stats) -> Report {
        let n = s.n_ops;
        let ents: &[packet::dev::StreamEnt] =
            if prog.gq_stream.is_empty() { &prog.stream } else { &prog.gq_stream };

        // consumers[c] = ops that wait on counter c; threshold as recorded by
        // the waiters (identical across them by construction — a counter has one
        // threshold — so the first is taken).
        let mut consumers: HashMap<u32, BTreeSet<u32>> = HashMap::new();
        let mut threshold: HashMap<u32, u32> = HashMap::new();
        for e in ents {
            for k in 0..e.wait_len as usize {
                let w = prog.waits[e.wait_ofs as usize + k];
                consumers.entry(w.id).or_default().insert(e.inst);
                threshold.entry(w.id).or_insert(w.threshold);
            }
        }

        let dead: HashSet<u32> = s.dead_ctr.iter().copied().collect();
        let blocks = |op: u32| -> u64 { s.blocks.get(op as usize).copied().unwrap_or(0) as u64 };

        let mut counters: Vec<Counter> = (0..s.n_counter)
            .map(|id| {
                let cons: Vec<u32> =
                    consumers.get(&id).map(|c| c.iter().copied().collect()).unwrap_or_default();
                // Coarse counter ids ARE op indices; anything at or past
                // `n_ops` is a fine per-slice counter with no single producer.
                let producer = ((id as usize) < n).then_some(id);
                Counter {
                    id,
                    threshold: threshold.get(&id).copied(),
                    producer,
                    poll_cost: cons.iter().map(|b| blocks(*b)).sum(),
                    bump_cost: producer.map(blocks).unwrap_or(0),
                    dead: dead.contains(&id),
                    consumers: cons,
                }
            })
            .collect();
        counters.sort_by_key(|c| c.id);

        let mut threshold_histogram: BTreeMap<u32, u32> = BTreeMap::new();
        for c in &counters {
            if let Some(t) = c.threshold {
                *threshold_histogram.entry(t).or_insert(0) += 1;
            }
        }

        let place = super::graph::placement(prog, n);
        let succ = adjacency(n, &s.edges);
        let redundant: Vec<RedundantEdge> = s
            .edges
            .difference(&s.edges_tr)
            .map(|&(a, b)| RedundantEdge {
                from: a,
                to: b,
                via: witness(&succ, a, b).unwrap_or_default(),
                polls_saved: blocks(b),
                co_placed: super::graph::placement_implied(&place, a, b),
            })
            .collect();

        Report {
            liveness: liveness(n, &counters),
            fine_ents: ents.iter().filter(|e| e.flags & SE_FINE != 0).count(),
            counters,
            redundant,
            threshold_histogram,
        }
    }

    /// Dead counters, worst first — the actionable subset.
    pub fn dead(&self) -> Vec<&Counter> {
        let mut v: Vec<&Counter> = self.counters.iter().filter(|c| c.dead).collect();
        v.sort_by_key(|c| std::cmp::Reverse(c.bump_cost));
        v
    }

    /// The counters costing the most polls, worst first.
    pub fn hottest(&self, k: usize) -> Vec<&Counter> {
        let mut v: Vec<&Counter> = self.counters.iter().collect();
        v.sort_by_key(|c| std::cmp::Reverse(c.poll_cost));
        v.truncate(k);
        v
    }
}

fn adjacency(n: usize, edges: &BTreeSet<(u32, u32)>) -> Vec<Vec<u32>> {
    let mut succ = vec![Vec::new(); n];
    for &(a, b) in edges {
        succ[a as usize].push(b);
    }
    succ
}

/// Shortest path `a -> … -> b` of length ≥ 2, or `None`.
///
/// BFS from each direct successor of `a` other than `b`. Bounded work: this runs
/// once per redundant edge, and redundant edges are a percent-scale minority.
fn witness(succ: &[Vec<u32>], a: u32, b: u32) -> Option<Vec<u32>> {
    for &m in succ.get(a as usize)? {
        if m == b {
            continue;
        }
        let mut prev: HashMap<u32, u32> = HashMap::new();
        let mut q = std::collections::VecDeque::from([m]);
        let mut seen: HashSet<u32> = HashSet::from([m]);
        while let Some(cur) = q.pop_front() {
            if cur == b {
                let mut path = vec![b];
                let mut p = b;
                while let Some(&q0) = prev.get(&p) {
                    path.push(q0);
                    p = q0;
                }
                path.push(a);
                path.reverse();
                return Some(path);
            }
            for &nx in succ.get(cur as usize).into_iter().flatten() {
                if seen.insert(nx) {
                    prev.insert(nx, cur);
                    q.push_back(nx);
                }
            }
        }
    }
    None
}

/// Sweep counter live intervals over instruction index.
///
/// `[producer, last consumer]`; a counter with no consumer is never live (it is
/// bumped and forgotten, which is exactly what makes it dead).
fn liveness(n: usize, counters: &[Counter]) -> Liveness {
    if n == 0 {
        return Liveness::default();
    }
    let mut delta = vec![0i32; n + 1];
    for c in counters {
        let (Some(p), Some(last)) = (c.producer, c.consumers.iter().max().copied()) else {
            continue;
        };
        if (p as usize) >= n || last <= p {
            continue;
        }
        delta[p as usize] += 1;
        delta[(last as usize).min(n)] -= 1;
    }
    let mut cur = 0i32;
    let mut profile = Vec::with_capacity(n);
    let (mut max, mut at) = (0u32, 0u32);
    for (i, d) in delta.iter().take(n).enumerate() {
        cur += d;
        let v = cur.max(0) as u32;
        if v > max {
            max = v;
            at = i as u32;
        }
        profile.push(v);
    }
    profile.sort_unstable();
    let pct = |p: usize| -> u32 {
        if profile.is_empty() {
            0
        } else {
            profile[(profile.len() - 1) * p / 100]
        }
    };
    Liveness { max_concurrent: max, at_inst: at, p50: pct(50), p99: pct(99) }
}
