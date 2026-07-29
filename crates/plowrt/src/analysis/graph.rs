//! Static analysis of a compiled program's dependency graph.
//!
//! Moved verbatim out of `examples/graphstat.rs` (PX-21) so that both the
//! `graphstat` example and `plowrt disasm --counters` compute it once, from the
//! same code. `graphstat`'s printed output is unchanged by the move — its
//! numbers appear in committed `perf-data/` write-ups, so a refactor that
//! shifted one would be a silent falsification of those.
//!
//! Nothing here touches a GPU: it reads a parsed blob and derives.

use std::collections::{BTreeSet, HashMap, HashSet};

use packet::dev::SE_FINE;

use crate::asset::devblob::DevProg;

/// Edges that survive transitive reduction: drop A→C when a path A→…→C of
/// length ≥ 2 exists. `succ[a]` is the direct-successor set.
pub fn transitive_reduction(n: usize, edges: &BTreeSet<(u32, u32)>) -> BTreeSet<(u32, u32)> {
    // Reachability by 2+ hops. The op DAG is emitted in topological order
    // (producer index < consumer index), so a single reverse sweep suffices.
    let mut succ: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    for &(a, b) in edges {
        succ[a as usize].insert(b);
    }
    // reach[a] = every node reachable from a (any distance ≥ 1).
    let mut reach: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    for a in (0..n).rev() {
        let mut r: HashSet<u32> = HashSet::new();
        for &b in &succ[a] {
            r.insert(b);
            for &c in &reach[b as usize] {
                r.insert(c);
            }
        }
        reach[a] = r;
    }
    let mut out = BTreeSet::new();
    for &(a, b) in edges {
        // Redundant iff some other direct successor of a reaches b.
        let redundant = succ[a as usize]
            .iter()
            .any(|&m| m != b && reach[m as usize].contains(&b));
        if !redundant {
            out.insert((a, b));
        }
    }
    out
}

pub struct Stats {
    pub n_ops: usize,
    pub n_counter: u32,
    pub fine_ents: usize,
    pub edges: BTreeSet<(u32, u32)>,
    pub edges_tr: BTreeSet<(u32, u32)>,
    pub dead_ctr: Vec<u32>,
    pub polls: u64,
    pub bumps: u64,
    /// Runtime polls that would remain after transitive reduction.
    pub polls_tr: u64,
    /// Runtime bumps that would remain after dropping dead counters.
    pub bumps_live: u64,
    pub n_ents: usize,
    pub blocks: Vec<u32>,
}

/// Longest path through the op DAG, measured in PACKETS (unit weight per op).
/// The emitted op order is topological (producer index < consumer index), so a
/// single forward sweep suffices. Returns `(depth, spine)`.
pub fn critical_path(n: usize, edges: &BTreeSet<(u32, u32)>) -> (u32, Vec<u32>) {
    let mut preds: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(a, b) in edges {
        preds[b as usize].push(a);
    }
    let mut depth = vec![1u32; n];
    let mut from = vec![u32::MAX; n];
    for i in 0..n {
        for &p in &preds[i] {
            if depth[p as usize] + 1 > depth[i] {
                depth[i] = depth[p as usize] + 1;
                from[i] = p;
            }
        }
    }
    let end = (0..n).max_by_key(|&i| depth[i]).unwrap_or(0);
    let mut spine = Vec::new();
    let mut cur = end as u32;
    loop {
        spine.push(cur);
        let p = from[cur as usize];
        if p == u32::MAX {
            break;
        }
        cur = p;
    }
    spine.reverse();
    (depth[end], spine)
}

pub fn analyse(prog: &DevProg) -> Stats {
    let n_ops = prog.insts.len();
    let ents: &[packet::dev::StreamEnt] =
        if prog.gq_stream.is_empty() { &prog.stream } else { &prog.gq_stream };

    let fine_ents = ents.iter().filter(|e| e.flags & SE_FINE != 0).count();

    // Op-level DAG from the coarse wait lists. Counter ids < n_ops are the
    // per-op coarse counters; ids >= n_ops are fine per-slice counters, which
    // we map back to their producing op via the fine base ranges (absent here
    // whenever fine_ents == 0).
    let mut edges: BTreeSet<(u32, u32)> = BTreeSet::new();
    let mut per_op_waitlen: HashMap<u32, u16> = HashMap::new();
    let mut polls = 0u64;
    let mut bumps = 0u64;
    let mut waited_on: HashSet<u32> = HashSet::new();

    for e in ents {
        polls += e.wait_len as u64;
        bumps += e.succ_len as u64;
        per_op_waitlen.entry(e.inst).or_insert(e.wait_len);
        for k in 0..e.wait_len as usize {
            let w = prog.waits[e.wait_ofs as usize + k];
            waited_on.insert(w.id);
            if (w.id as usize) < n_ops {
                edges.insert((w.id, e.inst));
            }
        }
    }

    let edges_tr = transitive_reduction(n_ops, &edges);

    // Counters nobody waits on: every slice still bumps them.
    let dead_ctr: Vec<u32> =
        (0..n_ops as u32).filter(|c| !waited_on.contains(c)).collect();
    let dead_set: HashSet<u32> = dead_ctr.iter().copied().collect();

    // Per-op slice counts, for the runtime arithmetic.
    let mut blocks = vec![0u32; n_ops];
    for e in ents {
        blocks[e.inst as usize] += 1;
    }

    // What the runtime would execute after each pruning.
    let kept_in: HashMap<u32, usize> = {
        let mut m: HashMap<u32, usize> = HashMap::new();
        for &(_, b) in &edges_tr {
            *m.entry(b).or_insert(0) += 1;
        }
        m
    };
    let polls_tr: u64 = (0..n_ops as u32)
        .map(|o| {
            let keep = *kept_in.get(&o).unwrap_or(&0) as u64;
            keep * blocks[o as usize] as u64
        })
        .sum();
    let bumps_live: u64 = (0..n_ops as u32)
        .map(|o| if dead_set.contains(&o) { 0 } else { blocks[o as usize] as u64 })
        .sum();

    Stats {
        n_ops,
        n_counter: prog.n_counter,
        fine_ents,
        edges,
        edges_tr,
        dead_ctr,
        polls,
        bumps,
        polls_tr,
        bumps_live,
        n_ents: ents.len(),
        blocks,
    }
}

/// Where each op actually RUNS: the set of CUs holding at least one of its
/// slices, plus the first/last position of those slices inside that CU's own
/// stream window.
///
/// The static interpreter walks a PER-CU stream — a workgroup executes every
/// entry it owns, in order, gate or no gate. So `stream_ofs`/`stream_len` are
/// not bookkeeping, they are the real execution order, and two packets in one
/// CU's window are serialised by that fact alone.
pub struct Placement {
    /// `cus[op]` — CUs with at least one slice of `op`.
    pub cus: Vec<BTreeSet<u32>>,
    /// `(cu, op)` → first and last index of `op` within that CU's window.
    pub span: HashMap<(u32, u32), (u32, u32)>,
}

pub fn placement(prog: &DevProg, n_ops: usize) -> Placement {
    let mut cus: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); n_ops];
    let mut span: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
    for cu in 0..prog.stream_ofs.len() as u32 {
        let ofs = prog.stream_ofs[cu as usize] as usize;
        let len = prog.stream_len[cu as usize] as usize;
        for (k, e) in prog.stream[ofs..ofs + len].iter().enumerate() {
            let k = k as u32;
            cus[e.inst as usize].insert(cu);
            span.entry((cu, e.inst))
                .and_modify(|(lo, hi)| {
                    *lo = (*lo).min(k);
                    *hi = (*hi).max(k);
                })
                .or_insert((k, k));
        }
    }
    Placement { cus, span }
}

/// Is edge `a -> b` ALREADY enforced by placement, with no gate needed?
///
/// A coarse counter gate demands that EVERY slice of `a` has finished. Per-CU
/// program order can only ever deliver that when every slice of `a` sits on the
/// one CU that runs `b`, ahead of it. Hence all three conditions:
///
///   * `a` occupies exactly ONE CU — otherwise a consumer on CU `c` learns
///     nothing about `a`'s slices on the other CUs;
///   * `b` occupies that SAME single CU — a slice of `b` anywhere else is
///     unordered with respect to `a`;
///   * `a`'s last entry precedes `b`'s first entry in that CU's window.
///
/// When this holds, deleting the gate cannot create overlap, because there was
/// no concurrency to unlock: the two packets are consecutive work items in one
/// workgroup's serial stream. This is the shape `mla.rs` mass-produces by
/// putting every 1-workgroup packet on CU 0.
pub fn placement_implied(p: &Placement, a: u32, b: u32) -> bool {
    if p.cus[a as usize].len() != 1 || p.cus[a as usize] != p.cus[b as usize] {
        return false;
    }
    let cu = *p.cus[a as usize].iter().next().unwrap();
    match (p.span.get(&(cu, a)), p.span.get(&(cu, b))) {
        (Some(&(_, a_hi)), Some(&(b_lo, _))) => a_hi < b_lo,
        _ => false,
    }
}

