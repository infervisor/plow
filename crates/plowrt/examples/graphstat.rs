//! PX-21 — measure the emitted dependency graph: counters, edges, and how many
//! of each survive **transitive reduction**.
//!
//! The devblob's gates are counter-gated and live on the [`StreamEnt`]: each
//! entry carries `(wait_ofs, wait_len)` into the program's `waits` table and
//! `(succ_ofs, succ_len)` into `succs`. A COARSE entry points at its op's shared
//! lists, so the op-level DAG is recoverable from any one slice of each op.
//!
//! What this prints, per program:
//!
//! * `ops` / `counters` — `n_counter == n_ops` iff no op has per-slice (fine)
//!   counters; anything above is `SE_FINE` machinery.
//! * `edges` — distinct producer→consumer op pairs in the emitted graph.
//! * `edges_tr` — edges surviving **transitive reduction** (an A→C edge is
//!   redundant iff a path A→…→C of length ≥2 exists).
//! * `dead_ctr` — counters no consumer ever waits on. Every slice of such an op
//!   still bumps it (`succs` always carries `op.counter`), so these are pure
//!   wasted atomics.
//! * `polls` / `bumps` — the RUNTIME totals: Σ over stream entries of
//!   `wait_len` / `succ_len`. This is what the interpreter actually executes.
//!
//! No GPU is touched; the blob is read and nothing else.
//!
//! Usage: `graphstat <asset-dir|model.pkt> [T]` (no T = every program).

use std::collections::{BTreeSet, HashMap, HashSet};

use packet::dev::SE_FINE;
use plowrt::asset::devblob::{DevBlob, DevProg};

fn op_name(op: u16) -> String {
    use packet::dev::DevOp::*;
    macro_rules! m {
        ($($v:ident),* $(,)?) => {
            $(if op == $v as u16 { return stringify!($v).to_string(); })*
        };
    }
    m!(
        Nop, RmsNorm, HeadNormRope, Residual, Glu, Embed, SoftCap, Gemm, Gemv, FlashPrefill,
        FlashDecode, FlashMerge, GemmSmall, GemmMed, NormResidual, AddNorm, Argmax, ArgmaxFin,
        GemvGlu, GemmGlu, GemvQkv, GemvFp8, GemvGluFp8, QuantFp8, GemmFp8, GemmMedFp8,
        GemmSmallFp8, GemmGluFp8, HeadNormRopeFp8, FlashPrefillFp8, NormResidualNorm,
    );
    format!("op{op}")
}

/// Edges that survive transitive reduction: drop A→C when a path A→…→C of
/// length ≥ 2 exists. `succ[a]` is the direct-successor set.
fn transitive_reduction(n: usize, edges: &BTreeSet<(u32, u32)>) -> BTreeSet<(u32, u32)> {
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

struct Stats {
    n_ops: usize,
    n_counter: u32,
    fine_ents: usize,
    edges: BTreeSet<(u32, u32)>,
    edges_tr: BTreeSet<(u32, u32)>,
    dead_ctr: Vec<u32>,
    polls: u64,
    bumps: u64,
    /// Runtime polls that would remain after transitive reduction.
    polls_tr: u64,
    /// Runtime bumps that would remain after dropping dead counters.
    bumps_live: u64,
    n_ents: usize,
    blocks: Vec<u32>,
}

fn analyse(prog: &DevProg) -> Stats {
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

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: graphstat <asset-dir|model.pkt> [T]");
    let want_t: Option<u32> = args.next().and_then(|s| s.parse().ok());
    let verbose = std::env::var("GRAPHSTAT_V").ok().as_deref() == Some("1");

    let p = std::path::Path::new(&path);
    let file = if p.is_dir() { p.join("model.pkt") } else { p.to_path_buf() };
    let buf = std::fs::read(&file).expect("read blob");
    let blob = DevBlob::parse(&buf).expect("parse devblob");

    println!("blob      {}", file.display());
    println!("n_cu      {}", blob.n_cu);
    println!(
        "programs  {:?}",
        blob.progs.iter().map(|p| p.t).collect::<Vec<_>>()
    );
    println!();
    println!(
        "{:>7} {:>5} {:>8} {:>7} {:>6} {:>6} {:>7} {:>5} {:>10} {:>10} {:>10} {:>10}",
        "T", "ops", "counters", "SE_FINE", "edges", "tr", "deadctr", "ents", "polls", "polls_tr",
        "bumps", "bumps_live"
    );

    for prog in &blob.progs {
        if let Some(t) = want_t {
            if prog.t != t {
                continue;
            }
        }
        let s = analyse(prog);
        println!(
            "{:>7} {:>5} {:>8} {:>7} {:>6} {:>6} {:>7} {:>5} {:>10} {:>10} {:>10} {:>10}",
            prog.t,
            s.n_ops,
            s.n_counter,
            s.fine_ents,
            s.edges.len(),
            s.edges_tr.len(),
            s.dead_ctr.len(),
            s.n_ents,
            s.polls,
            s.polls_tr,
            s.bumps,
            s.bumps_live
        );
        if verbose {
            let redundant: Vec<_> =
                s.edges.difference(&s.edges_tr).copied().collect();
            println!("  redundant edges ({}):", redundant.len());
            for (a, b) in &redundant {
                println!(
                    "    {a:>3} {:<18} -> {b:>3} {:<18}",
                    op_name(prog.insts[*a as usize].op),
                    op_name(prog.insts[*b as usize].op)
                );
            }
            println!("  dead counters ({}):", s.dead_ctr.len());
            for c in &s.dead_ctr {
                println!(
                    "    {c:>3} {:<18} blocks={} (wasted atomics)",
                    op_name(prog.insts[*c as usize].op),
                    s.blocks[*c as usize]
                );
            }
            println!("  ops:");
            for (i, inst) in prog.insts.iter().enumerate() {
                let ins: Vec<u32> = s
                    .edges
                    .iter()
                    .filter(|(_, b)| *b == i as u32)
                    .map(|(a, _)| *a)
                    .collect();
                let ins_tr: Vec<u32> = s
                    .edges_tr
                    .iter()
                    .filter(|(_, b)| *b == i as u32)
                    .map(|(a, _)| *a)
                    .collect();
                println!(
                    "    {i:>3} {:<18} blocks={:<5} deps={:?} tr={:?}",
                    op_name(inst.op),
                    s.blocks[i],
                    ins,
                    ins_tr
                );
            }
        }
    }
}
