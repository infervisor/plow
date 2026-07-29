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

/// Longest path through the op DAG, measured in PACKETS (unit weight per op).
/// The emitted op order is topological (producer index < consumer index), so a
/// single forward sweep suffices. Returns `(depth, spine)`.
fn critical_path(n: usize, edges: &BTreeSet<(u32, u32)>) -> (u32, Vec<u32>) {
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

/// Where each op actually RUNS: the set of CUs holding at least one of its
/// slices, plus the first/last position of those slices inside that CU's own
/// stream window.
///
/// The static interpreter walks a PER-CU stream — a workgroup executes every
/// entry it owns, in order, gate or no gate. So `stream_ofs`/`stream_len` are
/// not bookkeeping, they are the real execution order, and two packets in one
/// CU's window are serialised by that fact alone.
struct Placement {
    /// `cus[op]` — CUs with at least one slice of `op`.
    cus: Vec<BTreeSet<u32>>,
    /// `(cu, op)` → first and last index of `op` within that CU's window.
    span: HashMap<(u32, u32), (u32, u32)>,
}

fn placement(prog: &DevProg, n_ops: usize) -> Placement {
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
fn placement_implied(p: &Placement, a: u32, b: u32) -> bool {
    if p.cus[a as usize].len() != 1 || p.cus[a as usize] != p.cus[b as usize] {
        return false;
    }
    let cu = *p.cus[a as usize].iter().next().unwrap();
    match (p.span.get(&(cu, a)), p.span.get(&(cu, b))) {
        (Some(&(_, a_hi)), Some(&(b_lo, _))) => a_hi < b_lo,
        _ => false,
    }
}

/// `GRAPHSTAT_PLACE=1` — the edge census that decides whether an edge-REMOVAL
/// pass is worth writing.
///
/// Three populations, and only the third could ever pay:
///   1. every gate edge;
///   2. the transitively redundant ones (implied by a path of length ≥ 2);
///   3. the redundant ones whose endpoints are NOT co-placed.
///
/// Read the result with the ceiling in mind: deleting cache maintenance
/// outright (`PLOW_GATE_NOINV` / `PLOW_GATE_RELAXSIG`, real data races, so a
/// hard bound) prices ALL genuine local gating at 0.098 ms/CU = 0.34% of the
/// token. Nothing found here can exceed that.
fn place_census(prog: &DevProg, s: &Stats) {
    let p = placement(prog, s.n_ops);
    let redundant: Vec<(u32, u32)> = s.edges.difference(&s.edges_tr).copied().collect();

    let all_implied = s.edges.iter().filter(|&&(a, b)| placement_implied(&p, a, b)).count();
    let (mut red_coplaced, mut red_free) = (Vec::new(), Vec::new());
    for &(a, b) in &redundant {
        if placement_implied(&p, a, b) {
            red_coplaced.push((a, b));
        } else {
            red_free.push((a, b));
        }
    }

    // What removing an edge is actually WORTH: one poll per consumer slice.
    // It cannot buy overlap — transitive redundancy means the surviving path
    // already blocks the consumer for at least as long — so the poll is the
    // entire prize.
    let polls_of = |es: &[(u32, u32)]| -> u64 {
        es.iter().map(|&(_, b)| s.blocks[b as usize] as u64).sum()
    };

    let implied: Vec<(u32, u32)> =
        s.edges.iter().copied().filter(|&(a, b)| placement_implied(&p, a, b)).collect();
    debug_assert_eq!(implied.len(), all_implied);

    println!("  placement census (per-CU streams, {} CUs):", prog.stream_ofs.len());
    println!("    edges                              {:>8}", s.edges.len());
    println!("    transitively redundant             {:>8}", redundant.len());
    println!("    ... AND co-placed (cannot pay)     {:>8}", red_coplaced.len());
    println!("    ... AND NOT co-placed (could pay)  {:>8}", red_free.len());
    println!("    polls removable, redundant total   {:>8} of {}",
             polls_of(&redundant), s.polls);
    println!("    polls removable, not-co-placed     {:>8} of {}",
             polls_of(&red_free), s.polls);
    // A SECOND, DISJOINT removable population, and the one that matches the
    // narrative: edges that are NOT transitively redundant — the DAG genuinely
    // needs the ordering — but that placement already delivers for free. These
    // are the `mla.rs` CU-0 chains. Deleting one is only sound if the placement
    // is also frozen, which is a much stronger contract than the DAG-side
    // theorem, for a much smaller prize.
    println!("    edges implied by PLACEMENT alone   {:>8}  ({:.1}% of edges), polls {}",
             all_implied,
             100.0 * all_implied as f64 / s.edges.len().max(1) as f64,
             polls_of(&implied));
    println!("    ... of which also redundant        {:>8}", red_coplaced.len());

    let show = |label: &str, es: &[(u32, u32)]| {
        if es.is_empty() {
            return;
        }
        println!("    {label}:");
        for &(a, b) in es.iter().take(40) {
            println!(
                "      {a:>4} {:<18} cu={:<4} -> {b:>4} {:<18} cu={}",
                op_name(prog.insts[a as usize].op),
                p.cus[a as usize].len(),
                op_name(prog.insts[b as usize].op),
                p.cus[b as usize].len()
            );
        }
        if es.len() > 40 {
            println!("      ... {} more", es.len() - 40);
        }
    };
    show("redundant AND NOT co-placed", &red_free);
    if std::env::var("GRAPHSTAT_PLACE_ALL").ok().as_deref() == Some("1") {
        show("redundant AND co-placed", &red_coplaced);
        show("implied by placement alone", &implied);
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
        if std::env::var("GRAPHSTAT_PLACE").ok().as_deref() == Some("1") {
            place_census(prog, &s);
        }
        if std::env::var("GRAPHSTAT_CP").ok().as_deref() == Some("1") {
            let (d, spine) = critical_path(s.n_ops, &s.edges);
            println!("  critical path: {d} of {} packets ({:.1}%)", s.n_ops,
                     100.0 * d as f64 / s.n_ops as f64);
            let mut census: HashMap<String, u32> = HashMap::new();
            for &o in &spine {
                *census.entry(op_name(prog.insts[o as usize].op)).or_insert(0) += 1;
            }
            let mut c: Vec<_> = census.into_iter().collect();
            c.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
            println!("  spine census:");
            for (name, n) in &c {
                println!("    {name:<20} {n:>4}");
            }
            println!("  spine (idx op blocks):");
            for &o in &spine {
                println!(
                    "    {o:>3} {:<20} blocks={}",
                    op_name(prog.insts[o as usize].op),
                    s.blocks[o as usize]
                );
            }
        }
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
