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

use std::collections::HashMap;

use plowrt::asset::devblob::{DevBlob, DevProg};
// The local `op_name` here covered 31 of 108 opcodes and rendered the rest as
// `op69`. `packet::disasm::op_name` is derived from `DevOp::ALL`, so it cannot
// fall behind the enum.
use packet::disasm::op_label as op_name;

// `transitive_reduction`, `Stats`, `critical_path`, `analyse`, `Placement`,
// `placement` and `placement_implied` moved to `plowrt::analysis::graph` so
// that `plowrt disasm --counters` computes them from the same code. This
// example is now a printer: the numbers below are byte-identical to what it
// emitted before the move, which matters because they are quoted in
// `perf-data/`.
use plowrt::analysis::graph::{analyse, critical_path, placement, placement_implied, Stats};

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

    let all_implied = s
        .edges
        .iter()
        .filter(|&&(a, b)| placement_implied(&p, a, b))
        .count();
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
    let polls_of =
        |es: &[(u32, u32)]| -> u64 { es.iter().map(|&(_, b)| s.blocks[b as usize] as u64).sum() };

    let implied: Vec<(u32, u32)> = s
        .edges
        .iter()
        .copied()
        .filter(|&(a, b)| placement_implied(&p, a, b))
        .collect();
    debug_assert_eq!(implied.len(), all_implied);

    println!(
        "  placement census (per-CU streams, {} CUs):",
        prog.stream_ofs.len()
    );
    println!(
        "    edges                              {:>8}",
        s.edges.len()
    );
    println!(
        "    transitively redundant             {:>8}",
        redundant.len()
    );
    println!(
        "    ... AND co-placed (cannot pay)     {:>8}",
        red_coplaced.len()
    );
    println!(
        "    ... AND NOT co-placed (could pay)  {:>8}",
        red_free.len()
    );
    println!(
        "    polls removable, redundant total   {:>8} of {}",
        polls_of(&redundant),
        s.polls
    );
    println!(
        "    polls removable, not-co-placed     {:>8} of {}",
        polls_of(&red_free),
        s.polls
    );
    // A SECOND, DISJOINT removable population, and the one that matches the
    // narrative: edges that are NOT transitively redundant — the DAG genuinely
    // needs the ordering — but that placement already delivers for free. These
    // are the `mla.rs` CU-0 chains. Deleting one is only sound if the placement
    // is also frozen, which is a much stronger contract than the DAG-side
    // theorem, for a much smaller prize.
    println!(
        "    edges implied by PLACEMENT alone   {:>8}  ({:.1}% of edges), polls {}",
        all_implied,
        100.0 * all_implied as f64 / s.edges.len().max(1) as f64,
        polls_of(&implied)
    );
    println!(
        "    ... of which also redundant        {:>8}",
        red_coplaced.len()
    );

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
    let path = args
        .next()
        .expect("usage: graphstat <asset-dir|model.pkt> [T]");
    let want_t: Option<u32> = args.next().and_then(|s| s.parse().ok());
    let verbose = std::env::var("GRAPHSTAT_V").ok().as_deref() == Some("1");

    let p = std::path::Path::new(&path);
    let file = if p.is_dir() {
        p.join("model.pkt")
    } else {
        p.to_path_buf()
    };
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
        "T",
        "ops",
        "counters",
        "SE_FINE",
        "edges",
        "tr",
        "deadctr",
        "ents",
        "polls",
        "polls_tr",
        "bumps",
        "bumps_live"
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
            println!(
                "  critical path: {d} of {} packets ({:.1}%)",
                s.n_ops,
                100.0 * d as f64 / s.n_ops as f64
            );
            let mut census: HashMap<String, u32> = HashMap::new();
            for &o in &spine {
                *census
                    .entry(op_name(prog.insts[o as usize].op))
                    .or_insert(0) += 1;
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
            let redundant: Vec<_> = s.edges.difference(&s.edges_tr).copied().collect();
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
