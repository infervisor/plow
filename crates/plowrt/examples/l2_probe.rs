//! What locality a devblob carries, and what the runtime actually does with it.
//!
//! `cargo run --example l2_probe -- model.pkt [nodes]`
//!
//! Reports each program's declared L2 domain count, the cu -> domain map recovered from the
//! per-entry domain bits, the candidate node split, and then the decision the runtime reaches on
//! it. Candidate and accepted are separate lines on purpose: a mapping can divide cleanly over the
//! nodes and still be refused for leaving a node busier than `cu % nodes` would, which is the
//! common outcome. Use it to tell a placed blob from an unplaced one, and an accepted plan from a
//! merely well-formed one, before attributing a measurement to placement.
//!
//! Domain recovery and the decision come from `exec::cpu::engine` itself rather than a copy here,
//! so this can never drift from what the engine does.

use std::collections::BTreeMap;

use packet::dev::{SE_DOMAIN_MASK, SE_DOMAIN_SHIFT};
use plowrt::asset::devblob::DevBlob;
use plowrt::exec::cpu::engine::{cu_domains, cu_work, node_plan};

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: l2_probe <model.pkt> [nodes]");
    let nodes: usize = args.next().map_or(8, |a| a.parse().expect("nodes"));
    let buf = std::fs::read(&path).expect("read blob");
    let blob = DevBlob::parse_l2(&buf, true).expect("parse blob");
    let n_cu = blob.n_cu;
    println!("n_cu={n_cu} progs={} nodes={nodes}", blob.progs.len());

    for (pi, p) in blob.progs.iter().enumerate() {
        let mut seen: BTreeMap<u32, usize> = BTreeMap::new();
        for e in &p.stream {
            *seen
                .entry(((e.flags & SE_DOMAIN_MASK) >> SE_DOMAIN_SHIFT) as u32)
                .or_default() += 1;
        }
        println!(
            "  prog[{pi}] T={} l2_domains={} stream={} gq_windows={} distinct_domain_bits={:?}",
            p.t,
            p.l2_domains,
            p.stream.len(),
            p.gq_seg_ofs.len().saturating_sub(1),
            seen.keys().collect::<Vec<_>>()
        );
    }

    let Some((cu_dom, domains)) = cu_domains(&blob.progs, n_cu) else {
        println!("\nUNPLACED: no usable domain map; placement stays cu % nodes");
        return;
    };
    let mut per_domain = vec![0usize; domains as usize];
    for &d in &cu_dom {
        per_domain[d as usize] += 1;
    }
    println!("\ncus per domain: {per_domain:?}");
    println!(
        "cu -> domain (first 16): {:?}",
        cu_dom.iter().take(16).collect::<Vec<_>>()
    );

    if !(domains as usize >= nodes && (domains as usize).is_multiple_of(nodes)) {
        println!("CANDIDATE: none — {domains} domains do not divide over {nodes} nodes");
        println!("ACCEPTED:  no, placement stays cu % nodes");
        return;
    }
    let per = domains as usize / nodes;
    let mut cus_per_node = vec![0usize; nodes];
    for &d in &cu_dom {
        cus_per_node[d as usize / per] += 1;
    }
    println!("CANDIDATE: {domains} domains / {nodes} nodes = {per} per node, cus per node: {cus_per_node:?}");

    // What each node actually has to RUN, per program — programs are alternatives, so each one's
    // own busiest node is its own makespan and the guard has to hold for every one separately.
    let work = cu_work(&blob.progs, n_cu);
    let accepted = node_plan(&cu_dom, domains, nodes, &work).is_some();
    println!("\nwork per node (stream entries), busiest node sets the makespan:");
    for (pi, (p, w)) in blob.progs.iter().zip(&work).enumerate() {
        let (mut placed, mut rr) = (vec![0u64; nodes], vec![0u64; nodes]);
        for (cu, &n) in w.iter().enumerate() {
            placed[cu_dom[cu] as usize / per] += n;
            rr[cu % nodes] += n;
        }
        let (pmax, rmax) = (*placed.iter().max().unwrap(), *rr.iter().max().unwrap());
        println!(
            "  prog[{pi}] T={:<5} peak placed {pmax} vs round-robin {rmax}  {}\n    placed      {placed:?}\n    round-robin {rr:?}",
            p.t,
            if pmax <= rmax { "ok" } else { "WORSE — declines the plan" }
        );
    }
    println!(
        "\nACCEPTED:  {}",
        if accepted {
            "yes — the runtime places by domain (with --cpu-l2-place, which is off by default)"
        } else {
            "no — the guard refuses it; placement stays cu % nodes even with --cpu-l2-place"
        }
    );
}
