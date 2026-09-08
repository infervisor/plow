//! What locality a devblob carries, and how it lands on this host's NUMA nodes.
//!
//! `cargo run --example l2_probe -- model.pkt [nodes]`
//!
//! Reports each program's declared L2 domain count and the cu -> domain map recovered from the
//! per-entry domain bits, then the node split `exec::cpu::engine::node_plan` would choose. Use it
//! to tell a placed blob from an unplaced one before attributing a measurement to placement.

use std::collections::BTreeMap;

use packet::dev::{SE_DOMAIN_MASK, SE_DOMAIN_SHIFT};
use plowrt::asset::devblob::DevBlob;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: l2_probe <model.pkt> [nodes]");
    let nodes: usize = args.next().map_or(8, |a| a.parse().expect("nodes"));
    let buf = std::fs::read(&path).expect("read blob");
    let blob = DevBlob::parse_l2(&buf, true).expect("parse blob");
    println!(
        "n_cu={} progs={} nodes={nodes}",
        blob.n_cu,
        blob.progs.len()
    );

    let mut cu_dom: Vec<u32> = vec![u32::MAX; blob.n_cu as usize];
    let mut domains = 0u32;
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
        if p.l2_domains == 0 {
            continue;
        }
        domains = p.l2_domains;
        for (cu, slot) in cu_dom.iter_mut().enumerate() {
            let start = p.stream_ofs[cu] as usize;
            for e in &p.stream[start..start + p.stream_len[cu] as usize] {
                let d = ((e.flags & SE_DOMAIN_MASK) >> SE_DOMAIN_SHIFT) as u32;
                if *slot != u32::MAX && *slot != d {
                    println!("  !! cu {cu} carries domains {slot} and {d}");
                }
                *slot = d;
            }
        }
    }
    if domains == 0 {
        println!("UNPLACED: no program declares L2 domains; placement stays cu % nodes");
        return;
    }

    let mut per_domain = vec![0usize; domains as usize];
    for &d in cu_dom.iter().filter(|&&d| d != u32::MAX) {
        per_domain[d as usize] += 1;
    }
    println!("cus per domain: {per_domain:?}");
    let head: Vec<u32> = cu_dom.iter().take(16).copied().collect();
    println!("cu -> domain (first 16): {head:?}");

    if !(domains as usize >= nodes && (domains as usize).is_multiple_of(nodes)) {
        println!("PLACED but {domains} domains do not divide over {nodes} nodes: falls back to cu % nodes");
        return;
    }
    let per = domains as usize / nodes;
    let mut cus_per_node = vec![0usize; nodes];
    for &d in cu_dom.iter().filter(|&&d| d != u32::MAX) {
        cus_per_node[d as usize / per] += 1;
    }
    println!("PLACED: {domains} domains / {nodes} nodes = {per} per node, cus per node: {cus_per_node:?}");

    // What each node actually has to RUN. Stream entries per cu is the work unit the static
    // walk executes, and the makespan is set by the busiest node, so this is the number that
    // decides whether a locality plan costs more than it can buy.
    println!("\nwork per node (stream entries), busiest node sets the makespan:");
    for (pi, p) in blob.progs.iter().enumerate() {
        let (mut placed, mut rr) = (vec![0usize; nodes], vec![0usize; nodes]);
        for cu in 0..blob.n_cu as usize {
            let n = p.stream_len[cu] as usize;
            if cu_dom[cu] != u32::MAX {
                placed[cu_dom[cu] as usize / per] += n;
            }
            rr[cu % nodes] += n;
        }
        let ratio =
            |v: &[usize]| *v.iter().max().unwrap() as f64 / *v.iter().min().unwrap().max(&1) as f64;
        println!(
            "  prog[{pi}] T={:<5} placed max/min {:.2}x {placed:?}\n              {:11} round-robin max/min {:.2}x {rr:?}",
            p.t,
            ratio(&placed),
            "",
            ratio(&rr)
        );
    }
}
