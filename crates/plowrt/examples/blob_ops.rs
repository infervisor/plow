//! Per-program device-opcode histogram of a devblob — the kernel inventory a
//! backend must cover to run it.
//!
//! `cargo run --example blob_ops -- model.pkt [--l2]`
//! (`--l2` accepts an L2-domain-placed blob, as `PLOW_L2_PLACE_DISPATCH=1` does.)

use std::collections::BTreeMap;

use packet::dev::DevOp;
use plowrt::asset::devblob::DevBlob;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: blob_ops <model.pkt> [--l2]");
    let l2 = args.any(|a| a == "--l2");
    let buf = std::fs::read(&path).expect("read blob");
    let blob = DevBlob::parse_l2(&buf, l2).expect("parse blob");
    let mut union: BTreeMap<u16, usize> = BTreeMap::new();
    for (pi, p) in blob.progs.iter().enumerate() {
        let mut hist: BTreeMap<u16, (usize, usize)> = BTreeMap::new();
        for d in &p.insts {
            let e = hist.entry(d.op).or_default();
            e.0 += 1;
            e.1 += d.blocks as usize;
            *union.entry(d.op).or_default() += 1;
        }
        println!(
            "prog[{pi}] T={} insts={} stream={} gq={}",
            p.t,
            p.insts.len(),
            p.stream.len(),
            p.gq_stream.len()
        );
        for (op, (n, slices)) in &hist {
            println!("  {:>4} {:<32} insts={:<5} slices={}", op, name(*op), n, slices);
        }
    }
    println!("union: {} distinct ops", union.len());
    for op in union.keys() {
        println!("  {:>4} {}", op, name(*op));
    }
}

fn name(op: u16) -> &'static str {
    DevOp::from_u16(op).map(|o| o.c_name()).unwrap_or("?")
}
