//! Per-program device-opcode histogram of a devblob — the kernel inventory a
//! backend must cover to run it.
//!
//! `cargo run --example blob_ops -- model.pkt [--l2]`
//! (`--l2` accepts an L2-domain-placed blob, as `PLOW_L2_PLACE_DISPATCH=1` does.)
//!
//! `blob_ops PACKET... --speech-layout` prints one line per packet instead: the speech flag
//! layout it declares, its first instruction using a bit assigned since layout 0, and what a
//! loader of a speech object decides (`plow_asset::speech_flag_layout`); exit 1 on a refusal.

use std::collections::BTreeMap;

use packet::dev::DevOp;
use plowrt::asset::devblob::DevBlob;

/// Each packet against a speech object of this tree's layout; 1 if any is refused.
fn speech_layout() -> i32 {
    use plow_asset::speech_flag_layout as layout;
    let mut rc = 0;
    for path in std::env::args().skip(1).filter(|a| !a.starts_with("--")) {
        let buf = std::fs::read(&path).expect("read blob");
        let blob = DevBlob::parse_l2(&buf, true).expect("parse blob");
        let (declared, used) = blob.speech_flag_layout(&buf).expect("scan blob");
        let verdict = match layout::check(declared, used, Some(layout::LAYOUT)) {
            Ok(()) => "loads".to_string(),
            Err(e) => {
                rc = 1;
                format!("REFUSED: {e}")
            }
        };
        println!(
            "{path}: declared {declared:?} first protected {} -> {verdict}",
            used.map_or("none", name)
        );
    }
    rc
}

fn main() {
    if std::env::args().any(|a| a == "--speech-layout") {
        std::process::exit(speech_layout());
    }
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
        let fine = p
            .stream
            .iter()
            .filter(|e| e.flags & packet::dev::SE_FINE != 0)
            .count();
        let segs = p
            .stream
            .iter()
            .map(|e| e.seg)
            .max()
            .map_or(0, |m| m as usize + 1);
        println!(
            "prog[{pi}] T={} insts={} stream={} gq={} fine_entries={} segs={} counters={}",
            p.t,
            p.insts.len(),
            p.stream.len(),
            p.gq_stream.len(),
            fine,
            segs,
            p.n_counter
        );
        for (op, (n, slices)) in &hist {
            println!(
                "  {:>4} {:<32} insts={:<5} slices={}",
                op,
                name(*op),
                n,
                slices
            );
        }
        if std::env::args().any(|a| a == "--fields") {
            // First instance of every op: the operand contract as emitted for this model.
            let mut seen = std::collections::BTreeSet::new();
            for d in &p.insts {
                if seen.insert(d.op) {
                    println!(
                        "    {:<28} blocks={:<4} t={:?} i={:?} fj={:#x?}",
                        name(d.op),
                        d.blocks,
                        d.t,
                        d.i,
                        d.fj
                    );
                }
            }
        }
    }
    if std::env::args().any(|a| a == "--tensors") {
        for (i, t) in blob.tensors.iter().enumerate() {
            println!("  tensor {:>3} {:<40} {} B", i, t.name, t.bytes);
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
