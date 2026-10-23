//! Instruction stream of a devblob as TSV, one row per instruction with its operands named
//! (`packet::slots`): `prog idx op blocks slot=tensor... | int=value...`. The input of fusion
//! audits (which producer feeds which consumer).
//!
//! `cargo run --example op_tsv -- model.pkt [--l2]`

use packet::dev::{DevOp, TENSOR_NONE16};
use plowrt::asset::devblob::DevBlob;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: op_tsv <model.pkt> [--l2]");
    let l2 = args.any(|a| a == "--l2");
    let buf = std::fs::read(&path).expect("read blob");
    let blob = DevBlob::parse_l2(&buf, l2).expect("parse blob");
    for (pi, p) in blob.progs.iter().enumerate() {
        for (ii, d) in p.insts.iter().enumerate() {
            let Some(op) = DevOp::from_u16(d.op) else {
                println!("{pi}\t{ii}\top{}\t{}", d.op, d.blocks);
                continue;
            };
            let s = packet::slots::slots_for(op);
            let mut row = format!("{pi}\t{ii}\t{op:?}\t{}", d.blocks);
            for (k, &t) in d.t.iter().enumerate() {
                if t == TENSOR_NONE16 || t as usize >= blob.tensors.len() {
                    continue;
                }
                let name = s.t[k].map_or("t", |x| x.name);
                row.push_str(&format!("\t{name}={}", blob.tensors[t as usize].name));
            }
            row.push_str("\t|");
            for (k, &v) in d.i.iter().enumerate() {
                if let Some(name) = s.i[k] {
                    row.push_str(&format!("\t{name}={v}"));
                }
            }
            println!("{row}");
        }
    }
}
