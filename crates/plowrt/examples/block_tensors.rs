//! Dump the BlobTensor table (name + byte size) of a PLOWDEV `.pkt` — used by
//! the block-asset sanity harness to derive the exact weight tensors a synthetic
//! checkpoint must provide (loader binds `model.`/`fp8/` names by size).
//!
//!   block_tensors <blob.pkt>

use plowrt::asset::devblob::DevBlob;

fn main() {
    let path = std::env::args().nth(1).expect("usage: block_tensors <blob.pkt>");
    let buf = std::fs::read(&path).expect("read blob");
    let blob = DevBlob::parse(&buf).expect("parse blob");
    println!("# n_cu={} tensors={} progs={}", blob.n_cu, blob.tensors.len(), blob.progs.len());
    for t in &blob.tensors {
        let kind = if t.name.starts_with("model.") || t.name.starts_with("fp8/") {
            "WEIGHT" // bound from checkpoint by name+size
        } else if t.init.is_some() {
            "INIT" // compiler-filled (rope tables etc.)
        } else {
            "SCRATCH" // zeroed / activation / kv
        };
        println!("{kind}\t{}\t{}", t.bytes, t.name);
    }
}
