//! Parse a devblob with the RUNTIME's own parser and print what it recovers.
//!
//! The emitter's unit tests assert what it *wrote*; this asserts what the
//! loader *reads back*, which is a different question and the one that matters
//! for a blob nobody has run yet. In particular the TP degree is NOT a header
//! field — `DevBlob::parse` recovers it by scanning the emitted collectives —
//! so a blob whose XReduce operands are wrong reports the wrong world size here
//! rather than at rendezvous time on eight GPUs.
//!
//!   cargo run -p plowrt --features hsa --example parse_blob -- <model.pkt>

/// A stable 64-bit digest of one program's decoded instruction stream.
///
/// Hashed field by field rather than over the raw bytes, so it does not depend on padding the wire
/// format may or may not carry, and so a field ADDED to `DevInst64` shows up here as a compile
/// error instead of as a silently unchanged digest.
fn inst_digest(insts: &[packet::dev::DevInst64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for i in insts {
        i.op.hash(&mut h);
        i.blocks.hash(&mut h);
        i.fj.hash(&mut h);
        i.t.hash(&mut h);
        i.i.hash(&mut h);
    }
    h.finish()
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: parse_blob <model.pkt>");
    let raw = std::fs::read(&path).expect("read blob");
    let blob = plowrt::asset::devblob::DevBlob::parse(&raw).expect("parse blob");
    println!("blob      {path} ({} bytes)", raw.len());
    println!("n_cu      {}", blob.n_cu);
    println!("tensors   {}", blob.tensors.len());
    // `PLOW_DUMP_TENSORS=1` prints the DECLARED byte size of every tensor.
    //
    // That declaration is what `asset::shard::slice_for` reads to decide whether this
    // rank wants a 1/tp slice or the whole tensor, so diffing two blobs emitted at
    // different `--num-gpus` is the only direct way to see what the emitter chose to
    // shard — a name cannot tell you (K3 replicates `shared_experts.*`, GLM-5.2 shards
    // the identical spelling).
    if std::env::var("PLOW_DUMP_TENSORS").as_deref() == Ok("1") {
        for t in &blob.tensors {
            println!("T {:>12} {}", t.bytes, t.name);
        }
    }
    println!("programs  {}", blob.progs.len());
    // `prog_t` is the ROW COUNT each program was compiled for, and it is the field that says
    // whether a blob can prefill at all: the last entry is decode (T = 1) and everything before it
    // is a bucket. A blob whose `prog_t` is `[1]` walks a whole prompt through the decode program
    // one token at a time, which is the shape of TTFT gap this print exists to make visible.
    let last = blob.progs.len().saturating_sub(1);
    for (i, p) in blob.progs.iter().enumerate() {
        let kind = if i == last { "decode" } else { "prefill" };
        // The DIGEST covers the decoded instruction stream — opcodes, tensor handles, immediates,
        // block counts — and its point is comparability ACROSS BLOBS. Adding prefill buckets to a
        // model must not move the decode program by one byte, and the tensor table it indexes into
        // grows, so "the two blobs differ" is expected and says nothing. This is what makes the
        // claim checkable from outside the emitter: emit twice, diff this line.
        println!(
            "  prog[{i}]  T={:<6} {:>6} instructions  digest {:016x}  ({kind})",
            p.t,
            p.insts.len(),
            inst_digest(&p.insts)
        );
    }
    match &blob.tp {
        Some(tp) => println!(
            "TP        n_gpu={} hidden={} slot_bytes={}",
            tp.n_gpu, tp.hidden, tp.slot_bytes
        ),
        None => println!("TP        none (unsharded blob)"),
    }
}
