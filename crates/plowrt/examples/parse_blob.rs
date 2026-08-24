//! Parse a devblob with the RUNTIME's own parser and print what it recovers.
//!
//! The emitter's unit tests assert what it *wrote*; this asserts what the
//! loader *reads back*, which is a different question and the one that matters
//! for a blob nobody has run yet. In particular the TP degree is NOT a header
//! field — `DevBlob::parse` recovers it by scanning the emitted collectives —
//! so a blob whose XReduce operands are wrong reports the wrong world size here
//! rather than at rendezvous time on eight GPUs.
//!
//!   cargo run -p plowrt --example parse_blob -- <model.pkt> [--dump [prog]]
//!
//! `--dump` prints every tensor and every instruction (opcode, tensor names,
//! immediates). Pass a program index to dump one bucket (`0` = first prefill,
//! last = decode). Stream/wait tables print as counts only.

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

fn op_name(op: u16) -> String {
    packet::dev::DevOp::from_u16(op)
        .map(|o| format!("{o:?}"))
        .unwrap_or_else(|| format!("op#{op}"))
}

fn tensor_name(blob: &plowrt::asset::devblob::DevBlob, h: u16) -> String {
    if h == packet::dev::TENSOR_NONE16 {
        return "-".into();
    }
    blob.tensors
        .get(h as usize)
        .map(|t| t.name.clone())
        .unwrap_or_else(|| format!("t#{h}"))
}

fn dump_inst(blob: &plowrt::asset::devblob::DevBlob, i: usize, d: &packet::dev::DevInst64) {
    let t: Vec<String> = d.t.iter().map(|&h| tensor_name(blob, h)).collect();
    let i_slots: Vec<String> = d
        .i
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != 0)
        .map(|(k, v)| format!("i[{k}]={v}"))
        .collect();
    let f0 = f32::from_bits(d.fj[0]);
    let f1 = f32::from_bits(d.fj[1]);
    let mut extra = i_slots;
    if f0 != 0.0 {
        extra.push(format!("f0={f0}"));
    }
    if f1 != 0.0 {
        extra.push(format!("fj1={f1}"));
    }
    if d.fj[2] != 0 {
        extra.push(format!("j1={}", d.fj[2]));
    }
    let extras = if extra.is_empty() {
        String::new()
    } else {
        format!("  {}", extra.join(" "))
    };
    println!(
        "    [{i:>4}] {:<22} blocks={:<4} t=[{}]{extras}",
        op_name(d.op),
        d.blocks,
        t.join(", "),
    );
}

fn dump_prog(blob: &plowrt::asset::devblob::DevBlob, pi: usize) {
    let p = &blob.progs[pi];
    let last = blob.progs.len().saturating_sub(1);
    let kind = if pi == last { "decode" } else { "prefill" };
    println!(
        "\n== prog[{pi}] T={} {kind}  insts={} stream={} waits={} succs={} counters={} l2_domains={}",
        p.t,
        p.insts.len(),
        p.stream.len(),
        p.waits.len(),
        p.succs.len(),
        p.n_counter,
        p.l2_domains
    );
    for (i, d) in p.insts.iter().enumerate() {
        dump_inst(blob, i, d);
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: parse_blob <model.pkt> [--dump [prog]]");
    let dump = matches!(args.next().as_deref(), Some("--dump"));
    let dump_prog_i: Option<usize> = if dump {
        args.next().and_then(|s| s.parse().ok())
    } else {
        None
    };

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
    if !blob.sections.is_empty() {
        println!("sections  {}", blob.sections.len());
        for s in &blob.sections {
            println!("  kind={} name={:?} size={}", s.kind, s.name, s.size);
        }
    }

    if dump {
        println!("\n== tensors ({}) ==", blob.tensors.len());
        for (i, t) in blob.tensors.iter().enumerate() {
            let init = t
                .init
                .as_ref()
                .map(|r| format!(" init={r:?}"))
                .unwrap_or_default();
            println!("  [{i:>3}] {:>12} B  {}{init}", t.bytes, t.name);
        }
        match dump_prog_i {
            Some(i) if i < blob.progs.len() => dump_prog(&blob, i),
            Some(i) => {
                eprintln!("--dump {i}: program index out of range (0..{})", blob.progs.len());
                std::process::exit(1);
            }
            None => {
                for i in 0..blob.progs.len() {
                    dump_prog(&blob, i);
                }
            }
        }
    }
}
