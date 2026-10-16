use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let pkt_path = PathBuf::from(args.get(1).map(|s| s.as_str()).unwrap_or("/home/lava/bench/dsv41/pkt/dsv41_tp8_l0.pkt"));
    let raw = std::fs::read(&pkt_path)?;
    let blob = plowrt::asset::devblob::DevBlob::parse(&raw)?;

    println!("Blob has {} tensors, {} programs", blob.tensors.len(), blob.progs.len());
    for (ti, t) in blob.tensors.iter().enumerate() {
        println!("  tensor[{:2}]: name={:35} bytes={}", ti, t.name, t.bytes);
    }
    for (pi, p) in blob.progs.iter().enumerate() {
        println!("\n=== Program {} (t={}, role={:?}, insts={}) ===", pi, p.t, p.role, p.insts.len());
        for (ii, inst) in p.insts.iter().enumerate() {
            let op_name = match packet::dev::DevOp::from_u16(inst.op) {
                Some(op) => format!("{:?}", op),
                None => format!("Op#{}", inst.op),
            };
            println!("  inst[{:2}]: op {:?} ({}) | blocks={} i={:?} t={:?}", ii, inst.op, op_name, inst.blocks, inst.i, inst.t);
        }
        println!("Stream entries: {}", p.stream.len());
        let mut seg_insts: std::collections::BTreeMap<u16, std::collections::BTreeSet<u32>> = std::collections::BTreeMap::new();
        for se in &p.stream {
            seg_insts.entry(se.seg).or_default().insert(se.inst);
        }
        for (seg, insts) in &seg_insts {
            println!("  Segment {:2}: insts {:?}", seg, insts);
        }
        #[cfg(feature = "hsa")]
        if let Ok(classes) = plowrt::exec::amd::derive_segments(p) {
            println!("  Derived wave classes (len={}):", classes.len());
            for (seg, cls) in classes.iter().enumerate() {
                if let Some(insts) = seg_insts.get(&(seg as u16)) {
                    println!("    Segment {:2} (insts {:?}): wave class {}", seg, insts, cls);
                }
            }
        }
    }
    Ok(())
}
