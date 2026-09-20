use std::path::PathBuf;
use plowrt::asset::devblob::DevBlob;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let pkt_path = PathBuf::from(args.next().unwrap_or_else(|| "/opt/dlami/nvme/tmp/campaign-gemma4-ladder-opt/assets/model.pkt".to_string()));
    let prog_idx: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(7);
    let buf = std::fs::read(&pkt_path)?;
    let blob = DevBlob::parse(&buf)?;
    for (i, p) in blob.progs.iter().enumerate() {
        println!("Program {}: t={}, role={:?}, insts={}", i, p.t, p.role, p.insts.len());
    }
    return Ok(());
    let p = &blob.progs[prog_idx];
    for (i, inst) in p.insts.iter().enumerate().skip(520) {
        println!("Inst {:2}: op={:2} blocks={:3} t={:?} i={:?} fj={:?}", i, inst.op, inst.blocks, inst.t, inst.i, inst.fj);
    }
    Ok(())
}
