//! List one program's instructions (index, op) of a packet, for attributing `packet_bench --sweep`
//! marginal costs: `packet_ops PACKET PROGRAM_INDEX`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: packet_ops PACKET PROGRAM_INDEX".into());
    }
    let raw = std::fs::read(&args[1])?;
    let blob = plowrt::asset::devblob::DevBlob::parse_l2(&raw, true)?;
    let p = blob
        .progs
        .get(args[2].parse::<usize>()?)
        .ok_or("no such program")?;
    for (i, inst) in p.insts.iter().enumerate() {
        let op = packet::dev::DevOp::from_u16(inst.op)
            .map_or_else(|| format!("op{}", inst.op), |o| format!("{o:?}"));
        println!("{i} {op} rows={} i1={} i2={} flags={}", inst.i[0], inst.i[1], inst.i[2], inst.i[7]);
    }
    Ok(())
}
