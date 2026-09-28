//! Time one program role of a packet on the CUDA packet runtime (inputs left zeroed):
//! `packet_bench PACKET PIPELINE ROLE [ITERS] [--sweep]`. With `--sweep` every instruction cap
//! 0..=n_inst is timed in this process (JSON lines `{"cap":..,"us":..}`): the marginal cost of
//! instruction i is us(cap = i + 1) - us(cap = i). With `--each` and ROLE a prefix, every program
//! of the role sequence `ROLE.<n>` is timed alone (JSON lines `{"role":..,"program":..,"us":..}`);
//! with `--seq` the whole sequence is timed as one CUDA graph.
//! `PB_PRE=ROLE,..` runs those programs once first (e.g. the encoder, so a CFM program sees real
//! key lengths instead of zeroed inputs).

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("packet_bench requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::gpu::packet_exec::CudaPacketRuntime;
    use plowrt::exec::packet_runtime::{PacketAsset, PacketRuntime};
    let args: Vec<String> = std::env::args().collect();
    let sweep = args.iter().any(|a| a == "--sweep");
    let each = args.iter().any(|a| a == "--each");
    let pos: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with("--")).collect();
    if !(3..=4).contains(&pos.len()) {
        return Err("usage: packet_bench PACKET PIPELINE ROLE [ITERS] [--sweep|--each]".into());
    }
    let path = std::path::Path::new(pos[0]);
    let iters: usize = pos.get(3).map_or(Ok(50), |s| s.parse())?;
    let mut rt = CudaPacketRuntime::load(path, 0)?;
    let asset = PacketAsset::load(path)?;
    let pipeline = asset.bind(pos[1], &rt)?;
    if each {
        for (n, program) in pipeline.program_sequence(pos[2])?.into_iter().enumerate() {
            for _ in 0..3 {
                rt.run(program)?;
            }
            let mut us: Vec<f64> = (0..iters).map(|_| rt.run(program).map(|_| rt.last_run_us())).collect::<Result<_, _>>()?;
            us.sort_by(f64::total_cmp);
            println!("{{\"role\":\"{}.{n}\",\"program\":{program},\"us\":{:.2}}}", pos[2], us[iters / 2]);
        }
        return Ok(());
    }
    if args.iter().any(|a| a == "--seq") {
        // The whole role sequence `ROLE.<n>` as one CUDA graph (inputs zeroed).
        let programs = pipeline.program_sequence(pos[2])?;
        for _ in 0..3 {
            rt.run_sequence(&programs)?;
        }
        let mut us: Vec<f64> =
            (0..iters).map(|_| rt.run_sequence(&programs).map(|_| rt.last_run_us())).collect::<Result<_, _>>()?;
        us.sort_by(f64::total_cmp);
        println!("{{\"role\":\"{}\",\"programs\":{},\"us\":{:.2}}}", pos[2], programs.len(), us[iters / 2]);
        return Ok(());
    }
    let program = pipeline.program(pos[2])?;
    if let Ok(pre) = std::env::var("PB_PRE") {
        for role in pre.split(',') {
            rt.run(pipeline.program(role)?)?;
        }
    }
    let mut time = |rt: &mut CudaPacketRuntime| -> Result<f64, Box<dyn std::error::Error>> {
        for _ in 0..3 {
            rt.run(program)?;
        }
        let mut us = Vec::with_capacity(iters);
        for _ in 0..iters {
            rt.run(program)?;
            us.push(rt.last_run_us());
        }
        us.sort_by(f64::total_cmp);
        Ok(us[iters / 2])
    };
    if sweep {
        let raw = std::fs::read(path)?;
        let n_inst = plowrt::asset::devblob::DevBlob::parse(&raw)?.progs[program].insts.len();
        for cap in 0..=n_inst {
            rt.set_debug_max_inst(cap as u32)?;
            println!("{{\"cap\":{cap},\"us\":{:.2}}}", time(&mut rt)?);
        }
        rt.set_debug_max_inst(u32::MAX)?;
    }
    let us = time(&mut rt)?;
    println!("role={} program={program} median_us={us:.1}", pos[2]);
    Ok(())
}
