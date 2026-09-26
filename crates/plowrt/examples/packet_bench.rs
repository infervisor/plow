//! Time one program role of a packet on the CUDA packet runtime (inputs left zeroed):
//! `packet_bench PACKET PIPELINE ROLE [ITERS]`. With PLOW_DEBUG_MAX_INST=N only instructions
//! below N execute, which gives per-instruction marginal costs.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("packet_bench requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::packet_runtime::{load_packet_runtime, PacketAsset};
    let args: Vec<String> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: packet_bench PACKET PIPELINE ROLE [ITERS]".into());
    }
    let path = std::path::Path::new(&args[1]);
    let iters: usize = args.get(4).map_or(Ok(50), |s| s.parse())?;
    let mut loaded = load_packet_runtime(path, "cuda")?;
    let asset = PacketAsset::load(path)?;
    let pipeline = asset.bind(&args[2], loaded.runtime.as_ref())?;
    let program = pipeline.program(&args[3])?;
    for _ in 0..3 {
        loaded.runtime.run(program)?;
    }
    let mut us = Vec::with_capacity(iters);
    for _ in 0..iters {
        loaded.runtime.run(program)?;
        us.push(loaded.runtime.last_run_us());
    }
    us.sort_by(f64::total_cmp);
    println!("role={} program={program} median_us={:.1} min_us={:.1}", args[3], us[iters / 2], us[0]);
    Ok(())
}
