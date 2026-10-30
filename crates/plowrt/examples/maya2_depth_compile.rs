//! Maya 2 Native's DepthV2 (a `scripts/tts/maya2_prep.py` directory) to a standalone `depth.pkt`
//! for `packet_run` parity runs: `maya2_depth_compile PREP_DIR GPU N_CU OUT.pkt`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 {
        return Err("usage: maya2_depth_compile PREP_DIR GPU N_CU OUT.pkt".into());
    }
    let (model, section) = devgen::maya2_depth::lower_maya2_depth(
        std::path::Path::new(&args[1]),
        args[3].parse()?,
        packet::devbuild::gpu_fingerprint(&args[2]),
    )?;
    devgen::write_sidecar_packet(std::path::Path::new(&args[4]), &model, &[section]);
    Ok(())
}
