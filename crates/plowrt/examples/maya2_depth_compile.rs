//! Maya 2 Native's DepthV2 (a `scripts/tts/maya2_prep.py` directory) to a standalone `depth.pkt`
//! for `packet_run` parity runs: `maya2_depth_compile PREP_DIR GPU N_CU OUT.pkt`.
//! `--block LEVEL:LAYER --batch B` emits one layer (level starts at 1, layer at 0),
//! exposed as `maya2.depth.block` / `forward.0` for `packet_run` and `packet_bench`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 5 && !(args.len() == 9 && args[5] == "--block" && args[7] == "--batch") {
        return Err(
            "usage: maya2_depth_compile PREP_DIR GPU N_CU OUT.pkt [--block LEVEL:LAYER --batch B]"
                .into(),
        );
    }
    let dir = std::path::Path::new(&args[1]);
    let n_cu = args[3].parse()?;
    let target = packet::devbuild::gpu_fingerprint(&args[2]);
    let half =
        hwspec::registry::lookup(&args[2]).is_some_and(|gpu| gpu.arch == hwspec::Arch::Hopper);
    let (model, section) = if args.len() == 9 {
        let (level, layer) = args[6]
            .split_once(':')
            .ok_or("--block expects LEVEL:LAYER")?;
        devgen::maya2_depth::lower_maya2_depth_block(
            dir,
            n_cu,
            target,
            half,
            level.parse()?,
            layer.parse()?,
            args[8].parse()?,
        )?
    } else {
        devgen::maya2_depth::lower_maya2_depth(dir, n_cu, target, half)?
    };
    devgen::write_sidecar_packet(std::path::Path::new(&args[4]), &model, &[section]);
    Ok(())
}
