#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    use plowrt::asr::conformer::CpuConformerBlock;
    use plowrt::asr::nemotron::conformer_block_plan;
    use plowrt::asset::gguf::GgufFile;
    use plowrt::exec::apple::asr_conformer::MetalConformerBlock;

    let args: Vec<_> = std::env::args().collect();
    if !(3..=5).contains(&args.len()) {
        return Err("usage: asr_nemotron_conformer_metal MODEL FRAMES [LAYER] [REPEATS]".into());
    }
    let frames = args[2].parse::<usize>()?;
    let layer = args.get(3).map_or(Ok(0), |value| value.parse::<usize>())?;
    let repeats = args.get(4).map_or(Ok(20), |value| value.parse::<usize>())?;
    if frames == 0 || repeats == 0 {
        return Err("FRAMES and REPEATS must be positive".into());
    }
    let model = GgufFile::open(Path::new(&args[1]))?;
    let plan = conformer_block_plan(&model, layer)?;
    let input: Vec<_> = (0..frames * plan.width())
        .map(|index| ((index.wrapping_mul(17) % 251) as f32 - 125.0) / 127.0)
        .collect();
    let expected = CpuConformerBlock::new(plan).run(&input, frames)?;
    let mut executor = MetalConformerBlock::new(&plan, frames)?;
    let mut actual = vec![0.0; expected.len()];
    for _ in 0..3 {
        executor.run(&input, &mut actual)?;
    }
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        timings.push(executor.run(&input, &mut actual)?);
    }
    timings.sort_by(f64::total_cmp);
    let differences = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs());
    let max_abs = differences.clone().fold(0.0f32, f32::max);
    let mean_abs = differences.map(f64::from).sum::<f64>() / actual.len() as f64;
    if max_abs > 5e-2 {
        return Err(format!("Metal Conformer parity failed: max_abs={max_abs}").into());
    }
    println!(
        "{}",
        serde_json::json!({
            "layer": layer,
            "shape": [frames, plan.width()],
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "gpu_us_min": timings[0],
            "gpu_us_median": timings[timings.len() / 2],
            "repeats": repeats,
        })
    );
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires macOS and --features metal,gguf");
    std::process::exit(1);
}
