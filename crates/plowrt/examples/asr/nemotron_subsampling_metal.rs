#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    use plowrt::asr::frontend::decode_wav;
    use plowrt::asr::nemotron::{subsampling_plan, NemotronFrontend, NemotronSubsampler};
    use plowrt::asset::gguf::GgufFile;
    use plowrt::exec::apple::asr_subsampling::MetalSubsampler;

    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: asr_nemotron_subsampling_metal MODEL AUDIO [REPEATS]".into());
    }
    let repeats = args.get(3).map_or(Ok(20), |value| value.parse::<usize>())?;
    if repeats == 0 {
        return Err("REPEATS must be positive".into());
    }
    let model = GgufFile::open(Path::new(&args[1]))?;
    let frontend = NemotronFrontend::load(&model)?;
    let scalar = NemotronSubsampler::load(&model)?;
    let samples = decode_wav(&std::fs::read(&args[2])?)?;
    let features = frontend.extract(&samples)?;
    let reference = scalar.run(&features)?;
    let plan = subsampling_plan(&model)?;
    let mut kernel = MetalSubsampler::new(&plan, features.frames)?;
    if kernel.output_shape() != [reference.frames, reference.width, reference.channels] {
        return Err("Metal output shape mismatch".into());
    }
    let mut actual = vec![0.0; reference.values.len()];
    for _ in 0..3 {
        kernel.run(&features, &mut actual)?;
    }
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        timings.push(kernel.run(&features, &mut actual)?);
    }
    timings.sort_by(f64::total_cmp);
    let errors = actual
        .iter()
        .zip(&reference.values)
        .map(|(actual, reference)| (actual - reference).abs());
    let max_abs = errors.clone().fold(0.0f32, f32::max);
    let mean_abs = errors.map(f64::from).sum::<f64>() / actual.len() as f64;
    if max_abs > 2e-2 {
        return Err(format!("Nemotron Metal subsampling parity failed: {max_abs}").into());
    }
    println!(
        "{}",
        serde_json::json!({
            "input_shape": [features.frames, features.bins],
            "output_shape": kernel.output_shape(),
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
