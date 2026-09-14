#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    use plowrt::asr::conformer::CpuConformerEncoder;
    use plowrt::asr::frontend::decode_wav;
    use plowrt::asr::nemotron::{
        conformer_encoder_plan, pre_encode_projection, NemotronFrontend, NemotronSubsampler,
    };
    use plowrt::asset::gguf::GgufFile;
    use plowrt::exec::apple::asr_conformer::MetalConformerEncoder;

    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: asr_nemotron_encoder_metal MODEL AUDIO [REPEATS]".into());
    }
    let repeats = args.get(3).map_or(Ok(5), |value| value.parse::<usize>())?;
    if repeats == 0 {
        return Err("REPEATS must be positive".into());
    }
    let model = GgufFile::open(Path::new(&args[1]))?;
    let frontend = NemotronFrontend::load(&model)?;
    let subsampler = NemotronSubsampler::load(&model)?;
    let projection = pre_encode_projection(&model)?;
    let samples = decode_wav(&std::fs::read(&args[2])?)?;
    let features = frontend.extract(&samples)?;
    let subsampled = subsampler.run(&features)?;
    let frames = subsampled.frames;
    let mut projected = vec![0.0; frames * projection.weight().n()];
    projection.matmul(&subsampled.values, frames, &mut projected)?;
    let expected =
        CpuConformerEncoder::new(conformer_encoder_plan(&model)?).run(&projected, frames)?;
    let plan = conformer_encoder_plan(&model)?;
    let mut executor = MetalConformerEncoder::new(&plan, frames)?;
    let mut actual = vec![0.0; expected.len()];
    for _ in 0..2 {
        executor.run(&projected, &mut actual)?;
    }
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        timings.push(executor.run(&projected, &mut actual)?);
    }
    timings.sort_by(f64::total_cmp);
    let differences = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs());
    let max_abs = differences.clone().fold(0.0f32, f32::max);
    let mean_abs = differences.map(f64::from).sum::<f64>() / actual.len() as f64;
    if max_abs > 5e-2 {
        return Err(format!("Metal Conformer encoder parity failed: max_abs={max_abs}").into());
    }
    println!(
        "{}",
        serde_json::json!({
            "input_shape": [features.frames, features.bins],
            "encoder_shape": [frames, plan.width()],
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
