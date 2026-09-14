#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    use plowrt::asr::frontend::decode_wav;
    use plowrt::asr::nemotron::{pre_encode_projection, NemotronFrontend, NemotronSubsampler};
    use plowrt::asset::gguf::GgufFile;
    use plowrt::exec::apple::q8::Q8Gemm;

    let args: Vec<_> = std::env::args().collect();
    if !(3..=5).contains(&args.len()) {
        return Err(
            "usage: asr_nemotron_projection_metal MODEL AUDIO [REPEATS] [TILE_ROWS]".into(),
        );
    }
    let repeats = args.get(3).map_or(Ok(20), |value| value.parse::<usize>())?;
    if repeats == 0 {
        return Err("REPEATS must be positive".into());
    }
    let tile_rows = args
        .get(4)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    let model = GgufFile::open(Path::new(&args[1]))?;
    let frontend = NemotronFrontend::load(&model)?;
    let subsampler = NemotronSubsampler::load(&model)?;
    let projection = pre_encode_projection(&model)?;
    let samples = decode_wav(&std::fs::read(&args[2])?)?;
    let features = frontend.extract(&samples)?;
    let subsampled = subsampler.run(&features)?;
    let rows = subsampled.frames;
    let output_width = projection.weight().n();
    let mut expected = vec![0.0; rows * output_width];
    projection.matmul(&subsampled.values, rows, &mut expected)?;

    let mut kernel = match tile_rows {
        Some(tile_rows) => Q8Gemm::linear_with_tile_rows(&projection, rows, tile_rows)?,
        None => Q8Gemm::linear(&projection, rows)?,
    };
    let mut actual = vec![0.0; expected.len()];
    for _ in 0..3 {
        kernel.run(&subsampled.values, &mut actual)?;
    }
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        timings.push(kernel.run(&subsampled.values, &mut actual)?);
    }
    timings.sort_by(f64::total_cmp);
    let errors = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs());
    let max_abs = errors.clone().fold(0.0f32, f32::max);
    let mean_abs = errors.map(f64::from).sum::<f64>() / actual.len() as f64;
    if max_abs > 2e-2 {
        return Err(format!("Nemotron projection parity failed: {max_abs}").into());
    }
    println!(
        "{}",
        serde_json::json!({
            "input_shape": [rows, subsampled.width * subsampled.channels],
            "output_shape": [rows, output_width],
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "gpu_us_min": timings[0],
            "gpu_us_median": timings[timings.len() / 2],
            "repeats": repeats,
            "tile_rows": tile_rows,
        })
    );
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires macOS and --features metal,gguf");
    std::process::exit(1);
}
