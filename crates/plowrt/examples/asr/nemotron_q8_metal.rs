#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    use plowrt::asset::gguf::GgufFile;
    use plowrt::exec::apple::q8::Q8Gemv;

    let args: Vec<_> = std::env::args().collect();
    if !(3..=6).contains(&args.len()) {
        return Err(
            "usage: asr_nemotron_q8_metal MODEL TENSOR [REPEATS [SIMDGROUPS [ROWS_PER_SIMD]]]"
                .into(),
        );
    }
    let repeats = args.get(3).map_or(Ok(20), |value| value.parse::<usize>())?;
    if repeats == 0 {
        return Err("REPEATS must be positive".into());
    }
    let simdgroups = args.get(4).map_or(Ok(8), |value| value.parse::<usize>())?;
    let rows_per_simd = args.get(5).map_or(Ok(1), |value| value.parse::<usize>())?;
    let model = GgufFile::open(Path::new(&args[1]))?;
    let tensor = model.tensor(&args[2])?;
    let k = usize::try_from(tensor.dimensions[0])?;
    let n = usize::try_from(tensor.dimensions[1])?;
    let input: Vec<_> = (0..k)
        .map(|index| ((index.wrapping_mul(17) % 251) as f32 - 125.0) / 127.0)
        .collect();
    let mut reference = vec![0.0; n];
    tensor.q8_0_matvec(&input, &mut reference)?;
    let mut kernel = Q8Gemv::with_layout(tensor.q8_0_matrix()?, simdgroups, rows_per_simd)?;
    let mut actual = vec![0.0; n];
    for _ in 0..3 {
        kernel.run(&input, &mut actual)?;
    }
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        timings.push(kernel.run(&input, &mut actual)?);
    }
    timings.sort_by(f64::total_cmp);
    let max_abs = actual
        .iter()
        .zip(&reference)
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0f32, f32::max);
    let max_rel = actual
        .iter()
        .zip(&reference)
        .filter(|(_, reference)| reference.abs() > 1e-5)
        .map(|(actual, reference)| ((actual - reference) / reference).abs())
        .fold(0.0f32, f32::max);
    if max_abs > 2e-4 {
        return Err(format!("Q8_0 Metal parity failed: max_abs={max_abs}").into());
    }
    println!(
        "{}",
        serde_json::json!({
            "tensor": tensor.name,
            "dimensions": tensor.dimensions,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "gpu_us_min": timings[0],
            "gpu_us_median": timings[timings.len() / 2],
            "repeats": repeats,
            "simdgroups": simdgroups,
            "rows_per_simd": rows_per_simd,
        })
    );
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires macOS and --features metal,gguf");
    std::process::exit(1);
}
