#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    use plowrt::asset::gguf::GgufFile;
    use plowrt::exec::apple::q8::Q8Gemm;

    let args: Vec<_> = std::env::args().collect();
    if !(4..=6).contains(&args.len()) {
        return Err(
            "usage: asr_nemotron_q8_gemm_metal MODEL TENSOR M [REPEATS [TILE_ROWS]]".into(),
        );
    }
    let m = args[3].parse::<usize>()?;
    let repeats = args.get(4).map_or(Ok(20), |value| value.parse::<usize>())?;
    let tile_rows = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if m == 0 || repeats == 0 {
        return Err("M and REPEATS must be positive".into());
    }
    let model = GgufFile::open(Path::new(&args[1]))?;
    let tensor = model.tensor(&args[2])?;
    let k = usize::try_from(tensor.dimensions[0])?;
    let n = usize::try_from(tensor.dimensions[1])?;
    let tile_rows = tile_rows.unwrap_or(if m >= 192 || n >= 2048 { 64 } else { 32 });
    let input: Vec<_> = (0..m * k)
        .map(|index| ((index.wrapping_mul(17) % 251) as f32 - 125.0) / 127.0)
        .collect();

    let reference_rows = m.min(2);
    let mut reference = vec![0.0; reference_rows * n];
    tensor.q8_0_matmul(&input[..reference_rows * k], reference_rows, &mut reference)?;
    let mut kernel = Q8Gemm::with_tile_rows(tensor.q8_0_matrix()?, m, tile_rows)?;
    let mut actual = vec![0.0; m * n];
    for _ in 0..3 {
        kernel.run(&input, &mut actual)?;
    }
    let mut timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        timings.push(kernel.run(&input, &mut actual)?);
    }
    timings.sort_by(f64::total_cmp);
    let compared = &actual[..reference.len()];
    let max_abs = compared
        .iter()
        .zip(&reference)
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0f32, f32::max);
    let max_rel = compared
        .iter()
        .zip(&reference)
        .filter(|(_, reference)| reference.abs() > 1e-5)
        .map(|(actual, reference)| ((actual - reference) / reference).abs())
        .fold(0.0f32, f32::max);
    if max_abs > 1e-2 {
        return Err(format!("Q8_0 GEMM Metal parity failed: max_abs={max_abs}").into());
    }
    let median = timings[timings.len() / 2];
    let operations = 2.0 * m as f64 * n as f64 * k as f64;
    let weight_bytes_per_tile = tensor.bytes.len() as f64 * m.div_ceil(32) as f64;
    println!(
        "{}",
        serde_json::json!({
            "tensor": tensor.name,
            "dimensions": tensor.dimensions,
            "m": m,
            "reference_rows": reference_rows,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "gpu_us_min": timings[0],
            "gpu_us_median": median,
            "effective_tops": operations / median / 1e6,
            "effective_q8_weight_gbps": weight_bytes_per_tile / median / 1e3,
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
