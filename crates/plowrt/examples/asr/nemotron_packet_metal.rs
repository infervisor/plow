mod nemotron_packet_support;

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    use devgen::conformer::{self, ConformerSpec};
    use nemotron_packet_support::LayerNames;
    use plowrt::asr::conformer::AttentionMask;
    use plowrt::asr::frontend::decode_wav;
    use plowrt::asr::nemotron::{
        conformer_encoder_plan, pre_encode_projection, NemotronFrontend, NemotronSubsampler,
    };
    use plowrt::asset::gguf::GgufFile;

    let args: Vec<_> = std::env::args().collect();
    if !(3..=7).contains(&args.len()) {
        return Err(
            "usage: asr_nemotron_packet_metal MODEL AUDIO [REPEATS] [N_CU] [LAYERS] [OUTPUT_F32]"
                .into(),
        );
    }
    let repeats = args.get(3).map_or(Ok(5), |value| value.parse::<usize>())?;
    let n_cu = args.get(4).map_or(Ok(16), |value| value.parse::<u32>())?;
    let layer_limit = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if repeats == 0 || n_cu == 0 {
        return Err("REPEATS and N_CU must be positive".into());
    }

    let gguf = GgufFile::open(Path::new(&args[1]))?;
    let frontend = NemotronFrontend::load(&gguf)?;
    let subsampler = NemotronSubsampler::load(&gguf)?;
    let projection = pre_encode_projection(&gguf)?;
    let samples = decode_wav(&std::fs::read(&args[2])?)?;
    let features = frontend.extract(&samples)?;
    let subsampled = subsampler.run(&features)?;
    let frames = subsampled.frames;
    let mut projected = vec![0.0; frames * projection.weight().n()];
    projection.matmul(&subsampled.values, frames, &mut projected)?;

    let reference_plan = conformer_encoder_plan(&gguf)?;
    let layer_count = layer_limit
        .unwrap_or(reference_plan.blocks().len())
        .min(reference_plan.blocks().len());
    if layer_count == 0 {
        let output_path = args.get(6).ok_or("LAYERS=0 requires OUTPUT_F32")?;
        std::fs::write(output_path, bytemuck::cast_slice(&projected))?;
        println!(
            "{}",
            serde_json::json!({
                "input_shape": [features.frames, features.bins],
                "encoder_shape": [frames, reference_plan.width()],
                "layers": 0,
            })
        );
        return Ok(());
    }
    let gate_plan = plowrt::asr::conformer::ConformerEncoderPlan::new(
        reference_plan.blocks()[..layer_count].to_vec(),
    )?;
    let first = gate_plan
        .blocks()
        .first()
        .ok_or("encoder has no Conformer layers")?;
    let AttentionMask::ChunkedLimited {
        chunk_size,
        left_chunks,
    } = first.attention.mask
    else {
        return Err("packet gate requires chunk-limited attention".into());
    };
    let layer_names: Vec<_> = (0..gate_plan.blocks().len()).map(LayerNames::new).collect();
    let layers: Vec<_> = layer_names.iter().map(LayerNames::borrow).collect();
    let mut packets = conformer::lower(
        ConformerSpec {
            frames: frames.try_into()?,
            width: reference_plan.width().try_into()?,
            feed_forward_width: first.feed_forward1.expand.n().try_into()?,
            heads: first.attention.heads.try_into()?,
            convolution_kernel: first.convolution.kernel.try_into()?,
            chunk_size: chunk_size.try_into()?,
            left_chunks: left_chunks.try_into()?,
            position_table: "encoder.pos_enc.pe",
            position_count: first.attention.position_count.try_into()?,
            position_center: first.attention.position_center.try_into()?,
            epsilon: first.output_norm.epsilon(),
        },
        &layers,
        n_cu,
    )?;
    packets.embed_weights(|name| {
        gguf.tensor(name)
            .map(|tensor| tensor.bytes.to_vec())
            .map_err(|error| error.to_string())
    })?;

    let blob_path = std::env::temp_dir().join(format!(
        "plow-nemotron-packets-{}-{}.plowdev",
        std::process::id(),
        frames
    ));
    let section = packets.pipeline_section()?;
    std::fs::write(&blob_path, packets.model.to_blob_v6(&[section]))?;
    let output_path = args.get(6).map(std::path::PathBuf::from);
    let result = run_gate(
        &blob_path,
        &projected,
        packets.input as usize,
        repeats,
        &gate_plan,
        frames,
        features.frames,
        features.bins,
        n_cu,
        output_path.as_deref(),
    );
    let _ = std::fs::remove_file(&blob_path);
    result
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[allow(clippy::too_many_arguments)]
fn run_gate(
    blob_path: &std::path::Path,
    input: &[f32],
    input_handle: usize,
    repeats: usize,
    reference_plan: &plowrt::asr::conformer::ConformerEncoderPlan<'_>,
    frames: usize,
    feature_frames: usize,
    feature_bins: usize,
    n_cu: u32,
    output_path: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use plowrt::exec::apple::asr_conformer::MetalConformerEncoder;
    use plowrt::exec::apple::MetalEngine;

    let mut reference_engine = MetalConformerEncoder::new(reference_plan, frames)?;
    let mut expected = vec![0.0; input.len()];
    for _ in 0..2 {
        reference_engine.run(input, &mut expected)?;
    }
    let mut reference_timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = Instant::now();
        reference_engine.run(input, &mut expected)?;
        reference_timings.push(started.elapsed().as_secs_f64() * 1e6);
    }

    let mut engine = MetalEngine::load_packet(blob_path)?;
    let runtime_input = engine
        .packet_tensor("act.asr.io")
        .ok_or("packet input tensor is missing")?;
    if runtime_input != input_handle {
        return Err("serialized packet input handle changed".into());
    }
    engine.write_packet_f32(runtime_input, input)?;
    for _ in 0..2 {
        engine.run_packet(0)?;
        engine.write_packet_f32(runtime_input, input)?;
    }
    let mut timings = Vec::with_capacity(repeats);
    let mut device_timings = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        engine.set_profiling(true);
        engine.run_packet(0)?;
        timings.push(engine.last_run_us);
        device_timings.push(
            engine
                .profile
                .ok_or("packet profile is missing")?
                .gpu_device_ms
                * 1e3,
        );
        engine.write_packet_f32(runtime_input, input)?;
    }
    engine.run_packet(0)?;
    let actual = engine.read_packet_f32(runtime_input)?;
    if let Some(path) = output_path {
        std::fs::write(path, bytemuck::cast_slice(&actual))?;
    }
    reference_timings.sort_by(f64::total_cmp);
    timings.sort_by(f64::total_cmp);
    device_timings.sort_by(f64::total_cmp);
    let differences = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs());
    let max_abs = differences.clone().fold(0.0f32, f32::max);
    let mean_abs = differences.map(f64::from).sum::<f64>() / actual.len() as f64;
    println!(
        "{}",
        serde_json::json!({
            "input_shape": [feature_frames, feature_bins],
            "encoder_shape": [frames, reference_plan.width()],
            "layers": reference_plan.blocks().len(),
            "n_cu": n_cu,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "reference_us_min": reference_timings[0],
            "reference_us_median": reference_timings[reference_timings.len() / 2],
            "gpu_us_min": timings[0],
            "gpu_us_median": timings[timings.len() / 2],
            "gpu_device_us_min": device_timings[0],
            "gpu_device_us_median": device_timings[device_timings.len() / 2],
            "repeats": repeats,
        })
    );
    if max_abs > 5e-2 {
        return Err(format!("packet Conformer parity failed: max_abs={max_abs}").into());
    }
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires macOS and --features metal,gguf");
    std::process::exit(1);
}
