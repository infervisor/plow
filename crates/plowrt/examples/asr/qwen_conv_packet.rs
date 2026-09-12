use std::path::Path;

use devgen::asr::qwen::lower_audio_encoder;
use plowrt::asr::frontend::MelFeatures;
use plowrt::exec::apple::asr::QwenAudioEncoder;
use plowrt::exec::packet_runtime::{load_packet_runtime, PacketAsset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(5..=7).contains(&args.len()) {
        return Err(
            "usage: asr_qwen_conv_packet CHECKPOINT MEL_F32 FEATURE_FRAMES OUTPUT_PACKET [BACKEND] [REPEATS]"
                .into(),
        );
    }
    let checkpoint_path = Path::new(&args[1]);
    let feature_frames = args[3].parse::<usize>()?;
    let backend = args.get(5).map(String::as_str).unwrap_or("auto");
    let repeats = args.get(6).map_or(Ok(3), |value| value.parse::<usize>())?;
    if !(50..=3000).contains(&feature_frames) || repeats == 0 {
        return Err("FEATURE_FRAMES must be 50..=3000 and REPEATS must be positive".into());
    }
    let chunks = feature_frames.div_ceil(100);
    let chunk_frames = feature_frames.min(100);
    let mut lowered = lower_audio_encoder(feature_frames.try_into()?, 16)?;
    let input_handle = lowered.input;
    let convolution_handle = lowered.convolution;
    let projection_handle = lowered.projection;
    let positioned_handle = lowered.positioned;
    let norm_handle = lowered.first_norm;
    let qkv_handles = lowered.first_qkv;
    let attention_handle = lowered.first_attention;
    let layer_output_handle = lowered.first_output;
    let transformer_output_handle = lowered.transformer_output;
    let audio_output_handle = lowered.prefix.output;
    let rows = lowered.rows as usize;
    let output_shape = lowered.output_shape.map(|dimension| dimension as usize);
    lowered.embed_checkpoint(checkpoint_path)?;
    let prefix = lowered.prefix;
    let pipeline = prefix.forward_pipeline_section(
        "encode",
        plow_asset::packet_pipeline::PipelineDType::F32,
        plow_asset::packet_pipeline::PipelineDType::F32,
        vec![rows as u64, 2048],
    )?;
    std::fs::write(&args[4], prefix.model.to_blob_v6(&[pipeline]))?;

    let mel = read_f32(Path::new(&args[2]))?;
    if mel.len() != 128 * feature_frames {
        return Err("MEL_F32 size does not match FEATURE_FRAMES".into());
    }
    let mut input = vec![0.0f32; chunks * 128 * chunk_frames];
    for batch in 0..chunks {
        for bin in 0..128 {
            for frame in 0..chunk_frames {
                if batch * 100 + frame < feature_frames {
                    input[(batch * 128 + bin) * chunk_frames + frame] =
                        round_bf16(mel[bin * feature_frames + batch * 100 + frame]);
                }
            }
        }
    }
    let asset = PacketAsset::load(Path::new(&args[4]))?;
    let loaded = load_packet_runtime(Path::new(&args[4]), backend)?;
    let selected_backend = loaded.backend;
    let mut runtime = loaded.runtime;
    let pipeline = asset.bind("encode", runtime.as_ref())?;
    if pipeline.driver() != "forward.v1" {
        return Err("packet encoder uses an unsupported driver".into());
    }
    let programs = pipeline.program_sequence("forward")?;
    let input_tensor = pipeline.tensor("input")?;
    if input_tensor.handle != input_handle as usize {
        return Err("packet input tensor is missing".into());
    }
    let convolution_tensor = runtime
        .tensor("act.conv2d.2")
        .filter(|tensor| tensor.handle == convolution_handle as usize)
        .ok_or("packet convolution tensor is missing")?;
    let projection_tensor = runtime
        .tensor("act.qwen.conv_projection")
        .filter(|tensor| tensor.handle == projection_handle as usize)
        .ok_or("packet projection tensor is missing")?;
    let positioned_tensor = runtime
        .tensor("act.qwen.positioned")
        .filter(|tensor| tensor.handle == positioned_handle as usize)
        .ok_or("packet positioned tensor is missing")?;
    let norm_tensor = runtime
        .tensor("act.qwen.layers.0.self_attn_norm")
        .filter(|tensor| tensor.handle == norm_handle as usize)
        .ok_or("packet layer-normalization tensor is missing")?;
    let qkv_tensors = [
        "act.qwen.layers.0.query",
        "act.qwen.layers.0.key",
        "act.qwen.layers.0.value",
    ]
    .into_iter()
    .zip(qkv_handles)
    .map(|(name, handle)| {
        runtime
            .tensor(name)
            .filter(|tensor| tensor.handle == handle as usize)
            .ok_or_else(|| format!("packet {name} tensor is missing"))
    })
    .collect::<Result<Vec<_>, _>>()?;
    let attention_tensor = runtime
        .tensor("act.qwen.layers.0.attention")
        .filter(|tensor| tensor.handle == attention_handle as usize)
        .ok_or("packet attention tensor is missing")?;
    let layer_output_tensor = runtime
        .tensor("act.qwen.layers.0.output")
        .filter(|tensor| tensor.handle == layer_output_handle as usize)
        .ok_or("packet first-layer output tensor is missing")?;
    let transformer_output_tensor = runtime
        .tensor("act.qwen.layers.23.output")
        .filter(|tensor| tensor.handle == transformer_output_handle as usize)
        .ok_or("packet transformer output tensor is missing")?;
    let audio_output_tensor = pipeline.tensor("output")?;
    if audio_output_tensor.handle != audio_output_handle as usize {
        return Err("packet audio output tensor is missing".into());
    }
    runtime.write_tensor(input_tensor, bytemuck::cast_slice(&input))?;
    let mut timings = vec![Vec::with_capacity(repeats); programs.len()];
    for _ in 0..repeats {
        for (stage, &program) in programs.iter().enumerate() {
            runtime.run(program)?;
            timings[stage].push(runtime.last_run_us());
        }
    }
    let medians: Vec<_> = timings
        .iter_mut()
        .map(|samples| {
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        })
        .collect();
    let mut convolution_bytes = vec![0u8; convolution_tensor.bytes];
    runtime.read_tensor(convolution_tensor, &mut convolution_bytes)?;
    let convolution: &[f32] = bytemuck::cast_slice(&convolution_bytes);
    let mut projection_bytes = vec![0u8; projection_tensor.bytes];
    runtime.read_tensor(projection_tensor, &mut projection_bytes)?;
    let projection: &[f32] = bytemuck::cast_slice(&projection_bytes);
    let mut positioned_bytes = vec![0u8; positioned_tensor.bytes];
    runtime.read_tensor(positioned_tensor, &mut positioned_bytes)?;
    let positioned: &[f32] = bytemuck::cast_slice(&positioned_bytes);
    let mut norm_bytes = vec![0u8; norm_tensor.bytes];
    runtime.read_tensor(norm_tensor, &mut norm_bytes)?;
    let norm: &[f32] = bytemuck::cast_slice(&norm_bytes);
    let mut qkv_bytes = Vec::with_capacity(3);
    for tensor in &qkv_tensors {
        let mut bytes = vec![0u8; tensor.bytes];
        runtime.read_tensor(*tensor, &mut bytes)?;
        qkv_bytes.push(bytes);
    }
    let qkv: Vec<&[f32]> = qkv_bytes
        .iter()
        .map(|bytes| bytemuck::cast_slice(bytes))
        .collect();
    let mut attention_bytes = vec![0u8; attention_tensor.bytes];
    runtime.read_tensor(attention_tensor, &mut attention_bytes)?;
    let attention: &[f32] = bytemuck::cast_slice(&attention_bytes);
    let mut layer_output_bytes = vec![0u8; layer_output_tensor.bytes];
    runtime.read_tensor(layer_output_tensor, &mut layer_output_bytes)?;
    let layer_output: &[f32] = bytemuck::cast_slice(&layer_output_bytes);
    let mut transformer_output_bytes = vec![0u8; transformer_output_tensor.bytes];
    runtime.read_tensor(transformer_output_tensor, &mut transformer_output_bytes)?;
    let transformer_output: &[f32] = bytemuck::cast_slice(&transformer_output_bytes);
    let mut audio_output_bytes = vec![0u8; audio_output_tensor.bytes];
    runtime.read_tensor(audio_output_tensor, &mut audio_output_bytes)?;
    let audio_output: &[f32] = bytemuck::cast_slice(&audio_output_bytes);
    let mut sequence_ms = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let started = std::time::Instant::now();
        runtime.run_sequence(&programs)?;
        sequence_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    sequence_ms.sort_by(f64::total_cmp);
    let sequence_ms = sequence_ms[sequence_ms.len() / 2];
    let mut sequence_output = vec![0u8; audio_output_tensor.bytes];
    runtime.read_tensor(audio_output_tensor, &mut sequence_output)?;
    if sequence_output != audio_output_bytes {
        return Err("packet sequence changed the audio encoder output".into());
    }
    if convolution
        .iter()
        .chain(projection)
        .chain(positioned)
        .chain(norm)
        .chain(qkv.iter().flat_map(|values| values.iter()))
        .chain(attention)
        .chain(layer_output)
        .chain(transformer_output)
        .chain(audio_output)
        .any(|value| !value.is_finite())
    {
        return Err("packet convolution produced a non-finite value".into());
    }
    let mut oracle = QwenAudioEncoder::load(checkpoint_path)?;
    oracle.set_packed_conv(true);
    oracle.set_tiled_linear(true);
    oracle.set_wide_linear(oracle.supports_wide_linear())?;
    oracle.set_large_linear(oracle.supports_large_linear())?;
    oracle.set_simd_attention(oracle.supports_simd_attention())?;
    if oracle.validate_direct_epilogue().unwrap_or(false) {
        oracle.set_direct_epilogue(true)?;
    }
    let features = MelFeatures {
        values: mel,
        frames: feature_frames,
    };
    let (convolution_reference, reference_shape) = oracle.convolution_output(&features)?;
    if reference_shape != output_shape || convolution_reference.len() != convolution.len() {
        return Err("packet and direct convolution shapes differ".into());
    }
    let (convolution_max_abs, convolution_mean_abs) = error(convolution, &convolution_reference);
    if convolution_max_abs > 1e-4 {
        return Err(
            format!("packet convolution parity failed: max_abs={convolution_max_abs}").into(),
        );
    }
    let (projection_reference, projection_shape) =
        oracle.convolution_projection_output(&features)?;
    if projection_shape != [rows, 1024] || projection_reference.len() != projection.len() {
        return Err("packet and direct projection shapes differ".into());
    }
    let (projection_max_abs, projection_mean_abs) = error(projection, &projection_reference);
    if projection_max_abs > 1e-4 {
        return Err(
            format!("packet projection parity failed: max_abs={projection_max_abs}").into(),
        );
    }
    let (positioned_reference, positioned_shape) =
        oracle.positioned_convolution_output(&features)?;
    if positioned_shape != [rows, 1024] || positioned_reference.len() != positioned.len() {
        return Err("packet and direct positioned shapes differ".into());
    }
    let (positioned_max_abs, positioned_mean_abs) = error(positioned, &positioned_reference);
    if positioned_max_abs > 1e-4 {
        return Err(
            format!("packet positioned parity failed: max_abs={positioned_max_abs}").into(),
        );
    }
    let (norm_reference, norm_shape) = oracle.first_layer_norm_output(&features)?;
    if norm_shape != [rows, 1024] || norm_reference.len() != norm.len() {
        return Err("packet and direct layer-normalization shapes differ".into());
    }
    let (norm_max_abs, norm_mean_abs) = error(norm, &norm_reference);
    if norm_max_abs > 1e-4 {
        return Err(
            format!("packet layer-normalization parity failed: max_abs={norm_max_abs}").into(),
        );
    }
    let (qkv_reference, qkv_shape) = oracle.first_layer_qkv_output(&features)?;
    if qkv_shape != [rows, 1024]
        || qkv_reference
            .iter()
            .any(|values| values.len() != rows * 1024)
    {
        return Err("packet and direct QKV shapes differ".into());
    }
    let qkv_errors: Vec<_> = qkv
        .iter()
        .zip(&qkv_reference)
        .map(|(&actual, expected)| error(actual, expected))
        .collect();
    if qkv_errors.iter().any(|&(maximum, _)| maximum > 1e-4) {
        return Err(format!("packet QKV parity failed: {qkv_errors:?}").into());
    }
    let (attention_reference, attention_shape) = oracle.first_layer_attention_output(&features)?;
    if attention_shape != [rows, 1024] || attention_reference.len() != attention.len() {
        return Err("packet and direct attention shapes differ".into());
    }
    let (attention_max_abs, attention_mean_abs) = error(attention, &attention_reference);
    if attention_max_abs > 1e-4 {
        return Err(format!("packet attention parity failed: max_abs={attention_max_abs}").into());
    }
    let (layer_output_reference, layer_output_shape) = oracle.first_layer_output(&features)?;
    if layer_output_shape != [rows, 1024] || layer_output_reference.len() != layer_output.len() {
        return Err("packet and direct first-layer output shapes differ".into());
    }
    let (layer_output_max_abs, layer_output_mean_abs) =
        error(layer_output, &layer_output_reference);
    if layer_output_max_abs > 1e-4 {
        return Err(
            format!("packet first-layer parity failed: max_abs={layer_output_max_abs}").into(),
        );
    }
    let (transformer_reference, transformer_shape) = oracle.transformer_output(&features)?;
    if transformer_shape != [rows, 1024] || transformer_reference.len() != transformer_output.len()
    {
        return Err("packet and direct transformer output shapes differ".into());
    }
    let (transformer_max_abs, transformer_mean_abs) =
        error(transformer_output, &transformer_reference);
    if transformer_max_abs > 1e-4 {
        return Err(
            format!("packet transformer parity failed: max_abs={transformer_max_abs}").into(),
        );
    }
    let audio_reference = oracle.encode(&features)?;
    if audio_reference.len() != rows * 2048 || audio_reference.len() != audio_output.len() {
        return Err("packet and direct audio encoder output shapes differ".into());
    }
    let (audio_max_abs, audio_mean_abs) = error(audio_output, &audio_reference);
    if audio_max_abs > 1e-4 {
        return Err(format!("packet audio encoder parity failed: max_abs={audio_max_abs}").into());
    }
    println!(
        "{}",
        serde_json::json!({
            "backend": selected_backend,
            "input_shape": [chunks, 1, 128, chunk_frames],
            "convolution_shape": output_shape,
            "projection_shape": projection_shape,
            "stage_milliseconds": medians.iter().map(|value| value / 1000.0).collect::<Vec<_>>(),
            "milliseconds": medians.iter().sum::<f64>() / 1000.0,
            "sequence_milliseconds": sequence_ms,
            "convolution_max_abs": convolution_max_abs,
            "convolution_mean_abs": convolution_mean_abs,
            "projection_max_abs": projection_max_abs,
            "projection_mean_abs": projection_mean_abs,
            "positioned_max_abs": positioned_max_abs,
            "positioned_mean_abs": positioned_mean_abs,
            "norm_max_abs": norm_max_abs,
            "norm_mean_abs": norm_mean_abs,
            "qkv_errors": qkv_errors,
            "attention_max_abs": attention_max_abs,
            "attention_mean_abs": attention_mean_abs,
            "layer_output_max_abs": layer_output_max_abs,
            "layer_output_mean_abs": layer_output_mean_abs,
            "transformer_max_abs": transformer_max_abs,
            "transformer_mean_abs": transformer_mean_abs,
            "audio_max_abs": audio_max_abs,
            "audio_mean_abs": audio_mean_abs,
            "audio_shape": [rows, 2048],
            "sum": audio_output.iter().map(|&value| f64::from(value)).sum::<f64>(),
            "l2": audio_output.iter().map(|&value| f64::from(value).powi(2)).sum::<f64>().sqrt(),
            "first": &audio_output[..audio_output.len().min(8)],
        })
    );
    Ok(())
}

fn error(actual: &[f32], expected: &[f32]) -> (f32, f64) {
    let (max_abs, sum) = actual.iter().zip(expected).fold(
        (0.0f32, 0.0f64),
        |(maximum, sum), (&actual, &expected)| {
            let error = (actual - expected).abs();
            (maximum.max(error), sum + f64::from(error))
        },
    );
    (max_abs, sum / actual.len() as f64)
}

fn read_f32(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err("F32 input length is not divisible by four".into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect())
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    f32::from_bits((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000)
}
