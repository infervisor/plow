use std::path::Path;

use plowrt::asr::conformer::CpuConformerEncoder;
use plowrt::asr::frontend::decode_wav;
use plowrt::asr::nemotron::{
    conformer_encoder_plan, pre_encode_projection, NemotronFrontend, NemotronSubsampler,
};
use plowrt::asset::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: asr_nemotron_encoder MODEL AUDIO [OUTPUT_F32]".into());
    }
    let model = GgufFile::open(Path::new(&args[1]))?;
    let frontend = NemotronFrontend::load(&model)?;
    let subsampler = NemotronSubsampler::load(&model)?;
    let projection = pre_encode_projection(&model)?;
    let encoder_plan = conformer_encoder_plan(&model)?;
    let encoder_width = encoder_plan.width();
    let encoder = CpuConformerEncoder::new(encoder_plan);
    let samples = decode_wav(&std::fs::read(&args[2])?)?;
    let features = frontend.extract(&samples)?;
    let subsampled = subsampler.run(&features)?;
    let frames = subsampled.frames;
    let mut projected = vec![0.0; frames * projection.weight().n()];
    projection.matmul(&subsampled.values, frames, &mut projected)?;
    let started = std::time::Instant::now();
    let output = encoder.run(&projected, frames)?;
    let elapsed = started.elapsed();
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Conformer encoder produced a non-finite output".into());
    }
    if let Some(path) = args.get(3) {
        let bytes: Vec<_> = output
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path, bytes)?;
    }
    println!(
        "{}",
        serde_json::json!({
            "input_shape": [features.frames, features.bins],
            "encoder_shape": [frames, encoder_width],
            "encoder_seconds": elapsed.as_secs_f64(),
            "sum": output.iter().map(|&value| f64::from(value)).sum::<f64>(),
            "l2": output.iter().map(|&value| f64::from(value).powi(2)).sum::<f64>().sqrt(),
            "first": &output[..output.len().min(8)],
        })
    );
    Ok(())
}
