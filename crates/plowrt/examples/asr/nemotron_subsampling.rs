use std::path::Path;
use std::time::Instant;

use plowrt::asr::frontend::decode_wav;
use plowrt::asr::nemotron::{NemotronFrontend, NemotronSubsampler};
use plowrt::asset::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: asr_nemotron_subsampling MODEL AUDIO [OUTPUT_F32]".into());
    }
    let model = GgufFile::open(Path::new(&args[1]))?;
    let frontend = NemotronFrontend::load(&model)?;
    let subsampler = NemotronSubsampler::load(&model)?;
    let samples = decode_wav(&std::fs::read(&args[2])?)?;
    let features = frontend.extract(&samples)?;
    let start = Instant::now();
    let output = subsampler.run(&features)?;
    let elapsed = start.elapsed();
    if let Some(path) = args.get(3) {
        let bytes: Vec<_> = output
            .values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path, bytes)?;
    }
    println!(
        "{}",
        serde_json::json!({
            "input_shape": [features.frames, features.bins],
            "output_shape": [output.frames, output.width, output.channels],
            "milliseconds": elapsed.as_secs_f64() * 1e3,
            "sum": output.values.iter().map(|&value| f64::from(value)).sum::<f64>(),
            "l2": output.values.iter().map(|&value| f64::from(value).powi(2)).sum::<f64>().sqrt(),
            "first": &output.values[..output.values.len().min(8)],
        })
    );
    Ok(())
}
