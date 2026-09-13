use std::path::Path;

use plowrt::asr::frontend::decode_wav;
use plowrt::asr::nemotron::NemotronFrontend;
use plowrt::asset::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: asr_nemotron_frontend MODEL AUDIO [OUTPUT_F32]".into());
    }
    let model = GgufFile::open(Path::new(&args[1]))?;
    let frontend = NemotronFrontend::load(&model)?;
    let samples = decode_wav(&std::fs::read(&args[2])?)?;
    let features = frontend.extract(&samples)?;
    if let Some(path) = args.get(3) {
        let bytes: Vec<_> = features
            .values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path, bytes)?;
    }
    println!(
        "{}",
        serde_json::json!({
            "samples": samples.len(),
            "frames": features.frames,
            "bins": features.bins,
            "sum": features.values.iter().map(|&value| f64::from(value)).sum::<f64>(),
            "l2": features.values.iter().map(|&value| f64::from(value).powi(2)).sum::<f64>().sqrt(),
            "first": &features.values[..features.values.len().min(8)],
            "last": &features.values[features.values.len().saturating_sub(8)..],
        })
    );
    Ok(())
}
