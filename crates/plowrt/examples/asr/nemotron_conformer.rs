use std::path::Path;

use plowrt::asr::conformer::CpuConformerBlock;
use plowrt::asr::nemotron::conformer_block_plan;
use plowrt::asset::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=5).contains(&args.len()) {
        return Err("usage: asr_nemotron_conformer MODEL LAYER [FRAMES [OUTPUT_F32]]".into());
    }
    let model = GgufFile::open(Path::new(&args[1]))?;
    let layer = args[2].parse::<usize>()?;
    let plan = conformer_block_plan(&model, layer)?;
    let Some(frames) = args
        .get(3)
        .map(|value| value.parse::<usize>())
        .transpose()?
    else {
        println!(
            "{}",
            serde_json::json!({"layer": layer, "width": plan.width()})
        );
        return Ok(());
    };
    let input: Vec<_> = (0..frames * plan.width())
        .map(|index| ((index.wrapping_mul(17) % 251) as f32 - 125.0) / 127.0)
        .collect();
    let started = std::time::Instant::now();
    let output = CpuConformerBlock::new(plan).run(&input, frames)?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Conformer block produced a non-finite output".into());
    }
    if let Some(path) = args.get(4) {
        let bytes: Vec<_> = output
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path, bytes)?;
    }
    println!(
        "{}",
        serde_json::json!({
            "layer": layer,
            "frames": frames,
            "width": output.len() / frames,
            "seconds": started.elapsed().as_secs_f64(),
            "sum": output.iter().map(|&value| f64::from(value)).sum::<f64>(),
            "l2": output.iter().map(|&value| f64::from(value).powi(2)).sum::<f64>().sqrt(),
            "first": &output[..output.len().min(8)],
        })
    );
    Ok(())
}
