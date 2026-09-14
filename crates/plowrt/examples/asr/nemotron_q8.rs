use std::path::Path;

use gguf_rs_lib::format::types::GGUFTensorType;
use plowrt::asset::gguf::GgufFile;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: asr_nemotron_q8 MODEL TENSOR".into());
    }
    let model = GgufFile::open(Path::new(&args[1]))?;
    if args[2] == "--metadata" {
        println!("{}", serde_json::to_string(model.metadata())?);
        return Ok(());
    }
    if matches!(args[2].as_str(), "--list" | "--list-all") {
        let q8_only = args[2] == "--list";
        let tensors: Vec<_> = model
            .tensors()
            .filter(|tensor| !q8_only || tensor.dtype == GGUFTensorType::Q8_0)
            .map(|tensor| {
                serde_json::json!({
                    "name": tensor.name,
                    "dimensions": tensor.dimensions,
                    "dtype": format!("{:?}", tensor.dtype),
                })
            })
            .collect();
        println!("{}", serde_json::to_string(&tensors)?);
        return Ok(());
    }
    let tensor = model.tensor(&args[2])?;
    if tensor.dimensions.len() != 2 {
        return Err("TENSOR must be a matrix".into());
    }
    let k = usize::try_from(tensor.dimensions[0])?;
    let n = usize::try_from(tensor.dimensions[1])?;
    let input: Vec<_> = (0..k)
        .map(|index| ((index.wrapping_mul(17) % 251) as f32 - 125.0) / 127.0)
        .collect();
    let mut output = vec![0.0; n];
    tensor.q8_0_matvec(&input, &mut output)?;
    println!(
        "{}",
        serde_json::json!({
            "tensor": tensor.name,
            "dimensions": tensor.dimensions,
            "dtype": format!("{:?}", tensor.dtype),
            "sum": output.iter().map(|&value| f64::from(value)).sum::<f64>(),
            "l2": output.iter().map(|&value| f64::from(value).powi(2)).sum::<f64>().sqrt(),
            "first": &output[..output.len().min(8)],
        })
    );
    Ok(())
}
