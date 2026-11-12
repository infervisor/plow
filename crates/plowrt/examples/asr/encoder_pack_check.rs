//! Packed encoder launches against single ones: each recording's encoder rows from a packed launch
//! of PACK recordings must equal, bit for bit, its rows from its own launch. One JSON line per
//! differing recording, then a summary; exits 1 on a difference.
//!
//!   asr_encoder_pack_check ASSETS PACK AUDIO.wav...

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("asr_encoder_pack_check requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::audio_lm::{encoder_packing_differences, AudioLmPrompt};
    use plowrt::asr::frontend::decode_wav;
    use serde_json::json;
    use std::path::Path;

    let args: Vec<_> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: asr_encoder_pack_check ASSETS PACK AUDIO.wav...".into());
    }
    let assets = Path::new(&args[1]);
    let pack: usize = args[2].parse()?;
    let model = assets.join("model.pkt");
    let prompt = AudioLmPrompt::load(&model, &assets.join("checkpoint"))?;
    let encoder =
        plowrt::exec::packet_runtime::stage_packet(&model, "encoder.packet", "encoder.pkt")?;
    let items = args[3..]
        .iter()
        .map(|path| Ok(prompt.features(&decode_wav(&std::fs::read(path)?)?)?))
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    let differences = encoder_packing_differences(&encoder, &items, pack)?;
    let mut differing = 0;
    for ((path, features), (elements, worst)) in args[3..].iter().zip(&items).zip(&differences) {
        if *elements != 0 {
            differing += 1;
            println!(
                "{}",
                json!({"audio": path, "frames": features.frames, "elements": elements, "max_abs": worst})
            );
        }
    }
    println!(
        "{}",
        json!({"pack": pack, "recordings": items.len(), "differing": differing})
    );
    std::process::exit(i32::from(differing != 0));
}
