#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::frontend::{decode_wav, QwenFrontend};
    let args: Vec<_> = std::env::args().collect();
    if !(5..=6).contains(&args.len()) {
        return Err("usage: asr_check CHECKPOINT AUDIO.wav REFERENCE_DIR MODE [BLOB|REPEATS]; modes: encoder, encoder-reference-input, encoder-profile, encoder-profile-tiled, encoder-tiled, encoder-compare, encoder-conv-compare, encoder-packed-compare, encoder-packed, encoder-profile-wide, encoder-wide-compare, encoder-large-compare, encoder-bf16-compare, encoder-attention-compare, encoder-direct-compare, encoder-tile64-compare, encoder-tile64-selective, encoder-profile-current, encoder-profile-tile64, decoder, model, model-tiled".into());
    }
    if !matches!(
        args[4].as_str(),
        "encoder"
            | "encoder-reference-input"
            | "encoder-profile"
            | "encoder-profile-tiled"
            | "encoder-tiled"
            | "encoder-compare"
            | "encoder-conv-compare"
            | "encoder-packed-compare"
            | "encoder-packed"
            | "encoder-profile-wide"
            | "encoder-wide-compare"
            | "encoder-large-compare"
            | "encoder-bf16-compare"
            | "encoder-attention-compare"
            | "encoder-direct-compare"
            | "encoder-tile64-compare"
            | "encoder-tile64-selective"
            | "encoder-profile-current"
            | "encoder-profile-tile64"
            | "decoder"
            | "model"
            | "model-tiled"
            | "model-device"
    ) {
        return Err(format!("unknown check mode: {}", args[4]).into());
    }
    let checkpoint = std::path::Path::new(&args[1]);
    let samples = decode_wav(&std::fs::read(&args[2])?)?;
    let mut features = QwenFrontend::default().extract(&samples)?;
    let reference = std::path::Path::new(&args[3]);
    let compare = |name: &str,
                   values: &[f32],
                   limit: f64|
     -> Result<(), Box<dyn std::error::Error>> {
        let bytes = std::fs::read(reference.join(format!("{name}.f32")))?;
        let expected: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        if values.len() != expected.len() {
            return Err(format!("{name}: {} vs {} elements", values.len(), expected.len()).into());
        }
        let mut max = 0f64;
        let mut delta = 0f64;
        let mut norm = 0f64;
        for (&a, &b) in values.iter().zip(&expected) {
            if !a.is_finite() || !b.is_finite() {
                return Err("nonfinite output".into());
            }
            let d = a as f64 - b as f64;
            max = max.max(d.abs());
            delta += d * d;
            norm += (b as f64).powi(2);
        }
        let relative = (delta / norm.max(1e-20)).sqrt();
        println!(
            "{name}: max_abs={max:.8} rel_l2={relative:.8} elements={}",
            values.len()
        );
        if relative > limit {
            return Err(format!("{name}: rel_l2 exceeds {limit}").into());
        }
        Ok(())
    };
    compare("mel", &features.values, 1e-5)?;
    let tokenizer = plowrt::text::tokenizer::load_tokenizer(checkpoint);
    let input: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference.join("input.json"))?)?;
    let rows = plowrt::asr::qwen_audio_rows(features.frames);
    let prompt = input["prompt"]
        .as_str()
        .unwrap()
        .replace("<|audio_pad|>", &"<|audio_pad|>".repeat(rows));
    let expected: Vec<u32> = serde_json::from_value(input["prompt_ids"][0].clone())?;
    assert_eq!(tokenizer.encode(&prompt), expected, "prompt token parity");
    println!("prompt: {} tokens match", expected.len());
    if args.get(4).is_some_and(|s| {
        s == "encoder"
            || s == "encoder-reference-input"
            || s == "encoder-profile"
            || s == "encoder-profile-tiled"
            || s == "encoder-tiled"
            || s == "encoder-compare"
            || s == "encoder-conv-compare"
            || s == "encoder-packed-compare"
            || s == "encoder-packed"
            || s == "encoder-profile-wide"
            || s == "encoder-wide-compare"
            || s == "encoder-large-compare"
            || s == "encoder-bf16-compare"
            || s == "encoder-attention-compare"
            || s == "encoder-direct-compare"
            || s == "encoder-tile64-compare"
            || s == "encoder-tile64-selective"
            || s == "encoder-profile-current"
            || s == "encoder-profile-tile64"
    }) {
        if args[4] != "encoder" {
            features.values = std::fs::read(reference.join("mel.f32"))?
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            println!("encoder input: reference mel (isolates frontend rounding)");
        }
        let mut encoder = plowrt::exec::apple::asr::QwenAudioEncoder::load(checkpoint)?;
        encoder.set_kernel_profiling(
            args[4] == "encoder-profile"
                || args[4] == "encoder-profile-tiled"
                || args[4] == "encoder-packed"
                || args[4] == "encoder-profile-wide"
                || args[4] == "encoder-profile-current"
                || args[4] == "encoder-profile-tile64",
        );
        encoder.set_tiled_conv(args[4] == "encoder-profile-tiled");
        encoder.set_tiled_linear(
            args[4] == "encoder-tiled"
                || args[4] == "encoder-conv-compare"
                || args[4] == "encoder-packed-compare"
                || args[4] == "encoder-profile-tiled"
                || args[4] == "encoder-packed"
                || args[4] == "encoder-wide-compare"
                || args[4] == "encoder-large-compare"
                || args[4] == "encoder-bf16-compare"
                || args[4] == "encoder-attention-compare"
                || args[4] == "encoder-direct-compare"
                || args[4] == "encoder-tile64-compare"
                || args[4] == "encoder-tile64-selective"
                || args[4] == "encoder-profile-wide",
        );
        if args[4] == "encoder-packed"
            || args[4] == "encoder-wide-compare"
            || args[4] == "encoder-large-compare"
            || args[4] == "encoder-bf16-compare"
            || args[4] == "encoder-attention-compare"
            || args[4] == "encoder-direct-compare"
            || args[4] == "encoder-tile64-compare"
            || args[4] == "encoder-tile64-selective"
            || args[4] == "encoder-profile-wide"
        {
            encoder.set_packed_conv(true);
        }
        encoder.set_wide_linear(
            args[4] == "encoder-profile-wide" || args[4] == "encoder-large-compare",
        )?;
        if args[4] == "encoder-bf16-compare" {
            encoder.set_large_linear(true)?;
            encoder.set_bf16_linear_weights(true)?;
        }
        if args[4] == "encoder-attention-compare" {
            encoder.set_large_linear(true)?;
        }
        if args[4] == "encoder-direct-compare" {
            encoder.set_large_linear(true)?;
            encoder.set_simd_attention(true)?;
        }
        if args[4] == "encoder-tile64-compare" || args[4] == "encoder-tile64-selective" {
            encoder.set_large_linear(true)?;
            encoder.set_simd_attention(true)?;
            encoder.set_direct_epilogue(true)?;
        }
        if args[4] == "encoder-profile-current" || args[4] == "encoder-profile-tile64" {
            encoder.set_tiled_linear(true);
            encoder.set_packed_conv(true);
            encoder.set_wide_linear(encoder.supports_wide_linear())?;
            encoder.set_large_linear(encoder.supports_large_linear())?;
            encoder.set_simd_attention(encoder.supports_simd_attention())?;
            if encoder.validate_direct_epilogue()? {
                encoder.set_direct_epilogue(true)?;
            }
        }
        if args[4] == "encoder-profile-tile64" {
            encoder.set_tile64(true)?;
        }
        let repeats = args
            .get(5)
            .map(|s| s.parse::<usize>())
            .transpose()?
            .unwrap_or(1);
        if !(1..=100).contains(&repeats) {
            return Err("encoder repeats must be 1..=100".into());
        }
        for iteration in 0..repeats {
            if args[4] == "encoder-compare"
                || args[4] == "encoder-conv-compare"
                || args[4] == "encoder-packed-compare"
                || args[4] == "encoder-wide-compare"
                || args[4] == "encoder-large-compare"
                || args[4] == "encoder-bf16-compare"
                || args[4] == "encoder-attention-compare"
                || args[4] == "encoder-direct-compare"
                || args[4] == "encoder-tile64-compare"
                || args[4] == "encoder-tile64-selective"
            {
                let mut outputs = Vec::new();
                for tiled in [iteration % 2 == 0, iteration % 2 != 0] {
                    if args[4] == "encoder-tile64-compare" || args[4] == "encoder-tile64-selective"
                    {
                        if args[4] == "encoder-tile64-selective" {
                            encoder.set_tile64_selective(tiled)?;
                        } else {
                            encoder.set_tile64(tiled)?;
                        }
                    } else if args[4] == "encoder-direct-compare" {
                        encoder.set_direct_epilogue(tiled)?;
                    } else if args[4] == "encoder-attention-compare" {
                        encoder.set_simd_attention(tiled)?;
                    } else if args[4] == "encoder-bf16-compare" {
                        encoder.set_bf16_linear_weights(tiled)?;
                    } else if args[4] == "encoder-large-compare" {
                        encoder.set_large_linear(tiled)?;
                    } else if args[4] == "encoder-wide-compare" {
                        encoder.set_wide_linear(tiled)?;
                    } else if args[4] == "encoder-packed-compare" {
                        encoder.set_packed_conv(tiled);
                    } else if args[4] == "encoder-conv-compare" {
                        encoder.set_tiled_conv(tiled);
                    } else {
                        encoder.set_tiled_linear(tiled);
                    }
                    let start = std::time::Instant::now();
                    outputs.push(encoder.encode(&features)?);
                    println!(
                        "encoder {iteration}: tiled={tiled} {:.3}s",
                        start.elapsed().as_secs_f64()
                    );
                }
                assert_eq!(outputs[0], outputs[1], "scalar/tiled projected embeddings");
                compare("audio", &outputs[0], 0.02)?;
                continue;
            }
            let start = std::time::Instant::now();
            let audio = encoder.encode(&features)?;
            println!("encoder {iteration}: {:.3}s", start.elapsed().as_secs_f64());
            compare("audio", &audio, 0.02)?;
        }
    }
    if args.get(4).is_some_and(|s| s == "decoder") {
        let blob = args.get(5).ok_or("decoder check requires BLOB")?;
        let mut decoder =
            plowrt::exec::apple::MetalEngine::load(std::path::Path::new(blob), checkpoint)?;
        let embeddings: Vec<u16> = std::fs::read(reference.join("spliced.f32"))?
            .chunks_exact(4)
            .map(|b| {
                let u = f32::from_le_bytes(b.try_into().unwrap()).to_bits();
                (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
            })
            .collect();
        let mut token = decoder.prefill_embeddings(&expected, &embeddings)?;
        let logits = decoder.tensor_bytes(decoder.model.wk.logits.ok_or("missing logits")?);
        let logits: Vec<_> = logits
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
            .collect();
        let logit_parity = compare("logits", &logits, 0.02);
        let tokens: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference.join("tokens.json"))?)?;
        let tokens: Vec<u32> = serde_json::from_value(tokens["ids"].clone())?;
        let mut actual = vec![token];
        for step in 0..tokens.len().saturating_sub(1) {
            token = decoder.decode_step(
                (expected.len() + step) as u32,
                (expected.len() + step + 1) as u32,
            )?;
            actual.push(token);
        }
        println!("decoder: {}", tokenizer.decode(&actual));
        assert_eq!(
            actual, tokens,
            "greedy decoder token parity with reference audio embeddings"
        );
        logit_parity?;
    }
    if args
        .get(4)
        .is_some_and(|s| s == "model" || s == "model-tiled" || s == "model-device")
    {
        let blob = args.get(5).ok_or("model check requires BLOB")?;
        let mut engine = plowrt::asr::qwen::QwenAsr::load(std::path::Path::new(blob), checkpoint)?;
        if args[4] == "model-tiled" {
            engine.set_tiled_linear(true);
        }
        engine.set_device_handoff(args[4] == "model-device");
        let start = std::time::Instant::now();
        let result = engine.transcribe(
            &samples,
            None,
            "",
            &std::sync::atomic::AtomicBool::new(false),
        )?;
        println!(
            "model: {:.3}s {}",
            start.elapsed().as_secs_f64(),
            serde_json::to_string(&result)?
        );
        let expected: serde_json::Value =
            serde_json::from_slice(&std::fs::read(reference.join("transcript.json"))?)?;
        assert_eq!(
            result.text,
            expected[0]["text"].as_str().unwrap(),
            "end-to-end transcript parity"
        );
        assert_eq!(
            result.language.as_deref().unwrap_or(""),
            expected[0]["language"].as_str().unwrap()
        );
    }
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("requires macOS and --features metal");
    std::process::exit(1);
}
