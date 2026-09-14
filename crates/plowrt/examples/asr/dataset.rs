#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asr::{frontend::decode_wav, qwen::QwenAsr};
    use serde_json::{json, Value};
    use std::{io::Write, path::Path, sync::atomic::AtomicBool, time::Instant};
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len())
        || args.get(4).is_some_and(|s| {
            s != "--compare-conv"
                && s != "--compare-packed-conv"
                && s != "--compare-handoff"
                && s != "--compare-wide"
                && s != "--single"
                && s != "--compare-large"
                && s != "--compare-bf16-weights"
                && s != "--compare-attention"
                && s != "--compare-direct"
                && s != "--compare-tile64"
        })
    {
        return Err("usage: asr_dataset CHECKPOINT BLOB MANIFEST.jsonl [--compare-conv|--compare-packed-conv|--compare-handoff|--compare-wide|--compare-large|--compare-bf16-weights|--compare-attention|--compare-direct|--compare-tile64|--single]".into());
    }
    let single = args.get(4).is_some_and(|s| s == "--single");
    let compare_bf16 = args.get(4).is_some_and(|s| s == "--compare-bf16-weights");
    let compare_attention = args.get(4).is_some_and(|s| s == "--compare-attention");
    let compare_direct = args.get(4).is_some_and(|s| s == "--compare-direct");
    let compare_tile64 = args.get(4).is_some_and(|s| s == "--compare-tile64");
    let compare_large = args.get(4).is_some_and(|s| s == "--compare-large");
    let compare_wide = args.get(4).is_some_and(|s| s == "--compare-wide");
    let compare_conv = args.len() == 5;
    let compare_packed = args.get(4).is_some_and(|s| s == "--compare-packed-conv");
    let compare_handoff = args.get(4).is_some_and(|s| s == "--compare-handoff");
    let rows: Vec<Value> = std::fs::read_to_string(&args[3])?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let mut engine = QwenAsr::load(Path::new(&args[2]), Path::new(&args[1]))?;
    if !single {
        engine.set_tile64_selective(false)?;
        engine.set_direct_epilogue(false)?;
        engine.set_wide_linear(false)?;
        engine.set_large_linear(false)?;
    }
    let cancel = AtomicBool::new(false);
    let mut warmed = false;
    let mut failed = false;
    for (index, row) in rows.iter().enumerate() {
        if row.get("excluded_reason").is_some() {
            println!(
                "{}",
                json!({"kind":"excluded","id":row["id"],"reason":row["excluded_reason"]})
            );
            continue;
        }
        let samples = decode_wav(&std::fs::read(
            row["audio"].as_str().ok_or("missing audio")?,
        )?)?;
        if !warmed {
            for tiled in [false, true].into_iter().take(if single { 1 } else { 2 }) {
                if !single {
                    engine.set_tiled_linear(compare_conv || tiled);
                    engine.set_tiled_conv(compare_conv && tiled);
                    if compare_packed {
                        engine.set_packed_conv(tiled);
                    }
                    if compare_wide {
                        engine.set_packed_conv(true);
                        engine.set_wide_linear(tiled)?;
                    }
                    if compare_large {
                        engine.set_packed_conv(true);
                        engine.set_wide_linear(true)?;
                        engine.set_large_linear(tiled)?;
                    }
                    if compare_handoff {
                        engine.set_packed_conv(true);
                        engine.set_device_handoff(tiled);
                    }
                    if compare_bf16 {
                        engine.set_packed_conv(true);
                        engine.set_large_linear(true)?;
                        engine.set_bf16_linear_weights(tiled)?;
                    }
                    if compare_attention {
                        engine.set_packed_conv(true);
                        engine.set_large_linear(true)?;
                        engine.set_simd_attention(tiled)?;
                    }
                    if compare_direct {
                        engine.set_packed_conv(true);
                        engine.set_large_linear(true)?;
                        engine.set_simd_attention(true)?;
                        engine.set_direct_epilogue(tiled)?;
                    }
                    if compare_tile64 {
                        engine.set_packed_conv(true);
                        engine.set_large_linear(true)?;
                        engine.set_simd_attention(true)?;
                        engine.set_direct_epilogue(true)?;
                        engine.set_tile64_selective(tiled)?;
                    }
                }
                engine.transcribe(&samples, None, "", &cancel)?;
            }
            warmed = true;
        }
        for tiled in [index % 2 == 0, index % 2 != 0]
            .into_iter()
            .take(if single { 1 } else { 2 })
        {
            if !single {
                engine.set_tiled_linear(compare_conv || tiled);
                engine.set_tiled_conv(compare_conv && tiled);
                if compare_packed {
                    engine.set_packed_conv(tiled);
                }
                if compare_wide {
                    engine.set_packed_conv(true);
                    engine.set_wide_linear(tiled)?;
                }
                if compare_large {
                    engine.set_packed_conv(true);
                    engine.set_wide_linear(true)?;
                    engine.set_large_linear(tiled)?;
                }
                if compare_handoff {
                    engine.set_packed_conv(true);
                    engine.set_device_handoff(tiled);
                }
                if compare_bf16 {
                    engine.set_packed_conv(true);
                    engine.set_large_linear(true)?;
                    engine.set_bf16_linear_weights(tiled)?;
                }
                if compare_attention {
                    engine.set_packed_conv(true);
                    engine.set_large_linear(true)?;
                    engine.set_simd_attention(tiled)?;
                }
                if compare_direct {
                    engine.set_packed_conv(true);
                    engine.set_large_linear(true)?;
                    engine.set_simd_attention(true)?;
                    engine.set_direct_epilogue(tiled)?;
                }
                if compare_tile64 {
                    engine.set_packed_conv(true);
                    engine.set_large_linear(true)?;
                    engine.set_simd_attention(true)?;
                    engine.set_direct_epilogue(true)?;
                    engine.set_tile64_selective(tiled)?;
                }
            }
            let start = Instant::now();
            let result = engine.transcribe(&samples, None, "", &cancel);
            let seconds = start.elapsed().as_secs_f64();
            let mut output = json!({"kind":"result","id":row["id"],
                "backend":if single { "native" } else if compare_tile64 {
                    if tiled { "native-selective64" } else { "native-direct32" }
                } else if compare_direct {
                    if tiled { "native-direct" } else { "native-shared" }
                } else if compare_attention {
                    if tiled { "native-simd-attention" } else { "native-scalar-attention" }
                } else if compare_bf16 {
                    if tiled { "native-bf16-weights" } else { "native-f32-weights" }
                } else if compare_large {
                    if tiled { "native-large" } else { "native-wide" }
                } else if compare_wide {
                    if tiled { "native-wide" } else { "native-tiled" }
                } else if compare_handoff {
                    if tiled { "native-device-handoff" } else { "native-host-handoff" }
                } else if compare_packed {
                    if tiled { "native-packed-conv" } else { "native-tiled-conv" }
                } else { match (compare_conv, tiled) {
                    (true, true) => "native-tiled-conv",
                    (true, false) | (false, true) => "native-tiled",
                    (false, false) => "native-scalar",
                } },
                "duration_seconds":samples.len() as f64/16000.0,"seconds":seconds,
                "reference":row["reference"]});
            match result {
                Ok(result) => {
                    output["text"] = result.text.into();
                    output["language"] = json!(result.language);
                }
                Err(error) => {
                    failed = true;
                    output["text"] = "".into();
                    output["error"] = error.to_string().into();
                }
            }
            println!("{output}");
            std::io::stdout().flush()?;
        }
    }
    if failed {
        return Err("one or more transcriptions failed; retained in results".into());
    }
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("asr_dataset requires macOS and --features metal");
    std::process::exit(1);
}
