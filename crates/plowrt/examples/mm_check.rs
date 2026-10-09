//! Multimodal parity against `scripts/mm/hf_ref.py`: host preprocessing (pixel patches, log-mel)
//! on CPU, and with `--encode` the encoder sidecars on the GPU (projected soft tokens), each fed
//! both its own preprocessing and HF's, so encoder and preprocessing gaps are told apart.
//!
//!   cargo run --release -p plowrt --features cuda --example mm_check -- <ref dir> <model.pkt> [--encode]
//!
//! Prints one JSON line per item; exits non-zero when a gate fails.

use plowrt::serve::mm::media;

fn read_f32(dir: &std::path::Path, entry: &serde_json::Value) -> Vec<f32> {
    let bytes = std::fs::read(dir.join(entry["file"].as_str().expect("file"))).expect("ref array");
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn stats(a: &[f32], b: &[f32]) -> (f64, f64, f64) {
    let (mut dot, mut na, mut nb, mut diff, mut maxd) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (f64::from(x), f64::from(y));
        dot += x * y;
        na += x * x;
        nb += y * y;
        diff += (x - y) * (x - y);
        maxd = maxd.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-30), (diff / nb.max(1e-30)).sqrt(), maxd)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(&args[1]);
    let pkt = std::path::PathBuf::from(&args[2]);
    let encode = args.iter().any(|a| a == "--encode");
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("ref.json")).unwrap()).unwrap();
    let contract = plowrt::asset::serve::read_multimodal(&pkt).expect("contract").expect("packet has no multimodal contract");
    let mut ok = true;
    let mut encoders: std::collections::HashMap<String, plowrt::serve::mm::encoder::Encoder> = Default::default();
    for (i, item) in reference["items"].as_array().unwrap().iter().enumerate() {
        let kind = item["kind"].as_str().unwrap();
        let path = item["path"].as_str().unwrap();
        let m = contract.modality(kind).expect("modality");
        let bytes = std::fs::read(path).unwrap();
        let mut line = serde_json::json!({ "item": i, "kind": kind, "path": path });
        let (ours, theirs_input): (Box<dyn std::any::Any>, Box<dyn std::any::Any>);
        if kind == "image" {
            let p = media::PatchParams {
                patch: m.param("patch_size").unwrap() as u32,
                pool: m.param("pool").unwrap() as u32,
                max_soft_tokens: m.param("max_soft_tokens").unwrap() as u32,
                rescale: m.param_f32("rescale_f32").unwrap(),
                normalize: m.param("normalize") == Some(1),
                mean: [0.0; 3],
                std: [1.0; 3],
            };
            let img = media::decode_image(&bytes, u64::MAX).unwrap();
            let pt = media::image_patches(&img, &p).unwrap();
            let hf = read_f32(&dir, &item["patches"]);
            let positions: Vec<[u32; 2]> = item["positions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| [v[0].as_u64().unwrap() as u32, v[1].as_u64().unwrap() as u32])
                .collect();
            let same_grid = pt.values.len() == hf.len() && pt.positions == positions;
            let (_, rel, maxd) = if same_grid { stats(&pt.values, &hf) } else { (0.0, f64::NAN, f64::NAN) };
            let mean_abs = if same_grid {
                pt.values.iter().zip(&hf).map(|(a, b)| f64::from((a - b).abs())).sum::<f64>() / hf.len() as f64
            } else {
                f64::NAN
            };
            // PIL vs torchvision's uint8 antialiased bicubic: 2 levels; JPEG IDCT (image vs libjpeg) adds 1.
            let pass = same_grid && maxd * 255.0 <= 3.01 && mean_abs * 255.0 < 0.05 && pt.soft_tokens as u64 == item["soft_tokens"].as_u64().unwrap();
            ok &= pass;
            line["preprocess"] = serde_json::json!({
                "pass": pass, "soft_tokens": pt.soft_tokens, "hf_soft_tokens": item["soft_tokens"],
                "max_abs_levels": maxd * 255.0, "mean_abs_levels": mean_abs * 255.0, "rel_l2": rel,
            });
            let hf_pt = media::Patches { values: hf, positions, grid: pt.grid, soft_tokens: item["soft_tokens"].as_u64().unwrap() as u32 };
            ours = Box::new(pt);
            theirs_input = Box::new(hf_pt);
        } else if m.processor == "waveform_frames" {
            let frame = m.param("frame_samples").unwrap() as usize;
            let (samples, rate) = media::decode_wav(&bytes).unwrap();
            let samples = media::resample(&samples, rate, m.param("sample_rate").unwrap() as u32);
            let frames = media::waveform_frames(&samples, frame);
            let hf = read_f32(&dir, &item["mel"]);
            let valid = item["valid_frames"].as_u64().unwrap() as usize;
            let (_, rel, maxd) = stats(&frames.values, &hf);
            let pass = frames.valid_frames == valid && frames.values.len() == hf.len() && maxd < 1e-6;
            ok &= pass;
            line["preprocess"] = serde_json::json!({ "pass": pass, "tokens": frames.valid_frames, "hf_tokens": valid, "max_abs": maxd, "rel_l2": rel });
            let hf_frames = media::Mel { values: hf, frames: valid, valid_frames: valid };
            ours = Box::new(frames);
            theirs_input = Box::new(hf_frames);
        } else {
            let p = media::MelParams {
                sample_rate: m.param("sample_rate").unwrap() as u32,
                frame_length: m.param("frame_length").unwrap() as usize,
                hop_length: m.param("hop_length").unwrap() as usize,
                fft_length: m.param("fft_length").unwrap() as usize,
                mel_bins: m.param("mel_bins").unwrap() as usize,
                min_frequency: f64::from(m.param_f32("min_frequency_f32").unwrap()),
                max_frequency: f64::from(m.param_f32("max_frequency_f32").unwrap()),
                mel_floor: f64::from(m.param_f32("mel_floor_f32").unwrap()),
                pad_multiple: m.param("pad_multiple").unwrap_or(1) as usize,
            };
            let (samples, rate) = media::decode_wav(&bytes).unwrap();
            let samples = media::resample(&samples, rate, p.sample_rate);
            let mel = media::log_mel(&samples, &p);
            let hf = read_f32(&dir, &item["mel"]);
            let valid = item["valid_frames"].as_u64().unwrap() as usize;
            let n = valid * p.mel_bins;
            let (_, rel, maxd) = stats(&mel.values[..n.min(mel.values.len())], &hf);
            let pass = mel.valid_frames == valid && maxd < 1e-3;
            ok &= pass;
            line["preprocess"] = serde_json::json!({ "pass": pass, "valid_frames": mel.valid_frames, "hf_valid_frames": valid, "max_abs": maxd, "rel_l2": rel });
            let hf_mel = media::Mel { values: hf, frames: valid, valid_frames: valid };
            ours = Box::new(mel);
            theirs_input = Box::new(hf_mel);
        }
        if encode {
            let enc = encoders.entry(kind.to_string()).or_insert_with(|| {
                plowrt::serve::mm::encoder::Encoder::load(&pkt.with_file_name(&m.packet), 0).expect("encoder")
            });
            let hf_rows = read_f32(&dir, &item["encoded"]);
            let mut run = |label: &str, input: &dyn std::any::Any| {
                let rows = if kind == "image" {
                    let p = input.downcast_ref::<media::Patches>().unwrap();
                    enc.encode_images(&[p]).unwrap().remove(0)
                } else {
                    let mel = input.downcast_ref::<media::Mel>().unwrap();
                    let tokens = if m.processor == "waveform_frames" { mel.valid_frames } else { media::audio_tokens(mel.valid_frames, m.param("subsample").unwrap_or(4) as usize) };
                    enc.encode_audio(mel, tokens).unwrap()
                };
                let same = rows.len() == hf_rows.len();
                let (cos, rel, maxd) = if same { stats(&rows, &hf_rows) } else { (0.0, f64::NAN, f64::NAN) };
                // bf16 towers: rows agree to bf16 noise accumulated over the layers.
                let pass = same && cos > 0.995 && rel < 0.1;
                line[label] = serde_json::json!({ "pass": pass, "rows": rows.len() / contract.hidden as usize, "cosine": cos, "rel_l2": rel, "max_abs": maxd });
                pass
            };
            ok &= run("encode_hf_input", theirs_input.as_ref());
            ok &= run("encode_own_input", ours.as_ref());
        }
        println!("{line}");
    }
    if !ok {
        std::process::exit(1);
    }
}
