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
    let mut firsts: Vec<(String, plow_asset::multimodal::MmModality, Box<Input>, Vec<f32>)> = Vec::new();
    for (i, item) in reference["items"].as_array().unwrap().iter().enumerate() {
        let kind = item["kind"].as_str().unwrap();
        let path = item["path"].as_str().unwrap();
        let m = contract.modality(kind).expect("modality");
        let bytes = std::fs::read(path).unwrap();
        let mut line = serde_json::json!({ "item": i, "kind": kind, "path": path });
        let (ours, theirs_input): (Box<Input>, Box<Input>);
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
        } else if m.processor() == Ok(plow_asset::multimodal::Processor::WaveformFrames) {
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
            let mut run = |label: &str, input: &Input| {
                let rows = encode_one(enc, m, input);
                let same = rows.len() == hf_rows.len();
                let (cos, rel, maxd) = if same { stats(&rows, &hf_rows) } else { (0.0, f64::NAN, f64::NAN) };
                // bf16 towers: rows agree to bf16 noise accumulated over the layers.
                let pass = same && cos > 0.995 && rel < 0.1;
                line[label] = serde_json::json!({ "pass": pass, "rows": rows.len() / contract.hidden as usize, "cosine": cos, "rel_l2": rel, "max_abs": maxd });
                (pass, rows)
            };
            ok &= run("encode_hf_input", theirs_input.as_ref()).0;
            let (pass, rows) = run("encode_own_input", ours.as_ref());
            ok &= pass;
            firsts.push((kind.to_string(), m.clone(), ours, rows));
        }
        println!("{line}");
    }
    if encode {
        // Rows are a function of the input alone: the same bits re-encoded in reverse order (after
        // other rungs ran), and with every modality's encoder running concurrently on its own
        // thread, as the server runs them.
        drop(encoders);
        let mut reverse_equal = true;
        let mut fresh: std::collections::HashMap<String, plowrt::serve::mm::encoder::Encoder> = Default::default();
        for (kind, m, input, rows) in firsts.iter().rev() {
            let enc = fresh.entry(kind.clone()).or_insert_with(|| {
                plowrt::serve::mm::encoder::Encoder::load(&pkt.with_file_name(&m.packet), 0).expect("encoder")
            });
            reverse_equal &= bits(&encode_one(enc, m, input.as_ref())) == bits(rows);
        }
        drop(fresh);
        let kinds: std::collections::BTreeSet<&str> = firsts.iter().map(|f| f.0.as_str()).collect();
        let concurrent_equal = std::thread::scope(|s| {
            let handles: Vec<_> = kinds
                .iter()
                .map(|&kind| {
                    let (firsts, pkt) = (&firsts, &pkt);
                    s.spawn(move || {
                        let mut enc: Option<plowrt::serve::mm::encoder::Encoder> = None;
                        firsts.iter().filter(|f| f.0 == kind).all(|(_, m, input, rows)| {
                            let e = enc.get_or_insert_with(|| {
                                plowrt::serve::mm::encoder::Encoder::load(&pkt.with_file_name(&m.packet), 0).expect("encoder")
                            });
                            bits(&encode_one(e, m, input.as_ref())) == bits(rows)
                        })
                    })
                })
                .collect();
            handles.into_iter().all(|h| h.join().unwrap())
        });
        ok &= reverse_equal && concurrent_equal;
        println!("{}", serde_json::json!({ "determinism": { "pass": reverse_equal && concurrent_equal, "reverse_order_identical": reverse_equal, "concurrent_identical": concurrent_equal } }));
    }
    if !ok {
        std::process::exit(1);
    }
}

type Input = dyn std::any::Any + Send + Sync;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn encode_one(enc: &mut plowrt::serve::mm::encoder::Encoder, m: &plow_asset::multimodal::MmModality, input: &Input) -> Vec<f32> {
    if let Some(p) = input.downcast_ref::<media::Patches>() {
        return enc.encode_images(&[p]).unwrap().remove(0);
    }
    let mel = input.downcast_ref::<media::Mel>().unwrap();
    let tokens = if m.processor() == Ok(plow_asset::multimodal::Processor::WaveformFrames) {
        mel.valid_frames
    } else {
        media::audio_tokens(mel.valid_frames, m.param("subsample").unwrap_or(4) as usize)
    };
    enc.encode_audio(mel, tokens).unwrap()
}
