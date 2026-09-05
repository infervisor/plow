//! Teacher-forced CUDA decode dumps: decode_dump <assets> <out> <id,id,...>
//! <tensor>... [--input-f32 <file>]. Block assets require one little-endian f32
//! hidden-state row per token in --input-f32. Tensor files retain their raw dtype.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("decode_dump requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::{device::cuda::CudaBackend, exec::gpu::GpuEngine};
    use std::{path::PathBuf, sync::Arc};

    tracing_subscriber::fmt().with_env_filter("info").init();
    let mut args = std::env::args().skip(1);
    let assets = PathBuf::from(args.next().ok_or("missing assets directory")?);
    let out = PathBuf::from(args.next().ok_or("missing output directory")?);
    let ids: Vec<u32> = args
        .next()
        .ok_or("missing comma-separated token IDs")?
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let mut names = Vec::new();
    let mut input_path = None;
    let mut verify_lifecycle = false;
    let mut prefill_prefix = None;
    let mut batch_reference = None;
    while let Some(arg) = args.next() {
        if arg == "--verify-batch-reference" {
            batch_reference = Some(PathBuf::from(
                args.next().ok_or("missing B1 reference directory")?,
            ));
        } else if arg == "--verify-lifecycle" {
            verify_lifecycle = true;
        } else if arg == "--prefill-prefix" {
            prefill_prefix = Some(
                args.next()
                    .ok_or("missing prefill prefix length")?
                    .parse::<usize>()?,
            );
        } else if arg == "--input-f32" {
            input_path = Some(PathBuf::from(args.next().ok_or("missing input file")?));
        } else {
            names.push(arg);
        }
    }
    let input = if assets.join("block.json").exists() {
        let desc: plow_asset::BlockDescriptor =
            serde_json::from_slice(&std::fs::read(assets.join("block.json"))?)?;
        let raw = std::fs::read(input_path.ok_or("block asset requires --input-f32")?)?;
        let hidden = desc.hidden as usize;
        if raw.len() != ids.len() * hidden * 4 {
            return Err("input must contain one f32 hidden-state row per token".into());
        }
        Some((
            hidden,
            raw.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>(),
        ))
    } else {
        if input_path.is_some() {
            return Err("--input-f32 requires a block asset".into());
        }
        None
    };
    let checkpoint = std::env::var_os("PLOW_CHECKPOINT")
        .map(PathBuf::from)
        .unwrap_or_else(|| assets.join("checkpoint"));
    if verify_lifecycle && input.is_some() {
        return Err("--verify-lifecycle requires a full-model asset".into());
    }
    if let Some(n) = prefill_prefix {
        if n == 0 || n > ids.len() || input.is_some() || verify_lifecycle {
            return Err("--prefill-prefix requires 1..=prompt length, a full model, and no --verify-lifecycle".into());
        }
    }
    if batch_reference.is_some() && (input.is_some() || verify_lifecycle || ids.len() < 2) {
        return Err(
            "batch verification requires a full model, at least two IDs, and no --verify-lifecycle"
                .into(),
        );
    }
    let mut engine = GpuEngine::load(Arc::new(CudaBackend::new(0)?), &assets, &checkpoint)?;
    if ids.iter().any(|&id| id as usize >= engine.vocab()) {
        return Err("token ID exceeds vocabulary".into());
    }
    let tensors: Vec<_> = names
        .iter()
        .map(|name| {
            engine
                .tensor_bytes(name)
                .map(|bytes| (name.clone(), bytes))
                .ok_or_else(|| format!("no tensor {name:?}"))
        })
        .collect::<Result<_, _>>()?;
    std::fs::create_dir_all(&out)?;
    engine.begin_slot(0, ids.len())?;
    let mut tokens = Vec::new();
    let mut logits = Vec::new();
    let mut steps = Vec::new();
    let mut reference_rows = Vec::new();
    let start = prefill_prefix.map_or(0, |n| n - 1);
    for (step, &id) in ids.iter().enumerate().skip(start) {
        if let Some((hidden, values)) = &input {
            engine.upload_activation("act.x", &values[step * hidden..(step + 1) * hidden])?;
        }
        if step == start && prefill_prefix.is_some() {
            tokens.push(engine.prefill_slot(0, &ids[..step + 1])?);
        } else {
            engine.step_slots(&[(0, id)], &mut tokens)?;
        }
        let mut dumps = Vec::new();
        for (index, (name, bytes)) in tensors.iter().enumerate() {
            let mut raw = vec![0u8; usize::try_from(*bytes)?];
            engine.read_tensor(name, &mut raw)?;
            let file = format!("step{step:04}-tensor{index:03}.raw");
            std::fs::write(out.join(&file), raw)?;
            dumps.push(serde_json::json!({"name": name, "bytes": bytes, "file": file}));
        }
        let logits_file = if input.is_none() {
            engine.logits_row(0, &mut logits)?;
            if verify_lifecycle {
                reference_rows.push(logits.clone());
            }
            let file = format!("step{step:04}-logits.f32");
            let raw: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(out.join(&file), raw)?;
            Some(file)
        } else {
            None
        };
        steps.push(serde_json::json!({"step": step, "input_token": id,
            "argmax": tokens.first(), "tensors": dumps, "logits_f32": logits_file}));
        std::fs::write(
            out.join("manifest.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"assets": assets, "batch": engine.batch(), "mode": if prefill_prefix.is_some() { "prefill-then-decode" } else { "decode-only" },
                "prefill_prefix": prefill_prefix, "token_ids": ids,
                "block": input.is_some(), "steps": steps}),
            )?,
        )?;
    }
    if let Some(reference) = batch_reference {
        verify_batch(
            &mut engine,
            &checkpoint,
            &out,
            &ids,
            &reference,
            prefill_prefix,
        )?;
    }
    if verify_lifecycle {
        verify_reuse(&mut engine, &checkpoint, &out, &ids, &reference_rows)?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn verify_reuse(
    engine: &mut plowrt::exec::gpu::GpuEngine,
    checkpoint: &std::path::Path,
    out: &std::path::Path,
    ids: &[u32],
    reference: &[Vec<f32>],
) -> Result<(), Box<dyn std::error::Error>> {
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(checkpoint.join("config.json"))?)?;
    let text = config.get("text_config").unwrap_or(&config);
    let layers = text["num_hidden_layers"].as_u64().unwrap_or(0) as usize;
    let states: Vec<_> = (0..layers)
        .flat_map(|l| {
            [
                format!("state.qwen.{l}.conv"),
                format!("state.qwen.{l}.gdn"),
            ]
        })
        .filter_map(|name| engine.tensor_bytes(&name).map(|bytes| (name, bytes)))
        .collect();
    let batch = engine.batch();
    let mut runs = vec![0, 0];
    if batch > 1 {
        runs.push(batch - 1);
    }
    let mut tokens = Vec::new();
    let mut logits = Vec::new();
    let mut completed = Vec::new();
    for (repeat, slot) in runs.into_iter().enumerate() {
        let prior = inactive_state(engine, &states, slot, batch)?;
        engine.begin_slot(slot, ids.len())?;
        if inactive_state(engine, &states, slot, batch)? != prior {
            return Err(format!("reset changed inactive recurrent state for slot {slot}").into());
        }
        for (step, &id) in ids.iter().enumerate() {
            engine.step_slots(&[(slot, id)], &mut tokens)?;
            engine.logits_row(slot, &mut logits)?;
            if logits.len() != reference[step].len()
                || logits
                    .iter()
                    .zip(&reference[step])
                    .any(|(a, b)| !a.is_finite() || a.to_bits() != b.to_bits())
            {
                return Err(format!(
                    "lifecycle logits differ: repeat {repeat}, slot {slot}, step {step}"
                )
                .into());
            }
            if inactive_state(engine, &states, slot, batch)? != prior {
                return Err(format!(
                    "decode changed inactive recurrent state: slot {slot}, step {step}"
                )
                .into());
            }
        }
        completed.push(serde_json::json!({"repeat": repeat, "slot": slot, "steps": ids.len(),
            "logits_bit_exact": true, "inactive_state_tensors_checked": if batch > 1 { states.len() } else { 0 }}));
        std::fs::write(
            out.join("lifecycle.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"batch": batch, "completed": completed}),
            )?,
        )?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn inactive_state(
    engine: &mut plowrt::exec::gpu::GpuEngine,
    states: &[(String, u64)],
    active: usize,
    batch: usize,
) -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
    if batch == 1 {
        return Ok(Vec::new());
    }
    states
        .iter()
        .map(|(name, bytes)| {
            let bytes = usize::try_from(*bytes)?;
            if bytes % batch != 0 {
                return Err(format!("state {name} is not evenly slot-strided").into());
            }
            let stride = bytes / batch;
            let mut raw = vec![0; bytes];
            engine.read_tensor(name, &mut raw)?;
            raw.drain(active * stride..(active + 1) * stride);
            Ok(raw)
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn verify_batch(
    engine: &mut plowrt::exec::gpu::GpuEngine,
    checkpoint: &std::path::Path,
    out: &std::path::Path,
    ids: &[u32],
    reference_dir: &std::path::Path,
    prefill_prefix: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    if engine.batch() != 4 {
        return Err("batch verification requires B4".into());
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(reference_dir.join("manifest.json"))?)?;
    if manifest["batch"] != 1 || manifest["token_ids"] != serde_json::json!(ids) {
        return Err("B1 reference must contain the same token IDs".into());
    }
    let first_row = match manifest["mode"].as_str() {
        Some("decode-only") => 0,
        Some("prefill-then-decode") => {
            let first = manifest["prefill_prefix"]
                .as_u64()
                .ok_or("missing reference prefill prefix")? as usize;
            let tested = prefill_prefix.ok_or("native reference requires a prefill batch check")?;
            if first == 0 || first > tested || first > ids.len() {
                return Err(
                    "reference prefill prefix must be positive and no later than tested prefix"
                        .into(),
                );
            }
            first - 1
        }
        _ => return Err("unsupported B1 reference mode".into()),
    };
    let mut reference = vec![Vec::new(); ids.len()];
    for (step, row) in reference.iter_mut().enumerate().skip(first_row) {
        let raw = std::fs::read(reference_dir.join(format!("step{step:04}-logits.f32")))?;
        if raw.len() != engine.vocab() * 4 {
            return Err("reference vocabulary mismatch".into());
        }
        *row = raw
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
    }
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(checkpoint.join("config.json"))?)?;
    let text = config.get("text_config").unwrap_or(&config);
    let layers = text["num_hidden_layers"]
        .as_u64()
        .ok_or("missing layer count")?;
    let states: Vec<_> = (0..layers)
        .flat_map(|l| {
            [
                format!("state.qwen.{l}.conv"),
                format!("state.qwen.{l}.gdn"),
            ]
        })
        .filter_map(|name| engine.tensor_bytes(&name).map(|bytes| (name, bytes)))
        .collect();
    if states.is_empty() {
        return Err("batch diagnostic requires recurrent state".into());
    }
    if let Some(prefix) = prefill_prefix {
        let mut persistent = states;
        persistent.extend(
            (0..layers)
                .flat_map(|l| [format!("kv.{l}.k"), format!("kv.{l}.v")])
                .filter_map(|name| engine.tensor_bytes(&name).map(|bytes| (name, bytes))),
        );
        return verify_batch_prefill(engine, out, ids, &reference, &persistent, prefix);
    }
    let mut rows = Vec::new();
    let mut logits = Vec::new();
    let mut tokens = Vec::new();
    for delay in [0usize, 1] {
        engine.begin_slot(0, ids.len())?;
        let preserved = inactive_state(engine, &states, 3, 4)?;
        engine.begin_slot(3, ids.len())?;
        if inactive_state(engine, &states, 3, 4)? != preserved {
            return Err("slot3 reset changed another slot".into());
        }
        for tick in 0..ids.len() + delay {
            let mut feeds = Vec::new();
            if tick < ids.len() {
                feeds.push((0, ids[tick]));
            }
            if tick >= delay && tick - delay < ids.len() {
                feeds.push((3, ids[tick - delay]));
            }
            let idle: Vec<_> = (0..4)
                .filter(|slot| !feeds.iter().any(|(s, _)| s == slot))
                .collect();
            let snapshot = |engine: &mut plowrt::exec::gpu::GpuEngine| -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
                states.iter().map(|(name, bytes)| {
                    let mut raw = vec![0; *bytes as usize]; engine.read_tensor(name, &mut raw)?;
                    let stride = raw.len()/4;
                    Ok(idle.iter().flat_map(|&slot| raw[slot*stride..(slot+1)*stride].iter().copied()).collect())
                }).collect()
            };
            let before = snapshot(engine)?;
            engine.step_slots(&feeds, &mut tokens)?;
            if snapshot(engine)? != before {
                return Err(format!("idle state changed at delay{delay} tick{tick}").into());
            }
            for &(slot, _) in &feeds {
                let step = if slot == 0 { tick } else { tick - delay };
                engine.logits_row(slot, &mut logits)?;
                let expected = &reference[step];
                let mut squared = 0f64;
                let mut norm = 0f64;
                let mut max_abs = 0f32;
                for (&a, &b) in logits.iter().zip(expected) {
                    if !a.is_finite() || !b.is_finite() {
                        return Err("nonfinite batch/reference logits".into());
                    }
                    squared += f64::from(a - b).powi(2);
                    norm += f64::from(b).powi(2);
                    max_abs = max_abs.max((a - b).abs());
                }
                let rel_l2 = (squared / norm.max(1e-30)).sqrt();
                let top = |v: &[f32]| {
                    v.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|x| x.0)
                };
                let top_equal = top(&logits) == top(expected);
                rows.push(serde_json::json!({"delay":delay,"slot":slot,"step":step,"rel_l2":rel_l2,"max_abs":max_abs,"top1_equal":top_equal}));
                std::fs::write(
                    out.join("batch-lifecycle.json"),
                    serde_json::to_vec_pretty(
                        &serde_json::json!({"reference":reference_dir,"relative_l2_limit":0.01,"rows":rows,"complete":false}),
                    )?,
                )?;
                if logits.len() != expected.len() || rel_l2 > 0.01 || !top_equal {
                    return Err(format!(
                        "B4/B1 logits differ at slot{slot} step{step}: rel_l2={rel_l2}"
                    )
                    .into());
                }
            }
        }
        let preserved = inactive_state(engine, &states, 3, 4)?;
        engine.begin_slot(3, ids.len())?;
        if inactive_state(engine, &states, 3, 4)? != preserved {
            return Err("slot3 reset changed completed slot0".into());
        }
        for &id in ids {
            engine.step_slots(&[(3, id)], &mut tokens)?;
        }
        if inactive_state(engine, &states, 3, 4)? != preserved {
            return Err("slot3 replay changed idle slot0".into());
        }
    }
    std::fs::write(
        out.join("batch-lifecycle.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"reference":reference_dir,"relative_l2_limit":0.01,"rows":rows,"complete":true,"idle_state_exact":true,"slot3_reset_preserves_slot0":true}),
        )?,
    )?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn verify_batch_prefill(
    engine: &mut plowrt::exec::gpu::GpuEngine,
    out: &std::path::Path,
    ids: &[u32],
    reference: &[Vec<f32>],
    persistent: &[(String, u64)],
    prefix: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if prefix < 128 || prefix + 2 >= ids.len() {
        return Err(
            "batch prefill verification needs prefix>=128 and at least three continuation IDs"
                .into(),
        );
    }
    let mut records = Vec::new();
    let mut tokens = Vec::new();
    let mut logits = Vec::new();
    let mut check = |engine: &mut plowrt::exec::gpu::GpuEngine,
                     row: usize,
                     slot: usize,
                     step: usize,
                     mixed: bool,
                     phase: &str|
     -> Result<(), Box<dyn std::error::Error>> {
        engine.logits_row(row, &mut logits)?;
        let expected = reference
            .get(step)
            .filter(|row| !row.is_empty())
            .ok_or("missing required B1 reference row")?;
        if logits.len() != expected.len() || logits.iter().chain(expected).any(|x| !x.is_finite()) {
            return Err("invalid native prefill/reference logits".into());
        }
        let norm = expected.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
        let rel_l2 = (logits
            .iter()
            .zip(expected)
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum::<f64>()
            / norm.max(1e-30))
        .sqrt();
        let top = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|x| x.0)
        };
        let top_equal = top(&logits) == top(expected);
        records.push(serde_json::json!({"mixed":mixed,"slot":slot,"logits_row":row,"step":step,"phase":phase,"rel_l2":rel_l2,"top1_equal":top_equal}));
        std::fs::write(
            out.join("batch-prefill-lifecycle.json"),
            serde_json::to_vec_pretty(
                &serde_json::json!({"prefix":prefix,"rows":records,"complete":false,"relative_l2_limit":0.01}),
            )?,
        )?;
        if rel_l2 > 0.01 || !top_equal {
            return Err(
                format!("native B4/B1 mismatch slot{slot} step{step}: rel_l2={rel_l2}").into(),
            );
        }
        Ok(())
    };
    for mixed in [false, true] {
        let mut consumed = [prefix, prefix + if mixed { 2 } else { 0 }];
        for (index, slot) in [0usize, 3].into_iter().enumerate() {
            let before = inactive_state(engine, persistent, slot, 4)?;
            engine.begin_slot(slot, ids.len())?;
            if inactive_state(engine, persistent, slot, 4)? != before {
                return Err("prefill reset changed inactive state/KV".into());
            }
            engine.prefill_slot(slot, &ids[..consumed[index]])?;
            if inactive_state(engine, persistent, slot, 4)? != before {
                return Err("serial prefill changed inactive state/KV".into());
            }
            // Native buckets are multiples of128; a decode remainder leaves logits in its slot.
            let row = if consumed[index] % 128 == 0 { 0 } else { slot };
            check(engine, row, slot, consumed[index] - 1, mixed, "prefill")?;
        }
        while consumed.iter().any(|&n| n < ids.len()) {
            let feeds: Vec<_> = [0usize, 3]
                .into_iter()
                .enumerate()
                .filter_map(|(i, s)| (consumed[i] < ids.len()).then(|| (s, ids[consumed[i]])))
                .collect();
            let idle: Vec<_> = (0..4)
                .filter(|slot| !feeds.iter().any(|(s, _)| s == slot))
                .collect();
            let snapshot = |engine: &mut plowrt::exec::gpu::GpuEngine| -> Result<Vec<Vec<u8>>, Box<dyn std::error::Error>> {
                persistent.iter().map(|(name,bytes)| {
                    let mut raw = vec![0; *bytes as usize]; engine.read_tensor(name,&mut raw)?;
                    let stride = raw.len()/4;
                    Ok(idle.iter().flat_map(|&slot| raw[slot*stride..(slot+1)*stride].iter().copied()).collect())
                }).collect()
            };
            let before = snapshot(engine)?;
            engine.step_slots(&feeds, &mut tokens)?;
            if snapshot(engine)? != before {
                return Err("continuation changed inactive state/KV".into());
            }
            for (index, slot) in [0usize, 3].into_iter().enumerate() {
                if consumed[index] < ids.len() {
                    check(engine, slot, slot, consumed[index], mixed, "decode")?;
                    consumed[index] += 1;
                }
            }
        }
    }
    std::fs::write(
        out.join("batch-prefill-lifecycle.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"prefix":prefix,"rows":records,"complete":true,"relative_l2_limit":0.01,"inactive_state_and_kv_exact":true}),
        )?,
    )?;
    Ok(())
}
