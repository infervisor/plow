#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::{apple::MetalEngine, cpu::engine::Chunk};
    use serde_json::{json, Value};
    use std::path::Path;
    let args: Vec<_> = std::env::args().collect();
    if !(5..=6).contains(&args.len()) || args.get(5).is_some_and(|s| s != "--profile-decode") {
        return Err(
            "usage: asr_decoder_trace CHECKPOINT BLOB REFERENCE_DIR OUT_DIR [--profile-decode]"
                .into(),
        );
    }
    let reference = Path::new(&args[3]);
    let out = Path::new(&args[4]);
    std::fs::create_dir_all(out)?;
    let input: Value = serde_json::from_slice(&std::fs::read(reference.join("input.json"))?)?;
    let prompt: Vec<u32> = serde_json::from_value(input["prompt_ids"][0].clone())?;
    let mut engine = MetalEngine::load(Path::new(&args[2]), Path::new(&args[1]))?;
    let (prog, _) = engine
        .prefill_buckets()
        .into_iter()
        .find(|(_, t)| *t as usize >= prompt.len())
        .ok_or("trace needs one prefill bucket large enough for the prompt")?;
    let x = engine
        .model
        .names
        .iter()
        .position(|name| name == "act.x")
        .ok_or("missing act.x")?;
    let embeddings: Vec<u16> = std::fs::read(reference.join("spliced.f32"))?
        .chunks_exact(4)
        .map(|bytes| {
            let u = f32::from_le_bytes(bytes.try_into().unwrap()).to_bits();
            (u.wrapping_add(0x7fff + ((u >> 16) & 1)) >> 16) as u16
        })
        .collect();
    let first_token = engine.prefill_embeddings(&prompt, &embeddings)?;
    let logits = engine.model.wk.logits.ok_or("missing logits")?;
    if args.len() == 6 {
        use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
        let mut records = Vec::new();
        for iteration in 0..4 {
            engine.prefill_embeddings(&prompt, &embeddings)?;
            let started = std::time::Instant::now();
            engine.decode_step(prompt.len() as u32, prompt.len() as u32 + 1)?;
            let normal_seconds = started.elapsed().as_secs_f64();
            let expected = engine.tensor_bytes(logits).to_vec();
            if iteration == 0 {
                std::fs::write(out.join("decode-logits.bf16"), &expected)?;
            }
            engine.prefill_embeddings(&prompt, &embeddings)?;
            let dp =
                engine.prepare_decode(prompt.len() as u32, prompt.len() as u32 + 1, first_token)?;
            let insts = engine.insts_host(dp).to_vec();
            let mut instructions = Vec::new();
            for (index, inst) in insts.iter().enumerate() {
                let cb = engine.run_inst_async(dp, index)?;
                cb.waitUntilCompleted();
                if cb.status() != MTLCommandBufferStatus::Completed {
                    return Err(format!("decode instruction {index}: {:?}", cb.error()).into());
                }
                let op = packet::dev::DevOp::ALL
                    .iter()
                    .find(|&&op| op as u16 == inst.op)
                    .map(|op| format!("{op:?}"));
                let names: Vec<_> = inst
                    .t
                    .iter()
                    .map(|&t| engine.model.names.get(t as usize))
                    .collect();
                instructions.push(json!({"index":index,"op":op,"tensors":names,"i":inst.i,
                    "gpu_us":(cb.GPUEndTime()-cb.GPUStartTime())*1e6}));
            }
            if engine.tensor_bytes(logits) != expected {
                return Err("profile replay differs from normal decode logits".into());
            }
            records.push(
                json!({"iteration":iteration,"normal_seconds":normal_seconds,
                "logits_match":true,"instructions":instructions}),
            );
        }
        std::fs::write(
            out.join("decode-profile.json"),
            serde_json::to_vec_pretty(&records)?,
        )?;
        println!("profiled four decode steps; all replay logits match normal execution");
        return Ok(());
    }
    let normal_logits = engine.tensor_bytes(logits).to_vec();
    engine.prepare_prefill_chunk(
        &prompt,
        Chunk {
            prog,
            c0: 0,
            clen: prompt.len() as u32,
        },
    )?;
    if embeddings.len() * 2 > engine.tensor_bytes(x).len() {
        return Err("embedding buffer too small".into());
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            embeddings.as_ptr().cast::<u8>(),
            engine.host_ptr(x),
            embeddings.len() * 2,
        );
    }
    let insts = engine.insts_host(prog).to_vec();
    let mut metadata = Vec::new();
    let mut layer = 0usize;
    for (index, inst) in insts.iter().enumerate() {
        let names: Vec<_> = inst
            .t
            .iter()
            .map(|&t| {
                engine
                    .model
                    .names
                    .get(t as usize)
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();
        for name in &names {
            if let Some(rest) = name.strip_prefix("thinker.model.layers.") {
                layer = rest.split('.').next().ok_or("layer name")?.parse()?;
            }
        }
        if inst.op == packet::dev::DevOp::Embed as u16 {
            continue;
        }
        engine.run_inst(prog, index)?;
        let mut saved = std::collections::BTreeMap::new();
        for (&handle, name) in inst.t.iter().zip(&names) {
            let capture =
                layer == 0 || name == "act.x" && inst.op == packet::dev::DevOp::Residual as u16;
            if capture && name.starts_with("act.") && !saved.contains_key(name) {
                let file = format!("{index:03}-{name}.bin");
                let bytes = engine.tensor_bytes(handle as usize);
                let bytes = if name == "act.x" {
                    &bytes[..embeddings.len() * 2]
                } else {
                    bytes
                };
                std::fs::write(out.join(&file), bytes)?;
                saved.insert(name.clone(), file);
            }
        }
        let op = packet::dev::DevOp::ALL
            .iter()
            .find(|&&op| op as u16 == inst.op)
            .map(|op| format!("{op:?}"));
        metadata.push(
            json!({"index":index,"layer":layer,"op":op,"tensors":names,"i":inst.i,"fj":inst.fj,"saved":saved}),
        );
    }
    if engine.tensor_bytes(logits) != normal_logits {
        return Err("instruction replay logits differ from normal prefill".into());
    }
    std::fs::write(out.join("logits.bin"), &normal_logits)?;
    std::fs::write(
        out.join("metadata.json"),
        serde_json::to_vec_pretty(&metadata)?,
    )?;
    println!(
        "captured {} instructions to {}",
        metadata.len(),
        out.display()
    );
    Ok(())
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("asr_decoder_trace requires macOS and --features metal");
    std::process::exit(1);
}
