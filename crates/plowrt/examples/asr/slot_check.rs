#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder,
        MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions,
    };
    use plowrt::exec::apple::MetalEngine;
    use serde_json::{json, Value};
    use std::path::Path;
    use std::ptr::NonNull;
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err("usage: asr_slot_check CHECKPOINT B1_PACKET BATCH_PACKET REFERENCE OUT".into());
    }
    let reference = Path::new(&args[4]);
    let output = Path::new(&args[5]);
    std::fs::create_dir_all(output)?;
    let input: Value = serde_json::from_slice(&std::fs::read(reference.join("input.json"))?)?;
    let prompt: Vec<u32> = serde_json::from_value(input["prompt_ids"][0].clone())?;
    let embeddings: Vec<u16> = std::fs::read(reference.join("spliced.f32"))?
        .chunks_exact(4)
        .map(|b| {
            let bits = f32::from_le_bytes(b.try_into().unwrap()).to_bits();
            (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        })
        .collect();
    assert!(!prompt.is_empty() && embeddings.len() % prompt.len() == 0);
    let hidden = embeddings.len() / prompt.len();
    let mut engine = MetalEngine::load(Path::new(&args[3]), Path::new(&args[1]))?;
    let batch = engine.model.batch;
    assert!(batch >= 2 && batch <= prompt.len());
    let lengths: Vec<_> = (1..=batch).map(|n| prompt.len() / n).collect();
    let mut first = Vec::new();
    let mut expected = Vec::new();
    let mut expected_tokens = Vec::new();
    for &length in &lengths {
        let mut engine = MetalEngine::load(Path::new(&args[2]), Path::new(&args[1]))?;
        assert_eq!(engine.model.batch, 1);
        first.push(engine.prefill_embeddings(&prompt[..length], &embeddings[..length * hidden])?);
        let logits = engine.model.wk.logits.ok_or("missing logits")?;
        let mut steps = Vec::new();
        let mut step_tokens = Vec::new();
        for step in 0..4 {
            step_tokens
                .push(engine.decode_step((length + step) as u32, (length + step + 1) as u32)?);
            steps.push(engine.tensor_bytes(logits).to_vec());
        }
        expected.push(steps);
        expected_tokens.push(step_tokens);
    }
    let logits = engine.model.wk.logits.ok_or("missing logits")?;
    let device = MTLCreateSystemDefaultDevice().ok_or("Metal device")?;
    // The copied fixture remains owned through every synchronous prefill.
    let source = unsafe {
        device.newBufferWithBytes_length_options(
            NonNull::new(embeddings.as_ptr() as *mut std::ffi::c_void).unwrap(),
            embeddings.len() * 2,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or("fixture buffer")?;
    let mut records = Vec::new();
    for widths in [vec![batch; 4], vec![batch, 1, 2, batch]] {
        for device_stage in [false, true] {
            for order in [(0..batch).collect::<Vec<_>>(), (0..batch).rev().collect()] {
                let mut tokens = vec![0; batch];
                for slot in 0..batch {
                    let request = order[slot];
                    let length = lengths[request];
                    tokens[slot] = if device_stage {
                        engine.prefill_slot_embeddings_staged(
                        slot,
                        &prompt[..length],
                        length * hidden,
                        true,
                        |cb, output, _, ch, width, _| {
                            assert_eq!(width, hidden);
                            let cb = cb.ok_or_else(|| {
                                plowrt::RuntimeError::Device("expected device staging".into())
                            })?;
                            let offset = ch.c0 as usize * hidden * 2;
                            let bytes = ch.clen as usize * hidden * 2;
                            assert!(offset + bytes <= source.length() && bytes <= output.length());
                            let blit = cb.blitCommandEncoder().ok_or_else(|| {
                                plowrt::RuntimeError::Device("blit encoder".into())
                            })?;
                            unsafe {
                                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                                    &source, offset, output, 0, bytes,
                                );
                            }
                            blit.endEncoding();
                            Ok(())
                        },
                    )?
                    } else {
                        engine.prefill_slot_embeddings(
                            slot,
                            &prompt[..length],
                            &embeddings[..length * hidden],
                        )?
                    };
                    assert_eq!(tokens[slot], first[request], "prefill slot {slot}");
                    assert_eq!(engine.model.kv_slot(), 0);
                }
                assert!(engine.prefill_slot_embeddings(1, &[], &[]).is_err());
                assert_eq!(engine.model.kv_slot(), 0, "error must restore table base");
                assert!(engine
                    .prefill_slot_embeddings(batch, &prompt, &embeddings)
                    .is_err());
                assert_eq!(engine.model.kv_slot(), 0);
                let mut callback_called = false;
                let error = engine.prefill_slot_embeddings_staged(
                    1,
                    &prompt,
                    embeddings.len(),
                    true,
                    |_, _, _, _, _, _| {
                        callback_called = true;
                        Err(plowrt::RuntimeError::Rejected(
                            "injected staging failure".into(),
                        ))
                    },
                );
                assert!(callback_called);
                assert!(
                    matches!(error, Err(plowrt::RuntimeError::Rejected(message)) if message == "injected staging failure")
                );
                assert_eq!(
                    engine.model.kv_slot(),
                    0,
                    "staging failure must restore table base"
                );
                let mut steps = vec![0; batch];
                for &width in &widths {
                    let dp = engine.model.decode_prog_for(width);
                    let rows = engine.model.blob.progs[dp].t as usize;
                    let pos: Vec<_> = order
                        .iter()
                        .enumerate()
                        .map(|(slot, &r)| (lengths[r] + steps[slot]) as u32)
                        .collect();
                    let kvlen: Vec<_> = pos.iter().map(|&p| p + 1).collect();
                    let next = engine.decode_step_batched_at(&pos, &kvlen, &tokens, dp)?;
                    tokens[..rows].copy_from_slice(&next[..rows]);
                    let actual = engine.tensor_bytes(logits);
                    for slot in 0..rows {
                        let step = steps[slot];
                        assert_eq!(
                            tokens[slot], expected_tokens[order[slot]][step],
                            "sampled token"
                        );
                        let wanted = &expected[order[slot]][step];
                        let row = actual
                            .get(slot * wanted.len()..(slot + 1) * wanted.len())
                            .ok_or("logit row bounds")?;
                        let differences = row
                            .chunks_exact(2)
                            .zip(wanted.chunks_exact(2))
                            .filter(|(a, b)| a != b)
                            .count();
                        records.push(json!({"widths":widths,"rows":rows,"device_stage":device_stage,"order":order,"step":step,"slot":slot,
                    "length":lengths[order[slot]],"different_bf16":differences,"token":tokens[slot]}));
                        std::fs::write(
                            output.join("results.json"),
                            serde_json::to_vec_pretty(&records)?,
                        )?;
                        steps[slot] += 1;
                        if differences != 0 {
                            return Err(format!(
                            "batched logits differ: slot {slot}, step {step}, values {differences}"
                        )
                            .into());
                        }
                    }
                }
            }
        }
    }
    println!("{} batched logit rows match isolated execution; host/device staging, swapped slot reuse and error restoration pass", records.len());
    Ok(())
}
#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("asr_slot_check requires macOS and the metal feature");
    std::process::exit(1);
}
