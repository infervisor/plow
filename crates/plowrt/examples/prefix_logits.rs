//! Compare full-prompt prefill, split prefill plus decode, and packed prefill.
//! prefix_logits <assets> <ordinary|split|packed|packed-tail> <cases.json> <out> [reference]

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("prefix_logits requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::gpu::{GpuEngine, PfBatchReq, PrefillStep};
    use std::{fs, io::Write, path::PathBuf, sync::Arc};

    let mut args = std::env::args().skip(1);
    let assets = PathBuf::from(args.next().ok_or("assets required")?);
    let mode = args.next().ok_or("mode required")?;
    assert!(matches!(
        mode.as_str(),
        "ordinary" | "split" | "packed" | "packed-tail"
    ));
    let packed = mode.starts_with("packed");
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(args.next().ok_or("cases required")?)?)?;
    let out = PathBuf::from(args.next().ok_or("output directory required")?);
    let reference = args.next().map(PathBuf::from);
    assert_eq!(reference.is_some(), mode != "ordinary");
    assert_eq!(plowrt::config::RuntimeConfig::get().nv.pf_batch, packed);
    assert!(cfg!(target_endian = "little"));
    fs::create_dir(&out)?;
    let tokenizer = plowrt::text::tokenizer::load_tokenizer(&assets);
    let template =
        plowrt::serve::template::ChatTemplate::load(&assets).ok_or("chat template required")?;
    let be = Arc::new(plowrt::device::cuda::CudaBackend::new(0)?);
    for (index, case) in cases.iter().enumerate() {
        let rendered = template.render(case["messages"].as_array().ok_or("messages required")?)?;
        let prompt = tokenizer.encode(&rendered);
        assert_eq!(prompt.len() as u64, case["prompt_tokens"].as_u64().unwrap());
        let steps = case["steps"].as_u64().unwrap() as usize;
        assert!(steps > 0 && prompt.len() > 1);
        let reference_tokens: Option<Vec<u32>> = reference.as_ref().map(|dir| {
            let value: serde_json::Value =
                serde_json::from_slice(&fs::read(dir.join(format!("{index}.json"))).unwrap())
                    .unwrap();
            assert_eq!(value["prompt_ids"], serde_json::json!(prompt));
            serde_json::from_value(value["selected_tokens"].clone()).unwrap()
        });
        if let Some(tokens) = &reference_tokens {
            assert_eq!(tokens.len(), steps);
        }
        let mut e = GpuEngine::load(Arc::clone(&be), &assets, &assets.join("checkpoint"))?;
        e.begin_slot(0, prompt.len() + steps)?;
        let mut tokens = Vec::new();
        let mut next = if mode == "ordinary" {
            e.prefill_slot(0, &prompt)?
        } else {
            assert_eq!(e.attach_prompt(0, &prompt)?, 0);
            let mut position = 0;
            while position + 1 < prompt.len() {
                let len = (prompt.len() - 1 - position).min(e.pf_max_rows());
                if packed {
                    e.prefill_batched(&[PfBatchReq {
                        slot: 0,
                        prompt: &prompt,
                        c0: position,
                        len,
                    }])?;
                    position += len;
                } else {
                    position = match e.prefill_chunk(0, &prompt[..prompt.len() - 1], len)? {
                        PrefillStep::Progress(end) => end,
                        PrefillStep::Done(_) => prompt.len() - 1,
                    };
                }
            }
            if mode == "packed-tail" {
                match e.prefill_chunk(0, &prompt, 1)? {
                    PrefillStep::Done(token) => token,
                    PrefillStep::Progress(_) => return Err("final prompt row not consumed".into()),
                }
            } else {
                e.step_slots(&[(0, *prompt.last().unwrap())], &mut tokens)?;
                tokens[0]
            }
        };
        let mut selected = Vec::with_capacity(steps);
        let mut logits = Vec::new();
        let mut file = std::io::BufWriter::new(fs::File::create(out.join(format!("{index}.f32")))?);
        for step in 0..steps {
            e.logits_row(0, &mut logits)?;
            assert!(!logits.is_empty() && logits.iter().all(|x| x.is_finite()));
            file.write_all(bytemuck::cast_slice(&logits))?;
            selected.push(next);
            if step + 1 < steps {
                let feed = reference_tokens.as_ref().map_or(next, |ids| ids[step]);
                e.step_slots(&[(0, feed)], &mut tokens)?;
                next = tokens[0];
            }
        }
        file.flush()?;
        fs::write(
            out.join(format!("{index}.json")),
            serde_json::to_vec_pretty(&serde_json::json!({
                "mode": mode, "case": case, "prompt_ids": prompt, "vocab": logits.len(),
                "frames": steps, "teacher_forced": reference_tokens.is_some(), "selected_tokens": selected,
            }))?,
        )?;
        println!(
            "case={index} mode={mode} frames={steps} vocab={}",
            logits.len()
        );
    }
    Ok(())
}
