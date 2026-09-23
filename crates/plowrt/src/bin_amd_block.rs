use std::{path::PathBuf, sync::Arc, time::Instant};

use plowrt::{
    asset::devblob::DevBlob,
    device::hsa::HsaBackend,
    exec::{amd::AmdEngine, amd_tp::AmdTpGroup},
};

type Error = Box<dyn std::error::Error>;

fn check_output(bytes: &[u8], reference: &[u8], tolerance: f64) -> Result<f64, Error> {
    if bytes.len() != reference.len() || bytes.len() % 2 != 0 || bytes.is_empty() {
        return Err("reference residual size differs from block output".into());
    }
    let mut error = 0.0;
    let mut norm = 0.0;
    for (a, b) in bytes.chunks_exact(2).zip(reference.chunks_exact(2)) {
        let value =
            |v: &[u8]| f32::from_bits(u32::from(u16::from_le_bytes([v[0], v[1]])) << 16) as f64;
        let (a, b) = (value(a), value(b));
        if !a.is_finite() || !b.is_finite() {
            return Err("non-finite block output or reference".into());
        }
        error += (a - b).powi(2);
        norm += b * b;
    }
    let relative = (error / norm).sqrt();
    if !relative.is_finite() || relative > tolerance {
        return Err(format!("block oracle FAIL: rel-L2={relative}, limit={tolerance}").into());
    }
    Ok(relative)
}

fn check_rows(bytes: &[u8], reference: &[u8], batch: usize, tolerance: f64) -> Result<f64, Error> {
    if batch == 0
        || bytes.len() != reference.len()
        || bytes.len() % (2 * batch) != 0
        || bytes.is_empty()
    {
        return Err("invalid batched reference shape".into());
    }
    let row_bytes = bytes.len() / batch;
    let mut worst = 0.0f64;
    for (row, (actual, expected)) in bytes
        .chunks_exact(row_bytes)
        .zip(reference.chunks_exact(row_bytes))
        .enumerate()
    {
        let relative =
            check_output(actual, expected, tolerance).map_err(|e| format!("row {row}: {e}"))?;
        worst = worst.max(relative);
    }
    Ok(worst)
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    blob: PathBuf,
    hsaco: PathBuf,
    checkpoint: Option<PathBuf>,
    inputs: PathBuf,
    ctx: u32,
    repeat: u32,
    warmup: u32,
    tp: u32,
    dump: Option<PathBuf>,
    report: Option<PathBuf>,
) -> Result<(), Error> {
    if ctx == 0 || repeat == 0 {
        return Err("context and repeat must be positive".into());
    }
    let parsed = DevBlob::parse_l2(&std::fs::read(&blob)?, true)?;
    if parsed.tp_degree() != tp {
        return Err(format!("block packet is TP{}, requested TP{tp}", parsed.tp_degree()).into());
    }
    let uses_logits = parsed
        .tensors
        .iter()
        .position(|t| t.name == "act.logits")
        .is_some_and(|handle| {
            parsed.progs.iter().any(|p| {
                p.insts
                    .iter()
                    .any(|inst| inst.t.iter().any(|&t| usize::from(t) == handle))
            })
        });
    if checkpoint.is_none() || uses_logits {
        return Err("--input-dir requires a weight-bound act.x-only block".into());
    }
    let mut operands = Vec::new();
    for entry in std::fs::read_dir(&inputs)? {
        let path = entry?.path();
        if path.extension().is_none_or(|e| e != "bin") {
            continue;
        }
        let name = path
            .file_stem()
            .ok_or("invalid operand filename")?
            .to_string_lossy()
            .into_owned();
        if name != "act.x" && name != "act.iidx" && !name.starts_with("kv.") {
            return Err(format!("{name}: inputs may only contain act.x, act.iidx and kv.*").into());
        }
        let tensor = parsed
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("{name}: tensor absent from packet"))?;
        let bytes = std::fs::read(path)?;
        if bytes.is_empty() || bytes.len() as u64 > tensor.bytes || bytes.len() % 2 != 0 {
            return Err(format!(
                "{name}: invalid operand size {} (capacity {})",
                bytes.len(),
                tensor.bytes
            )
            .into());
        }
        if name == "act.x" && bytes.len() as u64 != tensor.bytes {
            return Err("act.x.bin must cover the full packet input tensor".into());
        }
        operands.push((name, bytes));
    }
    if !operands.iter().any(|(n, _)| n == "act.x") {
        return Err("missing act.x.bin".into());
    }
    for tensor in &parsed.tensors {
        if ((ctx > 1 && tensor.name.starts_with("kv.")) || tensor.name == "act.iidx")
            && !operands.iter().any(|(n, _)| n == &tensor.name)
        {
            return Err(format!("missing carried-state operand {}.bin", tensor.name).into());
        }
    }
    let output = "act.xnext";
    let output_bytes = parsed
        .tensors
        .iter()
        .find(|t| t.name == output)
        .ok_or("packet has no act.xnext; use a single-layer block")?
        .bytes as usize;
    let reference = std::fs::read(inputs.join("reference.bf16"))?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(inputs.join("reference.json"))?)?;
    let batch = usize::try_from(metadata["batch"].as_u64().unwrap_or(1))?;
    if batch == 0 || batch > 64 {
        return Err("capture batch must be in 1..=64".into());
    }
    let positions = vec![ctx - 1; batch];
    let kv_lengths = vec![ctx; batch];
    if metadata["ctx"].as_u64() != Some(u64::from(ctx)) || reference.len() != output_bytes {
        return Err("reference context or residual shape differs from the requested block".into());
    }
    let tolerance = metadata["tolerance_rel_l2"]
        .as_f64()
        .ok_or("reference has no tolerance_rel_l2")?;
    if !tolerance.is_finite() || tolerance <= 0.0 || tolerance > 0.03 {
        return Err("reference tolerance must be in (0, 0.03]".into());
    }
    let iterations = warmup
        .checked_add(repeat)
        .and_then(|n| n.checked_add(1))
        .ok_or("iteration count overflow")?;
    let stages = metadata["stages"]
        .as_array()
        .ok_or("reference needs per-stage gates; residual alone is insufficient")?
        .iter()
        .map(|name| -> Result<_, Error> {
            let name = name.as_str().ok_or("invalid stage name")?.to_string();
            let bytes = std::fs::read(inputs.join(format!("{name}.reference.bf16")))?;
            let tensor = parsed
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or("reference stage absent from packet")?;
            if bytes.len() as u64 != tensor.bytes {
                return Err(format!("{name}: stage shape mismatch").into());
            }
            Ok((name, bytes))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if stages.is_empty() {
        return Err("empty per-stage reference gates".into());
    }
    let check_stages = |engine: &AmdEngine, rank: usize| -> Result<(), Error> {
        if let Some(dir) = &dump {
            std::fs::create_dir_all(dir)?;
            for name in ["in.icos", "in.isin"] {
                if let Some(size) = engine.tensor_bytes(name) {
                    if size as usize % engine.max_ctx() != 0 {
                        return Err("invalid indexer RoPE table capacity".into());
                    }
                    let mut bytes = vec![0; size as usize / engine.max_ctx() * ctx as usize];
                    engine.read_tensor(name, &mut bytes)?;
                    std::fs::write(dir.join(format!("rank{rank}.{name}.bin")), bytes)?;
                }
            }
            for name in [
                "act.xnext",
                "act.xmid",
                "act.xn",
                "act.qlr",
                "act.ckvraw",
                "act.krr",
                "act.qkva_xq",
                "act.qkva_xs",
                "act.qlat",
                "act.qb",
                "act.qb_xq",
                "act.qb_xs",
                "act.qidx",
                "act.kidx_raw",
                "act.kidx_normed",
                "act.widx",
                "act.iidx",
                "act.qa",
                "act.qrr",
                "act.qr",
                "act.olat",
                "act.opart",
                "act.mlpart",
                "act.xn2",
                "act.attn",
                "act.rlogit",
                "act.tab",
                "act.shared",
                "act.shfu",
                "act.shfu_up",
                "act.sh_gate",
                "act.sh_xq",
                "act.sh_xs",
                "act.sh_hq",
                "act.sh_hs",
                "act.routed_xq",
                "act.routed_xs",
                "act.routed_hq",
                "act.routed_hs",
                "act.moe_fug",
                "act.moe_meta",
                "act.moe_rowtok",
                "act.moe_rowpart",
                "act.moe_rowgate",
                "act.part",
                "act.oat",
                "act.blk_xq",
                "act.blk_xs",
                "act.og_tp",
            ] {
                if let Some(size) = engine.tensor_bytes(name) {
                    let mut bytes = vec![0; size as usize];
                    engine.read_tensor(name, &mut bytes)?;
                    std::fs::write(dir.join(format!("rank{rank}.{name}.bin")), bytes)?;
                }
            }
            for tensor in parsed.tensors.iter().filter(|t| {
                t.name.ends_with(".self_attn.o_proj.weight_fp8")
                    || t.name.ends_with(".self_attn.fused_qkv_a_proj.weight_fp8")
                    || t.name.ends_with(".self_attn.fused_qkv_a_proj.weight_scale_inv")
                    || t.name.ends_with(".self_attn.q_b_proj.weight")
                    || t.name.ends_with(".self_attn.q_b_proj.weight_scale_inv")
                    || t.name.contains(".self_attn.indexer.")
                    || t.name.contains(".self_attn.derived.mla_fp8_tp8.")
                    || t.name.ends_with(".self_attn.q_a_layernorm.weight")
                    || t.name.ends_with(".self_attn.kv_a_layernorm.weight")
                    || t.name.ends_with(".self_attn.o_proj.weight_scale_inv")
                    || (t.name.contains(".mlp.shared_experts.")
                        && (t.name.ends_with(".weight_fp8") || t.name.ends_with(".weight_scale_inv")))
            }) {
                let mut bytes = vec![0; tensor.bytes as usize];
                engine.read_tensor(&tensor.name, &mut bytes)?;
                std::fs::write(dir.join(format!("rank{rank}.{}.bin", tensor.name)), bytes)?;
            }
            for slot in 0..batch {
                engine.dump_slot_kv(dir, &format!("rank{rank}"), slot, ctx)?;
            }
        }
        for (name, reference) in &stages {
            let mut bytes = vec![0; reference.len()];
            engine.read_tensor(name, &mut bytes)?;
            check_rows(&bytes, reference, batch, tolerance).map_err(|e| format!("{name}: {e}"))?;
        }
        Ok(())
    };
    let tp = u8::try_from(tp).map_err(|_| "TP degree exceeds device ordinal range")?;
    let backends = (0..tp)
        .map(|r| HsaBackend::new(r).map(Arc::new))
        .collect::<plowrt::Result<Vec<_>>>()?;
    let mut single;
    let mut group;
    let mut samples = Vec::new();
    let mut outputs = Vec::new();
    let validation_failure = |errors: Vec<String>| -> Result<(), Error> {
        if errors.is_empty() {
            return Ok(());
        }
        if let Some(path) = &report {
            std::fs::write(path, serde_json::to_string_pretty(&serde_json::json!({
                "scope": "single-block-decode", "batch": batch, "ctx": ctx, "tp": tp,
                "oracle_verified": false, "validation_errors": errors,
                "input_dir": inputs, "blob": blob, "hsaco": hsaco, "checkpoint": checkpoint
            }))?)?;
        }
        Err(errors.join("; ").into())
    };
    let run = |engines: &mut [&mut AmdEngine]| -> Result<(), Error> {
        for engine in engines {
            if engine.batch() != batch {
                return Err("capture batch must equal the packet's allocated decode batch".into());
            }
            if ctx as usize > engine.max_ctx() {
                return Err("context exceeds packet capacity".into());
            }
            engine.decode_prepare_batched(&positions, &kv_lengths)?;
            for (name, bytes) in &operands {
                if name.starts_with("kv.") {
                    let capacity = engine.tensor_bytes(name).ok_or("missing KV tensor")? as usize;
                    if capacity % (batch * engine.max_ctx()) != 0
                        || bytes.len() != capacity / engine.max_ctx() * ctx as usize
                    {
                        return Err(format!(
                            "{name}: capture must cover exactly {batch} rows of context {ctx}"
                        )
                        .into());
                    }
                    let stride = capacity / batch;
                    for (row, data) in bytes.chunks_exact(bytes.len() / batch).enumerate() {
                        engine.write_tensor_at(name, (row * stride) as u64, data)?;
                    }
                } else {
                    engine.write_tensor(name, bytes)?;
                }
            }
        }
        Ok(())
    };
    if tp == 1 {
        single = AmdEngine::load(backends[0].clone(), &blob, &hsaco, checkpoint.as_deref())?;
        for i in 0..iterations {
            run(&mut [&mut single])?;
            let start = Instant::now();
            let dp = single.decode_prog_for(batch);
            single.run(dp, single.decode_kernel_for(dp))?;
            let elapsed = start.elapsed();
            plowrt::obs::dstep::token(elapsed.as_nanos() as u64);
            let us = elapsed.as_secs_f64() * 1e6;
            if i == 0 {
                let mut bytes = vec![0; output_bytes];
                single.read_tensor(output, &mut bytes)?;
                let mut errors = Vec::new();
                if let Err(e) = check_stages(&single, 0) { errors.push(e.to_string()); }
                if let Err(e) = check_rows(&bytes, &reference, batch, tolerance) { errors.push(e.to_string()); }
                validation_failure(errors)?;
            }
            if i > warmup {
                samples.push(us);
            }
        }
        let mut bytes = vec![0; output_bytes];
        single.read_tensor(output, &mut bytes)?;
        outputs.push(bytes);
        check_stages(&single, 0)?;
        crate::trace_dump_1(&single, "")?;
    } else {
        group = AmdTpGroup::load(backends, &blob, &hsaco, checkpoint.as_deref())?;
        if !group.counter_audit_enabled() {
            return Err("block measurements require the TP counter audit".into());
        }
        for i in 0..iterations {
            for rank in 0..tp as usize {
                run(&mut [group.rank_mut(rank)])?;
            }
            let start = Instant::now();
            let dp = group.rank(0).decode_prog_for(batch);
            group.submit_decode_batched_at(&positions, &kv_lengths, dp)?;
            group.complete_block()?;
            let elapsed = start.elapsed();
            plowrt::obs::dstep::token(elapsed.as_nanos() as u64);
            let us = elapsed.as_secs_f64() * 1e6;
            if i == 0 {
                let mut errors = Vec::new();
                for rank in 0..tp as usize {
                    let mut bytes = vec![0; output_bytes];
                    group.rank(rank).read_tensor(output, &mut bytes)?;
                    if let Err(e) = check_stages(group.rank(rank), rank) {
                        errors.push(format!("rank{rank}: {e}"));
                    }
                    if let Err(e) = check_rows(&bytes, &reference, batch, tolerance) {
                        errors.push(format!("rank{rank}: {e}"));
                    }
                }
                validation_failure(errors)?;
            }
            if i > warmup {
                samples.push(us);
            }
        }
        for rank in 0..tp as usize {
            let mut bytes = vec![0; output_bytes];
            group.rank(rank).read_tensor(output, &mut bytes)?;
            outputs.push(bytes);
            check_stages(group.rank(rank), rank)?;
        }
        crate::trace_dump(&group)?;
    }
    let relative_l2 = outputs
        .iter()
        .map(|b| check_rows(b, &reference, batch, tolerance))
        .collect::<Result<Vec<_>, _>>()?;
    if outputs.iter().any(|b| b != &outputs[0]) {
        return Err("block residual differs across TP ranks".into());
    }
    if let Some(dir) = dump {
        std::fs::create_dir_all(&dir)?;
        for (rank, bytes) in outputs.iter().enumerate() {
            std::fs::write(dir.join(format!("rank{rank}.{output}.bin")), bytes)?;
        }
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let record = serde_json::json!({
        "scope": "single-block-decode", "clock": if tp == 1 { "host-dispatch-drain" } else { "host-prepare-dispatch-drain-audit" },
        "batch": batch, "ctx": ctx, "tp": tp, "warmup": warmup,
        "latency_us_median": sorted[sorted.len()/2], "samples_us": samples,
        "input_dir": inputs, "blob": blob, "hsaco": hsaco, "checkpoint": checkpoint,
        "finiteness": true, "rank_identity": true, "oracle_verified": true, "relative_l2": relative_l2,
        "relative_l2_reduction": "max over individual rows, per rank",
        "cache_policy": "operand-uploads-before-each-dispatch; no-L2-flush",
        "trace_instrumented": plowrt::config::RuntimeConfig::get().amd.trace_raw.is_some(),
        "host_timing_instrumented": plowrt::obs::dstep::on()
    });
    let json = serde_json::to_string_pretty(&record)?;
    if let Some(path) = report {
        std::fs::write(path, &json)?;
    }
    println!("{json}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_rejects_silent_zeros_nonfinite_and_shape_mismatch() {
        let one = 0x3f80u16.to_le_bytes();
        assert_eq!(check_output(&one, &one, 0.03).unwrap(), 0.0);
        assert!(check_output(&[0, 0], &one, 0.03).is_err());
        assert!(check_output(&0x7fc0u16.to_le_bytes(), &one, 0.03).is_err());
        assert!(check_output(&[], &one, 0.03).is_err());
        assert!(check_output(&[0, 0], &[0, 0], 0.03).is_err());
    }

    #[test]
    fn row_gate_rejects_error_hidden_by_batch_average() {
        let reference = vec![0x3f80u16; 64];
        let mut actual = reference.clone();
        actual[63] = 0x3f70;
        let bytes = |values: &[u16]| {
            values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let (actual, reference) = (bytes(&actual), bytes(&reference));
        assert!(check_output(&actual, &reference, 0.03).is_ok());
        assert!(check_rows(&actual, &reference, 64, 0.03)
            .unwrap_err()
            .to_string()
            .contains("row 63"));
        assert_eq!(check_rows(&reference, &reference, 64, 0.03).unwrap(), 0.0);
        assert!(check_rows(&actual, &reference, 0, 0.03).is_err());
    }
}
