//! Op-lockstep check of the Metal kernels against the CPU kernels on ONE loaded model (unified
//! memory: the Metal engine's tensors are the host tensors). For every instruction of a prefill
//! chunk and one decode step: run the CPU kernel with its outputs redirected to scratch, run the
//! same instruction on the GPU, compare. The GPU result stays as the state, so each op is judged
//! on the same inputs the CPU saw.
//!
//! `cargo run --release --features metal --example apple_lockstep -- <model.pkt> <ckpt> [--prompt "..."] [--chat "..."]`

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
    use plowrt::exec::apple::MetalEngine;
    use plowrt::exec::cpu::engine::plan_chunks;
    use plowrt::exec::cpu::ffi::{self, Isa, PlowCpuCtx};
    use plowrt::text::tokenizer::load_tokenizer;
    use std::ffi::c_void;
    use std::path::PathBuf;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args
        .next()
        .expect("usage: apple_lockstep <model.pkt> <ckpt>")
        .into();
    let ckpt: PathBuf = args
        .next()
        .expect("usage: apple_lockstep <model.pkt> <ckpt>")
        .into();
    let mut prompt = String::from("The capital of France is");
    let mut time_only = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--time" => time_only = true,
            "--prompt" => prompt = args.next().unwrap(),
            "--chat" => {
                let q = args.next().unwrap();
                prompt = format!(
                    "<bos><|turn>user\n{q}<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
                );
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let tok = load_tokenizer(&ckpt);
    let ids = tok.encode_with_special_tokens(&prompt, true);
    let mut gpu = MetalEngine::load(&blob, &ckpt).expect("metal load");
    let isa = ffi::init(Isa::Amx).expect("cpu kernels");
    println!("cpu tier {isa:?}; {} tokens", ids.len());
    let n = gpu.model.names.len();
    let mut ctx = PlowCpuCtx::new(0, 0);
    let scratch_bytes = ffi::scratch_bytes().max(64) as usize;
    let mut scratch = vec![0u64; scratch_bytes / 8 + 8];
    ctx.scratch = scratch.as_mut_ptr() as *mut c_void;
    ctx.scratch_bytes = scratch_bytes as u32;
    ffi::thread_init(&mut ctx).expect("thread init");

    let outs_of = |op: u16| -> Vec<usize> {
        match op {
            11 => vec![0, 1, 5],
            12 => vec![0, 1],
            22 => vec![0, 3, 5],
            21 | 23 => vec![0, 1],
            _ => vec![0],
        }
    };
    let bf = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect()
    };
    let f32s = |b: &[u8]| -> Vec<f32> {
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };

    // Base pointer table = the model's host tensors (shared with the GPU).
    let base: Vec<*mut c_void> = (0..n).map(|h| gpu.host_ptr(h) as *mut c_void).collect();

    let mut check_program = |gpu: &mut MetalEngine, p: usize, label: &str| {
        let insts: Vec<DevInst64> = gpu.insts_host(p).to_vec();
        println!("== {label}: program {p}, {} instructions", insts.len());
        if time_only {
            // GPU time per op class, one dispatch per instruction (includes per-dispatch overhead
            // of ~0.1 ms; a persistent run has none of that, so read this as relative weight).
            let mut by_op: std::collections::BTreeMap<u16, (usize, f64)> =
                std::collections::BTreeMap::new();
            let t_all = std::time::Instant::now();
            for (i, d) in insts.iter().enumerate() {
                let t = std::time::Instant::now();
                gpu.run_inst(p, i).expect("run_inst");
                let e = by_op.entry(d.op).or_default();
                e.0 += 1;
                e.1 += t.elapsed().as_secs_f64() * 1e3;
            }
            let total = t_all.elapsed().as_secs_f64() * 1e3;
            let mut rows: Vec<_> = by_op.into_iter().collect();
            rows.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
            println!("  {:<24} {:>6} {:>10} {:>7}", "op", "insts", "ms", "%");
            for (op, (n, ms)) in rows {
                let name = DevOp::from_u16(op).map(|o| o.c_name()).unwrap_or("?");
                println!(
                    "  {name:<24} {n:>6} {ms:>10.1} {:>6.1}%",
                    100.0 * ms / total
                );
            }
            println!("  total {total:.1} ms (serial dispatch)");
            return;
        }
        let mut bad_ops = 0;
        for (i, d) in insts.iter().enumerate() {
            let op = DevOp::from_u16(d.op).map(|o| o.c_name()).unwrap_or("?");
            let outs: Vec<usize> = outs_of(d.op)
                .into_iter()
                .filter(|&k| d.t[k] != TENSOR_NONE16)
                .collect();
            // CPU: outputs redirected to scratch copies (inputs stay the real tensors).
            let mut table = base.clone();
            let mut tmp: Vec<(usize, Vec<u8>)> = Vec::new();
            for &k in &outs {
                let h = d.t[k] as usize;
                let cur = gpu.tensor_bytes(h).to_vec(); // start from current contents (in-place ops)
                tmp.push((h, cur));
            }
            for (j, &k) in outs.iter().enumerate() {
                table[d.t[k] as usize] = tmp[j].1.as_mut_ptr() as *mut c_void;
            }
            if let Some(f) = ffi::kernel(d.op) {
                for s in 0..d.blocks as u32 {
                    // SAFETY: the kernel contract (slice of nblk, validated handles).
                    unsafe { f(d, s, d.blocks as u32, table.as_ptr(), &mut ctx) };
                }
            } else {
                println!("  inst {i:4} {op:<24}: no CPU kernel");
            }
            // GPU
            if let Err(e) = gpu.run_inst(p, i) {
                println!("  inst {i:4} {op:<24}: GPU error {e}");
                bad_ops += 1;
                continue;
            }
            for (j, &k) in outs.iter().enumerate() {
                let h = d.t[k] as usize;
                let g = gpu.tensor_bytes(h);
                let c = &tmp[j].1;
                let name = &gpu.model.names[h];
                let (vg, vc, kind) = match d.op {
                    11 | 12 if k < 2 => (f32s(g), f32s(c), "f32"),
                    17 | 18 => (Vec::new(), Vec::new(), "raw"),
                    _ => (bf(g), bf(c), "bf16"),
                };
                let (mut worst, mut wi, mut bad, mut nn) = (0f32, 0usize, 0usize, 0usize);
                if kind == "raw" {
                    let m = g.len().min(c.len());
                    bad = g[..m].iter().zip(&c[..m]).filter(|(x, y)| x != y).count();
                    nn = m;
                } else {
                    for (e, (x, y)) in vg.iter().zip(vc.iter()).enumerate() {
                        nn += 1;
                        if x.is_nan() && y.is_nan() {
                            continue;
                        }
                        let dif = (x - y).abs();
                        let tol = 2e-2 * y.abs() + 2e-2;
                        if !(dif <= tol) {
                            bad += 1;
                        }
                        if dif > worst || dif.is_nan() {
                            worst = dif;
                            wi = e;
                        }
                    }
                }
                if bad > 0 {
                    bad_ops += 1;
                    if bad_ops <= 25 {
                        println!(
                            "  inst {i:4} {op:<22} t{k} {name:<28} {kind}: {bad}/{nn} off, worst {worst:.4} at {wi} (cpu {:.4} gpu {:.4})",
                            vc.get(wi).copied().unwrap_or(0.0), vg.get(wi).copied().unwrap_or(0.0)
                        );
                    }
                }
            }
        }
        println!("== {label}: {bad_ops} mismatching outputs");
    };

    let buckets = gpu.prefill_buckets();
    let plan = plan_chunks(&buckets, ids.len() as u32);
    for ch in plan {
        gpu.prepare_prefill_chunk(&ids, ch).expect("prepare");
        check_program(
            &mut gpu,
            ch.prog,
            &format!("prefill chunk c0={} clen={}", ch.c0, ch.clen),
        );
    }
    let t_ids = gpu.model.wk.ids.expect("ids");
    let first = gpu.read_u32(t_ids, 0);
    println!("first token {first} {:?}", tok.decode(&[first]));
    let pos = ids.len() as u32;
    let dp = gpu
        .prepare_decode(pos, pos + 1, first)
        .expect("prepare decode");
    check_program(&mut gpu, dp, "decode step");
    let next = gpu.read_u32(t_ids, 0);
    println!("next token {next} {:?}", tok.decode(&[next]));
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("build with --features metal on macOS");
}
