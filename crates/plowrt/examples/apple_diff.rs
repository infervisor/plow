//! Differential check of the Metal engine against the CPU engine: run the same prefill (and
//! one decode step) on both, then walk the program's instructions in order and compare each
//! op's output tensor. The first mismatching op is the one to look at.
//!
//! `cargo run --release --features metal --example apple_diff -- <model.pkt> <ckpt> [--prompt "..."] [--decode]`

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use packet::dev::{DevOp, TENSOR_NONE16};
    use plowrt::exec::apple::MetalEngine;
    use plowrt::exec::cpu::engine::{CpuEngine, CpuEngineOpts};
    use plowrt::text::tokenizer::load_tokenizer;
    use std::path::PathBuf;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args
        .next()
        .expect("usage: apple_diff <model.pkt> <ckpt>")
        .into();
    let ckpt: PathBuf = args
        .next()
        .expect("usage: apple_diff <model.pkt> <ckpt>")
        .into();
    let mut prompt = String::from("The capital of France is");
    let mut decode = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--prompt" => prompt = args.next().unwrap(),
            "--decode" => decode = true,
            other => panic!("unknown arg {other}"),
        }
    }
    let tok = load_tokenizer(&ckpt);
    let ids = tok.encode_with_special_tokens(&prompt, true);
    let mut opts = CpuEngineOpts::default();
    opts.threads = 8;
    let mut cpu = CpuEngine::load(&blob, &ckpt, &opts).expect("cpu load");
    let mut gpu = MetalEngine::load(&blob, &ckpt).expect("metal load");
    let tc = cpu.prefill(&ids).expect("cpu prefill");
    let tg = gpu.prefill(&ids).expect("gpu prefill");
    println!("prefill token: cpu {tc} gpu {tg}");
    let mut prog = gpu
        .model
        .blob
        .progs
        .iter()
        .position(|p| p.t as usize >= ids.len())
        .unwrap_or(0);
    if decode {
        let pos = ids.len() as u32;
        gpu.set_token(tc).unwrap();
        let dc = cpu.decode_step(pos, pos + 1).expect("cpu decode");
        let dg = gpu.decode_step(pos, pos + 1).expect("gpu decode");
        println!("decode token: cpu {dc} gpu {dg}");
        prog = gpu.model.dec_ix;
    }
    let p = &gpu.model.blob.progs[prog];
    println!(
        "comparing program {prog} (T={}), {} instructions",
        p.t,
        p.insts.len()
    );
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
    let mut shown = 0;
    let mut seen = std::collections::HashSet::new();
    for (ii, d) in p.insts.iter().enumerate() {
        let op = DevOp::from_u16(d.op).map(|o| o.c_name()).unwrap_or("?");
        let outs: Vec<usize> = match d.op {
            11 | 12 => vec![0, 1],
            22 => vec![0, 3, 5],
            _ => vec![0],
        };
        for k in outs {
            let h = d.t[k];
            if h == TENSOR_NONE16 || !seen.insert((h, ii)) {
                continue;
            }
            let h = h as usize;
            let a = unsafe { cpu.model().tensor(h).as_slice() };
            let b = gpu.tensor_bytes(h);
            let name = &gpu.model.names[h];
            let (va, vb, kind) = match d.op {
                11 | 12 => (f32s(a), f32s(b), "f32"),
                17 | 18 => (Vec::new(), Vec::new(), "raw"),
                _ => (bf(a), bf(b), "bf16"),
            };
            let (mut worst, mut wi, mut bad, mut n) = (0f32, 0usize, 0usize, 0usize);
            if kind == "raw" {
                let m = a.len().min(b.len());
                bad = a[..m].iter().zip(&b[..m]).filter(|(x, y)| x != y).count();
                n = m;
            } else {
                for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
                    n += 1;
                    if x.is_nan() && y.is_nan() {
                        continue;
                    }
                    let dif = (x - y).abs();
                    let tol = 1e-2 * x.abs() + 1e-2;
                    if !(dif <= tol) {
                        bad += 1;
                    }
                    if dif > worst || dif.is_nan() {
                        worst = dif;
                        wi = i;
                    }
                }
            }
            if bad > 0 && shown < 12 {
                shown += 1;
                println!(
                    "  inst {ii:4} {op:<24} t{k} {name:<40} {kind}: {bad}/{n} off, worst {worst:.4} at {wi} (cpu {:.4} gpu {:.4})",
                    va.get(wi).copied().unwrap_or(0.0), vb.get(wi).copied().unwrap_or(0.0)
                );
            }
        }
    }
    if shown == 0 {
        println!("all compared outputs match within 1e-2");
    }
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("build with --features metal on macOS");
}
