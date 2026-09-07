//! Metal prefill GEMM microbenchmark + golden check on REAL tensors of a loaded blob: for each
//! (op, M, N, K) cell an instruction is crafted over the model's weights/activations, dispatched
//! alone (`run_inst`, 16 threadgroups like the emitted stream), timed (min of reps) and compared
//! against the CPU tier's kernel on the same operands.
//!
//! `cargo run --release --features ane --example metal_gemm_bench -- <model.pkt> <ckpt> [--reps 5] [--check] [--m 25,64,128,512]`
//!
//! Cells come from the blob itself (every distinct GEMM-family instruction shape of the prefill
//! programs, at the requested M values), so the table is the model's real prefill mix.

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
    use plowrt::exec::apple::MetalEngine;
    use plowrt::exec::cpu::ffi::{self, Isa, PlowCpuCtx};
    use std::collections::BTreeMap;
    use std::ffi::c_void;
    use std::path::PathBuf;
    use std::time::Instant;

    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage: <model.pkt> <ckpt>").into();
    let ckpt: PathBuf = args.next().expect("usage").into();
    let mut reps = 5usize;
    let mut check = false;
    let mut all_tiles = false;
    let mut ms_list: Vec<u32> = vec![25, 64, 128, 512];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--check" => check = true,
            "--all-tiles" => all_tiles = true,
            "--m" => ms_list = args.next().unwrap().split(',').map(|v| v.parse().unwrap()).collect(),
            other => panic!("unknown arg {other}"),
        }
    }
    let mut gpu = MetalEngine::load(&blob, &ckpt).expect("metal load");
    let isa = ffi::init(Isa::Amx).expect("cpu kernels");
    let n = gpu.model.names.len();
    let base: Vec<*mut c_void> = (0..n).map(|h| gpu.host_ptr(h) as *mut c_void).collect();

    // Distinct GEMM-family instructions of the prefill programs, keyed by (op, N, K, operand handles).
    let gemm_ops = [8u16, 14, 15, 20, 33, 34, 35, 36];
    let mut cells: BTreeMap<(u16, u32, u32, [u16; 8]), (usize, usize)> = BTreeMap::new();
    for p in 0..gpu.model.dec_ix {
        for (i, d) in gpu.insts_host(p).iter().enumerate() {
            if gemm_ops.contains(&d.op) {
                cells.entry((d.op, d.i[1], d.i[2], d.t)).or_insert((p, i));
            }
        }
    }
    // One representative per (op, N, K): the first layer's.
    let mut seen = std::collections::HashSet::new();
    let mut plan: Vec<(DevInst64, usize, usize)> = Vec::new();
    for ((op, nn, k, _), (p, i)) in &cells {
        if seen.insert((*op, *nn, *k)) {
            plan.push((gpu.insts_host(*p)[*i], *p, *i));
        }
    }
    // Random A operand (bf16 in [-1, 1]) so the check is not on zeros.
    let mut rng = 0x9E3779B9u32;
    let mut fill = |h: usize| {
        let bytes = gpu.tensor_bytes(h).len();
        let p = gpu.host_ptr(h) as *mut u16;
        for i in 0..bytes / 2 {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((rng >> 8) & 0xFFFF) as f32 / 32768.0 - 1.0;
            // SAFETY: in-bounds bf16 store into an activation tensor nothing else touches.
            unsafe { *p.add(i) = (v.to_bits() >> 16) as u16 };
        }
    };
    let mut filled = std::collections::HashSet::new();
    for (d, _, _) in &plan {
        let a = d.t[1] as usize;
        if filled.insert(a) {
            fill(a);
        }
    }
    println!(
        "cpu tier {isa:?}; {} GEMM shapes; M in {ms_list:?}; reps {reps}{}",
        plan.len(),
        if check { "; checking against the CPU kernel" } else { "" }
    );
    println!(
        "{:<22} {:>5} {:>6} {:>5} {:>9} {:>8} {:>8}  {}",
        "op", "M", "N", "K", "ms", "GFLOPS", "GB/s(W)", "check"
    );
    let mut ctx = PlowCpuCtx::new(0, 0);
    let scratch_bytes = ffi::scratch_bytes().max(64) as usize;
    let mut scratch = vec![0u64; scratch_bytes / 8 + 8];
    ctx.scratch = scratch.as_mut_ptr() as *mut c_void;
    ctx.scratch_bytes = scratch_bytes as u32;
    ffi::thread_init(&mut ctx).expect("thread init");
    // Dispatch floor: the first GEMM shape at K = 16 (one staging chunk) is almost pure launch
    // + walk overhead; every cell below includes it.
    if let Some((d0, p, i)) = plan.first().copied() {
        let mut d = d0;
        d.i[0] = 32;
        d.i[2] = 16;
        d.blocks = 16;
        gpu.insts_host_mut(p)[i] = d;
        let mut best = f64::INFINITY;
        for _ in 0..reps.max(10) {
            let t = Instant::now();
            gpu.run_inst(p, i).expect("run_inst");
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        gpu.insts_host_mut(p)[i] = d0;
        println!("dispatch floor (32 x N x 16 GEMM): {best:.3} ms");
        // A near-empty elementwise op for the pure launch cost.
        if let Some(ri) = gpu.insts_host(p).iter().position(|d| d.op == 4) {
            let r0 = gpu.insts_host(p)[ri];
            let mut r = r0;
            r.i[0] = 64;
            r.blocks = 16;
            gpu.insts_host_mut(p)[ri] = r;
            let mut best = f64::INFINITY;
            for _ in 0..20 {
                let t = Instant::now();
                gpu.run_inst(p, ri).expect("run_inst");
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            gpu.insts_host_mut(p)[ri] = r0;
            println!("launch floor (64-element RESIDUAL, 16 threadgroups): {best:.3} ms");
        }
    }
    let mut total_ms = 0.0;
    // `--all-tiles`: every plain GEMM shape also with the other two tile ops of its precision.
    let plan: Vec<(DevInst64, usize, usize)> = if all_tiles {
        plan.iter()
            .flat_map(|&(d, p, i)| {
                let alts: &[u16] = match d.op {
                    33 | 34 | 35 => &[33, 34, 35],
                    8 | 14 | 15 => &[8, 15, 14],
                    _ => &[],
                };
                if alts.is_empty() {
                    vec![(d, p, i)]
                } else {
                    alts.iter().map(|&op| { let mut e = d; e.op = op; (e, p, i) }).collect()
                }
            })
            .collect()
    } else {
        plan
    };
    for (d0, p, i) in plan {
        let name = DevOp::from_u16(d0.op).map(|o| o.c_name()).unwrap_or("?");
        let (nn, k) = (d0.i[1], d0.i[2]);
        let fp8 = matches!(d0.op, 33 | 34 | 35 | 36);
        let glu = matches!(d0.op, 20 | 36);
        // Rows the operands can hold (the lm_head writes a 1-row logits tensor).
        let cap = (gpu.tensor_bytes(d0.t[0] as usize).len() as u32 / (nn * 2))
            .min(gpu.tensor_bytes(d0.t[1] as usize).len() as u32 / (k * 2));
        let ks: Vec<u32> = std::env::var("PLOW_BENCH_KS")
            .ok()
            .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
            .unwrap_or_else(|| vec![k]);
        for &m in &ms_list {
            if m > cap {
                continue;
            }
            for &kk in &ks {
            let k = kk;
            let nn = std::env::var("PLOW_BENCH_N").ok().and_then(|v| v.parse().ok()).unwrap_or(nn);
            let mut d = d0;
            d.i[0] = m;
            d.i[1] = nn;
            d.i[2] = k;
            d.i[4] = 0;
            d.i[5] = 0;
            d.blocks = 16;
            gpu.insts_host_mut(p)[i] = d;
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t = Instant::now();
                gpu.run_inst(p, i).expect("run_inst");
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            total_ms += best;
            let flops = 2.0 * m as f64 * nn as f64 * k as f64 * if glu { 2.0 } else { 1.0 };
            let wbytes = nn as f64 * k as f64 * if fp8 { 1.0 } else { 2.0 } * if glu { 2.0 } else { 1.0 };
            let mut verdict = String::new();
            if check {
                // CPU kernel with its output redirected to a scratch copy, then compare row block [0, m).
                let c = d.t[0] as usize;
                let mut table = base.clone();
                let cur = gpu.tensor_bytes(c).to_vec();
                let mut tmp = cur.clone();
                table[c] = tmp.as_mut_ptr() as *mut c_void;
                let f = ffi::kernel(d.op).expect("cpu kernel");
                let nblk = 8u32;
                for s in 0..nblk {
                    // SAFETY: the kernel contract (validated handles, slice of nblk).
                    unsafe { f(&d, s, nblk, table.as_ptr(), &mut ctx) };
                }
                gpu.run_inst(p, i).expect("run_inst");
                let g = gpu.tensor_bytes(c);
                let (mut bad, mut worst, mut cnt) = (0usize, 0f32, 0usize);
                for e in 0..(m * nn) as usize {
                    let x = f32::from_bits((u16::from_le_bytes([g[2 * e], g[2 * e + 1]]) as u32) << 16);
                    let y = f32::from_bits((u16::from_le_bytes([tmp[2 * e], tmp[2 * e + 1]]) as u32) << 16);
                    cnt += 1;
                    let dif = (x - y).abs();
                    if !(dif <= 2e-2 * y.abs() + 2e-2) {
                        bad += 1;
                    }
                    if dif > worst || dif.is_nan() {
                        worst = dif;
                    }
                }
                verdict = if bad == 0 {
                    format!("ok (worst {worst:.4})")
                } else {
                    format!("{bad}/{cnt} OFF, worst {worst:.4}")
                };
            }
            println!(
                "{:<22} {:>5} {:>6} {:>5} {:>9.3} {:>8.0} {:>8.0}  {}",
                name.trim_start_matches("PLOW_DOP_"),
                m,
                nn,
                k,
                best,
                flops / best / 1e6,
                wbytes / best / 1e6,
                verdict
            );
            let _ = TENSOR_NONE16;
            }
        }
        gpu.insts_host_mut(p)[i] = d0;
    }
    println!("sum of best times {total_ms:.1} ms");
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("build with --features metal on macOS");
}
