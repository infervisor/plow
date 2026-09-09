//! Experimental N-axis GPU/ANE GEMM split, including host packing and concatenation.
//! Usage: <w8a16.pkt> <checkpoint> [--m 128,256] [--shares 0,25,50,75] [--reps 9]

#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    use objc2_core_ml::MLComputeUnits;
    use objc2_metal::MTLCommandBuffer;
    use plowrt::exec::ane::{f32_to_f16, AneNet, Layer, NetSpec};
    use plowrt::exec::apple::MetalEngine;
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("<w8a16.pkt> <checkpoint>").into();
    let checkpoint: PathBuf = args.next().expect("<checkpoint>").into();
    let mut rows = vec![128u32, 256];
    let mut shares = vec![0u32, 25, 50, 75];
    let mut reps = 9usize;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--m" => {
                rows = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--shares" => {
                shares = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.parse().unwrap())
                    .collect()
            }
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            _ => panic!("unknown argument {arg}"),
        }
    }
    assert!(reps > 0 && rows.iter().all(|&m| m > 0));
    assert!(shares.iter().all(|&s| s < 100));
    let mut gpu = MetalEngine::load(&blob, &checkpoint).expect("Metal load");
    let mut shapes = BTreeMap::new();
    for p in 0..gpu.model.dec_ix {
        for (i, d) in gpu.insts_host(p).iter().enumerate() {
            if matches!(d.op, 33 | 34 | 35) && d.i[1] == 3072 && matches!(d.i[2], 3072 | 8192) {
                assert_eq!(d.t[3], packet::dev::TENSOR_NONE16, "requires W8A16");
                shapes.entry((d.i[1], d.i[2])).or_insert((*d, p, i));
            }
        }
    }
    assert!(!shapes.is_empty(), "no requested W8A16 GEMMs in asset");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let cache = PathBuf::from("/private/tmp").join(format!(
        "plow-tensor-parallel-{}-{nonce}",
        std::process::id()
    ));
    std::fs::create_dir(&cache).expect("unique experimental cache");
    println!(
        "cache={} (retained for compute-plan inspection)",
        cache.display()
    );
    println!(
        "N K M ANE% GPU_N ANE_N min_ms median_ms gpu_ms pack_ms ane_ms join_ms rel_l2 max_abs bad"
    );
    let bf = |v: u16| f32::from_bits((v as u32) << 16);
    let to_bf = |v: f32| {
        let bits = v.to_bits();
        (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
    };
    let mut cache_weight_bytes = 0usize;
    for ((n, k), (original, program, instruction)) in shapes {
        let capacity = (gpu.tensor_bytes(original.t[0] as usize).len() / (2 * n as usize))
            .min(gpu.tensor_bytes(original.t[1] as usize).len() / (2 * k as usize));
        let valid_rows: Vec<u32> = rows
            .iter()
            .copied()
            .filter(|&m| m as usize <= capacity)
            .collect();
        for &m in &rows {
            if m as usize > capacity {
                eprintln!("skip M={m}, N={n}, K={k}: capacity={capacity}");
            }
        }
        if valid_rows.is_empty() {
            continue;
        }
        let wb = gpu.tensor_bytes(original.t[2] as usize);
        let scales = gpu.tensor_bytes(original.t[4] as usize);
        let w16: Vec<u16> = wb[..(n * k) as usize]
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let sign = if b & 128 != 0 { -1.0 } else { 1.0 };
                let exponent = (b >> 3) & 15;
                let mantissa = (b & 7) as f32;
                let value = if exponent == 0 {
                    mantissa / 512.0
                } else {
                    (1.0 + mantissa / 8.0) * 2f32.powi(exponent as i32 - 7)
                };
                let row = i / k as usize;
                let scale = f32::from_le_bytes(scales[4 * row..4 * row + 4].try_into().unwrap());
                f32_to_f16(sign * value * scale)
            })
            .collect();
        for m in valid_rows {
            let mut nets = BTreeMap::new();
            for &share in &shares {
                let na = n * share / 100 / 128 * 128;
                if na == 0 || nets.contains_key(&na) {
                    continue;
                }
                cache_weight_bytes += 2 * (na * k) as usize;
                assert!(
                    cache_weight_bytes < 256 * 1024 * 1024,
                    "bounded experimental weight budget"
                );
                let spec = NetSpec {
                    inputs: vec![("x".into(), k as usize)],
                    outputs: vec![("y".into(), na as usize)],
                    t_enum: vec![m as usize],
                    flex_outputs: false,
                    range: false,
                    out_range: false,
                    w8: false,
                    layers: vec![Layer::InnerProduct {
                        input: "x".into(),
                        output: "y".into(),
                        k: k as usize,
                        n: na as usize,
                        w_f16: w16[((n - na) * k) as usize..].to_vec(),
                    }],
                };
                let io = (spec.inputs.clone(), spec.outputs.clone());
                let name = format!("ip-m{m}-n{na}-k{k}");
                let net = AneNet::new(
                    &cache,
                    &name,
                    || spec,
                    io,
                    MLComputeUnits::CPUAndNeuralEngine,
                )
                .expect("ANE compile/load");
                eprintln!("compiled {name} in {:.1} ms", net.compile_ms);
                nets.insert(na, net);
            }
            let mut rng = 0x9e3779b9u32;
            let input_ptr = gpu.host_ptr(original.t[1] as usize) as *mut u16;
            // SAFETY: M is bounded by both tensor capacities and GPU work is not in flight.
            let input = unsafe { std::slice::from_raw_parts_mut(input_ptr, (m * k) as usize) };
            for x in input.iter_mut() {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                *x = to_bf(((rng >> 8) & 65535) as f32 / 32768.0 - 1.0);
            }
            let mut inst = original;
            inst.op = 34;
            inst.i[0] = m;
            inst.i[4] = 0;
            inst.i[5] = 0;
            inst.blocks = 16;
            gpu.insts_host_mut(program)[instruction] = inst;
            gpu.run_inst(program, instruction).expect("GPU reference");
            let reference =
                gpu.tensor_bytes(original.t[0] as usize)[..(2 * m * n) as usize].to_vec();
            let mut xin = vec![0f32; (m * k) as usize];
            let mut yane = vec![0f32; (m * n) as usize];
            let mut assembled = vec![0u16; (m * n) as usize];
            let mut times: Vec<Vec<[f64; 5]>> = vec![Vec::new(); shares.len()];
            let mut checks = vec![(0f64, 0f32, 0usize); shares.len()];
            for round in 0..reps + 2 {
                for order in 0..shares.len() {
                    let index = (round + order) % shares.len();
                    let na = n * shares[index] / 100 / 128 * 128;
                    let ng = n - na;
                    inst.i[1] = ng;
                    gpu.insts_host_mut(program)[instruction] = inst;
                    let start = Instant::now();
                    let cb = gpu
                        .run_inst_async(program, instruction)
                        .expect("GPU submit");
                    let pack_start = Instant::now();
                    if na > 0 {
                        for (dst, &src) in xin.iter_mut().zip(input.iter()) {
                            *dst = bf(src);
                        }
                    }
                    let pack_ms = pack_start.elapsed().as_secs_f64() * 1e3;
                    let ane_start = Instant::now();
                    if na > 0 {
                        nets.get_mut(&na)
                            .unwrap()
                            .run(m as usize, &[&xin], &mut [&mut yane[..(m * na) as usize]])
                            .expect("ANE predict");
                    }
                    let ane_ms = ane_start.elapsed().as_secs_f64() * 1e3;
                    cb.waitUntilCompleted();
                    let gpu_ms = (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3;
                    let join_start = Instant::now();
                    let packed = gpu.tensor_bytes(original.t[0] as usize);
                    if na > 0 {
                        for row in 0..m as usize {
                            for col in 0..ng as usize {
                                let off = 2 * (row * ng as usize + col);
                                assembled[row * n as usize + col] =
                                    u16::from_le_bytes([packed[off], packed[off + 1]]);
                            }
                            for col in 0..na as usize {
                                assembled[row * n as usize + ng as usize + col] =
                                    to_bf(yane[row * na as usize + col]);
                            }
                        }
                        // SAFETY: the GPU has joined; C has capacity for the complete result.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                assembled.as_ptr(),
                                gpu.host_ptr(original.t[0] as usize) as *mut u16,
                                assembled.len(),
                            );
                        }
                    }
                    let join_ms = join_start.elapsed().as_secs_f64() * 1e3;
                    let total = start.elapsed().as_secs_f64() * 1e3;
                    if round >= 2 {
                        times[index].push([total, gpu_ms, pack_ms, ane_ms, join_ms]);
                    }
                    if round == reps + 1 {
                        let published = gpu.tensor_bytes(original.t[0] as usize);
                        let (mut error, mut norm, mut worst, mut bad) = (0f64, 0f64, 0f32, 0usize);
                        for e in 0..(m * n) as usize {
                            let actual =
                                bf(u16::from_le_bytes([published[2 * e], published[2 * e + 1]]));
                            let expected =
                                bf(u16::from_le_bytes([reference[2 * e], reference[2 * e + 1]]));
                            let diff = (actual - expected).abs();
                            error += f64::from(diff).powi(2);
                            norm += f64::from(expected).powi(2);
                            worst = worst.max(diff);
                            if !actual.is_finite() || diff > 0.02 + 0.02 * expected.abs() {
                                bad += 1;
                            }
                        }
                        checks[index] = ((error / norm).sqrt(), worst, bad);
                        assert_eq!(bad, 0, "N-axis split differs from GPU reference");
                    }
                }
            }
            for (index, values) in times.iter_mut().enumerate() {
                values.sort_by(|a, b| a[0].total_cmp(&b[0]));
                let mid = values[values.len() / 2];
                let na = n * shares[index] / 100 / 128 * 128;
                let (l2, worst, bad) = checks[index];
                println!("{n} {k} {m} {} {} {na} {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} {l2:.6} {worst:.5} {bad}", shares[index], n - na, values[0][0], mid[0], mid[1], mid[2], mid[3], mid[4]);
            }
            gpu.insts_host_mut(program)[instruction] = original;
        }
    }
    println!("Compute units request CPUAndNeuralEngine; not proof of actual ANE placement.");
}

#[cfg(not(all(feature = "ane", target_os = "macos")))]
fn main() {
    eprintln!("build with --features ane on macOS");
}
