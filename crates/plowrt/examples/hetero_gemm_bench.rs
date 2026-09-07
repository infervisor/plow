//! One GEMM, three units: `C[M][N] = A[M][K] . W^T` with the rows of `M` split into a GPU block
//! (Metal, fp8 weights), an ANE block (a CoreML innerProduct program over the same weights,
//! fp16) and a CPU block (the NEON fp8 GEMM kernel), all three running CONCURRENTLY on the
//! unified-memory tensors of a loaded blob. Every split is checked against the GPU-only result
//! and reported against per-unit rooflines, over the model's real (N, K) shapes and a list of M.
//!
//! `cargo run --release --features ane --example hetero_gemm_bench -- <model.pkt> <ckpt>
//!     [--m 128,512] [--splits gpu100,ane100,cpu100,g50a50,g60a40,g45a45c10,g80c20] [--reps 5]`
//!
//! Rooflines (M4 Pro, from the microbenchmarks in this tree): GPU 4.1 TFLOPS matrix peak and
//! 273 GB/s bus; ANE 4.0 TFLOPS with a ~0.3 ms program call; CPU (NEON, 8 P-cores) 0.5 TFLOPS.
//! A unit's floor for its rows is max(FLOP / peak, weight bytes / bus); the split's floor is the
//! slowest unit's floor (they run in parallel); "ideal" is the whole GEMM at the summed peaks.

#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    use objc2_core_ml::MLComputeUnits;
    use objc2_metal::MTLCommandBuffer;
    use packet::dev::DevInst64;
    use plowrt::exec::ane::{f32_to_f16, AneNet, Layer, NetSpec};
    use plowrt::exec::apple::MetalEngine;
    use plowrt::exec::cpu::ffi::{self, Isa, PlowCpuCtx};
    use std::collections::{BTreeMap, HashMap};
    use std::ffi::c_void;
    use std::path::PathBuf;
    use std::time::Instant;

    const GPU_TFLOPS: f64 = 4.1;
    const ANE_TFLOPS: f64 = 4.0;
    const ANE_CALL_MS: f64 = 0.3;
    const CPU_TFLOPS: f64 = 0.5;
    const BUS_GBS: f64 = 273.0;

    let mut args = std::env::args().skip(1);
    let blob: PathBuf = args.next().expect("usage: <model.pkt> <ckpt>").into();
    let ckpt: PathBuf = args.next().expect("usage").into();
    let mut ms_list: Vec<u32> = vec![128, 512];
    let mut splits: Vec<(u32, u32, u32)> = vec![
        (100, 0, 0),
        (0, 100, 0),
        (0, 0, 100),
        (50, 50, 0),
        (60, 40, 0),
        (45, 45, 10),
        (80, 0, 20),
    ];
    let mut reps = 5usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--m" => ms_list = args.next().unwrap().split(',').map(|v| v.parse().unwrap()).collect(),
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--splits" => {
                splits = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| {
                        let mut g = 0;
                        let mut a = 0;
                        let mut c = 0;
                        let mut cur = ' ';
                        let mut num = String::new();
                        for ch in s.chars().chain(std::iter::once(' ')) {
                            if ch.is_ascii_digit() {
                                num.push(ch);
                            } else {
                                if !num.is_empty() {
                                    let v: u32 = num.parse().unwrap();
                                    match cur {
                                        'g' => g = v,
                                        'a' => a = v,
                                        'c' => c = v,
                                        _ => {}
                                    }
                                    num.clear();
                                }
                                cur = ch;
                                if s.starts_with("gpu") { cur = 'g'; }
                                if s.starts_with("ane") { cur = 'a'; }
                                if s.starts_with("cpu") { cur = 'c'; }
                            }
                        }
                        (g, a, c)
                    })
                    .collect()
            }
            other => panic!("unknown arg {other}"),
        }
    }
    let mut gpu = MetalEngine::load(&blob, &ckpt).expect("metal load");
    ffi::init(Isa::Amx).expect("cpu kernels");
    let n_t = gpu.model.names.len();
    let base: Vec<*mut c_void> = (0..n_t).map(|h| gpu.host_ptr(h) as *mut c_void).collect();

    // Plain fp8 GEMM shapes of the prefill programs (one per (N, K)).
    let mut shapes: BTreeMap<(u32, u32), (DevInst64, usize, usize)> = BTreeMap::new();
    for p in 0..gpu.model.dec_ix {
        for (i, d) in gpu.insts_host(p).iter().enumerate() {
            if matches!(d.op, 33 | 34 | 35) {
                shapes.entry((d.i[1], d.i[2])).or_insert((*d, p, i));
            }
        }
    }
    // Random A (bf16 in [-1, 1]).
    let mut rng = 0x9E3779B9u32;
    let mut filled = std::collections::HashSet::new();
    for (d, _, _) in shapes.values() {
        let h = d.t[1] as usize;
        if !filled.insert(h) {
            continue;
        }
        let bytes = gpu.tensor_bytes(h).len();
        let p = gpu.host_ptr(h) as *mut u16;
        for i in 0..bytes / 2 {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let v = ((rng >> 8) & 0xFFFF) as f32 / 32768.0 - 1.0;
            // SAFETY: in-bounds bf16 store into an activation tensor.
            unsafe { *p.add(i) = (v.to_bits() >> 16) as u16 };
        }
    }
    let lut: Vec<f32> = (0..256)
        .map(|b: usize| {
            let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
            let e = ((b >> 3) & 0xf) as i32;
            let m = (b & 7) as f32;
            if e == 0 {
                s * m / 8.0 * 2f32.powi(-6)
            } else if e == 15 && m == 7.0 {
                f32::NAN
            } else {
                s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
            }
        })
        .collect();
    let ane_dir = std::env::temp_dir().join("plow-hetero-gemm-bench");
    let _ = std::fs::remove_dir_all(&ane_dir);
    let mut nets: HashMap<(u32, u32, u32), AneNet> = HashMap::new();
    let threads = 8usize;
    let round8 = |v: u32| v / 8 * 8;
    println!(
        "{:<11} {:>4} {:>12} {:>8} {:>7} {:>7} {:>7} {:>8} {:>8} {:>6}  {}",
        "shape N.K", "M", "split g/a/c", "total ms", "gpu ms", "ane ms", "cpu ms", "floor ms", "ideal ms", "eff%", "check"
    );
    for ((nn, k), (d0, p, i)) in &shapes {
        let (nn, k) = (*nn, *k);
        let cap = (gpu.tensor_bytes(d0.t[0] as usize).len() as u32 / (nn * 2))
            .min(gpu.tensor_bytes(d0.t[1] as usize).len() as u32 / (k * 2));
        // fp16 weights with the per-channel scale folded, for the ANE programs of this shape.
        let wb = gpu.tensor_bytes(d0.t[2] as usize);
        let sb = gpu.tensor_bytes(d0.t[4] as usize);
        let w16: Vec<u16> = (0..(nn * k) as usize)
            .map(|e| {
                let n = e / k as usize;
                let s = f32::from_le_bytes([sb[4 * n], sb[4 * n + 1], sb[4 * n + 2], sb[4 * n + 3]]);
                f32_to_f16(lut[wb[e] as usize] * s)
            })
            .collect();
        for &m in &ms_list {
            if m > cap {
                continue;
            }
            let flops = 2.0 * m as f64 * nn as f64 * k as f64;
            let wbytes = nn as f64 * k as f64;
            // GPU-only reference output (bf16 rows [0, m)).
            let mut dref = *d0;
            dref.op = 34;
            dref.i[0] = m;
            dref.i[4] = 0;
            dref.i[5] = 0;
            dref.blocks = 16;
            gpu.insts_host_mut(*p)[*i] = dref;
            gpu.run_inst(*p, *i).expect("ref");
            let cref = gpu.tensor_bytes(d0.t[0] as usize)[..(m * nn * 2) as usize].to_vec();
            for &(pg, pa, pc) in &splits {
                let (mut ra, mut rc) = (round8(m * pa / 100), round8(m * pc / 100));
                if pg == 0 && pa + pc > 0 {
                    // No GPU rows: give the rounding remainder to the first non-GPU unit.
                    if pa > 0 { ra = m - rc; } else { rc = m; }
                }
                let rg = m - ra - rc;
                let (row_a, row_c) = (rg, rg + ra);
                // ANE program for (N, K, ra) rows.
                if ra > 0 && !nets.contains_key(&(nn, k, ra)) {
                    let spec = NetSpec {
                        inputs: vec![("x".into(), k as usize)],
                        outputs: vec![("y".into(), nn as usize)],
                        t_enum: vec![ra as usize],
                        flex_outputs: false,
                        range: false,
                        out_range: false,
                        layers: vec![Layer::InnerProduct {
                            input: "x".into(),
                            output: "y".into(),
                            k: k as usize,
                            n: nn as usize,
                            w_f16: w16.clone(),
                        }],
                    };
                    let io = (spec.inputs.clone(), spec.outputs.clone());
                    let net = AneNet::new(&ane_dir, &format!("ip-{nn}x{k}-t{ra}"), move || spec, io, MLComputeUnits::CPUAndNeuralEngine)
                        .expect("ane program");
                    nets.insert((nn, k, ra), net);
                }
                // Clear the output rows so a unit that did nothing is caught by the check.
                {
                    let cp = gpu.host_ptr(d0.t[0] as usize) as *mut u16;
                    // SAFETY: rows [0, m) of C.
                    unsafe { std::ptr::write_bytes(cp, 0, (m * nn) as usize) };
                }
                let mut best = f64::INFINITY;
                let (mut t_gpu, mut t_ane, mut t_cpu) = (0.0f64, 0.0f64, 0.0f64);
                let mut xin = vec![0f32; (ra * k) as usize];
                let mut yout = vec![0f32; (ra * nn) as usize];
                for _ in 0..reps {
                    let t0 = Instant::now();
                    // GPU rows [0, rg): async command buffer.
                    let cb = if rg > 0 {
                        let mut d = dref;
                        d.i[0] = rg;
                        gpu.insts_host_mut(*p)[*i] = d;
                        Some(gpu.run_inst_async(*p, *i).expect("gpu"))
                    } else {
                        None
                    };
                    let tg0 = Instant::now();
                    // CPU rows [row_c, m): NEON fp8 GEMM on `threads` scoped threads.
                    let tc = Instant::now();
                    let cpu_ms = if rc > 0 {
                        let mut d = dref;
                        d.op = 33;
                        d.i[0] = rc;
                        let mut table = base.clone();
                        let c_h = d.t[0] as usize;
                        let a_h = d.t[1] as usize;
                        // SAFETY: row offsets inside the tensors.
                        unsafe {
                            table[c_h] = (table[c_h] as *mut u8).add((row_c * nn * 2) as usize) as *mut c_void;
                            table[a_h] = (table[a_h] as *mut u8).add((row_c * k * 2) as usize) as *mut c_void;
                        }
                        let f = ffi::kernel(33).expect("cpu fp8 gemm");
                        let tp = table.as_ptr() as usize;
                        let scratch_bytes = ffi::scratch_bytes().max(64) as usize;
                        std::thread::scope(|sc| {
                            for w in 0..threads {
                                sc.spawn(move || {
                                    let mut ctx = PlowCpuCtx::new(w as u32, 0);
                                    let mut scratch = vec![0u64; scratch_bytes / 8 + 8];
                                    ctx.scratch = scratch.as_mut_ptr() as *mut c_void;
                                    ctx.scratch_bytes = scratch_bytes as u32;
                                    let _ = ffi::thread_init(&mut ctx);
                                    // SAFETY: disjoint tiles of one op on rows no other unit writes.
                                    unsafe { f(&d, w as u32, threads as u32, tp as *const *mut c_void, &mut ctx) };
                                });
                            }
                            // ANE rows [row_a, row_c) on this thread while the CPU threads run.
                            if ra > 0 {
                                let ta = Instant::now();
                                let ap = gpu.host_ptr(d0.t[1] as usize) as *const u16;
                                // SAFETY: rows [row_a, row_a+ra) of A.
                                unsafe {
                                    for e in 0..(ra * k) as usize {
                                        xin[e] = f32::from_bits((*ap.add((row_a * k) as usize + e) as u32) << 16);
                                    }
                                }
                                let net = nets.get_mut(&(nn, k, ra)).unwrap();
                                net.run(ra as usize, &[&xin], &mut [&mut yout]).expect("ane run");
                                let cp = gpu.host_ptr(d0.t[0] as usize) as *mut u16;
                                // SAFETY: rows [row_a, row_a+ra) of C.
                                unsafe {
                                    for e in 0..(ra * nn) as usize {
                                        let v = yout[e];
                                        let x = v.to_bits();
                                        let r = 0x7fff + ((x >> 16) & 1);
                                        *cp.add((row_a * nn) as usize + e) = (x.wrapping_add(r) >> 16) as u16;
                                    }
                                }
                                t_ane = ta.elapsed().as_secs_f64() * 1e3;
                            }
                        });
                        tc.elapsed().as_secs_f64() * 1e3
                    } else {
                        if ra > 0 {
                            let ta = Instant::now();
                            let ap = gpu.host_ptr(d0.t[1] as usize) as *const u16;
                            // SAFETY: as above.
                            unsafe {
                                for e in 0..(ra * k) as usize {
                                    xin[e] = f32::from_bits((*ap.add((row_a * k) as usize + e) as u32) << 16);
                                }
                            }
                            let net = nets.get_mut(&(nn, k, ra)).unwrap();
                            net.run(ra as usize, &[&xin], &mut [&mut yout]).expect("ane run");
                            let cp = gpu.host_ptr(d0.t[0] as usize) as *mut u16;
                            // SAFETY: as above.
                            unsafe {
                                for e in 0..(ra * nn) as usize {
                                    let x = yout[e].to_bits();
                                    let r = 0x7fff + ((x >> 16) & 1);
                                    *cp.add((row_a * nn) as usize + e) = (x.wrapping_add(r) >> 16) as u16;
                                }
                            }
                            t_ane = ta.elapsed().as_secs_f64() * 1e3;
                        }
                        0.0
                    };
                    if let Some(cb) = cb {
                        cb.waitUntilCompleted();
                    }
                    t_gpu = if rg > 0 { tg0.elapsed().as_secs_f64() * 1e3 } else { 0.0 };
                    t_cpu = cpu_ms;
                    best = best.min(t0.elapsed().as_secs_f64() * 1e3);
                }
                // Check against the GPU-only reference.
                let cnow = gpu.tensor_bytes(d0.t[0] as usize);
                let (mut bad, mut worst) = (0usize, 0f32);
                for e in 0..(m * nn) as usize {
                    let x = f32::from_bits((u16::from_le_bytes([cnow[2 * e], cnow[2 * e + 1]]) as u32) << 16);
                    let y = f32::from_bits((u16::from_le_bytes([cref[2 * e], cref[2 * e + 1]]) as u32) << 16);
                    let dif = (x - y).abs();
                    if !(dif <= 2e-2 * y.abs() + 2e-2) {
                        bad += 1;
                    }
                    if dif > worst || dif.is_nan() {
                        worst = dif;
                    }
                }
                let floor_unit = |rows: u32, peak: f64, fixed: f64| -> f64 {
                    if rows == 0 {
                        0.0
                    } else {
                        let f = 2.0 * rows as f64 * nn as f64 * k as f64;
                        let compute = f / (peak * 1e9);
                        let bw = wbytes / (BUS_GBS * 1e6);
                        compute.max(bw) + fixed
                    }
                };
                let floor = floor_unit(rg, GPU_TFLOPS, 0.0)
                    .max(floor_unit(ra, ANE_TFLOPS, ANE_CALL_MS))
                    .max(floor_unit(rc, CPU_TFLOPS, 0.0));
                let ideal = flops / ((GPU_TFLOPS + ANE_TFLOPS + CPU_TFLOPS) * 1e9);
                println!(
                    "{:<11} {:>4} {:>12} {:>8.3} {:>7.2} {:>7.2} {:>7.2} {:>8.3} {:>8.3} {:>5.0}%  {}",
                    format!("{nn}.{k}"),
                    m,
                    format!("{pg}/{pa}/{pc}"),
                    best,
                    t_gpu,
                    t_ane,
                    t_cpu,
                    floor,
                    ideal,
                    100.0 * floor / best,
                    if bad == 0 { format!("ok {worst:.3}") } else { format!("{bad} OFF {worst:.3}") }
                );
            }
            gpu.insts_host_mut(*p)[*i] = *d0;
        }
    }
    let _ = std::fs::remove_dir_all(&ane_dir);
}

#[cfg(not(all(feature = "ane", target_os = "macos")))]
fn main() {
    eprintln!("build with --features ane on macOS");
}
