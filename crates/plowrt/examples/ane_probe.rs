//! Rung 0 probe (f): a dense GEMM on the Apple Neural Engine through the plow ANE executor.
//! Builds the CoreML program from random fp16 weights, checks it against a CPU reference, and
//! times it under CPU-only versus CPU+ANE compute units (the ANE has no direct timer; a large
//! speed-up over CPU-only is the evidence it ran there).
//!
//! `cargo run --release --features ane --example ane_probe -- [T] [K] [N]`

#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    use objc2_core_ml::MLComputeUnits;
    use plowrt::exec::ane::{f32_to_f16, AneGemm};
    use std::time::Instant;

    let mut args = std::env::args().skip(1);
    let t: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(128);
    let k: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(3072);
    let n: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(8192);
    let mut rng = 0x9E3779B9u32;
    let mut frand = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 8) & 0xFFFFFF) as f32 / 8388608.0 - 1.0
    };
    let w: Vec<f32> = (0..n * k).map(|_| frand() * 0.05).collect();
    let w16: Vec<u16> = w.iter().map(|&v| f32_to_f16(v)).collect();
    let wq: Vec<f32> = w16.iter().map(|&h| half_to_f32(h)).collect();
    let x: Vec<f32> = (0..t * k).map(|_| frand()).collect();
    let mut y = vec![0f32; t * n];
    // Reference with the fp16-rounded weights (what the ANE holds) and fp16-rounded x.
    let xq: Vec<f32> = x.iter().map(|&v| half_to_f32(f32_to_f16(v))).collect();
    let t0 = Instant::now();
    let mut yref = vec![0f32; t * n];
    for i in 0..t {
        for j in 0..n {
            let mut acc = 0f32;
            let xr = &xq[i * k..(i + 1) * k];
            let wr = &wq[j * k..(j + 1) * k];
            for kk in 0..k {
                acc += xr[kk] * wr[kk];
            }
            yref[i * n + j] = acc;
        }
    }
    println!(
        "reference: {:.1} ms (scalar)",
        t0.elapsed().as_secs_f64() * 1e3
    );
    let dir = std::env::temp_dir().join("plow-ane-probe");
    let _ = std::fs::remove_dir_all(&dir);
    for (label, units) in [
        ("cpu-only", MLComputeUnits::CPUOnly),
        ("cpu+ane", MLComputeUnits::CPUAndNeuralEngine),
    ] {
        let mut g = AneGemm::new(&dir, &format!("gemm_{t}x{k}x{n}"), t, k, n, &w16, units)
            .expect("ane gemm");
        g.run(&x, &mut y).expect("warm");
        let mut best = f64::INFINITY;
        for _ in 0..8 {
            g.run(&x, &mut y).expect("run");
            best = best.min(g.last_ms);
        }
        let (mut worst, mut bad) = (0f32, 0usize);
        for (a, b) in y.iter().zip(&yref) {
            let d = (a - b).abs();
            if d > 2e-2 * b.abs() + 5e-2 {
                bad += 1;
            }
            worst = worst.max(d);
        }
        let flops = 2.0 * t as f64 * k as f64 * n as f64;
        println!(
            "{label:<9} T={t} K={k} N={n}: compile+load {:.0} ms, best {:.2} ms = {:.0} GFLOPS, mismatches {bad}/{} (worst {worst:.4})",
            g.compile_ms, best, flops / best / 1e6, y.len()
        );
    }
}

#[cfg(all(feature = "ane", target_os = "macos"))]
fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal
            let mut e2 = -14i32;
            let mut m2 = m;
            while m2 & 0x400 == 0 {
                m2 <<= 1;
                e2 -= 1;
            }
            m2 &= 0x3ff;
            (s << 31) | (((e2 + 127) as u32) << 23) | (m2 << 13)
        }
    } else if e == 0x1f {
        (s << 31) | 0x7f80_0000 | (m << 13)
    } else {
        (s << 31) | (((e - 15 + 127) as u32) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[cfg(not(all(feature = "ane", target_os = "macos")))]
fn main() {
    eprintln!("build with --features ane on macOS");
}
