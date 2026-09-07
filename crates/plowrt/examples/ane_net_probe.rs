//! Numeric check of the CoreML graph layers `exec::ane::AneNet` uses for a transformer layer
//! (RMSNorm as reduceSumSquare+rsqrt+multiply, innerProduct, sigmoid/GELU, broadcast mul/add,
//! enumerated row shapes) against an f32 reference, on CPU-only and CPU+ANE compute units.
//!
//! `cargo run --release --features ane --example ane_net_probe -- [K] [I] [N]`

#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    use objc2_core_ml::MLComputeUnits;
    use plowrt::exec::ane::{f32_to_f16, AneNet, Layer, NetSpec};

    let mut args = std::env::args().skip(1);
    let k: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(64);
    let inter: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(96);
    let n: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(48);
    let t_enum: Vec<usize> = std::env::var("PLOW_NET_TENUM")
        .ok()
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or(vec![8usize, 24]);
    let eps = 1e-5f32;
    let mut rng = 0x9E3779B9u32;
    let mut frand = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 8) & 0xFFFFFF) as f32 / 8388608.0 - 1.0
    };
    let h16 = |v: f32| half_to_f32(f32_to_f16(v));
    let mk =
        |rows: usize, cols: usize, s: f32, fr: &mut dyn FnMut() -> f32| -> (Vec<u16>, Vec<f32>) {
            let w: Vec<f32> = (0..rows * cols).map(|_| fr() * s).collect();
            let w16: Vec<u16> = w.iter().map(|&v| f32_to_f16(v)).collect();
            let wq: Vec<f32> = w16.iter().map(|&h| half_to_f32(h)).collect();
            (w16, wq)
        };
    let (wo16, wo) = mk(k, k, 0.1, &mut frand);
    let (wg16, wg) = mk(inter, k, 0.1, &mut frand);
    let (wu16, wu) = mk(inter, k, 0.1, &mut frand);
    let (wd16, wd) = mk(k, inter, 0.1, &mut frand);
    let (wq16, wqv) = mk(n, k, 0.1, &mut frand);
    // mid-like graph: o = at.Wo; x1 = x + o; hn = rms(x1); g = hn.Wg; u = hn.Wu; h = silu(g)*u;
    // d = h.Wd; xo = x1 + d; hn2 = rms(xo); qkv = hn2.Wq
    let nl: usize = std::env::var("PLOW_NET_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    let spec = || NetSpec {
        inputs: vec![("x".into(), k), ("at".into(), k)],
        outputs: match nl {
            12 => vec![("xo".into(), k), ("qkv".into(), n)],
            11 => vec![("hn2".into(), k)],
            10 => vec![("xo".into(), k)],
            9 => vec![("d".into(), k)],
            8 => vec![("h".into(), inter)],
            7 => vec![("silu".into(), inter)],
            6 => vec![("sg".into(), inter)],
            5 => vec![("u".into(), inter)],
            4 => vec![("g".into(), inter)],
            3 => vec![("hn".into(), k)],
            2 => vec![("x1".into(), k)],
            _ => vec![("o".into(), k)],
        },
        t_enum: t_enum.clone(),
        flex_outputs: std::env::var("PLOW_FLEX_OUT").is_ok(),
        range: std::env::var("PLOW_RANGE").is_ok(),
        out_range: std::env::var("PLOW_OUT_RANGE").is_ok(),
        layers: vec![
            Layer::InnerProduct {
                input: "at".into(),
                output: "o".into(),
                k,
                n: k,
                w_f16: wo16.clone(),
            },
            Layer::Add {
                a: "x".into(),
                b: "o".into(),
                output: "x1".into(),
            },
            Layer::RmsNorm {
                input: "x1".into(),
                output: "hn".into(),
                c: k,
                eps,
            },
            Layer::InnerProduct {
                input: "hn".into(),
                output: "g".into(),
                k,
                n: inter,
                w_f16: wg16.clone(),
            },
            Layer::InnerProduct {
                input: "hn".into(),
                output: "u".into(),
                k,
                n: inter,
                w_f16: wu16.clone(),
            },
            Layer::Sigmoid {
                input: "g".into(),
                output: "sg".into(),
            },
            Layer::Mul {
                a: "g".into(),
                b: "sg".into(),
                output: "silu".into(),
            },
            Layer::Mul {
                a: "silu".into(),
                b: "u".into(),
                output: "h".into(),
            },
            Layer::InnerProduct {
                input: "h".into(),
                output: "d".into(),
                k: inter,
                n: k,
                w_f16: wd16.clone(),
            },
            Layer::Add {
                a: "x1".into(),
                b: "d".into(),
                output: "xo".into(),
            },
            Layer::RmsNorm {
                input: "xo".into(),
                output: "hn2".into(),
                c: k,
                eps,
            },
            Layer::InnerProduct {
                input: "hn2".into(),
                output: "qkv".into(),
                k,
                n,
                w_f16: wq16.clone(),
            },
        ]
        .into_iter()
        .take(nl)
        .collect(),
    };
    let dir = std::env::temp_dir().join("plow-ane-net-probe");
    let _ = std::fs::remove_dir_all(&dir);
    let io = (
        vec![("x".to_string(), k), ("at".to_string(), k)],
        vec![("xo".to_string(), k), ("qkv".to_string(), n)],
    );
    let matmul = |a: &[f32], t: usize, kk: usize, w: &[f32], nn: usize| -> Vec<f32> {
        let mut y = vec![0f32; t * nn];
        for i in 0..t {
            for j in 0..nn {
                let mut acc = 0f32;
                for q in 0..kk {
                    acc += a[i * kk + q] * w[j * kk + q];
                }
                y[i * nn + j] = acc;
            }
        }
        y
    };
    let rms = |x: &[f32], t: usize, c: usize| -> Vec<f32> {
        let mut y = vec![0f32; t * c];
        for i in 0..t {
            let ss: f32 = x[i * c..(i + 1) * c].iter().map(|v| v * v).sum();
            let r = 1.0 / (ss / c as f32 + eps).sqrt();
            for q in 0..c {
                y[i * c + q] = x[i * c + q] * r;
            }
        }
        y
    };
    for (label, units) in [
        ("cpu-only", MLComputeUnits::CPUOnly),
        ("cpu+ane", MLComputeUnits::CPUAndNeuralEngine),
    ] {
        let mut net =
            AneNet::new(&dir, &format!("mid_{label}"), spec, io.clone(), units).expect("net");
        for &t in &t_enum {
            // A "massive activation" row to exercise the fp16 range of the norm: |x| up to 2000/16.
            let mut x: Vec<f32> = (0..t * k).map(|_| frand()).collect();
            x[3] = 120.0;
            let at: Vec<f32> = (0..t * k).map(|_| frand()).collect();
            let xq: Vec<f32> = x.iter().map(|&v| h16(v)).collect();
            let atq: Vec<f32> = at.iter().map(|&v| h16(v)).collect();
            let o = matmul(&atq, t, k, &wo, k);
            let x1: Vec<f32> = xq.iter().zip(&o).map(|(a, b)| a + b).collect();
            let hn = rms(&x1, t, k);
            let g = matmul(&hn, t, k, &wg, inter);
            let u = matmul(&hn, t, k, &wu, inter);
            let hh: Vec<f32> = g
                .iter()
                .zip(&u)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            let d = matmul(&hh, t, inter, &wd, k);
            let xo_ref: Vec<f32> = x1.iter().zip(&d).map(|(a, b)| a + b).collect();
            let hn2 = rms(&xo_ref, t, k);
            let qkv_ref = matmul(&hn2, t, k, &wqv, n);
            let mut xo = vec![0f32; t * k];
            let mut qkv = vec![0f32; t * n];
            net.run(t, &[&x, &at], &mut [&mut xo, &mut qkv])
                .expect("run");
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                net.run(t, &[&x, &at], &mut [&mut xo, &mut qkv])
                    .expect("run");
                best = best.min(net.last_ms);
            }
            let cmp = |a: &[f32], b: &[f32]| -> (usize, f32) {
                let mut bad = 0;
                let mut worst = 0f32;
                for (x, y) in a.iter().zip(b) {
                    let dif = (x - y).abs();
                    if dif > 2e-2 * y.abs() + 2e-2 {
                        bad += 1;
                    }
                    worst = worst.max(dif);
                }
                (bad, worst)
            };
            let (bx, wx) = cmp(&xo, &xo_ref);
            let (bq, wq) = cmp(&qkv, &qkv_ref);
            println!(
                "{label:<9} T={t:<3}: compile+load {:.0} ms, best {:.3} ms; xo mismatches {bx}/{} (worst {wx:.4}), qkv mismatches {bq}/{} (worst {wq:.4})",
                net.compile_ms, best, xo.len(), qkv.len()
            );
        }
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
