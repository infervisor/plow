//! Bisect which CoreML layer/feature crashes or fails the compiler: builds incremental graphs.
//! `cargo run --release --features ane --example ane_layer_probe`
#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    if std::env::args().any(|a| a == "--enum-big") {
        return enum_big();
    }
    use objc2_core_ml::MLComputeUnits;
    use plowrt::exec::ane::{f32_to_f16, AneNet, Layer, NetSpec};
    let k = 32usize;
    let w16: Vec<u16> = (0..k * k)
        .map(|i| f32_to_f16((i % 7) as f32 * 0.01))
        .collect();
    let dir = std::env::temp_dir().join("plow-ane-layer-probe");
    let _ = std::fs::remove_dir_all(&dir);
    let stages: Vec<(&str, bool, Vec<Layer>)> = vec![
        (
            "ip",
            false,
            vec![Layer::InnerProduct {
                input: "x".into(),
                output: "y".into(),
                k,
                n: k,
                w_f16: w16.clone(),
            }],
        ),
        (
            "ip-enum",
            true,
            vec![Layer::InnerProduct {
                input: "x".into(),
                output: "y".into(),
                k,
                n: k,
                w_f16: w16.clone(),
            }],
        ),
        (
            "add",
            false,
            vec![Layer::Add {
                a: "x".into(),
                b: "x".into(),
                output: "y".into(),
            }],
        ),
        (
            "mul",
            false,
            vec![Layer::Mul {
                a: "x".into(),
                b: "x".into(),
                output: "y".into(),
            }],
        ),
        (
            "sigmoid",
            false,
            vec![Layer::Sigmoid {
                input: "x".into(),
                output: "y".into(),
            }],
        ),
        (
            "gelu",
            false,
            vec![Layer::Gelu {
                input: "x".into(),
                output: "y".into(),
            }],
        ),
        (
            "rms",
            false,
            vec![Layer::RmsNorm {
                input: "x".into(),
                output: "y".into(),
                c: k,
                eps: 1e-5,
            }],
        ),
        (
            "two-io",
            false,
            vec![Layer::Add {
                a: "x".into(),
                b: "x".into(),
                output: "y".into(),
            }],
        ),
        (
            "two-io-enum",
            true,
            vec![Layer::Add {
                a: "x".into(),
                b: "x".into(),
                output: "y".into(),
            }],
        ),
        (
            "chain",
            true,
            vec![
                Layer::InnerProduct {
                    input: "x".into(),
                    output: "o".into(),
                    k,
                    n: k,
                    w_f16: w16.clone(),
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
                    eps: 1e-5,
                },
                Layer::InnerProduct {
                    input: "hn".into(),
                    output: "g".into(),
                    k,
                    n: k,
                    w_f16: w16.clone(),
                },
                Layer::Sigmoid {
                    input: "g".into(),
                    output: "sg".into(),
                },
                Layer::Mul {
                    a: "g".into(),
                    b: "sg".into(),
                    output: "y".into(),
                },
            ],
        ),
    ];
    for (name, en, layers) in stages {
        println!("stage {name}: building");
        let two = name.starts_with("two-io");
        let mut layers = layers;
        if two {
            layers.push(Layer::Mul {
                a: "at".into(),
                b: "y".into(),
                output: "y2".into(),
            });
        }
        let spec = NetSpec {
            inputs: if two {
                vec![("x".into(), k), ("at".into(), k)]
            } else {
                vec![("x".into(), k)]
            },
            outputs: if two {
                vec![("y".into(), k), ("y2".into(), k)]
            } else {
                vec![("y".into(), k)]
            },
            t_enum: if en { vec![8, 16] } else { vec![8] },
            flex_outputs: std::env::var("PLOW_FLEX_OUT").is_ok(),
            range: std::env::var("PLOW_RANGE").is_ok(),
            out_range: std::env::var("PLOW_OUT_RANGE").is_ok(),
        w8: std::env::var("PLOW_ANE_W8").as_deref() == Ok("1"),
            layers,
        };
        let io = (spec.inputs.clone(), spec.outputs.clone());
        let units = if std::env::var("PLOW_CPU_ONLY").is_ok() {
            MLComputeUnits::CPUOnly
        } else {
            MLComputeUnits::CPUAndNeuralEngine
        };
        match AneNet::new(&dir, name, move || spec, io, units) {
            Ok(mut net) => {
                let x: Vec<f32> = (0..8 * k).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect();
                let mut y = vec![0f32; 8 * k];
                let mut y2 = vec![0f32; 8 * k];
                let r = if two {
                    net.run(8, &[&x, &x], &mut [&mut y, &mut y2])
                } else {
                    net.run(8, &[&x], &mut [&mut y])
                };
                match r {
                    Ok(()) => println!("stage {name}: ok, y[0..4] = {:?}", &y[..4]),
                    Err(e) => println!("stage {name}: run error {e}"),
                }
            }
            Err(e) => println!("stage {name}: build error {e}"),
        }
    }
}
#[cfg(not(all(feature = "ane", target_os = "macos")))]
fn main() {}

/// `ane_layer_probe --enum-big`: one input, one output, enumerated rows {64,128,256} at a real
/// projection size — does the ANE serve the non-default shapes or does CoreML fall back to CPU?
#[cfg(all(feature = "ane", target_os = "macos"))]
#[allow(dead_code)]
pub fn enum_big() {
    use objc2_core_ml::MLComputeUnits;
    use plowrt::exec::ane::{f32_to_f16, AneNet, Layer, NetSpec};
    let (k, n) = (3072usize, 8192usize);
    let w16: Vec<u16> = (0..n * k)
        .map(|i| f32_to_f16(((i % 13) as f32 - 6.0) * 0.01))
        .collect();
    let dir = std::env::temp_dir().join("plow-ane-enum-big");
    let _ = std::fs::remove_dir_all(&dir);
    let ts = vec![64usize, 128, 256];
    let spec = NetSpec {
        inputs: vec![("x".into(), k)],
        outputs: vec![("y".into(), n)],
        t_enum: ts.clone(),
        flex_outputs: std::env::var("PLOW_FLEX_OUT").is_ok(),
        range: false,
        out_range: std::env::var("PLOW_OUT_RANGE").is_ok(),
        w8: std::env::var("PLOW_ANE_W8").as_deref() == Ok("1"),
        layers: vec![Layer::InnerProduct {
            input: "x".into(),
            output: "y".into(),
            k,
            n,
            w_f16: w16,
        }],
    };
    let io = (spec.inputs.clone(), spec.outputs.clone());
    let mut net = AneNet::new(
        &dir,
        "enum-big",
        move || spec,
        io,
        MLComputeUnits::CPUAndNeuralEngine,
    )
    .expect("net");
    for &t in &ts {
        let x: Vec<f32> = (0..t * k).map(|i| (i % 7) as f32 * 0.1 - 0.3).collect();
        let mut y = vec![0f32; t * n];
        net.run(t, &[&x], &mut [&mut y]).expect("run");
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            net.run(t, &[&x], &mut [&mut y]).expect("run");
            best = best.min(net.last_ms);
        }
        let gf = 2.0 * t as f64 * k as f64 * n as f64 / best / 1e6;
        println!("enum-big T={t}: best {best:.2} ms = {gf:.0} GFLOPS");
    }
}
