//! Whole-MLP channel offload experiment; no production packet or policy changes.

#[cfg(all(feature = "ane", target_os = "macos"))]
#[path = "support/mlp_metal.rs"]
mod mlp_metal;
#[cfg(all(feature = "ane", target_os = "macos"))]
#[path = "support/mlp_weights.rs"]
mod mlp_weights;

#[cfg(all(feature = "ane", target_os = "macos"))]
mod bench {
    use super::{
        mlp_metal as metal,
        mlp_weights::{bf, mlp_spec, to_bf, Matrix},
    };
    use clap::Parser;
    use objc2_core_ml::MLComputeUnits;
    use objc2_metal::{MTLBuffer, MTLCommandBuffer};
    use plowrt::{
        asset::Checkpoint,
        exec::{
            ane::{AneNet, RunTimings},
            cpu::ffi,
        },
    };
    use serde_json::{json, Value};
    use std::{
        path::PathBuf,
        process::Command,
        time::{Instant, SystemTime, UNIX_EPOCH},
    };

    #[derive(Parser)]
    struct Args {
        checkpoint: PathBuf,
        #[arg(long)]
        twin: Option<PathBuf>,
        #[arg(long, default_value="fp8", value_parser=["bf16","fp8","mxfp4"])]
        encoding: String,
        #[arg(long, default_value_t = 0)]
        layer: usize,
        #[arg(long, value_delimiter = ',', default_value = "32,64,91,128,256,512")]
        rows: Vec<usize>,
        #[arg(long, value_delimiter = ',', default_value = "2048,3072,4096,5120")]
        channels: Vec<usize>,
        #[arg(long, default_value_t = 20)]
        reps: usize,
        #[arg(long, default_value_t = 5)]
        warmup: usize,
        /// 0 = compile exact M; 64 = existing padded-call policy.
        #[arg(long, default_value_t = 0)]
        call_rows: usize,
        #[arg(long, default_value = "/private/tmp/plow-ane-placement")]
        placement: PathBuf,
        #[arg(long)]
        output: PathBuf,
        /// Retain only this invocation's compiled graphs. Default deletes each after measurement.
        #[arg(long)]
        keep_cache: bool,
        #[arg(long)]
        profile: bool,
        #[arg(long, default_value_t = 1.0)]
        input_scale: f32,
        #[arg(long)]
        host_reduce: bool,
        #[arg(long, default_value_t = 0)]
        deep_k: usize,
        #[arg(long)]
        prepared_io: bool,
        /// Diagnostic alternative; default matches production BF16 down-output rounding.
        #[arg(long)]
        f32_baseline: bool,
    }

    fn memory() -> Value {
        let get = |p: &str, a: &[&str]| {
            let r = Command::new(p).args(a).output().unwrap();
            assert!(r.status.success());
            String::from_utf8(r.stdout).unwrap()
        };
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        let peak = unsafe {
            assert_eq!(libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()), 0);
            usage.assume_init().ru_maxrss
        };
        let services: Vec<_> = get("/bin/ps", &["-axo", "pid=,rss=,comm="])
            .lines()
            .filter_map(|line| {
                let mut fields = line.split_whitespace();
                let pid = fields.next()?;
                let rss = fields.next()?;
                let path = fields.next()?;
                (path.contains("/ANE") || path.ends_with("/aned") || path.ends_with("/aneuserd"))
                .then(||json!({"pid":pid,"rss_bytes":rss.parse::<u64>().unwrap()*1024,"path":path}))
            })
            .collect();
        json!({"process_peak_rss_bytes": peak, "ane_services":services,"vm": get("/usr/bin/vm_stat", &[]),
            "pressure": get("/usr/sbin/sysctl", &["-n","kern.memorystatus_vm_pressure_level"]).trim()})
    }

    fn disk_bytes(dir: &std::path::Path) -> u64 {
        std::fs::read_dir(dir)
            .unwrap()
            .map(|e| {
                let e = e.unwrap();
                let meta = e.metadata().unwrap();
                if meta.is_dir() {
                    disk_bytes(&e.path())
                } else {
                    meta.len()
                }
            })
            .sum()
    }

    fn quiet(a: &Value, b: &Value) -> bool {
        let count = |m: &Value, name: &str| {
            m["vm"].as_str().unwrap().lines().find_map(|line| {
                let (k, v) = line.split_once(':')?;
                (k == name).then(|| v.trim().trim_end_matches('.').parse::<u64>().unwrap())
            })
        };
        a["pressure"] == "1"
            && b["pressure"] == "1"
            && ["Swapins", "Swapouts"]
                .iter()
                .all(|n| count(a, n).is_some() && count(a, n) == count(b, n))
    }

    fn quality(a: impl Iterator<Item = f32>, b: impl Iterator<Item = f32>) -> Value {
        let (mut diff, mut norm, mut max_abs, mut count) = (0f64, 0f64, 0f32, 0usize);
        for (a, b) in a.zip(b) {
            if !a.is_finite() || !b.is_finite() {
                return json!({"pass":false,"reason":"nonfinite"});
            }
            let d = f64::from(a) - f64::from(b);
            diff += d * d;
            norm += f64::from(b).powi(2);
            max_abs = max_abs.max(d.abs() as f32);
            count += 1;
        }
        let rel = (diff / norm.max(1e-30)).sqrt();
        json!({"pass":count > 0 && rel < 0.03, "rel_l2":rel, "max_abs":max_abs})
    }

    fn golden(weights: &[Matrix; 3], x: &[u16], gpu_u: &[u16], gpu_y: &[f32]) -> Value {
        let (h, i) = (weights[0].k, weights[0].n);
        let mut u = vec![0u16; i];
        let mut y = vec![0u16; h];
        let mut d = packet::dev::DevInst64 {
            op: [20, 36, 113][weights[0].encoding as usize],
            blocks: 1,
            t: [packet::dev::TENSOR_NONE16; 8],
            ..Default::default()
        };
        d.i[0] = 1;
        d.i[1] = i as u32;
        d.i[2] = h as u32;
        d.i[5] = 1;
        d.t[0] = 0;
        d.t[1] = 1;
        d.t[2] = 2;
        d.t[5] = 3;
        if weights[0].encoding == 1 {
            d.t[4] = 4;
            d.t[6] = 5;
        }
        if weights[0].encoding == 2 {
            d.t[3] = 4;
            d.t[4] = 5;
        }
        let mut ctx = ffi::PlowCpuCtx::new(0, 0);
        ffi::thread_init(&mut ctx).unwrap();
        let mut ptrs = vec![
            u.as_mut_ptr() as *mut _,
            x.as_ptr() as *mut _,
            weights[0].data.as_ptr() as *mut _,
            weights[1].data.as_ptr() as *mut _,
            weights[0].scales.as_ptr() as *mut _,
            weights[1].scales.as_ptr() as *mut _,
        ];
        // SAFETY: one full row; these golden GEMMs need no context scratch.
        unsafe {
            ffi::exec(&d, 0, 1, &ptrs, &mut ctx).unwrap();
        }
        let gu = quality(
            gpu_u.iter().take(i).copied().map(bf),
            u.iter().copied().map(bf),
        );
        d.op = [15, 34, 98][weights[0].encoding as usize];
        d.t = [packet::dev::TENSOR_NONE16; 8];
        d.t[0] = 0;
        d.t[1] = 1;
        d.t[2] = 2;
        d.i[1] = h as u32;
        d.i[2] = i as u32;
        d.i[5] = 0;
        if weights[0].encoding == 1 {
            d.t[4] = 3;
        }
        if weights[0].encoding == 2 {
            d.t[3] = 3;
        }
        ptrs = vec![
            y.as_mut_ptr() as *mut _,
            u.as_ptr() as *mut _,
            weights[2].data.as_ptr() as *mut _,
            weights[2].scales.as_ptr() as *mut _,
        ];
        unsafe {
            ffi::exec(&d, 0, 1, &ptrs, &mut ctx).unwrap();
        }
        let gy = quality(gpu_y.iter().take(h).copied(), y.iter().copied().map(bf));
        assert_eq!(gu["pass"], true, "GPU GLU vs CPU golden: {gu}");
        assert_eq!(gy["pass"], true, "GPU MLP vs CPU golden: {gy}");
        json!({"rows":1,"glu":gu,"down":gy})
    }

    fn summary(samples: &[[f64; 6]]) -> Value {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let median = sorted[sorted.len() / 2];
        let mut dev: Vec<_> = sorted.iter().map(|s| (s[0] - median[0]).abs()).collect();
        dev.sort_by(f64::total_cmp);
        json!({"median_ms":median[0],"p95_ms":sorted[(sorted.len()*95).div_ceil(100)-1][0],
            "mad_ms":dev[dev.len()/2], "median_components":{"gpu_ms":median[1],"pack_ms":median[2],
                "coreml_ms":median[3],"wait_ms":median[4],"reduction_ms":median[5]},"samples":samples})
    }

    #[allow(deprecated)]
    fn mach_clock() -> impl Fn() -> f64 {
        let mut timebase = libc::mach_timebase_info_data_t { numer: 0, denom: 0 };
        assert_eq!(unsafe { libc::mach_timebase_info(&mut timebase) }, 0);
        let clock_scale = f64::from(timebase.numer) / f64::from(timebase.denom) / 1e9;
        move || unsafe { libc::mach_absolute_time() } as f64 * clock_scale
    }

    pub fn main() {
        let args = Args::parse();
        let host_clock = mach_clock();
        assert!(args.reps > 0 && args.warmup > 0 && !args.rows.is_empty());
        assert!(args.rows.iter().all(|&m| m > 0 && m <= 512));
        assert!(args
            .channels
            .iter()
            .all(|&c| c > 0 && c <= 8192 && c % 128 == 0));
        assert!(args.deep_k % 128 == 0);
        assert!(args.call_rows == 0 || args.call_rows == 64);
        assert!(args.input_scale.is_finite() && args.input_scale > 0.0);
        assert!(!args.output.exists(), "report will not overwrite");
        assert!(
            args.placement.is_file(),
            "build runtime/apple/probe/ane_placement.m first"
        );
        let initial = memory();
        assert_eq!(initial["pressure"], "1", "memory pressure");
        ffi::init(ffi::Isa::Scalar).unwrap();
        let ck = Checkpoint::open_with_twin(&args.checkpoint, args.twin.as_deref()).unwrap();
        let encoding = match args.encoding.as_str() {
            "bf16" => 0,
            "fp8" => 1,
            _ => 2,
        };
        let name = |p| format!("model.layers.{}.mlp.{p}_proj.weight", args.layer);
        let (h, i) = (3072, 8192);
        let weights = [
            Matrix::load(&ck, &name("gate"), i, h, encoding),
            Matrix::load(&ck, &name("up"), i, h, encoding),
            Matrix::load(&ck, &name("down"), h, i, encoding),
        ];
        let max_m = *args.rows.iter().max().unwrap();
        let gpu = metal::Metal::new(max_m, h);
        let baseline = metal::Mlp::new(&gpu, &weights, max_m);
        let mut rng = 12345u32;
        let mut input: Vec<u16> = (0..max_m * h)
            .map(|_| {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                to_bf(((rng >> 8) as f32 / 8388608.0 - 1.0) * args.input_scale)
            })
            .collect();
        let residual: Vec<u16> = (0..max_m * h)
            .map(|j| to_bf((j as i32 % 29 - 14) as f32 / 16.0))
            .collect();
        metal::write(&gpu.x, 0, &input);
        metal::write(&gpu.residual, 0, &residual);
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let cache = PathBuf::from("/private/tmp")
            .join(format!("plow-mlp-channel-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&cache).unwrap();
        println!(
            "cache={} encoding={} (no production changes)",
            cache.display(),
            args.encoding
        );
        let mut cells = Vec::new();
        for &m in &args.rows {
            metal::join(&baseline.submit(&gpu, m, true, false));
            let oracle = golden(
                &weights,
                &input,
                metal::read(&baseline.intermediate, m * i),
                metal::read(&baseline.output, m * h),
            );
            for &c in &args.channels {
                let start = Instant::now();
                let aw = [
                    weights[0].slice(0..c, 0..h),
                    weights[1].slice(0..c, 0..h),
                    weights[2].slice(0..h, 0..c),
                ];
                let gw = [
                    weights[0].slice(c..i, 0..h),
                    weights[1].slice(c..i, 0..h),
                    weights[2].slice(0..h, c..i),
                ];
                let partial = metal::Mlp::new(&gpu, &gw, m);
                if c == i {
                    metal::write(&partial.output, 0, &vec![0f32; m * h]);
                }
                let pack_weights_ms = start.elapsed().as_secs_f64() * 1e3;
                let t = if args.call_rows == 0 {
                    m
                } else {
                    args.call_rows
                };
                let name = format!("m{m}-c{c}-t{t}-k{}", args.deep_k);
                let dir = cache.join(&name);
                std::fs::create_dir(&dir).unwrap();
                let spec = mlp_spec(t, &aw[0], &aw[1], &aw[2], args.deep_k);
                let layers = spec.layers.len();
                let io = (spec.inputs.clone(), spec.outputs.clone());
                let mut net =
                    AneNet::new(&dir, "mlp", || spec, io, MLComputeUnits::CPUAndNeuralEngine)
                        .unwrap();
                let compile_ms = net.compile_ms;
                let compiled_bytes = disk_bytes(&dir);
                let placement = Command::new(&args.placement)
                    .arg(dir.join("mlp.mlmodelc"))
                    .output()
                    .unwrap();
                let placement_text = String::from_utf8(placement.stdout).unwrap();
                let placed = placement.status.success()
                    && placement_text.lines().count() == layers
                    && placement_text
                        .lines()
                        .all(|s| s.ends_with("preferred=MLNeuralEngineComputeDevice"));
                if !placed {
                    println!("M={m} C={c}: REJECT placement {placement_text}");
                    cells.push(json!({"m":m,"ane_channels":c,"eligible":false,"reason":"placement","placement":placement_text}));
                    drop(net);
                    if !args.keep_cache {
                        std::fs::remove_dir_all(&dir).unwrap();
                    }
                    continue;
                }
                let mut x = vec![0f32; if args.prepared_io { 0 } else { t * h }];
                let mut y = vec![0f32; m.div_ceil(t) * t * h];
                let mut prepared = if args.prepared_io {
                    Some(net.prepare_io(t).unwrap())
                } else {
                    None
                };
                let mut backing_hits = 0usize;
                let mut base_samples = Vec::new();
                let mut split_samples = Vec::new();
                let mut profiles = Vec::new();
                let mut traces = Vec::new();
                let mut q = Value::Null;
                let mut partial_quality = Value::Null;
                let mut all_quality_pass = true;
                let mut expected = vec![0u16; m * h];
                let mut actual = vec![0u16; m * h];
                let mut expected_consumed = vec![0u16; m * h];
                let mut actual_consumed = vec![0u16; m * h];
                let mut consumed_quality = Value::Null;
                let mut first_ms = [0f64; 2];
                let mut expected_partial = vec![0f32; m * h];
                let mut before = Value::Null;
                for round in 0..args.warmup + args.reps {
                    if round == args.warmup {
                        before = memory();
                    }
                    if round == args.warmup + args.reps / 2 {
                        for v in input.iter_mut().take(m * h) {
                            *v = to_bf(-bf(*v));
                        }
                        metal::write(&gpu.x, 0, &input[..m * h]);
                    }
                    for mode in if round % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let start = Instant::now();
                        if !mode {
                            let device =
                                metal::join(&baseline.submit(&gpu, m, true, !args.f32_baseline));
                            let total = start.elapsed().as_secs_f64() * 1e3;
                            if round == 0 {
                                first_ms[0] = total;
                            }
                            if round >= args.warmup {
                                base_samples.push([total, device, 0.0, 0.0, 0.0, 0.0]);
                            }
                            expected.copy_from_slice(metal::read(&gpu.output, m * h));
                            expected_consumed.copy_from_slice(metal::read(&gpu.consumed, m * h));
                            if args.f32_baseline {
                                expected_partial
                                    .copy_from_slice(metal::read(&baseline.output, m * h));
                            } else {
                                for (v, &bits) in expected_partial
                                    .iter_mut()
                                    .zip(metal::read::<u16>(&baseline.output, m * h))
                                {
                                    *v = bf(bits);
                                }
                            }
                        } else {
                            let cb = (c < i).then(|| partial.submit(&gpu, m, false, false));
                            let (mut pack_ms, mut prediction_ms) = (0.0, 0.0);
                            let mut timings = RunTimings::default();
                            let mut call_intervals = Vec::new();
                            let mut failed = None;
                            for offset in (0..m).step_by(t) {
                                let rows = (m - offset).min(t);
                                let p = Instant::now();
                                let input_array = if let Some(p) = &mut prepared {
                                    p.input_mut(0)
                                } else {
                                    &mut x
                                };
                                for (dst, &src) in input_array[..rows * h]
                                    .iter_mut()
                                    .zip(&input[offset * h..(offset + rows) * h])
                                {
                                    *dst = bf(src);
                                }
                                input_array[rows * h..].fill(0.0);
                                pack_ms += p.elapsed().as_secs_f64() * 1e3;
                                let p = Instant::now();
                                let call_start = args.profile.then(&host_clock);
                                let result = if let Some(p) = &mut prepared {
                                    let r = p.run(args.profile.then_some(&mut timings));
                                    if r.is_ok() {
                                        y[offset * h..(offset + t) * h]
                                            .copy_from_slice(p.output(0));
                                        backing_hits += p.backing_hits;
                                    }
                                    r
                                } else {
                                    net.run_timed(
                                        t,
                                        &[&x],
                                        &mut [&mut y[offset * h..(offset + t) * h]],
                                        args.profile.then_some(&mut timings),
                                    )
                                };
                                if let Some(start) = call_start {
                                    call_intervals.push([start, host_clock()]);
                                }
                                if let Err(e) = result {
                                    failed = Some(e);
                                    break;
                                }
                                prediction_ms += p.elapsed().as_secs_f64() * 1e3;
                            }
                            let wait = Instant::now();
                            let device = cb.as_ref().map_or(0.0, metal::join);
                            let wait_ms = wait.elapsed().as_secs_f64() * 1e3;
                            if let Some(e) = failed {
                                panic!("CoreML failed after GPU join: {e}");
                            }
                            let reduce = Instant::now();
                            if args.host_reduce {
                                let pg: &[f32] = metal::read(&partial.output, m * h);
                                for e in 0..m * h {
                                    actual[e] = to_bf((pg[e] + y[e]) + bf(residual[e]));
                                }
                                metal::write(&gpu.output, 0, &actual);
                                metal::join(&gpu.consume(m));
                            } else {
                                metal::write(&gpu.ane, 0, &y[..m * h]);
                                metal::join(&gpu.finish(&partial.output, m));
                            }
                            let reduce_ms = reduce.elapsed().as_secs_f64() * 1e3;
                            let total = start.elapsed().as_secs_f64() * 1e3;
                            if round == 0 {
                                first_ms[1] = total;
                            }
                            if round >= args.warmup {
                                split_samples.push([
                                    total,
                                    device,
                                    pack_ms,
                                    prediction_ms,
                                    wait_ms,
                                    reduce_ms,
                                ]);
                                profiles.push(timings);
                                if args.profile {
                                    let interval =
                                        cb.as_ref().map(|cb| [cb.GPUStartTime(), cb.GPUEndTime()]);
                                    let overlap = interval.map_or(0.0, |g| {
                                        call_intervals
                                            .iter()
                                            .map(|a| {
                                                (g[1].min(a[1]) - g[0].max(a[0])).max(0.0) * 1e3
                                            })
                                            .sum::<f64>()
                                    });
                                    traces.push(json!({"gpu_device_s":interval,"coreml_call_host_s":call_intervals,
                                        "overlap_ms":overlap}));
                                }
                            }
                            actual.copy_from_slice(metal::read(&gpu.output, m * h));
                            actual_consumed.copy_from_slice(metal::read(&gpu.consumed, m * h));
                        }
                    }
                    q = quality(
                        actual.iter().copied().map(bf),
                        expected.iter().copied().map(bf),
                    );
                    partial_quality = quality(
                        metal::read::<f32>(&partial.output, m * h)
                            .iter()
                            .zip(&y)
                            .map(|(&g, &a)| g + a),
                        expected_partial.iter().copied(),
                    );
                    consumed_quality = quality(
                        actual_consumed.iter().copied().map(bf),
                        expected_consumed.iter().copied().map(bf),
                    );
                    all_quality_pass &= q["pass"] == true
                        && partial_quality["pass"] == true
                        && consumed_quality["pass"] == true;
                }
                let after = memory();
                let base = summary(&base_samples);
                let split = summary(&split_samples);
                let speedup =
                    base["median_ms"].as_f64().unwrap() / split["median_ms"].as_f64().unwrap();
                println!("M={m} C={c} call_rows={t}: GPU={:.3} split={:.3} speedup={speedup:.3} q={q} memory_quiet={}",base["median_ms"].as_f64().unwrap(),split["median_ms"].as_f64().unwrap(),quiet(&before,&after));
                cells.push(json!({"m":m,"ane_channels":c,"gpu_channels":i-c,"call_rows":t,"calls":m.div_ceil(t),
                    "gpu":base,"split":split,"quality":q,"partial_quality":partial_quality,
                    "consumer_quality":consumed_quality,"all_quality_pass":all_quality_pass,
                    "cpu_golden":oracle,"profiles":profiles,"trace":traces,"placement":placement_text,
                    "memory_before":before,"memory_after":after,"memory_quiet":quiet(&before,&after),
                    "gpu_weight_bytes":baseline.weight_bytes,"extra_gpu_slice_bytes":partial.weight_bytes,
                    "ane_raw_weight_bytes":3*c*h*2,
                    "partial_and_staging_bytes":partial.output.length()+gpu.ane.length()+(y.len()+x.len())*4
                        +if args.prepared_io { t*h*8 } else { 0 },
                    "compile_load_ms":compile_ms,"compiled_disk_bytes":compiled_bytes,"first_gpu_ms":first_ms[0],"first_split_ms":first_ms[1],
                    "weight_slice_ms":pack_weights_ms,"backing_hits":backing_hits,"eligible":false}));
                drop(prepared);
                drop(net);
                if !args.keep_cache {
                    std::fs::remove_dir_all(&dir).unwrap();
                }
            }
        }
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.output)
            .unwrap();
        serde_json::to_writer_pretty(file,&json!({"schema":"apple-mlp-channel-probe-v1","encoding":args.encoding,
            "layer":args.layer,"input_scale":args.input_scale,"profile":args.profile,"host_reduce":args.host_reduce,
            "deep_k":args.deep_k,"prepared_io":args.prepared_io,"f32_baseline":args.f32_baseline,
            "next_consumer":"Metal RMSNorm, unit gamma, epsilon 1e-5",
            "h":h,"intermediate":i,"reps":args.reps,"warmup":args.warmup,
            "trace_clock":"system mach time seconds; CoreML host call is not a pure ANE device interval",
            "metal_kernel_sha256":plow_asset::decode_objects::image_sha256(include_bytes!("../../../runtime/apple/interp.metal")),
            "probe_kernel_sha256":plow_asset::decode_objects::image_sha256(include_bytes!("../../../runtime/apple/mlp_channel.metal")),
            "weight_sha256":weights.iter().map(|w|json!({"data":plow_asset::decode_objects::image_sha256(&w.data),
                "scales":plow_asset::decode_objects::image_sha256(&w.scales)})).collect::<Vec<_>>(),
            "cache":cache,"cache_retained":args.keep_cache,"cells":cells,"eligible_for_policy":false})).unwrap();
        println!("wrote {}", args.output.display());
    }
}

#[cfg(all(feature = "ane", target_os = "macos"))]
fn main() {
    bench::main();
}
#[cfg(not(all(feature = "ane", target_os = "macos")))]
fn main() {
    eprintln!("requires macOS --features ane");
}
