//! Single-block launcher. Loads a
//! block asset (a PLOWDEV blob compiled by `gemma4 --block`, its `block.json`
//! descriptor, and a checkpoint) and drives just that block on the real GPU
//! through two verbs on ONE loaded engine:
//!
//!   block_run <asset-dir> check [--in x.npy] [--out y.npy] [--ctx T]
//!   block_run <asset-dir> bench --batch 1,2,4,8 --ctx 128,512,1024,4096
//!                              [--iters 100] [--warmup 10] [--prefill-iters 10]
//!                              [--pf-chunk N]
//!
//! `check` feeds a hidden-state into `act.x` (an .npy or a seeded synthetic),
//! launches one prefill bucket, reads `act.x` back, and prints shape / min /
//! max / mean / NaN-Inf. SCOPE: shape + finiteness + self-consistency only —
//! there is NO PyTorch/HF parity here (no `transformers` in this environment;
//! numeric parity is the `scripts/block_oracle.py` job, deferred).
//!
//! `bench` sweeps decode batch B × context T on the block: prefill B slots to
//! T rows, then time N decode steps per (B,T) and write `sweep.json`. The
//! isolated block has no upstream, so `act.x` is not refreshed between decode
//! steps — the tokens are meaningless, but the per-step KERNEL time (the sweep
//! metric) is data-independent, which is the point.
//!
//! Cloned from `examples/step_bench.rs`; env `PLOW_CHECKPOINT` overrides the
//! default `<asset>/checkpoint`, `PLOW_STEP_TIME=1` adds the engine's host-op
//! breakdown, a `-DPLOW_NV_TRACE=1` cubin adds the per-op cycle profile.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("block_run requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuda::run()
}

#[cfg(feature = "cuda")]
mod cuda {
    /// Discarded prefill passes before timing begins (prefill is expensive; a
    /// couple of passes is enough to settle clocks and allocator state).
    const PF_WARMUP: usize = 2;

    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::Instant;

    /// Minimal NumPy v1.0 reader/writer for C-order little-endian f32 (no dep).
    mod npy {
        use std::io::{Read, Write};

        pub fn read_f32(path: &std::path::Path) -> std::io::Result<(Vec<usize>, Vec<f32>)> {
            let mut f = std::fs::File::open(path)?;
            let mut magic = [0u8; 8];
            f.read_exact(&mut magic)?;
            assert_eq!(
                &magic[..6],
                b"\x93NUMPY",
                "{}: not an npy file",
                path.display()
            );
            let mut hl = [0u8; 2];
            f.read_exact(&mut hl)?;
            let hlen = u16::from_le_bytes(hl) as usize;
            let mut hbuf = vec![0u8; hlen];
            f.read_exact(&mut hbuf)?;
            let hdr = String::from_utf8_lossy(&hbuf);
            assert!(
                hdr.contains("'<f4'") || hdr.contains("'|f4'") || hdr.contains("\"<f4\""),
                "{}: only <f4 (f32 LE) supported, header: {hdr}",
                path.display()
            );
            assert!(
                hdr.contains("'fortran_order': False"),
                "{}: only C-order supported",
                path.display()
            );
            // shape = (a, b, ...)
            let s = hdr.split("'shape':").nth(1).expect("shape key");
            let open = s.find('(').expect("shape (");
            let close = s[open..].find(')').expect("shape )") + open;
            let shape: Vec<usize> = s[open + 1..close]
                .split(',')
                .filter_map(|x| x.trim().parse().ok())
                .collect();
            let mut data = Vec::new();
            f.read_to_end(&mut data)?;
            let n: usize = shape.iter().product();
            let mut out = Vec::with_capacity(n);
            for c in data.chunks_exact(4).take(n) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
            Ok((shape, out))
        }

        pub fn write_f32(
            path: &std::path::Path,
            shape: &[usize],
            data: &[f32],
        ) -> std::io::Result<()> {
            let shape_str = if shape.len() == 1 {
                format!("({},)", shape[0])
            } else {
                let parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                format!("({})", parts.join(", "))
            };
            let mut hdr =
                format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_str}, }}");
            // Pad so that 10 (magic+len) + header is a multiple of 64, header
            // terminated by '\n'.
            let total = 10 + hdr.len() + 1;
            let pad = (64 - total % 64) % 64;
            hdr.push_str(&" ".repeat(pad));
            hdr.push('\n');
            let mut f = std::fs::File::create(path)?;
            f.write_all(b"\x93NUMPY\x01\x00")?;
            f.write_all(&(hdr.len() as u16).to_le_bytes())?;
            f.write_all(hdr.as_bytes())?;
            for &v in data {
                f.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        }
    }

    /// Deterministic seeded hidden state (reproducible without a checkpoint):
    /// a small bounded value per element.
    fn synth(t: usize, hidden: usize) -> Vec<f32> {
        (0..t * hidden)
            .map(|i| {
                let x = (i as f32 * 0.0007).sin() * 0.5;
                x
            })
            .collect()
    }

    fn parse_list(s: &str) -> Vec<usize> {
        s.split(',').filter_map(|x| x.trim().parse().ok()).collect()
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .init();

        let mut args = std::env::args().skip(1);
        let asset = PathBuf::from(
            args.next()
                .ok_or("usage: block_run <asset-dir> <check|bench> [flags]")?,
        );
        let verb = args
            .next()
            .ok_or("usage: block_run <asset-dir> <check|bench> [flags]")?;
        let rest: Vec<String> = args.collect();
        let flag = |name: &str| -> Option<String> {
            rest.iter()
                .position(|a| a == name)
                .and_then(|i| rest.get(i + 1).cloned())
        };

        // block.json descriptor (hidden width, dims) — written next to the blob
        // by `gemma4 --block`.
        let desc: plow_asset::BlockDescriptor = {
            let p = asset.join("block.json");
            let raw = std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            serde_json::from_slice(&raw)?
        };
        let hidden = desc.hidden as usize;

        let ckpt = std::env::var("PLOW_CHECKPOINT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| asset.join("checkpoint"));
        let be = Arc::new(plowrt::device::cuda::CudaBackend::new(0)?);
        let mut e = plowrt::exec::gpu::GpuEngine::load(be, &asset, &ckpt)?;
        println!(
            "block L{} arch={} hidden={hidden} engine batch={} max_ctx={} prefill={}",
            desc.layer,
            desc.arch,
            e.batch(),
            e.max_ctx(),
            e.has_prefill()
        );
        // The block output tensor (residual ping-pongs to `act.xnext` for an
        // odd layer count; decode-only MLA/Mamba blocks report it here).
        let out_name = desc
            .outputs
            .first()
            .map(|o| o.name.clone())
            .unwrap_or_else(|| "act.x".to_string());

        match verb.as_str() {
            "check" => check(&mut e, hidden, &out_name, &flag),
            "bench" => {
                // `bench` prefills every slot, so it still needs the _pf object.
                if !e.has_prefill() {
                    return Err(
                        "block_run bench needs the prefill (_pf) object — set PLOW_NV_CUBIN_PF"
                            .into(),
                    );
                }
                bench(&mut e, hidden, &flag)
            }
            other => Err(format!("unknown verb {other:?} (check|bench)").into()),
        }
    }

    fn check(
        e: &mut plowrt::exec::gpu::GpuEngine,
        hidden: usize,
        out_name: &str,
        flag: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Input: an .npy [T, hidden] or a seeded synthetic (default T=128).
        let (t, xin) = if let Some(p) = flag("--in") {
            let (shape, data) = npy::read_f32(Path::new(&p))?;
            assert_eq!(shape.len(), 2, "--in must be [T, hidden]");
            assert_eq!(
                shape[1], hidden,
                "--in hidden {} != block hidden {hidden}",
                shape[1]
            );
            (shape[0], data)
        } else {
            let t: usize = flag("--ctx").and_then(|s| s.parse().ok()).unwrap_or(128);
            (t, synth(t, hidden))
        };
        println!(
            "check: T={t} hidden={hidden} (input {})",
            if flag("--in").is_some() {
                "npy"
            } else {
                "synthetic"
            }
        );

        // Two launch modes on ONE loaded engine:
        //  - prefill blocks (gemma dense): upload [T,hidden] act.x, launch one
        //    prefill bucket (Embed elided in block mode, so token ids do not
        //    affect the hidden state — only positions / kv length matter).
        //  - decode-only blocks (GLM/Kimi MLA, Nemotron Mamba/GQA/MoE — the
        //    emit path has prefill_buckets=[]): drive ONE decode step (M=1) on a
        //    single row, mirroring step_bench's no-prefill branch.
        let t = if e.has_prefill() {
            e.begin_slot(0, t + 1)?;
            e.upload_activation("act.x", &xin)?;
            let prompt: Vec<u32> = (0..t as u32).map(|i| 100 + (i % 1000)).collect();
            let t0 = Instant::now();
            e.prefill_slot(0, &prompt)?;
            println!(
                "  launched prefill(T={t}) in {:.3} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
            t
        } else {
            // Decode processes one row; feed row 0 of the input.
            println!("  (decode-only block: single decode step, T forced to 1)");
            e.begin_slot(0, 2)?;
            e.upload_activation("act.x", &xin[..hidden])?;
            let mut toks = Vec::new();
            let t0 = Instant::now();
            e.step_slots(&[(0, 100)], &mut toks)?;
            println!(
                "  launched decode(M=1) in {:.3} ms",
                t0.elapsed().as_secs_f64() * 1e3
            );
            1
        };

        let out = e.download_activation(out_name)?;
        let out = &out[..t * hidden]; // trim pad rows past T
        let (mut mn, mut mx, mut sum, mut nan, mut inf) =
            (f32::INFINITY, f32::NEG_INFINITY, 0.0f64, 0usize, 0usize);
        for &v in out {
            if v.is_nan() {
                nan += 1;
                continue;
            }
            if v.is_infinite() {
                inf += 1;
                continue;
            }
            mn = mn.min(v);
            mx = mx.max(v);
            sum += v as f64;
        }
        let finite = out.len() - nan - inf;
        let mean = if finite > 0 { sum / finite as f64 } else { 0.0 };
        println!(
            "  {out_name} out: shape [{t}, {hidden}]  min={mn:.5} max={mx:.5} mean={mean:.6} \
             NaN={nan} Inf={inf}"
        );
        let ok = nan == 0 && inf == 0 && finite == out.len();
        println!(
            "  finiteness: {}",
            if ok {
                "PASS (all finite)"
            } else {
                "FAIL (non-finite present)"
            }
        );

        if let Some(p) = flag("--out") {
            npy::write_f32(Path::new(&p), &[t, hidden], out)?;
            println!("  wrote {p}");
        }
        if let Some(profile) = e.trace_summary()? {
            println!("{profile}");
        }
        if ok {
            Ok(())
        } else {
            Err("act.x contains non-finite values".into())
        }
    }

    fn bench(
        e: &mut plowrt::exec::gpu::GpuEngine,
        hidden: usize,
        flag: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let batches = flag("--batch")
            .map(|s| parse_list(&s))
            .unwrap_or_else(|| vec![1, 2, 4, 8]);
        let ctxs = flag("--ctx")
            .map(|s| parse_list(&s))
            .unwrap_or_else(|| vec![128, 512, 1024, 4096]);
        let iters: usize = flag("--iters").and_then(|s| s.parse().ok()).unwrap_or(100);
        let warmup: usize = flag("--warmup").and_then(|s| s.parse().ok()).unwrap_or(10);
        // Prefill is far more expensive per pass than a decode step, so it gets
        // its own (smaller) iteration count.
        let pf_iters: usize = flag("--prefill-iters")
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        // Rows of `act.x` uploaded before a prefill pass.
        //
        // `act.x` is sized by the packet's LARGEST PREFILL BUCKET, not by
        // max_ctx — on a 12B block that is 8192 rows — so uploading a full
        // ctx=32768 hidden state is rejected outright and the block simply
        // cannot be benched above the bucket. Capping the upload lets
        // `prefill_slot` chunk the prompt as it always does, and the KV cache
        // still grows to the full ctx, which is the only thing the decode-step
        // timing depends on.
        //
        // What this costs: chunks past the first read whatever act.x already
        // holds. That is not a new compromise — the header already says the
        // isolated block has no upstream and its tokens are meaningless — and
        // per-step kernel time stays data-independent, which is what makes the
        // sweep metric valid. It is still a hard rule that NOTHING numeric may
        // be read out of a run using this.
        let pf_chunk: Option<usize> = flag("--pf-chunk").and_then(|s| s.parse().ok());
        let cap = e.batch();

        let mut rows = Vec::new();
        for &t in &ctxs {
            if t > e.max_ctx() {
                eprintln!("skip ctx={t}: exceeds engine max_ctx {}", e.max_ctx());
                continue;
            }
            for &bsz in &batches {
                if bsz > cap {
                    eprintln!("skip batch={bsz}: engine decode batch is {cap}");
                    continue;
                }
                // Prefill each of `bsz` slots to a T-row context (seeded act.x so
                // every run is comparable; numerics irrelevant to step time).
                let prompt: Vec<u32> = (0..t as u32).map(|i| 100 + (i % 1000)).collect();
                let xin = synth(pf_chunk.map_or(t, |c| c.min(t)), hidden);
                let mut last = vec![0u32; bsz];
                let need = t + iters + warmup + 2;

                // PREFILL PHASE — timed with the same warmup/median/p95 treatment
                // as decode, so it is comparable against the baseline harness
                // (`scripts/block_layer_bench.py`, which reports prefill_ms_*).
                // Only `prefill_slot` is inside the timer: `begin_slot` and the
                // act.x upload are setup, and the baseline's prefill is likewise
                // compute-only. `prefill_slot` loops `prefill_chunk` until Done,
                // whose path ends in a D2H token download, so it SYNCHRONIZES —
                // wall-clock here is real execution, not a launch.
                let mut pf_ms: Vec<f64> = Vec::with_capacity(pf_iters);
                for pass in 0..(PF_WARMUP + pf_iters) {
                    let mut acc_us = 0.0f64;
                    for b in 0..bsz {
                        e.begin_slot(b, need)?;
                        e.upload_activation("act.x", &xin)?;
                        let t0 = Instant::now();
                        last[b] = e.prefill_slot(b, &prompt)?;
                        acc_us += t0.elapsed().as_secs_f64() * 1e6;
                    }
                    if pass >= PF_WARMUP {
                        pf_ms.push(acc_us / 1e3);
                    }
                }
                pf_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let pf_med = pf_ms[pf_ms.len() / 2];
                let pf_p95 = pf_ms[((pf_ms.len() as f64 * 0.95) as usize).min(pf_ms.len() - 1)];
                let pf_tok_s = (bsz * t) as f64 / (pf_med / 1e3);
                // The final pass left every slot prefilled to T with `last` set,
                // which is exactly the state the decode loop below expects.
                let feeds = |last: &[u32]| -> Vec<(usize, u32)> {
                    last.iter().enumerate().map(|(b, &tk)| (b, tk)).collect()
                };
                let mut toks = Vec::new();
                for _ in 0..warmup {
                    e.step_slots(&feeds(&last), &mut toks)?;
                    last.copy_from_slice(&toks);
                }
                e.trace_reset()?;
                let mut us: Vec<f64> = Vec::with_capacity(iters);
                for _ in 0..iters {
                    let t0 = Instant::now();
                    e.step_slots(&feeds(&last), &mut toks)?;
                    us.push(t0.elapsed().as_secs_f64() * 1e6);
                    last.copy_from_slice(&toks);
                }
                us.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = us[us.len() / 2];
                let p95 = us[((us.len() as f64 * 0.95) as usize).min(us.len() - 1)];
                let tok_s = 1e6 / median * bsz as f64;
                println!(
                    "  B={bsz:>2} T={t:>5}  decode median={median:>9.2} us p95={p95:>9.2} us \
                     tok/s={tok_s:>9.1} | prefill median={pf_med:>8.2} ms tok/s={pf_tok_s:>9.1}"
                );
                rows.push(serde_json::json!({
                    "batch": bsz,
                    "ctx": t,
                    "latency_us_median": (median * 100.0).round() / 100.0,
                    "latency_us_p95": (p95 * 100.0).round() / 100.0,
                    "tok_s": (tok_s * 10.0).round() / 10.0,
                    "prefill_ms_median": (pf_med * 1000.0).round() / 1000.0,
                    "prefill_ms_p95": (pf_p95 * 1000.0).round() / 1000.0,
                    "prefill_tok_s": (pf_tok_s * 10.0).round() / 10.0,
                }));
            }
        }

        let out_dir = PathBuf::from("/dev/shm/block-asset/bench");
        std::fs::create_dir_all(&out_dir)?;
        let out = out_dir.join("sweep.json");
        std::fs::write(
            &out,
            serde_json::to_vec_pretty(&serde_json::json!({ "sweep": rows }))?,
        )?;
        println!("wrote {}", out.display());
        if let Some(profile) = e.trace_summary()? {
            println!("{profile}");
        }
        Ok(())
    }
}
