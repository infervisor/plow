//! Single-block launcher. Loads a
//! block asset (a PLOWDEV blob compiled by `gemma4 --block`, its `block.json`
//! descriptor, and a checkpoint) and drives just that block on the real GPU
//! through two verbs on ONE loaded engine:
//!
//!   block_run <asset-dir> check [--in x.npy] [--out y.npy] [--ctx T] [--repeat N]
//!                              [--dump-tensors name,name --dump-dir dir]
//!   block_run <asset-dir> bench --batch 1,2,4,8 --ctx 128,512,1024,4096
//!                              [--iters 100] [--warmup 10] [--prefill-iters 10]
//!                              [--pf-chunk N] [--pf-cap ROWS]
//!   block_run <asset-dir> mixed-check --rows 128 --decode 1
//!   block_run <asset-dir> packed-check
//!   block_run <asset-dir> decode-check --dir <oracle ref_decode dir> [--out-tensor name] [--pf-chunk rows] [--dump-tensors name,name]
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
            let mut f =
                std::io::BufWriter::with_capacity(1024 * 1024, std::fs::File::create(path)?);
            f.write_all(b"\x93NUMPY\x01\x00")?;
            f.write_all(&(hdr.len() as u16).to_le_bytes())?;
            f.write_all(hdr.as_bytes())?;
            for &v in data {
                f.write_all(&v.to_le_bytes())?;
            }
            f.flush()
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
                .ok_or("usage: block_run <asset-dir> <check|bench|mixed-check> [flags]")?,
        );
        let verb = args
            .next()
            .ok_or("usage: block_run <asset-dir> <check|bench|mixed-check> [flags]")?;
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
        if verb == "tp-check" {
            return tp_check(&asset, &ckpt, &desc, &flag);
        }
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
            "check" => {
                // The input tensor and its row multiplier: rows = T x the product of the fixed
                // non-hidden dims (DeepSeek-V4.1's mHC residual is [4, T, hidden]).
                let (in_name, mult) = desc.inputs.first().map_or(("act.x".to_string(), 1), |i| {
                    let fixed: i64 = i.shape[..i.shape.len().saturating_sub(1)]
                        .iter()
                        .map(|d| match d {
                            plow_asset::Dim::Fixed(v) => *v,
                            plow_asset::Dim::Symbolic(_) => 1,
                        })
                        .product();
                    (i.name.clone(), fixed.max(1) as usize)
                });
                check(&mut e, hidden, &in_name, mult, &out_name, &flag)
            }
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
            "mixed-check" => mixed_check(&mut e, &desc, hidden, &out_name, &flag),
            "packed-check" => packed_check(&mut e, &desc, hidden, &out_name),
            "decode-check" => decode_check(&mut e, &desc, hidden, &out_name, &flag),
            other => Err(format!("unknown verb {other:?} (check|bench|mixed-check|decode-check)").into()),
        }
    }

    /// `tp-check`: the block's tensor-parallel packet on every visible GPU, one engine per rank on
    /// its own thread. Each run zeroes every rank's xctr between two barriers, then all ranks
    /// prefill concurrently (their collectives meet in the peer region). The replicated output must
    /// be byte-identical on every rank; rank 0's is written with `--out` and dumped like `check`.
    fn tp_check(
        asset: &Path,
        ckpt: &Path,
        desc: &plow_asset::BlockDescriptor,
        flag: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use packet::dev::DevOp;
        use plowrt::device::cuda::CudaBackend;
        use plowrt::exec::gpu::{GpuEngine, NvTpBind};
        use plowrt::exec::tp::{PeerLayout, TpGroup};
        use std::sync::Barrier;

        let pkt = plowrt::asset::devblob::DevBlob::find_in_dir(asset)?.ok_or("no packet")?;
        let raw = std::fs::read(&pkt)?;
        let blob = plowrt::asset::devblob::DevBlob::parse(&raw)?;
        let tp = blob.tp.clone().ok_or("tp-check needs a tensor-parallel packet")?;
        let n_xctr = blob
            .progs
            .iter()
            .flat_map(|p| &p.insts)
            .filter_map(|d| match DevOp::from_u16(d.op) {
                Some(DevOp::XReduce) => Some(d.i[3]),
                Some(DevOp::XReduceTwoShot) => Some(d.i[3].max(d.i[4])),
                Some(DevOp::XArgmaxFin) => {
                    Some(d.i[4] + packet::devbuild::xargmax_value_lines(d.i[1].max(1)).unwrap_or(1) - 1)
                }
                _ => None,
            })
            .max()
            .map_or(0, |g| g + 1);
        let max_tokens = (tp.slot_bytes / (tp.hidden as u64 * 2)) as u32;
        let layout = PeerLayout::new(tp.hidden, max_tokens, n_xctr).ok_or("peer layout not 128 B aligned")?;
        let n = tp.n_gpu as usize;
        let bes: Vec<Arc<CudaBackend>> = (0..n as u8).map(|d| CudaBackend::new(d).map(Arc::new)).collect::<Result<_, _>>()?;
        for a in &bes {
            for b in &bes {
                if !Arc::ptr_eq(a, b) {
                    a.enable_peer_access(b)?;
                }
            }
        }
        let group = TpGroup::bringup(bes.iter().map(|b| Arc::clone(b) as Arc<dyn plowrt::device::Backend>).collect(), layout)?;
        group.verify_peer_visibility()?;
        println!("tp-check: {n} ranks, hidden={} max_tokens={max_tokens} n_xctr={n_xctr} peer region {} KiB", tp.hidden, group.layout().bytes() / 1024);

        let binds: Vec<NvTpBind> = group
            .ranks()
            .iter()
            .map(|r| NvTpBind {
                rank: r.rank(),
                n_gpu: n as u32,
                peer_table: r.peer_scratch_table(),
                xctr: r.xctr(),
                scratch_base: r.scratch_base(),
                slot_b: tp.slot_bytes,
                slot_bytes: tp.slot_bytes,
            })
            .collect();
        let mut engines: Vec<GpuEngine> = std::thread::scope(|s| {
            let hs: Vec<_> = bes
                .iter()
                .zip(&binds)
                .map(|(be, bind)| {
                    let be = Arc::clone(be);
                    let bind = *bind;
                    s.spawn(move || GpuEngine::load_tp(be, asset, ckpt, Some(bind)))
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("load thread")).collect::<Result<Vec<_>, _>>()
        })?;

        let (in_name, mult) = desc.inputs.first().map_or(("act.x".to_string(), 1), |i| {
            let fixed: i64 = i.shape[..i.shape.len().saturating_sub(1)]
                .iter()
                .map(|d| match d {
                    plow_asset::Dim::Fixed(v) => *v,
                    plow_asset::Dim::Symbolic(_) => 1,
                })
                .product();
            (i.name.clone(), fixed.max(1) as usize)
        });
        let hidden = desc.hidden as usize;
        // `--decode-dir`: decode-check on every rank (oracle ref_decode files), rank 0's output written.
        // `--tokens`: a model packet (embed + head): prompts from `prompt_{b}.npy`, `--steps` greedy
        // decode steps, every sampled id written to `plow_tokens.npy` [1 + steps][slots].
        if let Some(dir) = flag("--decode-dir") {
            let dir = PathBuf::from(dir);
            let tokens = flag("--tokens").is_some();
            let out_tensor = if tokens {
                String::new()
            } else {
                flag("--out-tensor").ok_or("tp-check --decode-dir needs --out-tensor")?
            };
            let chunk: usize = flag("--pf-chunk").and_then(|v| v.parse().ok()).unwrap_or(1024);
            let src = if tokens { "prompt" } else { "pre" };
            let mut pre = Vec::new();
            while dir.join(format!("{src}_{}.npy", pre.len())).exists() {
                pre.push(npy::read_f32(&dir.join(format!("{src}_{}.npy", pre.len())))?.1);
            }
            let mut xs = Vec::new();
            if tokens {
                xs = vec![Vec::new(); flag("--steps").and_then(|v| v.parse().ok()).unwrap_or(3)];
            }
            while !tokens && dir.join(format!("dec_x_{}.npy", xs.len())).exists() {
                xs.push(npy::read_f32(&dir.join(format!("dec_x_{}.npy", xs.len())))?.1);
            }
            let (nb, steps) = (pre.len(), xs.len());
            let iters: usize = flag("--dec-iters").and_then(|v| v.parse().ok()).unwrap_or(0);
            let barrier = Barrier::new(n);
            let dumps = flag("--dump-tensors");
            let (group, pre, xs, barrier, in_name, out_tensor, dumps, dir) = (&group, &pre, &xs, &barrier, &in_name, &out_tensor, &dumps, &dir);
            // Every launch meets its peers in the collectives: all ranks enter it with zeroed counters.
            let fence = |rank: usize| -> Result<(), plowrt::RuntimeError> {
                barrier.wait();
                if rank == 0 {
                    group.zero_xctr()?;
                }
                barrier.wait();
                Ok(())
            };
            let outs: Vec<Vec<Vec<f32>>> = std::thread::scope(|s| {
                let hs: Vec<_> = engines
                    .iter_mut()
                    .enumerate()
                    .map(|(rank, e)| {
                        s.spawn(move || -> Result<Vec<Vec<f32>>, plowrt::RuntimeError> {
                            let mut last = vec![0u32; nb];
                            for (b, x) in pre.iter().enumerate() {
                                let t = if tokens { x.len() } else { x.len() / (mult * hidden) };
                                let prompt: Vec<u32> = if tokens {
                                    x.iter().map(|&v| v as u32).collect()
                                } else {
                                    (0..t as u32).map(|i| 100 + (i % 1000)).collect()
                                };
                                e.begin_slot(b, t + steps + iters + 2)?;
                                let mut c0 = 0;
                                last[b] = loop {
                                    let rows = chunk.min(t - c0) * mult * hidden;
                                    if !tokens {
                                        e.upload_activation(in_name, &x[c0 * mult * hidden..][..rows])?;
                                    }
                                    c0 += chunk;
                                    fence(rank)?;
                                    if let plowrt::exec::gpu::PrefillStep::Done(tok) = e.prefill_chunk(b, &prompt, chunk)? {
                                        break tok;
                                    }
                                };
                            }
                            let mut outs = Vec::new();
                            if tokens {
                                outs.push(last.iter().map(|&v| v as f32).collect());
                            }
                            let mut toks = Vec::new();
                            for x in xs {
                                if !tokens {
                                    e.upload_activation(in_name, x)?;
                                }
                                let feeds: Vec<_> = last.iter().enumerate().map(|(b, &tk)| (b, tk)).collect();
                                fence(rank)?;
                                let t0 = Instant::now();
                                e.step_slots(&feeds, &mut toks)?;
                                if rank == 0 {
                                    println!("tp-decode: step B={nb} {:.3} ms", t0.elapsed().as_secs_f64() * 1e3);
                                }
                                last.copy_from_slice(&toks[..nb]);
                                if tokens {
                                    outs.push(last.iter().map(|&v| v as f32).collect());
                                } else {
                                    outs.push(e.download_activation(out_tensor)?[..nb * mult * hidden].to_vec());
                                }
                                if rank == 0 && outs.len() == 1 + tokens as usize {
                                    for name in dumps.iter().flat_map(|n| n.split(',')) {
                                        let mut raw = vec![0u8; e.tensor_bytes(name).expect("unknown dump tensor") as usize];
                                        e.read_tensor(name, &mut raw)?;
                                        std::fs::write(dir.join(format!("plow0_{name}.bin")), raw).expect("dump write");
                                    }
                                }
                            }
                            let mut ms = Vec::with_capacity(iters);
                            e.trace_reset()?;
                            for _ in 0..iters {
                                let feeds: Vec<_> = last.iter().enumerate().map(|(b, &tk)| (b, tk)).collect();
                                fence(rank)?;
                                let t0 = Instant::now();
                                e.step_slots(&feeds, &mut toks)?;
                                ms.push(t0.elapsed().as_secs_f64() * 1e3);
                                last.copy_from_slice(&toks[..nb]);
                            }
                            if rank == 0 && iters > 0 {
                                let mean = ms.iter().sum::<f64>() / iters as f64;
                                let min = ms.iter().cloned().fold(f64::MAX, f64::min);
                                println!("tp-decode: {iters} timed steps B={nb} mean {mean:.3} ms min {min:.3} ms");
                                if let Some(tr) = e.trace_summary()? {
                                    println!("tp-decode trace: {tr}");
                                }
                            }
                            if iters > 0 {
                                // One more step, spans only: the per-instruction critical path.
                                let feeds: Vec<_> = last.iter().enumerate().map(|(b, &tk)| (b, tk)).collect();
                                fence(rank)?;
                                e.trace_spans_reset()?;
                                e.step_slots(&feeds, &mut toks)?;
                                if let (0, Some(sp)) = (rank, e.trace_spans()?) {
                                    let t0 = sp.iter().filter(|s| s.1 > 0).map(|s| s.0).min().unwrap_or(0);
                                    for (i, (a, b)) in sp.iter().enumerate().filter(|(_, s)| s.1 > 0) {
                                        println!("span {i:4} start {:8.1} us  end {:8.1} us  dur {:7.1} us", (a - t0) as f64 / 1e3, (b - t0) as f64 / 1e3, (b - a) as f64 / 1e3);
                                    }
                                    if let Some(bk) = e.trace_block_spans()? {
                                        let us: Vec<f32> = bk.iter().map(|&v| if v == 0 { -1.0 } else { (v as f64 - t0 as f64) as f32 / 1e3 }).collect();
                                        npy::write_f32(&dir.join("block_spans.npy"), &[3, 128, 256], &us).expect("block_spans.npy");
                                    }
                                    if let Some(p) = e.trace_global_u64("g_probe", 4096)? {
                                        let n = p[0] as usize;
                                        let us: Vec<f32> = p[1..1 + 32 * n.min(127)].iter().map(|&v| if v == 0 { -1.0 } else { (v as f64 - t0 as f64) as f32 / 1e3 }).collect();
                                        npy::write_f32(&dir.join("probe.npy"), &[n.min(127), 32], &us).expect("probe.npy");
                                    }
                                }
                            }
                            Ok(outs)
                        })
                    })
                    .collect();
                hs.into_iter().map(|h| h.join().expect("run thread")).collect::<Result<Vec<_>, _>>()
            })?;
            if flag("--tokens").is_some() {
                let same = outs.iter().all(|r| r == &outs[0]);
                let flat: Vec<f32> = outs[0].iter().flatten().copied().collect();
                npy::write_f32(&dir.join("plow_tokens.npy"), &[outs[0].len(), outs[0][0].len()], &flat)?;
                println!("tp-model: ranks identical={same} tokens={:?}", outs[0]);
                return Ok(());
            }
            for (st, o) in outs[0].iter().enumerate() {
                let same = outs.iter().all(|r| r[st] == *o);
                let bad = o.iter().filter(|v| !v.is_finite()).count();
                npy::write_f32(&dir.join(format!("dec_plow_{st}.npy")), &[nb * mult, hidden], o)?;
                println!("tp-decode: step {st} ranks identical={same} nonfinite={bad}");
            }
            return Ok(());
        }
        let (t, xin) = if let Some(p) = flag("--in") {
            let (shape, data) = npy::read_f32(Path::new(&p))?;
            assert_eq!(shape.len(), 2, "--in must be [rows, hidden]");
            assert_eq!(shape[1], hidden);
            (shape[0] / mult, data)
        } else {
            let t: usize = flag("--ctx").and_then(|s| s.parse().ok()).unwrap_or(128);
            (t, synth(t * mult, hidden))
        };
        let repeat: usize = flag("--repeat").and_then(|s| s.parse().ok()).unwrap_or(1).max(1);
        let prompt: Vec<u32> = (0..t as u32).map(|i| 100 + (i % 1000)).collect();
        let barrier = Barrier::new(n);
        let group = &group;
        let (xin, in_name, prompt, barrier) = (&xin, &in_name, &prompt, &barrier);
        std::thread::scope(|s| {
            let hs: Vec<_> = engines
                .iter_mut()
                .enumerate()
                .map(|(rank, e)| {
                    s.spawn(move || -> Result<(), plowrt::RuntimeError> {
                        for _ in 0..repeat {
                            e.begin_slot(0, t + 1)?;
                            e.upload_activation(in_name, xin)?;
                            barrier.wait();
                            if rank == 0 {
                                group.zero_xctr()?;
                            }
                            barrier.wait();
                            let t0 = Instant::now();
                            e.prefill_slot(0, prompt)?;
                            let ms = t0.elapsed().as_secs_f64() * 1e3;
                            barrier.wait();
                            if rank == 0 {
                                println!("  launched prefill(T={t}) on {n} ranks in {ms:.3} ms (rank 0)");
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            hs.into_iter().try_for_each(|h| h.join().expect("run thread"))
        })?;

        let out_name = desc.outputs.first().map(|o| o.name.clone()).unwrap_or_else(|| "act.x".to_string());
        let outs: Vec<Vec<f32>> = engines.iter_mut().map(|e| e.download_activation(&out_name)).collect::<Result<_, _>>()?;
        let rows = t * mult * hidden;
        let same = outs.iter().all(|o| o[..rows] == outs[0][..rows]);
        let nonfinite = outs[0][..rows].iter().filter(|v| !v.is_finite()).count();
        println!("  {out_name}: ranks identical={same} nonfinite={nonfinite}");
        if let Some(p) = flag("--out") {
            npy::write_f32(Path::new(&p), &[t * mult, hidden], &outs[0][..rows])?;
            println!("  wrote {p}");
        }
        if same && nonfinite == 0 {
            Ok(())
        } else {
            Err("tp-check: ranks disagree or non-finite output".into())
        }
    }

    fn packed_check(
        e: &mut plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        hidden: usize,
        out_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use plowrt::exec::gpu::PfBatchReq;

        let slots = [3usize, 15];
        let starts = [31usize, 95];
        let rows = [33usize, 31];
        if !e.pf_batch_enabled()
            || e.batch() <= slots[1]
            || e.max_ctx() < starts[1] + rows[1]
            || e.pf_max_rows() < rows.iter().sum()
        {
            return Err("packed-check asset lacks B16 packed-prefill capacity".into());
        }
        let prompts: Vec<Vec<u32>> = starts
            .iter()
            .zip(rows)
            .enumerate()
            .map(|(request, (&start, rows))| {
                (0..start + rows + 1)
                    .map(|position| 100 + ((position * 17 + request * 101) % 1000) as u32)
                    .collect()
            })
            .collect();
        let inputs = synth(rows.iter().sum(), hidden);

        let prepare = |e: &mut plowrt::exec::gpu::GpuEngine| {
            for request in 0..slots.len() {
                e.begin_slot(slots[request], prompts[request].len() + 1)?;
                e.upload_activation("act.x", &synth(starts[request], hidden))?;
                e.prefill_batched(&[PfBatchReq {
                    slot: slots[request],
                    prompt: &prompts[request],
                    c0: 0,
                    len: starts[request],
                }])?;
            }
            Ok::<_, plowrt::RuntimeError>(())
        };

        prepare(e)?;
        let mut reference = Vec::new();
        let mut row0 = 0;
        for request in 0..slots.len() {
            let row1 = row0 + rows[request];
            e.upload_activation("act.x", &inputs[row0 * hidden..row1 * hidden])?;
            e.prefill_batched(&[PfBatchReq {
                slot: slots[request],
                prompt: &prompts[request],
                c0: starts[request],
                len: rows[request],
            }])?;
            reference.push(e.download_activation(out_name)?[..rows[request] * hidden].to_vec());
            row0 = row1;
        }
        let ranges: Vec<_> = (0..slots.len())
            .map(|request| (slots[request], starts[request], rows[request]))
            .collect();
        let reference_kv = snapshot_kv_requests(e, desc, &ranges)?;

        prepare(e)?;
        e.upload_activation("act.x", &inputs)?;
        let requests: Vec<_> = (0..slots.len())
            .map(|request| PfBatchReq {
                slot: slots[request],
                prompt: &prompts[request],
                c0: starts[request],
                len: rows[request],
            })
            .collect();
        e.prefill_batched(&requests)?;
        let packed = e.download_activation(out_name)?;
        let mut row0 = 0;
        for (request, expected) in reference.iter().enumerate() {
            let row1 = row0 + rows[request];
            compare_f32(
                &format!("packed request {request} activation"),
                &packed[row0 * hidden..row1 * hidden],
                expected,
                6.0e-3,
                7.5e-2,
            )?;
            row0 = row1;
        }
        compare_bf16(
            "packed sparse-slot KV",
            &snapshot_kv_requests(e, desc, &ranges)?,
            &reference_kv,
            6.0e-3,
            5.0e-2,
        )?;
        println!(
            "packed-check: sparse slots 3/15, absolute starts 31/95, ragged rows 33/31 parity=PASS"
        );
        Ok(())
    }

    /// `decode-check`: replays `block_oracle.py ref_decode` output. Prefills slot b from
    /// `pre_{b}.npy`, then per step uploads `dec_x_{s}.npy` (slot-major rows) and runs one decode
    /// step over every slot, writing the block output to `dec_plow_{s}.npy`.
    fn decode_check(
        e: &mut plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        hidden: usize,
        out_name: &str,
        flag: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dir = PathBuf::from(flag("--dir").ok_or("decode-check needs --dir")?);
        let in_name = desc.inputs.first().map_or("act.x".to_string(), |i| i.name.clone());
        // The decode rung's output buffer need not be the prefill block's (attention-only rungs
        // stop at the ping-pong residual `act.hc_residual_b`).
        let out_tensor = flag("--out-tensor").unwrap_or_else(|| out_name.to_string());
        let mut pre = Vec::new();
        while dir.join(format!("pre_{}.npy", pre.len())).exists() {
            pre.push(npy::read_f32(&dir.join(format!("pre_{}.npy", pre.len())))?.1);
        }
        let mut steps = 0;
        while dir.join(format!("dec_x_{steps}.npy")).exists() {
            steps += 1;
        }
        let nb = pre.len();
        if nb == 0 || steps == 0 || nb > e.batch() {
            return Err(format!("decode-check: {nb} slots / {steps} steps (engine batch {})", e.batch()).into());
        }
        let x0 = npy::read_f32(&dir.join("dec_x_0.npy"))?;
        let mult = x0.0[0] / nb;
        let mut last = vec![0u32; nb];
        let chunk: usize = flag("--pf-chunk").and_then(|v| v.parse().ok()).unwrap_or(1024);
        for (b, x) in pre.iter().enumerate() {
            let t = x.len() / (mult * hidden);
            let prompt: Vec<u32> = (0..t as u32).map(|i| 100 + (i % 1000)).collect();
            e.begin_slot(b, t + steps + 2)?;
            // Block mode has no embed: each chunk's input rows are uploaded before it runs.
            let mut c0 = 0;
            last[b] = loop {
                let rows = chunk.min(t - c0) * mult * hidden;
                e.upload_activation(&in_name, &x[c0 * mult * hidden..][..rows])?;
                c0 += chunk;
                if let plowrt::exec::gpu::PrefillStep::Done(tok) = e.prefill_chunk(b, &prompt, chunk)? {
                    break tok;
                }
            };
            println!("decode-check: slot {b} prefilled T={t}");
        }
        let mut toks = Vec::new();
        for s in 0..steps {
            let x = npy::read_f32(&dir.join(format!("dec_x_{s}.npy")))?.1;
            e.upload_activation(&in_name, &x)?;
            let feeds: Vec<_> = last.iter().enumerate().map(|(b, &tk)| (b, tk)).collect();
            let t0 = Instant::now();
            e.step_slots(&feeds, &mut toks)?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            last.copy_from_slice(&toks[..nb]);
            let out = e.download_activation(&out_tensor)?;
            let rows = nb * mult;
            let bad = out[..rows * hidden].iter().filter(|v| !v.is_finite()).count();
            npy::write_f32(&dir.join(format!("dec_plow_{s}.npy")), &[rows, hidden], &out[..rows * hidden])?;
            if s == 0 {
                for name in flag("--dump-tensors").iter().flat_map(|n| n.split(',')) {
                    let mut raw = vec![0u8; usize::try_from(e.tensor_bytes(name).ok_or("unknown dump tensor")?)?];
                    e.read_tensor(name, &mut raw)?;
                    std::fs::write(dir.join(format!("plow0_{name}.bin")), raw)?;
                }
            }
            println!("decode-check: step {s} B={nb} {ms:.3} ms nonfinite={bad}");
        }
        Ok(())
    }

    fn mixed_check(
        e: &mut plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        hidden: usize,
        out_name: &str,
        flag: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let rows = flag("--rows").and_then(|s| s.parse().ok()).unwrap_or(128);
        let decode_rows = flag("--decode").and_then(|s| s.parse().ok()).unwrap_or(1);
        let spans = flag("--spans").and_then(|s| s.parse().ok()).unwrap_or(1);
        if decode_rows == 0 || decode_rows + 1 > e.batch() || decode_rows >= rows {
            return Err("mixed-check needs 0 < decode < rows and one free prefill slot".into());
        }
        if spans == 0 {
            return Err("mixed-check needs at least one span".into());
        }
        let prefill_rows = rows - decode_rows;
        for slot in 0..=decode_rows {
            e.begin_slot(slot, spans * rows + 1)?;
        }
        let input = synth(rows * spans, hidden);
        let prefill_tokens: Vec<_> = (0..prefill_rows * spans)
            .map(|row| 200 + row as u32)
            .collect();
        let start = Instant::now();
        let mut mixed_outputs = Vec::with_capacity(spans);
        let mut mixed_kv = Vec::with_capacity(spans);
        for span in 0..spans {
            e.upload_activation(
                "act.x",
                &input[span * rows * hidden..(span + 1) * rows * hidden],
            )?;
            let decode: Vec<_> = (0..decode_rows)
                .map(|slot| plow_asset::mixed_step::DecodeRequest {
                    slot: slot as u32,
                    state_slot: slot as u32,
                    token: 100 + (span * decode_rows + slot) as u32,
                })
                .collect();
            let pf_start = span * prefill_rows;
            let prefill = [plow_asset::mixed_step::PrefillRequest {
                slot: decode_rows as u32,
                state_slot: decode_rows as u32,
                start: pf_start as u32,
                tokens: &prefill_tokens[pf_start..pf_start + prefill_rows],
                prompt_len: prefill_tokens.len() as u32,
            }];
            e.mixed_step(rows as u32, &decode, &prefill, &mut [])?;
            let out = e.download_activation(out_name)?;
            if out[..rows * hidden].iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "mixed block output contains non-finite values at span {span}"
                )
                .into());
            }
            mixed_outputs.push(out[..rows * hidden].to_vec());
            mixed_kv.push(snapshot_kv_range(
                e,
                desc,
                decode_rows,
                span,
                prefill_rows,
                pf_start,
            )?);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;

        for slot in 0..=decode_rows {
            e.begin_slot(slot, spans * rows + 1)?;
        }
        for span in 0..spans {
            let span_input = &input[span * rows * hidden..(span + 1) * rows * hidden];
            e.upload_activation("act.x", &span_input[..decode_rows * hidden])?;
            let feeds: Vec<_> = (0..decode_rows)
                .map(|slot| (slot, 100 + (span * decode_rows + slot) as u32))
                .collect();
            let mut tokens = Vec::new();
            e.step_slots(&feeds, &mut tokens)?;
            let decode_out = e.download_activation(out_name)?;

            e.upload_activation("act.x", &span_input[decode_rows * hidden..])?;
            let pf_end = (span + 1) * prefill_rows;
            e.prefill_slot(decode_rows, &prefill_tokens[..pf_end])?;
            let prefill_out = e.download_activation(out_name)?;
            let reference_kv = snapshot_kv_range(
                e,
                desc,
                decode_rows,
                span,
                prefill_rows,
                span * prefill_rows,
            )?;
            let mut reference_out = Vec::with_capacity(rows * hidden);
            reference_out.extend_from_slice(&decode_out[..decode_rows * hidden]);
            reference_out.extend_from_slice(&prefill_out[..prefill_rows * hidden]);
            compare_f32(
                &format!("span {span} activation"),
                &mixed_outputs[span],
                &reference_out,
                6.0e-3,
                7.5e-2,
            )?;
            compare_bf16(
                &format!("span {span} written KV"),
                &mixed_kv[span],
                &reference_kv,
                6.0e-3,
                5.0e-2,
            )?;
        }
        println!(
            "mixed-check: rows={rows} decode={decode_rows} prefill_rows={prefill_rows} spans={spans} elapsed={elapsed_ms:.3} ms parity=PASS"
        );
        Ok(())
    }

    fn snapshot_kv_range(
        e: &plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        decode_rows: usize,
        decode_position: usize,
        prefill_rows: usize,
        prefill_position: usize,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let ranges: Vec<_> = (0..decode_rows)
            .map(|slot| (slot, decode_position, 1))
            .chain(std::iter::once((
                decode_rows,
                prefill_position,
                prefill_rows,
            )))
            .collect();
        snapshot_kv_requests(e, desc, &ranges)
    }

    fn snapshot_kv_requests(
        e: &plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        ranges: &[(usize, usize, usize)],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let heads = usize::try_from(desc.dims.kv_heads.ok_or("block has no KV-head count")?)?;
        let head_dim = usize::try_from(desc.dims.head_dim.ok_or("block has no head dimension")?)?;
        let row_bytes = head_dim.checked_mul(2).ok_or("KV row size overflow")?;
        let mut out = Vec::new();
        for state in &desc.carried_state {
            if state.role != "kv" {
                continue;
            }
            for name in &state.tensors {
                let tensor_bytes = e
                    .tensor_bytes(name)
                    .ok_or_else(|| format!("missing carried-state tensor {name:?}"))?;
                let slot_bytes = tensor_bytes / e.batch() as u64;
                let head_bytes = slot_bytes / heads as u64;
                for &(slot, position, written_rows) in ranges {
                    for head in 0..heads {
                        let offset = slot as u64 * slot_bytes
                            + head as u64 * head_bytes
                            + position as u64 * row_bytes as u64;
                        let begin = out.len();
                        out.resize(begin + written_rows * row_bytes, 0);
                        e.read_tensor_range(name, offset, &mut out[begin..])?;
                    }
                }
            }
        }
        if out.is_empty() {
            return Err("block descriptor has no KV carried state".into());
        }
        Ok(out)
    }

    fn compare_f32(
        what: &str,
        got: &[f32],
        reference: &[f32],
        rel_l2_limit: f64,
        abs_limit: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if got.len() != reference.len() {
            return Err(format!("{what}: length {} != {}", got.len(), reference.len()).into());
        }
        let mut err2 = 0.0;
        let mut ref2 = 0.0;
        let mut max_abs = 0.0f64;
        let mut max_ref = 0.0f64;
        for (&a, &b) in got.iter().zip(reference) {
            let delta = (a as f64 - b as f64).abs();
            err2 += delta * delta;
            ref2 += (b as f64) * (b as f64);
            max_abs = max_abs.max(delta);
            max_ref = max_ref.max((b as f64).abs());
        }
        let rel_l2 = (err2 / ref2.max(f64::MIN_POSITIVE)).sqrt();
        let scaled_abs_limit = abs_limit + rel_l2_limit * max_ref;
        println!("  {what}: rel_l2={rel_l2:.3e} max_abs={max_abs:.3e} max_ref={max_ref:.3e}");
        if rel_l2 > rel_l2_limit || max_abs > scaled_abs_limit {
            return Err(format!(
                "{what} parity failed: rel_l2 {rel_l2:.3e} > {rel_l2_limit:.3e} or max_abs {max_abs:.3e} > {scaled_abs_limit:.3e}"
            )
            .into());
        }
        Ok(())
    }

    fn compare_bf16(
        what: &str,
        got: &[u8],
        reference: &[u8],
        rel_l2_limit: f64,
        abs_limit: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let decode = |bytes: &[u8]| {
            bytes
                .chunks_exact(2)
                .map(|x| f32::from_bits(u32::from(u16::from_le_bytes([x[0], x[1]])) << 16))
                .collect::<Vec<_>>()
        };
        compare_f32(
            what,
            &decode(got),
            &decode(reference),
            rel_l2_limit,
            abs_limit,
        )
    }

    fn check(
        e: &mut plowrt::exec::gpu::GpuEngine,
        hidden: usize,
        in_name: &str,
        mult: usize,
        out_name: &str,
        flag: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dumps = match (flag("--dump-tensors"), flag("--dump-dir")) {
            (None, None) => None,
            (Some(names), Some(dir)) => {
                let mut tensors = Vec::new();
                for name in names.split(',').map(str::trim) {
                    if name.is_empty() || tensors.iter().any(|(prior, _)| prior == name) {
                        return Err("--dump-tensors requires distinct nonempty names".into());
                    }
                    let bytes = e
                        .tensor_bytes(name)
                        .ok_or_else(|| format!("unknown dump tensor {name:?}"))?;
                    tensors.push((name.to_string(), usize::try_from(bytes)?));
                }
                Some((PathBuf::from(dir), tensors))
            }
            _ => return Err("--dump-tensors and --dump-dir must be provided together".into()),
        };
        // Input: an .npy [T, hidden] or a seeded synthetic (default T=128).
        let (t, xin) = if let Some(p) = flag("--in") {
            let (shape, data) = npy::read_f32(Path::new(&p))?;
            assert_eq!(shape.len(), 2, "--in must be [rows, hidden]");
            assert_eq!(
                shape[1], hidden,
                "--in hidden {} != block hidden {hidden}",
                shape[1]
            );
            assert_eq!(shape[0] % mult, 0, "--in rows must be a multiple of {mult}");
            (shape[0] / mult, data)
        } else {
            let t: usize = flag("--ctx").and_then(|s| s.parse().ok()).unwrap_or(128);
            (t, synth(t * mult, hidden))
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
        // `--pf-chunk N`: the prompt in launches of at most N rows, each chunk's input uploaded
        // before it and its output rows collected after it (block mode has no embed).
        let mut chunked: Option<Vec<f32>> = None;
        let t = if e.has_prefill() {
            let prompt: Vec<u32> = (0..t as u32).map(|i| 100 + (i % 1000)).collect();
            // `--repeat N`: the same real input N times, so a timing read (PLOW_PF_SEG_TIME)
            // can skip the cold first launch.
            let repeat: usize = flag("--repeat").and_then(|s| s.parse().ok()).unwrap_or(1);
            if let Some(chunk) = flag("--pf-chunk").and_then(|s| s.parse::<usize>().ok()) {
                e.begin_slot(0, t + 1)?;
                let mut acc = Vec::with_capacity(t * mult * hidden);
                let mut c0 = 0;
                while c0 < t {
                    let rows = chunk.min(t - c0) * mult * hidden;
                    e.upload_activation(in_name, &xin[c0 * mult * hidden..][..rows])?;
                    e.prefill_chunk(0, &prompt, chunk)?;
                    acc.extend_from_slice(&e.download_activation(out_name)?[..rows]);
                    c0 += chunk;
                }
                println!("  launched prefill(T={t}) in chunks of {chunk}");
                chunked = Some(acc);
            }
            for _ in 0..if chunked.is_some() { 0 } else { repeat.max(1) } {
                e.begin_slot(0, t + 1)?;
                e.upload_activation(in_name, &xin)?;
                let t0 = Instant::now();
                e.prefill_slot(0, &prompt)?;
                println!(
                    "  launched prefill(T={t}) in {:.3} ms",
                    t0.elapsed().as_secs_f64() * 1e3
                );
            }
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

        let out = match chunked {
            Some(v) => v,
            None => e.download_activation(out_name)?,
        };
        let out = &out[..t * mult * hidden]; // trim pad rows past T
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
            npy::write_f32(Path::new(&p), &[t * mult, hidden], out)?;
            println!("  wrote {p}");
        }
        if let Some((dir, tensors)) = dumps {
            std::fs::create_dir_all(&dir)?;
            let mut rows = Vec::new();
            for (index, (name, bytes)) in tensors.into_iter().enumerate() {
                let mut raw = vec![0u8; bytes];
                e.read_tensor(&name, &mut raw)?;
                let file = format!("tensor-{index:03}.bin");
                std::fs::write(dir.join(&file), raw)?;
                rows.push(serde_json::json!({"name": name, "bytes": bytes, "file": file}));
            }
            std::fs::write(
                dir.join("manifest.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "scope": "raw complete allocations after block execution; may contain reused scratch or padding",
                    "input_rows": t,
                    "tensors": rows,
                }))?,
            )?;
            println!("  wrote raw tensor dumps to {}", dir.display());
        }
        let profile = if e.has_prefill() {
            e.trace_summary_pf()?
        } else {
            e.trace_summary()?
        };
        if let Some(profile) = profile {
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
        // Rows one prefill launch may take (the serve layer's per-launch cap), so one packet
        // A/Bs its launch geometry: `--pf-cap 4224` vs `8192` on an 8192-bucket block.
        let pf_cap: usize = flag("--pf-cap")
            .and_then(|s| s.parse().ok())
            .unwrap_or(usize::MAX);
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
                        last[b] = loop {
                            if let plowrt::exec::gpu::PrefillStep::Done(tok) =
                                e.prefill_chunk(b, &prompt, pf_cap)?
                            {
                                break tok;
                            }
                        };
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

        let out_dir = flag("--out-dir")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/dev/shm/block-asset/bench"));
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
