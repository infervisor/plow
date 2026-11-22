//! Single-block launcher. Loads a
//! block asset (a PLOWDEV blob compiled by `gemma4 --block`, its `block.json`
//! descriptor, and a checkpoint) and drives just that block on the real GPU
//! through two verbs on ONE loaded engine:
//!
//!   block_run <asset-dir> check [--in x.npy] [--out y.npy] [--ctx T] [--repeat N]
//!                              [--dump-tensors name,name --dump-dir dir]
//!   block_run <asset-dir> bench --batch 1,2,4,8 --ctx 128,512,1024,4096
//!                              [--iters 100] [--warmup 10] [--prefill-iters 10]
//!                              [--decode-in rows.npy]
//!                              [--pf-chunk N] [--pf-cap ROWS] [--fixed-ctx 1]
//!   block_run <asset-dir> decode-check [--batch B --ctx T --steps N | --batch-seq 48,1,8]
//!                              [--in x.npy] [--rows r.npy]
//!                              [--out y.npy] [--poison name,name]
//!                              [--dump-tensors name,name --dump-dir dir]
//!   block_run <asset-dir> mixed-check --rows 128 --decode 1
//!   block_run <asset-dir> packed-check [--requests 16 --query-rows 128 --kv-base 4096]
//!                                      [--lens 1,130,7 [--starts 0,300,0]] [--exact 1]
//!                                      [--dump-dir dir] [--dump-tensors name,name]
//!   block_run <asset-dir> prefill-bench --query-rows 512,2048 --prior 0,4096,16384
//!                                      [--requests N] [--riders D --rider-ctx C]
//!                                      [--iters 20] [--warmup 3] [--out-dir dir]
//!                                      [--in x.npy]
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
//! `prefill-bench` times one prefill launch per point: `--requests` chunks of `--query-rows`
//! rows, each after `--prior` rows of its own prefix (prefilled untimed first), plus `--riders`
//! one-row decode requests at `--rider-ctx` riding the same packed launch. Rows land in
//! `sweep.json` as `{"phase": "prefill", ...}` for `scripts/campaign/block_roofline.py`. Without
//! a packed-prefill contract only one request without riders is timed (the serial chunk path).
//! `--in` times the leading rows of a recorded hidden state instead of the synthetic one: an MoE
//! block's expert union, and so its time, depends on the routing the rows produce.
//!
//! Every `sweep.json` carries `provenance`: packet and binary sha256, device, the block
//! descriptor, the packet's static route inventory and the library-attention launch counts
//! (served vs native fallback), so a block number states which routes it measured.
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
        s.split(',')
            .map(|x| {
                x.trim()
                    .parse()
                    .unwrap_or_else(|_| panic!("bad list entry {x:?} in {s:?}"))
            })
            .collect()
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
                .ok_or("usage: block_run <asset-dir> <check|bench|prefill-bench|decode-check|mixed-check|packed-check> [flags]")?,
        );
        let verb = args
            .next()
            .ok_or("usage: block_run <asset-dir> <check|bench|prefill-bench|decode-check|mixed-check|packed-check> [flags]")?;
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
        let device = be.device_name().to_string();
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
            "bench" | "prefill-bench" => {
                // Both prefill slots, so they need the _pf object.
                if !e.has_prefill() {
                    return Err(format!(
                        "block_run {verb} needs the prefill (_pf) object — set PLOW_NV_CUBIN_PF"
                    )
                    .into());
                }
                let rows = if verb == "bench" {
                    bench(&mut e, hidden, &flag)?
                } else {
                    prefill_bench(&mut e, hidden, &flag)?
                };
                let out_dir = flag("--out-dir")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from("/dev/shm/block-asset/bench"));
                std::fs::create_dir_all(&out_dir)?;
                let out = out_dir.join("sweep.json");
                let doc = serde_json::json!({
                    "sweep": rows,
                    "provenance": provenance(&e, &asset, &desc, &device)?,
                });
                std::fs::write(&out, serde_json::to_vec_pretty(&doc)?)?;
                println!("wrote {}", out.display());
                if let Some(profile) = e.trace_summary()? {
                    println!("{profile}");
                }
                Ok(())
            }
            "decode-check" => decode_check(&mut e, hidden, &out_name, &flag),
            "mixed-check" => mixed_check(&mut e, &desc, kv_e4m3(&asset)?, hidden, &out_name, &flag),
            "packed-check" => packed_check(&mut e, &desc, kv_e4m3(&asset)?, hidden, &out_name, &flag),
            other => Err(format!(
                "unknown verb {other:?} (check|bench|prefill-bench|decode-check|mixed-check|packed-check)"
            )
            .into()),
        }
    }

    /// The packet's cache writers store unit-scale E4M3 rows (`live_kv::KV_E4M3`, object-wide, no
    /// scale tensors).
    fn kv_e4m3(asset: &Path) -> Result<bool, Box<dyn std::error::Error>> {
        let raw = std::fs::read(asset.join("model.pkt"))?;
        let blob = plowrt::asset::devblob::DevBlob::parse_l2(&raw, true)?;
        Ok(blob.progs.iter().flat_map(|p| &p.insts).any(|inst| {
            inst.op == packet::dev::DevOp::QwenHeadNormRope as u16
                && inst.i[7] & plow_asset::live_kv::KV_E4M3 != 0
        }))
    }

    fn sha256_file(path: &Path) -> Result<String, Box<dyn std::error::Error>> {
        use sha2::Digest;
        let digest = sha2::Sha256::digest(std::fs::read(path)?);
        Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
    }

    /// What a block number measured: packet, binary, device, descriptor, declared routes and
    /// the library-attention launches that actually ran.
    fn provenance(
        e: &plowrt::exec::gpu::GpuEngine,
        asset: &Path,
        desc: &plow_asset::BlockDescriptor,
        device: &str,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
        let pkt = asset.join("model.pkt");
        let raw = std::fs::read(&pkt)?;
        let blob = plowrt::asset::devblob::DevBlob::parse_l2(&raw, true)?;
        let routes = plowrt::inventory::inventory(&blob, &raw, Some(asset))?;
        let exe = std::env::current_exe()?;
        let env: serde_json::Map<String, serde_json::Value> = std::env::vars()
            .filter(|(k, _)| k.starts_with("PLOW_"))
            .map(|(k, v)| (k, serde_json::Value::String(v)))
            .collect();
        Ok(serde_json::json!({
            "asset": asset.display().to_string(),
            "packet_sha256": sha256_file(&pkt)?,
            "binary": exe.display().to_string(),
            "binary_sha256": sha256_file(&exe)?,
            "device": device,
            "block": desc,
            "routes": routes,
            "library_attention": e.library_attention_launches().map(|(served, fallback)| {
                serde_json::json!({"served": served, "native_fallback": fallback})
            }),
            "env": env,
        }))
    }

    fn median_p95(mut v: Vec<f64>) -> (f64, f64) {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (
            v[v.len() / 2],
            v[((v.len() as f64 * 0.95) as usize).min(v.len() - 1)],
        )
    }

    /// One prefill launch per point after an untimed prefix (see the module docs).
    fn prefill_bench(
        e: &mut plowrt::exec::gpu::GpuEngine,
        hidden: usize,
        flag: &dyn Fn(&str) -> Option<String>,
    ) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
        use plowrt::exec::gpu::{PfBatchReq, PrefillStep};
        let num = |name: &str, default: usize| -> Result<usize, Box<dyn std::error::Error>> {
            Ok(flag(name)
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(default))
        };
        let query_rows = flag("--query-rows").map_or_else(|| vec![128], |s| parse_list(&s));
        let priors = flag("--prior").map_or_else(|| vec![0], |s| parse_list(&s));
        let requests = num("--requests", 1)?;
        let riders = num("--riders", 0)?;
        let iters = num("--iters", 20)?.max(1);
        let warmup = num("--warmup", 3)?;
        let rider_ctx_flag = flag("--rider-ctx")
            .map(|s| s.parse::<usize>())
            .transpose()?;
        let xin = flag("--in")
            .map(|p| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                let (shape, data) = npy::read_f32(Path::new(&p))?;
                if shape.len() != 2 || shape[1] != hidden {
                    return Err(format!("--in must be [rows, {hidden}], got {shape:?}").into());
                }
                Ok(data)
            })
            .transpose()?;
        let timed_x = |t: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            match &xin {
                Some(x) if x.len() >= t * hidden => Ok(x[..t * hidden].to_vec()),
                Some(x) => {
                    Err(format!("--in has {} rows, the point needs {t}", x.len() / hidden).into())
                }
                None => Ok(synth(t, hidden)),
            }
        };
        let packed = e.pf_batch_enabled();
        if requests == 0 || requests + riders > e.batch() {
            return Err(format!(
                "prefill-bench needs 1 <= requests and requests + riders <= {} slots",
                e.batch()
            )
            .into());
        }
        if !packed && (requests != 1 || riders != 0) {
            return Err(
                "packet has no packed-prefill contract: only --requests 1 without riders".into(),
            );
        }
        // Prefix fill chunk: the widest launch the packet takes.
        let fill = e.pf_request_max_rows().max(1);
        let mut rows = Vec::new();
        for &q in &query_rows {
            for &prior in &priors {
                let rider_ctx = rider_ctx_flag.unwrap_or(prior.max(1));
                // Two spare tokens keep every timed chunk short of its prompt end, so no
                // iteration publishes a prefix snapshot.
                let need = (prior + q + 2).max(rider_ctx + 3);
                if need + 1 > e.max_ctx()
                    || (packed && (requests * q + riders > e.pf_max_rows() || q > fill))
                {
                    eprintln!(
                        "skip rows={q} prior={prior}: exceeds max_ctx {} or packed rows {}",
                        e.max_ctx(),
                        e.pf_max_rows()
                    );
                    continue;
                }
                let prompt = |seed: usize, len: usize| -> Vec<u32> {
                    (0..len)
                        .map(|i| 100 + ((i * 17 + seed * 101) % 1000) as u32)
                        .collect()
                };
                let prompts: Vec<Vec<u32>> = (0..requests)
                    .map(|r| prompt(r, prior + q + 2))
                    .chain((0..riders).map(|r| prompt(requests + r, rider_ctx + 3)))
                    .collect();
                let starts: Vec<usize> = (0..requests)
                    .map(|_| prior)
                    .chain((0..riders).map(|_| rider_ctx))
                    .collect();
                let lens: Vec<usize> = (0..requests)
                    .map(|_| q)
                    .chain((0..riders).map(|_| 1))
                    .collect();
                let total: usize = lens.iter().sum();
                let mut ms = Vec::with_capacity(iters);
                if packed {
                    // A packed request must start at its slot's frontier, so every pass refills
                    // the prefixes (untimed) before the timed launch.
                    let reqs: Vec<PfBatchReq> = (0..prompts.len())
                        .map(|slot| PfBatchReq {
                            slot,
                            prompt: &prompts[slot],
                            c0: starts[slot],
                            len: lens[slot],
                        })
                        .collect();
                    let x = timed_x(total)?;
                    for pass in 0..warmup + iters {
                        for (slot, (p, &start)) in prompts.iter().zip(&starts).enumerate() {
                            e.begin_slot(slot, p.len() + 1)?;
                            let mut c0 = 0;
                            while c0 < start {
                                let len = (start - c0).min(fill);
                                e.upload_activation("act.x", &synth(len, hidden))?;
                                e.prefill_batched(&[PfBatchReq {
                                    slot,
                                    prompt: p,
                                    c0,
                                    len,
                                }])?;
                                c0 += len;
                            }
                        }
                        e.upload_activation("act.x", &x)?;
                        let t0 = Instant::now();
                        e.prefill_batched(&reqs)?;
                        if pass >= warmup {
                            ms.push(t0.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                } else {
                    // Serial chunk path: the timed chunk needs the prefix replayed each pass.
                    let p = &prompts[0];
                    for pass in 0..warmup + iters {
                        e.begin_slot(0, p.len() + 1)?;
                        if prior > 0 {
                            e.upload_activation("act.x", &synth(prior.min(fill), hidden))?;
                            while !matches!(
                                e.prefill_chunk(0, &p[..prior], fill)?,
                                PrefillStep::Done(_)
                            ) {}
                        }
                        e.upload_activation("act.x", &timed_x(q)?)?;
                        let t0 = Instant::now();
                        let mut launches = 0;
                        while !matches!(
                            e.prefill_chunk(0, &p[..prior + q], q)?,
                            PrefillStep::Done(_)
                        ) {
                            launches += 1;
                        }
                        if launches > 0 {
                            return Err(format!(
                                "rows={q} after prior={prior} took more than one launch"
                            )
                            .into());
                        }
                        if pass >= warmup {
                            ms.push(t0.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                }
                let (median, p95) = median_p95(ms);
                println!(
                    "  prefill {requests}x{q} rows prior={prior} riders={riders}@{rider_ctx}: \
                     median={median:.3} ms p95={p95:.3} ms ({} path)",
                    if packed { "packed" } else { "serial" }
                );
                rows.push(serde_json::json!({
                    "phase": "prefill",
                    "requests": requests,
                    "query_rows": q,
                    "prior": prior,
                    "riders": riders,
                    "rider_ctx": rider_ctx,
                    "packed": packed,
                    "ms_median": median,
                    "ms_p95": p95,
                    "iters": iters,
                }));
            }
        }
        Ok(rows)
    }

    fn packed_check(
        e: &mut plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        e4m3: bool,
        hidden: usize,
        out_name: &str,
        flag: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use plowrt::exec::gpu::PfBatchReq;

        let list = |name: &str| -> Result<Option<Vec<usize>>, Box<dyn std::error::Error>> {
            flag(name)
                .map(|v| v.split(',').map(|x| x.trim().parse::<usize>()).collect::<Result<Vec<_>, _>>())
                .transpose()
                .map_err(|e| format!("{name}: {e}").into())
        };
        let (slots, starts, rows) = if let Some(lens) = list("--lens")? {
            // Mixed lengths, each request continuing from its own prior rows (`--starts`, default
            // 0) on its own slot.
            let starts = list("--starts")?.unwrap_or_else(|| vec![0; lens.len()]);
            if lens.is_empty() || lens.len() > e.batch() || starts.len() != lens.len() || lens.contains(&0) {
                return Err("packed-check --lens needs 1..=batch non-zero lengths and matching --starts".into());
            }
            let slots = (0..lens.len()).map(|r| (3 + 5 * r) % e.batch()).collect::<Vec<_>>();
            (slots, starts, lens)
        } else if let Some(count) = flag("--requests") {
            let count: usize = count.parse()?;
            let query_rows: usize = flag("--query-rows")
                .ok_or("packed-check --requests needs --query-rows")?
                .parse()?;
            let kv_base: usize = flag("--kv-base")
                .ok_or("packed-check --requests needs --kv-base")?
                .parse()?;
            if !(1..=16).contains(&count) || query_rows == 0 || kv_base < query_rows {
                return Err(
                    "packed-check needs 1..16 requests and kv-base >= query-rows > 0".into(),
                );
            }
            let slots = (0..count).map(|r| 3 + 8 * r).collect::<Vec<_>>();
            let starts = (0..count)
                .map(|r| kv_base + 64 * r - query_rows)
                .collect::<Vec<_>>();
            (slots, starts, vec![query_rows; count])
        } else {
            (vec![3usize, 15], vec![31usize, 95], vec![33usize, 31])
        };
        if !e.pf_batch_enabled()
            || e.batch() <= *slots.last().unwrap()
            || e.max_ctx() < starts.last().unwrap() + rows.last().unwrap()
            || e.pf_max_rows() < rows.iter().sum()
        {
            return Err("packed-check asset lacks requested packed-prefill capacity".into());
        }
        let prompts: Vec<Vec<u32>> = starts
            .iter()
            .zip(rows.iter())
            .enumerate()
            .map(|(request, (&start, &rows))| {
                (0..start + rows + 1)
                    .map(|position| 100 + ((position * 17 + request * 101) % 1000) as u32)
                    .collect()
            })
            .collect();
        let inputs = synth(rows.iter().sum(), hidden);

        let prepare = |e: &mut plowrt::exec::gpu::GpuEngine| {
            for request in 0..slots.len() {
                e.begin_slot(slots[request], prompts[request].len() + 1)?;
                let mut c0 = 0;
                while c0 < starts[request] {
                    let len = (starts[request] - c0).min(e.pf_max_rows()).min(1024);
                    e.upload_activation("act.x", &synth(len, hidden))?;
                    e.prefill_batched(&[PfBatchReq {
                        slot: slots[request],
                        prompt: &prompts[request],
                        c0,
                        len,
                    }])?;
                    c0 += len;
                }
            }
            Ok::<_, plowrt::RuntimeError>(())
        };

        prepare(e)?;
        let mut reference = Vec::new();
        let mut row0 = 0;
        let mut serial_ms = 0.0;
        for request in 0..slots.len() {
            let row1 = row0 + rows[request];
            e.upload_activation("act.x", &inputs[row0 * hidden..row1 * hidden])?;
            let started = Instant::now();
            e.prefill_batched(&[PfBatchReq {
                slot: slots[request],
                prompt: &prompts[request],
                c0: starts[request],
                len: rows[request],
            }])?;
            serial_ms += started.elapsed().as_secs_f64() * 1e3;
            reference.push(e.download_activation(out_name)?[..rows[request] * hidden].to_vec());
            if request == 0 {
                dump_packed_stages(e, flag, "serial")?;
            }
            row0 = row1;
        }
        let ranges: Vec<_> = (0..slots.len())
            .map(|request| (slots[request], starts[request], rows[request]))
            .collect();
        let reference_kv = snapshot_kv_requests(e, desc, e4m3, &ranges)?;
        let reference_state = snapshot_recurrent(e, desc, &slots)?;
        if let Some(dir) = flag("--dump-dir") {
            let dir = Path::new(&dir);
            std::fs::create_dir_all(dir)?;
            let serial: Vec<f32> = reference
                .iter()
                .flat_map(|rows| rows.iter().copied())
                .collect();
            npy::write_f32(
                &dir.join("serial-activation.npy"),
                &[rows.iter().sum(), hidden],
                &serial,
            )?;
            std::fs::write(dir.join("serial-kv.bin"), &reference_kv)?;
        }

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
        let started = Instant::now();
        e.prefill_batched(&requests)?;
        let packed_ms = started.elapsed().as_secs_f64() * 1e3;
        let packed = e.download_activation(out_name)?;
        dump_packed_stages(e, flag, "packed")?;
        let packed_kv = snapshot_kv_requests(e, desc, e4m3, &ranges)?;
        let packed_state = snapshot_recurrent(e, desc, &slots)?;
        if let Some(dir) = flag("--dump-dir") {
            let dir = Path::new(&dir);
            std::fs::create_dir_all(dir)?;
            npy::write_f32(
                &dir.join("packed-activation.npy"),
                &[rows.iter().sum(), hidden],
                &packed[..rows.iter().sum::<usize>() * hidden],
            )?;
            std::fs::write(dir.join("packed-kv.bin"), &packed_kv)?;
        }
        println!(
            "packed-check timing: requests={} query_rows={} kv_base={} serial_ms={serial_ms:.4} packed_ms={packed_ms:.4}",
            slots.len(), rows.iter().sum::<usize>(), starts[0] + rows[0]
        );
        let mut row0 = 0;
        let exact = flag("--exact").is_some_and(|v| v != "0");
        if !reference_kv.is_empty() {
            let same = packed_kv == reference_kv;
            println!("  packed KV bytes: {}", if same { "bit-identical" } else { "differ" });
            if exact && !same {
                return Err("packed KV differs from serial (--exact)".into());
            }
            compare_kv(e, desc, e4m3, "packed sparse-slot KV", &packed_kv, &reference_kv)?;
        }
        for (((name, slot, elem), got), (_, want)) in packed_state.iter().zip(&reference_state) {
            let diff = got.iter().zip(want).filter(|(a, b)| a != b).count();
            println!("  recurrent {name} slot {slot}: {diff} of {} bytes differ", got.len());
            if diff != 0 {
                if exact {
                    return Err(format!("recurrent {name} slot {slot} differs (--exact)").into());
                }
                let f = |b: &[u8]| -> Vec<f32> {
                    b.chunks_exact(2)
                        .map(|x| {
                            let h = u16::from_le_bytes([x[0], x[1]]);
                            if *elem == "f16" { f16_to_f32(h) } else { f32::from_bits(u32::from(h) << 16) }
                        })
                        .collect()
                };
                compare_f32(&format!("recurrent {name} slot {slot}"), &f(got), &f(want), 6.0e-3, 7.5e-2)?;
            }
        }
        for (request, expected) in reference.iter().enumerate() {
            let row1 = row0 + rows[request];
            let got = &packed[row0 * hidden..row1 * hidden];
            let diff = got.iter().zip(expected).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            println!("  packed request {request} ({} rows): {diff} of {} elements differ", rows[request], got.len());
            if exact && diff != 0 {
                return Err(format!("packed request {request} differs (--exact)").into());
            }
            compare_f32(
                &format!("packed request {request} activation"),
                &packed[row0 * hidden..row1 * hidden],
                expected,
                6.0e-3,
                7.5e-2,
            )?;
            row0 = row1;
        }
        println!("packed-check: requests={} query_rows={} kv_base={} serial_ms={serial_ms:.4} packed_ms={packed_ms:.4} parity=PASS",
            slots.len(), rows.iter().sum::<usize>(), starts[0] + rows[0]);
        Ok(())
    }

    fn dump_packed_stages(
        e: &plowrt::exec::gpu::GpuEngine,
        flag: &dyn Fn(&str) -> Option<String>,
        label: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(names) = flag("--dump-tensors") else {
            return Ok(());
        };
        let dir = PathBuf::from(flag("--dump-dir").ok_or("--dump-tensors needs --dump-dir")?);
        std::fs::create_dir_all(&dir)?;
        for name in names
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            let size = usize::try_from(
                e.tensor_bytes(name)
                    .ok_or_else(|| format!("unknown stage tensor {name:?}"))?,
            )?;
            let mut data = vec![0u8; size];
            e.read_tensor_range(name, 0, &mut data)?;
            let filename = format!("{label}-{}.bin", name.replace('/', "_").replace('.', "_"));
            std::fs::write(dir.join(filename), data)?;
        }
        Ok(())
    }

    /// `decode-check --batch B --ctx T --steps N --out y.npy`: prefill B slots to T rows, then N
    /// decode steps, each on a fresh seeded act.x of B rows; writes the N x B block outputs (byte
    /// comparison of two decode routes on one packet).
    fn decode_check(
        e: &mut plowrt::exec::gpu::GpuEngine,
        hidden: usize,
        out_name: &str,
        flag: &dyn Fn(&str) -> Option<String>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let t: usize = flag("--ctx").and_then(|s| s.parse().ok()).unwrap_or(1024);
        // `--batch-seq 48,1,48`: step s decodes slots 0..seq[s] (the rest idle), so the engine
        // changes rung between steps; otherwise `--batch` slots for `--steps` steps.
        let seq: Vec<usize> = match flag("--batch-seq") {
            Some(s) => parse_list(&s),
            None => {
                let bsz: usize = flag("--batch").and_then(|s| s.parse().ok()).unwrap_or(1);
                let steps: usize = flag("--steps").and_then(|s| s.parse().ok()).unwrap_or(4);
                vec![bsz; steps]
            }
        };
        let (bsz, steps) = (seq.iter().copied().max().unwrap_or(0), seq.len());
        if steps == 0 || seq.contains(&0) || bsz > e.batch() || !e.has_prefill() {
            return Err("decode-check needs 0 < batch <= engine batch and a prefill object".into());
        }
        // `--poison a,b`: every byte of these tensors set to 0xff before each step, so a step
        // that reads what it did not write this step shows up as non-finite or changed output.
        let poison: Vec<String> = flag("--poison")
            .map(|s| {
                s.split(',')
                    .map(str::trim)
                    .filter(|n| !n.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default();
        // `--in x.npy` ([T, hidden], every slot's prompt) and `--rows r.npy` ([steps * B, hidden],
        // the decode rows) replace the seeded synthetic inputs, for a reference comparison.
        let read_rows = |name: &str| -> Result<Option<Vec<f32>>, Box<dyn std::error::Error>> {
            let Some(p) = flag(name) else { return Ok(None) };
            let (shape, data) = npy::read_f32(Path::new(&p))?;
            if shape.len() != 2 || shape[1] != hidden {
                return Err(format!("{name} must be [rows, {hidden}], got {shape:?}").into());
            }
            Ok(Some(data))
        };
        let xin = read_rows("--in")?.unwrap_or_else(|| synth(t, hidden));
        let t = xin.len() / hidden;
        let prompt: Vec<u32> = (0..t as u32).map(|i| 100 + (i % 1000)).collect();
        let mut last = vec![0u32; bsz];
        let (x_rows, mut scratch) = (act_x_rows(e, hidden), Vec::new());
        for (b, tok) in last.iter_mut().enumerate() {
            e.begin_slot(b, t + steps + 2)?;
            *tok = prefill_rows(e, b, &prompt, &xin, hidden, x_rows, out_name, &mut scratch)?.0;
        }
        let rows = read_rows("--rows")?.unwrap_or_else(|| synth(steps * bsz, hidden));
        if rows.len() < steps * bsz * hidden {
            return Err(format!("--rows needs {} rows", steps * bsz).into());
        }
        let mut out = Vec::with_capacity(seq.iter().sum::<usize>() * hidden);
        let mut toks = Vec::new();
        for (s, &n) in seq.iter().enumerate() {
            for name in &poison {
                let bytes = e
                    .tensor_bytes(name)
                    .ok_or_else(|| format!("unknown poison tensor {name:?}"))?;
                e.write_tensor(name, 0, &vec![0xffu8; usize::try_from(bytes)?])?;
            }
            e.upload_activation("act.x", &rows[s * bsz * hidden..(s * bsz + n) * hidden])?;
            let feeds: Vec<_> = last[..n]
                .iter()
                .enumerate()
                .map(|(b, &tk)| (b, tk))
                .collect();
            e.step_slots(&feeds, &mut toks)?;
            if toks.len() >= n {
                last[..n].copy_from_slice(&toks[..n]);
            }
            let y = e.download_activation(out_name)?;
            out.extend_from_slice(&y[..n * hidden]);
        }
        let bad = out.iter().filter(|v| !v.is_finite()).count();
        println!("decode-check: B={bsz} T={t} steps={steps} seq={seq:?} non-finite={bad}");
        if let Some(p) = flag("--out") {
            npy::write_f32(Path::new(&p), &[out.len() / hidden, hidden], &out)?;
            println!("  wrote {p}");
        }
        // Raw allocations after the last step (`--dump-tensors a,b --dump-dir d`): d/<name>.bin.
        if let (Some(names), Some(dir)) = (flag("--dump-tensors"), flag("--dump-dir")) {
            std::fs::create_dir_all(&dir)?;
            for name in names.split(',').map(str::trim).filter(|n| !n.is_empty()) {
                let bytes = e
                    .tensor_bytes(name)
                    .ok_or_else(|| format!("unknown dump tensor {name:?}"))?;
                let mut raw = vec![0u8; usize::try_from(bytes)?];
                e.read_tensor(name, &mut raw)?;
                std::fs::write(Path::new(&dir).join(format!("{name}.bin")), raw)?;
            }
            println!("  wrote tensor dumps to {dir}");
        }
        if bad == 0 {
            Ok(())
        } else {
            Err("decode output contains non-finite values".into())
        }
    }

    fn mixed_check(
        e: &mut plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        e4m3: bool,
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
                e4m3,
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
                e4m3,
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
            compare_kv(
                e,
                desc,
                e4m3,
                &format!("span {span} written KV"),
                &mixed_kv[span],
                &reference_kv,
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
        e4m3: bool,
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
        snapshot_kv_requests(e, desc, e4m3, &ranges)
    }

    fn snapshot_kv_requests(
        e: &plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        e4m3: bool,
        ranges: &[(usize, usize, usize)],
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let heads = usize::try_from(desc.dims.kv_heads.ok_or("block has no KV-head count")?)?;
        let head_dim = usize::try_from(desc.dims.head_dim.ok_or("block has no head dimension")?)?;
        let mut out = Vec::new();
        for state in &desc.carried_state {
            if state.role != "kv" {
                continue;
            }
            for name in &state.tensors {
                let scale_name = format!("{name}_scale");
                let fp8 = e.tensor_bytes(&scale_name).is_some();
                let kv_row_bytes = head_dim
                    .checked_mul(if fp8 || e4m3 { 1 } else { 2 })
                    .ok_or("KV row size overflow")?;
                for (tensor, row_bytes) in [(name.as_str(), kv_row_bytes), (&scale_name, 4)] {
                    if tensor == scale_name.as_str() && !fp8 {
                        continue;
                    }
                    let tensor_bytes = e
                        .tensor_bytes(tensor)
                        .ok_or_else(|| format!("missing carried-state tensor {tensor:?}"))?;
                    let slot_bytes = tensor_bytes / e.batch() as u64;
                    let head_bytes = slot_bytes / heads as u64;
                    let row_bytes = row_bytes as u64;
                    if head_bytes % row_bytes != 0 || head_bytes == 0 {
                        return Err(format!("invalid KV head stride for {tensor}").into());
                    }
                    let capacity = head_bytes / row_bytes;
                    for &(slot, position, written_rows) in ranges {
                        if written_rows as u64 > capacity {
                            return Err("KV snapshot range exceeds physical ring capacity".into());
                        }
                        for head in 0..heads {
                            let mut absolute = position as u64;
                            let mut remaining = written_rows as u64;
                            while remaining != 0 {
                                let physical = absolute % capacity;
                                let take = remaining.min(capacity - physical);
                                let offset = slot as u64 * slot_bytes
                                    + head as u64 * head_bytes
                                    + physical * row_bytes;
                                let begin = out.len();
                                out.resize(begin + (take * row_bytes) as usize, 0);
                                e.read_tensor_range(tensor, offset, &mut out[begin..])?;
                                absolute += take;
                                remaining -= take;
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    /// Every recurrent carried-state tensor's bytes at each slot, as `((name, slot, element), bytes)`.
    #[allow(clippy::type_complexity)]
    fn snapshot_recurrent(
        e: &plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        slots: &[usize],
    ) -> Result<Vec<((String, usize, &'static str), Vec<u8>)>, Box<dyn std::error::Error>> {
        let mut out = Vec::new();
        for state in desc.carried_state.iter().filter(|s| s.role == "recurrent") {
            for name in &state.tensors {
                let bytes = e
                    .tensor_bytes(name)
                    .ok_or_else(|| format!("missing carried-state tensor {name:?}"))?;
                let stride = bytes / e.batch() as u64;
                let elem = if name.ends_with(".ssm") { "f16" } else { "bf16" };
                for &slot in slots {
                    let mut buf = vec![0u8; stride as usize];
                    e.read_tensor_range(name, slot as u64 * stride, &mut buf)?;
                    out.push(((name.clone(), slot, elem), buf));
                }
            }
        }
        Ok(out)
    }

    fn f16_to_f32(h: u16) -> f32 {
        let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
        let v = match exp {
            0 => man as f32 * 2f32.powi(-24),
            31 => if man == 0 { f32::INFINITY } else { f32::NAN },
            _ => f32::from_bits((exp + 112) << 23 | man << 13),
        };
        if sign == 1 { -v } else { v }
    }

    fn compare_kv(
        e: &plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        e4m3: bool,
        what: &str,
        got: &[u8],
        reference: &[u8],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let fp8 = desc.carried_state.iter().any(|state| {
            state.role == "kv"
                && state
                    .tensors
                    .iter()
                    .any(|name| e.tensor_bytes(&format!("{name}_scale")).is_some())
        });
        if fp8 {
            if got != reference {
                return Err(format!("{what}: FP8 KV or scale bytes differ").into());
            }
            println!("  {what}: {} FP8 KV and scale bytes exact", got.len());
            Ok(())
        } else if e4m3 {
            if got.len() != reference.len() {
                return Err(format!("{what}: length {} != {}", got.len(), reference.len()).into());
            }
            // A row whose bf16 source differs within the activation tolerance (another GEMM arm
            // for the request's own bucket) may round to the neighbouring E4M3 code: allow one
            // code step of the same sign, and bound the decoded rel-L2.
            let diff = got.iter().zip(reference).filter(|(a, b)| a != b).count();
            let far = got
                .iter()
                .zip(reference)
                .filter(|(&a, &b)| (a ^ b) & 0x80 != 0 || (a & 0x7f).abs_diff(b & 0x7f) > 1)
                .count();
            println!("  {what}: {diff} of {} E4M3 KV bytes differ, {far} by more than one code", got.len());
            if far != 0 {
                return Err(format!("{what}: {far} E4M3 KV bytes differ by more than one code").into());
            }
            if diff == 0 {
                return Ok(());
            }
            let decode = |b: &u8| {
                let (exp, man) = (i32::from((b >> 3) & 0xf), f32::from(b & 7));
                let v = if exp == 0 { man * 2f32.powi(-9) } else { (1.0 + man / 8.0) * 2f32.powi(exp - 7) };
                if b & 0x80 != 0 { -v } else { v }
            };
            let (g, r): (Vec<f32>, Vec<f32>) =
                (got.iter().map(decode).collect(), reference.iter().map(decode).collect());
            compare_f32(what, &g, &r, 6.0e-3, f64::INFINITY)
        } else {
            compare_bf16(what, got, reference, 6.0e-3, 5.0e-2)
        }
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

    /// Prefill slot `b` with `xin` as act.x, chunk by chunk: the engine splits an off-bucket
    /// prompt into several launches, and in block mode (no Embed) each chunk reads act.x from
    /// row 0, so each chunk gets its own input rows uploaded and its output rows read back
    /// into `out`. Returns (first token, chunks).
    #[allow(clippy::too_many_arguments)]
    fn prefill_rows(
        e: &mut plowrt::exec::gpu::GpuEngine,
        b: usize,
        prompt: &[u32],
        xin: &[f32],
        hidden: usize,
        x_rows: usize,
        out_name: &str,
        out: &mut Vec<f32>,
    ) -> Result<(u32, usize), Box<dyn std::error::Error>> {
        let t = prompt.len();
        out.clear();
        let (mut c0, mut chunks) = (0usize, 0usize);
        loop {
            let rows = (t - c0).min(x_rows);
            e.upload_activation("act.x", &xin[c0 * hidden..(c0 + rows) * hidden])?;
            let (c1, tok) = match e.prefill_chunk(b, prompt, usize::MAX)? {
                plowrt::exec::gpu::PrefillStep::Progress(p) => (p, None),
                plowrt::exec::gpu::PrefillStep::Done(tok) => (t, Some(tok)),
            };
            let o = e.download_activation(out_name)?;
            out.extend_from_slice(&o[..(c1 - c0) * hidden]);
            c0 = c1;
            chunks += 1;
            if let Some(tok) = tok {
                return Ok((tok, chunks));
            }
        }
    }

    fn act_x_rows(e: &plowrt::exec::gpu::GpuEngine, hidden: usize) -> usize {
        e.tensor_bytes("act.x")
            .map_or(usize::MAX, |b| usize::try_from(b).unwrap_or(0) / 2 / hidden)
    }

    fn check(
        e: &mut plowrt::exec::gpu::GpuEngine,
        hidden: usize,
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
        let mut pf_out: Vec<f32> = Vec::new();
        let t = if e.has_prefill() {
            let prompt: Vec<u32> = (0..t as u32).map(|i| 100 + (i % 1000)).collect();
            let x_rows = act_x_rows(e, hidden);
            // `--repeat N`: the same real input N times, so a timing read (PLOW_PF_SEG_TIME)
            // can skip the cold first launch.
            let repeat: usize = flag("--repeat").and_then(|s| s.parse().ok()).unwrap_or(1);
            for _ in 0..repeat.max(1) {
                e.begin_slot(0, t + 1)?;
                let t0 = Instant::now();
                let (_, chunks) =
                    prefill_rows(e, 0, &prompt, &xin, hidden, x_rows, out_name, &mut pf_out)?;
                println!(
                    "  launched prefill(T={t}) in {chunks} chunk(s), {:.3} ms",
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

        let out = if pf_out.is_empty() {
            e.download_activation(out_name)?
        } else {
            std::mem::take(&mut pf_out)
        };
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
    ) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error>> {
        // `--decode-in rows.npy` ([n, hidden] f32): every decode step's act.x rows, cycled,
        // uploaded outside the timer (distinct rows route a MoE block like real traffic; the
        // default leaves act.x at the prefill's near-identical synthetic rows).
        let decode_in = match flag("--decode-in") {
            Some(path) => {
                let (shape, data) = npy::read_f32(Path::new(&path))?;
                if shape.len() != 2 || shape[1] != hidden || shape[0] == 0 {
                    return Err(format!("--decode-in {path}: want [n, {hidden}], got {shape:?}").into());
                }
                Some(data)
            }
            None => None,
        };
        let mut decode_step = 0usize;
        let mut feed_rows = |e: &mut plowrt::exec::gpu::GpuEngine,
                             bsz: usize|
         -> Result<(), Box<dyn std::error::Error>> {
            let Some(rows) = &decode_in else { return Ok(()) };
            let n = rows.len() / hidden;
            let mut x = Vec::with_capacity(bsz * hidden);
            for b in 0..bsz {
                let r = (decode_step * bsz + b) % n;
                x.extend_from_slice(&rows[r * hidden..(r + 1) * hidden]);
            }
            decode_step += 1;
            e.upload_activation("act.x", &x)?;
            Ok(())
        };
        let batches = flag("--batch")
            .map(|s| parse_list(&s))
            .unwrap_or_else(|| vec![1, 2, 4, 8]);
        let ctxs = flag("--ctx")
            .map(|s| parse_list(&s))
            .unwrap_or_else(|| vec![128, 512, 1024, 4096]);
        let iters: usize = flag("--iters").and_then(|s| s.parse().ok()).unwrap_or(100);
        let warmup: usize = flag("--warmup").and_then(|s| s.parse().ok()).unwrap_or(10);
        let fixed_ctx = match flag("--fixed-ctx").as_deref() {
            None | Some("0" | "false") => false,
            Some("1" | "true") => true,
            _ => return Err("--fixed-ctx expects 0 or 1".into()),
        };
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
                if fixed_ctx {
                    for b in 0..bsz {
                        e.snapshot_slot(b)?;
                    }
                    e.sync_stream()?;
                }
                for _ in 0..warmup {
                    feed_rows(e, bsz)?;
                    e.step_slots(&feeds(&last), &mut toks)?;
                    last.copy_from_slice(&toks);
                    if fixed_ctx {
                        for b in 0..bsz {
                            e.rewind_slot(b, t)?;
                        }
                        e.sync_stream()?;
                    }
                }
                e.trace_reset()?;
                let mut us: Vec<f64> = Vec::with_capacity(iters);
                for _ in 0..iters {
                    feed_rows(e, bsz)?;
                    let t0 = Instant::now();
                    e.step_slots(&feeds(&last), &mut toks)?;
                    us.push(t0.elapsed().as_secs_f64() * 1e6);
                    last.copy_from_slice(&toks);
                    if fixed_ctx {
                        for b in 0..bsz {
                            e.rewind_slot(b, t)?;
                        }
                        e.sync_stream()?;
                    }
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
                    "ctx": if fixed_ctx { t + 1 } else { t },
                    "prefix_ctx": t,
                    "context_mode": if fixed_ctx { "fixed" } else { "growing" },
                    "latency_us_median": (median * 100.0).round() / 100.0,
                    "latency_us_p95": (p95 * 100.0).round() / 100.0,
                    "tok_s": (tok_s * 10.0).round() / 10.0,
                    "prefill_ms_median": pf_med,
                    "prefill_ms_p95": pf_p95,
                    "prefill_tok_s": (pf_tok_s * 10.0).round() / 10.0,
                }));
            }
        }

        Ok(rows)
    }
}
