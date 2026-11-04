//! Kernel-only decode-step control for the serving-overhead audit
//! (campaign S5-serve-tuning): drives `GpuEngine` directly — no HTTP, no mux,
//! no spawn_blocking, no SSE — so `served TPOT − this` isolates the serving
//! layer from the kernel at any batch. Same engine code the server runs
//! (`prefill_slot` to build ctx, then one `step_slots` per token).
//!
//! Usage:
//!   step_bench <assets_dir> [slots] [ctx] [steps] [--same [--kv-image] | --spread] [--warmup N] [--multistep] [--packed-prefill]
//!              [--dump-tensors name,name --dump-dir dir | --dump-prefill-logits dir]
//!              [--max-inst N | --max-segments N]
//! `--same` feeds every slot the SAME prompt and reports how many slots' greedy
//! streams agree with slot 0 (a within-batch consistency check); with `--kv-image` slots 1..
//! take slot 0's KV through an exported image instead of prefilling (`KV_IMAGE` line).
//! `--kv-image-peer DIR --kv-image-peer-device amd:N|nv:N` instead checks a cross-vendor
//! roundtrip into peer slot 1; the main engine is the other vendor on device 0.
//! `--spread` gives each slot a pseudo-random prompt over 64K ids, so a MoE model routes a decode batch the way served
//! traffic does (the default prompts share one 1000-id cycle). `--packed-prefill`
//! initializes prompts through packed request chunks and the compact terminal, printing each
//! launch's wall ms; `--pf-chunk N` caps the per-request slice below the packet's request chunk and
//! `--pf-reps N` repeats the whole packed prefill (the first pass pays graph capture);
//! with `--dump-prefill-logits` the last rep's compact-terminal logits are written per slot.
//! `--dump-tensors`
//! writes the named tensors raw after the last step (block_run's format), which
//! with `--max-inst N`, one step and zero warmup gives partial decode activations;
//! with 0 steps and zero warmup it gives the activations right after prefill (the last layer's).
//! Instruction caps require native decode routes. `--multistep` times the
//! engine's device multi-step quanta (`PLOW_MULTISTEP=K`) instead of single steps; the digest
//! covers the same tokens in the same order, so it compares directly with a single-step run.
//! Env: PLOW_CHECKPOINT (default <assets>/checkpoint), PLOW_STEP_TIME=1 for
//! the engine's host-op breakdown.

#[cfg(any(feature = "cuda", test))]
fn record_inputs(histories: &mut [Vec<u32>], last: &[u32], outputs: &[u32], steps: usize) {
    for (row, history) in histories.iter_mut().enumerate() {
        history.push(last[row]);
        history.extend_from_slice(&outputs[row * steps..(row + 1) * steps - 1]);
    }
}

/// Write `--dump-tensors` entries and a manifest (`meta` plus the tensor table) into `dir`.
#[cfg(feature = "cuda")]
fn write_dumps(
    e: &plowrt::exec::gpu::GpuEngine,
    names: &str,
    dir: &std::path::Path,
    mut meta: serde_json::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let mut rows = Vec::new();
    for (index, spec) in names.split(',').map(str::trim).enumerate() {
        let (name, offset, bytes) = dump_range(e, spec)?;
        let mut raw = vec![0u8; usize::try_from(bytes)?];
        e.read_tensor_range(name, offset, &mut raw)?;
        let file = format!("tensor-{index:03}.bin");
        std::fs::write(dir.join(&file), raw)?;
        rows.push(
            serde_json::json!({"name": name, "offset": offset, "bytes": bytes, "file": file}),
        );
    }
    meta["tensors"] = rows.into();
    std::fs::write(dir.join("manifest.json"), serde_json::to_vec_pretty(&meta)?)?;
    println!("  wrote raw tensor dumps to {}", dir.display());
    Ok(())
}

/// A `--dump-tensors` entry: `name` (the whole tensor) or `name@OFFSET:BYTES` (one byte range,
/// e.g. one slot of a KV cache).
#[cfg(feature = "cuda")]
fn dump_range<'a>(
    e: &plowrt::exec::gpu::GpuEngine,
    spec: &'a str,
) -> Result<(&'a str, u64, u64), Box<dyn std::error::Error>> {
    let (name, range) = spec
        .split_once('@')
        .map_or((spec, None), |(n, r)| (n, Some(r)));
    let total = e
        .tensor_bytes(name)
        .ok_or_else(|| format!("unknown dump tensor {name:?}"))?;
    let Some(range) = range else {
        return Ok((name, 0, total));
    };
    let (offset, bytes) = range
        .split_once(':')
        .ok_or("--dump-tensors name@OFFSET:BYTES")?;
    let (offset, bytes): (u64, u64) = (offset.parse()?, bytes.parse()?);
    if offset.checked_add(bytes).is_none_or(|end| end > total) {
        return Err(format!("{spec}: range outside the {total}-byte tensor").into());
    }
    Ok((name, offset, bytes))
}

#[cfg(all(feature = "cuda", feature = "hsa"))]
struct StreamingImage {
    image: plowrt::serve::pd::KvBytes,
    first: u32,
    prefill_ms: f64,
    export_ms: f64,
    import_ms: f64,
    handoff_ms: f64,
    lease_start_ms: Option<f64>,
    intervals: [f64; 4],
}

#[cfg(all(feature = "cuda", feature = "hsa"))]
struct ExportedImage {
    first: u32,
    handoff_started: std::time::Instant,
    image: plowrt::serve::pd::KvBytes,
}

#[cfg(all(feature = "cuda", feature = "hsa"))]
fn produce_image<S>(
    source: &mut S,
    prefill: impl FnOnce(&mut S) -> plowrt::Result<u32>,
    export: impl FnOnce(
        &mut S,
        &mut plowrt::serve::pd::KvBytes,
        &mut dyn FnMut(plowrt::serve::pd::KvBytes),
    ) -> plowrt::Result<()>,
    streaming: bool,
    tx: std::sync::mpsc::SyncSender<ExportedImage>,
) -> plowrt::Result<(std::time::Instant, std::time::Instant, std::time::Instant)> {
    let first = prefill(source)?;
    let prefill_end = std::time::Instant::now();
    let handoff_started = std::time::Instant::now();
    let mut image = plowrt::serve::pd::KvBytes::default();
    export(source, &mut image, &mut |image| {
        let _ = tx.send(ExportedImage {
            first,
            handoff_started,
            image,
        });
    })?;
    let export_end = std::time::Instant::now();
    if !streaming {
        let _ = tx.send(ExportedImage {
            first,
            handoff_started,
            image,
        });
    }
    Ok((prefill_end, handoff_started, export_end))
}

#[cfg(all(feature = "cuda", feature = "hsa"))]
fn finish_image(
    producer: std::thread::ScopedJoinHandle<
        '_,
        plowrt::Result<(std::time::Instant, std::time::Instant, std::time::Instant)>,
    >,
    rx: std::sync::mpsc::Receiver<ExportedImage>,
    request_started: std::time::Instant,
    import: impl FnOnce(&plowrt::serve::pd::KvBytes, u32) -> plowrt::Result<()>,
) -> Result<StreamingImage, Box<dyn std::error::Error>> {
    let frame = rx.recv();
    let imported = match &frame {
        Ok(frame) => {
            let start = std::time::Instant::now();
            import(&frame.image, frame.first).map(|()| (start, std::time::Instant::now()))
        }
        Err(_) => Err(plowrt::RuntimeError::Unavailable(
            "source produced no staged image".into(),
        )),
    };
    let (prefill_end, export_start, export_end) =
        producer.join().map_err(|_| "KV source actor panicked")??;
    let (import_start, import_end) = imported?;
    let frame = frame?;
    let handoff_started = frame.handoff_started;
    let lease_start_ms = frame
        .image
        .stream_started()
        .map(|epoch| epoch.duration_since(handoff_started).as_secs_f64() * 1e3);
    Ok(StreamingImage {
        image: frame.image,
        first: frame.first,
        prefill_ms: prefill_end.duration_since(request_started).as_secs_f64() * 1e3,
        export_ms: export_end.duration_since(export_start).as_secs_f64() * 1e3,
        import_ms: import_end.duration_since(import_start).as_secs_f64() * 1e3,
        handoff_ms: handoff_started.elapsed().as_secs_f64() * 1e3,
        lease_start_ms,
        intervals: [
            export_start.duration_since(handoff_started).as_secs_f64() * 1e3,
            export_end.duration_since(handoff_started).as_secs_f64() * 1e3,
            import_start.duration_since(handoff_started).as_secs_f64() * 1e3,
            import_end.duration_since(handoff_started).as_secs_f64() * 1e3,
        ],
    })
}

#[cfg(all(feature = "cuda", feature = "hsa"))]
fn kv_image_peer_bench(
    assets: &std::path::Path,
    peer: &std::path::Path,
    device: &str,
    ctx: usize,
    prompt_file: Option<&str>,
    early_import: bool,
    stream_import: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::asset::serve::checkpoint_dir;
    use std::sync::Arc;
    use std::time::Instant;

    let (vendor, ordinal) = device
        .split_once(':')
        .ok_or("peer device must be amd:N or nv:N")?;
    let ordinal: u8 = ordinal.parse()?;
    let (nv_dir, amd_dir, nv_device, amd_device, main_nv) = match vendor {
        "amd" => (assets, peer, 0, ordinal, true),
        "nv" => (peer, assets, ordinal, 0, false),
        _ => return Err("peer device must be amd:N or nv:N".into()),
    };
    let mut digests = std::collections::HashMap::new();
    let headers = [assets, peer].map(|dir| plowrt::serve::pd::kv_header(dir, &mut digests));
    plowrt::serve::pd::check_headers(
        "step_bench",
        plowrt::serve::pd::PdSpec {
            prefill: 1,
            decode: 2,
        },
        &headers,
    )?;
    let be = Arc::new(plowrt::device::cuda::CudaBackend::new(nv_device)?);
    let mut nv = plowrt::exec::gpu::GpuEngine::load(be.clone(), nv_dir, &checkpoint_dir(nv_dir))?;
    let blob =
        plowrt::asset::devblob::DevBlob::find_in_dir(amd_dir)?.ok_or("AMD peer has no packet")?;
    let hsaco = amd_dir.join("hsaco");
    let hsaco = if hsaco.is_dir() {
        hsaco.as_path()
    } else {
        amd_dir
    };
    let mut amd = plowrt::serve::engine::AmdServe::load_on(
        &blob,
        hsaco,
        Some(&checkpoint_dir(amd_dir)),
        amd_device,
    )?;
    if nv.kv_image_key().is_none() || nv.kv_image_key() != amd.kv_image_key() {
        return Err("engines do not expose matching KV image keys".into());
    }
    // The peer imports into its last slot: a different slot index where it has one, so the
    // image cannot depend on the slot.
    let (nv_slot, amd_slot) = if main_nv {
        (0, amd.batch().saturating_sub(1))
    } else {
        (nv.batch().saturating_sub(1), 0)
    };
    if nv.batch() <= nv_slot
        || amd.batch() <= amd_slot
        || ctx == 0
        || ctx >= nv.max_ctx()
        || ctx >= amd.max_ctx()
    {
        return Err("KV image check needs room for the prompt plus one decode row".into());
    }
    let prompt: Vec<u32> = if let Some(file) = prompt_file {
        let prompts: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(file)?)?;
        prompts
            .first()
            .filter(|p| p.len() >= ctx)
            .ok_or("prompt file has no ctx-row prompt")?[..ctx]
            .to_vec()
    } else {
        (0..ctx).map(|i| 100 + (i % 1000) as u32).collect()
    };
    let n = u32::try_from(ctx)?;
    #[cfg(target_os = "linux")]
    {
        let bytes = [nv_dir, amd_dir]
            .into_iter()
            .map(|dir| plowrt::serve::pd::kv_image_capacity(dir, Some(n)))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .max()
            .unwrap();
        plowrt::device::host_region::prepare_image_pool(
            bytes,
            &[be, amd.kv_image_staging().ok_or("AMD staging unavailable")?],
        )?;
    }
    let request_started = Instant::now();
    let mut preparation = (0., 0.);
    let transferred = if main_nv {
        std::thread::scope(
            |scope| -> Result<StreamingImage, Box<dyn std::error::Error>> {
                let (tx, rx) = std::sync::mpsc::sync_channel(1);
                let producer = scope.spawn(|| {
                    produce_image(
                        &mut nv,
                        |source| {
                            source.begin_slot(nv_slot, ctx + 1)?;
                            if source.has_prefill() {
                                source.prefill_slot(nv_slot, &prompt)
                            } else {
                                source.consume_prompt(nv_slot, &prompt, &mut Vec::new())
                            }
                        },
                        |source, out, publish| {
                            if stream_import {
                                source.export_kv_image_stream(nv_slot, n, out, publish)
                            } else {
                                source.export_kv_image(nv_slot, n, out)
                            }
                        },
                        stream_import,
                        tx,
                    )
                });
                finish_image(producer, rx, request_started, |image, first| {
                    amd.import_kv_image(amd_slot, &prompt, image, first)
                })
            },
        )?
    } else {
        std::thread::scope(
            |scope| -> Result<StreamingImage, Box<dyn std::error::Error>> {
                let (tx, rx) = std::sync::mpsc::sync_channel(1);
                let producer = scope.spawn(|| {
                    produce_image(
                        &mut amd,
                        |source| source.prefill(amd_slot, &prompt),
                        |source, out, publish| {
                            if stream_import {
                                source.export_kv_image_stream(amd_slot, n, out, publish)
                            } else {
                                source.export_kv_image(amd_slot, n, out)
                            }
                        },
                        stream_import,
                        tx,
                    )
                });
                if early_import {
                    let start = request_started.elapsed().as_secs_f64() * 1e3;
                    if let Err(error) = nv.prepare_kv_import(nv_slot, ctx, ctx + 1) {
                        producer.join().map_err(|_| "KV source actor panicked")??;
                        return Err(error.into());
                    }
                    preparation = (start, request_started.elapsed().as_secs_f64() * 1e3);
                }
                finish_image(producer, rx, request_started, |image, _| {
                    if !early_import {
                        nv.begin_slot(nv_slot, ctx + 1)?;
                    }
                    nv.import_kv_image(nv_slot, &prompt, image)
                })
            },
        )?
    };
    let first = transferred.first;
    let prefill_finished = transferred.prefill_ms;
    let export_ms = transferred.export_ms;
    let import_ms = transferred.import_ms;
    let pipeline = if stream_import {
        Some((
            transferred.handoff_ms,
            transferred
                .lease_start_ms
                .ok_or("stream image has no source epoch")?,
            transferred.intervals,
        ))
    } else {
        None
    };
    let a = transferred.image;
    let mut b = plowrt::serve::pd::KvBytes::default();
    let import_ready = request_started.elapsed().as_secs_f64() * 1e3;
    if !a.is_staged() {
        return Err("KV image peer benchmark requires a staging lease".into());
    }
    if main_nv {
        amd.export_kv_image(amd_slot, n, &mut b)?;
    } else {
        nv.export_kv_image(nv_slot, n, &mut b)?;
    }
    println!(
        "KV_IMAGE_XV bytes={} exact={} staged={} export_ms={export_ms:.4} import_ms={import_ms:.4}",
        a.len(),
        u8::from(*a == *b),
        u8::from(a.is_staged())
    );
    if *a != *b {
        return Err("cross-vendor KV image re-export differs".into());
    }
    let mut tokens = Vec::new();
    nv.step_slots(&[(nv_slot, first)], &mut tokens)?;
    let amd_token = amd.step(amd_slot, first)?;
    let (main_token, peer_token) = if main_nv {
        (tokens[0], amd_token)
    } else {
        (amd_token, tokens[0])
    };
    println!("KV_IMAGE_XV_TOKENS prefill={first} main={main_token} peer={peer_token}");
    println!("KV_IMAGE_XV_PHASES early={} prefill_ms={prefill_finished:.4} prepare_start_ms={:.4} prepare_end_ms={:.4} import_ready_ms={import_ready:.4} verified_decode_ms={:.4}",
        u8::from(early_import), preparation.0, preparation.1, request_started.elapsed().as_secs_f64() * 1e3);
    if let Some((
        handoff_ms,
        lease_start_ms,
        [export_start, export_end, import_start, import_end],
    )) = pipeline
    {
        println!("KV_IMAGE_XV_PIPE stream=1 handoff_ms={handoff_ms:.4} lease_start_ms={lease_start_ms:.4} export_start_ms={export_start:.4} export_end_ms={export_end:.4} import_start_ms={import_start:.4} import_end_ms={import_end:.4}");
    }
    if main_token != peer_token {
        return Err(format!("KV image peer decode differs: {main_token} vs {peer_token}").into());
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("step_bench requires --features cuda");
    std::process::exit(2);
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::time::Instant;

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let assets = std::path::PathBuf::from(
        args.next()
            .ok_or("usage: step_bench <assets> [slots] [ctx] [steps] [--same] [--warmup N] [--multistep] [--dump-tensors a,b --dump-dir d]")?,
    );
    let want_slots: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(1);
    let ctx: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(4137);
    let steps: usize = args.next().map(|v| v.parse()).transpose()?.unwrap_or(128);
    let mut same = false;
    let mut kv_image = false;
    let mut kv_image_prepare = false;
    let mut kv_image_stream = false;
    let mut kv_image_peer = None::<std::path::PathBuf>;
    let mut kv_image_peer_device = None::<String>;
    let mut spread = false;
    let mut multistep = false;
    let mut packed_prefill = false;
    let mut pf_chunk = usize::MAX;
    let mut pf_reps = 1usize;
    let mut warmup = 16usize;
    let mut sweep: Option<(u32, u32)> = None;
    let mut max_inst: Option<u32> = None;
    let mut max_segments: Option<usize> = None;
    let mut seg_sweep: Option<(usize, usize, usize)> = None;
    let mut prefill_logits_dir = None::<std::path::PathBuf>;
    let mut ride: Option<(usize, Vec<usize>)> = None;
    let mut ride_dump = None::<std::path::PathBuf>;
    let mut ride_prefix = 0usize;
    let mut ride_salt = 0u32;
    let (mut dump_names, mut dump_dir) = (None::<String>, None::<String>);
    let mut spec = SpecArgs::default();
    let mut prompt_file = None::<String>;
    while let Some(a) = args.next() {
        match a.as_str() {
            // `--spec oracle|adversarial|ngram`: speculative verify identity gate (see spec_bench).
            "--spec" => spec.mode = Some(args.next().ok_or("--spec MODE")?),
            "--spec-k" => spec.k = args.next().ok_or("--spec-k K")?.parse()?,
            "--spec-rung" => spec.rung = args.next().ok_or("--spec-rung R")?.parse()?,
            "--spec-poison" => spec.poison = true,
            // `--spec-sample T,TOP_K,TOP_P,MIN_P`: sampled verify (see spec_bench); `--spec-host`
            // samples the verify rows on the host instead of with `plow_sample`.
            "--spec-sample" => {
                let v: Vec<f32> = args
                    .next()
                    .ok_or("--spec-sample T,TOP_K,TOP_P,MIN_P")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                let [temp, top_k, top_p, min_p] = v[..] else {
                    return Err("--spec-sample T,TOP_K,TOP_P,MIN_P".into());
                };
                spec.sample = Some((temp, top_k as usize, top_p, min_p));
            }
            "--spec-host" => spec.host = true,
            // `--prompt-tokens FILE`: JSON array of token-id prompts; slot b takes prompt b mod n,
            // which must hold at least CTX tokens (truncated to CTX).
            "--prompt-tokens" => prompt_file = Some(args.next().ok_or("--prompt-tokens FILE")?),
            // `--draft-bench ITERS`: time the packet's draft programs at k = 1..=kmax (see draft_bench).
            "--draft-bench" => {
                spec.draft_iters = args.next().ok_or("--draft-bench ITERS")?.parse()?
            }
            "--nsplit-kvlen" => {
                spec.nsplit_kvlen = Some(args.next().ok_or("--nsplit-kvlen N")?.parse()?)
            }
            "--max-segments" => {
                max_segments = Some(args.next().ok_or("--max-segments N")?.parse()?)
            }
            "--max-inst" => max_inst = Some(args.next().ok_or("--max-inst N")?.parse()?),
            "--same" => same = true,
            "--kv-image" => kv_image = true,
            "--kv-image-prepare" => kv_image_prepare = true,
            "--kv-image-stream" => kv_image_stream = true,
            "--kv-image-peer" => {
                kv_image_peer = Some(args.next().ok_or("--kv-image-peer DIR")?.into())
            }
            "--kv-image-peer-device" => {
                kv_image_peer_device = Some(args.next().ok_or("--kv-image-peer-device amd:N|nv:N")?)
            }
            "--spread" => spread = true,
            "--packed-prefill" => packed_prefill = true,
            "--pf-chunk" => pf_chunk = args.next().ok_or("--pf-chunk N")?.parse()?,
            "--pf-reps" => pf_reps = args.next().ok_or("--pf-reps N")?.parse()?,
            "--multistep" => multistep = true,
            "--warmup" => warmup = args.next().ok_or("--warmup N")?.parse()?,
            // `--sweep LO..HI`: time decode steps with instruction caps LO..=HI in this process.
            "--sweep" => {
                let r = args.next().ok_or("--sweep LO..HI")?;
                let (lo, hi) = r.split_once("..").ok_or("--sweep LO..HI")?;
                sweep = Some((lo.parse()?, hi.parse()?));
            }
            // `--seg-sweep LO..HI[:STEP]`: time the widest library-routed decode rung captured up to
            // segment n for n in LO..=HI (the per-segment profile instruction caps cannot take).
            "--seg-sweep" => {
                let r = args.next().ok_or("--seg-sweep LO..HI[:STEP]")?;
                let (r, step) = r.split_once(':').unwrap_or((r.as_str(), "1"));
                let (lo, hi) = r.split_once("..").ok_or("--seg-sweep LO..HI[:STEP]")?;
                seg_sweep = Some((lo.parse()?, hi.parse()?, step.parse()?));
            }
            "--dump-tensors" => dump_names = Some(args.next().ok_or("--dump-tensors a,b")?),
            "--dump-dir" => dump_dir = Some(args.next().ok_or("--dump-dir d")?),
            "--dump-prefill-logits" => {
                prefill_logits_dir = Some(args.next().ok_or("--dump-prefill-logits dir")?.into())
            }
            "--ride" => {
                let rows = args.next().ok_or("--ride ROWS W1,W2,..")?.parse()?;
                let widths = args
                    .next()
                    .ok_or("--ride ROWS W1,W2,..")?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<Vec<usize>, _>>()?;
                ride = Some((rows, widths));
            }
            "--ride-dump" => ride_dump = Some(args.next().ok_or("--ride-dump dir")?.into()),
            "--ride-prefix" => ride_prefix = args.next().ok_or("--ride-prefix N")?.parse()?,
            "--ride-salt" => ride_salt = args.next().ok_or("--ride-salt N")?.parse()?,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let dumps = match (dump_names, dump_dir) {
        (None, None) => None,
        (Some(n), Some(d)) => Some((n, std::path::PathBuf::from(d))),
        _ => return Err("--dump-tensors and --dump-dir must be provided together".into()),
    };

    if max_inst.is_some() && max_segments.is_some() {
        return Err("choose --max-inst or --max-segments".into());
    }
    let partial = max_inst.is_some() || max_segments.is_some();
    if prefill_logits_dir.is_some() && (partial || dumps.is_some() || sweep.is_some() || multistep)
    {
        return Err("--dump-prefill-logits captures prefill only".into());
    }
    if partial && (steps != 1 || warmup != 0 || multistep || sweep.is_some() || dumps.is_none()) {
        return Err(
            "partial capture requires one step, --warmup 0, tensor dumps, and no sweep/multistep"
                .into(),
        );
    }

    if kv_image_peer.is_some() || kv_image_peer_device.is_some() {
        if kv_image_stream && !kv_image_prepare {
            return Err("--kv-image-stream requires --kv-image-prepare".into());
        }
        let (Some(peer), Some(device)) = (kv_image_peer, kv_image_peer_device) else {
            return Err("--kv-image-peer and --kv-image-peer-device are required together".into());
        };
        if !kv_image || !same || packed_prefill || warmup != 0 || steps == 0 || want_slots < 2 {
            return Err("cross-vendor KV image check needs --same --kv-image, 2+ slots, --warmup 0 and steps".into());
        }
        if spread
            || multistep
            || partial
            || dumps.is_some()
            || prefill_logits_dir.is_some()
            || sweep.is_some()
            || seg_sweep.is_some()
            || ride.is_some()
            || spec.mode.is_some()
            || spec.draft_iters > 0
        {
            return Err(
                "cross-vendor KV image check runs only the roundtrip and one decode step".into(),
            );
        }
        #[cfg(feature = "hsa")]
        return kv_image_peer_bench(
            &assets,
            &peer,
            &device,
            ctx,
            prompt_file.as_deref(),
            kv_image_prepare,
            kv_image_stream,
        );
        #[cfg(not(feature = "hsa"))]
        return Err("--kv-image-peer requires --features cuda,hsa".into());
    }

    if kv_image_prepare || kv_image_stream {
        return Err("--kv-image-prepare requires the cross-vendor KV image check".into());
    }

    let ckpt = std::env::var("PLOW_CHECKPOINT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| assets.join("checkpoint"));
    let be = Arc::new(plowrt::device::cuda::CudaBackend::new(0)?);
    let mut e = plowrt::exec::gpu::GpuEngine::load(be, &assets, &ckpt)?;
    let slots = want_slots.min(e.batch());
    if prefill_logits_dir.is_some() && !e.has_prefill() {
        return Err("prefill logit capture requires GPU prefill".into());
    }
    if (max_segments.is_some() || seg_sweep.is_some()) && slots != e.batch() {
        return Err("--max-segments requires the widest decode rung".into());
    }
    println!(
        "engine batch={} vocab={} max_ctx={} prefill={} -> slots={slots} ctx={ctx} steps={steps}",
        e.batch(),
        e.vocab(),
        e.max_ctx(),
        e.has_prefill()
    );

    // Synthetic prompts, ids in-vocab. One PER SLOT: identical rows route to the same top-k
    // experts, so a MoE model's B>1 step touched 8 experts where serving touches ~60 (slot 0
    // keeps the historical prompt, so B=1 numbers are unchanged). `--same` gives every slot
    // slot 0's prompt on purpose.
    let mut last = vec![0u32; slots];
    let mut histories = dumps.as_ref().map(|_| vec![Vec::new(); slots]);
    let mut prefill_cases = Vec::new();
    let mut prefill_raw = prefill_logits_dir
        .as_ref()
        .map(|dir| {
            std::fs::create_dir(dir)?;
            Ok::<_, Box<dyn std::error::Error>>(vec![0u8; e.vocab() * 2])
        })
        .transpose()?;
    let prefill_case = |prompt: &[u32], b: usize, file: String, token: u32| {
        let prompt_bytes: Vec<u8> = prompt.iter().flat_map(|id| id.to_le_bytes()).collect();
        serde_json::json!({
            "id": format!("prefill-s{b:03}"), "file": file, "dtype": "bf16",
            "prompt_token_ids": prompt, "prompt_len": prompt.len(),
            "prompt_sha256_u32le": plow_asset::knob::sha256_hex(&prompt_bytes),
            "sampled_token_id": token, "generation_step": 0,
            "execution_phase": "prefill_output",
        })
    };
    let file_prompts: Option<Vec<Vec<u32>>> = prompt_file
        .map(|f| -> Result<_, Box<dyn std::error::Error>> {
            let p: Vec<Vec<u32>> = serde_json::from_slice(&std::fs::read(&f)?)?;
            if p.is_empty() || p.iter().any(|x| x.len() < ctx) {
                return Err(format!("{f}: every prompt needs at least {ctx} tokens").into());
            }
            Ok(p)
        })
        .transpose()?;
    let prompt_for = |b: usize| -> Vec<u32> {
        if let Some(p) = &file_prompts {
            return p[b % p.len()][..ctx].to_vec();
        }
        let bb = if same { 0 } else { b as u32 };
        if spread {
            let mut x = 0x9e37_79b9u32 ^ (bb + 1).wrapping_mul(0x85eb_ca6b);
            return (0..ctx)
                .map(|_| {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    1000 + x % 65536
                })
                .collect();
        }
        (0..ctx as u32)
            .map(|i| 100 + ((i + 131 * bb) % 1000))
            .collect()
    };
    if packed_prefill {
        use plowrt::exec::gpu::PfBatchReq;
        if slots == 0 || ctx == 0 || !e.pf_batch_enabled() || !e.has_packed_terminal() {
            return Err(
                "--packed-prefill requires nonempty prompts, packed prefill and compact terminal"
                    .into(),
            );
        }
        let prompts: Vec<_> = (0..slots).map(prompt_for).collect();
        let chunk = e.pf_request_max_rows().min(e.pf_max_rows()).min(pf_chunk);
        if chunk == 0 {
            return Err("packed prefill has no chunk capacity".into());
        }
        let width = e.pf_max_rows() / chunk;
        let mut completed = Vec::new();
        let mut logits_row = Vec::new();
        for rep in 0..pf_reps.max(1) {
            for b in 0..slots {
                if rep > 0 {
                    e.retire_slot(b, false);
                }
                e.begin_slot(b, ctx + steps + 1)?;
                if let Some(histories) = &mut histories {
                    histories[b] = prompts[b].clone();
                }
            }
            let t0 = Instant::now();
            for c0 in (0..ctx).step_by(chunk) {
                let len = chunk.min(ctx - c0);
                for first in (0..slots).step_by(width) {
                    let end = (first + width).min(slots);
                    let requests: Vec<_> = (first..end)
                        .map(|slot| PfBatchReq {
                            slot,
                            prompt: &prompts[slot],
                            c0,
                            len,
                        })
                        .collect();
                    let t = Instant::now();
                    e.prefill_batched_complete(&requests, &mut completed)?;
                    println!("packed prefill: rep={rep} prefix={c0} chunk_rows={len} requests={} rows={} ms={:.3}",
                             requests.len(), requests.len() * len, t.elapsed().as_secs_f64() * 1e3);
                    if c0 + len == ctx && completed.len() != requests.len() {
                        return Err("packed prefill did not complete every request".into());
                    }
                    for (row, &(slot, token)) in completed.iter().enumerate() {
                        last[slot] = token;
                        // The compact terminal leaves row k's logits for completed[k]; the last
                        // rep's are the ones its timed launches produced.
                        if let (Some(dir), true) = (&prefill_logits_dir, rep + 1 == pf_reps.max(1))
                        {
                            e.logits_row(row, &mut logits_row)?;
                            let raw: Vec<u8> = logits_row
                                .iter()
                                .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
                                .collect();
                            let file = format!("prefill-s{slot:03}.bin");
                            std::fs::write(dir.join(&file), raw)?;
                            prefill_cases.push(prefill_case(&prompts[slot], slot, file, token));
                        }
                    }
                }
            }
            println!(
                "packed prompts consumed in {:.4} s (rep {rep}, first tokens {:?})",
                t0.elapsed().as_secs_f64(),
                &last[..slots.min(8)]
            );
        }
    } else {
        // `--kv-image`: only slot 0 prefills; the others are seated from its image below.
        for b in 0..if kv_image { 1 } else { slots } {
            let prompt = prompt_for(b);
            if let Some(histories) = &mut histories {
                histories[b] = prompt.clone();
            }
            e.begin_slot(b, ctx + steps + 1)?;
            let t0 = Instant::now();
            last[b] = if e.has_prefill() {
                e.prefill_slot(b, &prompt)?
            } else {
                let mut toks = Vec::new();
                e.consume_prompt(b, &prompt, &mut toks)?
            };
            if let (Some(dir), Some(raw)) = (&prefill_logits_dir, &mut prefill_raw) {
                e.read_tensor_range("act.logits", 0, raw)?;
                let file = format!("prefill-s{b:03}.bin");
                std::fs::write(dir.join(&file), raw)?;
                prefill_cases.push(prefill_case(&prompt, b, file, last[b]));
            }
            println!(
                "slot {b}: prompt consumed in {:.4} s",
                t0.elapsed().as_secs_f64()
            );
        }
    }
    if kv_image {
        // Slots 1.. take slot 0's KV through an exported image instead of their own prefill:
        // re-exporting each must give the same bytes, and SLOTS_AGREE then compares the decodes.
        if !same || packed_prefill || slots < 2 || warmup != 0 || steps == 0 {
            return Err(
                "--kv-image needs --same, the serial prefill, 2+ slots, --warmup 0 and steps"
                    .into(),
            );
        }
        let n = prompt_for(0).len() as u32;
        let (mut image, mut again) = (
            plowrt::serve::pd::KvBytes::default(),
            plowrt::serve::pd::KvBytes::default(),
        );
        let t0 = Instant::now();
        e.export_kv_image(0, n, &mut image)?;
        let export_s = t0.elapsed().as_secs_f64();
        let mut exact = 0;
        for b in 1..slots {
            e.begin_slot(b, ctx + steps + 1)?;
            e.import_kv_image(b, &prompt_for(b), &image)?;
            last[b] = last[0];
            e.export_kv_image(b, n, &mut again)?;
            exact += usize::from(*again == *image);
        }
        println!(
            "KV_IMAGE rows={n} bytes={} export_s={export_s:.4} roundtrip_exact={exact}/{} key={}",
            image.len(),
            slots - 1,
            e.kv_image_key().unwrap_or_default()
        );
        if exact != slots - 1 {
            return Err("KV image re-export differs from the exported image".into());
        }
    }
    if let Some(dir) = &prefill_logits_dir {
        std::fs::write(
            dir.join("manifest.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "schema": 1, "producer": "plow-step-bench-prefill", "name": "step-bench-prefill",
                "vocab_size": e.vocab(), "cases": prefill_cases,
            }))?,
        )?;
        return Ok(());
    }

    if let Some((rows, widths)) = &ride {
        return ride_bench(
            &mut e,
            &last,
            ctx,
            steps,
            *rows,
            widths,
            ride_prefix,
            ride_salt,
            ride_dump.as_deref(),
        );
    }

    if let Some(segments) = max_segments {
        e.capture_debug_decode_prefix(segments)?;
    }
    if let Some(cap) = max_inst {
        if !e.set_debug_max_inst(cap, slots)? {
            return Err("decode object has no instruction-cap global".into());
        }
    }

    // Warmup (repo convention: discard 16), then timed steps.
    let feeds_of = |last: &[u32]| -> Vec<(usize, u32)> {
        last.iter().enumerate().map(|(b, &t)| (b, t)).collect()
    };
    let mut toks = Vec::new();
    for _ in 0..warmup {
        e.step_slots(&feeds_of(&last), &mut toks)?;
        if let Some(histories) = &mut histories {
            record_inputs(histories, &last, &toks, 1);
        }
        last.copy_from_slice(&toks);
    }
    if spec.draft_iters > 0 {
        return draft_bench(&mut e, &assets, &last, spec.draft_iters);
    }
    if let Some(mode) = spec.mode.clone() {
        let prompts: Vec<Vec<u32>> = (0..slots).map(prompt_for).collect();
        let verdict = spec_bench(
            &mut e,
            &assets,
            &spec,
            &mode,
            &prompts,
            &last,
            ctx + warmup,
            steps,
        );
        if let Some((names, dir)) = &dumps {
            let pos: Vec<usize> = (0..slots).map(|b| e.slot_pos(b)).collect();
            write_dumps(
                &e,
                names,
                std::path::Path::new(dir),
                serde_json::json!({"scope": "raw allocations after the spec run", "mode": mode,
                                   "slots": slots, "ctx": ctx, "steps": steps, "warmup": warmup,
                                   "slot_pos": pos}),
            )?;
        }
        return verdict;
    }
    if let Some((lo, hi, step)) = seg_sweep {
        let base_pos: Vec<usize> = (0..slots).map(|_| ctx + warmup).collect();
        for b in 0..slots {
            e.snapshot_slot(b)?;
        }
        let base_last = last.clone();
        for n in (lo..=hi).step_by(step.max(1)) {
            e.capture_debug_decode_prefix(n)?;
            for (b, &p) in base_pos.iter().enumerate() {
                e.rewind_slot(b, p)?;
            }
            last.copy_from_slice(&base_last);
            for _ in 0..2 {
                e.step_slots(&feeds_of(&last), &mut toks)?;
            }
            let mut v: Vec<f64> = (0..steps)
                .map(|_| {
                    let t0 = Instant::now();
                    e.step_slots(&feeds_of(&last), &mut toks)
                        .map(|_| t0.elapsed().as_secs_f64() * 1e3)
                })
                .collect::<Result<_, _>>()?;
            v.sort_by(f64::total_cmp);
            println!("{{\"seg\":{n},\"ms\":{:.4}}}", v[v.len() / 2]);
        }
        return Ok(());
    }
    if let Some((lo, hi)) = sweep {
        let base_pos: Vec<usize> = (0..slots).map(|_| ctx + warmup).collect();
        // Recurrent (GDN) state is restored with the position at every cap.
        for b in 0..slots {
            e.snapshot_slot(b)?;
        }
        let base_last = last.clone();
        for cap in (lo..=hi).chain([u32::MAX]) {
            e.set_debug_max_inst(cap, slots)?;
            // Same kv length at every cap: a drifting position biases each delta by the
            // attention's per-token cost.
            for (b, &p) in base_pos.iter().enumerate() {
                e.rewind_slot(b, p)?;
            }
            last.copy_from_slice(&base_last);
            for _ in 0..4 {
                e.step_slots(&feeds_of(&last), &mut toks)?;
            }
            let mut v: Vec<f64> = (0..steps)
                .map(|_| {
                    let t0 = Instant::now();
                    e.step_slots(&feeds_of(&last), &mut toks)
                        .map(|_| t0.elapsed().as_secs_f64() * 1e3)
                })
                .collect::<Result<_, _>>()?;
            v.sort_by(f64::total_cmp);
            println!(
                "{{\"cap\":{},\"ms\":{:.4}}}",
                if cap == u32::MAX { -1 } else { cap as i64 },
                v[v.len() / 2]
            );
        }
        return Ok(());
    }
    // Drop prefill + warmup from the trace so the profile is timed-decode only.
    e.trace_reset()?;
    let mut ms: Vec<f64> = Vec::with_capacity(steps);
    // Greedy stream digest (FNV-1a over every slot's tokens): two decode objects on one packet
    // can be compared for token agreement without a server.
    let mut digest: u64 = 0xcbf29ce484222325;
    let mut streams: Vec<Vec<u32>> = vec![Vec::with_capacity(steps); slots];
    if multistep && e.multistep_quantum().is_none() {
        return Err("--multistep needs an engine with PLOW_MULTISTEP >= 2".into());
    }
    let mut done = 0;
    while done < steps {
        let t0 = Instant::now();
        if multistep {
            // Fed row r is slot r; `toks` is row-major, K tokens per row.
            let k = e.multi_step_at_most(&feeds_of(&last), steps - done, &mut toks)?;
            let per = t0.elapsed().as_secs_f64() * 1e3 / k as f64;
            if let Some(histories) = &mut histories {
                record_inputs(histories, &last, &toks, k);
            }
            for step in 0..k {
                ms.push(per);
                for (b, stream) in streams.iter_mut().enumerate() {
                    let t = toks[b * k + step];
                    digest = (digest ^ t as u64).wrapping_mul(0x100000001b3);
                    stream.push(t);
                }
            }
            for (b, l) in last.iter_mut().enumerate() {
                *l = toks[b * k + k - 1];
            }
            done += k;
            continue;
        }
        e.step_slots(&feeds_of(&last), &mut toks)?;
        ms.push(t0.elapsed().as_secs_f64() * 1e3);
        if let Some(histories) = &mut histories {
            record_inputs(histories, &last, &toks, 1);
        }
        last.copy_from_slice(&toks);
        for (b, &t) in toks.iter().enumerate() {
            digest = (digest ^ t as u64).wrapping_mul(0x100000001b3);
            streams[b].push(t);
        }
        done += 1;
    }
    if !partial {
        println!(
            "TOK_STREAM slots={slots} ctx={ctx} fnv={digest:016x} slot0={:?}",
            streams[0]
        );
    }
    if same && !partial {
        let agree = streams.iter().filter(|s| **s == streams[0]).count();
        println!("SLOTS_AGREE {agree}/{slots} (identical prompts on every slot)");
        if kv_image && agree != slots {
            return Err("slots seated from a KV image decode differently".into());
        }
        for (b, s) in streams.iter().enumerate().skip(1) {
            if *s != streams[0] {
                println!("  slot{b}={s:?}");
            }
        }
    }
    // 0 steps: no timing; a dump then holds the allocations right after prefill.
    if !ms.is_empty() {
        ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = ms.iter().sum::<f64>() / ms.len() as f64;
        let median = ms[ms.len() / 2];
        let sd = if ms.len() > 1 {
            (ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (ms.len() - 1) as f64).sqrt()
        } else {
            0.0
        };
        let measurement = if partial { "PARTIAL_STEP" } else { "RAW_STEP" };
        println!(
            "{measurement} slots={slots} ctx={ctx} n={} mean_ms={mean:.3} median_ms={median:.3} \
             sd_ms={sd:.3} min_ms={:.3} max_ms={:.3} per_user_tok_s={:.1} aggregate_tok_s={:.1}",
            ms.len(),
            ms[0],
            ms[ms.len() - 1],
            1000.0 / mean,
            1000.0 / mean * slots as f64,
        );
    }

    if let Some((names, dir)) = dumps {
        write_dumps(
            &e,
            &names,
            std::path::Path::new(&dir),
            serde_json::json!({
                "scope": if partial { "raw allocations after partial decode; outputs may be stale" }
                    else if steps == 0 { "raw complete allocations after prefill" }
                    else { "raw complete allocations after the last decode step" },
                "max_inst": max_inst, "max_segments": max_segments,
                "slots": slots, "ctx": ctx, "steps": steps, "warmup": warmup,
                "packed_prefill": packed_prefill,
                "token_histories": histories,
                "sampled_token_ids": if !partial { Some(last) } else { None },
            }),
        )?;
    }

    // Stage-7 profile: with a -DPLOW_NV_TRACE=1 decode cubin, dump block 0's
    // per-opcode gate/body/signal cycle attribution (None on a normal cubin).
    if let Some(profile) = e.trace_summary()? {
        println!("{profile}");
    }
    Ok(())
}

/// `--ride ROWS W1,W2,..`: the unified token batch's per-rider cost. Slots `0..W` (at `ctx`)
/// ride one packed launch of ROWS fresh prompt rows (intermediate chunks on the slots after the
/// prefilled ones, nothing sampled) against the same launch alone and a standalone decode step of
/// `W` rows. Every arm rewinds the slots first, so context and inputs are fixed; arms interleave
/// per iteration. The riders' logits are compared in process with the standalone step's;
/// `--ride-dump DIR` also writes both (bf16, `W x vocab`) for cross-run comparison.
/// `--ride-prefix N`: the prefill slots first hold N prompt tokens (a cached prefix), and every
/// arm rewinds them to N, so the launch's rows attend to N + their own rows.
/// `--ride-salt S`: other prefill token ids (a rider's logits must not depend on its launch's
/// other rows; compare `--ride-dump`s across salts).
#[cfg(feature = "cuda")]
fn ride_bench(
    e: &mut plowrt::exec::gpu::GpuEngine,
    last: &[u32],
    ctx: usize,
    iters: usize,
    rows: usize,
    widths: &[usize],
    prefix: usize,
    salt: u32,
    dump: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    use plow_asset::token_batch::{Phase, Request, Selection};
    use std::time::Instant;
    let first = last.len();
    let wmax = *widths.iter().max().ok_or("--ride needs a width")?;
    let chunk = e
        .pf_request_max_rows()
        .min(e.pf_max_rows())
        .min(rows)
        .max(1);
    let fills: Vec<usize> = (0..rows.div_ceil(chunk))
        .map(|i| chunk.min(rows - i * chunk))
        .collect();
    if !e.token_batch_enabled() || wmax > first || first + fills.len() > e.batch() {
        return Err(
            "--ride needs the token batch, widths <= slots, slots + prefill requests <= batch"
                .into(),
        );
    }
    let prompt: Vec<u32> = (0..2 * chunk as u32)
        .map(|i| 1000 + (i * 7919).wrapping_add(salt.wrapping_mul(104_729)) % 50000)
        .collect();
    let head: Vec<u32> = (0..prefix as u32)
        .map(|i| 2000 + (i * 6007) % 50000)
        .collect();
    for i in 0..fills.len() {
        e.begin_slot(first + i, prefix + prompt.len() + 1)?;
    }
    if prefix > 0 {
        let generations: Vec<u32> = (0..fills.len())
            .map(|i| e.slot_generation(first + i).expect("slot"))
            .collect();
        let mut out = Vec::new();
        for c0 in (0..prefix).step_by(chunk) {
            let part = &head[c0..(c0 + chunk).min(prefix)];
            let reqs: Vec<Request<'_>> = (0..fills.len())
                .map(|i| Request {
                    id: (first + i) as u32,
                    slot: (first + i) as u32,
                    state_slot: (first + i) as u32,
                    generation: generations[i],
                    phase: Phase::Prefill,
                    tokens: part,
                    prompt_len: (prefix + prompt.len()) as u32,
                    selection: Selection::default(),
                })
                .collect();
            e.token_batch_step(&reqs, &mut out)?;
        }
    }
    // A recurrent chain rewinds through the state snapshot taken at the rewind position: the
    // decode slots at ctx, the prefill slots fresh at 0.
    for b in 0..wmax {
        e.snapshot_slot(b)?;
    }
    for i in 0..fills.len() {
        e.snapshot_slot(first + i)?;
    }
    if let Some(dir) = dump {
        std::fs::create_dir_all(dir)?;
    }
    let reset = |e: &mut plowrt::exec::gpu::GpuEngine| -> Result<(), Box<dyn std::error::Error>> {
        for b in 0..wmax {
            e.rewind_slot(b, ctx)?;
        }
        for i in 0..fills.len() {
            e.rewind_slot(first + i, prefix)?;
        }
        Ok(())
    };
    let generations: Vec<u32> = (0..first + fills.len())
        .map(|s| e.slot_generation(s).expect("slot"))
        .collect();
    let requests = |w: usize| {
        let request = |slot: usize, phase, tokens| Request {
            id: slot as u32,
            slot: slot as u32,
            state_slot: slot as u32,
            generation: generations[slot],
            phase,
            tokens,
            prompt_len: match phase {
                Phase::Decode | Phase::Verify => ctx as u32,
                Phase::Prefill => (prefix + prompt.len()) as u32,
            },
            selection: Selection::default(),
        };
        (0..w)
            .map(|b| request(b, Phase::Decode, std::slice::from_ref(&last[b])))
            .chain(
                fills
                    .iter()
                    .enumerate()
                    .map(|(i, &n)| request(first + i, Phase::Prefill, &prompt[..n])),
            )
            .collect::<Vec<Request<'_>>>()
    };
    let feeds: Vec<(usize, u32)> = last.iter().copied().enumerate().collect();
    println!("RIDE_INPUTS ids={:?}", &last[..wmax]);
    let (mut output, mut toks, mut logits) = (Vec::new(), Vec::new(), Vec::new());
    let median = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    let bf16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    };
    let (mut pure, warm) = (Vec::new(), 4);
    for &w in widths {
        let (mut ride, mut step) = (Vec::new(), Vec::new());
        for it in 0..warm + iters {
            reset(e)?;
            let t = Instant::now();
            e.token_batch_step(&requests(0), &mut output)?;
            if it >= warm {
                pure.push(t.elapsed().as_secs_f64() * 1e3);
            }
            reset(e)?;
            let reqs = requests(w);
            let t = Instant::now();
            e.token_batch_step(&reqs, &mut output)?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let ride_ids: Vec<u32> = output.iter().map(|&(_, t)| t).collect();
            if output.iter().map(|&(id, _)| id as usize).ne(0..w) {
                return Err("rider outputs out of order".into());
            }
            let mut ride_logits = Vec::with_capacity(if it == warm { w } else { 0 });
            if it == warm {
                for r in 0..w {
                    e.logits_row(r, &mut logits)?;
                    ride_logits.push(logits.clone());
                }
            }
            reset(e)?;
            let t = Instant::now();
            e.step_slots(&feeds[..w], &mut toks)?;
            let step_ms = t.elapsed().as_secs_f64() * 1e3;
            if it >= warm {
                ride.push(ms);
                step.push(step_ms);
            }
            if it == warm {
                let (mut max_abs, mut kl_max, mut agree) = (0f32, 0f64, 0);
                let mut raw_ride = Vec::new();
                let mut raw_step = Vec::new();
                for (r, rl) in ride_logits.iter().enumerate() {
                    e.logits_row(r, &mut logits)?;
                    agree += usize::from(ride_ids[r] == toks[r]);
                    let lse = |v: &[f32]| {
                        let m = v.iter().copied().fold(f32::MIN, f32::max) as f64;
                        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
                    };
                    let (lp, lq) = (lse(&logits), lse(rl));
                    let kl: f64 = logits
                        .iter()
                        .zip(rl)
                        .map(|(&p, &q)| {
                            let lp_i = p as f64 - lp;
                            lp_i.exp() * (lp_i - (q as f64 - lq))
                        })
                        .sum();
                    kl_max = kl_max.max(kl);
                    max_abs = rl
                        .iter()
                        .zip(&logits)
                        .map(|(a, b)| (a - b).abs())
                        .fold(max_abs, f32::max);
                    if dump.is_some() {
                        raw_ride.extend(bf16(rl));
                        raw_step.extend(bf16(&logits));
                    }
                }
                println!(
                    "RIDE_VS_STEP w={w} ctx={ctx} top1_agree={agree}/{w} max_kl={kl_max:.3e} max_abs={max_abs:.4} ride_ids={ride_ids:?}"
                );
                if let Some(dir) = dump {
                    std::fs::write(dir.join(format!("ride-w{w}.bf16")), raw_ride)?;
                    std::fs::write(dir.join(format!("step-w{w}.bf16")), raw_step)?;
                }
            }
        }
        let (p, r, s) = (median(&mut pure), median(&mut ride), median(&mut step));
        println!(
            "RIDE rows={rows} w={w} ctx={ctx} prefix={prefix} n={iters} pure_ms={p:.3} ride_ms={r:.3} step_ms={s:.3} per_rider_ms={:.4} saved_ms={:.3}",
            (r - p) / w as f64,
            p + s - r
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::record_inputs;

    #[test]
    fn histories_include_inputs_but_exclude_the_final_sample() {
        let prompts = vec![vec![10, 11], vec![20, 21]];
        let mut single = prompts.clone();
        record_inputs(&mut single, &[12, 22], &[13, 23], 1);
        record_inputs(&mut single, &[13, 23], &[14, 24], 1);
        record_inputs(&mut single, &[14, 24], &[15, 25], 1);
        let mut multi = prompts;
        record_inputs(&mut multi, &[12, 22], &[13, 14, 15, 23, 24, 25], 3);
        assert_eq!(single, multi);
        assert_eq!(
            single,
            vec![vec![10, 11, 12, 13, 14], vec![20, 21, 22, 23, 24]]
        );
    }
}

#[derive(Default)]
struct SpecArgs {
    mode: Option<String>,
    k: usize,
    rung: usize,
    poison: bool,
    nsplit_kvlen: Option<u32>,
    draft_iters: usize,
    sample: Option<(f32, usize, f32, f32)>,
    host: bool,
}

/// `--spec-sample`'s draws: slot `b`'s uniform for output token `i` (the `seeded_unit_with` grid),
/// and each verify row's host sample for the device-vs-host check.
#[cfg(feature = "cuda")]
struct BenchSampler {
    params: plowrt::text::sample::SamplingParams,
    /// Per span: (slot, tokens it has emitted).
    spans: Vec<(usize, usize)>,
    prior: Vec<u32>,
}

#[cfg(feature = "cuda")]
fn bench_unit(b: usize, i: usize) -> f32 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64 ^ ((b as u64) << 32 | i as u64);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((h ^ (h >> 31)) % 10_000) as f32 / 10_000.0
}

#[cfg(feature = "cuda")]
impl plowrt::exec::gpu::spec::SpecSampler for BenchSampler {
    fn draws(&mut self, i: usize, _: &[u32], out: &mut [f32]) {
        let (b, at) = self.spans[i];
        for (j, u) in out.iter_mut().enumerate() {
            *u = bench_unit(b, at + j);
        }
    }

    fn host(&mut self, i: usize, tokens: &[u32], j: usize, logits: &mut [f32]) -> u32 {
        let (b, at) = self.spans[i];
        plowrt::orch::speculative::host_row_sample(
            logits,
            &self.params,
            &[],
            &tokens[1..=j],
            bench_unit(b, at + j),
            &mut self.prior,
        )
    }
}

#[cfg(feature = "cuda")]
/// `--draft-bench ITERS`: after the warmup steps, row `b` of the hidden activation is slot `b`'s
/// last decode row and `last[b]` the token it produced, which is exactly a draft's input. Times
/// the draft program at every `k` (ITERS back-to-back launches, inputs staged once), then checks
/// the deepest draft against the target's own greedy continuation (prefix agreement).
fn draft_bench(
    e: &mut plowrt::exec::gpu::GpuEngine,
    assets: &std::path::Path,
    last: &[u32],
    iters: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    let kmax = e.spec_draft_enable(assets)?;
    let mut toks = Vec::new();
    let rows: Vec<(usize, u32, u32)> = last
        .iter()
        .enumerate()
        .map(|(b, &t)| (b, t, b as u32))
        .collect();
    let block = e.spec_draft_is_block();
    let stage = |e: &mut plowrt::exec::gpu::GpuEngine, k: u32| {
        if block {
            let rows: Vec<(usize, u32)> = rows.iter().map(|&(b, t, _)| (b, t)).collect();
            e.spec_draft_stage_block(&rows, k)
        } else {
            e.spec_draft_stage(&rows, k)
        }
    };
    let mut drafts = Vec::new();
    for k in 1..=kmax {
        stage(e, k)?;
        for _ in 0..3 {
            e.spec_draft_launch()?;
        }
        e.spec_draft_sync()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            e.spec_draft_launch()?;
        }
        e.spec_draft_sync()?;
        let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
        e.spec_draft_run(&mut drafts)?;
        println!(
            "DRAFT B={} k={k} ms={ms:.4} per_token_ms={:.4} row0={:?}",
            rows.len(),
            ms / f64::from(k),
            &drafts[..k as usize]
        );
    }
    // `STEP_BENCH_DRAFT_TRACE=1` with a -DPLOW_NV_TRACE object: block 0's per-op profile of one
    // k = kmax draft launch (PLOW_TRACE_RAW gets the ordered packets).
    if std::env::var_os("STEP_BENCH_DRAFT_TRACE").is_some() {
        stage(e, kmax)?;
        e.trace_reset()?;
        e.spec_draft_launch()?;
        e.spec_draft_sync()?;
        if let Some(profile) = e.trace_summary()? {
            println!("DRAFT_TRACE B={} k={kmax}\n{profile}", rows.len());
        }
    }
    if let Some(dir) = std::env::var_os("STEP_BENCH_DRAFT_DUMP").map(std::path::PathBuf::from) {
        draft_dump(e, &dir, &rows)?;
    }
    let k = kmax as usize;
    let mut cur = last.to_vec();
    let mut reference = vec![Vec::with_capacity(k); rows.len()];
    for _ in 0..k {
        let feeds: Vec<(usize, u32)> = cur.iter().copied().enumerate().collect();
        e.step_slots(&feeds, &mut toks)?;
        for (b, &t) in toks.iter().enumerate() {
            reference[b].push(t);
        }
        cur.copy_from_slice(&toks);
    }
    let agree: Vec<usize> = (0..rows.len())
        .map(|b| {
            drafts[b * k..(b + 1) * k]
                .iter()
                .zip(&reference[b])
                .take_while(|(d, r)| d == r)
                .count()
        })
        .collect();
    println!(
        "DRAFT_AGREE B={} k={k} mean_prefix={:.3} per_row={agree:?} ref0={:?}",
        rows.len(),
        agree.iter().sum::<usize>() as f64 / rows.len() as f64,
        reference[0]
    );
    Ok(())
}

/// Bench-only: per slot, the draft's inputs, the activations one k=1 draft leaves behind (layer 3
/// for the per-layer ones), and the slot's target KV for the drafter's layers, for an offline
/// reference diff (`STEP_BENCH_DRAFT_DUMP=dir`).
#[cfg(feature = "cuda")]
fn draft_dump(
    e: &mut plowrt::exec::gpu::GpuEngine,
    dir: &std::path::Path,
    rows: &[(usize, u32, u32)],
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir)?;
    let slots = e.tensor_bytes("in.kvlen").ok_or("no in.kvlen")? as usize / 4;
    let read = |e: &plowrt::exec::gpu::GpuEngine, name: &str, off: u64, len: usize| {
        let mut v = vec![0u8; len];
        e.read_tensor_range(name, off, &mut v).map(|_| v)
    };
    let mut toks = Vec::new();
    let block = e.spec_draft_is_block();
    for &(b, token, hrow) in rows {
        let pos = e.slot_pos(b);
        let d = dir.join(format!("slot{b}"));
        std::fs::create_dir_all(&d)?;
        if block {
            // The block's raw logits (one sequence: rows 0..7) after a k = 7 draft.
            e.spec_draft_stage_block(&[(b, token)], 7)?;
            e.spec_draft_run(&mut toks)?;
            let vocab = e.vocab();
            std::fs::write(
                d.join("act.dspark.d.logits.bin"),
                read(e, "act.dspark.d.logits", 0, 7 * vocab * 2)?,
            )?;
        } else {
            e.spec_draft_stage(&[(b, token, hrow)], 7)?;
            e.spec_draft_run(&mut toks)?;
        }
        let drafts7 = toks.clone();
        if block {
            dump_kv(e, &d, b, pos, slots, &read)?;
            std::fs::write(
                d.join("meta.json"),
                serde_json::to_vec_pretty(&serde_json::json!({
                    "slot": b, "token": token, "pos": pos, "drafts7": drafts7,
                }))?,
            )?;
            continue;
        }
        e.spec_draft_stage(&[(b, token, hrow)], 1)?;
        e.spec_draft_run(&mut toks)?;
        let hidden = e.tensor_bytes("act.mtp.h").ok_or("no act.mtp.h")? as usize / 16;
        std::fs::write(
            d.join("act.hn.bin"),
            read(e, "act.hn", hrow as u64 * hidden as u64, hidden)?,
        )?;
        for n in [
            "e", "h", "x1", "x2", "x", "xn", "q", "qn", "at", "og", "hn", "fu", "dg", "logits",
        ] {
            let name = format!("act.mtp.{n}");
            let row = e.tensor_bytes(&name).ok_or("missing")? as usize / 16;
            std::fs::write(d.join(format!("{name}.bin")), read(e, &name, 0, row)?)?;
        }
        dump_kv(e, &d, b, pos, slots, &read)?;
        std::fs::write(
            d.join("meta.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "slot": b, "token": token, "hidden_row": hrow, "pos": pos,
                "draft1": toks[0], "drafts7": drafts7,
            }))?,
        )?;
    }
    Ok(())
}

/// `STEP_BENCH_DRAFT_DUMP_KV` layers of slot `b` (rows `[0, pos)` of full layers) into `d`.
#[cfg(feature = "cuda")]
fn dump_kv(
    e: &plowrt::exec::gpu::GpuEngine,
    d: &std::path::Path,
    b: usize,
    pos: usize,
    slots: usize,
    read: &dyn Fn(&plowrt::exec::gpu::GpuEngine, &str, u64, usize) -> plowrt::Result<Vec<u8>>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Sliding ring whole; full layer only the committed rows per head (the rest is unmapped).
    // `STEP_BENCH_DRAFT_DUMP_KV=<sliding layer>,<full layer>:<kv heads>:<head dim>` (default:
    // the 26B-A4B assistant's 28 and 29:2:512; the 12B assistant is 46,47:1:512).
    let spec = std::env::var("STEP_BENCH_DRAFT_DUMP_KV").unwrap_or_else(|_| "28,29:2:512".into());
    let kv_layers = spec
        .split(',')
        .map(|f| {
            let bad = || {
                format!("STEP_BENCH_DRAFT_DUMP_KV: {f:?} is not <layer> or <layer>:<heads>:<dim>")
            };
            let v = f
                .split(':')
                .map(|x| x.parse::<usize>().map_err(|_| bad()))
                .collect::<Result<Vec<_>, _>>()?;
            match v[..] {
                [l] => Ok((l, 0, 0)),
                [l, h, d] if h > 0 && d > 0 => Ok((l, h, d)),
                _ => Err(bad()),
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    for &(l, kvh, row_bytes) in &kv_layers {
        for (t, elt) in [("k", 1), ("v", 1), ("k_scale", 4), ("v_scale", 4)] {
            let name = format!("kv.{l}.{t}");
            let per = e.tensor_bytes(&name).ok_or("missing kv")? as usize / slots;
            let bytes = if kvh == 0 {
                read(e, &name, (b * per) as u64, per)?
            } else {
                let (head, row) = (per / kvh, if elt == 1 { row_bytes } else { 4 });
                if per % kvh != 0 || pos * row > head {
                    let m = format!("{name}: {kvh} x {pos} rows x {row} B exceed {per} B/slot");
                    return Err(m.into());
                }
                let mut v = Vec::new();
                for h in 0..kvh {
                    v.extend(read(e, &name, (b * per + h * head) as u64, pos * row)?);
                }
                v
            };
            std::fs::write(d.join(format!("{name}.bin")), bytes)?;
        }
    }
    Ok(())
}

/// `--spec MODE`: the span-mode verify identity gate. A plain greedy run of `steps` tokens per
/// slot at a fixed rung (`--spec-rung R`, and `--nsplit-kvlen N` for a fixed dense split) is the
/// reference; the slots are rewound and the same tokens regenerated through `spec_verify` +
/// `spec_commit` with drafts from MODE: `oracle` (the reference's next tokens, all accepted),
/// `adversarial` (every draft wrong, all rejected), `ngram` (prompt lookup over each slot's
/// history). `--spec-poison` NaN-fills every row a verify will write before it runs, and every
/// uncommitted row after it. `--spec-sample T,TOP_K,TOP_P,MIN_P` makes the reference a plain
/// sampled run (`plow_sample`, or the host sampler with `--spec-host`; draw `bench_unit(slot, i)`
/// for token `i`) and samples the verify rows with plain decode's sampler and draws, so the
/// streams must still match; each
/// device-sampled verdict is also recomputed on the host from the downloaded verify logits
/// (`SPEC_SAMPLE_HOST`). Prints `SPEC_IDENTITY ok|MISMATCH` and the verify cost.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn spec_bench(
    e: &mut plowrt::exec::gpu::GpuEngine,
    assets: &std::path::Path,
    a: &SpecArgs,
    mode: &str,
    prompts: &[Vec<u32>],
    last: &[u32],
    base: usize,
    steps: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::gpu::spec::{SpecDraw, SpecSpan};
    use plowrt::orch::speculative::ngram_propose;
    use std::time::Instant;
    let slots = last.len();
    let max_k = e.spec_enable(assets)?;
    if a.k > max_k || slots * (a.k + 1) > a.rung.max(1) && a.rung > 0 {
        return Err(format!(
            "k {} (max {max_k}) x {slots} slots exceeds rung {}",
            a.k, a.rung
        )
        .into());
    }
    e.set_debug_min_rung(a.rung);
    e.set_debug_nsplit_kvlen(a.nsplit_kvlen);
    for b in 0..slots {
        e.snapshot_slot(b)?;
    }
    let dev = a
        .sample
        .map(|(temp, top_k, top_p, min_p)| plowrt::exec::gpu::DevSample {
            temp,
            top_k: top_k as i32,
            top_p,
            min_p,
            rng01: 0.0,
            penalty: 1.0,
        });
    if dev.is_some() && !e.dev_sample_enabled() {
        return Err("--spec-sample needs the device sampler (sample_sm120.cubin)".into());
    }
    let mut sampler = BenchSampler {
        params: plowrt::text::sample::SamplingParams {
            temperature: dev.map_or(0.0, |d| d.temp),
            top_k: dev.map_or(0, |d| d.top_k as usize),
            top_p: dev.map_or(1.0, |d| d.top_p),
            min_p: dev.map_or(0.0, |d| d.min_p),
            ..Default::default()
        },
        spans: Vec::new(),
        prior: Vec::new(),
    };
    let draw = match dev {
        None => SpecDraw::Argmax,
        Some(_) if a.host => SpecDraw::Host,
        Some(d) => SpecDraw::Device(d),
    };
    let mut logits = Vec::new();
    // Reference: plain decode (greedy, device-sampled, or host-sampled) at the fixed rung.
    let mut cur = last.to_vec();
    let mut reference = vec![Vec::with_capacity(steps); slots];
    let feeds = |cur: &[u32]| -> Vec<(usize, u32)> { cur.iter().copied().enumerate().collect() };
    let mut toks = Vec::new();
    let mut specs = vec![plowrt::exec::gpu::DevSample::greedy(); e.batch()];
    let t0 = Instant::now();
    for i in 0..steps {
        if let Some(d) = dev {
            for (b, s) in specs.iter_mut().enumerate().take(slots) {
                *s = plowrt::exec::gpu::DevSample {
                    rng01: bench_unit(b, i),
                    ..d
                };
            }
        }
        if a.host && dev.is_some() {
            // `--spec-host`'s reference is plain host-sampled decode.
            e.step_slots(&feeds(&cur), &mut toks)?;
            for (b, t) in toks.iter_mut().enumerate() {
                e.logits_row(b, &mut logits)?;
                let u = bench_unit(b, i);
                *t = plowrt::orch::speculative::host_row_sample(
                    &mut logits,
                    &sampler.params,
                    &[],
                    &[],
                    u,
                    &mut sampler.prior,
                );
            }
        } else {
            e.step_slots_sampled(&feeds(&cur), dev.map(|_| specs.as_slice()), &mut toks)?;
        }
        for (b, &t) in toks.iter().enumerate() {
            reference[b].push(t);
        }
        cur.copy_from_slice(&toks);
    }
    let plain_ms = t0.elapsed().as_secs_f64() * 1e3 / steps.max(1) as f64;
    for b in 0..slots {
        e.rewind_slot(b, base)?;
    }

    let vocab = e.vocab() as u32;
    let mut out: Vec<Vec<u32>> = vec![Vec::with_capacity(steps); slots];
    let mut hist: Vec<Vec<u32>> = prompts.to_vec();
    for (h, &t) in hist.iter_mut().zip(last) {
        h.push(t);
    }
    let mut pending = last.to_vec();
    let (mut ticks, mut drafted, mut accepted, mut verify_ms) = (0usize, 0usize, 0usize, 0f64);
    let mut n_spans = 0usize;
    let mut inputs: Vec<Vec<u32>> = vec![Vec::new(); slots];
    let mut draft = Vec::new();
    let mut verdicts = Vec::new();
    // `mtp`: the packet's draft programs. A slot's draft input is its pending token and the verify
    // row that produced it; the first tick verifies the pending token alone to get that row.
    let mut hrow: Vec<Option<u32>> = vec![None; slots];
    let mut mtp = Vec::new();
    let mut draft_ms = 0f64;
    let (mut host_rows, mut host_spans, mut host_mismatch) = (0usize, 0usize, 0usize);
    if mode == "mtp" || mode == "dspark" {
        e.spec_draft_enable(assets)?;
    }
    while out.iter().any(|o| o.len() < steps) {
        let live: Vec<usize> = (0..slots).filter(|&b| out[b].len() < steps).collect();
        let drafting = mode == "dspark" || mode == "mtp" && live.iter().all(|&b| hrow[b].is_some());
        if mode == "dspark" {
            let rows: Vec<(usize, u32)> = live.iter().map(|&b| (b, pending[b])).collect();
            let t = Instant::now();
            e.spec_draft_stage_block(&rows, a.k as u32)?;
            e.spec_draft_run(&mut mtp)?;
            draft_ms += t.elapsed().as_secs_f64() * 1e3;
        } else if drafting {
            let rows: Vec<(usize, u32, u32)> = live
                .iter()
                .map(|&b| (b, pending[b], hrow[b].unwrap()))
                .collect();
            let t = Instant::now();
            e.spec_draft_stage(&rows, a.k as u32)?;
            e.spec_draft_run(&mut mtp)?;
            draft_ms += t.elapsed().as_secs_f64() * 1e3;
        }
        for (li, &b) in live.iter().enumerate() {
            let i = out[b].len();
            let k = a.k.min(steps - i - 1);
            match mode {
                "mtp" | "dspark" => {
                    draft.clear();
                    if drafting {
                        draft.extend_from_slice(&mtp[li * a.k..li * a.k + k]);
                    }
                }
                "oracle" => {
                    draft.clear();
                    draft.extend_from_slice(&reference[b][i..i + k]);
                }
                "adversarial" => {
                    draft.clear();
                    draft.extend(reference[b][i..i + k].iter().map(|&t| (t + 1) % vocab));
                }
                "ngram" => ngram_propose(&hist[b], k, 1, 4, &mut draft),
                m => return Err(format!("unknown --spec mode {m}").into()),
            }
            inputs[b].clear();
            inputs[b].push(pending[b]);
            inputs[b].extend_from_slice(&draft);
            drafted += inputs[b].len() - 1;
        }
        let spans: Vec<SpecSpan> = live
            .iter()
            .map(|&b| SpecSpan {
                slot: b,
                tokens: &inputs[b],
                draw,
            })
            .collect();
        sampler.spans.clear();
        sampler
            .spans
            .extend(live.iter().map(|&b| (b, out[b].len())));
        if a.poison {
            // Every row this launch writes holds NaN first: a row read before its own span's
            // store lands (a missing intra-span dependency) poisons the output.
            for sp in &spans {
                let p0 = e.slot_pos(sp.slot) as u32;
                e.spec_poison_rows(sp.slot, p0, p0 + sp.tokens.len() as u32)?;
            }
        }
        let t = Instant::now();
        e.spec_verify(&spans, &[], Some(&mut sampler), &mut verdicts)?;
        verify_ms += t.elapsed().as_secs_f64() * 1e3;
        if let SpecDraw::Device(_) = draw {
            use plowrt::exec::gpu::spec::SpecSampler;
            let mut r = 0u32;
            for (i, (sp, v)) in spans.iter().zip(&verdicts).enumerate() {
                let host = plowrt::orch::speculative::accept_sampled(
                    sp.tokens,
                    |j| {
                        e.logits_row(r as usize + j, &mut logits)?;
                        host_rows += 1;
                        Ok::<_, plowrt::RuntimeError>(sampler.host(i, sp.tokens, j, &mut logits))
                    },
                    r,
                    v.req_gen,
                    &[],
                )?;
                host_spans += 1;
                if host != *v {
                    host_mismatch += 1;
                    println!("  slot {}: device {v:?} vs host {host:?}", sp.slot);
                }
                r += sp.tokens.len() as u32;
            }
        }
        ticks += 1;
        n_spans += spans.len();
        for (sp, v) in spans.iter().zip(&verdicts) {
            let n = usize::from(v.n_emit);
            let p0 = e.slot_pos(sp.slot) as u32;
            if a.poison {
                e.spec_poison_rows(sp.slot, p0 + n as u32, p0 + sp.tokens.len() as u32)?;
            }
            e.spec_commit(sp, v)?;
            accepted += usize::from(v.n_acc);
            let emitted = &v.tokens[..n];
            out[sp.slot].extend_from_slice(emitted);
            hist[sp.slot].extend_from_slice(emitted);
            pending[sp.slot] = emitted[n - 1];
            hrow[sp.slot] = Some(v.hrow);
        }
    }
    if let SpecDraw::Device(_) = draw {
        println!(
            "SPEC_SAMPLE_HOST spans={host_spans} rows={host_rows} verdict_mismatch={host_mismatch}"
        );
    }
    let mut ok = true;
    for b in 0..slots {
        out[b].truncate(steps);
        if out[b] != reference[b] {
            ok = false;
            let at = out[b].iter().zip(&reference[b]).position(|(x, y)| x != y);
            println!("  slot {b}: first divergence at {at:?}");
        }
    }
    let digest = |s: &[Vec<u32>]| {
        let mut d: u64 = 0xcbf29ce484222325;
        for i in 0..steps {
            for v in s {
                d = (d ^ u64::from(v[i])).wrapping_mul(0x100000001b3);
            }
        }
        d
    };
    println!(
        "SPEC_IDENTITY {} mode={mode} slots={slots} k={} rung={} base={base} steps={steps} \
         poison={} sample={:?} draw={} ref_fnv={:016x} spec_fnv={:016x} ticks={ticks} drafted={drafted} accepted={accepted} \
         mean_emit={:.3} plain_ms={plain_ms:.3} verify_ms={:.3} draft_ms={:.3}",
        if ok { "ok" } else { "MISMATCH" },
        a.k,
        a.rung,
        a.poison,
        a.sample,
        match draw {
            SpecDraw::Argmax => "argmax",
            SpecDraw::Device(_) => "device",
            SpecDraw::Host => "host",
        },
        digest(&reference),
        digest(&out),
        (steps * slots) as f64 / n_spans as f64,
        verify_ms / ticks as f64,
        draft_ms / ticks as f64,
    );
    if ok {
        Ok(())
    } else {
        Err("speculative verify diverged from the plain reference".into())
    }
}
