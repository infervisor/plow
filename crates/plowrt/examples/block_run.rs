//! Single-block launcher. Loads a
//! block asset (a PLOWDEV blob compiled by `gemma4 --block`, its `block.json`
//! descriptor, and a checkpoint) and drives just that block on the real GPU
//! through two verbs on ONE loaded engine:
//!
//!   block_run <asset-dir> check [--in x.npy] [--out y.npy] [--ctx T]
//!                              [--dump-tensors name,name --dump-dir dir]
//!   block_run <asset-dir> bench --batch 1,2,4,8 --ctx 128,512,1024,4096
//!                              [--iters 100] [--warmup 10] [--prefill-iters 10]
//!                              [--pf-chunk N]
//!   block_run <asset-dir> mixed-check --rows 128 --decode 1
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
            "mixed-check" => mixed_check(&mut e, &desc, hidden, &out_name, &flag),
            other => Err(format!("unknown verb {other:?} (check|bench|mixed-check)").into()),
        }
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
        if decode_rows == 0 || decode_rows + 1 > e.batch() || decode_rows >= rows {
            return Err("mixed-check needs 0 < decode < rows and one free prefill slot".into());
        }
        for slot in 0..=decode_rows {
            e.begin_slot(slot, rows + 1)?;
        }
        let input = synth(rows, hidden);
        e.upload_activation("act.x", &input)?;
        let decode: Vec<_> = (0..decode_rows)
            .map(|slot| plow_asset::mixed_step::DecodeRequest {
                slot: slot as u32,
                state_slot: slot as u32,
                token: 100 + slot as u32,
            })
            .collect();
        let prefill_tokens: Vec<_> = (0..rows - decode_rows)
            .map(|row| 200 + row as u32)
            .collect();
        let prefill = [plow_asset::mixed_step::PrefillRequest {
            slot: decode_rows as u32,
            state_slot: decode_rows as u32,
            start: 0,
            tokens: &prefill_tokens,
            prompt_len: prefill_tokens.len() as u32,
        }];
        let start = Instant::now();
        e.mixed_step(rows as u32, &decode, &prefill, &mut [])?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1e3;
        let mixed_out = e.download_activation(out_name)?;
        let mixed_out = &mixed_out[..rows * hidden];
        if mixed_out.iter().any(|value| !value.is_finite()) {
            return Err("mixed block output contains non-finite values".into());
        }
        let mixed_kv = snapshot_kv(e, desc, decode_rows, rows - decode_rows)?;

        for slot in 0..=decode_rows {
            e.begin_slot(slot, rows + 1)?;
        }
        e.upload_activation("act.x", &input[..decode_rows * hidden])?;
        let feeds: Vec<_> = (0..decode_rows)
            .map(|slot| (slot, 100 + slot as u32))
            .collect();
        let mut tokens = Vec::new();
        e.step_slots(&feeds, &mut tokens)?;
        let decode_out = e.download_activation(out_name)?;

        e.upload_activation("act.x", &input[decode_rows * hidden..])?;
        e.prefill_slot(decode_rows, &prefill_tokens)?;
        let prefill_out = e.download_activation(out_name)?;
        let reference_kv = snapshot_kv(e, desc, decode_rows, rows - decode_rows)?;
        let mut reference_out = Vec::with_capacity(rows * hidden);
        reference_out.extend_from_slice(&decode_out[..decode_rows * hidden]);
        reference_out.extend_from_slice(&prefill_out[..(rows - decode_rows) * hidden]);
        compare_f32("activation", mixed_out, &reference_out, 6.0e-3, 5.0e-2)?;
        compare_bf16("written KV", &mixed_kv, &reference_kv, 6.0e-3, 5.0e-2)?;
        println!(
            "mixed-check: rows={rows} decode={decode_rows} prefill={} elapsed={elapsed_ms:.3} ms parity=PASS",
            prefill_tokens.len()
        );
        Ok(())
    }

    fn snapshot_kv(
        e: &plowrt::exec::gpu::GpuEngine,
        desc: &plow_asset::BlockDescriptor,
        decode_rows: usize,
        prefill_rows: usize,
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
                for (slot, written_rows) in (0..decode_rows)
                    .map(|slot| (slot, 1))
                    .chain(std::iter::once((decode_rows, prefill_rows)))
                {
                    for head in 0..heads {
                        let offset = slot as u64 * slot_bytes + head as u64 * head_bytes;
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
