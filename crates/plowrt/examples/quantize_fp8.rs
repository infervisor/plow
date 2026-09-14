//! Weight-only per-output-channel e4m3 fp8 twin of a bf16 checkpoint — the Rust twin of
//! `perf-data/tools/quantize_fp8.py` (per-channel mode) for hosts without torch.
//!
//! `cargo run --release --features cpu --example quantize_fp8 -- <src-model-dir> <out-dir> [prefix] [--head-only]`
//!
//! Method (settled, see the Python header): W is [N, K] row-major; scale[n] = amax(|W[n,:]|)/448
//! (1.0 for an all-zero row); W8[n,k] = e4m3fn(W[n,k] / scale[n]) round-to-nearest-even,
//! saturating at +-448. Output: `<out-dir>/model.safetensors` keyed `fp8/<name>` (F8_E4M3,
//! [N,K]) + `fp8/<name>_scale` (F32, [N]) for the 7 dense projections of every layer that
//! exists in the source (Gemma full layers have no v_proj). Prefix default
//! `model.language_model.` (Gemma-4 multimodal re-export); Llama/Qwen use `model.`.

#[cfg(feature = "cpu")]
fn main() {
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::path::{Path, PathBuf};

    const E4M3_MAX: f32 = 448.0;
    const PROJS: [&str; 7] = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ];

    /// f32 -> e4m3fn, RNE, saturating (torch.float8_e4m3fn semantics for finite inputs).
    fn to_e4m3(v: f32) -> u8 {
        let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
        let a = v.abs();
        if a.is_nan() {
            return 0x7f | sign;
        }
        if a >= E4M3_MAX {
            return 0x7e | sign; // 448 = 1.75 * 2^8 -> e=15, m=6
        }
        if a < 0.5f32.powi(9) / 2.0 {
            // below half the smallest subnormal (2^-9): rounds to zero
            return sign;
        }
        // Subnormal range: |v| < 2^-6 -> m = round(|v| * 2^9), e = 0 (m may round up to 8 -> 2^-6).
        if a < 0.015625 {
            let m = (a * 512.0).round_ties_even() as u32;
            if m >= 8 {
                return sign | (1 << 3);
            }
            return sign | m as u8;
        }
        let bits = a.to_bits();
        let e = ((bits >> 23) & 0xff) as i32 - 127; // -6 ..= 8
        let mant = bits & 0x7f_ffff;
        // Round the 23-bit mantissa to 3 bits, ties to even.
        let mut m = mant >> 20;
        let rem = mant & 0xf_ffff;
        let half = 0x8_0000;
        if rem > half || (rem == half && (m & 1) == 1) {
            m += 1;
        }
        let (mut e8, mut m8) = (e + 7, m);
        if m8 == 8 {
            m8 = 0;
            e8 += 1;
        }
        if e8 > 15 || (e8 == 15 && m8 == 7) {
            return 0x7e | sign; // saturate (never NaN)
        }
        sign | ((e8 as u8) << 3) | m8 as u8
    }

    let mut args = std::env::args().skip(1);
    let src: PathBuf = args
        .next()
        .expect("usage: quantize_fp8 <src-dir> <out-dir> [prefix]")
        .into();
    let out: PathBuf = args
        .next()
        .expect("usage: quantize_fp8 <src-dir> <out-dir> [prefix]")
        .into();
    let rest: Vec<_> = args.collect();
    let head_only = rest.iter().any(|arg| arg == "--head-only");
    let positional: Vec<_> = rest
        .into_iter()
        .filter(|arg| arg != "--head-only")
        .collect();
    assert!(
        positional.len() <= 1,
        "usage: quantize_fp8 <src-dir> <out-dir> [prefix] [--head-only]"
    );
    let prefix = positional
        .into_iter()
        .next()
        .unwrap_or_else(|| "model.language_model.".to_string());

    // Source tensors: sharded (index.json) or single file.
    let mut shards: Vec<memmap2::Mmap> = Vec::new();
    let mut index: BTreeMap<String, (usize, safetensors::tensor::TensorInfo)> = BTreeMap::new();
    let mut files: Vec<PathBuf> = Vec::new();
    let idx_path = src.join("model.safetensors.index.json");
    if idx_path.exists() {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&idx_path).expect("read index"))
                .expect("index json");
        let mut set: Vec<String> = v["weight_map"]
            .as_object()
            .expect("weight_map")
            .values()
            .map(|f| f.as_str().unwrap().to_string())
            .collect();
        set.sort();
        set.dedup();
        files.extend(set.into_iter().map(|f| src.join(f)));
    } else {
        files.push(src.join("model.safetensors"));
    }
    let mut data_off: Vec<usize> = Vec::new();
    for (si, f) in files.iter().enumerate() {
        let file = std::fs::File::open(f).unwrap_or_else(|e| panic!("open {}: {e}", f.display()));
        // SAFETY: read-only mapping of a checkpoint file nobody writes while we run.
        let map = unsafe { memmap2::Mmap::map(&file) }.expect("mmap");
        let (n, meta) = safetensors::SafeTensors::read_metadata(&map).expect("safetensors header");
        for (name, info) in meta.tensors() {
            index.insert(name.clone(), (si, info.clone()));
        }
        data_off.push(8 + n);
        shards.push(map);
    }
    let layers = 1 + index
        .keys()
        .filter_map(|k| {
            k.split("layers.")
                .nth(1)
                .and_then(|r| r.split('.').next())
                .and_then(|n| n.parse::<usize>().ok())
        })
        .max()
        .expect("no layers.* tensors");
    eprintln!(
        "src {}: {} tensors, {layers} layers, prefix {prefix:?}",
        src.display(),
        index.len()
    );

    // Plan in the Python's order: layer-major, PROJS order, only tensors that exist.
    let mut plan: Vec<(String, usize, usize)> = Vec::new(); // (name, N, K)
    if !head_only {
        for l in 0..layers {
            for p in PROJS {
                let name = format!("{prefix}layers.{l}.{p}");
                if let Some((_, info)) = index.get(&name) {
                    assert_eq!(info.dtype, safetensors::Dtype::BF16, "{name}: not bf16");
                    assert_eq!(info.shape.len(), 2, "{name}: not 2-D");
                    plan.push((name, info.shape[0], info.shape[1]));
                }
            }
        }
    }
    // The tied embedding / lm_head as well (`PLOW_FP8_HEAD=1` at emit reads `fp8/<prefix>embed_tokens.weight`):
    // on a 262k-vocab model the bf16 head is the largest single per-token read (E4B: 1.3 GB).
    let head = format!("{prefix}embed_tokens.weight");
    if let Some((_, info)) = index.get(&head) {
        if info.dtype == safetensors::Dtype::BF16 && info.shape.len() == 2 {
            plan.push((head, info.shape[0], info.shape[1]));
        }
    }
    assert!(
        !plan.is_empty(),
        "no supported projection weights found under prefix {prefix:?}"
    );

    // Header.
    let mut meta = serde_json::Map::new();
    let mut off = 0usize;
    for (name, n, k) in &plan {
        let wb = n * k;
        meta.insert(
            format!("fp8/{name}"),
            serde_json::json!({"dtype": "F8_E4M3", "shape": [n, k], "data_offsets": [off, off + wb]}),
        );
        off += wb;
        meta.insert(
            format!("fp8/{name}_scale"),
            serde_json::json!({"dtype": "F32", "shape": [n], "data_offsets": [off, off + n * 4]}),
        );
        off += n * 4;
    }
    let mut hdr = serde_json::to_vec(&serde_json::Value::Object(meta)).unwrap();
    while hdr.len() % 8 != 0 {
        hdr.push(b' ');
    }
    std::fs::create_dir_all(&out).expect("mkdir out");
    let out_path = out.join("model.safetensors");
    let mut o = std::io::BufWriter::with_capacity(
        16 << 20,
        std::fs::File::create(&out_path).expect("create out"),
    );
    o.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
    o.write_all(&hdr).unwrap();

    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let t0 = std::time::Instant::now();
    for (i, (name, n, k)) in plan.iter().enumerate() {
        let (si, info) = &index[name];
        let base = data_off[*si] + info.data_offsets.0;
        let bytes = &shards[*si][base..base + n * k * 2];
        let mut q = vec![0u8; n * k];
        let mut scales = vec![0f32; *n];
        let rows_per = (n + threads - 1) / threads;
        std::thread::scope(|s| {
            for (ti, (qc, sc)) in q
                .chunks_mut(rows_per * k)
                .zip(scales.chunks_mut(rows_per))
                .enumerate()
            {
                let r0 = ti * rows_per;
                let src_rows = &bytes[r0 * k * 2..(r0 + sc.len()) * k * 2];
                s.spawn(move || {
                    for (r, (qrow, scale)) in qc.chunks_mut(*k).zip(sc.iter_mut()).enumerate() {
                        let row = &src_rows[r * k * 2..(r + 1) * k * 2];
                        let mut amax = 0f32;
                        for j in 0..*k {
                            let v = f32::from_bits(
                                (u16::from_le_bytes([row[2 * j], row[2 * j + 1]]) as u32) << 16,
                            );
                            amax = amax.max(v.abs());
                        }
                        let sc = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
                        *scale = sc;
                        for j in 0..*k {
                            let v = f32::from_bits(
                                (u16::from_le_bytes([row[2 * j], row[2 * j + 1]]) as u32) << 16,
                            );
                            qrow[j] = to_e4m3(v / sc);
                        }
                    }
                });
            }
        });
        o.write_all(&q).unwrap();
        for sc in &scales {
            o.write_all(&sc.to_le_bytes()).unwrap();
        }
        if i % 40 == 0 {
            eprintln!(
                "  [{}/{}] {name}  N={n} K={k}  {:.1}s",
                i + 1,
                plan.len(),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    o.flush().unwrap();
    eprintln!(
        "wrote {} ({} tensors, {:.2} GiB) in {:.1}s",
        out_path.display(),
        plan.len() * 2,
        off as f64 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    );
    let _ = Path::new(&out_path);
}

#[cfg(not(feature = "cpu"))]
fn main() {
    eprintln!("build with --features cpu");
}
