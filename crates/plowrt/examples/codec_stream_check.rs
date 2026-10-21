//! Stateful stream steps vs whole decode through a `codec.pkt` (no GPU-free path):
//! `codec_stream_check CODEC.pkt FRAMES SEED CHUNK [STREAMS] [CODES.i32]`
//! CHUNK = new frames per step (1..=stream.step_frames) or `rand` (1..=17 per step). Phase 1
//! streams item 0 alone; phase 2 streams STREAMS items (default 4) in shared launches, each on its
//! own slot, joining one launch apart (different positions), each with its own chunk sizes.
//! CODES.i32 = [items][FRAMES][frame_codes] little-endian i32 (else random codes).
//! Prints per-stream rel-L2 / max |d| of the streamed PCM vs `decode.b1.f*` and sample counts.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("codec_stream_check requires --features cuda");
}

#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use plowrt::exec::packet_runtime::{load_packet_runtime_on, PacketAsset};

    let args: Vec<String> = std::env::args().collect();
    if !(5..=7).contains(&args.len()) {
        return Err(
            "usage: codec_stream_check CODEC.pkt FRAMES SEED CHUNK [STREAMS] [CODES.i32]".into(),
        );
    }
    let path = std::path::Path::new(&args[1]);
    let frames: usize = args[2].parse()?;
    let mut rng: u64 = args[3].parse::<u64>()? | 1;
    let chunk: Option<usize> = (args[4] != "rand").then(|| args[4].parse()).transpose()?;
    let streams: usize = args.get(5).map_or(Ok(4), |s| s.parse())?;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };

    let mut runtime = load_packet_runtime_on(path, "cuda", 0)?.runtime;
    let asset = PacketAsset::load(path)?;
    let pipe = asset.bind_driver("codec.v1", runtime.as_ref())?;
    let param = |k: &str| pipe.parameter(k).map(|v| v as usize);
    let (fc, fs) = (param("codec.frame_codes")?, param("codec.frame_samples")?);
    let (codebook, slots) = (param("codec.codebook")?, param("stream.slots")?);
    let step_frames = param("stream.step_frames")?;
    if param("stream.stateful")? != 1 || streams == 0 || streams > slots {
        return Err("packet has no stateful steps, or STREAMS exceeds its slots".into());
    }
    let scales: Vec<usize> = (0..param("lengths.count")?)
        .map(|k| param(&format!("lengths.{k}.rows_per_frame")))
        .collect::<Result<_, _>>()?;
    let (mut decodes, mut steps) = (Vec::new(), Vec::new());
    for (role, program) in pipe.programs() {
        let dims = |s: &str| -> Option<(usize, usize)> {
            let (b, f) = s.split_once('.')?;
            Some((
                b.strip_prefix('b')?.parse().ok()?,
                f.strip_prefix('f')?.parse().ok()?,
            ))
        };
        if let Some((b, f)) = role.strip_prefix("decode.").and_then(dims) {
            decodes.push((b, f, program));
        } else if let Some((b, n)) = role.strip_prefix("step.").and_then(dims) {
            steps.push((b, n, program));
        }
    }
    decodes.sort_by_key(|&(b, f, _)| (b * f, f));
    steps.sort_by_key(|&(b, n, _)| (b * n, n));
    let codes_t = pipe.tensor("codes")?;
    let pcm_t = pipe.tensor("pcm")?;
    let seed_t = pipe.tensor("seed")?;
    let (slot_t, pos_t, rope_t) = (
        pipe.tensor("stream.slot")?,
        pipe.tensor("stream.pos")?,
        pipe.tensor("stream.rope_pos")?,
    );
    let n_t: Vec<_> = (0..scales.len())
        .map(|k| pipe.tensor(&format!("stream.n.{k}")))
        .collect::<Result<_, _>>()?;

    let per = frames * fc;
    let codes: Vec<u32> = match args.get(6) {
        Some(file) => std::fs::read(file)?
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]).max(0) as u32)
            .collect(),
        None => (0..streams * per)
            .map(|_| (next() % codebook as u64) as u32)
            .collect(),
    };
    if codes.len() < streams * per {
        return Err(format!("CODES holds fewer than {streams} x {frames} frames").into());
    }
    let item = |s: usize| &codes[s * per..(s + 1) * per];

    // Whole decode, one item per launch.
    let &(_, cf, decode) = decodes
        .iter()
        .find(|&&(b, f, _)| b >= 1 && f >= frames)
        .ok_or("FRAMES exceeds every decode capacity")?;
    runtime.write_tensor(seed_t, &vec![0u8; seed_t.bytes])?;
    let mut whole = Vec::new();
    for s in 0..streams {
        let mut c = vec![0u32; cf * fc];
        c[..per].copy_from_slice(item(s));
        runtime.write_tensor_at(codes_t, 0, bytemuck::cast_slice(&c))?;
        for k in 0..scales.len() {
            let t = pipe.tensor(&format!("lengths.{k}"))?;
            let mut len = vec![0u32; t.bytes / 4];
            len[0] = (frames * scales[k]) as u32;
            runtime.write_tensor(t, bytemuck::cast_slice(&len))?;
        }
        runtime.run(decode)?;
        let mut pcm = vec![0f32; frames * fs];
        runtime.read_tensor_at(pcm_t, 0, bytemuck::cast_slice_mut(&mut pcm))?;
        whole.push(pcm);
    }

    // One launch: `(stream, slot, pos, n)` per active item. Padding items take unused slots with
    // n = 0 and pos = 1, which rewrites their state unchanged.
    let mut launch = |items: &[(usize, usize, usize, usize)]| -> Result<Vec<Vec<f32>>, String> {
        let nmax = items.iter().map(|i| i.3).max().unwrap_or(1);
        let &(cb, cn, program) = steps
            .iter()
            .find(|&&(b, n, _)| b >= items.len() && n >= nmax)
            .ok_or("no step capacity holds this launch")?;
        let e = |x: plowrt::RuntimeError| x.to_string();
        let mut c = vec![0u32; cb * cn * fc];
        let (mut slot, mut pos) = (vec![0u32; slot_t.bytes / 4], vec![1u32; pos_t.bytes / 4]);
        let mut rope = vec![0u32; rope_t.bytes / 4];
        let mut n = vec![0u32; cb];
        let used: Vec<usize> = items.iter().map(|i| i.1).collect();
        let mut spare = (0..slots).filter(|s| !used.contains(s));
        for b in 0..cb {
            match items.get(b) {
                Some(&(s, sl, p, k)) => {
                    c[b * cn * fc..][..k * fc].copy_from_slice(&item(s)[p * fc..][..k * fc]);
                    (slot[b], pos[b], n[b]) = (sl as u32, p as u32, k as u32);
                }
                None => slot[b] = spare.next().ok_or("no spare slot")? as u32,
            }
            for r in 0..cn {
                rope[b * cn + r] = pos[b] + r as u32;
            }
        }
        runtime
            .write_tensor_at(codes_t, 0, bytemuck::cast_slice(&c))
            .map_err(e)?;
        runtime
            .write_tensor(slot_t, bytemuck::cast_slice(&slot))
            .map_err(e)?;
        runtime
            .write_tensor(pos_t, bytemuck::cast_slice(&pos))
            .map_err(e)?;
        runtime
            .write_tensor(rope_t, bytemuck::cast_slice(&rope))
            .map_err(e)?;
        for (t, &scale) in n_t.iter().zip(&scales) {
            let mut v = vec![0u32; t.bytes / 4];
            for b in 0..cb {
                v[b] = n[b] * scale as u32;
            }
            runtime
                .write_tensor(*t, bytemuck::cast_slice(&v))
                .map_err(e)?;
        }
        runtime.run(program).map_err(e)?;
        let mut pcm = vec![0f32; cb * cn * fs];
        runtime
            .read_tensor_at(pcm_t, 0, bytemuck::cast_slice_mut(&mut pcm))
            .map_err(e)?;
        Ok(items
            .iter()
            .enumerate()
            .map(|(b, i)| pcm[b * cn * fs..][..i.3 * fs].to_vec())
            .collect())
    };
    let mut size = |left: usize| -> usize {
        chunk
            .unwrap_or_else(|| 1 + (next() % 17) as usize)
            .clamp(1, step_frames)
            .min(left)
    };
    let report = |label: &str, s: usize, got: &[f32]| {
        let want = &whole[s];
        let m = got.len().min(want.len());
        let (mut d2, mut w2, mut max) = (0f64, 0f64, 0f64);
        for (a, b) in got[..m].iter().zip(&want[..m]) {
            let d = f64::from(a - b);
            d2 += d * d;
            w2 += f64::from(*b) * f64::from(*b);
            max = max.max(d.abs());
        }
        println!(
            "{label} stream {s}: samples {} vs {} rel_l2 {:.3e} max {:.3e}",
            got.len(),
            want.len(),
            (d2 / w2.max(1e-30)).sqrt(),
            max
        );
    };

    // Phase 1: stream 0 alone on slot 0.
    let (mut got, mut t) = (Vec::new(), 0);
    while t < frames {
        let k = size(frames - t);
        got.extend(launch(&[(0, 0, t, k)])?.remove(0));
        t += k;
    }
    report("single", 0, &got);

    // Phase 2: every stream on its own slot, joining one launch apart.
    let slot_of = |s: usize| (slots - 1 - 5 * s % slots) % slots;
    let (mut got, mut at) = (vec![Vec::new(); streams], vec![0usize; streams]);
    let mut round = 0;
    while at.iter().any(|&t| t < frames) {
        let items: Vec<_> = (0..streams.min(round + 1))
            .filter(|&s| at[s] < frames)
            .map(|s| (s, slot_of(s), at[s], size(frames - at[s])))
            .collect();
        for (i, out) in items.iter().zip(launch(&items)?) {
            got[i.0].extend(out);
            at[i.0] += i.3;
        }
        round += 1;
    }
    for (s, g) in got.iter().enumerate() {
        report("mixed", s, g);
    }
    Ok(())
}
