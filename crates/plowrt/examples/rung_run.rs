//! Run ONE compiled block — a prefill "rung" — on real GPUs, across a TP group.
//!
//!   rung_run <blob.pkt> <hsaco-dir> [--checkpoint DIR] [--tp N] [--iters N]
//!            [--exit NAME] [--prog P]
//!
//! A rung is what `plowc --block L` emits: one layer, no embedding, no tail, no
//! tokenizer. Its entry is the activation `act.x` and its exit is whichever
//! buffer the layer leaves the answer in, so it cannot be driven through
//! `AmdTpGroup::prefill` (which begins by staging a prompt and a KV mapping) or
//! through `AmdServe` (which wants a servable model). It is driven by uploading
//! `act.x`, launching the program, and reading the exit back.
//!
//! SCOPE — shape, finiteness and timing. There is no parity check against a
//! reference implementation here: `act.x` is a seeded synthetic, so the numbers
//! that come back are meaningless as text. What they can show is that the layer
//! RAN: that every arm the packet names exists in the object, that nothing
//! faulted, that no NaN or Inf appeared, and how long the layer takes. Those are
//! the questions a first bring-up run has, and a wrong answer to any of them is
//! invisible without asking.
//!
//! The timing is the layer's, not the model's. Multiplying by the layer count
//! over-counts — a real model overlaps the seams this measures in isolation —
//! so it is a floor for one layer, not a projection.

#[cfg(not(feature = "hsa"))]
fn main() {
    eprintln!("rung_run requires --features hsa");
    std::process::exit(2);
}

#[cfg(feature = "hsa")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    hsa::run()
}

#[cfg(feature = "hsa")]
mod hsa {
    use plowrt::exec::amd_tp::AmdTpGroup;
    use std::path::PathBuf;
    use std::sync::Arc;

    /// `f32` to bf16, round-to-nearest-even — the same rounding the emitter's
    /// own conversions use, so a value written here reads back unchanged.
    fn bf16(x: f32) -> u16 {
        let b = x.to_bits();
        let lsb = (b >> 16) & 1;
        ((b + 0x7fff + lsb) >> 16) as u16
    }

    fn bf16_to_f32(h: u16) -> f32 {
        f32::from_bits((h as u32) << 16)
    }

    /// A deterministic hidden state. Small and centred, because the entry of a
    /// layer is a post-embedding residual and feeding it values a real model
    /// never sees would make an overflow look like a bug in the layer.
    fn seeded_activation(rows: usize, hidden: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(rows * hidden * 2);
        let mut s = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..rows * hidden {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = ((s >> 40) as f32) / (1u32 << 24) as f32; // [0, 1)
            out.extend_from_slice(&bf16((u - 0.5) * 0.08).to_le_bytes());
        }
        out
    }

    struct Stats {
        n: usize,
        nan: usize,
        inf: usize,
        min: f32,
        max: f32,
        mean: f64,
        zero: usize,
    }

    fn stats(bytes: &[u8]) -> Stats {
        let mut st = Stats {
            n: bytes.len() / 2,
            nan: 0,
            inf: 0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            mean: 0.0,
            zero: 0,
        };
        let mut sum = 0f64;
        for c in bytes.chunks_exact(2) {
            let v = bf16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            if v.is_nan() {
                st.nan += 1;
                continue;
            }
            if v.is_infinite() {
                st.inf += 1;
                continue;
            }
            if v == 0.0 {
                st.zero += 1;
            }
            st.min = st.min.min(v);
            st.max = st.max.max(v);
            sum += v as f64;
        }
        st.mean = sum / (st.n - st.nan - st.inf).max(1) as f64;
        st
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let mut args = std::env::args().skip(1);
        let blob = PathBuf::from(args.next().ok_or("usage: rung_run <blob.pkt> <hsaco-dir> ...")?);
        let hsaco = PathBuf::from(args.next().ok_or("usage: rung_run <blob.pkt> <hsaco-dir> ...")?);
        let mut checkpoint: Option<PathBuf> = None;
        let mut tp = 8u32;
        let mut iters = 10u32;
        let mut prog = 0usize;
        let mut exit = String::from("act.hc_residual_a");
        let mut entry = String::from("act.hc_residual_a");
        while let Some(a) = args.next() {
            match a.as_str() {
                "--checkpoint" => checkpoint = args.next().map(PathBuf::from),
                "--tp" => tp = args.next().ok_or("--tp needs a value")?.parse()?,
                "--iters" => iters = args.next().ok_or("--iters needs a value")?.parse()?,
                "--prog" => prog = args.next().ok_or("--prog needs a value")?.parse()?,
                "--exit" => exit = args.next().ok_or("--exit needs a value")?,
                "--entry" => entry = args.next().ok_or("--entry needs a value")?,
                other => return Err(format!("unknown argument {other}").into()),
            }
        }

        let mut backends = Vec::with_capacity(tp as usize);
        for d in 0..tp {
            backends.push(Arc::new(plowrt::device::hsa::HsaBackend::new(d as u8)?));
        }
        let t0 = std::time::Instant::now();
        let mut g = AmdTpGroup::load(backends, &blob, &hsaco, checkpoint.as_deref())?;
        println!(
            "loaded in {:.1} s: tp={}, weights_bound={}",
            t0.elapsed().as_secs_f64(),
            g.n_gpu(),
            g.weights_bound()
        );
        if !g.weights_bound() {
            println!(
                "  NOTE: no checkpoint bound. The run proves dispatch and finiteness only; \
                 every weight is whatever the allocator left behind, so the NUMBERS mean nothing."
            );
        }

        // The entry's declared size IS the row count: `act.x` is `[T][hidden]` bf16, and the
        // packet's own program width is the T that was compiled.
        let x_bytes = g.rank(0).tensor_bytes(&entry).ok_or_else(|| {
            format!("this packet has no `{entry}`; pass --entry with the right name")
        })? as usize;
        let out_bytes = g
            .rank(0)
            .tensor_bytes(&exit)
            .ok_or_else(|| format!("this packet has no `{exit}`; pass --exit with the right name"))?
            as usize;
        println!("entry {entry} = {x_bytes} B, exit {exit} = {out_bytes} B");

        // EVERY RANK gets the same entry. The residual stream is replicated under TP — the shard
        // is in the weights — so a rank seeded differently would diverge from its peers at the
        // first reduce and the collective would mix two different sequences.
        // THE WHOLE ENTRY, every byte of it. On a V4.1 rung the entry is `[hc_mult][T][hidden]` --
        // the mHC's parallel residual streams -- and filling only the first `[T][hidden]` leaves
        // `HyperConnPre` mixing three copies of whatever the arena holds. Not hypothetical: it
        // produced 167104 Inf out of 20971520 on a layer whose real input was bounded by +-0.04,
        // and it read as a numerics bug in the layer rather than as an unwritten buffer.
        let x = seeded_activation(x_bytes / 2, 1);
        for r in 0..g.n_gpu() {
            g.rank_mut(r).write_tensor(&entry, &x)?;
            // Positions 0..T for the rotary tables, and the attended length. A rung has no KV
            // history: it attends over its own rows, so kv_len is the bucket width.
            if let Some(b) = g.rank(r).tensor_bytes("in.pos") {
                let pos: Vec<u8> = (0..b / 4)
                    .flat_map(|i| (i as u32).to_le_bytes())
                    .collect();
                g.rank_mut(r).write_tensor("in.pos", &pos)?;
            }
            if g.rank(r).tensor_bytes("in.kvlen").is_some() {
                // Rows, not copies: the entry carries `hc_mult` of them.
                let rows = (x_bytes / 2 / 5120 / 4) as u32;
                g.rank_mut(r).write_tensor("in.kvlen", &rows.to_le_bytes())?;
            }
        }

        // Warm-up, then timed. The first launch pays for object load and page-in, which is a real
        // cost but not the layer's.
        g.run_rung(prog)?;
        let mut out = vec![0u8; out_bytes];
        g.rank(0).read_tensor(&exit, &mut out)?;
        let st = stats(&out);
        println!(
            "exit: {} elems  min {:.5}  max {:.5}  mean {:.6}  zero {}  NaN {}  Inf {}",
            st.n, st.min, st.max, st.mean, st.zero, st.nan, st.inf
        );
        if st.nan > 0 || st.inf > 0 {
            println!("  FAILED: the layer produced non-finite values");
        }
        if st.zero == st.n {
            println!(
                "  FAILED: every output element is zero. On AMD an unimplemented opcode does not \
                 trap — the dispatch `default:` leaves the buffer untouched — so an all-zero exit \
                 is what a missing arm looks like."
            );
        }

        let mut us: Vec<f64> = Vec::with_capacity(iters as usize);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            g.run_rung(prog)?;
            us.push(t.elapsed().as_secs_f64() * 1e6);
        }
        if !us.is_empty() {
            us.sort_by(f64::total_cmp);
            println!(
                "layer time over {iters} iters: min {:.1} us  median {:.1} us  max {:.1} us",
                us[0],
                us[us.len() / 2],
                us[us.len() - 1]
            );
            println!(
                "  a 40-layer model at this per-layer cost would be {:.1} ms, which OVER-COUNTS: \
                 a whole-model emit overlaps seams this measures in isolation.",
                us[us.len() / 2] * 40.0 / 1000.0
            );
        }
        Ok(())
    }
}
