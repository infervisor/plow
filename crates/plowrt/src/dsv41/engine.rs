//! The DeepSeek-V4.1 engine: four pipeline stages (one GPU each), the stage-to-stage handoff, the
//! host-side Engram gather, embedding, head and sampling.
//!
//! A step is a PREFILL of one sequence or a DECODE of a batch of slots. Stages run in order; each
//! hands its successor the hc stream, the pre-mix, the latest index picks and candidate mask, and
//! the rows it just wrote into any compressed cache the successor reads (its mirror of that cache).

use std::path::Path;
use std::sync::Arc;

use crate::asset::Checkpoint;
use crate::device::cuda::CudaBackend;
use crate::device::{Backend, DeviceMem};
use crate::error::{Result, RuntimeError};
use crate::text::engram::{build_compressed_token_map, v41_hash_tables, EngramHasher};

use super::config::Cfg;
use super::kernels::{cdiv, A};
use super::stage::{Shared, Stage, Step};
use super::weights::Layer;

pub struct EngineOpts {
    pub max_len: usize,
    pub max_slots: usize,
    /// Layer boundaries: stage s holds layers bounds[s]..bounds[s+1].
    pub bounds: Vec<usize>,
    pub arena_bytes: u64,
}

/// Per-slot host state: the Engram hasher (its n-gram lookback cache).
struct SlotHost {
    hasher: Option<EngramHasher>,
}

pub struct Engine {
    pub cfg: Arc<Cfg>,
    pub stages: Vec<Stage>,
    devs: Vec<Arc<CudaBackend>>,
    ck: Arc<Checkpoint>,
    embed: u64,
    head: u64,
    norm: u64,
    _globals: Vec<DeviceMem>,
    /// Engram: per engram layer, its table and scale tensors (mmapped) and its index among
    /// the hasher's layers.
    engram_tabs: Vec<(usize, &'static [u8], &'static [u8])>,
    hash_proto: Option<(Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<u32>, usize, usize, usize, usize)>,
    slots: Vec<SlotHost>,
    /// For each mirrored source layer: (producer stage, consumer stage).
    mirrors: Vec<(usize, usize, usize)>,
    /// `PLOW_DSV41_PROFILE=1`: sync after every stage and report per-stage milliseconds (perturbs timing:
    /// the stages stop overlapping host enqueue with device work).
    prof: Option<std::cell::RefCell<Prof>>,
}

#[derive(Default)]
struct Prof {
    steps: [u64; 2],
    host_ms: [f64; 2],
    stage_ms: [Vec<f64>; 2],
    head_ms: [f64; 2],
}

fn stage_of(bounds: &[usize], l: usize) -> usize {
    (0..bounds.len() - 1).find(|&s| l >= bounds[s] && l < bounds[s + 1]).expect("layer in range")
}

impl Engine {
    pub fn load(ckpt_dir: &Path, cubin: &[u8], opts: EngineOpts) -> Result<Engine> {
        let cfg = Cfg::load(ckpt_dir)?;
        let n_stages = opts.bounds.len() - 1;
        let ck = Arc::new(Checkpoint::open(ckpt_dir)?);
        let mut devs = Vec::new();
        for s in 0..n_stages {
            devs.push(Arc::new(CudaBackend::new(s as u8)?));
        }
        for a in 0..n_stages {
            for b in 0..n_stages {
                if a != b {
                    devs[a].enable_peer_access(&devs[b])?;
                }
            }
        }
        // mirrors: a kv source whose consumers sit on a later stage
        let mut mirrors = Vec::new();
        let mut mirrored: Vec<Vec<usize>> = vec![Vec::new(); n_stages];
        for l in 0..cfg.n_layers {
            if cfg.compress_ratios[l] == 0 {
                continue;
            }
            if let Some(src) = cfg.kv_src(l) {
                let (ps, cs) = (stage_of(&opts.bounds, src), stage_of(&opts.bounds, l));
                if ps != cs && !mirrored[cs].contains(&src) {
                    mirrored[cs].push(src);
                    mirrors.push((src, ps, cs));
                }
            }
        }
        // load stages in parallel, one thread per GPU
        let stages: Vec<Stage> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..n_stages)
                .map(|s| {
                    let dev = devs[s].clone();
                    let cfg = cfg.clone();
                    let ck = ck.clone();
                    let bounds = opts.bounds.clone();
                    let mir = mirrored[s].clone();
                    let (max_len, max_slots, arena) = (opts.max_len, opts.max_slots, opts.arena_bytes);
                    sc.spawn(move || -> Result<Stage> {
                        let mut layers = Vec::new();
                        for l in bounds[s]..bounds[s + 1] {
                            let t0 = std::time::Instant::now();
                            layers.push(Layer::load(&dev, &ck, &cfg, l)?);
                            eprintln!("dsv41: stage {s} layer {l} loaded in {:.1}s", t0.elapsed().as_secs_f32());
                        }
                        Stage::new(s, dev, cubin, cfg, layers, &mir, max_len, max_slots, arena)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("stage loader panicked")).collect::<Result<Vec<_>>>()
        })?;
        // embedding on stage 0, final norm + head on the last stage
        let mut globals = Vec::new();
        let mut upload = |d: &CudaBackend, name: &str| -> Result<u64> {
            let b = ck.tensor(name).ok_or_else(|| RuntimeError::Device(format!("dsv41: no {name}")))?;
            let m = d.alloc(d.device_ordinal, b.len() as u64)?;
            d.memcpy_htod(m.base, b)?;
            let p = m.base;
            globals.push(m);
            Ok(p)
        };
        let embed = upload(&devs[0], "embed.weight")?;
        let head = upload(&devs[n_stages - 1], "head.weight")?;
        let norm = upload(&devs[n_stages - 1], "norm.weight")?;
        // Engram tables stay in the mmap (host), gathered per step
        let mut engram_tabs = Vec::new();
        for &l in &cfg.engram_layers {
            let w = ck.tensor(&format!("layers.{l}.engram.embed.weight")).ok_or_else(|| RuntimeError::Device("dsv41: engram table".into()))?;
            let s = ck.tensor(&format!("layers.{l}.engram.embed.scale")).ok_or_else(|| RuntimeError::Device("dsv41: engram scale".into()))?;
            // SAFETY: the checkpoint mmap lives as long as `ck`, which the engine owns.
            let (w, s): (&'static [u8], &'static [u8]) = unsafe { (std::mem::transmute(w), std::mem::transmute(s)) };
            engram_tabs.push((l, w, s));
        }
        let hash_proto = if cfg.engram_layers.is_empty() {
            None
        } else {
            let tabs = v41_hash_tables(&cfg.engram_layers, cfg.engram_max_ngram, cfg.engram_n_heads, cfg.engram_vocab_size, cfg.engram_compressed_vocab)
                .ok_or_else(|| RuntimeError::Device("dsv41: no engram hash tables for this config".into()))?;
            let tok = tokenizers::Tokenizer::from_file(ckpt_dir.join("tokenizer.json")).map_err(|e| RuntimeError::Device(format!("tokenizer: {e}")))?;
            let (map, vocab) = build_compressed_token_map(&tok);
            Some((
                tabs.multipliers,
                tabs.primes,
                tabs.offsets,
                map,
                vocab,
                cfg.engram_compressed_vocab,
                cfg.engram_pad_id,
                opts.max_len + 64,
            ))
        };
        let mut eng = Engine {
            cfg: Arc::new(cfg),
            stages,
            devs,
            ck,
            embed,
            head,
            norm,
            _globals: globals,
            engram_tabs,
            hash_proto,
            slots: Vec::new(),
            mirrors,
            prof: std::env::var("PLOW_DSV41_PROFILE").ok().filter(|v| v != "0").map(|_| std::cell::RefCell::new(Prof::default())),
        };
        for _ in 0..opts.max_slots {
            let h = eng.new_hasher()?;
            eng.slots.push(SlotHost { hasher: h });
        }
        Ok(eng)
    }

    fn new_hasher(&self) -> Result<Option<EngramHasher>> {
        let Some((m, p, o, map, vocab, cfg_vocab, pad, len)) = &self.hash_proto else {
            return Ok(None);
        };
        let c = &self.cfg;
        EngramHasher::new(m.clone(), p.clone(), o.clone(), c.engram_max_ngram, c.engram_n_heads, map.clone(), *vocab, *cfg_vocab, *pad, *len)
            .map(Some)
            .map_err(RuntimeError::Device)
    }

    /// Forget a slot's sequence (its caches are overwritten by the next prefill).
    pub fn release(&mut self, slot: usize) -> Result<()> {
        if let Some(h) = self.slots[slot].hasher.as_mut() {
            h.truncate(0);
        }
        Ok(())
    }

    /// Engram gather for `ids` at `pos..` of `slot` (appended to that slot's hasher):
    /// per engram layer, bf16 [n][cols*head_dim] rows dequantized as ParallelEngramEmbedding does.
    fn engram_rows(&mut self, per_slot: &[(usize, &[u32], usize)]) -> Vec<Vec<u16>> {
        let c = &self.cfg;
        let hd = c.engram_head_dim;
        let blk = 32usize;
        let n_el = self.engram_tabs.len();
        let cols = (c.engram_max_ngram - 1) * c.engram_n_heads;
        let n_total: usize = per_slot.iter().map(|x| x.1.len()).sum();
        let mut out = vec![vec![0u16; n_total * cols * hd]; n_el];
        let mut row = 0usize;
        for &(slot, ids, pos) in per_slot {
            let mut hashes = Vec::new();
            if let Some(h) = self.slots[slot].hasher.as_mut() {
                h.hash(ids, pos, None, &mut hashes);
            }
            for i in 0..ids.len() {
                for (li, &(_, w, s)) in self.engram_tabs.iter().enumerate() {
                    for col in 0..cols {
                        let id = hashes[(i * n_el + li) * cols + col] as usize;
                        let wr = &w[id * hd..(id + 1) * hd];
                        let sr = &s[id * (hd / blk)..(id + 1) * (hd / blk)];
                        let dst = &mut out[li][((row + i) * cols + col) * hd..((row + i) * cols + col + 1) * hd];
                        for (j, d) in dst.iter_mut().enumerate() {
                            let v = e4m3_to_f32(wr[j]) * f32::from_bits((sr[j / blk] as u32) << 23);
                            *d = f32_to_bf16(v);
                        }
                    }
                }
            }
            row += ids.len();
        }
        out
    }

    /// Run one step. `seqs`: (slot, new token ids, start position). Prefill: exactly one sequence;
    /// decode: one token per sequence. Returns the greedy next token for each sequence.
    pub fn step(&mut self, seqs: &[(usize, Vec<u32>, usize)], decode: bool) -> Result<Vec<u32>> {
        let c = self.cfg.clone();
        let (h, hcm) = (c.hidden, c.hc_mult);
        let t: usize = seqs.iter().map(|s| s.1.len()).sum();
        let st = Step { decode, t, slots: seqs.iter().map(|s| s.0).collect(), pos: seqs.iter().map(|s| s.2).collect() };
        if !decode && seqs.len() != 1 {
            return Err(RuntimeError::Device("dsv41: prefill takes one sequence".into()));
        }
        // Engram gather (host)
        let per: Vec<(usize, &[u32], usize)> = seqs.iter().map(|(s, ids, p)| (*s, ids.as_slice(), *p)).collect();
        let emb_host = if self.engram_tabs.is_empty() { Vec::new() } else { self.engram_rows(&per) };
        let ids: Vec<i32> = seqs.iter().flat_map(|s| s.1.iter().map(|&x| x as i32)).collect();
        let x_bytes = (t * hcm * h * 2) as u64;
        let p_bytes = (t * hcm * 4) as u64;
        let mut prev: Option<(usize, u64, u64, Shared)> = None;
        let n_stages = self.stages.len();
        let mut result = Vec::new();
        let kind = decode as usize;
        let t_host = std::time::Instant::now();
        let mut t_stage = std::time::Instant::now();
        let mut stage_ms = vec![0f64; n_stages];
        for s in 0..n_stages {
            let stg = &self.stages[s];
            stg.arena.reset(0);
            let meta = stg.step_meta(&st)?;
            let xa = stg.alloc_persistent(x_bytes)?;
            let xb = stg.alloc_persistent(x_bytes)?;
            let pa = stg.alloc_persistent(p_bytes)?;
            let pb = stg.alloc_persistent(p_bytes)?;
            let mut sh = Shared::default();
            match prev {
                None => {
                    let d_ids = stg.alloc_persistent((t * 4) as u64)?;
                    upload(stg, d_ids, as_bytes(&ids))?;
                    stg.k.launch("dsv_embed_hc", [cdiv((t * h / 8) as u64, 256), 1, 1], 256, 0, &[A::P(xa), A::P(self.embed), A::P(d_ids), A::I(t as i32), A::I(h as i32)], &stg.stream)?;
                    let mut pm = vec![0f32; t * hcm];
                    for r in 0..t {
                        pm[r * hcm] = 1.0;
                    }
                    upload(stg, pa, as_bytes(&pm))?;
                }
                Some((ps, px, pp, psh)) => {
                    let src = &self.stages[ps];
                    src.stream_sync()?;
                    stg.dev.memcpy_peer_async(xa, &src.dev, px, x_bytes, &stg.stream)?;
                    stg.dev.memcpy_peer_async(pa, &src.dev, pp, p_bytes, &stg.stream)?;
                    if psh.kout > 0 {
                        let b = (t * psh.kout * 4) as u64;
                        sh.topk = stg.topk_buf;
                        sh.kout = psh.kout;
                        stg.dev.memcpy_peer_async(sh.topk, &src.dev, psh.topk, b, &stg.stream)?;
                    }
                    if psh.nblk > 0 {
                        let b = (t * psh.nblk) as u64;
                        sh.keep = stg.keep_buf;
                        sh.nblk = psh.nblk;
                        stg.dev.memcpy_peer_async(sh.keep, &src.dev, psh.keep, b, &stg.stream)?;
                    }
                    // mirrors: the rows the producer wrote this step
                    for &(src_l, pst, cst) in &self.mirrors {
                        if cst != s {
                            continue;
                        }
                        let prod = &self.stages[pst];
                        let r = c.compress_ratios[src_l];
                        for (reg_p, reg_c, row_b) in [
                            (prod.caches.cmp[&src_l], stg.caches.cmp[&src_l], (c.head_dim * 2) as u64),
                            (prod.caches.idxk[&src_l], stg.caches.idxk[&src_l], (c.index_dim * 2) as u64),
                        ] {
                            for (i, &slot) in st.slots.iter().enumerate() {
                                let (row0, rows) = if !decode {
                                    (0u64, (t / r) as u64)
                                } else if (st.pos[i] + 1) % r == 0 {
                                    ((st.pos[i] / r) as u64, 1)
                                } else {
                                    continue;
                                };
                                if rows == 0 {
                                    continue;
                                }
                                stg.dev.memcpy_peer_async(reg_c.at(slot) + row0 * row_b, &prod.dev, reg_p.at(slot) + row0 * row_b, rows * row_b, &stg.stream)?;
                            }
                        }
                    }
                }
            }
            // Engram rows for this stage's engram layers
            let mut emb_dev = std::collections::HashMap::new();
            for (li, &(l, _, _)) in self.engram_tabs.iter().enumerate() {
                if stg.layers.iter().any(|ly| ly.id == l) {
                    let d = stg.alloc_persistent((emb_host[li].len() * 2) as u64)?;
                    upload(stg, d, as_bytes(&emb_host[li]))?;
                    emb_dev.insert(l, d);
                }
            }
            let (mut x, mut pm, mut xo, mut po) = (xa, pa, xb, pb);
            for ly in &stg.layers {
                let emb = emb_dev.get(&ly.id).copied().unwrap_or(0);
                stg.layer(ly, x, pm, xo, po, emb, &st, &meta, &mut sh)?;
                std::mem::swap(&mut x, &mut xo);
                std::mem::swap(&mut pm, &mut po);
            }
            if self.prof.is_some() {
                stg.stream_sync()?;
                stage_ms[s] = t_stage.elapsed().as_secs_f64() * 1e3;
                t_stage = std::time::Instant::now();
            }
            if s + 1 < n_stages {
                prev = Some((s, x, pm, sh));
            } else {
                result = self.head_sample(stg, x, pm, &st)?;
            }
        }
        if let Some(p) = &self.prof {
            let mut p = p.borrow_mut();
            let head = t_stage.elapsed().as_secs_f64() * 1e3;
            p.steps[kind] += 1;
            p.host_ms[kind] += t_host.elapsed().as_secs_f64() * 1e3;
            p.head_ms[kind] += head;
            if p.stage_ms[kind].is_empty() {
                p.stage_ms[kind] = vec![0.0; n_stages];
            }
            for (a, b) in p.stage_ms[kind].iter_mut().zip(&stage_ms) {
                *a += b;
            }
            let every = if decode { 64 } else { 1 };
            if p.steps[kind] % every == 0 {
                let n = p.steps[kind] as f64;
                let per: Vec<String> = p.stage_ms[kind].iter().map(|v| format!("{:.2}", v / n)).collect();
                tracing::info!(
                    target: "dsv41",
                    "{} t={t} nb={}: step {:.2} ms (stages [{}] head {:.2}) avg over {n}",
                    if decode { "decode" } else { "prefill" },
                    st.nb(),
                    p.host_ms[kind] / n,
                    per.join(", "),
                    p.head_ms[kind] / n,
                );
                if decode {
                    p.steps[1] = 0;
                    p.host_ms[1] = 0.0;
                    p.head_ms[1] = 0.0;
                    p.stage_ms[1].iter_mut().for_each(|v| *v = 0.0);
                }
            }
        }
        Ok(result)
    }

    /// hc_pre with the final pre-mix, norm, fp32 head, argmax -- for the last row of each sequence.
    fn head_sample(&self, stg: &Stage, x: u64, pm: u64, st: &Step) -> Result<Vec<u32>> {
        let c = &self.cfg;
        let (h, hcm, v) = (c.hidden, c.hc_mult, c.vocab);
        let nb = st.nb();
        // rows to sample: prefill -> last row; decode -> all
        let rows: Vec<i32> = if st.decode { (0..nb as i32).collect() } else { vec![(st.t - 1) as i32] };
        let d_rows = stg.alloc_persistent((rows.len() * 4) as u64)?;
        upload(stg, d_rows, as_bytes(&rows))?;
        let xs = stg.alloc_persistent((nb * hcm * h * 2) as u64)?;
        stg.k.launch("dsv_gather_rows", [cdiv((hcm * h * 2 / 16) as u64, 256), nb as u32, 1], 256, 0, &[A::P(xs), A::P(x), A::P(d_rows), A::I(nb as i32), A::I((hcm * h * 2) as i32)], &stg.stream)?;
        let ps = stg.alloc_persistent((nb * hcm * 4) as u64)?;
        stg.k.launch("dsv_gather_rows", [1, nb as u32, 1], 32, 0, &[A::P(ps), A::P(pm), A::P(d_rows), A::I(nb as i32), A::I((hcm * 4) as i32)], &stg.stream)?;
        let hp = stg.alloc_persistent((nb * h * 2) as u64)?;
        stg.k.launch("dsv_hc_pre", [cdiv((nb * h) as u64, 256), 1, 1], 256, 0, &[A::P(hp), A::P(xs), A::P(ps), A::I(nb as i32), A::I(h as i32)], &stg.stream)?;
        let hn = stg.alloc_persistent((nb * h * 2) as u64)?;
        stg.k.launch("dsv_rmsnorm", [nb as u32, 1, 1], 256, 0, &[A::P(hn), A::P(hp), A::P(self.norm), A::I(h as i32), A::L(h as i64), A::L(h as i64), A::F(c.eps)], &stg.stream)?;
        let logits = stg.alloc_persistent((nb * v * 4) as u64)?;
        stg.k.gemm_f32(logits, hn, self.head, nb, v, h, h, true, true, &stg.stream)?;
        let tok = stg.alloc_persistent((nb * 4) as u64)?;
        stg.k.launch("dsv_argmax", [nb as u32, 1, 1], 1024, 0, &[A::P(tok), A::P(logits), A::I(v as i32)], &stg.stream)?;
        stg.stream_sync()?;
        let mut out = vec![0u8; nb * 4];
        stg.dev.memcpy_dtoh(&mut out, tok)?;
        Ok(out.chunks(4).map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect())
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }
    pub fn devices(&self) -> &[Arc<CudaBackend>] {
        &self.devs
    }
    pub fn checkpoint(&self) -> &Checkpoint {
        &self.ck
    }
}

fn upload(stg: &Stage, dst: u64, b: &[u8]) -> Result<()> {
    if b.is_empty() {
        return Ok(());
    }
    // SAFETY: pageable source staged by the driver before returning; ordered on the stage stream.
    unsafe { stg.dev.memcpy_htod_async(dst, b, &stg.stream) }
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data slice viewed as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

fn e4m3_to_f32(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0xf) as i32;
    let m = (b & 7) as f32;
    if e == 0 {
        s * m / 8.0 * 2f32.powi(-6)
    } else if e == 15 && (b & 7) == 7 {
        f32::NAN
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

fn f32_to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return 0x7fc0;
    }
    let round = 0x7fff + ((b >> 16) & 1);
    ((b + round) >> 16) as u16
}
