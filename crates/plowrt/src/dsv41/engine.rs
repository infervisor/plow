//! The DeepSeek-V4.1 engine: pipeline stages (one GPU each), the stage-to-stage handoff, the host-side
//! Engram gather, embedding, head and sampling.
//!
//! A step is a PREFILL of one sequence or a DECODE of a batch of slots. The host enqueues the whole
//! step without blocking: stage s+1's stream waits on an event stage s recorded after its layers,
//! then pulls the hc stream, pre-mix, index picks and candidate mask over P2P. A kv source whose
//! consumers sit on a later stage writes the rows it produced into that stage's mirror directly
//! (a copy kernel on the producer's stream through the peer mapping). The one host sync per step
//! is the sampled tokens' readback.

use std::path::Path;
use std::sync::Arc;

use rayon::prelude::*;

use crate::asset::Checkpoint;
use crate::device::cuda::{CudaBackend, CudaEvent, PinnedHost};
use crate::device::{Backend, DeviceMem};
use crate::error::{Result, RuntimeError};
use crate::text::engram::{build_compressed_token_map, v41_hash_tables, EngramHasher};

use super::config::Cfg;
use super::kernels::{cdiv, KTotal, A};
use super::stage::{Shared, Stage, Step};
use super::weights::Layer;

/// The kernels index with 32-bit element offsets in places and put `t` in grid.y; beyond this the
/// engine refuses to start rather than overflow on a long request.
pub const MAX_LEN_LIMIT: usize = 32768;

pub struct EngineOpts {
    pub max_len: usize,
    pub max_slots: usize,
    /// Layer boundaries: stage s holds layers bounds[s]..bounds[s+1].
    pub bounds: Vec<usize>,
    pub arena_bytes: u64,
}

/// Per-sequence sampling: temperature 0 is greedy; otherwise Gumbel-max over logits / temperature
/// with a counter RNG keyed by (seed, step).
#[derive(Clone, Copy, Debug, Default)]
pub struct Sampling {
    pub temperature: f32,
    pub seed: u64,
    pub step: u64,
}

impl Sampling {
    pub fn at(self, step: u64) -> Self {
        Sampling { step, ..self }
    }
}

/// Per-slot host state: the Engram hasher (its n-gram lookback cache).
struct SlotHost {
    hasher: Option<EngramHasher>,
}

/// One Engram layer's host side: its table names in the (owned, mmapped) checkpoint and the pinned
/// staging buffer its gathered rows are uploaded from.
struct EngramTab {
    layer: usize,
    stage: usize,
    weight: String,
    scale: String,
    staging: std::cell::RefCell<PinnedHost>,
}

struct Mirror {
    src: usize,
    producer: usize,
    consumer: usize,
    /// Whether an index source on the consumer stage reads the index keys (else only the latents
    /// are mirrored).
    idxk: bool,
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
    engram: Vec<EngramTab>,
    hash_proto: Option<(Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<Vec<i64>>, Vec<u32>, usize, usize, usize, usize)>,
    slots: Vec<SlotHost>,
    mirrors: Vec<Mirror>,
    /// One event per stage, recorded after its layers; the next stage's stream waits on it.
    done: Vec<CudaEvent>,
    /// `PLOW_DSV41_PROFILE=1`: sync after every stage and report per-stage milliseconds; `=2`
    /// also per-kernel GPU time. Both perturb timing (the stages stop overlapping).
    prof: Option<std::cell::RefCell<Prof>>,
    prof_kernels: bool,
    /// Set by [`Engine::rung_bench`]: the step report leaves the per-kernel totals to it.
    rung_mode: bool,
}

#[derive(Default)]
struct Prof {
    steps: [u64; 2],
    host_ms: [f64; 2],
    stage_ms: [Vec<f64>; 2],
}

fn stage_of(bounds: &[usize], l: usize) -> usize {
    (0..bounds.len() - 1).find(|&s| l >= bounds[s] && l < bounds[s + 1]).expect("bounds validated")
}

fn validate(cfg: &Cfg, opts: &EngineOpts, n_devices: u32) -> Result<()> {
    let b = &opts.bounds;
    let bad = |m: String| Err(RuntimeError::Device(format!("dsv41: {m}")));
    if b.len() < 2 || b[0] != 0 || *b.last().unwrap() != cfg.n_layers || b.windows(2).any(|w| w[1] <= w[0]) {
        return bad(format!("--bounds must rise strictly from 0 to {} (one stage per GPU), got {b:?}", cfg.n_layers));
    }
    if b.len() - 1 > n_devices as usize {
        return bad(format!("{} stages but only {n_devices} visible GPUs", b.len() - 1));
    }
    if opts.max_len == 0 || opts.max_len > MAX_LEN_LIMIT {
        return bad(format!("--max-len must be in 1..={MAX_LEN_LIMIT}"));
    }
    if opts.max_slots == 0 || opts.max_slots > 1024 {
        return bad("--max-slots must be in 1..=1024".into());
    }
    Ok(())
}

impl Engine {
    pub fn load(ckpt_dir: &Path, cubin: &[u8], opts: EngineOpts) -> Result<Engine> {
        let cfg = Cfg::load(ckpt_dir)?;
        let probe = CudaBackend::new(0)?;
        validate(&cfg, &opts, probe.device_count()?)?;
        drop(probe);
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
        let prof_level = std::env::var("PLOW_DSV41_PROFILE").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
        // mirrors: a kv source whose consumers sit on a later stage
        let mut mirrors: Vec<Mirror> = Vec::new();
        for l in 0..cfg.n_layers {
            if cfg.compress_ratios[l] == 0 {
                continue;
            }
            let Some(src) = cfg.kv_src(l) else { continue };
            let (ps, cs) = (stage_of(&opts.bounds, src), stage_of(&opts.bounds, l));
            if ps == cs {
                continue;
            }
            let reads_idxk = cfg.index_source.contains(&l);
            match mirrors.iter_mut().find(|m| m.src == src && m.consumer == cs) {
                Some(m) => m.idxk |= reads_idxk,
                None => mirrors.push(Mirror { src, producer: ps, consumer: cs, idxk: reads_idxk }),
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
                    let mir: Vec<usize> = mirrors.iter().filter(|m| m.consumer == s).map(|m| m.src).collect();
                    let (max_len, max_slots, arena) = (opts.max_len, opts.max_slots, opts.arena_bytes);
                    sc.spawn(move || -> Result<Stage> {
                        let t0 = std::time::Instant::now();
                        let mut layers = Vec::new();
                        for l in bounds[s]..bounds[s + 1] {
                            layers.push(Layer::load(&dev, &ck, &cfg, l)?);
                        }
                        tracing::info!(target: "dsv41", "stage {s}: layers {}..{} loaded in {:.1}s", bounds[s], bounds[s + 1], t0.elapsed().as_secs_f32());
                        Stage::new(s, dev, cubin, cfg, layers, &mir, max_len, max_slots, arena, prof_level >= 2)
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err(RuntimeError::Device("dsv41: stage loader panicked".into()))))
                .collect::<Result<Vec<_>>>()
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
        // Engram tables stay in the mmap (host); rows are gathered per step into pinned staging
        let cols = (cfg.engram_max_ngram.max(1) - 1) * cfg.engram_n_heads;
        let mut engram = Vec::new();
        for &l in &cfg.engram_layers {
            let (weight, scale) = (format!("layers.{l}.engram.embed.weight"), format!("layers.{l}.engram.embed.scale"));
            if ck.tensor(&weight).is_none() || ck.tensor(&scale).is_none() {
                return Err(RuntimeError::Device(format!("dsv41: layer {l} has no engram table")));
            }
            let stage = stage_of(&opts.bounds, l);
            let staging = devs[stage].host_alloc_pinned(opts.max_len * cols * cfg.engram_head_dim * 2)?;
            engram.push(EngramTab { layer: l, stage, weight, scale, staging: std::cell::RefCell::new(staging) });
        }
        let hash_proto = if cfg.engram_layers.is_empty() {
            None
        } else {
            let tabs = v41_hash_tables(&cfg.engram_layers, cfg.engram_max_ngram, cfg.engram_n_heads, cfg.engram_vocab_size, cfg.engram_compressed_vocab)
                .ok_or_else(|| RuntimeError::Device("dsv41: no engram hash tables for this config".into()))?;
            let tok = tokenizers::Tokenizer::from_file(ckpt_dir.join("tokenizer.json")).map_err(|e| RuntimeError::Device(format!("tokenizer: {e}")))?;
            let (map, vocab) = build_compressed_token_map(&tok);
            Some((tabs.multipliers, tabs.primes, tabs.offsets, map, vocab, cfg.engram_compressed_vocab, cfg.engram_pad_id, opts.max_len + 64))
        };
        let done = devs.iter().map(|d| d.event_create(false)).collect::<Result<Vec<_>>>()?;
        let mut eng = Engine {
            cfg: Arc::new(cfg),
            stages,
            devs,
            ck,
            embed,
            head,
            norm,
            _globals: globals,
            engram,
            hash_proto,
            slots: Vec::new(),
            mirrors,
            done,
            prof: (prof_level >= 1).then(|| std::cell::RefCell::new(Prof::default())),
            prof_kernels: prof_level >= 2,
            rung_mode: false,
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

    /// Hash this step's tokens for every Engram layer: [rows][layers][cols] row ids.
    fn engram_hashes(&mut self, seqs: &[(usize, Vec<u32>, usize)]) -> Vec<i64> {
        let mut out = Vec::new();
        for (slot, ids, pos) in seqs {
            if let Some(h) = self.slots[*slot].hasher.as_mut() {
                h.hash(ids, *pos, None, &mut out);
            }
        }
        out
    }

    /// Gather + dequantize one Engram layer's rows into its pinned staging buffer (as
    /// ParallelEngramEmbedding: e4m3 x ue8m0 per 32 -> bf16), rows in parallel. Returns the bytes.
    fn engram_gather(&self, li: usize, hashes: &[i64], rows: usize) -> Result<usize> {
        let c = &self.cfg;
        let (hd, blk) = (c.engram_head_dim, 32usize);
        let n_el = self.engram.len();
        let cols = (c.engram_max_ngram - 1) * c.engram_n_heads;
        let tab = &self.engram[li];
        let w = self.ck.tensor(&tab.weight).ok_or_else(|| RuntimeError::Device("dsv41: engram table".into()))?;
        let s = self.ck.tensor(&tab.scale).ok_or_else(|| RuntimeError::Device("dsv41: engram scale".into()))?;
        let n_rows = w.len() / hd;
        let mut staging = tab.staging.borrow_mut();
        let bytes = rows * cols * hd * 2;
        if bytes > staging.len() {
            return Err(RuntimeError::Device("dsv41: engram rows exceed the staging buffer".into()));
        }
        let dst: &mut [u8] = &mut staging.as_mut_slice()[..bytes];
        // SAFETY: the pinned buffer is 2-byte aligned (page aligned) and `bytes` is even.
        let dst: &mut [u16] = unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut u16, bytes / 2) };
        let bad = dst.par_chunks_mut(hd).enumerate().map(|(k, out)| {
            let (i, col) = (k / cols, k % cols);
            let id = hashes[(i * n_el + li) * cols + col] as usize;
            if id >= n_rows {
                return 1usize;
            }
            let wr = &w[id * hd..(id + 1) * hd];
            let sr = &s[id * (hd / blk)..(id + 1) * (hd / blk)];
            for (j, d) in out.iter_mut().enumerate() {
                *d = f32_to_bf16(E4M3[wr[j] as usize] * f32::from_bits((sr[j / blk] as u32) << 23));
            }
            0
        }).sum::<usize>();
        if bad > 0 {
            return Err(RuntimeError::Device(format!("dsv41: {bad} engram hash ids out of table range")));
        }
        Ok(bytes)
    }

    /// Run one step. `seqs`: (slot, new token ids, start position). Prefill: exactly one sequence;
    /// decode: one token per sequence. `samp`: one per sequence. Returns the next token per sequence.
    ///
    /// On any error every stage is drained before returning, so no later step's arena reuse can
    /// race a copy or kernel this step left in flight.
    pub fn step(&mut self, seqs: &[(usize, Vec<u32>, usize)], decode: bool, samp: &[Sampling]) -> Result<Vec<u32>> {
        let r = self.step_inner(seqs, decode, samp);
        if r.is_err() {
            for s in &self.stages {
                let _ = s.stream_sync();
            }
        }
        r
    }

    fn step_inner(&mut self, seqs: &[(usize, Vec<u32>, usize)], decode: bool, samp: &[Sampling]) -> Result<Vec<u32>> {
        let c = self.cfg.clone();
        let (h, hcm) = (c.hidden, c.hc_mult);
        let t: usize = seqs.iter().map(|s| s.1.len()).sum();
        if (!decode && seqs.len() != 1) || (decode && seqs.iter().any(|s| s.1.len() != 1)) || samp.len() != seqs.len() || t == 0 {
            return Err(RuntimeError::Device("dsv41: a step is one prefill sequence or single-token decodes".into()));
        }
        let st = Step { decode, t, slots: seqs.iter().map(|s| s.0).collect(), pos: seqs.iter().map(|s| s.2).collect() };
        let hashes = if self.engram.is_empty() { Vec::new() } else { self.engram_hashes(seqs) };
        let ids: Vec<i32> = seqs.iter().flat_map(|s| s.1.iter().map(|&x| x as i32)).collect();
        let x_bytes = (t * hcm * h * 2) as u64;
        let p_bytes = (t * hcm * 4) as u64;
        let mut prev: Option<(usize, u64, u64, Shared)> = None;
        let n_stages = self.stages.len();
        let kind = decode as usize;
        let t_host = std::time::Instant::now();
        let mut t_stage = std::time::Instant::now();
        let mut stage_ms = vec![0f64; n_stages];
        let mut result = Vec::new();
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
                    // no host sync: this stream waits on the producer's event, then pulls over P2P
                    let src = &self.stages[ps];
                    stg.dev.stream_wait_event(&stg.stream, &self.done[ps])?;
                    stg.dev.memcpy_peer_async(xa, &src.dev, px, x_bytes, &stg.stream)?;
                    stg.dev.memcpy_peer_async(pa, &src.dev, pp, p_bytes, &stg.stream)?;
                    if psh.kout > 0 {
                        sh.topk = stg.topk_buf;
                        sh.kout = psh.kout;
                        stg.dev.memcpy_peer_async(sh.topk, &src.dev, psh.topk, (t * psh.kout * 4) as u64, &stg.stream)?;
                    }
                    if psh.nblk > 0 {
                        sh.keep = stg.keep_buf;
                        sh.nblk = psh.nblk;
                        stg.dev.memcpy_peer_async(sh.keep, &src.dev, psh.keep, (t * psh.nblk) as u64, &stg.stream)?;
                    }
                }
            }
            // Engram rows for this stage's engram layers, gathered while earlier stages run
            let mut emb_dev = std::collections::HashMap::new();
            for (li, tab) in self.engram.iter().enumerate() {
                if tab.stage != s {
                    continue;
                }
                let bytes = self.engram_gather(li, &hashes, t)?;
                let d = stg.alloc_persistent(bytes as u64)?;
                let staging = tab.staging.borrow();
                // SAFETY: pinned source, stable until the step's final sync (the next step's gather
                // into it happens after that sync).
                unsafe { stg.dev.memcpy_htod_async(d, &staging.as_slice()[..bytes], &stg.stream)? };
                emb_dev.insert(tab.layer, d);
            }
            let (mut x, mut pm, mut xo, mut po) = (xa, pa, xb, pb);
            for ly in &stg.layers {
                let emb = emb_dev.get(&ly.id).copied().unwrap_or(0);
                stg.layer(ly, x, pm, xo, po, emb, &st, &meta, &mut sh)?;
                std::mem::swap(&mut x, &mut xo);
                std::mem::swap(&mut pm, &mut po);
            }
            self.write_mirrors(s, &st)?;
            stg.dev.event_record(&self.done[s], &stg.stream)?;
            if self.prof.is_some() {
                stg.stream_sync()?;
                stage_ms[s] = t_stage.elapsed().as_secs_f64() * 1e3;
                t_stage = std::time::Instant::now();
            }
            if s + 1 < n_stages {
                prev = Some((s, x, pm, sh));
            } else {
                result = self.head_sample(stg, x, pm, &st, samp)?;
            }
        }
        if self.prof.is_some() {
            self.report(kind, &st, t_host.elapsed().as_secs_f64() * 1e3, &stage_ms)?;
        }
        Ok(result)
    }

    /// Copy the compressed-cache rows stage `s`'s kv sources wrote this step into the mirrors on the
    /// stages that read them: one copy kernel per mirrored region, on the producer's stream, writing
    /// through the peer mapping. Prefill: rows 0..t/r of the one slot. Decode: row pos/r of each slot
    /// whose group just completed.
    fn write_mirrors(&self, s: usize, st: &Step) -> Result<()> {
        let c = &self.cfg;
        let stg = &self.stages[s];
        for m in self.mirrors.iter().filter(|m| m.producer == s) {
            let cons = &self.stages[m.consumer];
            let r = c.compress_ratios[m.src];
            let mut regions = vec![(stg.caches.cmp[&m.src], cons.caches.cmp[&m.src], (c.head_dim * 2) as u64)];
            if m.idxk {
                regions.push((stg.caches.idxk[&m.src], cons.caches.idxk[&m.src], (c.index_dim * 2) as u64));
            }
            for (rp, rc, row_b) in regions {
                let mut dst = Vec::with_capacity(st.nb());
                let mut src = Vec::with_capacity(st.nb());
                let mut rows = Vec::with_capacity(st.nb());
                for (i, &slot) in st.slots.iter().enumerate() {
                    let (row0, n) = if !st.decode {
                        (0u64, (st.t / r) as i32)
                    } else {
                        ((st.pos[i] / r) as u64, ((st.pos[i] + 1) % r == 0) as i32)
                    };
                    dst.push(rc.at(slot) + row0 * row_b);
                    src.push(rp.at(slot) + row0 * row_b);
                    rows.push(n);
                }
                if rows.iter().all(|&n| n == 0) {
                    continue;
                }
                let (d_dst, d_src, d_rows) = (stg.alloc_persistent((dst.len() * 8) as u64)?, stg.alloc_persistent((src.len() * 8) as u64)?, stg.alloc_persistent((rows.len() * 4) as u64)?);
                upload(stg, d_dst, as_bytes(&dst))?;
                upload(stg, d_src, as_bytes(&src))?;
                upload(stg, d_rows, as_bytes(&rows))?;
                let max_rows = *rows.iter().max().unwrap() as u64;
                let chunks = cdiv(max_rows * row_b / 16, 256).clamp(1, 1024);
                stg.k.launch("dsv_copy_rows", [chunks, st.nb() as u32, 1], 256, 0, &[A::P(d_dst), A::P(d_src), A::P(d_rows), A::I(st.nb() as i32), A::I(row_b as i32)], &stg.stream)?;
            }
        }
        Ok(())
    }

    /// hc_pre with the final pre-mix, norm, fp32 head, optional Gumbel noise, argmax -- for the last
    /// row of each sequence. The step's one host sync is the token readback here.
    fn head_sample(&self, stg: &Stage, x: u64, pm: u64, st: &Step, samp: &[Sampling]) -> Result<Vec<u32>> {
        let c = &self.cfg;
        let (h, hcm, v) = (c.hidden, c.hc_mult, c.vocab);
        let nb = st.nb();
        let (xs, ps) = if st.decode {
            (x, pm) // decode rows are the sequences, in order
        } else {
            let last = (st.t - 1) as u64;
            (x + last * (hcm * h * 2) as u64, pm + last * (hcm * 4) as u64)
        };
        let hp = stg.alloc_persistent((nb * h * 2) as u64)?;
        stg.k.launch("dsv_hc_pre", [cdiv((nb * h) as u64, 256), 1, 1], 256, 0, &[A::P(hp), A::P(xs), A::P(ps), A::I(nb as i32), A::I(h as i32)], &stg.stream)?;
        let hn = stg.alloc_persistent((nb * h * 2) as u64)?;
        stg.k.launch("dsv_rmsnorm", [nb as u32, 1, 1], 256, 0, &[A::P(hn), A::P(hp), A::P(self.norm), A::I(h as i32), A::L(h as i64), A::L(h as i64), A::F(c.eps)], &stg.stream)?;
        let logits = stg.alloc_persistent((nb * v * 4) as u64)?;
        stg.k.gemm_f32(logits, hn, self.head, nb, v, h, h, true, true, &stg.stream)?;
        if samp.iter().any(|s| s.temperature > 0.0) {
            let temps: Vec<f32> = samp.iter().map(|s| s.temperature).collect();
            let seeds: Vec<u64> = samp.iter().map(|s| s.seed).collect();
            let steps: Vec<u64> = samp.iter().map(|s| s.step).collect();
            let (dt, ds, dn) = (stg.alloc_persistent((nb * 4) as u64)?, stg.alloc_persistent((nb * 8) as u64)?, stg.alloc_persistent((nb * 8) as u64)?);
            upload(stg, dt, as_bytes(&temps))?;
            upload(stg, ds, as_bytes(&seeds))?;
            upload(stg, dn, as_bytes(&steps))?;
            stg.k.launch("dsv_gumbel", [cdiv(v as u64, 1024).min(128), nb as u32, 1], 256, 0, &[A::P(logits), A::P(dt), A::P(ds), A::P(dn), A::I(v as i32)], &stg.stream)?;
        }
        let tok = stg.alloc_persistent((nb * 4) as u64)?;
        stg.k.launch("dsv_argmax", [nb as u32, 1, 1], 1024, 0, &[A::P(tok), A::P(logits), A::I(v as i32)], &stg.stream)?;
        stg.stream_sync()?;
        let mut out = vec![0u8; nb * 4];
        stg.dev.memcpy_dtoh(&mut out, tok)?;
        Ok(out.chunks(4).map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]])).collect())
    }

    fn report(&self, kind: usize, st: &Step, host_ms: f64, stage_ms: &[f64]) -> Result<()> {
        let Some(p) = &self.prof else { return Ok(()) };
        let mut p = p.borrow_mut();
        p.steps[kind] += 1;
        p.host_ms[kind] += host_ms;
        if p.stage_ms[kind].is_empty() {
            p.stage_ms[kind] = vec![0.0; stage_ms.len()];
        }
        for (a, b) in p.stage_ms[kind].iter_mut().zip(stage_ms) {
            *a += b;
        }
        if self.prof_kernels && !self.rung_mode {
            for s in &self.stages {
                s.k.prof_collect()?;
            }
        }
        let every = if st.decode { 64 } else { 1 };
        if p.steps[kind] % every != 0 {
            return Ok(());
        }
        let n = p.steps[kind] as f64;
        let per: Vec<String> = p.stage_ms[kind].iter().map(|v| format!("{:.2}", v / n)).collect();
        tracing::info!(target: "dsv41", "{} t={} nb={}: step {:.2} ms (stages [{}]) avg over {n}", if st.decode { "decode" } else { "prefill" }, st.t, st.nb(), p.host_ms[kind] / n, per.join(", "));
        if self.prof_kernels && !self.rung_mode {
            let rows = self.kernel_totals();
            let lines: Vec<String> = rows.iter().take(12).map(|(k, t)| {
                let eff = if t.ms > 0.0 && t.floor_ms > 0.0 { format!(" {:.0}% of roofline", 100.0 * t.floor_ms / t.ms) } else { String::new() };
                format!("{k} {:.2}ms/step ({} calls){eff}", t.ms / n, t.calls / n as u64)
            }).collect();
            tracing::info!(target: "dsv41", "  kernels: {}", lines.join("; "));
        }
        p.steps[kind] = 0;
        p.host_ms[kind] = 0.0;
        p.stage_ms[kind].iter_mut().for_each(|v| *v = 0.0);
        Ok(())
    }

    /// Drain every stage's per-kernel totals, merged by kernel name, largest time first.
    fn kernel_totals(&self) -> Vec<(&'static str, KTotal)> {
        let mut all: std::collections::HashMap<&'static str, KTotal> = std::collections::HashMap::new();
        for s in &self.stages {
            if let Some(kp) = &s.k.prof {
                for (k, t) in kp.borrow_mut().totals.drain() {
                    let e = all.entry(k).or_default();
                    e.calls += t.calls;
                    e.ms += t.ms;
                    e.flops += t.flops;
                    e.bytes += t.bytes;
                    e.floor_ms += t.floor_ms;
                    e.peak = e.peak.or(t.peak);
                }
            }
        }
        let mut v: Vec<_> = all.into_iter().collect();
        v.sort_by(|a, b| b.1.ms.total_cmp(&a.1.ms));
        v
    }

    fn set_kernel_events(&self, on: bool) {
        for s in &self.stages {
            if let Some(kp) = &s.k.prof {
                kp.borrow_mut().on = on;
            }
        }
    }

    /// The perf campaign's measurement: for each rung, the unperturbed step time, then a per-kernel
    /// breakdown against the roofline (needs `PLOW_DSV41_PROFILE=2`). `spec`:
    /// `prefill=1024,4096;decode=1x1024,64x1024` -- prefill token counts, decode batch x context.
    /// Decode rungs run on synthetic positions (cache contents do not change the work done).
    pub fn rung_bench(&mut self, spec: &str, reps: usize) -> Result<String> {
        let mut out = String::new();
        use std::fmt::Write as _;
        let _ = writeln!(out, "# dsv41 rung bench ({} stages, H200 roofline: HBM 4.8 TB/s, fp8 1979 / bf16 989 / fp32 67 TFLOP/s dense)\n", self.stages.len());
        let mut rungs: Vec<(bool, usize, usize)> = Vec::new(); // (decode, batch or tokens, context)
        for part in spec.split(';').map(str::trim).filter(|p| !p.is_empty()) {
            let (kind, list) = part.split_once('=').ok_or_else(|| RuntimeError::Device(format!("rung spec: {part}")))?;
            for item in list.split(',').map(str::trim) {
                match kind.trim() {
                    "prefill" => rungs.push((false, item.parse().map_err(|_| RuntimeError::Device(format!("rung: {item}")))?, 0)),
                    "decode" => {
                        let (b, ctx) = item.split_once('x').ok_or_else(|| RuntimeError::Device(format!("decode rung {item}: want BxCTX")))?;
                        rungs.push((true, b.parse().map_err(|_| RuntimeError::Device(format!("rung: {item}")))?, ctx.parse().map_err(|_| RuntimeError::Device(format!("rung: {item}")))?));
                    }
                    k => return Err(RuntimeError::Device(format!("rung spec: unknown kind {k}"))),
                }
            }
        }
        self.rung_mode = true;
        let tok = |i: usize| ((i * 7919 + 13) % 120_000 + 10) as u32;
        let greedy = Sampling::default();
        for (decode, n, ctx) in rungs {
            if n > self.slots.len() && decode {
                let _ = writeln!(out, "## decode B={n} ctx={ctx}: skipped (max_slots {})\n", self.slots.len());
                continue;
            }
            let run = |eng: &mut Engine, pos_base: usize| -> Result<()> {
                if decode {
                    let seqs: Vec<(usize, Vec<u32>, usize)> = (0..n).map(|b| (b, vec![tok(b)], pos_base + b)).collect();
                    let samp = vec![greedy; n];
                    eng.step(&seqs, true, &samp).map(|_| ())
                } else {
                    eng.release(0)?;
                    let ids: Vec<u32> = (0..n).map(tok).collect();
                    eng.step(&[(0, ids, 0)], false, &[greedy]).map(|_| ())
                }
            };
            // unperturbed timing
            self.set_kernel_events(false);
            run(self, ctx)?; // warm-up
            let t0 = std::time::Instant::now();
            for r in 0..reps {
                run(self, ctx + r)?;
            }
            let step_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
            // per-kernel breakdown
            let mut rows = Vec::new();
            if self.prof_kernels {
                self.set_kernel_events(true);
                let _ = self.kernel_totals();
                run(self, ctx)?;
                for s in &self.stages {
                    s.k.prof_collect()?;
                }
                rows = self.kernel_totals();
            }
            let (gpu_ms, floor_ms): (f64, f64) = rows.iter().fold((0.0, 0.0), |a, (_, t)| (a.0 + t.ms, a.1 + t.floor_ms));
            let mark = out.len();
            let title = if decode { format!("decode B={n} ctx={ctx}") } else { format!("prefill T={n}") };
            let tok_s = if decode { n as f64 / step_ms * 1e3 } else { n as f64 / step_ms * 1e3 };
            let _ = writeln!(out, "## {title}: step {step_ms:.2} ms ({tok_s:.0} tok/s); kernels {gpu_ms:.2} ms GPU, roofline floor {floor_ms:.2} ms ({:.1}% of floor)\n", if gpu_ms > 0.0 { 100.0 * floor_ms / gpu_ms } else { 0.0 });
            if !rows.is_empty() {
                let _ = writeln!(out, "| kernel | calls | ms | % step | TB/s | TFLOP/s | floor ms | eff | bound |");
                let _ = writeln!(out, "|---|---:|---:|---:|---:|---:|---:|---:|---|");
                for (k, t) in &rows {
                    let s = t.ms / 1e3;
                    let bound = match t.peak {
                        Some(p) if t.flops / super::kernels::peak_flops(p) > t.bytes / super::kernels::HBM_BYTES_PER_S => format!("{p:?}"),
                        Some(_) => "mem".to_string(),
                        None => "-".to_string(),
                    };
                    let _ = writeln!(
                        out,
                        "| {k} | {} | {:.3} | {:.1} | {} | {} | {} | {} | {bound} |",
                        t.calls,
                        t.ms,
                        100.0 * t.ms / gpu_ms.max(1e-9),
                        if t.bytes > 0.0 { format!("{:.2}", t.bytes / s / 1e12) } else { "-".into() },
                        if t.flops > 0.0 { format!("{:.1}", t.flops / s / 1e12) } else { "-".into() },
                        if t.floor_ms > 0.0 { format!("{:.3}", t.floor_ms) } else { "-".into() },
                        if t.floor_ms > 0.0 { format!("{:.0}%", 100.0 * t.floor_ms / t.ms) } else { "-".into() },
                    );
                }
                let _ = writeln!(out);
            }
            eprint!("{}", &out[mark..]);
        }
        Ok(out)
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

/// e4m3 (OCP) -> f32 for all 256 codes.
static E4M3: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(|| {
    let mut t = [0f32; 256];
    for (b, v) in t.iter_mut().enumerate() {
        let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = ((b >> 3) & 0xf) as i32;
        let m = (b & 7) as f32;
        *v = if e == 0 {
            s * m / 8.0 * 2f32.powi(-6)
        } else if e == 15 && (b & 7) == 7 {
            f32::NAN
        } else {
            s * (1.0 + m / 8.0) * 2f32.powi(e - 7)
        };
    }
    t
});

fn f32_to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    if v.is_nan() {
        return 0x7fc0;
    }
    let round = 0x7fff + ((b >> 16) & 1);
    ((b + round) >> 16) as u16
}
