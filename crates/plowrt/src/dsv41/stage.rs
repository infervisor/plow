//! One pipeline stage of the DeepSeek-V4.1 sm_90a engine: a contiguous run of layers on one GPU,
//! their per-slot caches, and the per-layer launch schedule (verified against the reference in
//! `scripts/dsv41_nv/test_layers.py`, whose `pyengine.py` is this file's single-sequence twin).
//!
//! A step is either a PREFILL of one sequence (T rows at positions 0..T) or a DECODE of B slots
//! (one row each, at its own position).

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::device::cuda::{CudaBackend, CudaStream};
use crate::device::{Backend, DeviceMem};
use crate::error::{Result, RuntimeError};

use super::config::Cfg;
use super::kernels::{cdiv, Kernels, A, IX_SMEM, SA_SMEM};
use super::weights::Layer;

/// Bump allocator over one device allocation; `mark`/`reset` scope per-layer scratch.
pub struct Arena {
    mem: DeviceMem,
    off: Cell<u64>,
}

impl Arena {
    pub fn new(dev: &CudaBackend, bytes: u64) -> Result<Self> {
        Ok(Arena { mem: dev.alloc(dev.device_ordinal, bytes)?, off: Cell::new(0) })
    }
    pub fn alloc(&self, bytes: u64) -> Result<u64> {
        let at = self.off.get().next_multiple_of(256);
        if at + bytes > self.mem.len {
            return Err(RuntimeError::Device(format!("dsv41 arena: {} + {} > {}", at, bytes, self.mem.len)));
        }
        self.off.set(at + bytes);
        Ok(self.mem.base + at)
    }
    pub fn mark(&self) -> u64 {
        self.off.get()
    }
    pub fn reset(&self, m: u64) {
        self.off.set(m);
    }
}

/// Per-slot cache regions of one stage: `base + slot * stride`.
#[derive(Clone, Copy, Default)]
pub struct Region {
    pub base: u64,
    pub stride: u64,
}
impl Region {
    pub fn at(&self, slot: usize) -> u64 {
        self.base + self.stride * slot as u64
    }
}

pub struct Caches {
    /// Window ring per layer: [slots][WIN][HD] bf16.
    pub win: HashMap<usize, Region>,
    /// Compressed KV per kv-source layer (own or mirrored): [slots][cap][HD] bf16.
    pub cmp: HashMap<usize, Region>,
    /// Index keys per kv-source layer: [slots][cap][IXD] bf16.
    pub idxk: HashMap<usize, Region>,
    /// Compressor partial-group state per ratio>1 source: kv and score, [slots][r][HD] f32.
    pub st_kv: HashMap<usize, Region>,
    pub st_sc: HashMap<usize, Region>,
    _mem: Vec<DeviceMem>,
}

/// Shared attention state across a step's layers (model.py SharedAttentionRuntime).
#[derive(Clone, Copy, Default)]
pub struct Shared {
    /// Latest index picks: [rows][kout] int32, offset applied; kout = 0 when none.
    pub topk: u64,
    pub kout: usize,
    /// Candidate keep mask [rows][nblk] u8 from the candidate source, nblk = 0 when none.
    pub keep: u64,
    pub nblk: usize,
}

/// What one step computes, as the stage sees it.
pub struct Step {
    pub decode: bool,
    /// Rows (tokens) in this step.
    pub t: usize,
    /// Sequences: slot and start position (prefill: one, at 0).
    pub slots: Vec<usize>,
    pub pos: Vec<usize>,
}

impl Step {
    pub fn nb(&self) -> usize {
        self.slots.len()
    }
    pub fn q_per_b(&self) -> usize {
        if self.decode { 1 } else { self.t }
    }
}

pub struct Stage {
    pub idx: usize,
    pub dev: Arc<CudaBackend>,
    pub k: Kernels,
    pub stream: CudaStream,
    pub cfg: Cfg,
    pub layers: Vec<Layer>,
    pub caches: Caches,
    pub arena: Arena,
    pub max_len: usize,
    pub max_slots: usize,
    rope_w: (u64, u64),
    rope_c: (u64, u64),
    _rope_mem: DeviceMem,
    /// The latest index picks [max_len][index_topk] i32 and candidate keep mask
    /// [max_len][max_len / cand_block] u8: written by index / candidate sources, read by the layers
    /// after them. Outside the arena so the per-layer scratch reset never reclaims them.
    pub topk_buf: u64,
    pub keep_buf: u64,
    _shared_mem: Vec<DeviceMem>,
}

const WIN_DT: u64 = 2;

impl Stage {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        idx: usize,
        dev: Arc<CudaBackend>,
        cubin: &[u8],
        cfg: Cfg,
        layers: Vec<Layer>,
        mirrored_sources: &[usize],
        max_len: usize,
        max_slots: usize,
        arena_bytes: u64,
        profile_kernels: bool,
    ) -> Result<Stage> {
        let k = Kernels::load(dev.clone(), cubin, profile_kernels)?;
        let stream = dev.stream_create()?;
        let hd = cfg.head_dim as u64;
        let ixd = cfg.index_dim as u64;
        let win = cfg.window as u64;
        let ns = max_slots as u64;
        let mut mems = Vec::new();
        let mut region = |bytes_per_slot: u64| -> Result<Region> {
            let stride = bytes_per_slot.next_multiple_of(256);
            let m = dev.alloc(dev.device_ordinal, stride * ns)?;
            dev.memset_d8(m.base, 0, (stride * ns) as usize)?;
            let r = Region { base: m.base, stride };
            mems.push(m);
            Ok(r)
        };
        let (mut w, mut c, mut ik, mut sk, mut ss) = (HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new(), HashMap::new());
        let mut sources: Vec<usize> = layers.iter().filter(|l| l.kv_source).map(|l| l.id).collect();
        sources.extend_from_slice(mirrored_sources);
        for l in &layers {
            w.insert(l.id, region(win * hd * WIN_DT)?);
        }
        for &s in &sources {
            let r = cfg.compress_ratios[s] as u64;
            // + 64 rows: the indexer's score kernel reads whole 64-row key tiles
            let cap = max_len as u64 / r + 64;
            c.insert(s, region(cap * hd * 2)?);
            ik.insert(s, region(cap * ixd * 2)?);
        }
        for l in layers.iter().filter(|l| l.kv_source && l.ratio > 1) {
            sk.insert(l.id, region(l.ratio as u64 * hd * 4)?);
            ss.insert(l.id, region(l.ratio as u64 * hd * 4)?);
        }
        let caches = Caches { win: w, cmp: c, idxk: ik, st_kv: sk, st_sc: ss, _mem: mems };
        // rope tables: window-only (plain) and compressed (YaRN)
        let max_pos = max_len + 64;
        let half = cfg.rope_dim / 2;
        let tb = (max_pos * half * 4) as u64;
        let rope_mem = dev.alloc(dev.device_ordinal, 4 * tb.next_multiple_of(256))?;
        let at = |i: u64| rope_mem.base + i * tb.next_multiple_of(256);
        let (cw, sw) = cfg.rope_tables(max_pos, false);
        let (cc, sc) = cfg.rope_tables(max_pos, true);
        for (i, t) in [&cw, &sw, &cc, &sc].iter().enumerate() {
            dev.memcpy_htod(at(i as u64), bytemuck_f32(t))?;
        }
        let arena = Arena::new(&dev, arena_bytes)?;
        let topk_mem = dev.alloc(dev.device_ordinal, (max_len * cfg.index_topk * 4) as u64)?;
        let keep_mem = dev.alloc(dev.device_ordinal, (max_len * max_len.div_ceil(cfg.cand_block)) as u64)?;
        let (topk_buf, keep_buf) = (topk_mem.base, keep_mem.base);
        Ok(Stage {
            idx,
            rope_w: (at(0), at(1)),
            rope_c: (at(2), at(3)),
            _rope_mem: rope_mem,
            topk_buf,
            keep_buf,
            _shared_mem: vec![topk_mem, keep_mem],
            dev,
            k,
            stream,
            cfg,
            layers,
            caches,
            arena,
            max_len,
            max_slots,
        })
    }

    fn launch(&self, name: &str, grid: [u32; 3], block: u32, smem: u32, args: &[A]) -> Result<()> {
        self.k.launch(name, grid, block, smem, args, &self.stream)
    }

    /// Upload a small host array into the arena.
    fn put<T: Copy>(&self, v: &[T]) -> Result<u64> {
        let bytes = std::mem::size_of_val(v);
        let p = self.arena.alloc(bytes.max(4) as u64)?;
        // SAFETY: plain-old-data slice viewed as bytes.
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, bytes) };
        if !b.is_empty() {
            // Stream-ordered: the arena slot may still be read by an earlier kernel on this stream.
            // SAFETY: pageable source; the driver stages it before returning.
            unsafe { self.dev.memcpy_htod_async(p, b, &self.stream)? };
        }
        Ok(p)
    }

    // ------------------------------------------------------------------ primitives
    fn rmsnorm(&self, y: u64, x: u64, w: u64, m: usize, d: usize) -> Result<()> {
        self.launch("dsv_rmsnorm", [m as u32, 1, 1], 256, 0, &[A::P(y), A::P(x), A::P(w), A::I(d as i32), A::L(d as i64), A::L(d as i64), A::F(self.cfg.eps)])
    }
    /// e4m3 codes + ue8m0 scales of x [m][kd].
    fn quant(&self, x: u64, m: usize, kd: usize) -> Result<(u64, u64)> {
        let q = self.arena.alloc((m * kd) as u64)?;
        let s = self.arena.alloc((m * kd / 32) as u64)?;
        let groups = (m * kd / 32) as u64;
        self.launch("dsv_act_quant_fp8", [cdiv(groups, 8), 1, 1], 256, 0, &[A::P(q), A::P(s), A::P(0), A::P(x), A::I(m as i32), A::I(kd as i32), A::L(kd as i64)])?;
        Ok((q, s))
    }
    fn fq8(&self, x: u64, m: usize, kd: usize) -> Result<()> {
        let groups = (m * kd / 32) as u64;
        self.launch("dsv_act_quant_fp8", [cdiv(groups, 8), 1, 1], 256, 0, &[A::P(0), A::P(0), A::P(x), A::P(x), A::I(m as i32), A::I(kd as i32), A::L(kd as i64)])
    }
    fn fq4(&self, x: u64, n: usize, gs: i32, e4m3: i32) -> Result<()> {
        self.launch("dsv_fp4_fakequant", [cdiv(n as u64, 256), 1, 1], 256, 0, &[A::P(x), A::L(n as i64), A::I(gs), A::I(e4m3)])
    }
    fn w8a8(&self, c: u64, qs: (u64, u64), w: (u64, u64), m: usize, n: usize, kd: usize, f32out: bool) -> Result<()> {
        self.launch(
            "dsv_gemm_w8a8",
            [cdiv(n as u64, 128), cdiv(m as u64, 64), 1],
            128,
            0,
            &[A::P(c), A::P(qs.0), A::P(qs.1), A::P(w.0), A::P(w.1), A::I(m as i32), A::I(n as i32), A::I(kd as i32), A::L(n as i64), A::I(f32out as i32)],
        )
    }
    fn bf16w(&self, c: u64, a: u64, w: u64, ws: u64, m: usize, n: usize, kd: usize) -> Result<()> {
        self.launch(
            "dsv_gemm_bf16w",
            [cdiv(n as u64, 128), cdiv(m as u64, 64), 1],
            128,
            0,
            &[A::P(c), A::P(a), A::P(w), A::P(ws), A::I(m as i32), A::I(n as i32), A::I(kd as i32), A::L(kd as i64), A::L(n as i64), A::I((ws != 0) as i32), A::I(0), A::L(0), A::L(0), A::L(0), A::L(0)],
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn f32gemm(&self, c: u64, a: u64, w: u64, m: usize, n: usize, kd: usize, a_bf16: bool, w_bf16: bool) -> Result<()> {
        self.k.gemm_f32(c, a, w, m, n, kd, kd, a_bf16, w_bf16, &self.stream)
    }
    #[allow(clippy::too_many_arguments)]
    fn rope(&self, x: u64, n_tok: usize, n_head: usize, tok_stride: usize, head_stride: usize, width: usize, pos: u64, compressed: bool, inverse: bool, pos_mul: i32) -> Result<()> {
        let (c, s) = if compressed { self.rope_c } else { self.rope_w };
        let rd = self.cfg.rope_dim;
        let total = (n_tok * n_head * rd / 2) as u64;
        self.launch(
            "dsv_rope",
            [cdiv(total, 256), 1, 1],
            256,
            0,
            &[A::P(x), A::P(pos), A::P(c), A::P(s), A::I(n_tok as i32), A::I(n_head as i32), A::L(tok_stride as i64), A::L(head_stride as i64), A::I((width - rd) as i32), A::I(rd as i32), A::I(pos_mul), A::I(0), A::I(inverse as i32)],
        )
    }

    fn hc_mixes(&self, x: u64, hc: [u64; 3], t: usize) -> Result<(u64, u64, u64)> {
        let h = self.cfg.hidden;
        let hcm = self.cfg.hc_mult;
        let rsq = self.arena.alloc((t * 4) as u64)?;
        self.launch("dsv_row_rsqrt", [t as u32, 1, 1], 256, 0, &[A::P(rsq), A::P(x), A::I((hcm * h) as i32), A::F(self.cfg.eps)])?;
        let nmix = (2 + hcm) * hcm;
        let mixes = self.arena.alloc((t * nmix * 4) as u64)?;
        self.f32gemm(mixes, x, hc[0], t, nmix, hcm * h, true, false)?;
        let pre = self.arena.alloc((t * hcm * 4) as u64)?;
        let post = self.arena.alloc((t * hcm * 4) as u64)?;
        let comb = self.arena.alloc((t * hcm * hcm * 4) as u64)?;
        self.launch(
            "dsv_hc_sinkhorn",
            [cdiv(t as u64, 128), 1, 1],
            128,
            0,
            &[A::P(pre), A::P(post), A::P(comb), A::P(mixes), A::P(rsq), A::P(hc[1]), A::P(hc[2]), A::I(t as i32), A::I(self.cfg.sinkhorn_iters as i32), A::F(self.cfg.hc_eps)],
        )?;
        Ok((pre, post, comb))
    }
    fn hc_pre(&self, y: u64, x: u64, pre: u64, t: usize) -> Result<()> {
        let h = self.cfg.hidden;
        self.launch("dsv_hc_pre", [cdiv((t * h) as u64, 256), 1, 1], 256, 0, &[A::P(y), A::P(x), A::P(pre), A::I(t as i32), A::I(h as i32)])
    }
    fn hc_post(&self, out: u64, y: u64, res: u64, post: u64, comb: u64, t: usize) -> Result<()> {
        let h = self.cfg.hidden;
        self.launch("dsv_hc_post", [cdiv((t * h) as u64, 256), 1, 1], 256, 0, &[A::P(out), A::P(y), A::P(res), A::P(post), A::P(comb), A::I(t as i32), A::I(h as i32)])
    }

    // ------------------------------------------------------------------ step metadata
    /// Device arrays every layer of a step reads.
    pub fn step_meta(&self, st: &Step) -> Result<Meta> {
        // Every per-step array (positions, lengths, emit flags, and each cache region's per-slot
        // pointer table) is packed into one host buffer and uploaded with a single copy.
        let mut buf: Vec<u8> = Vec::new();
        let mut push = |bytes: &[u8]| -> u64 {
            let at = buf.len().next_multiple_of(16);
            buf.resize(at, 0);
            buf.extend_from_slice(bytes);
            at as u64
        };
        let i32s = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
        let t = st.t;
        let row_pos: Vec<i32> = if st.decode { st.pos.iter().map(|&p| p as i32).collect() } else { (0..t as i32).collect() };
        let o_row_pos = push(&i32s(&row_pos));
        let o_b_pos = push(&i32s(&st.pos.iter().map(|&p| p as i32).collect::<Vec<_>>()));
        let mut o_clen = HashMap::new();
        let mut o_gpos = HashMap::new();
        let mut o_emit = HashMap::new();
        let mut o_cmp_row = HashMap::new();
        for r in [1usize, 2] {
            let cl: Vec<i32> = if st.decode { st.pos.iter().map(|&p| ((p + 1) / r) as i32).collect() } else { (0..t).map(|i| ((i + 1) / r) as i32).collect() };
            o_clen.insert(r, push(&i32s(&cl)));
            // group index a latent row lands in (rope position = gidx * r)
            let gp: Vec<i32> = if st.decode { st.pos.iter().map(|&p| (p / r) as i32).collect() } else { (0..(t / r).max(1) as i32).collect() };
            o_gpos.insert(r, push(&i32s(&gp)));
            let em: Vec<i32> = if st.decode { st.pos.iter().map(|&p| ((p + 1) % r == 0) as i32).collect() } else { vec![0] };
            o_emit.insert(r, push(&i32s(&em)));
            o_cmp_row.insert(r, push(&i32s(&st.pos.iter().map(|&p| (p / r) as i32).collect::<Vec<_>>())));
        }
        let o_ones = push(&i32s(&vec![1i32; st.nb()]));
        let o_arange = push(&i32s(&(0..st.nb() as i32).collect::<Vec<_>>()));
        let o_win_slot = push(&i32s(&st.pos.iter().map(|&p| (p % self.cfg.window) as i32).collect::<Vec<_>>()));
        let mut o_ptrs = HashMap::new();
        let c = &self.caches;
        for reg in c.win.values().chain(c.cmp.values()).chain(c.idxk.values()).chain(c.st_kv.values()).chain(c.st_sc.values()) {
            let v: Vec<u8> = st.slots.iter().flat_map(|&sl| reg.at(sl).to_ne_bytes()).collect();
            o_ptrs.insert(reg.base, push(&v));
        }
        let base = self.put(&buf)?;
        let at = |o: u64| base + o;
        Ok(Meta {
            row_pos: at(o_row_pos),
            b_pos: at(o_b_pos),
            clen: o_clen.into_iter().map(|(k, v)| (k, at(v))).collect(),
            gpos: o_gpos.into_iter().map(|(k, v)| (k, at(v))).collect(),
            emit: o_emit.into_iter().map(|(k, v)| (k, at(v))).collect(),
            ones: at(o_ones),
            arange: at(o_arange),
            win_slot: at(o_win_slot),
            cmp_row: o_cmp_row.into_iter().map(|(k, v)| (k, at(v))).collect(),
            ptrs: o_ptrs.into_iter().map(|(k, v)| (k, at(v))).collect(),
        })
    }

    /// The per-slot pointer table of a cache region, uploaded with the step's metadata.
    fn ptrs(&self, reg: &Region, m: &Meta) -> Result<u64> {
        m.ptrs.get(&reg.base).copied().ok_or_else(|| RuntimeError::Device("dsv41: region has no pointer table".into()))
    }

    // ------------------------------------------------------------------ attention
    #[allow(clippy::too_many_arguments)]
    fn attention(&self, ly: &Layer, hn: u64, out: u64, st: &Step, m: &Meta, sh: &mut Shared) -> Result<()> {
        let c = &self.cfg;
        let t = st.t;
        let (h, nh, hd, rd, qr_n) = (c.hidden, c.n_heads, c.head_dim, c.rope_dim, c.q_lora);
        let win = c.window;
        let comp = ly.ratio > 0;
        let hq = self.quant(hn, t, h)?;
        let qa = self.arena.alloc((t * qr_n * 2) as u64)?;
        self.w8a8(qa, hq, (ly.wq_a.w, ly.wq_a.s), t, qr_n, h, false)?;
        let qr = self.arena.alloc((t * qr_n * 2) as u64)?;
        self.rmsnorm(qr, qa, ly.q_norm, t, qr_n)?;
        let qrq = self.quant(qr, t, qr_n)?;
        let q = self.arena.alloc((t * nh * hd * 2) as u64)?;
        self.w8a8(q, qrq, (ly.wq_b.w, ly.wq_b.s), t, nh * hd, qr_n, false)?;
        self.rope(q, t, nh, nh * hd, hd, hd, m.row_pos, comp, false, 1)?;
        let kv0 = self.arena.alloc((t * hd * 2) as u64)?;
        self.w8a8(kv0, hq, (ly.wkv.w, ly.wkv.s), t, hd, h, false)?;
        let kv = self.arena.alloc((t * hd * 2) as u64)?;
        self.rmsnorm(kv, kv0, ly.kv_norm, t, hd)?;
        self.rope(kv, t, 1, hd, hd, hd, m.row_pos, comp, false, 1)?;
        self.fq8(kv, t, hd)?;
        let wreg = self.caches.win[&ly.id];
        // window ring: position p at slot p % WIN
        let (win_ptrs, off) = if !st.decode {
            let n = t.min(win);
            let slot = st.slots[0];
            let first = t - n;
            let mut p = first;
            while p < t {
                let s0 = p % win;
                let run = (win - s0).min(t - p);
                self.dev.memcpy_dtod_async(
                    wreg.at(slot) + (s0 * hd * 2) as u64,
                    kv + (p * hd * 2) as u64,
                    (run * hd * 2) as u64,
                    &self.stream,
                )?;
                p += run;
            }
            (self.put(&[kv])?, t)
        } else {
            let wp = self.ptrs(&wreg, m)?;
            self.launch(
                "dsv_scatter_rows",
                [1, st.nb() as u32, 1],
                64,
                0,
                &[A::P(kv), A::I(st.nb() as i32), A::P(m.ones), A::P(m.arange), A::P(wp), A::P(m.win_slot), A::I((hd * 2) as i32)],
            )?;
            (wp, win)
        };
        let mut n_cmp = 0usize;
        let mut cmp_idx = 0u64;
        let mut cmp_ptrs = 0u64;
        if comp {
            let r = ly.ratio;
            let src = c.kv_src(ly.id).ok_or_else(|| RuntimeError::Device(format!("dsv41: layer {} has no kv source", ly.id)))?;
            let creg = self.caches.cmp[&src];
            let kreg = self.caches.idxk[&src];
            // latent rows this call produces (prefill: G groups; decode: one candidate row per slot)
            let mut latent = 0u64;
            let g = if st.decode { st.nb() } else { t / r };
            // A prefill shorter than one group produces no latent but still owes the compressor its
            // partial group (state rows 0..t%r), which the first decode steps pool.
            let tail = !st.decode && r > 1 && t % r != 0;
            if ly.kv_source && (g > 0 || tail) {
                let pooled = self.arena.alloc((g.max(1) * hd * 2) as u64)?;
                if r > 1 {
                    let kvf = self.arena.alloc((t * hd * 4) as u64)?;
                    let sc = self.arena.alloc((t * hd * 4) as u64)?;
                    self.f32gemm(kvf, hn, ly.c_wkv, t, hd, h, true, true)?;
                    self.f32gemm(sc, hn, ly.c_wgate, t, hd, h, true, true)?;
                    let sk = self.caches.st_kv[&ly.id];
                    let ss = self.caches.st_sc[&ly.id];
                    if !st.decode {
                        let rem = t % r;
                        let n_thr = (g * hd + rem * hd) as u64;
                        let slot = st.slots[0];
                        self.launch(
                            "dsv_compress_pool_prefill",
                            [cdiv(n_thr, 256), 1, 1],
                            256,
                            0,
                            &[A::P(pooled), A::P(kvf), A::P(sc), A::I(t as i32), A::I(hd as i32), A::I(r as i32), A::P(sk.at(slot)), A::P(ss.at(slot))],
                        )?;
                    } else {
                        let skp = self.ptrs(&sk, m)?;
                        let ssp = self.ptrs(&ss, m)?;
                        self.launch(
                            "dsv_compress_pool_decode",
                            [st.nb() as u32, 1, 1],
                            256,
                            0,
                            &[A::P(pooled), A::P(kvf), A::P(sc), A::I(st.nb() as i32), A::I(hd as i32), A::I(r as i32), A::P(m.b_pos), A::P(skp), A::P(ssp)],
                        )?;
                    }
                } else {
                    let raw = self.arena.alloc((t * hd * 2) as u64)?;
                    self.bf16w(raw, hn, ly.c_wkv, 0, t, hd, h)?;
                    self.dev.memcpy_dtod_async(pooled, raw, (t * hd * 2) as u64, &self.stream)?;
                }
                if g > 0 {
                    latent = self.arena.alloc((g * hd * 2) as u64)?;
                    self.rmsnorm(latent, pooled, ly.c_norm, g, hd)?;
                }
            }
            // clen: compressed positions each row may see
            let clen = m.clen[&r];
            let s_max = if st.decode { st.pos.iter().map(|&p| (p + 1) / r).max().unwrap_or(0) } else { t / r };
            if ly.index_source {
                let (ixh, ixd) = (c.index_heads, c.index_dim);
                if latent != 0 && ly.kv_source {
                    let k0 = self.arena.alloc((g * ixd * 2) as u64)?;
                    self.bf16w(k0, latent, ly.i_wk, 0, g, ixd, hd)?;
                    let k = self.arena.alloc((g * ixd * 2) as u64)?;
                    self.rmsnorm(k, k0, ly.i_knorm, g, ixd)?;
                    self.rope(k, g, 1, ixd, ixd, ixd, m.gpos[&r], true, false, r as i32)?;
                    self.fq4(k, g * ixd, 32, 0)?;
                    self.write_rows(&kreg, k, g, ixd * 2, st, m, r)?;
                }
                if s_max > 0 {
                    let qi = self.arena.alloc((t * ixh * ixd * 2) as u64)?;
                    self.w8a8(qi, qrq, (ly.i_wq_b.w, ly.i_wq_b.s), t, ixh * ixd, qr_n, false)?;
                    self.rope(qi, t, ixh, ixh * ixd, ixd, ixd, m.row_pos, true, false, 1)?;
                    self.fq4(qi, t * ixh * ixd, 32, 0)?;
                    let w = self.arena.alloc((t * ixh * 2) as u64)?;
                    self.bf16w(w, hn, ly.i_wproj, 0, t, ixh, h)?;
                    let n = (t * ixh) as u64;
                    let scale = (ixd as f32).powf(-0.5) * (ixh as f32).powf(-0.5);
                    self.launch("dsv_scale_bf16", [cdiv(n, 256), 1, 1], 256, 0, &[A::P(w), A::L(n as i64), A::F(scale)])?;
                    let s_ld = s_max.next_multiple_of(64);
                    let score = self.arena.alloc((t * s_ld * 2) as u64)?;
                    let kp = self.ptrs(&kreg, m)?;
                    self.launch(
                        "dsv_index_score",
                        [(s_ld / 64) as u32, cdiv(t as u64, 4), 1],
                        256,
                        IX_SMEM,
                        &[A::P(score), A::L(s_ld as i64), A::P(qi), A::P(w), A::P(kp), A::I(st.q_per_b() as i32), A::P(clen), A::I(t as i32)],
                    )?;
                    let cb = c.cand_block;
                    let mut keep = (0u64, 0usize);
                    if ly.cand_source {
                        let nb_ld = s_max.div_ceil(cb);
                        let bs = self.arena.alloc((t * nb_ld * 2) as u64)?;
                        self.launch("dsv_cand_block_scores", [cdiv(nb_ld as u64, 256), t as u32, 1], 256, 0, &[A::P(bs), A::L(nb_ld as i64), A::P(score), A::L(s_ld as i64), A::P(clen), A::I(cb as i32)])?;
                        let nbl: Vec<i32> = if st.decode {
                            st.pos.iter().map(|&p| ((p + 1) / r).div_ceil(cb) as i32).collect()
                        } else {
                            (0..t).map(|i| ((i + 1) / r).div_ceil(cb) as i32).collect()
                        };
                        let nbl = self.put(&nbl)?;
                        let kb = c.cand_topk_blocks.min(nb_ld);
                        let bidx = self.arena.alloc((t * kb * 4) as u64)?;
                        self.launch("dsv_topk_select", [t as u32, 1, 1], 1024, 0, &[A::P(bidx), A::I(kb as i32), A::P(bs), A::L(nb_ld as i64), A::P(nbl), A::I(kb as i32), A::I(0), A::P(0), A::L(0), A::I(1)])?;
                        let kp = self.keep_buf;
                        self.launch("dsv_keep_from_idx", [t as u32, 1, 1], 256, 0, &[A::P(kp), A::L(nb_ld as i64), A::I(nb_ld as i32), A::P(bidx), A::I(kb as i32)])?;
                        sh.keep = kp;
                        sh.nblk = nb_ld;
                        keep = (0, 0);
                    } else if ly.uses_cand && sh.nblk > 0 {
                        keep = (sh.keep, sh.nblk);
                    }
                    let kout = c.index_topk.min(s_max);
                    let idx = self.topk_buf;
                    self.launch(
                        "dsv_topk_select",
                        [t as u32, 1, 1],
                        1024,
                        0,
                        &[A::P(idx), A::I(kout as i32), A::P(score), A::L(s_ld as i64), A::P(clen), A::I(c.index_topk as i32), A::I(off as i32), A::P(keep.0), A::L(keep.1 as i64), A::I(cb as i32)],
                    )?;
                    sh.topk = idx;
                    sh.kout = kout;
                } else {
                    sh.topk = 0;
                    sh.kout = 0;
                }
            }
            if latent != 0 {
                self.rope(latent, g, 1, hd, hd, hd, m.gpos[&r], true, false, r as i32)?;
                self.fq4(latent, g * hd, 16, 1)?;
                self.write_rows(&creg, latent, g, hd * 2, st, m, r)?;
            }
            if sh.kout > 0 {
                n_cmp = sh.kout;
                cmp_idx = sh.topk;
            }
            cmp_ptrs = self.ptrs(&creg, m)?;
        }
        let n_idx = win + n_cmp;
        let table = self.arena.alloc((t * n_idx * 4) as u64)?;
        self.launch(
            "dsv_attn_index",
            [t as u32, 1, 1],
            256,
            0,
            &[A::P(table), A::I(t as i32), A::I(st.q_per_b() as i32), A::I(win as i32), A::I(st.decode as i32), A::P(m.b_pos), A::P(cmp_idx), A::I(n_cmp as i32)],
        )?;
        let o = self.arena.alloc((t * nh * hd * 2) as u64)?;
        self.launch(
            "dsv_sparse_attn",
            [t as u32, 1, 1],
            512,
            SA_SMEM,
            &[A::P(o), A::P(q), A::P(table), A::I(n_idx as i32), A::P(win_ptrs), A::P(cmp_ptrs), A::I(off as i32), A::I(st.q_per_b() as i32), A::P(ly.sink), A::F((hd as f32).powf(-0.5))],
        )?;
        self.rope(o, t, nh, nh * hd, hd, hd, m.row_pos, comp, true, 1)?;
        // wo_a: 8 groups of (4096 -> 1024), bf16 x fp8-dequantized weights
        let (og, or) = (c.o_groups, c.o_lora);
        let kg = nh * hd / og;
        let ga = self.arena.alloc((t * og * or * 2) as u64)?;
        self.launch(
            "dsv_gemm_bf16w",
            [cdiv(or as u64, 128), cdiv(t as u64, 64), og as u32],
            128,
            0,
            &[A::P(ga), A::P(o), A::P(ly.wo_a.w), A::P(ly.wo_a.s), A::I(t as i32), A::I(or as i32), A::I(kg as i32), A::L((nh * hd) as i64), A::L((og * or) as i64), A::I(1), A::I(0), A::L(kg as i64), A::L((or * kg) as i64), A::L(((or / 32) * (kg / 32)) as i64), A::L(or as i64)],
        )?;
        let gq = self.quant(ga, t, og * or)?;
        self.w8a8(out, gq, (ly.wo_b.w, ly.wo_b.s), t, h, og * or, false)?;
        let _ = rd;
        Ok(())
    }

    /// Write this call's latent/key rows into a per-slot compressed cache region.
    /// Prefill: rows 0..g of the one slot. Decode: row pos/r of each slot whose group just completed
    /// (ratio 1: every slot).
    #[allow(clippy::too_many_arguments)]
    fn write_rows(&self, reg: &Region, src: u64, g: usize, row_bytes: usize, st: &Step, m: &Meta, r: usize) -> Result<()> {
        if !st.decode {
            let slot = st.slots[0];
            self.dev.memcpy_dtod_async(reg.at(slot), src, (g * row_bytes) as u64, &self.stream)
        } else {
            let dp = self.ptrs(reg, m)?;
            self.launch(
                "dsv_scatter_rows",
                [1, st.nb() as u32, 1],
                64,
                0,
                &[A::P(src), A::I(st.nb() as i32), A::P(m.emit[&r]), A::P(m.arange), A::P(dp), A::P(m.cmp_row[&r]), A::I(row_bytes as i32)],
            )
        }
    }

    // ------------------------------------------------------------------ MoE
    fn moe(&self, ly: &Layer, hn: u64, y: u64, t: usize) -> Result<()> {
        let c = &self.cfg;
        let (h, e, tk, mi) = (c.hidden, c.n_experts, c.topk, c.moe_inter);
        let logits = self.arena.alloc((t * e * 4) as u64)?;
        self.f32gemm(logits, hn, ly.gate_w, t, e, h, true, true)?;
        let idx = self.arena.alloc((t * tk * 4) as u64)?;
        let wt = self.arena.alloc((t * tk * 4) as u64)?;
        self.launch("dsv_moe_route", [cdiv(t as u64, 8), 1, 1], 256, 0, &[A::P(idx), A::P(wt), A::P(logits), A::P(ly.gate_b), A::I(t as i32), A::I(e as i32), A::I(tk as i32), A::I(c.norm_topk as i32), A::F(c.route_scale)])?;
        let n = t * tk;
        let counts = self.arena.alloc((e * 4) as u64)?;
        self.dev.memset_d8_async(counts, 0, e * 4, &self.stream)?;
        self.launch("dsv_moe_count", [cdiv(n as u64, 256), 1, 1], 256, 0, &[A::P(counts), A::P(idx), A::I(n as i32)])?;
        let bm = 64usize;
        // sum over experts of ceil(c_e / bm) <= n / bm + (experts with any row) <= ceil(n / bm) + min(n, e)
        let max_tiles = n.div_ceil(bm) + n.min(e);
        let offs = self.arena.alloc(((e + 1) * 4) as u64)?;
        let tiles = self.arena.alloc((max_tiles * 8) as u64)?;
        let meta = self.arena.alloc(4)?;
        let ctr = self.arena.alloc((e * 4) as u64)?;
        self.launch("dsv_moe_offsets", [1, 1, 1], e.next_power_of_two().max(32) as u32, 0, &[A::P(offs), A::P(tiles), A::P(meta), A::P(ctr), A::P(counts), A::I(e as i32), A::I(bm as i32)])?;
        let rows = self.arena.alloc((n * 4) as u64)?;
        let rowpos = self.arena.alloc((n * 4) as u64)?;
        let roww = self.arena.alloc((n * 4) as u64)?;
        self.launch("dsv_moe_fill", [cdiv(n as u64, 256), 1, 1], 256, 0, &[A::P(rows), A::P(rowpos), A::P(roww), A::P(ctr), A::P(offs), A::P(idx), A::P(wt), A::I(n as i32), A::I(tk as i32)])?;
        let xq = self.quant(hn, t, h)?;
        let gu = self.arena.alloc((n * 2 * mi * 2) as u64)?;
        self.launch(
            "dsv_moe_gemm_fp4",
            [cdiv((2 * mi) as u64, 128), max_tiles as u32, 1],
            128,
            0,
            &[A::P(gu), A::P(xq.0), A::P(xq.1), A::P(ly.w13), A::P(ly.w13_s), A::P(tiles), A::P(meta), A::P(offs), A::P(rows), A::I(0), A::I((2 * mi) as i32), A::I(h as i32), A::L((2 * mi * h / 2) as i64), A::L((2 * mi * h / 32) as i64)],
        )?;
        let hq = self.arena.alloc((n * mi) as u64)?;
        let hs = self.arena.alloc((n * mi / 32) as u64)?;
        self.launch(
            "dsv_swiglu_quant",
            [cdiv((n * mi / 32) as u64, 8), 1, 1],
            256,
            0,
            &[A::P(hq), A::P(hs), A::P(gu), A::P(gu + (mi * 2) as u64), A::L((2 * mi) as i64), A::P(roww), A::I(n as i32), A::I(mi as i32), A::F(c.swiglu_limit), A::P(0)],
        )?;
        let down = self.arena.alloc((n * h * 2) as u64)?;
        self.launch(
            "dsv_moe_gemm_fp4",
            [cdiv(h as u64, 128), max_tiles as u32, 1],
            128,
            0,
            &[A::P(down), A::P(hq), A::P(hs), A::P(ly.w2), A::P(ly.w2_s), A::P(tiles), A::P(meta), A::P(offs), A::P(rows), A::I(1), A::I(h as i32), A::I(mi as i32), A::L((h * mi / 2) as i64), A::L((h * mi / 32) as i64)],
        )?;
        // shared expert (fp8 weights, no routing weight)
        let sgu = self.arena.alloc((t * 2 * mi * 2) as u64)?;
        self.w8a8(sgu, xq, (ly.sh_w13.w, ly.sh_w13.s), t, 2 * mi, h, false)?;
        let shq = self.arena.alloc((t * mi) as u64)?;
        let shs = self.arena.alloc((t * mi / 32) as u64)?;
        self.launch(
            "dsv_swiglu_quant",
            [cdiv((t * mi / 32) as u64, 8), 1, 1],
            256,
            0,
            &[A::P(shq), A::P(shs), A::P(sgu), A::P(sgu + (mi * 2) as u64), A::L((2 * mi) as i64), A::P(0), A::I(t as i32), A::I(mi as i32), A::F(c.swiglu_limit), A::P(0)],
        )?;
        let shared = self.arena.alloc((t * h * 2) as u64)?;
        self.w8a8(shared, (shq, shs), (ly.sh_w2.w, ly.sh_w2.s), t, h, mi, false)?;
        self.launch("dsv_moe_combine", [t as u32, 1, 1], 256, 0, &[A::P(y), A::P(down), A::P(shared), A::P(idx), A::P(rowpos), A::I(t as i32), A::I(h as i32), A::I(tk as i32)])
    }

    // ------------------------------------------------------------------ layer
    /// One layer: x [t][4][H] -> x_out, pre_mix [t][4] -> pre_out. `emb` is the Engram gather
    /// [t][n_cols*head_dim] bf16 for Engram layers.
    #[allow(clippy::too_many_arguments)]
    pub fn layer(&self, ly: &Layer, x: u64, pre_mix: u64, x_out: u64, pre_out: u64, emb: u64, st: &Step, m: &Meta, sh: &mut Shared) -> Result<()> {
        let c = &self.cfg;
        let t = st.t;
        let h = c.hidden;
        let mark = self.arena.mark();
        if ly.engram {
            let ncols = (c.engram_max_ngram - 1) * c.engram_n_heads * c.engram_head_dim;
            let eq = self.quant(emb, t, ncols)?;
            let kv = self.arena.alloc((t * (c.hc_mult + 1) * h * 2) as u64)?;
            self.w8a8(kv, eq, (ly.e_wkv.w, ly.e_wkv.s), t, (c.hc_mult + 1) * h, ncols, false)?;
            self.launch("dsv_engram_gate", [t as u32, c.hc_mult as u32, 1], 256, 0, &[A::P(x), A::P(kv), A::P(ly.e_qw), A::P(ly.e_kw), A::P(0), A::I(h as i32), A::F(c.eps)])?;
        }
        let (a_pre, a_post, a_comb) = self.hc_mixes(x, ly.hc_attn, t)?;
        let hpre = self.arena.alloc((t * h * 2) as u64)?;
        self.hc_pre(hpre, x, pre_mix, t)?;
        let hn = self.arena.alloc((t * h * 2) as u64)?;
        self.rmsnorm(hn, hpre, ly.attn_norm, t, h)?;
        let ao = self.arena.alloc((t * h * 2) as u64)?;
        self.attention(ly, hn, ao, st, m, sh)?;
        let x2 = self.arena.alloc((t * c.hc_mult * h * 2) as u64)?;
        self.hc_post(x2, ao, x, a_post, a_comb, t)?;
        let (f_pre, f_post, f_comb) = self.hc_mixes(x2, ly.hc_ffn, t)?;
        let hpre2 = self.arena.alloc((t * h * 2) as u64)?;
        self.hc_pre(hpre2, x2, a_pre, t)?;
        let hn2 = self.arena.alloc((t * h * 2) as u64)?;
        self.rmsnorm(hn2, hpre2, ly.ffn_norm, t, h)?;
        let y = self.arena.alloc((t * h * 2) as u64)?;
        self.moe(ly, hn2, y, t)?;
        self.hc_post(x_out, y, x2, f_post, f_comb, t)?;
        self.dev.memcpy_dtod_async(pre_out, f_pre, (t * c.hc_mult * 4) as u64, &self.stream)?;
        self.arena.reset(mark);
        Ok(())
    }

    pub fn stream_sync(&self) -> Result<()> {
        self.dev.stream_synchronize(&self.stream)
    }

    pub fn alloc_persistent(&self, bytes: u64) -> Result<u64> {
        self.arena.alloc(bytes)
    }
}

pub struct Meta {
    pub row_pos: u64,
    pub b_pos: u64,
    pub clen: HashMap<usize, u64>,
    pub gpos: HashMap<usize, u64>,
    pub emit: HashMap<usize, u64>,
    pub ones: u64,
    pub arange: u64,
    pub win_slot: u64,
    pub cmp_row: HashMap<usize, u64>,
    /// Region base -> device pointer table [nb] u64.
    pub ptrs: HashMap<u64, u64>,
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 slice viewed as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
