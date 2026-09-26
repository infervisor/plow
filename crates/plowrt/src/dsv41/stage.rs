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
use super::kernels::{cdiv, moe_gemm_name, moe_smem, Cost, Kernels, Peak, A, IX_SMEM, SA_SMEM, W8_SMEM, WG_SMEM};
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
    /// Prefill GEMMs may take the wgmma kernels (dsv41_wg.cu); `PLOW_DSV41_NO_WG` pins them to the
    /// mma.sync kernels (a kill switch, and the A/B for end-to-end checks).
    wg: bool,
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
        if cfg.hc_mult != 4 {
            return Err(RuntimeError::Device(format!("dsv41: hc_mult {} unsupported (the mHC kernels are built for 4)", cfg.hc_mult)));
        }
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
            wg: std::env::var_os("PLOW_DSV41_NO_WG").is_none(),
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
        self.k.cost(Cost::mem((m * d * 4) as f64));
        self.launch("dsv_rmsnorm", [m as u32, 1, 1], 256, 0, &[A::P(y), A::P(x), A::P(w), A::I(d as i32), A::L(d as i64), A::L(d as i64), A::F(self.cfg.eps)])
    }
    /// e4m3 codes + ue8m0 scales of x [m][kd], and at prefill sizes (m >= WG_MIN_M) the same values
    /// dequantized to bf16 (0 otherwise) for the wgmma GEMMs.
    fn quant(&self, x: u64, m: usize, kd: usize) -> Result<Q8> {
        let q = self.arena.alloc((m * kd) as u64)?;
        let s = self.arena.alloc((m * kd / 32) as u64)?;
        let fq = if self.wg && m >= WG_MIN_M { self.arena.alloc((m * kd * 2) as u64)? } else { 0 };
        let groups = (m * kd / 32) as u64;
        self.k.cost(Cost::mem((m * kd * 3) as f64 + (m * kd / 32) as f64 + if fq != 0 { (m * kd * 2) as f64 } else { 0.0 }));
        self.launch("dsv_act_quant_fp8", [cdiv(groups, 8), 1, 1], 256, 0, &[A::P(q), A::P(s), A::P(fq), A::P(x), A::I(m as i32), A::I(kd as i32), A::L(kd as i64)])?;
        Ok((q, s, fq))
    }
    /// C [z][m][ldc] = A [z][m][lda] (bf16) . deq(W8 [z][n][kd], [32 x 32] ue8m0)^T on the wgmma kernel.
    #[allow(clippy::too_many_arguments)]
    fn wg_fp8(&self, c: u64, a: u64, w: (u64, u64), m: usize, n: usize, kd: usize, lda: usize, ldc: usize, f32out: bool, batch: usize, strides: [usize; 4]) -> Result<()> {
        self.launch(
            "dsv_gemm_wg_fp8",
            [cdiv(n as u64, WG_BN as u64), cdiv(m as u64, 128), batch as u32],
            384,
            WG_SMEM,
            &[A::P(c), A::P(a), A::P(w.0), A::P(w.1), A::I(m as i32), A::I(n as i32), A::I(kd as i32), A::L(lda as i64), A::L(ldc as i64), A::I(f32out as i32), A::L(strides[0] as i64), A::L(strides[1] as i64), A::L(strides[2] as i64), A::L(strides[3] as i64)],
        )
    }
    fn fq8(&self, x: u64, m: usize, kd: usize) -> Result<()> {
        let groups = (m * kd / 32) as u64;
        self.k.cost(Cost::mem((m * kd * 4) as f64));
        self.launch("dsv_act_quant_fp8", [cdiv(groups, 8), 1, 1], 256, 0, &[A::P(0), A::P(0), A::P(x), A::P(x), A::I(m as i32), A::I(kd as i32), A::L(kd as i64)])
    }
    fn fq4(&self, x: u64, n: usize, gs: i32, e4m3: i32) -> Result<()> {
        self.k.cost(Cost::mem((n * 4) as f64));
        self.launch("dsv_fp4_fakequant", [cdiv(n as u64, 256), 1, 1], 256, 0, &[A::P(x), A::L(n as i64), A::I(gs), A::I(e4m3)])
    }
    fn w8a8(&self, c: u64, qs: Q8, w: (u64, u64), m: usize, n: usize, kd: usize, f32out: bool) -> Result<()> {
        let (mf, nf, kf) = (m as f64, n as f64, kd as f64);
        self.k.cost(Cost { flops: 2.0 * mf * nf * kf, bytes: mf * kf * 1.03 + nf * kf * 1.001 + mf * nf * if f32out { 4.0 } else { 2.0 }, peak: Peak::Fp8Mma });
        if qs.2 != 0 && wg_fits(m, n, kd, 1) {
            return self.wg_fp8(c, qs.2, w, m, n, kd, kd, n, f32out, 1, [0; 4]);
        }
        let tiles = cdiv(n as u64, 128) * cdiv(m as u64, 64);
        let ks = ksplit(tiles, kd / 32);
        let part = if ks > 1 { self.arena.alloc((ks * m * n * 4) as u64)? } else { 0 };
        self.launch(
            "dsv_gemm_w8a8",
            [cdiv(n as u64, 128), cdiv(m as u64, 64), ks as u32],
            128,
            W8_SMEM,
            &[A::P(c), A::P(qs.0), A::P(qs.1), A::P(w.0), A::P(w.1), A::I(m as i32), A::I(n as i32), A::I(kd as i32), A::L(n as i64), A::I(f32out as i32), A::I(ks as i32), A::P(part)],
        )?;
        self.splitk_reduce(c, part, ks, 1, m, n, n, 0, f32out)
    }
    /// Sum the split-K partials into C, in split order.
    #[allow(clippy::too_many_arguments)]
    fn splitk_reduce(&self, c: u64, part: u64, ks: usize, batch: usize, m: usize, n: usize, ldc: usize, c_bstride: usize, f32out: bool) -> Result<()> {
        if ks <= 1 {
            return Ok(());
        }
        let total = (batch * m * n) as u64;
        self.k.cost(Cost::mem((total * 4 * ks as u64 + total * if f32out { 4 } else { 2 }) as f64));
        self.launch(
            "dsv_splitk_reduce",
            [cdiv(total, 256).min(4096), 1, 1],
            256,
            0,
            &[A::P(c), A::P(part), A::I(ks as i32), A::I(batch as i32), A::I(m as i32), A::I(n as i32), A::L(ldc as i64), A::L(c_bstride as i64), A::I(f32out as i32)],
        )
    }
    fn bf16w(&self, c: u64, a: u64, w: u64, ws: u64, m: usize, n: usize, kd: usize) -> Result<()> {
        let (mf, nf, kf) = (m as f64, n as f64, kd as f64);
        self.k.cost(Cost { flops: 2.0 * mf * nf * kf, bytes: mf * kf * 2.0 + nf * kf * if ws != 0 { 1.001 } else { 2.0 } + mf * nf * 2.0, peak: Peak::Bf16Mma });
        if self.wg && ws != 0 && wg_fits(m, n, kd, 1) {
            return self.wg_fp8(c, a, (w, ws), m, n, kd, kd, n, false, 1, [0; 4]);
        }
        let ks = ksplit(cdiv(n as u64, 128) * cdiv(m as u64, 64), kd / 32);
        let part = if ks > 1 { self.arena.alloc((ks * m * n * 4) as u64)? } else { 0 };
        self.launch(
            "dsv_gemm_bf16w",
            [cdiv(n as u64, 128), cdiv(m as u64, 64), ks as u32],
            128,
            0,
            &[A::P(c), A::P(a), A::P(w), A::P(ws), A::I(m as i32), A::I(n as i32), A::I(kd as i32), A::L(kd as i64), A::L(n as i64), A::I((ws != 0) as i32), A::I(0), A::L(0), A::L(0), A::L(0), A::L(0), A::I(ks as i32), A::P(part)],
        )?;
        self.splitk_reduce(c, part, ks, 1, m, n, n, 0, false)
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
        self.k.cost(Cost::mem((n_tok * n_head * rd * 4) as f64 + (n_tok * rd * 4) as f64));
        self.launch(
            "dsv_rope",
            [cdiv(total, 256), 1, 1],
            256,
            0,
            &[A::P(x), A::P(pos), A::P(c), A::P(s), A::I(n_tok as i32), A::I(n_head as i32), A::L(tok_stride as i64), A::L(head_stride as i64), A::I((width - rd) as i32), A::I(rd as i32), A::I(pos_mul), A::I(0), A::I(inverse as i32)],
        )
    }

    /// The mHC coefficients for a sublayer: pre / post / comb from the flattened stream x [t][4H].
    /// Two launches: a split-K partial of the 24 projections plus the row's sum of squares (x read
    /// once), then the in-order reduction, the RMS scale and the Sinkhorn.
    fn hc_mixes(&self, x: u64, hc: [u64; 3], t: usize) -> Result<(u64, u64, u64)> {
        let k = self.cfg.hidden * self.cfg.hc_mult;
        let row_tiles = t.div_ceil(4);
        let splits = 264usize.div_ceil(row_tiles).clamp(1, k / 1024);
        let part = self.arena.alloc((splits * t * 25 * 4) as u64)?;
        self.k.cost(Cost { flops: 2.0 * (t * 25 * k) as f64, bytes: (t * k * 2 + 24 * k * 4) as f64, peak: Peak::Fp32 });
        self.launch("dsv_hc_mix_partial", [splits as u32, row_tiles as u32, 1], 256, 0, &[A::P(part), A::P(x), A::P(hc[0]), A::I(t as i32), A::I(k as i32)])?;
        let pre = self.arena.alloc((t * 4 * 4) as u64)?;
        let post = self.arena.alloc((t * 4 * 4) as u64)?;
        let comb = self.arena.alloc((t * 16 * 4) as u64)?;
        self.k.cost(Cost::mem((splits * t * 25 * 4 + t * 24 * 4) as f64));
        self.launch(
            "dsv_hc_mix_finish",
            [cdiv(t as u64, 8), 1, 1],
            128,
            0,
            &[A::P(pre), A::P(post), A::P(comb), A::P(part), A::I(splits as i32), A::I(t as i32), A::I(k as i32), A::P(hc[1]), A::P(hc[2]), A::I(self.cfg.sinkhorn_iters as i32), A::F(self.cfg.eps), A::F(self.cfg.hc_eps)],
        )?;
        Ok((pre, post, comb))
    }
    fn hc_pre(&self, y: u64, x: u64, pre: u64, t: usize) -> Result<()> {
        let h = self.cfg.hidden;
        self.k.cost(Cost::mem((t * h * 10) as f64));
        self.launch("dsv_hc_pre", [cdiv((t * h) as u64, 256), 1, 1], 256, 0, &[A::P(y), A::P(x), A::P(pre), A::I(t as i32), A::I(h as i32)])
    }
    fn hc_post(&self, out: u64, y: u64, res: u64, post: u64, comb: u64, t: usize) -> Result<()> {
        let h = self.cfg.hidden;
        self.k.cost(Cost::mem((t * h * 18) as f64));
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
                    // each (query, head) row against the visible keys; keys stream once per 4-query tile
                    let s_avg = if st.decode { s_max as f64 } else { s_max as f64 / 2.0 };
                    self.k.cost(Cost { flops: 2.0 * (t * ixh * ixd) as f64 * s_avg, bytes: (t * ixh * ixd * 2) as f64 + (t as f64) * s_avg * 2.0 + (st.nb() * s_max * ixd * 2) as f64, peak: Peak::Bf16Mma });
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
                    self.k.cost(Cost::mem((t * s_ld * 2 * 3) as f64 + (t * kout * 4) as f64));
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
        // q, o and the gathered KV rows; S = Q.K^T and P.V per head
        self.k.cost(Cost { flops: 4.0 * (t * nh * n_idx * hd) as f64, bytes: (t * nh * hd * 4) as f64 + (t * n_idx * hd * 2) as f64, peak: Peak::Bf16Mma });
        // split-KV when the query rows alone leave SMs idle (decode): about one wave of blocks
        let splits = if t < 66 { 132usize.div_ceil(t).min(n_idx.div_ceil(64)) } else { 1 };
        let part = if splits > 1 { self.arena.alloc((t * splits * nh * (hd + 4) * 4) as u64)? } else { 0 };
        self.launch(
            "dsv_sparse_attn",
            [t as u32, splits as u32, 1],
            512,
            SA_SMEM,
            &[A::P(o), A::P(q), A::P(table), A::I(n_idx as i32), A::P(win_ptrs), A::P(cmp_ptrs), A::I(off as i32), A::I(st.q_per_b() as i32), A::P(ly.sink), A::F((hd as f32).powf(-0.5)), A::P(part)],
        )?;
        if splits > 1 {
            self.k.cost(Cost::mem((t * splits * nh * (hd + 4) * 4 + t * nh * hd * 2) as f64));
            self.launch("dsv_sparse_attn_merge", [t as u32, nh as u32, 1], 128, 0, &[A::P(o), A::P(part), A::I(splits as i32), A::P(ly.sink)])?;
        }
        self.rope(o, t, nh, nh * hd, hd, hd, m.row_pos, comp, true, 1)?;
        // wo_a: 8 groups of (4096 -> 1024), bf16 x fp8-dequantized weights
        let (og, or) = (c.o_groups, c.o_lora);
        let kg = nh * hd / og;
        let ga = self.arena.alloc((t * og * or * 2) as u64)?;
        self.k.cost(Cost { flops: 2.0 * (t * og * or * kg) as f64, bytes: (t * nh * hd * 2) as f64 + (og * or * kg) as f64 * 1.001 + (t * og * or * 2) as f64, peak: Peak::Bf16Mma });
        if self.wg && wg_fits(t, or, kg, og) {
            // group g: ga[t][g*or ..] = o[t][g*kg ..] . deq(wo_a[g*or ..])^T
            self.wg_fp8(ga, o, (ly.wo_a.w, ly.wo_a.s), t, or, kg, nh * hd, og * or, false, og, [or * kg, (or / 32) * (kg / 32), kg, or])?;
            let gq = self.quant(ga, t, og * or)?;
            return self.w8a8(out, gq, (ly.wo_b.w, ly.wo_b.s), t, h, og * or, false);
        }
        let ks = ksplit(cdiv(or as u64, 128) * cdiv(t as u64, 64) * og as u32, kg / 32);
        let part = if ks > 1 { self.arena.alloc((ks * og * t * or * 4) as u64)? } else { 0 };
        self.launch(
            "dsv_gemm_bf16w",
            [cdiv(or as u64, 128), cdiv(t as u64, 64), (og * ks) as u32],
            128,
            0,
            &[A::P(ga), A::P(o), A::P(ly.wo_a.w), A::P(ly.wo_a.s), A::I(t as i32), A::I(or as i32), A::I(kg as i32), A::L((nh * hd) as i64), A::L((og * or) as i64), A::I(1), A::I(0), A::L(kg as i64), A::L((or * kg) as i64), A::L(((or / 32) * (kg / 32)) as i64), A::L(or as i64), A::I(ks as i32), A::P(part)],
        )?;
        // partials are [split][group][t][or]; the output row t holds the groups side by side
        self.splitk_reduce(ga, part, ks, og, t, or, og * or, or, false)?;
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
        self.k.cost(Cost::mem((t * e * 4) as f64));
        self.launch("dsv_moe_route", [cdiv(t as u64, 8), 1, 1], 256, 0, &[A::P(idx), A::P(wt), A::P(logits), A::P(ly.gate_b), A::I(t as i32), A::I(e as i32), A::I(tk as i32), A::I(c.norm_topk as i32), A::F(c.route_scale)])?;
        let n = t * tk;
        let counts = self.arena.alloc((e * 4) as u64)?;
        self.dev.memset_d8_async(counts, 0, e * 4, &self.stream)?;
        self.launch("dsv_moe_count", [cdiv(n as u64, 256), 1, 1], 256, 0, &[A::P(counts), A::P(idx), A::I(n as i32)])?;
        // distinct experts a batch of t tokens touches, in expectation (uniform routing)
        let touched = (e as f64) * (1.0 - (1.0 - tk as f64 / e as f64).powf(t as f64));
        // decode-sized steps: each routed expert holds a row or two, so the weight-streaming GEMV beats
        // an MMA tile that would be almost all padding
        let gemv = t <= MOE_GEMV_MAX_T;
        // MMA tile height from the rows an expert holds on average: the tiles stream every weight
        // column once each, so padding rows cost decode and MMA work but no bandwidth
        let bm = match moe_bm(n as f64 / touched.max(1.0)) {
            _ if gemv => 64,
            // the wgmma tiles read the bf16 activation copy, which quant() makes from WG_MIN_M rows
            MOE_WG_BM if t < WG_MIN_M || !self.wg => 64,
            b => b,
        };
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
        self.k.cost(Cost { flops: 2.0 * (n * 2 * mi * h) as f64, bytes: touched * (2 * mi) as f64 * (h as f64 / 2.0 + h as f64 / 32.0) + (n * h) as f64 + (n * 2 * mi * 2) as f64, peak: Peak::Fp8Mma });
        let wg = bm == MOE_WG_BM && xq.2 != 0;
        if wg {
            self.launch(
                "dsv_moe_gemm_wg_fp4",
                [cdiv((2 * mi) as u64, WG_BN as u64), max_tiles as u32, 1],
                384,
                WG_SMEM,
                &[A::P(gu), A::P(xq.2), A::P(ly.w13), A::P(ly.w13_s), A::P(tiles), A::P(meta), A::P(offs), A::P(rows), A::I(0), A::I((2 * mi) as i32), A::I(h as i32), A::L((2 * mi * h / 2) as i64), A::L((2 * mi * h / 32) as i64)],
            )?;
        } else {
            let (name, grid, block, smem) = if gemv {
                ("dsv_moe_gemv_fp4", [cdiv((2 * mi) as u64, 32), max_tiles as u32, 1], 256, 0)
            } else {
                (moe_gemm_name(bm), [cdiv((2 * mi) as u64, 128), max_tiles as u32, 1], 128, moe_smem(h, bm))
            };
            self.launch(
                name,
                grid,
                block,
                smem,
                &[A::P(gu), A::P(xq.0), A::P(xq.1), A::P(ly.w13), A::P(ly.w13_s), A::P(tiles), A::P(meta), A::P(offs), A::P(rows), A::I(0), A::I((2 * mi) as i32), A::I(h as i32), A::L((2 * mi * h / 2) as i64), A::L((2 * mi * h / 32) as i64)],
            )?;
        }
        let hq = self.arena.alloc((n * mi) as u64)?;
        let hs = self.arena.alloc((n * mi / 32) as u64)?;
        let hfq = if wg { self.arena.alloc((n * mi * 2) as u64)? } else { 0 };
        self.k.cost(Cost::mem((n * 2 * mi * 2 + n * mi + n * mi / 32 + n * 4 + if wg { n * mi * 2 } else { 0 }) as f64));
        self.launch(
            "dsv_swiglu_quant",
            [cdiv((n * mi / 32) as u64, 8), 1, 1],
            256,
            0,
            &[A::P(hq), A::P(hs), A::P(gu), A::P(gu + (mi * 2) as u64), A::L((2 * mi) as i64), A::P(roww), A::I(n as i32), A::I(mi as i32), A::F(c.swiglu_limit), A::P(0), A::P(hfq)],
        )?;
        let down = self.arena.alloc((n * h * 2) as u64)?;
        self.k.cost(Cost { flops: 2.0 * (n * h * mi) as f64, bytes: touched * h as f64 * (mi as f64 / 2.0 + mi as f64 / 32.0) + (n * mi) as f64 + (n * h * 2) as f64, peak: Peak::Fp8Mma });
        if wg {
            self.launch(
                "dsv_moe_gemm_wg_fp4",
                [cdiv(h as u64, WG_BN as u64), max_tiles as u32, 1],
                384,
                WG_SMEM,
                &[A::P(down), A::P(hfq), A::P(ly.w2), A::P(ly.w2_s), A::P(tiles), A::P(meta), A::P(offs), A::P(rows), A::I(1), A::I(h as i32), A::I(mi as i32), A::L((h * mi / 2) as i64), A::L((h * mi / 32) as i64)],
            )?;
        } else {
            let (name, grid, block, smem) = if gemv {
                ("dsv_moe_gemv_fp4", [cdiv(h as u64, 32), max_tiles as u32, 1], 256, 0)
            } else {
                (moe_gemm_name(bm), [cdiv(h as u64, 128), max_tiles as u32, 1], 128, moe_smem(mi, bm))
            };
            self.launch(
                name,
                grid,
                block,
                smem,
                &[A::P(down), A::P(hq), A::P(hs), A::P(ly.w2), A::P(ly.w2_s), A::P(tiles), A::P(meta), A::P(offs), A::P(rows), A::I(1), A::I(h as i32), A::I(mi as i32), A::L((h * mi / 2) as i64), A::L((h * mi / 32) as i64)],
            )?;
        }
        // shared expert (fp8 weights, no routing weight)
        let sgu = self.arena.alloc((t * 2 * mi * 2) as u64)?;
        self.w8a8(sgu, xq, (ly.sh_w13.w, ly.sh_w13.s), t, 2 * mi, h, false)?;
        let shq = self.arena.alloc((t * mi) as u64)?;
        let shs = self.arena.alloc((t * mi / 32) as u64)?;
        let shfq = if self.wg && t >= WG_MIN_M { self.arena.alloc((t * mi * 2) as u64)? } else { 0 };
        self.k.cost(Cost::mem((t * 2 * mi * 2 + t * mi + t * mi / 32) as f64));
        self.launch(
            "dsv_swiglu_quant",
            [cdiv((t * mi / 32) as u64, 8), 1, 1],
            256,
            0,
            &[A::P(shq), A::P(shs), A::P(sgu), A::P(sgu + (mi * 2) as u64), A::L((2 * mi) as i64), A::P(0), A::I(t as i32), A::I(mi as i32), A::F(c.swiglu_limit), A::P(0), A::P(shfq)],
        )?;
        let shared = self.arena.alloc((t * h * 2) as u64)?;
        self.w8a8(shared, (shq, shs, shfq), (ly.sh_w2.w, ly.sh_w2.s), t, h, mi, false)?;
        self.k.cost(Cost::mem((n * h * 2 + t * h * 4) as f64));
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

/// e4m3 codes, ue8m0 scales and (prefill) the dequantized bf16 copy of an activation.
type Q8 = (u64, u64, u64);

/// Rows from which activations also get their bf16 copy and GEMMs may take the wgmma kernel.
const WG_MIN_M: usize = 256;
/// `WG_BN` in dsv41_wg.cu.
const WG_BN: usize = 256;
/// The wgmma kernel when its 128 x 256 tiles fill the 132 SMs; below that the mma.sync kernels with
/// split-K win (bench_wg.py: 32 tiles run at under half their speed).
fn wg_fits(m: usize, n: usize, kd: usize, batch: usize) -> bool {
    m >= WG_MIN_M && kd % 64 == 0 && m.div_ceil(128) * n.div_ceil(WG_BN) * batch >= 132
}

/// The grouped fp4 GEMM's tile height for `rows` rows per touched expert: MOE_WG_BM selects the
/// 128-row wgmma kernel.
fn moe_bm(rows: f64) -> usize {
    if rows >= MOE_WG_MIN_ROWS {
        return MOE_WG_BM;
    }
    if rows <= MOE_BM16_MAX_ROWS {
        16
    } else if rows <= MOE_BM32_MAX_ROWS {
        32
    } else {
        64
    }
}
const MOE_BM16_MAX_ROWS: f64 = 8.0;
const MOE_BM32_MAX_ROWS: f64 = 64.0;
const MOE_WG_BM: usize = 128;
/// bench_wg.py: the wgmma kernel ties the 64-row mma.sync tile at 16 rows per expert (1k tokens) and
/// wins 1.5x at 64 (4k tokens).
const MOE_WG_MIN_ROWS: f64 = 48.0;

/// Steps of at most this many tokens run the routed experts through `dsv_moe_gemv_fp4`. Off: the
/// 16-row MMA tile streams the weights faster at every decode size (bench_moe.py).
const MOE_GEMV_MAX_T: usize = 0;

/// K splits for a GEMM of `tiles` output tiles over `kb` 32-wide K blocks: enough blocks for about
/// two waves on the 132 SMs, each split keeping at least 4 K blocks; 1 when the grid already fills.
fn ksplit(tiles: u32, kb: usize) -> usize {
    const TARGET: u32 = 264;
    if tiles >= TARGET / 2 {
        return 1;
    }
    (TARGET.div_ceil(tiles.max(1)) as usize).min(kb / 4).max(1)
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
