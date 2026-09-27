//! DeepSeek-V4.1 batched decode: one token per slot over `dbatch` slots, and the per-slot state
//! a prefill chunk seeds for it (model.py `Attention.forward` / `Compressor.forward` /
//! `Indexer.forward` at `start_pos > 0`).
//!
//! Every persistent tensor is `kv.*`, `[dbatch][..]` slot-major: the CUDA engine gives each slot
//! a table whose `kv.*` bases are shifted by `slot * bytes / dbatch`, so the prefill program writes
//! "slot 0" and lands in its own slot, while the decode program addresses all slots by row.
use super::dsv41::*;
use super::*;
use std::collections::BTreeMap;

/// The compressed ratio of layer `l`, 0 for a window-only layer.
pub(crate) fn dsv41_ratio(c: &Dsv41Cfg, l: u32) -> u32 {
    match c.raw.attn_kind(l) {
        nn_graph::models::config::V41Attn::Window => 0,
        nn_graph::models::config::V41Attn::Compressed { ratio } => ratio,
    }
}

/// The `kv_source` whose cache (and index keys) layer `l` reads: the last one at or before it
/// (`shared_attn.compress_kv` / `index_k` are rewritten by each source in layer order).
pub(crate) fn dsv41_src_of(c: &Dsv41Cfg, l: u32) -> Option<u32> {
    c.kv_source.iter().copied().filter(|&s| s <= l).max()
}

/// Per-slot decode state.
pub(crate) struct Dsv41State {
    pub(crate) dbatch: u32,
    /// `kv.l{l}.win`, `[dbatch][window][head_dim]` bf16: the sliding-window ring, post rope and
    /// fp8 fake quant, row `pos % window`.
    pub(crate) win: BTreeMap<u32, u32>,
    /// `kv.cmp{s}`, `[dbatch][ctx/ratio][head_dim]` bf16: source `s`'s compressed cache.
    pub(crate) cmp: BTreeMap<u32, u32>,
    /// `kv.ik{s}`, `[dbatch][ctx/ratio][index_dim]` bf16: source `s`'s index keys.
    pub(crate) ik: BTreeMap<u32, u32>,
    /// `kv.cst{s}.kv` / `.sc`, `[dbatch][ratio][head_dim]` f32: the compressor's incomplete group
    /// (`kv_state` / `score_state`), ratio > 1 sources only.
    pub(crate) cst: BTreeMap<u32, (u32, u32)>,
}

pub(crate) fn declare_dsv41_state(b: &mut Builder, c: &Dsv41Cfg, layers: &[u32], ctx: u32, dbatch: u32) -> Dsv41State {
    assert!(c.sliding_window.is_power_of_two(), "the window ring is addressed pos & (window - 1)");
    let (db, hd, di) = (dbatch as u64, c.head_dim as u64, c.index_dim as u64);
    let mut st = Dsv41State { dbatch, win: BTreeMap::new(), cmp: BTreeMap::new(), ik: BTreeMap::new(), cst: BTreeMap::new() };
    for &l in layers {
        st.win.insert(l, b.tensor(&format!("kv.l{l}.win"), db * c.sliding_window as u64 * hd * 2));
        let src = if c.kv_source.contains(&l) { Some(l) } else if dsv41_ratio(c, l) != 0 { dsv41_src_of(c, l) } else { None };
        if let Some(s) = src {
            let r = dsv41_ratio(c, s);
            let rows = (ctx / r) as u64;
            st.cmp.entry(s).or_insert_with(|| b.tensor(&format!("kv.cmp{s}"), db * rows * hd * 2));
            st.ik.entry(s).or_insert_with(|| b.tensor(&format!("kv.ik{s}"), db * rows * di * 2));
            if r > 1 {
                st.cst.entry(s).or_insert_with(|| {
                    (b.tensor(&format!("kv.cst{s}.kv"), db * r as u64 * hd * 4), b.tensor(&format!("kv.cst{s}.sc"), db * r as u64 * hd * 4))
                });
            }
        }
    }
    st
}

/// Prefill: seed this slot's window ring with the chunk's rows `[kvlen-W, kvlen)` of `kv` as the
/// attention core left them (roped and fp8 fake-quantized in place), copied without a second rope;
/// the re-quant at the same pow2 blocks is exact. It runs AFTER the core, because a later chunk's
/// core reads the previous chunk's tail from this ring.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dsv41_ring_seed(
    b: &mut Builder, c: &Dsv41Cfg, st: &Dsv41State, cus: &[u32], l: u32, kv: u32, pos: u32, kvlen: u32, t: u32, deps: &[u32],
) -> u32 {
    b.emit(DevOp::CompressRopeQuant, cus.to_vec(), deps, |d| {
        d.t[0] = st.win[&l];
        d.t[1] = kv;
        d.t[2] = TENSOR_NONE;
        d.t[3] = TENSOR_NONE;
        d.t[4] = pos;
        d.t[5] = kvlen;
        d.i[0] = t;
        d.i[1] = c.head_dim;
        d.i[2] = 0; // no rope: `kv` is already roped
        d.i[3] = 32;
        d.i[4] = 1;
        d.i[5] = 0;
        d.i[6] = 0; // PLOW_CMP_Q_FP8_POW2: act_quant(kv, 32, ue8m0)
        d.i[7] = 1;
        d.j[0] = 0;
        d.j[1] = (1u32 << 31) | (c.sliding_window - 1);
    })
}

/// Prefill: the chunk's incomplete tail group of the compressor projections into the state.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dsv41_state_seed(
    b: &mut Builder, c: &Dsv41Cfg, st: &Dsv41State, cus: &[u32], l: u32, kv: u32, score: u32, kvlen: u32, deps: &[u32],
) -> Option<u32> {
    let &(sk, ss) = st.cst.get(&l)?;
    Some(b.emit(DevOp::CompressDecodeStep, cus.to_vec(), deps, |d| {
        d.t[0] = TENSOR_NONE;
        d.t[1] = sk;
        d.t[2] = ss;
        d.t[3] = kv;
        d.t[4] = score;
        d.t[5] = TENSOR_NONE;
        d.t[6] = kvlen;
        d.i[0] = 1;
        d.i[1] = dsv41_ratio(c, l);
        d.i[2] = c.head_dim;
        d.i[3] = 1;
    }))
}

/// Decode-only scratch, `[dbatch][..]`.
pub(crate) struct Dsv41DecodeAct {
    /// `shared_attn.topk_idxs`: `[dbatch][index_topk]` i32, republished by each index layer.
    pub(crate) idx: u32,
    /// `[dbatch][ctx]` f32 indexer score (widest at ratio 1).
    pub(crate) score: u32,
    /// Sparse attention split partials.
    pub(crate) attn_part: u32,
    pub(crate) nsplit: u32,
}

/// `nsplit` for [`DevOp::SparseAttnDecode`]: the most splits that keep every (slot, 16-head
/// group, split) item in one round of the grid, at most 20 (a split is at least 32 rows of 640).
pub(crate) fn dsv41_attn_nsplit(n_cu: u32, dbatch: u32, heads: u32) -> u32 {
    (n_cu / (dbatch * (heads / 16)).max(1)).clamp(1, 20)
}

pub(crate) fn declare_dsv41_decode_act(b: &mut Builder, c: &Dsv41Cfg, tp: u32, ctx: u32, dbatch: u32) -> Dsv41DecodeAct {
    let nh_l = c.heads / tp;
    assert_eq!(nh_l % 16, 0, "sparse decode puts 16 heads on one MMA tile; {nh_l} per rank");
    let nsplit = dsv41_attn_nsplit(b.n_cu(), dbatch, nh_l);
    let db = dbatch as u64;
    Dsv41DecodeAct {
        idx: b.tensor("act.d_index_idx", db * c.index_topk as u64 * 4),
        score: b.tensor("act.d_index_score", db * ctx as u64 * 4),
        attn_part: b.tensor("act.d_attn_part", db * (nh_l / 16) as u64 * nsplit as u64 * (16 * c.head_dim as u64 + 32) * 4),
        nsplit,
    }
}

/// One layer's ATTENTION sublayer at decode, rows = slots. Returns the attention output
/// `[dbatch][heads/tp][head_dim]` completion; the caller runs `emit_dsv41_attn_out` and mHC.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_dsv41_attn_decode(
    b: &mut Builder,
    c: &Dsv41Cfg,
    w: &Dsv41Weights,
    st: &Dsv41State,
    da: &Dsv41DecodeAct,
    cp: Option<&Dsv41Compress>,
    ix: Option<&Dsv41Index>,
    cus: &[u32],
    l: u32,
    tp: u32,
    proj: &Dsv41ProjAct,
    pos: u32,
    cos: u32,
    sin: u32,
    ctx: u32,
    attn_o: u32,
    deps: &[u32],
) -> u32 {
    let bsz = st.dbatch;
    let (hd, rope) = (c.head_dim, c.qk_rope);
    let nope = hd - rope;
    let nh_l = c.heads / tp;
    let all = cus.to_vec();
    let ratio = dsv41_ratio(c, l);

    // q: interior rope at pos[row], in place (rows are slots).
    let c_q = b.emit(DevOp::QwenHeadNormRope, all.clone(), deps, |d| {
        d.t[0] = proj.q;
        d.t[1] = proj.q;
        d.t[2] = TENSOR_NONE;
        d.t[3] = cos;
        d.t[4] = sin;
        d.t[5] = pos;
        d.i[0] = nh_l;
        d.i[1] = hd;
        d.i[2] = rope | (1u32 << 31);
        d.i[3] = bsz;
        d.i[4] = 0;
        d.i[5] = 0;
        d.i[6] = 1;
        d.i[7] = nope;
    });
    // kv: rope + fp8 fake quant straight into ring row pos % W (`_window_kv`, decode branch).
    let c_ring = b.emit(DevOp::CompressRopeQuant, all.clone(), deps, |d| {
        d.t[0] = st.win[&l];
        d.t[1] = proj.kv;
        d.t[2] = cos;
        d.t[3] = sin;
        d.t[4] = pos;
        d.i[0] = bsz;
        d.i[1] = hd;
        d.i[2] = rope;
        d.i[3] = 32;
        d.i[4] = 1;
        d.i[5] = 0;
        d.i[6] = 0;
        d.i[7] = 1;
        d.j[0] = c.sliding_window;
        d.j[1] = (1u32 << 31) | (c.sliding_window - 1);
    });
    let mut attn_deps = vec![c_q, c_ring];

    // ---- compressor (kv_source layers): state step, then the new cache row and index key ----
    if c.kv_source.contains(&l) {
        let cp = cp.expect("a kv_source layer has compressor scratch");
        let rows = ctx / ratio;
        let c_lat = if ratio == 1 {
            let op = crate::pick_tile(bsz, hd, c.hidden, b.n_cu(), kernelcaps::QuantScheme::None);
            let c_kv = b.emit(op, all.clone(), deps, |d| {
                d.t[0] = cp.kv;
                d.t[1] = proj.xn;
                d.t[2] = w.get(l, "attn.compressor.wkv.weight");
                d.i[0] = bsz;
                d.i[1] = hd;
                d.i[2] = c.hidden;
            });
            b.emit(DevOp::RmsNorm, all.clone(), &[c_kv], |d| {
                d.t[0] = cp.latent;
                d.t[1] = cp.kv;
                d.t[2] = w.get(l, "attn.compressor.norm.weight");
                d.i[0] = bsz;
                d.i[1] = hd;
                d.f[0] = c.eps;
            })
        } else {
            let f32_gemm = |b: &mut Builder, out: u32, weight: u32, role: &str| {
                let (split, splits) = crate::mla::nv_gemm_f32_split(b, all.len(), role, bsz, hd, c.hidden);
                b.emit(DevOp::GemmF32, all.clone(), deps, |d| {
                    d.t[0] = out;
                    d.t[1] = proj.xn;
                    d.t[2] = weight;
                    d.t[3] = split;
                    d.i[0] = bsz;
                    d.i[1] = hd;
                    d.i[2] = c.hidden;
                    d.i[3] = splits;
                })
            };
            let c_kv = f32_gemm(b, cp.kv, w.get(l, "attn.compressor.wkv.weight"), "ckv");
            let c_gt = f32_gemm(b, cp.gate, w.get(l, "attn.compressor.wgate.weight"), "cgate");
            let (sk, ss) = st.cst[&l];
            b.emit(DevOp::CompressDecodeStep, all.clone(), &[c_kv, c_gt], |d| {
                d.t[0] = cp.latent;
                d.t[1] = sk;
                d.t[2] = ss;
                d.t[3] = cp.kv;
                d.t[4] = cp.gate;
                d.t[5] = w.get(l, "attn.compressor.norm.weight");
                d.t[6] = pos;
                d.i[0] = bsz;
                d.i[1] = ratio;
                d.i[2] = hd;
                d.f[0] = c.eps;
            })
        };
        // Index key from the PRE-rope latent (the indexer reads it before the cache row is roped).
        let ix = ix.expect("kv_source layers are index sources");
        let (di, hdi) = (c.index_dim, c.head_dim);
        let op = crate::pick_tile(bsz, di, hdi, b.n_cu(), kernelcaps::QuantScheme::None);
        let c_k = b.emit(op, all.clone(), &[c_lat], |d| {
            d.t[0] = ix.k;
            d.t[1] = cp.latent;
            d.t[2] = w.get(l, "attn.indexer.wk.weight");
            d.i[0] = bsz;
            d.i[1] = di;
            d.i[2] = hdi;
        });
        let c_kn = b.emit(DevOp::RmsNorm, all.clone(), &[c_k], |d| {
            d.t[0] = ix.kn;
            d.t[1] = ix.k;
            d.t[2] = w.get(l, "attn.indexer.k_norm.weight");
            d.i[0] = bsz;
            d.i[1] = di;
            d.f[0] = c.eps;
        });
        let append = |b: &mut Builder, out: u32, src: u32, d_: u32, qblk: u32, qmode: u32, dep: u32| {
            b.emit(DevOp::CompressRopeQuant, all.clone(), &[dep], |d| {
                d.t[0] = out;
                d.t[1] = src;
                d.t[2] = cos;
                d.t[3] = sin;
                d.t[4] = pos;
                d.i[0] = bsz;
                d.i[1] = d_;
                d.i[2] = rope;
                d.i[3] = qblk;
                d.i[4] = ratio;
                d.i[5] = 0;
                d.i[6] = qmode;
                d.i[7] = 1;
                d.j[0] = rows;
                d.j[1] = 1u32 << 31;
            })
        };
        attn_deps.push(append(b, st.ik[&l], ix.kn, di, DSV41_IDX_QBLK, 1, c_kn));
        attn_deps.push(append(b, st.cmp[&l], cp.latent, hd, DSV41_KV_QBLK, 2, c_lat));
    }

    // ---- indexer (index_source layers): score this slot's keys, publish its top-k ----
    if c.index_source.contains(&l) {
        let ix = ix.expect("index layers have indexer scratch");
        let src = dsv41_src_of(c, l).expect("an index layer reads some source's keys");
        let (hi, di) = (c.index_heads, c.index_dim);
        let c_q = crate::mla::emit_pf_gemm_fp8_mx(
            b, cus, ix.q, proj.q_an, w.get(l, "attn.indexer.wq_b.weight"), w.get(l, "attn.indexer.wq_b.scale"), bsz, hi * di, c.q_lora, deps,
        );
        let c_qr = b.emit(DevOp::CompressRopeQuant, all.clone(), &[c_q], |d| {
            d.t[0] = ix.qr;
            d.t[1] = ix.q;
            d.t[2] = cos;
            d.t[3] = sin;
            d.t[4] = pos;
            d.i[0] = bsz;
            d.i[1] = di;
            d.i[2] = rope;
            d.i[3] = DSV41_IDX_QBLK;
            d.i[4] = 1;
            d.i[5] = 0;
            d.i[6] = 1;
            d.i[7] = hi;
            d.j[0] = 0;
            d.j[1] = 1u32 << 31;
        });
        let op = crate::pick_tile(bsz, hi, c.hidden, b.n_cu(), kernelcaps::QuantScheme::None);
        let c_w = b.emit(op, all.clone(), deps, |d| {
            d.t[0] = ix.w;
            d.t[1] = proj.xn;
            d.t[2] = w.get(l, "attn.indexer.weights_proj.weight");
            d.i[0] = bsz;
            d.i[1] = hi;
            d.i[2] = c.hidden;
        });
        let cap = ctx / ratio;
        let mut sdeps = vec![c_qr, c_w];
        sdeps.extend_from_slice(&attn_deps);
        let c_sc = b.emit(DevOp::IndexScoreDecode, all.clone(), &sdeps, |d| {
            d.t[0] = da.score;
            d.t[1] = ix.qr;
            d.t[2] = ix.w;
            d.t[3] = st.ik[&src];
            d.t[4] = pos;
            d.i[0] = bsz;
            d.i[1] = hi;
            d.i[2] = cap;
            d.i[3] = ratio;
            d.f[0] = (di as f32).powf(-0.5) * (hi as f32).powf(-0.5);
        });
        attn_deps.push(b.emit(DevOp::IndexSelectDecode, all.clone(), &[c_sc], |d| {
            d.t[0] = da.idx;
            d.t[1] = da.score;
            d.t[2] = pos;
            d.i[0] = bsz;
            d.i[1] = cap;
            d.i[2] = ratio;
        }));
    }

    // ---- sparse attention over ring + selected compressed rows, sink folded at the merge ----
    let compressed = (ratio != 0).then(|| {
        let src = dsv41_src_of(c, l).expect("a compressed layer reads some source's cache");
        (st.cmp[&src], ctx / dsv41_ratio(c, src))
    });
    let sink = w.get(l, "attn.attn_sink");
    let c_at = b.emit(DevOp::SparseAttnDecode, all.clone(), &attn_deps, |d| {
        d.t[0] = attn_o;
        d.t[1] = proj.q;
        d.t[2] = st.win[&l];
        d.t[3] = compressed.map_or(TENSOR_NONE, |(t, _)| t);
        d.t[4] = if compressed.is_some() { da.idx } else { TENSOR_NONE };
        d.t[5] = pos;
        d.t[6] = sink;
        d.t[7] = da.attn_part;
        d.i[0] = bsz;
        d.i[1] = nh_l;
        d.i[2] = c.sliding_window;
        d.i[3] = compressed.map_or(0, |(_, r)| r);
        d.i[4] = if compressed.is_some() { c.index_topk } else { 0 };
        d.i[5] = da.nsplit;
        d.f[0] = 1.0 / (hd as f32).sqrt();
    });
    let c_at = if da.nsplit > 1 {
        b.emit(DevOp::SparseAttnMerge, all.clone(), &[c_at], |d| {
            d.t[0] = attn_o;
            d.t[1] = da.attn_part;
            d.t[2] = sink;
            d.i[0] = bsz;
            d.i[1] = nh_l;
            d.i[2] = da.nsplit;
        })
    } else {
        c_at
    };
    // `apply_rotary_emb(o[..., -rd:], freqs_cis, True)` at each slot's own position.
    b.emit(DevOp::RopeInverseO, all.clone(), &[c_at], |d| {
        d.t[0] = attn_o;
        d.t[1] = cos;
        d.t[2] = sin;
        d.t[3] = pos;
        d.i[0] = bsz;
        d.i[1] = nh_l;
        d.i[2] = hd;
        d.i[3] = rope;
        d.i[4] = 0;
        d.i[5] = 1;
    })
}
