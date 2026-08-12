//! DeepSeek-V4 decode emitter — the packet chain for one token, batch 1.
//!
//! This is the piece that turns the gated V4 kernels into something `plowrt
//! serve` can run. Every op below is either one this bring-up added (the
//! `V4*` opcodes) or one the tree already ships tuned; the split matters,
//! because ~90% of a decode step's BYTES are ordinary projections and experts
//! that `GemvFp8Blk` / `GemvMxfp4` / `MoeExpert*Fp8Blk` already carry at speed.
//! What is V4-specific is small in bytes and was the whole of the bring-up.
//!
//! # Scope of this file, stated plainly
//!
//! Decode, batch 1, TP 1. That is the shape the 200 tok/s target names, and it
//! is deliberately the narrowest thing that can produce a served number:
//!
//! * NO PREFILL BUCKETS. Without them `AmdServe::prefill` walks the prompt
//!   through the decode program one token at a time, which is what made GLM's
//!   TTFT 20x vLLM's. For a decode-throughput number that is acceptable and for
//!   a TTFT number it is not, so it is recorded here rather than discovered.
//! * NO TP. The V4 indexer has a hard TP constraint the emitter must honour
//!   when it lands — the score is all-reduced BEFORE the top-k, so the
//!   selection is a collective and ranks that disagree decode different tokens.
//! * NO DSPARK/MTP.
//!
//! # The per-layer chain
//!
//! ```text
//!   V4HcDot -> V4HcMix                  hyper-connection reduce (attn side)
//!   RmsNorm                             attn_norm
//!   GemvFp8Blk  wq_a                    q down-projection
//!   RmsNorm     q_norm
//!   GemvFp8Blk  wq_b                    q up-projection, 64 heads x 512
//!   HeadNormRope                        weightless per-head RMS + rope tail
//!   GemvFp8Blk  wkv                     the single shared KV head
//!   HeadNormRope                        kv rope tail, writes the KV ring row
//!   [compressed layers]
//!     GemvFp8Blk x2 -> V4KvCompressStep   compressor projections + window state
//!   [indexed layers]
//!     GemvFp8Blk -> V4IndexScore -> V4IndexTopk
//!   V4SparseAttn (split-K) -> merge
//!   HeadNormRope                        inverse rope on the output
//!   V4GroupedLinear wo_a
//!   GemvFp8Blk      wo_b
//!   V4HcExpand
//!   V4HcDot -> V4HcMix                  hyper-connection reduce (ffn side)
//!   RmsNorm                             ffn_norm
//!   GemvFp8Blk gate -> V4MoeRoute
//!   MoeExpertGluFp8Blk / GemvMxfp4 ...  experts
//!   V4ClampedSwiGlu
//!   V4HcExpand
//! ```
//!
//! # What is NOT here yet
//!
//! The host-side weight BIND. `plowrt` maps checkpoint tensors onto the handles
//! declared here by name, and V4's names are its own (`layers.N.attn.wq_a`, not
//! `model.layers.N.self_attn.*`); the fp8 block scales and the fp4 packing each
//! need their layout carried through. Until that lands this emitter builds a
//! structurally correct program that cannot load, which is why the entry point
//! is behind `PLOW_V4_FULL` and the default stays the capability report.

use crate::mla::deepseek_v4::{cfg_deepseek_v4, V4Cfg};
use packet::dev::DevOp;
use packet::devbuild::{Builder, Model};
use std::path::Path;

/// Tensor handles a V4 decode program needs. One table serves the whole
/// program, as `Model` requires.
pub(crate) struct V4Tn {
    pub ids: u32,
    pub pos: u32,
    pub cos: u32,
    pub sin: u32,
    /// Inverse-rope table: the same angles with `sin` negated, for the
    /// de-rotation V4 applies to the attention output.
    pub cos_i: u32,
    pub sin_i: u32,
    pub embed: u32,
    pub fin_norm: u32,
    pub head: u32,
    /// `[hc, hidden]` residual streams — V4 has no single residual vector.
    pub x: u32,
    pub xr: u32,
    pub xn: u32,
    pub q: u32,
    pub kv: u32,
    pub attn_out: u32,
    pub logits: u32,
    pub amax: u32,
    /// Per-site hyper-connection scratch: the `[1+mix]` partial the split
    /// reduce accumulates into, and the `[hc + hc*hc]` post/comb the paired
    /// expand reads.
    pub hc_partial: u32,
    pub hc_mix: u32,
}

const BF16: u64 = 2;
const F32: u64 = 4;
const I32: u64 = 4;

/// Declare the activation and scratch tensors for a decode program.
///
/// Weight handles are declared per layer by [`emit_v4_layer`] so their names
/// match the checkpoint exactly — that name IS the loader's contract.
pub(crate) fn declare_v4(b: &mut Builder, c: &V4Cfg, ctx: u32) -> V4Tn {
    let h = c.hidden as u64;
    let hc = c.hc_mult as u64;
    let mix = (2 + hc) * hc;
    let nh = c.heads as u64;
    let hd = c.head_dim as u64;

    let ids = b.tensor("in.ids", ctx as u64 * I32);
    // `in.kvlen` is what makes this a MODEL rather than a block asset: the
    // runtime's token-level entry points look for it by name and refuse the
    // blob outright when it is absent. Sized `batch * 4`, batch 1 here.
    b.tensor("in.kvlen", I32);
    let pos = b.tensor("in.pos", ctx as u64 * I32);
    let cos = b.tensor("in.cos", ctx as u64 * (hd / 2) * F32);
    let sin = b.tensor("in.sin", ctx as u64 * (hd / 2) * F32);
    let cos_i = b.tensor("in.cos_inv", ctx as u64 * (hd / 2) * F32);
    let sin_i = b.tensor("in.sin_inv", ctx as u64 * (hd / 2) * F32);

    let embed = b.tensor("embed.weight", c.vocab as u64 * h * BF16);
    let fin_norm = b.tensor("norm.weight", h * BF16);
    let head = b.tensor("head.weight", c.vocab as u64 * h * BF16);

    let ac = |b: &mut Builder, n: &str, sz: u64| b.tensor(&format!("act.{n}"), sz);
    // `x` is [hc, hidden]: the parallel residual streams, not one vector.
    let x = ac(b, "x", hc * h * BF16);
    let xr = ac(b, "xr", h * BF16);
    let xn = ac(b, "xn", h * BF16);
    let q = ac(b, "q", nh * hd * BF16);
    let kv = ac(b, "kv", hd * BF16);
    let attn_out = ac(b, "attn_out", nh * hd * BF16);
    let logits = ac(b, "logits", c.vocab as u64 * F32);
    let amax = ac(b, "amax", 256 * F32);
    let hc_partial = ac(b, "hc_partial", (1 + mix) * F32);
    let hc_mix = ac(b, "hc_mix", (hc + hc * hc) * F32);

    V4Tn {
        ids,
        pos,
        cos,
        sin,
        cos_i,
        sin_i,
        embed,
        fin_norm,
        head,
        x,
        xr,
        xn,
        q,
        kv,
        attn_out,
        logits,
        amax,
        hc_partial,
        hc_mix,
    }
}

/// The hyper-connection reduce, as the two packets the roofline work settled
/// on: a grid-parallel projection then the Sinkhorn tail.
///
/// `name` is the checkpoint prefix (`layers.3.hc_attn`), and both packets read
/// the same three weights — that is not a mistake, both nodes really do.
#[allow(clippy::too_many_arguments)]
fn emit_hc_reduce(
    b: &mut Builder,
    c: &V4Cfg,
    tn: &V4Tn,
    name: &str,
    out: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let h = c.hidden as u64;
    let hc = c.hc_mult as u64;
    let mix = (2 + hc) * hc;
    let fnw = b.tensor(&format!("{name}_fn"), mix * hc * h * F32);
    let scale = b.tensor(&format!("{name}_scale"), 3 * F32);
    let base = b.tensor(&format!("{name}_base"), mix * F32);
    let all: Vec<u32> = (0..n_cu).collect();

    // `V4HcDot` accumulates with atomics, so its partial must start at zero —
    // its opcode doc says exactly that. Fresh device memory made ONE reduce
    // site per program correct by accident; the second site in a layer, and
    // every site on the second token, accumulated onto the previous residue.
    let z = b.emit(DevOp::V4HcZero, all.clone(), deps, |d| {
        d.t[0] = tn.hc_partial;
        d.i[0] = 1 + mix as u32;
    });
    let dot = b.emit(DevOp::V4HcDot, all.clone(), &[z], |d| {
        d.t[0] = tn.hc_partial;
        d.t[1] = tn.x;
        d.t[2] = fnw;
        d.i[0] = 1;
        d.i[1] = c.hidden as u32;
        d.i[2] = c.hc_mult;
    });
    b.emit(DevOp::V4HcMix, all, &[dot], |d| {
        d.t[0] = out;
        d.t[1] = tn.x;
        d.t[2] = tn.hc_partial;
        d.t[3] = scale;
        d.t[4] = base;
        d.t[5] = tn.hc_mix;
        d.i[0] = 1;
        d.i[1] = c.hidden as u32;
        d.i[2] = c.hc_mult;
        d.i[3] = c.hc_iters;
        d.f[0] = 1e-6;
        d.f[1] = 1e-6; /* hc_eps */
    })
}

/// Write the branch output back into the residual streams.
#[allow(clippy::too_many_arguments)]
fn emit_hc_expand(
    b: &mut Builder,
    c: &V4Cfg,
    tn: &V4Tn,
    branch: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let all: Vec<u32> = (0..n_cu).collect();
    b.emit(DevOp::V4HcExpand, all, deps, |d| {
        d.t[0] = tn.x; // in place: every output reads the whole stream vector first
        d.t[1] = branch;
        d.t[2] = tn.x;
        d.t[3] = tn.hc_mix;
        d.i[0] = 1;
        d.i[1] = c.hidden as u32;
        d.i[2] = c.hc_mult;
    })
}

/// A block-fp8 decode GEMV, the arm every V4 projection uses.
///
/// The weight and its `[N/128][K/128]` scale grid are declared together and
/// named EXACTLY as the checkpoint spells them — `layers.3.attn.wq_b.weight`
/// and `.scale`. That pair of names is the loader's contract; a `.weight`
/// suffix or an `attn` vs `self_attn` prefix wrong here is a load failure, not
/// a wrong number.
#[allow(clippy::too_many_arguments)]
/// A plain bf16 GEMV.
///
/// NOT every V4 projection is quantized, and assuming so is a load failure at
/// best: `compressor.wkv` / `compressor.wgate` ship BF16 in the released
/// checkpoint, so the fp8 path declared half the bytes they occupy and invented
/// a `.scale` twin that does not exist.
#[allow(clippy::too_many_arguments)]
fn emit_bf16_gemv(
    b: &mut Builder,
    name: &str,
    out: u32,
    x: u32,
    n: u32,
    k: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let w = b.tensor(&format!("{name}.weight"), n as u64 * k as u64 * BF16);
    let all: Vec<u32> = (0..n_cu).collect();
    b.emit(DevOp::Gemv, all, deps, |d| {
        d.t[0] = out;
        d.t[1] = x;
        d.t[2] = w;
        d.i[0] = 1;
        d.i[1] = n;
        d.i[2] = k;
    })
}

fn emit_fp8_gemv(
    b: &mut Builder,
    name: &str,
    out: u32,
    x: u32,
    n: u32,
    k: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let w = b.tensor(&format!("{name}.weight"), n as u64 * k as u64);
    let ws = b.tensor(
        &format!("{name}.scale"),
        n.div_ceil(128) as u64 * k.div_ceil(128) as u64 * F32,
    );
    let all: Vec<u32> = (0..n_cu).collect();
    b.emit(DevOp::GemvFp8Blk, all, deps, |d| {
        d.t[0] = out;
        d.t[1] = x;
        d.t[2] = w;
        d.t[5] = ws;
        d.i[0] = 1;
        d.i[1] = n;
        d.i[2] = k;
    })
}

/// The attention sub-layer: reduce, norm, q/kv projections and ropes, the
/// compressor and indexer on the layers that have them, split-K attention, the
/// output de-rotation, and the block-diagonal output projection.
///
/// Returns the packet the FFN side must depend on.
#[allow(clippy::too_many_arguments)]
fn emit_v4_attn(
    b: &mut Builder,
    c: &V4Cfg,
    tn: &V4Tn,
    l: u32,
    ctx: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    let p = format!("layers.{l}");
    let all: Vec<u32> = (0..n_cu).collect();
    let h = c.hidden as u32;
    let nh = c.heads;
    let hd = c.head_dim;
    let ql = c.q_lora as u32;

    // hc reduce -> attn_norm
    let red = emit_hc_reduce(b, c, tn, &format!("{p}.hc_attn"), tn.xr, n_cu, deps);
    let anw = b.tensor(&format!("{p}.attn_norm.weight"), h as u64 * BF16);
    let nrm = b.emit(DevOp::RmsNorm, all.clone(), &[red], |d| {
        d.t[0] = tn.xn;
        d.t[1] = tn.xr;
        d.t[2] = anw;
        d.i[0] = 1;
        d.i[1] = h;
        d.f[0] = 1e-6;
    });

    // q: wq_a -> q_norm -> wq_b -> weightless per-head RMS + rope tail.
    let qlr = b.tensor("act.qlr", ql as u64 * BF16);
    let qa = emit_fp8_gemv(
        b,
        &format!("{p}.attn.wq_a"),
        qlr,
        tn.xn,
        ql,
        h,
        n_cu,
        &[nrm],
    );
    let qnw = b.tensor(&format!("{p}.attn.q_norm.weight"), ql as u64 * BF16);
    let qn = b.emit(DevOp::RmsNorm, all.clone(), &[qa], |d| {
        d.t[0] = qlr;
        d.t[1] = qlr;
        d.t[2] = qnw;
        d.i[0] = 1;
        d.i[1] = ql;
        d.f[0] = 1e-6;
    });
    let qb = emit_fp8_gemv(
        b,
        &format!("{p}.attn.wq_b"),
        tn.q,
        qlr,
        nh * hd,
        ql,
        n_cu,
        &[qn],
    );
    // `gamma == NONE` is the weightless rescale; i5 == 1 selects the
    // INTERLEAVED HD=512 template, which ropes the trailing lanes through an
    // identity-prefix table (see the v4_rope oracle).
    let qrope = b.emit(DevOp::HeadNormRope, all.clone(), &[qb], |d| {
        d.t[0] = tn.q;
        d.t[1] = tn.q;
        d.t[2] = packet::dev::TENSOR_NONE;
        d.t[3] = tn.cos;
        d.t[4] = tn.sin;
        d.t[5] = tn.pos;
        d.i[0] = 1;
        d.i[1] = nh;
        d.i[2] = hd;
        d.i[5] = 1;
        d.f[0] = 1e-6;
    });

    // The single shared KV head, roped and written into the sliding-window ring.
    let kvring = b.tensor(
        &format!("kv.{l}"),
        (c.window as u64 + ctx as u64 / 4) * hd as u64 * BF16,
    );
    let kva = emit_fp8_gemv(
        b,
        &format!("{p}.attn.wkv"),
        tn.kv,
        tn.xn,
        hd,
        h,
        n_cu,
        &[nrm],
    );
    let knw = b.tensor(&format!("{p}.attn.kv_norm.weight"), hd as u64 * BF16);
    let kvrope = b.emit(DevOp::HeadNormRope, all.clone(), &[kva], |d| {
        d.t[0] = kvring;
        d.t[1] = tn.kv;
        d.t[2] = knw;
        d.t[3] = tn.cos;
        d.t[4] = tn.sin;
        d.t[5] = tn.pos;
        d.i[0] = 1;
        d.i[1] = 1;
        d.i[2] = hd;
        d.i[5] = 1;
        d.f[0] = 1e-6;
        // `j1` is the ring mask: the window is a power-of-two ring, so the
        // write position wraps with one `v_and_b32`.
        d.j[1] = c.window - 1;
    });

    let mut adeps = vec![qrope, kvrope];

    // The compressor, on the 41 layers that have one.
    if let Some(r) = c.compress_ratio(l) {
        let coff = if V4Cfg::overlaps(r) { 2 } else { 1 };
        let w = coff * hd;
        let ckv = b.tensor(&format!("act.ckv.{l}"), w as u64 * F32);
        let cgt = b.tensor(&format!("act.cgate.{l}"), w as u64 * F32);
        let g1 = emit_bf16_gemv(
            b,
            &format!("{p}.attn.compressor.wkv"),
            ckv,
            tn.xn,
            w,
            h,
            n_cu,
            &[nrm],
        );
        let g2 = emit_bf16_gemv(
            b,
            &format!("{p}.attn.compressor.wgate"),
            cgt,
            tn.xn,
            w,
            h,
            n_cu,
            &[nrm],
        );
        let ape = b.tensor(
            &format!("{p}.attn.compressor.ape"),
            r as u64 * w as u64 * F32,
        );
        let cnw = b.tensor(
            &format!("{p}.attn.compressor.norm.weight"),
            hd as u64 * BF16,
        );
        // Window state is a RUNTIME resource and a PER-SEQUENCE one, exactly
        // like the KV ring — it carries the partial window between steps. So it
        // is named under `kv.`, which is what makes the runtime allocate and
        // rotate it per sequence. A `state.` prefix is not recognised as
        // runtime at all: the loader would go looking for it in the checkpoint
        // and fail with MISSING WEIGHT, which is exactly what the
        // name-classification test caught.
        let kvs = b.tensor(
            &format!("kv.cstate.{l}"),
            (coff * r) as u64 * w as u64 * F32,
        );
        let scs = b.tensor(
            &format!("kv.cscore.{l}"),
            (coff * r) as u64 * w as u64 * F32,
        );
        let step = b.emit(DevOp::V4KvCompress, vec![0], &[g1, g2], |d| {
            d.t[0] = kvring;
            d.t[1] = kvs;
            d.t[2] = scs;
            d.t[3] = ckv;
            d.t[4] = cgt;
            d.t[5] = ape;
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = hd;
            d.i[3] = r;
            d.i[4] = coff - 1;
            d.f[0] = 1e-6;
            let _ = cnw;
        });
        adeps.push(step);
    }

    // The indexer, on the 21 ratio-4 layers: its own compressed KV, a scorer,
    // and the top-k that decides what attention may read.
    let idx = b.tensor(&format!("act.topk.{l}"), c.index_topk as u64 * I32);
    if c.compress_ratio(l) == Some(4) {
        let ih = c.index_heads;
        let ihd = c.index_head_dim;
        let iq = b.tensor(&format!("act.iq.{l}"), (ih * ihd) as u64 * BF16);
        let isc = b.tensor(&format!("act.iscore.{l}"), (ctx / 4) as u64 * F32);
        let ickv = b.tensor(&format!("kv.idx.{l}"), (ctx / 4) as u64 * ihd as u64 * BF16);
        let iw = b.tensor(&format!("act.iw.{l}"), ih as u64 * BF16);
        let g = emit_fp8_gemv(
            b,
            &format!("{p}.attn.indexer.wq_b"),
            iq,
            qlr,
            ih * ihd,
            ql,
            n_cu,
            &[qn],
        );
        let sc = b.emit(DevOp::V4IndexScore, all.clone(), &[g], |d| {
            d.t[0] = isc;
            d.t[1] = iq;
            d.t[2] = ickv;
            d.t[3] = iw;
            d.i[0] = 1;
            d.i[1] = ih;
            d.i[2] = ihd;
            d.i[3] = ctx / 4;
        });
        // ONE block: the selection is a per-row reduction. Under TP the score
        // must be all-reduced BEFORE this packet.
        let tk = b.emit(DevOp::V4IndexTopk, vec![0], &[sc], |d| {
            d.t[0] = idx;
            d.t[1] = isc;
            d.i[0] = 1;
            d.i[1] = ctx / 4;
            d.i[2] = c.index_topk;
            d.i[3] = 0;
            d.i[4] = c.window;
        });
        adeps.push(tk);
    }

    // Split-K attention, then the merge that folds the partials and the sink.
    let topk = c.window
        + if c.compress_ratio(l).is_some() {
            c.index_topk
        } else {
            0
        };
    let sp = 4u32;
    let opart = b.tensor(&format!("act.opart.{l}"), (nh * sp * hd) as u64 * F32);
    let mlpart = b.tensor(&format!("act.mlpart.{l}"), (nh * sp * 2) as u64 * F32);
    let sink = b.tensor(&format!("{p}.attn.attn_sink"), nh as u64 * F32);
    let asp = b.emit(DevOp::V4SparseAttn, all.clone(), &adeps, |d| {
        d.t[0] = opart;
        d.t[1] = tn.q;
        d.t[2] = kvring;
        d.t[3] = idx;
        d.t[4] = mlpart;
        d.i[0] = 1;
        d.i[1] = nh;
        d.i[2] = hd;
        d.i[3] = topk;
        d.f[0] = 1.0 / (hd as f32).sqrt();
    });
    let amg = b.emit(DevOp::V4SparseAttn, all.clone(), &[asp], |d| {
        d.t[0] = tn.attn_out;
        d.t[1] = opart;
        d.t[2] = mlpart;
        d.t[3] = sink;
        d.i[0] = 1;
        d.i[1] = nh;
        d.i[2] = hd;
        d.i[3] = sp;
    });

    // The output de-rotation: the same arm with `sin` negated.
    let inv = b.emit(DevOp::HeadNormRope, all.clone(), &[amg], |d| {
        d.t[0] = tn.attn_out;
        d.t[1] = tn.attn_out;
        d.t[2] = packet::dev::TENSOR_NONE;
        d.t[3] = tn.cos_i;
        d.t[4] = tn.sin_i;
        d.t[5] = tn.pos;
        d.i[0] = 1;
        d.i[1] = nh;
        d.i[2] = hd;
        d.i[4] = 1; // skip_norm: the rescale already happened on q
        d.i[5] = 1;
    });

    // wo_a is BLOCK-DIAGONAL over the head groups, from one stacked tensor; a
    // dense linear of the same element count mixes groups the reference keeps
    // apart. wo_b is an ordinary projection.
    let og = c.o_groups;
    let orank = c.o_lora as u32;
    let ob = b.tensor("act.o_b", (og * orank) as u64 * BF16);
    let woa = b.tensor(
        &format!("{p}.attn.wo_a.weight"),
        (og * orank) as u64 * (nh * hd / og) as u64,
    );
    let gl = b.emit(DevOp::V4GroupedLinear, all, &[inv], |d| {
        d.t[0] = ob;
        d.t[1] = tn.attn_out;
        d.t[2] = woa;
        d.i[0] = 1;
        d.i[1] = og;
        d.i[2] = orank;
        d.i[3] = nh * hd / og;
    });
    let wob = emit_fp8_gemv(
        b,
        &format!("{p}.attn.wo_b"),
        tn.xr,
        ob,
        h,
        og * orank,
        n_cu,
        &[gl],
    );
    emit_hc_expand(b, c, tn, tn.xr, n_cu, &[wob])
}

/// The FFN sub-layer: reduce, norm, route, experts, and the expand back.
///
/// Routing is `sqrt(softplus(.))` with a selection bias that shifts WHICH
/// experts run without reaching their combine weights — and on the first
/// `num_hash_layers` there is no scoring at all, the set comes from a frozen
/// `[vocab, top_k]` table. Both are `V4MoeRoute`; which one is selected by
/// whether `tid2eid` is bound, so a hash layer emitted as a scored one routes
/// every token somewhere else and still runs.
/// The `i[]` slot the MoE weight ENCODING travels in on the decode expert ops
/// (45/46). It is deliberately not `i[3]`: those ops carry `n_exp` there, and
/// writing the encoding into it would set `n_exp = 2`, send every expert id >= 2
/// down the sentinel-skip path, and produce a silently DEAD MoE. Mirrors
/// `mla::MoeEnc::DECODE_SLOT`.
const MOE_ENC_SLOT: usize = 6;

/// The shared expert: fp8, ungated, and it runs for EVERY token.
///
/// Returned as `(output tensor, last packet)` because the combine needs both —
/// the tensor to read and the packet to wait on. Folding the shared expert into
/// the routed sum without gating it on its own producer is a race that shows up
/// as a small, load-dependent numeric drift, which is the hardest kind to find.
fn emit_v4_shared_expert(
    b: &mut Builder,
    c: &V4Cfg,
    tn: &V4Tn,
    l: u32,
    n_cu: u32,
    dep: u32,
) -> (u32, u32) {
    let p = format!("layers.{l}.ffn.shared_experts");
    let all: Vec<u32> = (0..n_cu).collect();
    let h = c.hidden as u32;
    let i = c.moe_inter as u32;
    let sh = b.tensor(&format!("act.shared.{l}"), h as u64 * BF16);
    let sg = b.tensor(&format!("act.shgate.{l}"), i as u64 * BF16);
    let su = b.tensor(&format!("act.shup.{l}"), i as u64 * BF16);
    // Block-fp8 `[N/128][K/128]`, one scale per 128x128 tile.
    let blk = |n: u32, kk: u32| (n as u64).div_ceil(128) * (kk as u64).div_ceil(128);
    let (w1, s1) = (
        b.tensor(&format!("{p}.w1.weight"), i as u64 * h as u64),
        b.tensor(&format!("{p}.w1.scale"), blk(i, h)),
    );
    let (w3, s3) = (
        b.tensor(&format!("{p}.w3.weight"), i as u64 * h as u64),
        b.tensor(&format!("{p}.w3.scale"), blk(i, h)),
    );
    let (w2, s2) = (
        b.tensor(&format!("{p}.w2.weight"), h as u64 * i as u64),
        b.tensor(&format!("{p}.w2.scale"), blk(h, i)),
    );
    #[allow(clippy::too_many_arguments)]
    fn gemv(
        b: &mut Builder,
        cus: &[u32],
        out: u32,
        w: u32,
        s: u32,
        n: u32,
        kk: u32,
        src: u32,
        d0: u32,
    ) -> u32 {
        b.emit(DevOp::GemvFp8Blk, cus.to_vec(), &[d0], |d| {
            d.t[0] = out;
            d.t[1] = src;
            d.t[2] = w;
            // t5, NOT t3. `interp.hip`'s arm reads `(const float*)TEN(5)`; with
            // the scale in t3 the kernel dereferenced an UNSET t5 — a null
            // pointer straight into an aperture violation on the first decode
            // step. `emit_fp8_gemv` had it right and this one did not, which is
            // exactly why the slot is now written the same way in both.
            d.t[5] = s;
            d.i[0] = 1;
            d.i[1] = n;
            d.i[2] = kk;
        })
    }
    let g = gemv(b, &all, sg, w1, s1, i, h, tn.xn, dep);
    let u = gemv(b, &all, su, w3, s3, i, h, tn.xn, dep);
    let a = b.emit(DevOp::V4ClampedSwiGlu, all.clone(), &[g, u], |d| {
        d.t[0] = sg;
        d.t[1] = sg;
        d.t[2] = su;
        d.i[0] = i;
        d.f[0] = c.swiglu_limit as f32;
    });
    let dn = gemv(b, &all, sh, w2, s2, h, i, sg, a);
    (sh, dn)
}

#[allow(clippy::too_many_arguments)]
fn emit_v4_ffn(b: &mut Builder, c: &V4Cfg, tn: &V4Tn, l: u32, n_cu: u32, deps: &[u32]) -> u32 {
    let p = format!("layers.{l}");
    let all: Vec<u32> = (0..n_cu).collect();
    let h = c.hidden as u32;
    let e = c.n_exp;
    let k = c.top_k;
    let imoe = c.moe_inter as u32;
    let hash = l < c.hash_layers;

    let red = emit_hc_reduce(b, c, tn, &format!("{p}.hc_ffn"), tn.xr, n_cu, deps);
    let fnw = b.tensor(&format!("{p}.ffn_norm.weight"), h as u64 * BF16);
    let nrm = b.emit(DevOp::RmsNorm, all.clone(), &[red], |d| {
        d.t[0] = tn.xn;
        d.t[1] = tn.xr;
        d.t[2] = fnw;
        d.i[0] = 1;
        d.i[1] = h;
        d.f[0] = 1e-6;
    });

    // Gate scores. The GEMM is bf16 and small; on a hash layer it is dead
    // weight the emitter still pays, which is a Stage-4 saving, not a bug.
    let glog = b.tensor(&format!("act.glogit.{l}"), e as u64 * F32);
    let gw = b.tensor(&format!("{p}.ffn.gate.weight"), e as u64 * h as u64 * BF16);
    let gg = b.emit(DevOp::Gemv, all.clone(), &[nrm], |d| {
        d.t[0] = glog;
        d.t[1] = tn.xn;
        d.t[2] = gw;
        d.i[0] = 1;
        d.i[1] = e;
        d.i[2] = h;
    });

    let sel = b.tensor(&format!("act.sel.{l}"), k as u64 * I32);
    let wts = b.tensor(&format!("act.wts.{l}"), k as u64 * F32);
    let bias = if hash {
        packet::dev::TENSOR_NONE
    } else {
        b.tensor(&format!("{p}.ffn.gate.bias"), e as u64 * F32)
    };
    let tid = if hash {
        b.tensor(
            &format!("{p}.ffn.gate.tid2eid"),
            c.vocab as u64 * k as u64 * 8,
        )
    } else {
        packet::dev::TENSOR_NONE
    };
    // `tab` is the selection in the layout ops 45/46 read — `[k]` of
    // `{u32 expert_id, f32 gate}` (`d_moe_router`, runtime/amd/op_moe.h). V4
    // SCORES differently from every other family, but what it hands the experts
    // is the ordinary table, which is why the shipped mxfp4 expert arms can
    // stream its experts with no V4-specific kernel at all.
    let tab = b.tensor(&format!("act.moetab.{l}"), k as u64 * 8);
    let rt = b.emit(DevOp::V4MoeRoute, vec![0], &[gg], |d| {
        d.t[0] = sel;
        d.t[1] = wts;
        d.t[2] = glog;
        d.t[3] = bias;
        d.t[4] = tid;
        d.t[5] = tn.ids;
        d.t[6] = tab;
        d.i[0] = 1;
        d.i[1] = e;
        d.i[2] = k;
        d.f[0] = c.route_scale as f32;
    });

    // Routed experts, fp4, through the existing mxfp4 expert arms; the shared
    // expert is fp8 and always runs.
    // `fu` is PER SLOT: the GLU arm writes `fu[slot * I_moe + n]` and the down
    // arm reads the same stride, so this is k rows of I_moe, not a gate|up pair.
    let fu = b.tensor(&format!("act.fu.{l}"), k as u64 * imoe as u64 * BF16);
    // k independent f32 partials of H, folded by the combine.
    let part = b.tensor(&format!("act.part.{l}"), k as u64 * h as u64 * F32);
    let (shared, shdep) = emit_v4_shared_expert(b, c, tn, l, n_cu, nrm);
    // HOST-FILLED POINTER TABLES, not checkpoint tensors: they hold device
    // addresses the loader computes after packing the 256 experts. The names
    // must end in `mlp.expert_weight_table` / `mlp.expert_scale_table` because
    // `packet::names::is_host_filled_table` SUFFIX-matches those exactly — the
    // prefix is ours to choose, the suffix is not. Named anything else they
    // classify as checkpoint weights and the load dies looking for a tensor no
    // checkpoint contains.
    // [E][3] u64 device addresses — gate, up, down per expert, in that slot
    // order. One handle per expert is not enough: the arms index
    // `expert_weight_table[eid * 3 + j]`.
    let tbl = 3 * 8 * e as u64;
    let ewt = b.tensor(&format!("layers.{l}.mlp.expert_weight_table"), tbl);
    let est = b.tensor(&format!("layers.{l}.mlp.expert_scale_table"), tbl);
    // ONE PACKET PAIR PER SLOT, and the slot index is what makes them different
    // work: `i[0]` tells the arm which of the k table entries to stream. A
    // single pair with `i[0]` left at zero runs the top-1 expert k times over
    // and drops the other five — arithmetic that completes, on a sixth of the
    // model.
    //
    // `i[6]` is the ENCODING (`MoeEnc::DECODE_SLOT`), and it is not optional
    // here: V4's routed experts are MXFP4, an unset `i[6]` reads as
    // `PLOW_MOE_ENC_BF16`, and the quantized dot answers an encoding it does not
    // implement with a NaN rather than a wrong number.
    let enc = 2u32; // PLOW_MOE_ENC_MXFP4
    let v4_clamp = 3u32; // PLOW_MOE_ACT_V4CLAMP
    let mut downs: Vec<u32> = Vec::with_capacity(k as usize);
    for sl in 0..k {
        let glu = b.emit(DevOp::MoeExpertGluFp8Blk, all.clone(), &[rt], |d| {
            d.t[0] = fu;
            d.t[1] = tn.xn;
            d.t[2] = tab;
            d.t[3] = ewt;
            d.t[4] = est;
            d.i[0] = sl;
            d.i[1] = imoe;
            d.i[2] = h;
            d.i[3] = e;
            // V4's activation is a CLAMPED SwiGLU (limit 10): one-sided on the
            // gate, two-sided on the up branch. It rides the GLU epilogue as an
            // activation code rather than a following packet because the arm
            // has `g` and `u` in registers there — a separate packet would
            // round `fu` to bf16, read it back, and clamp a number the clamp
            // was supposed to bound before the rounding.
            d.i[5] = v4_clamp;
            d.i[MOE_ENC_SLOT] = enc;
            d.f[0] = c.swiglu_limit as f32;
        });
        let down = b.emit(DevOp::MoeExpertDownFp8Blk, all.clone(), &[glu], |d| {
            d.t[0] = part;
            d.t[1] = fu;
            d.t[2] = tab;
            d.t[3] = ewt;
            d.t[4] = est;
            d.i[0] = sl;
            d.i[1] = h;
            d.i[2] = imoe;
            d.i[3] = e;
            d.i[MOE_ENC_SLOT] = enc;
        });
        downs.push(down);
    }
    // The k slots wrote k INDEPENDENT f32 partials; the combine folds them onto
    // the stream. V4 has no residual add, so the residual operand is `xr`
    // itself — the hyper-connection reduce's output, which is what the expert
    // sum is added to.
    downs.push(shdep);
    let cmb = b.emit(DevOp::MoeCombine, all, &downs, |d| {
        d.t[0] = tn.xr;
        d.t[1] = tn.xr;
        d.t[2] = shared;
        d.t[3] = part;
        d.i[0] = h;
        d.i[1] = k;
    });

    emit_hc_expand(b, c, tn, tn.xr, n_cu, &[cmb])
}

/// One whole V4 layer.
pub(crate) fn emit_v4_layer(
    b: &mut Builder,
    c: &V4Cfg,
    tn: &V4Tn,
    l: u32,
    ctx: u32,
    n_cu: u32,
    deps: &[u32],
) -> u32 {
    // DEBUG BISECT, same argument as PLOW_V4_LAYERS: a GPU fault names no
    // packet, so the halves are emitted separately to find out which one it is.
    // `attn` and `ffn` each still produce a well-formed program — neither is the
    // model, and both say so.
    let half = std::env::var("PLOW_V4_HALF").unwrap_or_default();
    if half == "ffn" {
        return emit_v4_ffn(b, c, tn, l, n_cu, deps);
    }
    let a = emit_v4_attn(b, c, tn, l, ctx, n_cu, deps);
    if half == "attn" {
        return a;
    }
    emit_v4_ffn(b, c, tn, l, n_cu, &[a])
}

/// The whole decode program for one token: embed, every layer, the final
/// hyper-connection reduce, the norm, the lm_head and the argmax.
///
/// The head reduce is a DIFFERENT formula from the per-layer one and not a
/// special case of it: `[hc, hc*D]` weights, a scalar scale, a sigmoid gate,
/// and no Sinkhorn — so it is `V4HcReduceHead`, not `V4HcDot` + `V4HcMix` with
/// the iteration count set to zero.
pub(crate) fn emit_v4_decode(b: &mut Builder, c: &V4Cfg, tn: &V4Tn, ctx: u32, n_cu: u32) {
    let all: Vec<u32> = (0..n_cu).collect();
    let h = c.hidden as u32;

    let emb = b.emit(DevOp::Embed, all.clone(), &[], |d| {
        d.t[0] = tn.x;
        d.t[1] = tn.embed;
        d.t[2] = tn.ids;
        d.i[0] = 1;
        d.i[1] = h;
        d.f[0] = 1.0;
    });

    // `Embed` wrote ONE [D] vector; the residual state is [hc, D]. Without this
    // the first reduce in layer 0 mixes uninitialised memory into every layer
    // after it — which is what a 43-layer program returning byte-identical
    // output to a 0-layer one looks like from the outside.
    let bc = b.emit(DevOp::V4HcBroadcast, all.clone(), &[emb], |d| {
        d.t[0] = tn.x;
        d.t[1] = tn.x;
        d.i[0] = 1;
        d.i[1] = h;
        d.i[2] = c.hc_mult;
    });
    let mut dep = bc;
    for l in 0..c.layers {
        dep = emit_v4_layer(b, c, tn, l, ctx, n_cu, &[dep]);
    }

    // The final reduce collapses the streams for the head.
    let hfn = b.tensor(
        "hc_head_fn",
        c.hc_mult as u64 * c.hc_mult as u64 * h as u64 * F32,
    );
    let hsc = b.tensor("hc_head_scale", F32);
    let hba = b.tensor("hc_head_base", c.hc_mult as u64 * F32);
    let hr = b.emit(DevOp::V4HcReduceHead, all.clone(), &[dep], |d| {
        d.t[0] = tn.xr;
        d.t[1] = tn.x;
        d.t[2] = hfn;
        d.t[3] = hsc;
        d.t[4] = hba;
        d.i[0] = 1;
        d.i[1] = h;
        d.i[2] = c.hc_mult;
        d.f[0] = 1e-6;
        d.f[1] = 1e-6;
    });
    let fnrm = b.emit(DevOp::RmsNorm, all.clone(), &[hr], |d| {
        d.t[0] = tn.xn;
        d.t[1] = tn.xr;
        d.t[2] = tn.fin_norm;
        d.i[0] = 1;
        d.i[1] = h;
        d.f[0] = 1e-6;
    });
    // lm_head is bf16 and replicated; at 129280 x 4096 it is 9.3% of a decode
    // step's bytes on its own (perf-data/tools/v4_decode_budget.py).
    let lm = b.emit(DevOp::Gemv, all.clone(), &[fnrm], |d| {
        d.t[0] = tn.logits;
        d.t[1] = tn.xn;
        d.t[2] = tn.head;
        d.i[0] = 1;
        d.i[1] = c.vocab as u32;
        d.i[2] = h;
    });
    let am = b.emit(DevOp::Argmax, all, &[lm], |d| {
        d.t[0] = tn.amax;
        d.t[1] = tn.logits;
        d.i[0] = c.vocab as u32;
        d.i[1] = 1;
    });
    b.emit(DevOp::ArgmaxFin, vec![0], &[am], |d| {
        d.t[0] = tn.ids;
        d.t[1] = tn.amax;
        d.i[0] = n_cu;
        d.i[1] = 1;
    });
}

/// Emit a V4 decode blob. Behind `PLOW_V4_FULL=1`; the default emit stays the
/// capability report, because a blob that cannot LOAD is worse than a refusal
/// that says why.
///
/// # The load contract, and why this is still gated
///
/// Every weight above is declared under the checkpoint's own name — including
/// the `.scale` companion of each block-fp8 tensor. `plowrt` binds by that
/// name, so the emit side of the contract is complete and checkable: the blob
/// says exactly which tensors it wants. What is NOT yet done is the host side
/// for V4's two quantized layouts — block-fp8 `[N/128][K/128]` scale grids and
/// the fp4 expert packing — plus the routed-expert weight/scale TABLES, which
/// are one handle per layer standing for 256 experts and need the loader to
/// build the table rather than bind a tensor.
///
/// So this writes a structurally complete program whose weights do not all
/// resolve yet. That is a deliberate intermediate: the packet DAG, the counts
/// and the ordering are all testable without a single byte of the 167 GB
/// checkpoint, and they are tested (`the_shipped_43_layer_program_builds`).
pub(crate) fn v4_emit_full(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, target: u32) {
    assert_eq!(
        tp, 1,
        "V4 TP is not wired: the indexer's score must be all-reduced BEFORE its top-k, so the \
         selection is a collective. Emitting per-rank selections would let ranks decode different \
         tokens — refused rather than approximated."
    );
    let mut c = cfg_deepseek_v4(dir);
    // DEBUG BISECT. A decode program that faults on the GPU gives no packet
    // index back — the AMD dispatch is a persistent counter-DAG interpreter, so
    // "which op" is not in the error. Truncating the layer count and re-running
    // is the probe that answers it, and it is here rather than in a scratch
    // branch because the alternative is guessing (which has already cost this
    // bring-up five wrong hypotheses).
    if let Some(n) = std::env::var("PLOW_V4_LAYERS").ok().and_then(|v| v.parse::<u32>().ok()) {
        c.layers = c.layers.min(n);
        eprintln!("deepseek_v4: PLOW_V4_LAYERS={n} — TRUNCATED, not the real model");
    }
    let mut b = Builder::new(n_cu);
    let tn = declare_v4(&mut b, &c, ctx);
    emit_v4_decode(&mut b, &c, &tn, ctx, n_cu);
    let prog = b.finish();

    eprintln!(
        "deepseek_v4: decode program built — {} packets, {} tensors, {} layers, ctx {ctx}",
        prog.insts.len(),
        b_tensor_count(&prog),
        c.layers
    );
    let m = Model {
        n_cu,
        target,
        tensors: prog.tensors.clone(),
        progs: vec![prog],
        kv_row_insts: Vec::new(),
        prog_t: vec![1],
        gen: Vec::new(),
    };
    let blob = m.to_blob();
    std::fs::write(out, &blob).expect("write blob");
    eprintln!("  wrote {} ({} bytes)", out, blob.len());
    eprintln!(
        "  the host-side bind of the two quantized layouts (block-fp8 scale grids, fp4 packing) \
         is what decides whether this LOADS; every weight is declared under the checkpoint's own \
         name, so a failure names the first tensor it cannot find."
    );
}

fn b_tensor_count(p: &packet::devbuild::Program) -> usize {
    p.tensors.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mla::deepseek_v4::cfg_deepseek_v4_for_test;

    /// The chain a V4 layer's attention side emits, in order. Asserted as a
    /// SEQUENCE because the order is the model: a hyper-connection expand that
    /// ran before its branch, or a reduce whose partial was not zeroed by the
    /// previous step's mix, is a program that runs and is wrong.
    #[test]
    fn hc_reduce_emits_dot_then_mix_sharing_one_partial() {
        let c = cfg_deepseek_v4_for_test();
        let mut b = Builder::new(8);
        let tn = declare_v4(&mut b, &c, 4096);
        let out = tn.xr;
        emit_hc_reduce(&mut b, &c, &tn, "layers.0.hc_attn", out, 8, &[]);
        let p = b.finish();
        let ops: Vec<u16> = p.insts.iter().map(|i| i.op).collect();
        assert_eq!(
            ops,
            vec![
                DevOp::V4HcZero as u16,
                DevOp::V4HcDot as u16,
                DevOp::V4HcMix as u16
            ],
            "zero, projection, Sinkhorn tail — the zero is not optional, the dot \
             accumulates with atomics"
        );
        // All three must name the SAME partial: the zero clears what the dot
        // accumulates into and the mix reads.
        assert_eq!(p.insts[0].t[0], p.insts[1].t[0], "zero clears the dot's target");
        assert_eq!(p.insts[1].t[0], p.insts[2].t[2], "the mix reads what the dot wrote");
    }

    /// The attention chain, as a SEQUENCE. Order is the model here: a
    /// de-rotation before the merge, or `wo_a` fed the raw heads instead of the
    /// de-rotated ones, is a program that runs and is a different network.
    #[test]
    fn attention_chain_is_emitted_in_reference_order() {
        let c = cfg_deepseek_v4_for_test();
        let mut b = Builder::new(8);
        let tn = declare_v4(&mut b, &c, 16384);
        // Layer 2: ratio 4, so it carries BOTH a compressor and an indexer.
        emit_v4_attn(&mut b, &c, &tn, 2, 16384, 8, &[]);
        let p = b.finish();
        let ops: Vec<DevOp> = p
            .insts
            .iter()
            .map(|i| DevOp::from_u16(i.op).expect("known opcode"))
            .collect();
        use DevOp::*;
        assert_eq!(
            ops,
            vec![
                V4HcZero,
                V4HcDot,
                V4HcMix, // reduce the streams
                RmsNorm, // attn_norm
                GemvFp8Blk,
                RmsNorm,
                GemvFp8Blk,
                HeadNormRope, // q path
                GemvFp8Blk,
                HeadNormRope, // kv + ring write
                // BF16, not block-fp8. This pair read `GemvFp8Blk` until the
                // load proved otherwise — `compressor.wkv.weight` is 8388608 B
                // on disk against the 4194304 B the fp8 path declared — so the
                // sequence was pinning a defect rather than the model.
                Gemv,
                Gemv,
                V4KvCompress, // compressor
                GemvFp8Blk,
                V4IndexScore,
                V4IndexTopk, // indexer
                V4SparseAttn,
                V4SparseAttn, // split + merge
                HeadNormRope, // INVERSE rope
                V4GroupedLinear,
                GemvFp8Blk, // wo_a (block-diag), wo_b
                V4HcExpand,
            ]
        );
    }

    /// A sliding-window-only layer emits neither compressor nor indexer, and a
    /// ratio-128 layer emits the compressor but NOT the indexer. Getting that
    /// split wrong is invisible in a shape check.
    #[test]
    fn only_ratio_four_layers_carry_an_indexer() {
        let c = cfg_deepseek_v4_for_test();
        for (l, want_comp, want_idx) in [(0u32, false, false), (2, true, true), (3, true, false)] {
            let mut b = Builder::new(8);
            let tn = declare_v4(&mut b, &c, 16384);
            emit_v4_attn(&mut b, &c, &tn, l, 16384, 8, &[]);
            let p = b.finish();
            let has = |op: DevOp| p.insts.iter().any(|i| i.op == op as u16);
            assert_eq!(has(DevOp::V4KvCompress), want_comp, "layer {l} compressor");
            assert_eq!(has(DevOp::V4IndexScore), want_idx, "layer {l} indexer");
        }
    }

    /// Hash-routed layers bind `tid2eid` and NO selection bias; scored layers
    /// are the other way round. The router op is the same either way, so this
    /// is the only place the split is visible before it reaches the device.
    #[test]
    fn hash_layers_bind_the_table_and_scored_layers_bind_the_bias() {
        let c = cfg_deepseek_v4_for_test();
        for (l, hash) in [(0u32, true), (2, true), (3, false)] {
            let mut b = Builder::new(8);
            let tn = declare_v4(&mut b, &c, 16384);
            emit_v4_ffn(&mut b, &c, &tn, l, 8, &[]);
            let p = b.finish();
            let r = p
                .insts
                .iter()
                .find(|i| i.op == DevOp::V4MoeRoute as u16)
                .expect("a router");
            // The BUILDER sentinel (0xFFFF_FFFF), not the packed one (0xFFFF):
            // comparing against TENSOR_NONE16 here reports every absent operand
            // as bound.
            let none = packet::dev::TENSOR_NONE;
            assert_eq!(r.t[4] != none, hash, "layer {l} tid2eid bound?");
            assert_eq!(r.t[3] != none, !hash, "layer {l} selection bias bound?");
        }
    }

    /// A whole layer, both sub-layers, with the expand appearing exactly twice
    /// — once per sub-layer boundary. V4 has no residual add, so the count of
    /// expands IS the count of residual writes.
    #[test]
    fn a_layer_has_exactly_two_hyper_connection_boundaries() {
        let c = cfg_deepseek_v4_for_test();
        let mut b = Builder::new(8);
        let tn = declare_v4(&mut b, &c, 16384);
        emit_v4_layer(&mut b, &c, &tn, 2, 16384, 8, &[]);
        let p = b.finish();
        let n = |op: DevOp| p.insts.iter().filter(|i| i.op == op as u16).count();
        assert_eq!(n(DevOp::V4HcExpand), 2);
        assert_eq!(n(DevOp::V4HcDot), 2);
        assert_eq!(n(DevOp::V4HcMix), 2);
        assert_eq!(
            p.insts
                .iter()
                .filter(|i| i.op == DevOp::Residual as u16)
                .count(),
            0,
            "V4 has no residual add anywhere"
        );
    }

    /// The whole decode program: one embed, every layer's boundaries, and a
    /// tail that ends in a token id. A program that does not end in ArgmaxFin
    /// cannot advance the sequence, and one whose head reduce is the per-layer
    /// op is a different function at the last step.
    #[test]
    fn decode_program_spans_embed_to_token() {
        let c = cfg_deepseek_v4_for_test();
        let mut b = Builder::new(8);
        let tn = declare_v4(&mut b, &c, 16384);
        emit_v4_decode(&mut b, &c, &tn, 16384, 8);
        let p = b.finish();
        let n = |op: DevOp| p.insts.iter().filter(|i| i.op == op as u16).count();
        assert_eq!(n(DevOp::Embed), 1);
        assert_eq!(n(DevOp::V4HcExpand), 2 * c.layers as usize);
        assert_eq!(n(DevOp::V4HcReduceHead), 1, "the head reduce is its OWN op");
        assert_eq!(n(DevOp::ArgmaxFin), 1);
        assert_eq!(n(DevOp::Residual), 0, "V4 has no residual add");
        let last = p.insts.last().expect("a program");
        assert_eq!(last.op, DevOp::ArgmaxFin as u16);
        assert_eq!(last.t[0] as u32, tn.ids, "the tail writes in.ids");
    }

    /// The REAL geometry: 43 layers, 21 of them indexed, at 16k. This is the
    /// program `plowrt serve` would run at batch 1, and the packet count is
    /// what the launch-floor argument in the decode-step harness was about —
    /// under the interpreter these are counter-gated packets, not launches.
    #[test]
    fn the_shipped_43_layer_program_builds() {
        let mut c = cfg_deepseek_v4_for_test();
        c.layers = 43;
        c.compress_ratios = (0..46)
            .map(|i| match i {
                0 | 1 | 43 | 44 | 45 => 0,
                n if n % 2 == 0 => 4,
                _ => 128,
            })
            .collect();
        let mut b = Builder::new(304);
        let tn = declare_v4(&mut b, &c, 16384);
        emit_v4_decode(&mut b, &c, &tn, 16384, 304);
        let p = b.finish();
        let n = |op: DevOp| p.insts.iter().filter(|i| i.op == op as u16).count();
        assert_eq!(n(DevOp::V4HcExpand), 86, "two boundaries x 43 layers");
        assert_eq!(n(DevOp::V4SparseAttn), 86, "split + merge per layer");
        assert_eq!(n(DevOp::V4IndexScore), 21, "the ratio-4 layers");
        assert_eq!(n(DevOp::V4KvCompress), 41, "every compressed layer");
        assert_eq!(n(DevOp::V4MoeRoute), 43);
        assert!(
            p.insts.len() > 500,
            "a 43-layer V4 step is many packets: {}",
            p.insts.len()
        );
        eprintln!("V4 decode program: {} packets", p.insts.len());
    }

    /// EVERY declared tensor must classify correctly for the loader, and the
    /// checkpoint-weight ones must match the checkpoint's real names and sizes.
    ///
    /// `plowrt` binds by EXACT NAME with a byte-size check
    /// (`exec/gpu.rs`: `ckpt.tensor(&td.name)` then `src.len() != td.bytes`),
    /// so a name this emitter spells wrong is a load failure and a size it
    /// computes wrong is a load failure — both loud, but only if someone runs
    /// the 167 GB load. This test is that check without the checkpoint: it
    /// asserts the SPLIT (runtime scratch vs checkpoint weight) and, for a
    /// sample of real tensors, the exact byte counts the safetensors headers
    /// carry.
    #[test]
    fn declared_names_classify_and_size_as_the_loader_expects() {
        use packet::names::{is_checkpoint_weight, is_runtime_tensor};
        let mut c = cfg_deepseek_v4_for_test();
        c.layers = 43;
        c.compress_ratios = (0..46)
            .map(|i| match i {
                0 | 1 | 43 | 44 | 45 => 0,
                n if n % 2 == 0 => 4,
                _ => 128,
            })
            .collect();
        let mut b = Builder::new(304);
        let tn = declare_v4(&mut b, &c, 16384);
        emit_v4_decode(&mut b, &c, &tn, 16384, 304);
        let p = b.finish();

        // Anything the model does not ship must be runtime-classified, or the
        // loader will go looking for it in the checkpoint and fail.
        for t in &p.tensors {
            let n = &t.name;
            let scratch = n.starts_with("act.") || n.starts_with("in.") || n.starts_with("kv.");
            if scratch {
                assert!(
                    is_runtime_tensor(n),
                    "`{n}` is scratch but the loader would look for it in the checkpoint"
                );
            }
        }

        // The real geometry, from the released checkpoint's headers.
        let want: &[(&str, u64)] = &[
            ("embed.weight", 129280 * 4096 * 2),
            ("head.weight", 129280 * 4096 * 2),
            ("norm.weight", 4096 * 2),
            ("hc_head_fn", 4 * 4 * 4096 * 4),
            ("layers.0.hc_attn_fn", 24 * 4 * 4096 * 4),
            ("layers.0.hc_ffn_base", 24 * 4),
            ("layers.0.attn.wq_a.weight", 1024 * 4096),
            ("layers.0.attn.wq_b.weight", 32768 * 1024),
            ("layers.0.attn.wkv.weight", 512 * 4096),
            ("layers.0.attn.wo_b.weight", 4096 * 8192),
            ("layers.0.attn.attn_sink", 64 * 4),
            ("layers.0.attn_norm.weight", 4096 * 2),
            ("layers.2.attn.compressor.ape", 4 * 1024 * 4),
            ("layers.2.attn.indexer.wq_b.weight", 8192 * 1024),
        ];
        // The expert pointer tables must classify as HOST-FILLED. Named
        // anything but the `mlp.expert_*_table` suffix they look like weights,
        // and the load dies hunting a tensor no checkpoint contains.
        for l in 0..c.layers {
            for suf in ["mlp.expert_weight_table", "mlp.expert_scale_table"] {
                let n = format!("layers.{l}.{suf}");
                let t = p.tensors.iter().find(|t| t.name == n);
                assert!(t.is_some(), "emitter never declared `{n}`");
                assert!(
                    packet::names::is_host_filled_table(&n),
                    "`{n}` must be host-filled, not looked up in the checkpoint"
                );
                assert!(!is_checkpoint_weight(&n));
            }
        }

        for (n, bytes) in want {
            let t = p
                .tensors
                .iter()
                .find(|t| &t.name == n)
                .unwrap_or_else(|| panic!("emitter never declared `{n}`"));
            assert!(
                is_checkpoint_weight(n),
                "`{n}` must bind from the checkpoint"
            );
            assert_eq!(t.bytes, *bytes, "`{n}` byte count the loader will check");
        }
    }

    /// Every routed-expert packet must name its SLOT and its ENCODING.
    ///
    /// Both defects this pins were live and neither was visible: the emitter
    /// sent ONE gate/up + down pair with `i[0]` unset, which runs the top-1
    /// expert and silently drops the other five, and it left `i[6]` at zero,
    /// which the quantized dot reads as bf16 and answers with a NaN. Packet
    /// counts alone cannot see either — the program built, the ops were right,
    /// the immediates were wrong — so this asserts the immediates.
    #[test]
    fn routed_experts_carry_their_slot_and_the_mxfp4_encoding() {
        let c = cfg_deepseek_v4_for_test();
        let mut b = Builder::new(8);
        let tn = declare_v4(&mut b, &c, 16384);
        emit_v4_decode(&mut b, &c, &tn, 16384, 8);
        let p = b.finish();
        let k = c.top_k;

        for (op, what) in [
            (DevOp::MoeExpertGluFp8Blk, "gate/up"),
            (DevOp::MoeExpertDownFp8Blk, "down"),
        ] {
            let ins: Vec<_> = p.insts.iter().filter(|i| i.op == op as u16).collect();
            assert_eq!(
                ins.len() as u32,
                k * c.layers,
                "{what}: one packet per slot per layer"
            );
            // The k slots of one layer must be 0..k, not k copies of slot 0.
            let mut slots: Vec<u32> = ins[..k as usize].iter().map(|i| i.i[0]).collect();
            slots.sort_unstable();
            assert_eq!(
                slots,
                (0..k).collect::<Vec<_>>(),
                "{what}: layer 0 must stream every routed slot exactly once"
            );
            for i in &ins {
                assert_eq!(
                    i.i[MOE_ENC_SLOT], 2,
                    "{what}: V4 experts are MXFP4; encoding 0 decodes as bf16 and poisons"
                );
                assert_eq!(i.i[3], c.n_exp, "{what}: n_exp must stay in i[3]");
            }
        }
        // The gate/up arm owns the clamped SwiGLU, so the clamp limit must ride
        // with it — a zero limit would clamp every activation to <= 0.
        let glu = p
            .insts
            .iter()
            .find(|i| i.op == DevOp::MoeExpertGluFp8Blk as u16)
            .expect("a routed gate/up packet");
        assert_eq!(glu.i[5], 3, "PLOW_MOE_ACT_V4CLAMP");
        assert_eq!(glu.f[0], c.swiglu_limit as f32);

        // One combine per layer folds the k partials plus the shared expert.
        let cmb: Vec<_> = p
            .insts
            .iter()
            .filter(|i| i.op == DevOp::MoeCombine as u16)
            .collect();
        assert_eq!(cmb.len() as u32, c.layers, "one combine per layer");
        assert_eq!(cmb[0].i[1], k, "the combine must fold ALL k partials");
    }

    /// EVERY block-fp8 GEMV must carry its scale in t5.
    ///
    /// `interp.hip` reads `(const float*)TEN(5)` on op 44. A scale written to
    /// any other slot leaves t5 UNSET, and the kernel then dereferences a null
    /// pointer — which is not a wrong number but an
    /// HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION on the first decode step.
    /// That is exactly what the shared expert did: it declared the right
    /// tensor, bound it, and put it one slot away from where it is read.
    #[test]
    fn every_block_fp8_gemv_carries_its_scale_in_t5() {
        let c = cfg_deepseek_v4_for_test();
        let mut b = Builder::new(8);
        let tn = declare_v4(&mut b, &c, 16384);
        emit_v4_decode(&mut b, &c, &tn, 16384, 8);
        let p = b.finish();
        let n = packet::dev::TENSOR_NONE;
        let mut seen = 0;
        for i in p.insts.iter().filter(|i| i.op == DevOp::GemvFp8Blk as u16) {
            seen += 1;
            assert!(
                i.t[5] != n,
                "a block-fp8 GEMV with no scale in t5 faults rather than mis-computes"
            );
        }
        // Per layer: wq_a, wq_b, wkv, wo_b, gate, and the shared expert's three.
        assert!(
            seen >= c.layers as usize * 5,
            "expected at least 5 block-fp8 GEMVs per layer, saw {seen} over {} layers",
            c.layers
        );
    }

    /// The expand reads `x` and writes `x`. That aliasing is deliberate and
    /// safe — every output element reads the whole `[hc]` vector at its own
    /// depth before storing any of it — so a future change that "fixes" it by
    /// introducing a copy is a regression, not a repair.
    #[test]
    fn hc_expand_is_in_place_over_the_streams() {
        let c = cfg_deepseek_v4_for_test();
        let mut b = Builder::new(8);
        let tn = declare_v4(&mut b, &c, 4096);
        emit_hc_expand(&mut b, &c, &tn, tn.attn_out, 8, &[]);
        let p = b.finish();
        assert_eq!(
            p.insts[0].t[0], p.insts[0].t[2],
            "expand is in place over x"
        );
    }
}
