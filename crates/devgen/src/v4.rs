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

use crate::mla::deepseek_v4::V4Cfg;
use packet::dev::DevOp;
use packet::devbuild::Builder;

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

    let dot = b.emit(DevOp::V4HcDot, all.clone(), deps, |d| {
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
        let g1 = emit_fp8_gemv(
            b,
            &format!("{p}.attn.compressor.wkv"),
            ckv,
            tn.xn,
            w,
            h,
            n_cu,
            &[nrm],
        );
        let g2 = emit_fp8_gemv(
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
        // Window state is a RUNTIME resource, like the KV ring: it persists
        // across steps and is not produced by any packet.
        let kvs = b.tensor(
            &format!("state.ckv.{l}"),
            (coff * r) as u64 * w as u64 * F32,
        );
        let scs = b.tensor(
            &format!("state.csc.{l}"),
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
            vec![DevOp::V4HcDot as u16, DevOp::V4HcMix as u16],
            "the reduce is two packets, projection then Sinkhorn tail"
        );
        // Both must name the SAME partial: the mix reads what the dot wrote.
        assert_eq!(p.insts[0].t[0], p.insts[1].t[2]);
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
                V4HcDot,
                V4HcMix, // reduce the streams
                RmsNorm, // attn_norm
                GemvFp8Blk,
                RmsNorm,
                GemvFp8Blk,
                HeadNormRope, // q path
                GemvFp8Blk,
                HeadNormRope, // kv + ring write
                GemvFp8Blk,
                GemvFp8Blk,
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
