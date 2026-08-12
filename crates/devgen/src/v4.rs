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
