//! `tinygemma` — compile a small but REAL Gemma-4-shaped prefill network into a device
//! packet program, and write the blob the runtime executes.
//!
//! # DEPRECATED
//!
//! Only built with `--features legacy-gemma-bins`; slated for removal along
//! with the `gemma4` binary. The C harness below is its last consumer.
//!
//! This exists because plow is a NETWORK compiler. Benchmarking one GEMM through the
//! interpreter measures almost nothing that matters: the interesting question is which
//! packets can be in flight at once, on which CUs, and where the machine goes idle
//! waiting on a counter. So we compile a whole block — embed, two decoder layers
//! (one sliding-window, one full-causal, as Gemma mixes them), final norm, tied
//! lm_head, softcap — and let the trace answer it.
//!
//! The numerics are Gemma 4's, and every one of these is a silent
//! fluent-but-wrong-output bug if you get it wrong:
//!   * RMSNorm has NO `+1` on the weight (Gemma 1/2/3 did; Gemma 4 does not) and eps
//!     is INSIDE the rsqrt.
//!   * attention scale is 1.0 — there is NO 1/sqrt(head_dim).
//!   * `v_norm` is a WEIGHTLESS RMSNorm over head_dim applied to V on every layer. It
//!     has no checkpoint tensor, so it is the easiest part of the model to omit.
//!   * MLP is GeGLU (gelu_tanh), not SwiGLU.
//!   * sandwich norms: the residual is added AFTER the post-norm.
//!   * the second residual is scaled by `layer_scalar`.
//!   * the embedding is scaled by the BF16-ROUNDED sqrt(hidden).
//!
//! The C harness (`runtime/tests/net_gemma_block_test.c`) generates the weights from a
//! fixed seed, runs this program on the interpreter, and checks it against an
//! independent fp32 CPU reference. Truth is the CPU reference.

use packet::dev::DevOp;
use packet::devbuild::Builder;

// ---- network shape. Small enough for an exact fp32 CPU reference, big enough that
// ---- the GEMMs actually tile and the schedule has something to schedule.
const T: u32 = 512; // prefill tokens
const H: u32 = 1024; // hidden
const I: u32 = 2048; // intermediate (GeGLU: gate and up are each I wide)
const N_HEAD: u32 = 4;
const HD: u32 = 256; // head_dim — the flash kernels are specialised on 256 or 512
const N_KV_HEAD: u32 = 2; // GQA 2:1
const LAYERS: u32 = 2;
const VOCAB: u32 = 1024;
const EPS: f32 = 1e-6;
const SOFTCAP: f32 = 30.0;
const LAYER_SCALAR: f32 = 0.75;
/// Layer 0 slides, layer 1 is full-causal — Gemma interleaves them, and the two take
/// different paths through flash, so a one-layer net would not exercise both.
const WINDOW: [u32; 2] = [256, 0]; // 0 == full causal

const QDIM: u32 = N_HEAD * HD;
const KVDIM: u32 = N_KV_HEAD * HD;
const BM: u32 = 256; // must match GM_BM / GM_BN in op_gemm.h
const BN: u32 = 256;

const BF16: u64 = 2;
const F32: u64 = 4;
const I32: u64 = 4;

fn gemm_tiles(m: u32, n: u32) -> u32 {
    m.div_ceil(BM) * n.div_ceil(BN)
}

fn main() {
    eprintln!(
        "warning: `tinygemma` is deprecated \
         (built only with --features legacy-gemma-bins)"
    );
    let n_cu: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);
    let out = std::env::args().nth(2).unwrap_or_else(|| "tinygemma.pkt".into());

    let mut b = Builder::new(n_cu);

    // ---- tensors -----------------------------------------------------------------
    let ids = b.tensor("ids", T as u64 * I32);
    let pos = b.tensor("pos", T as u64 * I32);
    let cos = b.tensor("cos", (T * HD / 2) as u64 * F32);
    let sin = b.tensor("sin", (T * HD / 2) as u64 * F32);
    // tied: the embedding table is also the lm_head weight.
    let emb = b.tensor("emb", (VOCAB * H) as u64 * BF16);

    let x = b.tensor("x", (T * H) as u64 * BF16); // the residual stream
    let hn = b.tensor("hn", (T * H) as u64 * BF16); // norm scratch
    let qg = b.tensor("qg", (T * QDIM) as u64 * BF16);
    let kg = b.tensor("kg", (T * KVDIM) as u64 * BF16);
    let vg = b.tensor("vg", (T * KVDIM) as u64 * BF16);
    let q = b.tensor("q", (T * QDIM) as u64 * BF16);
    let k = b.tensor("k", (T * KVDIM) as u64 * BF16);
    let v = b.tensor("v", (T * KVDIM) as u64 * BF16);
    let at = b.tensor("at", (T * QDIM) as u64 * BF16);
    // split-KV: flash emits unnormalized partials and a merge folds them
    const NSPLIT: u32 = 4;
    let opart = b.tensor("opart", (T * N_HEAD * NSPLIT * HD) as u64 * F32);
    let mlpart = b.tensor("mlpart", (T * N_HEAD * NSPLIT * 2) as u64 * F32);
    let og = b.tensor("og", (T * H) as u64 * BF16);
    let on = b.tensor("on", (T * H) as u64 * BF16);
    let gt = b.tensor("gt", (T * I) as u64 * BF16);
    let ut = b.tensor("ut", (T * I) as u64 * BF16);
    let fu = b.tensor("fu", (T * I) as u64 * BF16);
    let dg = b.tensor("dg", (T * H) as u64 * BF16);
    let dn = b.tensor("dn", (T * H) as u64 * BF16);
    let logits = b.tensor("logits", (T * VOCAB) as u64 * BF16);

    struct LW {
        wq: u32, wk: u32, wv: u32, wo: u32,
        wg: u32, wu: u32, wd: u32,
        g_in: u32, g_pa: u32, g_pf: u32, g_po: u32,
        qn: u32, kn: u32,
    }
    let lw: Vec<LW> = (0..LAYERS)
        .map(|l| LW {
            wq: b.tensor(&format!("l{l}.wq"), (QDIM * H) as u64 * BF16),
            wk: b.tensor(&format!("l{l}.wk"), (KVDIM * H) as u64 * BF16),
            wv: b.tensor(&format!("l{l}.wv"), (KVDIM * H) as u64 * BF16),
            wo: b.tensor(&format!("l{l}.wo"), (H * QDIM) as u64 * BF16),
            wg: b.tensor(&format!("l{l}.wg"), (I * H) as u64 * BF16),
            wu: b.tensor(&format!("l{l}.wu"), (I * H) as u64 * BF16),
            wd: b.tensor(&format!("l{l}.wd"), (H * I) as u64 * BF16),
            g_in: b.tensor(&format!("l{l}.g_in"), H as u64 * BF16),
            g_pa: b.tensor(&format!("l{l}.g_pa"), H as u64 * BF16),
            g_pf: b.tensor(&format!("l{l}.g_pf"), H as u64 * BF16),
            g_po: b.tensor(&format!("l{l}.g_po"), H as u64 * BF16),
            qn: b.tensor(&format!("l{l}.qn"), HD as u64 * BF16),
            kn: b.tensor(&format!("l{l}.kn"), HD as u64 * BF16),
        })
        .collect();
    let g_final = b.tensor("g_final", H as u64 * BF16);

    // ---- schedule ----------------------------------------------------------------
    // Ops that must serialise take the whole machine. Ops that are INDEPENDENT (q/k/v;
    // gate/up) get DISJOINT CU sets so they run at the same time — that is the entire
    // point of a packet interpreter, and the trace will show whether it happened.
    let all = b.all();
    let rows = T.min(n_cu); // one workgroup per row is the natural slicing for a norm
    let row_cus: Vec<u32> = (0..rows).collect();

    // embed: x = emb[ids] * bf16(sqrt(H))
    let embed_scale = bf16_round((H as f32).sqrt());
    let mut dep = b.emit(DevOp::Embed, row_cus.clone(), &[], |d| {
        d.t[0] = x; d.t[1] = emb; d.t[2] = ids;
        d.i[0] = T; d.i[1] = H;
        d.f[0] = embed_scale;
    });

    for l in 0..LAYERS as usize {
        let w = &lw[l];

        // ---- attention ----
        let c_norm = b.emit(DevOp::RmsNorm, row_cus.clone(), &[dep], |d| {
            d.t[0] = hn; d.t[1] = x; d.t[2] = w.g_in;
            d.i[0] = T; d.i[1] = H; d.f[0] = EPS;
        });

        // q, k, v are independent: give them disjoint CUs so they OVERLAP.
        let tq = gemm_tiles(T, QDIM);
        let tk = gemm_tiles(T, KVDIM);
        let tv = tk;
        let (cq, ck, cv) = three_way(n_cu, tq, tk, tv);
        let c_q = b.emit(DevOp::Gemm, cq, &[c_norm], |d| {
            d.t[0] = qg; d.t[1] = hn; d.t[2] = w.wq;
            d.i[0] = T; d.i[1] = QDIM; d.i[2] = H;
        });
        let c_k = b.emit(DevOp::Gemm, ck, &[c_norm], |d| {
            d.t[0] = kg; d.t[1] = hn; d.t[2] = w.wk;
            d.i[0] = T; d.i[1] = KVDIM; d.i[2] = H;
        });
        let c_v = b.emit(DevOp::Gemm, cv, &[c_norm], |d| {
            d.t[0] = vg; d.t[1] = hn; d.t[2] = w.wv;
            d.i[0] = T; d.i[1] = KVDIM; d.i[2] = H;
        });

        // q_norm + RoPE, k_norm + RoPE, and v_norm (weightless, NO RoPE).
        let c_qn = b.emit(DevOp::HeadNormRope, all.clone(), &[c_q], |d| {
            d.t[0] = q; d.t[1] = qg; d.t[2] = w.qn; d.t[3] = cos; d.t[4] = sin; d.t[5] = pos;
            d.i[0] = T; d.i[1] = N_HEAD; d.i[2] = HD; d.i[3] = 0; d.f[0] = EPS;
        });
        let c_kn = b.emit(DevOp::HeadNormRope, all.clone(), &[c_k], |d| {
            d.t[0] = k; d.t[1] = kg; d.t[2] = w.kn; d.t[3] = cos; d.t[4] = sin; d.t[5] = pos;
            d.i[0] = T; d.i[1] = N_KV_HEAD; d.i[2] = HD; d.i[3] = 0; d.f[0] = EPS;
        });
        // v_norm: gamma = NONE and cos = NONE. Leaving these set is the classic bug —
        // it applies RoPE to V and still produces fluent text.
        let c_vn = b.emit(DevOp::HeadNormRope, all.clone(), &[c_v], |d| {
            d.t[0] = v; d.t[1] = vg; /* t2=gamma, t3=cos, t4=sin stay TENSOR_NONE */
            d.t[5] = pos;
            d.i[0] = T; d.i[1] = N_KV_HEAD; d.i[2] = HD; d.i[3] = 0; d.f[0] = EPS;
        });

        let c_fa = b.emit(DevOp::FlashPrefill, all.clone(), &[c_qn, c_kn, c_vn], |d| {
            d.t[0] = opart; d.t[1] = mlpart; d.t[2] = q; d.t[3] = k; d.t[4] = v;
            d.i[0] = T; d.i[1] = T; d.i[2] = N_HEAD; d.i[3] = N_KV_HEAD;
            d.i[4] = 0; d.i[5] = WINDOW[l]; d.i[6] = HD; d.i[7] = NSPLIT;
            d.f[0] = 1.0; // Gemma: NO 1/sqrt(head_dim)
        });
        let c_mg = b.emit(DevOp::FlashMerge, all.clone(), &[c_fa], |d| {
            d.t[0] = at; d.t[1] = opart; d.t[2] = mlpart;
            d.i[0] = T; d.i[1] = N_HEAD; d.i[2] = NSPLIT; d.i[3] = HD;
        });

        let c_o = b.emit(DevOp::Gemm, all.clone(), &[c_mg], |d| {
            d.t[0] = og; d.t[1] = at; d.t[2] = w.wo;
            d.i[0] = T; d.i[1] = H; d.i[2] = QDIM;
        });
        // sandwich: post-norm FIRST, then add the residual.
        let c_on = b.emit(DevOp::RmsNorm, row_cus.clone(), &[c_o], |d| {
            d.t[0] = on; d.t[1] = og; d.t[2] = w.g_pa;
            d.i[0] = T; d.i[1] = H; d.f[0] = EPS;
        });
        let c_r1 = b.emit(DevOp::Residual, all.clone(), &[c_on], |d| {
            d.t[0] = x; d.t[1] = x; d.t[2] = on;
            d.i[0] = T * H; d.f[0] = 1.0; // first residual is unscaled
        });

        // ---- MLP ----
        let c_pf = b.emit(DevOp::RmsNorm, row_cus.clone(), &[c_r1], |d| {
            d.t[0] = hn; d.t[1] = x; d.t[2] = w.g_pf;
            d.i[0] = T; d.i[1] = H; d.f[0] = EPS;
        });
        // gate and up are independent — overlap them.
        let ti = gemm_tiles(T, I);
        let (cg, cu2) = two_way(n_cu, ti, ti);
        let c_g = b.emit(DevOp::Gemm, cg, &[c_pf], |d| {
            d.t[0] = gt; d.t[1] = hn; d.t[2] = w.wg;
            d.i[0] = T; d.i[1] = I; d.i[2] = H;
        });
        let c_u = b.emit(DevOp::Gemm, cu2, &[c_pf], |d| {
            d.t[0] = ut; d.t[1] = hn; d.t[2] = w.wu;
            d.i[0] = T; d.i[1] = I; d.i[2] = H;
        });
        let c_glu = b.emit(DevOp::Glu, all.clone(), &[c_g, c_u], |d| {
            d.t[0] = fu; d.t[1] = gt; d.t[2] = ut;
            d.i[0] = T * I; d.i[1] = 0; // 0 = gelu_tanh. Gemma is GeGLU, NOT SwiGLU.
        });
        let c_d = b.emit(DevOp::Gemm, all.clone(), &[c_glu], |d| {
            d.t[0] = dg; d.t[1] = fu; d.t[2] = w.wd;
            d.i[0] = T; d.i[1] = H; d.i[2] = I;
        });
        let c_dn = b.emit(DevOp::RmsNorm, row_cus.clone(), &[c_d], |d| {
            d.t[0] = dn; d.t[1] = dg; d.t[2] = w.g_po;
            d.i[0] = T; d.i[1] = H; d.f[0] = EPS;
        });
        dep = b.emit(DevOp::Residual, all.clone(), &[c_dn], |d| {
            d.t[0] = x; d.t[1] = x; d.t[2] = dn;
            d.i[0] = T * H;
            d.f[0] = LAYER_SCALAR; // Gemma folds layer_scalar into the SECOND residual
        });
    }

    // ---- head ----
    let c_fn = b.emit(DevOp::RmsNorm, row_cus.clone(), &[dep], |d| {
        d.t[0] = hn; d.t[1] = x; d.t[2] = g_final;
        d.i[0] = T; d.i[1] = H; d.f[0] = EPS;
    });
    let c_lm = b.emit(DevOp::Gemm, all.clone(), &[c_fn], |d| {
        d.t[0] = logits; d.t[1] = hn; d.t[2] = emb; // tied
        d.i[0] = T; d.i[1] = VOCAB; d.i[2] = H;
    });
    b.emit(DevOp::SoftCap, all.clone(), &[c_lm], |d| {
        d.t[0] = logits; d.t[1] = logits;
        d.i[0] = T * VOCAB; d.f[0] = SOFTCAP;
    });

    let prog = b.finish();
    let blob = prog.to_blob();
    std::fs::write(&out, &blob).expect("write blob");

    eprintln!("tinygemma: T={T} H={H} I={I} heads={N_HEAD}x{HD} kv={N_KV_HEAD} layers={LAYERS}");
    eprintln!(
        "  {} packets, {} workgroup-packets, {} counters, {} tensors -> {} ({} B)",
        prog.insts.len(),
        prog.n_trace(),
        prog.n_counter,
        prog.tensors.len(),
        out,
        blob.len()
    );
}

/// bf16 round-to-nearest-even. The embedding scale must be the BF16-ROUNDED sqrt(hidden)
/// — using the f32 value is a real (if small) mismatch against the reference.
fn bf16_round(f: f32) -> f32 {
    let u = f.to_bits();
    let r = u.wrapping_add(0x7fff).wrapping_add((u >> 16) & 1);
    f32::from_bits(r & 0xffff_0000)
}

/// Split the machine between two independent ops, proportional to their tile counts.
fn two_way(n_cu: u32, a: u32, b: u32) -> (Vec<u32>, Vec<u32>) {
    let split = ((n_cu as u64 * a as u64) / (a + b).max(1) as u64).max(1) as u32;
    let split = split.min(n_cu - 1);
    ((0..split).collect(), (split..n_cu).collect())
}

fn three_way(n_cu: u32, a: u32, b: u32, c: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tot = (a + b + c).max(1) as u64;
    let sa = ((n_cu as u64 * a as u64) / tot).max(1) as u32;
    let sb = ((n_cu as u64 * b as u64) / tot).max(1) as u32;
    let sa = sa.min(n_cu - 2);
    let sb = sb.min(n_cu - sa - 1);
    (
        (0..sa).collect(),
        (sa..sa + sb).collect(),
        (sa + sb..n_cu).collect(),
    )
}
