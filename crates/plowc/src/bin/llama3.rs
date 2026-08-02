//! `llama3` — compile a LLaMA 3.1 8B (or any Llama-architecture) BF16 prefill+decode
//! network into a device packet program, straight from the HuggingFace checkpoint.
//!
//! Reads `config.json` and the safetensors index, and emits packets whose weight tensors
//! are named EXACTLY as the checkpoint names them, so the runtime can bind them by name
//! and hard-fail on anything missing.
//!
//! # LLaMA 3.1 8B spec (verified against the checkpoint and modeling_llama.py)
//!
//! * **Pre-norm residual**: `x = x + attn(rmsnorm(x))`, then `x = x + mlp(rmsnorm(x))`.
//!   NO sandwich norms, NO post-norms.
//! * **RMSNorm**: `x * pow(mean(x^2) + eps, -0.5) * w`, eps INSIDE the power.
//! * **Attention scale**: `1/sqrt(head_dim)` — the standard scaling.
//! * **NO qk_norm, NO v_norm** — projections feed directly to RoPE/attention.
//! * **Full causal attention** on ALL layers (no sliding window).
//! * **RoPE over the full head_dim (128)**, θ = 500000.0 for 3.1.
//! * **MLP is SwiGLU**: `down(silu(gate(x)) * up(x))`.
//! * **Separate `lm_head.weight`** (`tie_word_embeddings: false` in 3.1 8B).
//! * **NO embedding scale** (unlike Gemma).
//! * **NO logit softcap** (unlike Gemma).
//! * **GQA**: 32 query heads, 8 KV heads (ratio 4:1).

use std::path::{Path, PathBuf};

use packet::dev::{DevOp, TENSOR_NONE};
use packet::devbuild::{Builder, Model};
use serde_json::Value;

const BF16: u64 = 2;
const F32: u64 = 4;
const I32: u64 = 4;
const BM: u32 = 256;
const BN: u32 = 256;

struct Cfg {
    hidden: u32,
    inter: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    eps: f32,
    vocab: u32,
    rope_theta: f64,
    tie_word_embeddings: bool,
}

fn cfg_from(dir: &Path) -> Cfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .unwrap();
    let g = |k: &str| v[k].as_u64().unwrap() as u32;
    let heads = g("num_attention_heads");
    let hidden = g("hidden_size");
    let kv_heads = v["num_key_value_heads"]
        .as_u64()
        .map(|x| x as u32)
        .unwrap_or(heads);
    let head_dim = v["head_dim"]
        .as_u64()
        .map(|x| x as u32)
        .unwrap_or(hidden / heads);
    Cfg {
        hidden,
        inter: g("intermediate_size"),
        layers: g("num_hidden_layers"),
        heads,
        kv_heads,
        head_dim,
        eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-5) as f32,
        vocab: g("vocab_size"),
        rope_theta: v["rope_theta"].as_f64().unwrap_or(500000.0),
        tie_word_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false),
    }
}

/// Pick the GEMM tile for one shape.
///
/// DELEGATES to [`devgen::gfx950_prefill_tile`], which is the picker the
/// emitters actually use. This function used to be a second implementation —
/// three hardcoded rungs and a 17-line inline `rounds * work / intensity`
/// heuristic — and it had drifted: the real ladder is FIVE rungs in each of
/// three weight encodings (`devgen`'s `GFX950_RUNGS`), chosen against measured
/// `tunedb` records with an analytical fallback, keyed by hardware fingerprint.
/// So this binary reported a tile the build would not emit.
///
/// `kernelcaps::select` names this exact pair as the problem it exists to end
/// ("two of those exist today … they disagree with each other"), and
/// `docs/arch/10-implementation-status.md` still lists it as the highest
/// outstanding item. `gfx950_prefill_tile` is `pub` precisely so a caller
/// outside the emitters can ask what the build would emit.
fn pick_tile(m: u32, n: u32, k: u32, n_cu: u32) -> DevOp {
    devgen::gfx950_prefill_tile(m, n, k, n_cu, kernelcaps::QuantScheme::None)
}

fn tiles(m: u32, n: u32) -> u32 {
    m.div_ceil(BM) * n.div_ceil(BN)
}

/// Every tensor the model touches.
struct Tn {
    ids: u32,
    pos: u32,
    kvlen: u32,
    cos: u32,
    sin: u32,
    emb: u32,
    lm_head: u32,
    fin: u32,
    x: u32,
    hn: u32,
    qg: u32,
    kg: u32,
    vg: u32,
    q: u32,
    opart: u32,
    mlpart: u32,
    at: u32,
    og: u32,
    gt: u32,
    ut: u32,
    fu: u32,
    dg: u32,
    logits: u32,
    amax: u32,
    kc: Vec<u32>,
    vc: Vec<u32>,
    lw: Vec<LW>,
}
struct LW {
    wq: u32,
    wk: u32,
    wv: u32,
    wo: u32,
    wg: u32,
    wu: u32,
    wd: u32,
    g_in: u32,
    g_pa: u32,
}

/// Largest prefill chunk.
const MAX_CHUNK: u32 = 8192;

/// Argmax partial reduction blocks.
const AMAX_BLOCKS: u32 = 64;

fn declare(b: &mut Builder, c: &Cfg, ctx: u32, ns_pre: u32) -> Tn {
    let rows = ctx.min(MAX_CHUNK);
    let qd = c.heads * c.head_dim;
    let kd = c.kv_heads * c.head_dim;
    let ac = |b: &mut Builder, n: &str, sz: u64| b.tensor(&format!("act.{n}"), sz);

    let (cs, sn) = rope_tables(ctx, c.head_dim, c.rope_theta);

    let emb = b.tensor(
        "model.embed_tokens.weight",
        (c.vocab * c.hidden) as u64 * BF16,
    );
    let lm_head = if c.tie_word_embeddings {
        emb // reuse embedding tensor
    } else {
        b.tensor("lm_head.weight", (c.vocab * c.hidden) as u64 * BF16)
    };

    let mut t = Tn {
        ids: b.tensor("in.ids", ctx as u64 * I32),
        pos: b.tensor("in.pos", ctx as u64 * I32),
        kvlen: b.tensor("in.kvlen", I32),
        cos: b.tensor_init("in.cos", cs),
        sin: b.tensor_init("in.sin", sn),
        emb,
        lm_head,
        fin: b.tensor("model.norm.weight", c.hidden as u64 * BF16),
        x: ac(b, "x", (rows * c.hidden) as u64 * BF16),
        hn: ac(b, "hn", (rows * c.hidden) as u64 * BF16),
        qg: ac(b, "qg", (rows * qd) as u64 * BF16),
        kg: ac(b, "kg", (rows * kd) as u64 * BF16),
        vg: ac(b, "vg", (rows * kd) as u64 * BF16),
        q: ac(b, "q", (rows * qd) as u64 * BF16),
        opart: ac(
            b,
            "opart",
            (rows.max(64) * c.heads * ns_pre * c.head_dim).max(c.heads * 64 * c.head_dim) as u64
                * F32,
        ),
        mlpart: ac(
            b,
            "mlpart",
            (rows.max(64) * c.heads * ns_pre * 2).max(c.heads * 64 * 2) as u64 * F32,
        ),
        at: ac(b, "at", (rows * qd) as u64 * BF16),
        og: ac(b, "og", (rows * c.hidden) as u64 * BF16),
        gt: ac(b, "gt", (rows * c.inter) as u64 * BF16),
        ut: ac(b, "ut", (rows * c.inter) as u64 * BF16),
        fu: ac(b, "fu", (rows * c.inter) as u64 * BF16),
        dg: ac(b, "dg", (rows * c.hidden) as u64 * BF16),
        logits: ac(b, "logits", c.vocab as u64 * BF16),
        amax: ac(b, "amax.part", AMAX_BLOCKS as u64 * 8),
        kc: Vec::new(),
        vc: Vec::new(),
        lw: Vec::new(),
    };
    for l in 0..c.layers {
        // Full-causal KV cache: linear, no ring.
        t.kc.push(b.tensor(
            &format!("kv.{l}.k"),
            (ctx * c.kv_heads * c.head_dim) as u64 * BF16,
        ));
        t.vc.push(b.tensor(
            &format!("kv.{l}.v"),
            (ctx * c.kv_heads * c.head_dim) as u64 * BF16,
        ));
        let w = |b: &mut Builder, s: &str, sz: u64| b.tensor(&format!("model.layers.{l}.{s}"), sz);
        t.lw.push(LW {
            wq: w(b, "self_attn.q_proj.weight", (qd * c.hidden) as u64 * BF16),
            wk: w(b, "self_attn.k_proj.weight", (kd * c.hidden) as u64 * BF16),
            wv: w(b, "self_attn.v_proj.weight", (kd * c.hidden) as u64 * BF16),
            wo: w(b, "self_attn.o_proj.weight", (c.hidden * qd) as u64 * BF16),
            wg: w(
                b,
                "mlp.gate_proj.weight",
                (c.inter * c.hidden) as u64 * BF16,
            ),
            wu: w(b, "mlp.up_proj.weight", (c.inter * c.hidden) as u64 * BF16),
            wd: w(
                b,
                "mlp.down_proj.weight",
                (c.hidden * c.inter) as u64 * BF16,
            ),
            g_in: w(b, "input_layernorm.weight", c.hidden as u64 * BF16),
            g_pa: w(b, "post_attention_layernorm.weight", c.hidden as u64 * BF16),
        });
    }
    t
}

/// LDS the GEMM arena holds, in halves.
const GM_LDS_HALVES: u64 = 2 * (256 + 256) * (64 + 8);

const Q_TILE_ROWS: u32 = 8 * 32; // PLOW_WAVES * FA_BQ

/// Emit one phase. `t == 1 && decode` is the decode step; otherwise a prefill bucket.
fn emit_phase(b: &mut Builder, c: &Cfg, n: &Tn, t: u32, ctx: u32, decode: bool, n_cu: u32) {
    let all = b.all();
    let rows: Vec<u32> = (0..t.min(n_cu).max(1)).collect();
    let elem = |ne: u32| -> Vec<u32> { (0..ne.div_ceil(512 * 8).max(1).min(n_cu)).collect() };
    let ns = if decode {
        n_cu.div_ceil(c.heads).max(1)
    } else {
        n_cu.div_ceil((t.div_ceil(Q_TILE_ROWS) * c.heads).max(1))
            .max(1)
    };
    let scale = 1.0 / (c.head_dim as f32).sqrt();

    // Embed: x = emb[ids]  (NO scale for LLaMA)
    let mut dep = b.emit(DevOp::Embed, rows.clone(), &[], |d| {
        d.t[0] = n.x;
        d.t[1] = n.emb;
        d.t[2] = n.ids;
        d.i[0] = t;
        d.i[1] = c.hidden;
        d.f[0] = 1.0; // no embedding scale
    });

    let eps = c.eps;
    let qd = c.heads * c.head_dim;
    let kd = c.kv_heads * c.head_dim;
    let kv_mask = 0xFFFF_FFFFu32; // full causal: no ring

    let proj = |b: &mut Builder,
                out: u32,
                a: u32,
                w: u32,
                m: u32,
                nn: u32,
                k: u32,
                gamma: u32,
                cus: Vec<u32>,
                deps: &[u32]|
     -> u32 {
        let fold = decode && gamma != TENSOR_NONE;
        let op = if decode {
            DevOp::Gemv
        } else {
            pick_tile(m, nn, k, n_cu)
        };
        b.emit(op, cus, deps, |d| {
            d.t[0] = out;
            d.t[1] = a;
            d.t[2] = w;
            if fold {
                d.t[4] = gamma;
            }
            d.i[0] = m;
            d.i[1] = nn;
            d.i[2] = k;
            d.i[3] = if fold { 2 } else { 0 };
            d.i[4] = 0;
            d.f[0] = eps;
        })
    };

    for l in 0..c.layers as usize {
        let w = &n.lw[l];

        // RMSNorm before attention.
        // In decode, fold into the consuming GEMV (norm mode 2).
        // In prefill, keep as a separate packet (T rows parallelise well).
        let c_n = b.emit(DevOp::RmsNorm, rows.clone(), &[dep], |d| {
            d.t[0] = n.hn;
            d.t[1] = n.x;
            d.t[2] = w.g_in;
            d.i[0] = t;
            d.i[1] = c.hidden;
            d.f[0] = eps;
        });
        let (qkv_src, qkv_g) = (n.hn, TENSOR_NONE);

        // q, k, v are independent — disjoint CU sets for overlap.
        let (cq, ck, cv) = if decode {
            split3(n_cu, qd, kd, kd)
        } else {
            split3(n_cu, tiles(t, qd), tiles(t, kd), tiles(t, kd))
        };
        let c_q = proj(b, n.qg, qkv_src, w.wq, t, qd, c.hidden, qkv_g, cq, &[c_n]);
        let c_k = proj(b, n.kg, qkv_src, w.wk, t, kd, c.hidden, qkv_g, ck, &[c_n]);
        let c_v = proj(b, n.vg, qkv_src, w.wv, t, kd, c.hidden, qkv_g, cv, &[c_n]);

        // RoPE on q and k — NO per-head norm (unlike Gemma).
        // HeadNormRope with gamma=TENSOR_NONE skips the norm; cos/sin apply RoPE.
        let hn_cus: Vec<u32> = (0..((t * c.heads).div_ceil(8)).min(n_cu).max(1)).collect();
        let c_qr = b.emit(DevOp::HeadNormRope, hn_cus.clone(), &[c_q], |d| {
            d.t[0] = n.q;
            d.t[1] = n.qg;
            // t2 = gamma: TENSOR_NONE (no head norm)
            d.t[3] = n.cos;
            d.t[4] = n.sin;
            d.t[5] = n.pos;
            d.i[0] = t;
            d.i[1] = c.heads;
            d.i[2] = c.head_dim;
            d.i[3] = 0;
            d.f[0] = eps;
        });
        let c_kr = b.emit(DevOp::HeadNormRope, hn_cus.clone(), &[c_k], |d| {
            d.t[0] = n.kc[l];
            d.t[1] = n.kg;
            // t2 = gamma: TENSOR_NONE (no head norm)
            d.t[3] = n.cos;
            d.t[4] = n.sin;
            d.t[5] = n.pos;
            d.i[0] = t;
            d.i[1] = c.kv_heads;
            d.i[2] = c.head_dim;
            d.i[3] = 0;
            d.f[0] = eps;
            // Full causal: linear KV cache, no ring.
            d.j[0] = ctx;
            d.j[1] = kv_mask;
        });
        // V: straight to the KV cache, NO norm, NO RoPE.
        let c_vr = b.emit(DevOp::HeadNormRope, hn_cus, &[c_v], |d| {
            d.t[0] = n.vc[l];
            d.t[1] = n.vg;
            // t2=gamma, t3=cos, t4=sin all TENSOR_NONE
            d.t[5] = n.pos;
            d.i[0] = t;
            d.i[1] = c.kv_heads;
            d.i[2] = c.head_dim;
            d.i[3] = 0;
            d.f[0] = eps;
            d.j[0] = ctx;
            d.j[1] = kv_mask;
        });

        // Flash attention.
        let c_fa = if decode {
            b.emit(DevOp::FlashDecode, all.clone(), &[c_qr, c_kr, c_vr], |d| {
                d.t[0] = n.opart;
                d.t[1] = n.mlpart;
                d.t[2] = n.q;
                d.t[3] = n.kc[l];
                d.t[4] = n.vc[l];
                d.t[5] = n.kvlen;
                d.i[0] = 1;
                d.i[1] = c.heads;
                d.i[2] = c.kv_heads;
                d.i[3] = ctx;
                d.i[4] = 0; // window=0 → full causal
                d.i[5] = ns;
                d.i[6] = c.head_dim;
                d.i[7] = kv_mask;
                d.f[0] = scale;
            })
        } else {
            b.emit(DevOp::FlashPrefill, all.clone(), &[c_qr, c_kr, c_vr], |d| {
                d.t[0] = n.opart;
                d.t[1] = n.mlpart;
                d.t[2] = n.q;
                d.t[3] = n.kc[l];
                d.t[4] = n.vc[l];
                d.i[0] = t;
                d.i[1] = t;
                d.i[2] = c.heads;
                d.i[3] = c.kv_heads;
                d.i[4] = 0;
                d.i[5] = 0; // window=0 → full causal
                d.i[6] = c.head_dim;
                d.i[7] = ns;
                d.f[0] = scale;
                d.j[0] = ctx;
                d.j[1] = kv_mask;
            })
        };
        let mg_cus: Vec<u32> = (0..(t * c.heads).min(n_cu).max(1)).collect();
        let c_mg = b.emit(DevOp::FlashMerge, mg_cus, &[c_fa], |d| {
            d.t[0] = n.at;
            d.t[1] = n.opart;
            d.t[2] = n.mlpart;
            d.i[0] = t;
            d.i[1] = c.heads;
            d.i[2] = ns;
            d.i[3] = c.head_dim;
        });

        // o_proj
        let c_o = proj(
            b,
            n.og,
            n.at,
            w.wo,
            t,
            c.hidden,
            qd,
            TENSOR_NONE,
            all.clone(),
            &[c_mg],
        );
        // Simple pre-norm residual: x = x + o_proj(attn). Scale = 1.0 (no layer_scalar).
        let c_r1 = b.emit(DevOp::Residual, elem(t * c.hidden), &[c_o], |d| {
            d.t[0] = n.x;
            d.t[1] = n.x;
            d.t[2] = n.og;
            d.i[0] = t * c.hidden;
            d.f[0] = 1.0;
        });

        // MLP: RMSNorm → gate/up → SwiGLU → down → residual
        let c_pn = b.emit(DevOp::RmsNorm, rows.clone(), &[c_r1], |d| {
            d.t[0] = n.hn;
            d.t[1] = n.x;
            d.t[2] = w.g_pa;
            d.i[0] = t;
            d.i[1] = c.hidden;
            d.f[0] = eps;
        });
        let (mlp_src, mlp_g) = (n.hn, TENSOR_NONE);

        // SwiGLU fusion (decode: GemvGlu, prefill: GemmGlu when the tile is 256x256).
        let glu_fused = decode && (t as u64 * c.hidden as u64) <= GM_LDS_HALVES;
        let gemm_glu = !decode && pick_tile(t, c.inter, c.hidden, n_cu) == DevOp::Gemm;
        let c_d = if glu_fused {
            let c_gl = b.emit(DevOp::GemvGlu, all.clone(), &[c_pn], |d| {
                d.t[0] = n.fu;
                d.t[1] = mlp_src;
                d.t[2] = w.wg;
                d.t[5] = w.wu;
                d.i[0] = t;
                d.i[1] = c.inter;
                d.i[2] = c.hidden;
                d.i[5] = 1; // 1 = SiLU (SwiGLU), NOT 0 (GeGLU)
            });
            proj(
                b,
                n.dg,
                n.fu,
                w.wd,
                t,
                c.hidden,
                c.inter,
                TENSOR_NONE,
                all.clone(),
                &[c_gl],
            )
        } else if gemm_glu {
            let c_gl = b.emit(DevOp::GemmGlu, all.clone(), &[c_pn], |d| {
                d.t[0] = n.fu;
                d.t[1] = mlp_src;
                d.t[2] = w.wg;
                d.t[5] = w.wu;
                d.i[0] = t;
                d.i[1] = c.inter;
                d.i[2] = c.hidden;
                d.i[5] = 1; // 1 = SiLU (SwiGLU)
            });
            proj(
                b,
                n.dg,
                n.fu,
                w.wd,
                t,
                c.hidden,
                c.inter,
                TENSOR_NONE,
                all.clone(),
                &[c_gl],
            )
        } else {
            // Unfused path: separate gate, up, GLU, down.
            let (cg, cu) = if decode {
                split2(n_cu, 1, 1)
            } else {
                split2(n_cu, tiles(t, c.inter), tiles(t, c.inter))
            };
            let c_g = proj(
                b,
                n.gt,
                mlp_src,
                w.wg,
                t,
                c.inter,
                c.hidden,
                mlp_g,
                cg,
                &[c_pn],
            );
            let c_u = proj(
                b,
                n.ut,
                mlp_src,
                w.wu,
                t,
                c.inter,
                c.hidden,
                mlp_g,
                cu,
                &[c_pn],
            );
            let c_gl = b.emit(DevOp::Glu, elem(t * c.inter), &[c_g, c_u], |d| {
                d.t[0] = n.fu;
                d.t[1] = n.gt;
                d.t[2] = n.ut;
                d.i[0] = t * c.inter;
                d.i[1] = 1; // 1 = SiLU (SwiGLU)
            });
            proj(
                b,
                n.dg,
                n.fu,
                w.wd,
                t,
                c.hidden,
                c.inter,
                TENSOR_NONE,
                all.clone(),
                &[c_gl],
            )
        };
        // x = x + down. Scale = 1.0 (no layer_scalar).
        dep = b.emit(DevOp::Residual, elem(t * c.hidden), &[c_d], |d| {
            d.t[0] = n.x;
            d.t[1] = n.x;
            d.t[2] = n.dg;
            d.i[0] = t * c.hidden;
            d.f[0] = 1.0;
        });
    }

    // Final norm + lm_head
    let c_f = b.emit(DevOp::RmsNorm, rows.clone(), &[dep], |d| {
        d.t[0] = n.hn;
        d.t[1] = n.x;
        d.t[2] = n.fin;
        d.i[0] = t;
        d.i[1] = c.hidden;
        d.f[0] = eps;
    });
    // lm_head over the LAST row only.
    let lm_op = if decode {
        DevOp::Gemv
    } else {
        pick_tile(1, c.vocab, c.hidden, n_cu)
    };
    let c_lm = b.emit(lm_op, all.clone(), &[c_f], |d| {
        d.t[0] = n.logits;
        d.t[1] = n.hn;
        d.t[2] = n.lm_head;
        d.i[0] = 1;
        d.i[1] = c.vocab;
        d.i[2] = c.hidden;
        d.i[3] = 0;
        d.i[4] = t - 1; // a_row0 = last token
    });
    // NO softcap for LLaMA.

    // Greedy argmax on device.
    let amax_cus: Vec<u32> = (0..AMAX_BLOCKS).collect();
    let c_am = b.emit(DevOp::Argmax, amax_cus, &[c_lm], |d| {
        d.t[0] = n.amax;
        d.t[1] = n.logits;
        d.i[0] = c.vocab;
    });
    b.emit(DevOp::ArgmaxFin, vec![0], &[c_am], |d| {
        d.t[0] = n.ids;
        d.t[1] = n.amax;
        d.i[0] = AMAX_BLOCKS;
    });
}

fn main() {
    let mut a = std::env::args().skip(1);
    let dir = PathBuf::from(
        a.next()
            .expect("usage: llama3 <model-dir> <max_ctx> <out.pkt> [n_cu]"),
    );
    let ctx: u32 = a.next().expect("max_ctx").parse().unwrap();
    let out = a.next().unwrap_or_else(|| "llama3.pkt".into());
    let n_cu: u32 = a.next().and_then(|s| s.parse().ok()).unwrap_or(256);

    let c = cfg_from(&dir);

    // Prefill buckets, capped at MAX_CHUNK.
    let buckets: Vec<u32> = [128u32, 512, 1024, 2048, 4096, 8192]
        .into_iter()
        .filter(|&x| x <= ctx.min(MAX_CHUNK))
        .collect();
    let arows = ctx.min(MAX_CHUNK);
    let ns_pre = n_cu
        .div_ceil((arows.div_ceil(Q_TILE_ROWS) * c.heads).max(1))
        .max(1);

    let mut tb = Builder::new(n_cu);
    let tn = declare(&mut tb, &c, ctx, ns_pre);
    let tensors = tb.tensors();

    let mut progs = Vec::new();
    let mut tlist = Vec::new();
    for &t in &buckets {
        let mut b = Builder::new(n_cu);
        b.adopt_tensors(tensors.clone());
        emit_phase(&mut b, &c, &tn, t, ctx, false, n_cu);
        progs.push(b.finish());
        tlist.push(t);
    }
    // Decode program
    let mut bd = Builder::new(n_cu);
    bd.adopt_tensors(tensors.clone());
    emit_phase(&mut bd, &c, &tn, 1, ctx, true, n_cu);
    progs.push(bd.finish());
    tlist.push(1);

    let m = Model {
        n_cu,
        target: 0,
        tensors,
        progs,
        kv_row_insts: Vec::new(),
        prog_t: tlist,
        // This deprecated path still expands its RoPE tables via `tensor_init`,
        // so it keeps emitting a v5 blob. `plowc --hf-dir` is the supported route.
        gen: Vec::new(),
    };
    std::fs::write(&out, m.to_blob()).unwrap();

    let wb: u64 = m
        .tensors
        .iter()
        .filter(|x| x.name.starts_with("model.") || x.name.starts_with("lm_head"))
        .map(|x| x.bytes)
        .sum();
    let kb: u64 = m
        .tensors
        .iter()
        .filter(|x| x.name.starts_with("kv."))
        .map(|x| x.bytes)
        .sum();
    let ab: u64 = m
        .tensors
        .iter()
        .filter(|x| x.name.starts_with("act."))
        .map(|x| x.bytes)
        .sum();
    eprintln!(
        "llama3: {} layers  hidden={} inter={}  heads={} kv_heads={}  hd={}  vocab={}",
        c.layers, c.hidden, c.inter, c.heads, c.kv_heads, c.head_dim, c.vocab
    );
    eprintln!("  max_ctx={}  prefill buckets {:?} + decode", ctx, buckets);
    for (i, p) in m.progs.iter().enumerate() {
        eprintln!(
            "    prog {} (T={:>4}): {:>5} packets, {:>7} workgroup-packets",
            i,
            m.prog_t[i],
            p.insts.len(),
            p.stream.len()
        );
    }
    eprintln!(
        "  weights {:.1} GiB   KV cache {:.2} GiB   activations {:.2} GiB   -> {}",
        wb as f64 / (1u64 << 30) as f64,
        kb as f64 / (1u64 << 30) as f64,
        ab as f64 / (1u64 << 30) as f64,
        out
    );
}

/// RoPE cos/sin tables for LLaMA: full head_dim, no partial rotary.
/// Layout: `cos[pos * (hd/2) + j]`.
fn rope_tables(t: u32, hd: u32, theta: f64) -> (Vec<u8>, Vec<u8>) {
    let h2 = (hd / 2) as usize;
    let mut cos = Vec::with_capacity(t as usize * h2 * 4);
    let mut sin = Vec::with_capacity(t as usize * h2 * 4);
    for p in 0..t as usize {
        for j in 0..h2 {
            let inv = 1.0 / theta.powf(2.0 * j as f64 / hd as f64);
            let a = p as f64 * inv;
            let (s, c) = (a.sin(), a.cos());
            cos.extend_from_slice(&(c as f32).to_le_bytes());
            sin.extend_from_slice(&(s as f32).to_le_bytes());
        }
    }
    (cos, sin)
}

fn split2(n: u32, a: u32, b: u32) -> (Vec<u32>, Vec<u32>) {
    let s = (((n as u64 * a as u64) / (a + b).max(1) as u64).max(1) as u32).min(n - 1);
    ((0..s).collect(), (s..n).collect())
}

fn split3(n: u32, a: u32, b: u32, c: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let tot = (a + b + c).max(1) as u64;
    let sa = (((n as u64 * a as u64) / tot).max(1) as u32).min(n - 2);
    let sb = (((n as u64 * b as u64) / tot).max(1) as u32).min(n - sa - 1);
    (
        (0..sa).collect(),
        (sa..sa + sb).collect(),
        (sa + sb..n).collect(),
    )
}
