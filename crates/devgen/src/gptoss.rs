//! GPT-OSS-20B emitter (`model_type: "gpt_oss"`), CPU-tier target.
//!
//! Per layer: `RMSNorm -> q/k/v (+bias) -> half-split YaRN RoPE -> flash (sliding 128 on even
//! layers, full on odd) + FLASH_MERGE with sinks -> o_proj (+bias) -> AddNorm -> router GEMV
//! (+bias) -> top-4 softmax -> MXFP4 experts (swiglu_oai, per-expert bias) -> combine ->
//! AddNorm`. Untied `lm_head`. Contracts: `plans/gptoss-isa.md`, `dev_isa.h` ops 147-150.
//!
//! One `Builder` per program in the qwen35 style: prefill buckets ascending, then the decode
//! batch ladder ascending, all sharing one tensor table sized at the widest use. Every weight
//! is declared under its CHECKPOINT name (the MXFP4 `*_blocks`/`*_scales` u8 tensors are bound
//! verbatim — their bytes are already plow's fp4 row layout), so the coverage gate refuses a
//! checkpoint with tensors this emitter does not consume.
use super::*;
use packet::dev::{ACT_SWIGLU_OAI, ROPE_PAIR_HALF};
use packet::devbuild::TensorDecl;
use std::path::Path;

/// Default prefill chunk (and bucket cap). The dense path derives it from the sliding window,
/// which at window 128 would make every prefill a 128-row pass over the whole weight set; the
/// ring cost that rule protects is negligible here (window + chunk rows of 8 x 64 bf16).
/// `PLOW_MAX_CHUNK` overrides.
const DEFAULT_CHUNK: u32 = 1024;
/// `MOE_ALIGN_PF` pads every expert segment to a row tile; the gathered arrays are sized for a
/// pad of up to this many rows per expert (the dense path's 128; AMD's MPF_BM is 64).
const MOE_PAD_ROWS: u32 = 128;
/// Router flags for `MoeRouterTopkPf`: bit0 = 0 softmax scoring, bit1 = renormalise the top-k.
/// softmax-over-all then renormalise the selected == softmax over the selected (HF).
const ROUTER_FLAGS: u32 = 2;

struct Emitter<'a> {
    b: Builder,
    c: &'a GptOssCfg,
    prefill: bool,
    /// Rows this program computes: the prefill bucket, or the decode rung (sequences).
    t: u32,
    /// Widest decode rung = KV slot count; sizes every per-slot tensor.
    dbatch: u32,
    ladder: bool,
    ctx: u32,
    chunk: u32,
    ns: u32,
    pos: u32,
    kvlen: u32,
    cos: u32,
    sin: u32,
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
    rlogit: u32,
    tab: u32,
    fu: u32,
    part: u32,
    moe: u32,
    meta: u32,
    row_token: u32,
    row_partidx: u32,
    row_gate: u32,
    kv_rows: Vec<u32>,
}

impl Emitter<'_> {
    fn w(&mut self, name: &str, bytes: u64) -> u32 {
        self.b.tensor(name, bytes)
    }
    fn rows(&self) -> Vec<u32> {
        (0..self.t.min(self.b.n_cu()).max(1)).collect()
    }
    /// `out[t, n] = src[t, k] . W[n, k]^T + bias`. Decode is the GEMV family, prefill the plain
    /// GEMM: the CPU tier has one GEMM body, so the GPU tile rungs are not selected here.
    fn proj(&mut self, out: u32, src: u32, w: u32, bias: u32, n: u32, k: u32, dep: u32) -> u32 {
        let op = if self.prefill { DevOp::Gemm } else { DevOp::Gemv };
        let t = self.t;
        self.b.emit(op, self.b.all(), &[dep], |d| {
            d.t[0] = out;
            d.t[1] = src;
            d.t[2] = w;
            d.t[7] = bias;
            d.i[0] = t;
            d.i[1] = n;
            d.i[2] = k;
        })
    }
    /// Decode-only MXFP4 twin of `proj`: `out[t, n] = src[t] . dequant(W4[n]) + bias[n]` (op 91, t7 bias).
    fn proj_mx4(&mut self, out: u32, src: u32, w4: u32, s4: u32, bias: u32, n: u32, k: u32, dep: u32) -> u32 {
        debug_assert!(!self.prefill && w4 != TENSOR_NONE);
        let t = self.t;
        self.b.emit(DevOp::GemvMxfp4, self.b.all(), &[dep], |d| {
            d.t[0] = out;
            d.t[1] = src;
            d.t[2] = w4;
            d.t[3] = s4;
            d.t[7] = bias;
            d.i[0] = t;
            d.i[1] = n;
            d.i[2] = k;
        })
    }
    /// `hn = RMSNorm(x + b, gamma)`, `x += b` in place (the Llama pre-norm tail).
    fn add_norm(&mut self, b_in: u32, gamma: u32, dep: u32) -> u32 {
        let (x, hn, t, h, eps) = (self.x, self.hn, self.t, self.c.hidden, self.c.eps);
        self.b.emit(DevOp::AddNorm, self.rows(), &[dep], |d| {
            d.t[0] = hn;
            d.t[1] = x;
            d.t[2] = x;
            d.t[3] = b_in;
            d.t[4] = gamma;
            d.i[0] = t;
            d.i[1] = h;
            d.f[0] = eps;
        })
    }
    fn rope(&mut self, out: u32, src: u32, nhead: u32, rotate: bool, cache: Option<(u32, u32)>, dep: u32) -> u32 {
        let (cos, sin, pos, t, hd, eps) = (self.cos, self.sin, self.pos, self.t, self.c.hd, self.c.eps);
        let batch_kv = if !self.prefill && (t > 1 || self.ladder) { t } else { 0 };
        self.b.emit(DevOp::HeadNormRope, self.b.all(), &[dep], |d| {
            d.t[0] = out;
            d.t[1] = src;
            d.t[5] = pos;
            if rotate {
                d.t[3] = cos;
                d.t[4] = sin;
                // GPT-OSS has no qk-norm and is NeoX half-split at hd 64 — the one case the
                // legacy hd==64 rule reads as interleaved.
                d.i[5] = ROPE_PAIR_HALF;
            }
            d.i[0] = t;
            d.i[1] = nhead;
            d.i[2] = hd;
            d.i[3] = 0;
            d.i[4] = 1; // skip_norm
            d.f[0] = eps;
            if let Some((ring, mask)) = cache {
                d.j[0] = ring;
                d.j[1] = mask;
                d.i[6] = batch_kv;
            }
        })
    }

    fn layer(&mut self, l: usize, dep: u32, gamma_next: u32) -> u32 {
        let c = self.c;
        let (t, h, hd, heads, kvh) = (self.t, c.hidden, c.hd, c.heads, c.kvh);
        let (qd, kd) = (heads * hd, kvh * hd);
        let p = format!("{}layers.{l}", c.prefix);
        let wq = self.w(&format!("{p}.self_attn.q_proj.weight"), qd as u64 * h as u64 * BF16);
        let bq = self.w(&format!("{p}.self_attn.q_proj.bias"), qd as u64 * BF16);
        let wk = self.w(&format!("{p}.self_attn.k_proj.weight"), kd as u64 * h as u64 * BF16);
        let bk = self.w(&format!("{p}.self_attn.k_proj.bias"), kd as u64 * BF16);
        let wv = self.w(&format!("{p}.self_attn.v_proj.weight"), kd as u64 * h as u64 * BF16);
        let bv = self.w(&format!("{p}.self_attn.v_proj.bias"), kd as u64 * BF16);
        let wo = self.w(&format!("{p}.self_attn.o_proj.weight"), h as u64 * qd as u64 * BF16);
        let bo = self.w(&format!("{p}.self_attn.o_proj.bias"), h as u64 * BF16);
        // PLOW_MXFP4: decode reads the dense projections from `mxfp4/<name>` twins (e2m1 + E8M0,
        // quantize_mxfp4.py) through biased GEMV_MXFP4 (t7 = bias, CPU tiers); prefill keeps the bf16
        // GEMMs, so the bf16 weights above stay bound.
        let mx4 = emit_config::active().mxfp4 && !self.prefill;
        let w4 = |em: &mut Self, s: &str, out: u64, k: u64| -> (u32, u32) {
            if mx4 {
                (
                    em.w(&format!("mxfp4/{p}.self_attn.{s}.weight"), out * k / 2),
                    em.w(&format!("mxfp4/{p}.self_attn.{s}.weight_scale"), out * k / 32),
                )
            } else {
                (TENSOR_NONE, TENSOR_NONE)
            }
        };
        let (wq4, sq4) = w4(self, "q_proj", qd as u64, h as u64);
        let (wk4, sk4) = w4(self, "k_proj", kd as u64, h as u64);
        let (wv4, sv4) = w4(self, "v_proj", kd as u64, h as u64);
        let (wo4, so4) = w4(self, "o_proj", h as u64, qd as u64);
        let sinks = self.w(&format!("{p}.self_attn.sinks"), heads as u64 * BF16);
        let g_post = self.w(&format!("{p}.post_attention_layernorm.weight"), h as u64 * BF16);
        let full = c.is_full[l];
        let win = if full { 0 } else { c.window };
        let (kvr, kvm) = kv_ring(full, self.ctx, c.window, self.chunk);
        let kv_bytes = self.dbatch as u64 * kvh as u64 * kvr as u64 * hd as u64 * BF16;
        let kc = self.b.tensor(&format!("kv.{l}.k"), kv_bytes);
        let vc = self.b.tensor(&format!("kv.{l}.v"), kv_bytes);

        // q/k/v projections. Decode fuses the three into one GEMV sweep (op 22) with the three
        // bias handles in i5/i6/i7; prefill is three biased GEMMs.
        let (hn, qg, kg, vg) = (self.hn, self.qg, self.kg, self.vg);
        let (cq, ck, cv) = if self.prefill {
            (
                self.proj(qg, hn, wq, bq, qd, h, dep),
                self.proj(kg, hn, wk, bk, kd, h, dep),
                self.proj(vg, hn, wv, bv, kd, h, dep),
            )
        } else if mx4 {
            (
                self.proj_mx4(qg, hn, wq4, sq4, bq, qd, h, dep),
                self.proj_mx4(kg, hn, wk4, sk4, bk, kd, h, dep),
                self.proj_mx4(vg, hn, wv4, sv4, bv, kd, h, dep),
            )
        } else {
            let f = self.b.emit(DevOp::GemvQkv, self.b.all(), &[dep], |d| {
                d.t[0] = qg;
                d.t[1] = hn;
                d.t[2] = wq;
                d.t[3] = kg;
                d.t[4] = wk;
                d.t[5] = vg;
                d.t[6] = wv;
                d.i[0] = t;
                d.i[1] = qd;
                d.i[2] = h;
                d.i[3] = kd;
                d.i[4] = kd;
                d.i[5] = bq;
                d.i[6] = bk;
                d.i[7] = bv;
            });
            (f, f, f)
        };
        let q = self.q;
        let c_qn = self.rope(q, qg, heads, true, None, cq);
        let c_kn = self.rope(kc, kg, kvh, true, Some((kvr, kvm)), ck);
        let c_vn = self.rope(vc, vg, kvh, false, Some((kvr, kvm)), cv);
        if !self.prefill {
            self.kv_rows.push(c_kn);
            self.kv_rows.push(c_vn);
        }

        let (opart, mlpart, kvlen, ns, scale) = (self.opart, self.mlpart, self.kvlen, self.ns, c.attn_scale);
        let c_fa = if self.prefill {
            // t5 (O_final) stays NONE: the sinks fold lives in FLASH_MERGE only (dev_isa.h op 13).
            self.b.emit(DevOp::FlashPrefill, self.b.all(), &[c_qn, c_kn, c_vn], |d| {
                d.t[0] = opart;
                d.t[1] = mlpart;
                d.t[2] = q;
                d.t[3] = kc;
                d.t[4] = vc;
                d.i[0] = t;
                d.i[1] = t;
                d.i[2] = heads;
                d.i[3] = kvh;
                d.i[4] = 0;
                d.i[5] = win;
                d.i[6] = hd;
                d.i[7] = ns;
                d.f[0] = scale;
                d.j[0] = kvr;
                d.j[1] = kvm;
            })
        } else {
            self.b.emit(DevOp::FlashDecode, self.b.all(), &[c_qn, c_kn, c_vn], |d| {
                d.t[0] = opart;
                d.t[1] = mlpart;
                d.t[2] = q;
                d.t[3] = kc;
                d.t[4] = vc;
                d.t[5] = kvlen;
                d.i[0] = t;
                d.i[1] = heads;
                d.i[2] = kvh;
                d.i[3] = kvr;
                d.i[4] = win;
                d.i[5] = ns;
                d.i[6] = hd;
                d.i[7] = kvm;
                d.f[0] = scale;
                if t > 1 {
                    d.j[0] = t * kvh * kvr;
                }
            })
        };
        // Always emitted, even at nsplit == 1: the sink enters the softmax denominator here.
        let at = self.at;
        let mg: Vec<u32> = (0..(t * heads).min(self.b.n_cu()).max(1)).collect();
        let c_mg = self.b.emit(DevOp::FlashMerge, mg, &[c_fa], |d| {
            d.t[0] = at;
            d.t[1] = opart;
            d.t[2] = mlpart;
            d.t[3] = sinks;
            d.i[0] = t;
            d.i[1] = heads;
            d.i[2] = ns;
            d.i[3] = hd;
        });
        let og = self.og;
        let c_o = if mx4 {
            self.proj_mx4(og, at, wo4, so4, bo, h, qd, c_mg)
        } else {
            self.proj(og, at, wo, bo, h, qd, c_mg)
        };
        let c_n1 = self.add_norm(og, g_post, c_o);

        // MoE. Router logits carry the linear bias in the GEMV (MoeRouterTopk's t3 is DeepSeek's
        // selection-only bias); flat top-k, softmax over the selected.
        let (e, k, inter) = (c.n_exp, c.top_k, c.inter);
        let wr = self.w(&format!("{p}.mlp.router.weight"), e as u64 * h as u64 * BF16);
        let br = self.w(&format!("{p}.mlp.router.bias"), e as u64 * BF16);
        let w_gu = self.w(
            &format!("{p}.mlp.experts.gate_up_proj_blocks"),
            e as u64 * (2 * inter) as u64 * (h / 2) as u64,
        );
        let s_gu = self.w(
            &format!("{p}.mlp.experts.gate_up_proj_scales"),
            e as u64 * (2 * inter) as u64 * (h / 32) as u64,
        );
        let b_gu = self.w(&format!("{p}.mlp.experts.gate_up_proj_bias"), e as u64 * (2 * inter) as u64 * BF16);
        let w_d = self.w(
            &format!("{p}.mlp.experts.down_proj_blocks"),
            e as u64 * h as u64 * (inter / 2) as u64,
        );
        let s_d = self.w(
            &format!("{p}.mlp.experts.down_proj_scales"),
            e as u64 * h as u64 * (inter / 32) as u64,
        );
        let b_d = self.w(&format!("{p}.mlp.experts.down_proj_bias"), e as u64 * h as u64 * BF16);
        let (rlogit, tab, fu, part, moe) = (self.rlogit, self.tab, self.fu, self.part, self.moe);
        let c_r = self.proj(rlogit, hn, wr, br, e, h, c_n1);
        let c_tk = self.b.emit(DevOp::MoeRouterTopkPf, self.rows(), &[c_r], |d| {
            d.t[0] = tab;
            d.t[1] = rlogit;
            d.i[1] = e;
            d.i[2] = k;
            d.i[3] = ROUTER_FLAGS;
            d.i[4] = t;
            d.f[0] = 1.0;
        });
        let (alpha, limit) = (c.swiglu_alpha, c.swiglu_limit);
        let c_d = if self.prefill {
            let (meta, row_token, row_partidx, row_gate) =
                (self.meta, self.row_token, self.row_partidx, self.row_gate);
            let c_al = self.b.emit(DevOp::MoeAlignPf, vec![0], &[c_tk], |d| {
                d.t[0] = meta;
                d.t[1] = tab;
                d.t[2] = row_token;
                d.t[3] = row_partidx;
                d.t[4] = row_gate;
                d.i[0] = t;
                d.i[1] = e;
                d.i[2] = k;
            });
            let c_g = self.b.emit(DevOp::MoeGluMxPf, self.b.all(), &[c_al], |d| {
                d.t[0] = fu;
                d.t[1] = hn;
                d.t[2] = w_gu;
                d.t[3] = s_gu;
                d.t[4] = meta;
                d.t[5] = row_token;
                d.t[6] = b_gu;
                d.i[0] = inter;
                d.i[1] = h;
                d.i[2] = e;
                d.i[3] = 0; // interleaved gate/up rows, as the checkpoint ships them
                d.i[5] = ACT_SWIGLU_OAI;
                d.f[0] = alpha;
                d.f[1] = limit;
            });
            self.b.emit(DevOp::MoeDownMxPf, self.b.all(), &[c_g], |d| {
                d.t[0] = part;
                d.t[1] = fu;
                d.t[2] = w_d;
                d.t[3] = s_d;
                d.t[4] = meta;
                d.t[5] = b_d;
                d.t[6] = row_partidx;
                d.t[7] = row_gate;
                d.i[0] = h;
                d.i[1] = inter;
                d.i[2] = e;
            })
        } else {
            let c_g = self.b.emit(DevOp::MoeGluMx, self.b.all(), &[c_tk], |d| {
                d.t[0] = fu;
                d.t[1] = hn;
                d.t[2] = tab;
                d.t[3] = w_gu;
                d.t[4] = s_gu;
                d.t[5] = b_gu;
                d.i[0] = k;
                d.i[1] = inter;
                d.i[2] = h;
                d.i[3] = e;
                d.i[4] = 0;
                d.i[5] = ACT_SWIGLU_OAI;
                d.i[6] = t;
                d.f[0] = alpha;
                d.f[1] = limit;
            });
            self.b.emit(DevOp::MoeDownMx, self.b.all(), &[c_g], |d| {
                d.t[0] = part;
                d.t[1] = fu;
                d.t[2] = tab;
                d.t[3] = w_d;
                d.t[4] = s_d;
                d.t[5] = b_d;
                d.i[0] = k;
                d.i[1] = h;
                d.i[2] = inter;
                d.i[3] = e;
                d.i[6] = t;
            })
        };
        // Fixed-slot-order sum of the k gated partials; the residual add rides the AddNorm.
        let c_c = self.b.emit(DevOp::MoeCombinePf, self.rows(), &[c_d], |d| {
            d.t[0] = moe;
            d.t[3] = part;
            d.i[0] = h;
            d.i[1] = k;
            d.i[2] = t;
        });
        self.add_norm(moe, gamma_next, c_c)
    }
}

struct Phase {
    prog: packet::devbuild::Program,
    tensors: Vec<TensorDecl>,
    gen: Vec<GenTensor>,
    kv_rows: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn phase(
    c: &GptOssCfg,
    ctx: u32,
    n_cu: u32,
    t: u32,
    dbatch: u32,
    chunk: u32,
    prefill: bool,
    tensors: Option<Vec<TensorDecl>>,
) -> Phase {
    let mut b = Builder::new(n_cu);
    if let Some(tensors) = tensors {
        b.adopt_tensors(tensors);
    }
    b.set_tensor_dedup(true);
    let (h, heads, hd, inter, e, k) = (c.hidden, c.heads, c.hd, c.inter, c.n_exp, c.top_k);
    let (qd, kd) = (heads * hd, c.kvh * hd);
    // `in.ids` FIRST: handle 0 is the "absent" value of the GEMV_QKV bias slots (dev_isa.h op 22).
    let ids = b.tensor("in.ids", ctx as u64 * F32);
    let pos = b.tensor("in.pos", ctx as u64 * F32);
    let kvlen = b.tensor("in.kvlen", dbatch as u64 * F32);
    let [gc, gs] = GenTensor::rope_pair(ctx, hd, c.theta, 1.0, c.rope_scale);
    let cos = b.tensor_gen("in.cos_full", gc.byte_len(), gc);
    let sin = b.tensor_gen("in.sin_full", gs.byte_len(), gs);
    // Flash split count. Prefill tiles the q axis; decode fills the machine per head. GF (the
    // decode GQA head fusion) is a kernel constant the CPU tier does not apply.
    let ns = if prefill {
        n_cu.div_ceil((t.div_ceil(Q_TILE_ROWS) * heads).max(1)).max(1)
    } else {
        n_cu.div_ceil(heads).max(1)
    };
    let tr = t as u64;
    let act = |b: &mut Builder, n: &str, bytes: u64| b.tensor(&format!("act.{n}"), bytes);
    let x = act(&mut b, "x", tr * h as u64 * BF16);
    let hn = act(&mut b, "hn", tr * h as u64 * BF16);
    let qg = act(&mut b, "qg", tr * qd as u64 * BF16);
    let kg = act(&mut b, "kg", tr * kd as u64 * BF16);
    let vg = act(&mut b, "vg", tr * kd as u64 * BF16);
    let q = act(&mut b, "q", tr * qd as u64 * BF16);
    let opart = act(&mut b, "opart", tr * heads as u64 * ns as u64 * hd as u64 * F32);
    let mlpart = act(&mut b, "mlpart", tr * heads as u64 * ns as u64 * 2 * F32);
    let at = act(&mut b, "at", tr * qd as u64 * BF16);
    let og = act(&mut b, "og", tr * h as u64 * BF16);
    let rlogit = act(&mut b, "rlogit", tr * e as u64 * BF16);
    let tab = act(&mut b, "tab", tr * k as u64 * 8);
    // Gathered (prefill) or per-slot (decode) expert intermediate; one buffer, sized at the max.
    let rows_pad = tr * k as u64 + e as u64 * MOE_PAD_ROWS as u64;
    let fu_rows = if prefill { rows_pad } else { tr * k as u64 };
    let fu = act(&mut b, "fu", fu_rows * inter as u64 * BF16);
    let part = act(&mut b, "part", tr * k as u64 * h as u64 * F32);
    let moe = act(&mut b, "moe", tr * h as u64 * BF16);
    let meta = act(&mut b, "meta", (3 * e as u64 + 1) * F32);
    let row_token = act(&mut b, "row_token", rows_pad * F32);
    let row_partidx = act(&mut b, "row_partidx", rows_pad * F32);
    let row_gate = act(&mut b, "row_gate", rows_pad * F32);
    let logits = act(&mut b, "logits", dbatch as u64 * c.vocab as u64 * BF16);
    let amax = act(&mut b, "amax", dbatch as u64 * AMAX_BLOCKS as u64 * 8);
    let ladder = emit_config::active().decode_ladder_on();
    let mut em = Emitter {
        b,
        c,
        prefill,
        t,
        dbatch,
        ladder,
        ctx,
        chunk,
        ns,
        pos,
        kvlen,
        cos,
        sin,
        x,
        hn,
        qg,
        kg,
        vg,
        q,
        opart,
        mlpart,
        at,
        og,
        rlogit,
        tab,
        fu,
        part,
        moe,
        meta,
        row_token,
        row_partidx,
        row_gate,
        kv_rows: Vec::new(),
    };
    let emb = em.w(
        &format!("{}embed_tokens.weight", c.prefix),
        c.vocab as u64 * h as u64 * BF16,
    );
    let rows = em.rows();
    let mut dep = em.b.emit(DevOp::Embed, rows.clone(), &[], |d| {
        d.t[0] = x;
        d.t[1] = emb;
        d.t[2] = ids;
        d.i[0] = t;
        d.i[1] = h;
        d.f[0] = 1.0;
    });
    let g_in: Vec<u32> = (0..c.layers as usize)
        .map(|l| em.w(&format!("{}layers.{l}.input_layernorm.weight", c.prefix), h as u64 * BF16))
        .collect();
    let g_final = em.w(&format!("{}norm.weight", c.prefix), h as u64 * BF16);
    let eps = c.eps;
    dep = em.b.emit(DevOp::RmsNorm, rows.clone(), &[dep], |d| {
        d.t[0] = hn;
        d.t[1] = x;
        d.t[2] = g_in[0];
        d.i[0] = t;
        d.i[1] = h;
        d.f[0] = eps;
    });
    for l in 0..c.layers as usize {
        // The layer tail's AddNorm already applies the NEXT norm (or the final one).
        let gamma_next = g_in.get(l + 1).copied().unwrap_or(g_final);
        dep = em.layer(l, dep, gamma_next);
    }
    // Untied head over the normed residual (`hn` after the last tail). Prefill scores the LAST
    // row only (a_row0 = t-1); decode scores every sequence row.
    let head = if c.tied {
        format!("{}embed_tokens.weight", c.prefix)
    } else {
        "lm_head.weight".to_string()
    };
    let lm = em.w(&head, c.vocab as u64 * h as u64 * BF16);
    let (vocab, all) = (c.vocab, em.b.all());
    let head_mx4 = emit_config::active().mxfp4 && !prefill;
    dep = if head_mx4 {
        // PLOW_MXFP4 decode head: the 1.16 GB bf16 lm_head is ~1/4 of a GPT-OSS decode step's bytes.
        let lm4 = em.w(&format!("mxfp4/{head}"), c.vocab as u64 * h as u64 / 2);
        let ls4 = em.w(&format!("mxfp4/{head}_scale"), c.vocab as u64 * h as u64 / 32);
        em.b.emit(DevOp::GemvMxfp4, all, &[dep], |d| {
            d.t[0] = logits;
            d.t[1] = hn;
            d.t[2] = lm4;
            d.t[3] = ls4;
            d.i[0] = t;
            d.i[1] = vocab;
            d.i[2] = h;
        })
    } else {
        em.b.emit(DevOp::Gemv, all, &[dep], |d| {
            d.t[0] = logits;
            d.t[1] = hn;
            d.t[2] = lm;
            d.i[0] = if prefill { 1 } else { t };
            d.i[1] = vocab;
            d.i[2] = h;
            d.i[4] = if prefill { t - 1 } else { 0 };
        })
    };
    let nb = if !prefill && t > 1 { t } else { 0 };
    dep = em.b.emit(DevOp::Argmax, (0..AMAX_BLOCKS).collect(), &[dep], |d| {
        d.t[0] = amax;
        d.t[1] = logits;
        d.i[0] = vocab;
        d.i[1] = nb;
    });
    em.b.emit(DevOp::ArgmaxFin, vec![0], &[dep], |d| {
        d.t[0] = ids;
        d.t[1] = amax;
        d.i[0] = AMAX_BLOCKS;
        d.i[1] = nb;
    });
    let tensors = em.b.tensors();
    let gen = em.b.gen_tensors();
    Phase {
        prog: em.b.finish(),
        tensors,
        gen,
        kv_rows: em.kv_rows,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    dir: &Path,
    ctx: u32,
    out: &str,
    n_cu: u32,
    tp: u32,
    block: Option<&str>,
    rope_gen: bool,
    arch: &str,
    gpu: &str,
    verify: Option<&VerifyHook>,
) {
    assert_eq!(tp, 1, "gpt_oss: tensor parallelism is not implemented");
    assert!(block.is_none(), "gpt_oss: --block extraction is not implemented");
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("gpt_oss config JSON");
    let c = cfg_gpt_oss(&v);
    let ecfg = emit_config::active();
    let rungs = ecfg.decode_rungs();
    let dbatch = *rungs.last().expect("decode_rungs is non-empty");
    let chunk = ecfg.max_chunk.unwrap_or(DEFAULT_CHUNK.min(ctx));
    assert!(
        chunk.is_power_of_two() && chunk <= MAX_CHUNK_MAX && chunk <= ctx,
        "gpt_oss: prefill chunk {chunk} must be a power of two <= min(ctx {ctx}, {MAX_CHUNK_MAX})"
    );
    let buckets: Vec<u32> = [128u32, 512, 1024, 2048, 4096, 8192]
        .into_iter()
        .filter(|&x| x <= chunk)
        .collect();
    assert!(
        rungs.iter().all(|&r| buckets.iter().all(|&pb| pb > r)),
        "gpt_oss: decode rungs {rungs:?} overlap the prefill buckets {buckets:?} \
         (`decode_rung_lo` separates them by width)"
    );
    let target = if gpu.is_empty() {
        0
    } else {
        packet::devbuild::gpu_fingerprint(gpu)
    };
    let _guard = EmitAmdGuard::set(target_is_amd(arch, gpu));

    let mut tensors: Option<Vec<TensorDecl>> = None;
    let mut progs = Vec::new();
    let mut prog_t = Vec::new();
    let mut gen = Vec::new();
    let mut kv_rows = Vec::new();
    for &t in &buckets {
        let ph = phase(&c, ctx, n_cu, t, dbatch, chunk, true, tensors.take());
        tensors = Some(ph.tensors);
        gen = ph.gen;
        progs.push(ph.prog);
        prog_t.push(t);
    }
    for &rb in &rungs {
        let ph = phase(&c, ctx, n_cu, rb, dbatch, chunk, false, tensors.take());
        tensors = Some(ph.tensors);
        gen = ph.gen;
        // The KV-append sites the host patches index the LAST (widest) program.
        kv_rows = ph.kv_rows;
        progs.push(ph.prog);
        prog_t.push(rb);
    }
    let mut m = Model {
        n_cu,
        target,
        tensors: tensors.expect("at least one program"),
        progs,
        kv_row_insts: kv_rows,
        prog_t,
        gen,
    };
    // Forward: every declared weight exists with the declared byte size (the loader re-checks
    // sizes). Reverse: a checkpoint tensor nothing declares is an op this emitter lacks — refuse.
    validate_coverage(
        dir,
        &c.prefix,
        &m.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        None,
        &[],
        &[],
        &[],
    )
    .unwrap_or_else(|e| panic!("gpt_oss: {e}"));
    if !rope_gen {
        m.bake_gen();
    }
    let lean = apply_verify_gate(&m, verify);
    let outp = Path::new(out);
    std::fs::write(outp, m.to_blob()).expect("write gpt_oss blob");
    if !arch.is_empty() {
        let man = manifest::build(&m, arch, &lean);
        manifest::write_config_header(&outp.with_file_name("plow_config.h"), &man)
            .expect("gpt_oss config header");
        std::fs::write(
            outp.with_file_name("build.json"),
            serde_json::to_vec_pretty(&man).unwrap(),
        )
        .expect("gpt_oss build manifest");
    }
    // Ops 147-150 (and the bias/sinks/pair_mode fields) have CPU-tier arms only. The gfx950
    // dispatch-coverage gate is not applied: this blob is not a GPU asset.
    eprintln!(
        "gpt_oss: {} layers ({} full)  hidden={} inter={} heads={} kvh={} hd={} experts={}x top{} \
         vocab={}\n  max_ctx={} chunk={} prefill buckets {:?} + decode rungs {:?} -> {}\n  \
         NOTE: MoE ops 147-150 + bias/sinks fields are CPU-tier only (no gfx950/sm arm).",
        c.layers,
        c.is_full.iter().filter(|x| **x).count(),
        c.hidden,
        c.inter,
        c.heads,
        c.kvh,
        c.hd,
        c.n_exp,
        c.top_k,
        c.vocab,
        ctx,
        chunk,
        buckets,
        rungs,
        out
    );
    for (i, p) in m.progs.iter().enumerate() {
        eprintln!(
            "    prog {} (T={:>4}): {:>5} packets, {:>7} workgroup-packets",
            i,
            m.prog_t[i],
            p.insts.len(),
            p.stream.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Value {
        serde_json::json!({
            "model_type":"gpt_oss","attention_bias":true,"hidden_act":"silu",
            "head_dim":64,"hidden_size":2880,"intermediate_size":2880,
            "layer_types":["sliding_attention","full_attention"],
            "num_attention_heads":64,"num_key_value_heads":8,"num_hidden_layers":2,
            "num_local_experts":32,"num_experts_per_tok":4,
            "quantization_config":{"quant_method":"mxfp4"},
            "rms_norm_eps":1e-5,
            "rope_scaling":{"beta_fast":32.0,"beta_slow":1.0,"factor":32.0,
                "original_max_position_embeddings":4096,"rope_type":"yarn","truncate":false},
            "rope_theta":150000,"sliding_window":128,"swiglu_limit":7.0,
            "tie_word_embeddings":false,"vocab_size":201088
        })
    }

    /// The contracts the kernel agent reads: op set, bias/sinks slots, half-split RoPE, MXFP4
    /// tensor byte sizes equal to the checkpoint's, `in.ids` at handle 0.
    #[test]
    fn decode_program_matches_the_isa_contracts() {
        let _env = crate::test_env::env_guard();
        let c = cfg_gpt_oss(&fixture());
        let ph = phase(&c, 4096, 16, 1, 1, 1024, false, None);
        assert_eq!(ph.tensors[0].name, "in.ids");
        let name = |h: u32| ph.tensors[h as usize].name.clone();
        let bytes = |n: &str| ph.tensors.iter().find(|t| t.name == n).unwrap().bytes;
        assert_eq!(bytes("model.layers.0.mlp.experts.gate_up_proj_blocks"), 32 * 5760 * 90 * 16);
        assert_eq!(bytes("model.layers.0.mlp.experts.gate_up_proj_scales"), 32 * 5760 * 90);
        assert_eq!(bytes("model.layers.0.mlp.experts.down_proj_blocks"), 32 * 2880 * 90 * 16);
        assert_eq!(bytes("model.layers.0.mlp.experts.gate_up_proj_bias"), 32 * 5760 * 2);
        assert_eq!(bytes("model.layers.0.self_attn.sinks"), 64 * 2);
        assert_eq!(bytes("lm_head.weight"), 201088 * 2880 * 2);
        // Sliding layer 0 rings (128 + 1024 - 1 -> 2048 rows), full layer 1 is linear (4096).
        assert_eq!(bytes("kv.0.k"), 8 * 2048 * 64 * 2);
        assert_eq!(bytes("kv.1.k"), 8 * 4096 * 64 * 2);
        assert_eq!(ph.kv_rows.len(), 4, "k and v append sites per layer");
        let ops: Vec<DevOp> = ph.prog.insts.iter().map(|d| DevOp::from_u16(d.op).unwrap()).collect();
        for want in [
            DevOp::Embed, DevOp::RmsNorm, DevOp::GemvQkv, DevOp::HeadNormRope, DevOp::FlashDecode,
            DevOp::FlashMerge, DevOp::Gemv, DevOp::AddNorm, DevOp::MoeRouterTopkPf, DevOp::MoeGluMx,
            DevOp::MoeDownMx, DevOp::MoeCombinePf, DevOp::Argmax, DevOp::ArgmaxFin,
        ] {
            assert!(ops.contains(&want), "missing {want:?}");
        }
        assert!(!ops.contains(&DevOp::MoeAlignPf) && !ops.contains(&DevOp::MoeGluMxPf));
        for d in &ph.prog.insts {
            match DevOp::from_u16(d.op).unwrap() {
                DevOp::GemvQkv => {
                    assert!(name(d.i[5]).ends_with("q_proj.bias"));
                    assert!(name(d.i[6]).ends_with("k_proj.bias"));
                    assert!(name(d.i[7]).ends_with("v_proj.bias"));
                    assert_eq!(d.t[7], TENSOR_NONE);
                }
                DevOp::HeadNormRope => {
                    assert_eq!(d.i[4], 1, "no qk-norm");
                    if d.t[3] != TENSOR_NONE {
                        assert_eq!(d.i[5], ROPE_PAIR_HALF);
                    }
                    if d.j[0] != 0 {
                        assert_eq!(d.i[6], 0, "no ladder in this fixture: legacy row patch");
                    }
                }
                DevOp::FlashMerge => assert!(name(d.t[3]).ends_with("self_attn.sinks")),
                DevOp::FlashDecode => {
                    assert_eq!(d.i[6], 64);
                    assert_eq!(d.f[0], 0.125);
                }
                DevOp::Gemv => {
                    if name(d.t[2]).ends_with("o_proj.weight") {
                        assert!(name(d.t[7]).ends_with("o_proj.bias"));
                    }
                    if name(d.t[2]).ends_with("router.weight") {
                        assert!(name(d.t[7]).ends_with("router.bias"));
                        assert_eq!(d.i[1], 32);
                    }
                    if name(d.t[2]) == "lm_head.weight" {
                        assert_eq!(d.t[7], TENSOR_NONE);
                    }
                }
                DevOp::MoeGluMx => {
                    assert_eq!(&d.i[..7], &[4, 2880, 2880, 32, 0, ACT_SWIGLU_OAI, 1]);
                    assert_eq!((d.f[0], d.f[1]), (1.702, 7.0));
                    assert!(name(d.t[5]).ends_with("gate_up_proj_bias"));
                }
                DevOp::MoeDownMx => {
                    assert_eq!(&d.i[..4], &[4, 2880, 2880, 32]);
                    assert!(name(d.t[5]).ends_with("down_proj_bias"));
                }
                DevOp::MoeRouterTopkPf => {
                    assert_eq!((d.i[1], d.i[2], d.i[3], d.i[4]), (32, 4, ROUTER_FLAGS, 1));
                    assert_eq!(d.t[3], TENSOR_NONE);
                }
                _ => {}
            }
        }
        // Window: even layer sliding (128), odd layer full (0).
        let wins: Vec<u32> = ph
            .prog
            .insts
            .iter()
            .filter(|d| d.op == DevOp::FlashDecode as u16)
            .map(|d| d.i[4])
            .collect();
        assert_eq!(wins, [128, 0]);
    }

    /// Under a decode ladder every rung — the one-row rung included — addresses the KV cache
    /// per sequence out of `pos[]` (`i6 = n_batch_kv`), and the widest rung declares the KV
    /// row-capacity bound on the flash. The k/v writers are the recorded KV-append sites.
    #[test]
    fn decode_ladder_arms_per_sequence_kv_addressing_at_every_rung() {
        let _env = crate::test_env::env_guard();
        struct Restore(emit_config::EmitConfig);
        impl Drop for Restore {
            fn drop(&mut self) {
                emit_config::install(self.0.clone());
            }
        }
        let restore = Restore(emit_config::active().clone());
        let mut cfg = restore.0.clone();
        cfg.decode_ladder = Some("1,2,4,8".into());
        emit_config::install(cfg);
        let c = cfg_gpt_oss(&fixture());
        for rung in [1u32, 8] {
            let ph = phase(&c, 4096, 16, rung, 8, 1024, false, None);
            let kv_write = |d: &packet::dev::DevInst| {
                d.op == DevOp::HeadNormRope as u16 && d.j[0] != 0
            };
            let writers: Vec<_> = ph.prog.insts.iter().filter(|d| kv_write(d)).collect();
            assert_eq!(writers.len(), 4);
            for d in &writers {
                assert_eq!(d.i[6], rung, "rung {rung}: n_batch_kv");
                assert_eq!(d.i[0], rung);
            }
            let sites: Vec<_> = ph
                .kv_rows
                .iter()
                .map(|&i| &ph.prog.insts[i as usize])
                .collect();
            assert!(sites.iter().all(|d| kv_write(d)));
            for d in ph.prog.insts.iter().filter(|d| d.op == DevOp::FlashDecode as u16) {
                assert_eq!(d.i[0], rung);
                assert_eq!(d.j[0], if rung > 1 { rung * 8 * d.i[3] } else { 0 });
            }
            for op in [DevOp::MoeGluMx, DevOp::MoeDownMx] {
                let d = ph.prog.insts.iter().find(|d| d.op == op as u16).unwrap();
                assert_eq!(d.i[6], rung, "{op:?} n_batch");
            }
            let tk = ph.prog.insts.iter().find(|d| d.op == DevOp::MoeRouterTopkPf as u16).unwrap();
            assert_eq!(tk.i[4], rung);
            let am = ph.prog.insts.iter().find(|d| d.op == DevOp::Argmax as u16).unwrap();
            assert_eq!(am.i[1], if rung > 1 { rung } else { 0 });
        }
    }

    #[test]
    fn prefill_program_uses_the_gathered_twins_and_no_direct_flash_write() {
        let _env = crate::test_env::env_guard();
        let c = cfg_gpt_oss(&fixture());
        let ph = phase(&c, 4096, 16, 128, 8, 1024, true, None);
        let ops: Vec<DevOp> = ph.prog.insts.iter().map(|d| DevOp::from_u16(d.op).unwrap()).collect();
        for want in [
            DevOp::Gemm, DevOp::FlashPrefill, DevOp::FlashMerge, DevOp::MoeRouterTopkPf,
            DevOp::MoeAlignPf, DevOp::MoeGluMxPf, DevOp::MoeDownMxPf, DevOp::MoeCombinePf,
        ] {
            assert!(ops.contains(&want), "missing {want:?}");
        }
        assert!(!ops.contains(&DevOp::GemvQkv) && !ops.contains(&DevOp::MoeGluMx));
        assert!(ph.kv_rows.is_empty());
        for d in &ph.prog.insts {
            match DevOp::from_u16(d.op).unwrap() {
                DevOp::FlashPrefill => assert_eq!(d.t[5], TENSOR_NONE, "sinks fold at the merge"),
                DevOp::Gemm => {
                    if d.i[1] != 201088 {
                        assert_ne!(d.t[7], TENSOR_NONE, "every layer projection is biased");
                    }
                }
                DevOp::MoeGluMxPf => {
                    assert_eq!((d.i[0], d.i[1], d.i[2], d.i[3], d.i[5]), (2880, 2880, 32, 0, ACT_SWIGLU_OAI));
                }
                DevOp::MoeCombinePf => assert_eq!(d.i[2], 128),
                _ => {}
            }
        }
        // Per-slot tensors are sized at the widest rung even in a prefill program.
        let bytes = |n: &str| ph.tensors.iter().find(|t| t.name == n).unwrap().bytes;
        assert_eq!(bytes("in.kvlen"), 8 * 4);
        assert_eq!(bytes("kv.1.v"), 8 * 8 * 4096 * 64 * 2);
        assert_eq!(bytes("act.fu"), (128 * 4 + 32 * 128) * 2880 * 2);
    }
}
