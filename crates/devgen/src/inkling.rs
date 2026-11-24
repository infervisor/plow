//! Inkling-2-Mini emitter (`model_type: "inkling2_loop"` / `"inkling2_loop_text"`).
//!
//! Per layer: `attn_norm -> q/k/v -> NoPE (skip_norm, no cos/sin) -> flash (sliding 1024 on local
//! layers, full on global) + FLASH_MERGE -> o_proj -> AddNorm(mlp_norm) -> router GEMV
//! (+selection bias in t3) -> top-8 sigmoid renorm -> MXFP4 experts (swiglu_oai, interleaved) -> combine ->
//! AddNorm`.
//!
//! One `Builder` per program: prefill buckets ascending, then decode batch ladder ascending,
//! all sharing one tensor table.

use super::*;
use packet::dev::ACT_SWIGLU_OAI;
use packet::devbuild::TensorDecl;
use std::path::Path;

const DEFAULT_CHUNK: u32 = 8192;
/// ROUTER_FLAGS: bit 0 = sigmoid, bit 1 = renormalize top-k, bit 2 = selection bias from TEN(3).
const ROUTER_FLAGS: u32 = 1 | 2 | 4;
const BF16: u64 = 2;
const F32: u64 = 4;

#[derive(Debug, Clone)]
pub(crate) struct InklingCfg {
    pub hidden: u32,
    pub inter: u32,
    pub layers: u32,
    pub heads: u32,
    pub kvh: u32,
    pub hd: u32,
    pub n_exp: u32,
    pub top_k: u32,
    pub vocab: u32,
    pub window: u32,
    pub swiglu_alpha: f32,
    pub swiglu_limit: f32,
    pub route_scale: f32,
    pub eps: f32,
    pub attn_scale: f32,
    pub prefix: String,
    pub is_full: Vec<bool>,
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
}

fn cfg_inkling(v: &Value) -> InklingCfg {
    let t = if v.get("text_config").is_some() {
        &v["text_config"]
    } else {
        v
    };
    let g = |k: &str| t[k].as_u64().unwrap_or_else(|| panic!("missing {k}")) as u32;
    let hidden = g("hidden_size");
    let inter = g("intermediate_size");
    let layers = g("num_hidden_layers");
    let heads = g("num_attention_heads");
    let kvh = g("num_key_value_heads");
    let hd = g("head_dim");
    let n_exp = g("n_routed_experts");
    let top_k = g("num_experts_per_tok");
    let vocab = g("vocab_size");
    let window = t
        .get("sliding_window_size")
        .and_then(|x| x.as_u64())
        .unwrap_or(1024) as u32;
    let swiglu_alpha = 1.0;
    let swiglu_limit = t
        .get("mlp_gate_softcap")
        .and_then(|x| x.as_f64())
        .unwrap_or(4.0) as f32;
    let route_scale = t
        .get("route_scale")
        .and_then(|x| x.as_f64())
        .unwrap_or(8.0) as f32;
    let eps = t
        .get("rms_norm_eps")
        .and_then(|x| x.as_f64())
        .unwrap_or(1e-6) as f32;
    let attn_scale = 1.0 / (hd as f32).sqrt();
    let prefix = "model.llm.".to_string();
    let local_layers: Vec<u32> = t
        .get("local_layer_ids")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|e| e.as_u64().map(|x| x as u32))
                .collect()
        })
        .unwrap_or_default();
    let is_full: Vec<bool> = (0..layers)
        .map(|l| !local_layers.contains(&l))
        .collect();
    let (core_start, core_end, num_passes) = t
        .get("loop_config")
        .map(|lc| {
            (
                lc.get("core_start").and_then(|x| x.as_u64()).unwrap_or(0) as usize,
                lc.get("core_end").and_then(|x| x.as_u64()).unwrap_or(layers as u64) as usize,
                lc.get("num_passes").and_then(|x| x.as_u64()).unwrap_or(1) as usize,
            )
        })
        .unwrap_or((0, layers as usize, 1));
    InklingCfg {
        hidden,
        inter,
        layers,
        heads,
        kvh,
        hd,
        n_exp,
        top_k,
        vocab,
        window,
        swiglu_alpha,
        swiglu_limit,
        route_scale,
        eps,
        attn_scale,
        prefix,
        is_full,
        core_start,
        core_end,
        num_passes,
    }
}

struct Emitter<'a> {
    b: Builder,
    c: &'a InklingCfg,
    prefill: bool,
    t: u32,
    dbatch: u32,
    ladder: bool,
    ctx: u32,
    chunk: u32,
    ns: u32,
    pos: u32,
    kvlen: u32,
    x: u32,
    hn: u32,
    kg: u32,
    vg: u32,
    q: u32,
    opart: u32,
    mlpart: u32,
    mrgc: u32,
    at: u32,
    og: u32,
    rlogit: u32,
    tab: u32,
    fu: u32,
    part: u32,
    moe: u32,
    kv_rows: Vec<u32>,
}

impl Emitter<'_> {
    fn w(&mut self, name: &str, bytes: u64) -> u32 {
        self.b.tensor(name, bytes)
    }

    fn rows(&self) -> Vec<u32> {
        (0..self.t.min(self.b.n_cu()).max(1)).collect()
    }

    fn proj(&mut self, out: u32, src: u32, w: u32, bias: u32, n: u32, k: u32, dep: u32) -> u32 {
        let t = self.t;
        let op = if !self.prefill {
            DevOp::Gemv
        } else if crate::emit_is_apple() {
            crate::apple_prefill_tile(t, n, self.b.n_cu(), kernelcaps::QuantScheme::None)
        } else {
            DevOp::Gemm
        };
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

    fn add_norm_moe_fold(&mut self, part: u32, gamma: u32, k: u32, dep: u32) -> u32 {
        let (x, hn, t, h, eps) = (self.x, self.hn, self.t, self.c.hidden, self.c.eps);
        self.b.emit(DevOp::AddNorm, self.rows(), &[dep], |d| {
            d.t[0] = hn;
            d.t[1] = x;
            d.t[2] = x;
            d.t[3] = part;
            d.t[4] = gamma;
            d.i[0] = t;
            d.i[1] = h;
            d.i[7] = k;
            d.f[0] = eps;
        })
    }

    fn rope(
        &mut self,
        out: u32,
        src: u32,
        nhead: u32,
        cache: Option<(u32, u32)>,
        dep: u32,
    ) -> u32 {
        let (pos, t, hd, eps) = (self.pos, self.t, self.c.hd, self.c.eps);
        let batch_kv = if !self.prefill && (t > 1 || self.ladder) {
            t
        } else {
            0
        };
        self.b.emit(DevOp::HeadNormRope, self.b.all(), &[dep], |d| {
            d.t[0] = out;
            d.t[1] = src;
            d.t[5] = pos;
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

    fn layer(&mut self, l: usize, pass_idx: usize, dep: u32, gamma_next: u32) -> u32 {
        let c = self.c;
        let (t, h, hd, heads, kvh) = (self.t, c.hidden, c.hd, c.heads, c.kvh);
        let (qd, kd) = (heads * hd, kvh * hd);
        let p = format!("{}layers.{l}.", c.prefix);

        let wq = self.w(&format!("{p}attn.wq_du.weight"), qd as u64 * h as u64 * BF16);
        let wk = self.w(&format!("{p}attn.wk_dv.weight"), kd as u64 * h as u64 * BF16);
        let wv = self.w(&format!("{p}attn.wv_dv.weight"), kd as u64 * h as u64 * BF16);
        let wo = self.w(&format!("{p}attn.wo_ud.weight"), h as u64 * qd as u64 * BF16);
        let g_mlp = self.w(&format!("{p}mlp_norm.weight"), h as u64 * BF16);

        let full = c.is_full[l];
        let win = if full { 0 } else { c.window };
        let (kvr, kvm) = kv_ring(full, self.ctx, c.window, self.chunk);
        let kv_bytes = self.dbatch as u64 * kvh as u64 * kvr as u64 * hd as u64 * BF16;
        let kc = self.b.tensor(&format!("kv.{pass_idx}.k"), kv_bytes);
        let vc = self.b.tensor(&format!("kv.{pass_idx}.v"), kv_bytes);

        // q/k/v projections. Decode fuses into GemvQkv when t == 1 and not under a ladder; batched decode and prefill use Gemm/GemmSmall.
        let (hn, kg, vg) = (self.hn, self.kg, self.vg);
        let q = self.q;
        let (cq, ck, cv) = if self.prefill || t > 1 || self.ladder {
            (
                self.proj(q, hn, wq, TENSOR_NONE, qd, h, dep),
                self.proj(kg, hn, wk, TENSOR_NONE, kd, h, dep),
                self.proj(vg, hn, wv, TENSOR_NONE, kd, h, dep),
            )
        } else {
            let f = self.b.emit(DevOp::GemvQkv, self.b.all(), &[dep], |d| {
                d.t[0] = q;
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
            });
            (f, f, f)
        };

        // Q has NoPE (skip_norm, no cos/sin), so projection writes directly to q.
        let c_qn = cq;
        let fuse_kv = !self.prefill && t == 1 && !self.ladder;
        let (c_kn, c_vn) = if fuse_kv {
            // Fuse K and V cache writes into a single HeadNormRope pair instruction (i[7] = 1).
            let (pos, hd, eps) = (self.pos, self.c.hd, self.c.eps);
            let c_kv = self.b.emit(DevOp::HeadNormRope, self.b.all(), &[ck], |d| {
                d.t[0] = kc;
                d.t[1] = kg;
                d.t[5] = pos;
                d.t[6] = vc;
                d.t[7] = vg;
                d.i[0] = t;
                d.i[1] = kvh;
                d.i[2] = hd;
                d.i[3] = 0;
                d.i[4] = 1; // skip_norm
                d.i[7] = 1; // fused KV pair (writes V cache from t6/t7)
                d.f[0] = eps;
                d.j[0] = kvr;
                d.j[1] = kvm;
            });
            (c_kv, c_kv)
        } else {
            (
                self.rope(kc, kg, kvh, Some((kvr, kvm)), ck),
                self.rope(vc, vg, kvh, Some((kvr, kvm)), cv),
            )
        };
        if !self.prefill {
            self.kv_rows.push(c_kn);
            if !fuse_kv {
                self.kv_rows.push(c_vn);
            }
        }

        let (opart, mlpart, kvlen, ns, scale) =
            (self.opart, self.mlpart, self.kvlen, self.ns, c.attn_scale);
        let merge_fold = !self.prefill
            && self.mrgc != TENSOR_NONE
            && crate::emit_config::active().nv_fa_fold
            && !crate::emit_is_amd()
            && !crate::emit_is_apple();
        let at = self.at;
        let c_fa = if self.prefill {
            self.b.emit(
                DevOp::FlashPrefill,
                self.b.all(),
                &[c_qn, c_kn, c_vn],
                |d| {
                    d.t[0] = opart;
                    d.t[1] = mlpart;
                    d.t[2] = q;
                    d.t[3] = kc;
                    d.t[4] = vc;
                    d.t[5] = at;
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
                },
            )
        } else {
            let fa_deps = if fuse_kv {
                vec![c_qn, c_kn]
            } else {
                vec![c_qn, c_kn, c_vn]
            };
            self.b
                .emit(DevOp::FlashDecode, self.b.all(), &fa_deps, |d| {
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
                    if t > 1 || self.ladder {
                        d.j[0] = t * kvh * kvr;
                    }
                    if merge_fold {
                        assert!(self.at < (1 << 16) && self.mrgc < (1 << 16));
                        d.j[1] = self.at | (self.mrgc << 16);
                    }
                })
        };

        let at = self.at;
        let c_attn_out = if merge_fold {
            c_fa
        } else {
            let mg: Vec<u32> = (0..(t * heads).min(self.b.n_cu()).max(1)).collect();
            self.b.emit(DevOp::FlashMerge, mg, &[c_fa], |d| {
                d.t[0] = at;
                d.t[1] = opart;
                d.t[2] = mlpart;
                d.t[3] = TENSOR_NONE;
                d.i[0] = t;
                d.i[1] = heads;
                d.i[2] = ns;
                d.i[3] = hd;
            })
        };
        let og = self.og;
        let c_o = self.proj(og, at, wo, TENSOR_NONE, h, qd, c_attn_out);
        let c_n1 = self.add_norm(og, g_mlp, c_o);

        // MoE
        let is_fp8 = l >= 20;
        let quant_mode = if is_fp8 { 1u32 } else { 2u32 };
        let (e, k, inter) = (c.n_exp, c.top_k, c.inter);
        let wr = self.w(&format!("{p}mlp.gate.weight"), e as u64 * h as u64 * BF16);
        let br = self.w(&format!("{p}mlp.gate.bias"), e as u64 * F32);
        let w_gu = self.w(
            &format!("{p}mlp.experts.w13_weight"),
            e as u64 * (2 * inter) as u64 * (if is_fp8 { h } else { h / 2 }) as u64,
        );
        let s_gu = self.w(
            &format!("{p}mlp.experts.w13_weight_scale"),
            e as u64 * (2 * inter) as u64 * (if is_fp8 { h / 32 } else { h / 16 }) as u64,
        );
        let w_d = self.w(
            &format!("{p}mlp.experts.w2_weight"),
            e as u64 * h as u64 * (if is_fp8 { inter } else { inter / 2 }) as u64,
        );
        let s_d = self.w(
            &format!("{p}mlp.experts.w2_weight_scale"),
            e as u64 * h as u64 * (if is_fp8 { inter / 32 } else { inter / 16 }) as u64,
        );

        let (rlogit, tab, fu, part, moe) = (self.rlogit, self.tab, self.fu, self.part, self.moe);
        let c_tk = if !self.prefill && self.t == 1 && !self.ladder && crate::emit_config::active().nv_moe_router_fold {
            self.b.emit(DevOp::MoeRouter, vec![0], &[c_n1], |d| {
                d.t[0] = tab;
                d.t[1] = hn;
                d.t[2] = wr;
                d.t[3] = br;
                d.i[0] = h;
                d.i[1] = e;
                d.i[2] = k;
                d.i[3] = ROUTER_FLAGS;
                d.f[0] = c.route_scale;
            })
        } else {
            let c_r = self.proj(rlogit, hn, wr, TENSOR_NONE, e, h, c_n1);
            self.b.emit(DevOp::MoeRouterTopkPf, self.rows(), &[c_r], |d| {
                d.t[0] = tab;
                d.t[1] = rlogit;
                d.t[3] = br;
                d.i[1] = e;
                d.i[2] = k;
                d.i[3] = ROUTER_FLAGS;
                d.i[4] = t;
                d.f[0] = c.route_scale;
            })
        };

        let (alpha, limit) = (c.swiglu_alpha, c.swiglu_limit);
        let c_g = self.b.emit(DevOp::MoeGluMx, self.b.all(), &[c_tk], |d| {
            d.t[0] = fu;
            d.t[1] = hn;
            d.t[2] = tab;
            d.t[3] = w_gu;
            d.t[4] = s_gu;
            d.t[5] = TENSOR_NONE;
            d.i[0] = k;
            d.i[1] = inter;
            d.i[2] = h;
            d.i[3] = e;
            d.i[4] = 0; // interleaved gate/up rows
            d.i[5] = ACT_SWIGLU_OAI;
            d.i[6] = t;
            d.i[7] = quant_mode;
            d.f[0] = alpha;
            d.f[1] = limit;
        });
        let c_d = self.b.emit(DevOp::MoeDownMx, self.b.all(), &[c_g], |d| {
            d.t[0] = part;
            d.t[1] = fu;
            d.t[2] = tab;
            d.t[3] = w_d;
            d.t[4] = s_d;
            d.t[5] = TENSOR_NONE;
            d.i[0] = k;
            d.i[1] = h;
            d.i[2] = inter;
            d.i[3] = e;
            d.i[6] = t;
            d.i[7] = quant_mode;
        });

        if crate::emit_config::active().nv_moe_combine_fold {
            self.add_norm_moe_fold(part, gamma_next, k, c_d)
        } else {
            let c_c = self.b.emit(DevOp::MoeCombinePf, self.b.all(), &[c_d], |d| {
                d.t[0] = moe;
                d.t[3] = part;
                d.i[0] = h;
                d.i[1] = k;
                d.i[2] = t;
            });

            self.add_norm(moe, gamma_next, c_c)
        }
    }
}

struct Phase {
    prog: packet::devbuild::Program,
    tensors: Vec<TensorDecl>,
    gen: Vec<GenTensor>,
    kv_rows: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
fn phase_range(
    c: &InklingCfg,
    ctx: u32,
    n_cu: u32,
    t: u32,
    dbatch: u32,
    chunk: u32,
    prefill: bool,
    tensors: Option<Vec<TensorDecl>>,
    layers: std::ops::Range<usize>,
) -> Phase {
    let mut b = Builder::new(n_cu);
    b.force_uniseg();
    let block_mode = layers.start != 0 || layers.end != c.layers as usize;
    if let Some(tensors) = tensors {
        b.adopt_tensors(tensors);
    }
    b.set_tensor_dedup(true);
    let (h, heads, hd, inter, e, k) = (c.hidden, c.heads, c.hd, c.inter, c.n_exp, c.top_k);
    let (qd, kd) = (heads * hd, c.kvh * hd);

    let ids = b.tensor("in.ids", ctx as u64 * F32);
    let pos = b.tensor("in.pos", ctx as u64 * F32);
    let kvlen = b.tensor("in.kvlen", dbatch as u64 * F32);

    let ns = if prefill {
        n_cu.div_ceil((t.div_ceil(Q_TILE_ROWS) * heads).max(1))
            .max(2)
    } else {
        n_cu.div_ceil(heads).max(1)
    };
    let tr = t as u64;
    let act = |b: &mut Builder, n: &str, bytes: u64| b.tensor(&format!("act.{n}"), bytes);
    let x = act(&mut b, "x", tr * h as u64 * BF16);
    let hn = act(&mut b, "hn", tr * h as u64 * BF16);
    let kg = act(&mut b, "kg", tr * kd as u64 * BF16);
    let vg = act(&mut b, "vg", tr * kd as u64 * BF16);
    let q = act(&mut b, "q", tr * qd as u64 * BF16);
    let opart = act(
        &mut b,
        "opart",
        tr * heads as u64 * ns as u64 * hd as u64 * F32,
    );
    let mlpart = act(&mut b, "mlpart", tr * heads as u64 * ns as u64 * 2 * F32);
    let mrgc = if !prefill && crate::emit_config::active().nv_fa_fold && !crate::emit_is_amd() && !crate::emit_is_apple() {
        act(&mut b, "mrgc", (dbatch.max(1) * heads) as u64 * 4)
    } else {
        TENSOR_NONE
    };
    let at = act(&mut b, "at", tr * qd as u64 * BF16);
    let og = act(&mut b, "og", tr * h as u64 * BF16);
    let rlogit = act(&mut b, "rlogit", tr * e as u64 * BF16);
    let tab = act(&mut b, "tab", tr * k as u64 * 8);

    let fu = act(&mut b, "fu", tr * k as u64 * inter as u64 * BF16);
    let part = act(&mut b, "part", tr * k as u64 * h as u64 * F32);
    let moe = act(&mut b, "moe", tr * h as u64 * BF16);
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
        x,
        hn,
        kg,
        vg,
        q,
        opart,
        mlpart,
        mrgc,
        at,
        og,
        rlogit,
        tab,
        fu,
        part,
        moe,
        kv_rows: Vec::new(),
    };
    let rows = em.rows();
    let mut dep = if block_mode {
        0
    } else {
        let emb = em.w(
            &format!("{}embed.weight", c.prefix),
            c.vocab as u64 * h as u64 * BF16,
        );
        em.b.emit(DevOp::Embed, rows.clone(), &[], |d| {
            d.t[0] = x;
            d.t[1] = emb;
            d.t[2] = ids;
            d.i[0] = t;
            d.i[1] = h;
            d.f[0] = 1.0;
        })
    };
    let g_in_for = |em: &mut Emitter, l: usize| -> u32 {
        let name = if c.is_full[l] {
            format!("{}layers.{l}.q_input_norm.weight", c.prefix)
        } else {
            format!("{}layers.{l}.attn_norm.weight", c.prefix)
        };
        em.w(&name, h as u64 * BF16)
    };
    let layer_seq: Vec<usize> = if block_mode {
        layers.collect()
    } else if c.num_passes > 1 {
        (0..c.core_start)
            .chain((0..c.num_passes).flat_map(|_| c.core_start..c.core_end))
            .chain(c.core_end..c.layers as usize)
            .collect()
    } else {
        (0..c.layers as usize).collect()
    };
    let g_in: Vec<u32> = if block_mode {
        vec![g_in_for(&mut em, layer_seq[0])]
    } else {
        (0..c.layers as usize).map(|l| g_in_for(&mut em, l)).collect()
    };
    let g_final = if block_mode {
        TENSOR_NONE
    } else {
        em.w(&format!("{}norm.weight", c.prefix), h as u64 * BF16)
    };
    let eps = c.eps;
    let norm_deps: Vec<u32> = if block_mode { Vec::new() } else { vec![dep] };
    let first_g = if block_mode { g_in[0] } else { g_in[layer_seq[0]] };
    dep = em.b.emit(DevOp::RmsNorm, rows.clone(), &norm_deps, |d| {
        d.t[0] = hn;
        d.t[1] = x;
        d.t[2] = first_g;
        d.i[0] = t;
        d.i[1] = h;
        d.f[0] = eps;
    });
    for (li, &l) in layer_seq.iter().enumerate() {
        let gamma_next = if li + 1 < layer_seq.len() {
            let next_l = layer_seq[li + 1];
            if block_mode {
                g_in[0]
            } else {
                g_in[next_l]
            }
        } else if block_mode {
            g_in[0]
        } else {
            g_final
        };
        let pass_idx = if block_mode { l } else { li };
        dep = em.layer(l, pass_idx, dep, gamma_next);
    }
    if block_mode {
        let tensors = em.b.tensors();
        let gen = em.b.gen_tensors();
        return Phase {
            prog: em.b.finish(),
            tensors,
            gen,
            kv_rows: em.kv_rows,
        };
    }

    let head = format!("{}unembed.weight", c.prefix);
    let lm = em.w(&head, c.vocab as u64 * h as u64 * BF16);
    let (vocab, all) = (c.vocab, em.b.all());
    dep = em.b.emit(DevOp::Gemv, all, &[dep], |d| {
        d.t[0] = logits;
        d.t[1] = hn;
        d.t[2] = lm;
        d.i[0] = if prefill { 1 } else { t };
        d.i[1] = vocab;
        d.i[2] = h;
        d.i[4] = if prefill { t - 1 } else { 0 };
    });
    let nb = if !prefill && (t > 1 || ladder) { t } else { 0 };
    dep = em
        .b
        .emit(DevOp::Argmax, (0..AMAX_BLOCKS).collect(), &[dep], |d| {
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
    assert_eq!(tp, 1, "inkling: tensor parallelism is not implemented");
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("inkling config JSON");
    let c = cfg_inkling(&v);
    let layers = block
        .map(|spec| parse_block(spec, c.layers as usize))
        .unwrap_or(0..c.layers as usize);
    assert!(
        block.is_none() || layers.len() == 1,
        "inkling: --block currently extracts one layer"
    );
    let ecfg = emit_config::active();
    let mut rungs = ecfg.decode_rungs();
    let mut dbatch = *rungs.last().expect("decode_rungs is non-empty");
    let chunk = ecfg.max_chunk.unwrap_or(DEFAULT_CHUNK.min(ctx));
    assert!(
        chunk.is_power_of_two() && chunk <= MAX_CHUNK_MAX && chunk <= ctx,
        "inkling: prefill chunk {chunk} must be a power of two <= min(ctx {ctx}, {MAX_CHUNK_MAX})"
    );
    if ecfg.decode_ladder_default {
        if let Some(target) = hwspec::registry::lookup(gpu) {
            let budget = target.mem.capacity.0.saturating_sub(2 << 30);
            loop {
                let trial = phase_range(
                    &c,
                    ctx,
                    n_cu,
                    dbatch,
                    dbatch,
                    chunk,
                    false,
                    None,
                    layers.clone(),
                );
                let resident: u64 = trial.tensors.iter().map(|t| t.bytes).sum();
                if resident <= budget || rungs.len() == 1 {
                    break;
                }
                rungs.pop();
                dbatch = *rungs.last().unwrap();
            }
            eprintln!("  inkling: default decode ladder after full-context memory sizing: {rungs:?}");
        }
    }
    let buckets: Vec<u32> = crate::with_always_rung(
        [32u32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
            .into_iter()
            .filter(|&x| x <= chunk)
            .collect(),
        ctx,
    );
    let chunk = buckets.last().copied().map_or(chunk, |w| w.max(chunk));
    assert!(
        rungs.iter().all(|&r| buckets.iter().all(|&pb| pb > r)),
        "inkling: decode rungs {rungs:?} overlap the prefill buckets {buckets:?}"
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
        let ph = phase_range(
            &c,
            ctx,
            n_cu,
            t,
            dbatch,
            chunk,
            true,
            tensors.take(),
            layers.clone(),
        );
        tensors = Some(ph.tensors);
        gen = ph.gen;
        progs.push(ph.prog);
        prog_t.push(t);
    }
    for &rb in &rungs {
        let ph = phase_range(
            &c,
            ctx,
            n_cu,
            rb,
            dbatch,
            chunk,
            false,
            tensors.take(),
            layers.clone(),
        );
        tensors = Some(ph.tensors);
        gen = ph.gen;
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

    validate_coverage(
        dir,
        &c.prefix,
        &m.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        block.map(|_| layers.clone()),
        &[
            "_sconv.",
            "gate_attn.",
            "attn_sink_proj.",
            "_amax",
            "_outer_scale_denominator",
            "wr_du",
            "rel_",
            "q_l2_clip",
            "attn_res.",
            "controllers.",
            "embed_norm.",
            "middle_cycle_cross_pass_",
            "k_input_norm.",
            "v_input_norm.",
        ],
        &[],
        &[],
    )
    .unwrap_or_else(|e| panic!("inkling: {e}"));

    crate::check_gfx950_opcode_coverage(&m, crate::target_is_amd(arch, gpu));
    if !rope_gen {
        m.bake_gen();
    }
    let lean = apply_verify_gate(&m, verify);
    let outp = Path::new(out);
    let mut sections = Vec::new();
    if let Some(section) = mxfp4_moe_role::apply_output_object(&mut m, arch, outp)
        .unwrap_or_else(|error| panic!("MXFP4 MoE object: {error}"))
    {
        sections.push(section);
    }
    if block.is_some() {
        use plow_asset::*;
        let l = layers.start;
        let hidden = c.hidden as i64;
        let ckpt = dir.to_string_lossy().into_owned();
        let desc = BlockDescriptor {
            model: ckpt.clone(),
            arch: "inkling_moe".into(),
            layer: l as u32,
            kind: vec![
                if c.is_full[l] {
                    "full_attn"
                } else {
                    "sliding_attn"
                }
                .into(),
                "moe_ffn".into(),
            ],
            hidden,
            dtype: "mxfp4_experts_bf16_activations".into(),
            dims: BlockDims {
                heads: Some(c.heads as i64),
                head_dim: Some(c.hd as i64),
                kv_heads: Some(c.kvh as i64),
                n_exp: Some(c.n_exp as i64),
                top_k: Some(c.top_k as i64),
                moe_inter: Some(c.inter as i64),
                ..Default::default()
            },
            dsa_role: None,
            inputs: vec![BlockTensor {
                name: "act.x".into(),
                shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(hidden)],
                dtype: "bf16".into(),
            }],
            outputs: vec![BlockTensor {
                name: "act.x".into(),
                shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(hidden)],
                dtype: "bf16".into(),
            }],
            carried_state: vec![CarriedState {
                role: "kv".into(),
                tensors: vec![format!("kv.{l}.k"), format!("kv.{l}.v")],
                layout: "head_major".into(),
            }],
            weights: BlockWeights {
                mode: "symlink".into(),
                ckpt,
                prefix: format!("{}layers.{l}.", c.prefix),
            },
            programs: BlockPrograms {
                prefill_buckets: buckets.iter().map(|&t| t as i64).collect(),
                decode_t: dbatch as i64,
            },
        };
        sections.push(write_block_descriptor(out, &desc));
    }
    let ecfg = emit_config::active();
    if block.is_some() || ecfg.block_packets || ecfg.pf_modular {
        let mut modular_progs = Vec::new();
        for (i, &t) in buckets.iter().enumerate() {
            modular_progs.push(modular::create_modular_block_prog(
                plow_asset::ModularBlockKind::DenseAttention,
                plow_asset::ModularPhase::Prefill,
                i as u32,
                t,
                None,
                false,
                false,
                false,
            ));
            modular_progs.push(modular::create_modular_block_prog(
                plow_asset::ModularBlockKind::Moe,
                plow_asset::ModularPhase::Prefill,
                i as u32,
                t,
                None,
                false,
                false,
                true,
            ));
        }
        for (i, &r) in rungs.iter().enumerate() {
            let prog_idx = (buckets.len() + i) as u32;
            modular_progs.push(modular::create_modular_block_prog(
                plow_asset::ModularBlockKind::DenseAttention,
                plow_asset::ModularPhase::Decode,
                prog_idx,
                r,
                None,
                false,
                false,
                false,
            ));
            modular_progs.push(modular::create_modular_block_prog(
                plow_asset::ModularBlockKind::Moe,
                plow_asset::ModularPhase::Decode,
                prog_idx,
                r,
                None,
                false,
                false,
                true,
            ));
        }
        let manifest =
            modular::build_modular_manifest(c.layers, modular_progs, &buckets, &rungs);
        sections.push(modular::modular_pipeline_section(&manifest));
    }
    modular::update_section_with_lean(&mut sections, &lean);
    if block.is_none() {
        sections.push(crate::serve_section::section_for(dir));
    }
    let bytes = if sections.is_empty() {
        m.to_blob()
    } else {
        m.to_blob_v6(&sections)
    };
    std::fs::write(outp, bytes).expect("write inkling blob");
    if !arch.is_empty() {
        let man = manifest::build_without_depth(&m, arch, &lean);
        crate::report_dispatch_audit(&man);
        manifest::write_config_header(&outp.with_file_name("plow_config.h"), &man)
            .expect("inkling config header");
        std::fs::write(
            outp.with_file_name("build.json"),
            serde_json::to_vec_pretty(&man).unwrap(),
        )
        .expect("inkling build manifest");
    }
    eprintln!(
        "inkling: {} layers ({} full)  hidden={} inter={} heads={} kvh={} hd={} experts={}x top{} \
         vocab={}\n  max_ctx={} chunk={} prefill buckets {:?} + decode rungs {:?} -> {}",
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
        outp.display()
    );
}
