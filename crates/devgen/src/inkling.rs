//! Inkling-2-Mini emitter (`model_type: "inkling2_loop"` / `"inkling2_loop_text"`).
//!
//! Per layer: `attn_norm -> q/k/v -> NoPE (skip_norm, no cos/sin) -> flash (sliding 1024 on local
//! layers, full on global) + FLASH_MERGE -> o_proj -> AddNorm(mlp_norm) -> router GEMV
//! (+selection bias in t3) -> top-8 sigmoid renorm -> NVFP4/MXFP8 experts (softcapped SwiGLU, interleaved) -> combine ->
//! AddNorm`.
//!
//! One `Builder` per program: prefill buckets ascending, then decode batch ladder ascending,
//! all sharing one tensor table.

use super::*;
use packet::dev::ACT_SWIGLU_SOFTCAP;
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
    pub logit_softcap: f32,
    pub window: u32,
    pub mlp_up_softcap: f32,
    pub mlp_gate_softcap: f32,
    pub route_scale: f32,
    pub eps: f32,
    pub attn_scale: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub rel_extent: u32,
    pub log_floor: u32,
    pub log_alpha: f32,
    pub prefix: String,
    pub is_full: Vec<bool>,
    pub residual_block_layers: usize,
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
    let mlp_up_softcap = t
        .get("mlp_up_softcap")
        .and_then(|x| x.as_f64())
        .unwrap_or(25.0) as f32;
    let mlp_gate_softcap = t
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
        logit_softcap: t["final_logit_softcapping"].as_f64().unwrap_or(0.0) as f32,
        window,
        mlp_up_softcap,
        mlp_gate_softcap,
        route_scale,
        eps,
        attn_scale,
        rel_clip: t["rel_l2_clip_threshold"].as_f64().expect("missing rel_l2_clip_threshold") as f32,
        rel_extent: g("rel_extent"),
        log_floor: g("log_scaling_n_floor"),
        log_alpha: t["log_scaling_alpha"].as_f64().expect("missing log_scaling_alpha") as f32,
        qk_clip: t["qk_l2_clip_threshold"].as_f64().expect("missing qk_l2_clip_threshold") as f32,
        prefix,
        is_full,
        residual_block_layers: t["n_layers_per_attn_res_block"].as_u64().unwrap_or(6) as usize,
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
    active: u32,
    block_mode: bool,
    k_conv: u32,
    v_conv: u32,
    k_taps: u32,
    v_taps: u32,
    x: u32,
    embed_snapshot: u32,
    hn: u32,
    kg: u32,
    vg: u32,
    q: u32,
    q_clipped: u32,
    k_clipped: u32,
    rel: u32,
    rel_logits: u32,
    log_scale: u32,
    opart: u32,
    mlpart: u32,
    attn_gate: u32,
    attn_sink: u32,
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
        normalize: bool,
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
            d.i[4] = u32::from(!normalize);
            d.f[0] = eps;
            if let Some((ring, mask)) = cache {
                d.j[0] = ring;
                d.j[1] = mask;
                d.i[6] = batch_kv;
            }
        })
    }

    fn clip_heads(&mut self, out: u32, src: u32, scale: u32, heads: u32, tau: u32, dep: u32) -> u32 {
        self.b.emit(DevOp::HeadNormRope, self.b.all(), &[dep], |d| {
            d.t[..3].copy_from_slice(&[out, src, scale]);
            d.t[3] = tau;
            d.t[5] = self.pos;
            d.i[0] = self.t;
            d.i[1] = heads;
            d.i[2] = self.c.hd;
            d.i[4] = self.c.qk_clip.to_bits();
            d.i[5] = packet::dev::ROPE_PAIR_L2_CLIP;
            d.f[0] = 1e-6;
        })
    }

    fn short_conv(&mut self, prefix: &str, kind: &str, visit: usize, full: bool,
                  out: u32, src: u32, projection: u32, input: (u32, u32)) -> u32 {
        let c = self.c;
        let channels = c.kvh * c.hd;
        let weight_name = if full { "static_w" } else { "weight" };
        let path = format!("{prefix}attn.{kind}_sconv");
        let weight = self.w(&format!("{path}.{weight_name}"), channels as u64 * 4 * BF16);
        let history = self.b.tensor(&format!("state.inkling.{visit}.{kind}"),
                                   self.dbatch as u64 * channels as u64 * 3 * BF16);
        let mut deps = vec![projection];
        let taps = if full {
            let w = self.w(&format!("{path}.weight_proj"), c.kvh as u64 * 4 * c.hidden as u64 * BF16);
            let taps = if kind == "k" { self.k_taps } else { self.v_taps };
            deps.push(self.proj(taps, input.0, w, TENSOR_NONE, c.kvh * 4, c.hidden, input.1));
            taps
        } else {
            TENSOR_NONE
        };
        let op = if self.prefill { DevOp::QwenGdnConvPrefill } else { DevOp::QwenGdnConv };
        self.b.emit(op, self.b.all(), &deps, |d| {
            d.t[..5].copy_from_slice(&[out, src, weight, history, self.active]);
            d.t[6] = taps;
            d.i[0] = channels;
            d.i[1] = 4;
            d.i[2] = self.t;
            d.i[3] = if self.prefill { self.t } else { 0 };
            d.i[4] = c.hd;
            d.f[0] = if full { 1.0 / 3.0 } else { 1.0 / 5.0f32.sqrt() };
        })
    }

    fn residual_conv(&mut self, prefix: &str, name: &str, visit: usize, out: u32, src: u32,
                     residual: u32, addend: u32, dep: u32) -> u32 {
        let h = self.c.hidden;
        let weight = self.w(&format!("{prefix}{name}_sconv.weight"), h as u64 * 4 * BF16);
        let history = self.b.tensor(&format!("state.inkling.{visit}.{name}"),
                                    self.dbatch as u64 * h as u64 * 3 * BF16);
        let op = if self.prefill { DevOp::QwenGdnConvPrefill } else { DevOp::QwenGdnConv };
        self.b.emit(op, self.b.all(), &[dep], |d| {
            d.t[..5].copy_from_slice(&[out, src, weight, history, self.active]);
            d.t[5] = residual;
            d.t[7] = addend;
            d.i[..5].copy_from_slice(&[h, 4, self.t, if self.prefill { self.t } else { 0 }, h]);
            d.f[0] = 1.0 / 5.0f32.sqrt();
        })
    }

    fn sliding_prelude(&self, layer: usize) -> bool {
        !self.block_mode && layer < self.c.core_start && layer < self.c.residual_block_layers
            && self.c.is_full[..=layer].iter().all(|&full| !full)
    }

    fn prelude_mix(&mut self, out: u32, last: u32, addend: u32, name: &str,
                   query: (u32, u32), gamma: u32, dep: u32) -> u32 {
        let h = self.c.hidden;
        let (query_row, query_rows) = query;
        let query = self.w(&format!("{}attn_res.{name}", self.c.prefix), query_rows as u64 * h as u64 * F32);
        self.b.emit(DevOp::AttnRes, self.rows(), &[dep], |d| {
            d.t[..4].copy_from_slice(&[out, last, self.embed_snapshot, query]);
            d.t[5] = gamma;
            d.t[7] = addend;
            d.i[..5].copy_from_slice(&[self.t, h, 1, 1, self.t]);
            d.i[5] = query_row;
            d.i[7] = 1;
            d.f[0] = 1e-6;
            d.f[1] = self.c.eps;
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
        let global_prelude = full && l > 0 && self.sliding_prelude(l - 1)
            && l < c.core_start && l < c.residual_block_layers;
        let (k_input, v_input) = if global_prelude {
            let query = format!("attn_queries.{}", l - 1);
            let mut kv_input = |kind, row| {
                let out = self.b.tensor(&format!("act.{kind}_input"), t as u64 * h as u64 * BF16);
                let gamma = self.w(&format!("{p}{kind}_input_norm.weight"), h as u64 * BF16);
                let ready = self.prelude_mix(out, self.x, TENSOR_NONE, &query, (row, 3), gamma, dep);
                (out, ready)
            };
            (kv_input("k", 1), kv_input("v", 2))
        } else {
            ((hn, dep), (hn, dep))
        };
        let (cq, ck, cv) = if self.prefill || t > 1 || self.ladder || k_input.0 != hn {
            (
                self.proj(q, hn, wq, TENSOR_NONE, qd, h, dep),
                self.proj(kg, k_input.0, wk, TENSOR_NONE, kd, h, k_input.1),
                self.proj(vg, v_input.0, wv, TENSOR_NONE, kd, h, v_input.1),
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

        let q_scale = self.w(&format!("{p}attn.q_l2_clip.scale"), qd as u64 * BF16);
        let q_in = q;
        let q = self.q_clipped;
        let tau = if full { self.log_scale } else { TENSOR_NONE };
        let c_qn = self.clip_heads(q, q_in, q_scale, heads, tau, cq);
        let rel_weight = self.w(&format!("{p}attn.wr_du.weight"), (heads * 16) as u64 * h as u64 * BF16);
        let rel_scale = self.w(&format!("{p}attn.rel_l2_clip.scale"), (heads * 16) as u64 * BF16);
        let rel_proj = self.w(&format!("{p}attn.rel_logits_proj.proj"), (16 * c.rel_extent) as u64 * BF16);
        let c_r = self.proj(self.rel, hn, rel_weight, TENSOR_NONE, heads * 16, h, dep);
        let rel_logits = self.rel_logits;
        let c_rel = self.b.emit(DevOp::HeadNormRope, self.b.all(), &[c_r], |d| {
            d.t[..6].copy_from_slice(&[rel_logits, self.rel, rel_scale, rel_proj, tau, self.pos]);
            d.i[..6].copy_from_slice(&[t, heads, 16, c.rel_extent, c.rel_clip.to_bits(),
                                      packet::dev::ROPE_PAIR_RELATIVE_LOGITS]);
            d.f[0] = 1e-6;
        });
        let ck = self.short_conv(&p, "k", pass_idx, full, self.k_conv, kg, ck, k_input);
        let cv = self.short_conv(&p, "v", pass_idx, full, self.v_conv, vg, cv, v_input);
        let ck = self.clip_heads(self.k_clipped, self.k_conv, TENSOR_NONE, kvh, TENSOR_NONE, ck);
        let c_kn = self.rope(kc, self.k_clipped, kvh, Some((kvr, kvm)), false, ck);
        let c_vn = self.rope(vc, self.v_conv, kvh, Some((kvr, kvm)), true, cv);
        if !self.prefill {
            self.kv_rows.extend([c_kn, c_vn]);
        }

        let (opart, mlpart, kvlen, ns, scale) =
            (self.opart, self.mlpart, self.kvlen, self.ns, c.attn_scale);
        let attn_gate = self.attn_gate;
        let attn_sink = self.attn_sink;
        let wg = self.w(&format!("{p}attn.gate_attn.weight"), qd as u64 * h as u64 * BF16);
        let ws = self.w(&format!("{p}attn.attn_sink_proj.weight"), heads as u64 * h as u64 * BF16);
        let c_gate = self.proj(attn_gate, hn, wg, TENSOR_NONE, qd, h, dep);
        let c_sink = self.proj(attn_sink, hn, ws, TENSOR_NONE, heads, h, dep);
        let c_fa = if self.prefill {
            self.b.emit(
                DevOp::FlashPrefill,
                self.b.all(),
                &[c_qn, c_kn, c_vn, c_rel],
                |d| {
                    d.t[0] = opart;
                    d.t[1] = mlpart;
                    d.t[2] = q;
                    d.t[3] = kc;
                    d.t[4] = vc;
                    d.t[6] = rel_logits;
                    d.t[7] = rel_logits;
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
            let fa_deps = [c_qn, c_kn, c_vn, c_rel];
            self.b
                .emit(DevOp::FlashDecode, self.b.all(), &fa_deps, |d| {
                    d.t[0] = opart;
                    d.t[1] = mlpart;
                    d.t[2] = q;
                    d.t[3] = kc;
                    d.t[4] = vc;
                    d.t[5] = kvlen;
                    d.t[7] = rel_logits;
                    d.j[1] = packet::dev::FLASH_DECODE_RELATIVE;
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
                })
        };

        let at = self.at;
        let mg: Vec<u32> = (0..(t * heads).min(self.b.n_cu()).max(1)).collect();
        let c_attn_out = self.b.emit(DevOp::FlashMerge, mg, &[c_fa, c_gate, c_sink], |d| {
            d.t[0] = at;
            d.t[1] = opart;
            d.t[2] = mlpart;
            d.t[4] = attn_gate;
            d.t[5] = attn_sink;
            d.i[0] = t;
            d.i[1] = heads;
            d.i[2] = ns;
            d.i[3] = hd;
        });
        let og = self.og;
        let c_o = self.proj(og, at, wo, TENSOR_NONE, h, qd, c_attn_out);
        let use_prelude_mix = l == 0 || self.sliding_prelude(l) || global_prelude;
        let attn_conv = if use_prelude_mix {
            self.b.tensor("act.attn_conv", t as u64 * h as u64 * BF16)
        } else {
            TENSOR_NONE
        };
        let residual = if use_prelude_mix && l != 0 { self.x } else { TENSOR_NONE };
        let c_n1 = if use_prelude_mix {
            let conv = self.residual_conv(&p, "attn", pass_idx, attn_conv, og,
                TENSOR_NONE, TENSOR_NONE, c_o);
            let (last, addend) = if l == 0 { (attn_conv, TENSOR_NONE) } else { (residual, attn_conv) };
            self.prelude_mix(hn, last, addend, &format!("mlp_queries.{l}"), (0, 1), g_mlp, conv)
        } else {
            self.add_norm(og, g_mlp, c_o)
        };

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

        let (gu_amax, gu_denominator, down_amax, down_denominator) = if is_fp8 {
            (TENSOR_NONE, TENSOR_NONE, TENSOR_NONE, TENSOR_NONE)
        } else {
            (
                self.w(&format!("{p}mlp.experts.w13_weight_amax"), e as u64 * F32),
                self.w(&format!("{p}mlp.experts.w13_weight_outer_scale_denominator"), F32),
                self.w(&format!("{p}mlp.experts.w2_weight_amax"), e as u64 * F32),
                self.w(&format!("{p}mlp.experts.w2_weight_outer_scale_denominator"), F32),
            )
        };

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

        let (gate_cap, up_cap) = (c.mlp_gate_softcap, c.mlp_up_softcap);
        let c_g = self.b.emit(DevOp::MoeGluMx, self.b.all(), &[c_tk], |d| {
            d.t[0] = fu;
            d.t[1] = hn;
            d.t[2] = tab;
            d.t[3] = w_gu;
            d.t[4] = s_gu;
            d.t[6] = gu_amax;
            d.t[7] = gu_denominator;
            d.i[0] = k;
            d.i[1] = inter;
            d.i[2] = h;
            d.i[3] = e;
            d.i[4] = 0; // interleaved gate/up rows
            d.i[5] = ACT_SWIGLU_SOFTCAP;
            d.i[6] = t;
            d.i[7] = quant_mode;
            d.f[0] = gate_cap;
            d.f[1] = up_cap;
        });
        let c_d = self.b.emit(DevOp::MoeDownMx, self.b.all(), &[c_g], |d| {
            d.t[0] = part;
            d.t[1] = fu;
            d.t[2] = tab;
            d.t[3] = w_d;
            d.t[4] = s_d;
            d.t[6] = down_amax;
            d.t[7] = down_denominator;
            d.i[0] = k;
            d.i[1] = h;
            d.i[2] = inter;
            d.i[3] = e;
            d.i[5] = 1; // Router weight was applied before the activation rounded to BF16.
            d.i[6] = t;
            d.i[7] = quant_mode;
        });

        if !use_prelude_mix && crate::emit_config::active().nv_moe_combine_fold {
            self.add_norm_moe_fold(part, gamma_next, k, c_d)
        } else {
            let c_c = self.b.emit(DevOp::MoeCombinePf, self.b.all(), &[c_d], |d| {
                d.t[0] = moe;
                d.t[3] = part;
                d.i[0] = h;
                d.i[1] = k;
                d.i[2] = t;
            });

            if use_prelude_mix {
                let next = if self.block_mode {
                    self.x
                } else {
                    self.b.tensor(if l % 2 == 0 { "act.partial_even" } else { "act.partial_odd" },
                        t as u64 * h as u64 * BF16)
                };
                let conv = self.residual_conv(&p, "mlp", pass_idx, next, moe, residual, attn_conv, c_c);
                if self.block_mode {
                    conv
                } else {
                    let query_rows = if c.is_full.get(l + 1) == Some(&true) { 3 } else { 1 };
                    let mix = self.prelude_mix(hn, next, TENSOR_NONE, &format!("attn_queries.{l}"),
                        (0, query_rows), gamma_next, conv);
                    self.x = next;
                    mix
                }
            } else {
                self.add_norm(moe, gamma_next, c_c)
            }
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
    let existing_log_scale = tensors.as_ref()
        .and_then(|table| table.iter().position(|t| t.name == "const.log_scale"));
    if let Some(tensors) = tensors {
        b.adopt_tensors(tensors);
    }
    b.set_tensor_dedup(true);
    let (h, heads, hd, inter, e, k) = (c.hidden, c.heads, c.hd, c.inter, c.n_exp, c.top_k);
    let (qd, kd) = (heads * hd, c.kvh * hd);

    let ids = b.tensor("in.ids", ctx as u64 * F32);
    let pos = b.tensor("in.pos", ctx as u64 * F32);
    let kvlen = b.tensor("in.kvlen", dbatch as u64 * F32);
    let active = b.tensor("in.active", dbatch as u64 * F32);

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
    let k_conv = act(&mut b, "k_conv", tr * kd as u64 * BF16);
    let v_conv = act(&mut b, "v_conv", tr * kd as u64 * BF16);
    let k_taps = act(&mut b, "k_taps", tr * c.kvh as u64 * 4 * BF16);
    let v_taps = act(&mut b, "v_taps", tr * c.kvh as u64 * 4 * BF16);
    let q = act(&mut b, "q", tr * qd as u64 * BF16);
    let q_clipped = act(&mut b, "q_clipped", tr * qd as u64 * BF16);
    let k_clipped = act(&mut b, "k_clipped", tr * kd as u64 * BF16);
    let rel = act(&mut b, "rel", tr * heads as u64 * 16 * BF16);
    let rel_logits = act(&mut b, "rel_logits", tr * heads as u64 * c.rel_extent as u64 * BF16);
    let log_scale = existing_log_scale.map_or_else(
        || b.tensor_init("const.log_scale", [c.log_floor.to_le_bytes(), c.log_alpha.to_le_bytes()].concat()),
        |handle| handle as u32,
    );
    let opart = act(
        &mut b,
        "opart",
        tr * heads as u64 * ns as u64 * hd as u64 * F32,
    );
    let mlpart = act(&mut b, "mlpart", tr * heads as u64 * ns as u64 * 2 * F32);
    let attn_gate = act(&mut b, "attn_gate", tr * qd as u64 * BF16);
    let attn_sink = act(&mut b, "attn_sink", tr * heads as u64 * BF16);
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
        active,
        block_mode,
        k_conv,
        v_conv,
        k_taps,
        v_taps,
        x,
        embed_snapshot: x,
        hn,
        kg,
        vg,
        q,
        q_clipped,
        k_clipped,
        rel,
        rel_logits,
        log_scale,
        opart,
        mlpart,
        attn_gate,
        attn_sink,
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
        let embed_norm = em.w(&format!("{}embed_norm.weight", c.prefix), h as u64 * BF16);
        let embed = em.b.emit(DevOp::Embed, rows.clone(), &[], |d| {
            d.t[0] = og;
            d.t[1] = emb;
            d.t[2] = ids;
            d.i[0] = t;
            d.i[1] = h;
            d.f[0] = 1.0;
        });
        em.b.emit(DevOp::RmsNorm, rows.clone(), &[embed], |d| {
            d.t[0] = x;
            d.t[1] = og;
            d.t[2] = embed_norm;
            d.i[0] = t;
            d.i[1] = h;
            d.f[0] = c.eps;
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
    let logit_rows = if prefill { 1 } else { t };
    dep = em.b.emit(DevOp::Gemv, all, &[dep], |d| {
        d.t[0] = logits;
        d.t[1] = hn;
        d.t[2] = lm;
        d.i[0] = logit_rows;
        d.i[1] = vocab;
        d.i[2] = h;
        d.i[4] = if prefill { t - 1 } else { 0 };
    });
    if c.logit_softcap > 0.0 {
        dep = em.b.emit(DevOp::SoftCap, em.b.all(), &[dep], |d| {
            d.t[0] = logits;
            d.t[1] = logits;
            d.i[0] = logit_rows * vocab;
            d.f[0] = c.logit_softcap;
        });
    }
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
    let text = v.get("text_config").unwrap_or(&v);
    assert_eq!(text["d_rel"].as_u64(), Some(16), "inkling requires 16 relative channels");
    assert_eq!(c.rel_extent, packet::dev::FLASH_RELATIVE_EXTENT);
    assert!(c.log_floor > 0 && c.log_alpha.is_finite() && c.log_alpha >= 0.0);
    assert_eq!(text["sconv_kernel_size"].as_u64(), Some(4), "inkling requires four convolution taps");
    assert_eq!(text["use_value_norm_no_gamma"].as_bool(), Some(true), "inkling requires unweighted V RMSNorm");
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
            ".attn_sconv.",
            ".mlp_sconv.",
            "attn_res.",
            "controllers.",
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
    let mut sections = vec![crate::qwen35::recurrent_state_section(&m, dbatch)];
    crate::object_requirements::append(&m, &mut sections)
        .unwrap_or_else(|error| panic!("object requirements: {error}"));
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
