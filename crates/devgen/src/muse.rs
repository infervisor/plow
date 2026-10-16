//! Muse Spark (`muse_spark` / `muse_spark_text`) device emitter, M1: `--block l[..r]` decode at
//! TP1 for gfx950. Hand-routed like gpt_oss / qwen35 (the rewrite path has no terms for the
//! Muse IR ops); the graph it lowers is `crates/nn-graph/src/models/muse_spark.rs`.
//!
//! Per layer (one decode token, `act.x` -> `act.xnext`):
//!
//! ```text
//!   hn   = centered_norm(x)                     MuseRmsNorm
//!   q,k,v,g = hn @ {q,k,v,gate}_proj            Gemv (bf16)
//!   q,k  = rope?(norm(q|k))  -> k into kv.{l}.k QwenHeadNormRope (weightless; NoPE on full layers)
//!   attn = flash(q, kv, window?)                FlashDecode/FlashMerge, scale = qk_scale/sqrt(hd)
//!   at   = norm_head(attn) * sigmoid(g)         QwenHeadNormRope + QwenSigmoidGate
//!   xmid = gate_res(x, norm(o_proj(at)))        MuseRmsNorm + MuseResidualGate
//!   hn2  = centered_norm(xmid)
//!   tab  = topk(sigmoid(0.17678 * hn2 @ Wr_f32) + bias)   GemvF32 + MoeRouterTopk (f1 scale)
//!   xe   = centered_norm(hn2 @ pre_expert_proj)
//!   ylat = sum_j w_j * input_scaled_norm(expert_j(xe))    grouped MXFP4 GLU/down + MuseMoeCombineNorm
//!   xnext = gate_res(xmid, input_scaled_norm(ylat @ post_expert_proj))
//! ```
//!
//! `q * qk_scale_factor` is folded into the flash scale. The routed experts run W4A16 (bf16
//! activations against dequantized MXFP4); the Quark export declares dynamic MXFP4 activations.
use super::*;
use std::path::Path;

const PREFIX: &str = "model.language_model.";
const MOE_ENC_MXFP4: u32 = 2;
const MOE_ACT_SILU: u32 = 1;
/// `MoeRouterTopk` flags: sigmoid | renormalize | bias bound | f32 logit.
const ROUTER_FLAGS: u32 = 1 | 2 | 4 | 8;
const NORM_PLAIN: u32 = 0;
const NORM_CENTERED: u32 = 1;
const NORM_INPUT_SCALED: u32 = 2;
/// `MuseResidualGate(residual_gate_eps=1e-3)`.
const RESIDUAL_GATE_EPS: f32 = 1e-3;
const BF16: u64 = 2;
const F32: u64 = 4;

/// What full-model emission still needs; the refusal names it.
const REMAINING: &str = "full-model emission (embed + weightless embed norm, final norm, untied \
    lm_head with output_multiplier and 20*tanh softcap, sampling), all 62 layers chained, prefill \
    buckets (hd64 FlashPrefill, grouped MoE prefill with the post-expert norm), batched decode \
    rungs, sliding-window KV rings, bf16-expert layers 0 and 61, TP8/EP layout and collectives, \
    and serve integration";

#[derive(Debug, Clone)]
pub(crate) struct MuseCfg {
    pub hidden: u32,
    pub heads: u32,
    pub kv_heads: u32,
    pub hd: u32,
    pub layers: u32,
    pub eps: f32,
    pub post_eps: f32,
    pub window: u32,
    pub sliding: Vec<bool>,
    pub rope: Vec<bool>,
    pub sparse: Vec<bool>,
    pub theta: f64,
    pub qk_scale: f32,
    pub output_multiplier: f32,
    pub temperature: f32,
    pub n_exp: u32,
    pub top_k: u32,
    pub moe_hidden: u32,
    pub moe_inter: u32,
    pub route_scale: f32,
    /// Per layer: routed experts are Quark MXFP4 (else bf16, `exclude`d).
    pub mxfp4_experts: Vec<bool>,
}

impl MuseCfg {
    pub(crate) fn parse(root: &Value) -> Self {
        let v = root.get("text_config").unwrap_or(root);
        let u = |k: &str| -> u32 {
            let n = v[k]
                .as_u64()
                .unwrap_or_else(|| panic!("muse_spark: missing or non-integer {k}"));
            u32::try_from(n).expect("muse_spark: dimension exceeds u32")
        };
        let f = |k: &str| -> f64 {
            v[k].as_f64()
                .unwrap_or_else(|| panic!("muse_spark: missing or non-numeric {k}"))
        };
        let strs = |k: &str| -> Vec<String> {
            v[k].as_array()
                .unwrap_or_else(|| panic!("muse_spark: missing {k}"))
                .iter()
                .map(|s| s.as_str().expect("muse_spark: non-string entry").to_string())
                .collect()
        };
        let layers = u("num_hidden_layers");
        let n = layers as usize;
        let types = strs("layer_types");
        let mlp = strs("mlp_layer_types");
        let thetas: Vec<f64> = v["layer_rope_theta"]
            .as_array()
            .expect("muse_spark: missing layer_rope_theta")
            .iter()
            .map(|t| t.as_f64().expect("muse_spark: non-numeric layer_rope_theta"))
            .collect();
        for (k, len) in [
            ("layer_types", types.len()),
            ("mlp_layer_types", mlp.len()),
            ("layer_rope_theta", thetas.len()),
        ] {
            assert_eq!(len, n, "muse_spark: {k} has {len} entries for {n} layers");
        }
        let rp = &v["rope_parameters"];
        assert_eq!(
            rp["rope_type"].as_str(),
            Some("default"),
            "muse_spark: only default RoPE"
        );
        let theta = rp["rope_theta"].as_f64().expect("muse_spark: rope_theta");
        assert!(
            thetas.iter().all(|&t| t == 0.0 || t == theta),
            "muse_spark: a layer_rope_theta differs from rope_parameters.rope_theta; the \
             reference applies the global theta to every RoPE layer"
        );
        assert_eq!(v["hidden_act"].as_str(), Some("silu"), "muse_spark: silu only");
        assert_eq!(
            v["num_shared_experts"].as_u64().unwrap_or(0),
            0,
            "muse_spark: shared experts are not modeled"
        );
        assert!(
            !v["attention_bias"].as_bool().unwrap_or(false),
            "muse_spark: attention_bias is not modeled"
        );
        let quant = root
            .get("quantization_config")
            .or_else(|| v.get("quantization_config"));
        let exclude: Vec<&str> = quant
            .and_then(|q| q["exclude"].as_array())
            .map(|a| a.iter().filter_map(|s| s.as_str()).collect())
            .unwrap_or_default();
        if let Some(q) = quant {
            let w = &q["global_quant_config"]["weight"];
            assert!(
                q["quant_method"] == "quark"
                    && w["dtype"] == "fp4"
                    && w["qscheme"] == "per_group"
                    && w["group_size"] == 32
                    && w["scale_format"] == "e8m0",
                "muse_spark: only the Quark MXFP4 export (fp4, per_group 32, e8m0) is modeled"
            );
        }
        let mxfp4_experts = (0..n)
            .map(|l| {
                quant.is_some()
                    && !exclude.contains(&format!("{PREFIX}layers.{l}.mlp.experts.0.gate_proj").as_str())
            })
            .collect();
        Self {
            hidden: u("hidden_size"),
            heads: u("num_attention_heads"),
            kv_heads: u("num_key_value_heads"),
            hd: u("head_dim"),
            layers,
            eps: f("rms_norm_eps") as f32,
            post_eps: f("post_norm_eps") as f32,
            window: u("sliding_window"),
            sliding: types
                .iter()
                .map(|t| match t.as_str() {
                    "sliding_attention" => true,
                    "full_attention" => false,
                    other => panic!("muse_spark: unknown layer type {other:?}"),
                })
                .collect(),
            rope: thetas.iter().map(|&t| t != 0.0).collect(),
            sparse: mlp
                .iter()
                .map(|t| match t.as_str() {
                    "sparse" => true,
                    "dense" => false,
                    other => panic!("muse_spark: unknown mlp layer type {other:?}"),
                })
                .collect(),
            theta,
            qk_scale: f("qk_scale_factor") as f32,
            output_multiplier: f("output_multiplier") as f32,
            temperature: f("residual_gate_temperature") as f32,
            n_exp: u("num_local_experts"),
            top_k: u("num_experts_per_tok"),
            moe_hidden: u("moe_hidden_size"),
            moe_inter: u("moe_intermediate_size"),
            route_scale: f("routed_scaling_factor") as f32,
            mxfp4_experts,
        }
    }

    /// The M1 refusals for one layer, or `Ok`.
    fn check_m1_layer(&self, l: usize) -> Result<(), String> {
        if !self.sparse[l] {
            return Err(format!("layer {l} has a dense MLP (no M1 route)"));
        }
        if !self.mxfp4_experts[l] {
            return Err(format!(
                "layer {l} routed experts are bf16 (Quark `exclude`); M1 routes MXFP4 experts only"
            ));
        }
        Ok(())
    }
}

struct Acts {
    x: u32,
    xnext: u32,
    hn: u32,
    q_raw: u32,
    k_raw: u32,
    v_raw: u32,
    gate: u32,
    q: u32,
    opart: u32,
    mlpart: u32,
    attn: u32,
    at: u32,
    o: u32,
    on: u32,
    xmid: u32,
    hn2: u32,
    rlogit: u32,
    tab: u32,
    xe0: u32,
    xe: u32,
    fu: u32,
    part: u32,
    ylat: u32,
    y: u32,
    yn: u32,
}

struct Emitter<'a> {
    b: Builder,
    c: &'a MuseCfg,
    ctx: u32,
    ns: u32,
    pos: u32,
    kvlen: u32,
    cos: u32,
    sin: u32,
    a: Acts,
}

impl Emitter<'_> {
    fn cus(&self, n: u32) -> Vec<u32> {
        (0..n.clamp(1, self.b.n_cu())).collect()
    }
    fn w(&mut self, name: &str, bytes: u64) -> u32 {
        self.b.tensor(name, bytes)
    }
    fn gemv(&mut self, out: u32, x: u32, w: u32, n: u32, k: u32, dep: u32) -> u32 {
        self.b.emit(DevOp::Gemv, self.b.all(), &[dep], |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = w;
            d.i[0] = 1;
            d.i[1] = n;
            d.i[2] = k;
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(&mut self, out: u32, x: u32, w: u32, dim: u32, mode: u32, eps: f32, dep: &[u32]) -> u32 {
        self.b.emit(DevOp::MuseRmsNorm, vec![0], dep, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = w;
            d.i[0] = 1;
            d.i[1] = dim;
            d.i[2] = mode;
            d.f[0] = eps;
        })
    }
    fn residual_gate(&mut self, out: u32, res: u32, h: u32, g: u32, deps: &[u32]) -> u32 {
        let n = self.c.hidden;
        let (t, cus) = (self.c.temperature, self.cus(n.div_ceil(512)));
        self.b.emit(DevOp::MuseResidualGate, cus, deps, |d| {
            d.t[0] = out;
            d.t[1] = res;
            d.t[2] = h;
            d.t[3] = g;
            d.i[0] = n;
            d.i[1] = 1;
            d.f[0] = t;
            d.f[1] = RESIDUAL_GATE_EPS;
        })
    }
    /// Weightless per-head RMSNorm, optionally NeoX-rotated, optionally written into a KV cache
    /// row at `pos` (`cache`). `normalize = false` is a plain copy (the V cache write).
    #[allow(clippy::too_many_arguments)]
    fn head_norm(
        &mut self,
        out: u32,
        src: u32,
        heads: u32,
        rope: bool,
        cache: bool,
        normalize: bool,
        dep: u32,
    ) -> u32 {
        let (hd, eps, ctx) = (self.c.hd, self.c.eps, self.ctx);
        let (cos, sin, pos) = (self.cos, self.sin, self.pos);
        let cus = self.cus(heads.div_ceil(8));
        self.b.emit(DevOp::QwenHeadNormRope, cus, &[dep], |d| {
            d.t[..7].copy_from_slice(&[
                out,
                src,
                TENSOR_NONE,
                if rope { cos } else { TENSOR_NONE },
                if rope { sin } else { TENSOR_NONE },
                pos,
                TENSOR_NONE,
            ]);
            d.i[..6].copy_from_slice(&[
                heads,
                hd,
                if rope { hd } else { 0 },
                1,
                if cache { ctx } else { 0 },
                normalize as u32,
            ]);
            d.f[0] = eps;
        })
    }

    fn layer(&mut self, l: usize, src: u32, dst: u32, dep: Option<u32>) -> u32 {
        let c = self.c;
        let p = format!("{PREFIX}layers.{l}.");
        let (h, nh, nkv, hd) = (c.hidden, c.heads, c.kv_heads, c.hd);
        let (qd, kd, m, e) = (nh * hd, nkv * hd, c.moe_hidden, c.n_exp);
        let (h64, qd64, kd64, m64) = (h as u64, qd as u64, kd as u64, m as u64);
        let a = &self.a;
        let (hn, q_raw, k_raw, v_raw, gate, q) = (a.hn, a.q_raw, a.k_raw, a.v_raw, a.gate, a.q);
        let (opart, mlpart, attn, at, o, on) = (a.opart, a.mlpart, a.attn, a.at, a.o, a.on);
        let (xmid, hn2, rlogit, tab, xe0, xe) = (a.xmid, a.hn2, a.rlogit, a.tab, a.xe0, a.xe);
        let (fu, part, ylat, y, yn) = (a.fu, a.part, a.ylat, a.y, a.yn);

        let w_in = self.w(&format!("{p}input_layernorm.weight"), h64 * BF16);
        let wq = self.w(&format!("{p}self_attn.q_proj.weight"), qd64 * h64 * BF16);
        let wk = self.w(&format!("{p}self_attn.k_proj.weight"), kd64 * h64 * BF16);
        let wv = self.w(&format!("{p}self_attn.v_proj.weight"), kd64 * h64 * BF16);
        let wg = self.w(&format!("{p}self_attn.gate_proj.weight"), qd64 * h64 * BF16);
        let wo = self.w(&format!("{p}self_attn.o_proj.weight"), h64 * qd64 * BF16);
        let g_attn = self.w(&format!("{p}post_attention_residual_gate.gate"), h64 * BF16);
        let w_ff = self.w(&format!("{p}pre_feedforward_layernorm.weight"), h64 * BF16);
        let w_post = self.w(&format!("{p}post_feedforward_layernorm.weight"), h64 * BF16);
        let g_ff = self.w(&format!("{p}post_feedforward_residual_gate.gate"), h64 * BF16);
        let wr = self.w(&format!("{p}mlp.gate.weight"), e as u64 * h64 * F32);
        let wr_bias = self.w(&format!("{p}mlp.gate.e_score_correction_bias"), e as u64 * F32);
        let w_pre = self.w(&format!("{p}mlp.pre_expert_proj.weight"), m64 * h64 * BF16);
        let w_pre_n = self.w(&format!("{p}mlp.pre_expert_norm.weight"), m64 * BF16);
        let w_pen = self.w(&format!("{p}mlp.experts.post_expert_norm.weight"), m64 * BF16);
        let w_postp = self.w(&format!("{p}mlp.post_expert_proj.weight"), h64 * m64 * BF16);
        let ewt = self.w(&format!("moe.{p}mlp.expert_weight_table"), e as u64 * 3 * 8);
        let est = self.w(&format!("moe.{p}mlp.expert_scale_table"), e as u64 * 3 * 8);
        let slots = 1u64;
        let kc = self.w(
            &format!("kv.{l}.k"),
            slots * kd64 * self.ctx as u64 * BF16,
        );
        let vc = self.w(
            &format!("kv.{l}.v"),
            slots * kd64 * self.ctx as u64 * BF16,
        );

        // ---- attention ----
        let d_hn = self.norm(hn, src, w_in, h, NORM_CENTERED, c.eps, dep.as_slice());
        let d_q = self.gemv(q_raw, hn, wq, qd, h, d_hn);
        let d_k = self.gemv(k_raw, hn, wk, kd, h, d_hn);
        let d_v = self.gemv(v_raw, hn, wv, kd, h, d_hn);
        let d_g = self.gemv(gate, hn, wg, qd, h, d_hn);
        let rope = c.rope[l];
        let d_qn = self.head_norm(q, q_raw, nh, rope, false, true, d_q);
        let d_kn = self.head_norm(kc, k_raw, nkv, rope, true, true, d_k);
        let d_vn = self.head_norm(vc, v_raw, nkv, false, true, false, d_v);
        let (ns, kvlen, ctx) = (self.ns, self.kvlen, self.ctx);
        let window = if c.sliding[l] { c.window } else { 0 };
        let scale = c.qk_scale / (hd as f32).sqrt();
        let d_fa = self
            .b
            .emit(DevOp::FlashDecode, self.b.all(), &[d_qn, d_kn, d_vn], |d| {
                d.t[..6].copy_from_slice(&[opart, mlpart, q, kc, vc, kvlen]);
                d.i.copy_from_slice(&[1, nh, nkv, ctx, window, ns, hd, u32::MAX]);
                d.f[0] = scale;
            });
        let d_fm = self.b.emit(DevOp::FlashMerge, self.cus(nh), &[d_fa], |d| {
            d.t[..3].copy_from_slice(&[attn, opart, mlpart]);
            d.i[..4].copy_from_slice(&[1, nh, ns, hd]);
        });
        let d_on = self.head_norm(at, attn, nh, false, false, true, d_fm);
        let d_gate = self
            .b
            .emit(DevOp::QwenSigmoidGate, self.cus(qd.div_ceil(512)), &[d_on, d_g], |d| {
                d.t[..4].copy_from_slice(&[at, at, gate, TENSOR_NONE]);
                d.i[..2].copy_from_slice(&[qd, 1]);
            });
        let d_o = self.gemv(o, at, wo, h, qd, d_gate);
        let d_pn = self.norm(on, o, TENSOR_NONE, h, NORM_PLAIN, c.post_eps, &[d_o]);
        let d_xm = self.residual_gate(xmid, src, on, g_attn, &[d_pn]);

        // ---- latent MoE ----
        let d_hn2 = self.norm(hn2, xmid, w_ff, h, NORM_CENTERED, c.eps, &[d_xm]);
        let d_rl = self.b.emit(DevOp::GemvF32, self.b.all(), &[d_hn2], |d| {
            d.t[0] = rlogit;
            d.t[1] = hn2;
            d.t[2] = wr;
            d.i[0] = 1;
            d.i[1] = e;
            d.i[2] = h;
        });
        let (top_k, route_scale, logit_scale) = (c.top_k, c.route_scale, c.output_multiplier);
        let d_rt = self.b.emit(DevOp::MoeRouterTopk, vec![0], &[d_rl], |d| {
            d.t[0] = tab;
            d.t[1] = rlogit;
            d.t[3] = wr_bias;
            d.i[1] = e;
            d.i[2] = top_k;
            d.i[3] = ROUTER_FLAGS;
            d.i[6] = 1;
            d.i[7] = 1;
            d.f[0] = route_scale;
            d.f[1] = logit_scale;
        });
        let d_xe0 = self.gemv(xe0, hn2, w_pre, m, h, d_hn2);
        let d_xe = self.norm(xe, xe0, w_pre_n, m, NORM_CENTERED, c.eps, &[d_xe0]);
        let imoe = c.moe_inter;
        let d_glu = self
            .b
            .emit(DevOp::MoeGroupGluFp8Blk, self.b.all(), &[d_rt, d_xe], |d| {
                d.t[..5].copy_from_slice(&[fu, xe, tab, ewt, est]);
                d.i[0] = top_k;
                d.i[1] = imoe;
                d.i[2] = m;
                d.i[3] = e;
                d.i[5] = MOE_ACT_SILU;
                d.i[6] = MOE_ENC_MXFP4;
            });
        let d_dn = self
            .b
            .emit(DevOp::MoeGroupDownFp8Blk, self.b.all(), &[d_glu], |d| {
                d.t[..5].copy_from_slice(&[part, fu, tab, ewt, est]);
                d.i[0] = top_k;
                d.i[1] = m;
                d.i[2] = imoe;
                d.i[3] = e;
                d.i[6] = MOE_ENC_MXFP4;
            });
        let post_eps = c.post_eps;
        let d_cmb = self
            .b
            .emit(DevOp::MuseMoeCombineNorm, self.cus(m.div_ceil(512)), &[d_dn], |d| {
                d.t[..4].copy_from_slice(&[ylat, part, tab, w_pen]);
                d.i[0] = m;
                d.i[1] = top_k;
                d.f[0] = post_eps;
            });
        let d_y = self.gemv(y, ylat, w_postp, h, m, d_cmb);
        let d_yn = self.norm(yn, y, w_post, h, NORM_INPUT_SCALED, c.post_eps, &[d_y]);
        self.residual_gate(dst, xmid, yn, g_ff, &[d_yn])
    }
}

/// One decode program over `layers`. The last layer writes `act.xnext`; earlier ones write
/// `act.x` back in place (it is dead after the layer's attention residual).
fn decode_program(c: &MuseCfg, ctx: u32, n_cu: u32, layers: std::ops::Range<usize>) -> Model {
    let mut b = Builder::new(n_cu);
    b.force_uniseg();
    let (h, nh, nkv, hd, m) = (c.hidden, c.heads, c.kv_heads, c.hd, c.moe_hidden);
    let (qd, kd) = ((nh * hd) as u64, (nkv * hd) as u64);
    // `in.ids` first: handle 0 is the "absent" value of several optional operand slots.
    b.tensor("in.ids", ctx as u64 * F32);
    // The engine reads its context capacity off `in.pos`.
    let pos = b.tensor("in.pos", ctx as u64 * F32);
    let kvlen = b.tensor("in.kvlen", F32);
    let [gc, gs] = GenTensor::rope_pair(ctx, hd, c.theta, 1.0, RopeScale::None);
    let cos = b.tensor_gen("in.cos_full", gc.byte_len(), gc);
    let sin = b.tensor_gen("in.sin_full", gs.byte_len(), gs);
    let ns = decode_ns(c, n_cu);
    let (h64, m64) = (h as u64, m as u64);
    let mut act = |n: &str, bytes: u64| b.tensor(&format!("act.{n}"), bytes);
    let a = Acts {
        x: act("x", h64 * BF16),
        xnext: act("xnext", h64 * BF16),
        hn: act("hn", h64 * BF16),
        q_raw: act("qg", qd * BF16),
        k_raw: act("kg", kd * BF16),
        v_raw: act("vg", kd * BF16),
        gate: act("ogate", qd * BF16),
        q: act("q", qd * BF16),
        opart: act("opart", nh as u64 * ns as u64 * hd as u64 * F32),
        mlpart: act("mlpart", nh as u64 * ns as u64 * 2 * F32),
        attn: act("attn", qd * BF16),
        at: act("at", qd * BF16),
        o: act("o", h64 * BF16),
        on: act("on", h64 * BF16),
        xmid: act("xmid", h64 * BF16),
        hn2: act("xn2", h64 * BF16),
        rlogit: act("rlogit", c.n_exp as u64 * F32),
        tab: act("tab", c.top_k as u64 * 8),
        xe0: act("xe0", m64 * BF16),
        xe: act("xe", m64 * BF16),
        fu: act("fu", c.top_k as u64 * c.moe_inter as u64 * BF16),
        part: act("part", c.top_k as u64 * m64 * F32),
        ylat: act("ylat", m64 * BF16),
        y: act("y", h64 * BF16),
        yn: act("yn", h64 * BF16),
    };
    let (x, xnext) = (a.x, a.xnext);
    let mut em = Emitter {
        b,
        c,
        ctx,
        ns,
        pos,
        kvlen,
        cos,
        sin,
        a,
    };
    let mut dep = None;
    let last = layers.end - 1;
    for l in layers {
        dep = Some(em.layer(l, x, if l == last { xnext } else { x }, dep));
    }
    let tensors = em.b.tensors();
    let gen = em.b.gen_tensors();
    Model {
        n_cu,
        target: 0,
        tensors,
        progs: vec![em.b.finish()],
        kv_row_insts: Vec::new(),
        prog_t: vec![1],
        gen,
    }
}

/// Validate `--block` and the M1 scope, then return the layer range.
pub(crate) fn m1_layers(
    c: &MuseCfg,
    block: Option<&str>,
    tp: u32,
    arch: &str,
    gpu: &str,
) -> std::ops::Range<usize> {
    let Some(spec) = block else {
        panic!(
            "muse_spark: only `--block l[..r]` (single-block decode, TP1, gfx950) is emitted. \
             Remaining: {REMAINING}"
        );
    };
    assert_eq!(
        tp, 1,
        "muse_spark --block: TP{tp} is not implemented (M1 is TP1). Remaining: {REMAINING}"
    );
    assert!(
        target_is_amd(arch, gpu),
        "muse_spark --block: M1 targets gfx950 only (got arch {arch:?} gpu {gpu:?})"
    );
    let layers = parse_block(spec, c.layers as usize);
    for l in layers.clone() {
        if let Err(e) = c.check_m1_layer(l) {
            panic!("muse_spark --block {spec}: {e}. Remaining: {REMAINING}");
        }
    }
    let rungs = emit_config::active().decode_rungs();
    assert_eq!(
        rungs,
        vec![1],
        "muse_spark --block: M1 emits the B1 decode program only (decode rungs {rungs:?})"
    );
    layers
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
    let root: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("muse_spark config JSON");
    let c = MuseCfg::parse(&root);
    let layers = m1_layers(&c, block, tp, arch, gpu);
    let _guard = EmitAmdGuard::set(true);
    let mut m = decode_program(&c, ctx, n_cu, layers.clone());
    if !gpu.is_empty() {
        m.target = packet::devbuild::gpu_fingerprint(gpu);
    }
    validate_coverage(
        dir,
        PREFIX,
        &m.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        Some(layers.clone()),
        &[".experts."],
        &[],
        &[],
    )
    .unwrap_or_else(|e| panic!("muse_spark: {e}"));
    check_gfx950_opcode_coverage(&m, true);
    if !rope_gen {
        m.bake_gen();
    }
    let lean = apply_verify_gate(&m, verify);
    let l = layers.start;
    let hidden = c.hidden as i64;
    let ckpt = dir.to_string_lossy().into_owned();
    use plow_asset::*;
    let desc = BlockDescriptor {
        model: ckpt.clone(),
        arch: "muse_spark_latent_moe".into(),
        layer: l as u32,
        kind: vec![
            if c.sliding[l] {
                "sliding_attn"
            } else {
                "full_nope_attn"
            }
            .into(),
            "latent_moe_ffn".into(),
        ],
        hidden,
        dtype: "mxfp4_experts_bf16_activations".into(),
        dims: BlockDims {
            heads: Some(c.heads as i64),
            head_dim: Some(c.hd as i64),
            kv_heads: Some(c.kv_heads as i64),
            n_exp: Some(c.n_exp as i64),
            top_k: Some(c.top_k as i64),
            moe_inter: Some(c.moe_inter as i64),
            ..Default::default()
        },
        dsa_role: None,
        inputs: vec![BlockTensor {
            name: "act.x".into(),
            shape: vec![Dim::Fixed(1), Dim::Fixed(hidden)],
            dtype: "bf16".into(),
        }],
        outputs: vec![BlockTensor {
            name: "act.xnext".into(),
            shape: vec![Dim::Fixed(1), Dim::Fixed(hidden)],
            dtype: "bf16".into(),
        }],
        carried_state: layers
            .clone()
            .map(|l| CarriedState {
                role: "kv".into(),
                tensors: vec![format!("kv.{l}.k"), format!("kv.{l}.v")],
                layout: "head_major".into(),
            })
            .collect(),
        weights: BlockWeights {
            mode: "symlink".into(),
            ckpt,
            prefix: format!("{PREFIX}layers.{l}."),
        },
        programs: BlockPrograms {
            prefill_buckets: Vec::new(),
            decode_t: 1,
        },
    };
    let section = write_block_descriptor(out, &desc);
    let outp = Path::new(out);
    std::fs::write(outp, m.to_blob_v6(&[section])).expect("write muse_spark blob");
    if !arch.is_empty() {
        write_lean_receipts(outp, &lean);
        let man = manifest::build(&m, arch, &lean);
        report_dispatch_audit(&man);
        manifest::write_config_header(&outp.with_file_name("plow_config.h"), &man)
            .expect("muse_spark config header");
        std::fs::write(
            outp.with_file_name("build.json"),
            serde_json::to_vec_pretty(&man).unwrap(),
        )
        .expect("muse_spark build manifest");
    }
    eprintln!(
        "muse_spark --block {layers:?}: {} decode packets, ctx={ctx}, ns={} -> {out}\n  NOTE: \
         M1 single-block decode only. Remaining: {REMAINING}",
        m.progs[0].insts.len(),
        decode_ns(&c, n_cu),
    );
}

fn decode_ns(c: &MuseCfg, n_cu: u32) -> u32 {
    emit_config::active().muse_decode_ns.unwrap_or_else(|| {
        attention_decode_ns(1, c.heads, c.kv_heads, n_cu, n_cu.div_ceil(c.heads / 2).max(1))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn config() -> Value {
        let n = 4;
        serde_json::json!({
            "model_type": "muse_spark",
            "text_config": {
                "model_type": "muse_spark_text",
                "hidden_size": 256, "num_attention_heads": 8, "num_key_value_heads": 2,
                "head_dim": 64, "num_hidden_layers": n, "rms_norm_eps": 1e-5,
                "post_norm_eps": 1e-8, "sliding_window": 128, "hidden_act": "silu",
                "layer_types": ["sliding_attention", "full_attention", "sliding_attention",
                                "sliding_attention"],
                "layer_rope_theta": [500000.0, 0.0, 500000.0, 500000.0],
                "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
                "mlp_layer_types": ["sparse", "sparse", "sparse", "sparse"],
                "qk_scale_factor": 5.4730064863875, "output_multiplier": 0.17677669529663687,
                "final_logit_softcapping": 20.0, "residual_gate_temperature": 0.3,
                "num_local_experts": 16, "num_experts_per_tok": 4, "moe_hidden_size": 128,
                "moe_intermediate_size": 128, "routed_scaling_factor": 1.0,
                "num_shared_experts": 0, "intermediate_size": 512, "vocab_size": 1000
            },
            "quantization_config": {
                "quant_method": "quark",
                "global_quant_config": {"weight": {"dtype": "fp4", "qscheme": "per_group",
                    "group_size": 32, "scale_format": "e8m0"}},
                "exclude": ["lm_head", "model.language_model.layers.0.mlp.experts.0.gate_proj"]
            }
        })
    }

    #[test]
    fn parses_layer_kinds_and_bf16_expert_layers() {
        let c = MuseCfg::parse(&config());
        assert_eq!(c.sliding, vec![true, false, true, true]);
        assert_eq!(c.rope, vec![true, false, true, true]);
        assert_eq!(c.mxfp4_experts, vec![false, true, true, true]);
        assert_eq!((c.heads, c.kv_heads, c.hd, c.window), (8, 2, 64, 128));
        assert!(c.check_m1_layer(0).unwrap_err().contains("bf16"));
        assert!(c.check_m1_layer(1).is_ok());
    }

    fn ops(m: &Model, op: DevOp) -> Vec<&packet::dev::DevInst> {
        m.progs[0].insts.iter().filter(|i| i.op == op as u16).collect()
    }

    #[test]
    fn full_nope_and_sliding_rope_layers_route_attention() {
        let c = MuseCfg::parse(&config());
        let full = decode_program(&c, 512, 256, 1..2);
        let fa = ops(&full, DevOp::FlashDecode);
        assert_eq!(fa.len(), 1);
        assert_eq!(fa[0].i[4], 0, "full layer: no window");
        assert!((fa[0].f[0] - c.qk_scale / 8.0).abs() < 1e-6, "q scale folded");
        let hnr = ops(&full, DevOp::QwenHeadNormRope);
        assert_eq!(hnr.len(), 4, "q, k, v cache writes and the output norm");
        assert!(hnr.iter().all(|i| i.i[2] == 0 && i.t[2] == TENSOR_NONE), "NoPE, weightless");

        let slide = decode_program(&c, 512, 256, 2..3);
        assert_eq!(ops(&slide, DevOp::FlashDecode)[0].i[4], 128);
        let hnr = ops(&slide, DevOp::QwenHeadNormRope);
        assert_eq!(hnr.iter().filter(|i| i.i[2] == 64).count(), 2, "q and k rotated");
    }

    #[test]
    fn latent_moe_routes_scaled_f32_logits_and_normed_combine() {
        let c = MuseCfg::parse(&config());
        let m = decode_program(&c, 512, 256, 1..2);
        let rt = ops(&m, DevOp::MoeRouterTopk);
        assert_eq!(rt.len(), 1);
        assert_eq!(rt[0].i[3], ROUTER_FLAGS);
        assert!((rt[0].f[1] - c.output_multiplier).abs() < 1e-7);
        assert_eq!(ops(&m, DevOp::GemvF32).len(), 1);
        let glu = ops(&m, DevOp::MoeGroupGluFp8Blk);
        assert_eq!((glu[0].i[1], glu[0].i[2], glu[0].i[6]), (128, 128, MOE_ENC_MXFP4));
        let cmb = ops(&m, DevOp::MuseMoeCombineNorm);
        assert_eq!((cmb[0].i[0], cmb[0].i[1]), (128, 4));
        let norms = ops(&m, DevOp::MuseRmsNorm);
        let modes: Vec<u32> = norms.iter().map(|i| i.i[2]).collect();
        assert_eq!(
            modes,
            vec![NORM_CENTERED, NORM_PLAIN, NORM_CENTERED, NORM_CENTERED, NORM_INPUT_SCALED]
        );
        assert_eq!(ops(&m, DevOp::MuseResidualGate).len(), 2);
        let names: Vec<&str> = m.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"moe.model.language_model.layers.1.mlp.expert_weight_table"));
        assert!(names.contains(&"act.xnext"));
    }

    #[test]
    fn multi_layer_block_chains_in_place_and_ends_in_xnext() {
        let c = MuseCfg::parse(&config());
        let m = decode_program(&c, 512, 256, 1..3);
        let x = m.tensors.iter().position(|t| t.name == "act.x").unwrap() as u32;
        let xnext = m.tensors.iter().position(|t| t.name == "act.xnext").unwrap() as u32;
        let outs: Vec<u32> = ops(&m, DevOp::MuseResidualGate).iter().map(|i| i.t[0]).collect();
        assert_eq!(outs.len(), 4);
        assert_eq!(outs[1], x);
        assert_eq!(outs[3], xnext);
        assert_eq!(ops(&m, DevOp::FlashDecode).len(), 2);
    }

    #[test]
    fn m1_scope_refusals_name_what_remains() {
        let c = MuseCfg::parse(&config());
        let full = std::panic::catch_unwind(|| m1_layers(&c, None, 1, "gfx950", ""));
        let msg = *full.unwrap_err().downcast::<String>().unwrap();
        assert!(msg.contains("lm_head") && msg.contains("TP8"), "{msg}");
        let bf16 = std::panic::catch_unwind(|| m1_layers(&c, Some("0"), 1, "gfx950", ""));
        assert!(bf16.unwrap_err().downcast::<String>().unwrap().contains("bf16"));
        assert!(std::panic::catch_unwind(|| m1_layers(&c, Some("1"), 8, "gfx950", "")).is_err());
        assert_eq!(m1_layers(&c, Some("1..3"), 1, "gfx950", ""), 1..3);
    }
}
