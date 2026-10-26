//! Muse Spark (`muse_spark` / `muse_spark_text`) device emitter for gfx950 at TP1 or TP
//! (`--block l[..r]`, or the full model ending in a greedy `in.ids`): prefill buckets, then the
//! decode rungs (one token per KV slot). Hand-routed like gpt_oss / qwen35 (the rewrite path has
//! no terms for the Muse IR ops); the graph it lowers is `crates/nn-graph/src/models/muse_spark.rs`.
//!
//! Per layer (`act.x` -> `act.xnext`; decode ops shown, prefill swaps in tiled GEMMs,
//! FlashPrefill and the grouped MoE prefill chain):
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
//! `q * qk_scale_factor` is folded into the flash scale. Decode experts run W4A16 (bf16
//! activations against dequantized MXFP4); prefill runs A4W4 (the dynamic MXFP4 activations the
//! Quark export declares), the only grouped MXFP4 prefill body gfx950 has.
use super::*;
use std::path::Path;

const PREFIX: &str = "model.language_model.";
const MOE_ENC_BF16: u32 = 0;
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
const PREFILL_LADDER: [u32; 4] = [128, 512, 1024, 2048];

/// What full-model emission still needs; the refusal names it.
const REMAINING: &str = "packed (multi-request) prefill, sliding-window KV rings, a \
    vocab-parallel head";

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
    pub softcap: f32,
    pub vocab: u32,
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
            softcap: f("final_logit_softcapping") as f32,
            vocab: u("vocab_size"),
            temperature: f("residual_gate_temperature") as f32,
            n_exp: u("num_local_experts"),
            top_k: u("num_experts_per_tok"),
            moe_hidden: u("moe_hidden_size"),
            moe_inter: u("moe_intermediate_size"),
            route_scale: f("routed_scaling_factor") as f32,
            mxfp4_experts,
        }
    }

    /// The block refusals for one layer, or `Ok`.
    fn check_m1_layer(&self, l: usize) -> Result<(), String> {
        if !self.sparse[l] {
            return Err(format!("layer {l} has a dense MLP (no route)"));
        }
        Ok(())
    }
}

/// Tensor parallelism. One blob serves every rank: weights are declared at the rank's shard
/// size (`plowrt::asset::shard::slice_for`) and the row-parallel partials go to the two peer
/// slots, which the collectives use strictly alternately — a rank rewrites a slot only after the
/// next rendezvous, which every peer reaches only after finishing its reads of that slot.
struct Tp {
    n: u32,
    /// Peer slots A (`act.og_tp`) and B (`act.dg_tp`).
    slots: [u32; 2],
    slot_b: u32,
    next: usize,
    xgate: u32,
    xr_cus: Vec<u32>,
    /// `pre_expert_proj` / `post_expert_proj` row-parallel (else replicated).
    latent: bool,
}

impl Tp {
    fn on(&self) -> bool {
        self.n > 1
    }
    fn local(&self, w: u32) -> u32 {
        assert_eq!(w % self.n, 0, "muse_spark: --num-gpus {} must divide {w}", self.n);
        w / self.n
    }
    /// The peer slot the next row-parallel producer writes.
    fn partial(&self) -> u32 {
        self.slots[self.next]
    }
}

/// One row norm's weight (`TENSOR_NONE` = weightless), mode, eps and output scale (0 = 1).
#[derive(Clone, Copy)]
struct NormSpec {
    w: u32,
    mode: u32,
    eps: f32,
    oscale: f32,
}

/// The norm that follows a layer when [`DevOp::MuseNormGateNorm`] takes it into the layer's
/// tail: the next layer's input norm, or the final norm.
#[derive(Clone, Copy)]
struct NextNorm {
    out: u32,
    n: NormSpec,
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
    /// TP: the reduced bf16 unweighted expert outputs `[rows][k][moe_hidden]`.
    ysum: u32,
    /// TP latent row-parallel: this rank's input slices of `xn2` / `ylat`. One row: column views
    /// of the row; several rows: views of the band-major copies `hn2_bm` / `ylat_bm` the norm and
    /// the combine write (`[tp][rows][width/tp]`).
    hn2_band: u32,
    ylat_band: u32,
    hn2_bm: u32,
    ylat_bm: u32,
    /// Grouped MoE prefill scratch (`MoeAlignPf` / `MoeGroup{Glu,Down}Pf`).
    meta: u32,
    row_token: u32,
    row_partidx: u32,
    row_gate: u32,
    fu_scale: u32,
}

/// The full-model ends: `in.ids` -> embed -> weightless norm -> layers -> head -> `in.ids`.
struct Ends {
    ids: u32,
    emb: u32,
    logits: u32,
    amax: u32,
}

struct Emitter<'a> {
    b: Builder,
    c: &'a MuseCfg,
    ctx: u32,
    /// Rows: the prefill bucket, or the decode rung (one token per KV slot).
    t: u32,
    prefill: bool,
    ns: u32,
    pos: u32,
    kvlen: u32,
    cos: u32,
    sin: u32,
    a: Acts,
    tp: Tp,
}

impl Emitter<'_> {
    fn cus(&self, n: u32) -> Vec<u32> {
        (0..n.clamp(1, self.b.n_cu())).collect()
    }
    /// One workgroup per row (the row ops' work split).
    fn rows(&self) -> Vec<u32> {
        self.cus(self.t)
    }
    fn w(&mut self, name: &str, bytes: u64) -> u32 {
        self.b.tensor(name, bytes)
    }
    /// `out[t, n] = x[t, k] . W[n, k]^T`: the GEMV at one row, the tile-picked GEMM in prefill,
    /// `GemmSmall` on a batched decode rung (the decode object's only tile).
    fn proj(&mut self, out: u32, x: u32, w: u32, n: u32, k: u32, dep: u32) -> u32 {
        let t = self.t;
        let op = if t == 1 {
            DevOp::Gemv
        } else if self.prefill {
            pick_tile(t, n, k, self.b.n_cu(), kernelcaps::QuantScheme::None)
        } else {
            DevOp::GemmSmall
        };
        self.b.emit(op, self.b.all(), &[dep], |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = w;
            d.i[0] = t;
            d.i[1] = n;
            d.i[2] = k;
        })
    }
    /// Row RMSNorm over `t` rows. `band = (out_bm, cols)` also writes the band-major copy
    /// `out_bm[c / cols][row][c % cols]` the row-parallel latent projections read.
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &mut self,
        out: u32,
        x: u32,
        w: u32,
        dim: u32,
        mode: u32,
        eps: f32,
        dep: &[u32],
        band: Option<(u32, u32)>,
    ) -> u32 {
        let t = self.t;
        self.b.emit(DevOp::MuseRmsNorm, self.rows(), dep, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = w;
            d.i[0] = t;
            d.i[1] = dim;
            d.i[2] = mode;
            d.f[0] = eps;
            if let Some((bm, cols)) = band {
                d.t[3] = bm;
                d.i[3] = cols;
            }
        })
    }
    /// `out_res = gate_res(res, norm(h | n1), g)`, then `out_n = norm(out_res | n2)` in one
    /// [`DevOp::MuseNormGateNorm`] (a [`Self::norm`] / [`Self::residual_gate`] / [`Self::norm`]
    /// chain).
    #[allow(clippy::too_many_arguments)]
    fn norm_gate_norm(
        &mut self,
        out_res: u32,
        h: u32,
        n1: NormSpec,
        res: u32,
        g: u32,
        out_n: u32,
        n2: NormSpec,
        dep: &[u32],
        band: Option<(u32, u32)>,
    ) -> u32 {
        let (dim, t, temp) = (self.c.hidden, self.t, self.c.temperature);
        self.b.emit(DevOp::MuseNormGateNorm, self.rows(), dep, |d| {
            d.t = [out_res, h, n1.w, res, g, out_n, n2.w, TENSOR_NONE];
            d.i[..4].copy_from_slice(&[t, dim, n1.mode, n2.mode]);
            if let Some((bm, cols)) = band {
                d.t[7] = bm;
                d.i[4] = cols;
            }
            d.i[5] = RESIDUAL_GATE_EPS.to_bits();
            d.i[6] = n2.oscale.to_bits();
            d.f[0] = n1.eps;
            d.f[1] = n2.eps;
            d.j[1] = temp.to_bits();
        })
    }
    fn residual_gate(&mut self, out: u32, res: u32, h: u32, g: u32, deps: &[u32]) -> u32 {
        let (n, t) = (self.c.hidden, self.t);
        let (temp, cus) = (self.c.temperature, self.cus((t * n).div_ceil(512)));
        self.b.emit(DevOp::MuseResidualGate, cus, deps, |d| {
            d.t[0] = out;
            d.t[1] = res;
            d.t[2] = h;
            d.t[3] = g;
            d.i[0] = n;
            d.i[1] = t;
            d.f[0] = temp;
            d.f[1] = RESIDUAL_GATE_EPS;
        })
    }
    /// All-reduce the partial the last producer wrote into [`Tp::partial`] into `out`: the
    /// one-shot at decode, the two-shot in prefill.
    fn reduce(&mut self, dep: u32, out: u32, elems: u32) -> u32 {
        let slot = self.tp.next as u32 * self.tp.slot_b;
        self.tp.next ^= 1;
        let (n, cus, decode) = (self.tp.n, self.tp.xr_cus.clone(), !self.prefill);
        crate::emit_xreduce(&mut self.b, &mut self.tp.xgate, decode, &cus, dep, out, elems, n, slot)
    }
    /// Weightless per-head RMSNorm over `t` rows, optionally NeoX-rotated, optionally written
    /// into the KV cache at `pos[row]` (`cache`; prefill rows share the slot the host rebased
    /// the cache onto, decode row `b` is slot `b`). `normalize = false` is a plain copy (V).
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
        let (hd, eps, ctx, t) = (self.c.hd, self.c.eps, self.ctx, self.t);
        let (cos, sin, pos) = (self.cos, self.sin, self.pos);
        let prefill = cache && self.prefill;
        let cus = self.cus((t * heads).div_ceil(8));
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
            d.i[..7].copy_from_slice(&[
                heads,
                hd,
                if rope { hd } else { 0 },
                t,
                if cache { ctx } else { 0 },
                normalize as u32,
                prefill as u32,
            ]);
            d.f[0] = eps;
        })
    }

    /// `normed`: `act.hn` already holds this layer's input norm (the previous tail wrote it).
    /// `next`: fuse the following norm into this layer's tail.
    fn layer(
        &mut self,
        l: usize,
        src: u32,
        dst: u32,
        dep: Option<u32>,
        normed: bool,
        next: Option<NextNorm>,
    ) -> u32 {
        let c = self.c;
        let p = format!("{PREFIX}layers.{l}.");
        let (h, hd, m, e, t) = (c.hidden, c.hd, c.moe_hidden, c.n_exp, self.t);
        let (nh, nkv) = (self.tp.local(c.heads), self.tp.local(c.kv_heads));
        let (qd, kd, imoe) = (nh * hd, nkv * hd, self.tp.local(c.moe_inter));
        let (h64, qd64, kd64, m64) = (h as u64, qd as u64, kd as u64, m as u64);
        let a = &self.a;
        let (hn, q_raw, k_raw, v_raw, gate, q) = (a.hn, a.q_raw, a.k_raw, a.v_raw, a.gate, a.q);
        let (opart, mlpart, attn, at, o, on) = (a.opart, a.mlpart, a.attn, a.at, a.o, a.on);
        let (xmid, hn2, rlogit, tab, xe0, xe) = (a.xmid, a.hn2, a.rlogit, a.tab, a.xe0, a.xe);
        let (fu, part, ylat, y, yn, ysum) = (a.fu, a.part, a.ylat, a.y, a.yn, a.ysum);
        let (hn2_band, ylat_band, hn2_bm, ylat_bm) = (a.hn2_band, a.ylat_band, a.hn2_bm, a.ylat_bm);
        let (meta, row_token, row_partidx, row_gate, fu_scale) =
            (a.meta, a.row_token, a.row_partidx, a.row_gate, a.fu_scale);
        let (tp, latent) = (self.tp.on(), self.tp.on() && self.tp.latent);
        // Row-parallel latent projections contract over this rank's 1/tp of their input.
        let (pre_k, post_k) = if latent {
            (self.tp.local(h), self.tp.local(m))
        } else {
            (h, m)
        };

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
        let w_pre = self.w(&format!("{p}mlp.pre_expert_proj.weight"), m64 * pre_k as u64 * BF16);
        let w_pre_n = self.w(&format!("{p}mlp.pre_expert_norm.weight"), m64 * BF16);
        let w_pen = self.w(&format!("{p}mlp.experts.post_expert_norm.weight"), m64 * BF16);
        let w_postp = self.w(&format!("{p}mlp.post_expert_proj.weight"), h64 * post_k as u64 * BF16);
        let ewt = self.w(&format!("moe.{p}mlp.expert_weight_table"), e as u64 * 3 * 8);
        let est = self.w(&format!("moe.{p}mlp.expert_scale_table"), e as u64 * 3 * 8);
        let slots = self.b.tensor_bytes(self.kvlen) / F32;
        let kc = self.w(&format!("kv.{l}.k"), slots * kd64 * self.ctx as u64 * BF16);
        let vc = self.w(&format!("kv.{l}.v"), slots * kd64 * self.ctx as u64 * BF16);

        // ---- attention ----
        let ngn = self.ngn();
        let d_hn = if normed {
            dep.expect("a fused input norm has a producer")
        } else {
            self.norm(hn, src, w_in, h, NORM_CENTERED, c.eps, dep.as_slice(), None)
        };
        let qkvg = t == 1
            && !self.prefill
            && emit_config::active().muse_qkvg
            && crate::gemv_staged_rows(t) as u64 * h64 <= crate::gm_lds_halves();
        let (d_q, d_k, d_v, d_g) = if qkvg {
            let f = self.b.emit(DevOp::GemvQkvg, self.b.all(), &[d_hn], |d| {
                d.t[..8].copy_from_slice(&[q_raw, hn, wq, k_raw, wk, v_raw, wv, gate]);
                d.i[..7].copy_from_slice(&[t, qd, h, kd, kd, qd, wg]);
            });
            (f, f, f, f)
        } else {
            (
                self.proj(q_raw, hn, wq, qd, h, d_hn),
                self.proj(k_raw, hn, wk, kd, h, d_hn),
                self.proj(v_raw, hn, wv, kd, h, d_hn),
                self.proj(gate, hn, wg, qd, h, d_hn),
            )
        };
        let rope = c.rope[l];
        let d_qn = self.head_norm(q, q_raw, nh, rope, false, true, d_q);
        let d_kn = self.head_norm(kc, k_raw, nkv, rope, true, true, d_k);
        let d_vn = self.head_norm(vc, v_raw, nkv, false, true, false, d_v);
        let (ns, kvlen, ctx) = (self.ns, self.kvlen, self.ctx);
        let window = if c.sliding[l] { c.window } else { 0 };
        let scale = c.qk_scale / (hd as f32).sqrt();
        let d_fa = if self.prefill {
            // n_kv / q_pos0 (i1 / i4) are the chunk's, patched by the host (`rebase_chunk_rows`).
            self.b
                .emit(DevOp::FlashPrefill, self.b.all(), &[d_qn, d_kn, d_vn], |d| {
                    d.t[..5].copy_from_slice(&[opart, mlpart, q, kc, vc]);
                    d.i.copy_from_slice(&[t, t, nh, nkv, 0, window, hd, ns]);
                    d.f[0] = scale;
                    d.j[0] = ctx;
                    d.j[1] = KV_MASK_NONE;
                })
        } else {
            self.b
                .emit(DevOp::FlashDecode, self.b.all(), &[d_qn, d_kn, d_vn], |d| {
                    d.t[..6].copy_from_slice(&[opart, mlpart, q, kc, vc, kvlen]);
                    d.i.copy_from_slice(&[t, nh, nkv, ctx, window, ns, hd, u32::MAX]);
                    d.f[0] = scale;
                })
        };
        let d_fm = self.b.emit(DevOp::FlashMerge, self.cus(t * nh), &[d_fa], |d| {
            d.t[..3].copy_from_slice(&[attn, opart, mlpart]);
            d.i[..4].copy_from_slice(&[t, nh, ns, hd]);
        });
        let d_on = self.head_norm(at, attn, nh, false, false, true, d_fm);
        let d_gate = self.b.emit(
            DevOp::QwenSigmoidGate,
            self.cus((t * qd).div_ceil(512)),
            &[d_on, d_g],
            |d| {
                d.t[..4].copy_from_slice(&[at, at, gate, TENSOR_NONE]);
                d.i[..2].copy_from_slice(&[qd, t]);
            },
        );
        let d_o = if tp {
            let og = self.tp.partial();
            let d = self.proj(og, at, wo, h, qd, d_gate);
            self.reduce(d, o, t * h)
        } else {
            self.proj(o, at, wo, h, qd, d_gate)
        };
        let band = (latent && t > 1).then(|| (hn2_bm, pre_k));
        let d_hn2 = if ngn {
            let n1 = NormSpec { w: TENSOR_NONE, mode: NORM_PLAIN, eps: c.post_eps, oscale: 0.0 };
            let n2 = NormSpec { w: w_ff, mode: NORM_CENTERED, eps: c.eps, oscale: 0.0 };
            self.norm_gate_norm(xmid, o, n1, src, g_attn, hn2, n2, &[d_o], band)
        } else {
            let d_pn = self.norm(on, o, TENSOR_NONE, h, NORM_PLAIN, c.post_eps, &[d_o], None);
            let d_xm = self.residual_gate(xmid, src, on, g_attn, &[d_pn]);
            self.norm(hn2, xmid, w_ff, h, NORM_CENTERED, c.eps, &[d_xm], band)
        };

        // ---- latent MoE ----
        let d_rl = self.b.emit(DevOp::GemvF32, self.b.all(), &[d_hn2], |d| {
            d.t[0] = rlogit;
            d.t[1] = hn2;
            d.t[2] = wr;
            d.i[0] = t;
            d.i[1] = e;
            d.i[2] = h;
        });
        let (top_k, route_scale, logit_scale) = (c.top_k, c.route_scale, c.output_multiplier);
        let (rt_op, rt_cus) = if self.prefill {
            (DevOp::MoeRouterTopkPf, self.rows())
        } else {
            (DevOp::MoeRouterTopk, vec![0])
        };
        let rt_cus = if t > 1 { self.rows() } else { rt_cus };
        let prefill = self.prefill;
        let d_rt = self.b.emit(rt_op, rt_cus, &[d_rl], |d| {
            d.t[0] = tab;
            d.t[1] = rlogit;
            d.t[3] = wr_bias;
            d.i[1] = e;
            d.i[2] = top_k;
            d.i[3] = ROUTER_FLAGS;
            // Rows: MoeRouterTopkPf's T, the decode router's row count (0 = one row).
            if prefill || t > 1 {
                d.i[4] = t;
            }
            d.i[6] = 1;
            d.i[7] = 1;
            d.f[0] = route_scale;
            d.f[1] = logit_scale;
        });
        let d_xe0 = if latent {
            let pg = self.tp.partial();
            let d = self.proj(pg, hn2_band, w_pre, m, pre_k, d_hn2);
            self.reduce(d, xe0, t * m)
        } else {
            self.proj(xe0, hn2, w_pre, m, h, d_hn2)
        };
        let d_xe = self.norm(xe, xe0, w_pre_n, m, NORM_CENTERED, c.eps, &[d_xe0], None);
        // TP: each rank's down writes its bf16 partial y into the peer slot (decode unweighted,
        // the prefill scatter gate-weighted); the post-expert norm is nonlinear per (token,
        // expert), so the reduce precedes it.
        let dn_out = if tp { self.tp.partial() } else { part };
        let ybf = match (tp, self.prefill) {
            (false, _) => 0,
            (true, false) => 1,
            (true, true) => 2,
        };
        // Quark `exclude`d layers keep bf16 experts: same grouped ops, bf16 encoding.
        let enc = if c.mxfp4_experts[l] { MOE_ENC_MXFP4 } else { MOE_ENC_BF16 };
        let d_dn = if self.prefill {
            let d_al = self.b.emit(DevOp::MoeAlignPf, vec![0], &[d_rt], |d| {
                d.t[..5].copy_from_slice(&[meta, tab, row_token, row_partidx, row_gate]);
                d.i[..3].copy_from_slice(&[t, e, top_k]);
            });
            let mx = enc == MOE_ENC_MXFP4;
            let d_glu = self
                .b
                .emit(DevOp::MoeGroupGluPf, self.b.all(), &[d_al, d_xe], |d| {
                    d.t[..6].copy_from_slice(&[fu, xe, ewt, est, meta, row_token]);
                    // A4W4: the fused bridge skips pad rows and writes the E8M0 rows DOWN reads.
                    if mx {
                        d.t[6] = row_partidx;
                        d.t[7] = fu_scale;
                    }
                    d.i[0] = imoe;
                    d.i[1] = m;
                    d.i[2] = e;
                    d.i[3] = enc;
                    d.i[5] = MOE_ACT_SILU;
                });
            self.b
                .emit(DevOp::MoeGroupDownPf, self.b.all(), &[d_glu], |d| {
                    d.t[..5].copy_from_slice(&[dn_out, fu, ewt, est, meta]);
                    if mx {
                        d.t[5] = fu_scale;
                    }
                    d.t[6] = row_partidx;
                    d.t[7] = row_gate;
                    d.i[0] = m;
                    d.i[1] = imoe;
                    d.i[2] = e;
                    d.i[3] = enc;
                    d.i[7] = u32::from(tp);
                })
        } else {
            // Batched rows: `t*k` flat slots, the GLU reads row `slot / k` of `xe`.
            let d_glu = self
                .b
                .emit(DevOp::MoeGroupGluFp8Blk, self.b.all(), &[d_rt, d_xe], |d| {
                    d.t[..5].copy_from_slice(&[fu, xe, tab, ewt, est]);
                    d.i[0] = t * top_k;
                    d.i[1] = imoe;
                    d.i[2] = m;
                    d.i[3] = e;
                    if t > 1 {
                        d.i[4] = top_k;
                    }
                    d.i[5] = MOE_ACT_SILU;
                    d.i[6] = enc;
                });
            self.b
                .emit(DevOp::MoeGroupDownFp8Blk, self.b.all(), &[d_glu], |d| {
                    d.t[..5].copy_from_slice(&[dn_out, fu, tab, ewt, est]);
                    d.i[0] = t * top_k;
                    d.i[1] = m;
                    d.i[2] = imoe;
                    d.i[3] = e;
                    d.i[6] = enc;
                    d.i[7] = ybf;
                })
        };
        let (d_dn, ysrc) = if tp {
            (self.reduce(d_dn, ysum, t * top_k * m), ysum)
        } else {
            (d_dn, part)
        };
        let post_eps = c.post_eps;
        let cmb_cus = if t > 1 { self.rows() } else { self.cus(m.div_ceil(512)) };
        let ylat_band_out = latent && t > 1;
        let d_cmb = self
            .b
            .emit(DevOp::MuseMoeCombineNorm, cmb_cus, &[d_dn], |d| {
                d.t[..4].copy_from_slice(&[ylat, ysrc, tab, w_pen]);
                d.i[0] = m;
                d.i[1] = top_k;
                d.i[2] = ybf;
                if t > 1 {
                    d.i[3] = t;
                }
                if ylat_band_out {
                    d.t[4] = ylat_bm;
                    d.i[4] = post_k;
                }
                d.f[0] = post_eps;
            });
        let d_y = if latent {
            let pg = self.tp.partial();
            let d = self.proj(pg, ylat_band, w_postp, h, post_k, d_cmb);
            self.reduce(d, y, t * h)
        } else {
            self.proj(y, ylat, w_postp, h, m, d_cmb)
        };
        if let Some(nx) = next.filter(|_| ngn) {
            let n1 = NormSpec { w: w_post, mode: NORM_INPUT_SCALED, eps: c.post_eps, oscale: 0.0 };
            return self.norm_gate_norm(dst, y, n1, xmid, g_ff, nx.out, nx.n, &[d_y], None);
        }
        let d_yn = self.norm(yn, y, w_post, h, NORM_INPUT_SCALED, c.post_eps, &[d_y], None);
        self.residual_gate(dst, xmid, yn, g_ff, &[d_yn])
    }

    /// [`DevOp::MuseNormGateNorm`] replaces the norm / gate / norm chains (one-row decode,
    /// opt-in; the batched rungs are not measured).
    fn ngn(&self) -> bool {
        !self.prefill && self.t == 1 && emit_config::active().muse_ngn
    }
}

/// One program's tensors, so the next program adopts them (one shared table, every tensor at
/// its widest use).
struct Phase {
    prog: packet::devbuild::Program,
    tensors: Vec<packet::devbuild::TensorDecl>,
    gen: Vec<GenTensor>,
}

/// What every program of one packet shares.
struct Shape {
    ctx: u32,
    n_cu: u32,
    tp: u32,
    layers: std::ops::Range<usize>,
    full: bool,
    /// KV slots: the widest decode rung.
    slots: u32,
    /// Peer-slot bytes, sized at the widest program (`devblob::recover_tp` reads `max(i[2])`).
    slot_b: u32,
}

/// One program over `s.layers`: a `t`-row prefill bucket or a `t`-row decode rung. The last
/// layer writes `act.xnext`; earlier ones write `act.x` back in place (it is dead after the
/// layer's attention residual).
fn program(
    c: &MuseCfg,
    s: &Shape,
    t: u32,
    prefill: bool,
    tensors: Option<Vec<packet::devbuild::TensorDecl>>,
) -> Phase {
    let (ctx, n_cu, tp, full) = (s.ctx, s.n_cu, s.tp, s.full);
    let mut b = Builder::new(n_cu);
    // Decode is one wave class; a prefill bucket keeps its FlashPrefill segments for the
    // 4-wave flash object.
    if !prefill {
        b.force_uniseg();
    }
    if let Some(tensors) = tensors {
        b.adopt_tensors(tensors);
    }
    b.set_tensor_dedup(true);
    let local = |w: u32| {
        assert_eq!(w % tp, 0, "muse_spark: --num-gpus {tp} must divide {w}");
        w / tp
    };
    let (h, hd, m, k, e) = (c.hidden, c.hd, c.moe_hidden, c.top_k, c.n_exp);
    let (nh, nkv, imoe) = (local(c.heads), local(c.kv_heads), local(c.moe_inter));
    let tr = t as u64;
    let (qd, kd) = ((nh * hd) as u64, (nkv * hd) as u64);
    // `in.ids` first: handle 0 is the "absent" value of several optional operand slots.
    let ids = b.tensor("in.ids", ctx as u64 * F32);
    // The engine reads its context capacity off `in.pos`.
    let pos = b.tensor("in.pos", ctx as u64 * F32);
    let kvlen = b.tensor("in.kvlen", s.slots as u64 * F32);
    let [gc, gs] = GenTensor::rope_pair(ctx, hd, c.theta, 1.0, RopeScale::None);
    let cos = b.tensor_gen("in.cos_full", gc.byte_len(), gc);
    let sin = b.tensor_gen("in.sin_full", gs.byte_len(), gs);
    let ns = if prefill {
        n_cu.div_ceil((t.div_ceil(Q_TILE_ROWS) * nh).max(1)).max(1)
    } else {
        decode_ns(t, nh, nkv, n_cu)
    };
    let (h64, m64) = (h as u64, m as u64);
    let latent = tp > 1 && emit_config::active().muse_latent_tp;
    // Gathered rows of the grouped prefill: every expert pads to a whole `MPF_BM` tile.
    let pad_rows = tr * k as u64 + (e * (crate::mla::MPF_BM - 1)) as u64;
    let fu_bytes = if prefill {
        pad_rows * imoe as u64 * BF16
    } else {
        tr * k as u64 * imoe as u64 * BF16
    };
    let mut act = |n: &str, bytes: u64| b.tensor(&format!("act.{n}"), bytes);
    let mut a = Acts {
        x: act("x", tr * h64 * BF16),
        xnext: act("xnext", tr * h64 * BF16),
        hn: act("hn", tr * h64 * BF16),
        q_raw: act("qg", tr * qd * BF16),
        k_raw: act("kg", tr * kd * BF16),
        v_raw: act("vg", tr * kd * BF16),
        gate: act("ogate", tr * qd * BF16),
        q: act("q", tr * qd * BF16),
        opart: act("opart", tr * nh as u64 * ns as u64 * hd as u64 * F32),
        mlpart: act("mlpart", tr * nh as u64 * ns as u64 * 2 * F32),
        attn: act("attn", tr * qd * BF16),
        at: act("at", tr * qd * BF16),
        o: act("o", tr * h64 * BF16),
        on: act("on", tr * h64 * BF16),
        xmid: act("xmid", tr * h64 * BF16),
        hn2: act("xn2", tr * h64 * BF16),
        rlogit: act("rlogit", tr * e as u64 * F32),
        tab: act("tab", tr * k as u64 * 8),
        xe0: act("xe0", tr * m64 * BF16),
        xe: act("xe", tr * m64 * BF16),
        fu: act("fu", fu_bytes),
        // TP writes the down partials into the peer slot instead.
        part: act("part", if tp > 1 { 1 } else { tr } * k as u64 * m64 * F32),
        ylat: act("ylat", tr * m64 * BF16),
        y: act("y", tr * h64 * BF16),
        yn: act("yn", tr * h64 * BF16),
        ysum: TENSOR_NONE,
        hn2_band: TENSOR_NONE,
        ylat_band: TENSOR_NONE,
        hn2_bm: TENSOR_NONE,
        ylat_bm: TENSOR_NONE,
        meta: TENSOR_NONE,
        row_token: TENSOR_NONE,
        row_partidx: TENSOR_NONE,
        row_gate: TENSOR_NONE,
        fu_scale: TENSOR_NONE,
    };
    if prefill {
        a.meta = act("moe_meta", (3 * e as u64 + 1) * F32);
        a.row_token = act("moe_rowtok", pad_rows * F32);
        a.row_partidx = act("moe_rowpart", pad_rows * F32);
        a.row_gate = act("moe_rowgate", pad_rows * F32);
        if c.mxfp4_experts.iter().any(|&mx| mx) {
            a.fu_scale = act("moe_fuscale", pad_rows * (imoe / 32) as u64);
        }
    }
    let ends = full.then(|| {
        let scored = if prefill { 1 } else { tr };
        Ends {
            ids,
            emb: b.tensor("act.emb", tr * h64 * BF16),
            logits: b.tensor("act.logits", scored * c.vocab as u64 * BF16),
            amax: b.tensor("act.amax", scored * AMAX_BLOCKS as u64 * 8),
        }
    });
    let tp = if tp > 1 {
        a.ysum = b.tensor("act.ysum", tr * k as u64 * m64 * BF16);
        if latent {
            // One row: column views of the row. Several: views of the band-major copies; named
            // `@bandr`, not `@band<t>`, so the sequence-parallel row-band rebind skips them.
            let (xn2, yl) = if t == 1 {
                (a.hn2, a.ylat)
            } else {
                a.hn2_bm = b.tensor("act.xn2_bm", tr * h64 * BF16);
                a.ylat_bm = b.tensor("act.ylat_bm", tr * m64 * BF16);
                (a.hn2_bm, a.ylat_bm)
            };
            let tag = if t == 1 { "band1".to_string() } else { format!("bandr{t}") };
            let mut view = |base: u32, n: u32| {
                let name = format!("{}@{tag}", b.tensor_name(base));
                b.tensor(&name, tr * n as u64 * BF16)
            };
            a.hn2_band = view(xn2, local(h));
            a.ylat_band = view(yl, local(m));
        }
        Tp {
            n: tp,
            slots: [
                b.tensor("act.og_tp", s.slot_b as u64),
                b.tensor("act.dg_tp", s.slot_b as u64),
            ],
            slot_b: s.slot_b,
            next: 0,
            xgate: 0,
            xr_cus: (0..emit_config::active().xr_cus.unwrap_or(n_cu).clamp(1, n_cu)).collect(),
            latent,
        }
    } else {
        Tp {
            n: 1,
            slots: [TENSOR_NONE; 2],
            slot_b: 0,
            next: 0,
            xgate: 0,
            xr_cus: Vec::new(),
            latent: false,
        }
    };
    let (x, xnext) = (a.x, a.xnext);
    let mut em = Emitter {
        b,
        c,
        ctx,
        t,
        prefill,
        ns,
        pos,
        kvlen,
        cos,
        sin,
        a,
        tp,
    };
    let mut dep = None;
    if let Some(e) = &ends {
        let table = em.w(&format!("{PREFIX}embed_tokens.weight"), c.vocab as u64 * h64 * BF16);
        let d_emb = em.b.emit(DevOp::Embed, em.b.all(), &[], |d| {
            d.t[..3].copy_from_slice(&[e.emb, table, e.ids]);
            d.i[0] = t;
            d.i[1] = h;
            d.f[0] = 1.0;
        });
        dep = Some(em.norm(x, e.emb, TENSOR_NONE, h, NORM_PLAIN, c.eps, &[d_emb], None));
    }
    let last = s.layers.end - 1;
    let ngn = em.ngn();
    let mut normed = false;
    for l in s.layers.clone() {
        let dst = if l == last && !full { xnext } else { x };
        // The norm after this layer, when the tail fuses it.
        let next = if !ngn {
            None
        } else if l < last {
            let w = em.w(&format!("{PREFIX}layers.{}.input_layernorm.weight", l + 1), h64 * BF16);
            Some(NextNorm { out: em.a.hn, n: NormSpec { w, mode: NORM_CENTERED, eps: c.eps, oscale: 0.0 } })
        } else if full {
            let w = em.w(&format!("{PREFIX}norm.weight"), h64 * BF16);
            let n = NormSpec { w, mode: NORM_PLAIN, eps: c.eps, oscale: c.output_multiplier };
            Some(NextNorm { out: em.a.hn, n })
        } else {
            None
        };
        dep = Some(em.layer(l, x, dst, dep, normed, next));
        normed = next.is_some();
    }
    if let Some(e) = &ends {
        let w_final = em.w(&format!("{PREFIX}norm.weight"), h64 * BF16);
        let lm = em.w("lm_head.weight", c.vocab as u64 * h64 * BF16);
        let (hn, vocab, mult, cap) = (em.a.hn, c.vocab, c.output_multiplier, c.softcap);
        // `cap * tanh(mult * logits / cap)`: the multiplier rides the final norm's output.
        let d_fn = if normed {
            dep.expect("the last layer's tail wrote the final norm")
        } else {
            em.b.emit(DevOp::MuseRmsNorm, em.rows(), dep.as_slice(), |d| {
                d.t[..3].copy_from_slice(&[hn, x, w_final]);
                d.i[..3].copy_from_slice(&[t, h, NORM_PLAIN]);
                d.f[0] = c.eps;
                d.f[1] = mult;
            })
        };
        // Prefill scores its last row only (a_row0, which the host moves to the chunk's last
        // real row); decode scores every slot.
        let (lm_op, lm_rows) = match (prefill, t) {
            (true, _) | (false, 1) => (DevOp::Gemv, 1),
            (false, _) => (DevOp::GemmSmall, t),
        };
        let d_lm = em.b.emit(lm_op, em.b.all(), &[d_fn], |d| {
            d.t[..3].copy_from_slice(&[e.logits, hn, lm]);
            d.i[..3].copy_from_slice(&[lm_rows, vocab, h]);
            if prefill {
                d.i[4] = t - 1;
            }
        });
        let d_cap = em.b.emit(DevOp::SoftCap, em.b.all(), &[d_lm], |d| {
            d.t[..2].copy_from_slice(&[e.logits, e.logits]);
            d.i[0] = lm_rows * vocab;
            d.f[0] = cap;
        });
        let nb = if lm_rows > 1 { lm_rows } else { 0 };
        let d_am = em.b.emit(DevOp::Argmax, (0..AMAX_BLOCKS).collect(), &[d_cap], |d| {
            d.t[..2].copy_from_slice(&[e.amax, e.logits]);
            d.i[0] = vocab;
            d.i[1] = nb;
        });
        dep = Some(em.b.emit(DevOp::ArgmaxFin, vec![0], &[d_am], |d| {
            d.t[..2].copy_from_slice(&[e.ids, e.amax]);
            d.i[0] = AMAX_BLOCKS;
            d.i[1] = nb;
        }));
    }
    let _ = dep;
    let tensors = em.b.tensors();
    let gen = em.b.gen_tensors();
    Phase {
        prog: em.b.finish(),
        tensors,
        gen,
    }
}

/// The packet: prefill `buckets` ascending, then the decode `rungs` ascending, one tensor table.
#[allow(clippy::too_many_arguments)]
fn model(
    c: &MuseCfg,
    ctx: u32,
    n_cu: u32,
    tp: u32,
    layers: std::ops::Range<usize>,
    full: bool,
    buckets: &[u32],
    rungs: &[u32],
) -> Model {
    let widest = buckets.iter().chain(rungs).copied().max().expect("a program");
    // Slot bytes: the widest partial (the k expert outputs) of the widest program, and a whole
    // number of the recovered `hidden` (the host sizes the peer region as max_tokens * hidden * 2).
    let shape = Shape {
        ctx,
        n_cu,
        tp,
        layers,
        full,
        slots: *rungs.last().expect("a decode rung"),
        slot_b: widest * c.hidden.max(c.top_k * c.moe_hidden) * 2,
    };
    let mut tensors = None;
    let (mut progs, mut prog_t, mut gen) = (Vec::new(), Vec::new(), Vec::new());
    for (t, prefill) in buckets.iter().map(|&t| (t, true)).chain(rungs.iter().map(|&t| (t, false))) {
        let ph = program(c, &shape, t, prefill, tensors.take());
        tensors = Some(ph.tensors);
        gen = ph.gen;
        progs.push(ph.prog);
        prog_t.push(t);
    }
    Model {
        n_cu,
        target: 0,
        tensors: tensors.expect("at least one program"),
        progs,
        kv_row_insts: Vec::new(),
        prog_t,
        gen,
    }
}

/// The B1 decode program alone (the M1/F1/F2 packet).
#[cfg(test)]
fn decode_program(
    c: &MuseCfg,
    ctx: u32,
    n_cu: u32,
    tp: u32,
    layers: std::ops::Range<usize>,
    full: bool,
) -> Model {
    model(c, ctx, n_cu, tp, layers, full, &[], &[1])
}

/// Prefill buckets: the ladder up to `PLOW_MAX_CHUNK` (default one sliding window), plus the
/// always-shipped rung once the context reaches it.
fn prefill_buckets(c: &MuseCfg, ctx: u32) -> Vec<u32> {
    let chunk = max_chunk(c.window).min(ctx);
    crate::with_always_rung(
        PREFILL_LADDER.into_iter().filter(|&b| b <= chunk).collect(),
        ctx,
    )
}

/// Validate `--block` and the M1 scope, then return the layer range.
pub(crate) fn m1_layers(
    c: &MuseCfg,
    block: Option<&str>,
    tp: u32,
    arch: &str,
    gpu: &str,
) -> std::ops::Range<usize> {
    let spec = block.unwrap_or("full model");
    assert!(
        c.kv_heads % tp == 0,
        "muse_spark {spec}: TP{tp} must divide the {} kv heads",
        c.kv_heads
    );
    assert!(
        target_is_amd(arch, gpu),
        "muse_spark {spec}: only gfx950 is emitted (got arch {arch:?} gpu {gpu:?})"
    );
    let layers = block.map_or(0..c.layers as usize, |b| parse_block(b, c.layers as usize));
    for l in layers.clone() {
        if let Err(e) = c.check_m1_layer(l) {
            panic!("muse_spark {spec}: {e}. Remaining: {REMAINING}");
        }
    }
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
    let rungs = emit_config::active().decode_rungs();
    let buckets = prefill_buckets(&c, ctx);
    assert!(
        rungs.iter().all(|&r| buckets.iter().all(|&b| b > r)),
        "muse_spark: decode rungs {rungs:?} overlap the prefill buckets {buckets:?}"
    );
    let _guard = EmitAmdGuard::set(true);
    let mut m = model(&c, ctx, n_cu, tp, layers.clone(), block.is_none(), &buckets, &rungs);
    if !gpu.is_empty() {
        m.target = packet::devbuild::gpu_fingerprint(gpu);
    }
    validate_coverage(
        dir,
        PREFIX,
        &m.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        block.map(|_| layers.clone()),
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
    let section = match block {
        Some(_) => block_section(&c, dir, out, layers.clone(), &buckets, &rungs),
        None => serve_section::section(dir, emit_config::active().serve_defaults.as_deref(), &[])
            .unwrap_or_else(|e| panic!("muse_spark serve manifest: {e}")),
    };
    let mut sections = vec![section];
    if tp > 1 && emit_config::active().muse_latent_tp {
        sections.push(latent_shards_section(&m));
    }
    let outp = Path::new(out);
    std::fs::write(outp, m.to_blob_v6(&sections)).expect("write muse_spark blob");
    if !arch.is_empty() {
        write_lean_receipts(outp, &lean);
        let mut man = manifest::build(&m, arch, &lean);
        if emit_config::active().fa_gf_64.is_some() {
            man["tuning"]["gf64"] = serde_json::json!(fa_gf(c.heads, c.kv_heads));
            man["pairing"]["hash"] =
                serde_json::json!(format!("0x{:016x}", manifest::pairing_hash(&man)));
        }
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
        "muse_spark {layers:?}: prefill {buckets:?} + decode {rungs:?} ({} packets), ctx={ctx} \
         -> {out}\n  Remaining: {REMAINING}",
        m.progs.iter().map(|p| p.insts.len()).sum::<usize>(),
    );
}

/// The row-parallel latent projections, by name: `shard_of` does not classify them.
fn latent_shards_section(m: &Model) -> packet::devbuild::SectionData {
    use plow_asset::weight_shards::{WeightShards, SECTION, VERSION};
    let row = m
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .filter(|n| n.ends_with("mlp.pre_expert_proj.weight") || n.ends_with("mlp.post_expert_proj.weight"))
        .collect();
    packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: SECTION.into(),
        data: serde_json::to_vec(&WeightShards { version: VERSION, row, column: Vec::new() })
            .expect("weight shards json"),
    }
}

fn block_section(
    c: &MuseCfg,
    dir: &Path,
    out: &str,
    layers: std::ops::Range<usize>,
    buckets: &[u32],
    rungs: &[u32],
) -> packet::devbuild::SectionData {
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
            prefill_buckets: buckets.iter().map(|&b| b as i64).collect(),
            decode_t: *rungs.last().expect("a decode rung") as i64,
        },
    };
    write_block_descriptor(out, &desc)
}

/// Query heads per flash-decode work item (`PLOW_FA_GF_64`; the packet header carries a set value
/// to the object). 2 by default: on MI350X TP1 B1 the full GQA group (8) is faster only on full
/// layers at long context (c32768 FlashDecode 144 -> 97 us) and slower on sliding layers and at
/// c4096 (20 -> 41 us), where flash decode is not bandwidth-bound.
fn fa_gf(heads: u32, kv_heads: u32) -> u32 {
    let gqa = heads / kv_heads;
    let gf = emit_config::active().fa_gf_64.unwrap_or(2);
    assert!(
        matches!(gf, 1 | 2 | 4 | 8) && gqa % gf == 0,
        "PLOW_FA_GF_64={gf} must be 1|2|4|8 and divide GQA {gqa}"
    );
    gf
}

/// Flash-decode split count for `rows` slots of `heads` / `kv_heads` (the rank-local counts
/// under TP).
fn decode_ns(rows: u32, heads: u32, kv_heads: u32, n_cu: u32) -> u32 {
    emit_config::active().muse_decode_ns.unwrap_or_else(|| {
        let groups = rows * heads / fa_gf(heads, kv_heads);
        attention_decode_ns(rows, heads, kv_heads, n_cu, n_cu.div_ceil(groups).max(1))
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
        assert_eq!((fa_gf(8, 2), decode_ns(1, 8, 2, 256), decode_ns(8, 8, 2, 256)), (2, 64, 8));
        assert!(c.check_m1_layer(0).is_ok() && c.check_m1_layer(1).is_ok());
    }

    fn ops(m: &Model, op: DevOp) -> Vec<&packet::dev::DevInst> {
        m.progs[0].insts.iter().filter(|i| i.op == op as u16).collect()
    }

    #[test]
    fn full_nope_and_sliding_rope_layers_route_attention() {
        let c = MuseCfg::parse(&config());
        let full = decode_program(&c, 512, 256, 1, 1..2, false);
        let fa = ops(&full, DevOp::FlashDecode);
        assert_eq!(fa.len(), 1);
        assert_eq!(fa[0].i[4], 0, "full layer: no window");
        assert!((fa[0].f[0] - c.qk_scale / 8.0).abs() < 1e-6, "q scale folded");
        let hnr = ops(&full, DevOp::QwenHeadNormRope);
        assert_eq!(hnr.len(), 4, "q, k, v cache writes and the output norm");
        assert!(hnr.iter().all(|i| i.i[2] == 0 && i.t[2] == TENSOR_NONE), "NoPE, weightless");

        let slide = decode_program(&c, 512, 256, 1, 2..3, false);
        assert_eq!(ops(&slide, DevOp::FlashDecode)[0].i[4], 128);
        let hnr = ops(&slide, DevOp::QwenHeadNormRope);
        assert_eq!(hnr.iter().filter(|i| i.i[2] == 64).count(), 2, "q and k rotated");
    }

    #[test]
    fn latent_moe_routes_scaled_f32_logits_and_normed_combine() {
        let c = MuseCfg::parse(&config());
        let m = decode_program(&c, 512, 256, 1, 1..2, false);
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
        let m = decode_program(&c, 512, 256, 1, 1..3, false);
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
        assert_eq!(m1_layers(&c, None, 1, "gfx950", ""), 0..4);
        let gpu = std::panic::catch_unwind(|| m1_layers(&c, None, 1, "", "H100"));
        let msg = *gpu.unwrap_err().downcast::<String>().unwrap();
        assert!(msg.contains("gfx950"), "{msg}");
        assert_eq!(m1_layers(&c, Some("0"), 1, "gfx950", ""), 0..1);
        assert!(std::panic::catch_unwind(|| m1_layers(&c, Some("1"), 8, "gfx950", "")).is_err());
        assert_eq!(m1_layers(&c, Some("1..3"), 2, "gfx950", ""), 1..3);
    }

    #[test]
    fn tp_shards_heads_and_experts_and_reduces_before_the_post_expert_norm() {
        let c = MuseCfg::parse(&config());
        let m = decode_program(&c, 512, 256, 2, 0..2, false);
        let size = |n: &str| m.tensors.iter().find(|t| t.name == n).unwrap().bytes;
        let p = "model.language_model.layers.1.";
        assert_eq!(size(&format!("{p}self_attn.q_proj.weight")), 256 * 256 * 2);
        assert_eq!(size(&format!("{p}self_attn.k_proj.weight")), 64 * 256 * 2);
        assert_eq!(size(&format!("{p}self_attn.o_proj.weight")), 256 * 256 * 2);
        assert_eq!(size(&format!("{p}mlp.pre_expert_proj.weight")), 128 * 128 * 2);
        assert_eq!(size(&format!("{p}mlp.gate.weight")), 16 * 256 * 4, "router replicated");
        let fa = ops(&m, DevOp::FlashDecode);
        assert_eq!((fa[0].i[1], fa[0].i[2]), (4, 1), "4 q heads / 1 kv head per rank");
        let xr: Vec<u32> = ops(&m, DevOp::XReduce).iter().map(|i| i.i[0]).collect();
        assert_eq!(xr, vec![256, 128, 4 * 128, 256, 256, 128, 4 * 128, 256]);
        let slots: Vec<u32> = ops(&m, DevOp::XReduce).iter().map(|i| i.i[2]).collect();
        let sb = 4 * 128 * 2;
        assert_eq!(slots, vec![0, sb, 0, sb, 0, sb, 0, sb], "slots strictly alternate");
        assert!(ops(&m, DevOp::MuseMoeCombineNorm).iter().all(|i| i.i[2] == 1));
        assert!(ops(&m, DevOp::MoeGroupDownFp8Blk).iter().all(|i| i.i[7] == 1));
        let downs = ops(&m, DevOp::MoeGroupDownFp8Blk);
        assert_eq!(downs.iter().map(|i| i.i[6]).collect::<Vec<_>>(), vec![MOE_ENC_BF16, MOE_ENC_MXFP4]);
        assert!(downs.iter().all(|i| i.i[2] == 64));
    }

    #[test]
    fn full_model_embeds_ids_and_ends_in_a_softcapped_greedy_id() {
        let c = MuseCfg::parse(&config());
        let m = decode_program(&c, 512, 256, 2, 0..4, true);
        let size = |n: &str| m.tensors.iter().find(|t| t.name == n).unwrap().bytes;
        let ins = &m.progs[0].insts;
        let id = |n: &str| m.tensors.iter().position(|t| t.name == n).unwrap() as u32;
        assert_eq!(ins[0].op, DevOp::Embed as u16);
        assert_eq!((ins[0].t[2], ins[0].f[0]), (id("in.ids"), 1.0));
        assert_eq!((ins[1].op, ins[1].t[2], ins[1].i[2]), (DevOp::MuseRmsNorm as u16, TENSOR_NONE, NORM_PLAIN));
        let n = ins.len();
        let ops: Vec<u16> = ins[n - 5..].iter().map(|i| i.op).collect();
        let want = [DevOp::MuseRmsNorm, DevOp::Gemv, DevOp::SoftCap, DevOp::Argmax, DevOp::ArgmaxFin];
        assert_eq!(ops, want.map(|o| o as u16));
        assert_eq!(ins[n - 5].f[1], c.output_multiplier, "multiplier rides the final norm");
        assert_eq!((ins[n - 3].i[0], ins[n - 3].f[0]), (1000, 20.0));
        assert_eq!(ins[n - 1].t[0], id("in.ids"));
        assert_eq!(size("lm_head.weight"), 1000 * 256 * 2, "head replicated at TP");
    }

    fn prog_ops(m: &Model, p: usize, op: DevOp) -> Vec<&packet::dev::DevInst> {
        m.progs[p].insts.iter().filter(|i| i.op == op as u16).collect()
    }

    #[test]
    fn prefill_bucket_routes_flash_prefill_and_the_grouped_moe_chain() {
        let c = MuseCfg::parse(&config());
        let m = model(&c, 512, 256, 1, 1..3, false, &[128], &[1]);
        assert_eq!(m.prog_t, vec![128, 1]);
        let fa = prog_ops(&m, 0, DevOp::FlashPrefill);
        assert_eq!(fa.len(), 2);
        assert_eq!((fa[0].i[0], fa[0].i[1], fa[0].i[5], fa[0].j[0]), (128, 128, 0, 512));
        assert_eq!(fa[1].i[5], 128, "sliding layer windowed");
        assert!(prog_ops(&m, 0, DevOp::QwenHeadNormRope).iter().all(|i| i.i[3] == 128));
        let kv = prog_ops(&m, 0, DevOp::QwenHeadNormRope);
        assert_eq!(kv.iter().filter(|i| i.i[6] == 1).count(), 4, "k/v cache writes per layer");
        let rt = prog_ops(&m, 0, DevOp::MoeRouterTopkPf);
        assert_eq!((rt[0].i[4], rt[0].i[3]), (128, ROUTER_FLAGS));
        assert!((rt[0].f[1] - c.output_multiplier).abs() < 1e-7);
        assert_eq!(prog_ops(&m, 0, DevOp::MoeAlignPf)[0].i[..3], [128, 16, 4]);
        let glu = prog_ops(&m, 0, DevOp::MoeGroupGluPf);
        assert_eq!((glu[0].i[3], glu[0].t[7] != TENSOR_NONE), (MOE_ENC_MXFP4, true));
        let dn = prog_ops(&m, 0, DevOp::MoeGroupDownPf);
        assert_eq!((dn[0].i[0], dn[0].i[1], dn[0].i[7]), (128, 128, 0));
        assert!(prog_ops(&m, 0, DevOp::MuseMoeCombineNorm).iter().all(|i| i.i[3] == 128));
        assert!(prog_ops(&m, 0, DevOp::Gemv).is_empty(), "prefill projections are tiled GEMMs");
        let size = |n: &str| m.tensors.iter().find(|t| t.name == n).unwrap().bytes;
        assert_eq!(size("act.x"), 128 * 256 * 2, "one table at the widest program");
        assert_eq!(size("kv.1.k"), 64 * 2 * 512 * 2);
    }

    #[test]
    fn tp_declares_the_latent_projections_row_parallel() {
        let c = MuseCfg::parse(&config());
        let m = model(&c, 512, 256, 2, 0..2, true, &[], &[1]);
        let sec = latent_shards_section(&m);
        let d = plow_asset::weight_shards::WeightShards::parse(&sec.data).unwrap();
        let mut row = d.row.clone();
        row.sort();
        let want: Vec<String> = (0..2)
            .flat_map(|l| {
                ["post", "pre"].map(|w| format!("{PREFIX}layers.{l}.mlp.{w}_expert_proj.weight"))
            })
            .collect();
        assert_eq!(row, want);
        assert!(d.column.is_empty());
    }

    #[test]
    fn tp_prefill_reduces_bf16_weighted_partials_two_shot() {
        let c = MuseCfg::parse(&config());
        let m = model(&c, 512, 256, 2, 0..2, true, &[128], &[1]);
        assert_eq!(prog_ops(&m, 0, DevOp::XReduce).len(), 0);
        let xr = prog_ops(&m, 0, DevOp::XReduceTwoShot);
        assert!(!xr.is_empty());
        assert!(prog_ops(&m, 0, DevOp::MoeGroupDownPf).iter().all(|i| i.i[7] == 1));
        assert!(prog_ops(&m, 0, DevOp::MuseMoeCombineNorm).iter().all(|i| i.i[2] == 2));
        let names: Vec<&str> = m.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"act.xn2_bm@bandr128") && names.contains(&"act.xn2@band1"));
        let norm = prog_ops(&m, 0, DevOp::MuseRmsNorm);
        assert_eq!(norm.iter().filter(|i| i.i[3] == 128).count(), 2, "band-major pre-MoE norms");
        let lm = prog_ops(&m, 0, DevOp::Gemv);
        assert_eq!((lm.len(), lm[0].i[0], lm[0].i[4]), (1, 1, 127), "last row scored");
        let slot = m.tensors.iter().find(|t| t.name == "act.og_tp").unwrap().bytes;
        assert_eq!(slot, 128 * 4 * 128 * 2);
    }

    #[test]
    fn batched_decode_rung_reads_each_slot_row() {
        let c = MuseCfg::parse(&config());
        let m = model(&c, 512, 256, 2, 0..2, true, &[128], &[1, 8]);
        assert_eq!(m.prog_t, vec![128, 1, 8]);
        let fa = prog_ops(&m, 2, DevOp::FlashDecode);
        assert_eq!(fa[0].i[0], 8);
        let rt = prog_ops(&m, 2, DevOp::MoeRouterTopk);
        assert_eq!((rt[0].i[4], rt[0].f[1]), (8, c.output_multiplier));
        let glu = prog_ops(&m, 2, DevOp::MoeGroupGluFp8Blk);
        assert_eq!((glu[0].i[0], glu[0].i[4]), (32, 4));
        assert!(prog_ops(&m, 2, DevOp::MoeGroupDownFp8Blk).iter().all(|i| i.i[0] == 32));
        let cmb = prog_ops(&m, 2, DevOp::MuseMoeCombineNorm);
        assert!(cmb.iter().all(|i| i.i[3] == 8 && i.i[4] == 64 && i.i[2] == 1));
        assert!(prog_ops(&m, 2, DevOp::Gemv).is_empty(), "rows > 1 run GemmSmall");
        assert_eq!(prog_ops(&m, 2, DevOp::Argmax)[0].i[1], 8);
        let size = |n: &str| m.tensors.iter().find(|t| t.name == n).unwrap().bytes;
        assert_eq!(size("in.kvlen"), 8 * 4);
        assert_eq!(size("kv.0.k"), 8 * 64 * 512 * 2, "one cache per slot");
        assert!(prog_ops(&m, 1, DevOp::Gemv).len() > 1, "the B1 rung keeps the GEMV");
    }
}
