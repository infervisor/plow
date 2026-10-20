//! Gemma-4 assistant (MTP) draft programs for speculative decoding (`PLOW_EMIT_SPEC`).
//!
//! One program per `(rows, k)`: `k` chained draft steps for `rows` sequences on the decode
//! interpreter. Step `j` embeds `tokens[j]` with the target table, projects it together with
//! the carried hidden (the verify's final-norm row at step 0, then the previous step's
//! `post_projection`), runs the four Q-only layers against the target's last sliding and last
//! full KV caches at the committed length, and writes the greedy draft to `tokens[j+1]`.
//! Weights bind from `model-mtp.safetensors` (`scripts/gemma4_mtp_rekey.py`) beside the
//! target shards. The draft never writes KV.
use std::path::Path;

use packet::dev::{DevInst, DevOp, TENSOR_NONE};
use packet::devbuild::{Builder, Program, SectionData, SECT_METADATA};
use plow_asset::spec_draft;
use serde_json::Value;

use crate::config::Cfg;
use crate::Tn;

pub(crate) const KMAX: u32 = 7;
const ROWS: [u32; 5] = [1, 2, 4, 8, 16];
const SHARD: &str = "model-mtp.safetensors";
const AMAX_BLOCKS: u32 = 64;
const BF16: u64 = 2;
const I32: u64 = 4;

pub(crate) struct DraftCfg {
    hidden: u32,
    inter: u32,
    full: Vec<bool>,
    eps: f32,
    layer_scalar: Vec<f32>,
}

fn refuse(what: String) -> ! {
    panic!("PLOW_EMIT_SPEC: {what}")
}

fn sha256_file(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_else(|e| refuse(format!("{}: {e}", path.display())));
    plow_asset::knob::sha256_hex(&bytes)
}

fn draft_dir() -> std::path::PathBuf {
    crate::emit_config::active()
        .spec_draft_dir
        .clone()
        .unwrap_or_else(|| refuse("PLOW_SPEC_DRAFT_DIR is unset".into()))
        .into()
}

fn safetensors_header(path: &Path) -> serde_json::Map<String, Value> {
    use std::io::Read;
    let mut f =
        std::fs::File::open(path).unwrap_or_else(|e| refuse(format!("{}: {e}", path.display())));
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .unwrap_or_else(|e| refuse(format!("{}: {e}", path.display())));
    let mut h = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut h)
        .unwrap_or_else(|e| refuse(format!("{}: {e}", path.display())));
    let mut v: serde_json::Map<String, Value> = serde_json::from_slice(&h)
        .unwrap_or_else(|e| refuse(format!("{}: bad header: {e}", path.display())));
    v.remove("__metadata__");
    v
}

impl DraftCfg {
    /// The assistant config, refused by name wherever it differs from what the draft program
    /// and the target layers it reads assume.
    pub(crate) fn load(c: &Cfg) -> Self {
        let dir = draft_dir();
        let v: Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .unwrap_or_else(|e| refuse(format!("{}/config.json: {e}", dir.display()))),
        )
        .unwrap_or_else(|e| refuse(format!("config.json: {e}")));
        let tc = &v["text_config"];
        let u = |x: &Value, k: &str| -> u32 {
            x[k].as_u64()
                .unwrap_or_else(|| refuse(format!("config: {k} missing"))) as u32
        };
        let want = |ok: bool, what: &str| {
            if !ok {
                refuse(format!("draft config: {what}"))
            }
        };
        want(
            v["model_type"] == "gemma4_assistant" || v["model_type"] == "gemma4_unified_assistant",
            "model_type is neither gemma4_assistant nor gemma4_unified_assistant",
        );
        want(
            v["use_ordered_embeddings"] != true,
            "use_ordered_embeddings (centroid head) is unsupported",
        );
        want(
            u(&v, "backbone_hidden_size") == c.hidden,
            "backbone_hidden_size != target hidden",
        );
        want(u(tc, "vocab_size") == c.vocab, "vocab_size != target vocab");
        want(
            u(tc, "num_attention_heads") == c.heads,
            "num_attention_heads != target heads",
        );
        want(
            u(tc, "head_dim") == c.hd_slide,
            "head_dim != target sliding head_dim",
        );
        want(
            u(tc, "global_head_dim") == c.hd_full,
            "global_head_dim != target full head_dim",
        );
        want(
            u(tc, "num_key_value_heads") == c.kvh_slide,
            "num_key_value_heads != target sliding kv heads",
        );
        want(
            u(tc, "num_global_key_value_heads") == c.kvh_full,
            "num_global_key_value_heads != target full kv heads",
        );
        want(
            u(tc, "sliding_window") == c.window,
            "sliding_window != target window",
        );
        want(
            tc["hidden_activation"] == "gelu_pytorch_tanh" && c.mlp_act == 0,
            "hidden_activation is not gelu_pytorch_tanh",
        );
        want(
            tc["final_logit_softcapping"].is_null(),
            "final_logit_softcapping is unsupported",
        );
        want(
            tc["enable_moe_block"] != true && tc["use_double_wide_mlp"] != true,
            "MoE / double-wide MLP draft layers are unsupported",
        );
        want(
            u(tc, "hidden_size_per_layer_input") == 0,
            "per-layer inputs are unsupported",
        );
        want(
            v["tie_word_embeddings"] == true || tc["tie_word_embeddings"] == true,
            "untied draft head is unsupported",
        );
        want(c.attn_scale == 1.0, "target attention scale is not 1.0");
        let layers = u(tc, "num_hidden_layers");
        want(
            u(tc, "num_kv_shared_layers") == layers,
            "every draft layer must read target KV",
        );
        let full: Vec<bool> = tc["layer_types"]
            .as_array()
            .unwrap_or_else(|| refuse("config: layer_types missing".into()))
            .iter()
            .map(|t| match t.as_str() {
                Some("full_attention") => true,
                Some("sliding_attention") => false,
                _ => refuse(format!("draft config: layer type {t}")),
            })
            .collect();
        want(full.len() == layers as usize, "layer_types length");
        let rp = &tc["rope_parameters"];
        want(
            rp["sliding_attention"]["rope_theta"].as_f64() == Some(c.theta_slide)
                && rp["full_attention"]["rope_theta"].as_f64() == Some(c.theta_full)
                && rp["full_attention"]["partial_rotary_factor"].as_f64() == Some(c.rope_frac_full),
            "rope parameters differ from the target layers the draft reads",
        );
        want(
            rp["full_attention"]["rope_type"] == "proportional"
                && rp["sliding_attention"]["rope_type"] == "default"
                && rp["sliding_attention"]["partial_rotary_factor"].as_f64().unwrap_or(1.0) == 1.0
                && ["full_attention", "sliding_attention"]
                    .iter()
                    .all(|t| rp[*t]["factor"].as_f64().unwrap_or(1.0) == 1.0),
            "rope types/scaling differ from the target's (proportional full, default sliding, unscaled)",
        );
        let reader = crate::checkpoint::TensorReader::open(&dir)
            .unwrap_or_else(|e| refuse(format!("draft checkpoint: {e}")));
        let layer_scalar = (0..layers)
            .map(|l| {
                let name = format!("model.layers.{l}.layer_scalar");
                let (dtype, bytes) = reader.read(&name).unwrap_or_else(|e| refuse(e));
                if dtype != "BF16" || bytes.len() != 2 {
                    refuse(format!("{name}: {dtype} {} bytes", bytes.len()));
                }
                f32::from_bits(u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16)
            })
            .collect();
        DraftCfg {
            hidden: u(tc, "hidden_size"),
            inter: u(tc, "intermediate_size"),
            full,
            eps: tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            layer_scalar,
        }
    }

    fn hd(&self, c: &Cfg, l: usize) -> u32 {
        if self.full[l] {
            c.hd_full
        } else {
            c.hd_slide
        }
    }

    /// `(name, shape)` of every tensor the rekeyed shard must hold.
    fn expected(&self, c: &Cfg) -> Vec<(String, Vec<u64>)> {
        let (h, i, v, bh) = (
            self.hidden as u64,
            self.inter as u64,
            c.vocab as u64,
            c.hidden as u64,
        );
        let mut e = vec![
            ("mtp.model.embed_tokens.weight".into(), vec![v, h]),
            ("mtp.model.norm.weight".into(), vec![h]),
            ("mtp.pre_projection_e.weight".into(), vec![h, bh]),
            ("mtp.pre_projection_h.weight".into(), vec![h, bh]),
            ("mtp.post_projection.weight".into(), vec![bh, h]),
        ];
        for l in 0..self.full.len() {
            let qd = (c.heads * self.hd(c, l)) as u64;
            let p = format!("mtp.model.layers.{l}.");
            for (s, shape) in [
                ("input_layernorm.weight", vec![h]),
                ("layer_scalar", vec![1]),
                ("mlp.down_proj.weight", vec![h, i]),
                ("mlp.gate_proj.weight", vec![i, h]),
                ("mlp.up_proj.weight", vec![i, h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("post_feedforward_layernorm.weight", vec![h]),
                ("pre_feedforward_layernorm.weight", vec![h]),
                ("self_attn.o_proj.weight", vec![h, qd]),
                ("self_attn.q_norm.weight", vec![self.hd(c, l) as u64]),
                ("self_attn.q_proj.weight", vec![qd, h]),
            ] {
                e.push((format!("{p}{s}"), shape));
            }
        }
        e
    }
}

struct LayerW {
    in_ln: u32,
    q: u32,
    q_norm: u32,
    o: u32,
    post_attn: u32,
    pre_ffn: u32,
    gate: u32,
    up: u32,
    down: u32,
    post_ffn: u32,
}

pub(crate) struct Tensors {
    cfg: DraftCfg,
    rows: u32,
    tokens: Vec<u32>,
    hidden_row: u32,
    position: u32,
    kv_len: u32,
    slot: u32,
    e: u32,
    h: u32,
    x1: u32,
    x2: u32,
    x: u32,
    xn: u32,
    q: u32,
    qn: u32,
    at: u32,
    og: u32,
    hn: u32,
    fu: u32,
    dg: u32,
    logits: u32,
    amax: u32,
    emb: u32,
    norm: u32,
    pre_e: u32,
    pre_h: u32,
    post: u32,
    layers: Vec<LayerW>,
}

pub(crate) fn declare(b: &mut Builder, c: &Cfg, dbatch: u32, span_verify: bool) -> Tensors {
    if !span_verify {
        refuse(
            "the draft reads the target KV through the span slot map: set PLOW_EMIT_SPAN_VERIFY"
                .into(),
        );
    }
    let cfg = DraftCfg::load(c);
    let rows = ROWS
        .iter()
        .copied()
        .filter(|&r| r <= dbatch)
        .max()
        .unwrap_or(1);
    let (r, h, bh, v) = (
        rows as u64,
        cfg.hidden as u64,
        c.hidden as u64,
        c.vocab as u64,
    );
    let qd = (c.heads * c.hd_slide.max(c.hd_full)) as u64;
    let w = |b: &mut Builder, s: &str, sz: u64| b.tensor(&format!("mtp.{s}"), sz * BF16);
    let tokens = (0..=KMAX)
        .map(|j| b.tensor(&format!("in.spec.tokens.{j}"), r * I32))
        .collect();
    let a = |b: &mut Builder, s: &str, sz: u64| b.tensor(&format!("act.mtp.{s}"), sz);
    let t = Tensors {
        rows,
        tokens,
        hidden_row: b.tensor("in.spec.hidden_row", r * I32),
        position: b.tensor("in.spec.position", r * I32),
        kv_len: b.tensor("in.spec.kv_len", r * I32),
        slot: b.tensor("in.spec.slot", r * I32),
        e: a(b, "e", r * bh * BF16),
        h: a(b, "h", r * bh * BF16),
        x1: a(b, "x1", r * h * BF16),
        x2: a(b, "x2", r * h * BF16),
        x: a(b, "x", r * h * BF16),
        xn: a(b, "xn", r * h * BF16),
        q: a(b, "q", r * qd * BF16),
        qn: a(b, "qn", r * qd * BF16),
        at: a(b, "at", r * qd * BF16),
        og: a(b, "og", r * h * BF16),
        hn: a(b, "hn", r * h * BF16),
        fu: a(b, "fu", r * cfg.inter as u64 * BF16),
        dg: a(b, "dg", r * h * BF16),
        logits: a(b, "logits", r * v * BF16),
        amax: a(b, "amax", r * AMAX_BLOCKS as u64 * 8),
        emb: w(b, "model.embed_tokens.weight", v * h),
        norm: w(b, "model.norm.weight", h),
        pre_e: w(b, "pre_projection_e.weight", h * bh),
        pre_h: w(b, "pre_projection_h.weight", h * bh),
        post: w(b, "post_projection.weight", bh * h),
        layers: (0..cfg.full.len())
            .map(|l| {
                let lq = (c.heads * cfg.hd(c, l)) as u64;
                let i = cfg.inter as u64;
                let p = format!("model.layers.{l}.");
                LayerW {
                    in_ln: w(b, &format!("{p}input_layernorm.weight"), h),
                    q: w(b, &format!("{p}self_attn.q_proj.weight"), lq * h),
                    q_norm: w(
                        b,
                        &format!("{p}self_attn.q_norm.weight"),
                        cfg.hd(c, l) as u64,
                    ),
                    o: w(b, &format!("{p}self_attn.o_proj.weight"), h * lq),
                    post_attn: w(b, &format!("{p}post_attention_layernorm.weight"), h),
                    pre_ffn: w(b, &format!("{p}pre_feedforward_layernorm.weight"), h),
                    gate: w(b, &format!("{p}mlp.gate_proj.weight"), i * h),
                    up: w(b, &format!("{p}mlp.up_proj.weight"), i * h),
                    down: w(b, &format!("{p}mlp.down_proj.weight"), h * i),
                    post_ffn: w(b, &format!("{p}post_feedforward_layernorm.weight"), h),
                }
            })
            .collect(),
        cfg,
    };
    t
}

/// The target decode rung's instructions the draft copies its geometry from.
struct Sites {
    embed: DevInst,
    /// Per draft layer: (q HeadNormRope, FlashDecodeFp8, FlashMerge) of the target layer it reads.
    attn: Vec<(DevInst, DevInst, DevInst)>,
}

fn sites(c: &Cfg, n: &Tn, d: &DraftCfg, target: &Program) -> Sites {
    let insts = &target.insts;
    let embed = insts
        .iter()
        .find(|i| i.op == DevOp::Embed as u16)
        .unwrap_or_else(|| refuse("target rung has no Embed".into()))
        .clone();
    let attn = (0..d.full.len())
        .map(|l| {
            let src = (0..c.layers as usize)
                .rev()
                .find(|&s| c.is_full[s] == d.full[l] && !c.kv_is_shared(s))
                .unwrap_or_else(|| refuse(format!("no target layer for draft layer {l}")));
            let fa = insts
                .iter()
                .position(|i| i.op == DevOp::FlashDecodeFp8 as u16 && i.t[3] == n.kc[src])
                .unwrap_or_else(|| {
                    refuse(format!("target layer {src} has no FP8-KV flash decode"))
                });
            let hnr = insts[..fa]
                .iter()
                .rev()
                .find(|i| i.op == DevOp::HeadNormRope as u16 && i.t[0] == insts[fa].t[2])
                .unwrap_or_else(|| refuse(format!("target layer {src}: no q HeadNormRope")));
            let merge = insts[fa..]
                .iter()
                .find(|i| i.op == DevOp::FlashMerge as u16 && i.t[1] == insts[fa].t[0])
                .unwrap_or_else(|| refuse(format!("target layer {src}: no FlashMerge")));
            let f = &insts[fa];
            if f.i[1] != c.heads
                || f.i[6] != d.hd(c, l)
                || f.t[4] == TENSOR_NONE
                || f.t[6] == TENSOR_NONE
                || f.t[7] == TENSOR_NONE
            {
                refuse(format!("target layer {src}: flash geometry"));
            }
            (hnr.clone(), f.clone(), merge.clone())
        })
        .collect();
    Sites { embed, attn }
}

/// The draft's token ids must mean what the target's do: identical vocab, and every draft added
/// token present in the target under the same id (the target may add more, e.g. media tokens).
fn check_tokenizer(draft: &Path, target: &Path) {
    let read = |p: &Path| -> Value {
        serde_json::from_slice(
            &std::fs::read(p).unwrap_or_else(|e| refuse(format!("{}: {e}", p.display()))),
        )
        .unwrap_or_else(|e| refuse(format!("{}: {e}", p.display())))
    };
    let (d, t) = (read(draft), read(target));
    if d["model"] != t["model"] {
        refuse("draft tokenizer vocab differs from the target's".into());
    }
    let added = t["added_tokens"].as_array().cloned().unwrap_or_default();
    for tok in d["added_tokens"].as_array().into_iter().flatten() {
        if !added.contains(tok) {
            refuse(format!("draft added token {tok} is not the target's"));
        }
    }
}

/// Operands only: `emit` owns the opcode, slice count and dependency fields.
fn args(i: &mut DevInst, src: &DevInst) {
    i.t = src.t;
    i.i = src.i;
    i.f = src.f;
    i.j = src.j;
}

#[allow(clippy::too_many_arguments)]
fn program(
    c: &Cfg,
    n: &Tn,
    t: &Tensors,
    tensors: &[packet::devbuild::TensorDecl],
    n_cu: u32,
    s: &Sites,
    rows: u32,
    k: u32,
) -> Program {
    let d = &t.cfg;
    let mut b = Builder::new(n_cu);
    b.adopt_tensors(tensors.to_vec());
    b.force_uniseg();
    let all: Vec<u32> = (0..n_cu).collect();
    let rset: Vec<u32> = (0..rows).collect();
    let span = |n: u32| -> Vec<u32> { (0..n.max(1).min(n_cu)).collect() };
    let nb = if rows > 1 { rows } else { 0 };
    let gemv = |b: &mut Builder, dep: &[u32], out: u32, x: u32, w: u32, nout: u32, kin: u32| {
        b.emit(DevOp::Gemv, all.clone(), dep, |i| {
            i.t[0] = out;
            i.t[1] = x;
            i.t[2] = w;
            i.i[0] = rows;
            i.i[1] = nout;
            i.i[2] = kin;
        })
    };
    let mut prev: Vec<u32> = Vec::new();
    let mut hdep: Vec<u32> = Vec::new();
    for j in 0..k as usize {
        let c_e = b.emit(DevOp::Embed, rset.clone(), &prev, |i| {
            i.t[0] = t.e;
            i.t[1] = n.emb;
            i.t[2] = t.tokens[j];
            i.i[0] = rows;
            i.i[1] = c.hidden;
            i.i[2] = s.embed.i[2];
            i.f[0] = s.embed.f[0];
        });
        if j == 0 {
            hdep = vec![b.emit(DevOp::RowGather, rset.clone(), &[], |i| {
                i.t[0] = t.h;
                i.t[1] = n.hn;
                i.t[2] = t.hidden_row;
                i.i[0] = rows;
                i.i[1] = c.hidden;
                i.i[2] = n.rowslot_rows;
            })];
        }
        let c_x1 = gemv(&mut b, &[c_e], t.x1, t.e, t.pre_e, d.hidden, c.hidden);
        let c_x2 = gemv(&mut b, &hdep, t.x2, t.h, t.pre_h, d.hidden, c.hidden);
        let c_x = b.emit(
            DevOp::Residual,
            span((rows * d.hidden).div_ceil(512 * 8)),
            &[c_x1, c_x2],
            |i| {
                i.t[0] = t.x;
                i.t[1] = t.x1;
                i.t[2] = t.x2;
                i.i[0] = rows * d.hidden;
                i.f[0] = 1.0;
            },
        );
        let mut dep = b.emit(DevOp::RmsNorm, rset.clone(), &[c_x], |i| {
            i.t[0] = t.xn;
            i.t[1] = t.x;
            i.t[2] = t.layers[0].in_ln;
            i.i[0] = rows;
            i.i[1] = d.hidden;
            i.f[0] = d.eps;
        });
        for (l, w) in t.layers.iter().enumerate() {
            let (hnr, fa, mg) = &s.attn[l];
            let qd = c.heads * d.hd(c, l);
            let c_q = gemv(&mut b, &[dep], t.q, t.xn, w.q, qd, d.hidden);
            let c_qn = b.emit(DevOp::HeadNormRope, span(hnr.blocks as u32), &[c_q], |i| {
                args(i, hnr);
                i.t[0] = t.qn;
                i.t[1] = t.q;
                i.t[2] = w.q_norm;
                i.t[5] = t.position;
                i.i[0] = rows;
                i.f[0] = d.eps;
            });
            let c_fa = b.emit(
                DevOp::FlashDecodeFp8,
                span(fa.blocks as u32),
                &[c_qn],
                |i| {
                    args(i, fa);
                    i.t[2] = t.qn;
                    i.t[5] = t.kv_len;
                    i.i[0] = rows;
                    i.j[1] = t.slot;
                },
            );
            let c_mg = b.emit(DevOp::FlashMerge, span(mg.blocks as u32), &[c_fa], |i| {
                args(i, mg);
                i.t[0] = t.at;
                i.i[0] = rows;
            });
            let c_o = gemv(&mut b, &[c_mg], t.og, t.at, w.o, d.hidden, qd);
            let c_r1 = b.emit(DevOp::NormResidualNorm, rset.clone(), &[c_o], |i| {
                i.t[0] = t.hn;
                i.t[1] = t.x;
                i.t[2] = t.x;
                i.t[3] = t.og;
                i.t[4] = w.post_attn;
                i.t[5] = w.pre_ffn;
                i.i[0] = rows;
                i.i[1] = d.hidden;
                i.f[0] = d.eps;
                i.f[1] = 1.0;
            });
            let c_gu = b.emit(DevOp::GemvGlu, all.clone(), &[c_r1], |i| {
                i.t[0] = t.fu;
                i.t[1] = t.hn;
                i.t[2] = w.gate;
                i.t[5] = w.up;
                i.i[0] = rows;
                i.i[1] = d.inter;
                i.i[2] = d.hidden;
                i.i[5] = c.mlp_act;
            });
            let c_d = gemv(&mut b, &[c_gu], t.dg, t.fu, w.down, d.hidden, d.inter);
            let next = t.layers.get(l + 1).map_or(t.norm, |n| n.in_ln);
            dep = b.emit(DevOp::NormResidualNorm, rset.clone(), &[c_d], |i| {
                i.t[0] = t.xn;
                i.t[1] = t.x;
                i.t[2] = t.x;
                i.t[3] = t.dg;
                i.t[4] = w.post_ffn;
                i.t[5] = next;
                i.i[0] = rows;
                i.i[1] = d.hidden;
                i.f[0] = d.eps;
                i.f[1] = d.layer_scalar[l];
            });
        }
        let c_lm = gemv(&mut b, &[dep], t.logits, t.xn, t.emb, c.vocab, d.hidden);
        let c_am = b.emit(DevOp::Argmax, (0..AMAX_BLOCKS).collect(), &[c_lm], |i| {
            i.t[0] = t.amax;
            i.t[1] = t.logits;
            i.i[0] = c.vocab;
            i.i[1] = nb;
        });
        prev = vec![b.emit(DevOp::ArgmaxFin, vec![0], &[c_am], |i| {
            i.t[0] = t.tokens[j + 1];
            i.t[1] = t.amax;
            i.i[0] = AMAX_BLOCKS;
            i.i[1] = nb;
        })];
        if j + 1 < k as usize {
            hdep = vec![gemv(&mut b, &[dep], t.h, t.xn, t.post, c.hidden, d.hidden)];
        }
    }
    b.finish()
}

/// The draft programs for every `(rows, k)` the target ladder has a rung for, plus `spec.json`.
pub(crate) fn sections(
    dir: &Path,
    c: &Cfg,
    n: &Tn,
    t: &Tensors,
    model: &packet::devbuild::Model,
    rung_progs: &[(u32, usize)],
) -> Vec<SectionData> {
    check_tokenizer(
        &draft_dir().join("tokenizer.json"),
        &dir.join("tokenizer.json"),
    );
    let shard = dir.join(SHARD);
    let header = safetensors_header(&shard);
    let expected = t.cfg.expected(c);
    for (name, shape) in &expected {
        let got = header
            .get(name)
            .unwrap_or_else(|| refuse(format!("{SHARD} lacks {name}")));
        let dims: Vec<u64> = got["shape"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_u64).collect())
            .unwrap_or_default();
        if &dims != shape || got["dtype"] != "BF16" {
            refuse(format!(
                "{SHARD} {name}: {:?} {} (expected {shape:?} BF16)",
                dims, got["dtype"]
            ));
        }
    }
    if header.len() != expected.len() {
        refuse(format!(
            "{SHARD} holds {} tensors, expected {}",
            header.len(),
            expected.len()
        ));
    }
    let mut programs = Vec::new();
    let mut table = Vec::new();
    for &rows in ROWS.iter().filter(|&&r| r <= t.rows) {
        let Some(&(_, pi)) = rung_progs.iter().find(|&&(r, _)| r == rows) else {
            continue;
        };
        let s = sites(c, n, &t.cfg, &model.progs[pi]);
        for k in 1..=KMAX {
            table.push(spec_draft::DraftProgram {
                rows,
                k,
                index: programs.len() as u32,
            });
            programs.push(program(c, n, t, &model.tensors, model.n_cu, &s, rows, k));
        }
    }
    if programs.is_empty() {
        refuse(format!(
            "no target decode rung of {ROWS:?} rows to copy attention geometry from"
        ));
    }
    let packed: Vec<Vec<_>> = programs
        .iter()
        .map(|p| p.insts.iter().map(|d| d.pack()).collect())
        .collect();
    let views: Vec<plow_asset::program::Program<'_>> = programs
        .iter()
        .zip(&packed)
        .zip(&table)
        .map(|((p, insts), e)| plow_asset::program::Program {
            rows: e.rows,
            role: packet::devbuild::ProgramRole::DecodeRung { rows: e.rows },
            n_counter: p.n_counter,
            insts,
            stream: &p.stream,
            stream_ofs: &p.stream_ofs,
            stream_len: &p.stream_len,
            waits: &p.waits,
            succs: &p.succs,
            gq_stream: &p.gq_stream,
            gq_seg_ofs: &p.gq_seg_ofs,
            l2_domains: p.l2_domains,
        })
        .collect();
    let bytes = plow_asset::aux_program::encode(model.n_cu, model.tensors.len(), &views)
        .unwrap_or_else(|e| refuse(format!("draft programs: {e}")));
    let h16 = |h: u32| u16::try_from(h).unwrap_or_else(|_| refuse(format!("handle {h}")));
    let manifest = spec_draft::Manifest {
        version: spec_draft::VERSION,
        family: "gemma4_assistant".into(),
        kmax: KMAX,
        programs: table,
        tensors: spec_draft::DraftTensors {
            tokens: t.tokens.iter().map(|&h| h16(h)).collect(),
            hidden_row: h16(t.hidden_row),
            position: h16(t.position),
            kv_len: h16(t.kv_len),
            slot: h16(t.slot),
            hidden: h16(n.hn),
            hidden_rows: n.rowslot_rows,
        },
        shard: SHARD.into(),
        shard_sha256: sha256_file(&shard),
    };
    manifest
        .validate(programs.len(), model.tensors.len())
        .unwrap_or_else(|e| refuse(e));
    eprintln!(
        "  spec draft: {} programs (rows {:?}, k 1..={KMAX})",
        programs.len(),
        manifest
            .programs
            .iter()
            .map(|p| p.rows)
            .collect::<std::collections::BTreeSet<_>>()
    );
    vec![
        SectionData {
            kind: SECT_METADATA,
            name: spec_draft::SECTION.into(),
            data: serde_json::to_vec(&manifest).expect("spec.json"),
        },
        SectionData {
            kind: SECT_METADATA,
            name: spec_draft::PROGRAMS_SECTION.into(),
            data: bytes,
        },
    ]
}
