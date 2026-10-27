//! DSpark block drafter, target side (`PLOW_EMIT_SPEC` with `PLOW_SPEC_KIND=gemma4_dspark`).
//!
//! The drafter keeps its own KV: five full-attention layers whose K/V for a committed row are
//! `k = k_proj_l(hidden_norm(sum_j fc_j(tap_j)))`, stored as `rope(k_norm(k))` / `v_norm(k)`,
//! where `tap_j` is the post-layer residual of target layer `taps[j]`. Every target program
//! copies those residuals as the layer loop passes them and ends with the context tail that
//! writes the drafter layers at each row's own position and slot, so the drafter context is
//! written in the same launch as the target KV and has no holes. The drafter layers are packet
//! KV layers `kv.{L+i}.*` (L = target layers) with a target full layer's geometry.
//! Weights bind from `model-dspark.safetensors` (`scripts/gemma4_dspark_rekey.py`) and, on an FP8
//! target, the twins in `model-dspark-fp8.safetensors` (`scripts/gemma4_dspark_fp8.py`).
use std::path::Path;

use packet::dev::{DevInst, DevOp};
use packet::devbuild::{Builder, Program, SectionData, SECT_METADATA};
use plow_asset::{spec_context, spec_draft};
use serde_json::Value;

use crate::config::{Arch, Cfg};
use crate::mtp::{draft_dir, kmax, refuse, safetensors_header, sha256_file};

const SHARD: &str = "model-dspark.safetensors";
const FP8_SHARD: &str = "model-dspark-fp8.safetensors";
const FAMILY: &str = "gemma4_dspark";
const BF16: u64 = 2;
const F32: u64 = 4;

/// Whether this emit carries the DSpark context tail.
pub(crate) fn enabled() -> bool {
    let e = crate::emit_config::active();
    e.emit_spec && e.spec_kind.as_deref() == Some(FAMILY)
}

pub(crate) struct DraftCfg {
    pub(crate) taps: Vec<u32>,
    layers: u32,
    heads: u32,
    inter: u32,
    markov_rank: u32,
    confidence: bool,
    eps: f32,
    block: u32,
    mask_token: u32,
    softcap: f32,
}

impl DraftCfg {
    /// The DSpark config, refused by name wherever it differs from what the context tail and the
    /// target layers it copies its KV writers from assume.
    pub(crate) fn load(c: &Cfg) -> Self {
        let dir = draft_dir();
        let v: Value = serde_json::from_slice(
            &std::fs::read(dir.join("config.json"))
                .unwrap_or_else(|e| refuse(format!("{}/config.json: {e}", dir.display()))),
        )
        .unwrap_or_else(|e| refuse(format!("config.json: {e}")));
        let u = |k: &str| -> u32 {
            v[k].as_u64()
                .unwrap_or_else(|| refuse(format!("draft config: {k} missing"))) as u32
        };
        let want = |ok: bool, what: &str| {
            if !ok {
                refuse(format!("draft config: {what}"))
            }
        };
        want(
            v["model_type"] == "gemma4_text" && v["architectures"][0] == "Gemma4DSparkModel",
            "not a Gemma4DSparkModel",
        );
        want(u("hidden_size") == c.hidden, "hidden_size != target hidden");
        want(u("vocab_size") == c.vocab, "vocab_size != target vocab");
        want(
            u("num_target_layers") == c.layers,
            "num_target_layers != target layers",
        );
        want(
            v["attention_k_eq_v"] == true,
            "attention_k_eq_v is required (V is the raw k projection)",
        );
        want(
            u("num_global_key_value_heads") == c.kvh_full,
            "num_global_key_value_heads != target full kv heads",
        );
        want(
            u("global_head_dim") == c.hd_full,
            "global_head_dim != target full head_dim",
        );
        want(
            u("hidden_size_per_layer_input") == 0,
            "per-layer inputs are unsupported",
        );
        want(
            v["markov_head_type"] == "vanilla",
            "markov_head_type is not vanilla",
        );
        let rp = &v["rope_parameters"]["full_attention"];
        want(
            rp["rope_theta"].as_f64() == Some(c.theta_full)
                && rp["partial_rotary_factor"].as_f64() == Some(c.rope_frac_full)
                && rp["rope_type"] == "proportional"
                && rp["factor"].as_f64().unwrap_or(1.0) == 1.0,
            "full-attention rope differs from the target's",
        );
        let layers = u("num_hidden_layers");
        let types = v["layer_types"].as_array().cloned().unwrap_or_default();
        want(
            layers > 0
                && types.len() == layers as usize
                && types.iter().all(|t| t == "full_attention"),
            "every draft layer must be full_attention",
        );
        let taps: Vec<u32> = v["target_layer_ids"]
            .as_array()
            .unwrap_or_else(|| refuse("draft config: target_layer_ids missing".into()))
            .iter()
            .map(|x| {
                x.as_u64()
                    .unwrap_or_else(|| refuse(format!("draft config: tap {x}")))
                    as u32
            })
            .collect();
        want(
            !taps.is_empty()
                && taps.windows(2).all(|w| w[0] < w[1])
                && taps.iter().all(|&l| l < c.layers),
            "target_layer_ids must ascend below the target layer count",
        );
        let block = u("block_size");
        want(
            block >= 1 && block <= kmax(),
            "block_size must be in 1..=PLOW_SPEC_KMAX",
        );
        want(
            u("mask_token_id") < c.vocab,
            "mask_token_id outside the vocab",
        );
        DraftCfg {
            taps,
            layers,
            heads: u("num_attention_heads"),
            inter: u("intermediate_size"),
            markov_rank: u("markov_rank"),
            confidence: v["enable_confidence_head"] == true,
            eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            block,
            mask_token: u("mask_token_id"),
            softcap: v["final_logit_softcapping"]
                .as_f64()
                .unwrap_or_else(|| refuse("draft config: final_logit_softcapping missing".into()))
                as f32,
        }
    }

    /// `(name, shape)` of every tensor the re-keyed shard must hold.
    fn expected(&self, c: &Cfg) -> Vec<(String, Vec<u64>)> {
        let (h, i, v, hd) = (
            c.hidden as u64,
            self.inter as u64,
            c.vocab as u64,
            c.hd_full as u64,
        );
        let (qd, kd, r) = (
            self.heads as u64 * hd,
            c.kvh_full as u64 * hd,
            self.markov_rank as u64,
        );
        let mut e: Vec<(String, Vec<u64>)> = vec![
            ("dspark.embed_tokens.weight".into(), vec![v, h]),
            ("dspark.hidden_norm.weight".into(), vec![h]),
            ("dspark.lm_head.weight".into(), vec![v, h]),
            ("dspark.markov_head.markov_w1.weight".into(), vec![v, r]),
            ("dspark.markov_head.markov_w2.weight".into(), vec![v, r]),
            ("dspark.norm.weight".into(), vec![h]),
        ];
        if self.confidence {
            e.push(("dspark.confidence_head.proj.bias".into(), vec![1]));
            e.push(("dspark.confidence_head.proj.weight".into(), vec![1, h + r]));
        }
        for j in 0..self.taps.len() {
            e.push((format!("dspark.fc.{j}.weight"), vec![h, h]));
        }
        for l in 0..self.layers {
            let p = format!("dspark.layers.{l}.");
            for (s, shape) in [
                ("input_layernorm.weight", vec![h]),
                ("layer_scalar", vec![1]),
                ("mlp.down_proj.weight", vec![h, i]),
                ("mlp.gate_proj.weight", vec![i, h]),
                ("mlp.up_proj.weight", vec![i, h]),
                ("post_attention_layernorm.weight", vec![h]),
                ("post_feedforward_layernorm.weight", vec![h]),
                ("pre_feedforward_layernorm.weight", vec![h]),
                ("self_attn.k_norm.weight", vec![hd]),
                ("self_attn.k_proj.weight", vec![kd, h]),
                ("self_attn.o_proj.weight", vec![h, qd]),
                ("self_attn.q_norm.weight", vec![hd]),
                ("self_attn.q_proj.weight", vec![qd, h]),
            ] {
                e.push((format!("{p}{s}"), shape));
            }
        }
        e
    }
}

/// One projection weight: the bf16 tensor on a bf16 target, the e4m3 twin and its per-row scale
/// on an FP8 one (the tail takes the target's projection route).
#[derive(Clone, Copy)]
pub(crate) struct Proj {
    pub(crate) w: u32,
    pub(crate) w8: u32,
    pub(crate) scale: u32,
    pub(crate) n: u32,
}

pub(crate) struct Tensors {
    pub(crate) cfg: DraftCfg,
    /// Per drafter layer: (k, v, k_scale, v_scale) cache handles (scales NONE under bf16 KV).
    pub(crate) kv: Vec<[u32; 4]>,
    /// Per tap: the copied post-layer residual `[rows][hidden]`.
    pub(crate) tap: Vec<u32>,
    pub(crate) fc: Vec<Proj>,
    pub(crate) k_proj: Vec<Proj>,
    pub(crate) k_norm: Vec<u32>,
    pub(crate) hidden_norm: u32,
    /// Tail scratch, per tap: activation quant (w8a8) and projection, so every projection of a
    /// phase runs back to back (one library-routed run per phase on the cuBLASLt rungs); then the
    /// sum, its norm and per-layer k.
    pub(crate) xq: Vec<u32>,
    pub(crate) xs: Vec<u32>,
    pub(crate) y: Vec<u32>,
    pub(crate) s: u32,
    pub(crate) xn: u32,
    pub(crate) k: Vec<u32>,
    fp8: bool,
    w8a8: bool,
    fp8_kv: bool,
    /// The draft program's tensors (FP8 target with span verify only).
    draft: Option<Draft>,
}

impl Tensors {
    /// Index of target layer `l` among the taps.
    pub(crate) fn tap_of(&self, l: usize) -> Option<usize> {
        self.cfg.taps.iter().position(|&t| t as usize == l)
    }

    pub(crate) fn eps(&self) -> f32 {
        self.cfg.eps
    }

    /// The packet KV layers the tail writes, after the target's own.
    pub(crate) fn kv_layers(&self, c: &Cfg) -> Vec<u32> {
        (c.layers..c.layers + self.cfg.layers).collect()
    }
}

/// Declare the drafter's KV layers (the bytes of target full layer `full_src`'s caches), the
/// context weights and the tail scratch for `rows` rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn declare(
    b: &mut Builder,
    c: &Cfg,
    rows: u32,
    fp8: bool,
    w8a8: bool,
    fp8_kv: bool,
    kv_src: [u32; 4],
    span_verify: bool,
) -> Tensors {
    if c.arch != Arch::Gemma4 || c.tp != 1 || c.moe || c.ple != 0 || c.kv_shared != 0 {
        refuse(
            "DSpark context needs a dense Gemma-4 target at TP 1 without PLE or KV sharing".into(),
        );
    }
    if !c.k_eq_v || !c.has_v_norm || c.attn_scale != 1.0 {
        refuse("DSpark context copies a k_eq_v, v-normed full-layer KV writer".into());
    }
    let e = crate::emit_config::active();
    if e.affine_q4 || e.mxfp4 {
        refuse("DSpark context takes bf16 or FP8 projections only".into());
    }
    if !e.decode_ladder_on() {
        refuse("DSpark context writers address KV by pos[]: set PLOW_DECODE_BATCH_LADDER".into());
    }
    let cfg = DraftCfg::load(c);
    let (r, h) = (rows as u64, c.hidden as u64);
    let kv = (0..cfg.layers)
        .map(|i| {
            let l = c.layers + i;
            let name = |s: &str| format!("kv.{l}.{s}");
            let decl = |b: &mut Builder, s: &str, src: u32| {
                if src == packet::dev::TENSOR_NONE {
                    packet::dev::TENSOR_NONE
                } else {
                    let bytes = b.tensor_bytes(src);
                    b.tensor(&name(s), bytes)
                }
            };
            [
                decl(b, "k", kv_src[0]),
                decl(b, "v", kv_src[1]),
                decl(b, "k_scale", kv_src[2]),
                decl(b, "v_scale", kv_src[3]),
            ]
        })
        .collect();
    let proj = |b: &mut Builder, name: &str, n: u32, k: u32| -> Proj {
        let (n64, k64) = (n as u64, k as u64);
        if fp8 {
            Proj {
                w: packet::dev::TENSOR_NONE,
                w8: b.tensor(&format!("fp8/{name}"), n64 * k64),
                scale: b.tensor(&format!("fp8/{name}_scale"), n64 * F32),
                n,
            }
        } else {
            Proj {
                w: b.tensor(name, n64 * k64 * BF16),
                w8: packet::dev::TENSOR_NONE,
                scale: packet::dev::TENSOR_NONE,
                n,
            }
        }
    };
    let a = |b: &mut Builder, s: &str, sz: u64| b.tensor(&format!("act.dspark.{s}"), sz);
    let kd = c.kvh_full * c.hd_full;
    let mut t = Tensors {
        kv,
        tap: (0..cfg.taps.len())
            .map(|j| a(b, &format!("tap{j}"), r * h * BF16))
            .collect(),
        fc: (0..cfg.taps.len())
            .map(|j| proj(b, &format!("dspark.fc.{j}.weight"), c.hidden, c.hidden))
            .collect(),
        k_proj: (0..cfg.layers)
            .map(|i| {
                proj(
                    b,
                    &format!("dspark.layers.{i}.self_attn.k_proj.weight"),
                    kd,
                    c.hidden,
                )
            })
            .collect(),
        k_norm: (0..cfg.layers)
            .map(|i| {
                b.tensor(
                    &format!("dspark.layers.{i}.self_attn.k_norm.weight"),
                    c.hd_full as u64 * BF16,
                )
            })
            .collect(),
        hidden_norm: b.tensor("dspark.hidden_norm.weight", h * BF16),
        xq: (0..cfg.taps.len())
            .map(|j| {
                if w8a8 {
                    a(b, &format!("xq{j}"), r * h)
                } else {
                    packet::dev::TENSOR_NONE
                }
            })
            .collect(),
        xs: (0..cfg.taps.len())
            .map(|j| {
                if w8a8 {
                    a(b, &format!("xs{j}"), r * F32)
                } else {
                    packet::dev::TENSOR_NONE
                }
            })
            .collect(),
        y: (0..cfg.taps.len())
            .map(|j| a(b, &format!("y{j}"), r * h * BF16))
            .collect(),
        s: a(b, "s", r * h * BF16),
        xn: a(b, "xn", r * h * BF16),
        k: (0..cfg.layers)
            .map(|i| a(b, &format!("k{i}"), r * kd as u64 * BF16))
            .collect(),
        cfg,
        fp8,
        w8a8,
        fp8_kv,
        draft: None,
    };
    if fp8 && span_verify {
        t.draft = Some(declare_draft(b, c, &t.cfg));
    }
    t
}

fn dims(entry: &Value) -> Vec<u64> {
    entry["shape"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default()
}

/// Check the shards against the drafter config, describe the context in `spec.context.json` and,
/// with the draft tensors declared, emit the draft programs (`spec.json`).
pub(crate) fn sections(
    dir: &Path,
    c: &Cfg,
    n: &crate::Tn,
    t: &Tensors,
    model: &packet::devbuild::Model,
    rung_progs: &[(u32, usize)],
) -> Vec<SectionData> {
    let shard = dir.join(SHARD);
    let (header, meta) = safetensors_header(&shard);
    // The fc column blocks are per tap: the shard must come from this very config.
    let config_sha256 = sha256_file(&draft_dir().join("config.json"));
    if meta["config_sha256"].as_str() != Some(config_sha256.as_str())
        || meta["fc_split"].as_str() != Some(t.cfg.taps.len().to_string().as_str())
    {
        refuse(format!(
            "{SHARD} was not re-keyed from {}/config.json (config_sha256 / fc_split)",
            draft_dir().display()
        ));
    }
    let expected = t.cfg.expected(c);
    for (name, shape) in &expected {
        let got = header
            .get(name)
            .unwrap_or_else(|| refuse(format!("{SHARD} lacks {name}")));
        if &dims(got) != shape || got["dtype"] != "BF16" {
            refuse(format!(
                "{SHARD} {name}: {:?} {} (expected {shape:?} BF16)",
                dims(got),
                got["dtype"]
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
    let shard_sha256 = sha256_file(&shard);
    if let Some(pin) = &crate::emit_config::active().spec_shard_sha256 {
        if *pin != shard_sha256 {
            refuse(format!(
                "{SHARD} sha256 {shard_sha256}, PLOW_SPEC_SHARD_SHA256 pins {pin}"
            ));
        }
    }
    let mut twins_sha256 = String::new();
    if t.fp8 {
        let (fp8, meta) = safetensors_header(&dir.join(FP8_SHARD));
        twins_sha256 = sha256_file(&dir.join(FP8_SHARD));
        if meta["source_shard_sha256"].as_str() != Some(shard_sha256.as_str()) {
            refuse(format!("{FP8_SHARD} was not made from this {SHARD}"));
        }
        let twins = (0..t.cfg.taps.len())
            .map(|j| format!("dspark.fc.{j}.weight"))
            .chain((0..t.cfg.layers).map(|i| format!("dspark.layers.{i}.self_attn.k_proj.weight")));
        for name in twins {
            let (_, shape) = expected
                .iter()
                .find(|(n, _)| *n == name)
                .expect("expected twin");
            for (suffix, dtype, want) in [
                ("", "F8_E4M3", shape.clone()),
                ("_scale", "F32", vec![shape[0]]),
            ] {
                let key = format!("fp8/{name}{suffix}");
                let got = fp8
                    .get(&key)
                    .unwrap_or_else(|| refuse(format!("{FP8_SHARD} lacks {key}")));
                if dims(got) != want || got["dtype"] != dtype {
                    refuse(format!(
                        "{FP8_SHARD} {key}: {:?} {} (expected {want:?} {dtype})",
                        dims(got),
                        got["dtype"]
                    ));
                }
            }
        }
    }
    let mut m = spec_context::Manifest {
        version: spec_context::VERSION,
        family: FAMILY.into(),
        taps: t.cfg.taps.clone(),
        kv_layers: t.kv_layers(c),
        kv_heads: c.kvh_full,
        head_dim: c.hd_full,
        kv_elem: if t.fp8_kv { "e4m3" } else { "bf16" }.into(),
        projection: match (t.fp8, t.w8a8) {
            (false, _) => "bf16",
            (true, false) => "w8a16",
            (true, true) => "w8a8",
        }
        .into(),
        twins_sha256,
        shard: SHARD.into(),
        shard_sha256,
        layout_sha256: String::new(),
    };
    m.layout_sha256 = m.layout_digest();
    m.validate(c.layers).unwrap_or_else(|e| refuse(e));
    eprintln!(
        "  spec context: {FAMILY} taps {:?} -> kv layers {:?} ({})",
        m.taps, m.kv_layers, m.kv_elem
    );
    let mut out = vec![SectionData {
        kind: SECT_METADATA,
        name: spec_context::SECTION.into(),
        data: serde_json::to_vec(&m).expect("spec.context.json"),
    }];
    if let Some(d) = &t.draft {
        let full = (0..c.layers as usize)
            .rev()
            .find(|&l| c.is_full[l] && !c.kv_is_shared(l))
            .expect("full layer");
        out.extend(draft_sections(
            dir,
            c,
            t,
            d,
            n.kc[full],
            model,
            rung_progs,
            &m.shard_sha256,
        ));
    }
    out
}

/// GEMV row ceiling of the decode interpreter's FP8 arms (`gemv_fp8_wgmma_supported`): a draft
/// program's block rows `size * rows` must fit.
const GEMV_ROWS: u32 = 64;
const AMAX_BLOCKS: u32 = 64;
const I32: u64 = 4;

/// One drafter layer of the draft program: FP8 twins `(e4m3, per-row scale)` for every GEMV.
struct DraftLayer {
    in_ln: u32,
    q: (u32, u32),
    q_norm: u32,
    o: (u32, u32),
    post_attn: u32,
    pre_ffn: u32,
    gate: (u32, u32),
    up: (u32, u32),
    down: (u32, u32),
    post_ffn: u32,
}

/// The draft program's tensors: inputs per block row (`[size * rows]`) and per request
/// (`tokens`, `[rows]`), activations sized for the widest program.
pub(crate) struct Draft {
    rows: u32,
    size: u32,
    mask: u32,
    tokens: Vec<u32>,
    ids: u32,
    pos: u32,
    kv_len: u32,
    slot: u32,
    store: u32,
    /// Per block position `i`: the block rows `r * size + i` (`[rows]` u32, packet constants).
    pick: Vec<u32>,
    x: u32,
    xn: u32,
    q: u32,
    qn: u32,
    k: u32,
    at: u32,
    og: u32,
    hn: u32,
    fu: u32,
    dg: u32,
    logits: u32,
    lg: u32,
    e: u32,
    bias: u32,
    sum: u32,
    amax: u32,
    emb: u32,
    norm: u32,
    head: (u32, u32),
    w1: u32,
    w2: u32,
    layers: Vec<DraftLayer>,
}

fn declare_draft(b: &mut Builder, c: &Cfg, cfg: &DraftCfg) -> Draft {
    let size = cfg.block;
    let rows = crate::mtp::rows_list()
        .into_iter()
        .filter(|&r| r * size <= GEMV_ROWS)
        .max()
        .unwrap_or_else(|| refuse(format!("no draft row count fits {GEMV_ROWS} block rows")));
    let (h, v, hd, r) = (
        c.hidden as u64,
        c.vocab as u64,
        c.hd_full as u64,
        cfg.markov_rank as u64,
    );
    let br = (rows * size) as u64;
    let qd = cfg.heads as u64 * hd;
    let inter = cfg.inter as u64;
    let w = |b: &mut Builder, name: &str, n: u64| b.tensor(name, n * BF16);
    let w8 = |b: &mut Builder, name: &str, n: u64, k: u64| {
        (
            b.tensor(&format!("fp8/{name}"), n * k),
            b.tensor(&format!("fp8/{name}_scale"), n * F32),
        )
    };
    let a = |b: &mut Builder, s: &str, sz: u64| b.tensor(&format!("act.dspark.d.{s}"), sz);
    let inp = |b: &mut Builder, s: &str, n: u64| b.tensor(&format!("in.spec.{s}"), n * I32);
    let layers = (0..cfg.layers)
        .map(|l| {
            let p = format!("dspark.layers.{l}.");
            DraftLayer {
                in_ln: w(b, &format!("{p}input_layernorm.weight"), h),
                q: w8(b, &format!("{p}self_attn.q_proj.weight"), qd, h),
                q_norm: w(b, &format!("{p}self_attn.q_norm.weight"), hd),
                o: w8(b, &format!("{p}self_attn.o_proj.weight"), h, qd),
                post_attn: w(b, &format!("{p}post_attention_layernorm.weight"), h),
                pre_ffn: w(b, &format!("{p}pre_feedforward_layernorm.weight"), h),
                gate: w8(b, &format!("{p}mlp.gate_proj.weight"), inter, h),
                up: w8(b, &format!("{p}mlp.up_proj.weight"), inter, h),
                down: w8(b, &format!("{p}mlp.down_proj.weight"), h, inter),
                post_ffn: w(b, &format!("{p}post_feedforward_layernorm.weight"), h),
            }
        })
        .collect();
    let pick = (0..size)
        .map(|i| {
            let init: Vec<u8> = (0..rows)
                .flat_map(|r| (r * size + i).to_le_bytes())
                .collect();
            b.tensor_init(&format!("in.dspark.pick.{i}"), init)
        })
        .collect();
    Draft {
        rows,
        size,
        mask: cfg.mask_token,
        tokens: (0..=kmax().min(size))
            .map(|j| inp(b, &format!("tokens.{j}"), rows as u64))
            .collect(),
        ids: inp(b, "block.ids", br),
        pos: inp(b, "block.pos", br),
        kv_len: inp(b, "block.kv_len", br),
        slot: inp(b, "block.slot", br),
        store: inp(b, "block.store", br),
        pick,
        x: a(b, "x", br * h * BF16),
        xn: a(b, "xn", br * h * BF16),
        q: a(b, "q", br * qd * BF16),
        qn: a(b, "qn", br * qd * BF16),
        k: a(b, "k", br * hd * BF16),
        at: a(b, "at", br * qd * BF16),
        og: a(b, "og", br * h * BF16),
        hn: a(b, "hn", br * h * BF16),
        fu: a(b, "fu", br * inter * BF16),
        dg: a(b, "dg", br * h * BF16),
        logits: a(b, "logits", br * v * BF16),
        lg: a(b, "lg", rows as u64 * v * BF16),
        e: a(b, "e", rows as u64 * r * BF16),
        bias: a(b, "bias", rows as u64 * v * BF16),
        sum: a(b, "sum", rows as u64 * v * BF16),
        amax: a(b, "amax", rows as u64 * AMAX_BLOCKS as u64 * 8),
        emb: w(b, "dspark.embed_tokens.weight", v * h),
        norm: w(b, "dspark.norm.weight", h),
        head: w8(b, "dspark.lm_head.weight", v, h),
        w1: w(b, "dspark.markov_head.markov_w1.weight", v * r),
        w2: w(b, "dspark.markov_head.markov_w2.weight", v * r),
        layers,
    }
}

/// Each drafter layer's `layer_scalar` from the re-keyed shard the packet binds (a bf16 scalar
/// the residual tail folds in at emit time).
fn layer_scalars(shard: &Path, layers: u32) -> Vec<f32> {
    use std::io::{Read, Seek, SeekFrom};
    let (header, _) = safetensors_header(shard);
    let mut f =
        std::fs::File::open(shard).unwrap_or_else(|e| refuse(format!("{}: {e}", shard.display())));
    let mut len = [0u8; 8];
    f.read_exact(&mut len)
        .unwrap_or_else(|e| refuse(format!("{SHARD}: {e}")));
    let base = 8 + u64::from_le_bytes(len);
    (0..layers)
        .map(|l| {
            let name = format!("dspark.layers.{l}.layer_scalar");
            let ofs = header[&name]["data_offsets"][0]
                .as_u64()
                .unwrap_or_else(|| refuse(format!("{SHARD} {name}: no offsets")));
            let mut v = [0u8; 2];
            f.seek(SeekFrom::Start(base + ofs))
                .and_then(|_| f.read_exact(&mut v))
                .unwrap_or_else(|e| refuse(format!("{SHARD} {name}: {e}")));
            f32::from_bits(u32::from(u16::from_le_bytes(v)) << 16)
        })
        .collect()
}

/// The target rung instructions the draft copies its geometry from: the embedding (scale), the
/// last full layer's q HeadNormRope, FlashDecodeFp8 and FlashMerge, and per drafter layer the
/// context tail's K and V writers (`kv.{L+l}`).
struct Sites {
    embed: DevInst,
    q_rope: DevInst,
    flash: DevInst,
    merge: DevInst,
    writers: Vec<(DevInst, DevInst)>,
}

fn sites(t: &Tensors, full_kc: u32, target: &Program) -> Sites {
    let insts = &target.insts;
    let find = |what: &str, f: &dyn Fn(&DevInst) -> bool| -> DevInst {
        *insts
            .iter()
            .find(|i| f(i))
            .unwrap_or_else(|| refuse(format!("draft geometry: target rung has no {what}")))
    };
    let embed = find("Embed", &|i| i.op == DevOp::Embed as u16);
    let fa = insts
        .iter()
        .position(|i| i.op == DevOp::FlashDecodeFp8 as u16 && i.t[3] == full_kc)
        .unwrap_or_else(|| refuse("draft geometry: no full-layer FP8 flash decode".into()));
    let flash = insts[fa];
    let q_rope = *insts[..fa]
        .iter()
        .rev()
        .find(|i| i.op == DevOp::HeadNormRope as u16 && i.t[0] == flash.t[2])
        .unwrap_or_else(|| refuse("draft geometry: no full-layer q HeadNormRope".into()));
    let merge = *insts[fa..]
        .iter()
        .find(|i| i.op == DevOp::FlashMerge as u16 && i.t[1] == flash.t[0])
        .unwrap_or_else(|| refuse("draft geometry: no full-layer FlashMerge".into()));
    let writers =
        t.kv.iter()
            .map(|kv| {
                let w = |cache: u32| {
                    find("drafter KV writer", &|i| {
                        i.op == DevOp::HeadNormRopeFp8 as u16 && i.t[0] == cache
                    })
                };
                (w(kv[0]), w(kv[1]))
            })
            .collect();
    Sites {
        embed,
        q_rope,
        flash,
        merge,
        writers,
    }
}

/// One `(rows, k)` draft program: the non-causal block forward over the drafter layers (block
/// K/V stored at `c..c+size` through the context tail's writers, every block row attending
/// `c + size` rows of its sequence), the drafter head over every block row, then `k` sequential
/// Markov stages: `d_{i+1} = argmax(cap(L_i) + cap(w2 . w1[d_i]))`, `d_0` the pending token.
fn program(
    c: &Cfg,
    t: &Tensors,
    d: &Draft,
    tensors: &[packet::devbuild::TensorDecl],
    n_cu: u32,
    s: &Sites,
    scalars: &[f32],
    rows: u32,
    k: u32,
) -> Program {
    let mut b = Builder::new(n_cu);
    b.adopt_tensors(tensors.to_vec());
    b.force_uniseg();
    let all: Vec<u32> = (0..n_cu).collect();
    let br = rows * d.size;
    let bset: Vec<u32> = (0..br).collect();
    let rset: Vec<u32> = (0..rows).collect();
    let span = |n: u32| -> Vec<u32> { (0..n.max(1).min(n_cu)).collect() };
    let elem = |n: u32| span(n.div_ceil(512 * 8));
    let (h, hd, v) = (c.hidden, c.hd_full, c.vocab);
    let qd = t.cfg.heads * hd;
    let gemv8 = |b: &mut Builder,
                 dep: &[u32],
                 out: u32,
                 x: u32,
                 w: (u32, u32),
                 m: u32,
                 n: u32,
                 kin: u32| {
        b.emit(DevOp::GemvFp8, all.clone(), dep, |i| {
            i.t[0] = out;
            i.t[1] = x;
            i.t[2] = w.0;
            i.t[5] = w.1;
            i.i[0] = m;
            i.i[1] = n;
            i.i[2] = kin;
        })
    };
    let c_e = b.emit(DevOp::Embed, bset.clone(), &[], |i| {
        i.t[0] = d.x;
        i.t[1] = d.emb;
        i.t[2] = d.ids;
        i.i[0] = br;
        i.i[1] = h;
        i.i[2] = s.embed.i[2];
        i.f[0] = s.embed.f[0];
    });
    let mut dep = b.emit(DevOp::RmsNorm, bset.clone(), &[c_e], |i| {
        i.t[0] = d.xn;
        i.t[1] = d.x;
        i.t[2] = d.layers[0].in_ln;
        i.i[0] = br;
        i.i[1] = h;
        i.f[0] = t.cfg.eps;
    });
    for (l, w) in d.layers.iter().enumerate() {
        let c_q = gemv8(&mut b, &[dep], d.q, d.xn, w.q, br, qd, h);
        let c_qn = b.emit(
            DevOp::HeadNormRope,
            span(s.q_rope.blocks as u32),
            &[c_q],
            |i| {
                crate::mtp::args(i, &s.q_rope);
                i.t[0] = d.qn;
                i.t[1] = d.q;
                i.t[2] = w.q_norm;
                i.t[5] = d.pos;
                i.i[0] = br;
                i.f[0] = t.cfg.eps;
            },
        );
        let kp = t.k_proj[l];
        let c_k = gemv8(&mut b, &[dep], d.k, d.xn, (kp.w8, kp.scale), br, kp.n, h);
        let (kw, vw) = &s.writers[l];
        let store = |b: &mut Builder, src: &DevInst| {
            b.emit(
                DevOp::HeadNormRopeFp8,
                span(src.blocks as u32),
                &[c_k],
                |i| {
                    crate::mtp::args(i, src);
                    i.t[1] = d.k;
                    i.t[5] = d.pos;
                    i.t[7] = d.store;
                    i.i[0] = br;
                },
            )
        };
        let c_kw = store(&mut b, kw);
        let c_vw = store(&mut b, vw);
        let [kc, vc, ks, vs] = t.kv[l];
        let c_fa = b.emit(
            DevOp::FlashDecodeFp8,
            span(s.flash.blocks as u32),
            &[c_qn, c_kw, c_vw],
            |i| {
                crate::mtp::args(i, &s.flash);
                i.t[2] = d.qn;
                i.t[3] = kc;
                i.t[4] = vc;
                i.t[5] = d.kv_len;
                i.t[6] = ks;
                i.t[7] = vs;
                i.i[0] = br;
                i.j[1] = d.slot;
            },
        );
        let c_mg = b.emit(
            DevOp::FlashMerge,
            span(s.merge.blocks as u32),
            &[c_fa],
            |i| {
                crate::mtp::args(i, &s.merge);
                i.t[0] = d.at;
                i.i[0] = br;
            },
        );
        let c_o = gemv8(&mut b, &[c_mg], d.og, d.at, w.o, br, h, qd);
        let c_r1 = b.emit(DevOp::NormResidualNorm, bset.clone(), &[c_o], |i| {
            i.t[0] = d.hn;
            i.t[1] = d.x;
            i.t[2] = d.x;
            i.t[3] = d.og;
            i.t[4] = w.post_attn;
            i.t[5] = w.pre_ffn;
            i.i[0] = br;
            i.i[1] = h;
            i.f[0] = t.cfg.eps;
            i.f[1] = 1.0;
        });
        let c_gu = b.emit(DevOp::GemvGluFp8, all.clone(), &[c_r1], |i| {
            i.t[0] = d.fu;
            i.t[1] = d.hn;
            i.t[2] = w.gate.0;
            i.t[5] = w.up.0;
            i.t[3] = w.gate.1;
            i.t[4] = w.up.1;
            i.i[0] = br;
            i.i[1] = t.cfg.inter;
            i.i[2] = h;
            i.i[5] = c.mlp_act;
        });
        let c_d = gemv8(&mut b, &[c_gu], d.dg, d.fu, w.down, br, h, t.cfg.inter);
        let next = d.layers.get(l + 1).map_or(d.norm, |n| n.in_ln);
        dep = b.emit(DevOp::NormResidualNorm, bset.clone(), &[c_d], |i| {
            i.t[0] = d.xn;
            i.t[1] = d.x;
            i.t[2] = d.x;
            i.t[3] = d.dg;
            i.t[4] = w.post_ffn;
            i.t[5] = next;
            i.i[0] = br;
            i.i[1] = h;
            i.f[0] = t.cfg.eps;
            i.f[1] = scalars[l];
        });
    }
    // Raw logits of every block row: each term is capped once, in the Markov stages.
    let c_lm = gemv8(&mut b, &[dep], d.logits, d.xn, d.head, br, v, h);
    let nb = if rows > 1 { rows } else { 0 };
    let mut prev = vec![c_lm];
    for i in 0..k as usize {
        let c_g = b.emit(DevOp::RowGather, rset.clone(), &prev, |x| {
            x.t[0] = d.lg;
            x.t[1] = d.logits;
            x.t[2] = d.pick[i];
            x.i[0] = rows;
            x.i[1] = v;
            x.i[2] = br;
        });
        let c_cl = b.emit(DevOp::SoftCap, elem(rows * v), &[c_g], |x| {
            x.t[0] = d.lg;
            x.t[1] = d.lg;
            x.i[0] = rows * v;
            x.f[0] = t.cfg.softcap;
        });
        let c_em = b.emit(DevOp::Embed, rset.clone(), &prev, |x| {
            x.t[0] = d.e;
            x.t[1] = d.w1;
            x.t[2] = d.tokens[i];
            x.i[0] = rows;
            x.i[1] = t.cfg.markov_rank;
            x.f[0] = 1.0;
        });
        let c_b = b.emit(DevOp::Gemv, all.clone(), &[c_em], |x| {
            x.t[0] = d.bias;
            x.t[1] = d.e;
            x.t[2] = d.w2;
            x.i[0] = rows;
            x.i[1] = v;
            x.i[2] = t.cfg.markov_rank;
        });
        let c_cb = b.emit(DevOp::SoftCap, elem(rows * v), &[c_b], |x| {
            x.t[0] = d.bias;
            x.t[1] = d.bias;
            x.i[0] = rows * v;
            x.f[0] = t.cfg.softcap;
        });
        let c_s = b.emit(DevOp::Residual, elem(rows * v), &[c_cl, c_cb], |x| {
            x.t[0] = d.sum;
            x.t[1] = d.lg;
            x.t[2] = d.bias;
            x.i[0] = rows * v;
            x.f[0] = 1.0;
        });
        let c_am = b.emit(DevOp::Argmax, (0..AMAX_BLOCKS).collect(), &[c_s], |x| {
            x.t[0] = d.amax;
            x.t[1] = d.sum;
            x.i[0] = v;
            x.i[1] = nb;
        });
        prev = vec![b.emit(DevOp::ArgmaxFin, vec![0], &[c_am], |x| {
            x.t[0] = d.tokens[i + 1];
            x.t[1] = d.amax;
            x.i[0] = AMAX_BLOCKS;
            x.i[1] = nb;
        })];
    }
    b.finish()
}

/// The draft programs for every `(rows, k)` whose block fits a target decode rung, as
/// `spec.json` + the programs section.
fn draft_sections(
    dir: &Path,
    c: &Cfg,
    t: &Tensors,
    d: &Draft,
    full_kc: u32,
    model: &packet::devbuild::Model,
    rung_progs: &[(u32, usize)],
    shard_sha256: &str,
) -> Vec<SectionData> {
    // The DSpark checkpoint ships no tokenizer (it drafts in the target's vocabulary, checked by
    // size in `DraftCfg::load`); one that does must agree with the target's.
    let draft_tok = draft_dir().join("tokenizer.json");
    if draft_tok.exists() {
        crate::mtp::check_tokenizer(&draft_tok, &dir.join("tokenizer.json"));
    }
    let kmax = kmax().min(d.size);
    let scalars = layer_scalars(&dir.join(SHARD), t.cfg.layers);
    let mut programs = Vec::new();
    let mut table = Vec::new();
    for rows in crate::mtp::rows_list().into_iter().filter(|&r| r <= d.rows) {
        // The flash/merge scratch of the narrowest rung holding the block rows.
        let Some(&(_, pi)) = rung_progs
            .iter()
            .filter(|&&(r, _)| r >= rows * d.size)
            .min_by_key(|&&(r, _)| r)
        else {
            continue;
        };
        let s = sites(t, full_kc, &model.progs[pi]);
        for k in 1..=kmax {
            table.push(spec_draft::DraftProgram {
                rows,
                k,
                index: programs.len() as u32,
            });
            programs.push(program(
                c,
                t,
                d,
                &model.tensors,
                model.n_cu,
                &s,
                &scalars,
                rows,
                k,
            ));
        }
    }
    if programs.is_empty() {
        refuse("no target decode rung holds a draft block".into());
    }
    let packed: Vec<Vec<_>> = programs
        .iter()
        .map(|p| p.insts.iter().map(|i| i.pack()).collect())
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
        family: FAMILY.into(),
        kmax,
        programs: table,
        tensors: spec_draft::DraftTensors {
            tokens: d.tokens.iter().map(|&h| h16(h)).collect(),
            hidden_row: None,
            position: h16(d.pos),
            kv_len: h16(d.kv_len),
            slot: h16(d.slot),
            hidden: None,
            hidden_rows: 0,
        },
        block: Some(spec_draft::BlockDraft {
            size: d.size,
            mask_token: d.mask,
            ids: h16(d.ids),
            store_slot: h16(d.store),
        }),
        shard: SHARD.into(),
        shard_sha256: shard_sha256.into(),
    };
    manifest
        .validate(programs.len(), model.tensors.len())
        .unwrap_or_else(|e| refuse(e));
    eprintln!(
        "  spec draft: {FAMILY} {} programs (rows {:?}, block {}, k 1..={kmax})",
        programs.len(),
        manifest
            .programs
            .iter()
            .map(|p| p.rows)
            .collect::<std::collections::BTreeSet<_>>(),
        d.size
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
