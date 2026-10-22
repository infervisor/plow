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

use packet::devbuild::{Builder, SectionData, SECT_METADATA};
use plow_asset::spec_context;
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
    Tensors {
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
    }
}

fn dims(entry: &Value) -> Vec<u64> {
    entry["shape"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default()
}

/// Check the shards against the drafter config and describe the context in `spec.context.json`.
pub(crate) fn sections(dir: &Path, c: &Cfg, t: &Tensors) -> Vec<SectionData> {
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
    vec![SectionData {
        kind: SECT_METADATA,
        name: spec_context::SECTION.into(),
        data: serde_json::to_vec(&m).expect("spec.context.json"),
    }]
}
